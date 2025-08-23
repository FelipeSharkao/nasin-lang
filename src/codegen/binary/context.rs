use std::borrow::Cow;
use std::collections::HashMap;

use cranelift_shim::{self as cl, Module};
use derive_new::new;
use itertools::repeat_n;

use super::func::FuncCodegen;
use super::types::ReturnPolicy;
use super::{types, FuncNS};
use crate::{bytecode as b, config, utils};

#[derive(Debug)]
pub struct FuncBinding {
    pub is_extrn: bool,
    pub is_virt: bool,
    pub symbol_name: String,
    pub func_id: Option<cl::FuncId>,
    pub proto: types::FuncPrototype,
}

#[derive(Debug, Clone)]
pub struct GlobalBinding<'a> {
    pub symbol_name: String,
    pub value: types::RuntimeValue,
    pub ty: Cow<'a, b::Type>,
    pub is_const: bool,
}

#[derive(Debug)]
pub struct FuncClosureBinding {
    pub symbol_name: String,
    pub func_id: cl::FuncId,
    pub func: cl::Function,
    pub ret_policy: ReturnPolicy,
}
impl Into<types::FuncPrototype> for &FuncClosureBinding {
    fn into(self) -> types::FuncPrototype {
        types::FuncPrototype::new(self.func.signature.clone(), self.ret_policy)
    }
}

#[derive(new)]
pub struct CodegenContext<'a> {
    pub modules: &'a [b::Module],
    pub cfg: &'a config::BuildConfig,
    pub cl_module: cl::ObjectModule,
    #[new(default)]
    pub funcs: HashMap<(usize, usize), FuncBinding>,
    #[new(default)]
    pub data: HashMap<cl::DataId, cl::DataDescription>,
    #[new(default)]
    pub globals: HashMap<(usize, usize), GlobalBinding<'a>>,
    #[new(default)]
    pub vtables_desc: HashMap<(usize, usize), types::VTableDesc>,
    #[new(default)]
    pub vtables_impl: HashMap<types::VTableRef, cl::DataId>,
    #[new(default)]
    pub funcs_closures: HashMap<(usize, usize), FuncClosureBinding>,
    #[new(default)]
    strings: HashMap<String, cl::DataId>,
    #[new(default)]
    tuples: HashMap<Vec<types::ValueSource>, cl::DataId>,
    #[new(default)]
    next_helper_id: u32,
}
impl<'a> CodegenContext<'a> {
    pub fn get_global(&self, mod_idx: usize, idx: usize) -> Option<&GlobalBinding> {
        self.globals.get(&(mod_idx, idx))
    }

    pub fn insert_global(&mut self, mod_idx: usize, idx: usize) {
        let module = &self.modules[mod_idx];
        let global = &module.globals[idx];
        let global_value = &module.values[global.value];

        // TODO: improve name mangling
        let symbol_name = global.name.clone();

        let (value, is_const) = utils::replace_with(self, |s| {
            let mut codegen = FuncCodegen::new(s, None);

            for instr in &global.body {
                if let b::InstrBody::Break(v) = &instr.body {
                    if let Some(v) = v {
                        codegen.values.insert(
                            (mod_idx, global.value),
                            codegen.values[&(mod_idx, *v)].clone(),
                        );
                    }
                    break;
                }

                if codegen.value_from_instr(instr, mod_idx).is_none() {
                    let data_id = codegen.ctx.create_writable_for_type(&global_value.ty);
                    let value =
                        types::RuntimeValue::new(data_id.into(), mod_idx, global.value);
                    return (codegen.ctx, (value, false));
                }
            }

            (
                codegen.ctx,
                (codegen.values[&(mod_idx, global.value)].clone(), true),
            )
        });

        self.globals.insert(
            (mod_idx, idx),
            GlobalBinding {
                symbol_name,
                value,
                ty: Cow::Borrowed(&global_value.ty),
                is_const,
            },
        );
    }

    pub fn insert_type(&mut self, mod_idx: usize, idx: usize) {
        let ty = &self.modules[mod_idx].typedefs[idx];
        match &ty.body {
            b::TypeDefBody::Record(rec) => {
                self.insert_record_type(mod_idx, idx, rec);
            }
            b::TypeDefBody::Interface(iface) => {
                self.insert_interface_type(mod_idx, idx, iface);
            }
        }
    }

    pub fn data_for_string(&mut self, value: &str) -> cl::DataId {
        if let Some(id) = self.strings.get(value) {
            return *id;
        }

        let data_id = self.cl_module.declare_anonymous_data(false, false).unwrap();
        let mut desc = cl::DataDescription::new();

        let mut bytes = value.as_bytes().to_vec();
        // Append a null terminator to avoid problems if used as a C string
        bytes.extend([0]);

        desc.define(bytes.into());
        self.cl_module.define_data(data_id, &desc).unwrap();

        self.data.insert(data_id, desc);
        self.strings.insert(value.to_string(), data_id);
        data_id
    }

    pub fn data_for_tuple(
        &mut self,
        values: Vec<types::ValueSource>,
    ) -> Option<cl::DataId> {
        if let Some(id) = self.tuples.get(&values) {
            return Some(*id);
        }

        let data_id = self.cl_module.declare_anonymous_data(false, false).unwrap();
        let mut desc = cl::DataDescription::new();

        let mut bytes = vec![];
        let mut included_datas = HashMap::new();
        let mut included_funcs = HashMap::new();

        for item in &values {
            match item {
                types::ValueSource::Data(field_data_id) => {
                    let offset = bytes.len();
                    bytes.extend(repeat_n(
                        0u8,
                        self.cl_module.isa().pointer_bytes() as usize,
                    ));

                    let field_gv =
                        included_datas.entry(field_data_id).or_insert_with(|| {
                            self.cl_module
                                .declare_data_in_data(*field_data_id, &mut desc)
                        });
                    desc.write_data_addr(offset as u32, field_gv.clone(), 0);
                }
                types::ValueSource::Func(func_id) => {
                    let offset = bytes.len();
                    bytes.extend(repeat_n(
                        0u8,
                        self.cl_module.isa().pointer_bytes() as usize,
                    ));

                    let func = included_funcs.entry(func_id).or_insert_with(|| {
                        self.cl_module.declare_func_in_data(*func_id, &mut desc)
                    });
                    desc.write_function_addr(offset as u32, func.clone());
                }
                _ => {
                    let res =
                        item.serialize(&mut bytes, self.cl_module.isa().endianness());
                    if res.is_err() {
                        return None;
                    }
                }
            }
        }

        desc.define(bytes.into());
        self.cl_module.define_data(data_id, &desc).unwrap();

        self.data.insert(data_id, desc);
        self.tuples.insert(values, data_id);
        Some(data_id)
    }

    pub fn create_writable_for_type(&mut self, ty: &b::Type) -> cl::DataId {
        let data_id = self.cl_module.declare_anonymous_data(false, false).unwrap();
        let mut desc = cl::DataDescription::new();
        desc.define_zeroinit(types::get_size(ty, self.modules, &self.cl_module) as usize);
        self.cl_module.define_data(data_id, &desc).unwrap();

        self.data.insert(data_id, desc);
        data_id
    }

    pub fn closure_for_func(
        &mut self,
        mod_idx: usize,
        func_idx: usize,
    ) -> (cl::FuncId, types::FuncPrototype) {
        let key = (mod_idx, func_idx);
        if let Some(binding) = self.funcs_closures.get(&key) {
            return (binding.func_id, binding.into());
        }

        let func_binding = &self.funcs[&key];
        let mut sig = func_binding.proto.signature.clone();
        sig.params.splice(
            0..0,
            [cl::AbiParam::new(self.cl_module.isa().pointer_type())],
        );

        let func = cl::Function::with_name_signature(
            cl::UserFuncName::user(FuncNS::Helper.into(), self.next_helper_id),
            sig,
        );
        self.next_helper_id += 1;

        // TODO: improve name mangling
        let symbol_name = format!("{}$$closure", &func_binding.symbol_name);

        let func_id = self
            .cl_module
            .declare_function(&symbol_name, cl::Linkage::Local, &func.signature)
            .unwrap();

        let binding = FuncClosureBinding {
            symbol_name,
            func_id,
            func,
            ret_policy: func_binding.proto.ret_policy,
        };
        let proto = (&binding).into();

        self.funcs_closures.insert(key, binding);
        (func_id, proto)
    }

    fn insert_record_type(
        &mut self,
        mod_idx: usize,
        idx: usize,
        rec: &b::RecordType,
    ) -> Vec<cl::DataId> {
        let key = (mod_idx, idx);

        rec.ifaces
            .iter()
            .map(|iface_key| {
                let key = types::VTableRef::new(*iface_key, key);
                if let Some(data_id) = self.vtables_impl.get(&key) {
                    return *data_id;
                }

                let b::TypeDefBody::Interface(iface) =
                    &self.modules[iface_key.0].typedefs[iface_key.1].body
                else {
                    panic!(
                        "type {}-{} should be an interface, I don't know what I got",
                        iface_key.0, iface_key.1
                    );
                };

                let vtable_desc =
                    self.insert_interface_type(iface_key.0, iface_key.1, iface);

                let tuple = vtable_desc
                    .methods
                    .clone()
                    .iter()
                    .map(|m| {
                        let func_ref = rec.methods[m].func_ref;
                        let func_id = self.funcs[&func_ref]
                            .func_id
                            .expect("Function should be defined");
                        types::ValueSource::Func(func_id)
                    })
                    .collect();

                let data_id = self
                    .data_for_tuple(tuple)
                    .expect("vtable should be serializable");

                self.vtables_impl.insert(key, data_id);
                data_id
            })
            .collect()
    }

    fn insert_interface_type(
        &mut self,
        mod_idx: usize,
        idx: usize,
        iface: &b::InterfaceType,
    ) -> &types::VTableDesc {
        self.vtables_desc.entry((mod_idx, idx)).or_insert_with(|| {
            types::VTableDesc::new(iface.methods.keys().cloned().collect())
        })
    }
}
