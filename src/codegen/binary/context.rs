use std::borrow::Cow;
use std::collections::HashMap;

use cranelift_shim::{self as cl, Module};
use derive_ctor::ctor;
use itertools::{Itertools, repeat_n};

use super::debug::DebugData;
use super::func::FuncCodegen;
use super::name_mangling::NameMangler;
use super::types::ReturnPolicy;
use super::{FuncNS, types};
use crate::{bytecode as b, config, utils};

#[derive(Debug)]
pub struct FuncBinding {
    pub is_extrn:    bool,
    pub is_virt:     bool,
    pub symbol_name: String,
    pub func_id:     Option<cl::FuncId>,
    pub proto:       types::FuncPrototype,
}

#[derive(Debug, Clone)]
pub struct GlobalBinding<'a> {
    #[allow(dead_code)]
    pub symbol_name: String,
    pub value:       types::RuntimeValue,
    #[allow(dead_code)]
    pub ty:          Cow<'a, b::Type>,
    pub is_const:    bool,
}

#[derive(Debug)]
pub struct FuncClosureBinding {
    pub symbol_name: String,
    pub func_id:     cl::FuncId,
    pub func:        cl::Function,
    pub ret_policy:  ReturnPolicy,
}
impl Into<types::FuncPrototype> for &FuncClosureBinding {
    fn into(self) -> types::FuncPrototype {
        types::FuncPrototype::new(self.func.signature.clone(), self.ret_policy)
    }
}

#[derive(ctor)]
pub struct CodegenContext<'a> {
    pub modules: &'a [b::Module],
    pub cfg: &'a config::BuildConfig,
    pub cl_module: cl::ObjectModule,
    pub debug: DebugData<'a>,
    #[ctor(default)]
    pub funcs: HashMap<(usize, usize), FuncBinding>,
    #[ctor(default)]
    pub data: HashMap<cl::DataId, cl::DataDescription>,
    #[ctor(default)]
    pub globals: HashMap<(usize, usize), GlobalBinding<'a>>,
    #[ctor(default)]
    pub vtables_desc: HashMap<(usize, usize), types::VTableDesc>,
    #[ctor(default)]
    pub vtables_impl: HashMap<types::VTableRef, cl::DataId>,
    #[ctor(default)]
    pub funcs_closures: HashMap<(usize, usize), FuncClosureBinding>,
    #[ctor(default)]
    strings: HashMap<String, cl::DataId>,
    #[ctor(default)]
    tuples: HashMap<Vec<types::ValueSource>, cl::DataId>,
    #[ctor(default)]
    next_helper_id: u32,
}
impl<'a> CodegenContext<'a> {
    pub fn get_global(&self, mod_idx: usize, idx: usize) -> Option<&GlobalBinding<'_>> {
        self.globals.get(&(mod_idx, idx))
    }

    pub fn insert_global(&mut self, mod_idx: usize, idx: usize) {
        let module = &self.modules[mod_idx];
        let global = &module.globals[idx];
        let global_value = &module.values[global.value];
        let body = module.blocks[global.body].body.to_vec();

        let symbol_name = NameMangler::new(self.modules).mangle(&global.name, []);

        let (value, is_const) = utils::replace_with(self, |s| {
            let mut codegen = FuncCodegen::new(s, None, ReturnPolicy::NoReturn);
            codegen.is_global = true;

            for instr in &body {
                if let b::InstrBody::Break(block_idx, Some(v)) = &instr.body
                    && *block_idx == global.body
                {
                    codegen.values.insert(
                        (mod_idx, global.value),
                        codegen.values[&(mod_idx, *v)].clone(),
                    );
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

    pub fn insert_type_meta(&mut self, mod_idx: usize, idx: b::TypeMetaIdx) {
        let type_meta = &self.modules[mod_idx].types_meta[idx];
        match &type_meta.body {
            b::TypeDefBody::Record(..) => {
                self.insert_record_type(mod_idx, idx, type_meta);
            }
            b::TypeDefBody::Interface => {
                self.insert_interface_type(mod_idx, idx, type_meta);
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
                    let res = item.serialize(&mut bytes, &self.cl_module);
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
        self.create_writable_sized(types::get_size(ty, self.modules, &self.cl_module))
    }

    pub fn create_writable_sized(&mut self, size: u32) -> cl::DataId {
        let data_id = self.cl_module.declare_anonymous_data(false, false).unwrap();
        let mut desc = cl::DataDescription::new();
        desc.define_zeroinit(size as usize);
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

        // If the function returns a struct, it is strictly required to be the first
        // argument, so the env will be the second
        let env_idx = match &func_binding.proto.ret_policy {
            ReturnPolicy::Struct(_) => 1,
            _ => 0,
        };
        sig.params.insert(
            env_idx,
            cl::AbiParam::new(self.cl_module.isa().pointer_type()),
        );

        let func = cl::Function::with_name_signature(
            cl::UserFuncName::user(FuncNS::Helper.into(), self.next_helper_id),
            sig,
        );
        self.next_helper_id += 1;

        let symbol_name = {
            let func_decl = &self.modules[mod_idx].funcs[func_idx];
            let mut params = func_decl
                .params
                .iter()
                .map(|v| &self.modules[mod_idx].values[*v].ty)
                .collect_vec();
            let void_ptr = b::Type::new(b::TypeBody::Ptr(None), None);
            params.insert(0, &void_ptr);
            NameMangler::new(self.modules).mangle(
                &func_decl.name.with("closure", b::NameIdentKind::Func, None),
                params,
            )
        };

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

    fn insert_type_impl(
        &mut self,
        mod_idx: usize,
        idx: usize,
        type_meta: &b::TypeMeta,
    ) -> Vec<cl::DataId> {
        let key = (mod_idx, idx);

        type_meta
            .ifaces
            .iter()
            .map(|iface_key| {
                let key = types::VTableRef::new(*iface_key, key);
                if let Some(data_id) = self.vtables_impl.get(&key) {
                    return *data_id;
                }

                let iface_def = &self.modules[iface_key.0].typedefs[iface_key.1];
                let b::TypeDefBody::Interface = &iface_def.body else {
                    panic!(
                        "type {}-{} should be an interface, I don't know what I got",
                        iface_key.0, iface_key.1
                    );
                };

                let vtable_desc =
                    self.insert_interface_type(iface_key.0, iface_key.1, iface_def);

                let tuple = vtable_desc
                    .methods
                    .clone()
                    .iter()
                    .map(|m| {
                        let func_ref = type_meta.methods[m].func_ref;
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
        typedef: &b::TypeDef,
    ) -> &types::VTableDesc {
        self.vtables_desc.entry((mod_idx, idx)).or_insert_with(|| {
            types::VTableDesc::new(typedef.methods.keys().cloned().collect())
        })
    }
}
