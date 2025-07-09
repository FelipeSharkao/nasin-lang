use std::collections::HashMap;

use cranelift_shim::{self as cl, Module};
use derive_new::new;
use itertools::repeat_n;

use super::func::{FuncCodegen, ResultPolicy};
use super::types;
use crate::{bytecode as b, config, utils};

#[derive(Debug)]
pub struct FuncBinding {
    pub is_extrn: bool,
    pub is_virt: bool,
    pub symbol_name: String,
    pub signature: cl::Signature,
    pub func_id: Option<cl::FuncId>,
    pub result_policy: ResultPolicy,
}

#[derive(Debug, Clone)]
pub struct GlobalBinding {
    pub symbol_name: String,
    pub value: types::RuntimeValue,
    pub is_const: bool,
    pub is_entry_point: bool,
}

#[derive(new)]
pub struct CodegenContext<'a> {
    pub modules: &'a [b::Module],
    pub cfg: &'a config::BuildConfig,
    pub obj_module: cl::ObjectModule,
    #[new(default)]
    pub funcs: HashMap<(usize, usize), FuncBinding>,
    #[new(default)]
    pub data: HashMap<cl::DataId, cl::DataDescription>,
    #[new(default)]
    pub globals: HashMap<(usize, usize), GlobalBinding>,
    #[new(default)]
    pub vtables_desc: HashMap<(usize, usize), types::VTableDesc>,
    #[new(default)]
    pub vtables_impl: HashMap<types::VTableRef, cl::DataId>,
    #[new(default)]
    strings: HashMap<String, cl::DataId>,
    #[new(default)]
    tuples: HashMap<Vec<types::ValueSource>, cl::DataId>,
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
        let symbol_name = format!("$global_{mod_idx}_{idx}");

        let (value, is_const) = utils::replace_with(self, |s| {
            let mut codegen = FuncCodegen::new(s, None);

            for instr in &global.body {
                if let b::InstrBody::Break(v) = &instr.body {
                    codegen.values.insert(global.value, codegen.values[v]);
                    break;
                }

                if codegen.value_from_instr(instr, mod_idx).is_none() {
                    let data_id = codegen.ctx.create_writable_for_type(&global_value.ty);
                    let value =
                        types::RuntimeValue::new(data_id.into(), mod_idx, global.value);
                    return (codegen.ctx, (value, false));
                }
            }

            (codegen.ctx, (codegen.values[&global.value], true))
        });

        self.globals.insert(
            (mod_idx, idx),
            GlobalBinding {
                symbol_name,
                value,
                is_const,
                is_entry_point: global.is_entry_point,
            },
        );
    }

    pub fn insert_type(&mut self, mod_idx: usize, idx: usize) {
        let typedef = &self.modules[mod_idx].typedefs[idx];

        if typedef.generics.len() > 0 {
            todo!()
        }

        match &typedef.body {
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

        let data_id = self
            .obj_module
            .declare_anonymous_data(false, false)
            .unwrap();
        let mut desc = cl::DataDescription::new();

        let mut bytes = vec![];

        match self.obj_module.isa().pointer_bytes() {
            1 => types::ValueSource::I8(value.len() as u8),
            2 => types::ValueSource::I16(value.len() as u16),
            4 => types::ValueSource::I32(value.len() as u32),
            8 => types::ValueSource::I64(value.len() as u64),
            _ => panic!("how many bytes?"),
        }
        .serialize(&mut bytes, self.obj_module.isa().endianness())
        .unwrap();

        bytes.extend(value.as_bytes());
        // Append a null terminator to avoid problems if used as a C string
        bytes.extend([0]);

        desc.define(bytes.into());
        self.obj_module.define_data(data_id, &desc).unwrap();

        self.data.insert(data_id, desc);
        self.strings.insert(value.to_string(), data_id);
        data_id
    }

    pub fn data_for_array(
        &mut self,
        mut values: Vec<types::ValueSource>,
    ) -> Option<cl::DataId> {
        let len = match self.obj_module.isa().pointer_bytes() {
            1 => types::ValueSource::I8(values.len() as u8),
            2 => types::ValueSource::I16(values.len() as u16),
            4 => types::ValueSource::I32(values.len() as u32),
            8 => types::ValueSource::I64(values.len() as u64),
            _ => panic!("how many bytes?"),
        };
        values.insert(0, len);
        self.data_for_tuple(values)
    }

    pub fn data_for_tuple(
        &mut self,
        values: Vec<types::ValueSource>,
    ) -> Option<cl::DataId> {
        if let Some(id) = self.tuples.get(&values) {
            return Some(*id);
        }

        let data_id = self
            .obj_module
            .declare_anonymous_data(false, false)
            .unwrap();
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
                        self.obj_module.isa().pointer_bytes() as usize,
                    ));

                    let field_gv =
                        included_datas.entry(field_data_id).or_insert_with(|| {
                            self.obj_module
                                .declare_data_in_data(*field_data_id, &mut desc)
                        });
                    desc.write_data_addr(offset as u32, field_gv.clone(), 0);
                }
                types::ValueSource::Func(func_id) => {
                    let offset = bytes.len();
                    bytes.extend(repeat_n(
                        0u8,
                        self.obj_module.isa().pointer_bytes() as usize,
                    ));

                    let func = included_funcs.entry(func_id).or_insert_with(|| {
                        self.obj_module.declare_func_in_data(*func_id, &mut desc)
                    });
                    desc.write_function_addr(offset as u32, func.clone());
                }
                _ => {
                    let res =
                        item.serialize(&mut bytes, self.obj_module.isa().endianness());
                    if res.is_err() {
                        return None;
                    }
                }
            }
        }

        desc.define(bytes.into());
        self.obj_module.define_data(data_id, &desc).unwrap();

        self.data.insert(data_id, desc);
        self.tuples.insert(values, data_id);
        Some(data_id)
    }

    pub fn create_writable_for_type(&mut self, ty: &b::Type) -> cl::DataId {
        let data_id = self
            .obj_module
            .declare_anonymous_data(false, false)
            .unwrap();
        let mut desc = cl::DataDescription::new();
        desc.define_zeroinit(types::get_size(ty, self.modules, &self.obj_module) as usize);
        self.obj_module.define_data(data_id, &desc).unwrap();

        self.data.insert(data_id, desc);
        data_id
    }

    fn insert_record_type(
        &mut self,
        mod_idx: usize,
        idx: usize,
        rec: &b::RecordType,
    ) -> Vec<cl::DataId> {
        let key = b::TypeRef::new(mod_idx, idx);

        rec.ifaces
            .iter()
            .map(|iface_impl| {
                let key = types::VTableRef::new(iface_impl.type_ref(), key);
                if let Some(data_id) = self.vtables_impl.get(&key) {
                    return *data_id;
                }

                let b::TypeDefBody::Interface(iface) =
                    &self.modules[iface_impl.mod_idx()].typedefs[iface_impl.ty_idx()]
                        .body
                else {
                    panic!(
                        "{} should be an interface, I don't know what I got",
                        iface_impl.type_ref()
                    );
                };

                let vtable_desc = self.insert_interface_type(
                    iface_impl.mod_idx(),
                    iface_impl.ty_idx(),
                    iface,
                );

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
