use std::collections::HashMap;
use std::mem;

use cranelift_shim::{self as cl, InstBuilder};
use derive_more::{Display, From};
use derive_new::new;
use derive_setters::Setters;
use itertools::Itertools;
use tracing::Value;

use crate::bytecode as b;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Display, Setters, new)]
#[display("{value_idx}")]
pub struct RuntimeValue {
    pub src: ValueSource,
    pub mod_idx: usize,
    pub value_idx: b::ValueIdx,
    #[new(value = "false")]
    pub is_ptr: bool,
}
impl RuntimeValue {
    pub fn ty<'m>(&self, modules: &'m [b::Module]) -> &'m b::Type {
        &modules[self.mod_idx].values[self.value_idx].ty
    }

    pub fn bytes(&self, modules: &[b::Module], obj_module: &impl cl::Module) -> u32 {
        get_size(self.ty(modules), modules, obj_module)
    }

    pub fn native_type(
        &self,
        modules: &[b::Module],
        obj_module: &impl cl::Module,
    ) -> Vec<cl::Type> {
        get_type(self.ty(modules), modules, obj_module)
    }

    /// Add value to function as a single Cranelift value. Values that are only logically
    /// grouped will be added to a stack slot and a pointer to it will be returned.
    pub fn add_value_to_func(
        &self,
        obj_module: &impl cl::Module,
        func: &mut cl::FunctionBuilder,
    ) -> cl::Value {
        match &self.src {
            ValueSource::Value(v) => *v,
            ValueSource::I8(n) => func.ins().iconst(cl::types::I8, *n as i64),
            ValueSource::I16(n) => func.ins().iconst(cl::types::I16, *n as i64),
            ValueSource::I32(n) => func.ins().iconst(cl::types::I32, *n as i64),
            ValueSource::I64(n) => {
                let n = unsafe { mem::transmute_copy::<u64, i64>(&n) };
                func.ins().iconst(cl::types::I64, n)
            }
            ValueSource::F32(n) => func.ins().f32const(n.to_float()),
            ValueSource::F64(n) => func.ins().f64const(n.to_float()),
            ValueSource::Data(data_id) => {
                let field_gv = obj_module.declare_data_in_func(*data_id, &mut func.func);
                func.ins()
                    .global_value(obj_module.isa().pointer_type(), field_gv)
            }
            ValueSource::StackSlot(ss) => {
                func.ins()
                    .stack_addr(obj_module.isa().pointer_type(), *ss, 0)
            }
            ValueSource::DynDispatched(..) => {
                todo!("dyn dispatched")
            }
            ValueSource::Func(..)
            | ValueSource::AppliedMethod(..)
            | ValueSource::AppliedMethodInderect(..) => {
                todo!("function references")
            }
        }
    }

    /// Add values to function as multiple Cranelift value. Values that are only logically
    /// grouped will continue to do so. Does not guarantee that the returned value was
    /// cloned and is inline.
    pub fn add_values_to_func(
        &self,
        obj_module: &impl cl::Module,
        func: &mut cl::FunctionBuilder,
    ) -> Vec<cl::Value> {
        match &self.src {
            ValueSource::Value(_)
            | ValueSource::I8(_)
            | ValueSource::I16(_)
            | ValueSource::I32(_)
            | ValueSource::I64(_)
            | ValueSource::F32(_)
            | ValueSource::F64(_)
            | ValueSource::Data(_)
            | ValueSource::StackSlot(_) => {
                vec![self.add_value_to_func(obj_module, func)]
            }
            ValueSource::DynDispatched(dispatched) => {
                vec![dispatched.src, dispatched.vtable]
            }
            ValueSource::Func(..)
            | ValueSource::AppliedMethod(..)
            | ValueSource::AppliedMethodInderect(..) => {
                todo!("function references")
            }
        }
    }
}

#[derive(Debug, Display, Clone, Copy, PartialEq, Eq, Hash, From)]
pub enum ValueSource {
    #[display("i8 {_0}")]
    I8(u8),
    #[display("i16 {_0}")]
    I16(u16),
    #[display("i32 {_0}")]
    I32(u32),
    #[display("i64 {_0}")]
    I64(u64),
    #[display("f32 {}", _0.to_float())]
    F32(F32Bits),
    #[display("f64 {}", _0.to_float())]
    F64(F64Bits),
    Value(cl::Value),
    Data(cl::DataId),
    StackSlot(cl::StackSlot),
    Func(cl::FuncId),
    #[display("{}-{} <- {_0}", _1.0, _1.1)]
    AppliedMethod(cl::Value, (usize, usize)),
    #[display("{_1} ({},{}) <- {_0}", _2.0, _2.1)]
    AppliedMethodInderect(cl::Value, cl::Value, (usize, usize)),
    #[display("{_0}")]
    DynDispatched(DynDispatched),
}
impl ValueSource {
    pub fn serialize(
        &self,
        bytes: &mut Vec<u8>,
        endianess: cl::Endianness,
    ) -> Result<(), ()> {
        macro_rules! serialize_number {
            ($n:expr) => {
                match endianess {
                    cl::Endianness::Little => bytes.extend(($n).to_le_bytes()),
                    cl::Endianness::Big => bytes.extend(($n).to_be_bytes()),
                }
            };
        }

        match self {
            ValueSource::I8(n) => bytes.push(*n),
            ValueSource::I16(n) => serialize_number!(n),
            ValueSource::I32(n) => serialize_number!(n),
            ValueSource::I64(n) => serialize_number!(n),
            ValueSource::F32(n) => serialize_number!(n.to_float()),
            ValueSource::F64(n) => serialize_number!(n.to_float()),
            ValueSource::Value(..)
            | ValueSource::Data(..)
            | ValueSource::StackSlot(..)
            | ValueSource::Func(..)
            | ValueSource::DynDispatched(..)
            | ValueSource::AppliedMethod(..)
            | ValueSource::AppliedMethodInderect(..) => return Err(()),
        }

        Ok(())
    }

    /// Creates a new ValueSource that replaces the inner value source with the provided
    /// values. The number of items and order required are defined by the kind of value
    /// specified, and should remain the same. Assumes that the place of the value is the
    /// same, i.g., referenced values stay referenced and inline values stays inline.
    pub fn with_values(&self, values: &[cl::Value]) -> Self {
        match self {
            ValueSource::Value(_)
            | ValueSource::I8(_)
            | ValueSource::I16(_)
            | ValueSource::I32(_)
            | ValueSource::I64(_)
            | ValueSource::F32(_)
            | ValueSource::F64(_)
            | ValueSource::Data(_)
            | ValueSource::StackSlot(_) => {
                assert_eq!(values.len(), 1);
                values[0].into()
            }
            ValueSource::DynDispatched(..) => {
                assert_eq!(values.len(), 2);
                DynDispatched::new(values[0], values[1]).into()
            }
            ValueSource::Func(..)
            | ValueSource::AppliedMethod(..)
            | ValueSource::AppliedMethodInderect(..) => {
                todo!("function references")
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct F32Bits(u32);
impl F32Bits {
    pub fn to_float(&self) -> f32 {
        f32::from_bits(self.0)
    }
}
impl From<f32> for F32Bits {
    fn from(value: f32) -> Self {
        Self(value.to_bits())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct F64Bits(u64);
impl F64Bits {
    pub fn to_float(&self) -> f64 {
        f64::from_bits(self.0)
    }
}
impl From<f64> for F64Bits {
    fn from(value: f64) -> Self {
        Self(value.to_bits())
    }
}

#[derive(Debug, Display, Copy, Clone, PartialEq, Eq, Hash, new)]
#[display("dyn {vtable} <- {src}")]
pub struct DynDispatched {
    pub src:    cl::Value,
    pub vtable: cl::Value,
}

pub fn tuple_from_record<'a>(
    fields: impl IntoIterator<Item = (&'a String, RuntimeValue)> + 'a,
    ty: &b::Type,
    modules: &[b::Module],
) -> Vec<RuntimeValue> {
    let fields: HashMap<_, _> = fields.into_iter().collect();

    let b::TypeBody::TypeRef(i, j) = &ty.body else {
        panic!("type is not a record type");
    };
    let b::TypeDefBody::Record(rec) = &modules[*i].typedefs[*j].body else {
        panic!("type is not a record type");
    };

    rec.fields
        .keys()
        .map(|key| *fields.get(key).expect(&format!("missing field: {key}")))
        .collect()
}

pub fn tuple_from_args(
    mod_idx: usize,
    values: &[b::ValueIdx],
    cl_values: &[cl::Value],
    modules: &[b::Module],
) -> Vec<RuntimeValue> {
    let mut cl_values = cl_values.iter();
    macro_rules! next_value {
        () => {
            *cl_values.next().unwrap()
        };
    }

    values
        .iter()
        .map(|v| {
            let ty = &modules[mod_idx].values[*v].ty;

            let (src, is_ptr) = match &ty.body {
                b::TypeBody::TypeRef(ty_mod_idx, ty_idx) => {
                    let typebody = &modules[*ty_mod_idx].typedefs[*ty_idx].body;
                    match typebody {
                        b::TypeDefBody::Interface(_) => {
                            let dispatched =
                                DynDispatched::new(next_value!(), next_value!());
                            (dispatched.into(), true)
                        }
                        b::TypeDefBody::Record(_) => (next_value!().into(), true),
                    }
                }
                b::TypeBody::Array(_) | b::TypeBody::String(_) => {
                    (next_value!().into(), true)
                }
                _ => (next_value!().into(), false),
            };
            RuntimeValue::new(src, mod_idx, *v).is_ptr(is_ptr)
        })
        .collect_vec()
}

pub fn get_type(
    ty: &b::Type,
    modules: &[b::Module],
    obj_module: &impl cl::Module,
) -> Vec<cl::Type> {
    match &ty.body {
        b::TypeBody::Bool => vec![cl::types::I8],
        b::TypeBody::I8 => vec![cl::types::I8],
        b::TypeBody::I16 => vec![cl::types::I16],
        b::TypeBody::I32 => vec![cl::types::I32],
        b::TypeBody::I64 => vec![cl::types::I64],
        b::TypeBody::U8 => vec![cl::types::I8],
        b::TypeBody::U16 => vec![cl::types::I16],
        b::TypeBody::U32 => vec![cl::types::I32],
        b::TypeBody::U64 => vec![cl::types::I64],
        b::TypeBody::F32 => vec![cl::types::F32],
        b::TypeBody::F64 => vec![cl::types::F64],
        b::TypeBody::USize
        | b::TypeBody::String(_)
        | b::TypeBody::Array(_)
        | b::TypeBody::Ptr(_)
        | b::TypeBody::SelfType(_, _) => vec![obj_module.isa().pointer_type()],
        b::TypeBody::TypeRef(i, j) => match &modules[*i].typedefs[*j].body {
            b::TypeDefBody::Record(_) => vec![obj_module.isa().pointer_type()],
            b::TypeDefBody::Interface(_) => vec![obj_module.isa().pointer_type(); 2],
        },
        b::TypeBody::AnyNumber
        | b::TypeBody::AnySignedNumber
        | b::TypeBody::AnyFloat
        | b::TypeBody::Inferred(_) => panic!("Type must be resolved before codegen"),
        b::TypeBody::Void => panic!("void type cannot be used directly"),
        b::TypeBody::Never => panic!("never type cannot be used directly"),
        b::TypeBody::AnyOpaque => panic!("anyopaque type cannot be used directly"),
        b::TypeBody::Func(_) => todo!("first-class functions are not supported yet"),
    }
}

pub fn get_size(
    ty: &b::Type,
    modules: &[b::Module],
    obj_module: &impl cl::Module,
) -> u32 {
    let ptr = obj_module.isa().pointer_bytes() as u32;

    match &ty.body {
        b::TypeBody::Void | b::TypeBody::Never => 0,
        b::TypeBody::String(s) => s.len.map_or(ptr, |len| ptr + len + 1),
        b::TypeBody::Array(a) => a.len.map_or(ptr, |len| {
            ptr + len
                * get_type(&a.item, modules, obj_module)
                    .into_iter()
                    .map(|ty| ty.bytes())
                    .sum::<u32>()
        }),
        b::TypeBody::TypeRef(i, j) => match &modules[*i].typedefs[*j].body {
            b::TypeDefBody::Record(rec) => rec
                .fields
                .values()
                .flat_map(|field| get_type(&field.ty, modules, obj_module))
                .map(|ty| ty.bytes())
                .sum(),
            b::TypeDefBody::Interface(_) => ptr * 2,
        },
        b::TypeBody::Bool
        | b::TypeBody::I8
        | b::TypeBody::U8
        | b::TypeBody::I16
        | b::TypeBody::U16
        | b::TypeBody::I32
        | b::TypeBody::U32
        | b::TypeBody::I64
        | b::TypeBody::U64
        | b::TypeBody::USize
        | b::TypeBody::F32
        | b::TypeBody::F64
        | b::TypeBody::Ptr(_)
        | b::TypeBody::SelfType(_, _) => get_type(ty, modules, obj_module)
            .into_iter()
            .map(|ty| ty.bytes())
            .sum(),
        b::TypeBody::AnyNumber
        | b::TypeBody::AnySignedNumber
        | b::TypeBody::AnyFloat
        | b::TypeBody::Inferred(_) => panic!("Type must be resolved before codegen"),
        b::TypeBody::AnyOpaque => panic!("anyopaque cannot be used directly"),
        b::TypeBody::Func(_) => todo!("first-class functions are not supported yet"),
    }
}

#[derive(new)]
pub struct VTableDesc {
    pub methods: Vec<String>,
}
impl VTableDesc {
    pub fn method_offset(
        &self,
        name: &str,
        obj_module: &impl cl::Module,
    ) -> Option<usize> {
        let ptr = obj_module.isa().pointer_bytes() as usize;
        self.methods
            .iter()
            .position(|m| *m == name)
            .map(|i| i * ptr)
    }
}

#[derive(new, Hash, PartialEq, Eq, Clone, Copy, Debug)]
pub struct VTableRef {
    pub iface: (usize, usize),
    pub ty:    (usize, usize),
}
