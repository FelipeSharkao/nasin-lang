use std::collections::HashMap;

use cranelift_shim as cl;
use derive_ctor::ctor;
use derive_more::{Display, From};
use itertools::{Itertools, enumerate};

use crate::bytecode as b;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Display, ctor)]
#[display("{value_idx}")]
pub struct RuntimeValue {
    pub src:       ValueSource,
    pub mod_idx:   usize,
    pub value_idx: b::ValueIdx,
}

#[derive(Debug, Display, Clone, PartialEq, Eq, Hash, From)]
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
    #[from(skip)]
    Primitive(cl::Value),
    #[from(skip)]
    PrimitiveAsPtr(cl::Value),
    #[from(skip)]
    #[display("ptr {}", _0)]
    Ptr(cl::Value),
    #[display("ptr ()")]
    UnitPtr,
    Data(cl::DataId),
    StackSlot(cl::StackSlot),
    #[display("{}", &*_0)]
    Slice(Box<Slice>),
    Func(cl::FuncId),
    FuncAsValue(FuncAsValue),
    #[display("method {}-{} <- {_0}", _1.0, _1.1)]
    AppliedMethod(cl::Value, (usize, usize)),
    #[display("method {_1} {_2} <- {_0}")]
    AppliedMethodInderect(cl::Value, cl::Value, FuncPrototype),
    #[display("{_0}")]
    DynDispatched(DynDispatched),
}
impl ValueSource {
    pub fn uint_ptr(v: u64, cl_module: &impl cl::Module) -> Self {
        match cl_module.isa().pointer_bytes() {
            1 => Self::I8(v as u8),
            2 => Self::I16(v as u16),
            4 => Self::I32(v as u32),
            8 => Self::I64(v as u64),
            _ => panic!("how many bytes?"),
        }
    }

    pub fn serialize(
        &self,
        bytes: &mut Vec<u8>,
        cl_module: &impl cl::Module,
    ) -> Result<(), ()> {
        let endianess = cl_module.isa().endianness();

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
            ValueSource::Slice(slice) => {
                slice.ptr.serialize(bytes, cl_module)?;
                slice.len.serialize(bytes, cl_module)?;
            }
            ValueSource::UnitPtr => match cl_module.isa().pointer_bytes() {
                1 => bytes.push(1),
                2 => serialize_number!(1u16),
                4 => serialize_number!(1u32),
                8 => serialize_number!(1u64),
                _ => panic!("how many bytes?"),
            },
            ValueSource::Primitive(..)
            | ValueSource::PrimitiveAsPtr(..)
            | ValueSource::Ptr(..)
            | ValueSource::Data(..)
            | ValueSource::StackSlot(..)
            | ValueSource::Func(..)
            | ValueSource::FuncAsValue(..)
            | ValueSource::DynDispatched(..)
            | ValueSource::AppliedMethod(..)
            | ValueSource::AppliedMethodInderect(..) => return Err(()),
        }

        Ok(())
    }

    /// Returns the number of values it would be needed to replace the inner value source.
    /// The number of items and order required are defined by the kind of value specified,
    /// and should remain the same. Assumes that the place of the value is the same, i.g.,
    /// referenced values stay referenced and inline values stays inline.
    pub fn count_values(&self) -> usize {
        match self {
            ValueSource::Primitive(_)
            | ValueSource::PrimitiveAsPtr(_)
            | ValueSource::Ptr(_)
            | ValueSource::UnitPtr
            | ValueSource::I8(_)
            | ValueSource::I16(_)
            | ValueSource::I32(_)
            | ValueSource::I64(_)
            | ValueSource::F32(_)
            | ValueSource::F64(_)
            | ValueSource::Data(_)
            | ValueSource::StackSlot(_) => 1,
            ValueSource::FuncAsValue(..) | ValueSource::DynDispatched(..) => 2,
            ValueSource::Slice(slice) => {
                slice.ptr.count_values() + slice.len.count_values()
            }
            ValueSource::Func(..)
            | ValueSource::AppliedMethod(..)
            | ValueSource::AppliedMethodInderect(..) => {
                todo!("function references")
            }
        }
    }

    /// Creates a new ValueSource that replaces the inner value source with the provided
    /// values. The number of items and order required are defined by the kind of value
    /// specified, and should remain the same. If this number is unknown, it should be
    /// queried with the `count_values` method. Assumes that the place of the value is the
    /// same, i.g., referenced values stay referenced and inline values stays inline.
    pub fn with_values(&self, values: &[cl::Value]) -> Self {
        assert_eq!(values.len(), self.count_values());
        match self {
            ValueSource::Primitive(_)
            | ValueSource::I8(_)
            | ValueSource::I16(_)
            | ValueSource::I32(_)
            | ValueSource::I64(_)
            | ValueSource::F32(_)
            | ValueSource::F64(_) => ValueSource::Primitive(values[0]),
            ValueSource::PrimitiveAsPtr(_) => ValueSource::PrimitiveAsPtr(values[0]),
            ValueSource::Ptr(_)
            | ValueSource::UnitPtr
            | ValueSource::Data(_)
            | ValueSource::StackSlot(_) => ValueSource::Ptr(values[0]),
            ValueSource::Slice(slice) => {
                let ptr_count = slice.ptr.count_values();
                let ptr = slice.ptr.with_values(&values[..ptr_count]);
                let len = slice.len.with_values(&values[ptr_count..]);
                Box::new(Slice::new(ptr, len)).into()
            }
            ValueSource::FuncAsValue(func_as_value) => {
                FuncAsValue::new(values[0], values[1], func_as_value.proto.clone()).into()
            }
            ValueSource::DynDispatched(..) => {
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

#[derive(Debug, Display, Clone, Copy, PartialEq, Eq, Hash, From)]
pub enum Const {
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
}

impl Const {
    pub fn uint_ptr(v: u64, cl_module: &impl cl::Module) -> Self {
        match cl_module.isa().pointer_bytes() {
            1 => Self::I8(v as u8),
            2 => Self::I16(v as u16),
            4 => Self::I32(v as u32),
            8 => Self::I64(v as u64),
            _ => panic!("how many bytes?"),
        }
    }
}

impl TryFrom<&ValueSource> for Const {
    type Error = ();

    fn try_from(value: &ValueSource) -> Result<Self, Self::Error> {
        match value {
            ValueSource::I8(n) => Ok(Self::I8(*n)),
            ValueSource::I16(n) => Ok(Self::I16(*n)),
            ValueSource::I32(n) => Ok(Self::I32(*n)),
            ValueSource::I64(n) => Ok(Self::I64(*n)),
            ValueSource::F32(n) => Ok(Self::F32(*n)),
            ValueSource::F64(n) => Ok(Self::F64(*n)),
            _ => Err(()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct F32Bits(u32);
impl F32Bits {
    pub fn to_float(&self) -> f32 {
        f32::from_bits(self.0)
    }

    pub fn from_float(value: f32) -> Self {
        Self(value.to_bits())
    }
}
impl From<f32> for F32Bits {
    fn from(value: f32) -> Self {
        Self::from_float(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct F64Bits(u64);
impl F64Bits {
    pub fn to_float(&self) -> f64 {
        f64::from_bits(self.0)
    }

    pub fn from_float(value: f64) -> Self {
        Self(value.to_bits())
    }
}
impl From<f64> for F64Bits {
    fn from(value: f64) -> Self {
        Self::from_float(value)
    }
}

#[derive(Debug, Display, Clone, PartialEq, Eq, Hash, ctor)]
#[display("[{ptr}; {len}]")]
pub struct Slice {
    pub ptr: ValueSource,
    pub len: ValueSource,
}

#[derive(Debug, Display, Clone, PartialEq, Eq, Hash, ctor)]
#[display("func {ptr} {proto} <- {env}")]
pub struct FuncAsValue {
    pub ptr:   cl::Value,
    pub env:   cl::Value,
    pub proto: FuncPrototype,
}

#[derive(Debug, Display, Copy, Clone, PartialEq, Eq, Hash, ctor)]
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

    let b::TypeBody::TypeRef(ty_ref) = &ty.body else {
        panic!("type is not a record type");
    };
    let b::TypeDefBody::Record(rec) = &ty_ref.get_typedef(modules).body else {
        panic!("type is not a record type");
    };

    rec.fields
        .keys()
        .map(|key| fields.get(key).expect(&format!("missing field: {key}")))
        .cloned()
        .collect()
}

pub fn tuple_from_args(
    mod_idx: usize,
    values: &[b::ValueIdx],
    cl_values: &[cl::Value],
    modules: &[b::Module],
    cl_module: &impl cl::Module,
    is_virtual: bool,
) -> Vec<RuntimeValue> {
    let mut c = 0;
    enumerate(values)
        .map(|(i, &v)| {
            let (res, n) = take_value_from_args(
                mod_idx,
                v,
                &cl_values[c..],
                modules,
                cl_module,
                is_virtual && i == 0,
            );
            c += n;
            res
        })
        .collect_vec()
}

pub fn take_value_from_args(
    mod_idx: usize,
    idx: usize,
    cl_values: &[cl::Value],
    modules: &[b::Module],
    cl_module: &impl cl::Module,
    is_receiver: bool,
) -> (RuntimeValue, usize) {
    let ty = &modules[mod_idx].values[idx].ty;

    let mut n = 0;
    let mut next = || {
        let value = cl_values[n];
        n += 1;
        value
    };

    let src = if is_receiver {
        match &ty.body {
            b::TypeBody::TypeRef(ty_ref) => match &ty_ref.get_typedef(modules).body {
                b::TypeDefBody::Record(_) | b::TypeDefBody::Interface => {
                    ValueSource::Ptr(next())
                }
                b::TypeDefBody::Builtin(builtin) => match builtin {
                    b::BuiltinType::String
                    | b::BuiltinType::Array
                    | b::BuiltinType::Ptr => ValueSource::Ptr(next()),
                    _ => ValueSource::PrimitiveAsPtr(next()),
                },
            },
            _ => ValueSource::Ptr(next()),
        }
    } else {
        match &ty.body {
            b::TypeBody::TypeRef(ty_ref) => match &ty_ref.get_typedef(modules).body {
                b::TypeDefBody::Record(_) => ValueSource::Ptr(next()),
                b::TypeDefBody::Interface => DynDispatched::new(next(), next()).into(),
                b::TypeDefBody::Builtin(builtin) => match builtin {
                    b::BuiltinType::String | b::BuiltinType::Array => {
                        Box::new(Slice::new(
                            ValueSource::Ptr(next()),
                            ValueSource::Primitive(next()),
                        ))
                        .into()
                    }
                    b::BuiltinType::Ptr => ValueSource::Ptr(next()),
                    _ => ValueSource::Primitive(next()),
                },
            },
            b::TypeBody::Func(func_ty) => {
                let proto = FuncPrototype::from_closure_type(func_ty, modules, cl_module);
                FuncAsValue::new(next(), next(), proto).into()
            }
            _ => ValueSource::Ptr(next()),
        }
    };

    (RuntimeValue::new(src, mod_idx, idx), n)
}

pub fn get_type_canonical(
    ty: &b::Type,
    modules: &[b::Module],
    cl_module: &impl cl::Module,
    is_receiver: bool,
) -> Vec<cl::Type> {
    if is_receiver {
        return vec![cl_module.isa().pointer_type()];
    }

    match &ty.body {
        b::TypeBody::TypeRef(t) => match &t.get_typedef(modules).body {
            b::TypeDefBody::Record(_) => vec![cl_module.isa().pointer_type()],
            b::TypeDefBody::Interface => vec![cl_module.isa().pointer_type(); 2],
            b::TypeDefBody::Builtin(builtin) => match builtin {
                b::BuiltinType::Ptr => vec![cl_module.isa().pointer_type()],
                _ => get_type_by_type(ty, modules, cl_module),
            },
        },
        _ => get_type_by_type(ty, modules, cl_module),
    }
}

pub fn get_type_by_type(
    ty: &b::Type,
    modules: &[b::Module],
    cl_module: &impl cl::Module,
) -> Vec<cl::Type> {
    match &ty.body {
        b::TypeBody::TypeRef(t) => match &t.get_typedef(modules).body {
            b::TypeDefBody::Record(rec) => rec
                .fields
                .values()
                .flat_map(|field| get_type_by_type(&field.ty, modules, cl_module))
                .collect_vec(),
            b::TypeDefBody::Interface => vec![cl_module.isa().pointer_type(); 2],
            b::TypeDefBody::Builtin(builtin) => match builtin {
                b::BuiltinType::Bool => vec![cl::types::I8],
                b::BuiltinType::I8 => vec![cl::types::I8],
                b::BuiltinType::I16 => vec![cl::types::I16],
                b::BuiltinType::I32 => vec![cl::types::I32],
                b::BuiltinType::I64 => vec![cl::types::I64],
                b::BuiltinType::U8 => vec![cl::types::I8],
                b::BuiltinType::U16 => vec![cl::types::I16],
                b::BuiltinType::U32 => vec![cl::types::I32],
                b::BuiltinType::U64 => vec![cl::types::I64],
                b::BuiltinType::F32 => vec![cl::types::F32],
                b::BuiltinType::F64 => vec![cl::types::F64],
                b::BuiltinType::USize | b::BuiltinType::Ptr => {
                    vec![cl_module.isa().pointer_type()]
                }
                b::BuiltinType::String | b::BuiltinType::Array => {
                    vec![cl_module.isa().pointer_type(); 2]
                }
                b::BuiltinType::Void => vec![],
                b::BuiltinType::AnyNumber
                | b::BuiltinType::AnySignedNumber
                | b::BuiltinType::AnyFloat => {
                    panic!("Type must be resolved before codegen")
                }
                b::BuiltinType::Never => panic!("never type cannot be used directly"),
                b::BuiltinType::AnyOpaque => {
                    panic!("anyopaque type cannot be used directly")
                }
            },
        },
        b::TypeBody::Func(_) => {
            vec![cl_module.isa().pointer_type(); 2]
        }
        b::TypeBody::TypeVar(_) => {
            panic!("TypeVar should have been instantiated by transform phase")
        }
        b::TypeBody::Inferred(_) => panic!("Type must be resolved before codegen"),
    }
}

pub fn get_size(ty: &b::Type, modules: &[b::Module], cl_module: &impl cl::Module) -> u32 {
    let ptr = cl_module.isa().pointer_bytes() as u32;

    match &ty.body {
        b::TypeBody::TypeRef(t) => match &t.get_typedef(modules).body {
            b::TypeDefBody::Record(rec) => rec
                .fields
                .values()
                .flat_map(|field| get_type_by_type(&field.ty, modules, cl_module))
                .map(|ty| ty.bytes())
                .sum(),
            b::TypeDefBody::Interface => ptr * 2,
            b::TypeDefBody::Builtin(builtin) => match builtin {
                b::BuiltinType::Void | b::BuiltinType::Never => 0,
                b::BuiltinType::Bool
                | b::BuiltinType::I8
                | b::BuiltinType::U8
                | b::BuiltinType::I16
                | b::BuiltinType::U16
                | b::BuiltinType::I32
                | b::BuiltinType::U32
                | b::BuiltinType::I64
                | b::BuiltinType::U64
                | b::BuiltinType::USize
                | b::BuiltinType::F32
                | b::BuiltinType::F64
                | b::BuiltinType::String
                | b::BuiltinType::Array
                | b::BuiltinType::Ptr => get_type_by_type(ty, modules, cl_module)
                    .into_iter()
                    .map(|ty| ty.bytes())
                    .sum(),
                b::BuiltinType::AnyNumber
                | b::BuiltinType::AnySignedNumber
                | b::BuiltinType::AnyFloat => {
                    panic!("Type must be resolved before codegen")
                }
                b::BuiltinType::AnyOpaque => panic!("anyopaque cannot be used directly"),
            },
        },
        b::TypeBody::Inferred(_) => panic!("Type must be resolved before codegen"),
        b::TypeBody::Func(_) => todo!("first-class functions are not supported yet"),
        b::TypeBody::TypeVar(_) => {
            panic!("TypeVar should have been instantiated by transform phase")
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Display)]
pub enum ReturnPolicy {
    #[display("normal")]
    Normal,
    #[display("struct({_0})")]
    Struct(u32),
    #[display("no_return")]
    NoReturn,
    #[display("void")]
    Void,
}
impl ReturnPolicy {
    pub fn from_func(
        mod_idx: usize,
        func_idx: usize,
        modules: &[b::Module],
        cl_module: &impl cl::Module,
    ) -> Self {
        let func = &modules[mod_idx].funcs[func_idx];
        let ret_ty = &modules[mod_idx].values[func.ret].ty;
        Self::from_ret_type(ret_ty, modules, cl_module)
    }

    pub fn from_ret_type(
        ty: &b::Type,
        modules: &[b::Module],
        cl_module: &impl cl::Module,
    ) -> Self {
        if ty.is_never(modules) {
            Self::NoReturn
        } else if ty.is_aggregate(modules) {
            let size = get_size(ty, modules, cl_module);
            Self::Struct(size as u32)
        } else if ty.is_void(modules) {
            Self::Void
        } else {
            Self::Normal
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Display, ctor)]
#[display("{signature} {ret_policy}")]
pub struct FuncPrototype {
    pub signature:  cl::Signature,
    pub ret_policy: ReturnPolicy,
}
impl FuncPrototype {
    pub fn from_func(
        mod_idx: usize,
        func_idx: usize,
        modules: &[b::Module],
        cl_module: &impl cl::Module,
    ) -> Self {
        let func = &modules[mod_idx].funcs[func_idx];
        let is_method = func.method.as_ref().is_some_and(|x| x.is_virtual);

        let mut sig = cl_module.make_signature();

        let ret_ty = &modules[mod_idx].values[func.ret].ty;
        let ret_policy = ReturnPolicy::from_ret_type(ret_ty, modules, cl_module);
        match ret_policy {
            ReturnPolicy::Struct(_) => {
                let ret_param = cl::AbiParam::special(
                    cl_module.isa().pointer_type(),
                    cl::ArgumentPurpose::StructReturn,
                );
                sig.params.push(ret_param);
            }
            ReturnPolicy::Normal => {
                let native_ty = get_type_canonical(ret_ty, modules, cl_module, false);
                assert_eq!(native_ty.len(), 1);
                sig.returns.push(cl::AbiParam::new(native_ty[0]));
            }
            ReturnPolicy::Void | ReturnPolicy::NoReturn => {}
        }

        for (i, param) in enumerate(&func.params) {
            let ty = &modules[mod_idx].values[*param].ty;
            for native_ty in
                get_type_canonical(ty, modules, cl_module, is_method && i == 0)
            {
                sig.params.push(cl::AbiParam::new(native_ty));
            }
        }

        Self::new(sig, ret_policy)
    }

    pub fn from_func_type(
        func_ty: &b::FuncType,
        modules: &[b::Module],
        cl_module: &impl cl::Module,
    ) -> Self {
        let mut sig = cl_module.make_signature();

        let ret_ty = &func_ty.ret;
        let ret_policy = ReturnPolicy::from_ret_type(ret_ty, modules, cl_module);
        match ret_policy {
            ReturnPolicy::Struct(_) => {
                let ret_param = cl::AbiParam::special(
                    cl_module.isa().pointer_type(),
                    cl::ArgumentPurpose::StructReturn,
                );
                sig.params.push(ret_param);
            }
            ReturnPolicy::Normal => {
                let native_ty = get_type_canonical(ret_ty, modules, cl_module, false);
                assert_eq!(native_ty.len(), 1);
                sig.returns.push(cl::AbiParam::new(native_ty[0]));
            }
            ReturnPolicy::Void | ReturnPolicy::NoReturn => {}
        }

        for param in &func_ty.params {
            for native_ty in get_type_canonical(param, modules, cl_module, false) {
                sig.params.push(cl::AbiParam::new(native_ty));
            }
        }

        Self::new(sig, ret_policy)
    }

    pub fn from_closure_type(
        func_ty: &b::FuncType,
        modules: &[b::Module],
        cl_module: &impl cl::Module,
    ) -> Self {
        let mut proto = Self::from_func_type(func_ty, modules, cl_module);

        let idx = proto.first_param_index();
        proto
            .signature
            .params
            .insert(idx, cl::AbiParam::new(cl_module.isa().pointer_type()));

        proto
    }

    /// Returns the index of the first parameter of the function after special parameters
    /// like struct return pointer.
    pub fn first_param_index(&self) -> usize {
        sig_first_param_index(&self.signature)
    }
}

/// Returns the index of the first parameter of the function after special parameters
/// like struct return pointer. If the function has none, returns the index where one
/// would be inserted.
pub fn sig_first_param_index(sig: &cl::Signature) -> usize {
    sig.params
        .iter()
        .position(|x| {
            matches!(
                x.purpose,
                cl::ArgumentPurpose::Normal | cl::ArgumentPurpose::StructArgument(_)
            )
        })
        .unwrap_or(sig.params.len())
}

#[derive(ctor)]
pub struct VTableDesc {
    pub methods: Vec<String>,
}
impl VTableDesc {
    pub fn method_offset(
        &self,
        name: &str,
        cl_module: &impl cl::Module,
    ) -> Option<usize> {
        let ptr = cl_module.isa().pointer_bytes() as usize;
        self.methods
            .iter()
            .position(|m| *m == name)
            .map(|i| i * ptr)
    }
}

#[derive(ctor, Hash, PartialEq, Eq, Clone, Copy, Debug)]
pub struct VTableRef {
    pub iface_key: b::TypeRefKey,
    pub ty_key:    b::TypeRefKey,
}
