use std::borrow::Cow;
use std::fmt;
use std::hash::Hash;

use derive_more::Display;
use derive_new::new;
use itertools::{chain, izip, Itertools};

use super::{Loc, Module, TypeDefBody};
use crate::utils;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TypeBody {
    Void,
    Never,
    Bool,
    AnyOpaque,
    // FIXME: use interface/trait for this
    AnyNumber,
    // FIXME: use interface/trait for this
    AnySignedNumber,
    // FIXME: use interface/trait for this
    AnyFloat,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    USize,
    F32,
    F64,
    Inferred(InferredType),
    String(StringType),
    Array(ArrayType),
    Ptr(Box<Type>),
    Func(Box<FuncType>),
    TypeRef(usize, usize),
}
impl Display for TypeBody {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TypeBody::Void => write!(f, "void")?,
            TypeBody::Never => write!(f, "never")?,
            TypeBody::Bool => write!(f, "bool")?,
            TypeBody::AnyNumber => write!(f, "AnyNumber")?,
            TypeBody::AnySignedNumber => write!(f, "AnySignedNumber")?,
            TypeBody::AnyFloat => write!(f, "AnyFloat")?,
            TypeBody::AnyOpaque => write!(f, "anyopaque")?,
            TypeBody::I8 => write!(f, "i8")?,
            TypeBody::I16 => write!(f, "i16")?,
            TypeBody::I32 => write!(f, "i32")?,
            TypeBody::I64 => write!(f, "i64")?,
            TypeBody::U8 => write!(f, "u8")?,
            TypeBody::U16 => write!(f, "u16")?,
            TypeBody::U32 => write!(f, "u32")?,
            TypeBody::U64 => write!(f, "u64")?,
            TypeBody::USize => write!(f, "usize")?,
            TypeBody::F32 => write!(f, "f32")?,
            TypeBody::F64 => write!(f, "f64")?,
            TypeBody::Inferred(v) => {
                write!(f, "infered {{")?;
                for (name, t) in &v.members {
                    write!(f, " {name}: {t}")?;
                }
                for (name, t) in &v.properties {
                    write!(f, " .{name}: {t}")?;
                }
                write!(f, " }}")?;
            }
            TypeBody::String(v) => {
                write!(f, "string")?;
                if let Some(len) = v.len {
                    write!(f, " {}", len)?;
                }
            }
            TypeBody::Array(v) => {
                write!(f, "array {}", v.item)?;
                if let Some(len) = v.len {
                    write!(f, " {}", len)?;
                }
            }
            TypeBody::Ptr(ty) => write!(f, "ptr {ty}")?,
            TypeBody::Func(func) => {
                write!(f, "func {}: {}", func.params.iter().join(", "), &func.ret)?
            }
            TypeBody::TypeRef(mod_idx, ty_idx) => write!(f, "type {mod_idx}-{ty_idx}")?,
        }
        Ok(())
    }
}
impl TypeBody {
    pub fn unknown() -> Self {
        TypeBody::Inferred(InferredType {
            members: utils::SortedMap::new(),
            properties: utils::SortedMap::new(),
        })
    }
    pub fn is_unknown(&self) -> bool {
        if let TypeBody::Inferred(i) = self {
            return i.members.is_empty() && i.properties.is_empty();
        }
        false
    }
}

#[derive(Debug, Clone, new)]
pub struct Type {
    pub body: TypeBody,
    pub loc: Option<Loc>,
}

macro_rules! unordered {
    ($a:pat, $b:pat) => {
        ($a, $b) | ($b, $a)
    };
}
macro_rules! number {
    ($var:ident $( , $gen:ident)*) => {
        unordered!(
            Type { body: TypeBody::$var, loc: _ },
            Type { body: TypeBody::AnyNumber $( | TypeBody::$gen )*, loc: _ })
    };
}
macro_rules! body {
    ($pat:pat) => {
        Type { body: $pat, loc: _ }
    };
}
impl Type {
    pub fn unknown(loc: Option<Loc>) -> Self {
        Type::new(TypeBody::unknown(), loc)
    }

    pub fn is_unknown(&self) -> bool {
        self.body.is_unknown()
    }

    pub fn is_inferred(&self) -> bool {
        matches!(&self.body, TypeBody::Inferred(_))
    }

    pub fn is_aggregate(&self, modules: &[Module]) -> bool {
        match &self.body {
            //TypeBody::String(_) | TypeBody::Array(_) => true,
            TypeBody::TypeRef(mod_idx, ty_idx) => {
                match &modules[*mod_idx].typedefs[*ty_idx].body {
                    TypeDefBody::Record(_) => true,
                }
            }
            _ => false,
        }
    }

    pub fn is_primitive(&self) -> bool {
        self.is_bool() || self.is_number() || matches!(&self.body, TypeBody::Ptr(_))
    }

    pub fn is_bool(&self) -> bool {
        matches!(&self.body, TypeBody::Bool)
    }

    pub fn is_number(&self) -> bool {
        matches!(&self.body, TypeBody::AnyNumber | TypeBody::AnySignedNumber)
            || self.is_sint()
            || self.is_uint()
            || self.is_float()
    }

    pub fn is_int(&self) -> bool {
        self.is_sint() || self.is_uint()
    }

    pub fn is_sint(&self) -> bool {
        matches!(
            &self.body,
            TypeBody::I8 | TypeBody::I16 | TypeBody::I32 | TypeBody::I64
        )
    }

    pub fn is_uint(&self) -> bool {
        matches!(
            &self.body,
            TypeBody::U8
                | TypeBody::U16
                | TypeBody::U32
                | TypeBody::U64
                | TypeBody::USize
        )
    }

    pub fn is_float(&self) -> bool {
        matches!(
            &self.body,
            TypeBody::AnyFloat | TypeBody::F32 | TypeBody::F64
        )
    }

    pub fn is_never(&self) -> bool {
        matches!(&self.body, TypeBody::Never)
    }

    pub fn field<'a>(&'a self, name: &str, modules: &'a [Module]) -> Option<&'a Type> {
        match &self.body {
            TypeBody::Inferred(v) => v.members.get(name),
            TypeBody::TypeRef(mod_idx, ty_idx) => {
                match &modules.get(*mod_idx)?.typedefs.get(*ty_idx)?.body {
                    TypeDefBody::Record(rec) => Some(&rec.fields.get(name)?.ty),
                }
            }
            _ => None,
        }
    }

    pub fn method<'a>(
        &'a self,
        name: &str,
        modules: &'a [Module],
    ) -> Option<Cow<'a, Type>> {
        match &self.body {
            TypeBody::TypeRef(mod_idx, ty_idx) => {
                let typedef = modules.get(*mod_idx)?.typedefs.get(*ty_idx)?;
                let method = match &typedef.body {
                    TypeDefBody::Record(rec) => rec.methods.get(name),
                }?;
                let method_mod = modules.get(method.func_ref.0)?;
                let func = &method_mod.funcs[method.func_ref.1];
                let params_tys = func
                    .params
                    .iter()
                    .map(|param| method_mod.values[*param].ty.clone())
                    .collect_vec();
                let ret_ty = method_mod.values[func.ret].ty.clone();
                Some(Cow::Owned(Type::new(
                    TypeBody::Func(Box::new(FuncType::new(params_tys, ret_ty))),
                    Some(method.loc),
                )))
            }
            _ => None,
        }
    }

    pub fn property<'a>(
        &'a self,
        name: &str,
        modules: &'a [Module],
    ) -> Option<Cow<'a, Type>> {
        if let Some(ty) = self.method(name, modules) {
            let TypeBody::Func(func) = &ty.body else {
                return None;
            };
            let [params @ .., self_param] = &func.params[..] else {
                return None;
            };
            // is static?
            if self_param.body != self.body {
                return None;
            }
            // functions without parameters are just values
            if params.len() == 0 {
                return Some(Cow::Owned(func.ret.clone()));
            }
            return Some(Cow::Owned(Type::new(
                TypeBody::Func(Box::new(FuncType::new(
                    params.to_vec(),
                    func.ret.clone(),
                ))),
                ty.loc,
            )));
        }
        if let TypeBody::Inferred(v) = &self.body {
            return v.properties.get(name).map(|v| Cow::Borrowed(v));
        }
        if let Some(ty) = self.field(name, modules) {
            return Some(Cow::Borrowed(ty));
        }
        None
    }

    pub fn intersection(&self, other: &Type, modules: &[Module]) -> Option<Type> {
        let body = match (self, other) {
            // INFO: This is not correct, an intersection with `never` and `a` should be
            // `never`, not `a`, but due to the way that `if` branches are checked, this
            // was necessary, and I reckon it won't be all that harmful. Maybe I'll fix it
            // later when it becomes a problem
            unordered!(body!(TypeBody::Never), body!(a)) => a.clone(),
            number!(U8) => TypeBody::U8,
            number!(U16) => TypeBody::U16,
            number!(U32) => TypeBody::U32,
            number!(U64) => TypeBody::U64,
            number!(USize) => TypeBody::USize,
            number!(I8, AnySignedNumber) => TypeBody::I8,
            number!(I16, AnySignedNumber) => TypeBody::I16,
            number!(I32, AnySignedNumber) => TypeBody::I32,
            number!(I64, AnySignedNumber) => TypeBody::I64,
            number!(F32, AnySignedNumber, AnyFloat) => TypeBody::F32,
            number!(F64, AnySignedNumber, AnyFloat) => TypeBody::F64,
            (body!(TypeBody::String(a)), body!(TypeBody::String(b))) => {
                let len = match (&a.len, &b.len) {
                    (a_len, b_len) if a_len == b_len => a_len.clone(),
                    (Some(len), None) | (None, Some(len)) => Some(*len),
                    _ => return None,
                };
                TypeBody::String(StringType { len })
            }
            (body!(TypeBody::Array(a)), body!(TypeBody::Array(b))) => {
                let len = match (&a.len, &b.len) {
                    (a_len, b_len) if a_len == b_len => a_len.clone(),
                    (Some(len), None) | (None, Some(len)) => Some(*len),
                    _ => return None,
                };
                TypeBody::Array(ArrayType {
                    len,
                    item: a.item.intersection(&b.item, modules)?.into(),
                })
            }
            (body!(TypeBody::Ptr(a)), body!(TypeBody::Ptr(b))) => {
                TypeBody::Ptr(a.intersection(&b, modules)?.into())
            }
            (body!(TypeBody::Func(a)), body!(TypeBody::Func(b))) => {
                TypeBody::Func(a.intersection(b, modules)?.into())
            }
            (body!(TypeBody::Inferred(a)), body!(TypeBody::Inferred(b))) => {
                let fields = chain!(a.members.keys(), b.members.keys())
                    .unique()
                    .map(|name| {
                        let ty = match (a.members.get(name), b.members.get(name)) {
                            (Some(a_member), Some(b_member)) => {
                                a_member.intersection(b_member, modules)?
                            }
                            unordered!(Some(field), None) => field.clone(),
                            _ => unreachable!(),
                        };
                        Some((name.to_string(), ty))
                    })
                    .collect::<Option<_>>()?;
                let methods = chain!(a.properties.keys(), b.properties.keys())
                    .unique()
                    .map(|name| {
                        let method =
                            match (a.properties.get(name), b.properties.get(name)) {
                                (Some(a_method), Some(b_method)) => {
                                    a_method.intersection(b_method, modules)?
                                }
                                unordered!(Some(method), None) => method.clone(),
                                _ => unreachable!(),
                            };
                        Some((name.to_string(), method))
                    })
                    .collect::<Option<_>>()?;
                TypeBody::Inferred(InferredType {
                    members: fields,
                    properties: methods,
                })
            }
            unordered!(body!(TypeBody::Inferred(a)), b) => {
                let has_all_members = a.members.iter().all(|(name, a_ty)| {
                    other
                        .field(name, modules)
                        .is_some_and(|b_ty| a_ty.intersection(b_ty, modules).is_some())
                });
                let has_all_props = a.properties.iter().all(|(name, a_ty)| {
                    other
                        .property(name, modules)
                        .is_some_and(|b_ty| a_ty.intersection(&b_ty, modules).is_some())
                });
                if has_all_members && has_all_props {
                    b.body.clone()
                } else {
                    return None;
                }
            }
            (body!(a), body!(b)) if a == b => a.clone(),
            _ => return None,
        };
        let loc = match (&self.loc, &other.loc) {
            unordered!(Some(loc), None) => Some(*loc),
            (Some(a), Some(b)) => {
                if a == b {
                    Some(*a)
                } else {
                    None
                }
            }
            (None, None) => None,
        };
        Some(Type::new(body, loc))
    }

    pub fn union(&self, other: &Type, modules: &[Module]) -> Option<Type> {
        let body = match (self, other) {
            unordered!(body!(TypeBody::Never), body!(a)) => a.clone(),
            (body!(TypeBody::String(a)), body!(TypeBody::String(b))) => {
                TypeBody::String(StringType {
                    len: if a.len == b.len { a.len.clone() } else { None },
                })
            }
            (body!(TypeBody::Array(a)), body!(TypeBody::Array(b))) => {
                TypeBody::Array(ArrayType {
                    item: a.item.union(&b.item, modules)?.into(),
                    len: if a.len == b.len { a.len.clone() } else { None },
                })
            }
            (body!(TypeBody::Ptr(a)), body!(TypeBody::Ptr(b))) => {
                TypeBody::Ptr(a.union(&b, modules)?.into())
            }
            (body!(TypeBody::Func(a)), body!(TypeBody::Func(b))) => {
                TypeBody::Func(a.union(b, modules)?.into())
            }
            (body!(TypeBody::Inferred(a)), body!(TypeBody::Inferred(b))) => {
                let fields = chain!(a.members.keys(), b.members.keys())
                    .unique()
                    .map(|name| {
                        let ty = match (a.members.get(name), b.members.get(name)) {
                            (Some(a_member), Some(b_member)) => {
                                a_member.union(b_member, modules)?
                            }
                            _ => unreachable!(),
                        };
                        Some((name.to_string(), ty))
                    })
                    .collect::<Option<_>>()?;
                let props = chain!(a.properties.keys(), b.properties.keys())
                    .unique()
                    .filter_map(|name| {
                        let prop = match (a.properties.get(name), b.properties.get(name))
                        {
                            (Some(a_prop), Some(b_prop)) => {
                                match a_prop.union(b_prop, modules) {
                                    Some(p) => p,
                                    // If there's no common signature, so the type is
                                    // impossible
                                    None => return Some(None),
                                }
                            }
                            // Omit field present in only one
                            unordered!(Some(_), None) => return None,
                            _ => unreachable!(),
                        };
                        Some(Some((name.to_string(), prop))) // that's so ugly
                    })
                    .collect::<Option<_>>()?;
                TypeBody::Inferred(InferredType {
                    members: fields,
                    properties: props,
                })
            }
            unordered!(body!(TypeBody::Inferred(a)), b) => {
                let has_all_members = a.members.iter().all(|(name, a_ty)| {
                    other
                        .field(name, modules)
                        .is_some_and(|b_ty| a_ty.union(b_ty, modules).is_some())
                });
                let has_all_props = a.properties.iter().all(|(name, a_ty)| {
                    other
                        .property(name, modules)
                        .is_some_and(|b_ty| a_ty.union(&b_ty, modules).is_some())
                });
                if has_all_members && has_all_props {
                    b.body.clone()
                } else {
                    return None;
                }
            }
            (a, b) => {
                if &a.body == &b.body {
                    a.body.clone()
                } else {
                    return a.intersection(b, modules);
                }
            }
        };
        let loc = match (&self.loc, &other.loc) {
            unordered!(Some(loc), None) => Some(loc.clone()),
            (Some(_), Some(_)) | (None, None) => None,
        };
        Some(Type::new(body, loc))
    }
}
impl PartialEq for Type {
    fn eq(&self, other: &Self) -> bool {
        &self.body == &other.body
    }
}
impl Eq for Type {}
impl Hash for Type {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.body.hash(state)
    }
}
impl Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}", &self.body)?;
        if let Some(loc) = &self.loc {
            write!(f, " {loc}")?;
        }
        write!(f, ")")?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct InferredType {
    /// Fields used in the constructors
    pub members: utils::SortedMap<String, Type>,
    /// Fields or applied methods
    pub properties: utils::SortedMap<String, Type>,
}

impl InferredType {
    pub fn new(
        members: impl IntoIterator<Item = (String, Type)>,
        props: impl IntoIterator<Item = (String, Type)>,
    ) -> Self {
        Self {
            members: members.into_iter().collect(),
            properties: props.into_iter().collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, new)]
pub struct StringType {
    pub len: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, new)]
pub struct ArrayType {
    pub item: Box<Type>,
    pub len: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Display, new)]
#[display("func({}): {ret}", params.iter().join(", "))]
pub struct FuncType {
    pub params: Vec<Type>,
    pub ret: Type,
}
impl FuncType {
    pub fn intersection(&self, other: &FuncType, modules: &[Module]) -> Option<FuncType> {
        if self.params.len() != other.params.len() {
            return None;
        }
        let params = izip!(&self.params, &other.params)
            .map(|(a_param, b_param)| a_param.union(b_param, modules))
            .collect::<Option<_>>()?;
        Some(FuncType::new(
            params,
            self.ret.intersection(&other.ret, modules)?,
        ))
    }
    pub fn union(&self, other: &FuncType, modules: &[Module]) -> Option<FuncType> {
        if self.params.len() != other.params.len() {
            return None;
        }
        let params = izip!(&self.params, &other.params)
            .map(|(a_param, b_param)| a_param.intersection(b_param, modules))
            .collect::<Option<_>>()?;
        Some(FuncType::new(params, self.ret.union(&other.ret, modules)?))
    }
}
