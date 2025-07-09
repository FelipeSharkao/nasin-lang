use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;

use derive_more::{Display, From};
use derive_new::new;
use derive_setters::Setters;
use itertools::{chain, izip, Itertools};

use super::{Loc, Module, TypeDefBody};
use crate::bytecode::InterfaceImpl;
use crate::utils::{self, unordered};

#[derive(Debug, Clone, PartialEq, Eq, Hash, From, Display)]
pub enum TypeBody {
    #[display("void")]
    Void,
    #[display("never")]
    Never,
    #[display("bool")]
    Bool,
    #[display("AnyNumber")]
    AnyOpaque,
    // FIXME: use interface/trait for this
    #[display("AnySignedNumber")]
    AnyNumber,
    // FIXME: use interface/trait for this
    #[display("AnyFloat")]
    AnySignedNumber,
    // FIXME: use interface/trait for this
    #[display("anyopaque")]
    AnyFloat,
    #[display("i8")]
    I8,
    #[display("i16")]
    I16,
    #[display("i32")]
    I32,
    #[display("i64")]
    I64,
    #[display("u8")]
    U8,
    #[display("u16")]
    U16,
    #[display("u32")]
    U32,
    #[display("u64")]
    U64,
    #[display("usize")]
    USize,
    #[display("f32")]
    F32,
    #[display("f64")]
    F64,
    #[display("{_0}")]
    Inferred(InferredType),
    #[display("{_0}")]
    String(StringType),
    #[display("{_0}")]
    Array(ArrayType),
    #[from(skip)]
    #[display("ptr {_0}")]
    Ptr(Box<Type>),
    #[display("{_0}")]
    Func(Box<FuncType>),
    #[display("{_0}")]
    TypeRef(TypeRef),
    #[display("{_0}")]
    Generic(GenericRef),
    #[display("{_0}")]
    GenericInstance(GenericInstanceType),
}
impl TypeBody {
    pub fn unknown() -> Self {
        TypeBody::Inferred(InferredType {
            members:    utils::SortedMap::new(),
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
    pub loc:  Option<Loc>,
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
        Type {
            body: $pat,
            loc:  _,
        }
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
            TypeBody::TypeRef(t) => match &modules[t.mod_idx].typedefs[t.idx].body {
                TypeDefBody::Record(_) | TypeDefBody::Interface(_) => true,
            },
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

    pub fn field_type<'a>(
        &'a self,
        name: &str,
        modules: &'a [Module],
    ) -> Option<Cow<'a, Type>> {
        match &self.body {
            TypeBody::Inferred(v) => Some(Cow::Borrowed(v.members.get(name)?)),
            TypeBody::TypeRef(t) => {
                let typedef = &modules.get(t.mod_idx)?.typedefs.get(t.idx)?;
                match &typedef.body {
                    TypeDefBody::Record(rec) => {
                        Some(Cow::Borrowed(&rec.fields.get(name)?.ty))
                    }
                    TypeDefBody::Interface(_) => None,
                }
            }
            TypeBody::GenericInstance(gen_ins) => {
                let mut field = gen_ins.ty.field_type(name, modules);
                if let Some(field) = &mut field {
                    let gens = gen_ins.ty.generics(modules);
                    let subs: HashMap<_, _> = izip!(gens, &gen_ins.args).collect();
                    field.to_mut().apply_generics(&subs)
                }
                field
            }
            _ => None,
        }
    }

    pub fn method_type<'a>(
        &'a self,
        name: &str,
        modules: &'a [Module],
    ) -> Option<Cow<'a, Type>> {
        match &self.body {
            TypeBody::TypeRef(t) => {
                let typedef = modules.get(t.mod_idx)?.typedefs.get(t.idx)?;

                let method = match &typedef.body {
                    TypeDefBody::Record(rec) => rec.methods.get(name),
                    TypeDefBody::Interface(iface) => iface.methods.get(name),
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
            TypeBody::GenericInstance(gen_ins) => {
                let mut method = gen_ins.ty.method_type(name, modules);
                if let Some(method) = &mut method {
                    let gens = gen_ins.ty.generics(modules);
                    let subs: HashMap<_, _> = izip!(gens, &gen_ins.args).collect();
                    method.to_mut().apply_generics(&subs)
                }
                method
            }
            _ => None,
        }
    }

    pub fn property_type<'a>(
        &'a self,
        name: &str,
        modules: &'a [Module],
    ) -> Option<Cow<'a, Type>> {
        if let Some(ty) = self.method_type(name, modules) {
            let TypeBody::Func(func) = &ty.body else {
                return None;
            };
            let [params @ .., self_param] = &func.params[..] else {
                return None;
            };
            // is static?
            if self.intersection(self_param, modules).is_none() {
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
        if let Some(ty) = self.field_type(name, modules) {
            return Some(ty);
        }
        None
    }

    pub fn intersection(&self, other: &Type, modules: &[Module]) -> Option<Type> {
        let body = match (self, other) {
            (body!(a), body!(b)) if a == b => a.clone(),
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
                    members:    fields,
                    properties: methods,
                })
            }
            unordered!(body!(TypeBody::Inferred(a)), b) => {
                let has_all_members = a.members.iter().all(|(name, a_ty)| {
                    other.field_type(name, modules).is_some_and(|b_ty| {
                        a_ty.intersection(b_ty.as_ref(), modules).is_some()
                    })
                });
                let has_all_props = a.properties.iter().all(|(name, a_ty)| {
                    other
                        .property_type(name, modules)
                        .is_some_and(|b_ty| a_ty.intersection(&b_ty, modules).is_some())
                });
                if has_all_members && has_all_props {
                    b.body.clone()
                } else {
                    return None;
                }
            }
            (body!(TypeBody::TypeRef(a)), body!(TypeBody::TypeRef(b))) => {
                let (mod_idx, ty_idx) = if a.is_same_of(&TypeBody::TypeRef(*b), modules) {
                    (a.mod_idx, a.idx)
                } else if a.extends(&TypeBody::TypeRef(*b), modules) {
                    (a.mod_idx, a.idx)
                } else if b.extends(&TypeBody::TypeRef(*b), modules) {
                    (b.mod_idx, b.idx)
                } else {
                    return None;
                };

                TypeRef::new(mod_idx, ty_idx)
                    .with_is_self(a.is_self || b.is_self)
                    .into()
            }
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
                    len:  if a.len == b.len { a.len.clone() } else { None },
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
                    members:    fields,
                    properties: props,
                })
            }
            unordered!(body!(TypeBody::Inferred(a)), b) => {
                let has_all_members = a.members.iter().all(|(name, a_ty)| {
                    other
                        .field_type(name, modules)
                        .is_some_and(|b_ty| a_ty.union(b_ty.as_ref(), modules).is_some())
                });
                let has_all_props = a.properties.iter().all(|(name, a_ty)| {
                    other
                        .property_type(name, modules)
                        .is_some_and(|b_ty| a_ty.union(&b_ty, modules).is_some())
                });
                if has_all_members && has_all_props {
                    b.body.clone()
                } else {
                    return None;
                }
            }
            unordered!(body!(TypeBody::TypeRef(a)), body!(TypeBody::TypeRef(b)))
                if a.is_same_of(&TypeBody::TypeRef(*b), modules) =>
            {
                TypeRef::new(a.mod_idx, a.idx)
                    .with_is_self(a.is_self && b.is_self)
                    .into()
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

    pub fn generics(&self, modules: &[Module]) -> Vec<GenericRef> {
        match &self.body {
            TypeBody::TypeRef(t) => modules[t.mod_idx].typedefs[t.idx]
                .generics
                .iter()
                .map(|i| GenericRef::new(t.mod_idx, *i))
                .collect_vec(),
            _ => vec![],
        }
    }

    pub fn apply_generics<'a>(&mut self, subs: &HashMap<GenericRef, &'a Type>) {
        match &mut self.body {
            TypeBody::Generic(gen_ref) => {
                if let Some(ty) = subs.get(gen_ref) {
                    *self = (*ty).clone();
                }
            }
            TypeBody::Inferred(t) => {
                for prop in t.properties.values_mut() {
                    prop.apply_generics(subs);
                }
                for member in t.members.values_mut() {
                    member.apply_generics(subs);
                }
            }
            TypeBody::Array(t) => {
                t.item.apply_generics(subs);
            }
            TypeBody::Ptr(_) => todo!(),
            TypeBody::Func(t) => {
                for param in &mut t.params {
                    param.apply_generics(subs);
                }
                t.ret.apply_generics(subs);
            }
            TypeBody::GenericInstance(t) => {
                t.ty.apply_generics(subs);
            }
            TypeBody::Void
            | TypeBody::Never
            | TypeBody::Bool
            | TypeBody::AnyOpaque
            | TypeBody::AnyNumber
            | TypeBody::AnySignedNumber
            | TypeBody::AnyFloat
            | TypeBody::I8
            | TypeBody::I16
            | TypeBody::I32
            | TypeBody::I64
            | TypeBody::U8
            | TypeBody::U16
            | TypeBody::U32
            | TypeBody::U64
            | TypeBody::USize
            | TypeBody::F32
            | TypeBody::F64
            | TypeBody::String(..)
            | TypeBody::TypeRef(..) => {}
        }
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

trait TypeOperations {
    /// Returns true if the type is compatible with `other`, not necessary the exact same
    /// type. If `self` is a supertype or a subtype of `other`, it will return false.
    fn is_same_of(&self, other: &TypeBody, modules: &[Module]) -> bool;
    /// Returns true if `self` is a subtype of `other`.
    fn extends(&self, other: &TypeBody, modules: &[Module]) -> bool;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Setters, new)]
#[setters(into, prefix = "with_")]
pub struct TypeRef {
    pub mod_idx: usize,
    pub idx:     usize,
    #[new(value = "false")]
    pub is_self: bool,
}
impl Display for TypeRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} {}-{}",
            if self.is_self { "self" } else { "type" },
            self.mod_idx,
            self.idx,
        )
    }
}
impl TypeOperations for TypeRef {
    fn is_same_of(&self, other: &TypeBody, _modules: &[Module]) -> bool {
        match other {
            TypeBody::TypeRef(other) => {
                (self.mod_idx, self.idx) == (other.mod_idx, other.idx)
            }
            _ => false,
        }
    }

    fn extends(&self, other: &TypeBody, modules: &[Module]) -> bool {
        match other {
            TypeBody::TypeRef(other) => {
                let self_ty = &modules[self.mod_idx].typedefs[self.idx];
                let other_ty = &modules[other.mod_idx].typedefs[other.idx];

                match (&self_ty.body, &other_ty.body) {
                    (TypeDefBody::Record(rec), TypeDefBody::Interface(..)) => {
                        rec.ifaces.iter().any(|iface| match iface {
                            InterfaceImpl::TypeRef(ity)
                            | InterfaceImpl::GenericInstance(ity, _) => {
                                (other.mod_idx, other.idx) == (ity.mod_idx, ity.idx)
                            }
                        })
                    }
                    _ => false,
                }
            }
            _ => false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Display, new)]
#[display("{ty} of {}", args.iter().join(", "))]
pub struct GenericInstanceType {
    pub ty:   Box<Type>,
    pub args: Vec<Type>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Display, new)]
#[display("generic {mod_idx}-{idx}")]
pub struct GenericRef {
    pub mod_idx: usize,
    pub idx:     usize,
}
impl GenericRef {
    pub fn is_same_of(&self, other: &GenericRef) -> bool {
        (self.mod_idx, self.idx) == (other.mod_idx, other.idx)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct InferredType {
    /// Fields used in the constructors
    pub members:    utils::SortedMap<String, Type>,
    /// Fields or applied methods
    pub properties: utils::SortedMap<String, Type>,
}
impl InferredType {
    pub fn new(
        members: impl IntoIterator<Item = (String, Type)>,
        props: impl IntoIterator<Item = (String, Type)>,
    ) -> Self {
        Self {
            members:    members.into_iter().collect(),
            properties: props.into_iter().collect(),
        }
    }
}
impl Display for InferredType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "infered {{")?;
        for (name, t) in &self.members {
            write!(f, " {name}: {t}")?;
        }
        for (name, t) in &self.properties {
            write!(f, " .{name}: {t}")?;
        }
        write!(f, " }}")?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, new)]
pub struct StringType {
    pub len: Option<u32>,
}
impl Display for StringType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "string")?;
        if let Some(len) = self.len {
            write!(f, " {}", len)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, new)]
pub struct ArrayType {
    pub item: Box<Type>,
    pub len:  Option<u32>,
}
impl Display for ArrayType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "array {}", self.item)?;
        if let Some(len) = self.len {
            write!(f, " {}", len)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Display, new)]
#[display("func({}): {ret}", params.iter().join(", "))]
pub struct FuncType {
    pub params: Vec<Type>,
    pub ret:    Type,
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
