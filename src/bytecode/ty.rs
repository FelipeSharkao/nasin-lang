use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;

use derive_ctor::ctor;
use derive_more::{Display, From};
use derive_setters::Setters;
use genawaiter::rc::Gen;
use itertools::{Itertools, chain, izip};

use super::{Loc, Module, TypeDef, TypeDefBody, TypeVarIdx};
use crate::utils::{self, SortedMap, unordered};

#[derive(Debug, Clone, PartialEq, Eq, Hash, From)]
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
    String,
    #[from(skip)]
    Array(Box<Type>),
    #[from(skip)]
    Ptr(Option<Box<Type>>),
    Func(Box<FuncType>),
    TypeRef(TypeRef),
    TypeVar(TypeVar),
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
            TypeBody::AnyOpaque => write!(f, "AnyOpaque")?,
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
            TypeBody::String => write!(f, "str")?,
            TypeBody::Inferred(v) => {
                write!(f, "{{")?;
                for (name, t) in &v.members {
                    write!(f, " {name}: {t}")?;
                }
                for (name, t) in &v.properties {
                    write!(f, " .{name}: {t}")?;
                }
                write!(f, " }}")?;
            }
            TypeBody::Array(v) => write!(f, "[{v}]")?,
            TypeBody::Ptr(ty) => {
                write!(f, "Ptr")?;
                if let Some(ty) = ty {
                    write!(f, "({ty})")?;
                }
            }
            TypeBody::Func(func) => write!(
                f,
                "Func({}): {}",
                utils::join(", ", &func.params),
                &func.ret
            )?,
            TypeBody::TypeRef(ty_ref) => write!(
                f,
                "{} {}-{}",
                if ty_ref.is_self { "self" } else { "type" },
                ty_ref.mod_idx,
                ty_ref.idx,
            )?,
            TypeBody::TypeVar(tv) => {
                write!(f, "typevar {}-{}", tv.mod_idx, tv.typevar_idx)?
            }
        }
        Ok(())
    }
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

    pub fn is_not_final(&self) -> bool {
        if matches!(
            self,
            TypeBody::AnyNumber
                | TypeBody::AnySignedNumber
                | TypeBody::AnyFloat
                | TypeBody::Inferred(_)
        ) {
            return true;
        }

        if let TypeBody::Func(func) = self {
            return func.params.iter().any(|ty| ty.body.is_not_final())
                || func.ret.body.is_not_final();
        }

        if let TypeBody::Array(ty) = self {
            return ty.body.is_not_final();
        }

        if let TypeBody::Ptr(Some(ty)) = self {
            return ty.body.is_not_final();
        }

        return false;
    }
}

#[derive(Debug, Display, Clone, ctor)]
#[display("{body}")]
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
            TypeBody::String | TypeBody::Array(_) => true,
            TypeBody::TypeRef(t) => match &modules[t.mod_idx].typedefs[t.idx].body {
                TypeDefBody::Record(_) | TypeDefBody::Interface(_) => true,
            },
            _ => false,
        }
    }

    pub fn is_primitive(&self) -> bool {
        self.is_bool() || self.is_number() || matches!(&self.body, TypeBody::Ptr(_))
    }

    pub fn is_ptr(&self) -> bool {
        matches!(&self.body, TypeBody::Ptr(_))
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

    pub fn field<'a>(
        &'a self,
        name: &str,
        modules: &'a [Module],
    ) -> Option<Cow<'a, Type>> {
        match &self.body {
            TypeBody::Inferred(v) => v.members.get(name).map(|ty| Cow::Borrowed(ty)),
            TypeBody::TypeRef(type_ref) => type_ref.field(name, modules),
            _ => None,
        }
    }

    pub fn method<'a>(
        &'a self,
        name: &str,
        modules: &'a [Module],
    ) -> Option<Cow<'a, Type>> {
        match &self.body {
            TypeBody::TypeRef(type_ref) => type_ref.method(name, modules),
            _ => None,
        }
    }

    pub fn property<'a>(
        &'a self,
        name: &str,
        modules: &'a [Module],
    ) -> Option<Cow<'a, Type>> {
        match &self.body {
            TypeBody::Inferred(v) => v.properties.get(name).map(|v| Cow::Borrowed(v)),
            TypeBody::TypeRef(type_ref) => type_ref.property(name, modules),
            _ => None,
        }
    }

    pub fn merge(
        &self,
        other: &Type,
        variance: Variance,
        modules: &[Module],
    ) -> Option<Type> {
        let body = match (self, other) {
            (body!(a), body!(b)) if a == b => a.clone(),
            // unordered!(body!(TypeBody::Never), body!(a)) => match variance {
            //     Variance::Covariant => TypeBody::Never,
            //     Variance::Contravariant => a.clone(),
            // },
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
            number!(AnySignedNumber) => TypeBody::AnySignedNumber,
            number!(AnyFloat, AnySignedNumber) => TypeBody::AnyFloat,
            (body!(TypeBody::Array(a)), body!(TypeBody::Array(b))) => {
                TypeBody::Array(a.merge(&b, variance, modules)?.into())
            }
            (body!(TypeBody::Ptr(a)), body!(TypeBody::Ptr(b))) => {
                let ty = match (a, b) {
                    (Some(a), Some(b)) => Some(a.merge(b, variance, modules)?.into()),
                    (None, None) => None,
                    unordered!(Some(a), None) => match variance {
                        Variance::Covariant => Some(a.clone()),
                        Variance::Contravariant => None,
                    },
                };
                TypeBody::Ptr(ty)
            }
            (body!(TypeBody::Func(a)), body!(TypeBody::Func(b))) => {
                TypeBody::Func(a.merge(b, variance, modules)?.into())
            }
            (body!(TypeBody::Inferred(a)), body!(TypeBody::Inferred(b))) => {
                let mut members = SortedMap::new();
                for name in chain!(a.members.keys(), b.members.keys()).unique() {
                    let ty = match (a.members.get(name), b.members.get(name)) {
                        (Some(a_member), Some(b_member)) => {
                            a_member.merge(b_member, variance, modules)?
                        }
                        // TODO: optional fields
                        unordered!(Some(_), None) => return None,
                        _ => unreachable!(),
                    };
                    members.insert(name.to_string(), ty);
                }

                let mut properties = SortedMap::new();
                for name in chain!(a.properties.keys(), b.properties.keys()).unique() {
                    let ty = match (a.properties.get(name), b.properties.get(name)) {
                        (Some(a_prop), Some(b_prop)) => {
                            a_prop.merge(b_prop, variance, modules)?
                        }
                        unordered!(Some(prop), None) => match variance {
                            Variance::Covariant => prop.clone(),
                            Variance::Contravariant => continue,
                        },
                        _ => unreachable!(),
                    };
                    properties.insert(name.to_string(), ty);
                }

                TypeBody::Inferred(InferredType {
                    members,
                    properties,
                })
            }
            unordered!(body!(TypeBody::Inferred(a)), b) => {
                let has_all_members = a.members.iter().all(|(name, a_ty)| {
                    other.field(name, modules).is_some_and(|b_ty| {
                        a_ty.merge(b_ty.as_ref(), variance, modules).is_some()
                    })
                });
                let has_all_props = a.properties.iter().all(|(name, a_ty)| {
                    other.property(name, modules).is_some_and(|b_ty| {
                        a_ty.merge(&b_ty, variance, modules).is_some()
                    })
                });
                if !has_all_members || !has_all_props {
                    return None;
                }

                if let TypeBody::TypeRef(type_ref) = &b.body
                    && !type_ref.args.is_empty()
                {
                    let generics =
                        &modules[type_ref.mod_idx].typedefs[type_ref.idx].generics;
                    let mut substitutions = HashMap::new();
                    if !type_ref.to_inferred(modules).collect_typevar_substitutions(
                        a,
                        variance,
                        &mut substitutions,
                        modules,
                    ) {
                        return None;
                    }
                    let args = izip!(generics, &type_ref.args)
                        .map(|(typevar, arg)| substitutions.get(typevar).unwrap_or(arg))
                        .cloned()
                        .collect_vec();
                    TypeRef {
                        args,
                        ..type_ref.clone()
                    }
                    .into()
                } else {
                    b.body.clone()
                }
            }
            (body!(TypeBody::TypeRef(a)), body!(TypeBody::TypeRef(b))) => {
                a.merge(b, variance, modules)?.into()
            }
            // TODO: when we add constraints to generics, we will have to intersect with
            // that. Since we don't have that yet, all typevars are blanket, they don't
            // change the type at all
            unordered!(body!(TypeBody::TypeVar(_)), body!(a)) => a.clone(),
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

    pub fn substitute_typevar<'m>(
        &self,
        substitutions: &'m HashMap<TypeVarIdx, Type>,
    ) -> Option<Type> {
        struct Substitutions<'a>(&'a HashMap<TypeVarIdx, Type>);
        impl<'a> Substitutions<'a> {
            fn substitute(&self, ty: &Type) -> Option<Type> {
                ty.substitute_typevar(self.0)
            }

            fn substitute_many<'s>(
                &'s self,
                iter: impl IntoIterator<Item = &'s Type>,
            ) -> Vec<Option<Type>>
            where
                'a: 's,
            {
                iter.into_iter().map(|ty| self.substitute(ty)).collect_vec()
            }

            fn mix(&self, old: &Type, new: Option<Type>) -> Type {
                new.unwrap_or_else(|| old.clone())
            }

            fn mix_many<'s>(
                &'s self,
                old: impl IntoIterator<Item = &'s Type>,
                new: impl IntoIterator<Item = Option<Type>>,
            ) -> impl Iterator<Item = Type>
            where
                'a: 's,
            {
                izip!(old, new).map(|(old_ty, new_ty)| self.mix(old_ty, new_ty))
            }
        }

        let subs = Substitutions(substitutions);

        macro_rules! validate {
            ($($iter:expr),* $(,)?) => {
                if chain!($($iter),*).all(|ty| ty.is_none()) {
                    return None;
                }
            };
        }

        let body = match &self.body {
            TypeBody::TypeVar(typevar) => {
                return substitutions.get(&typevar.typevar_idx).cloned();
            }
            TypeBody::TypeRef(type_ref) => {
                let args = subs.substitute_many(&type_ref.args);
                validate!(&args);
                TypeBody::TypeRef(TypeRef {
                    args: subs.mix_many(&type_ref.args, args).collect(),
                    ..type_ref.clone()
                })
            }
            TypeBody::Inferred(inferred) => {
                let members = subs.substitute_many(inferred.members.values());
                let properties = subs.substitute_many(inferred.properties.values());
                validate!(&members, &properties);
                TypeBody::Inferred(InferredType {
                    members: izip!(
                        inferred.members.keys().cloned(),
                        subs.mix_many(inferred.members.values(), members)
                    )
                    .collect(),
                    properties: izip!(
                        inferred.properties.keys().cloned(),
                        subs.mix_many(inferred.properties.values(), properties)
                    )
                    .collect(),
                    ..inferred.clone()
                })
            }
            TypeBody::Func(func_ty) => {
                let params = subs.substitute_many(&func_ty.params);
                let ret = subs.substitute(&func_ty.ret);
                validate!(&params, Some(&ret));
                TypeBody::Func(Box::new(FuncType::new(
                    subs.mix_many(&func_ty.params, params).collect(),
                    subs.mix(&func_ty.ret, ret),
                )))
            }
            TypeBody::Array(elem_ty) => {
                let new_elem_ty = subs.substitute(elem_ty.as_ref());
                validate!(Some(&new_elem_ty));
                TypeBody::Array(Box::new(subs.mix(elem_ty, new_elem_ty)))
            }
            TypeBody::Ptr(Some(elem_ty)) => {
                let new_elem_ty = subs.substitute(elem_ty.as_ref());
                validate!(Some(&new_elem_ty));
                TypeBody::Ptr(Some(Box::new(subs.mix(elem_ty, new_elem_ty))))
            }
            _ => return None,
        };

        Some(Type::new(body, self.loc))
    }

    pub fn typevars(&self) -> impl Iterator<Item = TypeVarIdx> {
        Gen::new(async move |co| match &self.body {
            TypeBody::TypeVar(typevar) => co.yield_(typevar.typevar_idx).await,
            TypeBody::TypeRef(type_ref) => {
                for arg_ty in &type_ref.args {
                    for typevar in arg_ty.typevars() {
                        co.yield_(typevar).await;
                    }
                }
            }
            TypeBody::Inferred(inferred) => {
                for ty in inferred.members.values() {
                    for typevar in ty.typevars() {
                        co.yield_(typevar).await;
                    }
                }
                for ty in inferred.properties.values() {
                    for typevar in ty.typevars() {
                        co.yield_(typevar).await;
                    }
                }
            }
            TypeBody::Func(func_ty) => {
                for param in &func_ty.params {
                    for typevar in param.typevars() {
                        co.yield_(typevar).await;
                    }
                }
                for typevar in func_ty.ret.typevars() {
                    co.yield_(typevar).await;
                }
            }
            TypeBody::Array(elem_ty) => {
                for typevar in elem_ty.typevars() {
                    co.yield_(typevar).await;
                }
            }
            TypeBody::Ptr(Some(elem_ty)) => {
                for typevar in elem_ty.typevars() {
                    co.yield_(typevar).await;
                }
            }
            _ => {}
        })
        .into_iter()
    }

    /// Compares the type `self` with `other` and updates a map of typevars that exist in
    /// `self` and the type they are mapped to in `other`. Returns false if `self` and
    /// `other` are incompatible.
    pub fn collect_typevar_substitutions(
        &self,
        other: &Type,
        variance: Variance,
        substitutions: &mut HashMap<TypeVarIdx, Type>,
        modules: &[Module],
    ) -> bool {
        macro_rules! rec_or_return {
            ($a:expr, $b:expr) => {
                if !($a).collect_typevar_substitutions(
                    $b,
                    variance,
                    substitutions,
                    modules,
                ) {
                    return false;
                }
            };
        }

        match (self, other) {
            (body!(TypeBody::TypeVar(typevar)), ty) => {
                let ty = if let Some(existing) = substitutions.get(&typevar.typevar_idx) {
                    let Some(merged) = existing.merge(ty, variance, modules) else {
                        return false;
                    };
                    merged
                } else {
                    ty.clone()
                };
                substitutions.insert(typevar.typevar_idx, ty);
            }
            (body!(TypeBody::TypeRef(a)), body!(TypeBody::TypeRef(b)))
                if a.is_same_of(b) =>
            {
                if a.args.len() != b.args.len() {
                    return false;
                }
                for (a_arg, b_arg) in izip!(&a.args, &b.args) {
                    rec_or_return!(a_arg, b_arg);
                }
            }
            (body!(TypeBody::TypeRef(a)), body!(TypeBody::Inferred(b))) => {
                rec_or_return!(&a.to_inferred(modules), b);
            }
            (body!(TypeBody::Inferred(a)), body!(TypeBody::TypeRef(b))) => {
                rec_or_return!(a, &b.to_inferred(modules));
            }
            (body!(TypeBody::Func(a)), body!(TypeBody::Func(b))) => {
                if a.params.len() != b.params.len() {
                    return false;
                }
                for (a_param, b_param) in izip!(&a.params, &b.params) {
                    rec_or_return!(a_param, b_param);
                }
                rec_or_return!(&a.ret, &b.ret);
            }
            (body!(TypeBody::Array(a)), body!(TypeBody::Array(b)))
            | (body!(TypeBody::Ptr(Some(a))), body!(TypeBody::Ptr(Some(b)))) => {
                rec_or_return!(a, b);
            }
            _ if self.merge(other, variance, modules).is_none() => return false,
            _ => {}
        }
        true
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variance {
    Covariant,
    Contravariant,
}

impl Variance {
    pub fn invert(self) -> Self {
        match self {
            Variance::Covariant => Variance::Contravariant,
            Variance::Contravariant => Variance::Covariant,
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

#[derive(Debug, Clone, PartialEq, Eq, Hash, Setters, ctor)]
#[setters(into, prefix = "with_")]
pub struct TypeRef {
    pub mod_idx: usize,
    pub idx:     usize,
    #[ctor(expr(false))]
    pub is_self: bool,
    #[ctor(default)]
    pub args:    Vec<Type>,
}

impl TypeRef {
    pub fn is_same_of(&self, other: &TypeRef) -> bool {
        (self.mod_idx, self.idx) == (other.mod_idx, other.idx)
    }

    pub fn merge(
        &self,
        other: &Self,
        variance: Variance,
        modules: &[Module],
    ) -> Option<Self> {
        let self_def = &modules[self.mod_idx].typedefs[self.idx];
        let other_def = &modules[other.mod_idx].typedefs[other.idx];

        macro_rules! def_body {
            ($pat:pat) => {
                TypeDef { body: $pat, .. }
            };
        }

        let (mod_idx, ty_idx) = match (
            (self.mod_idx, self.idx, self_def),
            (other.mod_idx, other.idx, other_def),
        ) {
            (
                (_, _, def_body!(TypeDefBody::Record(_))),
                (_, _, def_body!(TypeDefBody::Record(_))),
            )
            | (
                (_, _, def_body!(TypeDefBody::Interface(_))),
                (_, _, def_body!(TypeDefBody::Interface(_))),
            ) if self.is_same_of(other) => (self.mod_idx, other.idx),
            unordered!(
                (r_mod_idx, r_ty_idx, def_body!(TypeDefBody::Record(rec))),
                (i_mod_idx, i_ty_idx, def_body!(TypeDefBody::Interface(..))),
            ) => {
                let extends = rec.ifaces.iter().any(|(mod_idx, ty_idx)| {
                    i_mod_idx == *mod_idx && i_ty_idx == *ty_idx
                });
                if !extends {
                    return None;
                }

                if !self.args.is_empty() && !other.args.is_empty() {
                    todo!("merge of interface and record with generics");
                }

                match variance {
                    Variance::Covariant => (r_mod_idx, r_ty_idx),
                    Variance::Contravariant => (i_mod_idx, i_ty_idx),
                }
            }
            _ => return None,
        };

        let args = izip!(&self.args, &other.args)
            .map(|(self_arg, other_arg)| self_arg.merge(other_arg, variance, modules))
            .collect::<Option<Vec<_>>>()?;

        Some(
            TypeRef::new(mod_idx, ty_idx)
                .with_is_self(other.is_self || self.is_self)
                .with_args(args),
        )
    }

    pub fn field<'a>(
        &'a self,
        name: &str,
        modules: &'a [Module],
    ) -> Option<Cow<'a, Type>> {
        match &modules[self.mod_idx].typedefs[self.idx].body {
            TypeDefBody::Record(rec) => {
                let ty = &rec.fields.get(name)?.ty;
                let substitutions = self.typevar_substitutions(modules);
                if let Some(ty) = ty.substitute_typevar(&substitutions) {
                    Some(Cow::Owned(ty))
                } else {
                    Some(Cow::Borrowed(ty))
                }
            }
            TypeDefBody::Interface(_) => None,
        }
    }

    pub fn method<'a>(
        &'a self,
        name: &str,
        modules: &'a [Module],
    ) -> Option<Cow<'a, Type>> {
        let typedef = modules.get(self.mod_idx)?.typedefs.get(self.idx)?;
        let method = match &typedef.body {
            TypeDefBody::Record(rec) => rec.methods.get(name),
            TypeDefBody::Interface(iface) => iface.methods.get(name),
        }?;
        let method_mod = modules.get(method.func_ref.0)?;
        let func = &method_mod.funcs[method.func_ref.1];

        let substitutions = self.typevar_substitutions(modules);

        let params_tys = func
            .params
            .iter()
            .map(|param| {
                let ty = &method_mod.values[*param].ty;
                ty.substitute_typevar(&substitutions)
                    .unwrap_or_else(|| ty.clone())
            })
            .collect_vec();
        let ret_ty = &method_mod.values[func.ret].ty;
        let ret_ty = ret_ty
            .substitute_typevar(&substitutions)
            .unwrap_or_else(|| ret_ty.clone());

        Some(Cow::Owned(Type::new(
            TypeBody::Func(Box::new(FuncType::new(params_tys, ret_ty))),
            Some(method.loc),
        )))
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

            let [obj_param, params @ ..] = &func.params[..] else {
                return None;
            };

            // is static?
            let TypeBody::TypeRef(obj_type_ref) = &obj_param.body else {
                return None;
            };
            if obj_type_ref
                .merge(&self, Variance::Covariant, modules)
                .is_none()
            {
                return None;
            }

            // functions without parameters are just values
            if params.len() == 0 {
                return Some(Cow::Owned(func.ret.clone()));
            }

            Some(Cow::Owned(Type::new(
                TypeBody::Func(Box::new(FuncType::new(
                    params.to_vec(),
                    func.ret.clone(),
                ))),
                ty.loc,
            )))
        } else if let Some(ty) = self.field(name, modules) {
            Some(ty)
        } else {
            None
        }
    }

    pub fn typevar_substitutions(&self, modules: &[Module]) -> HashMap<TypeVarIdx, Type> {
        let def = &modules[self.mod_idx].typedefs[self.idx];
        izip!(&def.generics, &self.args)
            .map(|(&typevar, arg)| (typevar, arg.clone()))
            .collect()
    }

    pub fn to_inferred(&self, modules: &[Module]) -> InferredType {
        let def = &modules[self.mod_idx].typedefs[self.idx];

        let (fields, methods) = match &def.body {
            TypeDefBody::Record(rec) => (&rec.fields, &rec.methods),
            TypeDefBody::Interface(iface) => (&SortedMap::new(), &iface.methods),
        };

        let mut members = utils::SortedMap::new();
        let mut properties = utils::SortedMap::new();

        for name in fields.keys() {
            let Some(ty) = self.field(name, modules) else {
                continue;
            };
            members.insert(name.to_string(), ty.clone().into_owned());
        }

        for name in chain!(fields.keys(), methods.keys()).unique() {
            let Some(ty) = self.property(name, modules) else {
                continue;
            };
            properties.insert(name.to_string(), ty.into_owned());
        }

        InferredType::new(members, properties)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, ctor)]
pub struct TypeVar {
    pub mod_idx:     usize,
    pub typevar_idx: usize,
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

    /// Compares the type `self` with `other` and updates a map of typevars that exist in
    /// `self` and the type they are mapped to in `other`. Returns false if `self` and
    /// `other` are incompatible.
    pub fn collect_typevar_substitutions(
        &self,
        other: &Self,
        variance: Variance,
        substitutions: &mut HashMap<TypeVarIdx, Type>,
        modules: &[Module],
    ) -> bool {
        for (name, ty) in chain!(&self.members, &self.properties) {
            if let Some(other_ty) = other
                .members
                .get(name)
                .or_else(|| other.properties.get(name))
            {
                if !ty.collect_typevar_substitutions(
                    other_ty,
                    variance,
                    substitutions,
                    modules,
                ) {
                    return false;
                }
            }
        }

        true
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Display, ctor)]
#[display("func({}): {ret}", utils::join(", ", params))]
pub struct FuncType {
    pub params: Vec<Type>,
    pub ret:    Type,
}
impl FuncType {
    pub fn merge(
        &self,
        other: &FuncType,
        var: Variance,
        modules: &[Module],
    ) -> Option<FuncType> {
        if self.params.len() != other.params.len() {
            return None;
        }
        let params = izip!(&self.params, &other.params)
            .map(|(a_param, b_param)| a_param.merge(b_param, var.invert(), modules))
            .collect::<Option<_>>()?;
        Some(FuncType::new(
            params,
            self.ret.merge(&other.ret, var, modules)?,
        ))
    }
}
