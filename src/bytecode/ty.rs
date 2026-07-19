use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

use derive_ctor::ctor;
use derive_more::From;
use derive_setters::Setters;
use duplicate::duplicate_item;
use genawaiter::rc::Gen;
use itertools::{Itertools, chain, izip};

use super::Printer;
use super::module::*;
use crate::config::BuildConfig;
use crate::utils::{self, SortedMap, matches_if, unordered};

#[derive(Debug, Clone, PartialEq, Eq, Hash, From, PartialOrd, Ord)]
pub enum TypeBody {
    Inferred(InferredType),
    Func(FuncType),
    TypeRef(TypeRef),
    TypeVar(TypeVar),
}

impl TypeBody {
    pub fn unknown() -> Self {
        TypeBody::Inferred(InferredType {
            members:    utils::SortedMap::new(),
            properties: utils::SortedMap::new(),
        })
    }

    pub fn builtin(builtin: BuiltinType, args: impl IntoIterator<Item = Type>) -> Self {
        TypeRef::builtin(builtin, args).into()
    }

    pub fn is_unknown(&self) -> bool {
        if let TypeBody::Inferred(i) = self {
            return i.members.is_empty() && i.properties.is_empty();
        }
        false
    }

    pub fn is_not_final(&self, modules: &[Module]) -> bool {
        match self {
            TypeBody::Func(func) => {
                func.params.iter().any(|ty| ty.body.is_not_final(modules))
                    || func.ret.body.is_not_final(modules)
            }
            TypeBody::TypeRef(type_ref) => {
                if type_ref.args.iter().any(|ty| ty.body.is_not_final(modules)) {
                    return true;
                }
                match &type_ref.get_typedef(modules).body {
                    TypeDefBody::Builtin(builtin) => builtin.is_not_final(),
                    TypeDefBody::Record(_) | TypeDefBody::Interface => false,
                }
            }
            TypeBody::Inferred(_) => true,
            _ => false,
        }
    }

    pub fn is_never(&self, modules: &[Module]) -> bool {
        match self {
            TypeBody::TypeRef(t) => match &t.get_typedef(modules).body {
                TypeDefBody::Builtin(builtin) => matches!(builtin, BuiltinType::Never),
                TypeDefBody::Record(_) | TypeDefBody::Interface => false,
            },
            _ => false,
        }
    }

    pub fn is_void(&self, modules: &[Module]) -> bool {
        let TypeBody::TypeRef(type_ref) = &self else {
            return false;
        };
        match &type_ref.get_typedef(modules).body {
            TypeDefBody::Builtin(builtin) => matches!(builtin, BuiltinType::Void),
            TypeDefBody::Record(_) | TypeDefBody::Interface => false,
        }
    }

    pub fn field<'a>(
        &'a self,
        name: &str,
        modules: &'a [Module],
    ) -> Option<Cow<'a, Type>> {
        match self {
            Self::Inferred(v) => v.members.get(name).map(|ty| Cow::Borrowed(ty)),
            Self::TypeRef(type_ref) => type_ref.field(name, modules),
            Self::TypeVar(type_var) => type_var.field(name, modules),
            _ => None,
        }
    }

    pub fn method<'a>(
        &'a self,
        name: &str,
        modules: &'a [Module],
    ) -> Option<Cow<'a, Type>> {
        match self {
            Self::TypeRef(type_ref) => type_ref.method(name, modules),
            Self::TypeVar(type_var) => {
                let res = type_var.method(name, modules);
                res
            }
            _ => None,
        }
    }

    pub fn property<'a>(
        &'a self,
        name: &str,
        modules: &'a [Module],
    ) -> Option<Cow<'a, Type>> {
        match self {
            Self::Inferred(v) => v.properties.get(name).map(|v| Cow::Borrowed(v)),
            _ => {
                if let Some(ty) = self.method(name, modules) {
                    let TypeBody::Func(func) = &ty.body else {
                        return None;
                    };

                    let [obj_param, params @ ..] = &func.params[..] else {
                        return None;
                    };

                    // is static?
                    if self
                        .merge(
                            &obj_param.body.clone().with_rigid(false),
                            Variance::Covariant,
                            modules,
                        )
                        .is_none()
                    {
                        return None;
                    }

                    // functions without parameters are just values
                    if params.len() == 0 {
                        return Some(Cow::Owned(func.ret.as_ref().clone()));
                    }

                    Some(Cow::Owned(Type::new(
                        FuncType::new(params.to_vec(), func.ret.clone()).into(),
                        ty.loc,
                    )))
                } else if let Some(ty) = self.field(name, modules) {
                    Some(ty)
                } else {
                    None
                }
            }
        }
    }

    pub fn merge(
        &self,
        other: &Self,
        variance: Variance,
        modules: &[Module],
    ) -> Option<Self> {
        match (self, other) {
            (a, b) if a == b => Some(a.clone()),
            // INFO: the more """correct"" would be that a merge with a never type should
            // check the variance, returning a never type if the variance is covariant.
            // That doesn't work with our current implementation of the typechecker tho
            unordered!(a, b) if a.is_never(modules) && b.is_unknown() => Some(a.clone()),
            unordered!(a, b) if b.is_never(modules) => Some(a.clone()),
            (Self::Func(a), Self::Func(b)) => {
                Some(Self::Func(a.merge(b, variance, modules)?.into()))
            }
            (Self::Inferred(a), Self::Inferred(b)) => {
                Some(Self::Inferred(a.merge(b, variance, modules)?.into()))
            }
            unordered!(Self::Inferred(a), b) => {
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

                if let Self::TypeRef(type_ref) = b
                    && !type_ref.args.is_empty()
                {
                    let generics = &type_ref.get_typedef(modules).generics;
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
                    Some(
                        TypeRef {
                            args,
                            ..type_ref.clone()
                        }
                        .into(),
                    )
                } else {
                    Some(b.clone())
                }
            }
            (Self::TypeRef(a), Self::TypeRef(b)) => {
                Some(a.merge(b, variance, modules)?.into())
            }
            unordered!(Self::TypeVar(a), Self::TypeVar(b)) if a.rigid => {
                if a.extends(b, modules) {
                    Some(self.clone())
                } else {
                    None
                }
            }
            (Self::TypeVar(a), Self::TypeVar(b)) => {
                let def_a = &modules[a.mod_idx].typevars[a.typevar_idx];
                let def_b = &modules[b.mod_idx].typevars[b.typevar_idx];
                match (&def_a.constraint, &def_b.constraint) {
                    (Some(cons_a), Some(cons_b)) => {
                        cons_a.body.merge(&cons_b.body, variance, modules)
                    }
                    _ => None,
                }
            }
            unordered!(Self::TypeVar(a), b) if a.rigid => {
                let def = &modules[a.mod_idx].typevars[a.typevar_idx];
                match &def.constraint {
                    Some(cons) if b.extends(&cons.body, modules) => Some(self.clone()),
                    _ => None,
                }
            }
            unordered!(Self::TypeVar(a), b) => {
                let def = &modules[a.mod_idx].typevars[a.typevar_idx];
                match &def.constraint {
                    Some(cons) => cons.body.merge(b, variance, modules),
                    None => Some(b.clone()),
                }
            }
            _ => None,
        }
    }

    pub fn extends(&self, other: &Self, modules: &[Module]) -> bool {
        match (self, other) {
            (a, b) if a == b || a.is_never(modules) => true,
            (_, b) if b.is_unknown() => true,
            (_, b) if b.is_never(modules) => false,
            (Self::Func(a), Self::Func(b)) => a.extends(b, false, modules),
            (Self::Inferred(a), Self::Inferred(b)) => a.extends(b, modules),
            (Self::Inferred(a), Self::TypeRef(type_ref)) => {
                if let TypeRefKey::Builtin(_) = &type_ref.key
                    && !a.members.is_empty()
                {
                    return false;
                }
                a.extends(&type_ref.to_inferred(modules), modules)
            }
            (Self::TypeRef(a), Self::TypeRef(b)) => a.implements(b, modules),
            (Self::TypeVar(a), Self::TypeVar(b)) => a.extends(b, modules),
            (Self::TypeVar(a), b) => {
                let def = &modules[a.mod_idx].typevars[a.typevar_idx];
                match &def.constraint {
                    Some(cons) => cons.body.extends(b, modules),
                    None => !a.rigid,
                }
            }
            (a, Self::TypeVar(b)) => {
                let def = &modules[b.mod_idx].typevars[b.typevar_idx];
                match &def.constraint {
                    Some(cons) => a.extends(&cons.body, modules),
                    None => !b.rigid,
                }
            }
            _ => false,
        }
    }

    pub fn formated(
        &self,
        modules: &[Module],
        cfg: &BuildConfig,
        base_module: Option<usize>,
    ) -> String {
        let mut s = String::new();
        let mut printer = Printer::new(modules, cfg).with_reconstruct(true);
        if let Some(base_module) = base_module {
            printer = printer.with_cur_mod_idx(base_module);
        }
        printer.write_type_expr(&mut s, self).unwrap();
        s
    }

    #[duplicate_item(
        typevars       values_ref   reference(e);
        [typevars]     [values]     [&e];
        [typevars_mut] [values_mut] [&mut e];
    )]
    pub fn typevars(
        self: reference([Self]),
    ) -> impl Iterator<Item = reference([TypeVar])> {
        Gen::new(async move |co| match self {
            TypeBody::TypeVar(typevar) => co.yield_(typevar).await,
            TypeBody::TypeRef(type_ref) => {
                for arg_ty in reference([type_ref.args]) {
                    for typevar in arg_ty.body.typevars() {
                        co.yield_(typevar).await;
                    }
                }
            }
            TypeBody::Inferred(inferred) => {
                for ty in inferred.members.values_ref() {
                    for typevar in ty.body.typevars() {
                        co.yield_(typevar).await;
                    }
                }
                for ty in inferred.properties.values_ref() {
                    for typevar in ty.body.typevars() {
                        co.yield_(typevar).await;
                    }
                }
            }
            TypeBody::Func(func_ty) => {
                for param in reference([func_ty.params]) {
                    for typevar in param.body.typevars() {
                        co.yield_(typevar).await;
                    }
                }
                for typevar in func_ty.ret.body.typevars() {
                    co.yield_(typevar).await;
                }
            }
        })
        .into_iter()
    }

    pub fn substitute_typevar<'m>(
        &self,
        substitutions: &'m HashMap<TypeVarIdx, Type>,
    ) -> Option<Self> {
        let subs = TypevarSubstitutions(substitutions);

        macro_rules! validate {
            ($($iter:expr),* $(,)?) => {
                if chain!($($iter),*).all(|ty| ty.is_none()) {
                    return None;
                }
            };
        }

        match &self {
            Self::TypeVar(typevar) => substitutions
                .get(&typevar.typevar_idx)
                .map(|t| t.body.clone()),
            Self::TypeRef(type_ref) => {
                let args = subs.substitute_many(&type_ref.args);
                validate!(&args);
                Some(Self::TypeRef(TypeRef {
                    args: subs.mix_many(&type_ref.args, args).collect(),
                    ..type_ref.clone()
                }))
            }
            Self::Inferred(inferred) => {
                let members = subs.substitute_many(inferred.members.values());
                let properties = subs.substitute_many(inferred.properties.values());
                validate!(&members, &properties);
                Some(Self::Inferred(InferredType {
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
                }))
            }
            Self::Func(func_ty) => {
                let params = subs.substitute_many(&func_ty.params);
                let ret = subs.substitute(&func_ty.ret);
                validate!(&params, Some(&ret));
                let body = FuncType::new(
                    subs.mix_many(&func_ty.params, params).collect(),
                    subs.mix(&func_ty.ret, ret).into(),
                )
                .into();
                Some(body)
            }
        }
    }

    pub fn has_typevars(&self) -> bool {
        self.typevars().next().is_some()
    }

    /// Returns a new type with all typevars marked as rigid or not. See [`TypeVar`].
    pub fn with_rigid(mut self, rigid: bool) -> Self {
        for typevar in &mut self.typevars_mut() {
            typevar.rigid = rigid;
        }
        self
    }
}

#[derive(Debug, Clone, PartialOrd, Ord, ctor)]
pub struct Type {
    pub body: TypeBody,
    pub loc:  Option<Loc>,
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

    pub fn builtin(
        builtin: BuiltinType,
        args: impl IntoIterator<Item = Type>,
        loc: Option<Loc>,
    ) -> Self {
        Type::new(TypeBody::builtin(builtin, args), loc)
    }

    pub fn is_unknown(&self) -> bool {
        self.body.is_unknown()
    }

    pub fn is_inferred(&self) -> bool {
        matches!(&self.body, TypeBody::Inferred(_))
    }

    pub fn is_aggregate(&self, modules: &[Module]) -> bool {
        match &self.body {
            TypeBody::TypeRef(t) => match &t.get_typedef(modules).body {
                TypeDefBody::Record(_) | TypeDefBody::Interface => true,
                TypeDefBody::Builtin(builtin) => builtin.is_aggregate(),
            },
            _ => false,
        }
    }

    pub fn is_primitive(&self, modules: &[Module]) -> bool {
        match &self.body {
            TypeBody::TypeRef(t) => match &t.get_typedef(modules).body {
                TypeDefBody::Builtin(builtin) => builtin.is_primitive(),
                TypeDefBody::Record(_) | TypeDefBody::Interface => false,
            },
            _ => false,
        }
    }

    pub fn is_ptr(&self, modules: &[Module]) -> bool {
        match &self.body {
            TypeBody::TypeRef(t) => match &t.get_typedef(modules).body {
                TypeDefBody::Builtin(builtin) => matches!(builtin, BuiltinType::Ptr),
                TypeDefBody::Record(_) | TypeDefBody::Interface => false,
            },
            _ => false,
        }
    }

    pub fn is_number(&self, modules: &[Module]) -> bool {
        match &self.body {
            TypeBody::TypeRef(t) => match &t.get_typedef(modules).body {
                TypeDefBody::Builtin(builtin) => builtin.is_number(),
                TypeDefBody::Record(_) | TypeDefBody::Interface => false,
            },
            _ => false,
        }
    }

    pub fn is_int(&self, modules: &[Module]) -> bool {
        match &self.body {
            TypeBody::TypeRef(t) => match &t.get_typedef(modules).body {
                TypeDefBody::Builtin(builtin) => builtin.is_int(),
                TypeDefBody::Record(_) | TypeDefBody::Interface => false,
            },
            _ => false,
        }
    }

    pub fn is_sint(&self, modules: &[Module]) -> bool {
        match &self.body {
            TypeBody::TypeRef(t) => match &t.get_typedef(modules).body {
                TypeDefBody::Builtin(builtin) => builtin.is_sint(),
                TypeDefBody::Record(_) | TypeDefBody::Interface => false,
            },
            _ => false,
        }
    }

    pub fn is_uint(&self, modules: &[Module]) -> bool {
        match &self.body {
            TypeBody::TypeRef(t) => match &t.get_typedef(modules).body {
                TypeDefBody::Builtin(builtin) => builtin.is_uint(),
                TypeDefBody::Record(_) | TypeDefBody::Interface => false,
            },
            _ => false,
        }
    }

    pub fn is_float(&self, modules: &[Module]) -> bool {
        match &self.body {
            TypeBody::TypeRef(t) => match &t.get_typedef(modules).body {
                TypeDefBody::Builtin(builtin) => builtin.is_float(),
                TypeDefBody::Record(_) | TypeDefBody::Interface => false,
            },
            _ => false,
        }
    }

    pub fn is_never(&self, modules: &[Module]) -> bool {
        self.body.is_never(modules)
    }

    pub fn is_void(&self, modules: &[Module]) -> bool {
        self.body.is_void(modules)
    }

    pub fn merge(
        &self,
        other: &Type,
        variance: Variance,
        modules: &[Module],
    ) -> Option<Type> {
        let body = self.body.merge(&other.body, variance, modules)?;
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
    ) -> Option<Self> {
        let body = self.body.substitute_typevar(substitutions)?;
        Some(Type::new(body, self.loc))
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
                if !a.is_same_of(b) && b.implements(a, modules) =>
            {
                if a.args.is_empty() {
                    return true;
                }

                let b_def = b.get_typedef(modules);

                let impl_decl =
                    b_def.impls.iter().find(|decl| decl.iface == a.key).unwrap();
                assert!(impl_decl.iface_args.len() == a.args.len());

                let subs = izip!(&b_def.generics, &b.args)
                    .map(|(&typevar, arg)| (typevar, arg.clone()))
                    .collect();

                for (a_arg, iface_arg) in izip!(&a.args, &impl_decl.iface_args) {
                    let iface_arg_ty = Type::new(iface_arg.clone(), None);
                    let resolved_ty = iface_arg_ty
                        .substitute_typevar(&subs)
                        .unwrap_or(iface_arg_ty);

                    if !a_arg.collect_typevar_substitutions(
                        &Type::new(resolved_ty.body, None),
                        variance,
                        substitutions,
                        modules,
                    ) {
                        return false;
                    }
                }
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
            _ if self.merge(other, variance, modules).is_none() => return false,
            _ => {}
        }
        true
    }

    /// Returns a new type with all typevars marked as rigid or not. See [`TypeVar`].
    pub fn with_rigid(self, rigid: bool) -> Self {
        Self {
            body: self.body.with_rigid(rigid),
            loc:  self.loc,
        }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TypeRefKey {
    Builtin(BuiltinType),
    Custom { mod_idx: usize, idx: usize },
}

impl TypeRefKey {
    pub fn get_typedef<'a>(&self, modules: &'a [Module]) -> &'a TypeDef {
        match self {
            TypeRefKey::Builtin(needle) => modules[BUILTINS_MODULE_IDX]
                .typedefs
                .iter()
                .find(|def| matches!(&def.body, TypeDefBody::Builtin(b) if b == needle))
                .expect("builtin type not found in builtins module"),
            TypeRefKey::Custom { mod_idx, idx } => &modules[*mod_idx].typedefs[*idx],
        }
    }

    pub fn get_typedef_mut<'a>(&self, modules: &'a mut [Module]) -> &'a mut TypeDef {
        match self {
            TypeRefKey::Builtin(needle) => modules[BUILTINS_MODULE_IDX]
                .typedefs
                .iter_mut()
                .find(|def| matches!(&def.body, TypeDefBody::Builtin(b) if b == needle))
                .expect("builtin type not found in builtins module"),
            TypeRefKey::Custom { mod_idx, idx } => &mut modules[*mod_idx].typedefs[*idx],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Setters, ctor)]
#[setters(into, prefix = "with_")]
pub struct TypeRef {
    pub key:  TypeRefKey,
    #[ctor(default)]
    pub args: Vec<Type>,
}

impl TypeRef {
    pub fn builtin(builtin: BuiltinType, args: impl IntoIterator<Item = Type>) -> Self {
        TypeRef::new(TypeRefKey::Builtin(builtin))
            .with_args(args.into_iter().collect_vec())
    }

    pub fn get_typedef<'a>(&self, modules: &'a [Module]) -> &'a TypeDef {
        self.key.get_typedef(modules)
    }

    pub fn get_typedef_mut<'a>(&self, modules: &'a mut [Module]) -> &'a mut TypeDef {
        self.key.get_typedef_mut(modules)
    }

    pub fn is_same_of(&self, other: &TypeRef) -> bool {
        self.key == other.key
    }

    pub fn merge(
        &self,
        other: &Self,
        variance: Variance,
        modules: &[Module],
    ) -> Option<Self> {
        let result =
            |key: TypeRefKey, args: Vec<Type>| Some(TypeRef::new(key).with_args(args));

        if self.is_same_of(other) {
            if self.args.len() != other.args.len() {
                return None;
            }
            let args = izip!(&self.args, &other.args)
                .map(|(self_arg, other_arg)| self_arg.merge(other_arg, variance, modules))
                .collect::<Option<Vec<_>>>()?;
            return result(self.key, args);
        }

        match (self, other) {
            unordered!(a, b) if a.implements(b, modules) => match variance {
                Variance::Covariant => result(a.key, a.args.clone()),
                Variance::Contravariant => result(b.key, b.args.clone()),
            },
            unordered!(a, b) if a.replaces(b, modules) => result(a.key, a.args.clone()),
            _ => None,
        }
    }

    /// Returns true if `self` implements `other`, meaning that `self` can be used as a
    /// type for `other` in dynamic dispatch.
    pub fn implements(&self, other: &Self, modules: &[Module]) -> bool {
        if self.is_same_of(other) {
            todo!("does a type implement itself?");
        }

        let self_def = self.get_typedef(modules);
        let other_def = other.get_typedef(modules);

        match (&self_def.body, &other_def.body) {
            (_, TypeDefBody::Interface) => {
                let args: Vec<TypeBody> =
                    self.args.iter().map(|arg| arg.body.clone()).collect();

                self_def.impls.iter().any(|impl_decl| {
                    if impl_decl.iface != other.key {
                        return false;
                    }
                    if !impl_decl.constraints_satisfied(&args, modules) {
                        return false;
                    }

                    if impl_decl.iface_args.is_empty() {
                        return true;
                    }

                    if other.args.is_empty() {
                        return false;
                    }

                    let substitutions: HashMap<TypeVarIdx, Type> =
                        izip!(&self_def.generics, &self.args)
                            .map(|(&typevar, arg)| (typevar, arg.clone()))
                            .collect();

                    let resolved: Vec<TypeBody> = impl_decl
                        .iface_args
                        .iter()
                        .map(|arg| {
                            Type::new(arg.clone(), None)
                                .substitute_typevar(&substitutions)
                                .map_or_else(|| arg.clone(), |ty| ty.body)
                        })
                        .collect();

                    if resolved.len() != other.args.len() {
                        return false;
                    }

                    izip!(&resolved, &other.args).all(|(resolved, other_arg)| {
                        if let TypeBody::TypeVar(typevar) = &other_arg.body {
                            let def =
                                &modules[typevar.mod_idx].typevars[typevar.typevar_idx];
                            match &def.constraint {
                                Some(constraint) => {
                                    resolved.extends(&constraint.body, modules)
                                }
                                None => true,
                            }
                        } else {
                            resolved
                                .merge(&other_arg.body, Variance::Covariant, modules)
                                .is_some()
                        }
                    })
                })
            }
            _ => false,
        }
    }

    /// Returns true if `self` replaces `other`, meaning that `other` is not a final type
    /// and `self` can be used as a type for `other` statically, but not necessarily
    /// dynamically.
    pub fn replaces(&self, other: &Self, modules: &[Module]) -> bool {
        if self.is_same_of(other) || self.implements(other, modules) {
            return true;
        }

        let self_def = self.get_typedef(modules);
        let other_def = other.get_typedef(modules);

        macro_rules! number {
            ($var:ident $( , $gen:ident)* $(,)?) => {
                (
                    TypeDefBody::Builtin(BuiltinType::$var),
                    TypeDefBody::Builtin(BuiltinType::AnyNumber $( | BuiltinType::$gen)*)
                )
            };
        }

        match (&self_def.body, &other_def.body) {
            number!(U8)
            | number!(U16)
            | number!(U32)
            | number!(U64)
            | number!(USize)
            | number!(AnySignedNumber)
            | number!(I8, AnySignedNumber)
            | number!(I16, AnySignedNumber)
            | number!(I32, AnySignedNumber)
            | number!(I64, AnySignedNumber)
            | number!(AnyFloat, AnySignedNumber)
            | number!(F32, AnyFloat, AnySignedNumber)
            | number!(F64, AnyFloat, AnySignedNumber) => true,
            _ => false,
        }
    }

    pub fn field<'a>(
        &'a self,
        name: &str,
        modules: &'a [Module],
    ) -> Option<Cow<'a, Type>> {
        match &self.get_typedef(modules).body {
            TypeDefBody::Record(rec) => {
                let ty = &rec.fields.get(name)?.ty;
                let substitutions = self.typevar_substitutions(modules);
                if let Some(ty) = ty.substitute_typevar(&substitutions) {
                    Some(Cow::Owned(ty))
                } else {
                    Some(Cow::Borrowed(ty))
                }
            }
            TypeDefBody::Interface | TypeDefBody::Builtin(_) => None,
        }
    }

    pub fn method<'a>(
        &'a self,
        name: &str,
        modules: &'a [Module],
    ) -> Option<Cow<'a, Type>> {
        let typedef = self.get_typedef(modules);
        let method = typedef.methods.get(name)?;
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
            FuncType::new(params_tys, ret_ty.into()).into(),
            Some(method.loc),
        )))
    }

    pub fn method_instanced_ref(
        &self,
        name: &str,
        modules: &[Module],
    ) -> Option<(usize, usize)> {
        let typedef = self.get_typedef(modules);
        let base_method = &typedef.methods[name];

        let func = &modules[base_method.func_ref.0].funcs[base_method.func_ref.1];
        let instantiated_funcs = func
            .generic_instantiations
            .values()
            .map(|&func_idx| (base_method.func_ref.0, func_idx));

        chain!(Some(base_method.func_ref), instantiated_funcs).find_map(|func_ref| {
            let func = &modules[func_ref.0].funcs[func_ref.1];
            let recv_ty = &modules[base_method.func_ref.0].values[func.params[0]].ty;
            if recv_ty.body.has_typevars()
                || !matches_if!(
                    &recv_ty.body,
                    TypeBody::TypeRef(recv_ty_ref),
                    recv_ty_ref
                        .merge(self, Variance::Covariant, modules)
                        .is_some()
                )
            {
                return None;
            }
            Some(func_ref)
        })
    }

    pub fn typevar_substitutions(&self, modules: &[Module]) -> HashMap<TypeVarIdx, Type> {
        let def = self.get_typedef(modules);
        izip!(&def.generics, &self.args)
            .map(|(&typevar, arg)| (typevar, arg.clone()))
            .collect()
    }

    pub fn to_inferred(&self, modules: &[Module]) -> InferredType {
        let def = self.get_typedef(modules);

        let fields = match &def.body {
            TypeDefBody::Record(rec) => &rec.fields,
            TypeDefBody::Interface | TypeDefBody::Builtin(_) => &SortedMap::new(),
        };

        let mut members = utils::SortedMap::new();
        let mut properties = utils::SortedMap::new();

        for name in fields.keys() {
            let Some(ty) = self.field(name, modules) else {
                continue;
            };
            members.insert(name.to_string(), ty.clone().into_owned());
        }

        for name in chain!(fields.keys(), def.methods.keys()).unique() {
            let as_ty_body = TypeBody::TypeRef(self.clone());
            let Some(ty) = as_ty_body.property(name, modules) else {
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
    #[ctor(expr(true))]
    /// Represents a fixed concrete type within a single instantiation. It can only unify
    /// with itself, not with arbitrary types that satisfy its constraint.
    pub rigid:       bool,
}

impl TypeVar {
    pub fn is_same_of(&self, other: &Self) -> bool {
        self.mod_idx == other.mod_idx && self.typevar_idx == other.typevar_idx
    }

    pub fn field<'a>(
        &'a self,
        name: &str,
        modules: &'a [Module],
    ) -> Option<Cow<'a, Type>> {
        let def = &modules[self.mod_idx].typevars[self.typevar_idx];
        let cons = def.constraint.as_ref()?;
        cons.body.field(name, modules).map(|mut ty| {
            if ty.body.has_typevars() {
                utils::replace_with(&mut ty.to_mut().body, |tyb| {
                    tyb.with_rigid(self.rigid)
                });
            }
            ty
        })
    }

    pub fn method<'a>(
        &'a self,
        name: &str,
        modules: &'a [Module],
    ) -> Option<Cow<'a, Type>> {
        let def = &modules[self.mod_idx].typevars[self.typevar_idx];
        let cons = def.constraint.as_ref()?;
        let method = cons.body.method(name, modules);
        method.map(|mut ty| {
            if ty.body.has_typevars() {
                utils::replace_with(&mut ty.to_mut().body, |tyb| {
                    tyb.with_rigid(self.rigid)
                });
            }
            ty
        })
    }

    pub fn extends(&self, other: &Self, modules: &[Module]) -> bool {
        if other.rigid {
            return self.is_same_of(other);
        }
        let def_a = &modules[self.mod_idx].typevars[self.typevar_idx];
        let def_b = &modules[other.mod_idx].typevars[other.typevar_idx];
        match (&def_a.constraint, &def_b.constraint) {
            (Some(cons_a), Some(cons_b)) => cons_a.body.extends(&cons_b.body, modules),
            (Some(_), None) => true,
            _ => false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
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

    pub fn merge(
        &self,
        other: &Self,
        variance: Variance,
        modules: &[Module],
    ) -> Option<Self> {
        let mut members = SortedMap::new();
        for name in chain!(self.members.keys(), other.members.keys()).unique() {
            let ty = match (self.members.get(name), other.members.get(name)) {
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
        for name in chain!(self.properties.keys(), other.properties.keys()).unique() {
            let ty = match (self.properties.get(name), other.properties.get(name)) {
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

        Some(Self {
            members,
            properties,
        })
    }

    pub fn extends(&self, other: &Self, modules: &[Module]) -> bool {
        // TODO: optional fields
        self.members.len() == other.members.len()
            && other.members.iter().all(|(name, b)| {
                self.members
                    .get(name)
                    .is_some_and(|a| a.body.extends(&b.body, modules))
            })
            && other.properties.iter().all(|(name, b)| {
                self.properties
                    .get(name)
                    .is_some_and(|a| a.body.extends(&b.body, modules))
            })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, ctor)]
pub struct FuncType {
    pub params: Vec<Type>,
    pub ret:    Box<Type>,
}
impl FuncType {
    pub fn from_func(
        mod_idx: usize,
        func_idx: usize,
        rigid: bool,
        modules: &[Module],
    ) -> Self {
        let func = &modules[mod_idx].funcs[func_idx];
        let params = func
            .params
            .iter()
            .map(|&v| modules[mod_idx].values[v].ty.clone().with_rigid(rigid))
            .collect_vec();
        let ret = modules[mod_idx].values[func.ret]
            .ty
            .clone()
            .with_rigid(rigid);
        Self {
            params,
            ret: ret.into(),
        }
    }

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
            self.ret.merge(&other.ret, var, modules)?.into(),
        ))
    }

    pub fn extends(&self, other: &Self, is_method: bool, modules: &[Module]) -> bool {
        if self.params.len() != other.params.len() {
            return false;
        }
        izip!(&self.params, &other.params)
            .enumerate()
            .all(|(i, (a_param, b_param))| {
                (is_method && i == 0 && a_param.body.extends(&b_param.body, modules))
                    || b_param.body.extends(&a_param.body, modules)
            })
            && self.ret.body.extends(&other.ret.body, modules)
    }
}

struct TypevarSubstitutions<'a>(&'a HashMap<TypeVarIdx, Type>);

impl<'a> TypevarSubstitutions<'a> {
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
