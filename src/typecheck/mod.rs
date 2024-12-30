mod constraints;

use std::collections::HashSet;
use std::fmt::Debug;
use std::mem;

use derive_new::new;
use itertools::{enumerate, izip, Itertools};

use self::constraints::Constraint;
use crate::utils::SortedMap;
use crate::{bytecode as b, context, errors, utils};

#[derive(Debug, Clone, new)]
pub struct TypeChecker<'a> {
    ctx: &'a context::BuildContext,
    mod_idx: usize,
    #[new(default)]
    value_constraints: Vec<HashSet<Constraint>>,
}

impl<'a> TypeChecker<'a> {
    #[tracing::instrument(skip(self))]
    pub fn check(&mut self) {
        tracing::trace!("check started");

        let (globals_len, funcs_len, value_constraints) = {
            let module = &self.ctx.lock_modules()[self.mod_idx];
            let value_contraints = module
                .values
                .iter()
                .map(|value| {
                    if !value.ty.is_unknown() {
                        HashSet::from([Constraint::Is(value.ty.clone())])
                    } else {
                        HashSet::new()
                    }
                })
                .collect();
            (module.globals.len(), module.funcs.len(), value_contraints)
        };

        self.value_constraints = value_constraints;

        for i in 0..funcs_len {
            tracing::trace!(i, "adding func");
            let (body, ret) = {
                let module = &self.ctx.lock_modules()[self.mod_idx];
                (module.funcs[i].body.clone(), module.funcs[i].ret)
            };
            let mut scopes = utils::ScopeStack::new(ScopePayload::new());
            if let Some(result) = self.add_body(body, &mut scopes, Some(i)) {
                self.merge_types([&ret, &result]);
            }
        }

        for i in 0..globals_len {
            tracing::trace!(i, "adding global");
            let (body, value) = {
                let module = &self.ctx.lock_modules()[self.mod_idx];
                (module.globals[i].body.clone(), module.globals[i].value)
            };
            let mut scopes = utils::ScopeStack::new(ScopePayload::new());
            if let Some(result) = self.add_body(body, &mut scopes, None) {
                self.merge_types([&value, &result]);
            }
        }

        self.validate();

        for i in 0..globals_len {
            self.finish_body(
                |module| &module.globals[i].body,
                |module| &mut module.globals[i].body,
            );
        }
        for i in 0..funcs_len {
            self.finish_body(
                |module| &module.funcs[i].body,
                |module| &mut module.funcs[i].body,
            );
        }
    }

    #[tracing::instrument(level = "trace", skip_all)]
    fn add_body(
        &mut self,
        body: Vec<b::Instr>,
        scopes: &mut utils::ScopeStack<ScopePayload>,
        func_idx: Option<usize>,
    ) -> Option<b::ValueIdx> {
        if body.len() == 0 {
            return None;
        }

        let mut result = None;
        for instr in body {
            if let b::InstrBody::Break(v) = &instr.body {
                result = result.or(Some(*v));
                continue;
            }
            self.add_instr(instr, scopes, func_idx);
        }

        return result;
    }

    #[tracing::instrument(skip(self, scopes))]
    fn add_instr(
        &mut self,
        mut instr: b::Instr,
        scopes: &mut utils::ScopeStack<ScopePayload>,
        func_idx: Option<usize>,
    ) {
        tracing::trace!("add instr");

        match &mut instr.body {
            b::InstrBody::GetGlobal(mod_idx, idx) => {
                let v = instr.results[0];
                if *mod_idx == self.mod_idx {
                    let gv = { self.ctx.lock_modules()[*mod_idx].globals[*idx].value };
                    self.merge_types([&gv, &v]);
                } else {
                    let ty = {
                        let module = &self.ctx.lock_modules()[*mod_idx];
                        let gv = module.globals[*idx].value;
                        module.values[gv].ty.clone()
                    };
                    self.add_constraint(v, Constraint::Is(ty));
                };
            }
            b::InstrBody::GetProperty(source_v, name)
            | b::InstrBody::GetField(source_v, name)
            | b::InstrBody::GetMethod(source_v, name) => {
                let v = instr.results[0];
                self.define_property(*source_v, v, name, instr.loc);
            }
            b::InstrBody::CreateBool(_) => {
                let v = instr.results[0];
                let ty = b::Type::new(b::TypeBody::Bool, None);
                self.add_constraint(v, Constraint::Is(ty));
            }
            b::InstrBody::CreateNumber(num) => {
                let v = instr.results[0];
                // TODO: use better type
                let ty_body = if num.contains('.') {
                    b::TypeBody::AnyFloat
                } else if num.starts_with('-') {
                    b::TypeBody::AnySignedNumber
                } else {
                    b::TypeBody::AnyNumber
                };
                self.add_constraint(v, Constraint::Is(b::Type::new(ty_body, None)));
            }
            b::InstrBody::CreateString(x) => {
                let v = instr.results[0];
                let ty = b::Type::new(
                    b::TypeBody::String(b::StringType { len: Some(x.len()) }),
                    None,
                );
                self.add_constraint(v, Constraint::Is(ty.clone()));
            }
            b::InstrBody::CreateArray(vs) => {
                let v = instr.results[0];
                if vs.len() > 0 {
                    self.merge_types(&*vs);
                    self.add_constraint(v, Constraint::Array(vs[0]));
                } else {
                    let item_ty = b::Type::new(b::TypeBody::Never, None);
                    let arr_ty = b::Type::new(
                        b::TypeBody::Array(b::ArrayType {
                            len: Some(0),
                            item: item_ty.into(),
                        }),
                        None,
                    );
                    self.add_constraint(v, Constraint::Is(arr_ty));
                }
            }
            b::InstrBody::CreateRecord(fields) => {
                let v = instr.results[0];
                self.add_constraint(v, Constraint::Members(fields.clone()));
                for (name, fields_v) in fields {
                    self.define_property(v, *fields_v, name, instr.loc);
                }
            }
            b::InstrBody::Add(a, b)
            | b::InstrBody::Sub(a, b)
            | b::InstrBody::Mul(a, b)
            | b::InstrBody::Div(a, b)
            | b::InstrBody::Mod(a, b) => {
                let v = instr.results[0];
                self.merge_types([a, b, &v]);
                // FIXME: use interface/trait
                let ty = b::Type::new(b::TypeBody::AnyNumber, None);
                self.add_constraint(*a, Constraint::Is(ty));
            }
            b::InstrBody::Not(x) => {
                let v = instr.results[0];
                self.merge_types([x, &v]);
                // FIXME: use interface/trait
                self.add_constraint(
                    *x,
                    Constraint::Is(b::Type::new(b::TypeBody::Bool, None)),
                );
            }
            b::InstrBody::Eq(a, b)
            | b::InstrBody::Neq(a, b)
            | b::InstrBody::Gt(a, b)
            | b::InstrBody::Gte(a, b)
            | b::InstrBody::Lt(a, b)
            | b::InstrBody::Lte(a, b) => {
                let v = instr.results[0];
                self.merge_types([a, &*b]);
                // FIXME: use interface/trait
                let number_ty = b::Type::new(b::TypeBody::AnyNumber, None);
                let bool_ty = b::Type::new(b::TypeBody::Bool, None);
                self.add_constraint(*a, Constraint::Is(number_ty));
                self.add_constraint(v, Constraint::Is(bool_ty));
            }
            b::InstrBody::Call(mod_idx, idx, args) => {
                let v = instr.results[0];

                if *mod_idx == self.mod_idx {
                    let func = self.ctx.lock_modules()[self.mod_idx].funcs[*idx].clone();

                    if func_idx.is_some_and(|i| i == *idx) {
                        self.merge_types([&func.ret, &v]);
                        for (arg, param) in izip!(args, func.params) {
                            self.merge_types([&param, arg]);
                        }
                    } else {
                        for (arg, param) in izip!(args, func.params) {
                            self.add_constraint(*arg, Constraint::TypeOf(param));
                        }
                        self.add_constraint(v, Constraint::TypeOf(func.ret));
                    }
                } else {
                    let (params_tys, ret_ty) = {
                        let module = &self.ctx.lock_modules()[*mod_idx];
                        let func = &module.funcs[*idx];
                        (
                            func.params
                                .iter()
                                .map(|param| module.values[*param].ty.clone())
                                .collect_vec(),
                            module.values[func.ret].ty.clone(),
                        )
                    };

                    for (arg, param_ty) in izip!(args, params_tys) {
                        self.add_constraint(*arg, Constraint::Is(param_ty));
                    }
                    self.add_constraint(v, Constraint::Is(ret_ty))
                }
            }
            b::InstrBody::IndirectCall(func, args) => {
                let v = instr.results[0];

                self.add_constraint(*func, Constraint::Func(args.len()));

                for (i, arg) in enumerate(args) {
                    self.add_constraint(*arg, Constraint::ParameterOf(*func, i));
                }
                self.add_constraint(v, Constraint::ReturnOf(*func));
            }
            b::InstrBody::If(cond_v, then_, else_) => {
                self.add_constraint(
                    *cond_v,
                    Constraint::Is(b::Type::new(b::TypeBody::Bool, None)),
                );

                scopes.begin(ScopePayload::new());
                if let Some(then_v) =
                    self.add_body(std::mem::replace(then_, vec![]), scopes, func_idx)
                {
                    self.merge_types([&then_v, &instr.results[0]]);
                }

                scopes.branch();
                if let Some(else_v) =
                    self.add_body(std::mem::replace(else_, vec![]), scopes, func_idx)
                {
                    self.merge_types([&else_v, &instr.results[0]]);
                }

                scopes.end();
            }
            b::InstrBody::Loop(inputs, body) => {
                let scope = scopes.begin(ScopePayload::new());
                scope.is_loop = true;
                for (loop_v, initial_v) in &*inputs {
                    self.merge_types([initial_v, loop_v]);
                    scope.loop_args.push(*loop_v);
                }

                if let Some(result) =
                    self.add_body(std::mem::replace(body, vec![]), scopes, func_idx)
                {
                    self.merge_types([&result, &instr.results[0]]);
                }

                scopes.end();
            }
            b::InstrBody::Continue(vs) => {
                let loop_args = &scopes
                    .last_loop()
                    .expect("continue should be called inside a loop")
                    .loop_args;
                for (v, loop_v) in izip!(vs, loop_args) {
                    self.merge_types([v, loop_v]);
                }
            }
            b::InstrBody::ArrayLen(input) => {
                let v = instr.results[0];
                let item_ty = b::Type::unknown(None);
                let arr_ty = b::Type::new(
                    b::TypeBody::Array(b::ArrayType::new(item_ty.into(), None)),
                    None,
                );
                self.add_constraint(*input, Constraint::Is(arr_ty));
                let ty = b::Type::new(b::TypeBody::USize, None);
                self.add_constraint(v, Constraint::Is(ty));
            }
            b::InstrBody::ArrayPtr(input, _) => {
                let v = instr.results[0];
                let arr_ty = b::Type::new(
                    b::TypeBody::Array(b::ArrayType::new(
                        b::Type::unknown(None).into(),
                        None,
                    )),
                    None,
                );
                self.add_constraint(*input, Constraint::Is(arr_ty));
                self.add_constraint(v, Constraint::ArrayElemPtr(*input));
            }
            b::InstrBody::StrLen(input) => {
                let v = instr.results[0];
                let str_ty =
                    b::Type::new(b::TypeBody::String(b::StringType::new(None)), None);
                self.add_constraint(*input, Constraint::Is(str_ty));
                let ty = b::Type::new(b::TypeBody::USize, None);
                self.add_constraint(v, Constraint::Is(ty));
            }
            b::InstrBody::StrPtr(input, _) => {
                let v = instr.results[0];
                let str_ty =
                    b::Type::new(b::TypeBody::String(b::StringType::new(None)), None);
                self.add_constraint(*input, Constraint::Is(str_ty));
                let ty = b::Type::new(
                    b::TypeBody::Ptr(b::Type::new(b::TypeBody::U8, None).into()),
                    None,
                );
                self.add_constraint(v, Constraint::Is(ty));
            }
            b::InstrBody::Type(v, ty) => {
                self.add_constraint(*v, Constraint::Is(ty.clone()));
            }
            b::InstrBody::Break(_) | b::InstrBody::CompileError => {}
        }
    }

    #[tracing::instrument(skip(self))]
    fn add_constraint(&mut self, idx: b::ValueIdx, constraint: Constraint) {
        tracing::trace!("add constraint");

        let same_of = {
            let value = &self.ctx.lock_modules()[self.mod_idx].values[idx];
            if let Some(redirects_to) = &value.redirects_to {
                [*redirects_to].into()
            } else {
                value.same_type_of.clone()
            }
        };

        if same_of.len() > 0 {
            for idx in &same_of {
                self.add_constraint(*idx, constraint.clone());
            }
            return;
        }

        // Some constraints cannot be repeated, and instead indicates that two values have
        // the same type. In these cases, e merge the values types
        for c in &self.value_constraints[idx].clone() {
            match (c, &constraint) {
                (Constraint::Array(a), Constraint::Array(b))
                | (Constraint::Ptr(a), Constraint::Ptr(b))
                | (Constraint::ArrayElemPtr(a), Constraint::ArrayElemPtr(b)) => {
                    self.merge_types([a, b]);
                }
                (
                    Constraint::HasProperty(name_a, a),
                    Constraint::HasProperty(name_b, b),
                )
                | (
                    Constraint::IsProperty(a, name_a),
                    Constraint::IsProperty(b, name_b),
                ) if name_a == name_b => {
                    self.merge_types([a, b]);
                }
                _ => {}
            }
        }

        self.value_constraints[idx].insert(constraint);
    }

    #[tracing::instrument(level = "trace", skip(self))]
    fn merge_types<'i, I>(&mut self, items: I)
    where
        I: IntoIterator<Item = &'i b::ValueIdx>,
        I: Debug,
    {
        tracing::trace!("merge types");

        let mut merge_with = items
            .into_iter()
            .sorted_by(|a, b| a.cmp(b).reverse())
            .copied()
            .collect_vec();

        let head = merge_with.pop().unwrap();

        while let Some(idx) = merge_with.pop() {
            let constraints = {
                let values = &mut self.ctx.lock_modules_mut()[self.mod_idx].values;

                values[idx].same_type_of.insert(head);
                mem::replace(&mut self.value_constraints[idx], HashSet::new())
            };

            for constraint in constraints {
                self.add_constraint(head, constraint);
            }
        }
    }

    #[tracing::instrument(skip(self))]
    fn validate(&mut self) -> bool {
        let mut success = true;
        let mut visited = HashSet::new();

        let len = { self.ctx.lock_modules()[self.mod_idx].values.len() };

        for idx in 0..len {
            success = self.validate_value(idx, &mut visited) && success;
            tracing::trace!(idx, ?success, "validated value");
        }

        tracing::info!(success, "validation completed");

        success
    }

    #[tracing::instrument(level = "trace", skip(self, visited))]
    fn validate_value(
        &mut self,
        idx: b::ValueIdx,
        visited: &mut HashSet<b::ValueIdx>,
    ) -> bool {
        let already_visited = visited.contains(&idx);

        tracing::trace!(?already_visited, "validate_value");

        if already_visited {
            return true;
        }
        visited.insert(idx);

        let mut success = true;

        let same_of = {
            self.ctx.lock_modules()[self.mod_idx].values[idx]
                .same_type_of
                .clone()
        };

        if same_of.len() > 0 {
            let mut tys = vec![];
            for i in same_of {
                tracing::trace!(i, "will validate same_type_of");
                success = self.validate_value(i, visited) && success;
                if !success {
                    return false;
                };
                tys.push(self.ctx.lock_modules()[self.mod_idx].values[i].ty.clone());
            }

            let mut result_ty = tys[0].clone();
            for ty in &tys[1..] {
                if let Some(ty) = result_ty.union(ty, &self.ctx.lock_modules()) {
                    result_ty = ty;
                } else {
                    self.ctx.push_error(errors::Error::new(
                        errors::TypeMisatch::new(tys).into(),
                        self.ctx.lock_modules()[self.mod_idx].values[idx].loc,
                    ));
                    return false;
                }
            }

            self.ctx.lock_modules_mut()[self.mod_idx].values[idx].ty = result_ty;

            return success;
        }

        let constraints = self.value_constraints[idx]
            .iter()
            .cloned()
            .sorted_by(|a, b| b.priority().cmp(&a.priority()))
            .collect_vec();

        for c in constraints {
            tracing::trace!(?c, "checking constraint");
            let merge_with = match c {
                Constraint::Is(ty) => ty.clone(),
                Constraint::TypeOf(target) => {
                    tracing::trace!(target, "will validate TypeOf");
                    success = self.validate_value(target, visited) && success;
                    self.ctx.lock_modules()[self.mod_idx].values[target]
                        .ty
                        .clone()
                }
                Constraint::Array(target) => {
                    tracing::trace!(target, "will validate Array");
                    success = self.validate_value(target, visited) && success;
                    let ty = self.ctx.lock_modules()[self.mod_idx].values[target]
                        .ty
                        .clone();
                    b::Type::new(
                        b::TypeBody::Array(b::ArrayType::new(ty.into(), None)),
                        None,
                    )
                }
                Constraint::Ptr(target) => {
                    tracing::trace!(target, "will validate Ptr");
                    success = self.validate_value(target, visited) && success;
                    let ty = self.ctx.lock_modules()[self.mod_idx].values[target]
                        .ty
                        .clone();
                    b::Type::new(b::TypeBody::Ptr(ty.into()), None)
                }
                Constraint::ArrayElemPtr(target) => {
                    tracing::trace!(target, "will validate ArrayElemPtr");
                    success = self.validate_value(target, visited) && success;
                    let item_ty = if let b::TypeBody::Array(arr_ty) =
                        &self.ctx.lock_modules()[self.mod_idx].values[target].ty.body
                    {
                        arr_ty.item.clone()
                    } else {
                        b::Type::unknown(None).into()
                    };
                    b::Type::new(b::TypeBody::Ptr(item_ty), None)
                }
                Constraint::ReturnOf(target) => {
                    tracing::trace!(target, "will validate ReturnOf");
                    success = self.validate_value(target, visited) && success;
                    if let b::TypeBody::Func(func_ty) =
                        &self.ctx.lock_modules()[self.mod_idx].values[target].ty.body
                    {
                        func_ty.ret.clone()
                    } else {
                        b::Type::unknown(None)
                    }
                }
                Constraint::ParameterOf(target, idx) => {
                    tracing::trace!(target, idx, "will validate ParameterOf");
                    success = self.validate_value(target, visited) && success;
                    if let b::TypeBody::Func(func_ty) =
                        &self.ctx.lock_modules()[self.mod_idx].values[target].ty.body
                    {
                        func_ty
                            .params
                            .get(idx)
                            .cloned()
                            .unwrap_or(b::Type::unknown(None))
                    } else {
                        b::Type::unknown(None)
                    }
                }
                Constraint::IsProperty(target, key) => {
                    tracing::trace!(target, key, "will validate IsProperty");
                    success = self.validate_value(target, visited) && success;
                    for prop_dep in {
                        let modules = self.ctx.lock_modules();
                        self.get_property_deps(target, &key, &modules)
                    } {
                        tracing::trace!(prop_dep, "will validate property_deps");
                        success = self.validate_value(prop_dep, visited) && success;
                    }
                    if let Some(ty) =
                        self.get_property_type(target, &key, &self.ctx.lock_modules())
                    {
                        ty
                    } else {
                        success = false;
                        b::Type::unknown(None)
                    }
                }
                Constraint::Members(members) => {
                    for member in members.values() {
                        tracing::trace!(member, "will validate member");
                        success = self.validate_value(*member, visited) && success;
                    }
                    b::Type::new(
                        b::TypeBody::Inferred(b::InferredType {
                            members: members
                                .iter()
                                .map(|(k, v)| {
                                    let value =
                                        &self.ctx.lock_modules()[self.mod_idx].values[*v];
                                    (k.clone(), value.ty.clone())
                                })
                                .collect(),
                            properties: SortedMap::new(),
                        }),
                        None,
                    )
                }
                Constraint::HasProperty(key, target) => {
                    tracing::trace!(key, target, "will validate HasProperty");
                    success = self.validate_value(target, visited) && success;
                    let ty = {
                        self.ctx.lock_modules()[self.mod_idx].values[target]
                            .ty
                            .clone()
                    };
                    b::Type::new(
                        b::TypeBody::Inferred(b::InferredType {
                            properties: SortedMap::from([(key.clone(), ty)]),
                            members: SortedMap::new(),
                        }),
                        None,
                    )
                }
                Constraint::Func(n) => b::Type::new(
                    b::TypeBody::Func(Box::new(b::FuncType {
                        params: vec![b::Type::unknown(None); n],
                        ret: b::Type::unknown(None),
                    })),
                    None,
                ),
            };

            tracing::trace!(?merge_with, "got type");

            {
                let modules = &mut self.ctx.lock_modules_mut();
                if let Some(result_ty) = modules[self.mod_idx].values[idx]
                    .ty
                    .intersection(&merge_with, modules)
                {
                    modules[self.mod_idx].values[idx].ty = result_ty;
                } else {
                    self.ctx.push_error(errors::Error::new(
                        errors::UnexpectedType::new(
                            modules[self.mod_idx].values[idx].ty.to_owned(),
                            merge_with.clone(),
                        )
                        .into(),
                        modules[self.mod_idx].values[idx].loc,
                    ));
                    success = false;
                }
            }
        }

        {
            let value = &self.ctx.lock_modules()[self.mod_idx].values[idx];
            if success
                && matches!(
                    &value.ty.body,
                    b::TypeBody::AnyNumber
                        | b::TypeBody::AnySignedNumber
                        | b::TypeBody::AnyFloat
                        | b::TypeBody::Inferred(_)
                )
            {
                self.ctx.push_error(errors::Error::new(
                    errors::ErrorDetail::TypeNotFinal,
                    value.loc,
                ));
                success = false;
            }
        }

        success
    }

    fn define_property(
        &mut self,
        src_v: b::ValueIdx,
        prop_v: b::ValueIdx,
        prop_name: &str,
        loc: b::Loc,
    ) {
        let same_of = {
            self.ctx.lock_modules()[self.mod_idx].values[src_v]
                .same_type_of
                .clone()
        };

        if same_of.len() >= 1 {
            for v in &same_of {
                self.define_property(*v, prop_v, prop_name, loc);
            }
            return;
        }

        for item in &self.value_constraints[src_v] {
            if let Constraint::HasProperty(prop_name_, prop_v_) = item {
                if prop_name == prop_name_ {
                    self.merge_types(&[*prop_v_, prop_v]);
                    return;
                }
            }
        }

        self.add_constraint(
            src_v,
            Constraint::HasProperty(prop_name.to_string(), prop_v),
        );
        self.add_constraint(prop_v, Constraint::IsProperty(src_v, prop_name.to_string()));
    }

    fn get_property_deps(
        &self,
        v: b::ValueIdx,
        name: &str,
        modules: &[b::Module],
    ) -> Vec<b::ValueIdx> {
        let module = &modules[self.mod_idx];
        let parent = &module.values[v].ty;
        let b::TypeBody::TypeRef(mod_idx, ty_idx) = &parent.body else {
            return vec![];
        };
        let Some(func) = modules
            .get(*mod_idx)
            .and_then(|module| module.typedefs.get(*ty_idx))
            .and_then(|typedef| match &typedef.body {
                b::TypeDefBody::Record(rec) => rec.methods.get(name),
            })
            .and_then(|method| {
                if method.func_ref.0 == self.mod_idx {
                    return module.funcs.get(method.func_ref.1);
                } else {
                    return None;
                }
            })
        else {
            return vec![];
        };
        return func.params.iter().cloned().chain([func.ret]).collect();
    }

    fn get_property_type(
        &self,
        v: b::ValueIdx,
        key: &str,
        modules: &[b::Module],
    ) -> Option<b::Type> {
        let module = &modules[self.mod_idx];
        let parent = &module.values[v].ty;
        'from_entries: {
            let b::TypeBody::TypeRef(mod_idx, ty_idx) = &parent.body else {
                break 'from_entries;
            };
            let Some((func, loc)) = modules
                .get(*mod_idx)
                .and_then(|module| module.typedefs.get(*ty_idx))
                .and_then(|typedef| match &typedef.body {
                    b::TypeDefBody::Record(rec) => rec.methods.get(key),
                })
                .and_then(|method| {
                    if method.func_ref.0 == self.mod_idx {
                        return Some((module.funcs.get(method.func_ref.1)?, method.loc));
                    } else {
                        return None;
                    }
                })
            else {
                break 'from_entries;
            };

            let [params @ .., self_param] = &func.params[..] else {
                break 'from_entries;
            };
            // is static?
            if module.values[*self_param].ty.body != parent.body {
                return None;
            }
            // functions without parameters are just values
            if params.len() == 0 {
                return Some(module.values[func.ret].ty.clone());
            }
            return Some(b::Type::new(
                b::TypeBody::Func(
                    b::FuncType::new(
                        params
                            .iter()
                            .map(|x| module.values[*x].ty.clone())
                            .collect(),
                        module.values[func.ret].ty.clone(),
                    )
                    .into(),
                ),
                Some(loc),
            ));
        }
        return parent.property(key, modules).map(|ty| ty.into_owned());
    }

    #[tracing::instrument(level = "trace", skip_all)]
    fn finish_body(
        &self,
        get_body: impl for<'m> Fn(&'m b::Module) -> &'m [b::Instr],
        get_body_mut: impl for<'m> Fn(&'m mut b::Module) -> &'m mut [b::Instr],
    ) {
        let len = {
            let module = &self.ctx.lock_modules()[self.mod_idx];
            get_body(module).len()
        };
        for i in 0..len {
            let get_property = {
                let modules = &self.ctx.lock_modules()[self.mod_idx];
                let instr = &get_body(modules)[i];
                match &instr.body {
                    b::InstrBody::GetProperty(v, key) => Some((*v, key.clone())),
                    _ => None,
                }
            };

            if let Some((source_v, key)) = get_property {
                let (is_field, is_method) = {
                    let modules = &self.ctx.lock_modules();
                    let parent_ty = &modules[self.mod_idx].values[source_v].ty;
                    (
                        parent_ty.field(&key, &modules).is_some(),
                        parent_ty.method(&key, &modules).is_some(),
                    )
                };

                {
                    let modules = &mut self.ctx.lock_modules_mut();
                    let instr = &mut get_body_mut(&mut modules[self.mod_idx])[i];
                    if is_field {
                        instr.body = b::InstrBody::GetField(source_v, key.clone());
                    } else if is_method {
                        instr.body = b::InstrBody::GetMethod(source_v, key.clone());
                    }
                }
            }
        }
    }
}

#[derive(Debug, Clone, new)]
struct ScopePayload {
    #[new(default)]
    loop_args: Vec<b::ValueIdx>,
}
impl utils::SimpleScopePayload for ScopePayload {}
