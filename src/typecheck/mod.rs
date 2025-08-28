mod constraints;

use std::collections::HashSet;
use std::fmt::Debug;
use std::mem;

use derive_new::new;
use itertools::{enumerate, izip, Itertools};

use self::constraints::Constraint;
use crate::utils::{cfor, SortedMap};
use crate::{bytecode as b, context, errors, utils};

#[derive(Debug, Clone, new)]
struct TypeNode {
    #[new(default)]
    constraints: HashSet<Constraint>,
    #[new(default)]
    status:      TypeNodeStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, new)]
enum TypeNodeStatus {
    #[default]
    Unresolved,
    Failed,
    Resolved,
}
impl TypeNodeStatus {
    fn was_checked(&self) -> bool {
        *self != TypeNodeStatus::Unresolved
    }

    fn is_failed(&self) -> bool {
        *self == TypeNodeStatus::Failed
    }
}

#[derive(Debug, Clone, new)]
pub struct TypeChecker<'a> {
    ctx:     &'a context::BuildContext,
    mod_idx: usize,
    #[new(default)]
    nodes:   Vec<TypeNode>,
}

impl<'a> TypeChecker<'a> {
    #[tracing::instrument(skip(self))]
    pub fn check(&mut self) {
        tracing::trace!("check started");

        let (globals_len, funcs_len, nodes) = {
            let module = &self.ctx.lock_modules()[self.mod_idx];
            let nodes = module
                .values
                .iter()
                .map(|value| {
                    let mut node = TypeNode::new();
                    if !value.ty.is_unknown() {
                        node.constraints.insert(Constraint::Is(value.ty.clone()));
                    }
                    node
                })
                .collect();
            (module.globals.len(), module.funcs.len(), nodes)
        };

        self.nodes = nodes;

        for i in 0..funcs_len {
            self.add_func(i);
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
            self.finish_body(&|module| &module.globals[i].body, &|module| {
                &mut module.globals[i].body
            });
        }
        for i in 0..funcs_len {
            self.finish_body(&|module| &module.funcs[i].body, &|module| {
                &mut module.funcs[i].body
            });
        }
    }

    #[tracing::instrument(level = "trace", skip(self))]
    fn add_func(&mut self, idx: usize) {
        let (body, ret) = {
            let modules = &self.ctx.lock_modules();
            let func = &modules[self.mod_idx].funcs[idx];

            if let Some((mod_idx, ty_idx, name)) = &func.method {
                let parent_funcs = modules[*mod_idx].typedefs[*ty_idx]
                    .get_ifaces()
                    .into_iter()
                    .filter_map(|(mod_idx, ty_idx)| {
                        modules[mod_idx].typedefs[ty_idx].get_method(name)
                    })
                    .map(|f| f.func_ref);
                for (parent_mod_idx, parent_func_idx) in parent_funcs {
                    let parent_func = &modules[parent_mod_idx].funcs[parent_func_idx];
                    for (param, parent_param) in izip!(&func.params, &parent_func.params)
                    {
                        if *mod_idx == parent_mod_idx {
                            self.add_constraint(
                                *param,
                                Constraint::TypeOf(*parent_param),
                            );
                        } else {
                            let v = &modules[parent_mod_idx].values[*parent_param];
                            self.add_constraint(*param, Constraint::Is(v.ty.clone()));
                        }
                    }
                }
            }

            (func.body.clone(), func.ret)
        };
        let mut scopes = utils::ScopeStack::new(ScopePayload::new());
        if let Some(result) = self.add_body(body, &mut scopes, Some(idx)) {
            self.merge_types([&ret, &result]);
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
                result = result.or(*v);
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
            &mut b::InstrBody::GetFunc(mod_idx, func_idx) => {
                let v = instr.results[0];
                self.add_constraint(v, Constraint::GetFunc(mod_idx, func_idx));
            }
            b::InstrBody::GetProperty(source_v, name)
            | b::InstrBody::GetField(source_v, name)
            | b::InstrBody::GetMethod(source_v, name) => {
                let v = instr.results[0];
                self.define_property(*source_v, v, name);
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
                    b::TypeBody::String(b::StringType {
                        len: Some(x.len() as u32),
                    }),
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
                            len:  Some(0),
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
                    self.define_property(v, *fields_v, name);
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

                let mut has_get_func = false;
                for c in self
                    .get_contraints_with(*func, |c| matches!(c, Constraint::GetFunc(..)))
                {
                    let Constraint::GetFunc(mod_idx, func_idx) = c else {
                        continue;
                    };
                    has_get_func = true;

                    let (params, ret) = {
                        let func = &self.ctx.lock_modules()[mod_idx].funcs[func_idx];
                        (func.params.clone(), func.ret)
                    };

                    for (param, arg) in izip!(&params, &*args) {
                        self.add_constraint(*arg, Constraint::TypeOf(*param));
                    }
                    self.add_constraint(v, Constraint::TypeOf(ret));
                }
                if !has_get_func {
                    self.add_constraint(*func, Constraint::Func(args.clone(), v));
                }

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
            b::InstrBody::StrLen(input) => {
                let v = instr.results[0];
                let str_ty =
                    b::Type::new(b::TypeBody::String(b::StringType::new(None)), None);
                self.add_constraint(*input, Constraint::Is(str_ty));
                let ty = b::Type::new(b::TypeBody::USize, None);
                self.add_constraint(v, Constraint::Is(ty));
            }
            b::InstrBody::StrPtr(input) => {
                let v = instr.results[0];
                let str_ty =
                    b::Type::new(b::TypeBody::String(b::StringType::new(None)), None);
                self.add_constraint(*input, Constraint::Is(str_ty));
                let ty = b::Type::new(
                    b::TypeBody::Ptr(Some(b::Type::new(b::TypeBody::U8, None).into())),
                    None,
                );
                self.add_constraint(v, Constraint::Is(ty));
            }
            b::InstrBody::ArrayLen(input) => {
                let v = instr.results[0];
                let arr_ty = b::Type::new(
                    b::TypeBody::Array(b::ArrayType::new(
                        Box::new(b::Type::unknown(None)),
                        None,
                    )),
                    None,
                );
                self.add_constraint(*input, Constraint::Is(arr_ty));
                let ty = b::Type::new(b::TypeBody::USize, None);
                self.add_constraint(v, Constraint::Is(ty));
            }
            b::InstrBody::ArrayIndex(input, idx) => {
                let v = instr.results[0];
                self.add_constraint(*input, Constraint::Array(v));
                let idx_ty = b::Type::new(b::TypeBody::USize, None);
                self.add_constraint(*idx, Constraint::Is(idx_ty));
                self.add_constraint(v, Constraint::ArrayElem(*input));
            }
            b::InstrBody::Type(v, ty) => {
                self.add_constraint(*v, Constraint::Is(ty.clone()));
            }
            b::InstrBody::Dispatch(v, mod_idx, ty_idx) => {
                let ty = b::Type::new(b::TypeRef::new(*mod_idx, *ty_idx).into(), None);
                self.add_constraint(*v, Constraint::Is(ty));
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
        for c in &self.nodes[idx].constraints.clone() {
            match (c, &constraint) {
                (Constraint::Array(a), Constraint::Array(b))
                | (Constraint::Ptr(a), Constraint::Ptr(b))
                | (Constraint::ArrayElem(a), Constraint::ArrayElem(b)) => {
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

        self.nodes[idx].constraints.insert(constraint);
    }

    fn get_contraints_with(
        &self,
        idx: b::ValueIdx,
        f: impl Fn(&Constraint) -> bool,
    ) -> Vec<Constraint> {
        let mut constraints = vec![];
        self.get_contraints_with_and_write(&mut constraints, idx, &f);
        constraints
    }

    fn get_contraints_with_and_write<'w>(
        &self,
        constraints: &'w mut Vec<Constraint>,
        idx: b::ValueIdx,
        f: impl Fn(&Constraint) -> bool + Clone,
    ) {
        let same_of = {
            let value: &b::Value = &self.ctx.lock_modules()[self.mod_idx].values[idx];
            value.same_type_of.clone()
        };
        if same_of.len() > 0 {
            for i in same_of {
                self.get_contraints_with_and_write(constraints, i, f.clone());
            }
        } else {
            for c in &self.nodes[idx].constraints {
                if f(c) {
                    constraints.push(c.clone());
                }
            }
        }
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
                mem::replace(&mut self.nodes[idx].constraints, HashSet::new())
            };

            for constraint in constraints {
                self.add_constraint(head, constraint);
            }
        }
    }

    #[tracing::instrument(skip(self))]
    fn validate(&mut self) {
        tracing::trace!("validate started");

        let len = { self.ctx.lock_modules()[self.mod_idx].values.len() };

        let mut unresolved = HashSet::new();

        for idx in 0..len {
            tracing::trace!(idx, "will validate");

            let status = self.validate_value(idx, &mut HashSet::new());
            self.nodes[idx].status = status;

            if !status.was_checked() {
                unresolved.insert(idx);
            }
        }

        for idx in unresolved {
            tracing::trace!(idx, "will validate again");
            self.nodes[idx].status = self.validate_value(idx, &mut HashSet::new());
        }

        for idx in 0..len {
            if matches!(self.nodes[idx].status, TypeNodeStatus::Unresolved) {
                let value = &self.ctx.lock_modules()[self.mod_idx].values[idx];
                tracing::trace!(idx, ?value.ty, "is not final");
                self.ctx.push_error(errors::Error::new(
                    errors::ErrorDetail::TypeNotFinal,
                    value.loc,
                ));
            }
        }

        tracing::info!("validation completed");
    }

    #[tracing::instrument(level = "trace", skip(self, visited))]
    fn validate_value(
        &mut self,
        idx: b::ValueIdx,
        visited: &mut HashSet<b::ValueIdx>,
    ) -> TypeNodeStatus {
        let initial_status = self.nodes[idx].status;
        if initial_status.was_checked() || visited.contains(&idx) {
            tracing::trace!("already visited");
            return initial_status;
        }

        visited.insert(idx);

        let mut result_ty = self.ctx.lock_modules()[self.mod_idx].values[idx].ty.clone();
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
                success &= !self.validate_value(i, visited).is_failed();
                tys.push(self.ctx.lock_modules()[self.mod_idx].values[i].ty.clone());
            }

            let mut union_success = true;

            result_ty = b::Type::new(b::TypeBody::Never, None);
            for ty in &tys {
                if let Some(ty) = result_ty.union(ty, &self.ctx.lock_modules()) {
                    result_ty = ty;
                } else {
                    union_success = false;
                }
            }

            if !union_success {
                success = false;
                self.ctx.push_error(errors::Error::new(
                    errors::TypeMisatch::new(tys).into(),
                    self.ctx.lock_modules()[self.mod_idx].values[idx].loc,
                ));
            }
        } else {
            let constraints = self.nodes[idx]
                .constraints
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
                        success &= !self.validate_value(target, visited).is_failed();
                        self.ctx.lock_modules()[self.mod_idx].values[target]
                            .ty
                            .clone()
                    }
                    Constraint::Array(target) => {
                        tracing::trace!(target, "will validate Array");
                        success &= !self.validate_value(target, visited).is_failed();
                        let ty = self.ctx.lock_modules()[self.mod_idx].values[target]
                            .ty
                            .clone();
                        b::Type::new(
                            b::TypeBody::Array(b::ArrayType::new(ty.into(), None)),
                            None,
                        )
                    }
                    Constraint::ArrayElem(target) => {
                        tracing::trace!(target, "will validate ArrayElem");
                        success &= !self.validate_value(target, visited).is_failed();
                        if let b::TypeBody::Array(arr_ty) =
                            &self.ctx.lock_modules()[self.mod_idx].values[target].ty.body
                        {
                            (&*arr_ty.item).clone()
                        } else {
                            b::Type::unknown(None)
                        }
                    }
                    Constraint::Ptr(target) => {
                        tracing::trace!(target, "will validate Ptr");
                        success &= !self.validate_value(target, visited).is_failed();
                        let ty = self.ctx.lock_modules()[self.mod_idx].values[target]
                            .ty
                            .clone();
                        b::Type::new(b::TypeBody::Ptr(Some(ty.into())), None)
                    }
                    Constraint::ReturnOf(target) => {
                        tracing::trace!(target, "will validate ReturnOf");
                        success &= !self.validate_value(target, visited).is_failed();
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
                        success &= !self.validate_value(target, visited).is_failed();
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
                        success &= !self.validate_value(target, visited).is_failed();
                        for prop_dep in {
                            let modules = self.ctx.lock_modules();
                            self.get_property_deps(target, &key, &modules)
                        } {
                            tracing::trace!(prop_dep, "will validate property_deps");
                            success &=
                                !self.validate_value(prop_dep, visited).is_failed();
                        }
                        self.get_property_type(target, &key, &self.ctx.lock_modules())
                            .unwrap_or_else(|| b::Type::unknown(None))
                    }
                    Constraint::Members(members) => {
                        for member in members.values() {
                            tracing::trace!(member, "will validate member");
                            success = !self.validate_value(*member, visited).is_failed();
                        }
                        b::Type::new(
                            b::TypeBody::Inferred(b::InferredType {
                                members:    members
                                    .iter()
                                    .map(|(k, v)| {
                                        let value = &self.ctx.lock_modules()
                                            [self.mod_idx]
                                            .values[*v];
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
                        success = !self.validate_value(target, visited).is_failed();
                        let ty = {
                            self.ctx.lock_modules()[self.mod_idx].values[target]
                                .ty
                                .clone()
                        };
                        b::Type::new(
                            b::TypeBody::Inferred(b::InferredType {
                                properties: SortedMap::from([(key.clone(), ty)]),
                                members:    SortedMap::new(),
                            }),
                            None,
                        )
                    }
                    Constraint::GetFunc(mod_idx, func_idx) => {
                        let (params, ret) = {
                            let func = &self.ctx.lock_modules()[mod_idx].funcs[func_idx];
                            (func.params.clone(), func.ret)
                        };

                        if mod_idx == self.mod_idx {
                            for param in &params {
                                tracing::trace!(param, "will validate GetFunc param");
                                success =
                                    !self.validate_value(*param, visited).is_failed();
                            }
                            tracing::trace!(ret, "will validate GetFunc ret");
                            success = !self.validate_value(ret, visited).is_failed();
                        }

                        let (params, ret) = {
                            let module = &self.ctx.lock_modules()[self.mod_idx];
                            let params = params
                                .into_iter()
                                .map(|v| module.values[v].ty.clone())
                                .collect();
                            let ret = module.values[ret].ty.clone();
                            (params, ret)
                        };

                        b::Type::new(
                            b::TypeBody::Func(Box::new(b::FuncType::new(params, ret))),
                            None,
                        )
                    }
                    Constraint::Func(params, ret) => {
                        for param in &params {
                            tracing::trace!(param, "will validate Func param");
                            success = !self.validate_value(*param, visited).is_failed();
                        }
                        tracing::trace!(ret, "will validate Func ret");
                        success = !self.validate_value(ret, visited).is_failed();

                        let (params, ret) = {
                            let module = &self.ctx.lock_modules()[self.mod_idx];
                            let params = params
                                .into_iter()
                                .map(|v| module.values[v].ty.clone())
                                .collect();
                            let ret = module.values[ret].ty.clone();
                            (params, ret)
                        };

                        b::Type::new(
                            b::TypeBody::Func(Box::new(b::FuncType::new(params, ret))),
                            None,
                        )
                    }
                };

                tracing::trace!(?merge_with, "got type");

                let modules = &mut self.ctx.lock_modules_mut();
                if let Some(ty) = result_ty.intersection(&merge_with, modules) {
                    result_ty = ty;
                } else {
                    self.ctx.push_error(errors::Error::new(
                        errors::UnexpectedType::new(
                            vec![modules[self.mod_idx].values[idx].ty.to_owned()],
                            merge_with.clone(),
                        )
                        .into(),
                        modules[self.mod_idx].values[idx].loc,
                    ));
                    tracing::trace!(?result_ty, ?merge_with, "incompatible types");
                }
            }
        }

        let status = if !success {
            tracing::trace!("failed");
            TypeNodeStatus::Failed
        } else if result_ty.body.is_not_final() {
            tracing::trace!("unresolved");
            TypeNodeStatus::Unresolved
        } else {
            tracing::trace!("resolved");
            TypeNodeStatus::Resolved
        };

        self.ctx.lock_modules_mut()[self.mod_idx].values[idx].ty = result_ty;

        status
    }

    fn define_property(
        &mut self,
        src_v: b::ValueIdx,
        prop_v: b::ValueIdx,
        prop_name: &str,
    ) {
        let same_of = {
            self.ctx.lock_modules()[self.mod_idx].values[src_v]
                .same_type_of
                .clone()
        };

        if same_of.len() >= 1 {
            for v in &same_of {
                self.define_property(*v, prop_v, prop_name);
            }
            return;
        }

        for item in &self.nodes[src_v].constraints {
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
        let b::TypeBody::TypeRef(ty_ref) = &parent.body else {
            return vec![];
        };
        let Some(func) = modules.get(ty_ref.mod_idx).and_then(|module| {
            let method = match &module.typedefs.get(ty_ref.idx)?.body {
                b::TypeDefBody::Record(rec) => rec.methods.get(name)?,
                b::TypeDefBody::Interface(iface) => iface.methods.get(name)?,
            };
            if method.func_ref.0 == self.mod_idx {
                module.funcs.get(method.func_ref.1)
            } else {
                None
            }
        }) else {
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
        return parent.property(key, modules).map(|ty| ty.into_owned());
    }

    #[tracing::instrument(level = "trace", skip_all)]
    fn finish_body(
        &self,
        get_body: &impl for<'m> Fn(&'m b::Module) -> &'m Vec<b::Instr>,
        get_body_mut: &impl for<'m> Fn(&'m mut b::Module) -> &'m mut Vec<b::Instr>,
    ) {
        let len = {
            let module = &self.ctx.lock_modules()[self.mod_idx];
            get_body(module).len()
        };
        cfor!(let mut i = 0; i < len; i += 1; {
            self.finish_get_property_instr(i, &get_body, &get_body_mut);
            i += self.finish_dispatch_instr(i, &get_body, &get_body_mut);
        })
    }

    fn finish_get_property_instr(
        &self,
        idx: usize,
        get_body: &impl for<'m> Fn(&'m b::Module) -> &'m Vec<b::Instr>,
        get_body_mut: &impl for<'m> Fn(&'m mut b::Module) -> &'m mut Vec<b::Instr>,
    ) {
        let (source_v, key) = {
            let modules = &self.ctx.lock_modules()[self.mod_idx];
            let instr = &get_body(modules)[idx];
            match &instr.body {
                b::InstrBody::GetProperty(v, key) => (*v, key.clone()),
                _ => return,
            }
        };

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
            let instr = &mut get_body_mut(&mut modules[self.mod_idx])[idx];
            if is_field {
                instr.body = b::InstrBody::GetField(source_v, key.clone());
            } else if is_method {
                instr.body = b::InstrBody::GetMethod(source_v, key.clone());
            }
        }
    }

    fn finish_dispatch_instr(
        &self,
        idx: usize,
        get_body: &impl for<'m> Fn(&'m b::Module) -> &'m Vec<b::Instr>,
        get_body_mut: &impl for<'m> Fn(&'m mut b::Module) -> &'m mut Vec<b::Instr>,
    ) -> usize {
        let (params, loc) = {
            let modules = &self.ctx.lock_modules();
            let instr = &get_body(&modules[self.mod_idx])[idx];
            let params = match &instr.body {
                b::InstrBody::Call(mod_idx, func_idx, args) => {
                    let args_types =
                        args.iter().map(|v| &modules[self.mod_idx].values[*v].ty);
                    let params_types = modules[*mod_idx].funcs[*func_idx]
                        .params
                        .iter()
                        .map(|v| &modules[*mod_idx].values[*v].ty);
                    izip!(args, args_types, params_types).collect_vec()
                }
                b::InstrBody::IndirectCall(v, args) => {
                    let b::TypeBody::Func(func) =
                        &modules[self.mod_idx].values[*v].ty.body
                    else {
                        return 0;
                    };
                    let args_types =
                        args.iter().map(|v| &modules[self.mod_idx].values[*v].ty);
                    izip!(args, args_types, &func.params).collect_vec()
                }
                _ => return 0,
            };
            let params = params
                .into_iter()
                .enumerate()
                .filter_map(|(i, (v, arg_ty, param_ty))| {
                    if arg_ty.body == param_ty.body {
                        return None;
                    }
                    let b::TypeBody::TypeRef(param_ty_ref) = &param_ty.body else {
                        return None;
                    };
                    let param_ty_def =
                        &modules[param_ty_ref.mod_idx].typedefs[param_ty_ref.idx];
                    if matches!(&param_ty_def.body, b::TypeDefBody::Interface(_)) {
                        Some((i, *v, (param_ty_ref.mod_idx, param_ty_ref.idx)))
                    } else {
                        None
                    }
                })
                .collect_vec();
            (params, instr.loc)
        };

        let count = params.len();

        {
            let module = &mut self.ctx.lock_modules_mut()[self.mod_idx];

            let value_start = module.values.len();
            module.values.extend(params.iter().map(|(_, _, iface)| {
                let ty = b::Type::new(b::TypeRef::new(iface.0, iface.1).into(), None);
                b::Value::new(ty, loc)
            }));

            let body = get_body_mut(module);
            match &mut body[idx].body {
                b::InstrBody::Call(_, _, args) | b::InstrBody::IndirectCall(_, args) => {
                    for (i, (n, ..)) in params.iter().enumerate() {
                        args[*n] = value_start + i;
                    }
                }
                _ => {}
            };

            body.splice(
                idx..idx,
                params.into_iter().enumerate().map(|(i, (_, v, iface))| {
                    b::Instr::new(b::InstrBody::Dispatch(v, iface.0, iface.1), loc)
                        .with_results([value_start + i])
                }),
            );
        }

        count
    }
}

#[derive(Debug, Clone, new)]
struct ScopePayload {
    #[new(default)]
    loop_args: Vec<b::ValueIdx>,
}
impl utils::SimpleScopePayload for ScopePayload {}
