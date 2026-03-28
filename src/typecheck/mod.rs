mod constraints;

use std::collections::HashSet;
use std::fmt::Debug;
use std::mem;

use derive_ctor::ctor;
use itertools::{Itertools, enumerate, izip};

use self::constraints::{Constraint, ConstraintKind};
use crate::utils::SortedMap;
use crate::{bytecode as b, context, errors, utils};

#[derive(Debug, Clone, ctor)]
struct TypeNode {
    #[ctor(default)]
    constraints: HashSet<Constraint>,
    #[ctor(default)]
    status:      TypeNodeStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ctor)]
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

#[derive(Debug, Clone, ctor)]
pub struct TypeChecker<'a> {
    ctx:     &'a context::BuildContext,
    mod_idx: usize,
    #[ctor(default)]
    nodes:   Vec<TypeNode>,
}

impl<'a> TypeChecker<'a> {
    #[tracing::instrument(skip(self))]
    pub fn check(&mut self) {
        tracing::trace!("check started");

        // FIXME: use per-module locks instead of locking the entire module list
        let modules = &mut *self.ctx.lock_modules_mut();
        let module = &modules[self.mod_idx];

        self.nodes = module
            .values
            .iter()
            .map(|value| {
                let mut node = TypeNode::new();
                if !value.ty.is_unknown() {
                    node.constraints.insert(Constraint::new(
                        ConstraintKind::Is(value.ty.clone()),
                        value.ty.loc,
                    ));
                }
                node
            })
            .collect();

        let funcs_len = module.funcs.len();
        let globals_len = module.globals.len();

        for i in 0..funcs_len {
            self.add_func(i, modules);
        }

        for i in 0..globals_len {
            tracing::trace!(i, "adding global");
            let global = &modules[self.mod_idx].globals[i];
            let value = global.value;
            let mut cursor = b::BlockCursor::new(global.body);
            let mut scopes = utils::ScopeStack::new(ScopePayload::new());
            if let Some(result) = self.add_block(&mut cursor, &mut scopes, None, modules)
            {
                self.merge_types([&value, &result], modules);
            }
        }

        self.validate(modules);
    }

    #[tracing::instrument(level = "trace", skip(self, modules))]
    fn add_func(
        &mut self,
        idx: usize,
        // FIXME: use per-module locks instead of locking the entire module list
        modules: &mut [b::Module],
    ) {
        let func = &modules[self.mod_idx].funcs[idx];

        if let Some((mod_idx, ty_idx, name)) = &func.method {
            let parent_funcs = modules[*mod_idx].typedefs[*ty_idx]
                .get_ifaces()
                .into_iter()
                .filter_map(|(mod_idx, ty_idx)| {
                    modules[mod_idx].typedefs[ty_idx].get_method(name)
                })
                .map(|f| f.func_ref)
                .collect_vec();
            for (parent_mod_idx, parent_func_idx) in parent_funcs {
                let func = &modules[self.mod_idx].funcs[idx];
                let parent_func = &modules[parent_mod_idx].funcs[parent_func_idx];
                let pairs: Vec<_> = izip!(&func.params, &parent_func.params)
                    .map(|(p, pp)| (*p, *pp))
                    .collect();
                let parent_mod_idx_is_mod_idx =
                    func.method.as_ref().unwrap().0 == parent_mod_idx;
                for (param, parent_param) in pairs {
                    if parent_mod_idx_is_mod_idx {
                        self.add_constraint(
                            param,
                            Constraint::new(
                                ConstraintKind::TypeOf(parent_param),
                                modules[parent_mod_idx].funcs[parent_func_idx].loc,
                            ),
                            modules,
                        );
                    } else {
                        let v = &modules[parent_mod_idx].values[parent_param];
                        self.add_constraint(
                            param,
                            Constraint::new(
                                ConstraintKind::Is(v.ty.clone()),
                                modules[parent_mod_idx].funcs[parent_func_idx].loc,
                            ),
                            modules,
                        );
                    }
                }
            }
        }

        let func = &modules[self.mod_idx].funcs[idx];
        let ret = func.ret;

        let mut cursor = b::BlockCursor::new(func.body);
        let mut scopes = utils::ScopeStack::new(ScopePayload::new());
        if let Some(result) = self.add_block(&mut cursor, &mut scopes, Some(idx), modules)
        {
            self.merge_types([&ret, &result], modules);
        }
    }

    #[tracing::instrument(level = "trace", skip(self, scopes, modules))]
    fn add_block(
        &mut self,
        cursor: &mut b::BlockCursor,
        scopes: &mut utils::ScopeStack<ScopePayload>,
        func_idx: Option<usize>,
        // FIXME: use per-module locks instead of locking the entire module list
        modules: &mut [b::Module],
    ) -> Option<b::ValueIdx> {
        let mut result = None;
        while cursor.step_over(&modules[self.mod_idx]) {
            if let b::InstrBody::Break(v) =
                &cursor.instr(&modules[self.mod_idx]).unwrap().body
            {
                result = result.or(*v);
                continue;
            }
            self.add_instr(cursor, scopes, func_idx, modules);
        }

        return result;
    }

    #[tracing::instrument(level = "trace", skip(self, scopes, modules))]
    fn add_instr(
        &mut self,
        cursor: &mut b::BlockCursor,
        scopes: &mut utils::ScopeStack<ScopePayload>,
        func_idx: Option<usize>,
        // FIXME: use per-module locks instead of locking the entire module list
        modules: &mut [b::Module],
    ) {
        let instr = cursor.instr(&modules[self.mod_idx]).unwrap();
        let loc = instr.loc;

        tracing::trace!(?instr, "add instr");

        match &instr.body {
            b::InstrBody::GetGlobal(mod_idx, idx) => {
                let v = instr.results[0];
                if *mod_idx == self.mod_idx {
                    let gv = modules[*mod_idx].globals[*idx].value;
                    self.merge_types([&gv, &v], modules);
                } else {
                    let module = &modules[*mod_idx];
                    let gv = module.globals[*idx].value;
                    let ty = module.values[gv].ty.clone();
                    self.add_constraint(
                        v,
                        Constraint::new(ConstraintKind::Is(ty), loc),
                        modules,
                    );
                };
            }
            &b::InstrBody::GetFunc(mod_idx, func_idx) => {
                let v = instr.results[0];
                self.add_constraint(
                    v,
                    Constraint::new(ConstraintKind::GetFunc(mod_idx, func_idx), loc),
                    modules,
                );
            }
            b::InstrBody::GetProperty(source_v, name)
            | b::InstrBody::GetField(source_v, name)
            | b::InstrBody::GetMethod(source_v, name) => {
                let v = instr.results[0];
                self.define_property(*source_v, v, &name.clone(), loc, modules);
            }
            b::InstrBody::CreateBool(_) => {
                let v = instr.results[0];
                let ty = b::Type::new(b::TypeBody::Bool, None);
                self.add_constraint(
                    v,
                    Constraint::new(ConstraintKind::Is(ty), loc),
                    modules,
                );
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
                self.add_constraint(
                    v,
                    Constraint::new(ConstraintKind::Is(b::Type::new(ty_body, None)), loc),
                    modules,
                );
            }
            b::InstrBody::CreateString(_) => {
                let v = instr.results[0];
                let ty = b::Type::new(b::TypeBody::String, None);
                self.add_constraint(
                    v,
                    Constraint::new(ConstraintKind::Is(ty.clone()), loc),
                    modules,
                );
            }
            &b::InstrBody::CreateUninitializedString(len_v) => {
                let v = instr.results[0];
                let ty = b::Type::new(b::TypeBody::String, None);
                self.add_constraint(
                    v,
                    Constraint::new(ConstraintKind::Is(ty.clone()), loc),
                    modules,
                );

                let len_ty = b::Type::new(b::TypeBody::AnyNumber, None);
                self.add_constraint(
                    len_v,
                    Constraint::new(ConstraintKind::Is(len_ty), loc),
                    modules,
                );
            }
            b::InstrBody::CreateArray(vs) => {
                let v = instr.results[0];
                if vs.len() > 0 {
                    let vs = vs.clone();
                    self.merge_types(&vs, modules);
                    self.add_constraint(
                        v,
                        Constraint::new(ConstraintKind::Array(vs[0]), loc),
                        modules,
                    )
                } else {
                    let item_ty = b::Type::new(b::TypeBody::Never, None);
                    let arr_ty = b::Type::new(b::TypeBody::Array(item_ty.into()), None);
                    self.add_constraint(
                        v,
                        Constraint::new(ConstraintKind::Is(arr_ty), loc),
                        modules,
                    );
                }
            }
            b::InstrBody::CreateRecord(fields) => {
                let v = instr.results[0];
                let fields = fields.clone();
                self.add_constraint(
                    v,
                    Constraint::new(
                        ConstraintKind::Members(
                            fields.iter().map(|(k, v)| (k.clone(), *v)).collect(),
                        ),
                        loc,
                    ),
                    modules,
                );
                for (name, fields_v) in fields {
                    self.define_property(v, fields_v, &name, loc, modules);
                }
            }
            &b::InstrBody::Add(a, b)
            | &b::InstrBody::Sub(a, b)
            | &b::InstrBody::Mul(a, b)
            | &b::InstrBody::Div(a, b)
            | &b::InstrBody::Mod(a, b) => {
                let v = instr.results[0];
                self.merge_types([&a, &b, &v], modules);
                // FIXME: use interface/trait
                let ty = b::Type::new(b::TypeBody::AnyNumber, None);
                self.add_constraint(
                    a,
                    Constraint::new(ConstraintKind::Is(ty), loc),
                    modules,
                );
            }
            &b::InstrBody::Not(x) => {
                let v = instr.results[0];
                self.merge_types([&x, &v], modules);
                // FIXME: use interface/trait
                self.add_constraint(
                    x,
                    Constraint::new(
                        ConstraintKind::Is(b::Type::new(b::TypeBody::Bool, None)),
                        loc,
                    ),
                    modules,
                );
            }
            &b::InstrBody::Eq(a, b)
            | &b::InstrBody::Neq(a, b)
            | &b::InstrBody::Gt(a, b)
            | &b::InstrBody::Gte(a, b)
            | &b::InstrBody::Lt(a, b)
            | &b::InstrBody::Lte(a, b) => {
                let v = instr.results[0];
                self.merge_types([&a, &b], modules);
                // FIXME: use interface/trait
                let number_ty = b::Type::new(b::TypeBody::AnyNumber, None);
                let bool_ty = b::Type::new(b::TypeBody::Bool, None);
                self.add_constraint(
                    a,
                    Constraint::new(ConstraintKind::Is(number_ty), loc),
                    modules,
                );

                self.add_constraint(
                    v,
                    Constraint::new(ConstraintKind::Is(bool_ty), loc),
                    modules,
                );
            }
            &b::InstrBody::Call(call_mod_idx, idx, ref args) => {
                let v = instr.results[0];
                let func = &modules[call_mod_idx].funcs[idx];
                let args = args.clone();
                let params = func.params.clone();
                let ret = func.ret;

                if call_mod_idx == self.mod_idx && func_idx.is_some_and(|i| i == idx) {
                    self.merge_types([&ret, &v], modules);
                    for (arg, param) in izip!(&args, &params) {
                        self.merge_types([param, arg], modules);
                    }
                    return;
                }

                for (arg, param) in izip!(args, params) {
                    let param_ty = &modules[call_mod_idx].values[param].ty;
                    let kind =
                        if call_mod_idx == self.mod_idx && !param_ty.contains_typevar() {
                            ConstraintKind::TypeOf(param)
                        } else {
                            ConstraintKind::Is(param_ty.clone())
                        };
                    self.add_constraint(arg, Constraint::new(kind, loc), modules);
                }

                let ret_ty = &modules[call_mod_idx].values[ret].ty;
                let kind = if call_mod_idx == self.mod_idx && !ret_ty.contains_typevar() {
                    ConstraintKind::TypeOf(ret)
                } else {
                    ConstraintKind::Is(ret_ty.clone())
                };
                self.add_constraint(v, Constraint::new(kind, loc), modules);
            }
            &b::InstrBody::IndirectCall(func, ref args) => {
                let v = instr.results[0];
                let args = args.clone();

                let mut has_get_func = false;
                for c in self.get_contraints_with(func, modules, |c| {
                    matches!(c.kind, ConstraintKind::GetFunc(..))
                }) {
                    let ConstraintKind::GetFunc(mod_idx, func_idx) = c.kind else {
                        continue;
                    };
                    has_get_func = true;

                    let func = &modules[mod_idx].funcs[func_idx];
                    let params = func.params.clone();
                    let ret = func.ret;

                    for (param, arg) in izip!(&params, &args) {
                        self.add_constraint(
                            *arg,
                            Constraint::new(ConstraintKind::TypeOf(*param), loc),
                            modules,
                        );
                    }
                    self.add_constraint(
                        v,
                        Constraint::new(ConstraintKind::TypeOf(ret), loc),
                        modules,
                    );
                }
                if !has_get_func {
                    self.add_constraint(
                        func,
                        Constraint::new(ConstraintKind::Func(args.clone(), v), loc),
                        modules,
                    );
                }

                for (i, arg) in enumerate(args) {
                    self.add_constraint(
                        arg,
                        Constraint::new(ConstraintKind::ParameterOf(func, i), loc),
                        modules,
                    );
                }
                self.add_constraint(
                    v,
                    Constraint::new(ConstraintKind::ReturnOf(func), loc),
                    modules,
                );
            }
            &b::InstrBody::If(cond_v, _then_block, _else_block) => {
                let v = instr.results[0];

                self.add_constraint(
                    cond_v,
                    Constraint::new(
                        ConstraintKind::Is(b::Type::new(b::TypeBody::Bool, None)),
                        loc,
                    ),
                    modules,
                );

                scopes.begin(ScopePayload::new());
                cursor.step_in(&modules[self.mod_idx]);
                if let Some(then_v) = self.add_block(cursor, scopes, func_idx, modules) {
                    self.merge_types([&then_v, &v], modules);
                }

                scopes.branch();
                cursor.step_out(&modules[self.mod_idx]);
                if let Some(else_v) = self.add_block(cursor, scopes, func_idx, modules) {
                    self.merge_types([&else_v, &v], modules);
                }

                scopes.end();
                cursor.step_out(&modules[self.mod_idx]);
            }
            &b::InstrBody::Loop(ref inputs, _body_block) => {
                let v = instr.results[0];

                let scope = scopes.begin(ScopePayload::new());
                scope.is_loop = true;
                for (loop_v, initial_v) in inputs.clone() {
                    self.merge_types([&initial_v, &loop_v], modules);
                    scope.loop_args.push(loop_v);
                }

                cursor.step_in(&modules[self.mod_idx]);
                if let Some(result) = self.add_block(cursor, scopes, func_idx, modules) {
                    self.merge_types([&result, &v], modules);
                }

                scopes.end();
                cursor.step_out(&modules[self.mod_idx]);
            }
            b::InstrBody::Continue(vs) => {
                let loop_args = &scopes
                    .last_loop()
                    .expect("continue should be called inside a loop")
                    .loop_args;
                for (v, loop_v) in izip!(vs.clone(), loop_args) {
                    if v != *loop_v {
                        self.merge_types([&v, loop_v], modules);
                    }
                }
            }
            b::InstrBody::StrLen(input) => {
                let v = instr.results[0];
                let str_ty = b::Type::new(b::TypeBody::String, None);
                self.add_constraint(
                    *input,
                    Constraint::new(ConstraintKind::Is(str_ty), loc),
                    modules,
                );
                let ty = b::Type::new(b::TypeBody::USize, None);
                self.add_constraint(
                    v,
                    Constraint::new(ConstraintKind::Is(ty), loc),
                    modules,
                );
            }
            b::InstrBody::StrPtr(input) => {
                let v = instr.results[0];
                let str_ty = b::Type::new(b::TypeBody::String, None);
                self.add_constraint(
                    *input,
                    Constraint::new(ConstraintKind::Is(str_ty), loc),
                    modules,
                );
                let ty = b::Type::new(
                    b::TypeBody::Ptr(Some(b::Type::new(b::TypeBody::U8, None).into())),
                    None,
                );
                self.add_constraint(
                    v,
                    Constraint::new(ConstraintKind::Is(ty), loc),
                    modules,
                );
            }
            &b::InstrBody::StrFromPtr(ptr_v, len_v) => {
                let v = instr.results[0];

                let ptr_ty = b::Type::new(
                    b::TypeBody::Ptr(Some(b::Type::new(b::TypeBody::U8, None).into())),
                    None,
                );
                self.add_constraint(
                    ptr_v,
                    Constraint::new(ConstraintKind::Is(ptr_ty), loc),
                    modules,
                );

                let len_ty = b::Type::new(b::TypeBody::USize, None);
                self.add_constraint(
                    len_v,
                    Constraint::new(ConstraintKind::Is(len_ty), loc),
                    modules,
                );

                let ty = b::Type::new(b::TypeBody::String, None);
                self.add_constraint(
                    v,
                    Constraint::new(ConstraintKind::Is(ty), loc),
                    modules,
                );
            }
            &b::InstrBody::StrCopy(src_v, dst_v, offset_v) => {
                let str_ty = b::Type::new(b::TypeBody::String, None);
                self.add_constraint(
                    src_v,
                    Constraint::new(ConstraintKind::Is(str_ty.clone()), loc),
                    modules,
                );
                self.add_constraint(
                    dst_v,
                    Constraint::new(ConstraintKind::Is(str_ty), loc),
                    modules,
                );

                if let Some(offset_v) = offset_v {
                    let offset_ty = b::Type::new(b::TypeBody::USize, None);
                    self.add_constraint(
                        offset_v,
                        Constraint::new(ConstraintKind::Is(offset_ty), loc),
                        modules,
                    );
                }
            }
            b::InstrBody::ArrayLen(input) => {
                let v = instr.results[0];
                let arr_ty =
                    b::Type::new(b::TypeBody::Array(b::Type::unknown(None).into()), None);
                self.add_constraint(
                    *input,
                    Constraint::new(ConstraintKind::Is(arr_ty), loc),
                    modules,
                );
                let ty = b::Type::new(b::TypeBody::USize, None);
                self.add_constraint(
                    v,
                    Constraint::new(ConstraintKind::Is(ty), loc),
                    modules,
                );
            }
            &b::InstrBody::ArrayIndex(input, idx) => {
                let v = instr.results[0];
                self.add_constraint(
                    input,
                    Constraint::new(ConstraintKind::Array(v), loc),
                    modules,
                );
                let idx_ty = b::Type::new(b::TypeBody::USize, None);
                self.add_constraint(
                    idx,
                    Constraint::new(ConstraintKind::Is(idx_ty), loc),
                    modules,
                );
                self.add_constraint(
                    v,
                    Constraint::new(ConstraintKind::ArrayElem(input), loc),
                    modules,
                );
            }
            &b::InstrBody::PtrOffset(ptr, offset) => {
                let v = instr.results[0];
                self.merge_types([&v, &ptr], modules);

                let ptr_ty = b::Type::new(b::TypeBody::Ptr(None), None);
                self.add_constraint(
                    ptr,
                    Constraint::new(ConstraintKind::Is(ptr_ty), loc),
                    modules,
                );

                let offset_ty = b::Type::new(b::TypeBody::USize, None);
                self.add_constraint(
                    offset,
                    Constraint::new(ConstraintKind::Is(offset_ty), loc),
                    modules,
                );
            }
            &b::InstrBody::PtrSet(ptr, value) => {
                self.add_constraint(
                    ptr,
                    Constraint::new(ConstraintKind::Ptr(value), loc),
                    modules,
                );

                self.add_constraint(
                    value,
                    Constraint::new(ConstraintKind::Deref(ptr), loc),
                    modules,
                );
            }
            b::InstrBody::Type(v, ty) => {
                self.add_constraint(
                    *v,
                    Constraint::new(ConstraintKind::Is(ty.clone()), loc),
                    modules,
                );
            }
            b::InstrBody::Dispatch(v, mod_idx, ty_idx) => {
                let ty = b::Type::new(b::TypeRef::new(*mod_idx, *ty_idx).into(), None);
                self.add_constraint(
                    *v,
                    Constraint::new(ConstraintKind::Is(ty), loc),
                    modules,
                );
            }
            b::InstrBody::TypeName(_) => {
                let v = instr.results[0];
                let ty = b::Type::new(b::TypeBody::String, None);
                self.add_constraint(
                    v,
                    Constraint::new(ConstraintKind::Is(ty.clone()), loc),
                    modules,
                );
            }
            b::InstrBody::Break(_) | b::InstrBody::CompileError => {}
        }
    }

    #[tracing::instrument(skip(self, modules))]
    fn add_constraint(
        &mut self,
        idx: b::ValueIdx,
        constraint: Constraint,
        // FIXME: use per-module locks instead of locking the entire module list
        modules: &mut [b::Module],
    ) {
        tracing::trace!("add constraint");

        let value = &modules[self.mod_idx].values[idx];
        let same_of = if let Some(redirects_to) = &value.redirects_to {
            vec![*redirects_to]
        } else if value.same_type_of.len() > 0 {
            value.same_type_of.iter().copied().collect()
        } else {
            vec![]
        };

        if same_of.len() > 0 {
            for idx in same_of {
                self.add_constraint(idx, constraint.clone(), modules);
            }
            return;
        }

        // Some constraints cannot be repeated, and instead indicates that two values have
        // the same type. In these cases, e merge the values types
        for c in &self.nodes[idx].constraints.clone() {
            match (&c.kind, &constraint.kind) {
                (ConstraintKind::Array(a), ConstraintKind::Array(b))
                | (ConstraintKind::Ptr(a), ConstraintKind::Ptr(b))
                | (ConstraintKind::ArrayElem(a), ConstraintKind::ArrayElem(b)) => {
                    self.merge_types([a, b], modules);
                }
                (
                    ConstraintKind::HasProperty(name_a, a),
                    ConstraintKind::HasProperty(name_b, b),
                )
                | (
                    ConstraintKind::IsProperty(a, name_a),
                    ConstraintKind::IsProperty(b, name_b),
                ) if name_a == name_b => {
                    self.merge_types([a, b], modules);
                }
                _ => {}
            }
        }

        self.nodes[idx].constraints.insert(constraint);
    }

    fn get_contraints_with(
        &self,
        idx: b::ValueIdx,
        modules: &[b::Module],
        f: impl Fn(&Constraint) -> bool,
    ) -> Vec<Constraint> {
        let mut constraints = vec![];
        self.write_constraints_with(&mut constraints, idx, modules, &f);
        constraints
    }

    fn write_constraints_with(
        &self,
        target: &mut Vec<Constraint>,
        idx: b::ValueIdx,
        modules: &[b::Module],
        f: &impl Fn(&Constraint) -> bool,
    ) {
        let same_of = &modules[self.mod_idx].values[idx].same_type_of;
        if same_of.len() > 0 {
            for i in same_of.clone() {
                self.write_constraints_with(target, i, modules, f);
            }
        } else {
            for c in &self.nodes[idx].constraints {
                if f(c) {
                    target.push(c.clone());
                }
            }
        }
    }

    #[tracing::instrument(level = "trace", skip(self, modules))]
    fn merge_types<'i, I>(
        &mut self,
        items: I,
        // FIXME: use per-module locks instead of locking the entire module list
        modules: &mut [b::Module],
    ) where
        I: IntoIterator<Item = &'i b::ValueIdx>,
        I: Debug,
    {
        tracing::trace!("merge types");

        let mut merge_with = items
            .into_iter()
            .unique()
            .sorted_by(|a, b| a.cmp(b).reverse())
            .copied()
            .collect_vec();

        if merge_with.len() <= 1 {
            return;
        }

        let head = merge_with.pop().unwrap();

        while let Some(v) = merge_with.pop() {
            modules[self.mod_idx].values[v].same_type_of.insert(head);
            let constraints =
                mem::replace(&mut self.nodes[v].constraints, HashSet::new());

            for constraint in constraints {
                self.add_constraint(head, constraint, modules);
            }
        }
    }

    #[tracing::instrument(skip(self, modules))]
    fn validate(
        &mut self,
        modules:
        // FIXME: use per-module locks instead of locking the entire module list
        &mut [b::Module],
    ) {
        tracing::trace!("validate started");

        let len = modules[self.mod_idx].values.len();

        let mut unresolved = HashSet::new();

        for idx in 0..len {
            tracing::trace!(idx, "will validate");

            let status = self.validate_value(idx, &mut HashSet::new(), modules);
            self.nodes[idx].status = status;

            if !status.was_checked() {
                unresolved.insert(idx);
            }
        }

        for idx in unresolved {
            tracing::trace!(idx, "will validate again");
            self.nodes[idx].status =
                self.validate_value(idx, &mut HashSet::new(), modules);
        }

        for idx in 0..len {
            if matches!(self.nodes[idx].status, TypeNodeStatus::Unresolved) {
                let value = &modules[self.mod_idx].values[idx];
                tracing::trace!(idx, ?value.ty, "is not final");
                self.ctx.push_error(errors::Error::new(
                    errors::ErrorDetail::TypeNotFinal,
                    value.loc,
                ));
            }
        }

        tracing::info!("validation completed");
    }

    #[tracing::instrument(level = "trace", skip(self, visited, modules))]
    fn validate_value(
        &mut self,
        idx: b::ValueIdx,
        visited: &mut HashSet<b::ValueIdx>,
        // FIXME: use per-module locks instead of locking the entire module list
        modules: &mut [b::Module],
    ) -> TypeNodeStatus {
        let initial_status = self.nodes[idx].status;
        if initial_status.was_checked() || visited.contains(&idx) {
            tracing::trace!("already visited");
            return initial_status;
        }

        visited.insert(idx);

        let mut result_ty = modules[self.mod_idx].values[idx].ty.clone();
        let mut success = true;

        let same_of = modules[self.mod_idx].values[idx].same_type_of.clone();

        if same_of.len() > 0 {
            let mut tys = vec![];
            for i in same_of {
                tracing::trace!(i, "will validate same_type_of");
                success &= !self.validate_value(i, visited, modules).is_failed();
                tys.push(modules[self.mod_idx].values[i].ty.clone());
            }

            let mut union_success = true;

            result_ty = b::Type::new(b::TypeBody::Never, None);
            for ty in &tys {
                if let Some(ty) = result_ty.union(ty, modules) {
                    result_ty = ty;
                } else {
                    union_success = false;
                }
            }

            if !union_success {
                success = false;
                self.ctx.push_error(errors::Error::new(
                    errors::TypeMisatch::new(
                        tys.iter().collect(),
                        &modules,
                        &self.ctx.cfg,
                    )
                    .into(),
                    modules[self.mod_idx].values[idx].loc,
                ));
            }
        } else {
            let error_tys = self.nodes[idx]
                .constraints
                .iter()
                .cloned()
                .sorted_by(|a, b| b.priority().cmp(&a.priority()))
                .filter_map(|c| {
                    tracing::trace!(?c, "checking constraint");
                    let merge_with = match c.kind {
                        ConstraintKind::Is(ty) => ty.clone(),
                        ConstraintKind::TypeOf(target) => {
                            tracing::trace!(target, "will validate TypeOf");
                            success &= !self
                                .validate_value(target, visited, modules)
                                .is_failed();
                            modules[self.mod_idx].values[target].ty.clone()
                        }
                        ConstraintKind::Array(target) => {
                            tracing::trace!(target, "will validate Array");
                            success &= !self
                                .validate_value(target, visited, modules)
                                .is_failed();
                            let ty = modules[self.mod_idx].values[target].ty.clone();
                            b::Type::new(b::TypeBody::Array(ty.into()), None)
                        }
                        ConstraintKind::ArrayElem(target) => {
                            tracing::trace!(target, "will validate ArrayElem");
                            success &= !self
                                .validate_value(target, visited, modules)
                                .is_failed();
                            if let b::TypeBody::Array(item_ty) =
                                &modules[self.mod_idx].values[target].ty.body
                            {
                                item_ty.as_ref().clone()
                            } else {
                                b::Type::unknown(None)
                            }
                        }
                        ConstraintKind::Ptr(target) => {
                            tracing::trace!(target, "will validate Ptr");
                            success &= !self
                                .validate_value(target, visited, modules)
                                .is_failed();
                            let ty = modules[self.mod_idx].values[target].ty.clone();
                            b::Type::new(b::TypeBody::Ptr(Some(ty.into())), None)
                        }
                        ConstraintKind::Deref(target) => {
                            tracing::trace!(target, "will validate Deref");
                            success &= !self
                                .validate_value(target, visited, modules)
                                .is_failed();
                            let ty = modules[self.mod_idx].values[target].ty.clone();
                            match &ty.body {
                                b::TypeBody::Ptr(Some(ty)) => ty.as_ref().clone(),
                                _ => b::Type::unknown(None),
                            }
                        }
                        ConstraintKind::ReturnOf(target) => {
                            tracing::trace!(target, "will validate ReturnOf");
                            success &= !self
                                .validate_value(target, visited, modules)
                                .is_failed();
                            if let b::TypeBody::Func(func_ty) =
                                &modules[self.mod_idx].values[target].ty.body
                            {
                                func_ty.ret.clone()
                            } else {
                                b::Type::unknown(None)
                            }
                        }
                        ConstraintKind::ParameterOf(target, idx) => {
                            tracing::trace!(target, idx, "will validate ParameterOf");
                            success &= !self
                                .validate_value(target, visited, modules)
                                .is_failed();
                            if let b::TypeBody::Func(func_ty) =
                                &modules[self.mod_idx].values[target].ty.body
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
                        ConstraintKind::IsProperty(target, key) => {
                            tracing::trace!(target, key, "will validate IsProperty");
                            success &= !self
                                .validate_value(target, visited, modules)
                                .is_failed();
                            for prop_dep in self.get_property_deps(target, &key, modules)
                            {
                                tracing::trace!(prop_dep, "will validate property_deps");
                                success &= !self
                                    .validate_value(prop_dep, visited, modules)
                                    .is_failed();
                            }
                            self.get_property_type(target, &key, modules)
                                .unwrap_or_else(|| b::Type::unknown(None))
                        }
                        ConstraintKind::Members(members) => {
                            for member in members.values() {
                                tracing::trace!(member, "will validate member");
                                success &= !self
                                    .validate_value(*member, visited, modules)
                                    .is_failed();
                            }
                            b::Type::new(
                                b::TypeBody::Inferred(b::InferredType {
                                    members:    members
                                        .iter()
                                        .map(|(k, v)| {
                                            let value = &modules[self.mod_idx].values[*v];
                                            (k.clone(), value.ty.clone())
                                        })
                                        .collect(),
                                    properties: SortedMap::new(),
                                }),
                                None,
                            )
                        }
                        ConstraintKind::HasProperty(key, target) => {
                            tracing::trace!(key, target, "will validate HasProperty");
                            success &= !self
                                .validate_value(target, visited, modules)
                                .is_failed();
                            let ty = modules[self.mod_idx].values[target].ty.clone();
                            b::Type::new(
                                b::TypeBody::Inferred(b::InferredType {
                                    properties: SortedMap::from([(key.clone(), ty)]),
                                    members:    SortedMap::new(),
                                }),
                                None,
                            )
                        }
                        ConstraintKind::GetFunc(mod_idx, func_idx) => {
                            let func = &modules[mod_idx].funcs[func_idx];
                            let params = func.params.clone();
                            let ret = func.ret;

                            if mod_idx == self.mod_idx {
                                for param in &params {
                                    tracing::trace!(param, "will validate GetFunc param");
                                    success &= !self
                                        .validate_value(*param, visited, modules)
                                        .is_failed();
                                }
                                tracing::trace!(ret, "will validate GetFunc ret");
                                success &= !self
                                    .validate_value(ret, visited, modules)
                                    .is_failed();
                            }

                            let module = &modules[self.mod_idx];
                            let params = params
                                .into_iter()
                                .map(|v| module.values[v].ty.clone())
                                .collect();
                            let ret = module.values[ret].ty.clone();

                            b::Type::new(
                                b::TypeBody::Func(Box::new(b::FuncType::new(
                                    params, ret,
                                ))),
                                None,
                            )
                        }
                        ConstraintKind::Func(params, ret) => {
                            for param in &params {
                                tracing::trace!(param, "will validate Func param");
                                success &= !self
                                    .validate_value(*param, visited, modules)
                                    .is_failed();
                            }
                            tracing::trace!(ret, "will validate Func ret");
                            success &=
                                !self.validate_value(ret, visited, modules).is_failed();

                            let module = &modules[self.mod_idx];
                            let params = params
                                .into_iter()
                                .map(|v| module.values[v].ty.clone())
                                .collect();
                            let ret = module.values[ret].ty.clone();

                            b::Type::new(
                                b::TypeBody::Func(Box::new(b::FuncType::new(
                                    params, ret,
                                ))),
                                None,
                            )
                        }
                    };

                    tracing::trace!(?merge_with, "got type");

                    if let Some(ty) = result_ty.intersection(&merge_with, modules) {
                        result_ty = ty;
                        None
                    } else {
                        tracing::trace!(?result_ty, ?merge_with, "incompatible types");
                        Some((
                            merge_with,
                            c.loc.or(modules[self.mod_idx].values[idx].loc),
                        ))
                    }
                })
                .collect_vec();
            for (merge_with, loc) in &error_tys {
                self.ctx.push_error(errors::Error::new(
                    errors::UnexpectedType::new(
                        vec![merge_with],
                        &result_ty,
                        &modules,
                        &self.ctx.cfg,
                    )
                    .into(),
                    *loc,
                ));
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

        modules[self.mod_idx].values[idx].ty = result_ty;

        status
    }

    fn define_property(
        &mut self,
        src_v: b::ValueIdx,
        prop_v: b::ValueIdx,
        prop_name: &str,
        loc: Option<b::Loc>,
        // FIXME: use per-module locks instead of locking the entire module list
        modules: &mut [b::Module],
    ) {
        let same_of: Vec<_> = modules[self.mod_idx].values[src_v]
            .same_type_of
            .iter()
            .copied()
            .collect();

        if same_of.len() >= 1 {
            for v in same_of {
                self.define_property(v, prop_v, prop_name, loc, modules);
            }
            return;
        }

        for item in &self.nodes[src_v].constraints {
            if let ConstraintKind::HasProperty(prop_name_, prop_v_) = &item.kind {
                if prop_name == prop_name_ {
                    self.merge_types(&[*prop_v_, prop_v], modules);
                    return;
                }
            }
        }

        self.add_constraint(
            src_v,
            Constraint::new(
                ConstraintKind::HasProperty(prop_name.to_string(), prop_v),
                loc,
            ),
            modules,
        );
        self.add_constraint(
            prop_v,
            Constraint::new(
                ConstraintKind::IsProperty(src_v, prop_name.to_string()),
                loc,
            ),
            modules,
        );
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
}

#[derive(Debug, Clone, ctor)]
struct ScopePayload {
    #[ctor(default)]
    loop_args: Vec<b::ValueIdx>,
}
impl utils::SimpleScopePayload for ScopePayload {}
