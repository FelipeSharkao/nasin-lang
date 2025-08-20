use std::borrow::Cow;
use std::collections::HashMap;

use cl::InstBuilder;
use cranelift_shim::{self as cl, Module};
use derive_new::new;
use itertools::{izip, Itertools};

use super::context::CodegenContext;
use super::types::{self, ResultPolicy, ReturnPolicy};
use crate::utils::unwrap;
use crate::{bytecode as b, utils};

#[derive(Debug, Clone, Copy)]
pub enum Callee {
    Direct(cl::FuncId),
    Indirect(cl::SigRef, cl::Value),
}

#[derive(new)]
pub struct FuncCodegen<'a, 'b> {
    pub ctx: CodegenContext<'a>,
    pub builder: Option<cl::FunctionBuilder<'b>>,
    #[new(value = "utils::ScopeStack::empty()")]
    pub scopes: utils::ScopeStack<ScopePayload<'a>>,
    #[new(default)]
    pub values: HashMap<(usize, b::ValueIdx), types::RuntimeValue>,
    #[new(default)]
    imported_signatures: HashMap<cl::Signature, cl::SigRef>,
    #[new(default)]
    declared_funcs: HashMap<cl::FuncId, cl::FuncRef>,
}
macro_rules! expect_builder {
    ($self:expr) => {{
        ($self)
            .builder
            .as_mut()
            .expect("function builder should be defined")
    }};
}
impl<'a> FuncCodegen<'a, '_> {
    pub fn create_initial_block(
        &mut self,
        params: &'a [b::ValueIdx],
        result: Option<b::ValueIdx>,
        result_policy: ResultPolicy,
        mod_idx: usize,
    ) {
        let (block, mut cl_values) = {
            let func = expect_builder!(self);
            let block = func.create_block();
            func.append_block_params_for_function_params(block);
            (block, func.block_params(block).to_vec())
        };

        if let (ResultPolicy::Return(types::ReturnPolicy::Struct(_)), Some(result)) =
            (result_policy, result)
        {
            let cl_value = cl_values.remove(0);
            let runtime_value = types::RuntimeValue::new(
                types::ValueSource::Ptr(cl_value),
                mod_idx,
                result,
            );
            self.values.insert((mod_idx, result), runtime_value);
        }

        self.values.extend(izip!(
            params.iter().map(|v| (mod_idx, *v)),
            types::tuple_from_args(
                mod_idx,
                params,
                &cl_values,
                &self.ctx.modules,
                &self.ctx.cl_module
            ),
        ));

        expect_builder!(self).switch_to_block(block);
        self.scopes.begin(ScopePayload {
            start_block: block,
            block,
            next_branches: vec![],
            ty: result
                .clone()
                .map(|v| Cow::Borrowed(&self.ctx.modules[mod_idx].values[v].ty)),
            result,
        });
    }
    pub fn finish(self) -> CodegenContext<'a> {
        assert!(self.scopes.len() == 1);
        self.ctx
    }
    #[tracing::instrument(skip_all)]
    pub fn add_body(
        &mut self,
        body: impl IntoIterator<Item = &'a b::Instr>,
        mod_idx: usize,
        result_policy: ResultPolicy,
    ) {
        for instr in body {
            self.add_instr(instr, mod_idx, result_policy);
            if self.scopes.last().is_never()
                || matches!(&instr.body, b::InstrBody::Break(..))
            {
                break;
            }
        }
    }
    #[tracing::instrument(skip(self))]
    pub fn add_instr(
        &mut self,
        instr: &'a b::Instr,
        mod_idx: usize,
        result_policy: ResultPolicy,
    ) {
        if self.scopes.last().is_never() {
            return;
        }

        if self.value_from_instr(instr, mod_idx).is_some() {
            return;
        }

        match &instr.body {
            b::InstrBody::Add(a, b)
            | b::InstrBody::Sub(a, b)
            | b::InstrBody::Mul(a, b)
            | b::InstrBody::Div(a, b)
            | b::InstrBody::Mod(a, b)
            | b::InstrBody::Eq(a, b)
            | b::InstrBody::Neq(a, b)
            | b::InstrBody::Gt(a, b)
            | b::InstrBody::Lt(a, b)
            | b::InstrBody::Gte(a, b)
            | b::InstrBody::Lte(a, b) => {
                let lhs = self.use_value_by_value(mod_idx, *a);
                assert!(lhs.len() == 1);
                let lhs = lhs[0];

                let rhs = self.use_value_by_value(mod_idx, *b);
                assert!(rhs.len() == 1);
                let rhs = rhs[0];

                let ty = &self.ctx.modules[mod_idx].values[*a].ty;

                let builder = expect_builder!(self);

                let cl_value = match &instr.body {
                    b::InstrBody::Add(..) if ty.is_int() => builder.ins().iadd(lhs, rhs),
                    b::InstrBody::Add(..) if ty.is_float() => {
                        builder.ins().fadd(lhs, rhs)
                    }
                    b::InstrBody::Sub(..) if ty.is_int() => builder.ins().isub(lhs, rhs),
                    b::InstrBody::Sub(..) if ty.is_float() => {
                        builder.ins().fsub(lhs, rhs)
                    }
                    b::InstrBody::Mul(..) if ty.is_int() => builder.ins().imul(lhs, rhs),
                    b::InstrBody::Mul(..) if ty.is_float() => {
                        builder.ins().fmul(lhs, rhs)
                    }
                    b::InstrBody::Div(..) if ty.is_uint() => builder.ins().udiv(lhs, rhs),
                    b::InstrBody::Div(..) if ty.is_sint() => builder.ins().sdiv(lhs, rhs),
                    b::InstrBody::Div(..) if ty.is_float() => {
                        builder.ins().fdiv(lhs, rhs)
                    }
                    b::InstrBody::Mod(..) if ty.is_uint() => builder.ins().urem(lhs, rhs),
                    b::InstrBody::Mod(..) if ty.is_sint() => builder.ins().srem(lhs, rhs),
                    b::InstrBody::Mod(..) if ty.is_float() => {
                        let x = builder.ins().fdiv(lhs, rhs);
                        let x = builder.ins().trunc(x);
                        let y = builder.ins().fneg(rhs);
                        builder.ins().fma(x, y, lhs)
                    }
                    b::InstrBody::Eq(..) if ty.is_int() => {
                        builder.ins().icmp(cl::IntCC::Equal, lhs, rhs)
                    }
                    b::InstrBody::Eq(..) if ty.is_float() => {
                        builder.ins().fcmp(cl::FloatCC::Equal, lhs, rhs)
                    }
                    b::InstrBody::Neq(..) if ty.is_int() => {
                        builder.ins().icmp(cl::IntCC::NotEqual, lhs, rhs)
                    }
                    b::InstrBody::Neq(..) if ty.is_float() => {
                        builder.ins().fcmp(cl::FloatCC::NotEqual, lhs, rhs)
                    }
                    b::InstrBody::Lt(..) if ty.is_sint() => {
                        builder.ins().icmp(cl::IntCC::SignedLessThan, lhs, rhs)
                    }
                    b::InstrBody::Lt(..) if ty.is_uint() => {
                        builder.ins().icmp(cl::IntCC::UnsignedLessThan, lhs, rhs)
                    }
                    b::InstrBody::Lt(..) if ty.is_float() => {
                        builder.ins().fcmp(cl::FloatCC::LessThan, lhs, rhs)
                    }
                    b::InstrBody::Gt(..) if ty.is_sint() => {
                        builder.ins().icmp(cl::IntCC::SignedGreaterThan, lhs, rhs)
                    }
                    b::InstrBody::Gt(..) if ty.is_uint() => {
                        builder.ins().icmp(cl::IntCC::UnsignedGreaterThan, lhs, rhs)
                    }
                    b::InstrBody::Gt(..) if ty.is_float() => {
                        builder.ins().fcmp(cl::FloatCC::GreaterThan, lhs, rhs)
                    }
                    b::InstrBody::Lte(..) if ty.is_sint() => {
                        builder
                            .ins()
                            .icmp(cl::IntCC::SignedLessThanOrEqual, lhs, rhs)
                    }
                    b::InstrBody::Lte(..) if ty.is_uint() => {
                        builder
                            .ins()
                            .icmp(cl::IntCC::UnsignedLessThanOrEqual, lhs, rhs)
                    }
                    b::InstrBody::Lte(..) if ty.is_float() => {
                        builder.ins().fcmp(cl::FloatCC::LessThanOrEqual, lhs, rhs)
                    }
                    b::InstrBody::Gte(..) if ty.is_sint() => {
                        builder
                            .ins()
                            .icmp(cl::IntCC::SignedGreaterThanOrEqual, lhs, rhs)
                    }
                    b::InstrBody::Gte(..) if ty.is_uint() => builder.ins().icmp(
                        cl::IntCC::UnsignedGreaterThanOrEqual,
                        lhs,
                        rhs,
                    ),
                    b::InstrBody::Gte(..) if ty.is_float() => {
                        builder
                            .ins()
                            .fcmp(cl::FloatCC::GreaterThanOrEqual, lhs, rhs)
                    }
                    _ => unreachable!(),
                };

                self.values.insert(
                    (mod_idx, instr.results[0]),
                    types::RuntimeValue::new(
                        types::ValueSource::Primitive(cl_value),
                        mod_idx,
                        instr.results[0],
                    ),
                );
            }
            b::InstrBody::Not(cond) => {
                let cond = self.use_value_by_value(mod_idx, *cond);
                assert!(cond.len() == 1);
                let cond = cond[0];

                let builder = expect_builder!(self);

                let v_false = builder.ins().iconst(cl::types::I8, 0);
                let cl_value = builder.ins().icmp(cl::IntCC::Equal, cond, v_false);

                self.values.insert(
                    (mod_idx, instr.results[0]),
                    types::RuntimeValue::new(
                        types::ValueSource::Primitive(cl_value),
                        mod_idx,
                        instr.results[0],
                    ),
                );
            }
            b::InstrBody::If(cond, then_, else_) => {
                let cond = self.use_value_by_value(mod_idx, *cond);
                assert!(cond.len() == 1);
                let cond = cond[0];

                let builder = expect_builder!(self);

                let then_block = builder.create_block();
                let else_block = builder.create_block();

                builder.ins().brif(cond, then_block, &[], else_block, &[]);

                let mut scope = ScopePayload {
                    start_block: then_block,
                    block: then_block,
                    next_branches: vec![else_block],
                    result: None,
                    ty: None,
                };

                if instr.results.len() > 0 {
                    let module = &self.ctx.modules[mod_idx];
                    let ty = &module.values[instr.results[0]].ty;

                    let next_block = builder.create_block();
                    for native_ty in types::get_type_canonical(
                        ty,
                        self.ctx.modules,
                        &self.ctx.cl_module,
                    ) {
                        builder.append_block_param(next_block, native_ty);
                    }

                    scope.result = Some(instr.results[0]);
                    scope.ty = Some(Cow::Borrowed(ty));
                    self.scopes.last_mut().block = next_block;
                }

                self.scopes.begin(scope);

                builder.switch_to_block(then_block);
                self.add_body(then_, mod_idx, ResultPolicy::Normal);

                let builder = expect_builder!(self);

                self.scopes.branch();

                builder.switch_to_block(else_block);
                self.add_body(else_, mod_idx, ResultPolicy::Normal);

                let (scope, _) = self.scopes.end();

                let builder = expect_builder!(self);
                if !scope.is_never() {
                    let next_block = self.scopes.last().block;
                    builder.switch_to_block(next_block);
                }
            }
            b::InstrBody::Loop(inputs, body) => {
                let loop_block = {
                    let builder = expect_builder!(self);
                    builder.create_block()
                };

                let mut loop_args = vec![];
                for (loop_v, initial_v) in inputs {
                    let initial_runtime_value =
                        self.values[&(mod_idx, *initial_v)].clone();
                    let initial_values = self.use_value_canonical(mod_idx, *initial_v);
                    loop_args.extend(initial_values.iter().cloned());

                    let loop_values = initial_values
                        .into_iter()
                        .map(|initial_value| {
                            let builder = expect_builder!(self);
                            let native_ty = builder.func.dfg.value_type(initial_value);
                            builder.append_block_param(loop_block, native_ty)
                        })
                        .collect_vec();

                    let src = initial_runtime_value.src.with_values(&loop_values);
                    self.values.insert(
                        (mod_idx, *loop_v),
                        types::RuntimeValue::new(src, mod_idx, *loop_v),
                    );
                }

                let builder = expect_builder!(self);
                builder.ins().jump(loop_block, &loop_args);

                let continue_block = builder.create_block();
                let (result, ty) = if instr.results.len() > 0 {
                    assert_eq!(instr.results.len(), 1);
                    let result = instr.results[0];

                    let ty = &self.ctx.modules[mod_idx].values[result].ty;
                    for native_ty in types::get_type_canonical(
                        ty,
                        self.ctx.modules,
                        &self.ctx.cl_module,
                    ) {
                        builder.append_block_param(continue_block, native_ty);
                    }

                    (Some(result), Some(Cow::Borrowed(ty)))
                } else {
                    (None, None)
                };
                self.scopes.last_mut().block = continue_block;

                let scope = self.scopes.begin(ScopePayload {
                    start_block: loop_block,
                    block: loop_block,
                    next_branches: vec![],
                    result,
                    ty,
                });
                scope.is_loop = true;

                builder.switch_to_block(loop_block);
                self.add_body(body, mod_idx, ResultPolicy::Normal);

                let (scope, _) = self.scopes.end();

                let builder = expect_builder!(self);
                if !scope.is_never() {
                    let next_block = self.scopes.last().block;
                    builder.switch_to_block(next_block);
                }
            }
            b::InstrBody::Break(v) => {
                if let Some(v) = v {
                    let Some(runtime_value) = self.values.get(&(mod_idx, *v)).cloned()
                    else {
                        panic!("value should be present in scope: {v}");
                    };

                    match result_policy {
                        ResultPolicy::Normal => {
                            let cl_values = self.use_value_canonical(mod_idx, *v);

                            let builder = expect_builder!(self);

                            let prev_scope =
                                self.scopes.get(self.scopes.len() - 2).unwrap();
                            builder.ins().jump(prev_scope.block, &cl_values);
                            let block_params = builder.block_params(prev_scope.block);

                            if let Some(result) = self.scopes.last().result {
                                let src = runtime_value.src.with_values(block_params);
                                self.values.insert(
                                    (mod_idx, result),
                                    types::RuntimeValue::new(src, mod_idx, result),
                                );
                            }
                        }
                        ResultPolicy::Return(ReturnPolicy::Normal) => {
                            let cl_values = self.use_value_canonical(mod_idx, *v);
                            expect_builder!(self).ins().return_(&cl_values);
                        }
                        ResultPolicy::Return(ReturnPolicy::Void) => {
                            expect_builder!(self).ins().return_(&[]);
                        }
                        ResultPolicy::Return(ReturnPolicy::Struct(_)) => {
                            let cl_values = self.use_value_by_value(mod_idx, *v);
                            if let Some(res) = self.scopes.last().result {
                                let res_cl = self.use_value_by_ref(mod_idx, res);
                                let builder = expect_builder!(self);
                                let mut offset = 0;
                                for cl_value in &cl_values {
                                    builder.ins().store(
                                        cl::MemFlags::new(),
                                        *cl_value,
                                        res_cl,
                                        offset,
                                    );
                                    offset +=
                                        builder.func.dfg.value_type(*cl_value).bytes()
                                            as i32;
                                }
                            }
                            expect_builder!(self).ins().return_(&[]);
                        }
                        ResultPolicy::Return(ReturnPolicy::NoReturn) => unreachable!(),
                        ResultPolicy::Global => {
                            let cl_values = self.use_value_by_value(mod_idx, *v);
                            if let Some(res) = self.scopes.last().result {
                                let res_cl = self.use_value_by_ref(mod_idx, res);
                                let builder = expect_builder!(self);
                                let mut offset = 0;
                                for cl_value in &cl_values {
                                    builder.ins().store(
                                        cl::MemFlags::new(),
                                        *cl_value,
                                        res_cl,
                                        offset,
                                    );
                                    offset +=
                                        builder.func.dfg.value_type(*cl_value).bytes()
                                            as i32;
                                }
                            }
                        }
                    }
                } else {
                    if let ResultPolicy::Return(ReturnPolicy::Void) = result_policy {
                        expect_builder!(self).ins().return_(&[]);
                    } else {
                        unreachable!()
                    }
                }
            }
            b::InstrBody::Continue(vs) => {
                let block = self
                    .scopes
                    .last_loop()
                    .expect("continue instruction should be called in a loop")
                    .start_block;

                let values = vs
                    .into_iter()
                    .flat_map(|v| self.use_value_canonical(mod_idx, *v))
                    .collect_vec();

                expect_builder!(self).ins().jump(block, &values);
                self.scopes.last_mut().mark_as_never();
            }
            b::InstrBody::Call(func_mod_idx, func_idx, vs) => {
                let args = vs
                    .into_iter()
                    .flat_map(|v| self.use_value_canonical(mod_idx, *v))
                    .collect_vec();

                if let Some(value) = self.call_func(*func_mod_idx, *func_idx, args) {
                    self.save_value(mod_idx, instr.results[0], value);
                }
            }
            b::InstrBody::IndirectCall(func_v, vs) => {
                let func = self.values[&(mod_idx, *func_v)].clone();

                let mut args = vs
                    .into_iter()
                    .flat_map(|v| self.use_value_canonical(mod_idx, *v))
                    .collect_vec();

                let value = match &func.src {
                    types::ValueSource::AppliedMethod(
                        self_value,
                        (func_mod_idx, func_idx),
                    ) => {
                        args.push(*self_value);
                        self.call_func(*func_mod_idx, *func_idx, args)
                    }
                    types::ValueSource::AppliedMethodInderect(
                        self_value,
                        callee,
                        proto,
                    ) => {
                        args.push(*self_value);
                        self.call_indirect(proto, *callee, args)
                    }
                    types::ValueSource::FuncAsValue(func_as_value) => {
                        args.splice(0..0, [func_as_value.env]);
                        self.call_indirect(&func_as_value.proto, func_as_value.ptr, args)
                    }
                    src => todo!("call indirect: {src:?}"),
                };

                if let Some(value) = value {
                    self.save_value(mod_idx, instr.results[0], value);
                }
            }
            b::InstrBody::GetFunc(func_mod_idx, func_idx) => {
                let builder = expect_builder!(self);

                let (closure_func_id, proto) =
                    self.ctx.closure_for_func(*func_mod_idx, *func_idx);
                let closure_func_ref = self
                    .ctx
                    .cl_module
                    .declare_func_in_func(closure_func_id, builder.func);
                let closure_ptr = builder.ins().func_addr(
                    self.ctx.cl_module.isa().pointer_type(),
                    closure_func_ref,
                );

                let env = builder
                    .ins()
                    .iconst(self.ctx.cl_module.isa().pointer_type(), 0);
                let value = types::FuncAsValue::new(closure_ptr, env, proto);

                self.values.insert(
                    (mod_idx, instr.results[0]),
                    types::RuntimeValue::new(
                        value.into(),
                        *func_mod_idx,
                        instr.results[0],
                    ),
                );
            }
            b::InstrBody::GetField(source_v, name) => {
                let builder = expect_builder!(self);

                let source_ty = &self.ctx.modules[mod_idx].values[*source_v].ty;
                let source = &self.values[&(mod_idx, *source_v)];
                let b::TypeBody::TypeRef(ty_ref) = &source_ty.body else {
                    panic!("type should be a typeref");
                };
                let b::TypeDefBody::Record(rec) =
                    &self.ctx.modules[ty_ref.mod_idx].typedefs[ty_ref.idx].body
                else {
                    panic!("type should be a record type");
                };

                let mut offset = 0;
                for (k, v) in &rec.fields {
                    if k == name {
                        break;
                    }
                    for native_ty in types::get_type_canonical(
                        &v.ty,
                        self.ctx.modules,
                        &self.ctx.cl_module,
                    ) {
                        offset += native_ty.bytes();
                    }
                }

                let source_value =
                    source.src.add_by_ref(&mut self.ctx.cl_module, builder);

                let v = instr.results[0];
                let ty = &self.ctx.modules[mod_idx].values[v].ty;

                let mut values = vec![];
                for native_ty in
                    types::get_type_canonical(ty, self.ctx.modules, &self.ctx.cl_module)
                {
                    let value = builder.ins().load(
                        native_ty,
                        cl::MemFlags::new(),
                        source_value,
                        offset as i32,
                    );
                    values.push(value);
                    offset += native_ty.bytes();
                }

                let (value, n) = types::take_value_from_args(
                    mod_idx,
                    v,
                    &values,
                    self.ctx.modules,
                    &self.ctx.cl_module,
                );
                assert!(values.len() == n, "we should have consumed all values");

                self.values.insert((mod_idx, instr.results[0]), value);
            }
            b::InstrBody::GetMethod(source_v, name) => {
                let builder = expect_builder!(self);

                let source_ty = &self.ctx.modules[mod_idx].values[*source_v].ty;
                let source = &self.values[&(mod_idx, *source_v)];
                let b::TypeBody::TypeRef(ty_ref) = &source_ty.body else {
                    panic!("type should be a typeref");
                };

                match &self.ctx.modules[ty_ref.mod_idx].typedefs[ty_ref.idx].body {
                    b::TypeDefBody::Record(rec) => {
                        let value = source.src.add_by_ref(&self.ctx.cl_module, builder);
                        let method = &rec.methods[name];

                        self.values.insert(
                            (mod_idx, instr.results[0]),
                            types::RuntimeValue::new(
                                types::ValueSource::AppliedMethod(value, method.func_ref),
                                mod_idx,
                                instr.results[0],
                            ),
                        );
                    }
                    b::TypeDefBody::Interface(iface) => {
                        let (src, vtable) = match &source.src {
                            types::ValueSource::DynDispatched(dispatched) => {
                                (dispatched.src, dispatched.vtable)
                            }
                            _ => {
                                let ptr =
                                    source.src.add_by_ref(&self.ctx.cl_module, builder);
                                let src_value = builder.ins().load(
                                    self.ctx.cl_module.isa().pointer_type(),
                                    cl::MemFlags::new(),
                                    ptr,
                                    0,
                                );
                                let vtable_value = builder.ins().load(
                                    self.ctx.cl_module.isa().pointer_type(),
                                    cl::MemFlags::new(),
                                    ptr,
                                    self.ctx.cl_module.isa().pointer_bytes() as i32,
                                );
                                (src_value, vtable_value)
                            }
                        };

                        let method = &iface.methods[name];

                        let offset = self
                            .ctx
                            .vtables_desc
                            .get(&(ty_ref.mod_idx, ty_ref.idx))
                            .expect("Interface should already be defined")
                            .method_offset(name, &self.ctx.cl_module)
                            .unwrap();

                        let func_ptr = builder.ins().load(
                            self.ctx.cl_module.isa().pointer_type(),
                            cl::MemFlags::new(),
                            vtable,
                            offset as i32,
                        );

                        let proto = types::FuncPrototype::from_func(
                            method.func_ref.0,
                            method.func_ref.1,
                            self.ctx.modules,
                            &self.ctx.cl_module,
                        );

                        self.values.insert(
                            (mod_idx, instr.results[0]),
                            types::RuntimeValue::new(
                                types::ValueSource::AppliedMethodInderect(
                                    src, func_ptr, proto,
                                ),
                                mod_idx,
                                instr.results[0],
                            ),
                        );
                    }
                };
            }
            b::InstrBody::StrLen(source_v) => {
                let source = self.values[&(mod_idx, *source_v)].clone();

                let value = match &source.src {
                    types::ValueSource::Slice(slice) => slice.len.clone(),
                    src => todo!("str_ptr: {src:?}"),
                };

                self.values.insert(
                    (mod_idx, instr.results[0]),
                    types::RuntimeValue::new(value.into(), mod_idx, instr.results[0]),
                );
            }
            b::InstrBody::StrPtr(source_v) => {
                let source = self.values[&(mod_idx, *source_v)].clone();

                let value = match &source.src {
                    types::ValueSource::Slice(slice) => slice.ptr.clone(),
                    src => todo!("str_ptr: {src:?}"),
                };

                self.values.insert(
                    (mod_idx, instr.results[0]),
                    types::RuntimeValue::new(value.into(), mod_idx, instr.results[0]),
                );
            }
            b::InstrBody::Dispatch(v, iface_mod_idx, iface_ty_idx) => {
                let builder = expect_builder!(self);

                let ty = &self.ctx.modules[mod_idx].values[*v].ty;
                let b::TypeBody::TypeRef(ty_ref) = &ty.body else {
                    panic!("type should be a typeref");
                };
                let vtable_ref = types::VTableRef::new(
                    (*iface_mod_idx, *iface_ty_idx),
                    (ty_ref.mod_idx, ty_ref.idx),
                );
                let vtable_data = self.ctx.vtables_impl[&vtable_ref];
                let vtable_gv = self
                    .ctx
                    .cl_module
                    .declare_data_in_func(vtable_data, &mut builder.func);
                let vtable = builder
                    .ins()
                    .global_value(self.ctx.cl_module.isa().pointer_type(), vtable_gv);

                let src = self.use_value_by_ref(mod_idx, *v);

                let dispatched = types::DynDispatched::new(src, vtable.into());
                self.values.insert(
                    (mod_idx, instr.results[0]),
                    types::RuntimeValue::new(
                        dispatched.into(),
                        mod_idx,
                        instr.results[0],
                    ),
                );
            }
            b::InstrBody::Type(..) => {}
            b::InstrBody::GetProperty(..) | b::InstrBody::CompileError => {
                panic!("never should try to compile '{}'", &instr)
            }
            b::InstrBody::CreateNumber(..)
            | b::InstrBody::CreateBool(..)
            | b::InstrBody::CreateString(..)
            | b::InstrBody::CreateArray(..)
            | b::InstrBody::CreateRecord(..)
            | b::InstrBody::GetGlobal(..) => unreachable!(),
        }
    }

    pub fn value_from_instr(
        &mut self,
        instr: &'a b::Instr,
        mod_idx: usize,
    ) -> Option<types::RuntimeValue> {
        utils::replace_with(self, |mut this| {
            let value = 'match_b: {
                match &instr.body {
                    b::InstrBody::CreateNumber(n) => {
                        Some(this.create_number_inst(mod_idx, instr, n))
                    }
                    b::InstrBody::CreateBool(b) => Some(types::RuntimeValue::new(
                        (*b as u8).into(),
                        mod_idx,
                        instr.results[0],
                    )),
                    b::InstrBody::CreateString(s) => {
                        let data = this.ctx.data_for_string(s);
                        let len = types::ValueSource::uint_ptr(
                            s.len() as u64,
                            &this.ctx.cl_module,
                        );

                        Some(types::RuntimeValue::new(
                            Box::new(types::Slice::new(data.into(), len.into())).into(),
                            mod_idx,
                            instr.results[0],
                        ))
                    }
                    b::InstrBody::CreateArray(vs) => {
                        this.create_array_inst(mod_idx, instr, vs)
                    }
                    b::InstrBody::CreateRecord(fields) => {
                        let module = &self.ctx.modules[mod_idx];
                        let ty = &module.values[instr.results[0]].ty;

                        let values = types::tuple_from_record(
                            fields
                                .iter()
                                .map(|(name, v)| {
                                    (name, this.values[&(mod_idx, *v)].clone())
                                })
                                .collect_vec(),
                            ty,
                            this.ctx.modules,
                        );
                        let src = if values.len() > 0 {
                            let data = this.ctx.data_for_tuple(
                                values
                                    .iter()
                                    .map(|value| value.src.clone())
                                    .collect_vec(),
                            );
                            if let Some(data) = data {
                                data.into()
                            } else if this.builder.is_some() {
                                this.create_stack_slot(&values).into()
                            } else {
                                break 'match_b None;
                            }
                        } else {
                            types::ValueSource::I64(1)
                        };
                        Some(types::RuntimeValue::new(src, mod_idx, instr.results[0]))
                    }
                    b::InstrBody::GetGlobal(mod_idx, global_idx) => Some(
                        this.ctx
                            .get_global(*mod_idx, *global_idx)
                            .expect("global idx out of range")
                            .value
                            .clone(),
                    ),
                    _ => None,
                }
            };

            if let Some(value) = &value {
                this.values
                    .insert((mod_idx, instr.results[0]), value.clone());
            }

            (this, value)
        })
    }

    pub fn call_func(
        &mut self,
        func_mod_idx: usize,
        func_idx: usize,
        args: impl Into<Vec<cl::Value>>,
    ) -> Option<cl::Value> {
        let func_id = self
            .ctx
            .funcs
            .get(&(func_mod_idx, func_idx))
            .unwrap()
            .func_id
            .expect("Function should be declared");

        self.call(
            Callee::Direct(func_id),
            args,
            ReturnPolicy::from_func(
                func_mod_idx,
                func_idx,
                self.ctx.modules,
                &self.ctx.cl_module,
            ),
        )
    }

    pub fn call_indirect(
        &mut self,
        proto: &types::FuncPrototype,
        callee: cl::Value,
        args: impl Into<Vec<cl::Value>>,
    ) -> Option<cl::Value> {
        let builder = expect_builder!(self);

        let sig = &proto.signature;
        let sig_ref = match self.imported_signatures.get(sig) {
            Some(sig_ref) => *sig_ref,
            None => {
                let sig_ref = builder.import_signature(sig.clone());
                self.imported_signatures.insert(sig.clone(), sig_ref);
                sig_ref
            }
        };

        self.call(Callee::Indirect(sig_ref, callee), args, proto.ret_policy)
    }

    pub fn call(
        &mut self,
        callee: Callee,
        args: impl Into<Vec<cl::Value>>,
        ret_policy: ReturnPolicy,
    ) -> Option<cl::Value> {
        let builder = expect_builder!(self);

        let mut args = args.into();

        if let ReturnPolicy::Struct(size) = ret_policy {
            let ss_data = cl::StackSlotData::new(cl::StackSlotKind::ExplicitSlot, size);
            let ss = builder.create_sized_stack_slot(ss_data);
            let stack_addr =
                builder
                    .ins()
                    .stack_addr(self.ctx.cl_module.isa().pointer_type(), ss, 0);
            args.insert(0, stack_addr);
        }

        let instr = match callee {
            Callee::Direct(func_id) => {
                let func_ref = self.declared_funcs.entry(func_id).or_insert_with(|| {
                    let func_ref = self
                        .ctx
                        .cl_module
                        .declare_func_in_func(func_id, builder.func);
                    func_ref
                });

                builder.ins().call(*func_ref, &args)
            }
            Callee::Indirect(sig, ptr) => builder.ins().call_indirect(sig, ptr, &args),
        };

        let results = builder.inst_results(instr);
        assert!(results.len() <= 1);

        match ret_policy {
            ReturnPolicy::Normal => Some(results[0]),
            ReturnPolicy::Struct(..) => Some(args[0]),
            ReturnPolicy::NoReturn => {
                builder.ins().trap(cl::TrapCode::UnreachableCodeReached);
                self.scopes.last_mut().mark_as_never();
                None
            }
            ReturnPolicy::Void => None,
        }
    }

    fn create_array_inst(
        &mut self,
        mod_idx: usize,
        instr: &'a b::Instr,
        vs: &Vec<usize>,
    ) -> Option<types::RuntimeValue> {
        let data = self.ctx.data_for_tuple(
            vs.iter()
                .map(|v| self.values[&(mod_idx, *v)].src.clone())
                .collect_vec(),
        );

        let ptr = if let Some(data) = data {
            data.into()
        } else if self.builder.is_some() {
            let b::TypeBody::Array(array_ty) =
                &self.ctx.modules[mod_idx].values[instr.results[0]].ty.body
            else {
                panic!("type should be an array type");
            };

            let item_tys = types::get_type_by_value(
                &array_ty.item,
                self.ctx.modules,
                &self.ctx.cl_module,
            );

            let size =
                item_tys.iter().map(|ty| ty.bytes()).sum::<u32>() * vs.len() as u32;

            let ss_data = cl::StackSlotData::new(cl::StackSlotKind::ExplicitSlot, size);
            let ss = expect_builder!(self).create_sized_stack_slot(ss_data);

            let mut offset = 0;
            for v in vs {
                let native_values = self.use_value_by_value(mod_idx, *v);
                for (ty, value) in izip!(&item_tys, native_values) {
                    let builder = expect_builder!(self);
                    builder.ins().stack_store(value, ss, offset as i32);
                    offset += ty.bytes();
                }
            }

            ss.into()
        } else {
            return None;
        };

        let len = types::ValueSource::uint_ptr(vs.len() as u64, &self.ctx.cl_module);

        Some(types::RuntimeValue::new(
            Box::new(types::Slice::new(ptr, len.into())).into(),
            mod_idx,
            instr.results[0],
        ))
    }

    fn use_value_canonical(&mut self, mod_idx: usize, v: b::ValueIdx) -> Vec<cl::Value> {
        let runtime_value = unwrap!(
            self.values.get(&(mod_idx, v)),
            "value should be present in scope: {v}"
        );
        let ty = &self.ctx.modules[mod_idx].values[v].ty;
        runtime_value.src.add_canonical(
            ty,
            &self.ctx.modules,
            &mut self.ctx.cl_module,
            expect_builder!(self),
        )
    }

    fn use_value_by_ref(&mut self, mod_idx: usize, v: b::ValueIdx) -> cl::Value {
        let runtime_value = unwrap!(
            self.values.get(&(mod_idx, v)),
            "value should be present in scope: {v}"
        );
        runtime_value
            .src
            .add_by_ref(&mut self.ctx.cl_module, expect_builder!(self))
    }

    fn use_value_by_value(&mut self, mod_idx: usize, v: b::ValueIdx) -> Vec<cl::Value> {
        let runtime_value = unwrap!(
            self.values.get(&(mod_idx, v)),
            "value should be present in scope: {v}"
        );
        let ty = &self.ctx.modules[mod_idx].values[v].ty;
        runtime_value.src.add_by_value(
            ty,
            &self.ctx.modules,
            &mut self.ctx.cl_module,
            expect_builder!(self),
        )
    }

    fn save_value(&mut self, mod_idx: usize, v: b::ValueIdx, value: cl::Value) {
        let ty = &self.ctx.modules[mod_idx].values[v].ty;
        let src = if ty.is_aggregate(self.ctx.modules) {
            types::ValueSource::Ptr(value)
        } else {
            types::ValueSource::Primitive(value)
        };
        self.values
            .insert((mod_idx, v), types::RuntimeValue::new(src, mod_idx, v));
    }

    fn create_number_inst(
        &mut self,
        mod_idx: usize,
        instr: &'a b::Instr,
        n: &String,
    ) -> types::RuntimeValue {
        let module = &self.ctx.modules[mod_idx];
        let ty = &module.values[instr.results[0]].ty;

        macro_rules! parse_num {
            ($variant:ident $(, $($cast:tt)+ )?) => {{
                let value = n.parse().unwrap();
                let src = types::ValueSource::$variant($( $($cast)+ )? (value));
                types::RuntimeValue::new(
                    src,
                    mod_idx,
                    instr.results[0],
                )
            }};
        }

        match &ty.body {
            b::TypeBody::I8 => parse_num!(I8, i8::cast_unsigned),
            b::TypeBody::I16 => parse_num!(I16, i16::cast_unsigned),
            b::TypeBody::I32 => parse_num!(I32, i32::cast_unsigned),
            b::TypeBody::I64 => parse_num!(I64, i64::cast_unsigned),
            b::TypeBody::U8 => parse_num!(I8),
            b::TypeBody::U16 => parse_num!(I16),
            b::TypeBody::U32 => parse_num!(I32),
            b::TypeBody::U64 => parse_num!(I64),
            b::TypeBody::USize => match self.ctx.cl_module.isa().pointer_bytes() {
                1 => parse_num!(I8),
                2 => parse_num!(I16),
                4 => parse_num!(I32),
                8 => parse_num!(I64),
                _ => unreachable!("how many bytes?"),
            },
            b::TypeBody::F32 => {
                parse_num!(F32, types::F32Bits::from_float)
            }
            b::TypeBody::F64 => {
                parse_num!(F64, types::F64Bits::from_float)
            }
            _ => unreachable!("Cannot parse {n} as {ty}"),
        }
    }

    fn create_stack_slot<'v>(
        &mut self,
        values: impl IntoIterator<Item = &'v types::RuntimeValue> + 'v,
    ) -> cl::StackSlot {
        let Some(func) = &mut self.builder else {
            panic!("cannot add stack slot without a function");
        };

        let mut size = 0;
        let mut stored_values = Vec::new();
        for v in values {
            let v_ty = &self.ctx.modules[v.mod_idx].values[v.value_idx].ty;
            for v in v.src.add_canonical(
                v_ty,
                &self.ctx.modules,
                &mut self.ctx.cl_module,
                func,
            ) {
                stored_values.push((size, v));
                size += func.func.dfg.value_type(v).bytes();
            }
        }

        let ss_data = cl::StackSlotData::new(cl::StackSlotKind::ExplicitSlot, size);
        let ss = func.create_sized_stack_slot(ss_data);
        for (offset, value) in stored_values {
            func.ins().stack_store(value, ss, offset as i32);
        }

        ss
    }

    fn add_assert(&mut self, cond: cl::Value, code: cl::TrapCode) {
        let builder = expect_builder!(self);
        builder.ins().trapz(cond, code);
    }
}

#[derive(Debug)]
pub struct ScopePayload<'a> {
    pub start_block: cl::Block,
    pub block: cl::Block,
    pub next_branches: Vec<cl::Block>,
    pub result: Option<b::ValueIdx>,
    pub ty: Option<Cow<'a, b::Type>>,
}
impl utils::SimpleScopePayload for ScopePayload<'_> {
    fn branch(&mut self, _: Option<&Self>) {
        let block = self.next_branches.pop().unwrap();
        self.start_block = block;
        self.block = block;
    }
}
