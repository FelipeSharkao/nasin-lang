use std::borrow::Cow;
use std::collections::HashMap;
use std::mem;

use cl::InstBuilder;
use cranelift_shim::{self as cl, Module};
use derive_new::new;
use itertools::{izip, Itertools};

use super::context::{CodegenContext, GlobalBinding};
use super::types;
use crate::utils::unwrap;
use crate::{bytecode as b, utils};

#[derive(Debug, Clone, Copy)]
pub enum ResultPolicy {
    Normal,
    Global,
    Return,
    StructReturn,
}

#[derive(Debug, Clone, Copy)]
pub enum CallReturnPolicy {
    Normal,
    StructReturn(u32),
    NoReturn,
}

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
    pub values: HashMap<b::ValueIdx, types::RuntimeValue>,
    #[new(default)]
    imported_signatures: HashMap<(usize, usize), cl::SigRef>,
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

        if let (ResultPolicy::StructReturn, Some(result)) = (result_policy, result) {
            let cl_value = cl_values.remove(0);
            let runtime_value =
                types::RuntimeValue::new(cl_value.into(), mod_idx, result).is_ptr(true);
            self.values.insert(result, runtime_value);
        }

        self.values.extend(izip!(
            params.iter().copied(),
            types::tuple_from_args(mod_idx, params, &cl_values, &self.ctx.modules),
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
                let lhs = self.use_value(*a);
                let rhs = self.use_value(*b);
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
                    instr.results[0],
                    types::RuntimeValue::new(cl_value.into(), mod_idx, instr.results[0]),
                );
            }
            b::InstrBody::Not(cond) => {
                let cond = self.use_value(*cond);
                let builder = expect_builder!(self);

                let v_false = builder.ins().iconst(cl::types::I8, 0);
                let cl_value = builder.ins().icmp(cl::IntCC::Equal, cond, v_false);

                self.values.insert(
                    instr.results[0],
                    types::RuntimeValue::new(cl_value.into(), mod_idx, instr.results[0]),
                );
            }
            b::InstrBody::If(cond, then_, else_) => {
                let cond = self.use_value(*cond);
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
                    for native_ty in
                        types::get_type(ty, self.ctx.modules, &self.ctx.obj_module)
                    {
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
                    let initial_runtime_value = self.values[initial_v];
                    let initial_values = self.use_values(*initial_v);
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
                        *loop_v,
                        types::RuntimeValue::new(src, mod_idx, *loop_v)
                            .is_ptr(initial_runtime_value.is_ptr),
                    );
                }

                let builder = expect_builder!(self);
                builder.ins().jump(loop_block, &loop_args);

                let continue_block = builder.create_block();
                let (result, ty) = if instr.results.len() > 0 {
                    assert_eq!(instr.results.len(), 1);
                    let result = instr.results[0];

                    let ty = &self.ctx.modules[mod_idx].values[result].ty;
                    for native_ty in
                        types::get_type(ty, self.ctx.modules, &self.ctx.obj_module)
                    {
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
                let runtime_value = self.values[v];
                let ty = &self.ctx.modules[mod_idx].values[*v].ty;
                let mut cl_values: Vec<_>;

                match result_policy {
                    ResultPolicy::Normal => {
                        cl_values = self.use_values(*v);
                        if let Some(prev_scope) = self.scopes.get(self.scopes.len() - 2) {
                            let builder = expect_builder!(self);
                            builder.ins().jump(prev_scope.block, &cl_values);
                            cl_values = builder.block_params(prev_scope.block).to_vec();
                        }
                    }
                    ResultPolicy::Return => {
                        cl_values = self.use_values(*v);
                        expect_builder!(self).ins().return_(&cl_values);
                    }
                    ResultPolicy::StructReturn => {
                        let cl_value = self.use_value(*v);
                        if let Some(res) = self.scopes.last().result {
                            let size = types::get_size(
                                ty,
                                self.ctx.modules,
                                &self.ctx.obj_module,
                            );

                            let res_cl = self.use_value(res);

                            self.copy_bytes(res_cl, cl_value, size);
                        }
                        expect_builder!(self).ins().return_(&[]);
                        cl_values = vec![cl_value];
                    }
                    ResultPolicy::Global => {
                        cl_values = self.use_values(*v);
                    }
                }

                let scope = self.scopes.last();
                let result = scope.result.unwrap();
                let src = runtime_value.src.with_values(&cl_values);
                self.values.insert(
                    result,
                    types::RuntimeValue::new(src, mod_idx, result)
                        .is_ptr(runtime_value.is_ptr),
                );
            }
            b::InstrBody::Continue(vs) => {
                let block = self
                    .scopes
                    .last_loop()
                    .expect("continue instruction should be called in a loop")
                    .start_block;

                let values = vs
                    .into_iter()
                    .flat_map(|v| self.use_values(*v))
                    .collect_vec();

                expect_builder!(self).ins().jump(block, &values);
                self.scopes.last_mut().mark_as_never();
            }
            b::InstrBody::Call(func_mod_idx, func_idx, vs) => {
                let builder = expect_builder!(self);

                let args = vs
                    .into_iter()
                    .flat_map(|v| self.use_values(*v))
                    .collect_vec();

                if let Some(value) = self.call(*func_mod_idx, *func_idx, args) {
                    let ty = &self.ctx.modules[mod_idx].values[instr.results[0]].ty;

                    let mut is_ptr = false;
                    if let &b::TypeBody::TypeRef(ty_mod_idx, ty_idx) = &ty.body {
                        let typebody =
                            &self.ctx.modules[ty_mod_idx].typedefs[ty_idx].body;
                        is_ptr = matches!(typebody, b::TypeDefBody::Record(_))
                    };

                    self.values.insert(
                        instr.results[0],
                        types::RuntimeValue::new(value.into(), mod_idx, instr.results[0])
                            .is_ptr(is_ptr),
                    );
                }
            }
            b::InstrBody::IndirectCall(func_v, vs) => {
                let func = self.values[func_v];

                let mut args = vs
                    .into_iter()
                    .flat_map(|v| self.use_values(*v))
                    .collect_vec();

                match &func.src {
                    types::ValueSource::AppliedMethod(
                        self_value,
                        (func_mod_idx, func_idx),
                    ) => {
                        args.push(*self_value);

                        if let Some(value) = self.call(*func_mod_idx, *func_idx, args) {
                            let ty =
                                &self.ctx.modules[mod_idx].values[instr.results[0]].ty;

                            let mut is_ptr = false;
                            if let &b::TypeBody::TypeRef(ty_mod_idx, ty_idx) = &ty.body {
                                let typebody =
                                    &self.ctx.modules[ty_mod_idx].typedefs[ty_idx].body;
                                is_ptr = matches!(typebody, b::TypeDefBody::Record(_))
                            };

                            self.values.insert(
                                instr.results[0],
                                types::RuntimeValue::new(
                                    value.into(),
                                    mod_idx,
                                    instr.results[0],
                                ),
                            );
                        }
                    }
                    types::ValueSource::AppliedMethodInderect(
                        self_value,
                        callee,
                        ref_func_ref,
                    ) => {
                        args.push(*self_value);

                        if let Some(value) = self.call_indirect(
                            ref_func_ref.0,
                            ref_func_ref.1,
                            *callee,
                            args,
                        ) {
                            let ty =
                                &self.ctx.modules[mod_idx].values[instr.results[0]].ty;

                            let mut is_ptr = false;
                            if let &b::TypeBody::TypeRef(ty_mod_idx, ty_idx) = &ty.body {
                                let typebody =
                                    &self.ctx.modules[ty_mod_idx].typedefs[ty_idx].body;
                                is_ptr = matches!(typebody, b::TypeDefBody::Record(_))
                            };

                            self.values.insert(
                                instr.results[0],
                                types::RuntimeValue::new(
                                    value.into(),
                                    mod_idx,
                                    instr.results[0],
                                ),
                            );
                        }
                    }
                    _ => todo!("function as value"),
                }
            }
            b::InstrBody::GetField(source_v, name) => {
                let builder = expect_builder!(self);

                let source_ty = &self.ctx.modules[mod_idx].values[*source_v].ty;
                let source = &self.values[source_v];
                let b::Type {
                    body:
                        b::TypeBody::TypeRef(ty_mod_idx, ty_idx)
                        | b::TypeBody::SelfType(ty_mod_idx, ty_idx),
                    ..
                } = source_ty
                else {
                    panic!("type should be a typeref");
                };
                let b::TypeDefBody::Record(rec) =
                    &self.ctx.modules[*ty_mod_idx].typedefs[*ty_idx].body
                else {
                    panic!("type should be a record type");
                };

                let mut offset = 0;
                for (k, v) in &rec.fields {
                    if k == name {
                        break;
                    }
                    for native_ty in
                        types::get_type(&v.ty, self.ctx.modules, &self.ctx.obj_module)
                    {
                        offset += native_ty.bytes();
                    }
                }

                let ty = &self.ctx.modules[mod_idx].values[instr.results[0]].ty;

                let field_ty =
                    types::get_type(&ty, self.ctx.modules, &self.ctx.obj_module);
                assert_eq!(field_ty.len(), 1, "how do whe load this?");

                let source_value =
                    source.add_value_to_func(&mut self.ctx.obj_module, builder);
                let value = builder.ins().load(
                    field_ty[0],
                    cl::MemFlags::new(),
                    source_value,
                    offset as i32,
                );

                let mut is_ptr = false;
                if let &b::TypeBody::TypeRef(ty_mod_idx, ty_idx) = &ty.body {
                    let typebody = &self.ctx.modules[ty_mod_idx].typedefs[ty_idx].body;
                    is_ptr = matches!(
                        typebody,
                        b::TypeDefBody::Record(_) | b::TypeDefBody::Interface(_)
                    );
                };

                self.values.insert(
                    instr.results[0],
                    types::RuntimeValue::new(value.into(), mod_idx, instr.results[0])
                        .is_ptr(is_ptr),
                );
            }
            b::InstrBody::GetMethod(source_v, name) => {
                let builder = expect_builder!(self);

                let source_ty = &self.ctx.modules[mod_idx].values[*source_v].ty;
                let source = &self.values[source_v];
                let b::TypeBody::TypeRef(ty_mod_idx, ty_idx) = &source_ty.body else {
                    panic!("type should be a typeref");
                };

                match &self.ctx.modules[*ty_mod_idx].typedefs[*ty_idx].body {
                    b::TypeDefBody::Record(rec) => {
                        let value =
                            source.add_value_to_func(&self.ctx.obj_module, builder);
                        let method = &rec.methods[name];

                        self.values.insert(
                            instr.results[0],
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
                                let ptr = source
                                    .add_value_to_func(&self.ctx.obj_module, builder);
                                let src_value = builder.ins().load(
                                    self.ctx.obj_module.isa().pointer_type(),
                                    cl::MemFlags::new(),
                                    ptr,
                                    0,
                                );
                                let vtable_value = builder.ins().load(
                                    self.ctx.obj_module.isa().pointer_type(),
                                    cl::MemFlags::new(),
                                    ptr,
                                    self.ctx.obj_module.isa().pointer_bytes() as i32,
                                );
                                (src_value, vtable_value)
                            }
                        };

                        let method = &iface.methods[name];

                        let offset = self
                            .ctx
                            .vtables_desc
                            .get(&(*ty_mod_idx, *ty_idx))
                            .expect("Interface should already be defined")
                            .method_offset(name, &self.ctx.obj_module)
                            .unwrap();

                        let func_ptr = builder.ins().load(
                            self.ctx.obj_module.isa().pointer_type(),
                            cl::MemFlags::new(),
                            vtable,
                            offset as i32,
                        );

                        self.values.insert(
                            instr.results[0],
                            types::RuntimeValue::new(
                                types::ValueSource::AppliedMethodInderect(
                                    src,
                                    func_ptr,
                                    method.func_ref,
                                ),
                                mod_idx,
                                instr.results[0],
                            ),
                        );
                    }
                };
            }
            b::InstrBody::ArrayLen(source_v) | b::InstrBody::StrLen(source_v) => {
                let builder = expect_builder!(self);

                let source = self.values[source_v]
                    .add_value_to_func(&self.ctx.obj_module, builder);
                let value = builder.ins().load(
                    self.ctx.obj_module.isa().pointer_type(),
                    cl::MemFlags::new(),
                    source,
                    0,
                );
                self.values.insert(
                    instr.results[0],
                    types::RuntimeValue::new(
                        types::ValueSource::Value(value),
                        mod_idx,
                        instr.results[0],
                    ),
                );
            }
            b::InstrBody::ArrayPtr(source_v, idx)
            | b::InstrBody::StrPtr(source_v, idx) => {
                let source_ty = &self.ctx.modules[mod_idx].values[*source_v].ty;
                let source = self.values[source_v];
                let cl_source = source
                    .add_value_to_func(&mut self.ctx.obj_module, expect_builder!(self));

                let (item_size, len) = match &source_ty.body {
                    b::TypeBody::Array(array_ty) => (
                        types::get_size(
                            &array_ty.item,
                            &self.ctx.modules,
                            &self.ctx.obj_module,
                        ),
                        array_ty.len,
                    ),
                    b::TypeBody::String(str_ty) => (1, str_ty.len),
                    _ => panic!("type should be string or array"),
                };

                if let Some(len) = len {
                    assert!(*idx < len as u64);
                } else {
                    // Check length at runtime
                    let builder = expect_builder!(self);

                    let idx_value = builder
                        .ins()
                        .iconst(self.ctx.obj_module.isa().pointer_type(), unsafe {
                            mem::transmute::<_, i64>(*idx)
                        });
                    let len = builder.ins().load(
                        self.ctx.obj_module.isa().pointer_type(),
                        cl::MemFlags::new(),
                        cl_source,
                        0,
                    );
                    let cond =
                        builder
                            .ins()
                            .icmp(cl::IntCC::UnsignedLessThan, idx_value, len);
                    self.add_assert(cond, cl::TrapCode::NullReference);
                }

                let builder = expect_builder!(self);

                let offset = self.ctx.obj_module.isa().pointer_bytes() as u64
                    + idx * item_size as u64;
                let offset_value = builder
                    .ins()
                    .iconst(self.ctx.obj_module.isa().pointer_type(), unsafe {
                        mem::transmute::<_, i64>(offset)
                    });
                let value = builder.ins().iadd(cl_source, offset_value);

                self.values.insert(
                    instr.results[0],
                    types::RuntimeValue::new(value.into(), mod_idx, instr.results[0])
                        .is_ptr(true),
                );
            }
            b::InstrBody::Dispatch(v, iface_mod_idx, iface_ty_idx) => {
                let builder = expect_builder!(self);

                let ty = &self.ctx.modules[mod_idx].values[*v].ty;
                let b::TypeBody::TypeRef(ty_mod_idx, ty_idx) = &ty.body else {
                    panic!("type should be a typeref");
                };
                let vtable_ref = types::VTableRef::new(
                    (*iface_mod_idx, *iface_ty_idx),
                    (*ty_mod_idx, *ty_idx),
                );
                let vtable_data = self.ctx.vtables_impl[&vtable_ref];
                let vtable_gv = self
                    .ctx
                    .obj_module
                    .declare_data_in_func(vtable_data, &mut builder.func);
                let vtable = builder
                    .ins()
                    .global_value(self.ctx.obj_module.isa().pointer_type(), vtable_gv);

                let src = self.use_value(*v);

                let dispatched = types::DynDispatched::new(src, vtable.into());
                self.values.insert(
                    instr.results[0],
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
                        let module = &self.ctx.modules[mod_idx];
                        let ty = &module.values[instr.results[0]].ty;

                        macro_rules! parse_num {
                            ($ty:ty, $variant:ident) => {{
                                let value: $ty = n.parse().unwrap();
                                let src = types::ValueSource::$variant(unsafe {
                                    mem::transmute(value)
                                });
                                Some(types::RuntimeValue::new(
                                    src,
                                    mod_idx,
                                    instr.results[0],
                                ))
                            }};
                        }

                        match &ty.body {
                            b::TypeBody::I8 => parse_num!(i8, I8),
                            b::TypeBody::I16 => parse_num!(i16, I16),
                            b::TypeBody::I32 => parse_num!(i32, I32),
                            b::TypeBody::I64 => parse_num!(i64, I64),
                            b::TypeBody::U8 => parse_num!(u8, I8),
                            b::TypeBody::U16 => parse_num!(u16, I16),
                            b::TypeBody::U32 => parse_num!(u32, I32),
                            b::TypeBody::U64 => parse_num!(u64, I64),
                            b::TypeBody::USize => {
                                match this.ctx.obj_module.isa().pointer_bytes() {
                                    1 => parse_num!(u8, I8),
                                    2 => parse_num!(u16, I16),
                                    4 => parse_num!(u32, I32),
                                    8 => parse_num!(u64, I64),
                                    _ => unreachable!(),
                                }
                            }
                            b::TypeBody::F32 => parse_num!(f32, F32),
                            b::TypeBody::F64 => parse_num!(f64, F64),
                            b::TypeBody::Void
                            | b::TypeBody::Never
                            | b::TypeBody::Bool
                            | b::TypeBody::String(_)
                            | b::TypeBody::TypeRef(_, _)
                            | b::TypeBody::SelfType(_, _)
                            | b::TypeBody::Array(_)
                            | b::TypeBody::Ptr(_)
                            | b::TypeBody::Inferred(_)
                            | b::TypeBody::AnyOpaque
                            | b::TypeBody::AnyNumber
                            | b::TypeBody::AnySignedNumber
                            | b::TypeBody::AnyFloat
                            | b::TypeBody::Func(_) => panic!("Cannot parse {n} as {ty}"),
                        }
                    }
                    b::InstrBody::CreateBool(b) => Some(types::RuntimeValue::new(
                        (*b as u8).into(),
                        mod_idx,
                        instr.results[0],
                    )),
                    b::InstrBody::CreateString(s) => {
                        let data = this.ctx.data_for_string(s);
                        Some(types::RuntimeValue::new(
                            data.into(),
                            mod_idx,
                            instr.results[0],
                        ))
                    }
                    b::InstrBody::CreateArray(vs) => {
                        let data = this.ctx.data_for_array(
                            vs.iter().map(|v| this.values[v].src).collect_vec(),
                        );
                        let src = if let Some(data) = data {
                            data.into()
                        } else if this.builder.is_some() {
                            let values = vs.iter().map(|v| this.values[v]).collect_vec();
                            this.create_stack_slot(&values).into()
                        } else {
                            break 'match_b None;
                        };
                        Some(types::RuntimeValue::new(src, mod_idx, instr.results[0]))
                    }
                    b::InstrBody::CreateRecord(fields) => {
                        let module = &self.ctx.modules[mod_idx];
                        let ty = &module.values[instr.results[0]].ty;

                        let values = types::tuple_from_record(
                            fields
                                .iter()
                                .map(|(name, v)| (name, this.values[v]))
                                .collect_vec(),
                            ty,
                            this.ctx.modules,
                        );
                        let data = this.ctx.data_for_tuple(
                            values.iter().map(|value| value.src).collect_vec(),
                        );
                        let src = if let Some(data) = data {
                            data.into()
                        } else if this.builder.is_some() {
                            this.create_stack_slot(&values).into()
                        } else {
                            break 'match_b None;
                        };
                        Some(types::RuntimeValue::new(src, mod_idx, instr.results[0]))
                    }
                    b::InstrBody::GetGlobal(mod_idx, global_idx) => Some(
                        this.ctx
                            .get_global(*mod_idx, *global_idx)
                            .expect("global idx out of range")
                            .value,
                    ),
                    _ => None,
                }
            };

            if let Some(value) = &value {
                this.values.insert(instr.results[0], *value);
            }

            (this, value)
        })
    }

    pub fn store_global(&mut self, value: types::RuntimeValue, global: &GlobalBinding) {
        let types::ValueSource::Data(data_id) = &global.value.src else {
            panic!("should never try to store a global that is a const");
        };

        let builder = expect_builder!(self);

        let ty = value.native_type(self.ctx.modules, &self.ctx.obj_module);
        let global_value = self
            .ctx
            .obj_module
            .declare_data_in_func(*data_id, &mut builder.func);
        let ptr = builder
            .ins()
            .global_value(self.ctx.obj_module.isa().pointer_type(), global_value);

        let mut offset = 0;
        for v in value.add_values_to_func(&mut self.ctx.obj_module, builder) {
            builder
                .ins()
                .store(cl::MemFlags::new(), v, ptr, offset as i32);
            offset += builder.func.dfg.value_type(v).bytes();
        }
    }

    pub fn call(
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

        self.native_call(
            Callee::Direct(func_id),
            args,
            self.func_return_policy(func_mod_idx, func_idx),
        )
    }

    pub fn call_indirect(
        &mut self,
        ref_func_mod_idx: usize,
        ref_func_idx: usize,
        callee: cl::Value,
        args: impl Into<Vec<cl::Value>>,
    ) -> Option<cl::Value> {
        let builder = expect_builder!(self);

        let sig_ref = *self
            .imported_signatures
            .entry((ref_func_mod_idx, ref_func_idx))
            .or_insert_with(|| {
                let sig = self.ctx.funcs[&(ref_func_mod_idx, ref_func_idx)]
                    .signature
                    .clone();
                builder.import_signature(sig)
            });

        self.native_call(
            Callee::Indirect(sig_ref, callee),
            args,
            self.func_return_policy(ref_func_mod_idx, ref_func_idx),
        )
    }

    pub fn native_call(
        &mut self,
        callee: Callee,
        args: impl Into<Vec<cl::Value>>,
        ret_policy: CallReturnPolicy,
    ) -> Option<cl::Value> {
        let builder = expect_builder!(self);

        let mut args = args.into();

        if let CallReturnPolicy::StructReturn(size) = ret_policy {
            let ss_data = cl::StackSlotData::new(cl::StackSlotKind::ExplicitSlot, size);
            let ss = builder.create_sized_stack_slot(ss_data);
            let stack_addr =
                builder
                    .ins()
                    .stack_addr(self.ctx.obj_module.isa().pointer_type(), ss, 0);
            args.insert(0, stack_addr);
        }

        let instr = match callee {
            Callee::Direct(func_id) => {
                let func_ref = self.declared_funcs.entry(func_id).or_insert_with(|| {
                    let func_ref = self
                        .ctx
                        .obj_module
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
            CallReturnPolicy::Normal => Some(results[0]),
            CallReturnPolicy::StructReturn(..) => Some(args[0]),
            CallReturnPolicy::NoReturn => {
                builder.ins().trap(cl::TrapCode::UnreachableCodeReached);
                self.scopes.last_mut().mark_as_never();
                None
            }
        }
    }

    fn use_value(&mut self, v: b::ValueIdx) -> cl::Value {
        let runtime_value =
            unwrap!(self.values.get(&v), "value should be present in scope: {v}");
        runtime_value.add_value_to_func(&mut self.ctx.obj_module, expect_builder!(self))
    }

    fn use_values(&mut self, v: b::ValueIdx) -> Vec<cl::Value> {
        let runtime_value =
            unwrap!(self.values.get(&v), "value should be present in scope: {v}");
        runtime_value.add_values_to_func(&mut self.ctx.obj_module, expect_builder!(self))
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
            for v in v.add_values_to_func(&self.ctx.obj_module, func) {
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

    fn copy_bytes(&mut self, dst: cl::Value, src: cl::Value, size: u32) {
        let builder = expect_builder!(self);

        let mut offset: i32 = 0;
        loop {
            let remaining = size - (offset as u32);

            let mut copy = |ty: cl::types::Type| {
                let tmp = builder.ins().load(ty, cl::MemFlags::new(), src, offset);
                builder.ins().store(cl::MemFlags::new(), tmp, dst, offset);
                offset += ty.bytes() as i32;
            };

            if remaining >= 16 {
                copy(cl::types::I128);
            } else if remaining >= 8 {
                copy(cl::types::I64);
            } else if remaining >= 4 {
                copy(cl::types::I32);
            } else if remaining >= 2 {
                copy(cl::types::I16);
            } else if remaining >= 1 {
                copy(cl::types::I8);
            } else {
                break;
            }
        }
    }

    fn func_return_policy(
        &self,
        func_mod_idx: usize,
        func_idx: usize,
    ) -> CallReturnPolicy {
        let func = &self.ctx.modules[func_mod_idx].funcs[func_idx];
        let ret_ty = &self.ctx.modules[func_mod_idx].values[func.ret].ty;
        if ret_ty.is_never() {
            CallReturnPolicy::NoReturn
        } else if ret_ty.is_aggregate(&self.ctx.modules) {
            let size = types::get_size(ret_ty, &self.ctx.modules, &self.ctx.obj_module);
            CallReturnPolicy::StructReturn(size as u32)
        } else {
            CallReturnPolicy::Normal
        }
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
