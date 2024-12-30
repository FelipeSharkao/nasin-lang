use std::borrow::Cow;
use std::collections::HashMap;
use std::mem;

use cl::InstBuilder;
use cranelift_shim as cl;
use derive_new::new;
use itertools::{izip, Itertools};

use super::globals::{GlobalBinding, Globals};
use super::types::{self, get_size, get_type};
use super::FuncBinding;
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
    StructReturn,
    NoReturn,
}

#[derive(new)]
pub struct FuncCodegen<'a, 'b, M: cl::Module> {
    pub modules: &'a [b::Module],
    pub builder: Option<cl::FunctionBuilder<'b>>,
    pub obj_module: M,
    pub globals: Globals<'a>,
    pub funcs: HashMap<(usize, usize), FuncBinding>,
    #[new(value = "utils::ScopeStack::empty()")]
    pub scopes: utils::ScopeStack<ScopePayload<'a>>,
    #[new(default)]
    pub values: HashMap<b::ValueIdx, types::RuntimeValue>,
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
impl<'a, M: cl::Module> FuncCodegen<'a, '_, M> {
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
                types::RuntimeValue::new(cl_value.into(), mod_idx, result);
            self.values.insert(result, runtime_value);
        }

        for (v, cl_value) in izip!(params, cl_values) {
            let runtime_value = types::RuntimeValue::new(cl_value.into(), mod_idx, *v);
            self.values.insert(*v, runtime_value);
        }

        expect_builder!(self).switch_to_block(block);
        self.scopes.begin(ScopePayload {
            start_block: block,
            block,
            next_branches: vec![],
            ty: result
                .clone()
                .map(|v| Cow::Borrowed(&self.modules[mod_idx].values[v].ty)),
            result,
        });
    }
    pub fn finish(self) -> (M, Globals<'a>, HashMap<(usize, usize), FuncBinding>) {
        assert!(self.scopes.len() == 1);
        (self.obj_module, self.globals, self.funcs)
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
                let ty = &self.modules[mod_idx].values[*a].ty;

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
            b::InstrBody::If(cond_v, then_, else_) => {
                let builder = expect_builder!(self);
                let cond = self.values[cond_v].add_to_func(&mut self.obj_module, builder);

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
                    let module = &self.modules[mod_idx];
                    let ty = &module.values[instr.results[0]].ty;

                    let next_block = builder.create_block();
                    builder.append_block_param(
                        next_block,
                        get_type(ty, self.modules, &self.obj_module),
                    );

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
                let builder = expect_builder!(self);

                let loop_block = builder.create_block();

                let mut loop_args = vec![];
                for (loop_v, initial_v) in inputs {
                    let initial_runtime_value = self.values[initial_v];
                    let initial_value =
                        initial_runtime_value.add_to_func(&self.obj_module, builder);
                    loop_args.push(initial_value);

                    let native_ty =
                        initial_runtime_value.native_type(self.modules, &self.obj_module);
                    let loop_value = builder.append_block_param(loop_block, native_ty);
                    self.values.insert(
                        *loop_v,
                        types::RuntimeValue::new(loop_value.into(), mod_idx, *loop_v),
                    );
                }
                builder.ins().jump(loop_block, &loop_args);

                let continue_block = builder.create_block();
                let (result, ty) = if instr.results.len() > 0 {
                    let result = instr.results[0];
                    let ty = &self.modules[mod_idx].values[result].ty;
                    let native_ty = types::get_type(ty, self.modules, &self.obj_module);
                    builder.append_block_param(continue_block, native_ty);
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
                let builder = expect_builder!(self);

                let ty = &self.modules[mod_idx].values[*v].ty;
                let mut cl_value =
                    self.values[v].add_to_func(&mut self.obj_module, builder);

                match result_policy {
                    ResultPolicy::Normal => {
                        if let Some(prev_scope) = self.scopes.get(self.scopes.len() - 2) {
                            builder.ins().jump(prev_scope.block, &[cl_value]);
                            cl_value = builder.block_params(prev_scope.block)[0];
                        }
                    }
                    ResultPolicy::Return => {
                        builder.ins().return_(&[cl_value]);
                    }
                    ResultPolicy::StructReturn => {
                        if let Some(res) = self.scopes.last().result {
                            let size =
                                types::get_size(ty, self.modules, &self.obj_module);

                            let res_cl = self.values[&res]
                                .add_to_func(&mut self.obj_module, builder);

                            self.copy_bytes(res_cl, cl_value, size);
                        }
                        expect_builder!(self).ins().return_(&[]);
                    }
                    ResultPolicy::Global => {}
                }

                let scope = self.scopes.last();
                let result = scope.result.unwrap();
                self.values.insert(
                    result,
                    types::RuntimeValue::new(cl_value.into(), mod_idx, result),
                );
            }
            b::InstrBody::Continue(vs) => {
                let builder = expect_builder!(self);
                let block = self
                    .scopes
                    .last_loop()
                    .expect("continue instruction should be called in a loop")
                    .start_block;

                let values = vs
                    .into_iter()
                    .map(|v| self.values[v].add_to_func(&mut self.obj_module, builder))
                    .collect_vec();

                builder.ins().jump(block, &values);
                self.scopes.last_mut().mark_as_never();
            }
            b::InstrBody::Call(func_mod_idx, func_idx, vs) => {
                let builder = expect_builder!(self);

                let args = vs
                    .into_iter()
                    .map(|v| self.values[v].add_to_func(&mut self.obj_module, builder))
                    .collect_vec();

                if let Some(value) = self.call(*func_mod_idx, *func_idx, args) {
                    self.values.insert(
                        instr.results[0],
                        types::RuntimeValue::new(value.into(), mod_idx, instr.results[0]),
                    );
                }
            }
            b::InstrBody::IndirectCall(func_v, vs) => {
                let builder = expect_builder!(self);

                let func = self.values[func_v].clone();

                let mut args = vs
                    .into_iter()
                    .map(|v| self.values[v].add_to_func(&mut self.obj_module, builder))
                    .collect_vec();

                match &func.src {
                    types::ValueSource::AppliedMethod(
                        self_value,
                        (func_mod_idx, func_idx),
                    ) => {
                        args.push(*self_value);

                        if let Some(value) = self.call(*func_mod_idx, *func_idx, args) {
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

                let source_ty = &self.modules[mod_idx].values[*source_v].ty;
                let source = &self.values[source_v];
                let b::Type {
                    body: b::TypeBody::TypeRef(ty_mod_idx, ty_idx),
                    ..
                } = source_ty
                else {
                    panic!("type should be a typeref");
                };
                let b::TypeDefBody::Record(rec) =
                    &self.modules[*ty_mod_idx].typedefs[*ty_idx].body
                else {
                    panic!("type should be a record type");
                };

                let mut offset = 0;
                for (k, v) in &rec.fields {
                    if k == name {
                        break;
                    }
                    offset += get_type(&v.ty, self.modules, &self.obj_module).bytes();
                }

                let source_value = source.add_to_func(&mut self.obj_module, builder);
                let value = builder.ins().load(
                    source.native_type(self.modules, &self.obj_module).clone(),
                    cl::MemFlags::new(),
                    source_value,
                    offset as i32,
                );
                self.values.insert(
                    instr.results[0],
                    types::RuntimeValue::new(value.into(), mod_idx, instr.results[0]),
                );
            }
            b::InstrBody::GetMethod(source_v, name) => {
                let builder = expect_builder!(self);

                let source_ty = &self.modules[mod_idx].values[*source_v].ty;
                let source = &self.values[source_v];
                let b::Type {
                    body: b::TypeBody::TypeRef(ty_mod_idx, ty_idx),
                    ..
                } = source_ty
                else {
                    panic!("type should be a typeref");
                };

                let method = match &self.modules[*ty_mod_idx].typedefs[*ty_idx].body {
                    b::TypeDefBody::Record(rec) => rec
                        .methods
                        .iter()
                        .find(|(k, _)| k == name)
                        .map(|(_, v)| v)
                        .expect("method should be present in record"),
                };

                let value = source.add_to_func(&self.obj_module, builder);
                self.values.insert(
                    instr.results[0],
                    types::RuntimeValue::new(
                        types::ValueSource::AppliedMethod(value, method.func_ref),
                        mod_idx,
                        instr.results[0],
                    ),
                );
            }
            b::InstrBody::ArrayLen(source_v) | b::InstrBody::StrLen(source_v) => {
                let builder = expect_builder!(self);

                let source = self.values[source_v].add_to_func(&self.obj_module, builder);
                let value = builder.ins().load(
                    self.obj_module.isa().pointer_type(),
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
                let source_ty = &self.modules[mod_idx].values[*source_v].ty;
                let source = self.values[source_v].clone();
                let cl_source =
                    source.add_to_func(&mut self.obj_module, expect_builder!(self));

                let (item_size, len) = match &source_ty.body {
                    b::TypeBody::Array(array_ty) => (
                        get_size(&array_ty.item, &self.modules, &self.obj_module),
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
                        .iconst(self.obj_module.isa().pointer_type(), unsafe {
                            mem::transmute::<_, i64>(*idx)
                        });
                    let len = builder.ins().load(
                        self.obj_module.isa().pointer_type(),
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

                let offset =
                    self.obj_module.isa().pointer_bytes() as u64 + idx * item_size as u64;
                let offset_value = builder
                    .ins()
                    .iconst(self.obj_module.isa().pointer_type(), unsafe {
                        mem::transmute::<_, i64>(offset)
                    });
                let value = builder.ins().iadd(cl_source, offset_value);

                self.values.insert(
                    instr.results[0],
                    types::RuntimeValue::new(value.into(), mod_idx, instr.results[0]),
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
                        let module = &self.modules[mod_idx];
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
                                match this.obj_module.isa().pointer_bytes() {
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
                        let (data, module) =
                            this.globals.data_for_string(s, this.obj_module);
                        this.obj_module = module;
                        Some(types::RuntimeValue::new(
                            data.into(),
                            mod_idx,
                            instr.results[0],
                        ))
                    }
                    b::InstrBody::CreateArray(vs) => {
                        let (data, module) = this.globals.data_for_array(
                            vs.iter().map(|v| this.values[v].src).collect_vec(),
                            this.obj_module,
                        );
                        this.obj_module = module;
                        let src = if let Some(data) = data {
                            data.into()
                        } else if this.builder.is_some() {
                            this.create_stack_slot(
                                vs.iter().map(|v| this.values[v]).collect_vec(),
                            )
                            .into()
                        } else {
                            break 'match_b None;
                        };
                        Some(types::RuntimeValue::new(src, mod_idx, instr.results[0]))
                    }
                    b::InstrBody::CreateRecord(fields) => {
                        let module = &self.modules[mod_idx];
                        let ty = &module.values[instr.results[0]].ty;

                        let values = types::tuple_from_record(
                            fields
                                .iter()
                                .map(|(name, v)| (name, this.values[v].clone()))
                                .collect_vec(),
                            ty,
                            this.modules,
                        );
                        let (data, module) = this.globals.data_for_tuple(
                            values.iter().map(|value| value.src).collect_vec(),
                            this.obj_module,
                        );
                        this.obj_module = module;
                        let src = if let Some(data) = data {
                            data.into()
                        } else if this.builder.is_some() {
                            this.create_stack_slot(values).into()
                        } else {
                            break 'match_b None;
                        };
                        Some(types::RuntimeValue::new(src, mod_idx, instr.results[0]))
                    }
                    b::InstrBody::GetGlobal(mod_idx, global_idx) => Some(
                        this.globals
                            .get_global(*mod_idx, *global_idx)
                            .expect("global idx out of range")
                            .value
                            .clone(),
                    ),
                    _ => None,
                }
            };

            if let Some(value) = value {
                this.values.insert(instr.results[0], value);
            }

            (this, value)
        })
    }

    pub fn store_global(&mut self, value: types::RuntimeValue, global: &GlobalBinding) {
        let types::ValueSource::Data(data_id) = &global.value.src else {
            panic!("should never try to store a global that is a const");
        };

        let builder = expect_builder!(self);

        let ty = value.native_type(self.modules, &self.obj_module);
        let value = value.add_to_func(&mut self.obj_module, builder);
        let global_value = self
            .obj_module
            .declare_data_in_func(*data_id, &mut builder.func);
        let ptr = builder.ins().global_value(ty, global_value);
        builder.ins().store(cl::MemFlags::new(), value, ptr, 0);
    }

    pub fn call(
        &mut self,
        func_mod_idx: usize,
        func_idx: usize,
        args: impl Into<Vec<cl::Value>>,
    ) -> Option<cl::Value> {
        let builder = expect_builder!(self);

        let mut args: Vec<_> = args.into();

        let func_id = self.funcs.get(&(func_mod_idx, func_idx)).unwrap().func_id;
        let func = &self.modules[func_mod_idx].funcs[func_idx];

        let ret_ty = &self.modules[func_mod_idx].values[func.ret].ty;
        let ret_policy = if ret_ty.is_never() {
            CallReturnPolicy::NoReturn
        } else if ret_ty.is_aggregate(&self.modules) {
            let size = types::get_size(ret_ty, &self.modules, &self.obj_module);
            let ss_data =
                cl::StackSlotData::new(cl::StackSlotKind::ExplicitSlot, size as u32);
            let ss = builder.create_sized_stack_slot(ss_data);
            let stack_addr =
                builder
                    .ins()
                    .stack_addr(self.obj_module.isa().pointer_type(), ss, 0);
            args.insert(0, stack_addr);

            CallReturnPolicy::StructReturn
        } else {
            CallReturnPolicy::Normal
        };

        self.native_call(func_id, &args, ret_policy)
    }

    pub fn native_call(
        &mut self,
        func_id: cl::FuncId,
        args: &[cl::Value],
        ret_policy: CallReturnPolicy,
    ) -> Option<cl::Value> {
        let builder = expect_builder!(self);

        let func_ref = match self.declared_funcs.get(&func_id) {
            Some(func_ref) => *func_ref,
            None => {
                let func_ref =
                    self.obj_module.declare_func_in_func(func_id, builder.func);
                self.declared_funcs.insert(func_id, func_ref);
                func_ref
            }
        };

        let instr = builder.ins().call(func_ref, args);
        let results = builder.inst_results(instr);
        assert!(results.len() <= 1);

        match ret_policy {
            CallReturnPolicy::Normal => Some(results[0]),
            CallReturnPolicy::StructReturn => Some(args[0]),
            CallReturnPolicy::NoReturn => {
                builder.ins().trap(cl::TrapCode::UnreachableCodeReached);
                self.scopes.last_mut().mark_as_never();
                None
            }
        }
    }

    fn use_value(&mut self, v: b::ValueIdx) -> cl::Value {
        let Some(runtime_value) = self.values.get(&v) else {
            panic!("value should be present in scope: {v}");
        };
        let cl_value =
            runtime_value.add_to_func(&mut self.obj_module, expect_builder!(self));
        cl_value
    }

    fn create_stack_slot(
        &mut self,
        values: impl IntoIterator<Item = types::RuntimeValue>,
    ) -> cl::StackSlot {
        let Some(func) = &mut self.builder else {
            panic!("cannot add stack slot without a function");
        };

        let mut size = 0;
        let values = values
            .into_iter()
            .map(|v| {
                let offset = size;
                size += v.native_type(self.modules, &self.obj_module).bytes();
                (offset, v.add_to_func(&self.obj_module, func))
            })
            .collect_vec();

        let ss_data = cl::StackSlotData::new(cl::StackSlotKind::ExplicitSlot, size);
        let ss = func.create_sized_stack_slot(ss_data);
        for (offset, value) in values {
            func.ins().stack_store(value, ss, offset as i32);
        }

        ss
    }

    fn add_assert(&mut self, cond: cl::Value, code: cl::TrapCode) {
        let builder = expect_builder!(self);
        builder.ins().trapz(cond, code);
    }

    fn copy_bytes(&mut self, dst: cl::Value, src: cl::Value, size: usize) {
        let builder = expect_builder!(self);

        let mut offset: i32 = 0;
        loop {
            let remaining = size - (offset as usize);

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
