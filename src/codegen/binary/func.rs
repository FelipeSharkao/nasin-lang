use std::borrow::Cow;
use std::collections::HashMap;
use std::mem;

use cl::InstBuilder;
use cranelift_shim as cl;
use derive_new::new;
use itertools::{izip, Itertools};

use super::globals::{GlobalBinding, Globals};
use super::types::{self, get_size, get_type, RuntimeValue};
use super::FuncBinding;
use crate::{bytecode as b, utils};

#[derive(new)]
pub struct FuncCodegen<'a, 'b, M: cl::Module> {
    pub modules: &'a [b::Module],
    pub builder: Option<cl::FunctionBuilder<'b>>,
    pub obj_module: M,
    pub globals: Globals<'a>,
    pub funcs: HashMap<(usize, usize), FuncBinding>,
    #[new(value = "utils::ValueStack::new(ScopePayload::default())")]
    pub stack: utils::ValueStack<types::RuntimeValue<'a>, ScopePayload<'a>>,
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
    pub fn create_initial_block(&mut self, params: &'a [b::Param]) {
        let Some(func) = &mut self.builder else {
            panic!("cannot add stack slot without a function");
        };

        let block = func.create_block();
        func.append_block_params_for_function_params(block);
        for (param, value) in izip!(params, func.block_params(block)) {
            self.stack.push(types::RuntimeValue::new(
                Cow::Borrowed(&param.ty),
                (*value).into(),
            ));
        }

        func.switch_to_block(block);
        self.stack.get_scope_mut().payload.block = Some(block);
    }
    pub fn add_instr(&mut self, instr: &'a b::Instr, mod_idx: usize) {
        if self.stack.get_scope().is_never()
            && !matches!(&instr.body, b::InstrBody::End | b::InstrBody::Else)
        {
            return;
        }

        if let Some(value) = self.value_from_instr(instr, mod_idx) {
            self.stack.push(value);
            return;
        }

        match &instr.body {
            b::InstrBody::Add => {
                self.push_bin_op(|func, lhs, rhs, ty| {
                    if ty.is_int() {
                        func.ins().iadd(lhs, rhs)
                    } else if ty.is_float() {
                        func.ins().fadd(lhs, rhs)
                    } else {
                        unreachable!()
                    }
                });
            }
            b::InstrBody::Sub => {
                self.push_bin_op(|func, lhs, rhs, ty| {
                    if ty.is_int() {
                        func.ins().isub(lhs, rhs)
                    } else if ty.is_float() {
                        func.ins().fsub(lhs, rhs)
                    } else {
                        unreachable!()
                    }
                });
            }
            b::InstrBody::Mul => {
                self.push_bin_op(|func, lhs, rhs, ty| {
                    if ty.is_int() {
                        func.ins().imul(lhs, rhs)
                    } else if ty.is_float() {
                        func.ins().fmul(lhs, rhs)
                    } else {
                        unreachable!()
                    }
                });
            }
            b::InstrBody::Div => {
                self.push_bin_op(|func, lhs, rhs, ty| {
                    if ty.is_uint() {
                        func.ins().udiv(lhs, rhs)
                    } else if ty.is_sint() {
                        func.ins().sdiv(lhs, rhs)
                    } else if ty.is_float() {
                        func.ins().fdiv(lhs, rhs)
                    } else {
                        unreachable!()
                    }
                });
            }
            b::InstrBody::Mod => {
                self.push_bin_op(|func, lhs, rhs, ty| {
                    if ty.is_uint() {
                        func.ins().urem(lhs, rhs)
                    } else if ty.is_sint() {
                        func.ins().srem(lhs, rhs)
                    } else if ty.is_float() {
                        let x = func.ins().fdiv(lhs, rhs);
                        let x = func.ins().trunc(x);
                        let y = func.ins().fneg(rhs);
                        func.ins().fma(x, y, lhs)
                    } else {
                        unreachable!()
                    }
                });
            }
            b::InstrBody::Eq
            | b::InstrBody::Neq
            | b::InstrBody::Gt
            | b::InstrBody::Lt
            | b::InstrBody::Gte
            | b::InstrBody::Lte => {
                self.push_bin_op(|func, lhs, rhs, ty| {
                    if ty.is_int() {
                        let cond = match (&instr.body, ty.is_sint()) {
                            (b::InstrBody::Eq, _) => cl::IntCC::Equal,
                            (b::InstrBody::Neq, _) => cl::IntCC::NotEqual,
                            (b::InstrBody::Gt, true) => cl::IntCC::SignedGreaterThan,
                            (b::InstrBody::Gt, false) => cl::IntCC::UnsignedGreaterThan,
                            (b::InstrBody::Lt, true) => cl::IntCC::SignedLessThan,
                            (b::InstrBody::Lt, false) => cl::IntCC::UnsignedLessThan,
                            (b::InstrBody::Gte, true) => {
                                cl::IntCC::SignedGreaterThanOrEqual
                            }
                            (b::InstrBody::Gte, false) => {
                                cl::IntCC::UnsignedGreaterThanOrEqual
                            }
                            (b::InstrBody::Lte, true) => cl::IntCC::SignedLessThanOrEqual,
                            (b::InstrBody::Lte, false) => {
                                cl::IntCC::UnsignedLessThanOrEqual
                            }
                            _ => unreachable!(),
                        };
                        func.ins().icmp(cond, lhs, rhs)
                    } else if ty.is_float() {
                        let cond = match &instr.body {
                            b::InstrBody::Eq => cl::FloatCC::Equal,
                            b::InstrBody::Neq => cl::FloatCC::NotEqual,
                            b::InstrBody::Gt => cl::FloatCC::GreaterThan,
                            b::InstrBody::Lt => cl::FloatCC::LessThan,
                            b::InstrBody::Gte => cl::FloatCC::GreaterThanOrEqual,
                            b::InstrBody::Lte => cl::FloatCC::LessThanOrEqual,
                            _ => unreachable!(),
                        };
                        func.ins().fcmp(cond, lhs, rhs)
                    } else {
                        unreachable!()
                    }
                });
            }
            b::InstrBody::Not => {
                let builder = expect_builder!(self);
                let cond = self.stack.pop().add_to_func(&mut self.obj_module, builder);

                let next_block = builder.create_block();
                builder.append_block_param(next_block, cl::types::I8);

                let v_true = builder.ins().iconst(cl::types::I8, 1);
                let v_false = builder.ins().iconst(cl::types::I8, 0);

                builder
                    .ins()
                    .brif(cond, next_block, &[v_false], next_block, &[v_true]);
                builder.switch_to_block(next_block);

                let block_params = builder.block_params(next_block);
                assert!(block_params.len() == 1);

                self.stack.push(types::RuntimeValue::new(
                    Cow::Owned(b::Type::new(b::TypeBody::Bool, None)),
                    block_params[0].into(),
                ));
            }
            b::InstrBody::If(v) => {
                let builder = expect_builder!(self);
                let cond = self.stack.pop().add_to_func(&mut self.obj_module, builder);

                let then_block = builder.create_block();
                let else_block = builder.create_block();

                builder.ins().brif(cond, then_block, &[], else_block, &[]);
                builder.switch_to_block(then_block);

                let module = &self.modules[mod_idx];
                let ty = &module.values[*v].ty;

                let next_block = builder.create_block();
                builder.append_block_param(
                    next_block,
                    get_type(ty, self.modules, &self.obj_module),
                );

                self.stack.get_scope_mut().payload.block = Some(next_block);

                self.stack.create_scope(ScopePayload {
                    start_block: Some(then_block),
                    block: Some(then_block),
                    branches: vec![else_block],
                    next_block: Some(next_block),
                    ty: Some(Cow::Borrowed(ty)),
                });
            }
            b::InstrBody::Else => {
                let builder = expect_builder!(self);

                let (scope, values) = self.stack.branch_scope();
                let else_block = scope.payload.branches.pop().unwrap();
                scope.payload.start_block = Some(else_block);
                scope.payload.block = Some(else_block);

                if !scope.is_never() {
                    let value = values
                        .last()
                        .unwrap()
                        .add_to_func(&self.obj_module, builder);
                    builder
                        .ins()
                        .jump(scope.payload.next_block.unwrap(), &[value]);
                }

                builder.switch_to_block(else_block);
            }
            b::InstrBody::Loop(n) => {
                todo!();
                //let builder = expect_builder!(self);
                //let args = self.stack.pop_many(*n);
                //let mut args_values = vec![];
                //let mut loop_params = vec![];
                //
                //let loop_block = builder.create_block();
                //for arg in &args {
                //    let value = builder.append_block_param(
                //        loop_block,
                //        get_type(&arg.ty, self.modules, &self.obj_module),
                //    );
                //    args_values.push(arg.add_to_func(&mut self.obj_module, builder));
                //    loop_params
                //        .push(types::RuntimeValue::new(arg.ty.clone(), value.into()))
                //}
                //
                //builder.ins().jump(loop_block, &args_values);
                //builder.switch_to_block(loop_block);
                //
                //let next_block = builder.create_block();
                //builder.append_block_param(
                //    next_block,
                //    get_type(ty, self.modules, &self.obj_module),
                //);
                //
                //self.stack.get_scope_mut().payload.block = Some(next_block);
                //
                //let scope = self.stack.create_scope(ScopePayload {
                //    start_block: Some(loop_block),
                //    block: Some(loop_block),
                //    next_block: Some(next_block),
                //    ty: Some(Cow::Borrowed(ty)),
                //    branches: vec![],
                //});
                //scope.is_loop = true;
                //scope.loop_arity = *n;
                //
                //self.stack.extend(loop_params);
            }
            b::InstrBody::End => {
                let builder = expect_builder!(self);

                let (scope, values) = self.stack.end_scope();
                let next_block = scope.payload.next_block.unwrap();

                if !scope.is_never() {
                    let value = values
                        .last()
                        .unwrap()
                        .add_to_func(&self.obj_module, builder);
                    builder.ins().jump(next_block, &[value]);
                }

                builder.switch_to_block(next_block);

                let block_params = builder.block_params(next_block);
                assert!(block_params.len() <= 1);
                assert!(block_params.is_empty() == scope.payload.ty.is_none());

                if let [value] = block_params {
                    self.stack.push(types::RuntimeValue::new(
                        scope.payload.ty.unwrap(),
                        (*value).into(),
                    ));
                }
            }
            b::InstrBody::Continue => {
                let builder = expect_builder!(self);
                let (block, arity) = {
                    let scope = self
                        .stack
                        .get_loop_scope()
                        .expect("continue instruction should be called in a loop");
                    (scope.payload.start_block.unwrap(), scope.loop_arity)
                };

                let values = self
                    .stack
                    .pop_many(arity)
                    .iter()
                    .map(|arg| arg.add_to_func(&mut self.obj_module, builder))
                    .collect_vec();

                builder.ins().jump(block, &values);
                self.stack.get_scope_mut().mark_as_never();
            }
            b::InstrBody::Call(mod_idx, func_idx) => {
                let func_id = self.funcs.get(&(*mod_idx, *func_idx)).unwrap().func_id;
                let func = &self.modules[*mod_idx].funcs[*func_idx];
                let builder = expect_builder!(self);

                let args = self
                    .stack
                    .pop_many(func.params_desc.len())
                    .into_iter()
                    .map(|arg| arg.add_to_func(&mut self.obj_module, builder))
                    .collect_vec();

                if let Some(value) = self.call(func_id, &args) {
                    self.stack.push(types::RuntimeValue::new(
                        Cow::Borrowed(&func.ret_ty),
                        value.into(),
                    ));
                }
            }
            b::InstrBody::IndirectCall(n) => {
                let builder = expect_builder!(self);

                let func = self.stack.pop();

                let mut args = self
                    .stack
                    .pop_many(*n)
                    .into_iter()
                    .map(|arg| arg.add_to_func(&mut self.obj_module, builder))
                    .collect_vec();

                match &func.src {
                    types::ValueSource::AppliedMethod(
                        self_value,
                        (mod_idx, func_idx),
                    ) => {
                        let func_id =
                            self.funcs.get(&(*mod_idx, *func_idx)).unwrap().func_id;
                        let func = &self.modules[*mod_idx].funcs[*func_idx];

                        args.push(*self_value);

                        if let Some(value) = self.call(func_id, &args) {
                            self.stack.push(types::RuntimeValue::new(
                                Cow::Borrowed(&func.ret_ty),
                                value.into(),
                            ));
                        }
                    }
                    _ => todo!("function as value"),
                }
            }
            b::InstrBody::GetField(name) => {
                let builder = expect_builder!(self);

                let source = self.stack.pop();
                let b::Type {
                    body: b::TypeBody::TypeRef(mod_idx, ty_idx),
                    ..
                } = source.ty.as_ref()
                else {
                    panic!("type should be a typeref");
                };
                let b::TypeDefBody::Record(rec) =
                    &self.modules[*mod_idx].typedefs[*ty_idx].body
                else {
                    panic!("type should be a record type");
                };

                let mut offset = 0;
                let mut field = None;
                for (k, v) in &rec.fields {
                    if k == name {
                        field = Some(v);
                        break;
                    }
                    offset += get_type(&v.ty, self.modules, &self.obj_module).bytes();
                }
                let field = field.expect("field should be present in record");

                let source_value = source.add_to_func(&mut self.obj_module, builder);
                let value = builder.ins().load(
                    source.native_type(self.modules, &self.obj_module).clone(),
                    cl::MemFlags::new(),
                    source_value,
                    offset as i32,
                );
                self.stack.push(types::RuntimeValue::new(
                    Cow::Borrowed(&field.ty),
                    value.into(),
                ));
            }
            b::InstrBody::GetMethod(name) => {
                let builder = expect_builder!(self);

                let source = self.stack.pop();
                let b::Type {
                    body: b::TypeBody::TypeRef(mod_idx, ty_idx),
                    ..
                } = source.ty.as_ref()
                else {
                    panic!("type should be a typeref");
                };

                let method = match &self.modules[*mod_idx].typedefs[*ty_idx].body {
                    b::TypeDefBody::Record(rec) => rec
                        .methods
                        .iter()
                        .find(|(k, _)| k == name)
                        .map(|(_, v)| v)
                        .expect("method should be present in record"),
                };
                let ty = source
                    .ty
                    .property(name, &self.modules)
                    .expect("method should be present in type");

                let value = source.add_to_func(&self.obj_module, builder);
                self.stack.push(types::RuntimeValue::new(
                    Cow::Owned(ty.as_ref().clone()),
                    types::ValueSource::AppliedMethod(value, method.func_ref),
                ))
            }
            b::InstrBody::ArrayLen | b::InstrBody::StrLen => {
                let builder = expect_builder!(self);

                let source = self.stack.pop().add_to_func(&self.obj_module, builder);
                let value = builder.ins().load(
                    self.obj_module.isa().pointer_type(),
                    cl::MemFlags::new(),
                    source,
                    0,
                );
                self.stack.push(RuntimeValue::new(
                    Cow::Owned(b::Type::new(b::TypeBody::USize, None)),
                    types::ValueSource::Value(value),
                ));
            }
            b::InstrBody::ArrayPtr(idx) | b::InstrBody::StrPtr(idx) => {
                let source = self.stack.pop();
                let source_value =
                    source.add_to_func(&mut self.obj_module, expect_builder!(self));

                let (item_size, len) = match &source.ty.body {
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
                        source_value,
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
                let value = builder.ins().iadd(source_value, offset_value);

                let item_ty = match &source.ty.body {
                    b::TypeBody::Array(array_ty) => array_ty.item.clone(),
                    b::TypeBody::String(_) => b::Type::new(b::TypeBody::U8, None).into(),
                    _ => panic!("type should be string or array"),
                };
                self.stack.push(RuntimeValue::new(
                    Cow::Owned(b::Type::new(b::TypeBody::Ptr(item_ty), None)),
                    value.into(),
                ));
            }
            b::InstrBody::Type(_) => {}
            b::InstrBody::GetProperty(..) | b::InstrBody::CompileError => {
                panic!("never should try to compile '{}'", &instr)
            }
            b::InstrBody::Dup(..)
            | b::InstrBody::CreateNumber(..)
            | b::InstrBody::CreateBool(..)
            | b::InstrBody::CreateString(..)
            | b::InstrBody::CreateArray(..)
            | b::InstrBody::CreateRecord(..)
            | b::InstrBody::GetGlobal(..) => unreachable!(),
        }
    }
    pub fn return_value(
        mut self,
    ) -> (M, Globals<'a>, HashMap<(usize, usize), FuncBinding>) {
        let builder = expect_builder!(self);

        let value = self.stack.pop().add_to_func(&mut self.obj_module, builder);
        builder.ins().return_(&[value]);

        (self.obj_module, self.globals, self.funcs)
    }
    pub fn return_never(
        mut self,
    ) -> (M, Globals<'a>, HashMap<(usize, usize), FuncBinding>) {
        let func = expect_builder!(self);

        func.ins().trap(cl::TrapCode::UnreachableCodeReached);

        (self.obj_module, self.globals, self.funcs)
    }
    pub fn value_from_instr(
        &mut self,
        instr: &'a b::Instr,
        mod_idx: usize,
    ) -> Option<types::RuntimeValue<'a>> {
        utils::replace_with(self, |mut this| {
            let v = 'match_b: {
                match &instr.body {
                    b::InstrBody::Dup(n) => Some(this.stack.get(*n).unwrap().clone()),
                    b::InstrBody::CreateNumber(n) => {
                        let module = &self.modules[mod_idx];
                        let ty = &module.values[instr.results[0]].ty;

                        macro_rules! parse_num {
                            ($ty:ty, $variant:ident) => {{
                                let value: $ty = n.parse().unwrap();
                                let src = types::ValueSource::$variant(unsafe {
                                    mem::transmute(value)
                                });
                                Some(types::RuntimeValue::new(Cow::Borrowed(ty), src))
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
                        Cow::Owned(b::Type::new(b::TypeBody::Bool, None)),
                        (*b as u8).into(),
                    )),
                    b::InstrBody::CreateString(s) => {
                        let (data, module) =
                            this.globals.data_for_string(s, this.obj_module);
                        this.obj_module = module;
                        Some(types::RuntimeValue::new(
                            Cow::Owned(b::Type::new(
                                b::TypeBody::String(b::StringType::new(Some(s.len()))),
                                None,
                            )),
                            data.into(),
                        ))
                    }
                    b::InstrBody::CreateArray(n) => {
                        let module = &self.modules[mod_idx];
                        let ty = &module.values[instr.results[0]].ty;

                        let values = this.stack.pop_many(*n);
                        let (data, module) =
                            this.globals.data_for_array(values.clone(), this.obj_module);
                        this.obj_module = module;
                        let src = if let Some(data) = data {
                            data.into()
                        } else if this.builder.is_some() {
                            this.create_stack_slot(&values).into()
                        } else {
                            break 'match_b None;
                        };
                        Some(types::RuntimeValue::new(Cow::Borrowed(ty), src))
                    }
                    b::InstrBody::CreateRecord(fields) => {
                        let module = &self.modules[mod_idx];
                        let ty = &module.values[instr.results[0]].ty;

                        let values = types::tuple_from_record(
                            izip!(fields, this.stack.pop_many(fields.len())),
                            ty,
                            this.modules,
                        );
                        let (data, module) =
                            this.globals.data_for_tuple(values.clone(), this.obj_module);
                        this.obj_module = module;
                        let src = if let Some(data) = data {
                            data.into()
                        } else if this.builder.is_some() {
                            this.create_stack_slot(&values).into()
                        } else {
                            break 'match_b None;
                        };
                        Some(types::RuntimeValue::new(Cow::Borrowed(ty), src))
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

            (this, v)
        })
    }
    pub fn store_global(&mut self, value: RuntimeValue, global: &GlobalBinding) {
        let types::ValueSource::Data(data_id) = &global.value.src else {
            panic!("should never try to store a global that is a const");
        };

        let builder = expect_builder!(self);

        let ty = types::get_type(&value.ty, self.modules, &self.obj_module);
        let value = value.add_to_func(&mut self.obj_module, builder);
        let gv = self
            .obj_module
            .declare_data_in_func(*data_id, &mut builder.func);
        let ptr = builder.ins().global_value(ty, gv);
        builder.ins().store(cl::MemFlags::new(), value, ptr, 0);
    }
    pub fn call(&mut self, func_id: cl::FuncId, args: &[cl::Value]) -> Option<cl::Value> {
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

        let instr = builder.ins().call(func_ref, &args);
        let results = builder.inst_results(instr);
        assert!(results.len() <= 1);

        if results.is_empty() {
            None
        } else {
            Some(results[0])
        }
    }

    fn create_stack_slot(&mut self, values: &[types::RuntimeValue]) -> cl::StackSlot {
        let Some(func) = &mut self.builder else {
            panic!("cannot add stack slot without a function");
        };

        let mut size = 0;
        let values = values
            .iter()
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
    fn push_bin_op(
        &mut self,
        f: impl FnOnce(&mut cl::FunctionBuilder, cl::Value, cl::Value, &b::Type) -> cl::Value,
    ) {
        let operands = self.stack.pop_many(2);
        assert!(operands[0].ty == operands[1].ty);

        let lhs = operands[0].add_to_func(&mut self.obj_module, expect_builder!(self));
        let rhs = operands[1].add_to_func(&mut self.obj_module, expect_builder!(self));

        let value = f(expect_builder!(self), lhs, rhs, operands[0].ty.as_ref());
        self.stack.push(types::RuntimeValue::new(
            operands[0].ty.clone(),
            value.into(),
        ));
    }
    fn add_assert(&mut self, cond: cl::Value, code: cl::TrapCode) {
        let builder = expect_builder!(self);
        builder.ins().trapz(cond, code);
    }
}

#[derive(Debug, Default)]
pub struct ScopePayload<'a> {
    pub start_block: Option<cl::Block>,
    pub block: Option<cl::Block>,
    pub next_block: Option<cl::Block>,
    pub branches: Vec<cl::Block>,
    pub ty: Option<Cow<'a, b::Type>>,
}
