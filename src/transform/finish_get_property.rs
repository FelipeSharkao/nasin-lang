use derive_ctor::ctor;

use super::CodeTransformStep;
use crate::bytecode as b;
use crate::context::BuildContext;

#[derive(Clone, Copy, ctor)]
pub struct FinishGetPropertyStep<'a> {
    ctx: &'a BuildContext,
}

impl<'a> CodeTransformStep for FinishGetPropertyStep<'a> {
    #[tracing::instrument(skip(self))]
    fn transform(&mut self, mod_idx: usize, cursor: &mut b::BlockCursor) {
        let modules = &mut self.ctx.lock_modules_mut();
        let module = &modules[mod_idx];

        let Some(instr) = cursor.instr(module) else {
            return;
        };

        let b::InstrBody::GetProperty(source_v, ref key) = instr.body else {
            return;
        };
        let key = key.clone();

        assert_eq!(instr.results.len(), 1, "GetProperty should have one result");

        let loc = instr.loc;
        let result_ty = &module.values[instr.results[0]].ty;
        let parent_ty = &module.values[source_v].ty;
        let is_func = matches!(&result_ty.body, b::TypeBody::Func(..));
        let is_field = parent_ty.field(&key, &modules).is_some();
        let is_method = parent_ty.method(&key, &modules).is_some();

        let Some(instr) = cursor.instr_mut(&mut modules[mod_idx]) else {
            return;
        };
        if is_field {
            instr.body = b::InstrBody::GetField(source_v, key);
        } else if is_method {
            instr.body = b::InstrBody::GetMethod(source_v, key.clone());
            // methods with just the receiver are used as fields, so we have to call them
            if !is_func {
                let result = instr.results[0];

                let parent_ty = &modules[mod_idx].values[source_v].ty;
                let method = parent_ty.method(&key, &modules).unwrap().into_owned();

                let method_v = modules[mod_idx].add_value(b::Value::new(method, loc));
                let instr = cursor.instr_mut(&mut modules[mod_idx]).unwrap();
                instr.results[0] = method_v;

                cursor.step(&modules[mod_idx]);
                cursor.insert_instr(
                    &mut modules[mod_idx],
                    b::Instr::new(b::InstrBody::IndirectCall(method_v, vec![]), loc)
                        .with_results([result]),
                );
            }
        }
    }
}
