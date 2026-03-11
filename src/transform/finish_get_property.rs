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
        let Some(instr) = cursor.instr(&modules[mod_idx]) else {
            return;
        };
        let (source_v, key) = match &instr.body {
            b::InstrBody::GetProperty(v, key) => (*v, key.clone()),
            _ => return,
        };

        let parent_ty = &modules[mod_idx].values[source_v].ty;
        let is_field = parent_ty.field(&key, &modules).is_some();
        let is_method = parent_ty.method(&key, &modules).is_some();

        let Some(instr) = cursor.instr_mut(&mut modules[mod_idx]) else {
            return;
        };
        if is_field {
            instr.body = b::InstrBody::GetField(source_v, key);
        } else if is_method {
            instr.body = b::InstrBody::GetMethod(source_v, key);
        }
    }
}
