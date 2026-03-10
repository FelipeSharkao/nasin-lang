use derive_ctor::ctor;

use super::{CodeTransformCursor, CodeTransformStep};
use crate::bytecode as b;
use crate::context::BuildContext;

#[derive(Clone, Copy, ctor)]
pub struct FinishGetPropertyStep<'a> {
    ctx: &'a BuildContext,
}

impl<'a> CodeTransformStep for FinishGetPropertyStep<'a> {
    #[tracing::instrument(skip(self))]
    fn transform(&mut self, mod_idx: usize, cursor: &mut dyn CodeTransformCursor) {
        let (source_v, key) = {
            let modules = &self.ctx.lock_modules();
            let instr = cursor.get_instr(modules);
            match &instr.body {
                b::InstrBody::GetProperty(v, key) => (*v, key.clone()),
                _ => return,
            }
        };

        let (is_field, is_method) = {
            let modules = &self.ctx.lock_modules();
            let parent_ty = &modules[mod_idx].values[source_v].ty;
            (
                parent_ty.field(&key, &modules).is_some(),
                parent_ty.method(&key, &modules).is_some(),
            )
        };

        {
            let modules = &mut self.ctx.lock_modules_mut();
            let instr = cursor.get_instr_mut(modules);
            if is_field {
                instr.body = b::InstrBody::GetField(source_v, key.clone());
            } else if is_method {
                instr.body = b::InstrBody::GetMethod(source_v, key.clone());
            }
        }
    }
}
