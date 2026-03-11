mod finish_dispatch;
mod finish_get_property;
mod instantiate_generic_funcs;
mod lower_type_name;

use derive_ctor::ctor;
pub use finish_dispatch::FinishDispatchStep;
pub use finish_get_property::FinishGetPropertyStep;
pub use instantiate_generic_funcs::InstantiateGenericFuncsStep;
pub use lower_type_name::LowerTypeNameStep;

use crate::bytecode as b;
use crate::context::BuildContext;

#[derive(ctor)]
pub struct CodeTransform<'a> {
    ctx: &'a BuildContext,
}

impl<'a> CodeTransform<'a> {
    #[tracing::instrument(skip(self, step))]
    pub fn apply(&self, mut step: impl CodeTransformStep) {
        for mod_idx in 0..({ self.ctx.lock_modules().len() }) {
            for global_idx in 0..({ self.ctx.lock_modules()[mod_idx].globals.len() }) {
                tracing::trace!(mod_idx, global_idx, "transforming global");
                let block_idx =
                    { self.ctx.lock_modules()[mod_idx].globals[global_idx].body };
                self.transform_block(&mut step, mod_idx, block_idx);
            }
            for func_idx in 0..({ self.ctx.lock_modules()[mod_idx].funcs.len() }) {
                let (is_generic, block_idx) = {
                    let modules = &self.ctx.lock_modules();
                    let func = &modules[mod_idx].funcs[func_idx];
                    (func.generics.len() > 0, func.body)
                };
                if is_generic {
                    tracing::trace!(mod_idx, func_idx, "skipping generic function");
                    continue;
                }
                tracing::trace!(mod_idx, func_idx, "transforming function");
                self.transform_block(&mut step, mod_idx, block_idx);
            }
        }
    }

    #[tracing::instrument(skip(self, step))]
    fn transform_block(
        &self,
        step: &mut impl CodeTransformStep,
        mod_idx: usize,
        block_idx: b::BlockIdx,
    ) {
        let mut cursor = b::BlockCursor::new(block_idx);
        if cursor.is_done(&self.ctx.lock_modules()[mod_idx]) {
            return;
        }
        loop {
            tracing::trace!("transforming instruction");
            step.transform(mod_idx, &mut cursor);

            if !cursor.step(&self.ctx.lock_modules()[mod_idx]) {
                break;
            }
        }
    }
}

pub trait CodeTransformStep {
    fn transform(&mut self, mod_idx: usize, cursor: &mut b::BlockCursor);
}
