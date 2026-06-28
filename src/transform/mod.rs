mod finish_dispatch;
mod finish_get_property;
mod instantiate_generic_funcs;
mod lower_type_name;

use std::collections::VecDeque;

use derive_ctor::ctor;
pub use finish_dispatch::FinishDispatchStep;
pub use finish_get_property::FinishGetPropertyStep;
pub use instantiate_generic_funcs::InstantiateGenericFuncsStep;
pub use lower_type_name::LowerTypeNameStep;

use crate::bytecode as b;
use crate::context::BuildContext;

#[derive(ctor)]
pub struct CodeTransform<'a> {
    ctx:         &'a BuildContext,
    #[ctor(default)]
    added_funcs: VecDeque<(usize, usize)>,
}

impl<'a> CodeTransform<'a> {
    #[tracing::instrument(skip(self, step))]
    pub fn apply(&mut self, mut step: impl CodeTransformStep) {
        for mod_idx in 0..({ self.ctx.lock_modules().len() }) {
            tracing::trace!(mod_idx, "transforming module");

            for global_idx in 0..({ self.ctx.lock_modules()[mod_idx].globals.len() }) {
                self.transform_global(&mut step, mod_idx, global_idx);
            }

            for func_idx in 0..({ self.ctx.lock_modules()[mod_idx].funcs.len() }) {
                self.transform_func(&mut step, mod_idx, func_idx);
            }

            tracing::trace!(mod_idx, "transforming module done");
        }

        while let Some((mod_idx, func_idx)) = self.added_funcs.pop_front() {
            self.transform_func(&mut step, mod_idx, func_idx);
        }
    }

    #[tracing::instrument(skip(self, step))]
    pub fn transform_global(
        &mut self,
        step: &mut impl CodeTransformStep,
        mod_idx: usize,
        global_idx: usize,
    ) {
        let block_idx = { self.ctx.lock_modules()[mod_idx].globals[global_idx].body };
        self.transform_block(step, mod_idx, block_idx);
    }

    #[tracing::instrument(skip(self, step))]
    fn transform_func(
        &mut self,
        step: &mut impl CodeTransformStep,
        mod_idx: usize,
        func_idx: usize,
    ) {
        let (is_generic, block_idx) = {
            let modules = &self.ctx.lock_modules();
            let func = &modules[mod_idx].funcs[func_idx];
            (func.generics.len() > 0, func.body)
        };
        if is_generic {
            tracing::trace!("skipping generic function");
            return;
        }
        self.transform_block(step, mod_idx, block_idx);
    }

    #[tracing::instrument(skip(self, step))]
    fn transform_block(
        &mut self,
        step: &mut impl CodeTransformStep,
        mod_idx: usize,
        block_idx: b::BlockIdx,
    ) {
        let mut cursor = b::BlockCursor::new(block_idx);
        while cursor.step(&self.ctx.lock_modules()[mod_idx]) {
            tracing::trace!("transforming instruction");
            step.transform(mod_idx, &mut cursor);
        }

        self.added_funcs.extend(cursor.added_funcs);
    }
}

pub trait CodeTransformStep {
    fn transform(&mut self, mod_idx: usize, cursor: &mut b::BlockCursor);
}
