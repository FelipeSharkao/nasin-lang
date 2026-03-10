mod finish_dispatch;
mod finish_get_property;
mod instantiate_generic_funcs;
mod lower_type_name;

use derive_ctor::ctor;
use derive_more::Debug;
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
                self.transform_body(
                    &mut step,
                    mod_idx,
                    &mut GlobalCodeTransformCursor::new(mod_idx, global_idx),
                );
            }
            for func_idx in 0..({ self.ctx.lock_modules()[mod_idx].funcs.len() }) {
                let is_generic = {
                    let modules = &self.ctx.lock_modules();
                    let func = &modules[mod_idx].funcs[func_idx];
                    func.generics.len() > 0
                };
                if is_generic {
                    tracing::trace!(mod_idx, func_idx, "skipping generic function");
                    continue;
                }
                tracing::trace!(mod_idx, func_idx, "transforming function");
                self.transform_body(
                    &mut step,
                    mod_idx,
                    &mut FuncCodeTransformCursor::new(mod_idx, func_idx),
                );
            }
        }
    }

    #[tracing::instrument(skip(self, step))]
    fn transform_body(
        &self,
        step: &mut impl CodeTransformStep,
        mod_idx: usize,
        cursor: &mut dyn CodeTransformCursor,
    ) {
        loop {
            enum NestedBodyKind {
                If,
                Loop,
            }

            let nested_kind = {
                let modules = &self.ctx.lock_modules();
                if !cursor.has_next(modules) {
                    break;
                }

                match &cursor.get_instr(modules).body {
                    b::InstrBody::If(_, _, _) => Some(NestedBodyKind::If),
                    b::InstrBody::Loop(_, _) => Some(NestedBodyKind::Loop),
                    _ => None,
                }
            };

            match nested_kind {
                Some(NestedBodyKind::If) => {
                    tracing::trace!("transforming if-then");
                    self.transform_body(
                        step,
                        mod_idx,
                        &mut NestedCodeTransformCursor::new(
                            cursor,
                            |instrs| match &instrs[cursor.instr_idx()].body {
                                b::InstrBody::If(_, then_body, _) => then_body,
                                _ => unreachable!(),
                            },
                            |instrs| match &mut instrs[cursor.instr_idx()].body {
                                b::InstrBody::If(_, then_body, _) => then_body,
                                _ => unreachable!(),
                            },
                        ),
                    );
                    tracing::trace!("transforming if-else");
                    self.transform_body(
                        step,
                        mod_idx,
                        &mut NestedCodeTransformCursor::new(
                            cursor,
                            |instrs| match &instrs[cursor.instr_idx()].body {
                                b::InstrBody::If(_, _, else_body) => else_body,
                                _ => unreachable!(),
                            },
                            |instrs| match &mut instrs[cursor.instr_idx()].body {
                                b::InstrBody::If(_, _, else_body) => else_body,
                                _ => unreachable!(),
                            },
                        ),
                    );
                }
                Some(NestedBodyKind::Loop) => {
                    tracing::trace!("transforming loop");
                    self.transform_body(
                        step,
                        mod_idx,
                        &mut NestedCodeTransformCursor::new(
                            cursor,
                            |instrs| match &instrs[cursor.instr_idx()].body {
                                b::InstrBody::Loop(_, body) => body,
                                _ => unreachable!(),
                            },
                            |instrs| match &mut instrs[cursor.instr_idx()].body {
                                b::InstrBody::Loop(_, body) => body,
                                _ => unreachable!(),
                            },
                        ),
                    );
                }
                None => {}
            }

            tracing::trace!("transforming instruction");
            step.transform(mod_idx, cursor);
            cursor.shift(1);
        }
    }
}

pub trait CodeTransformStep {
    fn transform(&mut self, mod_idx: usize, cursor: &mut dyn CodeTransformCursor);
}

pub trait CodeTransformCursor: Debug {
    fn instr_idx(&self) -> usize;
    fn set_instr_idx(&mut self, idx: usize);
    fn get_body<'m>(&self, modules: &'m [b::Module]) -> &'m [b::Instr];
    fn get_body_mut<'m>(&self, modules: &'m mut [b::Module]) -> &'m mut Vec<b::Instr>;

    fn get_instr<'m>(&self, modules: &'m [b::Module]) -> &'m b::Instr {
        &self.get_body(modules)[self.instr_idx()]
    }

    fn get_instr_mut<'m>(&mut self, modules: &'m mut Vec<b::Module>) -> &'m mut b::Instr {
        &mut self.get_body_mut(modules)[self.instr_idx()]
    }

    fn has_next(&self, modules: &[b::Module]) -> bool {
        self.instr_idx() < self.get_body(modules).len()
    }

    #[tracing::instrument()]
    fn shift(&mut self, offset: isize) {
        let mut idx = self.instr_idx() as isize + offset;
        if idx < 0 {
            idx = 0;
        }
        self.set_instr_idx(idx as usize);
    }

    #[tracing::instrument()]
    fn insert_instr(&mut self, modules: &mut Vec<b::Module>, instr: b::Instr) {
        self.get_body_mut(modules).insert(self.instr_idx(), instr);
        self.shift(1);
    }
}

#[derive(Clone, Copy, Debug, ctor)]
struct GlobalCodeTransformCursor {
    mod_idx:    usize,
    global_idx: usize,
    #[ctor(default)]
    instr_idx:  usize,
}

impl CodeTransformCursor for GlobalCodeTransformCursor {
    fn instr_idx(&self) -> usize {
        self.instr_idx
    }

    fn set_instr_idx(&mut self, idx: usize) {
        self.instr_idx = idx;
    }

    fn get_body<'m>(&self, modules: &'m [b::Module]) -> &'m [b::Instr] {
        &modules[self.mod_idx].globals[self.global_idx].body
    }

    fn get_body_mut<'m>(&self, modules: &'m mut [b::Module]) -> &'m mut Vec<b::Instr> {
        &mut modules[self.mod_idx].globals[self.global_idx].body
    }
}

#[derive(Clone, Copy, Debug, ctor)]
struct FuncCodeTransformCursor {
    mod_idx:   usize,
    func_idx:  usize,
    #[ctor(default)]
    instr_idx: usize,
}

impl CodeTransformCursor for FuncCodeTransformCursor {
    fn instr_idx(&self) -> usize {
        self.instr_idx
    }

    fn set_instr_idx(&mut self, idx: usize) {
        self.instr_idx = idx;
    }

    fn get_body<'m>(&self, modules: &'m [b::Module]) -> &'m [b::Instr] {
        &modules[self.mod_idx].funcs[self.func_idx].body
    }

    fn get_body_mut<'m>(&self, modules: &'m mut [b::Module]) -> &'m mut Vec<b::Instr> {
        &mut modules[self.mod_idx].funcs[self.func_idx].body
    }
}

#[derive(Clone, Copy, Debug, ctor)]
struct NestedCodeTransformCursor<
    'a,
    F: for<'m> Fn(&'m [b::Instr]) -> &'m [b::Instr],
    FMut: for<'m> Fn(&'m mut Vec<b::Instr>) -> &'m mut Vec<b::Instr>,
> {
    cursor: &'a dyn CodeTransformCursor,
    #[debug(skip)]
    f: F,
    #[debug(skip)]
    fmut: FMut,
    #[ctor(default)]
    instr_idx: usize,
}

impl<
    'a,
    F: for<'m> Fn(&'m [b::Instr]) -> &'m [b::Instr],
    FMut: for<'m> Fn(&'m mut Vec<b::Instr>) -> &'m mut Vec<b::Instr>,
> CodeTransformCursor for NestedCodeTransformCursor<'a, F, FMut>
{
    fn instr_idx(&self) -> usize {
        self.instr_idx
    }

    fn set_instr_idx(&mut self, idx: usize) {
        self.instr_idx = idx;
    }

    fn get_body<'m>(&self, modules: &'m [b::Module]) -> &'m [b::Instr] {
        (self.f)(self.cursor.get_body(modules))
    }

    fn get_body_mut<'m>(&self, modules: &'m mut [b::Module]) -> &'m mut Vec<b::Instr> {
        (self.fmut)(self.cursor.get_body_mut(modules))
    }
}
