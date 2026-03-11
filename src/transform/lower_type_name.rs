use derive_ctor::ctor;

use super::CodeTransformStep;
use crate::bytecode as b;
use crate::context::BuildContext;

#[derive(Clone, Copy, ctor)]
pub struct LowerTypeNameStep<'a> {
    ctx: &'a BuildContext,
}

impl<'a> CodeTransformStep for LowerTypeNameStep<'a> {
    fn transform(&mut self, mod_idx: usize, cursor: &mut b::BlockCursor) {
        let modules = &mut self.ctx.lock_modules_mut();
        let instr = cursor.instr(&modules[mod_idx]).unwrap();
        let &b::InstrBody::TypeName(value_idx) = &instr.body else {
            return;
        };

        let ty = &modules[mod_idx].values[value_idx].ty;
        let type_name = b::printer::format_type_body(&ty.body, &modules);

        let instr = cursor.instr_mut(&mut modules[mod_idx]).unwrap();
        instr.body = b::InstrBody::CreateString(type_name);
    }
}
