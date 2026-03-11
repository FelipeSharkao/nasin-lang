use super::instr::*;
use super::module::*;

/// A position within a block's instruction list. Provides methods for reading,
/// writing, and navigating instructions within the block.
#[derive(Debug, Clone, Copy)]
pub struct BlockCursor {
    pub block_idx: BlockIdx,
    instr_idx:     usize,
}

impl BlockCursor {
    pub fn new(block_idx: BlockIdx) -> Self {
        Self {
            block_idx,
            instr_idx: 0,
        }
    }

    pub fn instr_idx(&self) -> usize {
        self.instr_idx
    }

    pub fn has_next(&self, module: &Module) -> bool {
        self.instr_idx < module.block(self.block_idx).len()
    }

    pub fn instr<'m>(&self, module: &'m Module) -> &'m Instr {
        &module.block(self.block_idx)[self.instr_idx]
    }

    pub fn instr_mut<'m>(&self, module: &'m mut Module) -> &'m mut Instr {
        &mut module.block_mut(self.block_idx)[self.instr_idx]
    }

    pub fn advance(&mut self) {
        self.instr_idx += 1;
    }

    pub fn shift(&mut self, offset: isize) {
        let idx = self.instr_idx as isize + offset;
        self.instr_idx = idx.max(0) as usize;
    }

    pub fn insert_instr(&mut self, module: &mut Module, instr: Instr) {
        module
            .block_mut(self.block_idx)
            .insert(self.instr_idx, instr);
        self.advance();
    }

    /// Returns the sub-blocks referenced by the current instruction, if any.
    /// For `If`: returns `(then_block, Some(else_block))`.
    /// For `Loop`: returns `(body_block, None)`.
    pub fn sub_blocks(&self, module: &Module) -> Option<(BlockIdx, Option<BlockIdx>)> {
        match &self.instr(module).body {
            InstrBody::If(_, then_block, else_block) => {
                Some((*then_block, Some(*else_block)))
            }
            InstrBody::Loop(_, body_block) => Some((*body_block, None)),
            _ => None,
        }
    }
}

/// Depth-first visit of all instructions in a block tree. Visits sub-blocks
/// before the parent instruction (pre-order on sub-blocks).
///
/// The visitor receives the module index, a cursor positioned at the current
/// instruction, and a shared reference to the modules slice.
pub fn visit_block(
    modules: &[Module],
    mod_idx: usize,
    block_idx: BlockIdx,
    visitor: &mut impl FnMut(usize, &BlockCursor, &[Module]),
) {
    let mut cursor = BlockCursor::new(block_idx);
    while cursor.has_next(&modules[mod_idx]) {
        if let Some((first, second)) = cursor.sub_blocks(&modules[mod_idx]) {
            visit_block(modules, mod_idx, first, visitor);
            if let Some(second) = second {
                visit_block(modules, mod_idx, second, visitor);
            }
        }
        visitor(mod_idx, &cursor, modules);
        cursor.advance();
    }
}
