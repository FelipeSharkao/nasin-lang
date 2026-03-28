use std::collections::VecDeque;
use std::fmt;

use derive_ctor::ctor;

use super::instr::*;
use super::module::*;

/// A position within a block's instruction list. Provides methods for reading,
/// writing, and navigating instructions within the block.
#[derive(Debug, Clone)]
pub struct BlockCursor {
    frames: Vec<BlockCursorFrame>,
}

impl BlockCursor {
    /// Creates a new cursor for the given block index. The cursor will not be moved to
    /// the first instruction until `step()` or `step_over()` is called.
    pub fn new(block_idx: BlockIdx) -> Self {
        Self {
            frames: vec![BlockCursorFrame::new(block_idx)],
        }
    }

    /// Returns the instruction at the current position, if any.
    pub fn instr<'m>(&self, module: &'m Module) -> Option<&'m Instr> {
        let frame = self.frames.last()?;
        module
            .blocks
            .get(frame.block_idx)?
            .body
            .get(frame.instr_idx?)
    }

    /// Returns the instruction at the current position, if any.
    pub fn instr_mut<'m>(&self, module: &'m mut Module) -> Option<&'m mut Instr> {
        let frame = self.frames.last()?;
        module
            .blocks
            .get_mut(frame.block_idx)?
            .body
            .get_mut(frame.instr_idx?)
    }

    /// Moves the cursor to the next instruction, if any. If the next instruction has
    /// nested blocks, the cursor will move to the first nested block and queue the
    /// others. if more than one. Returns `true` if the cursor was moved to a valid
    /// instruction.
    pub fn step(&mut self, module: &Module) -> bool {
        if self.step_over(module) {
            if self.step_in(module) {
                // `step_in` will move the cursor to the first nested block, but won't
                // start the frame, so we need to step to the first instruction of that
                // block, and maybe step in again (depth-first)
                return self.step(module);
            }
            true
        } else if self.step_out(module) {
            // `step_out` will move the cursor to the outer block, at the same instruction
            // that did `step_in`, which would cause an infinite loop, so we need to step
            // to the next instruction
            self.step(module)
        } else {
            false
        }
    }

    /// Moves the cursor to the next instruction, if any. The cursor will not move into
    /// nested blocks, and will not step out of the current block when it reaches the end.
    /// Returns `true` if the cursor was moved to a valid instruction.
    pub fn step_over(&mut self, module: &Module) -> bool {
        let Some(frame) = self.frames.last_mut() else {
            return false;
        };
        let instr_idx = frame.step_over();
        instr_idx < module.blocks[frame.block_idx].body.len()
    }

    /// Moves the cursor to the first nested block and queue the others, if more than one.
    /// The cursor will not move to the first instruction of the nested block until
    /// `step()` or `step_over()` is called. Returns `true` if the cursor was moved to a
    /// valid instruction.
    pub fn step_in(&mut self, module: &Module) -> bool {
        let mut next_branches = VecDeque::new();
        for nested_block in self.nested_blocks(module) {
            if module.blocks[nested_block].body.is_empty() {
                continue;
            }
            next_branches.push_back(nested_block);
        }
        let Some(block_idx) = next_branches.pop_front() else {
            return false;
        };
        let mut frame = BlockCursorFrame::new(block_idx);
        frame.next_branches = next_branches;
        self.frames.push(frame);
        true
    }

    /// Moves the cursor out the current block and to the next queued instruction. If the
    /// current block was the starting block, and there's no queued instruction, returns
    /// `false`.
    pub fn step_out(&mut self, module: &Module) -> bool {
        let Some(frame) = self.frames.last_mut() else {
            return false;
        };
        if let Some(next_branch) = frame.next_branches.pop_front() {
            frame.block_idx = next_branch;
            frame.instr_idx = None;
            return true;
        }
        while self.frames.len() > 1 {
            self.frames.pop();
            let frame = self.frames.last_mut().unwrap();
            if frame
                .instr_idx
                .is_none_or(|x| x < module.blocks[frame.block_idx].body.len())
            {
                return true;
            }
        }
        false
    }

    /// Inserts an instruction at the current position
    pub fn insert_instr(&mut self, module: &mut Module, instr: Instr) {
        let frame = self.frames.last_mut().unwrap();
        let Some(instr_idx) = frame.instr_idx else {
            panic!("Cannot insert instruction before starting the cursor");
        };

        let block = &mut module.blocks[frame.block_idx];
        block.body.insert(instr_idx.min(block.body.len()), instr);
    }

    fn nested_blocks(&self, module: &Module) -> Vec<BlockIdx> {
        let mut res = vec![];

        match self.instr(module).map(|instr| &instr.body) {
            Some(InstrBody::If(_, then_block, else_block)) => {
                res.push(*then_block);
                res.push(*else_block);
            }
            Some(InstrBody::Loop(_, body_block)) => res.push(*body_block),
            _ => {}
        }

        res
    }
}

#[derive(Clone, ctor)]
struct BlockCursorFrame {
    block_idx:     BlockIdx,
    #[ctor(default)]
    instr_idx:     Option<usize>,
    #[ctor(default)]
    next_branches: VecDeque<BlockIdx>,
}

impl BlockCursorFrame {
    /// Moves the frame to the next instruction. Returns the index of the instruction.
    /// This don't do any checks, so it's up to the caller to check if the instruction is
    /// valid.
    fn step_over(&mut self) -> usize {
        let instr_idx = self.instr_idx.map_or(0, |x| x + 1);
        self.instr_idx = Some(instr_idx);
        instr_idx
    }
}

impl fmt::Debug for BlockCursorFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{ block {}", self.block_idx)?;
        if let Some(instr_idx) = self.instr_idx {
            write!(f, ", instr {instr_idx}")?;
        }
        if !self.next_branches.is_empty() {
            write!(f, ", next_branches [")?;
            for (i, next_branch) in self.next_branches.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", next_branch)?;
            }
            write!(f, "]")?;
        }
        write!(f, " }}")
    }
}
