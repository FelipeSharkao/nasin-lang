use std::collections::VecDeque;

use derive_ctor::ctor;
use genawaiter::rc::r#gen as rc_gen;
use genawaiter::yield_;
use itertools::Itertools;

use super::instr::*;
use super::module::*;

/// A position within a block's instruction list. Provides methods for reading,
/// writing, and navigating instructions within the block.
#[derive(Debug, Clone)]
pub struct BlockCursor {
    frames: Vec<BlockCursorFrame>,
    frames_queue: VecDeque<BlockCursorFrame>,
}

impl BlockCursor {
    pub fn new(block_idx: BlockIdx) -> Self {
        Self {
            frames: vec![BlockCursorFrame::new(block_idx)],
            frames_queue: VecDeque::new(),
        }
    }

    /// Returns `true` if the cursor is over the last instruction of the block tree. If
    /// this function returns `true`, `instr()` and `instr_mut()` will return `None`, and
    /// all the `step*()` functions will return `false`.
    pub fn is_done(&self, module: &Module) -> bool {
        let Some(frame) = self.frames.last() else {
            return self.frames_queue.is_empty();
        };
        frame.instr_idx >= module.blocks[frame.block_idx].body.len()
    }

    /// Returns the instruction at the current position, if any.
    pub fn instr<'m>(&self, module: &'m Module) -> Option<&'m Instr> {
        let frame = self.frames.last()?;
        module
            .blocks
            .get(frame.block_idx)?
            .body
            .get(frame.instr_idx)
    }

    /// Returns the instruction at the current position, if any.
    pub fn instr_mut<'m>(&self, module: &'m mut Module) -> Option<&'m mut Instr> {
        let frame = self.frames.last()?;
        module
            .blocks
            .get_mut(frame.block_idx)?
            .body
            .get_mut(frame.instr_idx)
    }

    /// Moves the cursor to the next instruction, if any. If the next instruction has
    /// nested blocks, the cursor will move to the first nested block and queue the others
    /// if more than one. Returns `true` if the cursor was moved to a valid instruction.
    pub fn step(&mut self, module: &Module) -> bool {
        if !self.step_over(module) {
            return false;
        }
        for sub_block in self.nested_blocks(module).collect_vec() {
            if module.blocks[sub_block].body.is_empty() {
                continue;
            }
            self.frames_queue.push_back(BlockCursorFrame {
                block_idx: sub_block,
                instr_idx: 0,
            });
        }
        if let Some(frame) = self.frames_queue.pop_front() {
            self.frames.push(frame);
        }
        true
    }

    /// Moves the cursor to the next instruction, if any. The cursor will not move into
    /// nested blocks. Returns `true` if the cursor was moved to a valid instruction.
    pub fn step_over(&mut self, module: &Module) -> bool {
        let Some(frame) = self.frames.last_mut() else {
            return self.step_out(module);
        };
        frame.instr_idx += 1;
        if frame.instr_idx >= module.blocks[frame.block_idx].body.len() {
            if !self.step_out(module) {
                return false;
            }
        }
        true
    }

    /// Moves the cursor out the current block and to the next queued instruction. If the
    /// current block was the starting block, and there's no queued instruction, returns
    /// `false`.
    pub fn step_out(&mut self, module: &Module) -> bool {
        if let Some(frame) = self.frames_queue.pop_front() {
            self.frames.pop();
            self.frames.push(frame);
            return true;
        }
        while self.frames.len() > 1 {
            self.frames.pop();
            let frame = self.frames.last_mut().unwrap();
            frame.instr_idx += 1;
            if frame.instr_idx < module.blocks[frame.block_idx].body.len() {
                return true;
            }
        }
        false
    }

    /// Inserts an instruction at the current position
    pub fn insert_instr(&mut self, module: &mut Module, instr: Instr) {
        let frame = self.frames.last_mut().unwrap();
        let block = &mut module.blocks[frame.block_idx];
        block
            .body
            .insert(frame.instr_idx.min(block.body.len()), instr);
    }

    fn nested_blocks(&self, module: &Module) -> impl Iterator<Item = BlockIdx> {
        rc_gen!({
            let Some(instr) = self.instr(module) else {
                return;
            };
            match &instr.body {
                InstrBody::If(_, then_block, else_block) => {
                    yield_!(*then_block);
                    yield_!(*else_block);
                }
                InstrBody::Loop(_, body_block) => yield_!(*body_block),
                _ => {}
            }
        })
        .into_iter()
    }
}

#[derive(Debug, Clone, Copy, ctor)]
struct BlockCursorFrame {
    block_idx: BlockIdx,
    #[ctor(default)]
    instr_idx: usize,
}
