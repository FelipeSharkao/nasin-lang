# Bytecode IR Refactoring

This document describes the three-phase refactoring of the nasin compiler's bytecode intermediate representation (IR). The work flattened recursive instruction bodies into a central block table, added traversal utilities, and overhauled pretty-printing for both developer debugging and user-facing error messages.

---

## Table of Contents

1. [Motivation](#motivation)
2. [Phase 1: Flatten instruction bodies into a block table](#phase-1-flatten-instruction-bodies-into-a-block-table)
3. [Phase 2: Block cursor, visitors, and clone_block_tree](#phase-2-block-cursor-visitors-and-clone_block_tree)
4. [Phase 3: Pretty printers and error message formatting](#phase-3-pretty-printers-and-error-message-formatting)
5. [Summary of all modified files](#summary-of-all-modified-files)

---

## Motivation

Before this refactoring, the nasin bytecode IR represented control flow inline: an
`If` instruction contained two `Vec<Instr>` (then-branch and else-branch), and a `Loop`
instruction contained a `Vec<Instr>` for its body. This recursive nesting had several
drawbacks:

- **Traversal complexity.** Every compiler pass that walked instructions needed its own
  recursive descent logic. There was no shared cursor or visitor abstraction.
- **Cloning overhead.** Generic function instantiation required deep-cloning nested
  instruction trees. With inline `Vec<Instr>`, the cloning code had to recursively walk
  and rebuild the entire tree manually.
- **Transform ergonomics.** The old `CodeTransformCursor` trait hierarchy was tightly
  coupled to the recursive structure, making it difficult to insert, remove, or replace
  instructions during a transform pass.
- **Display limitations.** Printing the IR for `--dump-bytecode` interleaved recursive
  formatting with instruction formatting, producing inconsistent output. There was no
  way to cross-reference blocks or show block indices.
- **Error messages.** Type errors printed raw `TypeBody` values (e.g., `type 0-2`,
  `typevar 1-0`) instead of resolved human-readable names, because the error formatting
  path had no access to the module list.

The refactoring addresses all of these by introducing a flat block table, a cursor
abstraction, a generic block-cloning mechanism, and two distinct type printers (debug
and user-facing).

---

## Phase 1: Flatten instruction bodies into a block table

### Core idea

Replace all inline `Vec<Instr>` inside `InstrBody::If` and `InstrBody::Loop` with
`BlockIdx` references into a new `Module.blocks: Vec<Block>` field. Every sequence of
instructions — whether it is a function body, a global initializer, an if-branch, or a
loop body — is now a `Block` stored in the module's flat block table.

### Data structure changes

#### `src/bytecode/module.rs`

Added:

```rust
pub type BlockIdx = usize;

pub struct Block {
    pub body: Vec<Instr>,
}
```

The `Module` struct gained a new field:

```rust
pub struct Module {
    // ... existing fields ...
    pub blocks: Vec<Block>,
}
```

`Global.body` and `Func.body` changed from `Vec<Instr>` to `BlockIdx`. Three accessor
methods were added:

- `Module::add_block(body: Vec<Instr>) -> BlockIdx` — pushes a new block into the table
  and returns its index.
- `Module::block(idx: BlockIdx) -> &[Instr]` — borrows the block's instruction slice.
- `Module::block_mut(idx: BlockIdx) -> &mut Vec<Instr>` — mutably borrows the block's
  instruction vector.

#### `src/bytecode/instr.rs`

The `InstrBody` enum variants changed:

```rust
// Before:
If(ValueIdx, Vec<Instr>, Vec<Instr>)
Loop(Vec<(ValueIdx, ValueIdx)>, Vec<Instr>)

// After:
If(ValueIdx, BlockIdx, BlockIdx)
Loop(Vec<(ValueIdx, ValueIdx)>, BlockIdx)
```

`InstrBody::remap_values` was simplified: it no longer recursively descends into
sub-instruction vectors. It only remaps the value indices within the instruction itself
(the condition value for `If`, the init values for `Loop`). Sub-block traversal is now
the caller's responsibility.

The `Display for InstrBody` was updated to print block references in a flat format
(`block:N`) instead of inlining instruction lists. This Display impl is the "raw"
format; the pretty-printer (Phase 3) handles inlining.

#### `src/parser/module_parser.rs`

`ModuleParser` gained a `blocks: Vec<b::Block>` field and an `add_block` helper. The
`finish()` method was updated so that when it creates `Func` and `Global` entries, it
calls `add_block` to allocate the body block and stores the returned `BlockIdx` in
`func.body` / `global.body`. The blocks vector is moved into the `Module` during
`finish()`.

#### `src/parser/expr_parser.rs`

The `if` and `loop` expression parsing was updated. Previously, the parser built
`Vec<Instr>` inline and passed them directly to the `InstrBody::If` / `InstrBody::Loop`
constructors. Now:

- For `if`: after parsing the then-branch and else-branch instruction lists, the parser
  calls `self.module.add_block(then_body)` and `self.module.add_block(else_body)` to
  allocate blocks, then constructs `InstrBody::If(cond, then_block_idx, else_block_idx)`.
- For `loop`: similarly allocates the loop body block via `self.module.add_block(body)`.

#### `src/typecheck/mod.rs`

The `add_block` method was updated to take a `BlockIdx` parameter. It clones the block's
instruction list via `module.block(block_idx).to_vec()` and processes each instruction.
This clone is necessary because the typechecker mutates the module's value table while
iterating, and Rust's borrow checker does not allow simultaneous mutable access to the
module and immutable iteration over its blocks.

#### `src/codegen/binary/func.rs`

`FuncCodegen::add_block` was updated analogously: it takes a `BlockIdx`, clones the
body via `.to_vec()`, and iterates over the cloned instructions. The lifetime parameter
on several methods (`add_instr`, `value_from_instr`, `create_array_inst`,
`create_number_inst`) was relaxed from `&'a b::Instr` to `&b::Instr` since instructions
are now cloned rather than borrowed from the module.

#### `src/codegen/binary/mod.rs`

`build_entry`, `build_function`, and `insert_function` were updated to use
`module.block(func.body)` instead of accessing `func.body` as a `Vec<Instr>` directly.
The emptiness check changed from `func.body.is_empty()` to
`module.block(func.body).is_empty()`.

#### `src/codegen/binary/context.rs`

`insert_global` clones the global's block body via `module.block(global.body).to_vec()`
for iteration, matching the same pattern used in `func.rs` and `typecheck/mod.rs`.

#### `src/context/runtime.rs`

`RuntimeBuilder` gained a `blocks: Vec<b::Block>` field and an `add_block` helper. The
`build()` method includes these blocks when constructing the `Module`. The methods
`add_entry`, `add_print_array`, and `add_print_array_2d` were updated to allocate blocks
via `self.add_block(body)` when building if/loop IR for the runtime entry function.

---

## Phase 2: Block cursor, visitors, and clone_block_tree

### BlockCursor (`src/bytecode/cursor.rs`)

A new `BlockCursor` struct provides a position within a block's instruction list:

```rust
pub struct BlockCursor {
    pub block_idx: BlockIdx,
    instr_idx:     usize,
}
```

Methods:

- `new(block_idx)` — creates a cursor at the beginning of a block.
- `has_next(module)` — returns true if there are more instructions.
- `instr(module) -> &Instr` — borrows the current instruction.
- `instr_mut(module) -> &mut Instr` — mutably borrows the current instruction.
- `advance()` — moves to the next instruction.
- `shift(offset: isize)` — shifts the position by a signed offset (clamped to 0).
- `insert_instr(module, instr)` — inserts an instruction at the current position and
  advances past it. This is the key method for transform steps that need to inject
  new instructions before the current one.
- `sub_blocks(module)` — returns the sub-block indices referenced by the current
  instruction (`(then, Some(else))` for `If`, `(body, None)` for `Loop`, `None`
  otherwise).

### visit_block (`src/bytecode/cursor.rs`)

A free function for depth-first traversal:

```rust
pub fn visit_block(
    modules: &[Module],
    mod_idx: usize,
    block_idx: BlockIdx,
    visitor: &mut impl FnMut(usize, &BlockCursor, &[Module]),
)
```

This walks all instructions in a block tree in pre-order (sub-blocks before parent),
calling the visitor closure for each instruction. It is available as general-purpose
infrastructure but is not yet used by any consumer — callers currently iterate
modules/globals/funcs themselves and use `BlockCursor` directly.

### clone_block_tree (`src/bytecode/module.rs`)

A method on `Module` for deep-cloning a block and all its transitively referenced
sub-blocks:

```rust
impl Module {
    pub fn clone_block_tree<T: BlockCloneTransformer>(
        &mut self,
        block_idx: BlockIdx,
        transformer: &mut T,
    ) -> BlockIdx
}
```

This recursively walks the block tree, cloning each instruction. For each cloned
instruction:

1. Results are remapped via `transformer.remap_result(module, old_value_idx)`.
2. Operand values are remapped via `transformer.remap_instr_values(&mut body)`.
3. If the instruction is `If` or `Loop`, sub-blocks are recursively cloned and the
   block indices in the instruction body are updated to point to the new copies.

The `BlockCloneTransformer` trait is statically dispatched (generic, not `dyn`) to
avoid dynamic dispatch overhead during monomorphization:

```rust
pub trait BlockCloneTransformer {
    fn remap_result(&mut self, module: &mut Module, old: ValueIdx) -> ValueIdx;
    fn remap_instr_values(&self, body: &mut InstrBody);
}
```

### Transform infrastructure rewrite (`src/transform/mod.rs`)

The old `CodeTransformCursor` trait hierarchy was completely replaced. The new
`CodeTransform` struct iterates all modules, globals, and functions, calling
`transform_block` for each root block. `transform_block` uses `BlockCursor`:

```rust
fn transform_block(&self, step: &mut impl CodeTransformStep, mod_idx: usize, block_idx: BlockIdx) {
    let mut cursor = BlockCursor::new(block_idx);
    loop {
        // Check for sub-blocks and recurse into them first
        if let Some((first, second)) = cursor.sub_blocks(&module) {
            self.transform_block(step, mod_idx, first);
            if let Some(second) = second {
                self.transform_block(step, mod_idx, second);
            }
        }
        // Transform the current instruction
        step.transform(mod_idx, &mut cursor);
        cursor.advance();
    }
}
```

The `CodeTransformStep` trait was simplified to a single method:

```rust
pub trait CodeTransformStep {
    fn transform(&mut self, mod_idx: usize, cursor: &mut BlockCursor);
}
```

All four transform steps were updated:

#### `src/transform/instantiate_generic_funcs.rs`

The `GenericInstantiationTransformer` struct implements `BlockCloneTransformer`. Its
`remap_result` method clones the value, applies typevar substitution, pushes the new
value into the module, and records the old→new mapping. Its `remap_instr_values` method
delegates to `InstrBody::remap_values` with the accumulated mapping.

The `remap_func` function clones a generic function template: it creates new value slots
for params and the return value (with type substitution), then calls
`module.clone_block_tree(func.body, &mut transformer)` to deep-clone the function body
with all values remapped.

The deduplication cache (`Func::generic_instantiations: HashMap<Vec<TypeBody>, usize>`)
is checked before instantiation to avoid creating duplicate copies for the same type
substitution set.

#### `src/transform/finish_dispatch.rs`

Updated to use `BlockCursor`. Inspects `Call`/`IndirectCall` instructions and inserts
`Dispatch` instructions before the call when argument types differ from interface
parameter types. Uses `cursor.insert_instr()` to inject the dispatch instruction at the
current position.

#### `src/transform/finish_get_property.rs`

Updated to use `BlockCursor`. Resolves `GetProperty` instructions to either `GetField`
or `GetMethod` by inspecting the value's type and looking up the typedef's fields and
methods.

#### `src/transform/lower_type_name.rs`

Updated to use `BlockCursor`. Replaces `TypeName(v)` instructions with
`CreateString("<type name>")`, where the type name is computed from the value's resolved
type.

---

## Phase 3: Pretty printers and error message formatting

### Debug printer (`src/bytecode/printer.rs`)

#### ModulePrinter

A new `ModulePrinter` struct replaces the old `Display for Module` as the primary
display path used by `--dump-bytecode` and `--dump-transformed-bytecode`. Key
improvements:

1. **No top-level value listing.** The old format listed every value at the top of the
   module (`v0: (type) loc`, `v1: (type) loc`, ...) before any instructions. This was
   removed entirely. Types are now shown inline at instruction results.

2. **Inline type annotations on results.** Instructions print their result values with
   type suffixes:
   ```
   v1:string = create_string "Hello" :2:9-2:16
   v7:usize = add v5 v6 :5:5-5:23
   ```

3. **Type annotations on function params and return values.** Function and global
   headers include types:
   ```
   func 0 func_declaration.foo :4:1-5:23 (params v0:string) -> v9:string
   global 0 record_type.msg :5:1-5:38 -> v2:record_type.Foo(1-0)
   ```

4. **Block inlining with block indices.** Sub-blocks for `If` and `Loop` are inlined
   with indentation, and the block index is shown:
   ```
   v91:bool = if v86 then: #57 :46:5-50:13
       v87:bool = call core.internal_eprint 0-4 v8 :47:17-47:38
       ...
   else: #58
       v90:bool = create_bool true :50:9-50:13
       ...
   ```

5. **Name resolution for Call, GetFunc, GetGlobal, Dispatch.** Instead of bare indices,
   the printer resolves and shows the full qualified name:
   ```
   // Before:
   call 0-3 v0
   get_global 1-0

   // After:
   call core.internal_print 0-3 v0
   get_global record_type.msg 1-0
   ```
   The index is still shown alongside the name for cross-referencing.

6. **Name resolution for Dispatch.** Shows the typedef name:
   ```
   dispatch v0 interface.Lines 1-2
   ```

#### DebugTypePrinter

A new `DebugTypePrinter` struct formats types for the `--dump-bytecode` output. It shows
full qualified names AND indices for `TypeRef` and `TypeVar`:

```
record_type.Foo(1-0)    // TypeRef: full name + module-index
T(1-0)                  // TypeVar: name + module-typevar_index
self(1-0)               // self type with index
```

Other type bodies use a compact format without parentheses or location spans:

```
string                  // primitive
[string]                // array (bracket notation)
*u8                     // pointer (prefix notation)
func(string) -> bool    // function type
inferred { ... }        // inferred type
```

This replaced the old `TypePrinter` struct, which used parenthesized format with
location spans (e.g., `(string :2:11-2:14)`) and only showed the last identifier for
TypeRef (e.g., `Foo` instead of `record_type.Foo(1-0)`).

#### write_instr_body

A new function that handles module-aware instruction display. It intercepts the
following instruction variants and prints them with resolved names:

- `GetGlobal(mod_idx, global_idx)` → `get_global <full_name> <mod>-<idx>`
- `GetFunc(mod_idx, func_idx)` → `get_func <full_name> <mod>-<idx>`
- `Call(mod_idx, func_idx, args)` → `call <full_name> <mod>-<idx> <args>`
- `Dispatch(v, mod_idx, ty_idx)` → `dispatch v<v> <full_name> <mod>-<idx>`
- `Type(v, ty)` → `type v<v> <debug_type>`

All other instruction variants fall through to `InstrBody::Display`, which remains as
the flat-reference format for non-module-aware contexts.

### Simplified Display for Module (`src/bytecode/module.rs`)

The old `Display for Module` was a 100-line implementation that duplicated much of
what `ModulePrinter` now does (but without name resolution or block inlining). It was
replaced with a one-line summary format for quick debugging:

```rust
impl Display for Module {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "module {} ({} values, {} globals, {} funcs, {} blocks)",
            self.idx, self.values.len(), self.globals.len(),
            self.funcs.len(), self.blocks.len())
    }
}
```

### User-facing type printer (`src/bytecode/printer.rs`)

#### format_type_body

A function that formats a `TypeBody` into a user-friendly string for error messages:

```rust
pub fn format_type_body(body: &TypeBody, modules: &[Module]) -> String
```

This uses a separate formatting strategy from `DebugTypePrinter`:

| TypeBody          | Debug format               | User format       |
|-------------------|----------------------------|--------------------|
| `String`          | `string`                   | `str`              |
| `AnyNumber`       | `AnyNumber`                | `number`           |
| `AnySignedNumber` | `AnySignedNumber`          | `signed number`    |
| `AnyFloat`        | `AnyFloat`                 | `float`            |
| `Inferred(_)`     | `inferred { ... }`         | `(unknown)`        |
| `Array(T)`        | `[T]`                      | `[T]`              |
| `Ptr(T)`          | `*T`                       | `*T`               |
| `Func(P -> R)`    | `func(P) -> R`             | `(P) -> R`         |
| `TypeRef`         | `record_type.Foo(1-0)`     | `Foo`              |
| `TypeVar`         | `T(1-0)`                   | `T`                |

The user format omits indices and module prefixes, showing only the short type name. It
uses `str` instead of `string` and `number` instead of `AnyNumber` to match the
language's surface syntax.

### Error message wiring (`src/errors.rs`)

The three error types that display types were changed from storing raw `b::Type` objects
to storing pre-formatted `String` values:

#### UnexpectedType

```rust
// Before:
pub struct UnexpectedType {
    pub expected: Vec<b::Type>,
    pub actual: b::Type,
}

// After:
pub struct UnexpectedType {
    pub expected: Vec<String>,
    pub actual: String,
}
```

The constructor now takes references and the module slice, pre-formatting via
`format_type_body`:

```rust
impl UnexpectedType {
    pub fn new(expected: Vec<&b::Type>, actual: &b::Type, modules: &[b::Module]) -> Self {
        Self {
            expected: expected.iter()
                .map(|t| b::printer::format_type_body(&t.body, modules))
                .collect(),
            actual: b::printer::format_type_body(&actual.body, modules),
        }
    }
}
```

#### TypeMisatch

```rust
// Before:
pub struct TypeMisatch {
    pub types: Vec<b::Type>,
}

// After:
pub struct TypeMisatch {
    pub types: Vec<String>,
}
```

Constructor pre-formats each type via `format_type_body`.

#### TypeNotInterface

```rust
// Before:
pub struct TypeNotInterface {
    pub ty: b::Type,
}

// After:
pub struct TypeNotInterface {
    pub ty: String,
}
```

Constructor pre-formats the type via `format_type_body`.

### Call site updates

All error construction sites were updated to pass module references:

- **`src/typecheck/mod.rs:817`** — `TypeMisatch::new(tys.iter().collect(), &modules)`
- **`src/typecheck/mod.rs:1039`** — `UnexpectedType::new(vec![&merge_with], &result_ty, &modules)`
- **`src/context/runtime.rs:126`** — `UnexpectedType::new(vec![&str_ty, &array_ty, &array_2d_ty], main_ty, &modules)`
- **`src/parser/type_parser.rs:178`** — `TypeNotInterface::new(&ty, &self.ctx.lock_modules())`

### Bytecode dump integration (`src/lib.rs`, `src/context/mod.rs`)

The `--dump-bytecode` and `--dump-transformed-bytecode` flags now call
`bytecode::printer::print_modules(&modules)` instead of `println!("{module}")`. The
runtime module dump in `src/context/mod.rs` uses
`bytecode::printer::print_module(&module, &modules)`.

---

## Summary of all modified files

| File | Phase | Changes |
|------|-------|---------|
| `src/bytecode/module.rs` | 1, 2, 3 | Added `BlockIdx`, `Block`, `blocks` field, `add_block`/`block`/`block_mut`, `clone_block_tree`, `BlockCloneTransformer`. Simplified `Display for Module`. |
| `src/bytecode/instr.rs` | 1 | Changed `If`/`Loop` from `Vec<Instr>` to `BlockIdx`. Updated `remap_values` (no recursion). Updated `Display`. |
| `src/bytecode/cursor.rs` | 2 | New file. `BlockCursor` struct with navigation/mutation methods. `visit_block` free function. |
| `src/bytecode/printer.rs` | 3 | New file. `ModulePrinter`, `DebugTypePrinter`, `write_block_inline`, `write_results`, `write_instr_body`, `format_type_body`. |
| `src/bytecode/mod.rs` | 2, 3 | Added `cursor` and `printer` module declarations and re-exports. |
| `src/errors.rs` | 3 | Changed `UnexpectedType`, `TypeMisatch`, `TypeNotInterface` to store pre-formatted strings. Added constructors that call `format_type_body`. |
| `src/parser/module_parser.rs` | 1 | Added `blocks` field, `add_block` method. Updated `finish()`, `add_func`, `add_global`. |
| `src/parser/expr_parser.rs` | 1 | Updated `if` and `loop` parsing to allocate blocks via `add_block`. |
| `src/parser/type_parser.rs` | 3 | Updated `TypeNotInterface` construction to pass modules. |
| `src/typecheck/mod.rs` | 1, 3 | `add_block` takes `BlockIdx`, clones body. Error construction passes modules. |
| `src/transform/mod.rs` | 2 | Complete rewrite. `CodeTransform` uses `BlockCursor`. `CodeTransformStep` trait simplified. |
| `src/transform/instantiate_generic_funcs.rs` | 2 | `GenericInstantiationTransformer` implements `BlockCloneTransformer`. `remap_func` uses `clone_block_tree`. |
| `src/transform/finish_dispatch.rs` | 2 | Updated to use `BlockCursor`. |
| `src/transform/finish_get_property.rs` | 2 | Updated to use `BlockCursor`. |
| `src/transform/lower_type_name.rs` | 2 | Updated to use `BlockCursor`. |
| `src/codegen/binary/func.rs` | 1 | `add_block` takes `BlockIdx`, clones body. Relaxed lifetime on `&b::Instr`. |
| `src/codegen/binary/mod.rs` | 1 | Updated `build_entry`, `build_function`, `insert_function` to use block accessors. |
| `src/codegen/binary/context.rs` | 1 | `insert_global` clones body via `block().to_vec()`. |
| `src/context/runtime.rs` | 1, 3 | Added `blocks` field, `add_block`. Updated `build()`. Error construction passes modules. |
| `src/context/mod.rs` | 3 | Runtime module dump uses `printer::print_module`. |
| `src/lib.rs` | 3 | Bytecode dump uses `printer::print_modules`. |
