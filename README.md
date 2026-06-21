# Nasin

Nasin is an statically-typed pure functional programming language with
structural type inference, generics, records, and interfaces that compiles to
native binaries.

> [!WARNING]
>
> **Status: early/experimental.** The language and compiler are under active
> development and not ready for production use and are not guaranteed to be
> stable.

```nasin
type Point {
    x: i32
    y: i32
}

Point.shift(point: Self, n): Self =
    {x = point.x + n, y = point.y + n}

Point.side(point: Self) =
    if point.x > 0 then "right" else "left"

origin: Point = {x=10, y=0}

main =
    let shifted = origin.shift(-20)
    [origin.side, shifted.side]
```

More examples live in [`tests/`](tests/).

## Building

Requirements: a Rust nightly toolchain and a C compiler.

```bash
./first.rs --release
```

This builds the compiler and copies the binary to `bin/nasin`.

> Changes to the grammar will only be picked up after regenerating the parser.
> See [Development](#development) for more info.

## Usage

Build an executable from a source file:

```bash
nasin build -o myprog path/to/file.nsn
```

Or run it directly:

```bash
nasin run path/to/file.nsn
```

## Development

`./first.rs` is a rust script to manage the build process. Besides compiling the
rust code, without `--release` (or with `--generate-grammar`), it will generate
the tree-sitter parser from `tree-sitter-nasin/grammar.js` if needed, and copy
the binary to the `bin/` directory.

Generating the parser requires [Bun] and [tree-sitter-cli].

For convenience, the script also provides a `--run` flag that will run the
compiled binary directly:

```bash
./first.rs --run [args]...
```

## Testing

Snapshot tests use [rere.py]:

```bash
./rere.py replay tests/_test.list # replay snapshots
./rere.py record tests/_test.list # update snapshots
```

## Roadmap

Some of the planned features, in no particular order.

- [ ] Pure side-effect API (`IO` / `Effect` / `*World`-style) for I/O
- [ ] Module system / multi-file projects
- [ ] Pattern matching
- [ ] Sum types / tagged unions
- [ ] Closures
- [ ] Inferred mutability (let local mutation be used under a pure surface)
- [ ] Self-hosting

[Bun]: https://bun.sh/
[tree-sitter-cli]: https://github.com/tree-sitter/tree-sitter
[rere.py]: https://github.com/tsoding/rere.py
