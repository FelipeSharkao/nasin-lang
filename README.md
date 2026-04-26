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

Requirements: a recent Rust toolchain and a C compiler.

```bash
make
```

This builds the compiler with the standard library path baked in and copies the
binary to `bin/nasin`.

> Making changes to the grammar additionally requires [Bun]. Run
> `make -B GENERATE_GRAMMAR=true` to regenerate the tree-sitter parser, or run
> `make ENV=dev` to build in development mode.

## Usage

Build an executable from a source file:

```bash
./bin/nasin build -o myprog path/to/file.nsn
```

Or run it directly:

```bash
./bin/nasin run path/to/file.nsn
```

## Testing

Snapshot tests use [rere.py]:

```bash
make test           # replay snapshots
make record-test    # update snapshots
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
[rere.py]: https://github.com/tsoding/rere.py
