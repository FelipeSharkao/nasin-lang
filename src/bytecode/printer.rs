use std::fmt::{self, Write};

use super::instr::*;
use super::module::*;
use super::ty::*;
use crate::utils;

/// Prints a full bytecode dump of all modules. This is the debug format used
/// by `--dump-bytecode` and `--dump-transformed-bytecode`. It inlines block
/// contents into their parent functions/globals with indentation.
pub fn print_modules(modules: &[Module]) {
    for module in modules {
        println!("{}", ModulePrinter { module, modules });
    }
}

/// Print a single module, using the full modules slice for name resolution.
pub fn print_module(module: &Module, modules: &[Module]) {
    println!("{}", ModulePrinter { module, modules });
}

struct ModulePrinter<'a> {
    module:  &'a Module,
    modules: &'a [Module],
}

impl fmt::Display for ModulePrinter<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let m = self.module;

        write!(f, "module {}:", m.idx)?;

        for (i, typedef) in m.typedefs.iter().enumerate() {
            write!(f, "\n    type {i} {} {}", &typedef.name, &typedef.loc)?;

            match &typedef.body {
                TypeDefBody::Record(v) => {
                    write!(f, " record:")?;
                    if v.ifaces.len() > 0 {
                        write!(f, "\n        implements")?;
                        for iface in &v.ifaces {
                            write!(f, " ({}-{})", iface.0, iface.1)?;
                        }
                    }
                    for (name, field) in &v.fields {
                        write!(
                            f,
                            "\n        {name}: {}",
                            DebugTypePrinter::new(&field.ty, self.modules)
                        )?;
                    }
                    for (name, method) in &v.methods {
                        write!(f, "\n        {name}(): {method}")?;
                    }
                }
                TypeDefBody::Interface(v) => {
                    write!(f, " interface:")?;
                    for (name, method) in &v.methods {
                        write!(f, "\n        {name}(): {method}")?;
                    }
                }
            }
        }

        for (i, typevar) in m.typevars.iter().enumerate() {
            write!(f, "\n    typevar {i} {} {}", &typevar.name, &typevar.loc)?;
        }

        for (i, global) in m.globals.iter().enumerate() {
            write!(
                f,
                "\n    global {i} {} {} -> v{}:{}",
                &global.name,
                global.loc,
                global.value,
                DebugTypePrinter::new(&m.values[global.value].ty, self.modules),
            )?;
            write_block_inline(f, m, self.modules, global.body, 8)?;
        }

        for (i, func) in m.funcs.iter().enumerate() {
            write!(f, "\n    func {i} {}", &func.name)?;
            if let Some(loc) = &func.loc {
                write!(f, " {loc}")?;
            }

            if func.params.len() > 0 {
                write!(f, " (params")?;
                for &v in &func.params {
                    write!(
                        f,
                        " v{v}:{}",
                        DebugTypePrinter::new(&m.values[v].ty, self.modules)
                    )?;
                }
                write!(f, ")")?;
            }

            write!(
                f,
                " -> v{}:{}",
                func.ret,
                DebugTypePrinter::new(&m.values[func.ret].ty, self.modules),
            )?;

            if let Some((mod_idx, ty_idx, name)) = &func.method {
                write!(f, " (method {mod_idx}-{ty_idx} .{name})")?;
            }

            if let Some(Extern { name }) = &func.extrn {
                write!(f, " (extern {})", utils::encode_string_lit(name))?;
            }

            write_block_inline(f, m, self.modules, func.body, 8)?;
        }

        Ok(())
    }
}

/// Write a block's instructions inline with indentation, recursively inlining
/// sub-blocks for If and Loop. Shows the block index in a comment.
fn write_block_inline(
    f: &mut fmt::Formatter<'_>,
    module: &Module,
    modules: &[Module],
    block_idx: BlockIdx,
    indent: usize,
) -> fmt::Result {
    let body = module.block(block_idx);
    if body.is_empty() {
        return Ok(());
    }

    let pad = " ".repeat(indent);
    for instr in body {
        match &instr.body {
            InstrBody::If(cond, then_block, else_block) => {
                write!(f, "\n{pad}")?;
                write_results(f, instr, module, modules)?;
                write!(f, "if v{cond} then: #{then_block}")?;
                if let Some(loc) = &instr.loc {
                    write!(f, " {loc}")?;
                }
                write_block_inline(f, module, modules, *then_block, indent + 4)?;
                write!(f, "\n{pad}else: #{else_block}")?;
                write_block_inline(f, module, modules, *else_block, indent + 4)?;
            }
            InstrBody::Loop(inputs, body_block) => {
                write!(f, "\n{pad}")?;
                write_results(f, instr, module, modules)?;
                write!(f, "loop")?;
                for (v, initial_v) in inputs {
                    write!(f, " v{v}=v{initial_v}")?;
                }
                write!(f, ": #{body_block}")?;
                if let Some(loc) = &instr.loc {
                    write!(f, " {loc}")?;
                }
                write_block_inline(f, module, modules, *body_block, indent + 4)?;
            }
            _ => {
                write!(f, "\n{pad}")?;
                write_results(f, instr, module, modules)?;
                write_instr_body(f, &instr.body, modules)?;
                if let Some(loc) = &instr.loc {
                    write!(f, " {loc}")?;
                }
            }
        }
    }
    Ok(())
}

/// Write the result prefix for an instruction, with inline type annotations.
fn write_results(
    f: &mut fmt::Formatter<'_>,
    instr: &Instr,
    module: &Module,
    modules: &[Module],
) -> fmt::Result {
    if instr.results.is_empty() {
        return Ok(());
    }
    for (i, &v) in instr.results.iter().enumerate() {
        if i > 0 {
            write!(f, ", ")?;
        }
        write!(
            f,
            "v{v}:{}",
            DebugTypePrinter::new(&module.values[v].ty, modules)
        )?;
    }
    write!(f, " = ")
}

/// Write an instruction body with module-aware name resolution for
/// Call, GetFunc, GetGlobal, GetMethod, Dispatch, and Type.
fn write_instr_body(
    f: &mut fmt::Formatter<'_>,
    body: &InstrBody,
    modules: &[Module],
) -> fmt::Result {
    match body {
        InstrBody::GetGlobal(mod_idx, global_idx) => {
            let name = modules
                .get(*mod_idx)
                .and_then(|m| m.globals.get(*global_idx))
                .map(|g| g.name.to_string());
            write!(f, "get_global")?;
            if let Some(name) = name {
                write!(f, " {name}")?;
            }
            write!(f, " {mod_idx}-{global_idx}")
        }
        InstrBody::GetFunc(mod_idx, func_idx) => {
            let name = modules
                .get(*mod_idx)
                .and_then(|m| m.funcs.get(*func_idx))
                .map(|func| func.name.to_string());
            write!(f, "get_func")?;
            if let Some(name) = name {
                write!(f, " {name}")?;
            }
            write!(f, " {mod_idx}-{func_idx}")
        }
        InstrBody::Call(mod_idx, func_idx, args) => {
            let name = modules
                .get(*mod_idx)
                .and_then(|m| m.funcs.get(*func_idx))
                .map(|func| func.name.to_string());
            write!(f, "call")?;
            if let Some(name) = name {
                write!(f, " {name}")?;
            }
            write!(f, " {mod_idx}-{func_idx}")?;
            for arg in args {
                write!(f, " v{arg}")?;
            }
            Ok(())
        }
        InstrBody::GetMethod(v, method_name) => {
            write!(f, "get_method v{v} .{method_name}")
        }
        InstrBody::Dispatch(v, mod_idx, ty_idx) => {
            let name = modules
                .get(*mod_idx)
                .and_then(|m| m.typedefs.get(*ty_idx))
                .map(|td| td.name.to_string());
            write!(f, "dispatch v{v}")?;
            if let Some(name) = name {
                write!(f, " {name}")?;
            }
            write!(f, " {mod_idx}-{ty_idx}")
        }
        InstrBody::Type(v, ty) => {
            write!(f, "type v{v} {}", DebugTypePrinter::new(ty, modules))
        }
        // For all other instructions, fall through to the Display impl
        _ => write!(f, "{body}"),
    }
}

// ---------------------------------------------------------------------------
// Debug-mode type printer (for --dump-bytecode)
// Shows full name + index for TypeRef, full name for TypeVar.
// ---------------------------------------------------------------------------

/// Formats a `Type` for --dump-bytecode display. Shows full qualified names
/// AND indices for TypeRef (e.g. `record_type.Foo(1-0)`).
struct DebugTypePrinter<'a> {
    ty:      &'a Type,
    modules: &'a [Module],
}

impl<'a> DebugTypePrinter<'a> {
    fn new(ty: &'a Type, modules: &'a [Module]) -> Self {
        Self { ty, modules }
    }
}

impl fmt::Display for DebugTypePrinter<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_debug_type_body(f, &self.ty.body, self.modules)
    }
}

fn write_debug_type_body(
    f: &mut fmt::Formatter<'_>,
    body: &TypeBody,
    modules: &[Module],
) -> fmt::Result {
    match body {
        TypeBody::Void => write!(f, "void"),
        TypeBody::Never => write!(f, "never"),
        TypeBody::Bool => write!(f, "bool"),
        TypeBody::AnyNumber => write!(f, "AnyNumber"),
        TypeBody::AnySignedNumber => write!(f, "AnySignedNumber"),
        TypeBody::AnyFloat => write!(f, "AnyFloat"),
        TypeBody::AnyOpaque => write!(f, "anyopaque"),
        TypeBody::I8 => write!(f, "i8"),
        TypeBody::I16 => write!(f, "i16"),
        TypeBody::I32 => write!(f, "i32"),
        TypeBody::I64 => write!(f, "i64"),
        TypeBody::U8 => write!(f, "u8"),
        TypeBody::U16 => write!(f, "u16"),
        TypeBody::U32 => write!(f, "u32"),
        TypeBody::U64 => write!(f, "u64"),
        TypeBody::USize => write!(f, "usize"),
        TypeBody::F32 => write!(f, "f32"),
        TypeBody::F64 => write!(f, "f64"),
        TypeBody::String => write!(f, "string"),
        TypeBody::Inferred(v) => {
            write!(f, "inferred {{")?;
            for (name, t) in &v.members {
                write!(f, " {name}: ")?;
                write_debug_type_body(f, &t.body, modules)?;
            }
            for (name, t) in &v.properties {
                write!(f, " .{name}: ")?;
                write_debug_type_body(f, &t.body, modules)?;
            }
            write!(f, " }}")
        }
        TypeBody::Array(v) => {
            write!(f, "[")?;
            write_debug_type_body(f, &v.body, modules)?;
            write!(f, "]")
        }
        TypeBody::Ptr(ty) => {
            write!(f, "*")?;
            if let Some(ty) = ty {
                write_debug_type_body(f, &ty.body, modules)?;
            } else {
                write!(f, "anyopaque")?;
            }
            Ok(())
        }
        TypeBody::Func(func) => {
            write!(f, "func(")?;
            for (i, p) in func.params.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write_debug_type_body(f, &p.body, modules)?;
            }
            write!(f, ") -> ")?;
            write_debug_type_body(f, &func.ret.body, modules)
        }
        TypeBody::TypeRef(ty_ref) => {
            if ty_ref.is_self {
                write!(f, "self({}-{})", ty_ref.mod_idx, ty_ref.idx)
            } else if let Some(typedef) = modules
                .get(ty_ref.mod_idx)
                .and_then(|m| m.typedefs.get(ty_ref.idx))
            {
                write!(f, "{}({}-{})", typedef.name, ty_ref.mod_idx, ty_ref.idx)
            } else {
                write!(f, "type({}-{})", ty_ref.mod_idx, ty_ref.idx)
            }
        }
        TypeBody::TypeVar(tv) => {
            if let Some(typevar) = modules
                .get(tv.mod_idx)
                .and_then(|m| m.typevars.get(tv.typevar_idx))
            {
                write!(f, "{}({}-{})", typevar.name, tv.mod_idx, tv.typevar_idx)
            } else {
                write!(f, "typevar({}-{})", tv.mod_idx, tv.typevar_idx)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// User-facing type printer (for error messages)
// Shows friendly names without indices.
// ---------------------------------------------------------------------------

/// Formats a `TypeBody` into a user-friendly string for error messages.
/// Resolves TypeRef/TypeVar to their names.
pub fn format_type_body(body: &TypeBody, modules: &[Module]) -> String {
    let mut s = String::new();
    write_type_body_to_string(&mut s, body, modules);
    s
}

fn write_type_body_to_string(s: &mut String, body: &TypeBody, modules: &[Module]) {
    match body {
        TypeBody::Void => s.push_str("void"),
        TypeBody::Never => s.push_str("never"),
        TypeBody::Bool => s.push_str("bool"),
        TypeBody::AnyNumber => s.push_str("number"),
        TypeBody::AnySignedNumber => s.push_str("signed number"),
        TypeBody::AnyFloat => s.push_str("float"),
        TypeBody::AnyOpaque => s.push_str("anyopaque"),
        TypeBody::I8 => s.push_str("i8"),
        TypeBody::I16 => s.push_str("i16"),
        TypeBody::I32 => s.push_str("i32"),
        TypeBody::I64 => s.push_str("i64"),
        TypeBody::U8 => s.push_str("u8"),
        TypeBody::U16 => s.push_str("u16"),
        TypeBody::U32 => s.push_str("u32"),
        TypeBody::U64 => s.push_str("u64"),
        TypeBody::USize => s.push_str("usize"),
        TypeBody::F32 => s.push_str("f32"),
        TypeBody::F64 => s.push_str("f64"),
        TypeBody::String => s.push_str("str"),
        TypeBody::Inferred(_) => s.push_str("(unknown)"),
        TypeBody::Array(v) => {
            s.push('[');
            write_type_body_to_string(s, &v.body, modules);
            s.push(']');
        }
        TypeBody::Ptr(ty) => {
            s.push('*');
            if let Some(ty) = ty {
                write_type_body_to_string(s, &ty.body, modules);
            } else {
                s.push_str("anyopaque");
            }
        }
        TypeBody::Func(func) => {
            s.push('(');
            for (i, p) in func.params.iter().enumerate() {
                if i > 0 {
                    s.push_str(", ");
                }
                write_type_body_to_string(s, &p.body, modules);
            }
            s.push_str(") -> ");
            write_type_body_to_string(s, &func.ret.body, modules);
        }
        TypeBody::TypeRef(ty_ref) => {
            if ty_ref.is_self {
                s.push_str("self");
            } else if let Some(typedef) = modules
                .get(ty_ref.mod_idx)
                .and_then(|m| m.typedefs.get(ty_ref.idx))
            {
                s.push_str(typedef.name.last_ident());
            } else {
                write!(s, "type({}-{})", ty_ref.mod_idx, ty_ref.idx).unwrap();
            }
        }
        TypeBody::TypeVar(tv) => {
            if let Some(typevar) = modules
                .get(tv.mod_idx)
                .and_then(|m| m.typevars.get(tv.typevar_idx))
            {
                s.push_str(typevar.name.last_ident());
            } else {
                write!(s, "typevar({}-{})", tv.mod_idx, tv.typevar_idx).unwrap();
            }
        }
    }
}
