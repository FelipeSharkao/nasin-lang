use std::fmt::{self, Write};
use std::mem;

use bump_scope::traits::BumpAllocatorTyped;
use bump_scope::{Bump, BumpScope, BumpString, BumpVec, bump_vec};
use derive_ctor::ctor;
use derive_setters::Setters;

use super::instr::*;
use super::module::*;
use super::ty::*;
use crate::config::BuildConfig;
use crate::sources::SourceManager;
use crate::utils;

const S: &str = "";

#[derive(ctor, Setters)]
#[setters(into)]
pub struct Printer<'a> {
    modules: &'a [Module],
    cfg: &'a BuildConfig,
    #[ctor(default)]
    source_manager: Option<&'a SourceManager>,
    #[ctor(default)]
    show_ids: bool,
    #[ctor(default)]
    reconstruct: bool,
}

impl<'a> Printer<'a> {
    pub fn print_all(&mut self) {
        let mut p = utils::WriteIO::stdout();
        self.write_all(&mut p).unwrap();
        writeln!(p).unwrap();
    }

    pub fn print_module(&mut self, mod_idx: usize) {
        let mut p = utils::WriteIO::stdout();
        self.write_module(&mut p, mod_idx).unwrap();
        writeln!(p).unwrap();
    }

    pub fn write_all(&mut self, f: &mut impl Write) -> fmt::Result {
        let mut bump = Bump::new();
        let bump = bump.as_mut_scope();
        self.write_all_in(f, bump)
    }

    pub fn write_module(&mut self, f: &mut impl Write, mod_idx: usize) -> fmt::Result {
        let mut bump = Bump::new();
        let bump = bump.as_mut_scope();
        self.write_module_in(f, bump, mod_idx)
    }

    /// Writes a type body expression. Respects the `reconstruct` and `show_ids` flags.
    pub fn write_type_expr(
        &mut self,
        f: &mut impl Write,
        body: &TypeBody,
    ) -> fmt::Result {
        self.write_type_body(f, body)
    }

    fn write_all_in(&mut self, f: &mut impl Write, bump: &mut BumpScope) -> fmt::Result {
        for (i, _) in self.modules.iter().enumerate() {
            if i > 0 {
                writeln!(f)?;
            }
            self.write_module_in(f, bump, i)?;
        }
        Ok(())
    }

    fn write_module_in(
        &mut self,
        f: &mut impl Write,
        bump: &mut BumpScope,
        mod_idx: usize,
    ) -> fmt::Result {
        let mut guard = bump.scope_guard();
        let mut bump = guard.scope().by_value();

        let module = &self.modules[mod_idx];

        write!(f, "{}", module.name)?;
        if self.show_ids {
            write!(f, " (module {mod_idx})")?;
        }
        write!(f, ":")?;

        for (i, _) in module.typedefs.iter().enumerate() {
            writeln!(f)?;
            self.write_typedef(f, module, i, 2)?;
        }

        if module.typevars.len() > 0 {
            writeln!(f)?;

            let mut guard = bump.scope_guard();
            let bump = guard.scope().by_value();
            let mut table = BumpTable::new_in(&bump);

            for i in 0..module.typevars.len() {
                self.write_typevar_tabled(&mut table, module, i, 2)?;
            }

            writeln!(f, "{table}")?;
        }

        for (i, _) in module.globals.iter().enumerate() {
            writeln!(f)?;
            self.write_global_in(f, &mut bump, module, i, 2)?;
            writeln!(f)?;
        }

        for (i, _) in module.funcs.iter().enumerate() {
            writeln!(f)?;
            self.write_func_in(f, &mut bump, module, i, 2)?;
            writeln!(f)?;
        }

        Ok(())
    }

    fn write_typevar_tabled<'t, 'b: 't>(
        &mut self,
        table: &'t mut BumpTable<&'b BumpScope<'b>>,
        module: &Module,
        idx: usize,
        indent: usize,
    ) -> fmt::Result {
        let typevar = &module.typevars[idx];

        table.start_row();

        let line = table.push_cell();
        write!(line, "{S:indent$}typevar {}", &typevar.name)?;
        if self.show_ids {
            write!(line, " (typevar {idx})")?;
        }

        if !self.reconstruct {
            let loc_comment = table.push_cell();
            self.write_loc_comment(loc_comment, Some(&typevar.loc))?;
        }

        table.end_row();

        Ok(())
    }

    fn write_typedef(
        &mut self,
        f: &mut impl Write,
        module: &Module,
        idx: usize,
        indent: usize,
    ) -> fmt::Result {
        let typedef = &module.typedefs[idx];

        if self.reconstruct {
            write!(f, "{S:indent$}type {}", typedef.name.last_ident())?;
            match &typedef.body {
                TypeDefBody::Record(rec) => {
                    write!(f, " {{")?;
                    for (name, field) in &rec.fields {
                        writeln!(f)?;
                        write!(f, "{S:indent$}  {name}: ")?;
                        self.write_type_body(f, &field.ty.body)?;
                    }
                    for (name, method) in &rec.methods {
                        let func =
                            &self.modules[method.func_ref.0].funcs[method.func_ref.1];
                        writeln!(f)?;
                        write!(f, "{S:indent$}  ")?;
                        self.write_method_signature(
                            f,
                            name,
                            &self.modules[method.func_ref.0],
                            func,
                        )?;
                    }
                    writeln!(f)?;
                    write!(f, "{S:indent$}}}")?;
                }
                TypeDefBody::Interface(iface) => {
                    write!(f, " interface {{")?;
                    for (name, method) in &iface.methods {
                        let func =
                            &self.modules[method.func_ref.0].funcs[method.func_ref.1];
                        writeln!(f)?;
                        write!(f, "{S:indent$}  ")?;
                        self.write_method_signature(
                            f,
                            name,
                            &self.modules[method.func_ref.0],
                            func,
                        )?;
                    }
                    writeln!(f)?;
                    write!(f, "{S:indent$}}}")?;
                }
            }
        } else {
            let mod_idx = module.idx;
            write!(f, "{S:indent$}type {mod_idx}-{idx} {}", typedef.name)?;
            self.write_loc_comment(f, Some(&typedef.loc))?;

            match &typedef.body {
                TypeDefBody::Record(rec) => {
                    write!(f, " record:")?;
                    if !rec.ifaces.is_empty() {
                        write!(f, "\n{S:indent$}  implements")?;
                        let mut ifaces: Vec<_> = rec.ifaces.iter().collect();
                        ifaces.sort();
                        for (m, t) in ifaces {
                            write!(f, " ({m}-{t})")?;
                        }
                    }
                    for (name, field) in &rec.fields {
                        write!(f, "\n{S:indent$}  {name}: ")?;
                        self.write_type_body(f, &field.ty.body)?;
                    }
                    for (name, method) in &rec.methods {
                        write!(
                            f,
                            "\n{S:indent$}  {name}(): ({}-{})",
                            method.func_ref.0, method.func_ref.1
                        )?;
                    }
                }
                TypeDefBody::Interface(iface) => {
                    write!(f, " interface:")?;
                    for (name, method) in &iface.methods {
                        write!(
                            f,
                            "\n{S:indent$}  {name}(): ({}-{})",
                            method.func_ref.0, method.func_ref.1
                        )?;
                    }
                }
            }
        }

        Ok(())
    }

    fn write_global_in(
        &mut self,
        f: &mut impl Write,
        bump: &mut BumpScope,
        module: &Module,
        idx: usize,
        indent: usize,
    ) -> fmt::Result {
        let mut guard = bump.scope_guard();
        let bump = guard.scope().by_value();

        let global = &module.globals[idx];

        let mut table = BumpTable::new_in(&bump);

        let line = table.push_cell();
        write!(line, "{S:indent$}{}", global.name)?;

        if !self.reconstruct {
            write!(line, " v{}", global.value)?;
        }

        write!(line, ": ")?;
        self.write_type_body(line, &module.values[global.value].ty.body)?;
        if self.show_ids {
            write!(line, " (global {idx})")?;
        }

        if !self.reconstruct {
            let loc_comment = table.push_cell();
            self.write_loc_comment(loc_comment, Some(&global.loc))?;
        }

        self.write_block_tabled(&mut table, module, global.body, indent + 2)?;

        write!(f, "{table}")?;

        Ok(())
    }

    fn write_func_in(
        &mut self,
        f: &mut impl Write,
        bump: &mut BumpScope,
        module: &Module,
        idx: usize,
        indent: usize,
    ) -> fmt::Result {
        let mut guard = bump.scope_guard();
        let bump = guard.scope().by_value();

        let mut table = BumpTable::new_in(&bump);

        self.write_func_signature_tabled(&mut table, module.idx, idx, indent)?;
        self.write_block_tabled(&mut table, module, module.funcs[idx].body, indent + 2)?;

        write!(f, "{table}")
    }

    fn write_func_signature(
        &mut self,
        f: &mut impl Write,
        module: &Module,
        idx: usize,
        indent: usize,
    ) -> fmt::Result {
        let func = &module.funcs[idx];

        write!(f, "{S:indent$}{}", func.name)?;

        if !func.generics.is_empty() && !self.reconstruct {
            write!(f, "<")?;
            for (i, &tv_idx) in func.generics.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                let tv = &module.typevars[tv_idx];
                write!(f, "{}", tv.name)?;
                if self.show_ids {
                    write!(f, " ({tv_idx})")?;
                }
            }
            write!(f, ">")?;
        }

        write!(f, "(")?;
        for (i, &v) in func.params.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            if !self.reconstruct {
                write!(f, "v{v}: ")?;
            }
            self.write_type_body(f, &module.values[v].ty.body)?;
        }
        write!(f, ")")?;

        if !self.reconstruct {
            write!(f, " v{}", func.ret)?;
        }

        write!(f, ": ")?;
        self.write_type_body(f, &module.values[func.ret].ty.body)?;

        if let Some(Extern { name }) = &func.extrn {
            write!(f, " @extern({})", utils::encode_string_lit(name))?;
        }

        if self.show_ids && !self.reconstruct {
            write!(f, " (func {idx})")?;
        }

        Ok(())
    }

    fn write_func_signature_tabled<'t, 'b: 't>(
        &mut self,
        table: &'t mut BumpTable<&'b BumpScope<'b>>,
        mod_idx: usize,
        func_idx: usize,
        indent: usize,
    ) -> fmt::Result {
        let module = &self.modules[mod_idx];
        let func = &module.funcs[func_idx];

        table.start_row();

        let line = table.push_cell();
        self.write_func_signature(line, module, func_idx, indent)?;

        if !self.reconstruct {
            let loc_comment = table.push_cell();
            self.write_loc_comment(loc_comment, func.loc.as_ref())?;
        }

        table.end_row();

        Ok(())
    }

    fn write_method_signature(
        &mut self,
        f: &mut impl Write,
        name: &str,
        module: &Module,
        func: &Func,
    ) -> fmt::Result {
        write!(f, "{name}(")?;
        for (i, &v) in func.params.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "v{v}: ")?;
            self.write_type_body(f, &module.values[v].ty.body)?;
        }
        write!(f, ")")?;

        let ret_ty = &module.values[func.ret].ty.body;
        if !matches!(ret_ty, TypeBody::Void) {
            write!(f, ": ")?;
            self.write_type_body(f, ret_ty)?;
        }

        Ok(())
    }

    fn write_block_tabled<'t, 'b: 't>(
        &mut self,
        table: &'t mut BumpTable<&'b BumpScope<'b>>,
        module: &Module,
        block_idx: BlockIdx,
        indent: usize,
    ) -> fmt::Result {
        table.start_row();

        if self.reconstruct {
            table.push_cell_lit("...");
            table.end_row();
            return Ok(());
        }

        let body = &module.blocks[block_idx].body;

        let line = table.push_cell();
        write!(line, "{S:indent$}block{block_idx}:")?;

        let loc_comment = table.push_cell();
        self.write_loc_comment(loc_comment, module.blocks[block_idx].loc.as_ref())?;

        for instr in body {
            self.write_instr_tabled(table, module, instr, indent + 2)?;
        }

        Ok(())
    }

    fn write_instr_tabled<'t, 'b: 't>(
        &mut self,
        table: &'t mut BumpTable<&'b BumpScope<'b>>,
        module: &Module,
        instr: &Instr,
        indent: usize,
    ) -> fmt::Result {
        table.start_row();

        if self.reconstruct {
            table.push_cell_lit("...");
            table.end_row();
            return Ok(());
        }

        let line = table.push_cell();
        write!(line, "{S:indent$}")?;

        if !instr.results.is_empty() {
            for (i, &v) in instr.results.iter().enumerate() {
                if i > 0 {
                    write!(line, ", ")?;
                }
                write!(line, "v{v}: ")?;
                self.write_type_body(line, &module.values[v].ty.body)?;
            }
            write!(line, " = ")?;
        }

        self.write_instr_body(line, &instr.body)?;

        let loc_comment = table.push_cell();
        self.write_loc_comment(loc_comment, instr.loc.as_ref())?;

        table.end_row();

        match &instr.body {
            InstrBody::If(_, then_block, else_block) => {
                self.write_block_tabled(table, module, *then_block, indent + 2)?;
                self.write_block_tabled(table, module, *else_block, indent + 2)?;
            }
            InstrBody::Loop(_, body_block) => {
                self.write_block_tabled(table, module, *body_block, indent + 2)?;
            }
            _ => {}
        }

        Ok(())
    }

    fn write_instr_body(&mut self, f: &mut impl Write, body: &InstrBody) -> fmt::Result {
        match body {
            InstrBody::GetGlobal(mod_idx, global_idx) => {
                write!(f, "GetGlobal(")?;
                self.write_global_ref(f, *mod_idx, *global_idx)?;
                write!(f, ")")
            }
            InstrBody::GetFunc(mod_idx, func_idx) => {
                write!(f, "GetFunc(")?;
                self.write_func_ref(f, *mod_idx, *func_idx)?;
                write!(f, ")")
            }
            InstrBody::Call(mod_idx, func_idx, args) => {
                write!(f, "Call(")?;
                self.write_func_ref(f, *mod_idx, *func_idx)?;
                for arg in args {
                    write!(f, ", v{arg}")?;
                }
                write!(f, ")")
            }
            InstrBody::IndirectCall(v, args) => {
                write!(f, "IndirectCall(v{v}")?;
                for arg in args {
                    write!(f, ", v{arg}")?;
                }
                write!(f, ")")
            }
            InstrBody::GetProperty(v, prop) => write!(f, "GetProperty(v{v}, .{prop})"),
            InstrBody::GetField(v, field) => write!(f, "GetField(v{v}, .{field})"),
            InstrBody::GetMethod(v, name) => write!(f, "GetMethod(v{v}, .{name})"),
            InstrBody::CreateBool(b) => write!(f, "CreateBool({b})"),
            InstrBody::CreateNumber(n) => write!(f, "CreateNumber({n})"),
            InstrBody::CreateString(s) => {
                write!(f, "CreateString({})", utils::encode_string_lit(s))
            }
            InstrBody::CreateUninitializedString(len) => {
                write!(f, "CreateUninitializedString(v{len})")
            }
            InstrBody::CreateArray(vs) => {
                write!(f, "CreateArray(")?;
                for (i, v) in vs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "v{v}")?;
                }
                write!(f, ")")
            }
            InstrBody::CreateRecord(fields) => {
                write!(f, "CreateRecord(")?;
                for (i, (name, v)) in fields.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, ".{name}=v{v}")?;
                }
                write!(f, ")")
            }
            InstrBody::Add(a, b) => write!(f, "Add(v{a}, v{b})"),
            InstrBody::Sub(a, b) => write!(f, "Sub(v{a}, v{b})"),
            InstrBody::Mul(a, b) => write!(f, "Mul(v{a}, v{b})"),
            InstrBody::Div(a, b) => write!(f, "Div(v{a}, v{b})"),
            InstrBody::Mod(a, b) => write!(f, "Mod(v{a}, v{b})"),
            InstrBody::Eq(a, b) => write!(f, "Eq(v{a}, v{b})"),
            InstrBody::Neq(a, b) => write!(f, "Neq(v{a}, v{b})"),
            InstrBody::Gt(a, b) => write!(f, "Gt(v{a}, v{b})"),
            InstrBody::Gte(a, b) => write!(f, "Gte(v{a}, v{b})"),
            InstrBody::Lt(a, b) => write!(f, "Lt(v{a}, v{b})"),
            InstrBody::Lte(a, b) => write!(f, "Lte(v{a}, v{b})"),
            InstrBody::Not(v) => write!(f, "Not(v{v})"),
            InstrBody::If(cond, then_block, else_block) => {
                write!(f, "If(v{cond}, block{then_block}, block{else_block})")
            }
            InstrBody::Loop(inputs, body_block) => {
                write!(f, "Loop(")?;
                for (i, (v, init)) in inputs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "v{v}=v{init}")?;
                }
                write!(f, ", block{body_block})")
            }
            InstrBody::Break(v) => {
                write!(f, "Break(")?;
                if let Some(v) = v {
                    write!(f, "v{v}")?;
                }
                write!(f, ")")
            }
            InstrBody::Continue(vs) => {
                write!(f, "Continue(")?;
                for (i, v) in vs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "v{v}")?;
                }
                write!(f, ")")
            }
            InstrBody::StrLen(v) => write!(f, "StrLen(v{v})"),
            InstrBody::StrPtr(v) => write!(f, "StrPtr(v{v})"),
            InstrBody::StrFromPtr(ptr, len) => write!(f, "StrFromPtr(v{ptr}, v{len})"),
            InstrBody::StrCopy(src, dst, off) => {
                write!(f, "StrCopy(v{src}, v{dst}")?;
                if let Some(o) = off {
                    write!(f, "+v{o}")?;
                }
                write!(f, ")")
            }
            InstrBody::ArrayLen(v) => write!(f, "ArrayLen(v{v})"),
            InstrBody::ArrayIndex(v, i) => write!(f, "ArrayIndex(v{v}, v{i})"),
            InstrBody::PtrOffset(p, o) => write!(f, "PtrOffset(v{p}, v{o})"),
            InstrBody::PtrSet(p, val) => write!(f, "PtrSet(v{p}, v{val})"),
            InstrBody::TypeName(v) => write!(f, "TypeName(v{v})"),
            InstrBody::CompileError => write!(f, "CompileError()"),
            InstrBody::Dispatch(v, mod_idx, ty_idx) => {
                write!(f, "Dispatch(v{v}, ")?;
                self.write_type_ref(f, &TypeRef::new(*mod_idx, *ty_idx))?;
                write!(f, ")")
            }
            InstrBody::Type(v, ty) => {
                write!(f, "Type(v{v}, ")?;
                self.write_type_body(f, &ty.body)?;
                write!(f, ")")
            }
        }
    }

    fn write_type_body(&mut self, f: &mut impl Write, body: &TypeBody) -> fmt::Result {
        match body {
            TypeBody::Inferred(v) => {
                write!(f, "{{")?;
                for (name, t) in &v.members {
                    write!(f, " {name}: ")?;
                    self.write_type_body(f, &t.body)?;
                }
                for (name, t) in &v.properties {
                    write!(f, " .{name}: ")?;
                    self.write_type_body(f, &t.body)?;
                }
                write!(f, " }}")?;
            }
            TypeBody::Array(ty) => {
                write!(f, "[")?;
                self.write_type_body(f, &ty.body)?;
                write!(f, "]")?;
            }
            TypeBody::Ptr(ty) => {
                write!(f, "Ptr")?;
                if let Some(ty) = ty {
                    write!(f, "(")?;
                    self.write_type_body(f, &ty.body)?;
                    write!(f, ")")?;
                }
            }
            TypeBody::Func(func) => {
                write!(f, "Func(")?;
                for (i, p) in func.params.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    self.write_type_body(f, &p.body)?;
                }
                write!(f, "): ")?;
                self.write_type_body(f, &func.ret.body)?;
            }
            TypeBody::TypeRef(ty_ref) => self.write_type_ref(f, ty_ref)?,
            TypeBody::TypeVar(tv) => {
                let name = self
                    .modules
                    .get(tv.mod_idx)
                    .and_then(|m| m.typevars.get(tv.typevar_idx))
                    .map(|tv| tv.name.to_string());
                if let Some(name) = name {
                    write!(f, "{name}")?;
                }
                if self.show_ids {
                    write!(f, " ({}-{})", tv.mod_idx, tv.typevar_idx)?;
                }
            }
            TypeBody::Void
            | TypeBody::Never
            | TypeBody::Bool
            | TypeBody::AnyNumber
            | TypeBody::AnySignedNumber
            | TypeBody::AnyFloat
            | TypeBody::AnyOpaque
            | TypeBody::I8
            | TypeBody::I16
            | TypeBody::I32
            | TypeBody::I64
            | TypeBody::U8
            | TypeBody::U16
            | TypeBody::U32
            | TypeBody::U64
            | TypeBody::USize
            | TypeBody::F32
            | TypeBody::F64
            | TypeBody::String => write!(f, "{body}")?,
        }
        Ok(())
    }

    fn write_type_ref(&mut self, f: &mut impl Write, ty_ref: &TypeRef) -> fmt::Result {
        if ty_ref.is_self {
            write!(f, "Self")?;
            if self.show_ids {
                write!(f, " ({}-{})", ty_ref.mod_idx, ty_ref.idx)?;
            }
            return Ok(());
        }
        match self
            .modules
            .get(ty_ref.mod_idx)
            .and_then(|m| m.typedefs.get(ty_ref.idx))
        {
            Some(td) => {
                write!(f, "{}", td.name)?;
                if self.show_ids {
                    write!(f, " ({}-{})", ty_ref.mod_idx, ty_ref.idx)?;
                }
            }
            None => write!(f, "type({}-{})", ty_ref.mod_idx, ty_ref.idx)?,
        }
        Ok(())
    }

    fn write_func_ref(
        &mut self,
        f: &mut impl Write,
        mod_idx: usize,
        func_idx: usize,
    ) -> fmt::Result {
        match self
            .modules
            .get(mod_idx)
            .and_then(|m| m.funcs.get(func_idx))
        {
            Some(func) => {
                write!(f, "{}", func.name)?;
                if self.show_ids {
                    write!(f, " (func {mod_idx}-{func_idx})")?;
                }
            }
            None => write!(f, "func {mod_idx}-{func_idx}")?,
        }
        Ok(())
    }

    fn write_global_ref(
        &mut self,
        f: &mut impl Write,
        mod_idx: usize,
        global_idx: usize,
    ) -> fmt::Result {
        match self
            .modules
            .get(mod_idx)
            .and_then(|m| m.globals.get(global_idx))
        {
            Some(g) => {
                write!(f, "{}", g.name)?;
                if self.show_ids {
                    write!(f, " (gobal {mod_idx}-{global_idx})")?;
                }
            }
            None => write!(f, "global {mod_idx}-{global_idx}")?,
        }
        Ok(())
    }

    fn write_loc_comment(&self, f: &mut impl Write, loc: Option<&Loc>) -> fmt::Result {
        let Some(loc) = loc else {
            return Ok(());
        };

        let Some(source_manager) = self.source_manager else {
            return Ok(());
        };

        let Some(source) = source_manager.sources.get(loc.source_idx) else {
            return Ok(());
        };

        let path = self.cfg.strip_base_paths(&source.path).display();

        write!(f, "; {path}:{}:{}", loc.start_line, loc.start_col)
    }
}

struct BumpTable<A: BumpAllocatorTyped> {
    rows:     BumpVec<BumpVec<BumpString<A>, A>, A>,
    next_row: BumpVec<BumpString<A>, A>,
    alloc:    A,
}

impl<A: BumpAllocatorTyped + Clone> BumpTable<A> {
    fn new_in(alloc: A) -> Self {
        Self {
            rows: bump_vec![in alloc.clone()],
            next_row: bump_vec![in alloc.clone()],
            alloc,
        }
    }

    fn push_cell(&mut self) -> &mut BumpString<A> {
        let cell = BumpString::new_in(self.alloc.clone());
        self.next_row.push(cell);
        let idx = self.next_row.len() - 1;
        &mut self.next_row[idx]
    }

    fn push_cell_lit(&mut self, s: &'static str) -> &mut BumpString<A> {
        let cell = BumpString::from_str_in(s, self.alloc.clone());
        self.next_row.push(cell);
        let idx = self.next_row.len() - 1;
        &mut self.next_row[idx]
    }

    fn start_row(&mut self) {
        if self.next_row.len() > 0 {
            self.end_row();
        }
    }

    fn end_row(&mut self) {
        let curr_row = mem::replace(&mut self.next_row, bump_vec![in self.alloc.clone()]);
        self.rows.push(curr_row);
    }
}

impl<A: BumpAllocatorTyped + Clone> fmt::Display for BumpTable<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut cols_widths = bump_vec![in self.alloc.clone()];

        for row in self.rows.iter().chain([&self.next_row]) {
            if row.len() > cols_widths.len() {
                cols_widths.resize(row.len(), 0);
            }
            for (i, col) in row.iter().enumerate() {
                cols_widths[i] = cols_widths[i].max(col.len());
            }
        }

        let fmt_row = |f: &mut fmt::Formatter<'_>, row: &BumpVec<BumpString<A>, A>| {
            let cols_count = row.len();
            for (i, col) in row.iter().enumerate() {
                if i > 0 {
                    write!(f, "  ")?;
                }
                if i < cols_count - 1 {
                    write!(f, "{col:width$}", width = cols_widths[i])?;
                } else {
                    write!(f, "{col}")?;
                }
            }
            Ok(())
        };

        for (i, row) in self.rows.iter().enumerate() {
            if i > 0 {
                writeln!(f)?;
            }
            fmt_row(f, row)?;
        }

        if self.next_row.len() > 0 {
            if self.rows.len() > 0 {
                writeln!(f)?;
            }
            fmt_row(f, &self.next_row)?;
        }

        Ok(())
    }
}
