use std::borrow::{Borrow, Cow};
use std::fmt::{self, Write};
use std::mem;

use bump_scope::traits::BumpAllocatorTyped;
use bump_scope::{Bump, BumpScope, BumpString, BumpVec, bump_vec};
use derive_ctor::ctor;
use derive_setters::Setters;

use super::instr::*;
use super::module::*;
use super::ty::*;
use super::{Name, NameNode};
use crate::config::{BuildConfig, ShouldDump};
use crate::sources::SourceManager;
use crate::utils;

const S: &str = "";

#[derive(ctor, Setters)]
#[setters(into, prefix = "with_")]
pub struct Printer<'a> {
    modules: &'a [Module],
    cfg: &'a BuildConfig,
    #[ctor(default)]
    source_manager: Option<&'a SourceManager>,
    #[ctor(default)]
    show_ids: bool,
    #[ctor(default)]
    reconstruct: bool,
    #[ctor(default)]
    cur_mod_idx: Option<usize>,
}

impl<'a> Printer<'a> {
    pub fn print(&mut self, flag: impl ShouldDump) {
        let mut p = utils::WriteIO::stdout();
        self.write_all(&mut p, flag).unwrap();
        writeln!(p).unwrap();
    }

    pub fn print_module(&mut self, mod_idx: usize) {
        let mut p = utils::WriteIO::stdout();
        self.write_module(&mut p, mod_idx).unwrap();
        writeln!(p).unwrap();
    }

    pub fn write_all(
        &mut self,
        f: &mut impl Write,
        flag: impl ShouldDump,
    ) -> fmt::Result {
        let mut bump = Bump::new();
        let bump = bump.as_mut_scope();
        self.write_all_in(f, bump, flag)
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

    pub fn write_name(&mut self, f: &mut impl Write, name: &Name) -> fmt::Result {
        let name = match self.cur_mod_idx {
            Some(mod_idx) => name.strip_prefix(&self.modules[mod_idx].name),
            None => Cow::Borrowed(name),
        };
        self.write_name_nodes(f, name.nodes.iter())
    }

    pub fn write_name_nodes<'b>(
        &mut self,
        f: &mut impl Write,
        nodes: impl IntoIterator<Item = &'b NameNode>,
    ) -> fmt::Result {
        for (i, nome) in nodes.into_iter().enumerate() {
            match nome {
                NameNode::Ident(ident) => {
                    if i > 0 {
                        write!(f, ".")?;
                    }
                    write!(f, "{}", ident.ident)?;
                }
                NameNode::TypeParams(params) => {
                    write!(f, "<")?;
                    for (i, param) in params.params.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        self.write_type_expr(f, &param.body)?;
                    }
                    write!(f, ">")?;
                }
            }
        }

        Ok(())
    }

    pub fn write_signature(
        &mut self,
        f: &mut impl Write,
        params: impl IntoIterator<Item = &'a TypeBody>,
        ret: &'a TypeBody,
    ) -> fmt::Result {
        write!(f, "(")?;
        for (i, ty) in params.into_iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            self.write_type_body(f, ty)?;
        }
        write!(f, "): ")?;
        self.write_type_body(f, ret)?;
        Ok(())
    }

    fn write_all_in(
        &mut self,
        f: &mut impl Write,
        bump: &mut BumpScope,
        flag: impl ShouldDump,
    ) -> fmt::Result {
        let mut c = 0;
        for (i, module) in self.modules.iter().enumerate() {
            if flag.should_dump(&module.name) {
                if c > 0 {
                    writeln!(f)?;
                }

                self.write_module_in(f, bump, i)?;
                c += 1;
            }
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

        self.write_name(f, &module.name)?;
        if self.show_ids {
            write!(f, " (module {mod_idx})")?;
        }
        write!(f, ":")?;

        let prev_mod_idx = self.cur_mod_idx;
        self.cur_mod_idx = Some(mod_idx);

        for (i, _) in module.typedefs.iter().enumerate() {
            writeln!(f)?;
            self.write_typedef_in(f, &mut bump, module, i, 2)?;
            writeln!(f)?;

            if !self.reconstruct && !module.typedefs[i].impls.is_empty() {
                writeln!(f)?;
                self.write_impl_decls_in(f, &mut bump, module, i, 2)?;
                writeln!(f)?;
            }
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

        self.cur_mod_idx = prev_mod_idx;

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
        write!(line, "{S:indent$}typevar ")?;
        self.write_name(line, &typevar.name)?;

        if let Some(constraint) = &typevar.constraint {
            write!(line, ": ")?;
            self.write_type_body(line, &constraint.body)?;
        }

        if self.show_ids {
            write!(line, " (typevar {idx})")?;
        }

        if !self.reconstruct {
            let loc_comment = table.push_cell();
            self.write_loc_comment(loc_comment, typevar.loc.as_ref())?;
        }

        table.end_row();

        Ok(())
    }

    fn write_typedef_in(
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

        let typedef = &module.typedefs[idx];

        let header = table.push_cell();
        write!(header, "{S:indent$}type ")?;

        if let TypeDefBody::Builtin(builtin) = &typedef.body {
            self.write_builtin_type(
                header,
                builtin,
                typedef
                    .generics
                    .iter()
                    .map(|&idx| TypeBody::TypeVar(TypeVar::new(module.idx, idx))),
            )?;
        } else {
            self.write_name(header, &typedef.name)?;

            if typedef.generics.len() > 0 {
                write!(header, "(")?;
                for (i, &idx) in typedef.generics.iter().enumerate() {
                    if i > 0 {
                        write!(header, ", ")?;
                    }
                    self.write_typevar(header, module, idx)?;
                }
                write!(header, ")")?;
            }
        }

        if self.show_ids && !self.reconstruct {
            write!(header, " (type {idx})")?;
        }

        match &typedef.body {
            TypeDefBody::Record(_) | TypeDefBody::Builtin(_) => {}
            TypeDefBody::Interface => write!(header, " interface")?,
        }

        write!(header, " {{")?;

        if !self.reconstruct {
            let loc_comment = table.push_cell();
            self.write_loc_comment(loc_comment, Some(&typedef.loc))?;
        }

        match &typedef.body {
            TypeDefBody::Record(rec) => {
                for (name, field) in &rec.fields {
                    table.start_row();
                    let line = table.push_cell();
                    write!(line, "{S:indent$}  {name}: ")?;
                    self.write_type_body(line, &field.ty.body)?;

                    if !self.reconstruct {
                        let loc_comment = table.push_cell();
                        self.write_loc_comment(loc_comment, Some(&field.loc))?;
                    }
                }
            }
            TypeDefBody::Interface | TypeDefBody::Builtin(_) => {}
        };

        for (name, method) in &typedef.methods {
            table.start_row();
            let line = table.push_cell();
            let func = &self.modules[method.func_ref.0].funcs[method.func_ref.1];
            write!(line, "{S:indent$}  ")?;
            self.write_method_signature(
                line,
                name,
                &self.modules[method.func_ref.0],
                func,
            )?;

            if !self.reconstruct && self.show_ids {
                write!(line, " (func {}-{})", method.func_ref.0, method.func_ref.1)?;
            }

            if !self.reconstruct {
                let loc_comment = table.push_cell();
                self.write_loc_comment(loc_comment, Some(&method.loc))?;
            }
        }

        table.start_row();
        let line = table.push_cell();
        write!(line, "{S:indent$}}}")?;

        write!(f, "{table}")?;

        Ok(())
    }

    fn write_impl_decls_in(
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

        let typedef = &module.typedefs[idx];
        for impl_decl in &typedef.impls {
            self.write_impl_decl_tabled(&mut table, typedef, impl_decl, indent)?;
        }

        write!(f, "{table}")?;

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
        write!(line, "{S:indent$}")?;
        self.write_name(line, &global.name)?;

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

        self.write_func_decl_tabled(&mut table, module.idx, idx, indent)?;
        self.write_block_tabled(&mut table, module, module.funcs[idx].body, indent + 2)?;

        write!(f, "{table}")
    }

    fn write_func_decl(
        &mut self,
        f: &mut impl Write,
        module: &Module,
        idx: usize,
        indent: usize,
    ) -> fmt::Result {
        let func = &module.funcs[idx];

        write!(f, "{S:indent$}")?;
        self.write_name(f, &func.name)?;

        if !func.generics.is_empty() && !self.reconstruct {
            write!(f, "<")?;
            for (i, &idx) in func.generics.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                let typevar_def = &module.typevars[idx];
                self.write_name(f, &typevar_def.name)?;
                if self.show_ids {
                    write!(f, " (typevar {idx})")?;
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

    fn write_func_decl_tabled<'t, 'b: 't>(
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
        self.write_func_decl(line, module, func_idx, indent)?;

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
        write!(f, "): ")?;

        let ret_ty = &module.values[func.ret].ty.body;
        self.write_type_body(f, ret_ty)?;

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

    fn write_impl_decl_tabled<'t, 'b: 't>(
        &mut self,
        table: &'t mut BumpTable<&'b BumpScope<'b>>,
        typedef: &TypeDef,
        impl_decl: &ImplDecl,
        indent: usize,
    ) -> fmt::Result {
        table.start_row();
        let line = table.push_cell();

        write!(line, "{S:indent$}impl ")?;

        let constraints = impl_decl.type_args_constraints.iter().flatten();

        if let TypeDefBody::Builtin(builtin) = &typedef.body {
            self.write_builtin_type(line, builtin, constraints)?;
        } else {
            self.write_name(line, &typedef.name)?;

            if typedef.generics.len() > 0 {
                write!(line, "(")?;
                for (i, ty) in constraints.enumerate() {
                    if i > 0 {
                        write!(line, ", ")?;
                    }
                    self.write_type_body(line, ty)?;
                }
                write!(line, ")")?;
            }
        }

        write!(line, " : ")?;
        self.write_type_ref(line, &TypeRef::new(impl_decl.iface))?;

        if !self.reconstruct {
            let loc_comment = table.push_cell();
            self.write_loc_comment(loc_comment, Some(&impl_decl.loc))?;
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
            InstrBody::GetProperty(v, prop) => write!(f, "GetProperty(v{v}, {prop})"),
            InstrBody::GetField(v, field) => write!(f, "GetField(v{v}, {field})"),
            InstrBody::GetMethod(v, name) => write!(f, "GetMethod(v{v}, {name})"),
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
                    write!(f, "{name}=v{v}")?;
                }
                write!(f, ")")
            }
            InstrBody::Add(a, b) => write!(f, "Add(v{a}, v{b})"),
            InstrBody::Sub(a, b) => write!(f, "Sub(v{a}, v{b})"),
            InstrBody::Mul(a, b) => write!(f, "Mul(v{a}, v{b})"),
            InstrBody::Div(a, b) => write!(f, "Div(v{a}, v{b})"),
            InstrBody::Mod(a, b) => write!(f, "Mod(v{a}, v{b})"),
            InstrBody::Neg(v) => write!(f, "Neg(v{v})"),
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
            InstrBody::Break(block, v) => {
                write!(f, "Break(block{block}")?;
                if let Some(v) = v {
                    write!(f, ", v{v}")?;
                }
                write!(f, ")")
            }
            InstrBody::Continue(block, vs) => {
                write!(f, "Continue(block{block}")?;
                for v in vs {
                    write!(f, ", v{v}")?;
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
            InstrBody::CompileError => write!(f, "CompileError"),
            InstrBody::Dispatch(v, ty_key) => {
                write!(f, "Dispatch(v{v}, ")?;
                self.write_type_ref(f, &TypeRef::new(*ty_key))?;
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
            TypeBody::TypeVar(typevar) => {
                let typevar_def =
                    &self.modules[typevar.mod_idx].typevars[typevar.typevar_idx];
                self.write_name(f, &typevar_def.name)?;
                if !self.reconstruct && self.show_ids {
                    write!(f, " (typevar {}-{})", typevar.mod_idx, typevar.typevar_idx)?;
                }
            }
        }
        Ok(())
    }

    fn write_type_ref(&mut self, f: &mut impl Write, type_ref: &TypeRef) -> fmt::Result {
        let typedef = type_ref.get_typedef(self.modules);

        if let TypeDefBody::Builtin(builtin) = &typedef.body {
            self.write_builtin_type(
                f,
                builtin,
                type_ref.args.iter().map(|arg| &arg.body),
            )?;
        } else {
            self.write_name(f, &typedef.name)?;

            if !type_ref.args.is_empty() {
                write!(f, "(")?;
                for (i, arg) in type_ref.args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    self.write_type_body(f, &arg.body)?;
                }
                write!(f, ")")?;
            }
        }

        if let TypeRefKey::Custom { mod_idx, idx } = type_ref.key
            && self.show_ids
        {
            write!(f, " (type {}-{})", mod_idx, idx)?;
        }

        Ok(())
    }

    fn write_builtin_type<'b>(
        &mut self,
        f: &mut impl Write,
        builtin: &BuiltinType,
        args: impl IntoIterator<Item = impl Borrow<TypeBody>>,
    ) -> fmt::Result {
        let mut args = args.into_iter().peekable();

        match builtin {
            BuiltinType::Void => write!(f, "void")?,
            BuiltinType::Never => write!(f, "never")?,
            BuiltinType::Bool => write!(f, "bool")?,
            BuiltinType::AnyOpaque => write!(f, "AnyOpaque")?,
            BuiltinType::AnyNumber => write!(f, "AnyNumber")?,
            BuiltinType::AnySignedNumber => write!(f, "AnySignedNumber")?,
            BuiltinType::AnyFloat => write!(f, "AnyFloat")?,
            BuiltinType::I8 => write!(f, "i8")?,
            BuiltinType::I16 => write!(f, "i16")?,
            BuiltinType::I32 => write!(f, "i32")?,
            BuiltinType::I64 => write!(f, "i64")?,
            BuiltinType::U8 => write!(f, "u8")?,
            BuiltinType::U16 => write!(f, "u16")?,
            BuiltinType::U32 => write!(f, "u32")?,
            BuiltinType::U64 => write!(f, "u64")?,
            BuiltinType::USize => write!(f, "usize")?,
            BuiltinType::F32 => write!(f, "f32")?,
            BuiltinType::F64 => write!(f, "f64")?,
            BuiltinType::String => write!(f, "str")?,
            BuiltinType::Array => {
                write!(f, "[")?;
                match args.next() {
                    Some(ty) => self.write_type_body(f, ty.borrow())?,
                    None => write!(f, "_")?,
                }
                write!(f, "]")?;
            }
            BuiltinType::Ptr => write!(f, "Ptr")?,
        }
        if args.peek().is_some() {
            write!(f, "(")?;
            for (i, ty) in args.enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                self.write_type_body(f, ty.borrow())?;
            }
            write!(f, ")")?;
        }
        Ok(())
    }

    fn write_typevar(
        &mut self,
        f: &mut impl Write,
        module: &Module,
        typevar_idx: usize,
    ) -> fmt::Result {
        let typevar_def = &module.typevars[typevar_idx];
        self.write_name(f, &typevar_def.name)?;
        if self.show_ids {
            write!(f, " (typevar {typevar_idx})")?;
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
                self.write_name(f, &func.name)?;
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
            Some(global) => {
                self.write_name(f, &global.name)?;
                if self.show_ids {
                    write!(f, " (global {mod_idx}-{global_idx})")?;
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
