use std::collections::HashSet;

use derive_new::new;

use super::BuildContext;
use crate::bytecode as b;

#[derive(new)]
pub struct RuntimeBuilder<'a> {
    ctx: &'a BuildContext,
    #[new(default)]
    values: Vec<b::Value>,
    #[new(default)]
    funcs: Vec<b::Func>,
    #[new(default)]
    entry_func_idx: Option<usize>,
}
impl<'a> RuntimeBuilder<'a> {
    pub fn build(self) -> Option<(usize, usize)> {
        let entry_func_idx = self.entry_func_idx?;

        let mut modules = self.ctx.lock_modules_mut();

        let idx = modules.len();
        modules.push(b::Module {
            idx,
            values: self.values,
            funcs: self.funcs,
            globals: vec![],
            typedefs: vec![],
            sources: HashSet::new(),
        });

        Some((idx, entry_func_idx))
    }

    pub fn add_entry(mut self) -> Self {
        if self.ctx.core_mod_idx.is_none() {
            return self;
        }

        let Some(main_global) = *self.ctx.main.read().unwrap() else {
            return self;
        };

        let modules = self.ctx.lock_modules();

        let main_ty = &modules[main_global.0].values
            [modules[main_global.0].globals[main_global.1].value]
            .ty;

        let main_v = self.add_value(main_ty.body.clone());
        let entry_v = self.add_value(b::TypeBody::Void);

        let mut body = vec![b::Instr::get_global(
            main_global.0,
            main_global.1,
            main_v,
            None,
        )];

        match &main_ty.body {
            b::TypeBody::String(..) => {
                self.add_print_str(&mut body, main_v, &*modules);
            }
            b::TypeBody::Array(array_ty)
                if matches!(&array_ty.item.body, b::TypeBody::String(..)) =>
            {
                self.add_print_array(&mut body, main_v, &*modules);
            }
            _ => todo!("better error message for main return type"),
        }

        body.push(b::Instr::break_(None, None));

        let entry_idx = self.funcs.len();
        self.funcs.push(b::Func {
            name: "entry".to_string(),
            body,
            params: vec![],
            ret: entry_v,
            method: None,
            extrn: None,
            is_entry: true,
            is_virt: false,
            loc: None,
        });

        self.entry_func_idx = Some(entry_idx);
        self
    }

    fn add_print_str(
        &mut self,
        body: &mut Vec<b::Instr>,
        v: usize,
        modules: &[b::Module],
    ) {
        let core_mod_idx = self.ctx.core_mod_idx.expect("core should be defined");

        let (print_func_idx, print_func) = modules[core_mod_idx]
            .get_func("print")
            .expect("core.print should be defined");

        let print_ty = &modules[core_mod_idx].values[print_func.ret].ty;
        let print_v = self.add_value(print_ty.body.clone());

        body.push(b::Instr::call(
            core_mod_idx,
            print_func_idx,
            vec![v],
            print_v,
            None,
        ));
    }

    fn add_print_array<'s>(
        &mut self,
        body: &mut Vec<b::Instr>,
        v: usize,
        modules: &[b::Module],
    ) {
        let len_v = self.add_value(b::TypeBody::USize);
        body.push(b::Instr::array_len(v, len_v, None));

        let zero_v = self.add_value(b::TypeBody::USize);
        body.push(b::Instr::create_number("0".to_string(), zero_v, None));

        let one_v = self.add_value(b::TypeBody::USize);
        body.push(b::Instr::create_number("1".to_string(), one_v, None));

        let idx_v = self.add_value(b::TypeBody::USize);
        let cond_v = self.add_value(b::TypeBody::Bool);

        let mut then_body = vec![];

        let str_v = self.add_value(b::TypeBody::String(b::StringType::new(None)));
        then_body.push(b::Instr::array_index(v, idx_v, str_v, None));

        self.add_print_str(&mut then_body, str_v, &*modules);

        let new_idx_v = self.add_value(b::TypeBody::USize);
        then_body.push(b::Instr::add(idx_v, one_v, new_idx_v, None));
        then_body.push(b::Instr::continue_(vec![new_idx_v], None));

        body.push(b::Instr::loop_(
            vec![(idx_v, zero_v)],
            vec![
                b::Instr::lt(idx_v, len_v, cond_v, None),
                b::Instr::if_(cond_v, then_body, vec![], None, None),
                b::Instr::break_(None, None),
            ],
            None,
            None,
        ));
    }

    pub fn add_value(&mut self, ty: b::TypeBody) -> b::ValueIdx {
        let v = self.values.len();
        self.values
            .push(b::Value::new(b::Type::new(ty, None), None));
        v
    }
}
