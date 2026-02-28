use std::collections::HashSet;

use derive_new::new;

use super::BuildContext;
use crate::{bytecode as b, errors};

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

        let main_global_def = &modules[main_global.0].globals[main_global.1];
        let main_ty = &modules[main_global.0].values[main_global_def.value].ty;

        let main_v = self.add_value(main_ty.body.clone());
        let entry_v = self.add_value(b::TypeBody::Void);

        let mut body = vec![b::Instr::get_global(
            main_global.0,
            main_global.1,
            main_v,
            None,
        )];

        if let Err(e) = self.add_print(&mut body, main_v) {
            self.ctx.push_error(e);
            return self;
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

    fn add_print(
        &mut self,
        body: &mut Vec<b::Instr>,
        main_v: b::ValueIdx,
    ) -> Result<(), errors::Error> {
        let Some(main_global) = *self.ctx.main.read().unwrap() else {
            return Ok(());
        };

        let modules = self.ctx.lock_modules();

        let main_global_def = &modules[main_global.0].globals[main_global.1];
        let main_ty = &modules[main_global.0].values[main_global_def.value].ty;

        let str_ty = b::Type::new(b::TypeBody::String, None);
        if main_ty.intersection(&str_ty, &*modules).is_some() {
            self.add_print_str(body, main_v, &*modules);
            return Ok(());
        }

        let array_ty = b::Type::new(b::TypeBody::Array(str_ty.clone().into()), None);
        if main_ty.intersection(&array_ty, &*modules).is_some() {
            self.add_print_array(body, main_v, &*modules);
            return Ok(());
        }

        let array_2d_ty = b::Type::new(b::TypeBody::Array(array_ty.clone().into()), None);
        if main_ty.intersection(&array_2d_ty, &*modules).is_some() {
            self.add_print_array_2d(body, main_v, &*modules);
            return Ok(());
        }

        Err(errors::Error::new(
            errors::UnexpectedType::new(
                vec![str_ty.clone(), array_ty.clone(), array_2d_ty.clone()],
                main_ty.clone(),
            )
            .into(),
            Some(main_global_def.loc),
        ))
    }

    fn add_print_str(
        &mut self,
        body: &mut Vec<b::Instr>,
        v: usize,
        modules: &[b::Module],
    ) {
        let core_mod_idx = self.ctx.core_mod_idx.expect("core should be defined");

        let (print_func_idx, print_func) = modules[core_mod_idx]
            .get_func("internal_print")
            .expect("core.internal_print should be defined");

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

        let str_v = self.add_value(b::TypeBody::String);
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

    fn add_print_array_2d<'s>(
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

        let str_array_v = self.add_value(b::TypeBody::Array(
            b::Type::new(b::TypeBody::String, None).into(),
        ));
        then_body.push(b::Instr::array_index(v, idx_v, str_array_v, None));

        self.add_print_array(&mut then_body, str_array_v, &*modules);

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
