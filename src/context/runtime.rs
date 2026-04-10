use std::collections::HashSet;

use derive_ctor::ctor;

use super::BuildContext;
use crate::{bytecode as b, errors};

#[derive(ctor)]
pub struct RuntimeBuilder<'a> {
    ctx: &'a BuildContext,
    #[ctor(default)]
    values: Vec<b::Value>,
    #[ctor(default)]
    funcs: Vec<b::Func>,
    #[ctor(default)]
    blocks: Vec<b::Block>,
    #[ctor(default)]
    entry_func_idx: Option<usize>,
}
impl<'a> RuntimeBuilder<'a> {
    pub fn build(self) -> Option<(usize, usize)> {
        let entry_func_idx = self.entry_func_idx?;

        let mut modules = self.ctx.lock_modules_mut();

        let idx = modules.len();
        modules.push(b::Module {
            name: b::Name::from_ident("runtime", b::NameIdentKind::Module, None),
            idx,
            values: self.values,
            funcs: self.funcs,
            blocks: self.blocks,
            globals: vec![],
            typedefs: vec![],
            typevars: vec![],
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

        let ret_v = self.add_value(b::TypeBody::Void);
        let main_v = self.add_value(main_ty.body.clone());

        let body_block = self.add_block();

        self.blocks[body_block].extend([b::Instr::get_global(
            main_global.0,
            main_global.1,
            main_v,
            None,
        )]);

        if let Err(e) = self.add_print_main(body_block, main_v) {
            self.ctx.push_error(e);
            return self;
        }

        let entry_idx = self.funcs.len();
        self.funcs.push(b::Func {
            name: b::Name::from_ident("entry", b::NameIdentKind::Func, None),
            body: body_block,
            params: vec![],
            ret: ret_v,
            method: None,
            extrn: None,
            is_entry: true,
            is_virt: false,
            loc: None,
            generics: vec![],
            generic_instantiations: std::collections::HashMap::new(),
        });

        self.entry_func_idx = Some(entry_idx);
        self
    }

    fn add_print_main(
        &mut self,
        block: b::BlockIdx,
        main_v: b::ValueIdx,
    ) -> Result<(), errors::Error> {
        let Some(main_global) = *self.ctx.main.read().unwrap() else {
            return Ok(());
        };

        let modules = self.ctx.lock_modules();

        let main_global_def = &modules[main_global.0].globals[main_global.1];
        let main_ty = &modules[main_global.0].values[main_global_def.value].ty;

        let str_ty = b::Type::new(b::TypeBody::String, None);
        let array_ty = b::Type::new(b::TypeBody::Array(str_ty.clone().into()), None);
        let array_2d_ty = b::Type::new(b::TypeBody::Array(array_ty.clone().into()), None);

        if main_ty.intersection(&str_ty, &*modules).is_none()
            && main_ty.intersection(&array_ty, &*modules).is_none()
            && main_ty.intersection(&array_2d_ty, &*modules).is_none()
        {
            return Err(errors::Error::new(
                errors::UnexpectedType::new(
                    vec![&str_ty, &array_ty, &array_2d_ty],
                    main_ty,
                    &modules,
                    &self.ctx.cfg,
                )
                .into(),
                Some(main_global_def.loc),
            ));
        }

        self.add_print(block, Some(block), main_v, &*modules);
        Ok(())
    }

    fn add_print(
        &mut self,
        block: b::BlockIdx,
        result_block: Option<b::BlockIdx>,
        v: b::ValueIdx,
        modules: &[b::Module],
    ) {
        let ty = &self.values[v].ty;

        if matches!(ty.body, b::TypeBody::Array(_)) {
            self.add_print_array(block, result_block, v, &*modules);
        } else {
            self.add_print_str(block, v, &*modules);
            if let Some(result_block) = result_block {
                self.blocks[block].extend([b::Instr::break_(result_block, None, None)]);
            }
        }
    }

    fn add_print_str(&mut self, block: b::BlockIdx, v: usize, modules: &[b::Module]) {
        let core_mod_idx = self.ctx.core_mod_idx.expect("core should be defined");

        let (print_func_idx, print_func) = modules[core_mod_idx]
            .get_func("internal_print")
            .expect("core.internal_print should be defined");

        let print_ty = &modules[core_mod_idx].values[print_func.ret].ty;
        let print_v = self.add_value(print_ty.body.clone());

        self.blocks[block].extend([b::Instr::call(
            core_mod_idx,
            print_func_idx,
            vec![v],
            print_v,
            None,
        )]);
    }

    fn add_print_array<'s>(
        &mut self,
        block: b::BlockIdx,
        result_block: Option<b::BlockIdx>,
        v: usize,
        modules: &[b::Module],
    ) {
        let len_v = self.add_value(b::TypeBody::USize);
        let zero_v = self.add_value(b::TypeBody::USize);
        let one_v = self.add_value(b::TypeBody::USize);
        let idx_v = self.add_value(b::TypeBody::USize);
        let cond_v = self.add_value(b::TypeBody::Bool);

        let loop_block = self.add_block();
        let then_block = self.add_block();
        let else_block = self.add_block();

        self.blocks[block].extend([
            b::Instr::array_len(v, len_v, None),
            b::Instr::create_number("0".to_string(), zero_v, None),
            b::Instr::create_number("1".to_string(), one_v, None),
            b::Instr::loop_(vec![(idx_v, zero_v)], loop_block, None, None),
        ]);

        self.blocks[loop_block].extend(vec![
            b::Instr::lt(idx_v, len_v, cond_v, None),
            b::Instr::if_(cond_v, then_block, else_block, None, None),
        ]);

        let b::TypeBody::Array(item_ty) = &self.values[v].ty.body else {
            panic!("type should be an array type");
        };

        let str_v = self.add_value(item_ty.body.clone());
        let new_idx_v = self.add_value(b::TypeBody::USize);

        self.blocks[then_block].extend([b::Instr::array_index(v, idx_v, str_v, None)]);

        self.add_print(then_block, None, str_v, &*modules);

        self.blocks[then_block].extend([
            b::Instr::add(idx_v, one_v, new_idx_v, None),
            b::Instr::continue_(loop_block, vec![new_idx_v], None),
        ]);

        self.blocks[else_block].extend([b::Instr::break_(
            result_block.unwrap_or(loop_block),
            None,
            None,
        )]);
    }

    fn add_block(&mut self) -> b::BlockIdx {
        self.blocks.push(b::Block::default());
        self.blocks.len() - 1
    }

    pub fn add_value(&mut self, ty: b::TypeBody) -> b::ValueIdx {
        self.values
            .push(b::Value::new(b::Type::new(ty, None), None));
        self.values.len() - 1
    }
}
