use std::collections::HashMap;

use derive_ctor::ctor;

use super::{CodeTransformCursor, CodeTransformStep};
use crate::bytecode as b;
use crate::context::BuildContext;

#[derive(Clone, Copy, ctor)]
pub struct InstantiateGenericFuncsStep<'a> {
    ctx: &'a BuildContext,
}

impl<'a> CodeTransformStep for InstantiateGenericFuncsStep<'a> {
    #[tracing::instrument(skip(self))]
    fn transform(&mut self, mod_idx: usize, cursor: &mut dyn CodeTransformCursor) {
        let (args, func_mod_idx, func_idx) = {
            let modules = &self.ctx.lock_modules();
            let instr = cursor.get_instr(modules);

            let b::InstrBody::Call(func_mod_idx, func_idx, args) = &instr.body else {
                return;
            };

            let func = &modules[*func_mod_idx].funcs[*func_idx];
            if func.generics.is_empty() {
                return;
            }

            (args.clone(), *func_mod_idx, *func_idx)
        };

        let type_substitutions =
            self.find_type_substitutions(mod_idx, func_mod_idx, func_idx, &args);

        if type_substitutions.is_empty() {
            return;
        }

        let new_func_idx =
            self.instantiate_generic_func(func_mod_idx, func_idx, &type_substitutions);

        let modules = &mut self.ctx.lock_modules_mut();
        let instr = cursor.get_instr_mut(modules);
        if let b::InstrBody::Call(_, _, _) = &mut instr.body {
            instr.body = b::InstrBody::Call(func_mod_idx, new_func_idx, args);
        }
    }
}

impl<'a> InstantiateGenericFuncsStep<'a> {
    #[tracing::instrument(skip(self))]
    fn find_type_substitutions(
        &self,
        call_mod_idx: usize,
        func_mod_idx: usize,
        func_idx: usize,
        args: &[b::ValueIdx],
    ) -> HashMap<b::TypeVarIdx, b::Type> {
        let modules = &self.ctx.lock_modules();
        let func = &modules[func_mod_idx].funcs[func_idx];

        let mut substitutions = HashMap::new();

        for (param_idx, param_value_idx) in func.params.iter().enumerate() {
            let param_ty = &modules[func_mod_idx].values[*param_value_idx].ty;

            let b::TypeBody::TypeVar(tv) = &param_ty.body else {
                continue;
            };

            let arg_ty = &modules[call_mod_idx].values[args[param_idx]].ty;

            substitutions.insert(tv.typevar_idx, arg_ty.clone());
        }

        substitutions
    }

    #[tracing::instrument(skip(self))]
    fn instantiate_generic_func(
        &mut self,
        func_mod_idx: usize,
        func_idx: usize,
        substitutions: &HashMap<b::TypeVarIdx, b::Type>,
    ) -> usize {
        let key: Vec<b::TypeBody> = substitutions
            .iter()
            .map(|(_, ty)| ty.body.clone())
            .collect();

        let modules = &mut self.ctx.lock_modules_mut();

        if let Some(&existing_idx) = modules[func_mod_idx].funcs[func_idx]
            .generic_instantiations
            .get(&key)
        {
            return existing_idx;
        }

        let module = &mut modules[func_mod_idx];

        let new_func_idx = remap_func(module, func_idx, &substitutions);

        module.funcs[func_idx]
            .generic_instantiations
            .insert(key, new_func_idx);

        new_func_idx
    }
}

/// Clone a function and apply typevar substitutions to its parameters, return value and
/// body.
fn remap_func(
    module: &mut b::Module,
    func_idx: usize,
    substitutions: &HashMap<b::TypeVarIdx, b::Type>,
) -> usize {
    let mut new_func = module.funcs[func_idx].clone();

    let mut new_params = Vec::new();
    let mut value_remap = HashMap::new();

    for &param_idx in &new_func.params {
        let mut val = module.values[param_idx].clone();
        substitute_typevar(&mut val.ty, substitutions);
        let new_idx = module.values.len();
        module.values.push(val);
        new_params.push(new_idx);
        value_remap.insert(param_idx, new_idx);
    }

    let mut ret_val = module.values[new_func.ret].clone();
    substitute_typevar(&mut ret_val.ty, substitutions);
    let new_ret = module.values.len();
    module.values.push(ret_val);
    value_remap.insert(new_func.ret, new_ret);

    new_func.generics = Vec::new();
    new_func.params = new_params;
    new_func.ret = new_ret;
    new_func.generic_instantiations = HashMap::new();

    update_body(&mut new_func.body, module, &substitutions, &mut value_remap);

    let new_func_idx = module.funcs.len();
    module.funcs.push(new_func);

    new_func_idx
}

/// Clone body instructions, remapping value indices and substituting typevars
/// in any newly created result values.
fn update_body(
    body: &mut Vec<b::Instr>,
    module: &mut b::Module,
    substitutions: &HashMap<b::TypeVarIdx, b::Type>,
    value_remap: &mut HashMap<b::ValueIdx, b::ValueIdx>,
) {
    for instr in body {
        instr.results = instr
            .results
            .iter()
            .map(|&res| {
                *value_remap.entry(res).or_insert_with(|| {
                    let mut val = module.values[res].clone();
                    substitute_typevar(&mut val.ty, substitutions);
                    let new_idx = module.values.len();
                    module.values.push(val);
                    new_idx
                })
            })
            .collect();

        instr.body.remap_values(value_remap);
    }
}

fn substitute_typevar(ty: &mut b::Type, substitutions: &HashMap<b::TypeVarIdx, b::Type>) {
    if let b::TypeBody::TypeVar(tv) = &ty.body {
        if let Some(new_ty) = substitutions.get(&tv.typevar_idx) {
            *ty = new_ty.clone();
        }
    }
}
