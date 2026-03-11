use std::collections::HashMap;

use derive_ctor::ctor;
use itertools::Itertools;

use super::CodeTransformStep;
use crate::bytecode as b;
use crate::context::BuildContext;

#[derive(Clone, Copy, ctor)]
pub struct InstantiateGenericFuncsStep<'a> {
    ctx: &'a BuildContext,
}

impl<'a> CodeTransformStep for InstantiateGenericFuncsStep<'a> {
    #[tracing::instrument(skip(self))]
    fn transform(&mut self, mod_idx: usize, cursor: &mut b::BlockCursor) {
        let (args, func_mod_idx, func_idx) = {
            let modules = &self.ctx.lock_modules();
            let instr = cursor.instr(&modules[mod_idx]).unwrap();

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
        let instr = cursor.instr_mut(&mut modules[mod_idx]).unwrap();
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

/// Transformer that remaps values and substitutes typevars during generic
/// function instantiation.
#[derive(ctor)]
struct GenericInstantiationTransformer<'a> {
    substitutions: &'a HashMap<b::TypeVarIdx, b::Type>,
    #[ctor(default)]
    value_remap:   HashMap<b::ValueIdx, b::ValueIdx>,
}

impl b::BlockTransformer for GenericInstantiationTransformer<'_> {
    fn remap_instr(&mut self, module: &mut b::Module, instr: &mut b::Instr) {
        for res in &mut instr.results {
            let ty = &module.values[*res].ty;
            if let Some(new_ty) = ty.substitute_typevar(&self.substitutions) {
                *res = *self.value_remap.entry(*res).or_insert_with(|| {
                    let mut val = module.values[*res].clone();
                    val.ty = new_ty;
                    module.add_value(val)
                });
            }
        }

        instr.body.remap_values(&self.value_remap);
    }
}

fn remap_func(
    module: &mut b::Module,
    func_idx: usize,
    substitutions: &HashMap<b::TypeVarIdx, b::Type>,
) -> usize {
    let mut new_func = module.funcs[func_idx].clone();
    new_func.generics = Vec::new();
    new_func.generic_instantiations = HashMap::new();

    let mut transformer = GenericInstantiationTransformer::new(substitutions);

    new_func.params = new_func
        .params
        .iter()
        .map(|&param_idx| {
            let ty = &module.values[param_idx].ty;
            if let Some(new_ty) = ty.substitute_typevar(substitutions) {
                let mut val = module.values[param_idx].clone();
                val.ty = new_ty;
                let new_idx = module.add_value(val);
                transformer.value_remap.insert(param_idx, new_idx);
                new_idx
            } else {
                param_idx
            }
        })
        .collect_vec();

    let ret_ty = &module.values[new_func.ret].ty;
    new_func.ret = if let Some(new_ty) = ret_ty.substitute_typevar(substitutions) {
        let mut ret_val = module.values[new_func.ret].clone();
        ret_val.ty = new_ty;
        let new_ret = module.add_value(ret_val);
        transformer.value_remap.insert(new_func.ret, new_ret);
        new_ret
    } else {
        new_func.ret
    };

    new_func.body = module.clone_block_tree(new_func.body, &mut transformer);

    module.add_func(new_func)
}
