use std::collections::HashMap;

use derive_ctor::ctor;

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
            let instr = cursor.instr(&modules[mod_idx]);

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
        let instr = cursor.instr_mut(&mut modules[mod_idx]);
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
struct GenericInstantiationTransformer<'a> {
    substitutions: &'a HashMap<b::TypeVarIdx, b::Type>,
    value_remap:   HashMap<b::ValueIdx, b::ValueIdx>,
}

impl<'a> GenericInstantiationTransformer<'a> {
    fn new(substitutions: &'a HashMap<b::TypeVarIdx, b::Type>) -> Self {
        Self {
            substitutions,
            value_remap: HashMap::new(),
        }
    }

    fn seed_remap(&mut self, old: b::ValueIdx, new: b::ValueIdx) {
        self.value_remap.insert(old, new);
    }
}

impl b::BlockCloneTransformer for GenericInstantiationTransformer<'_> {
    fn remap_result(&mut self, module: &mut b::Module, old: b::ValueIdx) -> b::ValueIdx {
        *self.value_remap.entry(old).or_insert_with(|| {
            let mut val = module.values[old].clone();
            substitute_typevar(&mut val.ty, self.substitutions);
            let new_idx = module.values.len();
            module.values.push(val);
            new_idx
        })
    }

    fn remap_instr_values(&self, body: &mut b::InstrBody) {
        body.remap_values(&self.value_remap);
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
    let mut transformer = GenericInstantiationTransformer::new(substitutions);

    let mut new_params = Vec::new();
    for &param_idx in &new_func.params {
        let mut val = module.values[param_idx].clone();
        substitute_typevar(&mut val.ty, substitutions);
        let new_idx = module.values.len();
        module.values.push(val);
        new_params.push(new_idx);
        transformer.seed_remap(param_idx, new_idx);
    }

    let mut ret_val = module.values[new_func.ret].clone();
    substitute_typevar(&mut ret_val.ty, substitutions);
    let new_ret = module.values.len();
    module.values.push(ret_val);
    transformer.seed_remap(new_func.ret, new_ret);

    new_func.generics = Vec::new();
    new_func.params = new_params;
    new_func.ret = new_ret;
    new_func.generic_instantiations = HashMap::new();

    new_func.body = module.clone_block_tree(new_func.body, &mut transformer);

    let new_func_idx = module.funcs.len();
    module.funcs.push(new_func);

    new_func_idx
}

fn substitute_typevar(ty: &mut b::Type, substitutions: &HashMap<b::TypeVarIdx, b::Type>) {
    if let b::TypeBody::TypeVar(tv) = &ty.body {
        if let Some(new_ty) = substitutions.get(&tv.typevar_idx) {
            *ty = new_ty.clone();
        }
    }
}
