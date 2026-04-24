use std::collections::HashMap;

use derive_ctor::ctor;
use genawaiter::rc::Gen;
use itertools::{Itertools, izip};

use super::CodeTransformStep;
use crate::bytecode as b;
use crate::context::BuildContext;

#[derive(Clone, Debug, ctor)]
enum FuncArgs {
    Call {
        args: Vec<b::ValueIdx>,
    },
    GetFunc {
        result: b::ValueIdx,
    },
    GetMethod {
        source: b::ValueIdx,
        result: b::ValueIdx,
    },
}

#[derive(Clone, Copy, ctor)]
pub struct InstantiateGenericFuncsStep<'a> {
    ctx: &'a BuildContext,
}

impl<'a> CodeTransformStep for InstantiateGenericFuncsStep<'a> {
    #[tracing::instrument(skip(self))]
    fn transform(&mut self, mod_idx: usize, cursor: &mut b::BlockCursor) {
        let modules = &mut self.ctx.lock_modules_mut();
        let instr = cursor.instr(&modules[mod_idx]).unwrap();
        match &instr.body {
            &b::InstrBody::Call(func_mod_idx, func_idx, ref args) => {
                let args = args.clone();
                let Some((new_func_idx, _)) = self.instantiate_call(
                    modules,
                    mod_idx,
                    func_mod_idx,
                    func_idx,
                    FuncArgs::call(args),
                ) else {
                    return;
                };

                let instr = cursor.instr_mut(&mut modules[mod_idx]).unwrap();
                if let b::InstrBody::Call(_, func_idx, _) = &mut instr.body {
                    *func_idx = new_func_idx;
                }
            }
            &b::InstrBody::GetFunc(func_mod_idx, func_idx) => {
                assert!(instr.results.len() == 1);
                let result = instr.results[0];

                let Some((new_func_idx, _)) = self.instantiate_call(
                    modules,
                    mod_idx,
                    func_mod_idx,
                    func_idx,
                    FuncArgs::get_func(result),
                ) else {
                    return;
                };

                let instr = cursor.instr_mut(&mut modules[mod_idx]).unwrap();
                if let b::InstrBody::GetFunc(_, func_idx) = &mut instr.body {
                    *func_idx = new_func_idx;
                }
            }
            &b::InstrBody::GetProperty(source, ref prop)
            | &b::InstrBody::GetMethod(source, ref prop) => {
                assert!(instr.results.len() == 1);
                let result = instr.results[0];

                let source_ty = &modules[mod_idx].values[source].ty;
                let b::TypeBody::TypeRef(type_ref) = &source_ty.body else {
                    return;
                };

                let ty_mod_idx = type_ref.mod_idx;
                let ty_idx = type_ref.idx;
                let typedef = &modules[ty_mod_idx].typedefs[ty_idx];

                let prop = prop.clone();
                let Some(method) = typedef.get_method(&prop) else {
                    return;
                };

                let (func_mod_idx, func_idx) = method.func_ref;

                let Some((new_func_idx, tys)) = self.instantiate_call(
                    modules,
                    mod_idx,
                    func_mod_idx,
                    func_idx,
                    FuncArgs::get_method(source, result),
                ) else {
                    return;
                };

                let typedef = &mut modules[ty_mod_idx].typedefs[ty_idx];
                let mut new_method = typedef.get_method(&prop).unwrap().clone();
                new_method.func_ref.1 = new_func_idx;

                let new_prop = b::Name::from_ident(prop, b::NameIdentKind::Func, None)
                    .with_type_params(
                        tys.into_iter().map(|body| b::Type::new(body, None)),
                        None,
                    )
                    .to_string();

                typedef.add_method(new_prop.clone(), new_method);

                let instr = cursor.instr_mut(&mut modules[mod_idx]).unwrap();
                if let b::InstrBody::GetProperty(_, prop)
                | b::InstrBody::GetMethod(_, prop) = &mut instr.body
                {
                    *prop = new_prop;
                }
            }
            _ => {}
        }
    }
}

impl<'a> InstantiateGenericFuncsStep<'a> {
    #[tracing::instrument(skip(self))]
    fn instantiate_call<'b>(
        &mut self,
        modules: &mut [b::Module],
        mod_idx: usize,
        func_mod_idx: usize,
        func_idx: usize,
        args: FuncArgs,
    ) -> Option<(usize, Vec<b::TypeBody>)> {
        let type_substitutions =
            self.find_type_substitutions(modules, mod_idx, func_mod_idx, func_idx, args);

        if type_substitutions.is_empty() {
            return None;
        }

        let res = self.instantiate_generic_func(
            modules,
            func_mod_idx,
            func_idx,
            &type_substitutions,
        );

        Some(res)
    }

    #[tracing::instrument(skip(self))]
    fn find_type_substitutions<'b>(
        &self,
        modules: &[b::Module],
        mod_idx: usize,
        func_mod_idx: usize,
        func_idx: usize,
        args: FuncArgs,
    ) -> HashMap<b::TypeVarIdx, b::Type> {
        let func = &modules[func_mod_idx].funcs[func_idx];

        let mut substitutions = HashMap::new();

        let args_tys = Gen::new(async move |co| match args {
            FuncArgs::Call { args } => {
                for arg in args {
                    co.yield_(&modules[mod_idx].values[arg].ty).await;
                }
            }
            FuncArgs::GetFunc { result } => {
                let result_ty = &modules[mod_idx].values[result].ty;
                let b::TypeBody::Func(func_ty) = &result_ty.body else {
                    return;
                };
                for param in &func_ty.params {
                    co.yield_(param).await;
                }
            }
            FuncArgs::GetMethod { source, result } => {
                let func = &modules[func_mod_idx].funcs[func_idx];
                if func.params.len() > 1 {
                    let result_ty = &modules[mod_idx].values[result].ty;
                    let b::TypeBody::Func(func_ty) = &result_ty.body else {
                        return;
                    };

                    for param in &func_ty.params {
                        co.yield_(param).await;
                    }
                }

                co.yield_(&modules[mod_idx].values[source].ty).await;
            }
        });

        for (&param, arg_ty) in izip!(&func.params, args_tys) {
            let param_ty = &modules[func_mod_idx].values[param].ty;
            param_ty.collect_typevar_substitutions(
                arg_ty,
                b::Variance::Covariant,
                &mut substitutions,
                modules,
            );
        }

        substitutions
    }

    #[tracing::instrument(skip(self))]
    fn instantiate_generic_func(
        &mut self,
        modules: &mut [b::Module],
        func_mod_idx: usize,
        func_idx: usize,
        substitutions: &HashMap<b::TypeVarIdx, b::Type>,
    ) -> (usize, Vec<b::TypeBody>) {
        let tys: Vec<b::TypeBody> = substitutions
            .iter()
            .map(|(_, ty)| ty.body.clone())
            .collect();

        if let Some(&existing_idx) = modules[func_mod_idx].funcs[func_idx]
            .generic_instantiations
            .get(&tys)
        {
            return (existing_idx, tys);
        }

        let module = &mut modules[func_mod_idx];

        let new_func_idx = remap_func(module, func_idx, &substitutions);

        module.funcs[func_idx]
            .generic_instantiations
            .insert(tys.clone(), new_func_idx);

        (new_func_idx, tys)
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

    new_func.body =
        module.clone_block_tree(new_func.body, &mut transformer, &mut HashMap::new());

    module.add_func(new_func)
}
