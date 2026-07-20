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

#[derive(Debug, ctor)]
struct DispatchedTypeArgs {
    ty_key: b::TypeRefKey,
    iface_key: b::TypeRefKey,
    args: Vec<b::TypeBody>,
    #[ctor(default)]
    substitutions: HashMap<b::TypeVarIdx, b::Type>,
    #[ctor(default)]
    methods_substitutions: HashMap<usize, HashMap<b::TypeVarIdx, b::Type>>,
}

#[derive(Clone, Copy, ctor)]
pub struct InstantiateGenericFuncsStep<'a> {
    ctx: &'a BuildContext,
}

impl<'a> CodeTransformStep for InstantiateGenericFuncsStep<'a> {
    #[tracing::instrument(skip(self))]
    fn transform(&mut self, mod_idx: usize, cursor: &mut b::BlockCursor) {
        let modules = &mut self.ctx.lock_modules_mut();
        self.transform_func(mod_idx, cursor, modules);
        self.transform_method(mod_idx, cursor, modules);
        self.transform_dispatched_args(mod_idx, cursor, modules);
    }
}

impl<'a> InstantiateGenericFuncsStep<'a> {
    #[tracing::instrument(skip(self))]
    fn transform_func(
        &mut self,
        mod_idx: usize,
        cursor: &mut b::BlockCursor,
        modules: &mut [b::Module],
    ) {
        let instr = cursor.instr(&modules[mod_idx]).unwrap();

        let (args, func_mod_idx, func_idx) = match &instr.body {
            &b::InstrBody::Call(func_mod_idx, func_idx, ref args) => {
                let args = args.clone();
                (FuncArgs::call(args), func_mod_idx, func_idx)
            }
            &b::InstrBody::GetFunc(func_mod_idx, func_idx) => {
                assert!(instr.results.len() == 1);
                let result = instr.results[0];
                (FuncArgs::get_func(result), func_mod_idx, func_idx)
            }
            _ => return,
        };

        let Some((new_func_idx, _)) =
            self.instantiate_call(cursor, modules, mod_idx, func_mod_idx, func_idx, args)
        else {
            return;
        };

        let instr = cursor.instr_mut(&mut modules[mod_idx]).unwrap();
        match &mut instr.body {
            b::InstrBody::Call(_, func_idx, _) => {
                *func_idx = new_func_idx;
            }
            b::InstrBody::GetFunc(_, func_idx) => {
                *func_idx = new_func_idx;
            }
            _ => {}
        };
    }

    #[tracing::instrument(skip(self))]
    fn transform_method(
        &mut self,
        mod_idx: usize,
        cursor: &mut b::BlockCursor,
        modules: &mut [b::Module],
    ) {
        let instr = cursor.instr(&modules[mod_idx]).unwrap();

        let &b::InstrBody::GetMethod(source, idx) = &instr.body else {
            return;
        };

        assert!(instr.results.len() == 1);
        let result = instr.results[0];

        let source_ty = &modules[mod_idx].values[source].ty;
        let b::TypeBody::TypeRef(type_ref) = &source_ty.body else {
            return;
        };

        let type_ref_key = type_ref.key;
        let typedef = type_ref_key.get_typedef(modules);
        let method = &typedef.methods[idx];
        let (func_mod_idx, func_idx) = method.func_ref;

        let Some((new_func_idx, tys)) = self.instantiate_call(
            cursor,
            modules,
            mod_idx,
            func_mod_idx,
            func_idx,
            FuncArgs::get_method(source, result),
        ) else {
            return;
        };

        let typedef = type_ref_key.get_typedef(modules);
        let method = &typedef.methods[idx];

        let new_method_name =
            b::Name::from_ident(&method.name, b::NameIdentKind::Func, None)
                .with_type_params(
                    tys.into_iter().map(|body| b::Type::new(body, None)),
                    None,
                )
                .formated(&modules, &self.ctx.cfg, None);

        let typedef = &mut type_ref_key.get_typedef_mut(modules);

        let mut new_method = typedef.methods[idx].clone();
        new_method.name = new_method_name;
        new_method.func_ref.1 = new_func_idx;

        let new_method_idx = typedef.methods.len();
        typedef.methods.push(new_method);

        let instr = cursor.instr_mut(&mut modules[mod_idx]).unwrap();
        if let b::InstrBody::GetMethod(_, idx) = &mut instr.body {
            *idx = new_method_idx;
        }
    }

    #[tracing::instrument(skip(self))]
    fn transform_dispatched_args(
        &mut self,
        mod_idx: usize,
        cursor: &mut b::BlockCursor,
        modules: &mut [b::Module],
    ) {
        let instr = cursor.instr(&modules[mod_idx]).unwrap();

        let items = match &instr.body {
            &b::InstrBody::Call(func_mod_idx, func_idx, ref args) => {
                let func = &modules[func_mod_idx].funcs[func_idx];

                izip!(args, &func.params)
                    .filter_map(|(&arg, &param)| {
                        let arg_ty = &modules[mod_idx].values[arg].ty;
                        let param_ty = &modules[func_mod_idx].values[param].ty;
                        self.get_dispatched_type_args(arg_ty, param_ty, modules)
                    })
                    .collect_vec()
            }
            b::InstrBody::IndirectCall(_, args) => {
                let b::TypeBody::Func(func) = &modules[mod_idx].values[args[0]].ty.body
                else {
                    return;
                };
                izip!(args, &func.params)
                    .filter_map(|(&arg, param_ty)| {
                        let arg_ty = &modules[mod_idx].values[arg].ty;
                        self.get_dispatched_type_args(arg_ty, param_ty, modules)
                    })
                    .collect_vec()
            }
            _ => return,
        };

        for item in items {
            let mut new_methods_idxs = HashMap::new();

            for (method_idx, substitutions) in item.methods_substitutions {
                let typedef = item.ty_key.get_typedef(modules);
                let method = &typedef.methods[method_idx];

                let (new_func_idx, tys) = self.instantiate_generic_func(
                    cursor,
                    modules,
                    method.func_ref.0,
                    method.func_ref.1,
                    &substitutions,
                );

                let typedef = item.ty_key.get_typedef(modules);
                let method = &typedef.methods[method_idx];

                let new_method_name =
                    b::Name::from_ident(&method.name, b::NameIdentKind::Func, None)
                        .with_type_params(
                            tys.into_iter().map(|body| b::Type::new(body, None)),
                            None,
                        )
                        .formated(&modules, &self.ctx.cfg, None);

                let typedef = item.ty_key.get_typedef_mut(modules);

                let mut new_method = typedef.methods[method_idx].clone();
                new_method.name = new_method_name;
                new_method.func_ref.1 = new_func_idx;

                let new_method_idx = typedef.methods.len();
                typedef.methods.push(new_method);
                new_methods_idxs.insert(method_idx, new_method_idx);
            }

            let typedef = item.ty_key.get_typedef_mut(modules);
            let impl_idx = typedef.impls.iter().position(|d| d.iface == item.iface_key);

            if let Some(impl_idx) = impl_idx
                && !typedef.impls[impl_idx]
                    .generic_instantiations
                    .contains_key(&item.args)
            {
                let impl_decl = &typedef.impls[impl_idx];

                let mut iface_args = impl_decl.iface_args.clone();
                for iface_arg in &mut iface_args {
                    match iface_arg.substitute_typevar(&item.substitutions) {
                        Some(ty) => *iface_arg = ty,
                        None => {}
                    }
                }

                let mut new_impl = b::ImplDecl::new(
                    item.iface_key,
                    iface_args,
                    Some(item.args.clone()),
                    impl_decl.loc,
                );
                new_impl.methods = impl_decl
                    .methods
                    .iter()
                    .map(|i| *new_methods_idxs.get(i).unwrap_or(i))
                    .collect();

                let new_impl_idx = typedef.impls.len();
                typedef.impls.push(new_impl);

                typedef.impls[impl_idx]
                    .generic_instantiations
                    .insert(item.args.clone(), new_impl_idx);
            }
        }
    }

    #[tracing::instrument(skip(self))]
    fn instantiate_call<'b>(
        &mut self,
        cursor: &mut b::BlockCursor,
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
            cursor,
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
                co.yield_(&modules[mod_idx].values[source].ty).await;

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
        cursor: &mut b::BlockCursor,
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

        let new_func = remap_func(module, func_idx, &substitutions);
        let new_func_idx = cursor.add_func(module, new_func);

        module.funcs[func_idx]
            .generic_instantiations
            .insert(tys.clone(), new_func_idx);

        (new_func_idx, tys)
    }

    #[tracing::instrument(skip(self))]
    fn get_dispatched_type_args(
        &self,
        ty: &b::Type,
        param_ty: &b::Type,
        modules: &[b::Module],
    ) -> Option<DispatchedTypeArgs> {
        if ty.body == param_ty.body {
            return None;
        }

        let b::TypeBody::TypeRef(ty_ref) = &ty.body else {
            return None;
        };
        if ty_ref.args.is_empty() {
            return None;
        }

        let b::TypeBody::TypeRef(iface_ty_ref) = &param_ty.body else {
            return None;
        };
        if ty_ref.is_same_of(iface_ty_ref) {
            return None;
        }

        let typedef = ty_ref.get_typedef(modules);
        let impl_decl = typedef
            .impls
            .iter()
            .find(|v| v.iface == iface_ty_ref.key)
            .unwrap();

        let iface_typedef = iface_ty_ref.get_typedef(modules);
        if !matches!(iface_typedef.body, b::TypeDefBody::Interface) {
            return None;
        }

        let mut result = DispatchedTypeArgs::new(
            ty_ref.key,
            iface_ty_ref.key,
            ty_ref.args.iter().map(|arg| arg.body.clone()).collect(),
        );

        for i in 0..iface_typedef.methods.len() {
            let method_idx = impl_decl.methods[i];
            let method = &typedef.methods[method_idx];

            let func = &modules[method.func_ref.0].funcs[method.func_ref.1];
            let reciever_ty = &modules[method.func_ref.0].values[func.params[0]].ty;

            let mut method_substitutions = HashMap::new();
            reciever_ty.collect_typevar_substitutions(
                ty,
                b::Variance::Covariant,
                &mut method_substitutions,
                modules,
            );

            if !method_substitutions.is_empty() {
                for (&k, v) in &method_substitutions {
                    result.substitutions.insert(k, v.clone());
                }

                result
                    .methods_substitutions
                    .insert(method_idx, method_substitutions);
            }
        }

        Some(result)
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
) -> b::Func {
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

    new_func
}
