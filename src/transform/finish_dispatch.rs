use std::collections::HashMap;

use derive_ctor::ctor;
use itertools::{Itertools, izip};

use super::CodeTransformStep;
use crate::bytecode as b;
use crate::context::BuildContext;

#[derive(Clone, Copy, ctor)]
pub struct FinishDispatchStep<'a> {
    ctx: &'a BuildContext,
}

impl<'a> CodeTransformStep for FinishDispatchStep<'a> {
    #[tracing::instrument(skip(self))]
    fn transform(&mut self, mod_idx: usize, cursor: &mut b::BlockCursor) {
        let modules = &mut self.ctx.lock_modules_mut();
        let module = &modules[mod_idx];
        let instr = cursor.instr(module).unwrap();
        let loc = instr.loc;

        let params = match &instr.body {
            b::InstrBody::Call(call_mod_idx, func_idx, args) => {
                let params_types = modules[*call_mod_idx].funcs[*func_idx]
                    .params
                    .iter()
                    .map(|v| &modules[*call_mod_idx].values[*v].ty);
                collect_params(args, params_types, modules)
            }
            b::InstrBody::IndirectCall(v, args) => {
                let b::TypeBody::Func(func) = &module.values[*v].ty.body else {
                    return;
                };
                collect_params(args, &func.params, modules)
            }
            _ => return,
        };

        let module = &mut modules[mod_idx];

        let mut remap = HashMap::new();
        for (v, iface_mod_idx, iface_idx) in params {
            remap.entry(v).or_insert_with(|| {
                let ty =
                    b::Type::new(b::TypeRef::new(iface_mod_idx, iface_idx).into(), None);

                let idx = module.add_value(b::Value::new(ty, loc));
                cursor.insert_instr(
                    module,
                    b::Instr::new(
                        b::InstrBody::Dispatch(v, iface_mod_idx, iface_idx),
                        loc,
                    )
                    .with_results([idx]),
                );
                cursor.step(module);

                idx
            });
        }

        cursor.instr_mut(module).unwrap().body.remap_values(&remap);
    }
}

fn collect_params<'m>(
    args: impl IntoIterator<Item = &'m b::ValueIdx>,
    params: impl IntoIterator<Item = &'m b::Type>,
    modules: &'m [b::Module],
) -> Vec<(b::ValueIdx, usize, usize)> {
    izip!(args, params)
        .filter_map(|(&arg, param_ty)| {
            let b::TypeBody::TypeRef(param_ty_ref) = &param_ty.body else {
                return None;
            };
            let param_ty_def = &modules[param_ty_ref.mod_idx].typedefs[param_ty_ref.idx];
            if !matches!(&param_ty_def.body, b::TypeDefBody::Interface(_)) {
                return None;
            }
            Some((arg, param_ty_ref.mod_idx, param_ty_ref.idx))
        })
        .collect_vec()
}
