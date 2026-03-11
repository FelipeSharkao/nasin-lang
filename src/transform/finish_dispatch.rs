use derive_ctor::ctor;
use itertools::{Itertools, enumerate, izip};

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
        let (params, loc) = {
            let modules = &self.ctx.lock_modules();
            let instr = cursor.instr(&modules[mod_idx]);
            let params = match &instr.body {
                b::InstrBody::Call(call_mod_idx, func_idx, args) => {
                    let args_types = args.iter().map(|v| &modules[mod_idx].values[*v].ty);
                    let params_types = modules[*call_mod_idx].funcs[*func_idx]
                        .params
                        .iter()
                        .map(|v| &modules[*call_mod_idx].values[*v].ty);
                    izip!(args, args_types, params_types).collect_vec()
                }
                b::InstrBody::IndirectCall(v, args) => {
                    let b::TypeBody::Func(func) = &modules[mod_idx].values[*v].ty.body
                    else {
                        return;
                    };
                    let args_types = args.iter().map(|v| &modules[mod_idx].values[*v].ty);
                    izip!(args, args_types, &func.params).collect_vec()
                }
                _ => return,
            };
            let params = params
                .into_iter()
                .enumerate()
                .filter_map(|(i, (v, arg_ty, param_ty))| {
                    if arg_ty.body == param_ty.body {
                        return None;
                    }
                    let b::TypeBody::TypeRef(param_ty_ref) = &param_ty.body else {
                        return None;
                    };
                    let param_ty_def =
                        &modules[param_ty_ref.mod_idx].typedefs[param_ty_ref.idx];
                    if matches!(&param_ty_def.body, b::TypeDefBody::Interface(_)) {
                        Some((i, *v, (param_ty_ref.mod_idx, param_ty_ref.idx)))
                    } else {
                        None
                    }
                })
                .collect_vec();
            (params, instr.loc)
        };

        {
            let modules = &mut self.ctx.lock_modules_mut();
            let module = &mut modules[mod_idx];

            let value_start = module.values.len();
            module.values.extend(params.iter().map(|(_, _, iface)| {
                let ty = b::Type::new(b::TypeRef::new(iface.0, iface.1).into(), None);
                b::Value::new(ty, loc)
            }));

            match &mut cursor.instr_mut(&mut modules[mod_idx]).body {
                b::InstrBody::Call(_, _, args) | b::InstrBody::IndirectCall(_, args) => {
                    for (i, (n, ..)) in params.iter().enumerate() {
                        args[*n] = value_start + i;
                    }
                }
                _ => {}
            };

            for (i, (_, v, iface)) in enumerate(params) {
                cursor.insert_instr(
                    &mut modules[mod_idx],
                    b::Instr::new(b::InstrBody::Dispatch(v, iface.0, iface.1), loc)
                        .with_results([value_start + i]),
                );
            }
        }
    }
}
