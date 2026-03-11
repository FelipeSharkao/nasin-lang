use derive_ctor::ctor;

use super::CodeTransformStep;
use crate::bytecode as b;
use crate::context::BuildContext;

#[derive(Clone, Copy, ctor)]
pub struct LowerTypeNameStep<'a> {
    ctx: &'a BuildContext,
}

impl<'a> CodeTransformStep for LowerTypeNameStep<'a> {
    fn transform(&mut self, mod_idx: usize, cursor: &mut b::BlockCursor) {
        let (value_idx, _loc) = {
            let modules = &self.ctx.lock_modules();
            let instr = cursor.instr(&modules[mod_idx]);
            let b::InstrBody::TypeName(v) = &instr.body else {
                return;
            };
            (*v, instr.loc)
        };

        let type_name = {
            let modules = &self.ctx.lock_modules();
            let ty = &modules[mod_idx].values[value_idx].ty;
            Self::type_name_str(ty, &modules)
        };

        let modules = &mut self.ctx.lock_modules_mut();
        let instr = cursor.instr_mut(&mut modules[mod_idx]);
        instr.body = b::InstrBody::CreateString(type_name);
    }
}

impl<'a> LowerTypeNameStep<'a> {
    fn type_name_str(ty: &b::Type, modules: &[b::Module]) -> String {
        match &ty.body {
            b::TypeBody::Bool => "bool".to_string(),
            b::TypeBody::I8 => "i8".to_string(),
            b::TypeBody::I16 => "i16".to_string(),
            b::TypeBody::I32 => "i32".to_string(),
            b::TypeBody::I64 => "i64".to_string(),
            b::TypeBody::U8 => "u8".to_string(),
            b::TypeBody::U16 => "u16".to_string(),
            b::TypeBody::U32 => "u32".to_string(),
            b::TypeBody::U64 => "u64".to_string(),
            b::TypeBody::USize => "usize".to_string(),
            b::TypeBody::F32 => "f32".to_string(),
            b::TypeBody::F64 => "f64".to_string(),
            b::TypeBody::String => "str".to_string(),
            b::TypeBody::Void => "void".to_string(),
            b::TypeBody::Never => "never".to_string(),
            b::TypeBody::Array(inner) => {
                format!("[{}]", Self::type_name_str(inner, modules))
            }
            b::TypeBody::Ptr(inner) => {
                if let Some(inner) = inner {
                    format!("*{}", Self::type_name_str(inner, modules))
                } else {
                    "*anyopaque".to_string()
                }
            }
            b::TypeBody::Func(func_ty) => {
                let params: Vec<_> = func_ty
                    .params
                    .iter()
                    .map(|p| Self::type_name_str(p, modules))
                    .collect();
                format!(
                    "({}) -> {}",
                    params.join(", "),
                    Self::type_name_str(&func_ty.ret, modules)
                )
            }
            b::TypeBody::TypeRef(type_ref) => {
                if let Some(typedef) =
                    modules[type_ref.mod_idx].typedefs.get(type_ref.idx)
                {
                    typedef.name.last_ident().to_string()
                } else {
                    format!("type{}-{}", type_ref.mod_idx, type_ref.idx)
                }
            }
            b::TypeBody::TypeVar(tv) => {
                if let Some(typevar) = modules[tv.mod_idx].typevars.get(tv.typevar_idx) {
                    typevar.name.last_ident().to_string()
                } else {
                    format!("typevar{}-{}", tv.mod_idx, tv.typevar_idx)
                }
            }
            _ => "unknown".to_string(),
        }
    }
}
