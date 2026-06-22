use std::collections::HashSet;

use derive_ctor::ctor;

use super::BuildContext;
use crate::bytecode as b;

#[derive(ctor)]
pub struct BuiltinsBuilder<'a> {
    ctx:      &'a BuildContext,
    #[ctor(expr(b::Name::from_ident("builtins", b::NameIdentKind::Module, None)))]
    name:     b::Name,
    #[ctor(default)]
    typedefs: Vec<b::TypeDef>,
    #[ctor(default)]
    typevars: Vec<b::TypeVarDef>,
}

impl<'a> BuiltinsBuilder<'a> {
    pub fn build(mut self) {
        self.add_typevars();
        self.add_typedefs();

        let mut modules = self.ctx.lock_modules_mut();
        let idx = modules.len();
        assert_eq!(
            idx,
            b::BUILTINS_MODULE_IDX,
            "Builtins should be at index {:?}",
            b::BUILTINS_MODULE_IDX
        );

        let mut module = b::Module::new(idx, self.name, HashSet::new());
        module.typedefs = self.typedefs;
        module.typevars = self.typevars;

        modules.push(module);
    }

    pub fn add_typevars(&mut self) {
        self.typevars.push(b::TypeVarDef::new(
            self.name.with("T", b::NameIdentKind::Type, None),
            None,
            None,
        ));
    }

    pub fn add_typedefs(&mut self) {
        for &builtin in &b::BuiltinType::VALUES {
            let generics = match builtin {
                b::BuiltinType::Array | b::BuiltinType::Ptr => vec![0],
                _ => vec![],
            };

            let typedef = b::TypeDef::new(
                self.name
                    .with(format!("{builtin:?}"), b::NameIdentKind::Type, None),
                b::TypeDefBody::Builtin(builtin),
                // FIXME: change this loc to Option
                b::Loc::default(),
                generics,
            );

            self.typedefs.push(typedef);
        }
    }
}
