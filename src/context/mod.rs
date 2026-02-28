mod runtime;

use std::fs;
use std::ops::{Deref, DerefMut};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};

use derive_new::new;
use tree_sitter as ts;

use self::runtime::RuntimeBuilder;
use crate::{bytecode as b, codegen, config, errors, parser, sources, typecheck};

#[derive(Debug, new)]
pub struct BuildContext {
    pub cfg: config::BuildConfig,
    #[new(default)]
    pub source_manager: sources::SourceManager,
    #[new(default)]
    pub errors: Mutex<Vec<errors::Error>>,
    #[new(default)]
    modules: RwLock<Vec<b::Module>>,
    #[new(default)]
    pub main: RwLock<Option<(usize, usize)>>,
    #[new(default)]
    pub core_mod_idx: Option<usize>,
}
impl BuildContext {
    pub fn lock_modules(&self) -> impl Deref<Target = Vec<b::Module>> + '_ {
        self.modules.read().unwrap()
    }

    pub fn lock_modules_mut(&self) -> impl DerefMut<Target = Vec<b::Module>> + '_ {
        self.modules.write().unwrap()
    }

    pub fn push_error(&self, value: errors::Error) {
        self.errors.lock().unwrap().push(value);
    }

    pub fn has_errors(&self) -> bool {
        self.errors.lock().unwrap().len() > 0
    }

    pub fn into_compile_error(self) -> errors::CompilerError {
        let source_manager = Arc::new(self.source_manager);
        let errors = self.errors.into_inner().unwrap();
        errors::CompilerError::new(source_manager, errors)
    }

    pub fn parse(&self, src_idx: usize, is_entry: bool) -> usize {
        let mut ts_parser = ts::Parser::new();
        ts_parser
            .set_language(&tree_sitter_nasin::LANGUAGE.into())
            .unwrap();
        let tree = ts_parser
            .parse(&self.source_manager.source(src_idx).content().text, None)
            .expect("Could not parse this file");
        let root_node = tree.root_node();

        if self.cfg.dump_ast {
            println!("{}", root_node.to_sexp());
        }

        let mod_idx = {
            let mut modules = self.lock_modules_mut();
            let sources = self
                .source_manager
                .sources
                .iter()
                .map(|s| s.into())
                .collect();
            let mod_idx = modules.len();
            modules.push(b::Module::new(mod_idx, sources));
            mod_idx
        };

        let mut module_parser =
            parser::ModuleParser::new(self, src_idx, mod_idx, is_entry);
        if let Some(core_mod_idx) = self.core_mod_idx {
            module_parser.open_module(core_mod_idx);
        }
        module_parser.add_root(root_node);
        module_parser.finish();

        typecheck::TypeChecker::new(self, mod_idx).check();

        if self.cfg.dump_bytecode {
            println!("{}", &self.lock_modules()[mod_idx]);
        }

        mod_idx
    }

    pub fn parse_library(&mut self) -> bool {
        let lib_dir = PathBuf::from(
            option_env!("LIB_DIR").expect("env LIB_DIR should be provided"),
        );

        let Ok(core_src_idx) = self.preload(lib_dir.join("core.nsn")) else {
            return false;
        };

        self.core_mod_idx = Some(self.parse(core_src_idx, false));
        true
    }

    pub fn compile(&self) {
        let rt_entry = RuntimeBuilder::new(self).add_entry().build();
        // RuntimeBuilder can push errors
        if self.has_errors() {
            return;
        }

        let modules = self.lock_modules();

        if self.cfg.dump_bytecode {
            if let Some((mod_idx, _)) = rt_entry {
                println!("{}", &modules[mod_idx]);
            }
        }

        let codegen = codegen::BinaryCodegen::new(&modules, rt_entry, &self.cfg);

        fs::create_dir_all(self.cfg.out.parent().unwrap()).unwrap();
        codegen.write();

        if !self.cfg.silent && !self.cfg.run {
            println!("Compiled program to {}", self.cfg.out.to_string_lossy());
        }
    }

    pub fn open(&mut self, path: PathBuf) -> Result<usize, ()> {
        match self.source_manager.open(path) {
            Ok(idx) => Ok(idx),
            Err(err) => {
                self.push_error(err);
                Err(())
            }
        }
    }

    pub fn preload(&mut self, path: PathBuf) -> Result<usize, ()> {
        match self.source_manager.preload(path) {
            Ok(idx) => Ok(idx),
            Err(err) => {
                self.push_error(err);
                Err(())
            }
        }
    }
}
