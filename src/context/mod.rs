mod builtins;

use std::collections::HashSet;
use std::fs;
use std::ops::{Deref, DerefMut};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};

use tree_sitter as ts;

use crate::config::ShouldDump;
use crate::utils::TreeSitterUtils;
use crate::{bytecode as b, codegen, config, errors, parser, sources, typecheck, utils};

#[derive(Debug)]
pub struct BuildContext {
    pub cfg: config::BuildConfig,
    pub source_manager: sources::SourceManager,
    pub errors: Mutex<HashSet<errors::Error>>,
    pub main: RwLock<Option<(usize, usize)>>,
    pub prelude: Vec<usize>,
    modules: RwLock<Vec<b::Module>>,
}

impl BuildContext {
    pub fn new(cfg: config::BuildConfig) -> Self {
        let this = Self {
            cfg,
            source_manager: sources::SourceManager::default(),
            errors: Mutex::new(HashSet::new()),
            main: RwLock::new(None),
            prelude: vec![],
            modules: RwLock::new(vec![]),
        };

        builtins::BuiltinsBuilder::new(&this).build();
        this
    }

    pub fn lock_modules(&self) -> impl Deref<Target = Vec<b::Module>> + '_ {
        utils::DeadlockGuard::new(self.modules.read().unwrap())
    }

    pub fn lock_modules_mut(&self) -> impl DerefMut<Target = Vec<b::Module>> + '_ {
        utils::DeadlockGuard::new(self.modules.write().unwrap())
    }

    pub fn push_error(&self, value: errors::Error) {
        self.errors.lock().unwrap().insert(value);
    }

    pub fn has_errors(&self) -> bool {
        self.errors.lock().unwrap().len() > 0
    }

    pub fn into_compile_error(self) -> errors::CompilerError {
        let source_manager = Arc::new(self.source_manager);
        let errors = self.errors.into_inner().unwrap();
        errors::CompilerError::new(Some(source_manager), errors)
    }

    pub fn parse(&self, src_idx: usize) -> usize {
        let mut ts_parser = ts::Parser::new();
        ts_parser
            .set_language(&tree_sitter_nasin::LANGUAGE.into())
            .unwrap();
        let tree = ts_parser
            .parse(&self.source_manager.source(src_idx).content().text, None)
            .expect("Could not parse this file");
        let root_node = tree.root_node();

        let name =
            b::Name::from_path(&self.source_manager.source(src_idx).path, &self.cfg);

        if self.cfg.dump_ast.should_dump(&name) {
            let source = &self.source_manager.source(src_idx).content().text;
            println!("{}", root_node.display(source));
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
            modules.push(b::Module::new(mod_idx, name, sources));
            mod_idx
        };

        if root_node.has_error() {
            for err in root_node.iter_errors() {
                let source = &self.source_manager.source(src_idx).content().text;
                let token = err.child(0).unwrap_or(err).get_text(source).to_string();
                self.push_error(errors::Error::new(
                    errors::UnexpectedToken::new(token).into(),
                    Some(b::Loc::from_node(src_idx, &err)),
                ));
            }
        }

        let mut module_parser = parser::ModuleParser::new(self, src_idx, mod_idx);

        for &prelude_mod_idx in &self.prelude {
            module_parser.open_module(prelude_mod_idx);
        }

        module_parser.add_root(root_node);
        module_parser.finish();

        {
            let modules = self.lock_modules();

            if self
                .cfg
                .dump_untyped_bytecode
                .should_dump(&modules[mod_idx].name)
            {
                b::Printer::new(&modules, &self.cfg)
                    .with_show_ids(true)
                    .with_source_manager(&self.source_manager)
                    .print_module(mod_idx);
            }
        }

        typecheck::TypeChecker::new(self, mod_idx).check();

        mod_idx
    }

    pub fn parse_library(&mut self) {
        let mut core = None;
        for lib_dir in &self.cfg.lib_dirs {
            let file = lib_dir.join("core.nsn");
            if file.is_file() {
                core = Some(file);
                break;
            }
        }
        let Some(core) = core else {
            self.push_error(errors::Error::new(
                errors::ErrorDetail::MissingLib(errors::MissingLib::new(
                    "core.nsn".to_string(),
                )),
                None,
            ));
            return;
        };

        let Ok(core_src_idx) = self.open(core) else {
            return;
        };

        self.prelude.push(self.parse(core_src_idx));
    }

    pub fn parse_runtime(&mut self) {
        let Some((main_mod_idx, _)) = *self.main.read().unwrap() else {
            self.push_error(errors::Error::new(errors::ErrorDetail::MissingMain, None));
            return;
        };

        self.prelude.push(main_mod_idx);

        let mut runtime = None;
        for lib_dir in &self.cfg.lib_dirs {
            let file = lib_dir.join("runtime.nsn");
            if file.is_file() {
                runtime = Some(file);
                break;
            }
        }

        let Some(core) = runtime else {
            self.push_error(errors::Error::new(
                errors::ErrorDetail::MissingLib(errors::MissingLib::new(
                    "runtime.nsn".to_string(),
                )),
                None,
            ));
            return;
        };

        let Ok(src_idx) = self.open(core) else {
            return;
        };
        self.parse(src_idx);
    }

    pub fn compile(&mut self) {
        let modules = self.lock_modules();

        let codegen =
            codegen::BinaryCodegen::new(&modules, &self.cfg, &self.source_manager);

        fs::create_dir_all(self.cfg.out.parent().unwrap()).unwrap();
        if let Err(error) = codegen.write() {
            self.push_error(error);
        }

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
}
