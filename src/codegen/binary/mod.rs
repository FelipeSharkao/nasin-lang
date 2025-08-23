mod context;
mod dump;
mod func;
mod types;

use std::borrow::Cow;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufWriter;
use std::mem;

use cranelift_shim::settings::Configurable;
use cranelift_shim::{self as cl, InstBuilder, Module};
use itertools::Itertools;
use target_lexicon::Triple;

use self::context::{CodegenContext, FuncBinding};
use self::func::{Callee, FuncCodegen};
use self::types::{ResultPolicy, ReturnPolicy};
use crate::{bytecode as b, cmd, config, utils};

utils::number_enum!(pub FuncNS: u32 {
    User = 0,
    SystemFunc = 1,
    Helper = 2,
});

utils::number_enum!(pub SystemFunc: u32 {
    Start = 0,
    Exit = 1,
});

pub struct BinaryCodegen<'a> {
    ctx: CodegenContext<'a>,
    declared_funcs: HashMap<(usize, usize), cl::Function>,
    entry_func: Option<(cl::FuncId, cl::Function)>,
    module_ctx: cl::Context,
    rt_start: Option<(usize, usize)>,
    next_func_id: u32,
}
impl<'a> BinaryCodegen<'a> {
    pub fn new(
        modules: &'a [b::Module],
        rt_start: Option<(usize, usize)>,
        cfg: &'a config::BuildConfig,
    ) -> Self {
        let triple = Triple::host();

        let mut settings_builder = cl::settings::builder();
        settings_builder.set("opt_level", "speed").unwrap();
        if !cfg!(debug_assertions) {
            settings_builder.set("enable_verifier", "false").unwrap();
        }
        let flags = cl::settings::Flags::new(settings_builder);
        if cfg.dump_clif {
            println!("{flags}");
        }

        let isa_target = cl::isa::lookup(triple).unwrap().finish(flags).unwrap();

        let cl_module = cl::ObjectModule::new(
            cl::ObjectBuilder::new(isa_target, "main", cl::default_libcall_names())
                .unwrap(),
        );

        let module_ctx = cl_module.make_context();

        BinaryCodegen {
            ctx: CodegenContext::new(modules, cfg, cl_module),
            module_ctx,
            declared_funcs: HashMap::new(),
            entry_func: None,
            rt_start,
            next_func_id: 0,
        }
    }
}
impl BinaryCodegen<'_> {
    pub fn write(mut self) {
        for mod_idx in 0..self.ctx.modules.len() {
            for idx in 0..self.ctx.modules[mod_idx].globals.len() {
                self.insert_global(mod_idx, idx);
            }

            for idx in 0..self.ctx.modules[mod_idx].funcs.len() {
                self.insert_function(mod_idx, idx);
            }

            for idx in 0..self.ctx.modules[mod_idx].typedefs.len() {
                self.insert_type(mod_idx, idx);
            }
        }

        self.build_entry();

        for mod_idx in 0..self.ctx.modules.len() {
            for idx in 0..self.ctx.modules[mod_idx].funcs.len() {
                let func = &self.ctx.modules[mod_idx].funcs[idx];
                if func.extrn.is_some() || func.is_virt {
                    continue;
                }
                self.build_function(mod_idx, idx);
            }
        }

        self.build_func_closures();

        if self.ctx.cfg.dump_clif {
            self.dump_clif();
        }

        self.emit_functions();
        self.write_to_file();
    }

    fn insert_global(&mut self, mod_idx: usize, idx: usize) {
        self.ctx.insert_global(mod_idx, idx);
    }

    fn insert_function(&mut self, mod_idx: usize, idx: usize) {
        let module = &self.ctx.modules[mod_idx];
        let decl = &module.funcs[idx];
        let proto = types::FuncPrototype::from_func(
            mod_idx,
            idx,
            self.ctx.modules,
            &self.ctx.cl_module,
        );

        let user_func_name =
            cl::UserFuncName::user(FuncNS::User.into(), self.next_func_id);
        self.next_func_id += 1;

        let func =
            cl::Function::with_name_signature(user_func_name, proto.signature.clone());

        let symbol_name = if let Some(b::Extern { name }) = &decl.extrn {
            name.clone()
        } else {
            // TODO: improve name mangling
            decl.name.clone()
        };

        let func_id = if !decl.is_virt {
            let linkage = if decl.extrn.is_some() {
                if decl.body.is_empty() {
                    cl::Linkage::Import
                } else {
                    cl::Linkage::Export
                }
            } else {
                cl::Linkage::Local
            };

            let func_id = self
                .ctx
                .cl_module
                .declare_function(&symbol_name, linkage, &func.signature)
                .unwrap();
            Some(func_id)
        } else {
            None
        };

        self.ctx.funcs.insert(
            (mod_idx, idx),
            FuncBinding {
                symbol_name,
                is_extrn: decl.extrn.is_some(),
                is_virt: decl.is_virt,
                func_id,
                proto,
            },
        );
        self.declared_funcs.insert((mod_idx, idx), func);
    }

    fn insert_type(&mut self, mod_idx: usize, idx: usize) {
        self.ctx.insert_type(mod_idx, idx);
    }

    fn build_entry(&mut self) {
        let mut exit_sig = self.ctx.cl_module.make_signature();
        exit_sig.params.push(cl::AbiParam::new(cl::types::I32));
        let exit_func = cl::Function::with_name_signature(
            cl::UserFuncName::user(FuncNS::SystemFunc.into(), SystemFunc::Exit.into()),
            exit_sig,
        );
        let exit_func_id = self
            .ctx
            .cl_module
            .declare_function("exit", cl::Linkage::Import, &exit_func.signature)
            .unwrap();

        let mut func = cl::Function::with_name_signature(
            cl::UserFuncName::user(FuncNS::SystemFunc.into(), SystemFunc::Start.into()),
            self.ctx.cl_module.make_signature(),
        );
        let func_id = self
            .ctx
            .cl_module
            .declare_function("_start", cl::Linkage::Export, &func.signature)
            .unwrap();

        utils::replace_with(self, |mut this| {
            let mut func_ctx = cl::FunctionBuilderContext::new();
            let func_builder = cl::FunctionBuilder::new(&mut func, &mut func_ctx);
            let mut codegen = FuncCodegen::new(this.ctx, Some(func_builder), true);
            codegen.create_initial_block(&[], None, ResultPolicy::Normal, 0);

            for ((i, j), global) in codegen
                .ctx
                .globals
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .sorted_by(|a, b| a.0.cmp(&b.0))
            {
                if global.is_const {
                    continue;
                };

                let types::ValueSource::Data(data_id) = &global.value.src else {
                    panic!("non const global should be a writable data");
                };

                let gv = codegen.ctx.modules[i].globals[j].value;
                let ty = &codegen.ctx.modules[i].values[gv].ty;

                let start_block = codegen.scopes.last().block;

                codegen.scopes.begin(func::ScopePayload {
                    start_block,
                    block: start_block,
                    next_branches: vec![],
                    result: Some(gv),
                    ty: Some(Cow::Borrowed(ty)),
                });

                codegen
                    .values
                    .insert((i, gv), types::RuntimeValue::new((*data_id).into(), i, gv));

                codegen.add_body(
                    &codegen.ctx.modules[i].globals[j].body,
                    i,
                    ResultPolicy::Global,
                );

                codegen.scopes.end();
                codegen.values.clear();

                let builder = codegen.builder.as_mut().unwrap();

                let next_block = builder.create_block();
                builder.ins().jump(next_block, &[]);
                builder.switch_to_block(next_block);

                codegen.scopes.last_mut().block = next_block;
            }

            if let Some((mod_idx, func_idx)) = this.rt_start {
                codegen.call_func(mod_idx, func_idx, &[]);
            }

            let exit_code = codegen
                .builder
                .as_mut()
                .unwrap()
                .ins()
                .iconst(cl::types::I32, 0);
            codegen.call(
                Callee::Direct(exit_func_id),
                &[exit_code],
                ReturnPolicy::NoReturn,
            );

            this.ctx = codegen.finish();
            this
        });

        self.entry_func = Some((func_id, func));
    }

    fn build_function(&mut self, mod_idx: usize, idx: usize) {
        let decl = &self.ctx.modules[mod_idx].funcs[idx];
        let ret_policy = self
            .ctx
            .funcs
            .get(&(mod_idx, idx))
            .unwrap()
            .proto
            .ret_policy;
        utils::replace_with(self, |mut this| {
            let mut func_ctx = cl::FunctionBuilderContext::new();
            let func = this.declared_funcs.get_mut(&(mod_idx, idx)).unwrap();
            let func_builder = cl::FunctionBuilder::new(func, &mut func_ctx);
            let mut codegen = FuncCodegen::new(this.ctx, Some(func_builder), false);
            codegen.create_initial_block(
                &decl.params,
                Some(decl.ret),
                ResultPolicy::Return(ret_policy),
                mod_idx,
            );

            codegen.add_body(&decl.body, mod_idx, ResultPolicy::Return(ret_policy));

            this.ctx = codegen.finish();
            this
        })
    }

    fn build_func_closures(&mut self) {
        for ((mod_idx, func_idx), binding) in &mut self.ctx.funcs_closures {
            let target_func_id = self.ctx.funcs[&(*mod_idx, *func_idx)]
                .func_id
                .expect("Function should be defined");

            let mut func_ctx = cl::FunctionBuilderContext::new();
            let mut func_builder =
                cl::FunctionBuilder::new(&mut binding.func, &mut func_ctx);

            let block = func_builder.create_block();
            func_builder.append_block_params_for_function_params(block);
            func_builder.switch_to_block(block);

            let params = func_builder.block_params(block)[1..].to_vec();

            let target_func_ref = self
                .ctx
                .cl_module
                .declare_func_in_func(target_func_id, func_builder.func);
            let call_ins = func_builder.ins().call(target_func_ref, &params);

            let ret = func_builder.inst_results(call_ins).to_vec();
            func_builder.ins().return_(&ret);
        }
    }

    fn dump_clif(&mut self) {
        println!();

        let mut funcs = vec![];

        for (key, binding) in self.ctx.funcs.iter().sorted_by_key(|x| x.0) {
            if let Some(func) = self.declared_funcs.get(key) {
                funcs.push((binding.symbol_name.as_str(), func));
            } else {
                dump::dump_signature(&binding.symbol_name, &binding.proto.signature);
            }
        }

        for (_, binding) in self.ctx.funcs_closures.iter().sorted_by_key(|x| x.0) {
            funcs.push((&binding.symbol_name, &binding.func));
        }

        if let Some((_, func)) = &self.entry_func {
            funcs.push(("_start", func));
        }

        for (name, func) in funcs {
            dump::dump_func(name, func, &self.ctx.cl_module);
        }

        if self.ctx.data.len() > 0 {
            for (data_id, desc) in self.ctx.data.iter().sorted_by_key(|x| x.0) {
                dump::dump_data(data_id, desc, &self.ctx.cl_module);
            }

            println!();
        }
    }

    fn emit_functions(&mut self) {
        let funcs = mem::replace(&mut self.ctx.funcs, HashMap::new());
        for ((mod_idx, func_idx), binding) in funcs {
            if binding.is_extrn || binding.is_virt {
                continue;
            }

            let func = self.declared_funcs.remove(&(mod_idx, func_idx));
            let (Some(func), Some(func_id)) = (func, binding.func_id) else {
                continue;
            };

            self.emit_function(
                &self.ctx.modules[mod_idx].funcs[func_idx].name,
                func_id,
                func,
            );
        }

        let funcs_closures = mem::replace(&mut self.ctx.funcs_closures, HashMap::new());
        for ((mod_idx, func_idx), binding) in funcs_closures {
            self.emit_function(
                &self.ctx.modules[mod_idx].funcs[func_idx].name,
                binding.func_id,
                binding.func,
            );
        }

        if let Some((func_id, func)) = mem::replace(&mut self.entry_func, None) {
            self.emit_function("_start", func_id, func);
        }
    }

    fn emit_function(&mut self, name: &str, func_id: cl::FuncId, func: cl::Function) {
        self.module_ctx.func = func;
        match self
            .ctx
            .cl_module
            .define_function(func_id, &mut self.module_ctx)
        {
            Ok(()) => {}
            Err(err) => {
                panic!("Failed to emit function {name}: {err:?}",);
            }
        }
        self.ctx.cl_module.clear_context(&mut self.module_ctx)
    }

    fn write_to_file(self) {
        let obj_product = self.ctx.cl_module.finish();

        let obj_path = format!("{}.o", self.ctx.cfg.out.to_string_lossy());
        let out_file = File::create(&obj_path).expect("Failed to create object file");

        obj_product
            .object
            .write_stream(BufWriter::new(out_file))
            .unwrap();

        // TODO: windows support
        let status = cmd!(
            "ld",
            "-dynamic-linker",
            "/lib/ld-linux-x86-64.so.2",
            "-o",
            &self.ctx.cfg.out,
            "-lc",
            &obj_path
        )
        .status()
        .expect("failed to link object file");

        if !status.success() {
            panic!("failed to link object file");
        }
    }
}
