mod context;
mod func;
mod types;

use std::borrow::Cow;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufWriter;

use cranelift_shim::{self as cl, InstBuilder, Module};
use itertools::Itertools;
use target_lexicon::Triple;

use self::context::{CodegenContext, FuncBinding};
use self::func::{CallReturnPolicy, Callee, FuncCodegen, ResultPolicy};
use crate::{bytecode as b, config, utils};

utils::number_enum!(pub FuncNS: u32 {
    User = 0,
    SystemFunc = 1,
});

utils::number_enum!(pub SystemFunc: u32 {
    Start = 0,
    Exit = 1,
});

pub struct BinaryCodegen<'a> {
    ctx: CodegenContext<'a>,
    declared_funcs: HashMap<(usize, usize), cl::Function>,
    module_ctx: cl::Context,
    next_func_id: u32,
}
impl<'a> BinaryCodegen<'a> {
    pub fn new(modules: &'a [b::Module], cfg: &'a config::BuildConfig) -> Self {
        let triple = Triple::host();

        let settings_builder = cl::settings::builder();
        let flags = cl::settings::Flags::new(settings_builder);
        let isa_target = cl::isa::lookup(triple).unwrap().finish(flags).unwrap();

        let obj_module = cl::ObjectModule::new(
            cl::ObjectBuilder::new(isa_target, "main", cl::default_libcall_names())
                .unwrap(),
        );

        let module_ctx = obj_module.make_context();

        BinaryCodegen {
            ctx: CodegenContext::new(modules, cfg, obj_module),
            module_ctx,
            declared_funcs: HashMap::new(),
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

        for mod_idx in 0..self.ctx.modules.len() {
            for idx in 0..self.ctx.modules[mod_idx].funcs.len() {
                let func = &self.ctx.modules[mod_idx].funcs[idx];
                if func.extrn.is_some() || func.is_virt {
                    continue;
                }
                self.build_function(mod_idx, idx);
            }
        }

        self.write_to_file();
    }

    fn build_entry(&mut self) {
        let mut exit_sig = self.ctx.obj_module.make_signature();
        exit_sig.params.push(cl::AbiParam::new(cl::types::I32));
        let exit_func = cl::Function::with_name_signature(
            cl::UserFuncName::user(FuncNS::SystemFunc.into(), SystemFunc::Exit.into()),
            exit_sig,
        );
        let exit_func_id = self
            .ctx
            .obj_module
            .declare_function("exit", cl::Linkage::Import, &exit_func.signature)
            .unwrap();

        let mut func = cl::Function::with_name_signature(
            cl::UserFuncName::user(FuncNS::SystemFunc.into(), SystemFunc::Start.into()),
            self.ctx.obj_module.make_signature(),
        );
        let func_id = self
            .ctx
            .obj_module
            .declare_function("_start", cl::Linkage::Export, &func.signature)
            .unwrap();

        utils::replace_with(self, |mut this| {
            let mut func_ctx = cl::FunctionBuilderContext::new();
            let func_builder = cl::FunctionBuilder::new(&mut func, &mut func_ctx);
            let mut codegen = FuncCodegen::new(this.ctx, Some(func_builder));
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

                let gv = self.ctx.modules[i].globals[j].value;
                let ty = &self.ctx.modules[i].values[gv].ty;

                let start_block = codegen.scopes.last().block;

                codegen.scopes.begin(func::ScopePayload {
                    start_block,
                    block: start_block,
                    next_branches: vec![],
                    result: Some(gv),
                    ty: Some(Cow::Borrowed(ty)),
                });

                codegen.add_body(
                    &self.ctx.modules[i].globals[j].body,
                    i,
                    ResultPolicy::Global,
                );

                codegen.scopes.end();

                if !global.is_entry_point {
                    let v = &self.ctx.modules[i].globals[j].value;
                    let res = codegen.values[v].clone();
                    codegen.store_global(res, &global);
                }
                codegen.values.clear();
            }

            let exit_code = codegen
                .builder
                .as_mut()
                .unwrap()
                .ins()
                .iconst(cl::types::I32, 0);
            codegen.native_call(
                Callee::Direct(exit_func_id),
                &[exit_code],
                CallReturnPolicy::NoReturn,
            );

            this.ctx = codegen.finish();
            this
        });

        if self.ctx.cfg.dump_clif {
            println!("\n<_start> {func}");
        }

        self.module_ctx.func = func;
        self.ctx
            .obj_module
            .define_function(func_id, &mut self.module_ctx)
            .unwrap();
        self.ctx.obj_module.clear_context(&mut self.module_ctx)
    }

    fn insert_global(&mut self, mod_idx: usize, idx: usize) {
        self.ctx.insert_global(mod_idx, idx);
    }

    fn insert_function(&mut self, mod_idx: usize, idx: usize) {
        let module = &self.ctx.modules[mod_idx];
        let decl = &module.funcs[idx];
        let mut sig = self.ctx.obj_module.make_signature();

        let ret_ty = &module.values[decl.ret].ty;
        let result_policy = if ret_ty.is_aggregate(&self.ctx.modules) {
            let ret_param = cl::AbiParam::special(
                self.ctx.obj_module.isa().pointer_type(),
                cl::ArgumentPurpose::StructReturn,
            );
            sig.params.push(ret_param);
            ResultPolicy::StructReturn
        } else if !matches!(&ret_ty.body, b::TypeBody::Void | b::TypeBody::Never) {
            let native_ty =
                types::get_type(ret_ty, self.ctx.modules, &self.ctx.obj_module);
            assert_eq!(native_ty.len(), 1);
            sig.returns.push(cl::AbiParam::new(native_ty[0]));
            ResultPolicy::Return
        } else {
            ResultPolicy::Normal
        };

        for param in &decl.params {
            let ty = &module.values[*param].ty;
            for native_ty in types::get_type(ty, self.ctx.modules, &self.ctx.obj_module) {
                sig.params.push(cl::AbiParam::new(native_ty));
            }
        }

        let user_func_name =
            cl::UserFuncName::user(FuncNS::User.into(), self.next_func_id);
        self.next_func_id += 1;

        let func = cl::Function::with_name_signature(user_func_name, sig);

        let symbol_name = if let Some(b::Extern { name }) = &decl.extrn {
            name.clone()
        } else {
            // TODO: improve name mangling
            format!("$func_{mod_idx}_{idx}")
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
                .obj_module
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
                signature: func.signature.clone(),
                func_id,
                result_policy,
            },
        );
        self.declared_funcs.insert((mod_idx, idx), func);
    }

    fn insert_type(&mut self, mod_idx: usize, idx: usize) {
        self.ctx.insert_type(mod_idx, idx);
    }

    fn build_function(&mut self, mod_idx: usize, idx: usize) {
        let decl = &self.ctx.modules[mod_idx].funcs[idx];
        let result_policy = self.ctx.funcs.get(&(mod_idx, idx)).unwrap().result_policy;
        utils::replace_with(self, |mut this| {
            let mut func_ctx = cl::FunctionBuilderContext::new();
            let func = this.declared_funcs.get_mut(&(mod_idx, idx)).unwrap();
            let func_builder = cl::FunctionBuilder::new(func, &mut func_ctx);
            let mut codegen = FuncCodegen::new(this.ctx, Some(func_builder));
            codegen.create_initial_block(
                &decl.params,
                Some(decl.ret),
                result_policy,
                mod_idx,
            );

            codegen.add_body(&decl.body, mod_idx, result_policy);

            this.ctx = codegen.finish();
            this
        })
    }

    fn write_to_file(mut self) {
        if self.ctx.data.len() > 0 && self.ctx.cfg.dump_clif {
            println!();

            for (data_id, desc) in self.ctx.data.iter().sorted_by(|a, b| a.0.cmp(b.0)) {
                let data_init = &desc.init;
                print!("data {} [{}]", &data_id.to_string()[6..], data_init.size());
                if let cl::Init::Bytes { contents } = data_init {
                    print!(" =");
                    for byte in contents {
                        print!(" {byte:02X}");
                    }
                }

                println!();
            }
        }

        self.build_entry();

        for key in self.ctx.funcs.keys().cloned().sorted().collect_vec() {
            let func = self.declared_funcs.remove(&key);

            let func_binding = self.ctx.funcs.remove(&key).unwrap();

            if self.ctx.cfg.dump_clif {
                match &func {
                    Some(func) => println!("<{}> {func}", &func_binding.symbol_name),
                    None => println!(
                        "<{}> {}",
                        &func_binding.symbol_name, &func_binding.signature
                    ),
                }
            }

            if func_binding.is_extrn || func_binding.is_virt {
                continue;
            }
            let (Some(func), Some(func_id)) = (func, func_binding.func_id) else {
                continue;
            };

            self.module_ctx.func = func;
            self.ctx
                .obj_module
                .define_function(func_id, &mut self.module_ctx)
                .unwrap();
            self.ctx.obj_module.clear_context(&mut self.module_ctx)
        }

        let obj_product = self.ctx.obj_module.finish();

        let obj_path = format!("{}.o", self.ctx.cfg.out.to_string_lossy());
        let out_file = File::create(&obj_path).expect("Failed to create object file");

        obj_product
            .object
            .write_stream(BufWriter::new(out_file))
            .unwrap();

        // TODO: windows support
        let status = std::process::Command::new("ld")
            .arg("-dynamic-linker")
            .arg("/lib/ld-linux-x86-64.so.2")
            .arg("-o")
            .arg(&self.ctx.cfg.out)
            .arg(&obj_path)
            .arg("-lc")
            .status()
            .expect("failed to link object file");

        if !status.success() {
            panic!("failed to link object file");
        }
    }
}
