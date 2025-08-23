use cranelift_shim as cl;
use itertools::Itertools;

pub fn dump_func(name: &str, func: &cl::Function, cl_module: &impl cl::Module) {
    // Dump the table of symbols so we can understand which global value is which
    let user_refs = func
        .global_values
        .values()
        .filter_map(|gv_data| match gv_data {
            cl::GlobalValueData::Symbol {
                name: cl::ExternalName::User(user_ref),
                ..
            } => Some(user_ref),
            _ => None,
        })
        .unique();
    for user_ref in user_refs {
        let Some(&cl::UserExternalName { namespace, index }) =
            func.params.user_named_funcs().get(*user_ref)
        else {
            continue;
        };

        print!("{user_ref} = ");

        let target = cl::ModuleRelocTarget::user(namespace, index);
        if cl::ModuleDeclarations::is_function(&target) {
            let func_id = cl::FuncId::from_name(&target);
            let func_name = cl_module
                .declarations()
                .get_function_decl(func_id)
                .linkage_name(func_id);
            print!("{func_name}");
        } else {
            let data_id = cl::DataId::from_name(&target);
            print!("{}", get_data_name(data_id));
        }

        println!();
    }
    println!("<{name}> {func}");
}

pub fn dump_signature(name: &str, sig: &cl::Signature) {
    println!("<{name}> {sig}");
}

pub fn dump_data(
    data_id: &cranelift_shim::DataId,
    desc: &cranelift_shim::DataDescription,
    cl_module: &impl cl::Module,
) {
    let data_init = &desc.init;
    let start = format!("{} [{}]", get_data_name(*data_id), data_init.size());

    let contents = match data_init {
        cl::Init::Bytes { contents } if contents.len() > 0 => {
            println!("{start} =");
            contents
        }
        _ => {
            print!("{start}");
            return;
        }
    };

    let relocs = desc
        .data_relocs
        .iter()
        .map(|&(pos, gv, _)| (pos, Some(format!("data {}", desc.data_decls[gv]))))
        .chain(desc.function_relocs.iter().map(|&(pos, gv)| {
            (pos, Some(format!("function {}", desc.function_decls[gv])))
        }))
        .chain([(data_init.size() as u32, None)])
        .sorted_by_key(|(pos, _)| *pos);

    let mut i = 0;
    for (pos, reloc) in relocs {
        let content_part = &contents[i..pos as usize];

        for chunk in content_part.chunks(8) {
            print!("   ");
            for byte in chunk {
                print!(" {byte:02X}");
            }
            for _ in 0..(8 - chunk.len()) {
                print!("   ");
            }

            print!("  ; ");
            for byte in chunk {
                if byte.is_ascii_graphic() {
                    print!("{}", *byte as char);
                } else {
                    print!(".");
                }
            }
            println!();
        }

        i = pos as usize;
        if let Some(name) = reloc {
            let len = cl_module.isa().pointer_bytes() as usize;
            i += len;
            println!("   {}  ; <{}>", " 00".repeat(len), name);
        }
    }
}

pub fn get_data_name(data_id: cl::DataId) -> String {
    format!("data {}", &data_id.to_string()[6..])
}
