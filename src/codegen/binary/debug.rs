use std::collections::HashMap;
use std::path::Path;

use derive_ctor::ctor;
use gimli::write::{
    Address, AttributeValue, DirectoryId, DwarfUnit, EndianVec, FileId, LineProgram,
    LineString, RelocateWriter, Relocation, Sections, UnitEntryId, Writer,
};
use gimli::{self, LineEncoding, LittleEndian};
use object::write::{Object, SymbolId};
use object::{RelocationEncoding, RelocationKind, SectionKind};

use crate::{bytecode as b, config, sources, utils};

#[derive(Debug, Clone, ctor)]
pub struct DebugFunction {
    pub name: b::Name,
    pub symbol_name: String,
    pub loc: Option<b::Loc>,
}

#[derive(Debug, Clone, ctor)]
pub struct DebugData<'a> {
    pub cfg: &'a config::BuildConfig,
    pub source_manager: &'a sources::SourceManager,
    #[ctor(default)]
    funcs: Vec<DebugFunction>,
    #[ctor(default)]
    reloc_symbols: Vec<SymbolId>,
    #[ctor(default)]
    dir_ids: HashMap<&'a Path, DirectoryId>,
    #[ctor(default)]
    file_ids: HashMap<usize, FileId>,
    #[ctor(default)]
    name_ids: HashMap<UnitEntryId, HashMap<Vec<b::NameNode>, UnitEntryId>>,
}

impl<'a> DebugData<'a> {
    pub fn add_func(&mut self, func: DebugFunction) {
        self.funcs.push(func);
    }

    pub fn write_debug_sections(&mut self, object: &mut Object) {
        let encoding = gimli::Encoding {
            format: gimli::Format::Dwarf32,
            version: 5,
            address_size: 8,
        };

        let base_dir = &self.cfg.base_dir;

        let mut dwarf = DwarfUnit::new(encoding);
        let root = dwarf.unit.root();

        let producer_str = dwarf.strings.add("nasin".as_bytes());
        dwarf.unit.get_mut(root).set(
            gimli::DW_AT_producer,
            AttributeValue::StringRef(producer_str),
        );

        let name_str = dwarf.strings.add(self.cfg.name.as_bytes());
        dwarf
            .unit
            .get_mut(root)
            .set(gimli::DW_AT_name, AttributeValue::StringRef(name_str));

        let comp_dir_str = dwarf.strings.add(base_dir.to_string_lossy().as_bytes());
        dwarf.unit.get_mut(root).set(
            gimli::DW_AT_comp_dir,
            AttributeValue::StringRef(comp_dir_str),
        );

        let mut line_program = LineProgram::new(
            encoding,
            LineEncoding::default(),
            LineString::String(base_dir.to_string_lossy().as_bytes().to_vec()),
            None,
            LineString::String(
                self.source_manager
                    .source(0)
                    .path
                    .strip_prefix(base_dir)
                    .expect("source path should be a child of the base dir")
                    .to_string_lossy()
                    .as_bytes()
                    .to_vec(),
            ),
            None,
        );

        self.dir_ids
            .insert(base_dir, line_program.default_directory());

        for src_idx in 0..self.source_manager.sources.len() {
            self.add_source_file(&mut line_program, src_idx);
        }

        for i in 0..self.funcs.len() {
            let symbol_info = self.get_symbol_info(object, &self.funcs[i].symbol_name);
            let id = self.add_name(&mut dwarf, root, self.funcs[i].name.clone());
            self.add_subprogram(&mut dwarf, &mut line_program, id, i, symbol_info);
        }

        dwarf.unit.line_program = line_program;

        let mut sections = Sections::new(DwarfSection::new());
        dwarf
            .write(&mut sections)
            .expect("should be able to write debug info");

        sections
            .for_each_mut(|id, section| -> Result<(), ()> {
                if section.len() <= 0 {
                    return Ok(());
                }

                let kind = if id.is_string() {
                    SectionKind::OtherString
                } else {
                    SectionKind::Debug
                };
                let object_section = object.add_section(vec![], id.name().into(), kind);
                object.append_section_data(object_section, section.bytes.slice(), 16);

                section.object_section = Some(object_section);
                Ok(())
            })
            .unwrap();

        let sections_res = sections.for_each(|_, section| -> object::write::Result<()> {
            self.apply_relocs(object, &sections, section)
        });
        if let Err(err) = sections_res {
            panic!("should be able to write debug info: {err}");
        }
    }

    fn get_symbol_info(
        &self,
        object: &Object,
        symbol_name: &str,
    ) -> Option<(SymbolId, u64)> {
        object.symbol_id(symbol_name.as_bytes()).map(|symbol_id| {
            let symbol = object.symbol(symbol_id);
            (symbol_id, symbol.size)
        })
    }

    fn add_source_file(&mut self, line_program: &mut LineProgram, src_idx: usize) {
        let path = &self.source_manager.source(src_idx).path;

        let dir = path.parent().expect("source path should have a parent");
        let file = path
            .file_name()
            .expect("source path should have a file name");

        let dir_id = self.dir_ids.entry(dir).or_insert_with(|| {
            let dir_id = line_program.add_directory(LineString::String(
                utils::relative_path_if_child(dir, &self.cfg.base_dir)
                    .to_string_lossy()
                    .as_bytes()
                    .to_vec(),
            ));
            dir_id
        });

        let file_id = line_program.add_file(
            LineString::String(file.to_string_lossy().as_bytes().to_vec()),
            *dir_id,
            None,
        );

        self.file_ids.insert(src_idx, file_id);
    }

    fn add_subprogram(
        &mut self,
        dwarf: &mut DwarfUnit,
        line_program: &mut LineProgram,
        id: UnitEntryId,
        func_idx: usize,
        symbol_info: Option<(SymbolId, u64)>,
    ) {
        let symbol_index = symbol_info.map(|(symbol_id, size)| {
            let symbol_index = self.add_reloc_symbol(symbol_id);

            dwarf.unit.get_mut(id).set(
                gimli::DW_AT_low_pc,
                AttributeValue::Address(Address::Symbol {
                    symbol: symbol_index,
                    addend: 0,
                }),
            );

            dwarf.unit.get_mut(id).set(
                gimli::DW_AT_high_pc,
                AttributeValue::Address(Address::Symbol {
                    symbol: symbol_index,
                    addend: size as i64,
                }),
            );

            symbol_index
        });

        let func = &self.funcs[func_idx];

        if let Some(loc) = &func.loc {
            let file_id = self
                .file_ids
                .get(&loc.source_idx)
                .expect("file ID should exist");

            dwarf.unit.get_mut(id).set(
                gimli::DW_AT_decl_file,
                AttributeValue::FileIndex(Some(*file_id)),
            );

            dwarf.unit.get_mut(id).set(
                gimli::DW_AT_decl_line,
                AttributeValue::Udata(loc.start_line as u64),
            );

            dwarf.unit.get_mut(id).set(
                gimli::DW_AT_decl_column,
                AttributeValue::Udata(loc.start_col as u64),
            );

            line_program.begin_sequence(symbol_index.map(|symbol_index| {
                Address::Symbol {
                    symbol: symbol_index,
                    addend: 0,
                }
            }));
            line_program.row().file = *file_id;
            line_program.row().line = loc.start_line as u64;
            line_program.row().column = loc.start_col as u64;
            line_program.generate_row();
            line_program
                .end_sequence(symbol_info.map(|(_, size)| size as u64).unwrap_or(0));
        }
    }

    fn add_name(
        &mut self,
        dwarf: &mut DwarfUnit,
        parent: UnitEntryId,
        name: b::Name,
    ) -> UnitEntryId {
        let mut id = parent;

        let mut iter = name.nodes.into_iter().peekable();
        while let Some(node) = iter.next() {
            let mut nodes = vec![node];

            while let Some(next) = iter.peek() {
                if let b::NameNode::Ident(_) = next {
                    break;
                }
                nodes.push(iter.next().unwrap());
            }

            let map = self.name_ids.entry(id).or_default();
            id = *map.entry(nodes).or_insert_with_key(|nodes| {
                let node = &nodes[0];

                let b::NameNode::Ident(ident) = node else {
                    panic!("Unexpected node: {node:?}");
                };

                let tag = match ident.kind {
                    b::NameIdentKind::Module => gimli::DW_TAG_module,
                    b::NameIdentKind::Type => gimli::DW_TAG_structure_type,
                    b::NameIdentKind::Func => gimli::DW_TAG_subprogram,
                    b::NameIdentKind::Value => todo!("value name"),
                };

                let id = dwarf.unit.add(id, tag);

                let name_str = dwarf
                    .strings
                    .add(utils::join("", nodes).to_string().into_bytes());
                dwarf
                    .unit
                    .get_mut(id)
                    .set(gimli::DW_AT_name, AttributeValue::StringRef(name_str));

                id
            })
        }

        id
    }

    fn add_reloc_symbol(&mut self, symbol_id: SymbolId) -> usize {
        let symbol_index = self.reloc_symbols.len();
        self.reloc_symbols.push(symbol_id);
        symbol_index
    }

    fn apply_relocs(
        &mut self,
        object: &mut Object,
        sections: &Sections<DwarfSection>,
        section: &DwarfSection,
    ) -> Result<(), object::write::Error> {
        let Some(object_section) = section.object_section else {
            return Ok(());
        };

        for reloc in &section.relocs {
            let (symbol, kind) = match reloc.target {
                gimli::write::RelocationTarget::Section(section_id) => {
                    let Some(reloc_object_section) =
                        sections.get(section_id).and_then(|s| s.object_section)
                    else {
                        panic!("relocation target section not found");
                    };
                    let symbol = object.section_symbol(reloc_object_section);

                    // FIXME: from the example in the gimli repo, kind should be
                    // SectionOffset when compiling Coff
                    (symbol, RelocationKind::Absolute)
                }
                gimli::write::RelocationTarget::Symbol(symbol_idx) => {
                    let Some(symbol) = self.reloc_symbols.get(symbol_idx) else {
                        panic!("relocation symbol not found");
                    };
                    (symbol.clone(), RelocationKind::Absolute)
                }
            };

            object.add_relocation(
                object_section,
                object::write::Relocation {
                    offset: reloc.offset as u64,
                    size: reloc.size * 8,
                    kind,
                    encoding: RelocationEncoding::Generic,
                    symbol,
                    addend: reloc.addend,
                },
            )?;
        }

        Ok(())
    }
}

#[derive(Debug, Clone, ctor)]
struct DwarfSection {
    #[ctor(expr(EndianVec::new(LittleEndian)))]
    bytes: EndianVec<LittleEndian>,
    #[ctor(default)]
    relocs: Vec<Relocation>,
    #[ctor(default)]
    object_section: Option<object::write::SectionId>,
}

impl RelocateWriter for DwarfSection {
    type Writer = EndianVec<LittleEndian>;

    fn writer(&self) -> &Self::Writer {
        &self.bytes
    }

    fn writer_mut(&mut self) -> &mut Self::Writer {
        &mut self.bytes
    }

    fn relocate(&mut self, relocation: Relocation) {
        self.relocs.push(relocation);
    }
}
