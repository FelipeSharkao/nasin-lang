use gimli;

pub fn struct_sections(
) -> gimli::write::Sections<gimli::write::EndianVec<gimli::LittleEndian>> {
    // FIXME: read that from the compilation

    let encoding = gimli::Encoding {
        format: gimli::Format::Dwarf32,
        version: 5,
        address_size: 8,
    };
    let mut unit = gimli::write::DwarfUnit::new(encoding);

    let ulong = add_number_type(&mut unit, "unsigned long", 8, false);
    let char = add_number_type(&mut unit, "char", 1, false);
    let char_ptr = add_pointer_type(&mut unit, char);

    // str
    let str = add_struct_type(&mut unit, "str", 16);
    {
        add_member(&mut unit, str, "data", 0, char_ptr);
        add_member(&mut unit, str, "len", 8, ulong);
    }

    let str_ptr = add_pointer_type(&mut unit, str);

    // [str]
    let str_array = add_struct_type(&mut unit, "array<str>", 16);
    {
        add_member(&mut unit, str_array, "data", 0, str_ptr);
        add_member(&mut unit, str_array, "len", 8, ulong);
    }

    // write sections
    let mut sections =
        gimli::write::Sections::new(gimli::write::EndianVec::new(gimli::LittleEndian));
    unit.write(&mut sections)
        .expect("should be able to write debug info");

    sections
}

fn add_number_type(
    unit: &mut gimli::write::DwarfUnit,
    name: impl Into<Vec<u8>>,
    size: u64,
    signed: bool,
) -> gimli::write::UnitEntryId {
    let die = unit.unit.add(unit.unit.root(), gimli::DW_TAG_base_type);
    unit.unit.get_mut(die).set(
        gimli::DW_AT_name,
        gimli::write::AttributeValue::String(name.into()),
    );
    unit.unit.get_mut(die).set(
        gimli::DW_AT_byte_size,
        gimli::write::AttributeValue::Udata(size),
    );
    let encoding = if signed {
        gimli::DW_ATE_signed
    } else {
        gimli::DW_ATE_unsigned
    };
    unit.unit.get_mut(die).set(
        gimli::DW_AT_encoding,
        gimli::write::AttributeValue::Encoding(encoding),
    );
    die
}

fn add_pointer_type(
    unit: &mut gimli::write::DwarfUnit,
    ty: gimli::write::UnitEntryId,
) -> gimli::write::UnitEntryId {
    let char_ptr = unit.unit.add(unit.unit.root(), gimli::DW_TAG_pointer_type);
    unit.unit.get_mut(char_ptr).set(
        gimli::DW_AT_byte_size,
        gimli::write::AttributeValue::Udata(8),
    );
    unit.unit
        .get_mut(char_ptr)
        .set(gimli::DW_AT_type, gimli::write::AttributeValue::UnitRef(ty));
    char_ptr
}

fn add_struct_type(
    unit: &mut gimli::write::DwarfUnit,
    name: impl Into<Vec<u8>>,
    size: u64,
) -> gimli::write::UnitEntryId {
    let die = unit
        .unit
        .add(unit.unit.root(), gimli::DW_TAG_structure_type);
    unit.unit.get_mut(die).set(
        gimli::DW_AT_name,
        gimli::write::AttributeValue::String(name.into()),
    );
    unit.unit.get_mut(die).set(
        gimli::DW_AT_byte_size,
        gimli::write::AttributeValue::Udata(size),
    );
    die
}

fn add_member(
    unit: &mut gimli::write::DwarfUnit,
    struct_: gimli::write::UnitEntryId,
    name: impl Into<Vec<u8>>,
    offset: u64,
    ty: gimli::write::UnitEntryId,
) -> gimli::write::UnitEntryId {
    let member = unit.unit.add(struct_, gimli::DW_TAG_member);
    unit.unit.get_mut(member).set(
        gimli::DW_AT_name,
        gimli::write::AttributeValue::String(name.into()),
    );
    unit.unit.get_mut(member).set(
        gimli::DW_AT_data_member_location,
        gimli::write::AttributeValue::Udata(offset),
    );
    unit.unit
        .get_mut(member)
        .set(gimli::DW_AT_type, gimli::write::AttributeValue::UnitRef(ty));
    member
}
