pub fn encode_string_lit(s: &str) -> String {
    let mut result = String::with_capacity(s.len() + 2);
    result.push('"');
    for c in s.chars() {
        match c {
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            '\\' => result.push_str("\\\\"),
            '\0' => result.push_str("\\0"),
            '"' => result.push_str("\\\""),
            c => result.push(c),
        }
    }
    result.push('"');
    result
}

pub fn decode_string_lit(lit: &str) -> String {
    let mut result = String::with_capacity(lit.len());
    let mut chars = lit.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            let c = chars.next().unwrap();
            match c {
                'n' => result.push('\n'),
                'r' => result.push('\r'),
                't' => result.push('\t'),
                '\\' => result.push('\\'),
                '0' => result.push('\0'),
                _ => panic!("Unknown escape sequence: \\{c}"),
            }
        } else {
            result.push(c);
        }
    }
    result
}
