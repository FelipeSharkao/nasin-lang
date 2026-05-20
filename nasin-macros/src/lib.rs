mod number_enum;

use proc_macro::TokenStream;
use syn::{DeriveInput, parse_macro_input};

#[proc_macro_derive(NumberEnum)]
pub fn derive_number_enum(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    number_enum::derive(input)
        .unwrap_or_else(|err| err.to_compile_error())
        .into()
}
