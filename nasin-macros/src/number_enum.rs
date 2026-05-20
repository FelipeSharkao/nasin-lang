use proc_macro2::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput};

pub fn derive(input: DeriveInput) -> syn::Result<TokenStream> {
    let Data::Enum(data) = &input.data else {
        return Err(syn::Error::new_spanned(&input, "Only enums are supported"));
    };

    let repr = input.attrs.iter().find_map(|attr| {
        if attr.path().is_ident("repr") {
            Some(attr.parse_args::<syn::Ident>().ok()?)
        } else {
            None
        }
    });
    let Some(repr) = repr else {
        return Err(syn::Error::new_spanned(
            input,
            "Cannot derive NumberEnum without a repr attribute",
        ));
    };

    let variants = data
        .variants
        .iter()
        .map(|variant| {
            if variant.fields.len() > 0 {
                return Err(syn::Error::new_spanned(
                    variant,
                    "Cannot derive NumberEnum for enum with fields",
                ));
            }
            Ok(variant.ident.clone())
        })
        .collect::<syn::Result<Vec<_>>>()?;

    let count = variants.len();
    if count == 0 {
        return Err(syn::Error::new_spanned(
            input,
            "Cannot derive NumberEnum for an empty enum",
        ));
    }

    let last = variants.len() - 1;

    let vis = &input.vis;
    let name = &input.ident;

    Ok(quote! {
        #[allow(unused)]
        impl #name {
            #vis const COUNT: usize = #count;

            #vis const VALUES: [#name; #count] = {
                let mut values = [#(#name::#variants),*];
                let mut i = 1;
                while i < #count {
                    let key = values[i] as #repr;
                    let mut j = i - 1;
                    while j >= 0 && values[j] as #repr > key {
                        values[j + 1] = values[j];
                        j -= 1;
                    }
                    values[j + 1] = values[i];
                    i += 1;
                }
                values
            };

            #vis const MIN: #name = #name::VALUES[0];
            #vis const MAX: #name = #name::VALUES[#last];
        }

        impl ::std::convert::Into<#repr> for #name {
            #[inline]
            fn into(self) -> #repr {
                self as #repr
            }
        }

        impl ::std::convert::TryFrom<#repr> for #name {
            type Error = ();

            #[inline]
            fn try_from(value: #repr) -> Result<Self, ()> {
                #(if value == #name::#variants as #repr {
                    return Ok(#name::#variants);
                })*
                Err(())
            }
        }

        impl ::std::clone::Clone for #name {
            #[inline]
            fn clone(&self) -> Self {
                *self
            }
        }

        impl ::std::marker::Copy for #name {}

        impl ::std::cmp::PartialEq for #name {
            #[inline]
            fn eq(&self, other: &Self) -> bool {
                *self as #repr == *other as #repr
            }
        }

        impl ::std::cmp::Eq for #name {}

        impl ::std::cmp::PartialOrd for #name {
            #[inline]
            fn partial_cmp(&self, other: &Self) -> Option<::std::cmp::Ordering> {
                (*self as #repr).partial_cmp(&(*other as #repr))
            }
        }

        impl ::std::cmp::Ord for #name {
            #[inline]
            fn cmp(&self, other: &Self) -> ::std::cmp::Ordering {
                (*self as #repr).cmp(&(*other as #repr))
            }
        }
    })
}
