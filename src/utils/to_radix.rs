pub fn to_radix(mut n: usize, radix: usize) -> String {
    assert!(radix > 1 && radix <= 36, "radix must be between 2 and 36");
    let mut b = vec![];
    loop {
        let (div, rem) = (n / radix, n % radix);
        if rem < 10 {
            b.push(b'0' + rem as u8);
        } else {
            b.push(b'A' + (rem as u8 - 10));
        }
        n = div;
        if n == 0 {
            break;
        }
    }
    b.reverse();
    unsafe { String::from_utf8_unchecked(b) }
}
