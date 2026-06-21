:i count 21
:b shell 40
./target/release/nasin r tests/hello.nsn
:i returncode 0
:b stdout 6
Hello

:b stderr 0

:b shell 48
./target/release/nasin r tests/global_string.nsn
:i returncode 0
:b stdout 18
Hello from global

:b stderr 0

:b shell 46
./target/release/nasin r tests/hello_array.nsn
:i returncode 0
:b stdout 12
Hello
World

:b stderr 0

:b shell 37
./target/release/nasin r tests/if.nsn
:i returncode 0
:b stdout 21
it's true
it's false

:b stderr 0

:b shell 47
./target/release/nasin r tests/if_returning.nsn
:i returncode 0
:b stdout 21
it's true
it's false

:b stderr 0

:b shell 55
./target/release/nasin r tests/aritimetic_operators.nsn
:i returncode 70
:b stdout 0

:b stderr 66
sanity check, this will fail
aritimetic_operators: exited with 70

:b shell 55
./target/release/nasin r tests/comparison_operators.nsn
:i returncode 70
:b stdout 0

:b stderr 66
sanity check, this will fail
comparison_operators: exited with 70

:b shell 48
./target/release/nasin r tests/string_concat.nsn
:i returncode 0
:b stdout 11
HelloWorld

:b stderr 0

:b shell 51
./target/release/nasin r tests/func_declaration.nsn
:i returncode 0
:b stdout 15
Hello from foo

:b stderr 0

:b shell 44
./target/release/nasin r tests/recursion.nsn
:i returncode 0
:b stdout 79
rec: 0
rec: 1
rec: 2
rec: 3
rec: 4
rec: 5
rec: 6
rec: 7
rec: 8
rec: 9
rec: 10


:b stderr 0

:b shell 46
./target/release/nasin r tests/record_type.nsn
:i returncode 0
:b stdout 18
Hello from record

:b stderr 0

:b shell 48
./target/release/nasin r tests/return_record.nsn
:i returncode 0
:b stdout 18
Hello from record

:b stderr 0

:b shell 41
./target/release/nasin r tests/method.nsn
:i returncode 0
:b stdout 28
Hello from record
Hi method

:b stderr 0

:b shell 44
./target/release/nasin r tests/interface.nsn
:i returncode 0
:b stdout 110
LinesA
1
implementation omitted
LinesB
LinesB 1
implementation omitted
LinesB
LinesB 2
implementation omitted

:b stderr 0

:b shell 48
./target/release/nasin r tests/func_as_value.nsn
:i returncode 0
:b stdout 42
LinesA
Hello direct
LinesA
Hello indirect

:b stderr 0

:b shell 47
./target/release/nasin r tests/generic_func.nsn
:i returncode 0
:b stdout 15
u8
u8
str
bool

:b stderr 0

:b shell 49
./target/release/nasin r tests/generic_record.nsn
:i returncode 0
:b stdout 15
u8
u8
str
bool

:b stderr 0

:b shell 49
./target/release/nasin r tests/generic_method.nsn
:i returncode 0
:b stdout 15
u8
u8
str
bool

:b stderr 0

:b shell 51
./target/release/nasin r tests/conditional_impl.nsn
:i returncode 0
:b stdout 18
hello
hello
42
42

:b stderr 0

:b shell 57
./target/release/nasin r tests/conditional_impl_error.nsn
:i returncode 1
:b stdout 0

:b stderr 477
/home/felipe/Projects/nasin/nasin/tests/conditional_impl_error.nsn:22:8 - error: Expected type StringValue, but found Container(u8) instead
22 | main = string_with(int_value) ; ERROR: int_value cannot be used as StringValue
            ^
/home/felipe/Projects/nasin/nasin/tests/conditional_impl_error.nsn:22:8 - error: Expected type StringValue, but found Container(u8) instead
22 | main = string_with(int_value) ; ERROR: int_value cannot be used as StringValue
            ^


:b shell 49
./target/release/nasin r tests/readme_example.nsn
:i returncode 0
:b stdout 11
right
left

:b stderr 0

