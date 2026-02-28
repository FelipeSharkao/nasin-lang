:i count 15
:b shell 29
./bin/nasin r tests/hello.nsn
:i returncode 0
:b stdout 6
Hello

:b stderr 0

:b shell 37
./bin/nasin r tests/global_string.nsn
:i returncode 0
:b stdout 18
Hello from global

:b stderr 0

:b shell 35
./bin/nasin r tests/hello_array.nsn
:i returncode 0
:b stdout 12
Hello
World

:b stderr 0

:b shell 26
./bin/nasin r tests/if.nsn
:i returncode 0
:b stdout 21
it's true
it's false

:b stderr 0

:b shell 36
./bin/nasin r tests/if_returning.nsn
:i returncode 0
:b stdout 21
it's true
it's false

:b stderr 0

:b shell 44
./bin/nasin r tests/aritimetic_operators.nsn
:i returncode 70
:b stdout 0

:b stderr 29
sanity check, this will fail

:b shell 44
./bin/nasin r tests/comparison_operators.nsn
:i returncode 70
:b stdout 0

:b stderr 29
sanity check, this will fail

:b shell 37
./bin/nasin r tests/string_concat.nsn
:i returncode 0
:b stdout 11
HelloWorld

:b stderr 0

:b shell 40
./bin/nasin r tests/func_declaration.nsn
:i returncode 0
:b stdout 15
Hello from foo

:b stderr 0

:b shell 33
./bin/nasin r tests/recursion.nsn
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

:b shell 35
./bin/nasin r tests/record_type.nsn
:i returncode 0
:b stdout 18
Hello from record

:b stderr 0

:b shell 37
./bin/nasin r tests/return_record.nsn
:i returncode 0
:b stdout 18
Hello from record

:b stderr 0

:b shell 30
./bin/nasin r tests/method.nsn
:i returncode 0
:b stdout 28
Hello from record
Hi method

:b stderr 0

:b shell 33
./bin/nasin r tests/interface.nsn
:i returncode 0
:b stdout 108
LinesA
implementation omitted
LinesB
LinesB 1
implementation omitted
LinesB
LinesB 2
implementation omitted

:b stderr 0

:b shell 37
./bin/nasin r tests/func_as_value.nsn
:i returncode 0
:b stdout 42
LinesA
Hello direct
LinesA
Hello indirect

:b stderr 0

