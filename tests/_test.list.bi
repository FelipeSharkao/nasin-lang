:i count 15
:b shell 32
./bin/nasin r tests/01_hello.nsn
:i returncode 0
:b stdout 6
Hello

:b stderr 0

:b shell 40
./bin/nasin r tests/02_global_string.nsn
:i returncode 0
:b stdout 18
Hello from global

:b stderr 0

:b shell 38
./bin/nasin r tests/03_hello_array.nsn
:i returncode 0
:b stdout 12
Hello
World

:b stderr 0

:b shell 36
./bin/nasin r tests/04_operators.nsn
:i returncode 70
:b stdout 0

:b stderr 29
sanity check, this will fail

:b shell 39
./bin/nasin r tests/05_array_concat.nsn
:i returncode 0
:b stdout 14
Hello
World
!

:b stderr 0

:b shell 35
./bin/nasin r tests/06_template.nsn
:i returncode 0
:b stdout 14
Hello World 1

:b stderr 0

:b shell 43
./bin/nasin r tests/07_func_declaration.nsn
:i returncode 0
:b stdout 15
Hello from foo

:b stderr 0

:b shell 29
./bin/nasin r tests/08_if.nsn
:i returncode 0
:b stdout 21
it's true
it's false

:b stderr 0

:b shell 39
./bin/nasin r tests/09_if_returning.nsn
:i returncode 0
:b stdout 21
it's true
it's false

:b stderr 0

:b shell 36
./bin/nasin r tests/10_recursion.nsn
:i returncode 0
:b stdout 70
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

:b stderr 0

:b shell 38
./bin/nasin r tests/11_record_type.nsn
:i returncode 0
:b stdout 18
Hello from record

:b stderr 0

:b shell 40
./bin/nasin r tests/12_return_record.nsn
:i returncode 0
:b stdout 18
Hello from record

:b stderr 0

:b shell 33
./bin/nasin r tests/13_method.nsn
:i returncode 0
:b stdout 28
Hello from record
Hi method

:b stderr 0

:b shell 36
./bin/nasin r tests/14_interface.nsn
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

:b shell 40
./bin/nasin r tests/15_func_as_value.nsn
:i returncode 1
:b stdout 42
LinesA
Hello direct
LinesB
Hello indirect

:b stderr 0

