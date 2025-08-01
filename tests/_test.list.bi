:i count 13
:b shell 29
./bin/nasin r tests/hello.nsn
:i returncode 0
:b stdout 44
Compiled program to tests/build/hello
Hello

:b stderr 0

:b shell 33
./bin/nasin r tests/operators.nsn
:i returncode 70
:b stdout 42
Compiled program to tests/build/operators

:b stderr 29
sanity check, this will fail

:b shell 40
./bin/nasin r tests/func_declaration.nsn
:i returncode 0
:b stdout 55
Compiled program to tests/build/func_declaration
Hello

:b stderr 0

:b shell 37
./bin/nasin r tests/global_string.nsn
:i returncode 0
:b stdout 64
Compiled program to tests/build/global_string
Hello from global

:b stderr 0

:b shell 47
./bin/nasin r tests/global_string_from_func.nsn
:i returncode 0
:b stdout 74
Compiled program to tests/build/global_string_from_func
Hello from global

:b stderr 0

:b shell 26
./bin/nasin r tests/if.nsn
:i returncode 0
:b stdout 56
Compiled program to tests/build/if
it's true
it's false

:b stderr 0

:b shell 36
./bin/nasin r tests/if_returning.nsn
:i returncode 0
:b stdout 66
Compiled program to tests/build/if_returning
it's true
it's false

:b stderr 0

:b shell 33
./bin/nasin r tests/recursion.nsn
:i returncode 0
:b stdout 49
Compiled program to tests/build/recursion
got 10

:b stderr 0

:b shell 35
./bin/nasin r tests/record_type.nsn
:i returncode 0
:b stdout 62
Compiled program to tests/build/record_type
Hello from record

:b stderr 0

:b shell 37
./bin/nasin r tests/return_record.nsn
:i returncode 0
:b stdout 64
Compiled program to tests/build/return_record
Hello from record

:b stderr 0

:b shell 30
./bin/nasin r tests/method.nsn
:i returncode 0
:b stdout 67
Compiled program to tests/build/method
Hello from record
Hi method

:b stderr 0

:b shell 37
./bin/nasin r tests/func_as_value.nsn
:i returncode 0
:b stdout 88
Compiled program to tests/build/func_as_value
PrintA
Hello direct
PrintA
Hello indirect

:b stderr 0

:b shell 33
./bin/nasin r tests/interface.nsn
:i returncode 0
:b stdout 150
Compiled program to tests/build/interface
PrintA
implementation omitted
PrintB
PrintB 1
implementation omitted
PrintB
PrintB 2
implementation omitted

:b stderr 0

