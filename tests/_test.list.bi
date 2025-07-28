:i count 13
:b shell 69
./bin/nasin b tests/hello.nsn -o tests/out/hello && ./tests/out/hello
:i returncode 0
:b stdout 42
Compiled program to tests/out/hello
Hello

:b stderr 0

:b shell 81
./bin/nasin b tests/operators.nsn -o tests/out/operators && ./tests/out/operators
:i returncode 70
:b stdout 40
Compiled program to tests/out/operators

:b stderr 29
sanity check, this will fail

:b shell 102
./bin/nasin b tests/func_declaration.nsn -o tests/out/func_declaration && ./tests/out/func_declaration
:i returncode 0
:b stdout 53
Compiled program to tests/out/func_declaration
Hello

:b stderr 0

:b shell 93
./bin/nasin b tests/global_string.nsn -o tests/out/global_string && ./tests/out/global_string
:i returncode 0
:b stdout 62
Compiled program to tests/out/global_string
Hello from global

:b stderr 0

:b shell 123
./bin/nasin b tests/global_string_from_func.nsn -o tests/out/global_string_from_func && ./tests/out/global_string_from_func
:i returncode 0
:b stdout 72
Compiled program to tests/out/global_string_from_func
Hello from global

:b stderr 0

:b shell 60
./bin/nasin b tests/if.nsn -o tests/out/if && ./tests/out/if
:i returncode 0
:b stdout 54
Compiled program to tests/out/if
it's true
it's false

:b stderr 0

:b shell 90
./bin/nasin b tests/if_returning.nsn -o tests/out/if_returning && ./tests/out/if_returning
:i returncode 0
:b stdout 64
Compiled program to tests/out/if_returning
it's true
it's false

:b stderr 0

:b shell 81
./bin/nasin b tests/recursion.nsn -o tests/out/recursion && ./tests/out/recursion
:i returncode 0
:b stdout 47
Compiled program to tests/out/recursion
got 10

:b stderr 0

:b shell 87
./bin/nasin b tests/record_type.nsn -o tests/out/record_type && ./tests/out/record_type
:i returncode 0
:b stdout 60
Compiled program to tests/out/record_type
Hello from record

:b stderr 0

:b shell 93
./bin/nasin b tests/return_record.nsn -o tests/out/return_record && ./tests/out/return_record
:i returncode 0
:b stdout 62
Compiled program to tests/out/return_record
Hello from record

:b stderr 0

:b shell 72
./bin/nasin b tests/method.nsn -o tests/out/method && ./tests/out/method
:i returncode 0
:b stdout 65
Compiled program to tests/out/method
Hello from record
Hi method

:b stderr 0

:b shell 87
./bin/nasin b tests/func_as_arg.nsn -o tests/out/func_as_arg && ./tests/out/func_as_arg
:i returncode 101
:b stdout 0

:b stderr 209

thread 'main' panicked at src/main.rs:79:45:
file not found: Os { code: 2, kind: NotFound, message: "No such file or directory" }
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace

:b shell 81
./bin/nasin b tests/interface.nsn -o tests/out/interface && ./tests/out/interface
:i returncode 0
:b stdout 148
Compiled program to tests/out/interface
PrintA
implementation omitted
PrintB
PrintB 1
implementation omitted
PrintB
PrintB 2
implementation omitted

:b stderr 0

