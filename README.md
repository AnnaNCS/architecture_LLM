# Architecture LLM

## About

Computer Architecture course Assignment at Columbia University and New York University, Fall 2023.

## Usage
Part 2:
To run the code, use the following command:
gcc softmax.c -o softmax -lm
gcc softmax_opt.c -o softmax_opt -lm -fopenmp
gcc -Ofast softmax_opt.c -o softmax_opt -lm -fopenmp


./softmax 
./softmax_opt

Part 3: 

To run the code, use the following command:
gcc lookups.c -o lookups
gcc -O3 lookups_opt.c -o lookups_opt -fopenmp

Additional: 
gcc -O3 -fprofile-generate lookups_opt.c -o lookups_opt -fopenmp
./lookups_opt
gcc -O3 -fprofile-use lookups_opt.c -o lookups_opt -fopenmp

./lookups 
./lookups_opt
