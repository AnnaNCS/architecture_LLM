# architecture_LLM

To run the code, use the following command:
gcc softmax.c -o softmax -lm
gcc softmax_opt.c -o softmax_opt -lm -fopenmp
gcc -Ofast softmax_opt.c -o softmax_opt -lm -fopenmp

./softmax 
./softmax_opt