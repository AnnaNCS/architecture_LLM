#include <stdint.h> 
#include <immintrin.h>
#include <stdio.h>
#include <time.h>


int main() {

	uint32_t clock = 4294967295; // max int it can be
	uint64_t clockL = 18446744073709551615; // max int it can be

	printf("%u ticks.\n" ,clock);
	printf("%lu ticks.\n" ,clock);
	printf("%llu ticks.\n" ,clock);

	printf("%u ticks.\n" ,clockL);
	printf("%lu ticks.\n" ,clockL);
	printf("%llu ticks.\n" ,clockL);

}