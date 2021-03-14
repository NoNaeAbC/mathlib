matrixMul: matrixBenchmark.cpp
	g++ matrixBenchmark.cpp -lpthread -lbenchmark -Ofast -ffast-math -march=native -std=c++20 -Wall -Wextra -o matrix_bench_gcc
	clang++ matrixBenchmark.cpp -lpthread -lbenchmark -Ofast -ffast-math -march=native -std=c++20 -Wall -Wextra -o matrix_bench_clang
	/opt/intel/oneapi/compiler/latest/linux/bin/intel64/icpc matrixBenchmark.cpp -lpthread -lbenchmark -Ofast -ffast-math -march=native -std=c++20 -Wall -Wextra -o matrix_bench_icc -no-multibyte-chars
	./matrix_bench_gcc
	./matrix_bench_clang
	./matrix_bench_icc