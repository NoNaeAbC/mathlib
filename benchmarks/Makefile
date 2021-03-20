ARGS = -Ofast -march=native -flto
#ARGS =

matrixMul: matrixBenchmark.cpp
	g++ matrixBenchmark.cpp -lpthread -lbenchmark $(ARGS) -std=c++20 -Wall -Wextra -o matrix_bench_gcc
	clang++ matrixBenchmark.cpp -lpthread -lbenchmark $(ARGS) -std=c++20 -Wall -Wextra -o matrix_bench_clang
	/opt/intel/oneapi/compiler/latest/linux/bin/intel64/icpc matrixBenchmark.cpp -lpthread -lbenchmark $(ARGS) -std=c++20 -Wall -Wextra -o matrix_bench_icc -no-multibyte-chars
	./matrix_bench_gcc
	./matrix_bench_clang
	./matrix_bench_icc


complex: complexBenchmark.cpp
	g++ complexBenchmark.cpp -lpthread -lbenchmark $(ARGS) -std=c++20 -Wall -Wextra -o complex_bench_gcc
	clang++ complexBenchmark.cpp -lpthread -lbenchmark $(ARGS) -std=c++20 -Wall -Wextra -o complex_bench_clang
	/opt/intel/oneapi/compiler/latest/linux/bin/intel64/icpc complexBenchmark.cpp -lpthread -lbenchmark $(ARGS) -std=c++20 -Wall -Wextra -o complex_bench_icc -no-multibyte-chars
	./complex_bench_gcc
	./complex_bench_clang
	./complex_bench_icc