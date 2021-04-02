ARGS = -Ofast -march=native -flto
#ARGS =

matrixMul: matrixBenchmark.cpp
	g++ matrixBenchmark.cpp -lpthread -lbenchmark $(ARGS) -std=c++20 -Wall -Wextra -o matrix_bench_gcc
	clang++ matrixBenchmark.cpp -lpthread -lbenchmark $(ARGS) -std=c++20 -Wall -Wextra -o matrix_bench_clang
	/opt/intel/oneapi/compiler/latest/linux/bin/intel64/icpc matrixBenchmark.cpp -lpthread -lbenchmark $(ARGS) -std=c++20 -Wall -Wextra -o matrix_bench_icc -no-multibyte-chars
	/opt/nvidia/hpc_sdk/Linux_x86_64/2021/compilers/bin/pgc++ matrixBenchmark.cpp -tp=native -std=c++20 -O4 -o matrix_bench_pgc -lpthread -lbenchmark -DDEBUG
	./matrix_bench_gcc
	./matrix_bench_clang
	./matrix_bench_icc
	./matrix_bench_pgc


complex: complexBenchmark.cpp
	g++ complexBenchmark.cpp -lpthread -lbenchmark $(ARGS) -std=c++20 -Wall -Wextra -o complex_bench_gcc
	clang++ complexBenchmark.cpp -lpthread -lbenchmark $(ARGS) -std=c++20 -Wall -Wextra -o complex_bench_clang
	/opt/intel/oneapi/compiler/latest/linux/bin/intel64/icpc complexBenchmark.cpp -lpthread -lbenchmark $(ARGS) -std=c++20 -Wall -Wextra -o complex_bench_icc -no-multibyte-chars
	./complex_bench_gcc
	./complex_bench_clang
	./complex_bench_icc