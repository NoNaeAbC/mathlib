CXX = clang++

CXX_ARGS = -O3 -ffast-math -flto -march=native

mandelbrot: mandelbrot.cpp
	$(CXX) $(CXX_ARGS) mandelbrot.cpp -o mandelbrot

mandelbrot_win: mandelbrot.cpp
	x86_64-w64-mingw32-g++ -O3 -ffast-math mandelbrot.cpp -o mandelbrot.exe -march=native  -static-libstdc++ -static-libgcc -static

cu_mandelbrot: mandelbrot.cu
	nvcc mandelbrot.cu -std=c++17 -O4 -o mandelbrot

all: mandelbrot

clean:
	rm mandelbrot

run: all
	./mandelbrot