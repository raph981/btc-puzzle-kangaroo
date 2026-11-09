CXX      ?= g++
CXXFLAGS ?= -O3 -march=native -std=c++17 -Wall -Wextra -pthread
NVCC     ?= nvcc
NVFLAGS  ?= -O3 -std=c++17

SRC = src/main.cpp
HDR = src/kernel.h src/hash.h src/gpu.h

# CPU-only build (default)
kangaroo: $(SRC) $(HDR)
	$(CXX) $(CXXFLAGS) -o $@ $(SRC)

# CPU + NVIDIA GPU build (needs the CUDA toolkit)
gpu: $(SRC) $(HDR) src/gpu_cuda.cu
	$(NVCC) $(NVFLAGS) -DWITH_CUDA -c src/gpu_cuda.cu -o gpu_cuda.o
	$(CXX) $(CXXFLAGS) -DWITH_CUDA -o kangaroo $(SRC) gpu_cuda.o \
		-L$(dir $(shell which $(NVCC)))../lib64 -lcudart

test: kangaroo
	python3 tests/test_solver.py

clean:
	rm -f kangaroo gpu_cuda.o

.PHONY: gpu test clean
