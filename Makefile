CXX      ?= g++
CXXFLAGS ?= -O3 -march=native -std=c++17 -Wall -Wextra -pthread
NVCC     ?= nvcc
# Ship PTX only and let the NVIDIA driver JIT-compile it for the installed GPU.
# Do not switch to -arch=sm_XX/native: with CUDA 12.0, ptxas' sm_86 SASS for the
# kangaroo kernel hangs on an RTX 3060, while the driver's JIT of the same PTX
# is correct and just as fast. compute_52 PTX runs on every GPU CUDA 12 supports.
NVFLAGS  ?= -O3 -std=c++17 -gencode arch=compute_52,code=compute_52

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
