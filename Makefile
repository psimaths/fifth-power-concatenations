NVCC ?= nvcc
CXX ?= g++
CUDA_ARCH ?= native
NVCCFLAGS ?= -O3 -std=c++17 -Xcompiler=-fopenmp -arch=$(CUDA_ARCH)
PYTHON ?= python3

.PHONY: all test test-gpu clean
all: build/search

build/search: src/search.cu src/core.hpp
	mkdir -p build
	$(NVCC) $(NVCCFLAGS) src/search.cu -o $@

build/test_core: tests/test_core.cpp src/core.hpp
	mkdir -p build
	$(CXX) -O2 -std=c++17 tests/test_core.cpp -o $@

test: build/test_core
	./build/test_core
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py'

test-gpu: build/search
	./build/search --tiny > build/tiny.jsonl
	$(PYTHON) tests/check_gpu.py build/tiny.jsonl

clean:
	rm -rf build
