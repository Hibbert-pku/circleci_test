all:
	nvcc main.cu -c -o main.o
	g++ main.o -o main -L/usr/local/cuda/lib64 -lcuda -lcudart
