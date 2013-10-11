LDFLAGS   := -lrt -lm -lOpenCL

gpudb: gpu cpu
	 g++ -pg tableScan.o hashJoin.o materialize.o groupby.o  orderBy.o driver.o parseKernel.o $(LDFLAGS) -o GPUDATABASE

cpu: driver.cpp parseKernel.cpp
	g++ -c -g -Wall driver.cpp parseKernel.cpp

gpu: tableScan.cpp hashJoin.cpp  materialize.cpp groupby.cpp orderBy.cpp 
	g++ -c -g -Wall tableScan.cpp hashJoin.cpp  materialize.cpp groupby.cpp orderBy.cpp

all: gpudb

clean:
	rm -rf *.o GPUDATABASE
