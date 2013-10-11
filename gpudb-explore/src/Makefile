include gmm.mk

# Specify CUDA install path here
CUDAPATH = /usr/local/cuda

# Name of the GMM shared library
LIBGMM = libgmm.so

# Source and object files needed to create $(LIBGMM)
SRCS = client.c common.c core.c interfaces.c msq.c replacement.c
OBJS = client.o common.o core.o interfaces.o msq.o replacement.o

FLAG_DEBUG = -DGMM_DEBUG
ifeq ($(ENABLE_DEBUG), 0)
FLAG_DEBUG =
endif

FLAG_PROFILE = -DGMM_PROFILE
ifeq ($(ENABLE_PROFILE), 0)
FLAG_PROFILE =
endif

# The compiler/linker settings
CC := gcc
NVCC := $(CUDAPATH)/bin/nvcc
CFLAGS := -O -Wall -pthread -fPIC -fvisibility=hidden \
	-I$(CUDAPATH)/include -DCUDAPATH=\"$(CUDAPATH)\" \
	$(FLAG_DEBUG) $(FLAG_PROFILE) $(GMM_CONFIGS)
LDFLAGS := -shared -pthread -ldl -fPIC

.DEFAULT_GOAL := all
.PHONY : depend all run purerun clean

all : depend gmmctl $(LIBGMM)

# Generate dependencies for $(OBJS)
depend : .depend

.depend : $(SRCS)
	$(CC) $(CFLAGS) -MM $(SRCS) > .depend

-include .depend

# No rules for source files
%.c : ;

gmmctl : server.o
	$(NVCC) -L$(CUDAPATH)/lib64 -arch=sm_20 $^ -o $@

server.o : server.c protocol.h spinlock.h list.h atomic.h
	$(NVCC) -c -I$(CUDAPATH)/include -arch=sm_20 $< -o $@

$(LIBGMM): $(OBJS)
	$(CC) $(LDFLAGS) $(OBJS) -o $@

#test:
#	$(NVCC) -arch=sm_20 test.cu -o $@

run:
	./gmmctl --start
	LD_PRELOAD=`pwd`/$(LIBGMM) ./test
	./gmmctl --stop

purerun:
	./test

clean:
	-rm -f gmmctl $(LIBGMM) test *.o .depend

# TODO
install: ;

# TODO
uninstall: ;
