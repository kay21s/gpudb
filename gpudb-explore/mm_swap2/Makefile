all: server client test lib

cfu = -I/home/syma/src/libcfu/src -Wl,-rpath=/usr/local/lib -lcfu	

test: 
	nvcc -I${CUDA_SDK_PATH}/include -I/home/syma/etc/CUDA_5.0_SAMPLES/common/inc test.cu -arch=sm_20 -o test

server: gmm_server.c gmm_core.c
	gcc -I${CUDA_SDK_PATH}/include $(cfu) -pthread -ldl -fPIC gmm_server.c gmm_core.c -o gmm_server

client: gmm_client.c gmm_core.c
	gcc -I${CUDA_SDK_PATH}/include $(cfu) -pthread -ldl -fPIC gmm_client.c gmm_core.c -o gmm_client

lib: intercept_gmm.c gmm_core.c
	gcc -I${CUDA_SDK_PATH}/include -shared $(cfu) -ldl -fPIC intercept_gmm.c gmm_core.c -o ic_gmm.so

run:
	LD_PRELOAD=`pwd`/ic_gmm.so ./test
	
pure_run:
	./test
	
clean:
	rm -f gmm_server gmm_client ic_gmm.so test

