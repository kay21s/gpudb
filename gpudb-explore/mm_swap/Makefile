all: server client test lib
	
test: 
	nvcc -I${CUDA_SDK_PATH}/include -I/home/syma/etc/CUDA_5.0_SAMPLES/common/inc test.cu -arch=sm_20 -o test

server: gmm_server.c gmm_meta.c
	gcc -I${CUDA_SDK_PATH}/include -pthread -ldl -fPIC gmm_server.c gmm_meta.c -o gmm_server

client: gmm_client.c gmm_meta.c
	gcc -I${CUDA_SDK_PATH}/include -pthread -ldl -fPIC gmm_client.c gmm_meta.c -o gmm_client

lib: intercept_gmm.c gmm_meta.c
	gcc -I${CUDA_SDK_PATH}/include -shared -ldl -fPIC intercept_gmm.c gmm_meta.c -o ic_gmm.so

run:
	LD_PRELOAD=`pwd`/ic_gmm.so ./test
	
pure_run:
	./test
	
clean:
	rm -f gmm_server gmm_client ic_gmm.so test

