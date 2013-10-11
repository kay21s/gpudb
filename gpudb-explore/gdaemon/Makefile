all:
	nvcc -I${CUDA_SDK_PATH}/include -I/home/syma/etc/CUDA_5.0_SAMPLES/common/inc test.cu -arch=sm_20 -o test
	gcc -I${CUDA_SDK_PATH}/include -shared -ldl -fPIC intercept.c gdaemon.c -o gdaemon.so

async: 
	gcc -I${CUDA_SDK_PATH}/include -shared -ldl -fPIC intercept_async.c -o ic_async.so
stream: 
	gcc -I${CUDA_SDK_PATH}/include -shared -ldl -fPIC intercept_stream.c -o ic_stream.so
mstream: 
	gcc -I${CUDA_SDK_PATH}/include -shared -ldl -fPIC intercept_mstream.c -o ic_mstream.so

run:
	LD_PRELOAD=`pwd`/gdaemon.so ./test
run_async:
	LD_PRELOAD=`pwd`/ic_async.so ./test
run_stream:
	LD_PRELOAD=`pwd`/ic_stream.so ./test

test_async : test_async.cu
	nvcc -I${CUDA_SDK_PATH}/include -I/home/syma/etc/CUDA_5.0_SAMPLES/common/inc test_async.cu -arch=sm_20 -o test_async

run_test_async: test_async async
	time ./test_async
	time LD_PRELOAD=`pwd`/ic_async.so ./test_async
	
test_async2 : test_asyncP1.cu test_asyncP2.cu
	nvcc -I${CUDA_SDK_PATH}/include -I/home/syma/etc/CUDA_5.0_SAMPLES/common/inc test_asyncP1.cu -arch=sm_20 -o test_asyncP1
	nvcc -I${CUDA_SDK_PATH}/include -I/home/syma/etc/CUDA_5.0_SAMPLES/common/inc test_asyncP2.cu -arch=sm_20 -o test_asyncP2

run_test_async2: test_async2 async stream
	./test_asyncP1 
	echo -e "\n"
	time ./test_asyncP2
	echo -e "\n"
	./test_asyncP1 &
	time ./test_asyncP2
	echo -e "\n"
	./test_asyncP1 &
	time LD_PRELOAD=`pwd`/ic_stream.so ./test_asyncP2
	
clean:
	rm -f test gdaemon.so ic_async.so ic_stream.so ic_mstream.so test_async test_asyncP1 test_asyncP2
