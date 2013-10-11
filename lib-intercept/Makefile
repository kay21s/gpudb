all:
	#nvcc mm.cu -arch=sm_20 -lcutil_x86_64 -L/home/kai/NVIDIA_GPU_Computing_SDK/C/lib -I/home/kai/NVIDIA_GPU_Computing_SDK/C/common/inc -o mm
	gcc -shared -ldl -fPIC -I/usr/local/cuda/include/ intercept.c -o libicept.so
run:
	LD_PRELOAD=/home/kaibo/projects/gpu-vm/tools/lib-intercept/libicept.so ./mm
clean:
	rm -f mm libicept.so
