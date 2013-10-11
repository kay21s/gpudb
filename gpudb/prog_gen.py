#!/usr/bin/python
import os

if not os.path.exists('/home/kai/projects/corun/query_progs/'):
	os.mkdir('/home/kai/projects/corun/query_progs/')

for file in os.listdir("/home/kai/projects/gpudb/test/ssb_test/"):
	if file[-3:] == 'sql':
		os.chdir("/home/kai/projects/gpudb/")
		cmd = r'/home/kai/projects/gpudb/translate.py' + r' /home/kai/projects/gpudb/test/ssb_test/' + file + r' /home/kai/projects/gpudb/test/ssb_test/ssb.schema'
		os.system(cmd)
		#os.system('/home/kai/projects/gpudb/translate file /home/kai/projects/gpudb/test/ssb_test/ssb.schema')
		os.chdir("/home/kai/projects/gpudb/src/cuda")
		os.system('make gpudb')
		cmd = r'cp GPUDATABASE /home/kai/projects/corun/query_progs/' + file[:-4]
		os.system(cmd)
		output = file[0:-3] + 'solo'
		cmd = r'LD_PRELOAD=/home/kai/projects/lib-intercept/libicept.so /home/kai/projects/gpudb/src/cuda/GPUDATABASE --datadir /home/kai/projects/gpudb/data_10' + r' > ' + r'/home/kai/projects/trace/file/' + output
		os.system(cmd)
