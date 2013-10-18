#!/usr/bin/python
import os

os.chdir("../")
rootpath = os.getcwd()

ldpreload=''
#ldpreload=r'LD_PRELOAD='+rootpath+r'/lib-intercept/libicept.so '

if not os.path.exists(rootpath+'/corun/query_progs/'):
	os.mkdir(rootpath+'/corun/query_progs/')

if not os.path.exists(rootpath+'/trace/file/'):
	os.mkdir(rootpath+'/trace/file/')

for file in os.listdir(rootpath+"/gpudb/test/ssb_test/"):
	if file[-3:] == 'sql':
		os.chdir(rootpath+"/gpudb/")
		cmd = rootpath + r'/gpudb/translate.py' + r' ' + rootpath + '/gpudb/test/ssb_test/' + file + r' '+ rootpath + r'/gpudb/test/ssb_test/ssb.schema'
		os.system(cmd)
		#os.system('/home/kai/projects/gpudb/translate file /home/kai/projects/gpudb/test/ssb_test/ssb.schema')
		os.chdir(rootpath+"/gpudb/src/cuda")
		os.system('make gpudb')
		cmd = r'cp GPUDATABASE ' + rootpath + r'/corun/query_progs/' + file[:-4]
		os.system(cmd)
		output = file[0:-3] + 'solo'
		cmd = ldpreload + rootpath + r'/gpudb/src/cuda/GPUDATABASE 1 --datadir ' + rootpath + r'/gpudb/data_10' + r' > ' + rootpath + r'/trace/file/' + output
		os.system(cmd)
