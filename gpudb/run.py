#!/usr/bin/python
import os

os.chdir("../")
rootpath = os.getcwd()

ldpreload=''
#ldpreload=r'LD_PRELOAD='+rootpath+r'/lib-intercept/libicept.so '

for file in os.listdir(rootpath + "/gpudb/test/ssb_test/"):
	if file[-3:] == 'sql':
		os.chdir(rootpath+"/gpudb/")
		cmd = rootpath+r'/gpudb/translate.py ' + rootpath + r'/gpudb/test/ssb_test/' + file + ' ' + rootpath + r'/gpudb/test/ssb_test/ssb.schema'
		os.system(cmd)
		#os.system('/home/kai/projects/gpudb/translate file /home/kai/projects/gpudb/test/ssb_test/ssb.schema')
		os.chdir(rootpath + "/gpudb/src/cuda")
		os.system('make gpudb')
		output = file[0:-3] + 'solo'
		#! New GMM add
		os.chdir(rootpath + "/gpudb-explore/src")
		cmd = ldpreload + rootpath + r'/gpudb/src/cuda/GPUDATABASE --datadir '+ rootpath + r'/gpudb/data' + r' > ' + rootpath + r'/trace/file/' + output
		#! OLD libicept test
		#cmd = r'LD_PRELOAD=/home/kai/projects/lib-intercept/libicept.so /home/kai/projects/gpudb/src/cuda/GPUDATABASE --datadir /home/kai/projects/gpudb/data' + r' > ' + r'/home/kai/projects/trace/file/' + output
		os.system(cmd)
