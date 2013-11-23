#!/usr/bin/python
import os

os.chdir("../")
rootpath = os.getcwd()

LOAD_GMM = 1

if LOAD_GMM:
	ldpreload=r'LD_PRELOAD='+rootpath+r'/gdb/src/libgmm.so '
	make_command = 'make gmmdb'
else:
	ldpreload=''
	make_command = 'make gpudb'
#elif LOAD_LIBICEPT:
#	ldpreload=r'LD_PRELOAD='+rootpath+r'/lib-intercept/libicept.so '
#	make_command = 'make gpudb'


if not os.path.exists(rootpath+'/corun/query_progs/'):
	os.mkdir(rootpath+'/corun/query_progs/')

if not os.path.exists(rootpath+'/trace/'):
	os.mkdir(rootpath+'/trace/')
if not os.path.exists(rootpath+'/trace/file/'):
	os.mkdir(rootpath+'/trace/file/')

for file in os.listdir(rootpath+"/gpudb/test/ssb_test/"):
	if file[-3:] == 'sql':
		os.chdir(rootpath+"/gpudb/")
		cmd = rootpath + r'/gpudb/translate.py' + r' ' + rootpath+'/gpudb/test/ssb_test/'+file + r' '+ rootpath+r'/gpudb/test/ssb_test/ssb.schema'
		os.system(cmd)
		os.chdir(rootpath+"/gpudb/src/cuda")
		os.system(make_command)
		cmd = r'cp GPUDATABASE ' + rootpath + r'/corun/query_progs/' + file[:-4]
		os.system(cmd)
		output = file[0:-3] + 'solo'
		cmd = ldpreload + rootpath + r'/gpudb/src/cuda/GPUDATABASE --datadir ' + rootpath+r'/gpudb/data' + r' > ' + rootpath+r'/trace/file/'+output
		os.system(cmd)
