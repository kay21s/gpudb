#!/usr/bin/python
import os

os.chdir("../")
rootpath = os.getcwd()

ldpreload=''
#ldpreload=r'LD_PRELOAD='+rootpath+r'/lib-intercept/libicept.so '
#ldpreload=r'LD_PRELOAD='+rootpath+r'/gdb/src/libgmm.so '

make_command = 'make gpudb'
#make_command = 'make gmmdb'

if not os.path.exists(rootpath+'/trace/'):
	os.mkdir(rootpath+'/trace/')
if not os.path.exists(rootpath+'/trace/file/'):
	os.mkdir(rootpath+'/trace/file/')

for file in os.listdir(rootpath + "/gpudb/test/ssb_test/"):
	if file[-3:] == 'sql':
		os.chdir(rootpath+"/gpudb/")
		cmd = rootpath+r'/gpudb/translate.py ' + rootpath+r'/gpudb/test/ssb_test/' + file + ' ' + rootpath+r'/gpudb/test/ssb_test/ssb.schema'
		os.system(cmd)
		os.chdir(rootpath + "/gpudb/src/cuda")
		os.system(make_command)
		output = file[0:-3] + 'solo'
		#! New GMM add
		#os.chdir(rootpath + "/gpudb-explore/src")
		cmd = ldpreload + rootpath+r'/gpudb/src/cuda/GPUDATABASE --datadir '+ rootpath+r'/gpudb/data'
		os.system(cmd)
		cmd = ldpreload + rootpath+r'/gpudb/src/cuda/GPUDATABASE --datadir '+ rootpath+r'/gpudb/data' + r' > ' + rootpath+r'/trace/file/'+output
		os.system(cmd)
