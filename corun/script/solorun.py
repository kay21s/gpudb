#!/usr/bin/python
import os
import time

os.chdir("../../")
rootpath = os.getcwd()

outpath = rootpath + r'/trace/file/'
querypath = rootpath + r'/corun/query_progs/'
datapath = rootpath + r'/gpudb/data/'

LOAD_GMM = 1

if LOAD_GMM:
	preloadlib=r'LD_PRELOAD='+rootpath+r'/gdb/src/libgmm.so '
else:
	preloadlib=''
#preloadlib = ' '
#preloadlib = 'LD_PRELOAD=' + rootpath + r'/lib-intercept/libicept.so '
#preloadlib = 'LD_PRELOAD=' + rootpath + r'/gdb/src/libgmm.so '

os.chdir(querypath)

time = 5

for query in os.listdir(querypath):
	# load the column
	cmd = preloadlib + './' + query + ' --datadir ' + datapath
	os.system(cmd)
	cmd = 'rm -f ' + outpath + query + '.solo'
	os.system(cmd)
	for i in range(0, 5):
		cmd = preloadlib + './' + query + ' --datadir ' + datapath + ' >> ' + outpath + query + '.solo'
		os.system(cmd)
