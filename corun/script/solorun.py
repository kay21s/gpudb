#!/usr/bin/python
import os
import shutil

os.chdir("../../")
rootpath = os.getcwd()

outpath = rootpath + r'/trace/file/'
querypath = rootpath + r'/corun/query_progs/'
datapath = rootpath + r'/gpudb/data/'

rep = 6
LOAD_GMM = 1
if LOAD_GMM:
	preloadlib=r'LD_PRELOAD='+rootpath+r'/gdb/src/libgmm.so '
else:
	preloadlib=''
#preloadlib = ' '
#preloadlib = 'LD_PRELOAD=' + rootpath + r'/lib-intercept/libicept.so '
#preloadlib = 'LD_PRELOAD=' + rootpath + r'/gdb/src/libgmm.so '

if os.path.exists(outpath):
	shutil.rmtree(outpath)
os.mkdir(outpath)

os.chdir(querypath)

for query in os.listdir(querypath):
	# load the column
	cmd = preloadlib + './' + query + ' --datadir ' + datapath
	os.system(cmd)
	cmd = 'rm -f ' + outpath + query + '.solo'
	os.system(cmd)
	for i in range(0, rep):
		cmd = preloadlib + './' + query + ' --datadir ' + datapath + ' >> ' + outpath + query + '.solo'
		os.system(cmd)
