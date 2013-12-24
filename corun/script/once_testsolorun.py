#!/usr/bin/python
import os
import sys
import time

os.chdir("../../")
rootpath = os.getcwd()

preloadlib=r'LD_PRELOAD='+rootpath+r'/gdb/src/libgmm.so '
#preloadlib = ''
querypath = rootpath + r'/corun/query_progs/'
datapath = ' --datadir ' + rootpath + r'/gpudb/data/'

if len(sys.argv) == 2:
	q = querypath+sys.argv[1]
else:
	print 'specify the queries to corun'
	sys.exit(0)

cmd = preloadlib + q + datapath
print cmd
os.system(cmd)

