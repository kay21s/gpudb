#!/usr/bin/python
import os
import sys
import time

os.chdir("../../")
rootpath = os.getcwd()

preloadlib=r'LD_PRELOAD='+rootpath+r'/gdb/src/libgmm.so '
querypath = rootpath + r'/corun/query_progs/'
datapath = ' --datadir ' + rootpath + r'/gpudb/data/'

if len(sys.argv) == 2:
	q1 = querypath+sys.argv[1]
	q2 = querypath+sys.argv[1]
elif len(sys.argv) == 3:
	q1 = querypath+sys.argv[1]
	q2 = querypath+sys.argv[2]
else:
	print 'specify the queries to corun'
	sys.exit(0)

cmd = preloadlib + q1 + datapath + ' & ' + preloadlib + q2 + datapath
os.system(cmd)

