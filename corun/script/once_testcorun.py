#!/usr/bin/python
import os
import sys
import time
import shutil

os.chdir("../../")
rootpath = os.getcwd()

preloadlib=r'LD_PRELOAD='+rootpath+r'/gdb/src/libgmm.so '
#preloadlib = ''
querypath = rootpath + r'/corun/query_progs/'
currentpath = rootpath + r'/corun/script/'
datapath = ' --datadir ' + rootpath + r'/gpudb/data/'

query=[]
if len(sys.argv) >= 2:
	for i in range(1, len(sys.argv)):
		query.append(sys.argv[i])
		if i == len(sys.argv) -1:
			cmd = preloadlib + querypath+sys.argv[i] + datapath
		else:
			cmd = preloadlib + querypath+sys.argv[i] + datapath + ' & '
		os.system(cmd)


'''
if len(sys.argv) == 2:
	q1 = querypath+sys.argv[1]
	qq1 = sys.argv[1]
	q2 = querypath+sys.argv[1]
	qq2 = sys.argv[1]
elif len(sys.argv) == 3:
	q1 = querypath+sys.argv[1]
	qq1 = sys.argv[1]
	q2 = querypath+sys.argv[2]
	qq2 = sys.argv[2]
else:
	print 'specify the queries to corun'
	sys.exit(0)

dir= currentpath + qq1 +'.' + qq2 + '/'
if os.path.exists(dir):
	shutil.rmtree(dir)
os.mkdir(dir)

cmd = preloadlib + q1 + datapath + ' > ' + dir + qq1 + ' & ' + preloadlib + q2 + datapath + ' > ' + dir + qq2
os.system(cmd)
'''

