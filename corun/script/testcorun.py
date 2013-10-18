#!/usr/bin/python
import os
import time
import sys, getopt

os.chdir("../../")
rootpath = os.getcwd()

outpath = rootpath + r'/corun/output'
querypath = rootpath + r'/corun/query_progs'
datapath = rootpath + r'/gpudb/data_10'
rep = '1'

if sys.argv[1] is None or sys.argv[2] is None:
	print 'must specify the query'
	sys.exit()

print sys.argv[1], sys.argv[2]


querys = sys.argv[1], sys.argv[2]
os.chdir(querypath)
for query in querys:
	cmd = './' + query + ' ' + rep + ' --datadir ' + datapath + ' &'
	print cmd
	os.system(cmd)

time.sleep(1)
cmd = ''
os.system(cmd)
