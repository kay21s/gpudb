#!/usr/bin/python
import os
import time
import sys, getopt

outpath=r'/home/kai/projects/corun/output'
querypath=r'/home/kai/projects/corun/query_progs'
datapath=r'/home/kai/projects/gpudb/data_10'
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
