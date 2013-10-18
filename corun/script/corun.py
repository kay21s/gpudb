#!/usr/bin/python
import os
import time

os.chdir("../../")
rootpath = os.getcwd()

plan_file = rootpath + r'/corun/exec_plan/2q.plan'
outpath = rootpath + r'/corun/output'
querypath = rootpath + r'/corun/query_progs'
datapath = rootpath + r'/gpudb/data_10'
preloadlib = rootpath + r'/lib-intercept/libicept.so'
rep = '7'

if not os.path.exists(outpath):
	os.mkdir(outpath)

plans = open(plan_file, "r").readlines()

for plan in plans:
	output = outpath + '/' + plan.strip().replace(' ', '.')
	cmd = 'mkdir ' + output
	os.system(cmd)

	querys = plan.strip().split(" ")
	os.chdir(querypath)
	print plan
	for query in querys:
		oo = output + '/' + query
		if os.path.isfile(oo):
			cmd = 'LD_PRELOAD='+ preloadlib + ' ./' + query + ' ' + rep + ' --datadir ' + datapath + ' > ' + output + '/' + query + 'g &'
		else:
			cmd = 'LD_PRELOAD='+ preloadlib + ' ./' + query + ' ' + rep + ' --datadir ' + datapath + ' > ' + output + '/' + query + ' &'
		#print cmd
		os.system(cmd)

	time.sleep(10)
	cmd=''
	os.system(cmd) # like press an enter for the last '&'
	for query in querys:
		cmd = r'ps -C ' + query + ' -o pid=|xargs'
		pid = os.popen(cmd).read().strip()
		if pid:
			cmd ='kill -9 ' + pid
			os.system(cmd)
			oo = query+' is killed'
			print oo
		#cmd = oo+ ' > ' + output + '/'
		#os.system(oo)
