#!/usr/bin/python
import os
import time

plan_file=r'/home/kai/projects/corun/exec_plan/2q.plan'
outpath=r'/home/kai/projects/corun/output'
querypath=r'/home/kai/projects/corun/query_progs'
datapath=r'/home/kai/projects/gpudb/data_10'
preloadlib=r'/home/kai/projects/lib-intercept/libicept.so'
rep = '3'

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

	time.sleep(5)
	cmd=''
	os.system(cmd)
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

