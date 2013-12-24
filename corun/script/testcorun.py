#!/usr/bin/python
import os
import time
import sys, getopt
import shutil

os.chdir("../../")
rootpath = os.getcwd()

plan_file = rootpath + r'/corun/exec_plan/2q.plan'
outpath = rootpath + r'/corun/output'
querypath = rootpath + r'/corun/query_progs'
datapath = rootpath + r'/gpudb/data'

LOAD_GMM = 1

if LOAD_GMM:
	preloadlib=r'LD_PRELOAD='+rootpath+r'/gdb/src/libgmm.so '
else:
	preloadlib=''
#preloadlib = ''
#preloadlib = 'LD_PRELOAD=' + rootpath + r'/lib-intercept/libicept.so'
#preloadlib = 'LD_PRELOAD=' + rootpath + r'/gdb/src/libgmm.so'

# corun for #rep times
rep = '9'


if sys.argv[1] is None or sys.argv[2] is None:
	print 'must specify the query'
	sys.exit()


plan = ''
for i in range(1, len(sys.argv)):
	plan = plan + sys.argv[i] + ' '

print plan

plans = []
plans.append(plan)

for plan in plans:
	output = outpath + '/' + plan.strip().replace(' ', '.')
	if os.path.exists(output):
		shutil.rmtree(output)
	cmd = 'mkdir ' + output
	os.system(cmd)
	running_query = {}

	querys = plan.strip().split(" ")
	os.chdir(querypath)
	print plan
	# Solorun the querys to load data into memory first
	for query in querys:
		cmd = preloadlib + ' ./' + query + ' --datadir ' + datapath
		os.system(cmd)
	# Now we corun the querys
	for query in querys:
		if running_query.has_key(query) is True:
			oo = output + '/' + query + '_' + str(running_query[query])
			running_query[query] += 1
		else:
			oo = output + '/' + query
			running_query[query] = 1

		cmd = preloadlib + ' ./' + query + ' --datadir ' + datapath + ' >> ' + oo + ' 2>>' + output+'/error'
		script = rootpath+'/corun/script/reprun.py ' + rep + ' ' + cmd + ' &'
		print script
		os.system(script)

	time.sleep(45)
	cmd=' '
	os.system(cmd) # like press an enter for the last '&'
	for query in querys:
		cmd = r'ps -C ' + query + ' -o pid=|xargs'
		pid = os.popen(cmd).read().strip()
		if pid:
			cmd ='kill -9 ' + pid
			os.system(cmd)
			oo = query+' is killed, ' + pid
			print oo
		#cmd = oo+ ' > ' + output + '/'
		#os.system(oo)
