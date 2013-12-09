#!/usr/bin/python
import os
import time
import shutil

os.chdir("../../")
rootpath = os.getcwd()

plan_file = rootpath + r'/corun/exec_plan/2q.plan'
outpath = rootpath + r'/corun/output/'
querypath = rootpath + r'/corun/query_progs/'
datapath = rootpath + r'/gpudb/data/'

# corun for #rep times
rep = '1'

LOAD_GMM = 1
if LOAD_GMM:
	preloadlib=r'LD_PRELOAD='+rootpath+r'/gdb/src/libgmm.so '
else:
	preloadlib=''
#preloadlib = ''
#preloadlib = 'LD_PRELOAD=' + rootpath + r'/lib-intercept/libicept.so'
#preloadlib = 'LD_PRELOAD=' + rootpath + r'/gdb/src/libgmm.so'


if os.path.exists(outpath):
	shutil.rmtree(outpath)
os.mkdir(outpath)

plans = open(plan_file, "r").readlines()

for plan in plans:
	output = outpath + plan.strip().replace(' ', '.') + '/'
	if os.path.exists(output):
		print "Corun for the second time"
		sys.exit(0)
	cmd = 'mkdir ' + output
	os.system(cmd)
	running_query = {}

	querys = plan.strip().split(' ')
	os.chdir(querypath)
	print plan
	# Solorun the querys to load data into memory first
	#for query in querys:
	#	cmd = preloadlib + ' ./' + query + ' --datadir ' + datapath
	#	os.system(cmd)
	# Now we corun the querys
	i = 0
	for query in querys:
		if running_query.has_key(query) is True:
			oo = output + query + '_' + str(running_query[query])
			running_query[query] += 1
		else:
			oo = output + query
			running_query[query] = 1

		if i == 0:
			cmd = preloadlib + ' ./' + query + ' --datadir ' + datapath + ' >> ' + oo + ' 2>>' + output+'error &'
		else:
			cmd = preloadlib + ' ./' + query + ' --datadir ' + datapath + ' >> ' + oo + ' 2>>' + output+'error'
		os.system(cmd)
		i += 1

	time.sleep(10)
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

