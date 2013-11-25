
#!/usr/bin/python
import os
import sys

os.chdir("../../")
rootpath = os.getcwd()

corun_path = rootpath + r'/corun/output/'
querypath = rootpath + r'/corun/query_progs/'
solorun_trace_path = rootpath + r'/trace/file/'
context_path = rootpath + r'/corun/result/'

if len(sys.argv) != 2:
	print 'give a context name to be restored'
	sys.exit(0)
elif os.path.exists(context_path+sys.argv[1]):
	print 'context exists!'
	sys.exit(0)

the_path = context_path + sys.argv[1]
cmd = 'cp -rf ' + query_path + ' ' + the_path
os.system(cmd)
cmd = 'cp -r ' + solorun_trace_path + ' ' + the_path
os.system(cmd)
cmd = 'cp -r ' + corun_trace_path + ' ' + the_path
os.system(cmd)

os.chdir(rootpath + '/corun/script/')
cmd = 'cp result_corun speedup0 speedup1 plot_corun.plot speedup.png ' + the_path

