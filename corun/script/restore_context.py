#!/usr/bin/python
import os
import sys

os.chdir("../../")
rootpath = os.getcwd()

corun_path = rootpath + r'/corun/'
solorun_path = rootpath + r'/trace/'
context_path = rootpath + r'/corun/result/'

os.system('rm -f result_corun speedup0 speedup1 plot_corun.plot speedup.png')

if len(sys.argv) != 2:
	print 'give a context name to be restored'
	sys.exit(0)
elif not os.path.exists(context_path+sys.argv[1]):
	print 'context not exists!'
	sys.exit(0)

the_path = context_path + sys.argv[1]
os.chdir(the_path)
os.system('rm -rf ' + corun_path + 'query_progs/')
os.system('cp -r ' + './query_progs ' + corun_path)
os.system('rm -rf ' + corun_path + 'output/')
os.system('cp -r ' + './output ' + corun_path)
os.system('rm -rf ' + solorun_path + 'file/')
os.system('cp -r ' + './file ' + solorun_path)


