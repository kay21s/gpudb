#!/usr/bin/python
import os
import time

os.chdir("../../")
rootpath = os.getcwd()

outpath = rootpath + r'/corun/output'

os.chdir(outpath)

crush_corun = []

for dir in os.listdir("."):
	err = open(dir+'/error').read()
	if len(err) <> 0:
		crush_corun.append(dir)

crush_corun.sort()
for q in crush_corun:
	print q

