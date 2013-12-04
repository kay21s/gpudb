#!/usr/bin/python
import os
import sys
import time

os.chdir("../../")
rootpath = os.getcwd()

outpath = rootpath + r'/corun/output'
os.chdir(outpath)

crush_corun = []
evict_corun = []
coruns = []
for dir in os.listdir('.'):
	coruns.append(dir)
	err = open(dir+'/error').read()
	if len(err) <> 0:
		crush_corun.append(dir)
		continue

	flag = 0
	for file in os.listdir(dir):
		if file != 'error':
			trace_lines = open(dir+'/'+file, 'r').readlines()
			for line in trace_lines:
				if 'evicting region' in line:
					evict_corun.append(dir)
					flag = 1
					break
		if flag == 1:
			break

coruns.sort()

print 'Coruns:\t \t Crush \t Evict'
print '-----------------------------------'
for corun in coruns:
	querys = corun.split('.')
	if len(sys.argv) == 1:
		for q in querys:
			print q, '\t',
	if corun in crush_corun:
		print '*\t',
	else:
		print ' \t',
	if corun in evict_corun:
		print '+\t',
	print ''

'''
if crush_corun:
	crush_corun.sort()
	print 'Crushed Coruns :'
	print '---------------------------'
	for q in crush_corun:
		print q
else:
	print 'No Crushed Coruns'
	print '---------------------------'
print ''

if evict_corun:
	evict_corun.sort()
	print 'Coruns with Evictions:'
	print '---------------------------'
	for q in evict_corun:
		print q
else:
	print 'No Coruns have eviction'
	print '---------------------------'
'''
