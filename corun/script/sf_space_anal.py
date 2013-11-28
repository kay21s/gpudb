#!/usr/bin/python
from __future__ import division
import os
import sys
import locale

locale.setlocale(locale.LC_ALL, 'en_US')

os.chdir("../../")
rootpath = os.getcwd()

mem_anal_dir = rootpath+ '/analyze_trace/mem_analysis/vis/'

if len(sys.argv) == 2:
	sf = int(sys.argv[1])
else:
	sf = 10

os.chdir(mem_anal_dir)
query_max_space = {}
for file in os.listdir(mem_anal_dir):
	max = 0
	if 'memspace.vis' in file:
		query = file[:4]
		for line in open(file, 'r').readlines():
			if 'time' in line:
				continue
			space = int(line.strip().split('\t')[1])
			if max < space:
				max = space
		query_max_space[file[:4]] = max

gpu_mem = 1470000000

print 'With Scale Factor ', sf
print '--------------------------------\n'

num = 0
print 'Solorun space :'
space_dict = sorted(query_max_space.iteritems(), key=lambda d:d[0])
for (query, space) in space_dict:
	print query, '\t', locale.format("%d", int(space*(sf/10)), grouping=True),
	if space*(sf/10) > gpu_mem:
		print '\t*'
		num += 1
	else:
		print ''
print '--------------------------------'
print 'Larger than GPU memory : ', num, '/', len(space_dict)


querys = []
for key in query_max_space.keys():
	querys.append(key)
querys.sort()

num = 0
total = 0
print '\nCorun space :'
for i in range(0, len(querys)):
	for j in range(i, len(querys)):
		corun_space = float((query_max_space[querys[i]] + query_max_space[querys[j]]) * (sf/10))
		#print querys[i], '\t', querys[j], '\t', locale.format("%d", corun_space, grouping=True),
		print querys[i], '\t', querys[j], '\t', "%.2f"% (corun_space/1000000000.0),
		if corun_space > gpu_mem:
			print '\t*'
			num += 1
		else:
			print ''
		total +=1

print '--------------------------------'
print 'Larger than GPU memory : ', num, '/', total
