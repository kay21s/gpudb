#!/usr/bin/python
import os
import time

os.chdir("../../")
rootpath = os.getcwd()

solo_dir = rootpath + r'/trace/file/'

stat_num = 5
print 'Query\tSolo Run(ms)\t',
for i in range(1, stat_num+1):
	print i,'\t',
print ''

files = os.listdir(solo_dir)
files.sort()
for file in files:
	if file[-4:] == 'solo':
		avg = 0.0
		num = 0.0
		time_list = []
		for line in open(solo_dir+file, "r").readlines():
			if line[:12] == 'Total Time: ':
				time = float(line[12:])
				time_list.append(round(time,2))
				avg += time
				num += 1
		avg = round(avg/num, 2)
		print file[:-5],'\t',avg,'\t',
		for i in range(0, int(num)):
			print time_list[i],'\t',
		print ''

