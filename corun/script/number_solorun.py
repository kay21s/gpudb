#!/usr/bin/python
import os
import sys
import time

os.chdir("../../")
rootpath = os.getcwd()

solo_dir = rootpath + r'/trace/file/'

stat_num = 6
print 'Query\tSolo Run(ms)\t',
for i in range(1, stat_num):
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
				# the first time data is not used since it loads data into memory
				if num == 0:
					num += 1
					continue
				time = float(line[12:])
				time_list.append(round(time,2))
				avg += time
				num += 1
		if num != stat_num:
			print num, '<', stat_num
			sys.exit(0)
		avg = round(avg/(num-1), 2)
		print file[:-5],'\t',avg,'\t',
		for i in range(0, int(num-1)):
			print time_list[i],'\t',
		print ''

