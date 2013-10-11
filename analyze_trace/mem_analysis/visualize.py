#!/usr/bin/python

import sys
import os
import string
import math


def process_memspace(file_path):
	fin = open(file_path, "r")
	lines = fin.readlines()
	fin.close()

	dp_space = []
	space = 0

	for line in lines:
		words = line.split()
		if len(words) < 2:
			continue

#		tt = words[0].split('.')
#		t = float(tt[0]) + float(tt[1]) / 1000000.0
		t = float(words[0])
		d = int(words[1])

		dp_space.append([t, space])
		space += d
		dp_space.append([t, space])

	return dp_space


def process_memusage(file_path):
	fin = open(file_path, "r")
	lines = fin.readlines()
	fin.close()

	dp_usage = []

	for line in lines:
		words = line.split()
		if len(words) < 3:
			continue

#		tt = words[0].split('.')
#		t1 = float(tt[0]) + float(tt[1]) / 1000000.0
#		tt = words[1].split('.')
#		t2 = float(tt[0]) + float(tt[1]) / 1000000.0
		t1 = float(words[0])
		t2 = float(words[1])
		usage = int(words[2])

		dp_usage.append([t1, 0])
		dp_usage.append([t1, usage])
		dp_usage.append([t2, usage])
		dp_usage.append([t2, 0])

	return dp_usage


def process_memcpy(file_path):
	fin = open(file_path, "r")
	lines = fin.readlines()
	fin.close()

	dp_copy = []

	for line in lines:
		words = line.split()
		if len(words) < 3:
			continue

		t1 = float(words[0])
		t2 = float(words[1])
		copied = int(words[2])

		dp_copy.append([t1, 0])
		dp_copy.append([t1, copied])
		dp_copy.append([t2, copied])
		dp_copy.append([t2, 0])

	return dp_copy


# main
if len(sys.argv) < 4:
	print "USAGE: " + sys.argv[0] + " mem_space_file mem_usage_file mem_copy_file"

dp_space = process_memspace(sys.argv[1])
dp_usage = process_memusage(sys.argv[2])
dp_copy = process_memcpy(sys.argv[3])

t_min = 0.0
if dp_space[0][0] < dp_usage[0][0]:
	t_min = dp_space[0][0]
	if dp_copy[0][0] < t_min:
		t_min = dp_copy[0][0]
else:
	t_min = dp_usage[0][0]
	if dp_copy[0][0] < t_min:
		t_min = dp_copy[0][0]
t_min -= 0.001	# start drawing 1ms ahead

dp_space.insert(0, [t_min, 0])
dp_usage.insert(0, [t_min, 0])
dp_copy.insert(0, [t_min, 0])

t_max = 0.0
if dp_space[-1][0] < dp_usage[-1][0]:
	t_max = dp_usage[-1][0]
	if t_max < dp_copy[-1][0]:
		t_max = dp_copy[-1][0]
else:
	t_max = dp_space[-1][0]
	if t_max < dp_copy[-1][0]:
		t_max = dp_copy[-1][0]
t_max += 0.001	# end drawing 1ms afterwards

dp_space.append([t_max, 0])
dp_usage.append([t_max, 0])
dp_copy.append([t_max, 0])

# save results
print sys.argv[1]
ff = r'./vis/' + sys.argv[1][12:-3] + 'vis'
fout = open(ff, "w")
fout.write("#time\tspace\n")
for dp in dp_space:
	fout.write(('%.3f' % ((dp[0] - t_min) * 1000.0)) + "\t" + str(dp[1]) + "\n")
fout.close()

ff = r'./vis/' + sys.argv[2][12:-3] + 'vis'
fout = open(ff, "w")
fout.write("#time\tusage\n")
for dp in dp_usage:
	fout.write(('%.3f' % ((dp[0] - t_min) * 1000.0)) + "\t" + str(dp[1]) + "\n")
fout.close()

ff = r'./vis/' + sys.argv[3][12:-3] + 'vis'
fout = open(ff, "w")
fout.write("#time\tcopied\n")
for dp in dp_copy:
	fout.write(('%.3f' % ((dp[0] - t_min) * 1000.0)) + "\t" + str(dp[1]) + "\n")
fout.close()
