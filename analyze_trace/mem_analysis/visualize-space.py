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


# main
if len(sys.argv) < 2:
	print "USAGE: " + sys.argv[0] + " mem_space_file"

dp_space = process_memspace(sys.argv[1])

t_min = dp_space[0][0] - 0.001	# start drawing 1ms ahead
dp_space.insert(0, [t_min, 0])

t_max = dp_space[-1][0] + 0.01
dp_space.append([t_max, 0])

# save results
fout = open("memspace.vis", "w")
fout.write("#time\tspace\n")
for dp in dp_space:
	fout.write(('%.3f' % ((dp[0] - t_min) * 1000.0)) + "\t" + str(dp[1]) + "\n")
fout.close()
