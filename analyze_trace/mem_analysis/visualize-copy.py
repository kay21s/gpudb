#!/usr/bin/python

import sys
import os
import string
import math


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
if len(sys.argv) < 2:
	print "USAGE: " + sys.argv[0] + " mem_copy_file"

dp_copy = process_memcpy(sys.argv[1])

t_min = dp_copy[0][0] - 0.001	# start drawing 1ms ahead
dp_copy.insert(0, [t_min, 0])

t_max = dp_copy[-1][0] + 0.001	# end drawing 1ms afterwards
dp_copy.append([t_max, 0])

# save results
fout = open("memcpy.vis", "w")
fout.write("#time\tcopied\n")
for dp in dp_copy:
	fout.write(('%.3f' % ((dp[0] - t_min) * 1000.0)) + "\t" + str(dp[1]) + "\n")
fout.close()
