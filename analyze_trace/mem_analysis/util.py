#!/usr/bin/python

import sys
import os
import string
import math


def read_in_memspace(filepath):
	mem_space = []

	fin = open(filepath, "r")
	lines = fin.readlines()
	fin.close()

	prev_time = float("-inf")
	prev_space = 0.0

	for line in lines:
		words = line.split()
		if len(words) < 2:
			continue

		mem_space.append([prev_time, float(words[0]), prev_space])
		prev_time = float(words[0])
		prev_space += float(words[1])

	mem_space.append([prev_time, float("inf"), prev_space])
	return mem_space


def read_in_memusage(filepath):
	mem_usage = []

	fin = open(filepath, "r")
	lines = fin.readlines()
	fin.close()

	for line in lines:
		words = line.split()
		if len(words) < 3:
			continue
		mem_usage.append([float(words[0]), float(words[1]), float(words[2])])

	return mem_usage


def read_in_memcpy(filepath):
	mem_copy = []

	fin = open(filepath, "r")
	lines = fin.readlines()
	fin.close()

	for line in lines:
		words = line.split()
		if len(words) < 3:
			continue
		mem_copy.append([float(words[0]), float(words[1]), float(words[2])])

	return mem_copy


def anal_idle_interval(time_start, time_end, mem_space):
	if time_start < mem_space[0][1]:
		time_start = mem_space[0][1]
	if time_end > mem_space[-1][0]:
		time_end = mem_space[-1][0]

	l = time_end - time_start

	# eliminate zero intervals that fall into the interval
	for e in mem_space:
		if e[2] == 0.0 and e[0] >= time_start and e[1] <= time_end:
			l -= e[1] - e[0];
		elif e[0] > time_end:
			break

	return l


def get_space(t, mem_space):
	s = 0.0
	for e in mem_space:
		if t >= e[0] and t < e[1]:
			s = e[2]
			break
	return s


def anal_busy_interval(busy_item, mem_space):
	l = busy_item[1] - busy_item[0]
	t = (busy_item[1] + busy_item[0]) / 2.0
	s = get_space(t, mem_space)
	if s <= 0.0:
		print "error: memspace zero during a busy interval"
		sys.exit(1)
	return [l, busy_item[2] / s]


def compute_util(mem_space, mem_busy):
	utils = []

	# the first zero-usage interval
	l = anal_idle_interval(mem_space[0][1], mem_busy[0][0], mem_space)
	if l > 0.0:
		utils.append([l, 0.0])

	# for each busy interval and its immediate zero-usage interval
	for i in range(0, len(mem_busy)):
		[l, u] = anal_busy_interval(mem_busy[i], mem_space)
		if l > 0.0:
			utils.append([l, u])

		l = 0.0
		if i < len(mem_busy) - 1:
			l = anal_idle_interval(mem_busy[i][1], mem_busy[i+1][0], mem_space)
		else:
			l = anal_idle_interval(mem_busy[i][1], float("inf"), mem_space)
		if l > 0.0:
			utils.append([l, 0.0])

	# compute the utility
	l_tot = 0.0
	mem_util = 0.0
	for u in utils:
		l_tot += u[0]
		mem_util += u[0] * u[1]
	if l_tot > 0.0:
		mem_util /= l_tot

	return mem_util


# main
if len(sys.argv) < 4:
	print "USAGE: " + sys.argv[0] + " mem_space_file mem_usage_file mem_copy_file"

mem_space = read_in_memspace(sys.argv[1])
mem_usage = read_in_memusage(sys.argv[2])
mem_copy = read_in_memcpy(sys.argv[3])

mem_busy = sorted(mem_usage + mem_copy)

mem_util = compute_util(mem_space, mem_busy)
print "Memory utilization:", mem_util * 100.0, "%"
