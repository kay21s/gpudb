#!/usr/bin/python

import sys
import os
import string
import math


def merge_and_sort(list_alloc, list_free):
	list_merged = list_alloc + list_free
	return sorted(list_merged)


def get_log_events(log_file_path):
	log_events = []

	# read in log file
	log_file = open(log_file_path, "r")
	log_lines = log_file.readlines()
	log_file.close()

	# normalize lines to words
	for line in log_lines:
		words = line.split()
		if len(words) < 1 or words[0] <> "[gvm]" or words[2] == "BEGIN":
			continue

		del words[0]	# delete [gvm]
		log_events.append(words)

	return log_events


def anal_memspace(log_events, file):
	# get all `free' events; a free event struct is defined as
	# [idx, time, 0, free_addr, free_size]
	free_events = []
	for idx,event in enumerate(log_events):
		if event[1] == "intercepted" and event[2] == "cudaFree(":
			free_events.append([idx, event[0], 0, int(event[3], 16), 0])

	# get the corresponding `alloc' event for each `free' event; an alloc event
	# struct is defined as
	# [idx, time, 1, alloc_addr, alloc_size]
	alloc_events = []
	unfound_events = []
	for free_idx,free_event in enumerate(free_events):
		alloc_addr = free_event[3]
		alloc_size = 0

		# search from the free event backwards
		i = free_event[0] - 1
		while i >= 0:
			event = log_events[i]
			if event[1] == "intercepted" and event[2] == "cudaMalloc(" and int(event[3], 16) == alloc_addr:
				alloc_size = int(event[4])
				alloc_events.append([i, event[0], 1, alloc_addr, alloc_size])
				free_event[4] = alloc_size
				break
			i -= 1

		if i < 0:
			print "unfound", free_event
			unfound_events.append(free_idx)

	# delete unmatched free events
	for i in unfound_events:
		del free_events[i]

	# merge alloc and free events and sort
	memspace_events = merge_and_sort(alloc_events, free_events)

	# save memspace events
	ff = r'./immediate/' + file[:-5] + r'_memspace.txt'
	fout = open(ff, "w")
	for e in memspace_events:
		if e[2] == 0:
			fout.write(e[1] + "\t" + str(-e[-1]) + "\n")
		else:
			fout.write(e[1] + "\t" + str(e[-1]) + "\n")
	fout.close()

	return sorted(alloc_events)


def search_idx(idx, alloc_events):
	i = len(alloc_events) - 1
	while i >= 0 and alloc_events[i][0] > idx:
		i -= 1
	return i


def anal_memusage(log_events, alloc_events, file):
	nr_events = len(log_events)
	# memory usage list; each element is defined as
	# [start_time, end_time, size_used]
	memusage = []

	# search for cudaLaunch events; for each, compute used memory size
	# and the length of usage
	for idx,event in enumerate(log_events):
		if event[2] == "cudaLaunch" and event[1] == "intercepting":
			start_time = event[0]
		if event[2] == "cudaLaunch" and event[1] == "intercepted":
			end_time = event[0]
			size_used = 0

			# get the size of memory used by the kernel
			i = idx - 2
			while i >= 0 and log_events[i][2] <> "cudaConfigureCall":	# search backward
				if log_events[i][1] == "intercepted" and log_events[i][2] == "cudaSetupArgument(" and log_events[i][3] == "8":
					# get the possible address
					addr_value = 0
					for j in range(0, 8):
						addr_value += int(log_events[i][4+j], 16) * math.pow(256, j)	# little endian

					# search for its size if it's really an address
					size_value = 0
					j = search_idx(i, alloc_events)
					while j >= 0:
						if alloc_events[j][3] == addr_value or addr_value > alloc_events[j][3] and addr_value < alloc_events[j][3] + alloc_events[j][4]:
							size_value = int(alloc_events[j][4] - (addr_value - alloc_events[j][3]))
							break
						j -= 1

					# update size used
					size_used += size_value

				i -= 1

			if i < 0:
				print "error when search backward"
				sys.exit(1)

			# get the termination time of kernel execution by moving forward
			i = idx + 1
			#while i < nr_events:
			#	if log_events[i][2] == "cudaThreadSynchronize" and log_events[i][1] == "intercepted":
			#		end_time = log_events[i][0]
			#		break
			#	i += 1
			#if i >= nr_events:
			#	print "error when search forward; no cudaThreadSynchronize after kernel launch"
			#	sys.exit(1)

#			if size_used > 0:
			memusage.append([start_time, end_time, size_used])
#			else:	# fixme: set the value according to the max size_used
#				memusage.append([start_time, end_time, -1000])	# make it visible on the graph

	# save memusage elements
	ff = r'./immediate/' + file[:-5] + r'_memusage.txt'
	fout = open(ff, "w")
	for e in memusage:
		fout.write(e[0] + "\t" + e[1] + "\t" + str(e[2]) + "\n")
	fout.close()

	return


def anal_memcpy(log_events, alloc_events, file):
	nr_events = len(log_events)
	# memcpy list; each element is defined as [start_time, end_time, size_copied]
	memcpy = []

	for idx,event in enumerate(log_events):
		if event[2] == "cudaMemcpy" and event[1] == "intercepting":
			start_time = event[0]
			end_time = log_events[idx+1][0]
			size_copied = log_events[idx+1][5]
			memcpy.append([start_time, end_time, size_copied])

	ff = r'./immediate/' + file[:-5] + r'_memcpy.txt'
	fout = open(ff, "w")
	for e in memcpy:
		fout.write(e[0] + "\t" + e[1] + "\t" + e[2] + "\n")
	fout.close()

	return

# main
for file in os.listdir("./file/"):
	fname = r'./file/' + file
	log_events = get_log_events(fname)

	alloc_events = anal_memspace(log_events, file)
	anal_memusage(log_events, alloc_events, file)
	anal_memcpy(log_events, alloc_events, file)
