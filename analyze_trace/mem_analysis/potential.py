#!/usr/bin/python

import sys
import os
import string
import math
from subprocess import call

def get_log_events(log_file_path):
	log_events = []

	# read in log file
	log_file = open(log_file_path, "r")
	log_lines = log_file.readlines()
	log_file.close()

	# normalize lines to words
	for line in log_lines:
		words = line.split()
		if len(words) < 1 or words[0] <> "[gvm]":
			continue

		del words[0]	# delete [gvm]
		log_events.append(words)

	return log_events


def get_pid(addr, plist):
	pid = ""

	for k,v in plist.iteritems():
		if v[0] == addr:
			pid = k
			break

	return pid


def generate_plot_1(time_min, time_max, plist):
	# compute vertical offsets, save them to v[0]
	# leave an interval of 10 in space
	offset_curr = 10

	for k,v in plist.iteritems():
		v[0] = offset_curr

		size_max = 0
		for e in v[1:]:
			if e[1] == "malloc" and e[2] > size_max:
				size_max = e[2]

		offset_curr += size_max + 10

	# generate plot data files
	i = 0
	for k,v in plist.iteritems():
		fout = open("memspace" + str(i) + ".vis", "w")
		#t = time_min
		offset = v[0]
		y = 0

		fout.write("0\t" + str(offset) + "\n")
		for e in v[1:]:
			if e[1] == "malloc":
				fout.write(str((e[0] - time_min) * 1000.0) + "\t" + str(offset) + "\n")
				y = e[2]
				fout.write(str((e[0] - time_min) * 1000.0) + "\t" + str(offset + y) + "\n")
			elif e[1] == "free":
				fout.write(str((e[0] - time_min) * 1000.0) + "\t" + str(offset + y) + "\n")
				fout.write(str((e[0] - time_min) * 1000.0) + "\t" + str(offset) + "\n")
				y = 0
		fout.write(str((time_max - time_min) * 1000.0) + "\t" + str(offset) + "\n")

		fout.close()
		i += 1

	i = 0
	for k,v in plist.iteritems():
		fout = open("memusage" + str(i) + ".vis", "w")
		#t = time_min
		offset = v[0]
		y = 0

		fout.write("0\t" + str(offset) + "\n")
		for e in v[1:]:
			if e[1] == "malloc":
				y = e[2]
			elif e[1] == "free":
				y = 0
			elif e[1] == "copy_begin" or e[1] == "kernel_begin":
				fout.write(str((e[0] - time_min) * 1000.0) + "\t" + str(offset) + "\n")
				fout.write(str((e[0] - time_min) * 1000.0) + "\t" + str(offset + y) + "\n")
			elif e[1] == "copy_end" or e[1] == "kernel_end":
				fout.write(str((e[0] - time_min) * 1000.0) + "\t" + str(offset + y) + "\n")
				fout.write(str((e[0] - time_min) * 1000.0) + "\t" + str(offset) + "\n")
		fout.write(str((time_max - time_min) * 1000.0) + "\t" + str(offset) + "\n")

		fout.close()
		i += 1

	# generate gnuplot script file
	fout = open("pot.script", "w")

	fout.write("set xlabel \"Time (ms)\"\n")
	fout.write("set terminal png size 1200, 800\n")
	fout.write("set output \"pot1.png\"\n\n")

	i -= 1
	if i >= 0:
		fout.write("plot \"memspace" + str(i) + ".vis\" using 1:2 title \"space " + str(i) + "\" with lines lw 2,\\\n")
		fout.write("\t\"memusage" + str(i) + ".vis\" using 1:2 title \"usage " + str(i) + "\" with lines lw 1")

	i -= 1
	while i >= 0:
		fout.write(",\\\n\t\"memspace" + str(i) + ".vis\" using 1:2 title \"space " + str(i) + "\" with lines lw 2,\\\n")
		fout.write("\t\"memusage" + str(i) + ".vis\" using 1:2 title \"usage " + str(i) + "\" with lines lw 1")
		i -= 1

	fout.close()

	# execute the script
	the_cmd = "gnuplot pot.script"
	call(the_cmd, shell=True)


def generate_plot_2(time_min, time_max, plist):
	# compute vertical offsets, save them to v[0]
	# leave an interval of 10 in space
	offset_delta = 1000
	offset_curr = 100

	for k,v in plist.iteritems():
		v[0] = offset_curr
		offset_curr += offset_delta + 100

	# generate plot data files
	i = 0
	for k,v in plist.iteritems():
		fout = open("memspace" + str(i) + ".vis", "w")
		t = time_min
		offset = v[0]

		fout.write("0\t" + str(offset) + "\n")
		for e in v[1:]:
			if e[1] == "malloc":
				fout.write(str((e[0] - time_min) * 1000.0) + "\t" + str(offset) + "\n")
				fout.write(str((e[0] - time_min) * 1000.0) + "\t" + str(offset + offset_delta) + "\n")
			elif e[1] == "free":
				fout.write(str((e[0] - time_min) * 1000.0) + "\t" + str(offset + offset_delta) + "\n")
				fout.write(str((e[0] - time_min) * 1000.0) + "\t" + str(offset) + "\n")
		fout.write(str((time_max - time_min) * 1000.0) + "\t" + str(offset) + "\n")

		fout.close()
		i += 1

	i = 0
	for k,v in plist.iteritems():
		fout = open("memusage" + str(i) + ".vis", "w")
		t = time_min
		offset = v[0]

		fout.write("0\t" + str(offset) + "\n")
		for e in v[1:]:
			if e[1] == "copy_begin" or e[1] == "kernel_begin":
				fout.write(str((e[0] - time_min) * 1000.0) + "\t" + str(offset) + "\n")
				fout.write(str((e[0] - time_min) * 1000.0) + "\t" + str(offset + offset_delta) + "\n")
			elif e[1] == "copy_end" or e[1] == "kernel_end":
				fout.write(str((e[0] - time_min) * 1000.0) + "\t" + str(offset + offset_delta) + "\n")
				fout.write(str((e[0] - time_min) * 1000.0) + "\t" + str(offset) + "\n")
		fout.write(str((time_max - time_min) * 1000.0) + "\t" + str(offset) + "\n")

		fout.close()
		i += 1

	# generate gnuplot script file
	fout = open("pot.script", "w")

	fout.write("set xlabel \"Time (ms)\"\n")
	fout.write("set terminal png size 1200, 500\n")
	fout.write("set output \"pot2.png\"\n\n")

	i -= 1
	if i >= 0:
		fout.write("plot \"memspace" + str(i) + ".vis\" using 1:2 title \"space " + str(i) + "\" with lines lw 2,\\\n")
		fout.write("\t\"memusage" + str(i) + ".vis\" using 1:2 title \"usage " + str(i) + "\" with lines lw 1")

	i -= 1
	while i >= 0:
		fout.write(",\\\n\t\"memspace" + str(i) + ".vis\" using 1:2 title \"space " + str(i) + "\" with lines lw 2,\\\n")
		fout.write("\t\"memusage" + str(i) + ".vis\" using 1:2 title \"usage " + str(i) + "\" with lines lw 1")
		i -= 1

	fout.close()

	# execute the script
	the_cmd = "gnuplot pot.script"
	call(the_cmd, shell=True)


# main
if len(sys.argv) < 2:
	print "USAGE: " + sys.argv[0] + " log-file"
	sys.exit(1)

log_events = get_log_events(sys.argv[1])

# potential list; format: {id: [addr, events]}
# each event has the format:  [time, type, parameters]
# type is defined as: "malloc", "copy", "kernel", "free"
plist = {}

imax = len(log_events)
i = 0
while i < imax:
	#print i, imax
	if log_events[i][2] == "cudaMalloc":
		pid = log_events[i][4]
		i += 1
		addr = int(log_events[i][3], 16)
		size = int(log_events[i][4])
		t = float(log_events[i][0])

		if pid in plist:
			plist[pid][0] = addr
		else:
			plist[pid] = []
			plist[pid].append(addr)	# the first element is addr
		plist[pid].append([t, "malloc", size])
		i += 1

	elif log_events[i][2] == "cudaMemcpy":
		t1 = float(log_events[i][0])
		i += 1

		t2 = float(log_events[i][0])
		addr = 0
		if log_events[i][6] == "1":		# HtoD
			addr = int(log_events[i][3], 16)
		elif log_events[i][6] == "2":	# DtoH
			addr = int(log_events[i][4], 16)
		else:							# DtoD
			addr1 = int(log_events[i][3], 16)
			pid1 = get_pid(addr1, plist)
			if pid1 == "":
				print "Error 1: failed to find pid"
				#sys.exit(1)
				i += 1
				continue

			addr2 = int(log_events[i][4], 16)
			pid2 = get_pid(addr2, plist)
			if pid2 == "":
				print "Error 1: failed to find pid"
				#sys.exit(1)
				i += 1
				continue

			plist[pid1].append([t1, "copy_begin"])
			plist[pid1].append([t2, "copy_end"])
			plist[pid2].append([t1, "copy_begin"])
			plist[pid2].append([t2, "copy_end"])
			i += 1
			continue

		pid = get_pid(addr, plist)
		if pid == "":
			print "Error 1: failed to find pid"
			i += 1
			continue
			#sys.exit(1)

		plist[pid].append([t1, "copy_begin"])
		plist[pid].append([t2, "copy_end"])
		i += 1

	elif log_events[i][2] == "cudaConfigureCall" and log_events[i][1] == "intercepted":
		pids_affected = []
		i += 1
		while log_events[i][2] <> "cudaLaunch":
			if log_events[i][2] == "cudaSetupArgument(" and log_events[i][3] == "8":
				addr = 0
				for j in range(0, 8):
					addr += int(log_events[i][4+j], 16) * math.pow(256, j)	# little endian
				pid = get_pid(addr, plist)
				if pid <> "":
					pids_affected.append(pid)
			i += 1

		i += 1
		t1 = float(log_events[i][0])
		i += 2
		if log_events[i][2] <> "cudaThreadSynchronize":
			print "Error: no cuda_sync after kernel launch"
			sys.exit(1)
		t2 = float(log_events[i][0])

		for pid in pids_affected:
			plist[pid].append([t1, "kernel_begin"])
			plist[pid].append([t2, "kernel_end"])
		i += 1

	elif log_events[i][2] == "cudaFree(" and log_events[i][1] == "intercepted":
		t = float(log_events[i][0])
		addr = int(log_events[i][3], 16)

		pid = get_pid(addr, plist)
		if pid == "":
			print "Error: failed to find pid at cudaFree"
			#sys.exit(1)
			i += 1
			continue

		plist[pid].append([t, "free"])
		plist[pid][0] = 0
		i += 1

	else:
		i += 1


# compute time_min and time_max
time_min = float("inf")
time_max = float("-inf")

for k,v in plist.iteritems():
	if v[1][0] < time_min:
		time_min = v[1][0]
	if v[-1][0] > time_max:
		time_max = v[-1][0]

time_min -= 0.001
time_max += 0.001

generate_plot_1(time_min, time_max, plist)
generate_plot_2(time_min, time_max, plist)
