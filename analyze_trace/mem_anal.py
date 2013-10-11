#!/usr/bin/python

import sys
import os
import string

# analysis parameters
input_dir = "/home/kai/projects/trace/file"
output_dir = "/home/kai/projects/trace/anal"

solo_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
solos = []
for solo in solo_files:
	words = solo.split(".")
	solos.append(words[0])
solos.sort()

# the output buffer
outbuf = list([] for i in range(0,len(solos)))

# for each benchmark, parse solo run trace and put the result in its outbuf
for ibench in range(0, len(solos)):
	diskTime = 0.0
	infile = open(os.path.join(input_dir, solos[ibench] + ".solo"), 'r')
	lines = infile.readlines()
	infile.close()

	ifirst = -1
	for iline in range(0, len(lines)):
		words = lines[iline].split()
		if len(words) == 0 or words[2] <> "BEGIN":
			continue
		else:
			ifirst = iline
			break
	if ifirst == -1:
		print "Parse error: no begins found"
		sys.exit(1)

	startTime = float(lines[ifirst].split()[1][8:]) * 1000.0
	# now parse the traces recorded during the first solo run
	for iline in range(ifirst+1, len(lines)):
		words = lines[iline].split()
		if len(words) == 0 or words[0] <> "[gvm]" or words[2] <> "MEMSIZE":
			continue
		tg = [float(words[3])/1000000.0, float(words[4]) * 1000.0]
		outbuf[ibench].append(tg)


# write results to output file
for ibench in range(0, len(solos)):
	# normalize time
	tmin = outbuf[ibench][0][1]
	for i in range(0, len(outbuf[ibench])):
		outbuf[ibench][i][1] -= tmin

	# write normalized results to file
	outfile = open(os.path.join(output_dir, solos[ibench] + ".trace"), "w")
	outfile.write("#Time       Memory Size\n")
	for event in outbuf[ibench]:
		outfile.write(str(event[1]) + "\t" + str(event[0]) + "\n")
	outfile.close()

