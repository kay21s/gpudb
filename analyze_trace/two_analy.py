#!/usr/bin/python

import sys
import os
import string

# analysis parameters
input_dir = "../trace/file"
output_dir = "../trace/anal"

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
	infile = open(os.path.join(input_dir, solos[ibench] + ".solo"), 'r')
	lines = infile.readlines()
	infile.close()

	ifirst = -1
	isecond = -1
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

	for iline in range(ifirst+1, len(lines)):
		words = lines[iline].split()
		if len(words) == 0 or words[2] <> "BEGIN":
			continue
		else:
			isecond = iline
			break
	if isecond == -1:
		print "Parse error: no begins found"
		continue

	print isecond, len(lines), solos[ibench]
	startTime = float(lines[isecond].split()[1]) * 1000.0
	# now parse the traces recorded during the first solo run
	for iline in range(isecond+1, len(lines)):
		words = lines[iline].split()
		if len(words) == 0 or words[0] <> "[gvm]" or words[2] <> "intercepted":
			continue

		tg = [5, 0.0, float(words[1]) * 1000.0]
		if words[3] == "cudaMemcpy(" and words[7] == "1":	# H2D
			tg[0] = 0
			curWord = words[3][0:-1]
		elif words[3] == "cudaMemcpy(" and words[7] == "2":	# D2H
			tg[0] = 1
			curWord = words[3][0:-1]
		elif words[3] == "cudaLaunch":	# cudaLaunch
			tg[0] = 2
			curWord = words[3]
		elif words[3] == "diskIO":	# diskIO
			tg[0] = 3
			curWord = words[3]
		elif words[3] == "cudaMalloc(":	# cudaMalloc
			tg[0] = 4
			curWord = words[3][0:-1]

		if tg[0] <> 5:
			iprev = iline - 1
			preWord = lines[iprev].split()
			while len(preWord) == 0 or preWord[0] <> "[gvm]" or preWord[3] <> curWord:
				iprev -= 1
				preWord = lines[iprev].split()
			if preWord[2] <> "intercepting":
				print "ERROR!!!"
				sys.exit(1)

			tg[1] = float(preWord[1]) * 1000.0
			tc = [5, startTime, tg[1]]
			outbuf[ibench].append(tc)

			outbuf[ibench].append(tg)
			startTime = tg[2]


# write results to output file
for ibench in range(0, len(solos)):
	# normalize time
	tmin = outbuf[ibench][0][1]
	for i in range(0, len(outbuf[ibench])):
		#print outbuf[ibench][i][1]
		#print outbuf[ibench][i][2]
		outbuf[ibench][i][1] -= tmin
		outbuf[ibench][i][2] -= tmin

	# write normalized results to file
	outfile = open(os.path.join(output_dir, solos[ibench] + ".trace"), "w")
	outfile.write("Event\tStart Time\tEnd Time\n")
	for event in outbuf[ibench]:
		outfile.write(str(event[0]) + "\t" + str(event[1]) + "\t" + str(event[2]) + "\n")
	outfile.close()

