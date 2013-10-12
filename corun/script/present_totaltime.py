#!/usr/bin/python
import os

co_dir = "/home/kai/projects/corun/output/"
solo_dir = "/home/kai/projects/trace/file/"
cur_dir = "/home/kai/projects/corun/script/"

solo_dict = {}
for file in os.listdir(solo_dir):
	if file[-4:] == 'solo':
		for line in open(solo_dir+file, "r").readlines():
			if line[:12] == 'Total Time: ':
				time = float(line[12:])
				solo_dict[file[:-5]] = time
#print solo_dict

result = []
for file in os.listdir(co_dir):
#	print file, ' ',
	os.chdir(co_dir+file)
	querys = file.split('.')

	# 1) speedup = (1/co_q1 + 1/co_q2)/(1/(solo_q1 + solo_q2)) = (1/co_q1+1/co_q2) * (solo_q1+solo_q2)
	#speedup_right = 0.0 # (1/co_q1 + 1/co_q2)
	#speedup_left = 0.0 # (solo_q1 + solo_q2)

	# 2) Now We calculate as :speedup = solo_q1/co_q1 + solo_q2/co_q2

	speedup = 0.0
	for query in querys:
		time = 0.0
		num = 0
		lines = open(query, "r").readlines()
		if lines is None:
			print 'error',
			break
		# we only count for lines 2,3,4  for fully overlap
		for line in lines:
			if line[:12] == 'Total Time: ':
				num += 1
				if num == 1:
					continue # we donot count the first
				if num > 4:
					break
				time += float(line[12:])
		if num == 0:
			print 'deadlock',
			break
		if num != 5:
			print 'supposed to be 5 times'
			break
		#speedup_right += 1/(time/num)
		#speedup_left += solo_dict[query]
		speedup += solo_dict[query]/(time/3)
	if num > 0:
		#speedup = speedup_right * speedup_left
		#print speedup
		result.append(file + ' ' + str(speedup))
	else:
		print ''

result.sort()
os.chdir(cur_dir)
result_file = open("result_corun", "w+")
for item in result:
	result_file.write("%s\n" % item)
result_file.close()

