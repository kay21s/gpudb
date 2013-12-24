#!/usr/bin/python
import os,sys

os.chdir("../../")
rootpath = os.getcwd()

co_dir = rootpath + r'/corun/output/'
solo_dir = rootpath + r'/trace/file/'
cur_dir = rootpath + r'/corun/script/'

solo_dict = {}
for file in os.listdir(solo_dir):
	if file[-4:] == 'solo':
		avg = 0.0
		num = 0.0
		time_list = []
		for line in open(solo_dir+file, "r").readlines():
			if line[:12] == 'Total Time: ':
				if num == 0:
					num += 1
					continue
				time = float(line[12:])
				time_list.append(round(time,2))
				avg += time
				num += 1
		avg = round(avg/(num-1), 2)
		solo_dict[file[:-5]] = avg

#print solo_dict

#find the max query numbers
max_len = 0
for file in os.listdir(co_dir):
	length = len(file.split('.'))
	max_len = max(max_len, length)

#if max_len <> 2:
#	print "query corun number is supposed to be 2"
#	sys.exit(0)

# this is used to record the slow down of each query in the corun
# for example, when max_len = 2, it means at most 2 queries corun together
# then speedup0 is for the first query in corun, and speedup1 is for the
# second query in corun. This is for drawing the curve in gnuplot
# individual is the statistic for each query, ind_file is the file handler
individual = [[]]
ind_file = []
for i in range(0,max_len):
	os.chdir(cur_dir)
	individual.append([])
	fout = open("speedup"+str(i), "w+")
	ind_file.append(fout)

# 2-6 means we get the 2,3,4,5,6 corun performance as the valid ones for statistics
# For fully overlap
stat_s = 2
stat_e = 6
stat_num = stat_e - stat_s + 1

# we only print 3 times
print_runnum = 3

# two statistics, 0 or 1 to change
if len(sys.argv) == 2 and (sys.argv[1] == '0' or sys.argv[1] == '1'):
	statistic_1 = int(sys.argv[1])
else:
	statistic_1 = 1

if statistic_1:
	print "Left\tRight\tLeft Combo Runtime(ms)\tRight Combo Runtime(ms)\t",
	for i in range(1, print_runnum+1):
		print 'Left ', i, '\t',
	for i in range(1, print_runnum+1):
		print 'Right ', i, '\t',
	print ''
else:
	print "Left\tRight\tLeft Combo Runtime(ms)\tRight Combo Runtime(ms)\tCombo Speedup\tLeft Slowdown\tRight Slowdown"


files = os.listdir(co_dir)
files.sort()
for file in files:
#	print file, ' ',
	err = 0
	os.chdir(co_dir+file)


	q_static = {}
	querys=[]
	for q in file.split('.'):
		if q in querys:
			querys.append(q+'_'+str(q_static[q]))
			q_static[q] += 1
		else:
			querys.append(q)
			q_static[q] = 1

	'''
	err = open('error').read()
	if len(err) <> 0:
		for i in range(0, max_len):
			print querys[i][0:4].replace('_', ''),'\t',
		print ''
		continue
	'''

	# 1) speedup = (1/co_q1 + 1/co_q2)/(1/(solo_q1 + solo_q2)) = (1/co_q1+1/co_q2) * (solo_q1+solo_q2)
	#speedup_right = 0.0 # (1/co_q1 + 1/co_q2)
	#speedup_left = 0.0 # (solo_q1 + solo_q2)

	# 2) Now We calculate as :speedup = solo_q1/co_q1 + solo_q2/co_q2

	# init for each corun
	query_name = [] #record the query names
	avg_time = []  # record query average execution time
	individual = [[]] #record each execution time for statistic
	for i in range(0,max_len):
		individual.append([])

	speedup = 0.0
	query_no = 0
	for query in querys:
		time = 0.0
		num = 0
		lines = open(query, "r").readlines()
		if lines is None:
			print 'error', query, file
			sys.exit(0)
		# we only count for lines 2,3,4,5,6  for fully overlap
		for line in lines:
			if line[:12] == 'Total Time: ':
				num += 1
				if num < stat_s:
					continue # we donot count the first
				if num > stat_e:
					break
				time += float(line[12:])
				individual[query_no].append(round(float(line[12:]), 2))
		if num == 0:
			print 'deadlock', query, file
			sys.exit(0)
		if num != stat_e + 1:
			#print 'end before finding the required number of queries', num, query, file
			err = 1
			break
		#speedup_right += 1/(time/num)
		#speedup_left += solo_dict[query]
		query = query[0:4] # discard the last '_1' in like q3_1_1, which happens in same query corun
		query_name.append(query)
		avg_time.append(round(time/stat_num, 2))
		speedup += solo_dict[query]/(time/stat_num)
		query_no += 1

	if err == 1:
		for i in range(0, max_len):
			print file.split('.')[i].replace('_', ''),'\t',
		print ''
		continue

	if statistic_1:
		for i in range(0, len(query_name)):
			print query_name[i].replace('_', ''),'\t',
		for i in range(0, len(query_name)):
			print avg_time[i],'\t',
		for j in range(0, max_len):
			for i in range(0, print_runnum):
				print individual[j][i],'\t',
		print ''

	else:
		for i in range(0, len(query_name)):
			print query_name[i].replace('_', ''),'\t',
		for i in range(0, len(query_name)):
			print avg_time[i],'\t',
		print round(speedup,2),'\t',
		for i in range(0, len(query_name)):
			print round(avg_time[i]/solo_dict[query_name[i]],2),'\t',
		print ''

