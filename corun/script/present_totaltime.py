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
				time = float(line[12:])
				time_list.append(round(time,2))
				avg += time
				num += 1
		avg = round(avg/num, 2)
		solo_dict[file[:-5]] = avg

#print solo_dict

#find the max query numbers
max_len = 0
for file in os.listdir(co_dir):
	length = len(file.split('.'))
	max_len = max(max_len, length)

if max_len <> 2:
	print "query corun number is supposed to be 2"
	sys.exit(0)

# this is used to record the slow down of each query in the corun
# for example, when max_len = 2, it means at most 2 queries corun together
# then speedup0 is for the first query in corun, and speedup1 is for the
# second query in corun. This is for drawing the curve in gnuplot
individual = [[]]
ind_file = []
for i in range(0,max_len):
	os.chdir(cur_dir)
	individual.append([])
	fout = open("speedup"+str(i), "w+")
	ind_file.append(fout)

# 2-4 means we get the 2,3,4 Total time to calculate
# For fully overlap
stat_s = 2
stat_e = 4
stat_num = stat_e - stat_s + 1


result = []
for file in os.listdir(co_dir):
#	print file, ' ',
	os.chdir(co_dir+file)
	err = open('error').read()
	if len(err) <> 0:
		continue


	q_static = {}
	querys=[]
	for q in file.split('.'):
		if q in querys:
			querys.append(q+'_'+str(q_static[q]))
			q_static[q] += 1
		else:
			querys.append(q)
			q_static[q] = 1

	# 1) speedup = (1/co_q1 + 1/co_q2)/(1/(solo_q1 + solo_q2)) = (1/co_q1+1/co_q2) * (solo_q1+solo_q2)
	#speedup_right = 0.0 # (1/co_q1 + 1/co_q2)
	#speedup_left = 0.0 # (solo_q1 + solo_q2)

	# 2) Now We calculate as :speedup = solo_q1/co_q1 + solo_q2/co_q2

	speedup = 0.0
	query_no = 0
	for query in querys:
		time = 0.0
		num = 0
		lines = open(query, "r").readlines()
		if lines is None:
			print 'error', query, file
			sys.exit(0)
		# we only count for lines 2,3,4  for fully overlap
		for line in lines:
			if line[:12] == 'Total Time: ':
				num += 1
				if num < stat_s:
					continue # we donot count the first
				if num > stat_e:
					break
				time += float(line[12:])
		if num == 0:
			print 'deadlock', query, file
			sys.exit(0)
		if num != 5:
			print 'supposed to be 5 times', num, query, file
			sys.exit(0)
		#speedup_right += 1/(time/num)
		#speedup_left += solo_dict[query]
		query = query[0:4] # discard the last '_1' in like q3_1_1, which happens in same query corun
		speedup += solo_dict[query]/(time/stat_num)
		individual[query_no].append(file + ' ' + str(solo_dict[query]/(time/stat_num)))
		query_no += 1

	#speedup = speedup_right * speedup_left
	#print speedup
	result.append(file + ' ' + str(speedup))

result.sort()
total_corun = len(result)
os.chdir(cur_dir)
result_file = open("result_corun", "w+")
for item in result:
	result_file.write("%s\n" % item)
result_file.close()

for query_no in range(0, max_len):
	individual[query_no].sort()
	for item in individual[query_no]:
		ind_file[query_no].write("%s\n" % item)

plot = open("plot_corun.plot", "w+")
plot.write("set xlabel \"Co-run queries\"\n")
plot.write("set ylabel \"Speedup\"\n")
plot.write("set terminal png size 1200, 400\n")
plot.write("set output \"speedup.png\"\n")
plot.write("set xtics nomirror rotate by 30 font \",8\"\n")
plot.write("set yrange [0:2]\n")
plot.write("set xrange [-1:"+str(total_corun+1)+"]\n")
plot.write("plot \"speedup0\" using 2:xticlabels(1) title \"speedup of qx in qx.qy\" with lines, \"speedup1\" using 2:xticlabels(1) title \"speedup of qy in qx.qy\" with lines, \"result_corun\" using 2:xticlabels(1) title \"corun speedup\" with boxes\n")
plot.close()

os.system("gnuplot plot_corun.plot")
