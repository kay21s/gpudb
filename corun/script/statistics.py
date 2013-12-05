#!/usr/bin/python
import os,sys

os.chdir("../../")
rootpath = os.getcwd()

co_dir = rootpath + r'/corun/output/'
cur_dir = rootpath + r'/corun/script/'

stats_one = ['bytes_mem_alloc', 'bytes_mem_peak', 'bytes_mem_freed',
		'bytes_htod', 'bytes_htod_cow', 'bytes_dtoh', 'bytes_dtod',
		'bytes_memset']
stats_avg = ['bytes_evict_needed', 'bytes_evict_space', 'bytes_evict_data',
		'time_evict', 'time_attach', 'time_load', 'time_kernel',
		'time_dma_htod', 'time_dma_htod_cow', 'time_dma_dtoh',
		'time_total']

if len(sys.argv) == 2:
	print_stats_one = 1
	print_stats_avg = 0
else:
	print_stats_one = 0
	print_stats_avg = 1


print 'query\t',
if print_stats_one:
	for stat in stats_one:
		print stat, '\t',
if print_stats_avg:
	for stat in stats_avg:
		print stat, '\t',
print ''


#find the max query numbers
max_len = 0
for file in os.listdir(co_dir):
	length = len(file.split('.'))
	max_len = max(max_len, length)

if max_len <> 2:
	print "query corun number is supposed to be 2"
	sys.exit(0)

# 2-6 means we get the 2,3,4,5,6 corun performance as the valid ones for statistics
# For fully overlap
stat_s = 2
stat_e = 6
stat_num = stat_e - stat_s + 1

printed_queries = []

files = os.listdir(co_dir)
files.sort()
for file in files:
	#if print_stats_avg:
	#	print '--------------'
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

	err = open('error').read()
	if len(err) <> 0:
		#if print_stats_avg:
		#	print querys[0][0:4],'\n',querys[1][0:4]
		continue

	query_name = [] #record the query names
	avg_time = []  # record query average execution time
	individual = [[]] #record each execution time for statistic
	for i in range(0,max_len):
		individual.append([])


	for query in querys:
		lines = open(query, "r").readlines()
		if lines is None:
			print 'error', query, file
			sys.exit(0)

		num = 1
		dict_one = {}
		dict_avg = {}
		for stat in stats_avg:
			dict_avg[stat] = 0.0


		# we only count for lines 2,3,4,5,6  for fully overlap
		for line in lines:
			line = line.strip()
			for stat in stats_one:
				# ' ' + stat + ' ' avoid htod in  htod_cow
				if ' '+stat+' ' in line:
					if stat in dict_one:
						if dict_one[stat] != line.split(' ')[-1]:
							print 'error in stat', stat, query, file
							print dict_one[stat], '!=', line.split(' ')[-1]
							#print line
							sys.exit(0)
					else:
						dict_one[stat] = line.split(' ')[-1]

			for stat in stats_avg:
				if ' '+stat+' ' in line:
					if stat in dict_avg:
						if stat == 'time_total':
							#use this as mark
							num += 1
						if num < stat_s:
							continue
						elif num > stat_e:
							break
						else:
							dict_avg[stat] += float(line.split(' ')[-1])
					else:
						print 'error in stat', stat, query, file
						sys.exit(0)

		if num == 0:
			print 'deadlock', query, file
			sys.exit(0)
		if num < stat_e + 1:
			print 'end before finding the required number of queries', num, query, file
			sys.exit(0)

		query = query[0:4] # discard the last '_1' in like q3_1_1, which happens in same query corun

		if print_stats_one:
			if query not in printed_queries:
				printed_queries.append(query)
				print query, '\t',
				for stat in stats_one:
					print dict_one[stat], '\t',
				print ''
		if print_stats_avg:
			print query, '\t',
			for stat in stats_avg:
				print dict_avg[stat]/stat_num, '\t',
			print ''



