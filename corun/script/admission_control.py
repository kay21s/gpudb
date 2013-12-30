#!/usr/bin/python
import os
import sys
from random import randint
from time import time, sleep
import numpy
import threading

os.chdir("../../")
rootpath = os.getcwd()

querypath = rootpath + r'/corun/query_progs/'
datapath = rootpath + r'/'
curpath = rootpath + r'/corun/script/'

LOAD_GMM = 1
if LOAD_GMM:
	preloadlib=r'LD_PRELOAD='+rootpath+r'/gdb/src/libgmm.so '
else:
	preloadlib=''

poisson_avg = 20
lock = threading.Lock()
rq = []
querys = ['q1_1', 'q1_2', 'q1_3', 'q2_1', 'q2_2', 'q2_3',
		'q3_1', 'q3_2', 'q3_3', 'q3_4', 'q4_1', 'q4_2', 'q4_3']
mem_usage = {
		'q1_1':649009240,
		'q1_2':649009364,
		'q1_3':649009268,
		'q2_1':654157918,
		'q2_2':653320344,
		'q2_3':653521976,
		'q3_1':654927640,
		'q3_2':653599816,
		'q3_3':653263088,
		'q3_4':653263088,
		'q4_1':653314496,
		'q4_2':654922192,
		'q4_3':653228840
		}
data_sizes = ['7','14']

total_mem = 0
permitted_total_mem = 1600000000
#permitted_total_mem = 1400000000
#permitted_total_mem = 1500000000

gdb = 1
fix2 = 0

def get_query_random():
	return querys[randint(0, len(querys)-1)]

def get_datasize_random():
	return data_sizes[randint(0, len(data_sizes)-1)]

def preload_mem():
	for q in querys:
		for d in data_sizes:
			cmd = preloadlib + querypath + q + r' --datadir ' + datapath + 'data_' + d
			print cmd
			os.system(cmd)

def get_space(query, datasize):
	#print '---'
	#print datasize
	#print '---'
	if datasize == '14':
		return mem_usage[query]*3/2
	elif datasize == '7':
		return mem_usage[query]*3/4
	else:
		print 'error for datasize'
		sys.exit(0)

def wait_random_interval():
	s = numpy.random.poisson(lam = poisson_avg, size = None)
	#print '---------------sleeping', s, 'ms'
	sleep(s/1000)

def new_query(query, datasize):
	global total_mem
	cmd = preloadlib + querypath + query + r' --datadir ' + datapath + 'data_' + datasize
	#print 'start', query, datasize
	#cmd = preloadlib + querypath + query + r' --datadir ' + datapath
	os.system(cmd)
	with lock:
		for i in range(0, len(rq)):
			if [query, datasize] == rq[i]:
				del rq[i]
				total_mem -= get_space(query, datasize)
				break
	#print 'end', query, datasize

def admin_ctl(query, datasize):
	global total_mem
	if gdb:
		while total_mem + get_space(query, datasize) > permitted_total_mem:
			continue
	elif fix2:
		while len(rq) >= 2:
			continue

if __name__ == "__main__":

	global total_mem

	preload_mem()

	set = []
	for line in open(curpath+"random_combo", "r").readlines():
		set.append(line.strip().split(' '))

	start_time = time()

	for elem in set:
		[query, datasize] = elem

		#wait_random_interval()
		admin_ctl(query, datasize)

		with lock:
			rq.append(elem)
			total_mem += get_space(query, datasize)

		t = threading.Thread(target = new_query, args = (query, datasize))
		t.start()

	t.join()
	end_time = time()
	print 'total time = ', end_time - start_time
