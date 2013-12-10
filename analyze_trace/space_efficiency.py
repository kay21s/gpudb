#!/usr/bin/python
import os
import sys

def calc_area(file):
	i = 0
	area = 0
	for line in open(file, 'r').readlines():
		time, mem = line.strip().split('\t')
		# jump head
		if time == '#time':
			continue
		if i == 0:
			start_time = float(time)
			record_mem = float(mem)
			i = 1
		elif i == 1:
			end_time = float(time)
			if record_mem != float(mem):
				print 'error!!'
				sys.exit()
			i = 0
			if record_mem == 0.0:
				continue
			area += record_mem * (end_time - start_time)
	return area

def get_refs(ref_file):
	refs = []
	for line in open(ref_file, 'r').readlines():
		one_ref = line.strip().split(' ')
		if len(one_ref) < 4:
			continue
		if one_ref[2] == '----':
			refs.append(float(one_ref[3]))
	return refs

def calc_area_ref(file, refs):
	i = 0
	ref_num = 0
	area = 0
	for line in open(file, 'r').readlines():
		time, mem = line.strip().split('\t')
		# jump head
		if time == '#time':
			continue
		if i == 0:
			start_time = float(time)
			record_mem = float(mem)
			i = 1
		elif i == 1:
			end_time = float(time)
			if record_mem != float(mem):
				print 'error!!'
				sys.exit()
			i = 0
			if record_mem == 0.0:
				continue
			area += refs[ref_num] * (end_time - start_time)
			ref_num += 1
	return area


if __name__ == "__main__":
	querys = ['q1_1', 'q1_2', 'q1_3', 'q2_1', 'q2_2', 'q2_2',
			'q3_1', 'q3_2', 'q3_3', 'q3_4', 'q4_1', 'q4_2', 'q4_3']

	os.chdir("../")
	rootpath = os.getcwd()

	mem_anal_dir = rootpath+ '/analyze_trace/mem_analysis/vis/'
	ref_dir = rootpath + '/analyze_trace/kernel_mem_usage/'


	for query in querys:
		copy_area = calc_area(mem_anal_dir + query + '_memcpy.vis')
		space_area = calc_area(mem_anal_dir + query + '_memspace.vis')
		refs = get_refs(ref_dir + query + '.solo')
		usage_area = calc_area_ref(mem_anal_dir + query + '_memusage.vis', refs)
		print query, '       ', "%.2f" % ((usage_area+copy_area) / space_area)
