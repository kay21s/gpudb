#!/usr/bin/python
import os
from random import randint

querys = ['q1_1', 'q1_2', 'q1_3', 'q2_1', 'q2_2', 'q2_3',
		'q3_1', 'q3_2', 'q3_3', 'q3_4', 'q4_1', 'q4_2', 'q4_3']
data_sizes = ['7','14']

def get_query_random():
	return querys[randint(0, len(querys)-1)]

def get_datasize_random():
	return data_sizes[randint(0, len(data_sizes)-1)]

if __name__ == "__main__":
	for i in range(0, 500):
		query = get_query_random()
		datasize = get_datasize_random()
		print query, datasize
