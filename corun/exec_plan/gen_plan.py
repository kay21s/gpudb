#!/usr/bin/python
import os,sys

n_combo = 4

querys = ['q1_1', 'q1_2', 'q2_1', 'q2_2', 'q3_1', 'q4_1']
if n_combo == 3:
	for i in range(0, len(querys)-2):
		for j in range(i+1, len(querys)-1):
			for k in range(j+1, len(querys)):
				combo = querys[i]+' '+querys[j]+' '+querys[k]
				print combo
elif n_combo == 4:
	for i in range(0, len(querys)-3):
		for j in range(i+1, len(querys)-2):
			for k in range(j+1, len(querys)-1):
				for m in range(k+1, len(querys)):
					combo = querys[i]+' '+querys[j]+' '+querys[k]+' '+querys[m]
					print combo
