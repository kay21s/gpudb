#!/usr/bin/python
import sys
import os

if __name__ == '__main__':
	rep = int(sys.argv[1])
	cmd = ''
	for i in range(2, len(sys.argv)):
		cmd = cmd + ' ' + sys.argv[i]

	print cmd
	for i in range(0, rep):
		os.system(cmd)

