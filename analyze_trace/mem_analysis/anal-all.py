#!/usr/bin/python

import sys
import os
import string
import math
from subprocess import call

bench_suite_dir = "/home/kaibo/projects/gpu-vm/benchmarks/rodinia_2.0.1/cuda"
output_dir = "results"

# for each benchmark under bench_suite_dir, analyze its interception log
benchmarks = [f for f in os.listdir(bench_suite_dir) if os.path.isdir(os.path.join(bench_suite_dir, f)) and f <> "not-working"]

for bench in benchmarks:
	input_dir = os.path.join(bench_suite_dir, bench)

	# overal analysis
	the_cmd = "./anal.py " + os.path.join(input_dir, "gvm_intercept.log")
	call(the_cmd, shell=True)

	the_cmd = "make"
	call(the_cmd, shell=True)

	the_cmd = "mkdir " + os.path.join(output_dir, bench)
	call(the_cmd, shell=True)

	the_cmd = "cp plot.png " + os.path.join(output_dir, bench)
	call(the_cmd, shell=True)

	# compute memory utilization
	the_cmd = "./util.py memspace.txt memusage.txt memcpy.txt > util.txt"
	call(the_cmd, shell=True)

	the_cmd = "mv util.txt " + os.path.join(output_dir, bench)
	call(the_cmd, shell=True)

	# potential analysis
	the_cmd = "./potential.py " + os.path.join(input_dir, "gvm_intercept.log")
	call(the_cmd, shell=True)

	the_cmd = "cp pot1.png pot2.png " + os.path.join(output_dir, bench)
	call(the_cmd, shell=True)

	# clean
	the_cmd = "make clean"
	call(the_cmd, shell=True)
