#!/usr/bin/python
import os
if not os.path.exists('result'):
	os.mkdir('result')
if not os.path.exists('immediate'):
	os.mkdir('immediate')
if not os.path.exists('vis'):
	os.mkdir('vis')
if not os.path.exists('png'):
	os.mkdir('png')


cmd = r'./anal.py'
os.system(cmd)
for file in os.listdir("./file/"):
	memspace = r'./immediate/' + file[:-5] + r'_memspace.txt'
	memusage = r'./immediate/' + file[:-5] + r'_memusage.txt'
	memcpy = r'./immediate/' + file[:-5] + r'_memcpy.txt'
	cmd = r'./visualize.py ' + memspace + ' ' + memusage + ' ' +  memcpy
	os.system(cmd)
	ps = open('gnuplot.script', "w")
	ps.write(r'set xlabel "Time (ms)"' + '\n')
	ps.write(r'set ylabel "Bytes"' + '\n')
	ps.write(r'set terminal png size 1200, 500' + '\n')
	output = r'set output "./png/' + file[:-5] + r'.png' + '\n'
	ps.write(output)
	cmd = r'plot "./vis/' + file[:-5] + r'_memusage.vis" ' + r'using 1:2 title "usage" with lines lw 2,' + r' "./vis/' + file[:-5] + r'_memspace.vis" ' + r'using 1:2 title "space" with lines,' + r' "./vis/' + file[:-5] + r'_memcpy.vis" ' + r'using 1:2 title "copy" with lines' + '\n'
	ps.write(cmd)
	ps.close()
	os.system('gnuplot gnuplot.script')
