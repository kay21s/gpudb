all:
	./visualize.py memspace.txt memusage.txt memcpy.txt
	gnuplot gnuplot.script
space:
	./visualize-space.py memspace.txt
	gnuplot gnuplot.space.script
usage:
	./visualize-usage.py memusage.txt
	gnuplot gnuplot.usage.script
copy:
	./visualize-copy.py memcpy.txt
	gnuplot gnuplot.copy.script
util:
	./util.py memspace.txt memusage.txt memcpy.txt
potential:
	gnuplot pot.script
clean:
	rm -f mem*.txt mem*.vis *.png pot.script
	rm -f ./immediate/* ./png/* ./vis/* ./results/*
