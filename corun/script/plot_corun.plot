set xlabel "Co-run queries"
set ylabel "Speedup"

set terminal png size 1200, 400

set output "speedup.png"
set xtics nomirror rotate by 30 font ",8"
plot "result_corun" using 2:xticlabels(1) with boxes
