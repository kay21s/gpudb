set xlabel "Co-run queries"
set ylabel "Speedup"

set terminal png size 1200, 400

set output "speedup.png"
set xtics nomirror rotate by 30 font ",8"
plot "speedup0" using 2:xticlabels(1) title "speedup of qx in qx.qy" with lines, "speedup1" using 2:xticlabels(1) title "speedup of qy in qx.qy" with lines, "result_corun" using 2:xticlabels(1) title "corun speedup" with boxes
