#!/bin/bash


slowdown() {
	QS=$1
	Q1=$2
	Q2X=$3
	Q2Y=$4
	OCSV=$5
	
	R --slave --vanilla --quiet --no-save << EEE
	
	qs <- read.table("$QS")[[1]]
	q1 <- read.table("$Q1")
	q2x <- read.table("$Q2X")
	q2y <- read.table("$Q2Y")
	
	q2sd <- (sweep(q2x, MARGIN=2, t(q1), "/") + q2y/t(q1) )/2
	colnames(q2sd) <- qs
	rownames(q2sd) <- qs
	
	write.csv(q2sd, "$OCSV")
	
EEE
}

speedup() {
	QS=$1
	Q1=$2
	Q2X=$3
	Q2Y=$4
	OCSV=$5
	
	R --slave --vanilla --quiet --no-save << EEE
	
	qs <- read.table("$QS")[[1]]
	q1 <- read.table("$Q1")
	q2x <- read.table("$Q2X")
	q2y <- read.table("$Q2Y")
	
	q2sd <- (sweep(1/q2x, MARGIN=2, t(q1), "*") + t(q1)/q2y )
	colnames(q2sd) <- qs
	rownames(q2sd) <- qs
	
	write.csv(q2sd, "$OCSV")
	
EEE
}

a1() {
	QS=$1;
	Q1=$2;
	Q2PATH=$3;

	slowdown $QS $Q1 $Q2PATH/xc $Q2PATH/yc $Q2PATH/slowdown.csv;
	speedup $QS $Q1 $Q2PATH/xc $Q2PATH/yc $Q2PATH/speedup.csv;
}

########
##main##
########

$@;
