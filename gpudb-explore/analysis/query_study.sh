#!/bin/bash


gmm_usage_plot() {
	FILES=$1
	LEGS=$2
	OFILE=$3
	
	R --slave --vanilla --quiet --no-save << EEE
	
	require(ggplot2)
	qfs	<- c($FILES)
	legs<- c($LEGS)
	
	qnum <- length(qfs)
	if(length(legs) != qnum) {
		print(qfs);
		print(legs);
		stop("the number of the legends does not equal the number of the queries ");
	}

	data <- {};
	for(i in 1:qnum) {
		f <- read.table(qfs[i]);
		colnames(f) <- c("Time","Bytes");
		f["Bytes"] <- cumsum(f["Bytes"]);
		f["Query"] <- legs[i];

		data <- rbind(data, f);
	}
#print(data)
	p <- ggplot(data, aes(Time, Bytes, colour = as.factor(Query), group = as.factor(Query))) + geom_line()
	ggsave(filename="$OFILE", plot=p)
	
EEE
}

gmm_usage_parse() {
	#For log collected by ic_gmm debug mode
	LOG=$1;

	awk '{if(NR==1){base=$3}; if($1=="Malloc"){print $3-base,$6}; if($1=="Free"){print $3-base, "-"$6}}' $LOG;
}

gmm_batch() {
	LOGDIR=$1;
	LEGS=$2;
	OFILE=$3;

	for file in $(find $LOGDIR -type f); do
		gmm_usage_parse $file > ${file}.r;
	done

	FILES=$(find $LOGDIR -regex '.*.r$'| awk '{print "\""$1"\""}'| tr "\n" ","| sed 's/,$//g');
	gmm_usage_plot "$FILES" "$LEGS" "$OFILE";

	find $LOGDIR -regex '.*.r$'| xargs rm;
}

########
##main##
########

$@;
