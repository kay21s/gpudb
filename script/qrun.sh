#!/bin/bash

CDIR=`dirname $0`
source $CDIR/gpudb_env.sh

RUNPATH=$GPUDB_CUDA_PATH

usage() {
	echo "Usage: `echo $0| awk -F/ '{print $NF}'`  [-option]"
	echo "[option]:"
	echo "  -i path  : specify the folder path of the executables"
	echo "  -r repeat: repeat times for each query"
	echo "  -p plan  : execution plan where each line specifies which queires to corun"
	echo "  -o path  : specify the output path"
	echo
}

if [ $# -lt 8 ]
then
	usage
	exit
fi

while getopts "i:o:r:p:" OPTION
do
	case $OPTION in
		i)
			EXEPATH=$OPTARG;
			;;
		o)
			OUTPATH=$OPTARG;
			;;
		r)
			REP=$OPTARG;
			;;
		p)
			PLAN=$OPTARG;
			;;
		?)
			usage
			exit
			;;
	esac
done

while read line; do
	ODIR=$OUTPATH/$(echo $line| sed 's/ /./g')
	echo "$line"
	mkdir $ODIR

	echo "warm up ..."
	for query in $line; do
		cp $EXEPATH/$query $RUNPATH;
		cd $RUNPATH && $RUNPATH/$query 1
	done
	sleep 3

	echo "corun $REP times ..."
	for query in $line; do
		cd $RUNPATH && LD_PRELOAD=$PRELOAD $RUNPATH/$query $REP > $ODIR/$query 2>&1  &
	done

	echo "sync & cleanup ..."
	sleep 1
	wait;
	sleep 3
	for query in $line; do
		rm -f $RUNPATH/$query
	done
	echo ""

done < $PLAN
