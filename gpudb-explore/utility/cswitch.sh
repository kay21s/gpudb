#!/bin/bash

CDIR=`dirname $0`

mean() { awk '{sum+=$1} END { if(NR) print sum/NR}'; }
export -f mean

usage() {
	echo "Usage: `echo $0| awk -F/ '{print $NF}'`  command [args]"
	echo "[description]:"
	echo "  execute a listed command"
	echo "[commands]:"
	echo "  mean1: get mean of values for each file in a given folder"
	echo
}

if [ $# -lt 1 ]
then
	usage
	exit
fi

CMD=$1;

if [ "$CMD" == "parseAll" ]; then
	DPATH=$2;
	EXEC=$3;
	FILES=$(find $DPATH -type f| sort);
	for file in $FILES; do
		$EXEC $file;
	done
fi

if [ "$CMD" == "mean1" ]; then
	CENTER=$CDIR/center.sh;
	LNUM=$2;
	DPATH=$3;

	FILES=$(find $DPATH -type f| sort);
	for file in $FILES; do
		VAL=$($CENTER -n $LNUM -f $file| cut -d ':' -f 2| mean );
		echo -e "$file\t$VAL"
	done
fi


if [ "$CMD" == "grid1.xc" ]; then
	CENTER=$CDIR/center.sh;
	XCRDS=$2
	YCRDS=$3
	DPATH=$4
	FUNC=$5
	for yc in $XCRDS; do
		LINE=""
		for xc in $YCRDS;do
			VAL=$($CENTER -n 10 -f $DPATH/$xc.$yc/$xc| cut -d ':' -f 2| mean);
			LINE="$LINE $VAL";
		done
		echo $LINE
	done
fi

if [ "$CMD" == "grid1.yc" ]; then
	CENTER=$CDIR/center.sh;
	XCRDS=$2
	YCRDS=$3
	DPATH=$4
	FUNC=$5
	for yc in $XCRDS; do
		LINE=""
		for xc in $YCRDS;do
			VAL=$($CENTER -n 10 -f $DPATH/$xc.$yc/$yc| cut -d ':' -f 2| mean);
			LINE="$LINE $VAL";
		done
		echo $LINE
	done
fi

if [ "$CMD" == "a1" ]; then
	QS=$2;
	Q2S=$3;
	OUT=$4;	

	$0 grid1.xc "$(cat $QS)" "$(cat $QS)" $Q2S > $OUT/xc;
	$0 grid1.yc "$(cat $QS)" "$(cat $QS)" $Q2S > $OUT/yc;
fi
