#!/bin/bash

usage() {
	echo "Usage: `echo $0| awk -F/ '{print $NF}'`  [-option]"
	echo "[description]:"
	echo "  print n lines in the center"
	echo "[option]:"
	echo "  -n number_of_lines"
	echo "  -f path_of_the_file"
	echo
}

if [ $# -lt 4 ]
then
	usage
	exit
fi

while getopts "n:f:" OPTION
do
	case $OPTION in
		n)
			LNUM=$OPTARG;
			;;
		f)
			FPATH=$OPTARG;
			;;
		?)
			usage
			exit
			;;
	esac
done

LEN=$(cat $FPATH| wc -l)
HSKIP=$(($LEN - ($LEN- $LNUM) /2))

head -n $HSKIP $FPATH| tail -n $LNUM
