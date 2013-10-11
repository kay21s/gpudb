#!/bin/bash

CDIR=`dirname $0`
source $CDIR/gpudb_env.sh

TRANSLATE=$GPUDB_PATH/translate.py

usage() {
	echo "Usage: `echo $0| awk -F/ '{print $NF}'`  [-option]"
	echo "[option]:"
	echo "  -i path  : specify the path of the sqls"
	echo "  -s scheme: scheme file"
	echo "  -o path  : specify the output path for the executables"
	echo
}

if [ $# -lt 6 ]
then
	usage
	exit
fi

while getopts "i:s:o:" OPTION
do
	case $OPTION in
		i)
			INPATH=$OPTARG;
			;;
		o)
			OUTPATH=$OPTARG;
			;;
		s)
			SCHEME=$OPTARG;
			;;
		?)
			usage
			exit
			;;
	esac
done

SQLS=$(ls $INPATH|grep '.sql');
for sql in $SQLS; do
	echo "$sql"
	cd $GPUDB_PATH && $TRANSLATE $INPATH/$sql $SCHEME
	EXECUTABLE=$(echo $sql| sed 's/.sql//g')
	cd $GPUDB_CUDA_PATH && make >/dev/null 2>&1  && cp $GPUDB_CUDA_PATH/GPUDATABASE $OUTPATH/$EXECUTABLE
done

