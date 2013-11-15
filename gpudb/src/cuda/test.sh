#!/bin/sh

GMMPATH=`pwd`/../../../../gdb/src


#LD_PRELOAD=${GMMPATH}/libgmm.so ${GMMPATH}/../tests/tmp/test_evict_remote &
#LD_PRELOAD=${GMMPATH}/libgmm.so ./GPUDATABASE 1 --datadir ../../data > debug1.txt &
LD_PRELOAD=${GMMPATH}/libgmm.so ./GPUDATABASE 1 --datadir ../../data &
#LD_PRELOAD=${GMMPATH}/libgmm.so ./GPUDATABASE 1 --datadir ../../data > debug2.txt
LD_PRELOAD=${GMMPATH}/libgmm.so ./GPUDATABASE 1 --datadir ../../data
