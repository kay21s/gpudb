#!/bin/sh

GMMPATH=`pwd`/../../../../syma/src


#LD_PRELOAD=${GMMPATH}/libgmm.so ${GMMPATH}/../tests/tmp/test_evict_remote &
LD_PRELOAD=${GMMPATH}/libgmm.so ./GPUDATABASE 1 --datadir ../../data > debug1.txt &
LD_PRELOAD=${GMMPATH}/libgmm.so ./GPUDATABASE 1 --datadir ../../data > debug2.txt
