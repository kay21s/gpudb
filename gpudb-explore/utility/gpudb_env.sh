#!/bin/bash

ROOT="$( cd "$( dirname "$0" )" && pwd )/../";

export GPUDB_PATH=/home/syma/src/gpudb-read-only
export GPUDB_CUDA_PATH=$GPUDB_PATH/src/cuda
#export PRELOAD=$ROOT/gdaemon/ic_async.so
#export PRELOAD=$ROOT/gdaemon/ic_stream.so
#export PRELOAD=$ROOT/gdaemon/ic_mstream.so
#export PRELOAD=$ROOT/mm/ic_gmm.so
#export PRELOAD=$ROOT/mm_swap/ic_gmm.so
export PRELOAD=$ROOT/mm_swap2/ic_gmm.so
