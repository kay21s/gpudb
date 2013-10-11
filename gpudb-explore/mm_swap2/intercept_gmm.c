/*
 * Use nm to list the exact names of function calls in the program before
 * deciding which functions to intercept here.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <dlfcn.h>                               /* header required for dlsym() */
#include <driver_types.h>
#include <sys/time.h>
//#include <cuda.h>
//#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include <signal.h>
#include <execinfo.h>
#include <pthread.h>

#include "gmm_core_interface.h"


void main_constructor( void )
    __attribute__ ((no_instrument_function, constructor));

void main_constructor( void ) {
	gmm_attach();
}


void main_destructor( void )
        __attribute__ ((no_instrument_function, destructor));

void main_destructor( void ) {
	gmm_detach();
	//pthread_exit(NULL);
}

