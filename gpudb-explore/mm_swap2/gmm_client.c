#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <stdio.h>
#include <semaphore.h>

#include "./gmm_core_interface.h"

int main()
{
	fprintf(stdout, "[Before Attach]\n");
	gmm_attach();
	fprintf(stdout, "[Attached]\n");
	gmm_print_sdata();
	fprintf(stdout, "[Before Detach]\n");
	gmm_detach();
	fprintf(stdout, "[Detached]\n");

	return 0;
}

