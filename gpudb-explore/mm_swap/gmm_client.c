#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <stdio.h>
#include <semaphore.h>

#include "./gmm_meta_interface.h"
#include "./gmm_meta.h"

int main()
{
	gmm_attach();
	print_gmm_sdata();
	gmm_detach();

	return 0;
}

