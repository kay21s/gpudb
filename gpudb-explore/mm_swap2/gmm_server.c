#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <stdio.h>
#include <semaphore.h>

#include "./gmm_core_interface.h"

int main()
{
	char x;
	size_t memsize;
	gmm_init_attach();
	printf("GPU Memory Available: %ld\n", gmm_getFreeMem());
	while(1) {
		x=getchar();
		if (x == 'r') {
			printf("Set GPU Memory in Bytes (E.g. 1347483648): \n");
			scanf("%lu", &memsize);
			gmm_setFreeMem(memsize);
			printf("GPU Memory Available: %ld\n", gmm_getFreeMem());
		} else if (x == 'q') break;
	}
	gmm_detach();
	gmm_reclaim();	

	return 0;
}

