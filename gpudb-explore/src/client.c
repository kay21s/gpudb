#include <sys/mman.h>
#include <sys/stat.h> /* For mode constants */
#include <fcntl.h>
#include <semaphore.h>
#include <mqueue.h>
#include <string.h>
#include <errno.h>
#include <pthread.h>

#include "common.h"
#include "protocol.h"
#include "msq.h"


sem_t *sem_launch = SEM_FAILED;		// Guarding kernel launches
struct gmm_global *pglobal = NULL;	// Global shared memory
int cid = -1;						// Id of this client


static int client_alloc()
{
	int id = -1;

	// Get a unique client id
	acquire(&pglobal->lock);
	if (pglobal->nclients < NCLIENTS) {
		for (id = 0; id < NCLIENTS; id++) {
			if (pglobal->clients[id].index == -1)
				break;
		}
	}
	if (id >= 0 && id < NCLIENTS) {
		memset(pglobal->clients + id, 0, sizeof(pglobal->clients[0]));
		pglobal->clients[id].index = id;
		pglobal->clients[id].pid = gettid();
		ILIST_ADD(pglobal, id);
		pglobal->nclients++;
	}
	release(&pglobal->lock);

	return id;
}

static void client_free(int id)
{
	if (id >= 0 && id < NCLIENTS) {
		acquire(&pglobal->lock);
		ILIST_DEL(pglobal, id);
		pglobal->nclients--;
		pglobal->clients[id].index = -1;
		release(&pglobal->lock);
	}
}

// Attach this process to the global GMM arena.
int client_attach()
{
	int shmfd;

	sem_launch = sem_open(GMM_SEM_LAUNCH, 0);
	if (sem_launch == SEM_FAILED) {
		GMM_DPRINT("unable to open launch semaphore: %s\n", strerror(errno));
		return -1;
	}

	shmfd = shm_open(GMM_SHM_GLOBAL, O_RDWR, 0);
	if (shmfd == -1) {
		GMM_DPRINT("unable to open shared memory: %s\n", strerror(errno));
		goto fail_shm;
	}

	pglobal = (struct gmm_global *)mmap(NULL, sizeof(*pglobal),
			PROT_READ | PROT_WRITE, MAP_SHARED, shmfd, 0);
	if (pglobal == MAP_FAILED) {
		GMM_DPRINT("failed to mmap shared memory: %s\n", strerror(errno));
		goto fail_mmap;
	}

	if (msq_init() < 0) {
		GMM_DPRINT("message queue init failed\n");
		goto fail_msq;
	}

	cid = client_alloc();
	if (cid == -1) {
		GMM_DPRINT("failed to allocate client\n");
		goto fail_client;
	}

	close(shmfd);
	return 0;

fail_client:
	msq_fini();
fail_msq:
	munmap(pglobal, sizeof(*pglobal));
fail_mmap:
	pglobal = NULL;
	close(shmfd);
fail_shm:
	sem_close(sem_launch);
	sem_launch = SEM_FAILED;

	return -1;
}

void client_detach() {
	client_free(cid);
	cid = -1;
	msq_fini();
	if (pglobal != NULL) {
		munmap(pglobal, sizeof(*pglobal));
		pglobal = NULL;
	}
	if (sem_launch != SEM_FAILED) {
		sem_close(sem_launch);
		sem_launch = SEM_FAILED;
	}
}

long memsize_total()
{
	return pglobal->mem_total;
}

long memsize_free()
{
	long freesize = pglobal->mem_total - latomic_read(&pglobal->mem_used);
	return freesize < 0 ? 0 : freesize;
}

long memsize_free2()
{
	return pglobal->mem_total - latomic_read(&pglobal->mem_used);
}

void update_attached(long delta)
{
	latomic_add(&pglobal->mem_used, delta);
}

void update_detachable(long delta)
{
	latomic_add(&pglobal->clients[cid].size_detachable, delta);
}

void launch_wait()
{
	int ret;
	do {
		ret = sem_wait(sem_launch);
	} while (ret == -1 && errno == EINTR);
}

void launch_signal()
{
	sem_post(sem_launch);
}

// Get the id of the least recently used client with detachable
// device memory. The client is pinned if it is a remote client.
int client_lru_detachable()
{
	int iclient;

	acquire(&pglobal->lock);
	for (iclient = pglobal->ilru; iclient != -1;
			iclient = pglobal->clients[iclient].iprev) {
		if (pglobal->clients[iclient].size_detachable > 0)
			break;
	}
	if (iclient != -1 && iclient != cid)
		pglobal->clients[iclient].pinned++;
	release(&pglobal->lock);

	return iclient;
}

void client_unpin(int client)
{
	acquire(&pglobal->lock);
	pglobal->clients[client].pinned--;
	release(&pglobal->lock);
}

// Is the client a local client?
int is_client_local(int client)
{
	return client == cid;
}

int getcid()
{
	return cid;
}

pid_t cidtopid(int cid)
{
	return pglobal->clients[cid].pid;
}
