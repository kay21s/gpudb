// The protocol between GMM clients and GMM server

#ifndef _GMM_PROTOCOL_H_
#define _GMM_PROTOCOL_H_

#include <unistd.h>
#include "list.h"
#include "spinlock.h"
#include "atomic.h"

// The maximum number of concurrent processes managed by GMM
#define NCLIENTS	32

// A GMM client registered in the global shared memory.
// Each client has a POXIS message queue, named "/gmm_cli_%pid",
// that receives requests and/or notifications from other peer clients.
struct gmm_client {
	int index;				// index of this client; -1 means unoccupied
	int iprev;				// index of the previous client in the LRU list
	int inext;				// index of the next client in the LRU list

	int pinned;
	long size_detachable;	// TODO: update it correctly
	//long lru_size;		// TODO
	//long lru_cost;		// TODO
	pid_t pid;
};

// The global management info shared by all GMM clients
struct gmm_global {
	long mem_total;				// Total size of device memory.
	latomic_t mem_used;			// Size of used (attached) device memory
								// NOTE: in numbers, device memory may be
								// over-used, i.e., mem_used > mem_total.

	struct spinlock lock;		// This lock works only when the hardware cache
								// coherence protocol deals with virtual caches.
	struct gmm_client clients[NCLIENTS];
	int nclients;
	int imru;
	int ilru;
};

enum msgtype {
	MSG_REQ_EVICT,
	MSG_REP_ACK,
};

// Message header
struct msg {
	int type;
	int size;
};

// A message requesting for eviction.
struct msg_req {
	int type;
	int size;

	int from;
	long size_needed;
	int block;
};

// A message replying an eviction request.
struct msg_rep {
	int type;
	int size;

	int from;
	int ret;
};

#define GMM_SEM_LAUNCH	"/gmm_sem_launch"
#define GMM_SHM_GLOBAL	"/gmm_shm_global"

// Add the inew'th client to the MRU end of p's client list
static inline void ILIST_ADD(struct gmm_global *p, int inew)
{
	if (p->imru == -1) {
		p->ilru = p->imru = inew;
		p->clients[inew].iprev = -1;
		p->clients[inew].inext = -1;
	}
	else {
		p->clients[inew].iprev = -1;
		p->clients[inew].inext = p->imru;
		p->clients[p->imru].iprev = inew;
		p->imru = inew;
	}
}

// Delete a client from p's client list
static inline void ILIST_DEL(struct gmm_global *p, int idel)
{
	int iprev = p->clients[idel].iprev;
	int inext = p->clients[idel].inext;

	if (iprev != -1)
		p->clients[iprev].inext = inext;
	else
		p->imru = inext;

	if (inext != -1)
		p->clients[inext].iprev = iprev;
	else
		p->ilru = iprev;
}

// Move a client to the MRU end of p's client list
static inline void ILIST_MOV(struct gmm_global *p, int imov)
{
	ILIST_DEL(p, imov);
	ILIST_ADD(p, imov);
}


#endif
