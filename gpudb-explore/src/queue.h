// Unused
#ifndef _GMM_QUEUE_H_
#define _GMM_QUEUE_H_

#include "spinlock.h"

#define QLEN		64

// A simple queue
struct queue {
	struct spinlock lock;
	void elems[QLEN];
	int head;
	int tail;
};

static inline void qinit(struct queue *q)
{
	initlock(&q->lock);
	q->head = q->tail = 0;
}

static inline void qlock(struct queue *q)
{
	acquire(&q->lock);
}

static inline void qunlock(struct queue *q)
{
	release(&q->lock);
}

static inline int qpush(struct queue *q, void *elem)
{
	if ((q->tail + 1) % QLEN == q->head)
		return -1;
	q->elems[q->tail] = elem;
	q->tail++;
	return 0;
}

static inline void *qpop(struct queue *q)
{
	if (q->tail == q->head)
		return NULL;
	q->tail = (q->tail - 1 + QLEN) % QLEN;

	return q->elems[q->tail];
}

static inline void *qget(struct queue *q)
{
	void *elem;

	if (q->tail == q->head)
		elem = NULL;
	else {
		elem = q->elems[q->head];
		q->head = (q->head + 1) % QLEN;
	}

	return elem;
}

#endif
