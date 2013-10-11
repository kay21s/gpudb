// This spinlock implementation is copied from MIT's xv6.
#ifndef _GMM_SPINLOCK_H_
#define _GMM_SPINLOCK_H_

struct spinlock {
	unsigned int locked;
};

static inline unsigned int xchg(
		volatile unsigned int *addr,
		unsigned int newval)
{
  unsigned int result;

  // The + in "+m" denotes a read-modify-write operand.
  asm volatile("lock; xchgl %0, %1" :
               "+m" (*addr), "=a" (result) :
               "1" (newval) :
               "cc");
  return result;
}


static inline void initlock(struct spinlock *lk)
{
  lk->locked = 0;
}

// Acquire the lock.
// Loops (spins) until the lock is acquired.
static inline void acquire(struct spinlock *lk)
{
  // The xchg is atomic.
  // It also serializes, so that reads after acquire are not
  // reordered before it.
  while (xchg(&lk->locked, 1U) != 0)
    ;
}

// Try to acquire the lock.
// Return 1 if the locking was successful.
#define TRY_ACQUIRE_TIMES		5
static inline int try_acquire(struct spinlock *lk)
{
	int i = 0;

	while (i < TRY_ACQUIRE_TIMES && xchg(&lk->locked, 1U) != 0)
		i++;

	return i < TRY_ACQUIRE_TIMES;
}

// Release the lock.
static inline void release(struct spinlock *lk)
{
  // The xchg serializes, so that reads before release are
  // not reordered after it.  The 1996 PentiumPro manual (Volume 3,
  // 7.2) says reads can be carried out speculatively and in
  // any order, which implies we need to serialize here.
  // But the 2007 Intel 64 Architecture Memory Ordering White
  // Paper says that Intel 64 and IA-32 will not move a load
  // after a store. So lock->locked = 0 would work here.
  // The xchg being asm volatile ensures gcc emits it after
  // the above assignments (and after the critical section).
  xchg(&lk->locked, 0);
}

#endif
