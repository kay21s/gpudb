#ifndef _GMM_ATOMIC_H_
#define _GMM_ATOMIC_H_

typedef int atomic_t;
typedef long latomic_t;

// Integer atomics
static inline void atomic_set(atomic_t *ptr, int val)
{
	*ptr = val;
}

static inline int atomic_read(atomic_t *ptr)
{
	return *ptr;
}

static inline int atomic_add(atomic_t *ptr, int val)
{
	return __sync_fetch_and_add(ptr, val);
}

static inline int atomic_inc(atomic_t *ptr)
{
	return __sync_fetch_and_add(ptr, 1);
}

static inline int atomic_sub(atomic_t *ptr, int val)
{
	return __sync_fetch_and_sub(ptr, val);
}

static inline int atomic_dec(atomic_t *ptr)
{
	return __sync_fetch_and_sub(ptr, 1);
}

// Long atomics
static inline void latomic_set(latomic_t *ptr, long val)
{
	*ptr = val;
}

static inline int latomic_read(latomic_t *ptr)
{
	return *ptr;
}

static inline long latomic_add(latomic_t *ptr, long val)
{
	return __sync_fetch_and_add(ptr, val);
}

static inline long latomic_inc(latomic_t *ptr)
{
	return __sync_fetch_and_add(ptr, 1);
}

static inline long latomic_sub(latomic_t *ptr, long val)
{
	return __sync_fetch_and_sub(ptr, val);
}

static inline long latomic_dec(latomic_t *ptr)
{
	return __sync_fetch_and_sub(ptr, 1);
}

#endif
