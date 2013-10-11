#ifndef _GMM_REPLACEMENT_H_
#define _GMM_REPLACEMENT_H_

#include "list.h"
#include "core.h"

int victim_select_lru(
		long size_needed,
		struct region **excls,
		int nexcl,
		int local_only,
		struct list_head *victims);
int victim_select_lfu(
		long size_needed,
		struct region **excls,
		int nexcl,
		int local_only,
		struct list_head *victims);

#endif
