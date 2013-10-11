#include <stdio.h>
#include <stdlib.h>
#include "common.h"

static void show_stackframe() {
  void *trace[32];
  char **messages = (char **)NULL;
  int i, trace_size = 0;

  trace_size = backtrace(trace, 32);
  messages = backtrace_symbols(trace, trace_size);
  fprintf(stderr, "Printing stack frames:\n");
  for (i=0; i < trace_size; ++i)
        fprintf(stderr, "\t%s\n", messages[i]);
}

void panic(char *msg)
{
	fprintf(stderr, "[gmm:panic] %s\n", msg);
	show_stackframe();
	exit(-1);
}
