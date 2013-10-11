#include <stdio.h>

/* whether pointer q is included in pointer array a[0:n-1] */
#define is_included(a, n, q)	\
	do { \
		int i, included = 0; \
		for (i = 0; i < n; i++) { \
			if (q == a[i]) { \
				included = 1; \
				break; \
			} \
		} \
		included; \
	} while (0)

#define aaa(a) do { 1; } while (0)

int main()
{
	void *a[4] = {0, 1, 2, 3};
	int b[4] = {4, 5, 6, 7};

//	if (aaa(1))
//		printf("aaa\n");

//	if (is_included(a, 4, 2)) {
//		printf("shit\n");
//	}

	printf("%s\n", AAA);	// AAA passed from gcc -D

	return 0;
}
