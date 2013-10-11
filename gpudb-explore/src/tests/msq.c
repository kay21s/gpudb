// Testing POXIS message queue functionalities.
// Refer to http://linux.die.net/man/7/mq_overview
#include <fcntl.h>
#include <mqueue.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <errno.h>

int test_rw()
{
	#define MSQ_NAME	"/msq_test"
	struct mq_attr qattr;
	char *msgbuf;
	mqd_t mqid;
	pid_t pid;
	struct timespec t;
	int ret = 0;

	mqid = mq_open(MSQ_NAME, O_RDONLY | O_CREAT | O_EXCL, 0600, NULL);
	if (mqid == (mqd_t) -1) {
		perror("Failed to open message queue");
		return -1;
	}

	if (mq_getattr(mqid, &qattr) == -1) {
		perror("Failed to get msq attr");
		ret = -1;
		goto finish;
	}
	printf("Max message size: %ld\n", qattr.mq_msgsize);

	msgbuf = (char *)malloc(qattr.mq_msgsize + 1);
	if (!msgbuf) {
		perror("malloc failed");
		ret = -1;
		goto finish;
	}

	clock_gettime(CLOCK_REALTIME, &t);
	t.tv_sec++;

	pid = fork();
	if (pid < 0) {
		perror("Failed to spawn process");
		ret = -1;
		goto finish;
	}

	if (pid > 0) {
		// Parent: Read a message from the queue
		if (mq_timedreceive(mqid, msgbuf, qattr.mq_msgsize + 1, NULL, &t) == -1) {
			perror("mq_timedreceive failed");
			ret = -1;
			goto finish;
		}

		msgbuf[4] = '\0';
		printf("Message received: %s\n", msgbuf);
	}
	else {
		// Child: Send a message to the queue
		mq_close(mqid);
		mqid = mq_open(MSQ_NAME, O_WRONLY);
		if (mqid == (mqd_t) -1) {
			perror("Failed to open message queue in write-only mode");
			free(msgbuf);
			exit(1);
		}

		memcpy(msgbuf, "shit\0", 5);
		if (mq_timedsend(mqid, msgbuf, 5, 0, &t) == -1) {
			perror("mq_timedsend failed");
			free(msgbuf);
			mq_close(mqid);
			exit(1);
		}
		printf("Message sent\n");

		free(msgbuf);
		mq_close(mqid);
		exit(0);
	}

finish:
	free(msgbuf);
	mq_close(mqid);
	mq_unlink(MSQ_NAME);
	return ret;
}

int main()
{
	int ret = 0;

	ret |= test_rw();

	return ret;
}
