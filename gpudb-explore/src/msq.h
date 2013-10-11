#ifndef _GMM_MSQ_H_
#define _GMM_MSQ_H_

int msq_init();
void msq_fini();
int msq_send_req_evict(int client, long size_needed, int block);
int msq_send_rep_ack(int client, int ack);

#endif
