#ifndef _GDAEMON_H_

void init_gdaemon(void);
void scheduleCudaCall(void);

void addMemcpyCall();
void addMallocCall();
void addFreeCall();
void addConfigureCall();
void addLaunchCall();
void addSetupCall();
void addSyncCall();
void addMemsetCall();

//#define GD_DEBUG

#endif
