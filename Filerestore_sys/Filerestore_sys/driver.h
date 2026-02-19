/*
 * driver.h - Internal driver declarations
 */

#ifndef _FILERESTORE_DRIVER_H_
#define _FILERESTORE_DRIVER_H_

#include <fltKernel.h>
#include <ntstrsafe.h>
#include "common.h"

/* Pool tag: 'MsFR' (stored little-endian as 'RFsM') */
#define POOL_TAG                'RFsM'

/* Maximum pending events in the kernel ring buffer */
#define MAX_PENDING_EVENTS      256

/* ===== Global context ===== */

typedef struct _GLOBAL_CONTEXT {
    PFLT_FILTER     FilterHandle;
    PFLT_PORT       ServerPort;         /* server-side communication port */
    PFLT_PORT       ClientPort;         /* connected client port (single client) */

    /* Kernel event ring buffer */
    KSPIN_LOCK      BufferLock;
    LONG            EventCount;         /* number of events currently buffered */
    LONG            EventHead;          /* next write position */
    LONG            EventTail;          /* next read position */
    DELETE_NOTIFICATION Events[MAX_PENDING_EVENTS];

    /* Statistics */
    volatile LONG   TotalEvents;
    volatile LONG   DroppedEvents;
    volatile LONG   MonitorActive;      /* 1 = monitoring active */
} GLOBAL_CONTEXT, *PGLOBAL_CONTEXT;

extern GLOBAL_CONTEXT g_Context;

/* ===== Function prototypes ===== */

/* communication.c */
NTSTATUS
SetupCommunicationPort(VOID);

VOID
BufferPushEvent(
    _In_ const DELETE_NOTIFICATION* Event
    );

/* filter.c */
FLT_PREOP_CALLBACK_STATUS
PreSetInformation(
    _Inout_ PFLT_CALLBACK_DATA Data,
    _In_ PCFLT_RELATED_OBJECTS FltObjects,
    _Flt_CompletionContext_Outptr_ PVOID *CompletionContext
    );

#endif /* _FILERESTORE_DRIVER_H_ */
