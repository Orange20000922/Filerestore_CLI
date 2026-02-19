/*
 * common.h - Shared definitions between kernel driver and user-mode client
 *
 * This header is included by both the minifilter driver (kernel) and the
 * user-mode application (Filerestore_CLI). Keep it pure C compatible.
 */

#ifndef _FILERESTORE_COMMON_H_
#define _FILERESTORE_COMMON_H_

#ifdef __cplusplus
extern "C" {
#endif

/* Communication port name */
#define FILERESTORE_PORT_NAME   L"\\FileRestoreMonPort"

/* Command codes (user -> kernel) */
#define CMD_START_MONITOR       1
#define CMD_STOP_MONITOR        2
#define CMD_GET_EVENTS          3
#define CMD_GET_STATS           4
#define CMD_GET_VERSION         5

/* Delete types */
#define DELETE_TYPE_PERMANENT    0
#define DELETE_TYPE_RECYCLE      1

/* Limits */
#define MAX_LCN_ENTRIES         64
#define MAX_FILE_PATH_CHARS     520
#define DRIVER_VERSION_STRING   "1.0.0"

/* ===== Structures ===== */

/* LCN (Logical Cluster Number) entry for file extent mapping */
typedef struct _LCN_ENTRY {
    ULONGLONG   StartLCN;
    ULONG       ClusterCount;
    ULONG       Reserved;
} LCN_ENTRY, *PLCN_ENTRY;

/* Single delete event notification */
typedef struct _DELETE_NOTIFICATION {
    LARGE_INTEGER   Timestamp;
    ULONG           ProcessId;
    ULONG           DeleteType;
    LARGE_INTEGER   FileSize;
    LARGE_INTEGER   AllocationSize;
    LARGE_INTEGER   CreationTime;
    LARGE_INTEGER   LastWriteTime;
    ULONG           LCNCount;
    LCN_ENTRY       LCNEntries[MAX_LCN_ENTRIES];
    USHORT          FileNameLength;     /* in bytes */
    WCHAR           FileName[MAX_FILE_PATH_CHARS];
} DELETE_NOTIFICATION, *PDELETE_NOTIFICATION;

/* Command message (user -> kernel) */
typedef struct _COMMAND_MESSAGE {
    ULONG Command;
} COMMAND_MESSAGE, *PCOMMAND_MESSAGE;

/* Events reply header (kernel -> user, for CMD_GET_EVENTS) */
typedef struct _EVENTS_REPLY {
    ULONG               EventCount;
    DELETE_NOTIFICATION  Events[1];     /* variable-length array */
} EVENTS_REPLY, *PEVENTS_REPLY;

/* Statistics reply (kernel -> user, for CMD_GET_STATS) */
typedef struct _STATS_REPLY {
    ULONG   TotalEvents;
    ULONG   PendingEvents;
    ULONG   DroppedEvents;
    ULONG   MonitorActive;
} STATS_REPLY, *PSTATS_REPLY;

/* Version reply (kernel -> user, for CMD_GET_VERSION) */
typedef struct _VERSION_REPLY {
    CHAR    Version[64];
} VERSION_REPLY, *PVERSION_REPLY;

#ifdef __cplusplus
}
#endif

#endif /* _FILERESTORE_COMMON_H_ */
