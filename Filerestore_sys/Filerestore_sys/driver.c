/*
 * driver.c - DriverEntry, filter registration, unload
 */

#include "driver.h"

/* Global context - zero-initialized */
GLOBAL_CONTEXT g_Context = { 0 };

/* Forward declarations */
DRIVER_INITIALIZE DriverEntry;

static NTSTATUS
FilterUnloadCallback(
    _In_ FLT_FILTER_UNLOAD_FLAGS Flags
    );

/* ===== Filter registration ===== */

static const FLT_OPERATION_REGISTRATION FilterCallbacks[] = {
    {
        IRP_MJ_SET_INFORMATION,
        0,
        PreSetInformation,      /* PreOperation */
        NULL                    /* PostOperation - not needed */
    },
    { IRP_MJ_OPERATION_END }
};

static const FLT_REGISTRATION FilterRegistration = {
    sizeof(FLT_REGISTRATION),       /* Size */
    FLT_REGISTRATION_VERSION,       /* Version */
    0,                              /* Flags */
    NULL,                           /* ContextRegistration */
    FilterCallbacks,                /* OperationRegistration */
    FilterUnloadCallback,           /* FilterUnloadCallback */
    NULL,                           /* InstanceSetupCallback */
    NULL,                           /* InstanceQueryTeardownCallback */
    NULL,                           /* InstanceTeardownStartCallback */
    NULL,                           /* InstanceTeardownCompleteCallback */
    NULL,                           /* GenerateFileNameCallback */
    NULL,                           /* NormalizeNameComponentCallback */
    NULL                            /* NormalizeContextCleanupCallback */
};

/* ===== DriverEntry ===== */

NTSTATUS
DriverEntry(
    _In_ PDRIVER_OBJECT DriverObject,
    _In_ PUNICODE_STRING RegistryPath
    )
{
    NTSTATUS status;

    UNREFERENCED_PARAMETER(RegistryPath);

    /* 1. Initialize the event ring buffer lock */
    KeInitializeSpinLock(&g_Context.BufferLock);
    g_Context.EventCount = 0;
    g_Context.EventHead = 0;
    g_Context.EventTail = 0;
    g_Context.TotalEvents = 0;
    g_Context.DroppedEvents = 0;
    g_Context.MonitorActive = 0;

    /* 2. Register the minifilter */
    status = FltRegisterFilter(
        DriverObject,
        &FilterRegistration,
        &g_Context.FilterHandle
        );

    if (!NT_SUCCESS(status)) {
        return status;
    }

    /* 3. Create the communication port */
    status = SetupCommunicationPort();
    if (!NT_SUCCESS(status)) {
        FltUnregisterFilter(g_Context.FilterHandle);
        return status;
    }

    /* 4. Start filtering */
    status = FltStartFiltering(g_Context.FilterHandle);
    if (!NT_SUCCESS(status)) {
        FltCloseCommunicationPort(g_Context.ServerPort);
        FltUnregisterFilter(g_Context.FilterHandle);
        return status;
    }

    return STATUS_SUCCESS;
}

/* ===== FilterUnloadCallback ===== */

static NTSTATUS
FilterUnloadCallback(
    _In_ FLT_FILTER_UNLOAD_FLAGS Flags
    )
{
    UNREFERENCED_PARAMETER(Flags);

    /* Close the server port first - prevents new connections */
    if (g_Context.ServerPort != NULL) {
        FltCloseCommunicationPort(g_Context.ServerPort);
        g_Context.ServerPort = NULL;
    }

    /* Unregister the filter */
    if (g_Context.FilterHandle != NULL) {
        FltUnregisterFilter(g_Context.FilterHandle);
        g_Context.FilterHandle = NULL;
    }

    return STATUS_SUCCESS;
}
