/*
 * communication.c - FltMgr communication port management and event buffering
 */

#include "driver.h"

/* Forward declarations for port callbacks */
static NTSTATUS
ConnectNotifyCallback(
    _In_ PFLT_PORT ClientPort,
    _In_opt_ PVOID ServerPortCookie,
    _In_reads_bytes_opt_(SizeOfContext) PVOID ConnectionContext,
    _In_ ULONG SizeOfContext,
    _Outptr_result_maybenull_ PVOID *ConnectionCookie
    );

static VOID
DisconnectNotifyCallback(
    _In_opt_ PVOID ConnectionCookie
    );

static NTSTATUS
MessageNotifyCallback(
    _In_opt_ PVOID PortCookie,
    _In_reads_bytes_opt_(InputBufferLength) PVOID InputBuffer,
    _In_ ULONG InputBufferLength,
    _Out_writes_bytes_to_opt_(OutputBufferLength, *ReturnOutputBufferLength) PVOID OutputBuffer,
    _In_ ULONG OutputBufferLength,
    _Out_ PULONG ReturnOutputBufferLength
    );

/* ===== SetupCommunicationPort ===== */

NTSTATUS
SetupCommunicationPort(VOID)
{
    NTSTATUS status;
    UNICODE_STRING portName;
    PSECURITY_DESCRIPTOR sd = NULL;
    OBJECT_ATTRIBUTES oa;

    RtlInitUnicodeString(&portName, FILERESTORE_PORT_NAME);

    /* Build a security descriptor that only allows admin access */
    status = FltBuildDefaultSecurityDescriptor(&sd, FLT_PORT_ALL_ACCESS);
    if (!NT_SUCCESS(status)) {
        return status;
    }

    InitializeObjectAttributes(
        &oa,
        &portName,
        OBJ_CASE_INSENSITIVE | OBJ_KERNEL_HANDLE,
        NULL,
        sd
        );

    status = FltCreateCommunicationPort(
        g_Context.FilterHandle,
        &g_Context.ServerPort,
        &oa,
        NULL,                       /* ServerPortCookie */
        ConnectNotifyCallback,
        DisconnectNotifyCallback,
        MessageNotifyCallback,
        1                           /* MaxConnections: single client */
        );

    FltFreeSecurityDescriptor(sd);
    return status;
}

/* ===== VerifyCallerImage ===== */

/* ZwQueryInformationProcess is not declared in standard WDK headers */
NTSYSAPI NTSTATUS NTAPI ZwQueryInformationProcess(
    _In_ HANDLE ProcessHandle,
    _In_ PROCESSINFOCLASS ProcessInformationClass,
    _Out_writes_bytes_(ProcessInformationLength) PVOID ProcessInformation,
    _In_ ULONG ProcessInformationLength,
    _Out_opt_ PULONG ReturnLength
    );

/*
 * VerifyCallerImage - Verify the calling process image name matches the
 * expected client executable. Uses ZwQueryInformationProcess(ProcessImageFileName)
 * to get the NT path, then checks if the path suffix matches
 * FILERESTORE_CLIENT_IMAGE_NAME.
 */
static NTSTATUS VerifyCallerImage(VOID)
{
    NTSTATUS status;
    UCHAR buffer[512];
    ULONG returnedLength;
    PUNICODE_STRING imagePath;
    UNICODE_STRING expectedSuffix;
    UNICODE_STRING actualSuffix;
    USHORT suffixBytes;

    status = ZwQueryInformationProcess(
        NtCurrentProcess(),
        ProcessImageFileName,       /* = 27 */
        buffer,
        sizeof(buffer),
        &returnedLength
        );
    if (!NT_SUCCESS(status)) {
        return status;
    }

    imagePath = (PUNICODE_STRING)buffer;
    RtlInitUnicodeString(&expectedSuffix, FILERESTORE_CLIENT_IMAGE_NAME);

    suffixBytes = expectedSuffix.Length;
    if (imagePath->Length < suffixBytes) {
        return STATUS_ACCESS_DENIED;
    }

    /* Extract the suffix of the image path matching the expected length */
    actualSuffix.Buffer = (PWCH)((PUCHAR)imagePath->Buffer +
                                  imagePath->Length - suffixBytes);
    actualSuffix.Length = suffixBytes;
    actualSuffix.MaximumLength = suffixBytes;

    /* Case-insensitive comparison */
    if (RtlCompareUnicodeString(&actualSuffix, &expectedSuffix, TRUE) != 0) {
        return STATUS_ACCESS_DENIED;
    }

    return STATUS_SUCCESS;
}

/* ===== ConnectNotifyCallback ===== */

static NTSTATUS
ConnectNotifyCallback(
    _In_ PFLT_PORT ClientPort,
    _In_opt_ PVOID ServerPortCookie,
    _In_reads_bytes_opt_(SizeOfContext) PVOID ConnectionContext,
    _In_ ULONG SizeOfContext,
    _Outptr_result_maybenull_ PVOID *ConnectionCookie
    )
{
    PCONNECTION_CONTEXT ctx;

    UNREFERENCED_PARAMETER(ServerPortCookie);

    /* 1. Validate connection context (handshake) */
    if (ConnectionContext == NULL ||
        SizeOfContext < sizeof(CONNECTION_CONTEXT)) {
        return STATUS_ACCESS_DENIED;
    }

    ctx = (PCONNECTION_CONTEXT)ConnectionContext;
    if (ctx->Magic != FILERESTORE_CONNECTION_MAGIC ||
        ctx->Version != FILERESTORE_PROTOCOL_VERSION) {
        return STATUS_ACCESS_DENIED;
    }

    /* 2. Verify caller process image name */
    if (!NT_SUCCESS(VerifyCallerImage())) {
        return STATUS_ACCESS_DENIED;
    }

    g_Context.ClientPort = ClientPort;
    *ConnectionCookie = NULL;
    return STATUS_SUCCESS;
}

/* ===== DisconnectNotifyCallback ===== */

static VOID
DisconnectNotifyCallback(
    _In_opt_ PVOID ConnectionCookie
    )
{
    UNREFERENCED_PARAMETER(ConnectionCookie);

    if (g_Context.ClientPort != NULL) {
        FltCloseClientPort(g_Context.FilterHandle, &g_Context.ClientPort);
        g_Context.ClientPort = NULL;
    }

    /* Stop monitoring when client disconnects */
    InterlockedExchange(&g_Context.MonitorActive, 0);
}

/* ===== MessageNotifyCallback ===== */

static NTSTATUS
MessageNotifyCallback(
    _In_opt_ PVOID PortCookie,
    _In_reads_bytes_opt_(InputBufferLength) PVOID InputBuffer,
    _In_ ULONG InputBufferLength,
    _Out_writes_bytes_to_opt_(OutputBufferLength, *ReturnOutputBufferLength) PVOID OutputBuffer,
    _In_ ULONG OutputBufferLength,
    _Out_ PULONG ReturnOutputBufferLength
    )
{
    PCOMMAND_MESSAGE cmdMsg;
    KIRQL oldIrql;
    LONG count;
    LONG tail;
    LONG i;

    UNREFERENCED_PARAMETER(PortCookie);

    *ReturnOutputBufferLength = 0;

    if (InputBuffer == NULL || InputBufferLength < sizeof(COMMAND_MESSAGE)) {
        return STATUS_INVALID_PARAMETER;
    }

    cmdMsg = (PCOMMAND_MESSAGE)InputBuffer;

    switch (cmdMsg->Command) {

    case CMD_START_MONITOR:
        InterlockedExchange(&g_Context.MonitorActive, 1);
        return STATUS_SUCCESS;

    case CMD_STOP_MONITOR:
        InterlockedExchange(&g_Context.MonitorActive, 0);
        return STATUS_SUCCESS;

    case CMD_GET_EVENTS:
    {
        PEVENTS_REPLY reply;
        ULONG replySize;

        if (OutputBuffer == NULL || OutputBufferLength < sizeof(EVENTS_REPLY)) {
            return STATUS_BUFFER_TOO_SMALL;
        }

        reply = (PEVENTS_REPLY)OutputBuffer;

        KeAcquireSpinLock(&g_Context.BufferLock, &oldIrql);

        count = g_Context.EventCount;
        if (count == 0) {
            KeReleaseSpinLock(&g_Context.BufferLock, oldIrql);
            reply->EventCount = 0;
            *ReturnOutputBufferLength = FIELD_OFFSET(EVENTS_REPLY, Events);
            return STATUS_SUCCESS;
        }

        /* Calculate how many events fit in the output buffer */
        replySize = FIELD_OFFSET(EVENTS_REPLY, Events);
        {
            LONG maxEvents = (LONG)((OutputBufferLength - replySize) / sizeof(DELETE_NOTIFICATION)) + 1;
            if (count > maxEvents) {
                count = maxEvents;
            }
        }

        /* Copy events from ring buffer */
        tail = g_Context.EventTail;
        for (i = 0; i < count; i++) {
            RtlCopyMemory(
                &reply->Events[i],
                &g_Context.Events[tail],
                sizeof(DELETE_NOTIFICATION)
                );
            tail = (tail + 1) % MAX_PENDING_EVENTS;
        }

        g_Context.EventTail = tail;
        g_Context.EventCount -= count;

        KeReleaseSpinLock(&g_Context.BufferLock, oldIrql);

        reply->EventCount = (ULONG)count;
        *ReturnOutputBufferLength = FIELD_OFFSET(EVENTS_REPLY, Events) +
                                    (ULONG)count * sizeof(DELETE_NOTIFICATION);
        return STATUS_SUCCESS;
    }

    case CMD_GET_STATS:
    {
        PSTATS_REPLY statsReply;

        if (OutputBuffer == NULL || OutputBufferLength < sizeof(STATS_REPLY)) {
            return STATUS_BUFFER_TOO_SMALL;
        }

        statsReply = (PSTATS_REPLY)OutputBuffer;
        statsReply->TotalEvents = (ULONG)g_Context.TotalEvents;
        statsReply->DroppedEvents = (ULONG)g_Context.DroppedEvents;
        statsReply->MonitorActive = (ULONG)g_Context.MonitorActive;

        KeAcquireSpinLock(&g_Context.BufferLock, &oldIrql);
        statsReply->PendingEvents = (ULONG)g_Context.EventCount;
        KeReleaseSpinLock(&g_Context.BufferLock, oldIrql);

        *ReturnOutputBufferLength = sizeof(STATS_REPLY);
        return STATUS_SUCCESS;
    }

    case CMD_GET_VERSION:
    {
        PVERSION_REPLY verReply;

        if (OutputBuffer == NULL || OutputBufferLength < sizeof(VERSION_REPLY)) {
            return STATUS_BUFFER_TOO_SMALL;
        }

        verReply = (PVERSION_REPLY)OutputBuffer;
        RtlZeroMemory(verReply, sizeof(VERSION_REPLY));
        RtlStringCbCopyA(verReply->Version, sizeof(verReply->Version), DRIVER_VERSION_STRING);

        *ReturnOutputBufferLength = sizeof(VERSION_REPLY);
        return STATUS_SUCCESS;
    }

    default:
        return STATUS_INVALID_PARAMETER;
    }
}

/* ===== BufferPushEvent ===== */

VOID
BufferPushEvent(
    _In_ const DELETE_NOTIFICATION* Event
    )
{
    KIRQL oldIrql;

    KeAcquireSpinLock(&g_Context.BufferLock, &oldIrql);

    if (g_Context.EventCount >= MAX_PENDING_EVENTS) {
        /* Buffer full - drop the oldest event by advancing tail */
        g_Context.EventTail = (g_Context.EventTail + 1) % MAX_PENDING_EVENTS;
        g_Context.EventCount--;
        InterlockedIncrement(&g_Context.DroppedEvents);
    }

    RtlCopyMemory(
        &g_Context.Events[g_Context.EventHead],
        Event,
        sizeof(DELETE_NOTIFICATION)
        );

    g_Context.EventHead = (g_Context.EventHead + 1) % MAX_PENDING_EVENTS;
    g_Context.EventCount++;

    KeReleaseSpinLock(&g_Context.BufferLock, oldIrql);
}
