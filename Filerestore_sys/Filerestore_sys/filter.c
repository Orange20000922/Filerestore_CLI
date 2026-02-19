/*
 * filter.c - PreSetInformation callback and file metadata capture
 */

#include "driver.h"

/* Forward declarations */
static VOID
CaptureDeleteEvent(
    _Inout_ PFLT_CALLBACK_DATA Data,
    _In_ PCFLT_RELATED_OBJECTS FltObjects,
    _In_ FILE_INFORMATION_CLASS InfoClass
    );

static NTSTATUS
QueryFileLCNs(
    _In_ PCFLT_RELATED_OBJECTS FltObjects,
    _Inout_ PDELETE_NOTIFICATION Notification
    );

/* ===== PreSetInformation callback ===== */

FLT_PREOP_CALLBACK_STATUS
PreSetInformation(
    _Inout_ PFLT_CALLBACK_DATA Data,
    _In_ PCFLT_RELATED_OBJECTS FltObjects,
    _Flt_CompletionContext_Outptr_ PVOID *CompletionContext
    )
{
    FILE_INFORMATION_CLASS infoClass;

    *CompletionContext = NULL;

    /* Check if monitoring is active */
    if (!g_Context.MonitorActive) {
        return FLT_PREOP_SUCCESS_NO_CALLBACK;
    }

    infoClass = Data->Iopb->Parameters.SetFileInformation.FileInformationClass;

    switch (infoClass) {

    case FileDispositionInformation:
    {
        PFILE_DISPOSITION_INFORMATION dispInfo;
        dispInfo = (PFILE_DISPOSITION_INFORMATION)
            Data->Iopb->Parameters.SetFileInformation.InfoBuffer;
        if (dispInfo == NULL || !dispInfo->DeleteFile) {
            return FLT_PREOP_SUCCESS_NO_CALLBACK;
        }
        break;
    }

    case FileDispositionInformationEx:
    {
        PFILE_DISPOSITION_INFORMATION_EX dispInfoEx;
        dispInfoEx = (PFILE_DISPOSITION_INFORMATION_EX)
            Data->Iopb->Parameters.SetFileInformation.InfoBuffer;
        if (dispInfoEx == NULL ||
            !(dispInfoEx->Flags & FILE_DISPOSITION_DELETE)) {
            return FLT_PREOP_SUCCESS_NO_CALLBACK;
        }
        break;
    }

    default:
        /* Not a delete operation */
        return FLT_PREOP_SUCCESS_NO_CALLBACK;
    }

    /* Capture metadata before the delete reaches NTFS */
    CaptureDeleteEvent(Data, FltObjects, infoClass);

    /* Never block the original operation */
    return FLT_PREOP_SUCCESS_NO_CALLBACK;
}

/* ===== CaptureDeleteEvent ===== */

static VOID
CaptureDeleteEvent(
    _Inout_ PFLT_CALLBACK_DATA Data,
    _In_ PCFLT_RELATED_OBJECTS FltObjects,
    _In_ FILE_INFORMATION_CLASS InfoClass
    )
{
    DELETE_NOTIFICATION notif;
    PFLT_FILE_NAME_INFORMATION nameInfo = NULL;
    FILE_STANDARD_INFORMATION stdInfo;
    FILE_BASIC_INFORMATION basicInfo;
    NTSTATUS status;
    USHORT copyLen;

    UNREFERENCED_PARAMETER(InfoClass);

    RtlZeroMemory(&notif, sizeof(notif));

    __try {

        /* 1. Get file name */
        status = FltGetFileNameInformation(
            Data,
            FLT_FILE_NAME_NORMALIZED | FLT_FILE_NAME_QUERY_DEFAULT,
            &nameInfo
            );

        if (NT_SUCCESS(status)) {
            status = FltParseFileNameInformation(nameInfo);
            if (NT_SUCCESS(status)) {
                copyLen = nameInfo->Name.Length;
                if (copyLen > (MAX_FILE_PATH_CHARS - 1) * sizeof(WCHAR)) {
                    copyLen = (MAX_FILE_PATH_CHARS - 1) * sizeof(WCHAR);
                }
                RtlCopyMemory(notif.FileName, nameInfo->Name.Buffer, copyLen);
                notif.FileNameLength = copyLen;
                notif.FileName[copyLen / sizeof(WCHAR)] = L'\0';
            }
            FltReleaseFileNameInformation(nameInfo);
            nameInfo = NULL;
        }

        /* 2. Get file size */
        status = FltQueryInformationFile(
            FltObjects->Instance,
            FltObjects->FileObject,
            &stdInfo,
            sizeof(stdInfo),
            FileStandardInformation,
            NULL
            );

        if (NT_SUCCESS(status)) {
            notif.FileSize = stdInfo.EndOfFile;
            notif.AllocationSize = stdInfo.AllocationSize;
        }

        /* 3. Get timestamps */
        status = FltQueryInformationFile(
            FltObjects->Instance,
            FltObjects->FileObject,
            &basicInfo,
            sizeof(basicInfo),
            FileBasicInformation,
            NULL
            );

        if (NT_SUCCESS(status)) {
            notif.CreationTime = basicInfo.CreationTime;
            notif.LastWriteTime = basicInfo.LastWriteTime;
        }

        /* 4. Get LCN mapping (may fail for resident files - that's OK) */
        QueryFileLCNs(FltObjects, &notif);

        /* 5. Fill remaining fields */
        notif.ProcessId = FltGetRequestorProcessId(Data);
        notif.DeleteType = DELETE_TYPE_PERMANENT;
        KeQuerySystemTimePrecise(&notif.Timestamp);

        /* 6. Push to ring buffer */
        BufferPushEvent(&notif);
        InterlockedIncrement(&g_Context.TotalEvents);
    }
    __except (EXCEPTION_EXECUTE_HANDLER) {
        /* Silent failure - never interfere with the original delete */
        if (nameInfo != NULL) {
            FltReleaseFileNameInformation(nameInfo);
        }
    }
}

/* ===== QueryFileLCNs ===== */

static NTSTATUS
QueryFileLCNs(
    _In_ PCFLT_RELATED_OBJECTS FltObjects,
    _Inout_ PDELETE_NOTIFICATION Notification
    )
{
    NTSTATUS status;
    STARTING_VCN_INPUT_BUFFER vcnInput;
    PRETRIEVAL_POINTERS_BUFFER rpBuf = NULL;
    ULONG rpBufSize = 4096;
    ULONG returned;
    ULONG i;
    ULONG count;
    LARGE_INTEGER prevVcn;

    vcnInput.StartingVcn.QuadPart = 0;

    /* Allocate initial buffer */
    rpBuf = (PRETRIEVAL_POINTERS_BUFFER)ExAllocatePool2(
        POOL_FLAG_NON_PAGED,
        rpBufSize,
        POOL_TAG
        );

    if (rpBuf == NULL) {
        return STATUS_INSUFFICIENT_RESOURCES;
    }

    /* Query retrieval pointers */
    status = FltFsControlFile(
        FltObjects->Instance,
        FltObjects->FileObject,
        FSCTL_GET_RETRIEVAL_POINTERS,
        &vcnInput,
        sizeof(vcnInput),
        rpBuf,
        rpBufSize,
        &returned
        );

    if (status == STATUS_BUFFER_OVERFLOW) {
        /* Retry with larger buffer */
        ExFreePoolWithTag(rpBuf, POOL_TAG);
        rpBufSize = 16384;
        rpBuf = (PRETRIEVAL_POINTERS_BUFFER)ExAllocatePool2(
            POOL_FLAG_NON_PAGED,
            rpBufSize,
            POOL_TAG
            );
        if (rpBuf == NULL) {
            return STATUS_INSUFFICIENT_RESOURCES;
        }

        status = FltFsControlFile(
            FltObjects->Instance,
            FltObjects->FileObject,
            FSCTL_GET_RETRIEVAL_POINTERS,
            &vcnInput,
            sizeof(vcnInput),
            rpBuf,
            rpBufSize,
            &returned
            );
    }

    if (!NT_SUCCESS(status)) {
        /* Resident files or other failures - LCNCount stays 0 */
        ExFreePoolWithTag(rpBuf, POOL_TAG);
        return status;
    }

    /* Parse extents */
    count = rpBuf->ExtentCount;
    if (count > MAX_LCN_ENTRIES) {
        count = MAX_LCN_ENTRIES;
    }

    prevVcn = rpBuf->StartingVcn;
    for (i = 0; i < count; i++) {
        LARGE_INTEGER nextVcn = rpBuf->Extents[i].NextVcn;
        LARGE_INTEGER lcn = rpBuf->Extents[i].Lcn;

        /* LCN == -1 means sparse/unallocated extent, skip */
        if (lcn.QuadPart != -1) {
            Notification->LCNEntries[Notification->LCNCount].StartLCN =
                (ULONGLONG)lcn.QuadPart;
            Notification->LCNEntries[Notification->LCNCount].ClusterCount =
                (ULONG)(nextVcn.QuadPart - prevVcn.QuadPart);
            Notification->LCNEntries[Notification->LCNCount].Reserved = 0;
            Notification->LCNCount++;

            if (Notification->LCNCount >= MAX_LCN_ENTRIES) {
                break;
            }
        }

        prevVcn = nextVcn;
    }

    ExFreePoolWithTag(rpBuf, POOL_TAG);
    return STATUS_SUCCESS;
}
