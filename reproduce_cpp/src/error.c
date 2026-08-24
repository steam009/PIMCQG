#include "../include/error.h"
#include <string.h>

/* Global error info (using thread-local storage) */
__thread ErrorInfo g_last_error = {
    .code = ERR_OK,
    .message = "",
    .file = "",
    .line = 0,
    .sys_errno = 0
};

void error_init(void) {
    error_clear();
}

void error_clear(void) {
    g_last_error.code = ERR_OK;
    g_last_error.message[0] = '\0';
    g_last_error.file[0] = '\0';
    g_last_error.line = 0;
    g_last_error.sys_errno = 0;
}

ErrorCode error_get_code(void) {
    return g_last_error.code;
}

const char* error_get_message(void) {
    return g_last_error.message;
}

const char* error_code_to_string(ErrorCode code) {
    switch (code) {
        case ERR_OK:
            return "Success";
        case ERR_NOMEM:
            return "Out of memory";
        case ERR_INVALID_PARAM:
            return "Invalid parameter";
        case ERR_FILE_NOT_FOUND:
            return "File not found";
        case ERR_FILE_READ:
            return "File read error";
        case ERR_FILE_WRITE:
            return "File write error";
        case ERR_FILE_FORMAT:
            return "File format error";
        case ERR_INDEX_OUT_OF_BOUNDS:
            return "Index out of bounds";
        case ERR_NOT_FOUND:
            return "Not found";
        case ERR_ALREADY_EXISTS:
            return "Already exists";
        case ERR_NOT_INITIALIZED:
            return "Not initialized";
        case ERR_OPERATION_FAILED:
            return "Operation failed";
        case ERR_UNSUPPORTED:
            return "Unsupported operation";
        case ERR_INTERNAL:
            return "Internal error";
        default:
            return "Unknown error";
    }
}

void error_print(FILE* stream) {
    if (!stream) stream = stderr;
    
    if (g_last_error.code != ERR_OK) {
        fprintf(stream, "[ERROR] %s:%d: %s (code=%d - %s)\n",
               g_last_error.file,
               g_last_error.line,
               g_last_error.message,
               g_last_error.code,
               error_code_to_string(g_last_error.code));
    }
}

void error_print_verbose(FILE* stream) {
    if (!stream) stream = stderr;
    
    if (g_last_error.code != ERR_OK) {
        fprintf(stream, "╔══════════════════════════════════════════════════════════╗\n");
        fprintf(stream, "║                    ERROR DETAILS                         ║\n");
        fprintf(stream, "╠══════════════════════════════════════════════════════════╣\n");
        fprintf(stream, "║ Code:     %d (%s)\n", 
               g_last_error.code, error_code_to_string(g_last_error.code));
        fprintf(stream, "║ Message:  %s\n", g_last_error.message);
        fprintf(stream, "║ Location: %s:%d\n", g_last_error.file, g_last_error.line);
        
        if (g_last_error.sys_errno != 0) {
            fprintf(stream, "║ Errno:    %d (%s)\n",
                   g_last_error.sys_errno, strerror(g_last_error.sys_errno));
        }
        
        fprintf(stream, "╚══════════════════════════════════════════════════════════╝\n");
    }
}

int error_has_error(void) {
    return g_last_error.code != ERR_OK;
}

