#ifndef ERROR_H
#define ERROR_H

#include <stdio.h>
#include <errno.h>
#include <string.h>

/* Error code definitions */
typedef enum {
    ERR_OK = 0,                 /* success */
    ERR_NOMEM = -1,             /* out of memory */
    ERR_INVALID_PARAM = -2,     /* invalid parameter */
    ERR_FILE_NOT_FOUND = -3,    /* file not found */
    ERR_FILE_READ = -4,         /* file read error */
    ERR_FILE_WRITE = -5,        /* file write error */
    ERR_FILE_FORMAT = -6,       /* file format error */
    ERR_INDEX_OUT_OF_BOUNDS = -7, /* index out of bounds */
    ERR_NOT_FOUND = -8,         /* not found */
    ERR_ALREADY_EXISTS = -9,    /* already exists */
    ERR_NOT_INITIALIZED = -10,  /* not initialized */
    ERR_OPERATION_FAILED = -11, /* operation failed */
    ERR_UNSUPPORTED = -12,      /* unsupported operation */
    ERR_INTERNAL = -13          /* internal error */
} ErrorCode;

/* Error info structure */
typedef struct {
    ErrorCode code;             /* error code */
    char message[256];          /* error message */
    char file[128];             /* file where the error occurred */
    int line;                   /* line number where the error occurred */
    int sys_errno;              /* system errno (if applicable) */
} ErrorInfo;

/* Global error info (thread-local storage in multi-threaded environments) */
extern __thread ErrorInfo g_last_error;

/* =============== Error Handling Macros =============== */

/* Set error info */
#define SET_ERROR(error_code, fmt, ...) \
    do { \
        g_last_error.code = (error_code); \
        snprintf(g_last_error.message, sizeof(g_last_error.message), fmt, ##__VA_ARGS__); \
        snprintf(g_last_error.file, sizeof(g_last_error.file), "%s", __FILE__); \
        g_last_error.line = __LINE__; \
        g_last_error.sys_errno = errno; \
    } while(0)

/* Set system error (includes errno info) */
#define SET_SYS_ERROR(error_code, fmt, ...) \
    do { \
        g_last_error.code = (error_code); \
        snprintf(g_last_error.message, sizeof(g_last_error.message), \
                fmt ": %s", ##__VA_ARGS__, strerror(errno)); \
        snprintf(g_last_error.file, sizeof(g_last_error.file), "%s", __FILE__); \
        g_last_error.line = __LINE__; \
        g_last_error.sys_errno = errno; \
    } while(0)

/* Check if pointer is NULL; if so, set error and return */
#define CHECK_NULL_RETURN(ptr, ret_val, fmt, ...) \
    do { \
        if ((ptr) == NULL) { \
            SET_ERROR(ERR_NOMEM, fmt, ##__VA_ARGS__); \
            return (ret_val); \
        } \
    } while(0)

/* Check condition; if not met, set error and return */
#define CHECK_CONDITION_RETURN(cond, ret_val, error_code, fmt, ...) \
    do { \
        if (!(cond)) { \
            SET_ERROR(error_code, fmt, ##__VA_ARGS__); \
            return (ret_val); \
        } \
    } while(0)

/* Print error info to stderr */
#define PRINT_ERROR() \
    do { \
        if (g_last_error.code != ERR_OK) { \
            fprintf(stderr, "[ERROR] %s:%d: %s (code=%d)\n", \
                   g_last_error.file, g_last_error.line, \
                   g_last_error.message, g_last_error.code); \
        } \
    } while(0)

/* Print detailed error info */
#define PRINT_ERROR_VERBOSE() \
    do { \
        if (g_last_error.code != ERR_OK) { \
            fprintf(stderr, "=== Error Details ===\n"); \
            fprintf(stderr, "Code: %d\n", g_last_error.code); \
            fprintf(stderr, "Message: %s\n", g_last_error.message); \
            fprintf(stderr, "Location: %s:%d\n", g_last_error.file, g_last_error.line); \
            if (g_last_error.sys_errno != 0) { \
                fprintf(stderr, "System errno: %d (%s)\n", \
                       g_last_error.sys_errno, strerror(g_last_error.sys_errno)); \
            } \
            fprintf(stderr, "====================\n"); \
        } \
    } while(0)

/* =============== Error Handling Functions =============== */

/**
 * Initialize error system
 */
void error_init(void);

/**
 * Clear error info
 */
void error_clear(void);

/**
 * Get the last error code
 * @return error code
 */
ErrorCode error_get_code(void);

/**
 * Get the last error message
 * @return error message string
 */
const char* error_get_message(void);

/**
 * Get description for an error code
 * @param code error code
 * @return description string
 */
const char* error_code_to_string(ErrorCode code);

/**
 * Print current error info
 * @param stream output stream (stdout or stderr)
 */
void error_print(FILE* stream);

/**
 * Print detailed error info
 * @param stream output stream
 */
void error_print_verbose(FILE* stream);

/**
 * Check if there is an error
 * @return 1 if error, 0 if no error
 */
int error_has_error(void);

#endif /* ERROR_H */

