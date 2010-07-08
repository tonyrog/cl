#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <errno.h>
#include <sys/socket.h>
#ifdef DARWIN
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#if WORDSIZE==32
#include "config.32.h"
#elif WORDSIZE==64
#include "config.64.h"
#else
#error "WORDSIZE not defined"
#endif

#include "erl_driver.h"

#include "cbufv2.h"
#include "cl_hash.h"

#define ECL_REPLY_TYPE CBUF_FLAG_PUT_ETF

#define ECL_USE_ASYNC

// Tricky and buggy
// #define ASYNC_BUILD_PROGRAM
// #define ASYNC_CONTEXT_NOTIFY


#ifdef DEBUG
// #define CBUF_DBG(buf,msg) cbuf_print((buf),(msg))
#define CBUF_DBG(buf,msg)
#define DBG(...) cl_emit_error(__FILE__,__LINE__,__VA_ARGS__)
#else
#define CBUF_DBG(buf,msg)
#define DBG(...)
#endif

// debug async events
#define A_DBG(...) DBG(__VA_ARGS__)


typedef int (*get_fn_t)(cbuf_t*,void*,void*);

// Type map to external communication
#if SIZEOF_SHORT == 2
#define SHORT              INT16
#define USHORT             UINT16
#elif SIZEOF_SHORT == 4
#define SHORT              INT32
#define USHORT             UINT32
#endif

#if SIZEOF_INT == 4
#define INT                INT32
#define UINT               UINT32
#elif SIZEOF_INT == 8
#define INT                INT64
#define UINT               UINT64
#endif

#if SIZEOF_LONG == 4
#define LONG               UINT32
#define ULONG              UINT32
#elif SIZEOF_LONG == 8
#define LONG               UINT64
#define ULONG              UINT64
#endif

#if SIZEOF_SIZE_T == 4
#define SIZE_T             UINT32
static inline int put_size(cbuf_t* out, size_t sz)
{
    u_int64_t value = (u_int64_t) sz;
    return cbuf_put_uint64(out, value);
}

static inline int get_size(cbuf_t* in, size_t* ptr)
{
    u_int64_t value;
    if (!get_uint64(in, &value))
	return 0;
    *ptr = (size_t) value;
    return 1;
}

#elif SIZEOF_SIZE_T == 8
#define SIZE_T             UINT64

static inline int put_size(cbuf_t* out, size_t sz)
{
    return cbuf_put_uint64(out, (u_int64_t)sz);
}

static inline int get_size(cbuf_t* in, size_t* ptr)
{
    return get_uint64(in, (u_int64_t*)ptr);
}

#endif

#if SIZEOF_VOID_P == 4
typedef u_int32_t          pointer_t;
static inline int put_pointer(cbuf_t* out, pointer_t ptr)
{
    u_int64_t value = (u_int64_t) ptr;
    return cbuf_put_uint64(out, value);
}
static inline int get_pointer(cbuf_t* in, pointer_t* ptr)
{
    u_int64_t value;
    if (!get_uint64(in, &value))
	return 0;
    *ptr = (pointer_t) value;
    return 1;
}
#elif SIZEOF_VOID_P == 8
typedef u_int64_t          pointer_t;

static inline int put_pointer(cbuf_t* out, pointer_t ptr)
{
    return cbuf_put_uint64(out, ptr);
}
static inline int get_pointer(cbuf_t* in, pointer_t* ptr)
{
    return get_uint64(in, ptr);
}
#else
#error "check configure, unable to determine SIZEOF_VOID_P"
#endif

/* convert object to handle (just cast to handle setKernelArg) */
#define EPTR_HANDLE(ptr) ((pointer_t)(ptr))

#define STRING             STRING4

#define OCL_CHAR           INT8
#define OCL_UCHAR          UINT8
#define OCL_SHORT          INT16
#define OCL_USHORT         UINT16
#define OCL_INT            INT32
#define OCL_UINT           UINT32
#define OCL_LONG           INT64
#define OCL_ULONG          UINT64
#define OCL_HALF           UINT16
#define OCL_FLOAT          FLOAT32
#define OCL_DOUBLE         FLOAT64
#define OCL_BOOL           OCL_UINT //  not always same size as in kernel
#define OCL_STRING         STRING4
#define OCL_BITFIELD       OCL_ULONG
#define OCL_POINTER        POINTER
#define OCL_SIZE           SIZE_T
#define OCL_HANDLE         HANDLE

#define OCL_DEVICE_TYPE              OCL_BITFIELD
#define OCL_DEVICE_FP_CONFIG         OCL_BITFIELD
#define OCL_DEVICE_GLOBAL_MEM_CACHE_TYPE OCL_BITFIELD // ?
#define OCL_PLATFORM_INFO            OCL_UINT
#define OCL_DEVICE_INFO              OCL_UINT
#define OCL_DEVICE_GLOBAL_MEM_CACHE_TYPE  OCL_BITFIELD
#define OCL_DEVICE_FP_CONFIG         OCL_BITFIELD
#define OCL_DEVICE_EXEC_CAPABILITIES OCL_BITFIELD
#define OCL_QUEUE_PROPERTIES         OCL_BITFIELD
#define OCL_DEVICE_LOCAL_MEM_TYPE    OCL_BITFIELD
#define OCL_PLATFORM_ID              OCL_POINTER
#define OCL_MEM_OBJECT_TYPE          OCL_UINT
#define OCL_MEM_FLAGS                OCL_BITFIELD
#define OCL_SAMPLER_ADDRESSING_MODE  OCL_UINT
#define OCL_SAMPLER_FILTER_MODE      OCL_UINT
#define OCL_BUILD_STATUS             OCL_INT

#define MAX_INFO_SIZE   256    // ulong (2K or 4K buffer)
#define MAX_DEVICES     128   
#define MAX_PLATFORMS   128   
#define MAX_OPTION_LIST 1024
#define MAX_KERNEL_NAME 1024
#define MAX_KERNELS     1024
#define MAX_SOURCES     128
#define MAX_WAIT_LIST   128
#define MAX_WORK_SIZE   3

// COMMANDS  (cl_drv_ctl)
#define ECL_NOOP                         0x01
#define ECL_GET_PLATFORM_IDS             0x02
#define ECL_GET_DEVICE_IDS               0x03
#define ECL_GET_PLATFORM_INFO            0x04
#define ECL_GET_DEVICE_INFO              0x05
#define ECL_CREATE_CONTEXT               0x06
#define ECL_RELEASE_CONTEXT              0x07
#define ECL_RETAIN_CONTEXT               0x08
#define ECL_GET_CONTEXT_INFO             0x09
#define ECL_CREATE_QUEUE                 0x0A
#define ECL_RETAIN_QUEUE                 0x0B
#define ECL_RELEASE_QUEUE                0x0C
#define ECL_GET_QUEUE_INFO               0x0D
#define ECL_SET_QUEUE_PROPERTY           0x0E
#define ECL_CREATE_BUFFER                0x0F
#define ECL_ENQUEUE_READ_BUFFER          0x10
#define ECL_ENQUEUE_WRITE_BUFFER         0x11
#define ECL_ENQUEUE_COPY_BUFFER          0x12
#define ECL_RETAIN_MEM_OBJECT            0x13
#define ECL_RELEASE_MEM_OBJECT           0x14
#define ECL_CREATE_IMAGE2D               0x15
#define ECL_CREATE_IMAGE3D               0x16
#define ECL_GET_SUPPORTED_IMAGE_FORMATS  0x17
#define ECL_ENQUEUE_READ_IMAGE           0x18
#define ECL_ENQUEUE_WRITE_IMAGE          0x19
#define ECL_ENQUEUE_COPY_IMAGE           0x1A
#define ECL_ENQUEUE_COPY_IMAGE_TO_BUFFER 0x1B
#define ECL_ENQUEUE_COPY_BUFFER_TO_IMAGE 0x1C
#define ECL_ENQUEUE_MAP_BUFFER           0x1D
#define ECL_ENQUEUE_MAP_IMAGE            0x1E
#define ECL_ENQUEUE_UNMAP_MEM_OBEJCT     0x1F
#define ECL_GET_MEM_OBJECT_INFO          0x20
#define ECL_GET_IMAGE_INFO               0x21
#define ECL_CREATE_SAMPLER               0x22
#define ECL_RETAIN_SAMPLER               0x23
#define ECL_RELEASE_SAMPLER              0x24
#define ECL_GET_SAMPLER_INFO             0x25
#define ECL_CREATE_PROGRAM_WITH_SOURCE   0x26
#define ECL_CREATE_PROGRAM_WITH_BINARY   0x27
#define ECL_RELEASE_PROGRAM              0x28
#define ECL_RETAIN_PROGRAM               0x29
#define ECL_BUILD_PROGRAM                0x2A
#define ECL_UNLOAD_COMPILER              0x2B
#define ECL_GET_PROGRAM_INFO             0x2C
#define ECL_CREATE_KERNEL                0x2D
#define ECL_CREATE_KERNELS_IN_PROGRAM    0x2E
#define ECL_RETAIN_KERNEL                0x2F
#define ECL_RELEASE_KERNEL               0x30
#define ECL_SET_KERNEL_ARG               0x31
#define ECL_GET_KERNEL_INFO              0x32
#define ECL_GET_PROGRAM_BUILD_INFO       0x33
#define ECL_RETAIN_EVENT                 0x34
#define ECL_RELEASE_EVENT                0x35
#define ECL_GET_EVENT_INFO               0x36
#define ECL_GET_KERNEL_WORKGROUP_INFO    0x37
#define ECL_ENQUEUE_ND_RANGE_KERNEL      0x38
#define ECL_ENQUEUE_TASK                 0x39
#define ECL_FLUSH                        0x3A
#define ECL_FINISH                       0x3B
#define ECL_ENQUEUE_MARKER               0x3C
#define ECL_ENQUEUE_WAIT_FOR_EVENT       0x3D
#define ECL_ENQUEUE_BARRIER              0x3E
#define ECL_SET_KERNEL_ARG_POINTER_T     0x3F
#define ECL_SET_KERNEL_ARG_SIZE_T        0x40

/*
 * Environment keeps track on all allocated objects
 * and has a protective hash layer to check valididity
 * of object pointers. The driver will return a slightly
 * change pointer (down shift 2 bits, zeros anyway) as an
 * integer reference. Then the object is stored in a hash
 * table. The native OpenCL object pointer is used as the
 * key and the handle.
 * 
 */

typedef struct {
    char*          info_name;   // Display name
    cl_uint        info_id;     // Information
    bool           is_array;    // return type is a vector of data
    unsigned char  info_type;   // octet_buffer.h type
    unsigned char  extern_type; // octet_buffer.h type
    void*          extern_info; // Encode/Decode enum/bitfields
} ecl_info_t;

typedef struct {
    char*          key;
    u_int64_t      value;
} ecl_kv_t;

typedef enum {
    ECL_COMMAND_WAIT_STATUS=1,  // wait for completion
    ECL_COMMAND_WAIT_BIN=2,     // wait for completion, return binary
    ECL_COMMAND_FINISH=3        // wait for all events to complete
} ecl_command_type_t;

const char* ecl_command_name[8] =
{ "null", 
  "wait_status",
  "wait_bin",
  "finish",
  "???",
  "???",
  "???",
  "???"
};

#define LCAST(n) ((long)(n))


typedef struct {
    ecl_command_type_t type;    // command type
    ErlDrvTermData     caller;  // The caller that needs response
    u_int32_t          eref;    // Event reference (finish)
    union {
	cl_event         event;   // Event argument
	cl_command_queue queue;
    };
    ErlDrvBinary* bin;          // optional binary argument
} ecl_command_t;

typedef enum {
    ECL_RESPONSE_NONE=0,
    ECL_RESPONSE_EVENT_STATUS=1,
    ECL_RESPONSE_EVENT_BIN=2,
    ECL_RESPONSE_FINISH=3,
    ECL_RESPONSE_BUILD=4,
    ECL_RESPONSE_CONTEXT=5
} ecl_response_type_t;

const char* ecl_response_name[8] =
{ "null", 
  "event_status",
  "event_bin",
  "finish",
  "build",
  "context",
  "???",
  "???"
};

typedef struct {
    ecl_response_type_t  type;    // Response type
    ErlDrvTermData       caller;  // The caller that needs response
    u_int32_t            eref;    // async reference
    int                  err;     // reply error
    cl_int               status;  // exeuction status
    union {
	cl_event         event;   // EVENT argument
	cl_command_queue queue;   // QUEUE argument
	cl_program       program; // PROGRAM argument
	char*            errinfo; // Context error
    };
    ErlDrvBinary* bin;            // optionsl binary argument data
} ecl_response_t;

typedef union {
    ecl_command_t  command;
    ecl_response_t response;
} ecl_async_t;

#ifdef ECL_USE_ASYNC
static void ecl_async_invoke(ecl_async_t*);
static void ecl_async_free(ecl_async_t*);
#endif

/* environment */
typedef struct ocl_env {
    ErlDrvPort  port;   // Port reference
    lhash_t     ref;    // NativePointer => EclObject -> NativPointer
    ErlDrvTid   tid;    // Event thread dispatcher
    int         evt[2]; // Thread events evt[0]=main size, evt[1]=thread side
    u_int32_t   eref;   // event reference for event replies
} ecl_env_t;

typedef enum
{
    NO_TYPE       = 0,
    PLATFORM_TYPE = 1,  // special
    DEVICE_TYPE   = 2,  // special
    CONTEXT_TYPE  = 3,
    QUEUE_TYPE    = 4,
    MEM_TYPE      = 5,
    SAMPLER_TYPE  = 6,
    PROGRAM_TYPE  = 7,
    KERNEL_TYPE   = 8,
    EVENT_TYPE    = 9,
} ecl_object_type_t;

const char* ecl_type_name[] =
{  "NONE",
   "PLATFORM",
   "DEVICE",
   "CONTEXT",
   "QUEUE",
   "MEMOBJECT",
   "SAMPLER",
   "PROGRAM",
   "KERNEL",
   "EVENT"
};

struct _ecl_object_t;

typedef cl_int (*retain_fn)(void*);
typedef cl_int (*release_fn)(void*);
typedef cl_int (*info_fn)(void* ptr, cl_uint param_name, 
			  size_t param_value_size,
			  void* param_value, size_t* param_value_size_ret);

typedef struct _ecl_class_t {
    ecl_object_type_t type;
    retain_fn      retain;
    release_fn     release;
    info_fn        info;
    cl_uint        info_len;
    ecl_info_t*    info_vec;
} ecl_class_t;

/* generic object */
typedef struct _ecl_object_t {
    lhash_bucket_t    hbucket;
    ecl_class_t*      cl;
    unsigned int      refc;
    ecl_env_t*           env;
    union {
	cl_platform_id   platform;
	cl_device_id     device;
	cl_context       context;
	cl_command_queue queue;
	cl_mem           mem;
	cl_sampler       sampler;
	cl_program       program;
	cl_kernel        kernel;
	cl_event         event;
	void*            opaque;
    };
} ecl_object_t;


#define ECL_DEVICE_TYPE_DEFAULT      0x00000000
#define ECL_DEVICE_TYPE_CPU          0x00000001
#define ECL_DEVICE_TYPE_GPU          0x00000002
#define ECL_DEVICE_TYPE_ACCELERATOR  0x00000004
#define ECL_DEVICE_TYPE_ALL          0xFFFFFFFF

ecl_kv_t kv_device_type[] = {  // bitfield
    { "cpu",         CL_DEVICE_TYPE_CPU },
    { "gpu",         CL_DEVICE_TYPE_GPU },
    { "accelerator", CL_DEVICE_TYPE_ACCELERATOR },
    { 0, 0}
};

ecl_kv_t kv_fp_config[] = {  // bitfield
    { "denorm",      CL_FP_DENORM },
    { "inf_nan",     CL_FP_INF_NAN },
    { "round_to_nearest", CL_FP_ROUND_TO_NEAREST },
    { "round_to_zero", CL_FP_ROUND_TO_ZERO },
    { "round_to_inf", CL_FP_ROUND_TO_INF },
    { "fma", CL_FP_FMA },
    { 0, 0 }
};

ecl_kv_t kv_mem_cache_type[] = {  // enum
    { "none", CL_NONE },
    { "read_only", CL_READ_ONLY_CACHE },
    { "read_write", CL_READ_WRITE_CACHE },
    { 0, 0 }
};

ecl_kv_t kv_local_mem_type[] = {  // enum
    { "local", CL_LOCAL },
    { "global", CL_GLOBAL },
    { 0, 0 }
};

ecl_kv_t kv_exec_capabilities[] = {  // bit field
    { "kernel", CL_EXEC_KERNEL },
    { "native_kernel", CL_EXEC_NATIVE_KERNEL },
    { 0, 0 }
};

#define ECL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE 0x01
#define ECL_QUEUE_PROFILING_ENABLE              0x02

ecl_kv_t kv_command_queue_properties[] = { // bit field
    { "out_of_order_exec_mode_enable", CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE },
    { "profiling_enable", CL_QUEUE_PROFILING_ENABLE },
    { 0, 0}
};

#define ECL_MEM_READ_WRITE      0x01
#define ECL_MEM_WRITE_ONLY      0x02
#define ECL_MEM_READ_ONLY       0x04
#define ECL_MEM_USE_HOST_PTR    0x08
#define ECL_MEM_ALLOC_HOST_PTR  0x10
#define ECL_MEM_COPY_HOST_PTR   0x20

ecl_kv_t kv_mem_flags[] = { // bit field
    { "read_write", CL_MEM_READ_WRITE },
    { "write_only", CL_MEM_WRITE_ONLY },
    { "read_only",  CL_MEM_READ_ONLY },
    { "use_host_ptr", CL_MEM_USE_HOST_PTR },
    { "alloc_host_ptr", CL_MEM_ALLOC_HOST_PTR },
    { "copy_host_ptr", CL_MEM_COPY_HOST_PTR },
    { 0, 0 }
};

ecl_kv_t kv_mem_object_type[] = { // enum
    { "buffer", CL_MEM_OBJECT_BUFFER },
    { "image2d", CL_MEM_OBJECT_IMAGE2D },
    { "image3d", CL_MEM_OBJECT_IMAGE3D },
    { 0, 0 }
};

ecl_kv_t kv_addressing_mode[] = { // enum
    { "none", CL_ADDRESS_NONE },
    { "clamp_to_eded", CL_ADDRESS_CLAMP_TO_EDGE },
    { "clamp", CL_ADDRESS_CLAMP },
    { "repeat", CL_ADDRESS_REPEAT },
    { 0, 0 }
};

#define ADDRESSING_MODE_NUM ((int)(sizeof(kv_addressing_mode)/sizeof(ecl_kv_t))-1)


ecl_kv_t kv_filter_mode[] = { // enum
    { "nearest", CL_FILTER_NEAREST },
    { "linear",  CL_FILTER_LINEAR },
    { 0, 0 }
};

#define FILTER_MODE_NUM ((int)(sizeof(kv_filter_mode)/sizeof(ecl_kv_t))-1)

ecl_kv_t kv_map_flags[] = { // bitfield
    { "read", CL_MAP_READ },
    { "write", CL_MAP_WRITE },
    { 0, 0 }
};

ecl_kv_t kv_build_status[] = { // enum
    { "success", CL_BUILD_SUCCESS },
    { "none", CL_BUILD_NONE },
    { "error", CL_BUILD_ERROR },
    { "in_progress", CL_BUILD_IN_PROGRESS },
    { 0, 0 }
};

ecl_kv_t kv_command_type[] = { // enum
    { "ndrange_kernel", CL_COMMAND_NDRANGE_KERNEL },
    { "task",           CL_COMMAND_TASK },
    { "native_kernel",  CL_COMMAND_NATIVE_KERNEL },
    { "read_buffer",    CL_COMMAND_READ_BUFFER },
    { "write_buffer",   CL_COMMAND_WRITE_BUFFER },
    { "copy_buffer",    CL_COMMAND_COPY_BUFFER },
    { "read_image",     CL_COMMAND_READ_IMAGE },
    { "write_image",    CL_COMMAND_WRITE_IMAGE },
    { "copy_image",     CL_COMMAND_COPY_IMAGE },
    { "copy_image_to_buffer", CL_COMMAND_COPY_IMAGE_TO_BUFFER },
    { "copy_buffer_to_image", CL_COMMAND_COPY_BUFFER_TO_IMAGE },
    { "map_buffer", CL_COMMAND_MAP_BUFFER },
    { "map_image", CL_COMMAND_MAP_IMAGE },
    { "unmap_mem_object", CL_COMMAND_UNMAP_MEM_OBJECT },
    { "marker", CL_COMMAND_MARKER  },
    { "aquire_gl_objects", CL_COMMAND_ACQUIRE_GL_OBJECTS },
    { "release_gl_objects", CL_COMMAND_RELEASE_GL_OBJECTS },
    { 0, 0}
};

ecl_kv_t kv_execution_status[] = { // enum
    { "complete",   CL_COMPLETE   },   // same as CL_SUCCESS
    { "running",    CL_RUNNING    },
    { "submitted",  CL_SUBMITTED  },
    { "queued",     CL_QUEUED     },
    // the error codes (negative values)
    { "device_not_found", CL_DEVICE_NOT_FOUND },
    { "device_not_available", CL_DEVICE_NOT_AVAILABLE },
    { "compiler_not_available", CL_COMPILER_NOT_AVAILABLE },
    { "mem_object_allocation_failure", CL_MEM_OBJECT_ALLOCATION_FAILURE },
    { "out_of_resources", CL_OUT_OF_RESOURCES },
    { "out_of_host_memory", CL_OUT_OF_HOST_MEMORY },
    { "profiling_info_not_available", CL_PROFILING_INFO_NOT_AVAILABLE },
    { "mem_copy_overlap", CL_MEM_COPY_OVERLAP },
    { "image_format_mismatch", CL_IMAGE_FORMAT_MISMATCH },
    { "image_format_not_supported", CL_IMAGE_FORMAT_NOT_SUPPORTED },
    { "build_program_failure", CL_BUILD_PROGRAM_FAILURE },
    { "map_failure", CL_MAP_FAILURE },
    { "invalid_value", CL_INVALID_VALUE },
    { "invalid_device type", CL_INVALID_DEVICE_TYPE },
    { "invalid_platform", CL_INVALID_PLATFORM },
    { "invalid_device", CL_INVALID_DEVICE },
    { "invalid_context", CL_INVALID_CONTEXT },
    { "invalid_queue_properties", CL_INVALID_QUEUE_PROPERTIES },
    { "invalid_command_queue", CL_INVALID_COMMAND_QUEUE },
    { "invalid_host_ptr", CL_INVALID_HOST_PTR },
    { "invalid_mem_object", CL_INVALID_MEM_OBJECT },
    { "invalid_image_format_descriptor", CL_INVALID_IMAGE_FORMAT_DESCRIPTOR },
    { "invalid_image_size", CL_INVALID_IMAGE_SIZE },
    { "invalid_sampler", CL_INVALID_SAMPLER },
    { "invalid_binary", CL_INVALID_BINARY },
    { "invalid_build_options", CL_INVALID_BUILD_OPTIONS },
    { "invalid_program", CL_INVALID_PROGRAM },
    { "invalid_program_executable", CL_INVALID_PROGRAM_EXECUTABLE },
    { "invalid_kernel_name", CL_INVALID_KERNEL_NAME },
    { "invalid_kernel_definition", CL_INVALID_KERNEL_DEFINITION },
    { "invalid_kernel", CL_INVALID_KERNEL },
    { "invalid_arg_index", CL_INVALID_ARG_INDEX },
    { "invalid_arg_value", CL_INVALID_ARG_VALUE },
    { "invalid_arg_size", CL_INVALID_ARG_SIZE },
    { "invalid_kernel_args", CL_INVALID_KERNEL_ARGS },
    { "invalid_work_dimension", CL_INVALID_WORK_DIMENSION },
    { "invalid_work_group_size", CL_INVALID_WORK_GROUP_SIZE },
    { "invalid_work_item size", CL_INVALID_WORK_ITEM_SIZE },
    { "invalid_global_offset", CL_INVALID_GLOBAL_OFFSET },
    { "invalid_event_wait_list", CL_INVALID_EVENT_WAIT_LIST },
    { "invalid_event", CL_INVALID_EVENT },
    { "invalid_operation", CL_INVALID_OPERATION },
    { "invalid_gl_object", CL_INVALID_GL_OBJECT },
    { "invalid_buffer_size", CL_INVALID_BUFFER_SIZE },
    { "invalid_mip_level", CL_INVALID_MIP_LEVEL },
    { 0, 0 }
};

// Map device info index 0...N => cl_device_info x Data type
ecl_info_t device_info[] = 
{
    /* 00  */  { "type", CL_DEVICE_TYPE, false, OCL_DEVICE_TYPE, BITFIELD, kv_device_type },
    /* 01  */  { "vendor_id", CL_DEVICE_VENDOR_ID, false, OCL_UINT, UINT, 0 },
    /* 02  */  { "max_compute_units", CL_DEVICE_MAX_COMPUTE_UNITS, false, OCL_UINT, UINT, 0 },
    /* 03  */  { "max_work_item_dimensions", CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, false, OCL_UINT, UINT, 0 },
    /* 04  */  { "max_work_group_size", CL_DEVICE_MAX_WORK_GROUP_SIZE, false, OCL_SIZE, SIZE_T, 0 },
    /* 05  */  { "max_work_item_sizes", CL_DEVICE_MAX_WORK_ITEM_SIZES, true, OCL_SIZE, SIZE_T, 0 },
    /* 06  */  { "preferred_vector_width_char", CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, false, OCL_UINT, UINT, 0 },
    /* 07  */  { "preferred_vector_width_short", CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, false, OCL_UINT, UINT, 0 },
    /* 08  */  { "preferred_vector_width_int", CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, false, OCL_UINT, UINT, 0 },
    /* 09  */  { "preferred_vector_width_long", CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, false,OCL_UINT, UINT, 0 },
    /* 0A */  { "preferred_vector_width_float", CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, false, OCL_UINT, UINT, 0 },
    /* 0B */  { "preferred_vector_width_double", CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, false, OCL_UINT, UINT, 0 },
    /* 0C */  { "max_clock_frequency", CL_DEVICE_MAX_CLOCK_FREQUENCY, false, OCL_UINT, UINT, 0 },
    /* 0D */  { "address_bits", CL_DEVICE_ADDRESS_BITS, false, OCL_UINT, UINT, 0 },
    /* 0E */  { "max_read_image_args", CL_DEVICE_MAX_READ_IMAGE_ARGS, false, OCL_UINT, UINT, 0 },
    /* 0F */  { "max_write_image_args", CL_DEVICE_MAX_WRITE_IMAGE_ARGS, false, OCL_UINT, UINT, 0 },
    /* 10 */  { "max_mem_alloc_size", CL_DEVICE_MAX_MEM_ALLOC_SIZE, false, OCL_ULONG, ULONG, 0 },
    /* 11 */  { "image2d_max_width", CL_DEVICE_IMAGE2D_MAX_WIDTH, false, OCL_SIZE, SIZE_T, 0 },
    /* 12 */  { "image2d_max_height", CL_DEVICE_IMAGE2D_MAX_HEIGHT, false, OCL_SIZE, SIZE_T, 0 },
    /* 13 */  { "image3d_max_width", CL_DEVICE_IMAGE3D_MAX_WIDTH, false, OCL_SIZE, SIZE_T, 0 },
    /* 14 */  { "image3d_max_height", CL_DEVICE_IMAGE3D_MAX_HEIGHT, false, OCL_SIZE, SIZE_T, 0 },
    /* 15 */  { "image3d_max_depth", CL_DEVICE_IMAGE3D_MAX_DEPTH, false, OCL_SIZE, SIZE_T, 0 },
    /* 16 */  { "image_support", CL_DEVICE_IMAGE_SUPPORT, false, OCL_BOOL, BOOLEAN, 0 },
    /* 17 */  { "max_parameter_size", CL_DEVICE_MAX_PARAMETER_SIZE, false, OCL_SIZE, SIZE_T, 0 },
    /* 18 */  { "max_samplers", CL_DEVICE_MAX_SAMPLERS, false, OCL_UINT, UINT, 0 },
    /* 19 */  { "mem_base_addr_align", CL_DEVICE_MEM_BASE_ADDR_ALIGN, false, OCL_UINT, UINT, 0 },
    /* 1A */  { "min_data_type_align_size", CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE, false, OCL_UINT, UINT, 0 },
    /* 1B */  { "single_fp_config", CL_DEVICE_SINGLE_FP_CONFIG, false, OCL_DEVICE_FP_CONFIG, BITFIELD, kv_fp_config },
    /* 1C */  { "global_mem_cache_type", CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, false, OCL_DEVICE_GLOBAL_MEM_CACHE_TYPE, ENUM, kv_mem_cache_type },
    /* 1D */  { "global_mem_cacheline_size", CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, false, OCL_UINT, UINT, 0 },
    /* 1E */  { "global_mem_cache_size", CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, false, OCL_ULONG, ULONG, 0 },
    /* 1F */  { "global_mem_size", CL_DEVICE_GLOBAL_MEM_SIZE, false, OCL_ULONG, ULONG, 0 },
    /* 20 */  { "max_constant_buffer_size", CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,  false, OCL_ULONG, ULONG, 0 },
    /* 21 */  { "max_constant_args", CL_DEVICE_MAX_CONSTANT_ARGS, false, OCL_UINT, UINT, 0 },
    /* 22 */  { "local_mem_type", CL_DEVICE_LOCAL_MEM_TYPE, false, OCL_DEVICE_LOCAL_MEM_TYPE, ENUM, kv_local_mem_type },
    /* 23 */  { "local_mem_size", CL_DEVICE_LOCAL_MEM_SIZE,  false, OCL_ULONG, ULONG, 0 },
    /* 24 */  { "error_correction_support", CL_DEVICE_ERROR_CORRECTION_SUPPORT, false,  OCL_BOOL, BOOLEAN, 0 },
    /* 25 */  { "profiling_timer_resolution", CL_DEVICE_PROFILING_TIMER_RESOLUTION, false,  OCL_SIZE, SIZE_T, 0 },
    /* 26 */  { "endian_little", CL_DEVICE_ENDIAN_LITTLE, false, OCL_BOOL, BOOLEAN, 0},
    /* 27 */  { "available", CL_DEVICE_AVAILABLE,  false, OCL_BOOL, BOOLEAN, 0 },
    /* 28 */  { "compiler_available", CL_DEVICE_COMPILER_AVAILABLE, false, OCL_BOOL, BOOLEAN, 0 },
    /* 29 */  { "execution_capabilities", CL_DEVICE_EXECUTION_CAPABILITIES, false, OCL_DEVICE_EXEC_CAPABILITIES, BITFIELD, kv_exec_capabilities },
    /* 2A */  { "queue_properties", CL_DEVICE_QUEUE_PROPERTIES, false, OCL_QUEUE_PROPERTIES, BITFIELD, kv_command_queue_properties },
    /* 2B */  { "name", CL_DEVICE_NAME, false, OCL_STRING, STRING, 0 }, 
    /* 2C */  { "vendor", CL_DEVICE_VENDOR, false, OCL_STRING, STRING, 0 }, 
    /* 2D */  { "driver_version", CL_DRIVER_VERSION, false, OCL_STRING, STRING, 0 },
    /* 2E */  { "profile", CL_DEVICE_PROFILE, false, OCL_STRING, STRING, 0 },
    /* 2F */  { "version", CL_DEVICE_VERSION, false, OCL_STRING, STRING, 0 },
    /* 30 */  { "extensions", CL_DEVICE_EXTENSIONS, false, OCL_STRING, STRING, 0 },
    /* 31 */  { "platform", CL_DEVICE_PLATFORM, false, OCL_PLATFORM_ID, POINTER, 0 }
};

#define ECL_DEVICE_INFO_NUM  ((int)(sizeof(device_info)/sizeof(ecl_info_t)))



// Map device info index 0...N => cl_device_info x Data type
ecl_info_t platform_info[] = 
{
    /* 0  */  { "profile", CL_PLATFORM_PROFILE, false, OCL_STRING, STRING, 0 },
    /* 1  */  { "version", CL_PLATFORM_VERSION, false, OCL_STRING, STRING, 0 },
    /* 2  */  { "name",    CL_PLATFORM_NAME,    false, OCL_STRING, STRING, 0 },
    /* 3  */  { "vendor",  CL_PLATFORM_VENDOR,  false, OCL_STRING, STRING, 0 },
    /* 4  */  { "extensions", CL_PLATFORM_EXTENSIONS, false, OCL_STRING, STRING, 0}
};    

#define ECL_PLATFORM_INFO_NUM ((int)(sizeof(platform_info)/sizeof(ecl_info_t)))

ecl_info_t context_info[] =
{
    /* 0 */ { "reference_count", CL_CONTEXT_REFERENCE_COUNT, false, OCL_UINT, UINT, 0 },
    /* 1 */ { "devices", CL_CONTEXT_DEVICES, true, OCL_HANDLE, POINTER, 0 },
    /* 2 */ { "properties", CL_CONTEXT_PROPERTIES, true, OCL_INT, INT, 0 }
};

#define ECL_CONTEXT_INFO_NUM ((int)(sizeof(context_info)/sizeof(ecl_info_t)))

ecl_info_t queue_info[] = 
{
    /* 0 */ { "context", CL_QUEUE_CONTEXT, false, OCL_HANDLE, POINTER, 0 },
    /* 1 */ { "device",  CL_QUEUE_DEVICE, false, OCL_HANDLE, POINTER, 0 },
    /* 2 */ { "reference_count", CL_QUEUE_REFERENCE_COUNT, false, OCL_UINT, UINT, 0 },
    /* 3 */ { "properties", CL_QUEUE_PROPERTIES, false, OCL_QUEUE_PROPERTIES, BITFIELD, kv_command_queue_properties }
};

#define ECL_QUEUE_INFO_NUM ((int)(sizeof(queue_info)/sizeof(ecl_info_t)))

ecl_info_t mem_info[] =
{
    /* 0 */ { "object_type", CL_MEM_TYPE, false, OCL_MEM_OBJECT_TYPE, ENUM, kv_mem_object_type },
    /* 1 */ { "flags", CL_MEM_FLAGS, false, OCL_MEM_FLAGS, BITFIELD, kv_mem_flags },
    /* 2 */ { "size",  CL_MEM_SIZE,  false, OCL_SIZE, SIZE_T, 0 },
    // FIXME: pointer!! map it ?
    /* 3 */ { "host_ptr", CL_MEM_HOST_PTR, false, OCL_POINTER, POINTER, 0 }, 
    /* 4 */ { "map_count", CL_MEM_MAP_COUNT, false, OCL_UINT, UINT, 0 },
    /* 5 */ { "reference_count", CL_MEM_REFERENCE_COUNT, false, OCL_UINT, UINT, 0 },
    /* 6 */ { "context", CL_MEM_CONTEXT, false, OCL_HANDLE, POINTER, 0 }
};

#define ECL_MEM_INFO_NUM ((int)(sizeof(mem_info)/sizeof(ecl_info_t)))

ecl_info_t sampler_info[] = 
{
    { "reference_count", CL_SAMPLER_REFERENCE_COUNT, false, OCL_UINT, UINT, 0},
    { "context", CL_SAMPLER_CONTEXT, false,  OCL_HANDLE, POINTER, 0 },
    { "normalized_coords", CL_SAMPLER_NORMALIZED_COORDS, false, OCL_BOOL, BOOLEAN, 0 },
    {  "addressing_mode", CL_SAMPLER_ADDRESSING_MODE, false, OCL_SAMPLER_ADDRESSING_MODE, ENUM, kv_addressing_mode },
    { "filter_mode", CL_SAMPLER_FILTER_MODE, false, OCL_SAMPLER_FILTER_MODE, ENUM, kv_filter_mode }
};

#define ECL_SAMPLER_INFO_NUM ((int)(sizeof(sampler_info)/sizeof(ecl_info_t)))

ecl_info_t program_info[] = {
    { "reference_count", CL_PROGRAM_REFERENCE_COUNT, false, OCL_UINT, UINT, 0 },
    { "context", CL_PROGRAM_CONTEXT, false, OCL_HANDLE, POINTER, 0},
    { "num_devices", CL_PROGRAM_NUM_DEVICES, false, OCL_UINT, UINT, 0},
    { "devices", CL_PROGRAM_DEVICES, true, OCL_HANDLE, POINTER, 0 },
    { "source", CL_PROGRAM_SOURCE, false, OCL_STRING, STRING, 0 },
    { "binary_sizes", CL_PROGRAM_BINARY_SIZES, true, OCL_SIZE, SIZE_T, 0 },
    { "binaries", CL_PROGRAM_BINARIES, true, OCL_STRING, STRING, 0 }
};

#define ECL_PROGRAM_INFO_NUM ((int)(sizeof(program_info)/sizeof(ecl_info_t)))

ecl_info_t build_info[] = {
    { "status", CL_PROGRAM_BUILD_STATUS, false, OCL_BUILD_STATUS, ENUM, kv_build_status },
    { "options", CL_PROGRAM_BUILD_OPTIONS, false, OCL_STRING, STRING, 0 },
    { "log", CL_PROGRAM_BUILD_LOG, false, OCL_STRING, STRING, 0 }
};

#define ECL_BUILD_INFO_NUM ((int)(sizeof(build_info)/sizeof(ecl_info_t)))

ecl_info_t kernel_info[] = {
    { "function_name", CL_KERNEL_FUNCTION_NAME, false, OCL_STRING, STRING, 0 },
    { "num_args", CL_KERNEL_NUM_ARGS, false, OCL_UINT, UINT, 0},
    { "reference_count", CL_KERNEL_REFERENCE_COUNT, false, OCL_UINT, UINT, 0 },
    { "context", CL_KERNEL_CONTEXT, false, OCL_HANDLE, POINTER, 0},
    { "program", CL_KERNEL_PROGRAM, false, OCL_HANDLE, POINTER, 0}
};

#define ECL_KERNEL_INFO_NUM ((int)(sizeof(kernel_info)/sizeof(ecl_info_t)))


ecl_info_t workgroup_info[] = {
    { "work_group_size", CL_KERNEL_WORK_GROUP_SIZE, false, OCL_SIZE, SIZE_T, 0 },
    { "compile_work_group_size", CL_KERNEL_COMPILE_WORK_GROUP_SIZE, true, OCL_SIZE, SIZE_T, 0},
    { "local_mem_size", CL_KERNEL_LOCAL_MEM_SIZE, false, OCL_ULONG, ULONG, 0 },
};
#define ECL_WORKGROUP_INFO_NUM ((int)(sizeof(workgroup_info)/sizeof(ecl_info_t)))


ecl_info_t event_info[] = {
    { "command_queue",  CL_EVENT_COMMAND_QUEUE, false, OCL_HANDLE, POINTER, 0},
    { "command_type",   CL_EVENT_COMMAND_TYPE, false,  OCL_UINT, ENUM, kv_command_type },
    { "reference_count", CL_EVENT_REFERENCE_COUNT, false, OCL_UINT, UINT, 0 },
    { "execution_status", CL_EVENT_COMMAND_EXECUTION_STATUS, false, OCL_INT, ENUM, kv_execution_status }
};
#define ECL_EVENT_INFO_NUM ((int)(sizeof(event_info)/sizeof(ecl_info_t)))


ecl_class_t ecl_class_platform =
{
    PLATFORM_TYPE,
    (retain_fn) 0,
    (release_fn) 0,
    (info_fn) clGetPlatformInfo,
    ECL_PLATFORM_INFO_NUM,
    platform_info
};

ecl_class_t ecl_class_device =
{
    DEVICE_TYPE,
    (retain_fn) 0,
    (release_fn) 0,
    (info_fn) clGetDeviceInfo,
    ECL_DEVICE_INFO_NUM,
    device_info
};

ecl_class_t ecl_class_context = 
{
    CONTEXT_TYPE,
    (retain_fn) clRetainContext,
    (release_fn) clReleaseContext,
    (info_fn) clGetContextInfo,
    ECL_CONTEXT_INFO_NUM,
    context_info
};

ecl_class_t ecl_class_queue = 
{
    QUEUE_TYPE,
    (retain_fn) clRetainCommandQueue,
    (release_fn) clReleaseCommandQueue,
    (info_fn) clGetCommandQueueInfo,
    ECL_QUEUE_INFO_NUM,
    queue_info
};

ecl_class_t ecl_class_mem = 
{
    MEM_TYPE,
    (retain_fn) clRetainMemObject,
    (release_fn) clReleaseMemObject,
    (info_fn) clGetMemObjectInfo,
    ECL_MEM_INFO_NUM,
    mem_info
};

ecl_class_t ecl_class_sampler = 
{
    SAMPLER_TYPE,
    (retain_fn) clRetainSampler,
    (release_fn) clReleaseSampler,
    (info_fn) clGetSamplerInfo,
    ECL_SAMPLER_INFO_NUM,
    sampler_info
};

ecl_class_t ecl_class_program = 
{
    PROGRAM_TYPE,
    (retain_fn) clRetainProgram,
    (release_fn) clReleaseProgram,
    (info_fn) clGetProgramInfo,
    ECL_PROGRAM_INFO_NUM,
    program_info
};

ecl_class_t ecl_class_kernel = 
{
    KERNEL_TYPE,
    (retain_fn) clRetainKernel,
    (release_fn) clReleaseKernel,
    (info_fn) clGetKernelInfo,
    ECL_KERNEL_INFO_NUM,
    kernel_info
};

ecl_class_t ecl_class_event = 
{
    EVENT_TYPE,
    (retain_fn) clRetainEvent,
    (release_fn) clReleaseEvent,
    (info_fn) clGetEventInfo,
    ECL_EVENT_INFO_NUM,
    event_info
};

#ifdef DEBUG
#include <stdarg.h>
static void cl_emit_error(char* file, int line, ...);

static void cl_emit_error(char* file, int line, ...)
{
    va_list ap;
    char* fmt;

    va_start(ap, line);
    fmt = va_arg(ap, char*);

    fprintf(stderr, "%s:%d: ", file, line); 
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\r\n");
    va_end(ap);
}
#endif


static inline void ecl_object_destroy(ecl_object_t* obj)
{
    if (obj) {
	// fprintf(stderr, "destroy: %s handle=%d\r\n", 
	//   ecl_type_name[obj->cl->type], EPTR_HANDLE(obj->opaque));
	if (obj->opaque) {
	    if (obj->cl->release)
		obj->cl->release(obj->opaque);
	}
	lhash_erase(&obj->env->ref, (void*)EPTR_HANDLE(obj->opaque));
    }
}

static inline void ecl_object_retain(ecl_object_t* obj)
{
    if (obj) obj->refc++;  // atomic?
}

static inline void ecl_object_release(ecl_object_t* obj)
{
    if (obj) {
	if (obj->refc <= 1) {  // atomic?
	    ecl_object_destroy(obj);
	}
	else
	    obj->refc--;
    }
}

//
// Wrap info call
//
static cl_int ecl_object_info(ecl_object_t* obj,
			      cl_uint param_name, 
			      size_t param_value_size,
			      void* param_value, size_t* param_value_size_ret)
{
    return obj->cl->info(obj->opaque, param_name, 
			 param_value_size, param_value,
			 param_value_size_ret);
}

static lhash_value_t ref_hash(void* key)
{
    return (lhash_value_t) key;
}

static int ref_cmp(void* key, void* data)
{
    if (((pointer_t)key) == EPTR_HANDLE(((ecl_object_t*)data)->opaque))
	return 0;
    return 1;
}

static void ref_release(void *data)
{
    driver_free(data);
}

#if 0 
// Used with lhash_each to display contents of hash table
static void ref_display(lhash_t* lhash, void* elem, void* arg)
{
    ecl_object_t* obj = (ecl_object_t*) elem;
    (void) lhash;
    fprintf((FILE*)arg, "REF: handle=%d, obj=%p\r\n",
	    EPTR_HANDLE(obj->opaque), obj);
}
#endif

// Translate internal pointer to extern object handle
static inline pointer_t ecl_ptr(ecl_env_t* env, void* ptr)
{
    void* key = (void*) EPTR_HANDLE(ptr);
    if (lhash_lookup(&env->ref, key))
	return (pointer_t) key;
    return 0;
}

// Translate ecl_object_t external key 
static inline pointer_t ecl_handle(ecl_object_t* obj)
{
    if (obj)
	return EPTR_HANDLE(obj->opaque);
    return 0;
}

// Lookup object and type check
static inline ecl_object_t* ecl_object(ecl_env_t* env, pointer_t handle, 
				       ecl_object_type_t type)
{
    ecl_object_t* obj = (ecl_object_t*) lhash_lookup(&env->ref,(void*)handle);
    if (obj && (obj->cl->type == type))
	return obj;
    return 0;
}

static inline ecl_object_t* platform_object(ecl_env_t* env, pointer_t handle)
{
    return ecl_object(env, handle, PLATFORM_TYPE);
}

static inline ecl_object_t* device_object(ecl_env_t* env, pointer_t handle)
{
    return ecl_object(env, handle, DEVICE_TYPE);
}

static inline ecl_object_t* context_object(ecl_env_t* env, pointer_t handle)
{
    return ecl_object(env, handle, CONTEXT_TYPE);
}

static inline ecl_object_t* queue_object(ecl_env_t* env, pointer_t handle)
{
    return ecl_object(env, handle, QUEUE_TYPE);
}

static inline ecl_object_t* mem_object(ecl_env_t* env, pointer_t handle)
{
    return ecl_object(env, handle, MEM_TYPE);
}

static inline ecl_object_t* sampler_object(ecl_env_t* env, pointer_t handle)
{
    return ecl_object(env, handle, SAMPLER_TYPE);
}

static inline ecl_object_t* program_object(ecl_env_t* env, pointer_t handle)
{
    return ecl_object(env, handle, PROGRAM_TYPE);
}

static inline ecl_object_t* kernel_object(ecl_env_t* env, pointer_t handle)
{
    return ecl_object(env, handle, KERNEL_TYPE);
}

static inline ecl_object_t* event_object(ecl_env_t* env, pointer_t handle)
{
    return ecl_object(env, handle, EVENT_TYPE);
}

int get_event(cbuf_t* in, void* ptr, ecl_env_t* env)
{
    pointer_t handle;
    ecl_object_t* obj;

    if (!get_pointer(in, &handle))
	return 0;
    if (!(obj = event_object(env, handle)))
	return 0;
    *((cl_event*)ptr) = obj->event;
    return 1;
}

int get_device(cbuf_t* in, void* ptr, ecl_env_t* env)
{
    pointer_t handle;
    ecl_object_t* obj;

    if (!get_pointer(in, &handle))
	return 0;
    if (!(obj = device_object(env, handle)))
	return 0;
    *((cl_device_id*)ptr) = obj->device;
    return 1;
}


ecl_object_t* ecl_object_create(ecl_env_t* env, void* ptr, ecl_class_t* cl)
{
    ecl_object_t* obj;

    if (!(obj = (ecl_object_t*) driver_alloc(sizeof(ecl_object_t))))
	return 0;
    obj->cl      = cl;
    obj->refc    = 1;
    obj->env     = env;
    obj->opaque  = ptr;
    // retain an extra for the mapping entry
    if (cl->retain) cl->retain(ptr);
    // insert in hash table
    lhash_insert_new(&env->ref, (void*)EPTR_HANDLE(ptr), obj);
    // fprintf(stderr, "Object: %s create handle=%d\r\n", 
    //    ecl_type_name[cl->type], EPTR_HANDLE(ptr));
    return obj;
}

// lookup and possibly create a platform object
ecl_object_t* EclPlatform(ecl_env_t* env, cl_platform_id platform)
{
    ecl_object_t* obj = platform_object(env, EPTR_HANDLE(platform));
    if (!obj)
	return ecl_object_create(env,platform,&ecl_class_platform);
    return obj;
}

// lookup and possibly create a platform object
ecl_object_t* EclDevice(ecl_env_t* env, cl_device_id device)
{
    ecl_object_t* obj = device_object(env, EPTR_HANDLE(device));
    if (!obj)
	return ecl_object_create(env,device,&ecl_class_device);
    return obj;
}

ecl_object_t* EclContextCreate(ecl_env_t* env, cl_context context)
{
    return ecl_object_create(env, context, &ecl_class_context);
}

ecl_object_t* EclMemCreate(ecl_env_t* env, cl_mem mem)
{
    return ecl_object_create(env, mem, &ecl_class_mem);
}

ecl_object_t* EclQueueCreate(ecl_env_t* env,cl_command_queue queue)
{
    return ecl_object_create(env, queue, &ecl_class_queue);
}

ecl_object_t* EclSamplerCreate(ecl_env_t* env,cl_sampler sampler)
{
    return ecl_object_create(env, sampler, &ecl_class_sampler);
}

ecl_object_t* EclProgramCreate(ecl_env_t* env,cl_program program)
{
    return ecl_object_create(env, program, &ecl_class_program);
}

ecl_object_t* EclKernelCreate(ecl_env_t* env,cl_kernel kernel)
{
    return ecl_object_create(env, kernel, &ecl_class_kernel);
}

ecl_object_t* EclEventCreate(ecl_env_t* env,cl_event event)
{
    return ecl_object_create(env, event, &ecl_class_event);
}


static int        ecl_drv_init(void);
static void       ecl_drv_finish(void);
static void       ecl_drv_stop(ErlDrvData);
static void       ecl_drv_command(ErlDrvData, char*, int);
static void       ecl_drv_commandv(ErlDrvData, ErlIOVec*);
static void       ecl_drv_input(ErlDrvData, ErlDrvEvent);
static void       ecl_drv_output(ErlDrvData data, ErlDrvEvent event);
static ErlDrvData ecl_drv_start(ErlDrvPort, char* command);
static int        ecl_drv_ctl(ErlDrvData,unsigned int,char*, int,char**,int);
static void       ecl_drv_timeout(ErlDrvData);
static void       ecl_drv_ready_async(ErlDrvData, ErlDrvThreadData);


ErlDrvEntry  ecl_drv = {
    ecl_drv_init,
    ecl_drv_start,
    ecl_drv_stop,
    ecl_drv_command,               /* output */
    ecl_drv_input,                 /* ready_input */
    ecl_drv_output,                /* ready_output */
    "cl_drv",
    ecl_drv_finish,                /* finish */
    NULL,                           /* handle */
    ecl_drv_ctl,
    ecl_drv_timeout,               /* timeout */
    ecl_drv_commandv,
    ecl_drv_ready_async,            /* ready_async */
    NULL,                           /* flush */
    NULL,                           /* call */
    NULL,                           /* event */
    ERL_DRV_EXTENDED_MARKER,
    ERL_DRV_EXTENDED_MAJOR_VERSION,
    ERL_DRV_EXTENDED_MINOR_VERSION,
    ERL_DRV_FLAG_USE_PORT_LOCKING,  /* We really want NO locking! */
    NULL,                           /* handle2 */
    NULL,                           /* process_exit */
    NULL
};


static char* fmt_error(cl_int err)
{
    ecl_kv_t* kv = kv_execution_status;

    while(kv->key) {
	if ((cl_int)kv->value == err)
	    return kv->key;
	kv++;
    }
    return "unknown";
}

static void put_element(cbuf_t* data, u_int8_t tag, void* ptr, ecl_kv_t* kv)
{
    switch(tag) {
    case INT8:  cbuf_put_int8(data, *((cl_char*)ptr));   break;
    case INT16: cbuf_put_int16(data, *((cl_short*)ptr)); break;
    case INT32: cbuf_put_int32(data, *((cl_int*)ptr));   break;
    case INT64: cbuf_put_int64(data, *((cl_long*)ptr));  break;
    case UINT8: cbuf_put_uint8(data, *((cl_uchar*)ptr)); break;
    case UINT16: cbuf_put_uint16(data, *((cl_ushort*)ptr)); break;
    case UINT32: cbuf_put_uint32(data, *((cl_uint*)ptr)); break;
    case UINT64: cbuf_put_uint64(data, *((cl_ulong*)ptr)); break;
    case POINTER: put_pointer(data, *((pointer_t*)ptr)); break;
    case BOOLEAN: cbuf_put_boolean(data, (u_int8_t) *((cl_uint*)ptr)); break;
    case STRING1:
    case STRING4: {
	char* str = ((char*) ptr);
	int   len = strlen(str);
	cbuf_put_string(data, str, len);
	break;
    }
    case BITFIELD: {
	cl_bitfield v = *((cl_bitfield*)ptr);
	ecl_kv_t* kv0 = kv;
	size_t n = 0;

	// count number of elements
	if (v) {
	    while(kv->key) {
		if (kv->value & v)
		    n++;
		kv++;
	    }
	}
	cbuf_put_list_begin(data, n);
	if (v) {
	    kv = kv0;
	    while(kv->key) {
		if (kv->value & v)
		    cbuf_put_atom(data, kv->key);
		kv++;
	    }
	}
	cbuf_put_list_end(data, n);
	break;
    }

    case ENUM: {
	cl_int v = *((cl_int*)ptr);
	while(kv->key) {
	    if (v == (cl_int)kv->value) {
		cbuf_put_atom(data, kv->key);
		break;
	    }
	    kv++;
	}
	if (!kv->key) // not found, put as integer
	    put_element(data, OCL_INT, ptr, 0);
	break;
    }
    default:
	cbuf_put_atom(data, "undefined");
	break;
    }
}

static void put_value(cbuf_t* data, ecl_env_t* env, ecl_info_t* iptr,
		      void* buf, size_t len)
{
    u_int8_t* dptr = (u_int8_t*) buf;

    if (iptr->is_array) {
	// arrays are return as lists of items
	size_t elem_size = cbuf_sizeof(iptr->info_type);
	size_t n = (len / elem_size);
	cbuf_put_list_begin(data, n);
	while (len > 0) {
	    if (iptr->info_type == HANDLE) {
		void* ptr = *((void**) dptr);
		pointer_t handle = ecl_ptr(env, ptr);
		*((void**)dptr) = (void*)handle;
	    }
	    put_element(data, iptr->extern_type, dptr, iptr->extern_info);
	    len -= elem_size;
	    dptr += elem_size;
	}
	cbuf_put_list_end(data, n);
    }
    else {
	if (iptr->info_type == HANDLE) {
	    void* ptr = *((void**) dptr);
	    pointer_t handle = ecl_ptr(env, ptr);
	    *((void**)dptr) = (void*)handle;
	}
	put_element(data, iptr->extern_type, dptr, iptr->extern_info);
    }
}

//
// Special retrieve of program binaries
//
cl_int put_program_binaries(cbuf_t* out, cl_program program)
{
    cl_int err;
    size_t sizes[MAX_DEVICES]; 
    unsigned char* binaries[MAX_DEVICES];
    size_t returned_size;
    cl_uint num_devices;
    int i;

    err = clGetProgramInfo(program,
			   CL_PROGRAM_NUM_DEVICES,
			   sizeof(num_devices),
			   &num_devices,
			   &returned_size);
    if (err != CL_SUCCESS)
	return err;

    memset(sizes, 0,    sizeof(sizes));
    memset(binaries, 0, sizeof(binaries));

    err = clGetProgramInfo(program,
			   CL_PROGRAM_BINARY_SIZES,
			   sizeof(sizes),
			   sizes,
			   &returned_size);
    if (err != CL_SUCCESS)
	return err;
    if (returned_size != sizeof(size_t)*num_devices)
	return CL_INVALID_VALUE;
    
    i = 0;
    while (i < (int) num_devices) {
	binaries[i] = (unsigned char*) malloc(sizes[i]);
	if (!binaries[i]) {
	    err = CL_OUT_OF_HOST_MEMORY;
	    goto cleanup;
	}
	i++;
    }
    err = clGetProgramInfo(program,
			   CL_PROGRAM_BINARIES,
			   sizeof(unsigned char*)*num_devices,
			   binaries,
			   &returned_size);
    if (err != CL_SUCCESS)
	goto cleanup;

    cbuf_put_tuple_begin(out, 2);
    cbuf_put_tag_ok(out);
    cbuf_put_list_begin(out, num_devices);
    for (i = 0; i < (int)num_devices; i++)
	cbuf_put_binary(out, binaries[i], sizes[i]);
    cbuf_put_list_end(out, num_devices);
    cbuf_put_tuple_end(out, 2);    
cleanup:
    for (i = 0; i < (int)num_devices; i++)
	free(binaries[i]);
    return err;
}

// General info retrive
cl_int put_object_info(cbuf_t* out, ecl_object_t* obj, cl_uint info_arg)
{
    size_t returned_size = 0;
    cl_ulong buf[MAX_INFO_SIZE];
    cl_int err;

    if (info_arg >= obj->cl->info_len)
	return CL_INVALID_VALUE;
    err = ecl_object_info(obj,
			  obj->cl->info_vec[info_arg].info_id,
			  sizeof(buf),
			  buf, &returned_size);
    if (err == CL_SUCCESS) {
	cbuf_put_tuple_begin(out, 2);
	cbuf_put_tag_ok(out);
	put_value(out, obj->env, &obj->cl->info_vec[info_arg], 
		  buf, returned_size);
	cbuf_put_tuple_end(out, 2);
    }    
    return err;
}
//
// Load n elements into arrays using the get function
// each array element is of 'width' number of bytes wide.
//
int get_narray(cbuf_t* in,get_fn_t get,void* array, size_t width,
	       size_t n, void*arg)
{
    while(n) {
	if (!get(in, array, arg))
	    return 0;
	array = (void*)(((u_int8_t*)array)+width);
	n--;
    }
    return 1;
}
// Load arrays of objects from cbuf in each array element is of width bytes and 
// the get function reads elements from the input stream
// The maximum size is loaded from *size and then *size is updated
// with the actual size on return.
int get_array(cbuf_t* in,get_fn_t get, void* array, size_t width,
	      size_t* size, void*arg)
{
    u_int32_t n = *size;

    if (!get_size(in, size) || (*size > n))
	return 0;
    return get_narray(in, get, array, width, *size, arg);
}

// Main side send/recv
int ecl_send_command(ecl_env_t* env, ecl_command_t* cmd)
{
#ifdef ECL_USE_ASYNC
    ecl_async_t* async_data = driver_alloc(sizeof(ecl_async_t));
    async_data->command = *cmd;
    return driver_async(env->port, 0, (void (*)(void*)) ecl_async_invoke, 
			async_data, (void (*)(void*)) ecl_async_free);
#else
    return write(env->evt[0], cmd, sizeof(ecl_command_t));
#endif
}

int ecl_async_wait_event(ecl_object_t* obj)
{
    ecl_command_t cmd;

    cmd.type  = ECL_COMMAND_WAIT_STATUS;
    cmd.event = obj->event;
    cmd.caller = driver_caller(obj->env->port);
    cmd.bin   = 0;
    return ecl_send_command(obj->env, &cmd);
}

int ecl_async_wait_event_deref(ecl_object_t* obj, ErlDrvBinary* bin)
{
    ecl_command_t cmd;

    cmd.type  = ECL_COMMAND_WAIT_STATUS;    
    cmd.event = obj->event;
    cmd.caller = driver_caller(obj->env->port);
    cmd.bin   = bin;  // will be dereference when command is done
    return ecl_send_command(obj->env, &cmd);
}


int ecl_async_wait_event_bin(ecl_object_t* obj, ErlDrvBinary* bin)
{
    ecl_command_t cmd;

    cmd.type  = ECL_COMMAND_WAIT_BIN;    
    cmd.event = obj->event;
    cmd.caller = driver_caller(obj->env->port);
    cmd.bin   = bin;
    return ecl_send_command(obj->env, &cmd);
}

int ecl_async_finish(ecl_object_t* obj, u_int32_t eref)
{
    ecl_command_t cmd;
    cmd.type   = ECL_COMMAND_FINISH;
    cmd.queue  = obj->queue;
    cmd.caller = driver_caller(obj->env->port);
    cmd.bin    = 0;
    cmd.eref   = eref;
    return ecl_send_command(obj->env, &cmd);
}

int ecl_recv_response(ecl_env_t* env, ecl_response_t* resp)
{
    return read(env->evt[0], resp, sizeof(ecl_response_t));
}

// Thread side send/recv
int ecl_recv_command(ecl_env_t* env, ecl_command_t* cmd)
{
    return read(env->evt[1], cmd, sizeof(ecl_command_t));
}

int ecl_send_response(ecl_env_t* env, ecl_response_t* resp)
{
    A_DBG("ecl_send_response: resp.type=%s, resp.eref=%u",
	  ecl_response_name[resp->type], resp->eref);
    return write(env->evt[1], resp, sizeof(ecl_response_t));
}

// translate status into an atom, or fail
int ecl_event_status(ecl_kv_t* kv, cl_int status, ErlDrvTermData* ptr)
{
    while(kv->key) {
	if ((cl_int)kv->value == status) {
	    ptr[0] = ERL_DRV_ATOM;
	    ptr[1] = driver_mk_atom(kv->key);
	    return 0;
	}
	kv++;
    }
    ptr[0] = ERL_DRV_INT;
    ptr[1] = status;
    return -1;
}
//
// Thread dispatch loop, wait for events and report to main thread
//
void ecl_invoke(ecl_command_t* cmd, ecl_response_t* resp)
{
    ecl_command_type_t type = cmd->type;
    
    resp->caller = cmd->caller;
    resp->bin    = cmd->bin;
    resp->eref   = cmd->eref;

    A_DBG("ecl_invoke: cmd=%s, cmd_ref=%u",
	  ecl_command_name[type], cmd->eref);
    switch(type) {
    case ECL_COMMAND_WAIT_STATUS:
    case ECL_COMMAND_WAIT_BIN: {
	size_t info_size;
	resp->err = clWaitForEvents(1, &cmd->event);
	clGetEventInfo(cmd->event, CL_EVENT_COMMAND_EXECUTION_STATUS,
		       sizeof(cl_int), &resp->status, &info_size);
	resp->event = cmd->event;
	if (type == ECL_COMMAND_WAIT_STATUS)
	    resp->type = ECL_RESPONSE_EVENT_STATUS;
	else if (type == ECL_COMMAND_WAIT_BIN)
	    resp->type = ECL_RESPONSE_EVENT_BIN;
	break;
    }
	
    case ECL_COMMAND_FINISH:
	resp->err = clFinish(cmd->queue);
	resp->queue = cmd->queue;
	resp->type = ECL_RESPONSE_FINISH;
	break;

    default:
	resp->type = ECL_RESPONSE_NONE;
	resp->err = CL_INVALID_OPERATION;
	break;
    }
}

#ifdef ECL_USE_ASYNC
static void ecl_async_invoke(ecl_async_t* data)
{
    ecl_response_t resp;
    ecl_invoke(&data->command, &resp);
    // This copy could be avoided if we are carful in ecl_invoke!!!
    data->response = resp;
}

static void ecl_async_free(ecl_async_t* data)
{
    driver_free(data);
}
#else

static void* ecl_drv_dispatch(void* arg)
{
    ecl_env_t* env = (ecl_env_t*) arg;
    ecl_command_t cmd;
    int n;
    int err;

    A_DBG("ecl_drv_dispatch: started fd=%d", (int)env->evt[1]);

    while((n = ecl_recv_command(env, &cmd)) == sizeof(ecl_command_t)) {
	ecl_response_t resp;
	ecl_invoke(&cmd, &resp);
	ecl_send_response(env, &resp);
    }
    err = errno;
    A_DBG("ecl_drv_dispatch: closed err=%d", err);
    close(env->evt[1]);
    return (void*) LCAST(err);
}
#endif


typedef struct
{
    ecl_env_t*      env;
    u_int32_t       eref;
    ErlDrvTermData  caller;
} ecl_build_data_t;

//
// Notification functio for clBuildProgram
// Passed to main thread by sending a async response
// FIXME: lock needed?
//
void ecl_build_notify(cl_program program, void* user_data)
{
    ecl_build_data_t* dp = user_data;
    ecl_env_t* env = dp->env;
    ecl_response_t resp;

    A_DBG("ecl_build_notify: eref=%u", dp->eref);

    resp.type = ECL_RESPONSE_BUILD;
    resp.caller = dp->caller;
    resp.eref   = dp->eref;
    resp.err    = 0;
    resp.status = 0;
    resp.program = program;
    resp.bin = 0;
    // driver_free(user_data);
    ecl_send_response(env, &resp);
}

void ecl_context_notify(const char* errinfo,
			const void* private_info, size_t cb,
			void* user_data)
{
    ecl_env_t* env = (ecl_env_t*) user_data;
    ecl_response_t resp;
    (void) private_info;
    (void) cb;

    resp.type = ECL_RESPONSE_CONTEXT;
    resp.caller = 0;
    resp.eref   = 0;
    resp.errinfo = strdup(errinfo);
    resp.err    = 0;
    resp.bin    = 0;
    A_DBG("ecl_context_notify: errorinfo=%s", resp.errinfo);
    ecl_send_response(env, &resp);
}
			

DRIVER_INIT(ecl_drv)
{
    return &ecl_drv;
}


/* setup global object area */
static int ecl_drv_init(void)
{
    return 0;
}

static void ecl_drv_finish(void)
{
}

static ErlDrvData ecl_drv_start(ErlDrvPort port, char* command)
{
    ecl_env_t* env;
    (void) command;

    if ((env = (ecl_env_t*) driver_alloc(sizeof(ecl_env_t))) != NULL) {
	lhash_func_t func = { ref_hash, ref_cmp, ref_release, 0 };
	env->port = port;
	env->eref = 0xFEEDBABE;  // random start
	lhash_init(&env->ref, "ref", 2, &func);
	set_port_control_flags(port, PORT_CONTROL_FLAG_BINARY);
	env->evt[0] = -1;
	env->evt[1] = -1;
#ifdef ECL_USE_ASYNC
	{
	    ErlDrvSysInfo info;
	    driver_system_info(&info,sizeof(ErlDrvSysInfo));
	    if (info.async_threads == 0) {
		fprintf(stderr, "cl_drv: WARNING: missing +A option (async driver calls!)\r\n");
	    }
	}
#else
#ifdef HAVE_SOCKETPAIR
	{
	    int sockets[2];
	    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) < 0)
		return ERL_DRV_ERROR_ERRNO;
	    env->evt[0] =  sockets[0];
	    env->evt[1] =  sockets[1];
	}
#else
#error "cl_drv: currently need socketpair"
#endif
	if (erl_drv_thread_create("cl_drv_event_dispatch",
				  &env->tid, 
				  ecl_drv_dispatch,
				  env,
				  0) < 0) {
	    return ERL_DRV_ERROR_ERRNO;
	}
	driver_select(port, (ErlDrvEvent)LCAST(env->evt[0]), ERL_DRV_READ, 1);
#endif
	return (ErlDrvData) env;
    }
    return ERL_DRV_ERROR_ERRNO;
}

static void ecl_drv_stop(ErlDrvData d)
{
    ecl_env_t* env = (ecl_env_t*) d;
#ifndef ECL_USE_ASYNC
    void* resp;
    driver_select(env->port, (ErlDrvEvent)LCAST(env->evt[0]), ERL_DRV_READ, 0);
    close((int) env->evt[0]);
    erl_drv_thread_join(env->tid, &resp);
#endif
    lhash_delete(&env->ref);
    driver_free(env);
}

static void ecl_drv_command(ErlDrvData d, char* buf, int len)
{
    (void) d;
    (void) buf;
    (void) len;
}

static void ecl_response(ecl_env_t* env, ecl_response_t* resp)
{
    ErlDrvTermData term_data[16];
    // ecl_object_t* obj;
    int i = 0;

    A_DBG("ecl_drv_input: resp->type=%s resp->ref=%u",
	  ecl_response_name[resp->type], resp->eref);
    // Send {cl_event,Event,Status} to caller
    switch(resp->type) {
    case ECL_RESPONSE_NONE: {
	// response to bad command
	term_data[i++] = ERL_DRV_ATOM;
	term_data[i++] = driver_mk_atom("cl_reply");
	term_data[i++] = ERL_DRV_UINT;
	term_data[i++] = resp->eref;	
	term_data[i++] = ERL_DRV_ATOM;
	term_data[i++] = driver_mk_atom("error");
	term_data[i++] = ERL_DRV_ATOM;
	term_data[i++] = driver_mk_atom(fmt_error(resp->err));
	term_data[i++] = ERL_DRV_TUPLE;
	term_data[i++] = 2;
	term_data[i++] = ERL_DRV_TUPLE;
	term_data[i++] = 3;
	driver_send_term(env->port, resp->caller, term_data, i);
	break;
    }

    case ECL_RESPONSE_FINISH: {
	int i = 0;  // max 12 words
	term_data[i++] = ERL_DRV_ATOM;
	term_data[i++] = driver_mk_atom("cl_reply");
	term_data[i++] = ERL_DRV_UINT;
	term_data[i++] = resp->eref;
	if (!resp->err) {
	    term_data[i++] = ERL_DRV_ATOM;
	    term_data[i++] = driver_mk_atom("ok");
	}
	else {
	    term_data[i++] = ERL_DRV_ATOM;
	    term_data[i++] = driver_mk_atom("error");
	    term_data[i++] = ERL_DRV_ATOM;
	    term_data[i++] = driver_mk_atom(fmt_error(resp->err));
	    term_data[i++] = ERL_DRV_TUPLE;
	    term_data[i++] = 2;
	}
	term_data[i++] = ERL_DRV_TUPLE;
	term_data[i++] = 3;
	driver_send_term(env->port, resp->caller, term_data, i);
	break;
    }

    case ECL_RESPONSE_BUILD: { // 8 words
	term_data[0] = ERL_DRV_ATOM;
	term_data[1] = driver_mk_atom("cl_reply");
	term_data[2] = ERL_DRV_UINT;
	term_data[3] = resp->eref;
	term_data[4] = ERL_DRV_ATOM;
	term_data[5] = driver_mk_atom("ok");
	term_data[6] = ERL_DRV_TUPLE;
	term_data[7] = 3;
	driver_send_term(env->port, resp->caller, term_data, 8);
	break;
    }

    case ECL_RESPONSE_CONTEXT: { // 9 words
	term_data[0] = ERL_DRV_ATOM;
	term_data[1] = driver_mk_atom("cl_error");
	term_data[2] = ERL_DRV_UINT;
	term_data[3] = resp->eref;
	term_data[4] = ERL_DRV_STRING;
	term_data[5] = (ErlDrvTermData)resp->errinfo;
	term_data[6] = strlen(resp->errinfo);
	term_data[7] = ERL_DRV_TUPLE;
	term_data[8] = 3;
	if (!resp->caller)
	    driver_output_term(env->port, term_data, 9);
	else
	    driver_send_term(env->port, resp->caller, term_data, 9);
	driver_free(resp->errinfo);
	break;
    }

    case ECL_RESPONSE_EVENT_STATUS: { // max 8 words
	term_data[0] = ERL_DRV_ATOM;
	term_data[1] = driver_mk_atom("cl_event");
	term_data[2] = ERL_DRV_UINT;
	term_data[3] = EPTR_HANDLE(resp->event);
	ecl_event_status(kv_execution_status,resp->status,&term_data[4]);
	term_data[6] = ERL_DRV_TUPLE;
	term_data[7] = 3;
	/* get_event_info wont work! is this better?
	   if ((obj = event_object(env, EPTR_HANDLE(resp->event))))
	   ecl_object_release(obj);
	*/
	driver_send_term(env->port, resp->caller, term_data, 8);
	if (resp->bin)
	    driver_free_binary(resp->bin);
	break;
    }

    case ECL_RESPONSE_EVENT_BIN: { // max 10 words
	// This response is initially from clEnqueueReadBuffer
	// so if everythings is ok then pass the binary back to
	// the caller. Otherwise send an error.
	// Send {cl_event,Event,Bin|Status}
	term_data[0] = ERL_DRV_ATOM;
	term_data[1] = driver_mk_atom("cl_event");
	term_data[2] = ERL_DRV_UINT;
	term_data[3] = EPTR_HANDLE(resp->event);
	/* get_event_info wont work! is this better ?
	   if ((obj = event_object(env, EPTR_HANDLE(resp->event))))
	   ecl_object_release(obj);
	*/
	if (resp->status == CL_COMPLETE) {
	    term_data[4] = ERL_DRV_BINARY;
	    term_data[5] = (ErlDrvTermData) resp->bin;
	    term_data[6] = resp->bin->orig_size;
	    term_data[7] = 0;
	    term_data[8] = ERL_DRV_TUPLE;
	    term_data[9] = 3;
	    driver_send_term(env->port, resp->caller, term_data, 10);
	    driver_free_binary(resp->bin);
	}
	else {
	    driver_free_binary(resp->bin);
	    ecl_event_status(kv_execution_status,resp->status,
			     &term_data[4]);
	    term_data[6] = ERL_DRV_TUPLE;
	    term_data[7] = 3;
	    driver_send_term(env->port, resp->caller, term_data, 8);
	}
	return;
    }
	
	
    default:
	A_DBG("ecl_drv_input: warning: unkown response type = %d",
	      resp->type);
	return;
    }
}


static void ecl_drv_ready_async(ErlDrvData d, ErlDrvThreadData thread_data)
{
    ecl_env_t* env = (ecl_env_t*) d;
    ecl_response_t* resp = (ecl_response_t*) thread_data;
    ecl_response(env, resp);
    driver_free(resp);
}


// When the thread dispatcher has some thing ready we will wake up
// at this point. send to original caller
// here we dereference the event objects since they are used up already.
// (Or do we need them further?) Then use event_retain!
//
static void ecl_drv_input(ErlDrvData d, ErlDrvEvent handle)
{
    ecl_env_t* env = (ecl_env_t*) d;
    if (handle == (ErlDrvEvent)LCAST(env->evt[0])) {
	ecl_response_t resp;
	int n;
	A_DBG("ecl_drv_input: ready handle=%ld", LCAST(handle));	
	if ((n = ecl_recv_response(env, &resp)) == sizeof(ecl_response_t))
	    ecl_response(env, &resp);
    }
}

static void ecl_drv_output(ErlDrvData d, ErlDrvEvent e)
{
    (void) d;
    (void) e;
}

// Write OK
static inline void put_ok(cbuf_t* out)
{
    cbuf_put_tag_ok(out);
}

// Write ERROR,ATOM,String
static inline void put_error(cbuf_t* out, cl_int err)
{
    cbuf_put_tuple_begin(out, 2);
    cbuf_put_tag_error(out);
    cbuf_put_atom(out, fmt_error(err));
    cbuf_put_tuple_end(out, 2);
}

// Write EVENT,event-ref:32
static inline void put_event(cbuf_t* out, u_int32_t ref)
{
    cbuf_put_tuple_begin(out, 2);
    cbuf_put_tag_event(out);
    cbuf_put_uint32(out, ref);
    cbuf_put_tuple_end(out, 2);
}

// Write OK Object-Handle
static inline void put_object(cbuf_t* out, ecl_object_t* obj)
{
    pointer_t handle = ecl_handle(obj);
    cbuf_put_tuple_begin(out, 2);
    cbuf_put_tag_ok(out);
    put_pointer(out, handle);
    cbuf_put_tuple_end(out, 2);
}

#define RETURN_OK()       do { put_ok(&reply); goto done; } while(0)
#define RETURN_ERROR(err) do { put_error(&reply,(err)); goto done; } while(0)
#define RETURN_OBJ(obj)   do { put_object(&reply,(obj)); goto done; } while(0)
#define RETURN_EVENT(eref) do { put_event(&reply,(eref)); goto done; } while(0)
#define FRETURN_OK()       do { put_ok(reply); return; } while(0)
#define FRETURN_ERROR(err) do { put_error(reply,(err)); return; } while(0)
#define FRETURN_OBJ(obj)   do { put_object(reply,(obj)); return; } while(0)

//
// ECL_CREATE_BUFFER
// <<Context:Ptr, Flags:32, Size:32, Data/rest>>
static void ecl_create_buffer(ecl_env_t* env, cbuf_t* arg, cbuf_t* reply)
{
    pointer_t chandle;
    u_int32_t flags;
    u_int32_t size;
    ecl_object_t* ctx;
    int err = CL_INVALID_VALUE;

    if (get_pointer(arg, &chandle) &&
	get_uint32(arg, &flags) &&
	get_uint32(arg, &size) &&
	((ctx = context_object(env, chandle))) ) {
	void* host_ptr = 0;
	cl_mem_flags mem_flags = 0;
	cl_mem mem;
	size_t avail;

	//FIXME: how should we interpret COPY_HOST_PTR from user?
	// maybe we just remove the options from the interface?
	if (flags & ECL_MEM_READ_WRITE)
	    mem_flags |= CL_MEM_READ_WRITE;
	if (flags & ECL_MEM_WRITE_ONLY)
	    mem_flags |= CL_MEM_WRITE_ONLY;
	if (flags & ECL_MEM_READ_ONLY)
	    mem_flags |= CL_MEM_READ_ONLY;
	if (flags & ECL_MEM_USE_HOST_PTR)
	    mem_flags |= CL_MEM_USE_HOST_PTR;
	if (flags & ECL_MEM_ALLOC_HOST_PTR)
	    mem_flags |= CL_MEM_ALLOC_HOST_PTR;
	if (flags & ECL_MEM_COPY_HOST_PTR)	   
	    mem_flags |= CL_MEM_COPY_HOST_PTR; 

	avail = cbuf_r_avail(arg);
	if (avail > 0) {
	    if (avail == cbuf_seg_r_avail(arg))
		host_ptr = cbuf_seg_ptr(arg);
	    else {
		if (!(host_ptr = driver_alloc(avail)))
		    FRETURN_ERROR(CL_OUT_OF_HOST_MEMORY);
		cbuf_read(arg, host_ptr, avail);
	    }
	    mem_flags |= CL_MEM_COPY_HOST_PTR;
	}
	else if (size > 0) {
	    mem_flags |= CL_MEM_ALLOC_HOST_PTR;
	}

	mem = clCreateBuffer(ctx->context, mem_flags,
			     (size_t) size,
			     host_ptr, &err);
	if ((mem_flags & CL_MEM_COPY_HOST_PTR) && host_ptr &&
	    (host_ptr != cbuf_seg_ptr(arg)))
	    driver_free(host_ptr);
	if (mem) {
	    ecl_object_t* obj;
	    if (!(obj = EclMemCreate(env,mem))) {
		clReleaseMemObject(mem);
		FRETURN_ERROR(CL_OUT_OF_HOST_MEMORY);
	    }
	    FRETURN_OBJ(obj);
	}
    }
    FRETURN_ERROR(err);
}

//
// ECL_ENQUEUE_WRITE_BUFFER
// <<Queue:Ptr, Mem:Ptr, Offset:32, Cb:32,
//   NumEvents:32 Event1:Ptr, ..EventN:Ptr>>
//   Data/binary
//
static void ecl_enqueue_write_buffer(ecl_env_t* env, cbuf_t* arg, cbuf_t* reply)
{
    pointer_t  qh;
    pointer_t  mh;
    ecl_object_t* qobj;
    ecl_object_t* mobj;
    u_int32_t offset_arg;
    u_int32_t cb_arg;
    u_int32_t n_arg;
    cl_event wait_list[MAX_WAIT_LIST];
    size_t num_wait = MAX_WAIT_LIST;
    int err = CL_INVALID_VALUE;

    if (get_pointer(arg, &qh) && 
	get_pointer(arg, &mh) && 
	get_uint32(arg, &offset_arg) &&
	get_uint32(arg, &cb_arg) &&
	get_array(arg,(get_fn_t)get_event,wait_list,sizeof(cl_event),
		  &num_wait,env) &&
	(qobj = queue_object(env, qh)) &&
	(mobj = mem_object(env, mh)) &&
	((n_arg=cbuf_r_avail(arg)) >= cb_arg)) {
	cl_event event;
	ErlDrvBinary* bin;
	unsigned int bin_offset;

	// detect contigous binary memory and use it
	if ((bin=arg->v[arg->iv].bp) && (cbuf_seg_r_avail(arg) >= n_arg)) {
	    // found a binary offset=ip, do we have all data?
	    driver_binary_inc_refc(bin);
	    bin_offset = arg->ip;
	}
	else {
	    if (!(bin = driver_alloc_binary(n_arg)))
		FRETURN_ERROR(CL_OUT_OF_HOST_MEMORY);
	    bin_offset = 0;
	    cbuf_read(arg, bin->orig_bytes, n_arg);
	    // memcpy(bin->orig_bytes, arg->ptr, n_arg);
	}
	err = clEnqueueWriteBuffer(qobj->queue, mobj->mem,
				   CL_FALSE,
				   offset_arg,
				   cb_arg,
				   (void*) bin->orig_bytes+bin_offset,
				   (cl_uint) num_wait, 
				   (num_wait ? wait_list : 0),
				   &event);
	if (err == CL_SUCCESS) {
	    ecl_object_t* obj;
	    if (!(obj = EclEventCreate(env, event))) {
		clReleaseEvent(event);
		driver_free_binary(bin);
		FRETURN_ERROR(CL_OUT_OF_HOST_MEMORY);
	    }
	    // will dereference the binary when write is done (or failed)
	    ecl_async_wait_event_deref(obj, bin);
	    FRETURN_OBJ(obj);
	}
    }
    FRETURN_ERROR(err);
}
//
// <<Context:Ptr, Source/binary>>
// This command may be called with both port_command with an iolist
// where binaries will be preserved by outputv (ErlIOVec)
// Or port_control where iodata is merged into on buffer
//
// FIXME: run the actual cl command in a thread
//
static void ecl_create_program_with_source(ecl_env_t* env, cbuf_t* arg, cbuf_t* reply)
{
    pointer_t chandle;
    ecl_object_t* ctx;
    int err = CL_INVALID_VALUE;

    if (get_pointer(arg, &chandle) &&
	((ctx = context_object(env, chandle))) && 
	!cbuf_eob(arg)) {
	cl_program program;
	char* strings[MAX_SOURCES];
	size_t lengths[MAX_SOURCES];
	cl_uint count = 0;

	// install first vector (or linear buffer)
	strings[count] = (char*) cbuf_seg_ptr(arg);
	lengths[count] = cbuf_seg_r_avail(arg);
	DBG("source: length[%u]=%lu", count, lengths[count]);
	count++;
	// check if we have vectors to install
	if (arg->vlen > 1) {
	    int i = arg->iv+1;
	    while(i < (int)arg->vlen) {
		if (count == MAX_SOURCES)
		        FRETURN_ERROR(err);
		lengths[count] = arg->v[i].len;
		strings[count] = (char*) arg->v[i].base;
		DBG("source: length[%u]=%lu", count, lengths[count]);
		count++;
		i++;
	    }
	}
	program = clCreateProgramWithSource(ctx->context,
					    count,
					    (const char**) strings,
					    lengths,
					    &err);
	if (program) {
	    ecl_object_t* obj;
	    
	    if (!(obj = EclProgramCreate(env,program))) {
		clReleaseProgram(program);
		FRETURN_ERROR(CL_OUT_OF_HOST_MEMORY);
	    }
	    FRETURN_OBJ(obj);
	}
    }
    FRETURN_ERROR(err);
}

// ECL_CREATE_PROGRAM_WITH_BINARY:
// <<Context:Ptr, 
//   NumDevices:Size,  Device1:Ptr, ... DeviceN:Ptr,
//   NumBinaries:Size, Size1:32, Binary:Size1 ... 
//                     SizeN:32, Binary:SizeN>>
//
// FIXME: run as async command
//
static void ecl_create_program_with_binary(ecl_env_t* env, cbuf_t* arg, cbuf_t* reply)
{
    pointer_t     chandle;
    ecl_object_t* ctx;
    cl_device_id  device_list[MAX_DEVICES];
    size_t        num_devices = MAX_DEVICES;
    size_t        num_binaries = 0;
    int           err = CL_INVALID_VALUE;
    unsigned char* binaries[MAX_DEVICES];
    size_t lengths[MAX_DEVICES];
    u_int8_t allocated[MAX_DEVICES];
    size_t i;

    memset(allocated, 0, sizeof(allocated));

    if (get_pointer(arg, &chandle) &&
	(ctx = context_object(env, chandle)) &&
	get_array(arg,(get_fn_t)get_device,device_list,
		  sizeof(cl_device_id), &num_devices,env) &&
 	get_size(arg, &num_binaries) &&
	(num_binaries == num_devices)) {
	cl_program program;
	cl_int status;

	DBG("num_binaries=%lu", num_binaries);
	
	for (i = 0; i < num_binaries; i++) {
	    if (get_size(arg, &lengths[i])) {
		if (cbuf_seg_r_avail(arg) >= lengths[i]) {
		    // Handles the case both for port_control & when smaller
		    // binaries have been concatinated in the iodata.
		    binaries[i] = cbuf_seg_ptr(arg);
		    cbuf_forward(arg, lengths[i]);
		}
		else if (cbuf_r_avail(arg) >= lengths[i]) {
		    if (!(binaries[i] = driver_alloc(lengths[i]))) {
			put_error(reply,  CL_OUT_OF_HOST_MEMORY);
			goto clean_up;
		    }
		    allocated[i] = 1;
		    cbuf_read(arg, binaries[i], lengths[i]);
		}
	    }
	    else {
		put_error(reply, err);
		goto clean_up;
	    }
	}
	if (!cbuf_eob(arg)) {
	    put_error(reply, err);	    
	    goto clean_up;
	}

	program = clCreateProgramWithBinary(ctx->context,
					    num_devices,
					    (const cl_device_id*) device_list,
					    (const size_t*) lengths,
					    (const unsigned char**) binaries,
					    &status,
					    &err);
	if (program) {
	    ecl_object_t* obj;
	    
	    if (!(obj = EclProgramCreate(env,program))) {
		clReleaseProgram(program);
		put_error(reply, CL_OUT_OF_HOST_MEMORY);
		goto clean_up;
	    }
	    put_object(reply, obj);
	}
    }

clean_up:
    for (i = 0; i < num_binaries; i++) {
	if (allocated[i])
	    driver_free(binaries[i]);
    }
}

// This is tricky to type check
// <Kernel:Ptr, Index:32, ArgSize:32, Argument/binary>>
// type = POINTER | SIZE | 
//
static int ecl_set_kernel_arg(ecl_env_t* env, int type, cbuf_t* arg)
{
    pointer_t handle;
    u_int32_t arg_index;
    u_int32_t arg_size;
    size_t    arg_len;
    ecl_object_t* obj;
    int err = CL_INVALID_VALUE;

    if (get_pointer(arg, &handle) &&
	get_uint32(arg, &arg_index) &&
	get_uint32(arg, &arg_size) &&
	((obj = kernel_object(env, handle)) != 0)) {
	u_int8_t* arg_ptr = 0;
	u_int8_t  arg_alloc = 0;
	u_int8_t  arg_buf[64];

	arg_len = cbuf_r_avail(arg);
	switch(type) {
	case 0:
	    if (arg_len) {
		if (cbuf_seg_r_avail(arg) == arg_len)
		    arg_ptr = cbuf_seg_ptr(arg);
		else {
		    if (arg_len <= 64)
			arg_ptr = arg_buf;
		    else {
			if (!(arg_ptr = driver_alloc(arg_len)))
			    return CL_OUT_OF_HOST_MEMORY;
			arg_alloc = 1;
		    }
		    cbuf_read(arg, arg_ptr, arg_len);
		}
	    }
	    break;
	case POINTER:
	    if ((arg_len != 8) || !get_pointer(arg, (pointer_t*)&arg_buf[0]))
		return err;
	    arg_size = sizeof(pointer_t); // real size!!!
	    arg_ptr = arg_buf;
	    break;
	case USIZE:
	    if ((arg_len != 8) || !get_size(arg, (size_t*)&arg_buf[0]))
		return err;
	    arg_size = sizeof(size_t); // real size!!!
	    arg_ptr = arg_buf;	    
	    break;
	default:
	    return err;
	}
	DBG("set_kernel_arg:index=%d,size=%d,len=%ld",
	    arg_index,arg_size,arg_len);
	err = clSetKernelArg(obj->kernel,
			     (cl_uint) arg_index,
			     arg_size,
			     arg_ptr);
	if (arg_alloc)
	    driver_free(arg_ptr);
    }
    return err;
}


static int ecl_drv_ctl(ErlDrvData d, 
			unsigned int cmd, char* buf, int len,
			char** rbuf, int rsize)
{
    ecl_env_t* env = (ecl_env_t*) d;
    cbuf_t  arg;    // argument data stream
    cbuf_t  reply;  // reply data stream
    cl_int err = CL_INVALID_VALUE;

    // input data
    cbuf_init(&arg, (unsigned char*) buf, len, 0, 0);
    CBUF_DBG(&arg, "ctl_arg");
    // default data - will relloacte (as binary) on overflow
    cbuf_init(&reply, (unsigned char*) *rbuf, rsize, 0, 
	      CBUF_FLAG_BINARY | ECL_REPLY_TYPE);

    cbuf_put_begin(&reply);  // put respose header
    
    DBG("Cmd: %02X", cmd);
    switch(cmd) {
    case ECL_NOOP:
	RETURN_OK();

    case ECL_GET_PLATFORM_IDS: { 
        // no arguments - return list of platform_ids
	if (cbuf_eob(&arg)) {
	    cl_uint          num_platforms = 0;
	    cl_platform_id   platform_id[MAX_PLATFORMS];
	    cl_uint i;

	    err = clGetPlatformIDs(MAX_PLATFORMS, platform_id, &num_platforms);
	    if (err != CL_SUCCESS) 
		RETURN_ERROR(err);
	    cbuf_put_tuple_begin(&reply, 2),
	    cbuf_put_tag_ok(&reply);
	    cbuf_put_list_begin(&reply, num_platforms);
	    for (i = 0; i < num_platforms; i++) {
		ecl_object_t* obj = EclPlatform(env, platform_id[i]);
		put_pointer(&reply, ecl_handle(obj));
	    }
	    cbuf_put_list_end(&reply, num_platforms);
	    cbuf_put_tuple_end(&reply, 2);
	    goto done;
	}
	break;
    }

    case ECL_GET_DEVICE_IDS: {
        // <<platformid:Ptr, device_type:32>> - return list of device_ids
	pointer_t        handle;
	u_int32_t        type_arg;
	ecl_object_t*    obj = 0;
	
	if (get_pointer(&arg, &handle) && 
	    get_uint32(&arg, &type_arg) &&
	    (!handle || (obj = platform_object(env, handle))) &&
	    cbuf_eob(&arg)) {
	    cl_uint          num_devices = 0;
	    cl_device_id     device_id[MAX_DEVICES];
	    cl_platform_id   platform = obj ? obj->platform : 0;
	    cl_device_type   device_type = 0;
	    cl_uint i;

	    if (type_arg == ECL_DEVICE_TYPE_ALL)
		device_type = CL_DEVICE_TYPE_ALL;
	    else {
		if (type_arg & ECL_DEVICE_TYPE_CPU)
		    device_type |= CL_DEVICE_TYPE_CPU;
		if (type_arg & ECL_DEVICE_TYPE_GPU)
		    device_type |= CL_DEVICE_TYPE_GPU;
		if (type_arg & ECL_DEVICE_TYPE_ACCELERATOR)
		    device_type |= CL_DEVICE_TYPE_ACCELERATOR;
	    }
	    err = clGetDeviceIDs(platform, device_type, MAX_DEVICES, 
				 device_id, &num_devices);
	    if (err != CL_SUCCESS)
		RETURN_ERROR(err);
	    cbuf_put_tuple_begin(&reply, 2);
	    cbuf_put_tag_ok(&reply);
	    cbuf_put_list_begin(&reply, num_devices);
	    for (i = 0; i < num_devices; i++) {
		ecl_object_t* dobj = EclDevice(env, device_id[i]);
		put_pointer(&reply, ecl_handle(dobj));
	    }
	    cbuf_put_list_end(&reply,num_devices);
	    cbuf_put_tuple_end(&reply,2);
	    goto done;
	}
	break;
    }

    case ECL_GET_PLATFORM_INFO: {
	// <<Platform:Ptr,  Info:32>>
	pointer_t        handle;
	u_int32_t        info_arg;
	ecl_object_t*    obj = 0;

	if (get_pointer(&arg, &handle) && 
	    get_uint32(&arg, &info_arg) &&
	    (!handle || (obj = platform_object(env, handle))) && 
	    cbuf_eob(&arg)) {
	    cl_platform_id   platform = obj ? obj->platform : 0;
	    size_t returned_size = 0;
	    cl_ulong buf[256];   // 2k buffer

	    // can not run the ecl_class_platform (obj may be 0)
	    if (info_arg < ECL_PLATFORM_INFO_NUM) {
		err = clGetPlatformInfo(platform,
					platform_info[info_arg].info_id,
					sizeof(buf),
					buf, &returned_size);
		if (err == CL_SUCCESS) {
		    cbuf_put_tuple_begin(&reply, 2);
		    cbuf_put_tag_ok(&reply);
		    put_value(&reply, env, &platform_info[info_arg],
			      buf, returned_size);
		    cbuf_put_tuple_end(&reply, 2);
		    goto done;
		}
	    }
	}
	break;
    }

    case ECL_GET_DEVICE_INFO: {
	// <<Device:PtrSize,  Info:32>>
	pointer_t        handle;
	u_int32_t        info_arg;
	ecl_object_t*    obj;

	if (get_pointer(&arg, &handle) && 
	    get_uint32(&arg, &info_arg) &&
	    cbuf_eob(&arg) &&
	    ((obj = device_object(env, handle)))) {
	    err = put_object_info(&reply, obj, info_arg);
	    if (err == CL_SUCCESS)
		goto done;
	}
	break;
    }

    case ECL_CREATE_CONTEXT: {
	// <<NumDevices:Size,Dev1:Ptr, DevN:Ptr>>
	// FIXME: add cl_context_properties properties;
	cl_device_id     device_list[MAX_DEVICES]; 
	size_t           num_devices = MAX_DEVICES;

	// FIXME: add the platform in properties
	
	if (get_array(&arg,(get_fn_t)get_device,device_list,
		      sizeof(cl_device_id), &num_devices,env) &&
	    cbuf_eob(&arg)) {
	    cl_context       context;
#ifdef ASYNC_CONTEXT_NOTIFY
	    context = clCreateContext(0, num_devices, device_list, 
				      ecl_context_notify,
				      env,
				      &err);
#else
	    context = clCreateContext(0, num_devices, device_list, 
				      0, 0,
				      &err);
#endif
	    if (context) {
		ecl_object_t* obj;
		if (!(obj = EclContextCreate(env,context))) {
		    clReleaseContext(context);
		    RETURN_ERROR(CL_OUT_OF_HOST_MEMORY);
		}
		RETURN_OBJ(obj);
	    }
	}
	break;
    }

    case ECL_RETAIN_CONTEXT: {
	// <<Context:Ptr>>
	pointer_t handle;
	ecl_object_t* obj;

	if (get_pointer(&arg, &handle) &&
	    ((obj = context_object(env, handle))) &&
	    cbuf_eob(&arg)) {
	    ecl_object_retain(obj);
	    RETURN_OK();
	}
	break;
    }

    case ECL_RELEASE_CONTEXT: {
	// <<Context:Ptr>>
	pointer_t handle;
	ecl_object_t* obj;

	if (get_pointer(&arg, &handle) &&
	    cbuf_eob(&arg) &&
	    ((obj = context_object(env, handle)) != 0)) {
	    ecl_object_release(obj);
	    RETURN_OK();
	}
	break;
    }

    case ECL_GET_CONTEXT_INFO: {
	// <<Context:Ptr,  Info:32>>
	pointer_t   handle;
	u_int32_t   info_arg;
	ecl_object_t* obj;

	if (get_pointer(&arg, &handle) && 
	    get_uint32(&arg, &info_arg) &&
	    cbuf_eob(&arg) &&
	    ((obj = context_object(env, handle)) != 0)) {
	    err = put_object_info(&reply, obj, info_arg);
	    if (err == CL_SUCCESS)
		goto done;
	}
	break;
    }

    case ECL_CREATE_QUEUE: {
	// <<Context:Ptr, DeviceID:Ptr, Poperties:32>>
	pointer_t chandle;
	pointer_t dhandle;
	u_int32_t prop;
	ecl_object_t* ctx;
	ecl_object_t* dev;
	
	if (get_pointer(&arg, &chandle) &&
	    get_pointer(&arg, &dhandle) &&
	    get_uint32(&arg, &prop) &&
	    cbuf_eob(&arg) &&
	    ((ctx = context_object(env, chandle))) &&
	    ((dev = device_object(env, dhandle)))) {
	    cl_device_id device = dev->device;
	    cl_command_queue_properties properties = 0;
	    cl_command_queue queue;

	    if (prop & ECL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)
		properties |= CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
	    if (prop & ECL_QUEUE_PROFILING_ENABLE)
		properties |= CL_QUEUE_PROFILING_ENABLE;
	    queue = clCreateCommandQueue(ctx->context,
					 device, properties,
					 &err);
	    if (queue) {
		ecl_object_t* obj;
		if (!(obj = EclQueueCreate(env,queue))) {
		    clReleaseCommandQueue(queue);
		    RETURN_ERROR(CL_OUT_OF_HOST_MEMORY);
		}
		RETURN_OBJ(obj);
	    }
	}
	break;
    }

    case ECL_RETAIN_QUEUE: {
	// <<CommandQueue:Ptr>>
	pointer_t handle;
	ecl_object_t* obj;

	if (get_pointer(&arg, &handle) &&
	    cbuf_eob(&arg) &&
	    ((obj = queue_object(env, handle)) != 0)) {
	    ecl_object_retain(obj);
	    RETURN_OK();
	}
	break;
    }

    case ECL_RELEASE_QUEUE: {
	// <<CommandQueue:Ptr>>
	pointer_t handle;
	ecl_object_t* obj;

	if (get_pointer(&arg, &handle) &&
	    cbuf_eob(&arg) &&
	    ((obj = queue_object(env, handle)) != 0)) {
	    ecl_object_release(obj);
	    RETURN_OK();
	}
	break;
    }

    case ECL_GET_QUEUE_INFO: {
	// <<CommandQueue:Ptr,  Info:32>>
	pointer_t        handle;
	u_int32_t        info_arg;
	ecl_object_t* obj;

	if (get_pointer(&arg, &handle) && 
	    get_uint32(&arg, &info_arg) &&
	    cbuf_eob(&arg) &&
	    ((obj = queue_object(env, handle)) != 0)) {
	    err = put_object_info(&reply, obj, info_arg);
	    if (err == CL_SUCCESS)
		goto done;
	}
	break;
    }

    case ECL_SET_QUEUE_PROPERTY: {
	// <<Queue:Ptr, Properties:32, Enable:Bool>>
	pointer_t        handle;
	u_int32_t        prop;
	u_int32_t        enable;
	ecl_object_t* obj;

	if (get_pointer(&arg, &handle) && 
	    get_uint32(&arg, &prop) &&
	    get_uint32(&arg, &enable) &&
	    cbuf_eob(&arg) &&
	    ((obj = queue_object(env, handle)) != 0)) {
	    cl_command_queue_properties properties = 0;
	    cl_command_queue_properties old_properties;

	    if (prop & ECL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)
		properties |= CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
	    if (prop & ECL_QUEUE_PROFILING_ENABLE)
		properties |= CL_QUEUE_PROFILING_ENABLE;
	    err = clSetCommandQueueProperty(obj->queue, properties,
					    (enable != 0),
					    &old_properties);
	    if (err == CL_SUCCESS) {
		cbuf_put_tuple_begin(&reply, 2);
		cbuf_put_tag_ok(&reply);
		put_element(&reply, BITFIELD, &old_properties,
			    kv_command_queue_properties);
		cbuf_put_tuple_end(&reply, 2);
		goto done;
	    }
	}
	break;
    }

    case ECL_FLUSH: {	
	// <<Queue:Ptr>>
	pointer_t        handle;
	ecl_object_t* obj;
	if (get_pointer(&arg, &handle) && 
	    cbuf_eob(&arg) &&
	    ((obj = queue_object(env, handle)) != 0)) {
	    err = clFlush(obj->queue);
	}
	break;
    }

    case ECL_FINISH: {
	// <<Queue:Ptr>>
	pointer_t        handle;
	ecl_object_t* obj;
	if (get_pointer(&arg, &handle) &&
	    cbuf_eob(&arg) &&
	    ((obj = queue_object(env, handle)) != 0)) {
	    u_int32_t eref = env->eref++;
	    ecl_async_finish(obj, eref);
	    RETURN_EVENT(eref);
	}
	break;
    }
	

    case ECL_CREATE_BUFFER: {
	ecl_create_buffer(env, &arg, &reply);
	goto done;
    }

    case ECL_RETAIN_MEM_OBJECT: {
	// <<Mem:Ptr>>
	pointer_t handle;
	ecl_object_t* obj;

	if (get_pointer(&arg, &handle) &&
	    cbuf_eob(&arg) &&
	    ((obj = mem_object(env, handle)) != 0)) {
	    ecl_object_retain(obj);
	    RETURN_OK();
	}
	break;
    }

    case ECL_RELEASE_MEM_OBJECT: {
	// <<Mem:Ptr>>
	pointer_t handle;
	ecl_object_t* obj;

	if (get_pointer(&arg, &handle) &&
	    cbuf_eob(&arg) &&
	    ((obj = mem_object(env, handle)) != 0)) {
	    ecl_object_release(obj);
	    RETURN_OK();
	}
	break;
    }

    case ECL_GET_MEM_OBJECT_INFO: {
	// <<Mem:Ptr,  Info:32>>
	pointer_t  handle;
	u_int32_t  info_arg;
	ecl_object_t* obj;

	if (get_pointer(&arg, &handle) && 
	    get_uint32(&arg, &info_arg) &&
	    cbuf_eob(&arg) &&
	    ((obj = mem_object(env, handle)) != 0)) {
	    err = put_object_info(&reply, obj, info_arg);
	    if (err == CL_SUCCESS)
		goto done;
	}
	break;
    }

    case ECL_CREATE_SAMPLER: {
	// <<Context:Ptr, Normal:Bool, AddressMode:32, FilterMode:32>>
	pointer_t chandle;
	u_int32_t normal_arg;
	u_int32_t address_mode_ix;
	u_int32_t filter_mode_ix;
	ecl_object_t* ctx;

	if (get_pointer(&arg, &chandle) &&
	    ((ctx = context_object(env, chandle))) && 
	    get_uint32(&arg, &normal_arg) &&
	    get_uint32(&arg, &address_mode_ix) &&
	    (address_mode_ix < ADDRESSING_MODE_NUM) && 
	    get_uint32(&arg, &filter_mode_ix) &&
	    (filter_mode_ix < FILTER_MODE_NUM) && 
	    cbuf_eob(&arg)) {
	    cl_addressing_mode address_mode=
		kv_addressing_mode[address_mode_ix].value;
	    cl_filter_mode filter_mode=
		kv_filter_mode[filter_mode_ix].value;
	    cl_sampler sampler;

	    sampler = clCreateSampler(ctx->context, 
				      (cl_bool) (normal_arg != 0),
				      address_mode,
				      filter_mode,
				      &err);
	    if (sampler) {
		ecl_object_t* obj;
		
		if (!(obj = EclSamplerCreate(env,sampler))) {
		    clReleaseSampler(sampler);
		    RETURN_ERROR(CL_OUT_OF_HOST_MEMORY);
		}
		RETURN_OBJ(obj);
	    }
	}
	break;
    }

    case ECL_RETAIN_SAMPLER: {
	// <<Sampler:Ptr>>
	pointer_t handle;
	ecl_object_t* obj;

	if (get_pointer(&arg, &handle) &&
	    cbuf_eob(&arg) &&
	    ((obj = sampler_object(env, handle)) != 0)) {
	    ecl_object_retain(obj);
	    RETURN_OK();
	}
	break;
    }

    case ECL_RELEASE_SAMPLER: {
	// <<Sampler:Ptr>>
	pointer_t handle;
	ecl_object_t* obj;

	if (get_pointer(&arg, &handle) &&
	    cbuf_eob(&arg) &&
	    ((obj = sampler_object(env, handle)) != 0)) {
	    ecl_object_release(obj);
	    RETURN_OK();
	}
	break;
    }

    case ECL_GET_SAMPLER_INFO: {
	// <<Sampler:Ptr,  Info:32>>
	pointer_t  handle;
	u_int32_t  info_arg;
	ecl_object_t*    obj;

	if (get_pointer(&arg, &handle) && 
	    get_uint32(&arg, &info_arg) &&
	    cbuf_eob(&arg) &&
	    ((obj = sampler_object(env, handle)) != 0)) {
	    err = put_object_info(&reply, obj, info_arg);
	    if (err == CL_SUCCESS) 
		goto done;
	}
	break;
    }

    case ECL_CREATE_PROGRAM_WITH_SOURCE:
	ecl_create_program_with_source(env, &arg, &reply);
	goto done;

    case ECL_CREATE_PROGRAM_WITH_BINARY:
	ecl_create_program_with_binary(env, &arg, &reply);
	goto done;

    case ECL_RETAIN_PROGRAM: {
	// <<Program:Ptr>>
	pointer_t handle;
	ecl_object_t* obj;

	if (get_pointer(&arg, &handle) &&
	    cbuf_eob(&arg) &&
	    ((obj = program_object(env, handle)) != 0)) {
	    ecl_object_retain(obj);
	    RETURN_OK();
	}
	break;
    }

    case ECL_RELEASE_PROGRAM: {
	// <<Sampler:Ptr>>
	pointer_t handle;
	ecl_object_t* obj;

	if (get_pointer(&arg, &handle) &&
	    cbuf_eob(&arg) &&
	    ((obj = program_object(env, handle)) != 0)) {
	    ecl_object_release(obj);
	    RETURN_OK();
	}
	break;
    }

    case ECL_BUILD_PROGRAM: {
	// <<Program:Ptr, Num:32, Device:Ptr..., Options/binary>>
	pointer_t     handle;
	ecl_object_t* obj;
	cl_device_id  device_list[MAX_DEVICES];
	size_t        num_devices = MAX_DEVICES;

	if (get_pointer(&arg, &handle) &&
	    ((obj = program_object(env, handle))) &&
	    get_array(&arg,(get_fn_t)get_device,device_list,
		      sizeof(cl_device_id), &num_devices,env)) {
	    char options[MAX_OPTION_LIST];
	    size_t sz = cbuf_r_avail(&arg);
	    if (sz == 0) 
		options[0] = '\0';
	    else if (sz >= MAX_OPTION_LIST) {
		memcpy(options, cbuf_seg_ptr(&arg), MAX_OPTION_LIST);
		options[MAX_OPTION_LIST-1] = '\0';
	    }
	    else {
		memcpy(options, cbuf_seg_ptr(&arg), sz);
		options[sz] = '\0';
	    }
#ifdef ASYNC_BUILD_PROGRAM	    
	    {
		ecl_build_data_t* user_data;
		u_int32_t eref;

		if (!(user_data = driver_alloc(sizeof(ecl_build_data_t))))
		    RETURN_ERROR(CL_OUT_OF_HOST_MEMORY);
		eref = env->eref++;
		user_data->env = env;
		user_data->eref = eref;
		user_data->caller = driver_caller(env->port);
	    
		err = clBuildProgram(obj->program,
				     num_devices,
				     device_list,
				     (const char*) options,
				     ecl_build_notify,
				     user_data);
		if (err == CL_SUCCESS)
		    RETURN_EVENT(eref);
		else {
		    driver_free(user_data);
		}
	    }
#else
	    err = clBuildProgram(obj->program,
				 num_devices,
				 device_list,
				 (const char*) options,
				 0, 0);
#endif
	}
	break;
    }

    case ECL_UNLOAD_COMPILER: {
	// <<>>
	if (cbuf_eob(&arg))
	    err = clUnloadCompiler();
	break;
    }

    case ECL_GET_PROGRAM_INFO: {
	// <<Program:Ptr,  Info:32>>
	pointer_t  handle;
	u_int32_t  info_arg;
	ecl_object_t*    obj;

	if (get_pointer(&arg, &handle) && 
	    get_uint32(&arg, &info_arg) &&
	    cbuf_eob(&arg) &&
	    ((obj = program_object(env, handle)) != 0)) {
	    // Special case for PROGRAM binaries, must allocate special!
	    if ((info_arg < ECL_PROGRAM_INFO_NUM) &&
		(program_info[info_arg].info_id == CL_PROGRAM_BINARIES)) {
		err = put_program_binaries(&reply, obj->program);
	    }
	    else {
		err = put_object_info(&reply, obj, info_arg);
	    }
	    if (err == CL_SUCCESS)
		goto done;
	}
	break;
    }

    case ECL_GET_PROGRAM_BUILD_INFO: {
	// <<Program:Ptr, Device:Ptr, Info:32>>
	pointer_t  handle;
	pointer_t  dhandle;
	u_int32_t  info_arg;
	ecl_object_t* dev;
	ecl_object_t* obj;

	if (get_pointer(&arg, &handle) && 
	    get_pointer(&arg, &dhandle) && 
	    get_uint32(&arg, &info_arg) &&
	    cbuf_eob(&arg) &&
	    ((dev = device_object(env,dhandle))) &&
	    ((obj = program_object(env, handle)))) {
	    size_t returned_size = 0;
	    cl_ulong buf[2048];

	    if (info_arg < ECL_BUILD_INFO_NUM) {
		err = clGetProgramBuildInfo(obj->program,
					    dev->device,
					    build_info[info_arg].info_id,
					    sizeof(buf),
					    buf,
					    &returned_size);
		DBG("build_info: item=%d err=%d, returned_size=%ld",
		    info_arg, err, returned_size);

		if (err == CL_SUCCESS) {
		    cbuf_put_tuple_begin(&reply, 2);
		    cbuf_put_tag_ok(&reply);
		    put_value(&reply, env, &build_info[info_arg],
			      buf, returned_size);
		    cbuf_put_tuple_end(&reply, 2);
		    goto done;
		}
	    }
	}
	break;
    }

    case ECL_CREATE_KERNEL: {
	// <<Program:Ptr, NameLen:Size, Name/binary>>
	pointer_t      handle;
	ecl_object_t*  pobj;
	size_t         name_size;
	char kernel_name[MAX_KERNEL_NAME];

	if (get_pointer(&arg, &handle) && 
	    get_size(&arg, &name_size) &&
	    (name_size < MAX_KERNEL_NAME) &&
	    cbuf_read(&arg, kernel_name, name_size) &&
	    cbuf_eob(&arg) &&
	    ((pobj = program_object(env, handle)) != 0)) {
	    cl_kernel kernel;
	    kernel_name[name_size] = '\0';
	    DBG("clCreateKernel: size=%lu, '%s'", name_size, kernel_name);
	    kernel = clCreateKernel(pobj->program,
				    kernel_name,
				    &err);
	    if (kernel) {
		ecl_object_t* obj;
		if (!(obj = EclKernelCreate(env, kernel))) {
		    clReleaseKernel(kernel);
		    RETURN_ERROR(CL_OUT_OF_HOST_MEMORY);
		}
		RETURN_OBJ(obj);
	    }
	}
	break;
    }

    case ECL_CREATE_KERNELS_IN_PROGRAM: {
	// <<Program:Ptr>>
	pointer_t      handle;
	ecl_object_t*  pobj;	
	if (get_pointer(&arg, &handle) && 
	    cbuf_eob(&arg) &&
	    ((pobj = program_object(env, handle)) != 0)) {
	    cl_kernel kernels[MAX_KERNELS];
	    cl_uint num_kernels_ret;

	    err = clCreateKernelsInProgram(pobj->program,
					   MAX_KERNELS,
					   kernels,
					   &num_kernels_ret);
	    if (err == CL_SUCCESS) {
		ecl_object_t* kobj[MAX_KERNELS];
		int i = 0;
		cbuf_put_tuple_begin(&reply, 2);
		cbuf_put_tag_ok(&reply);
		cbuf_put_list_begin(&reply, num_kernels_ret);
		while(i < (int)num_kernels_ret) {
		    kobj[i] = EclKernelCreate(env, kernels[i]);
		    if (!kobj[i]) {
			while (i > 0) {
			    i--;
			    ecl_object_release(kobj[i]);
			}
			cbuf_reset(&reply, 0);
			cbuf_put_begin(&reply);  // reinit respose header
			RETURN_ERROR(CL_OUT_OF_HOST_MEMORY);
		    }
		    put_pointer(&reply, ecl_handle(kobj[i]));
		    i++;
		}
		cbuf_put_list_end(&reply, num_kernels_ret);
		cbuf_put_tuple_end(&reply, 2);
		goto done;
	    }
	}
	break;
    }

    case ECL_RETAIN_KERNEL: {
	// <<Kernel:Ptr>>
	pointer_t handle;
	ecl_object_t* obj;

	if (get_pointer(&arg, &handle) &&
	    cbuf_eob(&arg) &&
	    ((obj = kernel_object(env, handle)) != 0)) {
	    ecl_object_retain(obj);
	    RETURN_OK();
	}
	break;
    }

    case ECL_RELEASE_KERNEL: {
	// <<Kernel:Ptr>>
	pointer_t handle;
	ecl_object_t* obj;

	if (get_pointer(&arg, &handle) &&
	    cbuf_eob(&arg) &&
	    ((obj = kernel_object(env, handle)) != 0)) {
	    ecl_object_release(obj);
	    RETURN_OK();
	}
	break;
    }

    case ECL_SET_KERNEL_ARG: {
	err = ecl_set_kernel_arg(env, 0, &arg);
	break;
    }

    case ECL_SET_KERNEL_ARG_POINTER_T: {
	err = ecl_set_kernel_arg(env, POINTER, &arg);
	break;
    }

    case ECL_SET_KERNEL_ARG_SIZE_T: {
	err = ecl_set_kernel_arg(env, USIZE, &arg);
	break;
    }

    case ECL_GET_KERNEL_INFO: {
	// <<Kernel:Ptr,  Info:32>>
	pointer_t  handle;
	u_int32_t  info_arg;
	ecl_object_t*   obj;

	if (get_pointer(&arg, &handle) && 
	    get_uint32(&arg, &info_arg) &&
	    cbuf_eob(&arg) &&
	    ((obj = kernel_object(env, handle)) != 0)) {
	    err = put_object_info(&reply, obj, info_arg);
	    if (err == CL_SUCCESS)
		goto done;
	}
	break;	
    }

    case ECL_GET_KERNEL_WORKGROUP_INFO: {
	// <<Kernel:Ptr, Device:Ptr, Info:32>>
	pointer_t  handle;
	pointer_t  dhandle;
	u_int32_t  info_arg;
	ecl_object_t*   obj;
	ecl_object_t*   dev;

	if (get_pointer(&arg, &handle) && 
	    get_pointer(&arg, &dhandle) && 
	    get_uint32(&arg, &info_arg) &&
	    cbuf_eob(&arg) &&
	    ((obj = kernel_object(env, handle))) &&
	    ((dev = device_object(env, dhandle)))) {
	    size_t returned_size = 0;
	    cl_ulong buf[256];   // 2k buffer

	    if (info_arg < ECL_WORKGROUP_INFO_NUM) {
		err = clGetKernelWorkGroupInfo(obj->kernel,
					       dev->device,
					       workgroup_info[info_arg].info_id,
					       sizeof(buf),
					       buf, &returned_size);
		if (err == CL_SUCCESS) {
		    cbuf_put_tuple_begin(&reply, 2);
		    cbuf_put_tag_ok(&reply);
		    put_value(&reply, env, &workgroup_info[info_arg],
			      buf, returned_size);
		    cbuf_put_tuple_end(&reply, 2);
		    goto done;
		}
	    }
	}
	break;	
    }

    case ECL_RETAIN_EVENT: {
	// <<Event:Ptr>>
	pointer_t handle;
	ecl_object_t* obj;

	if (get_pointer(&arg, &handle) &&
	    cbuf_eob(&arg) &&
	    ((obj = event_object(env, handle)) != 0)) {
	    ecl_object_retain(obj);
	    RETURN_OK();
	}
	break;
    }

    case ECL_RELEASE_EVENT: {
	// <<Event:Ptr>>
	pointer_t handle;
	ecl_object_t* obj;

	if (get_pointer(&arg, &handle) &&
	    cbuf_eob(&arg) &&
	    ((obj = event_object(env, handle)) != 0)) {
	    ecl_object_release(obj);
	    RETURN_OK();
	}
	break;
    }

    case ECL_GET_EVENT_INFO: {
	// <<Event:Ptr,  Info:32>>
	pointer_t  handle;
	u_int32_t  info_arg;
	ecl_object_t*   obj;

	if (get_pointer(&arg, &handle) && 
	    get_uint32(&arg, &info_arg) &&
	    cbuf_eob(&arg) &&
	    ((obj = event_object(env, handle)) != 0)) {
	    err = put_object_info(&reply, obj, info_arg);
	    if (err == CL_SUCCESS)
		goto done;
	}
	break;	
    }

    case ECL_ENQUEUE_TASK: {
	// <<Queue:Ptr, Kernel:Ptr, 
	//   NumEvents:32, Event1:Ptr .. EventN:Ptr>>
	pointer_t  qh;
	pointer_t  kh;
	ecl_object_t* qobj;
	ecl_object_t* kobj;
	cl_event wait_list[MAX_WAIT_LIST];
	size_t num_wait = MAX_WAIT_LIST;

	if (get_pointer(&arg, &qh) && 
	    get_pointer(&arg, &kh) && 
	    (qobj = queue_object(env, qh)) &&
	    (kobj = kernel_object(env, kh)) &&
	    get_array(&arg,(get_fn_t)get_event,wait_list,sizeof(cl_event),
		      &num_wait,env) &&
	    cbuf_eob(&arg)) {
	    cl_event event;
	    err = clEnqueueTask(qobj->queue,
				kobj->kernel,
				(cl_uint) num_wait, 
				(num_wait ? wait_list : 0),
				&event);
	    if ((err == CL_SUCCESS) && event) {
		ecl_object_t* obj;
		if (!(obj = EclEventCreate(env, event))) {
		    clReleaseEvent(event);
		    RETURN_ERROR(CL_OUT_OF_HOST_MEMORY);
		}
		ecl_async_wait_event(obj);
		RETURN_OBJ(obj);
	    }
	}
	break;
    }

    case ECL_ENQUEUE_ND_RANGE_KERNEL: {
	// <<Queue:Ptr, Kernel:Ptr, WorkDim:size_t,
	//   Global:size_t,..Global:size_t, Local:size_t ... Local:size_t,
	//   NumEvents:32, Event1:Ptr .. EventN:Ptr>>
	pointer_t  qh;
	pointer_t  kh;
	ecl_object_t* qobj;
	ecl_object_t* kobj;
	size_t work_dim;

	if (get_pointer(&arg, &qh) && 
	    get_pointer(&arg, &kh) && 
	    get_size(&arg, &work_dim) &&
	    (work_dim > 0) && (work_dim <= MAX_WORK_SIZE) &&
	    (qobj = queue_object(env, qh)) &&
	    (kobj = kernel_object(env, kh)) &&
	    !cbuf_eob(&arg)) {
	    cl_event event;
	    const size_t* global_work_offset = 0;
	    size_t global_work_size[MAX_WORK_SIZE];
	    size_t local_work_size[MAX_WORK_SIZE];
	    cl_event wait_list[MAX_WAIT_LIST];
	    size_t num_wait = MAX_WAIT_LIST;

	    if (!get_narray(&arg,(get_fn_t)get_size,
			    global_work_size, sizeof(size_t), work_dim, 0))
		RETURN_ERROR(err);
	    if (!get_narray(&arg, (get_fn_t)get_size, 
			    local_work_size, sizeof(size_t), work_dim, 0))
		RETURN_ERROR(err);
	    if (!get_array(&arg,(get_fn_t)get_event,wait_list,sizeof(cl_event),
			   &num_wait,env))
		RETURN_ERROR(err);
	    if (!cbuf_eob(&arg))
		RETURN_ERROR(err);

#ifdef DEBUG
	    {
		int i;
		for (i = 0; i < (int)work_dim; i++) {
		    printf("global[%d] = %ld\r\n", i, global_work_size[i]);
		    printf("local[%d] = %ld\r\n", i,  local_work_size[i]);
		}
	    }
#endif

	    err = clEnqueueNDRangeKernel(qobj->queue, 
					 kobj->kernel,
					 work_dim, 
					 global_work_offset,
					 global_work_size,
					 local_work_size,
					 (cl_uint)num_wait,
					 (num_wait ? wait_list : 0),
					 &event);
	    if ((err == CL_SUCCESS) && event) {
		ecl_object_t* obj;
		if (!(obj = EclEventCreate(env, event))) {
		    clReleaseEvent(event);
		    RETURN_ERROR(CL_OUT_OF_HOST_MEMORY);
		}
		ecl_async_wait_event(obj);
		RETURN_OBJ(obj);
	    }
	}
	break;
    }

    case ECL_ENQUEUE_MARKER: {
	// <<Queue:Ptr>>
	pointer_t  qh;
	ecl_object_t* qobj;

	if (get_pointer(&arg, &qh) && 
	    (qobj = queue_object(env, qh)) &&
	    cbuf_eob(&arg)) {
	    cl_event event;
	    
	    err = clEnqueueMarker(qobj->queue, &event);
	    if (err == CL_SUCCESS) {
		ecl_object_t* obj;
		if (!(obj = EclEventCreate(env, event))) {
		    clReleaseEvent(event);
		    RETURN_ERROR(CL_OUT_OF_HOST_MEMORY);
		}
		ecl_async_wait_event(obj);
		RETURN_OBJ(obj);
	    }
	}
	break;
    }
	
    case ECL_ENQUEUE_WAIT_FOR_EVENT: {
	// <<Queue:Ptr,
	//   NumEvents:size_t, Event1:Ptr .. EventN:Ptr>>
	pointer_t  qh;
	ecl_object_t* qobj;
	cl_event wait_list[MAX_WAIT_LIST];
	size_t num_wait = MAX_WAIT_LIST;

	if (get_pointer(&arg, &qh) && 
	    (qobj = queue_object(env, qh)) &&
	    get_array(&arg,(get_fn_t)get_event,wait_list,sizeof(cl_event),
		      &num_wait,env) &&
	    cbuf_eob(&arg)) {
	    err = clEnqueueWaitForEvents(qobj->queue,
					 (cl_uint) num_wait,
					 (num_wait ? wait_list : 0));
	}
	break;
    }

    case ECL_ENQUEUE_BARRIER: {
	pointer_t  qh;
	ecl_object_t* qobj;

	if (get_pointer(&arg, &qh) && 
	    (qobj = queue_object(env, qh)) &&
	    cbuf_eob(&arg)) {
	    err = clEnqueueBarrier(qobj->queue);
	}
	break;
    }

    case ECL_ENQUEUE_READ_BUFFER: {
	// <<Queue:Ptr, Mem:Ptr, Offset:32, Cb:32, 
	//   NumEvents:size_t Event1:Ptr, ..EventN:Ptr>>
	pointer_t  qh;
	pointer_t  mh;
	ecl_object_t* qobj;
	ecl_object_t* mobj;
	u_int32_t offset_arg;
	u_int32_t cb_arg;
	cl_event wait_list[MAX_WAIT_LIST];
	size_t num_wait = MAX_WAIT_LIST;

	if (get_pointer(&arg, &qh) && 
	    get_pointer(&arg, &mh) && 
	    get_uint32(&arg, &offset_arg) &&
	    get_uint32(&arg, &cb_arg) &&
	    (qobj = queue_object(env, qh)) &&
	    (mobj = mem_object(env, mh)) &&
	    get_array(&arg,(get_fn_t)get_event,wait_list,sizeof(cl_event),
		      &num_wait,env) &&
	    cbuf_eob(&arg)) {
	    cl_event event;
	    ErlDrvBinary* bin;
	    
	    if (!(bin = driver_alloc_binary(cb_arg)))
		RETURN_ERROR(CL_OUT_OF_HOST_MEMORY);
	    err = clEnqueueReadBuffer(qobj->queue, mobj->mem,
				      CL_FALSE,
				      offset_arg,
				      cb_arg,
				      (void*) bin->orig_bytes,
				      (cl_uint) num_wait,
				      (num_wait ? wait_list : 0),
				      &event);
	    if (err == CL_SUCCESS) {
		ecl_object_t* obj;
		if (!(obj = EclEventCreate(env, event))) {
		    clReleaseEvent(event);
		    driver_free_binary(bin);
		    RETURN_ERROR(CL_OUT_OF_HOST_MEMORY);
		}
		// will return the binary as an event when done
		ecl_async_wait_event_bin(obj, bin);
		RETURN_OBJ(obj);
	    }
	}
	break;
    }

    case ECL_ENQUEUE_WRITE_BUFFER: {
	ecl_enqueue_write_buffer(env, &arg, &reply);
	goto done;
    }

    default:
	break;
    }

    if (err == CL_SUCCESS)
	RETURN_OK();
    else
	RETURN_ERROR(err);

done:
    cbuf_put_end(&reply);  // put response footer
    
    CBUF_DBG(&reply, "ctl_reply");

    if (reply.vlen == 1) {
	if (reply.v[0].bp) {
	    cbuf_trim(&reply);
	    *rbuf = (char*) reply.v[0].bp;
	}
	else
	    *rbuf = (char*) reply.v[0].base;
	return cbuf_seg_used(&reply);
    }
    else {
	// FIXME return a event handle and do a send_term
	// example: get_program_info(Program, binaries) 
	fprintf(stderr, "FIXME:vector reply!\r\n");
	return 0;
    }
}

//
// <<Cmd:8, ID:16, Command/binary>>
//
// Command:
//   ECL_CREATE_BUFFER:
//        <<Context:Ptr, Flags:32, Size:32, Data/rest>>
//
//   ECL_CREATE_PROGRAM_WITH_SOURCE
//	  <<Context:Ptr, Source/binary>>
//
//   ECL_CREATE_PROGRAM_WITH_BINARY
//        <<Context:Ptr, 
//             NumDevices:Size,  Device1:Ptr, ... DeviceN:Ptr,
//             NumBinaries:Size, Size1:32, Binary:Size1 ... SizeN:32, Binary:SizeN>>
//
//   ECL_ENQUEUE_WRITE_BUFFER
//        <<Queue:Ptr, Mem:Ptr, Offset:32, Cb:32,
//             NumEvents:Size Event1:Ptr, ..EventN:Ptr>>
//             Data/binary>>
// 
static void ecl_drv_commandv(ErlDrvData d, ErlIOVec* ev)
{
    ecl_env_t* env = (ecl_env_t*) d;
    cbuf_t     arg;    // argument data stream
    cbuf_t     reply;  // reply data stream
    u_int8_t   cmd;
    u_int32_t  cmd_ref;
    ErlDrvTermData term_data[10];

    cbuf_initv(&arg, ev);
    CBUF_DBG(&arg, "async_arg");
    cbuf_init(&reply, 0, 0, 0, CBUF_FLAG_BINARY | ECL_REPLY_TYPE);
    cbuf_put_begin(&reply);  // put respose header

    if (get_uint8(&arg, &cmd) &&
	get_uint32(&arg, &cmd_ref)) {
	DBG("Async Cmd: %02X ref=%u", cmd, cmd_ref);
	switch(cmd) {
	case ECL_CREATE_BUFFER:
	    ecl_create_buffer(env, &arg, &reply);
	    break;
	case ECL_CREATE_PROGRAM_WITH_SOURCE:
	    ecl_create_program_with_source(env, &arg, &reply);
	    break;
	case ECL_CREATE_PROGRAM_WITH_BINARY:
	    ecl_create_program_with_binary(env, &arg, &reply);
	    break;
	case ECL_ENQUEUE_WRITE_BUFFER:
	    ecl_enqueue_write_buffer(env, &arg, &reply);
	    break;
	default:
	    put_error(&reply, CL_INVALID_VALUE);	    
	    break;
	}
    }
    else {
	put_error(&reply, CL_INVALID_VALUE);
    }
    cbuf_put_end(&reply);  // put response footer

    // send reply to caller {cl_reply, CmdRef, ReplyData}
    cbuf_trim(&reply);
    A_DBG("ecl_drv_commandv: cmd=%02X, cmd_ref=%u", 
	  cmd, cmd_ref);
    CBUF_DBG(&reply, "async_reply");
    term_data[0] = ERL_DRV_ATOM;
    term_data[1] = driver_mk_atom("cl_reply");
    term_data[2] = ERL_DRV_UINT;
    term_data[3] = cmd_ref;
    term_data[4] = ERL_DRV_BINARY;
    term_data[5] = (ErlDrvTermData) reply.v[0].bp;
    term_data[6] = cbuf_seg_used(&reply);
    term_data[7] = 0;
    term_data[8] = ERL_DRV_TUPLE;
    term_data[9] = 3;

    driver_send_term(env->port, driver_caller(env->port), term_data, 10);
    cbuf_final(&reply);
}

static void ecl_drv_timeout(ErlDrvData d)
{
    (void) d;
}

