//
// NIF interface for OpenCL binding
//
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>

#ifndef WIN32
#include <stdbool.h>
#else
#include <windows.h>
#endif

#ifdef DARWIN
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

// Old cl_platform doesn't have the CL_CALLBACK
#ifndef CL_CALLBACK
#define CL_CALLBACK
#endif

#ifdef WIN32
typedef cl_bool bool;
#define true 1
#define false 0
#endif

#define UNUSED(a) ((void) a)

#include "erl_nif.h"
#include "cl_hash.h"

#define sizeof_array(a) (sizeof(a) / sizeof(a[0]))

#ifdef DEBUG
#include <stdarg.h>
static void ecl_emit_error(char* file, int line, ...);
#define DBG(...) ecl_emit_error(__FILE__,__LINE__,__VA_ARGS__)
#else
#define DBG(...)
#endif

#define CL_ERROR(...) ecl_emit_error(__FILE__,__LINE__,__VA_ARGS__)

// soft limits
#define MAX_INFO_SIZE   1024
#define MAX_DEVICES     128   
#define MAX_PLATFORMS   128   
#define MAX_OPTION_LIST 1024
#define MAX_KERNEL_NAME 1024
#define MAX_KERNELS     1024
#define MAX_SOURCES     128
#define MAX_WAIT_LIST   128
#define MAX_WORK_SIZE   3
#define MAX_IMAGE_FORMATS 128

// Atom macros
#define ATOM(name) atm_##name

#define DECL_ATOM(name) \
    ERL_NIF_TERM atm_##name = 0

// require env in context (ugly)
#define LOAD_ATOM(name)			\
    atm_##name = enif_make_atom(env,#name)

#define LOAD_ATOM_STRING(name,string)			\
    atm_##name = enif_make_atom(env,string)

// Wrapper to handle reource atom name etc.
typedef struct {
    char* name;
    ERL_NIF_TERM type;         // resource atom name
    ErlNifResourceType* res;   // the resource type
    size_t              size;  // "real" object size
} ecl_resource_t;

typedef struct _ecl_env_t {
    lhash_t     ref;        // cl -> ecl
    ErlNifRWLock* ref_lock; // lhash operation lock
} ecl_env_t;

typedef struct _ecl_object_t {
    lhash_bucket_t        hbucket;   // inheritance: map: cl->ecl
    ecl_env_t*            env;
    struct _ecl_object_t* parent;     // parent resource object
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

// "inherits" ecl_object_t and add special binary objects (read/write)
typedef struct _ecl_event_t {
    ecl_object_t obj;       // FIXED place for inhertiance
    bool          rd;       // Read binary operation
    bool          rl;       // Do not release if true
    ErlNifEnv*    bin_env;  // environment to hold binary term data
    ErlNifBinary* bin;      // read/write data
} ecl_event_t;

#define KERNEL_ARG_OTHER   0
#define KERNEL_ARG_MEM     1
#define KERNEL_ARG_SAMPLER 2

// This is a special construct inorder to kee
typedef struct {
    int type;    // 0=other, 1=mem, 2=samper
    union {
	cl_mem      mem;
	cl_sampler  sampler;
	void*       other;
	void*       value;
    };
} ecl_kernel_arg_t;

// "inherits" ecl_object_t and reference count kernel args
typedef struct _ecl_kernel_t {
    ecl_object_t      obj;       // FIXED place for inhertiance
    cl_uint           num_args;  // number of arguments used by the kernel
    ecl_kernel_arg_t* arg;       // array of current args 
} ecl_kernel_t;


typedef enum {
    OCL_CHAR,          // cl_char
    OCL_UCHAR,         // cl_uchar
    OCL_SHORT,         // cl_short
    OCL_USHORT,        // cl_ushort
    OCL_INT,           // cl_int
    OCL_UINT,          // cl_uint
    OCL_LONG,          // cl_long
    OCL_ULONG,         // cl_ulong
    OCL_HALF,          // cl_half
    OCL_FLOAT,         // cl_float
    OCL_DOUBLE,        // cl_double
    OCL_BOOL,          // cl_bool 
    OCL_STRING,        // cl_char*
    OCL_BITFIELD,      // cl_ulong
    OCL_ENUM,          // cl_int
    OCL_POINTER,       // void*
    OCL_SIZE,          // size_t
    OCL_PLATFORM,      // void*
    OCL_DEVICE,        // void*
    OCL_CONTEXT,       // void*
    OCL_PROGRAM,       // void*
    OCL_COMMAND_QUEUE, // void*
    OCL_IMAGE_FORMAT   // cl_image_format
} ocl_type_t;

#define OCL_DEVICE_TYPE                  OCL_BITFIELD
#define OCL_DEVICE_FP_CONFIG             OCL_BITFIELD
#define OCL_DEVICE_GLOBAL_MEM_CACHE_TYPE OCL_ENUM
#define OCL_PLATFORM_INFO                OCL_UINT
#define OCL_DEVICE_INFO                  OCL_UINT
#define OCL_DEVICE_FP_CONFIG             OCL_BITFIELD
#define OCL_DEVICE_EXEC_CAPABILITIES     OCL_BITFIELD
#define OCL_QUEUE_PROPERTIES             OCL_BITFIELD
#define OCL_DEVICE_LOCAL_MEM_TYPE        OCL_ENUM
#define OCL_MEM_OBJECT_TYPE              OCL_ENUM
#define OCL_MEM_FLAGS                    OCL_BITFIELD
#define OCL_SAMPLER_ADDRESSING_MODE      OCL_ENUM
#define OCL_SAMPLER_FILTER_MODE          OCL_ENUM
#define OCL_BUILD_STATUS                 OCL_ENUM

typedef struct {
    ERL_NIF_TERM*  key;
    ErlNifUInt64   value;
} ecl_kv_t;

typedef struct {
    ERL_NIF_TERM*  info_key;    // Atom
    cl_uint        info_id;     // Information
    bool           is_array;    // return type is a vector of data
    ocl_type_t     info_type;   // info data type
    void*          extern_info; // Encode/Decode enum/bitfields
} ecl_info_t;

typedef enum {
    ECL_MESSAGE_STOP,           // time to die
    ECL_MESSAGE_FLUSH,          // call clFlush
    ECL_MESSAGE_FINISH,         // call clFinish
    ECL_MESSAGE_WAIT_FOR_EVENT  // call clWaitForEvents (only one event!)
} ecl_message_type_t;

struct _ecl_thread_t;

typedef struct ecl_message_t
{
    ecl_message_type_t type;
    ErlNifPid        sender;  // sender pid
    ErlNifEnv*          env;  // message environment (ref, bin's etc)
    ERL_NIF_TERM        ref;  // ref (in env!)
    union {
	ecl_object_t* queue;  // ECL_MESSAGE_FLUSH/ECL_MESSAGE_FINISH
	ecl_event_t* event;   // ECL_MESSAGE_WAIT_FOR_EVENT
    };
} ecl_message_t;

typedef struct _ecl_qlink_t {
    struct _ecl_qlink_t* next;
    ecl_message_t mesg;
} ecl_qlink_t;

#define MAX_QLINK  8  // pre-allocated qlinks

typedef struct {
    ErlNifMutex*   mtx;
    ErlNifCond*    cv;
    int len;
    ecl_qlink_t*   front;   // pick from front
    ecl_qlink_t*   rear;    // insert at rear
    ecl_qlink_t*   free;    // free list in ql
    ecl_qlink_t  ql[MAX_QLINK];  // "pre" allocated qlinks
} ecl_queue_t;

typedef struct _ecl_thread_t {
    ErlNifTid   tid;     // thread id
    ecl_queue_t q;       // message queue
    void*       arg;     // thread init argument
} ecl_thread_t;

// "inherits" ecl_object_t and add keep track of the context thread
typedef struct _ecl_context_t {
    ecl_object_t obj;     // FIXED place for inhertiance
    ecl_thread_t* thr;    // The context thread
} ecl_context_t;

static void* ecl_context_main(void* arg);


static int ecl_load(ErlNifEnv* env, void** priv_data, ERL_NIF_TERM load_info);

static int ecl_reload(ErlNifEnv* env, void** priv_data, ERL_NIF_TERM load_info);

static int ecl_upgrade(ErlNifEnv* env, void** priv_data, void** old_priv_data, 
			 ERL_NIF_TERM load_info);

static void ecl_unload(ErlNifEnv* env, void* priv_data);


static ERL_NIF_TERM ecl_noop(ErlNifEnv* env, int argc, 
			    const ERL_NIF_TERM argv[]);

static ERL_NIF_TERM ecl_get_platform_ids(ErlNifEnv* env, int argc, 
					 const ERL_NIF_TERM argv[]);

static ERL_NIF_TERM ecl_get_platform_info(ErlNifEnv* env, int argc, 
					  const ERL_NIF_TERM argv[]);

static ERL_NIF_TERM ecl_get_device_ids(ErlNifEnv* env, int argc, 
				       const ERL_NIF_TERM argv[]);

static ERL_NIF_TERM ecl_get_device_info(ErlNifEnv* env, int argc, 
					const ERL_NIF_TERM argv[]);

static ERL_NIF_TERM ecl_create_context(ErlNifEnv* env, int argc, 
				       const ERL_NIF_TERM argv[]);

static ERL_NIF_TERM ecl_get_context_info(ErlNifEnv* env, int argc, 
					 const ERL_NIF_TERM argv[]);

static ERL_NIF_TERM ecl_create_queue(ErlNifEnv* env, int argc, 
				     const ERL_NIF_TERM argv[]);

static ERL_NIF_TERM ecl_get_queue_info(ErlNifEnv* env, int argc, 
				       const ERL_NIF_TERM argv[]);

static ERL_NIF_TERM ecl_create_buffer(ErlNifEnv* env, int argc, 
				      const ERL_NIF_TERM argv[]);

static ERL_NIF_TERM ecl_create_image2d(ErlNifEnv* env, int argc, 
				       const ERL_NIF_TERM argv[]);

static ERL_NIF_TERM ecl_create_image3d(ErlNifEnv* env, int argc, 
				       const ERL_NIF_TERM argv[]);

static ERL_NIF_TERM ecl_get_supported_image_formats(ErlNifEnv* env, int argc, 
						    const ERL_NIF_TERM argv[]);

static ERL_NIF_TERM ecl_get_mem_object_info(ErlNifEnv* env, int argc, 
					    const ERL_NIF_TERM argv[]);

static ERL_NIF_TERM ecl_get_image_info(ErlNifEnv* env, int argc, 
				       const ERL_NIF_TERM argv[]);

static ERL_NIF_TERM ecl_create_sampler(ErlNifEnv* env, int argc, 
				       const ERL_NIF_TERM argv[]);

static ERL_NIF_TERM ecl_get_sampler_info(ErlNifEnv* env, int argc, 
					 const ERL_NIF_TERM argv[]);

static ERL_NIF_TERM ecl_create_program_with_source(ErlNifEnv* env, int argc, 
						   const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM ecl_create_program_with_binary(ErlNifEnv* env, int argc, 
						   const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM ecl_async_build_program(ErlNifEnv* env, int argc, 
					    const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM ecl_unload_compiler(ErlNifEnv* env, int argc, 
					const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM ecl_get_program_info(ErlNifEnv* env, int argc, 
					 const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM ecl_get_program_build_info(ErlNifEnv* env, int argc, 
					       const ERL_NIF_TERM argv[]);

static ERL_NIF_TERM ecl_create_kernel(ErlNifEnv* env, int argc, 
				      const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM ecl_create_kernels_in_program(ErlNifEnv* env, int argc, 
						  const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM ecl_set_kernel_arg(ErlNifEnv* env, int argc, 
				       const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM ecl_set_kernel_arg_size(ErlNifEnv* env, int argc, 
					    const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM ecl_get_kernel_info(ErlNifEnv* env, int argc, 
					const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM ecl_get_kernel_workgroup_info(ErlNifEnv* env, int argc, 
						  const ERL_NIF_TERM argv[]);

static ERL_NIF_TERM ecl_enqueue_task(ErlNifEnv* env, int argc, 
				     const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM ecl_enqueue_nd_range_kernel(ErlNifEnv* env, int argc, 
						const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM ecl_enqueue_marker(ErlNifEnv* env, int argc, 
				       const ERL_NIF_TERM argv[]);

static ERL_NIF_TERM ecl_enqueue_barrier(ErlNifEnv* env, int argc, 
					const ERL_NIF_TERM argv[]);

static ERL_NIF_TERM ecl_enqueue_wait_for_events(ErlNifEnv* env, int argc, 
						const ERL_NIF_TERM argv[]);

static ERL_NIF_TERM ecl_enqueue_read_buffer(ErlNifEnv* env, int argc, 
					    const ERL_NIF_TERM argv[]);

static ERL_NIF_TERM ecl_enqueue_write_buffer(ErlNifEnv* env, int argc, 
					     const ERL_NIF_TERM argv[]);

static ERL_NIF_TERM ecl_enqueue_read_image(ErlNifEnv* env, int argc, 
					   const ERL_NIF_TERM argv[]);

static ERL_NIF_TERM ecl_enqueue_write_image(ErlNifEnv* env, int argc, 
					    const ERL_NIF_TERM argv[]);

static ERL_NIF_TERM ecl_enqueue_copy_image(ErlNifEnv* env, int argc, 
					   const ERL_NIF_TERM argv[]);

static ERL_NIF_TERM ecl_enqueue_copy_image_to_buffer(ErlNifEnv* env, int argc, 
						     const ERL_NIF_TERM argv[]);

static ERL_NIF_TERM ecl_enqueue_copy_buffer_to_image(ErlNifEnv* env, int argc, 
						     const ERL_NIF_TERM argv[]);

static ERL_NIF_TERM ecl_enqueue_map_buffer(ErlNifEnv* env, int argc, 
					   const ERL_NIF_TERM argv[]);

static ERL_NIF_TERM ecl_enqueue_map_image(ErlNifEnv* env, int argc, 
					  const ERL_NIF_TERM argv[]);

static ERL_NIF_TERM ecl_enqueue_unmap_mem_object(ErlNifEnv* env, int argc, 
						 const ERL_NIF_TERM argv[]);

static ERL_NIF_TERM ecl_async_flush(ErlNifEnv* env, int argc, 
				    const ERL_NIF_TERM argv[]);

static ERL_NIF_TERM ecl_async_finish(ErlNifEnv* env, int argc, 
				     const ERL_NIF_TERM argv[]);

// speical version of clWaitForEvents 
static ERL_NIF_TERM ecl_async_wait_for_event(ErlNifEnv* env, int argc, 
					     const ERL_NIF_TERM argv[]);

static ERL_NIF_TERM ecl_get_event_info(ErlNifEnv* env, int argc, 
				       const ERL_NIF_TERM argv[]);


ErlNifFunc ecl_funcs[] =
{
    { "noop",                        0, ecl_noop },

    // Platform
    { "get_platform_ids",           0, ecl_get_platform_ids },
    { "get_platform_info",          2, ecl_get_platform_info },

    // Devices
    { "get_device_ids",             2, ecl_get_device_ids },
    { "get_device_info",            2, ecl_get_device_info },

    // Context
    { "create_context",             1, ecl_create_context },
    { "get_context_info",           2, ecl_get_context_info },

    // Command queue
    { "create_queue",               3, ecl_create_queue },
    { "get_queue_info",             2, ecl_get_queue_info },

    // Memory object
    { "create_buffer",              4, ecl_create_buffer },
    { "get_mem_object_info",        2, ecl_get_mem_object_info },
    { "get_image_info",             2, ecl_get_image_info },

    { "create_image2d",            7, ecl_create_image2d },
    { "create_image3d",            9, ecl_create_image3d },
    { "get_supported_image_formats",3, ecl_get_supported_image_formats },

    // Sampler 
    { "create_sampler",             4, ecl_create_sampler },
    { "get_sampler_info",           2, ecl_get_sampler_info },

    // Program
    { "create_program_with_source", 2, ecl_create_program_with_source },
    { "create_program_with_binary", 3, ecl_create_program_with_binary },
    { "async_build_program",        3, ecl_async_build_program },
    { "unload_compiler",            0, ecl_unload_compiler },
    { "get_program_info",           2, ecl_get_program_info },
    { "get_program_build_info",     3, ecl_get_program_build_info },

    // Kernel
    { "create_kernel",              2, ecl_create_kernel },
    { "create_kernels_in_program",  1, ecl_create_kernels_in_program },
    { "set_kernel_arg",             3, ecl_set_kernel_arg },
    { "set_kernel_arg_size",        3, ecl_set_kernel_arg_size },
    { "get_kernel_info",            2, ecl_get_kernel_info },
    { "get_kernel_workgroup_info",  3, ecl_get_kernel_workgroup_info },

    // Events
    { "enqueue_task",               3, ecl_enqueue_task },
    { "enqueue_nd_range_kernel",    5, ecl_enqueue_nd_range_kernel },
    { "enqueue_marker",             1, ecl_enqueue_marker },
    { "enqueue_barrier",            1, ecl_enqueue_barrier },
    { "enqueue_wait_for_events",    2, ecl_enqueue_wait_for_events },
    { "enqueue_read_buffer",        5, ecl_enqueue_read_buffer },
    { "enqueue_write_buffer",       6, ecl_enqueue_write_buffer },
    { "enqueue_read_image",         7, ecl_enqueue_read_image },
    { "enqueue_write_image",        8, ecl_enqueue_write_image },
    { "enqueue_copy_image",         6, ecl_enqueue_copy_image },
    { "enqueue_copy_image_to_buffer", 7, ecl_enqueue_copy_image_to_buffer },
    { "enqueue_copy_buffer_to_image", 7, ecl_enqueue_copy_buffer_to_image },
    { "enqueue_map_buffer",           6, ecl_enqueue_map_buffer },
    { "enqueue_map_image",            6, ecl_enqueue_map_image },
    { "enqueue_unmap_mem_object",     3, ecl_enqueue_unmap_mem_object },
    { "async_flush",                  1, ecl_async_flush },
    { "async_finish",                 1, ecl_async_finish },
    { "async_wait_for_event",         1, ecl_async_wait_for_event },
    { "get_event_info",               2, ecl_get_event_info }
};

static ecl_resource_t platform_r;
static ecl_resource_t device_r;
static ecl_resource_t context_r;
static ecl_resource_t command_queue_r;
static ecl_resource_t mem_r;
static ecl_resource_t sampler_r;
static ecl_resource_t program_r;
static ecl_resource_t kernel_r;
static ecl_resource_t event_r;

// General atoms
DECL_ATOM(ok);
DECL_ATOM(error);
DECL_ATOM(unknown);
DECL_ATOM(undefined);
DECL_ATOM(true);
DECL_ATOM(false);

// async messages
DECL_ATOM(cl_async);
DECL_ATOM(cl_event);

// Type names
DECL_ATOM(platform_t);
DECL_ATOM(device_t);
DECL_ATOM(context_t);
DECL_ATOM(command_queue_t);
DECL_ATOM(mem_t);
DECL_ATOM(sampler_t);
DECL_ATOM(program_t);
DECL_ATOM(kernel_t);
DECL_ATOM(event_t);

// 'cl' type names
DECL_ATOM(char);
DECL_ATOM(char2);
DECL_ATOM(char4);
DECL_ATOM(char8);
DECL_ATOM(char16);

DECL_ATOM(uchar);
DECL_ATOM(uchar2);
DECL_ATOM(uchar4);
DECL_ATOM(uchar8);
DECL_ATOM(uchar16);

DECL_ATOM(short);
DECL_ATOM(short2);
DECL_ATOM(short4);
DECL_ATOM(short8);
DECL_ATOM(short16);

DECL_ATOM(ushort);
DECL_ATOM(ushort2);
DECL_ATOM(ushort4);
DECL_ATOM(ushort8);
DECL_ATOM(ushort16);

DECL_ATOM(int);
DECL_ATOM(int2);
DECL_ATOM(int4);
DECL_ATOM(int8);
DECL_ATOM(int16);

DECL_ATOM(uint);
DECL_ATOM(uint2);
DECL_ATOM(uint4);
DECL_ATOM(uint8);
DECL_ATOM(uint16);

DECL_ATOM(long);
DECL_ATOM(long2);
DECL_ATOM(long4);
DECL_ATOM(long8);
DECL_ATOM(long16);

DECL_ATOM(ulong);
DECL_ATOM(ulong2);
DECL_ATOM(ulong4);
DECL_ATOM(ulong8);
DECL_ATOM(ulong16);

DECL_ATOM(half);

DECL_ATOM(float);
DECL_ATOM(float2);
DECL_ATOM(float4);
DECL_ATOM(float8);
DECL_ATOM(float16);

DECL_ATOM(double);
DECL_ATOM(double2);
DECL_ATOM(double4);
DECL_ATOM(double8);
DECL_ATOM(double16);

// Device info
DECL_ATOM(type);
DECL_ATOM(vendor_id);
DECL_ATOM(max_compute_units);
DECL_ATOM(max_work_item_dimensions);
DECL_ATOM(max_work_group_size);
DECL_ATOM(max_work_item_sizes);
DECL_ATOM(preferred_vector_width_char);
DECL_ATOM(preferred_vector_width_short);
DECL_ATOM(preferred_vector_width_int);
DECL_ATOM(preferred_vector_width_long);
DECL_ATOM(preferred_vector_width_float);
DECL_ATOM(preferred_vector_width_double);
DECL_ATOM(max_clock_frequency);
DECL_ATOM(address_bits);
DECL_ATOM(max_read_image_args);
DECL_ATOM(max_write_image_args);
DECL_ATOM(max_mem_alloc_size);
DECL_ATOM(image2d_max_width);
DECL_ATOM(image2d_max_height);
DECL_ATOM(image3d_max_width);
DECL_ATOM(image3d_max_height);
DECL_ATOM(image3d_max_depth);
DECL_ATOM(image_support);
DECL_ATOM(max_parameter_size);
DECL_ATOM(max_samplers);
DECL_ATOM(mem_base_addr_align);
DECL_ATOM(min_data_type_align_size);
DECL_ATOM(single_fp_config);
DECL_ATOM(global_mem_cache_type);
DECL_ATOM(global_mem_cacheline_size);
DECL_ATOM(global_mem_cache_size);
DECL_ATOM(global_mem_size);
DECL_ATOM(max_constant_buffer_size);
DECL_ATOM(max_constant_args);
DECL_ATOM(local_mem_type);
DECL_ATOM(local_mem_size);
DECL_ATOM(error_correction_support);
DECL_ATOM(profiling_timer_resolution);
DECL_ATOM(endian_little);
DECL_ATOM(available);
DECL_ATOM(compiler_available);
DECL_ATOM(execution_capabilities);
DECL_ATOM(queue_properties);
DECL_ATOM(name);
DECL_ATOM(vendor);
DECL_ATOM(driver_version);
DECL_ATOM(profile);
DECL_ATOM(version);
DECL_ATOM(extensions);
DECL_ATOM(platform);

// Platform info
// DECL_ATOM(profile);
// DECL_ATOM(version);
// DECL_ATOM(name);
// DECL_ATOM(vendor);
// DECL_ATOM(extensions);

// Context info
DECL_ATOM(reference_count);
DECL_ATOM(devices);
DECL_ATOM(properties);

// Queue info
DECL_ATOM(context);
DECL_ATOM(num_devices);
DECL_ATOM(device);
// DECL_ATOM(reference_count);
// DECL_ATOM(properties);

// Mem info
DECL_ATOM(object_type);
DECL_ATOM(flags);
DECL_ATOM(size);
DECL_ATOM(host_ptr);
DECL_ATOM(map_count);
// DECL_ATOM(reference_count); 
// DECL_ATOM(context);

// Image info
DECL_ATOM(format);
DECL_ATOM(element_size);
DECL_ATOM(row_pitch);
DECL_ATOM(slice_pitch);
DECL_ATOM(width);
DECL_ATOM(height);
DECL_ATOM(depth);

// Sampler info
// DECL_ATOM(reference_count);
// DECL_ATOM(context);
DECL_ATOM(normalized_coords);
DECL_ATOM(addressing_mode);
DECL_ATOM(filter_mode);

// Program info
// DECL_ATOM(reference_count);
// DECL_ATOM(context);
DECL_ATOM(num_decices);
// DECL_ATOM(devices);
DECL_ATOM(source); 
DECL_ATOM(binary_sizes);
DECL_ATOM(binaries);

// Build Info
DECL_ATOM(status);
DECL_ATOM(options);
DECL_ATOM(log);

// Kernel Info
DECL_ATOM(function_name);
DECL_ATOM(num_args);
// DECL_ATOM(reference_count);
// DECL_ATOM(context);
DECL_ATOM(program);

// Event Info
DECL_ATOM(command_queue);
DECL_ATOM(command_type);
// DECL_ATOM(reference_count);
DECL_ATOM(execution_status);

// Workgroup info
DECL_ATOM(work_group_size);
DECL_ATOM(compile_work_group_size);
// DECL_ATOM(local_mem_size);

// Error codes
DECL_ATOM(device_not_found);
DECL_ATOM(device_not_available);
DECL_ATOM(compiler_not_available);
DECL_ATOM(mem_object_allocation_failure);
DECL_ATOM(out_of_resources);
DECL_ATOM(out_of_host_memory);
DECL_ATOM(profiling_info_not_available);
DECL_ATOM(mem_copy_overlap);
DECL_ATOM(image_format_mismatch);
DECL_ATOM(image_format_not_supported);
DECL_ATOM(build_program_failure);
DECL_ATOM(map_failure);
DECL_ATOM(invalid_value);
DECL_ATOM(invalid_device_type);
DECL_ATOM(invalid_platform);
DECL_ATOM(invalid_device);
DECL_ATOM(invalid_context);
DECL_ATOM(invalid_queue_properties);
DECL_ATOM(invalid_command_queue);
DECL_ATOM(invalid_host_ptr);
DECL_ATOM(invalid_mem_object);
DECL_ATOM(invalid_image_format_descriptor);
DECL_ATOM(invalid_image_size);
DECL_ATOM(invalid_sampler);
DECL_ATOM(invalid_binary);
DECL_ATOM(invalid_build_options);
DECL_ATOM(invalid_program);
DECL_ATOM(invalid_program_executable);
DECL_ATOM(invalid_kernel_name);
DECL_ATOM(invalid_kernel_definition);
DECL_ATOM(invalid_kernel);
DECL_ATOM(invalid_arg_index);
DECL_ATOM(invalid_arg_value);
DECL_ATOM(invalid_arg_size);
DECL_ATOM(invalid_kernel_args);
DECL_ATOM(invalid_work_dimension);
DECL_ATOM(invalid_work_group_size);
DECL_ATOM(invalid_work_item_size);
DECL_ATOM(invalid_global_offset);
DECL_ATOM(invalid_event_wait_list);
DECL_ATOM(invalid_event);
DECL_ATOM(invalid_operation);
DECL_ATOM(invalid_gl_object);
DECL_ATOM(invalid_buffer_size);
DECL_ATOM(invalid_mip_level);

// cl_device_type
DECL_ATOM(all);
DECL_ATOM(default);
DECL_ATOM(cpu);
DECL_ATOM(gpu);
DECL_ATOM(accelerator);

// fp_config
DECL_ATOM(denorm);
DECL_ATOM(inf_nan);
DECL_ATOM(round_to_nearest);
DECL_ATOM(round_to_zero);
DECL_ATOM(round_to_inf);
DECL_ATOM(fma);

// mem_cache_type
DECL_ATOM(none);
DECL_ATOM(read_only);
DECL_ATOM(read_write);

// local_mem_type
DECL_ATOM(local);
DECL_ATOM(global);

// exec capability
DECL_ATOM(kernel);
DECL_ATOM(native_kernel);

// command_queue_properties
DECL_ATOM(out_of_order_exec_mode_enable);
DECL_ATOM(profiling_enable);

// mem_flags
// DECL_ATOM(read_write);
DECL_ATOM(write_only);
// DECL_ATOM(read_only);
DECL_ATOM(use_host_ptr);
DECL_ATOM(alloc_host_ptr);
DECL_ATOM(copy_host_ptr);

// mem_object_type
DECL_ATOM(buffer);
DECL_ATOM(image2d);
DECL_ATOM(image3d);

// addressing_mode
// DECL_ATOM(none);
DECL_ATOM(clamp_to_edge);
DECL_ATOM(clamp);
DECL_ATOM(repeat);

// filter_mode
DECL_ATOM(nearest);
DECL_ATOM(linear);

// map_flags
DECL_ATOM(read);
DECL_ATOM(write);

// build_status
DECL_ATOM(success);
// DECL_ATOM(none);
// DECL_ATOM(error);
DECL_ATOM(in_progress);

// command_type
DECL_ATOM(ndrange_kernel);
DECL_ATOM(task);
// DECL_ATOM(native_kernel);
DECL_ATOM(read_buffer);
DECL_ATOM(write_buffer);
DECL_ATOM(copy_buffer);
DECL_ATOM(read_image);
DECL_ATOM(write_image);
DECL_ATOM(copy_image);
DECL_ATOM(copy_image_to_buffer);
DECL_ATOM(copy_buffer_to_image);
DECL_ATOM(map_buffer);
DECL_ATOM(map_image);
DECL_ATOM(unmap_mem_object);
DECL_ATOM(marker);
DECL_ATOM(aquire_gl_objects);
DECL_ATOM(release_gl_objects);

// execution_status
DECL_ATOM(complete);
DECL_ATOM(running);
DECL_ATOM(submitted);
DECL_ATOM(queued);

#define SIZE_1   0x010000
#define SIZE_2   0x020000
#define SIZE_4   0x040000
#define SIZE_8   0x080000
#define SIZE_16  0x100000

ecl_kv_t kv_cl_type[] = {
    { &ATOM(char),     SIZE_1 + OCL_CHAR },
    { &ATOM(char2),    SIZE_2 + OCL_CHAR },
    { &ATOM(char4),    SIZE_4 + OCL_CHAR },
    { &ATOM(char8),    SIZE_8 + OCL_CHAR },
    { &ATOM(char16),   SIZE_16 + OCL_CHAR },
    { &ATOM(uchar),    SIZE_1 + OCL_UCHAR },
    { &ATOM(uchar2),   SIZE_2 + OCL_UCHAR },
    { &ATOM(uchar4),   SIZE_4 + OCL_UCHAR },
    { &ATOM(uchar8),   SIZE_8 + OCL_UCHAR },
    { &ATOM(uchar16),  SIZE_16 + OCL_UCHAR },
    { &ATOM(short),    SIZE_1 + OCL_SHORT },
    { &ATOM(short2),   SIZE_2 + OCL_SHORT },
    { &ATOM(short4),   SIZE_4 + OCL_SHORT },
    { &ATOM(short8),   SIZE_8 + OCL_SHORT },
    { &ATOM(short16),  SIZE_16 + OCL_SHORT },
    { &ATOM(ushort),   SIZE_1 + OCL_USHORT },
    { &ATOM(ushort2),  SIZE_2 + OCL_USHORT },
    { &ATOM(ushort4),  SIZE_4 + OCL_USHORT },
    { &ATOM(ushort8),  SIZE_8 + OCL_USHORT },
    { &ATOM(ushort16), SIZE_16 + OCL_USHORT },
    { &ATOM(int),      SIZE_1 + OCL_INT },
    { &ATOM(int2),     SIZE_2 + OCL_INT },
    { &ATOM(int4),     SIZE_4 + OCL_INT },
    { &ATOM(int8),     SIZE_8 + OCL_INT },
    { &ATOM(int16),    SIZE_16 + OCL_INT },
    { &ATOM(uint),     SIZE_1 + OCL_UINT },
    { &ATOM(uint2),    SIZE_2 + OCL_UINT },
    { &ATOM(uint4),    SIZE_4 + OCL_UINT },
    { &ATOM(uint8),    SIZE_8 + OCL_UINT },
    { &ATOM(uint16),   SIZE_16 + OCL_UINT },
    { &ATOM(long),     SIZE_1 + OCL_LONG },
    { &ATOM(long2),    SIZE_2 + OCL_LONG },
    { &ATOM(long4),    SIZE_4 + OCL_LONG },
    { &ATOM(long8),    SIZE_8 + OCL_LONG },
    { &ATOM(long16),   SIZE_16 + OCL_LONG },
    { &ATOM(ulong),    SIZE_1 + OCL_ULONG },
    { &ATOM(ulong2),   SIZE_2 + OCL_ULONG },
    { &ATOM(ulong4),   SIZE_4 + OCL_ULONG },
    { &ATOM(ulong8),   SIZE_8 + OCL_ULONG },
    { &ATOM(ulong16),  SIZE_16 + OCL_ULONG },
    { &ATOM(half),     SIZE_1 + OCL_HALF },
    { &ATOM(float),    SIZE_1 + OCL_FLOAT },
    { &ATOM(float2),   SIZE_2 + OCL_FLOAT },
    { &ATOM(float4),   SIZE_4 + OCL_FLOAT },
    { &ATOM(float8),   SIZE_8 + OCL_FLOAT },
    { &ATOM(float16),  SIZE_16 + OCL_FLOAT },
    { &ATOM(double),   SIZE_1 + OCL_DOUBLE },
    { &ATOM(double2),  SIZE_2 + OCL_DOUBLE },
    { &ATOM(double4),  SIZE_4 + OCL_DOUBLE },
    { &ATOM(double8),  SIZE_8 + OCL_DOUBLE },
    { &ATOM(double16), SIZE_16 + OCL_DOUBLE },
    { 0, 0 }
};

ecl_kv_t kv_device_type[] = {  // bitfield
    { &ATOM(cpu),         CL_DEVICE_TYPE_CPU },
    { &ATOM(gpu),         CL_DEVICE_TYPE_GPU },
    { &ATOM(accelerator), CL_DEVICE_TYPE_ACCELERATOR },
    { &ATOM(default),     CL_DEVICE_TYPE_DEFAULT },
    { &ATOM(all),         CL_DEVICE_TYPE_ALL },
    { 0, 0}
};

ecl_kv_t kv_fp_config[] = {  // bitfield
    { &ATOM(denorm),      CL_FP_DENORM },
    { &ATOM(inf_nan),     CL_FP_INF_NAN },
    { &ATOM(round_to_nearest), CL_FP_ROUND_TO_NEAREST },
    { &ATOM(round_to_zero), CL_FP_ROUND_TO_ZERO },
    { &ATOM(round_to_inf), CL_FP_ROUND_TO_INF },
    { &ATOM(fma), CL_FP_FMA },
    { 0, 0 }
};

ecl_kv_t kv_mem_cache_type[] = {  // enum
    { &ATOM(none), CL_NONE },
    { &ATOM(read_only), CL_READ_ONLY_CACHE },
    { &ATOM(read_write), CL_READ_WRITE_CACHE },
    { 0, 0 }
};

ecl_kv_t kv_local_mem_type[] = {  // enum
    { &ATOM(local), CL_LOCAL },
    { &ATOM(global), CL_GLOBAL },
    { 0, 0 }
};

ecl_kv_t kv_exec_capabilities[] = {  // bit field
    { &ATOM(kernel), CL_EXEC_KERNEL },
    { &ATOM(native_kernel), CL_EXEC_NATIVE_KERNEL },
    { 0, 0 }
};


ecl_kv_t kv_command_queue_properties[] = { // bit field
    { &ATOM(out_of_order_exec_mode_enable), CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE },
    { &ATOM(profiling_enable), CL_QUEUE_PROFILING_ENABLE },
    { 0, 0}
};

ecl_kv_t kv_mem_flags[] = { // bit field
    { &ATOM(read_write), CL_MEM_READ_WRITE },
    { &ATOM(write_only), CL_MEM_WRITE_ONLY },
    { &ATOM(read_only),  CL_MEM_READ_ONLY },
    { &ATOM(use_host_ptr), CL_MEM_USE_HOST_PTR },
    { &ATOM(alloc_host_ptr), CL_MEM_ALLOC_HOST_PTR },
    { &ATOM(copy_host_ptr), CL_MEM_COPY_HOST_PTR },
    { 0, 0 }
};

ecl_kv_t kv_mem_object_type[] = { // enum
    { &ATOM(buffer), CL_MEM_OBJECT_BUFFER },
    { &ATOM(image2d), CL_MEM_OBJECT_IMAGE2D },
    { &ATOM(image3d), CL_MEM_OBJECT_IMAGE3D },
    { 0, 0 }
};

ecl_kv_t kv_addressing_mode[] = { // enum
    { &ATOM(none), CL_ADDRESS_NONE },
    { &ATOM(clamp_to_edge), CL_ADDRESS_CLAMP_TO_EDGE },
    { &ATOM(clamp), CL_ADDRESS_CLAMP },
    { &ATOM(repeat), CL_ADDRESS_REPEAT },
    { 0, 0 }
};

ecl_kv_t kv_filter_mode[] = { // enum
    { &ATOM(nearest), CL_FILTER_NEAREST },
    { &ATOM(linear),  CL_FILTER_LINEAR },
    { 0, 0 }
};

ecl_kv_t kv_map_flags[] = { // bitfield
    { &ATOM(read), CL_MAP_READ },
    { &ATOM(write), CL_MAP_WRITE },
    { 0, 0 }
};

ecl_kv_t kv_build_status[] = { // enum
    { &ATOM(success), CL_BUILD_SUCCESS },
    { &ATOM(none), CL_BUILD_NONE },
    { &ATOM(error), CL_BUILD_ERROR },
    { &ATOM(in_progress), CL_BUILD_IN_PROGRESS },
    { 0, 0 }
};

ecl_kv_t kv_command_type[] = { // enum
    { &ATOM(ndrange_kernel), CL_COMMAND_NDRANGE_KERNEL },
    { &ATOM(task),           CL_COMMAND_TASK },
    { &ATOM(native_kernel),  CL_COMMAND_NATIVE_KERNEL },
    { &ATOM(read_buffer),    CL_COMMAND_READ_BUFFER },
    { &ATOM(write_buffer),   CL_COMMAND_WRITE_BUFFER },
    { &ATOM(copy_buffer),    CL_COMMAND_COPY_BUFFER },
    { &ATOM(read_image),     CL_COMMAND_READ_IMAGE },
    { &ATOM(write_image),    CL_COMMAND_WRITE_IMAGE },
    { &ATOM(copy_image),     CL_COMMAND_COPY_IMAGE },
    { &ATOM(copy_image_to_buffer), CL_COMMAND_COPY_IMAGE_TO_BUFFER },
    { &ATOM(copy_buffer_to_image), CL_COMMAND_COPY_BUFFER_TO_IMAGE },
    { &ATOM(map_buffer), CL_COMMAND_MAP_BUFFER },
    { &ATOM(map_image), CL_COMMAND_MAP_IMAGE },
    { &ATOM(unmap_mem_object), CL_COMMAND_UNMAP_MEM_OBJECT },
    { &ATOM(marker), CL_COMMAND_MARKER  },
    { &ATOM(aquire_gl_objects), CL_COMMAND_ACQUIRE_GL_OBJECTS },
    { &ATOM(release_gl_objects), CL_COMMAND_RELEASE_GL_OBJECTS },
    { 0, 0}
};

ecl_kv_t kv_execution_status[] = { // enum
    { &ATOM(complete),   CL_COMPLETE   },   // same as CL_SUCCESS
    { &ATOM(running),    CL_RUNNING    },
    { &ATOM(submitted),  CL_SUBMITTED  },
    { &ATOM(queued),     CL_QUEUED     },
    // the error codes (negative values)
    { &ATOM(device_not_found), CL_DEVICE_NOT_FOUND },
    { &ATOM(device_not_available), CL_DEVICE_NOT_AVAILABLE },
    { &ATOM(compiler_not_available), CL_COMPILER_NOT_AVAILABLE },
    { &ATOM(mem_object_allocation_failure), CL_MEM_OBJECT_ALLOCATION_FAILURE },
    { &ATOM(out_of_resources), CL_OUT_OF_RESOURCES },
    { &ATOM(out_of_host_memory), CL_OUT_OF_HOST_MEMORY },
    { &ATOM(profiling_info_not_available), CL_PROFILING_INFO_NOT_AVAILABLE },
    { &ATOM(mem_copy_overlap), CL_MEM_COPY_OVERLAP },
    { &ATOM(image_format_mismatch), CL_IMAGE_FORMAT_MISMATCH },
    { &ATOM(image_format_not_supported), CL_IMAGE_FORMAT_NOT_SUPPORTED },
    { &ATOM(build_program_failure), CL_BUILD_PROGRAM_FAILURE },
    { &ATOM(map_failure), CL_MAP_FAILURE },
    { &ATOM(invalid_value), CL_INVALID_VALUE },
    { &ATOM(invalid_device_type), CL_INVALID_DEVICE_TYPE },
    { &ATOM(invalid_platform), CL_INVALID_PLATFORM },
    { &ATOM(invalid_device), CL_INVALID_DEVICE },
    { &ATOM(invalid_context), CL_INVALID_CONTEXT },
    { &ATOM(invalid_queue_properties), CL_INVALID_QUEUE_PROPERTIES },
    { &ATOM(invalid_command_queue), CL_INVALID_COMMAND_QUEUE },
    { &ATOM(invalid_host_ptr), CL_INVALID_HOST_PTR },
    { &ATOM(invalid_mem_object), CL_INVALID_MEM_OBJECT },
    { &ATOM(invalid_image_format_descriptor), CL_INVALID_IMAGE_FORMAT_DESCRIPTOR },
    { &ATOM(invalid_image_size), CL_INVALID_IMAGE_SIZE },
    { &ATOM(invalid_sampler), CL_INVALID_SAMPLER },
    { &ATOM(invalid_binary), CL_INVALID_BINARY },
    { &ATOM(invalid_build_options), CL_INVALID_BUILD_OPTIONS },
    { &ATOM(invalid_program), CL_INVALID_PROGRAM },
    { &ATOM(invalid_program_executable), CL_INVALID_PROGRAM_EXECUTABLE },
    { &ATOM(invalid_kernel_name), CL_INVALID_KERNEL_NAME },
    { &ATOM(invalid_kernel_definition), CL_INVALID_KERNEL_DEFINITION },
    { &ATOM(invalid_kernel), CL_INVALID_KERNEL },
    { &ATOM(invalid_arg_index), CL_INVALID_ARG_INDEX },
    { &ATOM(invalid_arg_value), CL_INVALID_ARG_VALUE },
    { &ATOM(invalid_arg_size), CL_INVALID_ARG_SIZE },
    { &ATOM(invalid_kernel_args), CL_INVALID_KERNEL_ARGS },
    { &ATOM(invalid_work_dimension), CL_INVALID_WORK_DIMENSION },
    { &ATOM(invalid_work_group_size), CL_INVALID_WORK_GROUP_SIZE },
    { &ATOM(invalid_work_item_size), CL_INVALID_WORK_ITEM_SIZE },
    { &ATOM(invalid_global_offset), CL_INVALID_GLOBAL_OFFSET },
    { &ATOM(invalid_event_wait_list), CL_INVALID_EVENT_WAIT_LIST },
    { &ATOM(invalid_event), CL_INVALID_EVENT },
    { &ATOM(invalid_operation), CL_INVALID_OPERATION },
    { &ATOM(invalid_gl_object), CL_INVALID_GL_OBJECT },
    { &ATOM(invalid_buffer_size), CL_INVALID_BUFFER_SIZE },
    { &ATOM(invalid_mip_level), CL_INVALID_MIP_LEVEL },
    { 0, 0 }
};

DECL_ATOM(snorm_int8);
DECL_ATOM(snorm_int16);
DECL_ATOM(unorm_int8);
DECL_ATOM(unorm_int16);
DECL_ATOM(unorm_short_565);
DECL_ATOM(unorm_short_555);
DECL_ATOM(unorm_int_101010);
DECL_ATOM(signed_int8);
DECL_ATOM(signed_int16);
DECL_ATOM(signed_int32);
DECL_ATOM(unsigned_int8);
DECL_ATOM(unsigned_int16);
DECL_ATOM(unsigned_int32);
DECL_ATOM(half_float);
// DECL_ATOM(float);

ecl_kv_t kv_channel_type[] = { // enum
    { &ATOM(snorm_int8), CL_SNORM_INT8 },
    { &ATOM(snorm_int16), CL_SNORM_INT16 },
    { &ATOM(unorm_int8), CL_UNORM_INT8 },
    { &ATOM(unorm_int16), CL_UNORM_INT16 },
    { &ATOM(unorm_short_565), CL_UNORM_SHORT_565 },
    { &ATOM(unorm_short_555), CL_UNORM_SHORT_555 },
    { &ATOM(unorm_int_101010), CL_UNORM_INT_101010 },
    { &ATOM(signed_int8), CL_SIGNED_INT8 },
    { &ATOM(signed_int16), CL_SIGNED_INT16 },
    { &ATOM(signed_int32), CL_SIGNED_INT32 },
    { &ATOM(unsigned_int8), CL_UNSIGNED_INT8 },
    { &ATOM(unsigned_int16), CL_UNSIGNED_INT16 },
    { &ATOM(unsigned_int32), CL_UNSIGNED_INT32 },
    { &ATOM(half_float), CL_HALF_FLOAT },
    { &ATOM(float), CL_FLOAT },
    { 0, 0 }
};

DECL_ATOM(r);
DECL_ATOM(a);
DECL_ATOM(rg);
DECL_ATOM(ra);
DECL_ATOM(rgb);
DECL_ATOM(rgba);
DECL_ATOM(bgra);
DECL_ATOM(argb);
DECL_ATOM(intensity);
DECL_ATOM(luminance);
DECL_ATOM(rx);
DECL_ATOM(rgx);
DECL_ATOM(rgbx);

// 1.1 features! in apple 1.0?
#ifndef CL_Rx
#define CL_Rx                                       0x10BA
#endif 

#ifndef CL_RGx
#define CL_RGx                                      0x10BB
#endif

#ifndef CL_RGBx
#define CL_RGBx                                     0x10BC
#endif

ecl_kv_t kv_channel_order[] = {
    { &ATOM(r), CL_R },
    { &ATOM(a), CL_A },
    { &ATOM(rg), CL_RG },
    { &ATOM(ra), CL_RA },
    { &ATOM(rgb), CL_RGB },
    { &ATOM(rgba), CL_RGBA },
    { &ATOM(bgra), CL_BGRA },
    { &ATOM(argb), CL_ARGB },
    { &ATOM(intensity), CL_INTENSITY },
    { &ATOM(luminance), CL_LUMINANCE },
    { &ATOM(rx), CL_Rx },
    { &ATOM(rgx), CL_RGx },
    { &ATOM(rgbx), CL_RGBx },
    { 0, 0 }
};

// Map device info index 0...N => cl_device_info x Data type
ecl_info_t device_info[] = 
{
    { &ATOM(type), CL_DEVICE_TYPE, false, OCL_DEVICE_TYPE, kv_device_type },
    { &ATOM(vendor_id), CL_DEVICE_VENDOR_ID, false, OCL_UINT, 0 },
    { &ATOM(max_compute_units), CL_DEVICE_MAX_COMPUTE_UNITS, false, OCL_UINT, 0 },
    { &ATOM(max_work_item_dimensions), CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, false, OCL_UINT, 0 },
    { &ATOM(max_work_group_size), CL_DEVICE_MAX_WORK_GROUP_SIZE, false, OCL_SIZE, 0 },
    { &ATOM(max_work_item_sizes), CL_DEVICE_MAX_WORK_ITEM_SIZES, true, OCL_SIZE, 0 },
    { &ATOM(preferred_vector_width_char), CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, false, OCL_UINT, 0 },
    { &ATOM(preferred_vector_width_short), CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, false, OCL_UINT,  0 },
    { &ATOM(preferred_vector_width_int), CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, false, OCL_UINT, 0 },
    { &ATOM(preferred_vector_width_long), CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, false,OCL_UINT, 0 },
    { &ATOM(preferred_vector_width_float), CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, false, OCL_UINT, 0 },
    { &ATOM(preferred_vector_width_double), CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, false, OCL_UINT, 0 },
    { &ATOM(max_clock_frequency), CL_DEVICE_MAX_CLOCK_FREQUENCY, false, OCL_UINT, 0 },
    { &ATOM(address_bits), CL_DEVICE_ADDRESS_BITS, false, OCL_UINT, 0 },
    { &ATOM(max_read_image_args), CL_DEVICE_MAX_READ_IMAGE_ARGS, false, OCL_UINT, 0 },
    { &ATOM(max_write_image_args), CL_DEVICE_MAX_WRITE_IMAGE_ARGS, false, OCL_UINT, 0 },
    { &ATOM(max_mem_alloc_size), CL_DEVICE_MAX_MEM_ALLOC_SIZE, false, OCL_ULONG, 0 },
    { &ATOM(image2d_max_width), CL_DEVICE_IMAGE2D_MAX_WIDTH, false, OCL_SIZE, 0 },
    { &ATOM(image2d_max_height), CL_DEVICE_IMAGE2D_MAX_HEIGHT, false, OCL_SIZE, 0 },
    { &ATOM(image3d_max_width), CL_DEVICE_IMAGE3D_MAX_WIDTH, false, OCL_SIZE, 0 },
    { &ATOM(image3d_max_height), CL_DEVICE_IMAGE3D_MAX_HEIGHT, false, OCL_SIZE, 0 },
    { &ATOM(image3d_max_depth), CL_DEVICE_IMAGE3D_MAX_DEPTH, false, OCL_SIZE, 0 },
    { &ATOM(image_support), CL_DEVICE_IMAGE_SUPPORT, false, OCL_BOOL, 0 },
    { &ATOM(max_parameter_size), CL_DEVICE_MAX_PARAMETER_SIZE, false, OCL_SIZE, 0 },
    { &ATOM(max_samplers), CL_DEVICE_MAX_SAMPLERS, false, OCL_UINT, 0 },
     { &ATOM(mem_base_addr_align), CL_DEVICE_MEM_BASE_ADDR_ALIGN, false, OCL_UINT, 0 },
    { &ATOM(min_data_type_align_size), CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE, false, OCL_UINT, 0 },
    { &ATOM(single_fp_config), CL_DEVICE_SINGLE_FP_CONFIG, false, OCL_DEVICE_FP_CONFIG, kv_fp_config },
    { &ATOM(global_mem_cache_type), CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, false, OCL_DEVICE_GLOBAL_MEM_CACHE_TYPE, kv_mem_cache_type },
    { &ATOM(global_mem_cacheline_size), CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, false, OCL_UINT, 0 },
    { &ATOM(global_mem_cache_size), CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, false, OCL_ULONG, 0 },
    { &ATOM(global_mem_size), CL_DEVICE_GLOBAL_MEM_SIZE, false, OCL_ULONG, 0 },
    { &ATOM(max_constant_buffer_size), CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,  false, OCL_ULONG, 0 },
    { &ATOM(max_constant_args), CL_DEVICE_MAX_CONSTANT_ARGS, false, OCL_UINT, 0 },
    { &ATOM(local_mem_type), CL_DEVICE_LOCAL_MEM_TYPE, false, OCL_DEVICE_LOCAL_MEM_TYPE, kv_local_mem_type },
    { &ATOM(local_mem_size), CL_DEVICE_LOCAL_MEM_SIZE,  false, OCL_ULONG, 0 },
    { &ATOM(error_correction_support), CL_DEVICE_ERROR_CORRECTION_SUPPORT, false,  OCL_BOOL, 0 },
    { &ATOM(profiling_timer_resolution), CL_DEVICE_PROFILING_TIMER_RESOLUTION, false,  OCL_SIZE, 0 },
    { &ATOM(endian_little), CL_DEVICE_ENDIAN_LITTLE, false, OCL_BOOL, 0},
    { &ATOM(available), CL_DEVICE_AVAILABLE,  false, OCL_BOOL, 0 },
    { &ATOM(compiler_available), CL_DEVICE_COMPILER_AVAILABLE, false, OCL_BOOL, 0 },
    { &ATOM(execution_capabilities), CL_DEVICE_EXECUTION_CAPABILITIES, false, OCL_DEVICE_EXEC_CAPABILITIES, kv_exec_capabilities },
    { &ATOM(queue_properties), CL_DEVICE_QUEUE_PROPERTIES, false, OCL_QUEUE_PROPERTIES, kv_command_queue_properties },
    { &ATOM(name), CL_DEVICE_NAME, false, OCL_STRING, 0 }, 
    { &ATOM(vendor), CL_DEVICE_VENDOR, false, OCL_STRING, 0 }, 
    { &ATOM(driver_version), CL_DRIVER_VERSION, false, OCL_STRING, 0 },
    { &ATOM(profile), CL_DEVICE_PROFILE, false, OCL_STRING, 0 },
    { &ATOM(version), CL_DEVICE_VERSION, false, OCL_STRING, 0 },
    { &ATOM(extensions), CL_DEVICE_EXTENSIONS, false, OCL_STRING, 0 },
    { &ATOM(platform), CL_DEVICE_PLATFORM, false, OCL_PLATFORM, 0 }
};

// Map device info index 0...N => cl_device_info x Data type
ecl_info_t platform_info[] = 
{
    { &ATOM(profile), CL_PLATFORM_PROFILE, false, OCL_STRING, 0 },
    { &ATOM(version), CL_PLATFORM_VERSION, false, OCL_STRING, 0 },
    { &ATOM(name),    CL_PLATFORM_NAME,    false, OCL_STRING, 0 },
    { &ATOM(vendor),  CL_PLATFORM_VENDOR,  false, OCL_STRING, 0 },
    { &ATOM(extensions), CL_PLATFORM_EXTENSIONS, false, OCL_STRING, 0}
};

ecl_info_t context_info[] =
{
    { &ATOM(reference_count), CL_CONTEXT_REFERENCE_COUNT, false, OCL_UINT, 0 },
    { &ATOM(devices), CL_CONTEXT_DEVICES, true, OCL_DEVICE, 0 },
    { &ATOM(properties), CL_CONTEXT_PROPERTIES, true, OCL_INT, 0 }
};

ecl_info_t queue_info[] = 
{
    { &ATOM(context), CL_QUEUE_CONTEXT, false, OCL_CONTEXT, 0 },
    { &ATOM(device),  CL_QUEUE_DEVICE, false, OCL_DEVICE, 0 },
    { &ATOM(reference_count), CL_QUEUE_REFERENCE_COUNT, false, OCL_UINT, 0 },
    { &ATOM(properties), CL_QUEUE_PROPERTIES, false, OCL_QUEUE_PROPERTIES, kv_command_queue_properties }
};

ecl_info_t mem_info[] =
{
    { &ATOM(object_type), CL_MEM_TYPE, false, OCL_MEM_OBJECT_TYPE, kv_mem_object_type },
    { &ATOM(flags), CL_MEM_FLAGS, false, OCL_MEM_FLAGS, kv_mem_flags },
    { &ATOM(size),  CL_MEM_SIZE,  false, OCL_SIZE, 0 },
    // FIXME: pointer!! map it (binary resource?)
    { &ATOM(host_ptr), CL_MEM_HOST_PTR, false, OCL_POINTER, 0 }, 
    { &ATOM(map_count), CL_MEM_MAP_COUNT, false, OCL_UINT, 0 },
    { &ATOM(reference_count), CL_MEM_REFERENCE_COUNT, false, OCL_UINT, 0 },
    { &ATOM(context), CL_MEM_CONTEXT, false, OCL_CONTEXT, 0 }
};

ecl_info_t image_info[] =
{
    { &ATOM(format), CL_IMAGE_FORMAT, false, OCL_IMAGE_FORMAT, 0 },
    { &ATOM(element_size), CL_IMAGE_ELEMENT_SIZE, false, OCL_SIZE, 0 },
    { &ATOM(row_pitch),  CL_IMAGE_ROW_PITCH,  false, OCL_SIZE, 0 },
    { &ATOM(slice_pitch), CL_IMAGE_SLICE_PITCH, false, OCL_SIZE, 0 },
    { &ATOM(width), CL_IMAGE_WIDTH, false, OCL_SIZE, 0 },
    { &ATOM(height), CL_IMAGE_HEIGHT, false, OCL_SIZE, 0 },
    { &ATOM(depth), CL_IMAGE_DEPTH, false, OCL_SIZE, 0 }
};

ecl_info_t sampler_info[] = 
{
    { &ATOM(reference_count), CL_SAMPLER_REFERENCE_COUNT, false, OCL_UINT, 0},
    { &ATOM(context), CL_SAMPLER_CONTEXT, false,  OCL_CONTEXT, 0 },
    { &ATOM(normalized_coords), CL_SAMPLER_NORMALIZED_COORDS, false, OCL_BOOL, 0 },
    {  &ATOM(addressing_mode), CL_SAMPLER_ADDRESSING_MODE, false, OCL_SAMPLER_ADDRESSING_MODE, kv_addressing_mode },
    { &ATOM(filter_mode), CL_SAMPLER_FILTER_MODE, false, OCL_SAMPLER_FILTER_MODE, kv_filter_mode }
};

ecl_info_t program_info[] = {
    { &ATOM(reference_count), CL_PROGRAM_REFERENCE_COUNT, false, OCL_UINT, 0 },
    { &ATOM(context), CL_PROGRAM_CONTEXT, false, OCL_CONTEXT, 0},
    { &ATOM(num_devices), CL_PROGRAM_NUM_DEVICES, false, OCL_UINT, 0},
    { &ATOM(devices), CL_PROGRAM_DEVICES, true, OCL_DEVICE, 0 },
    { &ATOM(source), CL_PROGRAM_SOURCE, false, OCL_STRING, 0 },
    { &ATOM(binary_sizes), CL_PROGRAM_BINARY_SIZES, true, OCL_SIZE, 0 },
    { &ATOM(binaries), CL_PROGRAM_BINARIES, true, OCL_STRING, 0 }
};

ecl_info_t build_info[] = {
    { &ATOM(status), CL_PROGRAM_BUILD_STATUS, false, OCL_BUILD_STATUS, kv_build_status },
    { &ATOM(options), CL_PROGRAM_BUILD_OPTIONS, false, OCL_STRING, 0 },
    { &ATOM(log), CL_PROGRAM_BUILD_LOG, false, OCL_STRING, 0 }
};

ecl_info_t kernel_info[] = {
    { &ATOM(function_name), CL_KERNEL_FUNCTION_NAME, false, OCL_STRING, 0 },
    { &ATOM(num_args), CL_KERNEL_NUM_ARGS, false, OCL_UINT, 0},
    { &ATOM(reference_count), CL_KERNEL_REFERENCE_COUNT, false, OCL_UINT, 0 },
    { &ATOM(context), CL_KERNEL_CONTEXT, false, OCL_CONTEXT, 0},
    { &ATOM(program), CL_KERNEL_PROGRAM, false, OCL_PROGRAM, 0}
};

ecl_info_t workgroup_info[] = {
    { &ATOM(work_group_size), CL_KERNEL_WORK_GROUP_SIZE, false, OCL_SIZE, 0 },
    { &ATOM(compile_work_group_size), CL_KERNEL_COMPILE_WORK_GROUP_SIZE, true, OCL_SIZE, 0},
    { &ATOM(local_mem_size), CL_KERNEL_LOCAL_MEM_SIZE, false, OCL_ULONG, 0 },
};

ecl_info_t event_info[] = {
    { &ATOM(command_queue),  CL_EVENT_COMMAND_QUEUE, false, OCL_COMMAND_QUEUE, 0},
    { &ATOM(command_type),   CL_EVENT_COMMAND_TYPE, false,  OCL_ENUM, kv_command_type },
    { &ATOM(reference_count), CL_EVENT_REFERENCE_COUNT, false, OCL_UINT, 0 },
    { &ATOM(execution_status), CL_EVENT_COMMAND_EXECUTION_STATUS, false, OCL_ENUM, kv_execution_status }
};


// Error reasons
ERL_NIF_TERM ecl_error(cl_int err)
{
    switch(err) {
    case CL_DEVICE_NOT_FOUND: 
	return ATOM(device_not_found);
    case CL_DEVICE_NOT_AVAILABLE: 
	return ATOM(device_not_available);
    case CL_COMPILER_NOT_AVAILABLE: 
	return ATOM(compiler_not_available);
    case CL_MEM_OBJECT_ALLOCATION_FAILURE: 
	return ATOM(mem_object_allocation_failure);
    case CL_OUT_OF_RESOURCES: 
	return ATOM(out_of_resources);
    case CL_OUT_OF_HOST_MEMORY: 
	return ATOM(out_of_host_memory);
    case CL_PROFILING_INFO_NOT_AVAILABLE: 
	return ATOM(profiling_info_not_available);
    case CL_MEM_COPY_OVERLAP: 
	return ATOM(mem_copy_overlap);
    case CL_IMAGE_FORMAT_MISMATCH:
	return ATOM(image_format_mismatch);
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:
	return ATOM(image_format_not_supported);
    case CL_BUILD_PROGRAM_FAILURE: 
	return ATOM(build_program_failure);
    case CL_MAP_FAILURE: 
	return ATOM(map_failure);
    case CL_INVALID_VALUE: 
	return ATOM(invalid_value);
    case CL_INVALID_DEVICE_TYPE: 
	return ATOM(invalid_device_type);
    case CL_INVALID_PLATFORM: 
	return ATOM(invalid_platform);
    case CL_INVALID_DEVICE: 
	return ATOM(invalid_device);
    case CL_INVALID_CONTEXT: 
	return ATOM(invalid_context);
    case CL_INVALID_QUEUE_PROPERTIES: 
	return ATOM(invalid_queue_properties);
    case CL_INVALID_COMMAND_QUEUE: 
	return ATOM(invalid_command_queue);
    case CL_INVALID_HOST_PTR: 
	return ATOM(invalid_host_ptr);
    case CL_INVALID_MEM_OBJECT: 
	return ATOM(invalid_mem_object);
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: 
	return ATOM(invalid_image_format_descriptor);
    case CL_INVALID_IMAGE_SIZE: 
	return ATOM(invalid_image_size);
    case CL_INVALID_SAMPLER: 
	return ATOM(invalid_sampler);
    case CL_INVALID_BINARY: 
	return ATOM(invalid_binary);
    case CL_INVALID_BUILD_OPTIONS: 
	return ATOM(invalid_build_options);
    case CL_INVALID_PROGRAM: 
	return ATOM(invalid_program);
    case CL_INVALID_PROGRAM_EXECUTABLE: 
	return ATOM(invalid_program_executable);
    case CL_INVALID_KERNEL_NAME: 
	return ATOM(invalid_kernel_name);
    case CL_INVALID_KERNEL_DEFINITION: 
	return ATOM(invalid_kernel_definition);
    case CL_INVALID_KERNEL: 
	return ATOM(invalid_kernel);
    case CL_INVALID_ARG_INDEX: 
	return ATOM(invalid_arg_index);
    case CL_INVALID_ARG_VALUE: 
	return ATOM(invalid_arg_value);
    case CL_INVALID_ARG_SIZE: 
	return ATOM(invalid_arg_size);
    case CL_INVALID_KERNEL_ARGS: 
	return ATOM(invalid_kernel_args);
    case CL_INVALID_WORK_DIMENSION: 
	return ATOM(invalid_work_dimension);
    case CL_INVALID_WORK_GROUP_SIZE: 
	return ATOM(invalid_work_group_size);
    case CL_INVALID_WORK_ITEM_SIZE: 
	return ATOM(invalid_work_item_size);
    case CL_INVALID_GLOBAL_OFFSET: 
	return ATOM(invalid_global_offset);
    case CL_INVALID_EVENT_WAIT_LIST: 
	return ATOM(invalid_event_wait_list);
    case CL_INVALID_EVENT: 
	return ATOM(invalid_event);
    case CL_INVALID_OPERATION: 
	return ATOM(invalid_operation);
    case CL_INVALID_GL_OBJECT: 
	return ATOM(invalid_gl_object);
    case CL_INVALID_BUFFER_SIZE: 
	return ATOM(invalid_buffer_size);
    case CL_INVALID_MIP_LEVEL: 
	return ATOM(invalid_mip_level);
    default: 
	return ATOM(unknown);
    }
}

ERL_NIF_TERM ecl_make_error(ErlNifEnv* env, cl_int err)
{
    return enif_make_tuple2(env, ATOM(error), ecl_error(err));
}

static void ecl_emit_error(char* file, int line, ...)
{
    va_list ap;
    char* fmt;

    va_start(ap, line);
    fmt = va_arg(ap, char*);

    fprintf(stderr, "%s:%d: ", file, line); 
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\r\n");
    va_end(ap);
    fflush(stderr);
}

// Parse bool
static int get_bool(ErlNifEnv* env, const ERL_NIF_TERM key, cl_bool* val)
{
    UNUSED(env);
    if (key == ATOM(true)) {
	*val = true;
	return 1;
    }
    else if (key == ATOM(false)) {
	*val = false;
	return 1;
    }
    return 0;
}


// Parse enum
static int get_enum(ErlNifEnv* env, const ERL_NIF_TERM key,
		    cl_uint* num, ecl_kv_t* kv)
{
    UNUSED(env);

    if (!enif_is_atom(env, key))
	return 0;
    while(kv->key) {
	if (*kv->key == key) {
	    *num = kv->value;
	    return 1;
	}
	kv++;
    }
    return 0;
}

// Parse bitfield
static int get_bitfield(ErlNifEnv* env, const ERL_NIF_TERM key,
			cl_bitfield* field, ecl_kv_t* kv)
{
    UNUSED(env);

    if (!enif_is_atom(env, key))
	return 0;
    while(kv->key) {
	if (*kv->key == key) {
	    *field = kv->value;
	    return 1;
	}
	kv++;
    }
    return 0;
}


static int get_bitfields(ErlNifEnv* env, const ERL_NIF_TERM term,
			 cl_bitfield* field, ecl_kv_t* kv)
{
    cl_bitfield t;

    if (enif_is_atom(env, term)) {
	if (!get_bitfield(env, term, &t, kv))
	    return 0;
	*field = t;
	return 1;
    }
    else if (enif_is_empty_list(env, term)) {
	*field = 0;
	return 1;
    }
    else if (enif_is_list(env, term)) {
	cl_bitfield fs = 0;
	ERL_NIF_TERM list = term;
	ERL_NIF_TERM head, tail;
	
	while(enif_get_list_cell(env, list, &head, &tail)) {
	    if (!get_bitfield(env, head, &t, kv))
		return 0;
	    fs |= t;
	    list = tail;
	}
	if (!enif_is_empty_list(env, list))
	    return 0;
	*field = fs;
	return 1;
    }
    return 0;
}

ERL_NIF_TERM make_enum(ErlNifEnv* env, cl_uint num, ecl_kv_t* kv)
{
    while(kv->key) {
	if (num == (cl_uint)kv->value)
	    return *kv->key;
	kv++;
    }
    return enif_make_uint(env, num);
}

ERL_NIF_TERM make_bitfields(ErlNifEnv* env, cl_bitfield v, ecl_kv_t* kv)
{
    ERL_NIF_TERM list = enif_make_list(env, 0);

    if (v) {
	int n = 0;
	while(kv->key) {
	    kv++;
	    n++;
	}
	while(n--) {
	    kv--;
	    if ((kv->value & v) == kv->value)
		list = enif_make_list_cell(env, *kv->key, list);
	}
    }
    return list;
}



/******************************************************************************
 *
 *   Linear hash functions
 *
 *****************************************************************************/

#define EPTR_HANDLE(ptr) ((intptr_t)(ptr))

static lhash_value_t ref_hash(void* key)
{
    return (lhash_value_t) key;
}

static int ref_cmp(void* key, void* data)
{
    if (((intptr_t)key) == EPTR_HANDLE(((ecl_object_t*)data)->opaque))
	return 0;
    return 1;
}

static void ref_release(void *data)
{
    UNUSED(data);
    // object's are free'd by garbage collection
}

// Remove object from hash 
static void object_erase(ecl_object_t* obj)
{
    ecl_env_t* ecl = obj->env;
    enif_rwlock_rwlock(ecl->ref_lock);
    lhash_erase(&ecl->ref, (void*)EPTR_HANDLE(obj->opaque));
    enif_rwlock_rwunlock(ecl->ref_lock);
}

/******************************************************************************
 *
 *   Message queue
 *
 *****************************************************************************/

// Peek at queue front
#if 0
static ecl_message_t* ecl_queue_peek(ecl_queue_t* q)
{
    ecl_qlink_t* ql;

    enif_mutex_lock(q->mtx);
    ql = q->front;
    enif_mutex_unlock(q->mtx);
    if (ql)
	return &ql->mesg;
    else
	return 0;
}
#endif

// Get message from queue front
static int ecl_queue_get(ecl_queue_t* q, ecl_message_t* m)
{
    ecl_qlink_t* ql;

    enif_mutex_lock(q->mtx);
    while(!(ql = q->front)) {
	enif_cond_wait(q->cv, q->mtx);
    }
    if (!(q->front = ql->next))
	q->rear = 0;
    q->len--;

    *m = ql->mesg;

    if ((ql >= &q->ql[0]) && (ql <= &q->ql[MAX_QLINK-1])) {
	ql->next = q->free;
	q->free = ql;
    }
    else 
	enif_free(ql);
    enif_mutex_unlock(q->mtx);
    return 0;
}

// Put message at queue rear
static int ecl_queue_put(ecl_queue_t* q, ecl_message_t* m)
{
    ecl_qlink_t* ql;
    ecl_qlink_t* qr;
    int res = 0;

    enif_mutex_lock(q->mtx);

    if ((ql = q->free))
	q->free = ql->next;
    else
	ql = enif_alloc(sizeof(ecl_qlink_t));
    if (!ql)
	res = -1;
    else {
	ql->mesg = *m;
	q->len++;
	ql->next = 0;
	if (!(qr = q->rear)) {
	    q->front = ql;
	    enif_cond_signal(q->cv);
	}
	else
	    qr->next = ql;
	q->rear = ql;
    }
    enif_mutex_unlock(q->mtx);
    return res;
}

static int ecl_queue_init(ecl_queue_t* q)
{
    int i;
    if (!(q->cv     = enif_cond_create("queue_cv")))
	return -1;
    if (!(q->mtx    = enif_mutex_create("queue_mtx")))
	return -1;
    q->front  = 0;
    q->rear   = 0;
    q->len    = 0;
    for (i = 0; i < MAX_QLINK-1; i++)
	q->ql[i].next = &q->ql[i+1];
    q->ql[MAX_QLINK-1].next = 0;
    q->free = &q->ql[0];
    return 0;
}

static void ecl_queue_destroy(ecl_queue_t* q)
{
    ecl_qlink_t* ql;

    enif_cond_destroy(q->cv);
    enif_mutex_destroy(q->mtx);

    ql = q->front;
    while(ql) {
	ecl_qlink_t* qln = ql->next;
	if ((ql >= &q->ql[0]) && (ql <= &q->ql[MAX_QLINK-1]))
	    ;
	else
	    enif_free(ql);
	ql = qln;
    }
}

/******************************************************************************
 *
 *   Threads
 *
 *****************************************************************************/

static int ecl_message_send(ecl_thread_t* thr, ecl_message_t* m)
{
    return ecl_queue_put(&thr->q, m);
}

static int ecl_message_recv(ecl_thread_t* thr, ecl_message_t* m)
{
    int r;
    if ((r = ecl_queue_get(&thr->q, m)) < 0)
	return r;
    return 0;
}

#if 0
static ecl_message_t* ecl_message_peek(ecl_thread_t* thr, ecl_thread_t** from)
{
    ecl_message_t* m;
    if ((m = ecl_queue_peek(&thr->q))) {
	if (from)
	    *from = m->sender;
    }
    return m;
}
#endif

static ecl_thread_t* ecl_thread_start(void* (*func)(void* arg),
				      void* arg, int stack_size)
{
    ErlNifThreadOpts* opts;
    ecl_thread_t* thr;

    if (!(thr = enif_alloc(sizeof(ecl_thread_t))))
	return 0;
    if (ecl_queue_init(&thr->q) < 0)
	goto error;
    if (!(opts = enif_thread_opts_create("ecl_thread_opts")))
	goto error;
    opts->suggested_stack_size = stack_size;
    thr->arg = arg;

    enif_thread_create("ecl_thread", &thr->tid, func, thr, opts);
    enif_thread_opts_destroy(opts);
    return thr;
error:
    enif_free(thr);
    return 0;
}

static int ecl_thread_stop(ecl_thread_t* thr, void** exit_value)
{
    ecl_message_t m;
    int r;

    m.type   = ECL_MESSAGE_STOP;
    m.env    = 0;
    ecl_message_send(thr, &m);
    r=enif_thread_join(thr->tid, exit_value);
    ecl_queue_destroy(&thr->q);
    enif_free(thr);
    return 0;
}

static void ecl_thread_exit(void* value)
{
    enif_thread_exit(value);
}

/******************************************************************************
 *
 *   Ecl resource
 *
 *****************************************************************************/

static int ecl_resource_init(ErlNifEnv* env,
			     ecl_resource_t* res,
			     char* name,
			     size_t size,  // object size
			     void (*dtor)(ErlNifEnv*, ecl_object_t*),
			     ErlNifResourceFlags flags,
			     ErlNifResourceFlags* tried)
{
    res->name = name;
    res->type = enif_make_atom(env, name);
    res->size = size;
    res->res  = enif_open_resource_type(env, 0, name, 
					(ErlNifResourceDtor*) dtor,
					flags, tried);
    return 0;
}

//
// Reference new kernel argument and Dereference old value
//

static void unref_kernel_arg(int type, void* val)
{
    switch(type) {
    case KERNEL_ARG_MEM:
	if (val)
	    clReleaseMemObject((cl_mem) val);
	break;
    case KERNEL_ARG_SAMPLER:
	if (val)
	    clReleaseSampler((cl_sampler) val);
	break;
    case KERNEL_ARG_OTHER:
    default:
	break;
    }
}

static void ref_kernel_arg(int type, void* val)
{
    switch(type) {
    case KERNEL_ARG_MEM:
	if (val)
	    clRetainMemObject((cl_mem) val);
	break;
    case KERNEL_ARG_SAMPLER:
	if (val)
	    clRetainSampler((cl_sampler) val);
	break;
    case KERNEL_ARG_OTHER:
    default:
	break;
    }
}

static int set_kernel_arg(ecl_kernel_t* kern, cl_uint i, int type, void* value)
{
    if (i < kern->num_args) {
	int   old_type  = kern->arg[i].type;
	void* old_value = kern->arg[i].value;
	ref_kernel_arg(type, value);
	kern->arg[i].type  = type;
	kern->arg[i].value = value;
	unref_kernel_arg(old_type, old_value);
	return 0;
    }
    return -1;
}

/******************************************************************************
 *
 *   Resource destructors
 *
 *****************************************************************************/

static void ecl_platform_dtor(ErlNifEnv* env, ecl_object_t* obj)
{
    UNUSED(env);
    UNUSED(obj);
    DBG("ecl_platform_dtor: %p", obj);
    object_erase(obj);
    if (obj->parent) enif_release_resource(obj->parent);
}

static void ecl_device_dtor(ErlNifEnv* env, ecl_object_t* obj)
{
    UNUSED(env);
    UNUSED(obj);
    DBG("ecl_device_dtor: %p", obj);
    object_erase(obj);
    if (obj->parent) enif_release_resource(obj->parent);
}

static void ecl_queue_dtor(ErlNifEnv* env, ecl_object_t* obj)
{
    UNUSED(env);
    DBG("ecl_queue_dtor: %p", obj);
    clReleaseCommandQueue(obj->queue);
    object_erase(obj);
    if (obj->parent) enif_release_resource(obj->parent);
}

static void ecl_mem_dtor(ErlNifEnv* env, ecl_object_t* obj)
{
    UNUSED(env);
    DBG("ecl_mem_dtor: %p", obj);
    clReleaseMemObject(obj->mem);
    object_erase(obj);
    if (obj->parent) enif_release_resource(obj->parent);
}

static void ecl_sampler_dtor(ErlNifEnv* env, ecl_object_t* obj)
{
    UNUSED(env);
    DBG("ecl_sampler_dtor: %p", obj);
    clReleaseSampler(obj->sampler);
    object_erase(obj);
    if (obj->parent) enif_release_resource(obj->parent);
}

static void ecl_program_dtor(ErlNifEnv* env, ecl_object_t* obj)
{
    UNUSED(env);
    DBG("ecl_program_dtor: %p", obj);
    clReleaseProgram(obj->program);
    object_erase(obj);
    if (obj->parent) enif_release_resource(obj->parent);
}

static void ecl_kernel_dtor(ErlNifEnv* env, ecl_object_t* obj)
{
    ecl_kernel_t* kern = (ecl_kernel_t*) obj;
    cl_uint i;
    UNUSED(env);
    DBG("ecl_kernel_dtor: %p", kern);
    for (i = 0; i < kern->num_args; i++)
	unref_kernel_arg(kern->arg[i].type, kern->arg[i].value);
    enif_free(kern->arg);
    clReleaseKernel(kern->obj.kernel);
    object_erase(obj);
    if (obj->parent) enif_release_resource(obj->parent);
}

static void ecl_event_dtor(ErlNifEnv* env, ecl_object_t* obj)
{
    ecl_event_t* evt = (ecl_event_t*) obj;
    UNUSED(env);
    DBG("ecl_event_dtor: %p", evt);
    clReleaseEvent(evt->obj.event);
    object_erase(obj);
    if (evt->bin) {
	if (!evt->rl)
	    enif_release_binary(evt->bin);
	enif_free(evt->bin);
    }
    if (evt->bin_env)
	enif_free_env(evt->bin_env);
    if (obj->parent) enif_release_resource(obj->parent);
}

static void ecl_context_dtor(ErlNifEnv* env, ecl_object_t* obj)
{
    void* exit_value;
    ecl_context_t* ctx = (ecl_context_t*) obj;
    UNUSED(env);

    DBG("ecl_context_dtor: %p", ctx);
    clReleaseContext(ctx->obj.context);
    object_erase(obj);
    // parent is always = 0
    // kill the event thread
    ecl_thread_stop(ctx->thr, &exit_value);
}


/******************************************************************************
 *
 *   make/get
 *
 *****************************************************************************/

// For now, wrap the resource object {type,pointer-val,handle}
static ERL_NIF_TERM make_object(ErlNifEnv* env, const ERL_NIF_TERM type,
				void* robject)
{
    if (!robject)
	return ATOM(undefined);
    else
	return enif_make_tuple3(env,
				type,
				enif_make_ulong(env, (unsigned long) robject),
				enif_make_resource(env, robject));
}

// Accept {type,pointer-val,handle}
static int get_ecl_object(ErlNifEnv* env, const ERL_NIF_TERM term,
			  ecl_resource_t* rtype, bool nullp,  
			  ecl_object_t** robjectp)
{
    const ERL_NIF_TERM* elem;
    int arity;
    unsigned long handle;

    if (nullp && (term == ATOM(undefined))) {
	*robjectp = 0;
	return 1;
    }
    if (!enif_get_tuple(env, term, &arity, &elem))
	return 0;
    if (arity != 3)
	return 0;
    if (!enif_is_atom(env, elem[0]) || (elem[0] != rtype->type))
	return 0;
    if (!enif_get_ulong(env, elem[1], &handle))
	return 0;
    if (!enif_get_resource(env, elem[2], rtype->res, (void**) robjectp))
	return 0;
    if ((unsigned long)*robjectp != handle)
	return 0;
    return 1;
}

#if 0
static int get_ecl_object_list(ErlNifEnv* env, const ERL_NIF_TERM term,
			       ecl_resource_t* rtype, bool nullp,
			       ecl_object_t** robjv, size_t* rlen)
{
    size_t maxlen = *rlen;
    size_t n = 0;
    ERL_NIF_TERM list = term;

    while(n < maxlen) {
	ERL_NIF_TERM head, tail;
	
	if (enif_get_list_cell(env, list, &head, &tail)) {
	    if (!get_ecl_object(env, head, rtype, nullp, robjv))
		return 0;
	    n++;
	    robjv++;
	    list = tail;
	}
	else if (enif_is_empty_list(env, list)) {
	    *rlen = n;
	    return 1;
	}
	else 
	    return 0;
    }
    return 0;
}
#endif

static int get_object(ErlNifEnv* env, const ERL_NIF_TERM term,
		      ecl_resource_t* rtype, bool nullp,  
		      void** rptr)
{
    ecl_object_t* obj;
    if (get_ecl_object(env, term, rtype, nullp, &obj)) {
	*rptr = obj ? obj->opaque : 0;
	return 1;
    }
    return 0;
}

static int get_object_list(ErlNifEnv* env, const ERL_NIF_TERM term,
			   ecl_resource_t* rtype, bool nullp,
			   void** robjv, size_t* rlen)
{
    size_t maxlen = *rlen;
    size_t n = 0;
    ERL_NIF_TERM list = term;

    while(n < maxlen) {
	ERL_NIF_TERM head, tail;
	
	if (enif_get_list_cell(env, list, &head, &tail)) {
	    if (!get_object(env, head, rtype, nullp, robjv))
		return 0;
	    n++;
	    robjv++;
	    list = tail;
	}
	else if (enif_is_empty_list(env, list)) {
	    *rlen = n;
	    return 1;
	}
	else 
	    return 0;
    }
    return 0;
}



static int get_ulong_list(ErlNifEnv* env, const ERL_NIF_TERM term,
			  unsigned long* rvec, size_t* rlen)
{
    size_t maxlen = *rlen;
    size_t n = 0;
    ERL_NIF_TERM list = term;

    while(n < maxlen) {
	ERL_NIF_TERM head, tail;
	
	if (enif_get_list_cell(env, list, &head, &tail)) {
	    if (!enif_get_ulong(env, head, rvec))
		return 0;
	    n++;
	    rvec++;
	    list = tail;
	}
	else if (enif_is_empty_list(env, list)) {
	    *rlen = n;
	    return 1;
	}
	else 
	    return 0;
    }
    return 0;
}

static int get_binary_list(ErlNifEnv* env, const ERL_NIF_TERM term,
			   ErlNifBinary* rvec, size_t* rlen)
{
    size_t maxlen = *rlen;
    size_t n = 0;
    ERL_NIF_TERM list = term;

    while(n < maxlen) {
	ERL_NIF_TERM head, tail;
	
	if (enif_get_list_cell(env, list, &head, &tail)) {
	    if (!enif_inspect_binary(env, head, rvec))
		return 0;
	    n++;
	    rvec++;
	    list = tail;
	}
	else if (enif_is_empty_list(env, list)) {
	    *rlen = n;
	    return 1;
	}
	else 
	    return 0;
    }
    return 0;
}

// Copy a "local" binary to a new process independent environment
// fill the binary structure with the new data and return it.
//
static ERL_NIF_TERM ecl_copy_binary(ErlNifEnv* src_env, ErlNifEnv* dst_env,
				    ErlNifBinary* bin)
{
    ERL_NIF_TERM b;

    // make it a complete binary, may be a io_list !
    b = enif_make_binary(src_env, bin);
    // copy to destination environment
    return enif_make_copy(dst_env, b); 
}


// Lookup a openCL object (native => reource ecl_object_t*)
static ecl_object_t* ecl_lookup(ErlNifEnv* env, void* ptr)
{
    if (!ptr)
	return 0;
    else {
	ecl_env_t* ecl = enif_priv_data(env);
	ecl_object_t* obj;

	enif_rwlock_rlock(ecl->ref_lock);
	obj = (ecl_object_t*) lhash_lookup(&ecl->ref,(void*)EPTR_HANDLE(ptr));
	enif_rwlock_runlock(ecl->ref_lock);
	return obj;
    }
}

// Create a new openCL resource object
static ecl_object_t* ecl_new(ErlNifEnv* env, ecl_resource_t* rtype, 
			     void* ptr, ecl_object_t* parent)
{
    if (!ptr) 
	return 0;
    else {
	ecl_env_t* ecl = enif_priv_data(env);
	ecl_object_t* obj;

	obj = enif_alloc_resource(rtype->res, rtype->size);
	if (obj) {
	    if (parent)	enif_keep_resource(parent);
	    obj->opaque = ptr;
	    obj->env    = ecl;
	    obj->parent = parent;
	    enif_rwlock_rwlock(ecl->ref_lock);
	    lhash_insert_new(&ecl->ref, (void*)EPTR_HANDLE(ptr), obj);
	    enif_rwlock_rwunlock(ecl->ref_lock);
	}
	return obj;
    }
}

// lookup or create a new ecl_object_t resource
static ecl_object_t* ecl_maybe_new(ErlNifEnv* env, ecl_resource_t* rtype, 
				   void* ptr, ecl_object_t* parent, 
				   bool* is_new)
{
    ecl_object_t* obj = ecl_lookup(env, ptr);
    if (!obj) {
	obj = ecl_new(env, rtype, ptr, parent);
	*is_new = true;
    }
    else
	*is_new = false;
    return obj;
}

static ERL_NIF_TERM ecl_make_object(ErlNifEnv* env, ecl_resource_t* rtype, 
				    void* ptr, ecl_object_t* parent)
{
    ecl_object_t* obj = ecl_new(env,rtype,ptr,parent);
    ERL_NIF_TERM  res;
    res = make_object(env, rtype->type, obj);
    if (obj)
	enif_release_resource(obj);
    return res;
}


static ERL_NIF_TERM ecl_make_kernel(ErlNifEnv* env, cl_kernel kernel,
				    ecl_object_t* parent)
{
    ecl_kernel_t* kern = (ecl_kernel_t*) ecl_new(env,&kernel_r,
						 (void*)kernel,parent);
    ERL_NIF_TERM  res;
    cl_uint num_args;
    size_t sz;

    // Get number of arguments, FIXME: check error return
    clGetKernelInfo(kernel,CL_KERNEL_NUM_ARGS,sizeof(num_args),&num_args,0);
    sz = num_args*sizeof(ecl_kernel_arg_t);

    kern->arg = (ecl_kernel_arg_t*) enif_alloc(sz);
    memset(kern->arg, 0, sz);
    kern->num_args = num_args;
    
    res = make_object(env, kernel_r.type, kern);
    if (kern)
	enif_release_resource(kern);
    return res;
}

static ERL_NIF_TERM ecl_make_event(ErlNifEnv* env, cl_event event,
				   bool rd, bool rl,
				   ErlNifEnv* bin_env,
				   ErlNifBinary* bin, 
				   ecl_object_t* parent)
{
    ecl_event_t* evt = (ecl_event_t*) ecl_new(env,&event_r,
					      (void*)event,parent);
    ERL_NIF_TERM res;
    evt->bin_env = bin_env;
    evt->bin = bin;
    evt->rd  = rd;
    evt->rl  = rl;
    res = make_object(env, event_r.type, (ecl_object_t*) evt);
    if (evt)
	enif_release_resource(evt);
    return res;    
}

static ERL_NIF_TERM ecl_make_context(ErlNifEnv* env, cl_context context)
{
    ERL_NIF_TERM  res;
    ecl_context_t* ctx = (ecl_context_t*) ecl_new(env,&context_r,
						  (void*)context,0);
    
    ctx->thr = ecl_thread_start(ecl_context_main, ctx, 8); // 8K stack!
    res = make_object(env, context_r.type, (ecl_object_t*) ctx);
    if (ctx)
	enif_release_resource(ctx);
    return res;
}

// lookup or create resource object, return as erlang term
static ERL_NIF_TERM ecl_lookup_object(ErlNifEnv* env, ecl_resource_t* rtype, 
				      void* ptr, ecl_object_t* parent)
{
    bool is_new;
    ERL_NIF_TERM  res;
    ecl_object_t* obj = ecl_maybe_new(env,rtype,ptr,parent,&is_new);
    
    res = make_object(env, rtype->type, obj);
    if (obj && is_new)
	enif_release_resource(obj);
    return res;
}


typedef cl_int CL_API_CALL info_fn_t(void* ptr, cl_uint param_name, 
				     size_t param_value_size,
				     void* param_value, size_t* param_value_size_ret);
typedef cl_int CL_API_CALL info2_fn_t(void* ptr1, void* ptr2, cl_uint param_name, 
				      size_t param_value_size,
				      void* param_value, size_t* param_value_size_ret);

// return size of type
static size_t ecl_sizeof(ocl_type_t type)
{
    switch(type) {
    case OCL_CHAR: return sizeof(cl_char);
    case OCL_UCHAR: return sizeof(cl_uchar);
    case OCL_SHORT: return sizeof(cl_short);
    case OCL_USHORT: return sizeof(cl_ushort);
    case OCL_INT: return sizeof(cl_int);
    case OCL_UINT: return sizeof(cl_uint);
    case OCL_LONG: return sizeof(cl_long);
    case OCL_ULONG: return sizeof(cl_ulong);
    case OCL_HALF: return sizeof(cl_half);
    case OCL_FLOAT: return sizeof(cl_float);
    case OCL_DOUBLE: return sizeof(cl_double);
    case OCL_BOOL: return sizeof(cl_bool);
    case OCL_STRING: return sizeof(cl_char*);
    case OCL_ENUM: return sizeof(cl_int);
    case OCL_BITFIELD: return sizeof(cl_bitfield);
    case OCL_POINTER: return sizeof(void*);
    case OCL_SIZE: return sizeof(size_t);
    case OCL_PLATFORM: return sizeof(void*);
    case OCL_DEVICE: return sizeof(void*);
    case OCL_CONTEXT: return sizeof(void*);
    case OCL_PROGRAM: return sizeof(void*);
    case OCL_COMMAND_QUEUE: return sizeof(void*);
    case OCL_IMAGE_FORMAT: return sizeof(cl_image_format);
    default:
	DBG("info_size: unknown type %d detected", type);
	return sizeof(cl_int);
    }
}

// put basic value types
static ERL_NIF_TERM make_info_element(ErlNifEnv* env, ocl_type_t type, void* ptr, ecl_kv_t* kv)
{
    switch(type) {
    case OCL_CHAR:  return enif_make_int(env, *((cl_char*)ptr));
    case OCL_SHORT: return enif_make_int(env, *((cl_short*)ptr));
    case OCL_INT: return enif_make_int(env, *((cl_int*)ptr));
    case OCL_LONG: return enif_make_int64(env, *((cl_long*)ptr));
    case OCL_UCHAR:  return enif_make_uint(env, *((cl_uchar*)ptr));
    case OCL_USHORT: return enif_make_uint(env, *((cl_ushort*)ptr));
    case OCL_UINT: return enif_make_uint(env, *((cl_uint*)ptr));
    case OCL_HALF: return enif_make_uint(env, *((cl_half*)ptr));
    case OCL_ULONG: return enif_make_uint64(env, *((cl_ulong*)ptr));
    case OCL_SIZE: return enif_make_ulong(env, *((size_t*)ptr));
    case OCL_FLOAT: return enif_make_double(env, *((cl_float*)ptr));
    case OCL_DOUBLE: return enif_make_double(env, *((cl_double*)ptr));
    case OCL_BOOL: return (*((cl_bool*)ptr)) ? ATOM(true) : ATOM(false);
    // case POINTER: cbuf_put_pointer(data, *((pointer_t*)ptr)); break;
    case OCL_STRING:
	return enif_make_string_len(env, (char*) ptr, strlen((char*) ptr), ERL_NIF_LATIN1);

    case OCL_BITFIELD:
	return make_bitfields(env, *((cl_bitfield*)ptr), kv);

    case OCL_ENUM:
	return make_enum(env, *((cl_int*)ptr), kv);

    case OCL_POINTER: 
	return enif_make_ulong(env, *((intptr_t*)ptr));

    case OCL_PLATFORM:
	return ecl_lookup_object(env,&platform_r,*(void**)ptr,0);

    case OCL_DEVICE:
	return ecl_lookup_object(env,&device_r,*(void**)ptr,0);

    case OCL_CONTEXT:
	return ecl_lookup_object(env,&context_r,*(void**)ptr,0);

    case OCL_PROGRAM:
	// FIXME: find context object, pass as parent
	return ecl_lookup_object(env,&program_r,*(void**)ptr,0);

    case OCL_COMMAND_QUEUE:
	// FIXME: find context object, pass as parent
	return ecl_lookup_object(env,&command_queue_r,*(void**)ptr,0);

    case OCL_IMAGE_FORMAT: {
	cl_image_format* fmt = (cl_image_format*) ptr;
	ERL_NIF_TERM channel_order;
	ERL_NIF_TERM channel_type;
	channel_order = make_enum(env,fmt->image_channel_order,
				  kv_channel_order);
	channel_type = make_enum(env,fmt->image_channel_data_type,
				 kv_channel_type);
	return enif_make_tuple2(env, channel_order, channel_type);
    }

    default:
	return ATOM(undefined);
    }
}


static ERL_NIF_TERM make_info_value(ErlNifEnv* env, ecl_info_t* iptr, void* buf, size_t buflen)
{
    char* dptr = (char*) buf;
    ERL_NIF_TERM value;

    if (iptr->is_array) {  // arrays are return as lists of items
	ERL_NIF_TERM list = enif_make_list(env, 0);
	size_t elem_size = ecl_sizeof(iptr->info_type);
	size_t n = (buflen / elem_size);
	dptr += (n*elem_size);  // run backwards!!!
	while (buflen >= elem_size) {
	    dptr -= elem_size;
	    value = make_info_element(env, iptr->info_type, dptr, iptr->extern_info);
	    list = enif_make_list_cell(env, value, list);
	    buflen -= elem_size;
	}
	value = list;
    }
    else {
	value = make_info_element(env, iptr->info_type, dptr, iptr->extern_info);
    }
    return value;
}

// Find object value
// return {ok,Value} | {error,Reason} | exception badarg
//
ERL_NIF_TERM make_object_info(ErlNifEnv* env,  ERL_NIF_TERM key, ecl_object_t* obj, info_fn_t* func, 
			      ecl_info_t* info, size_t num_info)
{
    size_t returned_size = 0;
    size_t size = MAX_INFO_SIZE;    
    unsigned char buf[MAX_INFO_SIZE];
    void* ptr = buf;
    ERL_NIF_TERM res;
    cl_int err;
    unsigned int i;

    if (!enif_is_atom(env, key))
	return enif_make_badarg(env);
    i = 0;
    while((i < num_info) && (*info[i].info_key != key))
	i++;
    if (i == num_info)
	return enif_make_badarg(env);  // or error ?

    err = (*func)(obj->opaque,info[i].info_id,size,ptr,&returned_size);
    if (err == CL_INVALID_VALUE) {
        // try again allocate returned_size, returned_size does not
	// (yet) return the actual needed bytes (by spec) 
	// but it looks like it... ;-)
	size = returned_size;
	if (!(ptr = enif_alloc(size)))
	    return ecl_make_error(env, CL_OUT_OF_HOST_MEMORY);
	err = (*func)(obj->opaque,info[i].info_id,size,ptr,&returned_size);
    }

    if (!err) {
	res = enif_make_tuple2(env, ATOM(ok), 
			       make_info_value(env,&info[i],ptr,returned_size));
    }
    else
	res = ecl_make_error(env, err);
    if (ptr != buf)
	enif_free(ptr);
    return res;
}


ERL_NIF_TERM make_object_info2(ErlNifEnv* env,  ERL_NIF_TERM key, ecl_object_t* obj1, ecl_object_t* obj2,
				   info2_fn_t* func, ecl_info_t* info, size_t num_info)
{
    size_t returned_size = 0;
    cl_long *buf;
    cl_int err;
    unsigned int i;
    ERL_NIF_TERM result;

    if (!enif_is_atom(env, key))
	return enif_make_badarg(env);
    i = 0;
    while((i < num_info) && (*info[i].info_key != key))
	i++;
    if (i == num_info)
	return enif_make_badarg(env);  // or error ?
    if (!(err = (*func)(obj1->opaque, obj2->opaque, info[i].info_id, 
			0, NULL, &returned_size))) {
	if (!(buf = enif_alloc(returned_size)))
	    return ecl_make_error(env, CL_OUT_OF_RESOURCES);
	if (!(err = (*func)(obj1->opaque, obj2->opaque, info[i].info_id, 
			    returned_size, buf, &returned_size))) {
	    result = enif_make_tuple2(env, ATOM(ok), make_info_value(env, &info[i], buf, returned_size));
	    enif_free(buf);
	    return result;
	}
    }
    return ecl_make_error(env, err);
}

/******************************************************************************
 *
 * main ecl event loop run as a thread.
 *  The main purpose is to dispatch and send messages to owners
 *
 *****************************************************************************/

static void* ecl_context_main(void* arg)
{
    ecl_thread_t* self = arg;
    // ecl_context_t* ctx = self->arg;

    DBG("ecl_context_main: started (%p)", self);

    while(1) {
	ecl_message_t m;
	ecl_message_recv(self, &m);

	switch(m.type) {
	case ECL_MESSAGE_STOP: {
	    DBG("ecl_context_main: stopped by command");
	    if (m.env) {
		enif_send(0, &m.sender, m.env, 
			  enif_make_tuple3(m.env, 
					   ATOM(cl_async), m.ref,
					   ATOM(ok)));
		enif_free_env(m.env);
	    }
	    ecl_thread_exit(self);
	    break;
	}

	case ECL_MESSAGE_FLUSH: {  // flush message queue
	    cl_int err;

	    DBG("ecl_context_main: flush q=%lu", (unsigned long) m.queue);
	    err = clFlush(m.queue->queue);
	    // send {cl_async, Ref, ok | {error,Reason}}
	    if (m.env) {
		ERL_NIF_TERM reply;
		int res;
		reply = !err ? ATOM(ok) : ecl_make_error(m.env, err);
		res = enif_send(0, &m.sender, m.env, 
				enif_make_tuple3(m.env, 
						 ATOM(cl_async),
						 m.ref,
						 reply));
		DBG("ecl_context_main: send r=%d", res);
		enif_free_env(m.env);
	    }
	    enif_release_resource(m.queue);
	    break;
	}

	case ECL_MESSAGE_FINISH: {  // finish message queue
	    cl_int err;
	    DBG("ecl_context_main: finish q=%lu", (unsigned long) m.queue);
	    err = clFlush(m.queue->queue);
	    // send {cl_async, Ref, ok | {error,Reason}}
	    if (m.env) {
		ERL_NIF_TERM reply;
		int res;

		reply = !err ? ATOM(ok) : ecl_make_error(m.env, err);
		res = enif_send(0, &m.sender, m.env, 
				enif_make_tuple3(m.env, 
						 ATOM(cl_async), m.ref,
						 reply));
		DBG("ecl_context_main: send r=%d", res);
		enif_free_env(m.env);
	    }
	    enif_release_resource(m.queue);
	    break;
	}

	case ECL_MESSAGE_WAIT_FOR_EVENT: { // wait for one event
	    cl_int err;
	    cl_event list[1];
	    DBG("ecl_context_main: wait_for_event e=%lu",
		(unsigned long) m.event);
	    list[0] = m.event->obj.event;
	    err = clWaitForEvents(1, list);
	    DBG("ecl_context_main: wait_for_event err=%d", err);
	    // reply to caller pid !
	    if (m.env) {
		ERL_NIF_TERM reply;
		int res;
		
		if (!err) {
		    cl_int status;
		    // read status COMPLETE | ERROR
		    // FIXME: check error
		    clGetEventInfo(m.event->obj.event,
				   CL_EVENT_COMMAND_EXECUTION_STATUS,
				   sizeof(status), &status, 0);
		    switch(status) {
		    case CL_COMPLETE:
			DBG("ecl_context_main: wait_for_event complete");
			if (m.event->bin && m.event->rd) {
			    m.event->rl = true;
			    reply = enif_make_binary(m.env, m.event->bin);
			}
			else
			    reply = ATOM(complete);
			break;
		    default:
			DBG("ecl_context_main: wait_for_event: status=%d");
			// must/should be an error
			reply = ecl_make_error(m.env, status);
			break;
		    }
		}
		else
		    reply = ecl_make_error(m.env, err);
		res = enif_send(0, &m.sender, m.env,
				enif_make_tuple3(m.env, 
						 ATOM(cl_event), m.ref,
						 reply));
		DBG("ecl_context_main: send r=%d", res);
		enif_free_env(m.env);
	    }
	    enif_release_resource(m.event);
	    break;
	}
	default:
	    break;
	}
    }
    return 0;
}


//
// API functions
//

// noop - no operation for NIF interface performance benchmarking
static ERL_NIF_TERM ecl_noop(ErlNifEnv* env, int argc,
			     const ERL_NIF_TERM argv[])
{
    UNUSED(env);
    UNUSED(argc);
    UNUSED(argv);
    return ATOM(ok);
}

static ERL_NIF_TERM ecl_get_platform_ids(ErlNifEnv* env, int argc,
					 const ERL_NIF_TERM argv[])
{
    cl_uint          num_platforms;
    cl_platform_id   platform_id[MAX_PLATFORMS];
    ERL_NIF_TERM     idv[MAX_PLATFORMS];
    ERL_NIF_TERM     platform_list;
    cl_uint i;
    cl_int err;
    UNUSED(argc);
    UNUSED(argv);

    if ((err = clGetPlatformIDs(MAX_PLATFORMS, platform_id, &num_platforms)))
	return ecl_make_error(env, err);

    for (i = 0; i < num_platforms; i++)
	idv[i] = ecl_make_object(env,&platform_r,platform_id[i],0);

    platform_list = enif_make_list_from_array(env, idv,num_platforms);
    return enif_make_tuple2(env, ATOM(ok), platform_list);
}

static ERL_NIF_TERM ecl_get_platform_info(ErlNifEnv* env, int argc,
					  const ERL_NIF_TERM argv[])
{
    ecl_object_t* o_platform;
    UNUSED(argc);

    if (!get_ecl_object(env, argv[0], &platform_r, false, &o_platform))
	return enif_make_badarg(env);
    return make_object_info(env, argv[1], o_platform, 
			    (info_fn_t*) clGetPlatformInfo, 
			    platform_info, 
			    sizeof_array(platform_info));
}


static ERL_NIF_TERM ecl_get_device_ids(ErlNifEnv* env, int argc, 
				       const ERL_NIF_TERM argv[])
{
    cl_device_type   device_type = 0;
    cl_device_id     device_id[MAX_DEVICES];
    ERL_NIF_TERM     idv[MAX_DEVICES];
    ERL_NIF_TERM     device_list;
    cl_uint          num_devices;
    cl_uint          i;
    cl_platform_id   platform;
    cl_int err;
    UNUSED(argc);
    
    if (!get_object(env, argv[0], &platform_r, true,(void**)&platform))
	return enif_make_badarg(env);
    if (!get_bitfields(env, argv[1], &device_type, kv_device_type))
	return enif_make_badarg(env);
    if ((err = clGetDeviceIDs(platform, device_type, MAX_DEVICES, 
			      device_id, &num_devices)))
	return ecl_make_error(env, err);
    
    for (i = 0; i < num_devices; i++)
	idv[i] = ecl_make_object(env, &device_r, device_id[i], 0);
    device_list = enif_make_list_from_array(env, idv, num_devices);
    return enif_make_tuple2(env, ATOM(ok), device_list);
}

static ERL_NIF_TERM ecl_get_device_info(ErlNifEnv* env, int argc, 
					const ERL_NIF_TERM argv[])
{
    ecl_object_t* o_device;
    UNUSED(argc);

    if (!get_ecl_object(env, argv[0], &device_r, false, &o_device))
	return enif_make_badarg(env);	
    return make_object_info(env, argv[1], o_device, 
			    (info_fn_t*) clGetDeviceInfo, 
			    device_info, 
			    sizeof_array(device_info));
}

typedef struct {
    ErlNifPid        sender;  // sender pid
    ErlNifEnv*        s_env;  // senders message environment (ref, bin's etc)
    ErlNifEnv*        r_env;  // receiver message environment (ref, bin's etc)
    ErlNifTid           tid;  // Calling thread
} ecl_notify_data_t;

void CL_CALLBACK ecl_context_notify(const char *errinfo, 
				    const void* private_info, size_t cb,
				    void * user_data)
{
    /* ecl_notify_data_t* bp = user_data; */
    /* ERL_NIF_TERM reply; */
    /* ErlNifEnv*   s_env; */
    /* int res; */

    DBG("ecl_context_notify:  user_data=%p", user_data);        
    DBG("ecl_context_notify:  priv_info=%p cb=%d", private_info, cb);
    CL_ERROR("CL ERROR ASYNC: %s", errinfo);
}

//
// cl:create_context([cl_device_id()]) -> 
//   {ok, cl_context()} | {error, cl_error()}
//
static ERL_NIF_TERM ecl_create_context(ErlNifEnv* env, int argc, 
				       const ERL_NIF_TERM argv[])
{
    cl_device_id     device_list[MAX_DEVICES];
    size_t           num_devices = MAX_DEVICES;
    cl_context       context;
    cl_int err;
    ecl_notify_data_t* bp;

    UNUSED(argc);

    if (!get_object_list(env, argv[0], &device_r, false, 
			 (void**) device_list, &num_devices))
	return enif_make_badarg(env);

    if (!(bp = enif_alloc(sizeof(ecl_notify_data_t))))
	return ecl_make_error(env, CL_OUT_OF_RESOURCES);  // enomem?
    
    if (!(bp->r_env = enif_alloc_env())) {
	enif_free(bp);
	return ecl_make_error(env, CL_OUT_OF_RESOURCES);  // enomem?
    }
    (void) enif_self(env, &bp->sender);
    bp->s_env = env;
    bp->tid = enif_thread_self();
    DBG("ecl_create_context: self %p", bp->tid);

    context = clCreateContext(0, num_devices, device_list, 
			      ecl_context_notify,
			      bp,
			      &err);
    if (context) {
	ERL_NIF_TERM t;
	t = ecl_make_context(env, context);
	return enif_make_tuple2(env, ATOM(ok), t);
    }
    return ecl_make_error(env, err);
}

static ERL_NIF_TERM ecl_get_context_info(ErlNifEnv* env, int argc, 
					 const ERL_NIF_TERM argv[])
{
    ecl_object_t* o_context;
    UNUSED(argc);

    if (!get_ecl_object(env, argv[0], &context_r, false, &o_context))
	return enif_make_badarg(env);
    return make_object_info(env, argv[1], o_context,
			    (info_fn_t*) clGetContextInfo,
			    context_info,
			    sizeof_array(context_info));
}

static ERL_NIF_TERM ecl_create_queue(ErlNifEnv* env, int argc, 
				     const ERL_NIF_TERM argv[])
{
    ecl_object_t* o_context;
    cl_device_id  device;
    cl_command_queue_properties properties;
    cl_command_queue queue;
    cl_int err;
    UNUSED(argc);

    if (!get_ecl_object(env, argv[0], &context_r, false, &o_context))
	return enif_make_badarg(env);
    if (!get_object(env, argv[1], &device_r, false, (void**) &device))
	return enif_make_badarg(env);
    if (!get_bitfields(env, argv[2], &properties,
		       kv_command_queue_properties))
	return enif_make_badarg(env);
    queue = clCreateCommandQueue(o_context->context, device, properties,
				 &err);
    if (queue) {
	ERL_NIF_TERM t;
	t = ecl_make_object(env, &command_queue_r,(void*) queue, o_context);
	return enif_make_tuple2(env, ATOM(ok), t);
    }
    return ecl_make_error(env, err);
}

static ERL_NIF_TERM ecl_get_queue_info(ErlNifEnv* env, int argc, 
				       const ERL_NIF_TERM argv[])
{
    ecl_object_t* o_queue;
    UNUSED(argc);

    if (!get_ecl_object(env, argv[0], &command_queue_r, false, &o_queue))
	return enif_make_badarg(env);
    return make_object_info(env, argv[1], o_queue, 
			    (info_fn_t*) clGetCommandQueueInfo, 
			    queue_info,
			    sizeof_array(queue_info));
}


static ERL_NIF_TERM ecl_create_buffer(ErlNifEnv* env, int argc, 
				      const ERL_NIF_TERM argv[])
{
    ecl_object_t* o_context;
    size_t size;
    cl_mem_flags mem_flags;
    cl_mem mem;
    ErlNifBinary bin;
    void* host_ptr = 0;
    cl_int err;
    UNUSED(argc);


    if (!get_ecl_object(env, argv[0], &context_r, false, &o_context))
	return enif_make_badarg(env);
    if (!get_bitfields(env, argv[1], &mem_flags, kv_mem_flags))
	return enif_make_badarg(env);
    if (!enif_get_ulong(env, argv[2], &size))
	return enif_make_badarg(env);
    if (!enif_inspect_iolist_as_binary(env, argv[3], &bin))
	return enif_make_badarg(env);
    // How do we keep binary data (CL_MEM_USE_HOST_PTR) 
    // We should probably make sure that the buffer is read_only in this
    // case!
    // we must be able to reference count the binary object!
    // USE enif_make_copy !!!! this copy is done to the thread environment!
    if (bin.size > 0) {
	host_ptr = bin.data;
	mem_flags |= CL_MEM_COPY_HOST_PTR;
	if (size < bin.size)
	    size = bin.size;
    }
    else if (size)
	mem_flags |= CL_MEM_ALLOC_HOST_PTR;

    mem = clCreateBuffer(o_context->context, mem_flags, size,
			 host_ptr, &err);

    if (!err) {
	ERL_NIF_TERM t;
	t = ecl_make_object(env, &mem_r,(void*) mem, o_context);
	return enif_make_tuple2(env, ATOM(ok), t);
    }
    return ecl_make_error(env, err);
}


static ERL_NIF_TERM ecl_create_image2d(ErlNifEnv* env, int argc, 
					const ERL_NIF_TERM argv[])
{
    ecl_object_t* o_context;
    size_t width;
    size_t height;
    size_t row_pitch;
    cl_image_format format;
    cl_mem_flags mem_flags;
    cl_mem mem;
    ErlNifBinary bin;
    void* host_ptr = 0;
    const ERL_NIF_TERM* array;
    int arity;
    cl_int err;
    UNUSED(argc);

    if (!get_ecl_object(env, argv[0], &context_r, false, &o_context))
	return enif_make_badarg(env);
    if (!get_bitfields(env, argv[1], &mem_flags, kv_mem_flags))
	return enif_make_badarg(env);

    if (!enif_get_tuple(env, argv[2], &arity, &array) || (arity != 2))
	return enif_make_badarg(env);
    if (!get_enum(env, array[0], &format.image_channel_order, kv_channel_order))
	return enif_make_badarg(env);	
    if (!get_enum(env, array[1], &format.image_channel_data_type,
		  kv_channel_type))
	return enif_make_badarg(env);	

    if (!enif_get_ulong(env, argv[3], &width))
	return enif_make_badarg(env);
    if (!enif_get_ulong(env, argv[4], &height))
	return enif_make_badarg(env);
    if (!enif_get_ulong(env, argv[5], &row_pitch))
	return enif_make_badarg(env);

    if (!enif_inspect_iolist_as_binary(env, argv[6], &bin))
	return enif_make_badarg(env);
    // How do we keep binary data (CL_MEM_USE_HOST_PTR) (read_only)
    // we must be able to reference count the binary object!
    if (bin.size > 0) {
	host_ptr = bin.data;
	mem_flags |= CL_MEM_COPY_HOST_PTR;
    }
    else if (width && height)
	mem_flags |= CL_MEM_ALLOC_HOST_PTR;

    mem = clCreateImage2D(o_context->context, mem_flags, &format,
			  width, height, row_pitch,
			  host_ptr, &err);

    if (!err) {
	ERL_NIF_TERM t;
	t = ecl_make_object(env, &mem_r,(void*) mem, o_context);
	return enif_make_tuple2(env, ATOM(ok), t);
    }
    return ecl_make_error(env, err);
}

static ERL_NIF_TERM ecl_create_image3d(ErlNifEnv* env, int argc, 
				       const ERL_NIF_TERM argv[])
{
    ecl_object_t* o_context;
    size_t width;
    size_t height;
    size_t depth;
    size_t row_pitch;
    size_t slice_pitch;
    cl_image_format format;
    cl_mem_flags mem_flags;
    cl_mem mem;
    ErlNifBinary bin;
    void* host_ptr = 0;
    const ERL_NIF_TERM* array;
    int arity;
    cl_int err;
    UNUSED(argc);

    if (!get_ecl_object(env, argv[0], &context_r, false, &o_context))
	return enif_make_badarg(env);
    if (!get_bitfields(env, argv[1], &mem_flags, kv_mem_flags))
	return enif_make_badarg(env);

    if (!enif_get_tuple(env, argv[2], &arity, &array) || (arity != 2))
	return enif_make_badarg(env);
    if (!get_enum(env, array[0], &format.image_channel_order, kv_channel_order))
	return enif_make_badarg(env);	
    if (!get_enum(env, array[1], &format.image_channel_data_type,
		  kv_channel_type))
	return enif_make_badarg(env);	

    if (!enif_get_ulong(env, argv[3], &width))
	return enif_make_badarg(env);
    if (!enif_get_ulong(env, argv[4], &height))
	return enif_make_badarg(env);
    if (!enif_get_ulong(env, argv[5], &depth))
	return enif_make_badarg(env);
    if (!enif_get_ulong(env, argv[6], &row_pitch))
	return enif_make_badarg(env);
    if (!enif_get_ulong(env, argv[7], &slice_pitch))
	return enif_make_badarg(env);

    if (!enif_inspect_iolist_as_binary(env, argv[8], &bin))
	return enif_make_badarg(env);
    // How do we keep binary data (CL_MEM_USE_HOST_PTR)  (read_only)
    // we must be able to reference count the binary object!
    if (bin.size > 0) {
	host_ptr = bin.data;
	mem_flags |= CL_MEM_COPY_HOST_PTR;
    }
    else if (width && height && depth)
	mem_flags |= CL_MEM_ALLOC_HOST_PTR;

    mem = clCreateImage3D(o_context->context, mem_flags, &format,
			  width, height, depth, row_pitch, slice_pitch,
			  host_ptr, &err);

    if (mem) {
	ERL_NIF_TERM t;
	t = ecl_make_object(env, &mem_r,(void*) mem, o_context);
	return enif_make_tuple2(env, ATOM(ok), t);
    }
    return ecl_make_error(env, err);
}

static ERL_NIF_TERM ecl_get_supported_image_formats(ErlNifEnv* env, int argc, 
						    const ERL_NIF_TERM argv[])
{
    cl_context context;
    cl_mem_flags flags;
    cl_mem_object_type image_type;
    cl_image_format image_format[MAX_IMAGE_FORMATS];
    cl_uint num_image_formats;
    cl_int err;
    UNUSED(argc);

    if (!get_object(env, argv[0], &context_r, false, (void**) &context))
	return enif_make_badarg(env);
    if (!get_bitfields(env, argv[1], &flags, kv_mem_flags))
	return enif_make_badarg(env);	
    if (!get_enum(env, argv[2], &image_type, kv_mem_object_type))
	return enif_make_badarg(env);	
    err = clGetSupportedImageFormats(context, flags, image_type,
				     MAX_IMAGE_FORMATS,
				     image_format,
				     &num_image_formats);
    if (!err) {
	int i = (int) num_image_formats;
	ERL_NIF_TERM list = enif_make_list(env, 0);

	while(i) {
	    ERL_NIF_TERM channel_order, channel_type;
	    ERL_NIF_TERM elem;
	    i--;
	    channel_order = make_enum(env,
				      image_format[i].image_channel_order, 
				      kv_channel_order);
	    channel_type = make_enum(env,
				     image_format[i].image_channel_data_type,
				     kv_channel_type);
	    elem = enif_make_tuple2(env, channel_order, channel_type);
	    list = enif_make_list_cell(env, elem, list);
	}
	return enif_make_tuple2(env, ATOM(ok), list);
    }
    return ecl_make_error(env, err);
}


static ERL_NIF_TERM ecl_get_mem_object_info(ErlNifEnv* env, int argc, 
					    const ERL_NIF_TERM argv[])
{
    ecl_object_t* o_mem;
    UNUSED(argc);

    if (!get_ecl_object(env, argv[0], &mem_r, false, &o_mem))
	return enif_make_badarg(env);
    return make_object_info(env, argv[1], o_mem, 
			    (info_fn_t*) clGetMemObjectInfo,
			    mem_info,
			    sizeof_array(mem_info));
}

static ERL_NIF_TERM ecl_get_image_info(ErlNifEnv* env, int argc, 
				       const ERL_NIF_TERM argv[])
{
    ecl_object_t* o_mem;
    UNUSED(argc);

    if (!get_ecl_object(env, argv[0], &mem_r, false, &o_mem))
	return enif_make_badarg(env);
    return make_object_info(env, argv[1], o_mem,
			    (info_fn_t*) clGetImageInfo,
			    image_info, 
			    sizeof_array(image_info));
}

//
// cl:create_sampler(Context::cl_context(),Normalized::boolean(),
//		     AddressingMode::cl_addressing_mode(),
//		     FilterMode::cl_filter_mode()) -> 
//    {'ok', cl_sampler()} | {'error', cl_error()}.
//

static ERL_NIF_TERM ecl_create_sampler(ErlNifEnv* env, int argc,
				       const ERL_NIF_TERM argv[])
{
    ecl_object_t* o_context;
    cl_bool normalized_coords;
    cl_addressing_mode addressing_mode;
    cl_filter_mode filter_mode;
    cl_sampler sampler;
    cl_int err;
    UNUSED(argc);

    if (!get_ecl_object(env, argv[0], &context_r, false, &o_context))
	return enif_make_badarg(env);
    if (!get_bool(env, argv[1], &normalized_coords))
	return enif_make_badarg(env);
    if (!get_enum(env, argv[2], &addressing_mode, kv_addressing_mode))
	return enif_make_badarg(env);
    if (!get_enum(env, argv[3], &filter_mode, kv_filter_mode))
	return enif_make_badarg(env);

    sampler = clCreateSampler(o_context->context,
			      normalized_coords, addressing_mode, filter_mode,
			      &err);
    if (!err) {
	ERL_NIF_TERM t;
	t = ecl_make_object(env, &sampler_r,(void*) sampler, o_context);
	return enif_make_tuple2(env, ATOM(ok), t);
    }
    return ecl_make_error(env, err);
}


static ERL_NIF_TERM ecl_get_sampler_info(ErlNifEnv* env, int argc, 
					 const ERL_NIF_TERM argv[])
{
    ecl_object_t* o_sampler;
    UNUSED(argc);

    if (!get_ecl_object(env, argv[0], &sampler_r, false, &o_sampler))
	return enif_make_badarg(env);
    return make_object_info(env, argv[1], o_sampler,
			    (info_fn_t*) clGetSamplerInfo,
			    sampler_info,
			    sizeof_array(sampler_info));
}

//
// cl:create_program_with_source(Context::cl_context(), Source::iodata()) ->
//   {'ok', cl_program()} | {'error', cl_error()}
//
static ERL_NIF_TERM ecl_create_program_with_source(ErlNifEnv* env, int argc, 
						   const ERL_NIF_TERM argv[])
{
    ecl_object_t* o_context;
    cl_program program;
    ErlNifBinary source;
    char* strings[1];
    size_t lengths[1];
    cl_int err;
    UNUSED(argc);

    if (!get_ecl_object(env, argv[0], &context_r, false, &o_context))
	return enif_make_badarg(env);
    if (!enif_inspect_iolist_as_binary(env, argv[1], &source))
	return enif_make_badarg(env);
    strings[0] = (char*) source.data;
    lengths[0] = source.size;
    program = clCreateProgramWithSource(o_context->context,
					1,
					(const char**) strings,
					lengths,
					&err);
    if (!err) {
	ERL_NIF_TERM t;
	t = ecl_make_object(env, &program_r,(void*) program, o_context);
	return enif_make_tuple2(env, ATOM(ok), t);
    }
    return ecl_make_error(env, err);
}

//
//  cl:create_program_with_binary(Context::cl_context(),
//                                  DeviceList::[cl_device_id()],
//                                  BinaryList::[binary()]) ->
//    {'ok', cl_program()} | {'error', cl_error()}
//
static ERL_NIF_TERM ecl_create_program_with_binary(ErlNifEnv* env, int argc, 
						   const ERL_NIF_TERM argv[])
{
    ecl_object_t* o_context;
    cl_program     program;
    cl_device_id   device_list[MAX_DEVICES];
    size_t         num_devices = MAX_DEVICES;
    ErlNifBinary   binary_list[MAX_DEVICES];
    size_t         num_binaries = MAX_DEVICES;
    size_t         lengths[MAX_DEVICES];
    unsigned char* data[MAX_DEVICES];
    cl_uint        i;
    cl_int         status[MAX_DEVICES];
    cl_int         err;
    UNUSED(argc);

    if (!get_ecl_object(env, argv[0], &context_r, false, &o_context))
	return enif_make_badarg(env);
    if (!get_object_list(env, argv[1], &device_r, false,
			 (void**) device_list, &num_devices))
	return enif_make_badarg(env);
    if (!get_binary_list(env, argv[2], binary_list, &num_binaries))
	return enif_make_badarg(env);
    if (num_binaries != num_devices)
	return enif_make_badarg(env);
	
    for (i = 0; i < num_devices; i++) {
	lengths[i] = binary_list[i].size;
	data[i]    = binary_list[i].data;
    }
    program = clCreateProgramWithBinary(o_context->context,
					num_devices,
					(const cl_device_id*) device_list,
					(const size_t*) lengths,
					(const unsigned char**) data,
					status,
					&err);
    if (!err) {
	ERL_NIF_TERM t;
	t = ecl_make_object(env, &program_r,(void*) program, o_context);
	return enif_make_tuple2(env, ATOM(ok), t);
    }
    // FIXME: handle the value in the status array
    // In cases of error we can then detect which binary was corrupt...
    return ecl_make_error(env, err);
}

//
// @spec async_build_program(Program::cl_program(),
//                     DeviceList::[cl_device_id()],
//                     Options::string()) ->
//  {'ok',Ref} | {'error', cl_error()}
//
//
// Notification functio for clBuildProgram
// Passed to main thread by sending a async response
// FIXME: lock needed?
//
typedef struct {
    ErlNifPid        sender;  // sender pid
    ErlNifEnv*        s_env;  // senders message environment (ref, bin's etc)
    ErlNifEnv*        r_env;  // receiver message environment (ref, bin's etc)
    ErlNifTid           tid;  // Calling thread
    ERL_NIF_TERM        ref;  // ref (in env!)
    ecl_object_t*  program;
} ecl_build_data_t;

void CL_CALLBACK ecl_build_notify(cl_program program, void* user_data)
{
    ecl_build_data_t* bp = user_data;
    ERL_NIF_TERM reply;
    ErlNifEnv*        s_env;
    int res;
    UNUSED(program);

    DBG("ecl_build_notify: done user_data=%p", user_data);

    // FIXME: check all devices for build_status!
    // clGetProgramBuildInfo(bp->program->program, CL_PROGRAM_BUILD_STATUS,

    // reply = !err ? ATOM(ok) : ecl_make_error(bp->env, err);
        
    if(enif_equal_tids(bp->tid, enif_thread_self()))
       s_env = bp->s_env;
    else
       s_env = 0;

    reply = ATOM(ok);
    res = enif_send(s_env, &bp->sender, bp->r_env, 
		    enif_make_tuple3(bp->r_env, 
				     ATOM(cl_async),
				     bp->ref,
				     reply));
    DBG("ecl_build_notify: send r=%d", res);
    enif_free_env(bp->r_env);
    enif_release_resource(bp->program);
    enif_free(bp);
}


static ERL_NIF_TERM ecl_async_build_program(ErlNifEnv* env, int argc, 
					    const ERL_NIF_TERM argv[])
{
    ecl_object_t*    o_program;
    cl_device_id     device_list[MAX_DEVICES];
    size_t           num_devices = MAX_DEVICES;
    char             options[MAX_OPTION_LIST];
    ERL_NIF_TERM     ref;
    ecl_build_data_t* bp;
    cl_int           err;
    UNUSED(argc);

    if (!get_ecl_object(env, argv[0], &program_r, false, &o_program))
	return enif_make_badarg(env);
    if (!get_object_list(env, argv[1], &device_r, false,
			 (void**) device_list, &num_devices))
	return enif_make_badarg(env);
    if (!enif_get_string(env, argv[2], options, sizeof(options),ERL_NIF_LATIN1))
	return enif_make_badarg(env);
    if (!(bp = enif_alloc(sizeof(ecl_build_data_t))))
	return ecl_make_error(env, CL_OUT_OF_RESOURCES);  // enomem?

    if (!(bp->r_env = enif_alloc_env())) {
	enif_free(bp);
	return ecl_make_error(env, CL_OUT_OF_RESOURCES);  // enomem?
    }
    ref = enif_make_ref(env);
    (void) enif_self(env, &bp->sender);
    bp->ref    = enif_make_copy(bp->r_env, ref);
    bp->program = o_program;
    bp->s_env = env;
    bp->tid = enif_thread_self();
    enif_keep_resource(o_program);    // keep while operation is running

    err = clBuildProgram(o_program->program,
			 num_devices,
			 device_list,
			 (const char*) options,
			 ecl_build_notify,
			 bp);
    DBG("ecl_async_build_program: err=%d user_data=%p", err, bp);

    if ((err==CL_SUCCESS) ||
	// This should not be returned, it is not according to spec!!!!
	(err==CL_BUILD_PROGRAM_FAILURE))
	return enif_make_tuple2(env, ATOM(ok), ref);
    else { 
        enif_free_env(bp->r_env);
	enif_release_resource(bp->program);
	enif_free(bp);
	return ecl_make_error(env, err);
    }
}

static ERL_NIF_TERM ecl_unload_compiler(ErlNifEnv* env, int argc, 
					const ERL_NIF_TERM argv[])
{
    cl_int err;
    UNUSED(argc);
    UNUSED(argv);

    err = clUnloadCompiler();
    if (err)
	return ecl_make_error(env, err);
    return ATOM(ok);    
}

// Special util to extract program binaries
static ERL_NIF_TERM make_program_binaries(ErlNifEnv* env, cl_program program)
{
    cl_int err;
    ERL_NIF_TERM list;
    size_t size[MAX_DEVICES];
    ErlNifBinary binary[MAX_DEVICES];
    unsigned char* data[MAX_DEVICES];
    size_t returned_size;
    cl_uint num_devices;
    int i;

    if ((err = clGetProgramInfo(program,
				CL_PROGRAM_NUM_DEVICES,
				sizeof(num_devices),
				&num_devices,
				&returned_size)))
	return ecl_make_error(env, err);

    memset(size, 0,     sizeof(size));
    memset(binary, 0,   sizeof(binary));

    if ((err = clGetProgramInfo(program,
				CL_PROGRAM_BINARY_SIZES,
				sizeof(size),
				size,
				&returned_size)))
	return ecl_make_error(env, err);
    if (returned_size != sizeof(size_t)*num_devices)
	return ecl_make_error(env, CL_INVALID_VALUE);
    i = 0;
    while (i < (int) num_devices) {
	if (!enif_alloc_binary(size[i], &binary[i])) {
	    err = CL_OUT_OF_HOST_MEMORY;
	    goto cleanup;
	}
	data[i] = binary[i].data;
	i++;
    }
    if ((err = clGetProgramInfo(program,
				CL_PROGRAM_BINARIES,
				sizeof(unsigned char*)*num_devices,
				data,
				&returned_size)))
	goto cleanup;

    list = enif_make_list(env, 0);
    for (i = num_devices-1; i >= 0; i--) {
	ERL_NIF_TERM elem = enif_make_binary(env, &binary[i]);
	list = enif_make_list_cell(env, elem, list);
    }
    return enif_make_tuple2(env, ATOM(ok), list);

cleanup:
    while(i > 0) {
	i--;
	enif_release_binary(&binary[i]);
    }
    return ecl_make_error(env, err);
}

static ERL_NIF_TERM ecl_get_program_info(ErlNifEnv* env, int argc, 
					 const ERL_NIF_TERM argv[])
{
    ecl_object_t* o_program;
    UNUSED(argc);

    if (!get_ecl_object(env, argv[0], &program_r, false, &o_program))
	return enif_make_badarg(env);

    if (argv[1] == ATOM(binaries))
	return make_program_binaries(env, o_program->program);
    else
	return make_object_info(env, argv[1], o_program,
				(info_fn_t*) clGetProgramInfo,
				program_info,
				sizeof_array(program_info));
}

static ERL_NIF_TERM ecl_get_program_build_info(ErlNifEnv* env, int argc, 
					       const ERL_NIF_TERM argv[])
{
    ecl_object_t* o_program;
    ecl_object_t* o_device;
    UNUSED(argc);

    if (!get_ecl_object(env, argv[0], &program_r, false, &o_program))
	return enif_make_badarg(env);
    if (!get_ecl_object(env, argv[1], &device_r, false, &o_device))
	return enif_make_badarg(env);
    return make_object_info2(env, argv[2], o_program, o_device, 
			     (info2_fn_t*) clGetProgramBuildInfo,
			     build_info,
			     sizeof_array(build_info));
}


static ERL_NIF_TERM ecl_create_kernel(ErlNifEnv* env, int argc, 
				      const ERL_NIF_TERM argv[])
{
    ecl_object_t* o_program;
    cl_kernel kernel;
    char kernel_name[MAX_KERNEL_NAME];
    cl_int err;
    UNUSED(argc);

    if (!get_ecl_object(env, argv[0], &program_r, false, &o_program))
	return enif_make_badarg(env);
    if (!enif_get_string(env, argv[1], kernel_name, sizeof(kernel_name),
			 ERL_NIF_LATIN1))
	return enif_make_badarg(env);

    kernel = clCreateKernel(o_program->program,kernel_name, &err);
    if (!err) {
	ERL_NIF_TERM t;
	t = ecl_make_kernel(env, kernel, o_program);
	return enif_make_tuple2(env, ATOM(ok), t);
    }
    return ecl_make_error(env, err);
}


//
// @spec create_kernels_in_program(Program::cl_program()) ->
//    {'ok', [cl_kernel()]} | {'error', cl_error()}
//
static ERL_NIF_TERM ecl_create_kernels_in_program(ErlNifEnv* env, int argc, 
						  const ERL_NIF_TERM argv[])
{
    ecl_object_t* o_program;
    ERL_NIF_TERM kernv[MAX_KERNELS];
    ERL_NIF_TERM kernel_list;
    cl_kernel kernel[MAX_KERNELS];
    cl_uint num_kernels_ret;
    cl_uint i;
    cl_int err;
    UNUSED(argc);

    if (!get_ecl_object(env, argv[0], &program_r, false, &o_program))
	return enif_make_badarg(env);

    err = clCreateKernelsInProgram(o_program->program,
				   MAX_KERNELS,
				   kernel,
				   &num_kernels_ret);
    if (err)
	return ecl_make_error(env, err);
    for (i = 0; i < num_kernels_ret; i++) {
	// FIXME: handle out of memory
	kernv[i] = ecl_make_kernel(env, kernel[i], o_program);
    }
    kernel_list = enif_make_list_from_array(env, kernv, num_kernels_ret);
    return enif_make_tuple2(env, ATOM(ok), kernel_list);
}


//
// cl:set_kernel_arg(Kernel::cl_kernel(), Index::non_neg_integer(),
//                   Argument::cl_kernel_arg()) -> 
// {Type,Value}
// {'size',Value}
// {ecl_object,Handle,<<Res>>}   object (special for sampler)
// integer()   ==  {'int', Value}
// float()     ==  {'float', Value}
// list        ==  Raw data
// binary      ==  Raw data
//
static ERL_NIF_TERM ecl_set_kernel_arg(ErlNifEnv* env, int argc, 
				       const ERL_NIF_TERM argv[])
{
    ecl_kernel_t* o_kernel;
    unsigned char arg_buf[16*sizeof(double)]; // vector type buffer
    cl_uint arg_index;
    size_t  arg_size;
    void*   arg_value;
    const ERL_NIF_TERM* array;
    double   fval;
    int      ival;
    long     lval;
    unsigned long luval;
    ErlNifUInt64 u64val;
    ErlNifSInt64 i64val;
    ErlNifBinary bval;
    cl_int   int_arg;
    cl_float float_arg;
    void*    ptr_arg = 0;
    int      arity;
    cl_int   err;
    int      arg_type = KERNEL_ARG_OTHER;
    UNUSED(argc);

    if (!get_ecl_object(env,argv[0],&kernel_r,false,(ecl_object_t**)&o_kernel))
	return enif_make_badarg(env);
    if (!enif_get_uint(env, argv[1], &arg_index))
	return enif_make_badarg(env);
    if (enif_get_tuple(env, argv[2], &arity, &array)) {
	if (arity == 3) {
	    if (array[0] == ATOM(mem_t)) {
		if (!get_object(env,argv[2],&mem_r,true,&ptr_arg))
		    return enif_make_badarg(env);
		arg_type = KERNEL_ARG_MEM;
		arg_value = &ptr_arg;
		arg_size = sizeof(cl_mem);
		goto do_kernel_arg;
	    }
	    else if (array[0] == ATOM(sampler_t)) {
		if (!get_object(env,argv[2],&sampler_r,false,&ptr_arg))
		    return enif_make_badarg(env);
		arg_type = KERNEL_ARG_SAMPLER;
		arg_value = &ptr_arg;
		arg_size = sizeof(cl_sampler);
		goto do_kernel_arg;
	    }
	    return enif_make_badarg(env);
	}
	else if (arity == 2) {
	    cl_uint typen;
	    ocl_type_t base_type;
	    size_t     base_size;
	    int       vec_size;
	    int value_arity;
	    const ERL_NIF_TERM* values;
	    unsigned char* ptr = arg_buf;
	    int i;

	    if (!get_enum(env, array[0], &typen, kv_cl_type))
		return enif_make_badarg(env);
	    vec_size = typen >> 16;
	    base_type = typen & 0xFFFF;
	    base_size = ecl_sizeof(base_type);
	    if ((vec_size == 1) && !enif_is_tuple(env, array[1])) {
		value_arity = 1;
		values = &array[1];
	    }
	    else if (!enif_get_tuple(env, array[1], &value_arity, &values))
		return enif_make_badarg(env);
	    if (value_arity != vec_size)
		return enif_make_badarg(env);
	    for (i = 0; i < vec_size; i++) {
		switch(base_type) {
		case OCL_CHAR:
		    if (!enif_get_long(env, values[i], &lval))
			return enif_make_badarg(env);
		    *((cl_char*)ptr) = lval;
		    break;
		case OCL_UCHAR:
		    if (!enif_get_ulong(env, values[i], &luval))
			return enif_make_badarg(env);
		    *((cl_uchar*)ptr) = luval;
		    break;
		case OCL_SHORT:
		    if (!enif_get_long(env, values[i], &lval))
			return enif_make_badarg(env);
		    *((cl_short*)ptr) = lval;
		    break;
		case OCL_USHORT:
		    if (!enif_get_ulong(env, values[i], &luval))
			return enif_make_badarg(env);
		    *((cl_ushort*)ptr) = luval;
		    break;
		case OCL_INT:
		    if (!enif_get_long(env, values[i], &lval))
			return enif_make_badarg(env);
		    *((cl_int*)ptr) = lval;
		    break;
		case OCL_UINT:
		    if (!enif_get_ulong(env, values[i], &luval))
			return enif_make_badarg(env);
		    *((cl_uint*)ptr) = luval;
		    break;
		case OCL_LONG:
		    if (!enif_get_int64(env, values[i], &i64val))
			return enif_make_badarg(env);
		    *((cl_long*)ptr) = i64val;
		    break;
		case OCL_ULONG:
		    if (!enif_get_uint64(env, values[i], &u64val))
			return enif_make_badarg(env);
		    *((cl_ulong*)ptr) = u64val;
		    break;
		case OCL_HALF:
		    if (!enif_get_ulong(env, values[i], &luval))
			return enif_make_badarg(env);
		    *((cl_half*)ptr) = luval;
		    break;
		case OCL_FLOAT:
		    if (!enif_get_double(env, values[i], &fval))
			return enif_make_badarg(env);
		    *((cl_float*)ptr) = fval;
		    break;

		case OCL_DOUBLE:
		    if (!enif_get_double(env, values[i], &fval))
			return enif_make_badarg(env);
		    *((cl_double*)ptr) = fval;
		    break;
		case OCL_SIZE:
		    if (!enif_get_ulong(env, values[i], &luval))
			return enif_make_badarg(env);
		    *((size_t*)ptr) = luval;
		    break;
		case OCL_BOOL:
		case OCL_STRING:
		case OCL_ENUM:
		case OCL_BITFIELD:
		case OCL_POINTER:
		case OCL_PLATFORM:
		case OCL_DEVICE: 
		case OCL_CONTEXT:
		case OCL_PROGRAM:
		case OCL_COMMAND_QUEUE:
		case OCL_IMAGE_FORMAT:
		default:
		    return enif_make_badarg(env);
		}
		ptr += base_size;
	    }
	    arg_value = arg_buf;
	    arg_size  = base_size*vec_size;
	    goto do_kernel_arg;
	}
	return enif_make_badarg(env);
    }
    else if (enif_get_int(env, argv[2], &ival)) {
	int_arg = ival;
	arg_value = &int_arg;
	arg_size = sizeof(int_arg);
	goto do_kernel_arg;
    }
    else if (enif_get_double(env, argv[2], &fval)) {
	float_arg = fval;
	arg_value = &float_arg;
	arg_size = sizeof(float_arg);
	goto do_kernel_arg;
    }
    else if (enif_inspect_iolist_as_binary(env, argv[2], &bval)) {
	// rule your own case 
	arg_value = bval.data;
	arg_size  = bval.size;
	goto do_kernel_arg;
    }
    return enif_make_badarg(env);

do_kernel_arg:
    err = clSetKernelArg(o_kernel->obj.kernel,
			 arg_index,
			 arg_size,
			 arg_value);
    if (!err) {
	set_kernel_arg(o_kernel, arg_index, arg_type, ptr_arg);
	return ATOM(ok);
    }
    return ecl_make_error(env, err);    
}

// cl:set_kernel_arg_size(Kernel::cl_kernel(), Index::non_neg_integer(),
//                        Size::non_neg_integer()) ->
//    'ok' | {'error', cl_error()}
//
// cl special to set kernel arg with size only (local mem etc)
//
static ERL_NIF_TERM ecl_set_kernel_arg_size(ErlNifEnv* env, int argc, 
					    const ERL_NIF_TERM argv[])
{
    ecl_kernel_t* o_kernel;
    cl_uint arg_index;
    size_t  arg_size;
    unsigned char* arg_value = 0;
    cl_int  err;
    UNUSED(argc);

    if (!get_ecl_object(env,argv[0],&kernel_r,false,(ecl_object_t**)&o_kernel))
	return enif_make_badarg(env);
    if (!enif_get_uint(env, argv[1], &arg_index))
	return enif_make_badarg(env);
    if (!enif_get_ulong(env, argv[1], &arg_size))
	return enif_make_badarg(env);

    err = clSetKernelArg(o_kernel->obj.kernel,
			 arg_index,
			 arg_size,
			 arg_value);
    if (!err) {
	set_kernel_arg(o_kernel, arg_index, KERNEL_ARG_OTHER, (void*) 0);
	return ATOM(ok);
    }
    return ecl_make_error(env, err);

}

static ERL_NIF_TERM ecl_get_kernel_info(ErlNifEnv* env, int argc, 
					const ERL_NIF_TERM argv[])
{
    ecl_object_t* o_kernel;
    UNUSED(argc);

    if (!get_ecl_object(env, argv[0], &kernel_r, false, &o_kernel))
	return enif_make_badarg(env);
    return make_object_info(env, argv[1], o_kernel,
			    (info_fn_t*) clGetKernelInfo, 
			    kernel_info,
			    sizeof_array(kernel_info));
}

static ERL_NIF_TERM ecl_get_kernel_workgroup_info(ErlNifEnv* env, int argc, 
						  const ERL_NIF_TERM argv[])
{
    ecl_object_t* o_kernel;
    ecl_object_t* o_device;
    UNUSED(argc);

    if (!get_ecl_object(env, argv[0], &kernel_r, false, &o_kernel))
	return enif_make_badarg(env);
    if (!get_ecl_object(env, argv[1], &device_r, false, &o_device))
	return enif_make_badarg(env);
    return make_object_info2(env, argv[2], o_kernel, o_device, 
			     (info2_fn_t*) clGetKernelWorkGroupInfo,
			     workgroup_info,
			     sizeof_array(workgroup_info));
}

//
// cl:enqueue_task(Queue::cl_queue(), Kernel::cl_kernel(),
//                   WaitList::[cl_event()]) ->
//    {'ok', cl_event()} | {'error', cl_error()}
//
static ERL_NIF_TERM ecl_enqueue_task(ErlNifEnv* env, int argc, 
				     const ERL_NIF_TERM argv[])
{
    ecl_object_t*    o_queue;
    cl_kernel        kernel;
    cl_event         wait_list[MAX_WAIT_LIST];
    size_t           num_events = MAX_WAIT_LIST;
    cl_event         event;
    cl_int           err;
    UNUSED(argc);

    if (!get_ecl_object(env, argv[0], &command_queue_r, false, &o_queue))
	return enif_make_badarg(env);
    if (!get_object(env, argv[1], &kernel_r, false,(void**)&kernel))
	return enif_make_badarg(env);
    if (!get_object_list(env, argv[2], &event_r, false,
			 (void**) wait_list, &num_events))
	return enif_make_badarg(env);
    err = clEnqueueTask(o_queue->queue, 
			kernel,
			num_events,
			num_events ? wait_list : 0,
			&event);
    if (!err) {
	ERL_NIF_TERM t;
	t = ecl_make_event(env, event, false, false, 0, 0, o_queue);
	return enif_make_tuple2(env, ATOM(ok), t);
    }
    return ecl_make_error(env, err);    
}
//
// cl:enqueue_nd_range_kernel(Queue::cl_queue(), Kernel::cl_kernel(),
//                            Global::[non_neg_integer()],
//                            Local::[non_neg_integer()],
//                            WaitList::[cl_event()]) ->
//    {'ok', cl_event()} | {'error', cl_error()}
//
static ERL_NIF_TERM ecl_enqueue_nd_range_kernel(ErlNifEnv* env, int argc, 
						const ERL_NIF_TERM argv[])
{
    ecl_object_t*    o_queue;
    cl_kernel     kernel;
    cl_event      wait_list[MAX_WAIT_LIST];
    size_t        num_events = MAX_WAIT_LIST;
    size_t        global_work_size[MAX_WORK_SIZE];
    size_t        local_work_size[MAX_WORK_SIZE];
    size_t        work_dim = MAX_WORK_SIZE;
    size_t        temp_dim = MAX_WORK_SIZE;
    cl_event      event;
    cl_int        err;
    UNUSED(argc);

    if (!get_ecl_object(env, argv[0], &command_queue_r, false, &o_queue))
	return enif_make_badarg(env);
    if (!get_object(env, argv[1], &kernel_r, false, (void**) &kernel))
	return enif_make_badarg(env);
    if (!get_ulong_list(env, argv[2], global_work_size, &work_dim))
	return enif_make_badarg(env);	
    if (!get_ulong_list(env, argv[3], local_work_size, &temp_dim))
	return enif_make_badarg(env);
    if (!get_object_list(env, argv[4], &event_r, false, 
			 (void**) wait_list, &num_events))
	return enif_make_badarg(env);
    if ((work_dim != temp_dim) || (work_dim == 0))
	return enif_make_badarg(env);

    err = clEnqueueNDRangeKernel(o_queue->queue, kernel,
				 work_dim,
				 0, // global_work_offset,
				 global_work_size,
				 local_work_size,
				 num_events, 
				 num_events ? wait_list : 0,
				 &event);
    if (!err) {
	ERL_NIF_TERM t;
	t = ecl_make_event(env, event, false, false, 0, 0, o_queue);
	return enif_make_tuple2(env, ATOM(ok), t);
    }
    return ecl_make_error(env, err);    
}

static ERL_NIF_TERM ecl_enqueue_marker(ErlNifEnv* env, int argc, 
				       const ERL_NIF_TERM argv[])
{
    ecl_object_t*    o_queue;
    cl_event event;
    cl_int err;
    UNUSED(argc);

    if (!get_ecl_object(env, argv[0], &command_queue_r, false, &o_queue))
	return enif_make_badarg(env);
    if (!(err = clEnqueueMarker(o_queue->queue, &event))) {
	ERL_NIF_TERM t;
	t = ecl_make_event(env, event, false, false, 0, 0, o_queue);
	return enif_make_tuple2(env, ATOM(ok), t);
    }
    return ecl_make_error(env, err);
}

//
// cl:enqueue_wait_for_events(Queue::cl_queue(), WaitList::[cl_event()]) ->
//    'ok' | {'error', cl_error()}
//
static ERL_NIF_TERM ecl_enqueue_wait_for_events(ErlNifEnv* env, int argc, 
						const ERL_NIF_TERM argv[])
{
    cl_command_queue queue;
    cl_event      wait_list[MAX_WAIT_LIST];
    size_t        num_events = MAX_WAIT_LIST;
    cl_int        err;
    UNUSED(argc);

    if (!get_object(env, argv[0], &command_queue_r, false, (void**)&queue))
	return enif_make_badarg(env);
    if (!get_object_list(env, argv[1], &event_r, false,
			 (void**) wait_list, &num_events))
	return enif_make_badarg(env);

    err = clEnqueueWaitForEvents(queue,
				 num_events,
				 num_events ? wait_list : 0);

    if (!err)
	return ATOM(ok);
    return ecl_make_error(env, err);    
}
//
// cl:enqueue_read_buffer(Queue::cl_queue(), Buffer::cl_mem(),
//                        Offset::non_neg_integer(), 
//                           Size::non_neg_integer(), 
//                           WaitList::[cl_event()]) ->
//    {'ok', cl_event()} | {'error', cl_error()}
static ERL_NIF_TERM ecl_enqueue_read_buffer(ErlNifEnv* env, int argc,
					    const ERL_NIF_TERM argv[])
{
    ecl_object_t*    o_queue;
    cl_mem           buffer;
    size_t           offset;
    size_t           size;
    cl_event         wait_list[MAX_WAIT_LIST];
    size_t           num_events = MAX_WAIT_LIST;
    cl_event         event;
    ErlNifBinary*    bin;
    cl_int           err;
    UNUSED(argc);

    if (!get_ecl_object(env, argv[0], &command_queue_r, false, &o_queue))
	return enif_make_badarg(env);
    if (!get_object(env, argv[1], &mem_r, false, (void**)&buffer))
	return enif_make_badarg(env);
    if (!enif_get_ulong(env, argv[2], &offset))
	return enif_make_badarg(env);	
    if (!enif_get_ulong(env, argv[3], &size))
	return enif_make_badarg(env);
    if (!get_object_list(env, argv[4], &event_r, false,
			 (void**) wait_list, &num_events))
	return enif_make_badarg(env);
    if (!(bin = enif_alloc(sizeof(ErlNifBinary))))
	return ecl_make_error(env, CL_OUT_OF_RESOURCES);  // enomem?
    if (!enif_alloc_binary(size, bin)) {
	enif_free(bin);
	return ecl_make_error(env, CL_OUT_OF_RESOURCES);  // enomem?	
    }
    err = clEnqueueReadBuffer(o_queue->queue, buffer,
			      CL_FALSE,
			      offset,
			      size,
			      bin->data,
			      num_events,
			      num_events ? wait_list : 0,
			      &event);
    if (!err) {
	ERL_NIF_TERM t;
	t = ecl_make_event(env, event, true, false, 0, bin, o_queue);
	return enif_make_tuple2(env, ATOM(ok), t);
    }
    else {
	enif_free(bin);
	return ecl_make_error(env, err);    
    }
}
//
// cl:enqueue_write_buffer(Queue::cl_queue(), Buffer::cl_mem(),
//                         Offset::non_neg_integer(), 
//                         Size::non_neg_integer(), 
//                         Data::binary(),
//                         WaitList::[cl_event()]) ->
//    {'ok', cl_event()} | {'error', cl_error()}
//
static ERL_NIF_TERM ecl_enqueue_write_buffer(ErlNifEnv* env, int argc, 
					     const ERL_NIF_TERM argv[])
{
    ecl_object_t*    o_queue;
    cl_mem           buffer;
    size_t           offset;
    size_t           size;
    cl_event         wait_list[MAX_WAIT_LIST];
    size_t           num_events = MAX_WAIT_LIST;
    cl_event         event;
    ErlNifBinary     bin;
    ErlNifEnv*       bin_env;
    cl_int           err;
    UNUSED(argc);

    if (!get_ecl_object(env, argv[0], &command_queue_r, false, &o_queue))
	return enif_make_badarg(env);
    if (!get_object(env, argv[1], &mem_r, false, (void**)&buffer))
	return enif_make_badarg(env);
    if (!enif_get_ulong(env, argv[2], &offset))
	return enif_make_badarg(env);
    if (!enif_get_ulong(env, argv[3], &size))
	return enif_make_badarg(env);
    if (!enif_inspect_binary(env, argv[4], &bin))
	return enif_make_badarg(env);
    if (!get_object_list(env, argv[5], &event_r, false,
			 (void**) wait_list, &num_events))
	return enif_make_badarg(env);

    // handle binary and iolist as binary
    if (bin.size < size) {   // FIXME: handle offset! 
	return enif_make_badarg(env);
    }

    // copy the binary new environment 
    if (!(bin_env = enif_alloc_env())) {  // create binary environment
	return ecl_make_error(env, CL_OUT_OF_RESOURCES);  // enomem?
    }

    (void) ecl_copy_binary(env, bin_env, &bin);

    err = clEnqueueWriteBuffer(o_queue->queue, buffer,
			       CL_FALSE,
			       offset,
			       size,
			       bin.data,
			       num_events,
			       num_events ? wait_list : 0,
			       &event);
    if (!err) {
	ERL_NIF_TERM t;
	t = ecl_make_event(env, event, false, true, bin_env, NULL, o_queue);
	return enif_make_tuple2(env, ATOM(ok), t);
    }
    else {
	enif_free_env(bin_env);	
	return ecl_make_error(env, err);
    }
}

//
// enqueue_read_image(_Queue, _Image, _Origin, _Region, _RowPitch, _SlicePitch,
//		   _WaitList) -> {'ok',Event} | {error,Error}
//
static ERL_NIF_TERM ecl_enqueue_read_image(ErlNifEnv* env, int argc, 
					   const ERL_NIF_TERM argv[])
{
    ecl_object_t*    o_queue;
    cl_mem           buffer;
    size_t           origin[3];
    size_t           region[3];
    size_t           row_pitch;
    size_t           slice_pitch;
    cl_event         wait_list[MAX_WAIT_LIST];
    size_t           num_events = MAX_WAIT_LIST;
    size_t           num_origin = 3;
    size_t           num_region = 3;
    size_t           psize;
    size_t           size;
    cl_event         event;
    ErlNifBinary*    bin;
    cl_int           err;
    UNUSED(argc);

    if (!get_ecl_object(env, argv[0], &command_queue_r, false, &o_queue))
	return enif_make_badarg(env);
    if (!get_object(env, argv[1], &mem_r, false, (void**)&buffer))
	return enif_make_badarg(env);
    origin[0] = origin[1] = origin[2] = 0;
    if (!get_ulong_list(env, argv[2], origin, &num_origin))
	return enif_make_badarg(env);
    region[0] = region[1] = region[2] = 1;
    if (!get_ulong_list(env, argv[3], region, &num_region))
	return enif_make_badarg(env);
    if (!enif_get_ulong(env, argv[4], &row_pitch))
	return enif_make_badarg(env);
    if (!enif_get_ulong(env, argv[5], &slice_pitch))
	return enif_make_badarg(env);
    if (!get_object_list(env, argv[6], &event_r, false,
			 (void**) wait_list, &num_events))
	return enif_make_badarg(env);
    if (!(bin = enif_alloc(sizeof(ErlNifBinary))))
	return ecl_make_error(env, CL_OUT_OF_RESOURCES);  // enomem?

    // calculate the read size of the image, FIXME: check error return
    clGetImageInfo(buffer, CL_IMAGE_ELEMENT_SIZE, sizeof(psize), &psize, 0);
    size = region[0]*region[1]*region[2]*psize;
    if (!enif_alloc_binary(size, bin)) {
	enif_free(bin);
	return ecl_make_error(env, CL_OUT_OF_RESOURCES);  // enomem?	
    }
    err = clEnqueueReadImage(o_queue->queue, buffer,
			     CL_FALSE,
			     origin,
			     region,
			     row_pitch,
			     slice_pitch,
			     bin->data,
			     num_events,
			     num_events ? wait_list : 0,
			     &event);
    if (!err) {
	ERL_NIF_TERM t;
	t = ecl_make_event(env, event, true, false, 0, bin, o_queue);
	return enif_make_tuple2(env, ATOM(ok), t);
    }
    else {
	enif_free(bin);
	return ecl_make_error(env, err);
    }
}

//
// enqueue_write_image(_Queue, _Image, _Origin, _Region, _RowPitch, _SlicePitch,
//		    _Data, _WaitList) ->
//
static ERL_NIF_TERM ecl_enqueue_write_image(ErlNifEnv* env, int argc, 
					    const ERL_NIF_TERM argv[])
{
    ecl_object_t*    o_queue;
    cl_mem           buffer;
    size_t           origin[3];
    size_t           region[3];
    size_t           row_pitch;
    size_t           slice_pitch;
    cl_event         wait_list[MAX_WAIT_LIST];
    size_t           num_events = MAX_WAIT_LIST;
    size_t           num_origin = 3;
    size_t           num_region = 3;
    size_t           psize;
    size_t           size;
    cl_event         event;
    ErlNifBinary     bin;
    ErlNifEnv*       bin_env;
    cl_int           err;
    UNUSED(argc);

    if (!get_ecl_object(env, argv[0], &command_queue_r, false, &o_queue))
	return enif_make_badarg(env);
    if (!get_object(env, argv[1], &mem_r, false, (void**)&buffer))
	return enif_make_badarg(env);
    origin[0] = origin[1] = origin[2] = 0;
    if (!get_ulong_list(env, argv[2], origin, &num_origin))
	return enif_make_badarg(env);
    region[0] = region[1] = region[2] = 1;
    if (!get_ulong_list(env, argv[3], region, &num_region))
	return enif_make_badarg(env);
    if (!enif_get_ulong(env, argv[4], &row_pitch))
	return enif_make_badarg(env);
    if (!enif_get_ulong(env, argv[5], &slice_pitch))
	return enif_make_badarg(env);
    if (!enif_inspect_binary(env, argv[6], &bin))
	return enif_make_badarg(env);
    if (!get_object_list(env, argv[7], &event_r, false,
			 (void**) wait_list, &num_events))
	return enif_make_badarg(env);


    // calculate the read size of the image FIXME: check error return
    clGetImageInfo(buffer, CL_IMAGE_ELEMENT_SIZE, sizeof(psize), &psize, 0);
    size = region[0]*region[1]*region[2]*psize;
    if (bin.size < size) {
	return enif_make_badarg(env);
    }
    if (!(bin_env = enif_alloc_env())) {  // create binary environment
	return ecl_make_error(env, CL_OUT_OF_RESOURCES);  // enomem?
    }
    (void) ecl_copy_binary(env, bin_env, &bin);

    err = clEnqueueWriteImage(o_queue->queue, buffer,
			      CL_FALSE,
			      origin,
			      region,
			      row_pitch,
			      slice_pitch,
			      bin.data,
			      num_events,
			      num_events ? wait_list : 0,
			      &event);
    if (!err) {
	ERL_NIF_TERM t;
	t = ecl_make_event(env, event, false, true, bin_env, NULL, o_queue);
	return enif_make_tuple2(env, ATOM(ok), t);
    }
    else {
	enif_free_env(bin_env);	
	return ecl_make_error(env, err);
    }
}

static ERL_NIF_TERM ecl_enqueue_copy_image(ErlNifEnv* env, int argc, 
					   const ERL_NIF_TERM argv[])
{
    ecl_object_t*    o_queue;
    cl_mem           src_image;
    cl_mem           dst_image;
    size_t           src_origin[3];
    size_t           dst_origin[3];
    size_t           region[3];
    cl_event         wait_list[MAX_WAIT_LIST];
    size_t           num_events = MAX_WAIT_LIST;
    size_t           num_src_origin = 3;
    size_t           num_dst_origin = 3;
    size_t           num_region = 3;
    cl_event         event;
    cl_int           err;
    UNUSED(argc);

    if (!get_ecl_object(env, argv[0], &command_queue_r, false, &o_queue))
	return enif_make_badarg(env);
    if (!get_object(env, argv[1], &mem_r, false, (void**)&src_image))
	return enif_make_badarg(env);
    if (!get_object(env, argv[2], &mem_r, false, (void**)&dst_image))
	return enif_make_badarg(env);
    src_origin[0] = src_origin[1] = src_origin[2] = 0;
    if (!get_ulong_list(env, argv[3], src_origin, &num_src_origin))
	return enif_make_badarg(env);
    dst_origin[0] = dst_origin[1] = dst_origin[2] = 0;
    if (!get_ulong_list(env, argv[4], dst_origin, &num_dst_origin))
	return enif_make_badarg(env);
    region[0] = region[1] = region[2] = 1;
    if (!get_ulong_list(env, argv[5], region, &num_region))
	return enif_make_badarg(env);
    if (!get_object_list(env, argv[6], &event_r, false,
			 (void**) wait_list, &num_events))
	return enif_make_badarg(env);
    err = clEnqueueCopyImage(o_queue->queue, src_image, dst_image,
			     src_origin,
			     dst_origin,
			     region,
			     num_events,
			     num_events ? wait_list : 0,
			     &event);
    if (!err) {
	ERL_NIF_TERM t;
	t = ecl_make_event(env, event, false, false, 0, 0, o_queue);
	return enif_make_tuple2(env, ATOM(ok), t);
    }
    return ecl_make_error(env, err);    
}

// cl:enqueue_copy_image_to_buffer(_Queue, _SrcImage, _DstBuffer, 
//                                 _Origin, _Region,
//			           _DstOffset, _WaitList) ->
static ERL_NIF_TERM ecl_enqueue_copy_image_to_buffer(ErlNifEnv* env, int argc, 
						     const ERL_NIF_TERM argv[])
{
    ecl_object_t*    o_queue;
    cl_mem           src_image;
    cl_mem           dst_buffer;
    size_t           origin[3];
    size_t           region[3];
    size_t           dst_offset;
    cl_event         wait_list[MAX_WAIT_LIST];
    size_t           num_events = MAX_WAIT_LIST;
    size_t           num_src_origin = 3;
    size_t           num_region = 3;
    cl_event         event;
    cl_int           err;
    UNUSED(argc);

    if (!get_ecl_object(env, argv[0], &command_queue_r, false, &o_queue))
	return enif_make_badarg(env);
    if (!get_object(env, argv[1], &mem_r, false, (void**)&src_image))
	return enif_make_badarg(env);
    if (!get_object(env, argv[2], &mem_r, false, (void**)&dst_buffer))
	return enif_make_badarg(env);
    origin[0] =  origin[1] = origin[2] = 0;
    if (!get_ulong_list(env, argv[3], origin, &num_src_origin))
	return enif_make_badarg(env);
    region[0] = region[1] = region[2] = 1;
    if (!get_ulong_list(env, argv[4], region, &num_region))
	return enif_make_badarg(env);
    if (!enif_get_ulong(env, argv[5], &dst_offset))
	return enif_make_badarg(env);
    if (!get_object_list(env, argv[6], &event_r, false,
			 (void**) wait_list, &num_events))
	return enif_make_badarg(env);
    err = clEnqueueCopyImageToBuffer(o_queue->queue, 
				     src_image,
				     dst_buffer,
				     origin,
				     region,
				     dst_offset,
				     num_events,
				     num_events ? wait_list : 0,
				     &event);
    if (!err) {
	ERL_NIF_TERM t;
	t = ecl_make_event(env, event, false, false, 0, 0, o_queue);
	return enif_make_tuple2(env, ATOM(ok), t);
    }
    return ecl_make_error(env, err);    
}
//
// cl:enqueue_copy_buffer_to_image(_Queue, _SrcBuffer, _DstImage,
//                                  _SrcOffset, _DstOrigin, 
//                                _Region, _WaitList) ->
//
static ERL_NIF_TERM ecl_enqueue_copy_buffer_to_image(ErlNifEnv* env, int argc, 
						     const ERL_NIF_TERM argv[])
{
    ecl_object_t*    o_queue;
    cl_mem           src_buffer;
    cl_mem           dst_image;
    size_t           src_offset;
    size_t           origin[3];
    size_t           region[3];
    cl_event         wait_list[MAX_WAIT_LIST];
    size_t           num_events = MAX_WAIT_LIST;
    size_t           num_src_origin = 3;
    size_t           num_region = 3;
    cl_event         event;
    cl_int           err;
    UNUSED(argc);

    if (!get_ecl_object(env, argv[0], &command_queue_r, false, &o_queue))
	return enif_make_badarg(env);
    if (!get_object(env, argv[1], &mem_r, false, (void**)&src_buffer))
	return enif_make_badarg(env);
    if (!get_object(env, argv[2], &mem_r, false, (void**)&dst_image))
	return enif_make_badarg(env);
    if (!enif_get_ulong(env, argv[3], &src_offset))
	return enif_make_badarg(env);
    origin[0] =  origin[1] = origin[2] = 0;
    if (!get_ulong_list(env, argv[4], origin, &num_src_origin))
	return enif_make_badarg(env);
    region[0] = region[1] = region[2] = 1;
    if (!get_ulong_list(env, argv[5], region, &num_region))
	return enif_make_badarg(env);
    if (!get_object_list(env, argv[6], &event_r, false,
			 (void**) wait_list, &num_events))
	return enif_make_badarg(env);
    err = clEnqueueCopyBufferToImage(o_queue->queue, 
				     src_buffer,
				     dst_image,
				     src_offset,
				     origin,
				     region,
				     num_events,
				     num_events ? wait_list : 0,
				     &event);
    if (!err) {
	ERL_NIF_TERM t;
	t = ecl_make_event(env, event, false, false, 0, 0, o_queue);
	return enif_make_tuple2(env, ATOM(ok), t);
    }
    return ecl_make_error(env, err);    
}

static ERL_NIF_TERM ecl_enqueue_map_buffer(ErlNifEnv* env, int argc, 
					   const ERL_NIF_TERM argv[])
{
    ecl_object_t*    o_queue;
    cl_mem           buffer;
    cl_map_flags     map_flags;
    size_t           offset;
    size_t           size;
    cl_event         wait_list[MAX_WAIT_LIST];
    size_t           num_events = MAX_WAIT_LIST;
    cl_event         event;
    cl_int           err;
    void*            ptr;
    UNUSED(argc);

    if (!get_ecl_object(env, argv[0], &command_queue_r, false, &o_queue))
	return enif_make_badarg(env);
    if (!get_object(env, argv[1], &mem_r, false, (void**)&buffer))
	return enif_make_badarg(env);
    if (!get_bitfields(env, argv[2], &map_flags, kv_map_flags))
	return enif_make_badarg(env);
    if (!enif_get_ulong(env, argv[3], &offset))
	return enif_make_badarg(env);
    if (!enif_get_ulong(env, argv[4], &size))
	return enif_make_badarg(env);
    if (!get_object_list(env, argv[5], &event_r, false,
			 (void**) wait_list, &num_events))
	return enif_make_badarg(env);

    ptr = clEnqueueMapBuffer(o_queue->queue,
			     buffer,
			     CL_FALSE,
			     map_flags,
			     offset,
			     size,
			     num_events,
			     num_events ? wait_list : 0,
			     &event,
			     &err);
    if (!err) {
	ERL_NIF_TERM t;
	// FIXME: how should we handle ptr????
	t = ecl_make_event(env, event, false, false, 0, 0, o_queue);
	return enif_make_tuple2(env, ATOM(ok), t);
    }
    return ecl_make_error(env, err);
}

//
// enqueue_map_image(_Queue, _Image, _MapFlags, _Origin, _Region, _WaitList) ->
//
static ERL_NIF_TERM ecl_enqueue_map_image(ErlNifEnv* env, int argc, 
					  const ERL_NIF_TERM argv[])
{
    ecl_object_t*    o_queue;
    cl_mem           image;
    cl_map_flags     map_flags;
    size_t           origin[3];
    size_t           region[3];
    size_t           row_pitch;
    size_t           slice_pitch;
    cl_event         wait_list[MAX_WAIT_LIST];
    size_t           num_events = MAX_WAIT_LIST;
    size_t           num_origin = 3;
    size_t           num_region = 3;
    cl_event         event;
    cl_int           err;
    void*            ptr;
    UNUSED(argc);

    if (!get_ecl_object(env, argv[0], &command_queue_r, false, &o_queue))
	return enif_make_badarg(env);
    if (!get_object(env, argv[1], &mem_r, false, (void**)&image))
	return enif_make_badarg(env);
    if (!get_bitfields(env, argv[2], &map_flags, kv_map_flags))
	return enif_make_badarg(env);
    origin[0] = origin[1] = origin[2] = 0;
    if (!get_ulong_list(env, argv[3], origin, &num_origin))
	return enif_make_badarg(env);
    region[0] = region[1] = region[2] = 1;
    if (!get_ulong_list(env, argv[4], region, &num_region))
	return enif_make_badarg(env);
    if (!get_object_list(env, argv[5], &event_r, false,
			 (void**) wait_list, &num_events))
	return enif_make_badarg(env);

    ptr = clEnqueueMapImage(o_queue->queue,
			    image,
			    CL_FALSE,
			    map_flags,
			    origin,
			    region,
			    &row_pitch,
			    &slice_pitch,
			    num_events,
			    num_events ? wait_list : 0,
			    &event,
			    &err);
    if (!err) {
	ERL_NIF_TERM t;
	// FIXME: send binary+event to event thread
	t = ecl_make_event(env, event, false, false, 0, 0, o_queue);
	return enif_make_tuple2(env, ATOM(ok), t);
    }
    return ecl_make_error(env, err);    
}

//
//  enqueue_unmap_mem_object(_Queue, _Mem, _WaitList) ->    
//    
//
static ERL_NIF_TERM ecl_enqueue_unmap_mem_object(ErlNifEnv* env, int argc, 
						 const ERL_NIF_TERM argv[])
{
    ecl_object_t*    o_queue;
    cl_mem           memobj;
    cl_event         wait_list[MAX_WAIT_LIST];
    size_t           num_events = MAX_WAIT_LIST;
    cl_event         event;
    void* mapped_ptr;
    cl_int err;
    UNUSED(argc);

    if (!get_ecl_object(env, argv[0], &command_queue_r, false, &o_queue))
	return enif_make_badarg(env);
    if (!get_object(env, argv[1], &mem_r, false, (void**)&memobj))
	return enif_make_badarg(env);
    if (!get_object_list(env, argv[3], &event_r, false,
			 (void**) wait_list, &num_events))
	return enif_make_badarg(env);
    mapped_ptr = 0;  // FIXME!!!!
    
    err = clEnqueueUnmapMemObject(o_queue->queue, memobj,
				  mapped_ptr,
				  num_events,
				  num_events ? wait_list : 0,
				  &event);
    if (!err) {
	ERL_NIF_TERM t;
	t = ecl_make_event(env, event, false, false, 0, 0, o_queue);
	return enif_make_tuple2(env, ATOM(ok), t);
    }
    return ecl_make_error(env, err);    
}

static ERL_NIF_TERM ecl_enqueue_barrier(ErlNifEnv* env, int argc, 
					const ERL_NIF_TERM argv[])
{
    cl_command_queue queue;
    cl_int           err;
    UNUSED(argc);

    if (!get_object(env, argv[0], &command_queue_r, false,(void**)&queue))
	return enif_make_badarg(env);
    if (!(err = clEnqueueBarrier(queue))) {
	return ATOM(ok);
    }
    return ecl_make_error(env, err);    
}

//
// cl:async_flush(Queue::cl_queue()) -> reference()
//
static ERL_NIF_TERM ecl_async_flush(ErlNifEnv* env, int argc, 
				    const ERL_NIF_TERM argv[])
{
    ecl_object_t* o_queue;
    ecl_context_t* o_context;
    ecl_message_t m;
    ERL_NIF_TERM ref;
    UNUSED(argc);

    if (!get_ecl_object(env, argv[0], &command_queue_r, false, &o_queue))
	return enif_make_badarg(env);
    if (!(o_context = (ecl_context_t*) o_queue->parent)) // must have context
	return enif_make_badarg(env);
    if (!(m.env = enif_alloc_env()))
	return ecl_make_error(env, CL_OUT_OF_RESOURCES);  // enomem?
    ref = enif_make_ref(env);

    m.type   = ECL_MESSAGE_FLUSH;
    (void) enif_self(env, &m.sender);
    m.ref    = enif_make_copy(m.env, ref);
    m.queue  = o_queue;
    enif_keep_resource(o_queue);    // keep while operation is running
    ecl_message_send(o_context->thr, &m);
    return enif_make_tuple2(env, ATOM(ok), ref);
}

//
// cl:async_finish(Queue::cl_queue()) -> reference()
//
static ERL_NIF_TERM ecl_async_finish(ErlNifEnv* env, int argc, 
				     const ERL_NIF_TERM argv[])
{
    ecl_object_t* o_queue;
    ecl_context_t* o_context;
    ecl_message_t m;
    ERL_NIF_TERM ref;
    UNUSED(argc);

    if (!get_ecl_object(env, argv[0], &command_queue_r, false, &o_queue))
	return enif_make_badarg(env);
    if (!(o_context = (ecl_context_t*) o_queue->parent)) // must have context
	return enif_make_badarg(env);
    if (!(m.env = enif_alloc_env()))
	return ecl_make_error(env, CL_OUT_OF_RESOURCES);  // enomem?
    ref = enif_make_ref(env);

    m.type   =  ECL_MESSAGE_FINISH;
    (void) enif_self(env, &m.sender);
    m.ref    = enif_make_copy(m.env, ref);
    m.queue  = o_queue;
    enif_keep_resource(o_queue);   // keep while operation is running
    ecl_message_send(o_context->thr, &m);
    return enif_make_tuple2(env, ATOM(ok), ref);
}
//
// cl:async_wait_for_event(Event) -> {ok,Ref} | {error,Reason}
// async reply {cl_event, Ref, Result}
//
static ERL_NIF_TERM ecl_async_wait_for_event(ErlNifEnv* env, int argc, 
					     const ERL_NIF_TERM argv[])
{
    ecl_event_t* o_event;
    ecl_object_t* o_queue;
    ecl_context_t* o_context;
    ecl_message_t m;
    ERL_NIF_TERM ref;
    UNUSED(argc);

    if (!get_ecl_object(env, argv[0],&event_r,false,(ecl_object_t**)&o_event))
	return enif_make_badarg(env);
    if (!(o_queue = o_event->obj.parent))  // queue not found !
	return enif_make_badarg(env);
    if (!(o_context = (ecl_context_t*) o_queue->parent)) // must have context
	return enif_make_badarg(env);
    if (!(m.env = enif_alloc_env()))
	return ecl_make_error(env, CL_OUT_OF_RESOURCES);  // enomem?
    ref = enif_make_ref(env);

    m.type   = ECL_MESSAGE_WAIT_FOR_EVENT;
    (void) enif_self(env, &m.sender);
    m.ref    = enif_make_copy(m.env, ref);
    m.event  = o_event;
    enif_keep_resource(o_event);   // keep while operation is running
    ecl_message_send(o_context->thr, &m);
    return enif_make_tuple2(env, ATOM(ok), ref);
}

// return event info
static ERL_NIF_TERM ecl_get_event_info(ErlNifEnv* env, int argc, 
				       const ERL_NIF_TERM argv[])
{
    ecl_object_t* o_event;
    UNUSED(argc);

    if (!get_ecl_object(env, argv[0], &event_r, false, &o_event))
	return enif_make_badarg(env);
    return make_object_info(env, argv[1], o_event,
			    (info_fn_t*) clGetEventInfo,
			    event_info,
			    sizeof_array(event_info));
}


static int  ecl_load(ErlNifEnv* env, void** priv_data, ERL_NIF_TERM load_info)
{
    ErlNifResourceFlags tried;
    ecl_env_t* ecl;
    lhash_func_t func = { ref_hash, ref_cmp, ref_release, 0 };
    UNUSED(env);
    UNUSED(load_info);

    if (!(ecl = enif_alloc(sizeof(ecl_env_t))))
	return -1;
    if (!(ecl->ref_lock = enif_rwlock_create("ref_lock")))
	return -1;
    lhash_init(&ecl->ref, "ref", 2, &func);

    // Load atoms

    // General atoms
    LOAD_ATOM(ok);
    LOAD_ATOM(error);
    LOAD_ATOM(unknown);
    LOAD_ATOM(undefined);
    LOAD_ATOM(true);
    LOAD_ATOM(false);

    // async messages
    LOAD_ATOM(cl_async);
    LOAD_ATOM(cl_event);
    
    // Type names
    LOAD_ATOM(platform_t);
    LOAD_ATOM(device_t);
    LOAD_ATOM(context_t);
    LOAD_ATOM(command_queue_t);
    LOAD_ATOM(mem_t);
    LOAD_ATOM(sampler_t);
    LOAD_ATOM(program_t);
    LOAD_ATOM(kernel_t);
    LOAD_ATOM(event_t);

    LOAD_ATOM(char);
    LOAD_ATOM(char2);
    LOAD_ATOM(char4);
    LOAD_ATOM(char8);
    LOAD_ATOM(char16);

    LOAD_ATOM(uchar);
    LOAD_ATOM(uchar2);
    LOAD_ATOM(uchar4);
    LOAD_ATOM(uchar8);
    LOAD_ATOM(uchar16);

    LOAD_ATOM(short);
    LOAD_ATOM(short2);
    LOAD_ATOM(short4);
    LOAD_ATOM(short8);
    LOAD_ATOM(short16);

    LOAD_ATOM(ushort);
    LOAD_ATOM(ushort2);
    LOAD_ATOM(ushort4);
    LOAD_ATOM(ushort8);
    LOAD_ATOM(ushort16);

    LOAD_ATOM(int);
    LOAD_ATOM(int2);
    LOAD_ATOM(int4);
    LOAD_ATOM(int8);
    LOAD_ATOM(int16);

    LOAD_ATOM(uint);
    LOAD_ATOM(uint2);
    LOAD_ATOM(uint4);
    LOAD_ATOM(uint8);
    LOAD_ATOM(uint16);

    LOAD_ATOM(long);
    LOAD_ATOM(long2);
    LOAD_ATOM(long4);
    LOAD_ATOM(long8);
    LOAD_ATOM(long16);

    LOAD_ATOM(ulong);
    LOAD_ATOM(ulong2);
    LOAD_ATOM(ulong4);
    LOAD_ATOM(ulong8);
    LOAD_ATOM(ulong16);

    LOAD_ATOM(half);

    LOAD_ATOM(float);
    LOAD_ATOM(float2);
    LOAD_ATOM(float4);
    LOAD_ATOM(float8);
    LOAD_ATOM(float16);

    LOAD_ATOM(double);
    LOAD_ATOM(double2);
    LOAD_ATOM(double4);
    LOAD_ATOM(double8);
    LOAD_ATOM(double16);

    // channel type
    LOAD_ATOM(snorm_int8);
    LOAD_ATOM(snorm_int16);
    LOAD_ATOM(unorm_int8);
    LOAD_ATOM(unorm_int16);
    LOAD_ATOM(unorm_short_565);
    LOAD_ATOM(unorm_short_555);
    LOAD_ATOM(unorm_int_101010);
    LOAD_ATOM(signed_int8);
    LOAD_ATOM(signed_int16);
    LOAD_ATOM(signed_int32);
    LOAD_ATOM(unsigned_int8);
    LOAD_ATOM(unsigned_int16);
    LOAD_ATOM(unsigned_int32);
    LOAD_ATOM(half_float);

    // channel order
    LOAD_ATOM(r);
    LOAD_ATOM(a);
    LOAD_ATOM(rg);
    LOAD_ATOM(ra);
    LOAD_ATOM(rgb);
    LOAD_ATOM(rgba);
    LOAD_ATOM(bgra);
    LOAD_ATOM(argb);
    LOAD_ATOM(intensity);
    LOAD_ATOM(luminance);
    LOAD_ATOM(rx);
    LOAD_ATOM(rgx);
    LOAD_ATOM(rgbx);

    // Load options & flags

    // Device info
    LOAD_ATOM(type);
    LOAD_ATOM(vendor_id);
    LOAD_ATOM(max_compute_units);
    LOAD_ATOM(max_work_item_dimensions);
    LOAD_ATOM(max_work_group_size);
    LOAD_ATOM(max_work_item_sizes);
    LOAD_ATOM(preferred_vector_width_char);
    LOAD_ATOM(preferred_vector_width_short);
    LOAD_ATOM(preferred_vector_width_int);
    LOAD_ATOM(preferred_vector_width_long);
    LOAD_ATOM(preferred_vector_width_float);
    LOAD_ATOM(preferred_vector_width_double);
    LOAD_ATOM(max_clock_frequency);
    LOAD_ATOM(address_bits);
    LOAD_ATOM(max_read_image_args);
    LOAD_ATOM(max_write_image_args);
    LOAD_ATOM(max_mem_alloc_size);
    LOAD_ATOM(image2d_max_width);
    LOAD_ATOM(image2d_max_height);
    LOAD_ATOM(image3d_max_width);
    LOAD_ATOM(image3d_max_height);
    LOAD_ATOM(image3d_max_depth);
    LOAD_ATOM(image_support);
    LOAD_ATOM(max_parameter_size);
    LOAD_ATOM(max_samplers);
    LOAD_ATOM(mem_base_addr_align);
    LOAD_ATOM(min_data_type_align_size);
    LOAD_ATOM(single_fp_config);
    LOAD_ATOM(global_mem_cache_type);
    LOAD_ATOM(global_mem_cacheline_size);
    LOAD_ATOM(global_mem_cache_size);
    LOAD_ATOM(global_mem_size);
    LOAD_ATOM(max_constant_buffer_size);
    LOAD_ATOM(max_constant_args);
    LOAD_ATOM(local_mem_type);
    LOAD_ATOM(local_mem_size);
    LOAD_ATOM(error_correction_support);
    LOAD_ATOM(profiling_timer_resolution);
    LOAD_ATOM(endian_little);
    LOAD_ATOM(available);
    LOAD_ATOM(compiler_available);
    LOAD_ATOM(execution_capabilities);
    LOAD_ATOM(queue_properties);
    LOAD_ATOM(name);
    LOAD_ATOM(vendor);
    LOAD_ATOM(driver_version);
    LOAD_ATOM(profile);
    LOAD_ATOM(version);
    LOAD_ATOM(extensions);
    LOAD_ATOM(platform);

     // Platform info
    LOAD_ATOM(profile);
    LOAD_ATOM(version);
    LOAD_ATOM(name);
    LOAD_ATOM(vendor);
    LOAD_ATOM(extensions);

     // Context info
    LOAD_ATOM(reference_count);
    LOAD_ATOM(devices);
    LOAD_ATOM(properties);

    // Queue info
    LOAD_ATOM(context);
    LOAD_ATOM(num_devices);
    LOAD_ATOM(device);
    LOAD_ATOM(reference_count);
    LOAD_ATOM(properties);

    // Mem info
    LOAD_ATOM(object_type);
    LOAD_ATOM(flags);
    LOAD_ATOM(size);
    LOAD_ATOM(host_ptr);
    LOAD_ATOM(map_count);
    LOAD_ATOM(reference_count); 
    LOAD_ATOM(context);

    // Image info
    LOAD_ATOM(format);
    LOAD_ATOM(element_size);
    LOAD_ATOM(row_pitch);
    LOAD_ATOM(slice_pitch);
    LOAD_ATOM(width);
    LOAD_ATOM(height);
    LOAD_ATOM(depth);

    // Sampler info
    LOAD_ATOM(reference_count);
    LOAD_ATOM(context);
    LOAD_ATOM(normalized_coords);
    LOAD_ATOM(addressing_mode);
    LOAD_ATOM(filter_mode);

    // Program info
    LOAD_ATOM(reference_count);
    LOAD_ATOM(context);
    LOAD_ATOM(num_decices);
    LOAD_ATOM(devices);
    LOAD_ATOM(source); 
    LOAD_ATOM(binary_sizes);
    LOAD_ATOM(binaries);

    // Build Info
    LOAD_ATOM(status);
    LOAD_ATOM(options);
    LOAD_ATOM(log);

    // Kernel Info
    LOAD_ATOM(function_name);
    LOAD_ATOM(num_args);
    LOAD_ATOM(reference_count);
    LOAD_ATOM(context);
    LOAD_ATOM(program);

    // Event Info
    LOAD_ATOM(command_queue);
    LOAD_ATOM(command_type);
    LOAD_ATOM(reference_count);
    LOAD_ATOM(execution_status);

    // Workgroup info
    LOAD_ATOM(work_group_size);
    LOAD_ATOM(compile_work_group_size);
    LOAD_ATOM(local_mem_size);

    // Error codes
    LOAD_ATOM(device_not_found);
    LOAD_ATOM(device_not_available);
    LOAD_ATOM(compiler_not_available);
    LOAD_ATOM(mem_object_allocation_failure);
    LOAD_ATOM(out_of_resources);
    LOAD_ATOM(out_of_host_memory);
    LOAD_ATOM(profiling_info_not_available);
    LOAD_ATOM(mem_copy_overlap);
    LOAD_ATOM(image_format_mismatch);
    LOAD_ATOM(image_format_not_supported);
    LOAD_ATOM(build_program_failure);
    LOAD_ATOM(map_failure);
    LOAD_ATOM(invalid_value);
    LOAD_ATOM(invalid_device_type);
    LOAD_ATOM(invalid_platform);
    LOAD_ATOM(invalid_device);
    LOAD_ATOM(invalid_context);
    LOAD_ATOM(invalid_queue_properties);
    LOAD_ATOM(invalid_command_queue);
    LOAD_ATOM(invalid_host_ptr);
    LOAD_ATOM(invalid_mem_object);
    LOAD_ATOM(invalid_image_format_descriptor);
    LOAD_ATOM(invalid_image_size);
    LOAD_ATOM(invalid_sampler);
    LOAD_ATOM(invalid_binary);
    LOAD_ATOM(invalid_build_options);
    LOAD_ATOM(invalid_program);
    LOAD_ATOM(invalid_program_executable);
    LOAD_ATOM(invalid_kernel_name);
    LOAD_ATOM(invalid_kernel_definition);
    LOAD_ATOM(invalid_kernel);
    LOAD_ATOM(invalid_arg_index);
    LOAD_ATOM(invalid_arg_value);
    LOAD_ATOM(invalid_arg_size);
    LOAD_ATOM(invalid_kernel_args);
    LOAD_ATOM(invalid_work_dimension);
    LOAD_ATOM(invalid_work_group_size);
    LOAD_ATOM(invalid_work_item_size);
    LOAD_ATOM(invalid_global_offset);
    LOAD_ATOM(invalid_event_wait_list);
    LOAD_ATOM(invalid_event);
    LOAD_ATOM(invalid_operation);
    LOAD_ATOM(invalid_gl_object);
    LOAD_ATOM(invalid_buffer_size);
    LOAD_ATOM(invalid_mip_level);

    // cl_device_type
    LOAD_ATOM(all);
    LOAD_ATOM(default);
    LOAD_ATOM(cpu);
    LOAD_ATOM(gpu);
    LOAD_ATOM(accelerator);

    // fp_config
    LOAD_ATOM(denorm);
    LOAD_ATOM(inf_nan);
    LOAD_ATOM(round_to_nearest);
    LOAD_ATOM(round_to_zero);
    LOAD_ATOM(round_to_inf);
    LOAD_ATOM(fma);

    // mem_cache_type
    LOAD_ATOM(none);
    LOAD_ATOM(read_only);
    LOAD_ATOM(read_write);

    // local_mem_type
    LOAD_ATOM(local);
    LOAD_ATOM(global);

    // exec capability
    LOAD_ATOM(kernel);
    LOAD_ATOM(native_kernel);

    // command_queue_properties
    LOAD_ATOM(out_of_order_exec_mode_enable);
    LOAD_ATOM(profiling_enable);

    // mem_flags
    LOAD_ATOM(read_write);
    LOAD_ATOM(write_only);
    LOAD_ATOM(read_only);
    LOAD_ATOM(use_host_ptr);
    LOAD_ATOM(alloc_host_ptr);
    LOAD_ATOM(copy_host_ptr);

    // mem_object_type
    LOAD_ATOM(buffer);
    LOAD_ATOM(image2d);
    LOAD_ATOM(image3d);

    // addressing_mode
    LOAD_ATOM(none);
    LOAD_ATOM(clamp_to_edge);
    LOAD_ATOM(clamp);
    LOAD_ATOM(repeat);

    // filter_mode
    LOAD_ATOM(nearest);
    LOAD_ATOM(linear);

    // map_flags
    LOAD_ATOM(read);
    LOAD_ATOM(write);

    // build_status
    LOAD_ATOM(success);
    LOAD_ATOM(none);
    LOAD_ATOM(error);
    LOAD_ATOM(in_progress);

    // command_type
    LOAD_ATOM(ndrange_kernel);
    LOAD_ATOM(task);
    LOAD_ATOM(native_kernel);
    LOAD_ATOM(read_buffer);
    LOAD_ATOM(write_buffer);
    LOAD_ATOM(copy_buffer);
    LOAD_ATOM(read_image);
    LOAD_ATOM(write_image);
    LOAD_ATOM(copy_image);
    LOAD_ATOM(copy_image_to_buffer);
    LOAD_ATOM(copy_buffer_to_image);
    LOAD_ATOM(map_buffer);
    LOAD_ATOM(map_image);
    LOAD_ATOM(unmap_mem_object);
    LOAD_ATOM(marker);
    LOAD_ATOM(aquire_gl_objects);
    LOAD_ATOM(release_gl_objects);

    // execution_status
    LOAD_ATOM(complete);
    LOAD_ATOM(running);
    LOAD_ATOM(submitted);
    LOAD_ATOM(queued);

    // Create resource types
    ecl_resource_init(env, &platform_r, "platform_t", 
		      sizeof(ecl_object_t),
		      ecl_platform_dtor,
		      ERL_NIF_RT_CREATE, &tried);
    ecl_resource_init(env, &device_r, "device_t",
		      sizeof(ecl_object_t),
		      ecl_device_dtor,
		      ERL_NIF_RT_CREATE, &tried);
    ecl_resource_init(env, &context_r, "context_t",
		      sizeof(ecl_context_t),     // NOTE! specialized!
		      ecl_context_dtor,
		      ERL_NIF_RT_CREATE, &tried);
    ecl_resource_init(env, &command_queue_r, "command_queue_t",
		      sizeof(ecl_object_t),
		      ecl_queue_dtor,
		      ERL_NIF_RT_CREATE, &tried);
    ecl_resource_init(env, &mem_r, "mem_t", 
		      sizeof(ecl_object_t),
		      ecl_mem_dtor,
		      ERL_NIF_RT_CREATE, &tried);
    ecl_resource_init(env, &sampler_r, "sampler_t",
		      sizeof(ecl_object_t),
		      ecl_sampler_dtor,
		      ERL_NIF_RT_CREATE, &tried);
    ecl_resource_init(env, &program_r, "program_t",
		      sizeof(ecl_object_t),
		      ecl_program_dtor,
		      ERL_NIF_RT_CREATE, &tried);
    ecl_resource_init(env, &kernel_r, "kernel_t",
		      sizeof(ecl_kernel_t),   // NOTE! specialized!
		      ecl_kernel_dtor,
		      ERL_NIF_RT_CREATE, &tried);
    ecl_resource_init(env, &event_r, "event_t",
		      sizeof(ecl_event_t),    // NOTE! specialized!
		      ecl_event_dtor,
		      ERL_NIF_RT_CREATE, &tried);
    *priv_data = ecl;
    return 0;
}

static int  ecl_reload(ErlNifEnv* env, void** priv_data, ERL_NIF_TERM load_info)
{
    UNUSED(env);
    UNUSED(load_info);
    UNUSED(priv_data);
    // FIXME
    return 0;
}

static int  ecl_upgrade(ErlNifEnv* env, void** priv_data, void** old_priv_data, 
			ERL_NIF_TERM load_info)
{
    UNUSED(env);
    UNUSED(load_info);
    // FIXME
    *priv_data = *old_priv_data;
    return 0;
}

static void ecl_unload(ErlNifEnv* env, void* priv_data)
{
    ecl_env_t* ecl = priv_data;
    UNUSED(env);

    enif_rwlock_rwlock(ecl->ref_lock);
    lhash_delete(&ecl->ref);
    enif_rwlock_rwunlock(ecl->ref_lock);

    enif_rwlock_destroy(ecl->ref_lock);
    enif_free(ecl);
}

ERL_NIF_INIT(cl, ecl_funcs, 
	     ecl_load, ecl_reload, 
	     ecl_upgrade, ecl_unload)
