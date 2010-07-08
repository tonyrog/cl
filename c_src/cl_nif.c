//
// NIF interface for OpenCL binding
//
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <errno.h>

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

#include "erl_nif.h"

#include "cbufv2.h"
#include "cl_hash.h"



ERL_NIF_TERM make_error(ErlNifEnv* env, cl_int err)
{

}


static ERL_NIF_TERM ecl_get_platform_ids(ErlNifEnv* env)
{
    cl_uint          num_platforms = 0;
    cl_platform_id   platform_id[MAX_PLATFORMS];
    cl_uint i;
    cl_int err = CL_INVALID_VALUE;

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
	    cbuf_put_list_end(&reply);
	    cbuf_put_tuple_end(&reply);    

}


static ErlNifFunc ecl_funcs[] =
{
    // Platform
    { "get_platform_ids", 0, ecl_get_platform_ids },
    { "platform_info", 0, ecl_platform_info },
    { "get_platform_info", 1, ecl_get_platform_info },
    { "get_platform_info", 2, ecl_get_platform_info },
    // Devices
    { "get_device_ids", 0,  ecl_get_device_ids },
    { "get_device_ids", 2, ecl_get_device_ids },
    { "device_info", 0, ecl_device_info },
    { "get_device_info", 1, ecl_get_device_info },
    { "get_device_info", 2, ecl_get_device_info },
    // Context
    { "create_context", 1, ecl_create_context },
    { "create_context_from_type", 1, ecl_create_context_from_type },
    { "release_context", 1, ecl_release_context },
    { "retain_context", 1, ecl_retain_context },
    { "context_info", 0, ecl_context_info },
    { "get_context_info", 1, ecl_get_context_info },
    { "get_context_info", 2, ecl_get_context_info },
    // Command queue
    { "create_queue", 3, ecl_create_queue },
    { "set_queue_property", 3, ecl_set_queue_property },
    { "release_queue", 1, ecl_release_queue },
    { "retain_queue", 1, ecl_retain_queue },
    { "queue_info", 0, ecl_queue_info },
    { "get_queue_info", 1, ecl_get_queue_info },
    { "get_queue_info", 2, ecl_get_queue_info },
    // Memory object
    { "create_buffer", 3, ecl_create_buffer },
    { "create_buffer", 4, ecl_create_buffer },
    { "release_mem_object", 1, ecl_release_mem_object },
    { "retain_mem_object", 1, ecl_retain_mem_object },
    { "get_mem_object_info", 1, ecl_get_mem_object_info },
    { "get_mem_object_info", 2, ecl_get_mem_object_info },
    // Sampler 
    { "create_sampler", 4, ecl_create_sampler },
    { "release_sampler", 1, ecl_release_sampler },
    { "retain_sampler", 1, ecl_retain_sampler },
    { "sampler_info", 0, ecl_sampler_info },
    { "get_sampler_info", 1, ecl_get_sampler_info },
    { "get_sampler_info", 2, ecl_get_sampler_info },
    // Program
    { "create_program_with_source", 2, ecl_create_program_with_source },
    { "create_program_with_binary", 3, ecl_create_program_with_binary },
    { "release_program", 1, ecl_release_program },
    { "retain_program", 1, ecl_retain_program },
    { "build_program", 3, ecl_build_program },
    { "unload_compiler", 0, ecl_unload_compiler },
    { "program_info", 0, ecl_program_info },
    { "get_program_info", 1, ecl_get_program_info },
    { "get_program_info", 2, ecl_get_program_info },
    { "program_build_info", 0, ecl_program_build_info },
    { "get_program_build_info", 2, ecl_get_program_build_info },
    { "get_program_build_info", 3, ecl_get_program_build_info },
    // Kernel
    { "create_kernel", 2, ecl_create_kernel },
    { "create_kernels_in_program", 1, ecl_create_kernels_in_program },
    { "set_kernel_arg", 3, ecl_set_kernel_arg },
    { "set_kernel_arg_size", 3, ecl_set_kernel_arg_size },
    { "encode_argument", 1, ecl_encode_argument },
    { "release_kernel", 1, ecl_release_kernel },
    { "retain_kernel", 1, ecl_retain_kernel },
    { "kernel_info", 0, ecl_kernel_info },
    { "get_kernel_info", 1, ecl_get_kernel_info },
    { "get_kernel_info", 2, ecl_get_kernel_info },
    { "kernel_workgroup_info", 0, ecl_kernel_workgroup_info },
    { "get_kernel_workgroup_info", 2, ecl_get_kernel_workgroup_info },
    { "get_kernel_workgroup_info", 3, ecl_get_kernel_workgroup_info },
    // Events
    { "enqueue_task", 3, ecl_enqueue_task },
    { "enqueue_nd_range_kernel", 5, ecl_enqueue_nd_range_kernel },
    { "enqueue_marker", 1, ecl_enqueue_marker },
    { "enqueue_wait_for_event", 2, ecl_enqueue_wait_for_event },
    { "enqueue_read_buffer", 5, ecl_enqueue_read_buffer },
    { "enqueue_write_buffer", 6, ecl_enqueue_write_buffer },
    { "enqueue_barrier", 1, ecl_enqueue_barrier },
    { "flush", 1, ecl_flush },
    { "finish", 1, ecl_finish },
    { "release_event", 1, ecl_release_event },
    { "retain_event", 1, ecl_retain_event },
    { "event_info", 0, ecl_event_info },
    { "get_event_info", 1, ecl_get_event_info },
    { "get_event_info", 2, ecl_get_event_info }
};

static int  ecl_load(ErlNifEnv*, void** priv_data, ERL_NIF_TERM load_info)
{
}

static int  ecl_reload(ErlNifEnv*, void** priv_data, ERL_NIF_TERM load_info)
{
}

static int  ecl_upgrade(ErlNifEnv*, void** priv_data, void** old_priv_data, 
			ERL_NIF_TERM load_info)
{
}

static void ecl_unload(ErlNifEnv*, void* priv_data)
{
}


ERL_NIF_INIT(cl, ecl_funcs, ecl_load, ecl_reload, ecl_upgrade, ecl_unload)



