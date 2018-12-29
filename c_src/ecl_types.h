//
// definition of types needed to implement cl functions 
// for a range of different version.
//
#ifndef __ECL_TYPES_H__
#define __ECL_TYPES_H__

//
#if !defined(CL_VERSION_1_2)
typedef struct _cl_image_desc {
    cl_mem_object_type      image_type;
    size_t                  image_width;
    size_t                  image_height;
    size_t                  image_depth;
    size_t                  image_array_size;
    size_t                  image_row_pitch;
    size_t                  image_slice_pitch;
    cl_uint                 num_mip_levels;
    cl_uint                 num_samples;
    cl_mem                  buffer;
} cl_image_desc;
#endif

// Function types t_<functionName>

typedef cl_int (CL_CALLBACK * t_clGetPlatformIDs)(cl_uint, cl_platform_id *, cl_uint *);

typedef cl_int (CL_CALLBACK * t_clGetPlatformInfo)(cl_platform_id, cl_platform_info, size_t , void *, size_t *);

typedef cl_int (CL_CALLBACK * t_clGetDeviceIDs)(cl_platform_id,
 cl_device_type, cl_uint, cl_device_id *, cl_uint *);

typedef cl_int (CL_CALLBACK * t_clGetDeviceInfo)(cl_device_id, cl_device_info, size_t, void *,
 size_t *);
 
typedef cl_int (CL_CALLBACK * t_clCreateSubDevices)(cl_device_id, const cl_device_partition_property *, cl_uint, cl_device_id *, cl_uint *);

typedef cl_int (CL_CALLBACK * t_clRetainDevice)(cl_device_id );
 
typedef cl_int (CL_CALLBACK * t_clReleaseDevice)(cl_device_id );
 

typedef cl_context (CL_CALLBACK * t_clCreateContext)(const cl_context_properties *,cl_uint,const cl_device_id *,void (CL_CALLBACK *)(const char *, const void *, size_t, void *),void *,cl_int *);

typedef cl_context (CL_CALLBACK * t_clCreateContextFromType)(const cl_context_properties *,cl_device_type,void (CL_CALLBACK *)(const char *, const void *, size_t, void *),void *,cl_int *);

typedef cl_int (CL_CALLBACK * t_clRetainContext)(cl_context );

typedef cl_int (CL_CALLBACK * t_clReleaseContext)(cl_context );

typedef cl_int (CL_CALLBACK * t_clGetContextInfo)(cl_context, cl_context_info, size_t, void *, size_t *);


typedef cl_command_queue (CL_CALLBACK * t_clCreateCommandQueue)(cl_context, cl_device_id, cl_command_queue_properties,cl_int *);

typedef cl_int (CL_CALLBACK * t_clRetainCommandQueue)(cl_command_queue );

typedef cl_int (CL_CALLBACK * t_clReleaseCommandQueue)(cl_command_queue );

typedef cl_int (CL_CALLBACK * t_clGetCommandQueueInfo)(cl_command_queue,cl_command_queue_info,size_t,void *,size_t *);


typedef cl_mem (CL_CALLBACK * t_clCreateBuffer)(cl_context,cl_mem_flags,size_t,void *,cl_int *);

typedef cl_mem (CL_CALLBACK * t_clCreateSubBuffer)(cl_mem,cl_mem_flags,cl_buffer_create_type,const void *,cl_int *);

typedef cl_mem (CL_CALLBACK * t_clCreateImage)(cl_context,cl_mem_flags,const cl_image_format *,const cl_image_desc *, void *,cl_int *);
 
typedef cl_int (CL_CALLBACK * t_clRetainMemObject)(cl_mem );

typedef cl_int (CL_CALLBACK * t_clReleaseMemObject)(cl_mem );

typedef cl_int (CL_CALLBACK * t_clGetSupportedImageFormats)(cl_context,cl_mem_flags,cl_mem_object_type,cl_uint,cl_image_format *,cl_uint *);
 
typedef cl_int (CL_CALLBACK * t_clGetMemObjectInfo)(cl_mem,cl_mem_info, size_t,void *,size_t *);

typedef cl_int (CL_CALLBACK * t_clGetImageInfo)(cl_mem,cl_image_info, size_t,void *,size_t *);

typedef cl_int (CL_CALLBACK * t_clSetMemObjectDestructorCallback)( cl_mem, void (CL_CALLBACK *)( cl_mem, void*), void *) ; 


typedef cl_sampler (CL_CALLBACK * t_clCreateSampler)(cl_context,cl_bool, cl_addressing_mode, cl_filter_mode,cl_int *);
typedef cl_int (CL_CALLBACK * t_clRetainSampler)(cl_sampler );

typedef cl_int (CL_CALLBACK * t_clReleaseSampler)(cl_sampler );

typedef cl_int (CL_CALLBACK * t_clGetSamplerInfo)(cl_sampler,cl_sampler_info,size_t,void *,size_t *);
 

typedef cl_program (CL_CALLBACK * t_clCreateProgramWithSource)(cl_context,cl_uint,const char **,const size_t *,cl_int *);

typedef cl_program (CL_CALLBACK * t_clCreateProgramWithBinary)(cl_context,cl_uint,const cl_device_id *,const size_t *,const unsigned char **,cl_int *,cl_int *);

typedef cl_program (CL_CALLBACK * t_clCreateProgramWithBuiltInKernels)(cl_context,cl_uint,const cl_device_id *,const char *,cl_int *);

typedef cl_int (CL_CALLBACK * t_clRetainProgram)(cl_program );

typedef cl_int (CL_CALLBACK * t_clReleaseProgram)(cl_program );

typedef cl_int (CL_CALLBACK * t_clBuildProgram)(cl_program,cl_uint,const cl_device_id *,const char *, void (CL_CALLBACK *)(cl_program, void *),void *);

typedef cl_int (CL_CALLBACK * t_clCompileProgram)(cl_program,cl_uint,const cl_device_id *,const char *, cl_uint,const cl_program *,const char **,void (CL_CALLBACK *)(cl_program, void *),void *);

typedef cl_program (CL_CALLBACK * t_clLinkProgram)(cl_context,cl_uint,const cl_device_id *,const char *, cl_uint,const cl_program *,void (CL_CALLBACK *)(cl_program, void *),void *,cl_int *);


typedef cl_int (CL_CALLBACK * t_clUnloadPlatformCompiler)(cl_platform_id );

typedef cl_int (CL_CALLBACK * t_clGetProgramInfo)(cl_program,cl_program_info,size_t,void *,size_t *);

typedef cl_int (CL_CALLBACK * t_clGetProgramBuildInfo)(cl_program,cl_device_id,cl_program_build_info,size_t,void *,size_t *);
 

typedef cl_kernel (CL_CALLBACK * t_clCreateKernel)(cl_program,const char *,cl_int *);

typedef cl_int (CL_CALLBACK * t_clCreateKernelsInProgram)(cl_program,cl_uint,cl_kernel *,cl_uint *);

typedef cl_int (CL_CALLBACK * t_clRetainKernel)(cl_kernel );

typedef cl_int (CL_CALLBACK * t_clReleaseKernel)(cl_kernel );

typedef cl_int (CL_CALLBACK * t_clSetKernelArg)(cl_kernel,cl_uint,size_t,const void *);

typedef cl_int (CL_CALLBACK * t_clGetKernelInfo)(cl_kernel,cl_kernel_info,size_t,void *,size_t *);

typedef cl_int (CL_CALLBACK * t_clGetKernelArgInfo)(cl_kernel,cl_uint,cl_kernel_arg_info,size_t,void *,size_t *);

typedef cl_int (CL_CALLBACK * t_clGetKernelWorkGroupInfo)(cl_kernel,cl_device_id,cl_kernel_work_group_info,size_t,void *,size_t *);


typedef cl_int (CL_CALLBACK * t_clWaitForEvents)(cl_uint,const cl_event *);

typedef cl_int (CL_CALLBACK * t_clGetEventInfo)(cl_event,cl_event_info,size_t,void *,size_t *);
 
typedef cl_event (CL_CALLBACK * t_clCreateUserEvent)(cl_context,cl_int *); 
 
typedef cl_int (CL_CALLBACK * t_clRetainEvent)(cl_event );

typedef cl_int (CL_CALLBACK * t_clReleaseEvent)(cl_event );

typedef cl_int (CL_CALLBACK * t_clSetUserEventStatus)(cl_event,cl_int );
 
typedef cl_int (CL_CALLBACK * t_clSetEventCallback)( cl_event,cl_int,void (CL_CALLBACK *)(cl_event, cl_int, void *),void *);

typedef cl_int (CL_CALLBACK * t_clGetEventProfilingInfo)(cl_event,cl_profiling_info,size_t,void *,size_t *);

typedef cl_int (CL_CALLBACK * t_clFlush)(cl_command_queue );

typedef cl_int (CL_CALLBACK * t_clFinish)(cl_command_queue );

typedef cl_int (CL_CALLBACK * t_clEnqueueReadBuffer)(cl_command_queue,cl_mem,cl_bool,size_t,size_t, void *,cl_uint,const cl_event *,cl_event *);
 
typedef cl_int (CL_CALLBACK * t_clEnqueueReadBufferRect)(cl_command_queue,cl_mem,cl_bool,const size_t *,const size_t *, const size_t *,size_t,size_t,size_t,size_t, void *,cl_uint,const cl_event *,cl_event *);
 
typedef cl_int (CL_CALLBACK * t_clEnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void *, cl_uint, const cl_event *, cl_event *);
 
typedef cl_int (CL_CALLBACK * t_clEnqueueWriteBufferRect)(cl_command_queue,cl_mem,cl_bool,const size_t *,const size_t *, const size_t *,size_t,size_t,size_t,size_t, const void *,cl_uint,const cl_event *,cl_event *);
 
typedef cl_int (CL_CALLBACK * t_clEnqueueFillBuffer)(cl_command_queue,cl_mem, const void *, size_t, size_t, size_t, cl_uint, const cl_event *, cl_event *);
 
typedef cl_int (CL_CALLBACK * t_clEnqueueCopyBuffer)(cl_command_queue, cl_mem,cl_mem, size_t,size_t,size_t, cl_uint,const cl_event *,cl_event *);
 
typedef cl_int (CL_CALLBACK * t_clEnqueueCopyBufferRect)(cl_command_queue, cl_mem,cl_mem, const size_t *,const size_t *,const size_t *, size_t,size_t,size_t,size_t,cl_uint,const cl_event *,cl_event *);
 
typedef cl_int (CL_CALLBACK * t_clEnqueueReadImage)(cl_command_queue,cl_mem,cl_bool, const size_t *,const size_t *,size_t,size_t, void *,cl_uint,const cl_event *,cl_event *);

typedef cl_int (CL_CALLBACK * t_clEnqueueWriteImage)(cl_command_queue,cl_mem,cl_bool, const size_t *,const size_t *,size_t,size_t, const void *,cl_uint,const cl_event *,cl_event *);

typedef cl_int (CL_CALLBACK * t_clEnqueueFillImage)(cl_command_queue,cl_mem, const void *, const size_t *, const size_t *, cl_uint, const cl_event *, cl_event *);
 
typedef cl_int (CL_CALLBACK * t_clEnqueueCopyImage)(cl_command_queue,cl_mem,cl_mem, const size_t *,const size_t *,const size_t *, cl_uint,const cl_event *,cl_event *);

typedef cl_int (CL_CALLBACK * t_clEnqueueCopyImageToBuffer)(cl_command_queue,cl_mem,cl_mem, const size_t *,const size_t *, size_t,cl_uint,const cl_event *,cl_event *);

typedef cl_int (CL_CALLBACK * t_clEnqueueCopyBufferToImage)(cl_command_queue,cl_mem,cl_mem, size_t,const size_t *,const size_t *, cl_uint,const cl_event *,cl_event *);

typedef void * (CL_CALLBACK * t_clEnqueueMapBuffer)(cl_command_queue,cl_mem,cl_bool, cl_map_flags,size_t,size_t,cl_uint,const cl_event *,cl_event *,cl_int *);

typedef void * (CL_CALLBACK * t_clEnqueueMapImage)(cl_command_queue,cl_mem, cl_bool, cl_map_flags, const size_t *,const size_t *,size_t *,size_t *,cl_uint,const cl_event *,cl_event *,cl_int *);

typedef cl_int (CL_CALLBACK * t_clEnqueueUnmapMemObject)(cl_command_queue,cl_mem,void *,cl_uint,const cl_event *,cl_event *);

typedef cl_int (CL_CALLBACK * t_clEnqueueMigrateMemObjects)(cl_command_queue,cl_uint,const cl_mem *,cl_mem_migration_flags,cl_uint,const cl_event *,cl_event *);

typedef cl_int (CL_CALLBACK * t_clEnqueueNDRangeKernel)(cl_command_queue,cl_kernel,cl_uint,const size_t *,const size_t *,const size_t *,cl_uint,const cl_event *,cl_event *);

typedef cl_int (CL_CALLBACK * t_clEnqueueTask)(cl_command_queue,cl_kernel,cl_uint,const cl_event *,cl_event *);

typedef cl_int (CL_CALLBACK * t_clEnqueueNativeKernel)(cl_command_queue, void (CL_CALLBACK *)(void *), void *,size_t, cl_uint,const cl_mem *,const void **,cl_uint,const cl_event *,cl_event *);

typedef cl_int (CL_CALLBACK * t_clEnqueueMarkerWithWaitList)(cl_command_queue,cl_uint,const cl_event *,cl_event *);

typedef cl_int (CL_CALLBACK * t_clEnqueueBarrierWithWaitList)(cl_command_queue,cl_uint,const cl_event *,cl_event *);

typedef void * (CL_CALLBACK * t_clGetExtensionFunctionAddressForPlatform)(cl_platform_id, const char *);
 
typedef cl_mem (CL_CALLBACK * t_clCreateImage2D)(cl_context,cl_mem_flags,const cl_image_format *,size_t,size_t,size_t, void *, cl_int *);

typedef cl_mem (CL_CALLBACK * t_clCreateImage3D)(cl_context,cl_mem_flags,const cl_image_format *,size_t, size_t,size_t, size_t, size_t, void *,cl_int *);
 
typedef cl_int (CL_CALLBACK * t_clEnqueueMarker)(cl_command_queue, cl_event *);
 
typedef cl_int (CL_CALLBACK * t_clEnqueueWaitForEvents)(cl_command_queue, cl_uint, const cl_event *);
typedef cl_int (CL_CALLBACK * t_clEnqueueBarrier)(cl_command_queue );
typedef cl_int (CL_CALLBACK * t_clUnloadCompiler)(void);
typedef void * (CL_CALLBACK * t_clGetExtensionFunctionAddress)(const char *);

#endif
