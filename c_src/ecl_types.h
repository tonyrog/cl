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

typedef cl_int (* t_clGetPlatformIDs)(cl_uint, cl_platform_id *, cl_uint *);

typedef cl_int (* t_clGetPlatformInfo)(cl_platform_id, cl_platform_info, size_t , void *, size_t *);

typedef cl_int (* t_clGetDeviceIDs)(cl_platform_id,
 cl_device_type, cl_uint, cl_device_id *, cl_uint *);

typedef cl_int (* t_clGetDeviceInfo)(cl_device_id, cl_device_info, size_t, void *,
 size_t *);
 
typedef cl_int (* t_clCreateSubDevices)(cl_device_id, const cl_device_partition_property *, cl_uint, cl_device_id *, cl_uint *);

typedef cl_int (* t_clRetainDevice)(cl_device_id );
 
typedef cl_int (* t_clReleaseDevice)(cl_device_id );
 

typedef cl_context (* t_clCreateContext)(const cl_context_properties *,cl_uint,const cl_device_id *,void (CL_CALLBACK *)(const char *, const void *, size_t, void *),void *,cl_int *);

typedef cl_context (* t_clCreateContextFromType)(const cl_context_properties *,cl_device_type,void (CL_CALLBACK *)(const char *, const void *, size_t, void *),void *,cl_int *);

typedef cl_int (* t_clRetainContext)(cl_context );

typedef cl_int (* t_clReleaseContext)(cl_context );

typedef cl_int (* t_clGetContextInfo)(cl_context, cl_context_info, size_t, void *, size_t *);


typedef cl_command_queue (* t_clCreateCommandQueue)(cl_context, cl_device_id, cl_command_queue_properties,cl_int *);

typedef cl_int (* t_clRetainCommandQueue)(cl_command_queue );

typedef cl_int (* t_clReleaseCommandQueue)(cl_command_queue );

typedef cl_int (* t_clGetCommandQueueInfo)(cl_command_queue,cl_command_queue_info,size_t,void *,size_t *);


typedef cl_mem (* t_clCreateBuffer)(cl_context,cl_mem_flags,size_t,void *,cl_int *);

typedef cl_mem (* t_clCreateSubBuffer)(cl_mem,cl_mem_flags,cl_buffer_create_type,const void *,cl_int *);

typedef cl_mem (* t_clCreateImage)(cl_context,cl_mem_flags,const cl_image_format *,const cl_image_desc *, void *,cl_int *);
 
typedef cl_int (* t_clRetainMemObject)(cl_mem );

typedef cl_int (* t_clReleaseMemObject)(cl_mem );

typedef cl_int (* t_clGetSupportedImageFormats)(cl_context,cl_mem_flags,cl_mem_object_type,cl_uint,cl_image_format *,cl_uint *);
 
typedef cl_int (* t_clGetMemObjectInfo)(cl_mem,cl_mem_info, size_t,void *,size_t *);

typedef cl_int (* t_clGetImageInfo)(cl_mem,cl_image_info, size_t,void *,size_t *);

typedef cl_int (* t_clSetMemObjectDestructorCallback)( cl_mem, void (CL_CALLBACK *)( cl_mem, void*), void *) ; 


typedef cl_sampler (* t_clCreateSampler)(cl_context,cl_bool, cl_addressing_mode, cl_filter_mode,cl_int *);
typedef cl_int (* t_clRetainSampler)(cl_sampler );

typedef cl_int (* t_clReleaseSampler)(cl_sampler );

typedef cl_int (* t_clGetSamplerInfo)(cl_sampler,cl_sampler_info,size_t,void *,size_t *);
 

typedef cl_program (* t_clCreateProgramWithSource)(cl_context,cl_uint,const char **,const size_t *,cl_int *);

typedef cl_program (* t_clCreateProgramWithBinary)(cl_context,cl_uint,const cl_device_id *,const size_t *,const unsigned char **,cl_int *,cl_int *);

typedef cl_program (* t_clCreateProgramWithBuiltInKernels)(cl_context,cl_uint,const cl_device_id *,const char *,cl_int *);

typedef cl_int (* t_clRetainProgram)(cl_program );

typedef cl_int (* t_clReleaseProgram)(cl_program );

typedef cl_int (* t_clBuildProgram)(cl_program,cl_uint,const cl_device_id *,const char *, void (CL_CALLBACK *)(cl_program, void *),void *);

typedef cl_int (* t_clCompileProgram)(cl_program,cl_uint,const cl_device_id *,const char *, cl_uint,const cl_program *,const char **,void (CL_CALLBACK *)(cl_program, void *),void *);

typedef cl_program (* t_clLinkProgram)(cl_context,cl_uint,const cl_device_id *,const char *, cl_uint,const cl_program *,void (CL_CALLBACK *)(cl_program, void *),void *,cl_int *);


typedef cl_int (* t_clUnloadPlatformCompiler)(cl_platform_id );

typedef cl_int (* t_clGetProgramInfo)(cl_program,cl_program_info,size_t,void *,size_t *);

typedef cl_int (* t_clGetProgramBuildInfo)(cl_program,cl_device_id,cl_program_build_info,size_t,void *,size_t *);
 

typedef cl_kernel (* t_clCreateKernel)(cl_program,const char *,cl_int *);

typedef cl_int (* t_clCreateKernelsInProgram)(cl_program,cl_uint,cl_kernel *,cl_uint *);

typedef cl_int (* t_clRetainKernel)(cl_kernel );

typedef cl_int (* t_clReleaseKernel)(cl_kernel );

typedef cl_int (* t_clSetKernelArg)(cl_kernel,cl_uint,size_t,const void *);

typedef cl_int (* t_clGetKernelInfo)(cl_kernel,cl_kernel_info,size_t,void *,size_t *);

typedef cl_int (* t_clGetKernelArgInfo)(cl_kernel,cl_uint,cl_kernel_arg_info,size_t,void *,size_t *);

typedef cl_int (* t_clGetKernelWorkGroupInfo)(cl_kernel,cl_device_id,cl_kernel_work_group_info,size_t,void *,size_t *);


typedef cl_int (* t_clWaitForEvents)(cl_uint,const cl_event *);

typedef cl_int (* t_clGetEventInfo)(cl_event,cl_event_info,size_t,void *,size_t *);
 
typedef cl_event (* t_clCreateUserEvent)(cl_context,cl_int *); 
 
typedef cl_int (* t_clRetainEvent)(cl_event );

typedef cl_int (* t_clReleaseEvent)(cl_event );

typedef cl_int (* t_clSetUserEventStatus)(cl_event,cl_int );
 
typedef cl_int (* t_clSetEventCallback)( cl_event,cl_int,void (CL_CALLBACK *)(cl_event, cl_int, void *),void *);

typedef cl_int (* t_clGetEventProfilingInfo)(cl_event,cl_profiling_info,size_t,void *,size_t *);

typedef cl_int (* t_clFlush)(cl_command_queue );

typedef cl_int (* t_clFinish)(cl_command_queue );

typedef cl_int (* t_clEnqueueReadBuffer)(cl_command_queue,cl_mem,cl_bool,size_t,size_t, void *,cl_uint,const cl_event *,cl_event *);
 
typedef cl_int (* t_clEnqueueReadBufferRect)(cl_command_queue,cl_mem,cl_bool,const size_t *,const size_t *, const size_t *,size_t,size_t,size_t,size_t, void *,cl_uint,const cl_event *,cl_event *);
 
typedef cl_int (* t_clEnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void *, cl_uint, const cl_event *, cl_event *);
 
typedef cl_int (* t_clEnqueueWriteBufferRect)(cl_command_queue,cl_mem,cl_bool,const size_t *,const size_t *, const size_t *,size_t,size_t,size_t,size_t, const void *,cl_uint,const cl_event *,cl_event *);
 
typedef cl_int (* t_clEnqueueFillBuffer)(cl_command_queue,cl_mem, const void *, size_t, size_t, size_t, cl_uint, const cl_event *, cl_event *);
 
typedef cl_int (* t_clEnqueueCopyBuffer)(cl_command_queue, cl_mem,cl_mem, size_t,size_t,size_t, cl_uint,const cl_event *,cl_event *);
 
typedef cl_int (* t_clEnqueueCopyBufferRect)(cl_command_queue, cl_mem,cl_mem, const size_t *,const size_t *,const size_t *, size_t,size_t,size_t,size_t,cl_uint,const cl_event *,cl_event *);
 
typedef cl_int (* t_clEnqueueReadImage)(cl_command_queue,cl_mem,cl_bool, const size_t *,const size_t *,size_t,size_t, void *,cl_uint,const cl_event *,cl_event *);

typedef cl_int (* t_clEnqueueWriteImage)(cl_command_queue,cl_mem,cl_bool, const size_t *,const size_t *,size_t,size_t, const void *,cl_uint,const cl_event *,cl_event *);

typedef cl_int (* t_clEnqueueFillImage)(cl_command_queue,cl_mem, const void *, const size_t *, const size_t *, cl_uint, const cl_event *, cl_event *);
 
typedef cl_int (* t_clEnqueueCopyImage)(cl_command_queue,cl_mem,cl_mem, const size_t *,const size_t *,const size_t *, cl_uint,const cl_event *,cl_event *);

typedef cl_int (* t_clEnqueueCopyImageToBuffer)(cl_command_queue,cl_mem,cl_mem, const size_t *,const size_t *, size_t,cl_uint,const cl_event *,cl_event *);

typedef cl_int (* t_clEnqueueCopyBufferToImage)(cl_command_queue,cl_mem,cl_mem, size_t,const size_t *,const size_t *, cl_uint,const cl_event *,cl_event *);

typedef void * (* t_clEnqueueMapBuffer)(cl_command_queue,cl_mem,cl_bool, cl_map_flags,size_t,size_t,cl_uint,const cl_event *,cl_event *,cl_int *);

typedef void * (* t_clEnqueueMapImage)(cl_command_queue,cl_mem, cl_bool, cl_map_flags, const size_t *,const size_t *,size_t *,size_t *,cl_uint,const cl_event *,cl_event *,cl_int *);

typedef cl_int (* t_clEnqueueUnmapMemObject)(cl_command_queue,cl_mem,void *,cl_uint,const cl_event *,cl_event *);

typedef cl_int (* t_clEnqueueMigrateMemObjects)(cl_command_queue,cl_uint,const cl_mem *,cl_mem_migration_flags,cl_uint,const cl_event *,cl_event *);

typedef cl_int (* t_clEnqueueNDRangeKernel)(cl_command_queue,cl_kernel,cl_uint,const size_t *,const size_t *,const size_t *,cl_uint,const cl_event *,cl_event *);

typedef cl_int (* t_clEnqueueTask)(cl_command_queue,cl_kernel,cl_uint,const cl_event *,cl_event *);

typedef cl_int (* t_clEnqueueNativeKernel)(cl_command_queue, void (CL_CALLBACK *)(void *), void *,size_t, cl_uint,const cl_mem *,const void **,cl_uint,const cl_event *,cl_event *);

typedef cl_int (* t_clEnqueueMarkerWithWaitList)(cl_command_queue,cl_uint,const cl_event *,cl_event *);

typedef cl_int (* t_clEnqueueBarrierWithWaitList)(cl_command_queue,cl_uint,const cl_event *,cl_event *);

typedef void * (* t_clGetExtensionFunctionAddressForPlatform)(cl_platform_id, const char *);
 
typedef cl_mem (* t_clCreateImage2D)(cl_context,cl_mem_flags,const cl_image_format *,size_t,size_t,size_t, void *, cl_int *);

typedef cl_mem (* t_clCreateImage3D)(cl_context,cl_mem_flags,const cl_image_format *,size_t, size_t,size_t, size_t, size_t, void *,cl_int *);
 
typedef cl_int (* t_clEnqueueMarker)(cl_command_queue, cl_event *);
 
typedef cl_int (* t_clEnqueueWaitForEvents)(cl_command_queue, cl_uint, const cl_event *);
typedef cl_int (* t_clEnqueueBarrier)(cl_command_queue );
typedef cl_int (* t_clUnloadCompiler)(void);
typedef void * (* t_clGetExtensionFunctionAddress)(const char *);

#endif
