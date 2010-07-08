%%% File    : cl.erl
%%% Author  : Tony Rogvall <tony@rogvall.se>
%%% Description : Erlang OpenCL  interface
%%% Created : 25 Oct 2009 by Tony Rogvall <tony@rogvall.se>

%% @doc The erlang api for <a href="http://www.khronos.org/opencl/">OpenCL</a>.
%%
%% OpenCL (Open Computing Language) is an open royalty-free standard
%% for general purpose parallel programming across CPUs, GPUs and
%% other processors, giving software developers portable and efficient
%% access to the power of these heterogeneous processing platforms.
%%
%% OpenCL supports a wide range of applications, ranging from embedded
%% and consumer software to HPC solutions, through a low-level,
%% high-performance, portable abstraction. By creating an efficient,
%% close-to-the-metal programming interface, OpenCL will form the
%% foundation layer of a parallel computing ecosystem of
%% platform-independent tools, middleware and applications.
%%
%% OpenCL consists of an API for coordinating parallel computation
%% across heterogeneous processors; and a cross-platform programming
%% language with a well-specified computation environment. The OpenCL
%% standard:
%%
%% <li> Supports both data- and task-based parallel programming models</li>
%% <li> Utilizes a subset of ISO C99 with extensions for parallelism </li>
%% <li> Defines consistent numerical requirements based on IEEE 754</li>
%% <li> Defines a configuration profile for handheld and embedded devices</li>
%% <li> Efficiently interoperates with OpenGL, OpenGL ES, and other graphics APIs</li>
%%
%% The specification is divided into a core specification that any
%% OpenCL compliant implementation must support; a handheld/embedded
%% profile which relaxes the OpenCL compliance requirements for
%% handheld and embedded devices; and a set of optional extensions
%% that are likely to move into the core specification in later
%% revisions of the OpenCL specification.
%%
%% The documentation is re-used with the following copyright:
%%
%% Copyright © 2007-2009 The Khronos Group Inc. Permission is hereby
%% granted, free of charge, to any person obtaining a copy of this
%% software and/or associated documentation files (the "Materials"),
%% to deal in the Materials without restriction, including without
%% limitation the rights to use, copy, modify, merge, publish,
%% distribute, sublicense, and/or sell copies of the Materials, and to
%% permit persons to whom the Materials are furnished to do so,
%% subject to the condition that this copyright notice and permission
%% notice shall be included in all copies or substantial portions of
%% the Materials.
%% 
%% @headerfile "../include/cl.hrl"
%% 
-module(cl).

-export([start/0, start/1, stop/0]).
-export([noop/0]).
%% Platform
-export([get_platform_ids/0]).
-export([platform_info/0]).
-export([get_platform_info/1,get_platform_info/2]).
%% Devices
-export([get_device_ids/0, get_device_ids/2]).
-export([device_info/0]).
-export([get_device_info/1,get_device_info/2]).
%% Context
-export([create_context/1]).
-export([create_context_from_type/1]).
-export([release_context/1]).
-export([retain_context/1]).
-export([context_info/0]).
-export([get_context_info/1,get_context_info/2]).
%% Command queue
-export([create_queue/3]).
-export([set_queue_property/3]).
-export([release_queue/1]).
-export([retain_queue/1]).
-export([queue_info/0]).
-export([get_queue_info/1,get_queue_info/2]).
%% Memory object
-export([create_buffer/3, create_buffer/4]).
-export([release_mem_object/1]).
-export([retain_mem_object/1]).
-export([get_mem_object_info/1,get_mem_object_info/2]).
%% Sampler 
-export([create_sampler/4]).
-export([release_sampler/1]).
-export([retain_sampler/1]).
-export([sampler_info/0]).
-export([get_sampler_info/1,get_sampler_info/2]).
%% Program
-export([create_program_with_source/2]).
-export([create_program_with_binary/3]).
-export([release_program/1]).
-export([retain_program/1]).
-export([build_program/3]).
-export([unload_compiler/0]).
-export([program_info/0]).
-export([get_program_info/1,get_program_info/2]).
-export([program_build_info/0]).
-export([get_program_build_info/2,get_program_build_info/3]).
%% Kernel
-export([create_kernel/2]).
-export([create_kernels_in_program/1]).
-export([set_kernel_arg/3]).
-export([set_kernel_arg_size/3]).
-export([encode_argument/1]).
-export([release_kernel/1]).
-export([retain_kernel/1]).
-export([kernel_info/0]).
-export([get_kernel_info/1,get_kernel_info/2]).
-export([kernel_workgroup_info/0]).
-export([get_kernel_workgroup_info/2,get_kernel_workgroup_info/3]).
%% Events
-export([enqueue_task/3]).
-export([enqueue_nd_range_kernel/5]).
-export([enqueue_marker/1]).
-export([enqueue_wait_for_event/2]).
-export([enqueue_read_buffer/5]).
-export([enqueue_write_buffer/6]).
-export([enqueue_barrier/1]).
-export([flush/1]).
-export([finish/1]).
-export([release_event/1]).
-export([retain_event/1]).
-export([event_info/0]).
-export([get_event_info/1, get_event_info/2]).
-export([wait/1, wait/2]).

-import(lists, [map/2, reverse/1]).

-include("../include/cl.hrl").
-include("cl_int.hrl").

%%
%% @type start_arg() = { {'debug',boolean()} }
%%
-type start_arg() :: { {'debug',boolean()} }.

%%
%% @spec start([start_arg()]) -> 'ok' | {'error', term()}
%%
%% @doc Start the OpenCL application
%% 
-spec start(Args::[start_arg()]) -> 'ok' | {'error', term()}.

start(Args) ->
    application:load(?MODULE),
    application:set_env(?MODULE, arguments, Args),
    application:start(?MODULE).

%%
%% @spec start() -> 'ok' | {'error', term()}
%%
%% @doc Start the OpenCL application
%%
%% @equiv start([])
%%
-spec start() -> 'ok' | {'error', term()}.

start() -> 
    start([]).

%%
%% @spec stop() -> 'ok' | {'error', term()}
%%
%% @doc Stop the OpenCL application
%%
%% @equiv application:stop(cl)
%%
-spec stop() -> 'ok' | {'error', term()}.

stop()  -> 
    application:stop(?MODULE).

%%
%% @spec noop() -> 'ok' | {'error', cl_error()}
%%
%% @doc Run a no operation towards the driver. This call can be used
%% to messure the call overhead to the driver.
%%
-spec noop() -> 'ok' | {'error', cl_error()}.

noop() ->
    cl_drv:call(?ECL_NOOP, []).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Platform
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
%% @type cl_platform_info_key() =
%%    { 'profile' | 'name' | 'vendor' | 'extensions' }.

-type cl_platform_info_key() ::
    { 'profile' | 'name' | 'vendor' | 'extensions' }.
%%
%% @type cl_platform_info() =
%%    { {'profile',string()} |
%%      {'name',string()} |
%%      {'vendor',string()} |
%%      {'extensions',string()} }.

-type cl_platform_info() ::
    { {'profile',string()} |
      {'name',string()} |
      {'vendor',string()} |
      {'extensions',string()} }.

%%
%% @spec get_platform_ids() ->
%%    {'ok',[cl_platform_id()]} | {'error', cl_error()}
%% @doc Obtain the list of platforms available.
-spec get_platform_ids() ->
    {'ok',[cl_platform_id()]} | {'error', cl_error()}.
    
get_platform_ids() ->
    cl_drv:call(?ECL_GET_PLATFORM_IDS, <<>>).
%%
%% @spec platform_info() ->
%%    [cl_platform_info_keys()]
%% @doc Returns a list of the possible platform info keys.
-spec platform_info() ->
    [cl_platform_info_key()].

platform_info() ->
    platform_info_keys().

%%
%% @spec get_platform_info(Platform :: cl_platform_id(), 
%%			Info :: cl_platform_info_key()) ->
%%    {'ok',term()} | {'error', cl_error()}
%% @doc Get specific information about the OpenCL platform.
%% <dl>
%%
%% <dt>name</dt>     <dd>Platform name string.</dd>
%%
%% <dt>vendor</dt>   <dd>Platform vendor string.</dd>
%%
%% <dt>profile</dt>  
%%        <dd> OpenCL profile string. Returns the profile name
%%        supported by the implementation. The profile name returned
%%        can be one of the following strings:
%%
%%        FULL_PROFILE - if the implementation supports the OpenCL
%%        specification (functionality defined as part of the core
%%        specification and does not require any extensions to be supported).
%%
%%        EMBEDDED_PROFILE - if the implementation supports the OpenCL
%%        embedded profile. The embedded profile is defined to be a subset for
%%        each version of OpenCL.</dd>
%%
%% <dt>version</dt>   
%%       <dd>OpenCL version string. Returns the OpenCL version supported by the implementation.</dd>
%%
%% <dt>extensions</dt> <dd>Returns a space-separated list of extension
%% names (the extension names themselves do not contain any spaces)
%% supported by the platform. Extensions defined here must be
%% supported by all devices associated with this platform. </dd> 
%%</dl>
-spec get_platform_info(Platform :: cl_platform_id(), 
			Info :: cl_platform_info_key()) ->
    {'ok',term()} | {'error', cl_error()}.

get_platform_info(Platform, Info) ->
    get_info(?ECL_GET_PLATFORM_INFO, Platform, Info, 
	     fun platform_info_map/1).

%%
%% @spec get_platform_info(Platform::cl_platform_id()) ->
%%     {'ok', [cl_platform_info()]} | {'error', cl_error()}
%% @doc Get all information about the OpenCL platform.
%% @see get_platform_info/2
-spec get_platform_info(Platform::cl_platform_id()) ->
    {'ok', [cl_platform_info()]} | {'error', cl_error()}.

get_platform_info(Platform) ->
    get_info_list(?ECL_GET_PLATFORM_INFO, Platform, 
		  platform_info_keys(), fun platform_info_map/1).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Devices
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
%% @type cl_device_type() =
%%   {'gpu' | 'cpu' | 'accelerator' | 'all' | 'default' }
%%
-type cl_device_type() :: {'gpu' | 'cpu' | 'accelerator' | 'all' | 'default' }.
%%
%%
%% @type cl_device_types() = {cl_device_type() | [cl_device_type()]}
%%
-type cl_device_types() :: {cl_device_type() | [cl_device_type()]}.

%%
%%
%% @type cl_device_info_key() = { 'type' | 'vendor_id' | 'max_compute_units' |
%%  'max_work_item_dimensions' | 'max_work_group_size' |
%%  'max_work_item_sizes' |
%%  'preferred_vector_width_char' | 'preferred_vector_width_short' |
%%  'preferred_vector_width_int' | 'preferred_vector_width_long' |
%%  'preferred_vector_width_float' | 'preferred_vector_width_double' |
%%  'max_clock_frequency' | 'address_bits' | 'max_read_image_args' |
%%  'max_write_image_args' | 'max_mem_alloc_size' | 
%%  'image2d_max_width' | 'image2d_max_height' | 'image3d_max_width' |
%%  'image3d_max_height' | 'image3d_max_depth' | 
%%  'image_support' |
%%  'max_parameter_size' | 'max_samplers' |
%%  'mem_base_addr_align' | 'min_data_type_align_size' |
%%  'single_fp_config' |  'global_mem_cache_type' |
%%  'global_mem_cacheline_size' | 'global_mem_cache_size' | 'global_mem_size' |
%%  'max_constant_buffer_size' | 'max_constant_args' |
%%  'local_mem_type' | 'local_mem_size' | 'error_correction_support' |
%%  'profiling_timer_resolution' | 'endian_little' | 'available' |
%%  'compiler_available' | 'execution_capabilities' | 'queue_properties' |
%%  'name' | 'vendor' | 'driver_version' | 'profile' | 'version' |
%%  'extensions' | 'platform' }
%%
-type cl_device_info_key() :: { 'type' | 'vendor_id' | 'max_compute_units' |
 'max_work_item_dimensions' | 'max_work_group_size' |
 'max_work_item_sizes' |
 'preferred_vector_width_char' | 'preferred_vector_width_short' |
 'preferred_vector_width_int' | 'preferred_vector_width_long' |
 'preferred_vector_width_float' | 'preferred_vector_width_double' |
 'max_clock_frequency' | 'address_bits' | 'max_read_image_args' |
 'max_write_image_args' | 'max_mem_alloc_size' | 
 'image2d_max_width' | 'image2d_max_height' | 'image3d_max_width' |
 'image3d_max_height' | 'image3d_max_depth' | 
 'image_support' |
 'max_parameter_size' | 'max_samplers' |
 'mem_base_addr_align' | 'min_data_type_align_size' |
 'single_fp_config' |  'global_mem_cache_type' |
 'global_mem_cacheline_size' | 'global_mem_cache_size' | 'global_mem_size' |
 'max_constant_buffer_size' | 'max_constant_args' |
 'local_mem_type' | 'local_mem_size' | 'error_correction_support' |
 'profiling_timer_resolution' | 'endian_little' | 'available' |
 'compiler_available' | 'execution_capabilities' | 'queue_properties' |
 'name' | 'vendor' | 'driver_version' | 'profile' | 'version' |
 'extensions' | 'platform' }.

%%
%% @type cl_device_info() = {cl_device_info_key(), term()}
%% @todo specifiy all info types
-type cl_device_info() :: {cl_device_info_key(), term()}.

%%
%% @spec get_device_ids() -> {'ok',[cl_device_id()]} | {'error',cl_error()}
%%
%% @equiv get_devive_ids(0,all)
%%
-spec get_device_ids() -> {'ok',[cl_device_id()]} | {'error',cl_error()}.
    
get_device_ids() ->
    get_device_ids(0, all).

%%
%% @spec get_device_ids(Platform::cl_platform_id(),Type::cl_device_types()) ->
%%     {'ok',[cl_device_id()]} | {'error',cl_error()}
%% @doc Obtain the list of devices available on a platform.
%% <dl> <dt>Platform</dt> <dd>
%%
%% Refers to the platform ID returned by <c>get_platform_ids</c> or can be
%% NULL. If platform is NULL, the behavior is implementation-defined. </dd>
%% 
%% <dt>Type</dt> <dd>
%% 
%% A list that identifies the type of OpenCL device. The
%% device_type can be used to query specific OpenCL devices or all
%% OpenCL devices available. </dd>
%%
%% </dl> 
%%
%%  get_device_ids/2 may return all or a subset of the actual
%%  physical devices present in the platform and that match
%%  device_type.
%%
%% The application can query specific capabilities of the OpenCL
%% device(s) returned by get_device_ids/2. This can be used by the
%% application to determine which device(s) to use.
%%
-spec get_device_ids(Platform::cl_platform_id(),Type::cl_device_types()) ->
    {'ok',[cl_device_id()]} | {'error',cl_error()}.

get_device_ids(Platform, Type) ->
    TypeID = encode_device_types(Type),
    cl_drv:call(?ECL_GET_DEVICE_IDS,
		 << ?pointer_t(Platform), ?u_int32_t(TypeID) >> ).

%%
%% @spec device_info() -> [cl_device_info_key()]
%% @doc Return a list of possible device info queries.
%% @see get_device_info/2
-spec device_info() -> [cl_device_info_key()].
    
device_info() ->
    device_info_keys().

%%
%% @spec get_device_info(DevID::cl_device_id(), Info::cl_device_info_key()) ->
%%   {'ok', term()} | {'error', cl_error()}
%% @doc Get information about an OpenCL device.
%% 
%% <dl> <dt>'type' </dt> <dd> <p>The OpenCL device type. Currently
%% supported values are one of or a combination of: CL_DEVICE_TYPE_CPU,
%% CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_ACCELERATOR, or
%% CL_DEVICE_TYPE_DEFAULT.</p></dd>
%%
%% <dt>'vendor_id'</dt> <dd> <p>A unique device vendor identifier. An
%% example of a unique device identifier could be the PCIe ID.</p> </dd>
%%
%% <dt>'max_compute_units'</dt> <dd> <p>The number of parallel compute
%% cores on the OpenCL device. The minimum value is 1.</p> </dd>
%%
%% <dt>'max_work_item_dimensions'</dt> <dd> <p>Maximum dimensions that
%% specify the global and local work-item IDs used by the data parallel
%% execution model. (@see enqueue_nd_range_kernel/5). The
%% minimum value is 3.</p></dd>
%%
%% <dt>'max_work_group_size'</dt> <dd> <p>Maximum number of
%% work-items in a work-group executing a kernel using the data parallel
%% execution model. (@see enqueue_nd_range_kernel/5). The minimum value
%% is 1.</p> </dd> 
%%
%% <dt>'max_work_item_sizes'</dt> <dd> <p>Maximum number of work-items
%% that can be specified in each dimension of the work-group to enqueue_nd_range_kernel/5.</p>
%% <p>Returns <code class="varname">n</code> entries, where <code
%% class="varname">n</code> is the value returned by the query for
%% CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS. The minimum value is (1, 1,
%% 1).</p></dd>
%%
%% <dt>'preferred_vector_width_TYPE'</dt> <dd> <p>Preferred native vector
%% width size for built-in scalar types that can be put into vectors. The
%% vector width is defined as the number of scalar elements that can be
%% stored in the vector.</p> <p>If the <c>cl_khr_fp64</c> extension is
%% not supported, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE must return
%% 0.</p></dd>
%%
%% <dt>'max_clock_frequency'</dt> <dd> <p>Maximum configured clock
%% frequency of the device in MHz.</p>
%%
%% </dd> <dt>'address_bits'</dt> <dd> The default compute device address
%% space size specified as an unsigned integer value in bits. Currently
%% supported values are 32 or 64 bits. </dd>
%%
%% <dt>'max_read_image_args'</dt> <dd> <p>Max number of simultaneous
%% image objects that can be read by a kernel. The minimum value is 128
%% if CL_DEVICE_IMAGE_SUPPORT is CL_TRUE.</p></dd>
%%
%% <dt>'max_write_image_args'</dt> <dd> <p>Max number of
%% simultaneous image objects that can be written to by a kernel. The
%% minimum value is 8 if CL_DEVICE_IMAGE_SUPPORT is CL_TRUE.</p> </dd>
%%
%% <dt>'max_mem_alloc_size'</dt> <dd> <p>Max size of memory object
%% allocation in bytes. The minimum value is max (1/4th of
%% CL_DEVICE_GLOBAL_MEM_SIZE, 128*1024*1024)</p></dd>
%%
%% <dt>'image2d_max_width'</dt> <dd> <p>Max width of 2D image in
%% pixels. The minimum value is 8192 if CL_DEVICE_IMAGE_SUPPORT is
%% CL_TRUE.</p> </dd>
%%
%% <dt>'image2d_max_height'</dt> <dd> <p>Max height of 2D image in
%% pixels. The minimum value is 8192 if CL_DEVICE_IMAGE_SUPPORT is
%% CL_TRUE.</p> </dd>
%%
%% <dt>'image3d_max_width'</dt> <dd> <p>Max width of 3D image in
%% pixels. The minimum value is 2048 if CL_DEVICE_IMAGE_SUPPORT is
%% CL_TRUE.</p> </dd> 
%%
%% <dt>'image3d_max_height'</dt> <dd> <p>Max height of 3D image in
%% pixels. The minimum value is 2048 if CL_DEVICE_IMAGE_SUPPORT is
%% CL_TRUE.</p> </dd>
%%
%% <dt>'image3d_max_depth'</dt> <dd> <p>Max depth of 3D image in
%% pixels. The minimum value is 2048 if CL_DEVICE_IMAGE_SUPPORT is
%% CL_TRUE.</p> </dd>
%%
%% <dt>'image_support'</dt> <dd> <p>Is CL_TRUE if images are supported by
%% the OpenCL device and CL_FALSE otherwise.</p> </dd>
%%
%% <dt>'max_parameter_size'</dt> <dd> <p>Max size in bytes of the
%% arguments that can be passed to a kernel. The minimum value is
%% 256.</p> </dd>
%%
%% <dt>'max_samplers'</dt> <dd> <p>Maximum number of samplers that can be
%% used in a kernel. The minimum value is 16 if CL_DEVICE_IMAGE_SUPPORT
%% is CL_TRUE.</p> </dd>
%%
%% <dt>'mem_base_addr_align'</dt> <dd> <p>Describes the alignment in bits
%% of the base address of any allocated memory object.</p> </dd>
%%
%% <dt>'min_data_type_align_size'</dt> <dd> <p>The smallest alignment in
%% bytes which can be used for any data type.</p> </dd>
%% <dt>'single_fp_config'</dt> <dd> <p>Describes single precision
%% floating-point capability of the device. This is a bit-field that
%% describes one or more of the following values:</p> <p>CL_FP_DENORM -
%% denorms are supported</p> <p>CL_FP_INF_NAN - INF and quiet NaNs are
%% supported</p> <p>CL_FP_ROUND_TO_NEAREST - round to nearest even
%% rounding mode supported</p>
%% <p>CL_FP_ROUND_TO_ZERO - round to zero rounding mode supported</p>
%% <p>CL_FP_ROUND_TO_INF - round to +ve and -ve infinity rounding modes supported</p>
%% <p>CL_FP_FMA - IEEE754-2008 fused multiply-add is supported</p>
%% <p>The mandated minimum floating-point capability is CL_FP_ROUND_TO_NEAREST | CL_FP_INF_NAN.</p>
%% </dd>
%%
%% <dt>'global_mem_cache_type'</dt> <dd> <p>Return type:
%% cl_device_mem_cache_type</p> <p>Type of global memory cache
%% supported. Valid values are: CL_NONE, CL_READ_ONLY_CACHE, and
%% CL_READ_WRITE_CACHE.</p> </dd>
%%
%% <dt>'global_mem_cacheline_size'</dt> <dd>
%% <p>Size of global memory cache line in bytes.</p>
%% </dd>
%%
%% <dt>'global_mem_cache_size'</dt> <dd>
%% <p>Size of global memory cache in bytes.</p>
%% </dd>
%%
%% <dt>'global_mem_size'</dt> <dd>
%% <p>Size of global device memory in bytes.</p>
%% </dd>
%%
%% <dt>'max_constant_buffer_size'</dt> <dd>
%% <p>Max size in bytes of a constant buffer allocation. The minimum value is 64 KB.</p></dd>
%%
%%  <dt>'max_constant_args'</dt> <dd> <p>Max number of arguments
%% declared with the <c>__constant</c> qualifier in a kernel. The minimum
%% value is 8.</p> </dd>
%%
%% <dt>'local_mem_type'</dt> <dd> <p>Type of local memory
%% supported. This can be set to CL_LOCAL implying dedicated local memory
%% storage such as SRAM, or CL_GLOBAL.</p> </dd>
%%
%% <dt>'local_mem_size'</dt> <dd> <p>Size of local memory arena in
%% bytes. The minimum value is 16 KB.</p></dd>
%%
%% <dt>'error_correction_support'</dt> <dd> Is CL_TRUE if the device
%% implements error correction for the memories, caches, registers
%% etc. in the device. Is CL_FALSE if the device does not implement error
%% correction. This can be a requirement for certain clients of
%% OpenCL.</dd>
%%
%% <dt>'profiling_timer_resolution'</dt> <dd> <p>Describes the resolution
%% of device timer. This is measured in nanoseconds.</p> </dd>
%%
%% <dt>'endian_little'</dt> <dd> Is CL_TRUE if the OpenCL device is a
%% little endian device and CL_FALSE otherwise.  </dd>
%%
%% <dt>'available'</dt> <dd> Is CL_TRUE if the device is available and
%% CL_FALSE if the device is not available.  </dd>
%%
%% <dt>'compiler_available'</dt> <dd> Is CL_FALSE if the implementation
%% does not have a compiler available to compile the program source. Is
%% CL_TRUE if the compiler is available. This can be CL_FALSE for the
%% embededed platform profile only.  </dd>
%%
%% <dt>'execution_capabilities'</dt> <dd> <p>Return type:
%% cl_device_exec_capabilities</p> <p>Describes the execution
%% capabilities of the device. This is a bit-field that describes one or
%% more of the following values:</p> <p>CL_EXEC_KERNEL - The OpenCL
%% device can execute OpenCL kernels.</p> <p>CL_EXEC_NATIVE_KERNEL - The
%% OpenCL device can execute native kernels.</p> <p>The mandated minimum
%% capability is CL_EXEC_KERNEL.</p> </dd>
%%
%% <dt>'queue_properties'</dt> <dd> <p>Describes the command-queue
%% properties supported by the device.  This is a bit-field that
%% describes one or more of the following values:</p>
%% <p>CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE</p>
%% <p>CL_QUEUE_PROFILING_ENABLE</p> <p>These properties are described in
%% the table for create_queue/3 .  The mandated minimum capability is
%% CL_QUEUE_PROFILING_ENABLE.</p> </dd>
%%
%% <dt>'name'</dt> <dd> <p>Device name string.</p> </dd>
%%
%% <dt>'vendor'</dt> <dd><p>Vendor name string.</p></dd>
%%
%% <dt>'driver_version'</dt> <dd><p>OpenCL software driver version string</p> </dd>
%%
%% <dt>'profile'</dt> <dd> <p>OpenCL profile string. Returns the profile
%% name supported by the device (see note). The profile name returned can
%% be one of the following strings:</p>
%% <p>FULL_PROFILE - if the device supports the OpenCL specification
%% (functionality defined as part of the core
%% specification and does not require any extensions
%% to be supported).</p> <p>EMBEDDED_PROFILE - if
%% the device supports the OpenCL embedded
%% profile.</p></dd>
%%
%% <dt>'version'</dt> <dd> <p>OpenCL version string.</p> </dd>
%%
%% <dt>'extensions' </dt> <dd><p>Returns a space separated list of extension names (the extension
%% names themselves do not contain any spaces). </p></dd>
%%
%% <dt>'platform' </dt> <dd> <p>The platform associated with this device.</p> </dd>
%%
%% </dl>
%%
%% <c>NOTE</c>: CL_DEVICE_PROFILE: The platform profile returns the profile that is
%% implemented by the OpenCL framework. If the platform profile
%% returned is FULL_PROFILE, the OpenCL framework will support devices
%% that are FULL_PROFILE and may also support devices that are
%% EMBEDDED_PROFILE. The compiler must be available for all devices
%% i.e. CL_DEVICE_COMPILER_AVAILABLE is CL_TRUE. If the platform
%% profile returned is EMBEDDED_PROFILE, then devices that are only
%% EMBEDDED_PROFILE are supported.

-spec get_device_info(Device::cl_device_id(), Info::cl_device_info_key()) ->
    {'ok', term()} | {'error', cl_error()}.

get_device_info(Device, Info) ->
    get_info(?ECL_GET_DEVICE_INFO, Device, Info, fun device_info_map/1).

%%
%% @spec get_device_info(Device) ->
%%    {'ok', [cl_device_info()]} | {'error', cl_error()}
%% @doc Get all device info.
%% @see get_device_info/2
-spec get_device_info(Device::cl_device_id()) ->
    {'ok', [cl_device_info()]} | {'error', cl_error()}.

get_device_info(Device) ->
    get_info_list(?ECL_GET_DEVICE_INFO, Device, 
		  device_info_keys(), fun device_info_map/1).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Context
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% @type cl_context_info_key() = {'reference_count' | 'devices' | 'properties'}
-type cl_context_info_key() :: {'reference_count' | 'devices' | 'properties'}.

%% @type cl_context_info() = 
%%  { {'reference_count', cl_uint()},
%%    {'devices', [cl_device()]},
%%    {'properties', [cl_int()]} }
-type cl_context_info() ::
  { {'reference_count', cl_uint()} |
    {'devices', [cl_device_id()]} |
    {'properties', [cl_int()]} }.

%%
%% @spec create_context(DeviceList::[cl_device_id()]) ->
%%    {'ok', cl_context()} | {'error', cl_error()}
%% @doc Creates an OpenCL context.
%%
%% An OpenCL context is created with one or more devices. Contexts are
%% used by the OpenCL runtime for managing objects such as
%% command-queues, memory, program and kernel objects and for
%% executing kernels on one or more devices specified in the context.
%%
%% NOTE: create_context/1 and create_context_from_type/1 perform an
%% implicit retain. This is very helpful for 3rd party libraries,
%% which typically get a context passed to them by the
%% application. However, it is possible that the application may
%% delete the context without informing the library. Allowing
%% functions to attach to (i.e. retain) and release a context solves
%% the problem of a context being used by a library no longer being
%% valid.

-spec create_context(DeviceList::[cl_device_id()]) ->
    {'ok', cl_context()} | {'error', cl_error()}.

create_context(DeviceList) ->
    DeviceData = encode_pointer_array(DeviceList),
    cl_drv:create(?ECL_CREATE_CONTEXT,
		   ?ECL_RELEASE_CONTEXT,
		  DeviceData ).

%%
%% @spec create_context_from_type(Type::cl_device_types())->
%%    {'ok', cl_context()} | {'error', cl_error()}
%% @doc Create an OpenCL context from a device type that identifies the specific device(s) to use. 
%%
%% NOTE: 
%% create_context_from_type/1 may return all or a subset of the
%% actual physical devices present in the platform and that match
%% device_type.
%% 
%% create_context/1 and create_context_from_type/1 perform an
%% implicit retain. This is very helpful for 3rd party libraries,
%% which typically get a context passed to them by the
%% application. However, it is possible that the application may
%% delete the context without informing the library. Allowing
%% functions to attach to (i.e. retain) and release a context solves
%% the problem of a context being used by a library no longer being
%% valid.
-spec create_context_from_type(Type::cl_device_types())->
    {'ok', cl_context()} | {'error', cl_error()}.

create_context_from_type(Type) ->
    case get_device_ids(0, Type) of
	{ok,DeviceList} ->
	    create_context(DeviceList);
	Error ->
	    Error
    end.

%%
%% @spec release_context(Context::cl_context()) ->
%%    'ok' | {'error', cl_error()}
%% @doc Decrement the context reference count. 
%% 
%% After the context reference count becomes zero and all the objects
%% attached to context (such as memory objects, command-queues) are
%% released, the context is deleted.
-spec release_context(Context::cl_context()) ->
    'ok' | {'error', cl_error()}.

release_context(Context) ->
    cl_drv:release(?ECL_RELEASE_CONTEXT, Context).

%%
%% @spec retain_context(Context::cl_context()) ->
%%     'ok' | {'error', cl_error()}
%% @doc Increment the context reference count. 
%% @see create_context
-spec retain_context(Context::cl_context()) ->
    'ok' | {'error', cl_error()}.

retain_context(Context) ->
    cl_drv:retain(?ECL_RETAIN_CONTEXT, Context).

%%
%% @spec context_info() -> [cl_context_info_key()]
%% @doc List context info queries.
-spec context_info() -> [cl_context_info_key()].

context_info() ->
    context_info_keys().

%%
%% @spec get_context_info(Context::cl_context(),Info::cl_context_info_key()) ->
%%   {'ok', term()} | {'error', cl_error()}
%% @doc  Query information about a context. 
%%
%% <dl> <dt>reference_count</dt> <dd> Return the context reference
%% count. The reference count returned should be considered
%% immediately stale. It is unsuitable for general use in
%% applications. This feature is provided for identifying memory
%% leaks. </dd>
%% 
%% <dt>devices</dt> <dd>Return the list of devices in context.</dd>
%%
%% <dt>properties</dt> <dd>Return the context properties.</dd>
%% </dl>
-spec get_context_info(Context::cl_context(), Info::cl_context_info_key()) ->
    {'ok', term()} | {'error', cl_error()}.

get_context_info(Context, Info) ->
    get_info(?ECL_GET_CONTEXT_INFO, Context, Info, fun context_info_map/1).

%% @spec get_context_info(Context::cl_context()) ->
%%    {'ok', [cl_context_info()]} | {'error', cl_error()}
%% @doc Get all context info.
%% @see get_context_info/2
-spec get_context_info(Context::cl_context()) ->
    {'ok', [cl_context_info()]} | {'error', cl_error()}.

get_context_info(Context) ->
    get_info_list(?ECL_GET_CONTEXT_INFO, Context, 
		  context_info_keys(), fun context_info_map/1).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Command Queue (Queue)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
-type cl_queue_property() :: { 'out_of_order_exec_mode_enable' | 
			       'profiling_enabled' }.
%%
%% @spec create_queue(Context::cl_context(),Device::cl_device_id(),
%%                    Properties::[cl_queue_property()]) ->
%%    {'ok', cl_queue()} | {'error', cl_error()}
%% @doc Create a command-queue on a specific device.
%%
%% <dl> 
%% <dt>'out_of_order_exec_mode_enable'</dt> <dd> Determines
%% whether the commands queued in the command-queue are executed
%% in-order or out-of-order. If set, the commands in the command-queue
%% are executed out-of-order. Otherwise, commands are executed
%% in-order.</dd>
%% 
%% <dt>'profiling_enabled'</dt> <dd> Enable or disable profiling of
%% commands in the command-queue. If set, the profiling of commands is
%% enabled. Otherwise profiling of commands is disabled. See
%% clGetEventProfilingInfo for more information.
%% </dd>
%% </dl>
%% 
%% The OpenCL functions that are submitted to a command-queue are
%% enqueued in the order the calls are made but can be configured to
%% execute in-order or out-of-order. The properties argument in
%% clCreateCommandQueue can be used to specify the execution order.
%%
%% If the CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE property of a
%% command-queue is not set, the commands enqueued to a command-queue
%% execute in order. For example, if an application calls
%% clEnqueueNDRangeKernel to execute kernel A followed by a
%% clEnqueueNDRangeKernel to execute kernel B, the application can
%% assume that kernel A finishes first and then kernel B is
%% executed. If the memory objects output by kernel A are inputs to
%% kernel B then kernel B will see the correct data in memory objects
%% produced by execution of kernel A. If the
%% CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE property of a commandqueue
%% is set, then there is no guarantee that kernel A will finish before
%% kernel B starts execution.
%%
%% Applications can configure the commands enqueued to a command-queue
%% to execute out-of-order by setting the
%% CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE property of the
%% command-queue. This can be specified when the command-queue is
%% created or can be changed dynamically using
%% clCreateCommandQueue. In out-of-order execution mode there is no
%% guarantee that the enqueued commands will finish execution in the
%% order they were queued. As there is no guarantee that kernels will
%% be executed in order, i.e. based on when the clEnqueueNDRangeKernel
%% calls are made within a command-queue, it is therefore possible
%% that an earlier clEnqueueNDRangeKernel call to execute kernel A
%% identified by event A may execute and/or finish later than a
%% clEnqueueNDRangeKernel call to execute kernel B which was called by
%% the application at a later point in time. To guarantee a specific
%% order of execution of kernels, a wait on a particular event (in
%% this case event A) can be used. The wait for event A can be
%% specified in the event_wait_list argument to clEnqueueNDRangeKernel
%% for kernel B.
%%
%% In addition, a wait for events or a barrier command can be enqueued
%% to the command-queue. The wait for events command ensures that
%% previously enqueued commands identified by the list of events to
%% wait for have finished before the next batch of commands is
%% executed. The barrier command ensures that all previously enqueued
%% commands in a command-queue have finished execution before the next
%% batch of commands is executed.
%%
%% Similarly, commands to read, write, copy or map memory objects that
%% are enqueued after clEnqueueNDRangeKernel, clEnqueueTask or
%% clEnqueueNativeKernel commands are not guaranteed to wait for
%% kernels scheduled for execution to have completed (if the
%% CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE property is set). To ensure
%% correct ordering of commands, the event object returned by
%% clEnqueueNDRangeKernel, clEnqueueTask or clEnqueueNativeKernel can
%% be used to enqueue a wait for event or a barrier command can be
%% enqueued that must complete before reads or writes to the memory
%% object(s) occur.
-spec create_queue(Context::cl_context(),Device::cl_device_id(),
		   Properties::[cl_queue_property()]) ->
    {'ok', cl_queue()} | {'error', cl_error()}.

create_queue(Context,Device,Properties) ->
    Prop = encode_queue_properties(Properties),
    cl_drv:create(?ECL_CREATE_QUEUE,
		   ?ECL_RELEASE_QUEUE,
		   << ?pointer_t(Context), ?pointer_t(Device), ?u_int32_t(Prop)>>).

%%
%% @spec set_queue_property(Queue::cl_queue(),
%%                          Properties::[cl_queue_property()],
%%                          Enable::bool()) ->
%%    'ok' | {'error', cl_error()}
%% @doc Enable or disable the properties of a command-queue.
%%
%% As specified for create_queue/3, the
%% CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE command-queue property
%% determines whether the commands in a command-queue are executed
%% in-order or out-of-order. Changing this command-queue property will
%% cause the OpenCL implementation to block until all previously
%% queued commands in command_queue have completed. This can be an
%% expensive operation and therefore changes to the
%% CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE property should be only done
%% when absolutely necessary.
%%
%% It is possible that a device(s) becomes unavailable after a context
%% and command-queues that use this device(s) have been created and
%% commands have been queued to command-queues. In this case the
%% behavior of OpenCL API calls that use this context (and
%% command-queues) are considered to be implementation-defined. The
%% user callback function, if specified when the context is created,
%% can be used to record appropriate information in the errinfo,
%% private_info arguments passed to the callback function when the
%% device becomes unavailable.
-spec set_queue_property(Queue::cl_queue(),
                         Properties::[cl_queue_property()],
                         Enable::boolean()) ->
    'ok' | {'error', cl_error()}.

set_queue_property(Queue, Properties, Enable) ->
    Prop = encode_queue_properties(Properties),
    Ena  = encode_bool(Enable),
    cl_drv:call(?ECL_SET_QUEUE_PROPERTY,
		<< ?pointer_t(Queue), ?u_int32_t(Prop), ?u_int32_t(Ena)>>).

%%
%% @spec release_queue(Queue::cl_queue()) ->
%%    'ok' | {'error', cl_error()}
%% @doc Decrements the command_queue reference count.
%%
%% After the command_queue reference count becomes zero and all
%% commands queued to command_queue have finished (e.g., kernel
%% executions, memory object updates, etc.), the command-queue is
%% deleted.
-spec release_queue(Queue::cl_queue()) ->
    'ok' | {'error', cl_error()}.
release_queue(Queue) ->
    cl_drv:release(?ECL_RELEASE_QUEUE, Queue).

%%
%% @spec retain_queue(Queue::cl_queue()) ->
%%    'ok' | {'error', cl_error()}
%% @doc Increments the command_queue reference count.
%%
%%  create_queue/3 performs an implicit retain. This is very
%%  helpful for 3rd party libraries, which typically get a
%%  command-queue passed to them by the application. However, it is
%%  possible that the application may delete the command-queue without
%%  informing the library. Allowing functions to attach to
%%  (i.e. retain) and release a command-queue solves the problem of a
%%  command-queue being used by a library no longer being valid.

-spec retain_queue(Queue::cl_queue()) ->
    'ok' | {'error', cl_error()}.

retain_queue(Queue) ->
    cl_drv:retain(?ECL_RETAIN_QUEUE, Queue).

%% @spec queue_info() -> [queue_info_keys()]
%% @doc Returns the list of possible queue info items.
queue_info() ->
    queue_info_keys().

%% @spec get_queue_info(Queue, Info) -> {ok, term()}
%% @doc Return the specified queue info
get_queue_info(Queue, Info) ->
    get_info(?ECL_GET_QUEUE_INFO, Queue, Info, 
	     fun queue_info_map/1).

%% @spec get_queue_info(Queue) -> [queue_info_keys()]
%% @doc Returns all queue info.
get_queue_info(Queue) ->
    get_info_list(?ECL_GET_QUEUE_INFO, Queue, 
		  queue_info_keys(), fun queue_info_map/1).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Mem
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
%% @type cl_mem_flag() = { 'read_write' | 'write_only' | 'read_only' |
%%                         'use_host_ptr' | 'alloc_host_ptr' |
%%                         'copy_host_ptr'}
%%
-type cl_mem_flag() :: { 'read_write' | 'write_only' | 'read_only' |
			 'use_host_ptr' | 'alloc_host_ptr' |
			 'copy_host_ptr'}.

%%
%% @spec create_buffer(Context::cl_context(),Flags::cl_mem_flags(),
%%                      Size::non_neg_integer()) ->
%%    {'ok', cl_mem()} | {'error', cl_error()}
%%
%% @equiv create_buffer(Context,Flags,Size,<<>>)
%%    
-spec create_buffer(Context::cl_context(),Flags::[cl_mem_flag()],
		    Size::non_neg_integer()) ->
    {'ok', cl_mem()} | {'error', cl_error()}.

create_buffer(Context,Flags,Size) ->
    FlagBits = encode_mem_flags(Flags),
    %% also: async_create
    cl_drv:create(?ECL_CREATE_BUFFER, 
		   ?ECL_RELEASE_MEM_OBJECT, 
		   <<?pointer_t(Context), 
		    ?u_int32_t(FlagBits), 
		    ?u_int32_t(Size)>>).

%%
%% @spec create_buffer(Context::cl_context(),Flags::[cl_mem_flag()],
%%                      Size::non_neg_integer(), Data::binary()) ->
%%    {'ok', cl_mem()} | {'error', cl_error()}
%% @doc  Creates a buffer object. 
%% 
-spec create_buffer(Context::cl_context(),Flags::[cl_mem_flag()],
		    Size::non_neg_integer(),Data::binary()) ->
    {'ok', cl_mem()} | {'error', cl_error()}.

create_buffer(Context,Flags,Size,Data) ->
    FlagData = encode_mem_flags(Flags),
    cl_drv:create(?ECL_CREATE_BUFFER, 
		   ?ECL_RELEASE_MEM_OBJECT, 
		   <<?pointer_t(Context), 
		    ?u_int32_t(FlagData), 
		    ?u_int32_t(Size),
		    Data/binary>>).

%%
%% @spec release_mem_object(Mem::cl_mem()) ->
%%    'ok' | {'error', cl_error()}
%% @doc  Decrements the memory object reference count. 
%%
%% After the memobj reference count becomes zero and commands queued
%% for execution on a command-queue(s) that use memobj have finished,
%% the memory object is deleted.
-spec release_mem_object(Mem::cl_mem()) ->
    'ok' | {'error', cl_error()}.

release_mem_object(Mem) ->
    cl_drv:release(?ECL_RELEASE_MEM_OBJECT, Mem).

%%
%% @spec retain_mem_object(Mem::cl_mem()) ->
%%    'ok' | {'error', cl_error()}
%% @doc Increments the memory object reference count. 
-spec retain_mem_object(Mem::cl_mem()) ->
    'ok' | {'error', cl_error()}.

retain_mem_object(Mem) ->
    cl_drv:release(?ECL_RETAIN_MEM_OBJECT, Mem).


-type cl_mem_info_key() :: {'object_type' | 'flags' | 'size' | 'host_ptr' | 'map_count' |
			    'reference_count' | 'context'}.
%%
%% @spec get_mem_object_info(Mem::cl_mem(), InfoType::cl_mem_info_key()) ->
%%    {'ok', term()} | {'error', cl_error()}
%%
%% @doc Used to get <c>InfoType</c> information that is common to all memory objects
%% (buffer and image objects).
get_mem_object_info(Mem, InfoType) ->
    get_info(?ECL_GET_MEM_OBJECT_INFO, Mem, InfoType, fun mem_info_map/1).
%%
%% @spec get_mem_object_info(Mem::cl_mem()) ->
%%    {'ok', term()} | {'error', cl_error()}
%%
%% @doc Used to get all information that is common to all memory objects
%% (buffer and image objects).
get_mem_object_info(Mem) ->
    get_info_list(?ECL_GET_MEM_OBJECT_INFO, Mem, 
		  mem_info_keys(), fun mem_info_map/1).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Sample
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% @type cl_addressing_mode() = {'none'|'clamp_to_edge'|'clamp'|'repeat'}
%%
-type cl_addressing_mode() :: {'none'|'clamp_to_edge'|'clamp'|'repeat'}.

%% @type cl_filter_mode() = {'nearest' | 'linear' }
-type cl_filter_mode() :: {'nearest' | 'linear' }.

%%
%% @spec create_sampler(Context::cl_context(),Normalized::boolean(),
%%                      AddressingMode::cl_addressing_mode(),
%%                      FilterMode::cl_filter_mode()) -> 
%%    {'ok', cl_sampler()} | {'error', cl_error()}
%% @doc Creates a sampler object. 
%%
%%  A sampler object describes how to sample an image when the image
%%  is read in the kernel. The built-in functions to read from an
%%  image in a kernel take a sampler as an argument. The sampler
%%  arguments to the image read function can be sampler objects
%%  created using OpenCL functions and passed as argument values to
%%  the kernel or can be samplers declared inside a kernel. In this
%%  section we discuss how sampler objects are created using OpenCL
%%  functions.
-spec create_sampler(Context::cl_context(),Normalized::boolean(),
		     AddressingMode::cl_addressing_mode(),
		     FilterMode::cl_filter_mode()) -> 
    {'ok', cl_sampler()} | {'error', cl_error()}.

create_sampler(Context, Normalized, AddressingMode, FilterMode) ->
    Norm = encode_bool(Normalized),
    Addr = encode_addressing_mode(AddressingMode),
    Filt = encode_filter_mode(FilterMode),
    cl_drv:create(?ECL_CREATE_SAMPLER,
		   ?ECL_RELEASE_SAMPLER,
		   <<?pointer_t(Context),
		    ?u_int32_t(Norm),
		    ?u_int32_t(Addr),
		    ?u_int32_t(Filt)>>).

%%
%% @spec release_sampler(Sampler::cl_sampler()) ->
%%    'ok' | {'error', cl_error()}
%% @doc Decrements the sampler reference count. 
%%
%%  The sampler object is deleted after the reference count becomes
%%  zero and commands queued for execution on a command-queue(s) that
%%  use sampler have finished.
-spec release_sampler(Sampler::cl_sampler()) ->
    'ok' | {'error', cl_error()}.

release_sampler(Mem) ->
    cl_drv:release(?ECL_RELEASE_SAMPLER, Mem).

%%
%% @spec retain_sampler(Sampler::cl_sampler()) ->
%%    'ok' | {'error', cl_error()}
%% @doc Increments the sampler reference count. 
-spec retain_sampler(Sampler::cl_sampler()) ->
    'ok' | {'error', cl_error()}.

retain_sampler(Sampler) ->
    cl_drv:release(?ECL_RETAIN_SAMPLER, Sampler).

sampler_info() ->
    sampler_info_keys().

%% @spec get_sampler_info(Sampler::cl_sampler(), InfoType::cl_sampler_info_type()) -> 
%%    {'ok', term()} | {'error', cl_error()}
%% @doc Returns <c>InfoType</c> information about the sampler object. 
get_sampler_info(Sampler, Info) ->
    get_info(?ECL_GET_SAMPLER_INFO, Sampler, Info, 
	     fun sampler_info_map/1).

%% @spec get_sampler_info(Sampler::cl_sampler()) -> {'ok', term()} | {'error', cl_error()}
%% @doc Returns all information about the sampler object. 
%% @see get_sampler_info/2
get_sampler_info(Sampler) ->
    get_info_list(?ECL_GET_SAMPLER_INFO, Sampler, 
		  sampler_info_keys(), fun sampler_info_map/1).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Program
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
%% @spec create_program_with_source(Context::cl_context(),
%%                                  Source::iodata()) ->
%%    {'ok', cl_program()} | {'error', cl_error()}
%%
%% @doc Creates a program object for a context, 
%% and loads the source code specified by the text strings in the
%% strings array into the program object.
%% 
%%  The devices associated with the program object are the devices associated with context.

%% OpenCL allows applications to create a program object using the
%% program source or binary and build appropriate program
%% executables. This allows applications to determine whether they
%% want to use the pre-built offline binary or load and compile the
%% program source and use the executable compiled/linked online as the
%% program executable. This can be very useful as it allows
%% applications to load and build program executables online on its
%% first instance for appropriate OpenCL devices in the system. These
%% executables can now be queried and cached by the
%% application. Future instances of the application launching will no
%% longer need to compile and build the program executables. The
%% cached executables can be read and loaded by the application, which
%% can help significantly reduce the application initialization time.

%% An OpenCL program consists of a set of kernels that are identified
%% as functions declared with the __kernel qualifier in the program
%% source. OpenCL programs may also contain auxiliary functions and
%% constant data that can be used by __kernel functions. The program
%% executable can be generated online or offline by the OpenCL
%% compiler for the appropriate target device(s).
%%
%% @todo allow iodata and handle multiple binaries in the driver
%%
-spec create_program_with_source(Context::cl_context(),
				 Source::iodata()) ->
    {'ok', cl_program()} | {'error', cl_error()}.

create_program_with_source(Context, Source) ->
    Data = if is_binary(Source) -> Source;
	      is_list(Source) -> list_to_binary(Source)
	   end,
    %% also: async_create
    cl_drv:create(?ECL_CREATE_PROGRAM_WITH_SOURCE, 
		  ?ECL_RELEASE_PROGRAM,
		  [<<?pointer_t(Context)>>,Data]).

%%
%% @spec create_program_with_binary(Context::cl_context(),
%%                                  DeviceList::[cl_device_id()],
%%                                  BinaryList::[binary()]) ->
%%    {'ok', cl_program()} | {'error', cl_error()}
%%
%% @doc  Creates a program object for a context, and loads specified binary data into the program object. 
%% 
%% OpenCL allows applications to create a program object using the
%% program source or binary and build appropriate program
%% executables. This allows applications to determine whether they
%% want to use the pre-built offline binary or load and compile the
%% program source and use the executable compiled/linked online as the
%% program executable. This can be very useful as it allows
%% applications to load and build program executables online on its
%% first instance for appropriate OpenCL devices in the system. These
%% executables can now be queried and cached by the
%% application. Future instances of the application launching will no
%% longer need to compile and build the program executables. The
%% cached executables can be read and loaded by the application, which
%% can help significantly reduce the application initialization time.
%%
%%  The binaries and device can be generated by calling:
%%  <code>
%%    {ok,P} = cl:create_program_with_source(Context,Source),
%%    ok = cl:build_program(P, DeviceList, Options),
%%    {ok,DeviceList} = cl:get_program_info(P, devices),
%%    {ok,BinaryList} = cl:get_program_info(P, binaries).
%%  </code>
%%
-spec create_program_with_binary(Context::cl_context(),
				 DeviceList::[cl_device_id()],
				 BinaryList::[binary()]) ->
    {'ok', cl_program()} | {'error', cl_error()}.

create_program_with_binary(Context, DeviceList, BinaryList) ->
    DeviceData = encode_pointer_array(DeviceList),
    BinaryData = encode_async_binary_array(BinaryList),
    %% also: async_create
    cl_drv:create(?ECL_CREATE_PROGRAM_WITH_BINARY,
		  ?ECL_RELEASE_PROGRAM,
		  [<<?pointer_t(Context),
		    DeviceData/binary>>,
		   BinaryData]).

%%
%% @spec retain_program(Program::cl_program()) ->
%%    'ok' | {'error', cl_error()}
%% @doc  Increments the program reference count. 
retain_program(Program) ->
    cl_drv:retain(?ECL_RETAIN_PROGRAM, Program).

%%
%% @spec release_program(Program::cl_program()) ->
%%    'ok' | {'error', cl_error()}
%% @doc Decrements the program reference count. 
%%
%% The program object is deleted after all kernel objects associated
%% with program have been deleted and the program reference count
%% becomes zero.
release_program(Program) ->
    cl_drv:release(?ECL_RELEASE_PROGRAM, Program).

%%
%% @spec build_program(Program::cl_program(),
%%                     DeviceList::[cl_device_id()],
%%                     Options::string()) ->
%%  'ok' | {'error', cl_error()}
%%
%% @doc Builds (compiles and links) a program executable from the
%% program source or binary.
%%
%% OpenCL allows program executables to be built using the source or the binary.
%% 
%% The build options are categorized as pre-processor options, options
%% for math intrinsics, options that control optimization and
%% miscellaneous options. This specification defines a standard set of
%% options that must be supported by an OpenCL compiler when building
%% program executables online or offline. These may be extended by a
%% set of vendor- or platform-specific options.
%% 
%% <h4>Preprocessor Options</h4> These options
%% control the OpenCL preprocessor which is run on each program source
%% before actual compilation. -D options are processed in the order
%% they are given in the options argument to
%% <code>build_program/3</code>.
%%
%% <dl>
%% <dt><span>-D name</span></dt><dd>
%% <p> Predefine <code>name</code> as a macro, with definition 1.</p></dd>
%% <dt>-D name=definition</dt><dd><p> The contents of <code>definition</code> 
%% are tokenized and processed as if they appeared during translation phase three in a `#define'
%% directive. In particular, the definition will be truncated by
%% embedded newline characters.  </p></dd>
%% <dt>-I dir</dt><dd> <p>Add the directory <code>dir</code> to the list of directories to be
%% searched for header files.</p> </dd></dl>
%% <br />
%%
%% <h4>Math Intrinsics Options</h4> These options control compiler
%% behavior regarding floating-point arithmetic. These options trade
%% off between speed and correctness.
%% <dl><dt>-cl-single-precision-constant</dt><dd><p> Treat double
%% precision floating-point constant as single precision constant.
%% </p></dd><dt>-cl-denorms-are-zero</dt><dd><p> This option controls
%% how single precision and double precision denormalized numbers are
%% handled. If specified as a build option, the single precision
%% denormalized numbers may be flushed to zero and if the optional
%% extension for double precision is supported, double precision
%% denormalized numbers may also be flushed to zero. This is intended
%% to be a performance hint and the OpenCL compiler can choose not to
%% flush denorms to zero if the device supports single precision (or
%% double precision) denormalized numbers.  </p><p> This option is
%% ignored for single precision numbers if the device does not support
%% single precision denormalized numbers i.e. CL_FP_DENORM bit is not
%% set in CL_DEVICE_SINGLE_FP_CONFIG.  </p><p> </p><p> This option is
%% ignored for double precision numbers if the device does not support
%% double precision or if it does support double precison but
%% CL_FP_DENORM bit is not set in CL_DEVICE_DOUBLE_FP_CONFIG.  </p><p>
%% 
%% This flag only applies for scalar and vector single precision
%% floating-point variables and computations on these floating-point
%% variables inside a program. It does not apply to reading from or
%% writing to image objects.  </p><p> </p></dd></dl><p><br />
%% </p>
%%
%% <h4>Optimization Options</h4> These options control various
%% sorts of optimizations. Turning on optimization flags makes the
%% compiler attempt to improve the performance and/or code size at the
%% expense of compilation time and possibly the ability to debug the
%% program.  <dl><dt>-cl-opt-disable</dt><dd><p> This option
%% disables all optimizations. The default is optimizations are
%% enabled.  </p></dd><dt>-cl-strict-aliasing</dt><dd><p> This option
%% allows the compiler to assume the strictest aliasing rules.
%% </p></dd></dl>
%%<p> The following options control compiler
%% behavior regarding floating-point arithmetic. These options trade
%% off between performance and correctness and must be specifically
%% enabled. These options are not turned on by default since it can
%% result in incorrect output for programs which depend on an exact
%% implementation of IEEE 754 rules/specifications for math functions.
%% </p><dl><dt>-cl-mad-enable</dt><dd><p> Allow <code>a * b + c</code>
%% to be replaced by a <code>mad</code>. The <code>mad</code> computes
%% <code>a * b + c</code> with reduced accuracy. For example, some
%% OpenCL devices implement <code>mad</code> as truncate
%% the result of <code>a * b</code> before adding it to
%% <code>c</code>.  </p></dd>
%% <dt>-cl-no-signed-zeros</dt><dd>
%% <p> Allow optimizations for floating-point arithmetic that ignore
%% the signedness of zero. IEEE 754 arithmetic specifies the behavior
%% of distinct <code>+0.0</code> and <code>-0.0</code> values, which
%% then prohibits simplification of expressions such as
%% <code>x+0.0</code> or <code>0.0*x</code> (even with -clfinite-math
%% only). This option implies that the sign of a zero result isn't
%% significant.  </p></dd>
%% <dt>-cl-unsafe-math-optimizations</dt><dd><p> Allow optimizations
%% for floating-point arithmetic that (a) assume that arguments and
%% results are valid, (b) may violate IEEE 754 standard and (c) may
%% violate the OpenCL numerical compliance requirements as defined in
%% section 7.4 for single-precision floating-point, section 9.3.9 for
%% double-precision floating-point, and edge case behavior in section
%% 7.5. This option includes the -cl-no-signed-zeros and
%% -cl-mad-enable options.  </p></dd>
%%<dt><span class="term">-cl-finite-math-only</span></dt><dd><p> 
%% Allow optimizations for floating-point arithmetic that assume that arguments and results
%% are not NaNs or ±infinity. This option may violate the OpenCL numerical compliance
%% requirements defined in in section 7.4 for single-precision floating-point,
%% section 9.3.9 for double-precision floating-point, and edge case behavior in section 7.5.
%% </p></dd>
%%<dt><span class="term">-cl-fast-relaxed-math</span></dt><dd><p> 
%% Sets the optimization options -cl-finite-math-only and -cl-unsafe-math-optimizations.
%% This allows optimizations for floating-point arithmetic that may violate the IEEE 754
%% standard and the OpenCL numerical compliance requirements defined in the specification in section 7.4 for single-precision floating-point, section 9.3.9 for double-precision floating-point,
%% and edge case behavior in section 7.5. This option causes the preprocessor macro
%%
%% <code>__FAST_RELAXED_MATH__</code> to be defined in the OpenCL program.
%% </p></dd></dl><p><br />
%% </p><h4>Options to Request or Suppress Warnings</h4>
%% Warnings are diagnostic messages that report constructions which are not inherently erroneous
%% but which are risky or suggest there may have been an error. The following languageindependent
%% options do not enable specific warnings but control the kinds of diagnostics
%% produced by the OpenCL compiler.
%% <dl><dt><span class="term">-w</span></dt><dd><p> 
%% Inhibit all warning messages.
%% </p></dd><dt><span class="term">-Werror</span></dt><dd><p> 
%% Make all warnings into errors.
%% </p></dd>
%%</dl>
build_program(Program, DeviceList, Options) ->
    DevData = encode_pointer_array(DeviceList),
    cl_drv:call(?ECL_BUILD_PROGRAM,
		 <<?pointer_t(Program),
		  DevData/binary,
		  (list_to_binary([Options]))/binary>>).

%%
%% @spec unload_compiler() -> 'ok' | {'error', cl_error()}
%% @doc Allows the implementation to release the resources allocated by the OpenCL compiler. 
%%
%% This is a hint from the application and does not guarantee that the
%% compiler will not be used in the future or that the compiler will
%% actually be unloaded by the implementation. Calls to build_program/3
%% after unload_compiler/0 will reload the compiler, if necessary, to
%% build the appropriate program executable.
unload_compiler() ->		    
    cl_drv:call(?ECL_UNLOAD_COMPILER, <<>>).

program_info() ->
    program_info_keys().

%% @doc  Returns specific information about the program object. 
get_program_info(Program, Info) ->
    get_info(?ECL_GET_PROGRAM_INFO, Program, Info, fun program_info_map/1).

%% @doc  Returns all information about the program object. 
get_program_info(Program) ->
    get_info_list(?ECL_GET_PROGRAM_INFO, Program, 
		  program_info_keys(), fun program_info_map/1).

program_build_info() ->
    build_info_keys().

%% @doc Returns specific build information for each device in the program object. 
get_program_build_info(Program, Device, Info) ->
    get_info(?ECL_GET_PROGRAM_BUILD_INFO, Program, 
	     <<?pointer_t(Device)>>, Info, fun build_info_map/1).
%% @doc Returns all build information for each device in the program object. 
get_program_build_info(Program, Device) ->
    get_info_list(?ECL_GET_PROGRAM_BUILD_INFO, Program,
		  <<?pointer_t(Device)>>, 
		  build_info_keys(), fun build_info_map/1).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Kernel
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
%% @spec create_kernel(Program::cl_program(),Name::string()) ->
%%    {'ok', cl_kernel()} | {'error', cl_error()}
%%
%% @doc  Creates a kernal object. 
%%
%%  A kernel is a function declared in a program. A kernel is
%%  identified by the __kernel qualifier applied to any function in a
%%  program. A kernel object encapsulates the specific __kernel
%%  function declared in a program and the argument values to be used
%%  when executing this __kernel function.
create_kernel(Program, Name) ->
    NameBin = list_to_binary([Name]),
    NameLen = byte_size(NameBin),
    cl_drv:create(?ECL_CREATE_KERNEL,
		   ?ECL_RELEASE_KERNEL,
		  <<?pointer_t(Program),?size_t(NameLen),NameBin/binary>>).

%%
%% @spec create_kernels_in_program(Program::cl_program()) ->
%%    {'ok', [cl_kernel()]} | {'error', cl_error()}
%%
%% @doc Creates kernel objects for all kernel functions in a program object. 
%%
%%  Creates kernel objects for all kernel functions in program. Kernel
%%  objects are not created for any __kernel functions in program that
%%  do not have the same function definition across all devices for
%%  which a program executable has been successfully built.

%% Kernel objects can only be created once you have a program object
%% with a valid program source or binary loaded into the program
%% object and the program executable has been successfully built for
%% one or more devices associated with program. No changes to the
%% program executable are allowed while there are kernel objects
%% associated with a program object. This means that calls to
%% clBuildProgram return CL_INVALID_OPERATION if there are kernel
%% objects attached to a program object. The OpenCL context associated
%% with program will be the context associated with kernel. The list
%% of devices associated with program are the devices associated with
%% kernel. Devices associated with a program object for which a valid
%% program executable has been built can be used to execute kernels
%% declared in the program object.
create_kernels_in_program(Program) ->
    cl_drv:create(?ECL_CREATE_KERNELS_IN_PROGRAM, 
		   ?ECL_RELEASE_KERNEL,
		   <<?pointer_t(Program)>>).

%%
%% @type cl_kernel_arg() = integer() | float() | binary()
%%
%% @spec set_kernel_arg(Kernel::cl_kernel(), Index::non_neg_integer(),
%%                      Argument::cl_kernel_arg()) -> 
%%    'ok' | {'error', cl_error()}
%% @doc Used to set the argument value for a specific argument of a kernel. 
%% 
%% For now set_kernel_arg handles integer and floats
%% to set any other type use `<<Foo:Bar/native...>>'
%% use the macros defined in cl.hrl to get it right (except for padding)
%% 
%% A kernel object does not update the reference count for objects
%% such as memory, sampler objects specified as argument values by
%% set_kernel_arg/3, Users may not rely on a kernel object to retain
%% objects specified as argument values to the kernel.
%%
%% Implementations shall not allow cl_kernel objects to hold reference
%% counts to cl_kernel arguments, because no mechanism is provided for
%% the user to tell the kernel to release that ownership right. If the
%% kernel holds ownership rights on kernel args, that would make it
%% impossible for the user to tell with certainty when he may safely
%% release user allocated resources associated with OpenCL objects
%% such as the cl_mem backing store used with CL_MEM_USE_HOST_PTR.

set_kernel_arg(Kernel,Index,{'pointer',Ptr}) ->
    cl_drv:call(?ECL_SET_KERNEL_ARG_POINTER_T,
		<<?pointer_t(Kernel),
		  ?u_int32_t(Index),
		  ?u_int32_t(8),
		  ?pointer_t(Ptr)>>);
set_kernel_arg(Kernel,Index,{'size',Sz}) ->
    cl_drv:call(?ECL_SET_KERNEL_ARG_SIZE_T,
		<<?pointer_t(Kernel),
		  ?u_int32_t(Index),
		  ?u_int32_t(8),
		  ?size_t(Sz)>>);
set_kernel_arg(Kernel,Index,Argument) ->
    Arg = encode_argument(Argument),
    Size  = byte_size(Arg),
    cl_drv:call(?ECL_SET_KERNEL_ARG,
		<<?pointer_t(Kernel),
		 ?u_int32_t(Index),
		 ?u_int32_t(Size),
		 Arg/binary>>).

%%
%% @spec set_kernel_arg_size(Kernel::cl_kernel(), Index::non_neg_integer(),
%%                           Size::non_neg_integer()) ->
%%    'ok' | {'error', cl_error()}
%%
%% @doc clErlang special to set kernel arg with size only (local mem etc)
%%
set_kernel_arg_size(Kernel,Index,Size) ->
    cl_drv:call(?ECL_SET_KERNEL_ARG,
		<<?pointer_t(Kernel),
		 ?u_int32_t(Index),
		 ?u_int32_t(Size)>>).

%%
%% @spec retain_kernel(Context::cl_kernel()) ->
%%    'ok' | {'error', cl_error()}
%% @doc  Increments the program kernel reference count. 
retain_kernel(Kernel) ->
    cl_drv:retain(?ECL_RETAIN_KERNEL, Kernel).

%%
%% @spec release_kernel(Context::cl_kernel()) ->
%%    'ok' | {'error', cl_error()}
%% @doc  Decrements the kernel reference count. 
release_kernel(Kernel) ->
    cl_drv:release(?ECL_RELEASE_KERNEL, Kernel).

kernel_info() ->
    kernel_info_keys().

%% @doc Returns specific information about the kernel object. 
get_kernel_info(Kernel, Info) ->
    get_info(?ECL_GET_KERNEL_INFO, Kernel, Info, fun kernel_info_map/1).

%% @doc Returns all information about the kernel object. 
get_kernel_info(Kernel) ->
    get_info_list(?ECL_GET_KERNEL_INFO, Kernel, 
		  kernel_info_keys(), fun kernel_info_map/1).

kernel_workgroup_info() ->
    workgroup_info_keys().

%% @doc Returns specific information about the kernel object that may
%% be specific to a device.
get_kernel_workgroup_info(Kernel, Device, Info) ->
    get_info(?ECL_GET_KERNEL_WORKGROUP_INFO, Kernel,
	     <<?pointer_t(Device)>>,  Info, fun workgroup_info_map/1).

%% @doc Returns all information about the kernel object that may be
%% specific to a device.
get_kernel_workgroup_info(Kernel, Device) ->
    get_info_list(?ECL_GET_KERNEL_WORKGROUP_INFO, Kernel, 
		  <<?pointer_t(Device)>>, 
		  workgroup_info_keys(), fun workgroup_info_map/1).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Events
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% @spec enqueue_task(Queue::cl_queue(), Kernel::cl_kernel(),
%%                    WaitList::[cl_event()]) ->
%%    {'ok', cl_event()} | {'error', cl_error()}
%%
%% @doc Enqueues a command to execute a kernel on a device. 
%%
%% The kernel is executed using a single work-item.
%% @see enqueue_nd_range_kernel/5
-spec enqueue_task(Queue::cl_queue(), Kernel::cl_kernel(),
		   WaitList::[cl_event()]) ->
    {'ok', cl_event()} | {'error', cl_error()}.

enqueue_task(Queue, Kernel, WaitList) ->
    EventData = encode_pointer_array(WaitList),
    cl_drv:create(?ECL_ENQUEUE_TASK,
		   ?ECL_RELEASE_EVENT,
		   <<?pointer_t(Queue), ?pointer_t(Kernel),
		    EventData/binary>>).

%%
%% @spec enqueue_nd_range_kernel(Queue::cl_queue(), Kernel::cl_kernel(),
%%                               Global::[non_neg_integer()],
%%                               Local::[non_neg_integer()],
%%                               WaitList::[cl_event()]) ->
%%    {'ok', cl_event()} | {'error', cl_error()}
%%
%% @doc Enqueues a command to execute a kernel on a device. 
%% 
%% Work-group instances are executed in parallel across multiple
%% compute units or concurrently on the same compute unit.
%%
%% Each work-item is uniquely identified by a global identifier. The
%% global ID, which can be read inside the kernel, is computed using
%% the value given by global_work_size and global_work_offset. In
%% OpenCL 1.0, the starting global ID is always (0, 0, ... 0). In
%% addition, a work-item is also identified within a work-group by a
%% unique local ID. The local ID, which can also be read by the
%% kernel, is computed using the value given by local_work_size. The
%% starting local ID is always (0, 0, ... 0).
-spec enqueue_nd_range_kernel(Queue::cl_queue(), Kernel::cl_kernel(),
			      Global::[non_neg_integer()],
			      Local::[non_neg_integer()],
			      WaitList::[cl_event()]) ->
    {'ok', cl_event()} | {'error', cl_error()}.


enqueue_nd_range_kernel(Queue, Kernel, Global, Local, WaitList) ->
    WorkDim = length(Global),
    WorkDim = length(Local),
    EventData = encode_pointer_array(WaitList),
    cl_drv:create(?ECL_ENQUEUE_ND_RANGE_KERNEL,
		  ?ECL_RELEASE_EVENT,
		  <<?pointer_t(Queue), ?pointer_t(Kernel),
		   ?size_t(WorkDim),
		   (<< <<?size_t(G)>> || G <- Global >>)/binary,
		   (<< <<?size_t(L)>> || L <- Local >>)/binary,
		   EventData/binary>>).

%% @spec enqueue_marker(Queue::cl_queue()) ->
%%    {'ok', cl_event()} | {'error', cl_error()}
%%
%% @doc  Enqueues a marker command. 
%%
%%  Enqueues a marker command to command_queue. The marker command
%%  returns an event which can be used to queue a wait on this marker
%%  event i.e. wait for all commands queued before the marker command
%%  to complete.
-spec enqueue_marker(Queue::cl_queue()) ->
    {'ok', cl_event()} | {'error', cl_error()}.

enqueue_marker(Queue) ->
    cl_drv:create(?ECL_ENQUEUE_MARKER,
		  ?ECL_RELEASE_EVENT,
		  <<?pointer_t(Queue)>>).

%%
%% @spec enqueue_wait_for_event(Queue::cl_queue(), WaitList::[cl_event()]) ->
%%    {'ok', cl_event()} | {'error', cl_error()}
%%
%% @doc Enqueues a wait for a specific event or a list of events 
%% to complete before any future commands queued in the command-queue are
%% executed.
%%
%% The context associated with events in WaitList and Queue must be the same. 
-spec enqueue_wait_for_event(Queue::cl_queue(),  WaitList::[cl_event()]) ->
    {'ok', cl_event()} | {'error', cl_error()}.

enqueue_wait_for_event(Queue, WaitList) ->
    EventData = encode_pointer_array(WaitList),
    cl_drv:call(?ECL_ENQUEUE_WAIT_FOR_EVENT,
		 <<?pointer_t(Queue), EventData/binary>>).

%%
%% @spec enqueue_read_buffer(Queue::cl_queue(), Buffer::cl_mem(),
%%                           Offset::non_neg_integer(), 
%%                           Size::non_neg_integer(), 
%%                           WaitList::[cl_event()]) ->
%%    {'ok', cl_event()} | {'error', cl_error()}
%%
%% @doc Enqueue commands to read from a buffer object to host memory. 
%% 
%% Calling <code>enqueue_read_buffer</code> to read a region of the
%% buffer object with the <code>Buffer</code> argument value set to
%% <code>host_ptr</code> + <code >offset</code>, where
%% <code>host_ptr</code> is a pointer to the memory region specified
%% when the buffer object being read is created with
%% <code>CL_MEM_USE_HOST_PTR</code>, must meet the following
%% requirements in order to avoid undefined behavior:
%%
%% <ul> <li>All commands that use this buffer object have finished
%% execution before the read command begins execution</li>
%% <li>The buffer object is not mapped</li>
%% <li>The buffer object is not used by any command-queue until the
%% read command has finished execution</li>
%% </ul>

-spec enqueue_read_buffer(Queue::cl_queue(), Buffer::cl_mem(),
			  Offset::non_neg_integer(), 
			  Size::non_neg_integer(), 
			  WaitList::[cl_event()]) ->
    {'ok', cl_event()} | {'error', cl_error()}.

enqueue_read_buffer(Queue, Buffer, Offset, Size, WaitList) ->
    EventData = encode_pointer_array(WaitList),
    cl_drv:create(?ECL_ENQUEUE_READ_BUFFER,
		  ?ECL_RELEASE_EVENT,
		  <<?pointer_t(Queue), ?pointer_t(Buffer),
		   ?u_int32_t(Offset), ?u_int32_t(Size), EventData/binary>>).

%%
%% @spec enqueue_write_buffer(Queue::cl_queue(), Buffer::cl_mem(),
%%                            Offset::non_neg_integer(), 
%%                            Size::non_neg_integer(), 
%%                            Data::binary(),
%%                            WaitList::[cl_event()]) ->
%%    {'ok', cl_event()} | {'error', cl_error()}
%%
%% @doc Enqueue commands to write to a buffer object from host memory. 
%% 
%% Calling <code>enqueue_write_buffer</code> to update the latest bits
%% in a region of the buffer object with the <code>Buffer</code>
%% argument value set to <code>host_ptr</code> + <code >offset</code>,
%% where <code>host_ptr</code> is a pointer to the memory region
%% specified when the buffer object being read is created with
%% <code>CL_MEM_USE_HOST_PTR</code>, must meet the following
%% requirements in order to avoid undefined behavior:
%%
%% <ul> <li>The host memory region given by <code>(host_ptr + offset, cb)</code>
%% contains the latest bits when the enqueued write command begins
%% execution. </li> 
%% <li>The buffer object is not mapped</li> 
%% <li>The buffer object is not used by any command-queue until the read
%% command has finished execution</li> </ul>
-spec enqueue_write_buffer(Queue::cl_queue(), Buffer::cl_mem(),
			   Offset::non_neg_integer(), 
			   Size::non_neg_integer(), 
			   Data::binary(),
			   WaitList::[cl_event()]) ->
    {'ok', cl_event()} | {'error', cl_error()}.


enqueue_write_buffer(Queue, Buffer, Offset, Size, Data, WaitList) ->
    EventData = encode_pointer_array(WaitList),
    %% also: async_create
    cl_drv:create(?ECL_ENQUEUE_WRITE_BUFFER,
		  ?ECL_RELEASE_EVENT,
		  <<?pointer_t(Queue), ?pointer_t(Buffer),
		   ?u_int32_t(Offset), ?u_int32_t(Size), EventData/binary,
		   Data/binary>>).

%% 
%% @spec enqueue_barrier(Queue::cl_queue()) ->
%%    'ok' | {'error', cl_error()}
%%
%% @doc A synchronization point that enqueues a barrier operation. 
%%
%%  enqueue_barrier/1 is a synchronization point that ensures that all
%%  queued commands in command_queue have finished execution before
%%  the next batch of commands can begin execution.
-spec enqueue_barrier(Queue::cl_queue()) ->
    'ok' | {'error', cl_error()}.

enqueue_barrier(Queue) ->
    cl_drv:call(?ECL_ENQUEUE_BARRIER,
		 <<?pointer_t(Queue)>>).

%%
%% @spec flush(Queue::cl_queue()) ->
%%    'ok' | {'error', cl_error()}
%%
%% @doc Issues all previously queued OpenCL commands 
%% in a command-queue to the device associated with the command-queue.
%%
%% flush only guarantees that all queued commands to command_queue get
%% issued to the appropriate device. There is no guarantee that they
%% will be complete after clFlush returns.
-spec flush(Queue::cl_queue()) ->
    'ok' | {'error', cl_error()}.

flush(Queue) ->
    cl_drv:call(?ECL_FLUSH, <<?pointer_t(Queue)>>).

%%
%% @spec finish(Queue::cl_queue()) ->
%%    'ok' | {'error', cl_error()}
%%
%% @doc Blocks until all previously queued OpenCL commands 
%% in a command-queue are issued to the associated device and have
%% completed.
%%
%% finish does not return until all queued commands in command_queue
%% have been processed and completed. clFinish is also a
%% synchronization point.
-spec finish(Queue::cl_queue()) ->
    'ok' | {'error', cl_error()}.

finish(Queue) ->
    cl_drv:call(?ECL_FINISH, <<?pointer_t(Queue)>>).

%%
%% @spec retain_event(Event::cl_event()) ->
%%    'ok' | {'error', cl_error()}
%% @doc  Increments the event reference count. 
%% NOTE: The OpenCL commands that return an event perform an implicit retain. 
retain_event(Event) ->
    cl_drv:retain(?ECL_RETAIN_EVENT, Event).

%%
%% @spec release_event(Event::cl_event()) ->
%%    'ok' | {'error', cl_error()}
%% @doc Decrements the event reference count. 
%%
%%  Decrements the event reference count. The event object is deleted
%%  once the reference count becomes zero, the specific command
%%  identified by this event has completed (or terminated) and there
%%  are no commands in the command-queues of a context that require a
%%  wait for this event to complete.
release_event(Event) ->
    cl_drv:release(?ECL_RELEASE_EVENT, Event).

%% @doc Returns all possible event_info items.
event_info() ->
    event_info_keys().

%% @doc Returns specific information about the event object. 
get_event_info(Event, Info) ->
    get_info(?ECL_GET_EVENT_INFO, Event, Info, fun event_info_map/1).

%% @doc Returns all specific information about the event object. 
get_event_info(Event) ->
    get_info_list(?ECL_GET_EVENT_INFO, Event, 
		  event_info_keys(), fun event_info_map/1).

%% @type timeout() = non_neg_integer() | 'infinity'
%%
%% @spec wait(Event::cl_event) -> 
%%    {'ok','completed'} | {'ok',Binary} | {'error',cl_error()}
%% @equiv wait(Event, infinity)
%%
wait(Event) ->
    wait(Event, infinity).

%%  
%% @spec wait(Event::cl_event, Timeout::timeout()) -> 
%%    {'ok','completed'} | {'ok',Binary} | 
%%    {'error',cl_error()} | {'error',timeout}
%% 
%%
%% @doc  Waits on the host thread for commands identified by event objects to complete. 
%%
%%  Waits on the host thread for commands identified by event objects
%%  in event_list to complete. A command is considered complete if its
%%  execution status is CL_COMPLETE or a negative value.
wait(Event, Timeout) ->
    receive
	{cl_event, Event, Binary} when is_binary(Binary) ->
	    release_event(Event),
	    {ok,Binary};
	{cl_event, Event, complete} ->
	    release_event(Event),
	    {ok,completed};
	{cl_event, Event, Err} ->
	    release_event(Event),
	    {error, Err}
    after Timeout ->
	    {error, timeout}
    end.

%% @hidden
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Utilities
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
get_info(Command, ID, Info, Map) ->
    get_info(Command, ID, <<>>, Info, Map).

get_info(Command, ID, Arg, Info, Map) ->
    InfoID = Map(Info),
    cl_drv:call(Command, <<?pointer_t(ID),Arg/binary,?u_int32_t(InfoID)>>).

%% @hidden
get_info_list(InfoCommand, ID, Keys, Map) ->
    get_info_list(InfoCommand, ID, <<>>, Keys, Map).

get_info_list(InfoCommand, ID, Arg, Keys, Map) ->
    get_info_list(InfoCommand, <<?pointer_t(ID),Arg/binary>>, Keys, Map,
		  [], ok).

get_info_list(Command, Arg, [K|Ks], Map, Acc, _Err) ->
    V = Map(K),    
    case cl_drv:call(Command, <<Arg/binary,?u_int32_t(V)>>) of
	{ok, Value} -> 
	    get_info_list(Command, Arg, Ks, Map, [{K,Value}|Acc], ok);
	Error ->
	    io:format("InfoError: ~s [~p]\n", [K,Error]),
	    get_info_list(Command, Arg, Ks, Map, Acc, Error)
    end;
get_info_list(_Command, _Arg, [], _Map, [], Error) ->
    Error;
get_info_list(_Command, _Arg, [], _Map, Acc, _Error) ->
    {ok, reverse(Acc)}.

%% @hidden
encode_mem_flag(read_write) -> ?ECL_MEM_READ_WRITE;
encode_mem_flag(write_only) -> ?ECL_MEM_WRITE_ONLY;
encode_mem_flag(read_only) -> ?ECL_MEM_READ_ONLY;
encode_mem_flag(use_host_ptr) -> ?ECL_MEM_USE_HOST_PTR;
encode_mem_flag(alloc_host_ptr) -> ?ECL_MEM_ALLOC_HOST_PTR;
encode_mem_flag(copy_host_ptr) -> ?ECL_MEM_COPY_HOST_PTR.
%% @hidden
encode_mem_flags([T|Ts]) ->
    encode_mem_flag(T) bor encode_mem_flags(Ts);
encode_mem_flags([]) ->
    0.

%% @hidden
encode_queue_property(out_of_order_exec_mode_enable) ->
    ?ECL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
encode_queue_property(profiling_enabled) ->
    ?ECL_QUEUE_PROFILING_ENABLE.
%% @hidden
encode_queue_properties([T|Ts]) ->
    encode_queue_property(T) bor encode_queue_properties(Ts);
encode_queue_properties([]) ->
    0.

%%
%% Encode kernel argument
%% FIXME: check pointers etc since OpenCL will crash if we 
%%        send bad pointers!
%%
%% @hidden
encode_argument(X) when is_integer(X) -> <<?cl_int(X)>>;
encode_argument(X) when is_float(X)   -> <<?cl_float(X)>>;
encode_argument(X) when is_list(X)    -> list_to_binary(X);
encode_argument(X) when is_binary(X)  -> X;
encode_argument({'char',X}) -> <<?cl_char(X)>>;
encode_argument({'uchar',X}) -> <<?cl_uchar(X)>>;
encode_argument({'short',X}) -> <<?cl_short(X)>>;
encode_argument({'ushort',X}) -> <<?cl_ushort(X)>>;
encode_argument({'int',X}) -> <<?cl_int(X)>>;
encode_argument({'uint',X}) -> <<?cl_uint(X)>>;
encode_argument({'long',X}) -> <<?cl_long(X)>>;
encode_argument({'ulong',X}) -> <<?cl_ulong(X)>>;
encode_argument({'half',X}) -> <<?cl_half(X)>>;
encode_argument({'float',X}) -> <<?cl_float(X)>>;
encode_argument({'double',X}) -> <<?cl_double(X)>>;

encode_argument({'char2',{X1,X2}}) ->
    <<?cl_char2(X1,X2)>>;    
encode_argument({'char4',{X1,X2,X3,X4}}) ->
    <<?cl_char4(X1,X2,X3,X4)>>;
encode_argument({'char8',{X1,X2,X3,X4,X5,X6,X7,X8}}) ->
    <<?cl_char8(X1,X2,X3,X4,X5,X6,X7,X8)>>;
encode_argument({'char16',{X1,X2,X3,X4,X5,X6,X7,X8,
			   X9,X10,X11,X12,X13,X14,X15,X16}}) ->
    <<?cl_char16(X1,X2,X3,X4,X5,X6,X7,X8,
		 X9,X10,X11,X12,X13,X14,X15,X16)>>;
encode_argument({'uchar2',{X1,X2}}) ->
    <<?cl_uchar2(X1,X2)>>;    
encode_argument({'uchar4',{X1,X2,X3,X4}}) ->
    <<?cl_uchar4(X1,X2,X3,X4)>>;
encode_argument({'uchar8',{X1,X2,X3,X4,X5,X6,X7,X8}}) ->
    <<?cl_uchar8(X1,X2,X3,X4,X5,X6,X7,X8)>>;
encode_argument({'uchar16',{X1,X2,X3,X4,X5,X6,X7,X8,
			   X9,X10,X11,X12,X13,X14,X15,X16}}) ->
    <<?cl_uchar16(X1,X2,X3,X4,X5,X6,X7,X8,
		 X9,X10,X11,X12,X13,X14,X15,X16)>>;

encode_argument({'short2',{X1,X2}}) ->
    <<?cl_short2(X1,X2)>>;    
encode_argument({'short4',{X1,X2,X3,X4}}) ->
    <<?cl_short4(X1,X2,X3,X4)>>;
encode_argument({'short8',{X1,X2,X3,X4,X5,X6,X7,X8}}) ->
    <<?cl_short8(X1,X2,X3,X4,X5,X6,X7,X8)>>;
encode_argument({'short16',{X1,X2,X3,X4,X5,X6,X7,X8,
			   X9,X10,X11,X12,X13,X14,X15,X16}}) ->
    <<?cl_short16(X1,X2,X3,X4,X5,X6,X7,X8,
		 X9,X10,X11,X12,X13,X14,X15,X16)>>;
encode_argument({'ushort2',{X1,X2}}) ->
    <<?cl_ushort2(X1,X2)>>;    
encode_argument({'ushort4',{X1,X2,X3,X4}}) ->
    <<?cl_ushort4(X1,X2,X3,X4)>>;
encode_argument({'ushort8',{X1,X2,X3,X4,X5,X6,X7,X8}}) ->
    <<?cl_ushort8(X1,X2,X3,X4,X5,X6,X7,X8)>>;
encode_argument({'ushort16',{X1,X2,X3,X4,X5,X6,X7,X8,
			   X9,X10,X11,X12,X13,X14,X15,X16}}) ->
    <<?cl_ushort16(X1,X2,X3,X4,X5,X6,X7,X8,
		 X9,X10,X11,X12,X13,X14,X15,X16)>>;


encode_argument({'int2',{X1,X2}}) ->
    <<?cl_int2(X1,X2)>>;    
encode_argument({'int4',{X1,X2,X3,X4}}) ->
    <<?cl_int4(X1,X2,X3,X4)>>;
encode_argument({'int8',{X1,X2,X3,X4,X5,X6,X7,X8}}) ->
    <<?cl_int8(X1,X2,X3,X4,X5,X6,X7,X8)>>;
encode_argument({'int16',{X1,X2,X3,X4,X5,X6,X7,X8,
			   X9,X10,X11,X12,X13,X14,X15,X16}}) ->
    <<?cl_int16(X1,X2,X3,X4,X5,X6,X7,X8,
		 X9,X10,X11,X12,X13,X14,X15,X16)>>;
encode_argument({'uint2',{X1,X2}}) ->
    <<?cl_uint2(X1,X2)>>;    
encode_argument({'uint4',{X1,X2,X3,X4}}) ->
    <<?cl_uint4(X1,X2,X3,X4)>>;
encode_argument({'uint8',{X1,X2,X3,X4,X5,X6,X7,X8}}) ->
    <<?cl_uint8(X1,X2,X3,X4,X5,X6,X7,X8)>>;
encode_argument({'uint16',{X1,X2,X3,X4,X5,X6,X7,X8,
			   X9,X10,X11,X12,X13,X14,X15,X16}}) ->
    <<?cl_uint16(X1,X2,X3,X4,X5,X6,X7,X8,
		 X9,X10,X11,X12,X13,X14,X15,X16)>>;

encode_argument({'long2',{X1,X2}}) ->
    <<?cl_long2(X1,X2)>>;    
encode_argument({'long4',{X1,X2,X3,X4}}) ->
    <<?cl_long4(X1,X2,X3,X4)>>;
encode_argument({'long8',{X1,X2,X3,X4,X5,X6,X7,X8}}) ->
    <<?cl_long8(X1,X2,X3,X4,X5,X6,X7,X8)>>;
encode_argument({'long16',{X1,X2,X3,X4,X5,X6,X7,X8,
			   X9,X10,X11,X12,X13,X14,X15,X16}}) ->
    <<?cl_long16(X1,X2,X3,X4,X5,X6,X7,X8,
		 X9,X10,X11,X12,X13,X14,X15,X16)>>;
encode_argument({'ulong2',{X1,X2}}) ->
    <<?cl_ulong2(X1,X2)>>;    
encode_argument({'ulong4',{X1,X2,X3,X4}}) ->
    <<?cl_ulong4(X1,X2,X3,X4)>>;
encode_argument({'ulong8',{X1,X2,X3,X4,X5,X6,X7,X8}}) ->
    <<?cl_ulong8(X1,X2,X3,X4,X5,X6,X7,X8)>>;
encode_argument({'ulong16',{X1,X2,X3,X4,X5,X6,X7,X8,
			   X9,X10,X11,X12,X13,X14,X15,X16}}) ->
    <<?cl_ulong16(X1,X2,X3,X4,X5,X6,X7,X8,
		 X9,X10,X11,X12,X13,X14,X15,X16)>>;

encode_argument({'float2',{X1,X2}}) ->
    <<?cl_float2(X1,X2)>>;    
encode_argument({'float4',{X1,X2,X3,X4}}) ->
    <<?cl_float4(X1,X2,X3,X4)>>;
encode_argument({'float8',{X1,X2,X3,X4,X5,X6,X7,X8}}) ->
    <<?cl_float8(X1,X2,X3,X4,X5,X6,X7,X8)>>;
encode_argument({'float16',{X1,X2,X3,X4,X5,X6,X7,X8,
			   X9,X10,X11,X12,X13,X14,X15,X16}}) ->
    <<?cl_float16(X1,X2,X3,X4,X5,X6,X7,X8,
		 X9,X10,X11,X12,X13,X14,X15,X16)>>.


%%
%% Encode pointer array <<N:32, Ptr1:Ptr, ... PtrN:Ptr>>
%%
%% @hidden
encode_pointer_array(Pointers) when is_list(Pointers) ->
    N = length(Pointers),
    <<?size_t(N), (<< <<?pointer_t(Ptr) >> || Ptr <- Pointers >>)/binary>>.

%%
%% Encode binary array <<N:32, Size1:Size, Binary1:Size1/binary ... >>
%%
%% @hidden
encode_async_binary_array(Binaries) when is_list(Binaries) ->
    N = length(Binaries),
    [<<?size_t(N)>> | 
     lists:map(fun(Bin) -> [<<?size_t((byte_size(Bin)))>>,Bin] end, 
	       Binaries)].

%% @hidden
%% boolean - passed as uint32
encode_bool(true) -> 1;
encode_bool(false) -> 0.

%% @hidden
%% addressing_mode - enum
encode_addressing_mode(Mode) ->
    addressing_mode_map(Mode).

%% @hidden
encode_filter_mode(Mode) ->
    filter_mode_map(Mode).

%% @hidden
%% device_type - bitfield
encode_device_type(cpu) -> ?ECL_DEVICE_TYPE_CPU;
encode_device_type(gpu) -> ?ECL_DEVICE_TYPE_GPU;
encode_device_type(accelerator) -> ?ECL_DEVICE_TYPE_ACCELERATOR;
encode_device_type(all) -> ?ECL_DEVICE_TYPE_ALL;
encode_device_type(default) -> ?ECL_DEVICE_TYPE_DEFAULT.

%% @hidden
encode_device_types(T) when is_atom(T) ->
    encode_device_type(T);    
encode_device_types([T|Ts]) when is_list(Ts) ->
    encode_device_type(T) bor encode_device_types(Ts);
encode_device_types([]) ->
    0.

%% @hidden
device_info_keys() ->
    [
	type,
	vendor_id,
	max_compute_units,
	max_work_item_dimensions,
	max_work_group_size,
	max_work_item_sizes,
	preferred_vector_width_char,
	preferred_vector_width_short,
	preferred_vector_width_int,
	preferred_vector_width_long,
	preferred_vector_width_float,
	preferred_vector_width_double,
	max_clock_frequency,
	address_bits,
	max_read_image_args,
	max_write_image_args,
	max_mem_alloc_size,
	image2d_max_width,
	image2d_max_height,
	image3d_max_width,
	image3d_max_height,
	image3d_max_depth,
	image_support,
	max_parameter_size,
	max_samplers,
	mem_base_addr_align,
	min_data_type_align_size,
	single_fp_config,
	global_mem_cache_type,
	global_mem_cacheline_size,
	global_mem_cache_size,
	global_mem_size,
	max_constant_buffer_size,
	max_constant_args,
	local_mem_type,
	local_mem_size,
	error_correction_support,
	profiling_timer_resolution,
	endian_little,
	available,
	compiler_available,
	execution_capabilities,
	queue_properties,
	name,
	vendor,
	driver_version,
	profile,
	version,
	extensions,
	platform].
%% @hidden
device_info_map(Key) ->
    case Key of
	type -> ?ECL_DEVICE_TYPE;
	vendor_id -> ?ECL_DEVICE_VENDOR_ID;
	max_compute_units -> ?ECL_DEVICE_MAX_COMPUTE_UNITS;
	max_work_item_dimensions -> ?ECL_DEVICE_MAX_WORK_ITEM_DIMENSIONS;
	max_work_group_size -> ?ECL_DEVICE_MAX_WORK_GROUP_SIZE;
	max_work_item_sizes -> ?ECL_DEVICE_MAX_WORK_ITEM_SIZES;
	preferred_vector_width_char -> ?ECL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR;
	preferred_vector_width_short -> ?ECL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT;
	preferred_vector_width_int -> ?ECL_DEVICE_PREFERRED_VECTOR_WIDTH_INT;
	preferred_vector_width_long -> ?ECL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG;
	preferred_vector_width_float -> ?ECL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT;
	preferred_vector_width_double -> ?ECL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE;
	max_clock_frequency -> ?ECL_DEVICE_MAX_CLOCK_FREQUENCY;
	address_bits -> ?ECL_DEVICE_ADDRESS_BITS;
	max_read_image_args -> ?ECL_DEVICE_MAX_READ_IMAGE_ARGS;
	max_write_image_args -> ?ECL_DEVICE_MAX_WRITE_IMAGE_ARGS;
	max_mem_alloc_size -> ?ECL_DEVICE_MAX_MEM_ALLOC_SIZE;
	image2d_max_width -> ?ECL_DEVICE_IMAGE2D_MAX_WIDTH;
	image2d_max_height -> ?ECL_DEVICE_IMAGE2D_MAX_HEIGHT;
	image3d_max_width -> ?ECL_DEVICE_IMAGE3D_MAX_WIDTH;
	image3d_max_height -> ?ECL_DEVICE_IMAGE3D_MAX_HEIGHT;
	image3d_max_depth -> ?ECL_DEVICE_IMAGE3D_MAX_DEPTH;
	image_support -> ?ECL_DEVICE_IMAGE_SUPPORT;
	max_parameter_size -> ?ECL_DEVICE_MAX_PARAMETER_SIZE;
	max_samplers -> ?ECL_DEVICE_MAX_SAMPLERS;
	mem_base_addr_align -> ?ECL_DEVICE_MEM_BASE_ADDR_ALIGN;
	min_data_type_align_size -> ?ECL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE;
	single_fp_config -> ?ECL_DEVICE_SINGLE_FP_CONFIG;
	global_mem_cache_type -> ?ECL_DEVICE_GLOBAL_MEM_CACHE_TYPE;
	global_mem_cacheline_size -> ?ECL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE;
	global_mem_cache_size -> ?ECL_DEVICE_GLOBAL_MEM_CACHE_SIZE;
	global_mem_size -> ?ECL_DEVICE_GLOBAL_MEM_SIZE;
	max_constant_buffer_size -> ?ECL_DEVICE_MAX_CONSTANT_BUFFER_SIZE;
	max_constant_args -> ?ECL_DEVICE_MAX_CONSTANT_ARGS;
	local_mem_type -> ?ECL_DEVICE_LOCAL_MEM_TYPE;
	local_mem_size -> ?ECL_DEVICE_LOCAL_MEM_SIZE;
	error_correction_support -> ?ECL_DEVICE_ERROR_CORRECTION_SUPPORT;
	profiling_timer_resolution -> ?ECL_DEVICE_PROFILING_TIMER_RESOLUTION;
	endian_little -> ?ECL_DEVICE_ENDIAN_LITTLE;
	available -> ?ECL_DEVICE_AVAILABLE;
	compiler_available -> ?ECL_DEVICE_COMPILER_AVAILABLE;
	execution_capabilities -> ?ECL_DEVICE_EXECUTION_CAPABILITIES;
	queue_properties -> ?ECL_DEVICE_QUEUE_PROPERTIES;
	name -> ?ECL_DEVICE_NAME;
	vendor -> ?ECL_DEVICE_VENDOR;
	driver_version -> ?ECL_DRIVER_VERSION;
	profile -> ?ECL_DEVICE_PROFILE;
	version -> ?ECL_DEVICE_VERSION;
	extensions -> ?ECL_DEVICE_EXTENSIONS;
	platform -> ?ECL_DEVICE_PLATFORM
    end.

%% @hidden
platform_info_keys() ->
    [ profile, version, name, vendor, extensions].

%% @hidden
platform_info_map(Key) ->
    case Key of
	profile -> ?ECL_PLATFORM_PROFILE;
	version -> ?ECL_PLATFORM_VERSION;
	name -> ?ECL_PLATFORM_NAME;
	vendor -> ?ECL_PLATFORM_VENDOR;
	extensions -> ?ECL_PLATFORM_EXTENSIONS
    end.

%% @hidden
context_info_keys() ->
    [ reference_count, devices, properties ].

%% @hidden
context_info_map(Key) ->
    case Key of
	reference_count -> ?ECL_CONTEXT_REFERENCE_COUNT;
	devices         -> ?ECL_CONTEXT_DEVICES;
	properties      -> ?ECL_CONTEXT_PROPERTIES
    end.

%% @hidden
queue_info_keys() ->
    [ context, device, reference_count, properties ].

%% @hidden
queue_info_map(Key) ->
    case Key of
	context         -> ?ECL_QUEUE_CONTEXT;
	device          -> ?ECL_QUEUE_DEVICE;
	reference_count -> ?ECL_QUEUE_REFERENCE_COUNT;
	properties      -> ?ECL_QUEUE_PROPERTIES
    end.

%% @hidden
mem_info_keys() ->
    [ object_type, flags, size, host_ptr, map_count,
      reference_count, context  ].

%% @hidden
mem_info_map(Key) ->
    case Key of
	object_type     -> ?ECL_MEM_TYPE;
	flags           -> ?ECL_MEM_FLAGS;
	size            -> ?ECL_MEM_SIZE;
	host_ptr        -> ?ECL_MEM_HOST_PTR;
	map_count       -> ?ECL_MEM_MAP_COUNT;
	reference_count -> ?ECL_MEM_REFERENCE_COUNT; 
	context         -> ?ECL_MEM_CONTEXT
    end.

%% @hidden
sampler_info_keys() ->
    [ reference_count, context, normalized_coords,
      addressing_mode, filter_mode ].

%% @hidden
sampler_info_map(Key) ->
    case Key of
	reference_count -> ?ECL_SAMPLER_REFERENCE_COUNT; 	    
	context -> ?ECL_SAMPLER_CONTEXT;
	normalized_coords -> ?ECL_SAMPLER_NORMALIZED_COORDS;
	addressing_mode -> ?ECL_SAMPLER_ADDRESSING_MODE;
	filter_mode -> ?ECL_SAMPLER_FILTER_MODE
    end.

%% @hidden
%% addressing_mode_keys() ->
%%    [ none, clamp_to_edge, clamp, repeat ].
addressing_mode_map(Key) ->
    case Key of
	none -> ?ECL_ADDRESS_NONE;
	clamp_to_edge -> ?ECL_ADDRESS_CLAMP_TO_EDGE;
	clamp -> ?ECL_ADDRESS_CLAMP;
	repeat -> ?ECL_ADDRESS_REPEAT
    end.

%% filter_mode_keys(Key) ->
%%     [ nearest, linear ].
%% @hidden
filter_mode_map(Key) ->
    case Key of
	nearest -> ?ECL_FILTER_NEAREST;
	linear  -> ?ECL_FILTER_LINEAR
    end.

%% @hidden
program_info_keys() ->
    [ reference_count, context, num_decices, devices,
      source, binary_sizes, binaries ].

%% @hidden
program_info_map(Key) ->
    case Key of
	reference_count -> ?ECL_PROGRAM_REFERENCE_COUNT;
	context -> ?ECL_PROGRAM_CONTEXT;
	num_decices -> ?ECL_PROGRAM_NUM_DEVICES;
	devices -> ?ECL_PROGRAM_DEVICES;
	source -> ?ECL_PROGRAM_SOURCE;
	binary_sizes -> ?ECL_PROGRAM_BINARY_SIZES; 
	binaries -> ?ECL_PROGRAM_BINARIES
    end.

%% @hidden
build_info_keys() ->
    [status, options, log ].

%% @hidden
build_info_map(Key) ->
    case Key of
	status -> ?ECL_PROGRAM_BUILD_STATUS;
	options -> ?ECL_PROGRAM_BUILD_OPTIONS;
	log -> ?ECL_PROGRAM_BUILD_LOG
    end.

%% @hidden
kernel_info_keys() ->
    [ function_name, num_args, reference_count,
      context, program ].

%% @hidden    
kernel_info_map(Key) ->
    case Key of
	function_name -> ?ECL_KERNEL_FUNCTION_NAME;
	num_args -> ?ECL_KERNEL_NUM_ARGS;
	reference_count -> ?ECL_KERNEL_REFERENCE_COUNT;
	context -> ?ECL_KERNEL_CONTEXT;
	program -> ?ECL_KERNEL_PROGRAM
    end.

%% @hidden
event_info_keys() ->
    [ command_queue, command_type, reference_count,
      execution_status ].

%% @hidden
event_info_map(Key) ->
    case Key of 
	command_queue -> ?ECL_EVENT_COMMAND_QUEUE;
	command_type  -> ?ECL_EVENT_COMMAND_TYPE;
	reference_count -> ?ECL_EVENT_REFERENCE_COUNT;
	execution_status -> ?ECL_EVENT_COMMAND_EXECUTION_STATUS
    end.

%% @hidden
workgroup_info_keys() ->
    [ work_group_size, compile_work_group_size,
      local_mem_size ].

%% @hidden	    
workgroup_info_map(Key) ->
    case Key of
	work_group_size -> ?ECL_KERNEL_WORK_GROUP_SIZE;
	compile_work_group_size -> ?ECL_KERNEL_COMPILE_WORK_GROUP_SIZE; 
	local_mem_size -> ?ECL_KERNEL_LOCAL_MEM_SIZE
    end.
