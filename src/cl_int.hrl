%%
%% ECL internal mappings
%%
-ifndef(__CL_INT_HRL__).
-define(__CL_INT_HRL__, true).

-define(CL_SERVER, ecl_serv).
-define(CL_PORT,   ecl_port).
-define(CL_REG,    ecl_reg).

-define(OK,             1).
-define(ERROR,          2).
-define(EVENT,          3).
-define(INT8,           4).
-define(UINT8,          5).
-define(INT16,          6).
-define(UINT16,         7).
-define(INT32,          8).
-define(UINT32,         9).
-define(INT64,          10).
-define(UINT64,         11).
-define(BOOLEAN,        12).
-define(FLOAT32,        13).
-define(FLOAT64,        14).
-define(STRING1,        15).
-define(STRING4,        16).
-define(ATOM,           17).
-define(BINARY,         18).
-define(LIST,           19).
-define(LIST_END,       20).
-define(TUPLE,          21).
-define(TUPLE_END,      22).
-define(ENUM,           23).
-define(BITFIELD,       24).
-define(HANDLE,         25).
-define(POINTER,        26).
-define(USIZE,          27).

%% object type codes
-define(PLATFORM_TYPE, 1).
-define(DEVICE_TYPE,   2).
-define(CONTEXT_TYPE,  3).
-define(QUEUE_TYPE,    4).
-define(MEM_TYPE,      5).
-define(SAMPLER_TYPE,  6).
-define(PROGRAM_TYPE,  7).
-define(KERNEL_TYPE,   8).
-define(EVENT_TYPE,    9).
%%
%% Commands
%%
-define(ECL_NOOP,                         16#01).
-define(ECL_GET_PLATFORM_IDS,             16#02).
-define(ECL_GET_DEVICE_IDS,               16#03).
-define(ECL_GET_PLATFORM_INFO,            16#04).
-define(ECL_GET_DEVICE_INFO,              16#05).
-define(ECL_CREATE_CONTEXT,               16#06).
-define(ECL_RELEASE_CONTEXT,              16#07).
-define(ECL_RETAIN_CONTEXT,               16#08).
-define(ECL_GET_CONTEXT_INFO,             16#09).
-define(ECL_CREATE_QUEUE,                 16#0A).
-define(ECL_RETAIN_QUEUE,                 16#0B).
-define(ECL_RELEASE_QUEUE,                16#0C).
-define(ECL_GET_QUEUE_INFO,               16#0D).
-define(ECL_SET_QUEUE_PROPERTY,           16#0E).
-define(ECL_CREATE_BUFFER,                16#0F).
-define(ECL_ENQUEUE_READ_BUFFER,          16#10).
-define(ECL_ENQUEUE_WRITE_BUFFER,         16#11).
-define(ECL_ENQUEUE_COPY_BUFFER,          16#12).
-define(ECL_RETAIN_MEM_OBJECT,            16#13).
-define(ECL_RELEASE_MEM_OBJECT,           16#14).
-define(ECL_CREATE_IMAGE2D,               16#15).
-define(ECL_CREATE_IMAGE3D,               16#16).
-define(ECL_GET_SUPPORTED_IMAGE_FORMATS,  16#17).
-define(ECL_ENQUEUE_READ_IMAGE,           16#18).
-define(ECL_ENQUEUE_WRITE_IMAGE,          16#19).
-define(ECL_ENQUEUE_COPY_IMAGE,           16#1A).
-define(ECL_ENQUEUE_COPY_IMAGE_TO_BUFFER, 16#1B).
-define(ECL_ENQUEUE_COPY_BUFFER_TO_IMAGE, 16#1C).
-define(ECL_ENQUEUE_MAP_BUFFER,           16#1D).
-define(ECL_ENQUEUE_MAP_IMAGE,            16#1E).
-define(ECL_ENQUEUE_UNMAP_MEM_OBEJCT,     16#1F).
-define(ECL_GET_MEM_OBJECT_INFO,          16#20).
-define(ECL_GET_IMAGE_INFO,               16#21).
-define(ECL_CREATE_SAMPLER,               16#22).
-define(ECL_RETAIN_SAMPLER,               16#23).
-define(ECL_RELEASE_SAMPLER,              16#24).
-define(ECL_GET_SAMPLER_INFO,             16#25).
-define(ECL_CREATE_PROGRAM_WITH_SOURCE,   16#26).
-define(ECL_CREATE_PROGRAM_WITH_BINARY,   16#27).
-define(ECL_RELEASE_PROGRAM,              16#28).
-define(ECL_RETAIN_PROGRAM,               16#29).
-define(ECL_BUILD_PROGRAM,                16#2A).
-define(ECL_UNLOAD_COMPILER,              16#2B).
-define(ECL_GET_PROGRAM_INFO,             16#2C).
-define(ECL_CREATE_KERNEL,                16#2D).
-define(ECL_CREATE_KERNELS_IN_PROGRAM,    16#2E).
-define(ECL_RETAIN_KERNEL,                16#2F).
-define(ECL_RELEASE_KERNEL,               16#30).
-define(ECL_SET_KERNEL_ARG,               16#31).
-define(ECL_GET_KERNEL_INFO,              16#32).
-define(ECL_GET_PROGRAM_BUILD_INFO,       16#33).
-define(ECL_RETAIN_EVENT,                 16#34).
-define(ECL_RELEASE_EVENT,                16#35).
-define(ECL_GET_EVENT_INFO,               16#36).
-define(ECL_GET_KERNEL_WORKGROUP_INFO,    16#37).
-define(ECL_ENQUEUE_ND_RANGE_KERNEL,      16#38).
-define(ECL_ENQUEUE_TASK,                 16#39).
-define(ECL_FLUSH,                        16#3A).
-define(ECL_FINISH,                       16#3B).
-define(ECL_ENQUEUE_MARKER,               16#3C).
-define(ECL_ENQUEUE_WAIT_FOR_EVENT,       16#3D).
-define(ECL_ENQUEUE_BARRIER,              16#3E).
-define(ECL_SET_KERNEL_ARG_POINTER_T,     16#3F).
-define(ECL_SET_KERNEL_ARG_SIZE_T,        16#40).

%% Platform info
-define(ECL_PLATFORM_PROFILE,    16#00).
-define(ECL_PLATFORM_VERSION,    16#01).
-define(ECL_PLATFORM_NAME,       16#02).
-define(ECL_PLATFORM_VENDOR,     16#03).
-define(ECL_PLATFORM_EXTENSIONS, 16#04).

%% Device info
-define(ECL_DEVICE_TYPE,                             16#00).
-define(ECL_DEVICE_VENDOR_ID,                        16#01).
-define(ECL_DEVICE_MAX_COMPUTE_UNITS,                16#02).
-define(ECL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,         16#03).
-define(ECL_DEVICE_MAX_WORK_GROUP_SIZE,              16#04).
-define(ECL_DEVICE_MAX_WORK_ITEM_SIZES,              16#05).
-define(ECL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR,      16#06).
-define(ECL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT,     16#07).
-define(ECL_DEVICE_PREFERRED_VECTOR_WIDTH_INT,       16#08).
-define(ECL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG,      16#09).
-define(ECL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT,     16#0A).
-define(ECL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE,    16#0B).
-define(ECL_DEVICE_MAX_CLOCK_FREQUENCY,              16#0C).
-define(ECL_DEVICE_ADDRESS_BITS,                     16#0D).
-define(ECL_DEVICE_MAX_READ_IMAGE_ARGS,              16#0E).
-define(ECL_DEVICE_MAX_WRITE_IMAGE_ARGS,             16#0F).
-define(ECL_DEVICE_MAX_MEM_ALLOC_SIZE,               16#10).
-define(ECL_DEVICE_IMAGE2D_MAX_WIDTH,                16#11).
-define(ECL_DEVICE_IMAGE2D_MAX_HEIGHT,               16#12).
-define(ECL_DEVICE_IMAGE3D_MAX_WIDTH,                16#13).
-define(ECL_DEVICE_IMAGE3D_MAX_HEIGHT,               16#14).
-define(ECL_DEVICE_IMAGE3D_MAX_DEPTH,                16#15).
-define(ECL_DEVICE_IMAGE_SUPPORT,                    16#16).
-define(ECL_DEVICE_MAX_PARAMETER_SIZE,               16#17).
-define(ECL_DEVICE_MAX_SAMPLERS,                     16#18).
-define(ECL_DEVICE_MEM_BASE_ADDR_ALIGN,              16#19).
-define(ECL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE,         16#1A).
-define(ECL_DEVICE_SINGLE_FP_CONFIG,                 16#1B).
-define(ECL_DEVICE_GLOBAL_MEM_CACHE_TYPE,            16#1C).
-define(ECL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE,        16#1D).
-define(ECL_DEVICE_GLOBAL_MEM_CACHE_SIZE,            16#1E).
-define(ECL_DEVICE_GLOBAL_MEM_SIZE,                  16#1F).
-define(ECL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,         16#20).
-define(ECL_DEVICE_MAX_CONSTANT_ARGS,                16#21).
-define(ECL_DEVICE_LOCAL_MEM_TYPE,                   16#22).
-define(ECL_DEVICE_LOCAL_MEM_SIZE,                   16#23).
-define(ECL_DEVICE_ERROR_CORRECTION_SUPPORT,         16#24).
-define(ECL_DEVICE_PROFILING_TIMER_RESOLUTION,       16#25).
-define(ECL_DEVICE_ENDIAN_LITTLE,                    16#26).
-define(ECL_DEVICE_AVAILABLE,                        16#27).
-define(ECL_DEVICE_COMPILER_AVAILABLE,               16#28).
-define(ECL_DEVICE_EXECUTION_CAPABILITIES,           16#29).
-define(ECL_DEVICE_QUEUE_PROPERTIES,                 16#2A).
-define(ECL_DEVICE_NAME,                             16#2B).
-define(ECL_DEVICE_VENDOR,                           16#2C).
-define(ECL_DRIVER_VERSION,                          16#2D).
-define(ECL_DEVICE_PROFILE,                          16#2E).
-define(ECL_DEVICE_VERSION,                          16#2F).
-define(ECL_DEVICE_EXTENSIONS,                       16#30).
-define(ECL_DEVICE_PLATFORM,                         16#31).

%% context info
-define(ECL_CONTEXT_REFERENCE_COUNT, 16#00).
-define(ECL_CONTEXT_DEVICES,         16#01).
-define(ECL_CONTEXT_PROPERTIES,      16#02).

%% device type
-define(ECL_DEVICE_TYPE_DEFAULT,      16#00000000).
-define(ECL_DEVICE_TYPE_CPU,          16#00000001).
-define(ECL_DEVICE_TYPE_GPU,          16#00000002).
-define(ECL_DEVICE_TYPE_ACCELERATOR,  16#00000004).
-define(ECL_DEVICE_TYPE_ALL,          16#FFFFFFFF).

%% command queue properties
-define(ECL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, 16#01).
-define(ECL_QUEUE_PROFILING_ENABLE,              16#02).

-define(ECL_QUEUE_CONTEXT,         16#00).
-define(ECL_QUEUE_DEVICE,          16#01).
-define(ECL_QUEUE_REFERENCE_COUNT, 16#02).
-define(ECL_QUEUE_PROPERTIES,      16#03).

%% cl_mem_flags bitfield
-define(ECL_MEM_READ_WRITE,     16#01).
-define(ECL_MEM_WRITE_ONLY,     16#02).
-define(ECL_MEM_READ_ONLY,      16#04).
-define(ECL_MEM_USE_HOST_PTR,   16#08).
-define(ECL_MEM_ALLOC_HOST_PTR, 16#10).
-define(ECL_MEM_COPY_HOST_PTR,  16#20).

%% cl_mem_info - enum
-define(ECL_MEM_TYPE,            0).
-define(ECL_MEM_FLAGS,           1).
-define(ECL_MEM_SIZE,            2).
-define(ECL_MEM_HOST_PTR,        3).
-define(ECL_MEM_MAP_COUNT,       4).
-define(ECL_MEM_REFERENCE_COUNT, 5).
-define(ECL_MEM_CONTEXT,         6).

%% cl_sampler_info - enum
-define(ECL_SAMPLER_REFERENCE_COUNT, 0).
-define(ECL_SAMPLER_CONTEXT, 1).
-define(ECL_SAMPLER_NORMALIZED_COORDS, 2).
-define(ECL_SAMPLER_ADDRESSING_MODE, 3).
-define(ECL_SAMPLER_FILTER_MODE, 4).


%% cl_addressing_mode - enum
-define(ECL_ADDRESS_NONE,          0).
-define(ECL_ADDRESS_CLAMP_TO_EDGE, 1).
-define(ECL_ADDRESS_CLAMP,         2).
-define(ECL_ADDRESS_REPEAT,        3).

%% cl_filter_mode - enum
-define(ECL_FILTER_NEAREST, 0).
-define(ECL_FILTER_LINEAR,  1).

%% cl_map_flags - bitfiels
-define(ECL_MAP_READ,  16#01).
-define(ECL_MAP_WRITE, 16#02).

%% cl_program_info
-define(ECL_PROGRAM_REFERENCE_COUNT, 0).
-define(ECL_PROGRAM_CONTEXT, 1).
-define(ECL_PROGRAM_NUM_DEVICES, 2).
-define(ECL_PROGRAM_DEVICES, 3).
-define(ECL_PROGRAM_SOURCE, 4).
-define(ECL_PROGRAM_BINARY_SIZES, 5).
-define(ECL_PROGRAM_BINARIES, 6).

%% cl_program_build_info
-define(ECL_PROGRAM_BUILD_STATUS, 0).
-define(ECL_PROGRAM_BUILD_OPTIONS, 1).
-define(ECL_PROGRAM_BUILD_LOG, 2).

%% cl_build_status
-define(ECL_BUILD_SUCCESS,      0).
-define(ECL_BUILD_NONE,        -1).
-define(ECL_BUILD_ERROR,       -2).
-define(ECL_BUILD_IN_PROGRESS, -3).

%% cl_kernel_info
-define(ECL_KERNEL_FUNCTION_NAME,   0).
-define(ECL_KERNEL_NUM_ARGS,        1).
-define(ECL_KERNEL_REFERENCE_COUNT, 2).
-define(ECL_KERNEL_CONTEXT,         3).
-define(ECL_KERNEL_PROGRAM,         4).

%% cl_event_info
-define(ECL_EVENT_COMMAND_QUEUE,    0).
-define(ECL_EVENT_COMMAND_TYPE,     1).
-define(ECL_EVENT_REFERENCE_COUNT,  2).
-define(ECL_EVENT_COMMAND_EXECUTION_STATUS, 3).

%% cl_workgroup_info
-define(ECL_KERNEL_WORK_GROUP_SIZE, 0).
-define(ECL_KERNEL_COMPILE_WORK_GROUP_SIZE, 1).
-define(ECL_KERNEL_LOCAL_MEM_SIZE, 2).

-endif.
