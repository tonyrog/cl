%%% File    : cl.erl
%%% Author  : Tony Rogvall <tony@rogvall.se>
%%% Description : Erlang OpenCL  interface
%%% Created : 25 Oct 2009 by Tony Rogvall <tony@rogvall.se>

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
%% @doc start the OpenCL application
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
%%
-spec get_platform_ids() ->
    {'ok',[cl_platform_id()]} | {'error', cl_error()}.
    
get_platform_ids() ->
    cl_drv:call(?ECL_GET_PLATFORM_IDS, <<>>).
%%
%% @spec platform_info() ->
%%    [cl_platform_info_keys()]
%%
-spec platform_info() ->
    [cl_platform_info_key()].

platform_info() ->
    platform_info_keys().

%%
%% @spec get_platform_info(Platform :: cl_platform_id(), 
%%			Info :: cl_platform_info_key()) ->
%%    {'ok',term()} | {'error', cl_error()}
%%
%%
-spec get_platform_info(Platform :: cl_platform_id(), 
			Info :: cl_platform_info_key()) ->
    {'ok',term()} | {'error', cl_error()}.

get_platform_info(Platform, Info) ->
    get_info(?ECL_GET_PLATFORM_INFO, Platform, Info, 
	     fun platform_info_map/1).

%%
%% @spec get_platform_info(Platform::cl_platform_id()) ->
%%     {'ok', [cl_platform_info()]} | {'error', cl_error()}
%%
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
%%
-spec get_device_ids(Platform::cl_platform_id(),Type::cl_device_types()) ->
    {'ok',[cl_device_id()]} | {'error',cl_error()}.

get_device_ids(Platform, Type) ->
    TypeID = encode_device_types(Type),
    cl_drv:call(?ECL_GET_DEVICE_IDS,
		 << ?pointer_t(Platform), ?u_int32_t(TypeID) >> ).

%%
%% @spec device_info() -> [cl_device_info_key()]
%%
-spec device_info() -> [cl_device_info_key()].
    
device_info() ->
    device_info_keys().

%%
%% @spec get_device_info(DevID::cl_device_id(), Info::cl_device_info_key()) ->
%%   {'ok', term()} | {'error', cl_error()}
%%
-spec get_device_info(Device::cl_device_id(), Info::cl_device_info_key()) ->
    {'ok', term()} | {'error', cl_error()}.

get_device_info(Device, Info) ->
    get_info(?ECL_GET_DEVICE_INFO, Device, Info, fun device_info_map/1).
%%
%% @spec get_device_info(Device) ->
%%    {'ok', [cl_device_info()]} | {'error', cl_error()}.
%%
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
%%
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
%%
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

-spec release_context(Context::cl_context()) ->
    'ok' | {'error', cl_error()}.

release_context(Context) ->
    cl_drv:release(?ECL_RELEASE_CONTEXT, Context).

%%
%% @spec retain_context(Context::cl_context()) ->
%%     'ok' | {'error', cl_error()}
%%
-spec retain_context(Context::cl_context()) ->
    'ok' | {'error', cl_error()}.

retain_context(Context) ->
    cl_drv:retain(?ECL_RETAIN_CONTEXT, Context).

%%
%% @spec context_info() -> [cl_context_info_key()]
%%
-spec context_info() -> [cl_context_info_key()].

context_info() ->
    context_info_keys().

%%
%% @spec get_context_info(Context::cl_context(),Info::cl_context_info_key()) ->
%%   {'ok', term()} | {'error', cl_error()}
%%
-spec get_context_info(Context::cl_context(), Info::cl_context_info_key()) ->
    {'ok', term()} | {'error', cl_error()}.

get_context_info(Context, Info) ->
    get_info(?ECL_GET_CONTEXT_INFO, Context, Info, fun context_info_map/1).
%%
%% @spec get_context_info(Context::cl_context()) ->
%%    {'ok', [cl_context_info()]} | {'error', cl_error()}
%%
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
%%
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
%%
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

-spec release_queue(Queue::cl_queue()) ->
    'ok' | {'error', cl_error()}.
release_queue(Queue) ->
    cl_drv:release(?ECL_RELEASE_QUEUE, Queue).

%%
%% @spec retain_queue(Queue::cl_queue()) ->
%%    'ok' | {'error', cl_error()}
-spec retain_queue(Queue::cl_queue()) ->
    'ok' | {'error', cl_error()}.

retain_queue(Queue) ->
    cl_drv:retain(?ECL_RETAIN_QUEUE, Queue).

queue_info() ->
    queue_info_keys().

get_queue_info(Queue, Info) ->
    get_info(?ECL_GET_QUEUE_INFO, Queue, Info, 
	     fun queue_info_map/1).

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
-spec release_mem_object(Mem::cl_mem()) ->
    'ok' | {'error', cl_error()}.

release_mem_object(Mem) ->
    cl_drv:release(?ECL_RELEASE_MEM_OBJECT, Mem).

%%
%% @spec retain_mem_object(Mem::cl_mem()) ->
%%    'ok' | {'error', cl_error()}
-spec retain_mem_object(Mem::cl_mem()) ->
    'ok' | {'error', cl_error()}.

retain_mem_object(Mem) ->
    cl_drv:release(?ECL_RETAIN_MEM_OBJECT, Mem).

get_mem_object_info(Mem, Info) ->
    get_info(?ECL_GET_MEM_OBJECT_INFO, Mem, Info, fun mem_info_map/1).

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
%%    
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
%%
-spec release_sampler(Sampler::cl_sampler()) ->
    'ok' | {'error', cl_error()}.

release_sampler(Mem) ->
    cl_drv:release(?ECL_RELEASE_SAMPLER, Mem).

%%
%% @spec retain_sampler(Sampler::cl_sampler()) ->
%%    'ok' | {'error', cl_error()}
%%
-spec retain_sampler(Sampler::cl_sampler()) ->
    'ok' | {'error', cl_error()}.

retain_sampler(Sampler) ->
    cl_drv:release(?ECL_RETAIN_SAMPLER, Sampler).

sampler_info() ->
    sampler_info_keys().

get_sampler_info(Sampler, Info) ->
    get_info(?ECL_GET_SAMPLER_INFO, Sampler, Info, 
	     fun sampler_info_map/1).

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
%% @doc Create program from, possibly, cached compilation data.
%%  The binaries and device can be generated by calling:
%%  <code>
%%    {ok,P} = cl:create_program_with_source(Context,Source),
%%    ok = cl:build_program(P, DeviceList, Options),
%%    {ok,DeviceList} = cl:get_program_info(P, devices),
%%    {ok,BinaryList} = cl:get_program_info(P, binaries).
%%  </code>
%%
%% @spec create_program_with_binary(Context::cl_context(),
%%                                  DeviceList::[cl_device_id()],
%%                                  BinaryList::[binary()]) ->
%%    {'ok', cl_program()} | {'error', cl_error()}
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

retain_program(Program) ->
    cl_drv:retain(?ECL_RETAIN_PROGRAM, Program).

%%
%% @spec release_program(Program::cl_program()) ->
%%    'ok' | {'error', cl_error()}

release_program(Program) ->
    cl_drv:release(?ECL_RELEASE_PROGRAM, Program).

%%
%% @spec build_program(Program::cl_program(),
%%                     DeviceList::[cl_device_id()],
%%                     Options::string()) ->
%%  'ok' | {'error', cl_error()}
%%
build_program(Program, DeviceList, Options) ->
    DevData = encode_pointer_array(DeviceList),
    cl_drv:call(?ECL_BUILD_PROGRAM,
		 <<?pointer_t(Program),
		  DevData/binary,
		  (list_to_binary([Options]))/binary>>).

%%
%% @spec unload_compiler() -> 'ok' | {'error', cl_error()}
%%
unload_compiler() ->		    
    cl_drv:call(?ECL_UNLOAD_COMPILER, <<>>).

program_info() ->
    program_info_keys().

get_program_info(Program, Info) ->
    get_info(?ECL_GET_PROGRAM_INFO, Program, Info, fun program_info_map/1).

get_program_info(Program) ->
    get_info_list(?ECL_GET_PROGRAM_INFO, Program, 
		  program_info_keys(), fun program_info_map/1).

program_build_info() ->
    build_info_keys().

get_program_build_info(Program, Device, Info) ->
    get_info(?ECL_GET_PROGRAM_BUILD_INFO, Program, 
	     <<?pointer_t(Device)>>, Info, fun build_info_map/1).

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
%%
%% For now set_kernel_arg handles integer and floats
%% to set any other type use <<Foo:Bar/native...>>
%% use the macros defined in cl.hrl to get it right (except for padding)
%%
%%
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
%% special to set kernel arg with size only (local mem etc)
%%
set_kernel_arg_size(Kernel,Index,Size) ->
    cl_drv:call(?ECL_SET_KERNEL_ARG,
		<<?pointer_t(Kernel),
		 ?u_int32_t(Index),
		 ?u_int32_t(Size)>>).

%%
%% @spec retain_kernel(Context::cl_kernel()) ->
%%    'ok' | {'error', cl_error()}

retain_kernel(Kernel) ->
    cl_drv:retain(?ECL_RETAIN_KERNEL, Kernel).

%%
%% @spec release_kernel(Context::cl_kernel()) ->
%%    'ok' | {'error', cl_error()}

release_kernel(Kernel) ->
    cl_drv:release(?ECL_RELEASE_KERNEL, Kernel).

kernel_info() ->
    kernel_info_keys().

get_kernel_info(Kernel, Info) ->
    get_info(?ECL_GET_KERNEL_INFO, Kernel, Info, fun kernel_info_map/1).

get_kernel_info(Kernel) ->
    get_info_list(?ECL_GET_KERNEL_INFO, Kernel, 
		  kernel_info_keys(), fun kernel_info_map/1).

kernel_workgroup_info() ->
    workgroup_info_keys().

get_kernel_workgroup_info(Kernel, Device, Info) ->
    get_info(?ECL_GET_KERNEL_WORKGROUP_INFO, Kernel,
	     <<?pointer_t(Device)>>,  Info, fun workgroup_info_map/1).

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
%% 
%% @spec enqueue_barrier(Queue::cl_queue()) ->
%%    'ok' | {'error', cl_error()}
%%

-spec enqueue_barrier(Queue::cl_queue()) ->
    'ok' | {'error', cl_error()}.

enqueue_barrier(Queue) ->
    cl_drv:call(?ECL_ENQUEUE_BARRIER,
		 <<?pointer_t(Queue)>>).
%%
%% @doc Make sure all operations are initiated on the device
%% they are queue for.
%%
%% @spec flush(Queue::cl_queue()) ->
%%    'ok' | {'error', cl_error()}
%%

-spec flush(Queue::cl_queue()) ->
    'ok' | {'error', cl_error()}.

flush(Queue) ->
    cl_drv:call(?ECL_FLUSH, <<?pointer_t(Queue)>>).

%%
%% @doc Block until are operations have completed or failed.
%%
%% @spec finish(Queue::cl_queue()) ->
%%    'ok' | {'error', cl_error()}
%%

-spec finish(Queue::cl_queue()) ->
    'ok' | {'error', cl_error()}.

finish(Queue) ->
    cl_drv:call(?ECL_FINISH, <<?pointer_t(Queue)>>).

%%
%% @spec retain_event(Event::cl_event()) ->
%%    'ok' | {'error', cl_error()}

retain_event(Event) ->
    cl_drv:retain(?ECL_RETAIN_EVENT, Event).

%%
%% @spec release_event(Event::cl_event()) ->
%%    'ok' | {'error', cl_error()}

release_event(Event) ->
    cl_drv:release(?ECL_RELEASE_EVENT, Event).

event_info() ->
    event_info_keys().

get_event_info(Event, Info) ->
    get_info(?ECL_GET_EVENT_INFO, Event, Info, fun event_info_map/1).

get_event_info(Event) ->
    get_info_list(?ECL_GET_EVENT_INFO, Event, 
		  event_info_keys(), fun event_info_map/1).
%%
%% @type timeout() = non_neg_integer() | 'infinity'
%%
%% @doc Wait for event completion.
%%
%% @spec wait(Event::cl_event) -> 
%%    {'ok','completed'} | {'ok',Binary} | {'error',cl_error()}
%% @equiv wait(Event, infinity)
%%
wait(Event) ->
    wait(Event, infinity).
%%
%% @doc Wait for event completion or timeout.
%%  
%% @spec wait(Event::cl_event, Timeout::timeout()) -> 
%%    {'ok','completed'} | {'ok',Binary} | 
%%    {'error',cl_error()} | {'error',timeout}
%% 
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


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Utilities
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
get_info(Command, ID, Info, Map) ->
    get_info(Command, ID, <<>>, Info, Map).

get_info(Command, ID, Arg, Info, Map) ->
    InfoID = Map(Info),
    cl_drv:call(Command, <<?pointer_t(ID),Arg/binary,?u_int32_t(InfoID)>>).

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

encode_mem_flag(read_write) -> ?ECL_MEM_READ_WRITE;
encode_mem_flag(write_only) -> ?ECL_MEM_WRITE_ONLY;
encode_mem_flag(read_only) -> ?ECL_MEM_READ_ONLY;
encode_mem_flag(use_host_ptr) -> ?ECL_MEM_USE_HOST_PTR;
encode_mem_flag(alloc_host_ptr) -> ?ECL_MEM_ALLOC_HOST_PTR;
encode_mem_flag(copy_host_ptr) -> ?ECL_MEM_COPY_HOST_PTR.

encode_mem_flags([T|Ts]) ->
    encode_mem_flag(T) bor encode_mem_flags(Ts);
encode_mem_flags([]) ->
    0.


encode_queue_property(out_of_order_exec_mode_enable) ->
    ?ECL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
encode_queue_property(profiling_enabled) ->
    ?ECL_QUEUE_PROFILING_ENABLE.

encode_queue_properties([T|Ts]) ->
    encode_queue_property(T) bor encode_queue_properties(Ts);
encode_queue_properties([]) ->
    0.
%%
%% Encode kernel argument
%% FIXME: check pointers etc since OpenCL will crash if we 
%%        send bad pointers!
%%
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
encode_argument({'pointer',X}) -> <<?cl_pointer(X)>>;
encode_argument({'size',X}) -> <<?cl_size(X)>>;

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
encode_pointer_array(Pointers) when is_list(Pointers) ->
    N = length(Pointers),
    <<?size_t(N), (<< <<?pointer_t(Ptr) >> || Ptr <- Pointers >>)/binary>>.
%%
%% Encode binary array <<N:32, Size1:Size, Binary1:Size1/binary ... >>
%%
encode_async_binary_array(Binaries) when is_list(Binaries) ->
    N = length(Binaries),
    [<<?size_t(N)>> | 
     lists:map(fun(Bin) -> [<<?size_t((byte_size(Bin)))>>,Bin] end, 
	       Binaries)].


%% boolean - passed as uint32
encode_bool(true) -> 1;
encode_bool(false) -> 0.

%% addressing_mode - enum
encode_addressing_mode(Mode) ->
    addressing_mode_map(Mode).

encode_filter_mode(Mode) ->
    filter_mode_map(Mode).

%% device_type - bitfield
encode_device_type(cpu) -> ?ECL_DEVICE_TYPE_CPU;
encode_device_type(gpu) -> ?ECL_DEVICE_TYPE_GPU;
encode_device_type(accelerator) -> ?ECL_DEVICE_TYPE_ACCELERATOR;
encode_device_type(all) -> ?ECL_DEVICE_TYPE_ALL;
encode_device_type(default) -> ?ECL_DEVICE_TYPE_DEFAULT.

encode_device_types(T) when is_atom(T) ->
    encode_device_type(T);    
encode_device_types([T|Ts]) when is_list(Ts) ->
    encode_device_type(T) bor encode_device_types(Ts);
encode_device_types([]) ->
    0.

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

platform_info_keys() ->
    [ profile, version, name, vendor, extensions].

platform_info_map(Key) ->
    case Key of
	profile -> ?ECL_PLATFORM_PROFILE;
	version -> ?ECL_PLATFORM_VERSION;
	name -> ?ECL_PLATFORM_NAME;
	vendor -> ?ECL_PLATFORM_VENDOR;
	extensions -> ?ECL_PLATFORM_EXTENSIONS
    end.

context_info_keys() ->
    [ reference_count, devices, properties ].

context_info_map(Key) ->
    case Key of
	reference_count -> ?ECL_CONTEXT_REFERENCE_COUNT;
	devices         -> ?ECL_CONTEXT_DEVICES;
	properties      -> ?ECL_CONTEXT_PROPERTIES
    end.

queue_info_keys() ->
    [ context, device, reference_count, properties ].

queue_info_map(Key) ->
    case Key of
	context         -> ?ECL_QUEUE_CONTEXT;
	device          -> ?ECL_QUEUE_DEVICE;
	reference_count -> ?ECL_QUEUE_REFERENCE_COUNT;
	properties      -> ?ECL_QUEUE_PROPERTIES
    end.

mem_info_keys() ->
    [ object_type, flags, size, host_ptr, map_count,
      reference_count, context  ].

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

sampler_info_keys() ->
    [ reference_count, context, normalized_coords,
      addressing_mode, filter_mode ].

sampler_info_map(Key) ->
    case Key of
	reference_count -> ?ECL_SAMPLER_REFERENCE_COUNT; 	    
	context -> ?ECL_SAMPLER_CONTEXT;
	normalized_coords -> ?ECL_SAMPLER_NORMALIZED_COORDS;
	addressing_mode -> ?ECL_SAMPLER_ADDRESSING_MODE;
	filter_mode -> ?ECL_SAMPLER_FILTER_MODE
    end.

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

filter_mode_map(Key) ->
    case Key of
	nearest -> ?ECL_FILTER_NEAREST;
	linear  -> ?ECL_FILTER_LINEAR
    end.

program_info_keys() ->
    [ reference_count, context, num_decices, devices,
      source, binary_sizes, binaries ].

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

build_info_keys() ->
    [status, options, log ].

build_info_map(Key) ->
    case Key of
	status -> ?ECL_PROGRAM_BUILD_STATUS;
	options -> ?ECL_PROGRAM_BUILD_OPTIONS;
	log -> ?ECL_PROGRAM_BUILD_LOG
    end.

kernel_info_keys() ->
    [ function_name, num_args, reference_count,
      context, program ].
    
kernel_info_map(Key) ->
    case Key of
	function_name -> ?ECL_KERNEL_FUNCTION_NAME;
	num_args -> ?ECL_KERNEL_NUM_ARGS;
	reference_count -> ?ECL_KERNEL_REFERENCE_COUNT;
	context -> ?ECL_KERNEL_CONTEXT;
	program -> ?ECL_KERNEL_PROGRAM
    end.

event_info_keys() ->
    [ command_queue, command_type, reference_count,
      execution_status ].

event_info_map(Key) ->
    case Key of 
	command_queue -> ?ECL_EVENT_COMMAND_QUEUE;
	command_type  -> ?ECL_EVENT_COMMAND_TYPE;
	reference_count -> ?ECL_EVENT_REFERENCE_COUNT;
	execution_status -> ?ECL_EVENT_COMMAND_EXECUTION_STATUS
    end.

workgroup_info_keys() ->
    [ work_group_size, compile_work_group_size,
      local_mem_size ].
	    
workgroup_info_map(Key) ->
    case Key of
	work_group_size -> ?ECL_KERNEL_WORK_GROUP_SIZE;
	compile_work_group_size -> ?ECL_KERNEL_COMPILE_WORK_GROUP_SIZE; 
	local_mem_size -> ?ECL_KERNEL_LOCAL_MEM_SIZE
    end.
