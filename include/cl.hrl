%%%---- BEGIN COPYRIGHT -------------------------------------------------------
%%%
%%% Copyright (C) 2007 - 2012, Rogvall Invest AB, <tony@rogvall.se>
%%%
%%% This software is licensed as described in the file COPYRIGHT, which
%%% you should have received as part of this distribution. The terms
%%% are also available at http://www.rogvall.se/docs/copyright.txt.
%%%
%%% You may opt to use, copy, modify, merge, publish, distribute and/or sell
%%% copies of the Software, and permit persons to whom the Software is
%%% furnished to do so, under the terms of the COPYRIGHT file.
%%%
%%% This software is distributed on an "AS IS" basis, WITHOUT WARRANTY OF ANY
%%% KIND, either express or implied.
%%%
%%%---- END COPYRIGHT ---------------------------------------------------------
%%
%% Definitions used here and there
%%
-ifndef(__CL_HRL__).
-define(__CL_HRL__, true).

-define(POINTER_SIZE, 64).  %% casted by driver
-define(SIZE_SIZE,    64).  %% casted by driver

%% transport types
-define(u_int8_t(X),    X:8/native-unsigned-integer).
-define(u_int16_t(X),   X:16/native-unsigned-integer).
-define(u_int32_t(X),   X:32/native-unsigned-integer).
-define(u_int64_t(X),   X:64/native-unsigned-integer).
-define(int8_t(X),      X:8/native-signed-integer).
-define(int16_t(X),     X:16/native-signed-integer).
-define(int32_t(X),     X:32/native-signed-integer).
-define(int64_t(X),     X:64/native-signed-integer).
-define(float_t(X),     X:32/native-float).
-define(double_t(X),    X:64/native-float).
-define(pointer_t(X),   X:?POINTER_SIZE/native-unsigned-integer).
-define(size_t(X),      X:?SIZE_SIZE/native-unsigned-integer).

%% scalar types
%% @type cl_char() = integer()
%% @type cl_uchar() = non_neg_integer()
%% @type cl_short() = integer()
%% @type cl_ushort() = non_neg_integer()
%% @type cl_int() = integer()
%% @type cl_uint() = non_neg_integer()
%% @type cl_long() = integer()
%% @type cl_ulong() = non_neg_integer()
%% @type cl_half() = float()
%% @type cl_float() = float()
%% @type cl_double() = float()

-type cl_char() :: integer().
-type cl_uchar() :: non_neg_integer().
-type cl_short() :: integer().
-type cl_ushort() :: non_neg_integer().
-type cl_int() :: integer().
-type cl_uint() :: non_neg_integer().
-type cl_long() :: integer().
-type cl_ulong() :: non_neg_integer().
-type cl_half() :: float().
-type cl_float() :: float().
-type cl_double() :: float().


-define(cl_char(X),     X:8/native-signed-integer).
-define(cl_uchar(X),    X:8/native-unsigned-integer).
-define(cl_short(X),    X:16/native-signed-integer).
-define(cl_ushort(X),   X:16/native-unsigned-integer).
-define(cl_int(X),      X:32/native-signed-integer).
-define(cl_uint(X),     X:32/native-unsigned-integer).
-define(cl_long(X),     X:64/native-signed-integer).
-define(cl_ulong(X),    X:64/native-unsigned-integer).
-define(cl_half(X),     X:16/native-unsigned-integer).
-define(cl_float(X),    X:32/native-float).
-define(cl_double(X),   X:64/native-float).

-define(cl_pointer(X),  X:?POINTER_SIZE/native-unsigned-integer).
-define(cl_size(X),     X:?SIZE_SIZE/native-unsigned-integer).

%% vector types,  OpenCL requires that all types be naturally aligned. 
-define(cl_char2(X1,X2), ?cl_char(X1), ?cl_char(X2)).
-define(cl_char4(X1,X2,X3,X4),
	?cl_char(X1), ?cl_char(X2), ?cl_char(X3), ?cl_char(X4)).
-define(cl_char8(X1,X2,X3,X4,X5,X6,X7,X8), 
	?cl_char(X1), ?cl_char(X2), ?cl_char(X3), ?cl_char(X4),
	?cl_char(X5), ?cl_char(X6), ?cl_char(X7), ?cl_char(X8)).
-define(cl_char16(X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,X11,X12,X13,X14,X15,X16), 
	?cl_char(X1), ?cl_char(X2), ?cl_char(X3), ?cl_char(X4),
	?cl_char(X5), ?cl_char(X6), ?cl_char(X7), ?cl_char(X8),
	?cl_char(X9), ?cl_char(X10), ?cl_char(X11), ?cl_char(X12),
	?cl_char(X13), ?cl_char(X14), ?cl_char(X15), ?cl_char(X16)).

-define(cl_uchar2(X1,X2), ?cl_uchar(X1), ?cl_uchar(X2)).
-define(cl_uchar4(X1,X2,X3,X4),
	?cl_uchar(X1), ?cl_uchar(X2), ?cl_uchar(X3), ?cl_uchar(X4)).
-define(cl_uchar8(X1,X2,X3,X4,X5,X6,X7,X8), 
	?cl_uchar(X1), ?cl_uchar(X2), ?cl_uchar(X3), ?cl_uchar(X4),
	?cl_uchar(X5), ?cl_uchar(X6), ?cl_uchar(X7), ?cl_uchar(X8)).
-define(cl_uchar16(X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,X11,X12,X13,X14,X15,X16), 
	?cl_uchar(X1), ?cl_uchar(X2), ?cl_uchar(X3), ?cl_uchar(X4),
	?cl_uchar(X5), ?cl_uchar(X6), ?cl_uchar(X7), ?cl_uchar(X8),
	?cl_uchar(X9), ?cl_uchar(X10), ?cl_uchar(X11), ?cl_uchar(X12),
	?cl_uchar(X13), ?cl_uchar(X14), ?cl_uchar(X15), ?cl_uchar(X16)).

-define(cl_short2(X1,X2), ?cl_short(X1), ?cl_short(X2)).
-define(cl_short4(X1,X2,X3,X4),
	?cl_short(X1), ?cl_short(X2), ?cl_short(X3), ?cl_short(X4)).
-define(cl_short8(X1,X2,X3,X4,X5,X6,X7,X8), 
	?cl_short(X1), ?cl_short(X2), ?cl_short(X3), ?cl_short(X4),
	?cl_short(X5), ?cl_short(X6), ?cl_short(X7), ?cl_short(X8)).
-define(cl_short16(X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,X11,X12,X13,X14,X15,X16), 
	?cl_short(X1), ?cl_short(X2), ?cl_short(X3), ?cl_short(X4),
	?cl_short(X5), ?cl_short(X6), ?cl_short(X7), ?cl_short(X8),
	?cl_short(X9), ?cl_short(X10), ?cl_short(X11), ?cl_short(X12),
	?cl_short(X13), ?cl_short(X14), ?cl_short(X15), ?cl_short(X16)).

-define(cl_ushort2(X1,X2), ?cl_ushort(X1), ?cl_ushort(X2)).
-define(cl_ushort4(X1,X2,X3,X4),
	?cl_ushort(X1), ?cl_ushort(X2), ?cl_ushort(X3), ?cl_ushort(X4)).
-define(cl_ushort8(X1,X2,X3,X4,X5,X6,X7,X8), 
	?cl_ushort(X1), ?cl_ushort(X2), ?cl_ushort(X3), ?cl_ushort(X4),
	?cl_ushort(X5), ?cl_ushort(X6), ?cl_ushort(X7), ?cl_ushort(X8)).
-define(cl_ushort16(X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,X11,X12,X13,X14,X15,X16), 
	?cl_ushort(X1), ?cl_ushort(X2), ?cl_ushort(X3), ?cl_ushort(X4),
	?cl_ushort(X5), ?cl_ushort(X6), ?cl_ushort(X7), ?cl_ushort(X8),
	?cl_ushort(X9), ?cl_ushort(X10), ?cl_ushort(X11), ?cl_ushort(X12),
	?cl_ushort(X13), ?cl_ushort(X14), ?cl_ushort(X15), ?cl_ushort(X16)).

-define(cl_int2(X1,X2), ?cl_int(X1), ?cl_int(X2)).
-define(cl_int4(X1,X2,X3,X4),
	?cl_int(X1), ?cl_int(X2), ?cl_int(X3), ?cl_int(X4)).
-define(cl_int8(X1,X2,X3,X4,X5,X6,X7,X8), 
	?cl_int(X1), ?cl_int(X2), ?cl_int(X3), ?cl_int(X4),
	?cl_int(X5), ?cl_int(X6), ?cl_int(X7), ?cl_int(X8)).
-define(cl_int16(X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,X11,X12,X13,X14,X15,X16), 
	?cl_int(X1), ?cl_int(X2), ?cl_int(X3), ?cl_int(X4),
	?cl_int(X5), ?cl_int(X6), ?cl_int(X7), ?cl_int(X8),
	?cl_int(X9), ?cl_int(X10), ?cl_int(X11), ?cl_int(X12),
	?cl_int(X13), ?cl_int(X14), ?cl_int(X15), ?cl_int(X16)).

-define(cl_uint2(X1,X2), ?cl_uint(X1), ?cl_uint(X2)).
-define(cl_uint4(X1,X2,X3,X4),
	?cl_uint(X1), ?cl_uint(X2), ?cl_uint(X3), ?cl_uint(X4)).
-define(cl_uint8(X1,X2,X3,X4,X5,X6,X7,X8), 
	?cl_uint(X1), ?cl_uint(X2), ?cl_uint(X3), ?cl_uint(X4),
	?cl_uint(X5), ?cl_uint(X6), ?cl_uint(X7), ?cl_uint(X8)).
-define(cl_uint16(X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,X11,X12,X13,X14,X15,X16), 
	?cl_uint(X1), ?cl_uint(X2), ?cl_uint(X3), ?cl_uint(X4),
	?cl_uint(X5), ?cl_uint(X6), ?cl_uint(X7), ?cl_uint(X8),
	?cl_uint(X9), ?cl_uint(X10), ?cl_uint(X11), ?cl_uint(X12),
	?cl_uint(X13), ?cl_uint(X14), ?cl_uint(X15), ?cl_uint(X16)).

-define(cl_long2(X1,X2), ?cl_long(X1), ?cl_long(X2)).
-define(cl_long4(X1,X2,X3,X4),
	?cl_long(X1), ?cl_long(X2), ?cl_long(X3), ?cl_long(X4)).
-define(cl_long8(X1,X2,X3,X4,X5,X6,X7,X8), 
	?cl_long(X1), ?cl_long(X2), ?cl_long(X3), ?cl_long(X4),
	?cl_long(X5), ?cl_long(X6), ?cl_long(X7), ?cl_long(X8)).
-define(cl_long16(X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,X11,X12,X13,X14,X15,X16), 
	?cl_long(X1), ?cl_long(X2), ?cl_long(X3), ?cl_long(X4),
	?cl_long(X5), ?cl_long(X6), ?cl_long(X7), ?cl_long(X8),
	?cl_long(X9), ?cl_long(X10), ?cl_long(X11), ?cl_long(X12),
	?cl_long(X13), ?cl_long(X14), ?cl_long(X15), ?cl_long(X16)).

-define(cl_ulong2(X1,X2), ?cl_ulong(X1), ?cl_ulong(X2)).
-define(cl_ulong4(X1,X2,X3,X4),
	?cl_ulong(X1), ?cl_ulong(X2), ?cl_ulong(X3), ?cl_ulong(X4)).
-define(cl_ulong8(X1,X2,X3,X4,X5,X6,X7,X8), 
	?cl_ulong(X1), ?cl_ulong(X2), ?cl_ulong(X3), ?cl_ulong(X4),
	?cl_ulong(X5), ?cl_ulong(X6), ?cl_ulong(X7), ?cl_ulong(X8)).
-define(cl_ulong16(X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,X11,X12,X13,X14,X15,X16), 
	?cl_ulong(X1), ?cl_ulong(X2), ?cl_ulong(X3), ?cl_ulong(X4),
	?cl_ulong(X5), ?cl_ulong(X6), ?cl_ulong(X7), ?cl_ulong(X8),
	?cl_ulong(X9), ?cl_ulong(X10), ?cl_ulong(X11), ?cl_ulong(X12),
	?cl_ulong(X13), ?cl_ulong(X14), ?cl_ulong(X15), ?cl_ulong(X16)).

-define(cl_float2(X1,X2), ?cl_float(X1), ?cl_float(X2)).
-define(cl_float4(X1,X2,X3,X4),
	?cl_float(X1), ?cl_float(X2), ?cl_float(X3), ?cl_float(X4)).
-define(cl_float8(X1,X2,X3,X4,X5,X6,X7,X8),
	?cl_float(X1), ?cl_float(X2), ?cl_float(X3), ?cl_float(X4),
	?cl_float(X5), ?cl_float(X6), ?cl_float(X7), ?cl_float(X8)).
-define(cl_float16(X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,X11,X12,X13,X14,X15,X16),
	?cl_float(X1),  ?cl_float(X2),  ?cl_float(X3),  ?cl_float(X4),
	?cl_float(X5),  ?cl_float(X6),  ?cl_float(X7),  ?cl_float(X8),
	?cl_float(X9),  ?cl_float(X10), ?cl_float(X11), ?cl_float(X12),
	?cl_float(X13), ?cl_float(X14), ?cl_float(X15), ?cl_float(X16)).

-define(cl_double2(X1,X2), ?cl_double(X1), ?cl_double(X2)).
-define(cl_double4(X1,X2,X3,X4),
	?cl_double(X1), ?cl_double(X2), ?cl_double(X3), ?cl_double(X4)).
-define(cl_double8(X1,X2,X3,X4,X5,X6,X7,X8), 
	?cl_double(X1), ?cl_double(X2), ?cl_double(X3), ?cl_double(X4),
	?cl_double(X5), ?cl_double(X6), ?cl_double(X7), ?cl_double(X8)).
-define(cl_double16(X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,X11,X12,X13,X14,X15,X16), 
	?cl_double(X1), ?cl_double(X2), ?cl_double(X3), ?cl_double(X4),
	?cl_double(X5), ?cl_double(X6), ?cl_double(X7), ?cl_double(X8),
	?cl_double(X9), ?cl_double(X10), ?cl_double(X11), ?cl_double(X12),
	?cl_double(X13), ?cl_double(X14), ?cl_double(X15), ?cl_double(X16)).

%% @type cl_platform_id() = { {'object', 1, non_neg_integer() } }
%% @type cl_device_id()   = { {'object', 2, non_neg_integer() } }
%% @type cl_context()     = { {'object', 3, non_neg_integer() } }
%% @type cl_queue()       = { {'object', 4, non_neg_integer() } }
%% @type cl_mem()         = { {'object', 5, non_neg_integer() } }
%% @type cl_sampler()     = { {'object', 6, non_neg_integer() } }
%% @type cl_program()     = { {'object', 7, non_neg_integer() } }
%% @type cl_kernel()      = { {'object', 8, non_neg_integer() } }
%% @type cl_event()       = { {'object', 9, non_neg_integer() } }

-type cl_platform_id() :: { {'object', 1, non_neg_integer() } }.
-type cl_device_id()   :: { {'object', 2, non_neg_integer() } }.
-type cl_context()     :: { {'object', 3, non_neg_integer() } }.
-type cl_queue()       :: { {'object', 4, non_neg_integer() } }.
-type cl_mem()         :: { {'object', 5, non_neg_integer() } }.
-type cl_sampler()     :: { {'object', 6, non_neg_integer() } }.
-type cl_program()     :: { {'object', 7, non_neg_integer() } }.
-type cl_kernel()      :: { {'object', 8, non_neg_integer() } }.
-type cl_event()       :: { {'object', 9, non_neg_integer() } }.

%% @type cl_error()  =  {
%%     'device_not_found' |
%%     'device_not_available' |
%%     'compiler_not_available' |
%%     'mem_object_allocation_failure' |
%%     'out_of_resources' |
%%     'out_of_host_memory' |
%%     'profiling_info_not_available' |
%%     'mem_copy_overlap' |
%%     'image_format_mismatch' |
%%     'image_format_not_supported' |
%%     'build_program_failure' |
%%     'map_failure' |
%%     'invalid_value' |
%%     'invalid_device type' |
%%     'invalid_platform' |
%%     'invalid_device' |
%%     'invalid_context' |
%%     'invalid_queue_properties' |
%%     'invalid_command_queue' |
%%     'invalid_host_ptr' |
%%     'invalid_mem_object' |
%%     'invalid_image_format_descriptor' |
%%     'invalid_image_size' |
%%     'invalid_sampler' |
%%     'invalid_binary' |
%%     'invalid_build_options' |
%%     'invalid_program' |
%%     'invalid_program_executable' |
%%     'invalid_kernel_name' |
%%     'invalid_kernel_definition' |
%%     'invalid_kernel' |
%%     'invalid_arg_index' |
%%     'invalid_arg_value' |
%%     'invalid_arg_size' |
%%     'invalid_kernel_args' |
%%     'invalid_work_dimension' |
%%     'invalid_work_group_size' |
%%     'invalid_work_item size' |
%%     'invalid_global_offset' |
%%     'invalid_event_wait_list' |
%%     'invalid_event' |
%%     'invalid_operation' |
%%     'invalid_gl_object' |
%%     'invalid_buffer_size' |
%%     'invalid_mip_level' |
%%     'unknown' }.

-type cl_error()  ::  {
    'device_not_found' |
    'device_not_available' |
    'compiler_not_available' |
    'mem_object_allocation_failure' |
    'out_of_resources' |
    'out_of_host_memory' |
    'profiling_info_not_available' |
    'mem_copy_overlap' |
    'image_format_mismatch' |
    'image_format_not_supported' |
    'build_program_failure' |
    'map_failure' |
    'invalid_value' |
    'invalid_device type' |
    'invalid_platform' |
    'invalid_device' |
    'invalid_context' |
    'invalid_queue_properties' |
    'invalid_command_queue' |
    'invalid_host_ptr' |
    'invalid_mem_object' |
    'invalid_image_format_descriptor' |
    'invalid_image_size' |
    'invalid_sampler' |
    'invalid_binary' |
    'invalid_build_options' |
    'invalid_program' |
    'invalid_program_executable' |
    'invalid_kernel_name' |
    'invalid_kernel_definition' |
    'invalid_kernel' |
    'invalid_arg_index' |
    'invalid_arg_value' |
    'invalid_arg_size' |
    'invalid_kernel_args' |
    'invalid_work_dimension' |
    'invalid_work_group_size' |
    'invalid_work_item size' |
    'invalid_global_offset' |
    'invalid_event_wait_list' |
    'invalid_event' |
    'invalid_operation' |
    'invalid_gl_object' |
    'invalid_buffer_size' |
    'invalid_mip_level' |
    'unknown' }.

-define(cl_platform_id(X),   ?cl_pointer(X)).
-define(cl_device_id(X),     ?cl_pointer(X)).
-define(cl_context(X),       ?cl_pointer(X)).
-define(cl_command_queue(X), ?cl_pointer(X)).
-define(cl_mem(X),           ?cl_pointer(X)).
-define(cl_program(X),       ?cl_pointer(X)).
-define(cl_kernel(X),        ?cl_pointer(X)).
-define(cl_event(X),         ?cl_pointer(X)).
-define(cl_sampler(X),       ?cl_pointer(X)).

-define(cl_bool(X),         ?cl_uint(X)).
-define(cl_bitfield(X),     ?cl_ulong(X)).
-define(cl_device_type(X),  ?cl_bitfield(X)).
-define(cl_platform_info(X),  ?cl_uint(X)).
-define(cl_device_info(X),  ?cl_uint(X)).
-define(cl_device_address_info(X),  ?cl_bitfield(X)).
-define(cl_device_fp_config(X),  ?cl_bitfield(X)).
-define(cl_device_mem_cache_type(X),  ?cl_uint(X)).
-define(cl_device_local_mem_type(X),  ?cl_uint(X)).
-define(cl_device_exec_capabilities(X),  ?cl_bitfield(X)).
-define(cl_command_queue_properties(X),  ?cl_bitfield(X)).

%% -define(cl_context_properties(X),  ?intptr_t(X)).
-define(cl_context_info(X),        ?cl_uint(X)).
-define(cl_command_queue_info(X),  ?cl_uint(X)).
-define(cl_channel_order(X),       ?cl_uint(X)).
-define(cl_channel_type(X),        ?cl_uint(X)).
-define(cl_mem_flags(X),           ?cl_bitfield(X)).
-define(cl_mem_object_type(X),     ?cl_uint(X)).
-define(cl_mem_info(X),            ?cl_uint(X)).
-define(cl_image_info(X),          ?cl_uint(X)).
-define(cl_addressing_mode(X),     ?cl_uint(X)).
-define(cl_filter_mode(X),         ?cl_uint(X)).
-define(cl_sampler_info(X),        ?cl_uint(X)).
-define(cl_map_flags(X),           ?cl_bitfield(X)).
-define(cl_program_info(X),        ?cl_uint(X)).
-define(cl_program_build_info(X),  ?cl_uint(X)).
-define(cl_build_status(X),        ?cl_int(X)).
-define(cl_kernel_info(X),         ?cl_uint(X)).
-define(cl_kernel_work_group_info(X),  ?cl_uint(X)).
-define(cl_event_info(X),          ?cl_uint(X)).
-define(cl_command_type(X),        ?cl_uint(X)).
-define(cl_profiling_info(X),      ?cl_uint(X)).

-define(CL_CHAR_BIT,        8).
-define(CL_SCHAR_MAX,       127).
-define(CL_SCHAR_MIN,       (-127-1)).
-define(CL_CHAR_MAX,        ?CL_SCHAR_MAX).
-define(CL_CHAR_MIN,        ?CL_SCHAR_MIN).
-define(CL_UCHAR_MAX,       255).
-define(CL_SHRT_MAX,        32767).
-define(CL_SHRT_MIN,        (-32767-1)).
-define(CL_USHRT_MAX,       65535).
-define(CL_INT_MAX,         2147483647).
-define(CL_INT_MIN,         (-2147483647-1)).
-define(CL_UINT_MAX,        16#ffffffff).
-define(CL_LONG_MAX,        16#7FFFFFFFFFFFFFFF).
-define(CL_LONG_MIN,        (-16#7FFFFFFFFFFFFFFF-1)).
-define(CL_ULONG_MAX,       16#FFFFFFFFFFFFFFFF).

-define(CL_FLT_DIG,          6).
-define(CL_FLT_MANT_DIG,     24).
-define(CL_FLT_MAX_10_EXP,   38).
-define(CL_FLT_MAX_EXP,      128).
-define(CL_FLT_MIN_10_EXP,   -37).
-define(CL_FLT_MIN_EXP,      -125).
-define(CL_FLT_RADIX,        2).
-define(CL_FLT_MAX,          3.40282347e+38).
-define(CL_FLT_MIN,          1.17549435e-38).
-define(CL_FLT_EPSILON,      1.19209290e-07).

-define(CL_DBL_DIG,          15).
-define(CL_DBL_MANT_DIG,     53).
-define(CL_DBL_MAX_10_EXP,   308).
-define(CL_DBL_MAX_EXP,      1024).
-define(CL_DBL_MIN_10_EXP,   -307).
-define(CL_DBL_MIN_EXP,      -1021).
-define(CL_DBL_RADIX,        2).
-define(CL_DBL_MAX,          1.7976931348623157e+308).
-define(CL_DBL_MIN,          2.2250738585072014e-308).
-define(CL_DBL_EPSILON,      2.2204460492503131e-16).

-type cl_channel_order() :: 
	r | a | rg | ra | rgb | rgba | rgba | bgra | argb |
	intensity | luminance | rx | rgx | rgbx |
	%% 1.2
	depth | depth_stencil.
	
-type cl_channel_type() :: 
	snorm_int8 | snorm_int16 | unorm_int8 | unorm_int16 |
	unorm_short_565 | unorm_short_555 | unorm_int_101010 |
	signed_int8 | signed_int16 | signed_int32 | unsigned_int8 |
	unsigned_int16 | unsigned_int32 | half_float | float |
	%% 1.2 
	unorm_int24.

-type cl_mem_object_type() ::
	buffer | image2d | image3d |
	%% 1.2
	image2d_array | image1d | image1d_array | image1d_buffer.

-record(cl_image_format,
	{
	  cl_channel_order :: cl_channel_order(),
	  cl_channel_type  :: cl_channel_type()
	}).

%% 1.2 

-record(cl_image_desc,
	{
	  image_type  :: cl_mem_object_type(),
	  image_width :: non_neg_integer(),
	  image_height :: non_neg_integer(),
	  image_depth  :: non_neg_integer(),
	  image_array_size :: non_neg_integer(),
	  image_row_pitch ::  non_neg_integer(),
	  image_slice_pitch = 1 ::  non_neg_integer(),
	  num_mip_levels  = 0 ::  non_neg_integer(),
	  num_samples  = 0 ::  non_neg_integer(),
	  buffer :: cl_mem() %% when CL_MEM_OBJECT_IMAGE1D_BUFFER
	}).

%% cl platform & default contex
-record(cl,
	{
	  platform,  %% one platform !
	  devices,   %% devices selected
	  context    %% context for devices
	 }).

-endif.


