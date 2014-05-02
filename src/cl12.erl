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
%%% @author Tony Rogvall <tony@rogvall.se>
%%% @copyright (C) 2013, Tony Rogvall
%%% @doc
%%%    OpenCL 1.2 API
%%% @end
%%% Created : 13 Jan 2013 by Tony Rogvall <tony@rogvall.se>

-module(cl12).

-on_load(init/0).

-export([start/0, start/1, stop/0]).
-export([get_platform_ids/0]).
-export([platform_info/0]).
-export([get_platform_info/1,get_platform_info/2]).
-export([get_device_ids/0, get_device_ids/2]).
-export([device_info/0]).
-export([get_device_info/1,get_device_info/2]).
-export([create_context/1]).
-export([create_context_from_type/1]).
-export([release_context/1]).
-export([retain_context/1]).
-export([context_info/0]).
-export([get_context_info/1,get_context_info/2]).
-export([create_queue/3]).
-export([set_queue_property/3]).
-export([release_queue/1]).
-export([retain_queue/1]).
-export([queue_info/0]).
-export([get_queue_info/1,get_queue_info/2]).
-export([create_buffer/3, create_buffer/4]).
-export([release_mem_object/1]).
-export([retain_mem_object/1]).
-export([mem_object_info/0]).
-export([get_mem_object_info/1,get_mem_object_info/2]).
-export([image_info/0]).
-export([get_image_info/1,get_image_info/2]).
-export([get_supported_image_formats/3]).
-export([create_image/5]).
-export([create_sampler/4]).
-export([release_sampler/1]).
-export([retain_sampler/1]).
-export([sampler_info/0]).
-export([get_sampler_info/1,get_sampler_info/2]).
-export([create_program_with_source/2]).
-export([create_program_with_binary/3]).
-export([release_program/1]).
-export([retain_program/1]).
-export([build_program/3, async_build_program/3]).
-export([unload_platform_compiler/1]).
-export([program_info/0]).
-export([get_program_info/1,get_program_info/2]).
-export([program_build_info/0]).
-export([get_program_build_info/2,get_program_build_info/3]).
-export([create_kernel/2]).
-export([create_kernels_in_program/1]).
-export([set_kernel_arg/3]).
-export([set_kernel_arg_size/3]).
-export([release_kernel/1]).
-export([retain_kernel/1]).
-export([kernel_info/0]).
-export([get_kernel_info/1,get_kernel_info/2]).
-export([kernel_workgroup_info/0]).
-export([get_kernel_workgroup_info/2,get_kernel_workgroup_info/3]).
-export([enqueue_task/3, enqueue_task/4]).
-export([nowait_enqueue_task/3]).
-export([enqueue_nd_range_kernel/5]).
-export([enqueue_nd_range_kernel/6]).
-export([nowait_enqueue_nd_range_kernel/5]).
-export([enqueue_marker_with_wait_list/2]).
-export([enqueue_barrier_with_wait_list/2]).
-export([enqueue_wait_for_events/2]).
-export([enqueue_read_buffer/5]).
-export([enqueue_write_buffer/6]).
-export([enqueue_write_buffer/7]).
-export([nowait_enqueue_write_buffer/6]).
-export([enqueue_read_image/7]).
-export([enqueue_write_image/8]).
-export([enqueue_write_image/9]).
-export([nowait_enqueue_write_image/8]).
-export([enqueue_copy_image/6]).
-export([enqueue_copy_image_to_buffer/7]).
-export([enqueue_copy_buffer_to_image/7]).
-export([enqueue_map_buffer/6]).
-export([enqueue_map_image/6]).
-export([enqueue_unmap_mem_object/3]).
-export([release_event/1]).
-export([retain_event/1]).
-export([event_info/0]).
-export([get_event_info/1, get_event_info/2]).
-export([wait/1, wait/2]).
-export([async_flush/1, flush/1]).
-export([async_finish/1, finish/1]).
-export([async_wait_for_event/1, wait_for_event/1]).

init() ->
    case lists:member({1,2}, cl:versions()) of
	false -> erlang:error(cl_1_2_not_supported);
	true -> ok
    end.

start(Args) ->  cl:start(Args).
start() -> cl:start().
stop() -> cl:stop().
get_platform_ids() -> cl:get_platform_ids().
platform_info() -> cl:platform_info().
get_platform_info(A1) -> cl:get_platform_info(A1).
get_platform_info(A1,A2) -> cl:get_platform_info(A1,A2).
get_device_ids() -> cl:get_device_ids().
get_device_ids(A1,A2) -> cl:get_device_ids(A1,A2).
device_info() ->
    cl:device_info_10(cl:device_info_11(cl:device_info_12([]))).
get_device_info(A1) -> cl:get_device_info(A1).
get_device_info(A1,A2) -> cl:get_device_info(A1,A2).
create_context(A1) -> cl:create_context(A1).
create_context_from_type(A1) -> cl:create_context_from_type(A1).
release_context(A1) -> cl:release_context(A1).
retain_context(A1) -> cl:retain_context(A1).
context_info() -> cl:context_info().
get_context_info(A1) -> cl:get_context_info(A1).
get_context_info(A1,A2) -> cl:get_context_info(A1,A2).
create_queue(A1,A2,A3) -> cl:create_queue(A1,A2,A3).
set_queue_property(A1,A2,A3) -> cl:set_queue_property(A1,A2,A3).
release_queue(A1) -> cl:release_queue(A1).
retain_queue(A1) -> cl:retain_queue(A1).
queue_info() -> cl:queue_info().
get_queue_info(A1) -> cl:get_queue_info(A1).
get_queue_info(A1,A2) -> cl:get_queue_info(A1,A2).
create_buffer(A1,A2,A3) -> cl:create_buffer(A1,A2,A3).
create_buffer(A1,A2,A3,A4) -> cl:create_buffer(A1,A2,A3,A4).
release_mem_object(A1) -> cl:release_mem_object(A1).
retain_mem_object(A1) -> cl:retain_mem_object(A1).
mem_object_info() -> cl:mem_object_info().
get_mem_object_info(A1) -> cl:get_mem_object_info(A1).
get_mem_object_info(A1,A2) -> cl:get_mem_object_info(A1,A2).
image_info() -> cl:image_info().
get_image_info(A1) -> cl:get_image_info(A1).
get_image_info(A1,A2) -> cl:get_image_info(A1,A2).
get_supported_image_formats(A1,A2,A3) -> cl:get_supported_image_formats(A1,A2,A3).
create_image(A1,A2,A3,A4,A5) -> cl:create_image(A1,A2,A3,A4,A5).
create_sampler(A1,A2,A3,A4) -> cl:create_sampler(A1,A2,A3,A4).
release_sampler(A1) -> cl:release_sampler(A1).
retain_sampler(A1) -> cl:retain_sampler(A1).
sampler_info() -> cl:sampler_info().
get_sampler_info(A1) -> cl:get_sampler_info(A1).
get_sampler_info(A1,A2) -> cl:get_sampler_info(A1,A2).
create_program_with_source(A1,A2) -> cl:create_program_with_source(A1,A2).
create_program_with_binary(A1,A2,A3) -> cl:create_program_with_binary(A1,A2,A3).
release_program(A1) -> cl:release_program(A1).
retain_program(A1) -> cl:retain_program(A1).
build_program(A1,A2,A3) -> cl:build_program(A1,A2,A3).
async_build_program(A1,A2,A3) -> cl:async_build_program(A1,A2,A3).
unload_platform_compiler(A1) -> cl:unload_platform_compiler(A1).
program_info() -> cl:program_info().
get_program_info(A1) -> cl:get_program_info(A1).
get_program_info(A1,A2) -> cl:get_program_info(A1,A2).
program_build_info() -> cl:program_build_info().
get_program_build_info(A1,A2) -> cl:get_program_build_info(A1,A2).
get_program_build_info(A1,A2,A3) -> cl:get_program_build_info(A1,A2,A3).
create_kernel(A1,A2) -> cl:create_kernel(A1,A2).
create_kernels_in_program(A1) -> cl:create_kernels_in_program(A1).
set_kernel_arg(A1,A2,A3) -> cl:set_kernel_arg(A1,A2,A3).
set_kernel_arg_size(A1,A2,A3) -> cl:set_kernel_arg_size(A1,A2,A3).
release_kernel(A1) -> cl:release_kernel(A1).
retain_kernel(A1) -> cl:retain_kernel(A1).
kernel_info() -> cl:kernel_info().
get_kernel_info(A1) -> cl:get_kernel_info(A1).
get_kernel_info(A1,A2) -> cl:get_kernel_info(A1,A2).
kernel_workgroup_info() -> cl:kernel_workgroup_info().
get_kernel_workgroup_info(A1,A2) -> cl:get_kernel_workgroup_info(A1,A2).
get_kernel_workgroup_info(A1,A2,A3) -> cl:get_kernel_workgroup_info(A1,A2,A3).
enqueue_task(A1,A2,A3) -> cl:enqueue_task(A1,A2,A3). 
enqueue_task(A1,A2,A3,A4) -> cl:enqueue_task(A1,A2,A3,A4).
nowait_enqueue_task(A1,A2,A3) -> cl:nowait_enqueue_task(A1,A2,A3).
enqueue_nd_range_kernel(A1,A2,A3,A4,A5) -> 
    cl:enqueue_nd_range_kernel(A1,A2,A3,A4,A5).
enqueue_nd_range_kernel(A1,A2,A3,A4,A5,A6) ->
    cl:enqueue_nd_range_kernel(A1,A2,A3,A4,A5,A6).
nowait_enqueue_nd_range_kernel(A1,A2,A3,A4,A5) ->
    cl:nowait_enqueue_nd_range_kernel(A1,A2,A3,A4,A5).
enqueue_marker_with_wait_list(A1,A2) -> 
    cl:enqueue_marker_with_wait_list(A1,A2).
enqueue_barrier_with_wait_list(A1,A2) ->
    cl:enqueue_barrier_with_wait_list(A1,A2).
enqueue_wait_for_events(A1,A2) -> 
    cl:enqueue_wait_for_events(A1,A2).
enqueue_read_buffer(A1,A2,A3,A4,A5) ->
    cl:enqueue_read_buffer(A1,A2,A3,A4,A5).
enqueue_write_buffer(A1,A2,A3,A4,A5,A6) ->
    cl:enqueue_write_buffer(A1,A2,A3,A4,A5,A6).
enqueue_write_buffer(A1,A2,A3,A4,A5,A6,A7) ->
    cl:enqueue_write_buffer(A1,A2,A3,A4,A5,A6,A7).
nowait_enqueue_write_buffer(A1,A2,A3,A4,A5,A6) ->
    cl:nowait_enqueue_write_buffer(A1,A2,A3,A4,A5,A6).
enqueue_read_image(A1,A2,A3,A4,A5,A6,A7) ->
    cl:enqueue_read_image(A1,A2,A3,A4,A5,A6,A7).
enqueue_write_image(A1,A2,A3,A4,A5,A6,A7,A8) ->
    cl:enqueue_write_image(A1,A2,A3,A4,A5,A6,A7,A8).
enqueue_write_image(A1,A2,A3,A4,A5,A6,A7,A8,A9) ->
    cl:enqueue_write_image(A1,A2,A3,A4,A5,A6,A7,A8,A9).
nowait_enqueue_write_image(A1,A2,A3,A4,A5,A6,A7,A8) ->
    cl:nowait_enqueue_write_image(A1,A2,A3,A4,A5,A6,A7,A8).
enqueue_copy_image(A1,A2,A3,A4,A5,A6) ->
    cl:enqueue_copy_image(A1,A2,A3,A4,A5,A6).
enqueue_copy_image_to_buffer(A1,A2,A3,A4,A5,A6,A7) ->
    cl:enqueue_copy_image_to_buffer(A1,A2,A3,A4,A5,A6,A7).
enqueue_copy_buffer_to_image(A1,A2,A3,A4,A5,A6,A7) ->
    cl:enqueue_copy_buffer_to_image(A1,A2,A3,A4,A5,A6,A7).
enqueue_map_buffer(A1,A2,A3,A4,A5,A6) ->
    cl:enqueue_map_buffer(A1,A2,A3,A4,A5,A6).
enqueue_map_image(A1,A2,A3,A4,A5,A6) ->
    cl:enqueue_map_image(A1,A2,A3,A4,A5,A6).
enqueue_unmap_mem_object(A1,A2,A3) ->
    cl:enqueue_unmap_mem_object(A1,A2,A3).
release_event(A1) -> cl:release_event(A1).
retain_event(A1) -> cl:retain_event(A1).
event_info() -> cl:event_info().
get_event_info(A1) -> cl:get_event_info(A1).
get_event_info(A1,A2) -> cl:get_event_info(A1,A2).
wait(A1) -> cl:wait(A1). 
wait(A1,A2) -> cl:wait(A1,A2).
async_flush(A1) -> cl:async_flush(A1). 
flush(A1) -> cl:flush(A1).
async_finish(A1) -> cl:async_finish(A1).
finish(A1) -> cl:finish(A1).
async_wait_for_event(A1) -> cl:async_wait_for_event(A1).
wait_for_event(A1) -> cl:wait_for_event(A1).
