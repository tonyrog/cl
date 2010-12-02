%%
%% SquareFloat program adpoted from "Hello World" OpenCL examples by apple
%%
-module(cl_square_float).

-compile(export_all).

-import(lists, [map/2]).

-include("../include/cl.hrl").

-define(DATA_SIZE, 1024).

source() ->
"
__kernel void square( __global float* input, 
                      __global float* output,
                      const unsigned int count)
{
   int i = get_global_id(0);
   if (i < count)
      output[i] = input[i]*input[i];
}
".

test_data() ->
    << <<X:32/native-float>> || X <- lists:seq(1,?DATA_SIZE) >>.

dump_data(Bin) ->
    io:format("data=~p\n", [[ X || <<X:32/native-float>> <= Bin ]]).

test() ->
    test(all).
    
test(DevType) ->
    %% Create binary with floating points 1.0 ... 1024.0
    Data = test_data(),
    run(Data, DevType).

%%
%% execute a kernel that squares floating point numbers
%% now only one device is used (We run on cpu for debugging)
%%
run(Data, DevType) ->
    E = clu:setup(DevType),
    io:format("platform created\n"),
    {ok,Program} = clu:build_source(E, source()),
    io:format("program built\n"),

    N = byte_size(Data), %% number of bytes in indata
    Count = N div 4,     %% number of floats in indata

    %% Create input data memory (implicit copy_host_ptr)
    {ok,Input} = cl:create_buffer(E#cl.context,[read_only],N),
    io:format("input memory created\n"),

    %% Create the output memory
    {ok,Output} = cl:create_buffer(E#cl.context,[write_only],N),
    io:format("output memory created\n"),

    %% Create the command queue for the first device
    {ok,Queue} = cl:create_queue(E#cl.context,hd(E#cl.devices),[]),
    io:format("queue created\n"),

    %% Create the squre kernel object
    {ok,Kernel} = cl:create_kernel(Program, "square"),
    io:format("kernel created: ~p\n", [Kernel]),

    clu:apply_kernel_args(Kernel, [Input, Output, Count]),
    io:format("kernel args set\n"),

    %% Write data into input array 
    {ok,Event1} = cl:enqueue_write_buffer(Queue, Input, 0, N, Data, []),
    io:format("write data enqueued\n"),
    erlang:display_string("enqueu write\n"),

    Device = hd(E#cl.devices),
    {ok,Local} = cl:get_kernel_workgroup_info(Kernel, Device, work_group_size),
    io:format("work_group_size = ~p\n", [Local]),

    %% Enqueue the kernel
    Global = Count,
    {ok,Event2} = cl:enqueue_nd_range_kernel(Queue, Kernel,
					     [Global], [Local], [Event1]),
    io:format("nd range [~p, ~p] kernel enqueued\n",
	      [[Global],[Local]]),
    
    %% Enqueue the read from device memory (wait for kernel to finish)
    {ok,Event3} = cl:enqueue_read_buffer(Queue,Output,0,N,[Event2]),
    io:format("read buffer enqueued\n"),

    %% Now flush the queue to make things happend 
    ok = cl:flush(Queue),
    io:format("flushed\n"),

    %% Wait for Result buffer to be written
    io:format("wait\n"),
    io:format("Event1 = ~p\n", [cl:wait(Event1)]),
    io:format("Event2 = ~p\n", [cl:wait(Event2)]),
    Event3Res = cl:wait(Event3),
    io:format("Event3 = ~p\n", [Event3Res]),

    %%
    cl:release_mem_object(Input),
    cl:release_mem_object(Output),
    cl:release_queue(Queue),
    cl:release_kernel(Kernel),
    cl:release_program(Program),

    clu:teardown(E),
    {ok,EventResData} = Event3Res,
    dump_data(EventResData).
