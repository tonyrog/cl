%%
%% SquareFloat program adpoted from "Hello World" OpenCL examples by apple
%%
-module(cl_bandwidth).

-compile(export_all).

-import(lists, [map/2]).

-include("../include/cl.hrl").

-define(DATA_SIZE, 1*1024*1024).

test_data(Length) ->
    << <<X:32/native-float>> || X <- lists:duplicate(Length, 1) >>.

test() ->
    test(all).
    
test(DevType) ->
    %% Create binary with floating points 1.0 ... 1024.0
    Data = test_data(?DATA_SIZE),
    run(Data, DevType).

test(Length, DevType) when is_number(Length) ->
    Data = test_data(Length),
    run(Data, DevType).    

%%
%% execute a kernel that squares floating point numbers
%% now only one device is used (We run on cpu for debugging)
%%
run(Data, DevType) ->
    E = clu:setup(DevType),
    io:format("platform created\n"),

    N = byte_size(Data), %% number of bytes in indata

    io:format("Testing with byte size: ~p \n", [N]),

    %% Create input data memory (implicit copy_host_ptr)
    {ok,Input} = cl:create_buffer(E#cl.context,[read_only],N),
    io:format("input memory created\n"),

    %% Create the command queue for the first device
    {ok,Queue} = cl:create_queue(E#cl.context,hd(E#cl.devices),[]),
    io:format("queue created\n"),

    %% run benchmark on data messuring best write time
    {WriteTotal, WriteQueueTotal} =
	write_loop(1000, Queue, Input, Data, N),

    io:format("Bandwidth tested with write size: ~p bytes\n\n", [N]),
    
    io:format("Write total milliseconds: ~p\n", [WriteTotal]),
    io:format("Bandwidth rate: ~p KB per second\n\n", [trunc((N / (WriteTotal/1000))/1024)]),

    io:format("Queue total milliseconds: ~p\n", [WriteQueueTotal]),
    io:format("Bandwidth rate: ~p KB per second\n\n", [trunc((N / (WriteQueueTotal/1000))/1024)]),
    %%
    cl:release_mem_object(Input),
    cl:release_queue(Queue),

    clu:teardown(E).

write_loop(Max, Queue, Mem, Data, N) ->
    write_loop(Max, Queue, Mem, Data, N, undefined, 0.0).

write_loop(0, _Queue, _Mem, _Data, _N, TBest, TQBest) ->
    {TBest, TQBest};
write_loop(I, Queue, Mem, Data, N, TBest, TQBest) ->
    WriteQueueStart = erlang:now(),
    {ok,E1} = cl:enqueue_write_buffer(Queue, Mem, 0, N, Data, []),
    WriteQueueEnd = erlang:now(),
    WQT = timer:now_diff(WriteQueueEnd, WriteQueueStart)/1000,

    WriteStart = erlang:now(),
    ok = cl:flush(Queue),
    {ok,completed} = cl:wait(E1),
    WriteEnd = erlang:now(),
    WT = timer:now_diff(WriteEnd, WriteStart)/1000,
    if TBest =:= undefined; WT < TBest ->
	    write_loop(I-1, Queue, Mem, Data, N, WT, WQT);
       true ->
	    write_loop(I-1, Queue, Mem, Data, N, TBest, TQBest)
    end.

	    
	    


    
    
