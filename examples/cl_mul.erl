%%% File    : cl_mul.erl
%%% Author  : Tony Rogvall <tony@rogvall.se>
%%% Description : Multiply matrix with list of matrices
%%% Created : 16 Nov 2009 by Tony Rogvall <tony@rogvall.se>
-module(cl_mul).

-compile(export_all).

-include("../include/cl.hrl").

-define(DATA_SIZE, 1024).
-define(ITEM_SIZE, (16*4)).

encode_matrix ({ X1, X2, X3, X4
               , X5, X6, X7, X8
               , X9, X10,X11,X12
               , X13,X14,X15,X16}) ->
    <<?cl_float16( X1, X2, X3, X4
                 , X5, X6, X7, X8
                 , X9, X10,X11,X12
                 , X13,X14,X15,X16)>>;
encode_matrix ({float16, M}) ->
    encode_matrix(M).

decode_matrix (Data) ->
    case Data of
    << ?cl_float16( A11, A12, A13, A14
                  , A21, A22, A23, A24
                  , A31, A32, A33, A34
                  , A41, A42, A43, A44
                  ),
       Rest/binary >> ->
            [{ A11, A12, A13, A14
             , A21, A22, A23, A24
             , A31, A32, A33, A34
             , A41, A42, A43, A44 } | decode_matrix(Rest)];
        <<>> ->
            []
    end.

id_matrix () ->
    {float16, { 1, 0, 0, 0
              , 0, 1, 0, 0
              , 0, 0, 1, 0
              , 0, 0, 0, 1}}.

zero_matrix () ->
    {float16, { 0, 0, 0, 0
              , 0, 0, 0, 0
              , 0, 0, 0, 0
              , 0, 0, 0, 0}}.

r () -> random:uniform().

random_matrices (N) ->
    list_to_binary(
      [begin
           M = { r(), r(), r(), r()
               , r(), r(), r(), r()
               , r(), r(), r(), r()
               , r(), r(), r(), r()},
           encode_matrix(M)
       end || _ <- lists:seq(1, N)]).

test_data () ->
    random_matrices(4).

dump_data (Bin) ->
    io:format("data=~p\n", [decode_matrix(Bin)]).

test () ->
    test(all).

test (DevType) ->
    %% Create binary with floating points 1.0 ... 1024.0
    Data = test_data(),
    run(Data, DevType).

examples_dir () ->
    filename:join(code:lib_dir(cl), "examples").

%%
%% execute a kernel that squares floating point numbers
%% now only one device is used (We run on cpu for debugging)
%%
run (Data, DevType) ->
    E = clu:setup(DevType),
    io:format("platform created\n"),

    Filename = filename:join(examples_dir(),"mul4x4.cl"),
    io:format("build: ~s\n", [Filename]),
    {ok,Program} = clu:build_source_file(E, Filename),
    io:format("program built\n"),

    N = byte_size(Data),       %% number of bytes in indata
    Count = N div ?ITEM_SIZE,  %% number of matrices in indata

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
    {ok,Kernel} = cl:create_kernel(Program, "mul4x4"),
    io:format("kernel created: ~p\n", [Kernel]),

    dump_data(Data),

    %% Write data into input array
    {ok,Event1} = cl:enqueue_write_buffer(Queue, Input, 0, N, Data, []),
    io:format("write data enqueued\n"),
    erlang:display_string("enqueue write\n"),

    %% Set kernel arguments
    clu:apply_kernel_args(Kernel, [Input,Output,encode_matrix(id_matrix()),{uint,Count}]),
    io:format("kernel args set\n"),

    Device = hd(E#cl.devices),
    {ok,Local} = cl:get_kernel_workgroup_info(Kernel, Device, work_group_size),
    io:format("work_group_size = ~p\n", [Local]),

    %% Enqueue the kernel
    Global = Count,
    if Local > Count ->  LocalWork = Count;
       true ->        LocalWork = Local
    end,
    {ok,Event2} = cl:enqueue_nd_range_kernel(Queue, Kernel,
                         [Global], [LocalWork], [Event1]),
    io:format("nd range [~w, ~w] kernel enqueued\n",
          [[Global],[LocalWork]]),

    %% Enqueue the read from device memory (wait for kernel to finish)
    {ok,Event3} = cl:enqueue_read_buffer(Queue,Output,0,N,[Event2]),
    io:format("read buffer enqueued\n"),

    %% Now flush the queue to make things happend
    ok = cl:flush(Queue),
    io:format("flushed\n"),

    %% Wait for Result buffer to be written
    io:format("wait\n"),
    io:format("Event1 = ~p\n", [cl:wait(Event1,1000)]),
    io:format("Event2 = ~p\n", [cl:wait(Event2,1000)]),
    Event3Res = cl:wait(Event3,1000),
    io:format("Event3 = ~p\n", [Event3Res]),

    %%
    cl:release_mem_object(Input),
    cl:release_mem_object(Output),
    cl:release_queue(Queue),
    cl:release_kernel(Kernel),
    cl:release_program(Program),

    clu:teardown(E),
    case Event3Res of
    {ok,ResData} ->
        dump_data(ResData);
    _ ->
        ok
    end,
    Event3Res.
