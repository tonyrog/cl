%% Basic tests
-module(cl_basic).

-compile(export_all).
-import(lists, [foreach/2]).

-include("../include/cl.hrl").

test() ->
    test(all).

test(DevType) ->
    E = clu:setup(DevType),
    {ok,PlatformInfo} = cl:get_platform_info(E#cl.platform),
    io:format("PlatformInfo: ~p\n", [PlatformInfo]),

    foreach(
      fun(Device) ->
	      io:format("Device: ~p\n", [Device]),
	      {ok,DeviceInfo} = cl:get_device_info(Device),
	      io:format("DeviceInfo: ~p\n", [DeviceInfo])
      end, E#cl.devices),

    {ok,ContextInfo} = cl:get_context_info(E#cl.context),
    io:format("ContextInfo: ~p\n", [ContextInfo]),
    cl:retain_context(E#cl.context),
    {ok,ContextInfo2} = cl:get_context_info(E#cl.context),
    io:format("Context2: ~p\n", [ContextInfo2]),

    foreach(fun(Device) ->
		    test_queue(E, Device)  end, 
	    E#cl.devices),

    foreach(fun(Device) ->
		    test_sampler(E, Device)  end, 
	    E#cl.devices),

    test_program(E#cl.context, E#cl.devices),

    clu:teardown(E).

test_program(Context, DeviceList) ->
    %% Program1
    Source1 = "
__kernel void program1(int n, int m) {
    int result = n + m;
}
",
    {ok,Program} = cl:create_program_with_source(Context,Source1),
    io:format("Program: ~p\n", [Program]),
    {ok,Info} = cl:get_program_info(Program),
    io:format("ProgramInfo: ~p\n", [Info]),
    foreach(
      fun(Device) ->
	      {ok,BuildInfo} = cl:get_program_build_info(Program,Device),
	      io:format("BuildInfo @ ~w: ~p\n", [Device,BuildInfo])
      end, DeviceList),

    case cl:build_program(Program, DeviceList, "-Dhello=1 -Dtest") of
	ok ->
	    foreach(
	      fun(Device) ->
		      {ok,BuildInfo} = cl:get_program_build_info(Program,Device),
		      io:format("BuildInfo @ ~w: ~p\n", [Device,BuildInfo])
	      end, DeviceList),
	    {ok,Info1} = cl:get_program_info(Program),
	    io:format("ProgramInfo1: ~p\n", [Info1]),
	    {ok,Kernels} = cl:create_kernels_in_program(Program),
	    foreach(
	      fun(Kernel) ->
		      {ok,KernelInfo} = cl:get_kernel_info(Kernel),
		      io:format("KernelInfo: ~p\n", [KernelInfo]),
		      foreach(
			fun(Device) ->
				{ok,I}=cl:get_kernel_workgroup_info(Kernel,Device),
				io:format("KernelWorkGroupInfo: ~p\n", [I])
			end, DeviceList)
	      end, Kernels),
	    foreach(
	      fun(Device) ->
		      {ok,Queue} = cl:create_queue(Context,Device,[]),
		      foreach(
			fun(Kernel) ->
				cl:set_kernel_arg(Kernel, 0, 12),
				cl:set_kernel_arg(Kernel, 1, 13),
				{ok,Event} = cl:enqueue_task(Queue, Kernel, []),
				{ok,EventInfo} = cl:get_event_info(Event),
				io:format("EventInfo: ~p\n", [EventInfo]),
				cl:flush(Queue),
				io:format("Event Status:=~p\n", 
					  [cl:wait(Event,1000)])
			end, Kernels)
	      end, DeviceList),
	    ok;
	Error ->
	    io:format("Build Error: ~p\n", [Error])
    end,
    cl:release_program(Program),
    ok.
    

test_queue(E, Device) ->
    {ok,Queue} = cl:create_queue(E#cl.context,Device,[]),
    io:format("Queue: ~p\n", [Queue]),
    {ok,QueueInfo} = cl:get_queue_info(Queue),
    io:format("QueueInfo: ~p\n", [QueueInfo]),
    cl:release_queue(Queue),
    ok.
    

test_buffer(E) ->
    %% Read/Write buffer
    {ok,Buffer} = cl:create_buffer(E#cl.context,[read_write],1024),
    io:format("Buffer: ~p\n", [Buffer]),
    {ok,BufferInfo} = cl:get_mem_object_info(Buffer),
    io:format("BufferInfo: ~p\n", [BufferInfo]),    
    cl:release_mem_object(Buffer),

    %% Read only buffer
    {ok,Buffer2} = cl:create_buffer(E#cl.context,[read_only],0,
				     <<"Hello brave new world">>),
    io:format("Buffer2: ~p\n", [Buffer2]),
    {ok,Buffer2Info} = cl:get_mem_object_info(Buffer2),
    io:format("Buffer2Info: ~p\n", [Buffer2Info]),
    cl:release_mem_object(Buffer2),
    ok.

    

test_sampler(E, Device) ->
    {ok,DeviceInfo} = cl:get_device_info(Device),
    Name = proplists:get_value(name, DeviceInfo),
    case proplists:get_value(image_support, DeviceInfo) of
	true ->
	    %% Sampler1
	    {ok,Sampler1} = cl:create_sampler(E#cl.context,true,clamp,nearest),
	    io:format("Sampler1: ~p\n", [Sampler1]),
	    {ok,Sampler1Info} = cl:get_sampler_info(Sampler1),
	    io:format("Sampler1Info: ~p\n", [Sampler1Info]),
	    cl:release_sampler(Sampler1),
	    
	    %% Sampler2
	    {ok,Sampler2} = cl:create_sampler(E#cl.context,false,repeat,linear),
	    io:format("Sampler2: ~p\n", [Sampler2]),
	    {ok,Sampler2Info} = cl:get_sampler_info(Sampler2),
	    io:format("Sampler2Info: ~p\n", [Sampler2Info]),
	    cl:release_sampler(Sampler2),
	    ok;
	false ->
	    io:format("No image support for device ~s ~n",[Name])	    
    end.



	      
    
    
    
