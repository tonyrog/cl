%%% @author Tony Rogvall <tony@rogvall.se>
%%% @copyright (C) 2014, Tony Rogvall
%%% @doc
%%%    A opencl compiler wrapper
%%% @end
%%% Created :  9 May 2014 by Tony Rogvall <tony@rogvall.se>

-module(cl_compile).

-compile(export_all).

%% compile File into binary,

file(File) ->  file(File,all).
file(File,Type) -> file(File,Type,"").

file(File,Type,Options) ->
    Clu = clu:setup(Type),
    case clu:build_source_file(Clu, File, Options) of
	Err = {error,_} ->
	    Err;
	{ok,Program} ->
	    info(Program)
    end.

info(Program) ->
    {ok,Ds} = cl:get_program_info(Program, devices),
    {ok,Bs} = cl:get_program_info(Program, binaries),
    lists:foreach(fun(Device) -> build_info(Program, Device)  end, Ds),
    program_info(Program),
    {ok,Kernels} = cl:create_kernels_in_program(Program),
    lists:foreach(
      fun(Kernel) ->
	      {ok,KernelInfo} = cl:get_kernel_info(Kernel),
	      io:format("KernelInfo: ~p\n", [KernelInfo]),
	      lists:foreach(
		fun(Device) ->
			{ok,I}=cl:get_kernel_workgroup_info(Kernel,Device),
			io:format("KernelWorkGroupInfo: ~p\n", [I])
		end, Ds),
	      case lists:member({1,2}, cl:versions()) of
		  true ->
		      {ok,ArgInfo} = cl:get_kernel_arg_info(Kernel),
		      io:format("arg_info: ~p\n", [ArgInfo]);
		  false ->
		      ok
	      end
      end, Kernels),
    {ok,Bs}.


program_info(Program) ->
    io:format("ProgramInfo:\n", []),
    lists:foreach(
      fun(Attr) ->
	      case cl:get_program_info(Program,Attr) of
		  {ok,Value} ->
		      io:format("  ~s: ~p\n", [Attr,Value]);
		  {error,Reason} ->
		      io:format("InfoError: ~s [~p]\n", 
				[Attr,Reason])
	      end
      end, cl:program_info()).

build_info(Program, Device) ->
    io:format("BuildInfo @ ~w\n", [Device]),
    {ok,BuildInfo} = cl:get_program_build_info(Program,Device),
    lists:foreach(
      fun({Attr,Value}) ->
	      io:format("  ~s: ~p\n", [Attr,Value])
      end, BuildInfo),
    case lists:member({1,2}, cl:versions()) of
	true ->
	    %% fixme: version handle program_build_info 
	    case cl:get_program_build_info(Program,Device,binary_type) of
		{ok,BinaryInfo} ->
		    io:format("  ~s: ~p\n", [binary_type,BinaryInfo]);
		{error,Reason} ->
		    io:format("InfoError: ~s [~p]\n", 
			      [binary_type,Reason])
	    end;
	false ->
	    ok
    end.

%% compile & link with openCL version 1.2

inc1() -> "
#define FOO 5
".

inc2() -> "
#define BAR 7
".

prog1() -> "
#include \"inc1.h\"\n
#include \"inc2.h\"\n

__kernel void sum(int x, int y, __global int* z)
{
  int i = get_global_id(0);
  z[i] = x + y + FOO + BAR + BAZ;
}
".

prog2() -> "
#define FOO 5
#define BAR 7

__kernel void prod(int x, int y, __global int* z)
{
  int i = get_global_id(0);
  z[i] = x*y*FOO*BAR + BAZ;
}
".

make_prog(Clu,prog1) ->
    {ok,Program} = cl:create_program_with_source(clu:context(Clu), prog1()),
    {ok,Inc1} = cl:create_program_with_source(clu:context(Clu), inc1()),
    {ok,Inc2} = cl:create_program_with_source(clu:context(Clu), inc2()),
    {Program, [Inc1,Inc2], ["inc1.h", "inc2.h"]};
make_prog(Clu,prog2) ->
    {ok,Program} = cl:create_program_with_source(clu:context(Clu), prog2()),
    {Program, [], []}.

%% MackBookPro, mac os x 10.9 with GEForce 9400M test_12(gpu,prog1)
%% fail with an error saying that the compiler can not find include
%% files 'inc1.h'
test_12() ->
    test_12(prog1, cpu).

test_12(Prog, Type) ->
    true = lists:member({1,2}, cl:versions()),
    Clu = clu:setup(Type),
    compile_12(Clu, Prog).

compile_12(Clu, Prog) ->
    {Program,Includes,IncludeNames} = make_prog(Clu,Prog),
    Ds = clu:device_list(Clu),
    case cl:compile_program(Program,Ds,"-DBAZ=11",
			    Includes, IncludeNames) of
	ok ->
	    Status = [get_build_status(Program, Dev) || Dev <- Ds],
	    case lists:any(fun(success) -> true;
			      (_) -> false end, Status) of
		true ->
		    {ok,Program};
		false ->
		    Logs = get_program_logs(Program),
		    io:format("Logs: ~s\n", [Logs]),
		    {error,{Status,Logs}}
	    end;
	Error ->
	    Logs = get_program_logs(Program),
	    io:format("Logs: ~s\n", [Logs]),
	    cl:release_program(Program),
	    {error,{Error,Logs}}
    end.

link_12(Type) ->
    link_12(prog1,Type).

link_12(Prog,Type) ->
    true = lists:member({1,2}, cl:versions()),
    Clu = clu:setup(Type),
    {ok,Prog1} = compile_12(Clu, Prog),
    io:format("Prog1 = ~p\n", [Prog1]),
%%    {ok,Prog2} = compile_12(Clu, prog2),
%%    io:format("Prog2 = ~p\n", [Prog2]),
    case cl:link_program(clu:context(Clu),
			 clu:device_list(Clu),
			 "",
			 [Prog1]) of
	{ok, Program} ->
	    %% check status & logs
	    get_program_binaries(Program);
	Error ->
	    Error
    end.

get_build_status(Program, Device) ->
    {ok,Status} = cl:get_program_build_info(Program, Device, status),
    {ok,BinaryType} = cl:get_program_build_info(Program, Device, binary_type),
    io:format("status: ~p, binary_type=~p\n", [Status, BinaryType]),
    Status.

get_program_logs(Program) ->
    {ok,DeviceList} = cl:get_program_info(Program, devices),
    lists:map(
      fun(Device) ->
	      {ok,Log} = cl:get_program_build_info(Program,Device,log),
	      Log
      end, DeviceList).

get_program_binaries(Program) ->
    {ok,DeviceList} = cl:get_program_info(Program, devices),
    {ok,BinaryList} = cl:get_program_info(Program, binaries),
    {ok,{DeviceList, BinaryList}}.
