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
