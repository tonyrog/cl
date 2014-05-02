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
%%% File    : clu.erl
%%% Author  : Tony Rogvall <tony@rogvall.se>
%%% Description : Utilities
%%% Created : 30 Oct 2009 by Tony Rogvall <tony@rogvall.se>

-module(clu).

-export([setup/0, setup/1, teardown/1]).
-export([context/1, device_list/1, device/1]).
-export([build_source/2]).
-export([build_binary/2]).
-export([build_source_file/2, compile_file/1]).
-export([get_program_binaries/1]).
-export([apply_kernel_args/2]).
-export([wait_complete/1]).

-include("../include/cl.hrl").
-import(lists, [map/2]).

%%
%% @type clu_state() = any()
%%
%%
%% @doc setup the platform and an initial context using
%%  devices of type DevType. Setup currently use the
%%  first platform found only.
%% @spec setup(DevType::cl_device_type()) ->  clu_state()
%%

setup(DevType) ->
    cl:start(),
    {ok,Ps} = cl:get_platform_ids(),
    setup(DevType, Ps).

setup(DevType, [Platform|Ps]) ->
    case cl:get_device_ids(Platform,DevType) of
	{ok, []} ->
	    setup(DevType, Ps);
	{ok,DeviceList} ->
	    case cl:create_context(DeviceList) of
		{ok,Context} ->
		    #cl { platform = Platform,
			  devices  = DeviceList,
			  context  = Context };
		{error, _} when Ps /= [] ->
		    setup(DevType, Ps);
		Other ->
		    Other
	    end;
	{error, device_not_found} ->
	    setup(DevType, Ps);
	{error, _} when Ps /= [] ->
	    setup(DevType, Ps);
	Other ->
	    Other
    end;
setup(DevType, []) ->
    {error, {device_not_found, DevType}}.


%%
%% @doc setup a clu context with all devices.
%%
%% @spec setup() -> clu_state()
%%
setup() ->
    setup(all).

%%
%% @doc Release the context setup by clu:setup().
%%
%% @spec teardown(E::clu_state()) -> 'ok' | {'error',cl_error()}
%%
teardown(E) ->
    cl:release_context(E#cl.context).
%%
%% Fetch context
%%
context(E) ->
    E#cl.context.
%%
%% Fetch device list
%%
device_list(E) ->
    E#cl.devices.

%%
%% Fetch first device
%%
device(E) ->
    hd(E#cl.devices).

%%
%% @doc Create and build a OpenCL program from a string.
%%
%% @spec build_source(E::clu_state(), Source::iodata()) ->
%%   {'ok',cl_program()} | {'error',{cl_error(), Logs}}
%%

build_source(E, Source) ->
    {ok,Program} = cl:create_program_with_source(E#cl.context,Source),
    case cl:build_program(Program, E#cl.devices, "") of
	ok ->
	    Status = [cl:get_program_build_info(Program, Dev, status)
		      || Dev <- E#cl.devices],
	    case lists:any(fun({ok, success}) -> true; 
			      (_) -> false end, Status) 
	    of
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


build_source_file(E, File) ->
    case file:read_file(File) of
	{ok,Binary} ->
	    build_source(E, Binary);
	Error ->
	    Error
    end.

compile_file(File) ->
    E = setup(all),
    Result = build_source_file(E, File),
    Res =
	case Result of
	    {error,{_,_Logs}} ->
		%% Listed in build_source, should it be?
		%% lists:foreach(
		%%   fun(Log) -> io:format("~s\n", [Log]) end, 
		%%   Logs),
		Result;
	    {ok,Program} ->
		BRes = get_program_binaries(Program),
		cl:release_program(Program),
		BRes;
	    Error ->
		Error
	end,
    teardown(E),
    Res.

%% @doc Retrieve the binaries associated with a program build.
%%  the binaries may be cached for later use with build_binary/2.
%%
%% @spec get_program_binaries(Program::cl_program()) ->
%%  {ok,{[cl_device_id()],[binary()]}}
%%

get_program_binaries(Program) ->
    {ok,DeviceList} = cl:get_program_info(Program, devices),
    {ok,BinaryList} = cl:get_program_info(Program, binaries),
    {ok,{DeviceList, BinaryList}}.

get_program_logs(Program) ->
    {ok,DeviceList} = cl:get_program_info(Program, devices),
    map(fun(Device) ->
		{ok,Log} = cl:get_program_build_info(Program,Device,log),
		Log
	end, DeviceList).


build_binary(E, {DeviceList,BinaryList}) ->
    {ok,Program} = cl:create_program_with_binary(E#cl.context, DeviceList, BinaryList),
    case cl:build_program(Program, DeviceList, "") of
	ok ->
	    {ok,Program};
	Error ->
	    Logs = 
		map(fun(Device) ->
			    {ok,Log} = cl:get_program_build_info(Program,
								  Device,log),
			    Log
		    end, E#cl.devices),
	    {error,{Error,Logs}}
    end.

%%
%% Utility to set all kernel arguments (and do arity check)
%%
apply_kernel_args (Kernel, Args) ->
    {ok,N} = cl:get_kernel_info(Kernel, num_args),
    Arity = length(Args),
    if N /= Arity ->
	    {ok,Name} = cl:get_kernel_info(Kernel, function_name),
	    erlang:error({bad_arity,Name,N});
       true ->
	    try
		apply_args(Kernel, 0, Args)
	    catch
		error:{badmatch,Error} ->
		    erlang:error(Error)
	    end
    end.

apply_args (Kernel, I, [{local,Size}|As]) ->
    %%io:format("kernel set arg ~w size to ~p\n", [I,Size]),
    ok = cl:set_kernel_arg_size(Kernel, I, Size),
    apply_args(Kernel, I+1, As);
apply_args (Kernel, I, [A|As]) ->
    %%io:format("kernel set arg ~w to ~p\n", [I,A]),
    ok = cl:set_kernel_arg(Kernel, I, A),
    apply_args(Kernel, I+1, As);
apply_args (_Kernel, _I, []) ->
    ok.

%% manual wait for event to complete (crash on failure)
%% should test for error status
wait_complete(Event) ->
    case cl:get_event_info(Event, execution_status) of
	{ok,complete} ->
	    ok;
	{ok,Other} ->
	    io:format("Status: ~p\n", [Other]),
	    timer:sleep(100),
	    wait_complete(Event)
    end.

