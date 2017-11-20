%%% @author Tony Rogvall <tony@rogvall.se>
%%% @copyright (C) 2010, Tony Rogvall
%%% @doc
%%%
%%% @end
%%% Created : 25 Dec 2010 by Tony Rogvall <tony@rogvall.se>

-module(cl_test).

-export([init_per_suite/1, end_per_suite/1]).
-export([all/0, ct_test0/1, ct_test1/1]).

-export([test0/0, test0/2]).
-export([test1/0, test1/2]).

-import(lists, [foreach/2]).

-define(BUFFER_SIZE, 1024*256).

-spec init_per_suite(Config0::list(tuple())) ->
                            (Config1::list(tuple())) | 
                            {skip,Reason::term()} | 
                            {skip_and_save,Reason::term(),
			     Config1::list(tuple())}.

init_per_suite(Config) -> cl_SUITE:init_per_suite(Config).

-spec end_per_suite(Config::list(tuple())) -> ok.

end_per_suite(_Config) ->
    ok.


all() -> [ct_test0, ct_test1].

test0() ->
    test0(cpu, ?BUFFER_SIZE).

ct_test0(Config) ->
    test0(proplists:get_value(type, Config, gpu), ?BUFFER_SIZE).

test0(Type, Size) ->
    {ok,[PI|_]} = cl:get_platform_ids(),
    {ok,[D]} = cl:get_device_ids(PI, Type),
    {ok,C} = cl:create_context([D]),
    {ok,Q} = cl:create_queue(C, D, []),
    {ok,Buf} = cl:create_buffer(C, [read_only], Size),
    N = Size div 2,
    Data = make_buffer(N),
    {ok,E1} = cl:enqueue_write_buffer(Q, Buf, 0, N, Data, []),
    {ok,E2} = cl:enqueue_write_buffer(Q, Buf, N, N, Data, []),
    Res1 = cl:wait(E1,3000),
    io:format("Res1 = ~p\n", [Res1]),
    Res2 = cl:wait(E2,3000),
    io:format("Res2 = ~p\n", [Res2]),
    {ok,E3} = cl:enqueue_read_buffer(Q, Buf, 0, N, []),
    case cl:wait(E3,3000) of
	{ok, Data} -> 
	    io:format("read_buffer: verified\n"),
	    ok;
	Res3 ->
	    io:format("Res3 = ~p\n", [Res3])
    end.

program(ok) -> "
__kernel void program1(int n, int m) {
    int result = n + m;
}
";
program(error) -> "
__kernel void program1(int n, int m) {
    int result = n + k;
}
".

ct_test1(Config) ->
    test1(proplists:get_value(type, Config, gpu), ok).

test1() ->
    test1(cpu, ok).

test1(Type, Prog) ->
    {ok,[PI|_]} = cl:get_platform_ids(),
    {ok,DeviceList} = cl:get_device_ids(PI, Type),
    {ok,C} = cl:create_context(DeviceList),
    {ok,P} = cl:create_program_with_source(C, program(Prog)),
    io:format("Program: ~p\n", [P]),
    {ok,Info} = cl:get_program_info(P),
    io:format("ProgramInfo: ~p\n", [Info]),
    foreach(
      fun(D) ->
	      {ok,BuildInfo} = cl:get_program_build_info(P,D),
	      io:format("BuildInfo @ ~w: ~p\n", [D,BuildInfo])
      end, DeviceList),
    case cl:build_program(P, DeviceList, "-Dhello=1 -Dtest") of
	ok ->
	    foreach(
	      fun(D) ->
		      {ok,BuildInfo} = cl:get_program_build_info(P,D),
		      io:format("BuildInfo @ ~w: ~p\n", [D,BuildInfo])
	      end, DeviceList),
	    ok;
	Error ->
	    io:format("\n\nBuild Error: ~p\n\n", [Error]),
	    foreach(
	      fun(D) ->
		      {ok,BuildInfo} = cl:get_program_build_info(P,D),
		      io:format("BuildInfo @ ~w: ~p\n", [D,BuildInfo])
	      end, DeviceList)
    end.


    

make_buffer(0) ->  <<>>;
make_buffer(1) ->  <<1>>;
make_buffer(2) ->  <<1,2>>;
make_buffer(N) ->
    Bin = make_buffer(N div 2),
    if N band 1 =:= 1 ->
	    list_to_binary([1,Bin,Bin]);
       true ->
	    list_to_binary([Bin,Bin])
    end.
    



