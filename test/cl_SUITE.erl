%%% File    : cl_SUITE
%%% Author  : Dan Gudmundsson
%%% Description : test cl

-module(cl_SUITE).
-export([all/0, init_per_suite/1, end_per_suite/1]).

-include("cl.hrl").

all() ->
    [{cl_test, all},
     {cl_basic, ct_test},
     {cl_binary_test, ct_test},
     {cl_buffer, all},
     {cl_image, all}
    ].

init_per_suite(Config) ->
    try
	io:format("Running init per SUITE: ~p~n", [Config]),
	CLU = clu:setup(),
	{ok, [Type|_]} = cl:get_device_info(clu:device(CLU), type),
	clu:teardown(CLU),
	[{type, Type}|Config]
    catch _:Reason ->
	    io:format("Skipping test case failed to figure out cl device~n"),
	    io:format("~p: ~p~n",[Reason, erlang:get_stacktrace()]),
	    {skip, "Can not find cl type"}
    end.

end_per_suite(_) ->
    ok.


