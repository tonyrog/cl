%%% @author Tony Rogvall <tony@rogvall.se>
%%% @copyright (C) 2023, Tony Rogvall
%%% @doc
%%%    Simple test to display platform and device extensions
%%% @end
%%% Created : 10 Sep 2023 by Tony Rogvall <tony@rogvall.se>

-module(cl_info).

-export([start/0]).

start() ->
    {ok,IDs} = cl:get_platform_ids(),
    lists:foreach(
      fun(ID) ->
	      io:format("~p\n", [cl:get_platform_info(ID)])
      end, IDs),

    CLU = clu:setup(),
    Ds = clu:device_list(CLU),
    lists:foreach(
      fun(D) ->    
	      io:format("~p\n", [cl:get_device_info(D)])
      end, Ds).

    

