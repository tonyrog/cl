%%% File    : cl_app.erl
%%% Description : 
%%% Created : 26 Oct 2009 by Tony Rogvall <tony@PBook.local>

%%% @hidden
%%% @author Tony Rogvall <tony@rogvall.se>

-module(cl_app).

-behaviour(application).
-export([start/2,stop/1]).

%% start
start(_Type, _StartArgs) ->
    {ok,Args} = application:get_env(arguments),
    cl_sup:start_link(Args).

%% stop FIXME
stop(_State) ->
  ok.
