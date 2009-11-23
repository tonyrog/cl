%%% File    : cl_sup.erl
%%% Description : 
%%% Created : 28 Aug 2006 by Tony Rogvall <tony@PBook.local>

%%% @hidden
%%% @author Tony Rogvall <tony@PBook.local>

-module(cl_sup).

-behaviour(supervisor).

%% External exports
-export([start_link/0, start_link/1]).

%% supervisor callbacks
-export([init/1]).

%%%----------------------------------------------------------------------
%%% API
%%%----------------------------------------------------------------------
start_link() ->
    start_link([]).
start_link(Args) ->
    supervisor:start_link({local, ?MODULE}, ?MODULE, Args).

%%%----------------------------------------------------------------------
%%% Callback functions from supervisor
%%%----------------------------------------------------------------------

%%----------------------------------------------------------------------
%%----------------------------------------------------------------------
init(Args) ->
    ClDrv = {cl_drv, {cl_drv, start_link, [Args]},
	     permanent, 5000, worker, [cl_drv]},
    {ok,{{one_for_all,0,300}, [ClDrv]}}.

