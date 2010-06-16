%%%-------------------------------------------------------------------
%%% File    : cl_drv.erl
%%% Description : Erlang OpenCL interface
%%% Created : 25 Oct 2009 by Tony Rogvall <tony@rogvall.se>
%%%-------------------------------------------------------------------

%%% @hidden
%%% @author Tony Rogvall <tony@rogvall.se>


-module(cl_drv).

-behaviour(gen_server).

%% API
-export([start_link/0, start/0, stop/0]).
-export([start_link/1, start/1]).

-export([create/3, async_create/3, release/2, retain/2, call/2, async_call/2]).
-export([encode/1, decode/1]).
%% gen_server callbacks
-export([init/1, handle_call/3, handle_cast/2, handle_info/2,
	 terminate/2, code_change/3]).

-import(lists, [foreach/2, seq/2]).

-include("../include/cl.hrl").
-include("cl_int.hrl").


-ifdef(debug).
-define(dbg(F,A), io:format((F)++"\n",(A))).
-define(dbg_hard(F,A), ok).
-define(drv_path, "debug").
-else.
-define(dbg(F,A), ok).
-define(dbg_hard(F,A), ok).
-define(drv_path, "release").
-endif.

-define(ASYNC_TIMEOUT, 10000).  %% 10s - compilation (may be more?)

-record(state, 
	{
	  port,  %% port to cl_drv
	  tab    %% CL_REG storage
	 }).

%% ECL_REG usage
%%  Handle  => ReleaseCode x ProcessesCounter
%%  Monitor => {Owner,Handler}
%%  {Owner,Handler} => MonitorxRetainCounter
%%
%%====================================================================
%% API
%%====================================================================
start_link() -> start_link([]).
start_link(Args) ->
    gen_server:start_link({local, ?CL_SERVER}, ?MODULE, Args, []).

start() -> 
    start([]).
start(Args) ->
    gen_server:start({local, ?CL_SERVER}, ?MODULE, Args, []).    

stop() ->
    gen_server:call(?CL_SERVER, stop).

create(Code, CodeDestroy, Args) ->
    case call(Code, Args) of
	{ok,Handle} when is_integer(Handle) ->  %% single new object
	    gen_server:cast(?CL_SERVER, {create,self(),Handle,CodeDestroy}),
	    {ok, Handle};
	{ok,Handles} when is_list(Handles) ->  %% list of new objects
	    foreach(fun(Handle) ->
			    gen_server:cast(?CL_SERVER, {create,self(),Handle,CodeDestroy})
		    end, Handles),
	    {ok,Handles};
	Error -> Error
    end.

async_create(Code, CodeDestroy, Args) ->
    case async_call(Code, Args) of
	{ok,Handle} when is_integer(Handle) ->  %% single new object
	    gen_server:cast(?CL_SERVER, {create,self(),Handle,CodeDestroy}),
	    {ok, Handle};
	{ok,Handles} when is_list(Handles) ->  %% list of new objects
	    foreach(fun(Handle) ->
			    gen_server:cast(?CL_SERVER, {create,self(),Handle,CodeDestroy})
		    end, Handles),
	    {ok,Handles};
	Error -> Error
    end.

release(Code, Handle) ->
    case call(Code, <<?cl_pointer(Handle)>>) of
	ok ->
	    gen_server:cast(?CL_SERVER, {release,self(),Handle}),
	    ok;
	Error ->
	    Error
    end.

retain(Code, Handle) ->
    case call(Code, <<?cl_pointer(Handle)>>) of
	ok ->
	    gen_server:cast(?CL_SERVER, {retain,self(),Handle}),
	    ok;
	Error ->
	    Error
    end.


call(Code, Args) ->
    Reply = erlang:port_control(?CL_PORT, Code, Args),
    case decode(Reply) of
	{event,Ref} ->
	    wait_reply(Ref);
	Result ->
	    Result
    end.


%% async call to command(v) interface
async_call(Code, Args) ->
    CmdRef = random:uniform(16#ffffffff),
    Header = <<?u_int8_t(Code),?u_int32_t(CmdRef)>>,
    erlang:port_command(?CL_PORT,[Header,Args]),
    wait_reply(CmdRef).

wait_reply(CmdRef) ->
    receive
	{cl_reply,CmdRef,Reply} when is_binary(Reply) ->
	    %% This is a reply from a port command
	    %% (passing binaries with zero copy)
	    decode(Reply);
	{cl_reply,CmdRef,Reply} ->
	    %% This is a reply from async calls (running in driver thread)
	    Reply
    after ?ASYNC_TIMEOUT ->
	    {error, no_reply}
    end.

%%====================================================================
%% gen_server callbacks
%%====================================================================

%%--------------------------------------------------------------------
%% Function: init(Args) -> {ok, State} |
%%                         {ok, State, Timeout} |
%%                         ignore               |
%%                         {stop, Reason}
%% Description: Initiates the server
%%--------------------------------------------------------------------
init(Args) ->
    DPath = case proplists:get_bool(debug, Args) of
		false -> ?drv_path;
		true -> "debug"
	    end,
    W = case erlang:system_info(wordsize) of
	    4 -> "32";
	    8 -> "64"
	end,
    Driver = "cl_drv",
    LibDir = code:lib_dir(cl),
    Path = filename:join([LibDir,"lib",DPath,W]),
    ?dbg("Load driver '~s' from: '~s'\n", [Driver, Path]),
    WPath = filename:join([LibDir,"ebin",W]),
    code:add_path(WPath),
    case erl_ddll:load_driver(Path, Driver) of
	ok ->
	    Port = erlang:open_port({spawn, Driver}, [binary]),
	    register(?CL_PORT, Port),
	    Tab  = ets:new(?CL_REG, [named_table, public, set]),
	    {ok, #state{ port = Port, tab  = Tab }};
	{error,Error} ->
	    error_logger:format("CL Error: ~s\n", [erl_ddll:format_error_int(Error)]),
	    {stop, Error}
    end.

%%--------------------------------------------------------------------
%% Function: %% handle_call(Request, From, State) -> {reply, Reply, State} |
%%                                      {reply, Reply, State, Timeout} |
%%                                      {noreply, State} |
%%                                      {noreply, State, Timeout} |
%%                                      {stop, Reason, Reply, State} |
%%                                      {stop, Reason, State}
%% Description: Handling call messages
%%--------------------------------------------------------------------
handle_call(stop, _From, State) ->
    {stop, normal, ok, State};
handle_call(_Request, _From, State) ->
    Reply = ok,
    {reply, Reply, State}.

%%--------------------------------------------------------------------
%% Function: handle_cast(Msg, State) -> {noreply, State} |
%%                                      {noreply, State, Timeout} |
%%                                      {stop, Reason, State}
%% Description: Handling cast messages
%%--------------------------------------------------------------------
handle_cast({create,Pid,Handle,ReleaseCode}, State) ->
    Mon = start_monitor(Pid),
    ets:insert_new(?CL_REG, {Handle,ReleaseCode,1}),
    ets:insert_new(?CL_REG, {Mon, {Pid,Handle}}),
    ets:insert_new(?CL_REG, {{Pid,Handle}, Mon, 1}),
    {noreply, State};
handle_cast({retain,Pid,Handle}, State) ->
    try ets:update_counter(?CL_REG,{Pid,Handle},{3,1}) of
	_N ->
	    {noreply,State}
    catch
	error:_ ->
	    Mon = start_monitor(Pid),
	    ets:update_counter(?CL_REG,Handle,{3,1}),
	    ets:insert_new(?CL_REG, {Mon, {Pid,Handle}}),
	    ets:insert_new(?CL_REG, {{Pid,Handle}, Mon, 1}),
	    {noreply,State}
    end;
handle_cast({release,Pid,Handle}, State) ->
    try ets:update_counter(?CL_REG,{Pid,Handle},{3,-1}) of
	0 ->
	    [{_,Mon,_}] = ets:lookup(?CL_REG, {Pid,Handle}),
	    stop_monitor(Mon),
	    ets:delete(?CL_REG, {Pid,Handle}),
	    ets:delete(?CL_REG, Mon),
	    case ets:update_counter(?CL_REG,Handle,{3,-1}) of
		0 -> ets:delete(?CL_REG, Handle);
		_ -> ok
	    end,
	    {noreply,State};
	_N ->
	    {noreply,State}
    catch
	error:_ ->
	    {noreply,State}
    end;
handle_cast(_Msg, State) ->
    {noreply, State}.

%%--------------------------------------------------------------------
%% Function: handle_info(Info, State) -> {noreply, State} |
%%                                       {noreply, State, Timeout} |
%%                                       {stop, Reason, State}
%% Description: Handling all non call/cast messages
%%--------------------------------------------------------------------
handle_info({Port,{data,Data}}, State) when Port == State#state.port ->
    ?dbg_hard("Port: data=~p", [Data]),
    case Data of
	<<?OK, ?u_int32_t(CmdId), ReplyData/binary>> ->
	    ?dbg("Got: OK cmdid=~w, data=~p", [CmdId,ReplyData]),
	    State1 = case decode(ReplyData) of
			 false ->
			     cl_reply(CmdId, ok, State);
			 {value,Decoded} ->
			     cl_reply(CmdId, {ok,Decoded}, State)
		     end,
	    {noreply, State1};
	<<?ERROR, ?u_int32_t(CmdId), ReplyData/binary>> ->
	    ?dbg("Got: ERROR cmdid=~w, data=~p", [CmdId,ReplyData]),
	    State1 = case decode(ReplyData) of
			 false ->
			     cl_reply(CmdId, error, State);
			 {value,Decoded} ->
			     cl_reply(CmdId, {error,Decoded}, State)
		     end,			 
	    {noreply, State1};
	<<?EVENT, ?u_int32_t(EventId), EventData/binary>> ->
	    ?dbg("Got: EVENT evtid=~w, data=~p", [EventId,EventData]),
	    case ets:lookup(?CL_REG, EventId) of
		[{_,Pid}|_] when is_pid(Pid) ->
		    cl_event(Pid, EventData, State);
		[_, {_,Pid}|_] when is_pid(Pid) ->
		    cl_event(Pid, EventData, State);
		_ ->
		    ?dbg("no receipient for evtid=~p data=~p", 
			 [EventId, decode(EventData)]),
		    {noreply, State}
	    end;
	_ ->
	    ?dbg("got bad info data ~p\n", [Data]),
	    {noreply, State}
    end;
handle_info({Port,eof}, State) when Port == State#state.port ->
    ?dbg("cl_drv closed",[]),
    erlang:port_close(Port),
    {stop, closed, State};
handle_info({cl_error,_Ref,Error}, State) ->
    %% This is the context error notification message
    io:format("CL Error: ~s\n", [Error]),
    {noreply, State};
handle_info({'DOWN',Mon,process,Pid,_Reason}, State) ->
    case ets:lookup(?CL_REG, Mon) of
	[{_,PH={Pid,Handle}}] ->
	    [{_, ReleaseCode,HCount}] = ets:lookup(?CL_REG, Handle),
	    [{_, Mon,Count}]  = ets:lookup(?CL_REG, PH),
	    foreach(fun(_I) ->
			    call(ReleaseCode,<<?cl_pointer(Handle)>>),
			    ?dbg_hard("Handle ~w released\n", [Handle])
		    end, seq(1, Count)),
	    ets:delete(?CL_REG, Mon),
	    ets:delete(?CL_REG, PH),
	    if HCount == 1 ->
		    ets:delete(?CL_REG, Handle);
	       true ->
		    ets:update_counter(?CL_REG,Handle,{3,-1})
	    end;
	[] ->
	    ok
    end,
    {noreply, State};
handle_info(_Info, State) ->
    {noreply, State}.


%%--------------------------------------------------------------------
%% Function: terminate(Reason, State) -> void()
%% Description: This function is called by a gen_server when it is about to
%% terminate. It should be the opposite of Module:init/1 and do any necessary
%% cleaning up. When it returns, the gen_server terminates with Reason.
%% The return value is ignored.
%%--------------------------------------------------------------------
terminate(_Reason, _State) ->
    ok.

%%--------------------------------------------------------------------
%% Func: code_change(OldVsn, State, Extra) -> {ok, NewState}
%% Description: Convert process state when code is changed
%%--------------------------------------------------------------------
code_change(_OldVsn, State, _Extra) ->
    {ok, State}.

%%--------------------------------------------------------------------
%%% Internal functions
%%--------------------------------------------------------------------

cl_event(Pid, EventData, State) ->	    
    case decode(EventData) of
	false ->
	    ?dbg("bad event data ~p", [EventData]),
	    {noreply, State};
	{value, _E={eevent,Win,destroyed}} -> 
	    %% cleanup window structure
	    ?dbg("event ~p", [_E]),
	    case ets:lookup(?CL_REG, Win) of
		[{_,Mon}|_] when is_reference(Mon) ->
		    ets:delete(?CL_REG, Win),
		    ets:delete(?CL_REG, Mon),
		    stop_monitor(Mon),
		    {noreply,State};
		[_,{_,Mon}|_] when is_reference(Mon) ->
		    ets:delete(?CL_REG, Win),
		    ets:delete(?CL_REG, Mon),
		    stop_monitor(Mon),
		    {noreply,State};
		[] ->
		    {noreply,State}
	    end;
	{value, E} ->
	    ?dbg("event ~p", [E]),
	    Pid ! E,
	    {noreply, State}
    end.

cl_reply(_CmdID, _Reply, State) ->
    ?dbg("got reply: ~w  ~p", [_CmdID, _Reply]),
    {noreply, State}.

start_monitor(Pid) ->
    erlang:monitor(process, Pid).

%% stop and flush
stop_monitor(undefined) -> 
    ok;
stop_monitor(Ref) ->
    erlang:demonitor(Ref),
    receive
	{'DOWN',Ref,_,_Pid,_Reason} ->
	    ok
    after 0 ->
	    ok
    end.

%% Encode/Decode reply data from driver
%% encode into Data format
encode(Term) ->
    list_to_binary([enc(Term)]).

enc(true) ->   
    <<?BOOLEAN, 1>>;
enc(false) ->  
    <<?BOOLEAN, 0>>;
enc(X) when is_atom(X) ->
    Val = atom_to_list(X),
    [<<?ATOM, (length(Val)):8>>, Val];
enc(X) when is_integer(X) ->
    if X >= 0, X =< 16#ffffffff -> <<?UINT32, ?u_int32_t(X)>> ;
       X < 0, X >= -16#8000000  -> <<?INT32, ?int32_t(X)>> ;
       X > 0 -> <<?UINT64, ?u_int64_t(X)>>;
       true -> (<<?INT64, ?int64_t(X)>>)
    end;
enc(X) when is_float(X) -> 
    <<?FLOAT64, ?double_t(X)>>;
enc(X) when is_list(X) ->
    case is_string(X) of
        true ->
            Len = length(X),
            if Len =< 255 ->
                    [<<?STRING1, ?u_int8_t(Len)>>, X];
               true ->
                    [<<?STRING4, ?u_int32_t(Len)>>, X]
            end;
        false ->
            [?LIST, lists:map(fun(E) -> enc(E) end, X), ?LIST_END]
    end;
enc(X) when is_tuple(X) ->
    [?TUPLE, lists:map(fun(E) -> enc(E) end, tuple_to_list(X)), ?TUPLE_END];
enc(X) when is_binary(X) ->
    Sz = byte_size(X),
    [<<?BINARY,?u_int32_t(Sz)>>, X].
     
is_string([X|Xs]) when X >= 0, X =< 255 -> is_string(Xs);
is_string([]) -> true;
is_string(_) -> false.


%% decode reply data
decode(Data = <<131,_/binary>>) ->
    binary_to_term(Data);
decode(Debug) -> 
    ?dbg_hard("decode: data = ~p\n", [Debug]),
    decode(Debug, []).
    
decode(<<>>, [Hd]) -> 
    ?dbg_hard("deocde = ~p\n", [Hd]),
    Hd;
decode(Data, Stack) ->
    case Data of
	<<?OK, Rest/binary>> -> 
            ?dbg_hard("OK",[]),
            decode(Rest, [ok|Stack]);
	<<?ERROR, Rest/binary>> -> 
            ?dbg_hard("ERROR",[]),
            decode(Rest, [error|Stack]);
	<<?EVENT, Rest/binary>> -> 
            ?dbg_hard("EVENT",[]),
            decode(Rest, [event|Stack]);
        <<?LIST, Rest/binary>> -> 
            ?dbg_hard("LIST",[]),
            decode(Rest, [list|Stack]);
        <<?TUPLE, Rest/binary>> ->
            ?dbg_hard("TUPLE",[]),
            decode(Rest, [tuple|Stack]);
        <<?BOOLEAN, B, Rest/binary>> -> 
            ?dbg_hard("BOOLEAN:~w",[B]),
            decode(Rest, [B =/= 0 | Stack]);
        <<?UINT8, ?u_int8_t(I), Rest/binary>> -> 
            ?dbg_hard("UINT8:~w",[I]),
            decode(Rest, [I|Stack]);
        <<?UINT16, ?u_int16_t(I), Rest/binary>> ->
            ?dbg_hard("UINT16:~w",[I]),
            decode(Rest, [I|Stack]);
        <<?UINT32, ?u_int32_t(I), Rest/binary>> ->
            ?dbg_hard("UINT32:~w",[I]),
            decode(Rest, [I|Stack]);
        <<?UINT64, ?u_int64_t(I), Rest/binary>> ->
            ?dbg_hard("UINT64:~w",[I]),
            decode(Rest, [I|Stack]);
        <<?INT8, ?int8_t(I), Rest/binary>> ->
            ?dbg_hard("INT8:~w",[I]),
            decode(Rest, [I|Stack]);
        <<?INT16, ?int16_t(I), Rest/binary>> ->
            ?dbg_hard("INT16:~w",[I]),
            decode(Rest, [I|Stack]);
        <<?INT32, ?int32_t(I), Rest/binary>> -> 
            ?dbg_hard("INT32:~w",[I]),
            decode(Rest, [I|Stack]);
        <<?INT64, ?int64_t(I), Rest/binary>> ->
            ?dbg_hard("INT64:~w",[I]),
            decode(Rest, [I|Stack]);
        <<?FLOAT32, ?float_t(F), Rest/binary>> ->
            ?dbg_hard("FLOAT32:~w",[F]),
            decode(Rest, [F|Stack]);
        <<?FLOAT64, ?double_t(F), Rest/binary>> -> 
            ?dbg_hard("FLOAT64:~w",[F]),
            decode(Rest, [F|Stack]);
        <<?STRING1, ?u_int8_t(N), String:N/binary, Rest/binary>> -> 
            ?dbg_hard("STRING1: len=~w, ~w",[N,String]),
            decode(Rest, [binary_to_list(String)  | Stack]);
        <<?STRING4, ?u_int32_t(N), String:N/binary, Rest/binary>> ->
            ?dbg_hard("STRING4: len=~w, ~w",[N,String]),
            decode(Rest, [binary_to_list(String)  | Stack]);
        <<?BINARY, ?u_int32_t(N), Bin:N/binary, Rest/binary>> -> 
            ?dbg_hard("BINARY: len=~w, ~w",[N,Bin]),
            decode(Rest, [Bin | Stack]);
	<<?ATOM, ?u_int8_t(N), Atom:N/binary, Rest/binary>> -> 
            ?dbg_hard("ATOM: len=~w, ~w",[N,Atom]),
            decode(Rest, [list_to_atom(binary_to_list(Atom)) | Stack]);
        <<?LIST_END, Rest/binary>> ->
            ?dbg_hard("LIST_END",[]),
            {L,[_|Stack1]} = lists:splitwith(fun(X) -> X =/= list end, Stack),
            decode(Rest, [lists:reverse(L) | Stack1]);
        <<?TUPLE_END, Rest/binary>> ->
            ?dbg_hard("TUPLE_END",[]),
            {L,[_|Stack1]}=lists:splitwith(fun(X) -> X =/= tuple end, Stack),
            decode(Rest, [list_to_tuple(lists:reverse(L)) | Stack1])
    end.
