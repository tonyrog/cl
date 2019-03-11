%%% @author Tony Rogvall <tony@rogvall.se>
%%% @copyright (C) 2019, Tony Rogvall
%%% @doc
%%%    Test cl nif calling overhead
%%% @end
%%% Created : 11 Mar 2019 by Tony Rogvall <tony@rogvall.se>

-module(cl_noop).

-compile(export_all).


test() ->
    T0 = erlang:monotonic_time(),
    loop_noop(1000000),
    T1 = erlang:monotonic_time(),
    Time1 = erlang:convert_time_unit(T1 - T0, native, microsecond),
    loop_noop_(1000000),
    T2 = erlang:monotonic_time(),
    Time2 = erlang:convert_time_unit(T2 - T1, native, microsecond),
    loop_dirty_noop(1000000),
    T3 = erlang:monotonic_time(),
    Time3 = erlang:convert_time_unit(T3 - T2, native, microsecond),
    {Time1/1000000, Time2/1000000, Time3/1000000}.

loop_noop(0) -> ok;
loop_noop(I) ->
    cl:noop(),
    loop_noop(I-1).

loop_noop_(0) -> ok;
loop_noop_(I) ->
    cl:noop_(),
    loop_noop_(I-1).

loop_dirty_noop(0) -> ok;
loop_dirty_noop(I) ->
    cl:dirty_noop(),
    loop_dirty_noop(I-1).

    
    
