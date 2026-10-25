%%% @doc
%%%    Round-trip latency for an empty kernel: Erlang -> GPU -> Erlang.
%%%
%%%    run() / run(N) measures, per iteration (microseconds):
%%%      enq    - time spent in enqueue_nd_range_kernel NIF
%%%      total  - enqueue start .. Erlang process sees completion
%%%      q2s    - profiling queued->submit (host clock)
%%%      s2e    - profiling start->end (GPU clock)
%%%      rest   - total - enq - q2s - s2e: doorbell -> GPU start plus
%%%               GPU done -> process running again
%%%    NOTE: ROCclr stamps queued/submit with the host clock and start/end
%%%    with the GPU clock, so cross-domain differences are meaningless.
%%%    for three wait strategies:
%%%      wait   - cl:wait/1 (async context thread, clWaitForEvents)
%%%      finish - cl:finish/1 (async context thread, clFinish)
%%%      poll   - cl:flush/1 then spin on get_event_info(execution_status)
%%% @end

-module(cl_rtt).

-export([run/0, run/1, run/2, run/3]).

-include("../include/cl.hrl").

-define(SRC, "__kernel void noop(__global int* x) { }\n").

run() -> run(2000).

run(N) -> run(N, [wait, finish, poll]).

run(N, Modes) -> run(N, Modes, 0).

%% Gap: idle milliseconds between iterations (exposes GPU/CPU power ramp)
run(N, Modes, Gap) ->
    E = clu:setup(gpu),
    {ok, Program} = clu:build_source(E, ?SRC),
    {ok, Kernel} = cl:create_kernel(Program, "noop"),
    {ok, Buf} = cl:create_buffer(clu:context(E), [read_write], 64),
    ok = cl:set_kernel_arg(Kernel, 0, Buf),
    {ok, Q} = cl:create_queue(clu:context(E), clu:device(E),
			      [profiling_enable]),
    Res = [begin
	       _ = loop(Mode, Q, Kernel, Gap, 100, []),  %% warmup
	       Samples = loop(Mode, Q, Kernel, Gap, N, []),
	       report(Mode, Samples),
	       {Mode, Samples}
	   end || Mode <- Modes],
    cl:release_queue(Q),
    clu:teardown(E),
    Res.

loop(_Mode, _Q, _K, _Gap, 0, Acc) ->
    Acc;
loop(Mode, Q, K, Gap, I, Acc) ->
    Gap > 0 andalso timer:sleep(Gap),
    T0 = erlang:monotonic_time(),
    {ok, Ev} = cl:enqueue_nd_range_kernel(Q, K, [1], [1], []),
    T1 = erlang:monotonic_time(),
    complete(Mode, Q, Ev),
    T2 = erlang:monotonic_time(),
    {ok, Qd} = cl:get_event_profiling_info(Ev, command_queued),
    {ok, Sb} = cl:get_event_profiling_info(Ev, command_submit),
    {ok, St} = cl:get_event_profiling_info(Ev, command_start),
    {ok, En} = cl:get_event_profiling_info(Ev, command_end),
    Enq = us(T1 - T0),
    Total = us(T2 - T0),
    Q2S = (Sb - Qd) / 1000,
    S2E = (En - St) / 1000,
    S = #{enq => Enq, total => Total, q2s => Q2S, s2e => S2E,
	  rest => Total - Enq - Q2S - S2E},
    loop(Mode, Q, K, Gap, I - 1, [S | Acc]).

complete(wait, _Q, Ev) ->
    {ok, completed} = cl:wait(Ev);
complete(finish, Q, _Ev) ->
    ok = cl:finish(Q);
complete(poll, Q, Ev) ->
    ok = cl:flush(Q),
    poll(Ev).

poll(Ev) ->
    case cl:get_event_info(Ev, execution_status) of
	{ok, complete} -> ok;
	{ok, _} -> poll(Ev)
    end.

us(T) -> erlang:convert_time_unit(T, native, nanosecond) / 1000.

report(Mode, Samples) ->
    io:format("~-6s ~10s ~10s ~10s ~10s ~10s ~10s ~10s~n",
	      [Mode, "", "min", "p50", "p90", "p99", "max", "mean"]),
    lists:foreach(
      fun(Key) ->
	      Vs = lists:sort([maps:get(Key, S) || S <- Samples]),
	      L = length(Vs),
	      P = fun(F) -> lists:nth(max(1, min(L, round(F * L))), Vs) end,
	      io:format("~-6s ~10s ~10.1f ~10.1f ~10.1f ~10.1f ~10.1f ~10.1f~n",
			["", Key, hd(Vs), P(0.5), P(0.9), P(0.99),
			 lists:last(Vs), lists:sum(Vs) / L])
      end, [enq, q2s, s2e, rest, total]).
