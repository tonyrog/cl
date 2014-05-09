%%% @author Tony Rogvall <tony@rogvall.se>
%%% @copyright (C) 2014, Tony Rogvall
%%% @doc
%%%    Buffer test/example
%%% @end
%%% Created :  8 May 2014 by Tony Rogvall <tony@rogvall.se>

-module(cl_buffer).
-compile(export_all).

%% test write/copy/read

%% test of copy buffer, require version 1.0
copy() ->
    C = clu:setup(gpu),
    {ok,Q} = cl:create_queue(clu:context(C),clu:device(C),[]),
    {ok,Buf1} = cl:create_buffer(clu:context(C),[read_write], 1024),
    {ok,Buf2} = cl:create_buffer(clu:context(C),[read_write], 1024),
    Data1 = erlang:iolist_to_binary(lists:duplicate(4,lists:seq(0,255))),
    {ok,E1} = cl:enqueue_write_buffer(Q, Buf1, 0, 1024, Data1, []),
    {ok,E2} = cl:enqueue_copy_buffer(Q, Buf1, Buf2, 0, 0, 1024, [E1]),
    {ok,E3} = cl:enqueue_read_buffer(Q, Buf2, 0, 1024, [E2]),
    cl:flush(Q),
    cl:wait_for_events([E1,E2]),
    {ok,Data2} = cl:wait(E3),
    clu:teardown(C),
    Data1 =:= Data2.

%% read rectangluar area, require version 1.1
read_rect() ->
    C = clu:setup(gpu),
    true = lists:member({1,1},cl:versions()),
    {ok,Q} = cl:create_queue(clu:context(C),clu:device(C),[]),
    {ok,Buf1} = cl:create_buffer(clu:context(C),[read_write], 8*8),
    Data1 = <<0,0,0,0,0,0,0,0,
	      0,0,0,0,0,0,0,0,
	      0,0,1,2,3,4,0,0,
	      0,0,5,6,7,8,0,0,
	      0,0,0,0,0,0,0,0,
	      0,0,0,0,0,0,0,0,
	      0,0,0,0,0,0,0,0,
	      0, 0,0,0,0,0,0,0>>,
    {ok,E1} = cl:enqueue_write_buffer(Q, Buf1, 0, 64, Data1, []),
    {ok,E2} = cl:enqueue_read_buffer_rect(Q, Buf1,
					  [2,2,0],
					  [0,0,0],
					  [4,2,1],
					  8, 0,
					  4, 0,
					  [E1]),
    cl:wait_for_events([E1]),
    {ok,Data2} = cl:wait(E2),
    clu:teardown(C),
    Data2 =:= <<1,2,3,4,5,6,7,8>>.

%% write rectangluar area, require version 1.1
write_rect() ->
    C = clu:setup(gpu),
    true = lists:member({1,1},cl:versions()),
    {ok,Q} = cl:create_queue(clu:context(C),clu:device(C),[]),
    {ok,Buf1} = cl:create_buffer(clu:context(C),[read_write], 8*8),
    Data0 = <<9,9,9,9,9,9,9,9,
	      9,9,9,9,9,9,9,9,
	      9,9,9,9,9,9,9,9,
	      9,9,9,9,9,9,9,9,
	      9,9,9,9,9,9,9,9,
	      9,9,9,9,9,9,9,9,
	      9,9,9,9,9,9,9,9,
	      9,9,9,9,9,9,9,9>>,
    {ok,E1} = cl:enqueue_write_buffer(Q, Buf1, 0, 64, Data0, []),
    Data1 = <<1,2,3,4,
	      5,6,7,8>>,
    {ok,E2} = cl:enqueue_write_buffer_rect(Q, Buf1,
					   [2,2,0],
					   [0,0,0],
					   [4,2,1],
					   8, 0,
					   4, 0,
					   Data1,
					   [E1]),
    {ok,E3} = cl:enqueue_read_buffer(Q, Buf1, 0, 64, [E2]),
    cl:flush(Q),
    cl:wait_for_events([E1,E2]),
    {ok,Data3} = cl:wait(E3),
    clu:teardown(C),
    Data3 =:= <<9,9,9,9,9,9,9,9,
		9,9,9,9,9,9,9,9,
		9,9,1,2,3,4,9,9,
		9,9,5,6,7,8,9,9,
		9,9,9,9,9,9,9,9,
		9,9,9,9,9,9,9,9,
		9,9,9,9,9,9,9,9,
		9,9,9,9,9,9,9,9>>.

%% cerate sub buffer, require version 1.1
sub() ->
    C = clu:setup(gpu),
    true = lists:member({1,1},cl:versions()),
    {ok,Q} = cl:create_queue(clu:context(C),clu:device(C),[]),
    {ok,Buf1} = cl:create_buffer(clu:context(C),[read_write], 8*8),
    Data1 = <<0,0,0,0,0,0,0,0,
	      0,0,0,0,0,0,0,0,
	      0,0,1,2,3,4,0,0,
	      0,0,5,6,7,8,0,0,
	      0,0,0,0,0,0,0,0,
	      0,0,0,0,0,0,0,0,
	      0,0,0,0,0,0,0,0,
	      0, 0,0,0,0,0,0,0>>,
    {ok,E1} = cl:enqueue_write_buffer(Q, Buf1, 0, 64, Data1, []),
    {ok,Buf2} = cl:create_sub_buffer(Buf1,[read_write],region,[18,14]),
    {ok,E2} = cl:enqueue_read_buffer(Q, Buf2, 0, 12, [E1]),
    cl:flush(Q),
    cl:wait_for_events([E1]),
    {ok,Data2} = cl:wait(E2),
    clu:teardown(C),
    Data2 =:= <<1,2,3,4,0,0,0,0,5,6,7,8>>.



    
    
