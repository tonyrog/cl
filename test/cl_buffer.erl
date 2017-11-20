%%% @author Tony Rogvall <tony@rogvall.se>
%%% @copyright (C) 2014, Tony Rogvall
%%% @doc
%%%    Buffer test/example
%%% @end
%%% Created :  8 May 2014 by Tony Rogvall <tony@rogvall.se>

-module(cl_buffer).

-export([init_per_suite/1, end_per_suite/1]).
-export([all/0,
	 copy/1,
	 read_rect/1,
	 write_rect/1,
	 sub/1,
	 fill/1,
	 migrate/1]).

-spec init_per_suite(Config0::list(tuple())) ->
                            (Config1::list(tuple())) | 
                            {skip,Reason::term()} | 
                            {skip_and_save,Reason::term(),
			     Config1::list(tuple())}.

init_per_suite(Config) -> cl_SUITE:init_per_suite(Config).

-spec end_per_suite(Config::list(tuple())) -> ok.

end_per_suite(_Config) ->
    ok.


all() ->
    [copy, read_rect, write_rect, sub, fill, migrate].

%% test write/copy/read

%% test of copy buffer, require version 1.0
copy(Config) ->
    C = clu:setup(proplists:get_value(type, Config, gpu)),
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
read_rect(Config) ->
    C = clu:setup(proplists:get_value(type, Config, gpu)),
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
write_rect(Config) ->
    C = clu:setup(proplists:get_value(type, Config, gpu)),
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
sub(Config) ->
    C = clu:setup(proplists:get_value(type, Config, gpu)),
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


%% fill buffer, require version 1.2
fill(Config) ->
    C = clu:setup(proplists:get_value(type, Config, gpu)),
    true = lists:member({1,2},cl:versions()),
    {ok,Q} = cl:create_queue(clu:context(C),clu:device(C),[]),
    {ok,Buf1} = cl:create_buffer(clu:context(C),[read_write], 8*8),
    {ok,E1} = cl:enqueue_fill_buffer(Q, Buf1, <<9>>, 0, 64, []),
    {ok,E2} = cl:enqueue_fill_buffer(Q, Buf1, <<1,2,3,4>>, 12, 4, [E1]),
    {ok,E3} = cl:enqueue_fill_buffer(Q, Buf1, <<5,6,7,8>>, 20, 4, [E2]),
    {ok,E4} = cl:enqueue_read_buffer(Q, Buf1, 0, 64, [E3]),
    cl:flush(Q),
    cl:wait_for_events([E1,E2,E3]),
    {ok,Data1} = cl:wait(E4),
    clu:teardown(C),
    Data1 =:= <<9,9,9,9,9,9,9,9,
		9,9,9,9,1,2,3,4,
		9,9,9,9,5,6,7,8,
		9,9,9,9,9,9,9,9,
		9,9,9,9,9,9,9,9,
		9,9,9,9,9,9,9,9,
		9,9,9,9,9,9,9,9,
		9,9,9,9,9,9,9,9>>.

migrate(_) ->
    C = clu:setup(all),
    true = lists:member({1,2},cl:versions()),
    case clu:device_list(C) of
	[D1,D2|_] ->
	    {ok,Q1} = cl:create_queue(clu:context(C),D1,[]),
	    {ok,Q2} = cl:create_queue(clu:context(C),D2,[]),
	    {ok,B1} = cl:create_buffer(clu:context(C),[read_write], 8*8),
	    {ok,E1} = cl:enqueue_fill_buffer(Q1, B1, <<9>>, 0, 64, []),
	    cl:flush(Q1),
	    {ok,completed} = cl:wait(E1),
	    {ok,E2} = cl:enqueue_migrate_mem_objects(Q2, [B1], [], []),
	    cl:flush(Q2),
	    %% fixme: add a kernel to check that the data was migrated
	    cl:wait(E2);
	_ -> ignore
    end.
