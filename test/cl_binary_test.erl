%%% File    : cl_binary_test.erl
%%% Author  : Tony Rogvall <tony@rogvall.se>
%%% Description : test build of binary programs
%%% Created :  7 Nov 2009 by Tony Rogvall <tony@rogvall.se>

-module(cl_binary_test).

-export([test/0, ct_test/1, init_per_suite/1, end_per_suite/1]).

-spec init_per_suite(Config0::list(tuple())) ->
                            (Config1::list(tuple())) | 
                            {skip,Reason::term()} | 
                            {skip_and_save,Reason::term(),
			     Config1::list(tuple())}.

init_per_suite(Config) -> cl_SUITE:init_per_suite(Config).

-spec end_per_suite(Config::list(tuple())) -> ok.

end_per_suite(_Config) ->
    ok.


ct_test(_) ->
    test().

test() ->
    E = clu:setup(),
    {ok,P1} = clu:build_source(E, "__kernel void foo(int n) { int x; x = n; }"),
    {ok,B} = clu:get_program_binaries(P1),
    ok = cl:release_program(P1),
    {ok,P2} = clu:build_binary(E, B),
    ok = cl:release_program(P2),
    ok.

    
    

