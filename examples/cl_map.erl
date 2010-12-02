-module(cl_map).

-include_lib("cl/include/cl.hrl").

-compile(export_all).
-import(lists, [map/2, foreach/2, foldl/3]).

-record(kwork,
	{
	  queue,   %% the queue
	  local,   %% kernel work_group_size
	  freq,    %% device max_clock_frequenct
	  units,   %% device max_compute_units
	  weight,  %% weight [0..1]
	  e1,e2,e3, %% events (fixme)
	  imem,    %% input memory object
	  omem,    %% output memory object
	  isize,   %% item size
	  idata    %% input data
	 }).

test() ->
    Args = << <<X:32/native-float>> || X <- lists:seq(1, 1024) >>,
    ResultList = run("fun(<<X/cl_float>>) -> X*X+1 end", Args),
    lists:flatmap(
      fun(Result) ->
	      [ X || <<X:32/native-float>> <= Result ]
      end, ResultList).
	
%% 
%% Run a map operation over data
%% Restrictions: the output must currently equal the size of
%%
%% 
run(Function, Data) ->
    E = clu:setup(all),  %% gpu needs more work
    {_NArgs,ItemSize,Source} = p_program(Function),
    io:format("Program:\n~s\n", [Source]),
    {ok,Program} = clu:build_source(E, Source),
    {ok,Kernel} = cl:create_kernel(Program, "example"),

    Kws =
	map(
	  fun(Device) ->
		  {ok,Queue} = cl:create_queue(E#cl.context,Device,[]),
		  {ok,Local} = cl:get_kernel_workgroup_info(Kernel,Device,
							    work_group_size),
		  {ok,Freq} = cl:get_device_info(Device,max_clock_frequency),
		  {ok,K} = cl:get_device_info(Device, max_compute_units),
		  #kwork{ queue=Queue, local=Local, freq=Freq, units=K,
			  isize=ItemSize }
	  end, E#cl.devices),
    io:format("Kws = ~p\n", [Kws]),

    %% Sum the weights and scale to [0..1]
    Tw = foldl(fun(K,Sum) -> Sum + K#kwork.freq*K#kwork.units end,
	       0, Kws),
    Kws1 = map(fun(K) ->
		       K#kwork { weight = (K#kwork.freq*K#kwork.units)/Tw }
	       end, Kws),
    io:format("Kws1 = ~p\n", [Kws1]),
    
    %% Split data according to Weights but start with data
    %% That have hard requirements on work_group_size
    Kws11 = lists:reverse(lists:keysort(#kwork.local,Kws1)),
    Kws2 = kwork_set_data(Kws11,  Data),
    io:format("Kws2 = ~p\n", [Kws2]),

    %% Create memory objects
    Kws3 = map(
	     fun(K) ->
		     Nk = byte_size(K#kwork.idata),
		     {ok,I}  = cl:create_buffer(E#cl.context,[read_only],Nk),
		     {ok,O} = cl:create_buffer(E#cl.context,[write_only],Nk),
		     K#kwork { imem=I, omem=O }
	     end, Kws2),
    io:format("Kws3 = ~p\n", [Kws3]),
    
    %% Enque input data
    Kws4 = map(
	fun(K) ->
		Nk = byte_size(K#kwork.idata),
		Count = Nk div K#kwork.isize,
		{ok,E1} = cl:enqueue_write_buffer(K#kwork.queue,
						  K#kwork.imem, 
						  0, Nk, 
						  K#kwork.idata, []),
		%% Set kernel arguments
		ok = cl:set_kernel_arg(Kernel, 0, K#kwork.imem),
		ok = cl:set_kernel_arg(Kernel, 1, K#kwork.omem),
		ok = cl:set_kernel_arg(Kernel, 2, Count),
	      
		%% Enqueue the kernel
		Global = Count,
		io:format("Global=~w, Local=~w\n", [Global,K#kwork.local]),
		{ok,E2} = cl:enqueue_nd_range_kernel(K#kwork.queue,
						     Kernel,
						     [Global], [K#kwork.local],
						     [E1]),
		%% Enqueue the read from device memory (wait for kernel to finish)
		{ok,E3} = cl:enqueue_read_buffer(K#kwork.queue,
						 K#kwork.omem,0,Nk,[E2]),
		%% Now flush the queue to make things happend 
		ok = cl:flush(K#kwork.queue),
		%% FIXME: here we should release E1,E2
		K#kwork { e1=E1,e2=E2,e3=E3 }
	end, Kws3),
    io:format("Kws4 = ~p\n", [Kws4]),

    %% Wait for Result buffer to be written
    Bs = map(
	   fun(K) ->
		   io:format("E1 = ~p\n", [cl:wait(K#kwork.e1)]),
		   io:format("E2 = ~p\n", [cl:wait(K#kwork.e2)]),
		   {ok,Bin} = cl:wait(K#kwork.e3),
		   cl:release_mem_object(K#kwork.imem),
		   cl:release_mem_object(K#kwork.omem),
		   cl:release_queue(K#kwork.queue),
		   %% Release built into cl:wait!
		   %% cl:release_event(K#kwork.e1),
		   %% cl:release_event(K#kwork.e2),
		   %% cl:release_event(K#kwork.e3),
		   Bin
	   end, Kws4),
    

    cl:release_kernel(Kernel),
    cl:release_program(Program),
    clu:teardown(E),
    Bs.
%%
%% Assume at least one kwork
%% Data must be a multiple of local (work_group_size)
%% FIXME: This must be reworked to handle all cases
%%
kwork_set_data([K], Data) ->
    [K#kwork { idata = Data }];
kwork_set_data([K|Ks], Data) ->
    N = byte_size(Data) div K#kwork.isize,
    M = trunc(K#kwork.weight * N),  %% make a multiple of local
    L = K#kwork.local,
    R = ((L - (M rem L)) rem L),
    ML = M + R,
    io:format("N=~w, M=~w, L=~w, R=~w, ML=~w\n", [N,M,L,R,ML]),
    if ML =< N ->
	    Md = ML*K#kwork.isize,
	    <<Data1:Md/binary, Data2/binary>> = Data,
	    [K#kwork { idata = Data1 } | kwork_set_data(Ks, Data2)];
       true ->
	    Rd = R*K#kwork.isize,
	    [K#kwork { idata = <<Data/binary, 0:Rd/unit:8>> } | Ks]
    end.
    
%%
%% Function:
%%     fun(<<X:32/T>>,P1,..,Pn) -> 
%%         F(X,P1,...Pn)
%%
%% Translates to
%%     __kernel main(__global T0* input, __global T0* output,
%%                   const unsigned int item_count,
%%                   T1 p1, T2 p2 .. Tn Pn)
%%     {
%%         int i = get_global_id(0);
%%         if (i < item_count) {
%%             output[i] = F(input[i],p1,..Pn)
%%         }
%%     }
%%
%%
%%
p_program(Function) ->
    case erl_scan:string(Function) of
	{ok,Ts,_Ln} ->
	    case erl_parse:parse_exprs(add_dot(Ts)) of
		{ok, Exprs} ->
		    p_fun(Exprs);
		Error ->
		    Error
	    end;
	Error ->
	    Error
    end.

add_dot(Ts) ->
    case lists:last(Ts) of
	{dot,_} -> Ts;
	E -> 
	    Ts ++ [{dot,element(2,E)}]
    end.
	    

p_fun([{'fun',_Ln1,{clauses,[{clause,_Ln3,H,[],B}]}}]) ->
    As = p_header(H),
    NArgs = length(As),
    {_MainVar,MainType} = hd(As),
    ItemSize = sizeof(MainType),
    {NArgs,ItemSize,
     lists:flatten([g_header(As), g_body(As,B)])};
p_fun(Fs) ->
    io:format("Fs=~p\n", [Fs]),
    erlang:error(not_supported).

p_header(Params) ->
    map(fun p_arg/1, Params).

g_header([{V,T}|Ps]) ->
    ["__kernel void example(",
     "__global ", g_type(T), "*", "in", ",",
     "__global ", g_type(T), "*", "out",",",
     "const uint n",
     map(fun({X,Tx}) ->
		 [",", "const ", g_type(Tx), " ",
		  atom_to_list(X)]
	 end, Ps),
     ")\n",
     "{",
     "  int i = get_global_id(0);\n",
     "  if (i < n) {\n"
     "  ", g_type(T), " ", atom_to_list(V), "= in[i];\n"
    ].

g_body(Vs,[E]) ->
    ["out[i] = ", p_expr(Vs, E),";\n",
     "  }\n",
     "}\n"];
g_body(Vs,[E|Es]) ->
    [p_expr(Vs,E),";\n",
     g_body(Vs, Es)];
g_body(_Vs,[]) ->
    ["  }\n",
     "}\n"].

p_arg({bin,_,[{bin_element,_,{var,_,V},Size,[Type]}]}) ->
    S = t_vector_size(Size),
    T = t_type(S,Type),
    {V,T}.

p_expr(Vs, {var,_,V}) ->
    true = lists:keymember(V, 1, Vs),
    [atom_to_list(V)];
p_expr(_Vs, {integer,_,I}) ->
    [integer_to_list(I)];
p_expr(_Vs, {float,_,F}) ->
    io_lib:format("~f", [F]);
p_expr(Vs, {op,_Ln,Op,L,R}) ->
    [p_expr(Vs,L),atom_to_list(Op),p_expr(Vs,R)];
p_expr(Vs, {op,_Ln,Op,M}) ->
    [atom_to_list(Op),p_expr(Vs,M)];
p_expr(Vs, {match,_Ln,L,R}) ->
    [p_expr(Vs,L),"=",p_expr(Vs,R)];
p_expr(Vs, {record_field,_Ln,{var,_,V},{atom,_,Selector}}) ->
    true = lists:keymember(V, 1, Vs),
    [atom_to_list(V),".",atom_to_list(Selector)];
p_expr(Vs, {record_field,_Ln,Expr,{atom,_,Selector}}) ->
    E = p_expr(Vs, Expr),
    %% fixme: normalize vector selector and check that
    %% the permutation is valid.
    [E,".",atom_to_list(Selector)];
p_expr(Vs, {call,_Ln,{atom,_,F},As}) ->
    Ps = map(fun(A) -> p_expr(Vs, A) end, As),
    [atom_to_list(F),"(", g_args(Ps), ")"].


t_vector_size(default) ->
    default;
t_vector_size({integer,_,Sz}) ->
    Sz.

g_args([]) -> [];
g_args([A]) ->  [A];
g_args([A|As]) ->  [A,"," | g_args(As)].

g_type({T,S}) when is_atom(T), is_integer(S) ->
    [atom_to_list(T),integer_to_list(T)];
g_type(T) when is_atom(T) ->
    [atom_to_list(T)].

%% size scalar type
sizeof('char') -> 1;
sizeof('uchar') -> 1;
sizeof('short') -> 2;
sizeof('ushort') -> 2;
sizeof('int') -> 4;
sizeof('uint') -> 4;
sizeof('long') -> 8;
sizeof('ulong') -> 8;
sizeof('float') -> 4;
sizeof('half') -> 2;
sizeof({T,default}) -> sizeof(T);
sizeof({T,S}) -> S*sizeof(T).

%% scalar types (api -> opencl)
t_type(Size,Type) ->
    Scalar = t_type(Type),
    if Size == default -> Scalar;
       Size == 1 -> Scalar;
       Scalar == 'half' ->
	    erlang:error({bad_vector_type,Scalar,Size});
       Size == 2 -> {Scalar,2};
       Size == 4 -> {Scalar,4};
       Size == 8 -> {Scalar,8};
       Size == 16 -> {Scalar,16};
       true -> erlang:error({bad_vector_type,Scalar,Size})
    end.
    
t_type(cl_char)   -> 'char';
t_type(cl_uchar)  -> 'uchar';
t_type(cl_short)  -> 'short';
t_type(cl_ushort) -> 'ushort';
t_type(cl_int)    -> 'int';
t_type(cl_uint)   -> 'uint';
t_type(cl_long)   -> 'long';
t_type(cl_ulong)  -> 'ulong';
t_type(cl_float)  -> 'float';
t_type(cl_half)   -> 'half';
t_type(T) ->
    erlang:error({bad_type,T}).

