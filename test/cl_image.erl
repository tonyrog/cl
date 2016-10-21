%%% @author Tony Rogvall <tony@rogvall.se>
%%% @copyright (C) 2014, Tony Rogvall
%%% @doc
%%%     cl_image test
%%% @end
%%% Created :  9 May 2014 by Tony Rogvall <tony@rogvall.se>

-module(cl_image).
-compile(export_all).

-include_lib("cl/include/cl.hrl").

init_per_suite(Config) -> cl_SUITE:init_per_suite(Config).
    

all() ->
    [create_image2d_a, create_image2d_b, create_image2d_c, create_image2d_d,
     create_image3d_a, create_image3d_b, create_image3d_c, create_image3d_d,
     pixop
    ].


create_image2d_a(Config) ->
    C = clu:setup(proplists:get_value(type,Config,gpu)),
    cl:create_image2d(clu:context(C),[read_write],
		      #cl_image_format { cl_channel_order = rgba,
					 cl_channel_type  = unorm_int8 },
		      64,
		      64,
		      0,
		      <<>>).

create_image2d_b(Config) ->
    C = clu:setup(proplists:get_value(type,Config,gpu)),
    ImageData = create_image2d_data(64, 64, 4),
    cl:create_image2d(clu:context(C),[read_write],
		      #cl_image_format { cl_channel_order = rgba,
					 cl_channel_type  = unorm_int8 },
		      64,
		      64,
		      64*4,
		      ImageData).

create_image2d_c(Config) ->
    true = lists:member({1,2},cl:versions()),
    C = clu:setup(proplists:get_value(type,Config,gpu)),
    cl:create_image(clu:context(C),[read_write],
		    #cl_image_format { cl_channel_order = rgba,
				       cl_channel_type  = unorm_int8 },
		    #cl_image_desc {
		       image_type = image2d,
		       image_width = 64,
		       image_height = 64,
		       image_depth = 1,
		       image_array_size = 1,
		       image_row_pitch = 0 },
		    <<>>).

create_image2d_d(Config) ->
    true = lists:member({1,2},cl:versions()),
    C = clu:setup(proplists:get_value(type,Config,gpu)),
    ImageData = create_image2d_data(64, 64, 4),
    cl:create_image(clu:context(C),[read_write],
		    #cl_image_format { cl_channel_order = rgba,
				       cl_channel_type  = unorm_int8 },
		    #cl_image_desc {
		       image_type = image2d,
		       image_width = 64,
		       image_height = 64,
		       image_depth = 1,
		       image_array_size = 1,
		       image_row_pitch = 64*4 },
		    ImageData).

create_image2d_data(W,H,BytesPerPixel) ->
    << <<1234:BytesPerPixel/unit:8>> ||
	_ <- lists:seq(1,W),
	_ <- lists:seq(1,H) >>.


create_image3d_a(Config) ->
    C = clu:setup(proplists:get_value(type,Config,gpu)),
    cl:create_image3d(clu:context(C),[read_write],
		      #cl_image_format { cl_channel_order = rgba,
					 cl_channel_type  = unorm_int8 },
		      64,
		      64,
		      64,
		      0,
		      0,
		      <<>>).

create_image3d_b(Config) ->
    C = clu:setup(proplists:get_value(type,Config,gpu)),
    ImageData = create_image3d_data(64, 64, 64, 4),
    cl:create_image3d(clu:context(C),[read_write],
		      #cl_image_format { cl_channel_order = rgba,
					 cl_channel_type  = unorm_int8 },
		      64,
		      64,
		      64,
		      64*4,
		      64*64*4,
		      ImageData).

create_image3d_c(Config) ->
    true = lists:member({1,2},cl:versions()),
    C = clu:setup(proplists:get_value(type,Config,gpu)),
    cl:create_image(clu:context(C),[read_write],
		    #cl_image_format { cl_channel_order = rgba,
				       cl_channel_type  = unorm_int8 },
		    #cl_image_desc {
		       image_type = image3d,
		       image_width = 64,
		       image_height = 64,
		       image_depth = 64,
		       image_array_size = 1,
		       image_row_pitch = 0,
		       image_slice_pitch = 0
		      },
		    <<>>).

create_image3d_d(Config) ->
    true = lists:member({1,2},cl:versions()),
    C = clu:setup(proplists:get_value(type,Config,gpu)),
    ImageData = create_image3d_data(64, 64, 64, 4),
    cl:create_image(clu:context(C),[read_write],
		    #cl_image_format { cl_channel_order = rgba,
				       cl_channel_type  = unorm_int8 },
		    #cl_image_desc {
		       image_type = image3d,
		       image_width = 64,
		       image_height = 64,
		       image_depth = 64,
		       image_array_size = 1,
		       image_row_pitch = 64*4,
		       image_slice_pitch = 64*64*4
		      },
		    ImageData).


create_image3d_data(W,H,D,BytesPerPixel) ->
    << <<Di:BytesPerPixel/unit:8>> ||
	Di <- lists:seq(1,D),
	_ <- lists:seq(1,H),
	_ <- lists:seq(1,W)
    >>.

%% test image pixel operations

pixop(Config) ->
    exit({skip, "Fails on linux machine"}),
    Clu = clu:setup(proplists:get_value(type,Config,cpu)),
    {ok,A} =
	cl:create_image2d(clu:context(Clu),[read_write],
			  #cl_image_format { cl_channel_order = rgba,
					     cl_channel_type  = unorm_int8 },
			  2,
			  2,
			  2*4,
			  <<100,200,50,127, 25,255,50,100,
			    30,64,10,20,    3,2,1,220>> ),
    %% {ok,E1} = cl:enqueue_write_image(Q, A, [0,0], [2,2], 2*4, 0, Data, []),
    {ok,B} =
	cl:create_image2d(clu:context(Clu),[read_write],
			  #cl_image_format { cl_channel_order = rgba,
					     cl_channel_type  = unorm_int8 },
			  2,
			  2,
			  2*4,
			  <<50,100,25,255,  100,100,100,127,
			    100,200,50,127, 1,2,3,20>>),
    {ok,C} =
	cl:create_image2d(clu:context(Clu),[read_write],
			  #cl_image_format { cl_channel_order = rgba,
					     cl_channel_type  = unorm_int8 },
			  2,
			  2,
			  0,
			  <<>>),

    {ok,Q} = cl:create_queue(clu:context(Clu),clu:device(Clu),[]),
    File =
	case proplists:get_value(data_dir, Config) of
	    false -> "pixop.cl";
	    Dir -> filename:join(filename:dirname(filename:dirname(Dir)), "pixop.cl")
	end,
    io:format("File: ~p~n", [File]),
    {ok,Program} = clu:build_source_file(Clu, File, ""),
    {ok,Kernel} = cl:create_kernel(Program, "pixmap_blend"),
    clu:apply_kernel_args(Kernel, [A,B,C,2,2]),
    {ok,E1} = cl:enqueue_nd_range_kernel(Q, Kernel,
					 [2,2], [],
					 []),
    cl:flush(Q),
    {ok,completed} = cl:wait(E1),

    {ok,E2}  = cl:enqueue_read_image(Q, C, [0,0], [2,2], 2*4, 0, []),
    cl:flush(Q),
    {ok,Data} = cl:wait(E2),
    Data.
