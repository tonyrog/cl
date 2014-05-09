%%% @author Tony Rogvall <tony@rogvall.se>
%%% @copyright (C) 2014, Tony Rogvall
%%% @doc
%%%     cl_image test
%%% @end
%%% Created :  9 May 2014 by Tony Rogvall <tony@rogvall.se>

-module(cl_image).
-compile(export_all).

-include_lib("cl/include/cl.hrl").

create_image2d_a() ->
    C = clu:setup(gpu),
    cl:create_image2d(clu:context(C),[read_write],
		      #cl_image_format { cl_channel_order = rgba,
					 cl_channel_type  = unorm_int8 },
		      64,
		      64,
		      0,
		      <<>>).

create_image2d_b() ->
    C = clu:setup(gpu),
    ImageData = create_image2d_data(64, 64, 4),
    cl:create_image2d(clu:context(C),[read_write],
		      #cl_image_format { cl_channel_order = rgba,
					 cl_channel_type  = unorm_int8 },
		      64,
		      64,
		      64*4,
		      ImageData).

create_image2d_c() ->
    true = lists:member({1,2},cl:versions()),
    C = clu:setup(gpu),
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

create_image2d_d() ->
    true = lists:member({1,2},cl:versions()),
    C = clu:setup(gpu),
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


create_image3d_a() ->
    C = clu:setup(gpu),
    cl:create_image3d(clu:context(C),[read_write],
		      #cl_image_format { cl_channel_order = rgba,
					 cl_channel_type  = unorm_int8 },
		      64,
		      64,
		      64,
		      0,
		      0,
		      <<>>).

create_image3d_b() ->
    C = clu:setup(gpu),
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

create_image3d_c() ->
    true = lists:member({1,2},cl:versions()),
    C = clu:setup(gpu),
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

create_image3d_d() ->
    true = lists:member({1,2},cl:versions()),
    C = clu:setup(gpu),
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
