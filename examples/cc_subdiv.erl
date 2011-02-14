%%%-------------------------------------------------------------------
%%% File    : cc_subdiv.erl
%%% Author  : Dan Gudmundsson
%%% Description : Catmull Clark subdivision in OpenCL 
%%%               The example is the same as I will use in wings3D
%%% Created : 8 Feb 2011
%%%-------------------------------------------------------------------
-module(cc_subdiv).
-compile(export_all).

-include_lib("wx/include/wx.hrl"). 
-include_lib("wx/include/gl.hrl"). 
-include_lib("cl/include/cl.hrl").

-record(cli,      {context, kernels, q, cl, device, 
		   %% CL temp buffers and respective sizes
		   vab, vab_sz=0, fl, fl_sz=0, fi, fi_sz=0}).
-record(cl_mem,   {v, v_no, f, fs_no, e, e_no, fi, fi0}).
-record(kernel,   {name, id, wg}).

-record(base, {v,    %% array of {x,y,z, {Valance, HardEdges}} nv
	       f,    %% array of [v0,v1..,vn]   nf
	       e,    %% array of v0,v1,f1,f2    ne
	       level %% Subdiv levels
	      }).
-define(I32,  32/signed-native).

-record(state, {f,    % wxFrame
		cl,   % CL record above
		gl,   % wxGLCanvas
		orig, % Orig Mesh 
		sd    % Sub Mesh
	       }).

start() ->
    WX = wx:new(),
    Frame   = wxFrame:new(WX,1,"OpenCL does CC subdivision",[{size, {800,600}}]),
    ok = wxFrame:connect(Frame, close_window),
    wxFrame:createStatusBar(Frame,[]),
    setup_menus(Frame),
    GLAttrs = [?WX_GL_RGBA,?WX_GL_DOUBLEBUFFER,0],
    Canvas = wxGLCanvas:new(Frame, [{attribList, GLAttrs},{size, {800,600}}]),
    Self = self(),
    Redraw = fun(_Ev,_) ->   
		     DC = wxPaintDC:new(Canvas),
 		     Self ! repaint,
 		     wxPaintDC:destroy(DC)
 	     end,
    wxFrame:connect(Canvas, paint, [{callback, Redraw}]),
    wxWindow:show(Frame),    %% Must show to initilize context.
    wxGLCanvas:setCurrent(Canvas), %% Init context
    Base = #base{v=verts(), f=faces(), e=edges(), level=4},
    initGL(Canvas),
    CL0 = initCL(),
    {In, Out, CL} = cl_allocate(Base, CL0),
    Wait0 = cl_write_input(Base, In, Out, CL),
    OrigMesh = setup_gl_buff(gen_va(size(faces()) div 16, In, Wait0, CL)),
    Wait1 = cl_write_input(Base, In, Out, CL),
    SDMesh   = setup_gl_buff(subdiv(4, In, Out, Wait1, CL)),
    gl:clear(?GL_COLOR_BUFFER_BIT bor ?GL_DEPTH_BUFFER_BIT),
    draw_buff(OrigMesh),
    wxGLCanvas:swapBuffers(Canvas),
    R = loop(0, #state{f=Frame, cl=CL, gl=Canvas, orig=OrigMesh, sd=SDMesh}),
    wx:destroy(),
    R.

loop(R, S = #state{f=Frame, cl=CL}) ->
    receive 
	#wx{event=#wxClose{}} ->
	    quit;
	#wx{id=?wxID_EXIT} ->
	    quit;
	#wx{id=?wxID_ABOUT} ->
	    about_box(Frame, CL),
	    loop(R, S);
	_Msg ->
	    draw(R, S),
	    loop(R, S)
    after 10 ->
	    draw(R, S),
	    _ = wxWindow:getSize(Frame),
	    loop(R+1, S)
    end.

draw(R, #state{gl=Canvas, orig=OrigMesh, sd=SDMesh}) ->
    gl:clear(?GL_COLOR_BUFFER_BIT bor ?GL_DEPTH_BUFFER_BIT),
    gl:matrixMode(?GL_MODELVIEW),
    gl:loadIdentity(),  
    glu:lookAt(15,15,15, 0,0,0, 0,1,0),    
    drawBox(R),
    gl:disable(?GL_BLEND),
    gl:color4f(1.0,1.0,0.0,1.0),
    draw_buff(SDMesh),
    gl:enable(?GL_BLEND),
    gl:color4f(0.5,0.5,0.5,0.5),
    draw_buff(OrigMesh),
    wxGLCanvas:swapBuffers(Canvas).

gen_va(NoFs, #cl_mem{v=Vs, f=Fs}, Wait, CL=#cli{q=Q, vab=Vab}) ->
    WVab = cl_apply(collect_face_info,[Vs,Fs,Vab,NoFs], NoFs, Wait,CL),
    {ok, WData} = cl:enqueue_read_buffer(Q,Vab,0,NoFs*4*6*4,[WVab]),
    {ok, Bin} = cl:wait(WData),
    Bin.

setup_gl_buff(Data) ->
    [Buff] = gl:genBuffers(1),
    gl:bindBuffer(?GL_ARRAY_BUFFER,Buff),
    gl:bufferData(?GL_ARRAY_BUFFER, size(Data), Data, ?GL_STATIC_DRAW),
    <<_:3/unit:32,Ns/bytes>> = Data,
    {Buff, Ns, size(Data) div (6*4)}.

draw_buff(Data = {Buff,_Ns,NoVs}) ->
    gl:bindBuffer(?GL_ARRAY_BUFFER,Buff),
    gl:vertexPointer(3, ?GL_FLOAT, 6*4, 0),
    gl:normalPointer(?GL_FLOAT, 6*4, 3*4),
    gl:enableClientState(?GL_VERTEX_ARRAY),
    gl:enableClientState(?GL_NORMAL_ARRAY),
    gl:drawArrays(?GL_QUADS, 0, NoVs),
    Data.

subdiv(N, In, Out, Wait0, CL) ->
    {Res, Wait} = subdiv_1(N, In, Out, CL, Wait0),
    gen_va(Res#cl_mem.fs_no, Res, Wait, CL).

subdiv_1(N,
	  In = #cl_mem{v=VsIn, f=FsIn, fi=FiIn, e=EsIn,
		       v_no=NoVs, fs_no=NoFs, e_no=NoEs},
	  Out= #cl_mem{v=VsOut, f=FsOut, e=EsOut, fi=FiOut,
		       v_no=NoVs1,fs_no=NoFs1, e_no=NoEs1},
	  CL, Wait0)
  when N > 0 ->
    Args1 = [VsIn, FsIn, FiIn, VsOut, FsOut, NoFs, NoVs],
    W0 = cl_apply(gen_faces, Args1, NoFs, Wait0, CL),
    [cl:release_event(Ev) || Ev <- Wait0],
    Args2 = [FsIn, FiIn, VsOut, NoFs, NoVs],
    W1 = cl_apply(add_center, Args2, 1, [W0], CL),

    Args3 = [VsIn, FsIn, EsIn, FiIn, 
	     VsOut, FsOut, EsOut, 
	     NoFs, NoVs, NoEs],
    W2 = cl_apply(gen_edges, Args3, NoEs, [W1], CL),
    Args4 = [VsIn, VsOut, EsIn, NoEs],
    W3 = cl_apply(add_edge_verts, Args4, 1, [W2], CL),

    Args5 = [VsIn,VsOut,NoVs,NoVs1],
    Wait = cl_apply(move_verts, Args5, NoVs1, [W3], CL),
    %% cl_vs("cvs_out3", N, VsOut, NoVs1, CL, Wait),
    [cl:release_event(Ev) || Ev <- [W0,W1,W2,W3]],
    subdiv_1(N-1, Out, 
	      In#cl_mem{fi=FiOut, v_no=NoVs1+NoFs1+NoEs1,
			fs_no=NoFs1*4, e_no=NoEs1*4},
	      CL, [Wait]);
subdiv_1(_C, ResultBuffs, _OutBuffs, _, Wait) ->
    {ResultBuffs,Wait}.

initCL() ->
    Opts = [],
    Prefered = proplists:get_value(cl_type, Opts, cpu),
    Other = [gpu,cpu] -- [Prefered],
    CL = case clu:setup(Prefered) of 
	     {error, _} -> 
		 case clu:setup(Other) of
		     {error, R} -> 
			 exit({no_opencl_device, R});
		     Cpu -> Cpu
		 end;
	     Gpu ->
		 Gpu
	 end,
    [Device|_] = CL#cl.devices,
    {ok,Queue} = cl:create_queue(CL#cl.context,Device,[]),
    %%% Compile
    Dir = filename:join(code:lib_dir(cl),"examples"),
    Bin = case file:read_file(filename:join([Dir, "cc_subdiv.cl"])) of
	      {ok, B} -> B;
	      {error, _} ->
		  io:format("OpenCL code not found run: erl -pa ABS_PATH/cl/ebin~n", []),
		  exit({file_not_found, Dir})
	  end,
    case clu:build_source(CL, Bin) of
	{error, {Err={error,build_program_failure}, _}} ->
	    %% io:format("~s", [Str]),
	    exit(Err);
	{ok, Program} -> 
	    {ok, MaxWGS} = cl:get_device_info(Device, max_work_group_size),
	    {ok, Kernels0} = cl:create_kernels_in_program(Program),
	    Kernels = [kernel_info(K,Device, MaxWGS) || K <- Kernels0],
	    %% io:format("Kernels ~p~n",[Kernels]),
	    CLI = #cli{context=CL#cl.context,kernels=Kernels,
		       q=Queue, device=Device, cl=CL},
	    cl:release_program(Program),
	    CLI
    end.

kernel_info(K,Device, MaxWGS) ->
    {ok, WG} = cl:get_kernel_workgroup_info(K, Device, work_group_size),
    {ok, Name} = cl:get_kernel_info(K, function_name),
    #kernel{name=list_to_atom(Name), wg=min(WG,MaxWGS), id=K}.

cl_apply(Name, Args, No, Wait, #cli{q=Q, kernels=Ks}) ->
    #kernel{id=K, wg=WG0} = lists:keyfind(Name, 2, Ks),
    try clu:apply_kernel_args(K, Args) of
	ok -> ok
    catch error:Reason ->
	    io:format("Bad args ~p: ~p~n",[Name, Args]),
	    erlang:raise(error,Reason, erlang:get_stacktrace())
    end,
    {GWG,WG} = if  No > WG0  -> 
		       {(1+(No div WG0))*WG0, WG0};
		   true -> {No,No}
	       end,
    {ok, Event} = cl:enqueue_nd_range_kernel(Q,K,[GWG],[WG],Wait),
    Event.

%% OpenCL Memory allocation
cl_allocate(Base, CL0=#cli{context=Ctxt}) ->
    {NoFs,NoEs,NoVs,NoFs1,MaxFs,MaxEs,MaxVs} = verify_size(Base, CL0),
    {ok,FsIn}  = cl:create_buffer(Ctxt, [], MaxFs*16),
    {ok,EsIn}  = cl:create_buffer(Ctxt, [], MaxEs*16),
    {ok,VsIn}  = cl:create_buffer(Ctxt, [], MaxVs*16),
    
    {ok,FsOut} = cl:create_buffer(Ctxt, [], MaxFs*16),
    {ok,EsOut} = cl:create_buffer(Ctxt, [], MaxEs*16),
    {ok,VsOut} = cl:create_buffer(Ctxt, [], MaxVs*16),
   
    CL = #cli{fi=FiOut} = check_temp_buffs(CL0, MaxFs),
    FiIn = FiOut,
    {#cl_mem{v=VsIn, f=FsIn, e=EsIn, fi=FiIn, fi0=FiIn,
	     v_no=NoVs, fs_no=NoFs, e_no=NoEs},
     #cl_mem{v=VsOut, f=FsOut, e=EsOut, fi=FiOut, fi0=FiIn,
	     v_no=NoVs+NoFs+NoEs, fs_no=NoFs1, e_no=NoEs*4},
     CL}.

cl_write_input(#base{f=Fs,e=Es,v=Vs}, 
	       #cl_mem{v=VsIn,f=FsIn,e=EsIn}, #cl_mem{v=VsOut}, 
	       #cli{q=Q}) ->
    {ok, W1} = cl:enqueue_write_buffer(Q,  VsIn, 0, byte_size(Vs), Vs, []),
    {ok, W2} = cl:enqueue_write_buffer(Q, VsOut, 0, byte_size(Vs), Vs, []),
    {ok, W3} = cl:enqueue_write_buffer(Q,  FsIn, 0, byte_size(Fs), Fs, []),
    {ok, W4} = cl:enqueue_write_buffer(Q,  EsIn, 0, byte_size(Es), Es, []),
    [W1,W2,W3,W4].
    
cl_release(#cl_mem{v=Vs,f=Fs,e=Es, fi0=Fi0}, All) ->
    Vs /= undefined andalso cl:release_mem_object(Vs),
    Fs /= undefined andalso cl:release_mem_object(Fs),
    Es /= undefined andalso cl:release_mem_object(Es),
    All andalso cl:release_mem_object(Fi0).

check_temp_buffs(CL=#cli{context=Ctxt, 
			 vab=Vab0, vab_sz=VabSz0, 
			 fl=FL0, fl_sz=FLSz0, 
			 fi=Fi0, fi_sz=FiSz0}, MaxFs0) ->
    MaxFs = trunc(MaxFs0*1.5),  
    %% Overallocate so we don't need new buffers all the time
    GenFi = fun() -> 
		    << <<(C*4):?I32, 4:?I32>> || 
			C <- lists:seq(0, MaxFs-1) >> 
	    end,
    {Vab,VabSz} = check_temp(Vab0,VabSz0,MaxFs*(3+3)*4*4,
			     Ctxt,[write_only],none),
    {FL,FLSz} = check_temp(FL0,FLSz0,MaxFs*3*4,
			   Ctxt,[read_only],none),
    {Fi,FiSz} = check_temp(Fi0,FiSz0,MaxFs*2*4,
			   Ctxt,[read_only],GenFi),
    CLI = CL#cli{vab=Vab, vab_sz=VabSz, 
		 fl=FL, fl_sz=FLSz, 
		 fi=Fi, fi_sz=FiSz},
    put({?MODULE, cl}, CLI),
    CLI.

check_temp(Buff, Current, Req, _, _, _) 
  when Current >= Req ->
    {Buff, Current};
check_temp(undefined, _, Req, Ctxt, Opt, none) ->
    {ok, Buff} = cl:create_buffer(Ctxt, Opt, Req),
    {Buff, Req};
check_temp(undefined, _, Req, Ctxt, Opt, Fun) ->
    {ok,Buff} = cl:create_buffer(Ctxt, Opt, Req, Fun()),
    {Buff, Req};
check_temp(Buff0, _, Req, Ctxt, Opt, Data) ->
    cl:release_mem_object(Buff0),
    check_temp(undefined, 0, Req, Ctxt, Opt, Data).

verify_size(#base{f=Fs, e=Es, v=Vs, level=N}, #cli{device=Device}) ->
    NoFs = size(Fs) div 16,
    NoEs = size(Es) div 16,
    NoVs = size(Vs) div 16,
    
    {ok, DevTotal} = cl:get_device_info(Device, max_mem_alloc_size),
    Res = verify_size_1(N-1, N, NoFs*4, NoEs*4, NoVs+NoEs+NoFs, DevTotal),
    case Res of
	false -> 
	    io:format("Can not subdivide, out of memory~n",[]),
	    exit(out_of_memory);
	{MaxFs, MaxEs, MaxVs} ->
	    {NoFs, NoEs, NoVs, NoFs*4, MaxFs, MaxEs, MaxVs}
    end.
	
verify_size_1(N, No, Fs, Es, Vs, CardMax) ->
    VertexSz = (3+3)*4*4,
    Total = Fs*VertexSz+2*(Fs*16+Es*16+Vs*16),
    case Total < CardMax of
	true when N == 0 ->
	    {Fs,Es,Vs};
	true -> 
	    case verify_size_1(N-1, No, Fs*4, Es*4, Vs+Fs+Es, CardMax) of
		false -> 
		    io:format("Out of memory, does not meet the number of sub-division"
			      "levels ~p(~p)~n",[No-N,No]),
		    {Fs,Es,Vs};
		Other -> Other
	    end;
	false ->
	    false
    end.

%%%%% OpenGL

initGL(Canvas) ->
    {W,H} = wxWindow:getClientSize(Canvas),
    io:format("Size ~p ~n",[{W,H}]),
    gl:viewport(0,0,W,H),

    gl:matrixMode(?GL_PROJECTION),
    gl:loadIdentity(),
    gl:ortho( -10.0, 10.0, -10.0*H/W, 10.0*H/W, -100.0, 100.0),
    
    gl:enable(?GL_DEPTH_TEST),
    gl:depthFunc(?GL_LESS),
    gl:clearColor(0.8,0.8,0.8,1.0),
    gl:shadeModel(?GL_SMOOTH),
    gl:disable(?GL_CULL_FACE),
    %% Nowadays you should really use a shader to do the lighting but I'm lazy.
    gl:enable(?GL_COLOR_MATERIAL),
    gl:enable(?GL_LIGHTING),
    gl:lightfv(?GL_LIGHT0, ?GL_DIFFUSE,  {1,1,1,1}), 
    gl:lightfv(?GL_LIGHT0, ?GL_SPECULAR, {0.5,0.5,0.5,1}),
    gl:lightfv(?GL_LIGHT0, ?GL_POSITION, {0.71,0.71,0.0,0.0}),
    gl:enable(?GL_LIGHT0),
    gl:enable(?GL_BLEND),
    gl:blendFunc(?GL_SRC_ALPHA, ?GL_ONE_MINUS_SRC_ALPHA),
    ok.

-define(VS, {{ 0.5,  0.5, -0.5},  %1
	     { 0.5, -0.5, -0.5},  %2
	     {-0.5, -0.5, -0.5},   
	     {-0.5,  0.5, -0.5},  %4
	     {-0.5,  0.5,  0.5},
	     { 0.5,  0.5,  0.5},  %6
	     { 0.5, -0.5,  0.5}, 
	     {-0.5, -0.5,  0.5}}).%8

-define(FS, 
	%% Faces    Normal   
	[{{1,2,3,4},{0,0,-1} },   % 
	 {{3,8,5,4},{-1,0,0}},   %
	 {{1,6,7,2},{1,0,0} },   %
	 {{6,5,8,7},{0,0,1} },   %
	 {{6,1,4,5},{0,1,0} },   %
	 {{7,8,3,2},{0,-1,0}}]).

drawBox(Deg) ->
    gl:matrixMode(?GL_MODELVIEW),
    gl:loadIdentity(),
    gl:rotatef(Deg, 0.0, 1.0, 0.3),
    gl:rotatef(20, 1.0, 0.0, 1.0),
    gl:'begin'(?GL_QUADS),    
    lists:foreach(fun(Face) -> drawFace(Face,?VS) end, ?FS),
    gl:'end'().

drawFace({{V1,V2,V3,V4},N={N1,N2,N3}}, Cube) ->
    gl:normal3fv(N),
    gl:color3f(abs(N1),abs(N2),abs(N3)),
    gl:texCoord2f(0.0, 1.0), gl:vertex3fv(element(V1, Cube)),
    gl:texCoord2f(0.0, 0.0), gl:vertex3fv(element(V2, Cube)),
    gl:texCoord2f(1.0, 0.0), gl:vertex3fv(element(V3, Cube)),
    gl:texCoord2f(1.0, 1.0), gl:vertex3fv(element(V4, Cube)).

setup_menus(Frame) ->
    MenuBar = wxMenuBar:new(),
    Menu    = wxMenu:new([]),
    true = wxMenuBar:append(MenuBar, Menu, "File"),
    wxMenu:append(Menu, ?wxID_ABOUT,"About"),
    wxMenu:append(Menu, ?wxID_EXIT, "Quit"),
    
    ok = wxFrame:connect(Frame, command_menu_selected), 
    ok = wxFrame:setMenuBar(Frame,MenuBar).

about_box(Frame, #cli{device=Device}) ->
    Env = wx:get_env(),
    OsInfo = [wx_misc:getOsDescription(),gl:getString(?GL_VENDOR),
	      gl:getString(?GL_RENDERER),gl:getString(?GL_VERSION)],

    DeviceInfo = [{Type, cl:get_device_info(Device, Type)} 
		  || Type <- [name, vendor, version]],

    spawn(fun() ->
		  wx:set_env(Env),
		  Str = "An OpenGL demo showing how to combine "
		      " OpenCL and OpenGL, Catmull-Clark subdivision is done in OpenCL\n"
		      " The transparent \"box\" is the original mesh and the subdivided"
		      " yellow pipes is the result of the subdivision\n\n",
		  
		  Info = io_lib:format("Os:         ~s~n~nGL Vendor:     ~s~n"
				       "GL Renderer:  ~s~nGL Version:    ~s~n",
				       OsInfo), 
		  CLInfo = [io_lib:format("~-25.w   ~s~n",[Type,I]) ||
			       {Type, {ok, I}} <- DeviceInfo],

		  MD = wxMessageDialog:new(Frame, [Str, Info, "\nOpenCL info:\n",CLInfo], 
					   [{style, ?wxOK}, 
					    {caption, "Opengl Example"}]),
		  wxDialog:showModal(MD),
		  wxDialog:destroy(MD)
	  end),
    ok.

faces() ->
    <<1,0,0,0,3,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,5,0,0,0,3,0,0,0,1,0,0,0,6,0,0,0,
      7,0,0,0,5,0,0,0,4,0,0,0,0,0,0,0,2,0,0,0,7,0,0,0,6,0,0,0,4,0,0,0,9,0,0,0,
      8,0,0,0,6,0,0,0,10,0,0,0,11,0,0,0,9,0,0,0,4,0,0,0,12,0,0,0,13,0,0,0,11,
      0,0,0,10,0,0,0,6,0,0,0,8,0,0,0,13,0,0,0,12,0,0,0,1,0,0,0,15,0,0,0,14,0,
      0,0,4,0,0,0,16,0,0,0,17,0,0,0,15,0,0,0,1,0,0,0,10,0,0,0,18,0,0,0,17,0,0,
      0,16,0,0,0,4,0,0,0,14,0,0,0,18,0,0,0,10,0,0,0,6,0,0,0,20,0,0,0,19,0,0,0,
      0,0,0,0,12,0,0,0,21,0,0,0,20,0,0,0,6,0,0,0,22,0,0,0,23,0,0,0,21,0,0,0,12,
      0,0,0,0,0,0,0,19,0,0,0,23,0,0,0,22,0,0,0,22,0,0,0,25,0,0,0,24,0,0,0,16,0,
      0,0,12,0,0,0,26,0,0,0,25,0,0,0,22,0,0,0,10,0,0,0,27,0,0,0,26,0,0,0,12,0,
      0,0,16,0,0,0,24,0,0,0,27,0,0,0,10,0,0,0,0,0,0,0,29,0,0,0,28,0,0,0,1,0,0,
      0,22,0,0,0,30,0,0,0,29,0,0,0,0,0,0,0,16,0,0,0,31,0,0,0,30,0,0,0,22,0,0,
      0,1,0,0,0,28,0,0,0,31,0,0,0,16,0,0,0,29,0,0,0,33,0,0,0,32,0,0,0,28,0,0,
      0,30,0,0,0,34,0,0,0,33,0,0,0,29,0,0,0,31,0,0,0,35,0,0,0,34,0,0,0,30,0,0,
      0,28,0,0,0,32,0,0,0,35,0,0,0,31,0,0,0,25,0,0,0,37,0,0,0,36,0,0,0,24,0,0,
      0,26,0,0,0,38,0,0,0,37,0,0,0,25,0,0,0,27,0,0,0,39,0,0,0,38,0,0,0,26,0,0,
      0,24,0,0,0,36,0,0,0,39,0,0,0,27,0,0,0,20,0,0,0,41,0,0,0,40,0,0,0,19,0,0,
      0,21,0,0,0,42,0,0,0,41,0,0,0,20,0,0,0,23,0,0,0,43,0,0,0,42,0,0,0,21,0,0,
      0,19,0,0,0,40,0,0,0,43,0,0,0,23,0,0,0,15,0,0,0,45,0,0,0,44,0,0,0,14,0,0,
      0,17,0,0,0,46,0,0,0,45,0,0,0,15,0,0,0,18,0,0,0,47,0,0,0,46,0,0,0,17,0,0,
      0,14,0,0,0,44,0,0,0,47,0,0,0,18,0,0,0,9,0,0,0,49,0,0,0,48,0,0,0,8,0,0,0,
      11,0,0,0,50,0,0,0,49,0,0,0,9,0,0,0,13,0,0,0,51,0,0,0,50,0,0,0,11,0,0,0,
      8,0,0,0,48,0,0,0,51,0,0,0,13,0,0,0,3,0,0,0,53,0,0,0,52,0,0,0,2,0,0,0,5,
      0,0,0,54,0,0,0,53,0,0,0,3,0,0,0,7,0,0,0,55,0,0,0,54,0,0,0,5,0,0,0,2,0,
      0,0,52,0,0,0,55,0,0,0,7,0,0,0>>.

edges() ->
    <<1,0,0,0,0,0,0,0,20,0,0,0,0,0,0,0,1,0,0,0,4,0,0,0,1,0,0,0,8,0,0,0,1,0,0,
      0,16,0,0,0,9,0,0,0,23,0,0,0,0,0,0,0,6,0,0,0,12,0,0,0,3,0,0,0,0,0,0,0,22,
      0,0,0,21,0,0,0,15,0,0,0,6,0,0,0,4,0,0,0,4,0,0,0,2,0,0,0,6,0,0,0,12,0,0,
      0,13,0,0,0,7,0,0,0,4,0,0,0,10,0,0,0,5,0,0,0,11,0,0,0,16,0,0,0,22,0,0,0,
      16,0,0,0,22,0,0,0,16,0,0,0,10,0,0,0,10,0,0,0,19,0,0,0,22,0,0,0,12,0,0,0,
      17,0,0,0,14,0,0,0,12,0,0,0,10,0,0,0,18,0,0,0,6,0,0,0,3,0,0,0,2,0,0,0,0,
      0,0,0,44,0,0,0,0,0,0,0,2,0,0,0,3,0,0,0,0,0,0,0,5,0,0,0,3,0,0,0,1,0,0,0,
      45,0,0,0,1,0,0,0,3,0,0,0,0,0,0,0,1,0,0,0,7,0,0,0,5,0,0,0,2,0,0,0,46,0,0,
      0,4,0,0,0,5,0,0,0,1,0,0,0,2,0,0,0,2,0,0,0,7,0,0,0,3,0,0,0,47,0,0,0,6,0,0,
      0,7,0,0,0,2,0,0,0,3,0,0,0,9,0,0,0,8,0,0,0,4,0,0,0,40,0,0,0,6,0,0,0,8,0,0,
      0,7,0,0,0,4,0,0,0,11,0,0,0,9,0,0,0,5,0,0,0,41,0,0,0,4,0,0,0,9,0,0,0,4,0,
      0,0,5,0,0,0,13,0,0,0,11,0,0,0,6,0,0,0,42,0,0,0,10,0,0,0,11,0,0,0,5,0,0,0,
      6,0,0,0,8,0,0,0,13,0,0,0,7,0,0,0,43,0,0,0,12,0,0,0,13,0,0,0,6,0,0,0,7,0,
      0,0,15,0,0,0,14,0,0,0,8,0,0,0,36,0,0,0,4,0,0,0,14,0,0,0,11,0,0,0,8,0,0,
      0,17,0,0,0,15,0,0,0,9,0,0,0,37,0,0,0,1,0,0,0,15,0,0,0,8,0,0,0,9,0,0,0,
      18,0,0,0,17,0,0,0,10,0,0,0,38,0,0,0,16,0,0,0,17,0,0,0,9,0,0,0,10,0,0,0,
      14,0,0,0,18,0,0,0,11,0,0,0,39,0,0,0,10,0,0,0,18,0,0,0,10,0,0,0,11,0,0,0,
      20,0,0,0,19,0,0,0,12,0,0,0,32,0,0,0,0,0,0,0,19,0,0,0,15,0,0,0,12,0,0,0,
      21,0,0,0,20,0,0,0,13,0,0,0,33,0,0,0,6,0,0,0,20,0,0,0,12,0,0,0,13,0,0,0,
      23,0,0,0,21,0,0,0,14,0,0,0,34,0,0,0,12,0,0,0,21,0,0,0,13,0,0,0,14,0,0,0,
      19,0,0,0,23,0,0,0,15,0,0,0,35,0,0,0,22,0,0,0,23,0,0,0,14,0,0,0,15,0,0,0,
      25,0,0,0,24,0,0,0,16,0,0,0,28,0,0,0,16,0,0,0,24,0,0,0,19,0,0,0,16,0,0,0,
      26,0,0,0,25,0,0,0,17,0,0,0,29,0,0,0,22,0,0,0,25,0,0,0,16,0,0,0,17,0,0,0,
      27,0,0,0,26,0,0,0,18,0,0,0,30,0,0,0,12,0,0,0,26,0,0,0,17,0,0,0,18,0,0,0,
      24,0,0,0,27,0,0,0,19,0,0,0,31,0,0,0,10,0,0,0,27,0,0,0,18,0,0,0,19,0,0,0,
      29,0,0,0,28,0,0,0,20,0,0,0,24,0,0,0,1,0,0,0,28,0,0,0,23,0,0,0,20,0,0,0,
      30,0,0,0,29,0,0,0,21,0,0,0,25,0,0,0,0,0,0,0,29,0,0,0,20,0,0,0,21,0,0,0,
      31,0,0,0,30,0,0,0,22,0,0,0,26,0,0,0,22,0,0,0,30,0,0,0,21,0,0,0,22,0,0,0,
      28,0,0,0,31,0,0,0,23,0,0,0,27,0,0,0,16,0,0,0,31,0,0,0,22,0,0,0,23,0,0,0,
      222,255,255,255,32,0,0,0,24,0,0,0,255,255,255,255,28,0,0,0,32,0,0,0,27,0,
      0,0,24,0,0,0,221,255,255,255,33,0,0,0,25,0,0,0,255,255,255,255,29,0,0,0,
      33,0,0,0,24,0,0,0,25,0,0,0,220,255,255,255,34,0,0,0,26,0,0,0,255,255,255,
      255,30,0,0,0,34,0,0,0,25,0,0,0,26,0,0,0,223,255,255,255,35,0,0,0,27,0,0,
      0,255,255,255,255,31,0,0,0,35,0,0,0,26,0,0,0,27,0,0,0,218,255,255,255,
      36,0,0,0,28,0,0,0,255,255,255,255,24,0,0,0,36,0,0,0,31,0,0,0,28,0,0,0,
      217,255,255,255,37,0,0,0,29,0,0,0,255,255,255,255,25,0,0,0,37,0,0,0,28,
      0,0,0,29,0,0,0,216,255,255,255,38,0,0,0,30,0,0,0,255,255,255,255,26,0,0,
      0,38,0,0,0,29,0,0,0,30,0,0,0,219,255,255,255,39,0,0,0,31,0,0,0,255,255,
      255,255,27,0,0,0,39,0,0,0,30,0,0,0,31,0,0,0,214,255,255,255,40,0,0,0,32,
      0,0,0,255,255,255,255,19,0,0,0,40,0,0,0,35,0,0,0,32,0,0,0,213,255,255,
      255,41,0,0,0,33,0,0,0,255,255,255,255,20,0,0,0,41,0,0,0,32,0,0,0,33,0,
      0,0,212,255,255,255,42,0,0,0,34,0,0,0,255,255,255,255,21,0,0,0,42,0,0,
      0,33,0,0,0,34,0,0,0,215,255,255,255,43,0,0,0,35,0,0,0,255,255,255,255,
      23,0,0,0,43,0,0,0,34,0,0,0,35,0,0,0,210,255,255,255,44,0,0,0,36,0,0,0,
      255,255,255,255,14,0,0,0,44,0,0,0,39,0,0,0,36,0,0,0,209,255,255,255,45,
      0,0,0,37,0,0,0,255,255,255,255,15,0,0,0,45,0,0,0,36,0,0,0,37,0,0,0,208,
      255,255,255,46,0,0,0,38,0,0,0,255,255,255,255,17,0,0,0,46,0,0,0,37,0,0,
      0,38,0,0,0,211,255,255,255,47,0,0,0,39,0,0,0,255,255,255,255,18,0,0,0,
      47,0,0,0,38,0,0,0,39,0,0,0,206,255,255,255,48,0,0,0,40,0,0,0,255,255,
      255,255,8,0,0,0,48,0,0,0,43,0,0,0,40,0,0,0,205,255,255,255,49,0,0,0,41,
      0,0,0,255,255,255,255,9,0,0,0,49,0,0,0,40,0,0,0,41,0,0,0,204,255,255,
      255,50,0,0,0,42,0,0,0,255,255,255,255,11,0,0,0,50,0,0,0,41,0,0,0,42,0,
      0,0,207,255,255,255,51,0,0,0,43,0,0,0,255,255,255,255,13,0,0,0,51,0,0,
      0,42,0,0,0,43,0,0,0,202,255,255,255,52,0,0,0,44,0,0,0,255,255,255,255,
      2,0,0,0,52,0,0,0,47,0,0,0,44,0,0,0,201,255,255,255,53,0,0,0,45,0,0,0,
      255,255,255,255,3,0,0,0,53,0,0,0,44,0,0,0,45,0,0,0,200,255,255,255,54,
      0,0,0,46,0,0,0,255,255,255,255,5,0,0,0,54,0,0,0,45,0,0,0,46,0,0,0,203,
      255,255,255,55,0,0,0,47,0,0,0,255,255,255,255,7,0,0,0,55,0,0,0,46,0,0,
      0,47,0,0,0>>.

verts() -> 
    <<0,0,128,191,0,0,128,63,0,0,128,63,0,0,192,65,0,0,128,191,0,0,128,191,0,
      0,128,63,0,0,192,65,0,0,128,191,0,0,128,63,205,204,140,63,0,0,128,65,0,
      0,128,191,0,0,128,191,205,204,140,63,0,0,128,65,0,0,128,63,0,0,128,191,
      0,0,128,63,0,0,192,65,0,0,128,63,0,0,128,191,205,204,140,63,0,0,128,65,
      0,0,128,63,0,0,128,63,0,0,128,63,0,0,192,65,0,0,128,63,0,0,128,63,205,
      204,140,63,0,0,128,65,205,204,140,63,0,0,128,63,0,0,128,63,0,0,128,65,
      205,204,140,63,0,0,128,191,0,0,128,63,0,0,128,65,0,0,128,63,0,0,128,191,
      0,0,128,191,0,0,192,65,205,204,140,63,0,0,128,191,0,0,128,191,0,0,128,
      65,0,0,128,63,0,0,128,63,0,0,128,191,0,0,192,65,205,204,140,63,0,0,128,
      63,0,0,128,191,0,0,128,65,0,0,128,63,205,204,140,191,0,0,128,63,0,0,128,
      65,0,0,128,191,205,204,140,191,0,0,128,63,0,0,128,65,0,0,128,191,0,0,
      128,191,0,0,128,191,0,0,192,65,0,0,128,191,205,204,140,191,0,0,128,191,
      0,0,128,65,0,0,128,63,205,204,140,191,0,0,128,191,0,0,128,65,0,0,128,
      191,205,204,140,63,0,0,128,63,0,0,128,65,0,0,128,63,205,204,140,63,0,0,
      128,63,0,0,128,65,0,0,128,63,205,204,140,63,0,0,128,191,0,0,128,65,0,0,
      128,191,0,0,128,63,0,0,128,191,0,0,192,65,0,0,128,191,205,204,140,63,0,
      0,128,191,0,0,128,65,0,0,128,191,0,0,128,191,205,204,140,191,0,0,128,65,
      0,0,128,191,0,0,128,63,205,204,140,191,0,0,128,65,0,0,128,63,0,0,128,63,
      205,204,140,191,0,0,128,65,0,0,128,63,0,0,128,191,205,204,140,191,0,0,
      128,65,205,204,140,191,0,0,128,191,0,0,128,63,0,0,128,65,205,204,140,
      191,0,0,128,63,0,0,128,63,0,0,128,65,205,204,140,191,0,0,128,63,0,0,128,
      191,0,0,128,65,205,204,140,191,0,0,128,191,0,0,128,191,0,0,128,65,51,51,
      163,192,0,0,128,191,0,0,128,63,0,0,96,65,51,51,163,192,0,0,128,63,0,0,
      128,63,0,0,96,65,51,51,163,192,0,0,128,63,0,0,128,191,0,0,96,65,51,51,
      163,192,0,0,128,191,0,0,128,191,0,0,96,65,0,0,128,191,0,0,128,191,51,
      51,163,192,0,0,96,65,0,0,128,191,0,0,128,63,51,51,163,192,0,0,96,65,0,
      0,128,63,0,0,128,63,51,51,163,192,0,0,96,65,0,0,128,63,0,0,128,191,51,
      51,163,192,0,0,96,65,0,0,128,191,51,51,163,64,0,0,128,63,0,0,96,65,0,
      0,128,63,51,51,163,64,0,0,128,63,0,0,96,65,0,0,128,63,51,51,163,64,0,
      0,128,191,0,0,96,65,0,0,128,191,51,51,163,64,0,0,128,191,0,0,96,65,0,
      0,128,63,51,51,163,192,0,0,128,63,0,0,96,65,0,0,128,191,51,51,163,192,
      0,0,128,63,0,0,96,65,0,0,128,191,51,51,163,192,0,0,128,191,0,0,96,65,
      0,0,128,63,51,51,163,192,0,0,128,191,0,0,96,65,51,51,163,64,0,0,128,
      63,0,0,128,63,0,0,96,65,51,51,163,64,0,0,128,191,0,0,128,63,0,0,96,
      65,51,51,163,64,0,0,128,191,0,0,128,191,0,0,96,65,51,51,163,64,0,0,
      128,63,0,0,128,191,0,0,96,65,0,0,128,191,0,0,128,63,51,51,163,64,0,
      0,96,65,0,0,128,191,0,0,128,191,51,51,163,64,0,0,96,65,0,0,128,63,
      0,0,128,191,51,51,163,64,0,0,96,65,0,0,128,63,0,0,128,63,51,51,163,
      64,0,0,96,65>>.
