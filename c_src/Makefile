OSNAME := $(shell uname -s)
MACHINE := $(shell uname -m)
CC = gcc
OUT_C = -o
OUT_L = -o
OBJ = o

ERLDIR	:= $(shell erl -noshell -eval "io:format([126,115,126,110],[code:root_dir()])" -s erlang halt)
ERL_C_INCLUDE_DIR := $(ERLDIR)/usr/include
OCL_DIR := $(shell cd ..; pwd)

MAC_OS_X  = No
WIN32_GCC = No
WIN32_CL  = No
LINUX     = No

ALTIVEC   = No
MMX       = No
SSE       = No
SSE2      = No
SSE3      = No
SSE4      = No
EXT       = so

ifeq ($(WORDSIZE), 64)
include config.64.build
endif
ifeq ($(WORDSIZE), 32)
include config.32.build
endif

ifeq ($(TYPE), debug)
OBJDIR=../obj/debug
LIBDIR=../lib/debug
CFLAGS += -Ddebug -DDEBUG -g -Wall -Wextra -Wswitch-default -Wswitch-enum -D_THREAD_SAFE 
CFLAGS += -D_REENTRANT -fno-common -I$(ERL_C_INCLUDE_DIR)
WIN_DEBUG = -Ddebug -DDEBUG
endif

ifeq ($(TYPE), release)
OBJDIR=../obj/release
LIBDIR=../lib/release
CFLAGS += -Wall -Wextra -Wswitch-default -Wswitch-enum -D_THREAD_SAFE -D_REENTRANT -fno-common
CFLAGS += -I$(ERL_C_INCLUDE_DIR)
endif

ifeq ($(OSNAME), Linux)
LINUX = Yes
ifeq ($(WORDSIZE), 32)
CFLAGS += -O3 -fPIC -m32 -DWORDSIZE=32
endif
ifeq ($(WORDSIZE), 64)
CFLAGS += -O3 -fPIC -m64 -DWORDSIZE=64
endif
LD_SHARED	:= $(CC) -shared -lOpenCL
endif

ifeq ($(OSNAME), Darwin)
MAC_OS_X = Yes
ifeq ($(WORDSIZE), 32)
CFLAGS += -O3 -fPIC -m32 -DWORDSIZE=32 -DDARWIN -no-cpp-precomp
LD_SHARED	:= $(CC) -m32 -bundle -flat_namespace -undefined suppress
endif
ifeq ($(WORDSIZE), 64)
CFLAGS += -O3 -fPIC -m64 -DWORDSIZE=64 -DDARWIN -no-cpp-precomp
LD_SHARED	:= $(CC) -m64 -bundle -flat_namespace -undefined suppress
endif
endif

## Windows start
ifeq ($(OSNAME), MINGW32_NT-6.0)
MINGW = Yes
endif
ifeq ($(OSNAME), MINGW32_NT-6.1)
MINGW = Yes
endif

ifeq ($(MINGW), Yes)
EXT = dll
MCL = $(shell which cl.exe)

#ifneq ($(MCL), )
ifeq ($(MCL), buggy)
## Use Microsoft CL
WIN32_CL = Yes
CC = cl.exe 
OUT_C = -Fo
OUT_L = -out:
LD_SHARED := link.exe -DLL 
OPENCL_DIR = c:/msysgit/opt/local/
else
## Use mingw-gcc
WIN32_GCC = Yes
CFLAGS += -D__WIN32__
ifeq ($(WORDSIZE), 32)
CFLAGS += -shared -I/opt/local/include -m32 -DWIN32 -DWORDSIZE=32
endif
ifeq ($(WORDSIZE), 64)
CFLAGS += -shared -I/opt/local/include -m64 -DWIN32 -DWORDSIZE=64
endif

LD_SHARED	:= $(CC) -shared 

## Optimizations is broken on mingw 4.4.0 (it crashes with it on)
GCC_VERSION = $(shell gcc -dumpversion)
ifneq ($(GCC_VERSION), 4.4.0)
CFLAGS += -O3
endif

endif
endif
## Windows end

ifeq ($(MACHINE), "Power Macintosh")
ALTIVEC   = Yes
endif

ifeq ($(MACHINE), i386)
MMX       = No
SSE       = No
SSE2      = Yes
SSE3      = No
endif

ifeq ($(MACHINE), i686)
MMX       = No
SSE       = No
SSE2      = Yes
SSE3      = No
endif

ifeq ($(ALTIVEC),Yes)
	CFLAGS += -maltivec -DUSE_ALTIVEC
endif

ifeq ($(SSE),Yes)
	CFLAGS += -msse
endif

ifeq ($(SSE2),Yes)
	CFLAGS += -msse2 -DUSE_SSE2
	SIMD_OBJS += ocl_simd_sse2.$(BIT).$(OBJ)
endif

ifeq ($(SSE3),Yes)
	MCFLAGS += -msse3
endif

ifeq ($(MMX),Yes)
	MCFLAGS += -mmmx -DUSE_MMX
	SIMD_OBJS += ocl_simd_mmx.$(BIT).$(OBJ)
endif


ifeq ($(MAC_OS_X),Yes)
	LDFLAGS += -framework OpenCL
endif

ifeq ($(WIN32_GCC),Yes)
	CFLAGS += -DWIN32
	LDFLAGS += -Wl--enable-stdcall-fixup -L/opt/local/lib -lOpenCL
endif

ifeq ($(WIN32_CL),Yes)
	CFLAGS = -I$(OPENCL_DIR)/include -I$(ERL_C_INCLUDE_DIR) -MD -D_THREAD_SAFE -DWIN32 -D__WIN32__ $(WIN_DEBUG)
	LDFLAGS = $(OPENCL_DIR)/lib/OpenCL.lib
endif

CL_DRV = $(LIBDIR)/$(WORDSIZE)/cl_drv.$(EXT)

CL_NIF = $(LIBDIR)/$(WORDSIZE)/cl_nif.$(EXT)

CL_DRV_OBJS = \
	$(OBJDIR)/$(WORDSIZE)/cl_drv.$(OBJ) \
	$(OBJDIR)/$(WORDSIZE)/cl_hash.$(OBJ)

CL_NIF_OBJS = \
	$(OBJDIR)/$(WORDSIZE)/cl_nif.$(OBJ) \
	$(OBJDIR)/$(WORDSIZE)/cl_hash.$(OBJ)

CL_ASM = \
	$(OBJDIR)/$(WORDSIZE)/cl_drv.s \
	$(OBJDIR)/$(WORDSIZE)/cl_hash.s

OCL_SIMD_OBJS = $(SIMD_OBJS:%=$(OBJDIR)/%)

all:	all32 all64

all32:	config.32.build
	$(MAKE) nif TYPE=release WORDSIZE=32
	$(MAKE) nif TYPE=debug   WORDSIZE=32

all64:	config.64.build
	$(MAKE) nif TYPE=release WORDSIZE=64
	$(MAKE) nif TYPE=debug   WORDSIZE=64

config.32.build:	configure
	./configure --with-wordsize=32

config.64.build:	configure
	./configure --with-wordsize=64

configure:	configure.in
	autoheader; autoconf

clean: 
	$(MAKE) clean_internal TYPE=release WORDSIZE=32
	$(MAKE) clean_internal TYPE=debug   WORDSIZE=32
	$(MAKE) clean_internal TYPE=release WORDSIZE=64
	$(MAKE) clean_internal TYPE=debug   WORDSIZE=64

# Make cl library
debug:
	$(MAKE) nif TYPE=debug

clean_debug:
	$(MAKE) clean_internal TYPE=debug

release:
	$(MAKE) nif TYPE=release

ifeq ($(CL_BUILD), false)
nif: 
	@echo 
	@echo Skipping nif for wordsize=$(WORDSIZE), not supported on this system
	@echo 
else
nif: $(CL_NIF)
endif

depend:

clean_internal:
	-rm -f $(OBJDIR)/$(WORDSIZE)/*.$(OBJ)

$(CL_NIF): $(OCL_LIB) $(CL_NIF_OBJS) # $(CL_ASM)
	$(LD_SHARED) $(OUT_L)$@ $(CL_NIF_OBJS) $(LDFLAGS)

$(OBJDIR)/$(WORDSIZE)/cl_nif.$(OBJ): cl_hash.h

$(OBJDIR)/$(WORDSIZE)/%.$(OBJ): %.c
	$(CC) -c $(OUT_C)$@ $(CFLAGS) $<

$(OBJDIR)/$(WORDSIZE)/%.s: %.c
	$(CC) -S $(OUT_C)$@ $(CFLAGS) $<

