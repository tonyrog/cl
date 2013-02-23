OSNAME := $(shell uname -s)
MACHINE := $(shell uname -m)
CC = gcc
OUT_C = -o
OUT_L = -o
OBJ = o

## make MCL=1  to build with microsoft compiler using msbuild (MSVC-10)

ERLDIR	:= $(shell erl -noshell -eval "io:format([126,115,126,110],[code:root_dir()])" -s erlang halt)
ERL_C_INCLUDE_DIR := $(ERLDIR)/usr/include
OCL_DIR := $(shell cd ..; pwd)

## Windows start
ifneq (, $(findstring MINGW32,$(OSNAME)))
MINGW = Yes
endif

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

ifneq ($(MINGW), Yes)
CONFIG_FILE_64 = config.64.build
CONFIG_FILE_32 = config.32.build
ifeq ($(WORDSIZE), 64)
include $(CONFIG_FILE_64)
endif
ifeq ($(WORDSIZE), 32)
include $(CONFIG_FILE_32)
endif
else
# Skip configure on windows for now we don't have
# 64bits support anyway, autoconf and lib are a mess
# to install
ifeq ($(WORDSIZE), 64)
   CL_BUILD=false
else
   CL_BUILD=true
endif
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
CFLAGS += -I/usr/include/nvidia-current
CFLAGS += -I/opt/AMDAPP/include
ifeq ($(WORDSIZE), 32)
CFLAGS += -O3 -fPIC -m32 -DWORDSIZE=32
endif
ifeq ($(WORDSIZE), 64)
CFLAGS += -O3 -fPIC -m64 -DWORDSIZE=64
endif
LD_SHARED	:= $(CC) -shared
LDFLAGS	        += -lOpenCL
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

ifeq ($(MINGW), Yes)
EXT = dll

ifneq ($(MCL), )
## Use Microsoft CL
WIN32_CL = Yes
ifeq ($(OPENCL_DIR), )
   OPENCL_DIR = c:/msysgit/opt/local/
endif
else
## Use mingw-gcc
WIN32_GCC = Yes
CFLAGS += -D__WIN32__
ifeq ($(OPENCL_DIR), )
   OPENCL_DIR = /opt/local/
endif

ifeq ($(WORDSIZE), 32)
CFLAGS += -shared -I$(OPENCL_DIR)/include -m32 -DWIN32 -DWORDSIZE=32
endif
ifeq ($(WORDSIZE), 64)
CFLAGS += -shared -I$(OPENCL_DIR)/include -m64 -DWIN32 -DWORDSIZE=64
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
	LDFLAGS += -Wl--enable-stdcall-fixup -L$(OPENCL_DIR)/lib -lOpenCL
endif

CL_DRV = $(LIBDIR)/$(WORDSIZE)/cl_drv.$(EXT)

CL_NIF = $(LIBDIR)/$(WORDSIZE)/cl_nif.$(EXT)

CL_DRV_OBJS = \
	$(OBJDIR)/$(WORDSIZE)/cl_drv.$(OBJ) \
	$(OBJDIR)/$(WORDSIZE)/cl_hash.$(OBJ)

CL_NIF_OBJS = \
	$(OBJDIR)/$(WORDSIZE)/cl_nif.$(OBJ) \
	$(OBJDIR)/$(WORDSIZE)/cl_hash.$(OBJ)

CL_NIF_SRC = \
	cl_nif.c \
	cl_hash.c

CL_ASM = \
	$(OBJDIR)/$(WORDSIZE)/cl_drv.s \
	$(OBJDIR)/$(WORDSIZE)/cl_hash.s

OCL_SIMD_OBJS = $(SIMD_OBJS:%=$(OBJDIR)/%)

all:	all32 all64

all32:	$(CONFIG_FILE_32)
	$(MAKE) nif TYPE=release WORDSIZE=32
	$(MAKE) nif TYPE=debug   WORDSIZE=32

all64:	$(CONFIG_FILE_64)
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

ifeq ($(WIN32_CL),No) 
clean_internal:
	-rm -f $(OBJDIR)/$(WORDSIZE)/*.$(OBJ)
	-rm -f $(LIBDIR)/$(WORDSIZE)/*.$(EXT)

$(CL_NIF): $(OCL_LIB) $(CL_NIF_OBJS) # $(CL_ASM)
	$(LD_SHARED) $(OUT_L) $@ $(CL_NIF_OBJS) $(LDFLAGS)
else
clean_internal:
	msbuild.exe CL.sln -t:clean

$(CL_NIF): $(CL_NIF_SRC) cl_hash.h
	msbuild.exe CL.sln -p:Configuration=$(TYPE) -p:ERL_C_INCLUDE_DIR=$(ERL_C_INCLUDE_DIR) \
	-p:OPENCL_DIR=$(OPENCL_DIR)
endif

$(OBJDIR)/$(WORDSIZE)/cl_nif.$(OBJ): cl_hash.h

$(OBJDIR)/$(WORDSIZE)/%.$(OBJ): %.c
	$(CC) -c $(OUT_C) $@ $(CFLAGS) $<

$(OBJDIR)/$(WORDSIZE)/%.s: %.c
	$(CC) -S $(OUT_C) $@ $(CFLAGS) $<

