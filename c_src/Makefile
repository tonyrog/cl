OSNAME := $(shell uname -s)
MACHINE := $(shell uname -m)
CC = gcc
OUT_C = -o
OUT_L = -o
OBJ = o

## make MCL=1  to build with microsoft compiler using msbuild (MSVC-10)

ERLDIR	:= $(shell erl -noshell -eval "io:format([126,115,126,110],[code:root_dir()])" -s erlang halt)

WORDSIZE = $(shell erl -noshell -eval "io:format([126,119,126,110],[erlang:system_info(wordsize)*8])" -s erlang halt)

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

EXT       = so
PRIVDIR=../priv

ifeq ($(TYPE), debug)
CFLAGS += -Ddebug -DDEBUG -g -Wall -Wextra -Wswitch-default -Wswitch-enum -D_THREAD_SAFE
CFLAGS += -D_REENTRANT -fno-common -I$(ERL_C_INCLUDE_DIR)
WIN_DEBUG = -Ddebug -DDEBUG
endif

ifeq ($(TYPE), release)
CFLAGS += -Wall -Wextra -Wswitch-default -Wswitch-enum -D_THREAD_SAFE -D_REENTRANT -fno-common
CFLAGS += -I$(ERL_C_INCLUDE_DIR)
endif

ifeq ($(OSNAME), Linux)
LINUX = Yes
CFLAGS += -I/usr/include/nvidia-current
CFLAGS += -I/opt/AMDAPP/include
ifeq ($(WORDSIZE), 32)
CFLAGS += -O3 -fPIC -m32
endif
ifeq ($(WORDSIZE), 64)
CFLAGS += -O3 -fPIC -m64
endif
LD_SHARED	:= $(CC) -shared
LDFLAGS	        += -lOpenCL
endif

ifeq ($(OSNAME), Darwin)
MAC_OS_X = Yes
ifeq ($(WORDSIZE), 32)
CFLAGS += -O3 -fPIC -m32 -DDARWIN -no-cpp-precomp
LD_SHARED	:= $(CC) -m32 -bundle -flat_namespace -undefined suppress
endif
ifeq ($(WORDSIZE), 64)
CFLAGS += -O3 -fPIC -m64 -DDARWIN -no-cpp-precomp
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
CFLAGS += -shared -I$(OPENCL_DIR)/include -m32 -DWIN32
endif
ifeq ($(WORDSIZE), 64)
CFLAGS += -shared -I$(OPENCL_DIR)/include -m64 -DWIN32
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

ifeq ($(MAC_OS_X),Yes)
	LDFLAGS += -framework OpenCL
endif

ifeq ($(WIN32_GCC),Yes)
	CFLAGS += -DWIN32
	LDFLAGS += -Wl--enable-stdcall-fixup -L$(OPENCL_DIR)/lib -lOpenCL
endif

CL_NIF = $(PRIVDIR)/cl_nif.$(EXT)

CL_NIF_OBJS = \
	cl_nif.$(OBJ) \
	cl_hash.$(OBJ)

CL_NIF_SRC = \
	cl_nif.c \
	cl_hash.c

all:
	$(MAKE) nif TYPE=release

debug:
	$(MAKE) nif TYPE=debug

clean:
	rm -f $(CL_NIF_OBJS)
	rm -f $(CL_NIF)

release:
	$(MAKE) nif TYPE=release

nif: $(CL_NIF)


ifeq ($(WIN32_CL),No)
clean_internal:
	-rm -f *.$(OBJ)
	-rm -f $(PRIVDIR)/*.$(EXT)

$(CL_NIF): $(OCL_LIB) $(CL_NIF_OBJS)
	$(LD_SHARED) $(OUT_L) $@ $(CL_NIF_OBJS) $(LDFLAGS)
else
clean_internal:
	msbuild.exe CL.sln -t:clean

$(CL_NIF): $(CL_NIF_SRC) cl_hash.h
	msbuild.exe CL.sln -p:Configuration=$(TYPE) -p:ERL_C_INCLUDE_DIR=$(ERL_C_INCLUDE_DIR) \
	-p:OPENCL_DIR=$(OPENCL_DIR)
endif

cl_nif.$(OBJ): cl_hash.h

%.$(OBJ): %.c
	$(CC) -c $(OUT_C) $@ $(CFLAGS) $<
