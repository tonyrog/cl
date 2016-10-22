#
# Copyright (C) 2016, Rogvall Invest AB, <tony@rogvall.se>
#
# This software is licensed as described in the file COPYRIGHT, which
# you should have received as part of this distribution. The terms
# are also available at http://www.rogvall.se/docs/copyright.txt.
#
# You may opt to use, copy, modify, merge, publish, distribute and/or sell
# copies of the Software, and permit persons to whom the Software is
# furnished to do so, under the terms of the COPYRIGHT file.
#
# This software is distributed on an "AS IS" basis, WITHOUT WARRANTY OF ANY
# KIND, either express or implied.
#

OSNAME := $(shell uname -s)
MACHINE := $(shell uname -m)
OUT_C = -o
OUT_L = -o
OBJ = o

## us NOCL=true make to force build with gcc on windows

WORDSIZE = $(shell erl -noshell -eval "io:format([126,119,126,110],[erlang:system_info(wordsize)*8])" -s erlang halt)

ifeq ($(ERLANG_ROOT_DIR), )
  ERLDIR := $(shell erl -noshell -eval "io:format([126,115,126,110],[code:root_dir()])" -s erlang halt)
  ERL_C_INCLUDE_DIR := "$(ERLDIR)/usr/include"
else
  ERL_C_INCLUDE_DIR := "$(ERLANG_ROOT_DIR)/usr/include"
endif

OCL_DIR := $(shell cd ..; pwd)

ifneq (, $(findstring MINGW,$(OSNAME)))
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
CFLAGS += -Wno-deprecated-declarations -Wno-missing-field-initializers -I$(ERL_C_INCLUDE_DIR)
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
LDFLAGS += -framework OpenCL
endif

############### WINDOWS HACK
ifeq ($(MINGW), Yes)
EXT = dll
CC_OR_NOCL=$(findstring which:,$(shell which cl.exe))$(NOCL)
ifeq (, $(CC_OR_NOCL))
## Use Microsoft CL
	WIN32_CL = Yes
	CC=cl.exe
	MS2C = MSYS2_ARG_CONV_EXCL=*
	OUT_C = /Fo
	ifeq ($(OPENCL_DIR), )
	   OPENCL_DIR = c:/msys64/opt/local/
	endif
	CFLAGS = /FS /Zi /nologo /W1 -DWIN32 -D__WIN32__
	CFLAGS += /I$(OPENCL_DIR)/include /I$(ERL_C_INCLUDE_DIR)
	LD_SHARED=link.exe /DLL
	OUT_L=/OUT:
	ifeq ($(WORDSIZE), 32)
	   LDFLAGS += /NOLOGO $(OPENCL_DIR)/lib/x86/OpenCL.lib
	else
	   CFLAGS  += -DWIN_X64
	   LDFLAGS += /NOLOGO $(OPENCL_DIR)/lib/x64/OpenCL.lib
	endif
else
############## Use mingw-gcc
	CC=gcc
	WIN32_GCC = Yes
	CFLAGS += -D__WIN32__
	ifeq ($(OPENCL_DIR), )
	   OPENCL_DIR = /opt/local/
	endif

	ifeq ($(WORDSIZE), 32)
	   CFLAGS += -shared -I$(OPENCL_DIR)/include -m32 -DWIN32
	   LDFLAGS += -L$(OPENCL_DIR)/lib/x86 -lOpenCL
	endif
	ifeq ($(WORDSIZE), 64)
	   CFLAGS += -shared -I$(OPENCL_DIR)/include -m64 -DWIN32
	   LDFLAGS += -L$(OPENCL_DIR)/lib/x64 -lOpenCL
	endif

	LD_SHARED	:= $(CC) -shared
## Optimizations is broken on mingw 4.4.0 (it crashes with it on)
	GCC_VERSION = $(shell gcc -dumpversion)
	ifneq ($(GCC_VERSION), 4.4.0)
	    CFLAGS += -O3
	endif
endif
endif
############### WINDOWS end

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

cl_nif.$(OBJ): cl_hash.h

clean_internal:
	-rm -f *.$(OBJ)
	-rm -f $(PRIVDIR)/*.$(EXT)

ifeq ($(WIN32_CL),No)

%.$(OBJ): %.c
	$(CC) -c $(OUT_C) $@ $(CFLAGS) $<

$(CL_NIF): $(OCL_LIB) $(CL_NIF_OBJS)
	@mkdir -p $(PRIVDIR)
	$(LD_SHARED) $(OUT_L) $@ $(CL_NIF_OBJS) $(LDFLAGS)
else

%.$(OBJ): %.c
	$(MS2C) $(CC) -c $(OUT_C)$@ $(CFLAGS) $<

$(CL_NIF): $(OCL_LIB) $(CL_NIF_OBJS)
	@mkdir -p $(PRIVDIR)
	$(MS2C) $(LD_SHARED) $(OUT_L)$@ $(CL_NIF_OBJS) $(LDFLAGS)

endif

