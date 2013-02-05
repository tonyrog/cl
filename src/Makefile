
MODULES = \
	cl  \
	cl10 \
	cl11 \
	cl12 \
	cl13 \
	clu

EBIN = ../ebin
ERLC = erlc

include ../vsn.mk
VSN=$(CL_VSN)

APP_NAME= cl
APP_FILE= $(APP_NAME).app
APP_SRC= $(APP_FILE).src
APP_TARGET= $(EBIN)/$(APP_FILE)

override ERLC_FLAGS = 


OBJS = $(MODULES:%=$(EBIN)/%.beam)

TARGET_FILES = $(OBJS) $(APP_TARGET)

debug: ERLC_FLAGS += -Ddebug 

all: $(TARGET_FILES)

debug: all

release: all

dialyze:
	dialyzer --src -o dia.out $(ERLC_FLAGS) -c $(MODULES:%=%.erl) 

edoc:
	erl -noshell -pa doc -run edoc_run application "cl" '".."' '[{private,true},no_packages,{sort_functions,false}]'
##erl -noshell -run edoc_run application "'$(APP_NAME)'"

clean:
	rm -f $(OBJS)

$(APP_TARGET): $(APP_SRC) ../vsn.mk
	sed -e 's;git;\"$(VSN)\";'  $< > $@

cl.erl:	../include/cl.hrl


$(EBIN)/%.beam:	%.erl
	$(ERLC) $(ERLC_FLAGS) -o $(EBIN) $<
