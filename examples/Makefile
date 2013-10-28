
MODULES = \
	cl_basic \
	cl_square_float \
	cl_map \
	cl_binary_test \
	cl_bandwidth \
	cl_test \
	cl_mul



EBIN = ../ebin
ERLC = erlc

override ERLC_FLAGS = -W -pa ../../cl/ebin

OBJS = $(MODULES:%=$(EBIN)/%.beam)

TARGET_FILES = $(OBJS)

debug: ERLC_FLAGS += -Ddebug

all: $(TARGET_FILES)

debug: all

release: all

depend:
	edep -MM -o ../ebin $(ERLC_FLAGS) $(MODULES:%=%.erl) > depend.mk

dialyze:
	dialyzer --src -o dia.out -pa ../ebin -c $(MODULES:%=%.erl)

clean:
	rm -f $(OBJS)

-include depend.mk

$(EBIN)/%.beam:	%.erl
	$(ERLC) $(ERLC_FLAGS) -o $(EBIN) $<
