#@BEGIN-ERL_SRC-DEFAULT-RULES@
ERLC="$(shell which erlc)"
ERLC_FLAGS=-MMD -MP -MF .$<.d -I ../.. +debug_info
YRL_SRC=$(wildcard *.yrl)
XRL_SRC=$(wildcard *.xrl)
ERL_SOURCES=$(wildcard *.erl) $(YRL_SRC:%.yrl=%.erl) $(XRL_SRC:%.xrl=%.erl)
ERL_OBJECTS=$(ERL_SOURCES:%.erl=../ebin/%.beam)
ALL_OBJECTS=$(ERL_OBJECTS)

.PRECIOUS: $(YRL_SRC:%.yrl=%.erl) $(XRL_SRC:%.xrl=%.erl)

all: $(ALL_OBJECTS)

clean:
	rm -f $(ALL_OBJECTS) *.core .*.d

../ebin/%.beam: %.erl
	$(ERLC) $(ERLC_FLAGS) -o ../ebin $<

%.erl: %.yrl
	$(ERLC) $<

%.erl: %.xrl
	$(ERLC) $<

.%.d: ;

-include .*.d
#@END-ERL_SRC-DEFAULT-RULES@
