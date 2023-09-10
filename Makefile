#@BEGIN-DIR-DEFAULT-RULES@
all:
	@if [ -d "src" -a -f "src/Makefile" ]; then (cd src && $(MAKE) all); fi
	@if [ -d "c_src" -a -f "c_src/Makefile" ]; then (cd c_src && $(MAKE) all); fi
	@if [ -d "test" -a -f "test/Makefile" ]; then (cd test && $(MAKE) all); fi

clean:
	@if [ -d "src" -a -f "src/Makefile" ]; then (cd src && $(MAKE) clean); fi
	@if [ -d "c_src" -a -f "c_src/Makefile" ]; then (cd c_src && $(MAKE) clean); fi
	@if [ -d "test" -a -f "test/Makefile" ]; then (cd test && $(MAKE) clean); fi
#@END-DIR-DEFAULT-RULES@
