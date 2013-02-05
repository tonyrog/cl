
all:
	(cd c_src; make $@)
	(cd src; make $@)

doc:
	(cd src; make edoc)

clean:
	(cd c_src; make $@)
	(cd src; make $@)
