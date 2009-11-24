
all:
	(cd c_src; make $@)
	(cd src; make $@)

all32:
	(cd c_src; make $@)
	(cd src; make $@)

doc:
	(cd src; make edoc)

