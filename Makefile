
all:
	(cd c_src; make)
	(cd src; make)

doc:
	(cd src; make edoc)

