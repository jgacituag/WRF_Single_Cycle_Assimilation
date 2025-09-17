# Usage: make && make install
PYTHON ?= python
F2PY   := $(PYTHON) -m numpy.f2py
FC     ?= gfortran
FFLAGS ?= -O3 -fopenmp

MOD    := cletkf_wloc
SRC    := common_da_wloc.f90 common_letkf.f90 common_mtx.f90 common_tools.f90

all: $(MOD).so

$(MOD).so: $(SRC)
	$(F2PY) -c -m $(MOD) $(SRC) --fcompiler=gnu95 --opt='-O3' --f90flags='$(FFLAGS)' -lgomp

install: $(MOD).so
	mkdir -p ../
	mv -v $(MOD)*.so ../

clean:
	rm -f $(MOD)*.so *.o *.mod