# Makefile for the Fortran-Python bindings of the 0D chemistry model

# Specify the source files
SOURCES = $(wildcard src/*.f)

# Specify the fortran compiler
COMPILER = gfortran

# Specify the compiler flags
FLAGS = -fmax-errors=3 -w -fbounds-check -fno-align-commons -shared -fPIC

0Dmodel-bindings: $(SOURCES) | bin
	# Compile the Fortran into a binary file
	$(COMPILER) -o bin/chemlib.so $(SOURCES) $(FLAGS)
    
bin:
	# Create a bin directory for the binary files
	mkdir -p $@