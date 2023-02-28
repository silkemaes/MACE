# SOURCES1=$(wildcard src/*.f)
# SOURCES2=$(filter-out scr/rate13main.f, $(SOURCES1))
# SOURCES3=$(filter-out scr/python_main.f, $(SOURCES2))
# SOURCES=$(filter-out scr/python_bindingsmain.f, $(SOURCES3))

# OBJECTS=$(patsubst scr/%.f, bin/%.o, $(SOURCES))
# OBJS=$(filter-out bin/*main.o, $(patsubst %.f, %.o, $(SOURCES)))


# 0Dmodel: src/rate13main.f $(SOURCES) 
# 	echo $(SOURCES)
# 	gfortran -w -o 0Dmodel src/rate13main.f $(SOURCES) -fbounds-check -fno-align-commons  -Wimplicit-procedure

# 0Dmodel-bindings: $(OBJECTS)
# 	gfortran -o chemlib.so python_main.o $(OBJ) -fbounds-check -fno-align-commons  -Wimplicit-procedure -shared -fPIC


# # $(OBJECTS): $(SOURCES)
# #	gfortran -c -O2 $(SOURCES) -fbounds-check -fno-align-commons -fPIC

# # dcodes: dcodes.f
# # 	gfortran -c -O2 dcodes.f -fbounds-check -fno-align-commons -fPIC
# # freeze: freeze.f
# # 	gfortran -c -O2 freeze.f -fbounds-check -fno-align-commons -fPIC
# # dvode: dvode.f
# # 	gfortran -c -O2 dvode.f -fbounds-check -fno-align-commons -fPIC
# # subs: subs.f
# # 	gfortran -c -O2 subs.f -fbounds-check -fno-align-commons -fPIC
# # dcrates: dcrates.f
# # 	gfortran -c -O2 dcrates.f -fbounds-check -fno-align-commons -fPIC
# # dcanalyse: dcanalyse.f
# # 	gfortran -c -O2 dcanalyse.f -fbounds-check -fno-align-commons -fPIC


# build:
# 	mkdir bin


#------------
# gfortran -c -O2 src/dcodes.f -fbounds-check -fno-align-commons -fPIC
# gfortran -c -O2 src/freeze.f -fbounds-check -fno-align-commons -fPIC
# gfortran -c -O2 src/dvode.f -fbounds-check -fno-align-commons -fPIC
# gfortran -c -O2 src/subs.f -fbounds-check -fno-align-commons -fPIC
# gfortran -c -O2 src/dcrates.f -fbounds-check -fno-align-commons -fPIC
# gfortran -c -O2 src/dcanalyse.f -fbounds-check -fno-align-commons -fPIC


# # # Compile regular main program
# # # ----------------------------

# gfortran -c -O2 src/rate13main.f -fbounds-check -fno-align-commons
# gfortran -o csmodel rate13main.o dcrates.o freeze.o dvode.o subs.o dcodes.o dcanalyse.o -fbounds-check -fno-align-commons  -Wimplicit-procedure


# # # Compile the python c-bindings
# # # -----------------------------
# # # Note: also added -fPIC to the compile flags above.

gfortran -c -O2 src/python_main.f -fbounds-check -fno-align-commons -fPIC
gfortran -o chemlib.so python_main.o dcrates.o freeze.o dvode.o subs.o dcodes.o dcanalyse.o -fbounds-check -fno-align-commons  -Wimplicit-procedure -shared -fPIC

# gfortran -c -O2 src/python_bindings.f -fbounds-check -fno-align-commons -fPIC
# gfortran -o chemlib.so python_bindings.o dcrates.o freeze.o dvode.o subs.o dcodes.o dcanalyse.o -fbounds-check -fno-align-commons  -Wimplicit-procedure -shared -fPIC



# gfortran-6 -std=legacy -ffixed-form  -w -O3 -o csmodel csmain.f acodes.f csrates.f dvode.f cssubs.f csanalyse.f 
# gfortran -std=legacy -ffixed-form  -w -O3 -o csmodel csmain.f acodes.f csrates.f dvode.f cssubs.f csanalyse.f 
