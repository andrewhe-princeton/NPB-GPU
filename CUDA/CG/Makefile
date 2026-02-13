SHELL=/bin/sh
BENCHMARK=cg
BENCHMARKU=CG

include ../config/make.def

OBJS = cg.o ${COMMON}/c_print_results.o  \
       ${COMMON}/c_${RAND}.o ${COMMON}/c_timers.o ${COMMON}/c_wtime.o

include ../sys/make.common

${PROGRAM}: config ${OBJS}
	${CLINK} ${CLINKFLAGS} -o ${PROGRAM} ${OBJS} ${C_LIB}

cg.o:             cg.cu npbparams.hpp
	${CCOMPILE} cg.cu

# Define these before including generic makefile
BMARK=cg
TEST_ARGS=1 10 10 10 10 # Placeholder
PERF_ARGS=0 2048 2048 2048 2048 # Placeholder

include ../Makefile.generic

# Explicitly ensure .ll targets depend on npbparams.hpp if generic rules are used
cg.ll: npbparams.hpp
${BMARK}-cuda-nvptx64-nvidia-cuda-sm_20.ll: npbparams.hpp

clean: clean_llm
	- rm -f *.o *~ 
	- rm -f npbparams.hpp core
