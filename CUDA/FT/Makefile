SHELL=/bin/sh
BENCHMARK=ft
BENCHMARKU=FT

include ../config/make.def

OBJS = ft.o ${COMMON}/c_${RAND}.o ${COMMON}/c_print_results.o \
       ${COMMON}/c_timers.o ${COMMON}/c_wtime.o

include ../sys/make.common

${PROGRAM}: config ${OBJS}
	${CLINK} ${CLINKFLAGS} -o ${PROGRAM} ${OBJS} ${C_LIB}

ft.o:             ft.cu npbparams.hpp
	${CCOMPILE} ft.cu

# Define these before including generic makefile
BMARK=ft
TEST_ARGS=1 10 10 10 10 # Placeholder
PERF_ARGS=0 2048 2048 2048 2048 # Placeholder

include ../Makefile.generic

# Explicitly ensure .ll targets depend on npbparams.hpp if generic rules are used
ft.ll: npbparams.hpp
${BMARK}-cuda-nvptx64-nvidia-cuda-sm_20.ll: npbparams.hpp

clean: clean_llm
	- rm -f *.o *~ mputil*
	- rm -f ft npbparams.hpp core
