/* Provide Declarations */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifndef __cplusplus
typedef unsigned char bool;
#endif

#ifndef _MSC_VER
#define __forceinline __attribute__((always_inline)) inline
#endif

#if defined(__GNUC__)
#define  __ATTRIBUTELIST__(x) __attribute__(x)
#else
#define  __ATTRIBUTELIST__(x)  
#endif

#ifdef _MSC_VER  /* Can only support "linkonce" vars with GCC */
#define __attribute__(X)
#endif

static __forceinline int llvm_fcmp_une(double X, double Y) { return X != Y; }
static __forceinline int llvm_fcmp_ole(double X, double Y) { return X <= Y; }


/* Global Declarations */
/* Helper union for bitcasts */
typedef union {
  uint32_t Int32;
  uint64_t Int64;
  float Float;
  double Double;
} llvmBitCastUnion;

/* Types Declarations */
struct __FIXME__l_struct_struct_OC_dcomplex;
struct __FIXME__l_struct_struct_OC_cudaDeviceProp;
struct __FIXME__l_unnamed_2;
struct __FIXME__l_struct_struct_OC_dim3;
struct __FIXME__l_unnamed_1;

/* Function definitions */

/* Types Definitions */
struct __FIXME__l_struct_struct_OC_dcomplex {
  double __FIXME__l_struct_struct_OC_dcomplex_field0;
  double __FIXME__l_struct_struct_OC_dcomplex_field1;
};
struct __FIXME__l_array_256_uint8_t {
  uint8_t array[256];
};
struct __FIXME__l_array_3_uint32_t {
  uint32_t array[3];
};
struct __FIXME__l_array_2_uint32_t {
  uint32_t array[2];
};
struct __FIXME__l_struct_struct_OC_cudaDeviceProp {
  uint8_t __FIXME__l_struct_struct_OC_cudaDeviceProp_field0[256];
  uint64_t __FIXME__l_struct_struct_OC_cudaDeviceProp_field1;
  uint64_t __FIXME__l_struct_struct_OC_cudaDeviceProp_field2;
  uint32_t __FIXME__l_struct_struct_OC_cudaDeviceProp_field3;
  uint32_t __FIXME__l_struct_struct_OC_cudaDeviceProp_field4;
  uint64_t __FIXME__l_struct_struct_OC_cudaDeviceProp_field5;
  uint32_t __FIXME__l_struct_struct_OC_cudaDeviceProp_field6;
  uint32_t __FIXME__l_struct_struct_OC_cudaDeviceProp_field7[3];
  uint32_t __FIXME__l_struct_struct_OC_cudaDeviceProp_field8[3];
  uint32_t __FIXME__l_struct_struct_OC_cudaDeviceProp_field9;
  uint64_t __FIXME__l_struct_struct_OC_cudaDeviceProp_field10;
  uint32_t __FIXME__l_struct_struct_OC_cudaDeviceProp_field11;
  uint32_t __FIXME__l_struct_struct_OC_cudaDeviceProp_field12;
  uint64_t __FIXME__l_struct_struct_OC_cudaDeviceProp_field13;
  uint64_t __FIXME__l_struct_struct_OC_cudaDeviceProp_field14;
  uint32_t __FIXME__l_struct_struct_OC_cudaDeviceProp_field15;
  uint32_t __FIXME__l_struct_struct_OC_cudaDeviceProp_field16;
  uint32_t __FIXME__l_struct_struct_OC_cudaDeviceProp_field17;
  uint32_t __FIXME__l_struct_struct_OC_cudaDeviceProp_field18;
  uint32_t __FIXME__l_struct_struct_OC_cudaDeviceProp_field19;
  uint32_t __FIXME__l_struct_struct_OC_cudaDeviceProp_field20;
  uint32_t __FIXME__l_struct_struct_OC_cudaDeviceProp_field21;
  uint32_t __FIXME__l_struct_struct_OC_cudaDeviceProp_field22;
  uint32_t __FIXME__l_struct_struct_OC_cudaDeviceProp_field23;
  uint32_t __FIXME__l_struct_struct_OC_cudaDeviceProp_field24[2];
  uint32_t __FIXME__l_struct_struct_OC_cudaDeviceProp_field25[2];
  uint32_t __FIXME__l_struct_struct_OC_cudaDeviceProp_field26[3];
  uint32_t __FIXME__l_struct_struct_OC_cudaDeviceProp_field27[2];
  uint32_t __FIXME__l_struct_struct_OC_cudaDeviceProp_field28[3];
  uint32_t __FIXME__l_struct_struct_OC_cudaDeviceProp_field29[3];
  uint32_t __FIXME__l_struct_struct_OC_cudaDeviceProp_field30;
  uint32_t __FIXME__l_struct_struct_OC_cudaDeviceProp_field31[2];
  uint32_t __FIXME__l_struct_struct_OC_cudaDeviceProp_field32[3];
  uint32_t __FIXME__l_struct_struct_OC_cudaDeviceProp_field33[2];
  uint32_t __FIXME__l_struct_struct_OC_cudaDeviceProp_field34;
  uint32_t __FIXME__l_struct_struct_OC_cudaDeviceProp_field35[2];
  uint32_t __FIXME__l_struct_struct_OC_cudaDeviceProp_field36[3];
  uint32_t __FIXME__l_struct_struct_OC_cudaDeviceProp_field37[2];
  uint32_t __FIXME__l_struct_struct_OC_cudaDeviceProp_field38[3];
  uint32_t __FIXME__l_struct_struct_OC_cudaDeviceProp_field39;
  uint32_t __FIXME__l_struct_struct_OC_cudaDeviceProp_field40[2];
  uint64_t __FIXME__l_struct_struct_OC_cudaDeviceProp_field41;
  uint32_t __FIXME__l_struct_struct_OC_cudaDeviceProp_field42;
  uint32_t __FIXME__l_struct_struct_OC_cudaDeviceProp_field43;
  uint32_t __FIXME__l_struct_struct_OC_cudaDeviceProp_field44;
  uint32_t __FIXME__l_struct_struct_OC_cudaDeviceProp_field45;
  uint32_t __FIXME__l_struct_struct_OC_cudaDeviceProp_field46;
  uint32_t __FIXME__l_struct_struct_OC_cudaDeviceProp_field47;
  uint32_t __FIXME__l_struct_struct_OC_cudaDeviceProp_field48;
  uint32_t __FIXME__l_struct_struct_OC_cudaDeviceProp_field49;
  uint32_t __FIXME__l_struct_struct_OC_cudaDeviceProp_field50;
  uint32_t __FIXME__l_struct_struct_OC_cudaDeviceProp_field51;
  uint32_t __FIXME__l_struct_struct_OC_cudaDeviceProp_field52;
  uint32_t __FIXME__l_struct_struct_OC_cudaDeviceProp_field53;
  uint32_t __FIXME__l_struct_struct_OC_cudaDeviceProp_field54;
  uint32_t __FIXME__l_struct_struct_OC_cudaDeviceProp_field55;
  uint32_t __FIXME__l_struct_struct_OC_cudaDeviceProp_field56;
  uint64_t __FIXME__l_struct_struct_OC_cudaDeviceProp_field57;
  uint32_t __FIXME__l_struct_struct_OC_cudaDeviceProp_field58;
  uint32_t __FIXME__l_struct_struct_OC_cudaDeviceProp_field59;
  uint32_t __FIXME__l_struct_struct_OC_cudaDeviceProp_field60;
  uint32_t __FIXME__l_struct_struct_OC_cudaDeviceProp_field61;
};
struct __FIXME__l_unnamed_2 {
  double __FIXME__l_unnamed_2_field0;
  double __FIXME__l_unnamed_2_field1;
};
struct __FIXME__l_struct_struct_OC_dim3 {
  uint32_t __FIXME__l_struct_struct_OC_dim3_field0;
  uint32_t __FIXME__l_struct_struct_OC_dim3_field1;
  uint32_t __FIXME__l_struct_struct_OC_dim3_field2;
};
struct __FIXME__l_unnamed_1 {
  uint64_t __FIXME__l_unnamed_1_field0;
  uint32_t __FIXME__l_unnamed_1_field1;
};

/* External Global Variable Declarations */
extern char /* (empty) */ extern_share_data;

/* Function Declarations */
uint32_t ilog2_device(uint32_t) __ATTRIBUTELIST__((noinline, nothrow));
void cffts3_gpu_cfftz_device(uint32_t, uint32_t, uint32_t, struct __FIXME__l_struct_struct_OC_dcomplex*, struct __FIXME__l_struct_struct_OC_dcomplex*, struct __FIXME__l_struct_struct_OC_dcomplex*, uint32_t, uint32_t) __ATTRIBUTELIST__((noinline, nothrow));
void cffts3_gpu_fftz2_device(uint32_t, uint32_t, uint32_t, uint32_t, struct __FIXME__l_struct_struct_OC_dcomplex*, struct __FIXME__l_struct_struct_OC_dcomplex*, struct __FIXME__l_struct_struct_OC_dcomplex*, uint32_t, uint32_t) __ATTRIBUTELIST__((noinline, nothrow));
double atomicAdd(double*, double) __ATTRIBUTELIST__((noinline, nothrow));
uint64_t atomicCAS(uint64_t*, uint64_t, uint64_t) __ATTRIBUTELIST__((noinline, nothrow));
void vranlc_device(uint32_t, double*, double, double*) __ATTRIBUTELIST__((noinline, nothrow));
void ipow46_device(double, uint32_t, double*) __ATTRIBUTELIST__((noinline, nothrow));
double randlc_device(double*, double) __ATTRIBUTELIST__((noinline, nothrow));
double randlc(double*, double) __ATTRIBUTELIST__((noinline, nothrow));
void c_print_results(uint8_t*, int8_t, uint32_t, uint32_t, uint32_t, uint32_t, double, double, uint8_t*, uint32_t, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*) __ATTRIBUTELIST__((noinline));
double pow(double, double) __ATTRIBUTELIST__((nothrow));
int main(int, char **) __ATTRIBUTELIST__((noinline));
void setup(void) __ATTRIBUTELIST__((noinline));
void setup_gpu(void) __ATTRIBUTELIST__((noinline));
void init_ui_gpu(struct __FIXME__l_struct_struct_OC_dcomplex*, struct __FIXME__l_struct_struct_OC_dcomplex*, double*) __ATTRIBUTELIST__((noinline));
void compute_indexmap_gpu(double*) __ATTRIBUTELIST__((noinline));
void compute_initial_conditions_gpu(struct __FIXME__l_struct_struct_OC_dcomplex*) __ATTRIBUTELIST__((noinline));
void fft_init_gpu(uint32_t) __ATTRIBUTELIST__((noinline));
void fft_gpu(uint32_t, struct __FIXME__l_struct_struct_OC_dcomplex*, struct __FIXME__l_struct_struct_OC_dcomplex*) __ATTRIBUTELIST__((noinline));
void evolve_gpu(struct __FIXME__l_struct_struct_OC_dcomplex*, struct __FIXME__l_struct_struct_OC_dcomplex*, double*) __ATTRIBUTELIST__((noinline));
void checksum_gpu(uint32_t, struct __FIXME__l_struct_struct_OC_dcomplex*) __ATTRIBUTELIST__((noinline));
void verify(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t*, uint8_t*) __ATTRIBUTELIST__((noinline));
double log(double) __ATTRIBUTELIST__((nothrow));
void release_gpu(void) __ATTRIBUTELIST__((noinline));
struct __FIXME__l_unnamed_2 dcomplex_div(double, double, double, double) __ATTRIBUTELIST__((noinline, nothrow));
double sqrt(double) __ATTRIBUTELIST__((nothrow));
void cffts1_gpu(uint32_t, struct __FIXME__l_struct_struct_OC_dcomplex*, struct __FIXME__l_struct_struct_OC_dcomplex*, struct __FIXME__l_struct_struct_OC_dcomplex*, struct __FIXME__l_struct_struct_OC_dcomplex*, struct __FIXME__l_struct_struct_OC_dcomplex*) __ATTRIBUTELIST__((noinline));
void cffts2_gpu(uint32_t, struct __FIXME__l_struct_struct_OC_dcomplex*, struct __FIXME__l_struct_struct_OC_dcomplex*, struct __FIXME__l_struct_struct_OC_dcomplex*, struct __FIXME__l_struct_struct_OC_dcomplex*, struct __FIXME__l_struct_struct_OC_dcomplex*) __ATTRIBUTELIST__((noinline));
void cffts3_gpu(uint32_t, struct __FIXME__l_struct_struct_OC_dcomplex*, struct __FIXME__l_struct_struct_OC_dcomplex*, struct __FIXME__l_struct_struct_OC_dcomplex*, struct __FIXME__l_struct_struct_OC_dcomplex*, struct __FIXME__l_struct_struct_OC_dcomplex*) __ATTRIBUTELIST__((noinline));
uint32_t ilog2(uint32_t) __ATTRIBUTELIST__((noinline, nothrow));
double cos(double) __ATTRIBUTELIST__((nothrow));
double sin(double) __ATTRIBUTELIST__((nothrow));
void ipow46(double, uint32_t, double*) __ATTRIBUTELIST__((noinline, nothrow));
void omp_set_num_threads(uint32_t);
double exp(double);
void init_ui_gpu_kernel(struct __FIXME__l_struct_struct_OC_dcomplex*, struct __FIXME__l_struct_struct_OC_dcomplex*, double*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) __ATTRIBUTELIST__((noinline, nothrow));
void compute_indexmap_gpu_kernel(double*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) __ATTRIBUTELIST__((noinline, nothrow));
void compute_initial_conditions_gpu_kernel(struct __FIXME__l_struct_struct_OC_dcomplex*, double*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) __ATTRIBUTELIST__((noinline, nothrow));
void evolve_gpu_kernel(struct __FIXME__l_struct_struct_OC_dcomplex*, struct __FIXME__l_struct_struct_OC_dcomplex*, double*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) __ATTRIBUTELIST__((noinline, nothrow));
void checksum_gpu_kernel0(uint32_t, struct __FIXME__l_struct_struct_OC_dcomplex*, struct __FIXME__l_struct_struct_OC_dcomplex*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) __ATTRIBUTELIST__((noinline, nothrow));
void cffts1_gpu_kernel_1(struct __FIXME__l_struct_struct_OC_dcomplex*, struct __FIXME__l_struct_struct_OC_dcomplex*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) __ATTRIBUTELIST__((noinline, nothrow));
void cffts1_gpu_kernel_2(uint32_t, struct __FIXME__l_struct_struct_OC_dcomplex*, struct __FIXME__l_struct_struct_OC_dcomplex*, struct __FIXME__l_struct_struct_OC_dcomplex*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) __ATTRIBUTELIST__((noinline, nothrow));
void cffts1_gpu_kernel_3(struct __FIXME__l_struct_struct_OC_dcomplex*, struct __FIXME__l_struct_struct_OC_dcomplex*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) __ATTRIBUTELIST__((noinline, nothrow));
void cffts2_gpu_kernel_1(struct __FIXME__l_struct_struct_OC_dcomplex*, struct __FIXME__l_struct_struct_OC_dcomplex*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) __ATTRIBUTELIST__((noinline, nothrow));
void cffts2_gpu_kernel_2(uint32_t, struct __FIXME__l_struct_struct_OC_dcomplex*, struct __FIXME__l_struct_struct_OC_dcomplex*, struct __FIXME__l_struct_struct_OC_dcomplex*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) __ATTRIBUTELIST__((noinline, nothrow));
void cffts2_gpu_kernel_3(struct __FIXME__l_struct_struct_OC_dcomplex*, struct __FIXME__l_struct_struct_OC_dcomplex*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) __ATTRIBUTELIST__((noinline, nothrow));
void cffts3_gpu_kernel_1(struct __FIXME__l_struct_struct_OC_dcomplex*, struct __FIXME__l_struct_struct_OC_dcomplex*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) __ATTRIBUTELIST__((noinline, nothrow));
void cffts3_gpu_kernel_2(uint32_t, struct __FIXME__l_struct_struct_OC_dcomplex*, struct __FIXME__l_struct_struct_OC_dcomplex*, struct __FIXME__l_struct_struct_OC_dcomplex*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) __ATTRIBUTELIST__((noinline, nothrow));
void cffts3_gpu_kernel_3(struct __FIXME__l_struct_struct_OC_dcomplex*, struct __FIXME__l_struct_struct_OC_dcomplex*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) __ATTRIBUTELIST__((noinline, nothrow));
void checksum_gpu_kernel1(uint32_t, struct __FIXME__l_struct_struct_OC_dcomplex*, struct __FIXME__l_struct_struct_OC_dcomplex*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) __ATTRIBUTELIST__((noinline, nothrow));


/* Global Variable Definitions and Initialization */
double* starts_device;
double* twiddle_device;
struct __FIXME__l_struct_struct_OC_dcomplex* sums_device;
struct __FIXME__l_struct_struct_OC_dcomplex* u_device;
struct __FIXME__l_struct_struct_OC_dcomplex* u0_device;
struct __FIXME__l_struct_struct_OC_dcomplex* u1_device;
struct __FIXME__l_struct_struct_OC_dcomplex* u2_device;
struct __FIXME__l_struct_struct_OC_dcomplex* y0_device;
struct __FIXME__l_struct_struct_OC_dcomplex* y1_device;
uint64_t size_sums_device;
uint64_t size_starts_device;
uint64_t size_twiddle_device;
uint64_t size_u_device;
uint64_t size_u0_device;
uint64_t size_u1_device;
uint64_t size_y0_device;
uint64_t size_y1_device;
uint64_t size_shared_data;
uint32_t blocks_per_grid_on_compute_indexmap;
uint32_t blocks_per_grid_on_compute_initial_conditions;
uint32_t blocks_per_grid_on_init_ui;
uint32_t blocks_per_grid_on_evolve;
uint32_t blocks_per_grid_on_fftx_1;
uint32_t blocks_per_grid_on_fftx_2;
uint32_t blocks_per_grid_on_fftx_3;
uint32_t blocks_per_grid_on_ffty_1;
uint32_t blocks_per_grid_on_ffty_2;
uint32_t blocks_per_grid_on_ffty_3;
uint32_t blocks_per_grid_on_fftz_1;
uint32_t blocks_per_grid_on_fftz_2;
uint32_t blocks_per_grid_on_fftz_3;
uint32_t blocks_per_grid_on_checksum;
uint32_t threads_per_block_on_compute_indexmap;
uint32_t threads_per_block_on_compute_initial_conditions;
uint32_t threads_per_block_on_init_ui;
uint32_t threads_per_block_on_evolve;
uint32_t threads_per_block_on_fftx_1;
uint32_t threads_per_block_on_fftx_2;
uint32_t threads_per_block_on_fftx_3;
uint32_t threads_per_block_on_ffty_1;
uint32_t threads_per_block_on_ffty_2;
uint32_t threads_per_block_on_ffty_3;
uint32_t threads_per_block_on_fftz_1;
uint32_t threads_per_block_on_fftz_2;
uint32_t threads_per_block_on_fftz_3;
uint32_t threads_per_block_on_checksum;
uint32_t gpu_device_id;
uint32_t total_devices;
struct __FIXME__l_struct_struct_OC_cudaDeviceProp gpu_device_properties;
uint8_t _OC_str[27] = { "\n\n %s Benchmark Completed\n" };
uint8_t _OC_str_OC_1[46] = { " class_npb       =                        %c\n" };
uint8_t _OC_str_OC_2[38] = { " Size            =             %12ld\n" };
uint8_t _OC_str_OC_3[44] = { " Size            =             %4dx%4dx%4d\n" };
uint8_t _OC_str_OC_4[8] = { "%15.0lf" };
uint8_t _OC_str_OC_5[34] = { " Size            =          %15s\n" };
uint8_t _OC_str_OC_6[37] = { " Size            =             %12d\n" };
uint8_t _OC_str_OC_7[42] = { " Size            =           %4dx%4dx%4d\n" };
uint8_t _OC_str_OC_8[37] = { " Iterations      =             %12d\n" };
uint8_t _OC_str_OC_9[39] = { " Time in seconds =             %12.2f\n" };
uint8_t _OC_str_OC_10[39] = { " Mop/s total     =             %12.2f\n" };
uint8_t _OC_str_OC_11[25] = { " Operation type  = %24s\n" };
uint8_t _OC_str_OC_12[45] = { " Verification    =            NOT PERFORMED\n" };
uint8_t _OC_str_OC_13[45] = { " Verification    =               SUCCESSFUL\n" };
uint8_t _OC_str_OC_14[45] = { " Verification    =             UNSUCCESSFUL\n" };
uint8_t _OC_str_OC_15[37] = { " Version         =             %12s\n" };
uint8_t _OC_str_OC_16[37] = { " Compile date    =             %12s\n" };
uint8_t _OC_str_OC_17[37] = { " NVCC version    =             %12s\n" };
uint8_t _OC_str_OC_18[37] = { " CUDA version    =             %12s\n" };
uint8_t _OC_str_OC_19[20] = { "\n Compile options:\n" };
uint8_t _OC_str_OC_20[23] = { "    CC           = %s\n" };
uint8_t _OC_str_OC_21[23] = { "    CLINK        = %s\n" };
uint8_t _OC_str_OC_22[23] = { "    C_LIB        = %s\n" };
uint8_t _OC_str_OC_23[23] = { "    C_INC        = %s\n" };
uint8_t _OC_str_OC_24[23] = { "    CFLAGS       = %s\n" };
uint8_t _OC_str_OC_25[23] = { "    CLINKFLAGS   = %s\n" };
uint8_t _OC_str_OC_26[23] = { "    RAND         = %s\n" };
uint8_t _OC_str_OC_27[13] = { "\n Hardware:\n" };
uint8_t _OC_str_OC_28[23] = { "    CPU device   = %s\n" };
uint8_t _OC_str_OC_29[23] = { "    GPU device   = %s\n" };
uint8_t _OC_str_OC_30[13] = { "\n Software:\n" };
uint8_t _OC_str_OC_31[23] = { "    Parameters   = %s\n" };
uint8_t _OC_str_OC_32[2] = { "\n" };
uint8_t _OC_str_OC_33[72] = { "----------------------------------------------------------------------\n" };
uint8_t _OC_str_OC_34[27] = { " NPB-CPP is developed by:\n" };
uint8_t _OC_str_OC_35[56] = { "            Dalvan Griebler <dalvangriebler@gmail.com>\n" };
uint8_t _OC_str_OC_36[52] = { "            Gabriell Araujo <hexenoften@gmail.com>\n" };
uint8_t _OC_str_OC_37[46] = { "            J\xC3\xBAnior L\xC3\xB6\x66\x66 <loffjh@gmail.com>\n" };
uint8_t _OC_str_OC_38[43] = { " In case of problems, send an email to us\n" };
struct __FIXME__l_struct_struct_OC_dcomplex* _ZL4sums;
double* _ZL7twiddle;
struct __FIXME__l_struct_struct_OC_dcomplex* _ZL1u;
struct __FIXME__l_struct_struct_OC_dcomplex* _ZL2u0;
struct __FIXME__l_struct_struct_OC_dcomplex* _ZL2u1;
uint32_t* _ZL4dims;
uint32_t _ZL5niter;
uint8_t _OC_str_OC_39[40] = { "T = %5d     Checksum = %22.12e %22.12e\n" };
uint8_t _OC_str_OC_40[10] = { "%5s\t%25s\n" };
uint8_t _OC_str_OC_41[11] = { "GPU Kernel" };
uint8_t _OC_str_OC_42[18] = { "Threads Per Block" };
uint8_t _OC_str_OC_43[11] = { "%29s\t%25d\n" };
uint8_t _OC_str_OC_44[10] = { " indexmap" };
uint8_t _OC_str_OC_45[20] = { " initial conditions" };
uint8_t _OC_str_OC_46[9] = { " init ui" };
uint8_t _OC_str_OC_47[8] = { " evolve" };
uint8_t _OC_str_OC_48[8] = { " fftx 1" };
uint8_t _OC_str_OC_49[8] = { " fftx 2" };
uint8_t _OC_str_OC_50[8] = { " fftx 3" };
uint8_t _OC_str_OC_51[8] = { " ffty 1" };
uint8_t _OC_str_OC_52[8] = { " ffty 2" };
uint8_t _OC_str_OC_53[8] = { " ffty 3" };
uint8_t _OC_str_OC_54[8] = { " fftz 1" };
uint8_t _OC_str_OC_55[8] = { " fftz 2" };
uint8_t _OC_str_OC_56[8] = { " fftz 3" };
uint8_t _OC_str_OC_57[10] = { " checksum" };
uint8_t _OC_str_OC_58[3] = { "FT" };
uint8_t _OC_str_OC_59[25] = { "          floating point" };
uint8_t _OC_str_OC_60[4] = { "4.1" };
uint8_t _OC_str_OC_61[12] = { "16 Feb 2026" };
uint8_t _OC_str_OC_62[6] = { "\xDC\x7FR\xFE\x7F" };
uint8_t _OC_str_OC_63[42] = { "Intel(R) Xeon(R) CPU E5-2697 v3 @ 2.60GHz" };
uint8_t _OC_str_OC_64[23] = { "${NVCC} ${EXTRA_STUFF}" };
uint8_t _OC_str_OC_65[6] = { "$(CC)" };
uint8_t _OC_str_OC_66[5] = { "-lm " };
uint8_t _OC_str_OC_67[13] = { "-I../common " };
uint8_t _OC_str_OC_68[4] = { "-O3" };
uint8_t _OC_str_OC_69[7] = { "randdp" };
uint8_t _OC_str_OC_73[33] = { " Result verification successful\n" };
uint8_t _OC_str_OC_74[29] = { " Result verification failed\n" };
uint8_t _OC_str_OC_75[17] = { " class_npb = %c\n" };
uint8_t _OC_str_OC_70[65] = { "\n\n NAS Parallel Benchmarks 4.1 CUDA C++ version - FT Benchmark\n\n" };
uint8_t _OC_str_OC_71[36] = { " Size                : %4dx%4dx%4d\n" };
uint8_t _OC_str_OC_72[35] = { " Iterations                  :%7d\n" };
__thread double extern_share_data_shared[1024];


/* LLVM Intrinsic Builtin Function Bodies */
static __forceinline uint32_t llvm_add_u32(uint32_t a, uint32_t b) {
  uint32_t r = a + b;
  return r;
}
static __forceinline uint64_t llvm_add_u64(uint64_t a, uint64_t b) {
  uint64_t r = a + b;
  return r;
}
static __forceinline uint32_t llvm_sub_u32(uint32_t a, uint32_t b) {
  uint32_t r = a - b;
  return r;
}
static __forceinline uint64_t llvm_sub_u64(uint64_t a, uint64_t b) {
  uint64_t r = a - b;
  return r;
}
static __forceinline uint32_t llvm_mul_u32(uint32_t a, uint32_t b) {
  uint32_t r = a * b;
  return r;
}
static __forceinline uint64_t llvm_mul_u64(uint64_t a, uint64_t b) {
  uint64_t r = a * b;
  return r;
}
static __forceinline uint32_t llvm_sdiv_u32(int32_t a, int32_t b) {
  uint32_t r = a / b;
  return r;
}
static __forceinline uint32_t llvm_srem_u32(int32_t a, int32_t b) {
  uint32_t r = a % b;
  return r;
}
static __forceinline uint32_t llvm_OC_nvvm_OC_d2i_OC_hi(double a) {
  uint32_t r;
  llvmBitCastUnion bc;
  bc.Double = a;
  r = (uint32_t)(bc.Int64 >> 32);
  return r;
}
static __forceinline double llvm_OC_nvvm_OC_mul_OC_rn_OC_d(double a, double b) {
  double r;
  r = a * b;
  return r;
}
static __forceinline double llvm_OC_nvvm_OC_add_OC_rn_OC_d(double a, double b) {
  double r;
  r = a + b;
  return r;
}
static __forceinline uint32_t llvm_OC_nvvm_OC_d2i_OC_lo(double a) {
  uint32_t r;
  llvmBitCastUnion bc;
  bc.Double = a;
  r = (uint32_t)(bc.Int64 & UINT64_C(0xFFFFFFFF));
  return r;
}
static __forceinline double llvm_OC_nvvm_OC_fma_OC_rn_OC_d(double a, double b, double c) {
  double r;
  r = fma(a, b, c);
  return r;
}
static __forceinline double llvm_OC_nvvm_OC_lohi_OC_i2d(uint32_t a, uint32_t b) {
  double r;
  llvmBitCastUnion bc;
  bc.Int64 = ((uint64_t)(uint32_t)b << 32) | (uint64_t)(uint32_t)a;
  r = bc.Double;
  return r;
}
static __forceinline double llvm_OC_ceil_OC_f64(double a) {
  double r;
  r = ceil(a);
  return r;
}


/* Function Bodies */

// FUNCTION ORDER ID 0 START
// INSERT COMMENT FUNCTION: ilog2_device
uint32_t ilog2_device(uint32_t n) {
  int32_t nn;
  uint32_t lg;

  if (n == 1) {
  return 0;
  }
  nn = 2;
  lg = 1;
// INSERT COMMENT LOOP: ilog2_device::while.cond
while (nn < ((int32_t)n)) {
  nn = (nn << 1);
  lg = lg + 1;
}
  return lg;
}
// FUNCTION ORDER ID 0 END


// FUNCTION ORDER ID 1 START
// INSERT COMMENT FUNCTION: cffts3_gpu_cfftz_device
void cffts3_gpu_cfftz_device(uint32_t is, uint32_t m, uint32_t n, struct __FIXME__l_struct_struct_OC_dcomplex* x, struct __FIXME__l_struct_struct_OC_dcomplex* y, struct __FIXME__l_struct_struct_OC_dcomplex* u_device, uint32_t index_arg, uint32_t size_arg) {
  int32_t l;
  int64_t j;

// INSERT COMMENT LOOP: cffts3_gpu_cfftz_device::for.cond
for(int32_t l = 1; l <= ((int32_t)m);   l = l + 2) {
  cffts3_gpu_fftz2_device(is, l, m, n, u_device, x, y, index_arg, size_arg);
;
  if (l == m) {
  break;
  }
cffts3_gpu_fftz2_device(is, l + 1, m, n, u_device, y, x, index_arg, size_arg);
}
// INSERT COMMENT IFELSE: cffts3_gpu_cfftz_device::for.end
  if ((int)m % (int)2 == 1) { // IFELSE MARKER: for.end IF
for(int64_t j = 0; j < n;   j = j + 1) {
  (x+(j * size_arg + index_arg))->__FIXME__l_struct_struct_OC_dcomplex_field0 = (y+(j * size_arg + index_arg))->__FIXME__l_struct_struct_OC_dcomplex_field0;
  (x+(j * size_arg + index_arg))->__FIXME__l_struct_struct_OC_dcomplex_field1 = (y+(j * size_arg + index_arg))->__FIXME__l_struct_struct_OC_dcomplex_field1;
}
  }
  return;
}
// FUNCTION ORDER ID 1 END


// FUNCTION ORDER ID 2 START
// INSERT COMMENT FUNCTION: cffts3_gpu_fftz2_device
void cffts3_gpu_fftz2_device(uint32_t is, uint32_t l, uint32_t m, uint32_t n, struct __FIXME__l_struct_struct_OC_dcomplex* u, struct __FIXME__l_struct_struct_OC_dcomplex* x, struct __FIXME__l_struct_struct_OC_dcomplex* y, uint32_t index_arg, uint32_t size_arg) {
  struct __FIXME__l_struct_struct_OC_dcomplex u1;    /* Address-exposed local */
  int32_t lk;
  int32_t __FIXME__shl2;
  int64_t i;
  int64_t k;

  lk = (1 << (l - 1));
  __FIXME__shl2 = (1 << (m - l));
// INSERT COMMENT LOOP: cffts3_gpu_fftz2_device::for.cond
for(int64_t i = 0; i < __FIXME__shl2;   i = i + 1) {
  uint64_t __FIXME__3 = i * lk;
  uint64_t __FIXME__4 = __FIXME__3 + n / 2;
  uint64_t __FIXME__5 = i * 2 * lk;
  uint64_t __FIXME__6 = __FIXME__5 + lk;
  if (((int32_t)is) >= 1) { // IFELSE MARKER: for.body IF
  u1.__FIXME__l_struct_struct_OC_dcomplex_field0 = (u+(__FIXME__shl2 + i))->__FIXME__l_struct_struct_OC_dcomplex_field0;
  u1.__FIXME__l_struct_struct_OC_dcomplex_field1 = (u+(__FIXME__shl2 + i))->__FIXME__l_struct_struct_OC_dcomplex_field1;
  } else { // IFELSE MARKER: for.body ELSE
  u1.__FIXME__l_struct_struct_OC_dcomplex_field0 = (u+(__FIXME__shl2 + i))->__FIXME__l_struct_struct_OC_dcomplex_field0;
  u1.__FIXME__l_struct_struct_OC_dcomplex_field1 = -((u+(__FIXME__shl2 + i))->__FIXME__l_struct_struct_OC_dcomplex_field1);
  }
for(int64_t k = 0; k < lk;   k = k + 1) {
  double x11real = (x+((__FIXME__3 + k) * size_arg + index_arg))->__FIXME__l_struct_struct_OC_dcomplex_field0;
  double x11imag = (x+((__FIXME__3 + k) * size_arg + index_arg))->__FIXME__l_struct_struct_OC_dcomplex_field1;
  double x21real = (x+((__FIXME__4 + k) * size_arg + index_arg))->__FIXME__l_struct_struct_OC_dcomplex_field0;
  double x21imag = (x+((__FIXME__4 + k) * size_arg + index_arg))->__FIXME__l_struct_struct_OC_dcomplex_field1;
  (y+((__FIXME__5 + k) * size_arg + index_arg))->__FIXME__l_struct_struct_OC_dcomplex_field0 = (x11real + x21real);
  (y+((__FIXME__5 + k) * size_arg + index_arg))->__FIXME__l_struct_struct_OC_dcomplex_field1 = (x11imag + x21imag);
  (y+((__FIXME__6 + k) * size_arg + index_arg))->__FIXME__l_struct_struct_OC_dcomplex_field0 = ((u1.__FIXME__l_struct_struct_OC_dcomplex_field0 * (x11real - x21real)) - ((u1.__FIXME__l_struct_struct_OC_dcomplex_field1 * (x11imag - x21imag))));
  (y+((__FIXME__6 + k) * size_arg + index_arg))->__FIXME__l_struct_struct_OC_dcomplex_field1 = ((u1.__FIXME__l_struct_struct_OC_dcomplex_field0 * (x11imag - x21imag)) + (u1.__FIXME__l_struct_struct_OC_dcomplex_field1 * (x11real - x21real)));
}
}
  return;
}
// FUNCTION ORDER ID 2 END


// FUNCTION ORDER ID 3 START
// INSERT COMMENT FUNCTION: vranlc_device
void vranlc_device(uint32_t n, double* x_seed, double a, double* y) {
  double a2;
  double x;
  int64_t i;

  a2 = (a - ((8388608 * ((double)((int32_t)((int32_t)(1.1920928955078125E-7 * a)))))));
  x = *x_seed;
// INSERT COMMENT LOOP: vranlc_device::for.cond
for(int64_t i = 0; i < n;   i = i + 1) {
  double x2 = (x - ((8388608 * ((double)((int32_t)((int32_t)(1.1920928955078125E-7 * x)))))));
  double t1 = ((((double)((int32_t)((int32_t)(1.1920928955078125E-7 * a)))) * x2) + (a2 * ((double)((int32_t)((int32_t)(1.1920928955078125E-7 * x))))));
  double t3 = ((8388608 * (t1 - ((8388608 * ((double)((int32_t)((int32_t)(1.1920928955078125E-7 * t1)))))))) + (a2 * x2));
  x = (t3 - ((70368744177664 * ((double)((int32_t)((int32_t)(1.4210854715202004E-14 * t3)))))));
  y[i] = (1.4210854715202004E-14 * x);
}
  *x_seed = x;
}
// FUNCTION ORDER ID 3 END


// FUNCTION ORDER ID 4 START
// INSERT COMMENT FUNCTION: randlc_device
double randlc_device(double* x, double a) {
  double a2;
  double x2;
  double t1;
  double t3;

  a2 = (a - ((8388608 * ((double)((int32_t)((int32_t)(1.1920928955078125E-7 * a)))))));
  x2 = (*x - ((8388608 * ((double)((int32_t)((int32_t)(1.1920928955078125E-7 * *x)))))));
  t1 = ((((double)((int32_t)((int32_t)(1.1920928955078125E-7 * a)))) * x2) + (a2 * ((double)((int32_t)((int32_t)(1.1920928955078125E-7 * *x))))));
  t3 = ((8388608 * (t1 - ((8388608 * ((double)((int32_t)((int32_t)(1.1920928955078125E-7 * t1)))))))) + (a2 * x2));
  *x = (t3 - ((70368744177664 * ((double)((int32_t)((int32_t)(1.4210854715202004E-14 * t3)))))));
  return (1.4210854715202004E-14 * *x);
}
// FUNCTION ORDER ID 4 END


// FUNCTION ORDER ID 5 START
// INSERT COMMENT FUNCTION: randlc
double randlc(double* x, double a) {
  double a2;
  double x2;
  double t1;
  double t3;

  a2 = (a - ((8388608 * ((double)((int32_t)((int32_t)(1.1920928955078125E-7 * a)))))));
  x2 = (*x - ((8388608 * ((double)((int32_t)((int32_t)(1.1920928955078125E-7 * *x)))))));
  t1 = ((((double)((int32_t)((int32_t)(1.1920928955078125E-7 * a)))) * x2) + (a2 * ((double)((int32_t)((int32_t)(1.1920928955078125E-7 * *x))))));
  t3 = ((8388608 * (t1 - ((8388608 * ((double)((int32_t)((int32_t)(1.1920928955078125E-7 * t1)))))))) + (a2 * x2));
  *x = (t3 - ((70368744177664 * ((double)((int32_t)((int32_t)(1.4210854715202004E-14 * t3)))))));
  return (1.4210854715202004E-14 * *x);
}
// FUNCTION ORDER ID 5 END


// FUNCTION ORDER ID 6 START
// INSERT COMMENT FUNCTION: c_print_results
void c_print_results(uint8_t* name, int8_t class_npb, uint32_t n1, uint32_t n2, uint32_t n3, uint32_t niter, double t, double mops, uint8_t* optype, uint32_t passed_verification, uint8_t* npbversion, uint8_t* compiletime, uint8_t* compilerversion, uint8_t* libversion, uint8_t* cpu_device, uint8_t* gpu_device, uint8_t* gpu_config, uint8_t* cc, uint8_t* clink, uint8_t* c_lib, uint8_t* c_inc, uint8_t* cflags, uint8_t* clinkflags, uint8_t* rand) {
  uint8_t size[16];    /* Address-exposed local */
  uint64_t nn;
  double __FIXME__call29;
  uint32_t j;

// INSERT COMMENT IFELSE: c_print_results::entry
  printf((_OC_str), name);
  printf((_OC_str_OC_1), class_npb);
  if (name[0] == 73) { // IFELSE MARKER: entry IF
  if (name[1] == 83) { // IFELSE MARKER: land.lhs.true IF
  if (n3 == 0) { // IFELSE MARKER: if.then IF
  if (n2 != 0) { // IFELSE MARKER: if.then7 IF
  nn = nn * n2;
  }
  printf((_OC_str_OC_2), nn);
  } else { // IFELSE MARKER: if.then ELSE
  printf((_OC_str_OC_3), n1, n2, n3);
  }
  }
  } else { // IFELSE MARKER: entry ELSE
  if (n2 == 0) { // IFELSE MARKER: if.else15 IF
  if (n3 == 0) { // IFELSE MARKER: land.lhs.true17 IF
  if (name[0] == 69) { // IFELSE MARKER: if.then19 IF
  if (name[1] == 80) { // IFELSE MARKER: land.lhs.true23 IF
  __FIXME__call29 = pow(2, ((double)((int32_t)n1)));
  sprintf(size, (_OC_str_OC_4), __FIXME__call29);
  j = 14;
  if (size[14] == 46) { // IFELSE MARKER: if.then27 IF
  size[14] = 32;
  j = 13;
  }
  size[(j + 1)] = 0;
  printf((_OC_str_OC_5), size);
  }
  } else { // IFELSE MARKER: if.then19 ELSE
  printf((_OC_str_OC_6), n1);
  }
  }
  } else { // IFELSE MARKER: if.else15 ELSE
  printf((_OC_str_OC_7), n1, n2, n3);
  }
  }
  printf((_OC_str_OC_8), niter);
  printf((_OC_str_OC_9), t);
  printf((_OC_str_OC_10), mops);
  printf((_OC_str_OC_11), optype);
  if (((int32_t)passed_verification) < 0) {
  return;
  }
// INSERT COMMENT IFELSE: c_print_results::if.else56
  if (passed_verification != 0) { // IFELSE MARKER: if.else56 IF
  printf((_OC_str_OC_13));
  } else { // IFELSE MARKER: if.else56 ELSE
  printf((_OC_str_OC_14));
  }
  printf((_OC_str_OC_15), npbversion);
  printf((_OC_str_OC_16), compiletime);
  printf((_OC_str_OC_17), compilerversion);
  printf((_OC_str_OC_18), libversion);
  printf((_OC_str_OC_19));
  printf((_OC_str_OC_20), cc);
  printf((_OC_str_OC_21), clink);
  printf((_OC_str_OC_22), c_lib);
  printf((_OC_str_OC_23), c_inc);
  printf((_OC_str_OC_24), cflags);
  printf((_OC_str_OC_25), clinkflags);
  printf((_OC_str_OC_26), rand);
  printf((_OC_str_OC_27));
  printf((_OC_str_OC_28), cpu_device);
  printf((_OC_str_OC_29), gpu_device);
  printf((_OC_str_OC_30));
  printf((_OC_str_OC_31), gpu_config);
  printf((_OC_str_OC_32));
  printf((_OC_str_OC_33));
  printf((_OC_str_OC_34));
  printf((_OC_str_OC_35));
  printf((_OC_str_OC_36));
  printf((_OC_str_OC_37));
  printf((_OC_str_OC_32));
  printf((_OC_str_OC_38));
  printf((_OC_str_OC_33));
  printf((_OC_str_OC_32));
}
// FUNCTION ORDER ID 6 END


// MAIN START
int main(int argc, char ** argv) {
  uint32_t verified;    /* Address-exposed local */
  uint8_t class_npb;    /* Address-exposed local */
  uint8_t gpu_config[256];    /* Address-exposed local */
  uint8_t gpu_config_string[2048];    /* Address-exposed local */
  uint8_t* __FIXME__call;
  uint8_t* __FIXME__call1;
  uint8_t* __FIXME__call2;
  uint8_t* __FIXME__call3;
  uint8_t* __FIXME__call4;
  uint8_t* __FIXME__call5;
  int32_t iter;
  double __FIXME__call19;
  double __FIXME__call20;
  double mflops;

  __FIXME__call = malloc(112);
  _ZL4sums = ((struct __FIXME__l_struct_struct_OC_dcomplex*)__FIXME__call);
  __FIXME__call1 = malloc(67108864);
  _ZL7twiddle = ((double*)__FIXME__call1);
  __FIXME__call2 = malloc(4096);
  _ZL1u = ((struct __FIXME__l_struct_struct_OC_dcomplex*)__FIXME__call2);
  __FIXME__call3 = malloc(134217728);
  _ZL2u0 = ((struct __FIXME__l_struct_struct_OC_dcomplex*)__FIXME__call3);
  __FIXME__call4 = malloc(134217728);
  _ZL2u1 = ((struct __FIXME__l_struct_struct_OC_dcomplex*)__FIXME__call4);
  __FIXME__call5 = malloc(12);
  _ZL4dims = ((uint32_t*)__FIXME__call5);
setup();
setup_gpu();
init_ui_gpu(u0_device, u1_device, twiddle_device);
compute_indexmap_gpu(twiddle_device);
compute_initial_conditions_gpu(u1_device);
fft_init_gpu(256);
fft_gpu(1, u1_device, u0_device);
compute_indexmap_gpu(twiddle_device);
compute_initial_conditions_gpu(u1_device);
fft_init_gpu(256);
fft_gpu(1, u1_device, u0_device);
// INSERT COMMENT LOOP: main::for.cond
for(int32_t iter = 1; iter <= ((int32_t)_ZL5niter);   iter = iter + 1) {
evolve_gpu(u0_device, u1_device, twiddle_device);
fft_gpu(-1, u1_device, u1_device);
checksum_gpu(iter, u1_device);
}
  ;
// INSERT COMMENT LOOP: main::for.cond9
for(int64_t iter = 1; iter <= _ZL5niter;   iter = iter + 1) {
  printf((_OC_str_OC_39), iter, (_ZL4sums+iter)->__FIXME__l_struct_struct_OC_dcomplex_field0, (_ZL4sums+iter)->__FIXME__l_struct_struct_OC_dcomplex_field1);
}
// INSERT COMMENT IFELSE: main::for.end17
  verify(256, 256, 128, _ZL5niter, (&verified), (&class_npb));
;
  if (llvm_fcmp_une(0, 0)) { // IFELSE MARKER: for.end17 IF
  __FIXME__call19 = log(8388608);
  __FIXME__call20 = log(8388608);
  mflops = ((8.3886079999999996 * ((14.8157 + (7.1964100000000002 * __FIXME__call19)) + ((5.2351799999999997 + (7.2111299999999998 * __FIXME__call20)) * ((double)((int32_t)_ZL5niter))))) / 0);
  } else { // IFELSE MARKER: for.end17 ELSE
  mflops = 0;
  }
  sprintf(gpu_config, (_OC_str_OC_40), (_OC_str_OC_41), (_OC_str_OC_42));
  strcpy(gpu_config_string, gpu_config);
  sprintf(gpu_config, (_OC_str_OC_43), (_OC_str_OC_44), threads_per_block_on_compute_indexmap);
  strcat(gpu_config_string, gpu_config);
  sprintf(gpu_config, (_OC_str_OC_43), (_OC_str_OC_45), threads_per_block_on_compute_initial_conditions);
  strcat(gpu_config_string, gpu_config);
  sprintf(gpu_config, (_OC_str_OC_43), (_OC_str_OC_46), threads_per_block_on_init_ui);
  strcat(gpu_config_string, gpu_config);
  sprintf(gpu_config, (_OC_str_OC_43), (_OC_str_OC_47), threads_per_block_on_evolve);
  strcat(gpu_config_string, gpu_config);
  sprintf(gpu_config, (_OC_str_OC_43), (_OC_str_OC_48), threads_per_block_on_fftx_1);
  strcat(gpu_config_string, gpu_config);
  sprintf(gpu_config, (_OC_str_OC_43), (_OC_str_OC_49), threads_per_block_on_fftx_2);
  strcat(gpu_config_string, gpu_config);
  sprintf(gpu_config, (_OC_str_OC_43), (_OC_str_OC_50), threads_per_block_on_fftx_3);
  strcat(gpu_config_string, gpu_config);
  sprintf(gpu_config, (_OC_str_OC_43), (_OC_str_OC_51), threads_per_block_on_ffty_1);
  strcat(gpu_config_string, gpu_config);
  sprintf(gpu_config, (_OC_str_OC_43), (_OC_str_OC_52), threads_per_block_on_ffty_2);
  strcat(gpu_config_string, gpu_config);
  sprintf(gpu_config, (_OC_str_OC_43), (_OC_str_OC_53), threads_per_block_on_ffty_3);
  strcat(gpu_config_string, gpu_config);
  sprintf(gpu_config, (_OC_str_OC_43), (_OC_str_OC_54), threads_per_block_on_fftz_1);
  strcat(gpu_config_string, gpu_config);
  sprintf(gpu_config, (_OC_str_OC_43), (_OC_str_OC_55), threads_per_block_on_fftz_2);
  strcat(gpu_config_string, gpu_config);
  sprintf(gpu_config, (_OC_str_OC_43), (_OC_str_OC_56), threads_per_block_on_fftz_3);
  strcat(gpu_config_string, gpu_config);
  sprintf(gpu_config, (_OC_str_OC_43), (_OC_str_OC_57), threads_per_block_on_checksum);
  strcat(gpu_config_string, gpu_config);
c_print_results((_OC_str_OC_58), class_npb, 256, 256, 128, _ZL5niter, 0, mflops, (_OC_str_OC_59), verified, (_OC_str_OC_60), (_OC_str_OC_61), (_OC_str_OC_62), (_OC_str_OC_62), (_OC_str_OC_63), (gpu_device_properties.__FIXME__l_struct_struct_OC_cudaDeviceProp_field0), gpu_config_string, (_OC_str_OC_64), (_OC_str_OC_65), (_OC_str_OC_66), (_OC_str_OC_67), (_OC_str_OC_68), (_OC_str_OC_68), (_OC_str_OC_69));
release_gpu();
free(((uint8_t*)_ZL4sums));
free(((uint8_t*)_ZL7twiddle));
free(((uint8_t*)_ZL1u));
free(((uint8_t*)_ZL2u0));
free(((uint8_t*)_ZL2u1));
free(((uint8_t*)_ZL4dims));
  return 0;
}
// MAIN END


// FUNCTION ORDER ID 7 START
// INSERT COMMENT FUNCTION: setup
void setup(void) {

  _ZL5niter = 6;
  printf((_OC_str_OC_70));
  printf((_OC_str_OC_71), 256, 256, 128);
  printf((_OC_str_OC_72), _ZL5niter);
  printf((_OC_str_OC_32));
}
// FUNCTION ORDER ID 7 END


// FUNCTION ORDER ID 8 START
// INSERT COMMENT FUNCTION: setup_gpu
void setup_gpu(void) {
  double __FIXME__8;
  double __FIXME__9;
  double __FIXME__10;
  double __FIXME__11;
  double __FIXME__12;
  double __FIXME__13;
  double __FIXME__14;
  double __FIXME__15;
  double __FIXME__16;
  double __FIXME__17;
  double __FIXME__18;
  double __FIXME__19;
  double __FIXME__20;
  double __FIXME__21;
  uint8_t* __FIXME__tulip_2e_host_2e_malloc;
  uint8_t* __FIXME__tulip_2e_host_2e_malloc1;
  uint8_t* __FIXME__tulip_2e_host_2e_malloc3;
  uint8_t* __FIXME__tulip_2e_host_2e_malloc5;
  uint8_t* __FIXME__tulip_2e_host_2e_malloc7;
  uint8_t* __FIXME__tulip_2e_host_2e_malloc9;
  uint8_t* __FIXME__tulip_2e_host_2e_malloc11;
  uint8_t* __FIXME__tulip_2e_host_2e_malloc13;

// INSERT COMMENT IFELSE: setup_gpu::entry
  *((&gpu_device_properties.__FIXME__l_struct_struct_OC_cudaDeviceProp_field4)) = 32;
  *((&gpu_device_properties.__FIXME__l_struct_struct_OC_cudaDeviceProp_field6)) = 32;
  if (32 <= ((int32_t)(gpu_device_properties.__FIXME__l_struct_struct_OC_cudaDeviceProp_field6))) { // IFELSE MARKER: entry IF
  threads_per_block_on_compute_indexmap = 32;
  } else { // IFELSE MARKER: entry ELSE
  threads_per_block_on_compute_indexmap = (gpu_device_properties.__FIXME__l_struct_struct_OC_cudaDeviceProp_field4);
  }
// INSERT COMMENT IFELSE: setup_gpu::if.end
  if (32 <= ((int32_t)(gpu_device_properties.__FIXME__l_struct_struct_OC_cudaDeviceProp_field6))) { // IFELSE MARKER: if.end IF
  threads_per_block_on_compute_initial_conditions = 32;
  } else { // IFELSE MARKER: if.end ELSE
  threads_per_block_on_compute_initial_conditions = (gpu_device_properties.__FIXME__l_struct_struct_OC_cudaDeviceProp_field4);
  }
// INSERT COMMENT IFELSE: setup_gpu::if.end4
  if (32 <= ((int32_t)(gpu_device_properties.__FIXME__l_struct_struct_OC_cudaDeviceProp_field6))) { // IFELSE MARKER: if.end4 IF
  threads_per_block_on_init_ui = 32;
  } else { // IFELSE MARKER: if.end4 ELSE
  threads_per_block_on_init_ui = (gpu_device_properties.__FIXME__l_struct_struct_OC_cudaDeviceProp_field4);
  }
// INSERT COMMENT IFELSE: setup_gpu::if.end8
  if (32 <= ((int32_t)(gpu_device_properties.__FIXME__l_struct_struct_OC_cudaDeviceProp_field6))) { // IFELSE MARKER: if.end8 IF
  threads_per_block_on_evolve = 32;
  } else { // IFELSE MARKER: if.end8 ELSE
  threads_per_block_on_evolve = (gpu_device_properties.__FIXME__l_struct_struct_OC_cudaDeviceProp_field4);
  }
// INSERT COMMENT IFELSE: setup_gpu::if.end12
  if (1024 <= ((int32_t)(gpu_device_properties.__FIXME__l_struct_struct_OC_cudaDeviceProp_field6))) { // IFELSE MARKER: if.end12 IF
  threads_per_block_on_fftx_1 = 1024;
  } else { // IFELSE MARKER: if.end12 ELSE
  threads_per_block_on_fftx_1 = (gpu_device_properties.__FIXME__l_struct_struct_OC_cudaDeviceProp_field4);
  }
// INSERT COMMENT IFELSE: setup_gpu::if.end16
  if (32 <= ((int32_t)(gpu_device_properties.__FIXME__l_struct_struct_OC_cudaDeviceProp_field6))) { // IFELSE MARKER: if.end16 IF
  threads_per_block_on_fftx_2 = 32;
  } else { // IFELSE MARKER: if.end16 ELSE
  threads_per_block_on_fftx_2 = (gpu_device_properties.__FIXME__l_struct_struct_OC_cudaDeviceProp_field4);
  }
// INSERT COMMENT IFELSE: setup_gpu::if.end20
  if (256 <= ((int32_t)(gpu_device_properties.__FIXME__l_struct_struct_OC_cudaDeviceProp_field6))) { // IFELSE MARKER: if.end20 IF
  threads_per_block_on_fftx_3 = 256;
  } else { // IFELSE MARKER: if.end20 ELSE
  threads_per_block_on_fftx_3 = (gpu_device_properties.__FIXME__l_struct_struct_OC_cudaDeviceProp_field4);
  }
// INSERT COMMENT IFELSE: setup_gpu::if.end24
  if (32 <= ((int32_t)(gpu_device_properties.__FIXME__l_struct_struct_OC_cudaDeviceProp_field6))) { // IFELSE MARKER: if.end24 IF
  threads_per_block_on_ffty_1 = 32;
  } else { // IFELSE MARKER: if.end24 ELSE
  threads_per_block_on_ffty_1 = (gpu_device_properties.__FIXME__l_struct_struct_OC_cudaDeviceProp_field4);
  }
// INSERT COMMENT IFELSE: setup_gpu::if.end28
  if (32 <= ((int32_t)(gpu_device_properties.__FIXME__l_struct_struct_OC_cudaDeviceProp_field6))) { // IFELSE MARKER: if.end28 IF
  threads_per_block_on_ffty_2 = 32;
  } else { // IFELSE MARKER: if.end28 ELSE
  threads_per_block_on_ffty_2 = (gpu_device_properties.__FIXME__l_struct_struct_OC_cudaDeviceProp_field4);
  }
// INSERT COMMENT IFELSE: setup_gpu::if.end32
  if (32 <= ((int32_t)(gpu_device_properties.__FIXME__l_struct_struct_OC_cudaDeviceProp_field6))) { // IFELSE MARKER: if.end32 IF
  threads_per_block_on_ffty_3 = 32;
  } else { // IFELSE MARKER: if.end32 ELSE
  threads_per_block_on_ffty_3 = (gpu_device_properties.__FIXME__l_struct_struct_OC_cudaDeviceProp_field4);
  }
// INSERT COMMENT IFELSE: setup_gpu::if.end36
  if (32 <= ((int32_t)(gpu_device_properties.__FIXME__l_struct_struct_OC_cudaDeviceProp_field6))) { // IFELSE MARKER: if.end36 IF
  threads_per_block_on_fftz_1 = 32;
  } else { // IFELSE MARKER: if.end36 ELSE
  threads_per_block_on_fftz_1 = (gpu_device_properties.__FIXME__l_struct_struct_OC_cudaDeviceProp_field4);
  }
// INSERT COMMENT IFELSE: setup_gpu::if.end40
  if (32 <= ((int32_t)(gpu_device_properties.__FIXME__l_struct_struct_OC_cudaDeviceProp_field6))) { // IFELSE MARKER: if.end40 IF
  threads_per_block_on_fftz_2 = 32;
  } else { // IFELSE MARKER: if.end40 ELSE
  threads_per_block_on_fftz_2 = (gpu_device_properties.__FIXME__l_struct_struct_OC_cudaDeviceProp_field4);
  }
// INSERT COMMENT IFELSE: setup_gpu::if.end44
  if (32 <= ((int32_t)(gpu_device_properties.__FIXME__l_struct_struct_OC_cudaDeviceProp_field6))) { // IFELSE MARKER: if.end44 IF
  threads_per_block_on_fftz_3 = 32;
  } else { // IFELSE MARKER: if.end44 ELSE
  threads_per_block_on_fftz_3 = (gpu_device_properties.__FIXME__l_struct_struct_OC_cudaDeviceProp_field4);
  }
// INSERT COMMENT IFELSE: setup_gpu::if.end48
  if (32 <= ((int32_t)(gpu_device_properties.__FIXME__l_struct_struct_OC_cudaDeviceProp_field6))) { // IFELSE MARKER: if.end48 IF
  threads_per_block_on_checksum = 32;
  } else { // IFELSE MARKER: if.end48 ELSE
  threads_per_block_on_checksum = (gpu_device_properties.__FIXME__l_struct_struct_OC_cudaDeviceProp_field4);
  }
  __FIXME__8 = /*__FIXME__INTRINSIC_CALL__*/llvm_OC_ceil_OC_f64((8388608 / ((double)((int32_t)threads_per_block_on_compute_indexmap))));
  blocks_per_grid_on_compute_indexmap = ((int32_t)__FIXME__8);
  __FIXME__9 = /*__FIXME__INTRINSIC_CALL__*/llvm_OC_ceil_OC_f64((128 / ((double)((int32_t)threads_per_block_on_compute_initial_conditions))));
  blocks_per_grid_on_compute_initial_conditions = ((int32_t)__FIXME__9);
  __FIXME__10 = /*__FIXME__INTRINSIC_CALL__*/llvm_OC_ceil_OC_f64((8388608 / ((double)((int32_t)threads_per_block_on_init_ui))));
  blocks_per_grid_on_init_ui = ((int32_t)__FIXME__10);
  __FIXME__11 = /*__FIXME__INTRINSIC_CALL__*/llvm_OC_ceil_OC_f64((8388608 / ((double)((int32_t)threads_per_block_on_evolve))));
  blocks_per_grid_on_evolve = ((int32_t)__FIXME__11);
  __FIXME__12 = /*__FIXME__INTRINSIC_CALL__*/llvm_OC_ceil_OC_f64((8388608 / ((double)((int32_t)threads_per_block_on_fftx_1))));
  blocks_per_grid_on_fftx_1 = ((int32_t)__FIXME__12);
  __FIXME__13 = /*__FIXME__INTRINSIC_CALL__*/llvm_OC_ceil_OC_f64((32768 / ((double)((int32_t)threads_per_block_on_fftx_2))));
  blocks_per_grid_on_fftx_2 = ((int32_t)__FIXME__13);
  __FIXME__14 = /*__FIXME__INTRINSIC_CALL__*/llvm_OC_ceil_OC_f64((8388608 / ((double)((int32_t)threads_per_block_on_fftx_3))));
  blocks_per_grid_on_fftx_3 = ((int32_t)__FIXME__14);
  __FIXME__15 = /*__FIXME__INTRINSIC_CALL__*/llvm_OC_ceil_OC_f64((8388608 / ((double)((int32_t)threads_per_block_on_ffty_1))));
  blocks_per_grid_on_ffty_1 = ((int32_t)__FIXME__15);
  __FIXME__16 = /*__FIXME__INTRINSIC_CALL__*/llvm_OC_ceil_OC_f64((32768 / ((double)((int32_t)threads_per_block_on_ffty_2))));
  blocks_per_grid_on_ffty_2 = ((int32_t)__FIXME__16);
  __FIXME__17 = /*__FIXME__INTRINSIC_CALL__*/llvm_OC_ceil_OC_f64((8388608 / ((double)((int32_t)threads_per_block_on_ffty_3))));
  blocks_per_grid_on_ffty_3 = ((int32_t)__FIXME__17);
  __FIXME__18 = /*__FIXME__INTRINSIC_CALL__*/llvm_OC_ceil_OC_f64((8388608 / ((double)((int32_t)threads_per_block_on_fftz_1))));
  blocks_per_grid_on_fftz_1 = ((int32_t)__FIXME__18);
  __FIXME__19 = /*__FIXME__INTRINSIC_CALL__*/llvm_OC_ceil_OC_f64((65536 / ((double)((int32_t)threads_per_block_on_fftz_2))));
  blocks_per_grid_on_fftz_2 = ((int32_t)__FIXME__19);
  __FIXME__20 = /*__FIXME__INTRINSIC_CALL__*/llvm_OC_ceil_OC_f64((8388608 / ((double)((int32_t)threads_per_block_on_fftz_3))));
  blocks_per_grid_on_fftz_3 = ((int32_t)__FIXME__20);
  __FIXME__21 = /*__FIXME__INTRINSIC_CALL__*/llvm_OC_ceil_OC_f64((1024 / ((double)((int32_t)threads_per_block_on_checksum))));
  blocks_per_grid_on_checksum = ((int32_t)__FIXME__21);
  size_sums_device = 112;
  size_starts_device = 1024;
  size_twiddle_device = 67108864;
  size_u_device = 4096;
  size_u0_device = 134217728;
  size_u1_device = 134217728;
  size_y0_device = 134217728;
  size_y1_device = 134217728;
  size_shared_data = threads_per_block_on_checksum * 16;
  __FIXME__tulip_2e_host_2e_malloc = malloc(size_sums_device);
  sums_device = ((struct __FIXME__l_struct_struct_OC_dcomplex*)__FIXME__tulip_2e_host_2e_malloc);
  __FIXME__tulip_2e_host_2e_malloc1 = malloc(size_starts_device);
  starts_device = ((double*)__FIXME__tulip_2e_host_2e_malloc1);
  __FIXME__tulip_2e_host_2e_malloc3 = malloc(size_twiddle_device);
  twiddle_device = ((double*)__FIXME__tulip_2e_host_2e_malloc3);
  __FIXME__tulip_2e_host_2e_malloc5 = malloc(size_u_device);
  u_device = ((struct __FIXME__l_struct_struct_OC_dcomplex*)__FIXME__tulip_2e_host_2e_malloc5);
  __FIXME__tulip_2e_host_2e_malloc7 = malloc(size_u0_device);
  u0_device = ((struct __FIXME__l_struct_struct_OC_dcomplex*)__FIXME__tulip_2e_host_2e_malloc7);
  __FIXME__tulip_2e_host_2e_malloc9 = malloc(size_u1_device);
  u1_device = ((struct __FIXME__l_struct_struct_OC_dcomplex*)__FIXME__tulip_2e_host_2e_malloc9);
  __FIXME__tulip_2e_host_2e_malloc11 = malloc(size_y0_device);
  y0_device = ((struct __FIXME__l_struct_struct_OC_dcomplex*)__FIXME__tulip_2e_host_2e_malloc11);
  __FIXME__tulip_2e_host_2e_malloc13 = malloc(size_y1_device);
  y1_device = ((struct __FIXME__l_struct_struct_OC_dcomplex*)__FIXME__tulip_2e_host_2e_malloc13);
omp_set_num_threads(3);
}
// FUNCTION ORDER ID 8 END


// FUNCTION ORDER ID 9 START
// INSERT COMMENT FUNCTION: init_ui_gpu
void init_ui_gpu(struct __FIXME__l_struct_struct_OC_dcomplex* u0, struct __FIXME__l_struct_struct_OC_dcomplex* u1, double* twiddle) {
  struct __FIXME__l_struct_struct_OC_dim3 __FIXME__agg_2e_tmp;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dim3 __FIXME__agg_2e_tmp1;    /* Address-exposed local */
  struct __FIXME__l_unnamed_1 __FIXME__agg_2e_tmp_2e_coerce;    /* Address-exposed local */
  struct __FIXME__l_unnamed_1 __FIXME__agg_2e_tmp1_2e_coerce;    /* Address-exposed local */
  uint32_t i;
  uint32_t j;

  __FIXME__agg_2e_tmp.__FIXME__l_struct_struct_OC_dim3_field0 = blocks_per_grid_on_init_ui;
  __FIXME__agg_2e_tmp.__FIXME__l_struct_struct_OC_dim3_field1 = 1;
  __FIXME__agg_2e_tmp.__FIXME__l_struct_struct_OC_dim3_field2 = 1;
  __FIXME__agg_2e_tmp1.__FIXME__l_struct_struct_OC_dim3_field0 = threads_per_block_on_init_ui;
  __FIXME__agg_2e_tmp1.__FIXME__l_struct_struct_OC_dim3_field1 = 1;
  __FIXME__agg_2e_tmp1.__FIXME__l_struct_struct_OC_dim3_field2 = 1;
  memcpy(((uint8_t*)(&__FIXME__agg_2e_tmp_2e_coerce)), ((uint8_t*)(&__FIXME__agg_2e_tmp)), 12);
  memcpy(((uint8_t*)(&__FIXME__agg_2e_tmp1_2e_coerce)), ((uint8_t*)(&__FIXME__agg_2e_tmp1)), 12);
// INSERT COMMENT LOOP: init_ui_gpu::header.0
#pragma omp parallel for collapse(2)
for(int32_t i = 0; i < blocks_per_grid_on_init_ui;   i = i + 1) {
for(int32_t j = 0; j < threads_per_block_on_init_ui;   j = j + 1) {
init_ui_gpu_kernel(u0, u1, twiddle, blocks_per_grid_on_init_ui, 1, 1, threads_per_block_on_init_ui, 1, 1, i, 0, 0, j, 0, 0);
}
}
  return;
}
// FUNCTION ORDER ID 9 END


// FUNCTION ORDER ID 10 START
// INSERT COMMENT FUNCTION: compute_indexmap_gpu
void compute_indexmap_gpu(double* twiddle) {
  struct __FIXME__l_struct_struct_OC_dim3 __FIXME__agg_2e_tmp;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dim3 __FIXME__agg_2e_tmp1;    /* Address-exposed local */
  struct __FIXME__l_unnamed_1 __FIXME__agg_2e_tmp_2e_coerce;    /* Address-exposed local */
  struct __FIXME__l_unnamed_1 __FIXME__agg_2e_tmp1_2e_coerce;    /* Address-exposed local */
  uint32_t i;
  uint32_t j;

  __FIXME__agg_2e_tmp.__FIXME__l_struct_struct_OC_dim3_field0 = blocks_per_grid_on_compute_indexmap;
  __FIXME__agg_2e_tmp.__FIXME__l_struct_struct_OC_dim3_field1 = 1;
  __FIXME__agg_2e_tmp.__FIXME__l_struct_struct_OC_dim3_field2 = 1;
  __FIXME__agg_2e_tmp1.__FIXME__l_struct_struct_OC_dim3_field0 = threads_per_block_on_compute_indexmap;
  __FIXME__agg_2e_tmp1.__FIXME__l_struct_struct_OC_dim3_field1 = 1;
  __FIXME__agg_2e_tmp1.__FIXME__l_struct_struct_OC_dim3_field2 = 1;
  memcpy(((uint8_t*)(&__FIXME__agg_2e_tmp_2e_coerce)), ((uint8_t*)(&__FIXME__agg_2e_tmp)), 12);
  memcpy(((uint8_t*)(&__FIXME__agg_2e_tmp1_2e_coerce)), ((uint8_t*)(&__FIXME__agg_2e_tmp1)), 12);
// INSERT COMMENT LOOP: compute_indexmap_gpu::header.0
#pragma omp parallel for collapse(2)
for(int32_t i = 0; i < blocks_per_grid_on_compute_indexmap;   i = i + 1) {
for(int32_t j = 0; j < threads_per_block_on_compute_indexmap;   j = j + 1) {
compute_indexmap_gpu_kernel(twiddle, blocks_per_grid_on_compute_indexmap, 1, 1, threads_per_block_on_compute_indexmap, 1, 1, i, 0, 0, j, 0, 0);
}
}
  return;
}
// FUNCTION ORDER ID 10 END


// FUNCTION ORDER ID 11 START
// INSERT COMMENT FUNCTION: compute_initial_conditions_gpu
void compute_initial_conditions_gpu(struct __FIXME__l_struct_struct_OC_dcomplex* u0) {
  double start;    /* Address-exposed local */
  double an;    /* Address-exposed local */
  double starts[128];    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dim3 __FIXME__agg_2e_tmp;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dim3 __FIXME__agg_2e_tmp4;    /* Address-exposed local */
  struct __FIXME__l_unnamed_1 __FIXME__agg_2e_tmp_2e_coerce;    /* Address-exposed local */
  struct __FIXME__l_unnamed_1 __FIXME__agg_2e_tmp4_2e_coerce;    /* Address-exposed local */
  int64_t z;
  uint32_t i;
  uint32_t j;

  start = 314159265;
ipow46(1220703125, 0, (&an));
  randlc((&start), an);
ipow46(1220703125, 131072, (&an));
  starts[0] = start;
// INSERT COMMENT LOOP: compute_initial_conditions_gpu::for.cond
for(int64_t z = 1; z < 128;   z = z + 1) {
  randlc((&start), an);
  starts[z] = start;
}
  ;
  __FIXME__agg_2e_tmp.__FIXME__l_struct_struct_OC_dim3_field0 = blocks_per_grid_on_compute_initial_conditions;
  __FIXME__agg_2e_tmp.__FIXME__l_struct_struct_OC_dim3_field1 = 1;
  __FIXME__agg_2e_tmp.__FIXME__l_struct_struct_OC_dim3_field2 = 1;
  __FIXME__agg_2e_tmp4.__FIXME__l_struct_struct_OC_dim3_field0 = threads_per_block_on_compute_initial_conditions;
  __FIXME__agg_2e_tmp4.__FIXME__l_struct_struct_OC_dim3_field1 = 1;
  __FIXME__agg_2e_tmp4.__FIXME__l_struct_struct_OC_dim3_field2 = 1;
  memcpy(((uint8_t*)(&__FIXME__agg_2e_tmp_2e_coerce)), ((uint8_t*)(&__FIXME__agg_2e_tmp)), 12);
  memcpy(((uint8_t*)(&__FIXME__agg_2e_tmp4_2e_coerce)), ((uint8_t*)(&__FIXME__agg_2e_tmp4)), 12);
// INSERT COMMENT LOOP: compute_initial_conditions_gpu::header.0
#pragma omp parallel for collapse(2)
for(int32_t i = 0; i < blocks_per_grid_on_compute_initial_conditions;   i = i + 1) {
for(int32_t j = 0; j < threads_per_block_on_compute_initial_conditions;   j = j + 1) {
compute_initial_conditions_gpu_kernel(u0, starts, blocks_per_grid_on_compute_initial_conditions, 1, 1, threads_per_block_on_compute_initial_conditions, 1, 1, i, 0, 0, j, 0, 0);
}
}
  return;
}
// FUNCTION ORDER ID 11 END


// FUNCTION ORDER ID 12 START
// INSERT COMMENT FUNCTION: fft_init_gpu
void fft_init_gpu(uint32_t n) {
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp6;    /* Address-exposed local */
  int32_t m;
  uint32_t ku;
  int32_t j;
  uint32_t ln;
  int64_t i;

  m = ilog2(n);
  __FIXME__ref_2e_tmp.__FIXME__l_struct_struct_OC_dcomplex_field0 = ((double)((int32_t)m));
  __FIXME__ref_2e_tmp.__FIXME__l_struct_struct_OC_dcomplex_field1 = 0;
  memcpy(((uint8_t*)_ZL1u), ((uint8_t*)(&__FIXME__ref_2e_tmp)), 16);
  ku = 2;
  ln = 1;
// INSERT COMMENT LOOP: fft_init_gpu::for.cond
for(int32_t j = 1; j <= m;   j = j + 1) {
  double t = (3.1415926535897931 / ((double)((int32_t)ln)));
for(int64_t i = 0; i <= (ln - 1);   i = i + 1) {
  double ti = (((double)((int32_t)i)) * t);
  double __FIXME__call8 = cos(ti);
  __FIXME__ref_2e_tmp6.__FIXME__l_struct_struct_OC_dcomplex_field0 = __FIXME__call8;
  double __FIXME__call10 = sin(ti);
  __FIXME__ref_2e_tmp6.__FIXME__l_struct_struct_OC_dcomplex_field1 = __FIXME__call10;
  memcpy(((uint8_t*)(_ZL1u+((i + ku) - 1))), ((uint8_t*)(&__FIXME__ref_2e_tmp6)), 16);
}
  ku = ku + ln;
  ln = 2 * ln;
}
  ;
}
// FUNCTION ORDER ID 12 END


// FUNCTION ORDER ID 13 START
// INSERT COMMENT FUNCTION: fft_gpu
void fft_gpu(uint32_t dir, struct __FIXME__l_struct_struct_OC_dcomplex* x1, struct __FIXME__l_struct_struct_OC_dcomplex* x2) {
// INSERT COMMENT IFELSE: fft_gpu::entry
  if (dir == 1) { // IFELSE MARKER: entry IF
cffts1_gpu(1, _ZL1u, x1, x1, y0_device, y1_device);
cffts2_gpu(1, _ZL1u, x1, x1, y0_device, y1_device);
cffts3_gpu(1, _ZL1u, x1, x2, y0_device, y1_device);
  } else { // IFELSE MARKER: entry ELSE
cffts3_gpu(-1, _ZL1u, x1, x1, y0_device, y1_device);
cffts2_gpu(-1, _ZL1u, x1, x1, y0_device, y1_device);
cffts1_gpu(-1, _ZL1u, x1, x2, y0_device, y1_device);
  }
  return;
}
// FUNCTION ORDER ID 13 END


// FUNCTION ORDER ID 14 START
// INSERT COMMENT FUNCTION: evolve_gpu
void evolve_gpu(struct __FIXME__l_struct_struct_OC_dcomplex* u0, struct __FIXME__l_struct_struct_OC_dcomplex* u1, double* twiddle) {
  struct __FIXME__l_struct_struct_OC_dim3 __FIXME__agg_2e_tmp;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dim3 __FIXME__agg_2e_tmp1;    /* Address-exposed local */
  struct __FIXME__l_unnamed_1 __FIXME__agg_2e_tmp_2e_coerce;    /* Address-exposed local */
  struct __FIXME__l_unnamed_1 __FIXME__agg_2e_tmp1_2e_coerce;    /* Address-exposed local */
  uint32_t i;
  uint32_t j;

  __FIXME__agg_2e_tmp.__FIXME__l_struct_struct_OC_dim3_field0 = blocks_per_grid_on_evolve;
  __FIXME__agg_2e_tmp.__FIXME__l_struct_struct_OC_dim3_field1 = 1;
  __FIXME__agg_2e_tmp.__FIXME__l_struct_struct_OC_dim3_field2 = 1;
  __FIXME__agg_2e_tmp1.__FIXME__l_struct_struct_OC_dim3_field0 = threads_per_block_on_evolve;
  __FIXME__agg_2e_tmp1.__FIXME__l_struct_struct_OC_dim3_field1 = 1;
  __FIXME__agg_2e_tmp1.__FIXME__l_struct_struct_OC_dim3_field2 = 1;
  memcpy(((uint8_t*)(&__FIXME__agg_2e_tmp_2e_coerce)), ((uint8_t*)(&__FIXME__agg_2e_tmp)), 12);
  memcpy(((uint8_t*)(&__FIXME__agg_2e_tmp1_2e_coerce)), ((uint8_t*)(&__FIXME__agg_2e_tmp1)), 12);
// INSERT COMMENT LOOP: evolve_gpu::header.0
#pragma omp parallel for collapse(2)
for(int32_t i = 0; i < blocks_per_grid_on_evolve;   i = i + 1) {
for(int32_t j = 0; j < threads_per_block_on_evolve;   j = j + 1) {
evolve_gpu_kernel(u0, u1, twiddle, blocks_per_grid_on_evolve, 1, 1, threads_per_block_on_evolve, 1, 1, i, 0, 0, j, 0, 0);
}
}
  return;
}
// FUNCTION ORDER ID 14 END


// FUNCTION ORDER ID 15 START
// INSERT COMMENT FUNCTION: checksum_gpu
void checksum_gpu(uint32_t iteration, struct __FIXME__l_struct_struct_OC_dcomplex* u1) {
  struct __FIXME__l_struct_struct_OC_dim3 __FIXME__agg_2e_tmp;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dim3 __FIXME__agg_2e_tmp1;    /* Address-exposed local */
  struct __FIXME__l_unnamed_1 __FIXME__agg_2e_tmp_2e_coerce;    /* Address-exposed local */
  struct __FIXME__l_unnamed_1 __FIXME__agg_2e_tmp1_2e_coerce;    /* Address-exposed local */
  uint32_t i;
  uint32_t j;

  __FIXME__agg_2e_tmp.__FIXME__l_struct_struct_OC_dim3_field0 = blocks_per_grid_on_checksum;
  __FIXME__agg_2e_tmp.__FIXME__l_struct_struct_OC_dim3_field1 = 1;
  __FIXME__agg_2e_tmp.__FIXME__l_struct_struct_OC_dim3_field2 = 1;
  __FIXME__agg_2e_tmp1.__FIXME__l_struct_struct_OC_dim3_field0 = threads_per_block_on_checksum;
  __FIXME__agg_2e_tmp1.__FIXME__l_struct_struct_OC_dim3_field1 = 1;
  __FIXME__agg_2e_tmp1.__FIXME__l_struct_struct_OC_dim3_field2 = 1;
  memcpy(((uint8_t*)(&__FIXME__agg_2e_tmp_2e_coerce)), ((uint8_t*)(&__FIXME__agg_2e_tmp)), 12);
  memcpy(((uint8_t*)(&__FIXME__agg_2e_tmp1_2e_coerce)), ((uint8_t*)(&__FIXME__agg_2e_tmp1)), 12);
// INSERT COMMENT LOOP: checksum_gpu::header.0
#pragma omp parallel for
for(int32_t i = 0; i < blocks_per_grid_on_checksum;   i = i + 1) {
for(int32_t j = 0; j < threads_per_block_on_checksum;   j = j + 1) {
checksum_gpu_kernel0(iteration, u1, _ZL4sums, blocks_per_grid_on_checksum, 1, 1, threads_per_block_on_checksum, 1, 1, i, 0, 0, j, 0, 0);
}
for(int32_t j = 0; j < threads_per_block_on_checksum;   j = j + 1) {
checksum_gpu_kernel1(iteration, u1, _ZL4sums, blocks_per_grid_on_checksum, 1, 1, threads_per_block_on_checksum, 1, 1, i, 0, 0, j, 0, 0);
}
}
  return;
}
// FUNCTION ORDER ID 15 END


// FUNCTION ORDER ID 16 START
// INSERT COMMENT FUNCTION: verify
void verify(uint32_t d1, uint32_t d2, uint32_t d3, uint32_t nt, uint32_t* verified, uint8_t* class_npb) {
  struct __FIXME__l_struct_struct_OC_dcomplex csum_ref[26];    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp6;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp10;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp14;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp18;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp22;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp34;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp38;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp42;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp46;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp50;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp54;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp67;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp71;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp75;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp79;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp83;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp87;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp100;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp104;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp108;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp112;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp116;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp120;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp124;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp128;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp132;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp136;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp140;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp144;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp148;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp152;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp156;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp160;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp164;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp168;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp172;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp176;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp189;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp193;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp197;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp201;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp205;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp209;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp213;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp217;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp221;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp225;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp229;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp233;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp237;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp241;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp245;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp249;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp253;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp257;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp261;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp265;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp278;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp282;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp286;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp290;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp294;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp298;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp302;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp306;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp310;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp314;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp318;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp322;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp326;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp330;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp334;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp338;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp342;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp346;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp350;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp354;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp358;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp362;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp366;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp370;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp374;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp387;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp391;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp395;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp399;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp403;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp407;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp411;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp415;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp419;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp423;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp427;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp431;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp435;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp439;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp443;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp447;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp451;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp455;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp459;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp463;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp467;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp471;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp475;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp479;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp483;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__agg_2e_tmp;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__agg_2e_tmp510;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__coerce;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__agg_2e_tmp514;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__agg_2e_tmp531;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__coerce535;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__agg_2e_tmp537;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__agg_2e_tmp554;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__coerce558;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__agg_2e_tmp560;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__agg_2e_tmp577;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__coerce581;    /* Address-exposed local */
  int64_t i;

// INSERT COMMENT IFELSE: verify::entry
  *class_npb = 85;
  *verified = 0;
  if (d1 == 64) { // IFELSE MARKER: entry IF
  if (d2 == 64) { // IFELSE MARKER: land.lhs.true IF
  if (d3 == 64) { // IFELSE MARKER: land.lhs.true2 IF
  if (nt == 6) { // IFELSE MARKER: land.lhs.true4 IF
  *class_npb = 83;
  __FIXME__ref_2e_tmp.__FIXME__l_struct_struct_OC_dcomplex_field0 = 554.60870049640005;
  __FIXME__ref_2e_tmp.__FIXME__l_struct_struct_OC_dcomplex_field1 = 484.53633319779999;
  memcpy(((uint8_t*)&csum_ref[1]), ((uint8_t*)(&__FIXME__ref_2e_tmp)), 16);
  __FIXME__ref_2e_tmp6.__FIXME__l_struct_struct_OC_dcomplex_field0 = 554.63854091890005;
  __FIXME__ref_2e_tmp6.__FIXME__l_struct_struct_OC_dcomplex_field1 = 486.53042695110003;
  memcpy(((uint8_t*)&csum_ref[2]), ((uint8_t*)(&__FIXME__ref_2e_tmp6)), 16);
  __FIXME__ref_2e_tmp10.__FIXME__l_struct_struct_OC_dcomplex_field0 = 554.61484061709996;
  __FIXME__ref_2e_tmp10.__FIXME__l_struct_struct_OC_dcomplex_field1 = 488.39107223360003;
  memcpy(((uint8_t*)&csum_ref[3]), ((uint8_t*)(&__FIXME__ref_2e_tmp10)), 16);
  __FIXME__ref_2e_tmp14.__FIXME__l_struct_struct_OC_dcomplex_field0 = 554.54236074150003;
  __FIXME__ref_2e_tmp14.__FIXME__l_struct_struct_OC_dcomplex_field1 = 490.12731690459998;
  memcpy(((uint8_t*)&csum_ref[4]), ((uint8_t*)(&__FIXME__ref_2e_tmp14)), 16);
  __FIXME__ref_2e_tmp18.__FIXME__l_struct_struct_OC_dcomplex_field0 = 554.42550396239994;
  __FIXME__ref_2e_tmp18.__FIXME__l_struct_struct_OC_dcomplex_field1 = 491.7475857993;
  memcpy(((uint8_t*)&csum_ref[5]), ((uint8_t*)(&__FIXME__ref_2e_tmp18)), 16);
  __FIXME__ref_2e_tmp22.__FIXME__l_struct_struct_OC_dcomplex_field0 = 554.26834119019998;
  __FIXME__ref_2e_tmp22.__FIXME__l_struct_struct_OC_dcomplex_field1 = 493.2597244941;
  memcpy(((uint8_t*)&csum_ref[6]), ((uint8_t*)(&__FIXME__ref_2e_tmp22)), 16);
  }
  }
  }
  } else { // IFELSE MARKER: entry ELSE
  if (d1 == 128) { // IFELSE MARKER: if.else IF
  if (d2 == 128) { // IFELSE MARKER: land.lhs.true27 IF
  if (d3 == 32) { // IFELSE MARKER: land.lhs.true29 IF
  if (nt == 6) { // IFELSE MARKER: land.lhs.true31 IF
  *class_npb = 87;
  __FIXME__ref_2e_tmp34.__FIXME__l_struct_struct_OC_dcomplex_field0 = 567.36121789440006;
  __FIXME__ref_2e_tmp34.__FIXME__l_struct_struct_OC_dcomplex_field1 = 529.32468491750001;
  memcpy(((uint8_t*)&csum_ref[1]), ((uint8_t*)(&__FIXME__ref_2e_tmp34)), 16);
  __FIXME__ref_2e_tmp38.__FIXME__l_struct_struct_OC_dcomplex_field0 = 563.14368852710004;
  __FIXME__ref_2e_tmp38.__FIXME__l_struct_struct_OC_dcomplex_field1 = 528.21499866290003;
  memcpy(((uint8_t*)&csum_ref[2]), ((uint8_t*)(&__FIXME__ref_2e_tmp38)), 16);
  __FIXME__ref_2e_tmp42.__FIXME__l_struct_struct_OC_dcomplex_field0 = 559.40240899699995;
  __FIXME__ref_2e_tmp42.__FIXME__l_struct_struct_OC_dcomplex_field1 = 527.09965580369999;
  memcpy(((uint8_t*)&csum_ref[3]), ((uint8_t*)(&__FIXME__ref_2e_tmp42)), 16);
  __FIXME__ref_2e_tmp46.__FIXME__l_struct_struct_OC_dcomplex_field0 = 556.06980470200006;
  __FIXME__ref_2e_tmp46.__FIXME__l_struct_struct_OC_dcomplex_field1 = 526.00279049250003;
  memcpy(((uint8_t*)&csum_ref[4]), ((uint8_t*)(&__FIXME__ref_2e_tmp46)), 16);
  __FIXME__ref_2e_tmp50.__FIXME__l_struct_struct_OC_dcomplex_field0 = 553.08989912499999;
  __FIXME__ref_2e_tmp50.__FIXME__l_struct_struct_OC_dcomplex_field1 = 524.94008456330005;
  memcpy(((uint8_t*)&csum_ref[5]), ((uint8_t*)(&__FIXME__ref_2e_tmp50)), 16);
  __FIXME__ref_2e_tmp54.__FIXME__l_struct_struct_OC_dcomplex_field0 = 550.41597345380001;
  __FIXME__ref_2e_tmp54.__FIXME__l_struct_struct_OC_dcomplex_field1 = 523.92122470859999;
  memcpy(((uint8_t*)&csum_ref[6]), ((uint8_t*)(&__FIXME__ref_2e_tmp54)), 16);
  }
  }
  }
  } else { // IFELSE MARKER: if.else ELSE
  if (d1 == 256) { // IFELSE MARKER: if.else58 IF
  if (d2 == 256) { // IFELSE MARKER: land.lhs.true60 IF
  if (d3 == 128) { // IFELSE MARKER: land.lhs.true62 IF
  if (nt == 6) { // IFELSE MARKER: land.lhs.true64 IF
  *class_npb = 65;
  __FIXME__ref_2e_tmp67.__FIXME__l_struct_struct_OC_dcomplex_field0 = 504.67350081929999;
  __FIXME__ref_2e_tmp67.__FIXME__l_struct_struct_OC_dcomplex_field1 = 511.40479055100002;
  memcpy(((uint8_t*)&csum_ref[1]), ((uint8_t*)(&__FIXME__ref_2e_tmp67)), 16);
  __FIXME__ref_2e_tmp71.__FIXME__l_struct_struct_OC_dcomplex_field0 = 505.94123197340002;
  __FIXME__ref_2e_tmp71.__FIXME__l_struct_struct_OC_dcomplex_field1 = 509.88096664329998;
  memcpy(((uint8_t*)&csum_ref[2]), ((uint8_t*)(&__FIXME__ref_2e_tmp71)), 16);
  __FIXME__ref_2e_tmp75.__FIXME__l_struct_struct_OC_dcomplex_field0 = 506.93768962870001;
  __FIXME__ref_2e_tmp75.__FIXME__l_struct_struct_OC_dcomplex_field1 = 509.81440422129998;
  memcpy(((uint8_t*)&csum_ref[3]), ((uint8_t*)(&__FIXME__ref_2e_tmp75)), 16);
  __FIXME__ref_2e_tmp79.__FIXME__l_struct_struct_OC_dcomplex_field0 = 507.78928684739998;
  __FIXME__ref_2e_tmp79.__FIXME__l_struct_struct_OC_dcomplex_field1 = 510.13361307589997;
  memcpy(((uint8_t*)&csum_ref[4]), ((uint8_t*)(&__FIXME__ref_2e_tmp79)), 16);
  __FIXME__ref_2e_tmp83.__FIXME__l_struct_struct_OC_dcomplex_field0 = 508.52330953910001;
  __FIXME__ref_2e_tmp83.__FIXME__l_struct_struct_OC_dcomplex_field1 = 510.49146551939998;
  memcpy(((uint8_t*)&csum_ref[5]), ((uint8_t*)(&__FIXME__ref_2e_tmp83)), 16);
  __FIXME__ref_2e_tmp87.__FIXME__l_struct_struct_OC_dcomplex_field0 = 509.14870999589999;
  __FIXME__ref_2e_tmp87.__FIXME__l_struct_struct_OC_dcomplex_field1 = 510.79178428030002;
  memcpy(((uint8_t*)&csum_ref[6]), ((uint8_t*)(&__FIXME__ref_2e_tmp87)), 16);
  }
  }
  }
  } else { // IFELSE MARKER: if.else58 ELSE
  if (d1 == 512) { // IFELSE MARKER: if.else91 IF
  if (d2 == 256) { // IFELSE MARKER: land.lhs.true93 IF
  if (d3 == 256) { // IFELSE MARKER: land.lhs.true95 IF
  if (nt == 20) { // IFELSE MARKER: land.lhs.true97 IF
  *class_npb = 66;
  __FIXME__ref_2e_tmp100.__FIXME__l_struct_struct_OC_dcomplex_field0 = 517.76435715790001;
  __FIXME__ref_2e_tmp100.__FIXME__l_struct_struct_OC_dcomplex_field1 = 507.78034585969999;
  memcpy(((uint8_t*)&csum_ref[1]), ((uint8_t*)(&__FIXME__ref_2e_tmp100)), 16);
  __FIXME__ref_2e_tmp104.__FIXME__l_struct_struct_OC_dcomplex_field0 = 515.45212912629995;
  __FIXME__ref_2e_tmp104.__FIXME__l_struct_struct_OC_dcomplex_field1 = 508.82494315989999;
  memcpy(((uint8_t*)&csum_ref[2]), ((uint8_t*)(&__FIXME__ref_2e_tmp104)), 16);
  __FIXME__ref_2e_tmp108.__FIXME__l_struct_struct_OC_dcomplex_field0 = 514.64092286489995;
  __FIXME__ref_2e_tmp108.__FIXME__l_struct_struct_OC_dcomplex_field1 = 509.62089126590001;
  memcpy(((uint8_t*)&csum_ref[3]), ((uint8_t*)(&__FIXME__ref_2e_tmp108)), 16);
  __FIXME__ref_2e_tmp112.__FIXME__l_struct_struct_OC_dcomplex_field0 = 514.23787562129996;
  __FIXME__ref_2e_tmp112.__FIXME__l_struct_struct_OC_dcomplex_field1 = 510.1023387619;
  memcpy(((uint8_t*)&csum_ref[4]), ((uint8_t*)(&__FIXME__ref_2e_tmp112)), 16);
  __FIXME__ref_2e_tmp116.__FIXME__l_struct_struct_OC_dcomplex_field0 = 513.96266677369999;
  __FIXME__ref_2e_tmp116.__FIXME__l_struct_struct_OC_dcomplex_field1 = 510.39766106169998;
  memcpy(((uint8_t*)&csum_ref[5]), ((uint8_t*)(&__FIXME__ref_2e_tmp116)), 16);
  __FIXME__ref_2e_tmp120.__FIXME__l_struct_struct_OC_dcomplex_field0 = 513.7423460082;
  __FIXME__ref_2e_tmp120.__FIXME__l_struct_struct_OC_dcomplex_field1 = 510.59480198019997;
  memcpy(((uint8_t*)&csum_ref[6]), ((uint8_t*)(&__FIXME__ref_2e_tmp120)), 16);
  __FIXME__ref_2e_tmp124.__FIXME__l_struct_struct_OC_dcomplex_field0 = 513.55470568780004;
  __FIXME__ref_2e_tmp124.__FIXME__l_struct_struct_OC_dcomplex_field1 = 510.74041657830003;
  memcpy(((uint8_t*)&csum_ref[7]), ((uint8_t*)(&__FIXME__ref_2e_tmp124)), 16);
  __FIXME__ref_2e_tmp128.__FIXME__l_struct_struct_OC_dcomplex_field0 = 513.39109254660002;
  __FIXME__ref_2e_tmp128.__FIXME__l_struct_struct_OC_dcomplex_field1 = 510.8576573661;
  memcpy(((uint8_t*)&csum_ref[8]), ((uint8_t*)(&__FIXME__ref_2e_tmp128)), 16);
  __FIXME__ref_2e_tmp132.__FIXME__l_struct_struct_OC_dcomplex_field0 = 513.24707053899999;
  __FIXME__ref_2e_tmp132.__FIXME__l_struct_struct_OC_dcomplex_field1 = 510.95772785230002;
  memcpy(((uint8_t*)&csum_ref[9]), ((uint8_t*)(&__FIXME__ref_2e_tmp132)), 16);
  __FIXME__ref_2e_tmp136.__FIXME__l_struct_struct_OC_dcomplex_field0 = 513.11977299839998;
  __FIXME__ref_2e_tmp136.__FIXME__l_struct_struct_OC_dcomplex_field1 = 511.04603044829997;
  memcpy(((uint8_t*)&csum_ref[10]), ((uint8_t*)(&__FIXME__ref_2e_tmp136)), 16);
  __FIXME__ref_2e_tmp140.__FIXME__l_struct_struct_OC_dcomplex_field0 = 513.00703192829997;
  __FIXME__ref_2e_tmp140.__FIXME__l_struct_struct_OC_dcomplex_field1 = 511.12524337999997;
  memcpy(((uint8_t*)&csum_ref[11]), ((uint8_t*)(&__FIXME__ref_2e_tmp140)), 16);
  __FIXME__ref_2e_tmp144.__FIXME__l_struct_struct_OC_dcomplex_field0 = 512.90705370319995;
  __FIXME__ref_2e_tmp144.__FIXME__l_struct_struct_OC_dcomplex_field1 = 511.19680777180002;
  memcpy(((uint8_t*)&csum_ref[12]), ((uint8_t*)(&__FIXME__ref_2e_tmp144)), 16);
  __FIXME__ref_2e_tmp148.__FIXME__l_struct_struct_OC_dcomplex_field0 = 512.81828835019996;
  __FIXME__ref_2e_tmp148.__FIXME__l_struct_struct_OC_dcomplex_field1 = 511.2616233064;
  memcpy(((uint8_t*)&csum_ref[13]), ((uint8_t*)(&__FIXME__ref_2e_tmp148)), 16);
  __FIXME__ref_2e_tmp152.__FIXME__l_struct_struct_OC_dcomplex_field0 = 512.73937333829997;
  __FIXME__ref_2e_tmp152.__FIXME__l_struct_struct_OC_dcomplex_field1 = 511.32036055510002;
  memcpy(((uint8_t*)&csum_ref[14]), ((uint8_t*)(&__FIXME__ref_2e_tmp152)), 16);
  __FIXME__ref_2e_tmp156.__FIXME__l_struct_struct_OC_dcomplex_field0 = 512.66910620199997;
  __FIXME__ref_2e_tmp156.__FIXME__l_struct_struct_OC_dcomplex_field1 = 511.37359280930002;
  memcpy(((uint8_t*)&csum_ref[15]), ((uint8_t*)(&__FIXME__ref_2e_tmp156)), 16);
  __FIXME__ref_2e_tmp160.__FIXME__l_struct_struct_OC_dcomplex_field0 = 512.60642760040002;
  __FIXME__ref_2e_tmp160.__FIXME__l_struct_struct_OC_dcomplex_field1 = 511.42184605480003;
  memcpy(((uint8_t*)&csum_ref[16]), ((uint8_t*)(&__FIXME__ref_2e_tmp160)), 16);
  __FIXME__ref_2e_tmp164.__FIXME__l_struct_struct_OC_dcomplex_field0 = 512.55040765700005;
  __FIXME__ref_2e_tmp164.__FIXME__l_struct_struct_OC_dcomplex_field1 = 511.46561397599999;
  memcpy(((uint8_t*)&csum_ref[17]), ((uint8_t*)(&__FIXME__ref_2e_tmp164)), 16);
  __FIXME__ref_2e_tmp168.__FIXME__l_struct_struct_OC_dcomplex_field0 = 512.50023317199998;
  __FIXME__ref_2e_tmp168.__FIXME__l_struct_struct_OC_dcomplex_field1 = 511.50535959659999;
  memcpy(((uint8_t*)&csum_ref[18]), ((uint8_t*)(&__FIXME__ref_2e_tmp168)), 16);
  __FIXME__ref_2e_tmp172.__FIXME__l_struct_struct_OC_dcomplex_field0 = 512.45519518460003;
  __FIXME__ref_2e_tmp172.__FIXME__l_struct_struct_OC_dcomplex_field1 = 511.54151304070001;
  memcpy(((uint8_t*)&csum_ref[19]), ((uint8_t*)(&__FIXME__ref_2e_tmp172)), 16);
  __FIXME__ref_2e_tmp176.__FIXME__l_struct_struct_OC_dcomplex_field0 = 512.41467700290002;
  __FIXME__ref_2e_tmp176.__FIXME__l_struct_struct_OC_dcomplex_field1 = 511.5744692211;
  memcpy(((uint8_t*)&csum_ref[20]), ((uint8_t*)(&__FIXME__ref_2e_tmp176)), 16);
  }
  }
  }
  } else { // IFELSE MARKER: if.else91 ELSE
  if (d1 == 512) { // IFELSE MARKER: if.else180 IF
  if (d2 == 512) { // IFELSE MARKER: land.lhs.true182 IF
  if (d3 == 512) { // IFELSE MARKER: land.lhs.true184 IF
  if (nt == 20) { // IFELSE MARKER: land.lhs.true186 IF
  *class_npb = 67;
  __FIXME__ref_2e_tmp189.__FIXME__l_struct_struct_OC_dcomplex_field0 = 519.50787074569996;
  __FIXME__ref_2e_tmp189.__FIXME__l_struct_struct_OC_dcomplex_field1 = 514.90196992380004;
  memcpy(((uint8_t*)&csum_ref[1]), ((uint8_t*)(&__FIXME__ref_2e_tmp189)), 16);
  __FIXME__ref_2e_tmp193.__FIXME__l_struct_struct_OC_dcomplex_field0 = 515.54221711340006;
  __FIXME__ref_2e_tmp193.__FIXME__l_struct_struct_OC_dcomplex_field1 = 512.75782019969995;
  memcpy(((uint8_t*)&csum_ref[2]), ((uint8_t*)(&__FIXME__ref_2e_tmp193)), 16);
  __FIXME__ref_2e_tmp197.__FIXME__l_struct_struct_OC_dcomplex_field0 = 514.46780222220002;
  __FIXME__ref_2e_tmp197.__FIXME__l_struct_struct_OC_dcomplex_field1 = 512.22518475139998;
  memcpy(((uint8_t*)&csum_ref[3]), ((uint8_t*)(&__FIXME__ref_2e_tmp197)), 16);
  __FIXME__ref_2e_tmp201.__FIXME__l_struct_struct_OC_dcomplex_field0 = 514.01505943279994;
  __FIXME__ref_2e_tmp201.__FIXME__l_struct_struct_OC_dcomplex_field1 = 512.1090289018;
  memcpy(((uint8_t*)&csum_ref[4]), ((uint8_t*)(&__FIXME__ref_2e_tmp201)), 16);
  __FIXME__ref_2e_tmp205.__FIXME__l_struct_struct_OC_dcomplex_field0 = 513.75504268099996;
  __FIXME__ref_2e_tmp205.__FIXME__l_struct_struct_OC_dcomplex_field1 = 512.11436858239995;
  memcpy(((uint8_t*)&csum_ref[5]), ((uint8_t*)(&__FIXME__ref_2e_tmp205)), 16);
  __FIXME__ref_2e_tmp209.__FIXME__l_struct_struct_OC_dcomplex_field0 = 513.58110567280005;
  __FIXME__ref_2e_tmp209.__FIXME__l_struct_struct_OC_dcomplex_field1 = 512.14967645679997;
  memcpy(((uint8_t*)&csum_ref[6]), ((uint8_t*)(&__FIXME__ref_2e_tmp209)), 16);
  __FIXME__ref_2e_tmp213.__FIXME__l_struct_struct_OC_dcomplex_field0 = 513.4569343165;
  __FIXME__ref_2e_tmp213.__FIXME__l_struct_struct_OC_dcomplex_field1 = 512.18709218929996;
  memcpy(((uint8_t*)&csum_ref[7]), ((uint8_t*)(&__FIXME__ref_2e_tmp213)), 16);
  __FIXME__ref_2e_tmp217.__FIXME__l_struct_struct_OC_dcomplex_field0 = 513.36519756610005;
  __FIXME__ref_2e_tmp217.__FIXME__l_struct_struct_OC_dcomplex_field1 = 512.2193250322;
  memcpy(((uint8_t*)&csum_ref[8]), ((uint8_t*)(&__FIXME__ref_2e_tmp217)), 16);
  __FIXME__ref_2e_tmp221.__FIXME__l_struct_struct_OC_dcomplex_field0 = 513.2955192805;
  __FIXME__ref_2e_tmp221.__FIXME__l_struct_struct_OC_dcomplex_field1 = 512.24547357940003;
  memcpy(((uint8_t*)&csum_ref[9]), ((uint8_t*)(&__FIXME__ref_2e_tmp221)), 16);
  __FIXME__ref_2e_tmp225.__FIXME__l_struct_struct_OC_dcomplex_field0 = 513.24104717379998;
  __FIXME__ref_2e_tmp225.__FIXME__l_struct_struct_OC_dcomplex_field1 = 512.26636496030005;
  memcpy(((uint8_t*)&csum_ref[10]), ((uint8_t*)(&__FIXME__ref_2e_tmp225)), 16);
  __FIXME__ref_2e_tmp229.__FIXME__l_struct_struct_OC_dcomplex_field0 = 513.19711416790005;
  __FIXME__ref_2e_tmp229.__FIXME__l_struct_struct_OC_dcomplex_field1 = 512.28308798269995;
  memcpy(((uint8_t*)&csum_ref[11]), ((uint8_t*)(&__FIXME__ref_2e_tmp229)), 16);
  __FIXME__ref_2e_tmp233.__FIXME__l_struct_struct_OC_dcomplex_field0 = 513.16052057160005;
  __FIXME__ref_2e_tmp233.__FIXME__l_struct_struct_OC_dcomplex_field1 = 512.29658697180002;
  memcpy(((uint8_t*)&csum_ref[12]), ((uint8_t*)(&__FIXME__ref_2e_tmp233)), 16);
  __FIXME__ref_2e_tmp237.__FIXME__l_struct_struct_OC_dcomplex_field0 = 513.12907341940002;
  __FIXME__ref_2e_tmp237.__FIXME__l_struct_struct_OC_dcomplex_field1 = 512.30759274449997;
  memcpy(((uint8_t*)&csum_ref[13]), ((uint8_t*)(&__FIXME__ref_2e_tmp237)), 16);
  __FIXME__ref_2e_tmp241.__FIXME__l_struct_struct_OC_dcomplex_field0 = 513.10127203139996;
  __FIXME__ref_2e_tmp241.__FIXME__l_struct_struct_OC_dcomplex_field1 = 512.31664865530001;
  memcpy(((uint8_t*)&csum_ref[14]), ((uint8_t*)(&__FIXME__ref_2e_tmp241)), 16);
  __FIXME__ref_2e_tmp245.__FIXME__l_struct_struct_OC_dcomplex_field0 = 513.0760908195;
  __FIXME__ref_2e_tmp245.__FIXME__l_struct_struct_OC_dcomplex_field1 = 512.32415416849994;
  memcpy(((uint8_t*)&csum_ref[15]), ((uint8_t*)(&__FIXME__ref_2e_tmp245)), 16);
  __FIXME__ref_2e_tmp249.__FIXME__l_struct_struct_OC_dcomplex_field0 = 513.05282959229999;
  __FIXME__ref_2e_tmp249.__FIXME__l_struct_struct_OC_dcomplex_field1 = 512.33040375990004;
  memcpy(((uint8_t*)&csum_ref[16]), ((uint8_t*)(&__FIXME__ref_2e_tmp249)), 16);
  __FIXME__ref_2e_tmp253.__FIXME__l_struct_struct_OC_dcomplex_field0 = 513.03101077730003;
  __FIXME__ref_2e_tmp253.__FIXME__l_struct_struct_OC_dcomplex_field1 = 512.33561679759998;
  memcpy(((uint8_t*)&csum_ref[17]), ((uint8_t*)(&__FIXME__ref_2e_tmp253)), 16);
  __FIXME__ref_2e_tmp257.__FIXME__l_struct_struct_OC_dcomplex_field0 = 513.0103090133;
  __FIXME__ref_2e_tmp257.__FIXME__l_struct_struct_OC_dcomplex_field1 = 512.3399592211;
  memcpy(((uint8_t*)&csum_ref[18]), ((uint8_t*)(&__FIXME__ref_2e_tmp257)), 16);
  __FIXME__ref_2e_tmp261.__FIXME__l_struct_struct_OC_dcomplex_field0 = 512.99050293330004;
  __FIXME__ref_2e_tmp261.__FIXME__l_struct_struct_OC_dcomplex_field1 = 512.34355889849996;
  memcpy(((uint8_t*)&csum_ref[19]), ((uint8_t*)(&__FIXME__ref_2e_tmp261)), 16);
  __FIXME__ref_2e_tmp265.__FIXME__l_struct_struct_OC_dcomplex_field0 = 512.97144211090006;
  __FIXME__ref_2e_tmp265.__FIXME__l_struct_struct_OC_dcomplex_field1 = 512.34651640080006;
  memcpy(((uint8_t*)&csum_ref[20]), ((uint8_t*)(&__FIXME__ref_2e_tmp265)), 16);
  }
  }
  }
  } else { // IFELSE MARKER: if.else180 ELSE
  if (d1 == 2048) { // IFELSE MARKER: if.else269 IF
  if (d2 == 1024) { // IFELSE MARKER: land.lhs.true271 IF
  if (d3 == 1024) { // IFELSE MARKER: land.lhs.true273 IF
  if (nt == 25) { // IFELSE MARKER: land.lhs.true275 IF
  *class_npb = 68;
  __FIXME__ref_2e_tmp278.__FIXME__l_struct_struct_OC_dcomplex_field0 = 512.22300652520005;
  __FIXME__ref_2e_tmp278.__FIXME__l_struct_struct_OC_dcomplex_field1 = 511.85340371090001;
  memcpy(((uint8_t*)&csum_ref[1]), ((uint8_t*)(&__FIXME__ref_2e_tmp278)), 16);
  __FIXME__ref_2e_tmp282.__FIXME__l_struct_struct_OC_dcomplex_field0 = 512.0463975765;
  __FIXME__ref_2e_tmp282.__FIXME__l_struct_struct_OC_dcomplex_field1 = 511.70611810819997;
  memcpy(((uint8_t*)&csum_ref[2]), ((uint8_t*)(&__FIXME__ref_2e_tmp282)), 16);
  __FIXME__ref_2e_tmp286.__FIXME__l_struct_struct_OC_dcomplex_field0 = 511.98657667600003;
  __FIXME__ref_2e_tmp286.__FIXME__l_struct_struct_OC_dcomplex_field1 = 511.7096364601;
  memcpy(((uint8_t*)&csum_ref[3]), ((uint8_t*)(&__FIXME__ref_2e_tmp286)), 16);
  __FIXME__ref_2e_tmp290.__FIXME__l_struct_struct_OC_dcomplex_field0 = 511.95187994880001;
  __FIXME__ref_2e_tmp290.__FIXME__l_struct_struct_OC_dcomplex_field1 = 511.73738639499999;
  memcpy(((uint8_t*)&csum_ref[4]), ((uint8_t*)(&__FIXME__ref_2e_tmp290)), 16);
  __FIXME__ref_2e_tmp294.__FIXME__l_struct_struct_OC_dcomplex_field0 = 511.92690882229999;
  __FIXME__ref_2e_tmp294.__FIXME__l_struct_struct_OC_dcomplex_field1 = 511.76803476319998;
  memcpy(((uint8_t*)&csum_ref[5]), ((uint8_t*)(&__FIXME__ref_2e_tmp294)), 16);
  __FIXME__ref_2e_tmp298.__FIXME__l_struct_struct_OC_dcomplex_field0 = 511.90824168580002;
  __FIXME__ref_2e_tmp298.__FIXME__l_struct_struct_OC_dcomplex_field1 = 511.79678755319998;
  memcpy(((uint8_t*)&csum_ref[6]), ((uint8_t*)(&__FIXME__ref_2e_tmp298)), 16);
  __FIXME__ref_2e_tmp302.__FIXME__l_struct_struct_OC_dcomplex_field0 = 511.89438146380002;
  __FIXME__ref_2e_tmp302.__FIXME__l_struct_struct_OC_dcomplex_field1 = 511.82252818410001;
  memcpy(((uint8_t*)&csum_ref[7]), ((uint8_t*)(&__FIXME__ref_2e_tmp302)), 16);
  __FIXME__ref_2e_tmp306.__FIXME__l_struct_struct_OC_dcomplex_field0 = 511.88423850570001;
  __FIXME__ref_2e_tmp306.__FIXME__l_struct_struct_OC_dcomplex_field1 = 511.84516293479999;
  memcpy(((uint8_t*)&csum_ref[8]), ((uint8_t*)(&__FIXME__ref_2e_tmp306)), 16);
  __FIXME__ref_2e_tmp310.__FIXME__l_struct_struct_OC_dcomplex_field0 = 511.87694356319997;
  __FIXME__ref_2e_tmp310.__FIXME__l_struct_struct_OC_dcomplex_field1 = 511.8649119387;
  memcpy(((uint8_t*)&csum_ref[9]), ((uint8_t*)(&__FIXME__ref_2e_tmp310)), 16);
  __FIXME__ref_2e_tmp314.__FIXME__l_struct_struct_OC_dcomplex_field0 = 511.87182034480003;
  __FIXME__ref_2e_tmp314.__FIXME__l_struct_struct_OC_dcomplex_field1 = 511.88208038440001;
  memcpy(((uint8_t*)&csum_ref[10]), ((uint8_t*)(&__FIXME__ref_2e_tmp314)), 16);
  __FIXME__ref_2e_tmp318.__FIXME__l_struct_struct_OC_dcomplex_field0 = 511.86835690610002;
  __FIXME__ref_2e_tmp318.__FIXME__l_struct_struct_OC_dcomplex_field1 = 511.89697810109999;
  memcpy(((uint8_t*)&csum_ref[11]), ((uint8_t*)(&__FIXME__ref_2e_tmp318)), 16);
  __FIXME__ref_2e_tmp322.__FIXME__l_struct_struct_OC_dcomplex_field0 = 511.86617085929998;
  __FIXME__ref_2e_tmp322.__FIXME__l_struct_struct_OC_dcomplex_field1 = 511.90989188349999;
  memcpy(((uint8_t*)&csum_ref[12]), ((uint8_t*)(&__FIXME__ref_2e_tmp322)), 16);
  __FIXME__ref_2e_tmp326.__FIXME__l_struct_struct_OC_dcomplex_field0 = 511.86497689499998;
  __FIXME__ref_2e_tmp326.__FIXME__l_struct_struct_OC_dcomplex_field1 = 511.92107770659999;
  memcpy(((uint8_t*)&csum_ref[13]), ((uint8_t*)(&__FIXME__ref_2e_tmp326)), 16);
  __FIXME__ref_2e_tmp330.__FIXME__l_struct_struct_OC_dcomplex_field0 = 511.86456056259999;
  __FIXME__ref_2e_tmp330.__FIXME__l_struct_struct_OC_dcomplex_field1 = 511.93076044840001;
  memcpy(((uint8_t*)&csum_ref[14]), ((uint8_t*)(&__FIXME__ref_2e_tmp330)), 16);
  __FIXME__ref_2e_tmp334.__FIXME__l_struct_struct_OC_dcomplex_field0 = 511.86475866180001;
  __FIXME__ref_2e_tmp334.__FIXME__l_struct_struct_OC_dcomplex_field1 = 511.93913626710003;
  memcpy(((uint8_t*)&csum_ref[15]), ((uint8_t*)(&__FIXME__ref_2e_tmp334)), 16);
  __FIXME__ref_2e_tmp338.__FIXME__l_struct_struct_OC_dcomplex_field0 = 511.86544515719999;
  __FIXME__ref_2e_tmp338.__FIXME__l_struct_struct_OC_dcomplex_field1 = 511.94637572409999;
  memcpy(((uint8_t*)&csum_ref[16]), ((uint8_t*)(&__FIXME__ref_2e_tmp338)), 16);
  __FIXME__ref_2e_tmp342.__FIXME__l_struct_struct_OC_dcomplex_field0 = 511.86652124509999;
  __FIXME__ref_2e_tmp342.__FIXME__l_struct_struct_OC_dcomplex_field1 = 511.95262692379998;
  memcpy(((uint8_t*)&csum_ref[17]), ((uint8_t*)(&__FIXME__ref_2e_tmp342)), 16);
  __FIXME__ref_2e_tmp346.__FIXME__l_struct_struct_OC_dcomplex_field0 = 511.86790838209998;
  __FIXME__ref_2e_tmp346.__FIXME__l_struct_struct_OC_dcomplex_field1 = 511.95801841079998;
  memcpy(((uint8_t*)&csum_ref[18]), ((uint8_t*)(&__FIXME__ref_2e_tmp346)), 16);
  __FIXME__ref_2e_tmp350.__FIXME__l_struct_struct_OC_dcomplex_field0 = 511.86954336640002;
  __FIXME__ref_2e_tmp350.__FIXME__l_struct_struct_OC_dcomplex_field1 = 511.96266175379998;
  memcpy(((uint8_t*)&csum_ref[19]), ((uint8_t*)(&__FIXME__ref_2e_tmp350)), 16);
  __FIXME__ref_2e_tmp354.__FIXME__l_struct_struct_OC_dcomplex_field0 = 511.87137482639997;
  __FIXME__ref_2e_tmp354.__FIXME__l_struct_struct_OC_dcomplex_field1 = 511.96665381380001;
  memcpy(((uint8_t*)&csum_ref[20]), ((uint8_t*)(&__FIXME__ref_2e_tmp354)), 16);
  __FIXME__ref_2e_tmp358.__FIXME__l_struct_struct_OC_dcomplex_field0 = 511.8733606701;
  __FIXME__ref_2e_tmp358.__FIXME__l_struct_struct_OC_dcomplex_field1 = 511.97007872189999;
  memcpy(((uint8_t*)&csum_ref[21]), ((uint8_t*)(&__FIXME__ref_2e_tmp358)), 16);
  __FIXME__ref_2e_tmp362.__FIXME__l_struct_struct_OC_dcomplex_field0 = 511.87546619739999;
  __FIXME__ref_2e_tmp362.__FIXME__l_struct_struct_OC_dcomplex_field1 = 511.97300959530003;
  memcpy(((uint8_t*)&csum_ref[22]), ((uint8_t*)(&__FIXME__ref_2e_tmp362)), 16);
  __FIXME__ref_2e_tmp366.__FIXME__l_struct_struct_OC_dcomplex_field0 = 511.87766267379999;
  __FIXME__ref_2e_tmp366.__FIXME__l_struct_struct_OC_dcomplex_field1 = 511.9755100241;
  memcpy(((uint8_t*)&csum_ref[23]), ((uint8_t*)(&__FIXME__ref_2e_tmp366)), 16);
  __FIXME__ref_2e_tmp370.__FIXME__l_struct_struct_OC_dcomplex_field0 = 511.87992623140002;
  __FIXME__ref_2e_tmp370.__FIXME__l_struct_struct_OC_dcomplex_field1 = 511.9776353561;
  memcpy(((uint8_t*)&csum_ref[24]), ((uint8_t*)(&__FIXME__ref_2e_tmp370)), 16);
  __FIXME__ref_2e_tmp374.__FIXME__l_struct_struct_OC_dcomplex_field0 = 511.88223700679998;
  __FIXME__ref_2e_tmp374.__FIXME__l_struct_struct_OC_dcomplex_field1 = 511.97943380599997;
  memcpy(((uint8_t*)&csum_ref[25]), ((uint8_t*)(&__FIXME__ref_2e_tmp374)), 16);
  }
  }
  }
  } else { // IFELSE MARKER: if.else269 ELSE
  if (d1 == 4096) { // IFELSE MARKER: if.else378 IF
  if (d2 == 2048) { // IFELSE MARKER: land.lhs.true380 IF
  if (d3 == 2048) { // IFELSE MARKER: land.lhs.true382 IF
  if (nt == 25) { // IFELSE MARKER: land.lhs.true384 IF
  *class_npb = 69;
  __FIXME__ref_2e_tmp387.__FIXME__l_struct_struct_OC_dcomplex_field0 = 512.16010453460001;
  __FIXME__ref_2e_tmp387.__FIXME__l_struct_struct_OC_dcomplex_field1 = 511.7395998266;
  memcpy(((uint8_t*)&csum_ref[1]), ((uint8_t*)(&__FIXME__ref_2e_tmp387)), 16);
  __FIXME__ref_2e_tmp391.__FIXME__l_struct_struct_OC_dcomplex_field0 = 512.09054036780003;
  __FIXME__ref_2e_tmp391.__FIXME__l_struct_struct_OC_dcomplex_field1 = 511.86147161820003;
  memcpy(((uint8_t*)&csum_ref[2]), ((uint8_t*)(&__FIXME__ref_2e_tmp391)), 16);
  __FIXME__ref_2e_tmp395.__FIXME__l_struct_struct_OC_dcomplex_field0 = 512.06232293059998;
  __FIXME__ref_2e_tmp395.__FIXME__l_struct_struct_OC_dcomplex_field1 = 511.9074203747;
  memcpy(((uint8_t*)&csum_ref[3]), ((uint8_t*)(&__FIXME__ref_2e_tmp395)), 16);
  __FIXME__ref_2e_tmp399.__FIXME__l_struct_struct_OC_dcomplex_field0 = 512.04384189970006;
  __FIXME__ref_2e_tmp399.__FIXME__l_struct_struct_OC_dcomplex_field1 = 511.9345900733;
  memcpy(((uint8_t*)&csum_ref[4]), ((uint8_t*)(&__FIXME__ref_2e_tmp399)), 16);
  __FIXME__ref_2e_tmp403.__FIXME__l_struct_struct_OC_dcomplex_field0 = 512.03115218719995;
  __FIXME__ref_2e_tmp403.__FIXME__l_struct_struct_OC_dcomplex_field1 = 511.95513255499998;
  memcpy(((uint8_t*)&csum_ref[5]), ((uint8_t*)(&__FIXME__ref_2e_tmp403)), 16);
  __FIXME__ref_2e_tmp407.__FIXME__l_struct_struct_OC_dcomplex_field0 = 512.02260888089995;
  __FIXME__ref_2e_tmp407.__FIXME__l_struct_struct_OC_dcomplex_field1 = 511.97201799189997;
  memcpy(((uint8_t*)&csum_ref[6]), ((uint8_t*)(&__FIXME__ref_2e_tmp407)), 16);
  __FIXME__ref_2e_tmp411.__FIXME__l_struct_struct_OC_dcomplex_field0 = 512.01692965339998;
  __FIXME__ref_2e_tmp411.__FIXME__l_struct_struct_OC_dcomplex_field1 = 511.98613716649999;
  memcpy(((uint8_t*)&csum_ref[7]), ((uint8_t*)(&__FIXME__ref_2e_tmp411)), 16);
  __FIXME__ref_2e_tmp415.__FIXME__l_struct_struct_OC_dcomplex_field0 = 512.01312251720003;
  __FIXME__ref_2e_tmp415.__FIXME__l_struct_struct_OC_dcomplex_field1 = 511.99793644020002;
  memcpy(((uint8_t*)&csum_ref[8]), ((uint8_t*)(&__FIXME__ref_2e_tmp415)), 16);
  __FIXME__ref_2e_tmp419.__FIXME__l_struct_struct_OC_dcomplex_field0 = 512.01047671080005;
  __FIXME__ref_2e_tmp419.__FIXME__l_struct_struct_OC_dcomplex_field1 = 512.00776740920003;
  memcpy(((uint8_t*)&csum_ref[9]), ((uint8_t*)(&__FIXME__ref_2e_tmp419)), 16);
  __FIXME__ref_2e_tmp423.__FIXME__l_struct_struct_OC_dcomplex_field0 = 512.00851279690005;
  __FIXME__ref_2e_tmp423.__FIXME__l_struct_struct_OC_dcomplex_field1 = 512.01594431210003;
  memcpy(((uint8_t*)&csum_ref[10]), ((uint8_t*)(&__FIXME__ref_2e_tmp423)), 16);
  __FIXME__ref_2e_tmp427.__FIXME__l_struct_struct_OC_dcomplex_field0 = 512.00692241269996;
  __FIXME__ref_2e_tmp427.__FIXME__l_struct_struct_OC_dcomplex_field1 = 512.02274536699997;
  memcpy(((uint8_t*)&csum_ref[11]), ((uint8_t*)(&__FIXME__ref_2e_tmp427)), 16);
  __FIXME__ref_2e_tmp431.__FIXME__l_struct_struct_OC_dcomplex_field0 = 512.00551581640002;
  __FIXME__ref_2e_tmp431.__FIXME__l_struct_struct_OC_dcomplex_field1 = 512.02840960410003;
  memcpy(((uint8_t*)&csum_ref[12]), ((uint8_t*)(&__FIXME__ref_2e_tmp431)), 16);
  __FIXME__ref_2e_tmp435.__FIXME__l_struct_struct_OC_dcomplex_field0 = 512.00418201590003;
  __FIXME__ref_2e_tmp435.__FIXME__l_struct_struct_OC_dcomplex_field1 = 512.0331373793;
  memcpy(((uint8_t*)&csum_ref[13]), ((uint8_t*)(&__FIXME__ref_2e_tmp435)), 16);
  __FIXME__ref_2e_tmp439.__FIXME__l_struct_struct_OC_dcomplex_field0 = 512.0028605402;
  __FIXME__ref_2e_tmp439.__FIXME__l_struct_struct_OC_dcomplex_field1 = 512.03709386790001;
  memcpy(((uint8_t*)&csum_ref[14]), ((uint8_t*)(&__FIXME__ref_2e_tmp439)), 16);
  __FIXME__ref_2e_tmp443.__FIXME__l_struct_struct_OC_dcomplex_field0 = 512.00152230109995;
  __FIXME__ref_2e_tmp443.__FIXME__l_struct_struct_OC_dcomplex_field1 = 512.04041388309997;
  memcpy(((uint8_t*)&csum_ref[15]), ((uint8_t*)(&__FIXME__ref_2e_tmp443)), 16);
  __FIXME__ref_2e_tmp447.__FIXME__l_struct_struct_OC_dcomplex_field0 = 512.00015700220001;
  __FIXME__ref_2e_tmp447.__FIXME__l_struct_struct_OC_dcomplex_field1 = 512.04320688370001;
  memcpy(((uint8_t*)&csum_ref[16]), ((uint8_t*)(&__FIXME__ref_2e_tmp447)), 16);
  __FIXME__ref_2e_tmp451.__FIXME__l_struct_struct_OC_dcomplex_field0 = 511.99876505549997;
  __FIXME__ref_2e_tmp451.__FIXME__l_struct_struct_OC_dcomplex_field1 = 512.04556158599996;
  memcpy(((uint8_t*)&csum_ref[17]), ((uint8_t*)(&__FIXME__ref_2e_tmp451)), 16);
  __FIXME__ref_2e_tmp455.__FIXME__l_struct_struct_OC_dcomplex_field0 = 511.99735250909998;
  __FIXME__ref_2e_tmp455.__FIXME__l_struct_struct_OC_dcomplex_field1 = 512.0475499442;
  memcpy(((uint8_t*)&csum_ref[18]), ((uint8_t*)(&__FIXME__ref_2e_tmp455)), 16);
  __FIXME__ref_2e_tmp459.__FIXME__l_struct_struct_OC_dcomplex_field0 = 511.99592794720002;
  __FIXME__ref_2e_tmp459.__FIXME__l_struct_struct_OC_dcomplex_field1 = 512.04923046290003;
  memcpy(((uint8_t*)&csum_ref[19]), ((uint8_t*)(&__FIXME__ref_2e_tmp459)), 16);
  __FIXME__ref_2e_tmp463.__FIXME__l_struct_struct_OC_dcomplex_field0 = 511.99450065579998;
  __FIXME__ref_2e_tmp463.__FIXME__l_struct_struct_OC_dcomplex_field1 = 512.05065089020002;
  memcpy(((uint8_t*)&csum_ref[20]), ((uint8_t*)(&__FIXME__ref_2e_tmp463)), 16);
  __FIXME__ref_2e_tmp467.__FIXME__l_struct_struct_OC_dcomplex_field0 = 511.99307959110001;
  __FIXME__ref_2e_tmp467.__FIXME__l_struct_struct_OC_dcomplex_field1 = 512.05185037820002;
  memcpy(((uint8_t*)&csum_ref[21]), ((uint8_t*)(&__FIXME__ref_2e_tmp467)), 16);
  __FIXME__ref_2e_tmp471.__FIXME__l_struct_struct_OC_dcomplex_field0 = 511.9916728462;
  __FIXME__ref_2e_tmp471.__FIXME__l_struct_struct_OC_dcomplex_field1 = 512.05286120159997;
  memcpy(((uint8_t*)&csum_ref[22]), ((uint8_t*)(&__FIXME__ref_2e_tmp471)), 16);
  __FIXME__ref_2e_tmp475.__FIXME__l_struct_struct_OC_dcomplex_field0 = 511.99028741849997;
  __FIXME__ref_2e_tmp475.__FIXME__l_struct_struct_OC_dcomplex_field1 = 512.05371011950001;
  memcpy(((uint8_t*)&csum_ref[23]), ((uint8_t*)(&__FIXME__ref_2e_tmp475)), 16);
  __FIXME__ref_2e_tmp479.__FIXME__l_struct_struct_OC_dcomplex_field0 = 511.98892915649998;
  __FIXME__ref_2e_tmp479.__FIXME__l_struct_struct_OC_dcomplex_field1 = 512.05441945140001;
  memcpy(((uint8_t*)&csum_ref[24]), ((uint8_t*)(&__FIXME__ref_2e_tmp479)), 16);
  __FIXME__ref_2e_tmp483.__FIXME__l_struct_struct_OC_dcomplex_field0 = 511.98760280490001;
  __FIXME__ref_2e_tmp483.__FIXME__l_struct_struct_OC_dcomplex_field1 = 512.05500792839996;
  memcpy(((uint8_t*)&csum_ref[25]), ((uint8_t*)(&__FIXME__ref_2e_tmp483)), 16);
  }
  }
  }
  }
  }
  }
  }
  }
  }
  }
// INSERT COMMENT IFELSE: verify::if.end492
  if (*class_npb != 85) { // IFELSE MARKER: if.end492 IF
  *verified = 1;
for(int64_t i = 1; i <= nt;   i = i + 1) {
  __FIXME__agg_2e_tmp.__FIXME__l_struct_struct_OC_dcomplex_field0 = ((_ZL4sums+i)->__FIXME__l_struct_struct_OC_dcomplex_field0 - (csum_ref[i].__FIXME__l_struct_struct_OC_dcomplex_field0));
  __FIXME__agg_2e_tmp.__FIXME__l_struct_struct_OC_dcomplex_field1 = ((_ZL4sums+i)->__FIXME__l_struct_struct_OC_dcomplex_field1 - (csum_ref[i].__FIXME__l_struct_struct_OC_dcomplex_field1));
  memcpy(((uint8_t*)(&__FIXME__agg_2e_tmp510)), ((uint8_t*)&csum_ref[i]), 16);
  struct __FIXME__l_unnamed_2 __FIXME__call = dcomplex_div(((struct __FIXME__l_unnamed_2*)(&__FIXME__agg_2e_tmp))->__FIXME__l_unnamed_2_field0, ((struct __FIXME__l_unnamed_2*)(&__FIXME__agg_2e_tmp))->__FIXME__l_unnamed_2_field1, ((struct __FIXME__l_unnamed_2*)(&__FIXME__agg_2e_tmp510))->__FIXME__l_unnamed_2_field0, ((struct __FIXME__l_unnamed_2*)(&__FIXME__agg_2e_tmp510))->__FIXME__l_unnamed_2_field1);
  ((struct __FIXME__l_unnamed_2*)(&__FIXME__coerce))->__FIXME__l_unnamed_2_field0 = (__FIXME__call.__FIXME__l_unnamed_2_field0);
  ((struct __FIXME__l_unnamed_2*)(&__FIXME__coerce))->__FIXME__l_unnamed_2_field1 = (__FIXME__call.__FIXME__l_unnamed_2_field1);
  __FIXME__agg_2e_tmp514.__FIXME__l_struct_struct_OC_dcomplex_field0 = ((_ZL4sums+i)->__FIXME__l_struct_struct_OC_dcomplex_field0 - (csum_ref[i].__FIXME__l_struct_struct_OC_dcomplex_field0));
  __FIXME__agg_2e_tmp514.__FIXME__l_struct_struct_OC_dcomplex_field1 = ((_ZL4sums+i)->__FIXME__l_struct_struct_OC_dcomplex_field1 - (csum_ref[i].__FIXME__l_struct_struct_OC_dcomplex_field1));
  memcpy(((uint8_t*)(&__FIXME__agg_2e_tmp531)), ((uint8_t*)&csum_ref[i]), 16);
  struct __FIXME__l_unnamed_2 __FIXME__call534 = dcomplex_div(((struct __FIXME__l_unnamed_2*)(&__FIXME__agg_2e_tmp514))->__FIXME__l_unnamed_2_field0, ((struct __FIXME__l_unnamed_2*)(&__FIXME__agg_2e_tmp514))->__FIXME__l_unnamed_2_field1, ((struct __FIXME__l_unnamed_2*)(&__FIXME__agg_2e_tmp531))->__FIXME__l_unnamed_2_field0, ((struct __FIXME__l_unnamed_2*)(&__FIXME__agg_2e_tmp531))->__FIXME__l_unnamed_2_field1);
  ((struct __FIXME__l_unnamed_2*)(&__FIXME__coerce535))->__FIXME__l_unnamed_2_field0 = (__FIXME__call534.__FIXME__l_unnamed_2_field0);
  ((struct __FIXME__l_unnamed_2*)(&__FIXME__coerce535))->__FIXME__l_unnamed_2_field1 = (__FIXME__call534.__FIXME__l_unnamed_2_field1);
  __FIXME__agg_2e_tmp537.__FIXME__l_struct_struct_OC_dcomplex_field0 = ((_ZL4sums+i)->__FIXME__l_struct_struct_OC_dcomplex_field0 - (csum_ref[i].__FIXME__l_struct_struct_OC_dcomplex_field0));
  __FIXME__agg_2e_tmp537.__FIXME__l_struct_struct_OC_dcomplex_field1 = ((_ZL4sums+i)->__FIXME__l_struct_struct_OC_dcomplex_field1 - (csum_ref[i].__FIXME__l_struct_struct_OC_dcomplex_field1));
  memcpy(((uint8_t*)(&__FIXME__agg_2e_tmp554)), ((uint8_t*)&csum_ref[i]), 16);
  struct __FIXME__l_unnamed_2 __FIXME__call557 = dcomplex_div(((struct __FIXME__l_unnamed_2*)(&__FIXME__agg_2e_tmp537))->__FIXME__l_unnamed_2_field0, ((struct __FIXME__l_unnamed_2*)(&__FIXME__agg_2e_tmp537))->__FIXME__l_unnamed_2_field1, ((struct __FIXME__l_unnamed_2*)(&__FIXME__agg_2e_tmp554))->__FIXME__l_unnamed_2_field0, ((struct __FIXME__l_unnamed_2*)(&__FIXME__agg_2e_tmp554))->__FIXME__l_unnamed_2_field1);
  ((struct __FIXME__l_unnamed_2*)(&__FIXME__coerce558))->__FIXME__l_unnamed_2_field0 = (__FIXME__call557.__FIXME__l_unnamed_2_field0);
  ((struct __FIXME__l_unnamed_2*)(&__FIXME__coerce558))->__FIXME__l_unnamed_2_field1 = (__FIXME__call557.__FIXME__l_unnamed_2_field1);
  __FIXME__agg_2e_tmp560.__FIXME__l_struct_struct_OC_dcomplex_field0 = ((_ZL4sums+i)->__FIXME__l_struct_struct_OC_dcomplex_field0 - (csum_ref[i].__FIXME__l_struct_struct_OC_dcomplex_field0));
  __FIXME__agg_2e_tmp560.__FIXME__l_struct_struct_OC_dcomplex_field1 = ((_ZL4sums+i)->__FIXME__l_struct_struct_OC_dcomplex_field1 - (csum_ref[i].__FIXME__l_struct_struct_OC_dcomplex_field1));
  memcpy(((uint8_t*)(&__FIXME__agg_2e_tmp577)), ((uint8_t*)&csum_ref[i]), 16);
  struct __FIXME__l_unnamed_2 __FIXME__call580 = dcomplex_div(((struct __FIXME__l_unnamed_2*)(&__FIXME__agg_2e_tmp560))->__FIXME__l_unnamed_2_field0, ((struct __FIXME__l_unnamed_2*)(&__FIXME__agg_2e_tmp560))->__FIXME__l_unnamed_2_field1, ((struct __FIXME__l_unnamed_2*)(&__FIXME__agg_2e_tmp577))->__FIXME__l_unnamed_2_field0, ((struct __FIXME__l_unnamed_2*)(&__FIXME__agg_2e_tmp577))->__FIXME__l_unnamed_2_field1);
  ((struct __FIXME__l_unnamed_2*)(&__FIXME__coerce581))->__FIXME__l_unnamed_2_field0 = (__FIXME__call580.__FIXME__l_unnamed_2_field0);
  ((struct __FIXME__l_unnamed_2*)(&__FIXME__coerce581))->__FIXME__l_unnamed_2_field1 = (__FIXME__call580.__FIXME__l_unnamed_2_field1);
  double err = sqrt(((__FIXME__coerce.__FIXME__l_struct_struct_OC_dcomplex_field0 * __FIXME__coerce535.__FIXME__l_struct_struct_OC_dcomplex_field0) + (__FIXME__coerce558.__FIXME__l_struct_struct_OC_dcomplex_field1 * __FIXME__coerce581.__FIXME__l_struct_struct_OC_dcomplex_field1)));
  if (!(llvm_fcmp_ole(err, 9.9999999999999998E-13))) {
  *verified = 0;
  break;
  }
}
  }
// INSERT COMMENT IFELSE: verify::if.end588
  if (*class_npb != 85) { // IFELSE MARKER: if.end588 IF
  if (*verified != 0) { // IFELSE MARKER: if.then591 IF
  printf((_OC_str_OC_73));
  } else { // IFELSE MARKER: if.then591 ELSE
  printf((_OC_str_OC_74));
  }
  }
  printf((_OC_str_OC_75), *class_npb);
}
// FUNCTION ORDER ID 16 END


// FUNCTION ORDER ID 17 START
// INSERT COMMENT FUNCTION: release_gpu
void release_gpu(void) {
  return;
}
// FUNCTION ORDER ID 17 END


// FUNCTION ORDER ID 18 START
// INSERT COMMENT FUNCTION: dcomplex_div
struct __FIXME__l_unnamed_2 dcomplex_div(double __FIXME__z1_2e_coerce0, double __FIXME__z1_2e_coerce1, double __FIXME__z2_2e_coerce0, double __FIXME__z2_2e_coerce1) {
  struct __FIXME__l_struct_struct_OC_dcomplex result;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex z1;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex z2;    /* Address-exposed local */
  double a;
  double b;
  double c;
  double d;
  double divisor;

  ((struct __FIXME__l_unnamed_2*)(&z1))->__FIXME__l_unnamed_2_field0 = __FIXME__z1_2e_coerce0;
  ((struct __FIXME__l_unnamed_2*)(&z1))->__FIXME__l_unnamed_2_field1 = __FIXME__z1_2e_coerce1;
  ((struct __FIXME__l_unnamed_2*)(&z2))->__FIXME__l_unnamed_2_field0 = __FIXME__z2_2e_coerce0;
  ((struct __FIXME__l_unnamed_2*)(&z2))->__FIXME__l_unnamed_2_field1 = __FIXME__z2_2e_coerce1;
  a = z1.__FIXME__l_struct_struct_OC_dcomplex_field0;
  b = z1.__FIXME__l_struct_struct_OC_dcomplex_field1;
  c = z2.__FIXME__l_struct_struct_OC_dcomplex_field0;
  d = z2.__FIXME__l_struct_struct_OC_dcomplex_field1;
  divisor = ((c * c) + (d * d));
  result.__FIXME__l_struct_struct_OC_dcomplex_field0 = (((a * c) + (b * d)) / divisor);
  result.__FIXME__l_struct_struct_OC_dcomplex_field1 = (((b * c) - ((a * d))) / divisor);
  return *((struct __FIXME__l_unnamed_2*)(&result));
}
// FUNCTION ORDER ID 18 END


// FUNCTION ORDER ID 19 START
// INSERT COMMENT FUNCTION: cffts1_gpu
void cffts1_gpu(uint32_t is, struct __FIXME__l_struct_struct_OC_dcomplex* u, struct __FIXME__l_struct_struct_OC_dcomplex* x_in, struct __FIXME__l_struct_struct_OC_dcomplex* x_out, struct __FIXME__l_struct_struct_OC_dcomplex* y0, struct __FIXME__l_struct_struct_OC_dcomplex* y1) {
  struct __FIXME__l_struct_struct_OC_dim3 __FIXME__agg_2e_tmp;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dim3 __FIXME__agg_2e_tmp1;    /* Address-exposed local */
  struct __FIXME__l_unnamed_1 __FIXME__agg_2e_tmp_2e_coerce;    /* Address-exposed local */
  struct __FIXME__l_unnamed_1 __FIXME__agg_2e_tmp1_2e_coerce;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dim3 __FIXME__agg_2e_tmp3;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dim3 __FIXME__agg_2e_tmp4;    /* Address-exposed local */
  struct __FIXME__l_unnamed_1 __FIXME__agg_2e_tmp3_2e_coerce;    /* Address-exposed local */
  struct __FIXME__l_unnamed_1 __FIXME__agg_2e_tmp4_2e_coerce;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dim3 __FIXME__agg_2e_tmp10;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dim3 __FIXME__agg_2e_tmp11;    /* Address-exposed local */
  struct __FIXME__l_unnamed_1 __FIXME__agg_2e_tmp10_2e_coerce;    /* Address-exposed local */
  struct __FIXME__l_unnamed_1 __FIXME__agg_2e_tmp11_2e_coerce;    /* Address-exposed local */
  uint32_t i;
  uint32_t j;

  __FIXME__agg_2e_tmp.__FIXME__l_struct_struct_OC_dim3_field0 = blocks_per_grid_on_fftx_1;
  __FIXME__agg_2e_tmp.__FIXME__l_struct_struct_OC_dim3_field1 = 1;
  __FIXME__agg_2e_tmp.__FIXME__l_struct_struct_OC_dim3_field2 = 1;
  __FIXME__agg_2e_tmp1.__FIXME__l_struct_struct_OC_dim3_field0 = threads_per_block_on_fftx_1;
  __FIXME__agg_2e_tmp1.__FIXME__l_struct_struct_OC_dim3_field1 = 1;
  __FIXME__agg_2e_tmp1.__FIXME__l_struct_struct_OC_dim3_field2 = 1;
  memcpy(((uint8_t*)(&__FIXME__agg_2e_tmp_2e_coerce)), ((uint8_t*)(&__FIXME__agg_2e_tmp)), 12);
  memcpy(((uint8_t*)(&__FIXME__agg_2e_tmp1_2e_coerce)), ((uint8_t*)(&__FIXME__agg_2e_tmp1)), 12);
// INSERT COMMENT LOOP: cffts1_gpu::header.0
#pragma omp parallel for collapse(2)
for(int32_t i = 0; i < blocks_per_grid_on_fftx_1;   i = i + 1) {
for(int32_t j = 0; j < threads_per_block_on_fftx_1;   j = j + 1) {
cffts1_gpu_kernel_1(x_in, y0, blocks_per_grid_on_fftx_1, 1, 1, threads_per_block_on_fftx_1, 1, 1, i, 0, 0, j, 0, 0);
}
}
  __FIXME__agg_2e_tmp3.__FIXME__l_struct_struct_OC_dim3_field0 = blocks_per_grid_on_fftx_2;
  __FIXME__agg_2e_tmp3.__FIXME__l_struct_struct_OC_dim3_field1 = 1;
  __FIXME__agg_2e_tmp3.__FIXME__l_struct_struct_OC_dim3_field2 = 1;
  __FIXME__agg_2e_tmp4.__FIXME__l_struct_struct_OC_dim3_field0 = threads_per_block_on_fftx_2;
  __FIXME__agg_2e_tmp4.__FIXME__l_struct_struct_OC_dim3_field1 = 1;
  __FIXME__agg_2e_tmp4.__FIXME__l_struct_struct_OC_dim3_field2 = 1;
  memcpy(((uint8_t*)(&__FIXME__agg_2e_tmp3_2e_coerce)), ((uint8_t*)(&__FIXME__agg_2e_tmp3)), 12);
  memcpy(((uint8_t*)(&__FIXME__agg_2e_tmp4_2e_coerce)), ((uint8_t*)(&__FIXME__agg_2e_tmp4)), 12);
// INSERT COMMENT LOOP: cffts1_gpu::header.016
#pragma omp parallel for collapse(2)
for(int32_t i = 0; i < blocks_per_grid_on_fftx_2;   i = i + 1) {
for(int32_t j = 0; j < threads_per_block_on_fftx_2;   j = j + 1) {
cffts1_gpu_kernel_2(is, y0, y1, u, blocks_per_grid_on_fftx_2, 1, 1, threads_per_block_on_fftx_2, 1, 1, i, 0, 0, j, 0, 0);
}
}
  __FIXME__agg_2e_tmp10.__FIXME__l_struct_struct_OC_dim3_field0 = blocks_per_grid_on_fftx_3;
  __FIXME__agg_2e_tmp10.__FIXME__l_struct_struct_OC_dim3_field1 = 1;
  __FIXME__agg_2e_tmp10.__FIXME__l_struct_struct_OC_dim3_field2 = 1;
  __FIXME__agg_2e_tmp11.__FIXME__l_struct_struct_OC_dim3_field0 = threads_per_block_on_fftx_3;
  __FIXME__agg_2e_tmp11.__FIXME__l_struct_struct_OC_dim3_field1 = 1;
  __FIXME__agg_2e_tmp11.__FIXME__l_struct_struct_OC_dim3_field2 = 1;
  memcpy(((uint8_t*)(&__FIXME__agg_2e_tmp10_2e_coerce)), ((uint8_t*)(&__FIXME__agg_2e_tmp10)), 12);
  memcpy(((uint8_t*)(&__FIXME__agg_2e_tmp11_2e_coerce)), ((uint8_t*)(&__FIXME__agg_2e_tmp11)), 12);
// INSERT COMMENT LOOP: cffts1_gpu::header.026
#pragma omp parallel for collapse(2)
for(int32_t i = 0; i < blocks_per_grid_on_fftx_3;   i = i + 1) {
for(int32_t j = 0; j < threads_per_block_on_fftx_3;   j = j + 1) {
cffts1_gpu_kernel_3(x_out, y0, blocks_per_grid_on_fftx_3, 1, 1, threads_per_block_on_fftx_3, 1, 1, i, 0, 0, j, 0, 0);
}
}
  return;
}
// FUNCTION ORDER ID 19 END


// FUNCTION ORDER ID 20 START
// INSERT COMMENT FUNCTION: cffts2_gpu
void cffts2_gpu(uint32_t is, struct __FIXME__l_struct_struct_OC_dcomplex* u, struct __FIXME__l_struct_struct_OC_dcomplex* x_in, struct __FIXME__l_struct_struct_OC_dcomplex* x_out, struct __FIXME__l_struct_struct_OC_dcomplex* y0, struct __FIXME__l_struct_struct_OC_dcomplex* y1) {
  struct __FIXME__l_struct_struct_OC_dim3 __FIXME__agg_2e_tmp;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dim3 __FIXME__agg_2e_tmp1;    /* Address-exposed local */
  struct __FIXME__l_unnamed_1 __FIXME__agg_2e_tmp_2e_coerce;    /* Address-exposed local */
  struct __FIXME__l_unnamed_1 __FIXME__agg_2e_tmp1_2e_coerce;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dim3 __FIXME__agg_2e_tmp3;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dim3 __FIXME__agg_2e_tmp4;    /* Address-exposed local */
  struct __FIXME__l_unnamed_1 __FIXME__agg_2e_tmp3_2e_coerce;    /* Address-exposed local */
  struct __FIXME__l_unnamed_1 __FIXME__agg_2e_tmp4_2e_coerce;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dim3 __FIXME__agg_2e_tmp10;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dim3 __FIXME__agg_2e_tmp11;    /* Address-exposed local */
  struct __FIXME__l_unnamed_1 __FIXME__agg_2e_tmp10_2e_coerce;    /* Address-exposed local */
  struct __FIXME__l_unnamed_1 __FIXME__agg_2e_tmp11_2e_coerce;    /* Address-exposed local */
  uint32_t i;
  uint32_t j;

  __FIXME__agg_2e_tmp.__FIXME__l_struct_struct_OC_dim3_field0 = blocks_per_grid_on_ffty_1;
  __FIXME__agg_2e_tmp.__FIXME__l_struct_struct_OC_dim3_field1 = 1;
  __FIXME__agg_2e_tmp.__FIXME__l_struct_struct_OC_dim3_field2 = 1;
  __FIXME__agg_2e_tmp1.__FIXME__l_struct_struct_OC_dim3_field0 = threads_per_block_on_ffty_1;
  __FIXME__agg_2e_tmp1.__FIXME__l_struct_struct_OC_dim3_field1 = 1;
  __FIXME__agg_2e_tmp1.__FIXME__l_struct_struct_OC_dim3_field2 = 1;
  memcpy(((uint8_t*)(&__FIXME__agg_2e_tmp_2e_coerce)), ((uint8_t*)(&__FIXME__agg_2e_tmp)), 12);
  memcpy(((uint8_t*)(&__FIXME__agg_2e_tmp1_2e_coerce)), ((uint8_t*)(&__FIXME__agg_2e_tmp1)), 12);
// INSERT COMMENT LOOP: cffts2_gpu::header.0
#pragma omp parallel for collapse(2)
for(int32_t i = 0; i < blocks_per_grid_on_ffty_1;   i = i + 1) {
for(int32_t j = 0; j < threads_per_block_on_ffty_1;   j = j + 1) {
cffts2_gpu_kernel_1(x_in, y0, blocks_per_grid_on_ffty_1, 1, 1, threads_per_block_on_ffty_1, 1, 1, i, 0, 0, j, 0, 0);
}
}
  __FIXME__agg_2e_tmp3.__FIXME__l_struct_struct_OC_dim3_field0 = blocks_per_grid_on_ffty_2;
  __FIXME__agg_2e_tmp3.__FIXME__l_struct_struct_OC_dim3_field1 = 1;
  __FIXME__agg_2e_tmp3.__FIXME__l_struct_struct_OC_dim3_field2 = 1;
  __FIXME__agg_2e_tmp4.__FIXME__l_struct_struct_OC_dim3_field0 = threads_per_block_on_ffty_2;
  __FIXME__agg_2e_tmp4.__FIXME__l_struct_struct_OC_dim3_field1 = 1;
  __FIXME__agg_2e_tmp4.__FIXME__l_struct_struct_OC_dim3_field2 = 1;
  memcpy(((uint8_t*)(&__FIXME__agg_2e_tmp3_2e_coerce)), ((uint8_t*)(&__FIXME__agg_2e_tmp3)), 12);
  memcpy(((uint8_t*)(&__FIXME__agg_2e_tmp4_2e_coerce)), ((uint8_t*)(&__FIXME__agg_2e_tmp4)), 12);
// INSERT COMMENT LOOP: cffts2_gpu::header.016
#pragma omp parallel for collapse(2)
for(int32_t i = 0; i < blocks_per_grid_on_ffty_2;   i = i + 1) {
for(int32_t j = 0; j < threads_per_block_on_ffty_2;   j = j + 1) {
cffts2_gpu_kernel_2(is, y0, y1, u, blocks_per_grid_on_ffty_2, 1, 1, threads_per_block_on_ffty_2, 1, 1, i, 0, 0, j, 0, 0);
}
}
  __FIXME__agg_2e_tmp10.__FIXME__l_struct_struct_OC_dim3_field0 = blocks_per_grid_on_ffty_3;
  __FIXME__agg_2e_tmp10.__FIXME__l_struct_struct_OC_dim3_field1 = 1;
  __FIXME__agg_2e_tmp10.__FIXME__l_struct_struct_OC_dim3_field2 = 1;
  __FIXME__agg_2e_tmp11.__FIXME__l_struct_struct_OC_dim3_field0 = threads_per_block_on_ffty_3;
  __FIXME__agg_2e_tmp11.__FIXME__l_struct_struct_OC_dim3_field1 = 1;
  __FIXME__agg_2e_tmp11.__FIXME__l_struct_struct_OC_dim3_field2 = 1;
  memcpy(((uint8_t*)(&__FIXME__agg_2e_tmp10_2e_coerce)), ((uint8_t*)(&__FIXME__agg_2e_tmp10)), 12);
  memcpy(((uint8_t*)(&__FIXME__agg_2e_tmp11_2e_coerce)), ((uint8_t*)(&__FIXME__agg_2e_tmp11)), 12);
// INSERT COMMENT LOOP: cffts2_gpu::header.026
#pragma omp parallel for collapse(2)
for(int32_t i = 0; i < blocks_per_grid_on_ffty_3;   i = i + 1) {
for(int32_t j = 0; j < threads_per_block_on_ffty_3;   j = j + 1) {
cffts2_gpu_kernel_3(x_out, y0, blocks_per_grid_on_ffty_3, 1, 1, threads_per_block_on_ffty_3, 1, 1, i, 0, 0, j, 0, 0);
}
}
  return;
}
// FUNCTION ORDER ID 20 END


// FUNCTION ORDER ID 21 START
// INSERT COMMENT FUNCTION: cffts3_gpu
void cffts3_gpu(uint32_t is, struct __FIXME__l_struct_struct_OC_dcomplex* u, struct __FIXME__l_struct_struct_OC_dcomplex* x_in, struct __FIXME__l_struct_struct_OC_dcomplex* x_out, struct __FIXME__l_struct_struct_OC_dcomplex* y0, struct __FIXME__l_struct_struct_OC_dcomplex* y1) {
  struct __FIXME__l_struct_struct_OC_dim3 __FIXME__agg_2e_tmp;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dim3 __FIXME__agg_2e_tmp1;    /* Address-exposed local */
  struct __FIXME__l_unnamed_1 __FIXME__agg_2e_tmp_2e_coerce;    /* Address-exposed local */
  struct __FIXME__l_unnamed_1 __FIXME__agg_2e_tmp1_2e_coerce;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dim3 __FIXME__agg_2e_tmp3;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dim3 __FIXME__agg_2e_tmp4;    /* Address-exposed local */
  struct __FIXME__l_unnamed_1 __FIXME__agg_2e_tmp3_2e_coerce;    /* Address-exposed local */
  struct __FIXME__l_unnamed_1 __FIXME__agg_2e_tmp4_2e_coerce;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dim3 __FIXME__agg_2e_tmp10;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dim3 __FIXME__agg_2e_tmp11;    /* Address-exposed local */
  struct __FIXME__l_unnamed_1 __FIXME__agg_2e_tmp10_2e_coerce;    /* Address-exposed local */
  struct __FIXME__l_unnamed_1 __FIXME__agg_2e_tmp11_2e_coerce;    /* Address-exposed local */
  uint32_t i;
  uint32_t j;

  __FIXME__agg_2e_tmp.__FIXME__l_struct_struct_OC_dim3_field0 = blocks_per_grid_on_fftz_1;
  __FIXME__agg_2e_tmp.__FIXME__l_struct_struct_OC_dim3_field1 = 1;
  __FIXME__agg_2e_tmp.__FIXME__l_struct_struct_OC_dim3_field2 = 1;
  __FIXME__agg_2e_tmp1.__FIXME__l_struct_struct_OC_dim3_field0 = threads_per_block_on_fftz_1;
  __FIXME__agg_2e_tmp1.__FIXME__l_struct_struct_OC_dim3_field1 = 1;
  __FIXME__agg_2e_tmp1.__FIXME__l_struct_struct_OC_dim3_field2 = 1;
  memcpy(((uint8_t*)(&__FIXME__agg_2e_tmp_2e_coerce)), ((uint8_t*)(&__FIXME__agg_2e_tmp)), 12);
  memcpy(((uint8_t*)(&__FIXME__agg_2e_tmp1_2e_coerce)), ((uint8_t*)(&__FIXME__agg_2e_tmp1)), 12);
// INSERT COMMENT LOOP: cffts3_gpu::header.0
#pragma omp parallel for collapse(2)
for(int32_t i = 0; i < blocks_per_grid_on_fftz_1;   i = i + 1) {
for(int32_t j = 0; j < threads_per_block_on_fftz_1;   j = j + 1) {
cffts3_gpu_kernel_1(x_in, y0, blocks_per_grid_on_fftz_1, 1, 1, threads_per_block_on_fftz_1, 1, 1, i, 0, 0, j, 0, 0);
}
}
  __FIXME__agg_2e_tmp3.__FIXME__l_struct_struct_OC_dim3_field0 = blocks_per_grid_on_fftz_2;
  __FIXME__agg_2e_tmp3.__FIXME__l_struct_struct_OC_dim3_field1 = 1;
  __FIXME__agg_2e_tmp3.__FIXME__l_struct_struct_OC_dim3_field2 = 1;
  __FIXME__agg_2e_tmp4.__FIXME__l_struct_struct_OC_dim3_field0 = threads_per_block_on_fftz_2;
  __FIXME__agg_2e_tmp4.__FIXME__l_struct_struct_OC_dim3_field1 = 1;
  __FIXME__agg_2e_tmp4.__FIXME__l_struct_struct_OC_dim3_field2 = 1;
  memcpy(((uint8_t*)(&__FIXME__agg_2e_tmp3_2e_coerce)), ((uint8_t*)(&__FIXME__agg_2e_tmp3)), 12);
  memcpy(((uint8_t*)(&__FIXME__agg_2e_tmp4_2e_coerce)), ((uint8_t*)(&__FIXME__agg_2e_tmp4)), 12);
// INSERT COMMENT LOOP: cffts3_gpu::header.016
#pragma omp parallel for collapse(2)
for(int32_t i = 0; i < blocks_per_grid_on_fftz_2;   i = i + 1) {
for(int32_t j = 0; j < threads_per_block_on_fftz_2;   j = j + 1) {
cffts3_gpu_kernel_2(is, y0, y1, u, blocks_per_grid_on_fftz_2, 1, 1, threads_per_block_on_fftz_2, 1, 1, i, 0, 0, j, 0, 0);
}
}
  __FIXME__agg_2e_tmp10.__FIXME__l_struct_struct_OC_dim3_field0 = blocks_per_grid_on_fftz_3;
  __FIXME__agg_2e_tmp10.__FIXME__l_struct_struct_OC_dim3_field1 = 1;
  __FIXME__agg_2e_tmp10.__FIXME__l_struct_struct_OC_dim3_field2 = 1;
  __FIXME__agg_2e_tmp11.__FIXME__l_struct_struct_OC_dim3_field0 = threads_per_block_on_fftz_3;
  __FIXME__agg_2e_tmp11.__FIXME__l_struct_struct_OC_dim3_field1 = 1;
  __FIXME__agg_2e_tmp11.__FIXME__l_struct_struct_OC_dim3_field2 = 1;
  memcpy(((uint8_t*)(&__FIXME__agg_2e_tmp10_2e_coerce)), ((uint8_t*)(&__FIXME__agg_2e_tmp10)), 12);
  memcpy(((uint8_t*)(&__FIXME__agg_2e_tmp11_2e_coerce)), ((uint8_t*)(&__FIXME__agg_2e_tmp11)), 12);
// INSERT COMMENT LOOP: cffts3_gpu::header.026
#pragma omp parallel for collapse(2)
for(int32_t i = 0; i < blocks_per_grid_on_fftz_3;   i = i + 1) {
for(int32_t j = 0; j < threads_per_block_on_fftz_3;   j = j + 1) {
cffts3_gpu_kernel_3(x_out, y0, blocks_per_grid_on_fftz_3, 1, 1, threads_per_block_on_fftz_3, 1, 1, i, 0, 0, j, 0, 0);
}
}
  return;
}
// FUNCTION ORDER ID 21 END


// FUNCTION ORDER ID 22 START
// INSERT COMMENT FUNCTION: ilog2
uint32_t ilog2(uint32_t n) {
  int32_t nn;
  uint32_t lg;

  if (n == 1) {
  return 0;
  }
  nn = 2;
  lg = 1;
// INSERT COMMENT LOOP: ilog2::while.cond
while (nn < ((int32_t)n)) {
  nn = (nn << 1);
  lg = lg + 1;
}
  return lg;
}
// FUNCTION ORDER ID 22 END


// FUNCTION ORDER ID 23 START
// INSERT COMMENT FUNCTION: ipow46
void ipow46(double a, uint32_t exponent, double* result) {
  double q;    /* Address-exposed local */
  double r;    /* Address-exposed local */
  int32_t n;

  *result = 1;
  if (exponent == 0) {
  return;
  }
  q = a;
  r = 1;
  n = exponent;
// INSERT COMMENT LOOP: ipow46::while.cond
while (n > 1) {
  uint32_t __FIXME__div = n / 2;
  if (__FIXME__div * 2 == n) { // IFELSE MARKER: while.body IF
  randlc((&q), q);
  n = __FIXME__div;
  } else { // IFELSE MARKER: while.body ELSE
  randlc((&r), q);
  n = (n - 1);
  }
}
  randlc((&r), q);
  *result = r;
  return;
}
// FUNCTION ORDER ID 23 END


// FUNCTION ORDER ID 24 START
// INSERT COMMENT FUNCTION: init_ui_gpu_kernel
void init_ui_gpu_kernel(struct __FIXME__l_struct_struct_OC_dcomplex* u0, struct __FIXME__l_struct_struct_OC_dcomplex* u1, double* twiddle, uint32_t __FIXME__gridDim_2e_x, uint32_t __FIXME__gridDim_2e_y, uint32_t __FIXME__gridDim_2e_z, uint32_t __FIXME__blockDim_2e_x, uint32_t __FIXME__blockDim_2e_y, uint32_t __FIXME__blockDim_2e_z, uint32_t __FIXME__blockIdx_2e_x, uint32_t __FIXME__blockIdx_2e_y, uint32_t __FIXME__blockIdx_2e_z, uint32_t __FIXME__threadIdx_2e_x, uint32_t __FIXME__threadIdx_2e_y, uint32_t __FIXME__threadIdx_2e_z) {
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp3;    /* Address-exposed local */
  int64_t thread_id;

// INSERT COMMENT IFELSE: init_ui_gpu_kernel::entry
  thread_id = __FIXME__blockIdx_2e_x * __FIXME__blockDim_2e_x + __FIXME__threadIdx_2e_x;
  if (thread_id >= 8388608) { // IFELSE MARKER: entry IF
  } else { // IFELSE MARKER: entry ELSE
  __FIXME__ref_2e_tmp.__FIXME__l_struct_struct_OC_dcomplex_field0 = 0;
  __FIXME__ref_2e_tmp.__FIXME__l_struct_struct_OC_dcomplex_field1 = 0;
  memcpy(((uint8_t*)(u0+thread_id)), ((uint8_t*)(&__FIXME__ref_2e_tmp)), 16);
  __FIXME__ref_2e_tmp3.__FIXME__l_struct_struct_OC_dcomplex_field0 = 0;
  __FIXME__ref_2e_tmp3.__FIXME__l_struct_struct_OC_dcomplex_field1 = 0;
  memcpy(((uint8_t*)(u1+thread_id)), ((uint8_t*)(&__FIXME__ref_2e_tmp3)), 16);
  twiddle[thread_id] = 0;
  }
  return;
}
// FUNCTION ORDER ID 24 END


// FUNCTION ORDER ID 25 START
// INSERT COMMENT FUNCTION: compute_indexmap_gpu_kernel
void compute_indexmap_gpu_kernel(double* twiddle, uint32_t __FIXME__gridDim_2e_x, uint32_t __FIXME__gridDim_2e_y, uint32_t __FIXME__gridDim_2e_z, uint32_t __FIXME__blockDim_2e_x, uint32_t __FIXME__blockDim_2e_y, uint32_t __FIXME__blockDim_2e_z, uint32_t __FIXME__blockIdx_2e_x, uint32_t __FIXME__blockIdx_2e_y, uint32_t __FIXME__blockIdx_2e_z, uint32_t __FIXME__threadIdx_2e_x, uint32_t __FIXME__threadIdx_2e_y, uint32_t __FIXME__threadIdx_2e_z) {
  int64_t thread_id;
  uint32_t kk;
  uint32_t jj;
  uint32_t ii;
  double __FIXME__exp_result;

  thread_id = __FIXME__blockIdx_2e_x * __FIXME__blockDim_2e_x + __FIXME__threadIdx_2e_x;
  if (thread_id >= 8388608) {
  return;
  }
  kk = ((int)(thread_id / 65536 + 64) % (int)128 - 64);
  jj = ((int)((int)thread_id / 256 % (int)256 + 128) % (int)256 - 128);
  ii = ((int)((int)thread_id % (int)256 + 128) % (int)256 - 128);
  __FIXME__exp_result = exp((-3.947841760435743E-5 * ((double)((int32_t)(ii * ii + (jj * jj + kk * kk))))));
  twiddle[thread_id] = __FIXME__exp_result;
  return;
}
// FUNCTION ORDER ID 25 END


// FUNCTION ORDER ID 26 START
// INSERT COMMENT FUNCTION: compute_initial_conditions_gpu_kernel
void compute_initial_conditions_gpu_kernel(struct __FIXME__l_struct_struct_OC_dcomplex* u0, double* starts, uint32_t __FIXME__gridDim_2e_x, uint32_t __FIXME__gridDim_2e_y, uint32_t __FIXME__gridDim_2e_z, uint32_t __FIXME__blockDim_2e_x, uint32_t __FIXME__blockDim_2e_y, uint32_t __FIXME__blockDim_2e_z, uint32_t __FIXME__blockIdx_2e_x, uint32_t __FIXME__blockIdx_2e_y, uint32_t __FIXME__blockIdx_2e_z, uint32_t __FIXME__threadIdx_2e_x, uint32_t __FIXME__threadIdx_2e_y, uint32_t __FIXME__threadIdx_2e_z) {
  double x0;    /* Address-exposed local */
  int64_t z;
  uint64_t y;

  z = __FIXME__blockIdx_2e_x * __FIXME__blockDim_2e_x + __FIXME__threadIdx_2e_x;
  if (z >= 128) {
  return;
  }
  x0 = starts[z];
// INSERT COMMENT LOOP: compute_initial_conditions_gpu_kernel::for.cond
for(int64_t y = 0; y < 256;   y = y + 1) {
vranlc_device(512, (&x0), 1220703125, ((double*)(u0+(y * 256 + z * 256 * 256))));
}
  return;
}
// FUNCTION ORDER ID 26 END


// FUNCTION ORDER ID 27 START
// INSERT COMMENT FUNCTION: evolve_gpu_kernel
void evolve_gpu_kernel(struct __FIXME__l_struct_struct_OC_dcomplex* u0, struct __FIXME__l_struct_struct_OC_dcomplex* u1, double* twiddle, uint32_t __FIXME__gridDim_2e_x, uint32_t __FIXME__gridDim_2e_y, uint32_t __FIXME__gridDim_2e_z, uint32_t __FIXME__blockDim_2e_x, uint32_t __FIXME__blockDim_2e_y, uint32_t __FIXME__blockDim_2e_z, uint32_t __FIXME__blockIdx_2e_x, uint32_t __FIXME__blockIdx_2e_y, uint32_t __FIXME__blockIdx_2e_z, uint32_t __FIXME__threadIdx_2e_x, uint32_t __FIXME__threadIdx_2e_y, uint32_t __FIXME__threadIdx_2e_z) {
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp;    /* Address-exposed local */
  int64_t thread_id;

// INSERT COMMENT IFELSE: evolve_gpu_kernel::entry
  thread_id = __FIXME__blockIdx_2e_x * __FIXME__blockDim_2e_x + __FIXME__threadIdx_2e_x;
  if (thread_id >= 8388608) { // IFELSE MARKER: entry IF
  } else { // IFELSE MARKER: entry ELSE
  __FIXME__ref_2e_tmp.__FIXME__l_struct_struct_OC_dcomplex_field0 = ((u0+thread_id)->__FIXME__l_struct_struct_OC_dcomplex_field0 * twiddle[thread_id]);
  __FIXME__ref_2e_tmp.__FIXME__l_struct_struct_OC_dcomplex_field1 = ((u0+thread_id)->__FIXME__l_struct_struct_OC_dcomplex_field1 * twiddle[thread_id]);
  memcpy(((uint8_t*)(u0+thread_id)), ((uint8_t*)(&__FIXME__ref_2e_tmp)), 16);
  memcpy(((uint8_t*)(u1+thread_id)), ((uint8_t*)(u0+thread_id)), 16);
  }
  return;
}
// FUNCTION ORDER ID 27 END


// FUNCTION ORDER ID 28 START
// INSERT COMMENT FUNCTION: checksum_gpu_kernel0
void checksum_gpu_kernel0(uint32_t iteration, struct __FIXME__l_struct_struct_OC_dcomplex* u1, struct __FIXME__l_struct_struct_OC_dcomplex* sums, uint32_t __FIXME__gridDim_2e_x, uint32_t __FIXME__gridDim_2e_y, uint32_t __FIXME__gridDim_2e_z, uint32_t __FIXME__blockDim_2e_x, uint32_t __FIXME__blockDim_2e_y, uint32_t __FIXME__blockDim_2e_z, uint32_t __FIXME__blockIdx_2e_x, uint32_t __FIXME__blockIdx_2e_y, uint32_t __FIXME__blockIdx_2e_z, uint32_t __FIXME__threadIdx_2e_x, uint32_t __FIXME__threadIdx_2e_y, uint32_t __FIXME__threadIdx_2e_z) {
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp;    /* Address-exposed local */
  int32_t j;

// INSERT COMMENT IFELSE: checksum_gpu_kernel0::entry
  j = __FIXME__blockIdx_2e_x * __FIXME__blockDim_2e_x + __FIXME__threadIdx_2e_x + 1;
  if (j <= 1024) { // IFELSE MARKER: entry IF
  memcpy(((uint8_t*)(((struct __FIXME__l_struct_struct_OC_dcomplex*)(&extern_share_data_shared))+__FIXME__threadIdx_2e_x)), ((uint8_t*)(u1+(((int)j % (int)256 + (int)3 * j % (int)256 * 256) + (int)5 * j % (int)128 * 256 * 256))), 16);
  } else { // IFELSE MARKER: entry ELSE
  __FIXME__ref_2e_tmp.__FIXME__l_struct_struct_OC_dcomplex_field0 = 0;
  __FIXME__ref_2e_tmp.__FIXME__l_struct_struct_OC_dcomplex_field1 = 0;
  memcpy(((uint8_t*)(((struct __FIXME__l_struct_struct_OC_dcomplex*)(&extern_share_data_shared))+__FIXME__threadIdx_2e_x)), ((uint8_t*)(&__FIXME__ref_2e_tmp)), 16);
  }
  return;
}
// FUNCTION ORDER ID 28 END


// FUNCTION ORDER ID 29 START
// INSERT COMMENT FUNCTION: cffts1_gpu_kernel_1
void cffts1_gpu_kernel_1(struct __FIXME__l_struct_struct_OC_dcomplex* x_in, struct __FIXME__l_struct_struct_OC_dcomplex* y0, uint32_t __FIXME__gridDim_2e_x, uint32_t __FIXME__gridDim_2e_y, uint32_t __FIXME__gridDim_2e_z, uint32_t __FIXME__blockDim_2e_x, uint32_t __FIXME__blockDim_2e_y, uint32_t __FIXME__blockDim_2e_z, uint32_t __FIXME__blockIdx_2e_x, uint32_t __FIXME__blockIdx_2e_y, uint32_t __FIXME__blockIdx_2e_z, uint32_t __FIXME__threadIdx_2e_x, uint32_t __FIXME__threadIdx_2e_y, uint32_t __FIXME__threadIdx_2e_z) {
  int64_t x_y_z;
  uint32_t x;
  uint32_t y;
  uint32_t z;

// INSERT COMMENT IFELSE: cffts1_gpu_kernel_1::entry
  x_y_z = __FIXME__blockIdx_2e_x * __FIXME__blockDim_2e_x + __FIXME__threadIdx_2e_x;
  if (x_y_z >= 8388608) { // IFELSE MARKER: entry IF
  } else { // IFELSE MARKER: entry ELSE
  x = (int)x_y_z % (int)256;
  y = (int)x_y_z / 256 % (int)256;
  z = x_y_z / 65536;
  (y0+((y + x * 256) + z * 256 * 256))->__FIXME__l_struct_struct_OC_dcomplex_field0 = (x_in+x_y_z)->__FIXME__l_struct_struct_OC_dcomplex_field0;
  (y0+((y + x * 256) + z * 256 * 256))->__FIXME__l_struct_struct_OC_dcomplex_field1 = (x_in+x_y_z)->__FIXME__l_struct_struct_OC_dcomplex_field1;
  }
  return;
}
// FUNCTION ORDER ID 29 END


// FUNCTION ORDER ID 30 START
// INSERT COMMENT FUNCTION: cffts1_gpu_kernel_2
void cffts1_gpu_kernel_2(uint32_t is, struct __FIXME__l_struct_struct_OC_dcomplex* gty1, struct __FIXME__l_struct_struct_OC_dcomplex* gty2, struct __FIXME__l_struct_struct_OC_dcomplex* u_device, uint32_t __FIXME__gridDim_2e_x, uint32_t __FIXME__gridDim_2e_y, uint32_t __FIXME__gridDim_2e_z, uint32_t __FIXME__blockDim_2e_x, uint32_t __FIXME__blockDim_2e_y, uint32_t __FIXME__blockDim_2e_z, uint32_t __FIXME__blockIdx_2e_x, uint32_t __FIXME__blockIdx_2e_y, uint32_t __FIXME__blockIdx_2e_z, uint32_t __FIXME__threadIdx_2e_x, uint32_t __FIXME__threadIdx_2e_y, uint32_t __FIXME__threadIdx_2e_z) {
  int32_t y_z;
  uint32_t j;
  uint32_t k;
  int32_t logd1;
  int32_t l;
  int64_t i1;
  int64_t k1;
  uint64_t j1;

  y_z = __FIXME__blockIdx_2e_x * __FIXME__blockDim_2e_x + __FIXME__threadIdx_2e_x;
  if (y_z >= 32768) {
  return;
  }
  j = (int)y_z % (int)256;
  k = (int)y_z / 256 % (int)128;
  logd1 = ilog2_device(256);
// INSERT COMMENT LOOP: cffts1_gpu_kernel_2::for.cond
for(int32_t l = 1; l <= logd1;   l = l + 2) {
  uint32_t lk = (1 << (l - 1));
  uint32_t __FIXME__shl7 = (1 << (logd1 - l));
for(int64_t i1 = 0; i1 <= (__FIXME__shl7 - 1);   i1 = i1 + 1) {
for(int64_t k1 = 0; k1 <= (lk - 1);   k1 = k1 + 1) {
  uint64_t __FIXME__166 = i1 * lk;
  uint64_t __FIXME__167 = __FIXME__166 + 128;
  uint64_t __FIXME__168 = i1 * 2 * lk;
  uint64_t __FIXME__169 = __FIXME__168 + lk;
  double uu1_real = (u_device+(__FIXME__shl7 + i1))->__FIXME__l_struct_struct_OC_dcomplex_field0;
  double uu1_imag = (((double)((int32_t)is)) * (u_device+(__FIXME__shl7 + i1))->__FIXME__l_struct_struct_OC_dcomplex_field1);
  double x11_real = (gty1+((j + (__FIXME__166 + k1) * 256) + k * 256 * 256))->__FIXME__l_struct_struct_OC_dcomplex_field0;
  double x11_imag = (gty1+((j + (__FIXME__166 + k1) * 256) + k * 256 * 256))->__FIXME__l_struct_struct_OC_dcomplex_field1;
  double x21_real = (gty1+((j + (__FIXME__167 + k1) * 256) + k * 256 * 256))->__FIXME__l_struct_struct_OC_dcomplex_field0;
  double x21_imag = (gty1+((j + (__FIXME__167 + k1) * 256) + k * 256 * 256))->__FIXME__l_struct_struct_OC_dcomplex_field1;
  (gty2+((j + (__FIXME__168 + k1) * 256) + k * 256 * 256))->__FIXME__l_struct_struct_OC_dcomplex_field0 = (x11_real + x21_real);
  (gty2+((j + (__FIXME__168 + k1) * 256) + k * 256 * 256))->__FIXME__l_struct_struct_OC_dcomplex_field1 = (x11_imag + x21_imag);
  double temp_real = (x11_real - x21_real);
  double temp_imag = (x11_imag - x21_imag);
  (gty2+((j + (__FIXME__169 + k1) * 256) + k * 256 * 256))->__FIXME__l_struct_struct_OC_dcomplex_field0 = ((uu1_real * temp_real) - ((uu1_imag * temp_imag)));
  (gty2+((j + (__FIXME__169 + k1) * 256) + k * 256 * 256))->__FIXME__l_struct_struct_OC_dcomplex_field1 = ((uu1_real * temp_imag) + (uu1_imag * temp_real));
}
}
  if (l == logd1) { // IFELSE MARKER: for.end110 IF
for(int64_t j1 = 0; j1 < 256;   j1 = j1 + 1) {
  (gty1+((j + j1 * 256) + k * 256 * 256))->__FIXME__l_struct_struct_OC_dcomplex_field0 = (gty2+((j + j1 * 256) + k * 256 * 256))->__FIXME__l_struct_struct_OC_dcomplex_field0;
  (gty1+((j + j1 * 256) + k * 256 * 256))->__FIXME__l_struct_struct_OC_dcomplex_field1 = (gty2+((j + j1 * 256) + k * 256 * 256))->__FIXME__l_struct_struct_OC_dcomplex_field1;
}
  } else { // IFELSE MARKER: for.end110 ELSE
  lk = (1 << (l + 1 - 1));
  uint32_t __FIXME__shl156 = (1 << (logd1 - (l + 1)));
for(int64_t i1 = 0; i1 <= (__FIXME__shl156 - 1);   i1 = i1 + 1) {
for(int64_t k1 = 0; k1 <= (lk - 1);   k1 = k1 + 1) {
  uint64_t __FIXME__170 = i1 * lk;
  uint64_t __FIXME__171 = __FIXME__170 + 128;
  uint64_t __FIXME__172 = i1 * 2 * lk;
  uint64_t __FIXME__173 = __FIXME__172 + lk;
  double uu2_real = (u_device+(__FIXME__shl156 + i1))->__FIXME__l_struct_struct_OC_dcomplex_field0;
  double uu2_imag = (((double)((int32_t)is)) * (u_device+(__FIXME__shl156 + i1))->__FIXME__l_struct_struct_OC_dcomplex_field1);
  double x12_real = (gty2+((j + (__FIXME__170 + k1) * 256) + k * 256 * 256))->__FIXME__l_struct_struct_OC_dcomplex_field0;
  double x12_imag = (gty2+((j + (__FIXME__170 + k1) * 256) + k * 256 * 256))->__FIXME__l_struct_struct_OC_dcomplex_field1;
  double x22_real = (gty2+((j + (__FIXME__171 + k1) * 256) + k * 256 * 256))->__FIXME__l_struct_struct_OC_dcomplex_field0;
  double x22_imag = (gty2+((j + (__FIXME__171 + k1) * 256) + k * 256 * 256))->__FIXME__l_struct_struct_OC_dcomplex_field1;
  (gty1+((j + (__FIXME__172 + k1) * 256) + k * 256 * 256))->__FIXME__l_struct_struct_OC_dcomplex_field0 = (x12_real + x22_real);
  (gty1+((j + (__FIXME__172 + k1) * 256) + k * 256 * 256))->__FIXME__l_struct_struct_OC_dcomplex_field1 = (x12_imag + x22_imag);
  double temp2_real = (x12_real - x22_real);
  double temp2_imag = (x12_imag - x22_imag);
  (gty1+((j + (__FIXME__173 + k1) * 256) + k * 256 * 256))->__FIXME__l_struct_struct_OC_dcomplex_field0 = ((uu2_real * temp2_real) - ((uu2_imag * temp2_imag)));
  (gty1+((j + (__FIXME__173 + k1) * 256) + k * 256 * 256))->__FIXME__l_struct_struct_OC_dcomplex_field1 = ((uu2_real * temp2_imag) + (uu2_imag * temp2_real));
}
}
  }
}
  return;
}
// FUNCTION ORDER ID 30 END


// FUNCTION ORDER ID 31 START
// INSERT COMMENT FUNCTION: cffts1_gpu_kernel_3
void cffts1_gpu_kernel_3(struct __FIXME__l_struct_struct_OC_dcomplex* x_out, struct __FIXME__l_struct_struct_OC_dcomplex* y0, uint32_t __FIXME__gridDim_2e_x, uint32_t __FIXME__gridDim_2e_y, uint32_t __FIXME__gridDim_2e_z, uint32_t __FIXME__blockDim_2e_x, uint32_t __FIXME__blockDim_2e_y, uint32_t __FIXME__blockDim_2e_z, uint32_t __FIXME__blockIdx_2e_x, uint32_t __FIXME__blockIdx_2e_y, uint32_t __FIXME__blockIdx_2e_z, uint32_t __FIXME__threadIdx_2e_x, uint32_t __FIXME__threadIdx_2e_y, uint32_t __FIXME__threadIdx_2e_z) {
  int64_t x_y_z;
  uint32_t x;
  uint32_t y;
  uint32_t z;

// INSERT COMMENT IFELSE: cffts1_gpu_kernel_3::entry
  x_y_z = __FIXME__blockIdx_2e_x * __FIXME__blockDim_2e_x + __FIXME__threadIdx_2e_x;
  if (x_y_z >= 8388608) { // IFELSE MARKER: entry IF
  } else { // IFELSE MARKER: entry ELSE
  x = (int)x_y_z % (int)256;
  y = (int)x_y_z / 256 % (int)256;
  z = x_y_z / 65536;
  (x_out+x_y_z)->__FIXME__l_struct_struct_OC_dcomplex_field0 = (y0+((y + x * 256) + z * 256 * 256))->__FIXME__l_struct_struct_OC_dcomplex_field0;
  (x_out+x_y_z)->__FIXME__l_struct_struct_OC_dcomplex_field1 = (y0+((y + x * 256) + z * 256 * 256))->__FIXME__l_struct_struct_OC_dcomplex_field1;
  }
  return;
}
// FUNCTION ORDER ID 31 END


// FUNCTION ORDER ID 32 START
// INSERT COMMENT FUNCTION: cffts2_gpu_kernel_1
void cffts2_gpu_kernel_1(struct __FIXME__l_struct_struct_OC_dcomplex* x_in, struct __FIXME__l_struct_struct_OC_dcomplex* y0, uint32_t __FIXME__gridDim_2e_x, uint32_t __FIXME__gridDim_2e_y, uint32_t __FIXME__gridDim_2e_z, uint32_t __FIXME__blockDim_2e_x, uint32_t __FIXME__blockDim_2e_y, uint32_t __FIXME__blockDim_2e_z, uint32_t __FIXME__blockIdx_2e_x, uint32_t __FIXME__blockIdx_2e_y, uint32_t __FIXME__blockIdx_2e_z, uint32_t __FIXME__threadIdx_2e_x, uint32_t __FIXME__threadIdx_2e_y, uint32_t __FIXME__threadIdx_2e_z) {
  int64_t x_y_z;

// INSERT COMMENT IFELSE: cffts2_gpu_kernel_1::entry
  x_y_z = __FIXME__blockIdx_2e_x * __FIXME__blockDim_2e_x + __FIXME__threadIdx_2e_x;
  if (x_y_z >= 8388608) { // IFELSE MARKER: entry IF
  } else { // IFELSE MARKER: entry ELSE
  (y0+x_y_z)->__FIXME__l_struct_struct_OC_dcomplex_field0 = (x_in+x_y_z)->__FIXME__l_struct_struct_OC_dcomplex_field0;
  (y0+x_y_z)->__FIXME__l_struct_struct_OC_dcomplex_field1 = (x_in+x_y_z)->__FIXME__l_struct_struct_OC_dcomplex_field1;
  }
  return;
}
// FUNCTION ORDER ID 32 END


// FUNCTION ORDER ID 33 START
// INSERT COMMENT FUNCTION: cffts2_gpu_kernel_2
void cffts2_gpu_kernel_2(uint32_t is, struct __FIXME__l_struct_struct_OC_dcomplex* gty1, struct __FIXME__l_struct_struct_OC_dcomplex* gty2, struct __FIXME__l_struct_struct_OC_dcomplex* u_device, uint32_t __FIXME__gridDim_2e_x, uint32_t __FIXME__gridDim_2e_y, uint32_t __FIXME__gridDim_2e_z, uint32_t __FIXME__blockDim_2e_x, uint32_t __FIXME__blockDim_2e_y, uint32_t __FIXME__blockDim_2e_z, uint32_t __FIXME__blockIdx_2e_x, uint32_t __FIXME__blockIdx_2e_y, uint32_t __FIXME__blockIdx_2e_z, uint32_t __FIXME__threadIdx_2e_x, uint32_t __FIXME__threadIdx_2e_y, uint32_t __FIXME__threadIdx_2e_z) {
  int32_t x_z;
  uint32_t i;
  uint32_t k;
  int32_t logd2;
  int32_t l;
  int64_t i1;
  int64_t k1;
  uint64_t j1;

  x_z = __FIXME__blockIdx_2e_x * __FIXME__blockDim_2e_x + __FIXME__threadIdx_2e_x;
  if (x_z >= 32768) {
  return;
  }
  i = (int)x_z % (int)256;
  k = (int)x_z / 256 % (int)128;
  logd2 = ilog2_device(256);
// INSERT COMMENT LOOP: cffts2_gpu_kernel_2::for.cond
for(int32_t l = 1; l <= logd2;   l = l + 2) {
  uint32_t lk = (1 << (l - 1));
  uint32_t __FIXME__shl7 = (1 << (logd2 - l));
for(int64_t i1 = 0; i1 <= (__FIXME__shl7 - 1);   i1 = i1 + 1) {
for(int64_t k1 = 0; k1 <= (lk - 1);   k1 = k1 + 1) {
  uint64_t __FIXME__174 = i1 * lk;
  uint64_t __FIXME__175 = __FIXME__174 + 128;
  uint64_t __FIXME__176 = i1 * 2 * lk;
  uint64_t __FIXME__177 = __FIXME__176 + lk;
  double uu1_real = (u_device+(__FIXME__shl7 + i1))->__FIXME__l_struct_struct_OC_dcomplex_field0;
  double uu1_imag = (((double)((int32_t)is)) * (u_device+(__FIXME__shl7 + i1))->__FIXME__l_struct_struct_OC_dcomplex_field1);
  double x11_real = (gty1+((i + (__FIXME__174 + k1) * 256) + k * 256 * 256))->__FIXME__l_struct_struct_OC_dcomplex_field0;
  double x11_imag = (gty1+((i + (__FIXME__174 + k1) * 256) + k * 256 * 256))->__FIXME__l_struct_struct_OC_dcomplex_field1;
  double x21_real = (gty1+((i + (__FIXME__175 + k1) * 256) + k * 256 * 256))->__FIXME__l_struct_struct_OC_dcomplex_field0;
  double x21_imag = (gty1+((i + (__FIXME__175 + k1) * 256) + k * 256 * 256))->__FIXME__l_struct_struct_OC_dcomplex_field1;
  (gty2+((i + (__FIXME__176 + k1) * 256) + k * 256 * 256))->__FIXME__l_struct_struct_OC_dcomplex_field0 = (x11_real + x21_real);
  (gty2+((i + (__FIXME__176 + k1) * 256) + k * 256 * 256))->__FIXME__l_struct_struct_OC_dcomplex_field1 = (x11_imag + x21_imag);
  double temp_real = (x11_real - x21_real);
  double temp_imag = (x11_imag - x21_imag);
  (gty2+((i + (__FIXME__177 + k1) * 256) + k * 256 * 256))->__FIXME__l_struct_struct_OC_dcomplex_field0 = ((uu1_real * temp_real) - ((uu1_imag * temp_imag)));
  (gty2+((i + (__FIXME__177 + k1) * 256) + k * 256 * 256))->__FIXME__l_struct_struct_OC_dcomplex_field1 = ((uu1_real * temp_imag) + (uu1_imag * temp_real));
}
}
  if (l == logd2) { // IFELSE MARKER: for.end110 IF
for(int64_t j1 = 0; j1 < 256;   j1 = j1 + 1) {
  (gty1+((i + j1 * 256) + k * 256 * 256))->__FIXME__l_struct_struct_OC_dcomplex_field0 = (gty2+((i + j1 * 256) + k * 256 * 256))->__FIXME__l_struct_struct_OC_dcomplex_field0;
  (gty1+((i + j1 * 256) + k * 256 * 256))->__FIXME__l_struct_struct_OC_dcomplex_field1 = (gty2+((i + j1 * 256) + k * 256 * 256))->__FIXME__l_struct_struct_OC_dcomplex_field1;
}
  } else { // IFELSE MARKER: for.end110 ELSE
  lk = (1 << (l + 1 - 1));
  uint32_t __FIXME__shl156 = (1 << (logd2 - (l + 1)));
for(int64_t i1 = 0; i1 <= (__FIXME__shl156 - 1);   i1 = i1 + 1) {
for(int64_t k1 = 0; k1 <= (lk - 1);   k1 = k1 + 1) {
  uint64_t __FIXME__178 = i1 * lk;
  uint64_t __FIXME__179 = __FIXME__178 + 128;
  uint64_t __FIXME__180 = i1 * 2 * lk;
  uint64_t __FIXME__181 = __FIXME__180 + lk;
  double uu2_real = (u_device+(__FIXME__shl156 + i1))->__FIXME__l_struct_struct_OC_dcomplex_field0;
  double uu2_imag = (((double)((int32_t)is)) * (u_device+(__FIXME__shl156 + i1))->__FIXME__l_struct_struct_OC_dcomplex_field1);
  double x12_real = (gty2+((i + (__FIXME__178 + k1) * 256) + k * 256 * 256))->__FIXME__l_struct_struct_OC_dcomplex_field0;
  double x12_imag = (gty2+((i + (__FIXME__178 + k1) * 256) + k * 256 * 256))->__FIXME__l_struct_struct_OC_dcomplex_field1;
  double x22_real = (gty2+((i + (__FIXME__179 + k1) * 256) + k * 256 * 256))->__FIXME__l_struct_struct_OC_dcomplex_field0;
  double x22_imag = (gty2+((i + (__FIXME__179 + k1) * 256) + k * 256 * 256))->__FIXME__l_struct_struct_OC_dcomplex_field1;
  (gty1+((i + (__FIXME__180 + k1) * 256) + k * 256 * 256))->__FIXME__l_struct_struct_OC_dcomplex_field0 = (x12_real + x22_real);
  (gty1+((i + (__FIXME__180 + k1) * 256) + k * 256 * 256))->__FIXME__l_struct_struct_OC_dcomplex_field1 = (x12_imag + x22_imag);
  double temp2_real = (x12_real - x22_real);
  double temp2_imag = (x12_imag - x22_imag);
  (gty1+((i + (__FIXME__181 + k1) * 256) + k * 256 * 256))->__FIXME__l_struct_struct_OC_dcomplex_field0 = ((uu2_real * temp2_real) - ((uu2_imag * temp2_imag)));
  (gty1+((i + (__FIXME__181 + k1) * 256) + k * 256 * 256))->__FIXME__l_struct_struct_OC_dcomplex_field1 = ((uu2_real * temp2_imag) + (uu2_imag * temp2_real));
}
}
  }
}
  return;
}
// FUNCTION ORDER ID 33 END


// FUNCTION ORDER ID 34 START
// INSERT COMMENT FUNCTION: cffts2_gpu_kernel_3
void cffts2_gpu_kernel_3(struct __FIXME__l_struct_struct_OC_dcomplex* x_out, struct __FIXME__l_struct_struct_OC_dcomplex* y0, uint32_t __FIXME__gridDim_2e_x, uint32_t __FIXME__gridDim_2e_y, uint32_t __FIXME__gridDim_2e_z, uint32_t __FIXME__blockDim_2e_x, uint32_t __FIXME__blockDim_2e_y, uint32_t __FIXME__blockDim_2e_z, uint32_t __FIXME__blockIdx_2e_x, uint32_t __FIXME__blockIdx_2e_y, uint32_t __FIXME__blockIdx_2e_z, uint32_t __FIXME__threadIdx_2e_x, uint32_t __FIXME__threadIdx_2e_y, uint32_t __FIXME__threadIdx_2e_z) {
  int64_t x_y_z;

// INSERT COMMENT IFELSE: cffts2_gpu_kernel_3::entry
  x_y_z = __FIXME__blockIdx_2e_x * __FIXME__blockDim_2e_x + __FIXME__threadIdx_2e_x;
  if (x_y_z >= 8388608) { // IFELSE MARKER: entry IF
  } else { // IFELSE MARKER: entry ELSE
  (x_out+x_y_z)->__FIXME__l_struct_struct_OC_dcomplex_field0 = (y0+x_y_z)->__FIXME__l_struct_struct_OC_dcomplex_field0;
  (x_out+x_y_z)->__FIXME__l_struct_struct_OC_dcomplex_field1 = (y0+x_y_z)->__FIXME__l_struct_struct_OC_dcomplex_field1;
  }
  return;
}
// FUNCTION ORDER ID 34 END


// FUNCTION ORDER ID 35 START
// INSERT COMMENT FUNCTION: cffts3_gpu_kernel_1
void cffts3_gpu_kernel_1(struct __FIXME__l_struct_struct_OC_dcomplex* x_in, struct __FIXME__l_struct_struct_OC_dcomplex* y0, uint32_t __FIXME__gridDim_2e_x, uint32_t __FIXME__gridDim_2e_y, uint32_t __FIXME__gridDim_2e_z, uint32_t __FIXME__blockDim_2e_x, uint32_t __FIXME__blockDim_2e_y, uint32_t __FIXME__blockDim_2e_z, uint32_t __FIXME__blockIdx_2e_x, uint32_t __FIXME__blockIdx_2e_y, uint32_t __FIXME__blockIdx_2e_z, uint32_t __FIXME__threadIdx_2e_x, uint32_t __FIXME__threadIdx_2e_y, uint32_t __FIXME__threadIdx_2e_z) {
  int64_t x_y_z;

// INSERT COMMENT IFELSE: cffts3_gpu_kernel_1::entry
  x_y_z = __FIXME__blockIdx_2e_x * __FIXME__blockDim_2e_x + __FIXME__threadIdx_2e_x;
  if (x_y_z >= 8388608) { // IFELSE MARKER: entry IF
  } else { // IFELSE MARKER: entry ELSE
  (y0+x_y_z)->__FIXME__l_struct_struct_OC_dcomplex_field0 = (x_in+x_y_z)->__FIXME__l_struct_struct_OC_dcomplex_field0;
  (y0+x_y_z)->__FIXME__l_struct_struct_OC_dcomplex_field1 = (x_in+x_y_z)->__FIXME__l_struct_struct_OC_dcomplex_field1;
  }
  return;
}
// FUNCTION ORDER ID 35 END


// FUNCTION ORDER ID 36 START
// INSERT COMMENT FUNCTION: cffts3_gpu_kernel_2
void cffts3_gpu_kernel_2(uint32_t is, struct __FIXME__l_struct_struct_OC_dcomplex* gty1, struct __FIXME__l_struct_struct_OC_dcomplex* gty2, struct __FIXME__l_struct_struct_OC_dcomplex* u_device, uint32_t __FIXME__gridDim_2e_x, uint32_t __FIXME__gridDim_2e_y, uint32_t __FIXME__gridDim_2e_z, uint32_t __FIXME__blockDim_2e_x, uint32_t __FIXME__blockDim_2e_y, uint32_t __FIXME__blockDim_2e_z, uint32_t __FIXME__blockIdx_2e_x, uint32_t __FIXME__blockIdx_2e_y, uint32_t __FIXME__blockIdx_2e_z, uint32_t __FIXME__threadIdx_2e_x, uint32_t __FIXME__threadIdx_2e_y, uint32_t __FIXME__threadIdx_2e_z) {
  int32_t x_y;
  int32_t __FIXME__call3;

// INSERT COMMENT IFELSE: cffts3_gpu_kernel_2::entry
  x_y = __FIXME__blockIdx_2e_x * __FIXME__blockDim_2e_x + __FIXME__threadIdx_2e_x;
  if (x_y >= 65536) { // IFELSE MARKER: entry IF
  } else { // IFELSE MARKER: entry ELSE
  __FIXME__call3 = ilog2_device(128);
cffts3_gpu_cfftz_device(is, __FIXME__call3, 128, gty1, gty2, u_device, x_y, 65536);
  }
  return;
}
// FUNCTION ORDER ID 36 END


// FUNCTION ORDER ID 37 START
// INSERT COMMENT FUNCTION: cffts3_gpu_kernel_3
void cffts3_gpu_kernel_3(struct __FIXME__l_struct_struct_OC_dcomplex* x_out, struct __FIXME__l_struct_struct_OC_dcomplex* y0, uint32_t __FIXME__gridDim_2e_x, uint32_t __FIXME__gridDim_2e_y, uint32_t __FIXME__gridDim_2e_z, uint32_t __FIXME__blockDim_2e_x, uint32_t __FIXME__blockDim_2e_y, uint32_t __FIXME__blockDim_2e_z, uint32_t __FIXME__blockIdx_2e_x, uint32_t __FIXME__blockIdx_2e_y, uint32_t __FIXME__blockIdx_2e_z, uint32_t __FIXME__threadIdx_2e_x, uint32_t __FIXME__threadIdx_2e_y, uint32_t __FIXME__threadIdx_2e_z) {
  int64_t x_y_z;

// INSERT COMMENT IFELSE: cffts3_gpu_kernel_3::entry
  x_y_z = __FIXME__blockIdx_2e_x * __FIXME__blockDim_2e_x + __FIXME__threadIdx_2e_x;
  if (x_y_z >= 8388608) { // IFELSE MARKER: entry IF
  } else { // IFELSE MARKER: entry ELSE
  (x_out+x_y_z)->__FIXME__l_struct_struct_OC_dcomplex_field0 = (y0+x_y_z)->__FIXME__l_struct_struct_OC_dcomplex_field0;
  (x_out+x_y_z)->__FIXME__l_struct_struct_OC_dcomplex_field1 = (y0+x_y_z)->__FIXME__l_struct_struct_OC_dcomplex_field1;
  }
  return;
}
// FUNCTION ORDER ID 37 END


// FUNCTION ORDER ID 38 START
// INSERT COMMENT FUNCTION: checksum_gpu_kernel1
void checksum_gpu_kernel1(uint32_t iteration, struct __FIXME__l_struct_struct_OC_dcomplex* u1, struct __FIXME__l_struct_struct_OC_dcomplex* sums, uint32_t __FIXME__gridDim_2e_x, uint32_t __FIXME__gridDim_2e_y, uint32_t __FIXME__gridDim_2e_z, uint32_t __FIXME__blockDim_2e_x, uint32_t __FIXME__blockDim_2e_y, uint32_t __FIXME__blockDim_2e_z, uint32_t __FIXME__blockIdx_2e_x, uint32_t __FIXME__blockIdx_2e_y, uint32_t __FIXME__blockIdx_2e_z, uint32_t __FIXME__threadIdx_2e_x, uint32_t __FIXME__threadIdx_2e_y, uint32_t __FIXME__threadIdx_2e_z) {
  struct __FIXME__l_struct_struct_OC_dcomplex __FIXME__ref_2e_tmp24;    /* Address-exposed local */
  int64_t i;

// INSERT COMMENT IFELSE: checksum_gpu_kernel1::syncpoint.1
  /*__FIXME__INTRINSIC_CALL__*///sync point
;
  if (__FIXME__threadIdx_2e_x == 0) { // IFELSE MARKER: syncpoint.1 IF
for(int64_t i = 1; ((uint64_t)i) < __FIXME__blockDim_2e_x;   i = i + 1) {
  __FIXME__ref_2e_tmp24.__FIXME__l_struct_struct_OC_dcomplex_field0 = (*((&((struct __FIXME__l_struct_struct_OC_dcomplex*)(&extern_share_data_shared))->__FIXME__l_struct_struct_OC_dcomplex_field0)) + *((&(((struct __FIXME__l_struct_struct_OC_dcomplex*)(&extern_share_data_shared))+i)->__FIXME__l_struct_struct_OC_dcomplex_field0)));
  __FIXME__ref_2e_tmp24.__FIXME__l_struct_struct_OC_dcomplex_field1 = (*((&((struct __FIXME__l_struct_struct_OC_dcomplex*)(&extern_share_data_shared))->__FIXME__l_struct_struct_OC_dcomplex_field1)) + *((&(((struct __FIXME__l_struct_struct_OC_dcomplex*)(&extern_share_data_shared))+i)->__FIXME__l_struct_struct_OC_dcomplex_field1)));
  memcpy(((uint8_t*)((struct __FIXME__l_struct_struct_OC_dcomplex*)(&extern_share_data_shared))), ((uint8_t*)(&__FIXME__ref_2e_tmp24)), 16);
}
  }
// INSERT COMMENT IFELSE: checksum_gpu_kernel1::if.end40
  if (__FIXME__threadIdx_2e_x == 0) { // IFELSE MARKER: if.end40 IF
  *((&((struct __FIXME__l_struct_struct_OC_dcomplex*)(&extern_share_data_shared))->__FIXME__l_struct_struct_OC_dcomplex_field0)) = (*((&((struct __FIXME__l_struct_struct_OC_dcomplex*)(&extern_share_data_shared))->__FIXME__l_struct_struct_OC_dcomplex_field0)) / 8388608);
  #pragma omp atomic update
  *(&(sums+iteration)->__FIXME__l_struct_struct_OC_dcomplex_field0) += *((&((struct __FIXME__l_struct_struct_OC_dcomplex*)(&extern_share_data_shared))->__FIXME__l_struct_struct_OC_dcomplex_field0));
  *((&((struct __FIXME__l_struct_struct_OC_dcomplex*)(&extern_share_data_shared))->__FIXME__l_struct_struct_OC_dcomplex_field1)) = (*((&((struct __FIXME__l_struct_struct_OC_dcomplex*)(&extern_share_data_shared))->__FIXME__l_struct_struct_OC_dcomplex_field1)) / 8388608);
  #pragma omp atomic update
  *(&(sums+iteration)->__FIXME__l_struct_struct_OC_dcomplex_field1) += *((&((struct __FIXME__l_struct_struct_OC_dcomplex*)(&extern_share_data_shared))->__FIXME__l_struct_struct_OC_dcomplex_field1));
  }
  return;
}
// FUNCTION ORDER ID 38 END

