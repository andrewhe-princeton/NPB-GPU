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

static __forceinline int llvm_fcmp_ole(double X, double Y) { return X <= Y; }
static __forceinline int llvm_fcmp_ogt(double X, double Y) { return X >  Y; }


/* Global Declarations */

/* Types Declarations */
struct __FIXME__l_struct_struct_OC_cudaDeviceProp;
struct __FIXME__l_struct_struct_OC_dim3;
struct __FIXME__l_unnamed_1;

/* Function definitions */

/* Types Definitions */
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

/* Function Declarations */
double randlc_device(double*, double) __ATTRIBUTELIST__((noinline, nothrow));
void vranlc_device(uint32_t, double*, double, double*) __ATTRIBUTELIST__((noinline, nothrow));
void _GLOBAL__sub_I_ep_OC_cu(void) __ATTRIBUTELIST__((noinline));
void __cxx_global_var_init(void) __ATTRIBUTELIST__((noinline));
double randlc(double*, double) __ATTRIBUTELIST__((noinline, nothrow));
void c_print_results(uint8_t*, int8_t, uint32_t, uint32_t, uint32_t, uint32_t, double, double, uint8_t*, uint32_t, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*) __ATTRIBUTELIST__((noinline));
double pow(double, double) __ATTRIBUTELIST__((nothrow));
int main(int, char **) __ATTRIBUTELIST__((noinline));
void setup_gpu(void) __ATTRIBUTELIST__((noinline));
void release_gpu(void) __ATTRIBUTELIST__((noinline));
double log(double);
double sqrt(double);
double fabs(double);
void gpu_kernel(double*, double*, double*, double, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) __ATTRIBUTELIST__((noinline, nothrow));


/* Global Variable Definitions and Initialization */
double* q_host;
double* q_device;
double* sx_host;
double* sx_device;
double* sy_host;
double* sy_device;
uint32_t threads_per_block;
uint32_t blocks_per_grid;
uint64_t size_q;
uint64_t size_sx;
uint64_t size_sy;
uint32_t gpu_device_id;
uint32_t total_devices;
struct __FIXME__l_struct_struct_OC_cudaDeviceProp gpu_device_properties;
double* _ZL1q;
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
uint8_t _OC_str_OC_39[7] = { "%15.0f" };
uint8_t _OC_str_OC_40[65] = { "\n\n NAS Parallel Benchmarks 4.1 CUDA C++ version - EP Benchmark\n\n" };
uint8_t _OC_str_OC_41[43] = { " Number of random numbers generated: %15s\n" };
uint8_t _OC_str_OC_42[26] = { "\n EP Benchmark Results:\n\n" };
uint8_t _OC_str_OC_43[19] = { " CPU Time =%10.4f\n" };
uint8_t _OC_str_OC_44[12] = { " N = 2^%5d\n" };
uint8_t _OC_str_OC_45[30] = { " No. Gaussian Pairs = %15.0f\n" };
uint8_t _OC_str_OC_46[25] = { " Sums = %25.15e %25.15e\n" };
uint8_t _OC_str_OC_47[11] = { " Counts: \n" };
uint8_t _OC_str_OC_48[11] = { "%3d%15.0f\n" };
uint8_t _OC_str_OC_49[10] = { "%5s\t%25s\n" };
uint8_t _OC_str_OC_50[11] = { "GPU Kernel" };
uint8_t _OC_str_OC_51[18] = { "Threads Per Block" };
uint8_t _OC_str_OC_52[11] = { "%29s\t%25d\n" };
uint8_t _OC_str_OC_53[4] = { " ep" };
uint8_t _OC_str_OC_54[3] = { "EP" };
uint8_t _OC_str_OC_55[25] = { "Random numbers generated" };
uint8_t _OC_str_OC_56[4] = { "4.1" };
uint8_t _OC_str_OC_57[12] = { "04 Feb 2026" };
uint8_t _OC_str_OC_58[6] = { "\x93\xE3\xBC\xFD\x7F" };
uint8_t _OC_str_OC_59[42] = { "Intel(R) Xeon(R) CPU E5-2697 v3 @ 2.60GHz" };
uint8_t _OC_str_OC_60[23] = { "${NVCC} ${EXTRA_STUFF}" };
uint8_t _OC_str_OC_61[6] = { "$(CC)" };
uint8_t _OC_str_OC_62[5] = { "-lm " };
uint8_t _OC_str_OC_63[13] = { "-I../common " };
uint8_t _OC_str_OC_64[4] = { "-O3" };
uint8_t _OC_str_OC_65[7] = { "randdp" };


/* LLVM Intrinsic Builtin Function Bodies */
static __forceinline uint32_t llvm_add_u32(uint32_t a, uint32_t b) {
  uint32_t r = a + b;
  return r;
}
static __forceinline uint64_t llvm_add_u64(uint64_t a, uint64_t b) {
  uint64_t r = a + b;
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
static __forceinline double llvm_OC_fabs_OC_f64(double a) {
  double r;
  r = fabs(a);
  return r;
}
static __forceinline double llvm_OC_ceil_OC_f64(double a) {
  double r;
  r = ceil(a);
  return r;
}


/* Function Bodies */

// FUNCTION ORDER ID 0 START
// INSERT COMMENT FUNCTION: randlc_device
double randlc_device(double* x, double a) {
  double a2;
  double x2;
  double t1;
  double t3;

  a2 = (a - (8388608 * (double)(((int32_t)(1.1920928955078125E-7 * a)))));
  x2 = (*x - (8388608 * (double)(((int32_t)(1.1920928955078125E-7 * *x)))));
  t1 = (((double)(((int32_t)(1.1920928955078125E-7 * a))) * x2) + (a2 * (double)(((int32_t)(1.1920928955078125E-7 * *x)))));
  t3 = ((8388608 * (t1 - (8388608 * (double)(((int32_t)(1.1920928955078125E-7 * t1)))))) + (a2 * x2));
  *x = (t3 - (70368744177664 * (double)(((int32_t)(1.4210854715202004E-14 * t3)))));
  return (1.4210854715202004E-14 * *x);
}
// FUNCTION ORDER ID 0 END


// FUNCTION ORDER ID 1 START
// INSERT COMMENT FUNCTION: vranlc_device
void vranlc_device(uint32_t n, double* x_seed, double a, double* y) {
  double a2;
  int64_t i;
  double x;

  a2 = (a - (8388608 * (double)(((int32_t)(1.1920928955078125E-7 * a)))));
  x = *x_seed;
// INSERT COMMENT LOOP: vranlc_device::for.cond
for(int64_t i = 0; i < n;   i = i + 1) {
  double x2 = (x - (8388608 * (double)(((int32_t)(1.1920928955078125E-7 * x)))));
  double t1 = (((double)(((int32_t)(1.1920928955078125E-7 * a))) * x2) + (a2 * (double)(((int32_t)(1.1920928955078125E-7 * x)))));
  double t3 = ((8388608 * (t1 - (8388608 * (double)(((int32_t)(1.1920928955078125E-7 * t1)))))) + (a2 * x2));
  x = (t3 - (70368744177664 * (double)(((int32_t)(1.4210854715202004E-14 * t3)))));
  y[i] = (1.4210854715202004E-14 * x);
  x = x;
}
  *x_seed = x;
}
// FUNCTION ORDER ID 1 END


// FUNCTION ORDER ID 2 START
// INSERT COMMENT FUNCTION: __cxx_global_var_init
void __cxx_global_var_init(void) {
  uint8_t* __FIXME__call;

  __FIXME__call = malloc(80);
  _ZL1q = ((double*)__FIXME__call);
}
// FUNCTION ORDER ID 2 END


// FUNCTION ORDER ID 3 START
// INSERT COMMENT FUNCTION: randlc
double randlc(double* x, double a) {
  double a2;
  double x2;
  double t1;
  double t3;

  a2 = (a - (8388608 * (double)(((int32_t)(1.1920928955078125E-7 * a)))));
  x2 = (*x - (8388608 * (double)(((int32_t)(1.1920928955078125E-7 * *x)))));
  t1 = (((double)(((int32_t)(1.1920928955078125E-7 * a))) * x2) + (a2 * (double)(((int32_t)(1.1920928955078125E-7 * *x)))));
  t3 = ((8388608 * (t1 - (8388608 * (double)(((int32_t)(1.1920928955078125E-7 * t1)))))) + (a2 * x2));
  *x = (t3 - (70368744177664 * (double)(((int32_t)(1.4210854715202004E-14 * t3)))));
  return (1.4210854715202004E-14 * *x);
}
// FUNCTION ORDER ID 3 END


// FUNCTION ORDER ID 4 START
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
  } else { // IFELSE MARKER: land.lhs.true ELSE
  if (n2 == 0) { // IFELSE MARKER: if.else15 IF
  if (n3 == 0) { // IFELSE MARKER: land.lhs.true17 IF
  if (name[0] == 69) { // IFELSE MARKER: if.then19 IF
  if (name[1] == 80) { // IFELSE MARKER: land.lhs.true23 IF
  __FIXME__call29 = pow(2, (double)(n1));
  sprintf(size, (_OC_str_OC_4), __FIXME__call29);
  if (size[14] == 46) { // IFELSE MARKER: if.then27 IF
  size[14] = 32;
  j = 13;
  }
  size[(j + 1)] = 0;
  printf((_OC_str_OC_5), size);
  } else { // IFELSE MARKER: land.lhs.true23 ELSE
  printf((_OC_str_OC_6), n1);
  }
  } else { // IFELSE MARKER: if.then19 ELSE
  printf((_OC_str_OC_6), n1);
  }
  } else { // IFELSE MARKER: land.lhs.true17 ELSE
  printf((_OC_str_OC_7), n1, n2, n3);
  }
  } else { // IFELSE MARKER: if.else15 ELSE
  printf((_OC_str_OC_7), n1, n2, n3);
  }
  }
  } else { // IFELSE MARKER: entry ELSE
  if (n2 == 0) { // IFELSE MARKER: if.else15 IF
  if (n3 == 0) { // IFELSE MARKER: land.lhs.true17 IF
  if (name[0] == 69) { // IFELSE MARKER: if.then19 IF
  if (name[1] == 80) { // IFELSE MARKER: land.lhs.true23 IF
  __FIXME__call29 = pow(2, (double)(n1));
  sprintf(size, (_OC_str_OC_4), __FIXME__call29);
  if (size[14] == 46) { // IFELSE MARKER: if.then27 IF
  size[14] = 32;
  j = 13;
  }
  size[(j + 1)] = 0;
  printf((_OC_str_OC_5), size);
  } else { // IFELSE MARKER: land.lhs.true23 ELSE
  printf((_OC_str_OC_6), n1);
  }
  } else { // IFELSE MARKER: if.then19 ELSE
  printf((_OC_str_OC_6), n1);
  }
  } else { // IFELSE MARKER: land.lhs.true17 ELSE
  printf((_OC_str_OC_7), n1, n2, n3);
  }
  } else { // IFELSE MARKER: if.else15 ELSE
  printf((_OC_str_OC_7), n1, n2, n3);
  }
  }
  printf((_OC_str_OC_8), niter);
  printf((_OC_str_OC_9), t);
  printf((_OC_str_OC_10), mops);
  printf((_OC_str_OC_11), optype);
  if (passed_verification < 0) {
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
// FUNCTION ORDER ID 4 END


// MAIN START
int main(int argc, char ** argv) {
  double t1;    /* Address-exposed local */
  uint8_t size[16];    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dim3 __FIXME__agg_2e_tmp;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dim3 __FIXME__agg_2e_tmp17;    /* Address-exposed local */
  struct __FIXME__l_unnamed_1 __FIXME__agg_2e_tmp_2e_coerce;    /* Address-exposed local */
  struct __FIXME__l_unnamed_1 __FIXME__agg_2e_tmp17_2e_coerce;    /* Address-exposed local */
  uint8_t gpu_config[256];    /* Address-exposed local */
  uint8_t gpu_config_string[2048];    /* Address-exposed local */
  double __FIXME__call;
  uint32_t j;
  int32_t i;
  int64_t block;
  double sy;
  double sx;
  double gc;
  double sx_err;
  double sy_err;
  bool __FIXME__1;
  uint32_t verified;
  double __FIXME__call63;
  double __FIXME__div64;

// INSERT COMMENT IFELSE: main::entry
  __FIXME__call = pow(2, 29);
  sprintf(size, (_OC_str_OC_39), __FIXME__call);
  if (size[14] == 46) { // IFELSE MARKER: entry IF
  j = 13;
  }
  size[(j + 1)] = 0;
  printf((_OC_str_OC_40));
  printf((_OC_str_OC_41), size);
  t1 = 1220703125;
// INSERT COMMENT LOOP: main::for.cond
for(int32_t i = 0; i < 17;   i = i + 1) {
  randlc((&t1), t1);
}
// INSERT COMMENT LOOP: main::for.cond9
for(int64_t i = 0; i < 10;   i = i + 1) {
  _ZL1q[i] = 0;
}
setup_gpu();
  __FIXME__agg_2e_tmp.__FIXME__l_struct_struct_OC_dim3_field0 = blocks_per_grid;
  __FIXME__agg_2e_tmp.__FIXME__l_struct_struct_OC_dim3_field1 = 1;
  __FIXME__agg_2e_tmp.__FIXME__l_struct_struct_OC_dim3_field2 = 1;
  __FIXME__agg_2e_tmp17.__FIXME__l_struct_struct_OC_dim3_field0 = threads_per_block;
  __FIXME__agg_2e_tmp17.__FIXME__l_struct_struct_OC_dim3_field1 = 1;
  __FIXME__agg_2e_tmp17.__FIXME__l_struct_struct_OC_dim3_field2 = 1;
  memcpy(((uint8_t*)(&__FIXME__agg_2e_tmp_2e_coerce)), ((uint8_t*)(&__FIXME__agg_2e_tmp)), 12);
  memcpy(((uint8_t*)(&__FIXME__agg_2e_tmp17_2e_coerce)), ((uint8_t*)(&__FIXME__agg_2e_tmp17)), 12);
// INSERT COMMENT LOOP: main::header.0
#pragma omp parallel for collapse(2)
for(int32_t i = 0; i < blocks_per_grid;   i = i + 1) {
for(int32_t j = 0; j < threads_per_block;   j = j + 1) {
gpu_kernel(q_host, sx_host, sy_host, t1, blocks_per_grid, 1, 1, threads_per_block, 1, 1, i, 0, 0, j, 0, 0);
}
}
  sy = 0;
  sx = 0;
// INSERT COMMENT LOOP: main::for.cond22
for(int64_t block = 0; block < blocks_per_grid;   block = block + 1) {
for(int64_t i = 0; i < 10;   i = i + 1) {
  _ZL1q[i] = (_ZL1q[i] + q_host[(block * 10 + i)]);
}
  sx = (sx + sx_host[block]);
  sy = (sy + sy_host[block]);
  sy = sy;
  sx = sx;
}
  gc = 0;
// INSERT COMMENT LOOP: main::for.cond46
for(int64_t i = 0; i < 10;   i = i + 1) {
  gc = (gc + _ZL1q[i]);
  gc = gc;
}
// INSERT COMMENT IFELSE: main::for.end54
  if (1 != 0) { // IFELSE MARKER: for.end54 IF
  sx_err = /*__FIXME__INTRINSIC_CALL__*/llvm_OC_fabs_OC_f64(((sx - -4295.8751656298919) / -4295.8751656298919));
  sy_err = /*__FIXME__INTRINSIC_CALL__*/llvm_OC_fabs_OC_f64(((sy - -15807.325736784311) / -15807.325736784311));
  if (llvm_fcmp_ole(sx_err, 1.0E-8)) { // IFELSE MARKER: if.then56 IF
  }
  }
  __FIXME__call63 = pow(2, 29);
  __FIXME__div64 = (__FIXME__call63 / /*UNDEF*/0);
  printf((_OC_str_OC_42));
  printf((_OC_str_OC_43), /*UNDEF*/0);
  printf((_OC_str_OC_44), 28);
  printf((_OC_str_OC_45), gc);
  printf((_OC_str_OC_46), sx, sy);
  printf((_OC_str_OC_47));
// INSERT COMMENT LOOP: main::for.cond72
for(int64_t i = 0; i < 10;   i = i + 1) {
  printf((_OC_str_OC_48), i, _ZL1q[i]);
}
  sprintf(gpu_config, (_OC_str_OC_49), (_OC_str_OC_50), (_OC_str_OC_51));
  strcpy(gpu_config_string, gpu_config);
  sprintf(gpu_config, (_OC_str_OC_52), (_OC_str_OC_53), threads_per_block);
  strcat(gpu_config_string, gpu_config);
c_print_results((_OC_str_OC_54), 65, 29, 0, 0, 0, /*UNDEF*/0, (__FIXME__div64 / 1.0E+6), (_OC_str_OC_55), verified, (_OC_str_OC_56), (_OC_str_OC_57), (_OC_str_OC_58), (_OC_str_OC_58), (_OC_str_OC_59), (&gpu_device_properties.__FIXME__l_struct_struct_OC_cudaDeviceProp_field0), gpu_config_string, (_OC_str_OC_60), (_OC_str_OC_61), (_OC_str_OC_62), (_OC_str_OC_63), (_OC_str_OC_64), (_OC_str_OC_64), (_OC_str_OC_65));
release_gpu();
  return 0;
}
// MAIN END


// FUNCTION ORDER ID 5 START
// INSERT COMMENT FUNCTION: setup_gpu
void setup_gpu(void) {
  double __FIXME__4;
  uint8_t* __FIXME__call;
  uint8_t* __FIXME__call8;
  uint8_t* __FIXME__call9;

  if (32 <= *(&gpu_device_properties.__FIXME__l_struct_struct_OC_cudaDeviceProp_field6)) {
  return;
  }
  threads_per_block = *(&gpu_device_properties.__FIXME__l_struct_struct_OC_cudaDeviceProp_field4);
  __FIXME__4 = /*__FIXME__INTRINSIC_CALL__*/llvm_OC_ceil_OC_f64((4096 / (double)(threads_per_block)));
  blocks_per_grid = ((int32_t)__FIXME__4);
  size_q = blocks_per_grid * 10 * 8;
  size_sx = blocks_per_grid * 8;
  size_sy = blocks_per_grid * 8;
  __FIXME__call = malloc(size_q);
  q_host = ((double*)__FIXME__call);
  __FIXME__call8 = malloc(size_sx);
  sx_host = ((double*)__FIXME__call8);
  __FIXME__call9 = malloc(size_sy);
  sy_host = ((double*)__FIXME__call9);
}
// FUNCTION ORDER ID 5 END


// FUNCTION ORDER ID 6 START
// INSERT COMMENT FUNCTION: release_gpu
void release_gpu(void) {
  return;
}
// FUNCTION ORDER ID 6 END


// FUNCTION ORDER ID 7 START
// INSERT COMMENT FUNCTION: gpu_kernel
void gpu_kernel(double* q_global, double* sx_global, double* sy_global, double an, uint32_t __FIXME__gridDim_2e_x, uint32_t __FIXME__gridDim_2e_y, uint32_t __FIXME__gridDim_2e_z, uint32_t __FIXME__blockDim_2e_x, uint32_t __FIXME__blockDim_2e_y, uint32_t __FIXME__blockDim_2e_z, uint32_t __FIXME__blockIdx_2e_x, uint32_t __FIXME__blockIdx_2e_y, uint32_t __FIXME__blockIdx_2e_z, uint32_t __FIXME__threadIdx_2e_x, uint32_t __FIXME__threadIdx_2e_y, uint32_t __FIXME__threadIdx_2e_z) {
  double x_local[256];    /* Address-exposed local */
  double q_local[10];    /* Address-exposed local */
  double t1;    /* Address-exposed local */
  double t2;    /* Address-exposed local */
  double seed;    /* Address-exposed local */
  int32_t kk;
  uint32_t i;
  double sx_local;
  double sy_local;
  uint32_t ii;
  double __FIXME__cond;

  q_local[0] = 0;
  q_local[1] = 0;
  q_local[2] = 0;
  q_local[3] = 0;
  q_local[4] = 0;
  q_local[5] = 0;
  q_local[6] = 0;
  q_local[7] = 0;
  q_local[8] = 0;
  q_local[9] = 0;
  kk = __FIXME__blockIdx_2e_x * __FIXME__blockDim_2e_x + __FIXME__threadIdx_2e_x;
  if (kk >= 4096) {
  return;
  }
  t1 = 271828183;
  t2 = an;
// INSERT COMMENT LOOP: gpu_kernel::for.cond
for(int32_t i = 1; i < 101;   i = i + 1) {
  uint32_t __FIXME__div = kk / 2;
  if (2 * __FIXME__div != kk) { // IFELSE MARKER: for.body IF
  randlc_device((&t1), t2);
  }
  if (__FIXME__div == 0) {
  break;
  }
  randlc_device((&t2), t2);
  kk = __FIXME__div;
}
  seed = t1;
  sx_local = 0;
  sy_local = 0;
// INSERT COMMENT LOOP: gpu_kernel::for.cond22
for(int32_t ii = 0; ii < 65536;   ii = ii + 128) {
vranlc_device(256, (&seed), 1220703125, x_local);
for(int64_t i = 0; i < 128;   i = i + 1) {
  double x1 = ((2 * x_local[2 * i]) - 1);
  double x2 = ((2 * x_local[(2 * i + 1)]) - 1);
  t1 = ((x1 * x1) + (x2 * x2));
  if (llvm_fcmp_ole(t1, 1)) { // IFELSE MARKER: for.body27 IF
  double __FIXME__log_result = log(t1);
  double __FIXME__5 = sqrt(((-2 * __FIXME__log_result) / t1));
  t2 = __FIXME__5;
  double t3 = (x1 * t2);
  double __FIXME__mul47 = (x2 * t2);
  double __FIXME__6 = fabs(t3);
  double __FIXME__7 = fabs(__FIXME__mul47);
  if (llvm_fcmp_ogt(__FIXME__6, __FIXME__7)) { // IFELSE MARKER: _ZL3logd.exit IF
  __FIXME__cond = fabs(t3);
  q_local[((int32_t)__FIXME__cond)] = (q_local[((int32_t)__FIXME__cond)] + 1);
  sx_local = (sx_local + t3);
  sy_local = (sy_local + __FIXME__mul47);
  } else { // IFELSE MARKER: _ZL3logd.exit ELSE
  __FIXME__cond = fabs(__FIXME__mul47);
  q_local[((int32_t)__FIXME__cond)] = (q_local[((int32_t)__FIXME__cond)] + 1);
  sx_local = (sx_local + t3);
  sy_local = (sy_local + __FIXME__mul47);
  }
  }
  sx_local = sx_local;
  sy_local = sy_local;
}
  sx_local = sx_local;
  sy_local = sy_local;
}
  return;
}
// FUNCTION ORDER ID 7 END

