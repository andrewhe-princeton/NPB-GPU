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
static __forceinline int llvm_fcmp_une(double X, double Y) { return X != Y; }


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
extern char /* (empty) */ extern_share_data;

/* Function Declarations */
void _GLOBAL__sub_I_cg_OC_cu(void) __ATTRIBUTELIST__((noinline));
void __cxx_global_var_init(void) __ATTRIBUTELIST__((noinline));
void __cxx_global_var_init_OC_1(void) __ATTRIBUTELIST__((noinline));
void __cxx_global_var_init_OC_2(void) __ATTRIBUTELIST__((noinline));
void __cxx_global_var_init_OC_3(void) __ATTRIBUTELIST__((noinline));
void __cxx_global_var_init_OC_4(void) __ATTRIBUTELIST__((noinline));
void __cxx_global_var_init_OC_5(void) __ATTRIBUTELIST__((noinline));
void __cxx_global_var_init_OC_6(void) __ATTRIBUTELIST__((noinline));
void __cxx_global_var_init_OC_7(void) __ATTRIBUTELIST__((noinline));
void __cxx_global_var_init_OC_8(void) __ATTRIBUTELIST__((noinline));
void __cxx_global_var_init_OC_9(void) __ATTRIBUTELIST__((noinline));
void __cxx_global_var_init_OC_10(void) __ATTRIBUTELIST__((noinline));
void __cxx_global_var_init_OC_11(void) __ATTRIBUTELIST__((noinline));
double randlc(double*, double) __ATTRIBUTELIST__((noinline, nothrow));
void c_print_results(uint8_t*, int8_t, uint32_t, uint32_t, uint32_t, uint32_t, double, double, uint8_t*, uint32_t, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*) __ATTRIBUTELIST__((noinline));
double pow(double, double) __ATTRIBUTELIST__((nothrow));
int main(int, char **) __ATTRIBUTELIST__((noinline));
void makea(uint32_t, uint32_t, double*, uint32_t*, uint32_t*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t*, uint32_t*, double*, uint32_t*) __ATTRIBUTELIST__((noinline));
void conj_grad(uint32_t*, uint32_t*, double*, double*, double*, double*, double*, double*, double*) __ATTRIBUTELIST__((noinline, nothrow));
double sqrt(double) __ATTRIBUTELIST__((nothrow));
void setup_gpu(void) __ATTRIBUTELIST__((noinline));
void conj_grad_gpu(double*) __ATTRIBUTELIST__((noinline));
void gpu_kernel_ten_host(double*, double*) __ATTRIBUTELIST__((noinline));
void gpu_kernel_eleven_host(double) __ATTRIBUTELIST__((noinline));
void release_gpu(void) __ATTRIBUTELIST__((noinline));
void gpu_kernel_one_host(void) __ATTRIBUTELIST__((noinline));
void gpu_kernel_two_host(double*) __ATTRIBUTELIST__((noinline));
void gpu_kernel_three_host(void) __ATTRIBUTELIST__((noinline));
void gpu_kernel_four_host(double*) __ATTRIBUTELIST__((noinline));
void gpu_kernel_five_host(double) __ATTRIBUTELIST__((noinline));
void gpu_kernel_six_host(double*) __ATTRIBUTELIST__((noinline));
void gpu_kernel_seven_host(double) __ATTRIBUTELIST__((noinline));
void gpu_kernel_eight_host(void) __ATTRIBUTELIST__((noinline));
void gpu_kernel_nine_host(double*) __ATTRIBUTELIST__((noinline));
void sprnvc(uint32_t, uint32_t, uint32_t, double*, uint32_t*) __ATTRIBUTELIST__((noinline, nothrow));
void vecset(uint32_t, double*, uint32_t*, uint32_t*, uint32_t, double) __ATTRIBUTELIST__((noinline, nothrow));
void sparse(double*, uint32_t*, uint32_t*, uint32_t, uint32_t, uint32_t, uint32_t*, uint32_t*, double*, uint32_t, uint32_t, uint32_t*, double, double) __ATTRIBUTELIST__((noinline));
void gpu_kernel_ten_10(double*, double*, double*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) __ATTRIBUTELIST__((noinline, nothrow));
void gpu_kernel_ten_20(double*, double*, double*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) __ATTRIBUTELIST__((noinline, nothrow));
void gpu_kernel_eleven_device(double, double*, double*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) __ATTRIBUTELIST__((noinline, nothrow));
void gpu_kernel_one_device(double*, double*, double*, double*, double*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) __ATTRIBUTELIST__((noinline, nothrow));
void gpu_kernel_two_device0(double*, double*, double*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) __ATTRIBUTELIST__((noinline, nothrow));
void gpu_kernel_three_device0(uint32_t*, uint32_t*, double*, double*, double*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) __ATTRIBUTELIST__((noinline, nothrow));
void gpu_kernel_four_device0(double*, double*, double*, double*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) __ATTRIBUTELIST__((noinline, nothrow));
void gpu_kernel_five_1(double, double*, double*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) __ATTRIBUTELIST__((noinline, nothrow));
void gpu_kernel_five_2(double, double*, double*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) __ATTRIBUTELIST__((noinline, nothrow));
void gpu_kernel_six_device0(double*, double*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) __ATTRIBUTELIST__((noinline, nothrow));
void gpu_kernel_seven_device(double, double*, double*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) __ATTRIBUTELIST__((noinline, nothrow));
void gpu_kernel_eight_device0(uint32_t*, uint32_t*, double*, double*, double*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) __ATTRIBUTELIST__((noinline, nothrow));
void gpu_kernel_nine_device0(double*, double*, double*, double*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) __ATTRIBUTELIST__((noinline, nothrow));
void gpu_kernel_ten_21(double*, double*, double*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) __ATTRIBUTELIST__((noinline, nothrow));
void gpu_kernel_ten_11(double*, double*, double*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) __ATTRIBUTELIST__((noinline, nothrow));
void gpu_kernel_three_device1(uint32_t*, uint32_t*, double*, double*, double*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) __ATTRIBUTELIST__((noinline, nothrow));
void gpu_kernel_nine_device1(double*, double*, double*, double*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) __ATTRIBUTELIST__((noinline, nothrow));
void gpu_kernel_six_device1(double*, double*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) __ATTRIBUTELIST__((noinline, nothrow));
void gpu_kernel_eight_device1(uint32_t*, uint32_t*, double*, double*, double*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) __ATTRIBUTELIST__((noinline, nothrow));
void gpu_kernel_two_device1(double*, double*, double*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) __ATTRIBUTELIST__((noinline, nothrow));
void gpu_kernel_four_device1(double*, double*, double*, double*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) __ATTRIBUTELIST__((noinline, nothrow));


/* Global Variable Definitions and Initialization */
uint32_t* colidx_device;
uint32_t* rowstr_device;
double* a_device;
double* p_device;
double* q_device;
double* r_device;
double* x_device;
double* z_device;
double* rho_device;
double* d_device;
double* alpha_device;
double* beta_device;
double* sum_device;
double* norm_temp1_device;
double* norm_temp2_device;
double* global_data;
double* global_data_two;
double* global_data_device;
double* global_data_two_device;
double global_data_reduce;
double global_data_two_reduce;
uint64_t global_data_elements;
uint64_t size_global_data;
uint64_t size_colidx_device;
uint64_t size_rowstr_device;
uint64_t size_iv_device;
uint64_t size_arow_device;
uint64_t size_acol_device;
uint64_t size_aelt_device;
uint64_t size_a_device;
uint64_t size_x_device;
uint64_t size_z_device;
uint64_t size_p_device;
uint64_t size_q_device;
uint64_t size_r_device;
uint64_t size_rho_device;
uint64_t size_d_device;
uint64_t size_alpha_device;
uint64_t size_beta_device;
uint64_t size_sum_device;
uint64_t size_norm_temp1_device;
uint64_t size_norm_temp2_device;
uint32_t blocks_per_grid_on_kernel_one;
uint32_t blocks_per_grid_on_kernel_two;
uint32_t blocks_per_grid_on_kernel_three;
uint32_t blocks_per_grid_on_kernel_four;
uint32_t blocks_per_grid_on_kernel_five;
uint32_t blocks_per_grid_on_kernel_six;
uint32_t blocks_per_grid_on_kernel_seven;
uint32_t blocks_per_grid_on_kernel_eight;
uint32_t blocks_per_grid_on_kernel_nine;
uint32_t blocks_per_grid_on_kernel_ten;
uint32_t blocks_per_grid_on_kernel_eleven;
uint32_t threads_per_block_on_kernel_one;
uint32_t threads_per_block_on_kernel_two;
uint32_t threads_per_block_on_kernel_three;
uint32_t threads_per_block_on_kernel_four;
uint32_t threads_per_block_on_kernel_five;
uint32_t threads_per_block_on_kernel_six;
uint32_t threads_per_block_on_kernel_seven;
uint32_t threads_per_block_on_kernel_eight;
uint32_t threads_per_block_on_kernel_nine;
uint32_t threads_per_block_on_kernel_ten;
uint32_t threads_per_block_on_kernel_eleven;
uint64_t size_shared_data_on_kernel_one;
uint64_t size_shared_data_on_kernel_two;
uint64_t size_shared_data_on_kernel_three;
uint64_t size_shared_data_on_kernel_four;
uint64_t size_shared_data_on_kernel_five;
uint64_t size_shared_data_on_kernel_six;
uint64_t size_shared_data_on_kernel_seven;
uint64_t size_shared_data_on_kernel_eight;
uint64_t size_shared_data_on_kernel_nine;
uint64_t size_shared_data_on_kernel_ten;
uint64_t size_shared_data_on_kernel_eleven;
uint64_t size_reduce_memory_on_kernel_one;
uint64_t size_reduce_memory_on_kernel_two;
uint64_t size_reduce_memory_on_kernel_three;
uint64_t size_reduce_memory_on_kernel_four;
uint64_t size_reduce_memory_on_kernel_five;
uint64_t size_reduce_memory_on_kernel_six;
uint64_t size_reduce_memory_on_kernel_seven;
uint64_t size_reduce_memory_on_kernel_eight;
uint64_t size_reduce_memory_on_kernel_nine;
uint64_t size_reduce_memory_on_kernel_ten;
uint64_t size_reduce_memory_on_kernel_eleven;
uint32_t gpu_device_id;
uint32_t total_devices;
struct __FIXME__l_struct_struct_OC_cudaDeviceProp gpu_device_properties;
double* _ZL1r;
double* _ZL1q;
double* _ZL1p;
double* _ZL1z;
double* _ZL1x;
double* _ZL1a;
double* _ZL4aelt;
uint32_t* _ZL4acol;
uint32_t* _ZL4arow;
uint32_t* _ZL2iv;
uint32_t* _ZL6rowstr;
uint32_t* _ZL6colidx;
uint8_t _OC_str[27] = { "\n\n %s Benchmark Completed\n" };
uint8_t _OC_str_OC_12[46] = { " class_npb       =                        %c\n" };
uint8_t _OC_str_OC_13[38] = { " Size            =             %12ld\n" };
uint8_t _OC_str_OC_14[44] = { " Size            =             %4dx%4dx%4d\n" };
uint8_t _OC_str_OC_15[8] = { "%15.0lf" };
uint8_t _OC_str_OC_16[34] = { " Size            =          %15s\n" };
uint8_t _OC_str_OC_17[37] = { " Size            =             %12d\n" };
uint8_t _OC_str_OC_18[42] = { " Size            =           %4dx%4dx%4d\n" };
uint8_t _OC_str_OC_19[37] = { " Iterations      =             %12d\n" };
uint8_t _OC_str_OC_20[39] = { " Time in seconds =             %12.2f\n" };
uint8_t _OC_str_OC_21[39] = { " Mop/s total     =             %12.2f\n" };
uint8_t _OC_str_OC_22[25] = { " Operation type  = %24s\n" };
uint8_t _OC_str_OC_23[45] = { " Verification    =            NOT PERFORMED\n" };
uint8_t _OC_str_OC_24[45] = { " Verification    =               SUCCESSFUL\n" };
uint8_t _OC_str_OC_25[45] = { " Verification    =             UNSUCCESSFUL\n" };
uint8_t _OC_str_OC_26[37] = { " Version         =             %12s\n" };
uint8_t _OC_str_OC_27[37] = { " Compile date    =             %12s\n" };
uint8_t _OC_str_OC_28[37] = { " NVCC version    =             %12s\n" };
uint8_t _OC_str_OC_29[37] = { " CUDA version    =             %12s\n" };
uint8_t _OC_str_OC_30[20] = { "\n Compile options:\n" };
uint8_t _OC_str_OC_31[23] = { "    CC           = %s\n" };
uint8_t _OC_str_OC_32[23] = { "    CLINK        = %s\n" };
uint8_t _OC_str_OC_33[23] = { "    C_LIB        = %s\n" };
uint8_t _OC_str_OC_34[23] = { "    C_INC        = %s\n" };
uint8_t _OC_str_OC_35[23] = { "    CFLAGS       = %s\n" };
uint8_t _OC_str_OC_36[23] = { "    CLINKFLAGS   = %s\n" };
uint8_t _OC_str_OC_37[23] = { "    RAND         = %s\n" };
uint8_t _OC_str_OC_38[13] = { "\n Hardware:\n" };
uint8_t _OC_str_OC_39[23] = { "    CPU device   = %s\n" };
uint8_t _OC_str_OC_40[23] = { "    GPU device   = %s\n" };
uint8_t _OC_str_OC_41[13] = { "\n Software:\n" };
uint8_t _OC_str_OC_42[23] = { "    Parameters   = %s\n" };
uint8_t _OC_str_OC_43[2] = { "\n" };
uint8_t _OC_str_OC_44[72] = { "----------------------------------------------------------------------\n" };
uint8_t _OC_str_OC_45[27] = { " NPB-CPP is developed by:\n" };
uint8_t _OC_str_OC_46[56] = { "            Dalvan Griebler <dalvangriebler@gmail.com>\n" };
uint8_t _OC_str_OC_47[52] = { "            Gabriell Araujo <hexenoften@gmail.com>\n" };
uint8_t _OC_str_OC_48[46] = { "            J\xC3\xBAnior L\xC3\xB6\x66\x66 <loffjh@gmail.com>\n" };
uint8_t _OC_str_OC_49[43] = { " In case of problems, send an email to us\n" };
uint32_t _ZL8firstrow;
uint32_t _ZL7lastrow;
uint32_t _ZL8firstcol;
uint32_t _ZL7lastcol;
uint8_t _OC_str_OC_50[65] = { "\n\n NAS Parallel Benchmarks 4.1 CUDA C++ version - CG Benchmark\n\n" };
uint8_t _OC_str_OC_51[13] = { " Size: %11d\n" };
uint8_t _OC_str_OC_52[18] = { " Iterations: %5d\n" };
uint32_t _ZL3naa;
uint32_t _ZL3nzz;
double _ZL4tran;
double _ZL5amult;
uint8_t _OC_str_OC_53[52] = { "\n   iteration           ||r||                 zeta\n" };
uint8_t _OC_str_OC_54[30] = { "    %5d       %20.14e%20.13e\n" };
uint8_t _OC_str_OC_55[22] = { " Benchmark completed\n" };
uint8_t _OC_str_OC_56[26] = { " VERIFICATION SUCCESSFUL\n" };
uint8_t _OC_str_OC_57[21] = { " Zeta is    %20.13e\n" };
uint8_t _OC_str_OC_58[21] = { " Error is   %20.13e\n" };
uint8_t _OC_str_OC_59[22] = { " VERIFICATION FAILED\n" };
uint8_t _OC_str_OC_60[30] = { " Zeta                %20.13e\n" };
uint8_t _OC_str_OC_61[30] = { " The correct zeta is %20.13e\n" };
uint8_t _OC_str_OC_62[23] = { " Problem size unknown\n" };
uint8_t _OC_str_OC_63[28] = { " NO VERIFICATION PERFORMED\n" };
uint8_t _OC_str_OC_64[10] = { "%5s\t%25s\n" };
uint8_t _OC_str_OC_65[11] = { "GPU Kernel" };
uint8_t _OC_str_OC_66[18] = { "Threads Per Block" };
uint8_t _OC_str_OC_67[11] = { "%29s\t%25d\n" };
uint8_t _OC_str_OC_68[5] = { " one" };
uint8_t _OC_str_OC_69[5] = { " two" };
uint8_t _OC_str_OC_70[7] = { " three" };
uint8_t _OC_str_OC_71[6] = { " four" };
uint8_t _OC_str_OC_72[6] = { " five" };
uint8_t _OC_str_OC_73[5] = { " six" };
uint8_t _OC_str_OC_74[7] = { " seven" };
uint8_t _OC_str_OC_75[7] = { " eight" };
uint8_t _OC_str_OC_76[6] = { " nine" };
uint8_t _OC_str_OC_77[5] = { " ten" };
uint8_t _OC_str_OC_78[8] = { " eleven" };
uint8_t _OC_str_OC_79[3] = { "CG" };
uint8_t _OC_str_OC_80[25] = { "          floating point" };
uint8_t _OC_str_OC_81[4] = { "4.1" };
uint8_t _OC_str_OC_82[12] = { "01 Feb 2026" };
uint8_t _OC_str_OC_83[6] = { "^\xE6Q\xFC\x7F" };
uint8_t _OC_str_OC_84[42] = { "Intel(R) Xeon(R) CPU E5-2697 v3 @ 2.60GHz" };
uint8_t _OC_str_OC_85[23] = { "${NVCC} ${EXTRA_STUFF}" };
uint8_t _OC_str_OC_86[6] = { "$(CC)" };
uint8_t _OC_str_OC_87[5] = { "-lm " };
uint8_t _OC_str_OC_88[13] = { "-I../common " };
uint8_t _OC_str_OC_89[4] = { "-O3" };
uint8_t _OC_str_OC_90[7] = { "randdp" };
uint8_t _OC_str_OC_91[46] = { "Space for matrix elements exceeded in sparse\n" };
uint8_t _OC_str_OC_92[21] = { "nza, nzmax = %d, %d\n" };
uint8_t _OC_str_OC_93[32] = { "internal error in sparse: i=%d\n" };
double extern_share_data_shared[1024];


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
static __forceinline uint32_t llvm_udiv_u32(uint32_t a, uint32_t b) {
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
// INSERT COMMENT FUNCTION: __cxx_global_var_init
void __cxx_global_var_init(void) {
  uint8_t* __FIXME__call;

  __FIXME__call = malloc(8064000);
  _ZL6colidx = ((uint32_t*)__FIXME__call);
}
// FUNCTION ORDER ID 0 END


// FUNCTION ORDER ID 1 START
// INSERT COMMENT FUNCTION: __cxx_global_var_init.1
void __cxx_global_var_init_OC_1(void) {
  uint8_t* __FIXME__call;

  __FIXME__call = malloc(56004);
  _ZL6rowstr = ((uint32_t*)__FIXME__call);
}
// FUNCTION ORDER ID 1 END


// FUNCTION ORDER ID 2 START
// INSERT COMMENT FUNCTION: __cxx_global_var_init.2
void __cxx_global_var_init_OC_2(void) {
  uint8_t* __FIXME__call;

  __FIXME__call = malloc(56000);
  _ZL2iv = ((uint32_t*)__FIXME__call);
}
// FUNCTION ORDER ID 2 END


// FUNCTION ORDER ID 3 START
// INSERT COMMENT FUNCTION: __cxx_global_var_init.3
void __cxx_global_var_init_OC_3(void) {
  uint8_t* __FIXME__call;

  __FIXME__call = malloc(56000);
  _ZL4arow = ((uint32_t*)__FIXME__call);
}
// FUNCTION ORDER ID 3 END


// FUNCTION ORDER ID 4 START
// INSERT COMMENT FUNCTION: __cxx_global_var_init.4
void __cxx_global_var_init_OC_4(void) {
  uint8_t* __FIXME__call;

  __FIXME__call = malloc(672000);
  _ZL4acol = ((uint32_t*)__FIXME__call);
}
// FUNCTION ORDER ID 4 END


// FUNCTION ORDER ID 5 START
// INSERT COMMENT FUNCTION: __cxx_global_var_init.5
void __cxx_global_var_init_OC_5(void) {
  uint8_t* __FIXME__call;

  __FIXME__call = malloc(1344000);
  _ZL4aelt = ((double*)__FIXME__call);
}
// FUNCTION ORDER ID 5 END


// FUNCTION ORDER ID 6 START
// INSERT COMMENT FUNCTION: __cxx_global_var_init.6
void __cxx_global_var_init_OC_6(void) {
  uint8_t* __FIXME__call;

  __FIXME__call = malloc(16128000);
  _ZL1a = ((double*)__FIXME__call);
}
// FUNCTION ORDER ID 6 END


// FUNCTION ORDER ID 7 START
// INSERT COMMENT FUNCTION: __cxx_global_var_init.7
void __cxx_global_var_init_OC_7(void) {
  uint8_t* __FIXME__call;

  __FIXME__call = malloc(112016);
  _ZL1x = ((double*)__FIXME__call);
}
// FUNCTION ORDER ID 7 END


// FUNCTION ORDER ID 8 START
// INSERT COMMENT FUNCTION: __cxx_global_var_init.8
void __cxx_global_var_init_OC_8(void) {
  uint8_t* __FIXME__call;

  __FIXME__call = malloc(112016);
  _ZL1z = ((double*)__FIXME__call);
}
// FUNCTION ORDER ID 8 END


// FUNCTION ORDER ID 9 START
// INSERT COMMENT FUNCTION: __cxx_global_var_init.9
void __cxx_global_var_init_OC_9(void) {
  uint8_t* __FIXME__call;

  __FIXME__call = malloc(112016);
  _ZL1p = ((double*)__FIXME__call);
}
// FUNCTION ORDER ID 9 END


// FUNCTION ORDER ID 10 START
// INSERT COMMENT FUNCTION: __cxx_global_var_init.10
void __cxx_global_var_init_OC_10(void) {
  uint8_t* __FIXME__call;

  __FIXME__call = malloc(112016);
  _ZL1q = ((double*)__FIXME__call);
}
// FUNCTION ORDER ID 10 END


// FUNCTION ORDER ID 11 START
// INSERT COMMENT FUNCTION: __cxx_global_var_init.11
void __cxx_global_var_init_OC_11(void) {
  uint8_t* __FIXME__call;

  __FIXME__call = malloc(112016);
  _ZL1r = ((double*)__FIXME__call);
}
// FUNCTION ORDER ID 11 END


// FUNCTION ORDER ID 12 START
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
// FUNCTION ORDER ID 12 END


// FUNCTION ORDER ID 13 START
// INSERT COMMENT FUNCTION: c_print_results
void c_print_results(uint8_t* name, int8_t class_npb, uint32_t n1, uint32_t n2, uint32_t n3, uint32_t niter, double t, double mops, uint8_t* optype, uint32_t passed_verification, uint8_t* npbversion, uint8_t* compiletime, uint8_t* compilerversion, uint8_t* libversion, uint8_t* cpu_device, uint8_t* gpu_device, uint8_t* gpu_config, uint8_t* cc, uint8_t* clink, uint8_t* c_lib, uint8_t* c_inc, uint8_t* cflags, uint8_t* clinkflags, uint8_t* rand) {
  uint8_t size[16];    /* Address-exposed local */
  uint64_t nn;
  double __FIXME__call29;
  uint32_t j;

// INSERT COMMENT IFELSE: c_print_results::entry
  printf((_OC_str), name);
  printf((_OC_str_OC_12), class_npb);
  if (name[0] == 73) { // IFELSE MARKER: entry IF
  if (name[1] == 83) { // IFELSE MARKER: land.lhs.true IF
  if (n3 == 0) { // IFELSE MARKER: if.then IF
  if (n2 != 0) { // IFELSE MARKER: if.then7 IF
  nn = nn * n2;
  }
  printf((_OC_str_OC_13), nn);
  } else { // IFELSE MARKER: if.then ELSE
  printf((_OC_str_OC_14), n1, n2, n3);
  }
  } else { // IFELSE MARKER: land.lhs.true ELSE
  if (n2 == 0) { // IFELSE MARKER: if.else15 IF
  if (n3 == 0) { // IFELSE MARKER: land.lhs.true17 IF
  if (name[0] == 69) { // IFELSE MARKER: if.then19 IF
  if (name[1] == 80) { // IFELSE MARKER: land.lhs.true23 IF
  __FIXME__call29 = pow(2, (double)(n1));
  sprintf(size, (_OC_str_OC_15), __FIXME__call29);
  if (size[14] == 46) { // IFELSE MARKER: if.then27 IF
  size[14] = 32;
  j = 13;
  }
  size[(j + 1)] = 0;
  printf((_OC_str_OC_16), size);
  } else { // IFELSE MARKER: land.lhs.true23 ELSE
  printf((_OC_str_OC_17), n1);
  }
  } else { // IFELSE MARKER: if.then19 ELSE
  printf((_OC_str_OC_17), n1);
  }
  } else { // IFELSE MARKER: land.lhs.true17 ELSE
  printf((_OC_str_OC_18), n1, n2, n3);
  }
  } else { // IFELSE MARKER: if.else15 ELSE
  printf((_OC_str_OC_18), n1, n2, n3);
  }
  }
  } else { // IFELSE MARKER: entry ELSE
  if (n2 == 0) { // IFELSE MARKER: if.else15 IF
  if (n3 == 0) { // IFELSE MARKER: land.lhs.true17 IF
  if (name[0] == 69) { // IFELSE MARKER: if.then19 IF
  if (name[1] == 80) { // IFELSE MARKER: land.lhs.true23 IF
  __FIXME__call29 = pow(2, (double)(n1));
  sprintf(size, (_OC_str_OC_15), __FIXME__call29);
  if (size[14] == 46) { // IFELSE MARKER: if.then27 IF
  size[14] = 32;
  j = 13;
  }
  size[(j + 1)] = 0;
  printf((_OC_str_OC_16), size);
  } else { // IFELSE MARKER: land.lhs.true23 ELSE
  printf((_OC_str_OC_17), n1);
  }
  } else { // IFELSE MARKER: if.then19 ELSE
  printf((_OC_str_OC_17), n1);
  }
  } else { // IFELSE MARKER: land.lhs.true17 ELSE
  printf((_OC_str_OC_18), n1, n2, n3);
  }
  } else { // IFELSE MARKER: if.else15 ELSE
  printf((_OC_str_OC_18), n1, n2, n3);
  }
  }
  printf((_OC_str_OC_19), niter);
  printf((_OC_str_OC_20), t);
  printf((_OC_str_OC_21), mops);
  printf((_OC_str_OC_22), optype);
  if (passed_verification < 0) {
  return;
  }
// INSERT COMMENT IFELSE: c_print_results::if.else56
  if (passed_verification != 0) { // IFELSE MARKER: if.else56 IF
  printf((_OC_str_OC_24));
  } else { // IFELSE MARKER: if.else56 ELSE
  printf((_OC_str_OC_25));
  }
  printf((_OC_str_OC_26), npbversion);
  printf((_OC_str_OC_27), compiletime);
  printf((_OC_str_OC_28), compilerversion);
  printf((_OC_str_OC_29), libversion);
  printf((_OC_str_OC_30));
  printf((_OC_str_OC_31), cc);
  printf((_OC_str_OC_32), clink);
  printf((_OC_str_OC_33), c_lib);
  printf((_OC_str_OC_34), c_inc);
  printf((_OC_str_OC_35), cflags);
  printf((_OC_str_OC_36), clinkflags);
  printf((_OC_str_OC_37), rand);
  printf((_OC_str_OC_38));
  printf((_OC_str_OC_39), cpu_device);
  printf((_OC_str_OC_40), gpu_device);
  printf((_OC_str_OC_41));
  printf((_OC_str_OC_42), gpu_config);
  printf((_OC_str_OC_43));
  printf((_OC_str_OC_44));
  printf((_OC_str_OC_45));
  printf((_OC_str_OC_46));
  printf((_OC_str_OC_47));
  printf((_OC_str_OC_48));
  printf((_OC_str_OC_43));
  printf((_OC_str_OC_49));
  printf((_OC_str_OC_44));
  printf((_OC_str_OC_43));
}
// FUNCTION ORDER ID 13 END


// MAIN START
int main(int argc, char ** argv) {
  double rnorm;    /* Address-exposed local */
  double norm_temp1;    /* Address-exposed local */
  double norm_temp2;    /* Address-exposed local */
  uint8_t gpu_config[256];    /* Address-exposed local */
  uint8_t gpu_config_string[2048];    /* Address-exposed local */
  int64_t j;
  int64_t k;
  int64_t i;
  uint32_t it;
  double __FIXME__1;
  double err;
  uint32_t verified;
  double mflops;

  /* Initialize global arrays */
  __cxx_global_var_init();
  __cxx_global_var_init_OC_1();
  __cxx_global_var_init_OC_2();
  __cxx_global_var_init_OC_3();
  __cxx_global_var_init_OC_4();
  __cxx_global_var_init_OC_5();
  __cxx_global_var_init_OC_6();
  __cxx_global_var_init_OC_7();
  __cxx_global_var_init_OC_8();
  __cxx_global_var_init_OC_9();
  __cxx_global_var_init_OC_10();
  __cxx_global_var_init_OC_11();

  _ZL8firstrow = 0;
  _ZL7lastrow = 13999;
  _ZL8firstcol = 0;
  _ZL7lastcol = 13999;
  printf((_OC_str_OC_50));
  printf((_OC_str_OC_51), 14000);
  printf((_OC_str_OC_52), 15);
  _ZL3naa = 14000;
  _ZL3nzz = 2016000;
  _ZL4tran = 314159265;
  _ZL5amult = 1220703125;
  randlc((&_ZL4tran), _ZL5amult);
makea(_ZL3naa, _ZL3nzz, _ZL1a, _ZL6colidx, _ZL6rowstr, _ZL8firstrow, _ZL7lastrow, _ZL8firstcol, _ZL7lastcol, _ZL4arow, ((uint32_t*)((uint8_t*)_ZL4acol)), ((double*)((uint8_t*)_ZL4aelt)), _ZL2iv);
// INSERT COMMENT LOOP: main::for.cond
for(int64_t j = 0; j < (_ZL7lastrow - _ZL8firstrow) + 1;   j = j + 1) {
for(int64_t k = _ZL6rowstr[j]; k < _ZL6rowstr[(j + 1)];   k = k + 1) {
  _ZL6colidx[k] = (_ZL6colidx[k] - _ZL8firstcol);
}
}
// INSERT COMMENT LOOP: main::for.cond18
for(int64_t i = 0; i < 14001;   i = i + 1) {
  _ZL1x[i] = 1;
}
// INSERT COMMENT LOOP: main::for.cond26
for(int64_t j = 0; j < (_ZL7lastcol - _ZL8firstcol) + 1;   j = j + 1) {
  _ZL1q[j] = 0;
  _ZL1z[j] = 0;
  _ZL1r[j] = 0;
  _ZL1p[j] = 0;
}
// INSERT COMMENT LOOP: main::for.cond42
for(int32_t it = 1; it < 2;   it = it + 1) {
conj_grad(_ZL6colidx, _ZL6rowstr, _ZL1x, _ZL1z, _ZL1a, _ZL1p, _ZL1q, _ZL1r, (&rnorm));
  norm_temp1 = 0;
  norm_temp2 = 0;
for(int64_t j = 0; j < (_ZL7lastcol - _ZL8firstcol) + 1;   j = j + 1) {
  norm_temp1 = (norm_temp1 + (_ZL1x[j] * _ZL1z[j]));
  norm_temp2 = (norm_temp2 + (_ZL1z[j] * _ZL1z[j]));
}
  double __FIXME__call64 = sqrt(norm_temp2);
  norm_temp2 = (1 / __FIXME__call64);
for(int64_t j = 0; j < (_ZL7lastcol - _ZL8firstcol) + 1;   j = j + 1) {
  _ZL1x[j] = (norm_temp2 * _ZL1z[j]);
}
}
// INSERT COMMENT LOOP: main::for.cond81
for(int64_t i = 0; i < 14001;   i = i + 1) {
  _ZL1x[i] = 1;
}
setup_gpu();
  double zeta = 0;
// INSERT COMMENT LOOP: main::for.cond89
for(int32_t it = 1; it < 16;   it = it + 1) {
  conj_grad_gpu((&rnorm));
;
  gpu_kernel_ten_host((&norm_temp1), (&norm_temp2));
;
  double __FIXME__call92 = sqrt(norm_temp2);
  norm_temp2 = (1 / __FIXME__call92);
  zeta = (20 + (1 / norm_temp1));
  if (it == 1) { // IFELSE MARKER: for.body91 IF
  printf((_OC_str_OC_53));
  }
  printf((_OC_str_OC_54), it, rnorm, zeta);
gpu_kernel_eleven_host(norm_temp2);
}
// INSERT COMMENT IFELSE: main::for.end101
  printf((_OC_str_OC_55));
  if (65 != 85) { // IFELSE MARKER: for.end101 IF
  __FIXME__1 = /*__FIXME__INTRINSIC_CALL__*/llvm_OC_fabs_OC_f64((zeta - 17.130235054029001));
  err = (__FIXME__1 / 17.130235054029001);
  if (llvm_fcmp_ole(err, 1.0E-10)) { // IFELSE MARKER: if.then104 IF
  printf((_OC_str_OC_56));
  printf((_OC_str_OC_57), zeta);
  printf((_OC_str_OC_58), err);
  verified = 1;
  } else { // IFELSE MARKER: if.then104 ELSE
  printf((_OC_str_OC_59));
  printf((_OC_str_OC_60), zeta);
  printf((_OC_str_OC_61), 17.130235054029001);
  verified = 0;
  }
  } else { // IFELSE MARKER: for.end101 ELSE
  printf((_OC_str_OC_62));
  printf((_OC_str_OC_63));
  verified = 0;
  }
// INSERT COMMENT IFELSE: main::if.end119
  if (llvm_fcmp_une(0, 0)) { // IFELSE MARKER: if.end119 IF
  mflops = ((1.49646E+9 / 0) / 1.0E+6);
  } else { // IFELSE MARKER: if.end119 ELSE
  mflops = 0;
  }
  sprintf(gpu_config, (_OC_str_OC_64), (_OC_str_OC_65), (_OC_str_OC_66));
  strcpy(gpu_config_string, gpu_config);
  sprintf(gpu_config, (_OC_str_OC_67), (_OC_str_OC_68), threads_per_block_on_kernel_one);
  strcat(gpu_config_string, gpu_config);
  sprintf(gpu_config, (_OC_str_OC_67), (_OC_str_OC_69), threads_per_block_on_kernel_two);
  strcat(gpu_config_string, gpu_config);
  sprintf(gpu_config, (_OC_str_OC_67), (_OC_str_OC_70), threads_per_block_on_kernel_three);
  strcat(gpu_config_string, gpu_config);
  sprintf(gpu_config, (_OC_str_OC_67), (_OC_str_OC_71), threads_per_block_on_kernel_four);
  strcat(gpu_config_string, gpu_config);
  sprintf(gpu_config, (_OC_str_OC_67), (_OC_str_OC_72), threads_per_block_on_kernel_five);
  strcat(gpu_config_string, gpu_config);
  sprintf(gpu_config, (_OC_str_OC_67), (_OC_str_OC_73), threads_per_block_on_kernel_six);
  strcat(gpu_config_string, gpu_config);
  sprintf(gpu_config, (_OC_str_OC_67), (_OC_str_OC_74), threads_per_block_on_kernel_seven);
  strcat(gpu_config_string, gpu_config);
  sprintf(gpu_config, (_OC_str_OC_67), (_OC_str_OC_75), threads_per_block_on_kernel_eight);
  strcat(gpu_config_string, gpu_config);
  sprintf(gpu_config, (_OC_str_OC_67), (_OC_str_OC_76), threads_per_block_on_kernel_nine);
  strcat(gpu_config_string, gpu_config);
  sprintf(gpu_config, (_OC_str_OC_67), (_OC_str_OC_77), threads_per_block_on_kernel_ten);
  strcat(gpu_config_string, gpu_config);
  sprintf(gpu_config, (_OC_str_OC_67), (_OC_str_OC_78), threads_per_block_on_kernel_eleven);
  strcat(gpu_config_string, gpu_config);
c_print_results((_OC_str_OC_79), 65, 14000, 0, 0, 15, 0, mflops, (_OC_str_OC_80), verified, (_OC_str_OC_81), (_OC_str_OC_82), (_OC_str_OC_83), (_OC_str_OC_83), (_OC_str_OC_84), (&gpu_device_properties.__FIXME__l_struct_struct_OC_cudaDeviceProp_field0), gpu_config_string, (_OC_str_OC_85), (_OC_str_OC_86), (_OC_str_OC_87), (_OC_str_OC_88), (_OC_str_OC_89), (_OC_str_OC_89), (_OC_str_OC_90));
release_gpu();
  return 0;
}
// MAIN END


// FUNCTION ORDER ID 14 START
// INSERT COMMENT FUNCTION: makea
void makea(uint32_t n, uint32_t nz, double* a, uint32_t* colidx, uint32_t* rowstr, uint32_t firstrow, uint32_t lastrow, uint32_t firstcol, uint32_t lastcol, uint32_t* arow, uint32_t* acol, double* aelt, uint32_t* iv) {
  uint32_t nzv;    /* Address-exposed local */
  uint32_t ivc[12];    /* Address-exposed local */
  double vc[12];    /* Address-exposed local */
  uint32_t nn1;
  uint32_t __FIXME__mul_2e_lcssa;
  int64_t iouter;
  int64_t ivelt;

  nn1 = 1;
// INSERT COMMENT LOOP: makea::do.body
do {
  uint32_t __FIXME__mul = 2 * nn1;
  nn1 = __FIXME__mul;
} while(nn1 < n);
// INSERT COMMENT LOOP: makea::for.cond
for(int64_t iouter = 0; iouter < n;   iouter = iouter + 1) {
  nzv = 11;
sprnvc(n, nzv, nn1, vc, ivc);
vecset(n, vc, ivc, (&nzv), (iouter + 1), 0.5);
  arow[iouter] = nzv;
for(int64_t ivelt = 0; ivelt < nzv;   ivelt = ivelt + 1) {
  (acol+12*iouter)[ivelt] = (ivc[ivelt] - 1);
  (aelt+12*iouter)[ivelt] = vc[ivelt];
}
}
sparse(a, colidx, rowstr, n, nz, 11, arow, acol, aelt, firstrow, lastrow, iv, 0.10000000000000001, 20);
}
// FUNCTION ORDER ID 14 END


// FUNCTION ORDER ID 15 START
// INSERT COMMENT FUNCTION: conj_grad
void conj_grad(uint32_t* colidx, uint32_t* rowstr, double* x, double* z, double* a, double* p, double* q, double* r, double* rnorm) {
  int64_t j;
  double rho;
  uint32_t cgit;
  int64_t k;
  double sum;
  double d;
  double rho0;
  double alpha;
  double beta;
  double __FIXME__call;

// INSERT COMMENT LOOP: conj_grad::for.cond
for(int64_t j = 0; j < _ZL3naa + 1;   j = j + 1) {
  q[j] = 0;
  z[j] = 0;
  r[j] = x[j];
  p[j] = r[j];
}
  rho = 0;
// INSERT COMMENT LOOP: conj_grad::for.cond11
for(int64_t j = 0; j < (_ZL7lastcol - _ZL8firstcol) + 1;   j = j + 1) {
  rho = rho + (r[j] * r[j]);
}
// INSERT COMMENT LOOP: conj_grad::for.cond23
for(int32_t cgit = 1; cgit < 26;   cgit = cgit + 1) {
for(int64_t j = 0; j < (_ZL7lastrow - _ZL8firstrow) + 1;   j = j + 1) {
  sum = 0;
for(int64_t k = rowstr[j]; k < rowstr[(j + 1)];   k = k + 1) {
  sum = sum + (a[k] * p[colidx[k]]);
}
  q[j] = sum;
}
  d = 0;
for(int64_t j = 0; j < (_ZL7lastcol - _ZL8firstcol) + 1;   j = j + 1) {
  d = d + (p[j] * q[j]);
}
  alpha = rho / d;
for(int64_t j = 0; j < (_ZL7lastcol - _ZL8firstcol) + 1;   j = j + 1) {
  z[j] = z[j] + (alpha * p[j]);
  r[j] = r[j] - (alpha * q[j]);
}
  rho0 = rho;
  rho = 0;
for(int64_t j = 0; j < (_ZL7lastcol - _ZL8firstcol) + 1;   j = j + 1) {
  rho = rho + (r[j] * r[j]);
}
  beta = rho / rho0;
for(int64_t j = 0; j < (_ZL7lastcol - _ZL8firstcol) + 1;   j = j + 1) {
  p[j] = r[j] + (beta * p[j]);
}
}
// INSERT COMMENT LOOP: conj_grad::for.cond127
for(int64_t j = 0; j < (_ZL7lastrow - _ZL8firstrow) + 1;   j = j + 1) {
  d = 0;
for(int64_t k = rowstr[j]; k < rowstr[(j + 1)];   k = k + 1) {
  d = d + (a[k] * z[colidx[k]]);
}
  r[j] = d;
}
  sum = 0;
// INSERT COMMENT LOOP: conj_grad::for.cond156
for(int64_t j = 0; j < (_ZL7lastcol - _ZL8firstcol) + 1;   j = j + 1) {
  d = (x[j] - r[j]);
  sum = sum + (d * d);
}
  __FIXME__call = sqrt(sum);
  *rnorm = __FIXME__call;
}
// FUNCTION ORDER ID 15 END


// FUNCTION ORDER ID 16 START
// INSERT COMMENT FUNCTION: setup_gpu
void setup_gpu(void) {
  double __FIXME__2;
  double __FIXME__3;
  double __FIXME__4;
  double __FIXME__5;
  double __FIXME__6;
  double __FIXME__7;
  double __FIXME__8;
  double __FIXME__9;
  double __FIXME__10;
  double __FIXME__11;
  uint8_t* __FIXME__call;
  uint8_t* __FIXME__call69;

  /* Initialize gpu_device_properties with CPU-compatible defaults */
  gpu_device_properties.__FIXME__l_struct_struct_OC_cudaDeviceProp_field4 = 32;  /* warpSize */
  gpu_device_properties.__FIXME__l_struct_struct_OC_cudaDeviceProp_field6 = 1024;  /* maxThreadsPerBlock */

// INSERT COMMENT IFELSE: setup_gpu::entry
  if (1024 <= *(&gpu_device_properties.__FIXME__l_struct_struct_OC_cudaDeviceProp_field6)) { // IFELSE MARKER: entry IF
  threads_per_block_on_kernel_one = 1024;
  } else { // IFELSE MARKER: entry ELSE
  threads_per_block_on_kernel_one = *(&gpu_device_properties.__FIXME__l_struct_struct_OC_cudaDeviceProp_field4);
  }
// INSERT COMMENT IFELSE: setup_gpu::if.end
  if (256 <= *(&gpu_device_properties.__FIXME__l_struct_struct_OC_cudaDeviceProp_field6)) { // IFELSE MARKER: if.end IF
  threads_per_block_on_kernel_two = 256;
  } else { // IFELSE MARKER: if.end ELSE
  threads_per_block_on_kernel_two = *(&gpu_device_properties.__FIXME__l_struct_struct_OC_cudaDeviceProp_field4);
  }
// INSERT COMMENT IFELSE: setup_gpu::if.end4
  if (64 <= *(&gpu_device_properties.__FIXME__l_struct_struct_OC_cudaDeviceProp_field6)) { // IFELSE MARKER: if.end4 IF
  threads_per_block_on_kernel_three = 64;
  } else { // IFELSE MARKER: if.end4 ELSE
  threads_per_block_on_kernel_three = *(&gpu_device_properties.__FIXME__l_struct_struct_OC_cudaDeviceProp_field4);
  }
// INSERT COMMENT IFELSE: setup_gpu::if.end8
  if (256 <= *(&gpu_device_properties.__FIXME__l_struct_struct_OC_cudaDeviceProp_field6)) { // IFELSE MARKER: if.end8 IF
  threads_per_block_on_kernel_four = 256;
  } else { // IFELSE MARKER: if.end8 ELSE
  threads_per_block_on_kernel_four = *(&gpu_device_properties.__FIXME__l_struct_struct_OC_cudaDeviceProp_field4);
  }
// INSERT COMMENT IFELSE: setup_gpu::if.end12
  if (64 <= *(&gpu_device_properties.__FIXME__l_struct_struct_OC_cudaDeviceProp_field6)) { // IFELSE MARKER: if.end12 IF
  threads_per_block_on_kernel_five = 64;
  } else { // IFELSE MARKER: if.end12 ELSE
  threads_per_block_on_kernel_five = *(&gpu_device_properties.__FIXME__l_struct_struct_OC_cudaDeviceProp_field4);
  }
// INSERT COMMENT IFELSE: setup_gpu::if.end16
  if (256 <= *(&gpu_device_properties.__FIXME__l_struct_struct_OC_cudaDeviceProp_field6)) { // IFELSE MARKER: if.end16 IF
  threads_per_block_on_kernel_six = 256;
  } else { // IFELSE MARKER: if.end16 ELSE
  threads_per_block_on_kernel_six = *(&gpu_device_properties.__FIXME__l_struct_struct_OC_cudaDeviceProp_field4);
  }
// INSERT COMMENT IFELSE: setup_gpu::if.end20
  if (512 <= *(&gpu_device_properties.__FIXME__l_struct_struct_OC_cudaDeviceProp_field6)) { // IFELSE MARKER: if.end20 IF
  threads_per_block_on_kernel_seven = 512;
  } else { // IFELSE MARKER: if.end20 ELSE
  threads_per_block_on_kernel_seven = *(&gpu_device_properties.__FIXME__l_struct_struct_OC_cudaDeviceProp_field4);
  }
// INSERT COMMENT IFELSE: setup_gpu::if.end24
  if (64 <= *(&gpu_device_properties.__FIXME__l_struct_struct_OC_cudaDeviceProp_field6)) { // IFELSE MARKER: if.end24 IF
  threads_per_block_on_kernel_eight = 64;
  } else { // IFELSE MARKER: if.end24 ELSE
  threads_per_block_on_kernel_eight = *(&gpu_device_properties.__FIXME__l_struct_struct_OC_cudaDeviceProp_field4);
  }
// INSERT COMMENT IFELSE: setup_gpu::if.end28
  if (512 <= *(&gpu_device_properties.__FIXME__l_struct_struct_OC_cudaDeviceProp_field6)) { // IFELSE MARKER: if.end28 IF
  threads_per_block_on_kernel_nine = 512;
  } else { // IFELSE MARKER: if.end28 ELSE
  threads_per_block_on_kernel_nine = *(&gpu_device_properties.__FIXME__l_struct_struct_OC_cudaDeviceProp_field4);
  }
// INSERT COMMENT IFELSE: setup_gpu::if.end32
  if (256 <= *(&gpu_device_properties.__FIXME__l_struct_struct_OC_cudaDeviceProp_field6)) { // IFELSE MARKER: if.end32 IF
  threads_per_block_on_kernel_ten = 256;
  } else { // IFELSE MARKER: if.end32 ELSE
  threads_per_block_on_kernel_ten = *(&gpu_device_properties.__FIXME__l_struct_struct_OC_cudaDeviceProp_field4);
  }
// INSERT COMMENT IFELSE: setup_gpu::if.end36
  if (512 <= *(&gpu_device_properties.__FIXME__l_struct_struct_OC_cudaDeviceProp_field6)) { // IFELSE MARKER: if.end36 IF
  threads_per_block_on_kernel_eleven = 512;
  } else { // IFELSE MARKER: if.end36 ELSE
  threads_per_block_on_kernel_eleven = *(&gpu_device_properties.__FIXME__l_struct_struct_OC_cudaDeviceProp_field4);
  }
  __FIXME__2 = /*__FIXME__INTRINSIC_CALL__*/llvm_OC_ceil_OC_f64((14000 / (double)(threads_per_block_on_kernel_one)));
  blocks_per_grid_on_kernel_one = ((int32_t)__FIXME__2);
  __FIXME__3 = /*__FIXME__INTRINSIC_CALL__*/llvm_OC_ceil_OC_f64((14000 / (double)(threads_per_block_on_kernel_two)));
  blocks_per_grid_on_kernel_two = ((int32_t)__FIXME__3);
  blocks_per_grid_on_kernel_three = 14000;
  __FIXME__4 = /*__FIXME__INTRINSIC_CALL__*/llvm_OC_ceil_OC_f64((14000 / (double)(threads_per_block_on_kernel_four)));
  blocks_per_grid_on_kernel_four = ((int32_t)__FIXME__4);
  __FIXME__5 = /*__FIXME__INTRINSIC_CALL__*/llvm_OC_ceil_OC_f64((14000 / (double)(threads_per_block_on_kernel_five)));
  blocks_per_grid_on_kernel_five = ((int32_t)__FIXME__5);
  __FIXME__6 = /*__FIXME__INTRINSIC_CALL__*/llvm_OC_ceil_OC_f64((14000 / (double)(threads_per_block_on_kernel_six)));
  blocks_per_grid_on_kernel_six = ((int32_t)__FIXME__6);
  __FIXME__7 = /*__FIXME__INTRINSIC_CALL__*/llvm_OC_ceil_OC_f64((14000 / (double)(threads_per_block_on_kernel_seven)));
  blocks_per_grid_on_kernel_seven = ((int32_t)__FIXME__7);
  blocks_per_grid_on_kernel_eight = 14000;
  __FIXME__8 = /*__FIXME__INTRINSIC_CALL__*/llvm_OC_ceil_OC_f64((14000 / (double)(threads_per_block_on_kernel_nine)));
  blocks_per_grid_on_kernel_nine = ((int32_t)__FIXME__8);
  __FIXME__9 = /*__FIXME__INTRINSIC_CALL__*/llvm_OC_ceil_OC_f64((14000 / (double)(threads_per_block_on_kernel_ten)));
  blocks_per_grid_on_kernel_ten = ((int32_t)__FIXME__9);
  __FIXME__10 = /*__FIXME__INTRINSIC_CALL__*/llvm_OC_ceil_OC_f64((14000 / (double)(threads_per_block_on_kernel_eleven)));
  blocks_per_grid_on_kernel_eleven = ((int32_t)__FIXME__10);
  __FIXME__11 = /*__FIXME__INTRINSIC_CALL__*/llvm_OC_ceil_OC_f64((14000 / (double)(*(&gpu_device_properties.__FIXME__l_struct_struct_OC_cudaDeviceProp_field4))));
  global_data_elements = ((uint64_t)__FIXME__11);
  size_global_data = global_data_elements * 8;
  size_colidx_device = 8064000;
  size_rowstr_device = 56004;
  size_iv_device = 56000;
  size_arow_device = 56000;
  size_acol_device = 672000;
  size_aelt_device = 1344000;
  size_a_device = 16128000;
  size_x_device = 112016;
  size_z_device = 112016;
  size_p_device = 112016;
  size_q_device = 112016;
  size_r_device = 112016;
  size_rho_device = 8;
  size_d_device = 8;
  size_alpha_device = 8;
  size_beta_device = 8;
  size_sum_device = 8;
  size_norm_temp1_device = 8;
  size_norm_temp2_device = 8;
  __FIXME__call = malloc(size_global_data);
  global_data = ((double*)__FIXME__call);
  __FIXME__call69 = malloc(size_global_data);
  global_data_two = ((double*)__FIXME__call69);
  size_shared_data_on_kernel_one = threads_per_block_on_kernel_one * 8;
  size_shared_data_on_kernel_two = threads_per_block_on_kernel_two * 8;
  size_shared_data_on_kernel_three = threads_per_block_on_kernel_three * 8;
  size_shared_data_on_kernel_four = threads_per_block_on_kernel_four * 8;
  size_shared_data_on_kernel_five = threads_per_block_on_kernel_five * 8;
  size_shared_data_on_kernel_six = threads_per_block_on_kernel_six * 8;
  size_shared_data_on_kernel_seven = threads_per_block_on_kernel_seven * 8;
  size_shared_data_on_kernel_eight = threads_per_block_on_kernel_eight * 8;
  size_shared_data_on_kernel_nine = threads_per_block_on_kernel_nine * 8;
  size_shared_data_on_kernel_ten = threads_per_block_on_kernel_ten * 8;
  size_shared_data_on_kernel_eleven = threads_per_block_on_kernel_eleven * 8;
  size_reduce_memory_on_kernel_one = blocks_per_grid_on_kernel_one * 8;
  size_reduce_memory_on_kernel_two = blocks_per_grid_on_kernel_two * 8;
  size_reduce_memory_on_kernel_three = blocks_per_grid_on_kernel_three * 8;
  size_reduce_memory_on_kernel_four = blocks_per_grid_on_kernel_four * 8;
  size_reduce_memory_on_kernel_five = blocks_per_grid_on_kernel_five * 8;
  size_reduce_memory_on_kernel_six = blocks_per_grid_on_kernel_six * 8;
  size_reduce_memory_on_kernel_seven = blocks_per_grid_on_kernel_seven * 8;
  size_reduce_memory_on_kernel_eight = blocks_per_grid_on_kernel_eight * 8;
  size_reduce_memory_on_kernel_nine = blocks_per_grid_on_kernel_nine * 8;
  size_reduce_memory_on_kernel_ten = blocks_per_grid_on_kernel_ten * 8;
  size_reduce_memory_on_kernel_eleven = blocks_per_grid_on_kernel_eleven * 8;
}
// FUNCTION ORDER ID 16 END


// FUNCTION ORDER ID 17 START
// INSERT COMMENT FUNCTION: conj_grad_gpu
void conj_grad_gpu(double* rnorm) {
  double d;    /* Address-exposed local */
  double sum;    /* Address-exposed local */
  double rho;    /* Address-exposed local */
  double rho0;   /* Previous rho for beta calculation */
  uint32_t cgit;
  double __FIXME__call;

gpu_kernel_one_host();
gpu_kernel_two_host((&rho));
// INSERT COMMENT LOOP: conj_grad_gpu::for.cond
for(int32_t cgit = 1; cgit < 26;   cgit = cgit + 1) {
gpu_kernel_three_host();
gpu_kernel_four_host((&d));
gpu_kernel_five_host((rho / d));
rho0 = rho;
gpu_kernel_six_host((&rho));
gpu_kernel_seven_host((rho / rho0));
}
gpu_kernel_eight_host();
gpu_kernel_nine_host((&sum));
  __FIXME__call = sqrt(sum);
  *rnorm = __FIXME__call;
}
// FUNCTION ORDER ID 17 END


// FUNCTION ORDER ID 18 START
// INSERT COMMENT FUNCTION: gpu_kernel_ten_host
void gpu_kernel_ten_host(double* norm_temp1, double* norm_temp2) {
  struct __FIXME__l_struct_struct_OC_dim3 __FIXME__agg_2e_tmp;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dim3 __FIXME__agg_2e_tmp1;    /* Address-exposed local */
  struct __FIXME__l_unnamed_1 __FIXME__agg_2e_tmp_2e_coerce;    /* Address-exposed local */
  struct __FIXME__l_unnamed_1 __FIXME__agg_2e_tmp1_2e_coerce;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dim3 __FIXME__agg_2e_tmp2;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dim3 __FIXME__agg_2e_tmp3;    /* Address-exposed local */
  struct __FIXME__l_unnamed_1 __FIXME__agg_2e_tmp2_2e_coerce;    /* Address-exposed local */
  struct __FIXME__l_unnamed_1 __FIXME__agg_2e_tmp3_2e_coerce;    /* Address-exposed local */
  uint32_t i;
  uint32_t j;

  __FIXME__agg_2e_tmp.__FIXME__l_struct_struct_OC_dim3_field0 = blocks_per_grid_on_kernel_ten;
  __FIXME__agg_2e_tmp.__FIXME__l_struct_struct_OC_dim3_field1 = 1;
  __FIXME__agg_2e_tmp.__FIXME__l_struct_struct_OC_dim3_field2 = 1;
  __FIXME__agg_2e_tmp1.__FIXME__l_struct_struct_OC_dim3_field0 = threads_per_block_on_kernel_ten;
  __FIXME__agg_2e_tmp1.__FIXME__l_struct_struct_OC_dim3_field1 = 1;
  __FIXME__agg_2e_tmp1.__FIXME__l_struct_struct_OC_dim3_field2 = 1;
  memcpy(((uint8_t*)(&__FIXME__agg_2e_tmp_2e_coerce)), ((uint8_t*)(&__FIXME__agg_2e_tmp)), 12);
  memcpy(((uint8_t*)(&__FIXME__agg_2e_tmp1_2e_coerce)), ((uint8_t*)(&__FIXME__agg_2e_tmp1)), 12);
// INSERT COMMENT LOOP: gpu_kernel_ten_host::header.0
#pragma omp parallel for collapse(2)
for(int32_t i = 0; i < blocks_per_grid_on_kernel_ten;   i = i + 1) {
for(int32_t j = 0; j < threads_per_block_on_kernel_ten;   j = j + 1) {
gpu_kernel_ten_10(global_data, _ZL1x, _ZL1z, blocks_per_grid_on_kernel_ten, 1, 1, threads_per_block_on_kernel_ten, 1, 1, i, 0, 0, j, 0, 0);
}
for(int32_t j = 0; j < threads_per_block_on_kernel_ten;   j = j + 1) {
gpu_kernel_ten_11(global_data, _ZL1x, _ZL1z, blocks_per_grid_on_kernel_ten, 1, 1, threads_per_block_on_kernel_ten, 1, 1, i, 0, 0, j, 0, 0);
}
}
  __FIXME__agg_2e_tmp2.__FIXME__l_struct_struct_OC_dim3_field0 = blocks_per_grid_on_kernel_ten;
  __FIXME__agg_2e_tmp2.__FIXME__l_struct_struct_OC_dim3_field1 = 1;
  __FIXME__agg_2e_tmp2.__FIXME__l_struct_struct_OC_dim3_field2 = 1;
  __FIXME__agg_2e_tmp3.__FIXME__l_struct_struct_OC_dim3_field0 = threads_per_block_on_kernel_ten;
  __FIXME__agg_2e_tmp3.__FIXME__l_struct_struct_OC_dim3_field1 = 1;
  __FIXME__agg_2e_tmp3.__FIXME__l_struct_struct_OC_dim3_field2 = 1;
  memcpy(((uint8_t*)(&__FIXME__agg_2e_tmp2_2e_coerce)), ((uint8_t*)(&__FIXME__agg_2e_tmp2)), 12);
  memcpy(((uint8_t*)(&__FIXME__agg_2e_tmp3_2e_coerce)), ((uint8_t*)(&__FIXME__agg_2e_tmp3)), 12);
// INSERT COMMENT LOOP: gpu_kernel_ten_host::header.010
#pragma omp parallel for collapse(2)
for(int32_t i = 0; i < blocks_per_grid_on_kernel_ten;   i = i + 1) {
for(int32_t j = 0; j < threads_per_block_on_kernel_ten;   j = j + 1) {
gpu_kernel_ten_20(global_data_two, _ZL1x, _ZL1z, blocks_per_grid_on_kernel_ten, 1, 1, threads_per_block_on_kernel_ten, 1, 1, i, 0, 0, j, 0, 0);
}
for(int32_t j = 0; j < threads_per_block_on_kernel_ten;   j = j + 1) {
gpu_kernel_ten_21(global_data_two, _ZL1x, _ZL1z, blocks_per_grid_on_kernel_ten, 1, 1, threads_per_block_on_kernel_ten, 1, 1, i, 0, 0, j, 0, 0);
}
}
  global_data_reduce = 0;
  global_data_two_reduce = 0;
// INSERT COMMENT LOOP: gpu_kernel_ten_host::for.cond
for(int64_t i = 0; i < blocks_per_grid_on_kernel_ten;   i = i + 1) {
  global_data_reduce = (global_data_reduce + global_data[i]);
  global_data_two_reduce = (global_data_two_reduce + global_data_two[i]);
}
  *norm_temp1 = global_data_reduce;
  *norm_temp2 = global_data_two_reduce;
}
// FUNCTION ORDER ID 18 END


// FUNCTION ORDER ID 19 START
// INSERT COMMENT FUNCTION: gpu_kernel_eleven_host
void gpu_kernel_eleven_host(double norm_temp2) {
  struct __FIXME__l_struct_struct_OC_dim3 __FIXME__agg_2e_tmp;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dim3 __FIXME__agg_2e_tmp1;    /* Address-exposed local */
  struct __FIXME__l_unnamed_1 __FIXME__agg_2e_tmp_2e_coerce;    /* Address-exposed local */
  struct __FIXME__l_unnamed_1 __FIXME__agg_2e_tmp1_2e_coerce;    /* Address-exposed local */
  uint32_t i;
  uint32_t j;

  __FIXME__agg_2e_tmp.__FIXME__l_struct_struct_OC_dim3_field0 = blocks_per_grid_on_kernel_eleven;
  __FIXME__agg_2e_tmp.__FIXME__l_struct_struct_OC_dim3_field1 = 1;
  __FIXME__agg_2e_tmp.__FIXME__l_struct_struct_OC_dim3_field2 = 1;
  __FIXME__agg_2e_tmp1.__FIXME__l_struct_struct_OC_dim3_field0 = threads_per_block_on_kernel_eleven;
  __FIXME__agg_2e_tmp1.__FIXME__l_struct_struct_OC_dim3_field1 = 1;
  __FIXME__agg_2e_tmp1.__FIXME__l_struct_struct_OC_dim3_field2 = 1;
  memcpy(((uint8_t*)(&__FIXME__agg_2e_tmp_2e_coerce)), ((uint8_t*)(&__FIXME__agg_2e_tmp)), 12);
  memcpy(((uint8_t*)(&__FIXME__agg_2e_tmp1_2e_coerce)), ((uint8_t*)(&__FIXME__agg_2e_tmp1)), 12);
// INSERT COMMENT LOOP: gpu_kernel_eleven_host::header.0
#pragma omp parallel for collapse(2)
for(int32_t i = 0; i < blocks_per_grid_on_kernel_eleven;   i = i + 1) {
for(int32_t j = 0; j < threads_per_block_on_kernel_eleven;   j = j + 1) {
gpu_kernel_eleven_device(norm_temp2, _ZL1x, _ZL1z, blocks_per_grid_on_kernel_eleven, 1, 1, threads_per_block_on_kernel_eleven, 1, 1, i, 0, 0, j, 0, 0);
}
}
  return;
}
// FUNCTION ORDER ID 19 END


// FUNCTION ORDER ID 20 START
// INSERT COMMENT FUNCTION: release_gpu
void release_gpu(void) {
  return;
}
// FUNCTION ORDER ID 20 END


// FUNCTION ORDER ID 21 START
// INSERT COMMENT FUNCTION: gpu_kernel_one_host
void gpu_kernel_one_host(void) {
  struct __FIXME__l_struct_struct_OC_dim3 __FIXME__agg_2e_tmp;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dim3 __FIXME__agg_2e_tmp1;    /* Address-exposed local */
  struct __FIXME__l_unnamed_1 __FIXME__agg_2e_tmp_2e_coerce;    /* Address-exposed local */
  struct __FIXME__l_unnamed_1 __FIXME__agg_2e_tmp1_2e_coerce;    /* Address-exposed local */
  uint32_t i;
  uint32_t j;

  __FIXME__agg_2e_tmp.__FIXME__l_struct_struct_OC_dim3_field0 = blocks_per_grid_on_kernel_one;
  __FIXME__agg_2e_tmp.__FIXME__l_struct_struct_OC_dim3_field1 = 1;
  __FIXME__agg_2e_tmp.__FIXME__l_struct_struct_OC_dim3_field2 = 1;
  __FIXME__agg_2e_tmp1.__FIXME__l_struct_struct_OC_dim3_field0 = threads_per_block_on_kernel_one;
  __FIXME__agg_2e_tmp1.__FIXME__l_struct_struct_OC_dim3_field1 = 1;
  __FIXME__agg_2e_tmp1.__FIXME__l_struct_struct_OC_dim3_field2 = 1;
  memcpy(((uint8_t*)(&__FIXME__agg_2e_tmp_2e_coerce)), ((uint8_t*)(&__FIXME__agg_2e_tmp)), 12);
  memcpy(((uint8_t*)(&__FIXME__agg_2e_tmp1_2e_coerce)), ((uint8_t*)(&__FIXME__agg_2e_tmp1)), 12);
// INSERT COMMENT LOOP: gpu_kernel_one_host::header.0
#pragma omp parallel for collapse(2)
for(int32_t i = 0; i < blocks_per_grid_on_kernel_one;   i = i + 1) {
for(int32_t j = 0; j < threads_per_block_on_kernel_one;   j = j + 1) {
gpu_kernel_one_device(_ZL1p, _ZL1q, _ZL1r, _ZL1x, _ZL1z, blocks_per_grid_on_kernel_one, 1, 1, threads_per_block_on_kernel_one, 1, 1, i, 0, 0, j, 0, 0);
}
}
  return;
}
// FUNCTION ORDER ID 21 END


// FUNCTION ORDER ID 22 START
// INSERT COMMENT FUNCTION: gpu_kernel_two_host
void gpu_kernel_two_host(double* rho_host) {
  struct __FIXME__l_struct_struct_OC_dim3 __FIXME__agg_2e_tmp;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dim3 __FIXME__agg_2e_tmp1;    /* Address-exposed local */
  struct __FIXME__l_unnamed_1 __FIXME__agg_2e_tmp_2e_coerce;    /* Address-exposed local */
  struct __FIXME__l_unnamed_1 __FIXME__agg_2e_tmp1_2e_coerce;    /* Address-exposed local */
  uint32_t i;
  uint32_t j;

  __FIXME__agg_2e_tmp.__FIXME__l_struct_struct_OC_dim3_field0 = blocks_per_grid_on_kernel_two;
  __FIXME__agg_2e_tmp.__FIXME__l_struct_struct_OC_dim3_field1 = 1;
  __FIXME__agg_2e_tmp.__FIXME__l_struct_struct_OC_dim3_field2 = 1;
  __FIXME__agg_2e_tmp1.__FIXME__l_struct_struct_OC_dim3_field0 = threads_per_block_on_kernel_two;
  __FIXME__agg_2e_tmp1.__FIXME__l_struct_struct_OC_dim3_field1 = 1;
  __FIXME__agg_2e_tmp1.__FIXME__l_struct_struct_OC_dim3_field2 = 1;
  memcpy(((uint8_t*)(&__FIXME__agg_2e_tmp_2e_coerce)), ((uint8_t*)(&__FIXME__agg_2e_tmp)), 12);
  memcpy(((uint8_t*)(&__FIXME__agg_2e_tmp1_2e_coerce)), ((uint8_t*)(&__FIXME__agg_2e_tmp1)), 12);
// INSERT COMMENT LOOP: gpu_kernel_two_host::header.0
#pragma omp parallel for collapse(2)
for(int32_t i = 0; i < blocks_per_grid_on_kernel_two;   i = i + 1) {
for(int32_t j = 0; j < threads_per_block_on_kernel_two;   j = j + 1) {
gpu_kernel_two_device0(_ZL1r, rho_device, global_data, blocks_per_grid_on_kernel_two, 1, 1, threads_per_block_on_kernel_two, 1, 1, i, 0, 0, j, 0, 0);
}
for(int32_t j = 0; j < threads_per_block_on_kernel_two;   j = j + 1) {
gpu_kernel_two_device1(_ZL1r, rho_device, global_data, blocks_per_grid_on_kernel_two, 1, 1, threads_per_block_on_kernel_two, 1, 1, i, 0, 0, j, 0, 0);
}
}
  global_data_reduce = 0;
// INSERT COMMENT LOOP: gpu_kernel_two_host::for.cond
for(int64_t i = 0; i < blocks_per_grid_on_kernel_two;   i = i + 1) {
  global_data_reduce = (global_data_reduce + global_data[i]);
}
  *rho_host = global_data_reduce;
}
// FUNCTION ORDER ID 22 END


// FUNCTION ORDER ID 23 START
// INSERT COMMENT FUNCTION: gpu_kernel_three_host
void gpu_kernel_three_host(void) {
  struct __FIXME__l_struct_struct_OC_dim3 __FIXME__agg_2e_tmp;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dim3 __FIXME__agg_2e_tmp1;    /* Address-exposed local */
  struct __FIXME__l_unnamed_1 __FIXME__agg_2e_tmp_2e_coerce;    /* Address-exposed local */
  struct __FIXME__l_unnamed_1 __FIXME__agg_2e_tmp1_2e_coerce;    /* Address-exposed local */
  uint32_t i;
  uint32_t j;

  __FIXME__agg_2e_tmp.__FIXME__l_struct_struct_OC_dim3_field0 = blocks_per_grid_on_kernel_three;
  __FIXME__agg_2e_tmp.__FIXME__l_struct_struct_OC_dim3_field1 = 1;
  __FIXME__agg_2e_tmp.__FIXME__l_struct_struct_OC_dim3_field2 = 1;
  __FIXME__agg_2e_tmp1.__FIXME__l_struct_struct_OC_dim3_field0 = threads_per_block_on_kernel_three;
  __FIXME__agg_2e_tmp1.__FIXME__l_struct_struct_OC_dim3_field1 = 1;
  __FIXME__agg_2e_tmp1.__FIXME__l_struct_struct_OC_dim3_field2 = 1;
  memcpy(((uint8_t*)(&__FIXME__agg_2e_tmp_2e_coerce)), ((uint8_t*)(&__FIXME__agg_2e_tmp)), 12);
  memcpy(((uint8_t*)(&__FIXME__agg_2e_tmp1_2e_coerce)), ((uint8_t*)(&__FIXME__agg_2e_tmp1)), 12);
// INSERT COMMENT LOOP: gpu_kernel_three_host::header.0
#pragma omp parallel for collapse(2)
for(int32_t i = 0; i < blocks_per_grid_on_kernel_three;   i = i + 1) {
for(int32_t j = 0; j < threads_per_block_on_kernel_three;   j = j + 1) {
gpu_kernel_three_device0(_ZL6colidx, _ZL6rowstr, _ZL1a, _ZL1p, _ZL1q, blocks_per_grid_on_kernel_three, 1, 1, threads_per_block_on_kernel_three, 1, 1, i, 0, 0, j, 0, 0);
}
for(int32_t j = 0; j < threads_per_block_on_kernel_three;   j = j + 1) {
gpu_kernel_three_device1(_ZL6colidx, _ZL6rowstr, _ZL1a, _ZL1p, _ZL1q, blocks_per_grid_on_kernel_three, 1, 1, threads_per_block_on_kernel_three, 1, 1, i, 0, 0, j, 0, 0);
}
}
  return;
}
// FUNCTION ORDER ID 23 END


// FUNCTION ORDER ID 24 START
// INSERT COMMENT FUNCTION: gpu_kernel_four_host
void gpu_kernel_four_host(double* d_host) {
  struct __FIXME__l_struct_struct_OC_dim3 __FIXME__agg_2e_tmp;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dim3 __FIXME__agg_2e_tmp1;    /* Address-exposed local */
  struct __FIXME__l_unnamed_1 __FIXME__agg_2e_tmp_2e_coerce;    /* Address-exposed local */
  struct __FIXME__l_unnamed_1 __FIXME__agg_2e_tmp1_2e_coerce;    /* Address-exposed local */
  uint32_t i;
  uint32_t j;

  __FIXME__agg_2e_tmp.__FIXME__l_struct_struct_OC_dim3_field0 = blocks_per_grid_on_kernel_four;
  __FIXME__agg_2e_tmp.__FIXME__l_struct_struct_OC_dim3_field1 = 1;
  __FIXME__agg_2e_tmp.__FIXME__l_struct_struct_OC_dim3_field2 = 1;
  __FIXME__agg_2e_tmp1.__FIXME__l_struct_struct_OC_dim3_field0 = threads_per_block_on_kernel_four;
  __FIXME__agg_2e_tmp1.__FIXME__l_struct_struct_OC_dim3_field1 = 1;
  __FIXME__agg_2e_tmp1.__FIXME__l_struct_struct_OC_dim3_field2 = 1;
  memcpy(((uint8_t*)(&__FIXME__agg_2e_tmp_2e_coerce)), ((uint8_t*)(&__FIXME__agg_2e_tmp)), 12);
  memcpy(((uint8_t*)(&__FIXME__agg_2e_tmp1_2e_coerce)), ((uint8_t*)(&__FIXME__agg_2e_tmp1)), 12);
// INSERT COMMENT LOOP: gpu_kernel_four_host::header.0
#pragma omp parallel for collapse(2)
for(int32_t i = 0; i < blocks_per_grid_on_kernel_four;   i = i + 1) {
for(int32_t j = 0; j < threads_per_block_on_kernel_four;   j = j + 1) {
gpu_kernel_four_device0(d_device, _ZL1p, _ZL1q, global_data, blocks_per_grid_on_kernel_four, 1, 1, threads_per_block_on_kernel_four, 1, 1, i, 0, 0, j, 0, 0);
}
for(int32_t j = 0; j < threads_per_block_on_kernel_four;   j = j + 1) {
gpu_kernel_four_device1(d_device, _ZL1p, _ZL1q, global_data, blocks_per_grid_on_kernel_four, 1, 1, threads_per_block_on_kernel_four, 1, 1, i, 0, 0, j, 0, 0);
}
}
  global_data_reduce = 0;
// INSERT COMMENT LOOP: gpu_kernel_four_host::for.cond
for(int64_t i = 0; i < blocks_per_grid_on_kernel_four;   i = i + 1) {
  global_data_reduce = (global_data_reduce + global_data[i]);
}
  *d_host = global_data_reduce;
}
// FUNCTION ORDER ID 24 END


// FUNCTION ORDER ID 25 START
// INSERT COMMENT FUNCTION: gpu_kernel_five_host
void gpu_kernel_five_host(double alpha_host) {
  struct __FIXME__l_struct_struct_OC_dim3 __FIXME__agg_2e_tmp;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dim3 __FIXME__agg_2e_tmp1;    /* Address-exposed local */
  struct __FIXME__l_unnamed_1 __FIXME__agg_2e_tmp_2e_coerce;    /* Address-exposed local */
  struct __FIXME__l_unnamed_1 __FIXME__agg_2e_tmp1_2e_coerce;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dim3 __FIXME__agg_2e_tmp2;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dim3 __FIXME__agg_2e_tmp3;    /* Address-exposed local */
  struct __FIXME__l_unnamed_1 __FIXME__agg_2e_tmp2_2e_coerce;    /* Address-exposed local */
  struct __FIXME__l_unnamed_1 __FIXME__agg_2e_tmp3_2e_coerce;    /* Address-exposed local */
  uint32_t i;
  uint32_t j;

  __FIXME__agg_2e_tmp.__FIXME__l_struct_struct_OC_dim3_field0 = blocks_per_grid_on_kernel_five;
  __FIXME__agg_2e_tmp.__FIXME__l_struct_struct_OC_dim3_field1 = 1;
  __FIXME__agg_2e_tmp.__FIXME__l_struct_struct_OC_dim3_field2 = 1;
  __FIXME__agg_2e_tmp1.__FIXME__l_struct_struct_OC_dim3_field0 = threads_per_block_on_kernel_five;
  __FIXME__agg_2e_tmp1.__FIXME__l_struct_struct_OC_dim3_field1 = 1;
  __FIXME__agg_2e_tmp1.__FIXME__l_struct_struct_OC_dim3_field2 = 1;
  memcpy(((uint8_t*)(&__FIXME__agg_2e_tmp_2e_coerce)), ((uint8_t*)(&__FIXME__agg_2e_tmp)), 12);
  memcpy(((uint8_t*)(&__FIXME__agg_2e_tmp1_2e_coerce)), ((uint8_t*)(&__FIXME__agg_2e_tmp1)), 12);
// INSERT COMMENT LOOP: gpu_kernel_five_host::header.0
#pragma omp parallel for collapse(2)
for(int32_t i = 0; i < blocks_per_grid_on_kernel_five;   i = i + 1) {
for(int32_t j = 0; j < threads_per_block_on_kernel_five;   j = j + 1) {
gpu_kernel_five_1(alpha_host, _ZL1p, _ZL1z, blocks_per_grid_on_kernel_five, 1, 1, threads_per_block_on_kernel_five, 1, 1, i, 0, 0, j, 0, 0);
}
}
  __FIXME__agg_2e_tmp2.__FIXME__l_struct_struct_OC_dim3_field0 = blocks_per_grid_on_kernel_five;
  __FIXME__agg_2e_tmp2.__FIXME__l_struct_struct_OC_dim3_field1 = 1;
  __FIXME__agg_2e_tmp2.__FIXME__l_struct_struct_OC_dim3_field2 = 1;
  __FIXME__agg_2e_tmp3.__FIXME__l_struct_struct_OC_dim3_field0 = threads_per_block_on_kernel_five;
  __FIXME__agg_2e_tmp3.__FIXME__l_struct_struct_OC_dim3_field1 = 1;
  __FIXME__agg_2e_tmp3.__FIXME__l_struct_struct_OC_dim3_field2 = 1;
  memcpy(((uint8_t*)(&__FIXME__agg_2e_tmp2_2e_coerce)), ((uint8_t*)(&__FIXME__agg_2e_tmp2)), 12);
  memcpy(((uint8_t*)(&__FIXME__agg_2e_tmp3_2e_coerce)), ((uint8_t*)(&__FIXME__agg_2e_tmp3)), 12);
// INSERT COMMENT LOOP: gpu_kernel_five_host::header.010
#pragma omp parallel for collapse(2)
for(int32_t i = 0; i < blocks_per_grid_on_kernel_five;   i = i + 1) {
for(int32_t j = 0; j < threads_per_block_on_kernel_five;   j = j + 1) {
gpu_kernel_five_2(alpha_host, _ZL1q, _ZL1r, blocks_per_grid_on_kernel_five, 1, 1, threads_per_block_on_kernel_five, 1, 1, i, 0, 0, j, 0, 0);
}
}
  return;
}
// FUNCTION ORDER ID 25 END


// FUNCTION ORDER ID 26 START
// INSERT COMMENT FUNCTION: gpu_kernel_six_host
void gpu_kernel_six_host(double* rho_host) {
  struct __FIXME__l_struct_struct_OC_dim3 __FIXME__agg_2e_tmp;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dim3 __FIXME__agg_2e_tmp1;    /* Address-exposed local */
  struct __FIXME__l_unnamed_1 __FIXME__agg_2e_tmp_2e_coerce;    /* Address-exposed local */
  struct __FIXME__l_unnamed_1 __FIXME__agg_2e_tmp1_2e_coerce;    /* Address-exposed local */
  uint32_t i;
  uint32_t j;

  __FIXME__agg_2e_tmp.__FIXME__l_struct_struct_OC_dim3_field0 = blocks_per_grid_on_kernel_six;
  __FIXME__agg_2e_tmp.__FIXME__l_struct_struct_OC_dim3_field1 = 1;
  __FIXME__agg_2e_tmp.__FIXME__l_struct_struct_OC_dim3_field2 = 1;
  __FIXME__agg_2e_tmp1.__FIXME__l_struct_struct_OC_dim3_field0 = threads_per_block_on_kernel_six;
  __FIXME__agg_2e_tmp1.__FIXME__l_struct_struct_OC_dim3_field1 = 1;
  __FIXME__agg_2e_tmp1.__FIXME__l_struct_struct_OC_dim3_field2 = 1;
  memcpy(((uint8_t*)(&__FIXME__agg_2e_tmp_2e_coerce)), ((uint8_t*)(&__FIXME__agg_2e_tmp)), 12);
  memcpy(((uint8_t*)(&__FIXME__agg_2e_tmp1_2e_coerce)), ((uint8_t*)(&__FIXME__agg_2e_tmp1)), 12);
// INSERT COMMENT LOOP: gpu_kernel_six_host::header.0
#pragma omp parallel for collapse(2)
for(int32_t i = 0; i < blocks_per_grid_on_kernel_six;   i = i + 1) {
for(int32_t j = 0; j < threads_per_block_on_kernel_six;   j = j + 1) {
gpu_kernel_six_device0(_ZL1r, global_data, blocks_per_grid_on_kernel_six, 1, 1, threads_per_block_on_kernel_six, 1, 1, i, 0, 0, j, 0, 0);
}
for(int32_t j = 0; j < threads_per_block_on_kernel_six;   j = j + 1) {
gpu_kernel_six_device1(_ZL1r, global_data, blocks_per_grid_on_kernel_six, 1, 1, threads_per_block_on_kernel_six, 1, 1, i, 0, 0, j, 0, 0);
}
}
  global_data_reduce = 0;
// INSERT COMMENT LOOP: gpu_kernel_six_host::for.cond
for(int64_t i = 0; i < blocks_per_grid_on_kernel_six;   i = i + 1) {
  global_data_reduce = (global_data_reduce + global_data[i]);
}
  *rho_host = global_data_reduce;
}
// FUNCTION ORDER ID 26 END


// FUNCTION ORDER ID 27 START
// INSERT COMMENT FUNCTION: gpu_kernel_seven_host
void gpu_kernel_seven_host(double beta_host) {
  struct __FIXME__l_struct_struct_OC_dim3 __FIXME__agg_2e_tmp;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dim3 __FIXME__agg_2e_tmp1;    /* Address-exposed local */
  struct __FIXME__l_unnamed_1 __FIXME__agg_2e_tmp_2e_coerce;    /* Address-exposed local */
  struct __FIXME__l_unnamed_1 __FIXME__agg_2e_tmp1_2e_coerce;    /* Address-exposed local */
  uint32_t i;
  uint32_t j;

  __FIXME__agg_2e_tmp.__FIXME__l_struct_struct_OC_dim3_field0 = blocks_per_grid_on_kernel_seven;
  __FIXME__agg_2e_tmp.__FIXME__l_struct_struct_OC_dim3_field1 = 1;
  __FIXME__agg_2e_tmp.__FIXME__l_struct_struct_OC_dim3_field2 = 1;
  __FIXME__agg_2e_tmp1.__FIXME__l_struct_struct_OC_dim3_field0 = threads_per_block_on_kernel_seven;
  __FIXME__agg_2e_tmp1.__FIXME__l_struct_struct_OC_dim3_field1 = 1;
  __FIXME__agg_2e_tmp1.__FIXME__l_struct_struct_OC_dim3_field2 = 1;
  memcpy(((uint8_t*)(&__FIXME__agg_2e_tmp_2e_coerce)), ((uint8_t*)(&__FIXME__agg_2e_tmp)), 12);
  memcpy(((uint8_t*)(&__FIXME__agg_2e_tmp1_2e_coerce)), ((uint8_t*)(&__FIXME__agg_2e_tmp1)), 12);
// INSERT COMMENT LOOP: gpu_kernel_seven_host::header.0
#pragma omp parallel for collapse(2)
for(int32_t i = 0; i < blocks_per_grid_on_kernel_seven;   i = i + 1) {
for(int32_t j = 0; j < threads_per_block_on_kernel_seven;   j = j + 1) {
gpu_kernel_seven_device(beta_host, _ZL1p, _ZL1r, blocks_per_grid_on_kernel_seven, 1, 1, threads_per_block_on_kernel_seven, 1, 1, i, 0, 0, j, 0, 0);
}
}
  return;
}
// FUNCTION ORDER ID 27 END


// FUNCTION ORDER ID 28 START
// INSERT COMMENT FUNCTION: gpu_kernel_eight_host
void gpu_kernel_eight_host(void) {
  struct __FIXME__l_struct_struct_OC_dim3 __FIXME__agg_2e_tmp;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dim3 __FIXME__agg_2e_tmp1;    /* Address-exposed local */
  struct __FIXME__l_unnamed_1 __FIXME__agg_2e_tmp_2e_coerce;    /* Address-exposed local */
  struct __FIXME__l_unnamed_1 __FIXME__agg_2e_tmp1_2e_coerce;    /* Address-exposed local */
  uint32_t i;
  uint32_t j;

  __FIXME__agg_2e_tmp.__FIXME__l_struct_struct_OC_dim3_field0 = blocks_per_grid_on_kernel_eight;
  __FIXME__agg_2e_tmp.__FIXME__l_struct_struct_OC_dim3_field1 = 1;
  __FIXME__agg_2e_tmp.__FIXME__l_struct_struct_OC_dim3_field2 = 1;
  __FIXME__agg_2e_tmp1.__FIXME__l_struct_struct_OC_dim3_field0 = threads_per_block_on_kernel_eight;
  __FIXME__agg_2e_tmp1.__FIXME__l_struct_struct_OC_dim3_field1 = 1;
  __FIXME__agg_2e_tmp1.__FIXME__l_struct_struct_OC_dim3_field2 = 1;
  memcpy(((uint8_t*)(&__FIXME__agg_2e_tmp_2e_coerce)), ((uint8_t*)(&__FIXME__agg_2e_tmp)), 12);
  memcpy(((uint8_t*)(&__FIXME__agg_2e_tmp1_2e_coerce)), ((uint8_t*)(&__FIXME__agg_2e_tmp1)), 12);
// INSERT COMMENT LOOP: gpu_kernel_eight_host::header.0
#pragma omp parallel for collapse(2)
for(int32_t i = 0; i < blocks_per_grid_on_kernel_eight;   i = i + 1) {
for(int32_t j = 0; j < threads_per_block_on_kernel_eight;   j = j + 1) {
gpu_kernel_eight_device0(_ZL6colidx, _ZL6rowstr, _ZL1a, _ZL1r, _ZL1z, blocks_per_grid_on_kernel_eight, 1, 1, threads_per_block_on_kernel_eight, 1, 1, i, 0, 0, j, 0, 0);
}
for(int32_t j = 0; j < threads_per_block_on_kernel_eight;   j = j + 1) {
gpu_kernel_eight_device1(_ZL6colidx, _ZL6rowstr, _ZL1a, _ZL1r, _ZL1z, blocks_per_grid_on_kernel_eight, 1, 1, threads_per_block_on_kernel_eight, 1, 1, i, 0, 0, j, 0, 0);
}
}
  return;
}
// FUNCTION ORDER ID 28 END


// FUNCTION ORDER ID 29 START
// INSERT COMMENT FUNCTION: gpu_kernel_nine_host
void gpu_kernel_nine_host(double* sum_host) {
  struct __FIXME__l_struct_struct_OC_dim3 __FIXME__agg_2e_tmp;    /* Address-exposed local */
  struct __FIXME__l_struct_struct_OC_dim3 __FIXME__agg_2e_tmp1;    /* Address-exposed local */
  struct __FIXME__l_unnamed_1 __FIXME__agg_2e_tmp_2e_coerce;    /* Address-exposed local */
  struct __FIXME__l_unnamed_1 __FIXME__agg_2e_tmp1_2e_coerce;    /* Address-exposed local */
  uint32_t i;
  uint32_t j;

  __FIXME__agg_2e_tmp.__FIXME__l_struct_struct_OC_dim3_field0 = blocks_per_grid_on_kernel_nine;
  __FIXME__agg_2e_tmp.__FIXME__l_struct_struct_OC_dim3_field1 = 1;
  __FIXME__agg_2e_tmp.__FIXME__l_struct_struct_OC_dim3_field2 = 1;
  __FIXME__agg_2e_tmp1.__FIXME__l_struct_struct_OC_dim3_field0 = threads_per_block_on_kernel_nine;
  __FIXME__agg_2e_tmp1.__FIXME__l_struct_struct_OC_dim3_field1 = 1;
  __FIXME__agg_2e_tmp1.__FIXME__l_struct_struct_OC_dim3_field2 = 1;
  memcpy(((uint8_t*)(&__FIXME__agg_2e_tmp_2e_coerce)), ((uint8_t*)(&__FIXME__agg_2e_tmp)), 12);
  memcpy(((uint8_t*)(&__FIXME__agg_2e_tmp1_2e_coerce)), ((uint8_t*)(&__FIXME__agg_2e_tmp1)), 12);
// INSERT COMMENT LOOP: gpu_kernel_nine_host::header.0
#pragma omp parallel for collapse(2)
for(int32_t i = 0; i < blocks_per_grid_on_kernel_nine;   i = i + 1) {
for(int32_t j = 0; j < threads_per_block_on_kernel_nine;   j = j + 1) {
gpu_kernel_nine_device0(_ZL1r, _ZL1x, sum_device, global_data, blocks_per_grid_on_kernel_nine, 1, 1, threads_per_block_on_kernel_nine, 1, 1, i, 0, 0, j, 0, 0);
}
for(int32_t j = 0; j < threads_per_block_on_kernel_nine;   j = j + 1) {
gpu_kernel_nine_device1(_ZL1r, _ZL1x, sum_device, global_data, blocks_per_grid_on_kernel_nine, 1, 1, threads_per_block_on_kernel_nine, 1, 1, i, 0, 0, j, 0, 0);
}
}
  global_data_reduce = 0;
// INSERT COMMENT LOOP: gpu_kernel_nine_host::for.cond
for(int64_t i = 0; i < blocks_per_grid_on_kernel_nine;   i = i + 1) {
  global_data_reduce = (global_data_reduce + global_data[i]);
}
  *sum_host = global_data_reduce;
}
// FUNCTION ORDER ID 29 END


// FUNCTION ORDER ID 30 START
// INSERT COMMENT FUNCTION: sprnvc
void sprnvc(uint32_t n, uint32_t nz, uint32_t nn1, double* v, uint32_t* iv) {
  uint32_t nzv = 0;
  uint32_t ii, i;
  double vecelt, vecloc;
  uint32_t was_gen;

  while(nzv < nz) {
    vecelt = randlc(&_ZL4tran, _ZL5amult);

    /* generate an integer between 1 and n in a portable manner */
    vecloc = randlc(&_ZL4tran, _ZL5amult);
    i = (uint32_t)(nn1 * vecloc) + 1;
    if(i > n) { continue; }

    /* was this integer generated already? */
    was_gen = 0;
    for(ii = 0; ii < nzv; ii++) {
      if(iv[ii] == i) {
        was_gen = 1;
        break;
      }
    }
    if(was_gen) { continue; }
    v[nzv] = vecelt;
    iv[nzv] = i;
    nzv = nzv + 1;
  }
}
// FUNCTION ORDER ID 30 END


// FUNCTION ORDER ID 31 START
// INSERT COMMENT FUNCTION: vecset
void vecset(uint32_t n, double* v, uint32_t* iv, uint32_t* nzv, uint32_t __FIXME__i, double val) {
  int64_t k;
  uint32_t set;

  set = 0;
// INSERT COMMENT LOOP: vecset::for.cond
for(int64_t k = 0; k < *nzv;   k = k + 1) {
  if (iv[k] == __FIXME__i) { // IFELSE MARKER: for.body IF
  v[k] = val;
  set = 1;
  }
  set = set;
}
// INSERT COMMENT IFELSE: vecset::for.end
  if (set == 0) { // IFELSE MARKER: for.end IF
  v[*nzv] = val;
  iv[*nzv] = __FIXME__i;
  *nzv = *nzv + 1;
  }
  return;
}
// FUNCTION ORDER ID 31 END


// FUNCTION ORDER ID 32 START
// INSERT COMMENT FUNCTION: sparse
void sparse(double* a, uint32_t* colidx, uint32_t* rowstr, uint32_t n, uint32_t nz, uint32_t nozer, uint32_t* arow, uint32_t* acol, double* aelt, uint32_t firstrow, uint32_t lastrow, uint32_t* nzloc, double rcond, double shift) {
  int64_t nrows;
  int64_t i, j, j1, j2, nza, k, kk, nzrow, jcol;
  double size, scale, ratio, va;
  uint32_t goto_40;

  nrows = (lastrow - firstrow) + 1;

  /* count the number of triples in each row */
  for(j = 0; j < nrows + 1; j++) {
    rowstr[j] = 0;
  }
  for(i = 0; i < n; i++) {
    for(nza = 0; nza < arow[i]; nza++) {
      j = (acol + 12*i)[nza] + 1;
      rowstr[j] = rowstr[j] + arow[i];
    }
  }
  rowstr[0] = 0;
  for(j = 1; j < nrows + 1; j++) {
    rowstr[j] = rowstr[j] + rowstr[j - 1];
  }
  nza = rowstr[nrows] - 1;

  if(nza > nz) {
    printf("Space for matrix elements exceeded in sparse\n");
    printf("nza, nzmax = %ld, %d\n", (long)nza, nz);
    exit(1);
  }

  /* preload data pages */
  for(j = 0; j < nrows; j++) {
    for(k = rowstr[j]; k < rowstr[j + 1]; k++) {
      a[k] = 0.0;
      colidx[k] = (uint32_t)-1;
    }
    nzloc[j] = 0;
  }

  /* generate actual values by summing duplicates */
  size = 1.0;
  ratio = pow(rcond, (1.0 / (double)(n)));
  for(i = 0; i < n; i++) {
    for(nza = 0; nza < arow[i]; nza++) {
      j = (acol + 12*i)[nza];

      scale = size * (aelt + 12*i)[nza];
      for(nzrow = 0; nzrow < arow[i]; nzrow++) {
        jcol = (acol + 12*i)[nzrow];
        va = (aelt + 12*i)[nzrow] * scale;

        /* add the identity * rcond to bound the smallest eigenvalue */
        if(jcol == j && j == (int64_t)i) {
          va = va + rcond - shift;
        }

        goto_40 = 0;
        for(k = rowstr[j]; k < rowstr[j + 1]; k++) {
          if(colidx[k] > (uint32_t)jcol) {
            /* insert colidx here orderly */
            for(kk = rowstr[j + 1] - 2; kk >= k; kk--) {
              if(colidx[kk] != (uint32_t)-1) {
                a[kk + 1] = a[kk];
                colidx[kk + 1] = colidx[kk];
              }
            }
            colidx[k] = jcol;
            a[k] = 0.0;
            goto_40 = 1;
            break;
          } else if(colidx[k] == (uint32_t)-1) {
            colidx[k] = jcol;
            goto_40 = 1;
            break;
          } else if(colidx[k] == (uint32_t)jcol) {
            /* mark the duplicated entry */
            nzloc[j] = nzloc[j] + 1;
            goto_40 = 1;
            break;
          }
        }
        if(goto_40 == 0) {
          printf("internal error in sparse: i=%ld\n", (long)i);
          exit(1);
        }
        a[k] = a[k] + va;
      }
    }
    size = size * ratio;
  }

  /* remove empty entries and generate final results */
  for(j = 1; j < nrows; j++) {
    nzloc[j] = nzloc[j] + nzloc[j - 1];
  }

  for(j = 0; j < nrows; j++) {
    if(j > 0) {
      j1 = rowstr[j] - nzloc[j - 1];
    } else {
      j1 = 0;
    }
    j2 = rowstr[j + 1] - nzloc[j];
    nza = rowstr[j];
    for(k = j1; k < j2; k++) {
      a[k] = a[nza];
      colidx[k] = colidx[nza];
      nza = nza + 1;
    }
  }
  for(j = 1; j < nrows + 1; j++) {
    rowstr[j] = rowstr[j] - nzloc[j - 1];
  }
}
// FUNCTION ORDER ID 32 END


// FUNCTION ORDER ID 33 START
// INSERT COMMENT FUNCTION: gpu_kernel_ten_10
void gpu_kernel_ten_10(double* norm_temp, double* x, double* z, uint32_t __FIXME__gridDim_2e_x, uint32_t __FIXME__gridDim_2e_y, uint32_t __FIXME__gridDim_2e_z, uint32_t __FIXME__blockDim_2e_x, uint32_t __FIXME__blockDim_2e_y, uint32_t __FIXME__blockDim_2e_z, uint32_t __FIXME__blockIdx_2e_x, uint32_t __FIXME__blockIdx_2e_y, uint32_t __FIXME__blockIdx_2e_z, uint32_t local_id, uint32_t __FIXME__threadIdx_2e_y, uint32_t __FIXME__threadIdx_2e_z) {
  int64_t thread_id;

// INSERT COMMENT IFELSE: gpu_kernel_ten_10::entry
  thread_id = __FIXME__blockIdx_2e_x * __FIXME__blockDim_2e_x + local_id;
  (&extern_share_data_shared[0])[local_id] = 0;
  if (thread_id < 14000) { // IFELSE MARKER: entry IF
  (&extern_share_data_shared[0])[local_id] = (x[thread_id] * z[thread_id]);
  }
  return;
}
// FUNCTION ORDER ID 33 END


// FUNCTION ORDER ID 34 START
// INSERT COMMENT FUNCTION: gpu_kernel_ten_20
void gpu_kernel_ten_20(double* norm_temp, double* x, double* z, uint32_t __FIXME__gridDim_2e_x, uint32_t __FIXME__gridDim_2e_y, uint32_t __FIXME__gridDim_2e_z, uint32_t __FIXME__blockDim_2e_x, uint32_t __FIXME__blockDim_2e_y, uint32_t __FIXME__blockDim_2e_z, uint32_t __FIXME__blockIdx_2e_x, uint32_t __FIXME__blockIdx_2e_y, uint32_t __FIXME__blockIdx_2e_z, uint32_t local_id, uint32_t __FIXME__threadIdx_2e_y, uint32_t __FIXME__threadIdx_2e_z) {
  int64_t thread_id;

// INSERT COMMENT IFELSE: gpu_kernel_ten_20::entry
  thread_id = __FIXME__blockIdx_2e_x * __FIXME__blockDim_2e_x + local_id;
  (&extern_share_data_shared[0])[local_id] = 0;
  if (thread_id < 14000) { // IFELSE MARKER: entry IF
  (&extern_share_data_shared[0])[local_id] = (z[thread_id] * z[thread_id]);
  }
  return;
}
// FUNCTION ORDER ID 34 END


// FUNCTION ORDER ID 35 START
// INSERT COMMENT FUNCTION: gpu_kernel_eleven_device
void gpu_kernel_eleven_device(double norm_temp2, double* x, double* z, uint32_t __FIXME__gridDim_2e_x, uint32_t __FIXME__gridDim_2e_y, uint32_t __FIXME__gridDim_2e_z, uint32_t __FIXME__blockDim_2e_x, uint32_t __FIXME__blockDim_2e_y, uint32_t __FIXME__blockDim_2e_z, uint32_t __FIXME__blockIdx_2e_x, uint32_t __FIXME__blockIdx_2e_y, uint32_t __FIXME__blockIdx_2e_z, uint32_t __FIXME__threadIdx_2e_x, uint32_t __FIXME__threadIdx_2e_y, uint32_t __FIXME__threadIdx_2e_z) {
  int64_t j;

// INSERT COMMENT IFELSE: gpu_kernel_eleven_device::entry
  j = __FIXME__blockIdx_2e_x * __FIXME__blockDim_2e_x + __FIXME__threadIdx_2e_x;
  if (j >= 14000) { // IFELSE MARKER: entry IF
  } else { // IFELSE MARKER: entry ELSE
  x[j] = (norm_temp2 * z[j]);
  }
  return;
}
// FUNCTION ORDER ID 35 END


// FUNCTION ORDER ID 36 START
// INSERT COMMENT FUNCTION: gpu_kernel_one_device
void gpu_kernel_one_device(double* p, double* q, double* r, double* x, double* z, uint32_t __FIXME__gridDim_2e_x, uint32_t __FIXME__gridDim_2e_y, uint32_t __FIXME__gridDim_2e_z, uint32_t __FIXME__blockDim_2e_x, uint32_t __FIXME__blockDim_2e_y, uint32_t __FIXME__blockDim_2e_z, uint32_t __FIXME__blockIdx_2e_x, uint32_t __FIXME__blockIdx_2e_y, uint32_t __FIXME__blockIdx_2e_z, uint32_t __FIXME__threadIdx_2e_x, uint32_t __FIXME__threadIdx_2e_y, uint32_t __FIXME__threadIdx_2e_z) {
  int64_t thread_id;

// INSERT COMMENT IFELSE: gpu_kernel_one_device::entry
  thread_id = __FIXME__blockIdx_2e_x * __FIXME__blockDim_2e_x + __FIXME__threadIdx_2e_x;
  if (thread_id >= 14000) { // IFELSE MARKER: entry IF
  } else { // IFELSE MARKER: entry ELSE
  q[thread_id] = 0;
  z[thread_id] = 0;
  r[thread_id] = x[thread_id];
  p[thread_id] = x[thread_id];
  }
  return;
}
// FUNCTION ORDER ID 36 END


// FUNCTION ORDER ID 37 START
// INSERT COMMENT FUNCTION: gpu_kernel_two_device0
void gpu_kernel_two_device0(double* r, double* rho, double* global_data, uint32_t __FIXME__gridDim_2e_x, uint32_t __FIXME__gridDim_2e_y, uint32_t __FIXME__gridDim_2e_z, uint32_t __FIXME__blockDim_2e_x, uint32_t __FIXME__blockDim_2e_y, uint32_t __FIXME__blockDim_2e_z, uint32_t __FIXME__blockIdx_2e_x, uint32_t __FIXME__blockIdx_2e_y, uint32_t __FIXME__blockIdx_2e_z, uint32_t local_id, uint32_t __FIXME__threadIdx_2e_y, uint32_t __FIXME__threadIdx_2e_z) {
  int64_t thread_id;

// INSERT COMMENT IFELSE: gpu_kernel_two_device0::entry
  thread_id = __FIXME__blockIdx_2e_x * __FIXME__blockDim_2e_x + local_id;
  (&extern_share_data_shared[0])[local_id] = 0;
  if (thread_id < 14000) { // IFELSE MARKER: entry IF
  (&extern_share_data_shared[0])[local_id] = (r[thread_id] * r[thread_id]);
  }
  return;
}
// FUNCTION ORDER ID 37 END


// FUNCTION ORDER ID 38 START
// INSERT COMMENT FUNCTION: gpu_kernel_three_device0
void gpu_kernel_three_device0(uint32_t* colidx, uint32_t* rowstr, double* a, double* p, double* q, uint32_t __FIXME__gridDim_2e_x, uint32_t __FIXME__gridDim_2e_y, uint32_t __FIXME__gridDim_2e_z, uint32_t __FIXME__blockDim_2e_x, uint32_t __FIXME__blockDim_2e_y, uint32_t __FIXME__blockDim_2e_z, uint32_t __FIXME__blockIdx_2e_x, uint32_t __FIXME__blockIdx_2e_y, uint32_t __FIXME__blockIdx_2e_z, uint32_t local_id, uint32_t __FIXME__threadIdx_2e_y, uint32_t __FIXME__threadIdx_2e_z) {
  int64_t j;
  int32_t k;
  double sum;

  j = ((__FIXME__blockIdx_2e_x * __FIXME__blockDim_2e_x + local_id) / __FIXME__blockDim_2e_x);
  k = rowstr[j] + local_id;
  sum = 0;
// INSERT COMMENT LOOP: gpu_kernel_three_device0::for.cond
for(int32_t k = rowstr[j] + local_id; k < rowstr[(j + 1)];   k = k + __FIXME__blockDim_2e_x) {
  sum = (sum + (a[k] * p[colidx[k]]));
  sum = sum;
}
  (&extern_share_data_shared[0])[local_id] = sum;
}
// FUNCTION ORDER ID 38 END


// FUNCTION ORDER ID 39 START
// INSERT COMMENT FUNCTION: gpu_kernel_four_device0
void gpu_kernel_four_device0(double* d, double* p, double* q, double* global_data, uint32_t __FIXME__gridDim_2e_x, uint32_t __FIXME__gridDim_2e_y, uint32_t __FIXME__gridDim_2e_z, uint32_t __FIXME__blockDim_2e_x, uint32_t __FIXME__blockDim_2e_y, uint32_t __FIXME__blockDim_2e_z, uint32_t __FIXME__blockIdx_2e_x, uint32_t __FIXME__blockIdx_2e_y, uint32_t __FIXME__blockIdx_2e_z, uint32_t local_id, uint32_t __FIXME__threadIdx_2e_y, uint32_t __FIXME__threadIdx_2e_z) {
  int64_t thread_id;

// INSERT COMMENT IFELSE: gpu_kernel_four_device0::entry
  thread_id = __FIXME__blockIdx_2e_x * __FIXME__blockDim_2e_x + local_id;
  (&extern_share_data_shared[0])[local_id] = 0;
  (&extern_share_data_shared[0])[local_id] = 0;
  if (thread_id < 14000) { // IFELSE MARKER: entry IF
  (&extern_share_data_shared[0])[local_id] = (p[thread_id] * q[thread_id]);
  }
  return;
}
// FUNCTION ORDER ID 39 END


// FUNCTION ORDER ID 40 START
// INSERT COMMENT FUNCTION: gpu_kernel_five_1
void gpu_kernel_five_1(double alpha, double* p, double* z, uint32_t __FIXME__gridDim_2e_x, uint32_t __FIXME__gridDim_2e_y, uint32_t __FIXME__gridDim_2e_z, uint32_t __FIXME__blockDim_2e_x, uint32_t __FIXME__blockDim_2e_y, uint32_t __FIXME__blockDim_2e_z, uint32_t __FIXME__blockIdx_2e_x, uint32_t __FIXME__blockIdx_2e_y, uint32_t __FIXME__blockIdx_2e_z, uint32_t __FIXME__threadIdx_2e_x, uint32_t __FIXME__threadIdx_2e_y, uint32_t __FIXME__threadIdx_2e_z) {
  int64_t j;

// INSERT COMMENT IFELSE: gpu_kernel_five_1::entry
  j = __FIXME__blockIdx_2e_x * __FIXME__blockDim_2e_x + __FIXME__threadIdx_2e_x;
  if (j >= 14000) { // IFELSE MARKER: entry IF
  } else { // IFELSE MARKER: entry ELSE
  z[j] = (z[j] + (alpha * p[j]));
  }
  return;
}
// FUNCTION ORDER ID 40 END


// FUNCTION ORDER ID 41 START
// INSERT COMMENT FUNCTION: gpu_kernel_five_2
void gpu_kernel_five_2(double alpha, double* q, double* r, uint32_t __FIXME__gridDim_2e_x, uint32_t __FIXME__gridDim_2e_y, uint32_t __FIXME__gridDim_2e_z, uint32_t __FIXME__blockDim_2e_x, uint32_t __FIXME__blockDim_2e_y, uint32_t __FIXME__blockDim_2e_z, uint32_t __FIXME__blockIdx_2e_x, uint32_t __FIXME__blockIdx_2e_y, uint32_t __FIXME__blockIdx_2e_z, uint32_t __FIXME__threadIdx_2e_x, uint32_t __FIXME__threadIdx_2e_y, uint32_t __FIXME__threadIdx_2e_z) {
  int64_t j;

// INSERT COMMENT IFELSE: gpu_kernel_five_2::entry
  j = __FIXME__blockIdx_2e_x * __FIXME__blockDim_2e_x + __FIXME__threadIdx_2e_x;
  if (j >= 14000) { // IFELSE MARKER: entry IF
  } else { // IFELSE MARKER: entry ELSE
  r[j] = (r[j] - (alpha * q[j]));
  }
  return;
}
// FUNCTION ORDER ID 41 END


// FUNCTION ORDER ID 42 START
// INSERT COMMENT FUNCTION: gpu_kernel_six_device0
void gpu_kernel_six_device0(double* r, double* global_data, uint32_t __FIXME__gridDim_2e_x, uint32_t __FIXME__gridDim_2e_y, uint32_t __FIXME__gridDim_2e_z, uint32_t __FIXME__blockDim_2e_x, uint32_t __FIXME__blockDim_2e_y, uint32_t __FIXME__blockDim_2e_z, uint32_t __FIXME__blockIdx_2e_x, uint32_t __FIXME__blockIdx_2e_y, uint32_t __FIXME__blockIdx_2e_z, uint32_t local_id, uint32_t __FIXME__threadIdx_2e_y, uint32_t __FIXME__threadIdx_2e_z) {
  int64_t thread_id;

// INSERT COMMENT IFELSE: gpu_kernel_six_device0::entry
  thread_id = __FIXME__blockIdx_2e_x * __FIXME__blockDim_2e_x + local_id;
  (&extern_share_data_shared[0])[local_id] = 0;
  if (thread_id < 14000) { // IFELSE MARKER: entry IF
  (&extern_share_data_shared[0])[local_id] = (r[thread_id] * r[thread_id]);
  }
  return;
}
// FUNCTION ORDER ID 42 END


// FUNCTION ORDER ID 43 START
// INSERT COMMENT FUNCTION: gpu_kernel_seven_device
void gpu_kernel_seven_device(double beta, double* p, double* r, uint32_t __FIXME__gridDim_2e_x, uint32_t __FIXME__gridDim_2e_y, uint32_t __FIXME__gridDim_2e_z, uint32_t __FIXME__blockDim_2e_x, uint32_t __FIXME__blockDim_2e_y, uint32_t __FIXME__blockDim_2e_z, uint32_t __FIXME__blockIdx_2e_x, uint32_t __FIXME__blockIdx_2e_y, uint32_t __FIXME__blockIdx_2e_z, uint32_t __FIXME__threadIdx_2e_x, uint32_t __FIXME__threadIdx_2e_y, uint32_t __FIXME__threadIdx_2e_z) {
  int64_t j;

// INSERT COMMENT IFELSE: gpu_kernel_seven_device::entry
  j = __FIXME__blockIdx_2e_x * __FIXME__blockDim_2e_x + __FIXME__threadIdx_2e_x;
  if (j >= 14000) { // IFELSE MARKER: entry IF
  } else { // IFELSE MARKER: entry ELSE
  p[j] = (r[j] + (beta * p[j]));
  }
  return;
}
// FUNCTION ORDER ID 43 END


// FUNCTION ORDER ID 44 START
// INSERT COMMENT FUNCTION: gpu_kernel_eight_device0
void gpu_kernel_eight_device0(uint32_t* colidx, uint32_t* rowstr, double* a, double* r, double* z, uint32_t __FIXME__gridDim_2e_x, uint32_t __FIXME__gridDim_2e_y, uint32_t __FIXME__gridDim_2e_z, uint32_t __FIXME__blockDim_2e_x, uint32_t __FIXME__blockDim_2e_y, uint32_t __FIXME__blockDim_2e_z, uint32_t __FIXME__blockIdx_2e_x, uint32_t __FIXME__blockIdx_2e_y, uint32_t __FIXME__blockIdx_2e_z, uint32_t local_id, uint32_t __FIXME__threadIdx_2e_y, uint32_t __FIXME__threadIdx_2e_z) {
  int64_t j;
  int32_t k;
  double sum;

  j = ((__FIXME__blockIdx_2e_x * __FIXME__blockDim_2e_x + local_id) / __FIXME__blockDim_2e_x);
  k = rowstr[j] + local_id;
  sum = 0;
// INSERT COMMENT LOOP: gpu_kernel_eight_device0::for.cond
for(int32_t k = rowstr[j] + local_id; k < rowstr[(j + 1)];   k = k + __FIXME__blockDim_2e_x) {
  sum = (sum + (a[k] * z[colidx[k]]));
  sum = sum;
}
  (&extern_share_data_shared[0])[local_id] = sum;
}
// FUNCTION ORDER ID 44 END


// FUNCTION ORDER ID 45 START
// INSERT COMMENT FUNCTION: gpu_kernel_nine_device0
void gpu_kernel_nine_device0(double* r, double* x, double* sum, double* global_data, uint32_t __FIXME__gridDim_2e_x, uint32_t __FIXME__gridDim_2e_y, uint32_t __FIXME__gridDim_2e_z, uint32_t __FIXME__blockDim_2e_x, uint32_t __FIXME__blockDim_2e_y, uint32_t __FIXME__blockDim_2e_z, uint32_t __FIXME__blockIdx_2e_x, uint32_t __FIXME__blockIdx_2e_y, uint32_t __FIXME__blockIdx_2e_z, uint32_t local_id, uint32_t __FIXME__threadIdx_2e_y, uint32_t __FIXME__threadIdx_2e_z) {
  int64_t thread_id;

// INSERT COMMENT IFELSE: gpu_kernel_nine_device0::entry
  thread_id = __FIXME__blockIdx_2e_x * __FIXME__blockDim_2e_x + local_id;
  (&extern_share_data_shared[0])[local_id] = 0;
  if (thread_id < 14000) { // IFELSE MARKER: entry IF
  (&extern_share_data_shared[0])[local_id] = (x[thread_id] - r[thread_id]);
  (&extern_share_data_shared[0])[local_id] = ((&extern_share_data_shared[0])[local_id] * (&extern_share_data_shared[0])[local_id]);
  }
  return;
}
// FUNCTION ORDER ID 45 END


// FUNCTION ORDER ID 46 START
// INSERT COMMENT FUNCTION: gpu_kernel_ten_21
void gpu_kernel_ten_21(double* norm_temp, double* x, double* z, uint32_t __FIXME__gridDim_2e_x, uint32_t __FIXME__gridDim_2e_y, uint32_t __FIXME__gridDim_2e_z, uint32_t __FIXME__blockDim_2e_x, uint32_t __FIXME__blockDim_2e_y, uint32_t __FIXME__blockDim_2e_z, uint32_t __FIXME__blockIdx_2e_x, uint32_t __FIXME__blockIdx_2e_y, uint32_t __FIXME__blockIdx_2e_z, uint32_t local_id, uint32_t __FIXME__threadIdx_2e_y, uint32_t __FIXME__threadIdx_2e_z) {
  int64_t i;

  (&extern_share_data_shared[0])[local_id] = 0;
  /*__FIXME__INTRINSIC_CALL__*///sync point
;
  if (!(local_id == 0)) {
  return;
  }
// INSERT COMMENT LOOP: gpu_kernel_ten_21::for.cond
for(int64_t i = 1; i < __FIXME__blockDim_2e_x;   i = i + 1) {
  (&extern_share_data_shared[0])[0] = ((&extern_share_data_shared[0])[0] + (&extern_share_data_shared[0])[i]);
}
  norm_temp[__FIXME__blockIdx_2e_x] = (&extern_share_data_shared[0])[0];
  return;
}
// FUNCTION ORDER ID 46 END


// FUNCTION ORDER ID 47 START
// INSERT COMMENT FUNCTION: gpu_kernel_ten_11
void gpu_kernel_ten_11(double* norm_temp, double* x, double* z, uint32_t __FIXME__gridDim_2e_x, uint32_t __FIXME__gridDim_2e_y, uint32_t __FIXME__gridDim_2e_z, uint32_t __FIXME__blockDim_2e_x, uint32_t __FIXME__blockDim_2e_y, uint32_t __FIXME__blockDim_2e_z, uint32_t __FIXME__blockIdx_2e_x, uint32_t __FIXME__blockIdx_2e_y, uint32_t __FIXME__blockIdx_2e_z, uint32_t local_id, uint32_t __FIXME__threadIdx_2e_y, uint32_t __FIXME__threadIdx_2e_z) {
  int64_t i;

  (&extern_share_data_shared[0])[local_id] = 0;
  /*__FIXME__INTRINSIC_CALL__*///sync point
;
  if (!(local_id == 0)) {
  return;
  }
// INSERT COMMENT LOOP: gpu_kernel_ten_11::for.cond
for(int64_t i = 1; i < __FIXME__blockDim_2e_x;   i = i + 1) {
  (&extern_share_data_shared[0])[0] = ((&extern_share_data_shared[0])[0] + (&extern_share_data_shared[0])[i]);
}
  norm_temp[__FIXME__blockIdx_2e_x] = (&extern_share_data_shared[0])[0];
  return;
}
// FUNCTION ORDER ID 47 END


// FUNCTION ORDER ID 48 START
// INSERT COMMENT FUNCTION: gpu_kernel_three_device1
void gpu_kernel_three_device1(uint32_t* colidx, uint32_t* rowstr, double* a, double* p, double* q, uint32_t __FIXME__gridDim_2e_x, uint32_t __FIXME__gridDim_2e_y, uint32_t __FIXME__gridDim_2e_z, uint32_t __FIXME__blockDim_2e_x, uint32_t __FIXME__blockDim_2e_y, uint32_t __FIXME__blockDim_2e_z, uint32_t __FIXME__blockIdx_2e_x, uint32_t __FIXME__blockIdx_2e_y, uint32_t __FIXME__blockIdx_2e_z, uint32_t local_id, uint32_t __FIXME__threadIdx_2e_y, uint32_t __FIXME__threadIdx_2e_z) {
  int64_t j;
  int64_t i;

  j = ((__FIXME__blockIdx_2e_x * __FIXME__blockDim_2e_x + local_id) / __FIXME__blockDim_2e_x);
  /*__FIXME__INTRINSIC_CALL__*///sync point
;
  if (!(local_id == 0)) {
  return;
  }
// INSERT COMMENT LOOP: gpu_kernel_three_device1::for.cond22
for(int64_t i = 1; i < __FIXME__blockDim_2e_x;   i = i + 1) {
  (&extern_share_data_shared[0])[0] = ((&extern_share_data_shared[0])[0] + (&extern_share_data_shared[0])[i]);
}
  q[j] = (&extern_share_data_shared[0])[0];
  return;
}
// FUNCTION ORDER ID 48 END


// FUNCTION ORDER ID 49 START
// INSERT COMMENT FUNCTION: gpu_kernel_nine_device1
void gpu_kernel_nine_device1(double* r, double* x, double* sum, double* global_data, uint32_t __FIXME__gridDim_2e_x, uint32_t __FIXME__gridDim_2e_y, uint32_t __FIXME__gridDim_2e_z, uint32_t __FIXME__blockDim_2e_x, uint32_t __FIXME__blockDim_2e_y, uint32_t __FIXME__blockDim_2e_z, uint32_t __FIXME__blockIdx_2e_x, uint32_t __FIXME__blockIdx_2e_y, uint32_t __FIXME__blockIdx_2e_z, uint32_t local_id, uint32_t __FIXME__threadIdx_2e_y, uint32_t __FIXME__threadIdx_2e_z) {
  int64_t i;

  (&extern_share_data_shared[0])[local_id] = 0;
  /*__FIXME__INTRINSIC_CALL__*///sync point
;
  if (!(local_id == 0)) {
  return;
  }
// INSERT COMMENT LOOP: gpu_kernel_nine_device1::for.cond
for(int64_t i = 1; i < __FIXME__blockDim_2e_x;   i = i + 1) {
  (&extern_share_data_shared[0])[0] = ((&extern_share_data_shared[0])[0] + (&extern_share_data_shared[0])[i]);
}
  global_data[__FIXME__blockIdx_2e_x] = (&extern_share_data_shared[0])[0];
  return;
}
// FUNCTION ORDER ID 49 END


// FUNCTION ORDER ID 50 START
// INSERT COMMENT FUNCTION: gpu_kernel_six_device1
void gpu_kernel_six_device1(double* r, double* global_data, uint32_t __FIXME__gridDim_2e_x, uint32_t __FIXME__gridDim_2e_y, uint32_t __FIXME__gridDim_2e_z, uint32_t __FIXME__blockDim_2e_x, uint32_t __FIXME__blockDim_2e_y, uint32_t __FIXME__blockDim_2e_z, uint32_t __FIXME__blockIdx_2e_x, uint32_t __FIXME__blockIdx_2e_y, uint32_t __FIXME__blockIdx_2e_z, uint32_t local_id, uint32_t __FIXME__threadIdx_2e_y, uint32_t __FIXME__threadIdx_2e_z) {
  int64_t i;

  (&extern_share_data_shared[0])[local_id] = 0;
  /*__FIXME__INTRINSIC_CALL__*///sync point
;
  if (!(local_id == 0)) {
  return;
  }
// INSERT COMMENT LOOP: gpu_kernel_six_device1::for.cond
for(int64_t i = 1; i < __FIXME__blockDim_2e_x;   i = i + 1) {
  (&extern_share_data_shared[0])[0] = ((&extern_share_data_shared[0])[0] + (&extern_share_data_shared[0])[i]);
}
  global_data[__FIXME__blockIdx_2e_x] = (&extern_share_data_shared[0])[0];
  return;
}
// FUNCTION ORDER ID 50 END


// FUNCTION ORDER ID 51 START
// INSERT COMMENT FUNCTION: gpu_kernel_eight_device1
void gpu_kernel_eight_device1(uint32_t* colidx, uint32_t* rowstr, double* a, double* r, double* z, uint32_t __FIXME__gridDim_2e_x, uint32_t __FIXME__gridDim_2e_y, uint32_t __FIXME__gridDim_2e_z, uint32_t __FIXME__blockDim_2e_x, uint32_t __FIXME__blockDim_2e_y, uint32_t __FIXME__blockDim_2e_z, uint32_t __FIXME__blockIdx_2e_x, uint32_t __FIXME__blockIdx_2e_y, uint32_t __FIXME__blockIdx_2e_z, uint32_t local_id, uint32_t __FIXME__threadIdx_2e_y, uint32_t __FIXME__threadIdx_2e_z) {
  int64_t j;
  int64_t i;

  j = ((__FIXME__blockIdx_2e_x * __FIXME__blockDim_2e_x + local_id) / __FIXME__blockDim_2e_x);
  /*__FIXME__INTRINSIC_CALL__*///sync point
;
  if (!(local_id == 0)) {
  return;
  }
// INSERT COMMENT LOOP: gpu_kernel_eight_device1::for.cond22
for(int64_t i = 1; i < __FIXME__blockDim_2e_x;   i = i + 1) {
  (&extern_share_data_shared[0])[0] = ((&extern_share_data_shared[0])[0] + (&extern_share_data_shared[0])[i]);
}
  r[j] = (&extern_share_data_shared[0])[0];
  return;
}
// FUNCTION ORDER ID 51 END


// FUNCTION ORDER ID 52 START
// INSERT COMMENT FUNCTION: gpu_kernel_two_device1
void gpu_kernel_two_device1(double* r, double* rho, double* global_data, uint32_t __FIXME__gridDim_2e_x, uint32_t __FIXME__gridDim_2e_y, uint32_t __FIXME__gridDim_2e_z, uint32_t __FIXME__blockDim_2e_x, uint32_t __FIXME__blockDim_2e_y, uint32_t __FIXME__blockDim_2e_z, uint32_t __FIXME__blockIdx_2e_x, uint32_t __FIXME__blockIdx_2e_y, uint32_t __FIXME__blockIdx_2e_z, uint32_t local_id, uint32_t __FIXME__threadIdx_2e_y, uint32_t __FIXME__threadIdx_2e_z) {
  int64_t i;

  (&extern_share_data_shared[0])[local_id] = 0;
  /*__FIXME__INTRINSIC_CALL__*///sync point
;
  if (!(local_id == 0)) {
  return;
  }
// INSERT COMMENT LOOP: gpu_kernel_two_device1::for.cond
for(int64_t i = 1; i < __FIXME__blockDim_2e_x;   i = i + 1) {
  (&extern_share_data_shared[0])[0] = ((&extern_share_data_shared[0])[0] + (&extern_share_data_shared[0])[i]);
}
  global_data[__FIXME__blockIdx_2e_x] = (&extern_share_data_shared[0])[0];
  return;
}
// FUNCTION ORDER ID 52 END


// FUNCTION ORDER ID 53 START
// INSERT COMMENT FUNCTION: gpu_kernel_four_device1
void gpu_kernel_four_device1(double* d, double* p, double* q, double* global_data, uint32_t __FIXME__gridDim_2e_x, uint32_t __FIXME__gridDim_2e_y, uint32_t __FIXME__gridDim_2e_z, uint32_t __FIXME__blockDim_2e_x, uint32_t __FIXME__blockDim_2e_y, uint32_t __FIXME__blockDim_2e_z, uint32_t __FIXME__blockIdx_2e_x, uint32_t __FIXME__blockIdx_2e_y, uint32_t __FIXME__blockIdx_2e_z, uint32_t local_id, uint32_t __FIXME__threadIdx_2e_y, uint32_t __FIXME__threadIdx_2e_z) {
  int64_t i;

  (&extern_share_data_shared[0])[local_id] = 0;
  (&extern_share_data_shared[0])[local_id] = 0;
  /*__FIXME__INTRINSIC_CALL__*///sync point
;
  if (!(local_id == 0)) {
  return;
  }
// INSERT COMMENT LOOP: gpu_kernel_four_device1::for.cond
for(int64_t i = 1; i < __FIXME__blockDim_2e_x;   i = i + 1) {
  (&extern_share_data_shared[0])[0] = ((&extern_share_data_shared[0])[0] + (&extern_share_data_shared[0])[i]);
}
  global_data[__FIXME__blockIdx_2e_x] = (&extern_share_data_shared[0])[0];
  return;
}
// FUNCTION ORDER ID 53 END

