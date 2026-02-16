; ModuleID = 'cg.cu'
source_filename = "cg.cu"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.cudaDeviceProp = type { [256 x i8], i64, i64, i32, i32, i64, i32, [3 x i32], [3 x i32], i32, i64, i32, i32, i64, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, [2 x i32], [2 x i32], [3 x i32], [2 x i32], [3 x i32], [3 x i32], i32, [2 x i32], [3 x i32], [2 x i32], i32, [2 x i32], [3 x i32], [2 x i32], [3 x i32], i32, [2 x i32], i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i64, i32, i32, i32, i32 }
%struct.dim3 = type { i32, i32, i32 }
%struct.CUstream_st = type opaque

$_ZN4dim3C2Ejjj = comdat any

@colidx_device = dso_local global i32* null, align 8, !dbg !0
@rowstr_device = dso_local global i32* null, align 8, !dbg !112
@a_device = dso_local global double* null, align 8, !dbg !114
@p_device = dso_local global double* null, align 8, !dbg !116
@q_device = dso_local global double* null, align 8, !dbg !118
@r_device = dso_local global double* null, align 8, !dbg !120
@x_device = dso_local global double* null, align 8, !dbg !122
@z_device = dso_local global double* null, align 8, !dbg !124
@rho_device = dso_local global double* null, align 8, !dbg !126
@d_device = dso_local global double* null, align 8, !dbg !128
@alpha_device = dso_local global double* null, align 8, !dbg !130
@beta_device = dso_local global double* null, align 8, !dbg !132
@sum_device = dso_local global double* null, align 8, !dbg !134
@norm_temp1_device = dso_local global double* null, align 8, !dbg !136
@norm_temp2_device = dso_local global double* null, align 8, !dbg !138
@global_data = dso_local global double* null, align 8, !dbg !140
@global_data_two = dso_local global double* null, align 8, !dbg !142
@global_data_device = dso_local global double* null, align 8, !dbg !144
@global_data_two_device = dso_local global double* null, align 8, !dbg !146
@global_data_reduce = dso_local global double 0.000000e+00, align 8, !dbg !148
@global_data_two_reduce = dso_local global double 0.000000e+00, align 8, !dbg !150
@global_data_elements = dso_local global i64 0, align 8, !dbg !152
@size_global_data = dso_local global i64 0, align 8, !dbg !157
@size_colidx_device = dso_local global i64 0, align 8, !dbg !159
@size_rowstr_device = dso_local global i64 0, align 8, !dbg !161
@size_iv_device = dso_local global i64 0, align 8, !dbg !163
@size_arow_device = dso_local global i64 0, align 8, !dbg !165
@size_acol_device = dso_local global i64 0, align 8, !dbg !167
@size_aelt_device = dso_local global i64 0, align 8, !dbg !169
@size_a_device = dso_local global i64 0, align 8, !dbg !171
@size_x_device = dso_local global i64 0, align 8, !dbg !173
@size_z_device = dso_local global i64 0, align 8, !dbg !175
@size_p_device = dso_local global i64 0, align 8, !dbg !177
@size_q_device = dso_local global i64 0, align 8, !dbg !179
@size_r_device = dso_local global i64 0, align 8, !dbg !181
@size_rho_device = dso_local global i64 0, align 8, !dbg !183
@size_d_device = dso_local global i64 0, align 8, !dbg !185
@size_alpha_device = dso_local global i64 0, align 8, !dbg !187
@size_beta_device = dso_local global i64 0, align 8, !dbg !189
@size_sum_device = dso_local global i64 0, align 8, !dbg !191
@size_norm_temp1_device = dso_local global i64 0, align 8, !dbg !193
@size_norm_temp2_device = dso_local global i64 0, align 8, !dbg !195
@blocks_per_grid_on_kernel_one = dso_local global i32 0, align 4, !dbg !197
@blocks_per_grid_on_kernel_two = dso_local global i32 0, align 4, !dbg !199
@blocks_per_grid_on_kernel_three = dso_local global i32 0, align 4, !dbg !201
@blocks_per_grid_on_kernel_four = dso_local global i32 0, align 4, !dbg !203
@blocks_per_grid_on_kernel_five = dso_local global i32 0, align 4, !dbg !205
@blocks_per_grid_on_kernel_six = dso_local global i32 0, align 4, !dbg !207
@blocks_per_grid_on_kernel_seven = dso_local global i32 0, align 4, !dbg !209
@blocks_per_grid_on_kernel_eight = dso_local global i32 0, align 4, !dbg !211
@blocks_per_grid_on_kernel_nine = dso_local global i32 0, align 4, !dbg !213
@blocks_per_grid_on_kernel_ten = dso_local global i32 0, align 4, !dbg !215
@blocks_per_grid_on_kernel_eleven = dso_local global i32 0, align 4, !dbg !217
@threads_per_block_on_kernel_one = dso_local global i32 0, align 4, !dbg !219
@threads_per_block_on_kernel_two = dso_local global i32 0, align 4, !dbg !221
@threads_per_block_on_kernel_three = dso_local global i32 0, align 4, !dbg !223
@threads_per_block_on_kernel_four = dso_local global i32 0, align 4, !dbg !225
@threads_per_block_on_kernel_five = dso_local global i32 0, align 4, !dbg !227
@threads_per_block_on_kernel_six = dso_local global i32 0, align 4, !dbg !229
@threads_per_block_on_kernel_seven = dso_local global i32 0, align 4, !dbg !231
@threads_per_block_on_kernel_eight = dso_local global i32 0, align 4, !dbg !233
@threads_per_block_on_kernel_nine = dso_local global i32 0, align 4, !dbg !235
@threads_per_block_on_kernel_ten = dso_local global i32 0, align 4, !dbg !237
@threads_per_block_on_kernel_eleven = dso_local global i32 0, align 4, !dbg !239
@size_shared_data_on_kernel_one = dso_local global i64 0, align 8, !dbg !241
@size_shared_data_on_kernel_two = dso_local global i64 0, align 8, !dbg !243
@size_shared_data_on_kernel_three = dso_local global i64 0, align 8, !dbg !245
@size_shared_data_on_kernel_four = dso_local global i64 0, align 8, !dbg !247
@size_shared_data_on_kernel_five = dso_local global i64 0, align 8, !dbg !249
@size_shared_data_on_kernel_six = dso_local global i64 0, align 8, !dbg !251
@size_shared_data_on_kernel_seven = dso_local global i64 0, align 8, !dbg !253
@size_shared_data_on_kernel_eight = dso_local global i64 0, align 8, !dbg !255
@size_shared_data_on_kernel_nine = dso_local global i64 0, align 8, !dbg !257
@size_shared_data_on_kernel_ten = dso_local global i64 0, align 8, !dbg !259
@size_shared_data_on_kernel_eleven = dso_local global i64 0, align 8, !dbg !261
@size_reduce_memory_on_kernel_one = dso_local global i64 0, align 8, !dbg !263
@size_reduce_memory_on_kernel_two = dso_local global i64 0, align 8, !dbg !265
@size_reduce_memory_on_kernel_three = dso_local global i64 0, align 8, !dbg !267
@size_reduce_memory_on_kernel_four = dso_local global i64 0, align 8, !dbg !269
@size_reduce_memory_on_kernel_five = dso_local global i64 0, align 8, !dbg !271
@size_reduce_memory_on_kernel_six = dso_local global i64 0, align 8, !dbg !273
@size_reduce_memory_on_kernel_seven = dso_local global i64 0, align 8, !dbg !275
@size_reduce_memory_on_kernel_eight = dso_local global i64 0, align 8, !dbg !277
@size_reduce_memory_on_kernel_nine = dso_local global i64 0, align 8, !dbg !279
@size_reduce_memory_on_kernel_ten = dso_local global i64 0, align 8, !dbg !281
@size_reduce_memory_on_kernel_eleven = dso_local global i64 0, align 8, !dbg !283
@gpu_device_id = dso_local global i32 0, align 4, !dbg !285
@total_devices = dso_local global i32 0, align 4, !dbg !287
@gpu_device_properties = dso_local global %struct.cudaDeviceProp zeroinitializer, align 8, !dbg !289
@.str = private unnamed_addr constant [27 x i8] c"\0A\0A %s Benchmark Completed\0A\00", align 1
@.str.1 = private unnamed_addr constant [46 x i8] c" class_npb       =                        %c\0A\00", align 1
@.str.2 = private unnamed_addr constant [38 x i8] c" Size            =             %12ld\0A\00", align 1
@.str.3 = private unnamed_addr constant [44 x i8] c" Size            =             %4dx%4dx%4d\0A\00", align 1
@.str.4 = private unnamed_addr constant [8 x i8] c"%15.0lf\00", align 1
@.str.5 = private unnamed_addr constant [34 x i8] c" Size            =          %15s\0A\00", align 1
@.str.6 = private unnamed_addr constant [37 x i8] c" Size            =             %12d\0A\00", align 1
@.str.7 = private unnamed_addr constant [42 x i8] c" Size            =           %4dx%4dx%4d\0A\00", align 1
@.str.8 = private unnamed_addr constant [37 x i8] c" Iterations      =             %12d\0A\00", align 1
@.str.9 = private unnamed_addr constant [39 x i8] c" Time in seconds =             %12.2f\0A\00", align 1
@.str.10 = private unnamed_addr constant [39 x i8] c" Mop/s total     =             %12.2f\0A\00", align 1
@.str.11 = private unnamed_addr constant [25 x i8] c" Operation type  = %24s\0A\00", align 1
@.str.12 = private unnamed_addr constant [45 x i8] c" Verification    =            NOT PERFORMED\0A\00", align 1
@.str.13 = private unnamed_addr constant [45 x i8] c" Verification    =               SUCCESSFUL\0A\00", align 1
@.str.14 = private unnamed_addr constant [45 x i8] c" Verification    =             UNSUCCESSFUL\0A\00", align 1
@.str.15 = private unnamed_addr constant [37 x i8] c" Version         =             %12s\0A\00", align 1
@.str.16 = private unnamed_addr constant [37 x i8] c" Compile date    =             %12s\0A\00", align 1
@.str.17 = private unnamed_addr constant [37 x i8] c" NVCC version    =             %12s\0A\00", align 1
@.str.18 = private unnamed_addr constant [37 x i8] c" CUDA version    =             %12s\0A\00", align 1
@.str.19 = private unnamed_addr constant [20 x i8] c"\0A Compile options:\0A\00", align 1
@.str.20 = private unnamed_addr constant [23 x i8] c"    CC           = %s\0A\00", align 1
@.str.21 = private unnamed_addr constant [23 x i8] c"    CLINK        = %s\0A\00", align 1
@.str.22 = private unnamed_addr constant [23 x i8] c"    C_LIB        = %s\0A\00", align 1
@.str.23 = private unnamed_addr constant [23 x i8] c"    C_INC        = %s\0A\00", align 1
@.str.24 = private unnamed_addr constant [23 x i8] c"    CFLAGS       = %s\0A\00", align 1
@.str.25 = private unnamed_addr constant [23 x i8] c"    CLINKFLAGS   = %s\0A\00", align 1
@.str.26 = private unnamed_addr constant [23 x i8] c"    RAND         = %s\0A\00", align 1
@.str.27 = private unnamed_addr constant [13 x i8] c"\0A Hardware:\0A\00", align 1
@.str.28 = private unnamed_addr constant [23 x i8] c"    CPU device   = %s\0A\00", align 1
@.str.29 = private unnamed_addr constant [23 x i8] c"    GPU device   = %s\0A\00", align 1
@.str.30 = private unnamed_addr constant [13 x i8] c"\0A Software:\0A\00", align 1
@.str.31 = private unnamed_addr constant [23 x i8] c"    Parameters   = %s\0A\00", align 1
@.str.32 = private unnamed_addr constant [2 x i8] c"\0A\00", align 1
@.str.33 = private unnamed_addr constant [72 x i8] c"----------------------------------------------------------------------\0A\00", align 1
@.str.34 = private unnamed_addr constant [27 x i8] c" NPB-CPP is developed by:\0A\00", align 1
@.str.35 = private unnamed_addr constant [56 x i8] c"            Dalvan Griebler <dalvangriebler@gmail.com>\0A\00", align 1
@.str.36 = private unnamed_addr constant [52 x i8] c"            Gabriell Araujo <hexenoften@gmail.com>\0A\00", align 1
@.str.37 = private unnamed_addr constant [46 x i8] c"            J\C3\BAnior L\C3\B6ff <loffjh@gmail.com>\0A\00", align 1
@.str.38 = private unnamed_addr constant [43 x i8] c" In case of problems, send an email to us\0A\00", align 1
@_ZL6colidx = internal global i32* null, align 8, !dbg !364
@_ZL6rowstr = internal global i32* null, align 8, !dbg !366
@_ZL2iv = internal global i32* null, align 8, !dbg !368
@_ZL4arow = internal global i32* null, align 8, !dbg !370
@_ZL4acol = internal global i32* null, align 8, !dbg !372
@_ZL4aelt = internal global double* null, align 8, !dbg !374
@_ZL1a = internal global double* null, align 8, !dbg !376
@_ZL1x = internal global double* null, align 8, !dbg !378
@_ZL1z = internal global double* null, align 8, !dbg !380
@_ZL1p = internal global double* null, align 8, !dbg !382
@_ZL1q = internal global double* null, align 8, !dbg !384
@_ZL1r = internal global double* null, align 8, !dbg !386
@.str.39 = private unnamed_addr constant [54 x i8] c" Using dynamically allocated arrays (C-style malloc)\0A\00", align 1
@_ZL8firstrow = internal global i32 0, align 4, !dbg !388
@_ZL7lastrow = internal global i32 0, align 4, !dbg !390
@_ZL8firstcol = internal global i32 0, align 4, !dbg !392
@_ZL7lastcol = internal global i32 0, align 4, !dbg !394
@.str.40 = private unnamed_addr constant [65 x i8] c"\0A\0A NAS Parallel Benchmarks 4.1 CUDA C++ version - CG Benchmark\0A\0A\00", align 1
@.str.41 = private unnamed_addr constant [13 x i8] c" Size: %11d\0A\00", align 1
@.str.42 = private unnamed_addr constant [18 x i8] c" Iterations: %5d\0A\00", align 1
@_ZL3naa = internal global i32 0, align 4, !dbg !396
@_ZL3nzz = internal global i32 0, align 4, !dbg !398
@_ZL4tran = internal global double 0.000000e+00, align 8, !dbg !400
@_ZL5amult = internal global double 0.000000e+00, align 8, !dbg !402
@.str.43 = private unnamed_addr constant [52 x i8] c"\0A   iteration           ||r||                 zeta\0A\00", align 1
@.str.44 = private unnamed_addr constant [30 x i8] c"    %5d       %20.14e%20.13e\0A\00", align 1
@.str.45 = private unnamed_addr constant [22 x i8] c" Benchmark completed\0A\00", align 1
@.str.46 = private unnamed_addr constant [26 x i8] c" VERIFICATION SUCCESSFUL\0A\00", align 1
@.str.47 = private unnamed_addr constant [21 x i8] c" Zeta is    %20.13e\0A\00", align 1
@.str.48 = private unnamed_addr constant [21 x i8] c" Error is   %20.13e\0A\00", align 1
@.str.49 = private unnamed_addr constant [22 x i8] c" VERIFICATION FAILED\0A\00", align 1
@.str.50 = private unnamed_addr constant [30 x i8] c" Zeta                %20.13e\0A\00", align 1
@.str.51 = private unnamed_addr constant [30 x i8] c" The correct zeta is %20.13e\0A\00", align 1
@.str.52 = private unnamed_addr constant [23 x i8] c" Problem size unknown\0A\00", align 1
@.str.53 = private unnamed_addr constant [28 x i8] c" NO VERIFICATION PERFORMED\0A\00", align 1
@.str.54 = private unnamed_addr constant [10 x i8] c"%5s\09%25s\0A\00", align 1
@.str.55 = private unnamed_addr constant [11 x i8] c"GPU Kernel\00", align 1
@.str.56 = private unnamed_addr constant [18 x i8] c"Threads Per Block\00", align 1
@.str.57 = private unnamed_addr constant [11 x i8] c"%29s\09%25d\0A\00", align 1
@.str.58 = private unnamed_addr constant [5 x i8] c" one\00", align 1
@.str.59 = private unnamed_addr constant [5 x i8] c" two\00", align 1
@.str.60 = private unnamed_addr constant [7 x i8] c" three\00", align 1
@.str.61 = private unnamed_addr constant [6 x i8] c" four\00", align 1
@.str.62 = private unnamed_addr constant [6 x i8] c" five\00", align 1
@.str.63 = private unnamed_addr constant [5 x i8] c" six\00", align 1
@.str.64 = private unnamed_addr constant [7 x i8] c" seven\00", align 1
@.str.65 = private unnamed_addr constant [7 x i8] c" eight\00", align 1
@.str.66 = private unnamed_addr constant [6 x i8] c" nine\00", align 1
@.str.67 = private unnamed_addr constant [5 x i8] c" ten\00", align 1
@.str.68 = private unnamed_addr constant [8 x i8] c" eleven\00", align 1
@.str.69 = private unnamed_addr constant [3 x i8] c"CG\00", align 1
@.str.70 = private unnamed_addr constant [25 x i8] c"          floating point\00", align 1
@.str.71 = private unnamed_addr constant [4 x i8] c"4.1\00", align 1
@.str.72 = private unnamed_addr constant [12 x i8] c"01 Feb 2026\00", align 1
@.str.73 = private unnamed_addr constant [6 x i8] c"^\E6Q\FC\7F\00", align 1
@.str.74 = private unnamed_addr constant [42 x i8] c"Intel(R) Xeon(R) CPU E5-2697 v3 @ 2.60GHz\00", align 1
@.str.75 = private unnamed_addr constant [23 x i8] c"${NVCC} ${EXTRA_STUFF}\00", align 1
@.str.76 = private unnamed_addr constant [6 x i8] c"$(CC)\00", align 1
@.str.77 = private unnamed_addr constant [5 x i8] c"-lm \00", align 1
@.str.78 = private unnamed_addr constant [13 x i8] c"-I../common \00", align 1
@.str.79 = private unnamed_addr constant [4 x i8] c"-O3\00", align 1
@.str.80 = private unnamed_addr constant [7 x i8] c"randdp\00", align 1
@.str.81 = private unnamed_addr constant [46 x i8] c"Space for matrix elements exceeded in sparse\0A\00", align 1
@.str.82 = private unnamed_addr constant [21 x i8] c"nza, nzmax = %d, %d\0A\00", align 1

; Function Attrs: noinline nounwind uwtable
define dso_local double @_Z6randlcPdd(double* %x, double %a) #0 !dbg !1160 {
entry:
  %x.addr = alloca double*, align 8
  %a.addr = alloca double, align 8
  %t1 = alloca double, align 8
  %t2 = alloca double, align 8
  %t3 = alloca double, align 8
  %t4 = alloca double, align 8
  %a1 = alloca double, align 8
  %a2 = alloca double, align 8
  %x1 = alloca double, align 8
  %x2 = alloca double, align 8
  %z = alloca double, align 8
  store double* %x, double** %x.addr, align 8
  call void @llvm.dbg.declare(metadata double** %x.addr, metadata !1164, metadata !DIExpression()), !dbg !1165
  store double %a, double* %a.addr, align 8
  call void @llvm.dbg.declare(metadata double* %a.addr, metadata !1166, metadata !DIExpression()), !dbg !1167
  call void @llvm.dbg.declare(metadata double* %t1, metadata !1168, metadata !DIExpression()), !dbg !1169
  call void @llvm.dbg.declare(metadata double* %t2, metadata !1170, metadata !DIExpression()), !dbg !1171
  call void @llvm.dbg.declare(metadata double* %t3, metadata !1172, metadata !DIExpression()), !dbg !1173
  call void @llvm.dbg.declare(metadata double* %t4, metadata !1174, metadata !DIExpression()), !dbg !1175
  call void @llvm.dbg.declare(metadata double* %a1, metadata !1176, metadata !DIExpression()), !dbg !1177
  call void @llvm.dbg.declare(metadata double* %a2, metadata !1178, metadata !DIExpression()), !dbg !1179
  call void @llvm.dbg.declare(metadata double* %x1, metadata !1180, metadata !DIExpression()), !dbg !1181
  call void @llvm.dbg.declare(metadata double* %x2, metadata !1182, metadata !DIExpression()), !dbg !1183
  call void @llvm.dbg.declare(metadata double* %z, metadata !1184, metadata !DIExpression()), !dbg !1185
  %0 = load double, double* %a.addr, align 8, !dbg !1186
  %mul = fmul contract double 0x3E80000000000000, %0, !dbg !1187
  store double %mul, double* %t1, align 8, !dbg !1188
  %1 = load double, double* %t1, align 8, !dbg !1189
  %conv = fptosi double %1 to i32, !dbg !1189
  %conv1 = sitofp i32 %conv to double, !dbg !1190
  store double %conv1, double* %a1, align 8, !dbg !1191
  %2 = load double, double* %a.addr, align 8, !dbg !1192
  %3 = load double, double* %a1, align 8, !dbg !1193
  %mul2 = fmul contract double 0x4160000000000000, %3, !dbg !1194
  %sub = fsub contract double %2, %mul2, !dbg !1195
  store double %sub, double* %a2, align 8, !dbg !1196
  %4 = load double*, double** %x.addr, align 8, !dbg !1197
  %5 = load double, double* %4, align 8, !dbg !1198
  %mul3 = fmul contract double 0x3E80000000000000, %5, !dbg !1199
  store double %mul3, double* %t1, align 8, !dbg !1200
  %6 = load double, double* %t1, align 8, !dbg !1201
  %conv4 = fptosi double %6 to i32, !dbg !1201
  %conv5 = sitofp i32 %conv4 to double, !dbg !1202
  store double %conv5, double* %x1, align 8, !dbg !1203
  %7 = load double*, double** %x.addr, align 8, !dbg !1204
  %8 = load double, double* %7, align 8, !dbg !1205
  %9 = load double, double* %x1, align 8, !dbg !1206
  %mul6 = fmul contract double 0x4160000000000000, %9, !dbg !1207
  %sub7 = fsub contract double %8, %mul6, !dbg !1208
  store double %sub7, double* %x2, align 8, !dbg !1209
  %10 = load double, double* %a1, align 8, !dbg !1210
  %11 = load double, double* %x2, align 8, !dbg !1211
  %mul8 = fmul contract double %10, %11, !dbg !1212
  %12 = load double, double* %a2, align 8, !dbg !1213
  %13 = load double, double* %x1, align 8, !dbg !1214
  %mul9 = fmul contract double %12, %13, !dbg !1215
  %add = fadd contract double %mul8, %mul9, !dbg !1216
  store double %add, double* %t1, align 8, !dbg !1217
  %14 = load double, double* %t1, align 8, !dbg !1218
  %mul10 = fmul contract double 0x3E80000000000000, %14, !dbg !1219
  %conv11 = fptosi double %mul10 to i32, !dbg !1220
  %conv12 = sitofp i32 %conv11 to double, !dbg !1221
  store double %conv12, double* %t2, align 8, !dbg !1222
  %15 = load double, double* %t1, align 8, !dbg !1223
  %16 = load double, double* %t2, align 8, !dbg !1224
  %mul13 = fmul contract double 0x4160000000000000, %16, !dbg !1225
  %sub14 = fsub contract double %15, %mul13, !dbg !1226
  store double %sub14, double* %z, align 8, !dbg !1227
  %17 = load double, double* %z, align 8, !dbg !1228
  %mul15 = fmul contract double 0x4160000000000000, %17, !dbg !1229
  %18 = load double, double* %a2, align 8, !dbg !1230
  %19 = load double, double* %x2, align 8, !dbg !1231
  %mul16 = fmul contract double %18, %19, !dbg !1232
  %add17 = fadd contract double %mul15, %mul16, !dbg !1233
  store double %add17, double* %t3, align 8, !dbg !1234
  %20 = load double, double* %t3, align 8, !dbg !1235
  %mul18 = fmul contract double 0x3D10000000000000, %20, !dbg !1236
  %conv19 = fptosi double %mul18 to i32, !dbg !1237
  %conv20 = sitofp i32 %conv19 to double, !dbg !1238
  store double %conv20, double* %t4, align 8, !dbg !1239
  %21 = load double, double* %t3, align 8, !dbg !1240
  %22 = load double, double* %t4, align 8, !dbg !1241
  %mul21 = fmul contract double 0x42D0000000000000, %22, !dbg !1242
  %sub22 = fsub contract double %21, %mul21, !dbg !1243
  %23 = load double*, double** %x.addr, align 8, !dbg !1244
  store double %sub22, double* %23, align 8, !dbg !1245
  %24 = load double*, double** %x.addr, align 8, !dbg !1246
  %25 = load double, double* %24, align 8, !dbg !1247
  %mul23 = fmul contract double 0x3D10000000000000, %25, !dbg !1248
  ret double %mul23, !dbg !1249
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: noinline uwtable
define dso_local void @_Z15c_print_resultsPcciiiiddS_iS_S_S_S_S_S_S_S_S_S_S_S_S_S_(i8* %name, i8 signext %class_npb, i32 %n1, i32 %n2, i32 %n3, i32 %niter, double %t, double %mops, i8* %optype, i32 %passed_verification, i8* %npbversion, i8* %compiletime, i8* %compilerversion, i8* %libversion, i8* %cpu_device, i8* %gpu_device, i8* %gpu_config, i8* %cc, i8* %clink, i8* %c_lib, i8* %c_inc, i8* %cflags, i8* %clinkflags, i8* %rand) #2 !dbg !1250 {
entry:
  %name.addr = alloca i8*, align 8
  %class_npb.addr = alloca i8, align 1
  %n1.addr = alloca i32, align 4
  %n2.addr = alloca i32, align 4
  %n3.addr = alloca i32, align 4
  %niter.addr = alloca i32, align 4
  %t.addr = alloca double, align 8
  %mops.addr = alloca double, align 8
  %optype.addr = alloca i8*, align 8
  %passed_verification.addr = alloca i32, align 4
  %npbversion.addr = alloca i8*, align 8
  %compiletime.addr = alloca i8*, align 8
  %compilerversion.addr = alloca i8*, align 8
  %libversion.addr = alloca i8*, align 8
  %cpu_device.addr = alloca i8*, align 8
  %gpu_device.addr = alloca i8*, align 8
  %gpu_config.addr = alloca i8*, align 8
  %cc.addr = alloca i8*, align 8
  %clink.addr = alloca i8*, align 8
  %c_lib.addr = alloca i8*, align 8
  %c_inc.addr = alloca i8*, align 8
  %cflags.addr = alloca i8*, align 8
  %clinkflags.addr = alloca i8*, align 8
  %rand.addr = alloca i8*, align 8
  %nn = alloca i64, align 8
  %size = alloca [16 x i8], align 16
  %j = alloca i32, align 4
  store i8* %name, i8** %name.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %name.addr, metadata !1253, metadata !DIExpression()), !dbg !1254
  store i8 %class_npb, i8* %class_npb.addr, align 1
  call void @llvm.dbg.declare(metadata i8* %class_npb.addr, metadata !1255, metadata !DIExpression()), !dbg !1256
  store i32 %n1, i32* %n1.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %n1.addr, metadata !1257, metadata !DIExpression()), !dbg !1258
  store i32 %n2, i32* %n2.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %n2.addr, metadata !1259, metadata !DIExpression()), !dbg !1260
  store i32 %n3, i32* %n3.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %n3.addr, metadata !1261, metadata !DIExpression()), !dbg !1262
  store i32 %niter, i32* %niter.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %niter.addr, metadata !1263, metadata !DIExpression()), !dbg !1264
  store double %t, double* %t.addr, align 8
  call void @llvm.dbg.declare(metadata double* %t.addr, metadata !1265, metadata !DIExpression()), !dbg !1266
  store double %mops, double* %mops.addr, align 8
  call void @llvm.dbg.declare(metadata double* %mops.addr, metadata !1267, metadata !DIExpression()), !dbg !1268
  store i8* %optype, i8** %optype.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %optype.addr, metadata !1269, metadata !DIExpression()), !dbg !1270
  store i32 %passed_verification, i32* %passed_verification.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %passed_verification.addr, metadata !1271, metadata !DIExpression()), !dbg !1272
  store i8* %npbversion, i8** %npbversion.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %npbversion.addr, metadata !1273, metadata !DIExpression()), !dbg !1274
  store i8* %compiletime, i8** %compiletime.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %compiletime.addr, metadata !1275, metadata !DIExpression()), !dbg !1276
  store i8* %compilerversion, i8** %compilerversion.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %compilerversion.addr, metadata !1277, metadata !DIExpression()), !dbg !1278
  store i8* %libversion, i8** %libversion.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %libversion.addr, metadata !1279, metadata !DIExpression()), !dbg !1280
  store i8* %cpu_device, i8** %cpu_device.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %cpu_device.addr, metadata !1281, metadata !DIExpression()), !dbg !1282
  store i8* %gpu_device, i8** %gpu_device.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %gpu_device.addr, metadata !1283, metadata !DIExpression()), !dbg !1284
  store i8* %gpu_config, i8** %gpu_config.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %gpu_config.addr, metadata !1285, metadata !DIExpression()), !dbg !1286
  store i8* %cc, i8** %cc.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %cc.addr, metadata !1287, metadata !DIExpression()), !dbg !1288
  store i8* %clink, i8** %clink.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %clink.addr, metadata !1289, metadata !DIExpression()), !dbg !1290
  store i8* %c_lib, i8** %c_lib.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %c_lib.addr, metadata !1291, metadata !DIExpression()), !dbg !1292
  store i8* %c_inc, i8** %c_inc.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %c_inc.addr, metadata !1293, metadata !DIExpression()), !dbg !1294
  store i8* %cflags, i8** %cflags.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %cflags.addr, metadata !1295, metadata !DIExpression()), !dbg !1296
  store i8* %clinkflags, i8** %clinkflags.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %clinkflags.addr, metadata !1297, metadata !DIExpression()), !dbg !1298
  store i8* %rand, i8** %rand.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %rand.addr, metadata !1299, metadata !DIExpression()), !dbg !1300
  %0 = load i8*, i8** %name.addr, align 8, !dbg !1301
  %call = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([27 x i8], [27 x i8]* @.str, i64 0, i64 0), i8* %0), !dbg !1302
  %1 = load i8, i8* %class_npb.addr, align 1, !dbg !1303
  %conv = sext i8 %1 to i32, !dbg !1303
  %call1 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([46 x i8], [46 x i8]* @.str.1, i64 0, i64 0), i32 %conv), !dbg !1304
  %2 = load i8*, i8** %name.addr, align 8, !dbg !1305
  %arrayidx = getelementptr inbounds i8, i8* %2, i64 0, !dbg !1305
  %3 = load i8, i8* %arrayidx, align 1, !dbg !1305
  %conv2 = sext i8 %3 to i32, !dbg !1305
  %cmp = icmp eq i32 %conv2, 73, !dbg !1307
  br i1 %cmp, label %land.lhs.true, label %if.else15, !dbg !1308

land.lhs.true:                                    ; preds = %entry
  %4 = load i8*, i8** %name.addr, align 8, !dbg !1309
  %arrayidx3 = getelementptr inbounds i8, i8* %4, i64 1, !dbg !1309
  %5 = load i8, i8* %arrayidx3, align 1, !dbg !1309
  %conv4 = sext i8 %5 to i32, !dbg !1309
  %cmp5 = icmp eq i32 %conv4, 83, !dbg !1310
  br i1 %cmp5, label %if.then, label %if.else15, !dbg !1311

if.then:                                          ; preds = %land.lhs.true
  %6 = load i32, i32* %n3.addr, align 4, !dbg !1312
  %cmp6 = icmp eq i32 %6, 0, !dbg !1315
  br i1 %cmp6, label %if.then7, label %if.else, !dbg !1316

if.then7:                                         ; preds = %if.then
  call void @llvm.dbg.declare(metadata i64* %nn, metadata !1317, metadata !DIExpression()), !dbg !1319
  %7 = load i32, i32* %n1.addr, align 4, !dbg !1320
  %conv8 = sext i32 %7 to i64, !dbg !1320
  store i64 %conv8, i64* %nn, align 8, !dbg !1319
  %8 = load i32, i32* %n2.addr, align 4, !dbg !1321
  %cmp9 = icmp ne i32 %8, 0, !dbg !1323
  br i1 %cmp9, label %if.then10, label %if.end, !dbg !1324

if.then10:                                        ; preds = %if.then7
  %9 = load i32, i32* %n2.addr, align 4, !dbg !1325
  %conv11 = sext i32 %9 to i64, !dbg !1325
  %10 = load i64, i64* %nn, align 8, !dbg !1327
  %mul = mul nsw i64 %10, %conv11, !dbg !1327
  store i64 %mul, i64* %nn, align 8, !dbg !1327
  br label %if.end, !dbg !1328

if.end:                                           ; preds = %if.then10, %if.then7
  %11 = load i64, i64* %nn, align 8, !dbg !1329
  %call12 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([38 x i8], [38 x i8]* @.str.2, i64 0, i64 0), i64 %11), !dbg !1330
  br label %if.end14, !dbg !1331

if.else:                                          ; preds = %if.then
  %12 = load i32, i32* %n1.addr, align 4, !dbg !1332
  %13 = load i32, i32* %n2.addr, align 4, !dbg !1334
  %14 = load i32, i32* %n3.addr, align 4, !dbg !1335
  %call13 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([44 x i8], [44 x i8]* @.str.3, i64 0, i64 0), i32 %12, i32 %13, i32 %14), !dbg !1336
  br label %if.end14

if.end14:                                         ; preds = %if.else, %if.end
  br label %if.end48, !dbg !1337

if.else15:                                        ; preds = %land.lhs.true, %entry
  call void @llvm.dbg.declare(metadata [16 x i8]* %size, metadata !1338, metadata !DIExpression()), !dbg !1343
  call void @llvm.dbg.declare(metadata i32* %j, metadata !1344, metadata !DIExpression()), !dbg !1345
  %15 = load i32, i32* %n2.addr, align 4, !dbg !1346
  %cmp16 = icmp eq i32 %15, 0, !dbg !1348
  br i1 %cmp16, label %land.lhs.true17, label %if.else45, !dbg !1349

land.lhs.true17:                                  ; preds = %if.else15
  %16 = load i32, i32* %n3.addr, align 4, !dbg !1350
  %cmp18 = icmp eq i32 %16, 0, !dbg !1351
  br i1 %cmp18, label %if.then19, label %if.else45, !dbg !1352

if.then19:                                        ; preds = %land.lhs.true17
  %17 = load i8*, i8** %name.addr, align 8, !dbg !1353
  %arrayidx20 = getelementptr inbounds i8, i8* %17, i64 0, !dbg !1353
  %18 = load i8, i8* %arrayidx20, align 1, !dbg !1353
  %conv21 = sext i8 %18 to i32, !dbg !1353
  %cmp22 = icmp eq i32 %conv21, 69, !dbg !1356
  br i1 %cmp22, label %land.lhs.true23, label %if.else42, !dbg !1357

land.lhs.true23:                                  ; preds = %if.then19
  %19 = load i8*, i8** %name.addr, align 8, !dbg !1358
  %arrayidx24 = getelementptr inbounds i8, i8* %19, i64 1, !dbg !1358
  %20 = load i8, i8* %arrayidx24, align 1, !dbg !1358
  %conv25 = sext i8 %20 to i32, !dbg !1358
  %cmp26 = icmp eq i32 %conv25, 80, !dbg !1359
  br i1 %cmp26, label %if.then27, label %if.else42, !dbg !1360

if.then27:                                        ; preds = %land.lhs.true23
  %arraydecay = getelementptr inbounds [16 x i8], [16 x i8]* %size, i64 0, i64 0, !dbg !1361
  %21 = load i32, i32* %n1.addr, align 4, !dbg !1363
  %conv28 = sitofp i32 %21 to double, !dbg !1363
  %call29 = call double @pow(double 2.000000e+00, double %conv28) #8, !dbg !1364
  %call30 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay, i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.4, i64 0, i64 0), double %call29) #8, !dbg !1365
  store i32 14, i32* %j, align 4, !dbg !1366
  %22 = load i32, i32* %j, align 4, !dbg !1367
  %idxprom = sext i32 %22 to i64, !dbg !1369
  %arrayidx31 = getelementptr inbounds [16 x i8], [16 x i8]* %size, i64 0, i64 %idxprom, !dbg !1369
  %23 = load i8, i8* %arrayidx31, align 1, !dbg !1369
  %conv32 = sext i8 %23 to i32, !dbg !1369
  %cmp33 = icmp eq i32 %conv32, 46, !dbg !1370
  br i1 %cmp33, label %if.then34, label %if.end37, !dbg !1371

if.then34:                                        ; preds = %if.then27
  %24 = load i32, i32* %j, align 4, !dbg !1372
  %idxprom35 = sext i32 %24 to i64, !dbg !1374
  %arrayidx36 = getelementptr inbounds [16 x i8], [16 x i8]* %size, i64 0, i64 %idxprom35, !dbg !1374
  store i8 32, i8* %arrayidx36, align 1, !dbg !1375
  %25 = load i32, i32* %j, align 4, !dbg !1376
  %dec = add nsw i32 %25, -1, !dbg !1376
  store i32 %dec, i32* %j, align 4, !dbg !1376
  br label %if.end37, !dbg !1377

if.end37:                                         ; preds = %if.then34, %if.then27
  %26 = load i32, i32* %j, align 4, !dbg !1378
  %add = add nsw i32 %26, 1, !dbg !1379
  %idxprom38 = sext i32 %add to i64, !dbg !1380
  %arrayidx39 = getelementptr inbounds [16 x i8], [16 x i8]* %size, i64 0, i64 %idxprom38, !dbg !1380
  store i8 0, i8* %arrayidx39, align 1, !dbg !1381
  %arraydecay40 = getelementptr inbounds [16 x i8], [16 x i8]* %size, i64 0, i64 0, !dbg !1382
  %call41 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([34 x i8], [34 x i8]* @.str.5, i64 0, i64 0), i8* %arraydecay40), !dbg !1383
  br label %if.end44, !dbg !1384

if.else42:                                        ; preds = %land.lhs.true23, %if.then19
  %27 = load i32, i32* %n1.addr, align 4, !dbg !1385
  %call43 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.6, i64 0, i64 0), i32 %27), !dbg !1387
  br label %if.end44

if.end44:                                         ; preds = %if.else42, %if.end37
  br label %if.end47, !dbg !1388

if.else45:                                        ; preds = %land.lhs.true17, %if.else15
  %28 = load i32, i32* %n1.addr, align 4, !dbg !1389
  %29 = load i32, i32* %n2.addr, align 4, !dbg !1391
  %30 = load i32, i32* %n3.addr, align 4, !dbg !1392
  %call46 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([42 x i8], [42 x i8]* @.str.7, i64 0, i64 0), i32 %28, i32 %29, i32 %30), !dbg !1393
  br label %if.end47

if.end47:                                         ; preds = %if.else45, %if.end44
  br label %if.end48

if.end48:                                         ; preds = %if.end47, %if.end14
  %31 = load i32, i32* %niter.addr, align 4, !dbg !1394
  %call49 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.8, i64 0, i64 0), i32 %31), !dbg !1395
  %32 = load double, double* %t.addr, align 8, !dbg !1396
  %call50 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([39 x i8], [39 x i8]* @.str.9, i64 0, i64 0), double %32), !dbg !1397
  %33 = load double, double* %mops.addr, align 8, !dbg !1398
  %call51 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([39 x i8], [39 x i8]* @.str.10, i64 0, i64 0), double %33), !dbg !1399
  %34 = load i8*, i8** %optype.addr, align 8, !dbg !1400
  %call52 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([25 x i8], [25 x i8]* @.str.11, i64 0, i64 0), i8* %34), !dbg !1401
  %35 = load i32, i32* %passed_verification.addr, align 4, !dbg !1402
  %cmp53 = icmp slt i32 %35, 0, !dbg !1404
  br i1 %cmp53, label %if.then54, label %if.else56, !dbg !1405

if.then54:                                        ; preds = %if.end48
  %call55 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([45 x i8], [45 x i8]* @.str.12, i64 0, i64 0)), !dbg !1406
  br label %if.end62, !dbg !1408

if.else56:                                        ; preds = %if.end48
  %36 = load i32, i32* %passed_verification.addr, align 4, !dbg !1409
  %tobool = icmp ne i32 %36, 0, !dbg !1409
  br i1 %tobool, label %if.then57, label %if.else59, !dbg !1411

if.then57:                                        ; preds = %if.else56
  %call58 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([45 x i8], [45 x i8]* @.str.13, i64 0, i64 0)), !dbg !1412
  br label %if.end61, !dbg !1414

if.else59:                                        ; preds = %if.else56
  %call60 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([45 x i8], [45 x i8]* @.str.14, i64 0, i64 0)), !dbg !1415
  br label %if.end61

if.end61:                                         ; preds = %if.else59, %if.then57
  br label %if.end62

if.end62:                                         ; preds = %if.end61, %if.then54
  %37 = load i8*, i8** %npbversion.addr, align 8, !dbg !1417
  %call63 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.15, i64 0, i64 0), i8* %37), !dbg !1418
  %38 = load i8*, i8** %compiletime.addr, align 8, !dbg !1419
  %call64 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.16, i64 0, i64 0), i8* %38), !dbg !1420
  %39 = load i8*, i8** %compilerversion.addr, align 8, !dbg !1421
  %call65 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.17, i64 0, i64 0), i8* %39), !dbg !1422
  %40 = load i8*, i8** %libversion.addr, align 8, !dbg !1423
  %call66 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.18, i64 0, i64 0), i8* %40), !dbg !1424
  %call67 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([20 x i8], [20 x i8]* @.str.19, i64 0, i64 0)), !dbg !1425
  %41 = load i8*, i8** %cc.addr, align 8, !dbg !1426
  %call68 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.20, i64 0, i64 0), i8* %41), !dbg !1427
  %42 = load i8*, i8** %clink.addr, align 8, !dbg !1428
  %call69 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.21, i64 0, i64 0), i8* %42), !dbg !1429
  %43 = load i8*, i8** %c_lib.addr, align 8, !dbg !1430
  %call70 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.22, i64 0, i64 0), i8* %43), !dbg !1431
  %44 = load i8*, i8** %c_inc.addr, align 8, !dbg !1432
  %call71 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.23, i64 0, i64 0), i8* %44), !dbg !1433
  %45 = load i8*, i8** %cflags.addr, align 8, !dbg !1434
  %call72 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.24, i64 0, i64 0), i8* %45), !dbg !1435
  %46 = load i8*, i8** %clinkflags.addr, align 8, !dbg !1436
  %call73 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.25, i64 0, i64 0), i8* %46), !dbg !1437
  %47 = load i8*, i8** %rand.addr, align 8, !dbg !1438
  %call74 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.26, i64 0, i64 0), i8* %47), !dbg !1439
  %call75 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.27, i64 0, i64 0)), !dbg !1440
  %48 = load i8*, i8** %cpu_device.addr, align 8, !dbg !1441
  %call76 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.28, i64 0, i64 0), i8* %48), !dbg !1442
  %49 = load i8*, i8** %gpu_device.addr, align 8, !dbg !1443
  %call77 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.29, i64 0, i64 0), i8* %49), !dbg !1444
  %call78 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.30, i64 0, i64 0)), !dbg !1445
  %50 = load i8*, i8** %gpu_config.addr, align 8, !dbg !1446
  %call79 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.31, i64 0, i64 0), i8* %50), !dbg !1447
  %call80 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.32, i64 0, i64 0)), !dbg !1448
  %call81 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([72 x i8], [72 x i8]* @.str.33, i64 0, i64 0)), !dbg !1449
  %call82 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([27 x i8], [27 x i8]* @.str.34, i64 0, i64 0)), !dbg !1450
  %call83 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([56 x i8], [56 x i8]* @.str.35, i64 0, i64 0)), !dbg !1451
  %call84 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([52 x i8], [52 x i8]* @.str.36, i64 0, i64 0)), !dbg !1452
  %call85 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([46 x i8], [46 x i8]* @.str.37, i64 0, i64 0)), !dbg !1453
  %call86 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.32, i64 0, i64 0)), !dbg !1454
  %call87 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([43 x i8], [43 x i8]* @.str.38, i64 0, i64 0)), !dbg !1455
  %call88 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([72 x i8], [72 x i8]* @.str.33, i64 0, i64 0)), !dbg !1456
  %call89 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.32, i64 0, i64 0)), !dbg !1457
  ret void, !dbg !1458
}

declare dso_local i32 @printf(i8*, ...) #3

; Function Attrs: nounwind
declare dso_local i32 @sprintf(i8*, i8*, ...) #4

; Function Attrs: nounwind
declare dso_local double @pow(double, double) #4

; Function Attrs: noinline norecurse uwtable
define dso_local i32 @main(i32 %argc, i8** %argv) #5 !dbg !1459 {
entry:
  %retval = alloca i32, align 4
  %argc.addr = alloca i32, align 4
  %argv.addr = alloca i8**, align 8
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  %k = alloca i32, align 4
  %it = alloca i32, align 4
  %zeta = alloca double, align 8
  %rnorm = alloca double, align 8
  %norm_temp1 = alloca double, align 8
  %norm_temp2 = alloca double, align 8
  %t = alloca double, align 8
  %mflops = alloca double, align 8
  %class_npb = alloca i8, align 1
  %verified = alloca i32, align 4
  %zeta_verify_value = alloca double, align 8
  %epsilon = alloca double, align 8
  %err = alloca double, align 8
  %gpu_config = alloca [256 x i8], align 16
  %gpu_config_string = alloca [2048 x i8], align 16
  store i32 0, i32* %retval, align 4
  store i32 %argc, i32* %argc.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %argc.addr, metadata !1462, metadata !DIExpression()), !dbg !1463
  store i8** %argv, i8*** %argv.addr, align 8
  call void @llvm.dbg.declare(metadata i8*** %argv.addr, metadata !1464, metadata !DIExpression()), !dbg !1465
  %call = call noalias i8* @malloc(i64 8064000) #8, !dbg !1466
  %0 = bitcast i8* %call to i32*, !dbg !1467
  store i32* %0, i32** @_ZL6colidx, align 8, !dbg !1468
  %call1 = call noalias i8* @malloc(i64 56004) #8, !dbg !1469
  %1 = bitcast i8* %call1 to i32*, !dbg !1470
  store i32* %1, i32** @_ZL6rowstr, align 8, !dbg !1471
  %call2 = call noalias i8* @malloc(i64 56000) #8, !dbg !1472
  %2 = bitcast i8* %call2 to i32*, !dbg !1473
  store i32* %2, i32** @_ZL2iv, align 8, !dbg !1474
  %call3 = call noalias i8* @malloc(i64 56000) #8, !dbg !1475
  %3 = bitcast i8* %call3 to i32*, !dbg !1476
  store i32* %3, i32** @_ZL4arow, align 8, !dbg !1477
  %call4 = call noalias i8* @malloc(i64 672000) #8, !dbg !1478
  %4 = bitcast i8* %call4 to i32*, !dbg !1479
  store i32* %4, i32** @_ZL4acol, align 8, !dbg !1480
  %call5 = call noalias i8* @malloc(i64 1344000) #8, !dbg !1481
  %5 = bitcast i8* %call5 to double*, !dbg !1482
  store double* %5, double** @_ZL4aelt, align 8, !dbg !1483
  %call6 = call noalias i8* @malloc(i64 16128000) #8, !dbg !1484
  %6 = bitcast i8* %call6 to double*, !dbg !1485
  store double* %6, double** @_ZL1a, align 8, !dbg !1486
  %call7 = call noalias i8* @malloc(i64 112016) #8, !dbg !1487
  %7 = bitcast i8* %call7 to double*, !dbg !1488
  store double* %7, double** @_ZL1x, align 8, !dbg !1489
  %call8 = call noalias i8* @malloc(i64 112016) #8, !dbg !1490
  %8 = bitcast i8* %call8 to double*, !dbg !1491
  store double* %8, double** @_ZL1z, align 8, !dbg !1492
  %call9 = call noalias i8* @malloc(i64 112016) #8, !dbg !1493
  %9 = bitcast i8* %call9 to double*, !dbg !1494
  store double* %9, double** @_ZL1p, align 8, !dbg !1495
  %call10 = call noalias i8* @malloc(i64 112016) #8, !dbg !1496
  %10 = bitcast i8* %call10 to double*, !dbg !1497
  store double* %10, double** @_ZL1q, align 8, !dbg !1498
  %call11 = call noalias i8* @malloc(i64 112016) #8, !dbg !1499
  %11 = bitcast i8* %call11 to double*, !dbg !1500
  store double* %11, double** @_ZL1r, align 8, !dbg !1501
  %call12 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([54 x i8], [54 x i8]* @.str.39, i64 0, i64 0)), !dbg !1502
  call void @llvm.dbg.declare(metadata i32* %i, metadata !1503, metadata !DIExpression()), !dbg !1504
  call void @llvm.dbg.declare(metadata i32* %j, metadata !1505, metadata !DIExpression()), !dbg !1506
  call void @llvm.dbg.declare(metadata i32* %k, metadata !1507, metadata !DIExpression()), !dbg !1508
  call void @llvm.dbg.declare(metadata i32* %it, metadata !1509, metadata !DIExpression()), !dbg !1510
  call void @llvm.dbg.declare(metadata double* %zeta, metadata !1511, metadata !DIExpression()), !dbg !1512
  call void @llvm.dbg.declare(metadata double* %rnorm, metadata !1513, metadata !DIExpression()), !dbg !1514
  call void @llvm.dbg.declare(metadata double* %norm_temp1, metadata !1515, metadata !DIExpression()), !dbg !1516
  call void @llvm.dbg.declare(metadata double* %norm_temp2, metadata !1517, metadata !DIExpression()), !dbg !1518
  call void @llvm.dbg.declare(metadata double* %t, metadata !1519, metadata !DIExpression()), !dbg !1520
  call void @llvm.dbg.declare(metadata double* %mflops, metadata !1521, metadata !DIExpression()), !dbg !1522
  call void @llvm.dbg.declare(metadata i8* %class_npb, metadata !1523, metadata !DIExpression()), !dbg !1524
  call void @llvm.dbg.declare(metadata i32* %verified, metadata !1525, metadata !DIExpression()), !dbg !1528
  call void @llvm.dbg.declare(metadata double* %zeta_verify_value, metadata !1529, metadata !DIExpression()), !dbg !1530
  call void @llvm.dbg.declare(metadata double* %epsilon, metadata !1531, metadata !DIExpression()), !dbg !1532
  call void @llvm.dbg.declare(metadata double* %err, metadata !1533, metadata !DIExpression()), !dbg !1534
  store i32 0, i32* @_ZL8firstrow, align 4, !dbg !1535
  store i32 13999, i32* @_ZL7lastrow, align 4, !dbg !1536
  store i32 0, i32* @_ZL8firstcol, align 4, !dbg !1537
  store i32 13999, i32* @_ZL7lastcol, align 4, !dbg !1538
  store i8 65, i8* %class_npb, align 1, !dbg !1539
  store double 0x4031215715A1D8EC, double* %zeta_verify_value, align 8, !dbg !1544
  %call13 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([65 x i8], [65 x i8]* @.str.40, i64 0, i64 0)), !dbg !1545
  %call14 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.41, i64 0, i64 0), i32 14000), !dbg !1546
  %call15 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([18 x i8], [18 x i8]* @.str.42, i64 0, i64 0), i32 15), !dbg !1547
  store i32 14000, i32* @_ZL3naa, align 4, !dbg !1548
  store i32 2016000, i32* @_ZL3nzz, align 4, !dbg !1549
  store double 0x41B2B9B0A1000000, double* @_ZL4tran, align 8, !dbg !1550
  store double 0x41D2309CE5400000, double* @_ZL5amult, align 8, !dbg !1551
  %12 = load double, double* @_ZL5amult, align 8, !dbg !1552
  %call16 = call double @_Z6randlcPdd(double* @_ZL4tran, double %12), !dbg !1553
  store double %call16, double* %zeta, align 8, !dbg !1554
  %13 = load i32, i32* @_ZL3naa, align 4, !dbg !1555
  %14 = load i32, i32* @_ZL3nzz, align 4, !dbg !1556
  %15 = load double*, double** @_ZL1a, align 8, !dbg !1557
  %16 = load i32*, i32** @_ZL6colidx, align 8, !dbg !1558
  %17 = load i32*, i32** @_ZL6rowstr, align 8, !dbg !1559
  %18 = load i32, i32* @_ZL8firstrow, align 4, !dbg !1560
  %19 = load i32, i32* @_ZL7lastrow, align 4, !dbg !1561
  %20 = load i32, i32* @_ZL8firstcol, align 4, !dbg !1562
  %21 = load i32, i32* @_ZL7lastcol, align 4, !dbg !1563
  %22 = load i32*, i32** @_ZL4arow, align 8, !dbg !1564
  %23 = load i32*, i32** @_ZL4acol, align 8, !dbg !1565
  %24 = bitcast i32* %23 to i8*, !dbg !1565
  %25 = bitcast i8* %24 to [12 x i32]*, !dbg !1566
  %26 = load double*, double** @_ZL4aelt, align 8, !dbg !1567
  %27 = bitcast double* %26 to i8*, !dbg !1567
  %28 = bitcast i8* %27 to [12 x double]*, !dbg !1568
  %29 = load i32*, i32** @_ZL2iv, align 8, !dbg !1569
  call void @_ZL5makeaiiPdPiS0_iiiiS0_PA12_iPA12_dS0_(i32 %13, i32 %14, double* %15, i32* %16, i32* %17, i32 %18, i32 %19, i32 %20, i32 %21, i32* %22, [12 x i32]* %25, [12 x double]* %28, i32* %29), !dbg !1570
  store i32 0, i32* %j, align 4, !dbg !1571
  br label %for.cond, !dbg !1573

for.cond:                                         ; preds = %for.inc28, %entry
  %30 = load i32, i32* %j, align 4, !dbg !1574
  %31 = load i32, i32* @_ZL7lastrow, align 4, !dbg !1576
  %32 = load i32, i32* @_ZL8firstrow, align 4, !dbg !1577
  %sub = sub nsw i32 %31, %32, !dbg !1578
  %add = add nsw i32 %sub, 1, !dbg !1579
  %cmp = icmp slt i32 %30, %add, !dbg !1580
  br i1 %cmp, label %for.body, label %for.end30, !dbg !1581

for.body:                                         ; preds = %for.cond
  %33 = load i32*, i32** @_ZL6rowstr, align 8, !dbg !1582
  %34 = load i32, i32* %j, align 4, !dbg !1585
  %idxprom = sext i32 %34 to i64, !dbg !1582
  %arrayidx = getelementptr inbounds i32, i32* %33, i64 %idxprom, !dbg !1582
  %35 = load i32, i32* %arrayidx, align 4, !dbg !1582
  store i32 %35, i32* %k, align 4, !dbg !1586
  br label %for.cond17, !dbg !1587

for.cond17:                                       ; preds = %for.inc, %for.body
  %36 = load i32, i32* %k, align 4, !dbg !1588
  %37 = load i32*, i32** @_ZL6rowstr, align 8, !dbg !1590
  %38 = load i32, i32* %j, align 4, !dbg !1591
  %add18 = add nsw i32 %38, 1, !dbg !1592
  %idxprom19 = sext i32 %add18 to i64, !dbg !1590
  %arrayidx20 = getelementptr inbounds i32, i32* %37, i64 %idxprom19, !dbg !1590
  %39 = load i32, i32* %arrayidx20, align 4, !dbg !1590
  %cmp21 = icmp slt i32 %36, %39, !dbg !1593
  br i1 %cmp21, label %for.body22, label %for.end, !dbg !1594

for.body22:                                       ; preds = %for.cond17
  %40 = load i32*, i32** @_ZL6colidx, align 8, !dbg !1595
  %41 = load i32, i32* %k, align 4, !dbg !1597
  %idxprom23 = sext i32 %41 to i64, !dbg !1595
  %arrayidx24 = getelementptr inbounds i32, i32* %40, i64 %idxprom23, !dbg !1595
  %42 = load i32, i32* %arrayidx24, align 4, !dbg !1595
  %43 = load i32, i32* @_ZL8firstcol, align 4, !dbg !1598
  %sub25 = sub nsw i32 %42, %43, !dbg !1599
  %44 = load i32*, i32** @_ZL6colidx, align 8, !dbg !1600
  %45 = load i32, i32* %k, align 4, !dbg !1601
  %idxprom26 = sext i32 %45 to i64, !dbg !1600
  %arrayidx27 = getelementptr inbounds i32, i32* %44, i64 %idxprom26, !dbg !1600
  store i32 %sub25, i32* %arrayidx27, align 4, !dbg !1602
  br label %for.inc, !dbg !1603

for.inc:                                          ; preds = %for.body22
  %46 = load i32, i32* %k, align 4, !dbg !1604
  %inc = add nsw i32 %46, 1, !dbg !1604
  store i32 %inc, i32* %k, align 4, !dbg !1604
  br label %for.cond17, !dbg !1605, !llvm.loop !1606

for.end:                                          ; preds = %for.cond17
  br label %for.inc28, !dbg !1608

for.inc28:                                        ; preds = %for.end
  %47 = load i32, i32* %j, align 4, !dbg !1609
  %inc29 = add nsw i32 %47, 1, !dbg !1609
  store i32 %inc29, i32* %j, align 4, !dbg !1609
  br label %for.cond, !dbg !1610, !llvm.loop !1611

for.end30:                                        ; preds = %for.cond
  store i32 0, i32* %i, align 4, !dbg !1613
  br label %for.cond31, !dbg !1615

for.cond31:                                       ; preds = %for.inc36, %for.end30
  %48 = load i32, i32* %i, align 4, !dbg !1616
  %cmp32 = icmp slt i32 %48, 14001, !dbg !1618
  br i1 %cmp32, label %for.body33, label %for.end38, !dbg !1619

for.body33:                                       ; preds = %for.cond31
  %49 = load double*, double** @_ZL1x, align 8, !dbg !1620
  %50 = load i32, i32* %i, align 4, !dbg !1622
  %idxprom34 = sext i32 %50 to i64, !dbg !1620
  %arrayidx35 = getelementptr inbounds double, double* %49, i64 %idxprom34, !dbg !1620
  store double 1.000000e+00, double* %arrayidx35, align 8, !dbg !1623
  br label %for.inc36, !dbg !1624

for.inc36:                                        ; preds = %for.body33
  %51 = load i32, i32* %i, align 4, !dbg !1625
  %inc37 = add nsw i32 %51, 1, !dbg !1625
  store i32 %inc37, i32* %i, align 4, !dbg !1625
  br label %for.cond31, !dbg !1626, !llvm.loop !1627

for.end38:                                        ; preds = %for.cond31
  store i32 0, i32* %j, align 4, !dbg !1629
  br label %for.cond39, !dbg !1631

for.cond39:                                       ; preds = %for.inc52, %for.end38
  %52 = load i32, i32* %j, align 4, !dbg !1632
  %53 = load i32, i32* @_ZL7lastcol, align 4, !dbg !1634
  %54 = load i32, i32* @_ZL8firstcol, align 4, !dbg !1635
  %sub40 = sub nsw i32 %53, %54, !dbg !1636
  %add41 = add nsw i32 %sub40, 1, !dbg !1637
  %cmp42 = icmp slt i32 %52, %add41, !dbg !1638
  br i1 %cmp42, label %for.body43, label %for.end54, !dbg !1639

for.body43:                                       ; preds = %for.cond39
  %55 = load double*, double** @_ZL1q, align 8, !dbg !1640
  %56 = load i32, i32* %j, align 4, !dbg !1642
  %idxprom44 = sext i32 %56 to i64, !dbg !1640
  %arrayidx45 = getelementptr inbounds double, double* %55, i64 %idxprom44, !dbg !1640
  store double 0.000000e+00, double* %arrayidx45, align 8, !dbg !1643
  %57 = load double*, double** @_ZL1z, align 8, !dbg !1644
  %58 = load i32, i32* %j, align 4, !dbg !1645
  %idxprom46 = sext i32 %58 to i64, !dbg !1644
  %arrayidx47 = getelementptr inbounds double, double* %57, i64 %idxprom46, !dbg !1644
  store double 0.000000e+00, double* %arrayidx47, align 8, !dbg !1646
  %59 = load double*, double** @_ZL1r, align 8, !dbg !1647
  %60 = load i32, i32* %j, align 4, !dbg !1648
  %idxprom48 = sext i32 %60 to i64, !dbg !1647
  %arrayidx49 = getelementptr inbounds double, double* %59, i64 %idxprom48, !dbg !1647
  store double 0.000000e+00, double* %arrayidx49, align 8, !dbg !1649
  %61 = load double*, double** @_ZL1p, align 8, !dbg !1650
  %62 = load i32, i32* %j, align 4, !dbg !1651
  %idxprom50 = sext i32 %62 to i64, !dbg !1650
  %arrayidx51 = getelementptr inbounds double, double* %61, i64 %idxprom50, !dbg !1650
  store double 0.000000e+00, double* %arrayidx51, align 8, !dbg !1652
  br label %for.inc52, !dbg !1653

for.inc52:                                        ; preds = %for.body43
  %63 = load i32, i32* %j, align 4, !dbg !1654
  %inc53 = add nsw i32 %63, 1, !dbg !1654
  store i32 %inc53, i32* %j, align 4, !dbg !1654
  br label %for.cond39, !dbg !1655, !llvm.loop !1656

for.end54:                                        ; preds = %for.cond39
  store double 0.000000e+00, double* %zeta, align 8, !dbg !1658
  store i32 1, i32* %it, align 4, !dbg !1659
  br label %for.cond55, !dbg !1661

for.cond55:                                       ; preds = %for.inc91, %for.end54
  %64 = load i32, i32* %it, align 4, !dbg !1662
  %cmp56 = icmp sle i32 %64, 1, !dbg !1664
  br i1 %cmp56, label %for.body57, label %for.end93, !dbg !1665

for.body57:                                       ; preds = %for.cond55
  %65 = load i32*, i32** @_ZL6colidx, align 8, !dbg !1666
  %66 = load i32*, i32** @_ZL6rowstr, align 8, !dbg !1668
  %67 = load double*, double** @_ZL1x, align 8, !dbg !1669
  %68 = load double*, double** @_ZL1z, align 8, !dbg !1670
  %69 = load double*, double** @_ZL1a, align 8, !dbg !1671
  %70 = load double*, double** @_ZL1p, align 8, !dbg !1672
  %71 = load double*, double** @_ZL1q, align 8, !dbg !1673
  %72 = load double*, double** @_ZL1r, align 8, !dbg !1674
  call void @_ZL9conj_gradPiS_PdS0_S0_S0_S0_S0_S0_(i32* %65, i32* %66, double* %67, double* %68, double* %69, double* %70, double* %71, double* %72, double* %rnorm), !dbg !1675
  store double 0.000000e+00, double* %norm_temp1, align 8, !dbg !1676
  store double 0.000000e+00, double* %norm_temp2, align 8, !dbg !1677
  store i32 0, i32* %j, align 4, !dbg !1678
  br label %for.cond58, !dbg !1680

for.cond58:                                       ; preds = %for.inc74, %for.body57
  %73 = load i32, i32* %j, align 4, !dbg !1681
  %74 = load i32, i32* @_ZL7lastcol, align 4, !dbg !1683
  %75 = load i32, i32* @_ZL8firstcol, align 4, !dbg !1684
  %sub59 = sub nsw i32 %74, %75, !dbg !1685
  %add60 = add nsw i32 %sub59, 1, !dbg !1686
  %cmp61 = icmp slt i32 %73, %add60, !dbg !1687
  br i1 %cmp61, label %for.body62, label %for.end76, !dbg !1688

for.body62:                                       ; preds = %for.cond58
  %76 = load double, double* %norm_temp1, align 8, !dbg !1689
  %77 = load double*, double** @_ZL1x, align 8, !dbg !1691
  %78 = load i32, i32* %j, align 4, !dbg !1692
  %idxprom63 = sext i32 %78 to i64, !dbg !1691
  %arrayidx64 = getelementptr inbounds double, double* %77, i64 %idxprom63, !dbg !1691
  %79 = load double, double* %arrayidx64, align 8, !dbg !1691
  %80 = load double*, double** @_ZL1z, align 8, !dbg !1693
  %81 = load i32, i32* %j, align 4, !dbg !1694
  %idxprom65 = sext i32 %81 to i64, !dbg !1693
  %arrayidx66 = getelementptr inbounds double, double* %80, i64 %idxprom65, !dbg !1693
  %82 = load double, double* %arrayidx66, align 8, !dbg !1693
  %mul = fmul contract double %79, %82, !dbg !1695
  %add67 = fadd contract double %76, %mul, !dbg !1696
  store double %add67, double* %norm_temp1, align 8, !dbg !1697
  %83 = load double, double* %norm_temp2, align 8, !dbg !1698
  %84 = load double*, double** @_ZL1z, align 8, !dbg !1699
  %85 = load i32, i32* %j, align 4, !dbg !1700
  %idxprom68 = sext i32 %85 to i64, !dbg !1699
  %arrayidx69 = getelementptr inbounds double, double* %84, i64 %idxprom68, !dbg !1699
  %86 = load double, double* %arrayidx69, align 8, !dbg !1699
  %87 = load double*, double** @_ZL1z, align 8, !dbg !1701
  %88 = load i32, i32* %j, align 4, !dbg !1702
  %idxprom70 = sext i32 %88 to i64, !dbg !1701
  %arrayidx71 = getelementptr inbounds double, double* %87, i64 %idxprom70, !dbg !1701
  %89 = load double, double* %arrayidx71, align 8, !dbg !1701
  %mul72 = fmul contract double %86, %89, !dbg !1703
  %add73 = fadd contract double %83, %mul72, !dbg !1704
  store double %add73, double* %norm_temp2, align 8, !dbg !1705
  br label %for.inc74, !dbg !1706

for.inc74:                                        ; preds = %for.body62
  %90 = load i32, i32* %j, align 4, !dbg !1707
  %inc75 = add nsw i32 %90, 1, !dbg !1707
  store i32 %inc75, i32* %j, align 4, !dbg !1707
  br label %for.cond58, !dbg !1708, !llvm.loop !1709

for.end76:                                        ; preds = %for.cond58
  %91 = load double, double* %norm_temp2, align 8, !dbg !1711
  %call77 = call double @sqrt(double %91) #8, !dbg !1712
  %div = fdiv double 1.000000e+00, %call77, !dbg !1713
  store double %div, double* %norm_temp2, align 8, !dbg !1714
  store i32 0, i32* %j, align 4, !dbg !1715
  br label %for.cond78, !dbg !1717

for.cond78:                                       ; preds = %for.inc88, %for.end76
  %92 = load i32, i32* %j, align 4, !dbg !1718
  %93 = load i32, i32* @_ZL7lastcol, align 4, !dbg !1720
  %94 = load i32, i32* @_ZL8firstcol, align 4, !dbg !1721
  %sub79 = sub nsw i32 %93, %94, !dbg !1722
  %add80 = add nsw i32 %sub79, 1, !dbg !1723
  %cmp81 = icmp slt i32 %92, %add80, !dbg !1724
  br i1 %cmp81, label %for.body82, label %for.end90, !dbg !1725

for.body82:                                       ; preds = %for.cond78
  %95 = load double, double* %norm_temp2, align 8, !dbg !1726
  %96 = load double*, double** @_ZL1z, align 8, !dbg !1728
  %97 = load i32, i32* %j, align 4, !dbg !1729
  %idxprom83 = sext i32 %97 to i64, !dbg !1728
  %arrayidx84 = getelementptr inbounds double, double* %96, i64 %idxprom83, !dbg !1728
  %98 = load double, double* %arrayidx84, align 8, !dbg !1728
  %mul85 = fmul contract double %95, %98, !dbg !1730
  %99 = load double*, double** @_ZL1x, align 8, !dbg !1731
  %100 = load i32, i32* %j, align 4, !dbg !1732
  %idxprom86 = sext i32 %100 to i64, !dbg !1731
  %arrayidx87 = getelementptr inbounds double, double* %99, i64 %idxprom86, !dbg !1731
  store double %mul85, double* %arrayidx87, align 8, !dbg !1733
  br label %for.inc88, !dbg !1734

for.inc88:                                        ; preds = %for.body82
  %101 = load i32, i32* %j, align 4, !dbg !1735
  %inc89 = add nsw i32 %101, 1, !dbg !1735
  store i32 %inc89, i32* %j, align 4, !dbg !1735
  br label %for.cond78, !dbg !1736, !llvm.loop !1737

for.end90:                                        ; preds = %for.cond78
  br label %for.inc91, !dbg !1739

for.inc91:                                        ; preds = %for.end90
  %102 = load i32, i32* %it, align 4, !dbg !1740
  %inc92 = add nsw i32 %102, 1, !dbg !1740
  store i32 %inc92, i32* %it, align 4, !dbg !1740
  br label %for.cond55, !dbg !1741, !llvm.loop !1742

for.end93:                                        ; preds = %for.cond55
  store i32 0, i32* %i, align 4, !dbg !1744
  br label %for.cond94, !dbg !1746

for.cond94:                                       ; preds = %for.inc99, %for.end93
  %103 = load i32, i32* %i, align 4, !dbg !1747
  %cmp95 = icmp slt i32 %103, 14001, !dbg !1749
  br i1 %cmp95, label %for.body96, label %for.end101, !dbg !1750

for.body96:                                       ; preds = %for.cond94
  %104 = load double*, double** @_ZL1x, align 8, !dbg !1751
  %105 = load i32, i32* %i, align 4, !dbg !1753
  %idxprom97 = sext i32 %105 to i64, !dbg !1751
  %arrayidx98 = getelementptr inbounds double, double* %104, i64 %idxprom97, !dbg !1751
  store double 1.000000e+00, double* %arrayidx98, align 8, !dbg !1754
  br label %for.inc99, !dbg !1755

for.inc99:                                        ; preds = %for.body96
  %106 = load i32, i32* %i, align 4, !dbg !1756
  %inc100 = add nsw i32 %106, 1, !dbg !1756
  store i32 %inc100, i32* %i, align 4, !dbg !1756
  br label %for.cond94, !dbg !1757, !llvm.loop !1758

for.end101:                                       ; preds = %for.cond94
  store double 0.000000e+00, double* %zeta, align 8, !dbg !1760
  call void @_ZL9setup_gpuv(), !dbg !1761
  store i32 1, i32* %it, align 4, !dbg !1762
  br label %for.cond102, !dbg !1764

for.cond102:                                      ; preds = %for.inc112, %for.end101
  %107 = load i32, i32* %it, align 4, !dbg !1765
  %cmp103 = icmp sle i32 %107, 15, !dbg !1767
  br i1 %cmp103, label %for.body104, label %for.end114, !dbg !1768

for.body104:                                      ; preds = %for.cond102
  call void @_ZL13conj_grad_gpuPd(double* %rnorm), !dbg !1769
  call void @_ZL19gpu_kernel_ten_hostPdS_(double* %norm_temp1, double* %norm_temp2), !dbg !1771
  %108 = load double, double* %norm_temp2, align 8, !dbg !1772
  %call105 = call double @sqrt(double %108) #8, !dbg !1773
  %div106 = fdiv double 1.000000e+00, %call105, !dbg !1774
  store double %div106, double* %norm_temp2, align 8, !dbg !1775
  %109 = load double, double* %norm_temp1, align 8, !dbg !1776
  %div107 = fdiv double 1.000000e+00, %109, !dbg !1777
  %add108 = fadd contract double 2.000000e+01, %div107, !dbg !1778
  store double %add108, double* %zeta, align 8, !dbg !1779
  %110 = load i32, i32* %it, align 4, !dbg !1780
  %cmp109 = icmp eq i32 %110, 1, !dbg !1782
  br i1 %cmp109, label %if.then, label %if.end, !dbg !1783

if.then:                                          ; preds = %for.body104
  %call110 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([52 x i8], [52 x i8]* @.str.43, i64 0, i64 0)), !dbg !1784
  br label %if.end, !dbg !1786

if.end:                                           ; preds = %if.then, %for.body104
  %111 = load i32, i32* %it, align 4, !dbg !1787
  %112 = load double, double* %rnorm, align 8, !dbg !1788
  %113 = load double, double* %zeta, align 8, !dbg !1789
  %call111 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([30 x i8], [30 x i8]* @.str.44, i64 0, i64 0), i32 %111, double %112, double %113), !dbg !1790
  %114 = load double, double* %norm_temp2, align 8, !dbg !1791
  call void @_ZL22gpu_kernel_eleven_hostd(double %114), !dbg !1792
  br label %for.inc112, !dbg !1793

for.inc112:                                       ; preds = %if.end
  %115 = load i32, i32* %it, align 4, !dbg !1794
  %inc113 = add nsw i32 %115, 1, !dbg !1794
  store i32 %inc113, i32* %it, align 4, !dbg !1794
  br label %for.cond102, !dbg !1795, !llvm.loop !1796

for.end114:                                       ; preds = %for.cond102
  store double 0.000000e+00, double* %t, align 8, !dbg !1798
  %call115 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([22 x i8], [22 x i8]* @.str.45, i64 0, i64 0)), !dbg !1799
  store double 1.000000e-10, double* %epsilon, align 8, !dbg !1800
  %116 = load i8, i8* %class_npb, align 1, !dbg !1801
  %conv = sext i8 %116 to i32, !dbg !1801
  %cmp116 = icmp ne i32 %conv, 85, !dbg !1803
  br i1 %cmp116, label %if.then117, label %if.else129, !dbg !1804

if.then117:                                       ; preds = %for.end114
  %117 = load double, double* %zeta, align 8, !dbg !1805
  %118 = load double, double* %zeta_verify_value, align 8, !dbg !1807
  %sub118 = fsub contract double %117, %118, !dbg !1808
  %119 = call double @llvm.fabs.f64(double %sub118), !dbg !1809
  %120 = load double, double* %zeta_verify_value, align 8, !dbg !1810
  %div119 = fdiv double %119, %120, !dbg !1811
  store double %div119, double* %err, align 8, !dbg !1812
  %121 = load double, double* %err, align 8, !dbg !1813
  %122 = load double, double* %epsilon, align 8, !dbg !1815
  %cmp120 = fcmp ole double %121, %122, !dbg !1816
  br i1 %cmp120, label %if.then121, label %if.else, !dbg !1817

if.then121:                                       ; preds = %if.then117
  store i32 1, i32* %verified, align 4, !dbg !1818
  %call122 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([26 x i8], [26 x i8]* @.str.46, i64 0, i64 0)), !dbg !1820
  %123 = load double, double* %zeta, align 8, !dbg !1821
  %call123 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([21 x i8], [21 x i8]* @.str.47, i64 0, i64 0), double %123), !dbg !1822
  %124 = load double, double* %err, align 8, !dbg !1823
  %call124 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([21 x i8], [21 x i8]* @.str.48, i64 0, i64 0), double %124), !dbg !1824
  br label %if.end128, !dbg !1825

if.else:                                          ; preds = %if.then117
  store i32 0, i32* %verified, align 4, !dbg !1826
  %call125 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([22 x i8], [22 x i8]* @.str.49, i64 0, i64 0)), !dbg !1828
  %125 = load double, double* %zeta, align 8, !dbg !1829
  %call126 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([30 x i8], [30 x i8]* @.str.50, i64 0, i64 0), double %125), !dbg !1830
  %126 = load double, double* %zeta_verify_value, align 8, !dbg !1831
  %call127 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([30 x i8], [30 x i8]* @.str.51, i64 0, i64 0), double %126), !dbg !1832
  br label %if.end128

if.end128:                                        ; preds = %if.else, %if.then121
  br label %if.end132, !dbg !1833

if.else129:                                       ; preds = %for.end114
  store i32 0, i32* %verified, align 4, !dbg !1834
  %call130 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.52, i64 0, i64 0)), !dbg !1836
  %call131 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([28 x i8], [28 x i8]* @.str.53, i64 0, i64 0)), !dbg !1837
  br label %if.end132

if.end132:                                        ; preds = %if.else129, %if.end128
  %127 = load double, double* %t, align 8, !dbg !1838
  %cmp133 = fcmp une double %127, 0.000000e+00, !dbg !1840
  br i1 %cmp133, label %if.then134, label %if.else137, !dbg !1841

if.then134:                                       ; preds = %if.end132
  %128 = load double, double* %t, align 8, !dbg !1842
  %div135 = fdiv double 1.496460e+09, %128, !dbg !1844
  %div136 = fdiv double %div135, 1.000000e+06, !dbg !1845
  store double %div136, double* %mflops, align 8, !dbg !1846
  br label %if.end138, !dbg !1847

if.else137:                                       ; preds = %if.end132
  store double 0.000000e+00, double* %mflops, align 8, !dbg !1848
  br label %if.end138

if.end138:                                        ; preds = %if.else137, %if.then134
  call void @llvm.dbg.declare(metadata [256 x i8]* %gpu_config, metadata !1850, metadata !DIExpression()), !dbg !1851
  call void @llvm.dbg.declare(metadata [2048 x i8]* %gpu_config_string, metadata !1852, metadata !DIExpression()), !dbg !1856
  %arraydecay = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1857
  %call139 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay, i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.54, i64 0, i64 0), i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.55, i64 0, i64 0), i8* getelementptr inbounds ([18 x i8], [18 x i8]* @.str.56, i64 0, i64 0)) #8, !dbg !1858
  %arraydecay140 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !1859
  %arraydecay141 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1860
  %call142 = call i8* @strcpy(i8* %arraydecay140, i8* %arraydecay141) #8, !dbg !1861
  %arraydecay143 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1862
  %129 = load i32, i32* @threads_per_block_on_kernel_one, align 4, !dbg !1863
  %call144 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay143, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.57, i64 0, i64 0), i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.58, i64 0, i64 0), i32 %129) #8, !dbg !1864
  %arraydecay145 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !1865
  %arraydecay146 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1866
  %call147 = call i8* @strcat(i8* %arraydecay145, i8* %arraydecay146) #8, !dbg !1867
  %arraydecay148 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1868
  %130 = load i32, i32* @threads_per_block_on_kernel_two, align 4, !dbg !1869
  %call149 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay148, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.57, i64 0, i64 0), i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.59, i64 0, i64 0), i32 %130) #8, !dbg !1870
  %arraydecay150 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !1871
  %arraydecay151 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1872
  %call152 = call i8* @strcat(i8* %arraydecay150, i8* %arraydecay151) #8, !dbg !1873
  %arraydecay153 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1874
  %131 = load i32, i32* @threads_per_block_on_kernel_three, align 4, !dbg !1875
  %call154 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay153, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.57, i64 0, i64 0), i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.60, i64 0, i64 0), i32 %131) #8, !dbg !1876
  %arraydecay155 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !1877
  %arraydecay156 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1878
  %call157 = call i8* @strcat(i8* %arraydecay155, i8* %arraydecay156) #8, !dbg !1879
  %arraydecay158 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1880
  %132 = load i32, i32* @threads_per_block_on_kernel_four, align 4, !dbg !1881
  %call159 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay158, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.57, i64 0, i64 0), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.61, i64 0, i64 0), i32 %132) #8, !dbg !1882
  %arraydecay160 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !1883
  %arraydecay161 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1884
  %call162 = call i8* @strcat(i8* %arraydecay160, i8* %arraydecay161) #8, !dbg !1885
  %arraydecay163 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1886
  %133 = load i32, i32* @threads_per_block_on_kernel_five, align 4, !dbg !1887
  %call164 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay163, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.57, i64 0, i64 0), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.62, i64 0, i64 0), i32 %133) #8, !dbg !1888
  %arraydecay165 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !1889
  %arraydecay166 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1890
  %call167 = call i8* @strcat(i8* %arraydecay165, i8* %arraydecay166) #8, !dbg !1891
  %arraydecay168 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1892
  %134 = load i32, i32* @threads_per_block_on_kernel_six, align 4, !dbg !1893
  %call169 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay168, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.57, i64 0, i64 0), i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.63, i64 0, i64 0), i32 %134) #8, !dbg !1894
  %arraydecay170 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !1895
  %arraydecay171 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1896
  %call172 = call i8* @strcat(i8* %arraydecay170, i8* %arraydecay171) #8, !dbg !1897
  %arraydecay173 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1898
  %135 = load i32, i32* @threads_per_block_on_kernel_seven, align 4, !dbg !1899
  %call174 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay173, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.57, i64 0, i64 0), i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.64, i64 0, i64 0), i32 %135) #8, !dbg !1900
  %arraydecay175 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !1901
  %arraydecay176 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1902
  %call177 = call i8* @strcat(i8* %arraydecay175, i8* %arraydecay176) #8, !dbg !1903
  %arraydecay178 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1904
  %136 = load i32, i32* @threads_per_block_on_kernel_eight, align 4, !dbg !1905
  %call179 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay178, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.57, i64 0, i64 0), i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.65, i64 0, i64 0), i32 %136) #8, !dbg !1906
  %arraydecay180 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !1907
  %arraydecay181 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1908
  %call182 = call i8* @strcat(i8* %arraydecay180, i8* %arraydecay181) #8, !dbg !1909
  %arraydecay183 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1910
  %137 = load i32, i32* @threads_per_block_on_kernel_nine, align 4, !dbg !1911
  %call184 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay183, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.57, i64 0, i64 0), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.66, i64 0, i64 0), i32 %137) #8, !dbg !1912
  %arraydecay185 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !1913
  %arraydecay186 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1914
  %call187 = call i8* @strcat(i8* %arraydecay185, i8* %arraydecay186) #8, !dbg !1915
  %arraydecay188 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1916
  %138 = load i32, i32* @threads_per_block_on_kernel_ten, align 4, !dbg !1917
  %call189 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay188, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.57, i64 0, i64 0), i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.67, i64 0, i64 0), i32 %138) #8, !dbg !1918
  %arraydecay190 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !1919
  %arraydecay191 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1920
  %call192 = call i8* @strcat(i8* %arraydecay190, i8* %arraydecay191) #8, !dbg !1921
  %arraydecay193 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1922
  %139 = load i32, i32* @threads_per_block_on_kernel_eleven, align 4, !dbg !1923
  %call194 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay193, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.57, i64 0, i64 0), i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.68, i64 0, i64 0), i32 %139) #8, !dbg !1924
  %arraydecay195 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !1925
  %arraydecay196 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1926
  %call197 = call i8* @strcat(i8* %arraydecay195, i8* %arraydecay196) #8, !dbg !1927
  %140 = load i8, i8* %class_npb, align 1, !dbg !1928
  %141 = load double, double* %t, align 8, !dbg !1929
  %142 = load double, double* %mflops, align 8, !dbg !1930
  %143 = load i32, i32* %verified, align 4, !dbg !1931
  %arraydecay198 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !1932
  call void @_Z15c_print_resultsPcciiiiddS_iS_S_S_S_S_S_S_S_S_S_S_S_S_S_(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.69, i64 0, i64 0), i8 signext %140, i32 14000, i32 0, i32 0, i32 15, double %141, double %142, i8* getelementptr inbounds ([25 x i8], [25 x i8]* @.str.70, i64 0, i64 0), i32 %143, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.71, i64 0, i64 0), i8* getelementptr inbounds ([12 x i8], [12 x i8]* @.str.72, i64 0, i64 0), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.73, i64 0, i64 0), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.73, i64 0, i64 0), i8* getelementptr inbounds ([42 x i8], [42 x i8]* @.str.74, i64 0, i64 0), i8* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 0, i64 0), i8* %arraydecay198, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.75, i64 0, i64 0), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.76, i64 0, i64 0), i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.77, i64 0, i64 0), i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.78, i64 0, i64 0), i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.79, i64 0, i64 0), i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.79, i64 0, i64 0), i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.80, i64 0, i64 0)), !dbg !1933
  call void @_ZL11release_gpuv(), !dbg !1934
  ret i32 0, !dbg !1935
}

; Function Attrs: nounwind
declare dso_local noalias i8* @malloc(i64) #4

; Function Attrs: noinline uwtable
define internal void @_ZL5makeaiiPdPiS0_iiiiS0_PA12_iPA12_dS0_(i32 %n, i32 %nz, double* %a, i32* %colidx, i32* %rowstr, i32 %firstrow, i32 %lastrow, i32 %firstcol, i32 %lastcol, i32* %arow, [12 x i32]* %acol, [12 x double]* %aelt, i32* %iv) #2 !dbg !1936 {
entry:
  %n.addr = alloca i32, align 4
  %nz.addr = alloca i32, align 4
  %a.addr = alloca double*, align 8
  %colidx.addr = alloca i32*, align 8
  %rowstr.addr = alloca i32*, align 8
  %firstrow.addr = alloca i32, align 4
  %lastrow.addr = alloca i32, align 4
  %firstcol.addr = alloca i32, align 4
  %lastcol.addr = alloca i32, align 4
  %arow.addr = alloca i32*, align 8
  %acol.addr = alloca [12 x i32]*, align 8
  %aelt.addr = alloca [12 x double]*, align 8
  %iv.addr = alloca i32*, align 8
  %iouter = alloca i32, align 4
  %ivelt = alloca i32, align 4
  %nzv = alloca i32, align 4
  %nn1 = alloca i32, align 4
  %ivc = alloca [12 x i32], align 16
  %vc = alloca [12 x double], align 16
  store i32 %n, i32* %n.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %n.addr, metadata !1939, metadata !DIExpression()), !dbg !1940
  store i32 %nz, i32* %nz.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %nz.addr, metadata !1941, metadata !DIExpression()), !dbg !1942
  store double* %a, double** %a.addr, align 8
  call void @llvm.dbg.declare(metadata double** %a.addr, metadata !1943, metadata !DIExpression()), !dbg !1944
  store i32* %colidx, i32** %colidx.addr, align 8
  call void @llvm.dbg.declare(metadata i32** %colidx.addr, metadata !1945, metadata !DIExpression()), !dbg !1946
  store i32* %rowstr, i32** %rowstr.addr, align 8
  call void @llvm.dbg.declare(metadata i32** %rowstr.addr, metadata !1947, metadata !DIExpression()), !dbg !1948
  store i32 %firstrow, i32* %firstrow.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %firstrow.addr, metadata !1949, metadata !DIExpression()), !dbg !1950
  store i32 %lastrow, i32* %lastrow.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %lastrow.addr, metadata !1951, metadata !DIExpression()), !dbg !1952
  store i32 %firstcol, i32* %firstcol.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %firstcol.addr, metadata !1953, metadata !DIExpression()), !dbg !1954
  store i32 %lastcol, i32* %lastcol.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %lastcol.addr, metadata !1955, metadata !DIExpression()), !dbg !1956
  store i32* %arow, i32** %arow.addr, align 8
  call void @llvm.dbg.declare(metadata i32** %arow.addr, metadata !1957, metadata !DIExpression()), !dbg !1958
  store [12 x i32]* %acol, [12 x i32]** %acol.addr, align 8
  call void @llvm.dbg.declare(metadata [12 x i32]** %acol.addr, metadata !1959, metadata !DIExpression()), !dbg !1960
  store [12 x double]* %aelt, [12 x double]** %aelt.addr, align 8
  call void @llvm.dbg.declare(metadata [12 x double]** %aelt.addr, metadata !1961, metadata !DIExpression()), !dbg !1962
  store i32* %iv, i32** %iv.addr, align 8
  call void @llvm.dbg.declare(metadata i32** %iv.addr, metadata !1963, metadata !DIExpression()), !dbg !1964
  call void @llvm.dbg.declare(metadata i32* %iouter, metadata !1965, metadata !DIExpression()), !dbg !1966
  call void @llvm.dbg.declare(metadata i32* %ivelt, metadata !1967, metadata !DIExpression()), !dbg !1968
  call void @llvm.dbg.declare(metadata i32* %nzv, metadata !1969, metadata !DIExpression()), !dbg !1970
  call void @llvm.dbg.declare(metadata i32* %nn1, metadata !1971, metadata !DIExpression()), !dbg !1972
  call void @llvm.dbg.declare(metadata [12 x i32]* %ivc, metadata !1973, metadata !DIExpression()), !dbg !1974
  call void @llvm.dbg.declare(metadata [12 x double]* %vc, metadata !1975, metadata !DIExpression()), !dbg !1976
  store i32 1, i32* %nn1, align 4, !dbg !1977
  br label %do.body, !dbg !1978

do.body:                                          ; preds = %do.cond, %entry
  %0 = load i32, i32* %nn1, align 4, !dbg !1979
  %mul = mul nsw i32 2, %0, !dbg !1981
  store i32 %mul, i32* %nn1, align 4, !dbg !1982
  br label %do.cond, !dbg !1983

do.cond:                                          ; preds = %do.body
  %1 = load i32, i32* %nn1, align 4, !dbg !1984
  %2 = load i32, i32* %n.addr, align 4, !dbg !1985
  %cmp = icmp slt i32 %1, %2, !dbg !1986
  br i1 %cmp, label %do.body, label %do.end, !dbg !1983, !llvm.loop !1987

do.end:                                           ; preds = %do.cond
  store i32 0, i32* %iouter, align 4, !dbg !1989
  br label %for.cond, !dbg !1991

for.cond:                                         ; preds = %for.inc20, %do.end
  %3 = load i32, i32* %iouter, align 4, !dbg !1992
  %4 = load i32, i32* %n.addr, align 4, !dbg !1994
  %cmp1 = icmp slt i32 %3, %4, !dbg !1995
  br i1 %cmp1, label %for.body, label %for.end22, !dbg !1996

for.body:                                         ; preds = %for.cond
  store i32 11, i32* %nzv, align 4, !dbg !1997
  %5 = load i32, i32* %n.addr, align 4, !dbg !1999
  %6 = load i32, i32* %nzv, align 4, !dbg !2000
  %7 = load i32, i32* %nn1, align 4, !dbg !2001
  %arraydecay = getelementptr inbounds [12 x double], [12 x double]* %vc, i64 0, i64 0, !dbg !2002
  %arraydecay2 = getelementptr inbounds [12 x i32], [12 x i32]* %ivc, i64 0, i64 0, !dbg !2003
  call void @_ZL6sprnvciiiPdPi(i32 %5, i32 %6, i32 %7, double* %arraydecay, i32* %arraydecay2), !dbg !2004
  %8 = load i32, i32* %n.addr, align 4, !dbg !2005
  %arraydecay3 = getelementptr inbounds [12 x double], [12 x double]* %vc, i64 0, i64 0, !dbg !2006
  %arraydecay4 = getelementptr inbounds [12 x i32], [12 x i32]* %ivc, i64 0, i64 0, !dbg !2007
  %9 = load i32, i32* %iouter, align 4, !dbg !2008
  %add = add nsw i32 %9, 1, !dbg !2009
  call void @_ZL6vecsetiPdPiS0_id(i32 %8, double* %arraydecay3, i32* %arraydecay4, i32* %nzv, i32 %add, double 5.000000e-01), !dbg !2010
  %10 = load i32, i32* %nzv, align 4, !dbg !2011
  %11 = load i32*, i32** %arow.addr, align 8, !dbg !2012
  %12 = load i32, i32* %iouter, align 4, !dbg !2013
  %idxprom = sext i32 %12 to i64, !dbg !2012
  %arrayidx = getelementptr inbounds i32, i32* %11, i64 %idxprom, !dbg !2012
  store i32 %10, i32* %arrayidx, align 4, !dbg !2014
  store i32 0, i32* %ivelt, align 4, !dbg !2015
  br label %for.cond5, !dbg !2017

for.cond5:                                        ; preds = %for.inc, %for.body
  %13 = load i32, i32* %ivelt, align 4, !dbg !2018
  %14 = load i32, i32* %nzv, align 4, !dbg !2020
  %cmp6 = icmp slt i32 %13, %14, !dbg !2021
  br i1 %cmp6, label %for.body7, label %for.end, !dbg !2022

for.body7:                                        ; preds = %for.cond5
  %15 = load i32, i32* %ivelt, align 4, !dbg !2023
  %idxprom8 = sext i32 %15 to i64, !dbg !2025
  %arrayidx9 = getelementptr inbounds [12 x i32], [12 x i32]* %ivc, i64 0, i64 %idxprom8, !dbg !2025
  %16 = load i32, i32* %arrayidx9, align 4, !dbg !2025
  %sub = sub nsw i32 %16, 1, !dbg !2026
  %17 = load [12 x i32]*, [12 x i32]** %acol.addr, align 8, !dbg !2027
  %18 = load i32, i32* %iouter, align 4, !dbg !2028
  %idxprom10 = sext i32 %18 to i64, !dbg !2027
  %arrayidx11 = getelementptr inbounds [12 x i32], [12 x i32]* %17, i64 %idxprom10, !dbg !2027
  %19 = load i32, i32* %ivelt, align 4, !dbg !2029
  %idxprom12 = sext i32 %19 to i64, !dbg !2027
  %arrayidx13 = getelementptr inbounds [12 x i32], [12 x i32]* %arrayidx11, i64 0, i64 %idxprom12, !dbg !2027
  store i32 %sub, i32* %arrayidx13, align 4, !dbg !2030
  %20 = load i32, i32* %ivelt, align 4, !dbg !2031
  %idxprom14 = sext i32 %20 to i64, !dbg !2032
  %arrayidx15 = getelementptr inbounds [12 x double], [12 x double]* %vc, i64 0, i64 %idxprom14, !dbg !2032
  %21 = load double, double* %arrayidx15, align 8, !dbg !2032
  %22 = load [12 x double]*, [12 x double]** %aelt.addr, align 8, !dbg !2033
  %23 = load i32, i32* %iouter, align 4, !dbg !2034
  %idxprom16 = sext i32 %23 to i64, !dbg !2033
  %arrayidx17 = getelementptr inbounds [12 x double], [12 x double]* %22, i64 %idxprom16, !dbg !2033
  %24 = load i32, i32* %ivelt, align 4, !dbg !2035
  %idxprom18 = sext i32 %24 to i64, !dbg !2033
  %arrayidx19 = getelementptr inbounds [12 x double], [12 x double]* %arrayidx17, i64 0, i64 %idxprom18, !dbg !2033
  store double %21, double* %arrayidx19, align 8, !dbg !2036
  br label %for.inc, !dbg !2037

for.inc:                                          ; preds = %for.body7
  %25 = load i32, i32* %ivelt, align 4, !dbg !2038
  %inc = add nsw i32 %25, 1, !dbg !2038
  store i32 %inc, i32* %ivelt, align 4, !dbg !2038
  br label %for.cond5, !dbg !2039, !llvm.loop !2040

for.end:                                          ; preds = %for.cond5
  br label %for.inc20, !dbg !2042

for.inc20:                                        ; preds = %for.end
  %26 = load i32, i32* %iouter, align 4, !dbg !2043
  %inc21 = add nsw i32 %26, 1, !dbg !2043
  store i32 %inc21, i32* %iouter, align 4, !dbg !2043
  br label %for.cond, !dbg !2044, !llvm.loop !2045

for.end22:                                        ; preds = %for.cond
  %27 = load double*, double** %a.addr, align 8, !dbg !2047
  %28 = load i32*, i32** %colidx.addr, align 8, !dbg !2048
  %29 = load i32*, i32** %rowstr.addr, align 8, !dbg !2049
  %30 = load i32, i32* %n.addr, align 4, !dbg !2050
  %31 = load i32, i32* %nz.addr, align 4, !dbg !2051
  %32 = load i32*, i32** %arow.addr, align 8, !dbg !2052
  %33 = load [12 x i32]*, [12 x i32]** %acol.addr, align 8, !dbg !2053
  %34 = load [12 x double]*, [12 x double]** %aelt.addr, align 8, !dbg !2054
  %35 = load i32, i32* %firstrow.addr, align 4, !dbg !2055
  %36 = load i32, i32* %lastrow.addr, align 4, !dbg !2056
  %37 = load i32*, i32** %iv.addr, align 8, !dbg !2057
  call void @_ZL6sparsePdPiS0_iiiS0_PA12_iPA12_diiS0_dd(double* %27, i32* %28, i32* %29, i32 %30, i32 %31, i32 11, i32* %32, [12 x i32]* %33, [12 x double]* %34, i32 %35, i32 %36, i32* %37, double 1.000000e-01, double 2.000000e+01), !dbg !2058
  ret void, !dbg !2059
}

; Function Attrs: noinline nounwind uwtable
define internal void @_ZL9conj_gradPiS_PdS0_S0_S0_S0_S0_S0_(i32* %colidx, i32* %rowstr, double* %x, double* %z, double* %a, double* %p, double* %q, double* %r, double* %rnorm) #0 !dbg !2060 {
entry:
  %colidx.addr = alloca i32*, align 8
  %rowstr.addr = alloca i32*, align 8
  %x.addr = alloca double*, align 8
  %z.addr = alloca double*, align 8
  %a.addr = alloca double*, align 8
  %p.addr = alloca double*, align 8
  %q.addr = alloca double*, align 8
  %r.addr = alloca double*, align 8
  %rnorm.addr = alloca double*, align 8
  %j = alloca i32, align 4
  %k = alloca i32, align 4
  %cgit = alloca i32, align 4
  %cgitmax = alloca i32, align 4
  %d = alloca double, align 8
  %sum = alloca double, align 8
  %rho = alloca double, align 8
  %rho0 = alloca double, align 8
  %alpha = alloca double, align 8
  %beta = alloca double, align 8
  store i32* %colidx, i32** %colidx.addr, align 8
  call void @llvm.dbg.declare(metadata i32** %colidx.addr, metadata !2063, metadata !DIExpression()), !dbg !2064
  store i32* %rowstr, i32** %rowstr.addr, align 8
  call void @llvm.dbg.declare(metadata i32** %rowstr.addr, metadata !2065, metadata !DIExpression()), !dbg !2066
  store double* %x, double** %x.addr, align 8
  call void @llvm.dbg.declare(metadata double** %x.addr, metadata !2067, metadata !DIExpression()), !dbg !2068
  store double* %z, double** %z.addr, align 8
  call void @llvm.dbg.declare(metadata double** %z.addr, metadata !2069, metadata !DIExpression()), !dbg !2070
  store double* %a, double** %a.addr, align 8
  call void @llvm.dbg.declare(metadata double** %a.addr, metadata !2071, metadata !DIExpression()), !dbg !2072
  store double* %p, double** %p.addr, align 8
  call void @llvm.dbg.declare(metadata double** %p.addr, metadata !2073, metadata !DIExpression()), !dbg !2074
  store double* %q, double** %q.addr, align 8
  call void @llvm.dbg.declare(metadata double** %q.addr, metadata !2075, metadata !DIExpression()), !dbg !2076
  store double* %r, double** %r.addr, align 8
  call void @llvm.dbg.declare(metadata double** %r.addr, metadata !2077, metadata !DIExpression()), !dbg !2078
  store double* %rnorm, double** %rnorm.addr, align 8
  call void @llvm.dbg.declare(metadata double** %rnorm.addr, metadata !2079, metadata !DIExpression()), !dbg !2080
  call void @llvm.dbg.declare(metadata i32* %j, metadata !2081, metadata !DIExpression()), !dbg !2082
  call void @llvm.dbg.declare(metadata i32* %k, metadata !2083, metadata !DIExpression()), !dbg !2084
  call void @llvm.dbg.declare(metadata i32* %cgit, metadata !2085, metadata !DIExpression()), !dbg !2086
  call void @llvm.dbg.declare(metadata i32* %cgitmax, metadata !2087, metadata !DIExpression()), !dbg !2088
  call void @llvm.dbg.declare(metadata double* %d, metadata !2089, metadata !DIExpression()), !dbg !2090
  call void @llvm.dbg.declare(metadata double* %sum, metadata !2091, metadata !DIExpression()), !dbg !2092
  call void @llvm.dbg.declare(metadata double* %rho, metadata !2093, metadata !DIExpression()), !dbg !2094
  call void @llvm.dbg.declare(metadata double* %rho0, metadata !2095, metadata !DIExpression()), !dbg !2096
  call void @llvm.dbg.declare(metadata double* %alpha, metadata !2097, metadata !DIExpression()), !dbg !2098
  call void @llvm.dbg.declare(metadata double* %beta, metadata !2099, metadata !DIExpression()), !dbg !2100
  store i32 25, i32* %cgitmax, align 4, !dbg !2101
  store double 0.000000e+00, double* %rho, align 8, !dbg !2102
  store i32 0, i32* %j, align 4, !dbg !2103
  br label %for.cond, !dbg !2105

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, i32* %j, align 4, !dbg !2106
  %1 = load i32, i32* @_ZL3naa, align 4, !dbg !2108
  %add = add nsw i32 %1, 1, !dbg !2109
  %cmp = icmp slt i32 %0, %add, !dbg !2110
  br i1 %cmp, label %for.body, label %for.end, !dbg !2111

for.body:                                         ; preds = %for.cond
  %2 = load double*, double** %q.addr, align 8, !dbg !2112
  %3 = load i32, i32* %j, align 4, !dbg !2114
  %idxprom = sext i32 %3 to i64, !dbg !2112
  %arrayidx = getelementptr inbounds double, double* %2, i64 %idxprom, !dbg !2112
  store double 0.000000e+00, double* %arrayidx, align 8, !dbg !2115
  %4 = load double*, double** %z.addr, align 8, !dbg !2116
  %5 = load i32, i32* %j, align 4, !dbg !2117
  %idxprom1 = sext i32 %5 to i64, !dbg !2116
  %arrayidx2 = getelementptr inbounds double, double* %4, i64 %idxprom1, !dbg !2116
  store double 0.000000e+00, double* %arrayidx2, align 8, !dbg !2118
  %6 = load double*, double** %x.addr, align 8, !dbg !2119
  %7 = load i32, i32* %j, align 4, !dbg !2120
  %idxprom3 = sext i32 %7 to i64, !dbg !2119
  %arrayidx4 = getelementptr inbounds double, double* %6, i64 %idxprom3, !dbg !2119
  %8 = load double, double* %arrayidx4, align 8, !dbg !2119
  %9 = load double*, double** %r.addr, align 8, !dbg !2121
  %10 = load i32, i32* %j, align 4, !dbg !2122
  %idxprom5 = sext i32 %10 to i64, !dbg !2121
  %arrayidx6 = getelementptr inbounds double, double* %9, i64 %idxprom5, !dbg !2121
  store double %8, double* %arrayidx6, align 8, !dbg !2123
  %11 = load double*, double** %r.addr, align 8, !dbg !2124
  %12 = load i32, i32* %j, align 4, !dbg !2125
  %idxprom7 = sext i32 %12 to i64, !dbg !2124
  %arrayidx8 = getelementptr inbounds double, double* %11, i64 %idxprom7, !dbg !2124
  %13 = load double, double* %arrayidx8, align 8, !dbg !2124
  %14 = load double*, double** %p.addr, align 8, !dbg !2126
  %15 = load i32, i32* %j, align 4, !dbg !2127
  %idxprom9 = sext i32 %15 to i64, !dbg !2126
  %arrayidx10 = getelementptr inbounds double, double* %14, i64 %idxprom9, !dbg !2126
  store double %13, double* %arrayidx10, align 8, !dbg !2128
  br label %for.inc, !dbg !2129

for.inc:                                          ; preds = %for.body
  %16 = load i32, i32* %j, align 4, !dbg !2130
  %inc = add nsw i32 %16, 1, !dbg !2130
  store i32 %inc, i32* %j, align 4, !dbg !2130
  br label %for.cond, !dbg !2131, !llvm.loop !2132

for.end:                                          ; preds = %for.cond
  store i32 0, i32* %j, align 4, !dbg !2134
  br label %for.cond11, !dbg !2136

for.cond11:                                       ; preds = %for.inc20, %for.end
  %17 = load i32, i32* %j, align 4, !dbg !2137
  %18 = load i32, i32* @_ZL7lastcol, align 4, !dbg !2139
  %19 = load i32, i32* @_ZL8firstcol, align 4, !dbg !2140
  %sub = sub nsw i32 %18, %19, !dbg !2141
  %add12 = add nsw i32 %sub, 1, !dbg !2142
  %cmp13 = icmp slt i32 %17, %add12, !dbg !2143
  br i1 %cmp13, label %for.body14, label %for.end22, !dbg !2144

for.body14:                                       ; preds = %for.cond11
  %20 = load double, double* %rho, align 8, !dbg !2145
  %21 = load double*, double** %r.addr, align 8, !dbg !2147
  %22 = load i32, i32* %j, align 4, !dbg !2148
  %idxprom15 = sext i32 %22 to i64, !dbg !2147
  %arrayidx16 = getelementptr inbounds double, double* %21, i64 %idxprom15, !dbg !2147
  %23 = load double, double* %arrayidx16, align 8, !dbg !2147
  %24 = load double*, double** %r.addr, align 8, !dbg !2149
  %25 = load i32, i32* %j, align 4, !dbg !2150
  %idxprom17 = sext i32 %25 to i64, !dbg !2149
  %arrayidx18 = getelementptr inbounds double, double* %24, i64 %idxprom17, !dbg !2149
  %26 = load double, double* %arrayidx18, align 8, !dbg !2149
  %mul = fmul contract double %23, %26, !dbg !2151
  %add19 = fadd contract double %20, %mul, !dbg !2152
  store double %add19, double* %rho, align 8, !dbg !2153
  br label %for.inc20, !dbg !2154

for.inc20:                                        ; preds = %for.body14
  %27 = load i32, i32* %j, align 4, !dbg !2155
  %inc21 = add nsw i32 %27, 1, !dbg !2155
  store i32 %inc21, i32* %j, align 4, !dbg !2155
  br label %for.cond11, !dbg !2156, !llvm.loop !2157

for.end22:                                        ; preds = %for.cond11
  store i32 1, i32* %cgit, align 4, !dbg !2159
  br label %for.cond23, !dbg !2161

for.cond23:                                       ; preds = %for.inc124, %for.end22
  %28 = load i32, i32* %cgit, align 4, !dbg !2162
  %29 = load i32, i32* %cgitmax, align 4, !dbg !2164
  %cmp24 = icmp sle i32 %28, %29, !dbg !2165
  br i1 %cmp24, label %for.body25, label %for.end126, !dbg !2166

for.body25:                                       ; preds = %for.cond23
  store i32 0, i32* %j, align 4, !dbg !2167
  br label %for.cond26, !dbg !2170

for.cond26:                                       ; preds = %for.inc52, %for.body25
  %30 = load i32, i32* %j, align 4, !dbg !2171
  %31 = load i32, i32* @_ZL7lastrow, align 4, !dbg !2173
  %32 = load i32, i32* @_ZL8firstrow, align 4, !dbg !2174
  %sub27 = sub nsw i32 %31, %32, !dbg !2175
  %add28 = add nsw i32 %sub27, 1, !dbg !2176
  %cmp29 = icmp slt i32 %30, %add28, !dbg !2177
  br i1 %cmp29, label %for.body30, label %for.end54, !dbg !2178

for.body30:                                       ; preds = %for.cond26
  store double 0.000000e+00, double* %sum, align 8, !dbg !2179
  %33 = load i32*, i32** %rowstr.addr, align 8, !dbg !2181
  %34 = load i32, i32* %j, align 4, !dbg !2183
  %idxprom31 = sext i32 %34 to i64, !dbg !2181
  %arrayidx32 = getelementptr inbounds i32, i32* %33, i64 %idxprom31, !dbg !2181
  %35 = load i32, i32* %arrayidx32, align 4, !dbg !2181
  store i32 %35, i32* %k, align 4, !dbg !2184
  br label %for.cond33, !dbg !2185

for.cond33:                                       ; preds = %for.inc47, %for.body30
  %36 = load i32, i32* %k, align 4, !dbg !2186
  %37 = load i32*, i32** %rowstr.addr, align 8, !dbg !2188
  %38 = load i32, i32* %j, align 4, !dbg !2189
  %add34 = add nsw i32 %38, 1, !dbg !2190
  %idxprom35 = sext i32 %add34 to i64, !dbg !2188
  %arrayidx36 = getelementptr inbounds i32, i32* %37, i64 %idxprom35, !dbg !2188
  %39 = load i32, i32* %arrayidx36, align 4, !dbg !2188
  %cmp37 = icmp slt i32 %36, %39, !dbg !2191
  br i1 %cmp37, label %for.body38, label %for.end49, !dbg !2192

for.body38:                                       ; preds = %for.cond33
  %40 = load double, double* %sum, align 8, !dbg !2193
  %41 = load double*, double** %a.addr, align 8, !dbg !2195
  %42 = load i32, i32* %k, align 4, !dbg !2196
  %idxprom39 = sext i32 %42 to i64, !dbg !2195
  %arrayidx40 = getelementptr inbounds double, double* %41, i64 %idxprom39, !dbg !2195
  %43 = load double, double* %arrayidx40, align 8, !dbg !2195
  %44 = load double*, double** %p.addr, align 8, !dbg !2197
  %45 = load i32*, i32** %colidx.addr, align 8, !dbg !2198
  %46 = load i32, i32* %k, align 4, !dbg !2199
  %idxprom41 = sext i32 %46 to i64, !dbg !2198
  %arrayidx42 = getelementptr inbounds i32, i32* %45, i64 %idxprom41, !dbg !2198
  %47 = load i32, i32* %arrayidx42, align 4, !dbg !2198
  %idxprom43 = sext i32 %47 to i64, !dbg !2197
  %arrayidx44 = getelementptr inbounds double, double* %44, i64 %idxprom43, !dbg !2197
  %48 = load double, double* %arrayidx44, align 8, !dbg !2197
  %mul45 = fmul contract double %43, %48, !dbg !2200
  %add46 = fadd contract double %40, %mul45, !dbg !2201
  store double %add46, double* %sum, align 8, !dbg !2202
  br label %for.inc47, !dbg !2203

for.inc47:                                        ; preds = %for.body38
  %49 = load i32, i32* %k, align 4, !dbg !2204
  %inc48 = add nsw i32 %49, 1, !dbg !2204
  store i32 %inc48, i32* %k, align 4, !dbg !2204
  br label %for.cond33, !dbg !2205, !llvm.loop !2206

for.end49:                                        ; preds = %for.cond33
  %50 = load double, double* %sum, align 8, !dbg !2208
  %51 = load double*, double** %q.addr, align 8, !dbg !2209
  %52 = load i32, i32* %j, align 4, !dbg !2210
  %idxprom50 = sext i32 %52 to i64, !dbg !2209
  %arrayidx51 = getelementptr inbounds double, double* %51, i64 %idxprom50, !dbg !2209
  store double %50, double* %arrayidx51, align 8, !dbg !2211
  br label %for.inc52, !dbg !2212

for.inc52:                                        ; preds = %for.end49
  %53 = load i32, i32* %j, align 4, !dbg !2213
  %inc53 = add nsw i32 %53, 1, !dbg !2213
  store i32 %inc53, i32* %j, align 4, !dbg !2213
  br label %for.cond26, !dbg !2214, !llvm.loop !2215

for.end54:                                        ; preds = %for.cond26
  store double 0.000000e+00, double* %d, align 8, !dbg !2217
  store i32 0, i32* %j, align 4, !dbg !2218
  br label %for.cond55, !dbg !2220

for.cond55:                                       ; preds = %for.inc66, %for.end54
  %54 = load i32, i32* %j, align 4, !dbg !2221
  %55 = load i32, i32* @_ZL7lastcol, align 4, !dbg !2223
  %56 = load i32, i32* @_ZL8firstcol, align 4, !dbg !2224
  %sub56 = sub nsw i32 %55, %56, !dbg !2225
  %add57 = add nsw i32 %sub56, 1, !dbg !2226
  %cmp58 = icmp slt i32 %54, %add57, !dbg !2227
  br i1 %cmp58, label %for.body59, label %for.end68, !dbg !2228

for.body59:                                       ; preds = %for.cond55
  %57 = load double, double* %d, align 8, !dbg !2229
  %58 = load double*, double** %p.addr, align 8, !dbg !2231
  %59 = load i32, i32* %j, align 4, !dbg !2232
  %idxprom60 = sext i32 %59 to i64, !dbg !2231
  %arrayidx61 = getelementptr inbounds double, double* %58, i64 %idxprom60, !dbg !2231
  %60 = load double, double* %arrayidx61, align 8, !dbg !2231
  %61 = load double*, double** %q.addr, align 8, !dbg !2233
  %62 = load i32, i32* %j, align 4, !dbg !2234
  %idxprom62 = sext i32 %62 to i64, !dbg !2233
  %arrayidx63 = getelementptr inbounds double, double* %61, i64 %idxprom62, !dbg !2233
  %63 = load double, double* %arrayidx63, align 8, !dbg !2233
  %mul64 = fmul contract double %60, %63, !dbg !2235
  %add65 = fadd contract double %57, %mul64, !dbg !2236
  store double %add65, double* %d, align 8, !dbg !2237
  br label %for.inc66, !dbg !2238

for.inc66:                                        ; preds = %for.body59
  %64 = load i32, i32* %j, align 4, !dbg !2239
  %inc67 = add nsw i32 %64, 1, !dbg !2239
  store i32 %inc67, i32* %j, align 4, !dbg !2239
  br label %for.cond55, !dbg !2240, !llvm.loop !2241

for.end68:                                        ; preds = %for.cond55
  %65 = load double, double* %rho, align 8, !dbg !2243
  %66 = load double, double* %d, align 8, !dbg !2244
  %div = fdiv double %65, %66, !dbg !2245
  store double %div, double* %alpha, align 8, !dbg !2246
  %67 = load double, double* %rho, align 8, !dbg !2247
  store double %67, double* %rho0, align 8, !dbg !2248
  store double 0.000000e+00, double* %rho, align 8, !dbg !2249
  store i32 0, i32* %j, align 4, !dbg !2250
  br label %for.cond69, !dbg !2252

for.cond69:                                       ; preds = %for.inc90, %for.end68
  %68 = load i32, i32* %j, align 4, !dbg !2253
  %69 = load i32, i32* @_ZL7lastcol, align 4, !dbg !2255
  %70 = load i32, i32* @_ZL8firstcol, align 4, !dbg !2256
  %sub70 = sub nsw i32 %69, %70, !dbg !2257
  %add71 = add nsw i32 %sub70, 1, !dbg !2258
  %cmp72 = icmp slt i32 %68, %add71, !dbg !2259
  br i1 %cmp72, label %for.body73, label %for.end92, !dbg !2260

for.body73:                                       ; preds = %for.cond69
  %71 = load double*, double** %z.addr, align 8, !dbg !2261
  %72 = load i32, i32* %j, align 4, !dbg !2263
  %idxprom74 = sext i32 %72 to i64, !dbg !2261
  %arrayidx75 = getelementptr inbounds double, double* %71, i64 %idxprom74, !dbg !2261
  %73 = load double, double* %arrayidx75, align 8, !dbg !2261
  %74 = load double, double* %alpha, align 8, !dbg !2264
  %75 = load double*, double** %p.addr, align 8, !dbg !2265
  %76 = load i32, i32* %j, align 4, !dbg !2266
  %idxprom76 = sext i32 %76 to i64, !dbg !2265
  %arrayidx77 = getelementptr inbounds double, double* %75, i64 %idxprom76, !dbg !2265
  %77 = load double, double* %arrayidx77, align 8, !dbg !2265
  %mul78 = fmul contract double %74, %77, !dbg !2267
  %add79 = fadd contract double %73, %mul78, !dbg !2268
  %78 = load double*, double** %z.addr, align 8, !dbg !2269
  %79 = load i32, i32* %j, align 4, !dbg !2270
  %idxprom80 = sext i32 %79 to i64, !dbg !2269
  %arrayidx81 = getelementptr inbounds double, double* %78, i64 %idxprom80, !dbg !2269
  store double %add79, double* %arrayidx81, align 8, !dbg !2271
  %80 = load double*, double** %r.addr, align 8, !dbg !2272
  %81 = load i32, i32* %j, align 4, !dbg !2273
  %idxprom82 = sext i32 %81 to i64, !dbg !2272
  %arrayidx83 = getelementptr inbounds double, double* %80, i64 %idxprom82, !dbg !2272
  %82 = load double, double* %arrayidx83, align 8, !dbg !2272
  %83 = load double, double* %alpha, align 8, !dbg !2274
  %84 = load double*, double** %q.addr, align 8, !dbg !2275
  %85 = load i32, i32* %j, align 4, !dbg !2276
  %idxprom84 = sext i32 %85 to i64, !dbg !2275
  %arrayidx85 = getelementptr inbounds double, double* %84, i64 %idxprom84, !dbg !2275
  %86 = load double, double* %arrayidx85, align 8, !dbg !2275
  %mul86 = fmul contract double %83, %86, !dbg !2277
  %sub87 = fsub contract double %82, %mul86, !dbg !2278
  %87 = load double*, double** %r.addr, align 8, !dbg !2279
  %88 = load i32, i32* %j, align 4, !dbg !2280
  %idxprom88 = sext i32 %88 to i64, !dbg !2279
  %arrayidx89 = getelementptr inbounds double, double* %87, i64 %idxprom88, !dbg !2279
  store double %sub87, double* %arrayidx89, align 8, !dbg !2281
  br label %for.inc90, !dbg !2282

for.inc90:                                        ; preds = %for.body73
  %89 = load i32, i32* %j, align 4, !dbg !2283
  %inc91 = add nsw i32 %89, 1, !dbg !2283
  store i32 %inc91, i32* %j, align 4, !dbg !2283
  br label %for.cond69, !dbg !2284, !llvm.loop !2285

for.end92:                                        ; preds = %for.cond69
  store i32 0, i32* %j, align 4, !dbg !2287
  br label %for.cond93, !dbg !2289

for.cond93:                                       ; preds = %for.inc104, %for.end92
  %90 = load i32, i32* %j, align 4, !dbg !2290
  %91 = load i32, i32* @_ZL7lastcol, align 4, !dbg !2292
  %92 = load i32, i32* @_ZL8firstcol, align 4, !dbg !2293
  %sub94 = sub nsw i32 %91, %92, !dbg !2294
  %add95 = add nsw i32 %sub94, 1, !dbg !2295
  %cmp96 = icmp slt i32 %90, %add95, !dbg !2296
  br i1 %cmp96, label %for.body97, label %for.end106, !dbg !2297

for.body97:                                       ; preds = %for.cond93
  %93 = load double, double* %rho, align 8, !dbg !2298
  %94 = load double*, double** %r.addr, align 8, !dbg !2300
  %95 = load i32, i32* %j, align 4, !dbg !2301
  %idxprom98 = sext i32 %95 to i64, !dbg !2300
  %arrayidx99 = getelementptr inbounds double, double* %94, i64 %idxprom98, !dbg !2300
  %96 = load double, double* %arrayidx99, align 8, !dbg !2300
  %97 = load double*, double** %r.addr, align 8, !dbg !2302
  %98 = load i32, i32* %j, align 4, !dbg !2303
  %idxprom100 = sext i32 %98 to i64, !dbg !2302
  %arrayidx101 = getelementptr inbounds double, double* %97, i64 %idxprom100, !dbg !2302
  %99 = load double, double* %arrayidx101, align 8, !dbg !2302
  %mul102 = fmul contract double %96, %99, !dbg !2304
  %add103 = fadd contract double %93, %mul102, !dbg !2305
  store double %add103, double* %rho, align 8, !dbg !2306
  br label %for.inc104, !dbg !2307

for.inc104:                                       ; preds = %for.body97
  %100 = load i32, i32* %j, align 4, !dbg !2308
  %inc105 = add nsw i32 %100, 1, !dbg !2308
  store i32 %inc105, i32* %j, align 4, !dbg !2308
  br label %for.cond93, !dbg !2309, !llvm.loop !2310

for.end106:                                       ; preds = %for.cond93
  %101 = load double, double* %rho, align 8, !dbg !2312
  %102 = load double, double* %rho0, align 8, !dbg !2313
  %div107 = fdiv double %101, %102, !dbg !2314
  store double %div107, double* %beta, align 8, !dbg !2315
  store i32 0, i32* %j, align 4, !dbg !2316
  br label %for.cond108, !dbg !2318

for.cond108:                                      ; preds = %for.inc121, %for.end106
  %103 = load i32, i32* %j, align 4, !dbg !2319
  %104 = load i32, i32* @_ZL7lastcol, align 4, !dbg !2321
  %105 = load i32, i32* @_ZL8firstcol, align 4, !dbg !2322
  %sub109 = sub nsw i32 %104, %105, !dbg !2323
  %add110 = add nsw i32 %sub109, 1, !dbg !2324
  %cmp111 = icmp slt i32 %103, %add110, !dbg !2325
  br i1 %cmp111, label %for.body112, label %for.end123, !dbg !2326

for.body112:                                      ; preds = %for.cond108
  %106 = load double*, double** %r.addr, align 8, !dbg !2327
  %107 = load i32, i32* %j, align 4, !dbg !2329
  %idxprom113 = sext i32 %107 to i64, !dbg !2327
  %arrayidx114 = getelementptr inbounds double, double* %106, i64 %idxprom113, !dbg !2327
  %108 = load double, double* %arrayidx114, align 8, !dbg !2327
  %109 = load double, double* %beta, align 8, !dbg !2330
  %110 = load double*, double** %p.addr, align 8, !dbg !2331
  %111 = load i32, i32* %j, align 4, !dbg !2332
  %idxprom115 = sext i32 %111 to i64, !dbg !2331
  %arrayidx116 = getelementptr inbounds double, double* %110, i64 %idxprom115, !dbg !2331
  %112 = load double, double* %arrayidx116, align 8, !dbg !2331
  %mul117 = fmul contract double %109, %112, !dbg !2333
  %add118 = fadd contract double %108, %mul117, !dbg !2334
  %113 = load double*, double** %p.addr, align 8, !dbg !2335
  %114 = load i32, i32* %j, align 4, !dbg !2336
  %idxprom119 = sext i32 %114 to i64, !dbg !2335
  %arrayidx120 = getelementptr inbounds double, double* %113, i64 %idxprom119, !dbg !2335
  store double %add118, double* %arrayidx120, align 8, !dbg !2337
  br label %for.inc121, !dbg !2338

for.inc121:                                       ; preds = %for.body112
  %115 = load i32, i32* %j, align 4, !dbg !2339
  %inc122 = add nsw i32 %115, 1, !dbg !2339
  store i32 %inc122, i32* %j, align 4, !dbg !2339
  br label %for.cond108, !dbg !2340, !llvm.loop !2341

for.end123:                                       ; preds = %for.cond108
  br label %for.inc124, !dbg !2343

for.inc124:                                       ; preds = %for.end123
  %116 = load i32, i32* %cgit, align 4, !dbg !2344
  %inc125 = add nsw i32 %116, 1, !dbg !2344
  store i32 %inc125, i32* %cgit, align 4, !dbg !2344
  br label %for.cond23, !dbg !2345, !llvm.loop !2346

for.end126:                                       ; preds = %for.cond23
  store double 0.000000e+00, double* %sum, align 8, !dbg !2348
  store i32 0, i32* %j, align 4, !dbg !2349
  br label %for.cond127, !dbg !2351

for.cond127:                                      ; preds = %for.inc153, %for.end126
  %117 = load i32, i32* %j, align 4, !dbg !2352
  %118 = load i32, i32* @_ZL7lastrow, align 4, !dbg !2354
  %119 = load i32, i32* @_ZL8firstrow, align 4, !dbg !2355
  %sub128 = sub nsw i32 %118, %119, !dbg !2356
  %add129 = add nsw i32 %sub128, 1, !dbg !2357
  %cmp130 = icmp slt i32 %117, %add129, !dbg !2358
  br i1 %cmp130, label %for.body131, label %for.end155, !dbg !2359

for.body131:                                      ; preds = %for.cond127
  store double 0.000000e+00, double* %d, align 8, !dbg !2360
  %120 = load i32*, i32** %rowstr.addr, align 8, !dbg !2362
  %121 = load i32, i32* %j, align 4, !dbg !2364
  %idxprom132 = sext i32 %121 to i64, !dbg !2362
  %arrayidx133 = getelementptr inbounds i32, i32* %120, i64 %idxprom132, !dbg !2362
  %122 = load i32, i32* %arrayidx133, align 4, !dbg !2362
  store i32 %122, i32* %k, align 4, !dbg !2365
  br label %for.cond134, !dbg !2366

for.cond134:                                      ; preds = %for.inc148, %for.body131
  %123 = load i32, i32* %k, align 4, !dbg !2367
  %124 = load i32*, i32** %rowstr.addr, align 8, !dbg !2369
  %125 = load i32, i32* %j, align 4, !dbg !2370
  %add135 = add nsw i32 %125, 1, !dbg !2371
  %idxprom136 = sext i32 %add135 to i64, !dbg !2369
  %arrayidx137 = getelementptr inbounds i32, i32* %124, i64 %idxprom136, !dbg !2369
  %126 = load i32, i32* %arrayidx137, align 4, !dbg !2369
  %cmp138 = icmp slt i32 %123, %126, !dbg !2372
  br i1 %cmp138, label %for.body139, label %for.end150, !dbg !2373

for.body139:                                      ; preds = %for.cond134
  %127 = load double, double* %d, align 8, !dbg !2374
  %128 = load double*, double** %a.addr, align 8, !dbg !2376
  %129 = load i32, i32* %k, align 4, !dbg !2377
  %idxprom140 = sext i32 %129 to i64, !dbg !2376
  %arrayidx141 = getelementptr inbounds double, double* %128, i64 %idxprom140, !dbg !2376
  %130 = load double, double* %arrayidx141, align 8, !dbg !2376
  %131 = load double*, double** %z.addr, align 8, !dbg !2378
  %132 = load i32*, i32** %colidx.addr, align 8, !dbg !2379
  %133 = load i32, i32* %k, align 4, !dbg !2380
  %idxprom142 = sext i32 %133 to i64, !dbg !2379
  %arrayidx143 = getelementptr inbounds i32, i32* %132, i64 %idxprom142, !dbg !2379
  %134 = load i32, i32* %arrayidx143, align 4, !dbg !2379
  %idxprom144 = sext i32 %134 to i64, !dbg !2378
  %arrayidx145 = getelementptr inbounds double, double* %131, i64 %idxprom144, !dbg !2378
  %135 = load double, double* %arrayidx145, align 8, !dbg !2378
  %mul146 = fmul contract double %130, %135, !dbg !2381
  %add147 = fadd contract double %127, %mul146, !dbg !2382
  store double %add147, double* %d, align 8, !dbg !2383
  br label %for.inc148, !dbg !2384

for.inc148:                                       ; preds = %for.body139
  %136 = load i32, i32* %k, align 4, !dbg !2385
  %inc149 = add nsw i32 %136, 1, !dbg !2385
  store i32 %inc149, i32* %k, align 4, !dbg !2385
  br label %for.cond134, !dbg !2386, !llvm.loop !2387

for.end150:                                       ; preds = %for.cond134
  %137 = load double, double* %d, align 8, !dbg !2389
  %138 = load double*, double** %r.addr, align 8, !dbg !2390
  %139 = load i32, i32* %j, align 4, !dbg !2391
  %idxprom151 = sext i32 %139 to i64, !dbg !2390
  %arrayidx152 = getelementptr inbounds double, double* %138, i64 %idxprom151, !dbg !2390
  store double %137, double* %arrayidx152, align 8, !dbg !2392
  br label %for.inc153, !dbg !2393

for.inc153:                                       ; preds = %for.end150
  %140 = load i32, i32* %j, align 4, !dbg !2394
  %inc154 = add nsw i32 %140, 1, !dbg !2394
  store i32 %inc154, i32* %j, align 4, !dbg !2394
  br label %for.cond127, !dbg !2395, !llvm.loop !2396

for.end155:                                       ; preds = %for.cond127
  store i32 0, i32* %j, align 4, !dbg !2398
  br label %for.cond156, !dbg !2400

for.cond156:                                      ; preds = %for.inc168, %for.end155
  %141 = load i32, i32* %j, align 4, !dbg !2401
  %142 = load i32, i32* @_ZL7lastcol, align 4, !dbg !2403
  %143 = load i32, i32* @_ZL8firstcol, align 4, !dbg !2404
  %sub157 = sub nsw i32 %142, %143, !dbg !2405
  %add158 = add nsw i32 %sub157, 1, !dbg !2406
  %cmp159 = icmp slt i32 %141, %add158, !dbg !2407
  br i1 %cmp159, label %for.body160, label %for.end170, !dbg !2408

for.body160:                                      ; preds = %for.cond156
  %144 = load double*, double** %x.addr, align 8, !dbg !2409
  %145 = load i32, i32* %j, align 4, !dbg !2411
  %idxprom161 = sext i32 %145 to i64, !dbg !2409
  %arrayidx162 = getelementptr inbounds double, double* %144, i64 %idxprom161, !dbg !2409
  %146 = load double, double* %arrayidx162, align 8, !dbg !2409
  %147 = load double*, double** %r.addr, align 8, !dbg !2412
  %148 = load i32, i32* %j, align 4, !dbg !2413
  %idxprom163 = sext i32 %148 to i64, !dbg !2412
  %arrayidx164 = getelementptr inbounds double, double* %147, i64 %idxprom163, !dbg !2412
  %149 = load double, double* %arrayidx164, align 8, !dbg !2412
  %sub165 = fsub contract double %146, %149, !dbg !2414
  store double %sub165, double* %d, align 8, !dbg !2415
  %150 = load double, double* %sum, align 8, !dbg !2416
  %151 = load double, double* %d, align 8, !dbg !2417
  %152 = load double, double* %d, align 8, !dbg !2418
  %mul166 = fmul contract double %151, %152, !dbg !2419
  %add167 = fadd contract double %150, %mul166, !dbg !2420
  store double %add167, double* %sum, align 8, !dbg !2421
  br label %for.inc168, !dbg !2422

for.inc168:                                       ; preds = %for.body160
  %153 = load i32, i32* %j, align 4, !dbg !2423
  %inc169 = add nsw i32 %153, 1, !dbg !2423
  store i32 %inc169, i32* %j, align 4, !dbg !2423
  br label %for.cond156, !dbg !2424, !llvm.loop !2425

for.end170:                                       ; preds = %for.cond156
  %154 = load double, double* %sum, align 8, !dbg !2427
  %call = call double @sqrt(double %154) #8, !dbg !2428
  %155 = load double*, double** %rnorm.addr, align 8, !dbg !2429
  store double %call, double* %155, align 8, !dbg !2430
  ret void, !dbg !2431
}

; Function Attrs: nounwind
declare dso_local double @sqrt(double) #4

; Function Attrs: noinline uwtable
define internal void @_ZL9setup_gpuv() #2 !dbg !2432 {
entry:
  store i32 32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !2433
  store i32 1024, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !2434
  %0 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !2435
  %cmp = icmp sle i32 1024, %0, !dbg !2437
  br i1 %cmp, label %if.then, label %if.else, !dbg !2438

if.then:                                          ; preds = %entry
  store i32 1024, i32* @threads_per_block_on_kernel_one, align 4, !dbg !2439
  br label %if.end, !dbg !2441

if.else:                                          ; preds = %entry
  %1 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !2442
  store i32 %1, i32* @threads_per_block_on_kernel_one, align 4, !dbg !2444
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %2 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !2445
  %cmp1 = icmp sle i32 256, %2, !dbg !2447
  br i1 %cmp1, label %if.then2, label %if.else3, !dbg !2448

if.then2:                                         ; preds = %if.end
  store i32 256, i32* @threads_per_block_on_kernel_two, align 4, !dbg !2449
  br label %if.end4, !dbg !2451

if.else3:                                         ; preds = %if.end
  %3 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !2452
  store i32 %3, i32* @threads_per_block_on_kernel_two, align 4, !dbg !2454
  br label %if.end4

if.end4:                                          ; preds = %if.else3, %if.then2
  %4 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !2455
  %cmp5 = icmp sle i32 64, %4, !dbg !2457
  br i1 %cmp5, label %if.then6, label %if.else7, !dbg !2458

if.then6:                                         ; preds = %if.end4
  store i32 64, i32* @threads_per_block_on_kernel_three, align 4, !dbg !2459
  br label %if.end8, !dbg !2461

if.else7:                                         ; preds = %if.end4
  %5 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !2462
  store i32 %5, i32* @threads_per_block_on_kernel_three, align 4, !dbg !2464
  br label %if.end8

if.end8:                                          ; preds = %if.else7, %if.then6
  %6 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !2465
  %cmp9 = icmp sle i32 256, %6, !dbg !2467
  br i1 %cmp9, label %if.then10, label %if.else11, !dbg !2468

if.then10:                                        ; preds = %if.end8
  store i32 256, i32* @threads_per_block_on_kernel_four, align 4, !dbg !2469
  br label %if.end12, !dbg !2471

if.else11:                                        ; preds = %if.end8
  %7 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !2472
  store i32 %7, i32* @threads_per_block_on_kernel_four, align 4, !dbg !2474
  br label %if.end12

if.end12:                                         ; preds = %if.else11, %if.then10
  %8 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !2475
  %cmp13 = icmp sle i32 64, %8, !dbg !2477
  br i1 %cmp13, label %if.then14, label %if.else15, !dbg !2478

if.then14:                                        ; preds = %if.end12
  store i32 64, i32* @threads_per_block_on_kernel_five, align 4, !dbg !2479
  br label %if.end16, !dbg !2481

if.else15:                                        ; preds = %if.end12
  %9 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !2482
  store i32 %9, i32* @threads_per_block_on_kernel_five, align 4, !dbg !2484
  br label %if.end16

if.end16:                                         ; preds = %if.else15, %if.then14
  %10 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !2485
  %cmp17 = icmp sle i32 256, %10, !dbg !2487
  br i1 %cmp17, label %if.then18, label %if.else19, !dbg !2488

if.then18:                                        ; preds = %if.end16
  store i32 256, i32* @threads_per_block_on_kernel_six, align 4, !dbg !2489
  br label %if.end20, !dbg !2491

if.else19:                                        ; preds = %if.end16
  %11 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !2492
  store i32 %11, i32* @threads_per_block_on_kernel_six, align 4, !dbg !2494
  br label %if.end20

if.end20:                                         ; preds = %if.else19, %if.then18
  %12 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !2495
  %cmp21 = icmp sle i32 512, %12, !dbg !2497
  br i1 %cmp21, label %if.then22, label %if.else23, !dbg !2498

if.then22:                                        ; preds = %if.end20
  store i32 512, i32* @threads_per_block_on_kernel_seven, align 4, !dbg !2499
  br label %if.end24, !dbg !2501

if.else23:                                        ; preds = %if.end20
  %13 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !2502
  store i32 %13, i32* @threads_per_block_on_kernel_seven, align 4, !dbg !2504
  br label %if.end24

if.end24:                                         ; preds = %if.else23, %if.then22
  %14 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !2505
  %cmp25 = icmp sle i32 64, %14, !dbg !2507
  br i1 %cmp25, label %if.then26, label %if.else27, !dbg !2508

if.then26:                                        ; preds = %if.end24
  store i32 64, i32* @threads_per_block_on_kernel_eight, align 4, !dbg !2509
  br label %if.end28, !dbg !2511

if.else27:                                        ; preds = %if.end24
  %15 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !2512
  store i32 %15, i32* @threads_per_block_on_kernel_eight, align 4, !dbg !2514
  br label %if.end28

if.end28:                                         ; preds = %if.else27, %if.then26
  %16 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !2515
  %cmp29 = icmp sle i32 512, %16, !dbg !2517
  br i1 %cmp29, label %if.then30, label %if.else31, !dbg !2518

if.then30:                                        ; preds = %if.end28
  store i32 512, i32* @threads_per_block_on_kernel_nine, align 4, !dbg !2519
  br label %if.end32, !dbg !2521

if.else31:                                        ; preds = %if.end28
  %17 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !2522
  store i32 %17, i32* @threads_per_block_on_kernel_nine, align 4, !dbg !2524
  br label %if.end32

if.end32:                                         ; preds = %if.else31, %if.then30
  %18 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !2525
  %cmp33 = icmp sle i32 256, %18, !dbg !2527
  br i1 %cmp33, label %if.then34, label %if.else35, !dbg !2528

if.then34:                                        ; preds = %if.end32
  store i32 256, i32* @threads_per_block_on_kernel_ten, align 4, !dbg !2529
  br label %if.end36, !dbg !2531

if.else35:                                        ; preds = %if.end32
  %19 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !2532
  store i32 %19, i32* @threads_per_block_on_kernel_ten, align 4, !dbg !2534
  br label %if.end36

if.end36:                                         ; preds = %if.else35, %if.then34
  %20 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !2535
  %cmp37 = icmp sle i32 512, %20, !dbg !2537
  br i1 %cmp37, label %if.then38, label %if.else39, !dbg !2538

if.then38:                                        ; preds = %if.end36
  store i32 512, i32* @threads_per_block_on_kernel_eleven, align 4, !dbg !2539
  br label %if.end40, !dbg !2541

if.else39:                                        ; preds = %if.end36
  %21 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !2542
  store i32 %21, i32* @threads_per_block_on_kernel_eleven, align 4, !dbg !2544
  br label %if.end40

if.end40:                                         ; preds = %if.else39, %if.then38
  %22 = load i32, i32* @threads_per_block_on_kernel_one, align 4, !dbg !2545
  %conv = sitofp i32 %22 to double, !dbg !2545
  %div = fdiv double 1.400000e+04, %conv, !dbg !2546
  %23 = call double @llvm.ceil.f64(double %div), !dbg !2547
  %conv41 = fptosi double %23 to i32, !dbg !2548
  store i32 %conv41, i32* @blocks_per_grid_on_kernel_one, align 4, !dbg !2549
  %24 = load i32, i32* @threads_per_block_on_kernel_two, align 4, !dbg !2550
  %conv42 = sitofp i32 %24 to double, !dbg !2550
  %div43 = fdiv double 1.400000e+04, %conv42, !dbg !2551
  %25 = call double @llvm.ceil.f64(double %div43), !dbg !2552
  %conv44 = fptosi double %25 to i32, !dbg !2553
  store i32 %conv44, i32* @blocks_per_grid_on_kernel_two, align 4, !dbg !2554
  store i32 14000, i32* @blocks_per_grid_on_kernel_three, align 4, !dbg !2555
  %26 = load i32, i32* @threads_per_block_on_kernel_four, align 4, !dbg !2556
  %conv45 = sitofp i32 %26 to double, !dbg !2556
  %div46 = fdiv double 1.400000e+04, %conv45, !dbg !2557
  %27 = call double @llvm.ceil.f64(double %div46), !dbg !2558
  %conv47 = fptosi double %27 to i32, !dbg !2559
  store i32 %conv47, i32* @blocks_per_grid_on_kernel_four, align 4, !dbg !2560
  %28 = load i32, i32* @threads_per_block_on_kernel_five, align 4, !dbg !2561
  %conv48 = sitofp i32 %28 to double, !dbg !2561
  %div49 = fdiv double 1.400000e+04, %conv48, !dbg !2562
  %29 = call double @llvm.ceil.f64(double %div49), !dbg !2563
  %conv50 = fptosi double %29 to i32, !dbg !2564
  store i32 %conv50, i32* @blocks_per_grid_on_kernel_five, align 4, !dbg !2565
  %30 = load i32, i32* @threads_per_block_on_kernel_six, align 4, !dbg !2566
  %conv51 = sitofp i32 %30 to double, !dbg !2566
  %div52 = fdiv double 1.400000e+04, %conv51, !dbg !2567
  %31 = call double @llvm.ceil.f64(double %div52), !dbg !2568
  %conv53 = fptosi double %31 to i32, !dbg !2569
  store i32 %conv53, i32* @blocks_per_grid_on_kernel_six, align 4, !dbg !2570
  %32 = load i32, i32* @threads_per_block_on_kernel_seven, align 4, !dbg !2571
  %conv54 = sitofp i32 %32 to double, !dbg !2571
  %div55 = fdiv double 1.400000e+04, %conv54, !dbg !2572
  %33 = call double @llvm.ceil.f64(double %div55), !dbg !2573
  %conv56 = fptosi double %33 to i32, !dbg !2574
  store i32 %conv56, i32* @blocks_per_grid_on_kernel_seven, align 4, !dbg !2575
  store i32 14000, i32* @blocks_per_grid_on_kernel_eight, align 4, !dbg !2576
  %34 = load i32, i32* @threads_per_block_on_kernel_nine, align 4, !dbg !2577
  %conv57 = sitofp i32 %34 to double, !dbg !2577
  %div58 = fdiv double 1.400000e+04, %conv57, !dbg !2578
  %35 = call double @llvm.ceil.f64(double %div58), !dbg !2579
  %conv59 = fptosi double %35 to i32, !dbg !2580
  store i32 %conv59, i32* @blocks_per_grid_on_kernel_nine, align 4, !dbg !2581
  %36 = load i32, i32* @threads_per_block_on_kernel_ten, align 4, !dbg !2582
  %conv60 = sitofp i32 %36 to double, !dbg !2582
  %div61 = fdiv double 1.400000e+04, %conv60, !dbg !2583
  %37 = call double @llvm.ceil.f64(double %div61), !dbg !2584
  %conv62 = fptosi double %37 to i32, !dbg !2585
  store i32 %conv62, i32* @blocks_per_grid_on_kernel_ten, align 4, !dbg !2586
  %38 = load i32, i32* @threads_per_block_on_kernel_eleven, align 4, !dbg !2587
  %conv63 = sitofp i32 %38 to double, !dbg !2587
  %div64 = fdiv double 1.400000e+04, %conv63, !dbg !2588
  %39 = call double @llvm.ceil.f64(double %div64), !dbg !2589
  %conv65 = fptosi double %39 to i32, !dbg !2590
  store i32 %conv65, i32* @blocks_per_grid_on_kernel_eleven, align 4, !dbg !2591
  %40 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !2592
  %conv66 = sitofp i32 %40 to double, !dbg !2593
  %div67 = fdiv double 1.400000e+04, %conv66, !dbg !2594
  %41 = call double @llvm.ceil.f64(double %div67), !dbg !2595
  %conv68 = fptoui double %41 to i64, !dbg !2595
  store i64 %conv68, i64* @global_data_elements, align 8, !dbg !2596
  %42 = load i64, i64* @global_data_elements, align 8, !dbg !2597
  %mul = mul i64 %42, 8, !dbg !2598
  store i64 %mul, i64* @size_global_data, align 8, !dbg !2599
  store i64 8064000, i64* @size_colidx_device, align 8, !dbg !2600
  store i64 56004, i64* @size_rowstr_device, align 8, !dbg !2601
  store i64 56000, i64* @size_iv_device, align 8, !dbg !2602
  store i64 56000, i64* @size_arow_device, align 8, !dbg !2603
  store i64 672000, i64* @size_acol_device, align 8, !dbg !2604
  store i64 1344000, i64* @size_aelt_device, align 8, !dbg !2605
  store i64 16128000, i64* @size_a_device, align 8, !dbg !2606
  store i64 112016, i64* @size_x_device, align 8, !dbg !2607
  store i64 112016, i64* @size_z_device, align 8, !dbg !2608
  store i64 112016, i64* @size_p_device, align 8, !dbg !2609
  store i64 112016, i64* @size_q_device, align 8, !dbg !2610
  store i64 112016, i64* @size_r_device, align 8, !dbg !2611
  store i64 8, i64* @size_rho_device, align 8, !dbg !2612
  store i64 8, i64* @size_d_device, align 8, !dbg !2613
  store i64 8, i64* @size_alpha_device, align 8, !dbg !2614
  store i64 8, i64* @size_beta_device, align 8, !dbg !2615
  store i64 8, i64* @size_sum_device, align 8, !dbg !2616
  store i64 8, i64* @size_norm_temp1_device, align 8, !dbg !2617
  store i64 8, i64* @size_norm_temp2_device, align 8, !dbg !2618
  %43 = load i64, i64* @size_global_data, align 8, !dbg !2619
  %call = call noalias i8* @malloc(i64 %43) #8, !dbg !2620
  %44 = bitcast i8* %call to double*, !dbg !2621
  store double* %44, double** @global_data, align 8, !dbg !2622
  %45 = load i64, i64* @size_global_data, align 8, !dbg !2623
  %call69 = call noalias i8* @malloc(i64 %45) #8, !dbg !2624
  %46 = bitcast i8* %call69 to double*, !dbg !2625
  store double* %46, double** @global_data_two, align 8, !dbg !2626
  %47 = load i64, i64* @size_colidx_device, align 8, !dbg !2627
  %call70 = call i32 @_ZL10cudaMallocIiE9cudaErrorPPT_m(i32** @colidx_device, i64 %47), !dbg !2628
  %48 = load i64, i64* @size_rowstr_device, align 8, !dbg !2629
  %call71 = call i32 @_ZL10cudaMallocIiE9cudaErrorPPT_m(i32** @rowstr_device, i64 %48), !dbg !2630
  %49 = load i64, i64* @size_a_device, align 8, !dbg !2631
  %call72 = call i32 @_ZL10cudaMallocIdE9cudaErrorPPT_m(double** @a_device, i64 %49), !dbg !2632
  %50 = load i64, i64* @size_p_device, align 8, !dbg !2633
  %call73 = call i32 @_ZL10cudaMallocIdE9cudaErrorPPT_m(double** @p_device, i64 %50), !dbg !2634
  %51 = load i64, i64* @size_q_device, align 8, !dbg !2635
  %call74 = call i32 @_ZL10cudaMallocIdE9cudaErrorPPT_m(double** @q_device, i64 %51), !dbg !2636
  %52 = load i64, i64* @size_r_device, align 8, !dbg !2637
  %call75 = call i32 @_ZL10cudaMallocIdE9cudaErrorPPT_m(double** @r_device, i64 %52), !dbg !2638
  %53 = load i64, i64* @size_x_device, align 8, !dbg !2639
  %call76 = call i32 @_ZL10cudaMallocIdE9cudaErrorPPT_m(double** @x_device, i64 %53), !dbg !2640
  %54 = load i64, i64* @size_z_device, align 8, !dbg !2641
  %call77 = call i32 @_ZL10cudaMallocIdE9cudaErrorPPT_m(double** @z_device, i64 %54), !dbg !2642
  %55 = load i64, i64* @size_rho_device, align 8, !dbg !2643
  %call78 = call i32 @_ZL10cudaMallocIdE9cudaErrorPPT_m(double** @rho_device, i64 %55), !dbg !2644
  %56 = load i64, i64* @size_d_device, align 8, !dbg !2645
  %call79 = call i32 @_ZL10cudaMallocIdE9cudaErrorPPT_m(double** @d_device, i64 %56), !dbg !2646
  %57 = load i64, i64* @size_alpha_device, align 8, !dbg !2647
  %call80 = call i32 @_ZL10cudaMallocIdE9cudaErrorPPT_m(double** @alpha_device, i64 %57), !dbg !2648
  %58 = load i64, i64* @size_beta_device, align 8, !dbg !2649
  %call81 = call i32 @_ZL10cudaMallocIdE9cudaErrorPPT_m(double** @beta_device, i64 %58), !dbg !2650
  %59 = load i64, i64* @size_sum_device, align 8, !dbg !2651
  %call82 = call i32 @_ZL10cudaMallocIdE9cudaErrorPPT_m(double** @sum_device, i64 %59), !dbg !2652
  %60 = load i64, i64* @size_norm_temp1_device, align 8, !dbg !2653
  %call83 = call i32 @_ZL10cudaMallocIdE9cudaErrorPPT_m(double** @norm_temp1_device, i64 %60), !dbg !2654
  %61 = load i64, i64* @size_norm_temp2_device, align 8, !dbg !2655
  %call84 = call i32 @_ZL10cudaMallocIdE9cudaErrorPPT_m(double** @norm_temp2_device, i64 %61), !dbg !2656
  %62 = load i64, i64* @size_global_data, align 8, !dbg !2657
  %call85 = call i32 @_ZL10cudaMallocIdE9cudaErrorPPT_m(double** @global_data_device, i64 %62), !dbg !2658
  %63 = load i64, i64* @size_global_data, align 8, !dbg !2659
  %call86 = call i32 @_ZL10cudaMallocIdE9cudaErrorPPT_m(double** @global_data_two_device, i64 %63), !dbg !2660
  %64 = load i32*, i32** @colidx_device, align 8, !dbg !2661
  %65 = bitcast i32* %64 to i8*, !dbg !2661
  %66 = load i32*, i32** @_ZL6colidx, align 8, !dbg !2662
  %67 = bitcast i32* %66 to i8*, !dbg !2662
  %68 = load i64, i64* @size_colidx_device, align 8, !dbg !2663
  %call87 = call i32 @cudaMemcpy(i8* %65, i8* %67, i64 %68, i32 1), !dbg !2664
  %69 = load i32*, i32** @rowstr_device, align 8, !dbg !2665
  %70 = bitcast i32* %69 to i8*, !dbg !2665
  %71 = load i32*, i32** @_ZL6rowstr, align 8, !dbg !2666
  %72 = bitcast i32* %71 to i8*, !dbg !2666
  %73 = load i64, i64* @size_rowstr_device, align 8, !dbg !2667
  %call88 = call i32 @cudaMemcpy(i8* %70, i8* %72, i64 %73, i32 1), !dbg !2668
  %74 = load double*, double** @a_device, align 8, !dbg !2669
  %75 = bitcast double* %74 to i8*, !dbg !2669
  %76 = load double*, double** @_ZL1a, align 8, !dbg !2670
  %77 = bitcast double* %76 to i8*, !dbg !2670
  %78 = load i64, i64* @size_a_device, align 8, !dbg !2671
  %call89 = call i32 @cudaMemcpy(i8* %75, i8* %77, i64 %78, i32 1), !dbg !2672
  %79 = load double*, double** @p_device, align 8, !dbg !2673
  %80 = bitcast double* %79 to i8*, !dbg !2673
  %81 = load double*, double** @_ZL1p, align 8, !dbg !2674
  %82 = bitcast double* %81 to i8*, !dbg !2674
  %83 = load i64, i64* @size_p_device, align 8, !dbg !2675
  %call90 = call i32 @cudaMemcpy(i8* %80, i8* %82, i64 %83, i32 1), !dbg !2676
  %84 = load double*, double** @q_device, align 8, !dbg !2677
  %85 = bitcast double* %84 to i8*, !dbg !2677
  %86 = load double*, double** @_ZL1q, align 8, !dbg !2678
  %87 = bitcast double* %86 to i8*, !dbg !2678
  %88 = load i64, i64* @size_q_device, align 8, !dbg !2679
  %call91 = call i32 @cudaMemcpy(i8* %85, i8* %87, i64 %88, i32 1), !dbg !2680
  %89 = load double*, double** @r_device, align 8, !dbg !2681
  %90 = bitcast double* %89 to i8*, !dbg !2681
  %91 = load double*, double** @_ZL1r, align 8, !dbg !2682
  %92 = bitcast double* %91 to i8*, !dbg !2682
  %93 = load i64, i64* @size_r_device, align 8, !dbg !2683
  %call92 = call i32 @cudaMemcpy(i8* %90, i8* %92, i64 %93, i32 1), !dbg !2684
  %94 = load double*, double** @x_device, align 8, !dbg !2685
  %95 = bitcast double* %94 to i8*, !dbg !2685
  %96 = load double*, double** @_ZL1x, align 8, !dbg !2686
  %97 = bitcast double* %96 to i8*, !dbg !2686
  %98 = load i64, i64* @size_x_device, align 8, !dbg !2687
  %call93 = call i32 @cudaMemcpy(i8* %95, i8* %97, i64 %98, i32 1), !dbg !2688
  %99 = load double*, double** @z_device, align 8, !dbg !2689
  %100 = bitcast double* %99 to i8*, !dbg !2689
  %101 = load double*, double** @_ZL1z, align 8, !dbg !2690
  %102 = bitcast double* %101 to i8*, !dbg !2690
  %103 = load i64, i64* @size_z_device, align 8, !dbg !2691
  %call94 = call i32 @cudaMemcpy(i8* %100, i8* %102, i64 %103, i32 1), !dbg !2692
  %104 = load i32, i32* @threads_per_block_on_kernel_one, align 4, !dbg !2693
  %conv95 = sext i32 %104 to i64, !dbg !2693
  %mul96 = mul i64 %conv95, 8, !dbg !2694
  store i64 %mul96, i64* @size_shared_data_on_kernel_one, align 8, !dbg !2695
  %105 = load i32, i32* @threads_per_block_on_kernel_two, align 4, !dbg !2696
  %conv97 = sext i32 %105 to i64, !dbg !2696
  %mul98 = mul i64 %conv97, 8, !dbg !2697
  store i64 %mul98, i64* @size_shared_data_on_kernel_two, align 8, !dbg !2698
  %106 = load i32, i32* @threads_per_block_on_kernel_three, align 4, !dbg !2699
  %conv99 = sext i32 %106 to i64, !dbg !2699
  %mul100 = mul i64 %conv99, 8, !dbg !2700
  store i64 %mul100, i64* @size_shared_data_on_kernel_three, align 8, !dbg !2701
  %107 = load i32, i32* @threads_per_block_on_kernel_four, align 4, !dbg !2702
  %conv101 = sext i32 %107 to i64, !dbg !2702
  %mul102 = mul i64 %conv101, 8, !dbg !2703
  store i64 %mul102, i64* @size_shared_data_on_kernel_four, align 8, !dbg !2704
  %108 = load i32, i32* @threads_per_block_on_kernel_five, align 4, !dbg !2705
  %conv103 = sext i32 %108 to i64, !dbg !2705
  %mul104 = mul i64 %conv103, 8, !dbg !2706
  store i64 %mul104, i64* @size_shared_data_on_kernel_five, align 8, !dbg !2707
  %109 = load i32, i32* @threads_per_block_on_kernel_six, align 4, !dbg !2708
  %conv105 = sext i32 %109 to i64, !dbg !2708
  %mul106 = mul i64 %conv105, 8, !dbg !2709
  store i64 %mul106, i64* @size_shared_data_on_kernel_six, align 8, !dbg !2710
  %110 = load i32, i32* @threads_per_block_on_kernel_seven, align 4, !dbg !2711
  %conv107 = sext i32 %110 to i64, !dbg !2711
  %mul108 = mul i64 %conv107, 8, !dbg !2712
  store i64 %mul108, i64* @size_shared_data_on_kernel_seven, align 8, !dbg !2713
  %111 = load i32, i32* @threads_per_block_on_kernel_eight, align 4, !dbg !2714
  %conv109 = sext i32 %111 to i64, !dbg !2714
  %mul110 = mul i64 %conv109, 8, !dbg !2715
  store i64 %mul110, i64* @size_shared_data_on_kernel_eight, align 8, !dbg !2716
  %112 = load i32, i32* @threads_per_block_on_kernel_nine, align 4, !dbg !2717
  %conv111 = sext i32 %112 to i64, !dbg !2717
  %mul112 = mul i64 %conv111, 8, !dbg !2718
  store i64 %mul112, i64* @size_shared_data_on_kernel_nine, align 8, !dbg !2719
  %113 = load i32, i32* @threads_per_block_on_kernel_ten, align 4, !dbg !2720
  %conv113 = sext i32 %113 to i64, !dbg !2720
  %mul114 = mul i64 %conv113, 8, !dbg !2721
  store i64 %mul114, i64* @size_shared_data_on_kernel_ten, align 8, !dbg !2722
  %114 = load i32, i32* @threads_per_block_on_kernel_eleven, align 4, !dbg !2723
  %conv115 = sext i32 %114 to i64, !dbg !2723
  %mul116 = mul i64 %conv115, 8, !dbg !2724
  store i64 %mul116, i64* @size_shared_data_on_kernel_eleven, align 8, !dbg !2725
  %115 = load i32, i32* @blocks_per_grid_on_kernel_one, align 4, !dbg !2726
  %conv117 = sext i32 %115 to i64, !dbg !2726
  %mul118 = mul i64 %conv117, 8, !dbg !2727
  store i64 %mul118, i64* @size_reduce_memory_on_kernel_one, align 8, !dbg !2728
  %116 = load i32, i32* @blocks_per_grid_on_kernel_two, align 4, !dbg !2729
  %conv119 = sext i32 %116 to i64, !dbg !2729
  %mul120 = mul i64 %conv119, 8, !dbg !2730
  store i64 %mul120, i64* @size_reduce_memory_on_kernel_two, align 8, !dbg !2731
  %117 = load i32, i32* @blocks_per_grid_on_kernel_three, align 4, !dbg !2732
  %conv121 = sext i32 %117 to i64, !dbg !2732
  %mul122 = mul i64 %conv121, 8, !dbg !2733
  store i64 %mul122, i64* @size_reduce_memory_on_kernel_three, align 8, !dbg !2734
  %118 = load i32, i32* @blocks_per_grid_on_kernel_four, align 4, !dbg !2735
  %conv123 = sext i32 %118 to i64, !dbg !2735
  %mul124 = mul i64 %conv123, 8, !dbg !2736
  store i64 %mul124, i64* @size_reduce_memory_on_kernel_four, align 8, !dbg !2737
  %119 = load i32, i32* @blocks_per_grid_on_kernel_five, align 4, !dbg !2738
  %conv125 = sext i32 %119 to i64, !dbg !2738
  %mul126 = mul i64 %conv125, 8, !dbg !2739
  store i64 %mul126, i64* @size_reduce_memory_on_kernel_five, align 8, !dbg !2740
  %120 = load i32, i32* @blocks_per_grid_on_kernel_six, align 4, !dbg !2741
  %conv127 = sext i32 %120 to i64, !dbg !2741
  %mul128 = mul i64 %conv127, 8, !dbg !2742
  store i64 %mul128, i64* @size_reduce_memory_on_kernel_six, align 8, !dbg !2743
  %121 = load i32, i32* @blocks_per_grid_on_kernel_seven, align 4, !dbg !2744
  %conv129 = sext i32 %121 to i64, !dbg !2744
  %mul130 = mul i64 %conv129, 8, !dbg !2745
  store i64 %mul130, i64* @size_reduce_memory_on_kernel_seven, align 8, !dbg !2746
  %122 = load i32, i32* @blocks_per_grid_on_kernel_eight, align 4, !dbg !2747
  %conv131 = sext i32 %122 to i64, !dbg !2747
  %mul132 = mul i64 %conv131, 8, !dbg !2748
  store i64 %mul132, i64* @size_reduce_memory_on_kernel_eight, align 8, !dbg !2749
  %123 = load i32, i32* @blocks_per_grid_on_kernel_nine, align 4, !dbg !2750
  %conv133 = sext i32 %123 to i64, !dbg !2750
  %mul134 = mul i64 %conv133, 8, !dbg !2751
  store i64 %mul134, i64* @size_reduce_memory_on_kernel_nine, align 8, !dbg !2752
  %124 = load i32, i32* @blocks_per_grid_on_kernel_ten, align 4, !dbg !2753
  %conv135 = sext i32 %124 to i64, !dbg !2753
  %mul136 = mul i64 %conv135, 8, !dbg !2754
  store i64 %mul136, i64* @size_reduce_memory_on_kernel_ten, align 8, !dbg !2755
  %125 = load i32, i32* @blocks_per_grid_on_kernel_eleven, align 4, !dbg !2756
  %conv137 = sext i32 %125 to i64, !dbg !2756
  %mul138 = mul i64 %conv137, 8, !dbg !2757
  store i64 %mul138, i64* @size_reduce_memory_on_kernel_eleven, align 8, !dbg !2758
  ret void, !dbg !2759
}

; Function Attrs: noinline uwtable
define internal void @_ZL13conj_grad_gpuPd(double* %rnorm) #2 !dbg !2760 {
entry:
  %rnorm.addr = alloca double*, align 8
  %d = alloca double, align 8
  %sum = alloca double, align 8
  %rho = alloca double, align 8
  %rho0 = alloca double, align 8
  %alpha = alloca double, align 8
  %beta = alloca double, align 8
  %cgit = alloca i32, align 4
  %cgitmax = alloca i32, align 4
  store double* %rnorm, double** %rnorm.addr, align 8
  call void @llvm.dbg.declare(metadata double** %rnorm.addr, metadata !2763, metadata !DIExpression()), !dbg !2764
  call void @llvm.dbg.declare(metadata double* %d, metadata !2765, metadata !DIExpression()), !dbg !2766
  call void @llvm.dbg.declare(metadata double* %sum, metadata !2767, metadata !DIExpression()), !dbg !2768
  call void @llvm.dbg.declare(metadata double* %rho, metadata !2769, metadata !DIExpression()), !dbg !2770
  call void @llvm.dbg.declare(metadata double* %rho0, metadata !2771, metadata !DIExpression()), !dbg !2772
  call void @llvm.dbg.declare(metadata double* %alpha, metadata !2773, metadata !DIExpression()), !dbg !2774
  call void @llvm.dbg.declare(metadata double* %beta, metadata !2775, metadata !DIExpression()), !dbg !2776
  call void @llvm.dbg.declare(metadata i32* %cgit, metadata !2777, metadata !DIExpression()), !dbg !2778
  call void @llvm.dbg.declare(metadata i32* %cgitmax, metadata !2779, metadata !DIExpression()), !dbg !2780
  store i32 25, i32* %cgitmax, align 4, !dbg !2780
  call void @_ZL19gpu_kernel_one_hostv(), !dbg !2781
  call void @_ZL19gpu_kernel_two_hostPd(double* %rho), !dbg !2782
  store i32 1, i32* %cgit, align 4, !dbg !2783
  br label %for.cond, !dbg !2785

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, i32* %cgit, align 4, !dbg !2786
  %1 = load i32, i32* %cgitmax, align 4, !dbg !2788
  %cmp = icmp sle i32 %0, %1, !dbg !2789
  br i1 %cmp, label %for.body, label %for.end, !dbg !2790

for.body:                                         ; preds = %for.cond
  call void @_ZL21gpu_kernel_three_hostv(), !dbg !2791
  call void @_ZL20gpu_kernel_four_hostPd(double* %d), !dbg !2793
  %2 = load double, double* %rho, align 8, !dbg !2794
  %3 = load double, double* %d, align 8, !dbg !2795
  %div = fdiv double %2, %3, !dbg !2796
  store double %div, double* %alpha, align 8, !dbg !2797
  %4 = load double, double* %rho, align 8, !dbg !2798
  store double %4, double* %rho0, align 8, !dbg !2799
  %5 = load double, double* %alpha, align 8, !dbg !2800
  call void @_ZL20gpu_kernel_five_hostd(double %5), !dbg !2801
  call void @_ZL19gpu_kernel_six_hostPd(double* %rho), !dbg !2802
  %6 = load double, double* %rho, align 8, !dbg !2803
  %7 = load double, double* %rho0, align 8, !dbg !2804
  %div1 = fdiv double %6, %7, !dbg !2805
  store double %div1, double* %beta, align 8, !dbg !2806
  %8 = load double, double* %beta, align 8, !dbg !2807
  call void @_ZL21gpu_kernel_seven_hostd(double %8), !dbg !2808
  br label %for.inc, !dbg !2809

for.inc:                                          ; preds = %for.body
  %9 = load i32, i32* %cgit, align 4, !dbg !2810
  %inc = add nsw i32 %9, 1, !dbg !2810
  store i32 %inc, i32* %cgit, align 4, !dbg !2810
  br label %for.cond, !dbg !2811, !llvm.loop !2812

for.end:                                          ; preds = %for.cond
  call void @_ZL21gpu_kernel_eight_hostv(), !dbg !2814
  call void @_ZL20gpu_kernel_nine_hostPd(double* %sum), !dbg !2815
  %10 = load double, double* %sum, align 8, !dbg !2816
  %call = call double @sqrt(double %10) #8, !dbg !2817
  %11 = load double*, double** %rnorm.addr, align 8, !dbg !2818
  store double %call, double* %11, align 8, !dbg !2819
  ret void, !dbg !2820
}

; Function Attrs: noinline uwtable
define internal void @_ZL19gpu_kernel_ten_hostPdS_(double* %norm_temp1, double* %norm_temp2) #2 !dbg !2821 {
entry:
  %norm_temp1.addr = alloca double*, align 8
  %norm_temp2.addr = alloca double*, align 8
  %agg.tmp = alloca %struct.dim3, align 4
  %agg.tmp1 = alloca %struct.dim3, align 4
  %agg.tmp.coerce = alloca { i64, i32 }, align 4
  %agg.tmp1.coerce = alloca { i64, i32 }, align 4
  %agg.tmp2 = alloca %struct.dim3, align 4
  %agg.tmp3 = alloca %struct.dim3, align 4
  %agg.tmp2.coerce = alloca { i64, i32 }, align 4
  %agg.tmp3.coerce = alloca { i64, i32 }, align 4
  %i = alloca i32, align 4
  store double* %norm_temp1, double** %norm_temp1.addr, align 8
  call void @llvm.dbg.declare(metadata double** %norm_temp1.addr, metadata !2824, metadata !DIExpression()), !dbg !2825
  store double* %norm_temp2, double** %norm_temp2.addr, align 8
  call void @llvm.dbg.declare(metadata double** %norm_temp2.addr, metadata !2826, metadata !DIExpression()), !dbg !2827
  %0 = load i32, i32* @blocks_per_grid_on_kernel_ten, align 4, !dbg !2828
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp, i32 %0, i32 1, i32 1), !dbg !2828
  %1 = load i32, i32* @threads_per_block_on_kernel_ten, align 4, !dbg !2829
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp1, i32 %1, i32 1, i32 1), !dbg !2829
  %2 = load i64, i64* @size_shared_data_on_kernel_ten, align 8, !dbg !2830
  %3 = bitcast { i64, i32 }* %agg.tmp.coerce to i8*, !dbg !2831
  %4 = bitcast %struct.dim3* %agg.tmp to i8*, !dbg !2831
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %3, i8* align 4 %4, i64 12, i1 false), !dbg !2831
  %5 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 0, !dbg !2831
  %6 = load i64, i64* %5, align 4, !dbg !2831
  %7 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 1, !dbg !2831
  %8 = load i32, i32* %7, align 4, !dbg !2831
  %9 = bitcast { i64, i32 }* %agg.tmp1.coerce to i8*, !dbg !2831
  %10 = bitcast %struct.dim3* %agg.tmp1 to i8*, !dbg !2831
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %9, i8* align 4 %10, i64 12, i1 false), !dbg !2831
  %11 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp1.coerce, i32 0, i32 0, !dbg !2831
  %12 = load i64, i64* %11, align 4, !dbg !2831
  %13 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp1.coerce, i32 0, i32 1, !dbg !2831
  %14 = load i32, i32* %13, align 4, !dbg !2831
  %call = call i32 @cudaConfigureCall(i64 %6, i32 %8, i64 %12, i32 %14, i64 %2, %struct.CUstream_st* null), !dbg !2831
  %tobool = icmp ne i32 %call, 0, !dbg !2831
  br i1 %tobool, label %kcall.end, label %kcall.configok, !dbg !2832

kcall.configok:                                   ; preds = %entry
  %15 = load double*, double** @global_data_device, align 8, !dbg !2833
  %16 = load double*, double** @x_device, align 8, !dbg !2834
  %17 = load double*, double** @z_device, align 8, !dbg !2835
  call void @_Z16gpu_kernel_ten_1PdS_S_(double* %15, double* %16, double* %17), !dbg !2832
  br label %kcall.end, !dbg !2832

kcall.end:                                        ; preds = %kcall.configok, %entry
  %18 = load i32, i32* @blocks_per_grid_on_kernel_ten, align 4, !dbg !2836
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp2, i32 %18, i32 1, i32 1), !dbg !2836
  %19 = load i32, i32* @threads_per_block_on_kernel_ten, align 4, !dbg !2837
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp3, i32 %19, i32 1, i32 1), !dbg !2837
  %20 = load i64, i64* @size_shared_data_on_kernel_ten, align 8, !dbg !2838
  %21 = bitcast { i64, i32 }* %agg.tmp2.coerce to i8*, !dbg !2839
  %22 = bitcast %struct.dim3* %agg.tmp2 to i8*, !dbg !2839
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %21, i8* align 4 %22, i64 12, i1 false), !dbg !2839
  %23 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp2.coerce, i32 0, i32 0, !dbg !2839
  %24 = load i64, i64* %23, align 4, !dbg !2839
  %25 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp2.coerce, i32 0, i32 1, !dbg !2839
  %26 = load i32, i32* %25, align 4, !dbg !2839
  %27 = bitcast { i64, i32 }* %agg.tmp3.coerce to i8*, !dbg !2839
  %28 = bitcast %struct.dim3* %agg.tmp3 to i8*, !dbg !2839
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %27, i8* align 4 %28, i64 12, i1 false), !dbg !2839
  %29 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp3.coerce, i32 0, i32 0, !dbg !2839
  %30 = load i64, i64* %29, align 4, !dbg !2839
  %31 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp3.coerce, i32 0, i32 1, !dbg !2839
  %32 = load i32, i32* %31, align 4, !dbg !2839
  %call4 = call i32 @cudaConfigureCall(i64 %24, i32 %26, i64 %30, i32 %32, i64 %20, %struct.CUstream_st* null), !dbg !2839
  %tobool5 = icmp ne i32 %call4, 0, !dbg !2839
  br i1 %tobool5, label %kcall.end7, label %kcall.configok6, !dbg !2840

kcall.configok6:                                  ; preds = %kcall.end
  %33 = load double*, double** @global_data_two_device, align 8, !dbg !2841
  %34 = load double*, double** @x_device, align 8, !dbg !2842
  %35 = load double*, double** @z_device, align 8, !dbg !2843
  call void @_Z16gpu_kernel_ten_2PdS_S_(double* %33, double* %34, double* %35), !dbg !2840
  br label %kcall.end7, !dbg !2840

kcall.end7:                                       ; preds = %kcall.configok6, %kcall.end
  store double 0.000000e+00, double* @global_data_reduce, align 8, !dbg !2844
  store double 0.000000e+00, double* @global_data_two_reduce, align 8, !dbg !2845
  %36 = load double*, double** @global_data, align 8, !dbg !2846
  %37 = bitcast double* %36 to i8*, !dbg !2846
  %38 = load double*, double** @global_data_device, align 8, !dbg !2847
  %39 = bitcast double* %38 to i8*, !dbg !2847
  %40 = load i64, i64* @size_reduce_memory_on_kernel_ten, align 8, !dbg !2848
  %call8 = call i32 @cudaMemcpy(i8* %37, i8* %39, i64 %40, i32 2), !dbg !2849
  %41 = load double*, double** @global_data_two, align 8, !dbg !2850
  %42 = bitcast double* %41 to i8*, !dbg !2850
  %43 = load double*, double** @global_data_two_device, align 8, !dbg !2851
  %44 = bitcast double* %43 to i8*, !dbg !2851
  %45 = load i64, i64* @size_reduce_memory_on_kernel_ten, align 8, !dbg !2852
  %call9 = call i32 @cudaMemcpy(i8* %42, i8* %44, i64 %45, i32 2), !dbg !2853
  call void @llvm.dbg.declare(metadata i32* %i, metadata !2854, metadata !DIExpression()), !dbg !2856
  store i32 0, i32* %i, align 4, !dbg !2856
  br label %for.cond, !dbg !2857

for.cond:                                         ; preds = %for.inc, %kcall.end7
  %46 = load i32, i32* %i, align 4, !dbg !2858
  %47 = load i32, i32* @blocks_per_grid_on_kernel_ten, align 4, !dbg !2860
  %cmp = icmp slt i32 %46, %47, !dbg !2861
  br i1 %cmp, label %for.body, label %for.end, !dbg !2862

for.body:                                         ; preds = %for.cond
  %48 = load double*, double** @global_data, align 8, !dbg !2863
  %49 = load i32, i32* %i, align 4, !dbg !2865
  %idxprom = sext i32 %49 to i64, !dbg !2863
  %arrayidx = getelementptr inbounds double, double* %48, i64 %idxprom, !dbg !2863
  %50 = load double, double* %arrayidx, align 8, !dbg !2863
  %51 = load double, double* @global_data_reduce, align 8, !dbg !2866
  %add = fadd contract double %51, %50, !dbg !2866
  store double %add, double* @global_data_reduce, align 8, !dbg !2866
  %52 = load double*, double** @global_data_two, align 8, !dbg !2867
  %53 = load i32, i32* %i, align 4, !dbg !2868
  %idxprom10 = sext i32 %53 to i64, !dbg !2867
  %arrayidx11 = getelementptr inbounds double, double* %52, i64 %idxprom10, !dbg !2867
  %54 = load double, double* %arrayidx11, align 8, !dbg !2867
  %55 = load double, double* @global_data_two_reduce, align 8, !dbg !2869
  %add12 = fadd contract double %55, %54, !dbg !2869
  store double %add12, double* @global_data_two_reduce, align 8, !dbg !2869
  br label %for.inc, !dbg !2870

for.inc:                                          ; preds = %for.body
  %56 = load i32, i32* %i, align 4, !dbg !2871
  %inc = add nsw i32 %56, 1, !dbg !2871
  store i32 %inc, i32* %i, align 4, !dbg !2871
  br label %for.cond, !dbg !2872, !llvm.loop !2873

for.end:                                          ; preds = %for.cond
  %57 = load double, double* @global_data_reduce, align 8, !dbg !2875
  %58 = load double*, double** %norm_temp1.addr, align 8, !dbg !2876
  store double %57, double* %58, align 8, !dbg !2877
  %59 = load double, double* @global_data_two_reduce, align 8, !dbg !2878
  %60 = load double*, double** %norm_temp2.addr, align 8, !dbg !2879
  store double %59, double* %60, align 8, !dbg !2880
  ret void, !dbg !2881
}

; Function Attrs: noinline uwtable
define internal void @_ZL22gpu_kernel_eleven_hostd(double %norm_temp2) #2 !dbg !2882 {
entry:
  %norm_temp2.addr = alloca double, align 8
  %agg.tmp = alloca %struct.dim3, align 4
  %agg.tmp1 = alloca %struct.dim3, align 4
  %agg.tmp.coerce = alloca { i64, i32 }, align 4
  %agg.tmp1.coerce = alloca { i64, i32 }, align 4
  store double %norm_temp2, double* %norm_temp2.addr, align 8
  call void @llvm.dbg.declare(metadata double* %norm_temp2.addr, metadata !2885, metadata !DIExpression()), !dbg !2886
  %0 = load i32, i32* @blocks_per_grid_on_kernel_eleven, align 4, !dbg !2887
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp, i32 %0, i32 1, i32 1), !dbg !2887
  %1 = load i32, i32* @threads_per_block_on_kernel_eleven, align 4, !dbg !2888
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp1, i32 %1, i32 1, i32 1), !dbg !2888
  %2 = bitcast { i64, i32 }* %agg.tmp.coerce to i8*, !dbg !2889
  %3 = bitcast %struct.dim3* %agg.tmp to i8*, !dbg !2889
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %2, i8* align 4 %3, i64 12, i1 false), !dbg !2889
  %4 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 0, !dbg !2889
  %5 = load i64, i64* %4, align 4, !dbg !2889
  %6 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 1, !dbg !2889
  %7 = load i32, i32* %6, align 4, !dbg !2889
  %8 = bitcast { i64, i32 }* %agg.tmp1.coerce to i8*, !dbg !2889
  %9 = bitcast %struct.dim3* %agg.tmp1 to i8*, !dbg !2889
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %8, i8* align 4 %9, i64 12, i1 false), !dbg !2889
  %10 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp1.coerce, i32 0, i32 0, !dbg !2889
  %11 = load i64, i64* %10, align 4, !dbg !2889
  %12 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp1.coerce, i32 0, i32 1, !dbg !2889
  %13 = load i32, i32* %12, align 4, !dbg !2889
  %call = call i32 @cudaConfigureCall(i64 %5, i32 %7, i64 %11, i32 %13, i64 0, %struct.CUstream_st* null), !dbg !2889
  %tobool = icmp ne i32 %call, 0, !dbg !2889
  br i1 %tobool, label %kcall.end, label %kcall.configok, !dbg !2890

kcall.configok:                                   ; preds = %entry
  %14 = load double, double* %norm_temp2.addr, align 8, !dbg !2891
  %15 = load double*, double** @x_device, align 8, !dbg !2892
  %16 = load double*, double** @z_device, align 8, !dbg !2893
  call void @_Z24gpu_kernel_eleven_devicedPdS_(double %14, double* %15, double* %16), !dbg !2890
  br label %kcall.end, !dbg !2890

kcall.end:                                        ; preds = %kcall.configok, %entry
  ret void, !dbg !2894
}

; Function Attrs: nounwind readnone speculatable
declare double @llvm.fabs.f64(double) #1

; Function Attrs: nounwind
declare dso_local i8* @strcpy(i8*, i8*) #4

; Function Attrs: nounwind
declare dso_local i8* @strcat(i8*, i8*) #4

; Function Attrs: noinline uwtable
define internal void @_ZL11release_gpuv() #2 !dbg !2895 {
entry:
  %0 = load i32*, i32** @colidx_device, align 8, !dbg !2896
  %1 = bitcast i32* %0 to i8*, !dbg !2896
  %call = call i32 @cudaFree(i8* %1), !dbg !2897
  %2 = load i32*, i32** @rowstr_device, align 8, !dbg !2898
  %3 = bitcast i32* %2 to i8*, !dbg !2898
  %call1 = call i32 @cudaFree(i8* %3), !dbg !2899
  %4 = load double*, double** @a_device, align 8, !dbg !2900
  %5 = bitcast double* %4 to i8*, !dbg !2900
  %call2 = call i32 @cudaFree(i8* %5), !dbg !2901
  %6 = load double*, double** @p_device, align 8, !dbg !2902
  %7 = bitcast double* %6 to i8*, !dbg !2902
  %call3 = call i32 @cudaFree(i8* %7), !dbg !2903
  %8 = load double*, double** @q_device, align 8, !dbg !2904
  %9 = bitcast double* %8 to i8*, !dbg !2904
  %call4 = call i32 @cudaFree(i8* %9), !dbg !2905
  %10 = load double*, double** @r_device, align 8, !dbg !2906
  %11 = bitcast double* %10 to i8*, !dbg !2906
  %call5 = call i32 @cudaFree(i8* %11), !dbg !2907
  %12 = load double*, double** @x_device, align 8, !dbg !2908
  %13 = bitcast double* %12 to i8*, !dbg !2908
  %call6 = call i32 @cudaFree(i8* %13), !dbg !2909
  %14 = load double*, double** @z_device, align 8, !dbg !2910
  %15 = bitcast double* %14 to i8*, !dbg !2910
  %call7 = call i32 @cudaFree(i8* %15), !dbg !2911
  %16 = load double*, double** @rho_device, align 8, !dbg !2912
  %17 = bitcast double* %16 to i8*, !dbg !2912
  %call8 = call i32 @cudaFree(i8* %17), !dbg !2913
  %18 = load double*, double** @d_device, align 8, !dbg !2914
  %19 = bitcast double* %18 to i8*, !dbg !2914
  %call9 = call i32 @cudaFree(i8* %19), !dbg !2915
  %20 = load double*, double** @alpha_device, align 8, !dbg !2916
  %21 = bitcast double* %20 to i8*, !dbg !2916
  %call10 = call i32 @cudaFree(i8* %21), !dbg !2917
  %22 = load double*, double** @beta_device, align 8, !dbg !2918
  %23 = bitcast double* %22 to i8*, !dbg !2918
  %call11 = call i32 @cudaFree(i8* %23), !dbg !2919
  %24 = load double*, double** @sum_device, align 8, !dbg !2920
  %25 = bitcast double* %24 to i8*, !dbg !2920
  %call12 = call i32 @cudaFree(i8* %25), !dbg !2921
  %26 = load double*, double** @norm_temp1_device, align 8, !dbg !2922
  %27 = bitcast double* %26 to i8*, !dbg !2922
  %call13 = call i32 @cudaFree(i8* %27), !dbg !2923
  %28 = load double*, double** @norm_temp2_device, align 8, !dbg !2924
  %29 = bitcast double* %28 to i8*, !dbg !2924
  %call14 = call i32 @cudaFree(i8* %29), !dbg !2925
  %30 = load double*, double** @global_data_device, align 8, !dbg !2926
  %31 = bitcast double* %30 to i8*, !dbg !2926
  %call15 = call i32 @cudaFree(i8* %31), !dbg !2927
  %32 = load double*, double** @global_data_two_device, align 8, !dbg !2928
  %33 = bitcast double* %32 to i8*, !dbg !2928
  %call16 = call i32 @cudaFree(i8* %33), !dbg !2929
  ret void, !dbg !2930
}

; Function Attrs: noinline uwtable
define dso_local void @_Z21gpu_kernel_one_devicePdS_S_S_S_(double* %p, double* %q, double* %r, double* %x, double* %z) #2 !dbg !2931 {
entry:
  %p.addr = alloca double*, align 8
  %q.addr = alloca double*, align 8
  %r.addr = alloca double*, align 8
  %x.addr = alloca double*, align 8
  %z.addr = alloca double*, align 8
  store double* %p, double** %p.addr, align 8
  call void @llvm.dbg.declare(metadata double** %p.addr, metadata !2934, metadata !DIExpression()), !dbg !2935
  store double* %q, double** %q.addr, align 8
  call void @llvm.dbg.declare(metadata double** %q.addr, metadata !2936, metadata !DIExpression()), !dbg !2937
  store double* %r, double** %r.addr, align 8
  call void @llvm.dbg.declare(metadata double** %r.addr, metadata !2938, metadata !DIExpression()), !dbg !2939
  store double* %x, double** %x.addr, align 8
  call void @llvm.dbg.declare(metadata double** %x.addr, metadata !2940, metadata !DIExpression()), !dbg !2941
  store double* %z, double** %z.addr, align 8
  call void @llvm.dbg.declare(metadata double** %z.addr, metadata !2942, metadata !DIExpression()), !dbg !2943
  %0 = bitcast double** %p.addr to i8*, !dbg !2944
  %1 = call i32 @cudaSetupArgument(i8* %0, i64 8, i64 0), !dbg !2944
  %2 = icmp eq i32 %1, 0, !dbg !2944
  br i1 %2, label %setup.next, label %setup.end, !dbg !2944

setup.next:                                       ; preds = %entry
  %3 = bitcast double** %q.addr to i8*, !dbg !2944
  %4 = call i32 @cudaSetupArgument(i8* %3, i64 8, i64 8), !dbg !2944
  %5 = icmp eq i32 %4, 0, !dbg !2944
  br i1 %5, label %setup.next1, label %setup.end, !dbg !2944

setup.next1:                                      ; preds = %setup.next
  %6 = bitcast double** %r.addr to i8*, !dbg !2944
  %7 = call i32 @cudaSetupArgument(i8* %6, i64 8, i64 16), !dbg !2944
  %8 = icmp eq i32 %7, 0, !dbg !2944
  br i1 %8, label %setup.next2, label %setup.end, !dbg !2944

setup.next2:                                      ; preds = %setup.next1
  %9 = bitcast double** %x.addr to i8*, !dbg !2944
  %10 = call i32 @cudaSetupArgument(i8* %9, i64 8, i64 24), !dbg !2944
  %11 = icmp eq i32 %10, 0, !dbg !2944
  br i1 %11, label %setup.next3, label %setup.end, !dbg !2944

setup.next3:                                      ; preds = %setup.next2
  %12 = bitcast double** %z.addr to i8*, !dbg !2944
  %13 = call i32 @cudaSetupArgument(i8* %12, i64 8, i64 32), !dbg !2944
  %14 = icmp eq i32 %13, 0, !dbg !2944
  br i1 %14, label %setup.next4, label %setup.end, !dbg !2944

setup.next4:                                      ; preds = %setup.next3
  %15 = call i32 @cudaLaunch(i8* bitcast (void (double*, double*, double*, double*, double*)* @_Z21gpu_kernel_one_devicePdS_S_S_S_ to i8*)), !dbg !2944
  br label %setup.end, !dbg !2944

setup.end:                                        ; preds = %setup.next4, %setup.next3, %setup.next2, %setup.next1, %setup.next, %entry
  ret void, !dbg !2945
}

declare dso_local i32 @cudaSetupArgument(i8*, i64, i64)

declare dso_local i32 @cudaLaunch(i8*)

; Function Attrs: noinline uwtable
define dso_local void @_Z21gpu_kernel_two_devicePdS_S_(double* %r, double* %rho, double* %global_data) #2 !dbg !2946 {
entry:
  %r.addr = alloca double*, align 8
  %rho.addr = alloca double*, align 8
  %global_data.addr = alloca double*, align 8
  store double* %r, double** %r.addr, align 8
  call void @llvm.dbg.declare(metadata double** %r.addr, metadata !2949, metadata !DIExpression()), !dbg !2950
  store double* %rho, double** %rho.addr, align 8
  call void @llvm.dbg.declare(metadata double** %rho.addr, metadata !2951, metadata !DIExpression()), !dbg !2952
  store double* %global_data, double** %global_data.addr, align 8
  call void @llvm.dbg.declare(metadata double** %global_data.addr, metadata !2953, metadata !DIExpression()), !dbg !2954
  %0 = bitcast double** %r.addr to i8*, !dbg !2955
  %1 = call i32 @cudaSetupArgument(i8* %0, i64 8, i64 0), !dbg !2955
  %2 = icmp eq i32 %1, 0, !dbg !2955
  br i1 %2, label %setup.next, label %setup.end, !dbg !2955

setup.next:                                       ; preds = %entry
  %3 = bitcast double** %rho.addr to i8*, !dbg !2955
  %4 = call i32 @cudaSetupArgument(i8* %3, i64 8, i64 8), !dbg !2955
  %5 = icmp eq i32 %4, 0, !dbg !2955
  br i1 %5, label %setup.next1, label %setup.end, !dbg !2955

setup.next1:                                      ; preds = %setup.next
  %6 = bitcast double** %global_data.addr to i8*, !dbg !2955
  %7 = call i32 @cudaSetupArgument(i8* %6, i64 8, i64 16), !dbg !2955
  %8 = icmp eq i32 %7, 0, !dbg !2955
  br i1 %8, label %setup.next2, label %setup.end, !dbg !2955

setup.next2:                                      ; preds = %setup.next1
  %9 = call i32 @cudaLaunch(i8* bitcast (void (double*, double*, double*)* @_Z21gpu_kernel_two_devicePdS_S_ to i8*)), !dbg !2955
  br label %setup.end, !dbg !2955

setup.end:                                        ; preds = %setup.next2, %setup.next1, %setup.next, %entry
  ret void, !dbg !2956
}

; Function Attrs: noinline uwtable
define dso_local void @_Z23gpu_kernel_three_devicePiS_PdS0_S0_(i32* %colidx, i32* %rowstr, double* %a, double* %p, double* %q) #2 !dbg !2957 {
entry:
  %colidx.addr = alloca i32*, align 8
  %rowstr.addr = alloca i32*, align 8
  %a.addr = alloca double*, align 8
  %p.addr = alloca double*, align 8
  %q.addr = alloca double*, align 8
  store i32* %colidx, i32** %colidx.addr, align 8
  call void @llvm.dbg.declare(metadata i32** %colidx.addr, metadata !2960, metadata !DIExpression()), !dbg !2961
  store i32* %rowstr, i32** %rowstr.addr, align 8
  call void @llvm.dbg.declare(metadata i32** %rowstr.addr, metadata !2962, metadata !DIExpression()), !dbg !2963
  store double* %a, double** %a.addr, align 8
  call void @llvm.dbg.declare(metadata double** %a.addr, metadata !2964, metadata !DIExpression()), !dbg !2965
  store double* %p, double** %p.addr, align 8
  call void @llvm.dbg.declare(metadata double** %p.addr, metadata !2966, metadata !DIExpression()), !dbg !2967
  store double* %q, double** %q.addr, align 8
  call void @llvm.dbg.declare(metadata double** %q.addr, metadata !2968, metadata !DIExpression()), !dbg !2969
  %0 = bitcast i32** %colidx.addr to i8*, !dbg !2970
  %1 = call i32 @cudaSetupArgument(i8* %0, i64 8, i64 0), !dbg !2970
  %2 = icmp eq i32 %1, 0, !dbg !2970
  br i1 %2, label %setup.next, label %setup.end, !dbg !2970

setup.next:                                       ; preds = %entry
  %3 = bitcast i32** %rowstr.addr to i8*, !dbg !2970
  %4 = call i32 @cudaSetupArgument(i8* %3, i64 8, i64 8), !dbg !2970
  %5 = icmp eq i32 %4, 0, !dbg !2970
  br i1 %5, label %setup.next1, label %setup.end, !dbg !2970

setup.next1:                                      ; preds = %setup.next
  %6 = bitcast double** %a.addr to i8*, !dbg !2970
  %7 = call i32 @cudaSetupArgument(i8* %6, i64 8, i64 16), !dbg !2970
  %8 = icmp eq i32 %7, 0, !dbg !2970
  br i1 %8, label %setup.next2, label %setup.end, !dbg !2970

setup.next2:                                      ; preds = %setup.next1
  %9 = bitcast double** %p.addr to i8*, !dbg !2970
  %10 = call i32 @cudaSetupArgument(i8* %9, i64 8, i64 24), !dbg !2970
  %11 = icmp eq i32 %10, 0, !dbg !2970
  br i1 %11, label %setup.next3, label %setup.end, !dbg !2970

setup.next3:                                      ; preds = %setup.next2
  %12 = bitcast double** %q.addr to i8*, !dbg !2970
  %13 = call i32 @cudaSetupArgument(i8* %12, i64 8, i64 32), !dbg !2970
  %14 = icmp eq i32 %13, 0, !dbg !2970
  br i1 %14, label %setup.next4, label %setup.end, !dbg !2970

setup.next4:                                      ; preds = %setup.next3
  %15 = call i32 @cudaLaunch(i8* bitcast (void (i32*, i32*, double*, double*, double*)* @_Z23gpu_kernel_three_devicePiS_PdS0_S0_ to i8*)), !dbg !2970
  br label %setup.end, !dbg !2970

setup.end:                                        ; preds = %setup.next4, %setup.next3, %setup.next2, %setup.next1, %setup.next, %entry
  ret void, !dbg !2971
}

; Function Attrs: noinline uwtable
define dso_local void @_Z22gpu_kernel_four_devicePdS_S_S_(double* %d, double* %p, double* %q, double* %global_data) #2 !dbg !2972 {
entry:
  %d.addr = alloca double*, align 8
  %p.addr = alloca double*, align 8
  %q.addr = alloca double*, align 8
  %global_data.addr = alloca double*, align 8
  store double* %d, double** %d.addr, align 8
  call void @llvm.dbg.declare(metadata double** %d.addr, metadata !2975, metadata !DIExpression()), !dbg !2976
  store double* %p, double** %p.addr, align 8
  call void @llvm.dbg.declare(metadata double** %p.addr, metadata !2977, metadata !DIExpression()), !dbg !2978
  store double* %q, double** %q.addr, align 8
  call void @llvm.dbg.declare(metadata double** %q.addr, metadata !2979, metadata !DIExpression()), !dbg !2980
  store double* %global_data, double** %global_data.addr, align 8
  call void @llvm.dbg.declare(metadata double** %global_data.addr, metadata !2981, metadata !DIExpression()), !dbg !2982
  %0 = bitcast double** %d.addr to i8*, !dbg !2983
  %1 = call i32 @cudaSetupArgument(i8* %0, i64 8, i64 0), !dbg !2983
  %2 = icmp eq i32 %1, 0, !dbg !2983
  br i1 %2, label %setup.next, label %setup.end, !dbg !2983

setup.next:                                       ; preds = %entry
  %3 = bitcast double** %p.addr to i8*, !dbg !2983
  %4 = call i32 @cudaSetupArgument(i8* %3, i64 8, i64 8), !dbg !2983
  %5 = icmp eq i32 %4, 0, !dbg !2983
  br i1 %5, label %setup.next1, label %setup.end, !dbg !2983

setup.next1:                                      ; preds = %setup.next
  %6 = bitcast double** %q.addr to i8*, !dbg !2983
  %7 = call i32 @cudaSetupArgument(i8* %6, i64 8, i64 16), !dbg !2983
  %8 = icmp eq i32 %7, 0, !dbg !2983
  br i1 %8, label %setup.next2, label %setup.end, !dbg !2983

setup.next2:                                      ; preds = %setup.next1
  %9 = bitcast double** %global_data.addr to i8*, !dbg !2983
  %10 = call i32 @cudaSetupArgument(i8* %9, i64 8, i64 24), !dbg !2983
  %11 = icmp eq i32 %10, 0, !dbg !2983
  br i1 %11, label %setup.next3, label %setup.end, !dbg !2983

setup.next3:                                      ; preds = %setup.next2
  %12 = call i32 @cudaLaunch(i8* bitcast (void (double*, double*, double*, double*)* @_Z22gpu_kernel_four_devicePdS_S_S_ to i8*)), !dbg !2983
  br label %setup.end, !dbg !2983

setup.end:                                        ; preds = %setup.next3, %setup.next2, %setup.next1, %setup.next, %entry
  ret void, !dbg !2984
}

; Function Attrs: noinline uwtable
define dso_local void @_Z17gpu_kernel_five_1dPdS_(double %alpha, double* %p, double* %z) #2 !dbg !2985 {
entry:
  %alpha.addr = alloca double, align 8
  %p.addr = alloca double*, align 8
  %z.addr = alloca double*, align 8
  store double %alpha, double* %alpha.addr, align 8
  call void @llvm.dbg.declare(metadata double* %alpha.addr, metadata !2988, metadata !DIExpression()), !dbg !2989
  store double* %p, double** %p.addr, align 8
  call void @llvm.dbg.declare(metadata double** %p.addr, metadata !2990, metadata !DIExpression()), !dbg !2991
  store double* %z, double** %z.addr, align 8
  call void @llvm.dbg.declare(metadata double** %z.addr, metadata !2992, metadata !DIExpression()), !dbg !2993
  %0 = bitcast double* %alpha.addr to i8*, !dbg !2994
  %1 = call i32 @cudaSetupArgument(i8* %0, i64 8, i64 0), !dbg !2994
  %2 = icmp eq i32 %1, 0, !dbg !2994
  br i1 %2, label %setup.next, label %setup.end, !dbg !2994

setup.next:                                       ; preds = %entry
  %3 = bitcast double** %p.addr to i8*, !dbg !2994
  %4 = call i32 @cudaSetupArgument(i8* %3, i64 8, i64 8), !dbg !2994
  %5 = icmp eq i32 %4, 0, !dbg !2994
  br i1 %5, label %setup.next1, label %setup.end, !dbg !2994

setup.next1:                                      ; preds = %setup.next
  %6 = bitcast double** %z.addr to i8*, !dbg !2994
  %7 = call i32 @cudaSetupArgument(i8* %6, i64 8, i64 16), !dbg !2994
  %8 = icmp eq i32 %7, 0, !dbg !2994
  br i1 %8, label %setup.next2, label %setup.end, !dbg !2994

setup.next2:                                      ; preds = %setup.next1
  %9 = call i32 @cudaLaunch(i8* bitcast (void (double, double*, double*)* @_Z17gpu_kernel_five_1dPdS_ to i8*)), !dbg !2994
  br label %setup.end, !dbg !2994

setup.end:                                        ; preds = %setup.next2, %setup.next1, %setup.next, %entry
  ret void, !dbg !2995
}

; Function Attrs: noinline uwtable
define dso_local void @_Z17gpu_kernel_five_2dPdS_(double %alpha, double* %q, double* %r) #2 !dbg !2996 {
entry:
  %alpha.addr = alloca double, align 8
  %q.addr = alloca double*, align 8
  %r.addr = alloca double*, align 8
  store double %alpha, double* %alpha.addr, align 8
  call void @llvm.dbg.declare(metadata double* %alpha.addr, metadata !2997, metadata !DIExpression()), !dbg !2998
  store double* %q, double** %q.addr, align 8
  call void @llvm.dbg.declare(metadata double** %q.addr, metadata !2999, metadata !DIExpression()), !dbg !3000
  store double* %r, double** %r.addr, align 8
  call void @llvm.dbg.declare(metadata double** %r.addr, metadata !3001, metadata !DIExpression()), !dbg !3002
  %0 = bitcast double* %alpha.addr to i8*, !dbg !3003
  %1 = call i32 @cudaSetupArgument(i8* %0, i64 8, i64 0), !dbg !3003
  %2 = icmp eq i32 %1, 0, !dbg !3003
  br i1 %2, label %setup.next, label %setup.end, !dbg !3003

setup.next:                                       ; preds = %entry
  %3 = bitcast double** %q.addr to i8*, !dbg !3003
  %4 = call i32 @cudaSetupArgument(i8* %3, i64 8, i64 8), !dbg !3003
  %5 = icmp eq i32 %4, 0, !dbg !3003
  br i1 %5, label %setup.next1, label %setup.end, !dbg !3003

setup.next1:                                      ; preds = %setup.next
  %6 = bitcast double** %r.addr to i8*, !dbg !3003
  %7 = call i32 @cudaSetupArgument(i8* %6, i64 8, i64 16), !dbg !3003
  %8 = icmp eq i32 %7, 0, !dbg !3003
  br i1 %8, label %setup.next2, label %setup.end, !dbg !3003

setup.next2:                                      ; preds = %setup.next1
  %9 = call i32 @cudaLaunch(i8* bitcast (void (double, double*, double*)* @_Z17gpu_kernel_five_2dPdS_ to i8*)), !dbg !3003
  br label %setup.end, !dbg !3003

setup.end:                                        ; preds = %setup.next2, %setup.next1, %setup.next, %entry
  ret void, !dbg !3004
}

; Function Attrs: noinline uwtable
define dso_local void @_Z21gpu_kernel_six_devicePdS_(double* %r, double* %global_data) #2 !dbg !3005 {
entry:
  %r.addr = alloca double*, align 8
  %global_data.addr = alloca double*, align 8
  store double* %r, double** %r.addr, align 8
  call void @llvm.dbg.declare(metadata double** %r.addr, metadata !3006, metadata !DIExpression()), !dbg !3007
  store double* %global_data, double** %global_data.addr, align 8
  call void @llvm.dbg.declare(metadata double** %global_data.addr, metadata !3008, metadata !DIExpression()), !dbg !3009
  %0 = bitcast double** %r.addr to i8*, !dbg !3010
  %1 = call i32 @cudaSetupArgument(i8* %0, i64 8, i64 0), !dbg !3010
  %2 = icmp eq i32 %1, 0, !dbg !3010
  br i1 %2, label %setup.next, label %setup.end, !dbg !3010

setup.next:                                       ; preds = %entry
  %3 = bitcast double** %global_data.addr to i8*, !dbg !3010
  %4 = call i32 @cudaSetupArgument(i8* %3, i64 8, i64 8), !dbg !3010
  %5 = icmp eq i32 %4, 0, !dbg !3010
  br i1 %5, label %setup.next1, label %setup.end, !dbg !3010

setup.next1:                                      ; preds = %setup.next
  %6 = call i32 @cudaLaunch(i8* bitcast (void (double*, double*)* @_Z21gpu_kernel_six_devicePdS_ to i8*)), !dbg !3010
  br label %setup.end, !dbg !3010

setup.end:                                        ; preds = %setup.next1, %setup.next, %entry
  ret void, !dbg !3011
}

; Function Attrs: noinline uwtable
define dso_local void @_Z23gpu_kernel_seven_devicedPdS_(double %beta, double* %p, double* %r) #2 !dbg !3012 {
entry:
  %beta.addr = alloca double, align 8
  %p.addr = alloca double*, align 8
  %r.addr = alloca double*, align 8
  store double %beta, double* %beta.addr, align 8
  call void @llvm.dbg.declare(metadata double* %beta.addr, metadata !3013, metadata !DIExpression()), !dbg !3014
  store double* %p, double** %p.addr, align 8
  call void @llvm.dbg.declare(metadata double** %p.addr, metadata !3015, metadata !DIExpression()), !dbg !3016
  store double* %r, double** %r.addr, align 8
  call void @llvm.dbg.declare(metadata double** %r.addr, metadata !3017, metadata !DIExpression()), !dbg !3018
  %0 = bitcast double* %beta.addr to i8*, !dbg !3019
  %1 = call i32 @cudaSetupArgument(i8* %0, i64 8, i64 0), !dbg !3019
  %2 = icmp eq i32 %1, 0, !dbg !3019
  br i1 %2, label %setup.next, label %setup.end, !dbg !3019

setup.next:                                       ; preds = %entry
  %3 = bitcast double** %p.addr to i8*, !dbg !3019
  %4 = call i32 @cudaSetupArgument(i8* %3, i64 8, i64 8), !dbg !3019
  %5 = icmp eq i32 %4, 0, !dbg !3019
  br i1 %5, label %setup.next1, label %setup.end, !dbg !3019

setup.next1:                                      ; preds = %setup.next
  %6 = bitcast double** %r.addr to i8*, !dbg !3019
  %7 = call i32 @cudaSetupArgument(i8* %6, i64 8, i64 16), !dbg !3019
  %8 = icmp eq i32 %7, 0, !dbg !3019
  br i1 %8, label %setup.next2, label %setup.end, !dbg !3019

setup.next2:                                      ; preds = %setup.next1
  %9 = call i32 @cudaLaunch(i8* bitcast (void (double, double*, double*)* @_Z23gpu_kernel_seven_devicedPdS_ to i8*)), !dbg !3019
  br label %setup.end, !dbg !3019

setup.end:                                        ; preds = %setup.next2, %setup.next1, %setup.next, %entry
  ret void, !dbg !3020
}

; Function Attrs: noinline uwtable
define dso_local void @_Z23gpu_kernel_eight_devicePiS_PdS0_S0_(i32* %colidx, i32* %rowstr, double* %a, double* %r, double* %z) #2 !dbg !3021 {
entry:
  %colidx.addr = alloca i32*, align 8
  %rowstr.addr = alloca i32*, align 8
  %a.addr = alloca double*, align 8
  %r.addr = alloca double*, align 8
  %z.addr = alloca double*, align 8
  store i32* %colidx, i32** %colidx.addr, align 8
  call void @llvm.dbg.declare(metadata i32** %colidx.addr, metadata !3022, metadata !DIExpression()), !dbg !3023
  store i32* %rowstr, i32** %rowstr.addr, align 8
  call void @llvm.dbg.declare(metadata i32** %rowstr.addr, metadata !3024, metadata !DIExpression()), !dbg !3025
  store double* %a, double** %a.addr, align 8
  call void @llvm.dbg.declare(metadata double** %a.addr, metadata !3026, metadata !DIExpression()), !dbg !3027
  store double* %r, double** %r.addr, align 8
  call void @llvm.dbg.declare(metadata double** %r.addr, metadata !3028, metadata !DIExpression()), !dbg !3029
  store double* %z, double** %z.addr, align 8
  call void @llvm.dbg.declare(metadata double** %z.addr, metadata !3030, metadata !DIExpression()), !dbg !3031
  %0 = bitcast i32** %colidx.addr to i8*, !dbg !3032
  %1 = call i32 @cudaSetupArgument(i8* %0, i64 8, i64 0), !dbg !3032
  %2 = icmp eq i32 %1, 0, !dbg !3032
  br i1 %2, label %setup.next, label %setup.end, !dbg !3032

setup.next:                                       ; preds = %entry
  %3 = bitcast i32** %rowstr.addr to i8*, !dbg !3032
  %4 = call i32 @cudaSetupArgument(i8* %3, i64 8, i64 8), !dbg !3032
  %5 = icmp eq i32 %4, 0, !dbg !3032
  br i1 %5, label %setup.next1, label %setup.end, !dbg !3032

setup.next1:                                      ; preds = %setup.next
  %6 = bitcast double** %a.addr to i8*, !dbg !3032
  %7 = call i32 @cudaSetupArgument(i8* %6, i64 8, i64 16), !dbg !3032
  %8 = icmp eq i32 %7, 0, !dbg !3032
  br i1 %8, label %setup.next2, label %setup.end, !dbg !3032

setup.next2:                                      ; preds = %setup.next1
  %9 = bitcast double** %r.addr to i8*, !dbg !3032
  %10 = call i32 @cudaSetupArgument(i8* %9, i64 8, i64 24), !dbg !3032
  %11 = icmp eq i32 %10, 0, !dbg !3032
  br i1 %11, label %setup.next3, label %setup.end, !dbg !3032

setup.next3:                                      ; preds = %setup.next2
  %12 = bitcast double** %z.addr to i8*, !dbg !3032
  %13 = call i32 @cudaSetupArgument(i8* %12, i64 8, i64 32), !dbg !3032
  %14 = icmp eq i32 %13, 0, !dbg !3032
  br i1 %14, label %setup.next4, label %setup.end, !dbg !3032

setup.next4:                                      ; preds = %setup.next3
  %15 = call i32 @cudaLaunch(i8* bitcast (void (i32*, i32*, double*, double*, double*)* @_Z23gpu_kernel_eight_devicePiS_PdS0_S0_ to i8*)), !dbg !3032
  br label %setup.end, !dbg !3032

setup.end:                                        ; preds = %setup.next4, %setup.next3, %setup.next2, %setup.next1, %setup.next, %entry
  ret void, !dbg !3033
}

; Function Attrs: noinline uwtable
define dso_local void @_Z22gpu_kernel_nine_devicePdS_S_S_(double* %r, double* %x, double* %sum, double* %global_data) #2 !dbg !3034 {
entry:
  %r.addr = alloca double*, align 8
  %x.addr = alloca double*, align 8
  %sum.addr = alloca double*, align 8
  %global_data.addr = alloca double*, align 8
  store double* %r, double** %r.addr, align 8
  call void @llvm.dbg.declare(metadata double** %r.addr, metadata !3035, metadata !DIExpression()), !dbg !3036
  store double* %x, double** %x.addr, align 8
  call void @llvm.dbg.declare(metadata double** %x.addr, metadata !3037, metadata !DIExpression()), !dbg !3038
  store double* %sum, double** %sum.addr, align 8
  call void @llvm.dbg.declare(metadata double** %sum.addr, metadata !3039, metadata !DIExpression()), !dbg !3040
  store double* %global_data, double** %global_data.addr, align 8
  call void @llvm.dbg.declare(metadata double** %global_data.addr, metadata !3041, metadata !DIExpression()), !dbg !3042
  %0 = bitcast double** %r.addr to i8*, !dbg !3043
  %1 = call i32 @cudaSetupArgument(i8* %0, i64 8, i64 0), !dbg !3043
  %2 = icmp eq i32 %1, 0, !dbg !3043
  br i1 %2, label %setup.next, label %setup.end, !dbg !3043

setup.next:                                       ; preds = %entry
  %3 = bitcast double** %x.addr to i8*, !dbg !3043
  %4 = call i32 @cudaSetupArgument(i8* %3, i64 8, i64 8), !dbg !3043
  %5 = icmp eq i32 %4, 0, !dbg !3043
  br i1 %5, label %setup.next1, label %setup.end, !dbg !3043

setup.next1:                                      ; preds = %setup.next
  %6 = bitcast double** %sum.addr to i8*, !dbg !3043
  %7 = call i32 @cudaSetupArgument(i8* %6, i64 8, i64 16), !dbg !3043
  %8 = icmp eq i32 %7, 0, !dbg !3043
  br i1 %8, label %setup.next2, label %setup.end, !dbg !3043

setup.next2:                                      ; preds = %setup.next1
  %9 = bitcast double** %global_data.addr to i8*, !dbg !3043
  %10 = call i32 @cudaSetupArgument(i8* %9, i64 8, i64 24), !dbg !3043
  %11 = icmp eq i32 %10, 0, !dbg !3043
  br i1 %11, label %setup.next3, label %setup.end, !dbg !3043

setup.next3:                                      ; preds = %setup.next2
  %12 = call i32 @cudaLaunch(i8* bitcast (void (double*, double*, double*, double*)* @_Z22gpu_kernel_nine_devicePdS_S_S_ to i8*)), !dbg !3043
  br label %setup.end, !dbg !3043

setup.end:                                        ; preds = %setup.next3, %setup.next2, %setup.next1, %setup.next, %entry
  ret void, !dbg !3044
}

; Function Attrs: noinline uwtable
define dso_local void @_Z16gpu_kernel_ten_1PdS_S_(double* %norm_temp, double* %x, double* %z) #2 !dbg !3045 {
entry:
  %norm_temp.addr = alloca double*, align 8
  %x.addr = alloca double*, align 8
  %z.addr = alloca double*, align 8
  store double* %norm_temp, double** %norm_temp.addr, align 8
  call void @llvm.dbg.declare(metadata double** %norm_temp.addr, metadata !3046, metadata !DIExpression()), !dbg !3047
  store double* %x, double** %x.addr, align 8
  call void @llvm.dbg.declare(metadata double** %x.addr, metadata !3048, metadata !DIExpression()), !dbg !3049
  store double* %z, double** %z.addr, align 8
  call void @llvm.dbg.declare(metadata double** %z.addr, metadata !3050, metadata !DIExpression()), !dbg !3051
  %0 = bitcast double** %norm_temp.addr to i8*, !dbg !3052
  %1 = call i32 @cudaSetupArgument(i8* %0, i64 8, i64 0), !dbg !3052
  %2 = icmp eq i32 %1, 0, !dbg !3052
  br i1 %2, label %setup.next, label %setup.end, !dbg !3052

setup.next:                                       ; preds = %entry
  %3 = bitcast double** %x.addr to i8*, !dbg !3052
  %4 = call i32 @cudaSetupArgument(i8* %3, i64 8, i64 8), !dbg !3052
  %5 = icmp eq i32 %4, 0, !dbg !3052
  br i1 %5, label %setup.next1, label %setup.end, !dbg !3052

setup.next1:                                      ; preds = %setup.next
  %6 = bitcast double** %z.addr to i8*, !dbg !3052
  %7 = call i32 @cudaSetupArgument(i8* %6, i64 8, i64 16), !dbg !3052
  %8 = icmp eq i32 %7, 0, !dbg !3052
  br i1 %8, label %setup.next2, label %setup.end, !dbg !3052

setup.next2:                                      ; preds = %setup.next1
  %9 = call i32 @cudaLaunch(i8* bitcast (void (double*, double*, double*)* @_Z16gpu_kernel_ten_1PdS_S_ to i8*)), !dbg !3052
  br label %setup.end, !dbg !3052

setup.end:                                        ; preds = %setup.next2, %setup.next1, %setup.next, %entry
  ret void, !dbg !3053
}

; Function Attrs: noinline uwtable
define dso_local void @_Z16gpu_kernel_ten_2PdS_S_(double* %norm_temp, double* %x, double* %z) #2 !dbg !3054 {
entry:
  %norm_temp.addr = alloca double*, align 8
  %x.addr = alloca double*, align 8
  %z.addr = alloca double*, align 8
  store double* %norm_temp, double** %norm_temp.addr, align 8
  call void @llvm.dbg.declare(metadata double** %norm_temp.addr, metadata !3055, metadata !DIExpression()), !dbg !3056
  store double* %x, double** %x.addr, align 8
  call void @llvm.dbg.declare(metadata double** %x.addr, metadata !3057, metadata !DIExpression()), !dbg !3058
  store double* %z, double** %z.addr, align 8
  call void @llvm.dbg.declare(metadata double** %z.addr, metadata !3059, metadata !DIExpression()), !dbg !3060
  %0 = bitcast double** %norm_temp.addr to i8*, !dbg !3061
  %1 = call i32 @cudaSetupArgument(i8* %0, i64 8, i64 0), !dbg !3061
  %2 = icmp eq i32 %1, 0, !dbg !3061
  br i1 %2, label %setup.next, label %setup.end, !dbg !3061

setup.next:                                       ; preds = %entry
  %3 = bitcast double** %x.addr to i8*, !dbg !3061
  %4 = call i32 @cudaSetupArgument(i8* %3, i64 8, i64 8), !dbg !3061
  %5 = icmp eq i32 %4, 0, !dbg !3061
  br i1 %5, label %setup.next1, label %setup.end, !dbg !3061

setup.next1:                                      ; preds = %setup.next
  %6 = bitcast double** %z.addr to i8*, !dbg !3061
  %7 = call i32 @cudaSetupArgument(i8* %6, i64 8, i64 16), !dbg !3061
  %8 = icmp eq i32 %7, 0, !dbg !3061
  br i1 %8, label %setup.next2, label %setup.end, !dbg !3061

setup.next2:                                      ; preds = %setup.next1
  %9 = call i32 @cudaLaunch(i8* bitcast (void (double*, double*, double*)* @_Z16gpu_kernel_ten_2PdS_S_ to i8*)), !dbg !3061
  br label %setup.end, !dbg !3061

setup.end:                                        ; preds = %setup.next2, %setup.next1, %setup.next, %entry
  ret void, !dbg !3062
}

; Function Attrs: noinline uwtable
define dso_local void @_Z24gpu_kernel_eleven_devicedPdS_(double %norm_temp2, double* %x, double* %z) #2 !dbg !3063 {
entry:
  %norm_temp2.addr = alloca double, align 8
  %x.addr = alloca double*, align 8
  %z.addr = alloca double*, align 8
  store double %norm_temp2, double* %norm_temp2.addr, align 8
  call void @llvm.dbg.declare(metadata double* %norm_temp2.addr, metadata !3064, metadata !DIExpression()), !dbg !3065
  store double* %x, double** %x.addr, align 8
  call void @llvm.dbg.declare(metadata double** %x.addr, metadata !3066, metadata !DIExpression()), !dbg !3067
  store double* %z, double** %z.addr, align 8
  call void @llvm.dbg.declare(metadata double** %z.addr, metadata !3068, metadata !DIExpression()), !dbg !3069
  %0 = bitcast double* %norm_temp2.addr to i8*, !dbg !3070
  %1 = call i32 @cudaSetupArgument(i8* %0, i64 8, i64 0), !dbg !3070
  %2 = icmp eq i32 %1, 0, !dbg !3070
  br i1 %2, label %setup.next, label %setup.end, !dbg !3070

setup.next:                                       ; preds = %entry
  %3 = bitcast double** %x.addr to i8*, !dbg !3070
  %4 = call i32 @cudaSetupArgument(i8* %3, i64 8, i64 8), !dbg !3070
  %5 = icmp eq i32 %4, 0, !dbg !3070
  br i1 %5, label %setup.next1, label %setup.end, !dbg !3070

setup.next1:                                      ; preds = %setup.next
  %6 = bitcast double** %z.addr to i8*, !dbg !3070
  %7 = call i32 @cudaSetupArgument(i8* %6, i64 8, i64 16), !dbg !3070
  %8 = icmp eq i32 %7, 0, !dbg !3070
  br i1 %8, label %setup.next2, label %setup.end, !dbg !3070

setup.next2:                                      ; preds = %setup.next1
  %9 = call i32 @cudaLaunch(i8* bitcast (void (double, double*, double*)* @_Z24gpu_kernel_eleven_devicedPdS_ to i8*)), !dbg !3070
  br label %setup.end, !dbg !3070

setup.end:                                        ; preds = %setup.next2, %setup.next1, %setup.next, %entry
  ret void, !dbg !3071
}

; Function Attrs: noinline uwtable
define internal void @_ZL19gpu_kernel_one_hostv() #2 !dbg !3072 {
entry:
  %agg.tmp = alloca %struct.dim3, align 4
  %agg.tmp1 = alloca %struct.dim3, align 4
  %agg.tmp.coerce = alloca { i64, i32 }, align 4
  %agg.tmp1.coerce = alloca { i64, i32 }, align 4
  %0 = load i32, i32* @blocks_per_grid_on_kernel_one, align 4, !dbg !3073
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp, i32 %0, i32 1, i32 1), !dbg !3073
  %1 = load i32, i32* @threads_per_block_on_kernel_one, align 4, !dbg !3074
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp1, i32 %1, i32 1, i32 1), !dbg !3074
  %2 = bitcast { i64, i32 }* %agg.tmp.coerce to i8*, !dbg !3075
  %3 = bitcast %struct.dim3* %agg.tmp to i8*, !dbg !3075
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %2, i8* align 4 %3, i64 12, i1 false), !dbg !3075
  %4 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 0, !dbg !3075
  %5 = load i64, i64* %4, align 4, !dbg !3075
  %6 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 1, !dbg !3075
  %7 = load i32, i32* %6, align 4, !dbg !3075
  %8 = bitcast { i64, i32 }* %agg.tmp1.coerce to i8*, !dbg !3075
  %9 = bitcast %struct.dim3* %agg.tmp1 to i8*, !dbg !3075
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %8, i8* align 4 %9, i64 12, i1 false), !dbg !3075
  %10 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp1.coerce, i32 0, i32 0, !dbg !3075
  %11 = load i64, i64* %10, align 4, !dbg !3075
  %12 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp1.coerce, i32 0, i32 1, !dbg !3075
  %13 = load i32, i32* %12, align 4, !dbg !3075
  %call = call i32 @cudaConfigureCall(i64 %5, i32 %7, i64 %11, i32 %13, i64 0, %struct.CUstream_st* null), !dbg !3075
  %tobool = icmp ne i32 %call, 0, !dbg !3075
  br i1 %tobool, label %kcall.end, label %kcall.configok, !dbg !3076

kcall.configok:                                   ; preds = %entry
  %14 = load double*, double** @p_device, align 8, !dbg !3077
  %15 = load double*, double** @q_device, align 8, !dbg !3078
  %16 = load double*, double** @r_device, align 8, !dbg !3079
  %17 = load double*, double** @x_device, align 8, !dbg !3080
  %18 = load double*, double** @z_device, align 8, !dbg !3081
  call void @_Z21gpu_kernel_one_devicePdS_S_S_S_(double* %14, double* %15, double* %16, double* %17, double* %18), !dbg !3076
  br label %kcall.end, !dbg !3076

kcall.end:                                        ; preds = %kcall.configok, %entry
  ret void, !dbg !3082
}

; Function Attrs: noinline uwtable
define internal void @_ZL19gpu_kernel_two_hostPd(double* %rho_host) #2 !dbg !3083 {
entry:
  %rho_host.addr = alloca double*, align 8
  %agg.tmp = alloca %struct.dim3, align 4
  %agg.tmp1 = alloca %struct.dim3, align 4
  %agg.tmp.coerce = alloca { i64, i32 }, align 4
  %agg.tmp1.coerce = alloca { i64, i32 }, align 4
  %i = alloca i32, align 4
  store double* %rho_host, double** %rho_host.addr, align 8
  call void @llvm.dbg.declare(metadata double** %rho_host.addr, metadata !3084, metadata !DIExpression()), !dbg !3085
  %0 = load i32, i32* @blocks_per_grid_on_kernel_two, align 4, !dbg !3086
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp, i32 %0, i32 1, i32 1), !dbg !3086
  %1 = load i32, i32* @threads_per_block_on_kernel_two, align 4, !dbg !3087
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp1, i32 %1, i32 1, i32 1), !dbg !3087
  %2 = load i64, i64* @size_shared_data_on_kernel_two, align 8, !dbg !3088
  %3 = bitcast { i64, i32 }* %agg.tmp.coerce to i8*, !dbg !3089
  %4 = bitcast %struct.dim3* %agg.tmp to i8*, !dbg !3089
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %3, i8* align 4 %4, i64 12, i1 false), !dbg !3089
  %5 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 0, !dbg !3089
  %6 = load i64, i64* %5, align 4, !dbg !3089
  %7 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 1, !dbg !3089
  %8 = load i32, i32* %7, align 4, !dbg !3089
  %9 = bitcast { i64, i32 }* %agg.tmp1.coerce to i8*, !dbg !3089
  %10 = bitcast %struct.dim3* %agg.tmp1 to i8*, !dbg !3089
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %9, i8* align 4 %10, i64 12, i1 false), !dbg !3089
  %11 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp1.coerce, i32 0, i32 0, !dbg !3089
  %12 = load i64, i64* %11, align 4, !dbg !3089
  %13 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp1.coerce, i32 0, i32 1, !dbg !3089
  %14 = load i32, i32* %13, align 4, !dbg !3089
  %call = call i32 @cudaConfigureCall(i64 %6, i32 %8, i64 %12, i32 %14, i64 %2, %struct.CUstream_st* null), !dbg !3089
  %tobool = icmp ne i32 %call, 0, !dbg !3089
  br i1 %tobool, label %kcall.end, label %kcall.configok, !dbg !3090

kcall.configok:                                   ; preds = %entry
  %15 = load double*, double** @r_device, align 8, !dbg !3091
  %16 = load double*, double** @rho_device, align 8, !dbg !3092
  %17 = load double*, double** @global_data_device, align 8, !dbg !3093
  call void @_Z21gpu_kernel_two_devicePdS_S_(double* %15, double* %16, double* %17), !dbg !3090
  br label %kcall.end, !dbg !3090

kcall.end:                                        ; preds = %kcall.configok, %entry
  store double 0.000000e+00, double* @global_data_reduce, align 8, !dbg !3094
  %18 = load double*, double** @global_data, align 8, !dbg !3095
  %19 = bitcast double* %18 to i8*, !dbg !3095
  %20 = load double*, double** @global_data_device, align 8, !dbg !3096
  %21 = bitcast double* %20 to i8*, !dbg !3096
  %22 = load i64, i64* @size_reduce_memory_on_kernel_two, align 8, !dbg !3097
  %call2 = call i32 @cudaMemcpy(i8* %19, i8* %21, i64 %22, i32 2), !dbg !3098
  call void @llvm.dbg.declare(metadata i32* %i, metadata !3099, metadata !DIExpression()), !dbg !3101
  store i32 0, i32* %i, align 4, !dbg !3101
  br label %for.cond, !dbg !3102

for.cond:                                         ; preds = %for.inc, %kcall.end
  %23 = load i32, i32* %i, align 4, !dbg !3103
  %24 = load i32, i32* @blocks_per_grid_on_kernel_two, align 4, !dbg !3105
  %cmp = icmp slt i32 %23, %24, !dbg !3106
  br i1 %cmp, label %for.body, label %for.end, !dbg !3107

for.body:                                         ; preds = %for.cond
  %25 = load double*, double** @global_data, align 8, !dbg !3108
  %26 = load i32, i32* %i, align 4, !dbg !3110
  %idxprom = sext i32 %26 to i64, !dbg !3108
  %arrayidx = getelementptr inbounds double, double* %25, i64 %idxprom, !dbg !3108
  %27 = load double, double* %arrayidx, align 8, !dbg !3108
  %28 = load double, double* @global_data_reduce, align 8, !dbg !3111
  %add = fadd contract double %28, %27, !dbg !3111
  store double %add, double* @global_data_reduce, align 8, !dbg !3111
  br label %for.inc, !dbg !3112

for.inc:                                          ; preds = %for.body
  %29 = load i32, i32* %i, align 4, !dbg !3113
  %inc = add nsw i32 %29, 1, !dbg !3113
  store i32 %inc, i32* %i, align 4, !dbg !3113
  br label %for.cond, !dbg !3114, !llvm.loop !3115

for.end:                                          ; preds = %for.cond
  %30 = load double, double* @global_data_reduce, align 8, !dbg !3117
  %31 = load double*, double** %rho_host.addr, align 8, !dbg !3118
  store double %30, double* %31, align 8, !dbg !3119
  ret void, !dbg !3120
}

; Function Attrs: noinline uwtable
define internal void @_ZL21gpu_kernel_three_hostv() #2 !dbg !3121 {
entry:
  %agg.tmp = alloca %struct.dim3, align 4
  %agg.tmp1 = alloca %struct.dim3, align 4
  %agg.tmp.coerce = alloca { i64, i32 }, align 4
  %agg.tmp1.coerce = alloca { i64, i32 }, align 4
  %0 = load i32, i32* @blocks_per_grid_on_kernel_three, align 4, !dbg !3122
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp, i32 %0, i32 1, i32 1), !dbg !3122
  %1 = load i32, i32* @threads_per_block_on_kernel_three, align 4, !dbg !3123
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp1, i32 %1, i32 1, i32 1), !dbg !3123
  %2 = load i64, i64* @size_shared_data_on_kernel_three, align 8, !dbg !3124
  %3 = bitcast { i64, i32 }* %agg.tmp.coerce to i8*, !dbg !3125
  %4 = bitcast %struct.dim3* %agg.tmp to i8*, !dbg !3125
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %3, i8* align 4 %4, i64 12, i1 false), !dbg !3125
  %5 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 0, !dbg !3125
  %6 = load i64, i64* %5, align 4, !dbg !3125
  %7 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 1, !dbg !3125
  %8 = load i32, i32* %7, align 4, !dbg !3125
  %9 = bitcast { i64, i32 }* %agg.tmp1.coerce to i8*, !dbg !3125
  %10 = bitcast %struct.dim3* %agg.tmp1 to i8*, !dbg !3125
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %9, i8* align 4 %10, i64 12, i1 false), !dbg !3125
  %11 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp1.coerce, i32 0, i32 0, !dbg !3125
  %12 = load i64, i64* %11, align 4, !dbg !3125
  %13 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp1.coerce, i32 0, i32 1, !dbg !3125
  %14 = load i32, i32* %13, align 4, !dbg !3125
  %call = call i32 @cudaConfigureCall(i64 %6, i32 %8, i64 %12, i32 %14, i64 %2, %struct.CUstream_st* null), !dbg !3125
  %tobool = icmp ne i32 %call, 0, !dbg !3125
  br i1 %tobool, label %kcall.end, label %kcall.configok, !dbg !3126

kcall.configok:                                   ; preds = %entry
  %15 = load i32*, i32** @colidx_device, align 8, !dbg !3127
  %16 = load i32*, i32** @rowstr_device, align 8, !dbg !3128
  %17 = load double*, double** @a_device, align 8, !dbg !3129
  %18 = load double*, double** @p_device, align 8, !dbg !3130
  %19 = load double*, double** @q_device, align 8, !dbg !3131
  call void @_Z23gpu_kernel_three_devicePiS_PdS0_S0_(i32* %15, i32* %16, double* %17, double* %18, double* %19), !dbg !3126
  br label %kcall.end, !dbg !3126

kcall.end:                                        ; preds = %kcall.configok, %entry
  ret void, !dbg !3132
}

; Function Attrs: noinline uwtable
define internal void @_ZL20gpu_kernel_four_hostPd(double* %d_host) #2 !dbg !3133 {
entry:
  %d_host.addr = alloca double*, align 8
  %agg.tmp = alloca %struct.dim3, align 4
  %agg.tmp1 = alloca %struct.dim3, align 4
  %agg.tmp.coerce = alloca { i64, i32 }, align 4
  %agg.tmp1.coerce = alloca { i64, i32 }, align 4
  %i = alloca i32, align 4
  store double* %d_host, double** %d_host.addr, align 8
  call void @llvm.dbg.declare(metadata double** %d_host.addr, metadata !3134, metadata !DIExpression()), !dbg !3135
  %0 = load i32, i32* @blocks_per_grid_on_kernel_four, align 4, !dbg !3136
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp, i32 %0, i32 1, i32 1), !dbg !3136
  %1 = load i32, i32* @threads_per_block_on_kernel_four, align 4, !dbg !3137
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp1, i32 %1, i32 1, i32 1), !dbg !3137
  %2 = load i64, i64* @size_shared_data_on_kernel_four, align 8, !dbg !3138
  %3 = bitcast { i64, i32 }* %agg.tmp.coerce to i8*, !dbg !3139
  %4 = bitcast %struct.dim3* %agg.tmp to i8*, !dbg !3139
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %3, i8* align 4 %4, i64 12, i1 false), !dbg !3139
  %5 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 0, !dbg !3139
  %6 = load i64, i64* %5, align 4, !dbg !3139
  %7 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 1, !dbg !3139
  %8 = load i32, i32* %7, align 4, !dbg !3139
  %9 = bitcast { i64, i32 }* %agg.tmp1.coerce to i8*, !dbg !3139
  %10 = bitcast %struct.dim3* %agg.tmp1 to i8*, !dbg !3139
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %9, i8* align 4 %10, i64 12, i1 false), !dbg !3139
  %11 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp1.coerce, i32 0, i32 0, !dbg !3139
  %12 = load i64, i64* %11, align 4, !dbg !3139
  %13 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp1.coerce, i32 0, i32 1, !dbg !3139
  %14 = load i32, i32* %13, align 4, !dbg !3139
  %call = call i32 @cudaConfigureCall(i64 %6, i32 %8, i64 %12, i32 %14, i64 %2, %struct.CUstream_st* null), !dbg !3139
  %tobool = icmp ne i32 %call, 0, !dbg !3139
  br i1 %tobool, label %kcall.end, label %kcall.configok, !dbg !3140

kcall.configok:                                   ; preds = %entry
  %15 = load double*, double** @d_device, align 8, !dbg !3141
  %16 = load double*, double** @p_device, align 8, !dbg !3142
  %17 = load double*, double** @q_device, align 8, !dbg !3143
  %18 = load double*, double** @global_data_device, align 8, !dbg !3144
  call void @_Z22gpu_kernel_four_devicePdS_S_S_(double* %15, double* %16, double* %17, double* %18), !dbg !3140
  br label %kcall.end, !dbg !3140

kcall.end:                                        ; preds = %kcall.configok, %entry
  store double 0.000000e+00, double* @global_data_reduce, align 8, !dbg !3145
  %19 = load double*, double** @global_data, align 8, !dbg !3146
  %20 = bitcast double* %19 to i8*, !dbg !3146
  %21 = load double*, double** @global_data_device, align 8, !dbg !3147
  %22 = bitcast double* %21 to i8*, !dbg !3147
  %23 = load i64, i64* @size_reduce_memory_on_kernel_four, align 8, !dbg !3148
  %call2 = call i32 @cudaMemcpy(i8* %20, i8* %22, i64 %23, i32 2), !dbg !3149
  call void @llvm.dbg.declare(metadata i32* %i, metadata !3150, metadata !DIExpression()), !dbg !3152
  store i32 0, i32* %i, align 4, !dbg !3152
  br label %for.cond, !dbg !3153

for.cond:                                         ; preds = %for.inc, %kcall.end
  %24 = load i32, i32* %i, align 4, !dbg !3154
  %25 = load i32, i32* @blocks_per_grid_on_kernel_four, align 4, !dbg !3156
  %cmp = icmp slt i32 %24, %25, !dbg !3157
  br i1 %cmp, label %for.body, label %for.end, !dbg !3158

for.body:                                         ; preds = %for.cond
  %26 = load double*, double** @global_data, align 8, !dbg !3159
  %27 = load i32, i32* %i, align 4, !dbg !3161
  %idxprom = sext i32 %27 to i64, !dbg !3159
  %arrayidx = getelementptr inbounds double, double* %26, i64 %idxprom, !dbg !3159
  %28 = load double, double* %arrayidx, align 8, !dbg !3159
  %29 = load double, double* @global_data_reduce, align 8, !dbg !3162
  %add = fadd contract double %29, %28, !dbg !3162
  store double %add, double* @global_data_reduce, align 8, !dbg !3162
  br label %for.inc, !dbg !3163

for.inc:                                          ; preds = %for.body
  %30 = load i32, i32* %i, align 4, !dbg !3164
  %inc = add nsw i32 %30, 1, !dbg !3164
  store i32 %inc, i32* %i, align 4, !dbg !3164
  br label %for.cond, !dbg !3165, !llvm.loop !3166

for.end:                                          ; preds = %for.cond
  %31 = load double, double* @global_data_reduce, align 8, !dbg !3168
  %32 = load double*, double** %d_host.addr, align 8, !dbg !3169
  store double %31, double* %32, align 8, !dbg !3170
  ret void, !dbg !3171
}

; Function Attrs: noinline uwtable
define internal void @_ZL20gpu_kernel_five_hostd(double %alpha_host) #2 !dbg !3172 {
entry:
  %alpha_host.addr = alloca double, align 8
  %agg.tmp = alloca %struct.dim3, align 4
  %agg.tmp1 = alloca %struct.dim3, align 4
  %agg.tmp.coerce = alloca { i64, i32 }, align 4
  %agg.tmp1.coerce = alloca { i64, i32 }, align 4
  %agg.tmp2 = alloca %struct.dim3, align 4
  %agg.tmp3 = alloca %struct.dim3, align 4
  %agg.tmp2.coerce = alloca { i64, i32 }, align 4
  %agg.tmp3.coerce = alloca { i64, i32 }, align 4
  store double %alpha_host, double* %alpha_host.addr, align 8
  call void @llvm.dbg.declare(metadata double* %alpha_host.addr, metadata !3173, metadata !DIExpression()), !dbg !3174
  %0 = load i32, i32* @blocks_per_grid_on_kernel_five, align 4, !dbg !3175
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp, i32 %0, i32 1, i32 1), !dbg !3175
  %1 = load i32, i32* @threads_per_block_on_kernel_five, align 4, !dbg !3176
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp1, i32 %1, i32 1, i32 1), !dbg !3176
  %2 = bitcast { i64, i32 }* %agg.tmp.coerce to i8*, !dbg !3177
  %3 = bitcast %struct.dim3* %agg.tmp to i8*, !dbg !3177
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %2, i8* align 4 %3, i64 12, i1 false), !dbg !3177
  %4 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 0, !dbg !3177
  %5 = load i64, i64* %4, align 4, !dbg !3177
  %6 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 1, !dbg !3177
  %7 = load i32, i32* %6, align 4, !dbg !3177
  %8 = bitcast { i64, i32 }* %agg.tmp1.coerce to i8*, !dbg !3177
  %9 = bitcast %struct.dim3* %agg.tmp1 to i8*, !dbg !3177
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %8, i8* align 4 %9, i64 12, i1 false), !dbg !3177
  %10 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp1.coerce, i32 0, i32 0, !dbg !3177
  %11 = load i64, i64* %10, align 4, !dbg !3177
  %12 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp1.coerce, i32 0, i32 1, !dbg !3177
  %13 = load i32, i32* %12, align 4, !dbg !3177
  %call = call i32 @cudaConfigureCall(i64 %5, i32 %7, i64 %11, i32 %13, i64 0, %struct.CUstream_st* null), !dbg !3177
  %tobool = icmp ne i32 %call, 0, !dbg !3177
  br i1 %tobool, label %kcall.end, label %kcall.configok, !dbg !3178

kcall.configok:                                   ; preds = %entry
  %14 = load double, double* %alpha_host.addr, align 8, !dbg !3179
  %15 = load double*, double** @p_device, align 8, !dbg !3180
  %16 = load double*, double** @z_device, align 8, !dbg !3181
  call void @_Z17gpu_kernel_five_1dPdS_(double %14, double* %15, double* %16), !dbg !3178
  br label %kcall.end, !dbg !3178

kcall.end:                                        ; preds = %kcall.configok, %entry
  %17 = load i32, i32* @blocks_per_grid_on_kernel_five, align 4, !dbg !3182
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp2, i32 %17, i32 1, i32 1), !dbg !3182
  %18 = load i32, i32* @threads_per_block_on_kernel_five, align 4, !dbg !3183
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp3, i32 %18, i32 1, i32 1), !dbg !3183
  %19 = bitcast { i64, i32 }* %agg.tmp2.coerce to i8*, !dbg !3184
  %20 = bitcast %struct.dim3* %agg.tmp2 to i8*, !dbg !3184
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %19, i8* align 4 %20, i64 12, i1 false), !dbg !3184
  %21 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp2.coerce, i32 0, i32 0, !dbg !3184
  %22 = load i64, i64* %21, align 4, !dbg !3184
  %23 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp2.coerce, i32 0, i32 1, !dbg !3184
  %24 = load i32, i32* %23, align 4, !dbg !3184
  %25 = bitcast { i64, i32 }* %agg.tmp3.coerce to i8*, !dbg !3184
  %26 = bitcast %struct.dim3* %agg.tmp3 to i8*, !dbg !3184
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %25, i8* align 4 %26, i64 12, i1 false), !dbg !3184
  %27 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp3.coerce, i32 0, i32 0, !dbg !3184
  %28 = load i64, i64* %27, align 4, !dbg !3184
  %29 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp3.coerce, i32 0, i32 1, !dbg !3184
  %30 = load i32, i32* %29, align 4, !dbg !3184
  %call4 = call i32 @cudaConfigureCall(i64 %22, i32 %24, i64 %28, i32 %30, i64 0, %struct.CUstream_st* null), !dbg !3184
  %tobool5 = icmp ne i32 %call4, 0, !dbg !3184
  br i1 %tobool5, label %kcall.end7, label %kcall.configok6, !dbg !3185

kcall.configok6:                                  ; preds = %kcall.end
  %31 = load double, double* %alpha_host.addr, align 8, !dbg !3186
  %32 = load double*, double** @q_device, align 8, !dbg !3187
  %33 = load double*, double** @r_device, align 8, !dbg !3188
  call void @_Z17gpu_kernel_five_2dPdS_(double %31, double* %32, double* %33), !dbg !3185
  br label %kcall.end7, !dbg !3185

kcall.end7:                                       ; preds = %kcall.configok6, %kcall.end
  ret void, !dbg !3189
}

; Function Attrs: noinline uwtable
define internal void @_ZL19gpu_kernel_six_hostPd(double* %rho_host) #2 !dbg !3190 {
entry:
  %rho_host.addr = alloca double*, align 8
  %agg.tmp = alloca %struct.dim3, align 4
  %agg.tmp1 = alloca %struct.dim3, align 4
  %agg.tmp.coerce = alloca { i64, i32 }, align 4
  %agg.tmp1.coerce = alloca { i64, i32 }, align 4
  %i = alloca i32, align 4
  store double* %rho_host, double** %rho_host.addr, align 8
  call void @llvm.dbg.declare(metadata double** %rho_host.addr, metadata !3191, metadata !DIExpression()), !dbg !3192
  %0 = load i32, i32* @blocks_per_grid_on_kernel_six, align 4, !dbg !3193
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp, i32 %0, i32 1, i32 1), !dbg !3193
  %1 = load i32, i32* @threads_per_block_on_kernel_six, align 4, !dbg !3194
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp1, i32 %1, i32 1, i32 1), !dbg !3194
  %2 = load i64, i64* @size_shared_data_on_kernel_six, align 8, !dbg !3195
  %3 = bitcast { i64, i32 }* %agg.tmp.coerce to i8*, !dbg !3196
  %4 = bitcast %struct.dim3* %agg.tmp to i8*, !dbg !3196
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %3, i8* align 4 %4, i64 12, i1 false), !dbg !3196
  %5 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 0, !dbg !3196
  %6 = load i64, i64* %5, align 4, !dbg !3196
  %7 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 1, !dbg !3196
  %8 = load i32, i32* %7, align 4, !dbg !3196
  %9 = bitcast { i64, i32 }* %agg.tmp1.coerce to i8*, !dbg !3196
  %10 = bitcast %struct.dim3* %agg.tmp1 to i8*, !dbg !3196
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %9, i8* align 4 %10, i64 12, i1 false), !dbg !3196
  %11 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp1.coerce, i32 0, i32 0, !dbg !3196
  %12 = load i64, i64* %11, align 4, !dbg !3196
  %13 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp1.coerce, i32 0, i32 1, !dbg !3196
  %14 = load i32, i32* %13, align 4, !dbg !3196
  %call = call i32 @cudaConfigureCall(i64 %6, i32 %8, i64 %12, i32 %14, i64 %2, %struct.CUstream_st* null), !dbg !3196
  %tobool = icmp ne i32 %call, 0, !dbg !3196
  br i1 %tobool, label %kcall.end, label %kcall.configok, !dbg !3197

kcall.configok:                                   ; preds = %entry
  %15 = load double*, double** @r_device, align 8, !dbg !3198
  %16 = load double*, double** @global_data_device, align 8, !dbg !3199
  call void @_Z21gpu_kernel_six_devicePdS_(double* %15, double* %16), !dbg !3197
  br label %kcall.end, !dbg !3197

kcall.end:                                        ; preds = %kcall.configok, %entry
  store double 0.000000e+00, double* @global_data_reduce, align 8, !dbg !3200
  %17 = load double*, double** @global_data, align 8, !dbg !3201
  %18 = bitcast double* %17 to i8*, !dbg !3201
  %19 = load double*, double** @global_data_device, align 8, !dbg !3202
  %20 = bitcast double* %19 to i8*, !dbg !3202
  %21 = load i64, i64* @size_reduce_memory_on_kernel_six, align 8, !dbg !3203
  %call2 = call i32 @cudaMemcpy(i8* %18, i8* %20, i64 %21, i32 2), !dbg !3204
  call void @llvm.dbg.declare(metadata i32* %i, metadata !3205, metadata !DIExpression()), !dbg !3207
  store i32 0, i32* %i, align 4, !dbg !3207
  br label %for.cond, !dbg !3208

for.cond:                                         ; preds = %for.inc, %kcall.end
  %22 = load i32, i32* %i, align 4, !dbg !3209
  %23 = load i32, i32* @blocks_per_grid_on_kernel_six, align 4, !dbg !3211
  %cmp = icmp slt i32 %22, %23, !dbg !3212
  br i1 %cmp, label %for.body, label %for.end, !dbg !3213

for.body:                                         ; preds = %for.cond
  %24 = load double*, double** @global_data, align 8, !dbg !3214
  %25 = load i32, i32* %i, align 4, !dbg !3216
  %idxprom = sext i32 %25 to i64, !dbg !3214
  %arrayidx = getelementptr inbounds double, double* %24, i64 %idxprom, !dbg !3214
  %26 = load double, double* %arrayidx, align 8, !dbg !3214
  %27 = load double, double* @global_data_reduce, align 8, !dbg !3217
  %add = fadd contract double %27, %26, !dbg !3217
  store double %add, double* @global_data_reduce, align 8, !dbg !3217
  br label %for.inc, !dbg !3218

for.inc:                                          ; preds = %for.body
  %28 = load i32, i32* %i, align 4, !dbg !3219
  %inc = add nsw i32 %28, 1, !dbg !3219
  store i32 %inc, i32* %i, align 4, !dbg !3219
  br label %for.cond, !dbg !3220, !llvm.loop !3221

for.end:                                          ; preds = %for.cond
  %29 = load double, double* @global_data_reduce, align 8, !dbg !3223
  %30 = load double*, double** %rho_host.addr, align 8, !dbg !3224
  store double %29, double* %30, align 8, !dbg !3225
  ret void, !dbg !3226
}

; Function Attrs: noinline uwtable
define internal void @_ZL21gpu_kernel_seven_hostd(double %beta_host) #2 !dbg !3227 {
entry:
  %beta_host.addr = alloca double, align 8
  %agg.tmp = alloca %struct.dim3, align 4
  %agg.tmp1 = alloca %struct.dim3, align 4
  %agg.tmp.coerce = alloca { i64, i32 }, align 4
  %agg.tmp1.coerce = alloca { i64, i32 }, align 4
  store double %beta_host, double* %beta_host.addr, align 8
  call void @llvm.dbg.declare(metadata double* %beta_host.addr, metadata !3228, metadata !DIExpression()), !dbg !3229
  %0 = load i32, i32* @blocks_per_grid_on_kernel_seven, align 4, !dbg !3230
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp, i32 %0, i32 1, i32 1), !dbg !3230
  %1 = load i32, i32* @threads_per_block_on_kernel_seven, align 4, !dbg !3231
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp1, i32 %1, i32 1, i32 1), !dbg !3231
  %2 = bitcast { i64, i32 }* %agg.tmp.coerce to i8*, !dbg !3232
  %3 = bitcast %struct.dim3* %agg.tmp to i8*, !dbg !3232
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %2, i8* align 4 %3, i64 12, i1 false), !dbg !3232
  %4 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 0, !dbg !3232
  %5 = load i64, i64* %4, align 4, !dbg !3232
  %6 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 1, !dbg !3232
  %7 = load i32, i32* %6, align 4, !dbg !3232
  %8 = bitcast { i64, i32 }* %agg.tmp1.coerce to i8*, !dbg !3232
  %9 = bitcast %struct.dim3* %agg.tmp1 to i8*, !dbg !3232
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %8, i8* align 4 %9, i64 12, i1 false), !dbg !3232
  %10 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp1.coerce, i32 0, i32 0, !dbg !3232
  %11 = load i64, i64* %10, align 4, !dbg !3232
  %12 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp1.coerce, i32 0, i32 1, !dbg !3232
  %13 = load i32, i32* %12, align 4, !dbg !3232
  %call = call i32 @cudaConfigureCall(i64 %5, i32 %7, i64 %11, i32 %13, i64 0, %struct.CUstream_st* null), !dbg !3232
  %tobool = icmp ne i32 %call, 0, !dbg !3232
  br i1 %tobool, label %kcall.end, label %kcall.configok, !dbg !3233

kcall.configok:                                   ; preds = %entry
  %14 = load double, double* %beta_host.addr, align 8, !dbg !3234
  %15 = load double*, double** @p_device, align 8, !dbg !3235
  %16 = load double*, double** @r_device, align 8, !dbg !3236
  call void @_Z23gpu_kernel_seven_devicedPdS_(double %14, double* %15, double* %16), !dbg !3233
  br label %kcall.end, !dbg !3233

kcall.end:                                        ; preds = %kcall.configok, %entry
  ret void, !dbg !3237
}

; Function Attrs: noinline uwtable
define internal void @_ZL21gpu_kernel_eight_hostv() #2 !dbg !3238 {
entry:
  %agg.tmp = alloca %struct.dim3, align 4
  %agg.tmp1 = alloca %struct.dim3, align 4
  %agg.tmp.coerce = alloca { i64, i32 }, align 4
  %agg.tmp1.coerce = alloca { i64, i32 }, align 4
  %0 = load i32, i32* @blocks_per_grid_on_kernel_eight, align 4, !dbg !3239
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp, i32 %0, i32 1, i32 1), !dbg !3239
  %1 = load i32, i32* @threads_per_block_on_kernel_eight, align 4, !dbg !3240
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp1, i32 %1, i32 1, i32 1), !dbg !3240
  %2 = load i64, i64* @size_shared_data_on_kernel_eight, align 8, !dbg !3241
  %3 = bitcast { i64, i32 }* %agg.tmp.coerce to i8*, !dbg !3242
  %4 = bitcast %struct.dim3* %agg.tmp to i8*, !dbg !3242
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %3, i8* align 4 %4, i64 12, i1 false), !dbg !3242
  %5 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 0, !dbg !3242
  %6 = load i64, i64* %5, align 4, !dbg !3242
  %7 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 1, !dbg !3242
  %8 = load i32, i32* %7, align 4, !dbg !3242
  %9 = bitcast { i64, i32 }* %agg.tmp1.coerce to i8*, !dbg !3242
  %10 = bitcast %struct.dim3* %agg.tmp1 to i8*, !dbg !3242
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %9, i8* align 4 %10, i64 12, i1 false), !dbg !3242
  %11 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp1.coerce, i32 0, i32 0, !dbg !3242
  %12 = load i64, i64* %11, align 4, !dbg !3242
  %13 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp1.coerce, i32 0, i32 1, !dbg !3242
  %14 = load i32, i32* %13, align 4, !dbg !3242
  %call = call i32 @cudaConfigureCall(i64 %6, i32 %8, i64 %12, i32 %14, i64 %2, %struct.CUstream_st* null), !dbg !3242
  %tobool = icmp ne i32 %call, 0, !dbg !3242
  br i1 %tobool, label %kcall.end, label %kcall.configok, !dbg !3243

kcall.configok:                                   ; preds = %entry
  %15 = load i32*, i32** @colidx_device, align 8, !dbg !3244
  %16 = load i32*, i32** @rowstr_device, align 8, !dbg !3245
  %17 = load double*, double** @a_device, align 8, !dbg !3246
  %18 = load double*, double** @r_device, align 8, !dbg !3247
  %19 = load double*, double** @z_device, align 8, !dbg !3248
  call void @_Z23gpu_kernel_eight_devicePiS_PdS0_S0_(i32* %15, i32* %16, double* %17, double* %18, double* %19), !dbg !3243
  br label %kcall.end, !dbg !3243

kcall.end:                                        ; preds = %kcall.configok, %entry
  ret void, !dbg !3249
}

; Function Attrs: noinline uwtable
define internal void @_ZL20gpu_kernel_nine_hostPd(double* %sum_host) #2 !dbg !3250 {
entry:
  %sum_host.addr = alloca double*, align 8
  %agg.tmp = alloca %struct.dim3, align 4
  %agg.tmp1 = alloca %struct.dim3, align 4
  %agg.tmp.coerce = alloca { i64, i32 }, align 4
  %agg.tmp1.coerce = alloca { i64, i32 }, align 4
  %i = alloca i32, align 4
  store double* %sum_host, double** %sum_host.addr, align 8
  call void @llvm.dbg.declare(metadata double** %sum_host.addr, metadata !3251, metadata !DIExpression()), !dbg !3252
  %0 = load i32, i32* @blocks_per_grid_on_kernel_nine, align 4, !dbg !3253
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp, i32 %0, i32 1, i32 1), !dbg !3253
  %1 = load i32, i32* @threads_per_block_on_kernel_nine, align 4, !dbg !3254
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp1, i32 %1, i32 1, i32 1), !dbg !3254
  %2 = load i64, i64* @size_shared_data_on_kernel_nine, align 8, !dbg !3255
  %3 = bitcast { i64, i32 }* %agg.tmp.coerce to i8*, !dbg !3256
  %4 = bitcast %struct.dim3* %agg.tmp to i8*, !dbg !3256
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %3, i8* align 4 %4, i64 12, i1 false), !dbg !3256
  %5 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 0, !dbg !3256
  %6 = load i64, i64* %5, align 4, !dbg !3256
  %7 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 1, !dbg !3256
  %8 = load i32, i32* %7, align 4, !dbg !3256
  %9 = bitcast { i64, i32 }* %agg.tmp1.coerce to i8*, !dbg !3256
  %10 = bitcast %struct.dim3* %agg.tmp1 to i8*, !dbg !3256
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %9, i8* align 4 %10, i64 12, i1 false), !dbg !3256
  %11 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp1.coerce, i32 0, i32 0, !dbg !3256
  %12 = load i64, i64* %11, align 4, !dbg !3256
  %13 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp1.coerce, i32 0, i32 1, !dbg !3256
  %14 = load i32, i32* %13, align 4, !dbg !3256
  %call = call i32 @cudaConfigureCall(i64 %6, i32 %8, i64 %12, i32 %14, i64 %2, %struct.CUstream_st* null), !dbg !3256
  %tobool = icmp ne i32 %call, 0, !dbg !3256
  br i1 %tobool, label %kcall.end, label %kcall.configok, !dbg !3257

kcall.configok:                                   ; preds = %entry
  %15 = load double*, double** @r_device, align 8, !dbg !3258
  %16 = load double*, double** @x_device, align 8, !dbg !3259
  %17 = load double*, double** @sum_device, align 8, !dbg !3260
  %18 = load double*, double** @global_data_device, align 8, !dbg !3261
  call void @_Z22gpu_kernel_nine_devicePdS_S_S_(double* %15, double* %16, double* %17, double* %18), !dbg !3257
  br label %kcall.end, !dbg !3257

kcall.end:                                        ; preds = %kcall.configok, %entry
  store double 0.000000e+00, double* @global_data_reduce, align 8, !dbg !3262
  %19 = load double*, double** @global_data, align 8, !dbg !3263
  %20 = bitcast double* %19 to i8*, !dbg !3263
  %21 = load double*, double** @global_data_device, align 8, !dbg !3264
  %22 = bitcast double* %21 to i8*, !dbg !3264
  %23 = load i64, i64* @size_reduce_memory_on_kernel_nine, align 8, !dbg !3265
  %call2 = call i32 @cudaMemcpy(i8* %20, i8* %22, i64 %23, i32 2), !dbg !3266
  call void @llvm.dbg.declare(metadata i32* %i, metadata !3267, metadata !DIExpression()), !dbg !3269
  store i32 0, i32* %i, align 4, !dbg !3269
  br label %for.cond, !dbg !3270

for.cond:                                         ; preds = %for.inc, %kcall.end
  %24 = load i32, i32* %i, align 4, !dbg !3271
  %25 = load i32, i32* @blocks_per_grid_on_kernel_nine, align 4, !dbg !3273
  %cmp = icmp slt i32 %24, %25, !dbg !3274
  br i1 %cmp, label %for.body, label %for.end, !dbg !3275

for.body:                                         ; preds = %for.cond
  %26 = load double*, double** @global_data, align 8, !dbg !3276
  %27 = load i32, i32* %i, align 4, !dbg !3278
  %idxprom = sext i32 %27 to i64, !dbg !3276
  %arrayidx = getelementptr inbounds double, double* %26, i64 %idxprom, !dbg !3276
  %28 = load double, double* %arrayidx, align 8, !dbg !3276
  %29 = load double, double* @global_data_reduce, align 8, !dbg !3279
  %add = fadd contract double %29, %28, !dbg !3279
  store double %add, double* @global_data_reduce, align 8, !dbg !3279
  br label %for.inc, !dbg !3280

for.inc:                                          ; preds = %for.body
  %30 = load i32, i32* %i, align 4, !dbg !3281
  %inc = add nsw i32 %30, 1, !dbg !3281
  store i32 %inc, i32* %i, align 4, !dbg !3281
  br label %for.cond, !dbg !3282, !llvm.loop !3283

for.end:                                          ; preds = %for.cond
  %31 = load double, double* @global_data_reduce, align 8, !dbg !3285
  %32 = load double*, double** %sum_host.addr, align 8, !dbg !3286
  store double %31, double* %32, align 8, !dbg !3287
  ret void, !dbg !3288
}

declare dso_local i32 @cudaConfigureCall(i64, i32, i64, i32, i64, %struct.CUstream_st*) #3

; Function Attrs: noinline nounwind uwtable
define linkonce_odr dso_local void @_ZN4dim3C2Ejjj(%struct.dim3* %this, i32 %vx, i32 %vy, i32 %vz) unnamed_addr #0 comdat align 2 !dbg !3289 {
entry:
  %this.addr = alloca %struct.dim3*, align 8
  %vx.addr = alloca i32, align 4
  %vy.addr = alloca i32, align 4
  %vz.addr = alloca i32, align 4
  store %struct.dim3* %this, %struct.dim3** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dim3** %this.addr, metadata !3312, metadata !DIExpression()), !dbg !3314
  store i32 %vx, i32* %vx.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %vx.addr, metadata !3315, metadata !DIExpression()), !dbg !3316
  store i32 %vy, i32* %vy.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %vy.addr, metadata !3317, metadata !DIExpression()), !dbg !3318
  store i32 %vz, i32* %vz.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %vz.addr, metadata !3319, metadata !DIExpression()), !dbg !3320
  %this1 = load %struct.dim3*, %struct.dim3** %this.addr, align 8
  %x = getelementptr inbounds %struct.dim3, %struct.dim3* %this1, i32 0, i32 0, !dbg !3321
  %0 = load i32, i32* %vx.addr, align 4, !dbg !3322
  store i32 %0, i32* %x, align 4, !dbg !3321
  %y = getelementptr inbounds %struct.dim3, %struct.dim3* %this1, i32 0, i32 1, !dbg !3323
  %1 = load i32, i32* %vy.addr, align 4, !dbg !3324
  store i32 %1, i32* %y, align 4, !dbg !3323
  %z = getelementptr inbounds %struct.dim3, %struct.dim3* %this1, i32 0, i32 2, !dbg !3325
  %2 = load i32, i32* %vz.addr, align 4, !dbg !3326
  store i32 %2, i32* %z, align 4, !dbg !3325
  ret void, !dbg !3327
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1 immarg) #6

declare dso_local i32 @cudaMemcpy(i8*, i8*, i64, i32) #3

; Function Attrs: noinline uwtable
define internal void @_ZL6sprnvciiiPdPi(i32 %n, i32 %nz, i32 %nn1, double* %v, i32* %iv) #2 !dbg !3328 {
entry:
  %n.addr = alloca i32, align 4
  %nz.addr = alloca i32, align 4
  %nn1.addr = alloca i32, align 4
  %v.addr = alloca double*, align 8
  %iv.addr = alloca i32*, align 8
  %nzv = alloca i32, align 4
  %ii = alloca i32, align 4
  %i = alloca i32, align 4
  %vecelt = alloca double, align 8
  %vecloc = alloca double, align 8
  %was_gen = alloca i32, align 4
  store i32 %n, i32* %n.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %n.addr, metadata !3331, metadata !DIExpression()), !dbg !3332
  store i32 %nz, i32* %nz.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %nz.addr, metadata !3333, metadata !DIExpression()), !dbg !3334
  store i32 %nn1, i32* %nn1.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %nn1.addr, metadata !3335, metadata !DIExpression()), !dbg !3336
  store double* %v, double** %v.addr, align 8
  call void @llvm.dbg.declare(metadata double** %v.addr, metadata !3337, metadata !DIExpression()), !dbg !3338
  store i32* %iv, i32** %iv.addr, align 8
  call void @llvm.dbg.declare(metadata i32** %iv.addr, metadata !3339, metadata !DIExpression()), !dbg !3340
  call void @llvm.dbg.declare(metadata i32* %nzv, metadata !3341, metadata !DIExpression()), !dbg !3342
  call void @llvm.dbg.declare(metadata i32* %ii, metadata !3343, metadata !DIExpression()), !dbg !3344
  call void @llvm.dbg.declare(metadata i32* %i, metadata !3345, metadata !DIExpression()), !dbg !3346
  call void @llvm.dbg.declare(metadata double* %vecelt, metadata !3347, metadata !DIExpression()), !dbg !3348
  call void @llvm.dbg.declare(metadata double* %vecloc, metadata !3349, metadata !DIExpression()), !dbg !3350
  store i32 0, i32* %nzv, align 4, !dbg !3351
  br label %while.cond, !dbg !3352

while.cond:                                       ; preds = %if.end9, %if.then8, %if.then, %entry
  %0 = load i32, i32* %nzv, align 4, !dbg !3353
  %1 = load i32, i32* %nz.addr, align 4, !dbg !3354
  %cmp = icmp slt i32 %0, %1, !dbg !3355
  br i1 %cmp, label %while.body, label %while.end, !dbg !3352

while.body:                                       ; preds = %while.cond
  %2 = load double, double* @_ZL5amult, align 8, !dbg !3356
  %call = call double @_Z6randlcPdd(double* @_ZL4tran, double %2), !dbg !3358
  store double %call, double* %vecelt, align 8, !dbg !3359
  %3 = load double, double* @_ZL5amult, align 8, !dbg !3360
  %call1 = call double @_Z6randlcPdd(double* @_ZL4tran, double %3), !dbg !3361
  store double %call1, double* %vecloc, align 8, !dbg !3362
  %4 = load double, double* %vecloc, align 8, !dbg !3363
  %5 = load i32, i32* %nn1.addr, align 4, !dbg !3364
  %call2 = call i32 @_ZL6icnvrtdi(double %4, i32 %5), !dbg !3365
  %add = add nsw i32 %call2, 1, !dbg !3366
  store i32 %add, i32* %i, align 4, !dbg !3367
  %6 = load i32, i32* %i, align 4, !dbg !3368
  %7 = load i32, i32* %n.addr, align 4, !dbg !3370
  %cmp3 = icmp sgt i32 %6, %7, !dbg !3371
  br i1 %cmp3, label %if.then, label %if.end, !dbg !3372

if.then:                                          ; preds = %while.body
  br label %while.cond, !dbg !3373, !llvm.loop !3375

if.end:                                           ; preds = %while.body
  call void @llvm.dbg.declare(metadata i32* %was_gen, metadata !3377, metadata !DIExpression()), !dbg !3378
  store i32 0, i32* %was_gen, align 4, !dbg !3378
  store i32 0, i32* %ii, align 4, !dbg !3379
  br label %for.cond, !dbg !3381

for.cond:                                         ; preds = %for.inc, %if.end
  %8 = load i32, i32* %ii, align 4, !dbg !3382
  %9 = load i32, i32* %nzv, align 4, !dbg !3384
  %cmp4 = icmp slt i32 %8, %9, !dbg !3385
  br i1 %cmp4, label %for.body, label %for.end, !dbg !3386

for.body:                                         ; preds = %for.cond
  %10 = load i32*, i32** %iv.addr, align 8, !dbg !3387
  %11 = load i32, i32* %ii, align 4, !dbg !3390
  %idxprom = sext i32 %11 to i64, !dbg !3387
  %arrayidx = getelementptr inbounds i32, i32* %10, i64 %idxprom, !dbg !3387
  %12 = load i32, i32* %arrayidx, align 4, !dbg !3387
  %13 = load i32, i32* %i, align 4, !dbg !3391
  %cmp5 = icmp eq i32 %12, %13, !dbg !3392
  br i1 %cmp5, label %if.then6, label %if.end7, !dbg !3393

if.then6:                                         ; preds = %for.body
  store i32 1, i32* %was_gen, align 4, !dbg !3394
  br label %for.end, !dbg !3396

if.end7:                                          ; preds = %for.body
  br label %for.inc, !dbg !3397

for.inc:                                          ; preds = %if.end7
  %14 = load i32, i32* %ii, align 4, !dbg !3398
  %inc = add nsw i32 %14, 1, !dbg !3398
  store i32 %inc, i32* %ii, align 4, !dbg !3398
  br label %for.cond, !dbg !3399, !llvm.loop !3400

for.end:                                          ; preds = %if.then6, %for.cond
  %15 = load i32, i32* %was_gen, align 4, !dbg !3402
  %tobool = icmp ne i32 %15, 0, !dbg !3402
  br i1 %tobool, label %if.then8, label %if.end9, !dbg !3404

if.then8:                                         ; preds = %for.end
  br label %while.cond, !dbg !3405, !llvm.loop !3375

if.end9:                                          ; preds = %for.end
  %16 = load double, double* %vecelt, align 8, !dbg !3407
  %17 = load double*, double** %v.addr, align 8, !dbg !3408
  %18 = load i32, i32* %nzv, align 4, !dbg !3409
  %idxprom10 = sext i32 %18 to i64, !dbg !3408
  %arrayidx11 = getelementptr inbounds double, double* %17, i64 %idxprom10, !dbg !3408
  store double %16, double* %arrayidx11, align 8, !dbg !3410
  %19 = load i32, i32* %i, align 4, !dbg !3411
  %20 = load i32*, i32** %iv.addr, align 8, !dbg !3412
  %21 = load i32, i32* %nzv, align 4, !dbg !3413
  %idxprom12 = sext i32 %21 to i64, !dbg !3412
  %arrayidx13 = getelementptr inbounds i32, i32* %20, i64 %idxprom12, !dbg !3412
  store i32 %19, i32* %arrayidx13, align 4, !dbg !3414
  %22 = load i32, i32* %nzv, align 4, !dbg !3415
  %add14 = add nsw i32 %22, 1, !dbg !3416
  store i32 %add14, i32* %nzv, align 4, !dbg !3417
  br label %while.cond, !dbg !3352, !llvm.loop !3375

while.end:                                        ; preds = %while.cond
  ret void, !dbg !3418
}

; Function Attrs: noinline nounwind uwtable
define internal void @_ZL6vecsetiPdPiS0_id(i32 %n, double* %v, i32* %iv, i32* %nzv, i32 %i, double %val) #0 !dbg !3419 {
entry:
  %n.addr = alloca i32, align 4
  %v.addr = alloca double*, align 8
  %iv.addr = alloca i32*, align 8
  %nzv.addr = alloca i32*, align 8
  %i.addr = alloca i32, align 4
  %val.addr = alloca double, align 8
  %k = alloca i32, align 4
  %set = alloca i32, align 4
  store i32 %n, i32* %n.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %n.addr, metadata !3422, metadata !DIExpression()), !dbg !3423
  store double* %v, double** %v.addr, align 8
  call void @llvm.dbg.declare(metadata double** %v.addr, metadata !3424, metadata !DIExpression()), !dbg !3425
  store i32* %iv, i32** %iv.addr, align 8
  call void @llvm.dbg.declare(metadata i32** %iv.addr, metadata !3426, metadata !DIExpression()), !dbg !3427
  store i32* %nzv, i32** %nzv.addr, align 8
  call void @llvm.dbg.declare(metadata i32** %nzv.addr, metadata !3428, metadata !DIExpression()), !dbg !3429
  store i32 %i, i32* %i.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %i.addr, metadata !3430, metadata !DIExpression()), !dbg !3431
  store double %val, double* %val.addr, align 8
  call void @llvm.dbg.declare(metadata double* %val.addr, metadata !3432, metadata !DIExpression()), !dbg !3433
  call void @llvm.dbg.declare(metadata i32* %k, metadata !3434, metadata !DIExpression()), !dbg !3435
  call void @llvm.dbg.declare(metadata i32* %set, metadata !3436, metadata !DIExpression()), !dbg !3437
  store i32 0, i32* %set, align 4, !dbg !3438
  store i32 0, i32* %k, align 4, !dbg !3439
  br label %for.cond, !dbg !3441

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, i32* %k, align 4, !dbg !3442
  %1 = load i32*, i32** %nzv.addr, align 8, !dbg !3444
  %2 = load i32, i32* %1, align 4, !dbg !3445
  %cmp = icmp slt i32 %0, %2, !dbg !3446
  br i1 %cmp, label %for.body, label %for.end, !dbg !3447

for.body:                                         ; preds = %for.cond
  %3 = load i32*, i32** %iv.addr, align 8, !dbg !3448
  %4 = load i32, i32* %k, align 4, !dbg !3451
  %idxprom = sext i32 %4 to i64, !dbg !3448
  %arrayidx = getelementptr inbounds i32, i32* %3, i64 %idxprom, !dbg !3448
  %5 = load i32, i32* %arrayidx, align 4, !dbg !3448
  %6 = load i32, i32* %i.addr, align 4, !dbg !3452
  %cmp1 = icmp eq i32 %5, %6, !dbg !3453
  br i1 %cmp1, label %if.then, label %if.end, !dbg !3454

if.then:                                          ; preds = %for.body
  %7 = load double, double* %val.addr, align 8, !dbg !3455
  %8 = load double*, double** %v.addr, align 8, !dbg !3457
  %9 = load i32, i32* %k, align 4, !dbg !3458
  %idxprom2 = sext i32 %9 to i64, !dbg !3457
  %arrayidx3 = getelementptr inbounds double, double* %8, i64 %idxprom2, !dbg !3457
  store double %7, double* %arrayidx3, align 8, !dbg !3459
  store i32 1, i32* %set, align 4, !dbg !3460
  br label %if.end, !dbg !3461

if.end:                                           ; preds = %if.then, %for.body
  br label %for.inc, !dbg !3462

for.inc:                                          ; preds = %if.end
  %10 = load i32, i32* %k, align 4, !dbg !3463
  %inc = add nsw i32 %10, 1, !dbg !3463
  store i32 %inc, i32* %k, align 4, !dbg !3463
  br label %for.cond, !dbg !3464, !llvm.loop !3465

for.end:                                          ; preds = %for.cond
  %11 = load i32, i32* %set, align 4, !dbg !3467
  %cmp4 = icmp eq i32 %11, 0, !dbg !3469
  br i1 %cmp4, label %if.then5, label %if.end10, !dbg !3470

if.then5:                                         ; preds = %for.end
  %12 = load double, double* %val.addr, align 8, !dbg !3471
  %13 = load double*, double** %v.addr, align 8, !dbg !3473
  %14 = load i32*, i32** %nzv.addr, align 8, !dbg !3474
  %15 = load i32, i32* %14, align 4, !dbg !3475
  %idxprom6 = sext i32 %15 to i64, !dbg !3473
  %arrayidx7 = getelementptr inbounds double, double* %13, i64 %idxprom6, !dbg !3473
  store double %12, double* %arrayidx7, align 8, !dbg !3476
  %16 = load i32, i32* %i.addr, align 4, !dbg !3477
  %17 = load i32*, i32** %iv.addr, align 8, !dbg !3478
  %18 = load i32*, i32** %nzv.addr, align 8, !dbg !3479
  %19 = load i32, i32* %18, align 4, !dbg !3480
  %idxprom8 = sext i32 %19 to i64, !dbg !3478
  %arrayidx9 = getelementptr inbounds i32, i32* %17, i64 %idxprom8, !dbg !3478
  store i32 %16, i32* %arrayidx9, align 4, !dbg !3481
  %20 = load i32*, i32** %nzv.addr, align 8, !dbg !3482
  %21 = load i32, i32* %20, align 4, !dbg !3483
  %add = add nsw i32 %21, 1, !dbg !3484
  %22 = load i32*, i32** %nzv.addr, align 8, !dbg !3485
  store i32 %add, i32* %22, align 4, !dbg !3486
  br label %if.end10, !dbg !3487

if.end10:                                         ; preds = %if.then5, %for.end
  ret void, !dbg !3488
}

; Function Attrs: noinline uwtable
define internal void @_ZL6sparsePdPiS0_iiiS0_PA12_iPA12_diiS0_dd(double* %a, i32* %colidx, i32* %rowstr, i32 %n, i32 %nz, i32 %nozer, i32* %arow, [12 x i32]* %acol, [12 x double]* %aelt, i32 %firstrow, i32 %lastrow, i32* %nzloc, double %rcond, double %shift) #2 !dbg !3489 {
entry:
  %a.addr = alloca double*, align 8
  %colidx.addr = alloca i32*, align 8
  %rowstr.addr = alloca i32*, align 8
  %n.addr = alloca i32, align 4
  %nz.addr = alloca i32, align 4
  %nozer.addr = alloca i32, align 4
  %arow.addr = alloca i32*, align 8
  %acol.addr = alloca [12 x i32]*, align 8
  %aelt.addr = alloca [12 x double]*, align 8
  %firstrow.addr = alloca i32, align 4
  %lastrow.addr = alloca i32, align 4
  %nzloc.addr = alloca i32*, align 8
  %rcond.addr = alloca double, align 8
  %shift.addr = alloca double, align 8
  %nrows = alloca i32, align 4
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  %j1 = alloca i32, align 4
  %j2 = alloca i32, align 4
  %nza = alloca i32, align 4
  %k = alloca i32, align 4
  %kk = alloca i32, align 4
  %nzrow = alloca i32, align 4
  %jcol = alloca i32, align 4
  %size = alloca double, align 8
  %scale = alloca double, align 8
  %ratio = alloca double, align 8
  %va = alloca double, align 8
  store double* %a, double** %a.addr, align 8
  call void @llvm.dbg.declare(metadata double** %a.addr, metadata !3492, metadata !DIExpression()), !dbg !3493
  store i32* %colidx, i32** %colidx.addr, align 8
  call void @llvm.dbg.declare(metadata i32** %colidx.addr, metadata !3494, metadata !DIExpression()), !dbg !3495
  store i32* %rowstr, i32** %rowstr.addr, align 8
  call void @llvm.dbg.declare(metadata i32** %rowstr.addr, metadata !3496, metadata !DIExpression()), !dbg !3497
  store i32 %n, i32* %n.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %n.addr, metadata !3498, metadata !DIExpression()), !dbg !3499
  store i32 %nz, i32* %nz.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %nz.addr, metadata !3500, metadata !DIExpression()), !dbg !3501
  store i32 %nozer, i32* %nozer.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %nozer.addr, metadata !3502, metadata !DIExpression()), !dbg !3503
  store i32* %arow, i32** %arow.addr, align 8
  call void @llvm.dbg.declare(metadata i32** %arow.addr, metadata !3504, metadata !DIExpression()), !dbg !3505
  store [12 x i32]* %acol, [12 x i32]** %acol.addr, align 8
  call void @llvm.dbg.declare(metadata [12 x i32]** %acol.addr, metadata !3506, metadata !DIExpression()), !dbg !3507
  store [12 x double]* %aelt, [12 x double]** %aelt.addr, align 8
  call void @llvm.dbg.declare(metadata [12 x double]** %aelt.addr, metadata !3508, metadata !DIExpression()), !dbg !3509
  store i32 %firstrow, i32* %firstrow.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %firstrow.addr, metadata !3510, metadata !DIExpression()), !dbg !3511
  store i32 %lastrow, i32* %lastrow.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %lastrow.addr, metadata !3512, metadata !DIExpression()), !dbg !3513
  store i32* %nzloc, i32** %nzloc.addr, align 8
  call void @llvm.dbg.declare(metadata i32** %nzloc.addr, metadata !3514, metadata !DIExpression()), !dbg !3515
  store double %rcond, double* %rcond.addr, align 8
  call void @llvm.dbg.declare(metadata double* %rcond.addr, metadata !3516, metadata !DIExpression()), !dbg !3517
  store double %shift, double* %shift.addr, align 8
  call void @llvm.dbg.declare(metadata double* %shift.addr, metadata !3518, metadata !DIExpression()), !dbg !3519
  call void @llvm.dbg.declare(metadata i32* %nrows, metadata !3520, metadata !DIExpression()), !dbg !3521
  call void @llvm.dbg.declare(metadata i32* %i, metadata !3522, metadata !DIExpression()), !dbg !3523
  call void @llvm.dbg.declare(metadata i32* %j, metadata !3524, metadata !DIExpression()), !dbg !3525
  call void @llvm.dbg.declare(metadata i32* %j1, metadata !3526, metadata !DIExpression()), !dbg !3527
  call void @llvm.dbg.declare(metadata i32* %j2, metadata !3528, metadata !DIExpression()), !dbg !3529
  call void @llvm.dbg.declare(metadata i32* %nza, metadata !3530, metadata !DIExpression()), !dbg !3531
  call void @llvm.dbg.declare(metadata i32* %k, metadata !3532, metadata !DIExpression()), !dbg !3533
  call void @llvm.dbg.declare(metadata i32* %kk, metadata !3534, metadata !DIExpression()), !dbg !3535
  call void @llvm.dbg.declare(metadata i32* %nzrow, metadata !3536, metadata !DIExpression()), !dbg !3537
  call void @llvm.dbg.declare(metadata i32* %jcol, metadata !3538, metadata !DIExpression()), !dbg !3539
  call void @llvm.dbg.declare(metadata double* %size, metadata !3540, metadata !DIExpression()), !dbg !3541
  call void @llvm.dbg.declare(metadata double* %scale, metadata !3542, metadata !DIExpression()), !dbg !3543
  call void @llvm.dbg.declare(metadata double* %ratio, metadata !3544, metadata !DIExpression()), !dbg !3545
  call void @llvm.dbg.declare(metadata double* %va, metadata !3546, metadata !DIExpression()), !dbg !3547
  %0 = load i32, i32* %lastrow.addr, align 4, !dbg !3548
  %1 = load i32, i32* %firstrow.addr, align 4, !dbg !3549
  %sub = sub nsw i32 %0, %1, !dbg !3550
  %add = add nsw i32 %sub, 1, !dbg !3551
  store i32 %add, i32* %nrows, align 4, !dbg !3552
  store i32 0, i32* %j, align 4, !dbg !3553
  br label %for.cond, !dbg !3555

for.cond:                                         ; preds = %for.inc, %entry
  %2 = load i32, i32* %j, align 4, !dbg !3556
  %3 = load i32, i32* %nrows, align 4, !dbg !3558
  %add1 = add nsw i32 %3, 1, !dbg !3559
  %cmp = icmp slt i32 %2, %add1, !dbg !3560
  br i1 %cmp, label %for.body, label %for.end, !dbg !3561

for.body:                                         ; preds = %for.cond
  %4 = load i32*, i32** %rowstr.addr, align 8, !dbg !3562
  %5 = load i32, i32* %j, align 4, !dbg !3564
  %idxprom = sext i32 %5 to i64, !dbg !3562
  %arrayidx = getelementptr inbounds i32, i32* %4, i64 %idxprom, !dbg !3562
  store i32 0, i32* %arrayidx, align 4, !dbg !3565
  br label %for.inc, !dbg !3566

for.inc:                                          ; preds = %for.body
  %6 = load i32, i32* %j, align 4, !dbg !3567
  %inc = add nsw i32 %6, 1, !dbg !3567
  store i32 %inc, i32* %j, align 4, !dbg !3567
  br label %for.cond, !dbg !3568, !llvm.loop !3569

for.end:                                          ; preds = %for.cond
  store i32 0, i32* %i, align 4, !dbg !3571
  br label %for.cond2, !dbg !3573

for.cond2:                                        ; preds = %for.inc25, %for.end
  %7 = load i32, i32* %i, align 4, !dbg !3574
  %8 = load i32, i32* %n.addr, align 4, !dbg !3576
  %cmp3 = icmp slt i32 %7, %8, !dbg !3577
  br i1 %cmp3, label %for.body4, label %for.end27, !dbg !3578

for.body4:                                        ; preds = %for.cond2
  store i32 0, i32* %nza, align 4, !dbg !3579
  br label %for.cond5, !dbg !3582

for.cond5:                                        ; preds = %for.inc22, %for.body4
  %9 = load i32, i32* %nza, align 4, !dbg !3583
  %10 = load i32*, i32** %arow.addr, align 8, !dbg !3585
  %11 = load i32, i32* %i, align 4, !dbg !3586
  %idxprom6 = sext i32 %11 to i64, !dbg !3585
  %arrayidx7 = getelementptr inbounds i32, i32* %10, i64 %idxprom6, !dbg !3585
  %12 = load i32, i32* %arrayidx7, align 4, !dbg !3585
  %cmp8 = icmp slt i32 %9, %12, !dbg !3587
  br i1 %cmp8, label %for.body9, label %for.end24, !dbg !3588

for.body9:                                        ; preds = %for.cond5
  %13 = load [12 x i32]*, [12 x i32]** %acol.addr, align 8, !dbg !3589
  %14 = load i32, i32* %i, align 4, !dbg !3591
  %idxprom10 = sext i32 %14 to i64, !dbg !3589
  %arrayidx11 = getelementptr inbounds [12 x i32], [12 x i32]* %13, i64 %idxprom10, !dbg !3589
  %15 = load i32, i32* %nza, align 4, !dbg !3592
  %idxprom12 = sext i32 %15 to i64, !dbg !3589
  %arrayidx13 = getelementptr inbounds [12 x i32], [12 x i32]* %arrayidx11, i64 0, i64 %idxprom12, !dbg !3589
  %16 = load i32, i32* %arrayidx13, align 4, !dbg !3589
  %add14 = add nsw i32 %16, 1, !dbg !3593
  store i32 %add14, i32* %j, align 4, !dbg !3594
  %17 = load i32*, i32** %rowstr.addr, align 8, !dbg !3595
  %18 = load i32, i32* %j, align 4, !dbg !3596
  %idxprom15 = sext i32 %18 to i64, !dbg !3595
  %arrayidx16 = getelementptr inbounds i32, i32* %17, i64 %idxprom15, !dbg !3595
  %19 = load i32, i32* %arrayidx16, align 4, !dbg !3595
  %20 = load i32*, i32** %arow.addr, align 8, !dbg !3597
  %21 = load i32, i32* %i, align 4, !dbg !3598
  %idxprom17 = sext i32 %21 to i64, !dbg !3597
  %arrayidx18 = getelementptr inbounds i32, i32* %20, i64 %idxprom17, !dbg !3597
  %22 = load i32, i32* %arrayidx18, align 4, !dbg !3597
  %add19 = add nsw i32 %19, %22, !dbg !3599
  %23 = load i32*, i32** %rowstr.addr, align 8, !dbg !3600
  %24 = load i32, i32* %j, align 4, !dbg !3601
  %idxprom20 = sext i32 %24 to i64, !dbg !3600
  %arrayidx21 = getelementptr inbounds i32, i32* %23, i64 %idxprom20, !dbg !3600
  store i32 %add19, i32* %arrayidx21, align 4, !dbg !3602
  br label %for.inc22, !dbg !3603

for.inc22:                                        ; preds = %for.body9
  %25 = load i32, i32* %nza, align 4, !dbg !3604
  %inc23 = add nsw i32 %25, 1, !dbg !3604
  store i32 %inc23, i32* %nza, align 4, !dbg !3604
  br label %for.cond5, !dbg !3605, !llvm.loop !3606

for.end24:                                        ; preds = %for.cond5
  br label %for.inc25, !dbg !3608

for.inc25:                                        ; preds = %for.end24
  %26 = load i32, i32* %i, align 4, !dbg !3609
  %inc26 = add nsw i32 %26, 1, !dbg !3609
  store i32 %inc26, i32* %i, align 4, !dbg !3609
  br label %for.cond2, !dbg !3610, !llvm.loop !3611

for.end27:                                        ; preds = %for.cond2
  %27 = load i32*, i32** %rowstr.addr, align 8, !dbg !3613
  %arrayidx28 = getelementptr inbounds i32, i32* %27, i64 0, !dbg !3613
  store i32 0, i32* %arrayidx28, align 4, !dbg !3614
  store i32 1, i32* %j, align 4, !dbg !3615
  br label %for.cond29, !dbg !3617

for.cond29:                                       ; preds = %for.inc41, %for.end27
  %28 = load i32, i32* %j, align 4, !dbg !3618
  %29 = load i32, i32* %nrows, align 4, !dbg !3620
  %add30 = add nsw i32 %29, 1, !dbg !3621
  %cmp31 = icmp slt i32 %28, %add30, !dbg !3622
  br i1 %cmp31, label %for.body32, label %for.end43, !dbg !3623

for.body32:                                       ; preds = %for.cond29
  %30 = load i32*, i32** %rowstr.addr, align 8, !dbg !3624
  %31 = load i32, i32* %j, align 4, !dbg !3626
  %idxprom33 = sext i32 %31 to i64, !dbg !3624
  %arrayidx34 = getelementptr inbounds i32, i32* %30, i64 %idxprom33, !dbg !3624
  %32 = load i32, i32* %arrayidx34, align 4, !dbg !3624
  %33 = load i32*, i32** %rowstr.addr, align 8, !dbg !3627
  %34 = load i32, i32* %j, align 4, !dbg !3628
  %sub35 = sub nsw i32 %34, 1, !dbg !3629
  %idxprom36 = sext i32 %sub35 to i64, !dbg !3627
  %arrayidx37 = getelementptr inbounds i32, i32* %33, i64 %idxprom36, !dbg !3627
  %35 = load i32, i32* %arrayidx37, align 4, !dbg !3627
  %add38 = add nsw i32 %32, %35, !dbg !3630
  %36 = load i32*, i32** %rowstr.addr, align 8, !dbg !3631
  %37 = load i32, i32* %j, align 4, !dbg !3632
  %idxprom39 = sext i32 %37 to i64, !dbg !3631
  %arrayidx40 = getelementptr inbounds i32, i32* %36, i64 %idxprom39, !dbg !3631
  store i32 %add38, i32* %arrayidx40, align 4, !dbg !3633
  br label %for.inc41, !dbg !3634

for.inc41:                                        ; preds = %for.body32
  %38 = load i32, i32* %j, align 4, !dbg !3635
  %inc42 = add nsw i32 %38, 1, !dbg !3635
  store i32 %inc42, i32* %j, align 4, !dbg !3635
  br label %for.cond29, !dbg !3636, !llvm.loop !3637

for.end43:                                        ; preds = %for.cond29
  %39 = load i32*, i32** %rowstr.addr, align 8, !dbg !3639
  %40 = load i32, i32* %nrows, align 4, !dbg !3640
  %idxprom44 = sext i32 %40 to i64, !dbg !3639
  %arrayidx45 = getelementptr inbounds i32, i32* %39, i64 %idxprom44, !dbg !3639
  %41 = load i32, i32* %arrayidx45, align 4, !dbg !3639
  %sub46 = sub nsw i32 %41, 1, !dbg !3641
  store i32 %sub46, i32* %nza, align 4, !dbg !3642
  %42 = load i32, i32* %nza, align 4, !dbg !3643
  %43 = load i32, i32* %nz.addr, align 4, !dbg !3645
  %cmp47 = icmp sgt i32 %42, %43, !dbg !3646
  br i1 %cmp47, label %if.then, label %if.end, !dbg !3647

if.then:                                          ; preds = %for.end43
  %call = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([46 x i8], [46 x i8]* @.str.81, i64 0, i64 0)), !dbg !3648
  %44 = load i32, i32* %nza, align 4, !dbg !3650
  %45 = load i32, i32* %nz.addr, align 4, !dbg !3651
  %call48 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([21 x i8], [21 x i8]* @.str.82, i64 0, i64 0), i32 %44, i32 %45), !dbg !3652
  call void @exit(i32 1) #9, !dbg !3653
  unreachable, !dbg !3653

if.end:                                           ; preds = %for.end43
  store i32 0, i32* %j, align 4, !dbg !3654
  br label %for.cond49, !dbg !3656

for.cond49:                                       ; preds = %for.inc69, %if.end
  %46 = load i32, i32* %j, align 4, !dbg !3657
  %47 = load i32, i32* %nrows, align 4, !dbg !3659
  %cmp50 = icmp slt i32 %46, %47, !dbg !3660
  br i1 %cmp50, label %for.body51, label %for.end71, !dbg !3661

for.body51:                                       ; preds = %for.cond49
  %48 = load i32*, i32** %rowstr.addr, align 8, !dbg !3662
  %49 = load i32, i32* %j, align 4, !dbg !3665
  %idxprom52 = sext i32 %49 to i64, !dbg !3662
  %arrayidx53 = getelementptr inbounds i32, i32* %48, i64 %idxprom52, !dbg !3662
  %50 = load i32, i32* %arrayidx53, align 4, !dbg !3662
  store i32 %50, i32* %k, align 4, !dbg !3666
  br label %for.cond54, !dbg !3667

for.cond54:                                       ; preds = %for.inc64, %for.body51
  %51 = load i32, i32* %k, align 4, !dbg !3668
  %52 = load i32*, i32** %rowstr.addr, align 8, !dbg !3670
  %53 = load i32, i32* %j, align 4, !dbg !3671
  %add55 = add nsw i32 %53, 1, !dbg !3672
  %idxprom56 = sext i32 %add55 to i64, !dbg !3670
  %arrayidx57 = getelementptr inbounds i32, i32* %52, i64 %idxprom56, !dbg !3670
  %54 = load i32, i32* %arrayidx57, align 4, !dbg !3670
  %cmp58 = icmp slt i32 %51, %54, !dbg !3673
  br i1 %cmp58, label %for.body59, label %for.end66, !dbg !3674

for.body59:                                       ; preds = %for.cond54
  %55 = load double*, double** %a.addr, align 8, !dbg !3675
  %56 = load i32, i32* %k, align 4, !dbg !3677
  %idxprom60 = sext i32 %56 to i64, !dbg !3675
  %arrayidx61 = getelementptr inbounds double, double* %55, i64 %idxprom60, !dbg !3675
  store double 0.000000e+00, double* %arrayidx61, align 8, !dbg !3678
  %57 = load i32*, i32** %colidx.addr, align 8, !dbg !3679
  %58 = load i32, i32* %k, align 4, !dbg !3680
  %idxprom62 = sext i32 %58 to i64, !dbg !3679
  %arrayidx63 = getelementptr inbounds i32, i32* %57, i64 %idxprom62, !dbg !3679
  store i32 -1, i32* %arrayidx63, align 4, !dbg !3681
  br label %for.inc64, !dbg !3682

for.inc64:                                        ; preds = %for.body59
  %59 = load i32, i32* %k, align 4, !dbg !3683
  %inc65 = add nsw i32 %59, 1, !dbg !3683
  store i32 %inc65, i32* %k, align 4, !dbg !3683
  br label %for.cond54, !dbg !3684, !llvm.loop !3685

for.end66:                                        ; preds = %for.cond54
  %60 = load i32*, i32** %nzloc.addr, align 8, !dbg !3687
  %61 = load i32, i32* %j, align 4, !dbg !3688
  %idxprom67 = sext i32 %61 to i64, !dbg !3687
  %arrayidx68 = getelementptr inbounds i32, i32* %60, i64 %idxprom67, !dbg !3687
  store i32 0, i32* %arrayidx68, align 4, !dbg !3689
  br label %for.inc69, !dbg !3690

for.inc69:                                        ; preds = %for.end66
  %62 = load i32, i32* %j, align 4, !dbg !3691
  %inc70 = add nsw i32 %62, 1, !dbg !3691
  store i32 %inc70, i32* %j, align 4, !dbg !3691
  br label %for.cond49, !dbg !3692, !llvm.loop !3693

for.end71:                                        ; preds = %for.cond49
  store double 1.000000e+00, double* %size, align 8, !dbg !3695
  %63 = load double, double* %rcond.addr, align 8, !dbg !3696
  %64 = load i32, i32* %n.addr, align 4, !dbg !3697
  %conv = sitofp i32 %64 to double, !dbg !3698
  %div = fdiv double 1.000000e+00, %conv, !dbg !3699
  %call72 = call double @pow(double %63, double %div) #8, !dbg !3700
  store double %call72, double* %ratio, align 8, !dbg !3701
  store i32 0, i32* %i, align 4, !dbg !3702
  br label %for.cond73, !dbg !3704

for.cond73:                                       ; preds = %for.inc183, %for.end71
  %65 = load i32, i32* %i, align 4, !dbg !3705
  %66 = load i32, i32* %n.addr, align 4, !dbg !3707
  %cmp74 = icmp slt i32 %65, %66, !dbg !3708
  br i1 %cmp74, label %for.body75, label %for.end185, !dbg !3709

for.body75:                                       ; preds = %for.cond73
  store i32 0, i32* %nza, align 4, !dbg !3710
  br label %for.cond76, !dbg !3713

for.cond76:                                       ; preds = %for.inc179, %for.body75
  %67 = load i32, i32* %nza, align 4, !dbg !3714
  %68 = load i32*, i32** %arow.addr, align 8, !dbg !3716
  %69 = load i32, i32* %i, align 4, !dbg !3717
  %idxprom77 = sext i32 %69 to i64, !dbg !3716
  %arrayidx78 = getelementptr inbounds i32, i32* %68, i64 %idxprom77, !dbg !3716
  %70 = load i32, i32* %arrayidx78, align 4, !dbg !3716
  %cmp79 = icmp slt i32 %67, %70, !dbg !3718
  br i1 %cmp79, label %for.body80, label %for.end181, !dbg !3719

for.body80:                                       ; preds = %for.cond76
  %71 = load [12 x i32]*, [12 x i32]** %acol.addr, align 8, !dbg !3720
  %72 = load i32, i32* %i, align 4, !dbg !3722
  %idxprom81 = sext i32 %72 to i64, !dbg !3720
  %arrayidx82 = getelementptr inbounds [12 x i32], [12 x i32]* %71, i64 %idxprom81, !dbg !3720
  %73 = load i32, i32* %nza, align 4, !dbg !3723
  %idxprom83 = sext i32 %73 to i64, !dbg !3720
  %arrayidx84 = getelementptr inbounds [12 x i32], [12 x i32]* %arrayidx82, i64 0, i64 %idxprom83, !dbg !3720
  %74 = load i32, i32* %arrayidx84, align 4, !dbg !3720
  store i32 %74, i32* %j, align 4, !dbg !3724
  %75 = load double, double* %size, align 8, !dbg !3725
  %76 = load [12 x double]*, [12 x double]** %aelt.addr, align 8, !dbg !3726
  %77 = load i32, i32* %i, align 4, !dbg !3727
  %idxprom85 = sext i32 %77 to i64, !dbg !3726
  %arrayidx86 = getelementptr inbounds [12 x double], [12 x double]* %76, i64 %idxprom85, !dbg !3726
  %78 = load i32, i32* %nza, align 4, !dbg !3728
  %idxprom87 = sext i32 %78 to i64, !dbg !3726
  %arrayidx88 = getelementptr inbounds [12 x double], [12 x double]* %arrayidx86, i64 0, i64 %idxprom87, !dbg !3726
  %79 = load double, double* %arrayidx88, align 8, !dbg !3726
  %mul = fmul contract double %75, %79, !dbg !3729
  store double %mul, double* %scale, align 8, !dbg !3730
  store i32 0, i32* %nzrow, align 4, !dbg !3731
  br label %for.cond89, !dbg !3733

for.cond89:                                       ; preds = %for.inc176, %for.body80
  %80 = load i32, i32* %nzrow, align 4, !dbg !3734
  %81 = load i32*, i32** %arow.addr, align 8, !dbg !3736
  %82 = load i32, i32* %i, align 4, !dbg !3737
  %idxprom90 = sext i32 %82 to i64, !dbg !3736
  %arrayidx91 = getelementptr inbounds i32, i32* %81, i64 %idxprom90, !dbg !3736
  %83 = load i32, i32* %arrayidx91, align 4, !dbg !3736
  %cmp92 = icmp slt i32 %80, %83, !dbg !3738
  br i1 %cmp92, label %for.body93, label %for.end178, !dbg !3739

for.body93:                                       ; preds = %for.cond89
  %84 = load [12 x i32]*, [12 x i32]** %acol.addr, align 8, !dbg !3740
  %85 = load i32, i32* %i, align 4, !dbg !3742
  %idxprom94 = sext i32 %85 to i64, !dbg !3740
  %arrayidx95 = getelementptr inbounds [12 x i32], [12 x i32]* %84, i64 %idxprom94, !dbg !3740
  %86 = load i32, i32* %nzrow, align 4, !dbg !3743
  %idxprom96 = sext i32 %86 to i64, !dbg !3740
  %arrayidx97 = getelementptr inbounds [12 x i32], [12 x i32]* %arrayidx95, i64 0, i64 %idxprom96, !dbg !3740
  %87 = load i32, i32* %arrayidx97, align 4, !dbg !3740
  store i32 %87, i32* %jcol, align 4, !dbg !3744
  %88 = load [12 x double]*, [12 x double]** %aelt.addr, align 8, !dbg !3745
  %89 = load i32, i32* %i, align 4, !dbg !3746
  %idxprom98 = sext i32 %89 to i64, !dbg !3745
  %arrayidx99 = getelementptr inbounds [12 x double], [12 x double]* %88, i64 %idxprom98, !dbg !3745
  %90 = load i32, i32* %nzrow, align 4, !dbg !3747
  %idxprom100 = sext i32 %90 to i64, !dbg !3745
  %arrayidx101 = getelementptr inbounds [12 x double], [12 x double]* %arrayidx99, i64 0, i64 %idxprom100, !dbg !3745
  %91 = load double, double* %arrayidx101, align 8, !dbg !3745
  %92 = load double, double* %scale, align 8, !dbg !3748
  %mul102 = fmul contract double %91, %92, !dbg !3749
  store double %mul102, double* %va, align 8, !dbg !3750
  %93 = load i32, i32* %jcol, align 4, !dbg !3751
  %94 = load i32, i32* %j, align 4, !dbg !3753
  %cmp103 = icmp eq i32 %93, %94, !dbg !3754
  br i1 %cmp103, label %land.lhs.true, label %if.end108, !dbg !3755

land.lhs.true:                                    ; preds = %for.body93
  %95 = load i32, i32* %j, align 4, !dbg !3756
  %96 = load i32, i32* %i, align 4, !dbg !3757
  %cmp104 = icmp eq i32 %95, %96, !dbg !3758
  br i1 %cmp104, label %if.then105, label %if.end108, !dbg !3759

if.then105:                                       ; preds = %land.lhs.true
  %97 = load double, double* %va, align 8, !dbg !3760
  %98 = load double, double* %rcond.addr, align 8, !dbg !3762
  %add106 = fadd contract double %97, %98, !dbg !3763
  %99 = load double, double* %shift.addr, align 8, !dbg !3764
  %sub107 = fsub contract double %add106, %99, !dbg !3765
  store double %sub107, double* %va, align 8, !dbg !3766
  br label %if.end108, !dbg !3767

if.end108:                                        ; preds = %if.then105, %land.lhs.true, %for.body93
  %100 = load i32*, i32** %rowstr.addr, align 8, !dbg !3768
  %101 = load i32, i32* %j, align 4, !dbg !3770
  %idxprom109 = sext i32 %101 to i64, !dbg !3768
  %arrayidx110 = getelementptr inbounds i32, i32* %100, i64 %idxprom109, !dbg !3768
  %102 = load i32, i32* %arrayidx110, align 4, !dbg !3768
  store i32 %102, i32* %k, align 4, !dbg !3771
  br label %for.cond111, !dbg !3772

for.cond111:                                      ; preds = %for.inc168, %if.end108
  %103 = load i32, i32* %k, align 4, !dbg !3773
  %104 = load i32*, i32** %rowstr.addr, align 8, !dbg !3775
  %105 = load i32, i32* %j, align 4, !dbg !3776
  %add112 = add nsw i32 %105, 1, !dbg !3777
  %idxprom113 = sext i32 %add112 to i64, !dbg !3775
  %arrayidx114 = getelementptr inbounds i32, i32* %104, i64 %idxprom113, !dbg !3775
  %106 = load i32, i32* %arrayidx114, align 4, !dbg !3775
  %cmp115 = icmp slt i32 %103, %106, !dbg !3778
  br i1 %cmp115, label %for.body116, label %for.end170, !dbg !3779

for.body116:                                      ; preds = %for.cond111
  %107 = load i32*, i32** %colidx.addr, align 8, !dbg !3780
  %108 = load i32, i32* %k, align 4, !dbg !3783
  %idxprom117 = sext i32 %108 to i64, !dbg !3780
  %arrayidx118 = getelementptr inbounds i32, i32* %107, i64 %idxprom117, !dbg !3780
  %109 = load i32, i32* %arrayidx118, align 4, !dbg !3780
  %110 = load i32, i32* %jcol, align 4, !dbg !3784
  %cmp119 = icmp sgt i32 %109, %110, !dbg !3785
  br i1 %cmp119, label %if.then120, label %if.else, !dbg !3786

if.then120:                                       ; preds = %for.body116
  %111 = load i32*, i32** %rowstr.addr, align 8, !dbg !3787
  %112 = load i32, i32* %j, align 4, !dbg !3790
  %add121 = add nsw i32 %112, 1, !dbg !3791
  %idxprom122 = sext i32 %add121 to i64, !dbg !3787
  %arrayidx123 = getelementptr inbounds i32, i32* %111, i64 %idxprom122, !dbg !3787
  %113 = load i32, i32* %arrayidx123, align 4, !dbg !3787
  %sub124 = sub nsw i32 %113, 2, !dbg !3792
  store i32 %sub124, i32* %kk, align 4, !dbg !3793
  br label %for.cond125, !dbg !3794

for.cond125:                                      ; preds = %for.inc143, %if.then120
  %114 = load i32, i32* %kk, align 4, !dbg !3795
  %115 = load i32, i32* %k, align 4, !dbg !3797
  %cmp126 = icmp sge i32 %114, %115, !dbg !3798
  br i1 %cmp126, label %for.body127, label %for.end144, !dbg !3799

for.body127:                                      ; preds = %for.cond125
  %116 = load i32*, i32** %colidx.addr, align 8, !dbg !3800
  %117 = load i32, i32* %kk, align 4, !dbg !3803
  %idxprom128 = sext i32 %117 to i64, !dbg !3800
  %arrayidx129 = getelementptr inbounds i32, i32* %116, i64 %idxprom128, !dbg !3800
  %118 = load i32, i32* %arrayidx129, align 4, !dbg !3800
  %cmp130 = icmp sgt i32 %118, -1, !dbg !3804
  br i1 %cmp130, label %if.then131, label %if.end142, !dbg !3805

if.then131:                                       ; preds = %for.body127
  %119 = load double*, double** %a.addr, align 8, !dbg !3806
  %120 = load i32, i32* %kk, align 4, !dbg !3808
  %idxprom132 = sext i32 %120 to i64, !dbg !3806
  %arrayidx133 = getelementptr inbounds double, double* %119, i64 %idxprom132, !dbg !3806
  %121 = load double, double* %arrayidx133, align 8, !dbg !3806
  %122 = load double*, double** %a.addr, align 8, !dbg !3809
  %123 = load i32, i32* %kk, align 4, !dbg !3810
  %add134 = add nsw i32 %123, 1, !dbg !3811
  %idxprom135 = sext i32 %add134 to i64, !dbg !3809
  %arrayidx136 = getelementptr inbounds double, double* %122, i64 %idxprom135, !dbg !3809
  store double %121, double* %arrayidx136, align 8, !dbg !3812
  %124 = load i32*, i32** %colidx.addr, align 8, !dbg !3813
  %125 = load i32, i32* %kk, align 4, !dbg !3814
  %idxprom137 = sext i32 %125 to i64, !dbg !3813
  %arrayidx138 = getelementptr inbounds i32, i32* %124, i64 %idxprom137, !dbg !3813
  %126 = load i32, i32* %arrayidx138, align 4, !dbg !3813
  %127 = load i32*, i32** %colidx.addr, align 8, !dbg !3815
  %128 = load i32, i32* %kk, align 4, !dbg !3816
  %add139 = add nsw i32 %128, 1, !dbg !3817
  %idxprom140 = sext i32 %add139 to i64, !dbg !3815
  %arrayidx141 = getelementptr inbounds i32, i32* %127, i64 %idxprom140, !dbg !3815
  store i32 %126, i32* %arrayidx141, align 4, !dbg !3818
  br label %if.end142, !dbg !3819

if.end142:                                        ; preds = %if.then131, %for.body127
  br label %for.inc143, !dbg !3820

for.inc143:                                       ; preds = %if.end142
  %129 = load i32, i32* %kk, align 4, !dbg !3821
  %dec = add nsw i32 %129, -1, !dbg !3821
  store i32 %dec, i32* %kk, align 4, !dbg !3821
  br label %for.cond125, !dbg !3822, !llvm.loop !3823

for.end144:                                       ; preds = %for.cond125
  %130 = load i32, i32* %jcol, align 4, !dbg !3825
  %131 = load i32*, i32** %colidx.addr, align 8, !dbg !3826
  %132 = load i32, i32* %k, align 4, !dbg !3827
  %idxprom145 = sext i32 %132 to i64, !dbg !3826
  %arrayidx146 = getelementptr inbounds i32, i32* %131, i64 %idxprom145, !dbg !3826
  store i32 %130, i32* %arrayidx146, align 4, !dbg !3828
  %133 = load double*, double** %a.addr, align 8, !dbg !3829
  %134 = load i32, i32* %k, align 4, !dbg !3830
  %idxprom147 = sext i32 %134 to i64, !dbg !3829
  %arrayidx148 = getelementptr inbounds double, double* %133, i64 %idxprom147, !dbg !3829
  store double 0.000000e+00, double* %arrayidx148, align 8, !dbg !3831
  br label %for.end170, !dbg !3832

if.else:                                          ; preds = %for.body116
  %135 = load i32*, i32** %colidx.addr, align 8, !dbg !3833
  %136 = load i32, i32* %k, align 4, !dbg !3835
  %idxprom149 = sext i32 %136 to i64, !dbg !3833
  %arrayidx150 = getelementptr inbounds i32, i32* %135, i64 %idxprom149, !dbg !3833
  %137 = load i32, i32* %arrayidx150, align 4, !dbg !3833
  %cmp151 = icmp eq i32 %137, -1, !dbg !3836
  br i1 %cmp151, label %if.then152, label %if.else155, !dbg !3837

if.then152:                                       ; preds = %if.else
  %138 = load i32, i32* %jcol, align 4, !dbg !3838
  %139 = load i32*, i32** %colidx.addr, align 8, !dbg !3840
  %140 = load i32, i32* %k, align 4, !dbg !3841
  %idxprom153 = sext i32 %140 to i64, !dbg !3840
  %arrayidx154 = getelementptr inbounds i32, i32* %139, i64 %idxprom153, !dbg !3840
  store i32 %138, i32* %arrayidx154, align 4, !dbg !3842
  br label %for.end170, !dbg !3843

if.else155:                                       ; preds = %if.else
  %141 = load i32*, i32** %colidx.addr, align 8, !dbg !3844
  %142 = load i32, i32* %k, align 4, !dbg !3846
  %idxprom156 = sext i32 %142 to i64, !dbg !3844
  %arrayidx157 = getelementptr inbounds i32, i32* %141, i64 %idxprom156, !dbg !3844
  %143 = load i32, i32* %arrayidx157, align 4, !dbg !3844
  %144 = load i32, i32* %jcol, align 4, !dbg !3847
  %cmp158 = icmp eq i32 %143, %144, !dbg !3848
  br i1 %cmp158, label %if.then159, label %if.end165, !dbg !3849

if.then159:                                       ; preds = %if.else155
  %145 = load i32*, i32** %nzloc.addr, align 8, !dbg !3850
  %146 = load i32, i32* %j, align 4, !dbg !3852
  %idxprom160 = sext i32 %146 to i64, !dbg !3850
  %arrayidx161 = getelementptr inbounds i32, i32* %145, i64 %idxprom160, !dbg !3850
  %147 = load i32, i32* %arrayidx161, align 4, !dbg !3850
  %add162 = add nsw i32 %147, 1, !dbg !3853
  %148 = load i32*, i32** %nzloc.addr, align 8, !dbg !3854
  %149 = load i32, i32* %j, align 4, !dbg !3855
  %idxprom163 = sext i32 %149 to i64, !dbg !3854
  %arrayidx164 = getelementptr inbounds i32, i32* %148, i64 %idxprom163, !dbg !3854
  store i32 %add162, i32* %arrayidx164, align 4, !dbg !3856
  br label %for.end170, !dbg !3857

if.end165:                                        ; preds = %if.else155
  br label %if.end166

if.end166:                                        ; preds = %if.end165
  br label %if.end167

if.end167:                                        ; preds = %if.end166
  br label %for.inc168, !dbg !3858

for.inc168:                                       ; preds = %if.end167
  %150 = load i32, i32* %k, align 4, !dbg !3859
  %inc169 = add nsw i32 %150, 1, !dbg !3859
  store i32 %inc169, i32* %k, align 4, !dbg !3859
  br label %for.cond111, !dbg !3860, !llvm.loop !3861

for.end170:                                       ; preds = %if.then159, %if.then152, %for.end144, %for.cond111
  %151 = load double*, double** %a.addr, align 8, !dbg !3863
  %152 = load i32, i32* %k, align 4, !dbg !3864
  %idxprom171 = sext i32 %152 to i64, !dbg !3863
  %arrayidx172 = getelementptr inbounds double, double* %151, i64 %idxprom171, !dbg !3863
  %153 = load double, double* %arrayidx172, align 8, !dbg !3863
  %154 = load double, double* %va, align 8, !dbg !3865
  %add173 = fadd contract double %153, %154, !dbg !3866
  %155 = load double*, double** %a.addr, align 8, !dbg !3867
  %156 = load i32, i32* %k, align 4, !dbg !3868
  %idxprom174 = sext i32 %156 to i64, !dbg !3867
  %arrayidx175 = getelementptr inbounds double, double* %155, i64 %idxprom174, !dbg !3867
  store double %add173, double* %arrayidx175, align 8, !dbg !3869
  br label %for.inc176, !dbg !3870

for.inc176:                                       ; preds = %for.end170
  %157 = load i32, i32* %nzrow, align 4, !dbg !3871
  %inc177 = add nsw i32 %157, 1, !dbg !3871
  store i32 %inc177, i32* %nzrow, align 4, !dbg !3871
  br label %for.cond89, !dbg !3872, !llvm.loop !3873

for.end178:                                       ; preds = %for.cond89
  br label %for.inc179, !dbg !3875

for.inc179:                                       ; preds = %for.end178
  %158 = load i32, i32* %nza, align 4, !dbg !3876
  %inc180 = add nsw i32 %158, 1, !dbg !3876
  store i32 %inc180, i32* %nza, align 4, !dbg !3876
  br label %for.cond76, !dbg !3877, !llvm.loop !3878

for.end181:                                       ; preds = %for.cond76
  %159 = load double, double* %size, align 8, !dbg !3880
  %160 = load double, double* %ratio, align 8, !dbg !3881
  %mul182 = fmul contract double %159, %160, !dbg !3882
  store double %mul182, double* %size, align 8, !dbg !3883
  br label %for.inc183, !dbg !3884

for.inc183:                                       ; preds = %for.end181
  %161 = load i32, i32* %i, align 4, !dbg !3885
  %inc184 = add nsw i32 %161, 1, !dbg !3885
  store i32 %inc184, i32* %i, align 4, !dbg !3885
  br label %for.cond73, !dbg !3886, !llvm.loop !3887

for.end185:                                       ; preds = %for.cond73
  store i32 1, i32* %j, align 4, !dbg !3889
  br label %for.cond186, !dbg !3891

for.cond186:                                      ; preds = %for.inc197, %for.end185
  %162 = load i32, i32* %j, align 4, !dbg !3892
  %163 = load i32, i32* %nrows, align 4, !dbg !3894
  %cmp187 = icmp slt i32 %162, %163, !dbg !3895
  br i1 %cmp187, label %for.body188, label %for.end199, !dbg !3896

for.body188:                                      ; preds = %for.cond186
  %164 = load i32*, i32** %nzloc.addr, align 8, !dbg !3897
  %165 = load i32, i32* %j, align 4, !dbg !3899
  %idxprom189 = sext i32 %165 to i64, !dbg !3897
  %arrayidx190 = getelementptr inbounds i32, i32* %164, i64 %idxprom189, !dbg !3897
  %166 = load i32, i32* %arrayidx190, align 4, !dbg !3897
  %167 = load i32*, i32** %nzloc.addr, align 8, !dbg !3900
  %168 = load i32, i32* %j, align 4, !dbg !3901
  %sub191 = sub nsw i32 %168, 1, !dbg !3902
  %idxprom192 = sext i32 %sub191 to i64, !dbg !3900
  %arrayidx193 = getelementptr inbounds i32, i32* %167, i64 %idxprom192, !dbg !3900
  %169 = load i32, i32* %arrayidx193, align 4, !dbg !3900
  %add194 = add nsw i32 %166, %169, !dbg !3903
  %170 = load i32*, i32** %nzloc.addr, align 8, !dbg !3904
  %171 = load i32, i32* %j, align 4, !dbg !3905
  %idxprom195 = sext i32 %171 to i64, !dbg !3904
  %arrayidx196 = getelementptr inbounds i32, i32* %170, i64 %idxprom195, !dbg !3904
  store i32 %add194, i32* %arrayidx196, align 4, !dbg !3906
  br label %for.inc197, !dbg !3907

for.inc197:                                       ; preds = %for.body188
  %172 = load i32, i32* %j, align 4, !dbg !3908
  %inc198 = add nsw i32 %172, 1, !dbg !3908
  store i32 %inc198, i32* %j, align 4, !dbg !3908
  br label %for.cond186, !dbg !3909, !llvm.loop !3910

for.end199:                                       ; preds = %for.cond186
  store i32 0, i32* %j, align 4, !dbg !3912
  br label %for.cond200, !dbg !3914

for.cond200:                                      ; preds = %for.inc236, %for.end199
  %173 = load i32, i32* %j, align 4, !dbg !3915
  %174 = load i32, i32* %nrows, align 4, !dbg !3917
  %cmp201 = icmp slt i32 %173, %174, !dbg !3918
  br i1 %cmp201, label %for.body202, label %for.end238, !dbg !3919

for.body202:                                      ; preds = %for.cond200
  %175 = load i32, i32* %j, align 4, !dbg !3920
  %cmp203 = icmp sgt i32 %175, 0, !dbg !3923
  br i1 %cmp203, label %if.then204, label %if.else211, !dbg !3924

if.then204:                                       ; preds = %for.body202
  %176 = load i32*, i32** %rowstr.addr, align 8, !dbg !3925
  %177 = load i32, i32* %j, align 4, !dbg !3927
  %idxprom205 = sext i32 %177 to i64, !dbg !3925
  %arrayidx206 = getelementptr inbounds i32, i32* %176, i64 %idxprom205, !dbg !3925
  %178 = load i32, i32* %arrayidx206, align 4, !dbg !3925
  %179 = load i32*, i32** %nzloc.addr, align 8, !dbg !3928
  %180 = load i32, i32* %j, align 4, !dbg !3929
  %sub207 = sub nsw i32 %180, 1, !dbg !3930
  %idxprom208 = sext i32 %sub207 to i64, !dbg !3928
  %arrayidx209 = getelementptr inbounds i32, i32* %179, i64 %idxprom208, !dbg !3928
  %181 = load i32, i32* %arrayidx209, align 4, !dbg !3928
  %sub210 = sub nsw i32 %178, %181, !dbg !3931
  store i32 %sub210, i32* %j1, align 4, !dbg !3932
  br label %if.end212, !dbg !3933

if.else211:                                       ; preds = %for.body202
  store i32 0, i32* %j1, align 4, !dbg !3934
  br label %if.end212

if.end212:                                        ; preds = %if.else211, %if.then204
  %182 = load i32*, i32** %rowstr.addr, align 8, !dbg !3936
  %183 = load i32, i32* %j, align 4, !dbg !3937
  %add213 = add nsw i32 %183, 1, !dbg !3938
  %idxprom214 = sext i32 %add213 to i64, !dbg !3936
  %arrayidx215 = getelementptr inbounds i32, i32* %182, i64 %idxprom214, !dbg !3936
  %184 = load i32, i32* %arrayidx215, align 4, !dbg !3936
  %185 = load i32*, i32** %nzloc.addr, align 8, !dbg !3939
  %186 = load i32, i32* %j, align 4, !dbg !3940
  %idxprom216 = sext i32 %186 to i64, !dbg !3939
  %arrayidx217 = getelementptr inbounds i32, i32* %185, i64 %idxprom216, !dbg !3939
  %187 = load i32, i32* %arrayidx217, align 4, !dbg !3939
  %sub218 = sub nsw i32 %184, %187, !dbg !3941
  store i32 %sub218, i32* %j2, align 4, !dbg !3942
  %188 = load i32*, i32** %rowstr.addr, align 8, !dbg !3943
  %189 = load i32, i32* %j, align 4, !dbg !3944
  %idxprom219 = sext i32 %189 to i64, !dbg !3943
  %arrayidx220 = getelementptr inbounds i32, i32* %188, i64 %idxprom219, !dbg !3943
  %190 = load i32, i32* %arrayidx220, align 4, !dbg !3943
  store i32 %190, i32* %nza, align 4, !dbg !3945
  %191 = load i32, i32* %j1, align 4, !dbg !3946
  store i32 %191, i32* %k, align 4, !dbg !3948
  br label %for.cond221, !dbg !3949

for.cond221:                                      ; preds = %for.inc233, %if.end212
  %192 = load i32, i32* %k, align 4, !dbg !3950
  %193 = load i32, i32* %j2, align 4, !dbg !3952
  %cmp222 = icmp slt i32 %192, %193, !dbg !3953
  br i1 %cmp222, label %for.body223, label %for.end235, !dbg !3954

for.body223:                                      ; preds = %for.cond221
  %194 = load double*, double** %a.addr, align 8, !dbg !3955
  %195 = load i32, i32* %nza, align 4, !dbg !3957
  %idxprom224 = sext i32 %195 to i64, !dbg !3955
  %arrayidx225 = getelementptr inbounds double, double* %194, i64 %idxprom224, !dbg !3955
  %196 = load double, double* %arrayidx225, align 8, !dbg !3955
  %197 = load double*, double** %a.addr, align 8, !dbg !3958
  %198 = load i32, i32* %k, align 4, !dbg !3959
  %idxprom226 = sext i32 %198 to i64, !dbg !3958
  %arrayidx227 = getelementptr inbounds double, double* %197, i64 %idxprom226, !dbg !3958
  store double %196, double* %arrayidx227, align 8, !dbg !3960
  %199 = load i32*, i32** %colidx.addr, align 8, !dbg !3961
  %200 = load i32, i32* %nza, align 4, !dbg !3962
  %idxprom228 = sext i32 %200 to i64, !dbg !3961
  %arrayidx229 = getelementptr inbounds i32, i32* %199, i64 %idxprom228, !dbg !3961
  %201 = load i32, i32* %arrayidx229, align 4, !dbg !3961
  %202 = load i32*, i32** %colidx.addr, align 8, !dbg !3963
  %203 = load i32, i32* %k, align 4, !dbg !3964
  %idxprom230 = sext i32 %203 to i64, !dbg !3963
  %arrayidx231 = getelementptr inbounds i32, i32* %202, i64 %idxprom230, !dbg !3963
  store i32 %201, i32* %arrayidx231, align 4, !dbg !3965
  %204 = load i32, i32* %nza, align 4, !dbg !3966
  %add232 = add nsw i32 %204, 1, !dbg !3967
  store i32 %add232, i32* %nza, align 4, !dbg !3968
  br label %for.inc233, !dbg !3969

for.inc233:                                       ; preds = %for.body223
  %205 = load i32, i32* %k, align 4, !dbg !3970
  %inc234 = add nsw i32 %205, 1, !dbg !3970
  store i32 %inc234, i32* %k, align 4, !dbg !3970
  br label %for.cond221, !dbg !3971, !llvm.loop !3972

for.end235:                                       ; preds = %for.cond221
  br label %for.inc236, !dbg !3974

for.inc236:                                       ; preds = %for.end235
  %206 = load i32, i32* %j, align 4, !dbg !3975
  %inc237 = add nsw i32 %206, 1, !dbg !3975
  store i32 %inc237, i32* %j, align 4, !dbg !3975
  br label %for.cond200, !dbg !3976, !llvm.loop !3977

for.end238:                                       ; preds = %for.cond200
  store i32 1, i32* %j, align 4, !dbg !3979
  br label %for.cond239, !dbg !3981

for.cond239:                                      ; preds = %for.inc251, %for.end238
  %207 = load i32, i32* %j, align 4, !dbg !3982
  %208 = load i32, i32* %nrows, align 4, !dbg !3984
  %add240 = add nsw i32 %208, 1, !dbg !3985
  %cmp241 = icmp slt i32 %207, %add240, !dbg !3986
  br i1 %cmp241, label %for.body242, label %for.end253, !dbg !3987

for.body242:                                      ; preds = %for.cond239
  %209 = load i32*, i32** %rowstr.addr, align 8, !dbg !3988
  %210 = load i32, i32* %j, align 4, !dbg !3990
  %idxprom243 = sext i32 %210 to i64, !dbg !3988
  %arrayidx244 = getelementptr inbounds i32, i32* %209, i64 %idxprom243, !dbg !3988
  %211 = load i32, i32* %arrayidx244, align 4, !dbg !3988
  %212 = load i32*, i32** %nzloc.addr, align 8, !dbg !3991
  %213 = load i32, i32* %j, align 4, !dbg !3992
  %sub245 = sub nsw i32 %213, 1, !dbg !3993
  %idxprom246 = sext i32 %sub245 to i64, !dbg !3991
  %arrayidx247 = getelementptr inbounds i32, i32* %212, i64 %idxprom246, !dbg !3991
  %214 = load i32, i32* %arrayidx247, align 4, !dbg !3991
  %sub248 = sub nsw i32 %211, %214, !dbg !3994
  %215 = load i32*, i32** %rowstr.addr, align 8, !dbg !3995
  %216 = load i32, i32* %j, align 4, !dbg !3996
  %idxprom249 = sext i32 %216 to i64, !dbg !3995
  %arrayidx250 = getelementptr inbounds i32, i32* %215, i64 %idxprom249, !dbg !3995
  store i32 %sub248, i32* %arrayidx250, align 4, !dbg !3997
  br label %for.inc251, !dbg !3998

for.inc251:                                       ; preds = %for.body242
  %217 = load i32, i32* %j, align 4, !dbg !3999
  %inc252 = add nsw i32 %217, 1, !dbg !3999
  store i32 %inc252, i32* %j, align 4, !dbg !3999
  br label %for.cond239, !dbg !4000, !llvm.loop !4001

for.end253:                                       ; preds = %for.cond239
  %218 = load i32*, i32** %rowstr.addr, align 8, !dbg !4003
  %219 = load i32, i32* %nrows, align 4, !dbg !4004
  %idxprom254 = sext i32 %219 to i64, !dbg !4003
  %arrayidx255 = getelementptr inbounds i32, i32* %218, i64 %idxprom254, !dbg !4003
  %220 = load i32, i32* %arrayidx255, align 4, !dbg !4003
  %sub256 = sub nsw i32 %220, 1, !dbg !4005
  store i32 %sub256, i32* %nza, align 4, !dbg !4006
  ret void, !dbg !4007
}

; Function Attrs: noinline nounwind uwtable
define internal i32 @_ZL6icnvrtdi(double %x, i32 %ipwr2) #0 !dbg !4008 {
entry:
  %x.addr = alloca double, align 8
  %ipwr2.addr = alloca i32, align 4
  store double %x, double* %x.addr, align 8
  call void @llvm.dbg.declare(metadata double* %x.addr, metadata !4011, metadata !DIExpression()), !dbg !4012
  store i32 %ipwr2, i32* %ipwr2.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %ipwr2.addr, metadata !4013, metadata !DIExpression()), !dbg !4014
  %0 = load i32, i32* %ipwr2.addr, align 4, !dbg !4015
  %conv = sitofp i32 %0 to double, !dbg !4015
  %1 = load double, double* %x.addr, align 8, !dbg !4016
  %mul = fmul contract double %conv, %1, !dbg !4017
  %conv1 = fptosi double %mul to i32, !dbg !4018
  ret i32 %conv1, !dbg !4019
}

; Function Attrs: noreturn nounwind
declare dso_local void @exit(i32) #7

declare dso_local i32 @cudaFree(i8*) #3

; Function Attrs: nounwind readnone speculatable
declare double @llvm.ceil.f64(double) #1

; Function Attrs: noinline uwtable
define internal i32 @_ZL10cudaMallocIiE9cudaErrorPPT_m(i32** %devPtr, i64 %size) #2 !dbg !4020 {
entry:
  %devPtr.addr = alloca i32**, align 8
  %size.addr = alloca i64, align 8
  store i32** %devPtr, i32*** %devPtr.addr, align 8
  call void @llvm.dbg.declare(metadata i32*** %devPtr.addr, metadata !4028, metadata !DIExpression()), !dbg !4029
  store i64 %size, i64* %size.addr, align 8
  call void @llvm.dbg.declare(metadata i64* %size.addr, metadata !4030, metadata !DIExpression()), !dbg !4031
  %0 = load i32**, i32*** %devPtr.addr, align 8, !dbg !4032
  %1 = bitcast i32** %0 to i8*, !dbg !4032
  %2 = bitcast i8* %1 to i8**, !dbg !4033
  %3 = load i64, i64* %size.addr, align 8, !dbg !4034
  %call = call i32 @cudaMalloc(i8** %2, i64 %3), !dbg !4035
  ret i32 %call, !dbg !4036
}

; Function Attrs: noinline uwtable
define internal i32 @_ZL10cudaMallocIdE9cudaErrorPPT_m(double** %devPtr, i64 %size) #2 !dbg !4037 {
entry:
  %devPtr.addr = alloca double**, align 8
  %size.addr = alloca i64, align 8
  store double** %devPtr, double*** %devPtr.addr, align 8
  call void @llvm.dbg.declare(metadata double*** %devPtr.addr, metadata !4043, metadata !DIExpression()), !dbg !4044
  store i64 %size, i64* %size.addr, align 8
  call void @llvm.dbg.declare(metadata i64* %size.addr, metadata !4045, metadata !DIExpression()), !dbg !4046
  %0 = load double**, double*** %devPtr.addr, align 8, !dbg !4047
  %1 = bitcast double** %0 to i8*, !dbg !4047
  %2 = bitcast i8* %1 to i8**, !dbg !4048
  %3 = load i64, i64* %size.addr, align 8, !dbg !4049
  %call = call i32 @cudaMalloc(i8** %2, i64 %3), !dbg !4050
  ret i32 %call, !dbg !4051
}

declare dso_local i32 @cudaMalloc(i8**, i64) #3

attributes #0 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { noinline uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { noinline norecurse uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { argmemonly nounwind }
attributes #7 = { noreturn nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #8 = { nounwind }
attributes #9 = { noreturn nounwind }

!llvm.module.flags = !{!1155, !1156, !1157, !1158}
!llvm.dbg.cu = !{!2}
!llvm.ident = !{!1159}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "colidx_device", scope: !2, file: !3, line: 118, type: !98, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 9.0.0 (https://github.com/yebinchon/llvm-project/ a1efa594106d738d0b74c9e4e2b9b779eb8b7d25)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !96, globals: !111, imports: !404, nameTableKind: None)
!3 = !DIFile(filename: "cg.cu", directory: "/scratch/ah7226/NPB-GPU/CUDA/CG")
!4 = !{!5, !14}
!5 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "cudaMemcpyKind", file: !6, line: 796, baseType: !7, size: 32, elements: !8, identifier: "_ZTS14cudaMemcpyKind")
!6 = !DIFile(filename: "/usr/local/cuda/include/driver_types.h", directory: "")
!7 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!8 = !{!9, !10, !11, !12, !13}
!9 = !DIEnumerator(name: "cudaMemcpyHostToHost", value: 0, isUnsigned: true)
!10 = !DIEnumerator(name: "cudaMemcpyHostToDevice", value: 1, isUnsigned: true)
!11 = !DIEnumerator(name: "cudaMemcpyDeviceToHost", value: 2, isUnsigned: true)
!12 = !DIEnumerator(name: "cudaMemcpyDeviceToDevice", value: 3, isUnsigned: true)
!13 = !DIEnumerator(name: "cudaMemcpyDefault", value: 4, isUnsigned: true)
!14 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "cudaError", file: !6, line: 150, baseType: !7, size: 32, elements: !15, identifier: "_ZTS9cudaError")
!15 = !{!16, !17, !18, !19, !20, !21, !22, !23, !24, !25, !26, !27, !28, !29, !30, !31, !32, !33, !34, !35, !36, !37, !38, !39, !40, !41, !42, !43, !44, !45, !46, !47, !48, !49, !50, !51, !52, !53, !54, !55, !56, !57, !58, !59, !60, !61, !62, !63, !64, !65, !66, !67, !68, !69, !70, !71, !72, !73, !74, !75, !76, !77, !78, !79, !80, !81, !82, !83, !84, !85, !86, !87, !88, !89, !90, !91, !92, !93, !94, !95}
!16 = !DIEnumerator(name: "cudaSuccess", value: 0, isUnsigned: true)
!17 = !DIEnumerator(name: "cudaErrorMissingConfiguration", value: 1, isUnsigned: true)
!18 = !DIEnumerator(name: "cudaErrorMemoryAllocation", value: 2, isUnsigned: true)
!19 = !DIEnumerator(name: "cudaErrorInitializationError", value: 3, isUnsigned: true)
!20 = !DIEnumerator(name: "cudaErrorLaunchFailure", value: 4, isUnsigned: true)
!21 = !DIEnumerator(name: "cudaErrorPriorLaunchFailure", value: 5, isUnsigned: true)
!22 = !DIEnumerator(name: "cudaErrorLaunchTimeout", value: 6, isUnsigned: true)
!23 = !DIEnumerator(name: "cudaErrorLaunchOutOfResources", value: 7, isUnsigned: true)
!24 = !DIEnumerator(name: "cudaErrorInvalidDeviceFunction", value: 8, isUnsigned: true)
!25 = !DIEnumerator(name: "cudaErrorInvalidConfiguration", value: 9, isUnsigned: true)
!26 = !DIEnumerator(name: "cudaErrorInvalidDevice", value: 10, isUnsigned: true)
!27 = !DIEnumerator(name: "cudaErrorInvalidValue", value: 11, isUnsigned: true)
!28 = !DIEnumerator(name: "cudaErrorInvalidPitchValue", value: 12, isUnsigned: true)
!29 = !DIEnumerator(name: "cudaErrorInvalidSymbol", value: 13, isUnsigned: true)
!30 = !DIEnumerator(name: "cudaErrorMapBufferObjectFailed", value: 14, isUnsigned: true)
!31 = !DIEnumerator(name: "cudaErrorUnmapBufferObjectFailed", value: 15, isUnsigned: true)
!32 = !DIEnumerator(name: "cudaErrorInvalidHostPointer", value: 16, isUnsigned: true)
!33 = !DIEnumerator(name: "cudaErrorInvalidDevicePointer", value: 17, isUnsigned: true)
!34 = !DIEnumerator(name: "cudaErrorInvalidTexture", value: 18, isUnsigned: true)
!35 = !DIEnumerator(name: "cudaErrorInvalidTextureBinding", value: 19, isUnsigned: true)
!36 = !DIEnumerator(name: "cudaErrorInvalidChannelDescriptor", value: 20, isUnsigned: true)
!37 = !DIEnumerator(name: "cudaErrorInvalidMemcpyDirection", value: 21, isUnsigned: true)
!38 = !DIEnumerator(name: "cudaErrorAddressOfConstant", value: 22, isUnsigned: true)
!39 = !DIEnumerator(name: "cudaErrorTextureFetchFailed", value: 23, isUnsigned: true)
!40 = !DIEnumerator(name: "cudaErrorTextureNotBound", value: 24, isUnsigned: true)
!41 = !DIEnumerator(name: "cudaErrorSynchronizationError", value: 25, isUnsigned: true)
!42 = !DIEnumerator(name: "cudaErrorInvalidFilterSetting", value: 26, isUnsigned: true)
!43 = !DIEnumerator(name: "cudaErrorInvalidNormSetting", value: 27, isUnsigned: true)
!44 = !DIEnumerator(name: "cudaErrorMixedDeviceExecution", value: 28, isUnsigned: true)
!45 = !DIEnumerator(name: "cudaErrorCudartUnloading", value: 29, isUnsigned: true)
!46 = !DIEnumerator(name: "cudaErrorUnknown", value: 30, isUnsigned: true)
!47 = !DIEnumerator(name: "cudaErrorNotYetImplemented", value: 31, isUnsigned: true)
!48 = !DIEnumerator(name: "cudaErrorMemoryValueTooLarge", value: 32, isUnsigned: true)
!49 = !DIEnumerator(name: "cudaErrorInvalidResourceHandle", value: 33, isUnsigned: true)
!50 = !DIEnumerator(name: "cudaErrorNotReady", value: 34, isUnsigned: true)
!51 = !DIEnumerator(name: "cudaErrorInsufficientDriver", value: 35, isUnsigned: true)
!52 = !DIEnumerator(name: "cudaErrorSetOnActiveProcess", value: 36, isUnsigned: true)
!53 = !DIEnumerator(name: "cudaErrorInvalidSurface", value: 37, isUnsigned: true)
!54 = !DIEnumerator(name: "cudaErrorNoDevice", value: 38, isUnsigned: true)
!55 = !DIEnumerator(name: "cudaErrorECCUncorrectable", value: 39, isUnsigned: true)
!56 = !DIEnumerator(name: "cudaErrorSharedObjectSymbolNotFound", value: 40, isUnsigned: true)
!57 = !DIEnumerator(name: "cudaErrorSharedObjectInitFailed", value: 41, isUnsigned: true)
!58 = !DIEnumerator(name: "cudaErrorUnsupportedLimit", value: 42, isUnsigned: true)
!59 = !DIEnumerator(name: "cudaErrorDuplicateVariableName", value: 43, isUnsigned: true)
!60 = !DIEnumerator(name: "cudaErrorDuplicateTextureName", value: 44, isUnsigned: true)
!61 = !DIEnumerator(name: "cudaErrorDuplicateSurfaceName", value: 45, isUnsigned: true)
!62 = !DIEnumerator(name: "cudaErrorDevicesUnavailable", value: 46, isUnsigned: true)
!63 = !DIEnumerator(name: "cudaErrorInvalidKernelImage", value: 47, isUnsigned: true)
!64 = !DIEnumerator(name: "cudaErrorNoKernelImageForDevice", value: 48, isUnsigned: true)
!65 = !DIEnumerator(name: "cudaErrorIncompatibleDriverContext", value: 49, isUnsigned: true)
!66 = !DIEnumerator(name: "cudaErrorPeerAccessAlreadyEnabled", value: 50, isUnsigned: true)
!67 = !DIEnumerator(name: "cudaErrorPeerAccessNotEnabled", value: 51, isUnsigned: true)
!68 = !DIEnumerator(name: "cudaErrorDeviceAlreadyInUse", value: 54, isUnsigned: true)
!69 = !DIEnumerator(name: "cudaErrorProfilerDisabled", value: 55, isUnsigned: true)
!70 = !DIEnumerator(name: "cudaErrorProfilerNotInitialized", value: 56, isUnsigned: true)
!71 = !DIEnumerator(name: "cudaErrorProfilerAlreadyStarted", value: 57, isUnsigned: true)
!72 = !DIEnumerator(name: "cudaErrorProfilerAlreadyStopped", value: 58, isUnsigned: true)
!73 = !DIEnumerator(name: "cudaErrorAssert", value: 59, isUnsigned: true)
!74 = !DIEnumerator(name: "cudaErrorTooManyPeers", value: 60, isUnsigned: true)
!75 = !DIEnumerator(name: "cudaErrorHostMemoryAlreadyRegistered", value: 61, isUnsigned: true)
!76 = !DIEnumerator(name: "cudaErrorHostMemoryNotRegistered", value: 62, isUnsigned: true)
!77 = !DIEnumerator(name: "cudaErrorOperatingSystem", value: 63, isUnsigned: true)
!78 = !DIEnumerator(name: "cudaErrorPeerAccessUnsupported", value: 64, isUnsigned: true)
!79 = !DIEnumerator(name: "cudaErrorLaunchMaxDepthExceeded", value: 65, isUnsigned: true)
!80 = !DIEnumerator(name: "cudaErrorLaunchFileScopedTex", value: 66, isUnsigned: true)
!81 = !DIEnumerator(name: "cudaErrorLaunchFileScopedSurf", value: 67, isUnsigned: true)
!82 = !DIEnumerator(name: "cudaErrorSyncDepthExceeded", value: 68, isUnsigned: true)
!83 = !DIEnumerator(name: "cudaErrorLaunchPendingCountExceeded", value: 69, isUnsigned: true)
!84 = !DIEnumerator(name: "cudaErrorNotPermitted", value: 70, isUnsigned: true)
!85 = !DIEnumerator(name: "cudaErrorNotSupported", value: 71, isUnsigned: true)
!86 = !DIEnumerator(name: "cudaErrorHardwareStackError", value: 72, isUnsigned: true)
!87 = !DIEnumerator(name: "cudaErrorIllegalInstruction", value: 73, isUnsigned: true)
!88 = !DIEnumerator(name: "cudaErrorMisalignedAddress", value: 74, isUnsigned: true)
!89 = !DIEnumerator(name: "cudaErrorInvalidAddressSpace", value: 75, isUnsigned: true)
!90 = !DIEnumerator(name: "cudaErrorInvalidPc", value: 76, isUnsigned: true)
!91 = !DIEnumerator(name: "cudaErrorIllegalAddress", value: 77, isUnsigned: true)
!92 = !DIEnumerator(name: "cudaErrorInvalidPtx", value: 78, isUnsigned: true)
!93 = !DIEnumerator(name: "cudaErrorInvalidGraphicsContext", value: 79, isUnsigned: true)
!94 = !DIEnumerator(name: "cudaErrorStartupFailure", value: 127, isUnsigned: true)
!95 = !DIEnumerator(name: "cudaErrorApiFailureBase", value: 10000, isUnsigned: true)
!96 = !{!97, !98, !99, !101, !105, !106, !100, !108, !110}
!97 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!98 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !97, size: 64)
!99 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !100, size: 64)
!100 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
!101 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !102, size: 64)
!102 = !DICompositeType(tag: DW_TAG_array_type, baseType: !97, size: 384, elements: !103)
!103 = !{!104}
!104 = !DISubrange(count: 12)
!105 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!106 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !107, size: 64)
!107 = !DICompositeType(tag: DW_TAG_array_type, baseType: !100, size: 768, elements: !103)
!108 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !109, size: 64)
!109 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!110 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !105, size: 64)
!111 = !{!0, !112, !114, !116, !118, !120, !122, !124, !126, !128, !130, !132, !134, !136, !138, !140, !142, !144, !146, !148, !150, !152, !157, !159, !161, !163, !165, !167, !169, !171, !173, !175, !177, !179, !181, !183, !185, !187, !189, !191, !193, !195, !197, !199, !201, !203, !205, !207, !209, !211, !213, !215, !217, !219, !221, !223, !225, !227, !229, !231, !233, !235, !237, !239, !241, !243, !245, !247, !249, !251, !253, !255, !257, !259, !261, !263, !265, !267, !269, !271, !273, !275, !277, !279, !281, !283, !285, !287, !289, !364, !366, !368, !370, !372, !374, !376, !378, !380, !382, !384, !386, !388, !390, !392, !394, !396, !398, !400, !402}
!112 = !DIGlobalVariableExpression(var: !113, expr: !DIExpression())
!113 = distinct !DIGlobalVariable(name: "rowstr_device", scope: !2, file: !3, line: 119, type: !98, isLocal: false, isDefinition: true)
!114 = !DIGlobalVariableExpression(var: !115, expr: !DIExpression())
!115 = distinct !DIGlobalVariable(name: "a_device", scope: !2, file: !3, line: 120, type: !99, isLocal: false, isDefinition: true)
!116 = !DIGlobalVariableExpression(var: !117, expr: !DIExpression())
!117 = distinct !DIGlobalVariable(name: "p_device", scope: !2, file: !3, line: 121, type: !99, isLocal: false, isDefinition: true)
!118 = !DIGlobalVariableExpression(var: !119, expr: !DIExpression())
!119 = distinct !DIGlobalVariable(name: "q_device", scope: !2, file: !3, line: 122, type: !99, isLocal: false, isDefinition: true)
!120 = !DIGlobalVariableExpression(var: !121, expr: !DIExpression())
!121 = distinct !DIGlobalVariable(name: "r_device", scope: !2, file: !3, line: 123, type: !99, isLocal: false, isDefinition: true)
!122 = !DIGlobalVariableExpression(var: !123, expr: !DIExpression())
!123 = distinct !DIGlobalVariable(name: "x_device", scope: !2, file: !3, line: 124, type: !99, isLocal: false, isDefinition: true)
!124 = !DIGlobalVariableExpression(var: !125, expr: !DIExpression())
!125 = distinct !DIGlobalVariable(name: "z_device", scope: !2, file: !3, line: 125, type: !99, isLocal: false, isDefinition: true)
!126 = !DIGlobalVariableExpression(var: !127, expr: !DIExpression())
!127 = distinct !DIGlobalVariable(name: "rho_device", scope: !2, file: !3, line: 126, type: !99, isLocal: false, isDefinition: true)
!128 = !DIGlobalVariableExpression(var: !129, expr: !DIExpression())
!129 = distinct !DIGlobalVariable(name: "d_device", scope: !2, file: !3, line: 127, type: !99, isLocal: false, isDefinition: true)
!130 = !DIGlobalVariableExpression(var: !131, expr: !DIExpression())
!131 = distinct !DIGlobalVariable(name: "alpha_device", scope: !2, file: !3, line: 128, type: !99, isLocal: false, isDefinition: true)
!132 = !DIGlobalVariableExpression(var: !133, expr: !DIExpression())
!133 = distinct !DIGlobalVariable(name: "beta_device", scope: !2, file: !3, line: 129, type: !99, isLocal: false, isDefinition: true)
!134 = !DIGlobalVariableExpression(var: !135, expr: !DIExpression())
!135 = distinct !DIGlobalVariable(name: "sum_device", scope: !2, file: !3, line: 130, type: !99, isLocal: false, isDefinition: true)
!136 = !DIGlobalVariableExpression(var: !137, expr: !DIExpression())
!137 = distinct !DIGlobalVariable(name: "norm_temp1_device", scope: !2, file: !3, line: 131, type: !99, isLocal: false, isDefinition: true)
!138 = !DIGlobalVariableExpression(var: !139, expr: !DIExpression())
!139 = distinct !DIGlobalVariable(name: "norm_temp2_device", scope: !2, file: !3, line: 132, type: !99, isLocal: false, isDefinition: true)
!140 = !DIGlobalVariableExpression(var: !141, expr: !DIExpression())
!141 = distinct !DIGlobalVariable(name: "global_data", scope: !2, file: !3, line: 133, type: !99, isLocal: false, isDefinition: true)
!142 = !DIGlobalVariableExpression(var: !143, expr: !DIExpression())
!143 = distinct !DIGlobalVariable(name: "global_data_two", scope: !2, file: !3, line: 134, type: !99, isLocal: false, isDefinition: true)
!144 = !DIGlobalVariableExpression(var: !145, expr: !DIExpression())
!145 = distinct !DIGlobalVariable(name: "global_data_device", scope: !2, file: !3, line: 135, type: !99, isLocal: false, isDefinition: true)
!146 = !DIGlobalVariableExpression(var: !147, expr: !DIExpression())
!147 = distinct !DIGlobalVariable(name: "global_data_two_device", scope: !2, file: !3, line: 136, type: !99, isLocal: false, isDefinition: true)
!148 = !DIGlobalVariableExpression(var: !149, expr: !DIExpression())
!149 = distinct !DIGlobalVariable(name: "global_data_reduce", scope: !2, file: !3, line: 137, type: !100, isLocal: false, isDefinition: true)
!150 = !DIGlobalVariableExpression(var: !151, expr: !DIExpression())
!151 = distinct !DIGlobalVariable(name: "global_data_two_reduce", scope: !2, file: !3, line: 138, type: !100, isLocal: false, isDefinition: true)
!152 = !DIGlobalVariableExpression(var: !153, expr: !DIExpression())
!153 = distinct !DIGlobalVariable(name: "global_data_elements", scope: !2, file: !3, line: 139, type: !154, isLocal: false, isDefinition: true)
!154 = !DIDerivedType(tag: DW_TAG_typedef, name: "size_t", file: !155, line: 46, baseType: !156)
!155 = !DIFile(filename: "llvm-install-tulip/lib/clang/9.0.0/include/stddef.h", directory: "/scratch/ah7226")
!156 = !DIBasicType(name: "long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!157 = !DIGlobalVariableExpression(var: !158, expr: !DIExpression())
!158 = distinct !DIGlobalVariable(name: "size_global_data", scope: !2, file: !3, line: 140, type: !154, isLocal: false, isDefinition: true)
!159 = !DIGlobalVariableExpression(var: !160, expr: !DIExpression())
!160 = distinct !DIGlobalVariable(name: "size_colidx_device", scope: !2, file: !3, line: 141, type: !154, isLocal: false, isDefinition: true)
!161 = !DIGlobalVariableExpression(var: !162, expr: !DIExpression())
!162 = distinct !DIGlobalVariable(name: "size_rowstr_device", scope: !2, file: !3, line: 142, type: !154, isLocal: false, isDefinition: true)
!163 = !DIGlobalVariableExpression(var: !164, expr: !DIExpression())
!164 = distinct !DIGlobalVariable(name: "size_iv_device", scope: !2, file: !3, line: 143, type: !154, isLocal: false, isDefinition: true)
!165 = !DIGlobalVariableExpression(var: !166, expr: !DIExpression())
!166 = distinct !DIGlobalVariable(name: "size_arow_device", scope: !2, file: !3, line: 144, type: !154, isLocal: false, isDefinition: true)
!167 = !DIGlobalVariableExpression(var: !168, expr: !DIExpression())
!168 = distinct !DIGlobalVariable(name: "size_acol_device", scope: !2, file: !3, line: 145, type: !154, isLocal: false, isDefinition: true)
!169 = !DIGlobalVariableExpression(var: !170, expr: !DIExpression())
!170 = distinct !DIGlobalVariable(name: "size_aelt_device", scope: !2, file: !3, line: 146, type: !154, isLocal: false, isDefinition: true)
!171 = !DIGlobalVariableExpression(var: !172, expr: !DIExpression())
!172 = distinct !DIGlobalVariable(name: "size_a_device", scope: !2, file: !3, line: 147, type: !154, isLocal: false, isDefinition: true)
!173 = !DIGlobalVariableExpression(var: !174, expr: !DIExpression())
!174 = distinct !DIGlobalVariable(name: "size_x_device", scope: !2, file: !3, line: 148, type: !154, isLocal: false, isDefinition: true)
!175 = !DIGlobalVariableExpression(var: !176, expr: !DIExpression())
!176 = distinct !DIGlobalVariable(name: "size_z_device", scope: !2, file: !3, line: 149, type: !154, isLocal: false, isDefinition: true)
!177 = !DIGlobalVariableExpression(var: !178, expr: !DIExpression())
!178 = distinct !DIGlobalVariable(name: "size_p_device", scope: !2, file: !3, line: 150, type: !154, isLocal: false, isDefinition: true)
!179 = !DIGlobalVariableExpression(var: !180, expr: !DIExpression())
!180 = distinct !DIGlobalVariable(name: "size_q_device", scope: !2, file: !3, line: 151, type: !154, isLocal: false, isDefinition: true)
!181 = !DIGlobalVariableExpression(var: !182, expr: !DIExpression())
!182 = distinct !DIGlobalVariable(name: "size_r_device", scope: !2, file: !3, line: 152, type: !154, isLocal: false, isDefinition: true)
!183 = !DIGlobalVariableExpression(var: !184, expr: !DIExpression())
!184 = distinct !DIGlobalVariable(name: "size_rho_device", scope: !2, file: !3, line: 153, type: !154, isLocal: false, isDefinition: true)
!185 = !DIGlobalVariableExpression(var: !186, expr: !DIExpression())
!186 = distinct !DIGlobalVariable(name: "size_d_device", scope: !2, file: !3, line: 154, type: !154, isLocal: false, isDefinition: true)
!187 = !DIGlobalVariableExpression(var: !188, expr: !DIExpression())
!188 = distinct !DIGlobalVariable(name: "size_alpha_device", scope: !2, file: !3, line: 155, type: !154, isLocal: false, isDefinition: true)
!189 = !DIGlobalVariableExpression(var: !190, expr: !DIExpression())
!190 = distinct !DIGlobalVariable(name: "size_beta_device", scope: !2, file: !3, line: 156, type: !154, isLocal: false, isDefinition: true)
!191 = !DIGlobalVariableExpression(var: !192, expr: !DIExpression())
!192 = distinct !DIGlobalVariable(name: "size_sum_device", scope: !2, file: !3, line: 157, type: !154, isLocal: false, isDefinition: true)
!193 = !DIGlobalVariableExpression(var: !194, expr: !DIExpression())
!194 = distinct !DIGlobalVariable(name: "size_norm_temp1_device", scope: !2, file: !3, line: 158, type: !154, isLocal: false, isDefinition: true)
!195 = !DIGlobalVariableExpression(var: !196, expr: !DIExpression())
!196 = distinct !DIGlobalVariable(name: "size_norm_temp2_device", scope: !2, file: !3, line: 159, type: !154, isLocal: false, isDefinition: true)
!197 = !DIGlobalVariableExpression(var: !198, expr: !DIExpression())
!198 = distinct !DIGlobalVariable(name: "blocks_per_grid_on_kernel_one", scope: !2, file: !3, line: 160, type: !97, isLocal: false, isDefinition: true)
!199 = !DIGlobalVariableExpression(var: !200, expr: !DIExpression())
!200 = distinct !DIGlobalVariable(name: "blocks_per_grid_on_kernel_two", scope: !2, file: !3, line: 161, type: !97, isLocal: false, isDefinition: true)
!201 = !DIGlobalVariableExpression(var: !202, expr: !DIExpression())
!202 = distinct !DIGlobalVariable(name: "blocks_per_grid_on_kernel_three", scope: !2, file: !3, line: 162, type: !97, isLocal: false, isDefinition: true)
!203 = !DIGlobalVariableExpression(var: !204, expr: !DIExpression())
!204 = distinct !DIGlobalVariable(name: "blocks_per_grid_on_kernel_four", scope: !2, file: !3, line: 163, type: !97, isLocal: false, isDefinition: true)
!205 = !DIGlobalVariableExpression(var: !206, expr: !DIExpression())
!206 = distinct !DIGlobalVariable(name: "blocks_per_grid_on_kernel_five", scope: !2, file: !3, line: 164, type: !97, isLocal: false, isDefinition: true)
!207 = !DIGlobalVariableExpression(var: !208, expr: !DIExpression())
!208 = distinct !DIGlobalVariable(name: "blocks_per_grid_on_kernel_six", scope: !2, file: !3, line: 165, type: !97, isLocal: false, isDefinition: true)
!209 = !DIGlobalVariableExpression(var: !210, expr: !DIExpression())
!210 = distinct !DIGlobalVariable(name: "blocks_per_grid_on_kernel_seven", scope: !2, file: !3, line: 166, type: !97, isLocal: false, isDefinition: true)
!211 = !DIGlobalVariableExpression(var: !212, expr: !DIExpression())
!212 = distinct !DIGlobalVariable(name: "blocks_per_grid_on_kernel_eight", scope: !2, file: !3, line: 167, type: !97, isLocal: false, isDefinition: true)
!213 = !DIGlobalVariableExpression(var: !214, expr: !DIExpression())
!214 = distinct !DIGlobalVariable(name: "blocks_per_grid_on_kernel_nine", scope: !2, file: !3, line: 168, type: !97, isLocal: false, isDefinition: true)
!215 = !DIGlobalVariableExpression(var: !216, expr: !DIExpression())
!216 = distinct !DIGlobalVariable(name: "blocks_per_grid_on_kernel_ten", scope: !2, file: !3, line: 169, type: !97, isLocal: false, isDefinition: true)
!217 = !DIGlobalVariableExpression(var: !218, expr: !DIExpression())
!218 = distinct !DIGlobalVariable(name: "blocks_per_grid_on_kernel_eleven", scope: !2, file: !3, line: 170, type: !97, isLocal: false, isDefinition: true)
!219 = !DIGlobalVariableExpression(var: !220, expr: !DIExpression())
!220 = distinct !DIGlobalVariable(name: "threads_per_block_on_kernel_one", scope: !2, file: !3, line: 171, type: !97, isLocal: false, isDefinition: true)
!221 = !DIGlobalVariableExpression(var: !222, expr: !DIExpression())
!222 = distinct !DIGlobalVariable(name: "threads_per_block_on_kernel_two", scope: !2, file: !3, line: 172, type: !97, isLocal: false, isDefinition: true)
!223 = !DIGlobalVariableExpression(var: !224, expr: !DIExpression())
!224 = distinct !DIGlobalVariable(name: "threads_per_block_on_kernel_three", scope: !2, file: !3, line: 173, type: !97, isLocal: false, isDefinition: true)
!225 = !DIGlobalVariableExpression(var: !226, expr: !DIExpression())
!226 = distinct !DIGlobalVariable(name: "threads_per_block_on_kernel_four", scope: !2, file: !3, line: 174, type: !97, isLocal: false, isDefinition: true)
!227 = !DIGlobalVariableExpression(var: !228, expr: !DIExpression())
!228 = distinct !DIGlobalVariable(name: "threads_per_block_on_kernel_five", scope: !2, file: !3, line: 175, type: !97, isLocal: false, isDefinition: true)
!229 = !DIGlobalVariableExpression(var: !230, expr: !DIExpression())
!230 = distinct !DIGlobalVariable(name: "threads_per_block_on_kernel_six", scope: !2, file: !3, line: 176, type: !97, isLocal: false, isDefinition: true)
!231 = !DIGlobalVariableExpression(var: !232, expr: !DIExpression())
!232 = distinct !DIGlobalVariable(name: "threads_per_block_on_kernel_seven", scope: !2, file: !3, line: 177, type: !97, isLocal: false, isDefinition: true)
!233 = !DIGlobalVariableExpression(var: !234, expr: !DIExpression())
!234 = distinct !DIGlobalVariable(name: "threads_per_block_on_kernel_eight", scope: !2, file: !3, line: 178, type: !97, isLocal: false, isDefinition: true)
!235 = !DIGlobalVariableExpression(var: !236, expr: !DIExpression())
!236 = distinct !DIGlobalVariable(name: "threads_per_block_on_kernel_nine", scope: !2, file: !3, line: 179, type: !97, isLocal: false, isDefinition: true)
!237 = !DIGlobalVariableExpression(var: !238, expr: !DIExpression())
!238 = distinct !DIGlobalVariable(name: "threads_per_block_on_kernel_ten", scope: !2, file: !3, line: 180, type: !97, isLocal: false, isDefinition: true)
!239 = !DIGlobalVariableExpression(var: !240, expr: !DIExpression())
!240 = distinct !DIGlobalVariable(name: "threads_per_block_on_kernel_eleven", scope: !2, file: !3, line: 181, type: !97, isLocal: false, isDefinition: true)
!241 = !DIGlobalVariableExpression(var: !242, expr: !DIExpression())
!242 = distinct !DIGlobalVariable(name: "size_shared_data_on_kernel_one", scope: !2, file: !3, line: 182, type: !154, isLocal: false, isDefinition: true)
!243 = !DIGlobalVariableExpression(var: !244, expr: !DIExpression())
!244 = distinct !DIGlobalVariable(name: "size_shared_data_on_kernel_two", scope: !2, file: !3, line: 183, type: !154, isLocal: false, isDefinition: true)
!245 = !DIGlobalVariableExpression(var: !246, expr: !DIExpression())
!246 = distinct !DIGlobalVariable(name: "size_shared_data_on_kernel_three", scope: !2, file: !3, line: 184, type: !154, isLocal: false, isDefinition: true)
!247 = !DIGlobalVariableExpression(var: !248, expr: !DIExpression())
!248 = distinct !DIGlobalVariable(name: "size_shared_data_on_kernel_four", scope: !2, file: !3, line: 185, type: !154, isLocal: false, isDefinition: true)
!249 = !DIGlobalVariableExpression(var: !250, expr: !DIExpression())
!250 = distinct !DIGlobalVariable(name: "size_shared_data_on_kernel_five", scope: !2, file: !3, line: 186, type: !154, isLocal: false, isDefinition: true)
!251 = !DIGlobalVariableExpression(var: !252, expr: !DIExpression())
!252 = distinct !DIGlobalVariable(name: "size_shared_data_on_kernel_six", scope: !2, file: !3, line: 187, type: !154, isLocal: false, isDefinition: true)
!253 = !DIGlobalVariableExpression(var: !254, expr: !DIExpression())
!254 = distinct !DIGlobalVariable(name: "size_shared_data_on_kernel_seven", scope: !2, file: !3, line: 188, type: !154, isLocal: false, isDefinition: true)
!255 = !DIGlobalVariableExpression(var: !256, expr: !DIExpression())
!256 = distinct !DIGlobalVariable(name: "size_shared_data_on_kernel_eight", scope: !2, file: !3, line: 189, type: !154, isLocal: false, isDefinition: true)
!257 = !DIGlobalVariableExpression(var: !258, expr: !DIExpression())
!258 = distinct !DIGlobalVariable(name: "size_shared_data_on_kernel_nine", scope: !2, file: !3, line: 190, type: !154, isLocal: false, isDefinition: true)
!259 = !DIGlobalVariableExpression(var: !260, expr: !DIExpression())
!260 = distinct !DIGlobalVariable(name: "size_shared_data_on_kernel_ten", scope: !2, file: !3, line: 191, type: !154, isLocal: false, isDefinition: true)
!261 = !DIGlobalVariableExpression(var: !262, expr: !DIExpression())
!262 = distinct !DIGlobalVariable(name: "size_shared_data_on_kernel_eleven", scope: !2, file: !3, line: 192, type: !154, isLocal: false, isDefinition: true)
!263 = !DIGlobalVariableExpression(var: !264, expr: !DIExpression())
!264 = distinct !DIGlobalVariable(name: "size_reduce_memory_on_kernel_one", scope: !2, file: !3, line: 193, type: !154, isLocal: false, isDefinition: true)
!265 = !DIGlobalVariableExpression(var: !266, expr: !DIExpression())
!266 = distinct !DIGlobalVariable(name: "size_reduce_memory_on_kernel_two", scope: !2, file: !3, line: 194, type: !154, isLocal: false, isDefinition: true)
!267 = !DIGlobalVariableExpression(var: !268, expr: !DIExpression())
!268 = distinct !DIGlobalVariable(name: "size_reduce_memory_on_kernel_three", scope: !2, file: !3, line: 195, type: !154, isLocal: false, isDefinition: true)
!269 = !DIGlobalVariableExpression(var: !270, expr: !DIExpression())
!270 = distinct !DIGlobalVariable(name: "size_reduce_memory_on_kernel_four", scope: !2, file: !3, line: 196, type: !154, isLocal: false, isDefinition: true)
!271 = !DIGlobalVariableExpression(var: !272, expr: !DIExpression())
!272 = distinct !DIGlobalVariable(name: "size_reduce_memory_on_kernel_five", scope: !2, file: !3, line: 197, type: !154, isLocal: false, isDefinition: true)
!273 = !DIGlobalVariableExpression(var: !274, expr: !DIExpression())
!274 = distinct !DIGlobalVariable(name: "size_reduce_memory_on_kernel_six", scope: !2, file: !3, line: 198, type: !154, isLocal: false, isDefinition: true)
!275 = !DIGlobalVariableExpression(var: !276, expr: !DIExpression())
!276 = distinct !DIGlobalVariable(name: "size_reduce_memory_on_kernel_seven", scope: !2, file: !3, line: 199, type: !154, isLocal: false, isDefinition: true)
!277 = !DIGlobalVariableExpression(var: !278, expr: !DIExpression())
!278 = distinct !DIGlobalVariable(name: "size_reduce_memory_on_kernel_eight", scope: !2, file: !3, line: 200, type: !154, isLocal: false, isDefinition: true)
!279 = !DIGlobalVariableExpression(var: !280, expr: !DIExpression())
!280 = distinct !DIGlobalVariable(name: "size_reduce_memory_on_kernel_nine", scope: !2, file: !3, line: 201, type: !154, isLocal: false, isDefinition: true)
!281 = !DIGlobalVariableExpression(var: !282, expr: !DIExpression())
!282 = distinct !DIGlobalVariable(name: "size_reduce_memory_on_kernel_ten", scope: !2, file: !3, line: 202, type: !154, isLocal: false, isDefinition: true)
!283 = !DIGlobalVariableExpression(var: !284, expr: !DIExpression())
!284 = distinct !DIGlobalVariable(name: "size_reduce_memory_on_kernel_eleven", scope: !2, file: !3, line: 203, type: !154, isLocal: false, isDefinition: true)
!285 = !DIGlobalVariableExpression(var: !286, expr: !DIExpression())
!286 = distinct !DIGlobalVariable(name: "gpu_device_id", scope: !2, file: !3, line: 204, type: !97, isLocal: false, isDefinition: true)
!287 = !DIGlobalVariableExpression(var: !288, expr: !DIExpression())
!288 = distinct !DIGlobalVariable(name: "total_devices", scope: !2, file: !3, line: 205, type: !97, isLocal: false, isDefinition: true)
!289 = !DIGlobalVariableExpression(var: !290, expr: !DIExpression())
!290 = distinct !DIGlobalVariable(name: "gpu_device_properties", scope: !2, file: !3, line: 206, type: !291, isLocal: false, isDefinition: true)
!291 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "cudaDeviceProp", file: !6, line: 1257, size: 5056, flags: DIFlagTypePassByValue, elements: !292, identifier: "_ZTS14cudaDeviceProp")
!292 = !{!293, !297, !298, !299, !300, !301, !302, !303, !307, !308, !309, !310, !311, !312, !313, !314, !315, !316, !317, !318, !319, !320, !321, !322, !323, !327, !328, !329, !330, !331, !332, !333, !334, !335, !336, !337, !338, !339, !340, !341, !342, !343, !344, !345, !346, !347, !348, !349, !350, !351, !352, !353, !354, !355, !356, !357, !358, !359, !360, !361, !362, !363}
!293 = !DIDerivedType(tag: DW_TAG_member, name: "name", scope: !291, file: !6, line: 1259, baseType: !294, size: 2048)
!294 = !DICompositeType(tag: DW_TAG_array_type, baseType: !109, size: 2048, elements: !295)
!295 = !{!296}
!296 = !DISubrange(count: 256)
!297 = !DIDerivedType(tag: DW_TAG_member, name: "totalGlobalMem", scope: !291, file: !6, line: 1260, baseType: !154, size: 64, offset: 2048)
!298 = !DIDerivedType(tag: DW_TAG_member, name: "sharedMemPerBlock", scope: !291, file: !6, line: 1261, baseType: !154, size: 64, offset: 2112)
!299 = !DIDerivedType(tag: DW_TAG_member, name: "regsPerBlock", scope: !291, file: !6, line: 1262, baseType: !97, size: 32, offset: 2176)
!300 = !DIDerivedType(tag: DW_TAG_member, name: "warpSize", scope: !291, file: !6, line: 1263, baseType: !97, size: 32, offset: 2208)
!301 = !DIDerivedType(tag: DW_TAG_member, name: "memPitch", scope: !291, file: !6, line: 1264, baseType: !154, size: 64, offset: 2240)
!302 = !DIDerivedType(tag: DW_TAG_member, name: "maxThreadsPerBlock", scope: !291, file: !6, line: 1265, baseType: !97, size: 32, offset: 2304)
!303 = !DIDerivedType(tag: DW_TAG_member, name: "maxThreadsDim", scope: !291, file: !6, line: 1266, baseType: !304, size: 96, offset: 2336)
!304 = !DICompositeType(tag: DW_TAG_array_type, baseType: !97, size: 96, elements: !305)
!305 = !{!306}
!306 = !DISubrange(count: 3)
!307 = !DIDerivedType(tag: DW_TAG_member, name: "maxGridSize", scope: !291, file: !6, line: 1267, baseType: !304, size: 96, offset: 2432)
!308 = !DIDerivedType(tag: DW_TAG_member, name: "clockRate", scope: !291, file: !6, line: 1268, baseType: !97, size: 32, offset: 2528)
!309 = !DIDerivedType(tag: DW_TAG_member, name: "totalConstMem", scope: !291, file: !6, line: 1269, baseType: !154, size: 64, offset: 2560)
!310 = !DIDerivedType(tag: DW_TAG_member, name: "major", scope: !291, file: !6, line: 1270, baseType: !97, size: 32, offset: 2624)
!311 = !DIDerivedType(tag: DW_TAG_member, name: "minor", scope: !291, file: !6, line: 1271, baseType: !97, size: 32, offset: 2656)
!312 = !DIDerivedType(tag: DW_TAG_member, name: "textureAlignment", scope: !291, file: !6, line: 1272, baseType: !154, size: 64, offset: 2688)
!313 = !DIDerivedType(tag: DW_TAG_member, name: "texturePitchAlignment", scope: !291, file: !6, line: 1273, baseType: !154, size: 64, offset: 2752)
!314 = !DIDerivedType(tag: DW_TAG_member, name: "deviceOverlap", scope: !291, file: !6, line: 1274, baseType: !97, size: 32, offset: 2816)
!315 = !DIDerivedType(tag: DW_TAG_member, name: "multiProcessorCount", scope: !291, file: !6, line: 1275, baseType: !97, size: 32, offset: 2848)
!316 = !DIDerivedType(tag: DW_TAG_member, name: "kernelExecTimeoutEnabled", scope: !291, file: !6, line: 1276, baseType: !97, size: 32, offset: 2880)
!317 = !DIDerivedType(tag: DW_TAG_member, name: "integrated", scope: !291, file: !6, line: 1277, baseType: !97, size: 32, offset: 2912)
!318 = !DIDerivedType(tag: DW_TAG_member, name: "canMapHostMemory", scope: !291, file: !6, line: 1278, baseType: !97, size: 32, offset: 2944)
!319 = !DIDerivedType(tag: DW_TAG_member, name: "computeMode", scope: !291, file: !6, line: 1279, baseType: !97, size: 32, offset: 2976)
!320 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture1D", scope: !291, file: !6, line: 1280, baseType: !97, size: 32, offset: 3008)
!321 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture1DMipmap", scope: !291, file: !6, line: 1281, baseType: !97, size: 32, offset: 3040)
!322 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture1DLinear", scope: !291, file: !6, line: 1282, baseType: !97, size: 32, offset: 3072)
!323 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture2D", scope: !291, file: !6, line: 1283, baseType: !324, size: 64, offset: 3104)
!324 = !DICompositeType(tag: DW_TAG_array_type, baseType: !97, size: 64, elements: !325)
!325 = !{!326}
!326 = !DISubrange(count: 2)
!327 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture2DMipmap", scope: !291, file: !6, line: 1284, baseType: !324, size: 64, offset: 3168)
!328 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture2DLinear", scope: !291, file: !6, line: 1285, baseType: !304, size: 96, offset: 3232)
!329 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture2DGather", scope: !291, file: !6, line: 1286, baseType: !324, size: 64, offset: 3328)
!330 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture3D", scope: !291, file: !6, line: 1287, baseType: !304, size: 96, offset: 3392)
!331 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture3DAlt", scope: !291, file: !6, line: 1288, baseType: !304, size: 96, offset: 3488)
!332 = !DIDerivedType(tag: DW_TAG_member, name: "maxTextureCubemap", scope: !291, file: !6, line: 1289, baseType: !97, size: 32, offset: 3584)
!333 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture1DLayered", scope: !291, file: !6, line: 1290, baseType: !324, size: 64, offset: 3616)
!334 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture2DLayered", scope: !291, file: !6, line: 1291, baseType: !304, size: 96, offset: 3680)
!335 = !DIDerivedType(tag: DW_TAG_member, name: "maxTextureCubemapLayered", scope: !291, file: !6, line: 1292, baseType: !324, size: 64, offset: 3776)
!336 = !DIDerivedType(tag: DW_TAG_member, name: "maxSurface1D", scope: !291, file: !6, line: 1293, baseType: !97, size: 32, offset: 3840)
!337 = !DIDerivedType(tag: DW_TAG_member, name: "maxSurface2D", scope: !291, file: !6, line: 1294, baseType: !324, size: 64, offset: 3872)
!338 = !DIDerivedType(tag: DW_TAG_member, name: "maxSurface3D", scope: !291, file: !6, line: 1295, baseType: !304, size: 96, offset: 3936)
!339 = !DIDerivedType(tag: DW_TAG_member, name: "maxSurface1DLayered", scope: !291, file: !6, line: 1296, baseType: !324, size: 64, offset: 4032)
!340 = !DIDerivedType(tag: DW_TAG_member, name: "maxSurface2DLayered", scope: !291, file: !6, line: 1297, baseType: !304, size: 96, offset: 4096)
!341 = !DIDerivedType(tag: DW_TAG_member, name: "maxSurfaceCubemap", scope: !291, file: !6, line: 1298, baseType: !97, size: 32, offset: 4192)
!342 = !DIDerivedType(tag: DW_TAG_member, name: "maxSurfaceCubemapLayered", scope: !291, file: !6, line: 1299, baseType: !324, size: 64, offset: 4224)
!343 = !DIDerivedType(tag: DW_TAG_member, name: "surfaceAlignment", scope: !291, file: !6, line: 1300, baseType: !154, size: 64, offset: 4288)
!344 = !DIDerivedType(tag: DW_TAG_member, name: "concurrentKernels", scope: !291, file: !6, line: 1301, baseType: !97, size: 32, offset: 4352)
!345 = !DIDerivedType(tag: DW_TAG_member, name: "ECCEnabled", scope: !291, file: !6, line: 1302, baseType: !97, size: 32, offset: 4384)
!346 = !DIDerivedType(tag: DW_TAG_member, name: "pciBusID", scope: !291, file: !6, line: 1303, baseType: !97, size: 32, offset: 4416)
!347 = !DIDerivedType(tag: DW_TAG_member, name: "pciDeviceID", scope: !291, file: !6, line: 1304, baseType: !97, size: 32, offset: 4448)
!348 = !DIDerivedType(tag: DW_TAG_member, name: "pciDomainID", scope: !291, file: !6, line: 1305, baseType: !97, size: 32, offset: 4480)
!349 = !DIDerivedType(tag: DW_TAG_member, name: "tccDriver", scope: !291, file: !6, line: 1306, baseType: !97, size: 32, offset: 4512)
!350 = !DIDerivedType(tag: DW_TAG_member, name: "asyncEngineCount", scope: !291, file: !6, line: 1307, baseType: !97, size: 32, offset: 4544)
!351 = !DIDerivedType(tag: DW_TAG_member, name: "unifiedAddressing", scope: !291, file: !6, line: 1308, baseType: !97, size: 32, offset: 4576)
!352 = !DIDerivedType(tag: DW_TAG_member, name: "memoryClockRate", scope: !291, file: !6, line: 1309, baseType: !97, size: 32, offset: 4608)
!353 = !DIDerivedType(tag: DW_TAG_member, name: "memoryBusWidth", scope: !291, file: !6, line: 1310, baseType: !97, size: 32, offset: 4640)
!354 = !DIDerivedType(tag: DW_TAG_member, name: "l2CacheSize", scope: !291, file: !6, line: 1311, baseType: !97, size: 32, offset: 4672)
!355 = !DIDerivedType(tag: DW_TAG_member, name: "maxThreadsPerMultiProcessor", scope: !291, file: !6, line: 1312, baseType: !97, size: 32, offset: 4704)
!356 = !DIDerivedType(tag: DW_TAG_member, name: "streamPrioritiesSupported", scope: !291, file: !6, line: 1313, baseType: !97, size: 32, offset: 4736)
!357 = !DIDerivedType(tag: DW_TAG_member, name: "globalL1CacheSupported", scope: !291, file: !6, line: 1314, baseType: !97, size: 32, offset: 4768)
!358 = !DIDerivedType(tag: DW_TAG_member, name: "localL1CacheSupported", scope: !291, file: !6, line: 1315, baseType: !97, size: 32, offset: 4800)
!359 = !DIDerivedType(tag: DW_TAG_member, name: "sharedMemPerMultiprocessor", scope: !291, file: !6, line: 1316, baseType: !154, size: 64, offset: 4864)
!360 = !DIDerivedType(tag: DW_TAG_member, name: "regsPerMultiprocessor", scope: !291, file: !6, line: 1317, baseType: !97, size: 32, offset: 4928)
!361 = !DIDerivedType(tag: DW_TAG_member, name: "managedMemory", scope: !291, file: !6, line: 1318, baseType: !97, size: 32, offset: 4960)
!362 = !DIDerivedType(tag: DW_TAG_member, name: "isMultiGpuBoard", scope: !291, file: !6, line: 1319, baseType: !97, size: 32, offset: 4992)
!363 = !DIDerivedType(tag: DW_TAG_member, name: "multiGpuBoardGroupID", scope: !291, file: !6, line: 1320, baseType: !97, size: 32, offset: 5024)
!364 = !DIGlobalVariableExpression(var: !365, expr: !DIExpression())
!365 = distinct !DIGlobalVariable(name: "colidx", linkageName: "_ZL6colidx", scope: !2, file: !3, line: 97, type: !98, isLocal: true, isDefinition: true)
!366 = !DIGlobalVariableExpression(var: !367, expr: !DIExpression())
!367 = distinct !DIGlobalVariable(name: "rowstr", linkageName: "_ZL6rowstr", scope: !2, file: !3, line: 98, type: !98, isLocal: true, isDefinition: true)
!368 = !DIGlobalVariableExpression(var: !369, expr: !DIExpression())
!369 = distinct !DIGlobalVariable(name: "iv", linkageName: "_ZL2iv", scope: !2, file: !3, line: 99, type: !98, isLocal: true, isDefinition: true)
!370 = !DIGlobalVariableExpression(var: !371, expr: !DIExpression())
!371 = distinct !DIGlobalVariable(name: "arow", linkageName: "_ZL4arow", scope: !2, file: !3, line: 100, type: !98, isLocal: true, isDefinition: true)
!372 = !DIGlobalVariableExpression(var: !373, expr: !DIExpression())
!373 = distinct !DIGlobalVariable(name: "acol", linkageName: "_ZL4acol", scope: !2, file: !3, line: 101, type: !98, isLocal: true, isDefinition: true)
!374 = !DIGlobalVariableExpression(var: !375, expr: !DIExpression())
!375 = distinct !DIGlobalVariable(name: "aelt", linkageName: "_ZL4aelt", scope: !2, file: !3, line: 102, type: !99, isLocal: true, isDefinition: true)
!376 = !DIGlobalVariableExpression(var: !377, expr: !DIExpression())
!377 = distinct !DIGlobalVariable(name: "a", linkageName: "_ZL1a", scope: !2, file: !3, line: 103, type: !99, isLocal: true, isDefinition: true)
!378 = !DIGlobalVariableExpression(var: !379, expr: !DIExpression())
!379 = distinct !DIGlobalVariable(name: "x", linkageName: "_ZL1x", scope: !2, file: !3, line: 104, type: !99, isLocal: true, isDefinition: true)
!380 = !DIGlobalVariableExpression(var: !381, expr: !DIExpression())
!381 = distinct !DIGlobalVariable(name: "z", linkageName: "_ZL1z", scope: !2, file: !3, line: 105, type: !99, isLocal: true, isDefinition: true)
!382 = !DIGlobalVariableExpression(var: !383, expr: !DIExpression())
!383 = distinct !DIGlobalVariable(name: "p", linkageName: "_ZL1p", scope: !2, file: !3, line: 106, type: !99, isLocal: true, isDefinition: true)
!384 = !DIGlobalVariableExpression(var: !385, expr: !DIExpression())
!385 = distinct !DIGlobalVariable(name: "q", linkageName: "_ZL1q", scope: !2, file: !3, line: 107, type: !99, isLocal: true, isDefinition: true)
!386 = !DIGlobalVariableExpression(var: !387, expr: !DIExpression())
!387 = distinct !DIGlobalVariable(name: "r", linkageName: "_ZL1r", scope: !2, file: !3, line: 108, type: !99, isLocal: true, isDefinition: true)
!388 = !DIGlobalVariableExpression(var: !389, expr: !DIExpression())
!389 = distinct !DIGlobalVariable(name: "firstrow", linkageName: "_ZL8firstrow", scope: !2, file: !3, line: 111, type: !97, isLocal: true, isDefinition: true)
!390 = !DIGlobalVariableExpression(var: !391, expr: !DIExpression())
!391 = distinct !DIGlobalVariable(name: "lastrow", linkageName: "_ZL7lastrow", scope: !2, file: !3, line: 112, type: !97, isLocal: true, isDefinition: true)
!392 = !DIGlobalVariableExpression(var: !393, expr: !DIExpression())
!393 = distinct !DIGlobalVariable(name: "firstcol", linkageName: "_ZL8firstcol", scope: !2, file: !3, line: 113, type: !97, isLocal: true, isDefinition: true)
!394 = !DIGlobalVariableExpression(var: !395, expr: !DIExpression())
!395 = distinct !DIGlobalVariable(name: "lastcol", linkageName: "_ZL7lastcol", scope: !2, file: !3, line: 114, type: !97, isLocal: true, isDefinition: true)
!396 = !DIGlobalVariableExpression(var: !397, expr: !DIExpression())
!397 = distinct !DIGlobalVariable(name: "naa", linkageName: "_ZL3naa", scope: !2, file: !3, line: 109, type: !97, isLocal: true, isDefinition: true)
!398 = !DIGlobalVariableExpression(var: !399, expr: !DIExpression())
!399 = distinct !DIGlobalVariable(name: "nzz", linkageName: "_ZL3nzz", scope: !2, file: !3, line: 110, type: !97, isLocal: true, isDefinition: true)
!400 = !DIGlobalVariableExpression(var: !401, expr: !DIExpression())
!401 = distinct !DIGlobalVariable(name: "tran", linkageName: "_ZL4tran", scope: !2, file: !3, line: 116, type: !100, isLocal: true, isDefinition: true)
!402 = !DIGlobalVariableExpression(var: !403, expr: !DIExpression())
!403 = distinct !DIGlobalVariable(name: "amult", linkageName: "_ZL5amult", scope: !2, file: !3, line: 115, type: !100, isLocal: true, isDefinition: true)
!404 = !{!405, !411, !416, !418, !420, !422, !424, !428, !430, !432, !434, !436, !438, !440, !442, !444, !446, !448, !450, !452, !454, !456, !460, !462, !464, !466, !470, !474, !476, !478, !483, !487, !489, !491, !493, !495, !497, !499, !501, !503, !508, !512, !514, !519, !523, !525, !527, !529, !531, !533, !537, !539, !541, !546, !552, !556, !558, !560, !562, !564, !568, !570, !572, !576, !578, !580, !582, !584, !586, !588, !590, !592, !594, !598, !604, !606, !608, !612, !614, !616, !618, !620, !622, !624, !626, !630, !634, !636, !638, !642, !644, !646, !648, !650, !652, !654, !658, !664, !668, !673, !675, !679, !683, !693, !697, !701, !705, !709, !713, !715, !719, !723, !727, !735, !739, !743, !747, !751, !755, !761, !765, !769, !771, !779, !783, !790, !792, !794, !798, !802, !806, !811, !815, !820, !821, !822, !823, !825, !826, !827, !828, !829, !830, !831, !833, !834, !835, !836, !837, !841, !842, !843, !844, !845, !846, !847, !848, !849, !850, !851, !852, !853, !854, !855, !856, !857, !858, !859, !860, !861, !862, !863, !864, !865, !869, !871, !873, !875, !877, !879, !881, !883, !886, !888, !890, !892, !894, !896, !898, !900, !902, !904, !906, !908, !910, !912, !914, !916, !918, !920, !922, !924, !926, !928, !930, !932, !934, !936, !938, !940, !942, !944, !946, !948, !950, !952, !954, !956, !958, !960, !962, !964, !966, !968, !970, !972, !974, !976, !978, !984, !990, !995, !999, !1001, !1003, !1005, !1007, !1014, !1018, !1022, !1026, !1030, !1034, !1039, !1043, !1045, !1049, !1055, !1059, !1064, !1066, !1068, !1072, !1076, !1080, !1082, !1084, !1086, !1088, !1092, !1094, !1096, !1100, !1104, !1108, !1112, !1116, !1118, !1120, !1126, !1130, !1134, !1138, !1140, !1142, !1146, !1150, !1151, !1152, !1153, !1154}
!405 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !407, file: !408, line: 223)
!406 = !DINamespace(name: "std", scope: null)
!407 = !DISubprogram(name: "abs", linkageName: "_ZL3absi", scope: !408, file: !408, line: 53, type: !409, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!408 = !DIFile(filename: "llvm-install-tulip/lib/clang/9.0.0/include/__clang_cuda_math_forward_declares.h", directory: "/scratch/ah7226")
!409 = !DISubroutineType(types: !410)
!410 = !{!97, !97}
!411 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !412, file: !408, line: 224)
!412 = !DISubprogram(name: "acos", linkageName: "_ZL4acosf", scope: !408, file: !408, line: 55, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!413 = !DISubroutineType(types: !414)
!414 = !{!415, !415}
!415 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!416 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !417, file: !408, line: 225)
!417 = !DISubprogram(name: "acosh", linkageName: "_ZL5acoshf", scope: !408, file: !408, line: 57, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!418 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !419, file: !408, line: 226)
!419 = !DISubprogram(name: "asin", linkageName: "_ZL4asinf", scope: !408, file: !408, line: 59, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!420 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !421, file: !408, line: 227)
!421 = !DISubprogram(name: "asinh", linkageName: "_ZL5asinhf", scope: !408, file: !408, line: 61, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!422 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !423, file: !408, line: 228)
!423 = !DISubprogram(name: "atan", linkageName: "_ZL4atanf", scope: !408, file: !408, line: 65, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!424 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !425, file: !408, line: 229)
!425 = !DISubprogram(name: "atan2", linkageName: "_ZL5atan2ff", scope: !408, file: !408, line: 63, type: !426, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!426 = !DISubroutineType(types: !427)
!427 = !{!415, !415, !415}
!428 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !429, file: !408, line: 230)
!429 = !DISubprogram(name: "atanh", linkageName: "_ZL5atanhf", scope: !408, file: !408, line: 67, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!430 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !431, file: !408, line: 231)
!431 = !DISubprogram(name: "cbrt", linkageName: "_ZL4cbrtf", scope: !408, file: !408, line: 69, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!432 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !433, file: !408, line: 232)
!433 = !DISubprogram(name: "ceil", linkageName: "_ZL4ceilf", scope: !408, file: !408, line: 71, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!434 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !435, file: !408, line: 233)
!435 = !DISubprogram(name: "copysign", linkageName: "_ZL8copysignff", scope: !408, file: !408, line: 73, type: !426, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!436 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !437, file: !408, line: 234)
!437 = !DISubprogram(name: "cos", linkageName: "_ZL3cosf", scope: !408, file: !408, line: 75, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!438 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !439, file: !408, line: 235)
!439 = !DISubprogram(name: "cosh", linkageName: "_ZL4coshf", scope: !408, file: !408, line: 77, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!440 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !441, file: !408, line: 236)
!441 = !DISubprogram(name: "erf", linkageName: "_ZL3erff", scope: !408, file: !408, line: 81, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!442 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !443, file: !408, line: 237)
!443 = !DISubprogram(name: "erfc", linkageName: "_ZL4erfcf", scope: !408, file: !408, line: 79, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!444 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !445, file: !408, line: 238)
!445 = !DISubprogram(name: "exp", linkageName: "_ZL3expf", scope: !408, file: !408, line: 85, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!446 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !447, file: !408, line: 239)
!447 = !DISubprogram(name: "exp2", linkageName: "_ZL4exp2f", scope: !408, file: !408, line: 83, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!448 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !449, file: !408, line: 240)
!449 = !DISubprogram(name: "expm1", linkageName: "_ZL5expm1f", scope: !408, file: !408, line: 87, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!450 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !451, file: !408, line: 241)
!451 = !DISubprogram(name: "fabs", linkageName: "_ZL4fabsf", scope: !408, file: !408, line: 89, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!452 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !453, file: !408, line: 242)
!453 = !DISubprogram(name: "fdim", linkageName: "_ZL4fdimff", scope: !408, file: !408, line: 91, type: !426, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!454 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !455, file: !408, line: 243)
!455 = !DISubprogram(name: "floor", linkageName: "_ZL5floorf", scope: !408, file: !408, line: 93, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!456 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !457, file: !408, line: 244)
!457 = !DISubprogram(name: "fma", linkageName: "_ZL3fmafff", scope: !408, file: !408, line: 95, type: !458, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!458 = !DISubroutineType(types: !459)
!459 = !{!415, !415, !415, !415}
!460 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !461, file: !408, line: 245)
!461 = !DISubprogram(name: "fmax", linkageName: "_ZL4fmaxff", scope: !408, file: !408, line: 97, type: !426, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!462 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !463, file: !408, line: 246)
!463 = !DISubprogram(name: "fmin", linkageName: "_ZL4fminff", scope: !408, file: !408, line: 99, type: !426, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!464 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !465, file: !408, line: 247)
!465 = !DISubprogram(name: "fmod", linkageName: "_ZL4fmodff", scope: !408, file: !408, line: 101, type: !426, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!466 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !467, file: !408, line: 248)
!467 = !DISubprogram(name: "fpclassify", linkageName: "_ZL10fpclassifyf", scope: !408, file: !408, line: 103, type: !468, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!468 = !DISubroutineType(types: !469)
!469 = !{!97, !415}
!470 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !471, file: !408, line: 249)
!471 = !DISubprogram(name: "frexp", linkageName: "_ZL5frexpfPi", scope: !408, file: !408, line: 105, type: !472, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!472 = !DISubroutineType(types: !473)
!473 = !{!415, !415, !98}
!474 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !475, file: !408, line: 250)
!475 = !DISubprogram(name: "hypot", linkageName: "_ZL5hypotff", scope: !408, file: !408, line: 107, type: !426, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!476 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !477, file: !408, line: 251)
!477 = !DISubprogram(name: "ilogb", linkageName: "_ZL5ilogbf", scope: !408, file: !408, line: 109, type: !468, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!478 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !479, file: !408, line: 252)
!479 = !DISubprogram(name: "isfinite", linkageName: "_ZL8isfinitef", scope: !408, file: !408, line: 114, type: !480, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!480 = !DISubroutineType(types: !481)
!481 = !{!482, !415}
!482 = !DIBasicType(name: "bool", size: 8, encoding: DW_ATE_boolean)
!483 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !484, file: !408, line: 253)
!484 = !DISubprogram(name: "isgreater", linkageName: "_ZL9isgreaterff", scope: !408, file: !408, line: 118, type: !485, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!485 = !DISubroutineType(types: !486)
!486 = !{!482, !415, !415}
!487 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !488, file: !408, line: 254)
!488 = !DISubprogram(name: "isgreaterequal", linkageName: "_ZL14isgreaterequalff", scope: !408, file: !408, line: 117, type: !485, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!489 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !490, file: !408, line: 255)
!490 = !DISubprogram(name: "isinf", linkageName: "_ZL5isinff", scope: !408, file: !408, line: 123, type: !480, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!491 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !492, file: !408, line: 256)
!492 = !DISubprogram(name: "isless", linkageName: "_ZL6islessff", scope: !408, file: !408, line: 127, type: !485, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!493 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !494, file: !408, line: 257)
!494 = !DISubprogram(name: "islessequal", linkageName: "_ZL11islessequalff", scope: !408, file: !408, line: 126, type: !485, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!495 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !496, file: !408, line: 258)
!496 = !DISubprogram(name: "islessgreater", linkageName: "_ZL13islessgreaterff", scope: !408, file: !408, line: 129, type: !485, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!497 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !498, file: !408, line: 259)
!498 = !DISubprogram(name: "isnan", linkageName: "_ZL5isnanf", scope: !408, file: !408, line: 134, type: !480, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!499 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !500, file: !408, line: 260)
!500 = !DISubprogram(name: "isnormal", linkageName: "_ZL8isnormalf", scope: !408, file: !408, line: 136, type: !480, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!501 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !502, file: !408, line: 261)
!502 = !DISubprogram(name: "isunordered", linkageName: "_ZL11isunorderedff", scope: !408, file: !408, line: 138, type: !485, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!503 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !504, file: !408, line: 262)
!504 = !DISubprogram(name: "labs", linkageName: "_ZL4labsl", scope: !408, file: !408, line: 139, type: !505, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!505 = !DISubroutineType(types: !506)
!506 = !{!507, !507}
!507 = !DIBasicType(name: "long int", size: 64, encoding: DW_ATE_signed)
!508 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !509, file: !408, line: 263)
!509 = !DISubprogram(name: "ldexp", linkageName: "_ZL5ldexpfi", scope: !408, file: !408, line: 141, type: !510, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!510 = !DISubroutineType(types: !511)
!511 = !{!415, !415, !97}
!512 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !513, file: !408, line: 264)
!513 = !DISubprogram(name: "lgamma", linkageName: "_ZL6lgammaf", scope: !408, file: !408, line: 143, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!514 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !515, file: !408, line: 265)
!515 = !DISubprogram(name: "llabs", linkageName: "_ZL5llabsx", scope: !408, file: !408, line: 144, type: !516, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!516 = !DISubroutineType(types: !517)
!517 = !{!518, !518}
!518 = !DIBasicType(name: "long long int", size: 64, encoding: DW_ATE_signed)
!519 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !520, file: !408, line: 266)
!520 = !DISubprogram(name: "llrint", linkageName: "_ZL6llrintf", scope: !408, file: !408, line: 146, type: !521, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!521 = !DISubroutineType(types: !522)
!522 = !{!518, !415}
!523 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !524, file: !408, line: 267)
!524 = !DISubprogram(name: "log", linkageName: "_ZL3logf", scope: !408, file: !408, line: 159, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!525 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !526, file: !408, line: 268)
!526 = !DISubprogram(name: "log10", linkageName: "_ZL5log10f", scope: !408, file: !408, line: 148, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!527 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !528, file: !408, line: 269)
!528 = !DISubprogram(name: "log1p", linkageName: "_ZL5log1pf", scope: !408, file: !408, line: 150, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!529 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !530, file: !408, line: 270)
!530 = !DISubprogram(name: "log2", linkageName: "_ZL4log2f", scope: !408, file: !408, line: 152, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!531 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !532, file: !408, line: 271)
!532 = !DISubprogram(name: "logb", linkageName: "_ZL4logbf", scope: !408, file: !408, line: 154, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!533 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !534, file: !408, line: 272)
!534 = !DISubprogram(name: "lrint", linkageName: "_ZL5lrintf", scope: !408, file: !408, line: 161, type: !535, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!535 = !DISubroutineType(types: !536)
!536 = !{!507, !415}
!537 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !538, file: !408, line: 273)
!538 = !DISubprogram(name: "lround", linkageName: "_ZL6lroundf", scope: !408, file: !408, line: 163, type: !535, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!539 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !540, file: !408, line: 274)
!540 = !DISubprogram(name: "llround", linkageName: "_ZL7llroundf", scope: !408, file: !408, line: 164, type: !521, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!541 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !542, file: !408, line: 275)
!542 = !DISubprogram(name: "modf", linkageName: "_ZL4modffPf", scope: !408, file: !408, line: 166, type: !543, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!543 = !DISubroutineType(types: !544)
!544 = !{!415, !415, !545}
!545 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !415, size: 64)
!546 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !547, file: !408, line: 276)
!547 = !DISubprogram(name: "nan", linkageName: "_ZL3nanPKc", scope: !408, file: !408, line: 167, type: !548, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!548 = !DISubroutineType(types: !549)
!549 = !{!100, !550}
!550 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !551, size: 64)
!551 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !109)
!552 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !553, file: !408, line: 277)
!553 = !DISubprogram(name: "nanf", linkageName: "_ZL4nanfPKc", scope: !408, file: !408, line: 168, type: !554, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!554 = !DISubroutineType(types: !555)
!555 = !{!415, !550}
!556 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !557, file: !408, line: 278)
!557 = !DISubprogram(name: "nearbyint", linkageName: "_ZL9nearbyintf", scope: !408, file: !408, line: 170, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!558 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !559, file: !408, line: 279)
!559 = !DISubprogram(name: "nextafter", linkageName: "_ZL9nextafterff", scope: !408, file: !408, line: 172, type: !426, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!560 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !561, file: !408, line: 280)
!561 = !DISubprogram(name: "pow", linkageName: "_ZL3powfi", scope: !408, file: !408, line: 176, type: !510, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!562 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !563, file: !408, line: 281)
!563 = !DISubprogram(name: "remainder", linkageName: "_ZL9remainderff", scope: !408, file: !408, line: 178, type: !426, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!564 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !565, file: !408, line: 282)
!565 = !DISubprogram(name: "remquo", linkageName: "_ZL6remquoffPi", scope: !408, file: !408, line: 180, type: !566, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!566 = !DISubroutineType(types: !567)
!567 = !{!415, !415, !415, !98}
!568 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !569, file: !408, line: 283)
!569 = !DISubprogram(name: "rint", linkageName: "_ZL4rintf", scope: !408, file: !408, line: 182, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!570 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !571, file: !408, line: 284)
!571 = !DISubprogram(name: "round", linkageName: "_ZL5roundf", scope: !408, file: !408, line: 184, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!572 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !573, file: !408, line: 285)
!573 = !DISubprogram(name: "scalbln", linkageName: "_ZL7scalblnfl", scope: !408, file: !408, line: 186, type: !574, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!574 = !DISubroutineType(types: !575)
!575 = !{!415, !415, !507}
!576 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !577, file: !408, line: 286)
!577 = !DISubprogram(name: "scalbn", linkageName: "_ZL6scalbnfi", scope: !408, file: !408, line: 188, type: !510, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!578 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !579, file: !408, line: 287)
!579 = !DISubprogram(name: "signbit", linkageName: "_ZL7signbitf", scope: !408, file: !408, line: 190, type: !480, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!580 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !581, file: !408, line: 288)
!581 = !DISubprogram(name: "sin", linkageName: "_ZL3sinf", scope: !408, file: !408, line: 192, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!582 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !583, file: !408, line: 289)
!583 = !DISubprogram(name: "sinh", linkageName: "_ZL4sinhf", scope: !408, file: !408, line: 194, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!584 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !585, file: !408, line: 290)
!585 = !DISubprogram(name: "sqrt", linkageName: "_ZL4sqrtf", scope: !408, file: !408, line: 196, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!586 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !587, file: !408, line: 291)
!587 = !DISubprogram(name: "tan", linkageName: "_ZL3tanf", scope: !408, file: !408, line: 198, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!588 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !589, file: !408, line: 292)
!589 = !DISubprogram(name: "tanh", linkageName: "_ZL4tanhf", scope: !408, file: !408, line: 200, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!590 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !591, file: !408, line: 293)
!591 = !DISubprogram(name: "tgamma", linkageName: "_ZL6tgammaf", scope: !408, file: !408, line: 202, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!592 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !593, file: !408, line: 294)
!593 = !DISubprogram(name: "trunc", linkageName: "_ZL5truncf", scope: !408, file: !408, line: 204, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!594 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !595, file: !597, line: 52)
!595 = !DISubprogram(name: "abs", scope: !596, file: !596, line: 848, type: !409, flags: DIFlagPrototyped, spFlags: 0)
!596 = !DIFile(filename: "/usr/include/stdlib.h", directory: "")
!597 = !DIFile(filename: "/usr/lib/gcc/x86_64-linux-gnu/11/../../../../include/c++/11/bits/std_abs.h", directory: "")
!598 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !599, file: !603, line: 83)
!599 = !DISubprogram(name: "acos", scope: !600, file: !600, line: 53, type: !601, flags: DIFlagPrototyped, spFlags: 0)
!600 = !DIFile(filename: "/usr/include/x86_64-linux-gnu/bits/mathcalls.h", directory: "")
!601 = !DISubroutineType(types: !602)
!602 = !{!100, !100}
!603 = !DIFile(filename: "/usr/lib/gcc/x86_64-linux-gnu/11/../../../../include/c++/11/cmath", directory: "")
!604 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !605, file: !603, line: 102)
!605 = !DISubprogram(name: "asin", scope: !600, file: !600, line: 55, type: !601, flags: DIFlagPrototyped, spFlags: 0)
!606 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !607, file: !603, line: 121)
!607 = !DISubprogram(name: "atan", scope: !600, file: !600, line: 57, type: !601, flags: DIFlagPrototyped, spFlags: 0)
!608 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !609, file: !603, line: 140)
!609 = !DISubprogram(name: "atan2", scope: !600, file: !600, line: 59, type: !610, flags: DIFlagPrototyped, spFlags: 0)
!610 = !DISubroutineType(types: !611)
!611 = !{!100, !100, !100}
!612 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !613, file: !603, line: 161)
!613 = !DISubprogram(name: "ceil", scope: !600, file: !600, line: 159, type: !601, flags: DIFlagPrototyped, spFlags: 0)
!614 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !615, file: !603, line: 180)
!615 = !DISubprogram(name: "cos", scope: !600, file: !600, line: 62, type: !601, flags: DIFlagPrototyped, spFlags: 0)
!616 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !617, file: !603, line: 199)
!617 = !DISubprogram(name: "cosh", scope: !600, file: !600, line: 71, type: !601, flags: DIFlagPrototyped, spFlags: 0)
!618 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !619, file: !603, line: 218)
!619 = !DISubprogram(name: "exp", scope: !600, file: !600, line: 95, type: !601, flags: DIFlagPrototyped, spFlags: 0)
!620 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !621, file: !603, line: 237)
!621 = !DISubprogram(name: "fabs", scope: !600, file: !600, line: 162, type: !601, flags: DIFlagPrototyped, spFlags: 0)
!622 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !623, file: !603, line: 256)
!623 = !DISubprogram(name: "floor", scope: !600, file: !600, line: 165, type: !601, flags: DIFlagPrototyped, spFlags: 0)
!624 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !625, file: !603, line: 275)
!625 = !DISubprogram(name: "fmod", scope: !600, file: !600, line: 168, type: !610, flags: DIFlagPrototyped, spFlags: 0)
!626 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !627, file: !603, line: 296)
!627 = !DISubprogram(name: "frexp", scope: !600, file: !600, line: 98, type: !628, flags: DIFlagPrototyped, spFlags: 0)
!628 = !DISubroutineType(types: !629)
!629 = !{!100, !100, !98}
!630 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !631, file: !603, line: 315)
!631 = !DISubprogram(name: "ldexp", scope: !600, file: !600, line: 101, type: !632, flags: DIFlagPrototyped, spFlags: 0)
!632 = !DISubroutineType(types: !633)
!633 = !{!100, !100, !97}
!634 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !635, file: !603, line: 334)
!635 = !DISubprogram(name: "log", scope: !600, file: !600, line: 104, type: !601, flags: DIFlagPrototyped, spFlags: 0)
!636 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !637, file: !603, line: 353)
!637 = !DISubprogram(name: "log10", scope: !600, file: !600, line: 107, type: !601, flags: DIFlagPrototyped, spFlags: 0)
!638 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !639, file: !603, line: 372)
!639 = !DISubprogram(name: "modf", scope: !600, file: !600, line: 110, type: !640, flags: DIFlagPrototyped, spFlags: 0)
!640 = !DISubroutineType(types: !641)
!641 = !{!100, !100, !99}
!642 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !643, file: !603, line: 384)
!643 = !DISubprogram(name: "pow", scope: !600, file: !600, line: 140, type: !610, flags: DIFlagPrototyped, spFlags: 0)
!644 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !645, file: !603, line: 421)
!645 = !DISubprogram(name: "sin", scope: !600, file: !600, line: 64, type: !601, flags: DIFlagPrototyped, spFlags: 0)
!646 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !647, file: !603, line: 440)
!647 = !DISubprogram(name: "sinh", scope: !600, file: !600, line: 73, type: !601, flags: DIFlagPrototyped, spFlags: 0)
!648 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !649, file: !603, line: 459)
!649 = !DISubprogram(name: "sqrt", scope: !600, file: !600, line: 143, type: !601, flags: DIFlagPrototyped, spFlags: 0)
!650 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !651, file: !603, line: 478)
!651 = !DISubprogram(name: "tan", scope: !600, file: !600, line: 66, type: !601, flags: DIFlagPrototyped, spFlags: 0)
!652 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !653, file: !603, line: 497)
!653 = !DISubprogram(name: "tanh", scope: !600, file: !600, line: 75, type: !601, flags: DIFlagPrototyped, spFlags: 0)
!654 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !655, file: !657, line: 127)
!655 = !DIDerivedType(tag: DW_TAG_typedef, name: "div_t", file: !596, line: 63, baseType: !656)
!656 = !DICompositeType(tag: DW_TAG_structure_type, file: !596, line: 59, flags: DIFlagFwdDecl, identifier: "_ZTS5div_t")
!657 = !DIFile(filename: "/usr/lib/gcc/x86_64-linux-gnu/11/../../../../include/c++/11/cstdlib", directory: "")
!658 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !659, file: !657, line: 128)
!659 = !DIDerivedType(tag: DW_TAG_typedef, name: "ldiv_t", file: !596, line: 71, baseType: !660)
!660 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !596, line: 67, size: 128, flags: DIFlagTypePassByValue, elements: !661, identifier: "_ZTS6ldiv_t")
!661 = !{!662, !663}
!662 = !DIDerivedType(tag: DW_TAG_member, name: "quot", scope: !660, file: !596, line: 69, baseType: !507, size: 64)
!663 = !DIDerivedType(tag: DW_TAG_member, name: "rem", scope: !660, file: !596, line: 70, baseType: !507, size: 64, offset: 64)
!664 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !665, file: !657, line: 130)
!665 = !DISubprogram(name: "abort", scope: !596, file: !596, line: 598, type: !666, flags: DIFlagPrototyped | DIFlagNoReturn, spFlags: 0)
!666 = !DISubroutineType(types: !667)
!667 = !{null}
!668 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !669, file: !657, line: 134)
!669 = !DISubprogram(name: "atexit", scope: !596, file: !596, line: 602, type: !670, flags: DIFlagPrototyped, spFlags: 0)
!670 = !DISubroutineType(types: !671)
!671 = !{!97, !672}
!672 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !666, size: 64)
!673 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !674, file: !657, line: 140)
!674 = !DISubprogram(name: "atof", scope: !596, file: !596, line: 102, type: !548, flags: DIFlagPrototyped, spFlags: 0)
!675 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !676, file: !657, line: 141)
!676 = !DISubprogram(name: "atoi", scope: !596, file: !596, line: 105, type: !677, flags: DIFlagPrototyped, spFlags: 0)
!677 = !DISubroutineType(types: !678)
!678 = !{!97, !550}
!679 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !680, file: !657, line: 142)
!680 = !DISubprogram(name: "atol", scope: !596, file: !596, line: 108, type: !681, flags: DIFlagPrototyped, spFlags: 0)
!681 = !DISubroutineType(types: !682)
!682 = !{!507, !550}
!683 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !684, file: !657, line: 143)
!684 = !DISubprogram(name: "bsearch", scope: !596, file: !596, line: 828, type: !685, flags: DIFlagPrototyped, spFlags: 0)
!685 = !DISubroutineType(types: !686)
!686 = !{!105, !687, !687, !154, !154, !689}
!687 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !688, size: 64)
!688 = !DIDerivedType(tag: DW_TAG_const_type, baseType: null)
!689 = !DIDerivedType(tag: DW_TAG_typedef, name: "__compar_fn_t", file: !596, line: 816, baseType: !690)
!690 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !691, size: 64)
!691 = !DISubroutineType(types: !692)
!692 = !{!97, !687, !687}
!693 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !694, file: !657, line: 144)
!694 = !DISubprogram(name: "calloc", scope: !596, file: !596, line: 543, type: !695, flags: DIFlagPrototyped, spFlags: 0)
!695 = !DISubroutineType(types: !696)
!696 = !{!105, !154, !154}
!697 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !698, file: !657, line: 145)
!698 = !DISubprogram(name: "div", scope: !596, file: !596, line: 860, type: !699, flags: DIFlagPrototyped, spFlags: 0)
!699 = !DISubroutineType(types: !700)
!700 = !{!655, !97, !97}
!701 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !702, file: !657, line: 146)
!702 = !DISubprogram(name: "exit", scope: !596, file: !596, line: 624, type: !703, flags: DIFlagPrototyped | DIFlagNoReturn, spFlags: 0)
!703 = !DISubroutineType(types: !704)
!704 = !{null, !97}
!705 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !706, file: !657, line: 147)
!706 = !DISubprogram(name: "free", scope: !596, file: !596, line: 555, type: !707, flags: DIFlagPrototyped, spFlags: 0)
!707 = !DISubroutineType(types: !708)
!708 = !{null, !105}
!709 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !710, file: !657, line: 148)
!710 = !DISubprogram(name: "getenv", scope: !596, file: !596, line: 641, type: !711, flags: DIFlagPrototyped, spFlags: 0)
!711 = !DISubroutineType(types: !712)
!712 = !{!108, !550}
!713 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !714, file: !657, line: 149)
!714 = !DISubprogram(name: "labs", scope: !596, file: !596, line: 849, type: !505, flags: DIFlagPrototyped, spFlags: 0)
!715 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !716, file: !657, line: 150)
!716 = !DISubprogram(name: "ldiv", scope: !596, file: !596, line: 862, type: !717, flags: DIFlagPrototyped, spFlags: 0)
!717 = !DISubroutineType(types: !718)
!718 = !{!659, !507, !507}
!719 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !720, file: !657, line: 151)
!720 = !DISubprogram(name: "malloc", scope: !596, file: !596, line: 540, type: !721, flags: DIFlagPrototyped, spFlags: 0)
!721 = !DISubroutineType(types: !722)
!722 = !{!105, !154}
!723 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !724, file: !657, line: 153)
!724 = !DISubprogram(name: "mblen", scope: !596, file: !596, line: 930, type: !725, flags: DIFlagPrototyped, spFlags: 0)
!725 = !DISubroutineType(types: !726)
!726 = !{!97, !550, !154}
!727 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !728, file: !657, line: 154)
!728 = !DISubprogram(name: "mbstowcs", scope: !596, file: !596, line: 941, type: !729, flags: DIFlagPrototyped, spFlags: 0)
!729 = !DISubroutineType(types: !730)
!730 = !{!154, !731, !734, !154}
!731 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !732)
!732 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !733, size: 64)
!733 = !DIBasicType(name: "wchar_t", size: 32, encoding: DW_ATE_signed)
!734 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !550)
!735 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !736, file: !657, line: 155)
!736 = !DISubprogram(name: "mbtowc", scope: !596, file: !596, line: 933, type: !737, flags: DIFlagPrototyped, spFlags: 0)
!737 = !DISubroutineType(types: !738)
!738 = !{!97, !731, !734, !154}
!739 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !740, file: !657, line: 157)
!740 = !DISubprogram(name: "qsort", scope: !596, file: !596, line: 838, type: !741, flags: DIFlagPrototyped, spFlags: 0)
!741 = !DISubroutineType(types: !742)
!742 = !{null, !105, !154, !154, !689}
!743 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !744, file: !657, line: 163)
!744 = !DISubprogram(name: "rand", scope: !596, file: !596, line: 454, type: !745, flags: DIFlagPrototyped, spFlags: 0)
!745 = !DISubroutineType(types: !746)
!746 = !{!97}
!747 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !748, file: !657, line: 164)
!748 = !DISubprogram(name: "realloc", scope: !596, file: !596, line: 551, type: !749, flags: DIFlagPrototyped, spFlags: 0)
!749 = !DISubroutineType(types: !750)
!750 = !{!105, !105, !154}
!751 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !752, file: !657, line: 165)
!752 = !DISubprogram(name: "srand", scope: !596, file: !596, line: 456, type: !753, flags: DIFlagPrototyped, spFlags: 0)
!753 = !DISubroutineType(types: !754)
!754 = !{null, !7}
!755 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !756, file: !657, line: 166)
!756 = !DISubprogram(name: "strtod", scope: !596, file: !596, line: 118, type: !757, flags: DIFlagPrototyped, spFlags: 0)
!757 = !DISubroutineType(types: !758)
!758 = !{!100, !734, !759}
!759 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !760)
!760 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !108, size: 64)
!761 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !762, file: !657, line: 167)
!762 = !DISubprogram(name: "strtol", scope: !596, file: !596, line: 177, type: !763, flags: DIFlagPrototyped, spFlags: 0)
!763 = !DISubroutineType(types: !764)
!764 = !{!507, !734, !759, !97}
!765 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !766, file: !657, line: 168)
!766 = !DISubprogram(name: "strtoul", scope: !596, file: !596, line: 181, type: !767, flags: DIFlagPrototyped, spFlags: 0)
!767 = !DISubroutineType(types: !768)
!768 = !{!156, !734, !759, !97}
!769 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !770, file: !657, line: 169)
!770 = !DISubprogram(name: "system", scope: !596, file: !596, line: 791, type: !677, flags: DIFlagPrototyped, spFlags: 0)
!771 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !772, file: !657, line: 171)
!772 = !DISubprogram(name: "wcstombs", scope: !596, file: !596, line: 945, type: !773, flags: DIFlagPrototyped, spFlags: 0)
!773 = !DISubroutineType(types: !774)
!774 = !{!154, !775, !776, !154}
!775 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !108)
!776 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !777)
!777 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !778, size: 64)
!778 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !733)
!779 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !780, file: !657, line: 172)
!780 = !DISubprogram(name: "wctomb", scope: !596, file: !596, line: 937, type: !781, flags: DIFlagPrototyped, spFlags: 0)
!781 = !DISubroutineType(types: !782)
!782 = !{!97, !108, !733}
!783 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !784, entity: !785, file: !657, line: 200)
!784 = !DINamespace(name: "__gnu_cxx", scope: null)
!785 = !DIDerivedType(tag: DW_TAG_typedef, name: "lldiv_t", file: !596, line: 81, baseType: !786)
!786 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !596, line: 77, size: 128, flags: DIFlagTypePassByValue, elements: !787, identifier: "_ZTS7lldiv_t")
!787 = !{!788, !789}
!788 = !DIDerivedType(tag: DW_TAG_member, name: "quot", scope: !786, file: !596, line: 79, baseType: !518, size: 64)
!789 = !DIDerivedType(tag: DW_TAG_member, name: "rem", scope: !786, file: !596, line: 80, baseType: !518, size: 64, offset: 64)
!790 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !784, entity: !791, file: !657, line: 206)
!791 = !DISubprogram(name: "_Exit", scope: !596, file: !596, line: 636, type: !703, flags: DIFlagPrototyped | DIFlagNoReturn, spFlags: 0)
!792 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !784, entity: !793, file: !657, line: 210)
!793 = !DISubprogram(name: "llabs", scope: !596, file: !596, line: 852, type: !516, flags: DIFlagPrototyped, spFlags: 0)
!794 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !784, entity: !795, file: !657, line: 216)
!795 = !DISubprogram(name: "lldiv", scope: !596, file: !596, line: 866, type: !796, flags: DIFlagPrototyped, spFlags: 0)
!796 = !DISubroutineType(types: !797)
!797 = !{!785, !518, !518}
!798 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !784, entity: !799, file: !657, line: 227)
!799 = !DISubprogram(name: "atoll", scope: !596, file: !596, line: 113, type: !800, flags: DIFlagPrototyped, spFlags: 0)
!800 = !DISubroutineType(types: !801)
!801 = !{!518, !550}
!802 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !784, entity: !803, file: !657, line: 228)
!803 = !DISubprogram(name: "strtoll", scope: !596, file: !596, line: 201, type: !804, flags: DIFlagPrototyped, spFlags: 0)
!804 = !DISubroutineType(types: !805)
!805 = !{!518, !734, !759, !97}
!806 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !784, entity: !807, file: !657, line: 229)
!807 = !DISubprogram(name: "strtoull", scope: !596, file: !596, line: 206, type: !808, flags: DIFlagPrototyped, spFlags: 0)
!808 = !DISubroutineType(types: !809)
!809 = !{!810, !734, !759, !97}
!810 = !DIBasicType(name: "long long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!811 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !784, entity: !812, file: !657, line: 231)
!812 = !DISubprogram(name: "strtof", scope: !596, file: !596, line: 124, type: !813, flags: DIFlagPrototyped, spFlags: 0)
!813 = !DISubroutineType(types: !814)
!814 = !{!415, !734, !759}
!815 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !784, entity: !816, file: !657, line: 232)
!816 = !DISubprogram(name: "strtold", scope: !596, file: !596, line: 127, type: !817, flags: DIFlagPrototyped, spFlags: 0)
!817 = !DISubroutineType(types: !818)
!818 = !{!819, !734, !759}
!819 = !DIBasicType(name: "long double", size: 128, encoding: DW_ATE_float)
!820 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !785, file: !657, line: 240)
!821 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !791, file: !657, line: 242)
!822 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !793, file: !657, line: 244)
!823 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !824, file: !657, line: 245)
!824 = !DISubprogram(name: "div", linkageName: "_ZN9__gnu_cxx3divExx", scope: !784, file: !657, line: 213, type: !796, flags: DIFlagPrototyped, spFlags: 0)
!825 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !795, file: !657, line: 246)
!826 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !799, file: !657, line: 248)
!827 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !812, file: !657, line: 249)
!828 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !803, file: !657, line: 250)
!829 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !807, file: !657, line: 251)
!830 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !816, file: !657, line: 252)
!831 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !665, file: !832, line: 38)
!832 = !DIFile(filename: "/usr/lib/gcc/x86_64-linux-gnu/11/../../../../include/c++/11/stdlib.h", directory: "")
!833 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !669, file: !832, line: 39)
!834 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !702, file: !832, line: 40)
!835 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !655, file: !832, line: 51)
!836 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !659, file: !832, line: 52)
!837 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !838, file: !832, line: 54)
!838 = !DISubprogram(name: "abs", linkageName: "_ZSt3abse", scope: !406, file: !597, line: 79, type: !839, flags: DIFlagPrototyped, spFlags: 0)
!839 = !DISubroutineType(types: !840)
!840 = !{!819, !819}
!841 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !674, file: !832, line: 55)
!842 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !676, file: !832, line: 56)
!843 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !680, file: !832, line: 57)
!844 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !684, file: !832, line: 58)
!845 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !694, file: !832, line: 59)
!846 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !824, file: !832, line: 60)
!847 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !706, file: !832, line: 61)
!848 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !710, file: !832, line: 62)
!849 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !714, file: !832, line: 63)
!850 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !716, file: !832, line: 64)
!851 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !720, file: !832, line: 65)
!852 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !724, file: !832, line: 67)
!853 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !728, file: !832, line: 68)
!854 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !736, file: !832, line: 69)
!855 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !740, file: !832, line: 71)
!856 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !744, file: !832, line: 72)
!857 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !748, file: !832, line: 73)
!858 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !752, file: !832, line: 74)
!859 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !756, file: !832, line: 75)
!860 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !762, file: !832, line: 76)
!861 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !766, file: !832, line: 77)
!862 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !770, file: !832, line: 78)
!863 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !772, file: !832, line: 80)
!864 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !780, file: !832, line: 81)
!865 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !866, file: !868, line: 414)
!866 = !DISubprogram(name: "acosf", linkageName: "_ZL5acosff", scope: !867, file: !867, line: 1126, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!867 = !DIFile(filename: "/usr/local/cuda/include/math_functions.hpp", directory: "")
!868 = !DIFile(filename: "llvm-install-tulip/lib/clang/9.0.0/include/__clang_cuda_cmath.h", directory: "/scratch/ah7226")
!869 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !870, file: !868, line: 415)
!870 = !DISubprogram(name: "acoshf", linkageName: "_ZL6acoshff", scope: !867, file: !867, line: 1154, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!871 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !872, file: !868, line: 416)
!872 = !DISubprogram(name: "asinf", linkageName: "_ZL5asinff", scope: !867, file: !867, line: 1121, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!873 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !874, file: !868, line: 417)
!874 = !DISubprogram(name: "asinhf", linkageName: "_ZL6asinhff", scope: !867, file: !867, line: 1159, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!875 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !876, file: !868, line: 418)
!876 = !DISubprogram(name: "atan2f", linkageName: "_ZL6atan2fff", scope: !867, file: !867, line: 1111, type: !426, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!877 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !878, file: !868, line: 419)
!878 = !DISubprogram(name: "atanf", linkageName: "_ZL5atanff", scope: !867, file: !867, line: 1116, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!879 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !880, file: !868, line: 420)
!880 = !DISubprogram(name: "atanhf", linkageName: "_ZL6atanhff", scope: !867, file: !867, line: 1164, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!881 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !882, file: !868, line: 421)
!882 = !DISubprogram(name: "cbrtf", linkageName: "_ZL5cbrtff", scope: !867, file: !867, line: 1199, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!883 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !884, file: !868, line: 422)
!884 = !DISubprogram(name: "ceilf", linkageName: "_ZL5ceilff", scope: !885, file: !885, line: 647, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!885 = !DIFile(filename: "/usr/local/cuda/include/device_functions.hpp", directory: "")
!886 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !887, file: !868, line: 423)
!887 = !DISubprogram(name: "copysignf", linkageName: "_ZL9copysignfff", scope: !867, file: !867, line: 973, type: !426, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!888 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !889, file: !868, line: 424)
!889 = !DISubprogram(name: "cosf", linkageName: "_ZL4cosff", scope: !867, file: !867, line: 1027, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!890 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !891, file: !868, line: 425)
!891 = !DISubprogram(name: "coshf", linkageName: "_ZL5coshff", scope: !867, file: !867, line: 1096, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!892 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !893, file: !868, line: 426)
!893 = !DISubprogram(name: "erfcf", linkageName: "_ZL5erfcff", scope: !867, file: !867, line: 1259, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!894 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !895, file: !868, line: 427)
!895 = !DISubprogram(name: "erff", linkageName: "_ZL4erfff", scope: !867, file: !867, line: 1249, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!896 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !897, file: !868, line: 428)
!897 = !DISubprogram(name: "exp2f", linkageName: "_ZL5exp2ff", scope: !885, file: !885, line: 637, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!898 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !899, file: !868, line: 429)
!899 = !DISubprogram(name: "expf", linkageName: "_ZL4expff", scope: !867, file: !867, line: 1078, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!900 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !901, file: !868, line: 430)
!901 = !DISubprogram(name: "expm1f", linkageName: "_ZL6expm1ff", scope: !867, file: !867, line: 1169, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!902 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !903, file: !868, line: 431)
!903 = !DISubprogram(name: "fabsf", linkageName: "_ZL5fabsff", scope: !885, file: !885, line: 582, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!904 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !905, file: !868, line: 432)
!905 = !DISubprogram(name: "fdimf", linkageName: "_ZL5fdimfff", scope: !867, file: !867, line: 1385, type: !426, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!906 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !907, file: !868, line: 433)
!907 = !DISubprogram(name: "floorf", linkageName: "_ZL6floorff", scope: !885, file: !885, line: 572, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!908 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !909, file: !868, line: 434)
!909 = !DISubprogram(name: "fmaf", linkageName: "_ZL4fmaffff", scope: !867, file: !867, line: 1337, type: !458, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!910 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !911, file: !868, line: 435)
!911 = !DISubprogram(name: "fmaxf", linkageName: "_ZL5fmaxfff", scope: !885, file: !885, line: 602, type: !426, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!912 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !913, file: !868, line: 436)
!913 = !DISubprogram(name: "fminf", linkageName: "_ZL5fminfff", scope: !885, file: !885, line: 597, type: !426, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!914 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !915, file: !868, line: 437)
!915 = !DISubprogram(name: "fmodf", linkageName: "_ZL5fmodfff", scope: !867, file: !867, line: 1322, type: !426, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!916 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !917, file: !868, line: 438)
!917 = !DISubprogram(name: "frexpf", linkageName: "_ZL6frexpffPi", scope: !867, file: !867, line: 1312, type: !472, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!918 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !919, file: !868, line: 439)
!919 = !DISubprogram(name: "hypotf", linkageName: "_ZL6hypotfff", scope: !867, file: !867, line: 1174, type: !426, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!920 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !921, file: !868, line: 440)
!921 = !DISubprogram(name: "ilogbf", linkageName: "_ZL6ilogbff", scope: !867, file: !867, line: 1390, type: !468, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!922 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !923, file: !868, line: 441)
!923 = !DISubprogram(name: "ldexpf", linkageName: "_ZL6ldexpffi", scope: !867, file: !867, line: 1289, type: !510, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!924 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !925, file: !868, line: 442)
!925 = !DISubprogram(name: "lgammaf", linkageName: "_ZL7lgammaff", scope: !867, file: !867, line: 1284, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!926 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !927, file: !868, line: 443)
!927 = !DISubprogram(name: "llrintf", linkageName: "_ZL7llrintff", scope: !867, file: !867, line: 933, type: !521, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!928 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !929, file: !868, line: 444)
!929 = !DISubprogram(name: "llroundf", linkageName: "_ZL8llroundff", scope: !867, file: !867, line: 1371, type: !521, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!930 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !931, file: !868, line: 445)
!931 = !DISubprogram(name: "log10f", linkageName: "_ZL6log10ff", scope: !867, file: !867, line: 1140, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!932 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !933, file: !868, line: 446)
!933 = !DISubprogram(name: "log1pf", linkageName: "_ZL6log1pff", scope: !867, file: !867, line: 1149, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!934 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !935, file: !868, line: 447)
!935 = !DISubprogram(name: "log2f", linkageName: "_ZL5log2ff", scope: !867, file: !867, line: 1069, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!936 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !937, file: !868, line: 448)
!937 = !DISubprogram(name: "logbf", linkageName: "_ZL5logbff", scope: !867, file: !867, line: 1395, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!938 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !939, file: !868, line: 449)
!939 = !DISubprogram(name: "logf", linkageName: "_ZL4logff", scope: !867, file: !867, line: 1131, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!940 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !941, file: !868, line: 450)
!941 = !DISubprogram(name: "lrintf", linkageName: "_ZL6lrintff", scope: !867, file: !867, line: 924, type: !535, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!942 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !943, file: !868, line: 451)
!943 = !DISubprogram(name: "lroundf", linkageName: "_ZL7lroundff", scope: !867, file: !867, line: 1376, type: !535, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!944 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !945, file: !868, line: 452)
!945 = !DISubprogram(name: "modff", linkageName: "_ZL5modfffPf", scope: !867, file: !867, line: 1317, type: !543, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!946 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !947, file: !868, line: 453)
!947 = !DISubprogram(name: "nearbyintf", linkageName: "_ZL10nearbyintff", scope: !867, file: !867, line: 938, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!948 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !949, file: !868, line: 454)
!949 = !DISubprogram(name: "nextafterf", linkageName: "_ZL10nextafterfff", scope: !867, file: !867, line: 1002, type: !426, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!950 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !951, file: !868, line: 455)
!951 = !DISubprogram(name: "powf", linkageName: "_ZL4powfff", scope: !867, file: !867, line: 1352, type: !426, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!952 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !953, file: !868, line: 456)
!953 = !DISubprogram(name: "remainderf", linkageName: "_ZL10remainderfff", scope: !867, file: !867, line: 1327, type: !426, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!954 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !955, file: !868, line: 457)
!955 = !DISubprogram(name: "remquof", linkageName: "_ZL7remquofffPi", scope: !867, file: !867, line: 1332, type: !566, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!956 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !957, file: !868, line: 458)
!957 = !DISubprogram(name: "rintf", linkageName: "_ZL5rintff", scope: !867, file: !867, line: 919, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!958 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !959, file: !868, line: 459)
!959 = !DISubprogram(name: "roundf", linkageName: "_ZL6roundff", scope: !867, file: !867, line: 1366, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!960 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !961, file: !868, line: 462)
!961 = !DISubprogram(name: "scalblnf", linkageName: "_ZL8scalblnffl", scope: !867, file: !867, line: 1299, type: !574, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!962 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !963, file: !868, line: 464)
!963 = !DISubprogram(name: "scalbnf", linkageName: "_ZL7scalbnffi", scope: !867, file: !867, line: 1294, type: !510, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!964 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !965, file: !868, line: 465)
!965 = !DISubprogram(name: "sinf", linkageName: "_ZL4sinff", scope: !867, file: !867, line: 1018, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!966 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !967, file: !868, line: 466)
!967 = !DISubprogram(name: "sinhf", linkageName: "_ZL5sinhff", scope: !867, file: !867, line: 1101, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!968 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !969, file: !868, line: 467)
!969 = !DISubprogram(name: "sqrtf", linkageName: "_ZL5sqrtff", scope: !885, file: !885, line: 887, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!970 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !971, file: !868, line: 468)
!971 = !DISubprogram(name: "tanf", linkageName: "_ZL4tanff", scope: !867, file: !867, line: 1060, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!972 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !973, file: !868, line: 469)
!973 = !DISubprogram(name: "tanhf", linkageName: "_ZL5tanhff", scope: !867, file: !867, line: 1106, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!974 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !975, file: !868, line: 470)
!975 = !DISubprogram(name: "tgammaf", linkageName: "_ZL7tgammaff", scope: !867, file: !867, line: 1361, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!976 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !977, file: !868, line: 471)
!977 = !DISubprogram(name: "truncf", linkageName: "_ZL6truncff", scope: !885, file: !885, line: 642, type: !413, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!978 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !979, file: !983, line: 98)
!979 = !DIDerivedType(tag: DW_TAG_typedef, name: "FILE", file: !980, line: 7, baseType: !981)
!980 = !DIFile(filename: "/usr/include/x86_64-linux-gnu/bits/types/FILE.h", directory: "")
!981 = !DICompositeType(tag: DW_TAG_structure_type, name: "_IO_FILE", file: !982, line: 49, flags: DIFlagFwdDecl, identifier: "_ZTS8_IO_FILE")
!982 = !DIFile(filename: "/usr/include/x86_64-linux-gnu/bits/types/struct_FILE.h", directory: "")
!983 = !DIFile(filename: "/usr/lib/gcc/x86_64-linux-gnu/11/../../../../include/c++/11/cstdio", directory: "")
!984 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !985, file: !983, line: 99)
!985 = !DIDerivedType(tag: DW_TAG_typedef, name: "fpos_t", file: !986, line: 84, baseType: !987)
!986 = !DIFile(filename: "/usr/include/stdio.h", directory: "")
!987 = !DIDerivedType(tag: DW_TAG_typedef, name: "__fpos_t", file: !988, line: 14, baseType: !989)
!988 = !DIFile(filename: "/usr/include/x86_64-linux-gnu/bits/types/__fpos_t.h", directory: "")
!989 = !DICompositeType(tag: DW_TAG_structure_type, name: "_G_fpos_t", file: !988, line: 10, flags: DIFlagFwdDecl, identifier: "_ZTS9_G_fpos_t")
!990 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !991, file: !983, line: 101)
!991 = !DISubprogram(name: "clearerr", scope: !986, file: !986, line: 786, type: !992, flags: DIFlagPrototyped, spFlags: 0)
!992 = !DISubroutineType(types: !993)
!993 = !{null, !994}
!994 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !979, size: 64)
!995 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !996, file: !983, line: 102)
!996 = !DISubprogram(name: "fclose", scope: !986, file: !986, line: 178, type: !997, flags: DIFlagPrototyped, spFlags: 0)
!997 = !DISubroutineType(types: !998)
!998 = !{!97, !994}
!999 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !1000, file: !983, line: 103)
!1000 = !DISubprogram(name: "feof", scope: !986, file: !986, line: 788, type: !997, flags: DIFlagPrototyped, spFlags: 0)
!1001 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !1002, file: !983, line: 104)
!1002 = !DISubprogram(name: "ferror", scope: !986, file: !986, line: 790, type: !997, flags: DIFlagPrototyped, spFlags: 0)
!1003 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !1004, file: !983, line: 105)
!1004 = !DISubprogram(name: "fflush", scope: !986, file: !986, line: 230, type: !997, flags: DIFlagPrototyped, spFlags: 0)
!1005 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !1006, file: !983, line: 106)
!1006 = !DISubprogram(name: "fgetc", scope: !986, file: !986, line: 513, type: !997, flags: DIFlagPrototyped, spFlags: 0)
!1007 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !1008, file: !983, line: 107)
!1008 = !DISubprogram(name: "fgetpos", scope: !986, file: !986, line: 760, type: !1009, flags: DIFlagPrototyped, spFlags: 0)
!1009 = !DISubroutineType(types: !1010)
!1010 = !{!97, !1011, !1012}
!1011 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !994)
!1012 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !1013)
!1013 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !985, size: 64)
!1014 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !1015, file: !983, line: 108)
!1015 = !DISubprogram(name: "fgets", scope: !986, file: !986, line: 592, type: !1016, flags: DIFlagPrototyped, spFlags: 0)
!1016 = !DISubroutineType(types: !1017)
!1017 = !{!108, !775, !97, !1011}
!1018 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !1019, file: !983, line: 109)
!1019 = !DISubprogram(name: "fopen", scope: !986, file: !986, line: 258, type: !1020, flags: DIFlagPrototyped, spFlags: 0)
!1020 = !DISubroutineType(types: !1021)
!1021 = !{!994, !734, !734}
!1022 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !1023, file: !983, line: 110)
!1023 = !DISubprogram(name: "fprintf", scope: !986, file: !986, line: 350, type: !1024, flags: DIFlagPrototyped, spFlags: 0)
!1024 = !DISubroutineType(types: !1025)
!1025 = !{!97, !1011, !734, null}
!1026 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !1027, file: !983, line: 111)
!1027 = !DISubprogram(name: "fputc", scope: !986, file: !986, line: 549, type: !1028, flags: DIFlagPrototyped, spFlags: 0)
!1028 = !DISubroutineType(types: !1029)
!1029 = !{!97, !97, !994}
!1030 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !1031, file: !983, line: 112)
!1031 = !DISubprogram(name: "fputs", scope: !986, file: !986, line: 655, type: !1032, flags: DIFlagPrototyped, spFlags: 0)
!1032 = !DISubroutineType(types: !1033)
!1033 = !{!97, !734, !1011}
!1034 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !1035, file: !983, line: 113)
!1035 = !DISubprogram(name: "fread", scope: !986, file: !986, line: 675, type: !1036, flags: DIFlagPrototyped, spFlags: 0)
!1036 = !DISubroutineType(types: !1037)
!1037 = !{!154, !1038, !154, !154, !1011}
!1038 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !105)
!1039 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !1040, file: !983, line: 114)
!1040 = !DISubprogram(name: "freopen", scope: !986, file: !986, line: 265, type: !1041, flags: DIFlagPrototyped, spFlags: 0)
!1041 = !DISubroutineType(types: !1042)
!1042 = !{!994, !734, !734, !1011}
!1043 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !1044, file: !983, line: 115)
!1044 = !DISubprogram(name: "fscanf", scope: !986, file: !986, line: 415, type: !1024, flags: DIFlagPrototyped, spFlags: 0)
!1045 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !1046, file: !983, line: 116)
!1046 = !DISubprogram(name: "fseek", scope: !986, file: !986, line: 713, type: !1047, flags: DIFlagPrototyped, spFlags: 0)
!1047 = !DISubroutineType(types: !1048)
!1048 = !{!97, !994, !507, !97}
!1049 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !1050, file: !983, line: 117)
!1050 = !DISubprogram(name: "fsetpos", scope: !986, file: !986, line: 765, type: !1051, flags: DIFlagPrototyped, spFlags: 0)
!1051 = !DISubroutineType(types: !1052)
!1052 = !{!97, !994, !1053}
!1053 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !1054, size: 64)
!1054 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !985)
!1055 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !1056, file: !983, line: 118)
!1056 = !DISubprogram(name: "ftell", scope: !986, file: !986, line: 718, type: !1057, flags: DIFlagPrototyped, spFlags: 0)
!1057 = !DISubroutineType(types: !1058)
!1058 = !{!507, !994}
!1059 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !1060, file: !983, line: 119)
!1060 = !DISubprogram(name: "fwrite", scope: !986, file: !986, line: 681, type: !1061, flags: DIFlagPrototyped, spFlags: 0)
!1061 = !DISubroutineType(types: !1062)
!1062 = !{!154, !1063, !154, !154, !1011}
!1063 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !687)
!1064 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !1065, file: !983, line: 120)
!1065 = !DISubprogram(name: "getc", scope: !986, file: !986, line: 514, type: !997, flags: DIFlagPrototyped, spFlags: 0)
!1066 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !1067, file: !983, line: 121)
!1067 = !DISubprogram(name: "getchar", scope: !986, file: !986, line: 520, type: !745, flags: DIFlagPrototyped, spFlags: 0)
!1068 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !1069, file: !983, line: 124)
!1069 = !DISubprogram(name: "gets", scope: !986, file: !986, line: 605, type: !1070, flags: DIFlagPrototyped, spFlags: 0)
!1070 = !DISubroutineType(types: !1071)
!1071 = !{!108, !108}
!1072 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !1073, file: !983, line: 126)
!1073 = !DISubprogram(name: "perror", scope: !986, file: !986, line: 804, type: !1074, flags: DIFlagPrototyped, spFlags: 0)
!1074 = !DISubroutineType(types: !1075)
!1075 = !{null, !550}
!1076 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !1077, file: !983, line: 127)
!1077 = !DISubprogram(name: "printf", scope: !986, file: !986, line: 356, type: !1078, flags: DIFlagPrototyped, spFlags: 0)
!1078 = !DISubroutineType(types: !1079)
!1079 = !{!97, !734, null}
!1080 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !1081, file: !983, line: 128)
!1081 = !DISubprogram(name: "putc", scope: !986, file: !986, line: 550, type: !1028, flags: DIFlagPrototyped, spFlags: 0)
!1082 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !1083, file: !983, line: 129)
!1083 = !DISubprogram(name: "putchar", scope: !986, file: !986, line: 556, type: !409, flags: DIFlagPrototyped, spFlags: 0)
!1084 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !1085, file: !983, line: 130)
!1085 = !DISubprogram(name: "puts", scope: !986, file: !986, line: 661, type: !677, flags: DIFlagPrototyped, spFlags: 0)
!1086 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !1087, file: !983, line: 131)
!1087 = !DISubprogram(name: "remove", scope: !986, file: !986, line: 152, type: !677, flags: DIFlagPrototyped, spFlags: 0)
!1088 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !1089, file: !983, line: 132)
!1089 = !DISubprogram(name: "rename", scope: !986, file: !986, line: 154, type: !1090, flags: DIFlagPrototyped, spFlags: 0)
!1090 = !DISubroutineType(types: !1091)
!1091 = !{!97, !550, !550}
!1092 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !1093, file: !983, line: 133)
!1093 = !DISubprogram(name: "rewind", scope: !986, file: !986, line: 723, type: !992, flags: DIFlagPrototyped, spFlags: 0)
!1094 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !1095, file: !983, line: 134)
!1095 = !DISubprogram(name: "scanf", scope: !986, file: !986, line: 421, type: !1078, flags: DIFlagPrototyped, spFlags: 0)
!1096 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !1097, file: !983, line: 135)
!1097 = !DISubprogram(name: "setbuf", scope: !986, file: !986, line: 328, type: !1098, flags: DIFlagPrototyped, spFlags: 0)
!1098 = !DISubroutineType(types: !1099)
!1099 = !{null, !1011, !775}
!1100 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !1101, file: !983, line: 136)
!1101 = !DISubprogram(name: "setvbuf", scope: !986, file: !986, line: 332, type: !1102, flags: DIFlagPrototyped, spFlags: 0)
!1102 = !DISubroutineType(types: !1103)
!1103 = !{!97, !1011, !775, !97, !154}
!1104 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !1105, file: !983, line: 137)
!1105 = !DISubprogram(name: "sprintf", scope: !986, file: !986, line: 358, type: !1106, flags: DIFlagPrototyped, spFlags: 0)
!1106 = !DISubroutineType(types: !1107)
!1107 = !{!97, !775, !734, null}
!1108 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !1109, file: !983, line: 138)
!1109 = !DISubprogram(name: "sscanf", scope: !986, file: !986, line: 423, type: !1110, flags: DIFlagPrototyped, spFlags: 0)
!1110 = !DISubroutineType(types: !1111)
!1111 = !{!97, !734, !734, null}
!1112 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !1113, file: !983, line: 139)
!1113 = !DISubprogram(name: "tmpfile", scope: !986, file: !986, line: 188, type: !1114, flags: DIFlagPrototyped, spFlags: 0)
!1114 = !DISubroutineType(types: !1115)
!1115 = !{!994}
!1116 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !1117, file: !983, line: 141)
!1117 = !DISubprogram(name: "tmpnam", scope: !986, file: !986, line: 205, type: !1070, flags: DIFlagPrototyped, spFlags: 0)
!1118 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !1119, file: !983, line: 143)
!1119 = !DISubprogram(name: "ungetc", scope: !986, file: !986, line: 668, type: !1028, flags: DIFlagPrototyped, spFlags: 0)
!1120 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !1121, file: !983, line: 144)
!1121 = !DISubprogram(name: "vfprintf", scope: !986, file: !986, line: 365, type: !1122, flags: DIFlagPrototyped, spFlags: 0)
!1122 = !DISubroutineType(types: !1123)
!1123 = !{!97, !1011, !734, !1124}
!1124 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !1125, size: 64)
!1125 = !DICompositeType(tag: DW_TAG_structure_type, name: "__va_list_tag", file: !3, flags: DIFlagFwdDecl, identifier: "_ZTS13__va_list_tag")
!1126 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !1127, file: !983, line: 145)
!1127 = !DISubprogram(name: "vprintf", scope: !986, file: !986, line: 371, type: !1128, flags: DIFlagPrototyped, spFlags: 0)
!1128 = !DISubroutineType(types: !1129)
!1129 = !{!97, !734, !1124}
!1130 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !1131, file: !983, line: 146)
!1131 = !DISubprogram(name: "vsprintf", scope: !986, file: !986, line: 373, type: !1132, flags: DIFlagPrototyped, spFlags: 0)
!1132 = !DISubroutineType(types: !1133)
!1133 = !{!97, !775, !734, !1124}
!1134 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !784, entity: !1135, file: !983, line: 175)
!1135 = !DISubprogram(name: "snprintf", scope: !986, file: !986, line: 378, type: !1136, flags: DIFlagPrototyped, spFlags: 0)
!1136 = !DISubroutineType(types: !1137)
!1137 = !{!97, !775, !154, !734, null}
!1138 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !784, entity: !1139, file: !983, line: 176)
!1139 = !DISubprogram(name: "vfscanf", scope: !986, file: !986, line: 459, type: !1122, flags: DIFlagPrototyped, spFlags: 0)
!1140 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !784, entity: !1141, file: !983, line: 177)
!1141 = !DISubprogram(name: "vscanf", scope: !986, file: !986, line: 467, type: !1128, flags: DIFlagPrototyped, spFlags: 0)
!1142 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !784, entity: !1143, file: !983, line: 178)
!1143 = !DISubprogram(name: "vsnprintf", scope: !986, file: !986, line: 382, type: !1144, flags: DIFlagPrototyped, spFlags: 0)
!1144 = !DISubroutineType(types: !1145)
!1145 = !{!97, !775, !154, !734, !1124}
!1146 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !784, entity: !1147, file: !983, line: 179)
!1147 = !DISubprogram(name: "vsscanf", scope: !986, file: !986, line: 471, type: !1148, flags: DIFlagPrototyped, spFlags: 0)
!1148 = !DISubroutineType(types: !1149)
!1149 = !{!97, !734, !734, !1124}
!1150 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !1135, file: !983, line: 185)
!1151 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !1139, file: !983, line: 186)
!1152 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !1141, file: !983, line: 187)
!1153 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !1143, file: !983, line: 188)
!1154 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !1147, file: !983, line: 189)
!1155 = !{i32 2, !"SDK Version", [2 x i32] [i32 7, i32 0]}
!1156 = !{i32 2, !"Dwarf Version", i32 4}
!1157 = !{i32 2, !"Debug Info Version", i32 3}
!1158 = !{i32 1, !"wchar_size", i32 4}
!1159 = !{!"clang version 9.0.0 (https://github.com/yebinchon/llvm-project/ a1efa594106d738d0b74c9e4e2b9b779eb8b7d25)"}
!1160 = distinct !DISubprogram(name: "randlc", linkageName: "_Z6randlcPdd", scope: !3, file: !3, line: 365, type: !1161, scopeLine: 365, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !1163)
!1161 = !DISubroutineType(types: !1162)
!1162 = !{!100, !99, !100}
!1163 = !{}
!1164 = !DILocalVariable(name: "x", arg: 1, scope: !1160, file: !3, line: 365, type: !99)
!1165 = !DILocation(line: 365, column: 23, scope: !1160)
!1166 = !DILocalVariable(name: "a", arg: 2, scope: !1160, file: !3, line: 365, type: !100)
!1167 = !DILocation(line: 365, column: 33, scope: !1160)
!1168 = !DILocalVariable(name: "t1", scope: !1160, file: !3, line: 366, type: !100)
!1169 = !DILocation(line: 366, column: 9, scope: !1160)
!1170 = !DILocalVariable(name: "t2", scope: !1160, file: !3, line: 366, type: !100)
!1171 = !DILocation(line: 366, column: 12, scope: !1160)
!1172 = !DILocalVariable(name: "t3", scope: !1160, file: !3, line: 366, type: !100)
!1173 = !DILocation(line: 366, column: 15, scope: !1160)
!1174 = !DILocalVariable(name: "t4", scope: !1160, file: !3, line: 366, type: !100)
!1175 = !DILocation(line: 366, column: 18, scope: !1160)
!1176 = !DILocalVariable(name: "a1", scope: !1160, file: !3, line: 366, type: !100)
!1177 = !DILocation(line: 366, column: 21, scope: !1160)
!1178 = !DILocalVariable(name: "a2", scope: !1160, file: !3, line: 366, type: !100)
!1179 = !DILocation(line: 366, column: 24, scope: !1160)
!1180 = !DILocalVariable(name: "x1", scope: !1160, file: !3, line: 366, type: !100)
!1181 = !DILocation(line: 366, column: 27, scope: !1160)
!1182 = !DILocalVariable(name: "x2", scope: !1160, file: !3, line: 366, type: !100)
!1183 = !DILocation(line: 366, column: 30, scope: !1160)
!1184 = !DILocalVariable(name: "z", scope: !1160, file: !3, line: 366, type: !100)
!1185 = !DILocation(line: 366, column: 33, scope: !1160)
!1186 = !DILocation(line: 373, column: 13, scope: !1160)
!1187 = !DILocation(line: 373, column: 11, scope: !1160)
!1188 = !DILocation(line: 373, column: 5, scope: !1160)
!1189 = !DILocation(line: 374, column: 12, scope: !1160)
!1190 = !DILocation(line: 374, column: 7, scope: !1160)
!1191 = !DILocation(line: 374, column: 5, scope: !1160)
!1192 = !DILocation(line: 375, column: 7, scope: !1160)
!1193 = !DILocation(line: 375, column: 17, scope: !1160)
!1194 = !DILocation(line: 375, column: 15, scope: !1160)
!1195 = !DILocation(line: 375, column: 9, scope: !1160)
!1196 = !DILocation(line: 375, column: 5, scope: !1160)
!1197 = !DILocation(line: 384, column: 15, scope: !1160)
!1198 = !DILocation(line: 384, column: 14, scope: !1160)
!1199 = !DILocation(line: 384, column: 11, scope: !1160)
!1200 = !DILocation(line: 384, column: 5, scope: !1160)
!1201 = !DILocation(line: 385, column: 12, scope: !1160)
!1202 = !DILocation(line: 385, column: 7, scope: !1160)
!1203 = !DILocation(line: 385, column: 5, scope: !1160)
!1204 = !DILocation(line: 386, column: 9, scope: !1160)
!1205 = !DILocation(line: 386, column: 8, scope: !1160)
!1206 = !DILocation(line: 386, column: 20, scope: !1160)
!1207 = !DILocation(line: 386, column: 18, scope: !1160)
!1208 = !DILocation(line: 386, column: 12, scope: !1160)
!1209 = !DILocation(line: 386, column: 5, scope: !1160)
!1210 = !DILocation(line: 387, column: 7, scope: !1160)
!1211 = !DILocation(line: 387, column: 12, scope: !1160)
!1212 = !DILocation(line: 387, column: 10, scope: !1160)
!1213 = !DILocation(line: 387, column: 17, scope: !1160)
!1214 = !DILocation(line: 387, column: 22, scope: !1160)
!1215 = !DILocation(line: 387, column: 20, scope: !1160)
!1216 = !DILocation(line: 387, column: 15, scope: !1160)
!1217 = !DILocation(line: 387, column: 5, scope: !1160)
!1218 = !DILocation(line: 388, column: 19, scope: !1160)
!1219 = !DILocation(line: 388, column: 17, scope: !1160)
!1220 = !DILocation(line: 388, column: 12, scope: !1160)
!1221 = !DILocation(line: 388, column: 7, scope: !1160)
!1222 = !DILocation(line: 388, column: 5, scope: !1160)
!1223 = !DILocation(line: 389, column: 6, scope: !1160)
!1224 = !DILocation(line: 389, column: 17, scope: !1160)
!1225 = !DILocation(line: 389, column: 15, scope: !1160)
!1226 = !DILocation(line: 389, column: 9, scope: !1160)
!1227 = !DILocation(line: 389, column: 4, scope: !1160)
!1228 = !DILocation(line: 390, column: 13, scope: !1160)
!1229 = !DILocation(line: 390, column: 11, scope: !1160)
!1230 = !DILocation(line: 390, column: 17, scope: !1160)
!1231 = !DILocation(line: 390, column: 22, scope: !1160)
!1232 = !DILocation(line: 390, column: 20, scope: !1160)
!1233 = !DILocation(line: 390, column: 15, scope: !1160)
!1234 = !DILocation(line: 390, column: 5, scope: !1160)
!1235 = !DILocation(line: 391, column: 19, scope: !1160)
!1236 = !DILocation(line: 391, column: 17, scope: !1160)
!1237 = !DILocation(line: 391, column: 12, scope: !1160)
!1238 = !DILocation(line: 391, column: 7, scope: !1160)
!1239 = !DILocation(line: 391, column: 5, scope: !1160)
!1240 = !DILocation(line: 392, column: 9, scope: !1160)
!1241 = !DILocation(line: 392, column: 20, scope: !1160)
!1242 = !DILocation(line: 392, column: 18, scope: !1160)
!1243 = !DILocation(line: 392, column: 12, scope: !1160)
!1244 = !DILocation(line: 392, column: 4, scope: !1160)
!1245 = !DILocation(line: 392, column: 7, scope: !1160)
!1246 = !DILocation(line: 394, column: 18, scope: !1160)
!1247 = !DILocation(line: 394, column: 17, scope: !1160)
!1248 = !DILocation(line: 394, column: 14, scope: !1160)
!1249 = !DILocation(line: 394, column: 2, scope: !1160)
!1250 = distinct !DISubprogram(name: "c_print_results", linkageName: "_Z15c_print_resultsPcciiiiddS_iS_S_S_S_S_S_S_S_S_S_S_S_S_S_", scope: !3, file: !3, line: 400, type: !1251, scopeLine: 423, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !1163)
!1251 = !DISubroutineType(types: !1252)
!1252 = !{null, !108, !109, !97, !97, !97, !97, !100, !100, !108, !97, !108, !108, !108, !108, !108, !108, !108, !108, !108, !108, !108, !108, !108, !108}
!1253 = !DILocalVariable(name: "name", arg: 1, scope: !1250, file: !3, line: 400, type: !108)
!1254 = !DILocation(line: 400, column: 28, scope: !1250)
!1255 = !DILocalVariable(name: "class_npb", arg: 2, scope: !1250, file: !3, line: 401, type: !109)
!1256 = !DILocation(line: 401, column: 8, scope: !1250)
!1257 = !DILocalVariable(name: "n1", arg: 3, scope: !1250, file: !3, line: 402, type: !97)
!1258 = !DILocation(line: 402, column: 7, scope: !1250)
!1259 = !DILocalVariable(name: "n2", arg: 4, scope: !1250, file: !3, line: 403, type: !97)
!1260 = !DILocation(line: 403, column: 7, scope: !1250)
!1261 = !DILocalVariable(name: "n3", arg: 5, scope: !1250, file: !3, line: 404, type: !97)
!1262 = !DILocation(line: 404, column: 7, scope: !1250)
!1263 = !DILocalVariable(name: "niter", arg: 6, scope: !1250, file: !3, line: 405, type: !97)
!1264 = !DILocation(line: 405, column: 7, scope: !1250)
!1265 = !DILocalVariable(name: "t", arg: 7, scope: !1250, file: !3, line: 406, type: !100)
!1266 = !DILocation(line: 406, column: 10, scope: !1250)
!1267 = !DILocalVariable(name: "mops", arg: 8, scope: !1250, file: !3, line: 407, type: !100)
!1268 = !DILocation(line: 407, column: 10, scope: !1250)
!1269 = !DILocalVariable(name: "optype", arg: 9, scope: !1250, file: !3, line: 408, type: !108)
!1270 = !DILocation(line: 408, column: 9, scope: !1250)
!1271 = !DILocalVariable(name: "passed_verification", arg: 10, scope: !1250, file: !3, line: 409, type: !97)
!1272 = !DILocation(line: 409, column: 7, scope: !1250)
!1273 = !DILocalVariable(name: "npbversion", arg: 11, scope: !1250, file: !3, line: 410, type: !108)
!1274 = !DILocation(line: 410, column: 9, scope: !1250)
!1275 = !DILocalVariable(name: "compiletime", arg: 12, scope: !1250, file: !3, line: 411, type: !108)
!1276 = !DILocation(line: 411, column: 9, scope: !1250)
!1277 = !DILocalVariable(name: "compilerversion", arg: 13, scope: !1250, file: !3, line: 412, type: !108)
!1278 = !DILocation(line: 412, column: 9, scope: !1250)
!1279 = !DILocalVariable(name: "libversion", arg: 14, scope: !1250, file: !3, line: 413, type: !108)
!1280 = !DILocation(line: 413, column: 9, scope: !1250)
!1281 = !DILocalVariable(name: "cpu_device", arg: 15, scope: !1250, file: !3, line: 414, type: !108)
!1282 = !DILocation(line: 414, column: 9, scope: !1250)
!1283 = !DILocalVariable(name: "gpu_device", arg: 16, scope: !1250, file: !3, line: 415, type: !108)
!1284 = !DILocation(line: 415, column: 9, scope: !1250)
!1285 = !DILocalVariable(name: "gpu_config", arg: 17, scope: !1250, file: !3, line: 416, type: !108)
!1286 = !DILocation(line: 416, column: 9, scope: !1250)
!1287 = !DILocalVariable(name: "cc", arg: 18, scope: !1250, file: !3, line: 417, type: !108)
!1288 = !DILocation(line: 417, column: 9, scope: !1250)
!1289 = !DILocalVariable(name: "clink", arg: 19, scope: !1250, file: !3, line: 418, type: !108)
!1290 = !DILocation(line: 418, column: 9, scope: !1250)
!1291 = !DILocalVariable(name: "c_lib", arg: 20, scope: !1250, file: !3, line: 419, type: !108)
!1292 = !DILocation(line: 419, column: 9, scope: !1250)
!1293 = !DILocalVariable(name: "c_inc", arg: 21, scope: !1250, file: !3, line: 420, type: !108)
!1294 = !DILocation(line: 420, column: 9, scope: !1250)
!1295 = !DILocalVariable(name: "cflags", arg: 22, scope: !1250, file: !3, line: 421, type: !108)
!1296 = !DILocation(line: 421, column: 9, scope: !1250)
!1297 = !DILocalVariable(name: "clinkflags", arg: 23, scope: !1250, file: !3, line: 422, type: !108)
!1298 = !DILocation(line: 422, column: 9, scope: !1250)
!1299 = !DILocalVariable(name: "rand", arg: 24, scope: !1250, file: !3, line: 423, type: !108)
!1300 = !DILocation(line: 423, column: 9, scope: !1250)
!1301 = !DILocation(line: 424, column: 44, scope: !1250)
!1302 = !DILocation(line: 424, column: 4, scope: !1250)
!1303 = !DILocation(line: 425, column: 61, scope: !1250)
!1304 = !DILocation(line: 425, column: 4, scope: !1250)
!1305 = !DILocation(line: 426, column: 8, scope: !1306)
!1306 = distinct !DILexicalBlock(scope: !1250, file: !3, line: 426, column: 7)
!1307 = !DILocation(line: 426, column: 15, scope: !1306)
!1308 = !DILocation(line: 426, column: 21, scope: !1306)
!1309 = !DILocation(line: 426, column: 24, scope: !1306)
!1310 = !DILocation(line: 426, column: 31, scope: !1306)
!1311 = !DILocation(line: 426, column: 7, scope: !1250)
!1312 = !DILocation(line: 427, column: 8, scope: !1313)
!1313 = distinct !DILexicalBlock(scope: !1314, file: !3, line: 427, column: 8)
!1314 = distinct !DILexicalBlock(scope: !1306, file: !3, line: 426, column: 38)
!1315 = !DILocation(line: 427, column: 10, scope: !1313)
!1316 = !DILocation(line: 427, column: 8, scope: !1314)
!1317 = !DILocalVariable(name: "nn", scope: !1318, file: !3, line: 428, type: !507)
!1318 = distinct !DILexicalBlock(scope: !1313, file: !3, line: 427, column: 14)
!1319 = !DILocation(line: 428, column: 11, scope: !1318)
!1320 = !DILocation(line: 428, column: 16, scope: !1318)
!1321 = !DILocation(line: 429, column: 9, scope: !1322)
!1322 = distinct !DILexicalBlock(scope: !1318, file: !3, line: 429, column: 9)
!1323 = !DILocation(line: 429, column: 11, scope: !1322)
!1324 = !DILocation(line: 429, column: 9, scope: !1318)
!1325 = !DILocation(line: 429, column: 20, scope: !1326)
!1326 = distinct !DILexicalBlock(scope: !1322, file: !3, line: 429, column: 15)
!1327 = !DILocation(line: 429, column: 18, scope: !1326)
!1328 = !DILocation(line: 429, column: 23, scope: !1326)
!1329 = !DILocation(line: 430, column: 55, scope: !1318)
!1330 = !DILocation(line: 430, column: 6, scope: !1318)
!1331 = !DILocation(line: 431, column: 5, scope: !1318)
!1332 = !DILocation(line: 432, column: 61, scope: !1333)
!1333 = distinct !DILexicalBlock(scope: !1313, file: !3, line: 431, column: 10)
!1334 = !DILocation(line: 432, column: 64, scope: !1333)
!1335 = !DILocation(line: 432, column: 67, scope: !1333)
!1336 = !DILocation(line: 432, column: 6, scope: !1333)
!1337 = !DILocation(line: 434, column: 4, scope: !1314)
!1338 = !DILocalVariable(name: "size", scope: !1339, file: !3, line: 435, type: !1340)
!1339 = distinct !DILexicalBlock(scope: !1306, file: !3, line: 434, column: 9)
!1340 = !DICompositeType(tag: DW_TAG_array_type, baseType: !109, size: 128, elements: !1341)
!1341 = !{!1342}
!1342 = !DISubrange(count: 16)
!1343 = !DILocation(line: 435, column: 10, scope: !1339)
!1344 = !DILocalVariable(name: "j", scope: !1339, file: !3, line: 436, type: !97)
!1345 = !DILocation(line: 436, column: 9, scope: !1339)
!1346 = !DILocation(line: 437, column: 9, scope: !1347)
!1347 = distinct !DILexicalBlock(scope: !1339, file: !3, line: 437, column: 8)
!1348 = !DILocation(line: 437, column: 11, scope: !1347)
!1349 = !DILocation(line: 437, column: 16, scope: !1347)
!1350 = !DILocation(line: 437, column: 20, scope: !1347)
!1351 = !DILocation(line: 437, column: 22, scope: !1347)
!1352 = !DILocation(line: 437, column: 8, scope: !1339)
!1353 = !DILocation(line: 438, column: 10, scope: !1354)
!1354 = distinct !DILexicalBlock(scope: !1355, file: !3, line: 438, column: 9)
!1355 = distinct !DILexicalBlock(scope: !1347, file: !3, line: 437, column: 27)
!1356 = !DILocation(line: 438, column: 17, scope: !1354)
!1357 = !DILocation(line: 438, column: 23, scope: !1354)
!1358 = !DILocation(line: 438, column: 26, scope: !1354)
!1359 = !DILocation(line: 438, column: 33, scope: !1354)
!1360 = !DILocation(line: 438, column: 9, scope: !1355)
!1361 = !DILocation(line: 439, column: 15, scope: !1362)
!1362 = distinct !DILexicalBlock(scope: !1354, file: !3, line: 438, column: 40)
!1363 = !DILocation(line: 439, column: 41, scope: !1362)
!1364 = !DILocation(line: 439, column: 32, scope: !1362)
!1365 = !DILocation(line: 439, column: 7, scope: !1362)
!1366 = !DILocation(line: 440, column: 9, scope: !1362)
!1367 = !DILocation(line: 441, column: 15, scope: !1368)
!1368 = distinct !DILexicalBlock(scope: !1362, file: !3, line: 441, column: 10)
!1369 = !DILocation(line: 441, column: 10, scope: !1368)
!1370 = !DILocation(line: 441, column: 18, scope: !1368)
!1371 = !DILocation(line: 441, column: 10, scope: !1362)
!1372 = !DILocation(line: 442, column: 13, scope: !1373)
!1373 = distinct !DILexicalBlock(scope: !1368, file: !3, line: 441, column: 25)
!1374 = !DILocation(line: 442, column: 8, scope: !1373)
!1375 = !DILocation(line: 442, column: 16, scope: !1373)
!1376 = !DILocation(line: 443, column: 9, scope: !1373)
!1377 = !DILocation(line: 444, column: 7, scope: !1373)
!1378 = !DILocation(line: 445, column: 12, scope: !1362)
!1379 = !DILocation(line: 445, column: 13, scope: !1362)
!1380 = !DILocation(line: 445, column: 7, scope: !1362)
!1381 = !DILocation(line: 445, column: 17, scope: !1362)
!1382 = !DILocation(line: 446, column: 52, scope: !1362)
!1383 = !DILocation(line: 446, column: 7, scope: !1362)
!1384 = !DILocation(line: 447, column: 6, scope: !1362)
!1385 = !DILocation(line: 448, column: 55, scope: !1386)
!1386 = distinct !DILexicalBlock(scope: !1354, file: !3, line: 447, column: 11)
!1387 = !DILocation(line: 448, column: 7, scope: !1386)
!1388 = !DILocation(line: 450, column: 5, scope: !1355)
!1389 = !DILocation(line: 451, column: 59, scope: !1390)
!1390 = distinct !DILexicalBlock(scope: !1347, file: !3, line: 450, column: 10)
!1391 = !DILocation(line: 451, column: 63, scope: !1390)
!1392 = !DILocation(line: 451, column: 67, scope: !1390)
!1393 = !DILocation(line: 451, column: 6, scope: !1390)
!1394 = !DILocation(line: 454, column: 52, scope: !1250)
!1395 = !DILocation(line: 454, column: 4, scope: !1250)
!1396 = !DILocation(line: 455, column: 54, scope: !1250)
!1397 = !DILocation(line: 455, column: 4, scope: !1250)
!1398 = !DILocation(line: 456, column: 54, scope: !1250)
!1399 = !DILocation(line: 456, column: 4, scope: !1250)
!1400 = !DILocation(line: 457, column: 40, scope: !1250)
!1401 = !DILocation(line: 457, column: 4, scope: !1250)
!1402 = !DILocation(line: 458, column: 7, scope: !1403)
!1403 = distinct !DILexicalBlock(scope: !1250, file: !3, line: 458, column: 7)
!1404 = !DILocation(line: 458, column: 27, scope: !1403)
!1405 = !DILocation(line: 458, column: 7, scope: !1250)
!1406 = !DILocation(line: 459, column: 5, scope: !1407)
!1407 = distinct !DILexicalBlock(scope: !1403, file: !3, line: 458, column: 31)
!1408 = !DILocation(line: 460, column: 4, scope: !1407)
!1409 = !DILocation(line: 460, column: 13, scope: !1410)
!1410 = distinct !DILexicalBlock(scope: !1403, file: !3, line: 460, column: 13)
!1411 = !DILocation(line: 460, column: 13, scope: !1403)
!1412 = !DILocation(line: 461, column: 5, scope: !1413)
!1413 = distinct !DILexicalBlock(scope: !1410, file: !3, line: 460, column: 33)
!1414 = !DILocation(line: 462, column: 4, scope: !1413)
!1415 = !DILocation(line: 463, column: 5, scope: !1416)
!1416 = distinct !DILexicalBlock(scope: !1410, file: !3, line: 462, column: 9)
!1417 = !DILocation(line: 465, column: 52, scope: !1250)
!1418 = !DILocation(line: 465, column: 4, scope: !1250)
!1419 = !DILocation(line: 466, column: 52, scope: !1250)
!1420 = !DILocation(line: 466, column: 4, scope: !1250)
!1421 = !DILocation(line: 467, column: 52, scope: !1250)
!1422 = !DILocation(line: 467, column: 4, scope: !1250)
!1423 = !DILocation(line: 468, column: 52, scope: !1250)
!1424 = !DILocation(line: 468, column: 4, scope: !1250)
!1425 = !DILocation(line: 469, column: 4, scope: !1250)
!1426 = !DILocation(line: 470, column: 38, scope: !1250)
!1427 = !DILocation(line: 470, column: 4, scope: !1250)
!1428 = !DILocation(line: 471, column: 38, scope: !1250)
!1429 = !DILocation(line: 471, column: 4, scope: !1250)
!1430 = !DILocation(line: 472, column: 38, scope: !1250)
!1431 = !DILocation(line: 472, column: 4, scope: !1250)
!1432 = !DILocation(line: 473, column: 38, scope: !1250)
!1433 = !DILocation(line: 473, column: 4, scope: !1250)
!1434 = !DILocation(line: 474, column: 38, scope: !1250)
!1435 = !DILocation(line: 474, column: 4, scope: !1250)
!1436 = !DILocation(line: 475, column: 38, scope: !1250)
!1437 = !DILocation(line: 475, column: 4, scope: !1250)
!1438 = !DILocation(line: 476, column: 38, scope: !1250)
!1439 = !DILocation(line: 476, column: 4, scope: !1250)
!1440 = !DILocation(line: 477, column: 4, scope: !1250)
!1441 = !DILocation(line: 478, column: 38, scope: !1250)
!1442 = !DILocation(line: 478, column: 4, scope: !1250)
!1443 = !DILocation(line: 479, column: 38, scope: !1250)
!1444 = !DILocation(line: 479, column: 4, scope: !1250)
!1445 = !DILocation(line: 480, column: 4, scope: !1250)
!1446 = !DILocation(line: 481, column: 38, scope: !1250)
!1447 = !DILocation(line: 481, column: 4, scope: !1250)
!1448 = !DILocation(line: 496, column: 4, scope: !1250)
!1449 = !DILocation(line: 497, column: 4, scope: !1250)
!1450 = !DILocation(line: 498, column: 4, scope: !1250)
!1451 = !DILocation(line: 499, column: 4, scope: !1250)
!1452 = !DILocation(line: 500, column: 4, scope: !1250)
!1453 = !DILocation(line: 501, column: 4, scope: !1250)
!1454 = !DILocation(line: 502, column: 4, scope: !1250)
!1455 = !DILocation(line: 503, column: 4, scope: !1250)
!1456 = !DILocation(line: 504, column: 4, scope: !1250)
!1457 = !DILocation(line: 505, column: 4, scope: !1250)
!1458 = !DILocation(line: 506, column: 3, scope: !1250)
!1459 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 510, type: !1460, scopeLine: 510, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !1163)
!1460 = !DISubroutineType(types: !1461)
!1461 = !{!97, !97, !760}
!1462 = !DILocalVariable(name: "argc", arg: 1, scope: !1459, file: !3, line: 510, type: !97)
!1463 = !DILocation(line: 510, column: 14, scope: !1459)
!1464 = !DILocalVariable(name: "argv", arg: 2, scope: !1459, file: !3, line: 510, type: !760)
!1465 = !DILocation(line: 510, column: 27, scope: !1459)
!1466 = !DILocation(line: 512, column: 17, scope: !1459)
!1467 = !DILocation(line: 512, column: 11, scope: !1459)
!1468 = !DILocation(line: 512, column: 9, scope: !1459)
!1469 = !DILocation(line: 513, column: 17, scope: !1459)
!1470 = !DILocation(line: 513, column: 11, scope: !1459)
!1471 = !DILocation(line: 513, column: 9, scope: !1459)
!1472 = !DILocation(line: 514, column: 13, scope: !1459)
!1473 = !DILocation(line: 514, column: 7, scope: !1459)
!1474 = !DILocation(line: 514, column: 5, scope: !1459)
!1475 = !DILocation(line: 515, column: 15, scope: !1459)
!1476 = !DILocation(line: 515, column: 9, scope: !1459)
!1477 = !DILocation(line: 515, column: 7, scope: !1459)
!1478 = !DILocation(line: 516, column: 15, scope: !1459)
!1479 = !DILocation(line: 516, column: 9, scope: !1459)
!1480 = !DILocation(line: 516, column: 7, scope: !1459)
!1481 = !DILocation(line: 517, column: 18, scope: !1459)
!1482 = !DILocation(line: 517, column: 9, scope: !1459)
!1483 = !DILocation(line: 517, column: 7, scope: !1459)
!1484 = !DILocation(line: 518, column: 15, scope: !1459)
!1485 = !DILocation(line: 518, column: 6, scope: !1459)
!1486 = !DILocation(line: 518, column: 4, scope: !1459)
!1487 = !DILocation(line: 519, column: 15, scope: !1459)
!1488 = !DILocation(line: 519, column: 6, scope: !1459)
!1489 = !DILocation(line: 519, column: 4, scope: !1459)
!1490 = !DILocation(line: 520, column: 15, scope: !1459)
!1491 = !DILocation(line: 520, column: 6, scope: !1459)
!1492 = !DILocation(line: 520, column: 4, scope: !1459)
!1493 = !DILocation(line: 521, column: 15, scope: !1459)
!1494 = !DILocation(line: 521, column: 6, scope: !1459)
!1495 = !DILocation(line: 521, column: 4, scope: !1459)
!1496 = !DILocation(line: 522, column: 15, scope: !1459)
!1497 = !DILocation(line: 522, column: 6, scope: !1459)
!1498 = !DILocation(line: 522, column: 4, scope: !1459)
!1499 = !DILocation(line: 523, column: 15, scope: !1459)
!1500 = !DILocation(line: 523, column: 6, scope: !1459)
!1501 = !DILocation(line: 523, column: 4, scope: !1459)
!1502 = !DILocation(line: 524, column: 2, scope: !1459)
!1503 = !DILocalVariable(name: "i", scope: !1459, file: !3, line: 529, type: !97)
!1504 = !DILocation(line: 529, column: 6, scope: !1459)
!1505 = !DILocalVariable(name: "j", scope: !1459, file: !3, line: 529, type: !97)
!1506 = !DILocation(line: 529, column: 9, scope: !1459)
!1507 = !DILocalVariable(name: "k", scope: !1459, file: !3, line: 529, type: !97)
!1508 = !DILocation(line: 529, column: 12, scope: !1459)
!1509 = !DILocalVariable(name: "it", scope: !1459, file: !3, line: 529, type: !97)
!1510 = !DILocation(line: 529, column: 15, scope: !1459)
!1511 = !DILocalVariable(name: "zeta", scope: !1459, file: !3, line: 530, type: !100)
!1512 = !DILocation(line: 530, column: 9, scope: !1459)
!1513 = !DILocalVariable(name: "rnorm", scope: !1459, file: !3, line: 531, type: !100)
!1514 = !DILocation(line: 531, column: 9, scope: !1459)
!1515 = !DILocalVariable(name: "norm_temp1", scope: !1459, file: !3, line: 532, type: !100)
!1516 = !DILocation(line: 532, column: 9, scope: !1459)
!1517 = !DILocalVariable(name: "norm_temp2", scope: !1459, file: !3, line: 532, type: !100)
!1518 = !DILocation(line: 532, column: 21, scope: !1459)
!1519 = !DILocalVariable(name: "t", scope: !1459, file: !3, line: 533, type: !100)
!1520 = !DILocation(line: 533, column: 9, scope: !1459)
!1521 = !DILocalVariable(name: "mflops", scope: !1459, file: !3, line: 533, type: !100)
!1522 = !DILocation(line: 533, column: 12, scope: !1459)
!1523 = !DILocalVariable(name: "class_npb", scope: !1459, file: !3, line: 534, type: !109)
!1524 = !DILocation(line: 534, column: 7, scope: !1459)
!1525 = !DILocalVariable(name: "verified", scope: !1459, file: !3, line: 535, type: !1526)
!1526 = !DIDerivedType(tag: DW_TAG_typedef, name: "boolean", file: !1527, line: 80, baseType: !97)
!1527 = !DIFile(filename: "./../common/npb-CPP.hpp", directory: "/scratch/ah7226/NPB-GPU/CUDA/CG")
!1528 = !DILocation(line: 535, column: 10, scope: !1459)
!1529 = !DILocalVariable(name: "zeta_verify_value", scope: !1459, file: !3, line: 536, type: !100)
!1530 = !DILocation(line: 536, column: 9, scope: !1459)
!1531 = !DILocalVariable(name: "epsilon", scope: !1459, file: !3, line: 536, type: !100)
!1532 = !DILocation(line: 536, column: 28, scope: !1459)
!1533 = !DILocalVariable(name: "err", scope: !1459, file: !3, line: 536, type: !100)
!1534 = !DILocation(line: 536, column: 37, scope: !1459)
!1535 = !DILocation(line: 553, column: 11, scope: !1459)
!1536 = !DILocation(line: 554, column: 11, scope: !1459)
!1537 = !DILocation(line: 555, column: 11, scope: !1459)
!1538 = !DILocation(line: 556, column: 11, scope: !1459)
!1539 = !DILocation(line: 565, column: 13, scope: !1540)
!1540 = distinct !DILexicalBlock(scope: !1541, file: !3, line: 564, column: 71)
!1541 = distinct !DILexicalBlock(scope: !1542, file: !3, line: 564, column: 11)
!1542 = distinct !DILexicalBlock(scope: !1543, file: !3, line: 561, column: 11)
!1543 = distinct !DILexicalBlock(scope: !1459, file: !3, line: 558, column: 5)
!1544 = !DILocation(line: 566, column: 21, scope: !1540)
!1545 = !DILocation(line: 583, column: 2, scope: !1459)
!1546 = !DILocation(line: 584, column: 2, scope: !1459)
!1547 = !DILocation(line: 585, column: 2, scope: !1459)
!1548 = !DILocation(line: 587, column: 6, scope: !1459)
!1549 = !DILocation(line: 588, column: 6, scope: !1459)
!1550 = !DILocation(line: 591, column: 10, scope: !1459)
!1551 = !DILocation(line: 592, column: 10, scope: !1459)
!1552 = !DILocation(line: 593, column: 27, scope: !1459)
!1553 = !DILocation(line: 593, column: 12, scope: !1459)
!1554 = !DILocation(line: 593, column: 10, scope: !1459)
!1555 = !DILocation(line: 595, column: 8, scope: !1459)
!1556 = !DILocation(line: 596, column: 4, scope: !1459)
!1557 = !DILocation(line: 597, column: 4, scope: !1459)
!1558 = !DILocation(line: 598, column: 4, scope: !1459)
!1559 = !DILocation(line: 599, column: 4, scope: !1459)
!1560 = !DILocation(line: 600, column: 4, scope: !1459)
!1561 = !DILocation(line: 601, column: 4, scope: !1459)
!1562 = !DILocation(line: 602, column: 4, scope: !1459)
!1563 = !DILocation(line: 603, column: 4, scope: !1459)
!1564 = !DILocation(line: 604, column: 4, scope: !1459)
!1565 = !DILocation(line: 605, column: 29, scope: !1459)
!1566 = !DILocation(line: 605, column: 4, scope: !1459)
!1567 = !DILocation(line: 606, column: 32, scope: !1459)
!1568 = !DILocation(line: 606, column: 4, scope: !1459)
!1569 = !DILocation(line: 607, column: 4, scope: !1459)
!1570 = !DILocation(line: 595, column: 2, scope: !1459)
!1571 = !DILocation(line: 619, column: 8, scope: !1572)
!1572 = distinct !DILexicalBlock(scope: !1459, file: !3, line: 619, column: 2)
!1573 = !DILocation(line: 619, column: 6, scope: !1572)
!1574 = !DILocation(line: 619, column: 13, scope: !1575)
!1575 = distinct !DILexicalBlock(scope: !1572, file: !3, line: 619, column: 2)
!1576 = !DILocation(line: 619, column: 17, scope: !1575)
!1577 = !DILocation(line: 619, column: 27, scope: !1575)
!1578 = !DILocation(line: 619, column: 25, scope: !1575)
!1579 = !DILocation(line: 619, column: 36, scope: !1575)
!1580 = !DILocation(line: 619, column: 15, scope: !1575)
!1581 = !DILocation(line: 619, column: 2, scope: !1572)
!1582 = !DILocation(line: 620, column: 11, scope: !1583)
!1583 = distinct !DILexicalBlock(scope: !1584, file: !3, line: 620, column: 3)
!1584 = distinct !DILexicalBlock(scope: !1575, file: !3, line: 619, column: 45)
!1585 = !DILocation(line: 620, column: 18, scope: !1583)
!1586 = !DILocation(line: 620, column: 9, scope: !1583)
!1587 = !DILocation(line: 620, column: 7, scope: !1583)
!1588 = !DILocation(line: 620, column: 22, scope: !1589)
!1589 = distinct !DILexicalBlock(scope: !1583, file: !3, line: 620, column: 3)
!1590 = !DILocation(line: 620, column: 26, scope: !1589)
!1591 = !DILocation(line: 620, column: 33, scope: !1589)
!1592 = !DILocation(line: 620, column: 34, scope: !1589)
!1593 = !DILocation(line: 620, column: 24, scope: !1589)
!1594 = !DILocation(line: 620, column: 3, scope: !1583)
!1595 = !DILocation(line: 621, column: 16, scope: !1596)
!1596 = distinct !DILexicalBlock(scope: !1589, file: !3, line: 620, column: 43)
!1597 = !DILocation(line: 621, column: 23, scope: !1596)
!1598 = !DILocation(line: 621, column: 28, scope: !1596)
!1599 = !DILocation(line: 621, column: 26, scope: !1596)
!1600 = !DILocation(line: 621, column: 4, scope: !1596)
!1601 = !DILocation(line: 621, column: 11, scope: !1596)
!1602 = !DILocation(line: 621, column: 14, scope: !1596)
!1603 = !DILocation(line: 622, column: 3, scope: !1596)
!1604 = !DILocation(line: 620, column: 40, scope: !1589)
!1605 = !DILocation(line: 620, column: 3, scope: !1589)
!1606 = distinct !{!1606, !1594, !1607}
!1607 = !DILocation(line: 622, column: 3, scope: !1583)
!1608 = !DILocation(line: 623, column: 2, scope: !1584)
!1609 = !DILocation(line: 619, column: 42, scope: !1575)
!1610 = !DILocation(line: 619, column: 2, scope: !1575)
!1611 = distinct !{!1611, !1581, !1612}
!1612 = !DILocation(line: 623, column: 2, scope: !1572)
!1613 = !DILocation(line: 626, column: 8, scope: !1614)
!1614 = distinct !DILexicalBlock(scope: !1459, file: !3, line: 626, column: 2)
!1615 = !DILocation(line: 626, column: 6, scope: !1614)
!1616 = !DILocation(line: 626, column: 13, scope: !1617)
!1617 = distinct !DILexicalBlock(scope: !1614, file: !3, line: 626, column: 2)
!1618 = !DILocation(line: 626, column: 15, scope: !1617)
!1619 = !DILocation(line: 626, column: 2, scope: !1614)
!1620 = !DILocation(line: 627, column: 3, scope: !1621)
!1621 = distinct !DILexicalBlock(scope: !1617, file: !3, line: 626, column: 27)
!1622 = !DILocation(line: 627, column: 5, scope: !1621)
!1623 = !DILocation(line: 627, column: 8, scope: !1621)
!1624 = !DILocation(line: 628, column: 2, scope: !1621)
!1625 = !DILocation(line: 626, column: 24, scope: !1617)
!1626 = !DILocation(line: 626, column: 2, scope: !1617)
!1627 = distinct !{!1627, !1619, !1628}
!1628 = !DILocation(line: 628, column: 2, scope: !1614)
!1629 = !DILocation(line: 629, column: 8, scope: !1630)
!1630 = distinct !DILexicalBlock(scope: !1459, file: !3, line: 629, column: 2)
!1631 = !DILocation(line: 629, column: 6, scope: !1630)
!1632 = !DILocation(line: 629, column: 13, scope: !1633)
!1633 = distinct !DILexicalBlock(scope: !1630, file: !3, line: 629, column: 2)
!1634 = !DILocation(line: 629, column: 15, scope: !1633)
!1635 = !DILocation(line: 629, column: 23, scope: !1633)
!1636 = !DILocation(line: 629, column: 22, scope: !1633)
!1637 = !DILocation(line: 629, column: 31, scope: !1633)
!1638 = !DILocation(line: 629, column: 14, scope: !1633)
!1639 = !DILocation(line: 629, column: 2, scope: !1630)
!1640 = !DILocation(line: 630, column: 3, scope: !1641)
!1641 = distinct !DILexicalBlock(scope: !1633, file: !3, line: 629, column: 39)
!1642 = !DILocation(line: 630, column: 5, scope: !1641)
!1643 = !DILocation(line: 630, column: 8, scope: !1641)
!1644 = !DILocation(line: 631, column: 3, scope: !1641)
!1645 = !DILocation(line: 631, column: 5, scope: !1641)
!1646 = !DILocation(line: 631, column: 8, scope: !1641)
!1647 = !DILocation(line: 632, column: 3, scope: !1641)
!1648 = !DILocation(line: 632, column: 5, scope: !1641)
!1649 = !DILocation(line: 632, column: 8, scope: !1641)
!1650 = !DILocation(line: 633, column: 3, scope: !1641)
!1651 = !DILocation(line: 633, column: 5, scope: !1641)
!1652 = !DILocation(line: 633, column: 8, scope: !1641)
!1653 = !DILocation(line: 634, column: 2, scope: !1641)
!1654 = !DILocation(line: 629, column: 36, scope: !1633)
!1655 = !DILocation(line: 629, column: 2, scope: !1633)
!1656 = distinct !{!1656, !1639, !1657}
!1657 = !DILocation(line: 634, column: 2, scope: !1630)
!1658 = !DILocation(line: 635, column: 7, scope: !1459)
!1659 = !DILocation(line: 643, column: 9, scope: !1660)
!1660 = distinct !DILexicalBlock(scope: !1459, file: !3, line: 643, column: 2)
!1661 = !DILocation(line: 643, column: 6, scope: !1660)
!1662 = !DILocation(line: 643, column: 14, scope: !1663)
!1663 = distinct !DILexicalBlock(scope: !1660, file: !3, line: 643, column: 2)
!1664 = !DILocation(line: 643, column: 17, scope: !1663)
!1665 = !DILocation(line: 643, column: 2, scope: !1660)
!1666 = !DILocation(line: 645, column: 13, scope: !1667)
!1667 = distinct !DILexicalBlock(scope: !1663, file: !3, line: 643, column: 28)
!1668 = !DILocation(line: 645, column: 21, scope: !1667)
!1669 = !DILocation(line: 645, column: 29, scope: !1667)
!1670 = !DILocation(line: 645, column: 32, scope: !1667)
!1671 = !DILocation(line: 645, column: 35, scope: !1667)
!1672 = !DILocation(line: 645, column: 38, scope: !1667)
!1673 = !DILocation(line: 645, column: 41, scope: !1667)
!1674 = !DILocation(line: 645, column: 44, scope: !1667)
!1675 = !DILocation(line: 645, column: 3, scope: !1667)
!1676 = !DILocation(line: 655, column: 14, scope: !1667)
!1677 = !DILocation(line: 656, column: 14, scope: !1667)
!1678 = !DILocation(line: 657, column: 9, scope: !1679)
!1679 = distinct !DILexicalBlock(scope: !1667, file: !3, line: 657, column: 3)
!1680 = !DILocation(line: 657, column: 7, scope: !1679)
!1681 = !DILocation(line: 657, column: 14, scope: !1682)
!1682 = distinct !DILexicalBlock(scope: !1679, file: !3, line: 657, column: 3)
!1683 = !DILocation(line: 657, column: 18, scope: !1682)
!1684 = !DILocation(line: 657, column: 28, scope: !1682)
!1685 = !DILocation(line: 657, column: 26, scope: !1682)
!1686 = !DILocation(line: 657, column: 37, scope: !1682)
!1687 = !DILocation(line: 657, column: 16, scope: !1682)
!1688 = !DILocation(line: 657, column: 3, scope: !1679)
!1689 = !DILocation(line: 658, column: 17, scope: !1690)
!1690 = distinct !DILexicalBlock(scope: !1682, file: !3, line: 657, column: 46)
!1691 = !DILocation(line: 658, column: 30, scope: !1690)
!1692 = !DILocation(line: 658, column: 32, scope: !1690)
!1693 = !DILocation(line: 658, column: 37, scope: !1690)
!1694 = !DILocation(line: 658, column: 39, scope: !1690)
!1695 = !DILocation(line: 658, column: 35, scope: !1690)
!1696 = !DILocation(line: 658, column: 28, scope: !1690)
!1697 = !DILocation(line: 658, column: 15, scope: !1690)
!1698 = !DILocation(line: 659, column: 17, scope: !1690)
!1699 = !DILocation(line: 659, column: 30, scope: !1690)
!1700 = !DILocation(line: 659, column: 32, scope: !1690)
!1701 = !DILocation(line: 659, column: 37, scope: !1690)
!1702 = !DILocation(line: 659, column: 39, scope: !1690)
!1703 = !DILocation(line: 659, column: 35, scope: !1690)
!1704 = !DILocation(line: 659, column: 28, scope: !1690)
!1705 = !DILocation(line: 659, column: 15, scope: !1690)
!1706 = !DILocation(line: 660, column: 3, scope: !1690)
!1707 = !DILocation(line: 657, column: 43, scope: !1682)
!1708 = !DILocation(line: 657, column: 3, scope: !1682)
!1709 = distinct !{!1709, !1688, !1710}
!1710 = !DILocation(line: 660, column: 3, scope: !1679)
!1711 = !DILocation(line: 661, column: 27, scope: !1667)
!1712 = !DILocation(line: 661, column: 22, scope: !1667)
!1713 = !DILocation(line: 661, column: 20, scope: !1667)
!1714 = !DILocation(line: 661, column: 14, scope: !1667)
!1715 = !DILocation(line: 664, column: 9, scope: !1716)
!1716 = distinct !DILexicalBlock(scope: !1667, file: !3, line: 664, column: 3)
!1717 = !DILocation(line: 664, column: 7, scope: !1716)
!1718 = !DILocation(line: 664, column: 14, scope: !1719)
!1719 = distinct !DILexicalBlock(scope: !1716, file: !3, line: 664, column: 3)
!1720 = !DILocation(line: 664, column: 18, scope: !1719)
!1721 = !DILocation(line: 664, column: 28, scope: !1719)
!1722 = !DILocation(line: 664, column: 26, scope: !1719)
!1723 = !DILocation(line: 664, column: 37, scope: !1719)
!1724 = !DILocation(line: 664, column: 16, scope: !1719)
!1725 = !DILocation(line: 664, column: 3, scope: !1716)
!1726 = !DILocation(line: 665, column: 11, scope: !1727)
!1727 = distinct !DILexicalBlock(scope: !1719, file: !3, line: 664, column: 46)
!1728 = !DILocation(line: 665, column: 24, scope: !1727)
!1729 = !DILocation(line: 665, column: 26, scope: !1727)
!1730 = !DILocation(line: 665, column: 22, scope: !1727)
!1731 = !DILocation(line: 665, column: 4, scope: !1727)
!1732 = !DILocation(line: 665, column: 6, scope: !1727)
!1733 = !DILocation(line: 665, column: 9, scope: !1727)
!1734 = !DILocation(line: 666, column: 3, scope: !1727)
!1735 = !DILocation(line: 664, column: 43, scope: !1719)
!1736 = !DILocation(line: 664, column: 3, scope: !1719)
!1737 = distinct !{!1737, !1725, !1738}
!1738 = !DILocation(line: 666, column: 3, scope: !1716)
!1739 = !DILocation(line: 667, column: 2, scope: !1667)
!1740 = !DILocation(line: 643, column: 25, scope: !1663)
!1741 = !DILocation(line: 643, column: 2, scope: !1663)
!1742 = distinct !{!1742, !1665, !1743}
!1743 = !DILocation(line: 667, column: 2, scope: !1660)
!1744 = !DILocation(line: 670, column: 8, scope: !1745)
!1745 = distinct !DILexicalBlock(scope: !1459, file: !3, line: 670, column: 2)
!1746 = !DILocation(line: 670, column: 6, scope: !1745)
!1747 = !DILocation(line: 670, column: 13, scope: !1748)
!1748 = distinct !DILexicalBlock(scope: !1745, file: !3, line: 670, column: 2)
!1749 = !DILocation(line: 670, column: 15, scope: !1748)
!1750 = !DILocation(line: 670, column: 2, scope: !1745)
!1751 = !DILocation(line: 671, column: 3, scope: !1752)
!1752 = distinct !DILexicalBlock(scope: !1748, file: !3, line: 670, column: 27)
!1753 = !DILocation(line: 671, column: 5, scope: !1752)
!1754 = !DILocation(line: 671, column: 8, scope: !1752)
!1755 = !DILocation(line: 672, column: 2, scope: !1752)
!1756 = !DILocation(line: 670, column: 24, scope: !1748)
!1757 = !DILocation(line: 670, column: 2, scope: !1748)
!1758 = distinct !{!1758, !1750, !1759}
!1759 = !DILocation(line: 672, column: 2, scope: !1745)
!1760 = !DILocation(line: 673, column: 7, scope: !1459)
!1761 = !DILocation(line: 675, column: 2, scope: !1459)
!1762 = !DILocation(line: 685, column: 9, scope: !1763)
!1763 = distinct !DILexicalBlock(scope: !1459, file: !3, line: 685, column: 2)
!1764 = !DILocation(line: 685, column: 6, scope: !1763)
!1765 = !DILocation(line: 685, column: 14, scope: !1766)
!1766 = distinct !DILexicalBlock(scope: !1763, file: !3, line: 685, column: 2)
!1767 = !DILocation(line: 685, column: 17, scope: !1766)
!1768 = !DILocation(line: 685, column: 2, scope: !1763)
!1769 = !DILocation(line: 687, column: 3, scope: !1770)
!1770 = distinct !DILexicalBlock(scope: !1766, file: !3, line: 685, column: 32)
!1771 = !DILocation(line: 697, column: 3, scope: !1770)
!1772 = !DILocation(line: 698, column: 27, scope: !1770)
!1773 = !DILocation(line: 698, column: 22, scope: !1770)
!1774 = !DILocation(line: 698, column: 20, scope: !1770)
!1775 = !DILocation(line: 698, column: 14, scope: !1770)
!1776 = !DILocation(line: 699, column: 24, scope: !1770)
!1777 = !DILocation(line: 699, column: 22, scope: !1770)
!1778 = !DILocation(line: 699, column: 16, scope: !1770)
!1779 = !DILocation(line: 699, column: 8, scope: !1770)
!1780 = !DILocation(line: 700, column: 6, scope: !1781)
!1781 = distinct !DILexicalBlock(scope: !1770, file: !3, line: 700, column: 6)
!1782 = !DILocation(line: 700, column: 8, scope: !1781)
!1783 = !DILocation(line: 700, column: 6, scope: !1770)
!1784 = !DILocation(line: 700, column: 13, scope: !1785)
!1785 = distinct !DILexicalBlock(scope: !1781, file: !3, line: 700, column: 12)
!1786 = !DILocation(line: 700, column: 77, scope: !1785)
!1787 = !DILocation(line: 701, column: 44, scope: !1770)
!1788 = !DILocation(line: 701, column: 48, scope: !1770)
!1789 = !DILocation(line: 701, column: 55, scope: !1770)
!1790 = !DILocation(line: 701, column: 3, scope: !1770)
!1791 = !DILocation(line: 704, column: 26, scope: !1770)
!1792 = !DILocation(line: 704, column: 3, scope: !1770)
!1793 = !DILocation(line: 705, column: 2, scope: !1770)
!1794 = !DILocation(line: 685, column: 29, scope: !1766)
!1795 = !DILocation(line: 685, column: 2, scope: !1766)
!1796 = distinct !{!1796, !1768, !1797}
!1797 = !DILocation(line: 705, column: 2, scope: !1763)
!1798 = !DILocation(line: 716, column: 4, scope: !1459)
!1799 = !DILocation(line: 719, column: 2, scope: !1459)
!1800 = !DILocation(line: 721, column: 10, scope: !1459)
!1801 = !DILocation(line: 722, column: 5, scope: !1802)
!1802 = distinct !DILexicalBlock(scope: !1459, file: !3, line: 722, column: 5)
!1803 = !DILocation(line: 722, column: 15, scope: !1802)
!1804 = !DILocation(line: 722, column: 5, scope: !1459)
!1805 = !DILocation(line: 723, column: 14, scope: !1806)
!1806 = distinct !DILexicalBlock(scope: !1802, file: !3, line: 722, column: 22)
!1807 = !DILocation(line: 723, column: 21, scope: !1806)
!1808 = !DILocation(line: 723, column: 19, scope: !1806)
!1809 = !DILocation(line: 723, column: 9, scope: !1806)
!1810 = !DILocation(line: 723, column: 42, scope: !1806)
!1811 = !DILocation(line: 723, column: 40, scope: !1806)
!1812 = !DILocation(line: 723, column: 7, scope: !1806)
!1813 = !DILocation(line: 724, column: 6, scope: !1814)
!1814 = distinct !DILexicalBlock(scope: !1806, file: !3, line: 724, column: 6)
!1815 = !DILocation(line: 724, column: 13, scope: !1814)
!1816 = !DILocation(line: 724, column: 10, scope: !1814)
!1817 = !DILocation(line: 724, column: 6, scope: !1806)
!1818 = !DILocation(line: 725, column: 13, scope: !1819)
!1819 = distinct !DILexicalBlock(scope: !1814, file: !3, line: 724, column: 21)
!1820 = !DILocation(line: 726, column: 4, scope: !1819)
!1821 = !DILocation(line: 727, column: 36, scope: !1819)
!1822 = !DILocation(line: 727, column: 4, scope: !1819)
!1823 = !DILocation(line: 728, column: 36, scope: !1819)
!1824 = !DILocation(line: 728, column: 4, scope: !1819)
!1825 = !DILocation(line: 729, column: 3, scope: !1819)
!1826 = !DILocation(line: 730, column: 13, scope: !1827)
!1827 = distinct !DILexicalBlock(scope: !1814, file: !3, line: 729, column: 8)
!1828 = !DILocation(line: 731, column: 4, scope: !1827)
!1829 = !DILocation(line: 732, column: 45, scope: !1827)
!1830 = !DILocation(line: 732, column: 4, scope: !1827)
!1831 = !DILocation(line: 733, column: 45, scope: !1827)
!1832 = !DILocation(line: 733, column: 4, scope: !1827)
!1833 = !DILocation(line: 735, column: 2, scope: !1806)
!1834 = !DILocation(line: 736, column: 12, scope: !1835)
!1835 = distinct !DILexicalBlock(scope: !1802, file: !3, line: 735, column: 7)
!1836 = !DILocation(line: 737, column: 3, scope: !1835)
!1837 = !DILocation(line: 738, column: 3, scope: !1835)
!1838 = !DILocation(line: 740, column: 5, scope: !1839)
!1839 = distinct !DILexicalBlock(scope: !1459, file: !3, line: 740, column: 5)
!1840 = !DILocation(line: 740, column: 7, scope: !1839)
!1841 = !DILocation(line: 740, column: 5, scope: !1459)
!1842 = !DILocation(line: 745, column: 6, scope: !1843)
!1843 = distinct !DILexicalBlock(scope: !1839, file: !3, line: 740, column: 14)
!1844 = !DILocation(line: 745, column: 4, scope: !1843)
!1845 = !DILocation(line: 745, column: 8, scope: !1843)
!1846 = !DILocation(line: 741, column: 10, scope: !1843)
!1847 = !DILocation(line: 746, column: 2, scope: !1843)
!1848 = !DILocation(line: 747, column: 10, scope: !1849)
!1849 = distinct !DILexicalBlock(scope: !1839, file: !3, line: 746, column: 7)
!1850 = !DILocalVariable(name: "gpu_config", scope: !1459, file: !3, line: 750, type: !294)
!1851 = !DILocation(line: 750, column: 7, scope: !1459)
!1852 = !DILocalVariable(name: "gpu_config_string", scope: !1459, file: !3, line: 751, type: !1853)
!1853 = !DICompositeType(tag: DW_TAG_array_type, baseType: !109, size: 16384, elements: !1854)
!1854 = !{!1855}
!1855 = !DISubrange(count: 2048)
!1856 = !DILocation(line: 751, column: 7, scope: !1459)
!1857 = !DILocation(line: 778, column: 10, scope: !1459)
!1858 = !DILocation(line: 778, column: 2, scope: !1459)
!1859 = !DILocation(line: 779, column: 9, scope: !1459)
!1860 = !DILocation(line: 779, column: 28, scope: !1459)
!1861 = !DILocation(line: 779, column: 2, scope: !1459)
!1862 = !DILocation(line: 780, column: 10, scope: !1459)
!1863 = !DILocation(line: 780, column: 46, scope: !1459)
!1864 = !DILocation(line: 780, column: 2, scope: !1459)
!1865 = !DILocation(line: 781, column: 9, scope: !1459)
!1866 = !DILocation(line: 781, column: 28, scope: !1459)
!1867 = !DILocation(line: 781, column: 2, scope: !1459)
!1868 = !DILocation(line: 782, column: 10, scope: !1459)
!1869 = !DILocation(line: 782, column: 46, scope: !1459)
!1870 = !DILocation(line: 782, column: 2, scope: !1459)
!1871 = !DILocation(line: 783, column: 9, scope: !1459)
!1872 = !DILocation(line: 783, column: 28, scope: !1459)
!1873 = !DILocation(line: 783, column: 2, scope: !1459)
!1874 = !DILocation(line: 784, column: 10, scope: !1459)
!1875 = !DILocation(line: 784, column: 48, scope: !1459)
!1876 = !DILocation(line: 784, column: 2, scope: !1459)
!1877 = !DILocation(line: 785, column: 9, scope: !1459)
!1878 = !DILocation(line: 785, column: 28, scope: !1459)
!1879 = !DILocation(line: 785, column: 2, scope: !1459)
!1880 = !DILocation(line: 786, column: 10, scope: !1459)
!1881 = !DILocation(line: 786, column: 47, scope: !1459)
!1882 = !DILocation(line: 786, column: 2, scope: !1459)
!1883 = !DILocation(line: 787, column: 9, scope: !1459)
!1884 = !DILocation(line: 787, column: 28, scope: !1459)
!1885 = !DILocation(line: 787, column: 2, scope: !1459)
!1886 = !DILocation(line: 788, column: 10, scope: !1459)
!1887 = !DILocation(line: 788, column: 47, scope: !1459)
!1888 = !DILocation(line: 788, column: 2, scope: !1459)
!1889 = !DILocation(line: 789, column: 9, scope: !1459)
!1890 = !DILocation(line: 789, column: 28, scope: !1459)
!1891 = !DILocation(line: 789, column: 2, scope: !1459)
!1892 = !DILocation(line: 790, column: 10, scope: !1459)
!1893 = !DILocation(line: 790, column: 46, scope: !1459)
!1894 = !DILocation(line: 790, column: 2, scope: !1459)
!1895 = !DILocation(line: 791, column: 9, scope: !1459)
!1896 = !DILocation(line: 791, column: 28, scope: !1459)
!1897 = !DILocation(line: 791, column: 2, scope: !1459)
!1898 = !DILocation(line: 792, column: 10, scope: !1459)
!1899 = !DILocation(line: 792, column: 48, scope: !1459)
!1900 = !DILocation(line: 792, column: 2, scope: !1459)
!1901 = !DILocation(line: 793, column: 9, scope: !1459)
!1902 = !DILocation(line: 793, column: 28, scope: !1459)
!1903 = !DILocation(line: 793, column: 2, scope: !1459)
!1904 = !DILocation(line: 794, column: 10, scope: !1459)
!1905 = !DILocation(line: 794, column: 48, scope: !1459)
!1906 = !DILocation(line: 794, column: 2, scope: !1459)
!1907 = !DILocation(line: 795, column: 9, scope: !1459)
!1908 = !DILocation(line: 795, column: 28, scope: !1459)
!1909 = !DILocation(line: 795, column: 2, scope: !1459)
!1910 = !DILocation(line: 796, column: 10, scope: !1459)
!1911 = !DILocation(line: 796, column: 47, scope: !1459)
!1912 = !DILocation(line: 796, column: 2, scope: !1459)
!1913 = !DILocation(line: 797, column: 9, scope: !1459)
!1914 = !DILocation(line: 797, column: 28, scope: !1459)
!1915 = !DILocation(line: 797, column: 2, scope: !1459)
!1916 = !DILocation(line: 798, column: 10, scope: !1459)
!1917 = !DILocation(line: 798, column: 46, scope: !1459)
!1918 = !DILocation(line: 798, column: 2, scope: !1459)
!1919 = !DILocation(line: 799, column: 9, scope: !1459)
!1920 = !DILocation(line: 799, column: 28, scope: !1459)
!1921 = !DILocation(line: 799, column: 2, scope: !1459)
!1922 = !DILocation(line: 800, column: 10, scope: !1459)
!1923 = !DILocation(line: 800, column: 49, scope: !1459)
!1924 = !DILocation(line: 800, column: 2, scope: !1459)
!1925 = !DILocation(line: 801, column: 9, scope: !1459)
!1926 = !DILocation(line: 801, column: 28, scope: !1459)
!1927 = !DILocation(line: 801, column: 2, scope: !1459)
!1928 = !DILocation(line: 805, column: 4, scope: !1459)
!1929 = !DILocation(line: 810, column: 4, scope: !1459)
!1930 = !DILocation(line: 811, column: 4, scope: !1459)
!1931 = !DILocation(line: 813, column: 4, scope: !1459)
!1932 = !DILocation(line: 820, column: 11, scope: !1459)
!1933 = !DILocation(line: 804, column: 2, scope: !1459)
!1934 = !DILocation(line: 829, column: 2, scope: !1459)
!1935 = !DILocation(line: 831, column: 2, scope: !1459)
!1936 = distinct !DISubprogram(name: "makea", linkageName: "_ZL5makeaiiPdPiS0_iiiiS0_PA12_iPA12_dS0_", scope: !3, file: !3, line: 1576, type: !1937, scopeLine: 1588, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1163)
!1937 = !DISubroutineType(types: !1938)
!1938 = !{null, !97, !97, !99, !98, !98, !97, !97, !97, !97, !98, !101, !106, !98}
!1939 = !DILocalVariable(name: "n", arg: 1, scope: !1936, file: !3, line: 1576, type: !97)
!1940 = !DILocation(line: 1576, column: 23, scope: !1936)
!1941 = !DILocalVariable(name: "nz", arg: 2, scope: !1936, file: !3, line: 1577, type: !97)
!1942 = !DILocation(line: 1577, column: 7, scope: !1936)
!1943 = !DILocalVariable(name: "a", arg: 3, scope: !1936, file: !3, line: 1578, type: !99)
!1944 = !DILocation(line: 1578, column: 10, scope: !1936)
!1945 = !DILocalVariable(name: "colidx", arg: 4, scope: !1936, file: !3, line: 1579, type: !98)
!1946 = !DILocation(line: 1579, column: 7, scope: !1936)
!1947 = !DILocalVariable(name: "rowstr", arg: 5, scope: !1936, file: !3, line: 1580, type: !98)
!1948 = !DILocation(line: 1580, column: 7, scope: !1936)
!1949 = !DILocalVariable(name: "firstrow", arg: 6, scope: !1936, file: !3, line: 1581, type: !97)
!1950 = !DILocation(line: 1581, column: 7, scope: !1936)
!1951 = !DILocalVariable(name: "lastrow", arg: 7, scope: !1936, file: !3, line: 1582, type: !97)
!1952 = !DILocation(line: 1582, column: 7, scope: !1936)
!1953 = !DILocalVariable(name: "firstcol", arg: 8, scope: !1936, file: !3, line: 1583, type: !97)
!1954 = !DILocation(line: 1583, column: 7, scope: !1936)
!1955 = !DILocalVariable(name: "lastcol", arg: 9, scope: !1936, file: !3, line: 1584, type: !97)
!1956 = !DILocation(line: 1584, column: 7, scope: !1936)
!1957 = !DILocalVariable(name: "arow", arg: 10, scope: !1936, file: !3, line: 1585, type: !98)
!1958 = !DILocation(line: 1585, column: 7, scope: !1936)
!1959 = !DILocalVariable(name: "acol", arg: 11, scope: !1936, file: !3, line: 1586, type: !101)
!1960 = !DILocation(line: 1586, column: 7, scope: !1936)
!1961 = !DILocalVariable(name: "aelt", arg: 12, scope: !1936, file: !3, line: 1587, type: !106)
!1962 = !DILocation(line: 1587, column: 10, scope: !1936)
!1963 = !DILocalVariable(name: "iv", arg: 13, scope: !1936, file: !3, line: 1588, type: !98)
!1964 = !DILocation(line: 1588, column: 7, scope: !1936)
!1965 = !DILocalVariable(name: "iouter", scope: !1936, file: !3, line: 1589, type: !97)
!1966 = !DILocation(line: 1589, column: 6, scope: !1936)
!1967 = !DILocalVariable(name: "ivelt", scope: !1936, file: !3, line: 1589, type: !97)
!1968 = !DILocation(line: 1589, column: 14, scope: !1936)
!1969 = !DILocalVariable(name: "nzv", scope: !1936, file: !3, line: 1589, type: !97)
!1970 = !DILocation(line: 1589, column: 21, scope: !1936)
!1971 = !DILocalVariable(name: "nn1", scope: !1936, file: !3, line: 1589, type: !97)
!1972 = !DILocation(line: 1589, column: 26, scope: !1936)
!1973 = !DILocalVariable(name: "ivc", scope: !1936, file: !3, line: 1590, type: !102)
!1974 = !DILocation(line: 1590, column: 6, scope: !1936)
!1975 = !DILocalVariable(name: "vc", scope: !1936, file: !3, line: 1591, type: !107)
!1976 = !DILocation(line: 1591, column: 9, scope: !1936)
!1977 = !DILocation(line: 1600, column: 6, scope: !1936)
!1978 = !DILocation(line: 1601, column: 2, scope: !1936)
!1979 = !DILocation(line: 1602, column: 13, scope: !1980)
!1980 = distinct !DILexicalBlock(scope: !1936, file: !3, line: 1601, column: 4)
!1981 = !DILocation(line: 1602, column: 11, scope: !1980)
!1982 = !DILocation(line: 1602, column: 7, scope: !1980)
!1983 = !DILocation(line: 1603, column: 2, scope: !1980)
!1984 = !DILocation(line: 1603, column: 9, scope: !1936)
!1985 = !DILocation(line: 1603, column: 15, scope: !1936)
!1986 = !DILocation(line: 1603, column: 13, scope: !1936)
!1987 = distinct !{!1987, !1978, !1988}
!1988 = !DILocation(line: 1603, column: 16, scope: !1936)
!1989 = !DILocation(line: 1610, column: 13, scope: !1990)
!1990 = distinct !DILexicalBlock(scope: !1936, file: !3, line: 1610, column: 2)
!1991 = !DILocation(line: 1610, column: 6, scope: !1990)
!1992 = !DILocation(line: 1610, column: 18, scope: !1993)
!1993 = distinct !DILexicalBlock(scope: !1990, file: !3, line: 1610, column: 2)
!1994 = !DILocation(line: 1610, column: 27, scope: !1993)
!1995 = !DILocation(line: 1610, column: 25, scope: !1993)
!1996 = !DILocation(line: 1610, column: 2, scope: !1990)
!1997 = !DILocation(line: 1611, column: 7, scope: !1998)
!1998 = distinct !DILexicalBlock(scope: !1993, file: !3, line: 1610, column: 39)
!1999 = !DILocation(line: 1612, column: 10, scope: !1998)
!2000 = !DILocation(line: 1612, column: 13, scope: !1998)
!2001 = !DILocation(line: 1612, column: 18, scope: !1998)
!2002 = !DILocation(line: 1612, column: 23, scope: !1998)
!2003 = !DILocation(line: 1612, column: 27, scope: !1998)
!2004 = !DILocation(line: 1612, column: 3, scope: !1998)
!2005 = !DILocation(line: 1613, column: 10, scope: !1998)
!2006 = !DILocation(line: 1613, column: 13, scope: !1998)
!2007 = !DILocation(line: 1613, column: 17, scope: !1998)
!2008 = !DILocation(line: 1613, column: 28, scope: !1998)
!2009 = !DILocation(line: 1613, column: 34, scope: !1998)
!2010 = !DILocation(line: 1613, column: 3, scope: !1998)
!2011 = !DILocation(line: 1614, column: 18, scope: !1998)
!2012 = !DILocation(line: 1614, column: 3, scope: !1998)
!2013 = !DILocation(line: 1614, column: 8, scope: !1998)
!2014 = !DILocation(line: 1614, column: 16, scope: !1998)
!2015 = !DILocation(line: 1615, column: 13, scope: !2016)
!2016 = distinct !DILexicalBlock(scope: !1998, file: !3, line: 1615, column: 3)
!2017 = !DILocation(line: 1615, column: 7, scope: !2016)
!2018 = !DILocation(line: 1615, column: 18, scope: !2019)
!2019 = distinct !DILexicalBlock(scope: !2016, file: !3, line: 1615, column: 3)
!2020 = !DILocation(line: 1615, column: 26, scope: !2019)
!2021 = !DILocation(line: 1615, column: 24, scope: !2019)
!2022 = !DILocation(line: 1615, column: 3, scope: !2016)
!2023 = !DILocation(line: 1616, column: 30, scope: !2024)
!2024 = distinct !DILexicalBlock(scope: !2019, file: !3, line: 1615, column: 39)
!2025 = !DILocation(line: 1616, column: 26, scope: !2024)
!2026 = !DILocation(line: 1616, column: 37, scope: !2024)
!2027 = !DILocation(line: 1616, column: 4, scope: !2024)
!2028 = !DILocation(line: 1616, column: 9, scope: !2024)
!2029 = !DILocation(line: 1616, column: 17, scope: !2024)
!2030 = !DILocation(line: 1616, column: 24, scope: !2024)
!2031 = !DILocation(line: 1617, column: 29, scope: !2024)
!2032 = !DILocation(line: 1617, column: 26, scope: !2024)
!2033 = !DILocation(line: 1617, column: 4, scope: !2024)
!2034 = !DILocation(line: 1617, column: 9, scope: !2024)
!2035 = !DILocation(line: 1617, column: 17, scope: !2024)
!2036 = !DILocation(line: 1617, column: 24, scope: !2024)
!2037 = !DILocation(line: 1618, column: 3, scope: !2024)
!2038 = !DILocation(line: 1615, column: 36, scope: !2019)
!2039 = !DILocation(line: 1615, column: 3, scope: !2019)
!2040 = distinct !{!2040, !2022, !2041}
!2041 = !DILocation(line: 1618, column: 3, scope: !2016)
!2042 = !DILocation(line: 1619, column: 2, scope: !1998)
!2043 = !DILocation(line: 1610, column: 36, scope: !1993)
!2044 = !DILocation(line: 1610, column: 2, scope: !1993)
!2045 = distinct !{!2045, !1996, !2046}
!2046 = !DILocation(line: 1619, column: 2, scope: !1990)
!2047 = !DILocation(line: 1627, column: 9, scope: !1936)
!2048 = !DILocation(line: 1628, column: 4, scope: !1936)
!2049 = !DILocation(line: 1629, column: 4, scope: !1936)
!2050 = !DILocation(line: 1630, column: 4, scope: !1936)
!2051 = !DILocation(line: 1631, column: 4, scope: !1936)
!2052 = !DILocation(line: 1633, column: 4, scope: !1936)
!2053 = !DILocation(line: 1634, column: 4, scope: !1936)
!2054 = !DILocation(line: 1635, column: 4, scope: !1936)
!2055 = !DILocation(line: 1636, column: 4, scope: !1936)
!2056 = !DILocation(line: 1637, column: 4, scope: !1936)
!2057 = !DILocation(line: 1638, column: 4, scope: !1936)
!2058 = !DILocation(line: 1627, column: 2, scope: !1936)
!2059 = !DILocation(line: 1641, column: 1, scope: !1936)
!2060 = distinct !DISubprogram(name: "conj_grad", linkageName: "_ZL9conj_gradPiS_PdS0_S0_S0_S0_S0_S0_", scope: !3, file: !3, line: 840, type: !2061, scopeLine: 848, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1163)
!2061 = !DISubroutineType(types: !2062)
!2062 = !{null, !98, !98, !99, !99, !99, !99, !99, !99, !99}
!2063 = !DILocalVariable(name: "colidx", arg: 1, scope: !2060, file: !3, line: 840, type: !98)
!2064 = !DILocation(line: 840, column: 27, scope: !2060)
!2065 = !DILocalVariable(name: "rowstr", arg: 2, scope: !2060, file: !3, line: 841, type: !98)
!2066 = !DILocation(line: 841, column: 7, scope: !2060)
!2067 = !DILocalVariable(name: "x", arg: 3, scope: !2060, file: !3, line: 842, type: !99)
!2068 = !DILocation(line: 842, column: 10, scope: !2060)
!2069 = !DILocalVariable(name: "z", arg: 4, scope: !2060, file: !3, line: 843, type: !99)
!2070 = !DILocation(line: 843, column: 10, scope: !2060)
!2071 = !DILocalVariable(name: "a", arg: 5, scope: !2060, file: !3, line: 844, type: !99)
!2072 = !DILocation(line: 844, column: 10, scope: !2060)
!2073 = !DILocalVariable(name: "p", arg: 6, scope: !2060, file: !3, line: 845, type: !99)
!2074 = !DILocation(line: 845, column: 10, scope: !2060)
!2075 = !DILocalVariable(name: "q", arg: 7, scope: !2060, file: !3, line: 846, type: !99)
!2076 = !DILocation(line: 846, column: 10, scope: !2060)
!2077 = !DILocalVariable(name: "r", arg: 8, scope: !2060, file: !3, line: 847, type: !99)
!2078 = !DILocation(line: 847, column: 10, scope: !2060)
!2079 = !DILocalVariable(name: "rnorm", arg: 9, scope: !2060, file: !3, line: 848, type: !99)
!2080 = !DILocation(line: 848, column: 11, scope: !2060)
!2081 = !DILocalVariable(name: "j", scope: !2060, file: !3, line: 849, type: !97)
!2082 = !DILocation(line: 849, column: 6, scope: !2060)
!2083 = !DILocalVariable(name: "k", scope: !2060, file: !3, line: 849, type: !97)
!2084 = !DILocation(line: 849, column: 9, scope: !2060)
!2085 = !DILocalVariable(name: "cgit", scope: !2060, file: !3, line: 850, type: !97)
!2086 = !DILocation(line: 850, column: 6, scope: !2060)
!2087 = !DILocalVariable(name: "cgitmax", scope: !2060, file: !3, line: 850, type: !97)
!2088 = !DILocation(line: 850, column: 12, scope: !2060)
!2089 = !DILocalVariable(name: "d", scope: !2060, file: !3, line: 851, type: !100)
!2090 = !DILocation(line: 851, column: 9, scope: !2060)
!2091 = !DILocalVariable(name: "sum", scope: !2060, file: !3, line: 851, type: !100)
!2092 = !DILocation(line: 851, column: 12, scope: !2060)
!2093 = !DILocalVariable(name: "rho", scope: !2060, file: !3, line: 851, type: !100)
!2094 = !DILocation(line: 851, column: 17, scope: !2060)
!2095 = !DILocalVariable(name: "rho0", scope: !2060, file: !3, line: 851, type: !100)
!2096 = !DILocation(line: 851, column: 22, scope: !2060)
!2097 = !DILocalVariable(name: "alpha", scope: !2060, file: !3, line: 851, type: !100)
!2098 = !DILocation(line: 851, column: 28, scope: !2060)
!2099 = !DILocalVariable(name: "beta", scope: !2060, file: !3, line: 851, type: !100)
!2100 = !DILocation(line: 851, column: 35, scope: !2060)
!2101 = !DILocation(line: 853, column: 10, scope: !2060)
!2102 = !DILocation(line: 855, column: 6, scope: !2060)
!2103 = !DILocation(line: 858, column: 8, scope: !2104)
!2104 = distinct !DILexicalBlock(scope: !2060, file: !3, line: 858, column: 2)
!2105 = !DILocation(line: 858, column: 6, scope: !2104)
!2106 = !DILocation(line: 858, column: 13, scope: !2107)
!2107 = distinct !DILexicalBlock(scope: !2104, file: !3, line: 858, column: 2)
!2108 = !DILocation(line: 858, column: 17, scope: !2107)
!2109 = !DILocation(line: 858, column: 20, scope: !2107)
!2110 = !DILocation(line: 858, column: 15, scope: !2107)
!2111 = !DILocation(line: 858, column: 2, scope: !2104)
!2112 = !DILocation(line: 859, column: 3, scope: !2113)
!2113 = distinct !DILexicalBlock(scope: !2107, file: !3, line: 858, column: 28)
!2114 = !DILocation(line: 859, column: 5, scope: !2113)
!2115 = !DILocation(line: 859, column: 8, scope: !2113)
!2116 = !DILocation(line: 860, column: 3, scope: !2113)
!2117 = !DILocation(line: 860, column: 5, scope: !2113)
!2118 = !DILocation(line: 860, column: 8, scope: !2113)
!2119 = !DILocation(line: 861, column: 10, scope: !2113)
!2120 = !DILocation(line: 861, column: 12, scope: !2113)
!2121 = !DILocation(line: 861, column: 3, scope: !2113)
!2122 = !DILocation(line: 861, column: 5, scope: !2113)
!2123 = !DILocation(line: 861, column: 8, scope: !2113)
!2124 = !DILocation(line: 862, column: 10, scope: !2113)
!2125 = !DILocation(line: 862, column: 12, scope: !2113)
!2126 = !DILocation(line: 862, column: 3, scope: !2113)
!2127 = !DILocation(line: 862, column: 5, scope: !2113)
!2128 = !DILocation(line: 862, column: 8, scope: !2113)
!2129 = !DILocation(line: 863, column: 2, scope: !2113)
!2130 = !DILocation(line: 858, column: 25, scope: !2107)
!2131 = !DILocation(line: 858, column: 2, scope: !2107)
!2132 = distinct !{!2132, !2111, !2133}
!2133 = !DILocation(line: 863, column: 2, scope: !2104)
!2134 = !DILocation(line: 871, column: 8, scope: !2135)
!2135 = distinct !DILexicalBlock(scope: !2060, file: !3, line: 871, column: 2)
!2136 = !DILocation(line: 871, column: 6, scope: !2135)
!2137 = !DILocation(line: 871, column: 13, scope: !2138)
!2138 = distinct !DILexicalBlock(scope: !2135, file: !3, line: 871, column: 2)
!2139 = !DILocation(line: 871, column: 17, scope: !2138)
!2140 = !DILocation(line: 871, column: 27, scope: !2138)
!2141 = !DILocation(line: 871, column: 25, scope: !2138)
!2142 = !DILocation(line: 871, column: 36, scope: !2138)
!2143 = !DILocation(line: 871, column: 15, scope: !2138)
!2144 = !DILocation(line: 871, column: 2, scope: !2135)
!2145 = !DILocation(line: 872, column: 9, scope: !2146)
!2146 = distinct !DILexicalBlock(scope: !2138, file: !3, line: 871, column: 45)
!2147 = !DILocation(line: 872, column: 15, scope: !2146)
!2148 = !DILocation(line: 872, column: 17, scope: !2146)
!2149 = !DILocation(line: 872, column: 20, scope: !2146)
!2150 = !DILocation(line: 872, column: 22, scope: !2146)
!2151 = !DILocation(line: 872, column: 19, scope: !2146)
!2152 = !DILocation(line: 872, column: 13, scope: !2146)
!2153 = !DILocation(line: 872, column: 7, scope: !2146)
!2154 = !DILocation(line: 873, column: 2, scope: !2146)
!2155 = !DILocation(line: 871, column: 42, scope: !2138)
!2156 = !DILocation(line: 871, column: 2, scope: !2138)
!2157 = distinct !{!2157, !2144, !2158}
!2158 = !DILocation(line: 873, column: 2, scope: !2135)
!2159 = !DILocation(line: 876, column: 11, scope: !2160)
!2160 = distinct !DILexicalBlock(scope: !2060, file: !3, line: 876, column: 2)
!2161 = !DILocation(line: 876, column: 6, scope: !2160)
!2162 = !DILocation(line: 876, column: 16, scope: !2163)
!2163 = distinct !DILexicalBlock(scope: !2160, file: !3, line: 876, column: 2)
!2164 = !DILocation(line: 876, column: 24, scope: !2163)
!2165 = !DILocation(line: 876, column: 21, scope: !2163)
!2166 = !DILocation(line: 876, column: 2, scope: !2160)
!2167 = !DILocation(line: 890, column: 9, scope: !2168)
!2168 = distinct !DILexicalBlock(scope: !2169, file: !3, line: 890, column: 3)
!2169 = distinct !DILexicalBlock(scope: !2163, file: !3, line: 876, column: 40)
!2170 = !DILocation(line: 890, column: 7, scope: !2168)
!2171 = !DILocation(line: 890, column: 14, scope: !2172)
!2172 = distinct !DILexicalBlock(scope: !2168, file: !3, line: 890, column: 3)
!2173 = !DILocation(line: 890, column: 18, scope: !2172)
!2174 = !DILocation(line: 890, column: 28, scope: !2172)
!2175 = !DILocation(line: 890, column: 26, scope: !2172)
!2176 = !DILocation(line: 890, column: 37, scope: !2172)
!2177 = !DILocation(line: 890, column: 16, scope: !2172)
!2178 = !DILocation(line: 890, column: 3, scope: !2168)
!2179 = !DILocation(line: 891, column: 8, scope: !2180)
!2180 = distinct !DILexicalBlock(scope: !2172, file: !3, line: 890, column: 46)
!2181 = !DILocation(line: 892, column: 12, scope: !2182)
!2182 = distinct !DILexicalBlock(scope: !2180, file: !3, line: 892, column: 4)
!2183 = !DILocation(line: 892, column: 19, scope: !2182)
!2184 = !DILocation(line: 892, column: 10, scope: !2182)
!2185 = !DILocation(line: 892, column: 8, scope: !2182)
!2186 = !DILocation(line: 892, column: 23, scope: !2187)
!2187 = distinct !DILexicalBlock(scope: !2182, file: !3, line: 892, column: 4)
!2188 = !DILocation(line: 892, column: 27, scope: !2187)
!2189 = !DILocation(line: 892, column: 34, scope: !2187)
!2190 = !DILocation(line: 892, column: 35, scope: !2187)
!2191 = !DILocation(line: 892, column: 25, scope: !2187)
!2192 = !DILocation(line: 892, column: 4, scope: !2182)
!2193 = !DILocation(line: 893, column: 11, scope: !2194)
!2194 = distinct !DILexicalBlock(scope: !2187, file: !3, line: 892, column: 44)
!2195 = !DILocation(line: 893, column: 17, scope: !2194)
!2196 = !DILocation(line: 893, column: 19, scope: !2194)
!2197 = !DILocation(line: 893, column: 22, scope: !2194)
!2198 = !DILocation(line: 893, column: 24, scope: !2194)
!2199 = !DILocation(line: 893, column: 31, scope: !2194)
!2200 = !DILocation(line: 893, column: 21, scope: !2194)
!2201 = !DILocation(line: 893, column: 15, scope: !2194)
!2202 = !DILocation(line: 893, column: 9, scope: !2194)
!2203 = !DILocation(line: 894, column: 4, scope: !2194)
!2204 = !DILocation(line: 892, column: 41, scope: !2187)
!2205 = !DILocation(line: 892, column: 4, scope: !2187)
!2206 = distinct !{!2206, !2192, !2207}
!2207 = !DILocation(line: 894, column: 4, scope: !2182)
!2208 = !DILocation(line: 895, column: 11, scope: !2180)
!2209 = !DILocation(line: 895, column: 4, scope: !2180)
!2210 = !DILocation(line: 895, column: 6, scope: !2180)
!2211 = !DILocation(line: 895, column: 9, scope: !2180)
!2212 = !DILocation(line: 896, column: 3, scope: !2180)
!2213 = !DILocation(line: 890, column: 43, scope: !2172)
!2214 = !DILocation(line: 890, column: 3, scope: !2172)
!2215 = distinct !{!2215, !2178, !2216}
!2216 = !DILocation(line: 896, column: 3, scope: !2168)
!2217 = !DILocation(line: 903, column: 5, scope: !2169)
!2218 = !DILocation(line: 904, column: 10, scope: !2219)
!2219 = distinct !DILexicalBlock(scope: !2169, file: !3, line: 904, column: 3)
!2220 = !DILocation(line: 904, column: 8, scope: !2219)
!2221 = !DILocation(line: 904, column: 15, scope: !2222)
!2222 = distinct !DILexicalBlock(scope: !2219, file: !3, line: 904, column: 3)
!2223 = !DILocation(line: 904, column: 19, scope: !2222)
!2224 = !DILocation(line: 904, column: 29, scope: !2222)
!2225 = !DILocation(line: 904, column: 27, scope: !2222)
!2226 = !DILocation(line: 904, column: 38, scope: !2222)
!2227 = !DILocation(line: 904, column: 17, scope: !2222)
!2228 = !DILocation(line: 904, column: 3, scope: !2219)
!2229 = !DILocation(line: 905, column: 8, scope: !2230)
!2230 = distinct !DILexicalBlock(scope: !2222, file: !3, line: 904, column: 48)
!2231 = !DILocation(line: 905, column: 12, scope: !2230)
!2232 = !DILocation(line: 905, column: 14, scope: !2230)
!2233 = !DILocation(line: 905, column: 17, scope: !2230)
!2234 = !DILocation(line: 905, column: 19, scope: !2230)
!2235 = !DILocation(line: 905, column: 16, scope: !2230)
!2236 = !DILocation(line: 905, column: 10, scope: !2230)
!2237 = !DILocation(line: 905, column: 6, scope: !2230)
!2238 = !DILocation(line: 906, column: 3, scope: !2230)
!2239 = !DILocation(line: 904, column: 44, scope: !2222)
!2240 = !DILocation(line: 904, column: 3, scope: !2222)
!2241 = distinct !{!2241, !2228, !2242}
!2242 = !DILocation(line: 906, column: 3, scope: !2219)
!2243 = !DILocation(line: 913, column: 11, scope: !2169)
!2244 = !DILocation(line: 913, column: 17, scope: !2169)
!2245 = !DILocation(line: 913, column: 15, scope: !2169)
!2246 = !DILocation(line: 913, column: 9, scope: !2169)
!2247 = !DILocation(line: 920, column: 10, scope: !2169)
!2248 = !DILocation(line: 920, column: 8, scope: !2169)
!2249 = !DILocation(line: 928, column: 7, scope: !2169)
!2250 = !DILocation(line: 929, column: 9, scope: !2251)
!2251 = distinct !DILexicalBlock(scope: !2169, file: !3, line: 929, column: 3)
!2252 = !DILocation(line: 929, column: 7, scope: !2251)
!2253 = !DILocation(line: 929, column: 14, scope: !2254)
!2254 = distinct !DILexicalBlock(scope: !2251, file: !3, line: 929, column: 3)
!2255 = !DILocation(line: 929, column: 18, scope: !2254)
!2256 = !DILocation(line: 929, column: 28, scope: !2254)
!2257 = !DILocation(line: 929, column: 26, scope: !2254)
!2258 = !DILocation(line: 929, column: 37, scope: !2254)
!2259 = !DILocation(line: 929, column: 16, scope: !2254)
!2260 = !DILocation(line: 929, column: 3, scope: !2251)
!2261 = !DILocation(line: 930, column: 11, scope: !2262)
!2262 = distinct !DILexicalBlock(scope: !2254, file: !3, line: 929, column: 46)
!2263 = !DILocation(line: 930, column: 13, scope: !2262)
!2264 = !DILocation(line: 930, column: 18, scope: !2262)
!2265 = !DILocation(line: 930, column: 24, scope: !2262)
!2266 = !DILocation(line: 930, column: 26, scope: !2262)
!2267 = !DILocation(line: 930, column: 23, scope: !2262)
!2268 = !DILocation(line: 930, column: 16, scope: !2262)
!2269 = !DILocation(line: 930, column: 4, scope: !2262)
!2270 = !DILocation(line: 930, column: 6, scope: !2262)
!2271 = !DILocation(line: 930, column: 9, scope: !2262)
!2272 = !DILocation(line: 931, column: 11, scope: !2262)
!2273 = !DILocation(line: 931, column: 13, scope: !2262)
!2274 = !DILocation(line: 931, column: 18, scope: !2262)
!2275 = !DILocation(line: 931, column: 24, scope: !2262)
!2276 = !DILocation(line: 931, column: 26, scope: !2262)
!2277 = !DILocation(line: 931, column: 23, scope: !2262)
!2278 = !DILocation(line: 931, column: 16, scope: !2262)
!2279 = !DILocation(line: 931, column: 4, scope: !2262)
!2280 = !DILocation(line: 931, column: 6, scope: !2262)
!2281 = !DILocation(line: 931, column: 9, scope: !2262)
!2282 = !DILocation(line: 932, column: 3, scope: !2262)
!2283 = !DILocation(line: 929, column: 43, scope: !2254)
!2284 = !DILocation(line: 929, column: 3, scope: !2254)
!2285 = distinct !{!2285, !2260, !2286}
!2286 = !DILocation(line: 932, column: 3, scope: !2251)
!2287 = !DILocation(line: 940, column: 9, scope: !2288)
!2288 = distinct !DILexicalBlock(scope: !2169, file: !3, line: 940, column: 3)
!2289 = !DILocation(line: 940, column: 7, scope: !2288)
!2290 = !DILocation(line: 940, column: 14, scope: !2291)
!2291 = distinct !DILexicalBlock(scope: !2288, file: !3, line: 940, column: 3)
!2292 = !DILocation(line: 940, column: 18, scope: !2291)
!2293 = !DILocation(line: 940, column: 28, scope: !2291)
!2294 = !DILocation(line: 940, column: 26, scope: !2291)
!2295 = !DILocation(line: 940, column: 37, scope: !2291)
!2296 = !DILocation(line: 940, column: 16, scope: !2291)
!2297 = !DILocation(line: 940, column: 3, scope: !2288)
!2298 = !DILocation(line: 941, column: 10, scope: !2299)
!2299 = distinct !DILexicalBlock(scope: !2291, file: !3, line: 940, column: 46)
!2300 = !DILocation(line: 941, column: 16, scope: !2299)
!2301 = !DILocation(line: 941, column: 18, scope: !2299)
!2302 = !DILocation(line: 941, column: 21, scope: !2299)
!2303 = !DILocation(line: 941, column: 23, scope: !2299)
!2304 = !DILocation(line: 941, column: 20, scope: !2299)
!2305 = !DILocation(line: 941, column: 14, scope: !2299)
!2306 = !DILocation(line: 941, column: 8, scope: !2299)
!2307 = !DILocation(line: 942, column: 3, scope: !2299)
!2308 = !DILocation(line: 940, column: 43, scope: !2291)
!2309 = !DILocation(line: 940, column: 3, scope: !2291)
!2310 = distinct !{!2310, !2297, !2311}
!2311 = !DILocation(line: 942, column: 3, scope: !2288)
!2312 = !DILocation(line: 949, column: 10, scope: !2169)
!2313 = !DILocation(line: 949, column: 16, scope: !2169)
!2314 = !DILocation(line: 949, column: 14, scope: !2169)
!2315 = !DILocation(line: 949, column: 8, scope: !2169)
!2316 = !DILocation(line: 956, column: 9, scope: !2317)
!2317 = distinct !DILexicalBlock(scope: !2169, file: !3, line: 956, column: 3)
!2318 = !DILocation(line: 956, column: 7, scope: !2317)
!2319 = !DILocation(line: 956, column: 14, scope: !2320)
!2320 = distinct !DILexicalBlock(scope: !2317, file: !3, line: 956, column: 3)
!2321 = !DILocation(line: 956, column: 18, scope: !2320)
!2322 = !DILocation(line: 956, column: 28, scope: !2320)
!2323 = !DILocation(line: 956, column: 26, scope: !2320)
!2324 = !DILocation(line: 956, column: 37, scope: !2320)
!2325 = !DILocation(line: 956, column: 16, scope: !2320)
!2326 = !DILocation(line: 956, column: 3, scope: !2317)
!2327 = !DILocation(line: 957, column: 11, scope: !2328)
!2328 = distinct !DILexicalBlock(scope: !2320, file: !3, line: 956, column: 46)
!2329 = !DILocation(line: 957, column: 13, scope: !2328)
!2330 = !DILocation(line: 957, column: 18, scope: !2328)
!2331 = !DILocation(line: 957, column: 23, scope: !2328)
!2332 = !DILocation(line: 957, column: 25, scope: !2328)
!2333 = !DILocation(line: 957, column: 22, scope: !2328)
!2334 = !DILocation(line: 957, column: 16, scope: !2328)
!2335 = !DILocation(line: 957, column: 4, scope: !2328)
!2336 = !DILocation(line: 957, column: 6, scope: !2328)
!2337 = !DILocation(line: 957, column: 9, scope: !2328)
!2338 = !DILocation(line: 958, column: 3, scope: !2328)
!2339 = !DILocation(line: 956, column: 43, scope: !2320)
!2340 = !DILocation(line: 956, column: 3, scope: !2320)
!2341 = distinct !{!2341, !2326, !2342}
!2342 = !DILocation(line: 958, column: 3, scope: !2317)
!2343 = !DILocation(line: 959, column: 2, scope: !2169)
!2344 = !DILocation(line: 876, column: 37, scope: !2163)
!2345 = !DILocation(line: 876, column: 2, scope: !2163)
!2346 = distinct !{!2346, !2166, !2347}
!2347 = !DILocation(line: 959, column: 2, scope: !2160)
!2348 = !DILocation(line: 968, column: 6, scope: !2060)
!2349 = !DILocation(line: 969, column: 8, scope: !2350)
!2350 = distinct !DILexicalBlock(scope: !2060, file: !3, line: 969, column: 2)
!2351 = !DILocation(line: 969, column: 6, scope: !2350)
!2352 = !DILocation(line: 969, column: 13, scope: !2353)
!2353 = distinct !DILexicalBlock(scope: !2350, file: !3, line: 969, column: 2)
!2354 = !DILocation(line: 969, column: 17, scope: !2353)
!2355 = !DILocation(line: 969, column: 27, scope: !2353)
!2356 = !DILocation(line: 969, column: 25, scope: !2353)
!2357 = !DILocation(line: 969, column: 36, scope: !2353)
!2358 = !DILocation(line: 969, column: 15, scope: !2353)
!2359 = !DILocation(line: 969, column: 2, scope: !2350)
!2360 = !DILocation(line: 970, column: 5, scope: !2361)
!2361 = distinct !DILexicalBlock(scope: !2353, file: !3, line: 969, column: 45)
!2362 = !DILocation(line: 971, column: 11, scope: !2363)
!2363 = distinct !DILexicalBlock(scope: !2361, file: !3, line: 971, column: 3)
!2364 = !DILocation(line: 971, column: 18, scope: !2363)
!2365 = !DILocation(line: 971, column: 9, scope: !2363)
!2366 = !DILocation(line: 971, column: 7, scope: !2363)
!2367 = !DILocation(line: 971, column: 22, scope: !2368)
!2368 = distinct !DILexicalBlock(scope: !2363, file: !3, line: 971, column: 3)
!2369 = !DILocation(line: 971, column: 26, scope: !2368)
!2370 = !DILocation(line: 971, column: 33, scope: !2368)
!2371 = !DILocation(line: 971, column: 34, scope: !2368)
!2372 = !DILocation(line: 971, column: 24, scope: !2368)
!2373 = !DILocation(line: 971, column: 3, scope: !2363)
!2374 = !DILocation(line: 972, column: 8, scope: !2375)
!2375 = distinct !DILexicalBlock(scope: !2368, file: !3, line: 971, column: 43)
!2376 = !DILocation(line: 972, column: 12, scope: !2375)
!2377 = !DILocation(line: 972, column: 14, scope: !2375)
!2378 = !DILocation(line: 972, column: 17, scope: !2375)
!2379 = !DILocation(line: 972, column: 19, scope: !2375)
!2380 = !DILocation(line: 972, column: 26, scope: !2375)
!2381 = !DILocation(line: 972, column: 16, scope: !2375)
!2382 = !DILocation(line: 972, column: 10, scope: !2375)
!2383 = !DILocation(line: 972, column: 6, scope: !2375)
!2384 = !DILocation(line: 973, column: 3, scope: !2375)
!2385 = !DILocation(line: 971, column: 40, scope: !2368)
!2386 = !DILocation(line: 971, column: 3, scope: !2368)
!2387 = distinct !{!2387, !2373, !2388}
!2388 = !DILocation(line: 973, column: 3, scope: !2363)
!2389 = !DILocation(line: 974, column: 10, scope: !2361)
!2390 = !DILocation(line: 974, column: 3, scope: !2361)
!2391 = !DILocation(line: 974, column: 5, scope: !2361)
!2392 = !DILocation(line: 974, column: 8, scope: !2361)
!2393 = !DILocation(line: 975, column: 2, scope: !2361)
!2394 = !DILocation(line: 969, column: 42, scope: !2353)
!2395 = !DILocation(line: 969, column: 2, scope: !2353)
!2396 = distinct !{!2396, !2359, !2397}
!2397 = !DILocation(line: 975, column: 2, scope: !2350)
!2398 = !DILocation(line: 982, column: 8, scope: !2399)
!2399 = distinct !DILexicalBlock(scope: !2060, file: !3, line: 982, column: 2)
!2400 = !DILocation(line: 982, column: 6, scope: !2399)
!2401 = !DILocation(line: 982, column: 13, scope: !2402)
!2402 = distinct !DILexicalBlock(scope: !2399, file: !3, line: 982, column: 2)
!2403 = !DILocation(line: 982, column: 17, scope: !2402)
!2404 = !DILocation(line: 982, column: 25, scope: !2402)
!2405 = !DILocation(line: 982, column: 24, scope: !2402)
!2406 = !DILocation(line: 982, column: 33, scope: !2402)
!2407 = !DILocation(line: 982, column: 15, scope: !2402)
!2408 = !DILocation(line: 982, column: 2, scope: !2399)
!2409 = !DILocation(line: 983, column: 9, scope: !2410)
!2410 = distinct !DILexicalBlock(scope: !2402, file: !3, line: 982, column: 41)
!2411 = !DILocation(line: 983, column: 11, scope: !2410)
!2412 = !DILocation(line: 983, column: 16, scope: !2410)
!2413 = !DILocation(line: 983, column: 18, scope: !2410)
!2414 = !DILocation(line: 983, column: 14, scope: !2410)
!2415 = !DILocation(line: 983, column: 7, scope: !2410)
!2416 = !DILocation(line: 984, column: 9, scope: !2410)
!2417 = !DILocation(line: 984, column: 15, scope: !2410)
!2418 = !DILocation(line: 984, column: 17, scope: !2410)
!2419 = !DILocation(line: 984, column: 16, scope: !2410)
!2420 = !DILocation(line: 984, column: 13, scope: !2410)
!2421 = !DILocation(line: 984, column: 7, scope: !2410)
!2422 = !DILocation(line: 985, column: 2, scope: !2410)
!2423 = !DILocation(line: 982, column: 38, scope: !2402)
!2424 = !DILocation(line: 982, column: 2, scope: !2402)
!2425 = distinct !{!2425, !2408, !2426}
!2426 = !DILocation(line: 985, column: 2, scope: !2399)
!2427 = !DILocation(line: 987, column: 16, scope: !2060)
!2428 = !DILocation(line: 987, column: 11, scope: !2060)
!2429 = !DILocation(line: 987, column: 3, scope: !2060)
!2430 = !DILocation(line: 987, column: 9, scope: !2060)
!2431 = !DILocation(line: 988, column: 1, scope: !2060)
!2432 = distinct !DISubprogram(name: "setup_gpu", linkageName: "_ZL9setup_gpuv", scope: !3, file: !3, line: 1663, type: !666, scopeLine: 1663, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1163)
!2433 = !DILocation(line: 1709, column: 33, scope: !2432)
!2434 = !DILocation(line: 1710, column: 43, scope: !2432)
!2435 = !DILocation(line: 1713, column: 63, scope: !2436)
!2436 = distinct !DILexicalBlock(scope: !2432, file: !3, line: 1712, column: 5)
!2437 = !DILocation(line: 1713, column: 39, scope: !2436)
!2438 = !DILocation(line: 1712, column: 5, scope: !2432)
!2439 = !DILocation(line: 1714, column: 35, scope: !2440)
!2440 = distinct !DILexicalBlock(scope: !2436, file: !3, line: 1713, column: 83)
!2441 = !DILocation(line: 1715, column: 2, scope: !2440)
!2442 = !DILocation(line: 1717, column: 59, scope: !2443)
!2443 = distinct !DILexicalBlock(scope: !2436, file: !3, line: 1716, column: 6)
!2444 = !DILocation(line: 1717, column: 35, scope: !2443)
!2445 = !DILocation(line: 1720, column: 63, scope: !2446)
!2446 = distinct !DILexicalBlock(scope: !2432, file: !3, line: 1719, column: 5)
!2447 = !DILocation(line: 1720, column: 39, scope: !2446)
!2448 = !DILocation(line: 1719, column: 5, scope: !2432)
!2449 = !DILocation(line: 1721, column: 35, scope: !2450)
!2450 = distinct !DILexicalBlock(scope: !2446, file: !3, line: 1720, column: 83)
!2451 = !DILocation(line: 1722, column: 2, scope: !2450)
!2452 = !DILocation(line: 1724, column: 59, scope: !2453)
!2453 = distinct !DILexicalBlock(scope: !2446, file: !3, line: 1723, column: 6)
!2454 = !DILocation(line: 1724, column: 35, scope: !2453)
!2455 = !DILocation(line: 1727, column: 65, scope: !2456)
!2456 = distinct !DILexicalBlock(scope: !2432, file: !3, line: 1726, column: 5)
!2457 = !DILocation(line: 1727, column: 41, scope: !2456)
!2458 = !DILocation(line: 1726, column: 5, scope: !2432)
!2459 = !DILocation(line: 1728, column: 37, scope: !2460)
!2460 = distinct !DILexicalBlock(scope: !2456, file: !3, line: 1727, column: 85)
!2461 = !DILocation(line: 1729, column: 2, scope: !2460)
!2462 = !DILocation(line: 1731, column: 61, scope: !2463)
!2463 = distinct !DILexicalBlock(scope: !2456, file: !3, line: 1730, column: 6)
!2464 = !DILocation(line: 1731, column: 37, scope: !2463)
!2465 = !DILocation(line: 1734, column: 64, scope: !2466)
!2466 = distinct !DILexicalBlock(scope: !2432, file: !3, line: 1733, column: 5)
!2467 = !DILocation(line: 1734, column: 40, scope: !2466)
!2468 = !DILocation(line: 1733, column: 5, scope: !2432)
!2469 = !DILocation(line: 1735, column: 36, scope: !2470)
!2470 = distinct !DILexicalBlock(scope: !2466, file: !3, line: 1734, column: 84)
!2471 = !DILocation(line: 1736, column: 2, scope: !2470)
!2472 = !DILocation(line: 1738, column: 60, scope: !2473)
!2473 = distinct !DILexicalBlock(scope: !2466, file: !3, line: 1737, column: 6)
!2474 = !DILocation(line: 1738, column: 36, scope: !2473)
!2475 = !DILocation(line: 1741, column: 64, scope: !2476)
!2476 = distinct !DILexicalBlock(scope: !2432, file: !3, line: 1740, column: 5)
!2477 = !DILocation(line: 1741, column: 40, scope: !2476)
!2478 = !DILocation(line: 1740, column: 5, scope: !2432)
!2479 = !DILocation(line: 1742, column: 36, scope: !2480)
!2480 = distinct !DILexicalBlock(scope: !2476, file: !3, line: 1741, column: 84)
!2481 = !DILocation(line: 1743, column: 2, scope: !2480)
!2482 = !DILocation(line: 1745, column: 60, scope: !2483)
!2483 = distinct !DILexicalBlock(scope: !2476, file: !3, line: 1744, column: 6)
!2484 = !DILocation(line: 1745, column: 36, scope: !2483)
!2485 = !DILocation(line: 1748, column: 63, scope: !2486)
!2486 = distinct !DILexicalBlock(scope: !2432, file: !3, line: 1747, column: 5)
!2487 = !DILocation(line: 1748, column: 39, scope: !2486)
!2488 = !DILocation(line: 1747, column: 5, scope: !2432)
!2489 = !DILocation(line: 1749, column: 35, scope: !2490)
!2490 = distinct !DILexicalBlock(scope: !2486, file: !3, line: 1748, column: 83)
!2491 = !DILocation(line: 1750, column: 2, scope: !2490)
!2492 = !DILocation(line: 1752, column: 59, scope: !2493)
!2493 = distinct !DILexicalBlock(scope: !2486, file: !3, line: 1751, column: 6)
!2494 = !DILocation(line: 1752, column: 35, scope: !2493)
!2495 = !DILocation(line: 1755, column: 65, scope: !2496)
!2496 = distinct !DILexicalBlock(scope: !2432, file: !3, line: 1754, column: 5)
!2497 = !DILocation(line: 1755, column: 41, scope: !2496)
!2498 = !DILocation(line: 1754, column: 5, scope: !2432)
!2499 = !DILocation(line: 1756, column: 37, scope: !2500)
!2500 = distinct !DILexicalBlock(scope: !2496, file: !3, line: 1755, column: 85)
!2501 = !DILocation(line: 1757, column: 2, scope: !2500)
!2502 = !DILocation(line: 1759, column: 61, scope: !2503)
!2503 = distinct !DILexicalBlock(scope: !2496, file: !3, line: 1758, column: 6)
!2504 = !DILocation(line: 1759, column: 37, scope: !2503)
!2505 = !DILocation(line: 1762, column: 65, scope: !2506)
!2506 = distinct !DILexicalBlock(scope: !2432, file: !3, line: 1761, column: 5)
!2507 = !DILocation(line: 1762, column: 41, scope: !2506)
!2508 = !DILocation(line: 1761, column: 5, scope: !2432)
!2509 = !DILocation(line: 1763, column: 37, scope: !2510)
!2510 = distinct !DILexicalBlock(scope: !2506, file: !3, line: 1762, column: 85)
!2511 = !DILocation(line: 1764, column: 2, scope: !2510)
!2512 = !DILocation(line: 1766, column: 61, scope: !2513)
!2513 = distinct !DILexicalBlock(scope: !2506, file: !3, line: 1765, column: 6)
!2514 = !DILocation(line: 1766, column: 37, scope: !2513)
!2515 = !DILocation(line: 1769, column: 64, scope: !2516)
!2516 = distinct !DILexicalBlock(scope: !2432, file: !3, line: 1768, column: 5)
!2517 = !DILocation(line: 1769, column: 40, scope: !2516)
!2518 = !DILocation(line: 1768, column: 5, scope: !2432)
!2519 = !DILocation(line: 1770, column: 36, scope: !2520)
!2520 = distinct !DILexicalBlock(scope: !2516, file: !3, line: 1769, column: 84)
!2521 = !DILocation(line: 1771, column: 2, scope: !2520)
!2522 = !DILocation(line: 1773, column: 58, scope: !2523)
!2523 = distinct !DILexicalBlock(scope: !2516, file: !3, line: 1772, column: 6)
!2524 = !DILocation(line: 1773, column: 35, scope: !2523)
!2525 = !DILocation(line: 1776, column: 63, scope: !2526)
!2526 = distinct !DILexicalBlock(scope: !2432, file: !3, line: 1775, column: 5)
!2527 = !DILocation(line: 1776, column: 39, scope: !2526)
!2528 = !DILocation(line: 1775, column: 5, scope: !2432)
!2529 = !DILocation(line: 1777, column: 35, scope: !2530)
!2530 = distinct !DILexicalBlock(scope: !2526, file: !3, line: 1776, column: 83)
!2531 = !DILocation(line: 1778, column: 2, scope: !2530)
!2532 = !DILocation(line: 1780, column: 57, scope: !2533)
!2533 = distinct !DILexicalBlock(scope: !2526, file: !3, line: 1779, column: 6)
!2534 = !DILocation(line: 1780, column: 34, scope: !2533)
!2535 = !DILocation(line: 1783, column: 66, scope: !2536)
!2536 = distinct !DILexicalBlock(scope: !2432, file: !3, line: 1782, column: 5)
!2537 = !DILocation(line: 1783, column: 42, scope: !2536)
!2538 = !DILocation(line: 1782, column: 5, scope: !2432)
!2539 = !DILocation(line: 1784, column: 38, scope: !2540)
!2540 = distinct !DILexicalBlock(scope: !2536, file: !3, line: 1783, column: 86)
!2541 = !DILocation(line: 1785, column: 2, scope: !2540)
!2542 = !DILocation(line: 1787, column: 62, scope: !2543)
!2543 = distinct !DILexicalBlock(scope: !2536, file: !3, line: 1786, column: 6)
!2544 = !DILocation(line: 1787, column: 38, scope: !2543)
!2545 = !DILocation(line: 1790, column: 57, scope: !2432)
!2546 = !DILocation(line: 1790, column: 48, scope: !2432)
!2547 = !DILocation(line: 1790, column: 33, scope: !2432)
!2548 = !DILocation(line: 1790, column: 32, scope: !2432)
!2549 = !DILocation(line: 1790, column: 31, scope: !2432)
!2550 = !DILocation(line: 1791, column: 57, scope: !2432)
!2551 = !DILocation(line: 1791, column: 48, scope: !2432)
!2552 = !DILocation(line: 1791, column: 33, scope: !2432)
!2553 = !DILocation(line: 1791, column: 32, scope: !2432)
!2554 = !DILocation(line: 1791, column: 31, scope: !2432)
!2555 = !DILocation(line: 1792, column: 33, scope: !2432)
!2556 = !DILocation(line: 1793, column: 58, scope: !2432)
!2557 = !DILocation(line: 1793, column: 49, scope: !2432)
!2558 = !DILocation(line: 1793, column: 34, scope: !2432)
!2559 = !DILocation(line: 1793, column: 33, scope: !2432)
!2560 = !DILocation(line: 1793, column: 32, scope: !2432)
!2561 = !DILocation(line: 1794, column: 58, scope: !2432)
!2562 = !DILocation(line: 1794, column: 49, scope: !2432)
!2563 = !DILocation(line: 1794, column: 34, scope: !2432)
!2564 = !DILocation(line: 1794, column: 33, scope: !2432)
!2565 = !DILocation(line: 1794, column: 32, scope: !2432)
!2566 = !DILocation(line: 1795, column: 57, scope: !2432)
!2567 = !DILocation(line: 1795, column: 48, scope: !2432)
!2568 = !DILocation(line: 1795, column: 33, scope: !2432)
!2569 = !DILocation(line: 1795, column: 32, scope: !2432)
!2570 = !DILocation(line: 1795, column: 31, scope: !2432)
!2571 = !DILocation(line: 1796, column: 51, scope: !2432)
!2572 = !DILocation(line: 1796, column: 50, scope: !2432)
!2573 = !DILocation(line: 1796, column: 35, scope: !2432)
!2574 = !DILocation(line: 1796, column: 34, scope: !2432)
!2575 = !DILocation(line: 1796, column: 33, scope: !2432)
!2576 = !DILocation(line: 1797, column: 33, scope: !2432)
!2577 = !DILocation(line: 1798, column: 58, scope: !2432)
!2578 = !DILocation(line: 1798, column: 49, scope: !2432)
!2579 = !DILocation(line: 1798, column: 34, scope: !2432)
!2580 = !DILocation(line: 1798, column: 33, scope: !2432)
!2581 = !DILocation(line: 1798, column: 32, scope: !2432)
!2582 = !DILocation(line: 1799, column: 57, scope: !2432)
!2583 = !DILocation(line: 1799, column: 48, scope: !2432)
!2584 = !DILocation(line: 1799, column: 33, scope: !2432)
!2585 = !DILocation(line: 1799, column: 32, scope: !2432)
!2586 = !DILocation(line: 1799, column: 31, scope: !2432)
!2587 = !DILocation(line: 1800, column: 60, scope: !2432)
!2588 = !DILocation(line: 1800, column: 51, scope: !2432)
!2589 = !DILocation(line: 1800, column: 36, scope: !2432)
!2590 = !DILocation(line: 1800, column: 35, scope: !2432)
!2591 = !DILocation(line: 1800, column: 34, scope: !2432)
!2592 = !DILocation(line: 1802, column: 68, scope: !2432)
!2593 = !DILocation(line: 1802, column: 46, scope: !2432)
!2594 = !DILocation(line: 1802, column: 38, scope: !2432)
!2595 = !DILocation(line: 1802, column: 23, scope: !2432)
!2596 = !DILocation(line: 1802, column: 22, scope: !2432)
!2597 = !DILocation(line: 1804, column: 19, scope: !2432)
!2598 = !DILocation(line: 1804, column: 39, scope: !2432)
!2599 = !DILocation(line: 1804, column: 18, scope: !2432)
!2600 = !DILocation(line: 1805, column: 20, scope: !2432)
!2601 = !DILocation(line: 1806, column: 20, scope: !2432)
!2602 = !DILocation(line: 1807, column: 16, scope: !2432)
!2603 = !DILocation(line: 1808, column: 18, scope: !2432)
!2604 = !DILocation(line: 1809, column: 18, scope: !2432)
!2605 = !DILocation(line: 1810, column: 18, scope: !2432)
!2606 = !DILocation(line: 1811, column: 15, scope: !2432)
!2607 = !DILocation(line: 1812, column: 15, scope: !2432)
!2608 = !DILocation(line: 1813, column: 15, scope: !2432)
!2609 = !DILocation(line: 1814, column: 15, scope: !2432)
!2610 = !DILocation(line: 1815, column: 15, scope: !2432)
!2611 = !DILocation(line: 1816, column: 15, scope: !2432)
!2612 = !DILocation(line: 1817, column: 17, scope: !2432)
!2613 = !DILocation(line: 1818, column: 15, scope: !2432)
!2614 = !DILocation(line: 1819, column: 19, scope: !2432)
!2615 = !DILocation(line: 1820, column: 18, scope: !2432)
!2616 = !DILocation(line: 1821, column: 17, scope: !2432)
!2617 = !DILocation(line: 1822, column: 24, scope: !2432)
!2618 = !DILocation(line: 1823, column: 24, scope: !2432)
!2619 = !DILocation(line: 1825, column: 30, scope: !2432)
!2620 = !DILocation(line: 1825, column: 23, scope: !2432)
!2621 = !DILocation(line: 1825, column: 14, scope: !2432)
!2622 = !DILocation(line: 1825, column: 13, scope: !2432)
!2623 = !DILocation(line: 1826, column: 34, scope: !2432)
!2624 = !DILocation(line: 1826, column: 27, scope: !2432)
!2625 = !DILocation(line: 1826, column: 18, scope: !2432)
!2626 = !DILocation(line: 1826, column: 17, scope: !2432)
!2627 = !DILocation(line: 1828, column: 29, scope: !2432)
!2628 = !DILocation(line: 1828, column: 2, scope: !2432)
!2629 = !DILocation(line: 1829, column: 29, scope: !2432)
!2630 = !DILocation(line: 1829, column: 2, scope: !2432)
!2631 = !DILocation(line: 1830, column: 24, scope: !2432)
!2632 = !DILocation(line: 1830, column: 2, scope: !2432)
!2633 = !DILocation(line: 1831, column: 24, scope: !2432)
!2634 = !DILocation(line: 1831, column: 2, scope: !2432)
!2635 = !DILocation(line: 1832, column: 24, scope: !2432)
!2636 = !DILocation(line: 1832, column: 2, scope: !2432)
!2637 = !DILocation(line: 1833, column: 24, scope: !2432)
!2638 = !DILocation(line: 1833, column: 2, scope: !2432)
!2639 = !DILocation(line: 1834, column: 24, scope: !2432)
!2640 = !DILocation(line: 1834, column: 2, scope: !2432)
!2641 = !DILocation(line: 1835, column: 24, scope: !2432)
!2642 = !DILocation(line: 1835, column: 2, scope: !2432)
!2643 = !DILocation(line: 1836, column: 26, scope: !2432)
!2644 = !DILocation(line: 1836, column: 2, scope: !2432)
!2645 = !DILocation(line: 1837, column: 24, scope: !2432)
!2646 = !DILocation(line: 1837, column: 2, scope: !2432)
!2647 = !DILocation(line: 1838, column: 28, scope: !2432)
!2648 = !DILocation(line: 1838, column: 2, scope: !2432)
!2649 = !DILocation(line: 1839, column: 27, scope: !2432)
!2650 = !DILocation(line: 1839, column: 2, scope: !2432)
!2651 = !DILocation(line: 1840, column: 26, scope: !2432)
!2652 = !DILocation(line: 1840, column: 2, scope: !2432)
!2653 = !DILocation(line: 1841, column: 33, scope: !2432)
!2654 = !DILocation(line: 1841, column: 2, scope: !2432)
!2655 = !DILocation(line: 1842, column: 33, scope: !2432)
!2656 = !DILocation(line: 1842, column: 2, scope: !2432)
!2657 = !DILocation(line: 1843, column: 34, scope: !2432)
!2658 = !DILocation(line: 1843, column: 2, scope: !2432)
!2659 = !DILocation(line: 1844, column: 38, scope: !2432)
!2660 = !DILocation(line: 1844, column: 2, scope: !2432)
!2661 = !DILocation(line: 1846, column: 13, scope: !2432)
!2662 = !DILocation(line: 1846, column: 28, scope: !2432)
!2663 = !DILocation(line: 1846, column: 36, scope: !2432)
!2664 = !DILocation(line: 1846, column: 2, scope: !2432)
!2665 = !DILocation(line: 1847, column: 13, scope: !2432)
!2666 = !DILocation(line: 1847, column: 28, scope: !2432)
!2667 = !DILocation(line: 1847, column: 36, scope: !2432)
!2668 = !DILocation(line: 1847, column: 2, scope: !2432)
!2669 = !DILocation(line: 1848, column: 13, scope: !2432)
!2670 = !DILocation(line: 1848, column: 23, scope: !2432)
!2671 = !DILocation(line: 1848, column: 26, scope: !2432)
!2672 = !DILocation(line: 1848, column: 2, scope: !2432)
!2673 = !DILocation(line: 1849, column: 13, scope: !2432)
!2674 = !DILocation(line: 1849, column: 23, scope: !2432)
!2675 = !DILocation(line: 1849, column: 26, scope: !2432)
!2676 = !DILocation(line: 1849, column: 2, scope: !2432)
!2677 = !DILocation(line: 1850, column: 13, scope: !2432)
!2678 = !DILocation(line: 1850, column: 23, scope: !2432)
!2679 = !DILocation(line: 1850, column: 26, scope: !2432)
!2680 = !DILocation(line: 1850, column: 2, scope: !2432)
!2681 = !DILocation(line: 1851, column: 13, scope: !2432)
!2682 = !DILocation(line: 1851, column: 23, scope: !2432)
!2683 = !DILocation(line: 1851, column: 26, scope: !2432)
!2684 = !DILocation(line: 1851, column: 2, scope: !2432)
!2685 = !DILocation(line: 1852, column: 13, scope: !2432)
!2686 = !DILocation(line: 1852, column: 23, scope: !2432)
!2687 = !DILocation(line: 1852, column: 26, scope: !2432)
!2688 = !DILocation(line: 1852, column: 2, scope: !2432)
!2689 = !DILocation(line: 1853, column: 13, scope: !2432)
!2690 = !DILocation(line: 1853, column: 23, scope: !2432)
!2691 = !DILocation(line: 1853, column: 26, scope: !2432)
!2692 = !DILocation(line: 1853, column: 2, scope: !2432)
!2693 = !DILocation(line: 1855, column: 33, scope: !2432)
!2694 = !DILocation(line: 1855, column: 64, scope: !2432)
!2695 = !DILocation(line: 1855, column: 32, scope: !2432)
!2696 = !DILocation(line: 1856, column: 33, scope: !2432)
!2697 = !DILocation(line: 1856, column: 64, scope: !2432)
!2698 = !DILocation(line: 1856, column: 32, scope: !2432)
!2699 = !DILocation(line: 1857, column: 35, scope: !2432)
!2700 = !DILocation(line: 1857, column: 68, scope: !2432)
!2701 = !DILocation(line: 1857, column: 34, scope: !2432)
!2702 = !DILocation(line: 1858, column: 34, scope: !2432)
!2703 = !DILocation(line: 1858, column: 66, scope: !2432)
!2704 = !DILocation(line: 1858, column: 33, scope: !2432)
!2705 = !DILocation(line: 1859, column: 34, scope: !2432)
!2706 = !DILocation(line: 1859, column: 66, scope: !2432)
!2707 = !DILocation(line: 1859, column: 33, scope: !2432)
!2708 = !DILocation(line: 1860, column: 33, scope: !2432)
!2709 = !DILocation(line: 1860, column: 64, scope: !2432)
!2710 = !DILocation(line: 1860, column: 32, scope: !2432)
!2711 = !DILocation(line: 1861, column: 35, scope: !2432)
!2712 = !DILocation(line: 1861, column: 68, scope: !2432)
!2713 = !DILocation(line: 1861, column: 34, scope: !2432)
!2714 = !DILocation(line: 1862, column: 35, scope: !2432)
!2715 = !DILocation(line: 1862, column: 68, scope: !2432)
!2716 = !DILocation(line: 1862, column: 34, scope: !2432)
!2717 = !DILocation(line: 1863, column: 34, scope: !2432)
!2718 = !DILocation(line: 1863, column: 66, scope: !2432)
!2719 = !DILocation(line: 1863, column: 33, scope: !2432)
!2720 = !DILocation(line: 1864, column: 33, scope: !2432)
!2721 = !DILocation(line: 1864, column: 64, scope: !2432)
!2722 = !DILocation(line: 1864, column: 32, scope: !2432)
!2723 = !DILocation(line: 1865, column: 36, scope: !2432)
!2724 = !DILocation(line: 1865, column: 70, scope: !2432)
!2725 = !DILocation(line: 1865, column: 35, scope: !2432)
!2726 = !DILocation(line: 1867, column: 35, scope: !2432)
!2727 = !DILocation(line: 1867, column: 64, scope: !2432)
!2728 = !DILocation(line: 1867, column: 34, scope: !2432)
!2729 = !DILocation(line: 1868, column: 35, scope: !2432)
!2730 = !DILocation(line: 1868, column: 64, scope: !2432)
!2731 = !DILocation(line: 1868, column: 34, scope: !2432)
!2732 = !DILocation(line: 1869, column: 37, scope: !2432)
!2733 = !DILocation(line: 1869, column: 68, scope: !2432)
!2734 = !DILocation(line: 1869, column: 36, scope: !2432)
!2735 = !DILocation(line: 1870, column: 36, scope: !2432)
!2736 = !DILocation(line: 1870, column: 66, scope: !2432)
!2737 = !DILocation(line: 1870, column: 35, scope: !2432)
!2738 = !DILocation(line: 1871, column: 36, scope: !2432)
!2739 = !DILocation(line: 1871, column: 66, scope: !2432)
!2740 = !DILocation(line: 1871, column: 35, scope: !2432)
!2741 = !DILocation(line: 1872, column: 35, scope: !2432)
!2742 = !DILocation(line: 1872, column: 64, scope: !2432)
!2743 = !DILocation(line: 1872, column: 34, scope: !2432)
!2744 = !DILocation(line: 1873, column: 37, scope: !2432)
!2745 = !DILocation(line: 1873, column: 68, scope: !2432)
!2746 = !DILocation(line: 1873, column: 36, scope: !2432)
!2747 = !DILocation(line: 1874, column: 37, scope: !2432)
!2748 = !DILocation(line: 1874, column: 68, scope: !2432)
!2749 = !DILocation(line: 1874, column: 36, scope: !2432)
!2750 = !DILocation(line: 1875, column: 36, scope: !2432)
!2751 = !DILocation(line: 1875, column: 66, scope: !2432)
!2752 = !DILocation(line: 1875, column: 35, scope: !2432)
!2753 = !DILocation(line: 1876, column: 35, scope: !2432)
!2754 = !DILocation(line: 1876, column: 64, scope: !2432)
!2755 = !DILocation(line: 1876, column: 34, scope: !2432)
!2756 = !DILocation(line: 1877, column: 38, scope: !2432)
!2757 = !DILocation(line: 1877, column: 70, scope: !2432)
!2758 = !DILocation(line: 1877, column: 37, scope: !2432)
!2759 = !DILocation(line: 1878, column: 1, scope: !2432)
!2760 = distinct !DISubprogram(name: "conj_grad_gpu", linkageName: "_ZL13conj_grad_gpuPd", scope: !3, file: !3, line: 990, type: !2761, scopeLine: 990, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1163)
!2761 = !DISubroutineType(types: !2762)
!2762 = !{null, !99}
!2763 = !DILocalVariable(name: "rnorm", arg: 1, scope: !2760, file: !3, line: 990, type: !99)
!2764 = !DILocation(line: 990, column: 35, scope: !2760)
!2765 = !DILocalVariable(name: "d", scope: !2760, file: !3, line: 991, type: !100)
!2766 = !DILocation(line: 991, column: 9, scope: !2760)
!2767 = !DILocalVariable(name: "sum", scope: !2760, file: !3, line: 991, type: !100)
!2768 = !DILocation(line: 991, column: 12, scope: !2760)
!2769 = !DILocalVariable(name: "rho", scope: !2760, file: !3, line: 991, type: !100)
!2770 = !DILocation(line: 991, column: 17, scope: !2760)
!2771 = !DILocalVariable(name: "rho0", scope: !2760, file: !3, line: 991, type: !100)
!2772 = !DILocation(line: 991, column: 22, scope: !2760)
!2773 = !DILocalVariable(name: "alpha", scope: !2760, file: !3, line: 991, type: !100)
!2774 = !DILocation(line: 991, column: 28, scope: !2760)
!2775 = !DILocalVariable(name: "beta", scope: !2760, file: !3, line: 991, type: !100)
!2776 = !DILocation(line: 991, column: 35, scope: !2760)
!2777 = !DILocalVariable(name: "cgit", scope: !2760, file: !3, line: 992, type: !97)
!2778 = !DILocation(line: 992, column: 6, scope: !2760)
!2779 = !DILocalVariable(name: "cgitmax", scope: !2760, file: !3, line: 992, type: !97)
!2780 = !DILocation(line: 992, column: 12, scope: !2760)
!2781 = !DILocation(line: 995, column: 2, scope: !2760)
!2782 = !DILocation(line: 998, column: 2, scope: !2760)
!2783 = !DILocation(line: 1001, column: 11, scope: !2784)
!2784 = distinct !DILexicalBlock(scope: !2760, file: !3, line: 1001, column: 2)
!2785 = !DILocation(line: 1001, column: 6, scope: !2784)
!2786 = !DILocation(line: 1001, column: 16, scope: !2787)
!2787 = distinct !DILexicalBlock(scope: !2784, file: !3, line: 1001, column: 2)
!2788 = !DILocation(line: 1001, column: 24, scope: !2787)
!2789 = !DILocation(line: 1001, column: 21, scope: !2787)
!2790 = !DILocation(line: 1001, column: 2, scope: !2784)
!2791 = !DILocation(line: 1003, column: 3, scope: !2792)
!2792 = distinct !DILexicalBlock(scope: !2787, file: !3, line: 1001, column: 40)
!2793 = !DILocation(line: 1006, column: 3, scope: !2792)
!2794 = !DILocation(line: 1008, column: 11, scope: !2792)
!2795 = !DILocation(line: 1008, column: 17, scope: !2792)
!2796 = !DILocation(line: 1008, column: 15, scope: !2792)
!2797 = !DILocation(line: 1008, column: 9, scope: !2792)
!2798 = !DILocation(line: 1011, column: 10, scope: !2792)
!2799 = !DILocation(line: 1011, column: 8, scope: !2792)
!2800 = !DILocation(line: 1014, column: 24, scope: !2792)
!2801 = !DILocation(line: 1014, column: 3, scope: !2792)
!2802 = !DILocation(line: 1017, column: 3, scope: !2792)
!2803 = !DILocation(line: 1020, column: 10, scope: !2792)
!2804 = !DILocation(line: 1020, column: 16, scope: !2792)
!2805 = !DILocation(line: 1020, column: 14, scope: !2792)
!2806 = !DILocation(line: 1020, column: 8, scope: !2792)
!2807 = !DILocation(line: 1023, column: 25, scope: !2792)
!2808 = !DILocation(line: 1023, column: 3, scope: !2792)
!2809 = !DILocation(line: 1024, column: 2, scope: !2792)
!2810 = !DILocation(line: 1001, column: 37, scope: !2787)
!2811 = !DILocation(line: 1001, column: 2, scope: !2787)
!2812 = distinct !{!2812, !2790, !2813}
!2813 = !DILocation(line: 1024, column: 2, scope: !2784)
!2814 = !DILocation(line: 1027, column: 2, scope: !2760)
!2815 = !DILocation(line: 1030, column: 2, scope: !2760)
!2816 = !DILocation(line: 1032, column: 16, scope: !2760)
!2817 = !DILocation(line: 1032, column: 11, scope: !2760)
!2818 = !DILocation(line: 1032, column: 3, scope: !2760)
!2819 = !DILocation(line: 1032, column: 9, scope: !2760)
!2820 = !DILocation(line: 1033, column: 1, scope: !2760)
!2821 = distinct !DISubprogram(name: "gpu_kernel_ten_host", linkageName: "_ZL19gpu_kernel_ten_hostPdS_", scope: !3, file: !3, line: 1433, type: !2822, scopeLine: 1434, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1163)
!2822 = !DISubroutineType(types: !2823)
!2823 = !{null, !99, !99}
!2824 = !DILocalVariable(name: "norm_temp1", arg: 1, scope: !2821, file: !3, line: 1433, type: !99)
!2825 = !DILocation(line: 1433, column: 41, scope: !2821)
!2826 = !DILocalVariable(name: "norm_temp2", arg: 2, scope: !2821, file: !3, line: 1434, type: !99)
!2827 = !DILocation(line: 1434, column: 11, scope: !2821)
!2828 = !DILocation(line: 1438, column: 21, scope: !2821)
!2829 = !DILocation(line: 1438, column: 51, scope: !2821)
!2830 = !DILocation(line: 1438, column: 83, scope: !2821)
!2831 = !DILocation(line: 1438, column: 18, scope: !2821)
!2832 = !DILocation(line: 1438, column: 2, scope: !2821)
!2833 = !DILocation(line: 1438, column: 117, scope: !2821)
!2834 = !DILocation(line: 1438, column: 136, scope: !2821)
!2835 = !DILocation(line: 1438, column: 145, scope: !2821)
!2836 = !DILocation(line: 1439, column: 21, scope: !2821)
!2837 = !DILocation(line: 1439, column: 51, scope: !2821)
!2838 = !DILocation(line: 1439, column: 83, scope: !2821)
!2839 = !DILocation(line: 1439, column: 18, scope: !2821)
!2840 = !DILocation(line: 1439, column: 2, scope: !2821)
!2841 = !DILocation(line: 1439, column: 117, scope: !2821)
!2842 = !DILocation(line: 1439, column: 140, scope: !2821)
!2843 = !DILocation(line: 1439, column: 149, scope: !2821)
!2844 = !DILocation(line: 1441, column: 20, scope: !2821)
!2845 = !DILocation(line: 1442, column: 24, scope: !2821)
!2846 = !DILocation(line: 1443, column: 13, scope: !2821)
!2847 = !DILocation(line: 1443, column: 26, scope: !2821)
!2848 = !DILocation(line: 1443, column: 46, scope: !2821)
!2849 = !DILocation(line: 1443, column: 2, scope: !2821)
!2850 = !DILocation(line: 1444, column: 13, scope: !2821)
!2851 = !DILocation(line: 1444, column: 30, scope: !2821)
!2852 = !DILocation(line: 1444, column: 54, scope: !2821)
!2853 = !DILocation(line: 1444, column: 2, scope: !2821)
!2854 = !DILocalVariable(name: "i", scope: !2855, file: !3, line: 1446, type: !97)
!2855 = distinct !DILexicalBlock(scope: !2821, file: !3, line: 1446, column: 2)
!2856 = !DILocation(line: 1446, column: 10, scope: !2855)
!2857 = !DILocation(line: 1446, column: 6, scope: !2855)
!2858 = !DILocation(line: 1446, column: 15, scope: !2859)
!2859 = distinct !DILexicalBlock(scope: !2855, file: !3, line: 1446, column: 2)
!2860 = !DILocation(line: 1446, column: 17, scope: !2859)
!2861 = !DILocation(line: 1446, column: 16, scope: !2859)
!2862 = !DILocation(line: 1446, column: 2, scope: !2855)
!2863 = !DILocation(line: 1446, column: 73, scope: !2864)
!2864 = distinct !DILexicalBlock(scope: !2859, file: !3, line: 1446, column: 52)
!2865 = !DILocation(line: 1446, column: 85, scope: !2864)
!2866 = !DILocation(line: 1446, column: 71, scope: !2864)
!2867 = !DILocation(line: 1446, column: 112, scope: !2864)
!2868 = !DILocation(line: 1446, column: 128, scope: !2864)
!2869 = !DILocation(line: 1446, column: 110, scope: !2864)
!2870 = !DILocation(line: 1446, column: 131, scope: !2864)
!2871 = !DILocation(line: 1446, column: 49, scope: !2859)
!2872 = !DILocation(line: 1446, column: 2, scope: !2859)
!2873 = distinct !{!2873, !2862, !2874}
!2874 = !DILocation(line: 1446, column: 131, scope: !2855)
!2875 = !DILocation(line: 1447, column: 14, scope: !2821)
!2876 = !DILocation(line: 1447, column: 3, scope: !2821)
!2877 = !DILocation(line: 1447, column: 13, scope: !2821)
!2878 = !DILocation(line: 1448, column: 14, scope: !2821)
!2879 = !DILocation(line: 1448, column: 3, scope: !2821)
!2880 = !DILocation(line: 1448, column: 13, scope: !2821)
!2881 = !DILocation(line: 1452, column: 1, scope: !2821)
!2882 = distinct !DISubprogram(name: "gpu_kernel_eleven_host", linkageName: "_ZL22gpu_kernel_eleven_hostd", scope: !3, file: !3, line: 1520, type: !2883, scopeLine: 1520, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1163)
!2883 = !DISubroutineType(types: !2884)
!2884 = !{null, !100}
!2885 = !DILocalVariable(name: "norm_temp2", arg: 1, scope: !2882, file: !3, line: 1520, type: !100)
!2886 = !DILocation(line: 1520, column: 43, scope: !2882)
!2887 = !DILocation(line: 1524, column: 29, scope: !2882)
!2888 = !DILocation(line: 1525, column: 3, scope: !2882)
!2889 = !DILocation(line: 1524, column: 26, scope: !2882)
!2890 = !DILocation(line: 1524, column: 2, scope: !2882)
!2891 = !DILocation(line: 1526, column: 5, scope: !2882)
!2892 = !DILocation(line: 1527, column: 5, scope: !2882)
!2893 = !DILocation(line: 1528, column: 5, scope: !2882)
!2894 = !DILocation(line: 1532, column: 1, scope: !2882)
!2895 = distinct !DISubprogram(name: "release_gpu", linkageName: "_ZL11release_gpuv", scope: !3, file: !3, line: 1643, type: !666, scopeLine: 1643, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1163)
!2896 = !DILocation(line: 1644, column: 11, scope: !2895)
!2897 = !DILocation(line: 1644, column: 2, scope: !2895)
!2898 = !DILocation(line: 1645, column: 11, scope: !2895)
!2899 = !DILocation(line: 1645, column: 2, scope: !2895)
!2900 = !DILocation(line: 1646, column: 11, scope: !2895)
!2901 = !DILocation(line: 1646, column: 2, scope: !2895)
!2902 = !DILocation(line: 1647, column: 11, scope: !2895)
!2903 = !DILocation(line: 1647, column: 2, scope: !2895)
!2904 = !DILocation(line: 1648, column: 11, scope: !2895)
!2905 = !DILocation(line: 1648, column: 2, scope: !2895)
!2906 = !DILocation(line: 1649, column: 11, scope: !2895)
!2907 = !DILocation(line: 1649, column: 2, scope: !2895)
!2908 = !DILocation(line: 1650, column: 11, scope: !2895)
!2909 = !DILocation(line: 1650, column: 2, scope: !2895)
!2910 = !DILocation(line: 1651, column: 11, scope: !2895)
!2911 = !DILocation(line: 1651, column: 2, scope: !2895)
!2912 = !DILocation(line: 1652, column: 11, scope: !2895)
!2913 = !DILocation(line: 1652, column: 2, scope: !2895)
!2914 = !DILocation(line: 1653, column: 11, scope: !2895)
!2915 = !DILocation(line: 1653, column: 2, scope: !2895)
!2916 = !DILocation(line: 1654, column: 11, scope: !2895)
!2917 = !DILocation(line: 1654, column: 2, scope: !2895)
!2918 = !DILocation(line: 1655, column: 11, scope: !2895)
!2919 = !DILocation(line: 1655, column: 2, scope: !2895)
!2920 = !DILocation(line: 1656, column: 11, scope: !2895)
!2921 = !DILocation(line: 1656, column: 2, scope: !2895)
!2922 = !DILocation(line: 1657, column: 11, scope: !2895)
!2923 = !DILocation(line: 1657, column: 2, scope: !2895)
!2924 = !DILocation(line: 1658, column: 11, scope: !2895)
!2925 = !DILocation(line: 1658, column: 2, scope: !2895)
!2926 = !DILocation(line: 1659, column: 11, scope: !2895)
!2927 = !DILocation(line: 1659, column: 2, scope: !2895)
!2928 = !DILocation(line: 1660, column: 11, scope: !2895)
!2929 = !DILocation(line: 1660, column: 2, scope: !2895)
!2930 = !DILocation(line: 1661, column: 1, scope: !2895)
!2931 = distinct !DISubprogram(name: "gpu_kernel_one_device", linkageName: "_Z21gpu_kernel_one_devicePdS_S_S_S_", scope: !3, file: !3, line: 1051, type: !2932, scopeLine: 1055, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !1163)
!2932 = !DISubroutineType(types: !2933)
!2933 = !{null, !99, !99, !99, !99, !99}
!2934 = !DILocalVariable(name: "p", arg: 1, scope: !2931, file: !3, line: 1051, type: !99)
!2935 = !DILocation(line: 1051, column: 46, scope: !2931)
!2936 = !DILocalVariable(name: "q", arg: 2, scope: !2931, file: !3, line: 1052, type: !99)
!2937 = !DILocation(line: 1052, column: 10, scope: !2931)
!2938 = !DILocalVariable(name: "r", arg: 3, scope: !2931, file: !3, line: 1053, type: !99)
!2939 = !DILocation(line: 1053, column: 10, scope: !2931)
!2940 = !DILocalVariable(name: "x", arg: 4, scope: !2931, file: !3, line: 1054, type: !99)
!2941 = !DILocation(line: 1054, column: 10, scope: !2931)
!2942 = !DILocalVariable(name: "z", arg: 5, scope: !2931, file: !3, line: 1055, type: !99)
!2943 = !DILocation(line: 1055, column: 10, scope: !2931)
!2944 = !DILocation(line: 1055, column: 14, scope: !2931)
!2945 = !DILocation(line: 1063, column: 1, scope: !2931)
!2946 = distinct !DISubprogram(name: "gpu_kernel_two_device", linkageName: "_Z21gpu_kernel_two_devicePdS_S_", scope: !3, file: !3, line: 1084, type: !2947, scopeLine: 1086, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !1163)
!2947 = !DISubroutineType(types: !2948)
!2948 = !{null, !99, !99, !99}
!2949 = !DILocalVariable(name: "r", arg: 1, scope: !2946, file: !3, line: 1084, type: !99)
!2950 = !DILocation(line: 1084, column: 46, scope: !2946)
!2951 = !DILocalVariable(name: "rho", arg: 2, scope: !2946, file: !3, line: 1085, type: !99)
!2952 = !DILocation(line: 1085, column: 11, scope: !2946)
!2953 = !DILocalVariable(name: "global_data", arg: 3, scope: !2946, file: !3, line: 1086, type: !99)
!2954 = !DILocation(line: 1086, column: 10, scope: !2946)
!2955 = !DILocation(line: 1086, column: 24, scope: !2946)
!2956 = !DILocation(line: 1116, column: 1, scope: !2946)
!2957 = distinct !DISubprogram(name: "gpu_kernel_three_device", linkageName: "_Z23gpu_kernel_three_devicePiS_PdS0_S0_", scope: !3, file: !3, line: 1135, type: !2958, scopeLine: 1139, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !1163)
!2958 = !DISubroutineType(types: !2959)
!2959 = !{null, !98, !98, !99, !99, !99}
!2960 = !DILocalVariable(name: "colidx", arg: 1, scope: !2957, file: !3, line: 1135, type: !98)
!2961 = !DILocation(line: 1135, column: 45, scope: !2957)
!2962 = !DILocalVariable(name: "rowstr", arg: 2, scope: !2957, file: !3, line: 1136, type: !98)
!2963 = !DILocation(line: 1136, column: 7, scope: !2957)
!2964 = !DILocalVariable(name: "a", arg: 3, scope: !2957, file: !3, line: 1137, type: !99)
!2965 = !DILocation(line: 1137, column: 10, scope: !2957)
!2966 = !DILocalVariable(name: "p", arg: 4, scope: !2957, file: !3, line: 1138, type: !99)
!2967 = !DILocation(line: 1138, column: 10, scope: !2957)
!2968 = !DILocalVariable(name: "q", arg: 5, scope: !2957, file: !3, line: 1139, type: !99)
!2969 = !DILocation(line: 1139, column: 10, scope: !2957)
!2970 = !DILocation(line: 1139, column: 14, scope: !2957)
!2971 = !DILocation(line: 1168, column: 1, scope: !2957)
!2972 = distinct !DISubprogram(name: "gpu_kernel_four_device", linkageName: "_Z22gpu_kernel_four_devicePdS_S_S_", scope: !3, file: !3, line: 1190, type: !2973, scopeLine: 1193, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !1163)
!2973 = !DISubroutineType(types: !2974)
!2974 = !{null, !99, !99, !99, !99}
!2975 = !DILocalVariable(name: "d", arg: 1, scope: !2972, file: !3, line: 1190, type: !99)
!2976 = !DILocation(line: 1190, column: 48, scope: !2972)
!2977 = !DILocalVariable(name: "p", arg: 2, scope: !2972, file: !3, line: 1191, type: !99)
!2978 = !DILocation(line: 1191, column: 11, scope: !2972)
!2979 = !DILocalVariable(name: "q", arg: 3, scope: !2972, file: !3, line: 1192, type: !99)
!2980 = !DILocation(line: 1192, column: 11, scope: !2972)
!2981 = !DILocalVariable(name: "global_data", arg: 4, scope: !2972, file: !3, line: 1193, type: !99)
!2982 = !DILocation(line: 1193, column: 10, scope: !2972)
!2983 = !DILocation(line: 1193, column: 24, scope: !2972)
!2984 = !DILocation(line: 1224, column: 1, scope: !2972)
!2985 = distinct !DISubprogram(name: "gpu_kernel_five_1", linkageName: "_Z17gpu_kernel_five_1dPdS_", scope: !3, file: !3, line: 1245, type: !2986, scopeLine: 1247, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !1163)
!2986 = !DISubroutineType(types: !2987)
!2987 = !{null, !100, !99, !99}
!2988 = !DILocalVariable(name: "alpha", arg: 1, scope: !2985, file: !3, line: 1245, type: !100)
!2989 = !DILocation(line: 1245, column: 42, scope: !2985)
!2990 = !DILocalVariable(name: "p", arg: 2, scope: !2985, file: !3, line: 1246, type: !99)
!2991 = !DILocation(line: 1246, column: 11, scope: !2985)
!2992 = !DILocalVariable(name: "z", arg: 3, scope: !2985, file: !3, line: 1247, type: !99)
!2993 = !DILocation(line: 1247, column: 11, scope: !2985)
!2994 = !DILocation(line: 1247, column: 13, scope: !2985)
!2995 = !DILocation(line: 1251, column: 1, scope: !2985)
!2996 = distinct !DISubprogram(name: "gpu_kernel_five_2", linkageName: "_Z17gpu_kernel_five_2dPdS_", scope: !3, file: !3, line: 1253, type: !2986, scopeLine: 1255, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !1163)
!2997 = !DILocalVariable(name: "alpha", arg: 1, scope: !2996, file: !3, line: 1253, type: !100)
!2998 = !DILocation(line: 1253, column: 42, scope: !2996)
!2999 = !DILocalVariable(name: "q", arg: 2, scope: !2996, file: !3, line: 1254, type: !99)
!3000 = !DILocation(line: 1254, column: 11, scope: !2996)
!3001 = !DILocalVariable(name: "r", arg: 3, scope: !2996, file: !3, line: 1255, type: !99)
!3002 = !DILocation(line: 1255, column: 11, scope: !2996)
!3003 = !DILocation(line: 1255, column: 13, scope: !2996)
!3004 = !DILocation(line: 1259, column: 1, scope: !2996)
!3005 = distinct !DISubprogram(name: "gpu_kernel_six_device", linkageName: "_Z21gpu_kernel_six_devicePdS_", scope: !3, file: !3, line: 1279, type: !2822, scopeLine: 1280, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !1163)
!3006 = !DILocalVariable(name: "r", arg: 1, scope: !3005, file: !3, line: 1279, type: !99)
!3007 = !DILocation(line: 1279, column: 46, scope: !3005)
!3008 = !DILocalVariable(name: "global_data", arg: 2, scope: !3005, file: !3, line: 1280, type: !99)
!3009 = !DILocation(line: 1280, column: 10, scope: !3005)
!3010 = !DILocation(line: 1280, column: 24, scope: !3005)
!3011 = !DILocation(line: 1305, column: 1, scope: !3005)
!3012 = distinct !DISubprogram(name: "gpu_kernel_seven_device", linkageName: "_Z23gpu_kernel_seven_devicedPdS_", scope: !3, file: !3, line: 1321, type: !2986, scopeLine: 1323, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !1163)
!3013 = !DILocalVariable(name: "beta", arg: 1, scope: !3012, file: !3, line: 1321, type: !100)
!3014 = !DILocation(line: 1321, column: 48, scope: !3012)
!3015 = !DILocalVariable(name: "p", arg: 2, scope: !3012, file: !3, line: 1322, type: !99)
!3016 = !DILocation(line: 1322, column: 11, scope: !3012)
!3017 = !DILocalVariable(name: "r", arg: 3, scope: !3012, file: !3, line: 1323, type: !99)
!3018 = !DILocation(line: 1323, column: 11, scope: !3012)
!3019 = !DILocation(line: 1323, column: 13, scope: !3012)
!3020 = !DILocation(line: 1327, column: 1, scope: !3012)
!3021 = distinct !DISubprogram(name: "gpu_kernel_eight_device", linkageName: "_Z23gpu_kernel_eight_devicePiS_PdS0_S0_", scope: !3, file: !3, line: 1346, type: !2958, scopeLine: 1350, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !1163)
!3022 = !DILocalVariable(name: "colidx", arg: 1, scope: !3021, file: !3, line: 1346, type: !98)
!3023 = !DILocation(line: 1346, column: 45, scope: !3021)
!3024 = !DILocalVariable(name: "rowstr", arg: 2, scope: !3021, file: !3, line: 1347, type: !98)
!3025 = !DILocation(line: 1347, column: 7, scope: !3021)
!3026 = !DILocalVariable(name: "a", arg: 3, scope: !3021, file: !3, line: 1348, type: !99)
!3027 = !DILocation(line: 1348, column: 10, scope: !3021)
!3028 = !DILocalVariable(name: "r", arg: 4, scope: !3021, file: !3, line: 1349, type: !99)
!3029 = !DILocation(line: 1349, column: 10, scope: !3021)
!3030 = !DILocalVariable(name: "z", arg: 5, scope: !3021, file: !3, line: 1350, type: !99)
!3031 = !DILocation(line: 1350, column: 11, scope: !3021)
!3032 = !DILocation(line: 1350, column: 13, scope: !3021)
!3033 = !DILocation(line: 1379, column: 1, scope: !3021)
!3034 = distinct !DISubprogram(name: "gpu_kernel_nine_device", linkageName: "_Z22gpu_kernel_nine_devicePdS_S_S_", scope: !3, file: !3, line: 1401, type: !2973, scopeLine: 1401, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !1163)
!3035 = !DILocalVariable(name: "r", arg: 1, scope: !3034, file: !3, line: 1401, type: !99)
!3036 = !DILocation(line: 1401, column: 47, scope: !3034)
!3037 = !DILocalVariable(name: "x", arg: 2, scope: !3034, file: !3, line: 1401, type: !99)
!3038 = !DILocation(line: 1401, column: 59, scope: !3034)
!3039 = !DILocalVariable(name: "sum", arg: 3, scope: !3034, file: !3, line: 1401, type: !99)
!3040 = !DILocation(line: 1401, column: 72, scope: !3034)
!3041 = !DILocalVariable(name: "global_data", arg: 4, scope: !3034, file: !3, line: 1401, type: !99)
!3042 = !DILocation(line: 1401, column: 84, scope: !3034)
!3043 = !DILocation(line: 1401, column: 98, scope: !3034)
!3044 = !DILocation(line: 1431, column: 1, scope: !3034)
!3045 = distinct !DISubprogram(name: "gpu_kernel_ten_1", linkageName: "_Z16gpu_kernel_ten_1PdS_S_", scope: !3, file: !3, line: 1454, type: !2947, scopeLine: 1456, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !1163)
!3046 = !DILocalVariable(name: "norm_temp", arg: 1, scope: !3045, file: !3, line: 1454, type: !99)
!3047 = !DILocation(line: 1454, column: 42, scope: !3045)
!3048 = !DILocalVariable(name: "x", arg: 2, scope: !3045, file: !3, line: 1455, type: !99)
!3049 = !DILocation(line: 1455, column: 10, scope: !3045)
!3050 = !DILocalVariable(name: "z", arg: 3, scope: !3045, file: !3, line: 1456, type: !99)
!3051 = !DILocation(line: 1456, column: 10, scope: !3045)
!3052 = !DILocation(line: 1456, column: 14, scope: !3045)
!3053 = !DILocation(line: 1485, column: 1, scope: !3045)
!3054 = distinct !DISubprogram(name: "gpu_kernel_ten_2", linkageName: "_Z16gpu_kernel_ten_2PdS_S_", scope: !3, file: !3, line: 1487, type: !2947, scopeLine: 1489, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !1163)
!3055 = !DILocalVariable(name: "norm_temp", arg: 1, scope: !3054, file: !3, line: 1487, type: !99)
!3056 = !DILocation(line: 1487, column: 42, scope: !3054)
!3057 = !DILocalVariable(name: "x", arg: 2, scope: !3054, file: !3, line: 1488, type: !99)
!3058 = !DILocation(line: 1488, column: 10, scope: !3054)
!3059 = !DILocalVariable(name: "z", arg: 3, scope: !3054, file: !3, line: 1489, type: !99)
!3060 = !DILocation(line: 1489, column: 10, scope: !3054)
!3061 = !DILocation(line: 1489, column: 14, scope: !3054)
!3062 = !DILocation(line: 1518, column: 1, scope: !3054)
!3063 = distinct !DISubprogram(name: "gpu_kernel_eleven_device", linkageName: "_Z24gpu_kernel_eleven_devicedPdS_", scope: !3, file: !3, line: 1534, type: !2986, scopeLine: 1534, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !1163)
!3064 = !DILocalVariable(name: "norm_temp2", arg: 1, scope: !3063, file: !3, line: 1534, type: !100)
!3065 = !DILocation(line: 1534, column: 49, scope: !3063)
!3066 = !DILocalVariable(name: "x", arg: 2, scope: !3063, file: !3, line: 1534, type: !99)
!3067 = !DILocation(line: 1534, column: 68, scope: !3063)
!3068 = !DILocalVariable(name: "z", arg: 3, scope: !3063, file: !3, line: 1534, type: !99)
!3069 = !DILocation(line: 1534, column: 80, scope: !3063)
!3070 = !DILocation(line: 1534, column: 84, scope: !3063)
!3071 = !DILocation(line: 1538, column: 1, scope: !3063)
!3072 = distinct !DISubprogram(name: "gpu_kernel_one_host", linkageName: "_ZL19gpu_kernel_one_hostv", scope: !3, file: !3, line: 1035, type: !666, scopeLine: 1035, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1163)
!3073 = !DILocation(line: 1039, column: 26, scope: !3072)
!3074 = !DILocation(line: 1040, column: 3, scope: !3072)
!3075 = !DILocation(line: 1039, column: 23, scope: !3072)
!3076 = !DILocation(line: 1039, column: 2, scope: !3072)
!3077 = !DILocation(line: 1041, column: 5, scope: !3072)
!3078 = !DILocation(line: 1042, column: 5, scope: !3072)
!3079 = !DILocation(line: 1043, column: 5, scope: !3072)
!3080 = !DILocation(line: 1044, column: 5, scope: !3072)
!3081 = !DILocation(line: 1045, column: 5, scope: !3072)
!3082 = !DILocation(line: 1049, column: 1, scope: !3072)
!3083 = distinct !DISubprogram(name: "gpu_kernel_two_host", linkageName: "_ZL19gpu_kernel_two_hostPd", scope: !3, file: !3, line: 1065, type: !2761, scopeLine: 1065, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1163)
!3084 = !DILocalVariable(name: "rho_host", arg: 1, scope: !3083, file: !3, line: 1065, type: !99)
!3085 = !DILocation(line: 1065, column: 41, scope: !3083)
!3086 = !DILocation(line: 1069, column: 26, scope: !3083)
!3087 = !DILocation(line: 1070, column: 3, scope: !3083)
!3088 = !DILocation(line: 1071, column: 3, scope: !3083)
!3089 = !DILocation(line: 1069, column: 23, scope: !3083)
!3090 = !DILocation(line: 1069, column: 2, scope: !3083)
!3091 = !DILocation(line: 1072, column: 5, scope: !3083)
!3092 = !DILocation(line: 1073, column: 5, scope: !3083)
!3093 = !DILocation(line: 1074, column: 5, scope: !3083)
!3094 = !DILocation(line: 1075, column: 20, scope: !3083)
!3095 = !DILocation(line: 1076, column: 13, scope: !3083)
!3096 = !DILocation(line: 1076, column: 26, scope: !3083)
!3097 = !DILocation(line: 1076, column: 46, scope: !3083)
!3098 = !DILocation(line: 1076, column: 2, scope: !3083)
!3099 = !DILocalVariable(name: "i", scope: !3100, file: !3, line: 1077, type: !97)
!3100 = distinct !DILexicalBlock(scope: !3083, file: !3, line: 1077, column: 2)
!3101 = !DILocation(line: 1077, column: 10, scope: !3100)
!3102 = !DILocation(line: 1077, column: 6, scope: !3100)
!3103 = !DILocation(line: 1077, column: 15, scope: !3104)
!3104 = distinct !DILexicalBlock(scope: !3100, file: !3, line: 1077, column: 2)
!3105 = !DILocation(line: 1077, column: 17, scope: !3104)
!3106 = !DILocation(line: 1077, column: 16, scope: !3104)
!3107 = !DILocation(line: 1077, column: 2, scope: !3100)
!3108 = !DILocation(line: 1077, column: 73, scope: !3109)
!3109 = distinct !DILexicalBlock(scope: !3104, file: !3, line: 1077, column: 52)
!3110 = !DILocation(line: 1077, column: 85, scope: !3109)
!3111 = !DILocation(line: 1077, column: 71, scope: !3109)
!3112 = !DILocation(line: 1077, column: 88, scope: !3109)
!3113 = !DILocation(line: 1077, column: 49, scope: !3104)
!3114 = !DILocation(line: 1077, column: 2, scope: !3104)
!3115 = distinct !{!3115, !3107, !3116}
!3116 = !DILocation(line: 1077, column: 88, scope: !3100)
!3117 = !DILocation(line: 1078, column: 12, scope: !3083)
!3118 = !DILocation(line: 1078, column: 3, scope: !3083)
!3119 = !DILocation(line: 1078, column: 11, scope: !3083)
!3120 = !DILocation(line: 1082, column: 1, scope: !3083)
!3121 = distinct !DISubprogram(name: "gpu_kernel_three_host", linkageName: "_ZL21gpu_kernel_three_hostv", scope: !3, file: !3, line: 1118, type: !666, scopeLine: 1118, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1163)
!3122 = !DILocation(line: 1122, column: 28, scope: !3121)
!3123 = !DILocation(line: 1123, column: 3, scope: !3121)
!3124 = !DILocation(line: 1124, column: 3, scope: !3121)
!3125 = !DILocation(line: 1122, column: 25, scope: !3121)
!3126 = !DILocation(line: 1122, column: 2, scope: !3121)
!3127 = !DILocation(line: 1125, column: 5, scope: !3121)
!3128 = !DILocation(line: 1126, column: 5, scope: !3121)
!3129 = !DILocation(line: 1127, column: 5, scope: !3121)
!3130 = !DILocation(line: 1128, column: 5, scope: !3121)
!3131 = !DILocation(line: 1129, column: 5, scope: !3121)
!3132 = !DILocation(line: 1133, column: 1, scope: !3121)
!3133 = distinct !DISubprogram(name: "gpu_kernel_four_host", linkageName: "_ZL20gpu_kernel_four_hostPd", scope: !3, file: !3, line: 1170, type: !2761, scopeLine: 1170, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1163)
!3134 = !DILocalVariable(name: "d_host", arg: 1, scope: !3133, file: !3, line: 1170, type: !99)
!3135 = !DILocation(line: 1170, column: 42, scope: !3133)
!3136 = !DILocation(line: 1174, column: 27, scope: !3133)
!3137 = !DILocation(line: 1175, column: 3, scope: !3133)
!3138 = !DILocation(line: 1176, column: 3, scope: !3133)
!3139 = !DILocation(line: 1174, column: 24, scope: !3133)
!3140 = !DILocation(line: 1174, column: 2, scope: !3133)
!3141 = !DILocation(line: 1177, column: 5, scope: !3133)
!3142 = !DILocation(line: 1178, column: 5, scope: !3133)
!3143 = !DILocation(line: 1179, column: 5, scope: !3133)
!3144 = !DILocation(line: 1180, column: 5, scope: !3133)
!3145 = !DILocation(line: 1181, column: 20, scope: !3133)
!3146 = !DILocation(line: 1182, column: 13, scope: !3133)
!3147 = !DILocation(line: 1182, column: 26, scope: !3133)
!3148 = !DILocation(line: 1182, column: 46, scope: !3133)
!3149 = !DILocation(line: 1182, column: 2, scope: !3133)
!3150 = !DILocalVariable(name: "i", scope: !3151, file: !3, line: 1183, type: !97)
!3151 = distinct !DILexicalBlock(scope: !3133, file: !3, line: 1183, column: 2)
!3152 = !DILocation(line: 1183, column: 10, scope: !3151)
!3153 = !DILocation(line: 1183, column: 6, scope: !3151)
!3154 = !DILocation(line: 1183, column: 15, scope: !3155)
!3155 = distinct !DILexicalBlock(scope: !3151, file: !3, line: 1183, column: 2)
!3156 = !DILocation(line: 1183, column: 17, scope: !3155)
!3157 = !DILocation(line: 1183, column: 16, scope: !3155)
!3158 = !DILocation(line: 1183, column: 2, scope: !3151)
!3159 = !DILocation(line: 1183, column: 74, scope: !3160)
!3160 = distinct !DILexicalBlock(scope: !3155, file: !3, line: 1183, column: 53)
!3161 = !DILocation(line: 1183, column: 86, scope: !3160)
!3162 = !DILocation(line: 1183, column: 72, scope: !3160)
!3163 = !DILocation(line: 1183, column: 89, scope: !3160)
!3164 = !DILocation(line: 1183, column: 50, scope: !3155)
!3165 = !DILocation(line: 1183, column: 2, scope: !3155)
!3166 = distinct !{!3166, !3158, !3167}
!3167 = !DILocation(line: 1183, column: 89, scope: !3151)
!3168 = !DILocation(line: 1184, column: 10, scope: !3133)
!3169 = !DILocation(line: 1184, column: 3, scope: !3133)
!3170 = !DILocation(line: 1184, column: 9, scope: !3133)
!3171 = !DILocation(line: 1188, column: 1, scope: !3133)
!3172 = distinct !DISubprogram(name: "gpu_kernel_five_host", linkageName: "_ZL20gpu_kernel_five_hostd", scope: !3, file: !3, line: 1226, type: !2883, scopeLine: 1226, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1163)
!3173 = !DILocalVariable(name: "alpha_host", arg: 1, scope: !3172, file: !3, line: 1226, type: !100)
!3174 = !DILocation(line: 1226, column: 41, scope: !3172)
!3175 = !DILocation(line: 1230, column: 22, scope: !3172)
!3176 = !DILocation(line: 1231, column: 3, scope: !3172)
!3177 = !DILocation(line: 1230, column: 19, scope: !3172)
!3178 = !DILocation(line: 1230, column: 2, scope: !3172)
!3179 = !DILocation(line: 1232, column: 5, scope: !3172)
!3180 = !DILocation(line: 1233, column: 5, scope: !3172)
!3181 = !DILocation(line: 1234, column: 5, scope: !3172)
!3182 = !DILocation(line: 1235, column: 22, scope: !3172)
!3183 = !DILocation(line: 1236, column: 3, scope: !3172)
!3184 = !DILocation(line: 1235, column: 19, scope: !3172)
!3185 = !DILocation(line: 1235, column: 2, scope: !3172)
!3186 = !DILocation(line: 1237, column: 5, scope: !3172)
!3187 = !DILocation(line: 1238, column: 5, scope: !3172)
!3188 = !DILocation(line: 1239, column: 5, scope: !3172)
!3189 = !DILocation(line: 1243, column: 1, scope: !3172)
!3190 = distinct !DISubprogram(name: "gpu_kernel_six_host", linkageName: "_ZL19gpu_kernel_six_hostPd", scope: !3, file: !3, line: 1261, type: !2761, scopeLine: 1261, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1163)
!3191 = !DILocalVariable(name: "rho_host", arg: 1, scope: !3190, file: !3, line: 1261, type: !99)
!3192 = !DILocation(line: 1261, column: 41, scope: !3190)
!3193 = !DILocation(line: 1265, column: 26, scope: !3190)
!3194 = !DILocation(line: 1266, column: 3, scope: !3190)
!3195 = !DILocation(line: 1267, column: 3, scope: !3190)
!3196 = !DILocation(line: 1265, column: 23, scope: !3190)
!3197 = !DILocation(line: 1265, column: 2, scope: !3190)
!3198 = !DILocation(line: 1268, column: 5, scope: !3190)
!3199 = !DILocation(line: 1269, column: 5, scope: !3190)
!3200 = !DILocation(line: 1270, column: 20, scope: !3190)
!3201 = !DILocation(line: 1271, column: 13, scope: !3190)
!3202 = !DILocation(line: 1271, column: 26, scope: !3190)
!3203 = !DILocation(line: 1271, column: 46, scope: !3190)
!3204 = !DILocation(line: 1271, column: 2, scope: !3190)
!3205 = !DILocalVariable(name: "i", scope: !3206, file: !3, line: 1272, type: !97)
!3206 = distinct !DILexicalBlock(scope: !3190, file: !3, line: 1272, column: 2)
!3207 = !DILocation(line: 1272, column: 10, scope: !3206)
!3208 = !DILocation(line: 1272, column: 6, scope: !3206)
!3209 = !DILocation(line: 1272, column: 15, scope: !3210)
!3210 = distinct !DILexicalBlock(scope: !3206, file: !3, line: 1272, column: 2)
!3211 = !DILocation(line: 1272, column: 17, scope: !3210)
!3212 = !DILocation(line: 1272, column: 16, scope: !3210)
!3213 = !DILocation(line: 1272, column: 2, scope: !3206)
!3214 = !DILocation(line: 1272, column: 73, scope: !3215)
!3215 = distinct !DILexicalBlock(scope: !3210, file: !3, line: 1272, column: 52)
!3216 = !DILocation(line: 1272, column: 85, scope: !3215)
!3217 = !DILocation(line: 1272, column: 71, scope: !3215)
!3218 = !DILocation(line: 1272, column: 88, scope: !3215)
!3219 = !DILocation(line: 1272, column: 49, scope: !3210)
!3220 = !DILocation(line: 1272, column: 2, scope: !3210)
!3221 = distinct !{!3221, !3213, !3222}
!3222 = !DILocation(line: 1272, column: 88, scope: !3206)
!3223 = !DILocation(line: 1273, column: 12, scope: !3190)
!3224 = !DILocation(line: 1273, column: 3, scope: !3190)
!3225 = !DILocation(line: 1273, column: 11, scope: !3190)
!3226 = !DILocation(line: 1277, column: 1, scope: !3190)
!3227 = distinct !DISubprogram(name: "gpu_kernel_seven_host", linkageName: "_ZL21gpu_kernel_seven_hostd", scope: !3, file: !3, line: 1307, type: !2883, scopeLine: 1307, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1163)
!3228 = !DILocalVariable(name: "beta_host", arg: 1, scope: !3227, file: !3, line: 1307, type: !100)
!3229 = !DILocation(line: 1307, column: 42, scope: !3227)
!3230 = !DILocation(line: 1311, column: 28, scope: !3227)
!3231 = !DILocation(line: 1312, column: 3, scope: !3227)
!3232 = !DILocation(line: 1311, column: 25, scope: !3227)
!3233 = !DILocation(line: 1311, column: 2, scope: !3227)
!3234 = !DILocation(line: 1313, column: 5, scope: !3227)
!3235 = !DILocation(line: 1314, column: 5, scope: !3227)
!3236 = !DILocation(line: 1315, column: 5, scope: !3227)
!3237 = !DILocation(line: 1319, column: 1, scope: !3227)
!3238 = distinct !DISubprogram(name: "gpu_kernel_eight_host", linkageName: "_ZL21gpu_kernel_eight_hostv", scope: !3, file: !3, line: 1329, type: !666, scopeLine: 1329, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1163)
!3239 = !DILocation(line: 1333, column: 28, scope: !3238)
!3240 = !DILocation(line: 1334, column: 3, scope: !3238)
!3241 = !DILocation(line: 1335, column: 3, scope: !3238)
!3242 = !DILocation(line: 1333, column: 25, scope: !3238)
!3243 = !DILocation(line: 1333, column: 2, scope: !3238)
!3244 = !DILocation(line: 1336, column: 5, scope: !3238)
!3245 = !DILocation(line: 1337, column: 5, scope: !3238)
!3246 = !DILocation(line: 1338, column: 5, scope: !3238)
!3247 = !DILocation(line: 1339, column: 5, scope: !3238)
!3248 = !DILocation(line: 1340, column: 5, scope: !3238)
!3249 = !DILocation(line: 1344, column: 1, scope: !3238)
!3250 = distinct !DISubprogram(name: "gpu_kernel_nine_host", linkageName: "_ZL20gpu_kernel_nine_hostPd", scope: !3, file: !3, line: 1381, type: !2761, scopeLine: 1381, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1163)
!3251 = !DILocalVariable(name: "sum_host", arg: 1, scope: !3250, file: !3, line: 1381, type: !99)
!3252 = !DILocation(line: 1381, column: 42, scope: !3250)
!3253 = !DILocation(line: 1385, column: 27, scope: !3250)
!3254 = !DILocation(line: 1386, column: 3, scope: !3250)
!3255 = !DILocation(line: 1387, column: 3, scope: !3250)
!3256 = !DILocation(line: 1385, column: 24, scope: !3250)
!3257 = !DILocation(line: 1385, column: 2, scope: !3250)
!3258 = !DILocation(line: 1388, column: 5, scope: !3250)
!3259 = !DILocation(line: 1389, column: 5, scope: !3250)
!3260 = !DILocation(line: 1390, column: 5, scope: !3250)
!3261 = !DILocation(line: 1391, column: 5, scope: !3250)
!3262 = !DILocation(line: 1392, column: 20, scope: !3250)
!3263 = !DILocation(line: 1393, column: 13, scope: !3250)
!3264 = !DILocation(line: 1393, column: 26, scope: !3250)
!3265 = !DILocation(line: 1393, column: 46, scope: !3250)
!3266 = !DILocation(line: 1393, column: 2, scope: !3250)
!3267 = !DILocalVariable(name: "i", scope: !3268, file: !3, line: 1394, type: !97)
!3268 = distinct !DILexicalBlock(scope: !3250, file: !3, line: 1394, column: 2)
!3269 = !DILocation(line: 1394, column: 10, scope: !3268)
!3270 = !DILocation(line: 1394, column: 6, scope: !3268)
!3271 = !DILocation(line: 1394, column: 15, scope: !3272)
!3272 = distinct !DILexicalBlock(scope: !3268, file: !3, line: 1394, column: 2)
!3273 = !DILocation(line: 1394, column: 17, scope: !3272)
!3274 = !DILocation(line: 1394, column: 16, scope: !3272)
!3275 = !DILocation(line: 1394, column: 2, scope: !3268)
!3276 = !DILocation(line: 1394, column: 74, scope: !3277)
!3277 = distinct !DILexicalBlock(scope: !3272, file: !3, line: 1394, column: 53)
!3278 = !DILocation(line: 1394, column: 86, scope: !3277)
!3279 = !DILocation(line: 1394, column: 72, scope: !3277)
!3280 = !DILocation(line: 1394, column: 89, scope: !3277)
!3281 = !DILocation(line: 1394, column: 50, scope: !3272)
!3282 = !DILocation(line: 1394, column: 2, scope: !3272)
!3283 = distinct !{!3283, !3275, !3284}
!3284 = !DILocation(line: 1394, column: 89, scope: !3268)
!3285 = !DILocation(line: 1395, column: 12, scope: !3250)
!3286 = !DILocation(line: 1395, column: 3, scope: !3250)
!3287 = !DILocation(line: 1395, column: 11, scope: !3250)
!3288 = !DILocation(line: 1399, column: 1, scope: !3250)
!3289 = distinct !DISubprogram(name: "dim3", linkageName: "_ZN4dim3C2Ejjj", scope: !3291, file: !3290, line: 421, type: !3297, scopeLine: 421, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, declaration: !3296, retainedNodes: !1163)
!3290 = !DIFile(filename: "/usr/local/cuda/include/vector_types.h", directory: "")
!3291 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "dim3", file: !3290, line: 417, size: 96, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !3292, identifier: "_ZTS4dim3")
!3292 = !{!3293, !3294, !3295, !3296, !3300, !3309}
!3293 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !3291, file: !3290, line: 419, baseType: !7, size: 32)
!3294 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !3291, file: !3290, line: 419, baseType: !7, size: 32, offset: 32)
!3295 = !DIDerivedType(tag: DW_TAG_member, name: "z", scope: !3291, file: !3290, line: 419, baseType: !7, size: 32, offset: 64)
!3296 = !DISubprogram(name: "dim3", scope: !3291, file: !3290, line: 421, type: !3297, scopeLine: 421, flags: DIFlagPrototyped, spFlags: 0)
!3297 = !DISubroutineType(types: !3298)
!3298 = !{null, !3299, !7, !7, !7}
!3299 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !3291, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!3300 = !DISubprogram(name: "dim3", scope: !3291, file: !3290, line: 422, type: !3301, scopeLine: 422, flags: DIFlagPrototyped, spFlags: 0)
!3301 = !DISubroutineType(types: !3302)
!3302 = !{null, !3299, !3303}
!3303 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint3", file: !3290, line: 383, baseType: !3304)
!3304 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "uint3", file: !3290, line: 190, size: 96, flags: DIFlagTypePassByValue, elements: !3305, identifier: "_ZTS5uint3")
!3305 = !{!3306, !3307, !3308}
!3306 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !3304, file: !3290, line: 192, baseType: !7, size: 32)
!3307 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !3304, file: !3290, line: 192, baseType: !7, size: 32, offset: 32)
!3308 = !DIDerivedType(tag: DW_TAG_member, name: "z", scope: !3304, file: !3290, line: 192, baseType: !7, size: 32, offset: 64)
!3309 = !DISubprogram(name: "operator uint3", linkageName: "_ZN4dim3cv5uint3Ev", scope: !3291, file: !3290, line: 423, type: !3310, scopeLine: 423, flags: DIFlagPrototyped, spFlags: 0)
!3310 = !DISubroutineType(types: !3311)
!3311 = !{!3303, !3299}
!3312 = !DILocalVariable(name: "this", arg: 1, scope: !3289, type: !3313, flags: DIFlagArtificial | DIFlagObjectPointer)
!3313 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !3291, size: 64)
!3314 = !DILocation(line: 0, scope: !3289)
!3315 = !DILocalVariable(name: "vx", arg: 2, scope: !3289, file: !3290, line: 421, type: !7)
!3316 = !DILocation(line: 421, column: 43, scope: !3289)
!3317 = !DILocalVariable(name: "vy", arg: 3, scope: !3289, file: !3290, line: 421, type: !7)
!3318 = !DILocation(line: 421, column: 64, scope: !3289)
!3319 = !DILocalVariable(name: "vz", arg: 4, scope: !3289, file: !3290, line: 421, type: !7)
!3320 = !DILocation(line: 421, column: 85, scope: !3289)
!3321 = !DILocation(line: 421, column: 95, scope: !3289)
!3322 = !DILocation(line: 421, column: 97, scope: !3289)
!3323 = !DILocation(line: 421, column: 102, scope: !3289)
!3324 = !DILocation(line: 421, column: 104, scope: !3289)
!3325 = !DILocation(line: 421, column: 109, scope: !3289)
!3326 = !DILocation(line: 421, column: 111, scope: !3289)
!3327 = !DILocation(line: 421, column: 116, scope: !3289)
!3328 = distinct !DISubprogram(name: "sprnvc", linkageName: "_ZL6sprnvciiiPdPi", scope: !3, file: !3, line: 2073, type: !3329, scopeLine: 2073, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1163)
!3329 = !DISubroutineType(types: !3330)
!3330 = !{null, !97, !97, !97, !99, !98}
!3331 = !DILocalVariable(name: "n", arg: 1, scope: !3328, file: !3, line: 2073, type: !97)
!3332 = !DILocation(line: 2073, column: 24, scope: !3328)
!3333 = !DILocalVariable(name: "nz", arg: 2, scope: !3328, file: !3, line: 2073, type: !97)
!3334 = !DILocation(line: 2073, column: 31, scope: !3328)
!3335 = !DILocalVariable(name: "nn1", arg: 3, scope: !3328, file: !3, line: 2073, type: !97)
!3336 = !DILocation(line: 2073, column: 39, scope: !3328)
!3337 = !DILocalVariable(name: "v", arg: 4, scope: !3328, file: !3, line: 2073, type: !99)
!3338 = !DILocation(line: 2073, column: 51, scope: !3328)
!3339 = !DILocalVariable(name: "iv", arg: 5, scope: !3328, file: !3, line: 2073, type: !98)
!3340 = !DILocation(line: 2073, column: 60, scope: !3328)
!3341 = !DILocalVariable(name: "nzv", scope: !3328, file: !3, line: 2074, type: !97)
!3342 = !DILocation(line: 2074, column: 6, scope: !3328)
!3343 = !DILocalVariable(name: "ii", scope: !3328, file: !3, line: 2074, type: !97)
!3344 = !DILocation(line: 2074, column: 11, scope: !3328)
!3345 = !DILocalVariable(name: "i", scope: !3328, file: !3, line: 2074, type: !97)
!3346 = !DILocation(line: 2074, column: 15, scope: !3328)
!3347 = !DILocalVariable(name: "vecelt", scope: !3328, file: !3, line: 2075, type: !100)
!3348 = !DILocation(line: 2075, column: 9, scope: !3328)
!3349 = !DILocalVariable(name: "vecloc", scope: !3328, file: !3, line: 2075, type: !100)
!3350 = !DILocation(line: 2075, column: 17, scope: !3328)
!3351 = !DILocation(line: 2077, column: 6, scope: !3328)
!3352 = !DILocation(line: 2079, column: 2, scope: !3328)
!3353 = !DILocation(line: 2079, column: 8, scope: !3328)
!3354 = !DILocation(line: 2079, column: 14, scope: !3328)
!3355 = !DILocation(line: 2079, column: 12, scope: !3328)
!3356 = !DILocation(line: 2080, column: 26, scope: !3357)
!3357 = distinct !DILexicalBlock(scope: !3328, file: !3, line: 2079, column: 17)
!3358 = !DILocation(line: 2080, column: 12, scope: !3357)
!3359 = !DILocation(line: 2080, column: 10, scope: !3357)
!3360 = !DILocation(line: 2087, column: 26, scope: !3357)
!3361 = !DILocation(line: 2087, column: 12, scope: !3357)
!3362 = !DILocation(line: 2087, column: 10, scope: !3357)
!3363 = !DILocation(line: 2088, column: 14, scope: !3357)
!3364 = !DILocation(line: 2088, column: 22, scope: !3357)
!3365 = !DILocation(line: 2088, column: 7, scope: !3357)
!3366 = !DILocation(line: 2088, column: 27, scope: !3357)
!3367 = !DILocation(line: 2088, column: 5, scope: !3357)
!3368 = !DILocation(line: 2089, column: 6, scope: !3369)
!3369 = distinct !DILexicalBlock(scope: !3357, file: !3, line: 2089, column: 6)
!3370 = !DILocation(line: 2089, column: 8, scope: !3369)
!3371 = !DILocation(line: 2089, column: 7, scope: !3369)
!3372 = !DILocation(line: 2089, column: 6, scope: !3357)
!3373 = !DILocation(line: 2089, column: 11, scope: !3374)
!3374 = distinct !DILexicalBlock(scope: !3369, file: !3, line: 2089, column: 10)
!3375 = distinct !{!3375, !3352, !3376}
!3376 = !DILocation(line: 2107, column: 2, scope: !3328)
!3377 = !DILocalVariable(name: "was_gen", scope: !3357, file: !3, line: 2096, type: !1526)
!3378 = !DILocation(line: 2096, column: 11, scope: !3357)
!3379 = !DILocation(line: 2097, column: 10, scope: !3380)
!3380 = distinct !DILexicalBlock(scope: !3357, file: !3, line: 2097, column: 3)
!3381 = !DILocation(line: 2097, column: 7, scope: !3380)
!3382 = !DILocation(line: 2097, column: 15, scope: !3383)
!3383 = distinct !DILexicalBlock(scope: !3380, file: !3, line: 2097, column: 3)
!3384 = !DILocation(line: 2097, column: 20, scope: !3383)
!3385 = !DILocation(line: 2097, column: 18, scope: !3383)
!3386 = !DILocation(line: 2097, column: 3, scope: !3380)
!3387 = !DILocation(line: 2098, column: 7, scope: !3388)
!3388 = distinct !DILexicalBlock(scope: !3389, file: !3, line: 2098, column: 7)
!3389 = distinct !DILexicalBlock(scope: !3383, file: !3, line: 2097, column: 30)
!3390 = !DILocation(line: 2098, column: 10, scope: !3388)
!3391 = !DILocation(line: 2098, column: 17, scope: !3388)
!3392 = !DILocation(line: 2098, column: 14, scope: !3388)
!3393 = !DILocation(line: 2098, column: 7, scope: !3389)
!3394 = !DILocation(line: 2099, column: 13, scope: !3395)
!3395 = distinct !DILexicalBlock(scope: !3388, file: !3, line: 2098, column: 19)
!3396 = !DILocation(line: 2100, column: 5, scope: !3395)
!3397 = !DILocation(line: 2102, column: 3, scope: !3389)
!3398 = !DILocation(line: 2097, column: 27, scope: !3383)
!3399 = !DILocation(line: 2097, column: 3, scope: !3383)
!3400 = distinct !{!3400, !3386, !3401}
!3401 = !DILocation(line: 2102, column: 3, scope: !3380)
!3402 = !DILocation(line: 2103, column: 6, scope: !3403)
!3403 = distinct !DILexicalBlock(scope: !3357, file: !3, line: 2103, column: 6)
!3404 = !DILocation(line: 2103, column: 6, scope: !3357)
!3405 = !DILocation(line: 2103, column: 15, scope: !3406)
!3406 = distinct !DILexicalBlock(scope: !3403, file: !3, line: 2103, column: 14)
!3407 = !DILocation(line: 2104, column: 12, scope: !3357)
!3408 = !DILocation(line: 2104, column: 3, scope: !3357)
!3409 = !DILocation(line: 2104, column: 5, scope: !3357)
!3410 = !DILocation(line: 2104, column: 10, scope: !3357)
!3411 = !DILocation(line: 2105, column: 13, scope: !3357)
!3412 = !DILocation(line: 2105, column: 3, scope: !3357)
!3413 = !DILocation(line: 2105, column: 6, scope: !3357)
!3414 = !DILocation(line: 2105, column: 11, scope: !3357)
!3415 = !DILocation(line: 2106, column: 9, scope: !3357)
!3416 = !DILocation(line: 2106, column: 13, scope: !3357)
!3417 = !DILocation(line: 2106, column: 7, scope: !3357)
!3418 = !DILocation(line: 2108, column: 1, scope: !3328)
!3419 = distinct !DISubprogram(name: "vecset", linkageName: "_ZL6vecsetiPdPiS0_id", scope: !3, file: !3, line: 2116, type: !3420, scopeLine: 2116, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1163)
!3420 = !DISubroutineType(types: !3421)
!3421 = !{null, !97, !99, !98, !98, !97, !100}
!3422 = !DILocalVariable(name: "n", arg: 1, scope: !3419, file: !3, line: 2116, type: !97)
!3423 = !DILocation(line: 2116, column: 24, scope: !3419)
!3424 = !DILocalVariable(name: "v", arg: 2, scope: !3419, file: !3, line: 2116, type: !99)
!3425 = !DILocation(line: 2116, column: 34, scope: !3419)
!3426 = !DILocalVariable(name: "iv", arg: 3, scope: !3419, file: !3, line: 2116, type: !98)
!3427 = !DILocation(line: 2116, column: 43, scope: !3419)
!3428 = !DILocalVariable(name: "nzv", arg: 4, scope: !3419, file: !3, line: 2116, type: !98)
!3429 = !DILocation(line: 2116, column: 54, scope: !3419)
!3430 = !DILocalVariable(name: "i", arg: 5, scope: !3419, file: !3, line: 2116, type: !97)
!3431 = !DILocation(line: 2116, column: 63, scope: !3419)
!3432 = !DILocalVariable(name: "val", arg: 6, scope: !3419, file: !3, line: 2116, type: !100)
!3433 = !DILocation(line: 2116, column: 73, scope: !3419)
!3434 = !DILocalVariable(name: "k", scope: !3419, file: !3, line: 2117, type: !97)
!3435 = !DILocation(line: 2117, column: 6, scope: !3419)
!3436 = !DILocalVariable(name: "set", scope: !3419, file: !3, line: 2118, type: !1526)
!3437 = !DILocation(line: 2118, column: 10, scope: !3419)
!3438 = !DILocation(line: 2120, column: 6, scope: !3419)
!3439 = !DILocation(line: 2121, column: 8, scope: !3440)
!3440 = distinct !DILexicalBlock(scope: !3419, file: !3, line: 2121, column: 2)
!3441 = !DILocation(line: 2121, column: 6, scope: !3440)
!3442 = !DILocation(line: 2121, column: 13, scope: !3443)
!3443 = distinct !DILexicalBlock(scope: !3440, file: !3, line: 2121, column: 2)
!3444 = !DILocation(line: 2121, column: 18, scope: !3443)
!3445 = !DILocation(line: 2121, column: 17, scope: !3443)
!3446 = !DILocation(line: 2121, column: 15, scope: !3443)
!3447 = !DILocation(line: 2121, column: 2, scope: !3440)
!3448 = !DILocation(line: 2122, column: 6, scope: !3449)
!3449 = distinct !DILexicalBlock(scope: !3450, file: !3, line: 2122, column: 6)
!3450 = distinct !DILexicalBlock(scope: !3443, file: !3, line: 2121, column: 27)
!3451 = !DILocation(line: 2122, column: 9, scope: !3449)
!3452 = !DILocation(line: 2122, column: 15, scope: !3449)
!3453 = !DILocation(line: 2122, column: 12, scope: !3449)
!3454 = !DILocation(line: 2122, column: 6, scope: !3450)
!3455 = !DILocation(line: 2123, column: 11, scope: !3456)
!3456 = distinct !DILexicalBlock(scope: !3449, file: !3, line: 2122, column: 17)
!3457 = !DILocation(line: 2123, column: 4, scope: !3456)
!3458 = !DILocation(line: 2123, column: 6, scope: !3456)
!3459 = !DILocation(line: 2123, column: 9, scope: !3456)
!3460 = !DILocation(line: 2124, column: 9, scope: !3456)
!3461 = !DILocation(line: 2125, column: 3, scope: !3456)
!3462 = !DILocation(line: 2126, column: 2, scope: !3450)
!3463 = !DILocation(line: 2121, column: 24, scope: !3443)
!3464 = !DILocation(line: 2121, column: 2, scope: !3443)
!3465 = distinct !{!3465, !3447, !3466}
!3466 = !DILocation(line: 2126, column: 2, scope: !3440)
!3467 = !DILocation(line: 2127, column: 5, scope: !3468)
!3468 = distinct !DILexicalBlock(scope: !3419, file: !3, line: 2127, column: 5)
!3469 = !DILocation(line: 2127, column: 9, scope: !3468)
!3470 = !DILocation(line: 2127, column: 5, scope: !3419)
!3471 = !DILocation(line: 2128, column: 14, scope: !3472)
!3472 = distinct !DILexicalBlock(scope: !3468, file: !3, line: 2127, column: 18)
!3473 = !DILocation(line: 2128, column: 3, scope: !3472)
!3474 = !DILocation(line: 2128, column: 6, scope: !3472)
!3475 = !DILocation(line: 2128, column: 5, scope: !3472)
!3476 = !DILocation(line: 2128, column: 12, scope: !3472)
!3477 = !DILocation(line: 2129, column: 14, scope: !3472)
!3478 = !DILocation(line: 2129, column: 3, scope: !3472)
!3479 = !DILocation(line: 2129, column: 7, scope: !3472)
!3480 = !DILocation(line: 2129, column: 6, scope: !3472)
!3481 = !DILocation(line: 2129, column: 12, scope: !3472)
!3482 = !DILocation(line: 2130, column: 15, scope: !3472)
!3483 = !DILocation(line: 2130, column: 14, scope: !3472)
!3484 = !DILocation(line: 2130, column: 19, scope: !3472)
!3485 = !DILocation(line: 2130, column: 4, scope: !3472)
!3486 = !DILocation(line: 2130, column: 12, scope: !3472)
!3487 = !DILocation(line: 2131, column: 2, scope: !3472)
!3488 = !DILocation(line: 2132, column: 1, scope: !3419)
!3489 = distinct !DISubprogram(name: "sparse", linkageName: "_ZL6sparsePdPiS0_iiiS0_PA12_iPA12_diiS0_dd", scope: !3, file: !3, line: 1886, type: !3490, scopeLine: 1899, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1163)
!3490 = !DISubroutineType(types: !3491)
!3491 = !{null, !99, !98, !98, !97, !97, !97, !98, !101, !106, !97, !97, !98, !100, !100}
!3492 = !DILocalVariable(name: "a", arg: 1, scope: !3489, file: !3, line: 1886, type: !99)
!3493 = !DILocation(line: 1886, column: 27, scope: !3489)
!3494 = !DILocalVariable(name: "colidx", arg: 2, scope: !3489, file: !3, line: 1887, type: !98)
!3495 = !DILocation(line: 1887, column: 7, scope: !3489)
!3496 = !DILocalVariable(name: "rowstr", arg: 3, scope: !3489, file: !3, line: 1888, type: !98)
!3497 = !DILocation(line: 1888, column: 7, scope: !3489)
!3498 = !DILocalVariable(name: "n", arg: 4, scope: !3489, file: !3, line: 1889, type: !97)
!3499 = !DILocation(line: 1889, column: 7, scope: !3489)
!3500 = !DILocalVariable(name: "nz", arg: 5, scope: !3489, file: !3, line: 1890, type: !97)
!3501 = !DILocation(line: 1890, column: 7, scope: !3489)
!3502 = !DILocalVariable(name: "nozer", arg: 6, scope: !3489, file: !3, line: 1891, type: !97)
!3503 = !DILocation(line: 1891, column: 7, scope: !3489)
!3504 = !DILocalVariable(name: "arow", arg: 7, scope: !3489, file: !3, line: 1892, type: !98)
!3505 = !DILocation(line: 1892, column: 7, scope: !3489)
!3506 = !DILocalVariable(name: "acol", arg: 8, scope: !3489, file: !3, line: 1893, type: !101)
!3507 = !DILocation(line: 1893, column: 7, scope: !3489)
!3508 = !DILocalVariable(name: "aelt", arg: 9, scope: !3489, file: !3, line: 1894, type: !106)
!3509 = !DILocation(line: 1894, column: 10, scope: !3489)
!3510 = !DILocalVariable(name: "firstrow", arg: 10, scope: !3489, file: !3, line: 1895, type: !97)
!3511 = !DILocation(line: 1895, column: 7, scope: !3489)
!3512 = !DILocalVariable(name: "lastrow", arg: 11, scope: !3489, file: !3, line: 1896, type: !97)
!3513 = !DILocation(line: 1896, column: 7, scope: !3489)
!3514 = !DILocalVariable(name: "nzloc", arg: 12, scope: !3489, file: !3, line: 1897, type: !98)
!3515 = !DILocation(line: 1897, column: 7, scope: !3489)
!3516 = !DILocalVariable(name: "rcond", arg: 13, scope: !3489, file: !3, line: 1898, type: !100)
!3517 = !DILocation(line: 1898, column: 10, scope: !3489)
!3518 = !DILocalVariable(name: "shift", arg: 14, scope: !3489, file: !3, line: 1899, type: !100)
!3519 = !DILocation(line: 1899, column: 10, scope: !3489)
!3520 = !DILocalVariable(name: "nrows", scope: !3489, file: !3, line: 1900, type: !97)
!3521 = !DILocation(line: 1900, column: 6, scope: !3489)
!3522 = !DILocalVariable(name: "i", scope: !3489, file: !3, line: 1908, type: !97)
!3523 = !DILocation(line: 1908, column: 6, scope: !3489)
!3524 = !DILocalVariable(name: "j", scope: !3489, file: !3, line: 1908, type: !97)
!3525 = !DILocation(line: 1908, column: 9, scope: !3489)
!3526 = !DILocalVariable(name: "j1", scope: !3489, file: !3, line: 1908, type: !97)
!3527 = !DILocation(line: 1908, column: 12, scope: !3489)
!3528 = !DILocalVariable(name: "j2", scope: !3489, file: !3, line: 1908, type: !97)
!3529 = !DILocation(line: 1908, column: 16, scope: !3489)
!3530 = !DILocalVariable(name: "nza", scope: !3489, file: !3, line: 1908, type: !97)
!3531 = !DILocation(line: 1908, column: 20, scope: !3489)
!3532 = !DILocalVariable(name: "k", scope: !3489, file: !3, line: 1908, type: !97)
!3533 = !DILocation(line: 1908, column: 25, scope: !3489)
!3534 = !DILocalVariable(name: "kk", scope: !3489, file: !3, line: 1908, type: !97)
!3535 = !DILocation(line: 1908, column: 28, scope: !3489)
!3536 = !DILocalVariable(name: "nzrow", scope: !3489, file: !3, line: 1908, type: !97)
!3537 = !DILocation(line: 1908, column: 32, scope: !3489)
!3538 = !DILocalVariable(name: "jcol", scope: !3489, file: !3, line: 1908, type: !97)
!3539 = !DILocation(line: 1908, column: 39, scope: !3489)
!3540 = !DILocalVariable(name: "size", scope: !3489, file: !3, line: 1909, type: !100)
!3541 = !DILocation(line: 1909, column: 9, scope: !3489)
!3542 = !DILocalVariable(name: "scale", scope: !3489, file: !3, line: 1909, type: !100)
!3543 = !DILocation(line: 1909, column: 15, scope: !3489)
!3544 = !DILocalVariable(name: "ratio", scope: !3489, file: !3, line: 1909, type: !100)
!3545 = !DILocation(line: 1909, column: 22, scope: !3489)
!3546 = !DILocalVariable(name: "va", scope: !3489, file: !3, line: 1909, type: !100)
!3547 = !DILocation(line: 1909, column: 29, scope: !3489)
!3548 = !DILocation(line: 1917, column: 10, scope: !3489)
!3549 = !DILocation(line: 1917, column: 20, scope: !3489)
!3550 = !DILocation(line: 1917, column: 18, scope: !3489)
!3551 = !DILocation(line: 1917, column: 29, scope: !3489)
!3552 = !DILocation(line: 1917, column: 8, scope: !3489)
!3553 = !DILocation(line: 1924, column: 8, scope: !3554)
!3554 = distinct !DILexicalBlock(scope: !3489, file: !3, line: 1924, column: 2)
!3555 = !DILocation(line: 1924, column: 6, scope: !3554)
!3556 = !DILocation(line: 1924, column: 13, scope: !3557)
!3557 = distinct !DILexicalBlock(scope: !3554, file: !3, line: 1924, column: 2)
!3558 = !DILocation(line: 1924, column: 17, scope: !3557)
!3559 = !DILocation(line: 1924, column: 22, scope: !3557)
!3560 = !DILocation(line: 1924, column: 15, scope: !3557)
!3561 = !DILocation(line: 1924, column: 2, scope: !3554)
!3562 = !DILocation(line: 1925, column: 3, scope: !3563)
!3563 = distinct !DILexicalBlock(scope: !3557, file: !3, line: 1924, column: 30)
!3564 = !DILocation(line: 1925, column: 10, scope: !3563)
!3565 = !DILocation(line: 1925, column: 13, scope: !3563)
!3566 = !DILocation(line: 1926, column: 2, scope: !3563)
!3567 = !DILocation(line: 1924, column: 27, scope: !3557)
!3568 = !DILocation(line: 1924, column: 2, scope: !3557)
!3569 = distinct !{!3569, !3561, !3570}
!3570 = !DILocation(line: 1926, column: 2, scope: !3554)
!3571 = !DILocation(line: 1927, column: 8, scope: !3572)
!3572 = distinct !DILexicalBlock(scope: !3489, file: !3, line: 1927, column: 2)
!3573 = !DILocation(line: 1927, column: 6, scope: !3572)
!3574 = !DILocation(line: 1927, column: 13, scope: !3575)
!3575 = distinct !DILexicalBlock(scope: !3572, file: !3, line: 1927, column: 2)
!3576 = !DILocation(line: 1927, column: 17, scope: !3575)
!3577 = !DILocation(line: 1927, column: 15, scope: !3575)
!3578 = !DILocation(line: 1927, column: 2, scope: !3572)
!3579 = !DILocation(line: 1928, column: 11, scope: !3580)
!3580 = distinct !DILexicalBlock(scope: !3581, file: !3, line: 1928, column: 3)
!3581 = distinct !DILexicalBlock(scope: !3575, file: !3, line: 1927, column: 24)
!3582 = !DILocation(line: 1928, column: 7, scope: !3580)
!3583 = !DILocation(line: 1928, column: 16, scope: !3584)
!3584 = distinct !DILexicalBlock(scope: !3580, file: !3, line: 1928, column: 3)
!3585 = !DILocation(line: 1928, column: 22, scope: !3584)
!3586 = !DILocation(line: 1928, column: 27, scope: !3584)
!3587 = !DILocation(line: 1928, column: 20, scope: !3584)
!3588 = !DILocation(line: 1928, column: 3, scope: !3580)
!3589 = !DILocation(line: 1929, column: 8, scope: !3590)
!3590 = distinct !DILexicalBlock(scope: !3584, file: !3, line: 1928, column: 37)
!3591 = !DILocation(line: 1929, column: 13, scope: !3590)
!3592 = !DILocation(line: 1929, column: 16, scope: !3590)
!3593 = !DILocation(line: 1929, column: 21, scope: !3590)
!3594 = !DILocation(line: 1929, column: 6, scope: !3590)
!3595 = !DILocation(line: 1930, column: 16, scope: !3590)
!3596 = !DILocation(line: 1930, column: 23, scope: !3590)
!3597 = !DILocation(line: 1930, column: 28, scope: !3590)
!3598 = !DILocation(line: 1930, column: 33, scope: !3590)
!3599 = !DILocation(line: 1930, column: 26, scope: !3590)
!3600 = !DILocation(line: 1930, column: 4, scope: !3590)
!3601 = !DILocation(line: 1930, column: 11, scope: !3590)
!3602 = !DILocation(line: 1930, column: 14, scope: !3590)
!3603 = !DILocation(line: 1931, column: 3, scope: !3590)
!3604 = !DILocation(line: 1928, column: 34, scope: !3584)
!3605 = !DILocation(line: 1928, column: 3, scope: !3584)
!3606 = distinct !{!3606, !3588, !3607}
!3607 = !DILocation(line: 1931, column: 3, scope: !3580)
!3608 = !DILocation(line: 1932, column: 2, scope: !3581)
!3609 = !DILocation(line: 1927, column: 21, scope: !3575)
!3610 = !DILocation(line: 1927, column: 2, scope: !3575)
!3611 = distinct !{!3611, !3578, !3612}
!3612 = !DILocation(line: 1932, column: 2, scope: !3572)
!3613 = !DILocation(line: 1933, column: 2, scope: !3489)
!3614 = !DILocation(line: 1933, column: 12, scope: !3489)
!3615 = !DILocation(line: 1934, column: 8, scope: !3616)
!3616 = distinct !DILexicalBlock(scope: !3489, file: !3, line: 1934, column: 2)
!3617 = !DILocation(line: 1934, column: 6, scope: !3616)
!3618 = !DILocation(line: 1934, column: 13, scope: !3619)
!3619 = distinct !DILexicalBlock(scope: !3616, file: !3, line: 1934, column: 2)
!3620 = !DILocation(line: 1934, column: 17, scope: !3619)
!3621 = !DILocation(line: 1934, column: 22, scope: !3619)
!3622 = !DILocation(line: 1934, column: 15, scope: !3619)
!3623 = !DILocation(line: 1934, column: 2, scope: !3616)
!3624 = !DILocation(line: 1935, column: 15, scope: !3625)
!3625 = distinct !DILexicalBlock(scope: !3619, file: !3, line: 1934, column: 30)
!3626 = !DILocation(line: 1935, column: 22, scope: !3625)
!3627 = !DILocation(line: 1935, column: 27, scope: !3625)
!3628 = !DILocation(line: 1935, column: 34, scope: !3625)
!3629 = !DILocation(line: 1935, column: 35, scope: !3625)
!3630 = !DILocation(line: 1935, column: 25, scope: !3625)
!3631 = !DILocation(line: 1935, column: 3, scope: !3625)
!3632 = !DILocation(line: 1935, column: 10, scope: !3625)
!3633 = !DILocation(line: 1935, column: 13, scope: !3625)
!3634 = !DILocation(line: 1936, column: 2, scope: !3625)
!3635 = !DILocation(line: 1934, column: 27, scope: !3619)
!3636 = !DILocation(line: 1934, column: 2, scope: !3619)
!3637 = distinct !{!3637, !3623, !3638}
!3638 = !DILocation(line: 1936, column: 2, scope: !3616)
!3639 = !DILocation(line: 1937, column: 8, scope: !3489)
!3640 = !DILocation(line: 1937, column: 15, scope: !3489)
!3641 = !DILocation(line: 1937, column: 22, scope: !3489)
!3642 = !DILocation(line: 1937, column: 6, scope: !3489)
!3643 = !DILocation(line: 1945, column: 5, scope: !3644)
!3644 = distinct !DILexicalBlock(scope: !3489, file: !3, line: 1945, column: 5)
!3645 = !DILocation(line: 1945, column: 11, scope: !3644)
!3646 = !DILocation(line: 1945, column: 9, scope: !3644)
!3647 = !DILocation(line: 1945, column: 5, scope: !3489)
!3648 = !DILocation(line: 1946, column: 3, scope: !3649)
!3649 = distinct !DILexicalBlock(scope: !3644, file: !3, line: 1945, column: 14)
!3650 = !DILocation(line: 1947, column: 35, scope: !3649)
!3651 = !DILocation(line: 1947, column: 40, scope: !3649)
!3652 = !DILocation(line: 1947, column: 3, scope: !3649)
!3653 = !DILocation(line: 1948, column: 3, scope: !3649)
!3654 = !DILocation(line: 1956, column: 8, scope: !3655)
!3655 = distinct !DILexicalBlock(scope: !3489, file: !3, line: 1956, column: 2)
!3656 = !DILocation(line: 1956, column: 6, scope: !3655)
!3657 = !DILocation(line: 1956, column: 13, scope: !3658)
!3658 = distinct !DILexicalBlock(scope: !3655, file: !3, line: 1956, column: 2)
!3659 = !DILocation(line: 1956, column: 17, scope: !3658)
!3660 = !DILocation(line: 1956, column: 15, scope: !3658)
!3661 = !DILocation(line: 1956, column: 2, scope: !3655)
!3662 = !DILocation(line: 1957, column: 11, scope: !3663)
!3663 = distinct !DILexicalBlock(scope: !3664, file: !3, line: 1957, column: 3)
!3664 = distinct !DILexicalBlock(scope: !3658, file: !3, line: 1956, column: 28)
!3665 = !DILocation(line: 1957, column: 18, scope: !3663)
!3666 = !DILocation(line: 1957, column: 9, scope: !3663)
!3667 = !DILocation(line: 1957, column: 7, scope: !3663)
!3668 = !DILocation(line: 1957, column: 22, scope: !3669)
!3669 = distinct !DILexicalBlock(scope: !3663, file: !3, line: 1957, column: 3)
!3670 = !DILocation(line: 1957, column: 26, scope: !3669)
!3671 = !DILocation(line: 1957, column: 33, scope: !3669)
!3672 = !DILocation(line: 1957, column: 34, scope: !3669)
!3673 = !DILocation(line: 1957, column: 24, scope: !3669)
!3674 = !DILocation(line: 1957, column: 3, scope: !3663)
!3675 = !DILocation(line: 1958, column: 4, scope: !3676)
!3676 = distinct !DILexicalBlock(scope: !3669, file: !3, line: 1957, column: 43)
!3677 = !DILocation(line: 1958, column: 6, scope: !3676)
!3678 = !DILocation(line: 1958, column: 9, scope: !3676)
!3679 = !DILocation(line: 1959, column: 4, scope: !3676)
!3680 = !DILocation(line: 1959, column: 11, scope: !3676)
!3681 = !DILocation(line: 1959, column: 14, scope: !3676)
!3682 = !DILocation(line: 1960, column: 3, scope: !3676)
!3683 = !DILocation(line: 1957, column: 40, scope: !3669)
!3684 = !DILocation(line: 1957, column: 3, scope: !3669)
!3685 = distinct !{!3685, !3674, !3686}
!3686 = !DILocation(line: 1960, column: 3, scope: !3663)
!3687 = !DILocation(line: 1961, column: 3, scope: !3664)
!3688 = !DILocation(line: 1961, column: 9, scope: !3664)
!3689 = !DILocation(line: 1961, column: 12, scope: !3664)
!3690 = !DILocation(line: 1962, column: 2, scope: !3664)
!3691 = !DILocation(line: 1956, column: 25, scope: !3658)
!3692 = !DILocation(line: 1956, column: 2, scope: !3658)
!3693 = distinct !{!3693, !3661, !3694}
!3694 = !DILocation(line: 1962, column: 2, scope: !3655)
!3695 = !DILocation(line: 1969, column: 7, scope: !3489)
!3696 = !DILocation(line: 1970, column: 14, scope: !3489)
!3697 = !DILocation(line: 1970, column: 37, scope: !3489)
!3698 = !DILocation(line: 1970, column: 36, scope: !3489)
!3699 = !DILocation(line: 1970, column: 26, scope: !3489)
!3700 = !DILocation(line: 1970, column: 10, scope: !3489)
!3701 = !DILocation(line: 1970, column: 8, scope: !3489)
!3702 = !DILocation(line: 1971, column: 8, scope: !3703)
!3703 = distinct !DILexicalBlock(scope: !3489, file: !3, line: 1971, column: 2)
!3704 = !DILocation(line: 1971, column: 6, scope: !3703)
!3705 = !DILocation(line: 1971, column: 13, scope: !3706)
!3706 = distinct !DILexicalBlock(scope: !3703, file: !3, line: 1971, column: 2)
!3707 = !DILocation(line: 1971, column: 17, scope: !3706)
!3708 = !DILocation(line: 1971, column: 15, scope: !3706)
!3709 = !DILocation(line: 1971, column: 2, scope: !3703)
!3710 = !DILocation(line: 1972, column: 11, scope: !3711)
!3711 = distinct !DILexicalBlock(scope: !3712, file: !3, line: 1972, column: 3)
!3712 = distinct !DILexicalBlock(scope: !3706, file: !3, line: 1971, column: 24)
!3713 = !DILocation(line: 1972, column: 7, scope: !3711)
!3714 = !DILocation(line: 1972, column: 16, scope: !3715)
!3715 = distinct !DILexicalBlock(scope: !3711, file: !3, line: 1972, column: 3)
!3716 = !DILocation(line: 1972, column: 22, scope: !3715)
!3717 = !DILocation(line: 1972, column: 27, scope: !3715)
!3718 = !DILocation(line: 1972, column: 20, scope: !3715)
!3719 = !DILocation(line: 1972, column: 3, scope: !3711)
!3720 = !DILocation(line: 1973, column: 8, scope: !3721)
!3721 = distinct !DILexicalBlock(scope: !3715, file: !3, line: 1972, column: 37)
!3722 = !DILocation(line: 1973, column: 13, scope: !3721)
!3723 = !DILocation(line: 1973, column: 16, scope: !3721)
!3724 = !DILocation(line: 1973, column: 6, scope: !3721)
!3725 = !DILocation(line: 1975, column: 12, scope: !3721)
!3726 = !DILocation(line: 1975, column: 19, scope: !3721)
!3727 = !DILocation(line: 1975, column: 24, scope: !3721)
!3728 = !DILocation(line: 1975, column: 27, scope: !3721)
!3729 = !DILocation(line: 1975, column: 17, scope: !3721)
!3730 = !DILocation(line: 1975, column: 10, scope: !3721)
!3731 = !DILocation(line: 1976, column: 14, scope: !3732)
!3732 = distinct !DILexicalBlock(scope: !3721, file: !3, line: 1976, column: 4)
!3733 = !DILocation(line: 1976, column: 8, scope: !3732)
!3734 = !DILocation(line: 1976, column: 19, scope: !3735)
!3735 = distinct !DILexicalBlock(scope: !3732, file: !3, line: 1976, column: 4)
!3736 = !DILocation(line: 1976, column: 27, scope: !3735)
!3737 = !DILocation(line: 1976, column: 32, scope: !3735)
!3738 = !DILocation(line: 1976, column: 25, scope: !3735)
!3739 = !DILocation(line: 1976, column: 4, scope: !3732)
!3740 = !DILocation(line: 1977, column: 12, scope: !3741)
!3741 = distinct !DILexicalBlock(scope: !3735, file: !3, line: 1976, column: 44)
!3742 = !DILocation(line: 1977, column: 17, scope: !3741)
!3743 = !DILocation(line: 1977, column: 20, scope: !3741)
!3744 = !DILocation(line: 1977, column: 10, scope: !3741)
!3745 = !DILocation(line: 1978, column: 10, scope: !3741)
!3746 = !DILocation(line: 1978, column: 15, scope: !3741)
!3747 = !DILocation(line: 1978, column: 18, scope: !3741)
!3748 = !DILocation(line: 1978, column: 27, scope: !3741)
!3749 = !DILocation(line: 1978, column: 25, scope: !3741)
!3750 = !DILocation(line: 1978, column: 8, scope: !3741)
!3751 = !DILocation(line: 1986, column: 8, scope: !3752)
!3752 = distinct !DILexicalBlock(scope: !3741, file: !3, line: 1986, column: 8)
!3753 = !DILocation(line: 1986, column: 16, scope: !3752)
!3754 = !DILocation(line: 1986, column: 13, scope: !3752)
!3755 = !DILocation(line: 1986, column: 18, scope: !3752)
!3756 = !DILocation(line: 1986, column: 21, scope: !3752)
!3757 = !DILocation(line: 1986, column: 26, scope: !3752)
!3758 = !DILocation(line: 1986, column: 23, scope: !3752)
!3759 = !DILocation(line: 1986, column: 8, scope: !3741)
!3760 = !DILocation(line: 1987, column: 11, scope: !3761)
!3761 = distinct !DILexicalBlock(scope: !3752, file: !3, line: 1986, column: 28)
!3762 = !DILocation(line: 1987, column: 16, scope: !3761)
!3763 = !DILocation(line: 1987, column: 14, scope: !3761)
!3764 = !DILocation(line: 1987, column: 24, scope: !3761)
!3765 = !DILocation(line: 1987, column: 22, scope: !3761)
!3766 = !DILocation(line: 1987, column: 9, scope: !3761)
!3767 = !DILocation(line: 1988, column: 5, scope: !3761)
!3768 = !DILocation(line: 1991, column: 13, scope: !3769)
!3769 = distinct !DILexicalBlock(scope: !3741, file: !3, line: 1991, column: 5)
!3770 = !DILocation(line: 1991, column: 20, scope: !3769)
!3771 = !DILocation(line: 1991, column: 11, scope: !3769)
!3772 = !DILocation(line: 1991, column: 9, scope: !3769)
!3773 = !DILocation(line: 1991, column: 24, scope: !3774)
!3774 = distinct !DILexicalBlock(scope: !3769, file: !3, line: 1991, column: 5)
!3775 = !DILocation(line: 1991, column: 28, scope: !3774)
!3776 = !DILocation(line: 1991, column: 35, scope: !3774)
!3777 = !DILocation(line: 1991, column: 36, scope: !3774)
!3778 = !DILocation(line: 1991, column: 26, scope: !3774)
!3779 = !DILocation(line: 1991, column: 5, scope: !3769)
!3780 = !DILocation(line: 1992, column: 9, scope: !3781)
!3781 = distinct !DILexicalBlock(scope: !3782, file: !3, line: 1992, column: 9)
!3782 = distinct !DILexicalBlock(scope: !3774, file: !3, line: 1991, column: 45)
!3783 = !DILocation(line: 1992, column: 16, scope: !3781)
!3784 = !DILocation(line: 1992, column: 21, scope: !3781)
!3785 = !DILocation(line: 1992, column: 19, scope: !3781)
!3786 = !DILocation(line: 1992, column: 9, scope: !3782)
!3787 = !DILocation(line: 1998, column: 16, scope: !3788)
!3788 = distinct !DILexicalBlock(scope: !3789, file: !3, line: 1998, column: 7)
!3789 = distinct !DILexicalBlock(scope: !3781, file: !3, line: 1992, column: 26)
!3790 = !DILocation(line: 1998, column: 23, scope: !3788)
!3791 = !DILocation(line: 1998, column: 24, scope: !3788)
!3792 = !DILocation(line: 1998, column: 27, scope: !3788)
!3793 = !DILocation(line: 1998, column: 14, scope: !3788)
!3794 = !DILocation(line: 1998, column: 11, scope: !3788)
!3795 = !DILocation(line: 1998, column: 31, scope: !3796)
!3796 = distinct !DILexicalBlock(scope: !3788, file: !3, line: 1998, column: 7)
!3797 = !DILocation(line: 1998, column: 37, scope: !3796)
!3798 = !DILocation(line: 1998, column: 34, scope: !3796)
!3799 = !DILocation(line: 1998, column: 7, scope: !3788)
!3800 = !DILocation(line: 1999, column: 11, scope: !3801)
!3801 = distinct !DILexicalBlock(scope: !3802, file: !3, line: 1999, column: 11)
!3802 = distinct !DILexicalBlock(scope: !3796, file: !3, line: 1998, column: 45)
!3803 = !DILocation(line: 1999, column: 18, scope: !3801)
!3804 = !DILocation(line: 1999, column: 22, scope: !3801)
!3805 = !DILocation(line: 1999, column: 11, scope: !3802)
!3806 = !DILocation(line: 2000, column: 19, scope: !3807)
!3807 = distinct !DILexicalBlock(scope: !3801, file: !3, line: 1999, column: 27)
!3808 = !DILocation(line: 2000, column: 21, scope: !3807)
!3809 = !DILocation(line: 2000, column: 9, scope: !3807)
!3810 = !DILocation(line: 2000, column: 11, scope: !3807)
!3811 = !DILocation(line: 2000, column: 13, scope: !3807)
!3812 = !DILocation(line: 2000, column: 17, scope: !3807)
!3813 = !DILocation(line: 2001, column: 24, scope: !3807)
!3814 = !DILocation(line: 2001, column: 31, scope: !3807)
!3815 = !DILocation(line: 2001, column: 9, scope: !3807)
!3816 = !DILocation(line: 2001, column: 16, scope: !3807)
!3817 = !DILocation(line: 2001, column: 18, scope: !3807)
!3818 = !DILocation(line: 2001, column: 22, scope: !3807)
!3819 = !DILocation(line: 2002, column: 8, scope: !3807)
!3820 = !DILocation(line: 2003, column: 7, scope: !3802)
!3821 = !DILocation(line: 1998, column: 42, scope: !3796)
!3822 = !DILocation(line: 1998, column: 7, scope: !3796)
!3823 = distinct !{!3823, !3799, !3824}
!3824 = !DILocation(line: 2003, column: 7, scope: !3788)
!3825 = !DILocation(line: 2004, column: 19, scope: !3789)
!3826 = !DILocation(line: 2004, column: 7, scope: !3789)
!3827 = !DILocation(line: 2004, column: 14, scope: !3789)
!3828 = !DILocation(line: 2004, column: 17, scope: !3789)
!3829 = !DILocation(line: 2005, column: 7, scope: !3789)
!3830 = !DILocation(line: 2005, column: 9, scope: !3789)
!3831 = !DILocation(line: 2005, column: 13, scope: !3789)
!3832 = !DILocation(line: 2007, column: 7, scope: !3789)
!3833 = !DILocation(line: 2008, column: 15, scope: !3834)
!3834 = distinct !DILexicalBlock(scope: !3781, file: !3, line: 2008, column: 15)
!3835 = !DILocation(line: 2008, column: 22, scope: !3834)
!3836 = !DILocation(line: 2008, column: 25, scope: !3834)
!3837 = !DILocation(line: 2008, column: 15, scope: !3781)
!3838 = !DILocation(line: 2009, column: 19, scope: !3839)
!3839 = distinct !DILexicalBlock(scope: !3834, file: !3, line: 2008, column: 31)
!3840 = !DILocation(line: 2009, column: 7, scope: !3839)
!3841 = !DILocation(line: 2009, column: 14, scope: !3839)
!3842 = !DILocation(line: 2009, column: 17, scope: !3839)
!3843 = !DILocation(line: 2011, column: 7, scope: !3839)
!3844 = !DILocation(line: 2012, column: 15, scope: !3845)
!3845 = distinct !DILexicalBlock(scope: !3834, file: !3, line: 2012, column: 15)
!3846 = !DILocation(line: 2012, column: 22, scope: !3845)
!3847 = !DILocation(line: 2012, column: 28, scope: !3845)
!3848 = !DILocation(line: 2012, column: 25, scope: !3845)
!3849 = !DILocation(line: 2012, column: 15, scope: !3834)
!3850 = !DILocation(line: 2018, column: 18, scope: !3851)
!3851 = distinct !DILexicalBlock(scope: !3845, file: !3, line: 2012, column: 33)
!3852 = !DILocation(line: 2018, column: 24, scope: !3851)
!3853 = !DILocation(line: 2018, column: 27, scope: !3851)
!3854 = !DILocation(line: 2018, column: 7, scope: !3851)
!3855 = !DILocation(line: 2018, column: 13, scope: !3851)
!3856 = !DILocation(line: 2018, column: 16, scope: !3851)
!3857 = !DILocation(line: 2020, column: 7, scope: !3851)
!3858 = !DILocation(line: 2022, column: 5, scope: !3782)
!3859 = !DILocation(line: 1991, column: 42, scope: !3774)
!3860 = !DILocation(line: 1991, column: 5, scope: !3774)
!3861 = distinct !{!3861, !3779, !3862}
!3862 = !DILocation(line: 2022, column: 5, scope: !3769)
!3863 = !DILocation(line: 2027, column: 12, scope: !3741)
!3864 = !DILocation(line: 2027, column: 14, scope: !3741)
!3865 = !DILocation(line: 2027, column: 19, scope: !3741)
!3866 = !DILocation(line: 2027, column: 17, scope: !3741)
!3867 = !DILocation(line: 2027, column: 5, scope: !3741)
!3868 = !DILocation(line: 2027, column: 7, scope: !3741)
!3869 = !DILocation(line: 2027, column: 10, scope: !3741)
!3870 = !DILocation(line: 2028, column: 4, scope: !3741)
!3871 = !DILocation(line: 1976, column: 41, scope: !3735)
!3872 = !DILocation(line: 1976, column: 4, scope: !3735)
!3873 = distinct !{!3873, !3739, !3874}
!3874 = !DILocation(line: 2028, column: 4, scope: !3732)
!3875 = !DILocation(line: 2029, column: 3, scope: !3721)
!3876 = !DILocation(line: 1972, column: 34, scope: !3715)
!3877 = !DILocation(line: 1972, column: 3, scope: !3715)
!3878 = distinct !{!3878, !3719, !3879}
!3879 = !DILocation(line: 2029, column: 3, scope: !3711)
!3880 = !DILocation(line: 2030, column: 10, scope: !3712)
!3881 = !DILocation(line: 2030, column: 17, scope: !3712)
!3882 = !DILocation(line: 2030, column: 15, scope: !3712)
!3883 = !DILocation(line: 2030, column: 8, scope: !3712)
!3884 = !DILocation(line: 2031, column: 2, scope: !3712)
!3885 = !DILocation(line: 1971, column: 21, scope: !3706)
!3886 = !DILocation(line: 1971, column: 2, scope: !3706)
!3887 = distinct !{!3887, !3709, !3888}
!3888 = !DILocation(line: 2031, column: 2, scope: !3703)
!3889 = !DILocation(line: 2038, column: 8, scope: !3890)
!3890 = distinct !DILexicalBlock(scope: !3489, file: !3, line: 2038, column: 2)
!3891 = !DILocation(line: 2038, column: 6, scope: !3890)
!3892 = !DILocation(line: 2038, column: 13, scope: !3893)
!3893 = distinct !DILexicalBlock(scope: !3890, file: !3, line: 2038, column: 2)
!3894 = !DILocation(line: 2038, column: 17, scope: !3893)
!3895 = !DILocation(line: 2038, column: 15, scope: !3893)
!3896 = !DILocation(line: 2038, column: 2, scope: !3890)
!3897 = !DILocation(line: 2039, column: 14, scope: !3898)
!3898 = distinct !DILexicalBlock(scope: !3893, file: !3, line: 2038, column: 28)
!3899 = !DILocation(line: 2039, column: 20, scope: !3898)
!3900 = !DILocation(line: 2039, column: 25, scope: !3898)
!3901 = !DILocation(line: 2039, column: 31, scope: !3898)
!3902 = !DILocation(line: 2039, column: 32, scope: !3898)
!3903 = !DILocation(line: 2039, column: 23, scope: !3898)
!3904 = !DILocation(line: 2039, column: 3, scope: !3898)
!3905 = !DILocation(line: 2039, column: 9, scope: !3898)
!3906 = !DILocation(line: 2039, column: 12, scope: !3898)
!3907 = !DILocation(line: 2040, column: 2, scope: !3898)
!3908 = !DILocation(line: 2038, column: 25, scope: !3893)
!3909 = !DILocation(line: 2038, column: 2, scope: !3893)
!3910 = distinct !{!3910, !3896, !3911}
!3911 = !DILocation(line: 2040, column: 2, scope: !3890)
!3912 = !DILocation(line: 2042, column: 8, scope: !3913)
!3913 = distinct !DILexicalBlock(scope: !3489, file: !3, line: 2042, column: 2)
!3914 = !DILocation(line: 2042, column: 6, scope: !3913)
!3915 = !DILocation(line: 2042, column: 13, scope: !3916)
!3916 = distinct !DILexicalBlock(scope: !3913, file: !3, line: 2042, column: 2)
!3917 = !DILocation(line: 2042, column: 17, scope: !3916)
!3918 = !DILocation(line: 2042, column: 15, scope: !3916)
!3919 = !DILocation(line: 2042, column: 2, scope: !3913)
!3920 = !DILocation(line: 2043, column: 6, scope: !3921)
!3921 = distinct !DILexicalBlock(scope: !3922, file: !3, line: 2043, column: 6)
!3922 = distinct !DILexicalBlock(scope: !3916, file: !3, line: 2042, column: 28)
!3923 = !DILocation(line: 2043, column: 8, scope: !3921)
!3924 = !DILocation(line: 2043, column: 6, scope: !3922)
!3925 = !DILocation(line: 2044, column: 9, scope: !3926)
!3926 = distinct !DILexicalBlock(scope: !3921, file: !3, line: 2043, column: 12)
!3927 = !DILocation(line: 2044, column: 16, scope: !3926)
!3928 = !DILocation(line: 2044, column: 21, scope: !3926)
!3929 = !DILocation(line: 2044, column: 27, scope: !3926)
!3930 = !DILocation(line: 2044, column: 28, scope: !3926)
!3931 = !DILocation(line: 2044, column: 19, scope: !3926)
!3932 = !DILocation(line: 2044, column: 7, scope: !3926)
!3933 = !DILocation(line: 2045, column: 3, scope: !3926)
!3934 = !DILocation(line: 2046, column: 7, scope: !3935)
!3935 = distinct !DILexicalBlock(scope: !3921, file: !3, line: 2045, column: 8)
!3936 = !DILocation(line: 2048, column: 8, scope: !3922)
!3937 = !DILocation(line: 2048, column: 15, scope: !3922)
!3938 = !DILocation(line: 2048, column: 16, scope: !3922)
!3939 = !DILocation(line: 2048, column: 22, scope: !3922)
!3940 = !DILocation(line: 2048, column: 28, scope: !3922)
!3941 = !DILocation(line: 2048, column: 20, scope: !3922)
!3942 = !DILocation(line: 2048, column: 6, scope: !3922)
!3943 = !DILocation(line: 2049, column: 9, scope: !3922)
!3944 = !DILocation(line: 2049, column: 16, scope: !3922)
!3945 = !DILocation(line: 2049, column: 7, scope: !3922)
!3946 = !DILocation(line: 2050, column: 11, scope: !3947)
!3947 = distinct !DILexicalBlock(scope: !3922, file: !3, line: 2050, column: 3)
!3948 = !DILocation(line: 2050, column: 9, scope: !3947)
!3949 = !DILocation(line: 2050, column: 7, scope: !3947)
!3950 = !DILocation(line: 2050, column: 15, scope: !3951)
!3951 = distinct !DILexicalBlock(scope: !3947, file: !3, line: 2050, column: 3)
!3952 = !DILocation(line: 2050, column: 19, scope: !3951)
!3953 = !DILocation(line: 2050, column: 17, scope: !3951)
!3954 = !DILocation(line: 2050, column: 3, scope: !3947)
!3955 = !DILocation(line: 2051, column: 11, scope: !3956)
!3956 = distinct !DILexicalBlock(scope: !3951, file: !3, line: 2050, column: 27)
!3957 = !DILocation(line: 2051, column: 13, scope: !3956)
!3958 = !DILocation(line: 2051, column: 4, scope: !3956)
!3959 = !DILocation(line: 2051, column: 6, scope: !3956)
!3960 = !DILocation(line: 2051, column: 9, scope: !3956)
!3961 = !DILocation(line: 2052, column: 16, scope: !3956)
!3962 = !DILocation(line: 2052, column: 23, scope: !3956)
!3963 = !DILocation(line: 2052, column: 4, scope: !3956)
!3964 = !DILocation(line: 2052, column: 11, scope: !3956)
!3965 = !DILocation(line: 2052, column: 14, scope: !3956)
!3966 = !DILocation(line: 2053, column: 10, scope: !3956)
!3967 = !DILocation(line: 2053, column: 14, scope: !3956)
!3968 = !DILocation(line: 2053, column: 8, scope: !3956)
!3969 = !DILocation(line: 2054, column: 3, scope: !3956)
!3970 = !DILocation(line: 2050, column: 24, scope: !3951)
!3971 = !DILocation(line: 2050, column: 3, scope: !3951)
!3972 = distinct !{!3972, !3954, !3973}
!3973 = !DILocation(line: 2054, column: 3, scope: !3947)
!3974 = !DILocation(line: 2055, column: 2, scope: !3922)
!3975 = !DILocation(line: 2042, column: 25, scope: !3916)
!3976 = !DILocation(line: 2042, column: 2, scope: !3916)
!3977 = distinct !{!3977, !3919, !3978}
!3978 = !DILocation(line: 2055, column: 2, scope: !3913)
!3979 = !DILocation(line: 2056, column: 8, scope: !3980)
!3980 = distinct !DILexicalBlock(scope: !3489, file: !3, line: 2056, column: 2)
!3981 = !DILocation(line: 2056, column: 6, scope: !3980)
!3982 = !DILocation(line: 2056, column: 13, scope: !3983)
!3983 = distinct !DILexicalBlock(scope: !3980, file: !3, line: 2056, column: 2)
!3984 = !DILocation(line: 2056, column: 17, scope: !3983)
!3985 = !DILocation(line: 2056, column: 22, scope: !3983)
!3986 = !DILocation(line: 2056, column: 15, scope: !3983)
!3987 = !DILocation(line: 2056, column: 2, scope: !3980)
!3988 = !DILocation(line: 2057, column: 15, scope: !3989)
!3989 = distinct !DILexicalBlock(scope: !3983, file: !3, line: 2056, column: 30)
!3990 = !DILocation(line: 2057, column: 22, scope: !3989)
!3991 = !DILocation(line: 2057, column: 27, scope: !3989)
!3992 = !DILocation(line: 2057, column: 33, scope: !3989)
!3993 = !DILocation(line: 2057, column: 34, scope: !3989)
!3994 = !DILocation(line: 2057, column: 25, scope: !3989)
!3995 = !DILocation(line: 2057, column: 3, scope: !3989)
!3996 = !DILocation(line: 2057, column: 10, scope: !3989)
!3997 = !DILocation(line: 2057, column: 13, scope: !3989)
!3998 = !DILocation(line: 2058, column: 2, scope: !3989)
!3999 = !DILocation(line: 2056, column: 27, scope: !3983)
!4000 = !DILocation(line: 2056, column: 2, scope: !3983)
!4001 = distinct !{!4001, !3987, !4002}
!4002 = !DILocation(line: 2058, column: 2, scope: !3980)
!4003 = !DILocation(line: 2059, column: 8, scope: !3489)
!4004 = !DILocation(line: 2059, column: 15, scope: !3489)
!4005 = !DILocation(line: 2059, column: 22, scope: !3489)
!4006 = !DILocation(line: 2059, column: 6, scope: !3489)
!4007 = !DILocation(line: 2060, column: 1, scope: !3489)
!4008 = distinct !DISubprogram(name: "icnvrt", linkageName: "_ZL6icnvrtdi", scope: !3, file: !3, line: 1545, type: !4009, scopeLine: 1545, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1163)
!4009 = !DISubroutineType(types: !4010)
!4010 = !{!97, !100, !97}
!4011 = !DILocalVariable(name: "x", arg: 1, scope: !4008, file: !3, line: 1545, type: !100)
!4012 = !DILocation(line: 1545, column: 26, scope: !4008)
!4013 = !DILocalVariable(name: "ipwr2", arg: 2, scope: !4008, file: !3, line: 1545, type: !97)
!4014 = !DILocation(line: 1545, column: 33, scope: !4008)
!4015 = !DILocation(line: 1546, column: 15, scope: !4008)
!4016 = !DILocation(line: 1546, column: 23, scope: !4008)
!4017 = !DILocation(line: 1546, column: 21, scope: !4008)
!4018 = !DILocation(line: 1546, column: 14, scope: !4008)
!4019 = !DILocation(line: 1546, column: 2, scope: !4008)
!4020 = distinct !DISubprogram(name: "cudaMalloc<int>", linkageName: "_ZL10cudaMallocIiE9cudaErrorPPT_m", scope: !4021, file: !4021, line: 490, type: !4022, scopeLine: 494, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, templateParams: !4026, retainedNodes: !1163)
!4021 = !DIFile(filename: "/usr/local/cuda/include/cuda_runtime.h", directory: "")
!4022 = !DISubroutineType(types: !4023)
!4023 = !{!4024, !4025, !154}
!4024 = !DIDerivedType(tag: DW_TAG_typedef, name: "cudaError_t", file: !6, line: 1419, baseType: !14)
!4025 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !98, size: 64)
!4026 = !{!4027}
!4027 = !DITemplateTypeParameter(name: "T", type: !97)
!4028 = !DILocalVariable(name: "devPtr", arg: 1, scope: !4020, file: !4021, line: 491, type: !4025)
!4029 = !DILocation(line: 491, column: 12, scope: !4020)
!4030 = !DILocalVariable(name: "size", arg: 2, scope: !4020, file: !4021, line: 492, type: !154)
!4031 = !DILocation(line: 492, column: 12, scope: !4020)
!4032 = !DILocation(line: 495, column: 38, scope: !4020)
!4033 = !DILocation(line: 495, column: 23, scope: !4020)
!4034 = !DILocation(line: 495, column: 46, scope: !4020)
!4035 = !DILocation(line: 495, column: 10, scope: !4020)
!4036 = !DILocation(line: 495, column: 3, scope: !4020)
!4037 = distinct !DISubprogram(name: "cudaMalloc<double>", linkageName: "_ZL10cudaMallocIdE9cudaErrorPPT_m", scope: !4021, file: !4021, line: 490, type: !4038, scopeLine: 494, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, templateParams: !4041, retainedNodes: !1163)
!4038 = !DISubroutineType(types: !4039)
!4039 = !{!4024, !4040, !154}
!4040 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !99, size: 64)
!4041 = !{!4042}
!4042 = !DITemplateTypeParameter(name: "T", type: !100)
!4043 = !DILocalVariable(name: "devPtr", arg: 1, scope: !4037, file: !4021, line: 491, type: !4040)
!4044 = !DILocation(line: 491, column: 12, scope: !4037)
!4045 = !DILocalVariable(name: "size", arg: 2, scope: !4037, file: !4021, line: 492, type: !154)
!4046 = !DILocation(line: 492, column: 12, scope: !4037)
!4047 = !DILocation(line: 495, column: 38, scope: !4037)
!4048 = !DILocation(line: 495, column: 23, scope: !4037)
!4049 = !DILocation(line: 495, column: 46, scope: !4037)
!4050 = !DILocation(line: 495, column: 10, scope: !4037)
!4051 = !DILocation(line: 495, column: 3, scope: !4037)
