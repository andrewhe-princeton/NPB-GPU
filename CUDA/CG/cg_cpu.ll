; ModuleID = 'cg_cpu.bc'
source_filename = "llvm-link-cudafe"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.cudaDeviceProp = type { [256 x i8], i64, i64, i32, i32, i64, i32, [3 x i32], [3 x i32], i32, i64, i32, i32, i64, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, [2 x i32], [2 x i32], [3 x i32], [2 x i32], [3 x i32], [3 x i32], i32, [2 x i32], [3 x i32], [2 x i32], i32, [2 x i32], [3 x i32], [2 x i32], [3 x i32], i32, [2 x i32], i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i64, i32, i32, i32, i32 }
%struct.dim3 = type { i32, i32, i32 }

@extern_share_data = external dso_local addrspace(3) global [0 x double], align 8
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
@.str.83 = private unnamed_addr constant [32 x i8] c"internal error in sparse: i=%d\0A\00", align 1
@extern_share_data_shared = internal global [1024 x double] zeroinitializer

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #0

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: convergent nounwind
declare void @llvm.nvvm.barrier0() #2

; Function Attrs: noinline nounwind uwtable
define dso_local double @_Z6randlcPdd(double* %x, double %a) #3 !dbg !1253 {
entry:
  call void @llvm.dbg.value(metadata double* %x, metadata !1256, metadata !DIExpression()), !dbg !1257
  call void @llvm.dbg.value(metadata double %a, metadata !1258, metadata !DIExpression()), !dbg !1257
  %mul = fmul contract double 0x3E80000000000000, %a, !dbg !1259
  call void @llvm.dbg.value(metadata double %mul, metadata !1260, metadata !DIExpression()), !dbg !1257
  %conv = fptosi double %mul to i32, !dbg !1261
  %conv1 = sitofp i32 %conv to double, !dbg !1262
  call void @llvm.dbg.value(metadata double %conv1, metadata !1263, metadata !DIExpression()), !dbg !1257
  %mul2 = fmul contract double 0x4160000000000000, %conv1, !dbg !1264
  %sub = fsub contract double %a, %mul2, !dbg !1265
  call void @llvm.dbg.value(metadata double %sub, metadata !1266, metadata !DIExpression()), !dbg !1257
  %0 = load double, double* %x, align 8, !dbg !1267
  %mul3 = fmul contract double 0x3E80000000000000, %0, !dbg !1268
  call void @llvm.dbg.value(metadata double %mul3, metadata !1260, metadata !DIExpression()), !dbg !1257
  %conv4 = fptosi double %mul3 to i32, !dbg !1269
  %conv5 = sitofp i32 %conv4 to double, !dbg !1270
  call void @llvm.dbg.value(metadata double %conv5, metadata !1271, metadata !DIExpression()), !dbg !1257
  %1 = load double, double* %x, align 8, !dbg !1272
  %mul6 = fmul contract double 0x4160000000000000, %conv5, !dbg !1273
  %sub7 = fsub contract double %1, %mul6, !dbg !1274
  call void @llvm.dbg.value(metadata double %sub7, metadata !1275, metadata !DIExpression()), !dbg !1257
  %mul8 = fmul contract double %conv1, %sub7, !dbg !1276
  %mul9 = fmul contract double %sub, %conv5, !dbg !1277
  %add = fadd contract double %mul8, %mul9, !dbg !1278
  call void @llvm.dbg.value(metadata double %add, metadata !1260, metadata !DIExpression()), !dbg !1257
  %mul10 = fmul contract double 0x3E80000000000000, %add, !dbg !1279
  %conv11 = fptosi double %mul10 to i32, !dbg !1280
  %conv12 = sitofp i32 %conv11 to double, !dbg !1281
  call void @llvm.dbg.value(metadata double %conv12, metadata !1282, metadata !DIExpression()), !dbg !1257
  %mul13 = fmul contract double 0x4160000000000000, %conv12, !dbg !1283
  %sub14 = fsub contract double %add, %mul13, !dbg !1284
  call void @llvm.dbg.value(metadata double %sub14, metadata !1285, metadata !DIExpression()), !dbg !1257
  %mul15 = fmul contract double 0x4160000000000000, %sub14, !dbg !1286
  %mul16 = fmul contract double %sub, %sub7, !dbg !1287
  %add17 = fadd contract double %mul15, %mul16, !dbg !1288
  call void @llvm.dbg.value(metadata double %add17, metadata !1289, metadata !DIExpression()), !dbg !1257
  %mul18 = fmul contract double 0x3D10000000000000, %add17, !dbg !1290
  %conv19 = fptosi double %mul18 to i32, !dbg !1291
  %conv20 = sitofp i32 %conv19 to double, !dbg !1292
  call void @llvm.dbg.value(metadata double %conv20, metadata !1293, metadata !DIExpression()), !dbg !1257
  %mul21 = fmul contract double 0x42D0000000000000, %conv20, !dbg !1294
  %sub22 = fsub contract double %add17, %mul21, !dbg !1295
  store double %sub22, double* %x, align 8, !dbg !1296
  %2 = load double, double* %x, align 8, !dbg !1297
  %mul23 = fmul contract double 0x3D10000000000000, %2, !dbg !1298
  ret double %mul23, !dbg !1299
}

; Function Attrs: noinline uwtable
define dso_local void @_Z15c_print_resultsPcciiiiddS_iS_S_S_S_S_S_S_S_S_S_S_S_S_S_(i8* %name, i8 signext %class_npb, i32 %n1, i32 %n2, i32 %n3, i32 %niter, double %t, double %mops, i8* %optype, i32 %passed_verification, i8* %npbversion, i8* %compiletime, i8* %compilerversion, i8* %libversion, i8* %cpu_device, i8* %gpu_device, i8* %gpu_config, i8* %cc, i8* %clink, i8* %c_lib, i8* %c_inc, i8* %cflags, i8* %clinkflags, i8* %rand) #4 !dbg !1300 {
entry:
  %size = alloca [16 x i8], align 16
  call void @llvm.dbg.value(metadata i8* %name, metadata !1303, metadata !DIExpression()), !dbg !1304
  call void @llvm.dbg.value(metadata i8 %class_npb, metadata !1305, metadata !DIExpression()), !dbg !1304
  call void @llvm.dbg.value(metadata i32 %n1, metadata !1306, metadata !DIExpression()), !dbg !1304
  call void @llvm.dbg.value(metadata i32 %n2, metadata !1307, metadata !DIExpression()), !dbg !1304
  call void @llvm.dbg.value(metadata i32 %n3, metadata !1308, metadata !DIExpression()), !dbg !1304
  call void @llvm.dbg.value(metadata i32 %niter, metadata !1309, metadata !DIExpression()), !dbg !1304
  call void @llvm.dbg.value(metadata double %t, metadata !1310, metadata !DIExpression()), !dbg !1304
  call void @llvm.dbg.value(metadata double %mops, metadata !1311, metadata !DIExpression()), !dbg !1304
  call void @llvm.dbg.value(metadata i8* %optype, metadata !1312, metadata !DIExpression()), !dbg !1304
  call void @llvm.dbg.value(metadata i32 %passed_verification, metadata !1313, metadata !DIExpression()), !dbg !1304
  call void @llvm.dbg.value(metadata i8* %npbversion, metadata !1314, metadata !DIExpression()), !dbg !1304
  call void @llvm.dbg.value(metadata i8* %compiletime, metadata !1315, metadata !DIExpression()), !dbg !1304
  call void @llvm.dbg.value(metadata i8* %compilerversion, metadata !1316, metadata !DIExpression()), !dbg !1304
  call void @llvm.dbg.value(metadata i8* %libversion, metadata !1317, metadata !DIExpression()), !dbg !1304
  call void @llvm.dbg.value(metadata i8* %cpu_device, metadata !1318, metadata !DIExpression()), !dbg !1304
  call void @llvm.dbg.value(metadata i8* %gpu_device, metadata !1319, metadata !DIExpression()), !dbg !1304
  call void @llvm.dbg.value(metadata i8* %gpu_config, metadata !1320, metadata !DIExpression()), !dbg !1304
  call void @llvm.dbg.value(metadata i8* %cc, metadata !1321, metadata !DIExpression()), !dbg !1304
  call void @llvm.dbg.value(metadata i8* %clink, metadata !1322, metadata !DIExpression()), !dbg !1304
  call void @llvm.dbg.value(metadata i8* %c_lib, metadata !1323, metadata !DIExpression()), !dbg !1304
  call void @llvm.dbg.value(metadata i8* %c_inc, metadata !1324, metadata !DIExpression()), !dbg !1304
  call void @llvm.dbg.value(metadata i8* %cflags, metadata !1325, metadata !DIExpression()), !dbg !1304
  call void @llvm.dbg.value(metadata i8* %clinkflags, metadata !1326, metadata !DIExpression()), !dbg !1304
  call void @llvm.dbg.value(metadata i8* %rand, metadata !1327, metadata !DIExpression()), !dbg !1304
  %call = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([27 x i8], [27 x i8]* @.str, i64 0, i64 0), i8* %name), !dbg !1328
  %conv = sext i8 %class_npb to i32, !dbg !1329
  %call1 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([46 x i8], [46 x i8]* @.str.1, i64 0, i64 0), i32 %conv), !dbg !1330
  %arrayidx = getelementptr inbounds i8, i8* %name, i64 0, !dbg !1331
  %0 = load i8, i8* %arrayidx, align 1, !dbg !1331
  %conv2 = sext i8 %0 to i32, !dbg !1331
  %cmp = icmp eq i32 %conv2, 73, !dbg !1333
  br i1 %cmp, label %land.lhs.true, label %if.else15, !dbg !1334

land.lhs.true:                                    ; preds = %entry
  %arrayidx3 = getelementptr inbounds i8, i8* %name, i64 1, !dbg !1335
  %1 = load i8, i8* %arrayidx3, align 1, !dbg !1335
  %conv4 = sext i8 %1 to i32, !dbg !1335
  %cmp5 = icmp eq i32 %conv4, 83, !dbg !1336
  br i1 %cmp5, label %if.then, label %if.else15, !dbg !1337

if.then:                                          ; preds = %land.lhs.true
  %cmp6 = icmp eq i32 %n3, 0, !dbg !1338
  br i1 %cmp6, label %if.then7, label %if.else, !dbg !1341

if.then7:                                         ; preds = %if.then
  %conv8 = sext i32 %n1 to i64, !dbg !1342
  call void @llvm.dbg.value(metadata i64 %conv8, metadata !1344, metadata !DIExpression()), !dbg !1345
  %cmp9 = icmp ne i32 %n2, 0, !dbg !1346
  br i1 %cmp9, label %if.then10, label %if.end, !dbg !1348

if.then10:                                        ; preds = %if.then7
  %conv11 = sext i32 %n2 to i64, !dbg !1349
  %mul = mul nsw i64 %conv8, %conv11, !dbg !1351
  call void @llvm.dbg.value(metadata i64 %mul, metadata !1344, metadata !DIExpression()), !dbg !1345
  br label %if.end, !dbg !1352

if.end:                                           ; preds = %if.then10, %if.then7
  %nn.0 = phi i64 [ %mul, %if.then10 ], [ %conv8, %if.then7 ], !dbg !1345
  call void @llvm.dbg.value(metadata i64 %nn.0, metadata !1344, metadata !DIExpression()), !dbg !1345
  %call12 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([38 x i8], [38 x i8]* @.str.2, i64 0, i64 0), i64 %nn.0), !dbg !1353
  br label %if.end14, !dbg !1354

if.else:                                          ; preds = %if.then
  %call13 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([44 x i8], [44 x i8]* @.str.3, i64 0, i64 0), i32 %n1, i32 %n2, i32 %n3), !dbg !1355
  br label %if.end14

if.end14:                                         ; preds = %if.else, %if.end
  br label %if.end48, !dbg !1357

if.else15:                                        ; preds = %land.lhs.true, %entry
  call void @llvm.dbg.declare(metadata [16 x i8]* %size, metadata !1358, metadata !DIExpression()), !dbg !1363
  %cmp16 = icmp eq i32 %n2, 0, !dbg !1364
  br i1 %cmp16, label %land.lhs.true17, label %if.else45, !dbg !1366

land.lhs.true17:                                  ; preds = %if.else15
  %cmp18 = icmp eq i32 %n3, 0, !dbg !1367
  br i1 %cmp18, label %if.then19, label %if.else45, !dbg !1368

if.then19:                                        ; preds = %land.lhs.true17
  %arrayidx20 = getelementptr inbounds i8, i8* %name, i64 0, !dbg !1369
  %2 = load i8, i8* %arrayidx20, align 1, !dbg !1369
  %conv21 = sext i8 %2 to i32, !dbg !1369
  %cmp22 = icmp eq i32 %conv21, 69, !dbg !1372
  br i1 %cmp22, label %land.lhs.true23, label %if.else42, !dbg !1373

land.lhs.true23:                                  ; preds = %if.then19
  %arrayidx24 = getelementptr inbounds i8, i8* %name, i64 1, !dbg !1374
  %3 = load i8, i8* %arrayidx24, align 1, !dbg !1374
  %conv25 = sext i8 %3 to i32, !dbg !1374
  %cmp26 = icmp eq i32 %conv25, 80, !dbg !1375
  br i1 %cmp26, label %if.then27, label %if.else42, !dbg !1376

if.then27:                                        ; preds = %land.lhs.true23
  %arraydecay = getelementptr inbounds [16 x i8], [16 x i8]* %size, i64 0, i64 0, !dbg !1377
  %conv28 = sitofp i32 %n1 to double, !dbg !1379
  %call29 = call double @pow(double 2.000000e+00, double %conv28) #11, !dbg !1380
  %call30 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay, i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.4, i64 0, i64 0), double %call29) #11, !dbg !1381
  call void @llvm.dbg.value(metadata i32 14, metadata !1382, metadata !DIExpression()), !dbg !1383
  %idxprom = sext i32 14 to i64, !dbg !1384
  %arrayidx31 = getelementptr inbounds [16 x i8], [16 x i8]* %size, i64 0, i64 %idxprom, !dbg !1384
  %4 = load i8, i8* %arrayidx31, align 1, !dbg !1384
  %conv32 = sext i8 %4 to i32, !dbg !1384
  %cmp33 = icmp eq i32 %conv32, 46, !dbg !1386
  br i1 %cmp33, label %if.then34, label %if.end37, !dbg !1387

if.then34:                                        ; preds = %if.then27
  %idxprom35 = sext i32 14 to i64, !dbg !1388
  %arrayidx36 = getelementptr inbounds [16 x i8], [16 x i8]* %size, i64 0, i64 %idxprom35, !dbg !1388
  store i8 32, i8* %arrayidx36, align 1, !dbg !1390
  %dec = add nsw i32 14, -1, !dbg !1391
  call void @llvm.dbg.value(metadata i32 %dec, metadata !1382, metadata !DIExpression()), !dbg !1383
  br label %if.end37, !dbg !1392

if.end37:                                         ; preds = %if.then34, %if.then27
  %j.0 = phi i32 [ %dec, %if.then34 ], [ 14, %if.then27 ], !dbg !1393
  call void @llvm.dbg.value(metadata i32 %j.0, metadata !1382, metadata !DIExpression()), !dbg !1383
  %add = add nsw i32 %j.0, 1, !dbg !1394
  %idxprom38 = sext i32 %add to i64, !dbg !1395
  %arrayidx39 = getelementptr inbounds [16 x i8], [16 x i8]* %size, i64 0, i64 %idxprom38, !dbg !1395
  store i8 0, i8* %arrayidx39, align 1, !dbg !1396
  %arraydecay40 = getelementptr inbounds [16 x i8], [16 x i8]* %size, i64 0, i64 0, !dbg !1397
  %call41 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([34 x i8], [34 x i8]* @.str.5, i64 0, i64 0), i8* %arraydecay40), !dbg !1398
  br label %if.end44, !dbg !1399

if.else42:                                        ; preds = %land.lhs.true23, %if.then19
  %call43 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.6, i64 0, i64 0), i32 %n1), !dbg !1400
  br label %if.end44

if.end44:                                         ; preds = %if.else42, %if.end37
  br label %if.end47, !dbg !1402

if.else45:                                        ; preds = %land.lhs.true17, %if.else15
  %call46 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([42 x i8], [42 x i8]* @.str.7, i64 0, i64 0), i32 %n1, i32 %n2, i32 %n3), !dbg !1403
  br label %if.end47

if.end47:                                         ; preds = %if.else45, %if.end44
  br label %if.end48

if.end48:                                         ; preds = %if.end47, %if.end14
  %call49 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.8, i64 0, i64 0), i32 %niter), !dbg !1405
  %call50 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([39 x i8], [39 x i8]* @.str.9, i64 0, i64 0), double %t), !dbg !1406
  %call51 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([39 x i8], [39 x i8]* @.str.10, i64 0, i64 0), double %mops), !dbg !1407
  %call52 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([25 x i8], [25 x i8]* @.str.11, i64 0, i64 0), i8* %optype), !dbg !1408
  %cmp53 = icmp slt i32 %passed_verification, 0, !dbg !1409
  br i1 %cmp53, label %if.then54, label %if.else56, !dbg !1411

if.then54:                                        ; preds = %if.end48
  %call55 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([45 x i8], [45 x i8]* @.str.12, i64 0, i64 0)), !dbg !1412
  br label %if.end62, !dbg !1414

if.else56:                                        ; preds = %if.end48
  %tobool = icmp ne i32 %passed_verification, 0, !dbg !1415
  br i1 %tobool, label %if.then57, label %if.else59, !dbg !1417

if.then57:                                        ; preds = %if.else56
  %call58 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([45 x i8], [45 x i8]* @.str.13, i64 0, i64 0)), !dbg !1418
  br label %if.end61, !dbg !1420

if.else59:                                        ; preds = %if.else56
  %call60 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([45 x i8], [45 x i8]* @.str.14, i64 0, i64 0)), !dbg !1421
  br label %if.end61

if.end61:                                         ; preds = %if.else59, %if.then57
  br label %if.end62

if.end62:                                         ; preds = %if.end61, %if.then54
  %call63 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.15, i64 0, i64 0), i8* %npbversion), !dbg !1423
  %call64 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.16, i64 0, i64 0), i8* %compiletime), !dbg !1424
  %call65 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.17, i64 0, i64 0), i8* %compilerversion), !dbg !1425
  %call66 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.18, i64 0, i64 0), i8* %libversion), !dbg !1426
  %call67 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([20 x i8], [20 x i8]* @.str.19, i64 0, i64 0)), !dbg !1427
  %call68 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.20, i64 0, i64 0), i8* %cc), !dbg !1428
  %call69 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.21, i64 0, i64 0), i8* %clink), !dbg !1429
  %call70 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.22, i64 0, i64 0), i8* %c_lib), !dbg !1430
  %call71 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.23, i64 0, i64 0), i8* %c_inc), !dbg !1431
  %call72 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.24, i64 0, i64 0), i8* %cflags), !dbg !1432
  %call73 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.25, i64 0, i64 0), i8* %clinkflags), !dbg !1433
  %call74 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.26, i64 0, i64 0), i8* %rand), !dbg !1434
  %call75 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.27, i64 0, i64 0)), !dbg !1435
  %call76 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.28, i64 0, i64 0), i8* %cpu_device), !dbg !1436
  %call77 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.29, i64 0, i64 0), i8* %gpu_device), !dbg !1437
  %call78 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.30, i64 0, i64 0)), !dbg !1438
  %call79 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.31, i64 0, i64 0), i8* %gpu_config), !dbg !1439
  %call80 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.32, i64 0, i64 0)), !dbg !1440
  %call81 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([72 x i8], [72 x i8]* @.str.33, i64 0, i64 0)), !dbg !1441
  %call82 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([27 x i8], [27 x i8]* @.str.34, i64 0, i64 0)), !dbg !1442
  %call83 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([56 x i8], [56 x i8]* @.str.35, i64 0, i64 0)), !dbg !1443
  %call84 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([52 x i8], [52 x i8]* @.str.36, i64 0, i64 0)), !dbg !1444
  %call85 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([46 x i8], [46 x i8]* @.str.37, i64 0, i64 0)), !dbg !1445
  %call86 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.32, i64 0, i64 0)), !dbg !1446
  %call87 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([43 x i8], [43 x i8]* @.str.38, i64 0, i64 0)), !dbg !1447
  %call88 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([72 x i8], [72 x i8]* @.str.33, i64 0, i64 0)), !dbg !1448
  %call89 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.32, i64 0, i64 0)), !dbg !1449
  ret void, !dbg !1450
}

declare dso_local i32 @printf(i8*, ...) #5

; Function Attrs: nounwind
declare dso_local double @pow(double, double) #6

; Function Attrs: nounwind
declare dso_local i32 @sprintf(i8*, i8*, ...) #6

; Function Attrs: noinline norecurse uwtable
define dso_local i32 @main(i32 %argc, i8** %argv) #7 !dbg !1451 {
entry:
  %rnorm = alloca double, align 8
  %norm_temp1 = alloca double, align 8
  %norm_temp2 = alloca double, align 8
  %gpu_config = alloca [256 x i8], align 16
  %gpu_config_string = alloca [2048 x i8], align 16
  call void @llvm.dbg.value(metadata i32 %argc, metadata !1454, metadata !DIExpression()), !dbg !1455
  call void @llvm.dbg.value(metadata i8** %argv, metadata !1456, metadata !DIExpression()), !dbg !1455
  %call = call noalias i8* @malloc(i64 8064000) #11, !dbg !1457, !tulip.target.mapdata.to !1458
  %0 = bitcast i8* %call to i32*, !dbg !1459
  store i32* %0, i32** @_ZL6colidx, align 8, !dbg !1460
  %call1 = call noalias i8* @malloc(i64 56004) #11, !dbg !1461, !tulip.target.mapdata.to !1462
  %1 = bitcast i8* %call1 to i32*, !dbg !1463
  store i32* %1, i32** @_ZL6rowstr, align 8, !dbg !1464
  %call2 = call noalias i8* @malloc(i64 56000) #11, !dbg !1465
  %2 = bitcast i8* %call2 to i32*, !dbg !1466
  store i32* %2, i32** @_ZL2iv, align 8, !dbg !1467
  %call3 = call noalias i8* @malloc(i64 56000) #11, !dbg !1468
  %3 = bitcast i8* %call3 to i32*, !dbg !1469
  store i32* %3, i32** @_ZL4arow, align 8, !dbg !1470
  %call4 = call noalias i8* @malloc(i64 672000) #11, !dbg !1471
  %4 = bitcast i8* %call4 to i32*, !dbg !1472
  store i32* %4, i32** @_ZL4acol, align 8, !dbg !1473
  %call5 = call noalias i8* @malloc(i64 1344000) #11, !dbg !1474
  %5 = bitcast i8* %call5 to double*, !dbg !1475
  store double* %5, double** @_ZL4aelt, align 8, !dbg !1476
  %call6 = call noalias i8* @malloc(i64 16128000) #11, !dbg !1477, !tulip.target.mapdata.to !1478
  %6 = bitcast i8* %call6 to double*, !dbg !1479
  store double* %6, double** @_ZL1a, align 8, !dbg !1480
  %call7 = call noalias i8* @malloc(i64 112016) #11, !dbg !1481, !tulip.target.mapdata.to !1482
  %7 = bitcast i8* %call7 to double*, !dbg !1483
  store double* %7, double** @_ZL1x, align 8, !dbg !1484
  %call8 = call noalias i8* @malloc(i64 112016) #11, !dbg !1485, !tulip.target.mapdata.to !1482
  %8 = bitcast i8* %call8 to double*, !dbg !1486
  store double* %8, double** @_ZL1z, align 8, !dbg !1487
  %call9 = call noalias i8* @malloc(i64 112016) #11, !dbg !1488, !tulip.target.mapdata.to !1482
  %9 = bitcast i8* %call9 to double*, !dbg !1489
  store double* %9, double** @_ZL1p, align 8, !dbg !1490
  %call10 = call noalias i8* @malloc(i64 112016) #11, !dbg !1491, !tulip.target.mapdata.to !1482
  %10 = bitcast i8* %call10 to double*, !dbg !1492
  store double* %10, double** @_ZL1q, align 8, !dbg !1493
  %call11 = call noalias i8* @malloc(i64 112016) #11, !dbg !1494, !tulip.target.mapdata.to !1482
  %11 = bitcast i8* %call11 to double*, !dbg !1495
  store double* %11, double** @_ZL1r, align 8, !dbg !1496
  %call12 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([54 x i8], [54 x i8]* @.str.39, i64 0, i64 0)), !dbg !1497
  call void @llvm.dbg.declare(metadata double* %rnorm, metadata !1498, metadata !DIExpression()), !dbg !1499
  call void @llvm.dbg.declare(metadata double* %norm_temp1, metadata !1500, metadata !DIExpression()), !dbg !1501
  call void @llvm.dbg.declare(metadata double* %norm_temp2, metadata !1502, metadata !DIExpression()), !dbg !1503
  store i32 0, i32* @_ZL8firstrow, align 4, !dbg !1504
  store i32 13999, i32* @_ZL7lastrow, align 4, !dbg !1505
  store i32 0, i32* @_ZL8firstcol, align 4, !dbg !1506
  store i32 13999, i32* @_ZL7lastcol, align 4, !dbg !1507
  call void @llvm.dbg.value(metadata i8 65, metadata !1508, metadata !DIExpression()), !dbg !1455
  call void @llvm.dbg.value(metadata double 0x4031215715A1D8EC, metadata !1509, metadata !DIExpression()), !dbg !1455
  %call13 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([65 x i8], [65 x i8]* @.str.40, i64 0, i64 0)), !dbg !1510
  %call14 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.41, i64 0, i64 0), i32 14000), !dbg !1511
  %call15 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([18 x i8], [18 x i8]* @.str.42, i64 0, i64 0), i32 15), !dbg !1512
  store i32 14000, i32* @_ZL3naa, align 4, !dbg !1513
  store i32 2016000, i32* @_ZL3nzz, align 4, !dbg !1514
  store double 0x41B2B9B0A1000000, double* @_ZL4tran, align 8, !dbg !1515
  store double 0x41D2309CE5400000, double* @_ZL5amult, align 8, !dbg !1516
  %12 = load double, double* @_ZL5amult, align 8, !dbg !1517
  %call16 = call double @_Z6randlcPdd(double* @_ZL4tran, double %12), !dbg !1518
  call void @llvm.dbg.value(metadata double %call16, metadata !1519, metadata !DIExpression()), !dbg !1455
  %13 = load i32, i32* @_ZL3naa, align 4, !dbg !1520
  %14 = load i32, i32* @_ZL3nzz, align 4, !dbg !1521
  %15 = load double*, double** @_ZL1a, align 8, !dbg !1522
  %16 = load i32*, i32** @_ZL6colidx, align 8, !dbg !1523
  %17 = load i32*, i32** @_ZL6rowstr, align 8, !dbg !1524
  %18 = load i32, i32* @_ZL8firstrow, align 4, !dbg !1525
  %19 = load i32, i32* @_ZL7lastrow, align 4, !dbg !1526
  %20 = load i32, i32* @_ZL8firstcol, align 4, !dbg !1527
  %21 = load i32, i32* @_ZL7lastcol, align 4, !dbg !1528
  %22 = load i32*, i32** @_ZL4arow, align 8, !dbg !1529
  %23 = load i32*, i32** @_ZL4acol, align 8, !dbg !1530
  %24 = bitcast i32* %23 to i8*, !dbg !1530
  %25 = bitcast i8* %24 to [12 x i32]*, !dbg !1531
  %26 = load double*, double** @_ZL4aelt, align 8, !dbg !1532
  %27 = bitcast double* %26 to i8*, !dbg !1532
  %28 = bitcast i8* %27 to [12 x double]*, !dbg !1533
  %29 = load i32*, i32** @_ZL2iv, align 8, !dbg !1534
  call void @_ZL5makeaiiPdPiS0_iiiiS0_PA12_iPA12_dS0_(i32 %13, i32 %14, double* %15, i32* %16, i32* %17, i32 %18, i32 %19, i32 %20, i32 %21, i32* %22, [12 x i32]* %25, [12 x double]* %28, i32* %29), !dbg !1535
  call void @llvm.dbg.value(metadata i32 0, metadata !1536, metadata !DIExpression()), !dbg !1455
  br label %for.cond, !dbg !1537

for.cond:                                         ; preds = %for.inc28, %entry
  %indvars.iv14 = phi i64 [ %indvars.iv.next15, %for.inc28 ], [ 0, %entry ], !dbg !1539
  call void @llvm.dbg.value(metadata i64 %indvars.iv14, metadata !1536, metadata !DIExpression()), !dbg !1455
  %30 = load i32, i32* @_ZL7lastrow, align 4, !dbg !1540
  %31 = load i32, i32* @_ZL8firstrow, align 4, !dbg !1542
  %sub = sub nsw i32 %30, %31, !dbg !1543
  %add = add nsw i32 %sub, 1, !dbg !1544
  %32 = sext i32 %add to i64, !dbg !1545
  %cmp = icmp slt i64 %indvars.iv14, %32, !dbg !1545
  br i1 %cmp, label %for.body, label %for.end30, !dbg !1546

for.body:                                         ; preds = %for.cond
  %33 = load i32*, i32** @_ZL6rowstr, align 8, !dbg !1547
  %arrayidx = getelementptr inbounds i32, i32* %33, i64 %indvars.iv14, !dbg !1547
  %34 = load i32, i32* %arrayidx, align 4, !dbg !1547
  call void @llvm.dbg.value(metadata i32 %34, metadata !1550, metadata !DIExpression()), !dbg !1455
  %35 = sext i32 %34 to i64, !dbg !1551
  br label %for.cond17, !dbg !1551

for.cond17:                                       ; preds = %for.inc, %for.body
  %indvars.iv12 = phi i64 [ %indvars.iv.next13, %for.inc ], [ %35, %for.body ], !dbg !1552
  call void @llvm.dbg.value(metadata i64 %indvars.iv12, metadata !1550, metadata !DIExpression()), !dbg !1455
  %36 = load i32*, i32** @_ZL6rowstr, align 8, !dbg !1553
  %37 = add nuw nsw i64 %indvars.iv14, 1, !dbg !1555
  %arrayidx20 = getelementptr inbounds i32, i32* %36, i64 %37, !dbg !1553
  %38 = load i32, i32* %arrayidx20, align 4, !dbg !1553
  %39 = sext i32 %38 to i64, !dbg !1556
  %cmp21 = icmp slt i64 %indvars.iv12, %39, !dbg !1556
  br i1 %cmp21, label %for.body22, label %for.end, !dbg !1557

for.body22:                                       ; preds = %for.cond17
  %40 = load i32*, i32** @_ZL6colidx, align 8, !dbg !1558
  %arrayidx24 = getelementptr inbounds i32, i32* %40, i64 %indvars.iv12, !dbg !1558
  %41 = load i32, i32* %arrayidx24, align 4, !dbg !1558
  %42 = load i32, i32* @_ZL8firstcol, align 4, !dbg !1560
  %sub25 = sub nsw i32 %41, %42, !dbg !1561
  %43 = load i32*, i32** @_ZL6colidx, align 8, !dbg !1562
  %arrayidx27 = getelementptr inbounds i32, i32* %43, i64 %indvars.iv12, !dbg !1562
  store i32 %sub25, i32* %arrayidx27, align 4, !dbg !1563
  br label %for.inc, !dbg !1564

for.inc:                                          ; preds = %for.body22
  %indvars.iv.next13 = add nsw i64 %indvars.iv12, 1, !dbg !1565
  call void @llvm.dbg.value(metadata i32 undef, metadata !1550, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !1455
  br label %for.cond17, !dbg !1566, !llvm.loop !1567

for.end:                                          ; preds = %for.cond17
  br label %for.inc28, !dbg !1569

for.inc28:                                        ; preds = %for.end
  %indvars.iv.next15 = add nuw nsw i64 %indvars.iv14, 1, !dbg !1570
  call void @llvm.dbg.value(metadata i32 undef, metadata !1536, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !1455
  br label %for.cond, !dbg !1571, !llvm.loop !1572

for.end30:                                        ; preds = %for.cond
  call void @llvm.dbg.value(metadata i32 0, metadata !1574, metadata !DIExpression()), !dbg !1455
  br label %for.cond31, !dbg !1575

for.cond31:                                       ; preds = %for.inc36, %for.end30
  %indvars.iv9 = phi i64 [ %indvars.iv.next10, %for.inc36 ], [ 0, %for.end30 ], !dbg !1577
  call void @llvm.dbg.value(metadata i64 %indvars.iv9, metadata !1574, metadata !DIExpression()), !dbg !1455
  %exitcond11 = icmp ne i64 %indvars.iv9, 14001, !dbg !1578
  br i1 %exitcond11, label %for.body33, label %for.end38, !dbg !1580

for.body33:                                       ; preds = %for.cond31
  %44 = load double*, double** @_ZL1x, align 8, !dbg !1581
  %arrayidx35 = getelementptr inbounds double, double* %44, i64 %indvars.iv9, !dbg !1581
  store double 1.000000e+00, double* %arrayidx35, align 8, !dbg !1583
  br label %for.inc36, !dbg !1584

for.inc36:                                        ; preds = %for.body33
  %indvars.iv.next10 = add nuw nsw i64 %indvars.iv9, 1, !dbg !1585
  call void @llvm.dbg.value(metadata i32 undef, metadata !1574, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !1455
  br label %for.cond31, !dbg !1586, !llvm.loop !1587

for.end38:                                        ; preds = %for.cond31
  call void @llvm.dbg.value(metadata i32 0, metadata !1536, metadata !DIExpression()), !dbg !1455
  br label %for.cond39, !dbg !1589

for.cond39:                                       ; preds = %for.inc52, %for.end38
  %indvars.iv7 = phi i64 [ %indvars.iv.next8, %for.inc52 ], [ 0, %for.end38 ], !dbg !1591
  call void @llvm.dbg.value(metadata i64 %indvars.iv7, metadata !1536, metadata !DIExpression()), !dbg !1455
  %45 = load i32, i32* @_ZL7lastcol, align 4, !dbg !1592
  %46 = load i32, i32* @_ZL8firstcol, align 4, !dbg !1594
  %sub40 = sub nsw i32 %45, %46, !dbg !1595
  %add41 = add nsw i32 %sub40, 1, !dbg !1596
  %47 = sext i32 %add41 to i64, !dbg !1597
  %cmp42 = icmp slt i64 %indvars.iv7, %47, !dbg !1597
  br i1 %cmp42, label %for.body43, label %for.end54, !dbg !1598

for.body43:                                       ; preds = %for.cond39
  %48 = load double*, double** @_ZL1q, align 8, !dbg !1599
  %arrayidx45 = getelementptr inbounds double, double* %48, i64 %indvars.iv7, !dbg !1599
  store double 0.000000e+00, double* %arrayidx45, align 8, !dbg !1601
  %49 = load double*, double** @_ZL1z, align 8, !dbg !1602
  %arrayidx47 = getelementptr inbounds double, double* %49, i64 %indvars.iv7, !dbg !1602
  store double 0.000000e+00, double* %arrayidx47, align 8, !dbg !1603
  %50 = load double*, double** @_ZL1r, align 8, !dbg !1604
  %arrayidx49 = getelementptr inbounds double, double* %50, i64 %indvars.iv7, !dbg !1604
  store double 0.000000e+00, double* %arrayidx49, align 8, !dbg !1605
  %51 = load double*, double** @_ZL1p, align 8, !dbg !1606
  %arrayidx51 = getelementptr inbounds double, double* %51, i64 %indvars.iv7, !dbg !1606
  store double 0.000000e+00, double* %arrayidx51, align 8, !dbg !1607
  br label %for.inc52, !dbg !1608

for.inc52:                                        ; preds = %for.body43
  %indvars.iv.next8 = add nuw nsw i64 %indvars.iv7, 1, !dbg !1609
  call void @llvm.dbg.value(metadata i32 undef, metadata !1536, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !1455
  br label %for.cond39, !dbg !1610, !llvm.loop !1611

for.end54:                                        ; preds = %for.cond39
  call void @llvm.dbg.value(metadata double 0.000000e+00, metadata !1519, metadata !DIExpression()), !dbg !1455
  call void @llvm.dbg.value(metadata i32 1, metadata !1613, metadata !DIExpression()), !dbg !1455
  br label %for.cond55, !dbg !1614

for.cond55:                                       ; preds = %for.inc91, %for.end54
  %it.0 = phi i32 [ 1, %for.end54 ], [ %inc92, %for.inc91 ], !dbg !1616
  call void @llvm.dbg.value(metadata i32 %it.0, metadata !1613, metadata !DIExpression()), !dbg !1455
  %exitcond6 = icmp ne i32 %it.0, 2, !dbg !1617
  br i1 %exitcond6, label %for.body57, label %for.end93, !dbg !1619

for.body57:                                       ; preds = %for.cond55
  %52 = load i32*, i32** @_ZL6colidx, align 8, !dbg !1620
  %53 = load i32*, i32** @_ZL6rowstr, align 8, !dbg !1622
  %54 = load double*, double** @_ZL1x, align 8, !dbg !1623
  %55 = load double*, double** @_ZL1z, align 8, !dbg !1624
  %56 = load double*, double** @_ZL1a, align 8, !dbg !1625
  %57 = load double*, double** @_ZL1p, align 8, !dbg !1626
  %58 = load double*, double** @_ZL1q, align 8, !dbg !1627
  %59 = load double*, double** @_ZL1r, align 8, !dbg !1628
  call void @_ZL9conj_gradPiS_PdS0_S0_S0_S0_S0_S0_(i32* %52, i32* %53, double* %54, double* %55, double* %56, double* %57, double* %58, double* %59, double* %rnorm), !dbg !1629
  store double 0.000000e+00, double* %norm_temp1, align 8, !dbg !1630
  store double 0.000000e+00, double* %norm_temp2, align 8, !dbg !1631
  call void @llvm.dbg.value(metadata i32 0, metadata !1536, metadata !DIExpression()), !dbg !1455
  br label %for.cond58, !dbg !1632

for.cond58:                                       ; preds = %for.inc74, %for.body57
  %indvars.iv2 = phi i64 [ %indvars.iv.next3, %for.inc74 ], [ 0, %for.body57 ], !dbg !1634
  call void @llvm.dbg.value(metadata i64 %indvars.iv2, metadata !1536, metadata !DIExpression()), !dbg !1455
  %60 = load i32, i32* @_ZL7lastcol, align 4, !dbg !1635
  %61 = load i32, i32* @_ZL8firstcol, align 4, !dbg !1637
  %sub59 = sub nsw i32 %60, %61, !dbg !1638
  %add60 = add nsw i32 %sub59, 1, !dbg !1639
  %62 = sext i32 %add60 to i64, !dbg !1640
  %cmp61 = icmp slt i64 %indvars.iv2, %62, !dbg !1640
  br i1 %cmp61, label %for.body62, label %for.end76, !dbg !1641

for.body62:                                       ; preds = %for.cond58
  %63 = load double, double* %norm_temp1, align 8, !dbg !1642
  %64 = load double*, double** @_ZL1x, align 8, !dbg !1644
  %arrayidx64 = getelementptr inbounds double, double* %64, i64 %indvars.iv2, !dbg !1644
  %65 = load double, double* %arrayidx64, align 8, !dbg !1644
  %66 = load double*, double** @_ZL1z, align 8, !dbg !1645
  %arrayidx66 = getelementptr inbounds double, double* %66, i64 %indvars.iv2, !dbg !1645
  %67 = load double, double* %arrayidx66, align 8, !dbg !1645
  %mul = fmul contract double %65, %67, !dbg !1646
  %add67 = fadd contract double %63, %mul, !dbg !1647
  store double %add67, double* %norm_temp1, align 8, !dbg !1648
  %68 = load double, double* %norm_temp2, align 8, !dbg !1649
  %69 = load double*, double** @_ZL1z, align 8, !dbg !1650
  %arrayidx69 = getelementptr inbounds double, double* %69, i64 %indvars.iv2, !dbg !1650
  %70 = load double, double* %arrayidx69, align 8, !dbg !1650
  %71 = load double*, double** @_ZL1z, align 8, !dbg !1651
  %arrayidx71 = getelementptr inbounds double, double* %71, i64 %indvars.iv2, !dbg !1651
  %72 = load double, double* %arrayidx71, align 8, !dbg !1651
  %mul72 = fmul contract double %70, %72, !dbg !1652
  %add73 = fadd contract double %68, %mul72, !dbg !1653
  store double %add73, double* %norm_temp2, align 8, !dbg !1654
  br label %for.inc74, !dbg !1655

for.inc74:                                        ; preds = %for.body62
  %indvars.iv.next3 = add nuw nsw i64 %indvars.iv2, 1, !dbg !1656
  call void @llvm.dbg.value(metadata i32 undef, metadata !1536, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !1455
  br label %for.cond58, !dbg !1657, !llvm.loop !1658

for.end76:                                        ; preds = %for.cond58
  %73 = load double, double* %norm_temp2, align 8, !dbg !1660
  %call77 = call double @sqrt(double %73) #11, !dbg !1661
  %div = fdiv double 1.000000e+00, %call77, !dbg !1662
  store double %div, double* %norm_temp2, align 8, !dbg !1663
  call void @llvm.dbg.value(metadata i32 0, metadata !1536, metadata !DIExpression()), !dbg !1455
  br label %for.cond78, !dbg !1664

for.cond78:                                       ; preds = %for.inc88, %for.end76
  %indvars.iv4 = phi i64 [ %indvars.iv.next5, %for.inc88 ], [ 0, %for.end76 ], !dbg !1666
  call void @llvm.dbg.value(metadata i64 %indvars.iv4, metadata !1536, metadata !DIExpression()), !dbg !1455
  %74 = load i32, i32* @_ZL7lastcol, align 4, !dbg !1667
  %75 = load i32, i32* @_ZL8firstcol, align 4, !dbg !1669
  %sub79 = sub nsw i32 %74, %75, !dbg !1670
  %add80 = add nsw i32 %sub79, 1, !dbg !1671
  %76 = sext i32 %add80 to i64, !dbg !1672
  %cmp81 = icmp slt i64 %indvars.iv4, %76, !dbg !1672
  br i1 %cmp81, label %for.body82, label %for.end90, !dbg !1673

for.body82:                                       ; preds = %for.cond78
  %77 = load double, double* %norm_temp2, align 8, !dbg !1674
  %78 = load double*, double** @_ZL1z, align 8, !dbg !1676
  %arrayidx84 = getelementptr inbounds double, double* %78, i64 %indvars.iv4, !dbg !1676
  %79 = load double, double* %arrayidx84, align 8, !dbg !1676
  %mul85 = fmul contract double %77, %79, !dbg !1677
  %80 = load double*, double** @_ZL1x, align 8, !dbg !1678
  %arrayidx87 = getelementptr inbounds double, double* %80, i64 %indvars.iv4, !dbg !1678
  store double %mul85, double* %arrayidx87, align 8, !dbg !1679
  br label %for.inc88, !dbg !1680

for.inc88:                                        ; preds = %for.body82
  %indvars.iv.next5 = add nuw nsw i64 %indvars.iv4, 1, !dbg !1681
  call void @llvm.dbg.value(metadata i32 undef, metadata !1536, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !1455
  br label %for.cond78, !dbg !1682, !llvm.loop !1683

for.end90:                                        ; preds = %for.cond78
  br label %for.inc91, !dbg !1685

for.inc91:                                        ; preds = %for.end90
  %inc92 = add nuw nsw i32 %it.0, 1, !dbg !1686
  call void @llvm.dbg.value(metadata i32 %inc92, metadata !1613, metadata !DIExpression()), !dbg !1455
  br label %for.cond55, !dbg !1687, !llvm.loop !1688

for.end93:                                        ; preds = %for.cond55
  call void @llvm.dbg.value(metadata i32 0, metadata !1574, metadata !DIExpression()), !dbg !1455
  br label %for.cond94, !dbg !1690

for.cond94:                                       ; preds = %for.inc99, %for.end93
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc99 ], [ 0, %for.end93 ], !dbg !1692
  call void @llvm.dbg.value(metadata i64 %indvars.iv, metadata !1574, metadata !DIExpression()), !dbg !1455
  %exitcond1 = icmp ne i64 %indvars.iv, 14001, !dbg !1693
  br i1 %exitcond1, label %for.body96, label %for.end101, !dbg !1695

for.body96:                                       ; preds = %for.cond94
  %81 = load double*, double** @_ZL1x, align 8, !dbg !1696
  %arrayidx98 = getelementptr inbounds double, double* %81, i64 %indvars.iv, !dbg !1696
  store double 1.000000e+00, double* %arrayidx98, align 8, !dbg !1698
  br label %for.inc99, !dbg !1699

for.inc99:                                        ; preds = %for.body96
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !1700
  call void @llvm.dbg.value(metadata i32 undef, metadata !1574, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !1455
  br label %for.cond94, !dbg !1701, !llvm.loop !1702

for.end101:                                       ; preds = %for.cond94
  call void @llvm.dbg.value(metadata double 0.000000e+00, metadata !1519, metadata !DIExpression()), !dbg !1455
  call void @_ZL9setup_gpuv(), !dbg !1704
  call void @llvm.dbg.value(metadata i32 1, metadata !1613, metadata !DIExpression()), !dbg !1455
  br label %for.cond102, !dbg !1705

for.cond102:                                      ; preds = %for.inc112, %for.end101
  %it.1 = phi i32 [ 1, %for.end101 ], [ %inc113, %for.inc112 ], !dbg !1707
  %zeta.0 = phi double [ 0.000000e+00, %for.end101 ], [ %add108, %for.inc112 ], !dbg !1455
  call void @llvm.dbg.value(metadata double %zeta.0, metadata !1519, metadata !DIExpression()), !dbg !1455
  call void @llvm.dbg.value(metadata i32 %it.1, metadata !1613, metadata !DIExpression()), !dbg !1455
  %exitcond = icmp ne i32 %it.1, 16, !dbg !1708
  br i1 %exitcond, label %for.body104, label %for.end114, !dbg !1710

for.body104:                                      ; preds = %for.cond102
  call void @_ZL13conj_grad_gpuPd(double* %rnorm), !dbg !1711
  call void @_ZL19gpu_kernel_ten_hostPdS_(double* %norm_temp1, double* %norm_temp2), !dbg !1713
  %82 = load double, double* %norm_temp2, align 8, !dbg !1714
  %call105 = call double @sqrt(double %82) #11, !dbg !1715
  %div106 = fdiv double 1.000000e+00, %call105, !dbg !1716
  store double %div106, double* %norm_temp2, align 8, !dbg !1717
  %83 = load double, double* %norm_temp1, align 8, !dbg !1718
  %div107 = fdiv double 1.000000e+00, %83, !dbg !1719
  %add108 = fadd contract double 2.000000e+01, %div107, !dbg !1720
  call void @llvm.dbg.value(metadata double %add108, metadata !1519, metadata !DIExpression()), !dbg !1455
  %cmp109 = icmp eq i32 %it.1, 1, !dbg !1721
  br i1 %cmp109, label %if.then, label %if.end, !dbg !1723

if.then:                                          ; preds = %for.body104
  %call110 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([52 x i8], [52 x i8]* @.str.43, i64 0, i64 0)), !dbg !1724
  br label %if.end, !dbg !1726

if.end:                                           ; preds = %if.then, %for.body104
  %84 = load double, double* %rnorm, align 8, !dbg !1727
  %call111 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([30 x i8], [30 x i8]* @.str.44, i64 0, i64 0), i32 %it.1, double %84, double %add108), !dbg !1728
  %85 = load double, double* %norm_temp2, align 8, !dbg !1729
  call void @_ZL22gpu_kernel_eleven_hostd(double %85), !dbg !1730
  br label %for.inc112, !dbg !1731

for.inc112:                                       ; preds = %if.end
  %inc113 = add nuw nsw i32 %it.1, 1, !dbg !1732
  call void @llvm.dbg.value(metadata i32 %inc113, metadata !1613, metadata !DIExpression()), !dbg !1455
  br label %for.cond102, !dbg !1733, !llvm.loop !1734

for.end114:                                       ; preds = %for.cond102
  %zeta.0.lcssa = phi double [ %zeta.0, %for.cond102 ], !dbg !1455
  call void @llvm.dbg.value(metadata double %zeta.0.lcssa, metadata !1519, metadata !DIExpression()), !dbg !1455
  call void @llvm.dbg.value(metadata double 0.000000e+00, metadata !1736, metadata !DIExpression()), !dbg !1455
  %call115 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([22 x i8], [22 x i8]* @.str.45, i64 0, i64 0)), !dbg !1737
  call void @llvm.dbg.value(metadata double 1.000000e-10, metadata !1738, metadata !DIExpression()), !dbg !1455
  %conv = sext i8 65 to i32, !dbg !1739
  %cmp116 = icmp ne i32 %conv, 85, !dbg !1741
  br i1 %cmp116, label %if.then117, label %if.else129, !dbg !1742

if.then117:                                       ; preds = %for.end114
  %sub118 = fsub contract double %zeta.0.lcssa, 0x4031215715A1D8EC, !dbg !1743
  %86 = call double @llvm.fabs.f64(double %sub118), !dbg !1745
  %div119 = fdiv double %86, 0x4031215715A1D8EC, !dbg !1746
  call void @llvm.dbg.value(metadata double %div119, metadata !1747, metadata !DIExpression()), !dbg !1455
  %cmp120 = fcmp ole double %div119, 1.000000e-10, !dbg !1748
  br i1 %cmp120, label %if.then121, label %if.else, !dbg !1750

if.then121:                                       ; preds = %if.then117
  call void @llvm.dbg.value(metadata i32 1, metadata !1751, metadata !DIExpression()), !dbg !1455
  %call122 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([26 x i8], [26 x i8]* @.str.46, i64 0, i64 0)), !dbg !1754
  %call123 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([21 x i8], [21 x i8]* @.str.47, i64 0, i64 0), double %zeta.0.lcssa), !dbg !1756
  %call124 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([21 x i8], [21 x i8]* @.str.48, i64 0, i64 0), double %div119), !dbg !1757
  br label %if.end128, !dbg !1758

if.else:                                          ; preds = %if.then117
  call void @llvm.dbg.value(metadata i32 0, metadata !1751, metadata !DIExpression()), !dbg !1455
  %call125 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([22 x i8], [22 x i8]* @.str.49, i64 0, i64 0)), !dbg !1759
  %call126 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([30 x i8], [30 x i8]* @.str.50, i64 0, i64 0), double %zeta.0.lcssa), !dbg !1761
  %call127 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([30 x i8], [30 x i8]* @.str.51, i64 0, i64 0), double 0x4031215715A1D8EC), !dbg !1762
  br label %if.end128

if.end128:                                        ; preds = %if.else, %if.then121
  %verified.0 = phi i32 [ 1, %if.then121 ], [ 0, %if.else ], !dbg !1763
  call void @llvm.dbg.value(metadata i32 %verified.0, metadata !1751, metadata !DIExpression()), !dbg !1455
  br label %if.end132, !dbg !1764

if.else129:                                       ; preds = %for.end114
  call void @llvm.dbg.value(metadata i32 0, metadata !1751, metadata !DIExpression()), !dbg !1455
  %call130 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.52, i64 0, i64 0)), !dbg !1765
  %call131 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([28 x i8], [28 x i8]* @.str.53, i64 0, i64 0)), !dbg !1767
  br label %if.end132

if.end132:                                        ; preds = %if.else129, %if.end128
  %verified.1 = phi i32 [ %verified.0, %if.end128 ], [ 0, %if.else129 ], !dbg !1768
  call void @llvm.dbg.value(metadata i32 %verified.1, metadata !1751, metadata !DIExpression()), !dbg !1455
  %cmp133 = fcmp une double 0.000000e+00, 0.000000e+00, !dbg !1769
  br i1 %cmp133, label %if.then134, label %if.else137, !dbg !1771

if.then134:                                       ; preds = %if.end132
  %div135 = fdiv double 1.496460e+09, 0.000000e+00, !dbg !1772
  %div136 = fdiv double %div135, 1.000000e+06, !dbg !1774
  call void @llvm.dbg.value(metadata double %div136, metadata !1775, metadata !DIExpression()), !dbg !1455
  br label %if.end138, !dbg !1776

if.else137:                                       ; preds = %if.end132
  call void @llvm.dbg.value(metadata double 0.000000e+00, metadata !1775, metadata !DIExpression()), !dbg !1455
  br label %if.end138

if.end138:                                        ; preds = %if.else137, %if.then134
  %mflops.0 = phi double [ %div136, %if.then134 ], [ 0.000000e+00, %if.else137 ], !dbg !1777
  call void @llvm.dbg.value(metadata double %mflops.0, metadata !1775, metadata !DIExpression()), !dbg !1455
  call void @llvm.dbg.declare(metadata [256 x i8]* %gpu_config, metadata !1778, metadata !DIExpression()), !dbg !1779
  call void @llvm.dbg.declare(metadata [2048 x i8]* %gpu_config_string, metadata !1780, metadata !DIExpression()), !dbg !1784
  %arraydecay = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1785
  %call139 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay, i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.54, i64 0, i64 0), i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.55, i64 0, i64 0), i8* getelementptr inbounds ([18 x i8], [18 x i8]* @.str.56, i64 0, i64 0)) #11, !dbg !1786
  %arraydecay140 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !1787
  %arraydecay141 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1788
  %call142 = call i8* @strcpy(i8* %arraydecay140, i8* %arraydecay141) #11, !dbg !1789
  %arraydecay143 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1790
  %87 = load i32, i32* @threads_per_block_on_kernel_one, align 4, !dbg !1791
  %call144 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay143, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.57, i64 0, i64 0), i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.58, i64 0, i64 0), i32 %87) #11, !dbg !1792
  %arraydecay145 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !1793
  %arraydecay146 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1794
  %call147 = call i8* @strcat(i8* %arraydecay145, i8* %arraydecay146) #11, !dbg !1795
  %arraydecay148 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1796
  %88 = load i32, i32* @threads_per_block_on_kernel_two, align 4, !dbg !1797
  %call149 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay148, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.57, i64 0, i64 0), i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.59, i64 0, i64 0), i32 %88) #11, !dbg !1798
  %arraydecay150 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !1799
  %arraydecay151 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1800
  %call152 = call i8* @strcat(i8* %arraydecay150, i8* %arraydecay151) #11, !dbg !1801
  %arraydecay153 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1802
  %89 = load i32, i32* @threads_per_block_on_kernel_three, align 4, !dbg !1803
  %call154 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay153, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.57, i64 0, i64 0), i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.60, i64 0, i64 0), i32 %89) #11, !dbg !1804
  %arraydecay155 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !1805
  %arraydecay156 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1806
  %call157 = call i8* @strcat(i8* %arraydecay155, i8* %arraydecay156) #11, !dbg !1807
  %arraydecay158 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1808
  %90 = load i32, i32* @threads_per_block_on_kernel_four, align 4, !dbg !1809
  %call159 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay158, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.57, i64 0, i64 0), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.61, i64 0, i64 0), i32 %90) #11, !dbg !1810
  %arraydecay160 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !1811
  %arraydecay161 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1812
  %call162 = call i8* @strcat(i8* %arraydecay160, i8* %arraydecay161) #11, !dbg !1813
  %arraydecay163 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1814
  %91 = load i32, i32* @threads_per_block_on_kernel_five, align 4, !dbg !1815
  %call164 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay163, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.57, i64 0, i64 0), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.62, i64 0, i64 0), i32 %91) #11, !dbg !1816
  %arraydecay165 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !1817
  %arraydecay166 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1818
  %call167 = call i8* @strcat(i8* %arraydecay165, i8* %arraydecay166) #11, !dbg !1819
  %arraydecay168 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1820
  %92 = load i32, i32* @threads_per_block_on_kernel_six, align 4, !dbg !1821
  %call169 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay168, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.57, i64 0, i64 0), i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.63, i64 0, i64 0), i32 %92) #11, !dbg !1822
  %arraydecay170 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !1823
  %arraydecay171 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1824
  %call172 = call i8* @strcat(i8* %arraydecay170, i8* %arraydecay171) #11, !dbg !1825
  %arraydecay173 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1826
  %93 = load i32, i32* @threads_per_block_on_kernel_seven, align 4, !dbg !1827
  %call174 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay173, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.57, i64 0, i64 0), i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.64, i64 0, i64 0), i32 %93) #11, !dbg !1828
  %arraydecay175 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !1829
  %arraydecay176 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1830
  %call177 = call i8* @strcat(i8* %arraydecay175, i8* %arraydecay176) #11, !dbg !1831
  %arraydecay178 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1832
  %94 = load i32, i32* @threads_per_block_on_kernel_eight, align 4, !dbg !1833
  %call179 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay178, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.57, i64 0, i64 0), i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.65, i64 0, i64 0), i32 %94) #11, !dbg !1834
  %arraydecay180 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !1835
  %arraydecay181 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1836
  %call182 = call i8* @strcat(i8* %arraydecay180, i8* %arraydecay181) #11, !dbg !1837
  %arraydecay183 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1838
  %95 = load i32, i32* @threads_per_block_on_kernel_nine, align 4, !dbg !1839
  %call184 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay183, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.57, i64 0, i64 0), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.66, i64 0, i64 0), i32 %95) #11, !dbg !1840
  %arraydecay185 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !1841
  %arraydecay186 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1842
  %call187 = call i8* @strcat(i8* %arraydecay185, i8* %arraydecay186) #11, !dbg !1843
  %arraydecay188 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1844
  %96 = load i32, i32* @threads_per_block_on_kernel_ten, align 4, !dbg !1845
  %call189 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay188, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.57, i64 0, i64 0), i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.67, i64 0, i64 0), i32 %96) #11, !dbg !1846
  %arraydecay190 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !1847
  %arraydecay191 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1848
  %call192 = call i8* @strcat(i8* %arraydecay190, i8* %arraydecay191) #11, !dbg !1849
  %arraydecay193 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1850
  %97 = load i32, i32* @threads_per_block_on_kernel_eleven, align 4, !dbg !1851
  %call194 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay193, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.57, i64 0, i64 0), i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.68, i64 0, i64 0), i32 %97) #11, !dbg !1852
  %arraydecay195 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !1853
  %arraydecay196 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1854
  %call197 = call i8* @strcat(i8* %arraydecay195, i8* %arraydecay196) #11, !dbg !1855
  %arraydecay198 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !1856
  call void @_Z15c_print_resultsPcciiiiddS_iS_S_S_S_S_S_S_S_S_S_S_S_S_S_(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.69, i64 0, i64 0), i8 signext 65, i32 14000, i32 0, i32 0, i32 15, double 0.000000e+00, double %mflops.0, i8* getelementptr inbounds ([25 x i8], [25 x i8]* @.str.70, i64 0, i64 0), i32 %verified.1, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.71, i64 0, i64 0), i8* getelementptr inbounds ([12 x i8], [12 x i8]* @.str.72, i64 0, i64 0), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.73, i64 0, i64 0), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.73, i64 0, i64 0), i8* getelementptr inbounds ([42 x i8], [42 x i8]* @.str.74, i64 0, i64 0), i8* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 0, i64 0), i8* %arraydecay198, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.75, i64 0, i64 0), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.76, i64 0, i64 0), i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.77, i64 0, i64 0), i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.78, i64 0, i64 0), i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.79, i64 0, i64 0), i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.79, i64 0, i64 0), i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.80, i64 0, i64 0)), !dbg !1857
  call void @_ZL11release_gpuv(), !dbg !1858
  ret i32 0, !dbg !1859
}

; Function Attrs: nounwind
declare dso_local noalias i8* @malloc(i64) #6

; Function Attrs: noinline uwtable
define internal void @_ZL5makeaiiPdPiS0_iiiiS0_PA12_iPA12_dS0_(i32 %n, i32 %nz, double* %a, i32* %colidx, i32* %rowstr, i32 %firstrow, i32 %lastrow, i32 %firstcol, i32 %lastcol, i32* %arow, [12 x i32]* %acol, [12 x double]* %aelt, i32* %iv) #4 !dbg !1860 {
entry:
  %nzv = alloca i32, align 4
  %ivc = alloca [12 x i32], align 16
  %vc = alloca [12 x double], align 16
  call void @llvm.dbg.value(metadata i32 %n, metadata !1863, metadata !DIExpression()), !dbg !1864
  call void @llvm.dbg.value(metadata i32 %nz, metadata !1865, metadata !DIExpression()), !dbg !1864
  call void @llvm.dbg.value(metadata double* %a, metadata !1866, metadata !DIExpression()), !dbg !1864
  call void @llvm.dbg.value(metadata i32* %colidx, metadata !1867, metadata !DIExpression()), !dbg !1864
  call void @llvm.dbg.value(metadata i32* %rowstr, metadata !1868, metadata !DIExpression()), !dbg !1864
  call void @llvm.dbg.value(metadata i32 %firstrow, metadata !1869, metadata !DIExpression()), !dbg !1864
  call void @llvm.dbg.value(metadata i32 %lastrow, metadata !1870, metadata !DIExpression()), !dbg !1864
  call void @llvm.dbg.value(metadata i32 %firstcol, metadata !1871, metadata !DIExpression()), !dbg !1864
  call void @llvm.dbg.value(metadata i32 %lastcol, metadata !1872, metadata !DIExpression()), !dbg !1864
  call void @llvm.dbg.value(metadata i32* %arow, metadata !1873, metadata !DIExpression()), !dbg !1864
  call void @llvm.dbg.value(metadata [12 x i32]* %acol, metadata !1874, metadata !DIExpression()), !dbg !1864
  call void @llvm.dbg.value(metadata [12 x double]* %aelt, metadata !1875, metadata !DIExpression()), !dbg !1864
  call void @llvm.dbg.value(metadata i32* %iv, metadata !1876, metadata !DIExpression()), !dbg !1864
  call void @llvm.dbg.declare(metadata i32* %nzv, metadata !1877, metadata !DIExpression()), !dbg !1878
  call void @llvm.dbg.declare(metadata [12 x i32]* %ivc, metadata !1879, metadata !DIExpression()), !dbg !1880
  call void @llvm.dbg.declare(metadata [12 x double]* %vc, metadata !1881, metadata !DIExpression()), !dbg !1882
  call void @llvm.dbg.value(metadata i32 1, metadata !1883, metadata !DIExpression()), !dbg !1864
  br label %do.body, !dbg !1884

do.body:                                          ; preds = %do.cond, %entry
  %nn1.0 = phi i32 [ 1, %entry ], [ %mul, %do.cond ], !dbg !1864
  call void @llvm.dbg.value(metadata i32 %nn1.0, metadata !1883, metadata !DIExpression()), !dbg !1864
  %mul = mul nuw nsw i32 2, %nn1.0, !dbg !1885
  call void @llvm.dbg.value(metadata i32 %mul, metadata !1883, metadata !DIExpression()), !dbg !1864
  br label %do.cond, !dbg !1887

do.cond:                                          ; preds = %do.body
  %cmp = icmp slt i32 %mul, %n, !dbg !1888
  br i1 %cmp, label %do.body, label %do.end, !dbg !1887, !llvm.loop !1889

do.end:                                           ; preds = %do.cond
  %mul.lcssa = phi i32 [ %mul, %do.cond ], !dbg !1885
  call void @llvm.dbg.value(metadata i32 0, metadata !1891, metadata !DIExpression()), !dbg !1864
  %0 = sext i32 %n to i64, !dbg !1892
  br label %for.cond, !dbg !1892

for.cond:                                         ; preds = %for.inc20, %do.end
  %indvars.iv1 = phi i64 [ %indvars.iv.next2, %for.inc20 ], [ 0, %do.end ], !dbg !1894
  call void @llvm.dbg.value(metadata i64 %indvars.iv1, metadata !1891, metadata !DIExpression()), !dbg !1864
  %cmp1 = icmp slt i64 %indvars.iv1, %0, !dbg !1895
  br i1 %cmp1, label %for.body, label %for.end22, !dbg !1897

for.body:                                         ; preds = %for.cond
  store i32 11, i32* %nzv, align 4, !dbg !1898
  %1 = load i32, i32* %nzv, align 4, !dbg !1900
  %arraydecay = getelementptr inbounds [12 x double], [12 x double]* %vc, i64 0, i64 0, !dbg !1901
  %arraydecay2 = getelementptr inbounds [12 x i32], [12 x i32]* %ivc, i64 0, i64 0, !dbg !1902
  call void @_ZL6sprnvciiiPdPi(i32 %n, i32 %1, i32 %mul.lcssa, double* %arraydecay, i32* %arraydecay2), !dbg !1903
  %arraydecay3 = getelementptr inbounds [12 x double], [12 x double]* %vc, i64 0, i64 0, !dbg !1904
  %arraydecay4 = getelementptr inbounds [12 x i32], [12 x i32]* %ivc, i64 0, i64 0, !dbg !1905
  %indvars.iv.next2 = add nuw nsw i64 %indvars.iv1, 1, !dbg !1906
  %2 = trunc i64 %indvars.iv.next2 to i32, !dbg !1907
  call void @_ZL6vecsetiPdPiS0_id(i32 %n, double* %arraydecay3, i32* %arraydecay4, i32* %nzv, i32 %2, double 5.000000e-01), !dbg !1907
  %3 = load i32, i32* %nzv, align 4, !dbg !1908
  %arrayidx = getelementptr inbounds i32, i32* %arow, i64 %indvars.iv1, !dbg !1909
  store i32 %3, i32* %arrayidx, align 4, !dbg !1910
  call void @llvm.dbg.value(metadata i32 0, metadata !1911, metadata !DIExpression()), !dbg !1864
  br label %for.cond5, !dbg !1912

for.cond5:                                        ; preds = %for.inc, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %for.body ], !dbg !1914
  call void @llvm.dbg.value(metadata i64 %indvars.iv, metadata !1911, metadata !DIExpression()), !dbg !1864
  %4 = load i32, i32* %nzv, align 4, !dbg !1915
  %5 = sext i32 %4 to i64, !dbg !1917
  %cmp6 = icmp slt i64 %indvars.iv, %5, !dbg !1917
  br i1 %cmp6, label %for.body7, label %for.end, !dbg !1918

for.body7:                                        ; preds = %for.cond5
  %arrayidx9 = getelementptr inbounds [12 x i32], [12 x i32]* %ivc, i64 0, i64 %indvars.iv, !dbg !1919
  %6 = load i32, i32* %arrayidx9, align 4, !dbg !1919
  %sub = sub nsw i32 %6, 1, !dbg !1921
  %arrayidx11 = getelementptr inbounds [12 x i32], [12 x i32]* %acol, i64 %indvars.iv1, !dbg !1922
  %arrayidx13 = getelementptr inbounds [12 x i32], [12 x i32]* %arrayidx11, i64 0, i64 %indvars.iv, !dbg !1922
  store i32 %sub, i32* %arrayidx13, align 4, !dbg !1923
  %arrayidx15 = getelementptr inbounds [12 x double], [12 x double]* %vc, i64 0, i64 %indvars.iv, !dbg !1924
  %7 = load double, double* %arrayidx15, align 8, !dbg !1924
  %arrayidx17 = getelementptr inbounds [12 x double], [12 x double]* %aelt, i64 %indvars.iv1, !dbg !1925
  %arrayidx19 = getelementptr inbounds [12 x double], [12 x double]* %arrayidx17, i64 0, i64 %indvars.iv, !dbg !1925
  store double %7, double* %arrayidx19, align 8, !dbg !1926
  br label %for.inc, !dbg !1927

for.inc:                                          ; preds = %for.body7
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !1928
  call void @llvm.dbg.value(metadata i32 undef, metadata !1911, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !1864
  br label %for.cond5, !dbg !1929, !llvm.loop !1930

for.end:                                          ; preds = %for.cond5
  br label %for.inc20, !dbg !1932

for.inc20:                                        ; preds = %for.end
  call void @llvm.dbg.value(metadata i32 undef, metadata !1891, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !1864
  br label %for.cond, !dbg !1933, !llvm.loop !1934

for.end22:                                        ; preds = %for.cond
  call void @_ZL6sparsePdPiS0_iiiS0_PA12_iPA12_diiS0_dd(double* %a, i32* %colidx, i32* %rowstr, i32 %n, i32 %nz, i32 11, i32* %arow, [12 x i32]* %acol, [12 x double]* %aelt, i32 %firstrow, i32 %lastrow, i32* %iv, double 1.000000e-01, double 2.000000e+01), !dbg !1936
  ret void, !dbg !1937
}

; Function Attrs: noinline nounwind uwtable
define internal void @_ZL9conj_gradPiS_PdS0_S0_S0_S0_S0_S0_(i32* %colidx, i32* %rowstr, double* %x, double* %z, double* %a, double* %p, double* %q, double* %r, double* %rnorm) #3 !dbg !1938 {
entry:
  call void @llvm.dbg.value(metadata i32* %colidx, metadata !1941, metadata !DIExpression()), !dbg !1942
  call void @llvm.dbg.value(metadata i32* %rowstr, metadata !1943, metadata !DIExpression()), !dbg !1942
  call void @llvm.dbg.value(metadata double* %x, metadata !1944, metadata !DIExpression()), !dbg !1942
  call void @llvm.dbg.value(metadata double* %z, metadata !1945, metadata !DIExpression()), !dbg !1942
  call void @llvm.dbg.value(metadata double* %a, metadata !1946, metadata !DIExpression()), !dbg !1942
  call void @llvm.dbg.value(metadata double* %p, metadata !1947, metadata !DIExpression()), !dbg !1942
  call void @llvm.dbg.value(metadata double* %q, metadata !1948, metadata !DIExpression()), !dbg !1942
  call void @llvm.dbg.value(metadata double* %r, metadata !1949, metadata !DIExpression()), !dbg !1942
  call void @llvm.dbg.value(metadata double* %rnorm, metadata !1950, metadata !DIExpression()), !dbg !1942
  call void @llvm.dbg.value(metadata i32 25, metadata !1951, metadata !DIExpression()), !dbg !1942
  call void @llvm.dbg.value(metadata double 0.000000e+00, metadata !1952, metadata !DIExpression()), !dbg !1942
  call void @llvm.dbg.value(metadata i32 0, metadata !1953, metadata !DIExpression()), !dbg !1942
  br label %for.cond, !dbg !1954

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv21 = phi i64 [ %indvars.iv.next22, %for.inc ], [ 0, %entry ], !dbg !1956
  call void @llvm.dbg.value(metadata i64 %indvars.iv21, metadata !1953, metadata !DIExpression()), !dbg !1942
  %0 = load i32, i32* @_ZL3naa, align 4, !dbg !1957
  %add = add nsw i32 %0, 1, !dbg !1959
  %1 = sext i32 %add to i64, !dbg !1960
  %cmp = icmp slt i64 %indvars.iv21, %1, !dbg !1960
  br i1 %cmp, label %for.body, label %for.end, !dbg !1961

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds double, double* %q, i64 %indvars.iv21, !dbg !1962
  store double 0.000000e+00, double* %arrayidx, align 8, !dbg !1964
  %arrayidx2 = getelementptr inbounds double, double* %z, i64 %indvars.iv21, !dbg !1965
  store double 0.000000e+00, double* %arrayidx2, align 8, !dbg !1966
  %arrayidx4 = getelementptr inbounds double, double* %x, i64 %indvars.iv21, !dbg !1967
  %2 = load double, double* %arrayidx4, align 8, !dbg !1967
  %arrayidx6 = getelementptr inbounds double, double* %r, i64 %indvars.iv21, !dbg !1968
  store double %2, double* %arrayidx6, align 8, !dbg !1969
  %arrayidx8 = getelementptr inbounds double, double* %r, i64 %indvars.iv21, !dbg !1970
  %3 = load double, double* %arrayidx8, align 8, !dbg !1970
  %arrayidx10 = getelementptr inbounds double, double* %p, i64 %indvars.iv21, !dbg !1971
  store double %3, double* %arrayidx10, align 8, !dbg !1972
  br label %for.inc, !dbg !1973

for.inc:                                          ; preds = %for.body
  %indvars.iv.next22 = add nuw nsw i64 %indvars.iv21, 1, !dbg !1974
  call void @llvm.dbg.value(metadata i32 undef, metadata !1953, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !1942
  br label %for.cond, !dbg !1975, !llvm.loop !1976

for.end:                                          ; preds = %for.cond
  call void @llvm.dbg.value(metadata i32 0, metadata !1953, metadata !DIExpression()), !dbg !1942
  br label %for.cond11, !dbg !1978

for.cond11:                                       ; preds = %for.inc20, %for.end
  %indvars.iv19 = phi i64 [ %indvars.iv.next20, %for.inc20 ], [ 0, %for.end ], !dbg !1980
  %rho.0 = phi double [ 0.000000e+00, %for.end ], [ %add19, %for.inc20 ], !dbg !1942
  call void @llvm.dbg.value(metadata double %rho.0, metadata !1952, metadata !DIExpression()), !dbg !1942
  call void @llvm.dbg.value(metadata i64 %indvars.iv19, metadata !1953, metadata !DIExpression()), !dbg !1942
  %4 = load i32, i32* @_ZL7lastcol, align 4, !dbg !1981
  %5 = load i32, i32* @_ZL8firstcol, align 4, !dbg !1983
  %sub = sub nsw i32 %4, %5, !dbg !1984
  %add12 = add nsw i32 %sub, 1, !dbg !1985
  %6 = sext i32 %add12 to i64, !dbg !1986
  %cmp13 = icmp slt i64 %indvars.iv19, %6, !dbg !1986
  br i1 %cmp13, label %for.body14, label %for.end22, !dbg !1987

for.body14:                                       ; preds = %for.cond11
  %arrayidx16 = getelementptr inbounds double, double* %r, i64 %indvars.iv19, !dbg !1988
  %7 = load double, double* %arrayidx16, align 8, !dbg !1988
  %arrayidx18 = getelementptr inbounds double, double* %r, i64 %indvars.iv19, !dbg !1990
  %8 = load double, double* %arrayidx18, align 8, !dbg !1990
  %mul = fmul contract double %7, %8, !dbg !1991
  %add19 = fadd contract double %rho.0, %mul, !dbg !1992
  call void @llvm.dbg.value(metadata double %add19, metadata !1952, metadata !DIExpression()), !dbg !1942
  br label %for.inc20, !dbg !1993

for.inc20:                                        ; preds = %for.body14
  %indvars.iv.next20 = add nuw nsw i64 %indvars.iv19, 1, !dbg !1994
  call void @llvm.dbg.value(metadata i32 undef, metadata !1953, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !1942
  br label %for.cond11, !dbg !1995, !llvm.loop !1996

for.end22:                                        ; preds = %for.cond11
  %rho.0.lcssa = phi double [ %rho.0, %for.cond11 ], !dbg !1942
  call void @llvm.dbg.value(metadata double %rho.0.lcssa, metadata !1952, metadata !DIExpression()), !dbg !1942
  call void @llvm.dbg.value(metadata i32 1, metadata !1998, metadata !DIExpression()), !dbg !1942
  br label %for.cond23, !dbg !1999

for.cond23:                                       ; preds = %for.inc124, %for.end22
  %cgit.0 = phi i32 [ 1, %for.end22 ], [ %inc125, %for.inc124 ], !dbg !2001
  %rho.1 = phi double [ %rho.0.lcssa, %for.end22 ], [ %rho.2.lcssa, %for.inc124 ], !dbg !1942
  call void @llvm.dbg.value(metadata double %rho.1, metadata !1952, metadata !DIExpression()), !dbg !1942
  call void @llvm.dbg.value(metadata i32 %cgit.0, metadata !1998, metadata !DIExpression()), !dbg !1942
  %exitcond = icmp ne i32 %cgit.0, 26, !dbg !2002
  br i1 %exitcond, label %for.body25, label %for.end126, !dbg !2004

for.body25:                                       ; preds = %for.cond23
  call void @llvm.dbg.value(metadata i32 0, metadata !1953, metadata !DIExpression()), !dbg !1942
  br label %for.cond26, !dbg !2005

for.cond26:                                       ; preds = %for.inc52, %for.body25
  %indvars.iv8 = phi i64 [ %indvars.iv.next9, %for.inc52 ], [ 0, %for.body25 ], !dbg !2008
  call void @llvm.dbg.value(metadata i64 %indvars.iv8, metadata !1953, metadata !DIExpression()), !dbg !1942
  %9 = load i32, i32* @_ZL7lastrow, align 4, !dbg !2009
  %10 = load i32, i32* @_ZL8firstrow, align 4, !dbg !2011
  %sub27 = sub nsw i32 %9, %10, !dbg !2012
  %add28 = add nsw i32 %sub27, 1, !dbg !2013
  %11 = sext i32 %add28 to i64, !dbg !2014
  %cmp29 = icmp slt i64 %indvars.iv8, %11, !dbg !2014
  br i1 %cmp29, label %for.body30, label %for.end54, !dbg !2015

for.body30:                                       ; preds = %for.cond26
  call void @llvm.dbg.value(metadata double 0.000000e+00, metadata !2016, metadata !DIExpression()), !dbg !1942
  %arrayidx32 = getelementptr inbounds i32, i32* %rowstr, i64 %indvars.iv8, !dbg !2017
  %12 = load i32, i32* %arrayidx32, align 4, !dbg !2017
  call void @llvm.dbg.value(metadata i32 %12, metadata !2020, metadata !DIExpression()), !dbg !1942
  %13 = sext i32 %12 to i64, !dbg !2021
  br label %for.cond33, !dbg !2021

for.cond33:                                       ; preds = %for.inc47, %for.body30
  %indvars.iv6 = phi i64 [ %indvars.iv.next7, %for.inc47 ], [ %13, %for.body30 ], !dbg !2022
  %sum.0 = phi double [ 0.000000e+00, %for.body30 ], [ %add46, %for.inc47 ], !dbg !2023
  call void @llvm.dbg.value(metadata double %sum.0, metadata !2016, metadata !DIExpression()), !dbg !1942
  call void @llvm.dbg.value(metadata i64 %indvars.iv6, metadata !2020, metadata !DIExpression()), !dbg !1942
  %14 = add nuw nsw i64 %indvars.iv8, 1, !dbg !2024
  %arrayidx36 = getelementptr inbounds i32, i32* %rowstr, i64 %14, !dbg !2026
  %15 = load i32, i32* %arrayidx36, align 4, !dbg !2026
  %16 = sext i32 %15 to i64, !dbg !2027
  %cmp37 = icmp slt i64 %indvars.iv6, %16, !dbg !2027
  br i1 %cmp37, label %for.body38, label %for.end49, !dbg !2028

for.body38:                                       ; preds = %for.cond33
  %arrayidx40 = getelementptr inbounds double, double* %a, i64 %indvars.iv6, !dbg !2029
  %17 = load double, double* %arrayidx40, align 8, !dbg !2029
  %arrayidx42 = getelementptr inbounds i32, i32* %colidx, i64 %indvars.iv6, !dbg !2031
  %18 = load i32, i32* %arrayidx42, align 4, !dbg !2031
  %idxprom43 = sext i32 %18 to i64, !dbg !2032
  %arrayidx44 = getelementptr inbounds double, double* %p, i64 %idxprom43, !dbg !2032
  %19 = load double, double* %arrayidx44, align 8, !dbg !2032
  %mul45 = fmul contract double %17, %19, !dbg !2033
  %add46 = fadd contract double %sum.0, %mul45, !dbg !2034
  call void @llvm.dbg.value(metadata double %add46, metadata !2016, metadata !DIExpression()), !dbg !1942
  br label %for.inc47, !dbg !2035

for.inc47:                                        ; preds = %for.body38
  %indvars.iv.next7 = add nsw i64 %indvars.iv6, 1, !dbg !2036
  call void @llvm.dbg.value(metadata i32 undef, metadata !2020, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !1942
  br label %for.cond33, !dbg !2037, !llvm.loop !2038

for.end49:                                        ; preds = %for.cond33
  %sum.0.lcssa = phi double [ %sum.0, %for.cond33 ], !dbg !2023
  call void @llvm.dbg.value(metadata double %sum.0.lcssa, metadata !2016, metadata !DIExpression()), !dbg !1942
  %arrayidx51 = getelementptr inbounds double, double* %q, i64 %indvars.iv8, !dbg !2040
  store double %sum.0.lcssa, double* %arrayidx51, align 8, !dbg !2041
  br label %for.inc52, !dbg !2042

for.inc52:                                        ; preds = %for.end49
  %indvars.iv.next9 = add nuw nsw i64 %indvars.iv8, 1, !dbg !2043
  call void @llvm.dbg.value(metadata i32 undef, metadata !1953, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !1942
  br label %for.cond26, !dbg !2044, !llvm.loop !2045

for.end54:                                        ; preds = %for.cond26
  call void @llvm.dbg.value(metadata double 0.000000e+00, metadata !2047, metadata !DIExpression()), !dbg !1942
  call void @llvm.dbg.value(metadata i32 0, metadata !1953, metadata !DIExpression()), !dbg !1942
  br label %for.cond55, !dbg !2048

for.cond55:                                       ; preds = %for.inc66, %for.end54
  %indvars.iv11 = phi i64 [ %indvars.iv.next12, %for.inc66 ], [ 0, %for.end54 ], !dbg !2050
  %d.0 = phi double [ 0.000000e+00, %for.end54 ], [ %add65, %for.inc66 ], !dbg !2051
  call void @llvm.dbg.value(metadata double %d.0, metadata !2047, metadata !DIExpression()), !dbg !1942
  call void @llvm.dbg.value(metadata i64 %indvars.iv11, metadata !1953, metadata !DIExpression()), !dbg !1942
  %20 = load i32, i32* @_ZL7lastcol, align 4, !dbg !2052
  %21 = load i32, i32* @_ZL8firstcol, align 4, !dbg !2054
  %sub56 = sub nsw i32 %20, %21, !dbg !2055
  %add57 = add nsw i32 %sub56, 1, !dbg !2056
  %22 = sext i32 %add57 to i64, !dbg !2057
  %cmp58 = icmp slt i64 %indvars.iv11, %22, !dbg !2057
  br i1 %cmp58, label %for.body59, label %for.end68, !dbg !2058

for.body59:                                       ; preds = %for.cond55
  %arrayidx61 = getelementptr inbounds double, double* %p, i64 %indvars.iv11, !dbg !2059
  %23 = load double, double* %arrayidx61, align 8, !dbg !2059
  %arrayidx63 = getelementptr inbounds double, double* %q, i64 %indvars.iv11, !dbg !2061
  %24 = load double, double* %arrayidx63, align 8, !dbg !2061
  %mul64 = fmul contract double %23, %24, !dbg !2062
  %add65 = fadd contract double %d.0, %mul64, !dbg !2063
  call void @llvm.dbg.value(metadata double %add65, metadata !2047, metadata !DIExpression()), !dbg !1942
  br label %for.inc66, !dbg !2064

for.inc66:                                        ; preds = %for.body59
  %indvars.iv.next12 = add nuw nsw i64 %indvars.iv11, 1, !dbg !2065
  call void @llvm.dbg.value(metadata i32 undef, metadata !1953, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !1942
  br label %for.cond55, !dbg !2066, !llvm.loop !2067

for.end68:                                        ; preds = %for.cond55
  %d.0.lcssa = phi double [ %d.0, %for.cond55 ], !dbg !2051
  call void @llvm.dbg.value(metadata double %d.0.lcssa, metadata !2047, metadata !DIExpression()), !dbg !1942
  %div = fdiv double %rho.1, %d.0.lcssa, !dbg !2069
  call void @llvm.dbg.value(metadata double %div, metadata !2070, metadata !DIExpression()), !dbg !1942
  call void @llvm.dbg.value(metadata double %rho.1, metadata !2071, metadata !DIExpression()), !dbg !1942
  call void @llvm.dbg.value(metadata double 0.000000e+00, metadata !1952, metadata !DIExpression()), !dbg !1942
  call void @llvm.dbg.value(metadata i32 0, metadata !1953, metadata !DIExpression()), !dbg !1942
  br label %for.cond69, !dbg !2072

for.cond69:                                       ; preds = %for.inc90, %for.end68
  %indvars.iv13 = phi i64 [ %indvars.iv.next14, %for.inc90 ], [ 0, %for.end68 ], !dbg !2074
  call void @llvm.dbg.value(metadata i64 %indvars.iv13, metadata !1953, metadata !DIExpression()), !dbg !1942
  %25 = load i32, i32* @_ZL7lastcol, align 4, !dbg !2075
  %26 = load i32, i32* @_ZL8firstcol, align 4, !dbg !2077
  %sub70 = sub nsw i32 %25, %26, !dbg !2078
  %add71 = add nsw i32 %sub70, 1, !dbg !2079
  %27 = sext i32 %add71 to i64, !dbg !2080
  %cmp72 = icmp slt i64 %indvars.iv13, %27, !dbg !2080
  br i1 %cmp72, label %for.body73, label %for.end92, !dbg !2081

for.body73:                                       ; preds = %for.cond69
  %arrayidx75 = getelementptr inbounds double, double* %z, i64 %indvars.iv13, !dbg !2082
  %28 = load double, double* %arrayidx75, align 8, !dbg !2082
  %arrayidx77 = getelementptr inbounds double, double* %p, i64 %indvars.iv13, !dbg !2084
  %29 = load double, double* %arrayidx77, align 8, !dbg !2084
  %mul78 = fmul contract double %div, %29, !dbg !2085
  %add79 = fadd contract double %28, %mul78, !dbg !2086
  %arrayidx81 = getelementptr inbounds double, double* %z, i64 %indvars.iv13, !dbg !2087
  store double %add79, double* %arrayidx81, align 8, !dbg !2088
  %arrayidx83 = getelementptr inbounds double, double* %r, i64 %indvars.iv13, !dbg !2089
  %30 = load double, double* %arrayidx83, align 8, !dbg !2089
  %arrayidx85 = getelementptr inbounds double, double* %q, i64 %indvars.iv13, !dbg !2090
  %31 = load double, double* %arrayidx85, align 8, !dbg !2090
  %mul86 = fmul contract double %div, %31, !dbg !2091
  %sub87 = fsub contract double %30, %mul86, !dbg !2092
  %arrayidx89 = getelementptr inbounds double, double* %r, i64 %indvars.iv13, !dbg !2093
  store double %sub87, double* %arrayidx89, align 8, !dbg !2094
  br label %for.inc90, !dbg !2095

for.inc90:                                        ; preds = %for.body73
  %indvars.iv.next14 = add nuw nsw i64 %indvars.iv13, 1, !dbg !2096
  call void @llvm.dbg.value(metadata i32 undef, metadata !1953, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !1942
  br label %for.cond69, !dbg !2097, !llvm.loop !2098

for.end92:                                        ; preds = %for.cond69
  call void @llvm.dbg.value(metadata i32 0, metadata !1953, metadata !DIExpression()), !dbg !1942
  br label %for.cond93, !dbg !2100

for.cond93:                                       ; preds = %for.inc104, %for.end92
  %indvars.iv15 = phi i64 [ %indvars.iv.next16, %for.inc104 ], [ 0, %for.end92 ], !dbg !2102
  %rho.2 = phi double [ 0.000000e+00, %for.end92 ], [ %add103, %for.inc104 ], !dbg !2051
  call void @llvm.dbg.value(metadata double %rho.2, metadata !1952, metadata !DIExpression()), !dbg !1942
  call void @llvm.dbg.value(metadata i64 %indvars.iv15, metadata !1953, metadata !DIExpression()), !dbg !1942
  %32 = load i32, i32* @_ZL7lastcol, align 4, !dbg !2103
  %33 = load i32, i32* @_ZL8firstcol, align 4, !dbg !2105
  %sub94 = sub nsw i32 %32, %33, !dbg !2106
  %add95 = add nsw i32 %sub94, 1, !dbg !2107
  %34 = sext i32 %add95 to i64, !dbg !2108
  %cmp96 = icmp slt i64 %indvars.iv15, %34, !dbg !2108
  br i1 %cmp96, label %for.body97, label %for.end106, !dbg !2109

for.body97:                                       ; preds = %for.cond93
  %arrayidx99 = getelementptr inbounds double, double* %r, i64 %indvars.iv15, !dbg !2110
  %35 = load double, double* %arrayidx99, align 8, !dbg !2110
  %arrayidx101 = getelementptr inbounds double, double* %r, i64 %indvars.iv15, !dbg !2112
  %36 = load double, double* %arrayidx101, align 8, !dbg !2112
  %mul102 = fmul contract double %35, %36, !dbg !2113
  %add103 = fadd contract double %rho.2, %mul102, !dbg !2114
  call void @llvm.dbg.value(metadata double %add103, metadata !1952, metadata !DIExpression()), !dbg !1942
  br label %for.inc104, !dbg !2115

for.inc104:                                       ; preds = %for.body97
  %indvars.iv.next16 = add nuw nsw i64 %indvars.iv15, 1, !dbg !2116
  call void @llvm.dbg.value(metadata i32 undef, metadata !1953, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !1942
  br label %for.cond93, !dbg !2117, !llvm.loop !2118

for.end106:                                       ; preds = %for.cond93
  %rho.2.lcssa = phi double [ %rho.2, %for.cond93 ], !dbg !2051
  call void @llvm.dbg.value(metadata double %rho.2.lcssa, metadata !1952, metadata !DIExpression()), !dbg !1942
  %div107 = fdiv double %rho.2.lcssa, %rho.1, !dbg !2120
  call void @llvm.dbg.value(metadata double %div107, metadata !2121, metadata !DIExpression()), !dbg !1942
  call void @llvm.dbg.value(metadata i32 0, metadata !1953, metadata !DIExpression()), !dbg !1942
  br label %for.cond108, !dbg !2122

for.cond108:                                      ; preds = %for.inc121, %for.end106
  %indvars.iv17 = phi i64 [ %indvars.iv.next18, %for.inc121 ], [ 0, %for.end106 ], !dbg !2124
  call void @llvm.dbg.value(metadata i64 %indvars.iv17, metadata !1953, metadata !DIExpression()), !dbg !1942
  %37 = load i32, i32* @_ZL7lastcol, align 4, !dbg !2125
  %38 = load i32, i32* @_ZL8firstcol, align 4, !dbg !2127
  %sub109 = sub nsw i32 %37, %38, !dbg !2128
  %add110 = add nsw i32 %sub109, 1, !dbg !2129
  %39 = sext i32 %add110 to i64, !dbg !2130
  %cmp111 = icmp slt i64 %indvars.iv17, %39, !dbg !2130
  br i1 %cmp111, label %for.body112, label %for.end123, !dbg !2131

for.body112:                                      ; preds = %for.cond108
  %arrayidx114 = getelementptr inbounds double, double* %r, i64 %indvars.iv17, !dbg !2132
  %40 = load double, double* %arrayidx114, align 8, !dbg !2132
  %arrayidx116 = getelementptr inbounds double, double* %p, i64 %indvars.iv17, !dbg !2134
  %41 = load double, double* %arrayidx116, align 8, !dbg !2134
  %mul117 = fmul contract double %div107, %41, !dbg !2135
  %add118 = fadd contract double %40, %mul117, !dbg !2136
  %arrayidx120 = getelementptr inbounds double, double* %p, i64 %indvars.iv17, !dbg !2137
  store double %add118, double* %arrayidx120, align 8, !dbg !2138
  br label %for.inc121, !dbg !2139

for.inc121:                                       ; preds = %for.body112
  %indvars.iv.next18 = add nuw nsw i64 %indvars.iv17, 1, !dbg !2140
  call void @llvm.dbg.value(metadata i32 undef, metadata !1953, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !1942
  br label %for.cond108, !dbg !2141, !llvm.loop !2142

for.end123:                                       ; preds = %for.cond108
  br label %for.inc124, !dbg !2144

for.inc124:                                       ; preds = %for.end123
  %inc125 = add nuw nsw i32 %cgit.0, 1, !dbg !2145
  call void @llvm.dbg.value(metadata i32 %inc125, metadata !1998, metadata !DIExpression()), !dbg !1942
  br label %for.cond23, !dbg !2146, !llvm.loop !2147

for.end126:                                       ; preds = %for.cond23
  call void @llvm.dbg.value(metadata double 0.000000e+00, metadata !2016, metadata !DIExpression()), !dbg !1942
  call void @llvm.dbg.value(metadata i32 0, metadata !1953, metadata !DIExpression()), !dbg !1942
  br label %for.cond127, !dbg !2149

for.cond127:                                      ; preds = %for.inc153, %for.end126
  %indvars.iv3 = phi i64 [ %indvars.iv.next4, %for.inc153 ], [ 0, %for.end126 ], !dbg !2151
  call void @llvm.dbg.value(metadata i64 %indvars.iv3, metadata !1953, metadata !DIExpression()), !dbg !1942
  %42 = load i32, i32* @_ZL7lastrow, align 4, !dbg !2152
  %43 = load i32, i32* @_ZL8firstrow, align 4, !dbg !2154
  %sub128 = sub nsw i32 %42, %43, !dbg !2155
  %add129 = add nsw i32 %sub128, 1, !dbg !2156
  %44 = sext i32 %add129 to i64, !dbg !2157
  %cmp130 = icmp slt i64 %indvars.iv3, %44, !dbg !2157
  br i1 %cmp130, label %for.body131, label %for.end155, !dbg !2158

for.body131:                                      ; preds = %for.cond127
  call void @llvm.dbg.value(metadata double 0.000000e+00, metadata !2047, metadata !DIExpression()), !dbg !1942
  %arrayidx133 = getelementptr inbounds i32, i32* %rowstr, i64 %indvars.iv3, !dbg !2159
  %45 = load i32, i32* %arrayidx133, align 4, !dbg !2159
  call void @llvm.dbg.value(metadata i32 %45, metadata !2020, metadata !DIExpression()), !dbg !1942
  %46 = sext i32 %45 to i64, !dbg !2162
  br label %for.cond134, !dbg !2162

for.cond134:                                      ; preds = %for.inc148, %for.body131
  %indvars.iv1 = phi i64 [ %indvars.iv.next2, %for.inc148 ], [ %46, %for.body131 ], !dbg !2163
  %d.1 = phi double [ 0.000000e+00, %for.body131 ], [ %add147, %for.inc148 ], !dbg !2164
  call void @llvm.dbg.value(metadata double %d.1, metadata !2047, metadata !DIExpression()), !dbg !1942
  call void @llvm.dbg.value(metadata i64 %indvars.iv1, metadata !2020, metadata !DIExpression()), !dbg !1942
  %47 = add nuw nsw i64 %indvars.iv3, 1, !dbg !2165
  %arrayidx137 = getelementptr inbounds i32, i32* %rowstr, i64 %47, !dbg !2167
  %48 = load i32, i32* %arrayidx137, align 4, !dbg !2167
  %49 = sext i32 %48 to i64, !dbg !2168
  %cmp138 = icmp slt i64 %indvars.iv1, %49, !dbg !2168
  br i1 %cmp138, label %for.body139, label %for.end150, !dbg !2169

for.body139:                                      ; preds = %for.cond134
  %arrayidx141 = getelementptr inbounds double, double* %a, i64 %indvars.iv1, !dbg !2170
  %50 = load double, double* %arrayidx141, align 8, !dbg !2170
  %arrayidx143 = getelementptr inbounds i32, i32* %colidx, i64 %indvars.iv1, !dbg !2172
  %51 = load i32, i32* %arrayidx143, align 4, !dbg !2172
  %idxprom144 = sext i32 %51 to i64, !dbg !2173
  %arrayidx145 = getelementptr inbounds double, double* %z, i64 %idxprom144, !dbg !2173
  %52 = load double, double* %arrayidx145, align 8, !dbg !2173
  %mul146 = fmul contract double %50, %52, !dbg !2174
  %add147 = fadd contract double %d.1, %mul146, !dbg !2175
  call void @llvm.dbg.value(metadata double %add147, metadata !2047, metadata !DIExpression()), !dbg !1942
  br label %for.inc148, !dbg !2176

for.inc148:                                       ; preds = %for.body139
  %indvars.iv.next2 = add nsw i64 %indvars.iv1, 1, !dbg !2177
  call void @llvm.dbg.value(metadata i32 undef, metadata !2020, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !1942
  br label %for.cond134, !dbg !2178, !llvm.loop !2179

for.end150:                                       ; preds = %for.cond134
  %d.1.lcssa = phi double [ %d.1, %for.cond134 ], !dbg !2164
  call void @llvm.dbg.value(metadata double %d.1.lcssa, metadata !2047, metadata !DIExpression()), !dbg !1942
  %arrayidx152 = getelementptr inbounds double, double* %r, i64 %indvars.iv3, !dbg !2181
  store double %d.1.lcssa, double* %arrayidx152, align 8, !dbg !2182
  br label %for.inc153, !dbg !2183

for.inc153:                                       ; preds = %for.end150
  %indvars.iv.next4 = add nuw nsw i64 %indvars.iv3, 1, !dbg !2184
  call void @llvm.dbg.value(metadata i32 undef, metadata !1953, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !1942
  br label %for.cond127, !dbg !2185, !llvm.loop !2186

for.end155:                                       ; preds = %for.cond127
  call void @llvm.dbg.value(metadata i32 0, metadata !1953, metadata !DIExpression()), !dbg !1942
  br label %for.cond156, !dbg !2188

for.cond156:                                      ; preds = %for.inc168, %for.end155
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc168 ], [ 0, %for.end155 ], !dbg !2190
  %sum.1 = phi double [ 0.000000e+00, %for.end155 ], [ %add167, %for.inc168 ], !dbg !1942
  call void @llvm.dbg.value(metadata double %sum.1, metadata !2016, metadata !DIExpression()), !dbg !1942
  call void @llvm.dbg.value(metadata i64 %indvars.iv, metadata !1953, metadata !DIExpression()), !dbg !1942
  %53 = load i32, i32* @_ZL7lastcol, align 4, !dbg !2191
  %54 = load i32, i32* @_ZL8firstcol, align 4, !dbg !2193
  %sub157 = sub nsw i32 %53, %54, !dbg !2194
  %add158 = add nsw i32 %sub157, 1, !dbg !2195
  %55 = sext i32 %add158 to i64, !dbg !2196
  %cmp159 = icmp slt i64 %indvars.iv, %55, !dbg !2196
  br i1 %cmp159, label %for.body160, label %for.end170, !dbg !2197

for.body160:                                      ; preds = %for.cond156
  %arrayidx162 = getelementptr inbounds double, double* %x, i64 %indvars.iv, !dbg !2198
  %56 = load double, double* %arrayidx162, align 8, !dbg !2198
  %arrayidx164 = getelementptr inbounds double, double* %r, i64 %indvars.iv, !dbg !2200
  %57 = load double, double* %arrayidx164, align 8, !dbg !2200
  %sub165 = fsub contract double %56, %57, !dbg !2201
  call void @llvm.dbg.value(metadata double %sub165, metadata !2047, metadata !DIExpression()), !dbg !1942
  %mul166 = fmul contract double %sub165, %sub165, !dbg !2202
  %add167 = fadd contract double %sum.1, %mul166, !dbg !2203
  call void @llvm.dbg.value(metadata double %add167, metadata !2016, metadata !DIExpression()), !dbg !1942
  br label %for.inc168, !dbg !2204

for.inc168:                                       ; preds = %for.body160
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !2205
  call void @llvm.dbg.value(metadata i32 undef, metadata !1953, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !1942
  br label %for.cond156, !dbg !2206, !llvm.loop !2207

for.end170:                                       ; preds = %for.cond156
  %sum.1.lcssa = phi double [ %sum.1, %for.cond156 ], !dbg !1942
  call void @llvm.dbg.value(metadata double %sum.1.lcssa, metadata !2016, metadata !DIExpression()), !dbg !1942
  %call = call double @sqrt(double %sum.1.lcssa) #11, !dbg !2209
  store double %call, double* %rnorm, align 8, !dbg !2210
  ret void, !dbg !2211
}

; Function Attrs: nounwind
declare dso_local double @sqrt(double) #6

; Function Attrs: noinline uwtable
define internal void @_ZL9setup_gpuv() #4 !dbg !2212 {
entry:
  %0 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !2213
  %cmp = icmp sle i32 1024, %0, !dbg !2215
  br i1 %cmp, label %if.then, label %if.else, !dbg !2216

if.then:                                          ; preds = %entry
  store i32 1024, i32* @threads_per_block_on_kernel_one, align 4, !dbg !2217
  br label %if.end, !dbg !2219

if.else:                                          ; preds = %entry
  %1 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !2220
  store i32 %1, i32* @threads_per_block_on_kernel_one, align 4, !dbg !2222
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %2 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !2223
  %cmp1 = icmp sle i32 256, %2, !dbg !2225
  br i1 %cmp1, label %if.then2, label %if.else3, !dbg !2226

if.then2:                                         ; preds = %if.end
  store i32 256, i32* @threads_per_block_on_kernel_two, align 4, !dbg !2227
  br label %if.end4, !dbg !2229

if.else3:                                         ; preds = %if.end
  %3 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !2230
  store i32 %3, i32* @threads_per_block_on_kernel_two, align 4, !dbg !2232
  br label %if.end4

if.end4:                                          ; preds = %if.else3, %if.then2
  %4 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !2233
  %cmp5 = icmp sle i32 64, %4, !dbg !2235
  br i1 %cmp5, label %if.then6, label %if.else7, !dbg !2236

if.then6:                                         ; preds = %if.end4
  store i32 64, i32* @threads_per_block_on_kernel_three, align 4, !dbg !2237
  br label %if.end8, !dbg !2239

if.else7:                                         ; preds = %if.end4
  %5 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !2240
  store i32 %5, i32* @threads_per_block_on_kernel_three, align 4, !dbg !2242
  br label %if.end8

if.end8:                                          ; preds = %if.else7, %if.then6
  %6 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !2243
  %cmp9 = icmp sle i32 256, %6, !dbg !2245
  br i1 %cmp9, label %if.then10, label %if.else11, !dbg !2246

if.then10:                                        ; preds = %if.end8
  store i32 256, i32* @threads_per_block_on_kernel_four, align 4, !dbg !2247
  br label %if.end12, !dbg !2249

if.else11:                                        ; preds = %if.end8
  %7 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !2250
  store i32 %7, i32* @threads_per_block_on_kernel_four, align 4, !dbg !2252
  br label %if.end12

if.end12:                                         ; preds = %if.else11, %if.then10
  %8 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !2253
  %cmp13 = icmp sle i32 64, %8, !dbg !2255
  br i1 %cmp13, label %if.then14, label %if.else15, !dbg !2256

if.then14:                                        ; preds = %if.end12
  store i32 64, i32* @threads_per_block_on_kernel_five, align 4, !dbg !2257
  br label %if.end16, !dbg !2259

if.else15:                                        ; preds = %if.end12
  %9 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !2260
  store i32 %9, i32* @threads_per_block_on_kernel_five, align 4, !dbg !2262
  br label %if.end16

if.end16:                                         ; preds = %if.else15, %if.then14
  %10 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !2263
  %cmp17 = icmp sle i32 256, %10, !dbg !2265
  br i1 %cmp17, label %if.then18, label %if.else19, !dbg !2266

if.then18:                                        ; preds = %if.end16
  store i32 256, i32* @threads_per_block_on_kernel_six, align 4, !dbg !2267
  br label %if.end20, !dbg !2269

if.else19:                                        ; preds = %if.end16
  %11 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !2270
  store i32 %11, i32* @threads_per_block_on_kernel_six, align 4, !dbg !2272
  br label %if.end20

if.end20:                                         ; preds = %if.else19, %if.then18
  %12 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !2273
  %cmp21 = icmp sle i32 512, %12, !dbg !2275
  br i1 %cmp21, label %if.then22, label %if.else23, !dbg !2276

if.then22:                                        ; preds = %if.end20
  store i32 512, i32* @threads_per_block_on_kernel_seven, align 4, !dbg !2277
  br label %if.end24, !dbg !2279

if.else23:                                        ; preds = %if.end20
  %13 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !2280
  store i32 %13, i32* @threads_per_block_on_kernel_seven, align 4, !dbg !2282
  br label %if.end24

if.end24:                                         ; preds = %if.else23, %if.then22
  %14 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !2283
  %cmp25 = icmp sle i32 64, %14, !dbg !2285
  br i1 %cmp25, label %if.then26, label %if.else27, !dbg !2286

if.then26:                                        ; preds = %if.end24
  store i32 64, i32* @threads_per_block_on_kernel_eight, align 4, !dbg !2287
  br label %if.end28, !dbg !2289

if.else27:                                        ; preds = %if.end24
  %15 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !2290
  store i32 %15, i32* @threads_per_block_on_kernel_eight, align 4, !dbg !2292
  br label %if.end28

if.end28:                                         ; preds = %if.else27, %if.then26
  %16 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !2293
  %cmp29 = icmp sle i32 512, %16, !dbg !2295
  br i1 %cmp29, label %if.then30, label %if.else31, !dbg !2296

if.then30:                                        ; preds = %if.end28
  store i32 512, i32* @threads_per_block_on_kernel_nine, align 4, !dbg !2297
  br label %if.end32, !dbg !2299

if.else31:                                        ; preds = %if.end28
  %17 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !2300
  store i32 %17, i32* @threads_per_block_on_kernel_nine, align 4, !dbg !2302
  br label %if.end32

if.end32:                                         ; preds = %if.else31, %if.then30
  %18 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !2303
  %cmp33 = icmp sle i32 256, %18, !dbg !2305
  br i1 %cmp33, label %if.then34, label %if.else35, !dbg !2306

if.then34:                                        ; preds = %if.end32
  store i32 256, i32* @threads_per_block_on_kernel_ten, align 4, !dbg !2307
  br label %if.end36, !dbg !2309

if.else35:                                        ; preds = %if.end32
  %19 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !2310
  store i32 %19, i32* @threads_per_block_on_kernel_ten, align 4, !dbg !2312
  br label %if.end36

if.end36:                                         ; preds = %if.else35, %if.then34
  %20 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !2313
  %cmp37 = icmp sle i32 512, %20, !dbg !2315
  br i1 %cmp37, label %if.then38, label %if.else39, !dbg !2316

if.then38:                                        ; preds = %if.end36
  store i32 512, i32* @threads_per_block_on_kernel_eleven, align 4, !dbg !2317
  br label %if.end40, !dbg !2319

if.else39:                                        ; preds = %if.end36
  %21 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !2320
  store i32 %21, i32* @threads_per_block_on_kernel_eleven, align 4, !dbg !2322
  br label %if.end40

if.end40:                                         ; preds = %if.else39, %if.then38
  %22 = load i32, i32* @threads_per_block_on_kernel_one, align 4, !dbg !2323
  %conv = sitofp i32 %22 to double, !dbg !2323
  %div = fdiv double 1.400000e+04, %conv, !dbg !2324
  %23 = call double @llvm.ceil.f64(double %div), !dbg !2325
  %conv41 = fptosi double %23 to i32, !dbg !2326
  store i32 %conv41, i32* @blocks_per_grid_on_kernel_one, align 4, !dbg !2327
  %24 = load i32, i32* @threads_per_block_on_kernel_two, align 4, !dbg !2328
  %conv42 = sitofp i32 %24 to double, !dbg !2328
  %div43 = fdiv double 1.400000e+04, %conv42, !dbg !2329
  %25 = call double @llvm.ceil.f64(double %div43), !dbg !2330
  %conv44 = fptosi double %25 to i32, !dbg !2331
  store i32 %conv44, i32* @blocks_per_grid_on_kernel_two, align 4, !dbg !2332
  store i32 14000, i32* @blocks_per_grid_on_kernel_three, align 4, !dbg !2333
  %26 = load i32, i32* @threads_per_block_on_kernel_four, align 4, !dbg !2334
  %conv45 = sitofp i32 %26 to double, !dbg !2334
  %div46 = fdiv double 1.400000e+04, %conv45, !dbg !2335
  %27 = call double @llvm.ceil.f64(double %div46), !dbg !2336
  %conv47 = fptosi double %27 to i32, !dbg !2337
  store i32 %conv47, i32* @blocks_per_grid_on_kernel_four, align 4, !dbg !2338
  %28 = load i32, i32* @threads_per_block_on_kernel_five, align 4, !dbg !2339
  %conv48 = sitofp i32 %28 to double, !dbg !2339
  %div49 = fdiv double 1.400000e+04, %conv48, !dbg !2340
  %29 = call double @llvm.ceil.f64(double %div49), !dbg !2341
  %conv50 = fptosi double %29 to i32, !dbg !2342
  store i32 %conv50, i32* @blocks_per_grid_on_kernel_five, align 4, !dbg !2343
  %30 = load i32, i32* @threads_per_block_on_kernel_six, align 4, !dbg !2344
  %conv51 = sitofp i32 %30 to double, !dbg !2344
  %div52 = fdiv double 1.400000e+04, %conv51, !dbg !2345
  %31 = call double @llvm.ceil.f64(double %div52), !dbg !2346
  %conv53 = fptosi double %31 to i32, !dbg !2347
  store i32 %conv53, i32* @blocks_per_grid_on_kernel_six, align 4, !dbg !2348
  %32 = load i32, i32* @threads_per_block_on_kernel_seven, align 4, !dbg !2349
  %conv54 = sitofp i32 %32 to double, !dbg !2349
  %div55 = fdiv double 1.400000e+04, %conv54, !dbg !2350
  %33 = call double @llvm.ceil.f64(double %div55), !dbg !2351
  %conv56 = fptosi double %33 to i32, !dbg !2352
  store i32 %conv56, i32* @blocks_per_grid_on_kernel_seven, align 4, !dbg !2353
  store i32 14000, i32* @blocks_per_grid_on_kernel_eight, align 4, !dbg !2354
  %34 = load i32, i32* @threads_per_block_on_kernel_nine, align 4, !dbg !2355
  %conv57 = sitofp i32 %34 to double, !dbg !2355
  %div58 = fdiv double 1.400000e+04, %conv57, !dbg !2356
  %35 = call double @llvm.ceil.f64(double %div58), !dbg !2357
  %conv59 = fptosi double %35 to i32, !dbg !2358
  store i32 %conv59, i32* @blocks_per_grid_on_kernel_nine, align 4, !dbg !2359
  %36 = load i32, i32* @threads_per_block_on_kernel_ten, align 4, !dbg !2360
  %conv60 = sitofp i32 %36 to double, !dbg !2360
  %div61 = fdiv double 1.400000e+04, %conv60, !dbg !2361
  %37 = call double @llvm.ceil.f64(double %div61), !dbg !2362
  %conv62 = fptosi double %37 to i32, !dbg !2363
  store i32 %conv62, i32* @blocks_per_grid_on_kernel_ten, align 4, !dbg !2364
  %38 = load i32, i32* @threads_per_block_on_kernel_eleven, align 4, !dbg !2365
  %conv63 = sitofp i32 %38 to double, !dbg !2365
  %div64 = fdiv double 1.400000e+04, %conv63, !dbg !2366
  %39 = call double @llvm.ceil.f64(double %div64), !dbg !2367
  %conv65 = fptosi double %39 to i32, !dbg !2368
  store i32 %conv65, i32* @blocks_per_grid_on_kernel_eleven, align 4, !dbg !2369
  %40 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !2370
  %conv66 = sitofp i32 %40 to double, !dbg !2371
  %div67 = fdiv double 1.400000e+04, %conv66, !dbg !2372
  %41 = call double @llvm.ceil.f64(double %div67), !dbg !2373
  %conv68 = fptoui double %41 to i64, !dbg !2373
  store i64 %conv68, i64* @global_data_elements, align 8, !dbg !2374
  %42 = load i64, i64* @global_data_elements, align 8, !dbg !2375
  %mul = mul i64 %42, 8, !dbg !2376
  store i64 %mul, i64* @size_global_data, align 8, !dbg !2377
  store i64 8064000, i64* @size_colidx_device, align 8, !dbg !2378
  store i64 56004, i64* @size_rowstr_device, align 8, !dbg !2379
  store i64 56000, i64* @size_iv_device, align 8, !dbg !2380
  store i64 56000, i64* @size_arow_device, align 8, !dbg !2381
  store i64 672000, i64* @size_acol_device, align 8, !dbg !2382
  store i64 1344000, i64* @size_aelt_device, align 8, !dbg !2383
  store i64 16128000, i64* @size_a_device, align 8, !dbg !2384
  store i64 112016, i64* @size_x_device, align 8, !dbg !2385
  store i64 112016, i64* @size_z_device, align 8, !dbg !2386
  store i64 112016, i64* @size_p_device, align 8, !dbg !2387
  store i64 112016, i64* @size_q_device, align 8, !dbg !2388
  store i64 112016, i64* @size_r_device, align 8, !dbg !2389
  store i64 8, i64* @size_rho_device, align 8, !dbg !2390
  store i64 8, i64* @size_d_device, align 8, !dbg !2391
  store i64 8, i64* @size_alpha_device, align 8, !dbg !2392
  store i64 8, i64* @size_beta_device, align 8, !dbg !2393
  store i64 8, i64* @size_sum_device, align 8, !dbg !2394
  store i64 8, i64* @size_norm_temp1_device, align 8, !dbg !2395
  store i64 8, i64* @size_norm_temp2_device, align 8, !dbg !2396
  %43 = load i64, i64* @size_global_data, align 8, !dbg !2397, !tulip.target.datasize !2398
  %call = call noalias i8* @malloc(i64 %43) #11, !dbg !2399, !tulip.target.mapdata.from !2400
  %44 = bitcast i8* %call to double*, !dbg !2401
  store double* %44, double** @global_data, align 8, !dbg !2402
  %45 = load i64, i64* @size_global_data, align 8, !dbg !2403, !tulip.target.datasize !2404
  %call69 = call noalias i8* @malloc(i64 %45) #11, !dbg !2405, !tulip.target.mapdata.from !2406
  %46 = bitcast i8* %call69 to double*, !dbg !2407
  store double* %46, double** @global_data_two, align 8, !dbg !2408
  %47 = load i32*, i32** @_ZL6colidx, align 8, !dbg !2409
  %48 = bitcast i32* %47 to i8*, !dbg !2409
  %49 = load i32*, i32** @_ZL6colidx, align 8, !dbg !2410
  %50 = bitcast i32* %49 to i8*, !dbg !2410
  %51 = load i64, i64* @size_colidx_device, align 8, !dbg !2411
  %call87 = call i32 @cudaMemcpy(i8* %48, i8* %50, i64 %51, i32 1), !dbg !2412, !tulip.target.start.of.map !2413
  %52 = load i32, i32* @threads_per_block_on_kernel_one, align 4, !dbg !2414
  %conv95 = sext i32 %52 to i64, !dbg !2414
  %mul96 = mul i64 %conv95, 8, !dbg !2415
  store i64 %mul96, i64* @size_shared_data_on_kernel_one, align 8, !dbg !2416
  %53 = load i32, i32* @threads_per_block_on_kernel_two, align 4, !dbg !2417
  %conv97 = sext i32 %53 to i64, !dbg !2417
  %mul98 = mul i64 %conv97, 8, !dbg !2418
  store i64 %mul98, i64* @size_shared_data_on_kernel_two, align 8, !dbg !2419
  %54 = load i32, i32* @threads_per_block_on_kernel_three, align 4, !dbg !2420
  %conv99 = sext i32 %54 to i64, !dbg !2420
  %mul100 = mul i64 %conv99, 8, !dbg !2421
  store i64 %mul100, i64* @size_shared_data_on_kernel_three, align 8, !dbg !2422
  %55 = load i32, i32* @threads_per_block_on_kernel_four, align 4, !dbg !2423
  %conv101 = sext i32 %55 to i64, !dbg !2423
  %mul102 = mul i64 %conv101, 8, !dbg !2424
  store i64 %mul102, i64* @size_shared_data_on_kernel_four, align 8, !dbg !2425
  %56 = load i32, i32* @threads_per_block_on_kernel_five, align 4, !dbg !2426
  %conv103 = sext i32 %56 to i64, !dbg !2426
  %mul104 = mul i64 %conv103, 8, !dbg !2427
  store i64 %mul104, i64* @size_shared_data_on_kernel_five, align 8, !dbg !2428
  %57 = load i32, i32* @threads_per_block_on_kernel_six, align 4, !dbg !2429
  %conv105 = sext i32 %57 to i64, !dbg !2429
  %mul106 = mul i64 %conv105, 8, !dbg !2430
  store i64 %mul106, i64* @size_shared_data_on_kernel_six, align 8, !dbg !2431
  %58 = load i32, i32* @threads_per_block_on_kernel_seven, align 4, !dbg !2432
  %conv107 = sext i32 %58 to i64, !dbg !2432
  %mul108 = mul i64 %conv107, 8, !dbg !2433
  store i64 %mul108, i64* @size_shared_data_on_kernel_seven, align 8, !dbg !2434
  %59 = load i32, i32* @threads_per_block_on_kernel_eight, align 4, !dbg !2435
  %conv109 = sext i32 %59 to i64, !dbg !2435
  %mul110 = mul i64 %conv109, 8, !dbg !2436
  store i64 %mul110, i64* @size_shared_data_on_kernel_eight, align 8, !dbg !2437
  %60 = load i32, i32* @threads_per_block_on_kernel_nine, align 4, !dbg !2438
  %conv111 = sext i32 %60 to i64, !dbg !2438
  %mul112 = mul i64 %conv111, 8, !dbg !2439
  store i64 %mul112, i64* @size_shared_data_on_kernel_nine, align 8, !dbg !2440
  %61 = load i32, i32* @threads_per_block_on_kernel_ten, align 4, !dbg !2441
  %conv113 = sext i32 %61 to i64, !dbg !2441
  %mul114 = mul i64 %conv113, 8, !dbg !2442
  store i64 %mul114, i64* @size_shared_data_on_kernel_ten, align 8, !dbg !2443
  %62 = load i32, i32* @threads_per_block_on_kernel_eleven, align 4, !dbg !2444
  %conv115 = sext i32 %62 to i64, !dbg !2444
  %mul116 = mul i64 %conv115, 8, !dbg !2445
  store i64 %mul116, i64* @size_shared_data_on_kernel_eleven, align 8, !dbg !2446
  %63 = load i32, i32* @blocks_per_grid_on_kernel_one, align 4, !dbg !2447
  %conv117 = sext i32 %63 to i64, !dbg !2447
  %mul118 = mul i64 %conv117, 8, !dbg !2448
  store i64 %mul118, i64* @size_reduce_memory_on_kernel_one, align 8, !dbg !2449
  %64 = load i32, i32* @blocks_per_grid_on_kernel_two, align 4, !dbg !2450
  %conv119 = sext i32 %64 to i64, !dbg !2450
  %mul120 = mul i64 %conv119, 8, !dbg !2451
  store i64 %mul120, i64* @size_reduce_memory_on_kernel_two, align 8, !dbg !2452
  %65 = load i32, i32* @blocks_per_grid_on_kernel_three, align 4, !dbg !2453
  %conv121 = sext i32 %65 to i64, !dbg !2453
  %mul122 = mul i64 %conv121, 8, !dbg !2454
  store i64 %mul122, i64* @size_reduce_memory_on_kernel_three, align 8, !dbg !2455
  %66 = load i32, i32* @blocks_per_grid_on_kernel_four, align 4, !dbg !2456
  %conv123 = sext i32 %66 to i64, !dbg !2456
  %mul124 = mul i64 %conv123, 8, !dbg !2457
  store i64 %mul124, i64* @size_reduce_memory_on_kernel_four, align 8, !dbg !2458
  %67 = load i32, i32* @blocks_per_grid_on_kernel_five, align 4, !dbg !2459
  %conv125 = sext i32 %67 to i64, !dbg !2459
  %mul126 = mul i64 %conv125, 8, !dbg !2460
  store i64 %mul126, i64* @size_reduce_memory_on_kernel_five, align 8, !dbg !2461
  %68 = load i32, i32* @blocks_per_grid_on_kernel_six, align 4, !dbg !2462
  %conv127 = sext i32 %68 to i64, !dbg !2462
  %mul128 = mul i64 %conv127, 8, !dbg !2463
  store i64 %mul128, i64* @size_reduce_memory_on_kernel_six, align 8, !dbg !2464
  %69 = load i32, i32* @blocks_per_grid_on_kernel_seven, align 4, !dbg !2465
  %conv129 = sext i32 %69 to i64, !dbg !2465
  %mul130 = mul i64 %conv129, 8, !dbg !2466
  store i64 %mul130, i64* @size_reduce_memory_on_kernel_seven, align 8, !dbg !2467
  %70 = load i32, i32* @blocks_per_grid_on_kernel_eight, align 4, !dbg !2468
  %conv131 = sext i32 %70 to i64, !dbg !2468
  %mul132 = mul i64 %conv131, 8, !dbg !2469
  store i64 %mul132, i64* @size_reduce_memory_on_kernel_eight, align 8, !dbg !2470
  %71 = load i32, i32* @blocks_per_grid_on_kernel_nine, align 4, !dbg !2471
  %conv133 = sext i32 %71 to i64, !dbg !2471
  %mul134 = mul i64 %conv133, 8, !dbg !2472
  store i64 %mul134, i64* @size_reduce_memory_on_kernel_nine, align 8, !dbg !2473
  %72 = load i32, i32* @blocks_per_grid_on_kernel_ten, align 4, !dbg !2474
  %conv135 = sext i32 %72 to i64, !dbg !2474
  %mul136 = mul i64 %conv135, 8, !dbg !2475
  store i64 %mul136, i64* @size_reduce_memory_on_kernel_ten, align 8, !dbg !2476
  %73 = load i32, i32* @blocks_per_grid_on_kernel_eleven, align 4, !dbg !2477
  %conv137 = sext i32 %73 to i64, !dbg !2477
  %mul138 = mul i64 %conv137, 8, !dbg !2478
  store i64 %mul138, i64* @size_reduce_memory_on_kernel_eleven, align 8, !dbg !2479
  ret void, !dbg !2480
}

; Function Attrs: noinline uwtable
define internal void @_ZL13conj_grad_gpuPd(double* %rnorm) #4 !dbg !2481 {
entry:
  %d = alloca double, align 8
  %sum = alloca double, align 8
  %rho = alloca double, align 8
  call void @llvm.dbg.value(metadata double* %rnorm, metadata !2484, metadata !DIExpression()), !dbg !2485
  call void @llvm.dbg.declare(metadata double* %d, metadata !2486, metadata !DIExpression()), !dbg !2487
  call void @llvm.dbg.declare(metadata double* %sum, metadata !2488, metadata !DIExpression()), !dbg !2489
  call void @llvm.dbg.declare(metadata double* %rho, metadata !2490, metadata !DIExpression()), !dbg !2491
  call void @llvm.dbg.value(metadata i32 25, metadata !2492, metadata !DIExpression()), !dbg !2485
  call void @_ZL19gpu_kernel_one_hostv(), !dbg !2493
  call void @_ZL19gpu_kernel_two_hostPd(double* %rho), !dbg !2494
  call void @llvm.dbg.value(metadata i32 1, metadata !2495, metadata !DIExpression()), !dbg !2485
  br label %for.cond, !dbg !2496

for.cond:                                         ; preds = %for.inc, %entry
  %cgit.0 = phi i32 [ 1, %entry ], [ %inc, %for.inc ], !dbg !2498
  call void @llvm.dbg.value(metadata i32 %cgit.0, metadata !2495, metadata !DIExpression()), !dbg !2485
  %exitcond = icmp ne i32 %cgit.0, 26, !dbg !2499
  br i1 %exitcond, label %for.body, label %for.end, !dbg !2501

for.body:                                         ; preds = %for.cond
  call void @_ZL21gpu_kernel_three_hostv(), !dbg !2502
  call void @_ZL20gpu_kernel_four_hostPd(double* %d), !dbg !2504
  %0 = load double, double* %rho, align 8, !dbg !2505
  %1 = load double, double* %d, align 8, !dbg !2506
  %div = fdiv double %0, %1, !dbg !2507
  call void @llvm.dbg.value(metadata double %div, metadata !2508, metadata !DIExpression()), !dbg !2485
  %2 = load double, double* %rho, align 8, !dbg !2509
  call void @llvm.dbg.value(metadata double %2, metadata !2510, metadata !DIExpression()), !dbg !2485
  call void @_ZL20gpu_kernel_five_hostd(double %div), !dbg !2511
  call void @_ZL19gpu_kernel_six_hostPd(double* %rho), !dbg !2512
  %3 = load double, double* %rho, align 8, !dbg !2513
  %div1 = fdiv double %3, %2, !dbg !2514
  call void @llvm.dbg.value(metadata double %div1, metadata !2515, metadata !DIExpression()), !dbg !2485
  call void @_ZL21gpu_kernel_seven_hostd(double %div1), !dbg !2516
  br label %for.inc, !dbg !2517

for.inc:                                          ; preds = %for.body
  %inc = add nuw nsw i32 %cgit.0, 1, !dbg !2518
  call void @llvm.dbg.value(metadata i32 %inc, metadata !2495, metadata !DIExpression()), !dbg !2485
  br label %for.cond, !dbg !2519, !llvm.loop !2520

for.end:                                          ; preds = %for.cond
  call void @_ZL21gpu_kernel_eight_hostv(), !dbg !2522
  call void @_ZL20gpu_kernel_nine_hostPd(double* %sum), !dbg !2523
  %4 = load double, double* %sum, align 8, !dbg !2524
  %call = call double @sqrt(double %4) #11, !dbg !2525
  store double %call, double* %rnorm, align 8, !dbg !2526
  ret void, !dbg !2527
}

; Function Attrs: noinline uwtable
define internal void @_ZL19gpu_kernel_ten_hostPdS_(double* %norm_temp1, double* %norm_temp2) #4 !dbg !2528 {
entry:
  %agg.tmp = alloca %struct.dim3, align 4
  %agg.tmp1 = alloca %struct.dim3, align 4
  %agg.tmp.coerce = alloca { i64, i32 }, align 4
  %agg.tmp1.coerce = alloca { i64, i32 }, align 4
  %agg.tmp2 = alloca %struct.dim3, align 4
  %agg.tmp3 = alloca %struct.dim3, align 4
  %agg.tmp2.coerce = alloca { i64, i32 }, align 4
  %agg.tmp3.coerce = alloca { i64, i32 }, align 4
  call void @llvm.dbg.value(metadata double* %norm_temp1, metadata !2531, metadata !DIExpression()), !dbg !2532
  call void @llvm.dbg.value(metadata double* %norm_temp2, metadata !2533, metadata !DIExpression()), !dbg !2532
  %0 = load i32, i32* @blocks_per_grid_on_kernel_ten, align 4, !dbg !2534
  %dim3gep.0 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp, i32 0, i32 0
  store i32 %0, i32* %dim3gep.0
  %dim3gep.1 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp, i32 0, i32 1
  store i32 1, i32* %dim3gep.1
  %dim3gep.2 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp, i32 0, i32 2
  store i32 1, i32* %dim3gep.2
  %1 = load i32, i32* @threads_per_block_on_kernel_ten, align 4, !dbg !2535
  %dim3gep.01 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp1, i32 0, i32 0
  store i32 %1, i32* %dim3gep.01
  %dim3gep.12 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp1, i32 0, i32 1
  store i32 1, i32* %dim3gep.12
  %dim3gep.23 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp1, i32 0, i32 2
  store i32 1, i32* %dim3gep.23
  %2 = bitcast { i64, i32 }* %agg.tmp.coerce to i8*, !dbg !2536
  %3 = bitcast %struct.dim3* %agg.tmp to i8*, !dbg !2536
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %2, i8* align 4 %3, i64 12, i1 false), !dbg !2536
  %4 = bitcast { i64, i32 }* %agg.tmp1.coerce to i8*, !dbg !2536
  %5 = bitcast %struct.dim3* %agg.tmp1 to i8*, !dbg !2536
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %4, i8* align 4 %5, i64 12, i1 false), !dbg !2536
  br label %header.0

header.0:                                         ; preds = %latch.0, %entry
  %indvar.0 = phi i32 [ 0, %entry ], [ %indvar.next.0, %latch.0 ]
  %exitcond24 = icmp ne i32 %indvar.0, %0
  br i1 %exitcond24, label %header.1.preheader.clone0, label %kcall.end, !tulip.doall.loop.grid !2413

header.1.preheader.clone0:                        ; preds = %header.0
  br label %header.1.clone0

header.1.clone0:                                  ; preds = %latch.1.clone0, %header.1.preheader.clone0
  %indvar.1.clone0 = phi i32 [ %indvar.next.1.clone0, %latch.1.clone0 ], [ 0, %header.1.preheader.clone0 ]
  %exitcond22 = icmp ne i32 %indvar.1.clone0, %1
  br i1 %exitcond22, label %kcall.configok.clone0, label %header.1.preheader, !tulip.doall.loop.block !2413

kcall.configok.clone0:                            ; preds = %header.1.clone0
  %6 = load double*, double** @global_data, align 8, !dbg !2537
  %7 = load double*, double** @_ZL1x, align 8, !dbg !2538
  %8 = load double*, double** @_ZL1z, align 8, !dbg !2539
  call void @gpu_kernel_ten_10(double* %6, double* %7, double* %8, i32 %0, i32 1, i32 1, i32 %1, i32 1, i32 1, i32 %indvar.0, i32 0, i32 0, i32 %indvar.1.clone0, i32 0, i32 0)
  br label %latch.1.clone0

latch.1.clone0:                                   ; preds = %kcall.configok.clone0
  %indvar.next.1.clone0 = add i32 %indvar.1.clone0, 1
  br label %header.1.clone0

header.1.preheader:                               ; preds = %header.1.clone0
  br label %header.1

header.1:                                         ; preds = %header.1.preheader, %latch.1
  %indvar.1 = phi i32 [ %indvar.next.1, %latch.1 ], [ 0, %header.1.preheader ]
  %exitcond23 = icmp ne i32 %indvar.1, %1
  br i1 %exitcond23, label %kcall.configok, label %latch.0, !tulip.doall.loop.block !2413

latch.1:                                          ; preds = %kcall.configok
  %indvar.next.1 = add i32 %indvar.1, 1
  br label %header.1

latch.0:                                          ; preds = %header.1
  %indvar.next.0 = add i32 %indvar.0, 1
  br label %header.0

kcall.configok:                                   ; preds = %header.1
  %9 = load double*, double** @global_data, align 8, !dbg !2537
  %10 = load double*, double** @_ZL1x, align 8, !dbg !2538
  %11 = load double*, double** @_ZL1z, align 8, !dbg !2539
  call void @gpu_kernel_ten_11(double* %9, double* %10, double* %11, i32 %0, i32 1, i32 1, i32 %1, i32 1, i32 1, i32 %indvar.0, i32 0, i32 0, i32 %indvar.1, i32 0, i32 0)
  br label %latch.1

kcall.end:                                        ; preds = %header.0
  %12 = load i32, i32* @blocks_per_grid_on_kernel_ten, align 4, !dbg !2540
  %dim3gep.04 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp2, i32 0, i32 0
  store i32 %12, i32* %dim3gep.04
  %dim3gep.15 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp2, i32 0, i32 1
  store i32 1, i32* %dim3gep.15
  %dim3gep.26 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp2, i32 0, i32 2
  store i32 1, i32* %dim3gep.26
  %13 = load i32, i32* @threads_per_block_on_kernel_ten, align 4, !dbg !2541
  %dim3gep.07 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp3, i32 0, i32 0
  store i32 %13, i32* %dim3gep.07
  %dim3gep.18 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp3, i32 0, i32 1
  store i32 1, i32* %dim3gep.18
  %dim3gep.29 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp3, i32 0, i32 2
  store i32 1, i32* %dim3gep.29
  %14 = bitcast { i64, i32 }* %agg.tmp2.coerce to i8*, !dbg !2542
  %15 = bitcast %struct.dim3* %agg.tmp2 to i8*, !dbg !2542
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %14, i8* align 4 %15, i64 12, i1 false), !dbg !2542
  %16 = bitcast { i64, i32 }* %agg.tmp3.coerce to i8*, !dbg !2542
  %17 = bitcast %struct.dim3* %agg.tmp3 to i8*, !dbg !2542
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %16, i8* align 4 %17, i64 12, i1 false), !dbg !2542
  br label %header.010

header.010:                                       ; preds = %latch.013, %kcall.end
  %indvar.017 = phi i32 [ 0, %kcall.end ], [ %indvar.next.019, %latch.013 ]
  %exitcond21 = icmp ne i32 %indvar.017, %12
  br i1 %exitcond21, label %header.111.preheader.clone0, label %kcall.end7, !tulip.doall.loop.grid !2413

header.111.preheader.clone0:                      ; preds = %header.010
  br label %header.111.clone0

header.111.clone0:                                ; preds = %latch.112.clone0, %header.111.preheader.clone0
  %indvar.114.clone0 = phi i32 [ %indvar.next.116.clone0, %latch.112.clone0 ], [ 0, %header.111.preheader.clone0 ]
  %exitcond = icmp ne i32 %indvar.114.clone0, %13
  br i1 %exitcond, label %kcall.configok6.clone0, label %header.111.preheader, !tulip.doall.loop.block !2413

kcall.configok6.clone0:                           ; preds = %header.111.clone0
  %18 = load double*, double** @global_data_two, align 8, !dbg !2543
  %19 = load double*, double** @_ZL1x, align 8, !dbg !2544
  %20 = load double*, double** @_ZL1z, align 8, !dbg !2545
  call void @gpu_kernel_ten_20(double* %18, double* %19, double* %20, i32 %12, i32 1, i32 1, i32 %13, i32 1, i32 1, i32 %indvar.017, i32 0, i32 0, i32 %indvar.114.clone0, i32 0, i32 0)
  br label %latch.112.clone0

latch.112.clone0:                                 ; preds = %kcall.configok6.clone0
  %indvar.next.116.clone0 = add i32 %indvar.114.clone0, 1
  br label %header.111.clone0

header.111.preheader:                             ; preds = %header.111.clone0
  br label %header.111

header.111:                                       ; preds = %header.111.preheader, %latch.112
  %indvar.114 = phi i32 [ %indvar.next.116, %latch.112 ], [ 0, %header.111.preheader ]
  %exitcond20 = icmp ne i32 %indvar.114, %13
  br i1 %exitcond20, label %kcall.configok6, label %latch.013, !tulip.doall.loop.block !2413

latch.112:                                        ; preds = %kcall.configok6
  %indvar.next.116 = add i32 %indvar.114, 1
  br label %header.111

latch.013:                                        ; preds = %header.111
  %indvar.next.019 = add i32 %indvar.017, 1
  br label %header.010

kcall.configok6:                                  ; preds = %header.111
  %21 = load double*, double** @global_data_two, align 8, !dbg !2543
  %22 = load double*, double** @_ZL1x, align 8, !dbg !2544
  %23 = load double*, double** @_ZL1z, align 8, !dbg !2545
  call void @gpu_kernel_ten_21(double* %21, double* %22, double* %23, i32 %12, i32 1, i32 1, i32 %13, i32 1, i32 1, i32 %indvar.017, i32 0, i32 0, i32 %indvar.114, i32 0, i32 0)
  br label %latch.112

kcall.end7:                                       ; preds = %header.010
  store double 0.000000e+00, double* @global_data_reduce, align 8, !dbg !2546
  store double 0.000000e+00, double* @global_data_two_reduce, align 8, !dbg !2547
  %24 = load double*, double** @global_data, align 8, !dbg !2548
  %25 = bitcast double* %24 to i8*, !dbg !2548
  %26 = load double*, double** @global_data, align 8, !dbg !2549
  %27 = bitcast double* %26 to i8*, !dbg !2549
  %28 = load i64, i64* @size_reduce_memory_on_kernel_ten, align 8, !dbg !2550
  %call8 = call i32 @cudaMemcpy(i8* %25, i8* %27, i64 %28, i32 2), !dbg !2551, !tulip.target.end.of.map !2413
  call void @llvm.dbg.value(metadata i32 0, metadata !2552, metadata !DIExpression()), !dbg !2554
  br label %for.cond, !dbg !2555

for.cond:                                         ; preds = %for.inc, %kcall.end7
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %kcall.end7 ], !dbg !2554
  call void @llvm.dbg.value(metadata i64 %indvars.iv, metadata !2552, metadata !DIExpression()), !dbg !2554
  %29 = load i32, i32* @blocks_per_grid_on_kernel_ten, align 4, !dbg !2556
  %30 = sext i32 %29 to i64, !dbg !2558
  %cmp = icmp slt i64 %indvars.iv, %30, !dbg !2558
  br i1 %cmp, label %for.body, label %for.end, !dbg !2559

for.body:                                         ; preds = %for.cond
  %31 = load double*, double** @global_data, align 8, !dbg !2560
  %arrayidx = getelementptr inbounds double, double* %31, i64 %indvars.iv, !dbg !2560
  %32 = load double, double* %arrayidx, align 8, !dbg !2560
  %33 = load double, double* @global_data_reduce, align 8, !dbg !2562
  %add = fadd contract double %33, %32, !dbg !2562
  store double %add, double* @global_data_reduce, align 8, !dbg !2562
  %34 = load double*, double** @global_data_two, align 8, !dbg !2563
  %arrayidx11 = getelementptr inbounds double, double* %34, i64 %indvars.iv, !dbg !2563
  %35 = load double, double* %arrayidx11, align 8, !dbg !2563
  %36 = load double, double* @global_data_two_reduce, align 8, !dbg !2564
  %add12 = fadd contract double %36, %35, !dbg !2564
  store double %add12, double* @global_data_two_reduce, align 8, !dbg !2564
  br label %for.inc, !dbg !2565

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !2566
  call void @llvm.dbg.value(metadata i32 undef, metadata !2552, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !2554
  br label %for.cond, !dbg !2567, !llvm.loop !2568

for.end:                                          ; preds = %for.cond
  %37 = load double, double* @global_data_reduce, align 8, !dbg !2570
  store double %37, double* %norm_temp1, align 8, !dbg !2571
  %38 = load double, double* @global_data_two_reduce, align 8, !dbg !2572
  store double %38, double* %norm_temp2, align 8, !dbg !2573
  ret void, !dbg !2574
}

; Function Attrs: noinline uwtable
define internal void @_ZL22gpu_kernel_eleven_hostd(double %norm_temp2) #4 !dbg !2575 {
entry:
  %agg.tmp = alloca %struct.dim3, align 4
  %agg.tmp1 = alloca %struct.dim3, align 4
  %agg.tmp.coerce = alloca { i64, i32 }, align 4
  %agg.tmp1.coerce = alloca { i64, i32 }, align 4
  call void @llvm.dbg.value(metadata double %norm_temp2, metadata !2578, metadata !DIExpression()), !dbg !2579
  %0 = load i32, i32* @blocks_per_grid_on_kernel_eleven, align 4, !dbg !2580
  %dim3gep.0 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp, i32 0, i32 0
  store i32 %0, i32* %dim3gep.0
  %dim3gep.1 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp, i32 0, i32 1
  store i32 1, i32* %dim3gep.1
  %dim3gep.2 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp, i32 0, i32 2
  store i32 1, i32* %dim3gep.2
  %1 = load i32, i32* @threads_per_block_on_kernel_eleven, align 4, !dbg !2581
  %dim3gep.01 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp1, i32 0, i32 0
  store i32 %1, i32* %dim3gep.01
  %dim3gep.12 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp1, i32 0, i32 1
  store i32 1, i32* %dim3gep.12
  %dim3gep.23 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp1, i32 0, i32 2
  store i32 1, i32* %dim3gep.23
  %2 = bitcast { i64, i32 }* %agg.tmp.coerce to i8*, !dbg !2582
  %3 = bitcast %struct.dim3* %agg.tmp to i8*, !dbg !2582
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %2, i8* align 4 %3, i64 12, i1 false), !dbg !2582
  %4 = bitcast { i64, i32 }* %agg.tmp1.coerce to i8*, !dbg !2582
  %5 = bitcast %struct.dim3* %agg.tmp1 to i8*, !dbg !2582
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %4, i8* align 4 %5, i64 12, i1 false), !dbg !2582
  br label %header.0

header.0:                                         ; preds = %latch.0, %entry
  %indvar.0 = phi i32 [ 0, %entry ], [ %indvar.next.0, %latch.0 ]
  %exitcond4 = icmp ne i32 %indvar.0, %0
  br i1 %exitcond4, label %header.1.preheader, label %kcall.end, !tulip.doall.loop.grid !2413

header.1.preheader:                               ; preds = %header.0
  br label %header.1

header.1:                                         ; preds = %header.1.preheader, %latch.1
  %indvar.1 = phi i32 [ %indvar.next.1, %latch.1 ], [ 0, %header.1.preheader ]
  %exitcond = icmp ne i32 %indvar.1, %1
  br i1 %exitcond, label %kcall.configok, label %latch.0, !tulip.doall.loop.block !2413

latch.1:                                          ; preds = %kcall.configok
  %indvar.next.1 = add i32 %indvar.1, 1
  br label %header.1

latch.0:                                          ; preds = %header.1
  %indvar.next.0 = add i32 %indvar.0, 1
  br label %header.0

kcall.configok:                                   ; preds = %header.1
  %6 = load double*, double** @_ZL1x, align 8, !dbg !2583
  %7 = load double*, double** @_ZL1z, align 8, !dbg !2584
  call void @gpu_kernel_eleven_device(double %norm_temp2, double* %6, double* %7, i32 %0, i32 1, i32 1, i32 %1, i32 1, i32 1, i32 %indvar.0, i32 0, i32 0, i32 %indvar.1, i32 0, i32 0)
  br label %latch.1

kcall.end:                                        ; preds = %header.0
  ret void, !dbg !2585
}

; Function Attrs: nounwind readnone speculatable
declare double @llvm.fabs.f64(double) #0

; Function Attrs: nounwind
declare dso_local i8* @strcpy(i8*, i8*) #6

; Function Attrs: nounwind
declare dso_local i8* @strcat(i8*, i8*) #6

; Function Attrs: noinline uwtable
define internal void @_ZL11release_gpuv() #4 !dbg !2586 {
entry:
  ret void, !dbg !2587
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1 immarg) #8

declare dso_local i32 @cudaSetupArgument(i8*, i64, i64)

declare dso_local i32 @cudaLaunch(i8*)

declare dso_local i32 @cudaMemcpy(i8*, i8*, i64, i32) #5

; Function Attrs: noinline uwtable
define internal void @_ZL19gpu_kernel_one_hostv() #4 !dbg !2588 {
entry:
  %agg.tmp = alloca %struct.dim3, align 4
  %agg.tmp1 = alloca %struct.dim3, align 4
  %agg.tmp.coerce = alloca { i64, i32 }, align 4
  %agg.tmp1.coerce = alloca { i64, i32 }, align 4
  %0 = load i32, i32* @blocks_per_grid_on_kernel_one, align 4, !dbg !2589
  %dim3gep.0 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp, i32 0, i32 0
  store i32 %0, i32* %dim3gep.0
  %dim3gep.1 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp, i32 0, i32 1
  store i32 1, i32* %dim3gep.1
  %dim3gep.2 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp, i32 0, i32 2
  store i32 1, i32* %dim3gep.2
  %1 = load i32, i32* @threads_per_block_on_kernel_one, align 4, !dbg !2590
  %dim3gep.01 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp1, i32 0, i32 0
  store i32 %1, i32* %dim3gep.01
  %dim3gep.12 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp1, i32 0, i32 1
  store i32 1, i32* %dim3gep.12
  %dim3gep.23 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp1, i32 0, i32 2
  store i32 1, i32* %dim3gep.23
  %2 = bitcast { i64, i32 }* %agg.tmp.coerce to i8*, !dbg !2591
  %3 = bitcast %struct.dim3* %agg.tmp to i8*, !dbg !2591
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %2, i8* align 4 %3, i64 12, i1 false), !dbg !2591
  %4 = bitcast { i64, i32 }* %agg.tmp1.coerce to i8*, !dbg !2591
  %5 = bitcast %struct.dim3* %agg.tmp1 to i8*, !dbg !2591
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %4, i8* align 4 %5, i64 12, i1 false), !dbg !2591
  br label %header.0

header.0:                                         ; preds = %latch.0, %entry
  %indvar.0 = phi i32 [ 0, %entry ], [ %indvar.next.0, %latch.0 ]
  %exitcond4 = icmp ne i32 %indvar.0, %0
  br i1 %exitcond4, label %header.1.preheader, label %kcall.end, !tulip.doall.loop.grid !2413

header.1.preheader:                               ; preds = %header.0
  br label %header.1

header.1:                                         ; preds = %header.1.preheader, %latch.1
  %indvar.1 = phi i32 [ %indvar.next.1, %latch.1 ], [ 0, %header.1.preheader ]
  %exitcond = icmp ne i32 %indvar.1, %1
  br i1 %exitcond, label %kcall.configok, label %latch.0, !tulip.doall.loop.block !2413

latch.1:                                          ; preds = %kcall.configok
  %indvar.next.1 = add i32 %indvar.1, 1
  br label %header.1

latch.0:                                          ; preds = %header.1
  %indvar.next.0 = add i32 %indvar.0, 1
  br label %header.0

kcall.configok:                                   ; preds = %header.1
  %6 = load double*, double** @_ZL1p, align 8, !dbg !2592
  %7 = load double*, double** @_ZL1q, align 8, !dbg !2593
  %8 = load double*, double** @_ZL1r, align 8, !dbg !2594
  %9 = load double*, double** @_ZL1x, align 8, !dbg !2595
  %10 = load double*, double** @_ZL1z, align 8, !dbg !2596
  call void @gpu_kernel_one_device(double* %6, double* %7, double* %8, double* %9, double* %10, i32 %0, i32 1, i32 1, i32 %1, i32 1, i32 1, i32 %indvar.0, i32 0, i32 0, i32 %indvar.1, i32 0, i32 0)
  br label %latch.1

kcall.end:                                        ; preds = %header.0
  ret void, !dbg !2597
}

; Function Attrs: noinline uwtable
define internal void @_ZL19gpu_kernel_two_hostPd(double* %rho_host) #4 !dbg !2598 {
entry:
  %agg.tmp = alloca %struct.dim3, align 4
  %agg.tmp1 = alloca %struct.dim3, align 4
  %agg.tmp.coerce = alloca { i64, i32 }, align 4
  %agg.tmp1.coerce = alloca { i64, i32 }, align 4
  call void @llvm.dbg.value(metadata double* %rho_host, metadata !2599, metadata !DIExpression()), !dbg !2600
  %0 = load i32, i32* @blocks_per_grid_on_kernel_two, align 4, !dbg !2601
  %dim3gep.0 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp, i32 0, i32 0
  store i32 %0, i32* %dim3gep.0
  %dim3gep.1 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp, i32 0, i32 1
  store i32 1, i32* %dim3gep.1
  %dim3gep.2 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp, i32 0, i32 2
  store i32 1, i32* %dim3gep.2
  %1 = load i32, i32* @threads_per_block_on_kernel_two, align 4, !dbg !2602
  %dim3gep.01 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp1, i32 0, i32 0
  store i32 %1, i32* %dim3gep.01
  %dim3gep.12 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp1, i32 0, i32 1
  store i32 1, i32* %dim3gep.12
  %dim3gep.23 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp1, i32 0, i32 2
  store i32 1, i32* %dim3gep.23
  %2 = bitcast { i64, i32 }* %agg.tmp.coerce to i8*, !dbg !2603
  %3 = bitcast %struct.dim3* %agg.tmp to i8*, !dbg !2603
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %2, i8* align 4 %3, i64 12, i1 false), !dbg !2603
  %4 = bitcast { i64, i32 }* %agg.tmp1.coerce to i8*, !dbg !2603
  %5 = bitcast %struct.dim3* %agg.tmp1 to i8*, !dbg !2603
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %4, i8* align 4 %5, i64 12, i1 false), !dbg !2603
  br label %header.0

header.0:                                         ; preds = %latch.0, %entry
  %indvar.0 = phi i32 [ 0, %entry ], [ %indvar.next.0, %latch.0 ]
  %exitcond5 = icmp ne i32 %indvar.0, %0
  br i1 %exitcond5, label %header.1.preheader.clone0, label %kcall.end, !tulip.doall.loop.grid !2413

header.1.preheader.clone0:                        ; preds = %header.0
  br label %header.1.clone0

header.1.clone0:                                  ; preds = %latch.1.clone0, %header.1.preheader.clone0
  %indvar.1.clone0 = phi i32 [ %indvar.next.1.clone0, %latch.1.clone0 ], [ 0, %header.1.preheader.clone0 ]
  %exitcond = icmp ne i32 %indvar.1.clone0, %1
  br i1 %exitcond, label %kcall.configok.clone0, label %header.1.preheader, !tulip.doall.loop.block !2413

kcall.configok.clone0:                            ; preds = %header.1.clone0
  %6 = load double*, double** @_ZL1r, align 8, !dbg !2604
  %7 = load double*, double** @rho_device, align 8, !dbg !2605
  %8 = load double*, double** @global_data, align 8, !dbg !2606
  call void @gpu_kernel_two_device0(double* %6, double* %7, double* %8, i32 %0, i32 1, i32 1, i32 %1, i32 1, i32 1, i32 %indvar.0, i32 0, i32 0, i32 %indvar.1.clone0, i32 0, i32 0)
  br label %latch.1.clone0

latch.1.clone0:                                   ; preds = %kcall.configok.clone0
  %indvar.next.1.clone0 = add i32 %indvar.1.clone0, 1
  br label %header.1.clone0

header.1.preheader:                               ; preds = %header.1.clone0
  br label %header.1

header.1:                                         ; preds = %header.1.preheader, %latch.1
  %indvar.1 = phi i32 [ %indvar.next.1, %latch.1 ], [ 0, %header.1.preheader ]
  %exitcond4 = icmp ne i32 %indvar.1, %1
  br i1 %exitcond4, label %kcall.configok, label %latch.0, !tulip.doall.loop.block !2413

latch.1:                                          ; preds = %kcall.configok
  %indvar.next.1 = add i32 %indvar.1, 1
  br label %header.1

latch.0:                                          ; preds = %header.1
  %indvar.next.0 = add i32 %indvar.0, 1
  br label %header.0

kcall.configok:                                   ; preds = %header.1
  %9 = load double*, double** @_ZL1r, align 8, !dbg !2604
  %10 = load double*, double** @rho_device, align 8, !dbg !2605
  %11 = load double*, double** @global_data, align 8, !dbg !2606
  call void @gpu_kernel_two_device1(double* %9, double* %10, double* %11, i32 %0, i32 1, i32 1, i32 %1, i32 1, i32 1, i32 %indvar.0, i32 0, i32 0, i32 %indvar.1, i32 0, i32 0)
  br label %latch.1

kcall.end:                                        ; preds = %header.0
  store double 0.000000e+00, double* @global_data_reduce, align 8, !dbg !2607
  %12 = load double*, double** @global_data, align 8, !dbg !2608
  %13 = bitcast double* %12 to i8*, !dbg !2608
  %14 = load double*, double** @global_data, align 8, !dbg !2609
  %15 = bitcast double* %14 to i8*, !dbg !2609
  %16 = load i64, i64* @size_reduce_memory_on_kernel_two, align 8, !dbg !2610
  %call2 = call i32 @cudaMemcpy(i8* %13, i8* %15, i64 %16, i32 2), !dbg !2611, !tulip.target.end.of.map !2413
  call void @llvm.dbg.value(metadata i32 0, metadata !2612, metadata !DIExpression()), !dbg !2614
  br label %for.cond, !dbg !2615

for.cond:                                         ; preds = %for.inc, %kcall.end
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %kcall.end ], !dbg !2614
  call void @llvm.dbg.value(metadata i64 %indvars.iv, metadata !2612, metadata !DIExpression()), !dbg !2614
  %17 = load i32, i32* @blocks_per_grid_on_kernel_two, align 4, !dbg !2616
  %18 = sext i32 %17 to i64, !dbg !2618
  %cmp = icmp slt i64 %indvars.iv, %18, !dbg !2618
  br i1 %cmp, label %for.body, label %for.end, !dbg !2619

for.body:                                         ; preds = %for.cond
  %19 = load double*, double** @global_data, align 8, !dbg !2620
  %arrayidx = getelementptr inbounds double, double* %19, i64 %indvars.iv, !dbg !2620
  %20 = load double, double* %arrayidx, align 8, !dbg !2620
  %21 = load double, double* @global_data_reduce, align 8, !dbg !2622
  %add = fadd contract double %21, %20, !dbg !2622
  store double %add, double* @global_data_reduce, align 8, !dbg !2622
  br label %for.inc, !dbg !2623

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !2624
  call void @llvm.dbg.value(metadata i32 undef, metadata !2612, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !2614
  br label %for.cond, !dbg !2625, !llvm.loop !2626

for.end:                                          ; preds = %for.cond
  %22 = load double, double* @global_data_reduce, align 8, !dbg !2628
  store double %22, double* %rho_host, align 8, !dbg !2629
  ret void, !dbg !2630
}

; Function Attrs: noinline uwtable
define internal void @_ZL21gpu_kernel_three_hostv() #4 !dbg !2631 {
entry:
  %agg.tmp = alloca %struct.dim3, align 4
  %agg.tmp1 = alloca %struct.dim3, align 4
  %agg.tmp.coerce = alloca { i64, i32 }, align 4
  %agg.tmp1.coerce = alloca { i64, i32 }, align 4
  %0 = load i32, i32* @blocks_per_grid_on_kernel_three, align 4, !dbg !2632
  %dim3gep.0 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp, i32 0, i32 0
  store i32 %0, i32* %dim3gep.0
  %dim3gep.1 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp, i32 0, i32 1
  store i32 1, i32* %dim3gep.1
  %dim3gep.2 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp, i32 0, i32 2
  store i32 1, i32* %dim3gep.2
  %1 = load i32, i32* @threads_per_block_on_kernel_three, align 4, !dbg !2633
  %dim3gep.01 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp1, i32 0, i32 0
  store i32 %1, i32* %dim3gep.01
  %dim3gep.12 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp1, i32 0, i32 1
  store i32 1, i32* %dim3gep.12
  %dim3gep.23 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp1, i32 0, i32 2
  store i32 1, i32* %dim3gep.23
  %2 = bitcast { i64, i32 }* %agg.tmp.coerce to i8*, !dbg !2634
  %3 = bitcast %struct.dim3* %agg.tmp to i8*, !dbg !2634
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %2, i8* align 4 %3, i64 12, i1 false), !dbg !2634
  %4 = bitcast { i64, i32 }* %agg.tmp1.coerce to i8*, !dbg !2634
  %5 = bitcast %struct.dim3* %agg.tmp1 to i8*, !dbg !2634
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %4, i8* align 4 %5, i64 12, i1 false), !dbg !2634
  br label %header.0

header.0:                                         ; preds = %latch.0, %entry
  %indvar.0 = phi i32 [ 0, %entry ], [ %indvar.next.0, %latch.0 ]
  %exitcond5 = icmp ne i32 %indvar.0, %0
  br i1 %exitcond5, label %header.1.preheader.clone0, label %kcall.end, !tulip.doall.loop.grid !2413

header.1.preheader.clone0:                        ; preds = %header.0
  br label %header.1.clone0

header.1.clone0:                                  ; preds = %latch.1.clone0, %header.1.preheader.clone0
  %indvar.1.clone0 = phi i32 [ %indvar.next.1.clone0, %latch.1.clone0 ], [ 0, %header.1.preheader.clone0 ]
  %exitcond = icmp ne i32 %indvar.1.clone0, %1
  br i1 %exitcond, label %kcall.configok.clone0, label %header.1.preheader, !tulip.doall.loop.block !2413

kcall.configok.clone0:                            ; preds = %header.1.clone0
  %6 = load i32*, i32** @_ZL6colidx, align 8, !dbg !2635
  %7 = load i32*, i32** @_ZL6rowstr, align 8, !dbg !2636
  %8 = load double*, double** @_ZL1a, align 8, !dbg !2637
  %9 = load double*, double** @_ZL1p, align 8, !dbg !2638
  %10 = load double*, double** @_ZL1q, align 8, !dbg !2639
  call void @gpu_kernel_three_device0(i32* %6, i32* %7, double* %8, double* %9, double* %10, i32 %0, i32 1, i32 1, i32 %1, i32 1, i32 1, i32 %indvar.0, i32 0, i32 0, i32 %indvar.1.clone0, i32 0, i32 0)
  br label %latch.1.clone0

latch.1.clone0:                                   ; preds = %kcall.configok.clone0
  %indvar.next.1.clone0 = add i32 %indvar.1.clone0, 1
  br label %header.1.clone0

header.1.preheader:                               ; preds = %header.1.clone0
  br label %header.1

header.1:                                         ; preds = %header.1.preheader, %latch.1
  %indvar.1 = phi i32 [ %indvar.next.1, %latch.1 ], [ 0, %header.1.preheader ]
  %exitcond4 = icmp ne i32 %indvar.1, %1
  br i1 %exitcond4, label %kcall.configok, label %latch.0, !tulip.doall.loop.block !2413

latch.1:                                          ; preds = %kcall.configok
  %indvar.next.1 = add i32 %indvar.1, 1
  br label %header.1

latch.0:                                          ; preds = %header.1
  %indvar.next.0 = add i32 %indvar.0, 1
  br label %header.0

kcall.configok:                                   ; preds = %header.1
  %11 = load i32*, i32** @_ZL6colidx, align 8, !dbg !2635
  %12 = load i32*, i32** @_ZL6rowstr, align 8, !dbg !2636
  %13 = load double*, double** @_ZL1a, align 8, !dbg !2637
  %14 = load double*, double** @_ZL1p, align 8, !dbg !2638
  %15 = load double*, double** @_ZL1q, align 8, !dbg !2639
  call void @gpu_kernel_three_device1(i32* %11, i32* %12, double* %13, double* %14, double* %15, i32 %0, i32 1, i32 1, i32 %1, i32 1, i32 1, i32 %indvar.0, i32 0, i32 0, i32 %indvar.1, i32 0, i32 0)
  br label %latch.1

kcall.end:                                        ; preds = %header.0
  ret void, !dbg !2640
}

; Function Attrs: noinline uwtable
define internal void @_ZL20gpu_kernel_four_hostPd(double* %d_host) #4 !dbg !2641 {
entry:
  %agg.tmp = alloca %struct.dim3, align 4
  %agg.tmp1 = alloca %struct.dim3, align 4
  %agg.tmp.coerce = alloca { i64, i32 }, align 4
  %agg.tmp1.coerce = alloca { i64, i32 }, align 4
  call void @llvm.dbg.value(metadata double* %d_host, metadata !2642, metadata !DIExpression()), !dbg !2643
  %0 = load i32, i32* @blocks_per_grid_on_kernel_four, align 4, !dbg !2644
  %dim3gep.0 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp, i32 0, i32 0
  store i32 %0, i32* %dim3gep.0
  %dim3gep.1 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp, i32 0, i32 1
  store i32 1, i32* %dim3gep.1
  %dim3gep.2 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp, i32 0, i32 2
  store i32 1, i32* %dim3gep.2
  %1 = load i32, i32* @threads_per_block_on_kernel_four, align 4, !dbg !2645
  %dim3gep.01 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp1, i32 0, i32 0
  store i32 %1, i32* %dim3gep.01
  %dim3gep.12 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp1, i32 0, i32 1
  store i32 1, i32* %dim3gep.12
  %dim3gep.23 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp1, i32 0, i32 2
  store i32 1, i32* %dim3gep.23
  %2 = bitcast { i64, i32 }* %agg.tmp.coerce to i8*, !dbg !2646
  %3 = bitcast %struct.dim3* %agg.tmp to i8*, !dbg !2646
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %2, i8* align 4 %3, i64 12, i1 false), !dbg !2646
  %4 = bitcast { i64, i32 }* %agg.tmp1.coerce to i8*, !dbg !2646
  %5 = bitcast %struct.dim3* %agg.tmp1 to i8*, !dbg !2646
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %4, i8* align 4 %5, i64 12, i1 false), !dbg !2646
  br label %header.0

header.0:                                         ; preds = %latch.0, %entry
  %indvar.0 = phi i32 [ 0, %entry ], [ %indvar.next.0, %latch.0 ]
  %exitcond5 = icmp ne i32 %indvar.0, %0
  br i1 %exitcond5, label %header.1.preheader.clone0, label %kcall.end, !tulip.doall.loop.grid !2413

header.1.preheader.clone0:                        ; preds = %header.0
  br label %header.1.clone0

header.1.clone0:                                  ; preds = %latch.1.clone0, %header.1.preheader.clone0
  %indvar.1.clone0 = phi i32 [ %indvar.next.1.clone0, %latch.1.clone0 ], [ 0, %header.1.preheader.clone0 ]
  %exitcond = icmp ne i32 %indvar.1.clone0, %1
  br i1 %exitcond, label %kcall.configok.clone0, label %header.1.preheader, !tulip.doall.loop.block !2413

kcall.configok.clone0:                            ; preds = %header.1.clone0
  %6 = load double*, double** @d_device, align 8, !dbg !2647
  %7 = load double*, double** @_ZL1p, align 8, !dbg !2648
  %8 = load double*, double** @_ZL1q, align 8, !dbg !2649
  %9 = load double*, double** @global_data, align 8, !dbg !2650
  call void @gpu_kernel_four_device0(double* %6, double* %7, double* %8, double* %9, i32 %0, i32 1, i32 1, i32 %1, i32 1, i32 1, i32 %indvar.0, i32 0, i32 0, i32 %indvar.1.clone0, i32 0, i32 0)
  br label %latch.1.clone0

latch.1.clone0:                                   ; preds = %kcall.configok.clone0
  %indvar.next.1.clone0 = add i32 %indvar.1.clone0, 1
  br label %header.1.clone0

header.1.preheader:                               ; preds = %header.1.clone0
  br label %header.1

header.1:                                         ; preds = %header.1.preheader, %latch.1
  %indvar.1 = phi i32 [ %indvar.next.1, %latch.1 ], [ 0, %header.1.preheader ]
  %exitcond4 = icmp ne i32 %indvar.1, %1
  br i1 %exitcond4, label %kcall.configok, label %latch.0, !tulip.doall.loop.block !2413

latch.1:                                          ; preds = %kcall.configok
  %indvar.next.1 = add i32 %indvar.1, 1
  br label %header.1

latch.0:                                          ; preds = %header.1
  %indvar.next.0 = add i32 %indvar.0, 1
  br label %header.0

kcall.configok:                                   ; preds = %header.1
  %10 = load double*, double** @d_device, align 8, !dbg !2647
  %11 = load double*, double** @_ZL1p, align 8, !dbg !2648
  %12 = load double*, double** @_ZL1q, align 8, !dbg !2649
  %13 = load double*, double** @global_data, align 8, !dbg !2650
  call void @gpu_kernel_four_device1(double* %10, double* %11, double* %12, double* %13, i32 %0, i32 1, i32 1, i32 %1, i32 1, i32 1, i32 %indvar.0, i32 0, i32 0, i32 %indvar.1, i32 0, i32 0)
  br label %latch.1

kcall.end:                                        ; preds = %header.0
  store double 0.000000e+00, double* @global_data_reduce, align 8, !dbg !2651
  %14 = load double*, double** @global_data, align 8, !dbg !2652
  %15 = bitcast double* %14 to i8*, !dbg !2652
  %16 = load double*, double** @global_data, align 8, !dbg !2653
  %17 = bitcast double* %16 to i8*, !dbg !2653
  %18 = load i64, i64* @size_reduce_memory_on_kernel_four, align 8, !dbg !2654
  %call2 = call i32 @cudaMemcpy(i8* %15, i8* %17, i64 %18, i32 2), !dbg !2655, !tulip.target.end.of.map !2413
  call void @llvm.dbg.value(metadata i32 0, metadata !2656, metadata !DIExpression()), !dbg !2658
  br label %for.cond, !dbg !2659

for.cond:                                         ; preds = %for.inc, %kcall.end
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %kcall.end ], !dbg !2658
  call void @llvm.dbg.value(metadata i64 %indvars.iv, metadata !2656, metadata !DIExpression()), !dbg !2658
  %19 = load i32, i32* @blocks_per_grid_on_kernel_four, align 4, !dbg !2660
  %20 = sext i32 %19 to i64, !dbg !2662
  %cmp = icmp slt i64 %indvars.iv, %20, !dbg !2662
  br i1 %cmp, label %for.body, label %for.end, !dbg !2663

for.body:                                         ; preds = %for.cond
  %21 = load double*, double** @global_data, align 8, !dbg !2664
  %arrayidx = getelementptr inbounds double, double* %21, i64 %indvars.iv, !dbg !2664
  %22 = load double, double* %arrayidx, align 8, !dbg !2664
  %23 = load double, double* @global_data_reduce, align 8, !dbg !2666
  %add = fadd contract double %23, %22, !dbg !2666
  store double %add, double* @global_data_reduce, align 8, !dbg !2666
  br label %for.inc, !dbg !2667

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !2668
  call void @llvm.dbg.value(metadata i32 undef, metadata !2656, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !2658
  br label %for.cond, !dbg !2669, !llvm.loop !2670

for.end:                                          ; preds = %for.cond
  %24 = load double, double* @global_data_reduce, align 8, !dbg !2672
  store double %24, double* %d_host, align 8, !dbg !2673
  ret void, !dbg !2674
}

; Function Attrs: noinline uwtable
define internal void @_ZL20gpu_kernel_five_hostd(double %alpha_host) #4 !dbg !2675 {
entry:
  %agg.tmp = alloca %struct.dim3, align 4
  %agg.tmp1 = alloca %struct.dim3, align 4
  %agg.tmp.coerce = alloca { i64, i32 }, align 4
  %agg.tmp1.coerce = alloca { i64, i32 }, align 4
  %agg.tmp2 = alloca %struct.dim3, align 4
  %agg.tmp3 = alloca %struct.dim3, align 4
  %agg.tmp2.coerce = alloca { i64, i32 }, align 4
  %agg.tmp3.coerce = alloca { i64, i32 }, align 4
  call void @llvm.dbg.value(metadata double %alpha_host, metadata !2676, metadata !DIExpression()), !dbg !2677
  %0 = load i32, i32* @blocks_per_grid_on_kernel_five, align 4, !dbg !2678
  %dim3gep.0 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp, i32 0, i32 0
  store i32 %0, i32* %dim3gep.0
  %dim3gep.1 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp, i32 0, i32 1
  store i32 1, i32* %dim3gep.1
  %dim3gep.2 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp, i32 0, i32 2
  store i32 1, i32* %dim3gep.2
  %1 = load i32, i32* @threads_per_block_on_kernel_five, align 4, !dbg !2679
  %dim3gep.01 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp1, i32 0, i32 0
  store i32 %1, i32* %dim3gep.01
  %dim3gep.12 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp1, i32 0, i32 1
  store i32 1, i32* %dim3gep.12
  %dim3gep.23 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp1, i32 0, i32 2
  store i32 1, i32* %dim3gep.23
  %2 = bitcast { i64, i32 }* %agg.tmp.coerce to i8*, !dbg !2680
  %3 = bitcast %struct.dim3* %agg.tmp to i8*, !dbg !2680
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %2, i8* align 4 %3, i64 12, i1 false), !dbg !2680
  %4 = bitcast { i64, i32 }* %agg.tmp1.coerce to i8*, !dbg !2680
  %5 = bitcast %struct.dim3* %agg.tmp1 to i8*, !dbg !2680
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %4, i8* align 4 %5, i64 12, i1 false), !dbg !2680
  br label %header.0

header.0:                                         ; preds = %latch.0, %entry
  %indvar.0 = phi i32 [ 0, %entry ], [ %indvar.next.0, %latch.0 ]
  %exitcond22 = icmp ne i32 %indvar.0, %0
  br i1 %exitcond22, label %header.1.preheader, label %kcall.end, !tulip.doall.loop.grid !2413

header.1.preheader:                               ; preds = %header.0
  br label %header.1

header.1:                                         ; preds = %header.1.preheader, %latch.1
  %indvar.1 = phi i32 [ %indvar.next.1, %latch.1 ], [ 0, %header.1.preheader ]
  %exitcond21 = icmp ne i32 %indvar.1, %1
  br i1 %exitcond21, label %kcall.configok, label %latch.0, !tulip.doall.loop.block !2413

latch.1:                                          ; preds = %kcall.configok
  %indvar.next.1 = add i32 %indvar.1, 1
  br label %header.1

latch.0:                                          ; preds = %header.1
  %indvar.next.0 = add i32 %indvar.0, 1
  br label %header.0

kcall.configok:                                   ; preds = %header.1
  %6 = load double*, double** @_ZL1p, align 8, !dbg !2681
  %7 = load double*, double** @_ZL1z, align 8, !dbg !2682
  call void @gpu_kernel_five_1(double %alpha_host, double* %6, double* %7, i32 %0, i32 1, i32 1, i32 %1, i32 1, i32 1, i32 %indvar.0, i32 0, i32 0, i32 %indvar.1, i32 0, i32 0)
  br label %latch.1

kcall.end:                                        ; preds = %header.0
  %8 = load i32, i32* @blocks_per_grid_on_kernel_five, align 4, !dbg !2683
  %dim3gep.04 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp2, i32 0, i32 0
  store i32 %8, i32* %dim3gep.04
  %dim3gep.15 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp2, i32 0, i32 1
  store i32 1, i32* %dim3gep.15
  %dim3gep.26 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp2, i32 0, i32 2
  store i32 1, i32* %dim3gep.26
  %9 = load i32, i32* @threads_per_block_on_kernel_five, align 4, !dbg !2684
  %dim3gep.07 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp3, i32 0, i32 0
  store i32 %9, i32* %dim3gep.07
  %dim3gep.18 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp3, i32 0, i32 1
  store i32 1, i32* %dim3gep.18
  %dim3gep.29 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp3, i32 0, i32 2
  store i32 1, i32* %dim3gep.29
  %10 = bitcast { i64, i32 }* %agg.tmp2.coerce to i8*, !dbg !2685
  %11 = bitcast %struct.dim3* %agg.tmp2 to i8*, !dbg !2685
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %10, i8* align 4 %11, i64 12, i1 false), !dbg !2685
  %12 = bitcast { i64, i32 }* %agg.tmp3.coerce to i8*, !dbg !2685
  %13 = bitcast %struct.dim3* %agg.tmp3 to i8*, !dbg !2685
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %12, i8* align 4 %13, i64 12, i1 false), !dbg !2685
  br label %header.010

header.010:                                       ; preds = %latch.013, %kcall.end
  %indvar.017 = phi i32 [ 0, %kcall.end ], [ %indvar.next.019, %latch.013 ]
  %exitcond20 = icmp ne i32 %indvar.017, %8
  br i1 %exitcond20, label %header.111.preheader, label %kcall.end7, !tulip.doall.loop.grid !2413

header.111.preheader:                             ; preds = %header.010
  br label %header.111

header.111:                                       ; preds = %header.111.preheader, %latch.112
  %indvar.114 = phi i32 [ %indvar.next.116, %latch.112 ], [ 0, %header.111.preheader ]
  %exitcond = icmp ne i32 %indvar.114, %9
  br i1 %exitcond, label %kcall.configok6, label %latch.013, !tulip.doall.loop.block !2413

latch.112:                                        ; preds = %kcall.configok6
  %indvar.next.116 = add i32 %indvar.114, 1
  br label %header.111

latch.013:                                        ; preds = %header.111
  %indvar.next.019 = add i32 %indvar.017, 1
  br label %header.010

kcall.configok6:                                  ; preds = %header.111
  %14 = load double*, double** @_ZL1q, align 8, !dbg !2686
  %15 = load double*, double** @_ZL1r, align 8, !dbg !2687
  call void @gpu_kernel_five_2(double %alpha_host, double* %14, double* %15, i32 %8, i32 1, i32 1, i32 %9, i32 1, i32 1, i32 %indvar.017, i32 0, i32 0, i32 %indvar.114, i32 0, i32 0)
  br label %latch.112

kcall.end7:                                       ; preds = %header.010
  ret void, !dbg !2688
}

; Function Attrs: noinline uwtable
define internal void @_ZL19gpu_kernel_six_hostPd(double* %rho_host) #4 !dbg !2689 {
entry:
  %agg.tmp = alloca %struct.dim3, align 4
  %agg.tmp1 = alloca %struct.dim3, align 4
  %agg.tmp.coerce = alloca { i64, i32 }, align 4
  %agg.tmp1.coerce = alloca { i64, i32 }, align 4
  call void @llvm.dbg.value(metadata double* %rho_host, metadata !2690, metadata !DIExpression()), !dbg !2691
  %0 = load i32, i32* @blocks_per_grid_on_kernel_six, align 4, !dbg !2692
  %dim3gep.0 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp, i32 0, i32 0
  store i32 %0, i32* %dim3gep.0
  %dim3gep.1 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp, i32 0, i32 1
  store i32 1, i32* %dim3gep.1
  %dim3gep.2 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp, i32 0, i32 2
  store i32 1, i32* %dim3gep.2
  %1 = load i32, i32* @threads_per_block_on_kernel_six, align 4, !dbg !2693
  %dim3gep.01 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp1, i32 0, i32 0
  store i32 %1, i32* %dim3gep.01
  %dim3gep.12 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp1, i32 0, i32 1
  store i32 1, i32* %dim3gep.12
  %dim3gep.23 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp1, i32 0, i32 2
  store i32 1, i32* %dim3gep.23
  %2 = bitcast { i64, i32 }* %agg.tmp.coerce to i8*, !dbg !2694
  %3 = bitcast %struct.dim3* %agg.tmp to i8*, !dbg !2694
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %2, i8* align 4 %3, i64 12, i1 false), !dbg !2694
  %4 = bitcast { i64, i32 }* %agg.tmp1.coerce to i8*, !dbg !2694
  %5 = bitcast %struct.dim3* %agg.tmp1 to i8*, !dbg !2694
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %4, i8* align 4 %5, i64 12, i1 false), !dbg !2694
  br label %header.0

header.0:                                         ; preds = %latch.0, %entry
  %indvar.0 = phi i32 [ 0, %entry ], [ %indvar.next.0, %latch.0 ]
  %exitcond5 = icmp ne i32 %indvar.0, %0
  br i1 %exitcond5, label %header.1.preheader.clone0, label %kcall.end, !tulip.doall.loop.grid !2413

header.1.preheader.clone0:                        ; preds = %header.0
  br label %header.1.clone0

header.1.clone0:                                  ; preds = %latch.1.clone0, %header.1.preheader.clone0
  %indvar.1.clone0 = phi i32 [ %indvar.next.1.clone0, %latch.1.clone0 ], [ 0, %header.1.preheader.clone0 ]
  %exitcond = icmp ne i32 %indvar.1.clone0, %1
  br i1 %exitcond, label %kcall.configok.clone0, label %header.1.preheader, !tulip.doall.loop.block !2413

kcall.configok.clone0:                            ; preds = %header.1.clone0
  %6 = load double*, double** @_ZL1r, align 8, !dbg !2695
  %7 = load double*, double** @global_data, align 8, !dbg !2696
  call void @gpu_kernel_six_device0(double* %6, double* %7, i32 %0, i32 1, i32 1, i32 %1, i32 1, i32 1, i32 %indvar.0, i32 0, i32 0, i32 %indvar.1.clone0, i32 0, i32 0)
  br label %latch.1.clone0

latch.1.clone0:                                   ; preds = %kcall.configok.clone0
  %indvar.next.1.clone0 = add i32 %indvar.1.clone0, 1
  br label %header.1.clone0

header.1.preheader:                               ; preds = %header.1.clone0
  br label %header.1

header.1:                                         ; preds = %header.1.preheader, %latch.1
  %indvar.1 = phi i32 [ %indvar.next.1, %latch.1 ], [ 0, %header.1.preheader ]
  %exitcond4 = icmp ne i32 %indvar.1, %1
  br i1 %exitcond4, label %kcall.configok, label %latch.0, !tulip.doall.loop.block !2413

latch.1:                                          ; preds = %kcall.configok
  %indvar.next.1 = add i32 %indvar.1, 1
  br label %header.1

latch.0:                                          ; preds = %header.1
  %indvar.next.0 = add i32 %indvar.0, 1
  br label %header.0

kcall.configok:                                   ; preds = %header.1
  %8 = load double*, double** @_ZL1r, align 8, !dbg !2695
  %9 = load double*, double** @global_data, align 8, !dbg !2696
  call void @gpu_kernel_six_device1(double* %8, double* %9, i32 %0, i32 1, i32 1, i32 %1, i32 1, i32 1, i32 %indvar.0, i32 0, i32 0, i32 %indvar.1, i32 0, i32 0)
  br label %latch.1

kcall.end:                                        ; preds = %header.0
  store double 0.000000e+00, double* @global_data_reduce, align 8, !dbg !2697
  %10 = load double*, double** @global_data, align 8, !dbg !2698
  %11 = bitcast double* %10 to i8*, !dbg !2698
  %12 = load double*, double** @global_data, align 8, !dbg !2699
  %13 = bitcast double* %12 to i8*, !dbg !2699
  %14 = load i64, i64* @size_reduce_memory_on_kernel_six, align 8, !dbg !2700
  %call2 = call i32 @cudaMemcpy(i8* %11, i8* %13, i64 %14, i32 2), !dbg !2701, !tulip.target.end.of.map !2413
  call void @llvm.dbg.value(metadata i32 0, metadata !2702, metadata !DIExpression()), !dbg !2704
  br label %for.cond, !dbg !2705

for.cond:                                         ; preds = %for.inc, %kcall.end
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %kcall.end ], !dbg !2704
  call void @llvm.dbg.value(metadata i64 %indvars.iv, metadata !2702, metadata !DIExpression()), !dbg !2704
  %15 = load i32, i32* @blocks_per_grid_on_kernel_six, align 4, !dbg !2706
  %16 = sext i32 %15 to i64, !dbg !2708
  %cmp = icmp slt i64 %indvars.iv, %16, !dbg !2708
  br i1 %cmp, label %for.body, label %for.end, !dbg !2709

for.body:                                         ; preds = %for.cond
  %17 = load double*, double** @global_data, align 8, !dbg !2710
  %arrayidx = getelementptr inbounds double, double* %17, i64 %indvars.iv, !dbg !2710
  %18 = load double, double* %arrayidx, align 8, !dbg !2710
  %19 = load double, double* @global_data_reduce, align 8, !dbg !2712
  %add = fadd contract double %19, %18, !dbg !2712
  store double %add, double* @global_data_reduce, align 8, !dbg !2712
  br label %for.inc, !dbg !2713

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !2714
  call void @llvm.dbg.value(metadata i32 undef, metadata !2702, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !2704
  br label %for.cond, !dbg !2715, !llvm.loop !2716

for.end:                                          ; preds = %for.cond
  %20 = load double, double* @global_data_reduce, align 8, !dbg !2718
  store double %20, double* %rho_host, align 8, !dbg !2719
  ret void, !dbg !2720
}

; Function Attrs: noinline uwtable
define internal void @_ZL21gpu_kernel_seven_hostd(double %beta_host) #4 !dbg !2721 {
entry:
  %agg.tmp = alloca %struct.dim3, align 4
  %agg.tmp1 = alloca %struct.dim3, align 4
  %agg.tmp.coerce = alloca { i64, i32 }, align 4
  %agg.tmp1.coerce = alloca { i64, i32 }, align 4
  call void @llvm.dbg.value(metadata double %beta_host, metadata !2722, metadata !DIExpression()), !dbg !2723
  %0 = load i32, i32* @blocks_per_grid_on_kernel_seven, align 4, !dbg !2724
  %dim3gep.0 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp, i32 0, i32 0
  store i32 %0, i32* %dim3gep.0
  %dim3gep.1 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp, i32 0, i32 1
  store i32 1, i32* %dim3gep.1
  %dim3gep.2 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp, i32 0, i32 2
  store i32 1, i32* %dim3gep.2
  %1 = load i32, i32* @threads_per_block_on_kernel_seven, align 4, !dbg !2725
  %dim3gep.01 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp1, i32 0, i32 0
  store i32 %1, i32* %dim3gep.01
  %dim3gep.12 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp1, i32 0, i32 1
  store i32 1, i32* %dim3gep.12
  %dim3gep.23 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp1, i32 0, i32 2
  store i32 1, i32* %dim3gep.23
  %2 = bitcast { i64, i32 }* %agg.tmp.coerce to i8*, !dbg !2726
  %3 = bitcast %struct.dim3* %agg.tmp to i8*, !dbg !2726
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %2, i8* align 4 %3, i64 12, i1 false), !dbg !2726
  %4 = bitcast { i64, i32 }* %agg.tmp1.coerce to i8*, !dbg !2726
  %5 = bitcast %struct.dim3* %agg.tmp1 to i8*, !dbg !2726
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %4, i8* align 4 %5, i64 12, i1 false), !dbg !2726
  br label %header.0

header.0:                                         ; preds = %latch.0, %entry
  %indvar.0 = phi i32 [ 0, %entry ], [ %indvar.next.0, %latch.0 ]
  %exitcond4 = icmp ne i32 %indvar.0, %0
  br i1 %exitcond4, label %header.1.preheader, label %kcall.end, !tulip.doall.loop.grid !2413

header.1.preheader:                               ; preds = %header.0
  br label %header.1

header.1:                                         ; preds = %header.1.preheader, %latch.1
  %indvar.1 = phi i32 [ %indvar.next.1, %latch.1 ], [ 0, %header.1.preheader ]
  %exitcond = icmp ne i32 %indvar.1, %1
  br i1 %exitcond, label %kcall.configok, label %latch.0, !tulip.doall.loop.block !2413

latch.1:                                          ; preds = %kcall.configok
  %indvar.next.1 = add i32 %indvar.1, 1
  br label %header.1

latch.0:                                          ; preds = %header.1
  %indvar.next.0 = add i32 %indvar.0, 1
  br label %header.0

kcall.configok:                                   ; preds = %header.1
  %6 = load double*, double** @_ZL1p, align 8, !dbg !2727
  %7 = load double*, double** @_ZL1r, align 8, !dbg !2728
  call void @gpu_kernel_seven_device(double %beta_host, double* %6, double* %7, i32 %0, i32 1, i32 1, i32 %1, i32 1, i32 1, i32 %indvar.0, i32 0, i32 0, i32 %indvar.1, i32 0, i32 0)
  br label %latch.1

kcall.end:                                        ; preds = %header.0
  ret void, !dbg !2729
}

; Function Attrs: noinline uwtable
define internal void @_ZL21gpu_kernel_eight_hostv() #4 !dbg !2730 {
entry:
  %agg.tmp = alloca %struct.dim3, align 4
  %agg.tmp1 = alloca %struct.dim3, align 4
  %agg.tmp.coerce = alloca { i64, i32 }, align 4
  %agg.tmp1.coerce = alloca { i64, i32 }, align 4
  %0 = load i32, i32* @blocks_per_grid_on_kernel_eight, align 4, !dbg !2731
  %dim3gep.0 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp, i32 0, i32 0
  store i32 %0, i32* %dim3gep.0
  %dim3gep.1 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp, i32 0, i32 1
  store i32 1, i32* %dim3gep.1
  %dim3gep.2 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp, i32 0, i32 2
  store i32 1, i32* %dim3gep.2
  %1 = load i32, i32* @threads_per_block_on_kernel_eight, align 4, !dbg !2732
  %dim3gep.01 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp1, i32 0, i32 0
  store i32 %1, i32* %dim3gep.01
  %dim3gep.12 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp1, i32 0, i32 1
  store i32 1, i32* %dim3gep.12
  %dim3gep.23 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp1, i32 0, i32 2
  store i32 1, i32* %dim3gep.23
  %2 = bitcast { i64, i32 }* %agg.tmp.coerce to i8*, !dbg !2733
  %3 = bitcast %struct.dim3* %agg.tmp to i8*, !dbg !2733
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %2, i8* align 4 %3, i64 12, i1 false), !dbg !2733
  %4 = bitcast { i64, i32 }* %agg.tmp1.coerce to i8*, !dbg !2733
  %5 = bitcast %struct.dim3* %agg.tmp1 to i8*, !dbg !2733
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %4, i8* align 4 %5, i64 12, i1 false), !dbg !2733
  br label %header.0

header.0:                                         ; preds = %latch.0, %entry
  %indvar.0 = phi i32 [ 0, %entry ], [ %indvar.next.0, %latch.0 ]
  %exitcond5 = icmp ne i32 %indvar.0, %0
  br i1 %exitcond5, label %header.1.preheader.clone0, label %kcall.end, !tulip.doall.loop.grid !2413

header.1.preheader.clone0:                        ; preds = %header.0
  br label %header.1.clone0

header.1.clone0:                                  ; preds = %latch.1.clone0, %header.1.preheader.clone0
  %indvar.1.clone0 = phi i32 [ %indvar.next.1.clone0, %latch.1.clone0 ], [ 0, %header.1.preheader.clone0 ]
  %exitcond = icmp ne i32 %indvar.1.clone0, %1
  br i1 %exitcond, label %kcall.configok.clone0, label %header.1.preheader, !tulip.doall.loop.block !2413

kcall.configok.clone0:                            ; preds = %header.1.clone0
  %6 = load i32*, i32** @_ZL6colidx, align 8, !dbg !2734
  %7 = load i32*, i32** @_ZL6rowstr, align 8, !dbg !2735
  %8 = load double*, double** @_ZL1a, align 8, !dbg !2736
  %9 = load double*, double** @_ZL1r, align 8, !dbg !2737
  %10 = load double*, double** @_ZL1z, align 8, !dbg !2738
  call void @gpu_kernel_eight_device0(i32* %6, i32* %7, double* %8, double* %9, double* %10, i32 %0, i32 1, i32 1, i32 %1, i32 1, i32 1, i32 %indvar.0, i32 0, i32 0, i32 %indvar.1.clone0, i32 0, i32 0)
  br label %latch.1.clone0

latch.1.clone0:                                   ; preds = %kcall.configok.clone0
  %indvar.next.1.clone0 = add i32 %indvar.1.clone0, 1
  br label %header.1.clone0

header.1.preheader:                               ; preds = %header.1.clone0
  br label %header.1

header.1:                                         ; preds = %header.1.preheader, %latch.1
  %indvar.1 = phi i32 [ %indvar.next.1, %latch.1 ], [ 0, %header.1.preheader ]
  %exitcond4 = icmp ne i32 %indvar.1, %1
  br i1 %exitcond4, label %kcall.configok, label %latch.0, !tulip.doall.loop.block !2413

latch.1:                                          ; preds = %kcall.configok
  %indvar.next.1 = add i32 %indvar.1, 1
  br label %header.1

latch.0:                                          ; preds = %header.1
  %indvar.next.0 = add i32 %indvar.0, 1
  br label %header.0

kcall.configok:                                   ; preds = %header.1
  %11 = load i32*, i32** @_ZL6colidx, align 8, !dbg !2734
  %12 = load i32*, i32** @_ZL6rowstr, align 8, !dbg !2735
  %13 = load double*, double** @_ZL1a, align 8, !dbg !2736
  %14 = load double*, double** @_ZL1r, align 8, !dbg !2737
  %15 = load double*, double** @_ZL1z, align 8, !dbg !2738
  call void @gpu_kernel_eight_device1(i32* %11, i32* %12, double* %13, double* %14, double* %15, i32 %0, i32 1, i32 1, i32 %1, i32 1, i32 1, i32 %indvar.0, i32 0, i32 0, i32 %indvar.1, i32 0, i32 0)
  br label %latch.1

kcall.end:                                        ; preds = %header.0
  ret void, !dbg !2739
}

; Function Attrs: noinline uwtable
define internal void @_ZL20gpu_kernel_nine_hostPd(double* %sum_host) #4 !dbg !2740 {
entry:
  %agg.tmp = alloca %struct.dim3, align 4
  %agg.tmp1 = alloca %struct.dim3, align 4
  %agg.tmp.coerce = alloca { i64, i32 }, align 4
  %agg.tmp1.coerce = alloca { i64, i32 }, align 4
  call void @llvm.dbg.value(metadata double* %sum_host, metadata !2741, metadata !DIExpression()), !dbg !2742
  %0 = load i32, i32* @blocks_per_grid_on_kernel_nine, align 4, !dbg !2743
  %dim3gep.0 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp, i32 0, i32 0
  store i32 %0, i32* %dim3gep.0
  %dim3gep.1 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp, i32 0, i32 1
  store i32 1, i32* %dim3gep.1
  %dim3gep.2 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp, i32 0, i32 2
  store i32 1, i32* %dim3gep.2
  %1 = load i32, i32* @threads_per_block_on_kernel_nine, align 4, !dbg !2744
  %dim3gep.01 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp1, i32 0, i32 0
  store i32 %1, i32* %dim3gep.01
  %dim3gep.12 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp1, i32 0, i32 1
  store i32 1, i32* %dim3gep.12
  %dim3gep.23 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp1, i32 0, i32 2
  store i32 1, i32* %dim3gep.23
  %2 = bitcast { i64, i32 }* %agg.tmp.coerce to i8*, !dbg !2745
  %3 = bitcast %struct.dim3* %agg.tmp to i8*, !dbg !2745
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %2, i8* align 4 %3, i64 12, i1 false), !dbg !2745
  %4 = bitcast { i64, i32 }* %agg.tmp1.coerce to i8*, !dbg !2745
  %5 = bitcast %struct.dim3* %agg.tmp1 to i8*, !dbg !2745
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %4, i8* align 4 %5, i64 12, i1 false), !dbg !2745
  br label %header.0

header.0:                                         ; preds = %latch.0, %entry
  %indvar.0 = phi i32 [ 0, %entry ], [ %indvar.next.0, %latch.0 ]
  %exitcond5 = icmp ne i32 %indvar.0, %0
  br i1 %exitcond5, label %header.1.preheader.clone0, label %kcall.end, !tulip.doall.loop.grid !2413

header.1.preheader.clone0:                        ; preds = %header.0
  br label %header.1.clone0

header.1.clone0:                                  ; preds = %latch.1.clone0, %header.1.preheader.clone0
  %indvar.1.clone0 = phi i32 [ %indvar.next.1.clone0, %latch.1.clone0 ], [ 0, %header.1.preheader.clone0 ]
  %exitcond = icmp ne i32 %indvar.1.clone0, %1
  br i1 %exitcond, label %kcall.configok.clone0, label %header.1.preheader, !tulip.doall.loop.block !2413

kcall.configok.clone0:                            ; preds = %header.1.clone0
  %6 = load double*, double** @_ZL1r, align 8, !dbg !2746
  %7 = load double*, double** @_ZL1x, align 8, !dbg !2747
  %8 = load double*, double** @sum_device, align 8, !dbg !2748
  %9 = load double*, double** @global_data, align 8, !dbg !2749
  call void @gpu_kernel_nine_device0(double* %6, double* %7, double* %8, double* %9, i32 %0, i32 1, i32 1, i32 %1, i32 1, i32 1, i32 %indvar.0, i32 0, i32 0, i32 %indvar.1.clone0, i32 0, i32 0)
  br label %latch.1.clone0

latch.1.clone0:                                   ; preds = %kcall.configok.clone0
  %indvar.next.1.clone0 = add i32 %indvar.1.clone0, 1
  br label %header.1.clone0

header.1.preheader:                               ; preds = %header.1.clone0
  br label %header.1

header.1:                                         ; preds = %header.1.preheader, %latch.1
  %indvar.1 = phi i32 [ %indvar.next.1, %latch.1 ], [ 0, %header.1.preheader ]
  %exitcond4 = icmp ne i32 %indvar.1, %1
  br i1 %exitcond4, label %kcall.configok, label %latch.0, !tulip.doall.loop.block !2413

latch.1:                                          ; preds = %kcall.configok
  %indvar.next.1 = add i32 %indvar.1, 1
  br label %header.1

latch.0:                                          ; preds = %header.1
  %indvar.next.0 = add i32 %indvar.0, 1
  br label %header.0

kcall.configok:                                   ; preds = %header.1
  %10 = load double*, double** @_ZL1r, align 8, !dbg !2746
  %11 = load double*, double** @_ZL1x, align 8, !dbg !2747
  %12 = load double*, double** @sum_device, align 8, !dbg !2748
  %13 = load double*, double** @global_data, align 8, !dbg !2749
  call void @gpu_kernel_nine_device1(double* %10, double* %11, double* %12, double* %13, i32 %0, i32 1, i32 1, i32 %1, i32 1, i32 1, i32 %indvar.0, i32 0, i32 0, i32 %indvar.1, i32 0, i32 0)
  br label %latch.1

kcall.end:                                        ; preds = %header.0
  store double 0.000000e+00, double* @global_data_reduce, align 8, !dbg !2750
  %14 = load double*, double** @global_data, align 8, !dbg !2751
  %15 = bitcast double* %14 to i8*, !dbg !2751
  %16 = load double*, double** @global_data, align 8, !dbg !2752
  %17 = bitcast double* %16 to i8*, !dbg !2752
  %18 = load i64, i64* @size_reduce_memory_on_kernel_nine, align 8, !dbg !2753
  %call2 = call i32 @cudaMemcpy(i8* %15, i8* %17, i64 %18, i32 2), !dbg !2754, !tulip.target.end.of.map !2413
  call void @llvm.dbg.value(metadata i32 0, metadata !2755, metadata !DIExpression()), !dbg !2757
  br label %for.cond, !dbg !2758

for.cond:                                         ; preds = %for.inc, %kcall.end
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %kcall.end ], !dbg !2757
  call void @llvm.dbg.value(metadata i64 %indvars.iv, metadata !2755, metadata !DIExpression()), !dbg !2757
  %19 = load i32, i32* @blocks_per_grid_on_kernel_nine, align 4, !dbg !2759
  %20 = sext i32 %19 to i64, !dbg !2761
  %cmp = icmp slt i64 %indvars.iv, %20, !dbg !2761
  br i1 %cmp, label %for.body, label %for.end, !dbg !2762

for.body:                                         ; preds = %for.cond
  %21 = load double*, double** @global_data, align 8, !dbg !2763
  %arrayidx = getelementptr inbounds double, double* %21, i64 %indvars.iv, !dbg !2763
  %22 = load double, double* %arrayidx, align 8, !dbg !2763
  %23 = load double, double* @global_data_reduce, align 8, !dbg !2765
  %add = fadd contract double %23, %22, !dbg !2765
  store double %add, double* @global_data_reduce, align 8, !dbg !2765
  br label %for.inc, !dbg !2766

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !2767
  call void @llvm.dbg.value(metadata i32 undef, metadata !2755, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !2757
  br label %for.cond, !dbg !2768, !llvm.loop !2769

for.end:                                          ; preds = %for.cond
  %24 = load double, double* @global_data_reduce, align 8, !dbg !2771
  store double %24, double* %sum_host, align 8, !dbg !2772
  ret void, !dbg !2773
}

; Function Attrs: nounwind readnone speculatable
declare double @llvm.ceil.f64(double) #0

; Function Attrs: noinline uwtable
define internal void @_ZL6sprnvciiiPdPi(i32 %n, i32 %nz, i32 %nn1, double* %v, i32* %iv) #4 !dbg !2774 {
entry:
  call void @llvm.dbg.value(metadata i32 %n, metadata !2777, metadata !DIExpression()), !dbg !2778
  call void @llvm.dbg.value(metadata i32 %nz, metadata !2779, metadata !DIExpression()), !dbg !2778
  call void @llvm.dbg.value(metadata i32 %nn1, metadata !2780, metadata !DIExpression()), !dbg !2778
  call void @llvm.dbg.value(metadata double* %v, metadata !2781, metadata !DIExpression()), !dbg !2778
  call void @llvm.dbg.value(metadata i32* %iv, metadata !2782, metadata !DIExpression()), !dbg !2778
  call void @llvm.dbg.value(metadata i32 0, metadata !2783, metadata !DIExpression()), !dbg !2778
  %0 = sext i32 %nz to i64, !dbg !2784
  br label %while.cond, !dbg !2784

while.cond:                                       ; preds = %for.end, %entry
  %indvars.iv1 = phi i64 [ %indvars.iv.next2, %for.end ], [ 0, %entry ], !dbg !2778
  call void @llvm.dbg.value(metadata i64 %indvars.iv1, metadata !2783, metadata !DIExpression()), !dbg !2778
  %cmp = icmp slt i64 %indvars.iv1, %0, !dbg !2785
  br i1 %cmp, label %while.body, label %while.end, !dbg !2784

while.body:                                       ; preds = %while.cond
  %1 = load double, double* @_ZL5amult, align 8, !dbg !2786
  %call = call double @_Z6randlcPdd(double* @_ZL4tran, double %1), !dbg !2788
  call void @llvm.dbg.value(metadata double %call, metadata !2789, metadata !DIExpression()), !dbg !2778
  %2 = load double, double* @_ZL5amult, align 8, !dbg !2790
  %call1 = call double @_Z6randlcPdd(double* @_ZL4tran, double %2), !dbg !2791
  call void @llvm.dbg.value(metadata double %call1, metadata !2792, metadata !DIExpression()), !dbg !2778
  %call2 = call i32 @_ZL6icnvrtdi(double %call1, i32 %nn1), !dbg !2793
  %add = add nsw i32 %call2, 1, !dbg !2794
  call void @llvm.dbg.value(metadata i32 %add, metadata !2795, metadata !DIExpression()), !dbg !2778
  call void @llvm.dbg.value(metadata i32 0, metadata !2796, metadata !DIExpression()), !dbg !2797
  call void @llvm.dbg.value(metadata i32 0, metadata !2798, metadata !DIExpression()), !dbg !2778
  br label %for.cond, !dbg !2799

for.cond:                                         ; preds = %for.inc, %while.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %while.body ], !dbg !2801
  call void @llvm.dbg.value(metadata i64 %indvars.iv, metadata !2798, metadata !DIExpression()), !dbg !2778
  %exitcond = icmp ne i64 %indvars.iv, %indvars.iv1, !dbg !2802
  br i1 %exitcond, label %for.body, label %for.end.loopexit, !dbg !2804

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds i32, i32* %iv, i64 %indvars.iv, !dbg !2805
  %3 = load i32, i32* %arrayidx, align 4, !dbg !2805
  %cmp4 = icmp eq i32 %3, %add, !dbg !2808
  br i1 %cmp4, label %if.then, label %if.end, !dbg !2809

if.then:                                          ; preds = %for.body
  call void @llvm.dbg.value(metadata i32 1, metadata !2796, metadata !DIExpression()), !dbg !2797
  br label %for.end, !dbg !2810

if.end:                                           ; preds = %for.body
  br label %for.inc, !dbg !2812

for.inc:                                          ; preds = %if.end
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !2813
  call void @llvm.dbg.value(metadata i32 undef, metadata !2798, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !2778
  br label %for.cond, !dbg !2814, !llvm.loop !2815

for.end.loopexit:                                 ; preds = %for.cond
  br label %for.end, !dbg !2817

for.end:                                          ; preds = %for.end.loopexit, %if.then
  %arrayidx6 = getelementptr inbounds double, double* %v, i64 %indvars.iv1, !dbg !2817
  store double %call, double* %arrayidx6, align 8, !dbg !2818
  %arrayidx8 = getelementptr inbounds i32, i32* %iv, i64 %indvars.iv1, !dbg !2819
  store i32 %add, i32* %arrayidx8, align 4, !dbg !2820
  %indvars.iv.next2 = add nuw nsw i64 %indvars.iv1, 1, !dbg !2821
  call void @llvm.dbg.value(metadata i32 undef, metadata !2783, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !2778
  br label %while.cond, !dbg !2784, !llvm.loop !2822

while.end:                                        ; preds = %while.cond
  ret void, !dbg !2824
}

; Function Attrs: noinline nounwind uwtable
define internal void @_ZL6vecsetiPdPiS0_id(i32 %n, double* %v, i32* %iv, i32* %nzv, i32 %i, double %val) #3 !dbg !2825 {
entry:
  call void @llvm.dbg.value(metadata i32 %n, metadata !2828, metadata !DIExpression()), !dbg !2829
  call void @llvm.dbg.value(metadata double* %v, metadata !2830, metadata !DIExpression()), !dbg !2829
  call void @llvm.dbg.value(metadata i32* %iv, metadata !2831, metadata !DIExpression()), !dbg !2829
  call void @llvm.dbg.value(metadata i32* %nzv, metadata !2832, metadata !DIExpression()), !dbg !2829
  call void @llvm.dbg.value(metadata i32 %i, metadata !2833, metadata !DIExpression()), !dbg !2829
  call void @llvm.dbg.value(metadata double %val, metadata !2834, metadata !DIExpression()), !dbg !2829
  call void @llvm.dbg.value(metadata i32 0, metadata !2835, metadata !DIExpression()), !dbg !2829
  call void @llvm.dbg.value(metadata i32 0, metadata !2836, metadata !DIExpression()), !dbg !2829
  br label %for.cond, !dbg !2837

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ], !dbg !2839
  %set.0 = phi i32 [ 0, %entry ], [ %set.1, %for.inc ], !dbg !2829
  call void @llvm.dbg.value(metadata i32 %set.0, metadata !2835, metadata !DIExpression()), !dbg !2829
  call void @llvm.dbg.value(metadata i64 %indvars.iv, metadata !2836, metadata !DIExpression()), !dbg !2829
  %0 = load i32, i32* %nzv, align 4, !dbg !2840
  %1 = sext i32 %0 to i64, !dbg !2842
  %cmp = icmp slt i64 %indvars.iv, %1, !dbg !2842
  br i1 %cmp, label %for.body, label %for.end, !dbg !2843

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds i32, i32* %iv, i64 %indvars.iv, !dbg !2844
  %2 = load i32, i32* %arrayidx, align 4, !dbg !2844
  %cmp1 = icmp eq i32 %2, %i, !dbg !2847
  br i1 %cmp1, label %if.then, label %if.end, !dbg !2848

if.then:                                          ; preds = %for.body
  %arrayidx3 = getelementptr inbounds double, double* %v, i64 %indvars.iv, !dbg !2849
  store double %val, double* %arrayidx3, align 8, !dbg !2851
  call void @llvm.dbg.value(metadata i32 1, metadata !2835, metadata !DIExpression()), !dbg !2829
  br label %if.end, !dbg !2852

if.end:                                           ; preds = %if.then, %for.body
  %set.1 = phi i32 [ 1, %if.then ], [ %set.0, %for.body ], !dbg !2829
  call void @llvm.dbg.value(metadata i32 %set.1, metadata !2835, metadata !DIExpression()), !dbg !2829
  br label %for.inc, !dbg !2853

for.inc:                                          ; preds = %if.end
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !2854
  call void @llvm.dbg.value(metadata i32 undef, metadata !2836, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !2829
  br label %for.cond, !dbg !2855, !llvm.loop !2856

for.end:                                          ; preds = %for.cond
  %set.0.lcssa = phi i32 [ %set.0, %for.cond ], !dbg !2829
  call void @llvm.dbg.value(metadata i32 %set.0.lcssa, metadata !2835, metadata !DIExpression()), !dbg !2829
  %cmp4 = icmp eq i32 %set.0.lcssa, 0, !dbg !2858
  br i1 %cmp4, label %if.then5, label %if.end10, !dbg !2860

if.then5:                                         ; preds = %for.end
  %3 = load i32, i32* %nzv, align 4, !dbg !2861
  %idxprom6 = sext i32 %3 to i64, !dbg !2863
  %arrayidx7 = getelementptr inbounds double, double* %v, i64 %idxprom6, !dbg !2863
  store double %val, double* %arrayidx7, align 8, !dbg !2864
  %4 = load i32, i32* %nzv, align 4, !dbg !2865
  %idxprom8 = sext i32 %4 to i64, !dbg !2866
  %arrayidx9 = getelementptr inbounds i32, i32* %iv, i64 %idxprom8, !dbg !2866
  store i32 %i, i32* %arrayidx9, align 4, !dbg !2867
  %5 = load i32, i32* %nzv, align 4, !dbg !2868
  %add = add nsw i32 %5, 1, !dbg !2869
  store i32 %add, i32* %nzv, align 4, !dbg !2870
  br label %if.end10, !dbg !2871

if.end10:                                         ; preds = %if.then5, %for.end
  ret void, !dbg !2872
}

; Function Attrs: noinline uwtable
define internal void @_ZL6sparsePdPiS0_iiiS0_PA12_iPA12_diiS0_dd(double* %a, i32* %colidx, i32* %rowstr, i32 %n, i32 %nz, i32 %nozer, i32* %arow, [12 x i32]* %acol, [12 x double]* %aelt, i32 %firstrow, i32 %lastrow, i32* %nzloc, double %rcond, double %shift) #4 !dbg !2873 {
entry:
  call void @llvm.dbg.value(metadata double* %a, metadata !2876, metadata !DIExpression()), !dbg !2877
  call void @llvm.dbg.value(metadata i32* %colidx, metadata !2878, metadata !DIExpression()), !dbg !2877
  call void @llvm.dbg.value(metadata i32* %rowstr, metadata !2879, metadata !DIExpression()), !dbg !2877
  call void @llvm.dbg.value(metadata i32 %n, metadata !2880, metadata !DIExpression()), !dbg !2877
  call void @llvm.dbg.value(metadata i32 %nz, metadata !2881, metadata !DIExpression()), !dbg !2877
  call void @llvm.dbg.value(metadata i32 %nozer, metadata !2882, metadata !DIExpression()), !dbg !2877
  call void @llvm.dbg.value(metadata i32* %arow, metadata !2883, metadata !DIExpression()), !dbg !2877
  call void @llvm.dbg.value(metadata [12 x i32]* %acol, metadata !2884, metadata !DIExpression()), !dbg !2877
  call void @llvm.dbg.value(metadata [12 x double]* %aelt, metadata !2885, metadata !DIExpression()), !dbg !2877
  call void @llvm.dbg.value(metadata i32 %firstrow, metadata !2886, metadata !DIExpression()), !dbg !2877
  call void @llvm.dbg.value(metadata i32 %lastrow, metadata !2887, metadata !DIExpression()), !dbg !2877
  call void @llvm.dbg.value(metadata i32* %nzloc, metadata !2888, metadata !DIExpression()), !dbg !2877
  call void @llvm.dbg.value(metadata double %rcond, metadata !2889, metadata !DIExpression()), !dbg !2877
  call void @llvm.dbg.value(metadata double %shift, metadata !2890, metadata !DIExpression()), !dbg !2877
  %sub = sub nsw i32 %lastrow, %firstrow, !dbg !2891
  %add = add nsw i32 %sub, 1, !dbg !2892
  call void @llvm.dbg.value(metadata i32 %add, metadata !2893, metadata !DIExpression()), !dbg !2877
  call void @llvm.dbg.value(metadata i32 0, metadata !2894, metadata !DIExpression()), !dbg !2877
  br label %for.cond, !dbg !2895

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv41 = phi i64 [ %indvars.iv.next42, %for.inc ], [ 0, %entry ], !dbg !2897
  call void @llvm.dbg.value(metadata i64 %indvars.iv41, metadata !2894, metadata !DIExpression()), !dbg !2877
  %add1 = add nsw i32 %add, 1, !dbg !2898
  %0 = sext i32 %add1 to i64, !dbg !2900
  %cmp = icmp slt i64 %indvars.iv41, %0, !dbg !2900
  br i1 %cmp, label %for.body, label %for.end, !dbg !2901

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds i32, i32* %rowstr, i64 %indvars.iv41, !dbg !2902
  store i32 0, i32* %arrayidx, align 4, !dbg !2904
  br label %for.inc, !dbg !2905

for.inc:                                          ; preds = %for.body
  %indvars.iv.next42 = add nuw nsw i64 %indvars.iv41, 1, !dbg !2906
  call void @llvm.dbg.value(metadata i32 undef, metadata !2894, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !2877
  br label %for.cond, !dbg !2907, !llvm.loop !2908

for.end:                                          ; preds = %for.cond
  call void @llvm.dbg.value(metadata i32 0, metadata !2910, metadata !DIExpression()), !dbg !2877
  %1 = sext i32 %n to i64, !dbg !2911
  br label %for.cond2, !dbg !2911

for.cond2:                                        ; preds = %for.inc25, %for.end
  %indvars.iv39 = phi i64 [ %indvars.iv.next40, %for.inc25 ], [ 0, %for.end ], !dbg !2913
  call void @llvm.dbg.value(metadata i64 %indvars.iv39, metadata !2910, metadata !DIExpression()), !dbg !2877
  %cmp3 = icmp slt i64 %indvars.iv39, %1, !dbg !2914
  br i1 %cmp3, label %for.body4, label %for.end27, !dbg !2916

for.body4:                                        ; preds = %for.cond2
  call void @llvm.dbg.value(metadata i32 0, metadata !2917, metadata !DIExpression()), !dbg !2877
  br label %for.cond5, !dbg !2918

for.cond5:                                        ; preds = %for.inc22, %for.body4
  %indvars.iv37 = phi i64 [ %indvars.iv.next38, %for.inc22 ], [ 0, %for.body4 ], !dbg !2921
  call void @llvm.dbg.value(metadata i64 %indvars.iv37, metadata !2917, metadata !DIExpression()), !dbg !2877
  %arrayidx7 = getelementptr inbounds i32, i32* %arow, i64 %indvars.iv39, !dbg !2922
  %2 = load i32, i32* %arrayidx7, align 4, !dbg !2922
  %3 = sext i32 %2 to i64, !dbg !2924
  %cmp8 = icmp slt i64 %indvars.iv37, %3, !dbg !2924
  br i1 %cmp8, label %for.body9, label %for.end24, !dbg !2925

for.body9:                                        ; preds = %for.cond5
  %arrayidx11 = getelementptr inbounds [12 x i32], [12 x i32]* %acol, i64 %indvars.iv39, !dbg !2926
  %arrayidx13 = getelementptr inbounds [12 x i32], [12 x i32]* %arrayidx11, i64 0, i64 %indvars.iv37, !dbg !2926
  %4 = load i32, i32* %arrayidx13, align 4, !dbg !2926
  %add14 = add nsw i32 %4, 1, !dbg !2928
  call void @llvm.dbg.value(metadata i32 %add14, metadata !2894, metadata !DIExpression()), !dbg !2877
  %idxprom15 = sext i32 %add14 to i64, !dbg !2929
  %arrayidx16 = getelementptr inbounds i32, i32* %rowstr, i64 %idxprom15, !dbg !2929
  %5 = load i32, i32* %arrayidx16, align 4, !dbg !2929
  %arrayidx18 = getelementptr inbounds i32, i32* %arow, i64 %indvars.iv39, !dbg !2930
  %6 = load i32, i32* %arrayidx18, align 4, !dbg !2930
  %add19 = add nsw i32 %5, %6, !dbg !2931
  %idxprom20 = sext i32 %add14 to i64, !dbg !2932
  %arrayidx21 = getelementptr inbounds i32, i32* %rowstr, i64 %idxprom20, !dbg !2932
  store i32 %add19, i32* %arrayidx21, align 4, !dbg !2933
  br label %for.inc22, !dbg !2934

for.inc22:                                        ; preds = %for.body9
  %indvars.iv.next38 = add nuw nsw i64 %indvars.iv37, 1, !dbg !2935
  call void @llvm.dbg.value(metadata i32 undef, metadata !2917, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !2877
  br label %for.cond5, !dbg !2936, !llvm.loop !2937

for.end24:                                        ; preds = %for.cond5
  br label %for.inc25, !dbg !2939

for.inc25:                                        ; preds = %for.end24
  %indvars.iv.next40 = add nuw nsw i64 %indvars.iv39, 1, !dbg !2940
  call void @llvm.dbg.value(metadata i32 undef, metadata !2910, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !2877
  br label %for.cond2, !dbg !2941, !llvm.loop !2942

for.end27:                                        ; preds = %for.cond2
  %arrayidx28 = getelementptr inbounds i32, i32* %rowstr, i64 0, !dbg !2944
  store i32 0, i32* %arrayidx28, align 4, !dbg !2945
  call void @llvm.dbg.value(metadata i32 1, metadata !2894, metadata !DIExpression()), !dbg !2877
  br label %for.cond29, !dbg !2946

for.cond29:                                       ; preds = %for.inc41, %for.end27
  %indvars.iv34 = phi i64 [ %indvars.iv.next35, %for.inc41 ], [ 1, %for.end27 ], !dbg !2948
  call void @llvm.dbg.value(metadata i64 %indvars.iv34, metadata !2894, metadata !DIExpression()), !dbg !2877
  %add30 = add nsw i32 %add, 1, !dbg !2949
  %7 = sext i32 %add30 to i64, !dbg !2951
  %cmp31 = icmp slt i64 %indvars.iv34, %7, !dbg !2951
  br i1 %cmp31, label %for.body32, label %for.end43, !dbg !2952

for.body32:                                       ; preds = %for.cond29
  %arrayidx34 = getelementptr inbounds i32, i32* %rowstr, i64 %indvars.iv34, !dbg !2953
  %8 = load i32, i32* %arrayidx34, align 4, !dbg !2953
  %9 = sub nuw nsw i64 %indvars.iv34, 1, !dbg !2955
  %arrayidx37 = getelementptr inbounds i32, i32* %rowstr, i64 %9, !dbg !2956
  %10 = load i32, i32* %arrayidx37, align 4, !dbg !2956
  %add38 = add nsw i32 %8, %10, !dbg !2957
  %arrayidx40 = getelementptr inbounds i32, i32* %rowstr, i64 %indvars.iv34, !dbg !2958
  store i32 %add38, i32* %arrayidx40, align 4, !dbg !2959
  br label %for.inc41, !dbg !2960

for.inc41:                                        ; preds = %for.body32
  %indvars.iv.next35 = add nuw nsw i64 %indvars.iv34, 1, !dbg !2961
  call void @llvm.dbg.value(metadata i32 undef, metadata !2894, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !2877
  br label %for.cond29, !dbg !2962, !llvm.loop !2963

for.end43:                                        ; preds = %for.cond29
  %idxprom44 = sext i32 %add to i64, !dbg !2965
  %arrayidx45 = getelementptr inbounds i32, i32* %rowstr, i64 %idxprom44, !dbg !2965
  %11 = load i32, i32* %arrayidx45, align 4, !dbg !2965
  %sub46 = sub nsw i32 %11, 1, !dbg !2966
  call void @llvm.dbg.value(metadata i32 %sub46, metadata !2917, metadata !DIExpression()), !dbg !2877
  %cmp47 = icmp sgt i32 %sub46, %nz, !dbg !2967
  br i1 %cmp47, label %if.then, label %if.end, !dbg !2969

if.then:                                          ; preds = %for.end43
  %call = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([46 x i8], [46 x i8]* @.str.81, i64 0, i64 0)), !dbg !2970
  %call48 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([21 x i8], [21 x i8]* @.str.82, i64 0, i64 0), i32 %sub46, i32 %nz), !dbg !2972
  call void @exit(i32 1) #12, !dbg !2973
  unreachable, !dbg !2973

if.end:                                           ; preds = %for.end43
  call void @llvm.dbg.value(metadata i32 0, metadata !2894, metadata !DIExpression()), !dbg !2877
  %12 = sext i32 %add to i64, !dbg !2974
  br label %for.cond49, !dbg !2974

for.cond49:                                       ; preds = %for.inc69, %if.end
  %indvars.iv31 = phi i64 [ %indvars.iv.next32, %for.inc69 ], [ 0, %if.end ], !dbg !2976
  call void @llvm.dbg.value(metadata i64 %indvars.iv31, metadata !2894, metadata !DIExpression()), !dbg !2877
  %cmp50 = icmp slt i64 %indvars.iv31, %12, !dbg !2977
  br i1 %cmp50, label %for.body51, label %for.end71, !dbg !2979

for.body51:                                       ; preds = %for.cond49
  %arrayidx53 = getelementptr inbounds i32, i32* %rowstr, i64 %indvars.iv31, !dbg !2980
  %13 = load i32, i32* %arrayidx53, align 4, !dbg !2980
  call void @llvm.dbg.value(metadata i32 %13, metadata !2983, metadata !DIExpression()), !dbg !2877
  %14 = sext i32 %13 to i64, !dbg !2984
  br label %for.cond54, !dbg !2984

for.cond54:                                       ; preds = %for.inc64, %for.body51
  %indvars.iv29 = phi i64 [ %indvars.iv.next30, %for.inc64 ], [ %14, %for.body51 ], !dbg !2985
  call void @llvm.dbg.value(metadata i64 %indvars.iv29, metadata !2983, metadata !DIExpression()), !dbg !2877
  %15 = add nuw nsw i64 %indvars.iv31, 1, !dbg !2986
  %arrayidx57 = getelementptr inbounds i32, i32* %rowstr, i64 %15, !dbg !2988
  %16 = load i32, i32* %arrayidx57, align 4, !dbg !2988
  %17 = sext i32 %16 to i64, !dbg !2989
  %cmp58 = icmp slt i64 %indvars.iv29, %17, !dbg !2989
  br i1 %cmp58, label %for.body59, label %for.end66, !dbg !2990

for.body59:                                       ; preds = %for.cond54
  %arrayidx61 = getelementptr inbounds double, double* %a, i64 %indvars.iv29, !dbg !2991
  store double 0.000000e+00, double* %arrayidx61, align 8, !dbg !2993
  %arrayidx63 = getelementptr inbounds i32, i32* %colidx, i64 %indvars.iv29, !dbg !2994
  store i32 -1, i32* %arrayidx63, align 4, !dbg !2995
  br label %for.inc64, !dbg !2996

for.inc64:                                        ; preds = %for.body59
  %indvars.iv.next30 = add nsw i64 %indvars.iv29, 1, !dbg !2997
  call void @llvm.dbg.value(metadata i32 undef, metadata !2983, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !2877
  br label %for.cond54, !dbg !2998, !llvm.loop !2999

for.end66:                                        ; preds = %for.cond54
  %arrayidx68 = getelementptr inbounds i32, i32* %nzloc, i64 %indvars.iv31, !dbg !3001
  store i32 0, i32* %arrayidx68, align 4, !dbg !3002
  br label %for.inc69, !dbg !3003

for.inc69:                                        ; preds = %for.end66
  %indvars.iv.next32 = add nuw nsw i64 %indvars.iv31, 1, !dbg !3004
  call void @llvm.dbg.value(metadata i32 undef, metadata !2894, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !2877
  br label %for.cond49, !dbg !3005, !llvm.loop !3006

for.end71:                                        ; preds = %for.cond49
  call void @llvm.dbg.value(metadata double 1.000000e+00, metadata !3008, metadata !DIExpression()), !dbg !2877
  %conv = sitofp i32 %n to double, !dbg !3009
  %div = fdiv double 1.000000e+00, %conv, !dbg !3010
  %call72 = call double @pow(double %rcond, double %div) #11, !dbg !3011
  call void @llvm.dbg.value(metadata double %call72, metadata !3012, metadata !DIExpression()), !dbg !2877
  call void @llvm.dbg.value(metadata i32 0, metadata !2910, metadata !DIExpression()), !dbg !2877
  %18 = sext i32 %n to i64, !dbg !3013
  br label %for.cond73, !dbg !3013

for.cond73:                                       ; preds = %for.inc187, %for.end71
  %indvars.iv27 = phi i64 [ %indvars.iv.next28, %for.inc187 ], [ 0, %for.end71 ], !dbg !3015
  %size.0 = phi double [ 1.000000e+00, %for.end71 ], [ %mul186, %for.inc187 ], !dbg !2877
  call void @llvm.dbg.value(metadata double %size.0, metadata !3008, metadata !DIExpression()), !dbg !2877
  call void @llvm.dbg.value(metadata i64 %indvars.iv27, metadata !2910, metadata !DIExpression()), !dbg !2877
  %cmp74 = icmp slt i64 %indvars.iv27, %18, !dbg !3016
  br i1 %cmp74, label %for.body75, label %for.end189, !dbg !3018

for.body75:                                       ; preds = %for.cond73
  call void @llvm.dbg.value(metadata i32 0, metadata !2917, metadata !DIExpression()), !dbg !2877
  br label %for.cond76, !dbg !3019

for.cond76:                                       ; preds = %for.inc183, %for.body75
  %indvars.iv25 = phi i64 [ %indvars.iv.next26, %for.inc183 ], [ 0, %for.body75 ], !dbg !3022
  call void @llvm.dbg.value(metadata i64 %indvars.iv25, metadata !2917, metadata !DIExpression()), !dbg !2877
  %arrayidx78 = getelementptr inbounds i32, i32* %arow, i64 %indvars.iv27, !dbg !3023
  %19 = load i32, i32* %arrayidx78, align 4, !dbg !3023
  %20 = sext i32 %19 to i64, !dbg !3025
  %cmp79 = icmp slt i64 %indvars.iv25, %20, !dbg !3025
  br i1 %cmp79, label %for.body80, label %for.end185, !dbg !3026

for.body80:                                       ; preds = %for.cond76
  %arrayidx82 = getelementptr inbounds [12 x i32], [12 x i32]* %acol, i64 %indvars.iv27, !dbg !3027
  %arrayidx84 = getelementptr inbounds [12 x i32], [12 x i32]* %arrayidx82, i64 0, i64 %indvars.iv25, !dbg !3027
  %21 = load i32, i32* %arrayidx84, align 4, !dbg !3027
  call void @llvm.dbg.value(metadata i32 %21, metadata !2894, metadata !DIExpression()), !dbg !2877
  %arrayidx86 = getelementptr inbounds [12 x double], [12 x double]* %aelt, i64 %indvars.iv27, !dbg !3029
  %arrayidx88 = getelementptr inbounds [12 x double], [12 x double]* %arrayidx86, i64 0, i64 %indvars.iv25, !dbg !3029
  %22 = load double, double* %arrayidx88, align 8, !dbg !3029
  %mul = fmul contract double %size.0, %22, !dbg !3030
  call void @llvm.dbg.value(metadata double %mul, metadata !3031, metadata !DIExpression()), !dbg !2877
  call void @llvm.dbg.value(metadata i32 0, metadata !3032, metadata !DIExpression()), !dbg !2877
  %23 = zext i32 %21 to i64, !dbg !3033
  br label %for.cond89, !dbg !3033

for.cond89:                                       ; preds = %for.inc180, %for.body80
  %indvars.iv23 = phi i64 [ %indvars.iv.next24, %for.inc180 ], [ 0, %for.body80 ], !dbg !3035
  call void @llvm.dbg.value(metadata i64 %indvars.iv23, metadata !3032, metadata !DIExpression()), !dbg !2877
  %arrayidx91 = getelementptr inbounds i32, i32* %arow, i64 %indvars.iv27, !dbg !3036
  %24 = load i32, i32* %arrayidx91, align 4, !dbg !3036
  %25 = sext i32 %24 to i64, !dbg !3038
  %cmp92 = icmp slt i64 %indvars.iv23, %25, !dbg !3038
  br i1 %cmp92, label %for.body93, label %for.end182, !dbg !3039

for.body93:                                       ; preds = %for.cond89
  %arrayidx95 = getelementptr inbounds [12 x i32], [12 x i32]* %acol, i64 %indvars.iv27, !dbg !3040
  %arrayidx97 = getelementptr inbounds [12 x i32], [12 x i32]* %arrayidx95, i64 0, i64 %indvars.iv23, !dbg !3040
  %26 = load i32, i32* %arrayidx97, align 4, !dbg !3040
  call void @llvm.dbg.value(metadata i32 %26, metadata !3042, metadata !DIExpression()), !dbg !2877
  %arrayidx99 = getelementptr inbounds [12 x double], [12 x double]* %aelt, i64 %indvars.iv27, !dbg !3043
  %arrayidx101 = getelementptr inbounds [12 x double], [12 x double]* %arrayidx99, i64 0, i64 %indvars.iv23, !dbg !3043
  %27 = load double, double* %arrayidx101, align 8, !dbg !3043
  %mul102 = fmul contract double %27, %mul, !dbg !3044
  call void @llvm.dbg.value(metadata double %mul102, metadata !3045, metadata !DIExpression()), !dbg !2877
  %cmp103 = icmp eq i32 %26, %21, !dbg !3046
  br i1 %cmp103, label %land.lhs.true, label %if.end108, !dbg !3048

land.lhs.true:                                    ; preds = %for.body93
  %cmp104 = icmp eq i64 %23, %indvars.iv27, !dbg !3049
  br i1 %cmp104, label %if.then105, label %if.end108, !dbg !3050

if.then105:                                       ; preds = %land.lhs.true
  %add106 = fadd contract double %mul102, %rcond, !dbg !3051
  %sub107 = fsub contract double %add106, %shift, !dbg !3053
  call void @llvm.dbg.value(metadata double %sub107, metadata !3045, metadata !DIExpression()), !dbg !2877
  br label %if.end108, !dbg !3054

if.end108:                                        ; preds = %if.then105, %land.lhs.true, %for.body93
  %va.0 = phi double [ %sub107, %if.then105 ], [ %mul102, %land.lhs.true ], [ %mul102, %for.body93 ], !dbg !3055
  call void @llvm.dbg.value(metadata double %va.0, metadata !3045, metadata !DIExpression()), !dbg !2877
  call void @llvm.dbg.value(metadata i32 0, metadata !3056, metadata !DIExpression()), !dbg !2877
  %idxprom109 = sext i32 %21 to i64, !dbg !3057
  %arrayidx110 = getelementptr inbounds i32, i32* %rowstr, i64 %idxprom109, !dbg !3057
  %28 = load i32, i32* %arrayidx110, align 4, !dbg !3057
  call void @llvm.dbg.value(metadata i32 %28, metadata !2983, metadata !DIExpression()), !dbg !2877
  %29 = sext i32 %28 to i64, !dbg !3059
  br label %for.cond111, !dbg !3059

for.cond111:                                      ; preds = %for.inc168, %if.end108
  %indvars.iv17 = phi i64 [ %indvars.iv.next18, %for.inc168 ], [ %29, %if.end108 ], !dbg !3060
  call void @llvm.dbg.value(metadata i64 %indvars.iv17, metadata !2983, metadata !DIExpression()), !dbg !2877
  %add112 = add nsw i32 %21, 1, !dbg !3061
  %idxprom113 = sext i32 %add112 to i64, !dbg !3063
  %arrayidx114 = getelementptr inbounds i32, i32* %rowstr, i64 %idxprom113, !dbg !3063
  %30 = load i32, i32* %arrayidx114, align 4, !dbg !3063
  %31 = sext i32 %30 to i64, !dbg !3064
  %cmp115 = icmp slt i64 %indvars.iv17, %31, !dbg !3064
  br i1 %cmp115, label %for.body116, label %for.end170.loopexit, !dbg !3065

for.body116:                                      ; preds = %for.cond111
  %arrayidx118 = getelementptr inbounds i32, i32* %colidx, i64 %indvars.iv17, !dbg !3066
  %32 = load i32, i32* %arrayidx118, align 4, !dbg !3066
  %cmp119 = icmp sgt i32 %32, %26, !dbg !3069
  br i1 %cmp119, label %if.then120, label %if.else, !dbg !3070

if.then120:                                       ; preds = %for.body116
  %k.1.lcssa1.wide = phi i64 [ %indvars.iv17, %for.body116 ]
  %33 = trunc i64 %k.1.lcssa1.wide to i32, !dbg !2877
  call void @llvm.dbg.value(metadata i32 %33, metadata !2983, metadata !DIExpression()), !dbg !2877
  %add121 = add nsw i32 %21, 1, !dbg !3071
  %idxprom122 = sext i32 %add121 to i64, !dbg !3074
  %arrayidx123 = getelementptr inbounds i32, i32* %rowstr, i64 %idxprom122, !dbg !3074
  %34 = load i32, i32* %arrayidx123, align 4, !dbg !3074
  call void @llvm.dbg.value(metadata i32 %34, metadata !3075, metadata !DIExpression(DW_OP_constu, 2, DW_OP_minus, DW_OP_stack_value)), !dbg !2877
  %35 = add i32 %34, -2, !dbg !3076
  %36 = sext i32 %35 to i64, !dbg !3076
  %37 = sext i32 %33 to i64, !dbg !3076
  br label %for.cond125, !dbg !3076

for.cond125:                                      ; preds = %for.inc143, %if.then120
  %indvars.iv19 = phi i64 [ %indvars.iv.next20, %for.inc143 ], [ %36, %if.then120 ], !dbg !3077
  call void @llvm.dbg.value(metadata i64 %indvars.iv19, metadata !3075, metadata !DIExpression()), !dbg !2877
  %cmp126 = icmp sge i64 %indvars.iv19, %37, !dbg !3078
  br i1 %cmp126, label %for.body127, label %for.end144, !dbg !3080

for.body127:                                      ; preds = %for.cond125
  %arrayidx129 = getelementptr inbounds i32, i32* %colidx, i64 %indvars.iv19, !dbg !3081
  %38 = load i32, i32* %arrayidx129, align 4, !dbg !3081
  %cmp130 = icmp sgt i32 %38, -1, !dbg !3084
  br i1 %cmp130, label %if.then131, label %if.end142, !dbg !3085

if.then131:                                       ; preds = %for.body127
  %arrayidx133 = getelementptr inbounds double, double* %a, i64 %indvars.iv19, !dbg !3086
  %39 = load double, double* %arrayidx133, align 8, !dbg !3086
  %40 = add nsw i64 %indvars.iv19, 1, !dbg !3088
  %arrayidx136 = getelementptr inbounds double, double* %a, i64 %40, !dbg !3089
  store double %39, double* %arrayidx136, align 8, !dbg !3090
  %arrayidx138 = getelementptr inbounds i32, i32* %colidx, i64 %indvars.iv19, !dbg !3091
  %41 = load i32, i32* %arrayidx138, align 4, !dbg !3091
  %42 = add nsw i64 %indvars.iv19, 1, !dbg !3092
  %arrayidx141 = getelementptr inbounds i32, i32* %colidx, i64 %42, !dbg !3093
  store i32 %41, i32* %arrayidx141, align 4, !dbg !3094
  br label %if.end142, !dbg !3095

if.end142:                                        ; preds = %if.then131, %for.body127
  br label %for.inc143, !dbg !3096

for.inc143:                                       ; preds = %if.end142
  %indvars.iv.next20 = add i64 %indvars.iv19, -1, !dbg !3097
  call void @llvm.dbg.value(metadata i32 undef, metadata !3075, metadata !DIExpression(DW_OP_constu, 1, DW_OP_minus, DW_OP_stack_value)), !dbg !2877
  br label %for.cond125, !dbg !3098, !llvm.loop !3099

for.end144:                                       ; preds = %for.cond125
  %idxprom145 = sext i32 %33 to i64, !dbg !3101
  %arrayidx146 = getelementptr inbounds i32, i32* %colidx, i64 %idxprom145, !dbg !3101
  store i32 %26, i32* %arrayidx146, align 4, !dbg !3102
  %idxprom147 = sext i32 %33 to i64, !dbg !3103
  %arrayidx148 = getelementptr inbounds double, double* %a, i64 %idxprom147, !dbg !3103
  store double 0.000000e+00, double* %arrayidx148, align 8, !dbg !3104
  call void @llvm.dbg.value(metadata i32 1, metadata !3056, metadata !DIExpression()), !dbg !2877
  br label %for.end170, !dbg !3105

if.else:                                          ; preds = %for.body116
  %arrayidx150 = getelementptr inbounds i32, i32* %colidx, i64 %indvars.iv17, !dbg !3106
  %43 = load i32, i32* %arrayidx150, align 4, !dbg !3106
  %cmp151 = icmp eq i32 %43, -1, !dbg !3108
  br i1 %cmp151, label %if.then152, label %if.else155, !dbg !3109

if.then152:                                       ; preds = %if.else
  %k.1.lcssa2.wide = phi i64 [ %indvars.iv17, %if.else ]
  %44 = trunc i64 %k.1.lcssa2.wide to i32, !dbg !2877
  call void @llvm.dbg.value(metadata i32 %44, metadata !2983, metadata !DIExpression()), !dbg !2877
  %idxprom153 = sext i32 %44 to i64, !dbg !3110
  %arrayidx154 = getelementptr inbounds i32, i32* %colidx, i64 %idxprom153, !dbg !3110
  store i32 %26, i32* %arrayidx154, align 4, !dbg !3112
  call void @llvm.dbg.value(metadata i32 1, metadata !3056, metadata !DIExpression()), !dbg !2877
  br label %for.end170, !dbg !3113

if.else155:                                       ; preds = %if.else
  %arrayidx157 = getelementptr inbounds i32, i32* %colidx, i64 %indvars.iv17, !dbg !3114
  %45 = load i32, i32* %arrayidx157, align 4, !dbg !3114
  %cmp158 = icmp eq i32 %45, %26, !dbg !3116
  br i1 %cmp158, label %if.then159, label %if.end165, !dbg !3117

if.then159:                                       ; preds = %if.else155
  %k.1.lcssa3.wide = phi i64 [ %indvars.iv17, %if.else155 ]
  %46 = trunc i64 %k.1.lcssa3.wide to i32, !dbg !2877
  call void @llvm.dbg.value(metadata i32 %46, metadata !2983, metadata !DIExpression()), !dbg !2877
  %idxprom160 = sext i32 %21 to i64, !dbg !3118
  %arrayidx161 = getelementptr inbounds i32, i32* %nzloc, i64 %idxprom160, !dbg !3118
  %47 = load i32, i32* %arrayidx161, align 4, !dbg !3118
  %add162 = add nsw i32 %47, 1, !dbg !3120
  %idxprom163 = sext i32 %21 to i64, !dbg !3121
  %arrayidx164 = getelementptr inbounds i32, i32* %nzloc, i64 %idxprom163, !dbg !3121
  store i32 %add162, i32* %arrayidx164, align 4, !dbg !3122
  call void @llvm.dbg.value(metadata i32 1, metadata !3056, metadata !DIExpression()), !dbg !2877
  br label %for.end170, !dbg !3123

if.end165:                                        ; preds = %if.else155
  br label %if.end166

if.end166:                                        ; preds = %if.end165
  br label %if.end167

if.end167:                                        ; preds = %if.end166
  br label %for.inc168, !dbg !3124

for.inc168:                                       ; preds = %if.end167
  %indvars.iv.next18 = add nsw i64 %indvars.iv17, 1, !dbg !3125
  call void @llvm.dbg.value(metadata i32 undef, metadata !2983, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !2877
  br label %for.cond111, !dbg !3126, !llvm.loop !3127

for.end170.loopexit:                              ; preds = %for.cond111
  %k.1.lcssa.wide = phi i64 [ %indvars.iv17, %for.cond111 ]
  %48 = trunc i64 %k.1.lcssa.wide to i32, !dbg !2877
  call void @llvm.dbg.value(metadata i32 %48, metadata !2983, metadata !DIExpression()), !dbg !2877
  br label %for.end170, !dbg !3129

for.end170:                                       ; preds = %for.end170.loopexit, %if.then159, %if.then152, %for.end144
  %k.14 = phi i32 [ %33, %for.end144 ], [ %44, %if.then152 ], [ %46, %if.then159 ], [ %48, %for.end170.loopexit ]
  %goto_40.0 = phi i32 [ 1, %for.end144 ], [ 1, %if.then152 ], [ 1, %if.then159 ], [ 0, %for.end170.loopexit ], !dbg !3055
  call void @llvm.dbg.value(metadata i32 %goto_40.0, metadata !3056, metadata !DIExpression()), !dbg !2877
  %cmp171 = icmp eq i32 %goto_40.0, 0, !dbg !3129
  br i1 %cmp171, label %if.then172, label %if.end174, !dbg !3131

if.then172:                                       ; preds = %for.end170
  %i.1.lcssa5.wide = phi i64 [ %indvars.iv27, %for.end170 ]
  %49 = trunc i64 %i.1.lcssa5.wide to i32, !dbg !2877
  call void @llvm.dbg.value(metadata i32 %49, metadata !2910, metadata !DIExpression()), !dbg !2877
  %call173 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([32 x i8], [32 x i8]* @.str.83, i64 0, i64 0), i32 %49), !dbg !3132
  call void @exit(i32 1) #12, !dbg !3134
  unreachable, !dbg !3134

if.end174:                                        ; preds = %for.end170
  %idxprom175 = sext i32 %k.14 to i64, !dbg !3135
  %arrayidx176 = getelementptr inbounds double, double* %a, i64 %idxprom175, !dbg !3135
  %50 = load double, double* %arrayidx176, align 8, !dbg !3135
  %add177 = fadd contract double %50, %va.0, !dbg !3136
  %idxprom178 = sext i32 %k.14 to i64, !dbg !3137
  %arrayidx179 = getelementptr inbounds double, double* %a, i64 %idxprom178, !dbg !3137
  store double %add177, double* %arrayidx179, align 8, !dbg !3138
  br label %for.inc180, !dbg !3139

for.inc180:                                       ; preds = %if.end174
  %indvars.iv.next24 = add nuw nsw i64 %indvars.iv23, 1, !dbg !3140
  call void @llvm.dbg.value(metadata i32 undef, metadata !3032, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !2877
  br label %for.cond89, !dbg !3141, !llvm.loop !3142

for.end182:                                       ; preds = %for.cond89
  br label %for.inc183, !dbg !3144

for.inc183:                                       ; preds = %for.end182
  %indvars.iv.next26 = add nuw nsw i64 %indvars.iv25, 1, !dbg !3145
  call void @llvm.dbg.value(metadata i32 undef, metadata !2917, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !2877
  br label %for.cond76, !dbg !3146, !llvm.loop !3147

for.end185:                                       ; preds = %for.cond76
  %mul186 = fmul contract double %size.0, %call72, !dbg !3149
  call void @llvm.dbg.value(metadata double %mul186, metadata !3008, metadata !DIExpression()), !dbg !2877
  br label %for.inc187, !dbg !3150

for.inc187:                                       ; preds = %for.end185
  %indvars.iv.next28 = add nuw nsw i64 %indvars.iv27, 1, !dbg !3151
  call void @llvm.dbg.value(metadata i32 undef, metadata !2910, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !2877
  br label %for.cond73, !dbg !3152, !llvm.loop !3153

for.end189:                                       ; preds = %for.cond73
  call void @llvm.dbg.value(metadata i32 1, metadata !2894, metadata !DIExpression()), !dbg !2877
  %51 = sext i32 %add to i64, !dbg !3155
  br label %for.cond190, !dbg !3155

for.cond190:                                      ; preds = %for.inc201, %for.end189
  %indvars.iv14 = phi i64 [ %indvars.iv.next15, %for.inc201 ], [ 1, %for.end189 ], !dbg !3157
  call void @llvm.dbg.value(metadata i64 %indvars.iv14, metadata !2894, metadata !DIExpression()), !dbg !2877
  %cmp191 = icmp slt i64 %indvars.iv14, %51, !dbg !3158
  br i1 %cmp191, label %for.body192, label %for.end203, !dbg !3160

for.body192:                                      ; preds = %for.cond190
  %arrayidx194 = getelementptr inbounds i32, i32* %nzloc, i64 %indvars.iv14, !dbg !3161
  %52 = load i32, i32* %arrayidx194, align 4, !dbg !3161
  %53 = sub nuw nsw i64 %indvars.iv14, 1, !dbg !3163
  %arrayidx197 = getelementptr inbounds i32, i32* %nzloc, i64 %53, !dbg !3164
  %54 = load i32, i32* %arrayidx197, align 4, !dbg !3164
  %add198 = add nsw i32 %52, %54, !dbg !3165
  %arrayidx200 = getelementptr inbounds i32, i32* %nzloc, i64 %indvars.iv14, !dbg !3166
  store i32 %add198, i32* %arrayidx200, align 4, !dbg !3167
  br label %for.inc201, !dbg !3168

for.inc201:                                       ; preds = %for.body192
  %indvars.iv.next15 = add nuw nsw i64 %indvars.iv14, 1, !dbg !3169
  call void @llvm.dbg.value(metadata i32 undef, metadata !2894, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !2877
  br label %for.cond190, !dbg !3170, !llvm.loop !3171

for.end203:                                       ; preds = %for.cond190
  call void @llvm.dbg.value(metadata i32 0, metadata !2894, metadata !DIExpression()), !dbg !2877
  %55 = sext i32 %add to i64, !dbg !3173
  br label %for.cond204, !dbg !3173

for.cond204:                                      ; preds = %for.inc240, %for.end203
  %indvars.iv11 = phi i64 [ %indvars.iv.next12, %for.inc240 ], [ 0, %for.end203 ], !dbg !3175
  call void @llvm.dbg.value(metadata i64 %indvars.iv11, metadata !2894, metadata !DIExpression()), !dbg !2877
  %cmp205 = icmp slt i64 %indvars.iv11, %55, !dbg !3176
  br i1 %cmp205, label %for.body206, label %for.end242, !dbg !3178

for.body206:                                      ; preds = %for.cond204
  %cmp207 = icmp ugt i64 %indvars.iv11, 0, !dbg !3179
  br i1 %cmp207, label %if.then208, label %if.else215, !dbg !3182

if.then208:                                       ; preds = %for.body206
  %arrayidx210 = getelementptr inbounds i32, i32* %rowstr, i64 %indvars.iv11, !dbg !3183
  %56 = load i32, i32* %arrayidx210, align 4, !dbg !3183
  %57 = sub nsw i64 %indvars.iv11, 1, !dbg !3185
  %arrayidx213 = getelementptr inbounds i32, i32* %nzloc, i64 %57, !dbg !3186
  %58 = load i32, i32* %arrayidx213, align 4, !dbg !3186
  %sub214 = sub nsw i32 %56, %58, !dbg !3187
  call void @llvm.dbg.value(metadata i32 %sub214, metadata !3188, metadata !DIExpression()), !dbg !2877
  br label %if.end216, !dbg !3189

if.else215:                                       ; preds = %for.body206
  call void @llvm.dbg.value(metadata i32 0, metadata !3188, metadata !DIExpression()), !dbg !2877
  br label %if.end216

if.end216:                                        ; preds = %if.else215, %if.then208
  %j1.0 = phi i32 [ %sub214, %if.then208 ], [ 0, %if.else215 ], !dbg !3190
  call void @llvm.dbg.value(metadata i32 %j1.0, metadata !3188, metadata !DIExpression()), !dbg !2877
  %indvars.iv.next12 = add nuw nsw i64 %indvars.iv11, 1, !dbg !3191
  %arrayidx219 = getelementptr inbounds i32, i32* %rowstr, i64 %indvars.iv.next12, !dbg !3192
  %59 = load i32, i32* %arrayidx219, align 4, !dbg !3192
  %arrayidx221 = getelementptr inbounds i32, i32* %nzloc, i64 %indvars.iv11, !dbg !3193
  %60 = load i32, i32* %arrayidx221, align 4, !dbg !3193
  %sub222 = sub nsw i32 %59, %60, !dbg !3194
  call void @llvm.dbg.value(metadata i32 %sub222, metadata !3195, metadata !DIExpression()), !dbg !2877
  %arrayidx224 = getelementptr inbounds i32, i32* %rowstr, i64 %indvars.iv11, !dbg !3196
  %61 = load i32, i32* %arrayidx224, align 4, !dbg !3196
  call void @llvm.dbg.value(metadata i32 %61, metadata !2917, metadata !DIExpression()), !dbg !2877
  call void @llvm.dbg.value(metadata i32 %j1.0, metadata !2983, metadata !DIExpression()), !dbg !2877
  %62 = sext i32 %61 to i64, !dbg !3197
  %63 = sext i32 %j1.0 to i64, !dbg !3197
  %64 = sext i32 %sub222 to i64, !dbg !3197
  br label %for.cond225, !dbg !3197

for.cond225:                                      ; preds = %for.inc237, %if.end216
  %indvars.iv9 = phi i64 [ %indvars.iv.next10, %for.inc237 ], [ %63, %if.end216 ], !dbg !3199
  %indvars.iv7 = phi i64 [ %indvars.iv.next8, %for.inc237 ], [ %62, %if.end216 ], !dbg !3199
  call void @llvm.dbg.value(metadata i64 %indvars.iv9, metadata !2983, metadata !DIExpression()), !dbg !2877
  call void @llvm.dbg.value(metadata i64 %indvars.iv7, metadata !2917, metadata !DIExpression()), !dbg !2877
  %cmp226 = icmp slt i64 %indvars.iv9, %64, !dbg !3200
  br i1 %cmp226, label %for.body227, label %for.end239, !dbg !3202

for.body227:                                      ; preds = %for.cond225
  %arrayidx229 = getelementptr inbounds double, double* %a, i64 %indvars.iv7, !dbg !3203
  %65 = load double, double* %arrayidx229, align 8, !dbg !3203
  %arrayidx231 = getelementptr inbounds double, double* %a, i64 %indvars.iv9, !dbg !3205
  store double %65, double* %arrayidx231, align 8, !dbg !3206
  %arrayidx233 = getelementptr inbounds i32, i32* %colidx, i64 %indvars.iv7, !dbg !3207
  %66 = load i32, i32* %arrayidx233, align 4, !dbg !3207
  %arrayidx235 = getelementptr inbounds i32, i32* %colidx, i64 %indvars.iv9, !dbg !3208
  store i32 %66, i32* %arrayidx235, align 4, !dbg !3209
  %indvars.iv.next8 = add nsw i64 %indvars.iv7, 1, !dbg !3210
  call void @llvm.dbg.value(metadata i32 undef, metadata !2917, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !2877
  br label %for.inc237, !dbg !3211

for.inc237:                                       ; preds = %for.body227
  %indvars.iv.next10 = add nsw i64 %indvars.iv9, 1, !dbg !3212
  call void @llvm.dbg.value(metadata i32 undef, metadata !2983, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !2877
  br label %for.cond225, !dbg !3213, !llvm.loop !3214

for.end239:                                       ; preds = %for.cond225
  br label %for.inc240, !dbg !3216

for.inc240:                                       ; preds = %for.end239
  call void @llvm.dbg.value(metadata i32 undef, metadata !2894, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !2877
  br label %for.cond204, !dbg !3217, !llvm.loop !3218

for.end242:                                       ; preds = %for.cond204
  call void @llvm.dbg.value(metadata i32 1, metadata !2894, metadata !DIExpression()), !dbg !2877
  br label %for.cond243, !dbg !3220

for.cond243:                                      ; preds = %for.inc255, %for.end242
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc255 ], [ 1, %for.end242 ], !dbg !3222
  call void @llvm.dbg.value(metadata i64 %indvars.iv, metadata !2894, metadata !DIExpression()), !dbg !2877
  %add244 = add nsw i32 %add, 1, !dbg !3223
  %67 = sext i32 %add244 to i64, !dbg !3225
  %cmp245 = icmp slt i64 %indvars.iv, %67, !dbg !3225
  br i1 %cmp245, label %for.body246, label %for.end257, !dbg !3226

for.body246:                                      ; preds = %for.cond243
  %arrayidx248 = getelementptr inbounds i32, i32* %rowstr, i64 %indvars.iv, !dbg !3227
  %68 = load i32, i32* %arrayidx248, align 4, !dbg !3227
  %69 = sub nuw nsw i64 %indvars.iv, 1, !dbg !3229
  %arrayidx251 = getelementptr inbounds i32, i32* %nzloc, i64 %69, !dbg !3230
  %70 = load i32, i32* %arrayidx251, align 4, !dbg !3230
  %sub252 = sub nsw i32 %68, %70, !dbg !3231
  %arrayidx254 = getelementptr inbounds i32, i32* %rowstr, i64 %indvars.iv, !dbg !3232
  store i32 %sub252, i32* %arrayidx254, align 4, !dbg !3233
  br label %for.inc255, !dbg !3234

for.inc255:                                       ; preds = %for.body246
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !3235
  call void @llvm.dbg.value(metadata i32 undef, metadata !2894, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !2877
  br label %for.cond243, !dbg !3236, !llvm.loop !3237

for.end257:                                       ; preds = %for.cond243
  call void @llvm.dbg.value(metadata !1156, metadata !2917, metadata !DIExpression(DW_OP_constu, 1, DW_OP_minus, DW_OP_stack_value)), !dbg !2877
  ret void, !dbg !3239
}

; Function Attrs: noreturn nounwind
declare dso_local void @exit(i32) #9

; Function Attrs: noinline nounwind uwtable
define internal i32 @_ZL6icnvrtdi(double %x, i32 %ipwr2) #3 !dbg !3240 {
entry:
  call void @llvm.dbg.value(metadata double %x, metadata !3243, metadata !DIExpression()), !dbg !3244
  call void @llvm.dbg.value(metadata i32 %ipwr2, metadata !3245, metadata !DIExpression()), !dbg !3244
  %conv = sitofp i32 %ipwr2 to double, !dbg !3246
  %mul = fmul contract double %conv, %x, !dbg !3247
  %conv1 = fptosi double %mul to i32, !dbg !3248
  ret i32 %conv1, !dbg !3249
}

; Function Attrs: convergent noinline nounwind
define dso_local void @gpu_kernel_ten_10(double* %norm_temp, double* %x, double* %z, i32 %gridDim.x, i32 %gridDim.y, i32 %gridDim.z, i32 %blockDim.x, i32 %blockDim.y, i32 %blockDim.z, i32 %blockIdx.x, i32 %blockIdx.y, i32 %blockIdx.z, i32 %threadIdx.x, i32 %threadIdx.y, i32 %threadIdx.z) #10 {
entry:
  call void @llvm.dbg.value(metadata double* %norm_temp, metadata !3250, metadata !DIExpression()), !dbg !3254
  call void @llvm.dbg.value(metadata double* %x, metadata !3255, metadata !DIExpression()), !dbg !3254
  call void @llvm.dbg.value(metadata double* %z, metadata !3256, metadata !DIExpression()), !dbg !3254
  %sharedMem.gep = getelementptr [1024 x double], [1024 x double]* @extern_share_data_shared, i64 0, i64 0
  call void @llvm.dbg.value(metadata double* %sharedMem.gep, metadata !3257, metadata !DIExpression()), !dbg !3254
  %mul = mul i32 %blockIdx.x, %blockDim.x, !dbg !3258
  %add = add i32 %mul, %threadIdx.x, !dbg !3259
  call void @llvm.dbg.value(metadata i32 %add, metadata !3260, metadata !DIExpression()), !dbg !3254
  call void @llvm.dbg.value(metadata i32 %threadIdx.x, metadata !3261, metadata !DIExpression()), !dbg !3254
  %idxprom = zext i32 %threadIdx.x to i64, !dbg !3262
  %arrayidx = getelementptr inbounds double, double* %sharedMem.gep, i64 %idxprom, !dbg !3262
  store double 0.000000e+00, double* %arrayidx, align 8, !dbg !3263
  %cmp = icmp slt i32 %add, 14000, !dbg !3264
  br i1 %cmp, label %if.then, label %if.end, !dbg !3266

if.then:                                          ; preds = %entry
  %idxprom5 = sext i32 %add to i64, !dbg !3267
  %arrayidx6 = getelementptr inbounds double, double* %x, i64 %idxprom5, !dbg !3267
  %0 = load double, double* %arrayidx6, align 8, !dbg !3267
  %idxprom7 = sext i32 %add to i64, !dbg !3269
  %arrayidx8 = getelementptr inbounds double, double* %z, i64 %idxprom7, !dbg !3269
  %1 = load double, double* %arrayidx8, align 8, !dbg !3269
  %mul9 = fmul contract double %0, %1, !dbg !3270
  %idxprom11 = zext i32 %threadIdx.x to i64, !dbg !3271
  %arrayidx12 = getelementptr inbounds double, double* %sharedMem.gep, i64 %idxprom11, !dbg !3271
  store double %mul9, double* %arrayidx12, align 8, !dbg !3272
  br label %if.end, !dbg !3273

if.end:                                           ; preds = %if.then, %entry
  ret void, !dbg !3274
}

; Function Attrs: convergent noinline nounwind
define dso_local void @gpu_kernel_ten_20(double* %norm_temp, double* %x, double* %z, i32 %gridDim.x, i32 %gridDim.y, i32 %gridDim.z, i32 %blockDim.x, i32 %blockDim.y, i32 %blockDim.z, i32 %blockIdx.x, i32 %blockIdx.y, i32 %blockIdx.z, i32 %threadIdx.x, i32 %threadIdx.y, i32 %threadIdx.z) #10 {
entry:
  call void @llvm.dbg.value(metadata double* %norm_temp, metadata !3275, metadata !DIExpression()), !dbg !3277
  call void @llvm.dbg.value(metadata double* %x, metadata !3278, metadata !DIExpression()), !dbg !3277
  call void @llvm.dbg.value(metadata double* %z, metadata !3279, metadata !DIExpression()), !dbg !3277
  %sharedMem.gep = getelementptr [1024 x double], [1024 x double]* @extern_share_data_shared, i64 0, i64 0
  call void @llvm.dbg.value(metadata double* %sharedMem.gep, metadata !3280, metadata !DIExpression()), !dbg !3277
  %mul = mul i32 %blockIdx.x, %blockDim.x, !dbg !3281
  %add = add i32 %mul, %threadIdx.x, !dbg !3282
  call void @llvm.dbg.value(metadata i32 %add, metadata !3283, metadata !DIExpression()), !dbg !3277
  call void @llvm.dbg.value(metadata i32 %threadIdx.x, metadata !3284, metadata !DIExpression()), !dbg !3277
  %idxprom = zext i32 %threadIdx.x to i64, !dbg !3285
  %arrayidx = getelementptr inbounds double, double* %sharedMem.gep, i64 %idxprom, !dbg !3285
  store double 0.000000e+00, double* %arrayidx, align 8, !dbg !3286
  %cmp = icmp slt i32 %add, 14000, !dbg !3287
  br i1 %cmp, label %if.then, label %if.end, !dbg !3289

if.then:                                          ; preds = %entry
  %idxprom5 = sext i32 %add to i64, !dbg !3290
  %arrayidx6 = getelementptr inbounds double, double* %z, i64 %idxprom5, !dbg !3290
  %0 = load double, double* %arrayidx6, align 8, !dbg !3290
  %idxprom7 = sext i32 %add to i64, !dbg !3292
  %arrayidx8 = getelementptr inbounds double, double* %z, i64 %idxprom7, !dbg !3292
  %1 = load double, double* %arrayidx8, align 8, !dbg !3292
  %mul9 = fmul contract double %0, %1, !dbg !3293
  %idxprom11 = zext i32 %threadIdx.x to i64, !dbg !3294
  %arrayidx12 = getelementptr inbounds double, double* %sharedMem.gep, i64 %idxprom11, !dbg !3294
  store double %mul9, double* %arrayidx12, align 8, !dbg !3295
  br label %if.end, !dbg !3296

if.end:                                           ; preds = %if.then, %entry
  ret void, !dbg !3297
}

; Function Attrs: convergent noinline nounwind
define dso_local void @gpu_kernel_eleven_device(double %norm_temp2, double* %x, double* %z, i32 %gridDim.x, i32 %gridDim.y, i32 %gridDim.z, i32 %blockDim.x, i32 %blockDim.y, i32 %blockDim.z, i32 %blockIdx.x, i32 %blockIdx.y, i32 %blockIdx.z, i32 %threadIdx.x, i32 %threadIdx.y, i32 %threadIdx.z) #10 {
entry:
  call void @llvm.dbg.value(metadata double %norm_temp2, metadata !3298, metadata !DIExpression()), !dbg !3302
  call void @llvm.dbg.value(metadata double* %x, metadata !3303, metadata !DIExpression()), !dbg !3302
  call void @llvm.dbg.value(metadata double* %z, metadata !3304, metadata !DIExpression()), !dbg !3302
  %mul = mul i32 %blockIdx.x, %blockDim.x, !dbg !3305
  %add = add i32 %mul, %threadIdx.x, !dbg !3306
  call void @llvm.dbg.value(metadata i32 %add, metadata !3307, metadata !DIExpression()), !dbg !3302
  %cmp = icmp sge i32 %add, 14000, !dbg !3308
  br i1 %cmp, label %if.then, label %if.end, !dbg !3310

if.then:                                          ; preds = %entry
  br label %return, !dbg !3311

if.end:                                           ; preds = %entry
  %idxprom = sext i32 %add to i64, !dbg !3313
  %arrayidx = getelementptr inbounds double, double* %z, i64 %idxprom, !dbg !3313
  %0 = load double, double* %arrayidx, align 8, !dbg !3313
  %mul3 = fmul contract double %norm_temp2, %0, !dbg !3314
  %idxprom4 = sext i32 %add to i64, !dbg !3315
  %arrayidx5 = getelementptr inbounds double, double* %x, i64 %idxprom4, !dbg !3315
  store double %mul3, double* %arrayidx5, align 8, !dbg !3316
  br label %return, !dbg !3317

return:                                           ; preds = %if.end, %if.then
  ret void, !dbg !3317
}

; Function Attrs: convergent noinline nounwind
define dso_local void @gpu_kernel_one_device(double* %p, double* %q, double* %r, double* %x, double* %z, i32 %gridDim.x, i32 %gridDim.y, i32 %gridDim.z, i32 %blockDim.x, i32 %blockDim.y, i32 %blockDim.z, i32 %blockIdx.x, i32 %blockIdx.y, i32 %blockIdx.z, i32 %threadIdx.x, i32 %threadIdx.y, i32 %threadIdx.z) #10 {
entry:
  call void @llvm.dbg.value(metadata double* %p, metadata !3318, metadata !DIExpression()), !dbg !3322
  call void @llvm.dbg.value(metadata double* %q, metadata !3323, metadata !DIExpression()), !dbg !3322
  call void @llvm.dbg.value(metadata double* %r, metadata !3324, metadata !DIExpression()), !dbg !3322
  call void @llvm.dbg.value(metadata double* %x, metadata !3325, metadata !DIExpression()), !dbg !3322
  call void @llvm.dbg.value(metadata double* %z, metadata !3326, metadata !DIExpression()), !dbg !3322
  %mul = mul i32 %blockIdx.x, %blockDim.x, !dbg !3327
  %add = add i32 %mul, %threadIdx.x, !dbg !3328
  call void @llvm.dbg.value(metadata i32 %add, metadata !3329, metadata !DIExpression()), !dbg !3322
  %cmp = icmp sge i32 %add, 14000, !dbg !3330
  br i1 %cmp, label %if.then, label %if.end, !dbg !3332

if.then:                                          ; preds = %entry
  br label %return, !dbg !3333

if.end:                                           ; preds = %entry
  %idxprom = sext i32 %add to i64, !dbg !3335
  %arrayidx = getelementptr inbounds double, double* %q, i64 %idxprom, !dbg !3335
  store double 0.000000e+00, double* %arrayidx, align 8, !dbg !3336
  %idxprom3 = sext i32 %add to i64, !dbg !3337
  %arrayidx4 = getelementptr inbounds double, double* %z, i64 %idxprom3, !dbg !3337
  store double 0.000000e+00, double* %arrayidx4, align 8, !dbg !3338
  %idxprom5 = sext i32 %add to i64, !dbg !3339
  %arrayidx6 = getelementptr inbounds double, double* %x, i64 %idxprom5, !dbg !3339
  %0 = load double, double* %arrayidx6, align 8, !dbg !3339
  call void @llvm.dbg.value(metadata double %0, metadata !3340, metadata !DIExpression()), !dbg !3322
  %idxprom7 = sext i32 %add to i64, !dbg !3341
  %arrayidx8 = getelementptr inbounds double, double* %r, i64 %idxprom7, !dbg !3341
  store double %0, double* %arrayidx8, align 8, !dbg !3342
  %idxprom9 = sext i32 %add to i64, !dbg !3343
  %arrayidx10 = getelementptr inbounds double, double* %p, i64 %idxprom9, !dbg !3343
  store double %0, double* %arrayidx10, align 8, !dbg !3344
  br label %return, !dbg !3345

return:                                           ; preds = %if.end, %if.then
  ret void, !dbg !3345
}

; Function Attrs: convergent noinline nounwind
define dso_local void @gpu_kernel_two_device0(double* %r, double* %rho, double* %global_data, i32 %gridDim.x, i32 %gridDim.y, i32 %gridDim.z, i32 %blockDim.x, i32 %blockDim.y, i32 %blockDim.z, i32 %blockIdx.x, i32 %blockIdx.y, i32 %blockIdx.z, i32 %threadIdx.x, i32 %threadIdx.y, i32 %threadIdx.z) #10 {
entry:
  call void @llvm.dbg.value(metadata double* %r, metadata !3346, metadata !DIExpression()), !dbg !3348
  call void @llvm.dbg.value(metadata double* %rho, metadata !3349, metadata !DIExpression()), !dbg !3348
  call void @llvm.dbg.value(metadata double* %global_data, metadata !3350, metadata !DIExpression()), !dbg !3348
  %sharedMem.gep = getelementptr [1024 x double], [1024 x double]* @extern_share_data_shared, i64 0, i64 0
  call void @llvm.dbg.value(metadata double* %sharedMem.gep, metadata !3351, metadata !DIExpression()), !dbg !3348
  %mul = mul i32 %blockIdx.x, %blockDim.x, !dbg !3352
  %add = add i32 %mul, %threadIdx.x, !dbg !3353
  call void @llvm.dbg.value(metadata i32 %add, metadata !3354, metadata !DIExpression()), !dbg !3348
  call void @llvm.dbg.value(metadata i32 %threadIdx.x, metadata !3355, metadata !DIExpression()), !dbg !3348
  %idxprom = sext i32 %threadIdx.x to i64, !dbg !3356
  %arrayidx = getelementptr inbounds double, double* %sharedMem.gep, i64 %idxprom, !dbg !3356
  store double 0.000000e+00, double* %arrayidx, align 8, !dbg !3357
  %cmp = icmp slt i32 %add, 14000, !dbg !3358
  br i1 %cmp, label %if.then, label %if.end, !dbg !3360

if.then:                                          ; preds = %entry
  %idxprom4 = sext i32 %add to i64, !dbg !3361
  %arrayidx5 = getelementptr inbounds double, double* %r, i64 %idxprom4, !dbg !3361
  %0 = load double, double* %arrayidx5, align 8, !dbg !3361
  call void @llvm.dbg.value(metadata double %0, metadata !3363, metadata !DIExpression()), !dbg !3364
  %mul6 = fmul contract double %0, %0, !dbg !3365
  %idxprom7 = sext i32 %threadIdx.x to i64, !dbg !3366
  %arrayidx8 = getelementptr inbounds double, double* %sharedMem.gep, i64 %idxprom7, !dbg !3366
  store double %mul6, double* %arrayidx8, align 8, !dbg !3367
  br label %if.end, !dbg !3368

if.end:                                           ; preds = %if.then, %entry
  ret void, !dbg !3369
}

; Function Attrs: convergent noinline nounwind
define dso_local void @gpu_kernel_three_device0(i32* %colidx, i32* %rowstr, double* %a, double* %p, double* %q, i32 %gridDim.x, i32 %gridDim.y, i32 %gridDim.z, i32 %blockDim.x, i32 %blockDim.y, i32 %blockDim.z, i32 %blockIdx.x, i32 %blockIdx.y, i32 %blockIdx.z, i32 %threadIdx.x, i32 %threadIdx.y, i32 %threadIdx.z) #10 {
entry:
  call void @llvm.dbg.value(metadata i32* %colidx, metadata !3370, metadata !DIExpression()), !dbg !3374
  call void @llvm.dbg.value(metadata i32* %rowstr, metadata !3375, metadata !DIExpression()), !dbg !3374
  call void @llvm.dbg.value(metadata double* %a, metadata !3376, metadata !DIExpression()), !dbg !3374
  call void @llvm.dbg.value(metadata double* %p, metadata !3377, metadata !DIExpression()), !dbg !3374
  call void @llvm.dbg.value(metadata double* %q, metadata !3378, metadata !DIExpression()), !dbg !3374
  call void @llvm.dbg.value(metadata double* %sharedMem.gep, metadata !3379, metadata !DIExpression()), !dbg !3374
  %mul = mul i32 %blockIdx.x, %blockDim.x, !dbg !3380
  %add = add i32 %mul, %threadIdx.x, !dbg !3381
  %div = udiv i32 %add, %blockDim.x, !dbg !3382
  call void @llvm.dbg.value(metadata i32 %div, metadata !3383, metadata !DIExpression()), !dbg !3374
  call void @llvm.dbg.value(metadata i32 %threadIdx.x, metadata !3384, metadata !DIExpression()), !dbg !3374
  %idxprom = sext i32 %div to i64, !dbg !3385
  %arrayidx = getelementptr inbounds i32, i32* %rowstr, i64 %idxprom, !dbg !3385
  %0 = load i32, i32* %arrayidx, align 4, !dbg !3385
  call void @llvm.dbg.value(metadata i32 %0, metadata !3386, metadata !DIExpression()), !dbg !3374
  %add5 = add nsw i32 %div, 1, !dbg !3387
  %idxprom6 = sext i32 %add5 to i64, !dbg !3388
  %arrayidx7 = getelementptr inbounds i32, i32* %rowstr, i64 %idxprom6, !dbg !3388
  %1 = load i32, i32* %arrayidx7, align 4, !dbg !3388
  call void @llvm.dbg.value(metadata i32 %1, metadata !3389, metadata !DIExpression()), !dbg !3374
  call void @llvm.dbg.value(metadata double 0.000000e+00, metadata !3390, metadata !DIExpression()), !dbg !3374
  %add8 = add nsw i32 %0, %threadIdx.x, !dbg !3391
  call void @llvm.dbg.value(metadata i32 %add8, metadata !3393, metadata !DIExpression()), !dbg !3394
  br label %for.cond, !dbg !3395

for.cond:                                         ; preds = %for.inc, %entry
  %sum.0 = phi double [ 0.000000e+00, %entry ], [ %add16, %for.inc ], !dbg !3374
  %k.0 = phi i32 [ %add8, %entry ], [ %add18, %for.inc ], !dbg !3394
  call void @llvm.dbg.value(metadata i32 %k.0, metadata !3393, metadata !DIExpression()), !dbg !3394
  call void @llvm.dbg.value(metadata double %sum.0, metadata !3390, metadata !DIExpression()), !dbg !3374
  %cmp = icmp slt i32 %k.0, %1, !dbg !3396
  br i1 %cmp, label %for.body, label %for.end, !dbg !3398

for.body:                                         ; preds = %for.cond
  %idxprom9 = sext i32 %k.0 to i64, !dbg !3399
  %arrayidx10 = getelementptr inbounds double, double* %a, i64 %idxprom9, !dbg !3399
  %2 = load double, double* %arrayidx10, align 8, !dbg !3399
  %idxprom11 = sext i32 %k.0 to i64, !dbg !3401
  %arrayidx12 = getelementptr inbounds i32, i32* %colidx, i64 %idxprom11, !dbg !3401
  %3 = load i32, i32* %arrayidx12, align 4, !dbg !3401
  %idxprom13 = sext i32 %3 to i64, !dbg !3402
  %arrayidx14 = getelementptr inbounds double, double* %p, i64 %idxprom13, !dbg !3402
  %4 = load double, double* %arrayidx14, align 8, !dbg !3402
  %mul15 = fmul contract double %2, %4, !dbg !3403
  %add16 = fadd contract double %sum.0, %mul15, !dbg !3404
  call void @llvm.dbg.value(metadata double %add16, metadata !3390, metadata !DIExpression()), !dbg !3374
  br label %for.inc, !dbg !3405

for.inc:                                          ; preds = %for.body
  %add18 = add i32 %k.0, %blockDim.x, !dbg !3406
  call void @llvm.dbg.value(metadata i32 %add18, metadata !3393, metadata !DIExpression()), !dbg !3394
  br label %for.cond, !dbg !3407, !llvm.loop !3408

for.end:                                          ; preds = %for.cond
  %sum.0.lcssa = phi double [ %sum.0, %for.cond ], !dbg !3374
  %sharedMem.gep = getelementptr [1024 x double], [1024 x double]* @extern_share_data_shared, i64 0, i64 0
  call void @llvm.dbg.value(metadata double %sum.0.lcssa, metadata !3390, metadata !DIExpression()), !dbg !3374
  %idxprom19 = sext i32 %threadIdx.x to i64, !dbg !3410
  %arrayidx20 = getelementptr inbounds double, double* %sharedMem.gep, i64 %idxprom19, !dbg !3410
  store double %sum.0.lcssa, double* %arrayidx20, align 8, !dbg !3411
  ret void, !dbg !3412
}

; Function Attrs: convergent noinline nounwind
define dso_local void @gpu_kernel_four_device0(double* %d, double* %p, double* %q, double* %global_data, i32 %gridDim.x, i32 %gridDim.y, i32 %gridDim.z, i32 %blockDim.x, i32 %blockDim.y, i32 %blockDim.z, i32 %blockIdx.x, i32 %blockIdx.y, i32 %blockIdx.z, i32 %threadIdx.x, i32 %threadIdx.y, i32 %threadIdx.z) #10 {
entry:
  call void @llvm.dbg.value(metadata double* %d, metadata !3413, metadata !DIExpression()), !dbg !3417
  call void @llvm.dbg.value(metadata double* %p, metadata !3418, metadata !DIExpression()), !dbg !3417
  call void @llvm.dbg.value(metadata double* %q, metadata !3419, metadata !DIExpression()), !dbg !3417
  call void @llvm.dbg.value(metadata double* %global_data, metadata !3420, metadata !DIExpression()), !dbg !3417
  %sharedMem.gep = getelementptr [1024 x double], [1024 x double]* @extern_share_data_shared, i64 0, i64 0
  call void @llvm.dbg.value(metadata double* %sharedMem.gep, metadata !3421, metadata !DIExpression()), !dbg !3417
  %mul = mul i32 %blockIdx.x, %blockDim.x, !dbg !3422
  %add = add i32 %mul, %threadIdx.x, !dbg !3423
  call void @llvm.dbg.value(metadata i32 %add, metadata !3424, metadata !DIExpression()), !dbg !3417
  call void @llvm.dbg.value(metadata i32 %threadIdx.x, metadata !3425, metadata !DIExpression()), !dbg !3417
  %idxprom = sext i32 %threadIdx.x to i64, !dbg !3426
  %arrayidx = getelementptr inbounds double, double* %sharedMem.gep, i64 %idxprom, !dbg !3426
  store double 0.000000e+00, double* %arrayidx, align 8, !dbg !3427
  %idxprom4 = sext i32 %threadIdx.x to i64, !dbg !3428
  %arrayidx5 = getelementptr inbounds double, double* %sharedMem.gep, i64 %idxprom4, !dbg !3428
  store double 0.000000e+00, double* %arrayidx5, align 8, !dbg !3429
  %cmp = icmp slt i32 %add, 14000, !dbg !3430
  br i1 %cmp, label %if.then, label %if.end, !dbg !3432

if.then:                                          ; preds = %entry
  %idxprom6 = sext i32 %add to i64, !dbg !3433
  %arrayidx7 = getelementptr inbounds double, double* %p, i64 %idxprom6, !dbg !3433
  %0 = load double, double* %arrayidx7, align 8, !dbg !3433
  %idxprom8 = sext i32 %add to i64, !dbg !3435
  %arrayidx9 = getelementptr inbounds double, double* %q, i64 %idxprom8, !dbg !3435
  %1 = load double, double* %arrayidx9, align 8, !dbg !3435
  %mul10 = fmul contract double %0, %1, !dbg !3436
  %idxprom12 = zext i32 %threadIdx.x to i64, !dbg !3437
  %arrayidx13 = getelementptr inbounds double, double* %sharedMem.gep, i64 %idxprom12, !dbg !3437
  store double %mul10, double* %arrayidx13, align 8, !dbg !3438
  br label %if.end, !dbg !3439

if.end:                                           ; preds = %if.then, %entry
  ret void, !dbg !3440
}

; Function Attrs: convergent noinline nounwind
define dso_local void @gpu_kernel_five_1(double %alpha, double* %p, double* %z, i32 %gridDim.x, i32 %gridDim.y, i32 %gridDim.z, i32 %blockDim.x, i32 %blockDim.y, i32 %blockDim.z, i32 %blockIdx.x, i32 %blockIdx.y, i32 %blockIdx.z, i32 %threadIdx.x, i32 %threadIdx.y, i32 %threadIdx.z) #10 {
entry:
  call void @llvm.dbg.value(metadata double %alpha, metadata !3441, metadata !DIExpression()), !dbg !3443
  call void @llvm.dbg.value(metadata double* %p, metadata !3444, metadata !DIExpression()), !dbg !3443
  call void @llvm.dbg.value(metadata double* %z, metadata !3445, metadata !DIExpression()), !dbg !3443
  %mul = mul i32 %blockIdx.x, %blockDim.x, !dbg !3446
  %add = add i32 %mul, %threadIdx.x, !dbg !3447
  call void @llvm.dbg.value(metadata i32 %add, metadata !3448, metadata !DIExpression()), !dbg !3443
  %cmp = icmp sge i32 %add, 14000, !dbg !3449
  br i1 %cmp, label %if.then, label %if.end, !dbg !3451

if.then:                                          ; preds = %entry
  br label %return, !dbg !3452

if.end:                                           ; preds = %entry
  %idxprom = sext i32 %add to i64, !dbg !3454
  %arrayidx = getelementptr inbounds double, double* %p, i64 %idxprom, !dbg !3454
  %0 = load double, double* %arrayidx, align 8, !dbg !3454
  %mul3 = fmul contract double %alpha, %0, !dbg !3455
  %idxprom4 = sext i32 %add to i64, !dbg !3456
  %arrayidx5 = getelementptr inbounds double, double* %z, i64 %idxprom4, !dbg !3456
  %1 = load double, double* %arrayidx5, align 8, !dbg !3457
  %add6 = fadd contract double %1, %mul3, !dbg !3457
  store double %add6, double* %arrayidx5, align 8, !dbg !3457
  br label %return, !dbg !3458

return:                                           ; preds = %if.end, %if.then
  ret void, !dbg !3458
}

; Function Attrs: convergent noinline nounwind
define dso_local void @gpu_kernel_five_2(double %alpha, double* %q, double* %r, i32 %gridDim.x, i32 %gridDim.y, i32 %gridDim.z, i32 %blockDim.x, i32 %blockDim.y, i32 %blockDim.z, i32 %blockIdx.x, i32 %blockIdx.y, i32 %blockIdx.z, i32 %threadIdx.x, i32 %threadIdx.y, i32 %threadIdx.z) #10 {
entry:
  call void @llvm.dbg.value(metadata double %alpha, metadata !3459, metadata !DIExpression()), !dbg !3461
  call void @llvm.dbg.value(metadata double* %q, metadata !3462, metadata !DIExpression()), !dbg !3461
  call void @llvm.dbg.value(metadata double* %r, metadata !3463, metadata !DIExpression()), !dbg !3461
  %mul = mul i32 %blockIdx.x, %blockDim.x, !dbg !3464
  %add = add i32 %mul, %threadIdx.x, !dbg !3465
  call void @llvm.dbg.value(metadata i32 %add, metadata !3466, metadata !DIExpression()), !dbg !3461
  %cmp = icmp sge i32 %add, 14000, !dbg !3467
  br i1 %cmp, label %if.then, label %if.end, !dbg !3469

if.then:                                          ; preds = %entry
  br label %return, !dbg !3470

if.end:                                           ; preds = %entry
  %idxprom = sext i32 %add to i64, !dbg !3472
  %arrayidx = getelementptr inbounds double, double* %q, i64 %idxprom, !dbg !3472
  %0 = load double, double* %arrayidx, align 8, !dbg !3472
  %mul3 = fmul contract double %alpha, %0, !dbg !3473
  %idxprom4 = sext i32 %add to i64, !dbg !3474
  %arrayidx5 = getelementptr inbounds double, double* %r, i64 %idxprom4, !dbg !3474
  %1 = load double, double* %arrayidx5, align 8, !dbg !3475
  %sub = fsub contract double %1, %mul3, !dbg !3475
  store double %sub, double* %arrayidx5, align 8, !dbg !3475
  br label %return, !dbg !3476

return:                                           ; preds = %if.end, %if.then
  ret void, !dbg !3476
}

; Function Attrs: convergent noinline nounwind
define dso_local void @gpu_kernel_six_device0(double* %r, double* %global_data, i32 %gridDim.x, i32 %gridDim.y, i32 %gridDim.z, i32 %blockDim.x, i32 %blockDim.y, i32 %blockDim.z, i32 %blockIdx.x, i32 %blockIdx.y, i32 %blockIdx.z, i32 %threadIdx.x, i32 %threadIdx.y, i32 %threadIdx.z) #10 {
entry:
  call void @llvm.dbg.value(metadata double* %r, metadata !3477, metadata !DIExpression()), !dbg !3479
  call void @llvm.dbg.value(metadata double* %global_data, metadata !3480, metadata !DIExpression()), !dbg !3479
  %sharedMem.gep = getelementptr [1024 x double], [1024 x double]* @extern_share_data_shared, i64 0, i64 0
  call void @llvm.dbg.value(metadata double* %sharedMem.gep, metadata !3481, metadata !DIExpression()), !dbg !3479
  %mul = mul i32 %blockIdx.x, %blockDim.x, !dbg !3482
  %add = add i32 %mul, %threadIdx.x, !dbg !3483
  call void @llvm.dbg.value(metadata i32 %add, metadata !3484, metadata !DIExpression()), !dbg !3479
  call void @llvm.dbg.value(metadata i32 %threadIdx.x, metadata !3485, metadata !DIExpression()), !dbg !3479
  %idxprom = sext i32 %threadIdx.x to i64, !dbg !3486
  %arrayidx = getelementptr inbounds double, double* %sharedMem.gep, i64 %idxprom, !dbg !3486
  store double 0.000000e+00, double* %arrayidx, align 8, !dbg !3487
  %cmp = icmp slt i32 %add, 14000, !dbg !3488
  br i1 %cmp, label %if.then, label %if.end, !dbg !3490

if.then:                                          ; preds = %entry
  %idxprom4 = sext i32 %add to i64, !dbg !3491
  %arrayidx5 = getelementptr inbounds double, double* %r, i64 %idxprom4, !dbg !3491
  %0 = load double, double* %arrayidx5, align 8, !dbg !3491
  call void @llvm.dbg.value(metadata double %0, metadata !3493, metadata !DIExpression()), !dbg !3494
  %mul6 = fmul contract double %0, %0, !dbg !3495
  %idxprom7 = sext i32 %threadIdx.x to i64, !dbg !3496
  %arrayidx8 = getelementptr inbounds double, double* %sharedMem.gep, i64 %idxprom7, !dbg !3496
  store double %mul6, double* %arrayidx8, align 8, !dbg !3497
  br label %if.end, !dbg !3498

if.end:                                           ; preds = %if.then, %entry
  ret void, !dbg !3499
}

; Function Attrs: convergent noinline nounwind
define dso_local void @gpu_kernel_seven_device(double %beta, double* %p, double* %r, i32 %gridDim.x, i32 %gridDim.y, i32 %gridDim.z, i32 %blockDim.x, i32 %blockDim.y, i32 %blockDim.z, i32 %blockIdx.x, i32 %blockIdx.y, i32 %blockIdx.z, i32 %threadIdx.x, i32 %threadIdx.y, i32 %threadIdx.z) #10 {
entry:
  call void @llvm.dbg.value(metadata double %beta, metadata !3500, metadata !DIExpression()), !dbg !3502
  call void @llvm.dbg.value(metadata double* %p, metadata !3503, metadata !DIExpression()), !dbg !3502
  call void @llvm.dbg.value(metadata double* %r, metadata !3504, metadata !DIExpression()), !dbg !3502
  %mul = mul i32 %blockIdx.x, %blockDim.x, !dbg !3505
  %add = add i32 %mul, %threadIdx.x, !dbg !3506
  call void @llvm.dbg.value(metadata i32 %add, metadata !3507, metadata !DIExpression()), !dbg !3502
  %cmp = icmp sge i32 %add, 14000, !dbg !3508
  br i1 %cmp, label %if.then, label %if.end, !dbg !3510

if.then:                                          ; preds = %entry
  br label %return, !dbg !3511

if.end:                                           ; preds = %entry
  %idxprom = sext i32 %add to i64, !dbg !3513
  %arrayidx = getelementptr inbounds double, double* %r, i64 %idxprom, !dbg !3513
  %0 = load double, double* %arrayidx, align 8, !dbg !3513
  %idxprom3 = sext i32 %add to i64, !dbg !3514
  %arrayidx4 = getelementptr inbounds double, double* %p, i64 %idxprom3, !dbg !3514
  %1 = load double, double* %arrayidx4, align 8, !dbg !3514
  %mul5 = fmul contract double %beta, %1, !dbg !3515
  %add6 = fadd contract double %0, %mul5, !dbg !3516
  %idxprom7 = sext i32 %add to i64, !dbg !3517
  %arrayidx8 = getelementptr inbounds double, double* %p, i64 %idxprom7, !dbg !3517
  store double %add6, double* %arrayidx8, align 8, !dbg !3518
  br label %return, !dbg !3519

return:                                           ; preds = %if.end, %if.then
  ret void, !dbg !3519
}

; Function Attrs: convergent noinline nounwind
define dso_local void @gpu_kernel_eight_device0(i32* %colidx, i32* %rowstr, double* %a, double* %r, double* %z, i32 %gridDim.x, i32 %gridDim.y, i32 %gridDim.z, i32 %blockDim.x, i32 %blockDim.y, i32 %blockDim.z, i32 %blockIdx.x, i32 %blockIdx.y, i32 %blockIdx.z, i32 %threadIdx.x, i32 %threadIdx.y, i32 %threadIdx.z) #10 {
entry:
  call void @llvm.dbg.value(metadata i32* %colidx, metadata !3520, metadata !DIExpression()), !dbg !3522
  call void @llvm.dbg.value(metadata i32* %rowstr, metadata !3523, metadata !DIExpression()), !dbg !3522
  call void @llvm.dbg.value(metadata double* %a, metadata !3524, metadata !DIExpression()), !dbg !3522
  call void @llvm.dbg.value(metadata double* %r, metadata !3525, metadata !DIExpression()), !dbg !3522
  call void @llvm.dbg.value(metadata double* %z, metadata !3526, metadata !DIExpression()), !dbg !3522
  call void @llvm.dbg.value(metadata double* %sharedMem.gep, metadata !3527, metadata !DIExpression()), !dbg !3522
  %mul = mul i32 %blockIdx.x, %blockDim.x, !dbg !3528
  %add = add i32 %mul, %threadIdx.x, !dbg !3529
  %div = udiv i32 %add, %blockDim.x, !dbg !3530
  call void @llvm.dbg.value(metadata i32 %div, metadata !3531, metadata !DIExpression()), !dbg !3522
  call void @llvm.dbg.value(metadata i32 %threadIdx.x, metadata !3532, metadata !DIExpression()), !dbg !3522
  %idxprom = sext i32 %div to i64, !dbg !3533
  %arrayidx = getelementptr inbounds i32, i32* %rowstr, i64 %idxprom, !dbg !3533
  %0 = load i32, i32* %arrayidx, align 4, !dbg !3533
  call void @llvm.dbg.value(metadata i32 %0, metadata !3534, metadata !DIExpression()), !dbg !3522
  %add5 = add nsw i32 %div, 1, !dbg !3535
  %idxprom6 = sext i32 %add5 to i64, !dbg !3536
  %arrayidx7 = getelementptr inbounds i32, i32* %rowstr, i64 %idxprom6, !dbg !3536
  %1 = load i32, i32* %arrayidx7, align 4, !dbg !3536
  call void @llvm.dbg.value(metadata i32 %1, metadata !3537, metadata !DIExpression()), !dbg !3522
  call void @llvm.dbg.value(metadata double 0.000000e+00, metadata !3538, metadata !DIExpression()), !dbg !3522
  %add8 = add nsw i32 %0, %threadIdx.x, !dbg !3539
  call void @llvm.dbg.value(metadata i32 %add8, metadata !3541, metadata !DIExpression()), !dbg !3542
  br label %for.cond, !dbg !3543

for.cond:                                         ; preds = %for.inc, %entry
  %sum.0 = phi double [ 0.000000e+00, %entry ], [ %add16, %for.inc ], !dbg !3522
  %k.0 = phi i32 [ %add8, %entry ], [ %add18, %for.inc ], !dbg !3542
  call void @llvm.dbg.value(metadata i32 %k.0, metadata !3541, metadata !DIExpression()), !dbg !3542
  call void @llvm.dbg.value(metadata double %sum.0, metadata !3538, metadata !DIExpression()), !dbg !3522
  %cmp = icmp slt i32 %k.0, %1, !dbg !3544
  br i1 %cmp, label %for.body, label %for.end, !dbg !3546

for.body:                                         ; preds = %for.cond
  %idxprom9 = sext i32 %k.0 to i64, !dbg !3547
  %arrayidx10 = getelementptr inbounds double, double* %a, i64 %idxprom9, !dbg !3547
  %2 = load double, double* %arrayidx10, align 8, !dbg !3547
  %idxprom11 = sext i32 %k.0 to i64, !dbg !3549
  %arrayidx12 = getelementptr inbounds i32, i32* %colidx, i64 %idxprom11, !dbg !3549
  %3 = load i32, i32* %arrayidx12, align 4, !dbg !3549
  %idxprom13 = sext i32 %3 to i64, !dbg !3550
  %arrayidx14 = getelementptr inbounds double, double* %z, i64 %idxprom13, !dbg !3550
  %4 = load double, double* %arrayidx14, align 8, !dbg !3550
  %mul15 = fmul contract double %2, %4, !dbg !3551
  %add16 = fadd contract double %sum.0, %mul15, !dbg !3552
  call void @llvm.dbg.value(metadata double %add16, metadata !3538, metadata !DIExpression()), !dbg !3522
  br label %for.inc, !dbg !3553

for.inc:                                          ; preds = %for.body
  %add18 = add i32 %k.0, %blockDim.x, !dbg !3554
  call void @llvm.dbg.value(metadata i32 %add18, metadata !3541, metadata !DIExpression()), !dbg !3542
  br label %for.cond, !dbg !3555, !llvm.loop !3556

for.end:                                          ; preds = %for.cond
  %sum.0.lcssa = phi double [ %sum.0, %for.cond ], !dbg !3522
  %sharedMem.gep = getelementptr [1024 x double], [1024 x double]* @extern_share_data_shared, i64 0, i64 0
  call void @llvm.dbg.value(metadata double %sum.0.lcssa, metadata !3538, metadata !DIExpression()), !dbg !3522
  %idxprom19 = sext i32 %threadIdx.x to i64, !dbg !3558
  %arrayidx20 = getelementptr inbounds double, double* %sharedMem.gep, i64 %idxprom19, !dbg !3558
  store double %sum.0.lcssa, double* %arrayidx20, align 8, !dbg !3559
  ret void, !dbg !3560
}

; Function Attrs: convergent noinline nounwind
define dso_local void @gpu_kernel_nine_device0(double* %r, double* %x, double* %sum, double* %global_data, i32 %gridDim.x, i32 %gridDim.y, i32 %gridDim.z, i32 %blockDim.x, i32 %blockDim.y, i32 %blockDim.z, i32 %blockIdx.x, i32 %blockIdx.y, i32 %blockIdx.z, i32 %threadIdx.x, i32 %threadIdx.y, i32 %threadIdx.z) #10 {
entry:
  call void @llvm.dbg.value(metadata double* %r, metadata !3561, metadata !DIExpression()), !dbg !3563
  call void @llvm.dbg.value(metadata double* %x, metadata !3564, metadata !DIExpression()), !dbg !3563
  call void @llvm.dbg.value(metadata double* %sum, metadata !3565, metadata !DIExpression()), !dbg !3563
  call void @llvm.dbg.value(metadata double* %global_data, metadata !3566, metadata !DIExpression()), !dbg !3563
  %sharedMem.gep = getelementptr [1024 x double], [1024 x double]* @extern_share_data_shared, i64 0, i64 0
  call void @llvm.dbg.value(metadata double* %sharedMem.gep, metadata !3567, metadata !DIExpression()), !dbg !3563
  %mul = mul i32 %blockIdx.x, %blockDim.x, !dbg !3568
  %add = add i32 %mul, %threadIdx.x, !dbg !3569
  call void @llvm.dbg.value(metadata i32 %add, metadata !3570, metadata !DIExpression()), !dbg !3563
  call void @llvm.dbg.value(metadata i32 %threadIdx.x, metadata !3571, metadata !DIExpression()), !dbg !3563
  %idxprom = sext i32 %threadIdx.x to i64, !dbg !3572
  %arrayidx = getelementptr inbounds double, double* %sharedMem.gep, i64 %idxprom, !dbg !3572
  store double 0.000000e+00, double* %arrayidx, align 8, !dbg !3573
  %cmp = icmp slt i32 %add, 14000, !dbg !3574
  br i1 %cmp, label %if.then, label %if.end, !dbg !3576

if.then:                                          ; preds = %entry
  %idxprom4 = sext i32 %add to i64, !dbg !3577
  %arrayidx5 = getelementptr inbounds double, double* %x, i64 %idxprom4, !dbg !3577
  %0 = load double, double* %arrayidx5, align 8, !dbg !3577
  %idxprom6 = sext i32 %add to i64, !dbg !3579
  %arrayidx7 = getelementptr inbounds double, double* %r, i64 %idxprom6, !dbg !3579
  %1 = load double, double* %arrayidx7, align 8, !dbg !3579
  %sub = fsub contract double %0, %1, !dbg !3580
  %idxprom8 = sext i32 %threadIdx.x to i64, !dbg !3581
  %arrayidx9 = getelementptr inbounds double, double* %sharedMem.gep, i64 %idxprom8, !dbg !3581
  store double %sub, double* %arrayidx9, align 8, !dbg !3582
  %idxprom10 = sext i32 %threadIdx.x to i64, !dbg !3583
  %arrayidx11 = getelementptr inbounds double, double* %sharedMem.gep, i64 %idxprom10, !dbg !3583
  %2 = load double, double* %arrayidx11, align 8, !dbg !3583
  %idxprom12 = sext i32 %threadIdx.x to i64, !dbg !3584
  %arrayidx13 = getelementptr inbounds double, double* %sharedMem.gep, i64 %idxprom12, !dbg !3584
  %3 = load double, double* %arrayidx13, align 8, !dbg !3584
  %mul14 = fmul contract double %2, %3, !dbg !3585
  %idxprom15 = sext i32 %threadIdx.x to i64, !dbg !3586
  %arrayidx16 = getelementptr inbounds double, double* %sharedMem.gep, i64 %idxprom15, !dbg !3586
  store double %mul14, double* %arrayidx16, align 8, !dbg !3587
  br label %if.end, !dbg !3588

if.end:                                           ; preds = %if.then, %entry
  ret void, !dbg !3589
}

; Function Attrs: convergent noinline nounwind
define dso_local void @gpu_kernel_ten_21(double* %norm_temp, double* %x, double* %z, i32 %gridDim.x, i32 %gridDim.y, i32 %gridDim.z, i32 %blockDim.x, i32 %blockDim.y, i32 %blockDim.z, i32 %blockIdx.x, i32 %blockIdx.y, i32 %blockIdx.z, i32 %threadIdx.x, i32 %threadIdx.y, i32 %threadIdx.z) #10 {
entry:
  call void @llvm.dbg.value(metadata double* %norm_temp, metadata !3275, metadata !DIExpression()), !dbg !3277
  call void @llvm.dbg.value(metadata double* %x, metadata !3278, metadata !DIExpression()), !dbg !3277
  call void @llvm.dbg.value(metadata double* %z, metadata !3279, metadata !DIExpression()), !dbg !3277
  %sharedMem.gep = getelementptr [1024 x double], [1024 x double]* @extern_share_data_shared, i64 0, i64 0
  call void @llvm.dbg.value(metadata double* %sharedMem.gep, metadata !3280, metadata !DIExpression()), !dbg !3277
  call void @llvm.dbg.value(metadata !1156, metadata !3283, metadata !DIExpression()), !dbg !3277
  call void @llvm.dbg.value(metadata i32 %threadIdx.x, metadata !3284, metadata !DIExpression()), !dbg !3277
  %idxprom = zext i32 %threadIdx.x to i64, !dbg !3285
  %arrayidx = getelementptr inbounds double, double* %sharedMem.gep, i64 %idxprom, !dbg !3285
  store double 0.000000e+00, double* %arrayidx, align 8, !dbg !3286
  br label %syncpoint.1, !dbg !3289

syncpoint.1:                                      ; preds = %entry
  call void @llvm.nvvm.barrier0(), !dbg !3297
  %cmp13 = icmp eq i32 %threadIdx.x, 0, !dbg !3590
  br i1 %cmp13, label %if.then14, label %if.end25, !dbg !3592

if.then14:                                        ; preds = %syncpoint.1
  call void @llvm.dbg.value(metadata i32 1, metadata !3593, metadata !DIExpression()), !dbg !3596
  %0 = zext i32 %blockDim.x to i64, !dbg !3597
  br label %for.cond, !dbg !3597

for.cond:                                         ; preds = %for.inc, %if.then14
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 1, %if.then14 ], !dbg !3596
  call void @llvm.dbg.value(metadata i64 %indvars.iv, metadata !3593, metadata !DIExpression()), !dbg !3596
  %cmp16 = icmp ult i64 %indvars.iv, %0, !dbg !3598
  br i1 %cmp16, label %for.body, label %for.end, !dbg !3600

for.body:                                         ; preds = %for.cond
  %arrayidx18 = getelementptr inbounds double, double* %sharedMem.gep, i64 %indvars.iv, !dbg !3601
  %1 = load double, double* %arrayidx18, align 8, !dbg !3601
  %arrayidx19 = getelementptr inbounds double, double* %sharedMem.gep, i64 0, !dbg !3603
  %2 = load double, double* %arrayidx19, align 8, !dbg !3604
  %add20 = fadd contract double %2, %1, !dbg !3604
  store double %add20, double* %arrayidx19, align 8, !dbg !3604
  br label %for.inc, !dbg !3605

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !3606
  call void @llvm.dbg.value(metadata i32 undef, metadata !3593, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !3596
  br label %for.cond, !dbg !3607, !llvm.loop !3608

for.end:                                          ; preds = %for.cond
  %arrayidx21 = getelementptr inbounds double, double* %sharedMem.gep, i64 0, !dbg !3610
  %3 = load double, double* %arrayidx21, align 8, !dbg !3610
  %idxprom23 = zext i32 %blockIdx.x to i64, !dbg !3611
  %arrayidx24 = getelementptr inbounds double, double* %norm_temp, i64 %idxprom23, !dbg !3611
  store double %3, double* %arrayidx24, align 8, !dbg !3612
  br label %if.end25, !dbg !3613

if.end25:                                         ; preds = %for.end, %syncpoint.1
  ret void, !dbg !3614
}

; Function Attrs: convergent noinline nounwind
define dso_local void @gpu_kernel_ten_11(double* %norm_temp, double* %x, double* %z, i32 %gridDim.x, i32 %gridDim.y, i32 %gridDim.z, i32 %blockDim.x, i32 %blockDim.y, i32 %blockDim.z, i32 %blockIdx.x, i32 %blockIdx.y, i32 %blockIdx.z, i32 %threadIdx.x, i32 %threadIdx.y, i32 %threadIdx.z) #10 {
entry:
  call void @llvm.dbg.value(metadata double* %norm_temp, metadata !3250, metadata !DIExpression()), !dbg !3254
  call void @llvm.dbg.value(metadata double* %x, metadata !3255, metadata !DIExpression()), !dbg !3254
  call void @llvm.dbg.value(metadata double* %z, metadata !3256, metadata !DIExpression()), !dbg !3254
  %sharedMem.gep = getelementptr [1024 x double], [1024 x double]* @extern_share_data_shared, i64 0, i64 0
  call void @llvm.dbg.value(metadata double* %sharedMem.gep, metadata !3257, metadata !DIExpression()), !dbg !3254
  call void @llvm.dbg.value(metadata !1156, metadata !3260, metadata !DIExpression()), !dbg !3254
  call void @llvm.dbg.value(metadata i32 %threadIdx.x, metadata !3261, metadata !DIExpression()), !dbg !3254
  %idxprom = zext i32 %threadIdx.x to i64, !dbg !3262
  %arrayidx = getelementptr inbounds double, double* %sharedMem.gep, i64 %idxprom, !dbg !3262
  store double 0.000000e+00, double* %arrayidx, align 8, !dbg !3263
  br label %syncpoint.1, !dbg !3266

syncpoint.1:                                      ; preds = %entry
  call void @llvm.nvvm.barrier0(), !dbg !3274
  %cmp13 = icmp eq i32 %threadIdx.x, 0, !dbg !3615
  br i1 %cmp13, label %if.then14, label %if.end25, !dbg !3617

if.then14:                                        ; preds = %syncpoint.1
  call void @llvm.dbg.value(metadata i32 1, metadata !3618, metadata !DIExpression()), !dbg !3621
  %0 = zext i32 %blockDim.x to i64, !dbg !3622
  br label %for.cond, !dbg !3622

for.cond:                                         ; preds = %for.inc, %if.then14
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 1, %if.then14 ], !dbg !3621
  call void @llvm.dbg.value(metadata i64 %indvars.iv, metadata !3618, metadata !DIExpression()), !dbg !3621
  %cmp16 = icmp ult i64 %indvars.iv, %0, !dbg !3623
  br i1 %cmp16, label %for.body, label %for.end, !dbg !3625

for.body:                                         ; preds = %for.cond
  %arrayidx18 = getelementptr inbounds double, double* %sharedMem.gep, i64 %indvars.iv, !dbg !3626
  %1 = load double, double* %arrayidx18, align 8, !dbg !3626
  %arrayidx19 = getelementptr inbounds double, double* %sharedMem.gep, i64 0, !dbg !3628
  %2 = load double, double* %arrayidx19, align 8, !dbg !3629
  %add20 = fadd contract double %2, %1, !dbg !3629
  store double %add20, double* %arrayidx19, align 8, !dbg !3629
  br label %for.inc, !dbg !3630

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !3631
  call void @llvm.dbg.value(metadata i32 undef, metadata !3618, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !3621
  br label %for.cond, !dbg !3632, !llvm.loop !3633

for.end:                                          ; preds = %for.cond
  %arrayidx21 = getelementptr inbounds double, double* %sharedMem.gep, i64 0, !dbg !3635
  %3 = load double, double* %arrayidx21, align 8, !dbg !3635
  %idxprom23 = zext i32 %blockIdx.x to i64, !dbg !3636
  %arrayidx24 = getelementptr inbounds double, double* %norm_temp, i64 %idxprom23, !dbg !3636
  store double %3, double* %arrayidx24, align 8, !dbg !3637
  br label %if.end25, !dbg !3638

if.end25:                                         ; preds = %for.end, %syncpoint.1
  ret void, !dbg !3639
}

; Function Attrs: convergent noinline nounwind
define dso_local void @gpu_kernel_nine_device1(double* %r, double* %x, double* %sum, double* %global_data, i32 %gridDim.x, i32 %gridDim.y, i32 %gridDim.z, i32 %blockDim.x, i32 %blockDim.y, i32 %blockDim.z, i32 %blockIdx.x, i32 %blockIdx.y, i32 %blockIdx.z, i32 %threadIdx.x, i32 %threadIdx.y, i32 %threadIdx.z) #10 {
entry:
  call void @llvm.dbg.value(metadata double* %r, metadata !3561, metadata !DIExpression()), !dbg !3563
  call void @llvm.dbg.value(metadata double* %x, metadata !3564, metadata !DIExpression()), !dbg !3563
  call void @llvm.dbg.value(metadata double* %sum, metadata !3565, metadata !DIExpression()), !dbg !3563
  call void @llvm.dbg.value(metadata double* %global_data, metadata !3566, metadata !DIExpression()), !dbg !3563
  %sharedMem.gep = getelementptr [1024 x double], [1024 x double]* @extern_share_data_shared, i64 0, i64 0
  call void @llvm.dbg.value(metadata double* %sharedMem.gep, metadata !3567, metadata !DIExpression()), !dbg !3563
  call void @llvm.dbg.value(metadata !1156, metadata !3570, metadata !DIExpression()), !dbg !3563
  call void @llvm.dbg.value(metadata i32 %threadIdx.x, metadata !3571, metadata !DIExpression()), !dbg !3563
  %idxprom = sext i32 %threadIdx.x to i64, !dbg !3572
  %arrayidx = getelementptr inbounds double, double* %sharedMem.gep, i64 %idxprom, !dbg !3572
  store double 0.000000e+00, double* %arrayidx, align 8, !dbg !3573
  br label %syncpoint.1, !dbg !3576

syncpoint.1:                                      ; preds = %entry
  call void @llvm.nvvm.barrier0(), !dbg !3589
  %cmp17 = icmp eq i32 %threadIdx.x, 0, !dbg !3640
  br i1 %cmp17, label %if.then18, label %if.end29, !dbg !3642

if.then18:                                        ; preds = %syncpoint.1
  call void @llvm.dbg.value(metadata i32 1, metadata !3643, metadata !DIExpression()), !dbg !3646
  %0 = zext i32 %blockDim.x to i64, !dbg !3647
  br label %for.cond, !dbg !3647

for.cond:                                         ; preds = %for.inc, %if.then18
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 1, %if.then18 ], !dbg !3646
  call void @llvm.dbg.value(metadata i64 %indvars.iv, metadata !3643, metadata !DIExpression()), !dbg !3646
  %cmp20 = icmp ult i64 %indvars.iv, %0, !dbg !3648
  br i1 %cmp20, label %for.body, label %for.end, !dbg !3650

for.body:                                         ; preds = %for.cond
  %arrayidx22 = getelementptr inbounds double, double* %sharedMem.gep, i64 %indvars.iv, !dbg !3651
  %1 = load double, double* %arrayidx22, align 8, !dbg !3651
  %arrayidx23 = getelementptr inbounds double, double* %sharedMem.gep, i64 0, !dbg !3653
  %2 = load double, double* %arrayidx23, align 8, !dbg !3654
  %add24 = fadd contract double %2, %1, !dbg !3654
  store double %add24, double* %arrayidx23, align 8, !dbg !3654
  br label %for.inc, !dbg !3655

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !3656
  call void @llvm.dbg.value(metadata i32 undef, metadata !3643, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !3646
  br label %for.cond, !dbg !3657, !llvm.loop !3658

for.end:                                          ; preds = %for.cond
  %arrayidx25 = getelementptr inbounds double, double* %sharedMem.gep, i64 0, !dbg !3660
  %3 = load double, double* %arrayidx25, align 8, !dbg !3660
  %idxprom27 = zext i32 %blockIdx.x to i64, !dbg !3661
  %arrayidx28 = getelementptr inbounds double, double* %global_data, i64 %idxprom27, !dbg !3661
  store double %3, double* %arrayidx28, align 8, !dbg !3662
  br label %if.end29, !dbg !3663

if.end29:                                         ; preds = %for.end, %syncpoint.1
  ret void, !dbg !3664
}

; Function Attrs: convergent noinline nounwind
define dso_local void @gpu_kernel_three_device1(i32* %colidx, i32* %rowstr, double* %a, double* %p, double* %q, i32 %gridDim.x, i32 %gridDim.y, i32 %gridDim.z, i32 %blockDim.x, i32 %blockDim.y, i32 %blockDim.z, i32 %blockIdx.x, i32 %blockIdx.y, i32 %blockIdx.z, i32 %threadIdx.x, i32 %threadIdx.y, i32 %threadIdx.z) #10 {
entry:
  call void @llvm.dbg.value(metadata i32* %colidx, metadata !3370, metadata !DIExpression()), !dbg !3374
  call void @llvm.dbg.value(metadata i32* %rowstr, metadata !3375, metadata !DIExpression()), !dbg !3374
  call void @llvm.dbg.value(metadata double* %a, metadata !3376, metadata !DIExpression()), !dbg !3374
  call void @llvm.dbg.value(metadata double* %p, metadata !3377, metadata !DIExpression()), !dbg !3374
  call void @llvm.dbg.value(metadata double* %q, metadata !3378, metadata !DIExpression()), !dbg !3374
  %sharedMem.gep = getelementptr [1024 x double], [1024 x double]* @extern_share_data_shared, i64 0, i64 0
  call void @llvm.dbg.value(metadata double* %sharedMem.gep, metadata !3379, metadata !DIExpression()), !dbg !3374
  %mul = mul i32 %blockIdx.x, %blockDim.x, !dbg !3380
  %add = add i32 %mul, %threadIdx.x, !dbg !3381
  %div = udiv i32 %add, %blockDim.x, !dbg !3382
  call void @llvm.dbg.value(metadata i32 %div, metadata !3383, metadata !DIExpression()), !dbg !3374
  call void @llvm.dbg.value(metadata i32 %threadIdx.x, metadata !3384, metadata !DIExpression()), !dbg !3374
  call void @llvm.dbg.value(metadata !1156, metadata !3386, metadata !DIExpression()), !dbg !3374
  call void @llvm.dbg.value(metadata double 0.000000e+00, metadata !3390, metadata !DIExpression()), !dbg !3374
  br label %syncpoint.1, !dbg !3395

syncpoint.1:                                      ; preds = %entry
  call void @llvm.nvvm.barrier0(), !dbg !3412
  %cmp21 = icmp eq i32 %threadIdx.x, 0, !dbg !3665
  br i1 %cmp21, label %if.then, label %if.end, !dbg !3667

if.then:                                          ; preds = %syncpoint.1
  call void @llvm.dbg.value(metadata i32 1, metadata !3668, metadata !DIExpression()), !dbg !3671
  %0 = zext i32 %blockDim.x to i64, !dbg !3672
  br label %for.cond22, !dbg !3672

for.cond22:                                       ; preds = %for.inc30, %if.then
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc30 ], [ 1, %if.then ], !dbg !3671
  call void @llvm.dbg.value(metadata i64 %indvars.iv, metadata !3668, metadata !DIExpression()), !dbg !3671
  %cmp24 = icmp ult i64 %indvars.iv, %0, !dbg !3673
  br i1 %cmp24, label %for.body25, label %for.end31, !dbg !3675

for.body25:                                       ; preds = %for.cond22
  %arrayidx27 = getelementptr inbounds double, double* %sharedMem.gep, i64 %indvars.iv, !dbg !3676
  %1 = load double, double* %arrayidx27, align 8, !dbg !3676
  %arrayidx28 = getelementptr inbounds double, double* %sharedMem.gep, i64 0, !dbg !3678
  %2 = load double, double* %arrayidx28, align 8, !dbg !3679
  %add29 = fadd contract double %2, %1, !dbg !3679
  store double %add29, double* %arrayidx28, align 8, !dbg !3679
  br label %for.inc30, !dbg !3680

for.inc30:                                        ; preds = %for.body25
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !3681
  call void @llvm.dbg.value(metadata i32 undef, metadata !3668, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !3671
  br label %for.cond22, !dbg !3682, !llvm.loop !3683

for.end31:                                        ; preds = %for.cond22
  %arrayidx32 = getelementptr inbounds double, double* %sharedMem.gep, i64 0, !dbg !3685
  %3 = load double, double* %arrayidx32, align 8, !dbg !3685
  %idxprom33 = sext i32 %div to i64, !dbg !3686
  %arrayidx34 = getelementptr inbounds double, double* %q, i64 %idxprom33, !dbg !3686
  store double %3, double* %arrayidx34, align 8, !dbg !3687
  br label %if.end, !dbg !3688

if.end:                                           ; preds = %for.end31, %syncpoint.1
  ret void, !dbg !3689
}

; Function Attrs: convergent noinline nounwind
define dso_local void @gpu_kernel_six_device1(double* %r, double* %global_data, i32 %gridDim.x, i32 %gridDim.y, i32 %gridDim.z, i32 %blockDim.x, i32 %blockDim.y, i32 %blockDim.z, i32 %blockIdx.x, i32 %blockIdx.y, i32 %blockIdx.z, i32 %threadIdx.x, i32 %threadIdx.y, i32 %threadIdx.z) #10 {
entry:
  call void @llvm.dbg.value(metadata double* %r, metadata !3477, metadata !DIExpression()), !dbg !3479
  call void @llvm.dbg.value(metadata double* %global_data, metadata !3480, metadata !DIExpression()), !dbg !3479
  %sharedMem.gep = getelementptr [1024 x double], [1024 x double]* @extern_share_data_shared, i64 0, i64 0
  call void @llvm.dbg.value(metadata double* %sharedMem.gep, metadata !3481, metadata !DIExpression()), !dbg !3479
  call void @llvm.dbg.value(metadata !1156, metadata !3484, metadata !DIExpression()), !dbg !3479
  call void @llvm.dbg.value(metadata i32 %threadIdx.x, metadata !3485, metadata !DIExpression()), !dbg !3479
  %idxprom = sext i32 %threadIdx.x to i64, !dbg !3486
  %arrayidx = getelementptr inbounds double, double* %sharedMem.gep, i64 %idxprom, !dbg !3486
  store double 0.000000e+00, double* %arrayidx, align 8, !dbg !3487
  br label %syncpoint.1, !dbg !3490

syncpoint.1:                                      ; preds = %entry
  call void @llvm.nvvm.barrier0(), !dbg !3499
  %cmp9 = icmp eq i32 %threadIdx.x, 0, !dbg !3690
  br i1 %cmp9, label %if.then10, label %if.end21, !dbg !3692

if.then10:                                        ; preds = %syncpoint.1
  call void @llvm.dbg.value(metadata i32 1, metadata !3693, metadata !DIExpression()), !dbg !3696
  %0 = zext i32 %blockDim.x to i64, !dbg !3697
  br label %for.cond, !dbg !3697

for.cond:                                         ; preds = %for.inc, %if.then10
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 1, %if.then10 ], !dbg !3696
  call void @llvm.dbg.value(metadata i64 %indvars.iv, metadata !3693, metadata !DIExpression()), !dbg !3696
  %cmp12 = icmp ult i64 %indvars.iv, %0, !dbg !3698
  br i1 %cmp12, label %for.body, label %for.end, !dbg !3700

for.body:                                         ; preds = %for.cond
  %arrayidx14 = getelementptr inbounds double, double* %sharedMem.gep, i64 %indvars.iv, !dbg !3701
  %1 = load double, double* %arrayidx14, align 8, !dbg !3701
  %arrayidx15 = getelementptr inbounds double, double* %sharedMem.gep, i64 0, !dbg !3703
  %2 = load double, double* %arrayidx15, align 8, !dbg !3704
  %add16 = fadd contract double %2, %1, !dbg !3704
  store double %add16, double* %arrayidx15, align 8, !dbg !3704
  br label %for.inc, !dbg !3705

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !3706
  call void @llvm.dbg.value(metadata i32 undef, metadata !3693, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !3696
  br label %for.cond, !dbg !3707, !llvm.loop !3708

for.end:                                          ; preds = %for.cond
  %arrayidx17 = getelementptr inbounds double, double* %sharedMem.gep, i64 0, !dbg !3710
  %3 = load double, double* %arrayidx17, align 8, !dbg !3710
  %idxprom19 = zext i32 %blockIdx.x to i64, !dbg !3711
  %arrayidx20 = getelementptr inbounds double, double* %global_data, i64 %idxprom19, !dbg !3711
  store double %3, double* %arrayidx20, align 8, !dbg !3712
  br label %if.end21, !dbg !3713

if.end21:                                         ; preds = %for.end, %syncpoint.1
  ret void, !dbg !3714
}

; Function Attrs: convergent noinline nounwind
define dso_local void @gpu_kernel_eight_device1(i32* %colidx, i32* %rowstr, double* %a, double* %r, double* %z, i32 %gridDim.x, i32 %gridDim.y, i32 %gridDim.z, i32 %blockDim.x, i32 %blockDim.y, i32 %blockDim.z, i32 %blockIdx.x, i32 %blockIdx.y, i32 %blockIdx.z, i32 %threadIdx.x, i32 %threadIdx.y, i32 %threadIdx.z) #10 {
entry:
  call void @llvm.dbg.value(metadata i32* %colidx, metadata !3520, metadata !DIExpression()), !dbg !3522
  call void @llvm.dbg.value(metadata i32* %rowstr, metadata !3523, metadata !DIExpression()), !dbg !3522
  call void @llvm.dbg.value(metadata double* %a, metadata !3524, metadata !DIExpression()), !dbg !3522
  call void @llvm.dbg.value(metadata double* %r, metadata !3525, metadata !DIExpression()), !dbg !3522
  call void @llvm.dbg.value(metadata double* %z, metadata !3526, metadata !DIExpression()), !dbg !3522
  %sharedMem.gep = getelementptr [1024 x double], [1024 x double]* @extern_share_data_shared, i64 0, i64 0
  call void @llvm.dbg.value(metadata double* %sharedMem.gep, metadata !3527, metadata !DIExpression()), !dbg !3522
  %mul = mul i32 %blockIdx.x, %blockDim.x, !dbg !3528
  %add = add i32 %mul, %threadIdx.x, !dbg !3529
  %div = udiv i32 %add, %blockDim.x, !dbg !3530
  call void @llvm.dbg.value(metadata i32 %div, metadata !3531, metadata !DIExpression()), !dbg !3522
  call void @llvm.dbg.value(metadata i32 %threadIdx.x, metadata !3532, metadata !DIExpression()), !dbg !3522
  call void @llvm.dbg.value(metadata !1156, metadata !3534, metadata !DIExpression()), !dbg !3522
  call void @llvm.dbg.value(metadata double 0.000000e+00, metadata !3538, metadata !DIExpression()), !dbg !3522
  br label %syncpoint.1, !dbg !3543

syncpoint.1:                                      ; preds = %entry
  call void @llvm.nvvm.barrier0(), !dbg !3560
  %cmp21 = icmp eq i32 %threadIdx.x, 0, !dbg !3715
  br i1 %cmp21, label %if.then, label %if.end, !dbg !3717

if.then:                                          ; preds = %syncpoint.1
  call void @llvm.dbg.value(metadata i32 1, metadata !3718, metadata !DIExpression()), !dbg !3721
  %0 = zext i32 %blockDim.x to i64, !dbg !3722
  br label %for.cond22, !dbg !3722

for.cond22:                                       ; preds = %for.inc30, %if.then
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc30 ], [ 1, %if.then ], !dbg !3721
  call void @llvm.dbg.value(metadata i64 %indvars.iv, metadata !3718, metadata !DIExpression()), !dbg !3721
  %cmp24 = icmp ult i64 %indvars.iv, %0, !dbg !3723
  br i1 %cmp24, label %for.body25, label %for.end31, !dbg !3725

for.body25:                                       ; preds = %for.cond22
  %arrayidx27 = getelementptr inbounds double, double* %sharedMem.gep, i64 %indvars.iv, !dbg !3726
  %1 = load double, double* %arrayidx27, align 8, !dbg !3726
  %arrayidx28 = getelementptr inbounds double, double* %sharedMem.gep, i64 0, !dbg !3728
  %2 = load double, double* %arrayidx28, align 8, !dbg !3729
  %add29 = fadd contract double %2, %1, !dbg !3729
  store double %add29, double* %arrayidx28, align 8, !dbg !3729
  br label %for.inc30, !dbg !3730

for.inc30:                                        ; preds = %for.body25
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !3731
  call void @llvm.dbg.value(metadata i32 undef, metadata !3718, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !3721
  br label %for.cond22, !dbg !3732, !llvm.loop !3733

for.end31:                                        ; preds = %for.cond22
  %arrayidx32 = getelementptr inbounds double, double* %sharedMem.gep, i64 0, !dbg !3735
  %3 = load double, double* %arrayidx32, align 8, !dbg !3735
  %idxprom33 = sext i32 %div to i64, !dbg !3736
  %arrayidx34 = getelementptr inbounds double, double* %r, i64 %idxprom33, !dbg !3736
  store double %3, double* %arrayidx34, align 8, !dbg !3737
  br label %if.end, !dbg !3738

if.end:                                           ; preds = %for.end31, %syncpoint.1
  ret void, !dbg !3739
}

; Function Attrs: convergent noinline nounwind
define dso_local void @gpu_kernel_two_device1(double* %r, double* %rho, double* %global_data, i32 %gridDim.x, i32 %gridDim.y, i32 %gridDim.z, i32 %blockDim.x, i32 %blockDim.y, i32 %blockDim.z, i32 %blockIdx.x, i32 %blockIdx.y, i32 %blockIdx.z, i32 %threadIdx.x, i32 %threadIdx.y, i32 %threadIdx.z) #10 {
entry:
  call void @llvm.dbg.value(metadata double* %r, metadata !3346, metadata !DIExpression()), !dbg !3348
  call void @llvm.dbg.value(metadata double* %rho, metadata !3349, metadata !DIExpression()), !dbg !3348
  call void @llvm.dbg.value(metadata double* %global_data, metadata !3350, metadata !DIExpression()), !dbg !3348
  %sharedMem.gep = getelementptr [1024 x double], [1024 x double]* @extern_share_data_shared, i64 0, i64 0
  call void @llvm.dbg.value(metadata double* %sharedMem.gep, metadata !3351, metadata !DIExpression()), !dbg !3348
  call void @llvm.dbg.value(metadata !1156, metadata !3354, metadata !DIExpression()), !dbg !3348
  call void @llvm.dbg.value(metadata i32 %threadIdx.x, metadata !3355, metadata !DIExpression()), !dbg !3348
  %idxprom = sext i32 %threadIdx.x to i64, !dbg !3356
  %arrayidx = getelementptr inbounds double, double* %sharedMem.gep, i64 %idxprom, !dbg !3356
  store double 0.000000e+00, double* %arrayidx, align 8, !dbg !3357
  br label %syncpoint.1, !dbg !3360

syncpoint.1:                                      ; preds = %entry
  call void @llvm.nvvm.barrier0(), !dbg !3369
  %cmp9 = icmp eq i32 %threadIdx.x, 0, !dbg !3740
  br i1 %cmp9, label %if.then10, label %if.end21, !dbg !3742

if.then10:                                        ; preds = %syncpoint.1
  call void @llvm.dbg.value(metadata i32 1, metadata !3743, metadata !DIExpression()), !dbg !3746
  %0 = zext i32 %blockDim.x to i64, !dbg !3747
  br label %for.cond, !dbg !3747

for.cond:                                         ; preds = %for.inc, %if.then10
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 1, %if.then10 ], !dbg !3746
  call void @llvm.dbg.value(metadata i64 %indvars.iv, metadata !3743, metadata !DIExpression()), !dbg !3746
  %cmp12 = icmp ult i64 %indvars.iv, %0, !dbg !3748
  br i1 %cmp12, label %for.body, label %for.end, !dbg !3750

for.body:                                         ; preds = %for.cond
  %arrayidx14 = getelementptr inbounds double, double* %sharedMem.gep, i64 %indvars.iv, !dbg !3751
  %1 = load double, double* %arrayidx14, align 8, !dbg !3751
  %arrayidx15 = getelementptr inbounds double, double* %sharedMem.gep, i64 0, !dbg !3753
  %2 = load double, double* %arrayidx15, align 8, !dbg !3754
  %add16 = fadd contract double %2, %1, !dbg !3754
  store double %add16, double* %arrayidx15, align 8, !dbg !3754
  br label %for.inc, !dbg !3755

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !3756
  call void @llvm.dbg.value(metadata i32 undef, metadata !3743, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !3746
  br label %for.cond, !dbg !3757, !llvm.loop !3758

for.end:                                          ; preds = %for.cond
  %arrayidx17 = getelementptr inbounds double, double* %sharedMem.gep, i64 0, !dbg !3760
  %3 = load double, double* %arrayidx17, align 8, !dbg !3760
  %idxprom19 = zext i32 %blockIdx.x to i64, !dbg !3761
  %arrayidx20 = getelementptr inbounds double, double* %global_data, i64 %idxprom19, !dbg !3761
  store double %3, double* %arrayidx20, align 8, !dbg !3762
  br label %if.end21, !dbg !3763

if.end21:                                         ; preds = %for.end, %syncpoint.1
  ret void, !dbg !3764
}

; Function Attrs: convergent noinline nounwind
define dso_local void @gpu_kernel_four_device1(double* %d, double* %p, double* %q, double* %global_data, i32 %gridDim.x, i32 %gridDim.y, i32 %gridDim.z, i32 %blockDim.x, i32 %blockDim.y, i32 %blockDim.z, i32 %blockIdx.x, i32 %blockIdx.y, i32 %blockIdx.z, i32 %threadIdx.x, i32 %threadIdx.y, i32 %threadIdx.z) #10 {
entry:
  call void @llvm.dbg.value(metadata double* %d, metadata !3413, metadata !DIExpression()), !dbg !3417
  call void @llvm.dbg.value(metadata double* %p, metadata !3418, metadata !DIExpression()), !dbg !3417
  call void @llvm.dbg.value(metadata double* %q, metadata !3419, metadata !DIExpression()), !dbg !3417
  call void @llvm.dbg.value(metadata double* %global_data, metadata !3420, metadata !DIExpression()), !dbg !3417
  %sharedMem.gep = getelementptr [1024 x double], [1024 x double]* @extern_share_data_shared, i64 0, i64 0
  call void @llvm.dbg.value(metadata double* %sharedMem.gep, metadata !3421, metadata !DIExpression()), !dbg !3417
  call void @llvm.dbg.value(metadata !1156, metadata !3424, metadata !DIExpression()), !dbg !3417
  call void @llvm.dbg.value(metadata i32 %threadIdx.x, metadata !3425, metadata !DIExpression()), !dbg !3417
  %idxprom = sext i32 %threadIdx.x to i64, !dbg !3426
  %arrayidx = getelementptr inbounds double, double* %sharedMem.gep, i64 %idxprom, !dbg !3426
  store double 0.000000e+00, double* %arrayidx, align 8, !dbg !3427
  %idxprom4 = sext i32 %threadIdx.x to i64, !dbg !3428
  %arrayidx5 = getelementptr inbounds double, double* %sharedMem.gep, i64 %idxprom4, !dbg !3428
  store double 0.000000e+00, double* %arrayidx5, align 8, !dbg !3429
  br label %syncpoint.1, !dbg !3432

syncpoint.1:                                      ; preds = %entry
  call void @llvm.nvvm.barrier0(), !dbg !3440
  %cmp14 = icmp eq i32 %threadIdx.x, 0, !dbg !3765
  br i1 %cmp14, label %if.then15, label %if.end26, !dbg !3767

if.then15:                                        ; preds = %syncpoint.1
  call void @llvm.dbg.value(metadata i32 1, metadata !3768, metadata !DIExpression()), !dbg !3771
  %0 = zext i32 %blockDim.x to i64, !dbg !3772
  br label %for.cond, !dbg !3772

for.cond:                                         ; preds = %for.inc, %if.then15
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 1, %if.then15 ], !dbg !3771
  call void @llvm.dbg.value(metadata i64 %indvars.iv, metadata !3768, metadata !DIExpression()), !dbg !3771
  %cmp17 = icmp ult i64 %indvars.iv, %0, !dbg !3773
  br i1 %cmp17, label %for.body, label %for.end, !dbg !3775

for.body:                                         ; preds = %for.cond
  %arrayidx19 = getelementptr inbounds double, double* %sharedMem.gep, i64 %indvars.iv, !dbg !3776
  %1 = load double, double* %arrayidx19, align 8, !dbg !3776
  %arrayidx20 = getelementptr inbounds double, double* %sharedMem.gep, i64 0, !dbg !3778
  %2 = load double, double* %arrayidx20, align 8, !dbg !3779
  %add21 = fadd contract double %2, %1, !dbg !3779
  store double %add21, double* %arrayidx20, align 8, !dbg !3779
  br label %for.inc, !dbg !3780

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !3781
  call void @llvm.dbg.value(metadata i32 undef, metadata !3768, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !3771
  br label %for.cond, !dbg !3782, !llvm.loop !3783

for.end:                                          ; preds = %for.cond
  %arrayidx22 = getelementptr inbounds double, double* %sharedMem.gep, i64 0, !dbg !3785
  %3 = load double, double* %arrayidx22, align 8, !dbg !3785
  %idxprom24 = zext i32 %blockIdx.x to i64, !dbg !3786
  %arrayidx25 = getelementptr inbounds double, double* %global_data, i64 %idxprom24, !dbg !3786
  store double %3, double* %arrayidx25, align 8, !dbg !3787
  br label %if.end26, !dbg !3788

if.end26:                                         ; preds = %for.end, %syncpoint.1
  ret void, !dbg !3789
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #0

attributes #0 = { nounwind readnone speculatable }
attributes #1 = { nounwind readnone }
attributes #2 = { convergent nounwind }
attributes #3 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { noinline uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #7 = { noinline norecurse uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #8 = { argmemonly nounwind }
attributes #9 = { noreturn nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #10 = { convergent noinline nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #11 = { nounwind }
attributes #12 = { noreturn nounwind }

!llvm.dbg.cu = !{!1155, !2}
!nvvm.annotations = !{!1229, !1230, !1231, !1232, !1233, !1234, !1235, !1236, !1237, !1238, !1239, !1240, !1241, !1242, !1243, !1242, !1244, !1244, !1244, !1244, !1245, !1245, !1244}
!llvm.ident = !{!1246, !1246}
!nvvm.internalize.after.link = !{}
!nvvmir.version = !{!1247}
!llvm.module.flags = !{!1248, !1249, !1250, !1251, !1252}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "colidx_device", scope: !2, file: !3, line: 118, type: !98, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 9.0.0 (https://github.com/yebinchon/llvm-project/ a1efa594106d738d0b74c9e4e2b9b779eb8b7d25)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !96, globals: !111, imports: !404, nameTableKind: None)
!3 = !DIFile(filename: "cg.cu", directory: "/scratch/ah7226/NPB-GPU/CUDA/CG")
!4 = !{!5, !14}
!5 = distinct !DICompositeType(tag: DW_TAG_enumeration_type, name: "cudaMemcpyKind", file: !6, line: 796, baseType: !7, size: 32, elements: !8, identifier: "_ZTS14cudaMemcpyKind")
!6 = !DIFile(filename: "/usr/local/cuda/include/driver_types.h", directory: "")
!7 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!8 = !{!9, !10, !11, !12, !13}
!9 = !DIEnumerator(name: "cudaMemcpyHostToHost", value: 0, isUnsigned: true)
!10 = !DIEnumerator(name: "cudaMemcpyHostToDevice", value: 1, isUnsigned: true)
!11 = !DIEnumerator(name: "cudaMemcpyDeviceToHost", value: 2, isUnsigned: true)
!12 = !DIEnumerator(name: "cudaMemcpyDeviceToDevice", value: 3, isUnsigned: true)
!13 = !DIEnumerator(name: "cudaMemcpyDefault", value: 4, isUnsigned: true)
!14 = distinct !DICompositeType(tag: DW_TAG_enumeration_type, name: "cudaError", file: !6, line: 150, baseType: !7, size: 32, elements: !15, identifier: "_ZTS9cudaError")
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
!155 = !DIFile(filename: "/u/NAS_SCRATCH/ah7226/tulip/llvm-install-tulip/lib/clang/9.0.0/include/stddef.h", directory: "")
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
!408 = !DIFile(filename: "/u/NAS_SCRATCH/ah7226/tulip/llvm-install-tulip/lib/clang/9.0.0/include/__clang_cuda_math_forward_declares.h", directory: "")
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
!656 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !596, line: 59, flags: DIFlagFwdDecl, identifier: "_ZTS5div_t")
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
!868 = !DIFile(filename: "/u/NAS_SCRATCH/ah7226/tulip/llvm-install-tulip/lib/clang/9.0.0/include/__clang_cuda_cmath.h", directory: "")
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
!981 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "_IO_FILE", file: !982, line: 49, flags: DIFlagFwdDecl, identifier: "_ZTS8_IO_FILE")
!982 = !DIFile(filename: "/usr/include/x86_64-linux-gnu/bits/types/struct_FILE.h", directory: "")
!983 = !DIFile(filename: "/usr/lib/gcc/x86_64-linux-gnu/11/../../../../include/c++/11/cstdio", directory: "")
!984 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !985, file: !983, line: 99)
!985 = !DIDerivedType(tag: DW_TAG_typedef, name: "fpos_t", file: !986, line: 84, baseType: !987)
!986 = !DIFile(filename: "/usr/include/stdio.h", directory: "")
!987 = !DIDerivedType(tag: DW_TAG_typedef, name: "__fpos_t", file: !988, line: 14, baseType: !989)
!988 = !DIFile(filename: "/usr/include/x86_64-linux-gnu/bits/types/__fpos_t.h", directory: "")
!989 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "_G_fpos_t", file: !988, line: 10, flags: DIFlagFwdDecl, identifier: "_ZTS9_G_fpos_t")
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
!1125 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "__va_list_tag", file: !3, flags: DIFlagFwdDecl, identifier: "_ZTS13__va_list_tag")
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
!1155 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 9.0.0 (https://github.com/yebinchon/llvm-project/ a1efa594106d738d0b74c9e4e2b9b779eb8b7d25)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !1156, retainedTypes: !1157, imports: !1158, nameTableKind: None)
!1156 = !{}
!1157 = !{!99, !97}
!1158 = !{!405, !411, !416, !418, !420, !422, !424, !428, !430, !432, !434, !436, !438, !440, !442, !444, !446, !448, !450, !452, !454, !456, !460, !462, !464, !466, !470, !474, !476, !478, !483, !487, !489, !491, !493, !495, !497, !499, !501, !503, !508, !512, !514, !519, !523, !525, !527, !529, !531, !533, !537, !539, !541, !546, !552, !556, !558, !560, !562, !564, !568, !570, !572, !576, !578, !580, !582, !584, !586, !588, !590, !592, !594, !598, !604, !606, !608, !612, !614, !616, !618, !620, !622, !624, !626, !630, !634, !636, !638, !642, !644, !646, !648, !650, !652, !654, !658, !664, !668, !673, !675, !679, !683, !693, !697, !701, !705, !709, !713, !715, !719, !723, !727, !735, !739, !743, !747, !751, !755, !761, !765, !769, !771, !779, !783, !790, !792, !794, !798, !802, !806, !811, !1159, !820, !821, !822, !823, !825, !826, !827, !828, !829, !1164, !1165, !1166, !1167, !1168, !1169, !1170, !1174, !1175, !1176, !1177, !1178, !1179, !1180, !1181, !1182, !1183, !1184, !1185, !1186, !1187, !1188, !1189, !1190, !1191, !1192, !1193, !1194, !1195, !1196, !1197, !865, !869, !871, !873, !875, !877, !879, !881, !883, !886, !888, !890, !892, !894, !896, !898, !900, !902, !904, !906, !908, !910, !912, !914, !916, !918, !920, !922, !924, !926, !928, !930, !932, !934, !936, !938, !940, !942, !944, !946, !948, !950, !952, !954, !956, !958, !960, !962, !964, !966, !968, !970, !972, !974, !976, !978, !984, !990, !995, !999, !1001, !1003, !1005, !1007, !1014, !1018, !1022, !1026, !1030, !1034, !1039, !1043, !1045, !1049, !1055, !1059, !1064, !1066, !1068, !1072, !1076, !1080, !1082, !1084, !1086, !1088, !1092, !1094, !1096, !1100, !1104, !1108, !1112, !1116, !1118, !1198, !1205, !1209, !1134, !1213, !1215, !1217, !1221, !1150, !1225, !1226, !1227, !1228}
!1159 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !784, entity: !1160, file: !657, line: 232)
!1160 = !DISubprogram(name: "strtold", scope: !596, file: !596, line: 127, type: !1161, flags: DIFlagPrototyped, spFlags: 0)
!1161 = !DISubroutineType(types: !1162)
!1162 = !{!1163, !734, !759}
!1163 = !DIBasicType(name: "long double", size: 64, encoding: DW_ATE_float)
!1164 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !1160, file: !657, line: 252)
!1165 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !1155, entity: !665, file: !832, line: 38)
!1166 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !1155, entity: !669, file: !832, line: 39)
!1167 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !1155, entity: !702, file: !832, line: 40)
!1168 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !1155, entity: !655, file: !832, line: 51)
!1169 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !1155, entity: !659, file: !832, line: 52)
!1170 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !1155, entity: !1171, file: !832, line: 54)
!1171 = !DISubprogram(name: "abs", linkageName: "_ZSt3abse", scope: !406, file: !597, line: 79, type: !1172, flags: DIFlagPrototyped, spFlags: 0)
!1172 = !DISubroutineType(types: !1173)
!1173 = !{!1163, !1163}
!1174 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !1155, entity: !674, file: !832, line: 55)
!1175 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !1155, entity: !676, file: !832, line: 56)
!1176 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !1155, entity: !680, file: !832, line: 57)
!1177 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !1155, entity: !684, file: !832, line: 58)
!1178 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !1155, entity: !694, file: !832, line: 59)
!1179 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !1155, entity: !824, file: !832, line: 60)
!1180 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !1155, entity: !706, file: !832, line: 61)
!1181 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !1155, entity: !710, file: !832, line: 62)
!1182 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !1155, entity: !714, file: !832, line: 63)
!1183 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !1155, entity: !716, file: !832, line: 64)
!1184 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !1155, entity: !720, file: !832, line: 65)
!1185 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !1155, entity: !724, file: !832, line: 67)
!1186 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !1155, entity: !728, file: !832, line: 68)
!1187 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !1155, entity: !736, file: !832, line: 69)
!1188 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !1155, entity: !740, file: !832, line: 71)
!1189 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !1155, entity: !744, file: !832, line: 72)
!1190 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !1155, entity: !748, file: !832, line: 73)
!1191 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !1155, entity: !752, file: !832, line: 74)
!1192 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !1155, entity: !756, file: !832, line: 75)
!1193 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !1155, entity: !762, file: !832, line: 76)
!1194 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !1155, entity: !766, file: !832, line: 77)
!1195 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !1155, entity: !770, file: !832, line: 78)
!1196 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !1155, entity: !772, file: !832, line: 80)
!1197 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !1155, entity: !780, file: !832, line: 81)
!1198 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !1199, file: !983, line: 144)
!1199 = !DISubprogram(name: "vfprintf", scope: !986, file: !986, line: 365, type: !1200, flags: DIFlagPrototyped, spFlags: 0)
!1200 = !DISubroutineType(types: !1201)
!1201 = !{!97, !1011, !734, !1202}
!1202 = !DIDerivedType(tag: DW_TAG_typedef, name: "__gnuc_va_list", file: !1203, line: 32, baseType: !1204)
!1203 = !DIFile(filename: "/u/NAS_SCRATCH/ah7226/tulip/llvm-install-tulip/lib/clang/9.0.0/include/stdarg.h", directory: "")
!1204 = !DIDerivedType(tag: DW_TAG_typedef, name: "__builtin_va_list", file: !3, baseType: !108)
!1205 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !1206, file: !983, line: 145)
!1206 = !DISubprogram(name: "vprintf", scope: !986, file: !986, line: 371, type: !1207, flags: DIFlagPrototyped, spFlags: 0)
!1207 = !DISubroutineType(types: !1208)
!1208 = !{!97, !734, !1202}
!1209 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !1210, file: !983, line: 146)
!1210 = !DISubprogram(name: "vsprintf", scope: !986, file: !986, line: 373, type: !1211, flags: DIFlagPrototyped, spFlags: 0)
!1211 = !DISubroutineType(types: !1212)
!1212 = !{!97, !775, !734, !1202}
!1213 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !784, entity: !1214, file: !983, line: 176)
!1214 = !DISubprogram(name: "vfscanf", scope: !986, file: !986, line: 459, type: !1200, flags: DIFlagPrototyped, spFlags: 0)
!1215 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !784, entity: !1216, file: !983, line: 177)
!1216 = !DISubprogram(name: "vscanf", scope: !986, file: !986, line: 467, type: !1207, flags: DIFlagPrototyped, spFlags: 0)
!1217 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !784, entity: !1218, file: !983, line: 178)
!1218 = !DISubprogram(name: "vsnprintf", scope: !986, file: !986, line: 382, type: !1219, flags: DIFlagPrototyped, spFlags: 0)
!1219 = !DISubroutineType(types: !1220)
!1220 = !{!97, !775, !154, !734, !1202}
!1221 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !784, entity: !1222, file: !983, line: 179)
!1222 = !DISubprogram(name: "vsscanf", scope: !986, file: !986, line: 471, type: !1223, flags: DIFlagPrototyped, spFlags: 0)
!1223 = !DISubroutineType(types: !1224)
!1224 = !{!97, !734, !734, !1202}
!1225 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !1214, file: !983, line: 186)
!1226 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !1216, file: !983, line: 187)
!1227 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !1218, file: !983, line: 188)
!1228 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !406, entity: !1222, file: !983, line: 189)
!1229 = distinct !{null, !"kernel", i32 1}
!1230 = distinct !{null, !"kernel", i32 1}
!1231 = distinct !{null, !"kernel", i32 1}
!1232 = distinct !{null, !"kernel", i32 1}
!1233 = distinct !{null, !"kernel", i32 1}
!1234 = distinct !{null, !"kernel", i32 1}
!1235 = distinct !{null, !"kernel", i32 1}
!1236 = distinct !{null, !"kernel", i32 1}
!1237 = distinct !{null, !"kernel", i32 1}
!1238 = distinct !{null, !"kernel", i32 1}
!1239 = distinct !{null, !"kernel", i32 1}
!1240 = distinct !{null, !"kernel", i32 1}
!1241 = distinct !{null, !"kernel", i32 1}
!1242 = !{null, !"align", i32 8}
!1243 = !{null, !"align", i32 8, !"align", i32 65544, !"align", i32 131080}
!1244 = !{null, !"align", i32 16}
!1245 = !{null, !"align", i32 16, !"align", i32 65552, !"align", i32 131088}
!1246 = !{!"clang version 9.0.0 (https://github.com/yebinchon/llvm-project/ a1efa594106d738d0b74c9e4e2b9b779eb8b7d25)"}
!1247 = !{i32 1, i32 2}
!1248 = !{i32 2, !"SDK Version", [2 x i32] [i32 7, i32 0]}
!1249 = !{i32 2, !"Dwarf Version", i32 2}
!1250 = !{i32 2, !"Debug Info Version", i32 3}
!1251 = !{i32 1, !"wchar_size", i32 4}
!1252 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!1253 = distinct !DISubprogram(name: "randlc", linkageName: "_Z6randlcPdd", scope: !3, file: !3, line: 364, type: !1254, scopeLine: 364, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !1156)
!1254 = !DISubroutineType(types: !1255)
!1255 = !{!100, !99, !100}
!1256 = !DILocalVariable(name: "x", arg: 1, scope: !1253, file: !3, line: 364, type: !99)
!1257 = !DILocation(line: 0, scope: !1253)
!1258 = !DILocalVariable(name: "a", arg: 2, scope: !1253, file: !3, line: 364, type: !100)
!1259 = !DILocation(line: 372, column: 11, scope: !1253)
!1260 = !DILocalVariable(name: "t1", scope: !1253, file: !3, line: 365, type: !100)
!1261 = !DILocation(line: 373, column: 12, scope: !1253)
!1262 = !DILocation(line: 373, column: 7, scope: !1253)
!1263 = !DILocalVariable(name: "a1", scope: !1253, file: !3, line: 365, type: !100)
!1264 = !DILocation(line: 374, column: 15, scope: !1253)
!1265 = !DILocation(line: 374, column: 9, scope: !1253)
!1266 = !DILocalVariable(name: "a2", scope: !1253, file: !3, line: 365, type: !100)
!1267 = !DILocation(line: 383, column: 14, scope: !1253)
!1268 = !DILocation(line: 383, column: 11, scope: !1253)
!1269 = !DILocation(line: 384, column: 12, scope: !1253)
!1270 = !DILocation(line: 384, column: 7, scope: !1253)
!1271 = !DILocalVariable(name: "x1", scope: !1253, file: !3, line: 365, type: !100)
!1272 = !DILocation(line: 385, column: 8, scope: !1253)
!1273 = !DILocation(line: 385, column: 18, scope: !1253)
!1274 = !DILocation(line: 385, column: 12, scope: !1253)
!1275 = !DILocalVariable(name: "x2", scope: !1253, file: !3, line: 365, type: !100)
!1276 = !DILocation(line: 386, column: 10, scope: !1253)
!1277 = !DILocation(line: 386, column: 20, scope: !1253)
!1278 = !DILocation(line: 386, column: 15, scope: !1253)
!1279 = !DILocation(line: 387, column: 17, scope: !1253)
!1280 = !DILocation(line: 387, column: 12, scope: !1253)
!1281 = !DILocation(line: 387, column: 7, scope: !1253)
!1282 = !DILocalVariable(name: "t2", scope: !1253, file: !3, line: 365, type: !100)
!1283 = !DILocation(line: 388, column: 15, scope: !1253)
!1284 = !DILocation(line: 388, column: 9, scope: !1253)
!1285 = !DILocalVariable(name: "z", scope: !1253, file: !3, line: 365, type: !100)
!1286 = !DILocation(line: 389, column: 11, scope: !1253)
!1287 = !DILocation(line: 389, column: 20, scope: !1253)
!1288 = !DILocation(line: 389, column: 15, scope: !1253)
!1289 = !DILocalVariable(name: "t3", scope: !1253, file: !3, line: 365, type: !100)
!1290 = !DILocation(line: 390, column: 17, scope: !1253)
!1291 = !DILocation(line: 390, column: 12, scope: !1253)
!1292 = !DILocation(line: 390, column: 7, scope: !1253)
!1293 = !DILocalVariable(name: "t4", scope: !1253, file: !3, line: 365, type: !100)
!1294 = !DILocation(line: 391, column: 18, scope: !1253)
!1295 = !DILocation(line: 391, column: 12, scope: !1253)
!1296 = !DILocation(line: 391, column: 7, scope: !1253)
!1297 = !DILocation(line: 393, column: 17, scope: !1253)
!1298 = !DILocation(line: 393, column: 14, scope: !1253)
!1299 = !DILocation(line: 393, column: 2, scope: !1253)
!1300 = distinct !DISubprogram(name: "c_print_results", linkageName: "_Z15c_print_resultsPcciiiiddS_iS_S_S_S_S_S_S_S_S_S_S_S_S_S_", scope: !3, file: !3, line: 399, type: !1301, scopeLine: 422, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !1156)
!1301 = !DISubroutineType(types: !1302)
!1302 = !{null, !108, !109, !97, !97, !97, !97, !100, !100, !108, !97, !108, !108, !108, !108, !108, !108, !108, !108, !108, !108, !108, !108, !108, !108}
!1303 = !DILocalVariable(name: "name", arg: 1, scope: !1300, file: !3, line: 399, type: !108)
!1304 = !DILocation(line: 0, scope: !1300)
!1305 = !DILocalVariable(name: "class_npb", arg: 2, scope: !1300, file: !3, line: 400, type: !109)
!1306 = !DILocalVariable(name: "n1", arg: 3, scope: !1300, file: !3, line: 401, type: !97)
!1307 = !DILocalVariable(name: "n2", arg: 4, scope: !1300, file: !3, line: 402, type: !97)
!1308 = !DILocalVariable(name: "n3", arg: 5, scope: !1300, file: !3, line: 403, type: !97)
!1309 = !DILocalVariable(name: "niter", arg: 6, scope: !1300, file: !3, line: 404, type: !97)
!1310 = !DILocalVariable(name: "t", arg: 7, scope: !1300, file: !3, line: 405, type: !100)
!1311 = !DILocalVariable(name: "mops", arg: 8, scope: !1300, file: !3, line: 406, type: !100)
!1312 = !DILocalVariable(name: "optype", arg: 9, scope: !1300, file: !3, line: 407, type: !108)
!1313 = !DILocalVariable(name: "passed_verification", arg: 10, scope: !1300, file: !3, line: 408, type: !97)
!1314 = !DILocalVariable(name: "npbversion", arg: 11, scope: !1300, file: !3, line: 409, type: !108)
!1315 = !DILocalVariable(name: "compiletime", arg: 12, scope: !1300, file: !3, line: 410, type: !108)
!1316 = !DILocalVariable(name: "compilerversion", arg: 13, scope: !1300, file: !3, line: 411, type: !108)
!1317 = !DILocalVariable(name: "libversion", arg: 14, scope: !1300, file: !3, line: 412, type: !108)
!1318 = !DILocalVariable(name: "cpu_device", arg: 15, scope: !1300, file: !3, line: 413, type: !108)
!1319 = !DILocalVariable(name: "gpu_device", arg: 16, scope: !1300, file: !3, line: 414, type: !108)
!1320 = !DILocalVariable(name: "gpu_config", arg: 17, scope: !1300, file: !3, line: 415, type: !108)
!1321 = !DILocalVariable(name: "cc", arg: 18, scope: !1300, file: !3, line: 416, type: !108)
!1322 = !DILocalVariable(name: "clink", arg: 19, scope: !1300, file: !3, line: 417, type: !108)
!1323 = !DILocalVariable(name: "c_lib", arg: 20, scope: !1300, file: !3, line: 418, type: !108)
!1324 = !DILocalVariable(name: "c_inc", arg: 21, scope: !1300, file: !3, line: 419, type: !108)
!1325 = !DILocalVariable(name: "cflags", arg: 22, scope: !1300, file: !3, line: 420, type: !108)
!1326 = !DILocalVariable(name: "clinkflags", arg: 23, scope: !1300, file: !3, line: 421, type: !108)
!1327 = !DILocalVariable(name: "rand", arg: 24, scope: !1300, file: !3, line: 422, type: !108)
!1328 = !DILocation(line: 423, column: 4, scope: !1300)
!1329 = !DILocation(line: 424, column: 61, scope: !1300)
!1330 = !DILocation(line: 424, column: 4, scope: !1300)
!1331 = !DILocation(line: 425, column: 8, scope: !1332)
!1332 = distinct !DILexicalBlock(scope: !1300, file: !3, line: 425, column: 7)
!1333 = !DILocation(line: 425, column: 15, scope: !1332)
!1334 = !DILocation(line: 425, column: 21, scope: !1332)
!1335 = !DILocation(line: 425, column: 24, scope: !1332)
!1336 = !DILocation(line: 425, column: 31, scope: !1332)
!1337 = !DILocation(line: 425, column: 7, scope: !1300)
!1338 = !DILocation(line: 426, column: 10, scope: !1339)
!1339 = distinct !DILexicalBlock(scope: !1340, file: !3, line: 426, column: 8)
!1340 = distinct !DILexicalBlock(scope: !1332, file: !3, line: 425, column: 38)
!1341 = !DILocation(line: 426, column: 8, scope: !1340)
!1342 = !DILocation(line: 427, column: 16, scope: !1343)
!1343 = distinct !DILexicalBlock(scope: !1339, file: !3, line: 426, column: 14)
!1344 = !DILocalVariable(name: "nn", scope: !1343, file: !3, line: 427, type: !507)
!1345 = !DILocation(line: 0, scope: !1343)
!1346 = !DILocation(line: 428, column: 11, scope: !1347)
!1347 = distinct !DILexicalBlock(scope: !1343, file: !3, line: 428, column: 9)
!1348 = !DILocation(line: 428, column: 9, scope: !1343)
!1349 = !DILocation(line: 428, column: 20, scope: !1350)
!1350 = distinct !DILexicalBlock(scope: !1347, file: !3, line: 428, column: 15)
!1351 = !DILocation(line: 428, column: 18, scope: !1350)
!1352 = !DILocation(line: 428, column: 23, scope: !1350)
!1353 = !DILocation(line: 429, column: 6, scope: !1343)
!1354 = !DILocation(line: 430, column: 5, scope: !1343)
!1355 = !DILocation(line: 431, column: 6, scope: !1356)
!1356 = distinct !DILexicalBlock(scope: !1339, file: !3, line: 430, column: 10)
!1357 = !DILocation(line: 433, column: 4, scope: !1340)
!1358 = !DILocalVariable(name: "size", scope: !1359, file: !3, line: 434, type: !1360)
!1359 = distinct !DILexicalBlock(scope: !1332, file: !3, line: 433, column: 9)
!1360 = !DICompositeType(tag: DW_TAG_array_type, baseType: !109, size: 128, elements: !1361)
!1361 = !{!1362}
!1362 = !DISubrange(count: 16)
!1363 = !DILocation(line: 434, column: 10, scope: !1359)
!1364 = !DILocation(line: 436, column: 11, scope: !1365)
!1365 = distinct !DILexicalBlock(scope: !1359, file: !3, line: 436, column: 8)
!1366 = !DILocation(line: 436, column: 16, scope: !1365)
!1367 = !DILocation(line: 436, column: 22, scope: !1365)
!1368 = !DILocation(line: 436, column: 8, scope: !1359)
!1369 = !DILocation(line: 437, column: 10, scope: !1370)
!1370 = distinct !DILexicalBlock(scope: !1371, file: !3, line: 437, column: 9)
!1371 = distinct !DILexicalBlock(scope: !1365, file: !3, line: 436, column: 27)
!1372 = !DILocation(line: 437, column: 17, scope: !1370)
!1373 = !DILocation(line: 437, column: 23, scope: !1370)
!1374 = !DILocation(line: 437, column: 26, scope: !1370)
!1375 = !DILocation(line: 437, column: 33, scope: !1370)
!1376 = !DILocation(line: 437, column: 9, scope: !1371)
!1377 = !DILocation(line: 438, column: 15, scope: !1378)
!1378 = distinct !DILexicalBlock(scope: !1370, file: !3, line: 437, column: 40)
!1379 = !DILocation(line: 438, column: 41, scope: !1378)
!1380 = !DILocation(line: 438, column: 32, scope: !1378)
!1381 = !DILocation(line: 438, column: 7, scope: !1378)
!1382 = !DILocalVariable(name: "j", scope: !1359, file: !3, line: 435, type: !97)
!1383 = !DILocation(line: 0, scope: !1359)
!1384 = !DILocation(line: 440, column: 10, scope: !1385)
!1385 = distinct !DILexicalBlock(scope: !1378, file: !3, line: 440, column: 10)
!1386 = !DILocation(line: 440, column: 18, scope: !1385)
!1387 = !DILocation(line: 440, column: 10, scope: !1378)
!1388 = !DILocation(line: 441, column: 8, scope: !1389)
!1389 = distinct !DILexicalBlock(scope: !1385, file: !3, line: 440, column: 25)
!1390 = !DILocation(line: 441, column: 16, scope: !1389)
!1391 = !DILocation(line: 442, column: 9, scope: !1389)
!1392 = !DILocation(line: 443, column: 7, scope: !1389)
!1393 = !DILocation(line: 0, scope: !1378)
!1394 = !DILocation(line: 444, column: 13, scope: !1378)
!1395 = !DILocation(line: 444, column: 7, scope: !1378)
!1396 = !DILocation(line: 444, column: 17, scope: !1378)
!1397 = !DILocation(line: 445, column: 52, scope: !1378)
!1398 = !DILocation(line: 445, column: 7, scope: !1378)
!1399 = !DILocation(line: 446, column: 6, scope: !1378)
!1400 = !DILocation(line: 447, column: 7, scope: !1401)
!1401 = distinct !DILexicalBlock(scope: !1370, file: !3, line: 446, column: 11)
!1402 = !DILocation(line: 449, column: 5, scope: !1371)
!1403 = !DILocation(line: 450, column: 6, scope: !1404)
!1404 = distinct !DILexicalBlock(scope: !1365, file: !3, line: 449, column: 10)
!1405 = !DILocation(line: 453, column: 4, scope: !1300)
!1406 = !DILocation(line: 454, column: 4, scope: !1300)
!1407 = !DILocation(line: 455, column: 4, scope: !1300)
!1408 = !DILocation(line: 456, column: 4, scope: !1300)
!1409 = !DILocation(line: 457, column: 27, scope: !1410)
!1410 = distinct !DILexicalBlock(scope: !1300, file: !3, line: 457, column: 7)
!1411 = !DILocation(line: 457, column: 7, scope: !1300)
!1412 = !DILocation(line: 458, column: 5, scope: !1413)
!1413 = distinct !DILexicalBlock(scope: !1410, file: !3, line: 457, column: 31)
!1414 = !DILocation(line: 459, column: 4, scope: !1413)
!1415 = !DILocation(line: 459, column: 13, scope: !1416)
!1416 = distinct !DILexicalBlock(scope: !1410, file: !3, line: 459, column: 13)
!1417 = !DILocation(line: 459, column: 13, scope: !1410)
!1418 = !DILocation(line: 460, column: 5, scope: !1419)
!1419 = distinct !DILexicalBlock(scope: !1416, file: !3, line: 459, column: 33)
!1420 = !DILocation(line: 461, column: 4, scope: !1419)
!1421 = !DILocation(line: 462, column: 5, scope: !1422)
!1422 = distinct !DILexicalBlock(scope: !1416, file: !3, line: 461, column: 9)
!1423 = !DILocation(line: 464, column: 4, scope: !1300)
!1424 = !DILocation(line: 465, column: 4, scope: !1300)
!1425 = !DILocation(line: 466, column: 4, scope: !1300)
!1426 = !DILocation(line: 467, column: 4, scope: !1300)
!1427 = !DILocation(line: 468, column: 4, scope: !1300)
!1428 = !DILocation(line: 469, column: 4, scope: !1300)
!1429 = !DILocation(line: 470, column: 4, scope: !1300)
!1430 = !DILocation(line: 471, column: 4, scope: !1300)
!1431 = !DILocation(line: 472, column: 4, scope: !1300)
!1432 = !DILocation(line: 473, column: 4, scope: !1300)
!1433 = !DILocation(line: 474, column: 4, scope: !1300)
!1434 = !DILocation(line: 475, column: 4, scope: !1300)
!1435 = !DILocation(line: 476, column: 4, scope: !1300)
!1436 = !DILocation(line: 477, column: 4, scope: !1300)
!1437 = !DILocation(line: 478, column: 4, scope: !1300)
!1438 = !DILocation(line: 479, column: 4, scope: !1300)
!1439 = !DILocation(line: 480, column: 4, scope: !1300)
!1440 = !DILocation(line: 495, column: 4, scope: !1300)
!1441 = !DILocation(line: 496, column: 4, scope: !1300)
!1442 = !DILocation(line: 497, column: 4, scope: !1300)
!1443 = !DILocation(line: 498, column: 4, scope: !1300)
!1444 = !DILocation(line: 499, column: 4, scope: !1300)
!1445 = !DILocation(line: 500, column: 4, scope: !1300)
!1446 = !DILocation(line: 501, column: 4, scope: !1300)
!1447 = !DILocation(line: 502, column: 4, scope: !1300)
!1448 = !DILocation(line: 503, column: 4, scope: !1300)
!1449 = !DILocation(line: 504, column: 4, scope: !1300)
!1450 = !DILocation(line: 505, column: 3, scope: !1300)
!1451 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 509, type: !1452, scopeLine: 509, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !1156)
!1452 = !DISubroutineType(types: !1453)
!1453 = !{!97, !97, !760}
!1454 = !DILocalVariable(name: "argc", arg: 1, scope: !1451, file: !3, line: 509, type: !97)
!1455 = !DILocation(line: 0, scope: !1451)
!1456 = !DILocalVariable(name: "argv", arg: 2, scope: !1451, file: !3, line: 509, type: !760)
!1457 = !DILocation(line: 511, column: 17, scope: !1451)
!1458 = !{i64 8064000}
!1459 = !DILocation(line: 511, column: 11, scope: !1451)
!1460 = !DILocation(line: 511, column: 9, scope: !1451)
!1461 = !DILocation(line: 512, column: 17, scope: !1451)
!1462 = !{i64 56004}
!1463 = !DILocation(line: 512, column: 11, scope: !1451)
!1464 = !DILocation(line: 512, column: 9, scope: !1451)
!1465 = !DILocation(line: 513, column: 13, scope: !1451)
!1466 = !DILocation(line: 513, column: 7, scope: !1451)
!1467 = !DILocation(line: 513, column: 5, scope: !1451)
!1468 = !DILocation(line: 514, column: 15, scope: !1451)
!1469 = !DILocation(line: 514, column: 9, scope: !1451)
!1470 = !DILocation(line: 514, column: 7, scope: !1451)
!1471 = !DILocation(line: 515, column: 15, scope: !1451)
!1472 = !DILocation(line: 515, column: 9, scope: !1451)
!1473 = !DILocation(line: 515, column: 7, scope: !1451)
!1474 = !DILocation(line: 516, column: 18, scope: !1451)
!1475 = !DILocation(line: 516, column: 9, scope: !1451)
!1476 = !DILocation(line: 516, column: 7, scope: !1451)
!1477 = !DILocation(line: 517, column: 15, scope: !1451)
!1478 = !{i64 16128000}
!1479 = !DILocation(line: 517, column: 6, scope: !1451)
!1480 = !DILocation(line: 517, column: 4, scope: !1451)
!1481 = !DILocation(line: 518, column: 15, scope: !1451)
!1482 = !{i64 112016}
!1483 = !DILocation(line: 518, column: 6, scope: !1451)
!1484 = !DILocation(line: 518, column: 4, scope: !1451)
!1485 = !DILocation(line: 519, column: 15, scope: !1451)
!1486 = !DILocation(line: 519, column: 6, scope: !1451)
!1487 = !DILocation(line: 519, column: 4, scope: !1451)
!1488 = !DILocation(line: 520, column: 15, scope: !1451)
!1489 = !DILocation(line: 520, column: 6, scope: !1451)
!1490 = !DILocation(line: 520, column: 4, scope: !1451)
!1491 = !DILocation(line: 521, column: 15, scope: !1451)
!1492 = !DILocation(line: 521, column: 6, scope: !1451)
!1493 = !DILocation(line: 521, column: 4, scope: !1451)
!1494 = !DILocation(line: 522, column: 15, scope: !1451)
!1495 = !DILocation(line: 522, column: 6, scope: !1451)
!1496 = !DILocation(line: 522, column: 4, scope: !1451)
!1497 = !DILocation(line: 523, column: 2, scope: !1451)
!1498 = !DILocalVariable(name: "rnorm", scope: !1451, file: !3, line: 530, type: !100)
!1499 = !DILocation(line: 530, column: 9, scope: !1451)
!1500 = !DILocalVariable(name: "norm_temp1", scope: !1451, file: !3, line: 531, type: !100)
!1501 = !DILocation(line: 531, column: 9, scope: !1451)
!1502 = !DILocalVariable(name: "norm_temp2", scope: !1451, file: !3, line: 531, type: !100)
!1503 = !DILocation(line: 531, column: 21, scope: !1451)
!1504 = !DILocation(line: 552, column: 11, scope: !1451)
!1505 = !DILocation(line: 553, column: 11, scope: !1451)
!1506 = !DILocation(line: 554, column: 11, scope: !1451)
!1507 = !DILocation(line: 555, column: 11, scope: !1451)
!1508 = !DILocalVariable(name: "class_npb", scope: !1451, file: !3, line: 533, type: !109)
!1509 = !DILocalVariable(name: "zeta_verify_value", scope: !1451, file: !3, line: 535, type: !100)
!1510 = !DILocation(line: 582, column: 2, scope: !1451)
!1511 = !DILocation(line: 583, column: 2, scope: !1451)
!1512 = !DILocation(line: 584, column: 2, scope: !1451)
!1513 = !DILocation(line: 586, column: 6, scope: !1451)
!1514 = !DILocation(line: 587, column: 6, scope: !1451)
!1515 = !DILocation(line: 590, column: 10, scope: !1451)
!1516 = !DILocation(line: 591, column: 10, scope: !1451)
!1517 = !DILocation(line: 592, column: 27, scope: !1451)
!1518 = !DILocation(line: 592, column: 12, scope: !1451)
!1519 = !DILocalVariable(name: "zeta", scope: !1451, file: !3, line: 529, type: !100)
!1520 = !DILocation(line: 594, column: 8, scope: !1451)
!1521 = !DILocation(line: 595, column: 4, scope: !1451)
!1522 = !DILocation(line: 596, column: 4, scope: !1451)
!1523 = !DILocation(line: 597, column: 4, scope: !1451)
!1524 = !DILocation(line: 598, column: 4, scope: !1451)
!1525 = !DILocation(line: 599, column: 4, scope: !1451)
!1526 = !DILocation(line: 600, column: 4, scope: !1451)
!1527 = !DILocation(line: 601, column: 4, scope: !1451)
!1528 = !DILocation(line: 602, column: 4, scope: !1451)
!1529 = !DILocation(line: 603, column: 4, scope: !1451)
!1530 = !DILocation(line: 604, column: 29, scope: !1451)
!1531 = !DILocation(line: 604, column: 4, scope: !1451)
!1532 = !DILocation(line: 605, column: 32, scope: !1451)
!1533 = !DILocation(line: 605, column: 4, scope: !1451)
!1534 = !DILocation(line: 606, column: 4, scope: !1451)
!1535 = !DILocation(line: 594, column: 2, scope: !1451)
!1536 = !DILocalVariable(name: "j", scope: !1451, file: !3, line: 528, type: !97)
!1537 = !DILocation(line: 618, column: 6, scope: !1538)
!1538 = distinct !DILexicalBlock(scope: !1451, file: !3, line: 618, column: 2)
!1539 = !DILocation(line: 0, scope: !1538)
!1540 = !DILocation(line: 618, column: 17, scope: !1541)
!1541 = distinct !DILexicalBlock(scope: !1538, file: !3, line: 618, column: 2)
!1542 = !DILocation(line: 618, column: 27, scope: !1541)
!1543 = !DILocation(line: 618, column: 25, scope: !1541)
!1544 = !DILocation(line: 618, column: 36, scope: !1541)
!1545 = !DILocation(line: 618, column: 15, scope: !1541)
!1546 = !DILocation(line: 618, column: 2, scope: !1538)
!1547 = !DILocation(line: 619, column: 11, scope: !1548)
!1548 = distinct !DILexicalBlock(scope: !1549, file: !3, line: 619, column: 3)
!1549 = distinct !DILexicalBlock(scope: !1541, file: !3, line: 618, column: 45)
!1550 = !DILocalVariable(name: "k", scope: !1451, file: !3, line: 528, type: !97)
!1551 = !DILocation(line: 619, column: 7, scope: !1548)
!1552 = !DILocation(line: 0, scope: !1548)
!1553 = !DILocation(line: 619, column: 26, scope: !1554)
!1554 = distinct !DILexicalBlock(scope: !1548, file: !3, line: 619, column: 3)
!1555 = !DILocation(line: 619, column: 34, scope: !1554)
!1556 = !DILocation(line: 619, column: 24, scope: !1554)
!1557 = !DILocation(line: 619, column: 3, scope: !1548)
!1558 = !DILocation(line: 620, column: 16, scope: !1559)
!1559 = distinct !DILexicalBlock(scope: !1554, file: !3, line: 619, column: 43)
!1560 = !DILocation(line: 620, column: 28, scope: !1559)
!1561 = !DILocation(line: 620, column: 26, scope: !1559)
!1562 = !DILocation(line: 620, column: 4, scope: !1559)
!1563 = !DILocation(line: 620, column: 14, scope: !1559)
!1564 = !DILocation(line: 621, column: 3, scope: !1559)
!1565 = !DILocation(line: 619, column: 40, scope: !1554)
!1566 = !DILocation(line: 619, column: 3, scope: !1554)
!1567 = distinct !{!1567, !1557, !1568}
!1568 = !DILocation(line: 621, column: 3, scope: !1548)
!1569 = !DILocation(line: 622, column: 2, scope: !1549)
!1570 = !DILocation(line: 618, column: 42, scope: !1541)
!1571 = !DILocation(line: 618, column: 2, scope: !1541)
!1572 = distinct !{!1572, !1546, !1573}
!1573 = !DILocation(line: 622, column: 2, scope: !1538)
!1574 = !DILocalVariable(name: "i", scope: !1451, file: !3, line: 528, type: !97)
!1575 = !DILocation(line: 625, column: 6, scope: !1576)
!1576 = distinct !DILexicalBlock(scope: !1451, file: !3, line: 625, column: 2)
!1577 = !DILocation(line: 0, scope: !1576)
!1578 = !DILocation(line: 625, column: 15, scope: !1579)
!1579 = distinct !DILexicalBlock(scope: !1576, file: !3, line: 625, column: 2)
!1580 = !DILocation(line: 625, column: 2, scope: !1576)
!1581 = !DILocation(line: 626, column: 3, scope: !1582)
!1582 = distinct !DILexicalBlock(scope: !1579, file: !3, line: 625, column: 27)
!1583 = !DILocation(line: 626, column: 8, scope: !1582)
!1584 = !DILocation(line: 627, column: 2, scope: !1582)
!1585 = !DILocation(line: 625, column: 24, scope: !1579)
!1586 = !DILocation(line: 625, column: 2, scope: !1579)
!1587 = distinct !{!1587, !1580, !1588}
!1588 = !DILocation(line: 627, column: 2, scope: !1576)
!1589 = !DILocation(line: 628, column: 6, scope: !1590)
!1590 = distinct !DILexicalBlock(scope: !1451, file: !3, line: 628, column: 2)
!1591 = !DILocation(line: 0, scope: !1590)
!1592 = !DILocation(line: 628, column: 15, scope: !1593)
!1593 = distinct !DILexicalBlock(scope: !1590, file: !3, line: 628, column: 2)
!1594 = !DILocation(line: 628, column: 23, scope: !1593)
!1595 = !DILocation(line: 628, column: 22, scope: !1593)
!1596 = !DILocation(line: 628, column: 31, scope: !1593)
!1597 = !DILocation(line: 628, column: 14, scope: !1593)
!1598 = !DILocation(line: 628, column: 2, scope: !1590)
!1599 = !DILocation(line: 629, column: 3, scope: !1600)
!1600 = distinct !DILexicalBlock(scope: !1593, file: !3, line: 628, column: 39)
!1601 = !DILocation(line: 629, column: 8, scope: !1600)
!1602 = !DILocation(line: 630, column: 3, scope: !1600)
!1603 = !DILocation(line: 630, column: 8, scope: !1600)
!1604 = !DILocation(line: 631, column: 3, scope: !1600)
!1605 = !DILocation(line: 631, column: 8, scope: !1600)
!1606 = !DILocation(line: 632, column: 3, scope: !1600)
!1607 = !DILocation(line: 632, column: 8, scope: !1600)
!1608 = !DILocation(line: 633, column: 2, scope: !1600)
!1609 = !DILocation(line: 628, column: 36, scope: !1593)
!1610 = !DILocation(line: 628, column: 2, scope: !1593)
!1611 = distinct !{!1611, !1598, !1612}
!1612 = !DILocation(line: 633, column: 2, scope: !1590)
!1613 = !DILocalVariable(name: "it", scope: !1451, file: !3, line: 528, type: !97)
!1614 = !DILocation(line: 642, column: 6, scope: !1615)
!1615 = distinct !DILexicalBlock(scope: !1451, file: !3, line: 642, column: 2)
!1616 = !DILocation(line: 0, scope: !1615)
!1617 = !DILocation(line: 642, column: 17, scope: !1618)
!1618 = distinct !DILexicalBlock(scope: !1615, file: !3, line: 642, column: 2)
!1619 = !DILocation(line: 642, column: 2, scope: !1615)
!1620 = !DILocation(line: 644, column: 13, scope: !1621)
!1621 = distinct !DILexicalBlock(scope: !1618, file: !3, line: 642, column: 28)
!1622 = !DILocation(line: 644, column: 21, scope: !1621)
!1623 = !DILocation(line: 644, column: 29, scope: !1621)
!1624 = !DILocation(line: 644, column: 32, scope: !1621)
!1625 = !DILocation(line: 644, column: 35, scope: !1621)
!1626 = !DILocation(line: 644, column: 38, scope: !1621)
!1627 = !DILocation(line: 644, column: 41, scope: !1621)
!1628 = !DILocation(line: 644, column: 44, scope: !1621)
!1629 = !DILocation(line: 644, column: 3, scope: !1621)
!1630 = !DILocation(line: 654, column: 14, scope: !1621)
!1631 = !DILocation(line: 655, column: 14, scope: !1621)
!1632 = !DILocation(line: 656, column: 7, scope: !1633)
!1633 = distinct !DILexicalBlock(scope: !1621, file: !3, line: 656, column: 3)
!1634 = !DILocation(line: 0, scope: !1633)
!1635 = !DILocation(line: 656, column: 18, scope: !1636)
!1636 = distinct !DILexicalBlock(scope: !1633, file: !3, line: 656, column: 3)
!1637 = !DILocation(line: 656, column: 28, scope: !1636)
!1638 = !DILocation(line: 656, column: 26, scope: !1636)
!1639 = !DILocation(line: 656, column: 37, scope: !1636)
!1640 = !DILocation(line: 656, column: 16, scope: !1636)
!1641 = !DILocation(line: 656, column: 3, scope: !1633)
!1642 = !DILocation(line: 657, column: 17, scope: !1643)
!1643 = distinct !DILexicalBlock(scope: !1636, file: !3, line: 656, column: 46)
!1644 = !DILocation(line: 657, column: 30, scope: !1643)
!1645 = !DILocation(line: 657, column: 37, scope: !1643)
!1646 = !DILocation(line: 657, column: 35, scope: !1643)
!1647 = !DILocation(line: 657, column: 28, scope: !1643)
!1648 = !DILocation(line: 657, column: 15, scope: !1643)
!1649 = !DILocation(line: 658, column: 17, scope: !1643)
!1650 = !DILocation(line: 658, column: 30, scope: !1643)
!1651 = !DILocation(line: 658, column: 37, scope: !1643)
!1652 = !DILocation(line: 658, column: 35, scope: !1643)
!1653 = !DILocation(line: 658, column: 28, scope: !1643)
!1654 = !DILocation(line: 658, column: 15, scope: !1643)
!1655 = !DILocation(line: 659, column: 3, scope: !1643)
!1656 = !DILocation(line: 656, column: 43, scope: !1636)
!1657 = !DILocation(line: 656, column: 3, scope: !1636)
!1658 = distinct !{!1658, !1641, !1659}
!1659 = !DILocation(line: 659, column: 3, scope: !1633)
!1660 = !DILocation(line: 660, column: 27, scope: !1621)
!1661 = !DILocation(line: 660, column: 22, scope: !1621)
!1662 = !DILocation(line: 660, column: 20, scope: !1621)
!1663 = !DILocation(line: 660, column: 14, scope: !1621)
!1664 = !DILocation(line: 663, column: 7, scope: !1665)
!1665 = distinct !DILexicalBlock(scope: !1621, file: !3, line: 663, column: 3)
!1666 = !DILocation(line: 0, scope: !1665)
!1667 = !DILocation(line: 663, column: 18, scope: !1668)
!1668 = distinct !DILexicalBlock(scope: !1665, file: !3, line: 663, column: 3)
!1669 = !DILocation(line: 663, column: 28, scope: !1668)
!1670 = !DILocation(line: 663, column: 26, scope: !1668)
!1671 = !DILocation(line: 663, column: 37, scope: !1668)
!1672 = !DILocation(line: 663, column: 16, scope: !1668)
!1673 = !DILocation(line: 663, column: 3, scope: !1665)
!1674 = !DILocation(line: 664, column: 11, scope: !1675)
!1675 = distinct !DILexicalBlock(scope: !1668, file: !3, line: 663, column: 46)
!1676 = !DILocation(line: 664, column: 24, scope: !1675)
!1677 = !DILocation(line: 664, column: 22, scope: !1675)
!1678 = !DILocation(line: 664, column: 4, scope: !1675)
!1679 = !DILocation(line: 664, column: 9, scope: !1675)
!1680 = !DILocation(line: 665, column: 3, scope: !1675)
!1681 = !DILocation(line: 663, column: 43, scope: !1668)
!1682 = !DILocation(line: 663, column: 3, scope: !1668)
!1683 = distinct !{!1683, !1673, !1684}
!1684 = !DILocation(line: 665, column: 3, scope: !1665)
!1685 = !DILocation(line: 666, column: 2, scope: !1621)
!1686 = !DILocation(line: 642, column: 25, scope: !1618)
!1687 = !DILocation(line: 642, column: 2, scope: !1618)
!1688 = distinct !{!1688, !1619, !1689}
!1689 = !DILocation(line: 666, column: 2, scope: !1615)
!1690 = !DILocation(line: 669, column: 6, scope: !1691)
!1691 = distinct !DILexicalBlock(scope: !1451, file: !3, line: 669, column: 2)
!1692 = !DILocation(line: 0, scope: !1691)
!1693 = !DILocation(line: 669, column: 15, scope: !1694)
!1694 = distinct !DILexicalBlock(scope: !1691, file: !3, line: 669, column: 2)
!1695 = !DILocation(line: 669, column: 2, scope: !1691)
!1696 = !DILocation(line: 670, column: 3, scope: !1697)
!1697 = distinct !DILexicalBlock(scope: !1694, file: !3, line: 669, column: 27)
!1698 = !DILocation(line: 670, column: 8, scope: !1697)
!1699 = !DILocation(line: 671, column: 2, scope: !1697)
!1700 = !DILocation(line: 669, column: 24, scope: !1694)
!1701 = !DILocation(line: 669, column: 2, scope: !1694)
!1702 = distinct !{!1702, !1695, !1703}
!1703 = !DILocation(line: 671, column: 2, scope: !1691)
!1704 = !DILocation(line: 674, column: 2, scope: !1451)
!1705 = !DILocation(line: 684, column: 6, scope: !1706)
!1706 = distinct !DILexicalBlock(scope: !1451, file: !3, line: 684, column: 2)
!1707 = !DILocation(line: 0, scope: !1706)
!1708 = !DILocation(line: 684, column: 17, scope: !1709)
!1709 = distinct !DILexicalBlock(scope: !1706, file: !3, line: 684, column: 2)
!1710 = !DILocation(line: 684, column: 2, scope: !1706)
!1711 = !DILocation(line: 686, column: 3, scope: !1712)
!1712 = distinct !DILexicalBlock(scope: !1709, file: !3, line: 684, column: 32)
!1713 = !DILocation(line: 696, column: 3, scope: !1712)
!1714 = !DILocation(line: 697, column: 27, scope: !1712)
!1715 = !DILocation(line: 697, column: 22, scope: !1712)
!1716 = !DILocation(line: 697, column: 20, scope: !1712)
!1717 = !DILocation(line: 697, column: 14, scope: !1712)
!1718 = !DILocation(line: 698, column: 24, scope: !1712)
!1719 = !DILocation(line: 698, column: 22, scope: !1712)
!1720 = !DILocation(line: 698, column: 16, scope: !1712)
!1721 = !DILocation(line: 699, column: 8, scope: !1722)
!1722 = distinct !DILexicalBlock(scope: !1712, file: !3, line: 699, column: 6)
!1723 = !DILocation(line: 699, column: 6, scope: !1712)
!1724 = !DILocation(line: 699, column: 13, scope: !1725)
!1725 = distinct !DILexicalBlock(scope: !1722, file: !3, line: 699, column: 12)
!1726 = !DILocation(line: 699, column: 77, scope: !1725)
!1727 = !DILocation(line: 700, column: 48, scope: !1712)
!1728 = !DILocation(line: 700, column: 3, scope: !1712)
!1729 = !DILocation(line: 703, column: 26, scope: !1712)
!1730 = !DILocation(line: 703, column: 3, scope: !1712)
!1731 = !DILocation(line: 704, column: 2, scope: !1712)
!1732 = !DILocation(line: 684, column: 29, scope: !1709)
!1733 = !DILocation(line: 684, column: 2, scope: !1709)
!1734 = distinct !{!1734, !1710, !1735}
!1735 = !DILocation(line: 704, column: 2, scope: !1706)
!1736 = !DILocalVariable(name: "t", scope: !1451, file: !3, line: 532, type: !100)
!1737 = !DILocation(line: 718, column: 2, scope: !1451)
!1738 = !DILocalVariable(name: "epsilon", scope: !1451, file: !3, line: 535, type: !100)
!1739 = !DILocation(line: 721, column: 5, scope: !1740)
!1740 = distinct !DILexicalBlock(scope: !1451, file: !3, line: 721, column: 5)
!1741 = !DILocation(line: 721, column: 15, scope: !1740)
!1742 = !DILocation(line: 721, column: 5, scope: !1451)
!1743 = !DILocation(line: 722, column: 19, scope: !1744)
!1744 = distinct !DILexicalBlock(scope: !1740, file: !3, line: 721, column: 22)
!1745 = !DILocation(line: 722, column: 9, scope: !1744)
!1746 = !DILocation(line: 722, column: 40, scope: !1744)
!1747 = !DILocalVariable(name: "err", scope: !1451, file: !3, line: 535, type: !100)
!1748 = !DILocation(line: 723, column: 10, scope: !1749)
!1749 = distinct !DILexicalBlock(scope: !1744, file: !3, line: 723, column: 6)
!1750 = !DILocation(line: 723, column: 6, scope: !1744)
!1751 = !DILocalVariable(name: "verified", scope: !1451, file: !3, line: 534, type: !1752)
!1752 = !DIDerivedType(tag: DW_TAG_typedef, name: "boolean", file: !1753, line: 80, baseType: !97)
!1753 = !DIFile(filename: "./../common/npb-CPP.hpp", directory: "/scratch/ah7226/NPB-GPU/CUDA/CG")
!1754 = !DILocation(line: 725, column: 4, scope: !1755)
!1755 = distinct !DILexicalBlock(scope: !1749, file: !3, line: 723, column: 21)
!1756 = !DILocation(line: 726, column: 4, scope: !1755)
!1757 = !DILocation(line: 727, column: 4, scope: !1755)
!1758 = !DILocation(line: 728, column: 3, scope: !1755)
!1759 = !DILocation(line: 730, column: 4, scope: !1760)
!1760 = distinct !DILexicalBlock(scope: !1749, file: !3, line: 728, column: 8)
!1761 = !DILocation(line: 731, column: 4, scope: !1760)
!1762 = !DILocation(line: 732, column: 4, scope: !1760)
!1763 = !DILocation(line: 0, scope: !1749)
!1764 = !DILocation(line: 734, column: 2, scope: !1744)
!1765 = !DILocation(line: 736, column: 3, scope: !1766)
!1766 = distinct !DILexicalBlock(scope: !1740, file: !3, line: 734, column: 7)
!1767 = !DILocation(line: 737, column: 3, scope: !1766)
!1768 = !DILocation(line: 0, scope: !1740)
!1769 = !DILocation(line: 739, column: 7, scope: !1770)
!1770 = distinct !DILexicalBlock(scope: !1451, file: !3, line: 739, column: 5)
!1771 = !DILocation(line: 739, column: 5, scope: !1451)
!1772 = !DILocation(line: 744, column: 4, scope: !1773)
!1773 = distinct !DILexicalBlock(scope: !1770, file: !3, line: 739, column: 14)
!1774 = !DILocation(line: 744, column: 8, scope: !1773)
!1775 = !DILocalVariable(name: "mflops", scope: !1451, file: !3, line: 532, type: !100)
!1776 = !DILocation(line: 745, column: 2, scope: !1773)
!1777 = !DILocation(line: 0, scope: !1770)
!1778 = !DILocalVariable(name: "gpu_config", scope: !1451, file: !3, line: 749, type: !294)
!1779 = !DILocation(line: 749, column: 7, scope: !1451)
!1780 = !DILocalVariable(name: "gpu_config_string", scope: !1451, file: !3, line: 750, type: !1781)
!1781 = !DICompositeType(tag: DW_TAG_array_type, baseType: !109, size: 16384, elements: !1782)
!1782 = !{!1783}
!1783 = !DISubrange(count: 2048)
!1784 = !DILocation(line: 750, column: 7, scope: !1451)
!1785 = !DILocation(line: 777, column: 10, scope: !1451)
!1786 = !DILocation(line: 777, column: 2, scope: !1451)
!1787 = !DILocation(line: 778, column: 9, scope: !1451)
!1788 = !DILocation(line: 778, column: 28, scope: !1451)
!1789 = !DILocation(line: 778, column: 2, scope: !1451)
!1790 = !DILocation(line: 779, column: 10, scope: !1451)
!1791 = !DILocation(line: 779, column: 46, scope: !1451)
!1792 = !DILocation(line: 779, column: 2, scope: !1451)
!1793 = !DILocation(line: 780, column: 9, scope: !1451)
!1794 = !DILocation(line: 780, column: 28, scope: !1451)
!1795 = !DILocation(line: 780, column: 2, scope: !1451)
!1796 = !DILocation(line: 781, column: 10, scope: !1451)
!1797 = !DILocation(line: 781, column: 46, scope: !1451)
!1798 = !DILocation(line: 781, column: 2, scope: !1451)
!1799 = !DILocation(line: 782, column: 9, scope: !1451)
!1800 = !DILocation(line: 782, column: 28, scope: !1451)
!1801 = !DILocation(line: 782, column: 2, scope: !1451)
!1802 = !DILocation(line: 783, column: 10, scope: !1451)
!1803 = !DILocation(line: 783, column: 48, scope: !1451)
!1804 = !DILocation(line: 783, column: 2, scope: !1451)
!1805 = !DILocation(line: 784, column: 9, scope: !1451)
!1806 = !DILocation(line: 784, column: 28, scope: !1451)
!1807 = !DILocation(line: 784, column: 2, scope: !1451)
!1808 = !DILocation(line: 785, column: 10, scope: !1451)
!1809 = !DILocation(line: 785, column: 47, scope: !1451)
!1810 = !DILocation(line: 785, column: 2, scope: !1451)
!1811 = !DILocation(line: 786, column: 9, scope: !1451)
!1812 = !DILocation(line: 786, column: 28, scope: !1451)
!1813 = !DILocation(line: 786, column: 2, scope: !1451)
!1814 = !DILocation(line: 787, column: 10, scope: !1451)
!1815 = !DILocation(line: 787, column: 47, scope: !1451)
!1816 = !DILocation(line: 787, column: 2, scope: !1451)
!1817 = !DILocation(line: 788, column: 9, scope: !1451)
!1818 = !DILocation(line: 788, column: 28, scope: !1451)
!1819 = !DILocation(line: 788, column: 2, scope: !1451)
!1820 = !DILocation(line: 789, column: 10, scope: !1451)
!1821 = !DILocation(line: 789, column: 46, scope: !1451)
!1822 = !DILocation(line: 789, column: 2, scope: !1451)
!1823 = !DILocation(line: 790, column: 9, scope: !1451)
!1824 = !DILocation(line: 790, column: 28, scope: !1451)
!1825 = !DILocation(line: 790, column: 2, scope: !1451)
!1826 = !DILocation(line: 791, column: 10, scope: !1451)
!1827 = !DILocation(line: 791, column: 48, scope: !1451)
!1828 = !DILocation(line: 791, column: 2, scope: !1451)
!1829 = !DILocation(line: 792, column: 9, scope: !1451)
!1830 = !DILocation(line: 792, column: 28, scope: !1451)
!1831 = !DILocation(line: 792, column: 2, scope: !1451)
!1832 = !DILocation(line: 793, column: 10, scope: !1451)
!1833 = !DILocation(line: 793, column: 48, scope: !1451)
!1834 = !DILocation(line: 793, column: 2, scope: !1451)
!1835 = !DILocation(line: 794, column: 9, scope: !1451)
!1836 = !DILocation(line: 794, column: 28, scope: !1451)
!1837 = !DILocation(line: 794, column: 2, scope: !1451)
!1838 = !DILocation(line: 795, column: 10, scope: !1451)
!1839 = !DILocation(line: 795, column: 47, scope: !1451)
!1840 = !DILocation(line: 795, column: 2, scope: !1451)
!1841 = !DILocation(line: 796, column: 9, scope: !1451)
!1842 = !DILocation(line: 796, column: 28, scope: !1451)
!1843 = !DILocation(line: 796, column: 2, scope: !1451)
!1844 = !DILocation(line: 797, column: 10, scope: !1451)
!1845 = !DILocation(line: 797, column: 46, scope: !1451)
!1846 = !DILocation(line: 797, column: 2, scope: !1451)
!1847 = !DILocation(line: 798, column: 9, scope: !1451)
!1848 = !DILocation(line: 798, column: 28, scope: !1451)
!1849 = !DILocation(line: 798, column: 2, scope: !1451)
!1850 = !DILocation(line: 799, column: 10, scope: !1451)
!1851 = !DILocation(line: 799, column: 49, scope: !1451)
!1852 = !DILocation(line: 799, column: 2, scope: !1451)
!1853 = !DILocation(line: 800, column: 9, scope: !1451)
!1854 = !DILocation(line: 800, column: 28, scope: !1451)
!1855 = !DILocation(line: 800, column: 2, scope: !1451)
!1856 = !DILocation(line: 819, column: 11, scope: !1451)
!1857 = !DILocation(line: 803, column: 2, scope: !1451)
!1858 = !DILocation(line: 828, column: 2, scope: !1451)
!1859 = !DILocation(line: 830, column: 2, scope: !1451)
!1860 = distinct !DISubprogram(name: "makea", linkageName: "_ZL5makeaiiPdPiS0_iiiiS0_PA12_iPA12_dS0_", scope: !3, file: !3, line: 1575, type: !1861, scopeLine: 1587, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1156)
!1861 = !DISubroutineType(types: !1862)
!1862 = !{null, !97, !97, !99, !98, !98, !97, !97, !97, !97, !98, !101, !106, !98}
!1863 = !DILocalVariable(name: "n", arg: 1, scope: !1860, file: !3, line: 1575, type: !97)
!1864 = !DILocation(line: 0, scope: !1860)
!1865 = !DILocalVariable(name: "nz", arg: 2, scope: !1860, file: !3, line: 1576, type: !97)
!1866 = !DILocalVariable(name: "a", arg: 3, scope: !1860, file: !3, line: 1577, type: !99)
!1867 = !DILocalVariable(name: "colidx", arg: 4, scope: !1860, file: !3, line: 1578, type: !98)
!1868 = !DILocalVariable(name: "rowstr", arg: 5, scope: !1860, file: !3, line: 1579, type: !98)
!1869 = !DILocalVariable(name: "firstrow", arg: 6, scope: !1860, file: !3, line: 1580, type: !97)
!1870 = !DILocalVariable(name: "lastrow", arg: 7, scope: !1860, file: !3, line: 1581, type: !97)
!1871 = !DILocalVariable(name: "firstcol", arg: 8, scope: !1860, file: !3, line: 1582, type: !97)
!1872 = !DILocalVariable(name: "lastcol", arg: 9, scope: !1860, file: !3, line: 1583, type: !97)
!1873 = !DILocalVariable(name: "arow", arg: 10, scope: !1860, file: !3, line: 1584, type: !98)
!1874 = !DILocalVariable(name: "acol", arg: 11, scope: !1860, file: !3, line: 1585, type: !101)
!1875 = !DILocalVariable(name: "aelt", arg: 12, scope: !1860, file: !3, line: 1586, type: !106)
!1876 = !DILocalVariable(name: "iv", arg: 13, scope: !1860, file: !3, line: 1587, type: !98)
!1877 = !DILocalVariable(name: "nzv", scope: !1860, file: !3, line: 1588, type: !97)
!1878 = !DILocation(line: 1588, column: 21, scope: !1860)
!1879 = !DILocalVariable(name: "ivc", scope: !1860, file: !3, line: 1589, type: !102)
!1880 = !DILocation(line: 1589, column: 6, scope: !1860)
!1881 = !DILocalVariable(name: "vc", scope: !1860, file: !3, line: 1590, type: !107)
!1882 = !DILocation(line: 1590, column: 9, scope: !1860)
!1883 = !DILocalVariable(name: "nn1", scope: !1860, file: !3, line: 1588, type: !97)
!1884 = !DILocation(line: 1600, column: 2, scope: !1860)
!1885 = !DILocation(line: 1601, column: 11, scope: !1886)
!1886 = distinct !DILexicalBlock(scope: !1860, file: !3, line: 1600, column: 4)
!1887 = !DILocation(line: 1602, column: 2, scope: !1886)
!1888 = !DILocation(line: 1602, column: 13, scope: !1860)
!1889 = distinct !{!1889, !1884, !1890}
!1890 = !DILocation(line: 1602, column: 16, scope: !1860)
!1891 = !DILocalVariable(name: "iouter", scope: !1860, file: !3, line: 1588, type: !97)
!1892 = !DILocation(line: 1609, column: 6, scope: !1893)
!1893 = distinct !DILexicalBlock(scope: !1860, file: !3, line: 1609, column: 2)
!1894 = !DILocation(line: 0, scope: !1893)
!1895 = !DILocation(line: 1609, column: 25, scope: !1896)
!1896 = distinct !DILexicalBlock(scope: !1893, file: !3, line: 1609, column: 2)
!1897 = !DILocation(line: 1609, column: 2, scope: !1893)
!1898 = !DILocation(line: 1610, column: 7, scope: !1899)
!1899 = distinct !DILexicalBlock(scope: !1896, file: !3, line: 1609, column: 39)
!1900 = !DILocation(line: 1611, column: 13, scope: !1899)
!1901 = !DILocation(line: 1611, column: 23, scope: !1899)
!1902 = !DILocation(line: 1611, column: 27, scope: !1899)
!1903 = !DILocation(line: 1611, column: 3, scope: !1899)
!1904 = !DILocation(line: 1612, column: 13, scope: !1899)
!1905 = !DILocation(line: 1612, column: 17, scope: !1899)
!1906 = !DILocation(line: 1609, column: 36, scope: !1896)
!1907 = !DILocation(line: 1612, column: 3, scope: !1899)
!1908 = !DILocation(line: 1613, column: 18, scope: !1899)
!1909 = !DILocation(line: 1613, column: 3, scope: !1899)
!1910 = !DILocation(line: 1613, column: 16, scope: !1899)
!1911 = !DILocalVariable(name: "ivelt", scope: !1860, file: !3, line: 1588, type: !97)
!1912 = !DILocation(line: 1614, column: 7, scope: !1913)
!1913 = distinct !DILexicalBlock(scope: !1899, file: !3, line: 1614, column: 3)
!1914 = !DILocation(line: 0, scope: !1913)
!1915 = !DILocation(line: 1614, column: 26, scope: !1916)
!1916 = distinct !DILexicalBlock(scope: !1913, file: !3, line: 1614, column: 3)
!1917 = !DILocation(line: 1614, column: 24, scope: !1916)
!1918 = !DILocation(line: 1614, column: 3, scope: !1913)
!1919 = !DILocation(line: 1615, column: 26, scope: !1920)
!1920 = distinct !DILexicalBlock(scope: !1916, file: !3, line: 1614, column: 39)
!1921 = !DILocation(line: 1615, column: 37, scope: !1920)
!1922 = !DILocation(line: 1615, column: 4, scope: !1920)
!1923 = !DILocation(line: 1615, column: 24, scope: !1920)
!1924 = !DILocation(line: 1616, column: 26, scope: !1920)
!1925 = !DILocation(line: 1616, column: 4, scope: !1920)
!1926 = !DILocation(line: 1616, column: 24, scope: !1920)
!1927 = !DILocation(line: 1617, column: 3, scope: !1920)
!1928 = !DILocation(line: 1614, column: 36, scope: !1916)
!1929 = !DILocation(line: 1614, column: 3, scope: !1916)
!1930 = distinct !{!1930, !1918, !1931}
!1931 = !DILocation(line: 1617, column: 3, scope: !1913)
!1932 = !DILocation(line: 1618, column: 2, scope: !1899)
!1933 = !DILocation(line: 1609, column: 2, scope: !1896)
!1934 = distinct !{!1934, !1897, !1935}
!1935 = !DILocation(line: 1618, column: 2, scope: !1893)
!1936 = !DILocation(line: 1626, column: 2, scope: !1860)
!1937 = !DILocation(line: 1640, column: 1, scope: !1860)
!1938 = distinct !DISubprogram(name: "conj_grad", linkageName: "_ZL9conj_gradPiS_PdS0_S0_S0_S0_S0_S0_", scope: !3, file: !3, line: 839, type: !1939, scopeLine: 847, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1156)
!1939 = !DISubroutineType(types: !1940)
!1940 = !{null, !98, !98, !99, !99, !99, !99, !99, !99, !99}
!1941 = !DILocalVariable(name: "colidx", arg: 1, scope: !1938, file: !3, line: 839, type: !98)
!1942 = !DILocation(line: 0, scope: !1938)
!1943 = !DILocalVariable(name: "rowstr", arg: 2, scope: !1938, file: !3, line: 840, type: !98)
!1944 = !DILocalVariable(name: "x", arg: 3, scope: !1938, file: !3, line: 841, type: !99)
!1945 = !DILocalVariable(name: "z", arg: 4, scope: !1938, file: !3, line: 842, type: !99)
!1946 = !DILocalVariable(name: "a", arg: 5, scope: !1938, file: !3, line: 843, type: !99)
!1947 = !DILocalVariable(name: "p", arg: 6, scope: !1938, file: !3, line: 844, type: !99)
!1948 = !DILocalVariable(name: "q", arg: 7, scope: !1938, file: !3, line: 845, type: !99)
!1949 = !DILocalVariable(name: "r", arg: 8, scope: !1938, file: !3, line: 846, type: !99)
!1950 = !DILocalVariable(name: "rnorm", arg: 9, scope: !1938, file: !3, line: 847, type: !99)
!1951 = !DILocalVariable(name: "cgitmax", scope: !1938, file: !3, line: 849, type: !97)
!1952 = !DILocalVariable(name: "rho", scope: !1938, file: !3, line: 850, type: !100)
!1953 = !DILocalVariable(name: "j", scope: !1938, file: !3, line: 848, type: !97)
!1954 = !DILocation(line: 857, column: 6, scope: !1955)
!1955 = distinct !DILexicalBlock(scope: !1938, file: !3, line: 857, column: 2)
!1956 = !DILocation(line: 0, scope: !1955)
!1957 = !DILocation(line: 857, column: 17, scope: !1958)
!1958 = distinct !DILexicalBlock(scope: !1955, file: !3, line: 857, column: 2)
!1959 = !DILocation(line: 857, column: 20, scope: !1958)
!1960 = !DILocation(line: 857, column: 15, scope: !1958)
!1961 = !DILocation(line: 857, column: 2, scope: !1955)
!1962 = !DILocation(line: 858, column: 3, scope: !1963)
!1963 = distinct !DILexicalBlock(scope: !1958, file: !3, line: 857, column: 28)
!1964 = !DILocation(line: 858, column: 8, scope: !1963)
!1965 = !DILocation(line: 859, column: 3, scope: !1963)
!1966 = !DILocation(line: 859, column: 8, scope: !1963)
!1967 = !DILocation(line: 860, column: 10, scope: !1963)
!1968 = !DILocation(line: 860, column: 3, scope: !1963)
!1969 = !DILocation(line: 860, column: 8, scope: !1963)
!1970 = !DILocation(line: 861, column: 10, scope: !1963)
!1971 = !DILocation(line: 861, column: 3, scope: !1963)
!1972 = !DILocation(line: 861, column: 8, scope: !1963)
!1973 = !DILocation(line: 862, column: 2, scope: !1963)
!1974 = !DILocation(line: 857, column: 25, scope: !1958)
!1975 = !DILocation(line: 857, column: 2, scope: !1958)
!1976 = distinct !{!1976, !1961, !1977}
!1977 = !DILocation(line: 862, column: 2, scope: !1955)
!1978 = !DILocation(line: 870, column: 6, scope: !1979)
!1979 = distinct !DILexicalBlock(scope: !1938, file: !3, line: 870, column: 2)
!1980 = !DILocation(line: 0, scope: !1979)
!1981 = !DILocation(line: 870, column: 17, scope: !1982)
!1982 = distinct !DILexicalBlock(scope: !1979, file: !3, line: 870, column: 2)
!1983 = !DILocation(line: 870, column: 27, scope: !1982)
!1984 = !DILocation(line: 870, column: 25, scope: !1982)
!1985 = !DILocation(line: 870, column: 36, scope: !1982)
!1986 = !DILocation(line: 870, column: 15, scope: !1982)
!1987 = !DILocation(line: 870, column: 2, scope: !1979)
!1988 = !DILocation(line: 871, column: 15, scope: !1989)
!1989 = distinct !DILexicalBlock(scope: !1982, file: !3, line: 870, column: 45)
!1990 = !DILocation(line: 871, column: 20, scope: !1989)
!1991 = !DILocation(line: 871, column: 19, scope: !1989)
!1992 = !DILocation(line: 871, column: 13, scope: !1989)
!1993 = !DILocation(line: 872, column: 2, scope: !1989)
!1994 = !DILocation(line: 870, column: 42, scope: !1982)
!1995 = !DILocation(line: 870, column: 2, scope: !1982)
!1996 = distinct !{!1996, !1987, !1997}
!1997 = !DILocation(line: 872, column: 2, scope: !1979)
!1998 = !DILocalVariable(name: "cgit", scope: !1938, file: !3, line: 849, type: !97)
!1999 = !DILocation(line: 875, column: 6, scope: !2000)
!2000 = distinct !DILexicalBlock(scope: !1938, file: !3, line: 875, column: 2)
!2001 = !DILocation(line: 0, scope: !2000)
!2002 = !DILocation(line: 875, column: 21, scope: !2003)
!2003 = distinct !DILexicalBlock(scope: !2000, file: !3, line: 875, column: 2)
!2004 = !DILocation(line: 875, column: 2, scope: !2000)
!2005 = !DILocation(line: 889, column: 7, scope: !2006)
!2006 = distinct !DILexicalBlock(scope: !2007, file: !3, line: 889, column: 3)
!2007 = distinct !DILexicalBlock(scope: !2003, file: !3, line: 875, column: 40)
!2008 = !DILocation(line: 0, scope: !2006)
!2009 = !DILocation(line: 889, column: 18, scope: !2010)
!2010 = distinct !DILexicalBlock(scope: !2006, file: !3, line: 889, column: 3)
!2011 = !DILocation(line: 889, column: 28, scope: !2010)
!2012 = !DILocation(line: 889, column: 26, scope: !2010)
!2013 = !DILocation(line: 889, column: 37, scope: !2010)
!2014 = !DILocation(line: 889, column: 16, scope: !2010)
!2015 = !DILocation(line: 889, column: 3, scope: !2006)
!2016 = !DILocalVariable(name: "sum", scope: !1938, file: !3, line: 850, type: !100)
!2017 = !DILocation(line: 891, column: 12, scope: !2018)
!2018 = distinct !DILexicalBlock(scope: !2019, file: !3, line: 891, column: 4)
!2019 = distinct !DILexicalBlock(scope: !2010, file: !3, line: 889, column: 46)
!2020 = !DILocalVariable(name: "k", scope: !1938, file: !3, line: 848, type: !97)
!2021 = !DILocation(line: 891, column: 8, scope: !2018)
!2022 = !DILocation(line: 0, scope: !2018)
!2023 = !DILocation(line: 0, scope: !2019)
!2024 = !DILocation(line: 891, column: 35, scope: !2025)
!2025 = distinct !DILexicalBlock(scope: !2018, file: !3, line: 891, column: 4)
!2026 = !DILocation(line: 891, column: 27, scope: !2025)
!2027 = !DILocation(line: 891, column: 25, scope: !2025)
!2028 = !DILocation(line: 891, column: 4, scope: !2018)
!2029 = !DILocation(line: 892, column: 17, scope: !2030)
!2030 = distinct !DILexicalBlock(scope: !2025, file: !3, line: 891, column: 44)
!2031 = !DILocation(line: 892, column: 24, scope: !2030)
!2032 = !DILocation(line: 892, column: 22, scope: !2030)
!2033 = !DILocation(line: 892, column: 21, scope: !2030)
!2034 = !DILocation(line: 892, column: 15, scope: !2030)
!2035 = !DILocation(line: 893, column: 4, scope: !2030)
!2036 = !DILocation(line: 891, column: 41, scope: !2025)
!2037 = !DILocation(line: 891, column: 4, scope: !2025)
!2038 = distinct !{!2038, !2028, !2039}
!2039 = !DILocation(line: 893, column: 4, scope: !2018)
!2040 = !DILocation(line: 894, column: 4, scope: !2019)
!2041 = !DILocation(line: 894, column: 9, scope: !2019)
!2042 = !DILocation(line: 895, column: 3, scope: !2019)
!2043 = !DILocation(line: 889, column: 43, scope: !2010)
!2044 = !DILocation(line: 889, column: 3, scope: !2010)
!2045 = distinct !{!2045, !2015, !2046}
!2046 = !DILocation(line: 895, column: 3, scope: !2006)
!2047 = !DILocalVariable(name: "d", scope: !1938, file: !3, line: 850, type: !100)
!2048 = !DILocation(line: 903, column: 8, scope: !2049)
!2049 = distinct !DILexicalBlock(scope: !2007, file: !3, line: 903, column: 3)
!2050 = !DILocation(line: 0, scope: !2049)
!2051 = !DILocation(line: 0, scope: !2007)
!2052 = !DILocation(line: 903, column: 19, scope: !2053)
!2053 = distinct !DILexicalBlock(scope: !2049, file: !3, line: 903, column: 3)
!2054 = !DILocation(line: 903, column: 29, scope: !2053)
!2055 = !DILocation(line: 903, column: 27, scope: !2053)
!2056 = !DILocation(line: 903, column: 38, scope: !2053)
!2057 = !DILocation(line: 903, column: 17, scope: !2053)
!2058 = !DILocation(line: 903, column: 3, scope: !2049)
!2059 = !DILocation(line: 904, column: 12, scope: !2060)
!2060 = distinct !DILexicalBlock(scope: !2053, file: !3, line: 903, column: 48)
!2061 = !DILocation(line: 904, column: 17, scope: !2060)
!2062 = !DILocation(line: 904, column: 16, scope: !2060)
!2063 = !DILocation(line: 904, column: 10, scope: !2060)
!2064 = !DILocation(line: 905, column: 3, scope: !2060)
!2065 = !DILocation(line: 903, column: 44, scope: !2053)
!2066 = !DILocation(line: 903, column: 3, scope: !2053)
!2067 = distinct !{!2067, !2058, !2068}
!2068 = !DILocation(line: 905, column: 3, scope: !2049)
!2069 = !DILocation(line: 912, column: 15, scope: !2007)
!2070 = !DILocalVariable(name: "alpha", scope: !1938, file: !3, line: 850, type: !100)
!2071 = !DILocalVariable(name: "rho0", scope: !1938, file: !3, line: 850, type: !100)
!2072 = !DILocation(line: 928, column: 7, scope: !2073)
!2073 = distinct !DILexicalBlock(scope: !2007, file: !3, line: 928, column: 3)
!2074 = !DILocation(line: 0, scope: !2073)
!2075 = !DILocation(line: 928, column: 18, scope: !2076)
!2076 = distinct !DILexicalBlock(scope: !2073, file: !3, line: 928, column: 3)
!2077 = !DILocation(line: 928, column: 28, scope: !2076)
!2078 = !DILocation(line: 928, column: 26, scope: !2076)
!2079 = !DILocation(line: 928, column: 37, scope: !2076)
!2080 = !DILocation(line: 928, column: 16, scope: !2076)
!2081 = !DILocation(line: 928, column: 3, scope: !2073)
!2082 = !DILocation(line: 929, column: 11, scope: !2083)
!2083 = distinct !DILexicalBlock(scope: !2076, file: !3, line: 928, column: 46)
!2084 = !DILocation(line: 929, column: 24, scope: !2083)
!2085 = !DILocation(line: 929, column: 23, scope: !2083)
!2086 = !DILocation(line: 929, column: 16, scope: !2083)
!2087 = !DILocation(line: 929, column: 4, scope: !2083)
!2088 = !DILocation(line: 929, column: 9, scope: !2083)
!2089 = !DILocation(line: 930, column: 11, scope: !2083)
!2090 = !DILocation(line: 930, column: 24, scope: !2083)
!2091 = !DILocation(line: 930, column: 23, scope: !2083)
!2092 = !DILocation(line: 930, column: 16, scope: !2083)
!2093 = !DILocation(line: 930, column: 4, scope: !2083)
!2094 = !DILocation(line: 930, column: 9, scope: !2083)
!2095 = !DILocation(line: 931, column: 3, scope: !2083)
!2096 = !DILocation(line: 928, column: 43, scope: !2076)
!2097 = !DILocation(line: 928, column: 3, scope: !2076)
!2098 = distinct !{!2098, !2081, !2099}
!2099 = !DILocation(line: 931, column: 3, scope: !2073)
!2100 = !DILocation(line: 939, column: 7, scope: !2101)
!2101 = distinct !DILexicalBlock(scope: !2007, file: !3, line: 939, column: 3)
!2102 = !DILocation(line: 0, scope: !2101)
!2103 = !DILocation(line: 939, column: 18, scope: !2104)
!2104 = distinct !DILexicalBlock(scope: !2101, file: !3, line: 939, column: 3)
!2105 = !DILocation(line: 939, column: 28, scope: !2104)
!2106 = !DILocation(line: 939, column: 26, scope: !2104)
!2107 = !DILocation(line: 939, column: 37, scope: !2104)
!2108 = !DILocation(line: 939, column: 16, scope: !2104)
!2109 = !DILocation(line: 939, column: 3, scope: !2101)
!2110 = !DILocation(line: 940, column: 16, scope: !2111)
!2111 = distinct !DILexicalBlock(scope: !2104, file: !3, line: 939, column: 46)
!2112 = !DILocation(line: 940, column: 21, scope: !2111)
!2113 = !DILocation(line: 940, column: 20, scope: !2111)
!2114 = !DILocation(line: 940, column: 14, scope: !2111)
!2115 = !DILocation(line: 941, column: 3, scope: !2111)
!2116 = !DILocation(line: 939, column: 43, scope: !2104)
!2117 = !DILocation(line: 939, column: 3, scope: !2104)
!2118 = distinct !{!2118, !2109, !2119}
!2119 = !DILocation(line: 941, column: 3, scope: !2101)
!2120 = !DILocation(line: 948, column: 14, scope: !2007)
!2121 = !DILocalVariable(name: "beta", scope: !1938, file: !3, line: 850, type: !100)
!2122 = !DILocation(line: 955, column: 7, scope: !2123)
!2123 = distinct !DILexicalBlock(scope: !2007, file: !3, line: 955, column: 3)
!2124 = !DILocation(line: 0, scope: !2123)
!2125 = !DILocation(line: 955, column: 18, scope: !2126)
!2126 = distinct !DILexicalBlock(scope: !2123, file: !3, line: 955, column: 3)
!2127 = !DILocation(line: 955, column: 28, scope: !2126)
!2128 = !DILocation(line: 955, column: 26, scope: !2126)
!2129 = !DILocation(line: 955, column: 37, scope: !2126)
!2130 = !DILocation(line: 955, column: 16, scope: !2126)
!2131 = !DILocation(line: 955, column: 3, scope: !2123)
!2132 = !DILocation(line: 956, column: 11, scope: !2133)
!2133 = distinct !DILexicalBlock(scope: !2126, file: !3, line: 955, column: 46)
!2134 = !DILocation(line: 956, column: 23, scope: !2133)
!2135 = !DILocation(line: 956, column: 22, scope: !2133)
!2136 = !DILocation(line: 956, column: 16, scope: !2133)
!2137 = !DILocation(line: 956, column: 4, scope: !2133)
!2138 = !DILocation(line: 956, column: 9, scope: !2133)
!2139 = !DILocation(line: 957, column: 3, scope: !2133)
!2140 = !DILocation(line: 955, column: 43, scope: !2126)
!2141 = !DILocation(line: 955, column: 3, scope: !2126)
!2142 = distinct !{!2142, !2131, !2143}
!2143 = !DILocation(line: 957, column: 3, scope: !2123)
!2144 = !DILocation(line: 958, column: 2, scope: !2007)
!2145 = !DILocation(line: 875, column: 37, scope: !2003)
!2146 = !DILocation(line: 875, column: 2, scope: !2003)
!2147 = distinct !{!2147, !2004, !2148}
!2148 = !DILocation(line: 958, column: 2, scope: !2000)
!2149 = !DILocation(line: 968, column: 6, scope: !2150)
!2150 = distinct !DILexicalBlock(scope: !1938, file: !3, line: 968, column: 2)
!2151 = !DILocation(line: 0, scope: !2150)
!2152 = !DILocation(line: 968, column: 17, scope: !2153)
!2153 = distinct !DILexicalBlock(scope: !2150, file: !3, line: 968, column: 2)
!2154 = !DILocation(line: 968, column: 27, scope: !2153)
!2155 = !DILocation(line: 968, column: 25, scope: !2153)
!2156 = !DILocation(line: 968, column: 36, scope: !2153)
!2157 = !DILocation(line: 968, column: 15, scope: !2153)
!2158 = !DILocation(line: 968, column: 2, scope: !2150)
!2159 = !DILocation(line: 970, column: 11, scope: !2160)
!2160 = distinct !DILexicalBlock(scope: !2161, file: !3, line: 970, column: 3)
!2161 = distinct !DILexicalBlock(scope: !2153, file: !3, line: 968, column: 45)
!2162 = !DILocation(line: 970, column: 7, scope: !2160)
!2163 = !DILocation(line: 0, scope: !2160)
!2164 = !DILocation(line: 0, scope: !2161)
!2165 = !DILocation(line: 970, column: 34, scope: !2166)
!2166 = distinct !DILexicalBlock(scope: !2160, file: !3, line: 970, column: 3)
!2167 = !DILocation(line: 970, column: 26, scope: !2166)
!2168 = !DILocation(line: 970, column: 24, scope: !2166)
!2169 = !DILocation(line: 970, column: 3, scope: !2160)
!2170 = !DILocation(line: 971, column: 12, scope: !2171)
!2171 = distinct !DILexicalBlock(scope: !2166, file: !3, line: 970, column: 43)
!2172 = !DILocation(line: 971, column: 19, scope: !2171)
!2173 = !DILocation(line: 971, column: 17, scope: !2171)
!2174 = !DILocation(line: 971, column: 16, scope: !2171)
!2175 = !DILocation(line: 971, column: 10, scope: !2171)
!2176 = !DILocation(line: 972, column: 3, scope: !2171)
!2177 = !DILocation(line: 970, column: 40, scope: !2166)
!2178 = !DILocation(line: 970, column: 3, scope: !2166)
!2179 = distinct !{!2179, !2169, !2180}
!2180 = !DILocation(line: 972, column: 3, scope: !2160)
!2181 = !DILocation(line: 973, column: 3, scope: !2161)
!2182 = !DILocation(line: 973, column: 8, scope: !2161)
!2183 = !DILocation(line: 974, column: 2, scope: !2161)
!2184 = !DILocation(line: 968, column: 42, scope: !2153)
!2185 = !DILocation(line: 968, column: 2, scope: !2153)
!2186 = distinct !{!2186, !2158, !2187}
!2187 = !DILocation(line: 974, column: 2, scope: !2150)
!2188 = !DILocation(line: 981, column: 6, scope: !2189)
!2189 = distinct !DILexicalBlock(scope: !1938, file: !3, line: 981, column: 2)
!2190 = !DILocation(line: 0, scope: !2189)
!2191 = !DILocation(line: 981, column: 17, scope: !2192)
!2192 = distinct !DILexicalBlock(scope: !2189, file: !3, line: 981, column: 2)
!2193 = !DILocation(line: 981, column: 25, scope: !2192)
!2194 = !DILocation(line: 981, column: 24, scope: !2192)
!2195 = !DILocation(line: 981, column: 33, scope: !2192)
!2196 = !DILocation(line: 981, column: 15, scope: !2192)
!2197 = !DILocation(line: 981, column: 2, scope: !2189)
!2198 = !DILocation(line: 982, column: 9, scope: !2199)
!2199 = distinct !DILexicalBlock(scope: !2192, file: !3, line: 981, column: 41)
!2200 = !DILocation(line: 982, column: 16, scope: !2199)
!2201 = !DILocation(line: 982, column: 14, scope: !2199)
!2202 = !DILocation(line: 983, column: 16, scope: !2199)
!2203 = !DILocation(line: 983, column: 13, scope: !2199)
!2204 = !DILocation(line: 984, column: 2, scope: !2199)
!2205 = !DILocation(line: 981, column: 38, scope: !2192)
!2206 = !DILocation(line: 981, column: 2, scope: !2192)
!2207 = distinct !{!2207, !2197, !2208}
!2208 = !DILocation(line: 984, column: 2, scope: !2189)
!2209 = !DILocation(line: 986, column: 11, scope: !1938)
!2210 = !DILocation(line: 986, column: 9, scope: !1938)
!2211 = !DILocation(line: 987, column: 1, scope: !1938)
!2212 = distinct !DISubprogram(name: "setup_gpu", linkageName: "_ZL9setup_gpuv", scope: !3, file: !3, line: 1662, type: !666, scopeLine: 1662, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1156)
!2213 = !DILocation(line: 1710, column: 63, scope: !2214)
!2214 = distinct !DILexicalBlock(scope: !2212, file: !3, line: 1709, column: 5)
!2215 = !DILocation(line: 1710, column: 39, scope: !2214)
!2216 = !DILocation(line: 1709, column: 5, scope: !2212)
!2217 = !DILocation(line: 1711, column: 35, scope: !2218)
!2218 = distinct !DILexicalBlock(scope: !2214, file: !3, line: 1710, column: 83)
!2219 = !DILocation(line: 1712, column: 2, scope: !2218)
!2220 = !DILocation(line: 1714, column: 59, scope: !2221)
!2221 = distinct !DILexicalBlock(scope: !2214, file: !3, line: 1713, column: 6)
!2222 = !DILocation(line: 1714, column: 35, scope: !2221)
!2223 = !DILocation(line: 1717, column: 63, scope: !2224)
!2224 = distinct !DILexicalBlock(scope: !2212, file: !3, line: 1716, column: 5)
!2225 = !DILocation(line: 1717, column: 39, scope: !2224)
!2226 = !DILocation(line: 1716, column: 5, scope: !2212)
!2227 = !DILocation(line: 1718, column: 35, scope: !2228)
!2228 = distinct !DILexicalBlock(scope: !2224, file: !3, line: 1717, column: 83)
!2229 = !DILocation(line: 1719, column: 2, scope: !2228)
!2230 = !DILocation(line: 1721, column: 59, scope: !2231)
!2231 = distinct !DILexicalBlock(scope: !2224, file: !3, line: 1720, column: 6)
!2232 = !DILocation(line: 1721, column: 35, scope: !2231)
!2233 = !DILocation(line: 1724, column: 65, scope: !2234)
!2234 = distinct !DILexicalBlock(scope: !2212, file: !3, line: 1723, column: 5)
!2235 = !DILocation(line: 1724, column: 41, scope: !2234)
!2236 = !DILocation(line: 1723, column: 5, scope: !2212)
!2237 = !DILocation(line: 1725, column: 37, scope: !2238)
!2238 = distinct !DILexicalBlock(scope: !2234, file: !3, line: 1724, column: 85)
!2239 = !DILocation(line: 1726, column: 2, scope: !2238)
!2240 = !DILocation(line: 1728, column: 61, scope: !2241)
!2241 = distinct !DILexicalBlock(scope: !2234, file: !3, line: 1727, column: 6)
!2242 = !DILocation(line: 1728, column: 37, scope: !2241)
!2243 = !DILocation(line: 1731, column: 64, scope: !2244)
!2244 = distinct !DILexicalBlock(scope: !2212, file: !3, line: 1730, column: 5)
!2245 = !DILocation(line: 1731, column: 40, scope: !2244)
!2246 = !DILocation(line: 1730, column: 5, scope: !2212)
!2247 = !DILocation(line: 1732, column: 36, scope: !2248)
!2248 = distinct !DILexicalBlock(scope: !2244, file: !3, line: 1731, column: 84)
!2249 = !DILocation(line: 1733, column: 2, scope: !2248)
!2250 = !DILocation(line: 1735, column: 60, scope: !2251)
!2251 = distinct !DILexicalBlock(scope: !2244, file: !3, line: 1734, column: 6)
!2252 = !DILocation(line: 1735, column: 36, scope: !2251)
!2253 = !DILocation(line: 1738, column: 64, scope: !2254)
!2254 = distinct !DILexicalBlock(scope: !2212, file: !3, line: 1737, column: 5)
!2255 = !DILocation(line: 1738, column: 40, scope: !2254)
!2256 = !DILocation(line: 1737, column: 5, scope: !2212)
!2257 = !DILocation(line: 1739, column: 36, scope: !2258)
!2258 = distinct !DILexicalBlock(scope: !2254, file: !3, line: 1738, column: 84)
!2259 = !DILocation(line: 1740, column: 2, scope: !2258)
!2260 = !DILocation(line: 1742, column: 60, scope: !2261)
!2261 = distinct !DILexicalBlock(scope: !2254, file: !3, line: 1741, column: 6)
!2262 = !DILocation(line: 1742, column: 36, scope: !2261)
!2263 = !DILocation(line: 1745, column: 63, scope: !2264)
!2264 = distinct !DILexicalBlock(scope: !2212, file: !3, line: 1744, column: 5)
!2265 = !DILocation(line: 1745, column: 39, scope: !2264)
!2266 = !DILocation(line: 1744, column: 5, scope: !2212)
!2267 = !DILocation(line: 1746, column: 35, scope: !2268)
!2268 = distinct !DILexicalBlock(scope: !2264, file: !3, line: 1745, column: 83)
!2269 = !DILocation(line: 1747, column: 2, scope: !2268)
!2270 = !DILocation(line: 1749, column: 59, scope: !2271)
!2271 = distinct !DILexicalBlock(scope: !2264, file: !3, line: 1748, column: 6)
!2272 = !DILocation(line: 1749, column: 35, scope: !2271)
!2273 = !DILocation(line: 1752, column: 65, scope: !2274)
!2274 = distinct !DILexicalBlock(scope: !2212, file: !3, line: 1751, column: 5)
!2275 = !DILocation(line: 1752, column: 41, scope: !2274)
!2276 = !DILocation(line: 1751, column: 5, scope: !2212)
!2277 = !DILocation(line: 1753, column: 37, scope: !2278)
!2278 = distinct !DILexicalBlock(scope: !2274, file: !3, line: 1752, column: 85)
!2279 = !DILocation(line: 1754, column: 2, scope: !2278)
!2280 = !DILocation(line: 1756, column: 61, scope: !2281)
!2281 = distinct !DILexicalBlock(scope: !2274, file: !3, line: 1755, column: 6)
!2282 = !DILocation(line: 1756, column: 37, scope: !2281)
!2283 = !DILocation(line: 1759, column: 65, scope: !2284)
!2284 = distinct !DILexicalBlock(scope: !2212, file: !3, line: 1758, column: 5)
!2285 = !DILocation(line: 1759, column: 41, scope: !2284)
!2286 = !DILocation(line: 1758, column: 5, scope: !2212)
!2287 = !DILocation(line: 1760, column: 37, scope: !2288)
!2288 = distinct !DILexicalBlock(scope: !2284, file: !3, line: 1759, column: 85)
!2289 = !DILocation(line: 1761, column: 2, scope: !2288)
!2290 = !DILocation(line: 1763, column: 61, scope: !2291)
!2291 = distinct !DILexicalBlock(scope: !2284, file: !3, line: 1762, column: 6)
!2292 = !DILocation(line: 1763, column: 37, scope: !2291)
!2293 = !DILocation(line: 1766, column: 64, scope: !2294)
!2294 = distinct !DILexicalBlock(scope: !2212, file: !3, line: 1765, column: 5)
!2295 = !DILocation(line: 1766, column: 40, scope: !2294)
!2296 = !DILocation(line: 1765, column: 5, scope: !2212)
!2297 = !DILocation(line: 1767, column: 36, scope: !2298)
!2298 = distinct !DILexicalBlock(scope: !2294, file: !3, line: 1766, column: 84)
!2299 = !DILocation(line: 1768, column: 2, scope: !2298)
!2300 = !DILocation(line: 1770, column: 58, scope: !2301)
!2301 = distinct !DILexicalBlock(scope: !2294, file: !3, line: 1769, column: 6)
!2302 = !DILocation(line: 1770, column: 35, scope: !2301)
!2303 = !DILocation(line: 1773, column: 63, scope: !2304)
!2304 = distinct !DILexicalBlock(scope: !2212, file: !3, line: 1772, column: 5)
!2305 = !DILocation(line: 1773, column: 39, scope: !2304)
!2306 = !DILocation(line: 1772, column: 5, scope: !2212)
!2307 = !DILocation(line: 1774, column: 35, scope: !2308)
!2308 = distinct !DILexicalBlock(scope: !2304, file: !3, line: 1773, column: 83)
!2309 = !DILocation(line: 1775, column: 2, scope: !2308)
!2310 = !DILocation(line: 1777, column: 57, scope: !2311)
!2311 = distinct !DILexicalBlock(scope: !2304, file: !3, line: 1776, column: 6)
!2312 = !DILocation(line: 1777, column: 34, scope: !2311)
!2313 = !DILocation(line: 1780, column: 66, scope: !2314)
!2314 = distinct !DILexicalBlock(scope: !2212, file: !3, line: 1779, column: 5)
!2315 = !DILocation(line: 1780, column: 42, scope: !2314)
!2316 = !DILocation(line: 1779, column: 5, scope: !2212)
!2317 = !DILocation(line: 1781, column: 38, scope: !2318)
!2318 = distinct !DILexicalBlock(scope: !2314, file: !3, line: 1780, column: 86)
!2319 = !DILocation(line: 1782, column: 2, scope: !2318)
!2320 = !DILocation(line: 1784, column: 62, scope: !2321)
!2321 = distinct !DILexicalBlock(scope: !2314, file: !3, line: 1783, column: 6)
!2322 = !DILocation(line: 1784, column: 38, scope: !2321)
!2323 = !DILocation(line: 1787, column: 57, scope: !2212)
!2324 = !DILocation(line: 1787, column: 48, scope: !2212)
!2325 = !DILocation(line: 1787, column: 33, scope: !2212)
!2326 = !DILocation(line: 1787, column: 32, scope: !2212)
!2327 = !DILocation(line: 1787, column: 31, scope: !2212)
!2328 = !DILocation(line: 1788, column: 57, scope: !2212)
!2329 = !DILocation(line: 1788, column: 48, scope: !2212)
!2330 = !DILocation(line: 1788, column: 33, scope: !2212)
!2331 = !DILocation(line: 1788, column: 32, scope: !2212)
!2332 = !DILocation(line: 1788, column: 31, scope: !2212)
!2333 = !DILocation(line: 1789, column: 33, scope: !2212)
!2334 = !DILocation(line: 1790, column: 58, scope: !2212)
!2335 = !DILocation(line: 1790, column: 49, scope: !2212)
!2336 = !DILocation(line: 1790, column: 34, scope: !2212)
!2337 = !DILocation(line: 1790, column: 33, scope: !2212)
!2338 = !DILocation(line: 1790, column: 32, scope: !2212)
!2339 = !DILocation(line: 1791, column: 58, scope: !2212)
!2340 = !DILocation(line: 1791, column: 49, scope: !2212)
!2341 = !DILocation(line: 1791, column: 34, scope: !2212)
!2342 = !DILocation(line: 1791, column: 33, scope: !2212)
!2343 = !DILocation(line: 1791, column: 32, scope: !2212)
!2344 = !DILocation(line: 1792, column: 57, scope: !2212)
!2345 = !DILocation(line: 1792, column: 48, scope: !2212)
!2346 = !DILocation(line: 1792, column: 33, scope: !2212)
!2347 = !DILocation(line: 1792, column: 32, scope: !2212)
!2348 = !DILocation(line: 1792, column: 31, scope: !2212)
!2349 = !DILocation(line: 1793, column: 51, scope: !2212)
!2350 = !DILocation(line: 1793, column: 50, scope: !2212)
!2351 = !DILocation(line: 1793, column: 35, scope: !2212)
!2352 = !DILocation(line: 1793, column: 34, scope: !2212)
!2353 = !DILocation(line: 1793, column: 33, scope: !2212)
!2354 = !DILocation(line: 1794, column: 33, scope: !2212)
!2355 = !DILocation(line: 1795, column: 58, scope: !2212)
!2356 = !DILocation(line: 1795, column: 49, scope: !2212)
!2357 = !DILocation(line: 1795, column: 34, scope: !2212)
!2358 = !DILocation(line: 1795, column: 33, scope: !2212)
!2359 = !DILocation(line: 1795, column: 32, scope: !2212)
!2360 = !DILocation(line: 1796, column: 57, scope: !2212)
!2361 = !DILocation(line: 1796, column: 48, scope: !2212)
!2362 = !DILocation(line: 1796, column: 33, scope: !2212)
!2363 = !DILocation(line: 1796, column: 32, scope: !2212)
!2364 = !DILocation(line: 1796, column: 31, scope: !2212)
!2365 = !DILocation(line: 1797, column: 60, scope: !2212)
!2366 = !DILocation(line: 1797, column: 51, scope: !2212)
!2367 = !DILocation(line: 1797, column: 36, scope: !2212)
!2368 = !DILocation(line: 1797, column: 35, scope: !2212)
!2369 = !DILocation(line: 1797, column: 34, scope: !2212)
!2370 = !DILocation(line: 1799, column: 68, scope: !2212)
!2371 = !DILocation(line: 1799, column: 46, scope: !2212)
!2372 = !DILocation(line: 1799, column: 38, scope: !2212)
!2373 = !DILocation(line: 1799, column: 23, scope: !2212)
!2374 = !DILocation(line: 1799, column: 22, scope: !2212)
!2375 = !DILocation(line: 1801, column: 19, scope: !2212)
!2376 = !DILocation(line: 1801, column: 39, scope: !2212)
!2377 = !DILocation(line: 1801, column: 18, scope: !2212)
!2378 = !DILocation(line: 1802, column: 20, scope: !2212)
!2379 = !DILocation(line: 1803, column: 20, scope: !2212)
!2380 = !DILocation(line: 1804, column: 16, scope: !2212)
!2381 = !DILocation(line: 1805, column: 18, scope: !2212)
!2382 = !DILocation(line: 1806, column: 18, scope: !2212)
!2383 = !DILocation(line: 1807, column: 18, scope: !2212)
!2384 = !DILocation(line: 1808, column: 15, scope: !2212)
!2385 = !DILocation(line: 1809, column: 15, scope: !2212)
!2386 = !DILocation(line: 1810, column: 15, scope: !2212)
!2387 = !DILocation(line: 1811, column: 15, scope: !2212)
!2388 = !DILocation(line: 1812, column: 15, scope: !2212)
!2389 = !DILocation(line: 1813, column: 15, scope: !2212)
!2390 = !DILocation(line: 1814, column: 17, scope: !2212)
!2391 = !DILocation(line: 1815, column: 15, scope: !2212)
!2392 = !DILocation(line: 1816, column: 19, scope: !2212)
!2393 = !DILocation(line: 1817, column: 18, scope: !2212)
!2394 = !DILocation(line: 1818, column: 17, scope: !2212)
!2395 = !DILocation(line: 1819, column: 24, scope: !2212)
!2396 = !DILocation(line: 1820, column: 24, scope: !2212)
!2397 = !DILocation(line: 1822, column: 30, scope: !2212)
!2398 = !{!"5"}
!2399 = !DILocation(line: 1822, column: 23, scope: !2212)
!2400 = !{!2398}
!2401 = !DILocation(line: 1822, column: 14, scope: !2212)
!2402 = !DILocation(line: 1822, column: 13, scope: !2212)
!2403 = !DILocation(line: 1823, column: 34, scope: !2212)
!2404 = !{!"1"}
!2405 = !DILocation(line: 1823, column: 27, scope: !2212)
!2406 = !{!2404}
!2407 = !DILocation(line: 1823, column: 18, scope: !2212)
!2408 = !DILocation(line: 1823, column: 17, scope: !2212)
!2409 = !DILocation(line: 1843, column: 13, scope: !2212)
!2410 = !DILocation(line: 1843, column: 28, scope: !2212)
!2411 = !DILocation(line: 1843, column: 36, scope: !2212)
!2412 = !DILocation(line: 1843, column: 2, scope: !2212)
!2413 = !{!""}
!2414 = !DILocation(line: 1852, column: 33, scope: !2212)
!2415 = !DILocation(line: 1852, column: 64, scope: !2212)
!2416 = !DILocation(line: 1852, column: 32, scope: !2212)
!2417 = !DILocation(line: 1853, column: 33, scope: !2212)
!2418 = !DILocation(line: 1853, column: 64, scope: !2212)
!2419 = !DILocation(line: 1853, column: 32, scope: !2212)
!2420 = !DILocation(line: 1854, column: 35, scope: !2212)
!2421 = !DILocation(line: 1854, column: 68, scope: !2212)
!2422 = !DILocation(line: 1854, column: 34, scope: !2212)
!2423 = !DILocation(line: 1855, column: 34, scope: !2212)
!2424 = !DILocation(line: 1855, column: 66, scope: !2212)
!2425 = !DILocation(line: 1855, column: 33, scope: !2212)
!2426 = !DILocation(line: 1856, column: 34, scope: !2212)
!2427 = !DILocation(line: 1856, column: 66, scope: !2212)
!2428 = !DILocation(line: 1856, column: 33, scope: !2212)
!2429 = !DILocation(line: 1857, column: 33, scope: !2212)
!2430 = !DILocation(line: 1857, column: 64, scope: !2212)
!2431 = !DILocation(line: 1857, column: 32, scope: !2212)
!2432 = !DILocation(line: 1858, column: 35, scope: !2212)
!2433 = !DILocation(line: 1858, column: 68, scope: !2212)
!2434 = !DILocation(line: 1858, column: 34, scope: !2212)
!2435 = !DILocation(line: 1859, column: 35, scope: !2212)
!2436 = !DILocation(line: 1859, column: 68, scope: !2212)
!2437 = !DILocation(line: 1859, column: 34, scope: !2212)
!2438 = !DILocation(line: 1860, column: 34, scope: !2212)
!2439 = !DILocation(line: 1860, column: 66, scope: !2212)
!2440 = !DILocation(line: 1860, column: 33, scope: !2212)
!2441 = !DILocation(line: 1861, column: 33, scope: !2212)
!2442 = !DILocation(line: 1861, column: 64, scope: !2212)
!2443 = !DILocation(line: 1861, column: 32, scope: !2212)
!2444 = !DILocation(line: 1862, column: 36, scope: !2212)
!2445 = !DILocation(line: 1862, column: 70, scope: !2212)
!2446 = !DILocation(line: 1862, column: 35, scope: !2212)
!2447 = !DILocation(line: 1864, column: 35, scope: !2212)
!2448 = !DILocation(line: 1864, column: 64, scope: !2212)
!2449 = !DILocation(line: 1864, column: 34, scope: !2212)
!2450 = !DILocation(line: 1865, column: 35, scope: !2212)
!2451 = !DILocation(line: 1865, column: 64, scope: !2212)
!2452 = !DILocation(line: 1865, column: 34, scope: !2212)
!2453 = !DILocation(line: 1866, column: 37, scope: !2212)
!2454 = !DILocation(line: 1866, column: 68, scope: !2212)
!2455 = !DILocation(line: 1866, column: 36, scope: !2212)
!2456 = !DILocation(line: 1867, column: 36, scope: !2212)
!2457 = !DILocation(line: 1867, column: 66, scope: !2212)
!2458 = !DILocation(line: 1867, column: 35, scope: !2212)
!2459 = !DILocation(line: 1868, column: 36, scope: !2212)
!2460 = !DILocation(line: 1868, column: 66, scope: !2212)
!2461 = !DILocation(line: 1868, column: 35, scope: !2212)
!2462 = !DILocation(line: 1869, column: 35, scope: !2212)
!2463 = !DILocation(line: 1869, column: 64, scope: !2212)
!2464 = !DILocation(line: 1869, column: 34, scope: !2212)
!2465 = !DILocation(line: 1870, column: 37, scope: !2212)
!2466 = !DILocation(line: 1870, column: 68, scope: !2212)
!2467 = !DILocation(line: 1870, column: 36, scope: !2212)
!2468 = !DILocation(line: 1871, column: 37, scope: !2212)
!2469 = !DILocation(line: 1871, column: 68, scope: !2212)
!2470 = !DILocation(line: 1871, column: 36, scope: !2212)
!2471 = !DILocation(line: 1872, column: 36, scope: !2212)
!2472 = !DILocation(line: 1872, column: 66, scope: !2212)
!2473 = !DILocation(line: 1872, column: 35, scope: !2212)
!2474 = !DILocation(line: 1873, column: 35, scope: !2212)
!2475 = !DILocation(line: 1873, column: 64, scope: !2212)
!2476 = !DILocation(line: 1873, column: 34, scope: !2212)
!2477 = !DILocation(line: 1874, column: 38, scope: !2212)
!2478 = !DILocation(line: 1874, column: 70, scope: !2212)
!2479 = !DILocation(line: 1874, column: 37, scope: !2212)
!2480 = !DILocation(line: 1875, column: 1, scope: !2212)
!2481 = distinct !DISubprogram(name: "conj_grad_gpu", linkageName: "_ZL13conj_grad_gpuPd", scope: !3, file: !3, line: 989, type: !2482, scopeLine: 989, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1156)
!2482 = !DISubroutineType(types: !2483)
!2483 = !{null, !99}
!2484 = !DILocalVariable(name: "rnorm", arg: 1, scope: !2481, file: !3, line: 989, type: !99)
!2485 = !DILocation(line: 0, scope: !2481)
!2486 = !DILocalVariable(name: "d", scope: !2481, file: !3, line: 990, type: !100)
!2487 = !DILocation(line: 990, column: 9, scope: !2481)
!2488 = !DILocalVariable(name: "sum", scope: !2481, file: !3, line: 990, type: !100)
!2489 = !DILocation(line: 990, column: 12, scope: !2481)
!2490 = !DILocalVariable(name: "rho", scope: !2481, file: !3, line: 990, type: !100)
!2491 = !DILocation(line: 990, column: 17, scope: !2481)
!2492 = !DILocalVariable(name: "cgitmax", scope: !2481, file: !3, line: 991, type: !97)
!2493 = !DILocation(line: 994, column: 2, scope: !2481)
!2494 = !DILocation(line: 997, column: 2, scope: !2481)
!2495 = !DILocalVariable(name: "cgit", scope: !2481, file: !3, line: 991, type: !97)
!2496 = !DILocation(line: 1000, column: 6, scope: !2497)
!2497 = distinct !DILexicalBlock(scope: !2481, file: !3, line: 1000, column: 2)
!2498 = !DILocation(line: 0, scope: !2497)
!2499 = !DILocation(line: 1000, column: 21, scope: !2500)
!2500 = distinct !DILexicalBlock(scope: !2497, file: !3, line: 1000, column: 2)
!2501 = !DILocation(line: 1000, column: 2, scope: !2497)
!2502 = !DILocation(line: 1002, column: 3, scope: !2503)
!2503 = distinct !DILexicalBlock(scope: !2500, file: !3, line: 1000, column: 40)
!2504 = !DILocation(line: 1005, column: 3, scope: !2503)
!2505 = !DILocation(line: 1007, column: 11, scope: !2503)
!2506 = !DILocation(line: 1007, column: 17, scope: !2503)
!2507 = !DILocation(line: 1007, column: 15, scope: !2503)
!2508 = !DILocalVariable(name: "alpha", scope: !2481, file: !3, line: 990, type: !100)
!2509 = !DILocation(line: 1010, column: 10, scope: !2503)
!2510 = !DILocalVariable(name: "rho0", scope: !2481, file: !3, line: 990, type: !100)
!2511 = !DILocation(line: 1013, column: 3, scope: !2503)
!2512 = !DILocation(line: 1016, column: 3, scope: !2503)
!2513 = !DILocation(line: 1019, column: 10, scope: !2503)
!2514 = !DILocation(line: 1019, column: 14, scope: !2503)
!2515 = !DILocalVariable(name: "beta", scope: !2481, file: !3, line: 990, type: !100)
!2516 = !DILocation(line: 1022, column: 3, scope: !2503)
!2517 = !DILocation(line: 1023, column: 2, scope: !2503)
!2518 = !DILocation(line: 1000, column: 37, scope: !2500)
!2519 = !DILocation(line: 1000, column: 2, scope: !2500)
!2520 = distinct !{!2520, !2501, !2521}
!2521 = !DILocation(line: 1023, column: 2, scope: !2497)
!2522 = !DILocation(line: 1026, column: 2, scope: !2481)
!2523 = !DILocation(line: 1029, column: 2, scope: !2481)
!2524 = !DILocation(line: 1031, column: 16, scope: !2481)
!2525 = !DILocation(line: 1031, column: 11, scope: !2481)
!2526 = !DILocation(line: 1031, column: 9, scope: !2481)
!2527 = !DILocation(line: 1032, column: 1, scope: !2481)
!2528 = distinct !DISubprogram(name: "gpu_kernel_ten_host", linkageName: "_ZL19gpu_kernel_ten_hostPdS_", scope: !3, file: !3, line: 1432, type: !2529, scopeLine: 1433, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1156)
!2529 = !DISubroutineType(types: !2530)
!2530 = !{null, !99, !99}
!2531 = !DILocalVariable(name: "norm_temp1", arg: 1, scope: !2528, file: !3, line: 1432, type: !99)
!2532 = !DILocation(line: 0, scope: !2528)
!2533 = !DILocalVariable(name: "norm_temp2", arg: 2, scope: !2528, file: !3, line: 1433, type: !99)
!2534 = !DILocation(line: 1437, column: 21, scope: !2528)
!2535 = !DILocation(line: 1437, column: 51, scope: !2528)
!2536 = !DILocation(line: 1437, column: 18, scope: !2528)
!2537 = !DILocation(line: 1437, column: 117, scope: !2528)
!2538 = !DILocation(line: 1437, column: 136, scope: !2528)
!2539 = !DILocation(line: 1437, column: 145, scope: !2528)
!2540 = !DILocation(line: 1438, column: 21, scope: !2528)
!2541 = !DILocation(line: 1438, column: 51, scope: !2528)
!2542 = !DILocation(line: 1438, column: 18, scope: !2528)
!2543 = !DILocation(line: 1438, column: 117, scope: !2528)
!2544 = !DILocation(line: 1438, column: 140, scope: !2528)
!2545 = !DILocation(line: 1438, column: 149, scope: !2528)
!2546 = !DILocation(line: 1440, column: 20, scope: !2528)
!2547 = !DILocation(line: 1441, column: 24, scope: !2528)
!2548 = !DILocation(line: 1442, column: 13, scope: !2528)
!2549 = !DILocation(line: 1442, column: 26, scope: !2528)
!2550 = !DILocation(line: 1442, column: 46, scope: !2528)
!2551 = !DILocation(line: 1442, column: 2, scope: !2528)
!2552 = !DILocalVariable(name: "i", scope: !2553, file: !3, line: 1445, type: !97)
!2553 = distinct !DILexicalBlock(scope: !2528, file: !3, line: 1445, column: 2)
!2554 = !DILocation(line: 0, scope: !2553)
!2555 = !DILocation(line: 1445, column: 6, scope: !2553)
!2556 = !DILocation(line: 1445, column: 17, scope: !2557)
!2557 = distinct !DILexicalBlock(scope: !2553, file: !3, line: 1445, column: 2)
!2558 = !DILocation(line: 1445, column: 16, scope: !2557)
!2559 = !DILocation(line: 1445, column: 2, scope: !2553)
!2560 = !DILocation(line: 1445, column: 73, scope: !2561)
!2561 = distinct !DILexicalBlock(scope: !2557, file: !3, line: 1445, column: 52)
!2562 = !DILocation(line: 1445, column: 71, scope: !2561)
!2563 = !DILocation(line: 1445, column: 112, scope: !2561)
!2564 = !DILocation(line: 1445, column: 110, scope: !2561)
!2565 = !DILocation(line: 1445, column: 131, scope: !2561)
!2566 = !DILocation(line: 1445, column: 49, scope: !2557)
!2567 = !DILocation(line: 1445, column: 2, scope: !2557)
!2568 = distinct !{!2568, !2559, !2569}
!2569 = !DILocation(line: 1445, column: 131, scope: !2553)
!2570 = !DILocation(line: 1446, column: 14, scope: !2528)
!2571 = !DILocation(line: 1446, column: 13, scope: !2528)
!2572 = !DILocation(line: 1447, column: 14, scope: !2528)
!2573 = !DILocation(line: 1447, column: 13, scope: !2528)
!2574 = !DILocation(line: 1451, column: 1, scope: !2528)
!2575 = distinct !DISubprogram(name: "gpu_kernel_eleven_host", linkageName: "_ZL22gpu_kernel_eleven_hostd", scope: !3, file: !3, line: 1519, type: !2576, scopeLine: 1519, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1156)
!2576 = !DISubroutineType(types: !2577)
!2577 = !{null, !100}
!2578 = !DILocalVariable(name: "norm_temp2", arg: 1, scope: !2575, file: !3, line: 1519, type: !100)
!2579 = !DILocation(line: 0, scope: !2575)
!2580 = !DILocation(line: 1523, column: 29, scope: !2575)
!2581 = !DILocation(line: 1524, column: 3, scope: !2575)
!2582 = !DILocation(line: 1523, column: 26, scope: !2575)
!2583 = !DILocation(line: 1526, column: 5, scope: !2575)
!2584 = !DILocation(line: 1527, column: 5, scope: !2575)
!2585 = !DILocation(line: 1531, column: 1, scope: !2575)
!2586 = distinct !DISubprogram(name: "release_gpu", linkageName: "_ZL11release_gpuv", scope: !3, file: !3, line: 1642, type: !666, scopeLine: 1642, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1156)
!2587 = !DILocation(line: 1660, column: 1, scope: !2586)
!2588 = distinct !DISubprogram(name: "gpu_kernel_one_host", linkageName: "_ZL19gpu_kernel_one_hostv", scope: !3, file: !3, line: 1034, type: !666, scopeLine: 1034, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1156)
!2589 = !DILocation(line: 1038, column: 26, scope: !2588)
!2590 = !DILocation(line: 1039, column: 3, scope: !2588)
!2591 = !DILocation(line: 1038, column: 23, scope: !2588)
!2592 = !DILocation(line: 1040, column: 5, scope: !2588)
!2593 = !DILocation(line: 1041, column: 5, scope: !2588)
!2594 = !DILocation(line: 1042, column: 5, scope: !2588)
!2595 = !DILocation(line: 1043, column: 5, scope: !2588)
!2596 = !DILocation(line: 1044, column: 5, scope: !2588)
!2597 = !DILocation(line: 1048, column: 1, scope: !2588)
!2598 = distinct !DISubprogram(name: "gpu_kernel_two_host", linkageName: "_ZL19gpu_kernel_two_hostPd", scope: !3, file: !3, line: 1064, type: !2482, scopeLine: 1064, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1156)
!2599 = !DILocalVariable(name: "rho_host", arg: 1, scope: !2598, file: !3, line: 1064, type: !99)
!2600 = !DILocation(line: 0, scope: !2598)
!2601 = !DILocation(line: 1068, column: 26, scope: !2598)
!2602 = !DILocation(line: 1069, column: 3, scope: !2598)
!2603 = !DILocation(line: 1068, column: 23, scope: !2598)
!2604 = !DILocation(line: 1071, column: 5, scope: !2598)
!2605 = !DILocation(line: 1072, column: 5, scope: !2598)
!2606 = !DILocation(line: 1073, column: 5, scope: !2598)
!2607 = !DILocation(line: 1074, column: 20, scope: !2598)
!2608 = !DILocation(line: 1075, column: 13, scope: !2598)
!2609 = !DILocation(line: 1075, column: 26, scope: !2598)
!2610 = !DILocation(line: 1075, column: 46, scope: !2598)
!2611 = !DILocation(line: 1075, column: 2, scope: !2598)
!2612 = !DILocalVariable(name: "i", scope: !2613, file: !3, line: 1076, type: !97)
!2613 = distinct !DILexicalBlock(scope: !2598, file: !3, line: 1076, column: 2)
!2614 = !DILocation(line: 0, scope: !2613)
!2615 = !DILocation(line: 1076, column: 6, scope: !2613)
!2616 = !DILocation(line: 1076, column: 17, scope: !2617)
!2617 = distinct !DILexicalBlock(scope: !2613, file: !3, line: 1076, column: 2)
!2618 = !DILocation(line: 1076, column: 16, scope: !2617)
!2619 = !DILocation(line: 1076, column: 2, scope: !2613)
!2620 = !DILocation(line: 1076, column: 73, scope: !2621)
!2621 = distinct !DILexicalBlock(scope: !2617, file: !3, line: 1076, column: 52)
!2622 = !DILocation(line: 1076, column: 71, scope: !2621)
!2623 = !DILocation(line: 1076, column: 88, scope: !2621)
!2624 = !DILocation(line: 1076, column: 49, scope: !2617)
!2625 = !DILocation(line: 1076, column: 2, scope: !2617)
!2626 = distinct !{!2626, !2619, !2627}
!2627 = !DILocation(line: 1076, column: 88, scope: !2613)
!2628 = !DILocation(line: 1077, column: 12, scope: !2598)
!2629 = !DILocation(line: 1077, column: 11, scope: !2598)
!2630 = !DILocation(line: 1081, column: 1, scope: !2598)
!2631 = distinct !DISubprogram(name: "gpu_kernel_three_host", linkageName: "_ZL21gpu_kernel_three_hostv", scope: !3, file: !3, line: 1117, type: !666, scopeLine: 1117, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1156)
!2632 = !DILocation(line: 1121, column: 28, scope: !2631)
!2633 = !DILocation(line: 1122, column: 3, scope: !2631)
!2634 = !DILocation(line: 1121, column: 25, scope: !2631)
!2635 = !DILocation(line: 1124, column: 5, scope: !2631)
!2636 = !DILocation(line: 1125, column: 5, scope: !2631)
!2637 = !DILocation(line: 1126, column: 5, scope: !2631)
!2638 = !DILocation(line: 1127, column: 5, scope: !2631)
!2639 = !DILocation(line: 1128, column: 5, scope: !2631)
!2640 = !DILocation(line: 1132, column: 1, scope: !2631)
!2641 = distinct !DISubprogram(name: "gpu_kernel_four_host", linkageName: "_ZL20gpu_kernel_four_hostPd", scope: !3, file: !3, line: 1169, type: !2482, scopeLine: 1169, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1156)
!2642 = !DILocalVariable(name: "d_host", arg: 1, scope: !2641, file: !3, line: 1169, type: !99)
!2643 = !DILocation(line: 0, scope: !2641)
!2644 = !DILocation(line: 1173, column: 27, scope: !2641)
!2645 = !DILocation(line: 1174, column: 3, scope: !2641)
!2646 = !DILocation(line: 1173, column: 24, scope: !2641)
!2647 = !DILocation(line: 1176, column: 5, scope: !2641)
!2648 = !DILocation(line: 1177, column: 5, scope: !2641)
!2649 = !DILocation(line: 1178, column: 5, scope: !2641)
!2650 = !DILocation(line: 1179, column: 5, scope: !2641)
!2651 = !DILocation(line: 1180, column: 20, scope: !2641)
!2652 = !DILocation(line: 1181, column: 13, scope: !2641)
!2653 = !DILocation(line: 1181, column: 26, scope: !2641)
!2654 = !DILocation(line: 1181, column: 46, scope: !2641)
!2655 = !DILocation(line: 1181, column: 2, scope: !2641)
!2656 = !DILocalVariable(name: "i", scope: !2657, file: !3, line: 1182, type: !97)
!2657 = distinct !DILexicalBlock(scope: !2641, file: !3, line: 1182, column: 2)
!2658 = !DILocation(line: 0, scope: !2657)
!2659 = !DILocation(line: 1182, column: 6, scope: !2657)
!2660 = !DILocation(line: 1182, column: 17, scope: !2661)
!2661 = distinct !DILexicalBlock(scope: !2657, file: !3, line: 1182, column: 2)
!2662 = !DILocation(line: 1182, column: 16, scope: !2661)
!2663 = !DILocation(line: 1182, column: 2, scope: !2657)
!2664 = !DILocation(line: 1182, column: 74, scope: !2665)
!2665 = distinct !DILexicalBlock(scope: !2661, file: !3, line: 1182, column: 53)
!2666 = !DILocation(line: 1182, column: 72, scope: !2665)
!2667 = !DILocation(line: 1182, column: 89, scope: !2665)
!2668 = !DILocation(line: 1182, column: 50, scope: !2661)
!2669 = !DILocation(line: 1182, column: 2, scope: !2661)
!2670 = distinct !{!2670, !2663, !2671}
!2671 = !DILocation(line: 1182, column: 89, scope: !2657)
!2672 = !DILocation(line: 1183, column: 10, scope: !2641)
!2673 = !DILocation(line: 1183, column: 9, scope: !2641)
!2674 = !DILocation(line: 1187, column: 1, scope: !2641)
!2675 = distinct !DISubprogram(name: "gpu_kernel_five_host", linkageName: "_ZL20gpu_kernel_five_hostd", scope: !3, file: !3, line: 1225, type: !2576, scopeLine: 1225, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1156)
!2676 = !DILocalVariable(name: "alpha_host", arg: 1, scope: !2675, file: !3, line: 1225, type: !100)
!2677 = !DILocation(line: 0, scope: !2675)
!2678 = !DILocation(line: 1229, column: 22, scope: !2675)
!2679 = !DILocation(line: 1230, column: 3, scope: !2675)
!2680 = !DILocation(line: 1229, column: 19, scope: !2675)
!2681 = !DILocation(line: 1232, column: 5, scope: !2675)
!2682 = !DILocation(line: 1233, column: 5, scope: !2675)
!2683 = !DILocation(line: 1234, column: 22, scope: !2675)
!2684 = !DILocation(line: 1235, column: 3, scope: !2675)
!2685 = !DILocation(line: 1234, column: 19, scope: !2675)
!2686 = !DILocation(line: 1237, column: 5, scope: !2675)
!2687 = !DILocation(line: 1238, column: 5, scope: !2675)
!2688 = !DILocation(line: 1242, column: 1, scope: !2675)
!2689 = distinct !DISubprogram(name: "gpu_kernel_six_host", linkageName: "_ZL19gpu_kernel_six_hostPd", scope: !3, file: !3, line: 1260, type: !2482, scopeLine: 1260, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1156)
!2690 = !DILocalVariable(name: "rho_host", arg: 1, scope: !2689, file: !3, line: 1260, type: !99)
!2691 = !DILocation(line: 0, scope: !2689)
!2692 = !DILocation(line: 1264, column: 26, scope: !2689)
!2693 = !DILocation(line: 1265, column: 3, scope: !2689)
!2694 = !DILocation(line: 1264, column: 23, scope: !2689)
!2695 = !DILocation(line: 1267, column: 5, scope: !2689)
!2696 = !DILocation(line: 1268, column: 5, scope: !2689)
!2697 = !DILocation(line: 1269, column: 20, scope: !2689)
!2698 = !DILocation(line: 1270, column: 13, scope: !2689)
!2699 = !DILocation(line: 1270, column: 26, scope: !2689)
!2700 = !DILocation(line: 1270, column: 46, scope: !2689)
!2701 = !DILocation(line: 1270, column: 2, scope: !2689)
!2702 = !DILocalVariable(name: "i", scope: !2703, file: !3, line: 1271, type: !97)
!2703 = distinct !DILexicalBlock(scope: !2689, file: !3, line: 1271, column: 2)
!2704 = !DILocation(line: 0, scope: !2703)
!2705 = !DILocation(line: 1271, column: 6, scope: !2703)
!2706 = !DILocation(line: 1271, column: 17, scope: !2707)
!2707 = distinct !DILexicalBlock(scope: !2703, file: !3, line: 1271, column: 2)
!2708 = !DILocation(line: 1271, column: 16, scope: !2707)
!2709 = !DILocation(line: 1271, column: 2, scope: !2703)
!2710 = !DILocation(line: 1271, column: 73, scope: !2711)
!2711 = distinct !DILexicalBlock(scope: !2707, file: !3, line: 1271, column: 52)
!2712 = !DILocation(line: 1271, column: 71, scope: !2711)
!2713 = !DILocation(line: 1271, column: 88, scope: !2711)
!2714 = !DILocation(line: 1271, column: 49, scope: !2707)
!2715 = !DILocation(line: 1271, column: 2, scope: !2707)
!2716 = distinct !{!2716, !2709, !2717}
!2717 = !DILocation(line: 1271, column: 88, scope: !2703)
!2718 = !DILocation(line: 1272, column: 12, scope: !2689)
!2719 = !DILocation(line: 1272, column: 11, scope: !2689)
!2720 = !DILocation(line: 1276, column: 1, scope: !2689)
!2721 = distinct !DISubprogram(name: "gpu_kernel_seven_host", linkageName: "_ZL21gpu_kernel_seven_hostd", scope: !3, file: !3, line: 1306, type: !2576, scopeLine: 1306, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1156)
!2722 = !DILocalVariable(name: "beta_host", arg: 1, scope: !2721, file: !3, line: 1306, type: !100)
!2723 = !DILocation(line: 0, scope: !2721)
!2724 = !DILocation(line: 1310, column: 28, scope: !2721)
!2725 = !DILocation(line: 1311, column: 3, scope: !2721)
!2726 = !DILocation(line: 1310, column: 25, scope: !2721)
!2727 = !DILocation(line: 1313, column: 5, scope: !2721)
!2728 = !DILocation(line: 1314, column: 5, scope: !2721)
!2729 = !DILocation(line: 1318, column: 1, scope: !2721)
!2730 = distinct !DISubprogram(name: "gpu_kernel_eight_host", linkageName: "_ZL21gpu_kernel_eight_hostv", scope: !3, file: !3, line: 1328, type: !666, scopeLine: 1328, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1156)
!2731 = !DILocation(line: 1332, column: 28, scope: !2730)
!2732 = !DILocation(line: 1333, column: 3, scope: !2730)
!2733 = !DILocation(line: 1332, column: 25, scope: !2730)
!2734 = !DILocation(line: 1335, column: 5, scope: !2730)
!2735 = !DILocation(line: 1336, column: 5, scope: !2730)
!2736 = !DILocation(line: 1337, column: 5, scope: !2730)
!2737 = !DILocation(line: 1338, column: 5, scope: !2730)
!2738 = !DILocation(line: 1339, column: 5, scope: !2730)
!2739 = !DILocation(line: 1343, column: 1, scope: !2730)
!2740 = distinct !DISubprogram(name: "gpu_kernel_nine_host", linkageName: "_ZL20gpu_kernel_nine_hostPd", scope: !3, file: !3, line: 1380, type: !2482, scopeLine: 1380, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1156)
!2741 = !DILocalVariable(name: "sum_host", arg: 1, scope: !2740, file: !3, line: 1380, type: !99)
!2742 = !DILocation(line: 0, scope: !2740)
!2743 = !DILocation(line: 1384, column: 27, scope: !2740)
!2744 = !DILocation(line: 1385, column: 3, scope: !2740)
!2745 = !DILocation(line: 1384, column: 24, scope: !2740)
!2746 = !DILocation(line: 1387, column: 5, scope: !2740)
!2747 = !DILocation(line: 1388, column: 5, scope: !2740)
!2748 = !DILocation(line: 1389, column: 5, scope: !2740)
!2749 = !DILocation(line: 1390, column: 5, scope: !2740)
!2750 = !DILocation(line: 1391, column: 20, scope: !2740)
!2751 = !DILocation(line: 1392, column: 13, scope: !2740)
!2752 = !DILocation(line: 1392, column: 26, scope: !2740)
!2753 = !DILocation(line: 1392, column: 46, scope: !2740)
!2754 = !DILocation(line: 1392, column: 2, scope: !2740)
!2755 = !DILocalVariable(name: "i", scope: !2756, file: !3, line: 1393, type: !97)
!2756 = distinct !DILexicalBlock(scope: !2740, file: !3, line: 1393, column: 2)
!2757 = !DILocation(line: 0, scope: !2756)
!2758 = !DILocation(line: 1393, column: 6, scope: !2756)
!2759 = !DILocation(line: 1393, column: 17, scope: !2760)
!2760 = distinct !DILexicalBlock(scope: !2756, file: !3, line: 1393, column: 2)
!2761 = !DILocation(line: 1393, column: 16, scope: !2760)
!2762 = !DILocation(line: 1393, column: 2, scope: !2756)
!2763 = !DILocation(line: 1393, column: 74, scope: !2764)
!2764 = distinct !DILexicalBlock(scope: !2760, file: !3, line: 1393, column: 53)
!2765 = !DILocation(line: 1393, column: 72, scope: !2764)
!2766 = !DILocation(line: 1393, column: 89, scope: !2764)
!2767 = !DILocation(line: 1393, column: 50, scope: !2760)
!2768 = !DILocation(line: 1393, column: 2, scope: !2760)
!2769 = distinct !{!2769, !2762, !2770}
!2770 = !DILocation(line: 1393, column: 89, scope: !2756)
!2771 = !DILocation(line: 1394, column: 12, scope: !2740)
!2772 = !DILocation(line: 1394, column: 11, scope: !2740)
!2773 = !DILocation(line: 1398, column: 1, scope: !2740)
!2774 = distinct !DISubprogram(name: "sprnvc", linkageName: "_ZL6sprnvciiiPdPi", scope: !3, file: !3, line: 2070, type: !2775, scopeLine: 2070, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1156)
!2775 = !DISubroutineType(types: !2776)
!2776 = !{null, !97, !97, !97, !99, !98}
!2777 = !DILocalVariable(name: "n", arg: 1, scope: !2774, file: !3, line: 2070, type: !97)
!2778 = !DILocation(line: 0, scope: !2774)
!2779 = !DILocalVariable(name: "nz", arg: 2, scope: !2774, file: !3, line: 2070, type: !97)
!2780 = !DILocalVariable(name: "nn1", arg: 3, scope: !2774, file: !3, line: 2070, type: !97)
!2781 = !DILocalVariable(name: "v", arg: 4, scope: !2774, file: !3, line: 2070, type: !99)
!2782 = !DILocalVariable(name: "iv", arg: 5, scope: !2774, file: !3, line: 2070, type: !98)
!2783 = !DILocalVariable(name: "nzv", scope: !2774, file: !3, line: 2071, type: !97)
!2784 = !DILocation(line: 2076, column: 2, scope: !2774)
!2785 = !DILocation(line: 2076, column: 12, scope: !2774)
!2786 = !DILocation(line: 2077, column: 26, scope: !2787)
!2787 = distinct !DILexicalBlock(scope: !2774, file: !3, line: 2076, column: 17)
!2788 = !DILocation(line: 2077, column: 12, scope: !2787)
!2789 = !DILocalVariable(name: "vecelt", scope: !2774, file: !3, line: 2072, type: !100)
!2790 = !DILocation(line: 2084, column: 26, scope: !2787)
!2791 = !DILocation(line: 2084, column: 12, scope: !2787)
!2792 = !DILocalVariable(name: "vecloc", scope: !2774, file: !3, line: 2072, type: !100)
!2793 = !DILocation(line: 2085, column: 7, scope: !2787)
!2794 = !DILocation(line: 2085, column: 27, scope: !2787)
!2795 = !DILocalVariable(name: "i", scope: !2774, file: !3, line: 2071, type: !97)
!2796 = !DILocalVariable(name: "was_gen", scope: !2787, file: !3, line: 2093, type: !1752)
!2797 = !DILocation(line: 0, scope: !2787)
!2798 = !DILocalVariable(name: "ii", scope: !2774, file: !3, line: 2071, type: !97)
!2799 = !DILocation(line: 2094, column: 7, scope: !2800)
!2800 = distinct !DILexicalBlock(scope: !2787, file: !3, line: 2094, column: 3)
!2801 = !DILocation(line: 0, scope: !2800)
!2802 = !DILocation(line: 2094, column: 18, scope: !2803)
!2803 = distinct !DILexicalBlock(scope: !2800, file: !3, line: 2094, column: 3)
!2804 = !DILocation(line: 2094, column: 3, scope: !2800)
!2805 = !DILocation(line: 2095, column: 7, scope: !2806)
!2806 = distinct !DILexicalBlock(scope: !2807, file: !3, line: 2095, column: 7)
!2807 = distinct !DILexicalBlock(scope: !2803, file: !3, line: 2094, column: 30)
!2808 = !DILocation(line: 2095, column: 14, scope: !2806)
!2809 = !DILocation(line: 2095, column: 7, scope: !2807)
!2810 = !DILocation(line: 2097, column: 5, scope: !2811)
!2811 = distinct !DILexicalBlock(scope: !2806, file: !3, line: 2095, column: 19)
!2812 = !DILocation(line: 2099, column: 3, scope: !2807)
!2813 = !DILocation(line: 2094, column: 27, scope: !2803)
!2814 = !DILocation(line: 2094, column: 3, scope: !2803)
!2815 = distinct !{!2815, !2804, !2816}
!2816 = !DILocation(line: 2099, column: 3, scope: !2800)
!2817 = !DILocation(line: 2101, column: 3, scope: !2787)
!2818 = !DILocation(line: 2101, column: 10, scope: !2787)
!2819 = !DILocation(line: 2102, column: 3, scope: !2787)
!2820 = !DILocation(line: 2102, column: 11, scope: !2787)
!2821 = !DILocation(line: 2103, column: 13, scope: !2787)
!2822 = distinct !{!2822, !2784, !2823}
!2823 = !DILocation(line: 2104, column: 2, scope: !2774)
!2824 = !DILocation(line: 2105, column: 1, scope: !2774)
!2825 = distinct !DISubprogram(name: "vecset", linkageName: "_ZL6vecsetiPdPiS0_id", scope: !3, file: !3, line: 2113, type: !2826, scopeLine: 2113, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1156)
!2826 = !DISubroutineType(types: !2827)
!2827 = !{null, !97, !99, !98, !98, !97, !100}
!2828 = !DILocalVariable(name: "n", arg: 1, scope: !2825, file: !3, line: 2113, type: !97)
!2829 = !DILocation(line: 0, scope: !2825)
!2830 = !DILocalVariable(name: "v", arg: 2, scope: !2825, file: !3, line: 2113, type: !99)
!2831 = !DILocalVariable(name: "iv", arg: 3, scope: !2825, file: !3, line: 2113, type: !98)
!2832 = !DILocalVariable(name: "nzv", arg: 4, scope: !2825, file: !3, line: 2113, type: !98)
!2833 = !DILocalVariable(name: "i", arg: 5, scope: !2825, file: !3, line: 2113, type: !97)
!2834 = !DILocalVariable(name: "val", arg: 6, scope: !2825, file: !3, line: 2113, type: !100)
!2835 = !DILocalVariable(name: "set", scope: !2825, file: !3, line: 2115, type: !1752)
!2836 = !DILocalVariable(name: "k", scope: !2825, file: !3, line: 2114, type: !97)
!2837 = !DILocation(line: 2118, column: 6, scope: !2838)
!2838 = distinct !DILexicalBlock(scope: !2825, file: !3, line: 2118, column: 2)
!2839 = !DILocation(line: 0, scope: !2838)
!2840 = !DILocation(line: 2118, column: 17, scope: !2841)
!2841 = distinct !DILexicalBlock(scope: !2838, file: !3, line: 2118, column: 2)
!2842 = !DILocation(line: 2118, column: 15, scope: !2841)
!2843 = !DILocation(line: 2118, column: 2, scope: !2838)
!2844 = !DILocation(line: 2119, column: 6, scope: !2845)
!2845 = distinct !DILexicalBlock(scope: !2846, file: !3, line: 2119, column: 6)
!2846 = distinct !DILexicalBlock(scope: !2841, file: !3, line: 2118, column: 27)
!2847 = !DILocation(line: 2119, column: 12, scope: !2845)
!2848 = !DILocation(line: 2119, column: 6, scope: !2846)
!2849 = !DILocation(line: 2120, column: 4, scope: !2850)
!2850 = distinct !DILexicalBlock(scope: !2845, file: !3, line: 2119, column: 17)
!2851 = !DILocation(line: 2120, column: 9, scope: !2850)
!2852 = !DILocation(line: 2122, column: 3, scope: !2850)
!2853 = !DILocation(line: 2123, column: 2, scope: !2846)
!2854 = !DILocation(line: 2118, column: 24, scope: !2841)
!2855 = !DILocation(line: 2118, column: 2, scope: !2841)
!2856 = distinct !{!2856, !2843, !2857}
!2857 = !DILocation(line: 2123, column: 2, scope: !2838)
!2858 = !DILocation(line: 2124, column: 9, scope: !2859)
!2859 = distinct !DILexicalBlock(scope: !2825, file: !3, line: 2124, column: 5)
!2860 = !DILocation(line: 2124, column: 5, scope: !2825)
!2861 = !DILocation(line: 2125, column: 5, scope: !2862)
!2862 = distinct !DILexicalBlock(scope: !2859, file: !3, line: 2124, column: 18)
!2863 = !DILocation(line: 2125, column: 3, scope: !2862)
!2864 = !DILocation(line: 2125, column: 12, scope: !2862)
!2865 = !DILocation(line: 2126, column: 6, scope: !2862)
!2866 = !DILocation(line: 2126, column: 3, scope: !2862)
!2867 = !DILocation(line: 2126, column: 12, scope: !2862)
!2868 = !DILocation(line: 2127, column: 14, scope: !2862)
!2869 = !DILocation(line: 2127, column: 19, scope: !2862)
!2870 = !DILocation(line: 2127, column: 12, scope: !2862)
!2871 = !DILocation(line: 2128, column: 2, scope: !2862)
!2872 = !DILocation(line: 2129, column: 1, scope: !2825)
!2873 = distinct !DISubprogram(name: "sparse", linkageName: "_ZL6sparsePdPiS0_iiiS0_PA12_iPA12_diiS0_dd", scope: !3, file: !3, line: 1883, type: !2874, scopeLine: 1896, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1156)
!2874 = !DISubroutineType(types: !2875)
!2875 = !{null, !99, !98, !98, !97, !97, !97, !98, !101, !106, !97, !97, !98, !100, !100}
!2876 = !DILocalVariable(name: "a", arg: 1, scope: !2873, file: !3, line: 1883, type: !99)
!2877 = !DILocation(line: 0, scope: !2873)
!2878 = !DILocalVariable(name: "colidx", arg: 2, scope: !2873, file: !3, line: 1884, type: !98)
!2879 = !DILocalVariable(name: "rowstr", arg: 3, scope: !2873, file: !3, line: 1885, type: !98)
!2880 = !DILocalVariable(name: "n", arg: 4, scope: !2873, file: !3, line: 1886, type: !97)
!2881 = !DILocalVariable(name: "nz", arg: 5, scope: !2873, file: !3, line: 1887, type: !97)
!2882 = !DILocalVariable(name: "nozer", arg: 6, scope: !2873, file: !3, line: 1888, type: !97)
!2883 = !DILocalVariable(name: "arow", arg: 7, scope: !2873, file: !3, line: 1889, type: !98)
!2884 = !DILocalVariable(name: "acol", arg: 8, scope: !2873, file: !3, line: 1890, type: !101)
!2885 = !DILocalVariable(name: "aelt", arg: 9, scope: !2873, file: !3, line: 1891, type: !106)
!2886 = !DILocalVariable(name: "firstrow", arg: 10, scope: !2873, file: !3, line: 1892, type: !97)
!2887 = !DILocalVariable(name: "lastrow", arg: 11, scope: !2873, file: !3, line: 1893, type: !97)
!2888 = !DILocalVariable(name: "nzloc", arg: 12, scope: !2873, file: !3, line: 1894, type: !98)
!2889 = !DILocalVariable(name: "rcond", arg: 13, scope: !2873, file: !3, line: 1895, type: !100)
!2890 = !DILocalVariable(name: "shift", arg: 14, scope: !2873, file: !3, line: 1896, type: !100)
!2891 = !DILocation(line: 1914, column: 18, scope: !2873)
!2892 = !DILocation(line: 1914, column: 29, scope: !2873)
!2893 = !DILocalVariable(name: "nrows", scope: !2873, file: !3, line: 1897, type: !97)
!2894 = !DILocalVariable(name: "j", scope: !2873, file: !3, line: 1905, type: !97)
!2895 = !DILocation(line: 1921, column: 6, scope: !2896)
!2896 = distinct !DILexicalBlock(scope: !2873, file: !3, line: 1921, column: 2)
!2897 = !DILocation(line: 0, scope: !2896)
!2898 = !DILocation(line: 1921, column: 22, scope: !2899)
!2899 = distinct !DILexicalBlock(scope: !2896, file: !3, line: 1921, column: 2)
!2900 = !DILocation(line: 1921, column: 15, scope: !2899)
!2901 = !DILocation(line: 1921, column: 2, scope: !2896)
!2902 = !DILocation(line: 1922, column: 3, scope: !2903)
!2903 = distinct !DILexicalBlock(scope: !2899, file: !3, line: 1921, column: 30)
!2904 = !DILocation(line: 1922, column: 13, scope: !2903)
!2905 = !DILocation(line: 1923, column: 2, scope: !2903)
!2906 = !DILocation(line: 1921, column: 27, scope: !2899)
!2907 = !DILocation(line: 1921, column: 2, scope: !2899)
!2908 = distinct !{!2908, !2901, !2909}
!2909 = !DILocation(line: 1923, column: 2, scope: !2896)
!2910 = !DILocalVariable(name: "i", scope: !2873, file: !3, line: 1905, type: !97)
!2911 = !DILocation(line: 1924, column: 6, scope: !2912)
!2912 = distinct !DILexicalBlock(scope: !2873, file: !3, line: 1924, column: 2)
!2913 = !DILocation(line: 0, scope: !2912)
!2914 = !DILocation(line: 1924, column: 15, scope: !2915)
!2915 = distinct !DILexicalBlock(scope: !2912, file: !3, line: 1924, column: 2)
!2916 = !DILocation(line: 1924, column: 2, scope: !2912)
!2917 = !DILocalVariable(name: "nza", scope: !2873, file: !3, line: 1905, type: !97)
!2918 = !DILocation(line: 1925, column: 7, scope: !2919)
!2919 = distinct !DILexicalBlock(scope: !2920, file: !3, line: 1925, column: 3)
!2920 = distinct !DILexicalBlock(scope: !2915, file: !3, line: 1924, column: 24)
!2921 = !DILocation(line: 0, scope: !2919)
!2922 = !DILocation(line: 1925, column: 22, scope: !2923)
!2923 = distinct !DILexicalBlock(scope: !2919, file: !3, line: 1925, column: 3)
!2924 = !DILocation(line: 1925, column: 20, scope: !2923)
!2925 = !DILocation(line: 1925, column: 3, scope: !2919)
!2926 = !DILocation(line: 1926, column: 8, scope: !2927)
!2927 = distinct !DILexicalBlock(scope: !2923, file: !3, line: 1925, column: 37)
!2928 = !DILocation(line: 1926, column: 21, scope: !2927)
!2929 = !DILocation(line: 1927, column: 16, scope: !2927)
!2930 = !DILocation(line: 1927, column: 28, scope: !2927)
!2931 = !DILocation(line: 1927, column: 26, scope: !2927)
!2932 = !DILocation(line: 1927, column: 4, scope: !2927)
!2933 = !DILocation(line: 1927, column: 14, scope: !2927)
!2934 = !DILocation(line: 1928, column: 3, scope: !2927)
!2935 = !DILocation(line: 1925, column: 34, scope: !2923)
!2936 = !DILocation(line: 1925, column: 3, scope: !2923)
!2937 = distinct !{!2937, !2925, !2938}
!2938 = !DILocation(line: 1928, column: 3, scope: !2919)
!2939 = !DILocation(line: 1929, column: 2, scope: !2920)
!2940 = !DILocation(line: 1924, column: 21, scope: !2915)
!2941 = !DILocation(line: 1924, column: 2, scope: !2915)
!2942 = distinct !{!2942, !2916, !2943}
!2943 = !DILocation(line: 1929, column: 2, scope: !2912)
!2944 = !DILocation(line: 1930, column: 2, scope: !2873)
!2945 = !DILocation(line: 1930, column: 12, scope: !2873)
!2946 = !DILocation(line: 1931, column: 6, scope: !2947)
!2947 = distinct !DILexicalBlock(scope: !2873, file: !3, line: 1931, column: 2)
!2948 = !DILocation(line: 0, scope: !2947)
!2949 = !DILocation(line: 1931, column: 22, scope: !2950)
!2950 = distinct !DILexicalBlock(scope: !2947, file: !3, line: 1931, column: 2)
!2951 = !DILocation(line: 1931, column: 15, scope: !2950)
!2952 = !DILocation(line: 1931, column: 2, scope: !2947)
!2953 = !DILocation(line: 1932, column: 15, scope: !2954)
!2954 = distinct !DILexicalBlock(scope: !2950, file: !3, line: 1931, column: 30)
!2955 = !DILocation(line: 1932, column: 35, scope: !2954)
!2956 = !DILocation(line: 1932, column: 27, scope: !2954)
!2957 = !DILocation(line: 1932, column: 25, scope: !2954)
!2958 = !DILocation(line: 1932, column: 3, scope: !2954)
!2959 = !DILocation(line: 1932, column: 13, scope: !2954)
!2960 = !DILocation(line: 1933, column: 2, scope: !2954)
!2961 = !DILocation(line: 1931, column: 27, scope: !2950)
!2962 = !DILocation(line: 1931, column: 2, scope: !2950)
!2963 = distinct !{!2963, !2952, !2964}
!2964 = !DILocation(line: 1933, column: 2, scope: !2947)
!2965 = !DILocation(line: 1934, column: 8, scope: !2873)
!2966 = !DILocation(line: 1934, column: 22, scope: !2873)
!2967 = !DILocation(line: 1942, column: 9, scope: !2968)
!2968 = distinct !DILexicalBlock(scope: !2873, file: !3, line: 1942, column: 5)
!2969 = !DILocation(line: 1942, column: 5, scope: !2873)
!2970 = !DILocation(line: 1943, column: 3, scope: !2971)
!2971 = distinct !DILexicalBlock(scope: !2968, file: !3, line: 1942, column: 14)
!2972 = !DILocation(line: 1944, column: 3, scope: !2971)
!2973 = !DILocation(line: 1945, column: 3, scope: !2971)
!2974 = !DILocation(line: 1953, column: 6, scope: !2975)
!2975 = distinct !DILexicalBlock(scope: !2873, file: !3, line: 1953, column: 2)
!2976 = !DILocation(line: 0, scope: !2975)
!2977 = !DILocation(line: 1953, column: 15, scope: !2978)
!2978 = distinct !DILexicalBlock(scope: !2975, file: !3, line: 1953, column: 2)
!2979 = !DILocation(line: 1953, column: 2, scope: !2975)
!2980 = !DILocation(line: 1954, column: 11, scope: !2981)
!2981 = distinct !DILexicalBlock(scope: !2982, file: !3, line: 1954, column: 3)
!2982 = distinct !DILexicalBlock(scope: !2978, file: !3, line: 1953, column: 28)
!2983 = !DILocalVariable(name: "k", scope: !2873, file: !3, line: 1905, type: !97)
!2984 = !DILocation(line: 1954, column: 7, scope: !2981)
!2985 = !DILocation(line: 0, scope: !2981)
!2986 = !DILocation(line: 1954, column: 34, scope: !2987)
!2987 = distinct !DILexicalBlock(scope: !2981, file: !3, line: 1954, column: 3)
!2988 = !DILocation(line: 1954, column: 26, scope: !2987)
!2989 = !DILocation(line: 1954, column: 24, scope: !2987)
!2990 = !DILocation(line: 1954, column: 3, scope: !2981)
!2991 = !DILocation(line: 1955, column: 4, scope: !2992)
!2992 = distinct !DILexicalBlock(scope: !2987, file: !3, line: 1954, column: 43)
!2993 = !DILocation(line: 1955, column: 9, scope: !2992)
!2994 = !DILocation(line: 1956, column: 4, scope: !2992)
!2995 = !DILocation(line: 1956, column: 14, scope: !2992)
!2996 = !DILocation(line: 1957, column: 3, scope: !2992)
!2997 = !DILocation(line: 1954, column: 40, scope: !2987)
!2998 = !DILocation(line: 1954, column: 3, scope: !2987)
!2999 = distinct !{!2999, !2990, !3000}
!3000 = !DILocation(line: 1957, column: 3, scope: !2981)
!3001 = !DILocation(line: 1958, column: 3, scope: !2982)
!3002 = !DILocation(line: 1958, column: 12, scope: !2982)
!3003 = !DILocation(line: 1959, column: 2, scope: !2982)
!3004 = !DILocation(line: 1953, column: 25, scope: !2978)
!3005 = !DILocation(line: 1953, column: 2, scope: !2978)
!3006 = distinct !{!3006, !2979, !3007}
!3007 = !DILocation(line: 1959, column: 2, scope: !2975)
!3008 = !DILocalVariable(name: "size", scope: !2873, file: !3, line: 1906, type: !100)
!3009 = !DILocation(line: 1967, column: 36, scope: !2873)
!3010 = !DILocation(line: 1967, column: 26, scope: !2873)
!3011 = !DILocation(line: 1967, column: 10, scope: !2873)
!3012 = !DILocalVariable(name: "ratio", scope: !2873, file: !3, line: 1906, type: !100)
!3013 = !DILocation(line: 1968, column: 6, scope: !3014)
!3014 = distinct !DILexicalBlock(scope: !2873, file: !3, line: 1968, column: 2)
!3015 = !DILocation(line: 0, scope: !3014)
!3016 = !DILocation(line: 1968, column: 15, scope: !3017)
!3017 = distinct !DILexicalBlock(scope: !3014, file: !3, line: 1968, column: 2)
!3018 = !DILocation(line: 1968, column: 2, scope: !3014)
!3019 = !DILocation(line: 1969, column: 7, scope: !3020)
!3020 = distinct !DILexicalBlock(scope: !3021, file: !3, line: 1969, column: 3)
!3021 = distinct !DILexicalBlock(scope: !3017, file: !3, line: 1968, column: 24)
!3022 = !DILocation(line: 0, scope: !3020)
!3023 = !DILocation(line: 1969, column: 22, scope: !3024)
!3024 = distinct !DILexicalBlock(scope: !3020, file: !3, line: 1969, column: 3)
!3025 = !DILocation(line: 1969, column: 20, scope: !3024)
!3026 = !DILocation(line: 1969, column: 3, scope: !3020)
!3027 = !DILocation(line: 1970, column: 8, scope: !3028)
!3028 = distinct !DILexicalBlock(scope: !3024, file: !3, line: 1969, column: 37)
!3029 = !DILocation(line: 1972, column: 19, scope: !3028)
!3030 = !DILocation(line: 1972, column: 17, scope: !3028)
!3031 = !DILocalVariable(name: "scale", scope: !2873, file: !3, line: 1906, type: !100)
!3032 = !DILocalVariable(name: "nzrow", scope: !2873, file: !3, line: 1905, type: !97)
!3033 = !DILocation(line: 1973, column: 8, scope: !3034)
!3034 = distinct !DILexicalBlock(scope: !3028, file: !3, line: 1973, column: 4)
!3035 = !DILocation(line: 0, scope: !3034)
!3036 = !DILocation(line: 1973, column: 27, scope: !3037)
!3037 = distinct !DILexicalBlock(scope: !3034, file: !3, line: 1973, column: 4)
!3038 = !DILocation(line: 1973, column: 25, scope: !3037)
!3039 = !DILocation(line: 1973, column: 4, scope: !3034)
!3040 = !DILocation(line: 1974, column: 12, scope: !3041)
!3041 = distinct !DILexicalBlock(scope: !3037, file: !3, line: 1973, column: 44)
!3042 = !DILocalVariable(name: "jcol", scope: !2873, file: !3, line: 1905, type: !97)
!3043 = !DILocation(line: 1975, column: 10, scope: !3041)
!3044 = !DILocation(line: 1975, column: 25, scope: !3041)
!3045 = !DILocalVariable(name: "va", scope: !2873, file: !3, line: 1906, type: !100)
!3046 = !DILocation(line: 1983, column: 13, scope: !3047)
!3047 = distinct !DILexicalBlock(scope: !3041, file: !3, line: 1983, column: 8)
!3048 = !DILocation(line: 1983, column: 18, scope: !3047)
!3049 = !DILocation(line: 1983, column: 23, scope: !3047)
!3050 = !DILocation(line: 1983, column: 8, scope: !3041)
!3051 = !DILocation(line: 1984, column: 14, scope: !3052)
!3052 = distinct !DILexicalBlock(scope: !3047, file: !3, line: 1983, column: 28)
!3053 = !DILocation(line: 1984, column: 22, scope: !3052)
!3054 = !DILocation(line: 1985, column: 5, scope: !3052)
!3055 = !DILocation(line: 0, scope: !3041)
!3056 = !DILocalVariable(name: "goto_40", scope: !2873, file: !3, line: 1907, type: !1752)
!3057 = !DILocation(line: 1988, column: 13, scope: !3058)
!3058 = distinct !DILexicalBlock(scope: !3041, file: !3, line: 1988, column: 5)
!3059 = !DILocation(line: 1988, column: 9, scope: !3058)
!3060 = !DILocation(line: 0, scope: !3058)
!3061 = !DILocation(line: 1988, column: 36, scope: !3062)
!3062 = distinct !DILexicalBlock(scope: !3058, file: !3, line: 1988, column: 5)
!3063 = !DILocation(line: 1988, column: 28, scope: !3062)
!3064 = !DILocation(line: 1988, column: 26, scope: !3062)
!3065 = !DILocation(line: 1988, column: 5, scope: !3058)
!3066 = !DILocation(line: 1989, column: 9, scope: !3067)
!3067 = distinct !DILexicalBlock(scope: !3068, file: !3, line: 1989, column: 9)
!3068 = distinct !DILexicalBlock(scope: !3062, file: !3, line: 1988, column: 45)
!3069 = !DILocation(line: 1989, column: 19, scope: !3067)
!3070 = !DILocation(line: 1989, column: 9, scope: !3068)
!3071 = !DILocation(line: 1995, column: 24, scope: !3072)
!3072 = distinct !DILexicalBlock(scope: !3073, file: !3, line: 1995, column: 7)
!3073 = distinct !DILexicalBlock(scope: !3067, file: !3, line: 1989, column: 26)
!3074 = !DILocation(line: 1995, column: 16, scope: !3072)
!3075 = !DILocalVariable(name: "kk", scope: !2873, file: !3, line: 1905, type: !97)
!3076 = !DILocation(line: 1995, column: 11, scope: !3072)
!3077 = !DILocation(line: 0, scope: !3072)
!3078 = !DILocation(line: 1995, column: 34, scope: !3079)
!3079 = distinct !DILexicalBlock(scope: !3072, file: !3, line: 1995, column: 7)
!3080 = !DILocation(line: 1995, column: 7, scope: !3072)
!3081 = !DILocation(line: 1996, column: 11, scope: !3082)
!3082 = distinct !DILexicalBlock(scope: !3083, file: !3, line: 1996, column: 11)
!3083 = distinct !DILexicalBlock(scope: !3079, file: !3, line: 1995, column: 45)
!3084 = !DILocation(line: 1996, column: 22, scope: !3082)
!3085 = !DILocation(line: 1996, column: 11, scope: !3083)
!3086 = !DILocation(line: 1997, column: 19, scope: !3087)
!3087 = distinct !DILexicalBlock(scope: !3082, file: !3, line: 1996, column: 27)
!3088 = !DILocation(line: 1997, column: 13, scope: !3087)
!3089 = !DILocation(line: 1997, column: 9, scope: !3087)
!3090 = !DILocation(line: 1997, column: 17, scope: !3087)
!3091 = !DILocation(line: 1998, column: 24, scope: !3087)
!3092 = !DILocation(line: 1998, column: 18, scope: !3087)
!3093 = !DILocation(line: 1998, column: 9, scope: !3087)
!3094 = !DILocation(line: 1998, column: 22, scope: !3087)
!3095 = !DILocation(line: 1999, column: 8, scope: !3087)
!3096 = !DILocation(line: 2000, column: 7, scope: !3083)
!3097 = !DILocation(line: 1995, column: 42, scope: !3079)
!3098 = !DILocation(line: 1995, column: 7, scope: !3079)
!3099 = distinct !{!3099, !3080, !3100}
!3100 = !DILocation(line: 2000, column: 7, scope: !3072)
!3101 = !DILocation(line: 2001, column: 7, scope: !3073)
!3102 = !DILocation(line: 2001, column: 17, scope: !3073)
!3103 = !DILocation(line: 2002, column: 7, scope: !3073)
!3104 = !DILocation(line: 2002, column: 13, scope: !3073)
!3105 = !DILocation(line: 2004, column: 7, scope: !3073)
!3106 = !DILocation(line: 2005, column: 15, scope: !3107)
!3107 = distinct !DILexicalBlock(scope: !3067, file: !3, line: 2005, column: 15)
!3108 = !DILocation(line: 2005, column: 25, scope: !3107)
!3109 = !DILocation(line: 2005, column: 15, scope: !3067)
!3110 = !DILocation(line: 2006, column: 7, scope: !3111)
!3111 = distinct !DILexicalBlock(scope: !3107, file: !3, line: 2005, column: 31)
!3112 = !DILocation(line: 2006, column: 17, scope: !3111)
!3113 = !DILocation(line: 2008, column: 7, scope: !3111)
!3114 = !DILocation(line: 2009, column: 15, scope: !3115)
!3115 = distinct !DILexicalBlock(scope: !3107, file: !3, line: 2009, column: 15)
!3116 = !DILocation(line: 2009, column: 25, scope: !3115)
!3117 = !DILocation(line: 2009, column: 15, scope: !3107)
!3118 = !DILocation(line: 2015, column: 18, scope: !3119)
!3119 = distinct !DILexicalBlock(scope: !3115, file: !3, line: 2009, column: 33)
!3120 = !DILocation(line: 2015, column: 27, scope: !3119)
!3121 = !DILocation(line: 2015, column: 7, scope: !3119)
!3122 = !DILocation(line: 2015, column: 16, scope: !3119)
!3123 = !DILocation(line: 2017, column: 7, scope: !3119)
!3124 = !DILocation(line: 2019, column: 5, scope: !3068)
!3125 = !DILocation(line: 1988, column: 42, scope: !3062)
!3126 = !DILocation(line: 1988, column: 5, scope: !3062)
!3127 = distinct !{!3127, !3065, !3128}
!3128 = !DILocation(line: 2019, column: 5, scope: !3058)
!3129 = !DILocation(line: 2020, column: 16, scope: !3130)
!3130 = distinct !DILexicalBlock(scope: !3041, file: !3, line: 2020, column: 8)
!3131 = !DILocation(line: 2020, column: 8, scope: !3041)
!3132 = !DILocation(line: 2021, column: 6, scope: !3133)
!3133 = distinct !DILexicalBlock(scope: !3130, file: !3, line: 2020, column: 25)
!3134 = !DILocation(line: 2022, column: 6, scope: !3133)
!3135 = !DILocation(line: 2024, column: 12, scope: !3041)
!3136 = !DILocation(line: 2024, column: 17, scope: !3041)
!3137 = !DILocation(line: 2024, column: 5, scope: !3041)
!3138 = !DILocation(line: 2024, column: 10, scope: !3041)
!3139 = !DILocation(line: 2025, column: 4, scope: !3041)
!3140 = !DILocation(line: 1973, column: 41, scope: !3037)
!3141 = !DILocation(line: 1973, column: 4, scope: !3037)
!3142 = distinct !{!3142, !3039, !3143}
!3143 = !DILocation(line: 2025, column: 4, scope: !3034)
!3144 = !DILocation(line: 2026, column: 3, scope: !3028)
!3145 = !DILocation(line: 1969, column: 34, scope: !3024)
!3146 = !DILocation(line: 1969, column: 3, scope: !3024)
!3147 = distinct !{!3147, !3026, !3148}
!3148 = !DILocation(line: 2026, column: 3, scope: !3020)
!3149 = !DILocation(line: 2027, column: 15, scope: !3021)
!3150 = !DILocation(line: 2028, column: 2, scope: !3021)
!3151 = !DILocation(line: 1968, column: 21, scope: !3017)
!3152 = !DILocation(line: 1968, column: 2, scope: !3017)
!3153 = distinct !{!3153, !3018, !3154}
!3154 = !DILocation(line: 2028, column: 2, scope: !3014)
!3155 = !DILocation(line: 2035, column: 6, scope: !3156)
!3156 = distinct !DILexicalBlock(scope: !2873, file: !3, line: 2035, column: 2)
!3157 = !DILocation(line: 0, scope: !3156)
!3158 = !DILocation(line: 2035, column: 15, scope: !3159)
!3159 = distinct !DILexicalBlock(scope: !3156, file: !3, line: 2035, column: 2)
!3160 = !DILocation(line: 2035, column: 2, scope: !3156)
!3161 = !DILocation(line: 2036, column: 14, scope: !3162)
!3162 = distinct !DILexicalBlock(scope: !3159, file: !3, line: 2035, column: 28)
!3163 = !DILocation(line: 2036, column: 32, scope: !3162)
!3164 = !DILocation(line: 2036, column: 25, scope: !3162)
!3165 = !DILocation(line: 2036, column: 23, scope: !3162)
!3166 = !DILocation(line: 2036, column: 3, scope: !3162)
!3167 = !DILocation(line: 2036, column: 12, scope: !3162)
!3168 = !DILocation(line: 2037, column: 2, scope: !3162)
!3169 = !DILocation(line: 2035, column: 25, scope: !3159)
!3170 = !DILocation(line: 2035, column: 2, scope: !3159)
!3171 = distinct !{!3171, !3160, !3172}
!3172 = !DILocation(line: 2037, column: 2, scope: !3156)
!3173 = !DILocation(line: 2039, column: 6, scope: !3174)
!3174 = distinct !DILexicalBlock(scope: !2873, file: !3, line: 2039, column: 2)
!3175 = !DILocation(line: 0, scope: !3174)
!3176 = !DILocation(line: 2039, column: 15, scope: !3177)
!3177 = distinct !DILexicalBlock(scope: !3174, file: !3, line: 2039, column: 2)
!3178 = !DILocation(line: 2039, column: 2, scope: !3174)
!3179 = !DILocation(line: 2040, column: 8, scope: !3180)
!3180 = distinct !DILexicalBlock(scope: !3181, file: !3, line: 2040, column: 6)
!3181 = distinct !DILexicalBlock(scope: !3177, file: !3, line: 2039, column: 28)
!3182 = !DILocation(line: 2040, column: 6, scope: !3181)
!3183 = !DILocation(line: 2041, column: 9, scope: !3184)
!3184 = distinct !DILexicalBlock(scope: !3180, file: !3, line: 2040, column: 12)
!3185 = !DILocation(line: 2041, column: 28, scope: !3184)
!3186 = !DILocation(line: 2041, column: 21, scope: !3184)
!3187 = !DILocation(line: 2041, column: 19, scope: !3184)
!3188 = !DILocalVariable(name: "j1", scope: !2873, file: !3, line: 1905, type: !97)
!3189 = !DILocation(line: 2042, column: 3, scope: !3184)
!3190 = !DILocation(line: 0, scope: !3180)
!3191 = !DILocation(line: 2039, column: 25, scope: !3177)
!3192 = !DILocation(line: 2045, column: 8, scope: !3181)
!3193 = !DILocation(line: 2045, column: 22, scope: !3181)
!3194 = !DILocation(line: 2045, column: 20, scope: !3181)
!3195 = !DILocalVariable(name: "j2", scope: !2873, file: !3, line: 1905, type: !97)
!3196 = !DILocation(line: 2046, column: 9, scope: !3181)
!3197 = !DILocation(line: 2047, column: 7, scope: !3198)
!3198 = distinct !DILexicalBlock(scope: !3181, file: !3, line: 2047, column: 3)
!3199 = !DILocation(line: 0, scope: !3181)
!3200 = !DILocation(line: 2047, column: 17, scope: !3201)
!3201 = distinct !DILexicalBlock(scope: !3198, file: !3, line: 2047, column: 3)
!3202 = !DILocation(line: 2047, column: 3, scope: !3198)
!3203 = !DILocation(line: 2048, column: 11, scope: !3204)
!3204 = distinct !DILexicalBlock(scope: !3201, file: !3, line: 2047, column: 27)
!3205 = !DILocation(line: 2048, column: 4, scope: !3204)
!3206 = !DILocation(line: 2048, column: 9, scope: !3204)
!3207 = !DILocation(line: 2049, column: 16, scope: !3204)
!3208 = !DILocation(line: 2049, column: 4, scope: !3204)
!3209 = !DILocation(line: 2049, column: 14, scope: !3204)
!3210 = !DILocation(line: 2050, column: 14, scope: !3204)
!3211 = !DILocation(line: 2051, column: 3, scope: !3204)
!3212 = !DILocation(line: 2047, column: 24, scope: !3201)
!3213 = !DILocation(line: 2047, column: 3, scope: !3201)
!3214 = distinct !{!3214, !3202, !3215}
!3215 = !DILocation(line: 2051, column: 3, scope: !3198)
!3216 = !DILocation(line: 2052, column: 2, scope: !3181)
!3217 = !DILocation(line: 2039, column: 2, scope: !3177)
!3218 = distinct !{!3218, !3178, !3219}
!3219 = !DILocation(line: 2052, column: 2, scope: !3174)
!3220 = !DILocation(line: 2053, column: 6, scope: !3221)
!3221 = distinct !DILexicalBlock(scope: !2873, file: !3, line: 2053, column: 2)
!3222 = !DILocation(line: 0, scope: !3221)
!3223 = !DILocation(line: 2053, column: 22, scope: !3224)
!3224 = distinct !DILexicalBlock(scope: !3221, file: !3, line: 2053, column: 2)
!3225 = !DILocation(line: 2053, column: 15, scope: !3224)
!3226 = !DILocation(line: 2053, column: 2, scope: !3221)
!3227 = !DILocation(line: 2054, column: 15, scope: !3228)
!3228 = distinct !DILexicalBlock(scope: !3224, file: !3, line: 2053, column: 30)
!3229 = !DILocation(line: 2054, column: 34, scope: !3228)
!3230 = !DILocation(line: 2054, column: 27, scope: !3228)
!3231 = !DILocation(line: 2054, column: 25, scope: !3228)
!3232 = !DILocation(line: 2054, column: 3, scope: !3228)
!3233 = !DILocation(line: 2054, column: 13, scope: !3228)
!3234 = !DILocation(line: 2055, column: 2, scope: !3228)
!3235 = !DILocation(line: 2053, column: 27, scope: !3224)
!3236 = !DILocation(line: 2053, column: 2, scope: !3224)
!3237 = distinct !{!3237, !3226, !3238}
!3238 = !DILocation(line: 2055, column: 2, scope: !3221)
!3239 = !DILocation(line: 2057, column: 1, scope: !2873)
!3240 = distinct !DISubprogram(name: "icnvrt", linkageName: "_ZL6icnvrtdi", scope: !3, file: !3, line: 1544, type: !3241, scopeLine: 1544, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1156)
!3241 = !DISubroutineType(types: !3242)
!3242 = !{!97, !100, !97}
!3243 = !DILocalVariable(name: "x", arg: 1, scope: !3240, file: !3, line: 1544, type: !100)
!3244 = !DILocation(line: 0, scope: !3240)
!3245 = !DILocalVariable(name: "ipwr2", arg: 2, scope: !3240, file: !3, line: 1544, type: !97)
!3246 = !DILocation(line: 1545, column: 15, scope: !3240)
!3247 = !DILocation(line: 1545, column: 21, scope: !3240)
!3248 = !DILocation(line: 1545, column: 14, scope: !3240)
!3249 = !DILocation(line: 1545, column: 2, scope: !3240)
!3250 = !DILocalVariable(name: "norm_temp", arg: 1, scope: !3251, file: !3, line: 1453, type: !99)
!3251 = distinct !DISubprogram(name: "gpu_kernel_ten_1", linkageName: "_Z16gpu_kernel_ten_1PdS_S_", scope: !3, file: !3, line: 1453, type: !3252, scopeLine: 1455, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !1155, retainedNodes: !1156)
!3252 = !DISubroutineType(types: !3253)
!3253 = !{null, !99, !99, !99}
!3254 = !DILocation(line: 0, scope: !3251)
!3255 = !DILocalVariable(name: "x", arg: 2, scope: !3251, file: !3, line: 1454, type: !99)
!3256 = !DILocalVariable(name: "z", arg: 3, scope: !3251, file: !3, line: 1455, type: !99)
!3257 = !DILocalVariable(name: "share_data", scope: !3251, file: !3, line: 1456, type: !99)
!3258 = !DILocation(line: 1458, column: 29, scope: !3251)
!3259 = !DILocation(line: 1458, column: 42, scope: !3251)
!3260 = !DILocalVariable(name: "thread_id", scope: !3251, file: !3, line: 1458, type: !97)
!3261 = !DILocalVariable(name: "local_id", scope: !3251, file: !3, line: 1459, type: !97)
!3262 = !DILocation(line: 1461, column: 2, scope: !3251)
!3263 = !DILocation(line: 1461, column: 26, scope: !3251)
!3264 = !DILocation(line: 1465, column: 15, scope: !3265)
!3265 = distinct !DILexicalBlock(scope: !3251, file: !3, line: 1465, column: 5)
!3266 = !DILocation(line: 1465, column: 5, scope: !3251)
!3267 = !DILocation(line: 1466, column: 35, scope: !3268)
!3268 = distinct !DILexicalBlock(scope: !3265, file: !3, line: 1465, column: 20)
!3269 = !DILocation(line: 1466, column: 48, scope: !3268)
!3270 = !DILocation(line: 1466, column: 47, scope: !3268)
!3271 = !DILocation(line: 1466, column: 9, scope: !3268)
!3272 = !DILocation(line: 1466, column: 33, scope: !3268)
!3273 = !DILocation(line: 1467, column: 5, scope: !3268)
!3274 = !DILocation(line: 1477, column: 2, scope: !3251)
!3275 = !DILocalVariable(name: "norm_temp", arg: 1, scope: !3276, file: !3, line: 1486, type: !99)
!3276 = distinct !DISubprogram(name: "gpu_kernel_ten_2", linkageName: "_Z16gpu_kernel_ten_2PdS_S_", scope: !3, file: !3, line: 1486, type: !3252, scopeLine: 1488, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !1155, retainedNodes: !1156)
!3277 = !DILocation(line: 0, scope: !3276)
!3278 = !DILocalVariable(name: "x", arg: 2, scope: !3276, file: !3, line: 1487, type: !99)
!3279 = !DILocalVariable(name: "z", arg: 3, scope: !3276, file: !3, line: 1488, type: !99)
!3280 = !DILocalVariable(name: "share_data", scope: !3276, file: !3, line: 1489, type: !99)
!3281 = !DILocation(line: 1491, column: 29, scope: !3276)
!3282 = !DILocation(line: 1491, column: 42, scope: !3276)
!3283 = !DILocalVariable(name: "thread_id", scope: !3276, file: !3, line: 1491, type: !97)
!3284 = !DILocalVariable(name: "local_id", scope: !3276, file: !3, line: 1492, type: !97)
!3285 = !DILocation(line: 1494, column: 2, scope: !3276)
!3286 = !DILocation(line: 1494, column: 26, scope: !3276)
!3287 = !DILocation(line: 1498, column: 15, scope: !3288)
!3288 = distinct !DILexicalBlock(scope: !3276, file: !3, line: 1498, column: 5)
!3289 = !DILocation(line: 1498, column: 5, scope: !3276)
!3290 = !DILocation(line: 1499, column: 35, scope: !3291)
!3291 = distinct !DILexicalBlock(scope: !3288, file: !3, line: 1498, column: 20)
!3292 = !DILocation(line: 1499, column: 48, scope: !3291)
!3293 = !DILocation(line: 1499, column: 47, scope: !3291)
!3294 = !DILocation(line: 1499, column: 9, scope: !3291)
!3295 = !DILocation(line: 1499, column: 33, scope: !3291)
!3296 = !DILocation(line: 1500, column: 5, scope: !3291)
!3297 = !DILocation(line: 1510, column: 2, scope: !3276)
!3298 = !DILocalVariable(name: "norm_temp2", arg: 1, scope: !3299, file: !3, line: 1533, type: !100)
!3299 = distinct !DISubprogram(name: "gpu_kernel_eleven_device", linkageName: "_Z24gpu_kernel_eleven_devicedPdS_", scope: !3, file: !3, line: 1533, type: !3300, scopeLine: 1533, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !1155, retainedNodes: !1156)
!3300 = !DISubroutineType(types: !3301)
!3301 = !{null, !100, !99, !99}
!3302 = !DILocation(line: 0, scope: !3299)
!3303 = !DILocalVariable(name: "x", arg: 2, scope: !3299, file: !3, line: 1533, type: !99)
!3304 = !DILocalVariable(name: "z", arg: 3, scope: !3299, file: !3, line: 1533, type: !99)
!3305 = !DILocation(line: 1534, column: 21, scope: !3299)
!3306 = !DILocation(line: 1534, column: 34, scope: !3299)
!3307 = !DILocalVariable(name: "j", scope: !3299, file: !3, line: 1534, type: !97)
!3308 = !DILocation(line: 1535, column: 7, scope: !3309)
!3309 = distinct !DILexicalBlock(scope: !3299, file: !3, line: 1535, column: 5)
!3310 = !DILocation(line: 1535, column: 5, scope: !3299)
!3311 = !DILocation(line: 1535, column: 14, scope: !3312)
!3312 = distinct !DILexicalBlock(scope: !3309, file: !3, line: 1535, column: 13)
!3313 = !DILocation(line: 1536, column: 18, scope: !3299)
!3314 = !DILocation(line: 1536, column: 17, scope: !3299)
!3315 = !DILocation(line: 1536, column: 2, scope: !3299)
!3316 = !DILocation(line: 1536, column: 6, scope: !3299)
!3317 = !DILocation(line: 1537, column: 1, scope: !3299)
!3318 = !DILocalVariable(name: "p", arg: 1, scope: !3319, file: !3, line: 1050, type: !99)
!3319 = distinct !DISubprogram(name: "gpu_kernel_one_device", linkageName: "_Z21gpu_kernel_one_devicePdS_S_S_S_", scope: !3, file: !3, line: 1050, type: !3320, scopeLine: 1054, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !1155, retainedNodes: !1156)
!3320 = !DISubroutineType(types: !3321)
!3321 = !{null, !99, !99, !99, !99, !99}
!3322 = !DILocation(line: 0, scope: !3319)
!3323 = !DILocalVariable(name: "q", arg: 2, scope: !3319, file: !3, line: 1051, type: !99)
!3324 = !DILocalVariable(name: "r", arg: 3, scope: !3319, file: !3, line: 1052, type: !99)
!3325 = !DILocalVariable(name: "x", arg: 4, scope: !3319, file: !3, line: 1053, type: !99)
!3326 = !DILocalVariable(name: "z", arg: 5, scope: !3319, file: !3, line: 1054, type: !99)
!3327 = !DILocation(line: 1055, column: 29, scope: !3319)
!3328 = !DILocation(line: 1055, column: 42, scope: !3319)
!3329 = !DILocalVariable(name: "thread_id", scope: !3319, file: !3, line: 1055, type: !97)
!3330 = !DILocation(line: 1056, column: 15, scope: !3331)
!3331 = distinct !DILexicalBlock(scope: !3319, file: !3, line: 1056, column: 5)
!3332 = !DILocation(line: 1056, column: 5, scope: !3319)
!3333 = !DILocation(line: 1056, column: 22, scope: !3334)
!3334 = distinct !DILexicalBlock(scope: !3331, file: !3, line: 1056, column: 21)
!3335 = !DILocation(line: 1057, column: 2, scope: !3319)
!3336 = !DILocation(line: 1057, column: 15, scope: !3319)
!3337 = !DILocation(line: 1058, column: 2, scope: !3319)
!3338 = !DILocation(line: 1058, column: 15, scope: !3319)
!3339 = !DILocation(line: 1059, column: 19, scope: !3319)
!3340 = !DILocalVariable(name: "x_value", scope: !3319, file: !3, line: 1059, type: !100)
!3341 = !DILocation(line: 1060, column: 2, scope: !3319)
!3342 = !DILocation(line: 1060, column: 15, scope: !3319)
!3343 = !DILocation(line: 1061, column: 2, scope: !3319)
!3344 = !DILocation(line: 1061, column: 15, scope: !3319)
!3345 = !DILocation(line: 1062, column: 1, scope: !3319)
!3346 = !DILocalVariable(name: "r", arg: 1, scope: !3347, file: !3, line: 1083, type: !99)
!3347 = distinct !DISubprogram(name: "gpu_kernel_two_device", linkageName: "_Z21gpu_kernel_two_devicePdS_S_", scope: !3, file: !3, line: 1083, type: !3252, scopeLine: 1085, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !1155, retainedNodes: !1156)
!3348 = !DILocation(line: 0, scope: !3347)
!3349 = !DILocalVariable(name: "rho", arg: 2, scope: !3347, file: !3, line: 1084, type: !99)
!3350 = !DILocalVariable(name: "global_data", arg: 3, scope: !3347, file: !3, line: 1085, type: !99)
!3351 = !DILocalVariable(name: "share_data", scope: !3347, file: !3, line: 1086, type: !99)
!3352 = !DILocation(line: 1088, column: 29, scope: !3347)
!3353 = !DILocation(line: 1088, column: 42, scope: !3347)
!3354 = !DILocalVariable(name: "thread_id", scope: !3347, file: !3, line: 1088, type: !97)
!3355 = !DILocalVariable(name: "local_id", scope: !3347, file: !3, line: 1089, type: !97)
!3356 = !DILocation(line: 1091, column: 2, scope: !3347)
!3357 = !DILocation(line: 1091, column: 23, scope: !3347)
!3358 = !DILocation(line: 1095, column: 15, scope: !3359)
!3359 = distinct !DILexicalBlock(scope: !3347, file: !3, line: 1095, column: 5)
!3360 = !DILocation(line: 1095, column: 5, scope: !3347)
!3361 = !DILocation(line: 1096, column: 26, scope: !3362)
!3362 = distinct !DILexicalBlock(scope: !3359, file: !3, line: 1095, column: 20)
!3363 = !DILocalVariable(name: "r_value", scope: !3362, file: !3, line: 1096, type: !100)
!3364 = !DILocation(line: 0, scope: !3362)
!3365 = !DILocation(line: 1097, column: 40, scope: !3362)
!3366 = !DILocation(line: 1097, column: 9, scope: !3362)
!3367 = !DILocation(line: 1097, column: 30, scope: !3362)
!3368 = !DILocation(line: 1098, column: 5, scope: !3362)
!3369 = !DILocation(line: 1108, column: 2, scope: !3347)
!3370 = !DILocalVariable(name: "colidx", arg: 1, scope: !3371, file: !3, line: 1134, type: !98)
!3371 = distinct !DISubprogram(name: "gpu_kernel_three_device", linkageName: "_Z23gpu_kernel_three_devicePiS_PdS0_S0_", scope: !3, file: !3, line: 1134, type: !3372, scopeLine: 1138, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !1155, retainedNodes: !1156)
!3372 = !DISubroutineType(types: !3373)
!3373 = !{null, !98, !98, !99, !99, !99}
!3374 = !DILocation(line: 0, scope: !3371)
!3375 = !DILocalVariable(name: "rowstr", arg: 2, scope: !3371, file: !3, line: 1135, type: !98)
!3376 = !DILocalVariable(name: "a", arg: 3, scope: !3371, file: !3, line: 1136, type: !99)
!3377 = !DILocalVariable(name: "p", arg: 4, scope: !3371, file: !3, line: 1137, type: !99)
!3378 = !DILocalVariable(name: "q", arg: 5, scope: !3371, file: !3, line: 1138, type: !99)
!3379 = !DILocalVariable(name: "share_data", scope: !3371, file: !3, line: 1139, type: !99)
!3380 = !DILocation(line: 1141, column: 28, scope: !3371)
!3381 = !DILocation(line: 1141, column: 39, scope: !3371)
!3382 = !DILocation(line: 1141, column: 53, scope: !3371)
!3383 = !DILocalVariable(name: "j", scope: !3371, file: !3, line: 1141, type: !97)
!3384 = !DILocalVariable(name: "local_id", scope: !3371, file: !3, line: 1142, type: !97)
!3385 = !DILocation(line: 1144, column: 14, scope: !3371)
!3386 = !DILocalVariable(name: "begin", scope: !3371, file: !3, line: 1144, type: !97)
!3387 = !DILocation(line: 1145, column: 20, scope: !3371)
!3388 = !DILocation(line: 1145, column: 12, scope: !3371)
!3389 = !DILocalVariable(name: "end", scope: !3371, file: !3, line: 1145, type: !97)
!3390 = !DILocalVariable(name: "sum", scope: !3371, file: !3, line: 1146, type: !100)
!3391 = !DILocation(line: 1147, column: 17, scope: !3392)
!3392 = distinct !DILexicalBlock(scope: !3371, file: !3, line: 1147, column: 2)
!3393 = !DILocalVariable(name: "k", scope: !3392, file: !3, line: 1147, type: !97)
!3394 = !DILocation(line: 0, scope: !3392)
!3395 = !DILocation(line: 1147, column: 6, scope: !3392)
!3396 = !DILocation(line: 1147, column: 29, scope: !3397)
!3397 = distinct !DILexicalBlock(scope: !3392, file: !3, line: 1147, column: 2)
!3398 = !DILocation(line: 1147, column: 2, scope: !3392)
!3399 = !DILocation(line: 1148, column: 15, scope: !3400)
!3400 = distinct !DILexicalBlock(scope: !3397, file: !3, line: 1147, column: 49)
!3401 = !DILocation(line: 1148, column: 22, scope: !3400)
!3402 = !DILocation(line: 1148, column: 20, scope: !3400)
!3403 = !DILocation(line: 1148, column: 19, scope: !3400)
!3404 = !DILocation(line: 1148, column: 13, scope: !3400)
!3405 = !DILocation(line: 1149, column: 2, scope: !3400)
!3406 = !DILocation(line: 1147, column: 36, scope: !3397)
!3407 = !DILocation(line: 1147, column: 2, scope: !3397)
!3408 = distinct !{!3408, !3398, !3409}
!3409 = !DILocation(line: 1149, column: 2, scope: !3392)
!3410 = !DILocation(line: 1150, column: 2, scope: !3371)
!3411 = !DILocation(line: 1150, column: 23, scope: !3371)
!3412 = !DILocation(line: 1160, column: 2, scope: !3371)
!3413 = !DILocalVariable(name: "d", arg: 1, scope: !3414, file: !3, line: 1189, type: !99)
!3414 = distinct !DISubprogram(name: "gpu_kernel_four_device", linkageName: "_Z22gpu_kernel_four_devicePdS_S_S_", scope: !3, file: !3, line: 1189, type: !3415, scopeLine: 1192, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !1155, retainedNodes: !1156)
!3415 = !DISubroutineType(types: !3416)
!3416 = !{null, !99, !99, !99, !99}
!3417 = !DILocation(line: 0, scope: !3414)
!3418 = !DILocalVariable(name: "p", arg: 2, scope: !3414, file: !3, line: 1190, type: !99)
!3419 = !DILocalVariable(name: "q", arg: 3, scope: !3414, file: !3, line: 1191, type: !99)
!3420 = !DILocalVariable(name: "global_data", arg: 4, scope: !3414, file: !3, line: 1192, type: !99)
!3421 = !DILocalVariable(name: "share_data", scope: !3414, file: !3, line: 1193, type: !99)
!3422 = !DILocation(line: 1195, column: 29, scope: !3414)
!3423 = !DILocation(line: 1195, column: 42, scope: !3414)
!3424 = !DILocalVariable(name: "thread_id", scope: !3414, file: !3, line: 1195, type: !97)
!3425 = !DILocalVariable(name: "local_id", scope: !3414, file: !3, line: 1196, type: !97)
!3426 = !DILocation(line: 1198, column: 2, scope: !3414)
!3427 = !DILocation(line: 1198, column: 23, scope: !3414)
!3428 = !DILocation(line: 1202, column: 2, scope: !3414)
!3429 = !DILocation(line: 1202, column: 23, scope: !3414)
!3430 = !DILocation(line: 1204, column: 15, scope: !3431)
!3431 = distinct !DILexicalBlock(scope: !3414, file: !3, line: 1204, column: 5)
!3432 = !DILocation(line: 1204, column: 5, scope: !3414)
!3433 = !DILocation(line: 1205, column: 35, scope: !3434)
!3434 = distinct !DILexicalBlock(scope: !3431, file: !3, line: 1204, column: 20)
!3435 = !DILocation(line: 1205, column: 50, scope: !3434)
!3436 = !DILocation(line: 1205, column: 48, scope: !3434)
!3437 = !DILocation(line: 1205, column: 9, scope: !3434)
!3438 = !DILocation(line: 1205, column: 33, scope: !3434)
!3439 = !DILocation(line: 1206, column: 5, scope: !3434)
!3440 = !DILocation(line: 1216, column: 2, scope: !3414)
!3441 = !DILocalVariable(name: "alpha", arg: 1, scope: !3442, file: !3, line: 1244, type: !100)
!3442 = distinct !DISubprogram(name: "gpu_kernel_five_1", linkageName: "_Z17gpu_kernel_five_1dPdS_", scope: !3, file: !3, line: 1244, type: !3300, scopeLine: 1246, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !1155, retainedNodes: !1156)
!3443 = !DILocation(line: 0, scope: !3442)
!3444 = !DILocalVariable(name: "p", arg: 2, scope: !3442, file: !3, line: 1245, type: !99)
!3445 = !DILocalVariable(name: "z", arg: 3, scope: !3442, file: !3, line: 1246, type: !99)
!3446 = !DILocation(line: 1247, column: 21, scope: !3442)
!3447 = !DILocation(line: 1247, column: 34, scope: !3442)
!3448 = !DILocalVariable(name: "j", scope: !3442, file: !3, line: 1247, type: !97)
!3449 = !DILocation(line: 1248, column: 7, scope: !3450)
!3450 = distinct !DILexicalBlock(scope: !3442, file: !3, line: 1248, column: 5)
!3451 = !DILocation(line: 1248, column: 5, scope: !3442)
!3452 = !DILocation(line: 1248, column: 14, scope: !3453)
!3453 = distinct !DILexicalBlock(scope: !3450, file: !3, line: 1248, column: 13)
!3454 = !DILocation(line: 1249, column: 18, scope: !3442)
!3455 = !DILocation(line: 1249, column: 16, scope: !3442)
!3456 = !DILocation(line: 1249, column: 2, scope: !3442)
!3457 = !DILocation(line: 1249, column: 7, scope: !3442)
!3458 = !DILocation(line: 1250, column: 1, scope: !3442)
!3459 = !DILocalVariable(name: "alpha", arg: 1, scope: !3460, file: !3, line: 1252, type: !100)
!3460 = distinct !DISubprogram(name: "gpu_kernel_five_2", linkageName: "_Z17gpu_kernel_five_2dPdS_", scope: !3, file: !3, line: 1252, type: !3300, scopeLine: 1254, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !1155, retainedNodes: !1156)
!3461 = !DILocation(line: 0, scope: !3460)
!3462 = !DILocalVariable(name: "q", arg: 2, scope: !3460, file: !3, line: 1253, type: !99)
!3463 = !DILocalVariable(name: "r", arg: 3, scope: !3460, file: !3, line: 1254, type: !99)
!3464 = !DILocation(line: 1255, column: 21, scope: !3460)
!3465 = !DILocation(line: 1255, column: 34, scope: !3460)
!3466 = !DILocalVariable(name: "j", scope: !3460, file: !3, line: 1255, type: !97)
!3467 = !DILocation(line: 1256, column: 7, scope: !3468)
!3468 = distinct !DILexicalBlock(scope: !3460, file: !3, line: 1256, column: 5)
!3469 = !DILocation(line: 1256, column: 5, scope: !3460)
!3470 = !DILocation(line: 1256, column: 14, scope: !3471)
!3471 = distinct !DILexicalBlock(scope: !3468, file: !3, line: 1256, column: 13)
!3472 = !DILocation(line: 1257, column: 18, scope: !3460)
!3473 = !DILocation(line: 1257, column: 16, scope: !3460)
!3474 = !DILocation(line: 1257, column: 2, scope: !3460)
!3475 = !DILocation(line: 1257, column: 7, scope: !3460)
!3476 = !DILocation(line: 1258, column: 1, scope: !3460)
!3477 = !DILocalVariable(name: "r", arg: 1, scope: !3478, file: !3, line: 1278, type: !99)
!3478 = distinct !DISubprogram(name: "gpu_kernel_six_device", linkageName: "_Z21gpu_kernel_six_devicePdS_", scope: !3, file: !3, line: 1278, type: !2529, scopeLine: 1279, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !1155, retainedNodes: !1156)
!3479 = !DILocation(line: 0, scope: !3478)
!3480 = !DILocalVariable(name: "global_data", arg: 2, scope: !3478, file: !3, line: 1279, type: !99)
!3481 = !DILocalVariable(name: "share_data", scope: !3478, file: !3, line: 1280, type: !99)
!3482 = !DILocation(line: 1281, column: 29, scope: !3478)
!3483 = !DILocation(line: 1281, column: 42, scope: !3478)
!3484 = !DILocalVariable(name: "thread_id", scope: !3478, file: !3, line: 1281, type: !97)
!3485 = !DILocalVariable(name: "local_id", scope: !3478, file: !3, line: 1282, type: !97)
!3486 = !DILocation(line: 1283, column: 2, scope: !3478)
!3487 = !DILocation(line: 1283, column: 23, scope: !3478)
!3488 = !DILocation(line: 1285, column: 15, scope: !3489)
!3489 = distinct !DILexicalBlock(scope: !3478, file: !3, line: 1285, column: 5)
!3490 = !DILocation(line: 1285, column: 5, scope: !3478)
!3491 = !DILocation(line: 1286, column: 26, scope: !3492)
!3492 = distinct !DILexicalBlock(scope: !3489, file: !3, line: 1285, column: 20)
!3493 = !DILocalVariable(name: "r_value", scope: !3492, file: !3, line: 1286, type: !100)
!3494 = !DILocation(line: 0, scope: !3492)
!3495 = !DILocation(line: 1287, column: 40, scope: !3492)
!3496 = !DILocation(line: 1287, column: 9, scope: !3492)
!3497 = !DILocation(line: 1287, column: 30, scope: !3492)
!3498 = !DILocation(line: 1288, column: 5, scope: !3492)
!3499 = !DILocation(line: 1297, column: 2, scope: !3478)
!3500 = !DILocalVariable(name: "beta", arg: 1, scope: !3501, file: !3, line: 1320, type: !100)
!3501 = distinct !DISubprogram(name: "gpu_kernel_seven_device", linkageName: "_Z23gpu_kernel_seven_devicedPdS_", scope: !3, file: !3, line: 1320, type: !3300, scopeLine: 1322, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !1155, retainedNodes: !1156)
!3502 = !DILocation(line: 0, scope: !3501)
!3503 = !DILocalVariable(name: "p", arg: 2, scope: !3501, file: !3, line: 1321, type: !99)
!3504 = !DILocalVariable(name: "r", arg: 3, scope: !3501, file: !3, line: 1322, type: !99)
!3505 = !DILocation(line: 1323, column: 21, scope: !3501)
!3506 = !DILocation(line: 1323, column: 34, scope: !3501)
!3507 = !DILocalVariable(name: "j", scope: !3501, file: !3, line: 1323, type: !97)
!3508 = !DILocation(line: 1324, column: 7, scope: !3509)
!3509 = distinct !DILexicalBlock(scope: !3501, file: !3, line: 1324, column: 5)
!3510 = !DILocation(line: 1324, column: 5, scope: !3501)
!3511 = !DILocation(line: 1324, column: 14, scope: !3512)
!3512 = distinct !DILexicalBlock(scope: !3509, file: !3, line: 1324, column: 13)
!3513 = !DILocation(line: 1325, column: 9, scope: !3501)
!3514 = !DILocation(line: 1325, column: 21, scope: !3501)
!3515 = !DILocation(line: 1325, column: 20, scope: !3501)
!3516 = !DILocation(line: 1325, column: 14, scope: !3501)
!3517 = !DILocation(line: 1325, column: 2, scope: !3501)
!3518 = !DILocation(line: 1325, column: 7, scope: !3501)
!3519 = !DILocation(line: 1326, column: 1, scope: !3501)
!3520 = !DILocalVariable(name: "colidx", arg: 1, scope: !3521, file: !3, line: 1345, type: !98)
!3521 = distinct !DISubprogram(name: "gpu_kernel_eight_device", linkageName: "_Z23gpu_kernel_eight_devicePiS_PdS0_S0_", scope: !3, file: !3, line: 1345, type: !3372, scopeLine: 1349, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !1155, retainedNodes: !1156)
!3522 = !DILocation(line: 0, scope: !3521)
!3523 = !DILocalVariable(name: "rowstr", arg: 2, scope: !3521, file: !3, line: 1346, type: !98)
!3524 = !DILocalVariable(name: "a", arg: 3, scope: !3521, file: !3, line: 1347, type: !99)
!3525 = !DILocalVariable(name: "r", arg: 4, scope: !3521, file: !3, line: 1348, type: !99)
!3526 = !DILocalVariable(name: "z", arg: 5, scope: !3521, file: !3, line: 1349, type: !99)
!3527 = !DILocalVariable(name: "share_data", scope: !3521, file: !3, line: 1350, type: !99)
!3528 = !DILocation(line: 1352, column: 28, scope: !3521)
!3529 = !DILocation(line: 1352, column: 39, scope: !3521)
!3530 = !DILocation(line: 1352, column: 53, scope: !3521)
!3531 = !DILocalVariable(name: "j", scope: !3521, file: !3, line: 1352, type: !97)
!3532 = !DILocalVariable(name: "local_id", scope: !3521, file: !3, line: 1353, type: !97)
!3533 = !DILocation(line: 1355, column: 14, scope: !3521)
!3534 = !DILocalVariable(name: "begin", scope: !3521, file: !3, line: 1355, type: !97)
!3535 = !DILocation(line: 1356, column: 20, scope: !3521)
!3536 = !DILocation(line: 1356, column: 12, scope: !3521)
!3537 = !DILocalVariable(name: "end", scope: !3521, file: !3, line: 1356, type: !97)
!3538 = !DILocalVariable(name: "sum", scope: !3521, file: !3, line: 1357, type: !100)
!3539 = !DILocation(line: 1358, column: 17, scope: !3540)
!3540 = distinct !DILexicalBlock(scope: !3521, file: !3, line: 1358, column: 2)
!3541 = !DILocalVariable(name: "k", scope: !3540, file: !3, line: 1358, type: !97)
!3542 = !DILocation(line: 0, scope: !3540)
!3543 = !DILocation(line: 1358, column: 6, scope: !3540)
!3544 = !DILocation(line: 1358, column: 29, scope: !3545)
!3545 = distinct !DILexicalBlock(scope: !3540, file: !3, line: 1358, column: 2)
!3546 = !DILocation(line: 1358, column: 2, scope: !3540)
!3547 = !DILocation(line: 1359, column: 15, scope: !3548)
!3548 = distinct !DILexicalBlock(scope: !3545, file: !3, line: 1358, column: 49)
!3549 = !DILocation(line: 1359, column: 22, scope: !3548)
!3550 = !DILocation(line: 1359, column: 20, scope: !3548)
!3551 = !DILocation(line: 1359, column: 19, scope: !3548)
!3552 = !DILocation(line: 1359, column: 13, scope: !3548)
!3553 = !DILocation(line: 1360, column: 2, scope: !3548)
!3554 = !DILocation(line: 1358, column: 36, scope: !3545)
!3555 = !DILocation(line: 1358, column: 2, scope: !3545)
!3556 = distinct !{!3556, !3546, !3557}
!3557 = !DILocation(line: 1360, column: 2, scope: !3540)
!3558 = !DILocation(line: 1361, column: 2, scope: !3521)
!3559 = !DILocation(line: 1361, column: 23, scope: !3521)
!3560 = !DILocation(line: 1371, column: 2, scope: !3521)
!3561 = !DILocalVariable(name: "r", arg: 1, scope: !3562, file: !3, line: 1400, type: !99)
!3562 = distinct !DISubprogram(name: "gpu_kernel_nine_device", linkageName: "_Z22gpu_kernel_nine_devicePdS_S_S_", scope: !3, file: !3, line: 1400, type: !3415, scopeLine: 1400, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !1155, retainedNodes: !1156)
!3563 = !DILocation(line: 0, scope: !3562)
!3564 = !DILocalVariable(name: "x", arg: 2, scope: !3562, file: !3, line: 1400, type: !99)
!3565 = !DILocalVariable(name: "sum", arg: 3, scope: !3562, file: !3, line: 1400, type: !99)
!3566 = !DILocalVariable(name: "global_data", arg: 4, scope: !3562, file: !3, line: 1400, type: !99)
!3567 = !DILocalVariable(name: "share_data", scope: !3562, file: !3, line: 1401, type: !99)
!3568 = !DILocation(line: 1403, column: 29, scope: !3562)
!3569 = !DILocation(line: 1403, column: 42, scope: !3562)
!3570 = !DILocalVariable(name: "thread_id", scope: !3562, file: !3, line: 1403, type: !97)
!3571 = !DILocalVariable(name: "local_id", scope: !3562, file: !3, line: 1404, type: !97)
!3572 = !DILocation(line: 1406, column: 2, scope: !3562)
!3573 = !DILocation(line: 1406, column: 23, scope: !3562)
!3574 = !DILocation(line: 1410, column: 15, scope: !3575)
!3575 = distinct !DILexicalBlock(scope: !3562, file: !3, line: 1410, column: 5)
!3576 = !DILocation(line: 1410, column: 5, scope: !3562)
!3577 = !DILocation(line: 1411, column: 32, scope: !3578)
!3578 = distinct !DILexicalBlock(scope: !3575, file: !3, line: 1410, column: 20)
!3579 = !DILocation(line: 1411, column: 47, scope: !3578)
!3580 = !DILocation(line: 1411, column: 45, scope: !3578)
!3581 = !DILocation(line: 1411, column: 9, scope: !3578)
!3582 = !DILocation(line: 1411, column: 30, scope: !3578)
!3583 = !DILocation(line: 1412, column: 32, scope: !3578)
!3584 = !DILocation(line: 1412, column: 55, scope: !3578)
!3585 = !DILocation(line: 1412, column: 53, scope: !3578)
!3586 = !DILocation(line: 1412, column: 9, scope: !3578)
!3587 = !DILocation(line: 1412, column: 30, scope: !3578)
!3588 = !DILocation(line: 1413, column: 5, scope: !3578)
!3589 = !DILocation(line: 1423, column: 2, scope: !3562)
!3590 = !DILocation(line: 1511, column: 13, scope: !3591)
!3591 = distinct !DILexicalBlock(scope: !3276, file: !3, line: 1511, column: 5)
!3592 = !DILocation(line: 1511, column: 5, scope: !3276)
!3593 = !DILocalVariable(name: "i", scope: !3594, file: !3, line: 1512, type: !97)
!3594 = distinct !DILexicalBlock(scope: !3595, file: !3, line: 1512, column: 3)
!3595 = distinct !DILexicalBlock(scope: !3591, file: !3, line: 1511, column: 17)
!3596 = !DILocation(line: 0, scope: !3594)
!3597 = !DILocation(line: 1512, column: 7, scope: !3594)
!3598 = !DILocation(line: 1512, column: 17, scope: !3599)
!3599 = distinct !DILexicalBlock(scope: !3594, file: !3, line: 1512, column: 3)
!3600 = !DILocation(line: 1512, column: 3, scope: !3594)
!3601 = !DILocation(line: 1513, column: 19, scope: !3602)
!3602 = distinct !DILexicalBlock(scope: !3599, file: !3, line: 1512, column: 34)
!3603 = !DILocation(line: 1513, column: 4, scope: !3602)
!3604 = !DILocation(line: 1513, column: 17, scope: !3602)
!3605 = !DILocation(line: 1514, column: 3, scope: !3602)
!3606 = !DILocation(line: 1512, column: 31, scope: !3599)
!3607 = !DILocation(line: 1512, column: 3, scope: !3599)
!3608 = distinct !{!3608, !3600, !3609}
!3609 = !DILocation(line: 1514, column: 3, scope: !3594)
!3610 = !DILocation(line: 1515, column: 25, scope: !3595)
!3611 = !DILocation(line: 1515, column: 3, scope: !3595)
!3612 = !DILocation(line: 1515, column: 24, scope: !3595)
!3613 = !DILocation(line: 1516, column: 2, scope: !3595)
!3614 = !DILocation(line: 1517, column: 1, scope: !3276)
!3615 = !DILocation(line: 1478, column: 13, scope: !3616)
!3616 = distinct !DILexicalBlock(scope: !3251, file: !3, line: 1478, column: 5)
!3617 = !DILocation(line: 1478, column: 5, scope: !3251)
!3618 = !DILocalVariable(name: "i", scope: !3619, file: !3, line: 1479, type: !97)
!3619 = distinct !DILexicalBlock(scope: !3620, file: !3, line: 1479, column: 3)
!3620 = distinct !DILexicalBlock(scope: !3616, file: !3, line: 1478, column: 17)
!3621 = !DILocation(line: 0, scope: !3619)
!3622 = !DILocation(line: 1479, column: 7, scope: !3619)
!3623 = !DILocation(line: 1479, column: 17, scope: !3624)
!3624 = distinct !DILexicalBlock(scope: !3619, file: !3, line: 1479, column: 3)
!3625 = !DILocation(line: 1479, column: 3, scope: !3619)
!3626 = !DILocation(line: 1480, column: 19, scope: !3627)
!3627 = distinct !DILexicalBlock(scope: !3624, file: !3, line: 1479, column: 34)
!3628 = !DILocation(line: 1480, column: 4, scope: !3627)
!3629 = !DILocation(line: 1480, column: 17, scope: !3627)
!3630 = !DILocation(line: 1481, column: 3, scope: !3627)
!3631 = !DILocation(line: 1479, column: 31, scope: !3624)
!3632 = !DILocation(line: 1479, column: 3, scope: !3624)
!3633 = distinct !{!3633, !3625, !3634}
!3634 = !DILocation(line: 1481, column: 3, scope: !3619)
!3635 = !DILocation(line: 1482, column: 25, scope: !3620)
!3636 = !DILocation(line: 1482, column: 3, scope: !3620)
!3637 = !DILocation(line: 1482, column: 24, scope: !3620)
!3638 = !DILocation(line: 1483, column: 2, scope: !3620)
!3639 = !DILocation(line: 1484, column: 1, scope: !3251)
!3640 = !DILocation(line: 1424, column: 13, scope: !3641)
!3641 = distinct !DILexicalBlock(scope: !3562, file: !3, line: 1424, column: 5)
!3642 = !DILocation(line: 1424, column: 5, scope: !3562)
!3643 = !DILocalVariable(name: "i", scope: !3644, file: !3, line: 1425, type: !97)
!3644 = distinct !DILexicalBlock(scope: !3645, file: !3, line: 1425, column: 3)
!3645 = distinct !DILexicalBlock(scope: !3641, file: !3, line: 1424, column: 17)
!3646 = !DILocation(line: 0, scope: !3644)
!3647 = !DILocation(line: 1425, column: 7, scope: !3644)
!3648 = !DILocation(line: 1425, column: 17, scope: !3649)
!3649 = distinct !DILexicalBlock(scope: !3644, file: !3, line: 1425, column: 3)
!3650 = !DILocation(line: 1425, column: 3, scope: !3644)
!3651 = !DILocation(line: 1426, column: 19, scope: !3652)
!3652 = distinct !DILexicalBlock(scope: !3649, file: !3, line: 1425, column: 34)
!3653 = !DILocation(line: 1426, column: 4, scope: !3652)
!3654 = !DILocation(line: 1426, column: 17, scope: !3652)
!3655 = !DILocation(line: 1427, column: 3, scope: !3652)
!3656 = !DILocation(line: 1425, column: 31, scope: !3649)
!3657 = !DILocation(line: 1425, column: 3, scope: !3649)
!3658 = distinct !{!3658, !3650, !3659}
!3659 = !DILocation(line: 1427, column: 3, scope: !3644)
!3660 = !DILocation(line: 1428, column: 27, scope: !3645)
!3661 = !DILocation(line: 1428, column: 3, scope: !3645)
!3662 = !DILocation(line: 1428, column: 26, scope: !3645)
!3663 = !DILocation(line: 1429, column: 2, scope: !3645)
!3664 = !DILocation(line: 1430, column: 1, scope: !3562)
!3665 = !DILocation(line: 1161, column: 13, scope: !3666)
!3666 = distinct !DILexicalBlock(scope: !3371, file: !3, line: 1161, column: 5)
!3667 = !DILocation(line: 1161, column: 5, scope: !3371)
!3668 = !DILocalVariable(name: "i", scope: !3669, file: !3, line: 1162, type: !97)
!3669 = distinct !DILexicalBlock(scope: !3670, file: !3, line: 1162, column: 3)
!3670 = distinct !DILexicalBlock(scope: !3666, file: !3, line: 1161, column: 17)
!3671 = !DILocation(line: 0, scope: !3669)
!3672 = !DILocation(line: 1162, column: 7, scope: !3669)
!3673 = !DILocation(line: 1162, column: 17, scope: !3674)
!3674 = distinct !DILexicalBlock(scope: !3669, file: !3, line: 1162, column: 3)
!3675 = !DILocation(line: 1162, column: 3, scope: !3669)
!3676 = !DILocation(line: 1163, column: 19, scope: !3677)
!3677 = distinct !DILexicalBlock(scope: !3674, file: !3, line: 1162, column: 34)
!3678 = !DILocation(line: 1163, column: 4, scope: !3677)
!3679 = !DILocation(line: 1163, column: 17, scope: !3677)
!3680 = !DILocation(line: 1164, column: 3, scope: !3677)
!3681 = !DILocation(line: 1162, column: 31, scope: !3674)
!3682 = !DILocation(line: 1162, column: 3, scope: !3674)
!3683 = distinct !{!3683, !3675, !3684}
!3684 = !DILocation(line: 1164, column: 3, scope: !3669)
!3685 = !DILocation(line: 1165, column: 8, scope: !3670)
!3686 = !DILocation(line: 1165, column: 3, scope: !3670)
!3687 = !DILocation(line: 1165, column: 7, scope: !3670)
!3688 = !DILocation(line: 1166, column: 2, scope: !3670)
!3689 = !DILocation(line: 1167, column: 1, scope: !3371)
!3690 = !DILocation(line: 1298, column: 13, scope: !3691)
!3691 = distinct !DILexicalBlock(scope: !3478, file: !3, line: 1298, column: 5)
!3692 = !DILocation(line: 1298, column: 5, scope: !3478)
!3693 = !DILocalVariable(name: "i", scope: !3694, file: !3, line: 1299, type: !97)
!3694 = distinct !DILexicalBlock(scope: !3695, file: !3, line: 1299, column: 3)
!3695 = distinct !DILexicalBlock(scope: !3691, file: !3, line: 1298, column: 17)
!3696 = !DILocation(line: 0, scope: !3694)
!3697 = !DILocation(line: 1299, column: 7, scope: !3694)
!3698 = !DILocation(line: 1299, column: 17, scope: !3699)
!3699 = distinct !DILexicalBlock(scope: !3694, file: !3, line: 1299, column: 3)
!3700 = !DILocation(line: 1299, column: 3, scope: !3694)
!3701 = !DILocation(line: 1300, column: 19, scope: !3702)
!3702 = distinct !DILexicalBlock(scope: !3699, file: !3, line: 1299, column: 34)
!3703 = !DILocation(line: 1300, column: 4, scope: !3702)
!3704 = !DILocation(line: 1300, column: 17, scope: !3702)
!3705 = !DILocation(line: 1301, column: 3, scope: !3702)
!3706 = !DILocation(line: 1299, column: 31, scope: !3699)
!3707 = !DILocation(line: 1299, column: 3, scope: !3699)
!3708 = distinct !{!3708, !3700, !3709}
!3709 = !DILocation(line: 1301, column: 3, scope: !3694)
!3710 = !DILocation(line: 1302, column: 27, scope: !3695)
!3711 = !DILocation(line: 1302, column: 3, scope: !3695)
!3712 = !DILocation(line: 1302, column: 26, scope: !3695)
!3713 = !DILocation(line: 1303, column: 2, scope: !3695)
!3714 = !DILocation(line: 1304, column: 1, scope: !3478)
!3715 = !DILocation(line: 1372, column: 13, scope: !3716)
!3716 = distinct !DILexicalBlock(scope: !3521, file: !3, line: 1372, column: 5)
!3717 = !DILocation(line: 1372, column: 5, scope: !3521)
!3718 = !DILocalVariable(name: "i", scope: !3719, file: !3, line: 1373, type: !97)
!3719 = distinct !DILexicalBlock(scope: !3720, file: !3, line: 1373, column: 3)
!3720 = distinct !DILexicalBlock(scope: !3716, file: !3, line: 1372, column: 17)
!3721 = !DILocation(line: 0, scope: !3719)
!3722 = !DILocation(line: 1373, column: 7, scope: !3719)
!3723 = !DILocation(line: 1373, column: 17, scope: !3724)
!3724 = distinct !DILexicalBlock(scope: !3719, file: !3, line: 1373, column: 3)
!3725 = !DILocation(line: 1373, column: 3, scope: !3719)
!3726 = !DILocation(line: 1374, column: 19, scope: !3727)
!3727 = distinct !DILexicalBlock(scope: !3724, file: !3, line: 1373, column: 34)
!3728 = !DILocation(line: 1374, column: 4, scope: !3727)
!3729 = !DILocation(line: 1374, column: 17, scope: !3727)
!3730 = !DILocation(line: 1375, column: 3, scope: !3727)
!3731 = !DILocation(line: 1373, column: 31, scope: !3724)
!3732 = !DILocation(line: 1373, column: 3, scope: !3724)
!3733 = distinct !{!3733, !3725, !3734}
!3734 = !DILocation(line: 1375, column: 3, scope: !3719)
!3735 = !DILocation(line: 1376, column: 8, scope: !3720)
!3736 = !DILocation(line: 1376, column: 3, scope: !3720)
!3737 = !DILocation(line: 1376, column: 7, scope: !3720)
!3738 = !DILocation(line: 1377, column: 2, scope: !3720)
!3739 = !DILocation(line: 1378, column: 1, scope: !3521)
!3740 = !DILocation(line: 1109, column: 13, scope: !3741)
!3741 = distinct !DILexicalBlock(scope: !3347, file: !3, line: 1109, column: 5)
!3742 = !DILocation(line: 1109, column: 5, scope: !3347)
!3743 = !DILocalVariable(name: "i", scope: !3744, file: !3, line: 1110, type: !97)
!3744 = distinct !DILexicalBlock(scope: !3745, file: !3, line: 1110, column: 3)
!3745 = distinct !DILexicalBlock(scope: !3741, file: !3, line: 1109, column: 17)
!3746 = !DILocation(line: 0, scope: !3744)
!3747 = !DILocation(line: 1110, column: 7, scope: !3744)
!3748 = !DILocation(line: 1110, column: 17, scope: !3749)
!3749 = distinct !DILexicalBlock(scope: !3744, file: !3, line: 1110, column: 3)
!3750 = !DILocation(line: 1110, column: 3, scope: !3744)
!3751 = !DILocation(line: 1111, column: 19, scope: !3752)
!3752 = distinct !DILexicalBlock(scope: !3749, file: !3, line: 1110, column: 34)
!3753 = !DILocation(line: 1111, column: 4, scope: !3752)
!3754 = !DILocation(line: 1111, column: 17, scope: !3752)
!3755 = !DILocation(line: 1112, column: 3, scope: !3752)
!3756 = !DILocation(line: 1110, column: 31, scope: !3749)
!3757 = !DILocation(line: 1110, column: 3, scope: !3749)
!3758 = distinct !{!3758, !3750, !3759}
!3759 = !DILocation(line: 1112, column: 3, scope: !3744)
!3760 = !DILocation(line: 1113, column: 27, scope: !3745)
!3761 = !DILocation(line: 1113, column: 3, scope: !3745)
!3762 = !DILocation(line: 1113, column: 26, scope: !3745)
!3763 = !DILocation(line: 1114, column: 2, scope: !3745)
!3764 = !DILocation(line: 1115, column: 1, scope: !3347)
!3765 = !DILocation(line: 1217, column: 13, scope: !3766)
!3766 = distinct !DILexicalBlock(scope: !3414, file: !3, line: 1217, column: 5)
!3767 = !DILocation(line: 1217, column: 5, scope: !3414)
!3768 = !DILocalVariable(name: "i", scope: !3769, file: !3, line: 1218, type: !97)
!3769 = distinct !DILexicalBlock(scope: !3770, file: !3, line: 1218, column: 3)
!3770 = distinct !DILexicalBlock(scope: !3766, file: !3, line: 1217, column: 17)
!3771 = !DILocation(line: 0, scope: !3769)
!3772 = !DILocation(line: 1218, column: 7, scope: !3769)
!3773 = !DILocation(line: 1218, column: 17, scope: !3774)
!3774 = distinct !DILexicalBlock(scope: !3769, file: !3, line: 1218, column: 3)
!3775 = !DILocation(line: 1218, column: 3, scope: !3769)
!3776 = !DILocation(line: 1219, column: 19, scope: !3777)
!3777 = distinct !DILexicalBlock(scope: !3774, file: !3, line: 1218, column: 34)
!3778 = !DILocation(line: 1219, column: 4, scope: !3777)
!3779 = !DILocation(line: 1219, column: 17, scope: !3777)
!3780 = !DILocation(line: 1220, column: 3, scope: !3777)
!3781 = !DILocation(line: 1218, column: 31, scope: !3774)
!3782 = !DILocation(line: 1218, column: 3, scope: !3774)
!3783 = distinct !{!3783, !3775, !3784}
!3784 = !DILocation(line: 1220, column: 3, scope: !3769)
!3785 = !DILocation(line: 1221, column: 27, scope: !3770)
!3786 = !DILocation(line: 1221, column: 3, scope: !3770)
!3787 = !DILocation(line: 1221, column: 26, scope: !3770)
!3788 = !DILocation(line: 1222, column: 2, scope: !3770)
!3789 = !DILocation(line: 1223, column: 1, scope: !3414)
