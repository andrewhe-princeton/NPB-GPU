; ModuleID = 'ft_linked.bc'
source_filename = "llvm-link-cudafe"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.dcomplex = type { double, double }
%struct.cudaDeviceProp = type { [256 x i8], i64, i64, i32, i32, i64, i32, [3 x i32], [3 x i32], i32, i64, i32, i32, i64, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, [2 x i32], [2 x i32], [3 x i32], [2 x i32], [3 x i32], [3 x i32], i32, [2 x i32], [3 x i32], [2 x i32], i32, [2 x i32], [3 x i32], [2 x i32], [3 x i32], i32, [2 x i32], i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i64, i32, i32, i32, i32 }
%struct.dim3 = type { i32, i32, i32 }
%struct.CUstream_st = type opaque

$_ZN4dim3C2Ejjj = comdat any

@extern_share_data = external dso_local addrspace(3) global [0 x double], align 8
@starts_device = dso_local global double* null, align 8, !dbg !0
@twiddle_device = dso_local global double* null, align 8, !dbg !113
@sums_device = dso_local global %struct.dcomplex* null, align 8, !dbg !115
@u_device = dso_local global %struct.dcomplex* null, align 8, !dbg !117
@u0_device = dso_local global %struct.dcomplex* null, align 8, !dbg !119
@u1_device = dso_local global %struct.dcomplex* null, align 8, !dbg !121
@u2_device = dso_local global %struct.dcomplex* null, align 8, !dbg !123
@y0_device = dso_local global %struct.dcomplex* null, align 8, !dbg !125
@y1_device = dso_local global %struct.dcomplex* null, align 8, !dbg !127
@size_sums_device = dso_local global i64 0, align 8, !dbg !129
@size_starts_device = dso_local global i64 0, align 8, !dbg !134
@size_twiddle_device = dso_local global i64 0, align 8, !dbg !136
@size_u_device = dso_local global i64 0, align 8, !dbg !138
@size_u0_device = dso_local global i64 0, align 8, !dbg !140
@size_u1_device = dso_local global i64 0, align 8, !dbg !142
@size_y0_device = dso_local global i64 0, align 8, !dbg !144
@size_y1_device = dso_local global i64 0, align 8, !dbg !146
@size_shared_data = dso_local global i64 0, align 8, !dbg !148
@blocks_per_grid_on_compute_indexmap = dso_local global i32 0, align 4, !dbg !150
@blocks_per_grid_on_compute_initial_conditions = dso_local global i32 0, align 4, !dbg !152
@blocks_per_grid_on_init_ui = dso_local global i32 0, align 4, !dbg !154
@blocks_per_grid_on_evolve = dso_local global i32 0, align 4, !dbg !156
@blocks_per_grid_on_fftx_1 = dso_local global i32 0, align 4, !dbg !158
@blocks_per_grid_on_fftx_2 = dso_local global i32 0, align 4, !dbg !160
@blocks_per_grid_on_fftx_3 = dso_local global i32 0, align 4, !dbg !162
@blocks_per_grid_on_ffty_1 = dso_local global i32 0, align 4, !dbg !164
@blocks_per_grid_on_ffty_2 = dso_local global i32 0, align 4, !dbg !166
@blocks_per_grid_on_ffty_3 = dso_local global i32 0, align 4, !dbg !168
@blocks_per_grid_on_fftz_1 = dso_local global i32 0, align 4, !dbg !170
@blocks_per_grid_on_fftz_2 = dso_local global i32 0, align 4, !dbg !172
@blocks_per_grid_on_fftz_3 = dso_local global i32 0, align 4, !dbg !174
@blocks_per_grid_on_checksum = dso_local global i32 0, align 4, !dbg !176
@threads_per_block_on_compute_indexmap = dso_local global i32 0, align 4, !dbg !178
@threads_per_block_on_compute_initial_conditions = dso_local global i32 0, align 4, !dbg !180
@threads_per_block_on_init_ui = dso_local global i32 0, align 4, !dbg !182
@threads_per_block_on_evolve = dso_local global i32 0, align 4, !dbg !184
@threads_per_block_on_fftx_1 = dso_local global i32 0, align 4, !dbg !186
@threads_per_block_on_fftx_2 = dso_local global i32 0, align 4, !dbg !188
@threads_per_block_on_fftx_3 = dso_local global i32 0, align 4, !dbg !190
@threads_per_block_on_ffty_1 = dso_local global i32 0, align 4, !dbg !192
@threads_per_block_on_ffty_2 = dso_local global i32 0, align 4, !dbg !194
@threads_per_block_on_ffty_3 = dso_local global i32 0, align 4, !dbg !196
@threads_per_block_on_fftz_1 = dso_local global i32 0, align 4, !dbg !198
@threads_per_block_on_fftz_2 = dso_local global i32 0, align 4, !dbg !200
@threads_per_block_on_fftz_3 = dso_local global i32 0, align 4, !dbg !202
@threads_per_block_on_checksum = dso_local global i32 0, align 4, !dbg !204
@gpu_device_id = dso_local global i32 0, align 4, !dbg !206
@total_devices = dso_local global i32 0, align 4, !dbg !208
@gpu_device_properties = dso_local global %struct.cudaDeviceProp zeroinitializer, align 8, !dbg !210
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
@_ZL4sums = internal global %struct.dcomplex* null, align 8, !dbg !285
@_ZL7twiddle = internal global double* null, align 8, !dbg !287
@_ZL1u = internal global %struct.dcomplex* null, align 8, !dbg !289
@_ZL2u0 = internal global %struct.dcomplex* null, align 8, !dbg !291
@_ZL2u1 = internal global %struct.dcomplex* null, align 8, !dbg !293
@_ZL4dims = internal global i32* null, align 8, !dbg !295
@_ZL5niter = internal global i32 0, align 4, !dbg !297
@.str.39 = private unnamed_addr constant [40 x i8] c"T = %5d     Checksum = %22.12e %22.12e\0A\00", align 1
@.str.40 = private unnamed_addr constant [10 x i8] c"%5s\09%25s\0A\00", align 1
@.str.41 = private unnamed_addr constant [11 x i8] c"GPU Kernel\00", align 1
@.str.42 = private unnamed_addr constant [18 x i8] c"Threads Per Block\00", align 1
@.str.43 = private unnamed_addr constant [11 x i8] c"%29s\09%25d\0A\00", align 1
@.str.44 = private unnamed_addr constant [10 x i8] c" indexmap\00", align 1
@.str.45 = private unnamed_addr constant [20 x i8] c" initial conditions\00", align 1
@.str.46 = private unnamed_addr constant [9 x i8] c" init ui\00", align 1
@.str.47 = private unnamed_addr constant [8 x i8] c" evolve\00", align 1
@.str.48 = private unnamed_addr constant [8 x i8] c" fftx 1\00", align 1
@.str.49 = private unnamed_addr constant [8 x i8] c" fftx 2\00", align 1
@.str.50 = private unnamed_addr constant [8 x i8] c" fftx 3\00", align 1
@.str.51 = private unnamed_addr constant [8 x i8] c" ffty 1\00", align 1
@.str.52 = private unnamed_addr constant [8 x i8] c" ffty 2\00", align 1
@.str.53 = private unnamed_addr constant [8 x i8] c" ffty 3\00", align 1
@.str.54 = private unnamed_addr constant [8 x i8] c" fftz 1\00", align 1
@.str.55 = private unnamed_addr constant [8 x i8] c" fftz 2\00", align 1
@.str.56 = private unnamed_addr constant [8 x i8] c" fftz 3\00", align 1
@.str.57 = private unnamed_addr constant [10 x i8] c" checksum\00", align 1
@.str.58 = private unnamed_addr constant [3 x i8] c"FT\00", align 1
@.str.59 = private unnamed_addr constant [25 x i8] c"          floating point\00", align 1
@.str.60 = private unnamed_addr constant [4 x i8] c"4.1\00", align 1
@.str.61 = private unnamed_addr constant [12 x i8] c"10 Feb 2026\00", align 1
@.str.62 = private unnamed_addr constant [6 x i8] c"Jr\88\FF\7F\00", align 1
@.str.63 = private unnamed_addr constant [42 x i8] c"Intel(R) Xeon(R) CPU E5-2697 v3 @ 2.60GHz\00", align 1
@.str.64 = private unnamed_addr constant [23 x i8] c"${NVCC} ${EXTRA_STUFF}\00", align 1
@.str.65 = private unnamed_addr constant [6 x i8] c"$(CC)\00", align 1
@.str.66 = private unnamed_addr constant [5 x i8] c"-lm \00", align 1
@.str.67 = private unnamed_addr constant [13 x i8] c"-I../common \00", align 1
@.str.68 = private unnamed_addr constant [4 x i8] c"-O3\00", align 1
@.str.69 = private unnamed_addr constant [7 x i8] c"randdp\00", align 1
@.str.73 = private unnamed_addr constant [33 x i8] c" Result verification successful\0A\00", align 1
@.str.74 = private unnamed_addr constant [29 x i8] c" Result verification failed\0A\00", align 1
@.str.75 = private unnamed_addr constant [17 x i8] c" class_npb = %c\0A\00", align 1
@.str.70 = private unnamed_addr constant [65 x i8] c"\0A\0A NAS Parallel Benchmarks 4.1 CUDA C++ version - FT Benchmark\0A\0A\00", align 1
@.str.71 = private unnamed_addr constant [36 x i8] c" Size                : %4dx%4dx%4d\0A\00", align 1
@.str.72 = private unnamed_addr constant [35 x i8] c" Iterations                  :%7d\0A\00", align 1

; Function Attrs: convergent noinline nounwind
define dso_local void @_Z19cffts1_gpu_kernel_1P8dcomplexS0_(%struct.dcomplex* %x_in, %struct.dcomplex* %y0) #0 !dbg !1152 {
entry:
  %x_in.addr = alloca %struct.dcomplex*, align 8
  %y0.addr = alloca %struct.dcomplex*, align 8
  %x_y_z = alloca i32, align 4
  %x = alloca i32, align 4
  %y = alloca i32, align 4
  %z = alloca i32, align 4
  store %struct.dcomplex* %x_in, %struct.dcomplex** %x_in.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %x_in.addr, metadata !1155, metadata !DIExpression()), !dbg !1156
  store %struct.dcomplex* %y0, %struct.dcomplex** %y0.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %y0.addr, metadata !1157, metadata !DIExpression()), !dbg !1158
  call void @llvm.dbg.declare(metadata i32* %x_y_z, metadata !1159, metadata !DIExpression()), !dbg !1160
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #11, !dbg !1161, !range !1198
  %1 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #11, !dbg !1199, !range !1243
  %mul = mul i32 %0, %1, !dbg !1244
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #11, !dbg !1245, !range !1273
  %add = add i32 %mul, %2, !dbg !1274
  store i32 %add, i32* %x_y_z, align 4, !dbg !1160
  %3 = load i32, i32* %x_y_z, align 4, !dbg !1275
  %cmp = icmp sge i32 %3, 8388608, !dbg !1277
  br i1 %cmp, label %if.then, label %if.end, !dbg !1278

if.then:                                          ; preds = %entry
  br label %return, !dbg !1279

if.end:                                           ; preds = %entry
  call void @llvm.dbg.declare(metadata i32* %x, metadata !1281, metadata !DIExpression()), !dbg !1282
  %4 = load i32, i32* %x_y_z, align 4, !dbg !1283
  %rem = srem i32 %4, 256, !dbg !1284
  store i32 %rem, i32* %x, align 4, !dbg !1282
  call void @llvm.dbg.declare(metadata i32* %y, metadata !1285, metadata !DIExpression()), !dbg !1286
  %5 = load i32, i32* %x_y_z, align 4, !dbg !1287
  %div = sdiv i32 %5, 256, !dbg !1288
  %rem3 = srem i32 %div, 256, !dbg !1289
  store i32 %rem3, i32* %y, align 4, !dbg !1286
  call void @llvm.dbg.declare(metadata i32* %z, metadata !1290, metadata !DIExpression()), !dbg !1291
  %6 = load i32, i32* %x_y_z, align 4, !dbg !1292
  %div4 = sdiv i32 %6, 65536, !dbg !1293
  store i32 %div4, i32* %z, align 4, !dbg !1291
  %7 = load %struct.dcomplex*, %struct.dcomplex** %x_in.addr, align 8, !dbg !1294
  %8 = load i32, i32* %x_y_z, align 4, !dbg !1295
  %idxprom = sext i32 %8 to i64, !dbg !1294
  %arrayidx = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %7, i64 %idxprom, !dbg !1294
  %real = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx, i32 0, i32 0, !dbg !1296
  %9 = load double, double* %real, align 8, !dbg !1296
  %10 = load %struct.dcomplex*, %struct.dcomplex** %y0.addr, align 8, !dbg !1297
  %11 = load i32, i32* %y, align 4, !dbg !1298
  %12 = load i32, i32* %x, align 4, !dbg !1299
  %mul5 = mul nsw i32 %12, 256, !dbg !1300
  %add6 = add nsw i32 %11, %mul5, !dbg !1301
  %13 = load i32, i32* %z, align 4, !dbg !1302
  %mul7 = mul nsw i32 %13, 256, !dbg !1303
  %mul8 = mul nsw i32 %mul7, 256, !dbg !1304
  %add9 = add nsw i32 %add6, %mul8, !dbg !1305
  %idxprom10 = sext i32 %add9 to i64, !dbg !1297
  %arrayidx11 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %10, i64 %idxprom10, !dbg !1297
  %real12 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx11, i32 0, i32 0, !dbg !1306
  store double %9, double* %real12, align 8, !dbg !1307
  %14 = load %struct.dcomplex*, %struct.dcomplex** %x_in.addr, align 8, !dbg !1308
  %15 = load i32, i32* %x_y_z, align 4, !dbg !1309
  %idxprom13 = sext i32 %15 to i64, !dbg !1308
  %arrayidx14 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %14, i64 %idxprom13, !dbg !1308
  %imag = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx14, i32 0, i32 1, !dbg !1310
  %16 = load double, double* %imag, align 8, !dbg !1310
  %17 = load %struct.dcomplex*, %struct.dcomplex** %y0.addr, align 8, !dbg !1311
  %18 = load i32, i32* %y, align 4, !dbg !1312
  %19 = load i32, i32* %x, align 4, !dbg !1313
  %mul15 = mul nsw i32 %19, 256, !dbg !1314
  %add16 = add nsw i32 %18, %mul15, !dbg !1315
  %20 = load i32, i32* %z, align 4, !dbg !1316
  %mul17 = mul nsw i32 %20, 256, !dbg !1317
  %mul18 = mul nsw i32 %mul17, 256, !dbg !1318
  %add19 = add nsw i32 %add16, %mul18, !dbg !1319
  %idxprom20 = sext i32 %add19 to i64, !dbg !1311
  %arrayidx21 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %17, i64 %idxprom20, !dbg !1311
  %imag22 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx21, i32 0, i32 1, !dbg !1320
  store double %16, double* %imag22, align 8, !dbg !1321
  br label %return, !dbg !1322

return:                                           ; preds = %if.end, %if.then
  ret void, !dbg !1322
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #2

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #2

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() #2

; Function Attrs: convergent noinline nounwind
define dso_local void @_Z19cffts1_gpu_kernel_2iP8dcomplexS0_S0_(i32 %is, %struct.dcomplex* %gty1, %struct.dcomplex* %gty2, %struct.dcomplex* %u_device) #0 !dbg !1323 {
entry:
  %is.addr = alloca i32, align 4
  %gty1.addr = alloca %struct.dcomplex*, align 8
  %gty2.addr = alloca %struct.dcomplex*, align 8
  %u_device.addr = alloca %struct.dcomplex*, align 8
  %y_z = alloca i32, align 4
  %j = alloca i32, align 4
  %k = alloca i32, align 4
  %l = alloca i32, align 4
  %j1 = alloca i32, align 4
  %i1 = alloca i32, align 4
  %k1 = alloca i32, align 4
  %n1 = alloca i32, align 4
  %li = alloca i32, align 4
  %lj = alloca i32, align 4
  %lk = alloca i32, align 4
  %ku = alloca i32, align 4
  %i11 = alloca i32, align 4
  %i12 = alloca i32, align 4
  %i21 = alloca i32, align 4
  %i22 = alloca i32, align 4
  %logd1 = alloca i32, align 4
  %uu1_real = alloca double, align 8
  %x11_real = alloca double, align 8
  %x21_real = alloca double, align 8
  %uu1_imag = alloca double, align 8
  %x11_imag = alloca double, align 8
  %x21_imag = alloca double, align 8
  %uu2_real = alloca double, align 8
  %x12_real = alloca double, align 8
  %x22_real = alloca double, align 8
  %uu2_imag = alloca double, align 8
  %x12_imag = alloca double, align 8
  %x22_imag = alloca double, align 8
  %temp_real = alloca double, align 8
  %temp2_real = alloca double, align 8
  %temp_imag = alloca double, align 8
  %temp2_imag = alloca double, align 8
  store i32 %is, i32* %is.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %is.addr, metadata !1327, metadata !DIExpression()), !dbg !1328
  store %struct.dcomplex* %gty1, %struct.dcomplex** %gty1.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %gty1.addr, metadata !1329, metadata !DIExpression()), !dbg !1330
  store %struct.dcomplex* %gty2, %struct.dcomplex** %gty2.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %gty2.addr, metadata !1331, metadata !DIExpression()), !dbg !1332
  store %struct.dcomplex* %u_device, %struct.dcomplex** %u_device.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %u_device.addr, metadata !1333, metadata !DIExpression()), !dbg !1334
  call void @llvm.dbg.declare(metadata i32* %y_z, metadata !1335, metadata !DIExpression()), !dbg !1336
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #11, !dbg !1337, !range !1198
  %1 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #11, !dbg !1339, !range !1243
  %mul = mul i32 %0, %1, !dbg !1341
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #11, !dbg !1342, !range !1273
  %add = add i32 %mul, %2, !dbg !1344
  store i32 %add, i32* %y_z, align 4, !dbg !1336
  %3 = load i32, i32* %y_z, align 4, !dbg !1345
  %cmp = icmp sge i32 %3, 32768, !dbg !1347
  br i1 %cmp, label %if.then, label %if.end, !dbg !1348

if.then:                                          ; preds = %entry
  br label %for.end271, !dbg !1349

if.end:                                           ; preds = %entry
  call void @llvm.dbg.declare(metadata i32* %j, metadata !1351, metadata !DIExpression()), !dbg !1352
  call void @llvm.dbg.declare(metadata i32* %k, metadata !1353, metadata !DIExpression()), !dbg !1354
  call void @llvm.dbg.declare(metadata i32* %l, metadata !1355, metadata !DIExpression()), !dbg !1356
  call void @llvm.dbg.declare(metadata i32* %j1, metadata !1357, metadata !DIExpression()), !dbg !1358
  call void @llvm.dbg.declare(metadata i32* %i1, metadata !1359, metadata !DIExpression()), !dbg !1360
  call void @llvm.dbg.declare(metadata i32* %k1, metadata !1361, metadata !DIExpression()), !dbg !1362
  call void @llvm.dbg.declare(metadata i32* %n1, metadata !1363, metadata !DIExpression()), !dbg !1364
  call void @llvm.dbg.declare(metadata i32* %li, metadata !1365, metadata !DIExpression()), !dbg !1366
  call void @llvm.dbg.declare(metadata i32* %lj, metadata !1367, metadata !DIExpression()), !dbg !1368
  call void @llvm.dbg.declare(metadata i32* %lk, metadata !1369, metadata !DIExpression()), !dbg !1370
  call void @llvm.dbg.declare(metadata i32* %ku, metadata !1371, metadata !DIExpression()), !dbg !1372
  call void @llvm.dbg.declare(metadata i32* %i11, metadata !1373, metadata !DIExpression()), !dbg !1374
  call void @llvm.dbg.declare(metadata i32* %i12, metadata !1375, metadata !DIExpression()), !dbg !1376
  call void @llvm.dbg.declare(metadata i32* %i21, metadata !1377, metadata !DIExpression()), !dbg !1378
  call void @llvm.dbg.declare(metadata i32* %i22, metadata !1379, metadata !DIExpression()), !dbg !1380
  %4 = load i32, i32* %y_z, align 4, !dbg !1381
  %rem = srem i32 %4, 256, !dbg !1382
  store i32 %rem, i32* %j, align 4, !dbg !1383
  %5 = load i32, i32* %y_z, align 4, !dbg !1384
  %div = sdiv i32 %5, 256, !dbg !1385
  %rem3 = srem i32 %div, 128, !dbg !1386
  store i32 %rem3, i32* %k, align 4, !dbg !1387
  call void @llvm.dbg.declare(metadata i32* %logd1, metadata !1388, metadata !DIExpression()), !dbg !1389
  %call4 = call i32 @_Z12ilog2_devicei(i32 256) #4, !dbg !1390
  store i32 %call4, i32* %logd1, align 4, !dbg !1389
  call void @llvm.dbg.declare(metadata double* %uu1_real, metadata !1391, metadata !DIExpression()), !dbg !1392
  call void @llvm.dbg.declare(metadata double* %x11_real, metadata !1393, metadata !DIExpression()), !dbg !1394
  call void @llvm.dbg.declare(metadata double* %x21_real, metadata !1395, metadata !DIExpression()), !dbg !1396
  call void @llvm.dbg.declare(metadata double* %uu1_imag, metadata !1397, metadata !DIExpression()), !dbg !1398
  call void @llvm.dbg.declare(metadata double* %x11_imag, metadata !1399, metadata !DIExpression()), !dbg !1400
  call void @llvm.dbg.declare(metadata double* %x21_imag, metadata !1401, metadata !DIExpression()), !dbg !1402
  call void @llvm.dbg.declare(metadata double* %uu2_real, metadata !1403, metadata !DIExpression()), !dbg !1404
  call void @llvm.dbg.declare(metadata double* %x12_real, metadata !1405, metadata !DIExpression()), !dbg !1406
  call void @llvm.dbg.declare(metadata double* %x22_real, metadata !1407, metadata !DIExpression()), !dbg !1408
  call void @llvm.dbg.declare(metadata double* %uu2_imag, metadata !1409, metadata !DIExpression()), !dbg !1410
  call void @llvm.dbg.declare(metadata double* %x12_imag, metadata !1411, metadata !DIExpression()), !dbg !1412
  call void @llvm.dbg.declare(metadata double* %x22_imag, metadata !1413, metadata !DIExpression()), !dbg !1414
  call void @llvm.dbg.declare(metadata double* %temp_real, metadata !1415, metadata !DIExpression()), !dbg !1416
  call void @llvm.dbg.declare(metadata double* %temp2_real, metadata !1417, metadata !DIExpression()), !dbg !1418
  call void @llvm.dbg.declare(metadata double* %temp_imag, metadata !1419, metadata !DIExpression()), !dbg !1420
  call void @llvm.dbg.declare(metadata double* %temp2_imag, metadata !1421, metadata !DIExpression()), !dbg !1422
  store i32 1, i32* %l, align 4, !dbg !1423
  br label %for.cond, !dbg !1425

for.cond:                                         ; preds = %for.inc269, %if.end
  %6 = load i32, i32* %l, align 4, !dbg !1426
  %7 = load i32, i32* %logd1, align 4, !dbg !1428
  %cmp5 = icmp sle i32 %6, %7, !dbg !1429
  br i1 %cmp5, label %for.body, label %for.end271, !dbg !1430

for.body:                                         ; preds = %for.cond
  store i32 128, i32* %n1, align 4, !dbg !1431
  %8 = load i32, i32* %l, align 4, !dbg !1433
  %sub = sub nsw i32 %8, 1, !dbg !1434
  %shl = shl i32 1, %sub, !dbg !1435
  store i32 %shl, i32* %lk, align 4, !dbg !1436
  %9 = load i32, i32* %logd1, align 4, !dbg !1437
  %10 = load i32, i32* %l, align 4, !dbg !1438
  %sub6 = sub nsw i32 %9, %10, !dbg !1439
  %shl7 = shl i32 1, %sub6, !dbg !1440
  store i32 %shl7, i32* %li, align 4, !dbg !1441
  %11 = load i32, i32* %lk, align 4, !dbg !1442
  %mul8 = mul nsw i32 2, %11, !dbg !1443
  store i32 %mul8, i32* %lj, align 4, !dbg !1444
  %12 = load i32, i32* %li, align 4, !dbg !1445
  store i32 %12, i32* %ku, align 4, !dbg !1446
  store i32 0, i32* %i1, align 4, !dbg !1447
  br label %for.cond9, !dbg !1449

for.cond9:                                        ; preds = %for.inc108, %for.body
  %13 = load i32, i32* %i1, align 4, !dbg !1450
  %14 = load i32, i32* %li, align 4, !dbg !1452
  %sub10 = sub nsw i32 %14, 1, !dbg !1453
  %cmp11 = icmp sle i32 %13, %sub10, !dbg !1454
  br i1 %cmp11, label %for.body12, label %for.end110, !dbg !1455

for.body12:                                       ; preds = %for.cond9
  store i32 0, i32* %k1, align 4, !dbg !1456
  br label %for.cond13, !dbg !1459

for.cond13:                                       ; preds = %for.inc, %for.body12
  %15 = load i32, i32* %k1, align 4, !dbg !1460
  %16 = load i32, i32* %lk, align 4, !dbg !1462
  %sub14 = sub nsw i32 %16, 1, !dbg !1463
  %cmp15 = icmp sle i32 %15, %sub14, !dbg !1464
  br i1 %cmp15, label %for.body16, label %for.end, !dbg !1465

for.body16:                                       ; preds = %for.cond13
  %17 = load i32, i32* %i1, align 4, !dbg !1466
  %18 = load i32, i32* %lk, align 4, !dbg !1468
  %mul17 = mul nsw i32 %17, %18, !dbg !1469
  store i32 %mul17, i32* %i11, align 4, !dbg !1470
  %19 = load i32, i32* %i11, align 4, !dbg !1471
  %20 = load i32, i32* %n1, align 4, !dbg !1472
  %add18 = add nsw i32 %19, %20, !dbg !1473
  store i32 %add18, i32* %i12, align 4, !dbg !1474
  %21 = load i32, i32* %i1, align 4, !dbg !1475
  %22 = load i32, i32* %lj, align 4, !dbg !1476
  %mul19 = mul nsw i32 %21, %22, !dbg !1477
  store i32 %mul19, i32* %i21, align 4, !dbg !1478
  %23 = load i32, i32* %i21, align 4, !dbg !1479
  %24 = load i32, i32* %lk, align 4, !dbg !1480
  %add20 = add nsw i32 %23, %24, !dbg !1481
  store i32 %add20, i32* %i22, align 4, !dbg !1482
  %25 = load %struct.dcomplex*, %struct.dcomplex** %u_device.addr, align 8, !dbg !1483
  %26 = load i32, i32* %ku, align 4, !dbg !1484
  %27 = load i32, i32* %i1, align 4, !dbg !1485
  %add21 = add nsw i32 %26, %27, !dbg !1486
  %idxprom = sext i32 %add21 to i64, !dbg !1483
  %arrayidx = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %25, i64 %idxprom, !dbg !1483
  %real = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx, i32 0, i32 0, !dbg !1487
  %28 = load double, double* %real, align 8, !dbg !1487
  store double %28, double* %uu1_real, align 8, !dbg !1488
  %29 = load i32, i32* %is.addr, align 4, !dbg !1489
  %conv = sitofp i32 %29 to double, !dbg !1489
  %30 = load %struct.dcomplex*, %struct.dcomplex** %u_device.addr, align 8, !dbg !1490
  %31 = load i32, i32* %ku, align 4, !dbg !1491
  %32 = load i32, i32* %i1, align 4, !dbg !1492
  %add22 = add nsw i32 %31, %32, !dbg !1493
  %idxprom23 = sext i32 %add22 to i64, !dbg !1490
  %arrayidx24 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %30, i64 %idxprom23, !dbg !1490
  %imag = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx24, i32 0, i32 1, !dbg !1494
  %33 = load double, double* %imag, align 8, !dbg !1494
  %mul25 = fmul contract double %conv, %33, !dbg !1495
  store double %mul25, double* %uu1_imag, align 8, !dbg !1496
  %34 = load %struct.dcomplex*, %struct.dcomplex** %gty1.addr, align 8, !dbg !1497
  %35 = load i32, i32* %j, align 4, !dbg !1498
  %36 = load i32, i32* %i11, align 4, !dbg !1499
  %37 = load i32, i32* %k1, align 4, !dbg !1500
  %add26 = add nsw i32 %36, %37, !dbg !1501
  %mul27 = mul nsw i32 %add26, 256, !dbg !1502
  %add28 = add nsw i32 %35, %mul27, !dbg !1503
  %38 = load i32, i32* %k, align 4, !dbg !1504
  %mul29 = mul nsw i32 %38, 256, !dbg !1505
  %mul30 = mul nsw i32 %mul29, 256, !dbg !1506
  %add31 = add nsw i32 %add28, %mul30, !dbg !1507
  %idxprom32 = sext i32 %add31 to i64, !dbg !1497
  %arrayidx33 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %34, i64 %idxprom32, !dbg !1497
  %real34 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx33, i32 0, i32 0, !dbg !1508
  %39 = load double, double* %real34, align 8, !dbg !1508
  store double %39, double* %x11_real, align 8, !dbg !1509
  %40 = load %struct.dcomplex*, %struct.dcomplex** %gty1.addr, align 8, !dbg !1510
  %41 = load i32, i32* %j, align 4, !dbg !1511
  %42 = load i32, i32* %i11, align 4, !dbg !1512
  %43 = load i32, i32* %k1, align 4, !dbg !1513
  %add35 = add nsw i32 %42, %43, !dbg !1514
  %mul36 = mul nsw i32 %add35, 256, !dbg !1515
  %add37 = add nsw i32 %41, %mul36, !dbg !1516
  %44 = load i32, i32* %k, align 4, !dbg !1517
  %mul38 = mul nsw i32 %44, 256, !dbg !1518
  %mul39 = mul nsw i32 %mul38, 256, !dbg !1519
  %add40 = add nsw i32 %add37, %mul39, !dbg !1520
  %idxprom41 = sext i32 %add40 to i64, !dbg !1510
  %arrayidx42 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %40, i64 %idxprom41, !dbg !1510
  %imag43 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx42, i32 0, i32 1, !dbg !1521
  %45 = load double, double* %imag43, align 8, !dbg !1521
  store double %45, double* %x11_imag, align 8, !dbg !1522
  %46 = load %struct.dcomplex*, %struct.dcomplex** %gty1.addr, align 8, !dbg !1523
  %47 = load i32, i32* %j, align 4, !dbg !1524
  %48 = load i32, i32* %i12, align 4, !dbg !1525
  %49 = load i32, i32* %k1, align 4, !dbg !1526
  %add44 = add nsw i32 %48, %49, !dbg !1527
  %mul45 = mul nsw i32 %add44, 256, !dbg !1528
  %add46 = add nsw i32 %47, %mul45, !dbg !1529
  %50 = load i32, i32* %k, align 4, !dbg !1530
  %mul47 = mul nsw i32 %50, 256, !dbg !1531
  %mul48 = mul nsw i32 %mul47, 256, !dbg !1532
  %add49 = add nsw i32 %add46, %mul48, !dbg !1533
  %idxprom50 = sext i32 %add49 to i64, !dbg !1523
  %arrayidx51 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %46, i64 %idxprom50, !dbg !1523
  %real52 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx51, i32 0, i32 0, !dbg !1534
  %51 = load double, double* %real52, align 8, !dbg !1534
  store double %51, double* %x21_real, align 8, !dbg !1535
  %52 = load %struct.dcomplex*, %struct.dcomplex** %gty1.addr, align 8, !dbg !1536
  %53 = load i32, i32* %j, align 4, !dbg !1537
  %54 = load i32, i32* %i12, align 4, !dbg !1538
  %55 = load i32, i32* %k1, align 4, !dbg !1539
  %add53 = add nsw i32 %54, %55, !dbg !1540
  %mul54 = mul nsw i32 %add53, 256, !dbg !1541
  %add55 = add nsw i32 %53, %mul54, !dbg !1542
  %56 = load i32, i32* %k, align 4, !dbg !1543
  %mul56 = mul nsw i32 %56, 256, !dbg !1544
  %mul57 = mul nsw i32 %mul56, 256, !dbg !1545
  %add58 = add nsw i32 %add55, %mul57, !dbg !1546
  %idxprom59 = sext i32 %add58 to i64, !dbg !1536
  %arrayidx60 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %52, i64 %idxprom59, !dbg !1536
  %imag61 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx60, i32 0, i32 1, !dbg !1547
  %57 = load double, double* %imag61, align 8, !dbg !1547
  store double %57, double* %x21_imag, align 8, !dbg !1548
  %58 = load double, double* %x11_real, align 8, !dbg !1549
  %59 = load double, double* %x21_real, align 8, !dbg !1550
  %add62 = fadd contract double %58, %59, !dbg !1551
  %60 = load %struct.dcomplex*, %struct.dcomplex** %gty2.addr, align 8, !dbg !1552
  %61 = load i32, i32* %j, align 4, !dbg !1553
  %62 = load i32, i32* %i21, align 4, !dbg !1554
  %63 = load i32, i32* %k1, align 4, !dbg !1555
  %add63 = add nsw i32 %62, %63, !dbg !1556
  %mul64 = mul nsw i32 %add63, 256, !dbg !1557
  %add65 = add nsw i32 %61, %mul64, !dbg !1558
  %64 = load i32, i32* %k, align 4, !dbg !1559
  %mul66 = mul nsw i32 %64, 256, !dbg !1560
  %mul67 = mul nsw i32 %mul66, 256, !dbg !1561
  %add68 = add nsw i32 %add65, %mul67, !dbg !1562
  %idxprom69 = sext i32 %add68 to i64, !dbg !1552
  %arrayidx70 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %60, i64 %idxprom69, !dbg !1552
  %real71 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx70, i32 0, i32 0, !dbg !1563
  store double %add62, double* %real71, align 8, !dbg !1564
  %65 = load double, double* %x11_imag, align 8, !dbg !1565
  %66 = load double, double* %x21_imag, align 8, !dbg !1566
  %add72 = fadd contract double %65, %66, !dbg !1567
  %67 = load %struct.dcomplex*, %struct.dcomplex** %gty2.addr, align 8, !dbg !1568
  %68 = load i32, i32* %j, align 4, !dbg !1569
  %69 = load i32, i32* %i21, align 4, !dbg !1570
  %70 = load i32, i32* %k1, align 4, !dbg !1571
  %add73 = add nsw i32 %69, %70, !dbg !1572
  %mul74 = mul nsw i32 %add73, 256, !dbg !1573
  %add75 = add nsw i32 %68, %mul74, !dbg !1574
  %71 = load i32, i32* %k, align 4, !dbg !1575
  %mul76 = mul nsw i32 %71, 256, !dbg !1576
  %mul77 = mul nsw i32 %mul76, 256, !dbg !1577
  %add78 = add nsw i32 %add75, %mul77, !dbg !1578
  %idxprom79 = sext i32 %add78 to i64, !dbg !1568
  %arrayidx80 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %67, i64 %idxprom79, !dbg !1568
  %imag81 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx80, i32 0, i32 1, !dbg !1579
  store double %add72, double* %imag81, align 8, !dbg !1580
  %72 = load double, double* %x11_real, align 8, !dbg !1581
  %73 = load double, double* %x21_real, align 8, !dbg !1582
  %sub82 = fsub contract double %72, %73, !dbg !1583
  store double %sub82, double* %temp_real, align 8, !dbg !1584
  %74 = load double, double* %x11_imag, align 8, !dbg !1585
  %75 = load double, double* %x21_imag, align 8, !dbg !1586
  %sub83 = fsub contract double %74, %75, !dbg !1587
  store double %sub83, double* %temp_imag, align 8, !dbg !1588
  %76 = load double, double* %uu1_real, align 8, !dbg !1589
  %77 = load double, double* %temp_real, align 8, !dbg !1590
  %mul84 = fmul contract double %76, %77, !dbg !1591
  %78 = load double, double* %uu1_imag, align 8, !dbg !1592
  %79 = load double, double* %temp_imag, align 8, !dbg !1593
  %mul85 = fmul contract double %78, %79, !dbg !1594
  %sub86 = fsub contract double %mul84, %mul85, !dbg !1595
  %80 = load %struct.dcomplex*, %struct.dcomplex** %gty2.addr, align 8, !dbg !1596
  %81 = load i32, i32* %j, align 4, !dbg !1597
  %82 = load i32, i32* %i22, align 4, !dbg !1598
  %83 = load i32, i32* %k1, align 4, !dbg !1599
  %add87 = add nsw i32 %82, %83, !dbg !1600
  %mul88 = mul nsw i32 %add87, 256, !dbg !1601
  %add89 = add nsw i32 %81, %mul88, !dbg !1602
  %84 = load i32, i32* %k, align 4, !dbg !1603
  %mul90 = mul nsw i32 %84, 256, !dbg !1604
  %mul91 = mul nsw i32 %mul90, 256, !dbg !1605
  %add92 = add nsw i32 %add89, %mul91, !dbg !1606
  %idxprom93 = sext i32 %add92 to i64, !dbg !1596
  %arrayidx94 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %80, i64 %idxprom93, !dbg !1596
  %real95 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx94, i32 0, i32 0, !dbg !1607
  store double %sub86, double* %real95, align 8, !dbg !1608
  %85 = load double, double* %uu1_real, align 8, !dbg !1609
  %86 = load double, double* %temp_imag, align 8, !dbg !1610
  %mul96 = fmul contract double %85, %86, !dbg !1611
  %87 = load double, double* %uu1_imag, align 8, !dbg !1612
  %88 = load double, double* %temp_real, align 8, !dbg !1613
  %mul97 = fmul contract double %87, %88, !dbg !1614
  %add98 = fadd contract double %mul96, %mul97, !dbg !1615
  %89 = load %struct.dcomplex*, %struct.dcomplex** %gty2.addr, align 8, !dbg !1616
  %90 = load i32, i32* %j, align 4, !dbg !1617
  %91 = load i32, i32* %i22, align 4, !dbg !1618
  %92 = load i32, i32* %k1, align 4, !dbg !1619
  %add99 = add nsw i32 %91, %92, !dbg !1620
  %mul100 = mul nsw i32 %add99, 256, !dbg !1621
  %add101 = add nsw i32 %90, %mul100, !dbg !1622
  %93 = load i32, i32* %k, align 4, !dbg !1623
  %mul102 = mul nsw i32 %93, 256, !dbg !1624
  %mul103 = mul nsw i32 %mul102, 256, !dbg !1625
  %add104 = add nsw i32 %add101, %mul103, !dbg !1626
  %idxprom105 = sext i32 %add104 to i64, !dbg !1616
  %arrayidx106 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %89, i64 %idxprom105, !dbg !1616
  %imag107 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx106, i32 0, i32 1, !dbg !1627
  store double %add98, double* %imag107, align 8, !dbg !1628
  br label %for.inc, !dbg !1629

for.inc:                                          ; preds = %for.body16
  %94 = load i32, i32* %k1, align 4, !dbg !1630
  %inc = add nsw i32 %94, 1, !dbg !1630
  store i32 %inc, i32* %k1, align 4, !dbg !1630
  br label %for.cond13, !dbg !1631, !llvm.loop !1632

for.end:                                          ; preds = %for.cond13
  br label %for.inc108, !dbg !1634

for.inc108:                                       ; preds = %for.end
  %95 = load i32, i32* %i1, align 4, !dbg !1635
  %inc109 = add nsw i32 %95, 1, !dbg !1635
  store i32 %inc109, i32* %i1, align 4, !dbg !1635
  br label %for.cond9, !dbg !1636, !llvm.loop !1637

for.end110:                                       ; preds = %for.cond9
  %96 = load i32, i32* %l, align 4, !dbg !1639
  %97 = load i32, i32* %logd1, align 4, !dbg !1641
  %cmp111 = icmp eq i32 %96, %97, !dbg !1642
  br i1 %cmp111, label %if.then112, label %if.else, !dbg !1643

if.then112:                                       ; preds = %for.end110
  store i32 0, i32* %j1, align 4, !dbg !1644
  br label %for.cond113, !dbg !1647

for.cond113:                                      ; preds = %for.inc148, %if.then112
  %98 = load i32, i32* %j1, align 4, !dbg !1648
  %cmp114 = icmp slt i32 %98, 256, !dbg !1650
  br i1 %cmp114, label %for.body115, label %for.end150, !dbg !1651

for.body115:                                      ; preds = %for.cond113
  %99 = load %struct.dcomplex*, %struct.dcomplex** %gty2.addr, align 8, !dbg !1652
  %100 = load i32, i32* %j, align 4, !dbg !1654
  %101 = load i32, i32* %j1, align 4, !dbg !1655
  %mul116 = mul nsw i32 %101, 256, !dbg !1656
  %add117 = add nsw i32 %100, %mul116, !dbg !1657
  %102 = load i32, i32* %k, align 4, !dbg !1658
  %mul118 = mul nsw i32 %102, 256, !dbg !1659
  %mul119 = mul nsw i32 %mul118, 256, !dbg !1660
  %add120 = add nsw i32 %add117, %mul119, !dbg !1661
  %idxprom121 = sext i32 %add120 to i64, !dbg !1652
  %arrayidx122 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %99, i64 %idxprom121, !dbg !1652
  %real123 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx122, i32 0, i32 0, !dbg !1662
  %103 = load double, double* %real123, align 8, !dbg !1662
  %104 = load %struct.dcomplex*, %struct.dcomplex** %gty1.addr, align 8, !dbg !1663
  %105 = load i32, i32* %j, align 4, !dbg !1664
  %106 = load i32, i32* %j1, align 4, !dbg !1665
  %mul124 = mul nsw i32 %106, 256, !dbg !1666
  %add125 = add nsw i32 %105, %mul124, !dbg !1667
  %107 = load i32, i32* %k, align 4, !dbg !1668
  %mul126 = mul nsw i32 %107, 256, !dbg !1669
  %mul127 = mul nsw i32 %mul126, 256, !dbg !1670
  %add128 = add nsw i32 %add125, %mul127, !dbg !1671
  %idxprom129 = sext i32 %add128 to i64, !dbg !1663
  %arrayidx130 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %104, i64 %idxprom129, !dbg !1663
  %real131 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx130, i32 0, i32 0, !dbg !1672
  store double %103, double* %real131, align 8, !dbg !1673
  %108 = load %struct.dcomplex*, %struct.dcomplex** %gty2.addr, align 8, !dbg !1674
  %109 = load i32, i32* %j, align 4, !dbg !1675
  %110 = load i32, i32* %j1, align 4, !dbg !1676
  %mul132 = mul nsw i32 %110, 256, !dbg !1677
  %add133 = add nsw i32 %109, %mul132, !dbg !1678
  %111 = load i32, i32* %k, align 4, !dbg !1679
  %mul134 = mul nsw i32 %111, 256, !dbg !1680
  %mul135 = mul nsw i32 %mul134, 256, !dbg !1681
  %add136 = add nsw i32 %add133, %mul135, !dbg !1682
  %idxprom137 = sext i32 %add136 to i64, !dbg !1674
  %arrayidx138 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %108, i64 %idxprom137, !dbg !1674
  %imag139 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx138, i32 0, i32 1, !dbg !1683
  %112 = load double, double* %imag139, align 8, !dbg !1683
  %113 = load %struct.dcomplex*, %struct.dcomplex** %gty1.addr, align 8, !dbg !1684
  %114 = load i32, i32* %j, align 4, !dbg !1685
  %115 = load i32, i32* %j1, align 4, !dbg !1686
  %mul140 = mul nsw i32 %115, 256, !dbg !1687
  %add141 = add nsw i32 %114, %mul140, !dbg !1688
  %116 = load i32, i32* %k, align 4, !dbg !1689
  %mul142 = mul nsw i32 %116, 256, !dbg !1690
  %mul143 = mul nsw i32 %mul142, 256, !dbg !1691
  %add144 = add nsw i32 %add141, %mul143, !dbg !1692
  %idxprom145 = sext i32 %add144 to i64, !dbg !1684
  %arrayidx146 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %113, i64 %idxprom145, !dbg !1684
  %imag147 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx146, i32 0, i32 1, !dbg !1693
  store double %112, double* %imag147, align 8, !dbg !1694
  br label %for.inc148, !dbg !1695

for.inc148:                                       ; preds = %for.body115
  %117 = load i32, i32* %j1, align 4, !dbg !1696
  %inc149 = add nsw i32 %117, 1, !dbg !1696
  store i32 %inc149, i32* %j1, align 4, !dbg !1696
  br label %for.cond113, !dbg !1697, !llvm.loop !1698

for.end150:                                       ; preds = %for.cond113
  br label %if.end268, !dbg !1700

if.else:                                          ; preds = %for.end110
  store i32 128, i32* %n1, align 4, !dbg !1701
  %118 = load i32, i32* %l, align 4, !dbg !1703
  %add151 = add nsw i32 %118, 1, !dbg !1704
  %sub152 = sub nsw i32 %add151, 1, !dbg !1705
  %shl153 = shl i32 1, %sub152, !dbg !1706
  store i32 %shl153, i32* %lk, align 4, !dbg !1707
  %119 = load i32, i32* %logd1, align 4, !dbg !1708
  %120 = load i32, i32* %l, align 4, !dbg !1709
  %add154 = add nsw i32 %120, 1, !dbg !1710
  %sub155 = sub nsw i32 %119, %add154, !dbg !1711
  %shl156 = shl i32 1, %sub155, !dbg !1712
  store i32 %shl156, i32* %li, align 4, !dbg !1713
  %121 = load i32, i32* %lk, align 4, !dbg !1714
  %mul157 = mul nsw i32 2, %121, !dbg !1715
  store i32 %mul157, i32* %lj, align 4, !dbg !1716
  %122 = load i32, i32* %li, align 4, !dbg !1717
  store i32 %122, i32* %ku, align 4, !dbg !1718
  store i32 0, i32* %i1, align 4, !dbg !1719
  br label %for.cond158, !dbg !1721

for.cond158:                                      ; preds = %for.inc265, %if.else
  %123 = load i32, i32* %i1, align 4, !dbg !1722
  %124 = load i32, i32* %li, align 4, !dbg !1724
  %sub159 = sub nsw i32 %124, 1, !dbg !1725
  %cmp160 = icmp sle i32 %123, %sub159, !dbg !1726
  br i1 %cmp160, label %for.body161, label %for.end267, !dbg !1727

for.body161:                                      ; preds = %for.cond158
  store i32 0, i32* %k1, align 4, !dbg !1728
  br label %for.cond162, !dbg !1731

for.cond162:                                      ; preds = %for.inc262, %for.body161
  %125 = load i32, i32* %k1, align 4, !dbg !1732
  %126 = load i32, i32* %lk, align 4, !dbg !1734
  %sub163 = sub nsw i32 %126, 1, !dbg !1735
  %cmp164 = icmp sle i32 %125, %sub163, !dbg !1736
  br i1 %cmp164, label %for.body165, label %for.end264, !dbg !1737

for.body165:                                      ; preds = %for.cond162
  %127 = load i32, i32* %i1, align 4, !dbg !1738
  %128 = load i32, i32* %lk, align 4, !dbg !1740
  %mul166 = mul nsw i32 %127, %128, !dbg !1741
  store i32 %mul166, i32* %i11, align 4, !dbg !1742
  %129 = load i32, i32* %i11, align 4, !dbg !1743
  %130 = load i32, i32* %n1, align 4, !dbg !1744
  %add167 = add nsw i32 %129, %130, !dbg !1745
  store i32 %add167, i32* %i12, align 4, !dbg !1746
  %131 = load i32, i32* %i1, align 4, !dbg !1747
  %132 = load i32, i32* %lj, align 4, !dbg !1748
  %mul168 = mul nsw i32 %131, %132, !dbg !1749
  store i32 %mul168, i32* %i21, align 4, !dbg !1750
  %133 = load i32, i32* %i21, align 4, !dbg !1751
  %134 = load i32, i32* %lk, align 4, !dbg !1752
  %add169 = add nsw i32 %133, %134, !dbg !1753
  store i32 %add169, i32* %i22, align 4, !dbg !1754
  %135 = load %struct.dcomplex*, %struct.dcomplex** %u_device.addr, align 8, !dbg !1755
  %136 = load i32, i32* %ku, align 4, !dbg !1756
  %137 = load i32, i32* %i1, align 4, !dbg !1757
  %add170 = add nsw i32 %136, %137, !dbg !1758
  %idxprom171 = sext i32 %add170 to i64, !dbg !1755
  %arrayidx172 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %135, i64 %idxprom171, !dbg !1755
  %real173 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx172, i32 0, i32 0, !dbg !1759
  %138 = load double, double* %real173, align 8, !dbg !1759
  store double %138, double* %uu2_real, align 8, !dbg !1760
  %139 = load i32, i32* %is.addr, align 4, !dbg !1761
  %conv174 = sitofp i32 %139 to double, !dbg !1761
  %140 = load %struct.dcomplex*, %struct.dcomplex** %u_device.addr, align 8, !dbg !1762
  %141 = load i32, i32* %ku, align 4, !dbg !1763
  %142 = load i32, i32* %i1, align 4, !dbg !1764
  %add175 = add nsw i32 %141, %142, !dbg !1765
  %idxprom176 = sext i32 %add175 to i64, !dbg !1762
  %arrayidx177 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %140, i64 %idxprom176, !dbg !1762
  %imag178 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx177, i32 0, i32 1, !dbg !1766
  %143 = load double, double* %imag178, align 8, !dbg !1766
  %mul179 = fmul contract double %conv174, %143, !dbg !1767
  store double %mul179, double* %uu2_imag, align 8, !dbg !1768
  %144 = load %struct.dcomplex*, %struct.dcomplex** %gty2.addr, align 8, !dbg !1769
  %145 = load i32, i32* %j, align 4, !dbg !1770
  %146 = load i32, i32* %i11, align 4, !dbg !1771
  %147 = load i32, i32* %k1, align 4, !dbg !1772
  %add180 = add nsw i32 %146, %147, !dbg !1773
  %mul181 = mul nsw i32 %add180, 256, !dbg !1774
  %add182 = add nsw i32 %145, %mul181, !dbg !1775
  %148 = load i32, i32* %k, align 4, !dbg !1776
  %mul183 = mul nsw i32 %148, 256, !dbg !1777
  %mul184 = mul nsw i32 %mul183, 256, !dbg !1778
  %add185 = add nsw i32 %add182, %mul184, !dbg !1779
  %idxprom186 = sext i32 %add185 to i64, !dbg !1769
  %arrayidx187 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %144, i64 %idxprom186, !dbg !1769
  %real188 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx187, i32 0, i32 0, !dbg !1780
  %149 = load double, double* %real188, align 8, !dbg !1780
  store double %149, double* %x12_real, align 8, !dbg !1781
  %150 = load %struct.dcomplex*, %struct.dcomplex** %gty2.addr, align 8, !dbg !1782
  %151 = load i32, i32* %j, align 4, !dbg !1783
  %152 = load i32, i32* %i11, align 4, !dbg !1784
  %153 = load i32, i32* %k1, align 4, !dbg !1785
  %add189 = add nsw i32 %152, %153, !dbg !1786
  %mul190 = mul nsw i32 %add189, 256, !dbg !1787
  %add191 = add nsw i32 %151, %mul190, !dbg !1788
  %154 = load i32, i32* %k, align 4, !dbg !1789
  %mul192 = mul nsw i32 %154, 256, !dbg !1790
  %mul193 = mul nsw i32 %mul192, 256, !dbg !1791
  %add194 = add nsw i32 %add191, %mul193, !dbg !1792
  %idxprom195 = sext i32 %add194 to i64, !dbg !1782
  %arrayidx196 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %150, i64 %idxprom195, !dbg !1782
  %imag197 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx196, i32 0, i32 1, !dbg !1793
  %155 = load double, double* %imag197, align 8, !dbg !1793
  store double %155, double* %x12_imag, align 8, !dbg !1794
  %156 = load %struct.dcomplex*, %struct.dcomplex** %gty2.addr, align 8, !dbg !1795
  %157 = load i32, i32* %j, align 4, !dbg !1796
  %158 = load i32, i32* %i12, align 4, !dbg !1797
  %159 = load i32, i32* %k1, align 4, !dbg !1798
  %add198 = add nsw i32 %158, %159, !dbg !1799
  %mul199 = mul nsw i32 %add198, 256, !dbg !1800
  %add200 = add nsw i32 %157, %mul199, !dbg !1801
  %160 = load i32, i32* %k, align 4, !dbg !1802
  %mul201 = mul nsw i32 %160, 256, !dbg !1803
  %mul202 = mul nsw i32 %mul201, 256, !dbg !1804
  %add203 = add nsw i32 %add200, %mul202, !dbg !1805
  %idxprom204 = sext i32 %add203 to i64, !dbg !1795
  %arrayidx205 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %156, i64 %idxprom204, !dbg !1795
  %real206 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx205, i32 0, i32 0, !dbg !1806
  %161 = load double, double* %real206, align 8, !dbg !1806
  store double %161, double* %x22_real, align 8, !dbg !1807
  %162 = load %struct.dcomplex*, %struct.dcomplex** %gty2.addr, align 8, !dbg !1808
  %163 = load i32, i32* %j, align 4, !dbg !1809
  %164 = load i32, i32* %i12, align 4, !dbg !1810
  %165 = load i32, i32* %k1, align 4, !dbg !1811
  %add207 = add nsw i32 %164, %165, !dbg !1812
  %mul208 = mul nsw i32 %add207, 256, !dbg !1813
  %add209 = add nsw i32 %163, %mul208, !dbg !1814
  %166 = load i32, i32* %k, align 4, !dbg !1815
  %mul210 = mul nsw i32 %166, 256, !dbg !1816
  %mul211 = mul nsw i32 %mul210, 256, !dbg !1817
  %add212 = add nsw i32 %add209, %mul211, !dbg !1818
  %idxprom213 = sext i32 %add212 to i64, !dbg !1808
  %arrayidx214 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %162, i64 %idxprom213, !dbg !1808
  %imag215 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx214, i32 0, i32 1, !dbg !1819
  %167 = load double, double* %imag215, align 8, !dbg !1819
  store double %167, double* %x22_imag, align 8, !dbg !1820
  %168 = load double, double* %x12_real, align 8, !dbg !1821
  %169 = load double, double* %x22_real, align 8, !dbg !1822
  %add216 = fadd contract double %168, %169, !dbg !1823
  %170 = load %struct.dcomplex*, %struct.dcomplex** %gty1.addr, align 8, !dbg !1824
  %171 = load i32, i32* %j, align 4, !dbg !1825
  %172 = load i32, i32* %i21, align 4, !dbg !1826
  %173 = load i32, i32* %k1, align 4, !dbg !1827
  %add217 = add nsw i32 %172, %173, !dbg !1828
  %mul218 = mul nsw i32 %add217, 256, !dbg !1829
  %add219 = add nsw i32 %171, %mul218, !dbg !1830
  %174 = load i32, i32* %k, align 4, !dbg !1831
  %mul220 = mul nsw i32 %174, 256, !dbg !1832
  %mul221 = mul nsw i32 %mul220, 256, !dbg !1833
  %add222 = add nsw i32 %add219, %mul221, !dbg !1834
  %idxprom223 = sext i32 %add222 to i64, !dbg !1824
  %arrayidx224 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %170, i64 %idxprom223, !dbg !1824
  %real225 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx224, i32 0, i32 0, !dbg !1835
  store double %add216, double* %real225, align 8, !dbg !1836
  %175 = load double, double* %x12_imag, align 8, !dbg !1837
  %176 = load double, double* %x22_imag, align 8, !dbg !1838
  %add226 = fadd contract double %175, %176, !dbg !1839
  %177 = load %struct.dcomplex*, %struct.dcomplex** %gty1.addr, align 8, !dbg !1840
  %178 = load i32, i32* %j, align 4, !dbg !1841
  %179 = load i32, i32* %i21, align 4, !dbg !1842
  %180 = load i32, i32* %k1, align 4, !dbg !1843
  %add227 = add nsw i32 %179, %180, !dbg !1844
  %mul228 = mul nsw i32 %add227, 256, !dbg !1845
  %add229 = add nsw i32 %178, %mul228, !dbg !1846
  %181 = load i32, i32* %k, align 4, !dbg !1847
  %mul230 = mul nsw i32 %181, 256, !dbg !1848
  %mul231 = mul nsw i32 %mul230, 256, !dbg !1849
  %add232 = add nsw i32 %add229, %mul231, !dbg !1850
  %idxprom233 = sext i32 %add232 to i64, !dbg !1840
  %arrayidx234 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %177, i64 %idxprom233, !dbg !1840
  %imag235 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx234, i32 0, i32 1, !dbg !1851
  store double %add226, double* %imag235, align 8, !dbg !1852
  %182 = load double, double* %x12_real, align 8, !dbg !1853
  %183 = load double, double* %x22_real, align 8, !dbg !1854
  %sub236 = fsub contract double %182, %183, !dbg !1855
  store double %sub236, double* %temp2_real, align 8, !dbg !1856
  %184 = load double, double* %x12_imag, align 8, !dbg !1857
  %185 = load double, double* %x22_imag, align 8, !dbg !1858
  %sub237 = fsub contract double %184, %185, !dbg !1859
  store double %sub237, double* %temp2_imag, align 8, !dbg !1860
  %186 = load double, double* %uu2_real, align 8, !dbg !1861
  %187 = load double, double* %temp2_real, align 8, !dbg !1862
  %mul238 = fmul contract double %186, %187, !dbg !1863
  %188 = load double, double* %uu2_imag, align 8, !dbg !1864
  %189 = load double, double* %temp2_imag, align 8, !dbg !1865
  %mul239 = fmul contract double %188, %189, !dbg !1866
  %sub240 = fsub contract double %mul238, %mul239, !dbg !1867
  %190 = load %struct.dcomplex*, %struct.dcomplex** %gty1.addr, align 8, !dbg !1868
  %191 = load i32, i32* %j, align 4, !dbg !1869
  %192 = load i32, i32* %i22, align 4, !dbg !1870
  %193 = load i32, i32* %k1, align 4, !dbg !1871
  %add241 = add nsw i32 %192, %193, !dbg !1872
  %mul242 = mul nsw i32 %add241, 256, !dbg !1873
  %add243 = add nsw i32 %191, %mul242, !dbg !1874
  %194 = load i32, i32* %k, align 4, !dbg !1875
  %mul244 = mul nsw i32 %194, 256, !dbg !1876
  %mul245 = mul nsw i32 %mul244, 256, !dbg !1877
  %add246 = add nsw i32 %add243, %mul245, !dbg !1878
  %idxprom247 = sext i32 %add246 to i64, !dbg !1868
  %arrayidx248 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %190, i64 %idxprom247, !dbg !1868
  %real249 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx248, i32 0, i32 0, !dbg !1879
  store double %sub240, double* %real249, align 8, !dbg !1880
  %195 = load double, double* %uu2_real, align 8, !dbg !1881
  %196 = load double, double* %temp2_imag, align 8, !dbg !1882
  %mul250 = fmul contract double %195, %196, !dbg !1883
  %197 = load double, double* %uu2_imag, align 8, !dbg !1884
  %198 = load double, double* %temp2_real, align 8, !dbg !1885
  %mul251 = fmul contract double %197, %198, !dbg !1886
  %add252 = fadd contract double %mul250, %mul251, !dbg !1887
  %199 = load %struct.dcomplex*, %struct.dcomplex** %gty1.addr, align 8, !dbg !1888
  %200 = load i32, i32* %j, align 4, !dbg !1889
  %201 = load i32, i32* %i22, align 4, !dbg !1890
  %202 = load i32, i32* %k1, align 4, !dbg !1891
  %add253 = add nsw i32 %201, %202, !dbg !1892
  %mul254 = mul nsw i32 %add253, 256, !dbg !1893
  %add255 = add nsw i32 %200, %mul254, !dbg !1894
  %203 = load i32, i32* %k, align 4, !dbg !1895
  %mul256 = mul nsw i32 %203, 256, !dbg !1896
  %mul257 = mul nsw i32 %mul256, 256, !dbg !1897
  %add258 = add nsw i32 %add255, %mul257, !dbg !1898
  %idxprom259 = sext i32 %add258 to i64, !dbg !1888
  %arrayidx260 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %199, i64 %idxprom259, !dbg !1888
  %imag261 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx260, i32 0, i32 1, !dbg !1899
  store double %add252, double* %imag261, align 8, !dbg !1900
  br label %for.inc262, !dbg !1901

for.inc262:                                       ; preds = %for.body165
  %204 = load i32, i32* %k1, align 4, !dbg !1902
  %inc263 = add nsw i32 %204, 1, !dbg !1902
  store i32 %inc263, i32* %k1, align 4, !dbg !1902
  br label %for.cond162, !dbg !1903, !llvm.loop !1904

for.end264:                                       ; preds = %for.cond162
  br label %for.inc265, !dbg !1906

for.inc265:                                       ; preds = %for.end264
  %205 = load i32, i32* %i1, align 4, !dbg !1907
  %inc266 = add nsw i32 %205, 1, !dbg !1907
  store i32 %inc266, i32* %i1, align 4, !dbg !1907
  br label %for.cond158, !dbg !1908, !llvm.loop !1909

for.end267:                                       ; preds = %for.cond158
  br label %if.end268

if.end268:                                        ; preds = %for.end267, %for.end150
  br label %for.inc269, !dbg !1911

for.inc269:                                       ; preds = %if.end268
  %206 = load i32, i32* %l, align 4, !dbg !1912
  %add270 = add nsw i32 %206, 2, !dbg !1912
  store i32 %add270, i32* %l, align 4, !dbg !1912
  br label %for.cond, !dbg !1913, !llvm.loop !1914

for.end271:                                       ; preds = %for.cond, %if.then
  ret void, !dbg !1916
}

; Function Attrs: convergent noinline nounwind
define dso_local i32 @_Z12ilog2_devicei(i32 %n) #0 !dbg !1917 {
entry:
  %retval = alloca i32, align 4
  %n.addr = alloca i32, align 4
  %nn = alloca i32, align 4
  %lg = alloca i32, align 4
  store i32 %n, i32* %n.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %n.addr, metadata !1918, metadata !DIExpression()), !dbg !1919
  call void @llvm.dbg.declare(metadata i32* %nn, metadata !1920, metadata !DIExpression()), !dbg !1921
  call void @llvm.dbg.declare(metadata i32* %lg, metadata !1922, metadata !DIExpression()), !dbg !1923
  %0 = load i32, i32* %n.addr, align 4, !dbg !1924
  %cmp = icmp eq i32 %0, 1, !dbg !1926
  br i1 %cmp, label %if.then, label %if.end, !dbg !1927

if.then:                                          ; preds = %entry
  store i32 0, i32* %retval, align 4, !dbg !1928
  br label %return, !dbg !1928

if.end:                                           ; preds = %entry
  store i32 1, i32* %lg, align 4, !dbg !1930
  store i32 2, i32* %nn, align 4, !dbg !1931
  br label %while.cond, !dbg !1932

while.cond:                                       ; preds = %while.body, %if.end
  %1 = load i32, i32* %nn, align 4, !dbg !1933
  %2 = load i32, i32* %n.addr, align 4, !dbg !1934
  %cmp1 = icmp slt i32 %1, %2, !dbg !1935
  br i1 %cmp1, label %while.body, label %while.end, !dbg !1932

while.body:                                       ; preds = %while.cond
  %3 = load i32, i32* %nn, align 4, !dbg !1936
  %shl = shl i32 %3, 1, !dbg !1938
  store i32 %shl, i32* %nn, align 4, !dbg !1939
  %4 = load i32, i32* %lg, align 4, !dbg !1940
  %inc = add nsw i32 %4, 1, !dbg !1940
  store i32 %inc, i32* %lg, align 4, !dbg !1940
  br label %while.cond, !dbg !1932, !llvm.loop !1941

while.end:                                        ; preds = %while.cond
  %5 = load i32, i32* %lg, align 4, !dbg !1943
  store i32 %5, i32* %retval, align 4, !dbg !1944
  br label %return, !dbg !1944

return:                                           ; preds = %while.end, %if.then
  %6 = load i32, i32* %retval, align 4, !dbg !1945
  ret i32 %6, !dbg !1945
}

; Function Attrs: convergent noinline nounwind
define dso_local void @_Z19cffts1_gpu_kernel_3P8dcomplexS0_(%struct.dcomplex* %x_out, %struct.dcomplex* %y0) #0 !dbg !1946 {
entry:
  %x_out.addr = alloca %struct.dcomplex*, align 8
  %y0.addr = alloca %struct.dcomplex*, align 8
  %x_y_z = alloca i32, align 4
  %x = alloca i32, align 4
  %y = alloca i32, align 4
  %z = alloca i32, align 4
  store %struct.dcomplex* %x_out, %struct.dcomplex** %x_out.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %x_out.addr, metadata !1947, metadata !DIExpression()), !dbg !1948
  store %struct.dcomplex* %y0, %struct.dcomplex** %y0.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %y0.addr, metadata !1949, metadata !DIExpression()), !dbg !1950
  call void @llvm.dbg.declare(metadata i32* %x_y_z, metadata !1951, metadata !DIExpression()), !dbg !1952
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #11, !dbg !1953, !range !1198
  %1 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #11, !dbg !1955, !range !1243
  %mul = mul i32 %0, %1, !dbg !1957
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #11, !dbg !1958, !range !1273
  %add = add i32 %mul, %2, !dbg !1960
  store i32 %add, i32* %x_y_z, align 4, !dbg !1952
  %3 = load i32, i32* %x_y_z, align 4, !dbg !1961
  %cmp = icmp sge i32 %3, 8388608, !dbg !1963
  br i1 %cmp, label %if.then, label %if.end, !dbg !1964

if.then:                                          ; preds = %entry
  br label %return, !dbg !1965

if.end:                                           ; preds = %entry
  call void @llvm.dbg.declare(metadata i32* %x, metadata !1967, metadata !DIExpression()), !dbg !1968
  %4 = load i32, i32* %x_y_z, align 4, !dbg !1969
  %rem = srem i32 %4, 256, !dbg !1970
  store i32 %rem, i32* %x, align 4, !dbg !1968
  call void @llvm.dbg.declare(metadata i32* %y, metadata !1971, metadata !DIExpression()), !dbg !1972
  %5 = load i32, i32* %x_y_z, align 4, !dbg !1973
  %div = sdiv i32 %5, 256, !dbg !1974
  %rem3 = srem i32 %div, 256, !dbg !1975
  store i32 %rem3, i32* %y, align 4, !dbg !1972
  call void @llvm.dbg.declare(metadata i32* %z, metadata !1976, metadata !DIExpression()), !dbg !1977
  %6 = load i32, i32* %x_y_z, align 4, !dbg !1978
  %div4 = sdiv i32 %6, 65536, !dbg !1979
  store i32 %div4, i32* %z, align 4, !dbg !1977
  %7 = load %struct.dcomplex*, %struct.dcomplex** %y0.addr, align 8, !dbg !1980
  %8 = load i32, i32* %y, align 4, !dbg !1981
  %9 = load i32, i32* %x, align 4, !dbg !1982
  %mul5 = mul nsw i32 %9, 256, !dbg !1983
  %add6 = add nsw i32 %8, %mul5, !dbg !1984
  %10 = load i32, i32* %z, align 4, !dbg !1985
  %mul7 = mul nsw i32 %10, 256, !dbg !1986
  %mul8 = mul nsw i32 %mul7, 256, !dbg !1987
  %add9 = add nsw i32 %add6, %mul8, !dbg !1988
  %idxprom = sext i32 %add9 to i64, !dbg !1980
  %arrayidx = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %7, i64 %idxprom, !dbg !1980
  %real = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx, i32 0, i32 0, !dbg !1989
  %11 = load double, double* %real, align 8, !dbg !1989
  %12 = load %struct.dcomplex*, %struct.dcomplex** %x_out.addr, align 8, !dbg !1990
  %13 = load i32, i32* %x_y_z, align 4, !dbg !1991
  %idxprom10 = sext i32 %13 to i64, !dbg !1990
  %arrayidx11 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %12, i64 %idxprom10, !dbg !1990
  %real12 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx11, i32 0, i32 0, !dbg !1992
  store double %11, double* %real12, align 8, !dbg !1993
  %14 = load %struct.dcomplex*, %struct.dcomplex** %y0.addr, align 8, !dbg !1994
  %15 = load i32, i32* %y, align 4, !dbg !1995
  %16 = load i32, i32* %x, align 4, !dbg !1996
  %mul13 = mul nsw i32 %16, 256, !dbg !1997
  %add14 = add nsw i32 %15, %mul13, !dbg !1998
  %17 = load i32, i32* %z, align 4, !dbg !1999
  %mul15 = mul nsw i32 %17, 256, !dbg !2000
  %mul16 = mul nsw i32 %mul15, 256, !dbg !2001
  %add17 = add nsw i32 %add14, %mul16, !dbg !2002
  %idxprom18 = sext i32 %add17 to i64, !dbg !1994
  %arrayidx19 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %14, i64 %idxprom18, !dbg !1994
  %imag = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx19, i32 0, i32 1, !dbg !2003
  %18 = load double, double* %imag, align 8, !dbg !2003
  %19 = load %struct.dcomplex*, %struct.dcomplex** %x_out.addr, align 8, !dbg !2004
  %20 = load i32, i32* %x_y_z, align 4, !dbg !2005
  %idxprom20 = sext i32 %20 to i64, !dbg !2004
  %arrayidx21 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %19, i64 %idxprom20, !dbg !2004
  %imag22 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx21, i32 0, i32 1, !dbg !2006
  store double %18, double* %imag22, align 8, !dbg !2007
  br label %return, !dbg !2008

return:                                           ; preds = %if.end, %if.then
  ret void, !dbg !2008
}

; Function Attrs: convergent noinline nounwind
define dso_local void @_Z19cffts2_gpu_kernel_1P8dcomplexS0_(%struct.dcomplex* %x_in, %struct.dcomplex* %y0) #0 !dbg !2009 {
entry:
  %x_in.addr = alloca %struct.dcomplex*, align 8
  %y0.addr = alloca %struct.dcomplex*, align 8
  %x_y_z = alloca i32, align 4
  store %struct.dcomplex* %x_in, %struct.dcomplex** %x_in.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %x_in.addr, metadata !2010, metadata !DIExpression()), !dbg !2011
  store %struct.dcomplex* %y0, %struct.dcomplex** %y0.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %y0.addr, metadata !2012, metadata !DIExpression()), !dbg !2013
  call void @llvm.dbg.declare(metadata i32* %x_y_z, metadata !2014, metadata !DIExpression()), !dbg !2015
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #11, !dbg !2016, !range !1198
  %1 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #11, !dbg !2018, !range !1243
  %mul = mul i32 %0, %1, !dbg !2020
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #11, !dbg !2021, !range !1273
  %add = add i32 %mul, %2, !dbg !2023
  store i32 %add, i32* %x_y_z, align 4, !dbg !2015
  %3 = load i32, i32* %x_y_z, align 4, !dbg !2024
  %cmp = icmp sge i32 %3, 8388608, !dbg !2026
  br i1 %cmp, label %if.then, label %if.end, !dbg !2027

if.then:                                          ; preds = %entry
  br label %return, !dbg !2028

if.end:                                           ; preds = %entry
  %4 = load %struct.dcomplex*, %struct.dcomplex** %x_in.addr, align 8, !dbg !2030
  %5 = load i32, i32* %x_y_z, align 4, !dbg !2031
  %idxprom = sext i32 %5 to i64, !dbg !2030
  %arrayidx = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %4, i64 %idxprom, !dbg !2030
  %real = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx, i32 0, i32 0, !dbg !2032
  %6 = load double, double* %real, align 8, !dbg !2032
  %7 = load %struct.dcomplex*, %struct.dcomplex** %y0.addr, align 8, !dbg !2033
  %8 = load i32, i32* %x_y_z, align 4, !dbg !2034
  %idxprom3 = sext i32 %8 to i64, !dbg !2033
  %arrayidx4 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %7, i64 %idxprom3, !dbg !2033
  %real5 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx4, i32 0, i32 0, !dbg !2035
  store double %6, double* %real5, align 8, !dbg !2036
  %9 = load %struct.dcomplex*, %struct.dcomplex** %x_in.addr, align 8, !dbg !2037
  %10 = load i32, i32* %x_y_z, align 4, !dbg !2038
  %idxprom6 = sext i32 %10 to i64, !dbg !2037
  %arrayidx7 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %9, i64 %idxprom6, !dbg !2037
  %imag = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx7, i32 0, i32 1, !dbg !2039
  %11 = load double, double* %imag, align 8, !dbg !2039
  %12 = load %struct.dcomplex*, %struct.dcomplex** %y0.addr, align 8, !dbg !2040
  %13 = load i32, i32* %x_y_z, align 4, !dbg !2041
  %idxprom8 = sext i32 %13 to i64, !dbg !2040
  %arrayidx9 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %12, i64 %idxprom8, !dbg !2040
  %imag10 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx9, i32 0, i32 1, !dbg !2042
  store double %11, double* %imag10, align 8, !dbg !2043
  br label %return, !dbg !2044

return:                                           ; preds = %if.end, %if.then
  ret void, !dbg !2044
}

; Function Attrs: convergent noinline nounwind
define dso_local void @_Z19cffts2_gpu_kernel_2iP8dcomplexS0_S0_(i32 %is, %struct.dcomplex* %gty1, %struct.dcomplex* %gty2, %struct.dcomplex* %u_device) #0 !dbg !2045 {
entry:
  %is.addr = alloca i32, align 4
  %gty1.addr = alloca %struct.dcomplex*, align 8
  %gty2.addr = alloca %struct.dcomplex*, align 8
  %u_device.addr = alloca %struct.dcomplex*, align 8
  %x_z = alloca i32, align 4
  %i = alloca i32, align 4
  %k = alloca i32, align 4
  %l = alloca i32, align 4
  %j1 = alloca i32, align 4
  %i1 = alloca i32, align 4
  %k1 = alloca i32, align 4
  %n1 = alloca i32, align 4
  %li = alloca i32, align 4
  %lj = alloca i32, align 4
  %lk = alloca i32, align 4
  %ku = alloca i32, align 4
  %i11 = alloca i32, align 4
  %i12 = alloca i32, align 4
  %i21 = alloca i32, align 4
  %i22 = alloca i32, align 4
  %logd2 = alloca i32, align 4
  %uu1_real = alloca double, align 8
  %x11_real = alloca double, align 8
  %x21_real = alloca double, align 8
  %uu1_imag = alloca double, align 8
  %x11_imag = alloca double, align 8
  %x21_imag = alloca double, align 8
  %uu2_real = alloca double, align 8
  %x12_real = alloca double, align 8
  %x22_real = alloca double, align 8
  %uu2_imag = alloca double, align 8
  %x12_imag = alloca double, align 8
  %x22_imag = alloca double, align 8
  %temp_real = alloca double, align 8
  %temp2_real = alloca double, align 8
  %temp_imag = alloca double, align 8
  %temp2_imag = alloca double, align 8
  store i32 %is, i32* %is.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %is.addr, metadata !2046, metadata !DIExpression()), !dbg !2047
  store %struct.dcomplex* %gty1, %struct.dcomplex** %gty1.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %gty1.addr, metadata !2048, metadata !DIExpression()), !dbg !2049
  store %struct.dcomplex* %gty2, %struct.dcomplex** %gty2.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %gty2.addr, metadata !2050, metadata !DIExpression()), !dbg !2051
  store %struct.dcomplex* %u_device, %struct.dcomplex** %u_device.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %u_device.addr, metadata !2052, metadata !DIExpression()), !dbg !2053
  call void @llvm.dbg.declare(metadata i32* %x_z, metadata !2054, metadata !DIExpression()), !dbg !2055
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #11, !dbg !2056, !range !1198
  %1 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #11, !dbg !2058, !range !1243
  %mul = mul i32 %0, %1, !dbg !2060
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #11, !dbg !2061, !range !1273
  %add = add i32 %mul, %2, !dbg !2063
  store i32 %add, i32* %x_z, align 4, !dbg !2055
  %3 = load i32, i32* %x_z, align 4, !dbg !2064
  %cmp = icmp sge i32 %3, 32768, !dbg !2066
  br i1 %cmp, label %if.then, label %if.end, !dbg !2067

if.then:                                          ; preds = %entry
  br label %for.end271, !dbg !2068

if.end:                                           ; preds = %entry
  call void @llvm.dbg.declare(metadata i32* %i, metadata !2070, metadata !DIExpression()), !dbg !2071
  call void @llvm.dbg.declare(metadata i32* %k, metadata !2072, metadata !DIExpression()), !dbg !2073
  call void @llvm.dbg.declare(metadata i32* %l, metadata !2074, metadata !DIExpression()), !dbg !2075
  call void @llvm.dbg.declare(metadata i32* %j1, metadata !2076, metadata !DIExpression()), !dbg !2077
  call void @llvm.dbg.declare(metadata i32* %i1, metadata !2078, metadata !DIExpression()), !dbg !2079
  call void @llvm.dbg.declare(metadata i32* %k1, metadata !2080, metadata !DIExpression()), !dbg !2081
  call void @llvm.dbg.declare(metadata i32* %n1, metadata !2082, metadata !DIExpression()), !dbg !2083
  call void @llvm.dbg.declare(metadata i32* %li, metadata !2084, metadata !DIExpression()), !dbg !2085
  call void @llvm.dbg.declare(metadata i32* %lj, metadata !2086, metadata !DIExpression()), !dbg !2087
  call void @llvm.dbg.declare(metadata i32* %lk, metadata !2088, metadata !DIExpression()), !dbg !2089
  call void @llvm.dbg.declare(metadata i32* %ku, metadata !2090, metadata !DIExpression()), !dbg !2091
  call void @llvm.dbg.declare(metadata i32* %i11, metadata !2092, metadata !DIExpression()), !dbg !2093
  call void @llvm.dbg.declare(metadata i32* %i12, metadata !2094, metadata !DIExpression()), !dbg !2095
  call void @llvm.dbg.declare(metadata i32* %i21, metadata !2096, metadata !DIExpression()), !dbg !2097
  call void @llvm.dbg.declare(metadata i32* %i22, metadata !2098, metadata !DIExpression()), !dbg !2099
  %4 = load i32, i32* %x_z, align 4, !dbg !2100
  %rem = srem i32 %4, 256, !dbg !2101
  store i32 %rem, i32* %i, align 4, !dbg !2102
  %5 = load i32, i32* %x_z, align 4, !dbg !2103
  %div = sdiv i32 %5, 256, !dbg !2104
  %rem3 = srem i32 %div, 128, !dbg !2105
  store i32 %rem3, i32* %k, align 4, !dbg !2106
  call void @llvm.dbg.declare(metadata i32* %logd2, metadata !2107, metadata !DIExpression()), !dbg !2108
  %call4 = call i32 @_Z12ilog2_devicei(i32 256) #4, !dbg !2109
  store i32 %call4, i32* %logd2, align 4, !dbg !2108
  call void @llvm.dbg.declare(metadata double* %uu1_real, metadata !2110, metadata !DIExpression()), !dbg !2111
  call void @llvm.dbg.declare(metadata double* %x11_real, metadata !2112, metadata !DIExpression()), !dbg !2113
  call void @llvm.dbg.declare(metadata double* %x21_real, metadata !2114, metadata !DIExpression()), !dbg !2115
  call void @llvm.dbg.declare(metadata double* %uu1_imag, metadata !2116, metadata !DIExpression()), !dbg !2117
  call void @llvm.dbg.declare(metadata double* %x11_imag, metadata !2118, metadata !DIExpression()), !dbg !2119
  call void @llvm.dbg.declare(metadata double* %x21_imag, metadata !2120, metadata !DIExpression()), !dbg !2121
  call void @llvm.dbg.declare(metadata double* %uu2_real, metadata !2122, metadata !DIExpression()), !dbg !2123
  call void @llvm.dbg.declare(metadata double* %x12_real, metadata !2124, metadata !DIExpression()), !dbg !2125
  call void @llvm.dbg.declare(metadata double* %x22_real, metadata !2126, metadata !DIExpression()), !dbg !2127
  call void @llvm.dbg.declare(metadata double* %uu2_imag, metadata !2128, metadata !DIExpression()), !dbg !2129
  call void @llvm.dbg.declare(metadata double* %x12_imag, metadata !2130, metadata !DIExpression()), !dbg !2131
  call void @llvm.dbg.declare(metadata double* %x22_imag, metadata !2132, metadata !DIExpression()), !dbg !2133
  call void @llvm.dbg.declare(metadata double* %temp_real, metadata !2134, metadata !DIExpression()), !dbg !2135
  call void @llvm.dbg.declare(metadata double* %temp2_real, metadata !2136, metadata !DIExpression()), !dbg !2137
  call void @llvm.dbg.declare(metadata double* %temp_imag, metadata !2138, metadata !DIExpression()), !dbg !2139
  call void @llvm.dbg.declare(metadata double* %temp2_imag, metadata !2140, metadata !DIExpression()), !dbg !2141
  store i32 1, i32* %l, align 4, !dbg !2142
  br label %for.cond, !dbg !2144

for.cond:                                         ; preds = %for.inc269, %if.end
  %6 = load i32, i32* %l, align 4, !dbg !2145
  %7 = load i32, i32* %logd2, align 4, !dbg !2147
  %cmp5 = icmp sle i32 %6, %7, !dbg !2148
  br i1 %cmp5, label %for.body, label %for.end271, !dbg !2149

for.body:                                         ; preds = %for.cond
  store i32 128, i32* %n1, align 4, !dbg !2150
  %8 = load i32, i32* %l, align 4, !dbg !2152
  %sub = sub nsw i32 %8, 1, !dbg !2153
  %shl = shl i32 1, %sub, !dbg !2154
  store i32 %shl, i32* %lk, align 4, !dbg !2155
  %9 = load i32, i32* %logd2, align 4, !dbg !2156
  %10 = load i32, i32* %l, align 4, !dbg !2157
  %sub6 = sub nsw i32 %9, %10, !dbg !2158
  %shl7 = shl i32 1, %sub6, !dbg !2159
  store i32 %shl7, i32* %li, align 4, !dbg !2160
  %11 = load i32, i32* %lk, align 4, !dbg !2161
  %mul8 = mul nsw i32 2, %11, !dbg !2162
  store i32 %mul8, i32* %lj, align 4, !dbg !2163
  %12 = load i32, i32* %li, align 4, !dbg !2164
  store i32 %12, i32* %ku, align 4, !dbg !2165
  store i32 0, i32* %i1, align 4, !dbg !2166
  br label %for.cond9, !dbg !2168

for.cond9:                                        ; preds = %for.inc108, %for.body
  %13 = load i32, i32* %i1, align 4, !dbg !2169
  %14 = load i32, i32* %li, align 4, !dbg !2171
  %sub10 = sub nsw i32 %14, 1, !dbg !2172
  %cmp11 = icmp sle i32 %13, %sub10, !dbg !2173
  br i1 %cmp11, label %for.body12, label %for.end110, !dbg !2174

for.body12:                                       ; preds = %for.cond9
  store i32 0, i32* %k1, align 4, !dbg !2175
  br label %for.cond13, !dbg !2178

for.cond13:                                       ; preds = %for.inc, %for.body12
  %15 = load i32, i32* %k1, align 4, !dbg !2179
  %16 = load i32, i32* %lk, align 4, !dbg !2181
  %sub14 = sub nsw i32 %16, 1, !dbg !2182
  %cmp15 = icmp sle i32 %15, %sub14, !dbg !2183
  br i1 %cmp15, label %for.body16, label %for.end, !dbg !2184

for.body16:                                       ; preds = %for.cond13
  %17 = load i32, i32* %i1, align 4, !dbg !2185
  %18 = load i32, i32* %lk, align 4, !dbg !2187
  %mul17 = mul nsw i32 %17, %18, !dbg !2188
  store i32 %mul17, i32* %i11, align 4, !dbg !2189
  %19 = load i32, i32* %i11, align 4, !dbg !2190
  %20 = load i32, i32* %n1, align 4, !dbg !2191
  %add18 = add nsw i32 %19, %20, !dbg !2192
  store i32 %add18, i32* %i12, align 4, !dbg !2193
  %21 = load i32, i32* %i1, align 4, !dbg !2194
  %22 = load i32, i32* %lj, align 4, !dbg !2195
  %mul19 = mul nsw i32 %21, %22, !dbg !2196
  store i32 %mul19, i32* %i21, align 4, !dbg !2197
  %23 = load i32, i32* %i21, align 4, !dbg !2198
  %24 = load i32, i32* %lk, align 4, !dbg !2199
  %add20 = add nsw i32 %23, %24, !dbg !2200
  store i32 %add20, i32* %i22, align 4, !dbg !2201
  %25 = load %struct.dcomplex*, %struct.dcomplex** %u_device.addr, align 8, !dbg !2202
  %26 = load i32, i32* %ku, align 4, !dbg !2203
  %27 = load i32, i32* %i1, align 4, !dbg !2204
  %add21 = add nsw i32 %26, %27, !dbg !2205
  %idxprom = sext i32 %add21 to i64, !dbg !2202
  %arrayidx = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %25, i64 %idxprom, !dbg !2202
  %real = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx, i32 0, i32 0, !dbg !2206
  %28 = load double, double* %real, align 8, !dbg !2206
  store double %28, double* %uu1_real, align 8, !dbg !2207
  %29 = load i32, i32* %is.addr, align 4, !dbg !2208
  %conv = sitofp i32 %29 to double, !dbg !2208
  %30 = load %struct.dcomplex*, %struct.dcomplex** %u_device.addr, align 8, !dbg !2209
  %31 = load i32, i32* %ku, align 4, !dbg !2210
  %32 = load i32, i32* %i1, align 4, !dbg !2211
  %add22 = add nsw i32 %31, %32, !dbg !2212
  %idxprom23 = sext i32 %add22 to i64, !dbg !2209
  %arrayidx24 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %30, i64 %idxprom23, !dbg !2209
  %imag = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx24, i32 0, i32 1, !dbg !2213
  %33 = load double, double* %imag, align 8, !dbg !2213
  %mul25 = fmul contract double %conv, %33, !dbg !2214
  store double %mul25, double* %uu1_imag, align 8, !dbg !2215
  %34 = load %struct.dcomplex*, %struct.dcomplex** %gty1.addr, align 8, !dbg !2216
  %35 = load i32, i32* %i, align 4, !dbg !2217
  %36 = load i32, i32* %i11, align 4, !dbg !2218
  %37 = load i32, i32* %k1, align 4, !dbg !2219
  %add26 = add nsw i32 %36, %37, !dbg !2220
  %mul27 = mul nsw i32 %add26, 256, !dbg !2221
  %add28 = add nsw i32 %35, %mul27, !dbg !2222
  %38 = load i32, i32* %k, align 4, !dbg !2223
  %mul29 = mul nsw i32 %38, 256, !dbg !2224
  %mul30 = mul nsw i32 %mul29, 256, !dbg !2225
  %add31 = add nsw i32 %add28, %mul30, !dbg !2226
  %idxprom32 = sext i32 %add31 to i64, !dbg !2216
  %arrayidx33 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %34, i64 %idxprom32, !dbg !2216
  %real34 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx33, i32 0, i32 0, !dbg !2227
  %39 = load double, double* %real34, align 8, !dbg !2227
  store double %39, double* %x11_real, align 8, !dbg !2228
  %40 = load %struct.dcomplex*, %struct.dcomplex** %gty1.addr, align 8, !dbg !2229
  %41 = load i32, i32* %i, align 4, !dbg !2230
  %42 = load i32, i32* %i11, align 4, !dbg !2231
  %43 = load i32, i32* %k1, align 4, !dbg !2232
  %add35 = add nsw i32 %42, %43, !dbg !2233
  %mul36 = mul nsw i32 %add35, 256, !dbg !2234
  %add37 = add nsw i32 %41, %mul36, !dbg !2235
  %44 = load i32, i32* %k, align 4, !dbg !2236
  %mul38 = mul nsw i32 %44, 256, !dbg !2237
  %mul39 = mul nsw i32 %mul38, 256, !dbg !2238
  %add40 = add nsw i32 %add37, %mul39, !dbg !2239
  %idxprom41 = sext i32 %add40 to i64, !dbg !2229
  %arrayidx42 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %40, i64 %idxprom41, !dbg !2229
  %imag43 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx42, i32 0, i32 1, !dbg !2240
  %45 = load double, double* %imag43, align 8, !dbg !2240
  store double %45, double* %x11_imag, align 8, !dbg !2241
  %46 = load %struct.dcomplex*, %struct.dcomplex** %gty1.addr, align 8, !dbg !2242
  %47 = load i32, i32* %i, align 4, !dbg !2243
  %48 = load i32, i32* %i12, align 4, !dbg !2244
  %49 = load i32, i32* %k1, align 4, !dbg !2245
  %add44 = add nsw i32 %48, %49, !dbg !2246
  %mul45 = mul nsw i32 %add44, 256, !dbg !2247
  %add46 = add nsw i32 %47, %mul45, !dbg !2248
  %50 = load i32, i32* %k, align 4, !dbg !2249
  %mul47 = mul nsw i32 %50, 256, !dbg !2250
  %mul48 = mul nsw i32 %mul47, 256, !dbg !2251
  %add49 = add nsw i32 %add46, %mul48, !dbg !2252
  %idxprom50 = sext i32 %add49 to i64, !dbg !2242
  %arrayidx51 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %46, i64 %idxprom50, !dbg !2242
  %real52 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx51, i32 0, i32 0, !dbg !2253
  %51 = load double, double* %real52, align 8, !dbg !2253
  store double %51, double* %x21_real, align 8, !dbg !2254
  %52 = load %struct.dcomplex*, %struct.dcomplex** %gty1.addr, align 8, !dbg !2255
  %53 = load i32, i32* %i, align 4, !dbg !2256
  %54 = load i32, i32* %i12, align 4, !dbg !2257
  %55 = load i32, i32* %k1, align 4, !dbg !2258
  %add53 = add nsw i32 %54, %55, !dbg !2259
  %mul54 = mul nsw i32 %add53, 256, !dbg !2260
  %add55 = add nsw i32 %53, %mul54, !dbg !2261
  %56 = load i32, i32* %k, align 4, !dbg !2262
  %mul56 = mul nsw i32 %56, 256, !dbg !2263
  %mul57 = mul nsw i32 %mul56, 256, !dbg !2264
  %add58 = add nsw i32 %add55, %mul57, !dbg !2265
  %idxprom59 = sext i32 %add58 to i64, !dbg !2255
  %arrayidx60 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %52, i64 %idxprom59, !dbg !2255
  %imag61 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx60, i32 0, i32 1, !dbg !2266
  %57 = load double, double* %imag61, align 8, !dbg !2266
  store double %57, double* %x21_imag, align 8, !dbg !2267
  %58 = load double, double* %x11_real, align 8, !dbg !2268
  %59 = load double, double* %x21_real, align 8, !dbg !2269
  %add62 = fadd contract double %58, %59, !dbg !2270
  %60 = load %struct.dcomplex*, %struct.dcomplex** %gty2.addr, align 8, !dbg !2271
  %61 = load i32, i32* %i, align 4, !dbg !2272
  %62 = load i32, i32* %i21, align 4, !dbg !2273
  %63 = load i32, i32* %k1, align 4, !dbg !2274
  %add63 = add nsw i32 %62, %63, !dbg !2275
  %mul64 = mul nsw i32 %add63, 256, !dbg !2276
  %add65 = add nsw i32 %61, %mul64, !dbg !2277
  %64 = load i32, i32* %k, align 4, !dbg !2278
  %mul66 = mul nsw i32 %64, 256, !dbg !2279
  %mul67 = mul nsw i32 %mul66, 256, !dbg !2280
  %add68 = add nsw i32 %add65, %mul67, !dbg !2281
  %idxprom69 = sext i32 %add68 to i64, !dbg !2271
  %arrayidx70 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %60, i64 %idxprom69, !dbg !2271
  %real71 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx70, i32 0, i32 0, !dbg !2282
  store double %add62, double* %real71, align 8, !dbg !2283
  %65 = load double, double* %x11_imag, align 8, !dbg !2284
  %66 = load double, double* %x21_imag, align 8, !dbg !2285
  %add72 = fadd contract double %65, %66, !dbg !2286
  %67 = load %struct.dcomplex*, %struct.dcomplex** %gty2.addr, align 8, !dbg !2287
  %68 = load i32, i32* %i, align 4, !dbg !2288
  %69 = load i32, i32* %i21, align 4, !dbg !2289
  %70 = load i32, i32* %k1, align 4, !dbg !2290
  %add73 = add nsw i32 %69, %70, !dbg !2291
  %mul74 = mul nsw i32 %add73, 256, !dbg !2292
  %add75 = add nsw i32 %68, %mul74, !dbg !2293
  %71 = load i32, i32* %k, align 4, !dbg !2294
  %mul76 = mul nsw i32 %71, 256, !dbg !2295
  %mul77 = mul nsw i32 %mul76, 256, !dbg !2296
  %add78 = add nsw i32 %add75, %mul77, !dbg !2297
  %idxprom79 = sext i32 %add78 to i64, !dbg !2287
  %arrayidx80 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %67, i64 %idxprom79, !dbg !2287
  %imag81 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx80, i32 0, i32 1, !dbg !2298
  store double %add72, double* %imag81, align 8, !dbg !2299
  %72 = load double, double* %x11_real, align 8, !dbg !2300
  %73 = load double, double* %x21_real, align 8, !dbg !2301
  %sub82 = fsub contract double %72, %73, !dbg !2302
  store double %sub82, double* %temp_real, align 8, !dbg !2303
  %74 = load double, double* %x11_imag, align 8, !dbg !2304
  %75 = load double, double* %x21_imag, align 8, !dbg !2305
  %sub83 = fsub contract double %74, %75, !dbg !2306
  store double %sub83, double* %temp_imag, align 8, !dbg !2307
  %76 = load double, double* %uu1_real, align 8, !dbg !2308
  %77 = load double, double* %temp_real, align 8, !dbg !2309
  %mul84 = fmul contract double %76, %77, !dbg !2310
  %78 = load double, double* %uu1_imag, align 8, !dbg !2311
  %79 = load double, double* %temp_imag, align 8, !dbg !2312
  %mul85 = fmul contract double %78, %79, !dbg !2313
  %sub86 = fsub contract double %mul84, %mul85, !dbg !2314
  %80 = load %struct.dcomplex*, %struct.dcomplex** %gty2.addr, align 8, !dbg !2315
  %81 = load i32, i32* %i, align 4, !dbg !2316
  %82 = load i32, i32* %i22, align 4, !dbg !2317
  %83 = load i32, i32* %k1, align 4, !dbg !2318
  %add87 = add nsw i32 %82, %83, !dbg !2319
  %mul88 = mul nsw i32 %add87, 256, !dbg !2320
  %add89 = add nsw i32 %81, %mul88, !dbg !2321
  %84 = load i32, i32* %k, align 4, !dbg !2322
  %mul90 = mul nsw i32 %84, 256, !dbg !2323
  %mul91 = mul nsw i32 %mul90, 256, !dbg !2324
  %add92 = add nsw i32 %add89, %mul91, !dbg !2325
  %idxprom93 = sext i32 %add92 to i64, !dbg !2315
  %arrayidx94 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %80, i64 %idxprom93, !dbg !2315
  %real95 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx94, i32 0, i32 0, !dbg !2326
  store double %sub86, double* %real95, align 8, !dbg !2327
  %85 = load double, double* %uu1_real, align 8, !dbg !2328
  %86 = load double, double* %temp_imag, align 8, !dbg !2329
  %mul96 = fmul contract double %85, %86, !dbg !2330
  %87 = load double, double* %uu1_imag, align 8, !dbg !2331
  %88 = load double, double* %temp_real, align 8, !dbg !2332
  %mul97 = fmul contract double %87, %88, !dbg !2333
  %add98 = fadd contract double %mul96, %mul97, !dbg !2334
  %89 = load %struct.dcomplex*, %struct.dcomplex** %gty2.addr, align 8, !dbg !2335
  %90 = load i32, i32* %i, align 4, !dbg !2336
  %91 = load i32, i32* %i22, align 4, !dbg !2337
  %92 = load i32, i32* %k1, align 4, !dbg !2338
  %add99 = add nsw i32 %91, %92, !dbg !2339
  %mul100 = mul nsw i32 %add99, 256, !dbg !2340
  %add101 = add nsw i32 %90, %mul100, !dbg !2341
  %93 = load i32, i32* %k, align 4, !dbg !2342
  %mul102 = mul nsw i32 %93, 256, !dbg !2343
  %mul103 = mul nsw i32 %mul102, 256, !dbg !2344
  %add104 = add nsw i32 %add101, %mul103, !dbg !2345
  %idxprom105 = sext i32 %add104 to i64, !dbg !2335
  %arrayidx106 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %89, i64 %idxprom105, !dbg !2335
  %imag107 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx106, i32 0, i32 1, !dbg !2346
  store double %add98, double* %imag107, align 8, !dbg !2347
  br label %for.inc, !dbg !2348

for.inc:                                          ; preds = %for.body16
  %94 = load i32, i32* %k1, align 4, !dbg !2349
  %inc = add nsw i32 %94, 1, !dbg !2349
  store i32 %inc, i32* %k1, align 4, !dbg !2349
  br label %for.cond13, !dbg !2350, !llvm.loop !2351

for.end:                                          ; preds = %for.cond13
  br label %for.inc108, !dbg !2353

for.inc108:                                       ; preds = %for.end
  %95 = load i32, i32* %i1, align 4, !dbg !2354
  %inc109 = add nsw i32 %95, 1, !dbg !2354
  store i32 %inc109, i32* %i1, align 4, !dbg !2354
  br label %for.cond9, !dbg !2355, !llvm.loop !2356

for.end110:                                       ; preds = %for.cond9
  %96 = load i32, i32* %l, align 4, !dbg !2358
  %97 = load i32, i32* %logd2, align 4, !dbg !2360
  %cmp111 = icmp eq i32 %96, %97, !dbg !2361
  br i1 %cmp111, label %if.then112, label %if.else, !dbg !2362

if.then112:                                       ; preds = %for.end110
  store i32 0, i32* %j1, align 4, !dbg !2363
  br label %for.cond113, !dbg !2366

for.cond113:                                      ; preds = %for.inc148, %if.then112
  %98 = load i32, i32* %j1, align 4, !dbg !2367
  %cmp114 = icmp slt i32 %98, 256, !dbg !2369
  br i1 %cmp114, label %for.body115, label %for.end150, !dbg !2370

for.body115:                                      ; preds = %for.cond113
  %99 = load %struct.dcomplex*, %struct.dcomplex** %gty2.addr, align 8, !dbg !2371
  %100 = load i32, i32* %i, align 4, !dbg !2373
  %101 = load i32, i32* %j1, align 4, !dbg !2374
  %mul116 = mul nsw i32 %101, 256, !dbg !2375
  %add117 = add nsw i32 %100, %mul116, !dbg !2376
  %102 = load i32, i32* %k, align 4, !dbg !2377
  %mul118 = mul nsw i32 %102, 256, !dbg !2378
  %mul119 = mul nsw i32 %mul118, 256, !dbg !2379
  %add120 = add nsw i32 %add117, %mul119, !dbg !2380
  %idxprom121 = sext i32 %add120 to i64, !dbg !2371
  %arrayidx122 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %99, i64 %idxprom121, !dbg !2371
  %real123 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx122, i32 0, i32 0, !dbg !2381
  %103 = load double, double* %real123, align 8, !dbg !2381
  %104 = load %struct.dcomplex*, %struct.dcomplex** %gty1.addr, align 8, !dbg !2382
  %105 = load i32, i32* %i, align 4, !dbg !2383
  %106 = load i32, i32* %j1, align 4, !dbg !2384
  %mul124 = mul nsw i32 %106, 256, !dbg !2385
  %add125 = add nsw i32 %105, %mul124, !dbg !2386
  %107 = load i32, i32* %k, align 4, !dbg !2387
  %mul126 = mul nsw i32 %107, 256, !dbg !2388
  %mul127 = mul nsw i32 %mul126, 256, !dbg !2389
  %add128 = add nsw i32 %add125, %mul127, !dbg !2390
  %idxprom129 = sext i32 %add128 to i64, !dbg !2382
  %arrayidx130 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %104, i64 %idxprom129, !dbg !2382
  %real131 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx130, i32 0, i32 0, !dbg !2391
  store double %103, double* %real131, align 8, !dbg !2392
  %108 = load %struct.dcomplex*, %struct.dcomplex** %gty2.addr, align 8, !dbg !2393
  %109 = load i32, i32* %i, align 4, !dbg !2394
  %110 = load i32, i32* %j1, align 4, !dbg !2395
  %mul132 = mul nsw i32 %110, 256, !dbg !2396
  %add133 = add nsw i32 %109, %mul132, !dbg !2397
  %111 = load i32, i32* %k, align 4, !dbg !2398
  %mul134 = mul nsw i32 %111, 256, !dbg !2399
  %mul135 = mul nsw i32 %mul134, 256, !dbg !2400
  %add136 = add nsw i32 %add133, %mul135, !dbg !2401
  %idxprom137 = sext i32 %add136 to i64, !dbg !2393
  %arrayidx138 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %108, i64 %idxprom137, !dbg !2393
  %imag139 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx138, i32 0, i32 1, !dbg !2402
  %112 = load double, double* %imag139, align 8, !dbg !2402
  %113 = load %struct.dcomplex*, %struct.dcomplex** %gty1.addr, align 8, !dbg !2403
  %114 = load i32, i32* %i, align 4, !dbg !2404
  %115 = load i32, i32* %j1, align 4, !dbg !2405
  %mul140 = mul nsw i32 %115, 256, !dbg !2406
  %add141 = add nsw i32 %114, %mul140, !dbg !2407
  %116 = load i32, i32* %k, align 4, !dbg !2408
  %mul142 = mul nsw i32 %116, 256, !dbg !2409
  %mul143 = mul nsw i32 %mul142, 256, !dbg !2410
  %add144 = add nsw i32 %add141, %mul143, !dbg !2411
  %idxprom145 = sext i32 %add144 to i64, !dbg !2403
  %arrayidx146 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %113, i64 %idxprom145, !dbg !2403
  %imag147 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx146, i32 0, i32 1, !dbg !2412
  store double %112, double* %imag147, align 8, !dbg !2413
  br label %for.inc148, !dbg !2414

for.inc148:                                       ; preds = %for.body115
  %117 = load i32, i32* %j1, align 4, !dbg !2415
  %inc149 = add nsw i32 %117, 1, !dbg !2415
  store i32 %inc149, i32* %j1, align 4, !dbg !2415
  br label %for.cond113, !dbg !2416, !llvm.loop !2417

for.end150:                                       ; preds = %for.cond113
  br label %if.end268, !dbg !2419

if.else:                                          ; preds = %for.end110
  store i32 128, i32* %n1, align 4, !dbg !2420
  %118 = load i32, i32* %l, align 4, !dbg !2422
  %add151 = add nsw i32 %118, 1, !dbg !2423
  %sub152 = sub nsw i32 %add151, 1, !dbg !2424
  %shl153 = shl i32 1, %sub152, !dbg !2425
  store i32 %shl153, i32* %lk, align 4, !dbg !2426
  %119 = load i32, i32* %logd2, align 4, !dbg !2427
  %120 = load i32, i32* %l, align 4, !dbg !2428
  %add154 = add nsw i32 %120, 1, !dbg !2429
  %sub155 = sub nsw i32 %119, %add154, !dbg !2430
  %shl156 = shl i32 1, %sub155, !dbg !2431
  store i32 %shl156, i32* %li, align 4, !dbg !2432
  %121 = load i32, i32* %lk, align 4, !dbg !2433
  %mul157 = mul nsw i32 2, %121, !dbg !2434
  store i32 %mul157, i32* %lj, align 4, !dbg !2435
  %122 = load i32, i32* %li, align 4, !dbg !2436
  store i32 %122, i32* %ku, align 4, !dbg !2437
  store i32 0, i32* %i1, align 4, !dbg !2438
  br label %for.cond158, !dbg !2440

for.cond158:                                      ; preds = %for.inc265, %if.else
  %123 = load i32, i32* %i1, align 4, !dbg !2441
  %124 = load i32, i32* %li, align 4, !dbg !2443
  %sub159 = sub nsw i32 %124, 1, !dbg !2444
  %cmp160 = icmp sle i32 %123, %sub159, !dbg !2445
  br i1 %cmp160, label %for.body161, label %for.end267, !dbg !2446

for.body161:                                      ; preds = %for.cond158
  store i32 0, i32* %k1, align 4, !dbg !2447
  br label %for.cond162, !dbg !2450

for.cond162:                                      ; preds = %for.inc262, %for.body161
  %125 = load i32, i32* %k1, align 4, !dbg !2451
  %126 = load i32, i32* %lk, align 4, !dbg !2453
  %sub163 = sub nsw i32 %126, 1, !dbg !2454
  %cmp164 = icmp sle i32 %125, %sub163, !dbg !2455
  br i1 %cmp164, label %for.body165, label %for.end264, !dbg !2456

for.body165:                                      ; preds = %for.cond162
  %127 = load i32, i32* %i1, align 4, !dbg !2457
  %128 = load i32, i32* %lk, align 4, !dbg !2459
  %mul166 = mul nsw i32 %127, %128, !dbg !2460
  store i32 %mul166, i32* %i11, align 4, !dbg !2461
  %129 = load i32, i32* %i11, align 4, !dbg !2462
  %130 = load i32, i32* %n1, align 4, !dbg !2463
  %add167 = add nsw i32 %129, %130, !dbg !2464
  store i32 %add167, i32* %i12, align 4, !dbg !2465
  %131 = load i32, i32* %i1, align 4, !dbg !2466
  %132 = load i32, i32* %lj, align 4, !dbg !2467
  %mul168 = mul nsw i32 %131, %132, !dbg !2468
  store i32 %mul168, i32* %i21, align 4, !dbg !2469
  %133 = load i32, i32* %i21, align 4, !dbg !2470
  %134 = load i32, i32* %lk, align 4, !dbg !2471
  %add169 = add nsw i32 %133, %134, !dbg !2472
  store i32 %add169, i32* %i22, align 4, !dbg !2473
  %135 = load %struct.dcomplex*, %struct.dcomplex** %u_device.addr, align 8, !dbg !2474
  %136 = load i32, i32* %ku, align 4, !dbg !2475
  %137 = load i32, i32* %i1, align 4, !dbg !2476
  %add170 = add nsw i32 %136, %137, !dbg !2477
  %idxprom171 = sext i32 %add170 to i64, !dbg !2474
  %arrayidx172 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %135, i64 %idxprom171, !dbg !2474
  %real173 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx172, i32 0, i32 0, !dbg !2478
  %138 = load double, double* %real173, align 8, !dbg !2478
  store double %138, double* %uu2_real, align 8, !dbg !2479
  %139 = load i32, i32* %is.addr, align 4, !dbg !2480
  %conv174 = sitofp i32 %139 to double, !dbg !2480
  %140 = load %struct.dcomplex*, %struct.dcomplex** %u_device.addr, align 8, !dbg !2481
  %141 = load i32, i32* %ku, align 4, !dbg !2482
  %142 = load i32, i32* %i1, align 4, !dbg !2483
  %add175 = add nsw i32 %141, %142, !dbg !2484
  %idxprom176 = sext i32 %add175 to i64, !dbg !2481
  %arrayidx177 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %140, i64 %idxprom176, !dbg !2481
  %imag178 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx177, i32 0, i32 1, !dbg !2485
  %143 = load double, double* %imag178, align 8, !dbg !2485
  %mul179 = fmul contract double %conv174, %143, !dbg !2486
  store double %mul179, double* %uu2_imag, align 8, !dbg !2487
  %144 = load %struct.dcomplex*, %struct.dcomplex** %gty2.addr, align 8, !dbg !2488
  %145 = load i32, i32* %i, align 4, !dbg !2489
  %146 = load i32, i32* %i11, align 4, !dbg !2490
  %147 = load i32, i32* %k1, align 4, !dbg !2491
  %add180 = add nsw i32 %146, %147, !dbg !2492
  %mul181 = mul nsw i32 %add180, 256, !dbg !2493
  %add182 = add nsw i32 %145, %mul181, !dbg !2494
  %148 = load i32, i32* %k, align 4, !dbg !2495
  %mul183 = mul nsw i32 %148, 256, !dbg !2496
  %mul184 = mul nsw i32 %mul183, 256, !dbg !2497
  %add185 = add nsw i32 %add182, %mul184, !dbg !2498
  %idxprom186 = sext i32 %add185 to i64, !dbg !2488
  %arrayidx187 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %144, i64 %idxprom186, !dbg !2488
  %real188 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx187, i32 0, i32 0, !dbg !2499
  %149 = load double, double* %real188, align 8, !dbg !2499
  store double %149, double* %x12_real, align 8, !dbg !2500
  %150 = load %struct.dcomplex*, %struct.dcomplex** %gty2.addr, align 8, !dbg !2501
  %151 = load i32, i32* %i, align 4, !dbg !2502
  %152 = load i32, i32* %i11, align 4, !dbg !2503
  %153 = load i32, i32* %k1, align 4, !dbg !2504
  %add189 = add nsw i32 %152, %153, !dbg !2505
  %mul190 = mul nsw i32 %add189, 256, !dbg !2506
  %add191 = add nsw i32 %151, %mul190, !dbg !2507
  %154 = load i32, i32* %k, align 4, !dbg !2508
  %mul192 = mul nsw i32 %154, 256, !dbg !2509
  %mul193 = mul nsw i32 %mul192, 256, !dbg !2510
  %add194 = add nsw i32 %add191, %mul193, !dbg !2511
  %idxprom195 = sext i32 %add194 to i64, !dbg !2501
  %arrayidx196 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %150, i64 %idxprom195, !dbg !2501
  %imag197 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx196, i32 0, i32 1, !dbg !2512
  %155 = load double, double* %imag197, align 8, !dbg !2512
  store double %155, double* %x12_imag, align 8, !dbg !2513
  %156 = load %struct.dcomplex*, %struct.dcomplex** %gty2.addr, align 8, !dbg !2514
  %157 = load i32, i32* %i, align 4, !dbg !2515
  %158 = load i32, i32* %i12, align 4, !dbg !2516
  %159 = load i32, i32* %k1, align 4, !dbg !2517
  %add198 = add nsw i32 %158, %159, !dbg !2518
  %mul199 = mul nsw i32 %add198, 256, !dbg !2519
  %add200 = add nsw i32 %157, %mul199, !dbg !2520
  %160 = load i32, i32* %k, align 4, !dbg !2521
  %mul201 = mul nsw i32 %160, 256, !dbg !2522
  %mul202 = mul nsw i32 %mul201, 256, !dbg !2523
  %add203 = add nsw i32 %add200, %mul202, !dbg !2524
  %idxprom204 = sext i32 %add203 to i64, !dbg !2514
  %arrayidx205 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %156, i64 %idxprom204, !dbg !2514
  %real206 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx205, i32 0, i32 0, !dbg !2525
  %161 = load double, double* %real206, align 8, !dbg !2525
  store double %161, double* %x22_real, align 8, !dbg !2526
  %162 = load %struct.dcomplex*, %struct.dcomplex** %gty2.addr, align 8, !dbg !2527
  %163 = load i32, i32* %i, align 4, !dbg !2528
  %164 = load i32, i32* %i12, align 4, !dbg !2529
  %165 = load i32, i32* %k1, align 4, !dbg !2530
  %add207 = add nsw i32 %164, %165, !dbg !2531
  %mul208 = mul nsw i32 %add207, 256, !dbg !2532
  %add209 = add nsw i32 %163, %mul208, !dbg !2533
  %166 = load i32, i32* %k, align 4, !dbg !2534
  %mul210 = mul nsw i32 %166, 256, !dbg !2535
  %mul211 = mul nsw i32 %mul210, 256, !dbg !2536
  %add212 = add nsw i32 %add209, %mul211, !dbg !2537
  %idxprom213 = sext i32 %add212 to i64, !dbg !2527
  %arrayidx214 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %162, i64 %idxprom213, !dbg !2527
  %imag215 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx214, i32 0, i32 1, !dbg !2538
  %167 = load double, double* %imag215, align 8, !dbg !2538
  store double %167, double* %x22_imag, align 8, !dbg !2539
  %168 = load double, double* %x12_real, align 8, !dbg !2540
  %169 = load double, double* %x22_real, align 8, !dbg !2541
  %add216 = fadd contract double %168, %169, !dbg !2542
  %170 = load %struct.dcomplex*, %struct.dcomplex** %gty1.addr, align 8, !dbg !2543
  %171 = load i32, i32* %i, align 4, !dbg !2544
  %172 = load i32, i32* %i21, align 4, !dbg !2545
  %173 = load i32, i32* %k1, align 4, !dbg !2546
  %add217 = add nsw i32 %172, %173, !dbg !2547
  %mul218 = mul nsw i32 %add217, 256, !dbg !2548
  %add219 = add nsw i32 %171, %mul218, !dbg !2549
  %174 = load i32, i32* %k, align 4, !dbg !2550
  %mul220 = mul nsw i32 %174, 256, !dbg !2551
  %mul221 = mul nsw i32 %mul220, 256, !dbg !2552
  %add222 = add nsw i32 %add219, %mul221, !dbg !2553
  %idxprom223 = sext i32 %add222 to i64, !dbg !2543
  %arrayidx224 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %170, i64 %idxprom223, !dbg !2543
  %real225 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx224, i32 0, i32 0, !dbg !2554
  store double %add216, double* %real225, align 8, !dbg !2555
  %175 = load double, double* %x12_imag, align 8, !dbg !2556
  %176 = load double, double* %x22_imag, align 8, !dbg !2557
  %add226 = fadd contract double %175, %176, !dbg !2558
  %177 = load %struct.dcomplex*, %struct.dcomplex** %gty1.addr, align 8, !dbg !2559
  %178 = load i32, i32* %i, align 4, !dbg !2560
  %179 = load i32, i32* %i21, align 4, !dbg !2561
  %180 = load i32, i32* %k1, align 4, !dbg !2562
  %add227 = add nsw i32 %179, %180, !dbg !2563
  %mul228 = mul nsw i32 %add227, 256, !dbg !2564
  %add229 = add nsw i32 %178, %mul228, !dbg !2565
  %181 = load i32, i32* %k, align 4, !dbg !2566
  %mul230 = mul nsw i32 %181, 256, !dbg !2567
  %mul231 = mul nsw i32 %mul230, 256, !dbg !2568
  %add232 = add nsw i32 %add229, %mul231, !dbg !2569
  %idxprom233 = sext i32 %add232 to i64, !dbg !2559
  %arrayidx234 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %177, i64 %idxprom233, !dbg !2559
  %imag235 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx234, i32 0, i32 1, !dbg !2570
  store double %add226, double* %imag235, align 8, !dbg !2571
  %182 = load double, double* %x12_real, align 8, !dbg !2572
  %183 = load double, double* %x22_real, align 8, !dbg !2573
  %sub236 = fsub contract double %182, %183, !dbg !2574
  store double %sub236, double* %temp2_real, align 8, !dbg !2575
  %184 = load double, double* %x12_imag, align 8, !dbg !2576
  %185 = load double, double* %x22_imag, align 8, !dbg !2577
  %sub237 = fsub contract double %184, %185, !dbg !2578
  store double %sub237, double* %temp2_imag, align 8, !dbg !2579
  %186 = load double, double* %uu2_real, align 8, !dbg !2580
  %187 = load double, double* %temp2_real, align 8, !dbg !2581
  %mul238 = fmul contract double %186, %187, !dbg !2582
  %188 = load double, double* %uu2_imag, align 8, !dbg !2583
  %189 = load double, double* %temp2_imag, align 8, !dbg !2584
  %mul239 = fmul contract double %188, %189, !dbg !2585
  %sub240 = fsub contract double %mul238, %mul239, !dbg !2586
  %190 = load %struct.dcomplex*, %struct.dcomplex** %gty1.addr, align 8, !dbg !2587
  %191 = load i32, i32* %i, align 4, !dbg !2588
  %192 = load i32, i32* %i22, align 4, !dbg !2589
  %193 = load i32, i32* %k1, align 4, !dbg !2590
  %add241 = add nsw i32 %192, %193, !dbg !2591
  %mul242 = mul nsw i32 %add241, 256, !dbg !2592
  %add243 = add nsw i32 %191, %mul242, !dbg !2593
  %194 = load i32, i32* %k, align 4, !dbg !2594
  %mul244 = mul nsw i32 %194, 256, !dbg !2595
  %mul245 = mul nsw i32 %mul244, 256, !dbg !2596
  %add246 = add nsw i32 %add243, %mul245, !dbg !2597
  %idxprom247 = sext i32 %add246 to i64, !dbg !2587
  %arrayidx248 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %190, i64 %idxprom247, !dbg !2587
  %real249 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx248, i32 0, i32 0, !dbg !2598
  store double %sub240, double* %real249, align 8, !dbg !2599
  %195 = load double, double* %uu2_real, align 8, !dbg !2600
  %196 = load double, double* %temp2_imag, align 8, !dbg !2601
  %mul250 = fmul contract double %195, %196, !dbg !2602
  %197 = load double, double* %uu2_imag, align 8, !dbg !2603
  %198 = load double, double* %temp2_real, align 8, !dbg !2604
  %mul251 = fmul contract double %197, %198, !dbg !2605
  %add252 = fadd contract double %mul250, %mul251, !dbg !2606
  %199 = load %struct.dcomplex*, %struct.dcomplex** %gty1.addr, align 8, !dbg !2607
  %200 = load i32, i32* %i, align 4, !dbg !2608
  %201 = load i32, i32* %i22, align 4, !dbg !2609
  %202 = load i32, i32* %k1, align 4, !dbg !2610
  %add253 = add nsw i32 %201, %202, !dbg !2611
  %mul254 = mul nsw i32 %add253, 256, !dbg !2612
  %add255 = add nsw i32 %200, %mul254, !dbg !2613
  %203 = load i32, i32* %k, align 4, !dbg !2614
  %mul256 = mul nsw i32 %203, 256, !dbg !2615
  %mul257 = mul nsw i32 %mul256, 256, !dbg !2616
  %add258 = add nsw i32 %add255, %mul257, !dbg !2617
  %idxprom259 = sext i32 %add258 to i64, !dbg !2607
  %arrayidx260 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %199, i64 %idxprom259, !dbg !2607
  %imag261 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx260, i32 0, i32 1, !dbg !2618
  store double %add252, double* %imag261, align 8, !dbg !2619
  br label %for.inc262, !dbg !2620

for.inc262:                                       ; preds = %for.body165
  %204 = load i32, i32* %k1, align 4, !dbg !2621
  %inc263 = add nsw i32 %204, 1, !dbg !2621
  store i32 %inc263, i32* %k1, align 4, !dbg !2621
  br label %for.cond162, !dbg !2622, !llvm.loop !2623

for.end264:                                       ; preds = %for.cond162
  br label %for.inc265, !dbg !2625

for.inc265:                                       ; preds = %for.end264
  %205 = load i32, i32* %i1, align 4, !dbg !2626
  %inc266 = add nsw i32 %205, 1, !dbg !2626
  store i32 %inc266, i32* %i1, align 4, !dbg !2626
  br label %for.cond158, !dbg !2627, !llvm.loop !2628

for.end267:                                       ; preds = %for.cond158
  br label %if.end268

if.end268:                                        ; preds = %for.end267, %for.end150
  br label %for.inc269, !dbg !2630

for.inc269:                                       ; preds = %if.end268
  %206 = load i32, i32* %l, align 4, !dbg !2631
  %add270 = add nsw i32 %206, 2, !dbg !2631
  store i32 %add270, i32* %l, align 4, !dbg !2631
  br label %for.cond, !dbg !2632, !llvm.loop !2633

for.end271:                                       ; preds = %for.cond, %if.then
  ret void, !dbg !2635
}

; Function Attrs: convergent noinline nounwind
define dso_local void @_Z19cffts2_gpu_kernel_3P8dcomplexS0_(%struct.dcomplex* %x_out, %struct.dcomplex* %y0) #0 !dbg !2636 {
entry:
  %x_out.addr = alloca %struct.dcomplex*, align 8
  %y0.addr = alloca %struct.dcomplex*, align 8
  %x_y_z = alloca i32, align 4
  store %struct.dcomplex* %x_out, %struct.dcomplex** %x_out.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %x_out.addr, metadata !2637, metadata !DIExpression()), !dbg !2638
  store %struct.dcomplex* %y0, %struct.dcomplex** %y0.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %y0.addr, metadata !2639, metadata !DIExpression()), !dbg !2640
  call void @llvm.dbg.declare(metadata i32* %x_y_z, metadata !2641, metadata !DIExpression()), !dbg !2642
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #11, !dbg !2643, !range !1198
  %1 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #11, !dbg !2645, !range !1243
  %mul = mul i32 %0, %1, !dbg !2647
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #11, !dbg !2648, !range !1273
  %add = add i32 %mul, %2, !dbg !2650
  store i32 %add, i32* %x_y_z, align 4, !dbg !2642
  %3 = load i32, i32* %x_y_z, align 4, !dbg !2651
  %cmp = icmp sge i32 %3, 8388608, !dbg !2653
  br i1 %cmp, label %if.then, label %if.end, !dbg !2654

if.then:                                          ; preds = %entry
  br label %return, !dbg !2655

if.end:                                           ; preds = %entry
  %4 = load %struct.dcomplex*, %struct.dcomplex** %y0.addr, align 8, !dbg !2657
  %5 = load i32, i32* %x_y_z, align 4, !dbg !2658
  %idxprom = sext i32 %5 to i64, !dbg !2657
  %arrayidx = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %4, i64 %idxprom, !dbg !2657
  %real = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx, i32 0, i32 0, !dbg !2659
  %6 = load double, double* %real, align 8, !dbg !2659
  %7 = load %struct.dcomplex*, %struct.dcomplex** %x_out.addr, align 8, !dbg !2660
  %8 = load i32, i32* %x_y_z, align 4, !dbg !2661
  %idxprom3 = sext i32 %8 to i64, !dbg !2660
  %arrayidx4 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %7, i64 %idxprom3, !dbg !2660
  %real5 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx4, i32 0, i32 0, !dbg !2662
  store double %6, double* %real5, align 8, !dbg !2663
  %9 = load %struct.dcomplex*, %struct.dcomplex** %y0.addr, align 8, !dbg !2664
  %10 = load i32, i32* %x_y_z, align 4, !dbg !2665
  %idxprom6 = sext i32 %10 to i64, !dbg !2664
  %arrayidx7 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %9, i64 %idxprom6, !dbg !2664
  %imag = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx7, i32 0, i32 1, !dbg !2666
  %11 = load double, double* %imag, align 8, !dbg !2666
  %12 = load %struct.dcomplex*, %struct.dcomplex** %x_out.addr, align 8, !dbg !2667
  %13 = load i32, i32* %x_y_z, align 4, !dbg !2668
  %idxprom8 = sext i32 %13 to i64, !dbg !2667
  %arrayidx9 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %12, i64 %idxprom8, !dbg !2667
  %imag10 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx9, i32 0, i32 1, !dbg !2669
  store double %11, double* %imag10, align 8, !dbg !2670
  br label %return, !dbg !2671

return:                                           ; preds = %if.end, %if.then
  ret void, !dbg !2671
}

; Function Attrs: convergent noinline nounwind
define dso_local void @_Z23cffts3_gpu_cfftz_deviceiiiP8dcomplexS0_S0_ii(i32 %is, i32 %m, i32 %n, %struct.dcomplex* %x, %struct.dcomplex* %y, %struct.dcomplex* %u_device, i32 %index_arg, i32 %size_arg) #0 !dbg !2672 {
entry:
  %is.addr = alloca i32, align 4
  %m.addr = alloca i32, align 4
  %n.addr = alloca i32, align 4
  %x.addr = alloca %struct.dcomplex*, align 8
  %y.addr = alloca %struct.dcomplex*, align 8
  %u_device.addr = alloca %struct.dcomplex*, align 8
  %index_arg.addr = alloca i32, align 4
  %size_arg.addr = alloca i32, align 4
  %j = alloca i32, align 4
  %l = alloca i32, align 4
  store i32 %is, i32* %is.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %is.addr, metadata !2675, metadata !DIExpression()), !dbg !2676
  store i32 %m, i32* %m.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %m.addr, metadata !2677, metadata !DIExpression()), !dbg !2678
  store i32 %n, i32* %n.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %n.addr, metadata !2679, metadata !DIExpression()), !dbg !2680
  store %struct.dcomplex* %x, %struct.dcomplex** %x.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %x.addr, metadata !2681, metadata !DIExpression()), !dbg !2682
  store %struct.dcomplex* %y, %struct.dcomplex** %y.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %y.addr, metadata !2683, metadata !DIExpression()), !dbg !2684
  store %struct.dcomplex* %u_device, %struct.dcomplex** %u_device.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %u_device.addr, metadata !2685, metadata !DIExpression()), !dbg !2686
  store i32 %index_arg, i32* %index_arg.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %index_arg.addr, metadata !2687, metadata !DIExpression()), !dbg !2688
  store i32 %size_arg, i32* %size_arg.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %size_arg.addr, metadata !2689, metadata !DIExpression()), !dbg !2690
  call void @llvm.dbg.declare(metadata i32* %j, metadata !2691, metadata !DIExpression()), !dbg !2692
  call void @llvm.dbg.declare(metadata i32* %l, metadata !2693, metadata !DIExpression()), !dbg !2694
  store i32 1, i32* %l, align 4, !dbg !2695
  br label %for.cond, !dbg !2697

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, i32* %l, align 4, !dbg !2698
  %1 = load i32, i32* %m.addr, align 4, !dbg !2700
  %cmp = icmp sle i32 %0, %1, !dbg !2701
  br i1 %cmp, label %for.body, label %for.end, !dbg !2702

for.body:                                         ; preds = %for.cond
  %2 = load i32, i32* %is.addr, align 4, !dbg !2703
  %3 = load i32, i32* %l, align 4, !dbg !2705
  %4 = load i32, i32* %m.addr, align 4, !dbg !2706
  %5 = load i32, i32* %n.addr, align 4, !dbg !2707
  %6 = load %struct.dcomplex*, %struct.dcomplex** %u_device.addr, align 8, !dbg !2708
  %7 = load %struct.dcomplex*, %struct.dcomplex** %x.addr, align 8, !dbg !2709
  %8 = load %struct.dcomplex*, %struct.dcomplex** %y.addr, align 8, !dbg !2710
  %9 = load i32, i32* %index_arg.addr, align 4, !dbg !2711
  %10 = load i32, i32* %size_arg.addr, align 4, !dbg !2712
  call void @_Z23cffts3_gpu_fftz2_deviceiiiiP8dcomplexS0_S0_ii(i32 %2, i32 %3, i32 %4, i32 %5, %struct.dcomplex* %6, %struct.dcomplex* %7, %struct.dcomplex* %8, i32 %9, i32 %10) #4, !dbg !2713
  %11 = load i32, i32* %l, align 4, !dbg !2714
  %12 = load i32, i32* %m.addr, align 4, !dbg !2716
  %cmp1 = icmp eq i32 %11, %12, !dbg !2717
  br i1 %cmp1, label %if.then, label %if.end, !dbg !2718

if.then:                                          ; preds = %for.body
  br label %for.end, !dbg !2719

if.end:                                           ; preds = %for.body
  %13 = load i32, i32* %is.addr, align 4, !dbg !2721
  %14 = load i32, i32* %l, align 4, !dbg !2722
  %add = add nsw i32 %14, 1, !dbg !2723
  %15 = load i32, i32* %m.addr, align 4, !dbg !2724
  %16 = load i32, i32* %n.addr, align 4, !dbg !2725
  %17 = load %struct.dcomplex*, %struct.dcomplex** %u_device.addr, align 8, !dbg !2726
  %18 = load %struct.dcomplex*, %struct.dcomplex** %y.addr, align 8, !dbg !2727
  %19 = load %struct.dcomplex*, %struct.dcomplex** %x.addr, align 8, !dbg !2728
  %20 = load i32, i32* %index_arg.addr, align 4, !dbg !2729
  %21 = load i32, i32* %size_arg.addr, align 4, !dbg !2730
  call void @_Z23cffts3_gpu_fftz2_deviceiiiiP8dcomplexS0_S0_ii(i32 %13, i32 %add, i32 %15, i32 %16, %struct.dcomplex* %17, %struct.dcomplex* %18, %struct.dcomplex* %19, i32 %20, i32 %21) #4, !dbg !2731
  br label %for.inc, !dbg !2732

for.inc:                                          ; preds = %if.end
  %22 = load i32, i32* %l, align 4, !dbg !2733
  %add2 = add nsw i32 %22, 2, !dbg !2733
  store i32 %add2, i32* %l, align 4, !dbg !2733
  br label %for.cond, !dbg !2734, !llvm.loop !2735

for.end:                                          ; preds = %if.then, %for.cond
  %23 = load i32, i32* %m.addr, align 4, !dbg !2737
  %rem = srem i32 %23, 2, !dbg !2739
  %cmp3 = icmp eq i32 %rem, 1, !dbg !2740
  br i1 %cmp3, label %if.then4, label %if.end25, !dbg !2741

if.then4:                                         ; preds = %for.end
  store i32 0, i32* %j, align 4, !dbg !2742
  br label %for.cond5, !dbg !2745

for.cond5:                                        ; preds = %for.inc23, %if.then4
  %24 = load i32, i32* %j, align 4, !dbg !2746
  %25 = load i32, i32* %n.addr, align 4, !dbg !2748
  %cmp6 = icmp slt i32 %24, %25, !dbg !2749
  br i1 %cmp6, label %for.body7, label %for.end24, !dbg !2750

for.body7:                                        ; preds = %for.cond5
  %26 = load %struct.dcomplex*, %struct.dcomplex** %y.addr, align 8, !dbg !2751
  %27 = load i32, i32* %j, align 4, !dbg !2753
  %28 = load i32, i32* %size_arg.addr, align 4, !dbg !2754
  %mul = mul nsw i32 %27, %28, !dbg !2755
  %29 = load i32, i32* %index_arg.addr, align 4, !dbg !2756
  %add8 = add nsw i32 %mul, %29, !dbg !2757
  %idxprom = sext i32 %add8 to i64, !dbg !2751
  %arrayidx = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %26, i64 %idxprom, !dbg !2751
  %real = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx, i32 0, i32 0, !dbg !2758
  %30 = load double, double* %real, align 8, !dbg !2758
  %31 = load %struct.dcomplex*, %struct.dcomplex** %x.addr, align 8, !dbg !2759
  %32 = load i32, i32* %j, align 4, !dbg !2760
  %33 = load i32, i32* %size_arg.addr, align 4, !dbg !2761
  %mul9 = mul nsw i32 %32, %33, !dbg !2762
  %34 = load i32, i32* %index_arg.addr, align 4, !dbg !2763
  %add10 = add nsw i32 %mul9, %34, !dbg !2764
  %idxprom11 = sext i32 %add10 to i64, !dbg !2759
  %arrayidx12 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %31, i64 %idxprom11, !dbg !2759
  %real13 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx12, i32 0, i32 0, !dbg !2765
  store double %30, double* %real13, align 8, !dbg !2766
  %35 = load %struct.dcomplex*, %struct.dcomplex** %y.addr, align 8, !dbg !2767
  %36 = load i32, i32* %j, align 4, !dbg !2768
  %37 = load i32, i32* %size_arg.addr, align 4, !dbg !2769
  %mul14 = mul nsw i32 %36, %37, !dbg !2770
  %38 = load i32, i32* %index_arg.addr, align 4, !dbg !2771
  %add15 = add nsw i32 %mul14, %38, !dbg !2772
  %idxprom16 = sext i32 %add15 to i64, !dbg !2767
  %arrayidx17 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %35, i64 %idxprom16, !dbg !2767
  %imag = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx17, i32 0, i32 1, !dbg !2773
  %39 = load double, double* %imag, align 8, !dbg !2773
  %40 = load %struct.dcomplex*, %struct.dcomplex** %x.addr, align 8, !dbg !2774
  %41 = load i32, i32* %j, align 4, !dbg !2775
  %42 = load i32, i32* %size_arg.addr, align 4, !dbg !2776
  %mul18 = mul nsw i32 %41, %42, !dbg !2777
  %43 = load i32, i32* %index_arg.addr, align 4, !dbg !2778
  %add19 = add nsw i32 %mul18, %43, !dbg !2779
  %idxprom20 = sext i32 %add19 to i64, !dbg !2774
  %arrayidx21 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %40, i64 %idxprom20, !dbg !2774
  %imag22 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx21, i32 0, i32 1, !dbg !2780
  store double %39, double* %imag22, align 8, !dbg !2781
  br label %for.inc23, !dbg !2782

for.inc23:                                        ; preds = %for.body7
  %44 = load i32, i32* %j, align 4, !dbg !2783
  %inc = add nsw i32 %44, 1, !dbg !2783
  store i32 %inc, i32* %j, align 4, !dbg !2783
  br label %for.cond5, !dbg !2784, !llvm.loop !2785

for.end24:                                        ; preds = %for.cond5
  br label %if.end25, !dbg !2787

if.end25:                                         ; preds = %for.end24, %for.end
  ret void, !dbg !2788
}

; Function Attrs: convergent noinline nounwind
define dso_local void @_Z23cffts3_gpu_fftz2_deviceiiiiP8dcomplexS0_S0_ii(i32 %is, i32 %l, i32 %m, i32 %n, %struct.dcomplex* %u, %struct.dcomplex* %x, %struct.dcomplex* %y, i32 %index_arg, i32 %size_arg) #0 !dbg !2789 {
entry:
  %is.addr = alloca i32, align 4
  %l.addr = alloca i32, align 4
  %m.addr = alloca i32, align 4
  %n.addr = alloca i32, align 4
  %u.addr = alloca %struct.dcomplex*, align 8
  %x.addr = alloca %struct.dcomplex*, align 8
  %y.addr = alloca %struct.dcomplex*, align 8
  %index_arg.addr = alloca i32, align 4
  %size_arg.addr = alloca i32, align 4
  %k = alloca i32, align 4
  %n1 = alloca i32, align 4
  %li = alloca i32, align 4
  %lj = alloca i32, align 4
  %lk = alloca i32, align 4
  %ku = alloca i32, align 4
  %i = alloca i32, align 4
  %i11 = alloca i32, align 4
  %i12 = alloca i32, align 4
  %i21 = alloca i32, align 4
  %i22 = alloca i32, align 4
  %x11real = alloca double, align 8
  %x11imag = alloca double, align 8
  %x21real = alloca double, align 8
  %x21imag = alloca double, align 8
  %u1 = alloca %struct.dcomplex, align 8
  store i32 %is, i32* %is.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %is.addr, metadata !2792, metadata !DIExpression()), !dbg !2793
  store i32 %l, i32* %l.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %l.addr, metadata !2794, metadata !DIExpression()), !dbg !2795
  store i32 %m, i32* %m.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %m.addr, metadata !2796, metadata !DIExpression()), !dbg !2797
  store i32 %n, i32* %n.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %n.addr, metadata !2798, metadata !DIExpression()), !dbg !2799
  store %struct.dcomplex* %u, %struct.dcomplex** %u.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %u.addr, metadata !2800, metadata !DIExpression()), !dbg !2801
  store %struct.dcomplex* %x, %struct.dcomplex** %x.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %x.addr, metadata !2802, metadata !DIExpression()), !dbg !2803
  store %struct.dcomplex* %y, %struct.dcomplex** %y.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %y.addr, metadata !2804, metadata !DIExpression()), !dbg !2805
  store i32 %index_arg, i32* %index_arg.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %index_arg.addr, metadata !2806, metadata !DIExpression()), !dbg !2807
  store i32 %size_arg, i32* %size_arg.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %size_arg.addr, metadata !2808, metadata !DIExpression()), !dbg !2809
  call void @llvm.dbg.declare(metadata i32* %k, metadata !2810, metadata !DIExpression()), !dbg !2811
  call void @llvm.dbg.declare(metadata i32* %n1, metadata !2812, metadata !DIExpression()), !dbg !2813
  call void @llvm.dbg.declare(metadata i32* %li, metadata !2814, metadata !DIExpression()), !dbg !2815
  call void @llvm.dbg.declare(metadata i32* %lj, metadata !2816, metadata !DIExpression()), !dbg !2817
  call void @llvm.dbg.declare(metadata i32* %lk, metadata !2818, metadata !DIExpression()), !dbg !2819
  call void @llvm.dbg.declare(metadata i32* %ku, metadata !2820, metadata !DIExpression()), !dbg !2821
  call void @llvm.dbg.declare(metadata i32* %i, metadata !2822, metadata !DIExpression()), !dbg !2823
  call void @llvm.dbg.declare(metadata i32* %i11, metadata !2824, metadata !DIExpression()), !dbg !2825
  call void @llvm.dbg.declare(metadata i32* %i12, metadata !2826, metadata !DIExpression()), !dbg !2827
  call void @llvm.dbg.declare(metadata i32* %i21, metadata !2828, metadata !DIExpression()), !dbg !2829
  call void @llvm.dbg.declare(metadata i32* %i22, metadata !2830, metadata !DIExpression()), !dbg !2831
  call void @llvm.dbg.declare(metadata double* %x11real, metadata !2832, metadata !DIExpression()), !dbg !2833
  call void @llvm.dbg.declare(metadata double* %x11imag, metadata !2834, metadata !DIExpression()), !dbg !2835
  call void @llvm.dbg.declare(metadata double* %x21real, metadata !2836, metadata !DIExpression()), !dbg !2837
  call void @llvm.dbg.declare(metadata double* %x21imag, metadata !2838, metadata !DIExpression()), !dbg !2839
  call void @llvm.dbg.declare(metadata %struct.dcomplex* %u1, metadata !2840, metadata !DIExpression()), !dbg !2841
  %0 = load i32, i32* %n.addr, align 4, !dbg !2842
  %div = sdiv i32 %0, 2, !dbg !2843
  store i32 %div, i32* %n1, align 4, !dbg !2844
  %1 = load i32, i32* %l.addr, align 4, !dbg !2845
  %sub = sub nsw i32 %1, 1, !dbg !2846
  %shl = shl i32 1, %sub, !dbg !2847
  store i32 %shl, i32* %lk, align 4, !dbg !2848
  %2 = load i32, i32* %m.addr, align 4, !dbg !2849
  %3 = load i32, i32* %l.addr, align 4, !dbg !2850
  %sub1 = sub nsw i32 %2, %3, !dbg !2851
  %shl2 = shl i32 1, %sub1, !dbg !2852
  store i32 %shl2, i32* %li, align 4, !dbg !2853
  %4 = load i32, i32* %lk, align 4, !dbg !2854
  %mul = mul nsw i32 2, %4, !dbg !2855
  store i32 %mul, i32* %lj, align 4, !dbg !2856
  %5 = load i32, i32* %li, align 4, !dbg !2857
  store i32 %5, i32* %ku, align 4, !dbg !2858
  store i32 0, i32* %i, align 4, !dbg !2859
  br label %for.cond, !dbg !2861

for.cond:                                         ; preds = %for.inc91, %entry
  %6 = load i32, i32* %i, align 4, !dbg !2862
  %7 = load i32, i32* %li, align 4, !dbg !2864
  %cmp = icmp slt i32 %6, %7, !dbg !2865
  br i1 %cmp, label %for.body, label %for.end93, !dbg !2866

for.body:                                         ; preds = %for.cond
  %8 = load i32, i32* %i, align 4, !dbg !2867
  %9 = load i32, i32* %lk, align 4, !dbg !2869
  %mul3 = mul nsw i32 %8, %9, !dbg !2870
  store i32 %mul3, i32* %i11, align 4, !dbg !2871
  %10 = load i32, i32* %i11, align 4, !dbg !2872
  %11 = load i32, i32* %n1, align 4, !dbg !2873
  %add = add nsw i32 %10, %11, !dbg !2874
  store i32 %add, i32* %i12, align 4, !dbg !2875
  %12 = load i32, i32* %i, align 4, !dbg !2876
  %13 = load i32, i32* %lj, align 4, !dbg !2877
  %mul4 = mul nsw i32 %12, %13, !dbg !2878
  store i32 %mul4, i32* %i21, align 4, !dbg !2879
  %14 = load i32, i32* %i21, align 4, !dbg !2880
  %15 = load i32, i32* %lk, align 4, !dbg !2881
  %add5 = add nsw i32 %14, %15, !dbg !2882
  store i32 %add5, i32* %i22, align 4, !dbg !2883
  %16 = load i32, i32* %is.addr, align 4, !dbg !2884
  %cmp6 = icmp sge i32 %16, 1, !dbg !2886
  br i1 %cmp6, label %if.then, label %if.else, !dbg !2887

if.then:                                          ; preds = %for.body
  %17 = load %struct.dcomplex*, %struct.dcomplex** %u.addr, align 8, !dbg !2888
  %18 = load i32, i32* %ku, align 4, !dbg !2890
  %19 = load i32, i32* %i, align 4, !dbg !2891
  %add7 = add nsw i32 %18, %19, !dbg !2892
  %idxprom = sext i32 %add7 to i64, !dbg !2888
  %arrayidx = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %17, i64 %idxprom, !dbg !2888
  %real = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx, i32 0, i32 0, !dbg !2893
  %20 = load double, double* %real, align 8, !dbg !2893
  %real8 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %u1, i32 0, i32 0, !dbg !2894
  store double %20, double* %real8, align 8, !dbg !2895
  %21 = load %struct.dcomplex*, %struct.dcomplex** %u.addr, align 8, !dbg !2896
  %22 = load i32, i32* %ku, align 4, !dbg !2897
  %23 = load i32, i32* %i, align 4, !dbg !2898
  %add9 = add nsw i32 %22, %23, !dbg !2899
  %idxprom10 = sext i32 %add9 to i64, !dbg !2896
  %arrayidx11 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %21, i64 %idxprom10, !dbg !2896
  %imag = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx11, i32 0, i32 1, !dbg !2900
  %24 = load double, double* %imag, align 8, !dbg !2900
  %imag12 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %u1, i32 0, i32 1, !dbg !2901
  store double %24, double* %imag12, align 8, !dbg !2902
  br label %if.end, !dbg !2903

if.else:                                          ; preds = %for.body
  %25 = load %struct.dcomplex*, %struct.dcomplex** %u.addr, align 8, !dbg !2904
  %26 = load i32, i32* %ku, align 4, !dbg !2906
  %27 = load i32, i32* %i, align 4, !dbg !2907
  %add13 = add nsw i32 %26, %27, !dbg !2908
  %idxprom14 = sext i32 %add13 to i64, !dbg !2904
  %arrayidx15 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %25, i64 %idxprom14, !dbg !2904
  %real16 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx15, i32 0, i32 0, !dbg !2909
  %28 = load double, double* %real16, align 8, !dbg !2909
  %real17 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %u1, i32 0, i32 0, !dbg !2910
  store double %28, double* %real17, align 8, !dbg !2911
  %29 = load %struct.dcomplex*, %struct.dcomplex** %u.addr, align 8, !dbg !2912
  %30 = load i32, i32* %ku, align 4, !dbg !2913
  %31 = load i32, i32* %i, align 4, !dbg !2914
  %add18 = add nsw i32 %30, %31, !dbg !2915
  %idxprom19 = sext i32 %add18 to i64, !dbg !2912
  %arrayidx20 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %29, i64 %idxprom19, !dbg !2912
  %imag21 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx20, i32 0, i32 1, !dbg !2916
  %32 = load double, double* %imag21, align 8, !dbg !2916
  %sub22 = fsub double -0.000000e+00, %32, !dbg !2917
  %imag23 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %u1, i32 0, i32 1, !dbg !2918
  store double %sub22, double* %imag23, align 8, !dbg !2919
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  store i32 0, i32* %k, align 4, !dbg !2920
  br label %for.cond24, !dbg !2922

for.cond24:                                       ; preds = %for.inc, %if.end
  %33 = load i32, i32* %k, align 4, !dbg !2923
  %34 = load i32, i32* %lk, align 4, !dbg !2925
  %cmp25 = icmp slt i32 %33, %34, !dbg !2926
  br i1 %cmp25, label %for.body26, label %for.end, !dbg !2927

for.body26:                                       ; preds = %for.cond24
  %35 = load %struct.dcomplex*, %struct.dcomplex** %x.addr, align 8, !dbg !2928
  %36 = load i32, i32* %i11, align 4, !dbg !2930
  %37 = load i32, i32* %k, align 4, !dbg !2931
  %add27 = add nsw i32 %36, %37, !dbg !2932
  %38 = load i32, i32* %size_arg.addr, align 4, !dbg !2933
  %mul28 = mul nsw i32 %add27, %38, !dbg !2934
  %39 = load i32, i32* %index_arg.addr, align 4, !dbg !2935
  %add29 = add nsw i32 %mul28, %39, !dbg !2936
  %idxprom30 = sext i32 %add29 to i64, !dbg !2928
  %arrayidx31 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %35, i64 %idxprom30, !dbg !2928
  %real32 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx31, i32 0, i32 0, !dbg !2937
  %40 = load double, double* %real32, align 8, !dbg !2937
  store double %40, double* %x11real, align 8, !dbg !2938
  %41 = load %struct.dcomplex*, %struct.dcomplex** %x.addr, align 8, !dbg !2939
  %42 = load i32, i32* %i11, align 4, !dbg !2940
  %43 = load i32, i32* %k, align 4, !dbg !2941
  %add33 = add nsw i32 %42, %43, !dbg !2942
  %44 = load i32, i32* %size_arg.addr, align 4, !dbg !2943
  %mul34 = mul nsw i32 %add33, %44, !dbg !2944
  %45 = load i32, i32* %index_arg.addr, align 4, !dbg !2945
  %add35 = add nsw i32 %mul34, %45, !dbg !2946
  %idxprom36 = sext i32 %add35 to i64, !dbg !2939
  %arrayidx37 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %41, i64 %idxprom36, !dbg !2939
  %imag38 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx37, i32 0, i32 1, !dbg !2947
  %46 = load double, double* %imag38, align 8, !dbg !2947
  store double %46, double* %x11imag, align 8, !dbg !2948
  %47 = load %struct.dcomplex*, %struct.dcomplex** %x.addr, align 8, !dbg !2949
  %48 = load i32, i32* %i12, align 4, !dbg !2950
  %49 = load i32, i32* %k, align 4, !dbg !2951
  %add39 = add nsw i32 %48, %49, !dbg !2952
  %50 = load i32, i32* %size_arg.addr, align 4, !dbg !2953
  %mul40 = mul nsw i32 %add39, %50, !dbg !2954
  %51 = load i32, i32* %index_arg.addr, align 4, !dbg !2955
  %add41 = add nsw i32 %mul40, %51, !dbg !2956
  %idxprom42 = sext i32 %add41 to i64, !dbg !2949
  %arrayidx43 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %47, i64 %idxprom42, !dbg !2949
  %real44 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx43, i32 0, i32 0, !dbg !2957
  %52 = load double, double* %real44, align 8, !dbg !2957
  store double %52, double* %x21real, align 8, !dbg !2958
  %53 = load %struct.dcomplex*, %struct.dcomplex** %x.addr, align 8, !dbg !2959
  %54 = load i32, i32* %i12, align 4, !dbg !2960
  %55 = load i32, i32* %k, align 4, !dbg !2961
  %add45 = add nsw i32 %54, %55, !dbg !2962
  %56 = load i32, i32* %size_arg.addr, align 4, !dbg !2963
  %mul46 = mul nsw i32 %add45, %56, !dbg !2964
  %57 = load i32, i32* %index_arg.addr, align 4, !dbg !2965
  %add47 = add nsw i32 %mul46, %57, !dbg !2966
  %idxprom48 = sext i32 %add47 to i64, !dbg !2959
  %arrayidx49 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %53, i64 %idxprom48, !dbg !2959
  %imag50 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx49, i32 0, i32 1, !dbg !2967
  %58 = load double, double* %imag50, align 8, !dbg !2967
  store double %58, double* %x21imag, align 8, !dbg !2968
  %59 = load double, double* %x11real, align 8, !dbg !2969
  %60 = load double, double* %x21real, align 8, !dbg !2970
  %add51 = fadd contract double %59, %60, !dbg !2971
  %61 = load %struct.dcomplex*, %struct.dcomplex** %y.addr, align 8, !dbg !2972
  %62 = load i32, i32* %i21, align 4, !dbg !2973
  %63 = load i32, i32* %k, align 4, !dbg !2974
  %add52 = add nsw i32 %62, %63, !dbg !2975
  %64 = load i32, i32* %size_arg.addr, align 4, !dbg !2976
  %mul53 = mul nsw i32 %add52, %64, !dbg !2977
  %65 = load i32, i32* %index_arg.addr, align 4, !dbg !2978
  %add54 = add nsw i32 %mul53, %65, !dbg !2979
  %idxprom55 = sext i32 %add54 to i64, !dbg !2972
  %arrayidx56 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %61, i64 %idxprom55, !dbg !2972
  %real57 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx56, i32 0, i32 0, !dbg !2980
  store double %add51, double* %real57, align 8, !dbg !2981
  %66 = load double, double* %x11imag, align 8, !dbg !2982
  %67 = load double, double* %x21imag, align 8, !dbg !2983
  %add58 = fadd contract double %66, %67, !dbg !2984
  %68 = load %struct.dcomplex*, %struct.dcomplex** %y.addr, align 8, !dbg !2985
  %69 = load i32, i32* %i21, align 4, !dbg !2986
  %70 = load i32, i32* %k, align 4, !dbg !2987
  %add59 = add nsw i32 %69, %70, !dbg !2988
  %71 = load i32, i32* %size_arg.addr, align 4, !dbg !2989
  %mul60 = mul nsw i32 %add59, %71, !dbg !2990
  %72 = load i32, i32* %index_arg.addr, align 4, !dbg !2991
  %add61 = add nsw i32 %mul60, %72, !dbg !2992
  %idxprom62 = sext i32 %add61 to i64, !dbg !2985
  %arrayidx63 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %68, i64 %idxprom62, !dbg !2985
  %imag64 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx63, i32 0, i32 1, !dbg !2993
  store double %add58, double* %imag64, align 8, !dbg !2994
  %real65 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %u1, i32 0, i32 0, !dbg !2995
  %73 = load double, double* %real65, align 8, !dbg !2995
  %74 = load double, double* %x11real, align 8, !dbg !2996
  %75 = load double, double* %x21real, align 8, !dbg !2997
  %sub66 = fsub contract double %74, %75, !dbg !2998
  %mul67 = fmul contract double %73, %sub66, !dbg !2999
  %imag68 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %u1, i32 0, i32 1, !dbg !3000
  %76 = load double, double* %imag68, align 8, !dbg !3000
  %77 = load double, double* %x11imag, align 8, !dbg !3001
  %78 = load double, double* %x21imag, align 8, !dbg !3002
  %sub69 = fsub contract double %77, %78, !dbg !3003
  %mul70 = fmul contract double %76, %sub69, !dbg !3004
  %sub71 = fsub contract double %mul67, %mul70, !dbg !3005
  %79 = load %struct.dcomplex*, %struct.dcomplex** %y.addr, align 8, !dbg !3006
  %80 = load i32, i32* %i22, align 4, !dbg !3007
  %81 = load i32, i32* %k, align 4, !dbg !3008
  %add72 = add nsw i32 %80, %81, !dbg !3009
  %82 = load i32, i32* %size_arg.addr, align 4, !dbg !3010
  %mul73 = mul nsw i32 %add72, %82, !dbg !3011
  %83 = load i32, i32* %index_arg.addr, align 4, !dbg !3012
  %add74 = add nsw i32 %mul73, %83, !dbg !3013
  %idxprom75 = sext i32 %add74 to i64, !dbg !3006
  %arrayidx76 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %79, i64 %idxprom75, !dbg !3006
  %real77 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx76, i32 0, i32 0, !dbg !3014
  store double %sub71, double* %real77, align 8, !dbg !3015
  %real78 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %u1, i32 0, i32 0, !dbg !3016
  %84 = load double, double* %real78, align 8, !dbg !3016
  %85 = load double, double* %x11imag, align 8, !dbg !3017
  %86 = load double, double* %x21imag, align 8, !dbg !3018
  %sub79 = fsub contract double %85, %86, !dbg !3019
  %mul80 = fmul contract double %84, %sub79, !dbg !3020
  %imag81 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %u1, i32 0, i32 1, !dbg !3021
  %87 = load double, double* %imag81, align 8, !dbg !3021
  %88 = load double, double* %x11real, align 8, !dbg !3022
  %89 = load double, double* %x21real, align 8, !dbg !3023
  %sub82 = fsub contract double %88, %89, !dbg !3024
  %mul83 = fmul contract double %87, %sub82, !dbg !3025
  %add84 = fadd contract double %mul80, %mul83, !dbg !3026
  %90 = load %struct.dcomplex*, %struct.dcomplex** %y.addr, align 8, !dbg !3027
  %91 = load i32, i32* %i22, align 4, !dbg !3028
  %92 = load i32, i32* %k, align 4, !dbg !3029
  %add85 = add nsw i32 %91, %92, !dbg !3030
  %93 = load i32, i32* %size_arg.addr, align 4, !dbg !3031
  %mul86 = mul nsw i32 %add85, %93, !dbg !3032
  %94 = load i32, i32* %index_arg.addr, align 4, !dbg !3033
  %add87 = add nsw i32 %mul86, %94, !dbg !3034
  %idxprom88 = sext i32 %add87 to i64, !dbg !3027
  %arrayidx89 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %90, i64 %idxprom88, !dbg !3027
  %imag90 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx89, i32 0, i32 1, !dbg !3035
  store double %add84, double* %imag90, align 8, !dbg !3036
  br label %for.inc, !dbg !3037

for.inc:                                          ; preds = %for.body26
  %95 = load i32, i32* %k, align 4, !dbg !3038
  %inc = add nsw i32 %95, 1, !dbg !3038
  store i32 %inc, i32* %k, align 4, !dbg !3038
  br label %for.cond24, !dbg !3039, !llvm.loop !3040

for.end:                                          ; preds = %for.cond24
  br label %for.inc91, !dbg !3042

for.inc91:                                        ; preds = %for.end
  %96 = load i32, i32* %i, align 4, !dbg !3043
  %inc92 = add nsw i32 %96, 1, !dbg !3043
  store i32 %inc92, i32* %i, align 4, !dbg !3043
  br label %for.cond, !dbg !3044, !llvm.loop !3045

for.end93:                                        ; preds = %for.cond
  ret void, !dbg !3047
}

; Function Attrs: convergent noinline nounwind
define dso_local void @_Z19cffts3_gpu_kernel_1P8dcomplexS0_(%struct.dcomplex* %x_in, %struct.dcomplex* %y0) #0 !dbg !3048 {
entry:
  %x_in.addr = alloca %struct.dcomplex*, align 8
  %y0.addr = alloca %struct.dcomplex*, align 8
  %x_y_z = alloca i32, align 4
  store %struct.dcomplex* %x_in, %struct.dcomplex** %x_in.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %x_in.addr, metadata !3049, metadata !DIExpression()), !dbg !3050
  store %struct.dcomplex* %y0, %struct.dcomplex** %y0.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %y0.addr, metadata !3051, metadata !DIExpression()), !dbg !3052
  call void @llvm.dbg.declare(metadata i32* %x_y_z, metadata !3053, metadata !DIExpression()), !dbg !3054
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #11, !dbg !3055, !range !1198
  %1 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #11, !dbg !3057, !range !1243
  %mul = mul i32 %0, %1, !dbg !3059
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #11, !dbg !3060, !range !1273
  %add = add i32 %mul, %2, !dbg !3062
  store i32 %add, i32* %x_y_z, align 4, !dbg !3054
  %3 = load i32, i32* %x_y_z, align 4, !dbg !3063
  %cmp = icmp sge i32 %3, 8388608, !dbg !3065
  br i1 %cmp, label %if.then, label %if.end, !dbg !3066

if.then:                                          ; preds = %entry
  br label %return, !dbg !3067

if.end:                                           ; preds = %entry
  %4 = load %struct.dcomplex*, %struct.dcomplex** %x_in.addr, align 8, !dbg !3069
  %5 = load i32, i32* %x_y_z, align 4, !dbg !3070
  %idxprom = sext i32 %5 to i64, !dbg !3069
  %arrayidx = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %4, i64 %idxprom, !dbg !3069
  %real = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx, i32 0, i32 0, !dbg !3071
  %6 = load double, double* %real, align 8, !dbg !3071
  %7 = load %struct.dcomplex*, %struct.dcomplex** %y0.addr, align 8, !dbg !3072
  %8 = load i32, i32* %x_y_z, align 4, !dbg !3073
  %idxprom3 = sext i32 %8 to i64, !dbg !3072
  %arrayidx4 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %7, i64 %idxprom3, !dbg !3072
  %real5 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx4, i32 0, i32 0, !dbg !3074
  store double %6, double* %real5, align 8, !dbg !3075
  %9 = load %struct.dcomplex*, %struct.dcomplex** %x_in.addr, align 8, !dbg !3076
  %10 = load i32, i32* %x_y_z, align 4, !dbg !3077
  %idxprom6 = sext i32 %10 to i64, !dbg !3076
  %arrayidx7 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %9, i64 %idxprom6, !dbg !3076
  %imag = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx7, i32 0, i32 1, !dbg !3078
  %11 = load double, double* %imag, align 8, !dbg !3078
  %12 = load %struct.dcomplex*, %struct.dcomplex** %y0.addr, align 8, !dbg !3079
  %13 = load i32, i32* %x_y_z, align 4, !dbg !3080
  %idxprom8 = sext i32 %13 to i64, !dbg !3079
  %arrayidx9 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %12, i64 %idxprom8, !dbg !3079
  %imag10 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx9, i32 0, i32 1, !dbg !3081
  store double %11, double* %imag10, align 8, !dbg !3082
  br label %return, !dbg !3083

return:                                           ; preds = %if.end, %if.then
  ret void, !dbg !3083
}

; Function Attrs: convergent noinline nounwind
define dso_local void @_Z19cffts3_gpu_kernel_2iP8dcomplexS0_S0_(i32 %is, %struct.dcomplex* %gty1, %struct.dcomplex* %gty2, %struct.dcomplex* %u_device) #0 !dbg !3084 {
entry:
  %is.addr = alloca i32, align 4
  %gty1.addr = alloca %struct.dcomplex*, align 8
  %gty2.addr = alloca %struct.dcomplex*, align 8
  %u_device.addr = alloca %struct.dcomplex*, align 8
  %x_y = alloca i32, align 4
  store i32 %is, i32* %is.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %is.addr, metadata !3085, metadata !DIExpression()), !dbg !3086
  store %struct.dcomplex* %gty1, %struct.dcomplex** %gty1.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %gty1.addr, metadata !3087, metadata !DIExpression()), !dbg !3088
  store %struct.dcomplex* %gty2, %struct.dcomplex** %gty2.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %gty2.addr, metadata !3089, metadata !DIExpression()), !dbg !3090
  store %struct.dcomplex* %u_device, %struct.dcomplex** %u_device.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %u_device.addr, metadata !3091, metadata !DIExpression()), !dbg !3092
  call void @llvm.dbg.declare(metadata i32* %x_y, metadata !3093, metadata !DIExpression()), !dbg !3094
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #11, !dbg !3095, !range !1198
  %1 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #11, !dbg !3097, !range !1243
  %mul = mul i32 %0, %1, !dbg !3099
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #11, !dbg !3100, !range !1273
  %add = add i32 %mul, %2, !dbg !3102
  store i32 %add, i32* %x_y, align 4, !dbg !3094
  %3 = load i32, i32* %x_y, align 4, !dbg !3103
  %cmp = icmp sge i32 %3, 65536, !dbg !3105
  br i1 %cmp, label %if.then, label %if.end, !dbg !3106

if.then:                                          ; preds = %entry
  br label %return, !dbg !3107

if.end:                                           ; preds = %entry
  %4 = load i32, i32* %is.addr, align 4, !dbg !3109
  %call3 = call i32 @_Z12ilog2_devicei(i32 128) #4, !dbg !3110
  %5 = load %struct.dcomplex*, %struct.dcomplex** %gty1.addr, align 8, !dbg !3111
  %6 = load %struct.dcomplex*, %struct.dcomplex** %gty2.addr, align 8, !dbg !3112
  %7 = load %struct.dcomplex*, %struct.dcomplex** %u_device.addr, align 8, !dbg !3113
  %8 = load i32, i32* %x_y, align 4, !dbg !3114
  call void @_Z23cffts3_gpu_cfftz_deviceiiiP8dcomplexS0_S0_ii(i32 %4, i32 %call3, i32 128, %struct.dcomplex* %5, %struct.dcomplex* %6, %struct.dcomplex* %7, i32 %8, i32 65536) #4, !dbg !3115
  br label %return, !dbg !3116

return:                                           ; preds = %if.end, %if.then
  ret void, !dbg !3116
}

; Function Attrs: convergent noinline nounwind
define dso_local void @_Z19cffts3_gpu_kernel_3P8dcomplexS0_(%struct.dcomplex* %x_out, %struct.dcomplex* %y0) #0 !dbg !3117 {
entry:
  %x_out.addr = alloca %struct.dcomplex*, align 8
  %y0.addr = alloca %struct.dcomplex*, align 8
  %x_y_z = alloca i32, align 4
  store %struct.dcomplex* %x_out, %struct.dcomplex** %x_out.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %x_out.addr, metadata !3118, metadata !DIExpression()), !dbg !3119
  store %struct.dcomplex* %y0, %struct.dcomplex** %y0.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %y0.addr, metadata !3120, metadata !DIExpression()), !dbg !3121
  call void @llvm.dbg.declare(metadata i32* %x_y_z, metadata !3122, metadata !DIExpression()), !dbg !3123
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #11, !dbg !3124, !range !1198
  %1 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #11, !dbg !3126, !range !1243
  %mul = mul i32 %0, %1, !dbg !3128
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #11, !dbg !3129, !range !1273
  %add = add i32 %mul, %2, !dbg !3131
  store i32 %add, i32* %x_y_z, align 4, !dbg !3123
  %3 = load i32, i32* %x_y_z, align 4, !dbg !3132
  %cmp = icmp sge i32 %3, 8388608, !dbg !3134
  br i1 %cmp, label %if.then, label %if.end, !dbg !3135

if.then:                                          ; preds = %entry
  br label %return, !dbg !3136

if.end:                                           ; preds = %entry
  %4 = load %struct.dcomplex*, %struct.dcomplex** %y0.addr, align 8, !dbg !3138
  %5 = load i32, i32* %x_y_z, align 4, !dbg !3139
  %idxprom = sext i32 %5 to i64, !dbg !3138
  %arrayidx = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %4, i64 %idxprom, !dbg !3138
  %real = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx, i32 0, i32 0, !dbg !3140
  %6 = load double, double* %real, align 8, !dbg !3140
  %7 = load %struct.dcomplex*, %struct.dcomplex** %x_out.addr, align 8, !dbg !3141
  %8 = load i32, i32* %x_y_z, align 4, !dbg !3142
  %idxprom3 = sext i32 %8 to i64, !dbg !3141
  %arrayidx4 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %7, i64 %idxprom3, !dbg !3141
  %real5 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx4, i32 0, i32 0, !dbg !3143
  store double %6, double* %real5, align 8, !dbg !3144
  %9 = load %struct.dcomplex*, %struct.dcomplex** %y0.addr, align 8, !dbg !3145
  %10 = load i32, i32* %x_y_z, align 4, !dbg !3146
  %idxprom6 = sext i32 %10 to i64, !dbg !3145
  %arrayidx7 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %9, i64 %idxprom6, !dbg !3145
  %imag = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx7, i32 0, i32 1, !dbg !3147
  %11 = load double, double* %imag, align 8, !dbg !3147
  %12 = load %struct.dcomplex*, %struct.dcomplex** %x_out.addr, align 8, !dbg !3148
  %13 = load i32, i32* %x_y_z, align 4, !dbg !3149
  %idxprom8 = sext i32 %13 to i64, !dbg !3148
  %arrayidx9 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %12, i64 %idxprom8, !dbg !3148
  %imag10 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx9, i32 0, i32 1, !dbg !3150
  store double %11, double* %imag10, align 8, !dbg !3151
  br label %return, !dbg !3152

return:                                           ; preds = %if.end, %if.then
  ret void, !dbg !3152
}

; Function Attrs: convergent noinline nounwind
define dso_local void @_Z19checksum_gpu_kerneliP8dcomplexS0_(i32 %iteration, %struct.dcomplex* %u1, %struct.dcomplex* %sums) #0 !dbg !3153 {
entry:
  %iteration.addr = alloca i32, align 4
  %u1.addr = alloca %struct.dcomplex*, align 8
  %sums.addr = alloca %struct.dcomplex*, align 8
  %share_sums = alloca %struct.dcomplex*, align 8
  %j = alloca i32, align 4
  %q = alloca i32, align 4
  %r = alloca i32, align 4
  %s = alloca i32, align 4
  %ref.tmp = alloca %struct.dcomplex, align 8
  %i = alloca i32, align 4
  %ref.tmp24 = alloca %struct.dcomplex, align 8
  store i32 %iteration, i32* %iteration.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %iteration.addr, metadata !3156, metadata !DIExpression()), !dbg !3157
  store %struct.dcomplex* %u1, %struct.dcomplex** %u1.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %u1.addr, metadata !3158, metadata !DIExpression()), !dbg !3159
  store %struct.dcomplex* %sums, %struct.dcomplex** %sums.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %sums.addr, metadata !3160, metadata !DIExpression()), !dbg !3161
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %share_sums, metadata !3162, metadata !DIExpression()), !dbg !3163
  store %struct.dcomplex* addrspacecast (%struct.dcomplex addrspace(3)* bitcast ([0 x double] addrspace(3)* @extern_share_data to %struct.dcomplex addrspace(3)*) to %struct.dcomplex*), %struct.dcomplex** %share_sums, align 8, !dbg !3163
  call void @llvm.dbg.declare(metadata i32* %j, metadata !3164, metadata !DIExpression()), !dbg !3165
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #11, !dbg !3166, !range !1198
  %1 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #11, !dbg !3168, !range !1243
  %mul = mul i32 %0, %1, !dbg !3170
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #11, !dbg !3171, !range !1273
  %add = add i32 %mul, %2, !dbg !3173
  %add3 = add i32 %add, 1, !dbg !3174
  store i32 %add3, i32* %j, align 4, !dbg !3165
  call void @llvm.dbg.declare(metadata i32* %q, metadata !3175, metadata !DIExpression()), !dbg !3176
  call void @llvm.dbg.declare(metadata i32* %r, metadata !3177, metadata !DIExpression()), !dbg !3178
  call void @llvm.dbg.declare(metadata i32* %s, metadata !3179, metadata !DIExpression()), !dbg !3180
  %3 = load i32, i32* %j, align 4, !dbg !3181
  %cmp = icmp sle i32 %3, 1024, !dbg !3183
  br i1 %cmp, label %if.then, label %if.else, !dbg !3184

if.then:                                          ; preds = %entry
  %4 = load i32, i32* %j, align 4, !dbg !3185
  %rem = srem i32 %4, 256, !dbg !3187
  store i32 %rem, i32* %q, align 4, !dbg !3188
  %5 = load i32, i32* %j, align 4, !dbg !3189
  %mul4 = mul nsw i32 3, %5, !dbg !3190
  %rem5 = srem i32 %mul4, 256, !dbg !3191
  store i32 %rem5, i32* %r, align 4, !dbg !3192
  %6 = load i32, i32* %j, align 4, !dbg !3193
  %mul6 = mul nsw i32 5, %6, !dbg !3194
  %rem7 = srem i32 %mul6, 128, !dbg !3195
  store i32 %rem7, i32* %s, align 4, !dbg !3196
  %7 = load %struct.dcomplex*, %struct.dcomplex** %u1.addr, align 8, !dbg !3197
  %8 = load i32, i32* %q, align 4, !dbg !3198
  %9 = load i32, i32* %r, align 4, !dbg !3199
  %mul8 = mul nsw i32 %9, 256, !dbg !3200
  %add9 = add nsw i32 %8, %mul8, !dbg !3201
  %10 = load i32, i32* %s, align 4, !dbg !3202
  %mul10 = mul nsw i32 %10, 256, !dbg !3203
  %mul11 = mul nsw i32 %mul10, 256, !dbg !3204
  %add12 = add nsw i32 %add9, %mul11, !dbg !3205
  %idxprom = sext i32 %add12 to i64, !dbg !3197
  %arrayidx = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %7, i64 %idxprom, !dbg !3197
  %11 = load %struct.dcomplex*, %struct.dcomplex** %share_sums, align 8, !dbg !3206
  %12 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #11, !dbg !3207, !range !1273
  %idxprom14 = zext i32 %12 to i64, !dbg !3206
  %arrayidx15 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %11, i64 %idxprom14, !dbg !3206
  %13 = bitcast %struct.dcomplex* %arrayidx15 to i8*, !dbg !3209
  %14 = bitcast %struct.dcomplex* %arrayidx to i8*, !dbg !3209
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %13, i8* align 8 %14, i64 16, i1 false), !dbg !3209
  br label %if.end, !dbg !3210

if.else:                                          ; preds = %entry
  %real = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp, i32 0, i32 0, !dbg !3211
  store double 0.000000e+00, double* %real, align 8, !dbg !3211
  %imag = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp, i32 0, i32 1, !dbg !3211
  store double 0.000000e+00, double* %imag, align 8, !dbg !3211
  %15 = load %struct.dcomplex*, %struct.dcomplex** %share_sums, align 8, !dbg !3213
  %16 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #11, !dbg !3214, !range !1273
  %idxprom17 = zext i32 %16 to i64, !dbg !3213
  %arrayidx18 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %15, i64 %idxprom17, !dbg !3213
  %17 = bitcast %struct.dcomplex* %arrayidx18 to i8*, !dbg !3216
  %18 = bitcast %struct.dcomplex* %ref.tmp to i8*, !dbg !3216
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %17, i8* align 8 %18, i64 16, i1 false), !dbg !3216
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  call void @llvm.nvvm.barrier0(), !dbg !3217
  %19 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #11, !dbg !3218, !range !1273
  %cmp20 = icmp eq i32 %19, 0, !dbg !3221
  br i1 %cmp20, label %if.then21, label %if.end40, !dbg !3222

if.then21:                                        ; preds = %if.end
  call void @llvm.dbg.declare(metadata i32* %i, metadata !3223, metadata !DIExpression()), !dbg !3226
  store i32 1, i32* %i, align 4, !dbg !3226
  br label %for.cond, !dbg !3227

for.cond:                                         ; preds = %for.inc, %if.then21
  %20 = load i32, i32* %i, align 4, !dbg !3228
  %21 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #11, !dbg !3230, !range !1243
  %cmp23 = icmp ult i32 %20, %21, !dbg !3232
  br i1 %cmp23, label %for.body, label %for.end, !dbg !3233

for.body:                                         ; preds = %for.cond
  %real25 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp24, i32 0, i32 0, !dbg !3234
  %22 = load %struct.dcomplex*, %struct.dcomplex** %share_sums, align 8, !dbg !3234
  %arrayidx26 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %22, i64 0, !dbg !3234
  %real27 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx26, i32 0, i32 0, !dbg !3234
  %23 = load double, double* %real27, align 8, !dbg !3234
  %24 = load %struct.dcomplex*, %struct.dcomplex** %share_sums, align 8, !dbg !3234
  %25 = load i32, i32* %i, align 4, !dbg !3234
  %idxprom28 = sext i32 %25 to i64, !dbg !3234
  %arrayidx29 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %24, i64 %idxprom28, !dbg !3234
  %real30 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx29, i32 0, i32 0, !dbg !3234
  %26 = load double, double* %real30, align 8, !dbg !3234
  %add31 = fadd contract double %23, %26, !dbg !3234
  store double %add31, double* %real25, align 8, !dbg !3234
  %imag32 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp24, i32 0, i32 1, !dbg !3234
  %27 = load %struct.dcomplex*, %struct.dcomplex** %share_sums, align 8, !dbg !3234
  %arrayidx33 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %27, i64 0, !dbg !3234
  %imag34 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx33, i32 0, i32 1, !dbg !3234
  %28 = load double, double* %imag34, align 8, !dbg !3234
  %29 = load %struct.dcomplex*, %struct.dcomplex** %share_sums, align 8, !dbg !3234
  %30 = load i32, i32* %i, align 4, !dbg !3234
  %idxprom35 = sext i32 %30 to i64, !dbg !3234
  %arrayidx36 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %29, i64 %idxprom35, !dbg !3234
  %imag37 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx36, i32 0, i32 1, !dbg !3234
  %31 = load double, double* %imag37, align 8, !dbg !3234
  %add38 = fadd contract double %28, %31, !dbg !3234
  store double %add38, double* %imag32, align 8, !dbg !3234
  %32 = load %struct.dcomplex*, %struct.dcomplex** %share_sums, align 8, !dbg !3236
  %arrayidx39 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %32, i64 0, !dbg !3236
  %33 = bitcast %struct.dcomplex* %arrayidx39 to i8*, !dbg !3237
  %34 = bitcast %struct.dcomplex* %ref.tmp24 to i8*, !dbg !3237
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %33, i8* align 8 %34, i64 16, i1 false), !dbg !3237
  br label %for.inc, !dbg !3238

for.inc:                                          ; preds = %for.body
  %35 = load i32, i32* %i, align 4, !dbg !3239
  %inc = add nsw i32 %35, 1, !dbg !3239
  store i32 %inc, i32* %i, align 4, !dbg !3239
  br label %for.cond, !dbg !3240, !llvm.loop !3241

for.end:                                          ; preds = %for.cond
  br label %if.end40, !dbg !3243

if.end40:                                         ; preds = %for.end, %if.end
  %36 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #11, !dbg !3244, !range !1273
  %cmp42 = icmp eq i32 %36, 0, !dbg !3247
  br i1 %cmp42, label %if.then43, label %if.end65, !dbg !3248

if.then43:                                        ; preds = %if.end40
  %37 = load %struct.dcomplex*, %struct.dcomplex** %share_sums, align 8, !dbg !3249
  %arrayidx44 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %37, i64 0, !dbg !3249
  %real45 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx44, i32 0, i32 0, !dbg !3251
  %38 = load double, double* %real45, align 8, !dbg !3251
  %div = fdiv double %38, 0x4160000000000000, !dbg !3252
  %39 = load %struct.dcomplex*, %struct.dcomplex** %share_sums, align 8, !dbg !3253
  %arrayidx46 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %39, i64 0, !dbg !3253
  %real47 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx46, i32 0, i32 0, !dbg !3254
  store double %div, double* %real47, align 8, !dbg !3255
  %40 = load %struct.dcomplex*, %struct.dcomplex** %sums.addr, align 8, !dbg !3256
  %41 = load i32, i32* %iteration.addr, align 4, !dbg !3257
  %idxprom48 = sext i32 %41 to i64, !dbg !3256
  %arrayidx49 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %40, i64 %idxprom48, !dbg !3256
  %real50 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx49, i32 0, i32 0, !dbg !3258
  %42 = load %struct.dcomplex*, %struct.dcomplex** %share_sums, align 8, !dbg !3259
  %arrayidx51 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %42, i64 0, !dbg !3259
  %real52 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx51, i32 0, i32 0, !dbg !3260
  %43 = load double, double* %real52, align 8, !dbg !3260
  %call53 = call double @_ZL9atomicAddPdd(double* %real50, double %43) #4, !dbg !3261
  %44 = load %struct.dcomplex*, %struct.dcomplex** %share_sums, align 8, !dbg !3262
  %arrayidx54 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %44, i64 0, !dbg !3262
  %imag55 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx54, i32 0, i32 1, !dbg !3263
  %45 = load double, double* %imag55, align 8, !dbg !3263
  %div56 = fdiv double %45, 0x4160000000000000, !dbg !3264
  %46 = load %struct.dcomplex*, %struct.dcomplex** %share_sums, align 8, !dbg !3265
  %arrayidx57 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %46, i64 0, !dbg !3265
  %imag58 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx57, i32 0, i32 1, !dbg !3266
  store double %div56, double* %imag58, align 8, !dbg !3267
  %47 = load %struct.dcomplex*, %struct.dcomplex** %sums.addr, align 8, !dbg !3268
  %48 = load i32, i32* %iteration.addr, align 4, !dbg !3269
  %idxprom59 = sext i32 %48 to i64, !dbg !3268
  %arrayidx60 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %47, i64 %idxprom59, !dbg !3268
  %imag61 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx60, i32 0, i32 1, !dbg !3270
  %49 = load %struct.dcomplex*, %struct.dcomplex** %share_sums, align 8, !dbg !3271
  %arrayidx62 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %49, i64 0, !dbg !3271
  %imag63 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx62, i32 0, i32 1, !dbg !3272
  %50 = load double, double* %imag63, align 8, !dbg !3272
  %call64 = call double @_ZL9atomicAddPdd(double* %imag61, double %50) #4, !dbg !3273
  br label %if.end65, !dbg !3274

if.end65:                                         ; preds = %if.then43, %if.end40
  ret void, !dbg !3275
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1 immarg) #3

; Function Attrs: convergent nounwind
declare void @llvm.nvvm.barrier0() #4

; Function Attrs: convergent noinline nounwind
define internal double @_ZL9atomicAddPdd(double* %address, double %val) #5 !dbg !3276 {
entry:
  %x.addr.i13 = alloca double, align 8
  call void @llvm.dbg.declare(metadata double* %x.addr.i13, metadata !3279, metadata !DIExpression()), !dbg !3283
  %x.addr.i12 = alloca i64, align 8
  call void @llvm.dbg.declare(metadata i64* %x.addr.i12, metadata !3288, metadata !DIExpression()), !dbg !3292
  %x.addr.i11 = alloca i64, align 8
  call void @llvm.dbg.declare(metadata i64* %x.addr.i11, metadata !3288, metadata !DIExpression()), !dbg !3294
  %x.addr.i10 = alloca i64, align 8
  call void @llvm.dbg.declare(metadata i64* %x.addr.i10, metadata !3288, metadata !DIExpression()), !dbg !3297
  %x.addr.i = alloca i64, align 8
  call void @llvm.dbg.declare(metadata i64* %x.addr.i, metadata !3288, metadata !DIExpression()), !dbg !3299
  %retval = alloca double, align 8
  %address.addr = alloca double*, align 8
  %val.addr = alloca double, align 8
  %address_as_ull = alloca i64*, align 8
  %old = alloca i64, align 8
  %assumed = alloca i64, align 8
  %i = alloca i32, align 4
  store double* %address, double** %address.addr, align 8
  call void @llvm.dbg.declare(metadata double** %address.addr, metadata !3302, metadata !DIExpression()), !dbg !3303
  store double %val, double* %val.addr, align 8
  call void @llvm.dbg.declare(metadata double* %val.addr, metadata !3304, metadata !DIExpression()), !dbg !3305
  call void @llvm.dbg.declare(metadata i64** %address_as_ull, metadata !3306, metadata !DIExpression()), !dbg !3307
  %0 = load double*, double** %address.addr, align 8, !dbg !3308
  %1 = bitcast double* %0 to i64*, !dbg !3309
  store i64* %1, i64** %address_as_ull, align 8, !dbg !3307
  call void @llvm.dbg.declare(metadata i64* %old, metadata !3310, metadata !DIExpression()), !dbg !3311
  %2 = load i64*, i64** %address_as_ull, align 8, !dbg !3312
  %3 = load i64, i64* %2, align 8, !dbg !3313
  store i64 %3, i64* %old, align 8, !dbg !3311
  call void @llvm.dbg.declare(metadata i64* %assumed, metadata !3314, metadata !DIExpression()), !dbg !3315
  %4 = load double, double* %val.addr, align 8, !dbg !3316
  %cmp = fcmp oeq double %4, 0.000000e+00, !dbg !3317
  br i1 %cmp, label %if.then, label %if.end, !dbg !3318

if.then:                                          ; preds = %entry
  %5 = load i64, i64* %old, align 8, !dbg !3319
  store i64 %5, i64* %x.addr.i, align 8
  %6 = load i64, i64* %x.addr.i, align 8, !dbg !3320
  %7 = bitcast i64 %6 to double, !dbg !3321
  store double %7, double* %retval, align 8, !dbg !3322
  br label %return, !dbg !3322

if.end:                                           ; preds = %entry
  call void @llvm.dbg.declare(metadata i32* %i, metadata !3323, metadata !DIExpression()), !dbg !3324
  store i32 0, i32* %i, align 4, !dbg !3324
  br label %for.cond, !dbg !3325

for.cond:                                         ; preds = %for.inc, %if.end
  %8 = load i32, i32* %i, align 4, !dbg !3326
  %cmp1 = icmp slt i32 %8, 100000, !dbg !3327
  br i1 %cmp1, label %for.body, label %for.end, !dbg !3328

for.body:                                         ; preds = %for.cond
  %9 = load i64, i64* %old, align 8, !dbg !3329
  store i64 %9, i64* %assumed, align 8, !dbg !3330
  %10 = load i64*, i64** %address_as_ull, align 8, !dbg !3331
  %11 = load i64, i64* %assumed, align 8, !dbg !3332
  %12 = load double, double* %val.addr, align 8, !dbg !3333
  %13 = load i64, i64* %assumed, align 8, !dbg !3334
  store i64 %13, i64* %x.addr.i12, align 8
  %14 = load i64, i64* %x.addr.i12, align 8, !dbg !3335
  %15 = bitcast i64 %14 to double, !dbg !3336
  %add = fadd contract double %12, %15, !dbg !3337
  store double %add, double* %x.addr.i13, align 8
  %16 = load double, double* %x.addr.i13, align 8, !dbg !3338
  %17 = bitcast double %16 to i64, !dbg !3339
  %call4 = call i64 @_ZL9atomicCASPyyy(i64* %10, i64 %11, i64 %17) #4, !dbg !3340
  store i64 %call4, i64* %old, align 8, !dbg !3341
  %18 = load i64, i64* %assumed, align 8, !dbg !3342
  %19 = load i64, i64* %old, align 8, !dbg !3343
  %cmp5 = icmp eq i64 %18, %19, !dbg !3344
  br i1 %cmp5, label %if.then6, label %if.end8, !dbg !3345

if.then6:                                         ; preds = %for.body
  %20 = load i64, i64* %old, align 8, !dbg !3346
  store i64 %20, i64* %x.addr.i11, align 8
  %21 = load i64, i64* %x.addr.i11, align 8, !dbg !3347
  %22 = bitcast i64 %21 to double, !dbg !3348
  store double %22, double* %retval, align 8, !dbg !3349
  br label %return, !dbg !3349

if.end8:                                          ; preds = %for.body
  br label %for.inc, !dbg !3350

for.inc:                                          ; preds = %if.end8
  %23 = load i32, i32* %i, align 4, !dbg !3351
  %inc = add nsw i32 %23, 1, !dbg !3351
  store i32 %inc, i32* %i, align 4, !dbg !3351
  br label %for.cond, !dbg !3352, !llvm.loop !3353

for.end:                                          ; preds = %for.cond
  %24 = load i64, i64* %old, align 8, !dbg !3355
  store i64 %24, i64* %x.addr.i10, align 8
  %25 = load i64, i64* %x.addr.i10, align 8, !dbg !3356
  %26 = bitcast i64 %25 to double, !dbg !3357
  store double %26, double* %retval, align 8, !dbg !3358
  br label %return, !dbg !3358

return:                                           ; preds = %for.end, %if.then6, %if.then
  %27 = load double, double* %retval, align 8, !dbg !3359
  ret double %27, !dbg !3359
}

; Function Attrs: convergent noinline nounwind
define internal i64 @_ZL9atomicCASPyyy(i64* %address, i64 %compare, i64 %val) #0 !dbg !3360 {
entry:
  %p.addr.i = alloca i64*, align 8
  call void @llvm.dbg.declare(metadata i64** %p.addr.i, metadata !3364, metadata !DIExpression()), !dbg !3366
  %compare.addr.i = alloca i64, align 8
  call void @llvm.dbg.declare(metadata i64* %compare.addr.i, metadata !3368, metadata !DIExpression()), !dbg !3369
  %val.addr.i = alloca i64, align 8
  call void @llvm.dbg.declare(metadata i64* %val.addr.i, metadata !3370, metadata !DIExpression()), !dbg !3371
  %address.addr = alloca i64*, align 8
  %compare.addr = alloca i64, align 8
  %val.addr = alloca i64, align 8
  store i64* %address, i64** %address.addr, align 8
  call void @llvm.dbg.declare(metadata i64** %address.addr, metadata !3372, metadata !DIExpression()), !dbg !3373
  store i64 %compare, i64* %compare.addr, align 8
  call void @llvm.dbg.declare(metadata i64* %compare.addr, metadata !3374, metadata !DIExpression()), !dbg !3375
  store i64 %val, i64* %val.addr, align 8
  call void @llvm.dbg.declare(metadata i64* %val.addr, metadata !3376, metadata !DIExpression()), !dbg !3377
  %0 = load i64*, i64** %address.addr, align 8, !dbg !3378
  %1 = load i64, i64* %compare.addr, align 8, !dbg !3379
  %2 = load i64, i64* %val.addr, align 8, !dbg !3380
  store i64* %0, i64** %p.addr.i, align 8
  store i64 %1, i64* %compare.addr.i, align 8
  store i64 %2, i64* %val.addr.i, align 8
  %3 = load i64*, i64** %p.addr.i, align 8, !dbg !3381
  %4 = load i64, i64* %compare.addr.i, align 8, !dbg !3382
  %5 = load i64, i64* %val.addr.i, align 8, !dbg !3383
  %6 = cmpxchg i64* %3, i64 %4, i64 %5 seq_cst seq_cst, !dbg !3384
  %7 = extractvalue { i64, i1 } %6, 0, !dbg !3384
  ret i64 %7, !dbg !3385
}

; Function Attrs: convergent noinline nounwind
define dso_local void @_Z27compute_indexmap_gpu_kernelPd(double* %twiddle) #5 !dbg !3386 {
entry:
  %a.addr.i = alloca double, align 8
  call void @llvm.dbg.declare(metadata double* %a.addr.i, metadata !3389, metadata !DIExpression()), !dbg !3392
  %twiddle.addr = alloca double*, align 8
  %thread_id = alloca i32, align 4
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  %k = alloca i32, align 4
  %kk = alloca i32, align 4
  %kk2 = alloca i32, align 4
  %jj = alloca i32, align 4
  %kj2 = alloca i32, align 4
  %ii = alloca i32, align 4
  store double* %twiddle, double** %twiddle.addr, align 8
  call void @llvm.dbg.declare(metadata double** %twiddle.addr, metadata !3394, metadata !DIExpression()), !dbg !3395
  call void @llvm.dbg.declare(metadata i32* %thread_id, metadata !3396, metadata !DIExpression()), !dbg !3397
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #11, !dbg !3398, !range !1198
  %1 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #11, !dbg !3400, !range !1243
  %mul = mul i32 %0, %1, !dbg !3402
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #11, !dbg !3403, !range !1273
  %add = add i32 %mul, %2, !dbg !3405
  store i32 %add, i32* %thread_id, align 4, !dbg !3397
  %3 = load i32, i32* %thread_id, align 4, !dbg !3406
  %cmp = icmp sge i32 %3, 8388608, !dbg !3408
  br i1 %cmp, label %if.then, label %if.end, !dbg !3409

if.then:                                          ; preds = %entry
  br label %return, !dbg !3410

if.end:                                           ; preds = %entry
  call void @llvm.dbg.declare(metadata i32* %i, metadata !3412, metadata !DIExpression()), !dbg !3413
  %4 = load i32, i32* %thread_id, align 4, !dbg !3414
  %rem = srem i32 %4, 256, !dbg !3415
  store i32 %rem, i32* %i, align 4, !dbg !3413
  call void @llvm.dbg.declare(metadata i32* %j, metadata !3416, metadata !DIExpression()), !dbg !3417
  %5 = load i32, i32* %thread_id, align 4, !dbg !3418
  %div = sdiv i32 %5, 256, !dbg !3419
  %rem3 = srem i32 %div, 256, !dbg !3420
  store i32 %rem3, i32* %j, align 4, !dbg !3417
  call void @llvm.dbg.declare(metadata i32* %k, metadata !3421, metadata !DIExpression()), !dbg !3422
  %6 = load i32, i32* %thread_id, align 4, !dbg !3423
  %div4 = sdiv i32 %6, 65536, !dbg !3424
  store i32 %div4, i32* %k, align 4, !dbg !3422
  call void @llvm.dbg.declare(metadata i32* %kk, metadata !3425, metadata !DIExpression()), !dbg !3426
  call void @llvm.dbg.declare(metadata i32* %kk2, metadata !3427, metadata !DIExpression()), !dbg !3428
  call void @llvm.dbg.declare(metadata i32* %jj, metadata !3429, metadata !DIExpression()), !dbg !3430
  call void @llvm.dbg.declare(metadata i32* %kj2, metadata !3431, metadata !DIExpression()), !dbg !3432
  call void @llvm.dbg.declare(metadata i32* %ii, metadata !3433, metadata !DIExpression()), !dbg !3434
  %7 = load i32, i32* %k, align 4, !dbg !3435
  %add5 = add nsw i32 %7, 64, !dbg !3436
  %rem6 = srem i32 %add5, 128, !dbg !3437
  %sub = sub nsw i32 %rem6, 64, !dbg !3438
  store i32 %sub, i32* %kk, align 4, !dbg !3439
  %8 = load i32, i32* %kk, align 4, !dbg !3440
  %9 = load i32, i32* %kk, align 4, !dbg !3441
  %mul7 = mul nsw i32 %8, %9, !dbg !3442
  store i32 %mul7, i32* %kk2, align 4, !dbg !3443
  %10 = load i32, i32* %j, align 4, !dbg !3444
  %add8 = add nsw i32 %10, 128, !dbg !3445
  %rem9 = srem i32 %add8, 256, !dbg !3446
  %sub10 = sub nsw i32 %rem9, 128, !dbg !3447
  store i32 %sub10, i32* %jj, align 4, !dbg !3448
  %11 = load i32, i32* %jj, align 4, !dbg !3449
  %12 = load i32, i32* %jj, align 4, !dbg !3450
  %mul11 = mul nsw i32 %11, %12, !dbg !3451
  %13 = load i32, i32* %kk2, align 4, !dbg !3452
  %add12 = add nsw i32 %mul11, %13, !dbg !3453
  store i32 %add12, i32* %kj2, align 4, !dbg !3454
  %14 = load i32, i32* %i, align 4, !dbg !3455
  %add13 = add nsw i32 %14, 128, !dbg !3456
  %rem14 = srem i32 %add13, 256, !dbg !3457
  %sub15 = sub nsw i32 %rem14, 128, !dbg !3458
  store i32 %sub15, i32* %ii, align 4, !dbg !3459
  %15 = load i32, i32* %ii, align 4, !dbg !3460
  %16 = load i32, i32* %ii, align 4, !dbg !3461
  %mul16 = mul nsw i32 %15, %16, !dbg !3462
  %17 = load i32, i32* %kj2, align 4, !dbg !3463
  %add17 = add nsw i32 %mul16, %17, !dbg !3464
  %conv = sitofp i32 %add17 to double, !dbg !3465
  %mul18 = fmul contract double 0xBF04B2B4199E149A, %conv, !dbg !3466
  store double %mul18, double* %a.addr.i, align 8
  %18 = load double, double* %a.addr.i, align 8, !dbg !3467
  %19 = call i32 @llvm.nvvm.d2i.hi(double %18) #11, !dbg !3468
  %20 = bitcast i32 %19 to float, !dbg !3468
  %21 = call float @llvm.nvvm.fabs.f(float %20) #11, !dbg !3468
  %22 = fcmp olt float %21, 0x4010E92220000000, !dbg !3468
  %23 = zext i1 %22 to i32, !dbg !3468
  br i1 %22, label %24, label %57, !dbg !3468

24:                                               ; preds = %if.end
  %25 = call double @llvm.nvvm.mul.rn.d(double %18, double 0x3FF71547652B82FE) #11, !dbg !3468
  %26 = call double @llvm.nvvm.add.rn.d(double %25, double 0x4338000000000000) #11, !dbg !3468
  %27 = call i32 @llvm.nvvm.d2i.lo(double %26) #11, !dbg !3468
  %28 = call double @llvm.nvvm.add.rn.d(double %25, double 0x4338000000000000) #11, !dbg !3468
  %29 = call double @llvm.nvvm.add.rn.d(double %28, double 0xC338000000000000) #11, !dbg !3468
  %30 = call double @llvm.nvvm.fma.rn.d(double %29, double 0xBFE62E42FEFA39EF, double %18) #11, !dbg !3468
  %31 = call double @llvm.nvvm.fma.rn.d(double %29, double 0xBC7ABC9E3B39803F, double %30) #11, !dbg !3468
  %32 = call double @llvm.nvvm.fma.rn.d(double 0x3E5ADE1569CE2BDF, double %31, double 0x3E928AF3FCA213EA) #11, !dbg !3468
  %33 = call double @llvm.nvvm.fma.rn.d(double %32, double %31, double 0x3EC71DEE62401315) #11, !dbg !3468
  %34 = call double @llvm.nvvm.fma.rn.d(double %33, double %31, double 0x3EFA01997C89EB71) #11, !dbg !3468
  %35 = call double @llvm.nvvm.fma.rn.d(double %34, double %31, double 0x3F2A01A014761F65) #11, !dbg !3468
  %36 = call double @llvm.nvvm.fma.rn.d(double %35, double %31, double 0x3F56C16C1852B7AF) #11, !dbg !3468
  %37 = call double @llvm.nvvm.fma.rn.d(double %36, double %31, double 0x3F81111111122322) #11, !dbg !3468
  %38 = call double @llvm.nvvm.fma.rn.d(double %37, double %31, double 0x3FA55555555502A1) #11, !dbg !3468
  %39 = call double @llvm.nvvm.fma.rn.d(double %38, double %31, double 0x3FC5555555555511) #11, !dbg !3468
  %40 = call double @llvm.nvvm.fma.rn.d(double %39, double %31, double 0x3FE000000000000B) #11, !dbg !3468
  %41 = call double @llvm.nvvm.fma.rn.d(double %40, double %31, double 1.000000e+00) #11, !dbg !3468
  %42 = call double @llvm.nvvm.fma.rn.d(double %41, double %31, double 1.000000e+00) #11, !dbg !3468
  %neg.i.i = sub i32 0, %27, !dbg !3468
  %abs.cond.i.i = icmp sge i32 %27, 0, !dbg !3468
  %abs.i.i = select i1 %abs.cond.i.i, i32 %27, i32 %neg.i.i, !dbg !3468
  %43 = icmp slt i32 %abs.i.i, 1023, !dbg !3468
  br i1 %43, label %44, label %47, !dbg !3468

44:                                               ; preds = %24
  %45 = shl i32 %27, 20, !dbg !3468
  %46 = add nsw i32 %45, 1072693248, !dbg !3468
  br label %__internal_exp_kernel.exit.i.i, !dbg !3468

47:                                               ; preds = %24
  %48 = add nsw i32 %27, 2046, !dbg !3468
  %49 = udiv i32 %48, 2, !dbg !3468
  %50 = shl i32 %49, 20, !dbg !3468
  %51 = shl i32 %48, 20, !dbg !3468
  %52 = sub i32 %51, %50, !dbg !3468
  %53 = call double @llvm.nvvm.lohi.i2d(i32 0, i32 %50) #11, !dbg !3468
  %54 = fmul double %42, %53, !dbg !3468
  br label %__internal_exp_kernel.exit.i.i, !dbg !3468

__internal_exp_kernel.exit.i.i:                   ; preds = %47, %44
  %a.addr.0.i.i.i.i = phi double [ %42, %44 ], [ %54, %47 ], !dbg !3468
  %k.0.i.i.i.i = phi i32 [ %46, %44 ], [ %52, %47 ], !dbg !3468
  %55 = call double @llvm.nvvm.lohi.i2d(i32 0, i32 %k.0.i.i.i.i) #11, !dbg !3468
  %56 = fmul double %a.addr.0.i.i.i.i, %55, !dbg !3468
  br label %_ZL3expd.exit, !dbg !3468

57:                                               ; preds = %if.end
  %58 = icmp slt i32 %19, 0, !dbg !3468
  br i1 %58, label %59, label %60, !dbg !3468

59:                                               ; preds = %57
  br label %61, !dbg !3468

60:                                               ; preds = %57
  br label %61, !dbg !3468

61:                                               ; preds = %60, %59
  %62 = phi double [ 0.000000e+00, %59 ], [ 0x7FF0000000000000, %60 ], !dbg !3468
  %63 = call double @llvm.nvvm.fabs.d(double %18) #11, !dbg !3468
  %64 = fcmp ole double %63, 0x7FF0000000000000, !dbg !3468
  %65 = xor i1 %64, true, !dbg !3468
  %66 = zext i1 %65 to i32, !dbg !3468
  br i1 %65, label %67, label %69, !dbg !3468

67:                                               ; preds = %61
  %68 = fadd double %18, %18, !dbg !3468
  br label %69, !dbg !3468

69:                                               ; preds = %67, %61
  %t.0.i.i = phi double [ %68, %67 ], [ %62, %61 ], !dbg !3468
  br label %_ZL3expd.exit, !dbg !3468

_ZL3expd.exit:                                    ; preds = %69, %__internal_exp_kernel.exit.i.i
  %t.1.i.i = phi double [ %56, %__internal_exp_kernel.exit.i.i ], [ %t.0.i.i, %69 ], !dbg !3468
  %70 = load double*, double** %twiddle.addr, align 8, !dbg !3469
  %71 = load i32, i32* %thread_id, align 4, !dbg !3470
  %idxprom = sext i32 %71 to i64, !dbg !3469
  %arrayidx = getelementptr inbounds double, double* %70, i64 %idxprom, !dbg !3469
  store double %t.1.i.i, double* %arrayidx, align 8, !dbg !3471
  br label %return, !dbg !3472

return:                                           ; preds = %_ZL3expd.exit, %if.then
  ret void, !dbg !3472
}

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.d2i.hi(double) #2

; Function Attrs: nounwind readnone
declare float @llvm.nvvm.fabs.f(float) #2

; Function Attrs: nounwind readnone
declare double @llvm.nvvm.mul.rn.d(double, double) #2

; Function Attrs: nounwind readnone
declare double @llvm.nvvm.add.rn.d(double, double) #2

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.d2i.lo(double) #2

; Function Attrs: nounwind readnone
declare double @llvm.nvvm.fma.rn.d(double, double, double) #2

; Function Attrs: nounwind readnone
declare double @llvm.nvvm.lohi.i2d(i32, i32) #2

; Function Attrs: nounwind readnone
declare double @llvm.nvvm.fabs.d(double) #2

; Function Attrs: convergent noinline nounwind
define dso_local void @_Z37compute_initial_conditions_gpu_kernelP8dcomplexPd(%struct.dcomplex* %u0, double* %starts) #0 !dbg !3473 {
entry:
  %u0.addr = alloca %struct.dcomplex*, align 8
  %starts.addr = alloca double*, align 8
  %z = alloca i32, align 4
  %x0 = alloca double, align 8
  %y = alloca i32, align 4
  store %struct.dcomplex* %u0, %struct.dcomplex** %u0.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %u0.addr, metadata !3476, metadata !DIExpression()), !dbg !3477
  store double* %starts, double** %starts.addr, align 8
  call void @llvm.dbg.declare(metadata double** %starts.addr, metadata !3478, metadata !DIExpression()), !dbg !3479
  call void @llvm.dbg.declare(metadata i32* %z, metadata !3480, metadata !DIExpression()), !dbg !3481
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #11, !dbg !3482, !range !1198
  %1 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #11, !dbg !3484, !range !1243
  %mul = mul i32 %0, %1, !dbg !3486
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #11, !dbg !3487, !range !1273
  %add = add i32 %mul, %2, !dbg !3489
  store i32 %add, i32* %z, align 4, !dbg !3481
  %3 = load i32, i32* %z, align 4, !dbg !3490
  %cmp = icmp sge i32 %3, 128, !dbg !3492
  br i1 %cmp, label %if.then, label %if.end, !dbg !3493

if.then:                                          ; preds = %entry
  br label %for.end, !dbg !3494

if.end:                                           ; preds = %entry
  call void @llvm.dbg.declare(metadata double* %x0, metadata !3496, metadata !DIExpression()), !dbg !3497
  %4 = load double*, double** %starts.addr, align 8, !dbg !3498
  %5 = load i32, i32* %z, align 4, !dbg !3499
  %idxprom = sext i32 %5 to i64, !dbg !3498
  %arrayidx = getelementptr inbounds double, double* %4, i64 %idxprom, !dbg !3498
  %6 = load double, double* %arrayidx, align 8, !dbg !3498
  store double %6, double* %x0, align 8, !dbg !3497
  call void @llvm.dbg.declare(metadata i32* %y, metadata !3500, metadata !DIExpression()), !dbg !3502
  store i32 0, i32* %y, align 4, !dbg !3502
  br label %for.cond, !dbg !3503

for.cond:                                         ; preds = %for.inc, %if.end
  %7 = load i32, i32* %y, align 4, !dbg !3504
  %cmp3 = icmp slt i32 %7, 256, !dbg !3506
  br i1 %cmp3, label %for.body, label %for.end, !dbg !3507

for.body:                                         ; preds = %for.cond
  %8 = load %struct.dcomplex*, %struct.dcomplex** %u0.addr, align 8, !dbg !3508
  %9 = load i32, i32* %y, align 4, !dbg !3510
  %mul4 = mul nsw i32 %9, 256, !dbg !3511
  %add5 = add nsw i32 0, %mul4, !dbg !3512
  %10 = load i32, i32* %z, align 4, !dbg !3513
  %mul6 = mul nsw i32 %10, 256, !dbg !3514
  %mul7 = mul nsw i32 %mul6, 256, !dbg !3515
  %add8 = add nsw i32 %add5, %mul7, !dbg !3516
  %idxprom9 = sext i32 %add8 to i64, !dbg !3508
  %arrayidx10 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %8, i64 %idxprom9, !dbg !3508
  %11 = bitcast %struct.dcomplex* %arrayidx10 to double*, !dbg !3517
  call void @_Z13vranlc_deviceiPddS_(i32 512, double* %x0, double 0x41D2309CE5400000, double* %11) #4, !dbg !3518
  br label %for.inc, !dbg !3519

for.inc:                                          ; preds = %for.body
  %12 = load i32, i32* %y, align 4, !dbg !3520
  %inc = add nsw i32 %12, 1, !dbg !3520
  store i32 %inc, i32* %y, align 4, !dbg !3520
  br label %for.cond, !dbg !3521, !llvm.loop !3522

for.end:                                          ; preds = %for.cond, %if.then
  ret void, !dbg !3524
}

; Function Attrs: convergent noinline nounwind
define dso_local void @_Z13vranlc_deviceiPddS_(i32 %n, double* %x_seed, double %a, double* %y) #0 !dbg !3525 {
entry:
  %n.addr = alloca i32, align 4
  %x_seed.addr = alloca double*, align 8
  %a.addr = alloca double, align 8
  %y.addr = alloca double*, align 8
  %i = alloca i32, align 4
  %x = alloca double, align 8
  %t1 = alloca double, align 8
  %t2 = alloca double, align 8
  %t3 = alloca double, align 8
  %t4 = alloca double, align 8
  %a1 = alloca double, align 8
  %a2 = alloca double, align 8
  %x1 = alloca double, align 8
  %x2 = alloca double, align 8
  %z = alloca double, align 8
  store i32 %n, i32* %n.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %n.addr, metadata !3528, metadata !DIExpression()), !dbg !3529
  store double* %x_seed, double** %x_seed.addr, align 8
  call void @llvm.dbg.declare(metadata double** %x_seed.addr, metadata !3530, metadata !DIExpression()), !dbg !3531
  store double %a, double* %a.addr, align 8
  call void @llvm.dbg.declare(metadata double* %a.addr, metadata !3532, metadata !DIExpression()), !dbg !3533
  store double* %y, double** %y.addr, align 8
  call void @llvm.dbg.declare(metadata double** %y.addr, metadata !3534, metadata !DIExpression()), !dbg !3535
  call void @llvm.dbg.declare(metadata i32* %i, metadata !3536, metadata !DIExpression()), !dbg !3537
  call void @llvm.dbg.declare(metadata double* %x, metadata !3538, metadata !DIExpression()), !dbg !3539
  call void @llvm.dbg.declare(metadata double* %t1, metadata !3540, metadata !DIExpression()), !dbg !3541
  call void @llvm.dbg.declare(metadata double* %t2, metadata !3542, metadata !DIExpression()), !dbg !3543
  call void @llvm.dbg.declare(metadata double* %t3, metadata !3544, metadata !DIExpression()), !dbg !3545
  call void @llvm.dbg.declare(metadata double* %t4, metadata !3546, metadata !DIExpression()), !dbg !3547
  call void @llvm.dbg.declare(metadata double* %a1, metadata !3548, metadata !DIExpression()), !dbg !3549
  call void @llvm.dbg.declare(metadata double* %a2, metadata !3550, metadata !DIExpression()), !dbg !3551
  call void @llvm.dbg.declare(metadata double* %x1, metadata !3552, metadata !DIExpression()), !dbg !3553
  call void @llvm.dbg.declare(metadata double* %x2, metadata !3554, metadata !DIExpression()), !dbg !3555
  call void @llvm.dbg.declare(metadata double* %z, metadata !3556, metadata !DIExpression()), !dbg !3557
  %0 = load double, double* %a.addr, align 8, !dbg !3558
  %mul = fmul contract double 0x3E80000000000000, %0, !dbg !3559
  store double %mul, double* %t1, align 8, !dbg !3560
  %1 = load double, double* %t1, align 8, !dbg !3561
  %conv = fptosi double %1 to i32, !dbg !3561
  %conv1 = sitofp i32 %conv to double, !dbg !3562
  store double %conv1, double* %a1, align 8, !dbg !3563
  %2 = load double, double* %a.addr, align 8, !dbg !3564
  %3 = load double, double* %a1, align 8, !dbg !3565
  %mul2 = fmul contract double 0x4160000000000000, %3, !dbg !3566
  %sub = fsub contract double %2, %mul2, !dbg !3567
  store double %sub, double* %a2, align 8, !dbg !3568
  %4 = load double*, double** %x_seed.addr, align 8, !dbg !3569
  %5 = load double, double* %4, align 8, !dbg !3570
  store double %5, double* %x, align 8, !dbg !3571
  store i32 0, i32* %i, align 4, !dbg !3572
  br label %for.cond, !dbg !3574

for.cond:                                         ; preds = %for.inc, %entry
  %6 = load i32, i32* %i, align 4, !dbg !3575
  %7 = load i32, i32* %n.addr, align 4, !dbg !3577
  %cmp = icmp slt i32 %6, %7, !dbg !3578
  br i1 %cmp, label %for.body, label %for.end, !dbg !3579

for.body:                                         ; preds = %for.cond
  %8 = load double, double* %x, align 8, !dbg !3580
  %mul3 = fmul contract double 0x3E80000000000000, %8, !dbg !3582
  store double %mul3, double* %t1, align 8, !dbg !3583
  %9 = load double, double* %t1, align 8, !dbg !3584
  %conv4 = fptosi double %9 to i32, !dbg !3584
  %conv5 = sitofp i32 %conv4 to double, !dbg !3585
  store double %conv5, double* %x1, align 8, !dbg !3586
  %10 = load double, double* %x, align 8, !dbg !3587
  %11 = load double, double* %x1, align 8, !dbg !3588
  %mul6 = fmul contract double 0x4160000000000000, %11, !dbg !3589
  %sub7 = fsub contract double %10, %mul6, !dbg !3590
  store double %sub7, double* %x2, align 8, !dbg !3591
  %12 = load double, double* %a1, align 8, !dbg !3592
  %13 = load double, double* %x2, align 8, !dbg !3593
  %mul8 = fmul contract double %12, %13, !dbg !3594
  %14 = load double, double* %a2, align 8, !dbg !3595
  %15 = load double, double* %x1, align 8, !dbg !3596
  %mul9 = fmul contract double %14, %15, !dbg !3597
  %add = fadd contract double %mul8, %mul9, !dbg !3598
  store double %add, double* %t1, align 8, !dbg !3599
  %16 = load double, double* %t1, align 8, !dbg !3600
  %mul10 = fmul contract double 0x3E80000000000000, %16, !dbg !3601
  %conv11 = fptosi double %mul10 to i32, !dbg !3602
  %conv12 = sitofp i32 %conv11 to double, !dbg !3603
  store double %conv12, double* %t2, align 8, !dbg !3604
  %17 = load double, double* %t1, align 8, !dbg !3605
  %18 = load double, double* %t2, align 8, !dbg !3606
  %mul13 = fmul contract double 0x4160000000000000, %18, !dbg !3607
  %sub14 = fsub contract double %17, %mul13, !dbg !3608
  store double %sub14, double* %z, align 8, !dbg !3609
  %19 = load double, double* %z, align 8, !dbg !3610
  %mul15 = fmul contract double 0x4160000000000000, %19, !dbg !3611
  %20 = load double, double* %a2, align 8, !dbg !3612
  %21 = load double, double* %x2, align 8, !dbg !3613
  %mul16 = fmul contract double %20, %21, !dbg !3614
  %add17 = fadd contract double %mul15, %mul16, !dbg !3615
  store double %add17, double* %t3, align 8, !dbg !3616
  %22 = load double, double* %t3, align 8, !dbg !3617
  %mul18 = fmul contract double 0x3D10000000000000, %22, !dbg !3618
  %conv19 = fptosi double %mul18 to i32, !dbg !3619
  %conv20 = sitofp i32 %conv19 to double, !dbg !3620
  store double %conv20, double* %t4, align 8, !dbg !3621
  %23 = load double, double* %t3, align 8, !dbg !3622
  %24 = load double, double* %t4, align 8, !dbg !3623
  %mul21 = fmul contract double 0x42D0000000000000, %24, !dbg !3624
  %sub22 = fsub contract double %23, %mul21, !dbg !3625
  store double %sub22, double* %x, align 8, !dbg !3626
  %25 = load double, double* %x, align 8, !dbg !3627
  %mul23 = fmul contract double 0x3D10000000000000, %25, !dbg !3628
  %26 = load double*, double** %y.addr, align 8, !dbg !3629
  %27 = load i32, i32* %i, align 4, !dbg !3630
  %idxprom = sext i32 %27 to i64, !dbg !3629
  %arrayidx = getelementptr inbounds double, double* %26, i64 %idxprom, !dbg !3629
  store double %mul23, double* %arrayidx, align 8, !dbg !3631
  br label %for.inc, !dbg !3632

for.inc:                                          ; preds = %for.body
  %28 = load i32, i32* %i, align 4, !dbg !3633
  %inc = add nsw i32 %28, 1, !dbg !3633
  store i32 %inc, i32* %i, align 4, !dbg !3633
  br label %for.cond, !dbg !3634, !llvm.loop !3635

for.end:                                          ; preds = %for.cond
  %29 = load double, double* %x, align 8, !dbg !3637
  %30 = load double*, double** %x_seed.addr, align 8, !dbg !3638
  store double %29, double* %30, align 8, !dbg !3639
  ret void, !dbg !3640
}

; Function Attrs: convergent noinline nounwind
define dso_local void @_Z17evolve_gpu_kernelP8dcomplexS0_Pd(%struct.dcomplex* %u0, %struct.dcomplex* %u1, double* %twiddle) #0 !dbg !3641 {
entry:
  %u0.addr = alloca %struct.dcomplex*, align 8
  %u1.addr = alloca %struct.dcomplex*, align 8
  %twiddle.addr = alloca double*, align 8
  %thread_id = alloca i32, align 4
  %ref.tmp = alloca %struct.dcomplex, align 8
  store %struct.dcomplex* %u0, %struct.dcomplex** %u0.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %u0.addr, metadata !3644, metadata !DIExpression()), !dbg !3645
  store %struct.dcomplex* %u1, %struct.dcomplex** %u1.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %u1.addr, metadata !3646, metadata !DIExpression()), !dbg !3647
  store double* %twiddle, double** %twiddle.addr, align 8
  call void @llvm.dbg.declare(metadata double** %twiddle.addr, metadata !3648, metadata !DIExpression()), !dbg !3649
  call void @llvm.dbg.declare(metadata i32* %thread_id, metadata !3650, metadata !DIExpression()), !dbg !3651
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #11, !dbg !3652, !range !1198
  %1 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #11, !dbg !3654, !range !1243
  %mul = mul i32 %0, %1, !dbg !3656
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #11, !dbg !3657, !range !1273
  %add = add i32 %mul, %2, !dbg !3659
  store i32 %add, i32* %thread_id, align 4, !dbg !3651
  %3 = load i32, i32* %thread_id, align 4, !dbg !3660
  %cmp = icmp sge i32 %3, 8388608, !dbg !3662
  br i1 %cmp, label %if.then, label %if.end, !dbg !3663

if.then:                                          ; preds = %entry
  br label %return, !dbg !3664

if.end:                                           ; preds = %entry
  %real = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp, i32 0, i32 0, !dbg !3666
  %4 = load %struct.dcomplex*, %struct.dcomplex** %u0.addr, align 8, !dbg !3666
  %5 = load i32, i32* %thread_id, align 4, !dbg !3666
  %idxprom = sext i32 %5 to i64, !dbg !3666
  %arrayidx = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %4, i64 %idxprom, !dbg !3666
  %real3 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx, i32 0, i32 0, !dbg !3666
  %6 = load double, double* %real3, align 8, !dbg !3666
  %7 = load double*, double** %twiddle.addr, align 8, !dbg !3666
  %8 = load i32, i32* %thread_id, align 4, !dbg !3666
  %idxprom4 = sext i32 %8 to i64, !dbg !3666
  %arrayidx5 = getelementptr inbounds double, double* %7, i64 %idxprom4, !dbg !3666
  %9 = load double, double* %arrayidx5, align 8, !dbg !3666
  %mul6 = fmul contract double %6, %9, !dbg !3666
  store double %mul6, double* %real, align 8, !dbg !3666
  %imag = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp, i32 0, i32 1, !dbg !3666
  %10 = load %struct.dcomplex*, %struct.dcomplex** %u0.addr, align 8, !dbg !3666
  %11 = load i32, i32* %thread_id, align 4, !dbg !3666
  %idxprom7 = sext i32 %11 to i64, !dbg !3666
  %arrayidx8 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %10, i64 %idxprom7, !dbg !3666
  %imag9 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx8, i32 0, i32 1, !dbg !3666
  %12 = load double, double* %imag9, align 8, !dbg !3666
  %13 = load double*, double** %twiddle.addr, align 8, !dbg !3666
  %14 = load i32, i32* %thread_id, align 4, !dbg !3666
  %idxprom10 = sext i32 %14 to i64, !dbg !3666
  %arrayidx11 = getelementptr inbounds double, double* %13, i64 %idxprom10, !dbg !3666
  %15 = load double, double* %arrayidx11, align 8, !dbg !3666
  %mul12 = fmul contract double %12, %15, !dbg !3666
  store double %mul12, double* %imag, align 8, !dbg !3666
  %16 = load %struct.dcomplex*, %struct.dcomplex** %u0.addr, align 8, !dbg !3667
  %17 = load i32, i32* %thread_id, align 4, !dbg !3668
  %idxprom13 = sext i32 %17 to i64, !dbg !3667
  %arrayidx14 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %16, i64 %idxprom13, !dbg !3667
  %18 = bitcast %struct.dcomplex* %arrayidx14 to i8*, !dbg !3669
  %19 = bitcast %struct.dcomplex* %ref.tmp to i8*, !dbg !3669
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %18, i8* align 8 %19, i64 16, i1 false), !dbg !3669
  %20 = load %struct.dcomplex*, %struct.dcomplex** %u0.addr, align 8, !dbg !3670
  %21 = load i32, i32* %thread_id, align 4, !dbg !3671
  %idxprom15 = sext i32 %21 to i64, !dbg !3670
  %arrayidx16 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %20, i64 %idxprom15, !dbg !3670
  %22 = load %struct.dcomplex*, %struct.dcomplex** %u1.addr, align 8, !dbg !3672
  %23 = load i32, i32* %thread_id, align 4, !dbg !3673
  %idxprom17 = sext i32 %23 to i64, !dbg !3672
  %arrayidx18 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %22, i64 %idxprom17, !dbg !3672
  %24 = bitcast %struct.dcomplex* %arrayidx18 to i8*, !dbg !3674
  %25 = bitcast %struct.dcomplex* %arrayidx16 to i8*, !dbg !3674
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %24, i8* align 8 %25, i64 16, i1 false), !dbg !3674
  br label %return, !dbg !3675

return:                                           ; preds = %if.end, %if.then
  ret void, !dbg !3675
}

; Function Attrs: convergent noinline nounwind
define dso_local void @_Z18init_ui_gpu_kernelP8dcomplexS0_Pd(%struct.dcomplex* %u0, %struct.dcomplex* %u1, double* %twiddle) #0 !dbg !3676 {
entry:
  %u0.addr = alloca %struct.dcomplex*, align 8
  %u1.addr = alloca %struct.dcomplex*, align 8
  %twiddle.addr = alloca double*, align 8
  %thread_id = alloca i32, align 4
  %ref.tmp = alloca %struct.dcomplex, align 8
  %ref.tmp3 = alloca %struct.dcomplex, align 8
  store %struct.dcomplex* %u0, %struct.dcomplex** %u0.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %u0.addr, metadata !3677, metadata !DIExpression()), !dbg !3678
  store %struct.dcomplex* %u1, %struct.dcomplex** %u1.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %u1.addr, metadata !3679, metadata !DIExpression()), !dbg !3680
  store double* %twiddle, double** %twiddle.addr, align 8
  call void @llvm.dbg.declare(metadata double** %twiddle.addr, metadata !3681, metadata !DIExpression()), !dbg !3682
  call void @llvm.dbg.declare(metadata i32* %thread_id, metadata !3683, metadata !DIExpression()), !dbg !3684
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #11, !dbg !3685, !range !1198
  %1 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #11, !dbg !3687, !range !1243
  %mul = mul i32 %0, %1, !dbg !3689
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #11, !dbg !3690, !range !1273
  %add = add i32 %mul, %2, !dbg !3692
  store i32 %add, i32* %thread_id, align 4, !dbg !3684
  %3 = load i32, i32* %thread_id, align 4, !dbg !3693
  %cmp = icmp sge i32 %3, 8388608, !dbg !3695
  br i1 %cmp, label %if.then, label %if.end, !dbg !3696

if.then:                                          ; preds = %entry
  br label %return, !dbg !3697

if.end:                                           ; preds = %entry
  %real = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp, i32 0, i32 0, !dbg !3699
  store double 0.000000e+00, double* %real, align 8, !dbg !3699
  %imag = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp, i32 0, i32 1, !dbg !3699
  store double 0.000000e+00, double* %imag, align 8, !dbg !3699
  %4 = load %struct.dcomplex*, %struct.dcomplex** %u0.addr, align 8, !dbg !3700
  %5 = load i32, i32* %thread_id, align 4, !dbg !3701
  %idxprom = sext i32 %5 to i64, !dbg !3700
  %arrayidx = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %4, i64 %idxprom, !dbg !3700
  %6 = bitcast %struct.dcomplex* %arrayidx to i8*, !dbg !3702
  %7 = bitcast %struct.dcomplex* %ref.tmp to i8*, !dbg !3702
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %6, i8* align 8 %7, i64 16, i1 false), !dbg !3702
  %real4 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp3, i32 0, i32 0, !dbg !3703
  store double 0.000000e+00, double* %real4, align 8, !dbg !3703
  %imag5 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp3, i32 0, i32 1, !dbg !3703
  store double 0.000000e+00, double* %imag5, align 8, !dbg !3703
  %8 = load %struct.dcomplex*, %struct.dcomplex** %u1.addr, align 8, !dbg !3704
  %9 = load i32, i32* %thread_id, align 4, !dbg !3705
  %idxprom6 = sext i32 %9 to i64, !dbg !3704
  %arrayidx7 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %8, i64 %idxprom6, !dbg !3704
  %10 = bitcast %struct.dcomplex* %arrayidx7 to i8*, !dbg !3706
  %11 = bitcast %struct.dcomplex* %ref.tmp3 to i8*, !dbg !3706
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %10, i8* align 8 %11, i64 16, i1 false), !dbg !3706
  %12 = load double*, double** %twiddle.addr, align 8, !dbg !3707
  %13 = load i32, i32* %thread_id, align 4, !dbg !3708
  %idxprom8 = sext i32 %13 to i64, !dbg !3707
  %arrayidx9 = getelementptr inbounds double, double* %12, i64 %idxprom8, !dbg !3707
  store double 0.000000e+00, double* %arrayidx9, align 8, !dbg !3709
  br label %return, !dbg !3710

return:                                           ; preds = %if.end, %if.then
  ret void, !dbg !3710
}

; Function Attrs: convergent noinline nounwind
define dso_local void @_Z13ipow46_devicediPd(double %a, i32 %exponent, double* %result) #0 !dbg !3711 {
entry:
  %a.addr = alloca double, align 8
  %exponent.addr = alloca i32, align 4
  %result.addr = alloca double*, align 8
  %q = alloca double, align 8
  %r = alloca double, align 8
  %n = alloca i32, align 4
  %n2 = alloca i32, align 4
  store double %a, double* %a.addr, align 8
  call void @llvm.dbg.declare(metadata double* %a.addr, metadata !3714, metadata !DIExpression()), !dbg !3715
  store i32 %exponent, i32* %exponent.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %exponent.addr, metadata !3716, metadata !DIExpression()), !dbg !3717
  store double* %result, double** %result.addr, align 8
  call void @llvm.dbg.declare(metadata double** %result.addr, metadata !3718, metadata !DIExpression()), !dbg !3719
  call void @llvm.dbg.declare(metadata double* %q, metadata !3720, metadata !DIExpression()), !dbg !3721
  call void @llvm.dbg.declare(metadata double* %r, metadata !3722, metadata !DIExpression()), !dbg !3723
  call void @llvm.dbg.declare(metadata i32* %n, metadata !3724, metadata !DIExpression()), !dbg !3725
  call void @llvm.dbg.declare(metadata i32* %n2, metadata !3726, metadata !DIExpression()), !dbg !3727
  %0 = load double*, double** %result.addr, align 8, !dbg !3728
  store double 1.000000e+00, double* %0, align 8, !dbg !3729
  %1 = load i32, i32* %exponent.addr, align 4, !dbg !3730
  %cmp = icmp eq i32 %1, 0, !dbg !3732
  br i1 %cmp, label %if.then, label %if.end, !dbg !3733

if.then:                                          ; preds = %entry
  br label %return, !dbg !3734

if.end:                                           ; preds = %entry
  %2 = load double, double* %a.addr, align 8, !dbg !3736
  store double %2, double* %q, align 8, !dbg !3737
  store double 1.000000e+00, double* %r, align 8, !dbg !3738
  %3 = load i32, i32* %exponent.addr, align 4, !dbg !3739
  store i32 %3, i32* %n, align 4, !dbg !3740
  br label %while.cond, !dbg !3741

while.cond:                                       ; preds = %if.end5, %if.end
  %4 = load i32, i32* %n, align 4, !dbg !3742
  %cmp1 = icmp sgt i32 %4, 1, !dbg !3743
  br i1 %cmp1, label %while.body, label %while.end, !dbg !3741

while.body:                                       ; preds = %while.cond
  %5 = load i32, i32* %n, align 4, !dbg !3744
  %div = sdiv i32 %5, 2, !dbg !3746
  store i32 %div, i32* %n2, align 4, !dbg !3747
  %6 = load i32, i32* %n2, align 4, !dbg !3748
  %mul = mul nsw i32 %6, 2, !dbg !3750
  %7 = load i32, i32* %n, align 4, !dbg !3751
  %cmp2 = icmp eq i32 %mul, %7, !dbg !3752
  br i1 %cmp2, label %if.then3, label %if.else, !dbg !3753

if.then3:                                         ; preds = %while.body
  %8 = load double, double* %q, align 8, !dbg !3754
  %call = call double @_Z13randlc_devicePdd(double* %q, double %8) #4, !dbg !3756
  %9 = load i32, i32* %n2, align 4, !dbg !3757
  store i32 %9, i32* %n, align 4, !dbg !3758
  br label %if.end5, !dbg !3759

if.else:                                          ; preds = %while.body
  %10 = load double, double* %q, align 8, !dbg !3760
  %call4 = call double @_Z13randlc_devicePdd(double* %r, double %10) #4, !dbg !3762
  %11 = load i32, i32* %n, align 4, !dbg !3763
  %sub = sub nsw i32 %11, 1, !dbg !3764
  store i32 %sub, i32* %n, align 4, !dbg !3765
  br label %if.end5

if.end5:                                          ; preds = %if.else, %if.then3
  br label %while.cond, !dbg !3741, !llvm.loop !3766

while.end:                                        ; preds = %while.cond
  %12 = load double, double* %q, align 8, !dbg !3768
  %call6 = call double @_Z13randlc_devicePdd(double* %r, double %12) #4, !dbg !3769
  %13 = load double, double* %r, align 8, !dbg !3770
  %14 = load double*, double** %result.addr, align 8, !dbg !3771
  store double %13, double* %14, align 8, !dbg !3772
  br label %return, !dbg !3773

return:                                           ; preds = %while.end, %if.then
  ret void, !dbg !3773
}

; Function Attrs: convergent noinline nounwind
define dso_local double @_Z13randlc_devicePdd(double* %x, double %a) #0 !dbg !3774 {
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
  call void @llvm.dbg.declare(metadata double** %x.addr, metadata !3775, metadata !DIExpression()), !dbg !3776
  store double %a, double* %a.addr, align 8
  call void @llvm.dbg.declare(metadata double* %a.addr, metadata !3777, metadata !DIExpression()), !dbg !3778
  call void @llvm.dbg.declare(metadata double* %t1, metadata !3779, metadata !DIExpression()), !dbg !3780
  call void @llvm.dbg.declare(metadata double* %t2, metadata !3781, metadata !DIExpression()), !dbg !3782
  call void @llvm.dbg.declare(metadata double* %t3, metadata !3783, metadata !DIExpression()), !dbg !3784
  call void @llvm.dbg.declare(metadata double* %t4, metadata !3785, metadata !DIExpression()), !dbg !3786
  call void @llvm.dbg.declare(metadata double* %a1, metadata !3787, metadata !DIExpression()), !dbg !3788
  call void @llvm.dbg.declare(metadata double* %a2, metadata !3789, metadata !DIExpression()), !dbg !3790
  call void @llvm.dbg.declare(metadata double* %x1, metadata !3791, metadata !DIExpression()), !dbg !3792
  call void @llvm.dbg.declare(metadata double* %x2, metadata !3793, metadata !DIExpression()), !dbg !3794
  call void @llvm.dbg.declare(metadata double* %z, metadata !3795, metadata !DIExpression()), !dbg !3796
  %0 = load double, double* %a.addr, align 8, !dbg !3797
  %mul = fmul contract double 0x3E80000000000000, %0, !dbg !3798
  store double %mul, double* %t1, align 8, !dbg !3799
  %1 = load double, double* %t1, align 8, !dbg !3800
  %conv = fptosi double %1 to i32, !dbg !3800
  %conv1 = sitofp i32 %conv to double, !dbg !3801
  store double %conv1, double* %a1, align 8, !dbg !3802
  %2 = load double, double* %a.addr, align 8, !dbg !3803
  %3 = load double, double* %a1, align 8, !dbg !3804
  %mul2 = fmul contract double 0x4160000000000000, %3, !dbg !3805
  %sub = fsub contract double %2, %mul2, !dbg !3806
  store double %sub, double* %a2, align 8, !dbg !3807
  %4 = load double*, double** %x.addr, align 8, !dbg !3808
  %5 = load double, double* %4, align 8, !dbg !3809
  %mul3 = fmul contract double 0x3E80000000000000, %5, !dbg !3810
  store double %mul3, double* %t1, align 8, !dbg !3811
  %6 = load double, double* %t1, align 8, !dbg !3812
  %conv4 = fptosi double %6 to i32, !dbg !3812
  %conv5 = sitofp i32 %conv4 to double, !dbg !3813
  store double %conv5, double* %x1, align 8, !dbg !3814
  %7 = load double*, double** %x.addr, align 8, !dbg !3815
  %8 = load double, double* %7, align 8, !dbg !3816
  %9 = load double, double* %x1, align 8, !dbg !3817
  %mul6 = fmul contract double 0x4160000000000000, %9, !dbg !3818
  %sub7 = fsub contract double %8, %mul6, !dbg !3819
  store double %sub7, double* %x2, align 8, !dbg !3820
  %10 = load double, double* %a1, align 8, !dbg !3821
  %11 = load double, double* %x2, align 8, !dbg !3822
  %mul8 = fmul contract double %10, %11, !dbg !3823
  %12 = load double, double* %a2, align 8, !dbg !3824
  %13 = load double, double* %x1, align 8, !dbg !3825
  %mul9 = fmul contract double %12, %13, !dbg !3826
  %add = fadd contract double %mul8, %mul9, !dbg !3827
  store double %add, double* %t1, align 8, !dbg !3828
  %14 = load double, double* %t1, align 8, !dbg !3829
  %mul10 = fmul contract double 0x3E80000000000000, %14, !dbg !3830
  %conv11 = fptosi double %mul10 to i32, !dbg !3831
  %conv12 = sitofp i32 %conv11 to double, !dbg !3832
  store double %conv12, double* %t2, align 8, !dbg !3833
  %15 = load double, double* %t1, align 8, !dbg !3834
  %16 = load double, double* %t2, align 8, !dbg !3835
  %mul13 = fmul contract double 0x4160000000000000, %16, !dbg !3836
  %sub14 = fsub contract double %15, %mul13, !dbg !3837
  store double %sub14, double* %z, align 8, !dbg !3838
  %17 = load double, double* %z, align 8, !dbg !3839
  %mul15 = fmul contract double 0x4160000000000000, %17, !dbg !3840
  %18 = load double, double* %a2, align 8, !dbg !3841
  %19 = load double, double* %x2, align 8, !dbg !3842
  %mul16 = fmul contract double %18, %19, !dbg !3843
  %add17 = fadd contract double %mul15, %mul16, !dbg !3844
  store double %add17, double* %t3, align 8, !dbg !3845
  %20 = load double, double* %t3, align 8, !dbg !3846
  %mul18 = fmul contract double 0x3D10000000000000, %20, !dbg !3847
  %conv19 = fptosi double %mul18 to i32, !dbg !3848
  %conv20 = sitofp i32 %conv19 to double, !dbg !3849
  store double %conv20, double* %t4, align 8, !dbg !3850
  %21 = load double, double* %t3, align 8, !dbg !3851
  %22 = load double, double* %t4, align 8, !dbg !3852
  %mul21 = fmul contract double 0x42D0000000000000, %22, !dbg !3853
  %sub22 = fsub contract double %21, %mul21, !dbg !3854
  %23 = load double*, double** %x.addr, align 8, !dbg !3855
  store double %sub22, double* %23, align 8, !dbg !3856
  %24 = load double*, double** %x.addr, align 8, !dbg !3857
  %25 = load double, double* %24, align 8, !dbg !3858
  %mul23 = fmul contract double 0x3D10000000000000, %25, !dbg !3859
  ret double %mul23, !dbg !3860
}

; Function Attrs: noinline nounwind uwtable
define dso_local double @_Z6randlcPdd(double* %x, double %a) #6 !dbg !3861 {
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
  call void @llvm.dbg.declare(metadata double** %x.addr, metadata !3862, metadata !DIExpression()), !dbg !3863
  store double %a, double* %a.addr, align 8
  call void @llvm.dbg.declare(metadata double* %a.addr, metadata !3864, metadata !DIExpression()), !dbg !3865
  call void @llvm.dbg.declare(metadata double* %t1, metadata !3866, metadata !DIExpression()), !dbg !3867
  call void @llvm.dbg.declare(metadata double* %t2, metadata !3868, metadata !DIExpression()), !dbg !3869
  call void @llvm.dbg.declare(metadata double* %t3, metadata !3870, metadata !DIExpression()), !dbg !3871
  call void @llvm.dbg.declare(metadata double* %t4, metadata !3872, metadata !DIExpression()), !dbg !3873
  call void @llvm.dbg.declare(metadata double* %a1, metadata !3874, metadata !DIExpression()), !dbg !3875
  call void @llvm.dbg.declare(metadata double* %a2, metadata !3876, metadata !DIExpression()), !dbg !3877
  call void @llvm.dbg.declare(metadata double* %x1, metadata !3878, metadata !DIExpression()), !dbg !3879
  call void @llvm.dbg.declare(metadata double* %x2, metadata !3880, metadata !DIExpression()), !dbg !3881
  call void @llvm.dbg.declare(metadata double* %z, metadata !3882, metadata !DIExpression()), !dbg !3883
  %0 = load double, double* %a.addr, align 8, !dbg !3884
  %mul = fmul contract double 0x3E80000000000000, %0, !dbg !3885
  store double %mul, double* %t1, align 8, !dbg !3886
  %1 = load double, double* %t1, align 8, !dbg !3887
  %conv = fptosi double %1 to i32, !dbg !3887
  %conv1 = sitofp i32 %conv to double, !dbg !3888
  store double %conv1, double* %a1, align 8, !dbg !3889
  %2 = load double, double* %a.addr, align 8, !dbg !3890
  %3 = load double, double* %a1, align 8, !dbg !3891
  %mul2 = fmul contract double 0x4160000000000000, %3, !dbg !3892
  %sub = fsub contract double %2, %mul2, !dbg !3893
  store double %sub, double* %a2, align 8, !dbg !3894
  %4 = load double*, double** %x.addr, align 8, !dbg !3895
  %5 = load double, double* %4, align 8, !dbg !3896
  %mul3 = fmul contract double 0x3E80000000000000, %5, !dbg !3897
  store double %mul3, double* %t1, align 8, !dbg !3898
  %6 = load double, double* %t1, align 8, !dbg !3899
  %conv4 = fptosi double %6 to i32, !dbg !3899
  %conv5 = sitofp i32 %conv4 to double, !dbg !3900
  store double %conv5, double* %x1, align 8, !dbg !3901
  %7 = load double*, double** %x.addr, align 8, !dbg !3902
  %8 = load double, double* %7, align 8, !dbg !3903
  %9 = load double, double* %x1, align 8, !dbg !3904
  %mul6 = fmul contract double 0x4160000000000000, %9, !dbg !3905
  %sub7 = fsub contract double %8, %mul6, !dbg !3906
  store double %sub7, double* %x2, align 8, !dbg !3907
  %10 = load double, double* %a1, align 8, !dbg !3908
  %11 = load double, double* %x2, align 8, !dbg !3909
  %mul8 = fmul contract double %10, %11, !dbg !3910
  %12 = load double, double* %a2, align 8, !dbg !3911
  %13 = load double, double* %x1, align 8, !dbg !3912
  %mul9 = fmul contract double %12, %13, !dbg !3913
  %add = fadd contract double %mul8, %mul9, !dbg !3914
  store double %add, double* %t1, align 8, !dbg !3915
  %14 = load double, double* %t1, align 8, !dbg !3916
  %mul10 = fmul contract double 0x3E80000000000000, %14, !dbg !3917
  %conv11 = fptosi double %mul10 to i32, !dbg !3918
  %conv12 = sitofp i32 %conv11 to double, !dbg !3919
  store double %conv12, double* %t2, align 8, !dbg !3920
  %15 = load double, double* %t1, align 8, !dbg !3921
  %16 = load double, double* %t2, align 8, !dbg !3922
  %mul13 = fmul contract double 0x4160000000000000, %16, !dbg !3923
  %sub14 = fsub contract double %15, %mul13, !dbg !3924
  store double %sub14, double* %z, align 8, !dbg !3925
  %17 = load double, double* %z, align 8, !dbg !3926
  %mul15 = fmul contract double 0x4160000000000000, %17, !dbg !3927
  %18 = load double, double* %a2, align 8, !dbg !3928
  %19 = load double, double* %x2, align 8, !dbg !3929
  %mul16 = fmul contract double %18, %19, !dbg !3930
  %add17 = fadd contract double %mul15, %mul16, !dbg !3931
  store double %add17, double* %t3, align 8, !dbg !3932
  %20 = load double, double* %t3, align 8, !dbg !3933
  %mul18 = fmul contract double 0x3D10000000000000, %20, !dbg !3934
  %conv19 = fptosi double %mul18 to i32, !dbg !3935
  %conv20 = sitofp i32 %conv19 to double, !dbg !3936
  store double %conv20, double* %t4, align 8, !dbg !3937
  %21 = load double, double* %t3, align 8, !dbg !3938
  %22 = load double, double* %t4, align 8, !dbg !3939
  %mul21 = fmul contract double 0x42D0000000000000, %22, !dbg !3940
  %sub22 = fsub contract double %21, %mul21, !dbg !3941
  %23 = load double*, double** %x.addr, align 8, !dbg !3942
  store double %sub22, double* %23, align 8, !dbg !3943
  %24 = load double*, double** %x.addr, align 8, !dbg !3944
  %25 = load double, double* %24, align 8, !dbg !3945
  %mul23 = fmul contract double 0x3D10000000000000, %25, !dbg !3946
  ret double %mul23, !dbg !3947
}

; Function Attrs: noinline uwtable
define dso_local void @_Z15c_print_resultsPcciiiiddS_iS_S_S_S_S_S_S_S_S_S_S_S_S_S_(i8* %name, i8 signext %class_npb, i32 %n1, i32 %n2, i32 %n3, i32 %niter, double %t, double %mops, i8* %optype, i32 %passed_verification, i8* %npbversion, i8* %compiletime, i8* %compilerversion, i8* %libversion, i8* %cpu_device, i8* %gpu_device, i8* %gpu_config, i8* %cc, i8* %clink, i8* %c_lib, i8* %c_inc, i8* %cflags, i8* %clinkflags, i8* %rand) #7 !dbg !3948 {
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
  call void @llvm.dbg.declare(metadata i8** %name.addr, metadata !3951, metadata !DIExpression()), !dbg !3952
  store i8 %class_npb, i8* %class_npb.addr, align 1
  call void @llvm.dbg.declare(metadata i8* %class_npb.addr, metadata !3953, metadata !DIExpression()), !dbg !3954
  store i32 %n1, i32* %n1.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %n1.addr, metadata !3955, metadata !DIExpression()), !dbg !3956
  store i32 %n2, i32* %n2.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %n2.addr, metadata !3957, metadata !DIExpression()), !dbg !3958
  store i32 %n3, i32* %n3.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %n3.addr, metadata !3959, metadata !DIExpression()), !dbg !3960
  store i32 %niter, i32* %niter.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %niter.addr, metadata !3961, metadata !DIExpression()), !dbg !3962
  store double %t, double* %t.addr, align 8
  call void @llvm.dbg.declare(metadata double* %t.addr, metadata !3963, metadata !DIExpression()), !dbg !3964
  store double %mops, double* %mops.addr, align 8
  call void @llvm.dbg.declare(metadata double* %mops.addr, metadata !3965, metadata !DIExpression()), !dbg !3966
  store i8* %optype, i8** %optype.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %optype.addr, metadata !3967, metadata !DIExpression()), !dbg !3968
  store i32 %passed_verification, i32* %passed_verification.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %passed_verification.addr, metadata !3969, metadata !DIExpression()), !dbg !3970
  store i8* %npbversion, i8** %npbversion.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %npbversion.addr, metadata !3971, metadata !DIExpression()), !dbg !3972
  store i8* %compiletime, i8** %compiletime.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %compiletime.addr, metadata !3973, metadata !DIExpression()), !dbg !3974
  store i8* %compilerversion, i8** %compilerversion.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %compilerversion.addr, metadata !3975, metadata !DIExpression()), !dbg !3976
  store i8* %libversion, i8** %libversion.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %libversion.addr, metadata !3977, metadata !DIExpression()), !dbg !3978
  store i8* %cpu_device, i8** %cpu_device.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %cpu_device.addr, metadata !3979, metadata !DIExpression()), !dbg !3980
  store i8* %gpu_device, i8** %gpu_device.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %gpu_device.addr, metadata !3981, metadata !DIExpression()), !dbg !3982
  store i8* %gpu_config, i8** %gpu_config.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %gpu_config.addr, metadata !3983, metadata !DIExpression()), !dbg !3984
  store i8* %cc, i8** %cc.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %cc.addr, metadata !3985, metadata !DIExpression()), !dbg !3986
  store i8* %clink, i8** %clink.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %clink.addr, metadata !3987, metadata !DIExpression()), !dbg !3988
  store i8* %c_lib, i8** %c_lib.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %c_lib.addr, metadata !3989, metadata !DIExpression()), !dbg !3990
  store i8* %c_inc, i8** %c_inc.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %c_inc.addr, metadata !3991, metadata !DIExpression()), !dbg !3992
  store i8* %cflags, i8** %cflags.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %cflags.addr, metadata !3993, metadata !DIExpression()), !dbg !3994
  store i8* %clinkflags, i8** %clinkflags.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %clinkflags.addr, metadata !3995, metadata !DIExpression()), !dbg !3996
  store i8* %rand, i8** %rand.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %rand.addr, metadata !3997, metadata !DIExpression()), !dbg !3998
  %0 = load i8*, i8** %name.addr, align 8, !dbg !3999
  %call = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([27 x i8], [27 x i8]* @.str, i64 0, i64 0), i8* %0), !dbg !4000
  %1 = load i8, i8* %class_npb.addr, align 1, !dbg !4001
  %conv = sext i8 %1 to i32, !dbg !4001
  %call1 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([46 x i8], [46 x i8]* @.str.1, i64 0, i64 0), i32 %conv), !dbg !4002
  %2 = load i8*, i8** %name.addr, align 8, !dbg !4003
  %arrayidx = getelementptr inbounds i8, i8* %2, i64 0, !dbg !4003
  %3 = load i8, i8* %arrayidx, align 1, !dbg !4003
  %conv2 = sext i8 %3 to i32, !dbg !4003
  %cmp = icmp eq i32 %conv2, 73, !dbg !4005
  br i1 %cmp, label %land.lhs.true, label %if.else15, !dbg !4006

land.lhs.true:                                    ; preds = %entry
  %4 = load i8*, i8** %name.addr, align 8, !dbg !4007
  %arrayidx3 = getelementptr inbounds i8, i8* %4, i64 1, !dbg !4007
  %5 = load i8, i8* %arrayidx3, align 1, !dbg !4007
  %conv4 = sext i8 %5 to i32, !dbg !4007
  %cmp5 = icmp eq i32 %conv4, 83, !dbg !4008
  br i1 %cmp5, label %if.then, label %if.else15, !dbg !4009

if.then:                                          ; preds = %land.lhs.true
  %6 = load i32, i32* %n3.addr, align 4, !dbg !4010
  %cmp6 = icmp eq i32 %6, 0, !dbg !4013
  br i1 %cmp6, label %if.then7, label %if.else, !dbg !4014

if.then7:                                         ; preds = %if.then
  call void @llvm.dbg.declare(metadata i64* %nn, metadata !4015, metadata !DIExpression()), !dbg !4017
  %7 = load i32, i32* %n1.addr, align 4, !dbg !4018
  %conv8 = sext i32 %7 to i64, !dbg !4018
  store i64 %conv8, i64* %nn, align 8, !dbg !4017
  %8 = load i32, i32* %n2.addr, align 4, !dbg !4019
  %cmp9 = icmp ne i32 %8, 0, !dbg !4021
  br i1 %cmp9, label %if.then10, label %if.end, !dbg !4022

if.then10:                                        ; preds = %if.then7
  %9 = load i32, i32* %n2.addr, align 4, !dbg !4023
  %conv11 = sext i32 %9 to i64, !dbg !4023
  %10 = load i64, i64* %nn, align 8, !dbg !4025
  %mul = mul nsw i64 %10, %conv11, !dbg !4025
  store i64 %mul, i64* %nn, align 8, !dbg !4025
  br label %if.end, !dbg !4026

if.end:                                           ; preds = %if.then10, %if.then7
  %11 = load i64, i64* %nn, align 8, !dbg !4027
  %call12 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([38 x i8], [38 x i8]* @.str.2, i64 0, i64 0), i64 %11), !dbg !4028
  br label %if.end14, !dbg !4029

if.else:                                          ; preds = %if.then
  %12 = load i32, i32* %n1.addr, align 4, !dbg !4030
  %13 = load i32, i32* %n2.addr, align 4, !dbg !4032
  %14 = load i32, i32* %n3.addr, align 4, !dbg !4033
  %call13 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([44 x i8], [44 x i8]* @.str.3, i64 0, i64 0), i32 %12, i32 %13, i32 %14), !dbg !4034
  br label %if.end14

if.end14:                                         ; preds = %if.else, %if.end
  br label %if.end48, !dbg !4035

if.else15:                                        ; preds = %land.lhs.true, %entry
  call void @llvm.dbg.declare(metadata [16 x i8]* %size, metadata !4036, metadata !DIExpression()), !dbg !4041
  call void @llvm.dbg.declare(metadata i32* %j, metadata !4042, metadata !DIExpression()), !dbg !4043
  %15 = load i32, i32* %n2.addr, align 4, !dbg !4044
  %cmp16 = icmp eq i32 %15, 0, !dbg !4046
  br i1 %cmp16, label %land.lhs.true17, label %if.else45, !dbg !4047

land.lhs.true17:                                  ; preds = %if.else15
  %16 = load i32, i32* %n3.addr, align 4, !dbg !4048
  %cmp18 = icmp eq i32 %16, 0, !dbg !4049
  br i1 %cmp18, label %if.then19, label %if.else45, !dbg !4050

if.then19:                                        ; preds = %land.lhs.true17
  %17 = load i8*, i8** %name.addr, align 8, !dbg !4051
  %arrayidx20 = getelementptr inbounds i8, i8* %17, i64 0, !dbg !4051
  %18 = load i8, i8* %arrayidx20, align 1, !dbg !4051
  %conv21 = sext i8 %18 to i32, !dbg !4051
  %cmp22 = icmp eq i32 %conv21, 69, !dbg !4054
  br i1 %cmp22, label %land.lhs.true23, label %if.else42, !dbg !4055

land.lhs.true23:                                  ; preds = %if.then19
  %19 = load i8*, i8** %name.addr, align 8, !dbg !4056
  %arrayidx24 = getelementptr inbounds i8, i8* %19, i64 1, !dbg !4056
  %20 = load i8, i8* %arrayidx24, align 1, !dbg !4056
  %conv25 = sext i8 %20 to i32, !dbg !4056
  %cmp26 = icmp eq i32 %conv25, 80, !dbg !4057
  br i1 %cmp26, label %if.then27, label %if.else42, !dbg !4058

if.then27:                                        ; preds = %land.lhs.true23
  %arraydecay = getelementptr inbounds [16 x i8], [16 x i8]* %size, i64 0, i64 0, !dbg !4059
  %21 = load i32, i32* %n1.addr, align 4, !dbg !4061
  %conv28 = sitofp i32 %21 to double, !dbg !4061
  %call29 = call double @pow(double 2.000000e+00, double %conv28) #11, !dbg !4062
  %call30 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay, i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.4, i64 0, i64 0), double %call29) #11, !dbg !4063
  store i32 14, i32* %j, align 4, !dbg !4064
  %22 = load i32, i32* %j, align 4, !dbg !4065
  %idxprom = sext i32 %22 to i64, !dbg !4067
  %arrayidx31 = getelementptr inbounds [16 x i8], [16 x i8]* %size, i64 0, i64 %idxprom, !dbg !4067
  %23 = load i8, i8* %arrayidx31, align 1, !dbg !4067
  %conv32 = sext i8 %23 to i32, !dbg !4067
  %cmp33 = icmp eq i32 %conv32, 46, !dbg !4068
  br i1 %cmp33, label %if.then34, label %if.end37, !dbg !4069

if.then34:                                        ; preds = %if.then27
  %24 = load i32, i32* %j, align 4, !dbg !4070
  %idxprom35 = sext i32 %24 to i64, !dbg !4072
  %arrayidx36 = getelementptr inbounds [16 x i8], [16 x i8]* %size, i64 0, i64 %idxprom35, !dbg !4072
  store i8 32, i8* %arrayidx36, align 1, !dbg !4073
  %25 = load i32, i32* %j, align 4, !dbg !4074
  %dec = add nsw i32 %25, -1, !dbg !4074
  store i32 %dec, i32* %j, align 4, !dbg !4074
  br label %if.end37, !dbg !4075

if.end37:                                         ; preds = %if.then34, %if.then27
  %26 = load i32, i32* %j, align 4, !dbg !4076
  %add = add nsw i32 %26, 1, !dbg !4077
  %idxprom38 = sext i32 %add to i64, !dbg !4078
  %arrayidx39 = getelementptr inbounds [16 x i8], [16 x i8]* %size, i64 0, i64 %idxprom38, !dbg !4078
  store i8 0, i8* %arrayidx39, align 1, !dbg !4079
  %arraydecay40 = getelementptr inbounds [16 x i8], [16 x i8]* %size, i64 0, i64 0, !dbg !4080
  %call41 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([34 x i8], [34 x i8]* @.str.5, i64 0, i64 0), i8* %arraydecay40), !dbg !4081
  br label %if.end44, !dbg !4082

if.else42:                                        ; preds = %land.lhs.true23, %if.then19
  %27 = load i32, i32* %n1.addr, align 4, !dbg !4083
  %call43 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.6, i64 0, i64 0), i32 %27), !dbg !4085
  br label %if.end44

if.end44:                                         ; preds = %if.else42, %if.end37
  br label %if.end47, !dbg !4086

if.else45:                                        ; preds = %land.lhs.true17, %if.else15
  %28 = load i32, i32* %n1.addr, align 4, !dbg !4087
  %29 = load i32, i32* %n2.addr, align 4, !dbg !4089
  %30 = load i32, i32* %n3.addr, align 4, !dbg !4090
  %call46 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([42 x i8], [42 x i8]* @.str.7, i64 0, i64 0), i32 %28, i32 %29, i32 %30), !dbg !4091
  br label %if.end47

if.end47:                                         ; preds = %if.else45, %if.end44
  br label %if.end48

if.end48:                                         ; preds = %if.end47, %if.end14
  %31 = load i32, i32* %niter.addr, align 4, !dbg !4092
  %call49 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.8, i64 0, i64 0), i32 %31), !dbg !4093
  %32 = load double, double* %t.addr, align 8, !dbg !4094
  %call50 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([39 x i8], [39 x i8]* @.str.9, i64 0, i64 0), double %32), !dbg !4095
  %33 = load double, double* %mops.addr, align 8, !dbg !4096
  %call51 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([39 x i8], [39 x i8]* @.str.10, i64 0, i64 0), double %33), !dbg !4097
  %34 = load i8*, i8** %optype.addr, align 8, !dbg !4098
  %call52 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([25 x i8], [25 x i8]* @.str.11, i64 0, i64 0), i8* %34), !dbg !4099
  %35 = load i32, i32* %passed_verification.addr, align 4, !dbg !4100
  %cmp53 = icmp slt i32 %35, 0, !dbg !4102
  br i1 %cmp53, label %if.then54, label %if.else56, !dbg !4103

if.then54:                                        ; preds = %if.end48
  %call55 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([45 x i8], [45 x i8]* @.str.12, i64 0, i64 0)), !dbg !4104
  br label %if.end62, !dbg !4106

if.else56:                                        ; preds = %if.end48
  %36 = load i32, i32* %passed_verification.addr, align 4, !dbg !4107
  %tobool = icmp ne i32 %36, 0, !dbg !4107
  br i1 %tobool, label %if.then57, label %if.else59, !dbg !4109

if.then57:                                        ; preds = %if.else56
  %call58 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([45 x i8], [45 x i8]* @.str.13, i64 0, i64 0)), !dbg !4110
  br label %if.end61, !dbg !4112

if.else59:                                        ; preds = %if.else56
  %call60 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([45 x i8], [45 x i8]* @.str.14, i64 0, i64 0)), !dbg !4113
  br label %if.end61

if.end61:                                         ; preds = %if.else59, %if.then57
  br label %if.end62

if.end62:                                         ; preds = %if.end61, %if.then54
  %37 = load i8*, i8** %npbversion.addr, align 8, !dbg !4115
  %call63 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.15, i64 0, i64 0), i8* %37), !dbg !4116
  %38 = load i8*, i8** %compiletime.addr, align 8, !dbg !4117
  %call64 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.16, i64 0, i64 0), i8* %38), !dbg !4118
  %39 = load i8*, i8** %compilerversion.addr, align 8, !dbg !4119
  %call65 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.17, i64 0, i64 0), i8* %39), !dbg !4120
  %40 = load i8*, i8** %libversion.addr, align 8, !dbg !4121
  %call66 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.18, i64 0, i64 0), i8* %40), !dbg !4122
  %call67 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([20 x i8], [20 x i8]* @.str.19, i64 0, i64 0)), !dbg !4123
  %41 = load i8*, i8** %cc.addr, align 8, !dbg !4124
  %call68 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.20, i64 0, i64 0), i8* %41), !dbg !4125
  %42 = load i8*, i8** %clink.addr, align 8, !dbg !4126
  %call69 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.21, i64 0, i64 0), i8* %42), !dbg !4127
  %43 = load i8*, i8** %c_lib.addr, align 8, !dbg !4128
  %call70 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.22, i64 0, i64 0), i8* %43), !dbg !4129
  %44 = load i8*, i8** %c_inc.addr, align 8, !dbg !4130
  %call71 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.23, i64 0, i64 0), i8* %44), !dbg !4131
  %45 = load i8*, i8** %cflags.addr, align 8, !dbg !4132
  %call72 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.24, i64 0, i64 0), i8* %45), !dbg !4133
  %46 = load i8*, i8** %clinkflags.addr, align 8, !dbg !4134
  %call73 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.25, i64 0, i64 0), i8* %46), !dbg !4135
  %47 = load i8*, i8** %rand.addr, align 8, !dbg !4136
  %call74 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.26, i64 0, i64 0), i8* %47), !dbg !4137
  %call75 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.27, i64 0, i64 0)), !dbg !4138
  %48 = load i8*, i8** %cpu_device.addr, align 8, !dbg !4139
  %call76 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.28, i64 0, i64 0), i8* %48), !dbg !4140
  %49 = load i8*, i8** %gpu_device.addr, align 8, !dbg !4141
  %call77 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.29, i64 0, i64 0), i8* %49), !dbg !4142
  %call78 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.30, i64 0, i64 0)), !dbg !4143
  %50 = load i8*, i8** %gpu_config.addr, align 8, !dbg !4144
  %call79 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.31, i64 0, i64 0), i8* %50), !dbg !4145
  %call80 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.32, i64 0, i64 0)), !dbg !4146
  %call81 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([72 x i8], [72 x i8]* @.str.33, i64 0, i64 0)), !dbg !4147
  %call82 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([27 x i8], [27 x i8]* @.str.34, i64 0, i64 0)), !dbg !4148
  %call83 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([56 x i8], [56 x i8]* @.str.35, i64 0, i64 0)), !dbg !4149
  %call84 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([52 x i8], [52 x i8]* @.str.36, i64 0, i64 0)), !dbg !4150
  %call85 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([46 x i8], [46 x i8]* @.str.37, i64 0, i64 0)), !dbg !4151
  %call86 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.32, i64 0, i64 0)), !dbg !4152
  %call87 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([43 x i8], [43 x i8]* @.str.38, i64 0, i64 0)), !dbg !4153
  %call88 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([72 x i8], [72 x i8]* @.str.33, i64 0, i64 0)), !dbg !4154
  %call89 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.32, i64 0, i64 0)), !dbg !4155
  ret void, !dbg !4156
}

declare dso_local i32 @printf(i8*, ...) #8

; Function Attrs: nounwind
declare dso_local double @pow(double, double) #9

; Function Attrs: nounwind
declare dso_local i32 @sprintf(i8*, i8*, ...) #9

; Function Attrs: noinline norecurse uwtable
define dso_local i32 @main(i32 %argc, i8** %argv) #10 !dbg !4157 {
entry:
  %retval = alloca i32, align 4
  %argc.addr = alloca i32, align 4
  %argv.addr = alloca i8**, align 8
  %iter = alloca i32, align 4
  %total_time = alloca double, align 8
  %mflops = alloca double, align 8
  %verified = alloca i32, align 4
  %class_npb = alloca i8, align 1
  %gpu_config = alloca [256 x i8], align 16
  %gpu_config_string = alloca [2048 x i8], align 16
  store i32 0, i32* %retval, align 4
  store i32 %argc, i32* %argc.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %argc.addr, metadata !4160, metadata !DIExpression()), !dbg !4161
  store i8** %argv, i8*** %argv.addr, align 8
  call void @llvm.dbg.declare(metadata i8*** %argv.addr, metadata !4162, metadata !DIExpression()), !dbg !4163
  call void @llvm.dbg.declare(metadata i32* %iter, metadata !4164, metadata !DIExpression()), !dbg !4165
  store i32 0, i32* %iter, align 4, !dbg !4165
  call void @llvm.dbg.declare(metadata double* %total_time, metadata !4166, metadata !DIExpression()), !dbg !4167
  call void @llvm.dbg.declare(metadata double* %mflops, metadata !4168, metadata !DIExpression()), !dbg !4169
  call void @llvm.dbg.declare(metadata i32* %verified, metadata !4170, metadata !DIExpression()), !dbg !4172
  call void @llvm.dbg.declare(metadata i8* %class_npb, metadata !4173, metadata !DIExpression()), !dbg !4174
  %call = call noalias i8* @malloc(i64 112) #11, !dbg !4175
  %0 = bitcast i8* %call to %struct.dcomplex*, !dbg !4176
  store %struct.dcomplex* %0, %struct.dcomplex** @_ZL4sums, align 8, !dbg !4177
  %call1 = call noalias i8* @malloc(i64 67108864) #11, !dbg !4178
  %1 = bitcast i8* %call1 to double*, !dbg !4179
  store double* %1, double** @_ZL7twiddle, align 8, !dbg !4180
  %call2 = call noalias i8* @malloc(i64 4096) #11, !dbg !4181
  %2 = bitcast i8* %call2 to %struct.dcomplex*, !dbg !4182
  store %struct.dcomplex* %2, %struct.dcomplex** @_ZL1u, align 8, !dbg !4183
  %call3 = call noalias i8* @malloc(i64 134217728) #11, !dbg !4184
  %3 = bitcast i8* %call3 to %struct.dcomplex*, !dbg !4185
  store %struct.dcomplex* %3, %struct.dcomplex** @_ZL2u0, align 8, !dbg !4186
  %call4 = call noalias i8* @malloc(i64 134217728) #11, !dbg !4187
  %4 = bitcast i8* %call4 to %struct.dcomplex*, !dbg !4188
  store %struct.dcomplex* %4, %struct.dcomplex** @_ZL2u1, align 8, !dbg !4189
  %call5 = call noalias i8* @malloc(i64 12) #11, !dbg !4190
  %5 = bitcast i8* %call5 to i32*, !dbg !4191
  store i32* %5, i32** @_ZL4dims, align 8, !dbg !4192
  call void @_ZL5setupv(), !dbg !4193
  call void @_ZL9setup_gpuv(), !dbg !4194
  %6 = load %struct.dcomplex*, %struct.dcomplex** @u0_device, align 8, !dbg !4195
  %7 = load %struct.dcomplex*, %struct.dcomplex** @u1_device, align 8, !dbg !4196
  %8 = load double*, double** @twiddle_device, align 8, !dbg !4197
  call void @_ZL11init_ui_gpuP8dcomplexS0_Pd(%struct.dcomplex* %6, %struct.dcomplex* %7, double* %8), !dbg !4198
  %call6 = call i32 @omp_get_thread_num(), !dbg !4199
  %cmp = icmp eq i32 %call6, 0, !dbg !4202
  br i1 %cmp, label %if.then, label %if.else, !dbg !4203

if.then:                                          ; preds = %entry
  %9 = load double*, double** @twiddle_device, align 8, !dbg !4204
  call void @_ZL20compute_indexmap_gpuPd(double* %9), !dbg !4206
  br label %if.end15, !dbg !4207

if.else:                                          ; preds = %entry
  %call7 = call i32 @omp_get_thread_num(), !dbg !4208
  %cmp8 = icmp eq i32 %call7, 1, !dbg !4210
  br i1 %cmp8, label %if.then9, label %if.else10, !dbg !4211

if.then9:                                         ; preds = %if.else
  %10 = load %struct.dcomplex*, %struct.dcomplex** @u1_device, align 8, !dbg !4212
  call void @_ZL30compute_initial_conditions_gpuP8dcomplex(%struct.dcomplex* %10), !dbg !4214
  br label %if.end14, !dbg !4215

if.else10:                                        ; preds = %if.else
  %call11 = call i32 @omp_get_thread_num(), !dbg !4216
  %cmp12 = icmp eq i32 %call11, 2, !dbg !4218
  br i1 %cmp12, label %if.then13, label %if.end, !dbg !4219

if.then13:                                        ; preds = %if.else10
  call void @_ZL12fft_init_gpui(i32 256), !dbg !4220
  br label %if.end, !dbg !4222

if.end:                                           ; preds = %if.then13, %if.else10
  br label %if.end14

if.end14:                                         ; preds = %if.end, %if.then9
  br label %if.end15

if.end15:                                         ; preds = %if.end14, %if.then
  %call16 = call i32 @cudaDeviceSynchronize(), !dbg !4223
  %11 = load %struct.dcomplex*, %struct.dcomplex** @u1_device, align 8, !dbg !4224
  %12 = load %struct.dcomplex*, %struct.dcomplex** @u0_device, align 8, !dbg !4225
  call void @_ZL7fft_gpuiP8dcomplexS0_(i32 1, %struct.dcomplex* %11, %struct.dcomplex* %12), !dbg !4226
  %call17 = call i32 @omp_get_thread_num(), !dbg !4227
  %cmp18 = icmp eq i32 %call17, 0, !dbg !4230
  br i1 %cmp18, label %if.then19, label %if.else20, !dbg !4231

if.then19:                                        ; preds = %if.end15
  %13 = load double*, double** @twiddle_device, align 8, !dbg !4232
  call void @_ZL20compute_indexmap_gpuPd(double* %13), !dbg !4234
  br label %if.end30, !dbg !4235

if.else20:                                        ; preds = %if.end15
  %call21 = call i32 @omp_get_thread_num(), !dbg !4236
  %cmp22 = icmp eq i32 %call21, 1, !dbg !4238
  br i1 %cmp22, label %if.then23, label %if.else24, !dbg !4239

if.then23:                                        ; preds = %if.else20
  %14 = load %struct.dcomplex*, %struct.dcomplex** @u1_device, align 8, !dbg !4240
  call void @_ZL30compute_initial_conditions_gpuP8dcomplex(%struct.dcomplex* %14), !dbg !4242
  br label %if.end29, !dbg !4243

if.else24:                                        ; preds = %if.else20
  %call25 = call i32 @omp_get_thread_num(), !dbg !4244
  %cmp26 = icmp eq i32 %call25, 2, !dbg !4246
  br i1 %cmp26, label %if.then27, label %if.end28, !dbg !4247

if.then27:                                        ; preds = %if.else24
  call void @_ZL12fft_init_gpui(i32 256), !dbg !4248
  br label %if.end28, !dbg !4250

if.end28:                                         ; preds = %if.then27, %if.else24
  br label %if.end29

if.end29:                                         ; preds = %if.end28, %if.then23
  br label %if.end30

if.end30:                                         ; preds = %if.end29, %if.then19
  %call31 = call i32 @cudaDeviceSynchronize(), !dbg !4251
  %15 = load %struct.dcomplex*, %struct.dcomplex** @u1_device, align 8, !dbg !4252
  %16 = load %struct.dcomplex*, %struct.dcomplex** @u0_device, align 8, !dbg !4253
  call void @_ZL7fft_gpuiP8dcomplexS0_(i32 1, %struct.dcomplex* %15, %struct.dcomplex* %16), !dbg !4254
  store i32 1, i32* %iter, align 4, !dbg !4255
  br label %for.cond, !dbg !4257

for.cond:                                         ; preds = %for.inc, %if.end30
  %17 = load i32, i32* %iter, align 4, !dbg !4258
  %18 = load i32, i32* @_ZL5niter, align 4, !dbg !4260
  %cmp32 = icmp sle i32 %17, %18, !dbg !4261
  br i1 %cmp32, label %for.body, label %for.end, !dbg !4262

for.body:                                         ; preds = %for.cond
  %19 = load %struct.dcomplex*, %struct.dcomplex** @u0_device, align 8, !dbg !4263
  %20 = load %struct.dcomplex*, %struct.dcomplex** @u1_device, align 8, !dbg !4265
  %21 = load double*, double** @twiddle_device, align 8, !dbg !4266
  call void @_ZL10evolve_gpuP8dcomplexS0_Pd(%struct.dcomplex* %19, %struct.dcomplex* %20, double* %21), !dbg !4267
  %22 = load %struct.dcomplex*, %struct.dcomplex** @u1_device, align 8, !dbg !4268
  %23 = load %struct.dcomplex*, %struct.dcomplex** @u1_device, align 8, !dbg !4269
  call void @_ZL7fft_gpuiP8dcomplexS0_(i32 -1, %struct.dcomplex* %22, %struct.dcomplex* %23), !dbg !4270
  %24 = load i32, i32* %iter, align 4, !dbg !4271
  %25 = load %struct.dcomplex*, %struct.dcomplex** @u1_device, align 8, !dbg !4272
  call void @_ZL12checksum_gpuiP8dcomplex(i32 %24, %struct.dcomplex* %25), !dbg !4273
  br label %for.inc, !dbg !4274

for.inc:                                          ; preds = %for.body
  %26 = load i32, i32* %iter, align 4, !dbg !4275
  %inc = add nsw i32 %26, 1, !dbg !4275
  store i32 %inc, i32* %iter, align 4, !dbg !4275
  br label %for.cond, !dbg !4276, !llvm.loop !4277

for.end:                                          ; preds = %for.cond
  %27 = load %struct.dcomplex*, %struct.dcomplex** @_ZL4sums, align 8, !dbg !4279
  %28 = bitcast %struct.dcomplex* %27 to i8*, !dbg !4279
  %29 = load %struct.dcomplex*, %struct.dcomplex** @sums_device, align 8, !dbg !4280
  %30 = bitcast %struct.dcomplex* %29 to i8*, !dbg !4280
  %31 = load i64, i64* @size_sums_device, align 8, !dbg !4281
  %call33 = call i32 @cudaMemcpy(i8* %28, i8* %30, i64 %31, i32 2), !dbg !4282
  store i32 1, i32* %iter, align 4, !dbg !4283
  br label %for.cond34, !dbg !4285

for.cond34:                                       ; preds = %for.inc40, %for.end
  %32 = load i32, i32* %iter, align 4, !dbg !4286
  %33 = load i32, i32* @_ZL5niter, align 4, !dbg !4288
  %cmp35 = icmp sle i32 %32, %33, !dbg !4289
  br i1 %cmp35, label %for.body36, label %for.end42, !dbg !4290

for.body36:                                       ; preds = %for.cond34
  %34 = load i32, i32* %iter, align 4, !dbg !4291
  %35 = load %struct.dcomplex*, %struct.dcomplex** @_ZL4sums, align 8, !dbg !4293
  %36 = load i32, i32* %iter, align 4, !dbg !4294
  %idxprom = sext i32 %36 to i64, !dbg !4293
  %arrayidx = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %35, i64 %idxprom, !dbg !4293
  %real = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx, i32 0, i32 0, !dbg !4295
  %37 = load double, double* %real, align 8, !dbg !4295
  %38 = load %struct.dcomplex*, %struct.dcomplex** @_ZL4sums, align 8, !dbg !4296
  %39 = load i32, i32* %iter, align 4, !dbg !4297
  %idxprom37 = sext i32 %39 to i64, !dbg !4296
  %arrayidx38 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %38, i64 %idxprom37, !dbg !4296
  %imag = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx38, i32 0, i32 1, !dbg !4298
  %40 = load double, double* %imag, align 8, !dbg !4298
  %call39 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([40 x i8], [40 x i8]* @.str.39, i64 0, i64 0), i32 %34, double %37, double %40), !dbg !4299
  br label %for.inc40, !dbg !4300

for.inc40:                                        ; preds = %for.body36
  %41 = load i32, i32* %iter, align 4, !dbg !4301
  %inc41 = add nsw i32 %41, 1, !dbg !4301
  store i32 %inc41, i32* %iter, align 4, !dbg !4301
  br label %for.cond34, !dbg !4302, !llvm.loop !4303

for.end42:                                        ; preds = %for.cond34
  %42 = load i32, i32* @_ZL5niter, align 4, !dbg !4305
  call void @_ZL6verifyiiiiPiPc(i32 256, i32 256, i32 128, i32 %42, i32* %verified, i8* %class_npb), !dbg !4306
  store double 0.000000e+00, double* %total_time, align 8, !dbg !4307
  %43 = load double, double* %total_time, align 8, !dbg !4308
  %cmp43 = fcmp une double %43, 0.000000e+00, !dbg !4310
  br i1 %cmp43, label %if.then44, label %if.else52, !dbg !4311

if.then44:                                        ; preds = %for.end42
  %call45 = call double @log(double 0x4160000000000000) #11, !dbg !4312
  %mul = fmul contract double 7.196410e+00, %call45, !dbg !4314
  %add = fadd contract double 1.481570e+01, %mul, !dbg !4315
  %call46 = call double @log(double 0x4160000000000000) #11, !dbg !4316
  %mul47 = fmul contract double 7.211130e+00, %call46, !dbg !4317
  %add48 = fadd contract double 5.235180e+00, %mul47, !dbg !4318
  %44 = load i32, i32* @_ZL5niter, align 4, !dbg !4319
  %conv = sitofp i32 %44 to double, !dbg !4319
  %mul49 = fmul contract double %add48, %conv, !dbg !4320
  %add50 = fadd contract double %add, %mul49, !dbg !4321
  %mul51 = fmul contract double 0x4020C6F7A0B5ED8D, %add50, !dbg !4322
  %45 = load double, double* %total_time, align 8, !dbg !4323
  %div = fdiv double %mul51, %45, !dbg !4324
  store double %div, double* %mflops, align 8, !dbg !4325
  br label %if.end53, !dbg !4326

if.else52:                                        ; preds = %for.end42
  store double 0.000000e+00, double* %mflops, align 8, !dbg !4327
  br label %if.end53

if.end53:                                         ; preds = %if.else52, %if.then44
  call void @llvm.dbg.declare(metadata [256 x i8]* %gpu_config, metadata !4329, metadata !DIExpression()), !dbg !4330
  call void @llvm.dbg.declare(metadata [2048 x i8]* %gpu_config_string, metadata !4331, metadata !DIExpression()), !dbg !4335
  %arraydecay = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !4336
  %call54 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay, i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.40, i64 0, i64 0), i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.41, i64 0, i64 0), i8* getelementptr inbounds ([18 x i8], [18 x i8]* @.str.42, i64 0, i64 0)) #11, !dbg !4337
  %arraydecay55 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !4338
  %arraydecay56 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !4339
  %call57 = call i8* @strcpy(i8* %arraydecay55, i8* %arraydecay56) #11, !dbg !4340
  %arraydecay58 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !4341
  %46 = load i32, i32* @threads_per_block_on_compute_indexmap, align 4, !dbg !4342
  %call59 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay58, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.43, i64 0, i64 0), i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.44, i64 0, i64 0), i32 %46) #11, !dbg !4343
  %arraydecay60 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !4344
  %arraydecay61 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !4345
  %call62 = call i8* @strcat(i8* %arraydecay60, i8* %arraydecay61) #11, !dbg !4346
  %arraydecay63 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !4347
  %47 = load i32, i32* @threads_per_block_on_compute_initial_conditions, align 4, !dbg !4348
  %call64 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay63, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.43, i64 0, i64 0), i8* getelementptr inbounds ([20 x i8], [20 x i8]* @.str.45, i64 0, i64 0), i32 %47) #11, !dbg !4349
  %arraydecay65 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !4350
  %arraydecay66 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !4351
  %call67 = call i8* @strcat(i8* %arraydecay65, i8* %arraydecay66) #11, !dbg !4352
  %arraydecay68 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !4353
  %48 = load i32, i32* @threads_per_block_on_init_ui, align 4, !dbg !4354
  %call69 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay68, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.43, i64 0, i64 0), i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str.46, i64 0, i64 0), i32 %48) #11, !dbg !4355
  %arraydecay70 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !4356
  %arraydecay71 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !4357
  %call72 = call i8* @strcat(i8* %arraydecay70, i8* %arraydecay71) #11, !dbg !4358
  %arraydecay73 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !4359
  %49 = load i32, i32* @threads_per_block_on_evolve, align 4, !dbg !4360
  %call74 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay73, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.43, i64 0, i64 0), i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.47, i64 0, i64 0), i32 %49) #11, !dbg !4361
  %arraydecay75 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !4362
  %arraydecay76 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !4363
  %call77 = call i8* @strcat(i8* %arraydecay75, i8* %arraydecay76) #11, !dbg !4364
  %arraydecay78 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !4365
  %50 = load i32, i32* @threads_per_block_on_fftx_1, align 4, !dbg !4366
  %call79 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay78, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.43, i64 0, i64 0), i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.48, i64 0, i64 0), i32 %50) #11, !dbg !4367
  %arraydecay80 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !4368
  %arraydecay81 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !4369
  %call82 = call i8* @strcat(i8* %arraydecay80, i8* %arraydecay81) #11, !dbg !4370
  %arraydecay83 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !4371
  %51 = load i32, i32* @threads_per_block_on_fftx_2, align 4, !dbg !4372
  %call84 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay83, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.43, i64 0, i64 0), i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.49, i64 0, i64 0), i32 %51) #11, !dbg !4373
  %arraydecay85 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !4374
  %arraydecay86 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !4375
  %call87 = call i8* @strcat(i8* %arraydecay85, i8* %arraydecay86) #11, !dbg !4376
  %arraydecay88 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !4377
  %52 = load i32, i32* @threads_per_block_on_fftx_3, align 4, !dbg !4378
  %call89 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay88, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.43, i64 0, i64 0), i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.50, i64 0, i64 0), i32 %52) #11, !dbg !4379
  %arraydecay90 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !4380
  %arraydecay91 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !4381
  %call92 = call i8* @strcat(i8* %arraydecay90, i8* %arraydecay91) #11, !dbg !4382
  %arraydecay93 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !4383
  %53 = load i32, i32* @threads_per_block_on_ffty_1, align 4, !dbg !4384
  %call94 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay93, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.43, i64 0, i64 0), i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.51, i64 0, i64 0), i32 %53) #11, !dbg !4385
  %arraydecay95 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !4386
  %arraydecay96 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !4387
  %call97 = call i8* @strcat(i8* %arraydecay95, i8* %arraydecay96) #11, !dbg !4388
  %arraydecay98 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !4389
  %54 = load i32, i32* @threads_per_block_on_ffty_2, align 4, !dbg !4390
  %call99 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay98, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.43, i64 0, i64 0), i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.52, i64 0, i64 0), i32 %54) #11, !dbg !4391
  %arraydecay100 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !4392
  %arraydecay101 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !4393
  %call102 = call i8* @strcat(i8* %arraydecay100, i8* %arraydecay101) #11, !dbg !4394
  %arraydecay103 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !4395
  %55 = load i32, i32* @threads_per_block_on_ffty_3, align 4, !dbg !4396
  %call104 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay103, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.43, i64 0, i64 0), i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.53, i64 0, i64 0), i32 %55) #11, !dbg !4397
  %arraydecay105 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !4398
  %arraydecay106 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !4399
  %call107 = call i8* @strcat(i8* %arraydecay105, i8* %arraydecay106) #11, !dbg !4400
  %arraydecay108 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !4401
  %56 = load i32, i32* @threads_per_block_on_fftz_1, align 4, !dbg !4402
  %call109 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay108, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.43, i64 0, i64 0), i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.54, i64 0, i64 0), i32 %56) #11, !dbg !4403
  %arraydecay110 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !4404
  %arraydecay111 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !4405
  %call112 = call i8* @strcat(i8* %arraydecay110, i8* %arraydecay111) #11, !dbg !4406
  %arraydecay113 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !4407
  %57 = load i32, i32* @threads_per_block_on_fftz_2, align 4, !dbg !4408
  %call114 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay113, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.43, i64 0, i64 0), i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.55, i64 0, i64 0), i32 %57) #11, !dbg !4409
  %arraydecay115 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !4410
  %arraydecay116 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !4411
  %call117 = call i8* @strcat(i8* %arraydecay115, i8* %arraydecay116) #11, !dbg !4412
  %arraydecay118 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !4413
  %58 = load i32, i32* @threads_per_block_on_fftz_3, align 4, !dbg !4414
  %call119 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay118, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.43, i64 0, i64 0), i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.56, i64 0, i64 0), i32 %58) #11, !dbg !4415
  %arraydecay120 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !4416
  %arraydecay121 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !4417
  %call122 = call i8* @strcat(i8* %arraydecay120, i8* %arraydecay121) #11, !dbg !4418
  %arraydecay123 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !4419
  %59 = load i32, i32* @threads_per_block_on_checksum, align 4, !dbg !4420
  %call124 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay123, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.43, i64 0, i64 0), i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.57, i64 0, i64 0), i32 %59) #11, !dbg !4421
  %arraydecay125 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !4422
  %arraydecay126 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !4423
  %call127 = call i8* @strcat(i8* %arraydecay125, i8* %arraydecay126) #11, !dbg !4424
  %60 = load i8, i8* %class_npb, align 1, !dbg !4425
  %61 = load i32, i32* @_ZL5niter, align 4, !dbg !4426
  %62 = load double, double* %total_time, align 8, !dbg !4427
  %63 = load double, double* %mflops, align 8, !dbg !4428
  %64 = load i32, i32* %verified, align 4, !dbg !4429
  %arraydecay128 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !4430
  call void @_Z15c_print_resultsPcciiiiddS_iS_S_S_S_S_S_S_S_S_S_S_S_S_S_(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.58, i64 0, i64 0), i8 signext %60, i32 256, i32 256, i32 128, i32 %61, double %62, double %63, i8* getelementptr inbounds ([25 x i8], [25 x i8]* @.str.59, i64 0, i64 0), i32 %64, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.60, i64 0, i64 0), i8* getelementptr inbounds ([12 x i8], [12 x i8]* @.str.61, i64 0, i64 0), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.62, i64 0, i64 0), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.62, i64 0, i64 0), i8* getelementptr inbounds ([42 x i8], [42 x i8]* @.str.63, i64 0, i64 0), i8* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 0, i64 0), i8* %arraydecay128, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.64, i64 0, i64 0), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.65, i64 0, i64 0), i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.66, i64 0, i64 0), i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.67, i64 0, i64 0), i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.68, i64 0, i64 0), i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.68, i64 0, i64 0), i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.69, i64 0, i64 0)), !dbg !4431
  call void @_ZL11release_gpuv(), !dbg !4432
  %65 = load %struct.dcomplex*, %struct.dcomplex** @_ZL4sums, align 8, !dbg !4433
  %66 = bitcast %struct.dcomplex* %65 to i8*, !dbg !4433
  call void @free(i8* %66) #11, !dbg !4434
  %67 = load double*, double** @_ZL7twiddle, align 8, !dbg !4435
  %68 = bitcast double* %67 to i8*, !dbg !4435
  call void @free(i8* %68) #11, !dbg !4436
  %69 = load %struct.dcomplex*, %struct.dcomplex** @_ZL1u, align 8, !dbg !4437
  %70 = bitcast %struct.dcomplex* %69 to i8*, !dbg !4437
  call void @free(i8* %70) #11, !dbg !4438
  %71 = load %struct.dcomplex*, %struct.dcomplex** @_ZL2u0, align 8, !dbg !4439
  %72 = bitcast %struct.dcomplex* %71 to i8*, !dbg !4439
  call void @free(i8* %72) #11, !dbg !4440
  %73 = load %struct.dcomplex*, %struct.dcomplex** @_ZL2u1, align 8, !dbg !4441
  %74 = bitcast %struct.dcomplex* %73 to i8*, !dbg !4441
  call void @free(i8* %74) #11, !dbg !4442
  %75 = load i32*, i32** @_ZL4dims, align 8, !dbg !4443
  %76 = bitcast i32* %75 to i8*, !dbg !4443
  call void @free(i8* %76) #11, !dbg !4444
  ret i32 0, !dbg !4445
}

; Function Attrs: nounwind
declare dso_local noalias i8* @malloc(i64) #9

; Function Attrs: noinline uwtable
define internal void @_ZL5setupv() #7 !dbg !4446 {
entry:
  store i32 6, i32* @_ZL5niter, align 4, !dbg !4447
  %call = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([65 x i8], [65 x i8]* @.str.70, i64 0, i64 0)), !dbg !4448
  %call1 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([36 x i8], [36 x i8]* @.str.71, i64 0, i64 0), i32 256, i32 256, i32 128), !dbg !4449
  %0 = load i32, i32* @_ZL5niter, align 4, !dbg !4450
  %call2 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([35 x i8], [35 x i8]* @.str.72, i64 0, i64 0), i32 %0), !dbg !4451
  %call3 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.32, i64 0, i64 0)), !dbg !4452
  ret void, !dbg !4453
}

; Function Attrs: noinline uwtable
define internal void @_ZL9setup_gpuv() #7 !dbg !4454 {
entry:
  store i32 32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !4455
  store i32 32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !4456
  %0 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !4457
  %cmp = icmp sle i32 32, %0, !dbg !4459
  br i1 %cmp, label %if.then, label %if.else, !dbg !4460

if.then:                                          ; preds = %entry
  store i32 32, i32* @threads_per_block_on_compute_indexmap, align 4, !dbg !4461
  br label %if.end, !dbg !4463

if.else:                                          ; preds = %entry
  %1 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !4464
  store i32 %1, i32* @threads_per_block_on_compute_indexmap, align 4, !dbg !4466
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %2 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !4467
  %cmp1 = icmp sle i32 32, %2, !dbg !4469
  br i1 %cmp1, label %if.then2, label %if.else3, !dbg !4470

if.then2:                                         ; preds = %if.end
  store i32 32, i32* @threads_per_block_on_compute_initial_conditions, align 4, !dbg !4471
  br label %if.end4, !dbg !4473

if.else3:                                         ; preds = %if.end
  %3 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !4474
  store i32 %3, i32* @threads_per_block_on_compute_initial_conditions, align 4, !dbg !4476
  br label %if.end4

if.end4:                                          ; preds = %if.else3, %if.then2
  %4 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !4477
  %cmp5 = icmp sle i32 32, %4, !dbg !4479
  br i1 %cmp5, label %if.then6, label %if.else7, !dbg !4480

if.then6:                                         ; preds = %if.end4
  store i32 32, i32* @threads_per_block_on_init_ui, align 4, !dbg !4481
  br label %if.end8, !dbg !4483

if.else7:                                         ; preds = %if.end4
  %5 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !4484
  store i32 %5, i32* @threads_per_block_on_init_ui, align 4, !dbg !4486
  br label %if.end8

if.end8:                                          ; preds = %if.else7, %if.then6
  %6 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !4487
  %cmp9 = icmp sle i32 32, %6, !dbg !4489
  br i1 %cmp9, label %if.then10, label %if.else11, !dbg !4490

if.then10:                                        ; preds = %if.end8
  store i32 32, i32* @threads_per_block_on_evolve, align 4, !dbg !4491
  br label %if.end12, !dbg !4493

if.else11:                                        ; preds = %if.end8
  %7 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !4494
  store i32 %7, i32* @threads_per_block_on_evolve, align 4, !dbg !4496
  br label %if.end12

if.end12:                                         ; preds = %if.else11, %if.then10
  %8 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !4497
  %cmp13 = icmp sle i32 1024, %8, !dbg !4499
  br i1 %cmp13, label %if.then14, label %if.else15, !dbg !4500

if.then14:                                        ; preds = %if.end12
  store i32 1024, i32* @threads_per_block_on_fftx_1, align 4, !dbg !4501
  br label %if.end16, !dbg !4503

if.else15:                                        ; preds = %if.end12
  %9 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !4504
  store i32 %9, i32* @threads_per_block_on_fftx_1, align 4, !dbg !4506
  br label %if.end16

if.end16:                                         ; preds = %if.else15, %if.then14
  %10 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !4507
  %cmp17 = icmp sle i32 32, %10, !dbg !4509
  br i1 %cmp17, label %if.then18, label %if.else19, !dbg !4510

if.then18:                                        ; preds = %if.end16
  store i32 32, i32* @threads_per_block_on_fftx_2, align 4, !dbg !4511
  br label %if.end20, !dbg !4513

if.else19:                                        ; preds = %if.end16
  %11 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !4514
  store i32 %11, i32* @threads_per_block_on_fftx_2, align 4, !dbg !4516
  br label %if.end20

if.end20:                                         ; preds = %if.else19, %if.then18
  %12 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !4517
  %cmp21 = icmp sle i32 256, %12, !dbg !4519
  br i1 %cmp21, label %if.then22, label %if.else23, !dbg !4520

if.then22:                                        ; preds = %if.end20
  store i32 256, i32* @threads_per_block_on_fftx_3, align 4, !dbg !4521
  br label %if.end24, !dbg !4523

if.else23:                                        ; preds = %if.end20
  %13 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !4524
  store i32 %13, i32* @threads_per_block_on_fftx_3, align 4, !dbg !4526
  br label %if.end24

if.end24:                                         ; preds = %if.else23, %if.then22
  %14 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !4527
  %cmp25 = icmp sle i32 32, %14, !dbg !4529
  br i1 %cmp25, label %if.then26, label %if.else27, !dbg !4530

if.then26:                                        ; preds = %if.end24
  store i32 32, i32* @threads_per_block_on_ffty_1, align 4, !dbg !4531
  br label %if.end28, !dbg !4533

if.else27:                                        ; preds = %if.end24
  %15 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !4534
  store i32 %15, i32* @threads_per_block_on_ffty_1, align 4, !dbg !4536
  br label %if.end28

if.end28:                                         ; preds = %if.else27, %if.then26
  %16 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !4537
  %cmp29 = icmp sle i32 32, %16, !dbg !4539
  br i1 %cmp29, label %if.then30, label %if.else31, !dbg !4540

if.then30:                                        ; preds = %if.end28
  store i32 32, i32* @threads_per_block_on_ffty_2, align 4, !dbg !4541
  br label %if.end32, !dbg !4543

if.else31:                                        ; preds = %if.end28
  %17 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !4544
  store i32 %17, i32* @threads_per_block_on_ffty_2, align 4, !dbg !4546
  br label %if.end32

if.end32:                                         ; preds = %if.else31, %if.then30
  %18 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !4547
  %cmp33 = icmp sle i32 32, %18, !dbg !4549
  br i1 %cmp33, label %if.then34, label %if.else35, !dbg !4550

if.then34:                                        ; preds = %if.end32
  store i32 32, i32* @threads_per_block_on_ffty_3, align 4, !dbg !4551
  br label %if.end36, !dbg !4553

if.else35:                                        ; preds = %if.end32
  %19 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !4554
  store i32 %19, i32* @threads_per_block_on_ffty_3, align 4, !dbg !4556
  br label %if.end36

if.end36:                                         ; preds = %if.else35, %if.then34
  %20 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !4557
  %cmp37 = icmp sle i32 32, %20, !dbg !4559
  br i1 %cmp37, label %if.then38, label %if.else39, !dbg !4560

if.then38:                                        ; preds = %if.end36
  store i32 32, i32* @threads_per_block_on_fftz_1, align 4, !dbg !4561
  br label %if.end40, !dbg !4563

if.else39:                                        ; preds = %if.end36
  %21 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !4564
  store i32 %21, i32* @threads_per_block_on_fftz_1, align 4, !dbg !4566
  br label %if.end40

if.end40:                                         ; preds = %if.else39, %if.then38
  %22 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !4567
  %cmp41 = icmp sle i32 32, %22, !dbg !4569
  br i1 %cmp41, label %if.then42, label %if.else43, !dbg !4570

if.then42:                                        ; preds = %if.end40
  store i32 32, i32* @threads_per_block_on_fftz_2, align 4, !dbg !4571
  br label %if.end44, !dbg !4573

if.else43:                                        ; preds = %if.end40
  %23 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !4574
  store i32 %23, i32* @threads_per_block_on_fftz_2, align 4, !dbg !4576
  br label %if.end44

if.end44:                                         ; preds = %if.else43, %if.then42
  %24 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !4577
  %cmp45 = icmp sle i32 32, %24, !dbg !4579
  br i1 %cmp45, label %if.then46, label %if.else47, !dbg !4580

if.then46:                                        ; preds = %if.end44
  store i32 32, i32* @threads_per_block_on_fftz_3, align 4, !dbg !4581
  br label %if.end48, !dbg !4583

if.else47:                                        ; preds = %if.end44
  %25 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !4584
  store i32 %25, i32* @threads_per_block_on_fftz_3, align 4, !dbg !4586
  br label %if.end48

if.end48:                                         ; preds = %if.else47, %if.then46
  %26 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !4587
  %cmp49 = icmp sle i32 32, %26, !dbg !4589
  br i1 %cmp49, label %if.then50, label %if.else51, !dbg !4590

if.then50:                                        ; preds = %if.end48
  store i32 32, i32* @threads_per_block_on_checksum, align 4, !dbg !4591
  br label %if.end52, !dbg !4593

if.else51:                                        ; preds = %if.end48
  %27 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !4594
  store i32 %27, i32* @threads_per_block_on_checksum, align 4, !dbg !4596
  br label %if.end52

if.end52:                                         ; preds = %if.else51, %if.then50
  %28 = load i32, i32* @threads_per_block_on_compute_indexmap, align 4, !dbg !4597
  %conv = sitofp i32 %28 to double, !dbg !4597
  %div = fdiv double 0x4160000000000000, %conv, !dbg !4598
  %29 = call double @llvm.ceil.f64(double %div), !dbg !4599
  %conv53 = fptosi double %29 to i32, !dbg !4599
  store i32 %conv53, i32* @blocks_per_grid_on_compute_indexmap, align 4, !dbg !4600
  %30 = load i32, i32* @threads_per_block_on_compute_initial_conditions, align 4, !dbg !4601
  %conv54 = sitofp i32 %30 to double, !dbg !4601
  %div55 = fdiv double 1.280000e+02, %conv54, !dbg !4602
  %31 = call double @llvm.ceil.f64(double %div55), !dbg !4603
  %conv56 = fptosi double %31 to i32, !dbg !4603
  store i32 %conv56, i32* @blocks_per_grid_on_compute_initial_conditions, align 4, !dbg !4604
  %32 = load i32, i32* @threads_per_block_on_init_ui, align 4, !dbg !4605
  %conv57 = sitofp i32 %32 to double, !dbg !4605
  %div58 = fdiv double 0x4160000000000000, %conv57, !dbg !4606
  %33 = call double @llvm.ceil.f64(double %div58), !dbg !4607
  %conv59 = fptosi double %33 to i32, !dbg !4607
  store i32 %conv59, i32* @blocks_per_grid_on_init_ui, align 4, !dbg !4608
  %34 = load i32, i32* @threads_per_block_on_evolve, align 4, !dbg !4609
  %conv60 = sitofp i32 %34 to double, !dbg !4609
  %div61 = fdiv double 0x4160000000000000, %conv60, !dbg !4610
  %35 = call double @llvm.ceil.f64(double %div61), !dbg !4611
  %conv62 = fptosi double %35 to i32, !dbg !4611
  store i32 %conv62, i32* @blocks_per_grid_on_evolve, align 4, !dbg !4612
  %36 = load i32, i32* @threads_per_block_on_fftx_1, align 4, !dbg !4613
  %conv63 = sitofp i32 %36 to double, !dbg !4613
  %div64 = fdiv double 0x4160000000000000, %conv63, !dbg !4614
  %37 = call double @llvm.ceil.f64(double %div64), !dbg !4615
  %conv65 = fptosi double %37 to i32, !dbg !4615
  store i32 %conv65, i32* @blocks_per_grid_on_fftx_1, align 4, !dbg !4616
  %38 = load i32, i32* @threads_per_block_on_fftx_2, align 4, !dbg !4617
  %conv66 = sitofp i32 %38 to double, !dbg !4617
  %div67 = fdiv double 3.276800e+04, %conv66, !dbg !4618
  %39 = call double @llvm.ceil.f64(double %div67), !dbg !4619
  %conv68 = fptosi double %39 to i32, !dbg !4619
  store i32 %conv68, i32* @blocks_per_grid_on_fftx_2, align 4, !dbg !4620
  %40 = load i32, i32* @threads_per_block_on_fftx_3, align 4, !dbg !4621
  %conv69 = sitofp i32 %40 to double, !dbg !4621
  %div70 = fdiv double 0x4160000000000000, %conv69, !dbg !4622
  %41 = call double @llvm.ceil.f64(double %div70), !dbg !4623
  %conv71 = fptosi double %41 to i32, !dbg !4623
  store i32 %conv71, i32* @blocks_per_grid_on_fftx_3, align 4, !dbg !4624
  %42 = load i32, i32* @threads_per_block_on_ffty_1, align 4, !dbg !4625
  %conv72 = sitofp i32 %42 to double, !dbg !4625
  %div73 = fdiv double 0x4160000000000000, %conv72, !dbg !4626
  %43 = call double @llvm.ceil.f64(double %div73), !dbg !4627
  %conv74 = fptosi double %43 to i32, !dbg !4627
  store i32 %conv74, i32* @blocks_per_grid_on_ffty_1, align 4, !dbg !4628
  %44 = load i32, i32* @threads_per_block_on_ffty_2, align 4, !dbg !4629
  %conv75 = sitofp i32 %44 to double, !dbg !4629
  %div76 = fdiv double 3.276800e+04, %conv75, !dbg !4630
  %45 = call double @llvm.ceil.f64(double %div76), !dbg !4631
  %conv77 = fptosi double %45 to i32, !dbg !4631
  store i32 %conv77, i32* @blocks_per_grid_on_ffty_2, align 4, !dbg !4632
  %46 = load i32, i32* @threads_per_block_on_ffty_3, align 4, !dbg !4633
  %conv78 = sitofp i32 %46 to double, !dbg !4633
  %div79 = fdiv double 0x4160000000000000, %conv78, !dbg !4634
  %47 = call double @llvm.ceil.f64(double %div79), !dbg !4635
  %conv80 = fptosi double %47 to i32, !dbg !4635
  store i32 %conv80, i32* @blocks_per_grid_on_ffty_3, align 4, !dbg !4636
  %48 = load i32, i32* @threads_per_block_on_fftz_1, align 4, !dbg !4637
  %conv81 = sitofp i32 %48 to double, !dbg !4637
  %div82 = fdiv double 0x4160000000000000, %conv81, !dbg !4638
  %49 = call double @llvm.ceil.f64(double %div82), !dbg !4639
  %conv83 = fptosi double %49 to i32, !dbg !4639
  store i32 %conv83, i32* @blocks_per_grid_on_fftz_1, align 4, !dbg !4640
  %50 = load i32, i32* @threads_per_block_on_fftz_2, align 4, !dbg !4641
  %conv84 = sitofp i32 %50 to double, !dbg !4641
  %div85 = fdiv double 6.553600e+04, %conv84, !dbg !4642
  %51 = call double @llvm.ceil.f64(double %div85), !dbg !4643
  %conv86 = fptosi double %51 to i32, !dbg !4643
  store i32 %conv86, i32* @blocks_per_grid_on_fftz_2, align 4, !dbg !4644
  %52 = load i32, i32* @threads_per_block_on_fftz_3, align 4, !dbg !4645
  %conv87 = sitofp i32 %52 to double, !dbg !4645
  %div88 = fdiv double 0x4160000000000000, %conv87, !dbg !4646
  %53 = call double @llvm.ceil.f64(double %div88), !dbg !4647
  %conv89 = fptosi double %53 to i32, !dbg !4647
  store i32 %conv89, i32* @blocks_per_grid_on_fftz_3, align 4, !dbg !4648
  %54 = load i32, i32* @threads_per_block_on_checksum, align 4, !dbg !4649
  %conv90 = sitofp i32 %54 to double, !dbg !4649
  %div91 = fdiv double 1.024000e+03, %conv90, !dbg !4650
  %55 = call double @llvm.ceil.f64(double %div91), !dbg !4651
  %conv92 = fptosi double %55 to i32, !dbg !4651
  store i32 %conv92, i32* @blocks_per_grid_on_checksum, align 4, !dbg !4652
  store i64 112, i64* @size_sums_device, align 8, !dbg !4653
  store i64 1024, i64* @size_starts_device, align 8, !dbg !4654
  store i64 67108864, i64* @size_twiddle_device, align 8, !dbg !4655
  store i64 4096, i64* @size_u_device, align 8, !dbg !4656
  store i64 134217728, i64* @size_u0_device, align 8, !dbg !4657
  store i64 134217728, i64* @size_u1_device, align 8, !dbg !4658
  store i64 134217728, i64* @size_y0_device, align 8, !dbg !4659
  store i64 134217728, i64* @size_y1_device, align 8, !dbg !4660
  %56 = load i32, i32* @threads_per_block_on_checksum, align 4, !dbg !4661
  %conv93 = sext i32 %56 to i64, !dbg !4661
  %mul = mul i64 %conv93, 16, !dbg !4662
  store i64 %mul, i64* @size_shared_data, align 8, !dbg !4663
  %57 = load i64, i64* @size_sums_device, align 8, !dbg !4664
  %call = call i32 @_ZL10cudaMallocI8dcomplexE9cudaErrorPPT_m(%struct.dcomplex** @sums_device, i64 %57), !dbg !4665
  %58 = load i64, i64* @size_starts_device, align 8, !dbg !4666
  %call94 = call i32 @_ZL10cudaMallocIdE9cudaErrorPPT_m(double** @starts_device, i64 %58), !dbg !4667
  %59 = load i64, i64* @size_twiddle_device, align 8, !dbg !4668
  %call95 = call i32 @_ZL10cudaMallocIdE9cudaErrorPPT_m(double** @twiddle_device, i64 %59), !dbg !4669
  %60 = load i64, i64* @size_u_device, align 8, !dbg !4670
  %call96 = call i32 @_ZL10cudaMallocI8dcomplexE9cudaErrorPPT_m(%struct.dcomplex** @u_device, i64 %60), !dbg !4671
  %61 = load i64, i64* @size_u0_device, align 8, !dbg !4672
  %call97 = call i32 @_ZL10cudaMallocI8dcomplexE9cudaErrorPPT_m(%struct.dcomplex** @u0_device, i64 %61), !dbg !4673
  %62 = load i64, i64* @size_u1_device, align 8, !dbg !4674
  %call98 = call i32 @_ZL10cudaMallocI8dcomplexE9cudaErrorPPT_m(%struct.dcomplex** @u1_device, i64 %62), !dbg !4675
  %63 = load i64, i64* @size_y0_device, align 8, !dbg !4676
  %call99 = call i32 @_ZL10cudaMallocI8dcomplexE9cudaErrorPPT_m(%struct.dcomplex** @y0_device, i64 %63), !dbg !4677
  %64 = load i64, i64* @size_y1_device, align 8, !dbg !4678
  %call100 = call i32 @_ZL10cudaMallocI8dcomplexE9cudaErrorPPT_m(%struct.dcomplex** @y1_device, i64 %64), !dbg !4679
  call void @omp_set_num_threads(i32 3), !dbg !4680
  ret void, !dbg !4681
}

; Function Attrs: noinline uwtable
define internal void @_ZL11init_ui_gpuP8dcomplexS0_Pd(%struct.dcomplex* %u0, %struct.dcomplex* %u1, double* %twiddle) #7 !dbg !4682 {
entry:
  %u0.addr = alloca %struct.dcomplex*, align 8
  %u1.addr = alloca %struct.dcomplex*, align 8
  %twiddle.addr = alloca double*, align 8
  %agg.tmp = alloca %struct.dim3, align 4
  %agg.tmp1 = alloca %struct.dim3, align 4
  %agg.tmp.coerce = alloca { i64, i32 }, align 4
  %agg.tmp1.coerce = alloca { i64, i32 }, align 4
  store %struct.dcomplex* %u0, %struct.dcomplex** %u0.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %u0.addr, metadata !4683, metadata !DIExpression()), !dbg !4684
  store %struct.dcomplex* %u1, %struct.dcomplex** %u1.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %u1.addr, metadata !4685, metadata !DIExpression()), !dbg !4686
  store double* %twiddle, double** %twiddle.addr, align 8
  call void @llvm.dbg.declare(metadata double** %twiddle.addr, metadata !4687, metadata !DIExpression()), !dbg !4688
  %0 = load i32, i32* @blocks_per_grid_on_init_ui, align 4, !dbg !4689
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp, i32 %0, i32 1, i32 1), !dbg !4689
  %1 = load i32, i32* @threads_per_block_on_init_ui, align 4, !dbg !4690
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp1, i32 %1, i32 1, i32 1), !dbg !4690
  %2 = bitcast { i64, i32 }* %agg.tmp.coerce to i8*, !dbg !4691
  %3 = bitcast %struct.dim3* %agg.tmp to i8*, !dbg !4691
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %2, i8* align 4 %3, i64 12, i1 false), !dbg !4691
  %4 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 0, !dbg !4691
  %5 = load i64, i64* %4, align 4, !dbg !4691
  %6 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 1, !dbg !4691
  %7 = load i32, i32* %6, align 4, !dbg !4691
  %8 = bitcast { i64, i32 }* %agg.tmp1.coerce to i8*, !dbg !4691
  %9 = bitcast %struct.dim3* %agg.tmp1 to i8*, !dbg !4691
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %8, i8* align 4 %9, i64 12, i1 false), !dbg !4691
  %10 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp1.coerce, i32 0, i32 0, !dbg !4691
  %11 = load i64, i64* %10, align 4, !dbg !4691
  %12 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp1.coerce, i32 0, i32 1, !dbg !4691
  %13 = load i32, i32* %12, align 4, !dbg !4691
  %call = call i32 @cudaConfigureCall(i64 %5, i32 %7, i64 %11, i32 %13, i64 0, %struct.CUstream_st* null), !dbg !4691
  %tobool = icmp ne i32 %call, 0, !dbg !4691
  br i1 %tobool, label %kcall.end, label %kcall.configok, !dbg !4692

kcall.configok:                                   ; preds = %entry
  %14 = load %struct.dcomplex*, %struct.dcomplex** %u0.addr, align 8, !dbg !4693
  %15 = load %struct.dcomplex*, %struct.dcomplex** %u1.addr, align 8, !dbg !4694
  %16 = load double*, double** %twiddle.addr, align 8, !dbg !4695
  call void @ft.ll_CudaFE__Z18init_ui_gpu_kernelP8dcomplexS0_Pd(%struct.dcomplex* %14, %struct.dcomplex* %15, double* %16), !dbg !4692
  br label %kcall.end, !dbg !4692

kcall.end:                                        ; preds = %kcall.configok, %entry
  ret void, !dbg !4696
}

declare dso_local i32 @omp_get_thread_num() #8

; Function Attrs: noinline uwtable
define internal void @_ZL20compute_indexmap_gpuPd(double* %twiddle) #7 !dbg !4697 {
entry:
  %twiddle.addr = alloca double*, align 8
  %agg.tmp = alloca %struct.dim3, align 4
  %agg.tmp1 = alloca %struct.dim3, align 4
  %agg.tmp.coerce = alloca { i64, i32 }, align 4
  %agg.tmp1.coerce = alloca { i64, i32 }, align 4
  store double* %twiddle, double** %twiddle.addr, align 8
  call void @llvm.dbg.declare(metadata double** %twiddle.addr, metadata !4698, metadata !DIExpression()), !dbg !4699
  %0 = load i32, i32* @blocks_per_grid_on_compute_indexmap, align 4, !dbg !4700
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp, i32 %0, i32 1, i32 1), !dbg !4700
  %1 = load i32, i32* @threads_per_block_on_compute_indexmap, align 4, !dbg !4701
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp1, i32 %1, i32 1, i32 1), !dbg !4701
  %2 = bitcast { i64, i32 }* %agg.tmp.coerce to i8*, !dbg !4702
  %3 = bitcast %struct.dim3* %agg.tmp to i8*, !dbg !4702
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %2, i8* align 4 %3, i64 12, i1 false), !dbg !4702
  %4 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 0, !dbg !4702
  %5 = load i64, i64* %4, align 4, !dbg !4702
  %6 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 1, !dbg !4702
  %7 = load i32, i32* %6, align 4, !dbg !4702
  %8 = bitcast { i64, i32 }* %agg.tmp1.coerce to i8*, !dbg !4702
  %9 = bitcast %struct.dim3* %agg.tmp1 to i8*, !dbg !4702
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %8, i8* align 4 %9, i64 12, i1 false), !dbg !4702
  %10 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp1.coerce, i32 0, i32 0, !dbg !4702
  %11 = load i64, i64* %10, align 4, !dbg !4702
  %12 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp1.coerce, i32 0, i32 1, !dbg !4702
  %13 = load i32, i32* %12, align 4, !dbg !4702
  %call = call i32 @cudaConfigureCall(i64 %5, i32 %7, i64 %11, i32 %13, i64 0, %struct.CUstream_st* null), !dbg !4702
  %tobool = icmp ne i32 %call, 0, !dbg !4702
  br i1 %tobool, label %kcall.end, label %kcall.configok, !dbg !4703

kcall.configok:                                   ; preds = %entry
  %14 = load double*, double** %twiddle.addr, align 8, !dbg !4704
  call void @ft.ll_CudaFE__Z27compute_indexmap_gpu_kernelPd(double* %14), !dbg !4703
  br label %kcall.end, !dbg !4703

kcall.end:                                        ; preds = %kcall.configok, %entry
  ret void, !dbg !4705
}

; Function Attrs: noinline uwtable
define internal void @_ZL30compute_initial_conditions_gpuP8dcomplex(%struct.dcomplex* %u0) #7 !dbg !4706 {
entry:
  %u0.addr = alloca %struct.dcomplex*, align 8
  %z = alloca i32, align 4
  %start = alloca double, align 8
  %an = alloca double, align 8
  %starts = alloca [128 x double], align 16
  %agg.tmp = alloca %struct.dim3, align 4
  %agg.tmp4 = alloca %struct.dim3, align 4
  %agg.tmp.coerce = alloca { i64, i32 }, align 4
  %agg.tmp4.coerce = alloca { i64, i32 }, align 4
  store %struct.dcomplex* %u0, %struct.dcomplex** %u0.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %u0.addr, metadata !4709, metadata !DIExpression()), !dbg !4710
  call void @llvm.dbg.declare(metadata i32* %z, metadata !4711, metadata !DIExpression()), !dbg !4712
  call void @llvm.dbg.declare(metadata double* %start, metadata !4713, metadata !DIExpression()), !dbg !4714
  call void @llvm.dbg.declare(metadata double* %an, metadata !4715, metadata !DIExpression()), !dbg !4716
  call void @llvm.dbg.declare(metadata [128 x double]* %starts, metadata !4717, metadata !DIExpression()), !dbg !4721
  store double 0x41B2B9B0A1000000, double* %start, align 8, !dbg !4722
  call void @_ZL6ipow46diPd(double 0x41D2309CE5400000, i32 0, double* %an), !dbg !4723
  %0 = load double, double* %an, align 8, !dbg !4724
  %call = call double @_Z6randlcPdd(double* %start, double %0), !dbg !4725
  call void @_ZL6ipow46diPd(double 0x41D2309CE5400000, i32 131072, double* %an), !dbg !4726
  %1 = load double, double* %start, align 8, !dbg !4727
  %arrayidx = getelementptr inbounds [128 x double], [128 x double]* %starts, i64 0, i64 0, !dbg !4728
  store double %1, double* %arrayidx, align 16, !dbg !4729
  store i32 1, i32* %z, align 4, !dbg !4730
  br label %for.cond, !dbg !4732

for.cond:                                         ; preds = %for.inc, %entry
  %2 = load i32, i32* %z, align 4, !dbg !4733
  %cmp = icmp slt i32 %2, 128, !dbg !4735
  br i1 %cmp, label %for.body, label %for.end, !dbg !4736

for.body:                                         ; preds = %for.cond
  %3 = load double, double* %an, align 8, !dbg !4737
  %call1 = call double @_Z6randlcPdd(double* %start, double %3), !dbg !4739
  %4 = load double, double* %start, align 8, !dbg !4740
  %5 = load i32, i32* %z, align 4, !dbg !4741
  %idxprom = sext i32 %5 to i64, !dbg !4742
  %arrayidx2 = getelementptr inbounds [128 x double], [128 x double]* %starts, i64 0, i64 %idxprom, !dbg !4742
  store double %4, double* %arrayidx2, align 8, !dbg !4743
  br label %for.inc, !dbg !4744

for.inc:                                          ; preds = %for.body
  %6 = load i32, i32* %z, align 4, !dbg !4745
  %inc = add nsw i32 %6, 1, !dbg !4745
  store i32 %inc, i32* %z, align 4, !dbg !4745
  br label %for.cond, !dbg !4746, !llvm.loop !4747

for.end:                                          ; preds = %for.cond
  %7 = load double*, double** @starts_device, align 8, !dbg !4749
  %8 = bitcast double* %7 to i8*, !dbg !4749
  %arraydecay = getelementptr inbounds [128 x double], [128 x double]* %starts, i64 0, i64 0, !dbg !4750
  %9 = bitcast double* %arraydecay to i8*, !dbg !4750
  %10 = load i64, i64* @size_starts_device, align 8, !dbg !4751
  %call3 = call i32 @cudaMemcpy(i8* %8, i8* %9, i64 %10, i32 1), !dbg !4752
  %11 = load i32, i32* @blocks_per_grid_on_compute_initial_conditions, align 4, !dbg !4753
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp, i32 %11, i32 1, i32 1), !dbg !4753
  %12 = load i32, i32* @threads_per_block_on_compute_initial_conditions, align 4, !dbg !4754
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp4, i32 %12, i32 1, i32 1), !dbg !4754
  %13 = bitcast { i64, i32 }* %agg.tmp.coerce to i8*, !dbg !4755
  %14 = bitcast %struct.dim3* %agg.tmp to i8*, !dbg !4755
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %13, i8* align 4 %14, i64 12, i1 false), !dbg !4755
  %15 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 0, !dbg !4755
  %16 = load i64, i64* %15, align 4, !dbg !4755
  %17 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 1, !dbg !4755
  %18 = load i32, i32* %17, align 4, !dbg !4755
  %19 = bitcast { i64, i32 }* %agg.tmp4.coerce to i8*, !dbg !4755
  %20 = bitcast %struct.dim3* %agg.tmp4 to i8*, !dbg !4755
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %19, i8* align 4 %20, i64 12, i1 false), !dbg !4755
  %21 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp4.coerce, i32 0, i32 0, !dbg !4755
  %22 = load i64, i64* %21, align 4, !dbg !4755
  %23 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp4.coerce, i32 0, i32 1, !dbg !4755
  %24 = load i32, i32* %23, align 4, !dbg !4755
  %call5 = call i32 @cudaConfigureCall(i64 %16, i32 %18, i64 %22, i32 %24, i64 0, %struct.CUstream_st* null), !dbg !4755
  %tobool = icmp ne i32 %call5, 0, !dbg !4755
  br i1 %tobool, label %kcall.end, label %kcall.configok, !dbg !4756

kcall.configok:                                   ; preds = %for.end
  %25 = load %struct.dcomplex*, %struct.dcomplex** %u0.addr, align 8, !dbg !4757
  %26 = load double*, double** @starts_device, align 8, !dbg !4758
  call void @ft.ll_CudaFE__Z37compute_initial_conditions_gpu_kernelP8dcomplexPd(%struct.dcomplex* %25, double* %26), !dbg !4756
  br label %kcall.end, !dbg !4756

kcall.end:                                        ; preds = %kcall.configok, %for.end
  ret void, !dbg !4759
}

; Function Attrs: noinline uwtable
define internal void @_ZL12fft_init_gpui(i32 %n) #7 !dbg !4760 {
entry:
  %n.addr = alloca i32, align 4
  %m = alloca i32, align 4
  %ku = alloca i32, align 4
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  %ln = alloca i32, align 4
  %t = alloca double, align 8
  %ti = alloca double, align 8
  %ref.tmp = alloca %struct.dcomplex, align 8
  %ref.tmp6 = alloca %struct.dcomplex, align 8
  store i32 %n, i32* %n.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %n.addr, metadata !4761, metadata !DIExpression()), !dbg !4762
  call void @llvm.dbg.declare(metadata i32* %m, metadata !4763, metadata !DIExpression()), !dbg !4764
  call void @llvm.dbg.declare(metadata i32* %ku, metadata !4765, metadata !DIExpression()), !dbg !4766
  call void @llvm.dbg.declare(metadata i32* %i, metadata !4767, metadata !DIExpression()), !dbg !4768
  call void @llvm.dbg.declare(metadata i32* %j, metadata !4769, metadata !DIExpression()), !dbg !4770
  call void @llvm.dbg.declare(metadata i32* %ln, metadata !4771, metadata !DIExpression()), !dbg !4772
  call void @llvm.dbg.declare(metadata double* %t, metadata !4773, metadata !DIExpression()), !dbg !4774
  call void @llvm.dbg.declare(metadata double* %ti, metadata !4775, metadata !DIExpression()), !dbg !4776
  %0 = load i32, i32* %n.addr, align 4, !dbg !4777
  %call = call i32 @_ZL5ilog2i(i32 %0), !dbg !4778
  store i32 %call, i32* %m, align 4, !dbg !4779
  %real = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp, i32 0, i32 0, !dbg !4780
  %1 = load i32, i32* %m, align 4, !dbg !4780
  %conv = sitofp i32 %1 to double, !dbg !4780
  store double %conv, double* %real, align 8, !dbg !4780
  %imag = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp, i32 0, i32 1, !dbg !4780
  store double 0.000000e+00, double* %imag, align 8, !dbg !4780
  %2 = load %struct.dcomplex*, %struct.dcomplex** @_ZL1u, align 8, !dbg !4781
  %arrayidx = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %2, i64 0, !dbg !4781
  %3 = bitcast %struct.dcomplex* %arrayidx to i8*, !dbg !4782
  %4 = bitcast %struct.dcomplex* %ref.tmp to i8*, !dbg !4782
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %3, i8* align 8 %4, i64 16, i1 false), !dbg !4782
  store i32 2, i32* %ku, align 4, !dbg !4783
  store i32 1, i32* %ln, align 4, !dbg !4784
  store i32 1, i32* %j, align 4, !dbg !4785
  br label %for.cond, !dbg !4787

for.cond:                                         ; preds = %for.inc15, %entry
  %5 = load i32, i32* %j, align 4, !dbg !4788
  %6 = load i32, i32* %m, align 4, !dbg !4790
  %cmp = icmp sle i32 %5, %6, !dbg !4791
  br i1 %cmp, label %for.body, label %for.end17, !dbg !4792

for.body:                                         ; preds = %for.cond
  %7 = load i32, i32* %ln, align 4, !dbg !4793
  %conv1 = sitofp i32 %7 to double, !dbg !4793
  %div = fdiv double 0x400921FB54442D18, %conv1, !dbg !4795
  store double %div, double* %t, align 8, !dbg !4796
  store i32 0, i32* %i, align 4, !dbg !4797
  br label %for.cond2, !dbg !4799

for.cond2:                                        ; preds = %for.inc, %for.body
  %8 = load i32, i32* %i, align 4, !dbg !4800
  %9 = load i32, i32* %ln, align 4, !dbg !4802
  %sub = sub nsw i32 %9, 1, !dbg !4803
  %cmp3 = icmp sle i32 %8, %sub, !dbg !4804
  br i1 %cmp3, label %for.body4, label %for.end, !dbg !4805

for.body4:                                        ; preds = %for.cond2
  %10 = load i32, i32* %i, align 4, !dbg !4806
  %conv5 = sitofp i32 %10 to double, !dbg !4806
  %11 = load double, double* %t, align 8, !dbg !4808
  %mul = fmul contract double %conv5, %11, !dbg !4809
  store double %mul, double* %ti, align 8, !dbg !4810
  %real7 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp6, i32 0, i32 0, !dbg !4811
  %12 = load double, double* %ti, align 8, !dbg !4811
  %call8 = call double @cos(double %12) #11, !dbg !4811
  store double %call8, double* %real7, align 8, !dbg !4811
  %imag9 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp6, i32 0, i32 1, !dbg !4811
  %13 = load double, double* %ti, align 8, !dbg !4811
  %call10 = call double @sin(double %13) #11, !dbg !4811
  store double %call10, double* %imag9, align 8, !dbg !4811
  %14 = load %struct.dcomplex*, %struct.dcomplex** @_ZL1u, align 8, !dbg !4812
  %15 = load i32, i32* %i, align 4, !dbg !4813
  %16 = load i32, i32* %ku, align 4, !dbg !4814
  %add = add nsw i32 %15, %16, !dbg !4815
  %sub11 = sub nsw i32 %add, 1, !dbg !4816
  %idxprom = sext i32 %sub11 to i64, !dbg !4812
  %arrayidx12 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %14, i64 %idxprom, !dbg !4812
  %17 = bitcast %struct.dcomplex* %arrayidx12 to i8*, !dbg !4817
  %18 = bitcast %struct.dcomplex* %ref.tmp6 to i8*, !dbg !4817
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %17, i8* align 8 %18, i64 16, i1 false), !dbg !4817
  br label %for.inc, !dbg !4818

for.inc:                                          ; preds = %for.body4
  %19 = load i32, i32* %i, align 4, !dbg !4819
  %inc = add nsw i32 %19, 1, !dbg !4819
  store i32 %inc, i32* %i, align 4, !dbg !4819
  br label %for.cond2, !dbg !4820, !llvm.loop !4821

for.end:                                          ; preds = %for.cond2
  %20 = load i32, i32* %ku, align 4, !dbg !4823
  %21 = load i32, i32* %ln, align 4, !dbg !4824
  %add13 = add nsw i32 %20, %21, !dbg !4825
  store i32 %add13, i32* %ku, align 4, !dbg !4826
  %22 = load i32, i32* %ln, align 4, !dbg !4827
  %mul14 = mul nsw i32 2, %22, !dbg !4828
  store i32 %mul14, i32* %ln, align 4, !dbg !4829
  br label %for.inc15, !dbg !4830

for.inc15:                                        ; preds = %for.end
  %23 = load i32, i32* %j, align 4, !dbg !4831
  %inc16 = add nsw i32 %23, 1, !dbg !4831
  store i32 %inc16, i32* %j, align 4, !dbg !4831
  br label %for.cond, !dbg !4832, !llvm.loop !4833

for.end17:                                        ; preds = %for.cond
  %24 = load %struct.dcomplex*, %struct.dcomplex** @u_device, align 8, !dbg !4835
  %25 = bitcast %struct.dcomplex* %24 to i8*, !dbg !4835
  %26 = load %struct.dcomplex*, %struct.dcomplex** @_ZL1u, align 8, !dbg !4836
  %27 = bitcast %struct.dcomplex* %26 to i8*, !dbg !4836
  %28 = load i64, i64* @size_u_device, align 8, !dbg !4837
  %call18 = call i32 @cudaMemcpy(i8* %25, i8* %27, i64 %28, i32 1), !dbg !4838
  ret void, !dbg !4839
}

declare dso_local i32 @cudaDeviceSynchronize() #8

; Function Attrs: noinline uwtable
define internal void @_ZL7fft_gpuiP8dcomplexS0_(i32 %dir, %struct.dcomplex* %x1, %struct.dcomplex* %x2) #7 !dbg !4840 {
entry:
  %dir.addr = alloca i32, align 4
  %x1.addr = alloca %struct.dcomplex*, align 8
  %x2.addr = alloca %struct.dcomplex*, align 8
  store i32 %dir, i32* %dir.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %dir.addr, metadata !4841, metadata !DIExpression()), !dbg !4842
  store %struct.dcomplex* %x1, %struct.dcomplex** %x1.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %x1.addr, metadata !4843, metadata !DIExpression()), !dbg !4844
  store %struct.dcomplex* %x2, %struct.dcomplex** %x2.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %x2.addr, metadata !4845, metadata !DIExpression()), !dbg !4846
  %0 = load i32, i32* %dir.addr, align 4, !dbg !4847
  %cmp = icmp eq i32 %0, 1, !dbg !4849
  br i1 %cmp, label %if.then, label %if.else, !dbg !4850

if.then:                                          ; preds = %entry
  %1 = load %struct.dcomplex*, %struct.dcomplex** @u_device, align 8, !dbg !4851
  %2 = load %struct.dcomplex*, %struct.dcomplex** %x1.addr, align 8, !dbg !4853
  %3 = load %struct.dcomplex*, %struct.dcomplex** %x1.addr, align 8, !dbg !4854
  %4 = load %struct.dcomplex*, %struct.dcomplex** @y0_device, align 8, !dbg !4855
  %5 = load %struct.dcomplex*, %struct.dcomplex** @y1_device, align 8, !dbg !4856
  call void @_ZL10cffts1_gpuiP8dcomplexS0_S0_S0_S0_(i32 1, %struct.dcomplex* %1, %struct.dcomplex* %2, %struct.dcomplex* %3, %struct.dcomplex* %4, %struct.dcomplex* %5), !dbg !4857
  %6 = load %struct.dcomplex*, %struct.dcomplex** @u_device, align 8, !dbg !4858
  %7 = load %struct.dcomplex*, %struct.dcomplex** %x1.addr, align 8, !dbg !4859
  %8 = load %struct.dcomplex*, %struct.dcomplex** %x1.addr, align 8, !dbg !4860
  %9 = load %struct.dcomplex*, %struct.dcomplex** @y0_device, align 8, !dbg !4861
  %10 = load %struct.dcomplex*, %struct.dcomplex** @y1_device, align 8, !dbg !4862
  call void @_ZL10cffts2_gpuiP8dcomplexS0_S0_S0_S0_(i32 1, %struct.dcomplex* %6, %struct.dcomplex* %7, %struct.dcomplex* %8, %struct.dcomplex* %9, %struct.dcomplex* %10), !dbg !4863
  %11 = load %struct.dcomplex*, %struct.dcomplex** @u_device, align 8, !dbg !4864
  %12 = load %struct.dcomplex*, %struct.dcomplex** %x1.addr, align 8, !dbg !4865
  %13 = load %struct.dcomplex*, %struct.dcomplex** %x2.addr, align 8, !dbg !4866
  %14 = load %struct.dcomplex*, %struct.dcomplex** @y0_device, align 8, !dbg !4867
  %15 = load %struct.dcomplex*, %struct.dcomplex** @y1_device, align 8, !dbg !4868
  call void @_ZL10cffts3_gpuiP8dcomplexS0_S0_S0_S0_(i32 1, %struct.dcomplex* %11, %struct.dcomplex* %12, %struct.dcomplex* %13, %struct.dcomplex* %14, %struct.dcomplex* %15), !dbg !4869
  br label %if.end, !dbg !4870

if.else:                                          ; preds = %entry
  %16 = load %struct.dcomplex*, %struct.dcomplex** @u_device, align 8, !dbg !4871
  %17 = load %struct.dcomplex*, %struct.dcomplex** %x1.addr, align 8, !dbg !4873
  %18 = load %struct.dcomplex*, %struct.dcomplex** %x1.addr, align 8, !dbg !4874
  %19 = load %struct.dcomplex*, %struct.dcomplex** @y0_device, align 8, !dbg !4875
  %20 = load %struct.dcomplex*, %struct.dcomplex** @y1_device, align 8, !dbg !4876
  call void @_ZL10cffts3_gpuiP8dcomplexS0_S0_S0_S0_(i32 -1, %struct.dcomplex* %16, %struct.dcomplex* %17, %struct.dcomplex* %18, %struct.dcomplex* %19, %struct.dcomplex* %20), !dbg !4877
  %21 = load %struct.dcomplex*, %struct.dcomplex** @u_device, align 8, !dbg !4878
  %22 = load %struct.dcomplex*, %struct.dcomplex** %x1.addr, align 8, !dbg !4879
  %23 = load %struct.dcomplex*, %struct.dcomplex** %x1.addr, align 8, !dbg !4880
  %24 = load %struct.dcomplex*, %struct.dcomplex** @y0_device, align 8, !dbg !4881
  %25 = load %struct.dcomplex*, %struct.dcomplex** @y1_device, align 8, !dbg !4882
  call void @_ZL10cffts2_gpuiP8dcomplexS0_S0_S0_S0_(i32 -1, %struct.dcomplex* %21, %struct.dcomplex* %22, %struct.dcomplex* %23, %struct.dcomplex* %24, %struct.dcomplex* %25), !dbg !4883
  %26 = load %struct.dcomplex*, %struct.dcomplex** @u_device, align 8, !dbg !4884
  %27 = load %struct.dcomplex*, %struct.dcomplex** %x1.addr, align 8, !dbg !4885
  %28 = load %struct.dcomplex*, %struct.dcomplex** %x2.addr, align 8, !dbg !4886
  %29 = load %struct.dcomplex*, %struct.dcomplex** @y0_device, align 8, !dbg !4887
  %30 = load %struct.dcomplex*, %struct.dcomplex** @y1_device, align 8, !dbg !4888
  call void @_ZL10cffts1_gpuiP8dcomplexS0_S0_S0_S0_(i32 -1, %struct.dcomplex* %26, %struct.dcomplex* %27, %struct.dcomplex* %28, %struct.dcomplex* %29, %struct.dcomplex* %30), !dbg !4889
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void, !dbg !4890
}

; Function Attrs: noinline uwtable
define internal void @_ZL10evolve_gpuP8dcomplexS0_Pd(%struct.dcomplex* %u0, %struct.dcomplex* %u1, double* %twiddle) #7 !dbg !4891 {
entry:
  %u0.addr = alloca %struct.dcomplex*, align 8
  %u1.addr = alloca %struct.dcomplex*, align 8
  %twiddle.addr = alloca double*, align 8
  %agg.tmp = alloca %struct.dim3, align 4
  %agg.tmp1 = alloca %struct.dim3, align 4
  %agg.tmp.coerce = alloca { i64, i32 }, align 4
  %agg.tmp1.coerce = alloca { i64, i32 }, align 4
  store %struct.dcomplex* %u0, %struct.dcomplex** %u0.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %u0.addr, metadata !4892, metadata !DIExpression()), !dbg !4893
  store %struct.dcomplex* %u1, %struct.dcomplex** %u1.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %u1.addr, metadata !4894, metadata !DIExpression()), !dbg !4895
  store double* %twiddle, double** %twiddle.addr, align 8
  call void @llvm.dbg.declare(metadata double** %twiddle.addr, metadata !4896, metadata !DIExpression()), !dbg !4897
  %0 = load i32, i32* @blocks_per_grid_on_evolve, align 4, !dbg !4898
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp, i32 %0, i32 1, i32 1), !dbg !4898
  %1 = load i32, i32* @threads_per_block_on_evolve, align 4, !dbg !4899
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp1, i32 %1, i32 1, i32 1), !dbg !4899
  %2 = bitcast { i64, i32 }* %agg.tmp.coerce to i8*, !dbg !4900
  %3 = bitcast %struct.dim3* %agg.tmp to i8*, !dbg !4900
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %2, i8* align 4 %3, i64 12, i1 false), !dbg !4900
  %4 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 0, !dbg !4900
  %5 = load i64, i64* %4, align 4, !dbg !4900
  %6 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 1, !dbg !4900
  %7 = load i32, i32* %6, align 4, !dbg !4900
  %8 = bitcast { i64, i32 }* %agg.tmp1.coerce to i8*, !dbg !4900
  %9 = bitcast %struct.dim3* %agg.tmp1 to i8*, !dbg !4900
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %8, i8* align 4 %9, i64 12, i1 false), !dbg !4900
  %10 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp1.coerce, i32 0, i32 0, !dbg !4900
  %11 = load i64, i64* %10, align 4, !dbg !4900
  %12 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp1.coerce, i32 0, i32 1, !dbg !4900
  %13 = load i32, i32* %12, align 4, !dbg !4900
  %call = call i32 @cudaConfigureCall(i64 %5, i32 %7, i64 %11, i32 %13, i64 0, %struct.CUstream_st* null), !dbg !4900
  %tobool = icmp ne i32 %call, 0, !dbg !4900
  br i1 %tobool, label %kcall.end, label %kcall.configok, !dbg !4901

kcall.configok:                                   ; preds = %entry
  %14 = load %struct.dcomplex*, %struct.dcomplex** %u0.addr, align 8, !dbg !4902
  %15 = load %struct.dcomplex*, %struct.dcomplex** %u1.addr, align 8, !dbg !4903
  %16 = load double*, double** %twiddle.addr, align 8, !dbg !4904
  call void @ft.ll_CudaFE__Z17evolve_gpu_kernelP8dcomplexS0_Pd(%struct.dcomplex* %14, %struct.dcomplex* %15, double* %16), !dbg !4901
  br label %kcall.end, !dbg !4901

kcall.end:                                        ; preds = %kcall.configok, %entry
  %call2 = call i32 @cudaDeviceSynchronize(), !dbg !4905
  ret void, !dbg !4906
}

; Function Attrs: noinline uwtable
define internal void @_ZL12checksum_gpuiP8dcomplex(i32 %iteration, %struct.dcomplex* %u1) #7 !dbg !4907 {
entry:
  %iteration.addr = alloca i32, align 4
  %u1.addr = alloca %struct.dcomplex*, align 8
  %agg.tmp = alloca %struct.dim3, align 4
  %agg.tmp1 = alloca %struct.dim3, align 4
  %agg.tmp.coerce = alloca { i64, i32 }, align 4
  %agg.tmp1.coerce = alloca { i64, i32 }, align 4
  store i32 %iteration, i32* %iteration.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %iteration.addr, metadata !4910, metadata !DIExpression()), !dbg !4911
  store %struct.dcomplex* %u1, %struct.dcomplex** %u1.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %u1.addr, metadata !4912, metadata !DIExpression()), !dbg !4913
  %0 = load i32, i32* @blocks_per_grid_on_checksum, align 4, !dbg !4914
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp, i32 %0, i32 1, i32 1), !dbg !4914
  %1 = load i32, i32* @threads_per_block_on_checksum, align 4, !dbg !4915
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp1, i32 %1, i32 1, i32 1), !dbg !4915
  %2 = load i64, i64* @size_shared_data, align 8, !dbg !4916
  %3 = bitcast { i64, i32 }* %agg.tmp.coerce to i8*, !dbg !4917
  %4 = bitcast %struct.dim3* %agg.tmp to i8*, !dbg !4917
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %3, i8* align 4 %4, i64 12, i1 false), !dbg !4917
  %5 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 0, !dbg !4917
  %6 = load i64, i64* %5, align 4, !dbg !4917
  %7 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 1, !dbg !4917
  %8 = load i32, i32* %7, align 4, !dbg !4917
  %9 = bitcast { i64, i32 }* %agg.tmp1.coerce to i8*, !dbg !4917
  %10 = bitcast %struct.dim3* %agg.tmp1 to i8*, !dbg !4917
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %9, i8* align 4 %10, i64 12, i1 false), !dbg !4917
  %11 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp1.coerce, i32 0, i32 0, !dbg !4917
  %12 = load i64, i64* %11, align 4, !dbg !4917
  %13 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp1.coerce, i32 0, i32 1, !dbg !4917
  %14 = load i32, i32* %13, align 4, !dbg !4917
  %call = call i32 @cudaConfigureCall(i64 %6, i32 %8, i64 %12, i32 %14, i64 %2, %struct.CUstream_st* null), !dbg !4917
  %tobool = icmp ne i32 %call, 0, !dbg !4917
  br i1 %tobool, label %kcall.end, label %kcall.configok, !dbg !4918

kcall.configok:                                   ; preds = %entry
  %15 = load i32, i32* %iteration.addr, align 4, !dbg !4919
  %16 = load %struct.dcomplex*, %struct.dcomplex** %u1.addr, align 8, !dbg !4920
  %17 = load %struct.dcomplex*, %struct.dcomplex** @sums_device, align 8, !dbg !4921
  call void @ft.ll_CudaFE__Z19checksum_gpu_kerneliP8dcomplexS0_(i32 %15, %struct.dcomplex* %16, %struct.dcomplex* %17), !dbg !4918
  br label %kcall.end, !dbg !4918

kcall.end:                                        ; preds = %kcall.configok, %entry
  ret void, !dbg !4922
}

declare dso_local i32 @cudaMemcpy(i8*, i8*, i64, i32) #8

; Function Attrs: noinline uwtable
define internal void @_ZL6verifyiiiiPiPc(i32 %d1, i32 %d2, i32 %d3, i32 %nt, i32* %verified, i8* %class_npb) #7 !dbg !4923 {
entry:
  %d1.addr = alloca i32, align 4
  %d2.addr = alloca i32, align 4
  %d3.addr = alloca i32, align 4
  %nt.addr = alloca i32, align 4
  %verified.addr = alloca i32*, align 8
  %class_npb.addr = alloca i8*, align 8
  %i = alloca i32, align 4
  %err = alloca double, align 8
  %epsilon = alloca double, align 8
  %csum_ref = alloca [26 x %struct.dcomplex], align 16
  %ref.tmp = alloca %struct.dcomplex, align 8
  %ref.tmp6 = alloca %struct.dcomplex, align 8
  %ref.tmp10 = alloca %struct.dcomplex, align 8
  %ref.tmp14 = alloca %struct.dcomplex, align 8
  %ref.tmp18 = alloca %struct.dcomplex, align 8
  %ref.tmp22 = alloca %struct.dcomplex, align 8
  %ref.tmp34 = alloca %struct.dcomplex, align 8
  %ref.tmp38 = alloca %struct.dcomplex, align 8
  %ref.tmp42 = alloca %struct.dcomplex, align 8
  %ref.tmp46 = alloca %struct.dcomplex, align 8
  %ref.tmp50 = alloca %struct.dcomplex, align 8
  %ref.tmp54 = alloca %struct.dcomplex, align 8
  %ref.tmp67 = alloca %struct.dcomplex, align 8
  %ref.tmp71 = alloca %struct.dcomplex, align 8
  %ref.tmp75 = alloca %struct.dcomplex, align 8
  %ref.tmp79 = alloca %struct.dcomplex, align 8
  %ref.tmp83 = alloca %struct.dcomplex, align 8
  %ref.tmp87 = alloca %struct.dcomplex, align 8
  %ref.tmp100 = alloca %struct.dcomplex, align 8
  %ref.tmp104 = alloca %struct.dcomplex, align 8
  %ref.tmp108 = alloca %struct.dcomplex, align 8
  %ref.tmp112 = alloca %struct.dcomplex, align 8
  %ref.tmp116 = alloca %struct.dcomplex, align 8
  %ref.tmp120 = alloca %struct.dcomplex, align 8
  %ref.tmp124 = alloca %struct.dcomplex, align 8
  %ref.tmp128 = alloca %struct.dcomplex, align 8
  %ref.tmp132 = alloca %struct.dcomplex, align 8
  %ref.tmp136 = alloca %struct.dcomplex, align 8
  %ref.tmp140 = alloca %struct.dcomplex, align 8
  %ref.tmp144 = alloca %struct.dcomplex, align 8
  %ref.tmp148 = alloca %struct.dcomplex, align 8
  %ref.tmp152 = alloca %struct.dcomplex, align 8
  %ref.tmp156 = alloca %struct.dcomplex, align 8
  %ref.tmp160 = alloca %struct.dcomplex, align 8
  %ref.tmp164 = alloca %struct.dcomplex, align 8
  %ref.tmp168 = alloca %struct.dcomplex, align 8
  %ref.tmp172 = alloca %struct.dcomplex, align 8
  %ref.tmp176 = alloca %struct.dcomplex, align 8
  %ref.tmp189 = alloca %struct.dcomplex, align 8
  %ref.tmp193 = alloca %struct.dcomplex, align 8
  %ref.tmp197 = alloca %struct.dcomplex, align 8
  %ref.tmp201 = alloca %struct.dcomplex, align 8
  %ref.tmp205 = alloca %struct.dcomplex, align 8
  %ref.tmp209 = alloca %struct.dcomplex, align 8
  %ref.tmp213 = alloca %struct.dcomplex, align 8
  %ref.tmp217 = alloca %struct.dcomplex, align 8
  %ref.tmp221 = alloca %struct.dcomplex, align 8
  %ref.tmp225 = alloca %struct.dcomplex, align 8
  %ref.tmp229 = alloca %struct.dcomplex, align 8
  %ref.tmp233 = alloca %struct.dcomplex, align 8
  %ref.tmp237 = alloca %struct.dcomplex, align 8
  %ref.tmp241 = alloca %struct.dcomplex, align 8
  %ref.tmp245 = alloca %struct.dcomplex, align 8
  %ref.tmp249 = alloca %struct.dcomplex, align 8
  %ref.tmp253 = alloca %struct.dcomplex, align 8
  %ref.tmp257 = alloca %struct.dcomplex, align 8
  %ref.tmp261 = alloca %struct.dcomplex, align 8
  %ref.tmp265 = alloca %struct.dcomplex, align 8
  %ref.tmp278 = alloca %struct.dcomplex, align 8
  %ref.tmp282 = alloca %struct.dcomplex, align 8
  %ref.tmp286 = alloca %struct.dcomplex, align 8
  %ref.tmp290 = alloca %struct.dcomplex, align 8
  %ref.tmp294 = alloca %struct.dcomplex, align 8
  %ref.tmp298 = alloca %struct.dcomplex, align 8
  %ref.tmp302 = alloca %struct.dcomplex, align 8
  %ref.tmp306 = alloca %struct.dcomplex, align 8
  %ref.tmp310 = alloca %struct.dcomplex, align 8
  %ref.tmp314 = alloca %struct.dcomplex, align 8
  %ref.tmp318 = alloca %struct.dcomplex, align 8
  %ref.tmp322 = alloca %struct.dcomplex, align 8
  %ref.tmp326 = alloca %struct.dcomplex, align 8
  %ref.tmp330 = alloca %struct.dcomplex, align 8
  %ref.tmp334 = alloca %struct.dcomplex, align 8
  %ref.tmp338 = alloca %struct.dcomplex, align 8
  %ref.tmp342 = alloca %struct.dcomplex, align 8
  %ref.tmp346 = alloca %struct.dcomplex, align 8
  %ref.tmp350 = alloca %struct.dcomplex, align 8
  %ref.tmp354 = alloca %struct.dcomplex, align 8
  %ref.tmp358 = alloca %struct.dcomplex, align 8
  %ref.tmp362 = alloca %struct.dcomplex, align 8
  %ref.tmp366 = alloca %struct.dcomplex, align 8
  %ref.tmp370 = alloca %struct.dcomplex, align 8
  %ref.tmp374 = alloca %struct.dcomplex, align 8
  %ref.tmp387 = alloca %struct.dcomplex, align 8
  %ref.tmp391 = alloca %struct.dcomplex, align 8
  %ref.tmp395 = alloca %struct.dcomplex, align 8
  %ref.tmp399 = alloca %struct.dcomplex, align 8
  %ref.tmp403 = alloca %struct.dcomplex, align 8
  %ref.tmp407 = alloca %struct.dcomplex, align 8
  %ref.tmp411 = alloca %struct.dcomplex, align 8
  %ref.tmp415 = alloca %struct.dcomplex, align 8
  %ref.tmp419 = alloca %struct.dcomplex, align 8
  %ref.tmp423 = alloca %struct.dcomplex, align 8
  %ref.tmp427 = alloca %struct.dcomplex, align 8
  %ref.tmp431 = alloca %struct.dcomplex, align 8
  %ref.tmp435 = alloca %struct.dcomplex, align 8
  %ref.tmp439 = alloca %struct.dcomplex, align 8
  %ref.tmp443 = alloca %struct.dcomplex, align 8
  %ref.tmp447 = alloca %struct.dcomplex, align 8
  %ref.tmp451 = alloca %struct.dcomplex, align 8
  %ref.tmp455 = alloca %struct.dcomplex, align 8
  %ref.tmp459 = alloca %struct.dcomplex, align 8
  %ref.tmp463 = alloca %struct.dcomplex, align 8
  %ref.tmp467 = alloca %struct.dcomplex, align 8
  %ref.tmp471 = alloca %struct.dcomplex, align 8
  %ref.tmp475 = alloca %struct.dcomplex, align 8
  %ref.tmp479 = alloca %struct.dcomplex, align 8
  %ref.tmp483 = alloca %struct.dcomplex, align 8
  %agg.tmp = alloca %struct.dcomplex, align 8
  %agg.tmp510 = alloca %struct.dcomplex, align 8
  %coerce = alloca %struct.dcomplex, align 8
  %agg.tmp514 = alloca %struct.dcomplex, align 8
  %agg.tmp531 = alloca %struct.dcomplex, align 8
  %coerce535 = alloca %struct.dcomplex, align 8
  %agg.tmp537 = alloca %struct.dcomplex, align 8
  %agg.tmp554 = alloca %struct.dcomplex, align 8
  %coerce558 = alloca %struct.dcomplex, align 8
  %agg.tmp560 = alloca %struct.dcomplex, align 8
  %agg.tmp577 = alloca %struct.dcomplex, align 8
  %coerce581 = alloca %struct.dcomplex, align 8
  store i32 %d1, i32* %d1.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %d1.addr, metadata !4927, metadata !DIExpression()), !dbg !4928
  store i32 %d2, i32* %d2.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %d2.addr, metadata !4929, metadata !DIExpression()), !dbg !4930
  store i32 %d3, i32* %d3.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %d3.addr, metadata !4931, metadata !DIExpression()), !dbg !4932
  store i32 %nt, i32* %nt.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %nt.addr, metadata !4933, metadata !DIExpression()), !dbg !4934
  store i32* %verified, i32** %verified.addr, align 8
  call void @llvm.dbg.declare(metadata i32** %verified.addr, metadata !4935, metadata !DIExpression()), !dbg !4936
  store i8* %class_npb, i8** %class_npb.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %class_npb.addr, metadata !4937, metadata !DIExpression()), !dbg !4938
  call void @llvm.dbg.declare(metadata i32* %i, metadata !4939, metadata !DIExpression()), !dbg !4940
  call void @llvm.dbg.declare(metadata double* %err, metadata !4941, metadata !DIExpression()), !dbg !4942
  call void @llvm.dbg.declare(metadata double* %epsilon, metadata !4943, metadata !DIExpression()), !dbg !4944
  call void @llvm.dbg.declare(metadata [26 x %struct.dcomplex]* %csum_ref, metadata !4945, metadata !DIExpression()), !dbg !4949
  %0 = load i8*, i8** %class_npb.addr, align 8, !dbg !4950
  store i8 85, i8* %0, align 1, !dbg !4951
  store double 0x3D719799812DEA11, double* %epsilon, align 8, !dbg !4952
  %1 = load i32*, i32** %verified.addr, align 8, !dbg !4953
  store i32 0, i32* %1, align 4, !dbg !4954
  %2 = load i32, i32* %d1.addr, align 4, !dbg !4955
  %cmp = icmp eq i32 %2, 64, !dbg !4957
  br i1 %cmp, label %land.lhs.true, label %if.else, !dbg !4958

land.lhs.true:                                    ; preds = %entry
  %3 = load i32, i32* %d2.addr, align 4, !dbg !4959
  %cmp1 = icmp eq i32 %3, 64, !dbg !4960
  br i1 %cmp1, label %land.lhs.true2, label %if.else, !dbg !4961

land.lhs.true2:                                   ; preds = %land.lhs.true
  %4 = load i32, i32* %d3.addr, align 4, !dbg !4962
  %cmp3 = icmp eq i32 %4, 64, !dbg !4963
  br i1 %cmp3, label %land.lhs.true4, label %if.else, !dbg !4964

land.lhs.true4:                                   ; preds = %land.lhs.true2
  %5 = load i32, i32* %nt.addr, align 4, !dbg !4965
  %cmp5 = icmp eq i32 %5, 6, !dbg !4966
  br i1 %cmp5, label %if.then, label %if.else, !dbg !4967

if.then:                                          ; preds = %land.lhs.true4
  %6 = load i8*, i8** %class_npb.addr, align 8, !dbg !4968
  store i8 83, i8* %6, align 1, !dbg !4970
  %real = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp, i32 0, i32 0, !dbg !4971
  store double 0x408154DE9E5DA8C7, double* %real, align 8, !dbg !4971
  %imag = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp, i32 0, i32 1, !dbg !4971
  store double 0x407E4894D21E84F6, double* %imag, align 8, !dbg !4971
  %arrayidx = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 1, !dbg !4972
  %7 = bitcast %struct.dcomplex* %arrayidx to i8*, !dbg !4973
  %8 = bitcast %struct.dcomplex* %ref.tmp to i8*, !dbg !4973
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %7, i8* align 8 %8, i64 16, i1 false), !dbg !4973
  %real7 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp6, i32 0, i32 0, !dbg !4974
  store double 0x4081551BBB575EAB, double* %real7, align 8, !dbg !4974
  %imag8 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp6, i32 0, i32 1, !dbg !4974
  store double 0x407E687CA0F87E44, double* %imag8, align 8, !dbg !4974
  %arrayidx9 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 2, !dbg !4975
  %9 = bitcast %struct.dcomplex* %arrayidx9 to i8*, !dbg !4976
  %10 = bitcast %struct.dcomplex* %ref.tmp6 to i8*, !dbg !4976
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %9, i8* align 8 %10, i64 16, i1 false), !dbg !4976
  %real11 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp10, i32 0, i32 0, !dbg !4977
  store double 0x408154EB318EB593, double* %real11, align 8, !dbg !4977
  %imag12 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp10, i32 0, i32 1, !dbg !4977
  store double 0x407E8641D4F55AF9, double* %imag12, align 8, !dbg !4977
  %arrayidx13 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 3, !dbg !4978
  %11 = bitcast %struct.dcomplex* %arrayidx13 to i8*, !dbg !4979
  %12 = bitcast %struct.dcomplex* %ref.tmp10 to i8*, !dbg !4979
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %11, i8* align 8 %12, i64 16, i1 false), !dbg !4979
  %real15 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp14, i32 0, i32 0, !dbg !4980
  store double 0x40815456C13A7B04, double* %real15, align 8, !dbg !4980
  %imag16 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp14, i32 0, i32 1, !dbg !4980
  store double 0x407EA2097D7357C2, double* %imag16, align 8, !dbg !4980
  %arrayidx17 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 4, !dbg !4981
  %13 = bitcast %struct.dcomplex* %arrayidx17 to i8*, !dbg !4982
  %14 = bitcast %struct.dcomplex* %ref.tmp14 to i8*, !dbg !4982
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %13, i8* align 8 %14, i64 16, i1 false), !dbg !4982
  %real19 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp18, i32 0, i32 0, !dbg !4983
  store double 0x408153676E9F169C, double* %real19, align 8, !dbg !4983
  %imag20 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp18, i32 0, i32 1, !dbg !4983
  store double 0x407EBBF61C86EF29, double* %imag20, align 8, !dbg !4983
  %arrayidx21 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 5, !dbg !4984
  %15 = bitcast %struct.dcomplex* %arrayidx21 to i8*, !dbg !4985
  %16 = bitcast %struct.dcomplex* %ref.tmp18 to i8*, !dbg !4985
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %15, i8* align 8 %16, i64 16, i1 false), !dbg !4985
  %real23 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp22, i32 0, i32 0, !dbg !4986
  store double 0x408152259010E0A1, double* %real23, align 8, !dbg !4986
  %imag24 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp22, i32 0, i32 1, !dbg !4986
  store double 0x407ED427D4DF0213, double* %imag24, align 8, !dbg !4986
  %arrayidx25 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 6, !dbg !4987
  %17 = bitcast %struct.dcomplex* %arrayidx25 to i8*, !dbg !4988
  %18 = bitcast %struct.dcomplex* %ref.tmp22 to i8*, !dbg !4988
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %17, i8* align 8 %18, i64 16, i1 false), !dbg !4988
  br label %if.end492, !dbg !4989

if.else:                                          ; preds = %land.lhs.true4, %land.lhs.true2, %land.lhs.true, %entry
  %19 = load i32, i32* %d1.addr, align 4, !dbg !4990
  %cmp26 = icmp eq i32 %19, 128, !dbg !4992
  br i1 %cmp26, label %land.lhs.true27, label %if.else58, !dbg !4993

land.lhs.true27:                                  ; preds = %if.else
  %20 = load i32, i32* %d2.addr, align 4, !dbg !4994
  %cmp28 = icmp eq i32 %20, 128, !dbg !4995
  br i1 %cmp28, label %land.lhs.true29, label %if.else58, !dbg !4996

land.lhs.true29:                                  ; preds = %land.lhs.true27
  %21 = load i32, i32* %d3.addr, align 4, !dbg !4997
  %cmp30 = icmp eq i32 %21, 32, !dbg !4998
  br i1 %cmp30, label %land.lhs.true31, label %if.else58, !dbg !4999

land.lhs.true31:                                  ; preds = %land.lhs.true29
  %22 = load i32, i32* %nt.addr, align 4, !dbg !5000
  %cmp32 = icmp eq i32 %22, 6, !dbg !5001
  br i1 %cmp32, label %if.then33, label %if.else58, !dbg !5002

if.then33:                                        ; preds = %land.lhs.true31
  %23 = load i8*, i8** %class_npb.addr, align 8, !dbg !5003
  store i8 87, i8* %23, align 1, !dbg !5005
  %real35 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp34, i32 0, i32 0, !dbg !5006
  store double 0x4081BAE3C635196D, double* %real35, align 8, !dbg !5006
  %imag36 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp34, i32 0, i32 1, !dbg !5006
  store double 0x40808A98F467F156, double* %imag36, align 8, !dbg !5006
  %arrayidx37 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 1, !dbg !5007
  %24 = bitcast %struct.dcomplex* %arrayidx37 to i8*, !dbg !5008
  %25 = bitcast %struct.dcomplex* %ref.tmp34 to i8*, !dbg !5008
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %24, i8* align 8 %25, i64 16, i1 false), !dbg !5008
  %real39 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp38, i32 0, i32 0, !dbg !5009
  store double 0x40819926462BA5A4, double* %real39, align 8, !dbg !5009
  %imag40 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp38, i32 0, i32 1, !dbg !5009
  store double 0x408081B851380EB7, double* %imag40, align 8, !dbg !5009
  %arrayidx41 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 2, !dbg !5010
  %26 = bitcast %struct.dcomplex* %arrayidx41 to i8*, !dbg !5011
  %27 = bitcast %struct.dcomplex* %ref.tmp38 to i8*, !dbg !5011
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %26, i8* align 8 %27, i64 16, i1 false), !dbg !5011
  %real43 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp42, i32 0, i32 0, !dbg !5012
  store double 0x40817B3822354DD9, double* %real43, align 8, !dbg !5012
  %imag44 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp42, i32 0, i32 1, !dbg !5012
  store double 0x408078CC18578DFC, double* %imag44, align 8, !dbg !5012
  %arrayidx45 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 3, !dbg !5013
  %28 = bitcast %struct.dcomplex* %arrayidx45 to i8*, !dbg !5014
  %29 = bitcast %struct.dcomplex* %ref.tmp42 to i8*, !dbg !5014
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %28, i8* align 8 %29, i64 16, i1 false), !dbg !5014
  %real47 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp46, i32 0, i32 0, !dbg !5015
  store double 0x4081608EF5C48194, double* %real47, align 8, !dbg !5015
  %imag48 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp46, i32 0, i32 1, !dbg !5015
  store double 0x40807005B7059038, double* %imag48, align 8, !dbg !5015
  %arrayidx49 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 4, !dbg !5016
  %30 = bitcast %struct.dcomplex* %arrayidx49 to i8*, !dbg !5017
  %31 = bitcast %struct.dcomplex* %ref.tmp46 to i8*, !dbg !5017
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %30, i8* align 8 %31, i64 16, i1 false), !dbg !5017
  %real51 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp50, i32 0, i32 0, !dbg !5018
  store double 0x408148B81D084E83, double* %real51, align 8, !dbg !5018
  %imag52 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp50, i32 0, i32 1, !dbg !5018
  store double 0x408067854B0E36C9, double* %imag52, align 8, !dbg !5018
  %arrayidx53 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 5, !dbg !5019
  %32 = bitcast %struct.dcomplex* %arrayidx53 to i8*, !dbg !5020
  %33 = bitcast %struct.dcomplex* %ref.tmp50 to i8*, !dbg !5020
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %32, i8* align 8 %33, i64 16, i1 false), !dbg !5020
  %real55 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp54, i32 0, i32 0, !dbg !5021
  store double 0x40813353E9E3E09A, double* %real55, align 8, !dbg !5021
  %imag56 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp54, i32 0, i32 1, !dbg !5021
  store double 0x40805F5EAB0F5DA2, double* %imag56, align 8, !dbg !5021
  %arrayidx57 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 6, !dbg !5022
  %34 = bitcast %struct.dcomplex* %arrayidx57 to i8*, !dbg !5023
  %35 = bitcast %struct.dcomplex* %ref.tmp54 to i8*, !dbg !5023
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %34, i8* align 8 %35, i64 16, i1 false), !dbg !5023
  br label %if.end491, !dbg !5024

if.else58:                                        ; preds = %land.lhs.true31, %land.lhs.true29, %land.lhs.true27, %if.else
  %36 = load i32, i32* %d1.addr, align 4, !dbg !5025
  %cmp59 = icmp eq i32 %36, 256, !dbg !5027
  br i1 %cmp59, label %land.lhs.true60, label %if.else91, !dbg !5028

land.lhs.true60:                                  ; preds = %if.else58
  %37 = load i32, i32* %d2.addr, align 4, !dbg !5029
  %cmp61 = icmp eq i32 %37, 256, !dbg !5030
  br i1 %cmp61, label %land.lhs.true62, label %if.else91, !dbg !5031

land.lhs.true62:                                  ; preds = %land.lhs.true60
  %38 = load i32, i32* %d3.addr, align 4, !dbg !5032
  %cmp63 = icmp eq i32 %38, 128, !dbg !5033
  br i1 %cmp63, label %land.lhs.true64, label %if.else91, !dbg !5034

land.lhs.true64:                                  ; preds = %land.lhs.true62
  %39 = load i32, i32* %nt.addr, align 4, !dbg !5035
  %cmp65 = icmp eq i32 %39, 6, !dbg !5036
  br i1 %cmp65, label %if.then66, label %if.else91, !dbg !5037

if.then66:                                        ; preds = %land.lhs.true64
  %40 = load i8*, i8** %class_npb.addr, align 8, !dbg !5038
  store i8 65, i8* %40, align 1, !dbg !5040
  %real68 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp67, i32 0, i32 0, !dbg !5041
  store double 0x407F8AC6A8CB8B90, double* %real68, align 8, !dbg !5041
  %imag69 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp67, i32 0, i32 1, !dbg !5041
  store double 0x407FF67A05A82466, double* %imag69, align 8, !dbg !5041
  %arrayidx70 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 1, !dbg !5042
  %41 = bitcast %struct.dcomplex* %arrayidx70 to i8*, !dbg !5043
  %42 = bitcast %struct.dcomplex* %ref.tmp67 to i8*, !dbg !5043
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %41, i8* align 8 %42, i64 16, i1 false), !dbg !5043
  %real72 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp71, i32 0, i32 0, !dbg !5044
  store double 0x407F9F0F4941FB3E, double* %real72, align 8, !dbg !5044
  %imag73 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp71, i32 0, i32 1, !dbg !5044
  store double 0x407FDE18707A9D72, double* %imag73, align 8, !dbg !5044
  %arrayidx74 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 2, !dbg !5045
  %43 = bitcast %struct.dcomplex* %arrayidx74 to i8*, !dbg !5046
  %44 = bitcast %struct.dcomplex* %ref.tmp71 to i8*, !dbg !5046
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %43, i8* align 8 %44, i64 16, i1 false), !dbg !5046
  %real76 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp75, i32 0, i32 0, !dbg !5047
  store double 0x407FAF00C6D7110A, double* %real76, align 8, !dbg !5047
  %imag77 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp75, i32 0, i32 1, !dbg !5047
  store double 0x407FDD07CCB88353, double* %imag77, align 8, !dbg !5047
  %arrayidx78 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 3, !dbg !5048
  %45 = bitcast %struct.dcomplex* %arrayidx78 to i8*, !dbg !5049
  %46 = bitcast %struct.dcomplex* %ref.tmp75 to i8*, !dbg !5049
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %45, i8* align 8 %46, i64 16, i1 false), !dbg !5049
  %real80 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp79, i32 0, i32 0, !dbg !5050
  store double 0x407FBCA0EB3ECBEF, double* %real80, align 8, !dbg !5050
  %imag81 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp79, i32 0, i32 1, !dbg !5050
  store double 0x407FE2234776F4EF, double* %imag81, align 8, !dbg !5050
  %arrayidx82 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 4, !dbg !5051
  %47 = bitcast %struct.dcomplex* %arrayidx82 to i8*, !dbg !5052
  %48 = bitcast %struct.dcomplex* %ref.tmp79 to i8*, !dbg !5052
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %47, i8* align 8 %48, i64 16, i1 false), !dbg !5052
  %real84 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp83, i32 0, i32 0, !dbg !5053
  store double 0x407FC85F79D2C1E9, double* %real84, align 8, !dbg !5053
  %imag85 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp83, i32 0, i32 1, !dbg !5053
  store double 0x407FE7DD0AF2CEF4, double* %imag85, align 8, !dbg !5053
  %arrayidx86 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 5, !dbg !5054
  %49 = bitcast %struct.dcomplex* %arrayidx86 to i8*, !dbg !5055
  %50 = bitcast %struct.dcomplex* %ref.tmp83 to i8*, !dbg !5055
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %49, i8* align 8 %50, i64 16, i1 false), !dbg !5055
  %real88 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp87, i32 0, i32 0, !dbg !5056
  store double 0x407FD2611DBB8FA9, double* %real88, align 8, !dbg !5056
  %imag89 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp87, i32 0, i32 1, !dbg !5056
  store double 0x407FECAB25FE5602, double* %imag89, align 8, !dbg !5056
  %arrayidx90 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 6, !dbg !5057
  %51 = bitcast %struct.dcomplex* %arrayidx90 to i8*, !dbg !5058
  %52 = bitcast %struct.dcomplex* %ref.tmp87 to i8*, !dbg !5058
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %51, i8* align 8 %52, i64 16, i1 false), !dbg !5058
  br label %if.end490, !dbg !5059

if.else91:                                        ; preds = %land.lhs.true64, %land.lhs.true62, %land.lhs.true60, %if.else58
  %53 = load i32, i32* %d1.addr, align 4, !dbg !5060
  %cmp92 = icmp eq i32 %53, 512, !dbg !5062
  br i1 %cmp92, label %land.lhs.true93, label %if.else180, !dbg !5063

land.lhs.true93:                                  ; preds = %if.else91
  %54 = load i32, i32* %d2.addr, align 4, !dbg !5064
  %cmp94 = icmp eq i32 %54, 256, !dbg !5065
  br i1 %cmp94, label %land.lhs.true95, label %if.else180, !dbg !5066

land.lhs.true95:                                  ; preds = %land.lhs.true93
  %55 = load i32, i32* %d3.addr, align 4, !dbg !5067
  %cmp96 = icmp eq i32 %55, 256, !dbg !5068
  br i1 %cmp96, label %land.lhs.true97, label %if.else180, !dbg !5069

land.lhs.true97:                                  ; preds = %land.lhs.true95
  %56 = load i32, i32* %nt.addr, align 4, !dbg !5070
  %cmp98 = icmp eq i32 %56, 20, !dbg !5071
  br i1 %cmp98, label %if.then99, label %if.else180, !dbg !5072

if.then99:                                        ; preds = %land.lhs.true97
  %57 = load i8*, i8** %class_npb.addr, align 8, !dbg !5073
  store i8 66, i8* %57, align 1, !dbg !5075
  %real101 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp100, i32 0, i32 0, !dbg !5076
  store double 0x40802E1D67491D27, double* %real101, align 8, !dbg !5076
  %imag102 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp100, i32 0, i32 1, !dbg !5076
  store double 0x407FBC7C4BF0AFB0, double* %imag102, align 8, !dbg !5076
  %arrayidx103 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 1, !dbg !5077
  %58 = bitcast %struct.dcomplex* %arrayidx103 to i8*, !dbg !5078
  %59 = bitcast %struct.dcomplex* %ref.tmp100 to i8*, !dbg !5078
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %58, i8* align 8 %59, i64 16, i1 false), !dbg !5078
  %real105 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp104, i32 0, i32 0, !dbg !5079
  store double 0x40801B9DF5E01838, double* %real105, align 8, !dbg !5079
  %imag106 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp104, i32 0, i32 1, !dbg !5079
  store double 0x407FCD32F7994D45, double* %imag106, align 8, !dbg !5079
  %arrayidx107 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 2, !dbg !5080
  %60 = bitcast %struct.dcomplex* %arrayidx107 to i8*, !dbg !5081
  %61 = bitcast %struct.dcomplex* %ref.tmp104 to i8*, !dbg !5081
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %60, i8* align 8 %61, i64 16, i1 false), !dbg !5081
  %real109 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp108, i32 0, i32 0, !dbg !5082
  store double 0x408015209C2AC008, double* %real109, align 8, !dbg !5082
  %imag110 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp108, i32 0, i32 1, !dbg !5082
  store double 0x407FD9EF2BAE169A, double* %imag110, align 8, !dbg !5082
  %arrayidx111 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 3, !dbg !5083
  %62 = bitcast %struct.dcomplex* %arrayidx111 to i8*, !dbg !5084
  %63 = bitcast %struct.dcomplex* %ref.tmp108 to i8*, !dbg !5084
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %62, i8* align 8 %63, i64 16, i1 false), !dbg !5084
  %real113 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp112, i32 0, i32 0, !dbg !5085
  store double 0x408011E72B556FFE, double* %real113, align 8, !dbg !5085
  %imag114 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp112, i32 0, i32 1, !dbg !5085
  store double 0x407FE1A32DF83794, double* %imag114, align 8, !dbg !5085
  %arrayidx115 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 4, !dbg !5086
  %64 = bitcast %struct.dcomplex* %arrayidx115 to i8*, !dbg !5087
  %65 = bitcast %struct.dcomplex* %ref.tmp112 to i8*, !dbg !5087
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %64, i8* align 8 %65, i64 16, i1 false), !dbg !5087
  %real117 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp116, i32 0, i32 0, !dbg !5088
  store double 0x40800FB38AA32FE6, double* %real117, align 8, !dbg !5088
  %imag118 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp116, i32 0, i32 1, !dbg !5088
  store double 0x407FE65CD1D86E4E, double* %imag118, align 8, !dbg !5088
  %arrayidx119 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 5, !dbg !5089
  %66 = bitcast %struct.dcomplex* %arrayidx119 to i8*, !dbg !5090
  %67 = bitcast %struct.dcomplex* %ref.tmp116 to i8*, !dbg !5090
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %66, i8* align 8 %67, i64 16, i1 false), !dbg !5090
  %real121 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp120, i32 0, i32 0, !dbg !5091
  store double 0x40800DF0531A9C48, double* %real121, align 8, !dbg !5091
  %imag122 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp120, i32 0, i32 1, !dbg !5091
  store double 0x407FE9844F14C8E1, double* %imag122, align 8, !dbg !5091
  %arrayidx123 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 6, !dbg !5092
  %68 = bitcast %struct.dcomplex* %arrayidx123 to i8*, !dbg !5093
  %69 = bitcast %struct.dcomplex* %ref.tmp120 to i8*, !dbg !5093
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %68, i8* align 8 %69, i64 16, i1 false), !dbg !5093
  %real125 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp124, i32 0, i32 0, !dbg !5094
  store double 0x40800C700989200D, double* %real125, align 8, !dbg !5094
  %imag126 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp124, i32 0, i32 1, !dbg !5094
  store double 0x407FEBD8BF0DD370, double* %imag126, align 8, !dbg !5094
  %arrayidx127 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 7, !dbg !5095
  %70 = bitcast %struct.dcomplex* %arrayidx127 to i8*, !dbg !5096
  %71 = bitcast %struct.dcomplex* %ref.tmp124 to i8*, !dbg !5096
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %70, i8* align 8 %71, i64 16, i1 false), !dbg !5096
  %real129 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp128, i32 0, i32 0, !dbg !5097
  store double 0x40800B20F5210ADA, double* %real129, align 8, !dbg !5097
  %imag130 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp128, i32 0, i32 1, !dbg !5097
  store double 0x407FEDB8F6EE292B, double* %imag130, align 8, !dbg !5097
  %arrayidx131 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 8, !dbg !5098
  %72 = bitcast %struct.dcomplex* %arrayidx131 to i8*, !dbg !5099
  %73 = bitcast %struct.dcomplex* %ref.tmp128 to i8*, !dbg !5099
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %72, i8* align 8 %73, i64 16, i1 false), !dbg !5099
  %real133 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp132, i32 0, i32 0, !dbg !5100
  store double 0x408009FA001E667B, double* %real133, align 8, !dbg !5100
  %imag134 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp132, i32 0, i32 1, !dbg !5100
  store double 0x407FEF52DA70C18D, double* %imag134, align 8, !dbg !5100
  %arrayidx135 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 9, !dbg !5101
  %74 = bitcast %struct.dcomplex* %arrayidx135 to i8*, !dbg !5102
  %75 = bitcast %struct.dcomplex* %ref.tmp132 to i8*, !dbg !5102
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %74, i8* align 8 %75, i64 16, i1 false), !dbg !5102
  %real137 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp136, i32 0, i32 0, !dbg !5103
  store double 0x408008F54B8BB893, double* %real137, align 8, !dbg !5103
  %imag138 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp136, i32 0, i32 1, !dbg !5103
  store double 0x407FF0BC8A6C6119, double* %imag138, align 8, !dbg !5103
  %arrayidx139 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 10, !dbg !5104
  %76 = bitcast %struct.dcomplex* %arrayidx139 to i8*, !dbg !5105
  %77 = bitcast %struct.dcomplex* %ref.tmp136 to i8*, !dbg !5105
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %76, i8* align 8 %77, i64 16, i1 false), !dbg !5105
  %real141 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp140, i32 0, i32 0, !dbg !5106
  store double 0x4080080E66C1709C, double* %real141, align 8, !dbg !5106
  %imag142 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp140, i32 0, i32 1, !dbg !5106
  store double 0x407FF200FF33D23F, double* %imag142, align 8, !dbg !5106
  %arrayidx143 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 11, !dbg !5107
  %78 = bitcast %struct.dcomplex* %arrayidx143 to i8*, !dbg !5108
  %79 = bitcast %struct.dcomplex* %ref.tmp140 to i8*, !dbg !5108
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %78, i8* align 8 %79, i64 16, i1 false), !dbg !5108
  %real145 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp144, i32 0, i32 0, !dbg !5109
  store double 0x40800741A55F37AD, double* %real145, align 8, !dbg !5109
  %imag146 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp144, i32 0, i32 1, !dbg !5109
  store double 0x407FF3261FE7F7AD, double* %imag146, align 8, !dbg !5109
  %arrayidx147 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 12, !dbg !5110
  %80 = bitcast %struct.dcomplex* %arrayidx147 to i8*, !dbg !5111
  %81 = bitcast %struct.dcomplex* %ref.tmp144 to i8*, !dbg !5111
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %80, i8* align 8 %81, i64 16, i1 false), !dbg !5111
  %real149 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp148, i32 0, i32 0, !dbg !5112
  store double 0x4080068BDAC33674, double* %real149, align 8, !dbg !5112
  %imag150 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp148, i32 0, i32 1, !dbg !5112
  store double 0x407FF42F9BEB8DC0, double* %imag150, align 8, !dbg !5112
  %arrayidx151 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 13, !dbg !5113
  %82 = bitcast %struct.dcomplex* %arrayidx151 to i8*, !dbg !5114
  %83 = bitcast %struct.dcomplex* %ref.tmp148 to i8*, !dbg !5114
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %82, i8* align 8 %83, i64 16, i1 false), !dbg !5114
  %real153 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp152, i32 0, i32 0, !dbg !5115
  store double 0x408005EA3C919C43, double* %real153, align 8, !dbg !5115
  %imag154 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp152, i32 0, i32 1, !dbg !5115
  store double 0x407FF5203263B154, double* %imag154, align 8, !dbg !5115
  %arrayidx155 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 14, !dbg !5116
  %84 = bitcast %struct.dcomplex* %arrayidx155 to i8*, !dbg !5117
  %85 = bitcast %struct.dcomplex* %ref.tmp152 to i8*, !dbg !5117
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %84, i8* align 8 %85, i64 16, i1 false), !dbg !5117
  %real157 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp156, i32 0, i32 0, !dbg !5118
  store double 0x4080055A545A3920, double* %real157, align 8, !dbg !5118
  %imag158 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp156, i32 0, i32 1, !dbg !5118
  store double 0x407FF5FA3C741F6E, double* %imag158, align 8, !dbg !5118
  %arrayidx159 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 15, !dbg !5119
  %86 = bitcast %struct.dcomplex* %arrayidx159 to i8*, !dbg !5120
  %87 = bitcast %struct.dcomplex* %ref.tmp156 to i8*, !dbg !5120
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %86, i8* align 8 %87, i64 16, i1 false), !dbg !5120
  %real161 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp160, i32 0, i32 0, !dbg !5121
  store double 0x408004D9F6B6B8E1, double* %real161, align 8, !dbg !5121
  %imag162 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp160, i32 0, i32 1, !dbg !5121
  store double 0x407FF6BFE1A61501, double* %imag162, align 8, !dbg !5121
  %arrayidx163 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 16, !dbg !5122
  %88 = bitcast %struct.dcomplex* %arrayidx163 to i8*, !dbg !5123
  %89 = bitcast %struct.dcomplex* %ref.tmp160 to i8*, !dbg !5123
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %88, i8* align 8 %89, i64 16, i1 false), !dbg !5123
  %real165 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp164, i32 0, i32 0, !dbg !5124
  store double 0x408004673C213244, double* %real165, align 8, !dbg !5124
  %imag166 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp164, i32 0, i32 1, !dbg !5124
  store double 0x407FF77327A3F7B0, double* %imag166, align 8, !dbg !5124
  %arrayidx167 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 17, !dbg !5125
  %90 = bitcast %struct.dcomplex* %arrayidx167 to i8*, !dbg !5126
  %91 = bitcast %struct.dcomplex* %ref.tmp164 to i8*, !dbg !5126
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %90, i8* align 8 %91, i64 16, i1 false), !dbg !5126
  %real169 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp168, i32 0, i32 0, !dbg !5127
  store double 0x408004007A3FD0EA, double* %real169, align 8, !dbg !5127
  %imag170 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp168, i32 0, i32 1, !dbg !5127
  store double 0x407FF815F3F1C1DE, double* %imag170, align 8, !dbg !5127
  %arrayidx171 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 18, !dbg !5128
  %92 = bitcast %struct.dcomplex* %arrayidx171 to i8*, !dbg !5129
  %93 = bitcast %struct.dcomplex* %ref.tmp168 to i8*, !dbg !5129
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %92, i8* align 8 %93, i64 16, i1 false), !dbg !5129
  %real173 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp172, i32 0, i32 0, !dbg !5130
  store double 0x408003A43D5F793B, double* %real173, align 8, !dbg !5130
  %imag174 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp172, i32 0, i32 1, !dbg !5130
  store double 0x407FF8AA099402A0, double* %imag174, align 8, !dbg !5130
  %arrayidx175 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 19, !dbg !5131
  %94 = bitcast %struct.dcomplex* %arrayidx175 to i8*, !dbg !5132
  %95 = bitcast %struct.dcomplex* %ref.tmp172 to i8*, !dbg !5132
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %94, i8* align 8 %95, i64 16, i1 false), !dbg !5132
  %real177 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp176, i32 0, i32 0, !dbg !5133
  store double 0x40800351422D2EDF, double* %real177, align 8, !dbg !5133
  %imag178 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp176, i32 0, i32 1, !dbg !5133
  store double 0x407FF93106A352EE, double* %imag178, align 8, !dbg !5133
  %arrayidx179 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 20, !dbg !5134
  %96 = bitcast %struct.dcomplex* %arrayidx179 to i8*, !dbg !5135
  %97 = bitcast %struct.dcomplex* %ref.tmp176 to i8*, !dbg !5135
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %96, i8* align 8 %97, i64 16, i1 false), !dbg !5135
  br label %if.end489, !dbg !5136

if.else180:                                       ; preds = %land.lhs.true97, %land.lhs.true95, %land.lhs.true93, %if.else91
  %98 = load i32, i32* %d1.addr, align 4, !dbg !5137
  %cmp181 = icmp eq i32 %98, 512, !dbg !5139
  br i1 %cmp181, label %land.lhs.true182, label %if.else269, !dbg !5140

land.lhs.true182:                                 ; preds = %if.else180
  %99 = load i32, i32* %d2.addr, align 4, !dbg !5141
  %cmp183 = icmp eq i32 %99, 512, !dbg !5142
  br i1 %cmp183, label %land.lhs.true184, label %if.else269, !dbg !5143

land.lhs.true184:                                 ; preds = %land.lhs.true182
  %100 = load i32, i32* %d3.addr, align 4, !dbg !5144
  %cmp185 = icmp eq i32 %100, 512, !dbg !5145
  br i1 %cmp185, label %land.lhs.true186, label %if.else269, !dbg !5146

land.lhs.true186:                                 ; preds = %land.lhs.true184
  %101 = load i32, i32* %nt.addr, align 4, !dbg !5147
  %cmp187 = icmp eq i32 %101, 20, !dbg !5148
  br i1 %cmp187, label %if.then188, label %if.else269, !dbg !5149

if.then188:                                       ; preds = %land.lhs.true186
  %102 = load i8*, i8** %class_npb.addr, align 8, !dbg !5150
  store i8 67, i8* %102, align 1, !dbg !5152
  %real190 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp189, i32 0, i32 0, !dbg !5153
  store double 0x40803C101E899B03, double* %real190, align 8, !dbg !5153
  %imag191 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp189, i32 0, i32 1, !dbg !5153
  store double 0x408017373C01E593, double* %imag191, align 8, !dbg !5153
  %arrayidx192 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 1, !dbg !5154
  %103 = bitcast %struct.dcomplex* %arrayidx192 to i8*, !dbg !5155
  %104 = bitcast %struct.dcomplex* %ref.tmp189 to i8*, !dbg !5155
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %103, i8* align 8 %104, i64 16, i1 false), !dbg !5155
  %real194 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp193, i32 0, i32 0, !dbg !5156
  store double 0x40801C5675ED0B14, double* %real194, align 8, !dbg !5156
  %imag195 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp193, i32 0, i32 1, !dbg !5156
  store double 0x4080061004096FAD, double* %imag195, align 8, !dbg !5156
  %arrayidx196 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 2, !dbg !5157
  %105 = bitcast %struct.dcomplex* %arrayidx196 to i8*, !dbg !5158
  %106 = bitcast %struct.dcomplex* %ref.tmp193 to i8*, !dbg !5158
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %105, i8* align 8 %106, i64 16, i1 false), !dbg !5158
  %real198 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp197, i32 0, i32 0, !dbg !5159
  store double 0x408013BE0F176AC3, double* %real198, align 8, !dbg !5159
  %imag199 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp197, i32 0, i32 1, !dbg !5159
  store double 0x408001CD2DA9B691, double* %imag199, align 8, !dbg !5159
  %arrayidx200 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 3, !dbg !5160
  %107 = bitcast %struct.dcomplex* %arrayidx200 to i8*, !dbg !5161
  %108 = bitcast %struct.dcomplex* %ref.tmp197 to i8*, !dbg !5161
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %107, i8* align 8 %108, i64 16, i1 false), !dbg !5161
  %real202 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp201, i32 0, i32 0, !dbg !5162
  store double 0x4080101ED77ADAFA, double* %real202, align 8, !dbg !5162
  %imag203 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp201, i32 0, i32 1, !dbg !5162
  store double 0x408000DF4A8B7C66, double* %imag203, align 8, !dbg !5162
  %arrayidx204 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 4, !dbg !5163
  %109 = bitcast %struct.dcomplex* %arrayidx204 to i8*, !dbg !5164
  %110 = bitcast %struct.dcomplex* %ref.tmp201 to i8*, !dbg !5164
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %109, i8* align 8 %110, i64 16, i1 false), !dbg !5164
  %real206 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp205, i32 0, i32 0, !dbg !5165
  store double 0x40800E0A53D12FD5, double* %real206, align 8, !dbg !5165
  %imag207 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp205, i32 0, i32 1, !dbg !5165
  store double 0x408000EA3A1348C8, double* %imag207, align 8, !dbg !5165
  %arrayidx208 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 5, !dbg !5166
  %111 = bitcast %struct.dcomplex* %arrayidx208 to i8*, !dbg !5167
  %112 = bitcast %struct.dcomplex* %ref.tmp205 to i8*, !dbg !5167
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %111, i8* align 8 %112, i64 16, i1 false), !dbg !5167
  %real210 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp209, i32 0, i32 0, !dbg !5168
  store double 0x40800CA61ABB2192, double* %real210, align 8, !dbg !5168
  %imag211 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp209, i32 0, i32 1, !dbg !5168
  store double 0x408001328991F77F, double* %imag211, align 8, !dbg !5168
  %arrayidx212 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 6, !dbg !5169
  %113 = bitcast %struct.dcomplex* %arrayidx212 to i8*, !dbg !5170
  %114 = bitcast %struct.dcomplex* %ref.tmp209 to i8*, !dbg !5170
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %113, i8* align 8 %114, i64 16, i1 false), !dbg !5170
  %real214 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp213, i32 0, i32 0, !dbg !5171
  store double 0x40800BA7CD2DCE4D, double* %real214, align 8, !dbg !5171
  %imag215 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp213, i32 0, i32 1, !dbg !5171
  store double 0x4080017F2A30930B, double* %imag215, align 8, !dbg !5171
  %arrayidx216 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 7, !dbg !5172
  %115 = bitcast %struct.dcomplex* %arrayidx216 to i8*, !dbg !5173
  %116 = bitcast %struct.dcomplex* %ref.tmp213 to i8*, !dbg !5173
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %115, i8* align 8 %116, i64 16, i1 false), !dbg !5173
  %real218 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp217, i32 0, i32 0, !dbg !5174
  store double 0x40800AEBECB397D4, double* %real218, align 8, !dbg !5174
  %imag219 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp217, i32 0, i32 1, !dbg !5174
  store double 0x408001C12D7B83F2, double* %imag219, align 8, !dbg !5174
  %arrayidx220 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 8, !dbg !5175
  %117 = bitcast %struct.dcomplex* %arrayidx220 to i8*, !dbg !5176
  %118 = bitcast %struct.dcomplex* %ref.tmp217 to i8*, !dbg !5176
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %117, i8* align 8 %118, i64 16, i1 false), !dbg !5176
  %real222 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp221, i32 0, i32 0, !dbg !5177
  store double 0x40800A5D393668AE, double* %real222, align 8, !dbg !5177
  %imag223 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp221, i32 0, i32 1, !dbg !5177
  store double 0x408001F6BADA1C71, double* %imag223, align 8, !dbg !5177
  %arrayidx224 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 9, !dbg !5178
  %119 = bitcast %struct.dcomplex* %arrayidx224 to i8*, !dbg !5179
  %120 = bitcast %struct.dcomplex* %ref.tmp221 to i8*, !dbg !5179
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %119, i8* align 8 %120, i64 16, i1 false), !dbg !5179
  %real226 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp225, i32 0, i32 0, !dbg !5180
  store double 0x408009EDAA24021D, double* %real226, align 8, !dbg !5180
  %imag227 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp225, i32 0, i32 1, !dbg !5180
  store double 0x4080022183F3CA50, double* %imag227, align 8, !dbg !5180
  %arrayidx228 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 10, !dbg !5181
  %121 = bitcast %struct.dcomplex* %arrayidx228 to i8*, !dbg !5182
  %122 = bitcast %struct.dcomplex* %ref.tmp225 to i8*, !dbg !5182
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %121, i8* align 8 %122, i64 16, i1 false), !dbg !5182
  %real230 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp229, i32 0, i32 0, !dbg !5183
  store double 0x40800993B097C5AC, double* %real230, align 8, !dbg !5183
  %imag231 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp229, i32 0, i32 1, !dbg !5183
  store double 0x40800243C3A1DCB2, double* %imag231, align 8, !dbg !5183
  %arrayidx232 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 11, !dbg !5184
  %123 = bitcast %struct.dcomplex* %arrayidx232 to i8*, !dbg !5185
  %124 = bitcast %struct.dcomplex* %ref.tmp229 to i8*, !dbg !5185
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %123, i8* align 8 %124, i64 16, i1 false), !dbg !5185
  %real234 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp233, i32 0, i32 0, !dbg !5186
  store double 0x40800948BF026ADC, double* %real234, align 8, !dbg !5186
  %imag235 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp233, i32 0, i32 1, !dbg !5186
  store double 0x4080025F68FD8268, double* %imag235, align 8, !dbg !5186
  %arrayidx236 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 12, !dbg !5187
  %125 = bitcast %struct.dcomplex* %arrayidx236 to i8*, !dbg !5188
  %126 = bitcast %struct.dcomplex* %ref.tmp233 to i8*, !dbg !5188
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %125, i8* align 8 %126, i64 16, i1 false), !dbg !5188
  %real238 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp237, i32 0, i32 0, !dbg !5189
  store double 0x4080090857A518D9, double* %real238, align 8, !dbg !5189
  %imag239 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp237, i32 0, i32 1, !dbg !5189
  store double 0x40800275F32F50EA, double* %imag239, align 8, !dbg !5189
  %arrayidx240 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 13, !dbg !5190
  %127 = bitcast %struct.dcomplex* %arrayidx240 to i8*, !dbg !5191
  %128 = bitcast %struct.dcomplex* %ref.tmp237 to i8*, !dbg !5191
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %127, i8* align 8 %128, i64 16, i1 false), !dbg !5191
  %real242 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp241, i32 0, i32 0, !dbg !5192
  store double 0x408008CF67B5F6E6, double* %real242, align 8, !dbg !5192
  %imag243 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp241, i32 0, i32 1, !dbg !5192
  store double 0x408002887F1716B0, double* %imag243, align 8, !dbg !5192
  %arrayidx244 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 14, !dbg !5193
  %129 = bitcast %struct.dcomplex* %arrayidx244 to i8*, !dbg !5194
  %130 = bitcast %struct.dcomplex* %ref.tmp241 to i8*, !dbg !5194
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %129, i8* align 8 %130, i64 16, i1 false), !dbg !5194
  %real246 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp245, i32 0, i32 0, !dbg !5195
  store double 0x4080089BD580EA3A, double* %real246, align 8, !dbg !5195
  %imag247 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp245, i32 0, i32 1, !dbg !5195
  store double 0x40800297DE24048E, double* %imag247, align 8, !dbg !5195
  %arrayidx248 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 15, !dbg !5196
  %131 = bitcast %struct.dcomplex* %arrayidx248 to i8*, !dbg !5197
  %132 = bitcast %struct.dcomplex* %ref.tmp245 to i8*, !dbg !5197
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %131, i8* align 8 %132, i64 16, i1 false), !dbg !5197
  %real250 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp249, i32 0, i32 0, !dbg !5198
  store double 0x4080086C31EBD984, double* %real250, align 8, !dbg !5198
  %imag251 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp249, i32 0, i32 1, !dbg !5198
  store double 0x408002A4AAB9F9F8, double* %imag251, align 8, !dbg !5198
  %arrayidx252 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 16, !dbg !5199
  %133 = bitcast %struct.dcomplex* %arrayidx252 to i8*, !dbg !5200
  %134 = bitcast %struct.dcomplex* %ref.tmp249 to i8*, !dbg !5200
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %133, i8* align 8 %134, i64 16, i1 false), !dbg !5200
  %real254 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp253, i32 0, i32 0, !dbg !5201
  store double 0x4080083F8294129E, double* %real254, align 8, !dbg !5201
  %imag255 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp253, i32 0, i32 1, !dbg !5201
  store double 0x408002AF57DC0D71, double* %imag255, align 8, !dbg !5201
  %arrayidx256 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 17, !dbg !5202
  %135 = bitcast %struct.dcomplex* %arrayidx256 to i8*, !dbg !5203
  %136 = bitcast %struct.dcomplex* %ref.tmp253 to i8*, !dbg !5203
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %135, i8* align 8 %136, i64 16, i1 false), !dbg !5203
  %real258 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp257, i32 0, i32 0, !dbg !5204
  store double 0x408008151CE457D2, double* %real258, align 8, !dbg !5204
  %imag259 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp257, i32 0, i32 1, !dbg !5204
  store double 0x408002B83C8A44C9, double* %imag259, align 8, !dbg !5204
  %arrayidx260 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 18, !dbg !5205
  %137 = bitcast %struct.dcomplex* %arrayidx260 to i8*, !dbg !5206
  %138 = bitcast %struct.dcomplex* %ref.tmp257 to i8*, !dbg !5206
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %137, i8* align 8 %138, i64 16, i1 false), !dbg !5206
  %real262 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp261, i32 0, i32 0, !dbg !5207
  store double 0x408007EC8CCD48ED, double* %real262, align 8, !dbg !5207
  %imag263 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp261, i32 0, i32 1, !dbg !5207
  store double 0x408002BF9BCECA75, double* %imag263, align 8, !dbg !5207
  %arrayidx264 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 19, !dbg !5208
  %139 = bitcast %struct.dcomplex* %arrayidx264 to i8*, !dbg !5209
  %140 = bitcast %struct.dcomplex* %ref.tmp261 to i8*, !dbg !5209
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %139, i8* align 8 %140, i64 16, i1 false), !dbg !5209
  %real266 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp265, i32 0, i32 0, !dbg !5210
  store double 0x408007C58371022F, double* %real266, align 8, !dbg !5210
  %imag267 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp265, i32 0, i32 1, !dbg !5210
  store double 0x408002C5AA6407B6, double* %imag267, align 8, !dbg !5210
  %arrayidx268 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 20, !dbg !5211
  %141 = bitcast %struct.dcomplex* %arrayidx268 to i8*, !dbg !5212
  %142 = bitcast %struct.dcomplex* %ref.tmp265 to i8*, !dbg !5212
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %141, i8* align 8 %142, i64 16, i1 false), !dbg !5212
  br label %if.end488, !dbg !5213

if.else269:                                       ; preds = %land.lhs.true186, %land.lhs.true184, %land.lhs.true182, %if.else180
  %143 = load i32, i32* %d1.addr, align 4, !dbg !5214
  %cmp270 = icmp eq i32 %143, 2048, !dbg !5216
  br i1 %cmp270, label %land.lhs.true271, label %if.else378, !dbg !5217

land.lhs.true271:                                 ; preds = %if.else269
  %144 = load i32, i32* %d2.addr, align 4, !dbg !5218
  %cmp272 = icmp eq i32 %144, 1024, !dbg !5219
  br i1 %cmp272, label %land.lhs.true273, label %if.else378, !dbg !5220

land.lhs.true273:                                 ; preds = %land.lhs.true271
  %145 = load i32, i32* %d3.addr, align 4, !dbg !5221
  %cmp274 = icmp eq i32 %145, 1024, !dbg !5222
  br i1 %cmp274, label %land.lhs.true275, label %if.else378, !dbg !5223

land.lhs.true275:                                 ; preds = %land.lhs.true273
  %146 = load i32, i32* %nt.addr, align 4, !dbg !5224
  %cmp276 = icmp eq i32 %146, 25, !dbg !5225
  br i1 %cmp276, label %if.then277, label %if.else378, !dbg !5226

if.then277:                                       ; preds = %land.lhs.true275
  %147 = load i8*, i8** %class_npb.addr, align 8, !dbg !5227
  store i8 68, i8* %147, align 1, !dbg !5229
  %real279 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp278, i32 0, i32 0, !dbg !5230
  store double 0x408001C8B7A5243B, double* %real279, align 8, !dbg !5230
  %imag280 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp278, i32 0, i32 1, !dbg !5230
  store double 0x407FFDA78AA6499C, double* %imag280, align 8, !dbg !5230
  %arrayidx281 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 1, !dbg !5231
  %148 = bitcast %struct.dcomplex* %arrayidx281 to i8*, !dbg !5232
  %149 = bitcast %struct.dcomplex* %ref.tmp278 to i8*, !dbg !5232
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %148, i8* align 8 %149, i64 16, i1 false), !dbg !5232
  %real283 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp282, i32 0, i32 0, !dbg !5233
  store double 0x4080005F05B14D73, double* %real283, align 8, !dbg !5233
  %imag284 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp282, i32 0, i32 1, !dbg !5233
  store double 0x407FFB4C42805D51, double* %imag284, align 8, !dbg !5233
  %arrayidx285 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 2, !dbg !5234
  %150 = bitcast %struct.dcomplex* %arrayidx285 to i8*, !dbg !5235
  %151 = bitcast %struct.dcomplex* %ref.tmp282 to i8*, !dbg !5235
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %150, i8* align 8 %151, i64 16, i1 false), !dbg !5235
  %real287 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp286, i32 0, i32 0, !dbg !5236
  store double 0x407FFFC9049FE6AA, double* %real287, align 8, !dbg !5236
  %imag288 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp286, i32 0, i32 1, !dbg !5236
  store double 0x407FFB5AABC2C2DC, double* %imag288, align 8, !dbg !5236
  %arrayidx289 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 3, !dbg !5237
  %152 = bitcast %struct.dcomplex* %arrayidx289 to i8*, !dbg !5238
  %153 = bitcast %struct.dcomplex* %ref.tmp286 to i8*, !dbg !5238
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %152, i8* align 8 %153, i64 16, i1 false), !dbg !5238
  %real291 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp290, i32 0, i32 0, !dbg !5239
  store double 0x407FFF3AE6781D07, double* %real291, align 8, !dbg !5239
  %imag292 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp290, i32 0, i32 1, !dbg !5239
  store double 0x407FFBCC55AD30A5, double* %imag292, align 8, !dbg !5239
  %arrayidx293 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 4, !dbg !5240
  %154 = bitcast %struct.dcomplex* %arrayidx293 to i8*, !dbg !5241
  %155 = bitcast %struct.dcomplex* %ref.tmp290 to i8*, !dbg !5241
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %154, i8* align 8 %155, i64 16, i1 false), !dbg !5241
  %real295 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp294, i32 0, i32 0, !dbg !5242
  store double 0x407FFED49E586270, double* %real295, align 8, !dbg !5242
  %imag296 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp294, i32 0, i32 1, !dbg !5242
  store double 0x407FFC49DED1E229, double* %imag296, align 8, !dbg !5242
  %arrayidx297 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 5, !dbg !5243
  %156 = bitcast %struct.dcomplex* %arrayidx297 to i8*, !dbg !5244
  %157 = bitcast %struct.dcomplex* %ref.tmp294 to i8*, !dbg !5244
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %156, i8* align 8 %157, i64 16, i1 false), !dbg !5244
  %real299 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp298, i32 0, i32 0, !dbg !5245
  store double 0x407FFE88286F1600, double* %real299, align 8, !dbg !5245
  %imag300 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp298, i32 0, i32 1, !dbg !5245
  store double 0x407FFCBFA44E2DA9, double* %imag300, align 8, !dbg !5245
  %arrayidx301 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 6, !dbg !5246
  %158 = bitcast %struct.dcomplex* %arrayidx301 to i8*, !dbg !5247
  %159 = bitcast %struct.dcomplex* %ref.tmp298 to i8*, !dbg !5247
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %158, i8* align 8 %159, i64 16, i1 false), !dbg !5247
  %real303 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp302, i32 0, i32 0, !dbg !5248
  store double 0x407FFE4F62F012B7, double* %real303, align 8, !dbg !5248
  %imag304 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp302, i32 0, i32 1, !dbg !5248
  store double 0x407FFD2913502BF7, double* %imag304, align 8, !dbg !5248
  %arrayidx305 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 7, !dbg !5249
  %160 = bitcast %struct.dcomplex* %arrayidx305 to i8*, !dbg !5250
  %161 = bitcast %struct.dcomplex* %ref.tmp302 to i8*, !dbg !5250
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %160, i8* align 8 %161, i64 16, i1 false), !dbg !5250
  %real307 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp306, i32 0, i32 0, !dbg !5251
  store double 0x407FFE25D7467D87, double* %real307, align 8, !dbg !5251
  %imag308 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp306, i32 0, i32 1, !dbg !5251
  store double 0x407FFD85C991CC1E, double* %imag308, align 8, !dbg !5251
  %arrayidx309 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 8, !dbg !5252
  %162 = bitcast %struct.dcomplex* %arrayidx309 to i8*, !dbg !5253
  %163 = bitcast %struct.dcomplex* %ref.tmp306 to i8*, !dbg !5253
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %162, i8* align 8 %163, i64 16, i1 false), !dbg !5253
  %real311 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp310, i32 0, i32 0, !dbg !5254
  store double 0x407FFE07F5F9461B, double* %real311, align 8, !dbg !5254
  %imag312 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp310, i32 0, i32 1, !dbg !5254
  store double 0x407FFDD6ADE6AA2F, double* %imag312, align 8, !dbg !5254
  %arrayidx313 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 9, !dbg !5255
  %164 = bitcast %struct.dcomplex* %arrayidx313 to i8*, !dbg !5256
  %165 = bitcast %struct.dcomplex* %ref.tmp310 to i8*, !dbg !5256
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %164, i8* align 8 %165, i64 16, i1 false), !dbg !5256
  %real315 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp314, i32 0, i32 0, !dbg !5257
  store double 0x407FFDF2F9E3CE75, double* %real315, align 8, !dbg !5257
  %imag316 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp314, i32 0, i32 1, !dbg !5257
  store double 0x407FFE1D0052370F, double* %imag316, align 8, !dbg !5257
  %arrayidx317 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 10, !dbg !5258
  %166 = bitcast %struct.dcomplex* %arrayidx317 to i8*, !dbg !5259
  %167 = bitcast %struct.dcomplex* %ref.tmp314 to i8*, !dbg !5259
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %166, i8* align 8 %167, i64 16, i1 false), !dbg !5259
  %real319 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp318, i32 0, i32 0, !dbg !5260
  store double 0x407FFDE4CA360F49, double* %real319, align 8, !dbg !5260
  %imag320 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp318, i32 0, i32 1, !dbg !5260
  store double 0x407FFE5A05B5973E, double* %imag320, align 8, !dbg !5260
  %arrayidx321 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 11, !dbg !5261
  %168 = bitcast %struct.dcomplex* %arrayidx321 to i8*, !dbg !5262
  %169 = bitcast %struct.dcomplex* %ref.tmp318 to i8*, !dbg !5262
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %168, i8* align 8 %169, i64 16, i1 false), !dbg !5262
  %real323 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp322, i32 0, i32 0, !dbg !5263
  store double 0x407FFDDBD5F99711, double* %real323, align 8, !dbg !5263
  %imag324 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp322, i32 0, i32 1, !dbg !5263
  store double 0x407FFE8EEACAA874, double* %imag324, align 8, !dbg !5263
  %arrayidx325 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 12, !dbg !5264
  %170 = bitcast %struct.dcomplex* %arrayidx325 to i8*, !dbg !5265
  %171 = bitcast %struct.dcomplex* %ref.tmp322 to i8*, !dbg !5265
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %170, i8* align 8 %171, i64 16, i1 false), !dbg !5265
  %real327 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp326, i32 0, i32 0, !dbg !5266
  store double 0x407FFDD6F2033D21, double* %real327, align 8, !dbg !5266
  %imag328 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp326, i32 0, i32 1, !dbg !5266
  store double 0x407FFEBCBBFA2EBF, double* %imag328, align 8, !dbg !5266
  %arrayidx329 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 13, !dbg !5267
  %172 = bitcast %struct.dcomplex* %arrayidx329 to i8*, !dbg !5268
  %173 = bitcast %struct.dcomplex* %ref.tmp326 to i8*, !dbg !5268
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %172, i8* align 8 %173, i64 16, i1 false), !dbg !5268
  %real331 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp330, i32 0, i32 0, !dbg !5269
  store double 0x407FFDD53D74DC74, double* %real331, align 8, !dbg !5269
  %imag332 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp330, i32 0, i32 1, !dbg !5269
  store double 0x407FFEE46511649D, double* %imag332, align 8, !dbg !5269
  %arrayidx333 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 14, !dbg !5270
  %174 = bitcast %struct.dcomplex* %arrayidx333 to i8*, !dbg !5271
  %175 = bitcast %struct.dcomplex* %ref.tmp330 to i8*, !dbg !5271
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %174, i8* align 8 %175, i64 16, i1 false), !dbg !5271
  %real335 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp334, i32 0, i32 0, !dbg !5272
  store double 0x407FFDD60D2DB5D2, double* %real335, align 8, !dbg !5272
  %imag336 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp334, i32 0, i32 1, !dbg !5272
  store double 0x407FFF06B3C01AEA, double* %imag336, align 8, !dbg !5272
  %arrayidx337 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 15, !dbg !5273
  %176 = bitcast %struct.dcomplex* %arrayidx337 to i8*, !dbg !5274
  %177 = bitcast %struct.dcomplex* %ref.tmp334 to i8*, !dbg !5274
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %176, i8* align 8 %177, i64 16, i1 false), !dbg !5274
  %real339 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp338, i32 0, i32 0, !dbg !5275
  store double 0x407FFDD8DD056A7D, double* %real339, align 8, !dbg !5275
  %imag340 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp338, i32 0, i32 1, !dbg !5275
  store double 0x407FFF245ADF0BCE, double* %imag340, align 8, !dbg !5275
  %arrayidx341 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 16, !dbg !5276
  %178 = bitcast %struct.dcomplex* %arrayidx341 to i8*, !dbg !5277
  %179 = bitcast %struct.dcomplex* %ref.tmp338 to i8*, !dbg !5277
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %178, i8* align 8 %179, i64 16, i1 false), !dbg !5277
  %real343 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp342, i32 0, i32 0, !dbg !5278
  store double 0x407FFDDD45618FE6, double* %real343, align 8, !dbg !5278
  %imag344 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp342, i32 0, i32 1, !dbg !5278
  store double 0x407FFF3DF5BAB029, double* %imag344, align 8, !dbg !5278
  %arrayidx345 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 17, !dbg !5279
  %180 = bitcast %struct.dcomplex* %arrayidx345 to i8*, !dbg !5280
  %181 = bitcast %struct.dcomplex* %ref.tmp342 to i8*, !dbg !5280
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %180, i8* align 8 %181, i64 16, i1 false), !dbg !5280
  %real347 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp346, i32 0, i32 0, !dbg !5281
  store double 0x407FFDE2F3E650B3, double* %real347, align 8, !dbg !5281
  %imag348 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp346, i32 0, i32 1, !dbg !5281
  store double 0x407FFF540B1CF5A1, double* %imag348, align 8, !dbg !5281
  %arrayidx349 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 18, !dbg !5282
  %182 = bitcast %struct.dcomplex* %arrayidx349 to i8*, !dbg !5283
  %183 = bitcast %struct.dcomplex* %ref.tmp346 to i8*, !dbg !5283
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %182, i8* align 8 %183, i64 16, i1 false), !dbg !5283
  %real351 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp350, i32 0, i32 0, !dbg !5284
  store double 0x407FFDE9A64E1245, double* %real351, align 8, !dbg !5284
  %imag352 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp350, i32 0, i32 1, !dbg !5284
  store double 0x407FFF671002DAE5, double* %imag352, align 8, !dbg !5284
  %arrayidx353 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 19, !dbg !5285
  %184 = bitcast %struct.dcomplex* %arrayidx353 to i8*, !dbg !5286
  %185 = bitcast %struct.dcomplex* %ref.tmp350 to i8*, !dbg !5286
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %184, i8* align 8 %185, i64 16, i1 false), !dbg !5286
  %real355 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp354, i32 0, i32 0, !dbg !5287
  store double 0x407FFDF126BADF21, double* %real355, align 8, !dbg !5287
  %imag356 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp354, i32 0, i32 1, !dbg !5287
  store double 0x407FFF7769FD4D32, double* %imag356, align 8, !dbg !5287
  %arrayidx357 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 20, !dbg !5288
  %186 = bitcast %struct.dcomplex* %arrayidx357 to i8*, !dbg !5289
  %187 = bitcast %struct.dcomplex* %ref.tmp354 to i8*, !dbg !5289
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %186, i8* align 8 %187, i64 16, i1 false), !dbg !5289
  %real359 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp358, i32 0, i32 0, !dbg !5290
  store double 0x407FFDF94909BB13, double* %real359, align 8, !dbg !5290
  %imag360 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp358, i32 0, i32 1, !dbg !5290
  store double 0x407FFF85714411B2, double* %imag360, align 8, !dbg !5290
  %arrayidx361 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 21, !dbg !5291
  %188 = bitcast %struct.dcomplex* %arrayidx361 to i8*, !dbg !5292
  %189 = bitcast %struct.dcomplex* %ref.tmp358 to i8*, !dbg !5292
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %188, i8* align 8 %189, i64 16, i1 false), !dbg !5292
  %real363 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp362, i32 0, i32 0, !dbg !5293
  store double 0x407FFE01E8D7E962, double* %real363, align 8, !dbg !5293
  %imag364 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp362, i32 0, i32 1, !dbg !5293
  store double 0x407FFF9172826820, double* %imag364, align 8, !dbg !5293
  %arrayidx365 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 22, !dbg !5294
  %190 = bitcast %struct.dcomplex* %arrayidx365 to i8*, !dbg !5295
  %191 = bitcast %struct.dcomplex* %ref.tmp362 to i8*, !dbg !5295
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %190, i8* align 8 %191, i64 16, i1 false), !dbg !5295
  %real367 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp366, i32 0, i32 0, !dbg !5296
  store double 0x407FFE0AE8040E41, double* %real367, align 8, !dbg !5296
  %imag368 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp366, i32 0, i32 1, !dbg !5296
  store double 0x407FFF9BB06626E0, double* %imag368, align 8, !dbg !5296
  %arrayidx369 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 23, !dbg !5297
  %192 = bitcast %struct.dcomplex* %arrayidx369 to i8*, !dbg !5298
  %193 = bitcast %struct.dcomplex* %ref.tmp366 to i8*, !dbg !5298
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %192, i8* align 8 %193, i64 16, i1 false), !dbg !5298
  %real371 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp370, i32 0, i32 0, !dbg !5299
  store double 0x407FFE142D872C17, double* %real371, align 8, !dbg !5299
  %imag372 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp370, i32 0, i32 1, !dbg !5299
  store double 0x407FFFA464F89DCE, double* %imag372, align 8, !dbg !5299
  %arrayidx373 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 24, !dbg !5300
  %194 = bitcast %struct.dcomplex* %arrayidx373 to i8*, !dbg !5301
  %195 = bitcast %struct.dcomplex* %ref.tmp370 to i8*, !dbg !5301
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %194, i8* align 8 %195, i64 16, i1 false), !dbg !5301
  %real375 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp374, i32 0, i32 0, !dbg !5302
  store double 0x407FFE1DA48D386E, double* %real375, align 8, !dbg !5302
  %imag376 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp374, i32 0, i32 1, !dbg !5302
  store double 0x407FFFABC2C855DE, double* %imag376, align 8, !dbg !5302
  %arrayidx377 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 25, !dbg !5303
  %196 = bitcast %struct.dcomplex* %arrayidx377 to i8*, !dbg !5304
  %197 = bitcast %struct.dcomplex* %ref.tmp374 to i8*, !dbg !5304
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %196, i8* align 8 %197, i64 16, i1 false), !dbg !5304
  br label %if.end487, !dbg !5305

if.else378:                                       ; preds = %land.lhs.true275, %land.lhs.true273, %land.lhs.true271, %if.else269
  %198 = load i32, i32* %d1.addr, align 4, !dbg !5306
  %cmp379 = icmp eq i32 %198, 4096, !dbg !5308
  br i1 %cmp379, label %land.lhs.true380, label %if.end, !dbg !5309

land.lhs.true380:                                 ; preds = %if.else378
  %199 = load i32, i32* %d2.addr, align 4, !dbg !5310
  %cmp381 = icmp eq i32 %199, 2048, !dbg !5311
  br i1 %cmp381, label %land.lhs.true382, label %if.end, !dbg !5312

land.lhs.true382:                                 ; preds = %land.lhs.true380
  %200 = load i32, i32* %d3.addr, align 4, !dbg !5313
  %cmp383 = icmp eq i32 %200, 2048, !dbg !5314
  br i1 %cmp383, label %land.lhs.true384, label %if.end, !dbg !5315

land.lhs.true384:                                 ; preds = %land.lhs.true382
  %201 = load i32, i32* %nt.addr, align 4, !dbg !5316
  %cmp385 = icmp eq i32 %201, 25, !dbg !5317
  br i1 %cmp385, label %if.then386, label %if.end, !dbg !5318

if.then386:                                       ; preds = %land.lhs.true384
  %202 = load i8*, i8** %class_npb.addr, align 8, !dbg !5319
  store i8 69, i8* %202, align 1, !dbg !5321
  %real388 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp387, i32 0, i32 0, !dbg !5322
  store double 0x40800147E4E2E063, double* %real388, align 8, !dbg !5322
  %imag389 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp387, i32 0, i32 1, !dbg !5322
  store double 0x407FFBD566A0B5FD, double* %imag389, align 8, !dbg !5322
  %arrayidx390 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 1, !dbg !5323
  %203 = bitcast %struct.dcomplex* %arrayidx390 to i8*, !dbg !5324
  %204 = bitcast %struct.dcomplex* %ref.tmp387 to i8*, !dbg !5324
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %203, i8* align 8 %204, i64 16, i1 false), !dbg !5324
  %real392 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp391, i32 0, i32 0, !dbg !5325
  store double 0x408000B96D3A755A, double* %real392, align 8, !dbg !5325
  %imag393 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp391, i32 0, i32 1, !dbg !5325
  store double 0x407FFDC89676A99F, double* %imag393, align 8, !dbg !5325
  %arrayidx394 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 2, !dbg !5326
  %205 = bitcast %struct.dcomplex* %arrayidx394 to i8*, !dbg !5327
  %206 = bitcast %struct.dcomplex* %ref.tmp391 to i8*, !dbg !5327
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %205, i8* align 8 %206, i64 16, i1 false), !dbg !5327
  %real396 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp395, i32 0, i32 0, !dbg !5328
  store double 0x4080007FA32A25BE, double* %real396, align 8, !dbg !5328
  %imag397 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp395, i32 0, i32 1, !dbg !5328
  store double 0x407FFE84CB3A10F8, double* %imag397, align 8, !dbg !5328
  %arrayidx398 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 3, !dbg !5329
  %207 = bitcast %struct.dcomplex* %arrayidx398 to i8*, !dbg !5330
  %208 = bitcast %struct.dcomplex* %ref.tmp395 to i8*, !dbg !5330
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %207, i8* align 8 %208, i64 16, i1 false), !dbg !5330
  %real400 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp399, i32 0, i32 0, !dbg !5331
  store double 0x40800059C9C82B40, double* %real400, align 8, !dbg !5331
  %imag401 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp399, i32 0, i32 1, !dbg !5331
  store double 0x407FFEF414B87FD6, double* %imag401, align 8, !dbg !5331
  %arrayidx402 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 4, !dbg !5332
  %209 = bitcast %struct.dcomplex* %arrayidx402 to i8*, !dbg !5333
  %210 = bitcast %struct.dcomplex* %ref.tmp399 to i8*, !dbg !5333
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %209, i8* align 8 %210, i64 16, i1 false), !dbg !5333
  %real404 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp403, i32 0, i32 0, !dbg !5334
  store double 0x4080003FCCB7C9C8, double* %real404, align 8, !dbg !5334
  %imag405 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp403, i32 0, i32 1, !dbg !5334
  store double 0x407FFF483912F11E, double* %imag405, align 8, !dbg !5334
  %arrayidx406 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 5, !dbg !5335
  %211 = bitcast %struct.dcomplex* %arrayidx406 to i8*, !dbg !5336
  %212 = bitcast %struct.dcomplex* %ref.tmp403 to i8*, !dbg !5336
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %211, i8* align 8 %212, i64 16, i1 false), !dbg !5336
  %real408 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp407, i32 0, i32 0, !dbg !5337
  store double 0x4080002E4D90A084, double* %real408, align 8, !dbg !5337
  %imag409 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp407, i32 0, i32 1, !dbg !5337
  store double 0x407FFF8D62BCE558, double* %imag409, align 8, !dbg !5337
  %arrayidx410 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 6, !dbg !5338
  %213 = bitcast %struct.dcomplex* %arrayidx410 to i8*, !dbg !5339
  %214 = bitcast %struct.dcomplex* %ref.tmp407 to i8*, !dbg !5339
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %213, i8* align 8 %214, i64 16, i1 false), !dbg !5339
  %real412 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp411, i32 0, i32 0, !dbg !5340
  store double 0x40800022AC039D7C, double* %real412, align 8, !dbg !5340
  %imag413 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp411, i32 0, i32 1, !dbg !5340
  store double 0x407FFFC737C3F7CD, double* %imag413, align 8, !dbg !5340
  %arrayidx414 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 7, !dbg !5341
  %215 = bitcast %struct.dcomplex* %arrayidx414 to i8*, !dbg !5342
  %216 = bitcast %struct.dcomplex* %ref.tmp411 to i8*, !dbg !5342
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %215, i8* align 8 %216, i64 16, i1 false), !dbg !5342
  %real416 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp415, i32 0, i32 0, !dbg !5343
  store double 0x4080001ADFFA71B9, double* %real416, align 8, !dbg !5343
  %imag417 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp415, i32 0, i32 1, !dbg !5343
  store double 0x407FFFF78C336255, double* %imag417, align 8, !dbg !5343
  %arrayidx418 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 8, !dbg !5344
  %217 = bitcast %struct.dcomplex* %arrayidx418 to i8*, !dbg !5345
  %218 = bitcast %struct.dcomplex* %ref.tmp415 to i8*, !dbg !5345
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %217, i8* align 8 %218, i64 16, i1 false), !dbg !5345
  %real420 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp419, i32 0, i32 0, !dbg !5346
  store double 0x4080001574D0520C, double* %real420, align 8, !dbg !5346
  %imag421 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp419, i32 0, i32 1, !dbg !5346
  store double 0x4080000FE85C03E9, double* %imag421, align 8, !dbg !5346
  %arrayidx422 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 9, !dbg !5347
  %219 = bitcast %struct.dcomplex* %arrayidx422 to i8*, !dbg !5348
  %220 = bitcast %struct.dcomplex* %ref.tmp419 to i8*, !dbg !5348
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %219, i8* align 8 %220, i64 16, i1 false), !dbg !5348
  %real424 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp423, i32 0, i32 0, !dbg !5349
  store double 0x408000116F284244, double* %real424, align 8, !dbg !5349
  %imag425 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp423, i32 0, i32 1, !dbg !5349
  store double 0x40800020A7695837, double* %imag425, align 8, !dbg !5349
  %arrayidx426 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 10, !dbg !5350
  %221 = bitcast %struct.dcomplex* %arrayidx426 to i8*, !dbg !5351
  %222 = bitcast %struct.dcomplex* %ref.tmp423 to i8*, !dbg !5351
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %221, i8* align 8 %222, i64 16, i1 false), !dbg !5351
  %real428 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp427, i32 0, i32 0, !dbg !5352
  store double 0x4080000E2D56813F, double* %real428, align 8, !dbg !5352
  %imag429 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp427, i32 0, i32 1, !dbg !5352
  store double 0x4080002E951F7B34, double* %imag429, align 8, !dbg !5352
  %arrayidx430 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 11, !dbg !5353
  %223 = bitcast %struct.dcomplex* %arrayidx430 to i8*, !dbg !5354
  %224 = bitcast %struct.dcomplex* %ref.tmp427 to i8*, !dbg !5354
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %223, i8* align 8 %224, i64 16, i1 false), !dbg !5354
  %real432 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp431, i32 0, i32 0, !dbg !5355
  store double 0x4080000B4BE05864, double* %real432, align 8, !dbg !5355
  %imag433 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp431, i32 0, i32 1, !dbg !5355
  store double 0x4080003A2ED08404, double* %imag433, align 8, !dbg !5355
  %arrayidx434 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 12, !dbg !5356
  %225 = bitcast %struct.dcomplex* %arrayidx434 to i8*, !dbg !5357
  %226 = bitcast %struct.dcomplex* %ref.tmp431 to i8*, !dbg !5357
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %225, i8* align 8 %226, i64 16, i1 false), !dbg !5357
  %real436 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp435, i32 0, i32 0, !dbg !5358
  store double 0x408000089094AC2D, double* %real436, align 8, !dbg !5358
  %imag437 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp435, i32 0, i32 1, !dbg !5358
  store double 0x40800043DD87C2F3, double* %imag437, align 8, !dbg !5358
  %arrayidx438 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 13, !dbg !5359
  %227 = bitcast %struct.dcomplex* %arrayidx438 to i8*, !dbg !5360
  %228 = bitcast %struct.dcomplex* %ref.tmp435 to i8*, !dbg !5360
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %227, i8* align 8 %228, i64 16, i1 false), !dbg !5360
  %real440 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp439, i32 0, i32 0, !dbg !5361
  store double 0x40800005DBBF34DD, double* %real440, align 8, !dbg !5361
  %imag441 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp439, i32 0, i32 1, !dbg !5361
  store double 0x4080004BF7DEAC1A, double* %imag441, align 8, !dbg !5361
  %arrayidx442 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 14, !dbg !5362
  %229 = bitcast %struct.dcomplex* %arrayidx442 to i8*, !dbg !5363
  %230 = bitcast %struct.dcomplex* %ref.tmp439 to i8*, !dbg !5363
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %229, i8* align 8 %230, i64 16, i1 false), !dbg !5363
  %real444 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp443, i32 0, i32 0, !dbg !5364
  store double 0x408000031E1FCB83, double* %real444, align 8, !dbg !5364
  %imag445 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp443, i32 0, i32 1, !dbg !5364
  store double 0x40800052C48391C0, double* %imag445, align 8, !dbg !5364
  %arrayidx446 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 15, !dbg !5365
  %231 = bitcast %struct.dcomplex* %arrayidx446 to i8*, !dbg !5366
  %232 = bitcast %struct.dcomplex* %ref.tmp443 to i8*, !dbg !5366
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %231, i8* align 8 %232, i64 16, i1 false), !dbg !5366
  %real448 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp447, i32 0, i32 0, !dbg !5367
  store double 0x4080000052507A84, double* %real448, align 8, !dbg !5367
  %imag449 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp447, i32 0, i32 1, !dbg !5367
  store double 0x408000587CD9C3A1, double* %imag449, align 8, !dbg !5367
  %arrayidx450 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 16, !dbg !5368
  %233 = bitcast %struct.dcomplex* %arrayidx450 to i8*, !dbg !5369
  %234 = bitcast %struct.dcomplex* %ref.tmp447 to i8*, !dbg !5369
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %233, i8* align 8 %234, i64 16, i1 false), !dbg !5369
  %real452 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp451, i32 0, i32 0, !dbg !5370
  store double 0x407FFFFAF1111C29, double* %real452, align 8, !dbg !5370
  %imag453 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp451, i32 0, i32 1, !dbg !5370
  store double 0x4080005D4F648E97, double* %imag453, align 8, !dbg !5370
  %arrayidx454 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 17, !dbg !5371
  %235 = bitcast %struct.dcomplex* %arrayidx454 to i8*, !dbg !5372
  %236 = bitcast %struct.dcomplex* %ref.tmp451 to i8*, !dbg !5372
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %235, i8* align 8 %236, i64 16, i1 false), !dbg !5372
  %real456 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp455, i32 0, i32 0, !dbg !5373
  store double 0x407FFFF527E792B0, double* %real456, align 8, !dbg !5373
  %imag457 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp455, i32 0, i32 1, !dbg !5373
  store double 0x4080006161DD7A20, double* %imag457, align 8, !dbg !5373
  %arrayidx458 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 18, !dbg !5374
  %237 = bitcast %struct.dcomplex* %arrayidx458 to i8*, !dbg !5375
  %238 = bitcast %struct.dcomplex* %ref.tmp455 to i8*, !dbg !5375
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %237, i8* align 8 %238, i64 16, i1 false), !dbg !5375
  %real460 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp459, i32 0, i32 0, !dbg !5376
  store double 0x407FFFEF5224A658, double* %real460, align 8, !dbg !5376
  %imag461 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp459, i32 0, i32 1, !dbg !5376
  store double 0x40800064D2F0E0FB, double* %imag461, align 8, !dbg !5376
  %arrayidx462 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 19, !dbg !5377
  %239 = bitcast %struct.dcomplex* %arrayidx462 to i8*, !dbg !5378
  %240 = bitcast %struct.dcomplex* %ref.tmp459 to i8*, !dbg !5378
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %239, i8* align 8 %240, i64 16, i1 false), !dbg !5378
  %real464 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp463, i32 0, i32 0, !dbg !5379
  store double 0x407FFFE97985082F, double* %real464, align 8, !dbg !5379
  %imag465 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp463, i32 0, i32 1, !dbg !5379
  store double 0x40800067BBA76761, double* %imag465, align 8, !dbg !5379
  %arrayidx466 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 20, !dbg !5380
  %241 = bitcast %struct.dcomplex* %arrayidx466 to i8*, !dbg !5381
  %242 = bitcast %struct.dcomplex* %ref.tmp463 to i8*, !dbg !5381
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %241, i8* align 8 %242, i64 16, i1 false), !dbg !5381
  %real468 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp467, i32 0, i32 0, !dbg !5382
  store double 0x407FFFE3A76CE198, double* %real468, align 8, !dbg !5382
  %imag469 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp467, i32 0, i32 1, !dbg !5382
  store double 0x4080006A3087F53C, double* %imag469, align 8, !dbg !5382
  %arrayidx470 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 21, !dbg !5383
  %243 = bitcast %struct.dcomplex* %arrayidx470 to i8*, !dbg !5384
  %244 = bitcast %struct.dcomplex* %ref.tmp467 to i8*, !dbg !5384
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %243, i8* align 8 %244, i64 16, i1 false), !dbg !5384
  %real472 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp471, i32 0, i32 0, !dbg !5385
  store double 0x407FFFDDE458AC2A, double* %real472, align 8, !dbg !5385
  %imag473 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp471, i32 0, i32 1, !dbg !5385
  store double 0x4080006C427E60CB, double* %imag473, align 8, !dbg !5385
  %arrayidx474 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 22, !dbg !5386
  %245 = bitcast %struct.dcomplex* %arrayidx474 to i8*, !dbg !5387
  %246 = bitcast %struct.dcomplex* %ref.tmp471 to i8*, !dbg !5387
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %245, i8* align 8 %246, i64 16, i1 false), !dbg !5387
  %real476 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp475, i32 0, i32 0, !dbg !5388
  store double 0x407FFFD8379EC190, double* %real476, align 8, !dbg !5388
  %imag477 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp475, i32 0, i32 1, !dbg !5388
  store double 0x4080006DFF9235BC, double* %imag477, align 8, !dbg !5388
  %arrayidx478 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 23, !dbg !5389
  %247 = bitcast %struct.dcomplex* %arrayidx478 to i8*, !dbg !5390
  %248 = bitcast %struct.dcomplex* %ref.tmp475 to i8*, !dbg !5390
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %247, i8* align 8 %248, i64 16, i1 false), !dbg !5390
  %real480 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp479, i32 0, i32 0, !dbg !5391
  store double 0x407FFFD2A76113A7, double* %real480, align 8, !dbg !5391
  %imag481 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp479, i32 0, i32 1, !dbg !5391
  store double 0x4080006F7377203C, double* %imag481, align 8, !dbg !5391
  %arrayidx482 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 24, !dbg !5392
  %249 = bitcast %struct.dcomplex* %arrayidx482 to i8*, !dbg !5393
  %250 = bitcast %struct.dcomplex* %ref.tmp479 to i8*, !dbg !5393
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %249, i8* align 8 %250, i64 16, i1 false), !dbg !5393
  %real484 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp483, i32 0, i32 0, !dbg !5394
  store double 0x407FFFCD389947BC, double* %real484, align 8, !dbg !5394
  %imag485 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp483, i32 0, i32 1, !dbg !5394
  store double 0x40800070A7FF2BFD, double* %imag485, align 8, !dbg !5394
  %arrayidx486 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 25, !dbg !5395
  %251 = bitcast %struct.dcomplex* %arrayidx486 to i8*, !dbg !5396
  %252 = bitcast %struct.dcomplex* %ref.tmp483 to i8*, !dbg !5396
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %251, i8* align 8 %252, i64 16, i1 false), !dbg !5396
  br label %if.end, !dbg !5397

if.end:                                           ; preds = %if.then386, %land.lhs.true384, %land.lhs.true382, %land.lhs.true380, %if.else378
  br label %if.end487

if.end487:                                        ; preds = %if.end, %if.then277
  br label %if.end488

if.end488:                                        ; preds = %if.end487, %if.then188
  br label %if.end489

if.end489:                                        ; preds = %if.end488, %if.then99
  br label %if.end490

if.end490:                                        ; preds = %if.end489, %if.then66
  br label %if.end491

if.end491:                                        ; preds = %if.end490, %if.then33
  br label %if.end492

if.end492:                                        ; preds = %if.end491, %if.then
  %253 = load i8*, i8** %class_npb.addr, align 8, !dbg !5398
  %254 = load i8, i8* %253, align 1, !dbg !5400
  %conv = sext i8 %254 to i32, !dbg !5400
  %cmp493 = icmp ne i32 %conv, 85, !dbg !5401
  br i1 %cmp493, label %if.then494, label %if.end588, !dbg !5402

if.then494:                                       ; preds = %if.end492
  %255 = load i32*, i32** %verified.addr, align 8, !dbg !5403
  store i32 1, i32* %255, align 4, !dbg !5405
  store i32 1, i32* %i, align 4, !dbg !5406
  br label %for.cond, !dbg !5408

for.cond:                                         ; preds = %for.inc, %if.then494
  %256 = load i32, i32* %i, align 4, !dbg !5409
  %257 = load i32, i32* %nt.addr, align 4, !dbg !5411
  %cmp495 = icmp sle i32 %256, %257, !dbg !5412
  br i1 %cmp495, label %for.body, label %for.end, !dbg !5413

for.body:                                         ; preds = %for.cond
  %real496 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %agg.tmp, i32 0, i32 0, !dbg !5414
  %258 = load %struct.dcomplex*, %struct.dcomplex** @_ZL4sums, align 8, !dbg !5414
  %259 = load i32, i32* %i, align 4, !dbg !5414
  %idxprom = sext i32 %259 to i64, !dbg !5414
  %arrayidx497 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %258, i64 %idxprom, !dbg !5414
  %real498 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx497, i32 0, i32 0, !dbg !5414
  %260 = load double, double* %real498, align 8, !dbg !5414
  %261 = load i32, i32* %i, align 4, !dbg !5414
  %idxprom499 = sext i32 %261 to i64, !dbg !5414
  %arrayidx500 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 %idxprom499, !dbg !5414
  %real501 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx500, i32 0, i32 0, !dbg !5414
  %262 = load double, double* %real501, align 16, !dbg !5414
  %sub = fsub contract double %260, %262, !dbg !5414
  store double %sub, double* %real496, align 8, !dbg !5414
  %imag502 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %agg.tmp, i32 0, i32 1, !dbg !5414
  %263 = load %struct.dcomplex*, %struct.dcomplex** @_ZL4sums, align 8, !dbg !5414
  %264 = load i32, i32* %i, align 4, !dbg !5414
  %idxprom503 = sext i32 %264 to i64, !dbg !5414
  %arrayidx504 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %263, i64 %idxprom503, !dbg !5414
  %imag505 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx504, i32 0, i32 1, !dbg !5414
  %265 = load double, double* %imag505, align 8, !dbg !5414
  %266 = load i32, i32* %i, align 4, !dbg !5414
  %idxprom506 = sext i32 %266 to i64, !dbg !5414
  %arrayidx507 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 %idxprom506, !dbg !5414
  %imag508 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx507, i32 0, i32 1, !dbg !5414
  %267 = load double, double* %imag508, align 8, !dbg !5414
  %sub509 = fsub contract double %265, %267, !dbg !5414
  store double %sub509, double* %imag502, align 8, !dbg !5414
  %268 = load i32, i32* %i, align 4, !dbg !5414
  %idxprom511 = sext i32 %268 to i64, !dbg !5414
  %arrayidx512 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 %idxprom511, !dbg !5414
  %269 = bitcast %struct.dcomplex* %agg.tmp510 to i8*, !dbg !5414
  %270 = bitcast %struct.dcomplex* %arrayidx512 to i8*, !dbg !5414
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %269, i8* align 16 %270, i64 16, i1 false), !dbg !5414
  %271 = bitcast %struct.dcomplex* %agg.tmp to { double, double }*, !dbg !5414
  %272 = getelementptr inbounds { double, double }, { double, double }* %271, i32 0, i32 0, !dbg !5414
  %273 = load double, double* %272, align 8, !dbg !5414
  %274 = getelementptr inbounds { double, double }, { double, double }* %271, i32 0, i32 1, !dbg !5414
  %275 = load double, double* %274, align 8, !dbg !5414
  %276 = bitcast %struct.dcomplex* %agg.tmp510 to { double, double }*, !dbg !5414
  %277 = getelementptr inbounds { double, double }, { double, double }* %276, i32 0, i32 0, !dbg !5414
  %278 = load double, double* %277, align 8, !dbg !5414
  %279 = getelementptr inbounds { double, double }, { double, double }* %276, i32 0, i32 1, !dbg !5414
  %280 = load double, double* %279, align 8, !dbg !5414
  %call = call { double, double } @_ZL12dcomplex_div8dcomplexS_(double %273, double %275, double %278, double %280), !dbg !5414
  %281 = bitcast %struct.dcomplex* %coerce to { double, double }*, !dbg !5414
  %282 = getelementptr inbounds { double, double }, { double, double }* %281, i32 0, i32 0, !dbg !5414
  %283 = extractvalue { double, double } %call, 0, !dbg !5414
  store double %283, double* %282, align 8, !dbg !5414
  %284 = getelementptr inbounds { double, double }, { double, double }* %281, i32 0, i32 1, !dbg !5414
  %285 = extractvalue { double, double } %call, 1, !dbg !5414
  store double %285, double* %284, align 8, !dbg !5414
  %real513 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %coerce, i32 0, i32 0, !dbg !5414
  %286 = load double, double* %real513, align 8, !dbg !5414
  %real515 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %agg.tmp514, i32 0, i32 0, !dbg !5414
  %287 = load %struct.dcomplex*, %struct.dcomplex** @_ZL4sums, align 8, !dbg !5414
  %288 = load i32, i32* %i, align 4, !dbg !5414
  %idxprom516 = sext i32 %288 to i64, !dbg !5414
  %arrayidx517 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %287, i64 %idxprom516, !dbg !5414
  %real518 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx517, i32 0, i32 0, !dbg !5414
  %289 = load double, double* %real518, align 8, !dbg !5414
  %290 = load i32, i32* %i, align 4, !dbg !5414
  %idxprom519 = sext i32 %290 to i64, !dbg !5414
  %arrayidx520 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 %idxprom519, !dbg !5414
  %real521 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx520, i32 0, i32 0, !dbg !5414
  %291 = load double, double* %real521, align 16, !dbg !5414
  %sub522 = fsub contract double %289, %291, !dbg !5414
  store double %sub522, double* %real515, align 8, !dbg !5414
  %imag523 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %agg.tmp514, i32 0, i32 1, !dbg !5414
  %292 = load %struct.dcomplex*, %struct.dcomplex** @_ZL4sums, align 8, !dbg !5414
  %293 = load i32, i32* %i, align 4, !dbg !5414
  %idxprom524 = sext i32 %293 to i64, !dbg !5414
  %arrayidx525 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %292, i64 %idxprom524, !dbg !5414
  %imag526 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx525, i32 0, i32 1, !dbg !5414
  %294 = load double, double* %imag526, align 8, !dbg !5414
  %295 = load i32, i32* %i, align 4, !dbg !5414
  %idxprom527 = sext i32 %295 to i64, !dbg !5414
  %arrayidx528 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 %idxprom527, !dbg !5414
  %imag529 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx528, i32 0, i32 1, !dbg !5414
  %296 = load double, double* %imag529, align 8, !dbg !5414
  %sub530 = fsub contract double %294, %296, !dbg !5414
  store double %sub530, double* %imag523, align 8, !dbg !5414
  %297 = load i32, i32* %i, align 4, !dbg !5414
  %idxprom532 = sext i32 %297 to i64, !dbg !5414
  %arrayidx533 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 %idxprom532, !dbg !5414
  %298 = bitcast %struct.dcomplex* %agg.tmp531 to i8*, !dbg !5414
  %299 = bitcast %struct.dcomplex* %arrayidx533 to i8*, !dbg !5414
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %298, i8* align 16 %299, i64 16, i1 false), !dbg !5414
  %300 = bitcast %struct.dcomplex* %agg.tmp514 to { double, double }*, !dbg !5414
  %301 = getelementptr inbounds { double, double }, { double, double }* %300, i32 0, i32 0, !dbg !5414
  %302 = load double, double* %301, align 8, !dbg !5414
  %303 = getelementptr inbounds { double, double }, { double, double }* %300, i32 0, i32 1, !dbg !5414
  %304 = load double, double* %303, align 8, !dbg !5414
  %305 = bitcast %struct.dcomplex* %agg.tmp531 to { double, double }*, !dbg !5414
  %306 = getelementptr inbounds { double, double }, { double, double }* %305, i32 0, i32 0, !dbg !5414
  %307 = load double, double* %306, align 8, !dbg !5414
  %308 = getelementptr inbounds { double, double }, { double, double }* %305, i32 0, i32 1, !dbg !5414
  %309 = load double, double* %308, align 8, !dbg !5414
  %call534 = call { double, double } @_ZL12dcomplex_div8dcomplexS_(double %302, double %304, double %307, double %309), !dbg !5414
  %310 = bitcast %struct.dcomplex* %coerce535 to { double, double }*, !dbg !5414
  %311 = getelementptr inbounds { double, double }, { double, double }* %310, i32 0, i32 0, !dbg !5414
  %312 = extractvalue { double, double } %call534, 0, !dbg !5414
  store double %312, double* %311, align 8, !dbg !5414
  %313 = getelementptr inbounds { double, double }, { double, double }* %310, i32 0, i32 1, !dbg !5414
  %314 = extractvalue { double, double } %call534, 1, !dbg !5414
  store double %314, double* %313, align 8, !dbg !5414
  %real536 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %coerce535, i32 0, i32 0, !dbg !5414
  %315 = load double, double* %real536, align 8, !dbg !5414
  %mul = fmul contract double %286, %315, !dbg !5414
  %real538 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %agg.tmp537, i32 0, i32 0, !dbg !5414
  %316 = load %struct.dcomplex*, %struct.dcomplex** @_ZL4sums, align 8, !dbg !5414
  %317 = load i32, i32* %i, align 4, !dbg !5414
  %idxprom539 = sext i32 %317 to i64, !dbg !5414
  %arrayidx540 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %316, i64 %idxprom539, !dbg !5414
  %real541 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx540, i32 0, i32 0, !dbg !5414
  %318 = load double, double* %real541, align 8, !dbg !5414
  %319 = load i32, i32* %i, align 4, !dbg !5414
  %idxprom542 = sext i32 %319 to i64, !dbg !5414
  %arrayidx543 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 %idxprom542, !dbg !5414
  %real544 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx543, i32 0, i32 0, !dbg !5414
  %320 = load double, double* %real544, align 16, !dbg !5414
  %sub545 = fsub contract double %318, %320, !dbg !5414
  store double %sub545, double* %real538, align 8, !dbg !5414
  %imag546 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %agg.tmp537, i32 0, i32 1, !dbg !5414
  %321 = load %struct.dcomplex*, %struct.dcomplex** @_ZL4sums, align 8, !dbg !5414
  %322 = load i32, i32* %i, align 4, !dbg !5414
  %idxprom547 = sext i32 %322 to i64, !dbg !5414
  %arrayidx548 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %321, i64 %idxprom547, !dbg !5414
  %imag549 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx548, i32 0, i32 1, !dbg !5414
  %323 = load double, double* %imag549, align 8, !dbg !5414
  %324 = load i32, i32* %i, align 4, !dbg !5414
  %idxprom550 = sext i32 %324 to i64, !dbg !5414
  %arrayidx551 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 %idxprom550, !dbg !5414
  %imag552 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx551, i32 0, i32 1, !dbg !5414
  %325 = load double, double* %imag552, align 8, !dbg !5414
  %sub553 = fsub contract double %323, %325, !dbg !5414
  store double %sub553, double* %imag546, align 8, !dbg !5414
  %326 = load i32, i32* %i, align 4, !dbg !5414
  %idxprom555 = sext i32 %326 to i64, !dbg !5414
  %arrayidx556 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 %idxprom555, !dbg !5414
  %327 = bitcast %struct.dcomplex* %agg.tmp554 to i8*, !dbg !5414
  %328 = bitcast %struct.dcomplex* %arrayidx556 to i8*, !dbg !5414
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %327, i8* align 16 %328, i64 16, i1 false), !dbg !5414
  %329 = bitcast %struct.dcomplex* %agg.tmp537 to { double, double }*, !dbg !5414
  %330 = getelementptr inbounds { double, double }, { double, double }* %329, i32 0, i32 0, !dbg !5414
  %331 = load double, double* %330, align 8, !dbg !5414
  %332 = getelementptr inbounds { double, double }, { double, double }* %329, i32 0, i32 1, !dbg !5414
  %333 = load double, double* %332, align 8, !dbg !5414
  %334 = bitcast %struct.dcomplex* %agg.tmp554 to { double, double }*, !dbg !5414
  %335 = getelementptr inbounds { double, double }, { double, double }* %334, i32 0, i32 0, !dbg !5414
  %336 = load double, double* %335, align 8, !dbg !5414
  %337 = getelementptr inbounds { double, double }, { double, double }* %334, i32 0, i32 1, !dbg !5414
  %338 = load double, double* %337, align 8, !dbg !5414
  %call557 = call { double, double } @_ZL12dcomplex_div8dcomplexS_(double %331, double %333, double %336, double %338), !dbg !5414
  %339 = bitcast %struct.dcomplex* %coerce558 to { double, double }*, !dbg !5414
  %340 = getelementptr inbounds { double, double }, { double, double }* %339, i32 0, i32 0, !dbg !5414
  %341 = extractvalue { double, double } %call557, 0, !dbg !5414
  store double %341, double* %340, align 8, !dbg !5414
  %342 = getelementptr inbounds { double, double }, { double, double }* %339, i32 0, i32 1, !dbg !5414
  %343 = extractvalue { double, double } %call557, 1, !dbg !5414
  store double %343, double* %342, align 8, !dbg !5414
  %imag559 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %coerce558, i32 0, i32 1, !dbg !5414
  %344 = load double, double* %imag559, align 8, !dbg !5414
  %real561 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %agg.tmp560, i32 0, i32 0, !dbg !5414
  %345 = load %struct.dcomplex*, %struct.dcomplex** @_ZL4sums, align 8, !dbg !5414
  %346 = load i32, i32* %i, align 4, !dbg !5414
  %idxprom562 = sext i32 %346 to i64, !dbg !5414
  %arrayidx563 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %345, i64 %idxprom562, !dbg !5414
  %real564 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx563, i32 0, i32 0, !dbg !5414
  %347 = load double, double* %real564, align 8, !dbg !5414
  %348 = load i32, i32* %i, align 4, !dbg !5414
  %idxprom565 = sext i32 %348 to i64, !dbg !5414
  %arrayidx566 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 %idxprom565, !dbg !5414
  %real567 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx566, i32 0, i32 0, !dbg !5414
  %349 = load double, double* %real567, align 16, !dbg !5414
  %sub568 = fsub contract double %347, %349, !dbg !5414
  store double %sub568, double* %real561, align 8, !dbg !5414
  %imag569 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %agg.tmp560, i32 0, i32 1, !dbg !5414
  %350 = load %struct.dcomplex*, %struct.dcomplex** @_ZL4sums, align 8, !dbg !5414
  %351 = load i32, i32* %i, align 4, !dbg !5414
  %idxprom570 = sext i32 %351 to i64, !dbg !5414
  %arrayidx571 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %350, i64 %idxprom570, !dbg !5414
  %imag572 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx571, i32 0, i32 1, !dbg !5414
  %352 = load double, double* %imag572, align 8, !dbg !5414
  %353 = load i32, i32* %i, align 4, !dbg !5414
  %idxprom573 = sext i32 %353 to i64, !dbg !5414
  %arrayidx574 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 %idxprom573, !dbg !5414
  %imag575 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx574, i32 0, i32 1, !dbg !5414
  %354 = load double, double* %imag575, align 8, !dbg !5414
  %sub576 = fsub contract double %352, %354, !dbg !5414
  store double %sub576, double* %imag569, align 8, !dbg !5414
  %355 = load i32, i32* %i, align 4, !dbg !5414
  %idxprom578 = sext i32 %355 to i64, !dbg !5414
  %arrayidx579 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 %idxprom578, !dbg !5414
  %356 = bitcast %struct.dcomplex* %agg.tmp577 to i8*, !dbg !5414
  %357 = bitcast %struct.dcomplex* %arrayidx579 to i8*, !dbg !5414
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %356, i8* align 16 %357, i64 16, i1 false), !dbg !5414
  %358 = bitcast %struct.dcomplex* %agg.tmp560 to { double, double }*, !dbg !5414
  %359 = getelementptr inbounds { double, double }, { double, double }* %358, i32 0, i32 0, !dbg !5414
  %360 = load double, double* %359, align 8, !dbg !5414
  %361 = getelementptr inbounds { double, double }, { double, double }* %358, i32 0, i32 1, !dbg !5414
  %362 = load double, double* %361, align 8, !dbg !5414
  %363 = bitcast %struct.dcomplex* %agg.tmp577 to { double, double }*, !dbg !5414
  %364 = getelementptr inbounds { double, double }, { double, double }* %363, i32 0, i32 0, !dbg !5414
  %365 = load double, double* %364, align 8, !dbg !5414
  %366 = getelementptr inbounds { double, double }, { double, double }* %363, i32 0, i32 1, !dbg !5414
  %367 = load double, double* %366, align 8, !dbg !5414
  %call580 = call { double, double } @_ZL12dcomplex_div8dcomplexS_(double %360, double %362, double %365, double %367), !dbg !5414
  %368 = bitcast %struct.dcomplex* %coerce581 to { double, double }*, !dbg !5414
  %369 = getelementptr inbounds { double, double }, { double, double }* %368, i32 0, i32 0, !dbg !5414
  %370 = extractvalue { double, double } %call580, 0, !dbg !5414
  store double %370, double* %369, align 8, !dbg !5414
  %371 = getelementptr inbounds { double, double }, { double, double }* %368, i32 0, i32 1, !dbg !5414
  %372 = extractvalue { double, double } %call580, 1, !dbg !5414
  store double %372, double* %371, align 8, !dbg !5414
  %imag582 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %coerce581, i32 0, i32 1, !dbg !5414
  %373 = load double, double* %imag582, align 8, !dbg !5414
  %mul583 = fmul contract double %344, %373, !dbg !5414
  %add = fadd contract double %mul, %mul583, !dbg !5414
  %call584 = call double @sqrt(double %add) #11, !dbg !5414
  store double %call584, double* %err, align 8, !dbg !5416
  %374 = load double, double* %err, align 8, !dbg !5417
  %375 = load double, double* %epsilon, align 8, !dbg !5419
  %cmp585 = fcmp ole double %374, %375, !dbg !5420
  br i1 %cmp585, label %if.end587, label %if.then586, !dbg !5421

if.then586:                                       ; preds = %for.body
  %376 = load i32*, i32** %verified.addr, align 8, !dbg !5422
  store i32 0, i32* %376, align 4, !dbg !5424
  br label %for.end, !dbg !5425

if.end587:                                        ; preds = %for.body
  br label %for.inc, !dbg !5426

for.inc:                                          ; preds = %if.end587
  %377 = load i32, i32* %i, align 4, !dbg !5427
  %inc = add nsw i32 %377, 1, !dbg !5427
  store i32 %inc, i32* %i, align 4, !dbg !5427
  br label %for.cond, !dbg !5428, !llvm.loop !5429

for.end:                                          ; preds = %if.then586, %for.cond
  br label %if.end588, !dbg !5431

if.end588:                                        ; preds = %for.end, %if.end492
  %378 = load i8*, i8** %class_npb.addr, align 8, !dbg !5432
  %379 = load i8, i8* %378, align 1, !dbg !5434
  %conv589 = sext i8 %379 to i32, !dbg !5434
  %cmp590 = icmp ne i32 %conv589, 85, !dbg !5435
  br i1 %cmp590, label %if.then591, label %if.end597, !dbg !5436

if.then591:                                       ; preds = %if.end588
  %380 = load i32*, i32** %verified.addr, align 8, !dbg !5437
  %381 = load i32, i32* %380, align 4, !dbg !5440
  %tobool = icmp ne i32 %381, 0, !dbg !5440
  br i1 %tobool, label %if.then592, label %if.else594, !dbg !5441

if.then592:                                       ; preds = %if.then591
  %call593 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([33 x i8], [33 x i8]* @.str.73, i64 0, i64 0)), !dbg !5442
  br label %if.end596, !dbg !5444

if.else594:                                       ; preds = %if.then591
  %call595 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([29 x i8], [29 x i8]* @.str.74, i64 0, i64 0)), !dbg !5445
  br label %if.end596

if.end596:                                        ; preds = %if.else594, %if.then592
  br label %if.end597, !dbg !5447

if.end597:                                        ; preds = %if.end596, %if.end588
  %382 = load i8*, i8** %class_npb.addr, align 8, !dbg !5448
  %383 = load i8, i8* %382, align 1, !dbg !5449
  %conv598 = sext i8 %383 to i32, !dbg !5449
  %call599 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([17 x i8], [17 x i8]* @.str.75, i64 0, i64 0), i32 %conv598), !dbg !5450
  ret void, !dbg !5451
}

; Function Attrs: nounwind
declare dso_local double @log(double) #9

; Function Attrs: nounwind
declare dso_local i8* @strcpy(i8*, i8*) #9

; Function Attrs: nounwind
declare dso_local i8* @strcat(i8*, i8*) #9

; Function Attrs: noinline uwtable
define internal void @_ZL11release_gpuv() #7 !dbg !5452 {
entry:
  %0 = load %struct.dcomplex*, %struct.dcomplex** @sums_device, align 8, !dbg !5453
  %1 = bitcast %struct.dcomplex* %0 to i8*, !dbg !5453
  %call = call i32 @cudaFree(i8* %1), !dbg !5454
  %2 = load double*, double** @starts_device, align 8, !dbg !5455
  %3 = bitcast double* %2 to i8*, !dbg !5455
  %call1 = call i32 @cudaFree(i8* %3), !dbg !5456
  %4 = load double*, double** @twiddle_device, align 8, !dbg !5457
  %5 = bitcast double* %4 to i8*, !dbg !5457
  %call2 = call i32 @cudaFree(i8* %5), !dbg !5458
  %6 = load %struct.dcomplex*, %struct.dcomplex** @u_device, align 8, !dbg !5459
  %7 = bitcast %struct.dcomplex* %6 to i8*, !dbg !5459
  %call3 = call i32 @cudaFree(i8* %7), !dbg !5460
  %8 = load %struct.dcomplex*, %struct.dcomplex** @u0_device, align 8, !dbg !5461
  %9 = bitcast %struct.dcomplex* %8 to i8*, !dbg !5461
  %call4 = call i32 @cudaFree(i8* %9), !dbg !5462
  %10 = load %struct.dcomplex*, %struct.dcomplex** @u1_device, align 8, !dbg !5463
  %11 = bitcast %struct.dcomplex* %10 to i8*, !dbg !5463
  %call5 = call i32 @cudaFree(i8* %11), !dbg !5464
  %12 = load %struct.dcomplex*, %struct.dcomplex** @y0_device, align 8, !dbg !5465
  %13 = bitcast %struct.dcomplex* %12 to i8*, !dbg !5465
  %call6 = call i32 @cudaFree(i8* %13), !dbg !5466
  %14 = load %struct.dcomplex*, %struct.dcomplex** @y1_device, align 8, !dbg !5467
  %15 = bitcast %struct.dcomplex* %14 to i8*, !dbg !5467
  %call7 = call i32 @cudaFree(i8* %15), !dbg !5468
  ret void, !dbg !5469
}

; Function Attrs: nounwind
declare dso_local void @free(i8*) #9

declare dso_local i32 @cudaFree(i8*) #8

; Function Attrs: noinline nounwind uwtable
define internal { double, double } @_ZL12dcomplex_div8dcomplexS_(double %z1.coerce0, double %z1.coerce1, double %z2.coerce0, double %z2.coerce1) #6 !dbg !5470 {
entry:
  %retval = alloca %struct.dcomplex, align 8
  %z1 = alloca %struct.dcomplex, align 8
  %z2 = alloca %struct.dcomplex, align 8
  %a = alloca double, align 8
  %b = alloca double, align 8
  %c = alloca double, align 8
  %d = alloca double, align 8
  %divisor = alloca double, align 8
  %real4 = alloca double, align 8
  %imag8 = alloca double, align 8
  %0 = bitcast %struct.dcomplex* %z1 to { double, double }*
  %1 = getelementptr inbounds { double, double }, { double, double }* %0, i32 0, i32 0
  store double %z1.coerce0, double* %1, align 8
  %2 = getelementptr inbounds { double, double }, { double, double }* %0, i32 0, i32 1
  store double %z1.coerce1, double* %2, align 8
  %3 = bitcast %struct.dcomplex* %z2 to { double, double }*
  %4 = getelementptr inbounds { double, double }, { double, double }* %3, i32 0, i32 0
  store double %z2.coerce0, double* %4, align 8
  %5 = getelementptr inbounds { double, double }, { double, double }* %3, i32 0, i32 1
  store double %z2.coerce1, double* %5, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex* %z1, metadata !5473, metadata !DIExpression()), !dbg !5474
  call void @llvm.dbg.declare(metadata %struct.dcomplex* %z2, metadata !5475, metadata !DIExpression()), !dbg !5476
  call void @llvm.dbg.declare(metadata double* %a, metadata !5477, metadata !DIExpression()), !dbg !5478
  %real = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %z1, i32 0, i32 0, !dbg !5479
  %6 = load double, double* %real, align 8, !dbg !5479
  store double %6, double* %a, align 8, !dbg !5478
  call void @llvm.dbg.declare(metadata double* %b, metadata !5480, metadata !DIExpression()), !dbg !5481
  %imag = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %z1, i32 0, i32 1, !dbg !5482
  %7 = load double, double* %imag, align 8, !dbg !5482
  store double %7, double* %b, align 8, !dbg !5481
  call void @llvm.dbg.declare(metadata double* %c, metadata !5483, metadata !DIExpression()), !dbg !5484
  %real1 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %z2, i32 0, i32 0, !dbg !5485
  %8 = load double, double* %real1, align 8, !dbg !5485
  store double %8, double* %c, align 8, !dbg !5484
  call void @llvm.dbg.declare(metadata double* %d, metadata !5486, metadata !DIExpression()), !dbg !5487
  %imag2 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %z2, i32 0, i32 1, !dbg !5488
  %9 = load double, double* %imag2, align 8, !dbg !5488
  store double %9, double* %d, align 8, !dbg !5487
  call void @llvm.dbg.declare(metadata double* %divisor, metadata !5489, metadata !DIExpression()), !dbg !5490
  %10 = load double, double* %c, align 8, !dbg !5491
  %11 = load double, double* %c, align 8, !dbg !5492
  %mul = fmul contract double %10, %11, !dbg !5493
  %12 = load double, double* %d, align 8, !dbg !5494
  %13 = load double, double* %d, align 8, !dbg !5495
  %mul3 = fmul contract double %12, %13, !dbg !5496
  %add = fadd contract double %mul, %mul3, !dbg !5497
  store double %add, double* %divisor, align 8, !dbg !5490
  call void @llvm.dbg.declare(metadata double* %real4, metadata !5498, metadata !DIExpression()), !dbg !5499
  %14 = load double, double* %a, align 8, !dbg !5500
  %15 = load double, double* %c, align 8, !dbg !5501
  %mul5 = fmul contract double %14, %15, !dbg !5502
  %16 = load double, double* %b, align 8, !dbg !5503
  %17 = load double, double* %d, align 8, !dbg !5504
  %mul6 = fmul contract double %16, %17, !dbg !5505
  %add7 = fadd contract double %mul5, %mul6, !dbg !5506
  %18 = load double, double* %divisor, align 8, !dbg !5507
  %div = fdiv double %add7, %18, !dbg !5508
  store double %div, double* %real4, align 8, !dbg !5499
  call void @llvm.dbg.declare(metadata double* %imag8, metadata !5509, metadata !DIExpression()), !dbg !5510
  %19 = load double, double* %b, align 8, !dbg !5511
  %20 = load double, double* %c, align 8, !dbg !5512
  %mul9 = fmul contract double %19, %20, !dbg !5513
  %21 = load double, double* %a, align 8, !dbg !5514
  %22 = load double, double* %d, align 8, !dbg !5515
  %mul10 = fmul contract double %21, %22, !dbg !5516
  %sub = fsub contract double %mul9, %mul10, !dbg !5517
  %23 = load double, double* %divisor, align 8, !dbg !5518
  %div11 = fdiv double %sub, %23, !dbg !5519
  store double %div11, double* %imag8, align 8, !dbg !5510
  call void @llvm.dbg.declare(metadata %struct.dcomplex* %retval, metadata !5520, metadata !DIExpression()), !dbg !5521
  %real12 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %retval, i32 0, i32 0, !dbg !5522
  %24 = load double, double* %real4, align 8, !dbg !5523
  store double %24, double* %real12, align 8, !dbg !5522
  %imag13 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %retval, i32 0, i32 1, !dbg !5522
  %25 = load double, double* %imag8, align 8, !dbg !5524
  store double %25, double* %imag13, align 8, !dbg !5522
  %26 = bitcast %struct.dcomplex* %retval to { double, double }*, !dbg !5525
  %27 = load { double, double }, { double, double }* %26, align 8, !dbg !5525
  ret { double, double } %27, !dbg !5525
}

; Function Attrs: nounwind
declare dso_local double @sqrt(double) #9

; Function Attrs: noinline nounwind uwtable
define linkonce_odr dso_local void @_ZN4dim3C2Ejjj(%struct.dim3* %this, i32 %vx, i32 %vy, i32 %vz) unnamed_addr #6 comdat align 2 !dbg !5526 {
entry:
  %this.addr = alloca %struct.dim3*, align 8
  %vx.addr = alloca i32, align 4
  %vy.addr = alloca i32, align 4
  %vz.addr = alloca i32, align 4
  store %struct.dim3* %this, %struct.dim3** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dim3** %this.addr, metadata !5527, metadata !DIExpression()), !dbg !5529
  store i32 %vx, i32* %vx.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %vx.addr, metadata !5530, metadata !DIExpression()), !dbg !5531
  store i32 %vy, i32* %vy.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %vy.addr, metadata !5532, metadata !DIExpression()), !dbg !5533
  store i32 %vz, i32* %vz.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %vz.addr, metadata !5534, metadata !DIExpression()), !dbg !5535
  %this1 = load %struct.dim3*, %struct.dim3** %this.addr, align 8
  %x = getelementptr inbounds %struct.dim3, %struct.dim3* %this1, i32 0, i32 0, !dbg !5536
  %0 = load i32, i32* %vx.addr, align 4, !dbg !5537
  store i32 %0, i32* %x, align 4, !dbg !5536
  %y = getelementptr inbounds %struct.dim3, %struct.dim3* %this1, i32 0, i32 1, !dbg !5538
  %1 = load i32, i32* %vy.addr, align 4, !dbg !5539
  store i32 %1, i32* %y, align 4, !dbg !5538
  %z = getelementptr inbounds %struct.dim3, %struct.dim3* %this1, i32 0, i32 2, !dbg !5540
  %2 = load i32, i32* %vz.addr, align 4, !dbg !5541
  store i32 %2, i32* %z, align 4, !dbg !5540
  ret void, !dbg !5542
}

declare dso_local i32 @cudaConfigureCall(i64, i32, i64, i32, i64, %struct.CUstream_st*) #8

; Function Attrs: noinline uwtable
define dso_local void @ft.ll_CudaFE__Z19checksum_gpu_kerneliP8dcomplexS0_(i32 %iteration, %struct.dcomplex* %u1, %struct.dcomplex* %sums) #7 !dbg !5543 {
entry:
  %iteration.addr = alloca i32, align 4
  %u1.addr = alloca %struct.dcomplex*, align 8
  %sums.addr = alloca %struct.dcomplex*, align 8
  store i32 %iteration, i32* %iteration.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %iteration.addr, metadata !5544, metadata !DIExpression()), !dbg !5545
  store %struct.dcomplex* %u1, %struct.dcomplex** %u1.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %u1.addr, metadata !5546, metadata !DIExpression()), !dbg !5547
  store %struct.dcomplex* %sums, %struct.dcomplex** %sums.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %sums.addr, metadata !5548, metadata !DIExpression()), !dbg !5549
  %0 = bitcast i32* %iteration.addr to i8*, !dbg !5550
  %1 = call i32 @cudaSetupArgument(i8* %0, i64 4, i64 0), !dbg !5550
  %2 = icmp eq i32 %1, 0, !dbg !5550
  br i1 %2, label %setup.next, label %setup.end, !dbg !5550

setup.next:                                       ; preds = %entry
  %3 = bitcast %struct.dcomplex** %u1.addr to i8*, !dbg !5550
  %4 = call i32 @cudaSetupArgument(i8* %3, i64 8, i64 8), !dbg !5550
  %5 = icmp eq i32 %4, 0, !dbg !5550
  br i1 %5, label %setup.next1, label %setup.end, !dbg !5550

setup.next1:                                      ; preds = %setup.next
  %6 = bitcast %struct.dcomplex** %sums.addr to i8*, !dbg !5550
  %7 = call i32 @cudaSetupArgument(i8* %6, i64 8, i64 16), !dbg !5550
  %8 = icmp eq i32 %7, 0, !dbg !5550
  br i1 %8, label %setup.next2, label %setup.end, !dbg !5550

setup.next2:                                      ; preds = %setup.next1
  %9 = call i32 @cudaLaunch(i8* bitcast (void (i32, %struct.dcomplex*, %struct.dcomplex*)* @ft.ll_CudaFE__Z19checksum_gpu_kerneliP8dcomplexS0_ to i8*)), !dbg !5550
  br label %setup.end, !dbg !5550

setup.end:                                        ; preds = %setup.next2, %setup.next1, %setup.next, %entry
  ret void, !dbg !5551
}

declare dso_local i32 @cudaSetupArgument(i8*, i64, i64)

declare dso_local i32 @cudaLaunch(i8*)

; Function Attrs: noinline uwtable
define dso_local void @ft.ll_CudaFE__Z17evolve_gpu_kernelP8dcomplexS0_Pd(%struct.dcomplex* %u0, %struct.dcomplex* %u1, double* %twiddle) #7 !dbg !5552 {
entry:
  %u0.addr = alloca %struct.dcomplex*, align 8
  %u1.addr = alloca %struct.dcomplex*, align 8
  %twiddle.addr = alloca double*, align 8
  store %struct.dcomplex* %u0, %struct.dcomplex** %u0.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %u0.addr, metadata !5553, metadata !DIExpression()), !dbg !5554
  store %struct.dcomplex* %u1, %struct.dcomplex** %u1.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %u1.addr, metadata !5555, metadata !DIExpression()), !dbg !5556
  store double* %twiddle, double** %twiddle.addr, align 8
  call void @llvm.dbg.declare(metadata double** %twiddle.addr, metadata !5557, metadata !DIExpression()), !dbg !5558
  %0 = bitcast %struct.dcomplex** %u0.addr to i8*, !dbg !5559
  %1 = call i32 @cudaSetupArgument(i8* %0, i64 8, i64 0), !dbg !5559
  %2 = icmp eq i32 %1, 0, !dbg !5559
  br i1 %2, label %setup.next, label %setup.end, !dbg !5559

setup.next:                                       ; preds = %entry
  %3 = bitcast %struct.dcomplex** %u1.addr to i8*, !dbg !5559
  %4 = call i32 @cudaSetupArgument(i8* %3, i64 8, i64 8), !dbg !5559
  %5 = icmp eq i32 %4, 0, !dbg !5559
  br i1 %5, label %setup.next1, label %setup.end, !dbg !5559

setup.next1:                                      ; preds = %setup.next
  %6 = bitcast double** %twiddle.addr to i8*, !dbg !5559
  %7 = call i32 @cudaSetupArgument(i8* %6, i64 8, i64 16), !dbg !5559
  %8 = icmp eq i32 %7, 0, !dbg !5559
  br i1 %8, label %setup.next2, label %setup.end, !dbg !5559

setup.next2:                                      ; preds = %setup.next1
  %9 = call i32 @cudaLaunch(i8* bitcast (void (%struct.dcomplex*, %struct.dcomplex*, double*)* @ft.ll_CudaFE__Z17evolve_gpu_kernelP8dcomplexS0_Pd to i8*)), !dbg !5559
  br label %setup.end, !dbg !5559

setup.end:                                        ; preds = %setup.next2, %setup.next1, %setup.next, %entry
  ret void, !dbg !5560
}

; Function Attrs: noinline uwtable
define internal void @_ZL10cffts1_gpuiP8dcomplexS0_S0_S0_S0_(i32 %is, %struct.dcomplex* %u, %struct.dcomplex* %x_in, %struct.dcomplex* %x_out, %struct.dcomplex* %y0, %struct.dcomplex* %y1) #7 !dbg !5561 {
entry:
  %is.addr = alloca i32, align 4
  %u.addr = alloca %struct.dcomplex*, align 8
  %x_in.addr = alloca %struct.dcomplex*, align 8
  %x_out.addr = alloca %struct.dcomplex*, align 8
  %y0.addr = alloca %struct.dcomplex*, align 8
  %y1.addr = alloca %struct.dcomplex*, align 8
  %agg.tmp = alloca %struct.dim3, align 4
  %agg.tmp1 = alloca %struct.dim3, align 4
  %agg.tmp.coerce = alloca { i64, i32 }, align 4
  %agg.tmp1.coerce = alloca { i64, i32 }, align 4
  %agg.tmp3 = alloca %struct.dim3, align 4
  %agg.tmp4 = alloca %struct.dim3, align 4
  %agg.tmp3.coerce = alloca { i64, i32 }, align 4
  %agg.tmp4.coerce = alloca { i64, i32 }, align 4
  %agg.tmp10 = alloca %struct.dim3, align 4
  %agg.tmp11 = alloca %struct.dim3, align 4
  %agg.tmp10.coerce = alloca { i64, i32 }, align 4
  %agg.tmp11.coerce = alloca { i64, i32 }, align 4
  store i32 %is, i32* %is.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %is.addr, metadata !5564, metadata !DIExpression()), !dbg !5565
  store %struct.dcomplex* %u, %struct.dcomplex** %u.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %u.addr, metadata !5566, metadata !DIExpression()), !dbg !5567
  store %struct.dcomplex* %x_in, %struct.dcomplex** %x_in.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %x_in.addr, metadata !5568, metadata !DIExpression()), !dbg !5569
  store %struct.dcomplex* %x_out, %struct.dcomplex** %x_out.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %x_out.addr, metadata !5570, metadata !DIExpression()), !dbg !5571
  store %struct.dcomplex* %y0, %struct.dcomplex** %y0.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %y0.addr, metadata !5572, metadata !DIExpression()), !dbg !5573
  store %struct.dcomplex* %y1, %struct.dcomplex** %y1.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %y1.addr, metadata !5574, metadata !DIExpression()), !dbg !5575
  %0 = load i32, i32* @blocks_per_grid_on_fftx_1, align 4, !dbg !5576
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp, i32 %0, i32 1, i32 1), !dbg !5576
  %1 = load i32, i32* @threads_per_block_on_fftx_1, align 4, !dbg !5577
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp1, i32 %1, i32 1, i32 1), !dbg !5577
  %2 = bitcast { i64, i32 }* %agg.tmp.coerce to i8*, !dbg !5578
  %3 = bitcast %struct.dim3* %agg.tmp to i8*, !dbg !5578
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %2, i8* align 4 %3, i64 12, i1 false), !dbg !5578
  %4 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 0, !dbg !5578
  %5 = load i64, i64* %4, align 4, !dbg !5578
  %6 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 1, !dbg !5578
  %7 = load i32, i32* %6, align 4, !dbg !5578
  %8 = bitcast { i64, i32 }* %agg.tmp1.coerce to i8*, !dbg !5578
  %9 = bitcast %struct.dim3* %agg.tmp1 to i8*, !dbg !5578
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %8, i8* align 4 %9, i64 12, i1 false), !dbg !5578
  %10 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp1.coerce, i32 0, i32 0, !dbg !5578
  %11 = load i64, i64* %10, align 4, !dbg !5578
  %12 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp1.coerce, i32 0, i32 1, !dbg !5578
  %13 = load i32, i32* %12, align 4, !dbg !5578
  %call = call i32 @cudaConfigureCall(i64 %5, i32 %7, i64 %11, i32 %13, i64 0, %struct.CUstream_st* null), !dbg !5578
  %tobool = icmp ne i32 %call, 0, !dbg !5578
  br i1 %tobool, label %kcall.end, label %kcall.configok, !dbg !5579

kcall.configok:                                   ; preds = %entry
  %14 = load %struct.dcomplex*, %struct.dcomplex** %x_in.addr, align 8, !dbg !5580
  %15 = load %struct.dcomplex*, %struct.dcomplex** %y0.addr, align 8, !dbg !5581
  call void @ft.ll_CudaFE__Z19cffts1_gpu_kernel_1P8dcomplexS0_(%struct.dcomplex* %14, %struct.dcomplex* %15), !dbg !5579
  br label %kcall.end, !dbg !5579

kcall.end:                                        ; preds = %kcall.configok, %entry
  %call2 = call i32 @cudaDeviceSynchronize(), !dbg !5582
  %16 = load i32, i32* @blocks_per_grid_on_fftx_2, align 4, !dbg !5583
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp3, i32 %16, i32 1, i32 1), !dbg !5583
  %17 = load i32, i32* @threads_per_block_on_fftx_2, align 4, !dbg !5584
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp4, i32 %17, i32 1, i32 1), !dbg !5584
  %18 = bitcast { i64, i32 }* %agg.tmp3.coerce to i8*, !dbg !5585
  %19 = bitcast %struct.dim3* %agg.tmp3 to i8*, !dbg !5585
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %18, i8* align 4 %19, i64 12, i1 false), !dbg !5585
  %20 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp3.coerce, i32 0, i32 0, !dbg !5585
  %21 = load i64, i64* %20, align 4, !dbg !5585
  %22 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp3.coerce, i32 0, i32 1, !dbg !5585
  %23 = load i32, i32* %22, align 4, !dbg !5585
  %24 = bitcast { i64, i32 }* %agg.tmp4.coerce to i8*, !dbg !5585
  %25 = bitcast %struct.dim3* %agg.tmp4 to i8*, !dbg !5585
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %24, i8* align 4 %25, i64 12, i1 false), !dbg !5585
  %26 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp4.coerce, i32 0, i32 0, !dbg !5585
  %27 = load i64, i64* %26, align 4, !dbg !5585
  %28 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp4.coerce, i32 0, i32 1, !dbg !5585
  %29 = load i32, i32* %28, align 4, !dbg !5585
  %call5 = call i32 @cudaConfigureCall(i64 %21, i32 %23, i64 %27, i32 %29, i64 0, %struct.CUstream_st* null), !dbg !5585
  %tobool6 = icmp ne i32 %call5, 0, !dbg !5585
  br i1 %tobool6, label %kcall.end8, label %kcall.configok7, !dbg !5586

kcall.configok7:                                  ; preds = %kcall.end
  %30 = load i32, i32* %is.addr, align 4, !dbg !5587
  %31 = load %struct.dcomplex*, %struct.dcomplex** %y0.addr, align 8, !dbg !5588
  %32 = load %struct.dcomplex*, %struct.dcomplex** %y1.addr, align 8, !dbg !5589
  %33 = load %struct.dcomplex*, %struct.dcomplex** %u.addr, align 8, !dbg !5590
  call void @ft.ll_CudaFE__Z19cffts1_gpu_kernel_2iP8dcomplexS0_S0_(i32 %30, %struct.dcomplex* %31, %struct.dcomplex* %32, %struct.dcomplex* %33), !dbg !5586
  br label %kcall.end8, !dbg !5586

kcall.end8:                                       ; preds = %kcall.configok7, %kcall.end
  %call9 = call i32 @cudaDeviceSynchronize(), !dbg !5591
  %34 = load i32, i32* @blocks_per_grid_on_fftx_3, align 4, !dbg !5592
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp10, i32 %34, i32 1, i32 1), !dbg !5592
  %35 = load i32, i32* @threads_per_block_on_fftx_3, align 4, !dbg !5593
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp11, i32 %35, i32 1, i32 1), !dbg !5593
  %36 = bitcast { i64, i32 }* %agg.tmp10.coerce to i8*, !dbg !5594
  %37 = bitcast %struct.dim3* %agg.tmp10 to i8*, !dbg !5594
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %36, i8* align 4 %37, i64 12, i1 false), !dbg !5594
  %38 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp10.coerce, i32 0, i32 0, !dbg !5594
  %39 = load i64, i64* %38, align 4, !dbg !5594
  %40 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp10.coerce, i32 0, i32 1, !dbg !5594
  %41 = load i32, i32* %40, align 4, !dbg !5594
  %42 = bitcast { i64, i32 }* %agg.tmp11.coerce to i8*, !dbg !5594
  %43 = bitcast %struct.dim3* %agg.tmp11 to i8*, !dbg !5594
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %42, i8* align 4 %43, i64 12, i1 false), !dbg !5594
  %44 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp11.coerce, i32 0, i32 0, !dbg !5594
  %45 = load i64, i64* %44, align 4, !dbg !5594
  %46 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp11.coerce, i32 0, i32 1, !dbg !5594
  %47 = load i32, i32* %46, align 4, !dbg !5594
  %call12 = call i32 @cudaConfigureCall(i64 %39, i32 %41, i64 %45, i32 %47, i64 0, %struct.CUstream_st* null), !dbg !5594
  %tobool13 = icmp ne i32 %call12, 0, !dbg !5594
  br i1 %tobool13, label %kcall.end15, label %kcall.configok14, !dbg !5595

kcall.configok14:                                 ; preds = %kcall.end8
  %48 = load %struct.dcomplex*, %struct.dcomplex** %x_out.addr, align 8, !dbg !5596
  %49 = load %struct.dcomplex*, %struct.dcomplex** %y0.addr, align 8, !dbg !5597
  call void @ft.ll_CudaFE__Z19cffts1_gpu_kernel_3P8dcomplexS0_(%struct.dcomplex* %48, %struct.dcomplex* %49), !dbg !5595
  br label %kcall.end15, !dbg !5595

kcall.end15:                                      ; preds = %kcall.configok14, %kcall.end8
  %call16 = call i32 @cudaDeviceSynchronize(), !dbg !5598
  ret void, !dbg !5599
}

; Function Attrs: noinline uwtable
define internal void @_ZL10cffts2_gpuiP8dcomplexS0_S0_S0_S0_(i32 %is, %struct.dcomplex* %u, %struct.dcomplex* %x_in, %struct.dcomplex* %x_out, %struct.dcomplex* %y0, %struct.dcomplex* %y1) #7 !dbg !5600 {
entry:
  %is.addr = alloca i32, align 4
  %u.addr = alloca %struct.dcomplex*, align 8
  %x_in.addr = alloca %struct.dcomplex*, align 8
  %x_out.addr = alloca %struct.dcomplex*, align 8
  %y0.addr = alloca %struct.dcomplex*, align 8
  %y1.addr = alloca %struct.dcomplex*, align 8
  %agg.tmp = alloca %struct.dim3, align 4
  %agg.tmp1 = alloca %struct.dim3, align 4
  %agg.tmp.coerce = alloca { i64, i32 }, align 4
  %agg.tmp1.coerce = alloca { i64, i32 }, align 4
  %agg.tmp3 = alloca %struct.dim3, align 4
  %agg.tmp4 = alloca %struct.dim3, align 4
  %agg.tmp3.coerce = alloca { i64, i32 }, align 4
  %agg.tmp4.coerce = alloca { i64, i32 }, align 4
  %agg.tmp10 = alloca %struct.dim3, align 4
  %agg.tmp11 = alloca %struct.dim3, align 4
  %agg.tmp10.coerce = alloca { i64, i32 }, align 4
  %agg.tmp11.coerce = alloca { i64, i32 }, align 4
  store i32 %is, i32* %is.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %is.addr, metadata !5603, metadata !DIExpression()), !dbg !5604
  store %struct.dcomplex* %u, %struct.dcomplex** %u.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %u.addr, metadata !5605, metadata !DIExpression()), !dbg !5606
  store %struct.dcomplex* %x_in, %struct.dcomplex** %x_in.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %x_in.addr, metadata !5607, metadata !DIExpression()), !dbg !5608
  store %struct.dcomplex* %x_out, %struct.dcomplex** %x_out.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %x_out.addr, metadata !5609, metadata !DIExpression()), !dbg !5610
  store %struct.dcomplex* %y0, %struct.dcomplex** %y0.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %y0.addr, metadata !5611, metadata !DIExpression()), !dbg !5612
  store %struct.dcomplex* %y1, %struct.dcomplex** %y1.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %y1.addr, metadata !5613, metadata !DIExpression()), !dbg !5614
  %0 = load i32, i32* @blocks_per_grid_on_ffty_1, align 4, !dbg !5615
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp, i32 %0, i32 1, i32 1), !dbg !5615
  %1 = load i32, i32* @threads_per_block_on_ffty_1, align 4, !dbg !5616
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp1, i32 %1, i32 1, i32 1), !dbg !5616
  %2 = bitcast { i64, i32 }* %agg.tmp.coerce to i8*, !dbg !5617
  %3 = bitcast %struct.dim3* %agg.tmp to i8*, !dbg !5617
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %2, i8* align 4 %3, i64 12, i1 false), !dbg !5617
  %4 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 0, !dbg !5617
  %5 = load i64, i64* %4, align 4, !dbg !5617
  %6 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 1, !dbg !5617
  %7 = load i32, i32* %6, align 4, !dbg !5617
  %8 = bitcast { i64, i32 }* %agg.tmp1.coerce to i8*, !dbg !5617
  %9 = bitcast %struct.dim3* %agg.tmp1 to i8*, !dbg !5617
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %8, i8* align 4 %9, i64 12, i1 false), !dbg !5617
  %10 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp1.coerce, i32 0, i32 0, !dbg !5617
  %11 = load i64, i64* %10, align 4, !dbg !5617
  %12 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp1.coerce, i32 0, i32 1, !dbg !5617
  %13 = load i32, i32* %12, align 4, !dbg !5617
  %call = call i32 @cudaConfigureCall(i64 %5, i32 %7, i64 %11, i32 %13, i64 0, %struct.CUstream_st* null), !dbg !5617
  %tobool = icmp ne i32 %call, 0, !dbg !5617
  br i1 %tobool, label %kcall.end, label %kcall.configok, !dbg !5618

kcall.configok:                                   ; preds = %entry
  %14 = load %struct.dcomplex*, %struct.dcomplex** %x_in.addr, align 8, !dbg !5619
  %15 = load %struct.dcomplex*, %struct.dcomplex** %y0.addr, align 8, !dbg !5620
  call void @ft.ll_CudaFE__Z19cffts2_gpu_kernel_1P8dcomplexS0_(%struct.dcomplex* %14, %struct.dcomplex* %15), !dbg !5618
  br label %kcall.end, !dbg !5618

kcall.end:                                        ; preds = %kcall.configok, %entry
  %call2 = call i32 @cudaDeviceSynchronize(), !dbg !5621
  %16 = load i32, i32* @blocks_per_grid_on_ffty_2, align 4, !dbg !5622
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp3, i32 %16, i32 1, i32 1), !dbg !5622
  %17 = load i32, i32* @threads_per_block_on_ffty_2, align 4, !dbg !5623
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp4, i32 %17, i32 1, i32 1), !dbg !5623
  %18 = bitcast { i64, i32 }* %agg.tmp3.coerce to i8*, !dbg !5624
  %19 = bitcast %struct.dim3* %agg.tmp3 to i8*, !dbg !5624
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %18, i8* align 4 %19, i64 12, i1 false), !dbg !5624
  %20 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp3.coerce, i32 0, i32 0, !dbg !5624
  %21 = load i64, i64* %20, align 4, !dbg !5624
  %22 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp3.coerce, i32 0, i32 1, !dbg !5624
  %23 = load i32, i32* %22, align 4, !dbg !5624
  %24 = bitcast { i64, i32 }* %agg.tmp4.coerce to i8*, !dbg !5624
  %25 = bitcast %struct.dim3* %agg.tmp4 to i8*, !dbg !5624
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %24, i8* align 4 %25, i64 12, i1 false), !dbg !5624
  %26 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp4.coerce, i32 0, i32 0, !dbg !5624
  %27 = load i64, i64* %26, align 4, !dbg !5624
  %28 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp4.coerce, i32 0, i32 1, !dbg !5624
  %29 = load i32, i32* %28, align 4, !dbg !5624
  %call5 = call i32 @cudaConfigureCall(i64 %21, i32 %23, i64 %27, i32 %29, i64 0, %struct.CUstream_st* null), !dbg !5624
  %tobool6 = icmp ne i32 %call5, 0, !dbg !5624
  br i1 %tobool6, label %kcall.end8, label %kcall.configok7, !dbg !5625

kcall.configok7:                                  ; preds = %kcall.end
  %30 = load i32, i32* %is.addr, align 4, !dbg !5626
  %31 = load %struct.dcomplex*, %struct.dcomplex** %y0.addr, align 8, !dbg !5627
  %32 = load %struct.dcomplex*, %struct.dcomplex** %y1.addr, align 8, !dbg !5628
  %33 = load %struct.dcomplex*, %struct.dcomplex** %u.addr, align 8, !dbg !5629
  call void @ft.ll_CudaFE__Z19cffts2_gpu_kernel_2iP8dcomplexS0_S0_(i32 %30, %struct.dcomplex* %31, %struct.dcomplex* %32, %struct.dcomplex* %33), !dbg !5625
  br label %kcall.end8, !dbg !5625

kcall.end8:                                       ; preds = %kcall.configok7, %kcall.end
  %call9 = call i32 @cudaDeviceSynchronize(), !dbg !5630
  %34 = load i32, i32* @blocks_per_grid_on_ffty_3, align 4, !dbg !5631
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp10, i32 %34, i32 1, i32 1), !dbg !5631
  %35 = load i32, i32* @threads_per_block_on_ffty_3, align 4, !dbg !5632
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp11, i32 %35, i32 1, i32 1), !dbg !5632
  %36 = bitcast { i64, i32 }* %agg.tmp10.coerce to i8*, !dbg !5633
  %37 = bitcast %struct.dim3* %agg.tmp10 to i8*, !dbg !5633
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %36, i8* align 4 %37, i64 12, i1 false), !dbg !5633
  %38 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp10.coerce, i32 0, i32 0, !dbg !5633
  %39 = load i64, i64* %38, align 4, !dbg !5633
  %40 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp10.coerce, i32 0, i32 1, !dbg !5633
  %41 = load i32, i32* %40, align 4, !dbg !5633
  %42 = bitcast { i64, i32 }* %agg.tmp11.coerce to i8*, !dbg !5633
  %43 = bitcast %struct.dim3* %agg.tmp11 to i8*, !dbg !5633
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %42, i8* align 4 %43, i64 12, i1 false), !dbg !5633
  %44 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp11.coerce, i32 0, i32 0, !dbg !5633
  %45 = load i64, i64* %44, align 4, !dbg !5633
  %46 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp11.coerce, i32 0, i32 1, !dbg !5633
  %47 = load i32, i32* %46, align 4, !dbg !5633
  %call12 = call i32 @cudaConfigureCall(i64 %39, i32 %41, i64 %45, i32 %47, i64 0, %struct.CUstream_st* null), !dbg !5633
  %tobool13 = icmp ne i32 %call12, 0, !dbg !5633
  br i1 %tobool13, label %kcall.end15, label %kcall.configok14, !dbg !5634

kcall.configok14:                                 ; preds = %kcall.end8
  %48 = load %struct.dcomplex*, %struct.dcomplex** %x_out.addr, align 8, !dbg !5635
  %49 = load %struct.dcomplex*, %struct.dcomplex** %y0.addr, align 8, !dbg !5636
  call void @ft.ll_CudaFE__Z19cffts2_gpu_kernel_3P8dcomplexS0_(%struct.dcomplex* %48, %struct.dcomplex* %49), !dbg !5634
  br label %kcall.end15, !dbg !5634

kcall.end15:                                      ; preds = %kcall.configok14, %kcall.end8
  %call16 = call i32 @cudaDeviceSynchronize(), !dbg !5637
  ret void, !dbg !5638
}

; Function Attrs: noinline uwtable
define internal void @_ZL10cffts3_gpuiP8dcomplexS0_S0_S0_S0_(i32 %is, %struct.dcomplex* %u, %struct.dcomplex* %x_in, %struct.dcomplex* %x_out, %struct.dcomplex* %y0, %struct.dcomplex* %y1) #7 !dbg !5639 {
entry:
  %is.addr = alloca i32, align 4
  %u.addr = alloca %struct.dcomplex*, align 8
  %x_in.addr = alloca %struct.dcomplex*, align 8
  %x_out.addr = alloca %struct.dcomplex*, align 8
  %y0.addr = alloca %struct.dcomplex*, align 8
  %y1.addr = alloca %struct.dcomplex*, align 8
  %agg.tmp = alloca %struct.dim3, align 4
  %agg.tmp1 = alloca %struct.dim3, align 4
  %agg.tmp.coerce = alloca { i64, i32 }, align 4
  %agg.tmp1.coerce = alloca { i64, i32 }, align 4
  %agg.tmp3 = alloca %struct.dim3, align 4
  %agg.tmp4 = alloca %struct.dim3, align 4
  %agg.tmp3.coerce = alloca { i64, i32 }, align 4
  %agg.tmp4.coerce = alloca { i64, i32 }, align 4
  %agg.tmp10 = alloca %struct.dim3, align 4
  %agg.tmp11 = alloca %struct.dim3, align 4
  %agg.tmp10.coerce = alloca { i64, i32 }, align 4
  %agg.tmp11.coerce = alloca { i64, i32 }, align 4
  store i32 %is, i32* %is.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %is.addr, metadata !5640, metadata !DIExpression()), !dbg !5641
  store %struct.dcomplex* %u, %struct.dcomplex** %u.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %u.addr, metadata !5642, metadata !DIExpression()), !dbg !5643
  store %struct.dcomplex* %x_in, %struct.dcomplex** %x_in.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %x_in.addr, metadata !5644, metadata !DIExpression()), !dbg !5645
  store %struct.dcomplex* %x_out, %struct.dcomplex** %x_out.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %x_out.addr, metadata !5646, metadata !DIExpression()), !dbg !5647
  store %struct.dcomplex* %y0, %struct.dcomplex** %y0.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %y0.addr, metadata !5648, metadata !DIExpression()), !dbg !5649
  store %struct.dcomplex* %y1, %struct.dcomplex** %y1.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %y1.addr, metadata !5650, metadata !DIExpression()), !dbg !5651
  %0 = load i32, i32* @blocks_per_grid_on_fftz_1, align 4, !dbg !5652
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp, i32 %0, i32 1, i32 1), !dbg !5652
  %1 = load i32, i32* @threads_per_block_on_fftz_1, align 4, !dbg !5653
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp1, i32 %1, i32 1, i32 1), !dbg !5653
  %2 = bitcast { i64, i32 }* %agg.tmp.coerce to i8*, !dbg !5654
  %3 = bitcast %struct.dim3* %agg.tmp to i8*, !dbg !5654
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %2, i8* align 4 %3, i64 12, i1 false), !dbg !5654
  %4 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 0, !dbg !5654
  %5 = load i64, i64* %4, align 4, !dbg !5654
  %6 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 1, !dbg !5654
  %7 = load i32, i32* %6, align 4, !dbg !5654
  %8 = bitcast { i64, i32 }* %agg.tmp1.coerce to i8*, !dbg !5654
  %9 = bitcast %struct.dim3* %agg.tmp1 to i8*, !dbg !5654
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %8, i8* align 4 %9, i64 12, i1 false), !dbg !5654
  %10 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp1.coerce, i32 0, i32 0, !dbg !5654
  %11 = load i64, i64* %10, align 4, !dbg !5654
  %12 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp1.coerce, i32 0, i32 1, !dbg !5654
  %13 = load i32, i32* %12, align 4, !dbg !5654
  %call = call i32 @cudaConfigureCall(i64 %5, i32 %7, i64 %11, i32 %13, i64 0, %struct.CUstream_st* null), !dbg !5654
  %tobool = icmp ne i32 %call, 0, !dbg !5654
  br i1 %tobool, label %kcall.end, label %kcall.configok, !dbg !5655

kcall.configok:                                   ; preds = %entry
  %14 = load %struct.dcomplex*, %struct.dcomplex** %x_in.addr, align 8, !dbg !5656
  %15 = load %struct.dcomplex*, %struct.dcomplex** %y0.addr, align 8, !dbg !5657
  call void @ft.ll_CudaFE__Z19cffts3_gpu_kernel_1P8dcomplexS0_(%struct.dcomplex* %14, %struct.dcomplex* %15), !dbg !5655
  br label %kcall.end, !dbg !5655

kcall.end:                                        ; preds = %kcall.configok, %entry
  %call2 = call i32 @cudaDeviceSynchronize(), !dbg !5658
  %16 = load i32, i32* @blocks_per_grid_on_fftz_2, align 4, !dbg !5659
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp3, i32 %16, i32 1, i32 1), !dbg !5659
  %17 = load i32, i32* @threads_per_block_on_fftz_2, align 4, !dbg !5660
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp4, i32 %17, i32 1, i32 1), !dbg !5660
  %18 = bitcast { i64, i32 }* %agg.tmp3.coerce to i8*, !dbg !5661
  %19 = bitcast %struct.dim3* %agg.tmp3 to i8*, !dbg !5661
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %18, i8* align 4 %19, i64 12, i1 false), !dbg !5661
  %20 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp3.coerce, i32 0, i32 0, !dbg !5661
  %21 = load i64, i64* %20, align 4, !dbg !5661
  %22 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp3.coerce, i32 0, i32 1, !dbg !5661
  %23 = load i32, i32* %22, align 4, !dbg !5661
  %24 = bitcast { i64, i32 }* %agg.tmp4.coerce to i8*, !dbg !5661
  %25 = bitcast %struct.dim3* %agg.tmp4 to i8*, !dbg !5661
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %24, i8* align 4 %25, i64 12, i1 false), !dbg !5661
  %26 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp4.coerce, i32 0, i32 0, !dbg !5661
  %27 = load i64, i64* %26, align 4, !dbg !5661
  %28 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp4.coerce, i32 0, i32 1, !dbg !5661
  %29 = load i32, i32* %28, align 4, !dbg !5661
  %call5 = call i32 @cudaConfigureCall(i64 %21, i32 %23, i64 %27, i32 %29, i64 0, %struct.CUstream_st* null), !dbg !5661
  %tobool6 = icmp ne i32 %call5, 0, !dbg !5661
  br i1 %tobool6, label %kcall.end8, label %kcall.configok7, !dbg !5662

kcall.configok7:                                  ; preds = %kcall.end
  %30 = load i32, i32* %is.addr, align 4, !dbg !5663
  %31 = load %struct.dcomplex*, %struct.dcomplex** %y0.addr, align 8, !dbg !5664
  %32 = load %struct.dcomplex*, %struct.dcomplex** %y1.addr, align 8, !dbg !5665
  %33 = load %struct.dcomplex*, %struct.dcomplex** %u.addr, align 8, !dbg !5666
  call void @ft.ll_CudaFE__Z19cffts3_gpu_kernel_2iP8dcomplexS0_S0_(i32 %30, %struct.dcomplex* %31, %struct.dcomplex* %32, %struct.dcomplex* %33), !dbg !5662
  br label %kcall.end8, !dbg !5662

kcall.end8:                                       ; preds = %kcall.configok7, %kcall.end
  %call9 = call i32 @cudaDeviceSynchronize(), !dbg !5667
  %34 = load i32, i32* @blocks_per_grid_on_fftz_3, align 4, !dbg !5668
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp10, i32 %34, i32 1, i32 1), !dbg !5668
  %35 = load i32, i32* @threads_per_block_on_fftz_3, align 4, !dbg !5669
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp11, i32 %35, i32 1, i32 1), !dbg !5669
  %36 = bitcast { i64, i32 }* %agg.tmp10.coerce to i8*, !dbg !5670
  %37 = bitcast %struct.dim3* %agg.tmp10 to i8*, !dbg !5670
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %36, i8* align 4 %37, i64 12, i1 false), !dbg !5670
  %38 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp10.coerce, i32 0, i32 0, !dbg !5670
  %39 = load i64, i64* %38, align 4, !dbg !5670
  %40 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp10.coerce, i32 0, i32 1, !dbg !5670
  %41 = load i32, i32* %40, align 4, !dbg !5670
  %42 = bitcast { i64, i32 }* %agg.tmp11.coerce to i8*, !dbg !5670
  %43 = bitcast %struct.dim3* %agg.tmp11 to i8*, !dbg !5670
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %42, i8* align 4 %43, i64 12, i1 false), !dbg !5670
  %44 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp11.coerce, i32 0, i32 0, !dbg !5670
  %45 = load i64, i64* %44, align 4, !dbg !5670
  %46 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp11.coerce, i32 0, i32 1, !dbg !5670
  %47 = load i32, i32* %46, align 4, !dbg !5670
  %call12 = call i32 @cudaConfigureCall(i64 %39, i32 %41, i64 %45, i32 %47, i64 0, %struct.CUstream_st* null), !dbg !5670
  %tobool13 = icmp ne i32 %call12, 0, !dbg !5670
  br i1 %tobool13, label %kcall.end15, label %kcall.configok14, !dbg !5671

kcall.configok14:                                 ; preds = %kcall.end8
  %48 = load %struct.dcomplex*, %struct.dcomplex** %x_out.addr, align 8, !dbg !5672
  %49 = load %struct.dcomplex*, %struct.dcomplex** %y0.addr, align 8, !dbg !5673
  call void @ft.ll_CudaFE__Z19cffts3_gpu_kernel_3P8dcomplexS0_(%struct.dcomplex* %48, %struct.dcomplex* %49), !dbg !5671
  br label %kcall.end15, !dbg !5671

kcall.end15:                                      ; preds = %kcall.configok14, %kcall.end8
  %call16 = call i32 @cudaDeviceSynchronize(), !dbg !5674
  ret void, !dbg !5675
}

; Function Attrs: noinline uwtable
define dso_local void @ft.ll_CudaFE__Z19cffts3_gpu_kernel_1P8dcomplexS0_(%struct.dcomplex* %x_in, %struct.dcomplex* %y0) #7 !dbg !5676 {
entry:
  %x_in.addr = alloca %struct.dcomplex*, align 8
  %y0.addr = alloca %struct.dcomplex*, align 8
  store %struct.dcomplex* %x_in, %struct.dcomplex** %x_in.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %x_in.addr, metadata !5677, metadata !DIExpression()), !dbg !5678
  store %struct.dcomplex* %y0, %struct.dcomplex** %y0.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %y0.addr, metadata !5679, metadata !DIExpression()), !dbg !5680
  %0 = bitcast %struct.dcomplex** %x_in.addr to i8*, !dbg !5681
  %1 = call i32 @cudaSetupArgument(i8* %0, i64 8, i64 0), !dbg !5681
  %2 = icmp eq i32 %1, 0, !dbg !5681
  br i1 %2, label %setup.next, label %setup.end, !dbg !5681

setup.next:                                       ; preds = %entry
  %3 = bitcast %struct.dcomplex** %y0.addr to i8*, !dbg !5681
  %4 = call i32 @cudaSetupArgument(i8* %3, i64 8, i64 8), !dbg !5681
  %5 = icmp eq i32 %4, 0, !dbg !5681
  br i1 %5, label %setup.next1, label %setup.end, !dbg !5681

setup.next1:                                      ; preds = %setup.next
  %6 = call i32 @cudaLaunch(i8* bitcast (void (%struct.dcomplex*, %struct.dcomplex*)* @ft.ll_CudaFE__Z19cffts3_gpu_kernel_1P8dcomplexS0_ to i8*)), !dbg !5681
  br label %setup.end, !dbg !5681

setup.end:                                        ; preds = %setup.next1, %setup.next, %entry
  ret void, !dbg !5682
}

; Function Attrs: noinline uwtable
define dso_local void @ft.ll_CudaFE__Z19cffts3_gpu_kernel_2iP8dcomplexS0_S0_(i32 %is, %struct.dcomplex* %gty1, %struct.dcomplex* %gty2, %struct.dcomplex* %u_device) #7 !dbg !5683 {
entry:
  %is.addr = alloca i32, align 4
  %gty1.addr = alloca %struct.dcomplex*, align 8
  %gty2.addr = alloca %struct.dcomplex*, align 8
  %u_device.addr = alloca %struct.dcomplex*, align 8
  store i32 %is, i32* %is.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %is.addr, metadata !5684, metadata !DIExpression()), !dbg !5685
  store %struct.dcomplex* %gty1, %struct.dcomplex** %gty1.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %gty1.addr, metadata !5686, metadata !DIExpression()), !dbg !5687
  store %struct.dcomplex* %gty2, %struct.dcomplex** %gty2.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %gty2.addr, metadata !5688, metadata !DIExpression()), !dbg !5689
  store %struct.dcomplex* %u_device, %struct.dcomplex** %u_device.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %u_device.addr, metadata !5690, metadata !DIExpression()), !dbg !5691
  %0 = bitcast i32* %is.addr to i8*, !dbg !5692
  %1 = call i32 @cudaSetupArgument(i8* %0, i64 4, i64 0), !dbg !5692
  %2 = icmp eq i32 %1, 0, !dbg !5692
  br i1 %2, label %setup.next, label %setup.end, !dbg !5692

setup.next:                                       ; preds = %entry
  %3 = bitcast %struct.dcomplex** %gty1.addr to i8*, !dbg !5692
  %4 = call i32 @cudaSetupArgument(i8* %3, i64 8, i64 8), !dbg !5692
  %5 = icmp eq i32 %4, 0, !dbg !5692
  br i1 %5, label %setup.next1, label %setup.end, !dbg !5692

setup.next1:                                      ; preds = %setup.next
  %6 = bitcast %struct.dcomplex** %gty2.addr to i8*, !dbg !5692
  %7 = call i32 @cudaSetupArgument(i8* %6, i64 8, i64 16), !dbg !5692
  %8 = icmp eq i32 %7, 0, !dbg !5692
  br i1 %8, label %setup.next2, label %setup.end, !dbg !5692

setup.next2:                                      ; preds = %setup.next1
  %9 = bitcast %struct.dcomplex** %u_device.addr to i8*, !dbg !5692
  %10 = call i32 @cudaSetupArgument(i8* %9, i64 8, i64 24), !dbg !5692
  %11 = icmp eq i32 %10, 0, !dbg !5692
  br i1 %11, label %setup.next3, label %setup.end, !dbg !5692

setup.next3:                                      ; preds = %setup.next2
  %12 = call i32 @cudaLaunch(i8* bitcast (void (i32, %struct.dcomplex*, %struct.dcomplex*, %struct.dcomplex*)* @ft.ll_CudaFE__Z19cffts3_gpu_kernel_2iP8dcomplexS0_S0_ to i8*)), !dbg !5692
  br label %setup.end, !dbg !5692

setup.end:                                        ; preds = %setup.next3, %setup.next2, %setup.next1, %setup.next, %entry
  ret void, !dbg !5693
}

; Function Attrs: noinline uwtable
define dso_local void @ft.ll_CudaFE__Z19cffts3_gpu_kernel_3P8dcomplexS0_(%struct.dcomplex* %x_out, %struct.dcomplex* %y0) #7 !dbg !5694 {
entry:
  %x_out.addr = alloca %struct.dcomplex*, align 8
  %y0.addr = alloca %struct.dcomplex*, align 8
  store %struct.dcomplex* %x_out, %struct.dcomplex** %x_out.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %x_out.addr, metadata !5695, metadata !DIExpression()), !dbg !5696
  store %struct.dcomplex* %y0, %struct.dcomplex** %y0.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %y0.addr, metadata !5697, metadata !DIExpression()), !dbg !5698
  %0 = bitcast %struct.dcomplex** %x_out.addr to i8*, !dbg !5699
  %1 = call i32 @cudaSetupArgument(i8* %0, i64 8, i64 0), !dbg !5699
  %2 = icmp eq i32 %1, 0, !dbg !5699
  br i1 %2, label %setup.next, label %setup.end, !dbg !5699

setup.next:                                       ; preds = %entry
  %3 = bitcast %struct.dcomplex** %y0.addr to i8*, !dbg !5699
  %4 = call i32 @cudaSetupArgument(i8* %3, i64 8, i64 8), !dbg !5699
  %5 = icmp eq i32 %4, 0, !dbg !5699
  br i1 %5, label %setup.next1, label %setup.end, !dbg !5699

setup.next1:                                      ; preds = %setup.next
  %6 = call i32 @cudaLaunch(i8* bitcast (void (%struct.dcomplex*, %struct.dcomplex*)* @ft.ll_CudaFE__Z19cffts3_gpu_kernel_3P8dcomplexS0_ to i8*)), !dbg !5699
  br label %setup.end, !dbg !5699

setup.end:                                        ; preds = %setup.next1, %setup.next, %entry
  ret void, !dbg !5700
}

; Function Attrs: noinline uwtable
define dso_local void @ft.ll_CudaFE__Z19cffts2_gpu_kernel_1P8dcomplexS0_(%struct.dcomplex* %x_in, %struct.dcomplex* %y0) #7 !dbg !5701 {
entry:
  %x_in.addr = alloca %struct.dcomplex*, align 8
  %y0.addr = alloca %struct.dcomplex*, align 8
  store %struct.dcomplex* %x_in, %struct.dcomplex** %x_in.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %x_in.addr, metadata !5702, metadata !DIExpression()), !dbg !5703
  store %struct.dcomplex* %y0, %struct.dcomplex** %y0.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %y0.addr, metadata !5704, metadata !DIExpression()), !dbg !5705
  %0 = bitcast %struct.dcomplex** %x_in.addr to i8*, !dbg !5706
  %1 = call i32 @cudaSetupArgument(i8* %0, i64 8, i64 0), !dbg !5706
  %2 = icmp eq i32 %1, 0, !dbg !5706
  br i1 %2, label %setup.next, label %setup.end, !dbg !5706

setup.next:                                       ; preds = %entry
  %3 = bitcast %struct.dcomplex** %y0.addr to i8*, !dbg !5706
  %4 = call i32 @cudaSetupArgument(i8* %3, i64 8, i64 8), !dbg !5706
  %5 = icmp eq i32 %4, 0, !dbg !5706
  br i1 %5, label %setup.next1, label %setup.end, !dbg !5706

setup.next1:                                      ; preds = %setup.next
  %6 = call i32 @cudaLaunch(i8* bitcast (void (%struct.dcomplex*, %struct.dcomplex*)* @ft.ll_CudaFE__Z19cffts2_gpu_kernel_1P8dcomplexS0_ to i8*)), !dbg !5706
  br label %setup.end, !dbg !5706

setup.end:                                        ; preds = %setup.next1, %setup.next, %entry
  ret void, !dbg !5707
}

; Function Attrs: noinline uwtable
define dso_local void @ft.ll_CudaFE__Z19cffts2_gpu_kernel_2iP8dcomplexS0_S0_(i32 %is, %struct.dcomplex* %gty1, %struct.dcomplex* %gty2, %struct.dcomplex* %u_device) #7 !dbg !5708 {
entry:
  %is.addr = alloca i32, align 4
  %gty1.addr = alloca %struct.dcomplex*, align 8
  %gty2.addr = alloca %struct.dcomplex*, align 8
  %u_device.addr = alloca %struct.dcomplex*, align 8
  store i32 %is, i32* %is.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %is.addr, metadata !5709, metadata !DIExpression()), !dbg !5710
  store %struct.dcomplex* %gty1, %struct.dcomplex** %gty1.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %gty1.addr, metadata !5711, metadata !DIExpression()), !dbg !5712
  store %struct.dcomplex* %gty2, %struct.dcomplex** %gty2.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %gty2.addr, metadata !5713, metadata !DIExpression()), !dbg !5714
  store %struct.dcomplex* %u_device, %struct.dcomplex** %u_device.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %u_device.addr, metadata !5715, metadata !DIExpression()), !dbg !5716
  %0 = bitcast i32* %is.addr to i8*, !dbg !5717
  %1 = call i32 @cudaSetupArgument(i8* %0, i64 4, i64 0), !dbg !5717
  %2 = icmp eq i32 %1, 0, !dbg !5717
  br i1 %2, label %setup.next, label %setup.end, !dbg !5717

setup.next:                                       ; preds = %entry
  %3 = bitcast %struct.dcomplex** %gty1.addr to i8*, !dbg !5717
  %4 = call i32 @cudaSetupArgument(i8* %3, i64 8, i64 8), !dbg !5717
  %5 = icmp eq i32 %4, 0, !dbg !5717
  br i1 %5, label %setup.next1, label %setup.end, !dbg !5717

setup.next1:                                      ; preds = %setup.next
  %6 = bitcast %struct.dcomplex** %gty2.addr to i8*, !dbg !5717
  %7 = call i32 @cudaSetupArgument(i8* %6, i64 8, i64 16), !dbg !5717
  %8 = icmp eq i32 %7, 0, !dbg !5717
  br i1 %8, label %setup.next2, label %setup.end, !dbg !5717

setup.next2:                                      ; preds = %setup.next1
  %9 = bitcast %struct.dcomplex** %u_device.addr to i8*, !dbg !5717
  %10 = call i32 @cudaSetupArgument(i8* %9, i64 8, i64 24), !dbg !5717
  %11 = icmp eq i32 %10, 0, !dbg !5717
  br i1 %11, label %setup.next3, label %setup.end, !dbg !5717

setup.next3:                                      ; preds = %setup.next2
  %12 = call i32 @cudaLaunch(i8* bitcast (void (i32, %struct.dcomplex*, %struct.dcomplex*, %struct.dcomplex*)* @ft.ll_CudaFE__Z19cffts2_gpu_kernel_2iP8dcomplexS0_S0_ to i8*)), !dbg !5717
  br label %setup.end, !dbg !5717

setup.end:                                        ; preds = %setup.next3, %setup.next2, %setup.next1, %setup.next, %entry
  ret void, !dbg !5718
}

; Function Attrs: noinline uwtable
define dso_local void @ft.ll_CudaFE__Z19cffts2_gpu_kernel_3P8dcomplexS0_(%struct.dcomplex* %x_out, %struct.dcomplex* %y0) #7 !dbg !5719 {
entry:
  %x_out.addr = alloca %struct.dcomplex*, align 8
  %y0.addr = alloca %struct.dcomplex*, align 8
  store %struct.dcomplex* %x_out, %struct.dcomplex** %x_out.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %x_out.addr, metadata !5720, metadata !DIExpression()), !dbg !5721
  store %struct.dcomplex* %y0, %struct.dcomplex** %y0.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %y0.addr, metadata !5722, metadata !DIExpression()), !dbg !5723
  %0 = bitcast %struct.dcomplex** %x_out.addr to i8*, !dbg !5724
  %1 = call i32 @cudaSetupArgument(i8* %0, i64 8, i64 0), !dbg !5724
  %2 = icmp eq i32 %1, 0, !dbg !5724
  br i1 %2, label %setup.next, label %setup.end, !dbg !5724

setup.next:                                       ; preds = %entry
  %3 = bitcast %struct.dcomplex** %y0.addr to i8*, !dbg !5724
  %4 = call i32 @cudaSetupArgument(i8* %3, i64 8, i64 8), !dbg !5724
  %5 = icmp eq i32 %4, 0, !dbg !5724
  br i1 %5, label %setup.next1, label %setup.end, !dbg !5724

setup.next1:                                      ; preds = %setup.next
  %6 = call i32 @cudaLaunch(i8* bitcast (void (%struct.dcomplex*, %struct.dcomplex*)* @ft.ll_CudaFE__Z19cffts2_gpu_kernel_3P8dcomplexS0_ to i8*)), !dbg !5724
  br label %setup.end, !dbg !5724

setup.end:                                        ; preds = %setup.next1, %setup.next, %entry
  ret void, !dbg !5725
}

; Function Attrs: noinline uwtable
define dso_local void @ft.ll_CudaFE__Z19cffts1_gpu_kernel_1P8dcomplexS0_(%struct.dcomplex* %x_in, %struct.dcomplex* %y0) #7 !dbg !5726 {
entry:
  %x_in.addr = alloca %struct.dcomplex*, align 8
  %y0.addr = alloca %struct.dcomplex*, align 8
  store %struct.dcomplex* %x_in, %struct.dcomplex** %x_in.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %x_in.addr, metadata !5727, metadata !DIExpression()), !dbg !5728
  store %struct.dcomplex* %y0, %struct.dcomplex** %y0.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %y0.addr, metadata !5729, metadata !DIExpression()), !dbg !5730
  %0 = bitcast %struct.dcomplex** %x_in.addr to i8*, !dbg !5731
  %1 = call i32 @cudaSetupArgument(i8* %0, i64 8, i64 0), !dbg !5731
  %2 = icmp eq i32 %1, 0, !dbg !5731
  br i1 %2, label %setup.next, label %setup.end, !dbg !5731

setup.next:                                       ; preds = %entry
  %3 = bitcast %struct.dcomplex** %y0.addr to i8*, !dbg !5731
  %4 = call i32 @cudaSetupArgument(i8* %3, i64 8, i64 8), !dbg !5731
  %5 = icmp eq i32 %4, 0, !dbg !5731
  br i1 %5, label %setup.next1, label %setup.end, !dbg !5731

setup.next1:                                      ; preds = %setup.next
  %6 = call i32 @cudaLaunch(i8* bitcast (void (%struct.dcomplex*, %struct.dcomplex*)* @ft.ll_CudaFE__Z19cffts1_gpu_kernel_1P8dcomplexS0_ to i8*)), !dbg !5731
  br label %setup.end, !dbg !5731

setup.end:                                        ; preds = %setup.next1, %setup.next, %entry
  ret void, !dbg !5732
}

; Function Attrs: noinline uwtable
define dso_local void @ft.ll_CudaFE__Z19cffts1_gpu_kernel_2iP8dcomplexS0_S0_(i32 %is, %struct.dcomplex* %gty1, %struct.dcomplex* %gty2, %struct.dcomplex* %u_device) #7 !dbg !5733 {
entry:
  %is.addr = alloca i32, align 4
  %gty1.addr = alloca %struct.dcomplex*, align 8
  %gty2.addr = alloca %struct.dcomplex*, align 8
  %u_device.addr = alloca %struct.dcomplex*, align 8
  store i32 %is, i32* %is.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %is.addr, metadata !5734, metadata !DIExpression()), !dbg !5735
  store %struct.dcomplex* %gty1, %struct.dcomplex** %gty1.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %gty1.addr, metadata !5736, metadata !DIExpression()), !dbg !5737
  store %struct.dcomplex* %gty2, %struct.dcomplex** %gty2.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %gty2.addr, metadata !5738, metadata !DIExpression()), !dbg !5739
  store %struct.dcomplex* %u_device, %struct.dcomplex** %u_device.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %u_device.addr, metadata !5740, metadata !DIExpression()), !dbg !5741
  %0 = bitcast i32* %is.addr to i8*, !dbg !5742
  %1 = call i32 @cudaSetupArgument(i8* %0, i64 4, i64 0), !dbg !5742
  %2 = icmp eq i32 %1, 0, !dbg !5742
  br i1 %2, label %setup.next, label %setup.end, !dbg !5742

setup.next:                                       ; preds = %entry
  %3 = bitcast %struct.dcomplex** %gty1.addr to i8*, !dbg !5742
  %4 = call i32 @cudaSetupArgument(i8* %3, i64 8, i64 8), !dbg !5742
  %5 = icmp eq i32 %4, 0, !dbg !5742
  br i1 %5, label %setup.next1, label %setup.end, !dbg !5742

setup.next1:                                      ; preds = %setup.next
  %6 = bitcast %struct.dcomplex** %gty2.addr to i8*, !dbg !5742
  %7 = call i32 @cudaSetupArgument(i8* %6, i64 8, i64 16), !dbg !5742
  %8 = icmp eq i32 %7, 0, !dbg !5742
  br i1 %8, label %setup.next2, label %setup.end, !dbg !5742

setup.next2:                                      ; preds = %setup.next1
  %9 = bitcast %struct.dcomplex** %u_device.addr to i8*, !dbg !5742
  %10 = call i32 @cudaSetupArgument(i8* %9, i64 8, i64 24), !dbg !5742
  %11 = icmp eq i32 %10, 0, !dbg !5742
  br i1 %11, label %setup.next3, label %setup.end, !dbg !5742

setup.next3:                                      ; preds = %setup.next2
  %12 = call i32 @cudaLaunch(i8* bitcast (void (i32, %struct.dcomplex*, %struct.dcomplex*, %struct.dcomplex*)* @ft.ll_CudaFE__Z19cffts1_gpu_kernel_2iP8dcomplexS0_S0_ to i8*)), !dbg !5742
  br label %setup.end, !dbg !5742

setup.end:                                        ; preds = %setup.next3, %setup.next2, %setup.next1, %setup.next, %entry
  ret void, !dbg !5743
}

; Function Attrs: noinline uwtable
define dso_local void @ft.ll_CudaFE__Z19cffts1_gpu_kernel_3P8dcomplexS0_(%struct.dcomplex* %x_out, %struct.dcomplex* %y0) #7 !dbg !5744 {
entry:
  %x_out.addr = alloca %struct.dcomplex*, align 8
  %y0.addr = alloca %struct.dcomplex*, align 8
  store %struct.dcomplex* %x_out, %struct.dcomplex** %x_out.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %x_out.addr, metadata !5745, metadata !DIExpression()), !dbg !5746
  store %struct.dcomplex* %y0, %struct.dcomplex** %y0.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %y0.addr, metadata !5747, metadata !DIExpression()), !dbg !5748
  %0 = bitcast %struct.dcomplex** %x_out.addr to i8*, !dbg !5749
  %1 = call i32 @cudaSetupArgument(i8* %0, i64 8, i64 0), !dbg !5749
  %2 = icmp eq i32 %1, 0, !dbg !5749
  br i1 %2, label %setup.next, label %setup.end, !dbg !5749

setup.next:                                       ; preds = %entry
  %3 = bitcast %struct.dcomplex** %y0.addr to i8*, !dbg !5749
  %4 = call i32 @cudaSetupArgument(i8* %3, i64 8, i64 8), !dbg !5749
  %5 = icmp eq i32 %4, 0, !dbg !5749
  br i1 %5, label %setup.next1, label %setup.end, !dbg !5749

setup.next1:                                      ; preds = %setup.next
  %6 = call i32 @cudaLaunch(i8* bitcast (void (%struct.dcomplex*, %struct.dcomplex*)* @ft.ll_CudaFE__Z19cffts1_gpu_kernel_3P8dcomplexS0_ to i8*)), !dbg !5749
  br label %setup.end, !dbg !5749

setup.end:                                        ; preds = %setup.next1, %setup.next, %entry
  ret void, !dbg !5750
}

; Function Attrs: noinline nounwind uwtable
define internal i32 @_ZL5ilog2i(i32 %n) #6 !dbg !5751 {
entry:
  %retval = alloca i32, align 4
  %n.addr = alloca i32, align 4
  %nn = alloca i32, align 4
  %lg = alloca i32, align 4
  store i32 %n, i32* %n.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %n.addr, metadata !5752, metadata !DIExpression()), !dbg !5753
  call void @llvm.dbg.declare(metadata i32* %nn, metadata !5754, metadata !DIExpression()), !dbg !5755
  call void @llvm.dbg.declare(metadata i32* %lg, metadata !5756, metadata !DIExpression()), !dbg !5757
  %0 = load i32, i32* %n.addr, align 4, !dbg !5758
  %cmp = icmp eq i32 %0, 1, !dbg !5760
  br i1 %cmp, label %if.then, label %if.end, !dbg !5761

if.then:                                          ; preds = %entry
  store i32 0, i32* %retval, align 4, !dbg !5762
  br label %return, !dbg !5762

if.end:                                           ; preds = %entry
  store i32 1, i32* %lg, align 4, !dbg !5764
  store i32 2, i32* %nn, align 4, !dbg !5765
  br label %while.cond, !dbg !5766

while.cond:                                       ; preds = %while.body, %if.end
  %1 = load i32, i32* %nn, align 4, !dbg !5767
  %2 = load i32, i32* %n.addr, align 4, !dbg !5768
  %cmp1 = icmp slt i32 %1, %2, !dbg !5769
  br i1 %cmp1, label %while.body, label %while.end, !dbg !5766

while.body:                                       ; preds = %while.cond
  %3 = load i32, i32* %nn, align 4, !dbg !5770
  %shl = shl i32 %3, 1, !dbg !5772
  store i32 %shl, i32* %nn, align 4, !dbg !5773
  %4 = load i32, i32* %lg, align 4, !dbg !5774
  %inc = add nsw i32 %4, 1, !dbg !5774
  store i32 %inc, i32* %lg, align 4, !dbg !5774
  br label %while.cond, !dbg !5766, !llvm.loop !5775

while.end:                                        ; preds = %while.cond
  %5 = load i32, i32* %lg, align 4, !dbg !5777
  store i32 %5, i32* %retval, align 4, !dbg !5778
  br label %return, !dbg !5778

return:                                           ; preds = %while.end, %if.then
  %6 = load i32, i32* %retval, align 4, !dbg !5779
  ret i32 %6, !dbg !5779
}

; Function Attrs: nounwind
declare dso_local double @cos(double) #9

; Function Attrs: nounwind
declare dso_local double @sin(double) #9

; Function Attrs: noinline nounwind uwtable
define internal void @_ZL6ipow46diPd(double %a, i32 %exponent, double* %result) #6 !dbg !5780 {
entry:
  %a.addr = alloca double, align 8
  %exponent.addr = alloca i32, align 4
  %result.addr = alloca double*, align 8
  %q = alloca double, align 8
  %r = alloca double, align 8
  %n = alloca i32, align 4
  %n2 = alloca i32, align 4
  store double %a, double* %a.addr, align 8
  call void @llvm.dbg.declare(metadata double* %a.addr, metadata !5781, metadata !DIExpression()), !dbg !5782
  store i32 %exponent, i32* %exponent.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %exponent.addr, metadata !5783, metadata !DIExpression()), !dbg !5784
  store double* %result, double** %result.addr, align 8
  call void @llvm.dbg.declare(metadata double** %result.addr, metadata !5785, metadata !DIExpression()), !dbg !5786
  call void @llvm.dbg.declare(metadata double* %q, metadata !5787, metadata !DIExpression()), !dbg !5788
  call void @llvm.dbg.declare(metadata double* %r, metadata !5789, metadata !DIExpression()), !dbg !5790
  call void @llvm.dbg.declare(metadata i32* %n, metadata !5791, metadata !DIExpression()), !dbg !5792
  call void @llvm.dbg.declare(metadata i32* %n2, metadata !5793, metadata !DIExpression()), !dbg !5794
  %0 = load double*, double** %result.addr, align 8, !dbg !5795
  store double 1.000000e+00, double* %0, align 8, !dbg !5796
  %1 = load i32, i32* %exponent.addr, align 4, !dbg !5797
  %cmp = icmp eq i32 %1, 0, !dbg !5799
  br i1 %cmp, label %if.then, label %if.end, !dbg !5800

if.then:                                          ; preds = %entry
  br label %return, !dbg !5801

if.end:                                           ; preds = %entry
  %2 = load double, double* %a.addr, align 8, !dbg !5803
  store double %2, double* %q, align 8, !dbg !5804
  store double 1.000000e+00, double* %r, align 8, !dbg !5805
  %3 = load i32, i32* %exponent.addr, align 4, !dbg !5806
  store i32 %3, i32* %n, align 4, !dbg !5807
  br label %while.cond, !dbg !5808

while.cond:                                       ; preds = %if.end5, %if.end
  %4 = load i32, i32* %n, align 4, !dbg !5809
  %cmp1 = icmp sgt i32 %4, 1, !dbg !5810
  br i1 %cmp1, label %while.body, label %while.end, !dbg !5808

while.body:                                       ; preds = %while.cond
  %5 = load i32, i32* %n, align 4, !dbg !5811
  %div = sdiv i32 %5, 2, !dbg !5813
  store i32 %div, i32* %n2, align 4, !dbg !5814
  %6 = load i32, i32* %n2, align 4, !dbg !5815
  %mul = mul nsw i32 %6, 2, !dbg !5817
  %7 = load i32, i32* %n, align 4, !dbg !5818
  %cmp2 = icmp eq i32 %mul, %7, !dbg !5819
  br i1 %cmp2, label %if.then3, label %if.else, !dbg !5820

if.then3:                                         ; preds = %while.body
  %8 = load double, double* %q, align 8, !dbg !5821
  %call = call double @_Z6randlcPdd(double* %q, double %8), !dbg !5823
  %9 = load i32, i32* %n2, align 4, !dbg !5824
  store i32 %9, i32* %n, align 4, !dbg !5825
  br label %if.end5, !dbg !5826

if.else:                                          ; preds = %while.body
  %10 = load double, double* %q, align 8, !dbg !5827
  %call4 = call double @_Z6randlcPdd(double* %r, double %10), !dbg !5829
  %11 = load i32, i32* %n, align 4, !dbg !5830
  %sub = sub nsw i32 %11, 1, !dbg !5831
  store i32 %sub, i32* %n, align 4, !dbg !5832
  br label %if.end5

if.end5:                                          ; preds = %if.else, %if.then3
  br label %while.cond, !dbg !5808, !llvm.loop !5833

while.end:                                        ; preds = %while.cond
  %12 = load double, double* %q, align 8, !dbg !5835
  %call6 = call double @_Z6randlcPdd(double* %r, double %12), !dbg !5836
  %13 = load double, double* %r, align 8, !dbg !5837
  %14 = load double*, double** %result.addr, align 8, !dbg !5838
  store double %13, double* %14, align 8, !dbg !5839
  br label %return, !dbg !5840

return:                                           ; preds = %while.end, %if.then
  ret void, !dbg !5840
}

; Function Attrs: noinline uwtable
define dso_local void @ft.ll_CudaFE__Z37compute_initial_conditions_gpu_kernelP8dcomplexPd(%struct.dcomplex* %u0, double* %starts) #7 !dbg !5841 {
entry:
  %u0.addr = alloca %struct.dcomplex*, align 8
  %starts.addr = alloca double*, align 8
  store %struct.dcomplex* %u0, %struct.dcomplex** %u0.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %u0.addr, metadata !5842, metadata !DIExpression()), !dbg !5843
  store double* %starts, double** %starts.addr, align 8
  call void @llvm.dbg.declare(metadata double** %starts.addr, metadata !5844, metadata !DIExpression()), !dbg !5845
  %0 = bitcast %struct.dcomplex** %u0.addr to i8*, !dbg !5846
  %1 = call i32 @cudaSetupArgument(i8* %0, i64 8, i64 0), !dbg !5846
  %2 = icmp eq i32 %1, 0, !dbg !5846
  br i1 %2, label %setup.next, label %setup.end, !dbg !5846

setup.next:                                       ; preds = %entry
  %3 = bitcast double** %starts.addr to i8*, !dbg !5846
  %4 = call i32 @cudaSetupArgument(i8* %3, i64 8, i64 8), !dbg !5846
  %5 = icmp eq i32 %4, 0, !dbg !5846
  br i1 %5, label %setup.next1, label %setup.end, !dbg !5846

setup.next1:                                      ; preds = %setup.next
  %6 = call i32 @cudaLaunch(i8* bitcast (void (%struct.dcomplex*, double*)* @ft.ll_CudaFE__Z37compute_initial_conditions_gpu_kernelP8dcomplexPd to i8*)), !dbg !5846
  br label %setup.end, !dbg !5846

setup.end:                                        ; preds = %setup.next1, %setup.next, %entry
  ret void, !dbg !5847
}

; Function Attrs: noinline uwtable
define dso_local void @ft.ll_CudaFE__Z27compute_indexmap_gpu_kernelPd(double* %twiddle) #7 !dbg !5848 {
entry:
  %twiddle.addr = alloca double*, align 8
  store double* %twiddle, double** %twiddle.addr, align 8
  call void @llvm.dbg.declare(metadata double** %twiddle.addr, metadata !5849, metadata !DIExpression()), !dbg !5850
  %0 = bitcast double** %twiddle.addr to i8*, !dbg !5851
  %1 = call i32 @cudaSetupArgument(i8* %0, i64 8, i64 0), !dbg !5851
  %2 = icmp eq i32 %1, 0, !dbg !5851
  br i1 %2, label %setup.next, label %setup.end, !dbg !5851

setup.next:                                       ; preds = %entry
  %3 = call i32 @cudaLaunch(i8* bitcast (void (double*)* @ft.ll_CudaFE__Z27compute_indexmap_gpu_kernelPd to i8*)), !dbg !5851
  br label %setup.end, !dbg !5851

setup.end:                                        ; preds = %setup.next, %entry
  ret void, !dbg !5852
}

; Function Attrs: noinline uwtable
define dso_local void @ft.ll_CudaFE__Z18init_ui_gpu_kernelP8dcomplexS0_Pd(%struct.dcomplex* %u0, %struct.dcomplex* %u1, double* %twiddle) #7 !dbg !5853 {
entry:
  %u0.addr = alloca %struct.dcomplex*, align 8
  %u1.addr = alloca %struct.dcomplex*, align 8
  %twiddle.addr = alloca double*, align 8
  store %struct.dcomplex* %u0, %struct.dcomplex** %u0.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %u0.addr, metadata !5854, metadata !DIExpression()), !dbg !5855
  store %struct.dcomplex* %u1, %struct.dcomplex** %u1.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %u1.addr, metadata !5856, metadata !DIExpression()), !dbg !5857
  store double* %twiddle, double** %twiddle.addr, align 8
  call void @llvm.dbg.declare(metadata double** %twiddle.addr, metadata !5858, metadata !DIExpression()), !dbg !5859
  %0 = bitcast %struct.dcomplex** %u0.addr to i8*, !dbg !5860
  %1 = call i32 @cudaSetupArgument(i8* %0, i64 8, i64 0), !dbg !5860
  %2 = icmp eq i32 %1, 0, !dbg !5860
  br i1 %2, label %setup.next, label %setup.end, !dbg !5860

setup.next:                                       ; preds = %entry
  %3 = bitcast %struct.dcomplex** %u1.addr to i8*, !dbg !5860
  %4 = call i32 @cudaSetupArgument(i8* %3, i64 8, i64 8), !dbg !5860
  %5 = icmp eq i32 %4, 0, !dbg !5860
  br i1 %5, label %setup.next1, label %setup.end, !dbg !5860

setup.next1:                                      ; preds = %setup.next
  %6 = bitcast double** %twiddle.addr to i8*, !dbg !5860
  %7 = call i32 @cudaSetupArgument(i8* %6, i64 8, i64 16), !dbg !5860
  %8 = icmp eq i32 %7, 0, !dbg !5860
  br i1 %8, label %setup.next2, label %setup.end, !dbg !5860

setup.next2:                                      ; preds = %setup.next1
  %9 = call i32 @cudaLaunch(i8* bitcast (void (%struct.dcomplex*, %struct.dcomplex*, double*)* @ft.ll_CudaFE__Z18init_ui_gpu_kernelP8dcomplexS0_Pd to i8*)), !dbg !5860
  br label %setup.end, !dbg !5860

setup.end:                                        ; preds = %setup.next2, %setup.next1, %setup.next, %entry
  ret void, !dbg !5861
}

; Function Attrs: nounwind readnone speculatable
declare double @llvm.ceil.f64(double) #1

; Function Attrs: noinline uwtable
define internal i32 @_ZL10cudaMallocI8dcomplexE9cudaErrorPPT_m(%struct.dcomplex** %devPtr, i64 %size) #7 !dbg !5862 {
entry:
  %devPtr.addr = alloca %struct.dcomplex**, align 8
  %size.addr = alloca i64, align 8
  store %struct.dcomplex** %devPtr, %struct.dcomplex*** %devPtr.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex*** %devPtr.addr, metadata !5871, metadata !DIExpression()), !dbg !5872
  store i64 %size, i64* %size.addr, align 8
  call void @llvm.dbg.declare(metadata i64* %size.addr, metadata !5873, metadata !DIExpression()), !dbg !5874
  %0 = load %struct.dcomplex**, %struct.dcomplex*** %devPtr.addr, align 8, !dbg !5875
  %1 = bitcast %struct.dcomplex** %0 to i8*, !dbg !5875
  %2 = bitcast i8* %1 to i8**, !dbg !5876
  %3 = load i64, i64* %size.addr, align 8, !dbg !5877
  %call = call i32 @cudaMalloc(i8** %2, i64 %3), !dbg !5878
  ret i32 %call, !dbg !5879
}

; Function Attrs: noinline uwtable
define internal i32 @_ZL10cudaMallocIdE9cudaErrorPPT_m(double** %devPtr, i64 %size) #7 !dbg !5880 {
entry:
  %devPtr.addr = alloca double**, align 8
  %size.addr = alloca i64, align 8
  store double** %devPtr, double*** %devPtr.addr, align 8
  call void @llvm.dbg.declare(metadata double*** %devPtr.addr, metadata !5886, metadata !DIExpression()), !dbg !5887
  store i64 %size, i64* %size.addr, align 8
  call void @llvm.dbg.declare(metadata i64* %size.addr, metadata !5888, metadata !DIExpression()), !dbg !5889
  %0 = load double**, double*** %devPtr.addr, align 8, !dbg !5890
  %1 = bitcast double** %0 to i8*, !dbg !5890
  %2 = bitcast i8* %1 to i8**, !dbg !5891
  %3 = load i64, i64* %size.addr, align 8, !dbg !5892
  %call = call i32 @cudaMalloc(i8** %2, i64 %3), !dbg !5893
  ret i32 %call, !dbg !5894
}

declare dso_local void @omp_set_num_threads(i32) #8

declare dso_local i32 @cudaMalloc(i8**, i64) #8

attributes #0 = { convergent noinline nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="sm_20" "target-features"="+ptx42,+sm_20" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { nounwind readnone }
attributes #3 = { argmemonly nounwind }
attributes #4 = { convergent nounwind }
attributes #5 = { convergent noinline nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="sm_20" "target-features"="+ptx42,+sm_20" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #7 = { noinline uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #8 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #9 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #10 = { noinline norecurse uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #11 = { nounwind }

!llvm.dbg.cu = !{!1050, !2}
!nvvm.annotations = !{!1127, !1128, !1129, !1130, !1131, !1132, !1133, !1134, !1135, !1136, !1137, !1138, !1139, !1140, !1141, !1142, !1141, !1143, !1143, !1143, !1143, !1144, !1144, !1143}
!llvm.ident = !{!1145, !1145}
!nvvm.internalize.after.link = !{}
!nvvmir.version = !{!1146}
!llvm.module.flags = !{!1147, !1148, !1149, !1150, !1151}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "starts_device", scope: !2, file: !3, line: 173, type: !106, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 9.0.0 (https://github.com/yebinchon/llvm-project/ a1efa594106d738d0b74c9e4e2b9b779eb8b7d25)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !96, globals: !112, imports: !299, nameTableKind: None)
!3 = !DIFile(filename: "ft.cu", directory: "/scratch/ah7226/NPB-GPU/CUDA/FT")
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
!96 = !{!97, !98, !106, !107, !104, !108, !110, !111}
!97 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!98 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !99, size: 64)
!99 = !DIDerivedType(tag: DW_TAG_typedef, name: "dcomplex", file: !100, line: 81, baseType: !101)
!100 = !DIFile(filename: "./../common/npb-CPP.hpp", directory: "/scratch/ah7226/NPB-GPU/CUDA/FT")
!101 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !100, line: 81, size: 128, flags: DIFlagTypePassByValue, elements: !102, identifier: "_ZTS8dcomplex")
!102 = !{!103, !105}
!103 = !DIDerivedType(tag: DW_TAG_member, name: "real", scope: !101, file: !100, line: 81, baseType: !104, size: 64)
!104 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
!105 = !DIDerivedType(tag: DW_TAG_member, name: "imag", scope: !101, file: !100, line: 81, baseType: !104, size: 64, offset: 64)
!106 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !104, size: 64)
!107 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !97, size: 64)
!108 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !109, size: 64)
!109 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!110 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !111, size: 64)
!111 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!112 = !{!0, !113, !115, !117, !119, !121, !123, !125, !127, !129, !134, !136, !138, !140, !142, !144, !146, !148, !150, !152, !154, !156, !158, !160, !162, !164, !166, !168, !170, !172, !174, !176, !178, !180, !182, !184, !186, !188, !190, !192, !194, !196, !198, !200, !202, !204, !206, !208, !210, !285, !287, !289, !291, !293, !295, !297}
!113 = !DIGlobalVariableExpression(var: !114, expr: !DIExpression())
!114 = distinct !DIGlobalVariable(name: "twiddle_device", scope: !2, file: !3, line: 174, type: !106, isLocal: false, isDefinition: true)
!115 = !DIGlobalVariableExpression(var: !116, expr: !DIExpression())
!116 = distinct !DIGlobalVariable(name: "sums_device", scope: !2, file: !3, line: 175, type: !98, isLocal: false, isDefinition: true)
!117 = !DIGlobalVariableExpression(var: !118, expr: !DIExpression())
!118 = distinct !DIGlobalVariable(name: "u_device", scope: !2, file: !3, line: 176, type: !98, isLocal: false, isDefinition: true)
!119 = !DIGlobalVariableExpression(var: !120, expr: !DIExpression())
!120 = distinct !DIGlobalVariable(name: "u0_device", scope: !2, file: !3, line: 177, type: !98, isLocal: false, isDefinition: true)
!121 = !DIGlobalVariableExpression(var: !122, expr: !DIExpression())
!122 = distinct !DIGlobalVariable(name: "u1_device", scope: !2, file: !3, line: 178, type: !98, isLocal: false, isDefinition: true)
!123 = !DIGlobalVariableExpression(var: !124, expr: !DIExpression())
!124 = distinct !DIGlobalVariable(name: "u2_device", scope: !2, file: !3, line: 179, type: !98, isLocal: false, isDefinition: true)
!125 = !DIGlobalVariableExpression(var: !126, expr: !DIExpression())
!126 = distinct !DIGlobalVariable(name: "y0_device", scope: !2, file: !3, line: 180, type: !98, isLocal: false, isDefinition: true)
!127 = !DIGlobalVariableExpression(var: !128, expr: !DIExpression())
!128 = distinct !DIGlobalVariable(name: "y1_device", scope: !2, file: !3, line: 181, type: !98, isLocal: false, isDefinition: true)
!129 = !DIGlobalVariableExpression(var: !130, expr: !DIExpression())
!130 = distinct !DIGlobalVariable(name: "size_sums_device", scope: !2, file: !3, line: 182, type: !131, isLocal: false, isDefinition: true)
!131 = !DIDerivedType(tag: DW_TAG_typedef, name: "size_t", file: !132, line: 46, baseType: !133)
!132 = !DIFile(filename: "/u/NAS_SCRATCH/ah7226/tulip/llvm-install-tulip/lib/clang/9.0.0/include/stddef.h", directory: "")
!133 = !DIBasicType(name: "long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!134 = !DIGlobalVariableExpression(var: !135, expr: !DIExpression())
!135 = distinct !DIGlobalVariable(name: "size_starts_device", scope: !2, file: !3, line: 183, type: !131, isLocal: false, isDefinition: true)
!136 = !DIGlobalVariableExpression(var: !137, expr: !DIExpression())
!137 = distinct !DIGlobalVariable(name: "size_twiddle_device", scope: !2, file: !3, line: 184, type: !131, isLocal: false, isDefinition: true)
!138 = !DIGlobalVariableExpression(var: !139, expr: !DIExpression())
!139 = distinct !DIGlobalVariable(name: "size_u_device", scope: !2, file: !3, line: 185, type: !131, isLocal: false, isDefinition: true)
!140 = !DIGlobalVariableExpression(var: !141, expr: !DIExpression())
!141 = distinct !DIGlobalVariable(name: "size_u0_device", scope: !2, file: !3, line: 186, type: !131, isLocal: false, isDefinition: true)
!142 = !DIGlobalVariableExpression(var: !143, expr: !DIExpression())
!143 = distinct !DIGlobalVariable(name: "size_u1_device", scope: !2, file: !3, line: 187, type: !131, isLocal: false, isDefinition: true)
!144 = !DIGlobalVariableExpression(var: !145, expr: !DIExpression())
!145 = distinct !DIGlobalVariable(name: "size_y0_device", scope: !2, file: !3, line: 188, type: !131, isLocal: false, isDefinition: true)
!146 = !DIGlobalVariableExpression(var: !147, expr: !DIExpression())
!147 = distinct !DIGlobalVariable(name: "size_y1_device", scope: !2, file: !3, line: 189, type: !131, isLocal: false, isDefinition: true)
!148 = !DIGlobalVariableExpression(var: !149, expr: !DIExpression())
!149 = distinct !DIGlobalVariable(name: "size_shared_data", scope: !2, file: !3, line: 190, type: !131, isLocal: false, isDefinition: true)
!150 = !DIGlobalVariableExpression(var: !151, expr: !DIExpression())
!151 = distinct !DIGlobalVariable(name: "blocks_per_grid_on_compute_indexmap", scope: !2, file: !3, line: 191, type: !97, isLocal: false, isDefinition: true)
!152 = !DIGlobalVariableExpression(var: !153, expr: !DIExpression())
!153 = distinct !DIGlobalVariable(name: "blocks_per_grid_on_compute_initial_conditions", scope: !2, file: !3, line: 192, type: !97, isLocal: false, isDefinition: true)
!154 = !DIGlobalVariableExpression(var: !155, expr: !DIExpression())
!155 = distinct !DIGlobalVariable(name: "blocks_per_grid_on_init_ui", scope: !2, file: !3, line: 193, type: !97, isLocal: false, isDefinition: true)
!156 = !DIGlobalVariableExpression(var: !157, expr: !DIExpression())
!157 = distinct !DIGlobalVariable(name: "blocks_per_grid_on_evolve", scope: !2, file: !3, line: 194, type: !97, isLocal: false, isDefinition: true)
!158 = !DIGlobalVariableExpression(var: !159, expr: !DIExpression())
!159 = distinct !DIGlobalVariable(name: "blocks_per_grid_on_fftx_1", scope: !2, file: !3, line: 195, type: !97, isLocal: false, isDefinition: true)
!160 = !DIGlobalVariableExpression(var: !161, expr: !DIExpression())
!161 = distinct !DIGlobalVariable(name: "blocks_per_grid_on_fftx_2", scope: !2, file: !3, line: 196, type: !97, isLocal: false, isDefinition: true)
!162 = !DIGlobalVariableExpression(var: !163, expr: !DIExpression())
!163 = distinct !DIGlobalVariable(name: "blocks_per_grid_on_fftx_3", scope: !2, file: !3, line: 197, type: !97, isLocal: false, isDefinition: true)
!164 = !DIGlobalVariableExpression(var: !165, expr: !DIExpression())
!165 = distinct !DIGlobalVariable(name: "blocks_per_grid_on_ffty_1", scope: !2, file: !3, line: 198, type: !97, isLocal: false, isDefinition: true)
!166 = !DIGlobalVariableExpression(var: !167, expr: !DIExpression())
!167 = distinct !DIGlobalVariable(name: "blocks_per_grid_on_ffty_2", scope: !2, file: !3, line: 199, type: !97, isLocal: false, isDefinition: true)
!168 = !DIGlobalVariableExpression(var: !169, expr: !DIExpression())
!169 = distinct !DIGlobalVariable(name: "blocks_per_grid_on_ffty_3", scope: !2, file: !3, line: 200, type: !97, isLocal: false, isDefinition: true)
!170 = !DIGlobalVariableExpression(var: !171, expr: !DIExpression())
!171 = distinct !DIGlobalVariable(name: "blocks_per_grid_on_fftz_1", scope: !2, file: !3, line: 201, type: !97, isLocal: false, isDefinition: true)
!172 = !DIGlobalVariableExpression(var: !173, expr: !DIExpression())
!173 = distinct !DIGlobalVariable(name: "blocks_per_grid_on_fftz_2", scope: !2, file: !3, line: 202, type: !97, isLocal: false, isDefinition: true)
!174 = !DIGlobalVariableExpression(var: !175, expr: !DIExpression())
!175 = distinct !DIGlobalVariable(name: "blocks_per_grid_on_fftz_3", scope: !2, file: !3, line: 203, type: !97, isLocal: false, isDefinition: true)
!176 = !DIGlobalVariableExpression(var: !177, expr: !DIExpression())
!177 = distinct !DIGlobalVariable(name: "blocks_per_grid_on_checksum", scope: !2, file: !3, line: 204, type: !97, isLocal: false, isDefinition: true)
!178 = !DIGlobalVariableExpression(var: !179, expr: !DIExpression())
!179 = distinct !DIGlobalVariable(name: "threads_per_block_on_compute_indexmap", scope: !2, file: !3, line: 205, type: !97, isLocal: false, isDefinition: true)
!180 = !DIGlobalVariableExpression(var: !181, expr: !DIExpression())
!181 = distinct !DIGlobalVariable(name: "threads_per_block_on_compute_initial_conditions", scope: !2, file: !3, line: 206, type: !97, isLocal: false, isDefinition: true)
!182 = !DIGlobalVariableExpression(var: !183, expr: !DIExpression())
!183 = distinct !DIGlobalVariable(name: "threads_per_block_on_init_ui", scope: !2, file: !3, line: 207, type: !97, isLocal: false, isDefinition: true)
!184 = !DIGlobalVariableExpression(var: !185, expr: !DIExpression())
!185 = distinct !DIGlobalVariable(name: "threads_per_block_on_evolve", scope: !2, file: !3, line: 208, type: !97, isLocal: false, isDefinition: true)
!186 = !DIGlobalVariableExpression(var: !187, expr: !DIExpression())
!187 = distinct !DIGlobalVariable(name: "threads_per_block_on_fftx_1", scope: !2, file: !3, line: 209, type: !97, isLocal: false, isDefinition: true)
!188 = !DIGlobalVariableExpression(var: !189, expr: !DIExpression())
!189 = distinct !DIGlobalVariable(name: "threads_per_block_on_fftx_2", scope: !2, file: !3, line: 210, type: !97, isLocal: false, isDefinition: true)
!190 = !DIGlobalVariableExpression(var: !191, expr: !DIExpression())
!191 = distinct !DIGlobalVariable(name: "threads_per_block_on_fftx_3", scope: !2, file: !3, line: 211, type: !97, isLocal: false, isDefinition: true)
!192 = !DIGlobalVariableExpression(var: !193, expr: !DIExpression())
!193 = distinct !DIGlobalVariable(name: "threads_per_block_on_ffty_1", scope: !2, file: !3, line: 212, type: !97, isLocal: false, isDefinition: true)
!194 = !DIGlobalVariableExpression(var: !195, expr: !DIExpression())
!195 = distinct !DIGlobalVariable(name: "threads_per_block_on_ffty_2", scope: !2, file: !3, line: 213, type: !97, isLocal: false, isDefinition: true)
!196 = !DIGlobalVariableExpression(var: !197, expr: !DIExpression())
!197 = distinct !DIGlobalVariable(name: "threads_per_block_on_ffty_3", scope: !2, file: !3, line: 214, type: !97, isLocal: false, isDefinition: true)
!198 = !DIGlobalVariableExpression(var: !199, expr: !DIExpression())
!199 = distinct !DIGlobalVariable(name: "threads_per_block_on_fftz_1", scope: !2, file: !3, line: 215, type: !97, isLocal: false, isDefinition: true)
!200 = !DIGlobalVariableExpression(var: !201, expr: !DIExpression())
!201 = distinct !DIGlobalVariable(name: "threads_per_block_on_fftz_2", scope: !2, file: !3, line: 216, type: !97, isLocal: false, isDefinition: true)
!202 = !DIGlobalVariableExpression(var: !203, expr: !DIExpression())
!203 = distinct !DIGlobalVariable(name: "threads_per_block_on_fftz_3", scope: !2, file: !3, line: 217, type: !97, isLocal: false, isDefinition: true)
!204 = !DIGlobalVariableExpression(var: !205, expr: !DIExpression())
!205 = distinct !DIGlobalVariable(name: "threads_per_block_on_checksum", scope: !2, file: !3, line: 218, type: !97, isLocal: false, isDefinition: true)
!206 = !DIGlobalVariableExpression(var: !207, expr: !DIExpression())
!207 = distinct !DIGlobalVariable(name: "gpu_device_id", scope: !2, file: !3, line: 219, type: !97, isLocal: false, isDefinition: true)
!208 = !DIGlobalVariableExpression(var: !209, expr: !DIExpression())
!209 = distinct !DIGlobalVariable(name: "total_devices", scope: !2, file: !3, line: 220, type: !97, isLocal: false, isDefinition: true)
!210 = !DIGlobalVariableExpression(var: !211, expr: !DIExpression())
!211 = distinct !DIGlobalVariable(name: "gpu_device_properties", scope: !2, file: !3, line: 221, type: !212, isLocal: false, isDefinition: true)
!212 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "cudaDeviceProp", file: !6, line: 1257, size: 5056, flags: DIFlagTypePassByValue, elements: !213, identifier: "_ZTS14cudaDeviceProp")
!213 = !{!214, !218, !219, !220, !221, !222, !223, !224, !228, !229, !230, !231, !232, !233, !234, !235, !236, !237, !238, !239, !240, !241, !242, !243, !244, !248, !249, !250, !251, !252, !253, !254, !255, !256, !257, !258, !259, !260, !261, !262, !263, !264, !265, !266, !267, !268, !269, !270, !271, !272, !273, !274, !275, !276, !277, !278, !279, !280, !281, !282, !283, !284}
!214 = !DIDerivedType(tag: DW_TAG_member, name: "name", scope: !212, file: !6, line: 1259, baseType: !215, size: 2048)
!215 = !DICompositeType(tag: DW_TAG_array_type, baseType: !109, size: 2048, elements: !216)
!216 = !{!217}
!217 = !DISubrange(count: 256)
!218 = !DIDerivedType(tag: DW_TAG_member, name: "totalGlobalMem", scope: !212, file: !6, line: 1260, baseType: !131, size: 64, offset: 2048)
!219 = !DIDerivedType(tag: DW_TAG_member, name: "sharedMemPerBlock", scope: !212, file: !6, line: 1261, baseType: !131, size: 64, offset: 2112)
!220 = !DIDerivedType(tag: DW_TAG_member, name: "regsPerBlock", scope: !212, file: !6, line: 1262, baseType: !97, size: 32, offset: 2176)
!221 = !DIDerivedType(tag: DW_TAG_member, name: "warpSize", scope: !212, file: !6, line: 1263, baseType: !97, size: 32, offset: 2208)
!222 = !DIDerivedType(tag: DW_TAG_member, name: "memPitch", scope: !212, file: !6, line: 1264, baseType: !131, size: 64, offset: 2240)
!223 = !DIDerivedType(tag: DW_TAG_member, name: "maxThreadsPerBlock", scope: !212, file: !6, line: 1265, baseType: !97, size: 32, offset: 2304)
!224 = !DIDerivedType(tag: DW_TAG_member, name: "maxThreadsDim", scope: !212, file: !6, line: 1266, baseType: !225, size: 96, offset: 2336)
!225 = !DICompositeType(tag: DW_TAG_array_type, baseType: !97, size: 96, elements: !226)
!226 = !{!227}
!227 = !DISubrange(count: 3)
!228 = !DIDerivedType(tag: DW_TAG_member, name: "maxGridSize", scope: !212, file: !6, line: 1267, baseType: !225, size: 96, offset: 2432)
!229 = !DIDerivedType(tag: DW_TAG_member, name: "clockRate", scope: !212, file: !6, line: 1268, baseType: !97, size: 32, offset: 2528)
!230 = !DIDerivedType(tag: DW_TAG_member, name: "totalConstMem", scope: !212, file: !6, line: 1269, baseType: !131, size: 64, offset: 2560)
!231 = !DIDerivedType(tag: DW_TAG_member, name: "major", scope: !212, file: !6, line: 1270, baseType: !97, size: 32, offset: 2624)
!232 = !DIDerivedType(tag: DW_TAG_member, name: "minor", scope: !212, file: !6, line: 1271, baseType: !97, size: 32, offset: 2656)
!233 = !DIDerivedType(tag: DW_TAG_member, name: "textureAlignment", scope: !212, file: !6, line: 1272, baseType: !131, size: 64, offset: 2688)
!234 = !DIDerivedType(tag: DW_TAG_member, name: "texturePitchAlignment", scope: !212, file: !6, line: 1273, baseType: !131, size: 64, offset: 2752)
!235 = !DIDerivedType(tag: DW_TAG_member, name: "deviceOverlap", scope: !212, file: !6, line: 1274, baseType: !97, size: 32, offset: 2816)
!236 = !DIDerivedType(tag: DW_TAG_member, name: "multiProcessorCount", scope: !212, file: !6, line: 1275, baseType: !97, size: 32, offset: 2848)
!237 = !DIDerivedType(tag: DW_TAG_member, name: "kernelExecTimeoutEnabled", scope: !212, file: !6, line: 1276, baseType: !97, size: 32, offset: 2880)
!238 = !DIDerivedType(tag: DW_TAG_member, name: "integrated", scope: !212, file: !6, line: 1277, baseType: !97, size: 32, offset: 2912)
!239 = !DIDerivedType(tag: DW_TAG_member, name: "canMapHostMemory", scope: !212, file: !6, line: 1278, baseType: !97, size: 32, offset: 2944)
!240 = !DIDerivedType(tag: DW_TAG_member, name: "computeMode", scope: !212, file: !6, line: 1279, baseType: !97, size: 32, offset: 2976)
!241 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture1D", scope: !212, file: !6, line: 1280, baseType: !97, size: 32, offset: 3008)
!242 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture1DMipmap", scope: !212, file: !6, line: 1281, baseType: !97, size: 32, offset: 3040)
!243 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture1DLinear", scope: !212, file: !6, line: 1282, baseType: !97, size: 32, offset: 3072)
!244 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture2D", scope: !212, file: !6, line: 1283, baseType: !245, size: 64, offset: 3104)
!245 = !DICompositeType(tag: DW_TAG_array_type, baseType: !97, size: 64, elements: !246)
!246 = !{!247}
!247 = !DISubrange(count: 2)
!248 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture2DMipmap", scope: !212, file: !6, line: 1284, baseType: !245, size: 64, offset: 3168)
!249 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture2DLinear", scope: !212, file: !6, line: 1285, baseType: !225, size: 96, offset: 3232)
!250 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture2DGather", scope: !212, file: !6, line: 1286, baseType: !245, size: 64, offset: 3328)
!251 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture3D", scope: !212, file: !6, line: 1287, baseType: !225, size: 96, offset: 3392)
!252 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture3DAlt", scope: !212, file: !6, line: 1288, baseType: !225, size: 96, offset: 3488)
!253 = !DIDerivedType(tag: DW_TAG_member, name: "maxTextureCubemap", scope: !212, file: !6, line: 1289, baseType: !97, size: 32, offset: 3584)
!254 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture1DLayered", scope: !212, file: !6, line: 1290, baseType: !245, size: 64, offset: 3616)
!255 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture2DLayered", scope: !212, file: !6, line: 1291, baseType: !225, size: 96, offset: 3680)
!256 = !DIDerivedType(tag: DW_TAG_member, name: "maxTextureCubemapLayered", scope: !212, file: !6, line: 1292, baseType: !245, size: 64, offset: 3776)
!257 = !DIDerivedType(tag: DW_TAG_member, name: "maxSurface1D", scope: !212, file: !6, line: 1293, baseType: !97, size: 32, offset: 3840)
!258 = !DIDerivedType(tag: DW_TAG_member, name: "maxSurface2D", scope: !212, file: !6, line: 1294, baseType: !245, size: 64, offset: 3872)
!259 = !DIDerivedType(tag: DW_TAG_member, name: "maxSurface3D", scope: !212, file: !6, line: 1295, baseType: !225, size: 96, offset: 3936)
!260 = !DIDerivedType(tag: DW_TAG_member, name: "maxSurface1DLayered", scope: !212, file: !6, line: 1296, baseType: !245, size: 64, offset: 4032)
!261 = !DIDerivedType(tag: DW_TAG_member, name: "maxSurface2DLayered", scope: !212, file: !6, line: 1297, baseType: !225, size: 96, offset: 4096)
!262 = !DIDerivedType(tag: DW_TAG_member, name: "maxSurfaceCubemap", scope: !212, file: !6, line: 1298, baseType: !97, size: 32, offset: 4192)
!263 = !DIDerivedType(tag: DW_TAG_member, name: "maxSurfaceCubemapLayered", scope: !212, file: !6, line: 1299, baseType: !245, size: 64, offset: 4224)
!264 = !DIDerivedType(tag: DW_TAG_member, name: "surfaceAlignment", scope: !212, file: !6, line: 1300, baseType: !131, size: 64, offset: 4288)
!265 = !DIDerivedType(tag: DW_TAG_member, name: "concurrentKernels", scope: !212, file: !6, line: 1301, baseType: !97, size: 32, offset: 4352)
!266 = !DIDerivedType(tag: DW_TAG_member, name: "ECCEnabled", scope: !212, file: !6, line: 1302, baseType: !97, size: 32, offset: 4384)
!267 = !DIDerivedType(tag: DW_TAG_member, name: "pciBusID", scope: !212, file: !6, line: 1303, baseType: !97, size: 32, offset: 4416)
!268 = !DIDerivedType(tag: DW_TAG_member, name: "pciDeviceID", scope: !212, file: !6, line: 1304, baseType: !97, size: 32, offset: 4448)
!269 = !DIDerivedType(tag: DW_TAG_member, name: "pciDomainID", scope: !212, file: !6, line: 1305, baseType: !97, size: 32, offset: 4480)
!270 = !DIDerivedType(tag: DW_TAG_member, name: "tccDriver", scope: !212, file: !6, line: 1306, baseType: !97, size: 32, offset: 4512)
!271 = !DIDerivedType(tag: DW_TAG_member, name: "asyncEngineCount", scope: !212, file: !6, line: 1307, baseType: !97, size: 32, offset: 4544)
!272 = !DIDerivedType(tag: DW_TAG_member, name: "unifiedAddressing", scope: !212, file: !6, line: 1308, baseType: !97, size: 32, offset: 4576)
!273 = !DIDerivedType(tag: DW_TAG_member, name: "memoryClockRate", scope: !212, file: !6, line: 1309, baseType: !97, size: 32, offset: 4608)
!274 = !DIDerivedType(tag: DW_TAG_member, name: "memoryBusWidth", scope: !212, file: !6, line: 1310, baseType: !97, size: 32, offset: 4640)
!275 = !DIDerivedType(tag: DW_TAG_member, name: "l2CacheSize", scope: !212, file: !6, line: 1311, baseType: !97, size: 32, offset: 4672)
!276 = !DIDerivedType(tag: DW_TAG_member, name: "maxThreadsPerMultiProcessor", scope: !212, file: !6, line: 1312, baseType: !97, size: 32, offset: 4704)
!277 = !DIDerivedType(tag: DW_TAG_member, name: "streamPrioritiesSupported", scope: !212, file: !6, line: 1313, baseType: !97, size: 32, offset: 4736)
!278 = !DIDerivedType(tag: DW_TAG_member, name: "globalL1CacheSupported", scope: !212, file: !6, line: 1314, baseType: !97, size: 32, offset: 4768)
!279 = !DIDerivedType(tag: DW_TAG_member, name: "localL1CacheSupported", scope: !212, file: !6, line: 1315, baseType: !97, size: 32, offset: 4800)
!280 = !DIDerivedType(tag: DW_TAG_member, name: "sharedMemPerMultiprocessor", scope: !212, file: !6, line: 1316, baseType: !131, size: 64, offset: 4864)
!281 = !DIDerivedType(tag: DW_TAG_member, name: "regsPerMultiprocessor", scope: !212, file: !6, line: 1317, baseType: !97, size: 32, offset: 4928)
!282 = !DIDerivedType(tag: DW_TAG_member, name: "managedMemory", scope: !212, file: !6, line: 1318, baseType: !97, size: 32, offset: 4960)
!283 = !DIDerivedType(tag: DW_TAG_member, name: "isMultiGpuBoard", scope: !212, file: !6, line: 1319, baseType: !97, size: 32, offset: 4992)
!284 = !DIDerivedType(tag: DW_TAG_member, name: "multiGpuBoardGroupID", scope: !212, file: !6, line: 1320, baseType: !97, size: 32, offset: 5024)
!285 = !DIGlobalVariableExpression(var: !286, expr: !DIExpression())
!286 = distinct !DIGlobalVariable(name: "sums", linkageName: "_ZL4sums", scope: !2, file: !3, line: 164, type: !98, isLocal: true, isDefinition: true)
!287 = !DIGlobalVariableExpression(var: !288, expr: !DIExpression())
!288 = distinct !DIGlobalVariable(name: "twiddle", linkageName: "_ZL7twiddle", scope: !2, file: !3, line: 165, type: !106, isLocal: true, isDefinition: true)
!289 = !DIGlobalVariableExpression(var: !290, expr: !DIExpression())
!290 = distinct !DIGlobalVariable(name: "u", linkageName: "_ZL1u", scope: !2, file: !3, line: 166, type: !98, isLocal: true, isDefinition: true)
!291 = !DIGlobalVariableExpression(var: !292, expr: !DIExpression())
!292 = distinct !DIGlobalVariable(name: "u0", linkageName: "_ZL2u0", scope: !2, file: !3, line: 167, type: !98, isLocal: true, isDefinition: true)
!293 = !DIGlobalVariableExpression(var: !294, expr: !DIExpression())
!294 = distinct !DIGlobalVariable(name: "u1", linkageName: "_ZL2u1", scope: !2, file: !3, line: 168, type: !98, isLocal: true, isDefinition: true)
!295 = !DIGlobalVariableExpression(var: !296, expr: !DIExpression())
!296 = distinct !DIGlobalVariable(name: "dims", linkageName: "_ZL4dims", scope: !2, file: !3, line: 169, type: !107, isLocal: true, isDefinition: true)
!297 = !DIGlobalVariableExpression(var: !298, expr: !DIExpression())
!298 = distinct !DIGlobalVariable(name: "niter", linkageName: "_ZL5niter", scope: !2, file: !3, line: 171, type: !97, isLocal: true, isDefinition: true)
!299 = !{!300, !306, !311, !313, !315, !317, !319, !323, !325, !327, !329, !331, !333, !335, !337, !339, !341, !343, !345, !347, !349, !351, !355, !357, !359, !361, !365, !369, !371, !373, !378, !382, !384, !386, !388, !390, !392, !394, !396, !398, !403, !407, !409, !414, !418, !420, !422, !424, !426, !428, !432, !434, !436, !441, !447, !451, !453, !455, !457, !459, !463, !465, !467, !471, !473, !475, !477, !479, !481, !483, !485, !487, !489, !493, !499, !501, !503, !507, !509, !511, !513, !515, !517, !519, !521, !525, !529, !531, !533, !537, !539, !541, !543, !545, !547, !549, !553, !559, !563, !568, !570, !574, !578, !588, !592, !596, !600, !604, !608, !610, !614, !618, !622, !630, !634, !638, !642, !646, !650, !656, !660, !664, !666, !674, !678, !685, !687, !689, !693, !697, !701, !706, !710, !715, !716, !717, !718, !720, !721, !722, !723, !724, !725, !726, !728, !729, !730, !731, !732, !736, !737, !738, !739, !740, !741, !742, !743, !744, !745, !746, !747, !748, !749, !750, !751, !752, !753, !754, !755, !756, !757, !758, !759, !760, !764, !766, !768, !770, !772, !774, !776, !778, !781, !783, !785, !787, !789, !791, !793, !795, !797, !799, !801, !803, !805, !807, !809, !811, !813, !815, !817, !819, !821, !823, !825, !827, !829, !831, !833, !835, !837, !839, !841, !843, !845, !847, !849, !851, !853, !855, !857, !859, !861, !863, !865, !867, !869, !871, !873, !879, !885, !890, !894, !896, !898, !900, !902, !909, !913, !917, !921, !925, !929, !934, !938, !940, !944, !950, !954, !959, !961, !963, !967, !971, !975, !977, !979, !981, !983, !987, !989, !991, !995, !999, !1003, !1007, !1011, !1013, !1015, !1021, !1025, !1029, !1033, !1035, !1037, !1041, !1045, !1046, !1047, !1048, !1049}
!300 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !302, file: !303, line: 223)
!301 = !DINamespace(name: "std", scope: null)
!302 = !DISubprogram(name: "abs", linkageName: "_ZL3absi", scope: !303, file: !303, line: 53, type: !304, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!303 = !DIFile(filename: "/u/NAS_SCRATCH/ah7226/tulip/llvm-install-tulip/lib/clang/9.0.0/include/__clang_cuda_math_forward_declares.h", directory: "")
!304 = !DISubroutineType(types: !305)
!305 = !{!97, !97}
!306 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !307, file: !303, line: 224)
!307 = !DISubprogram(name: "acos", linkageName: "_ZL4acosf", scope: !303, file: !303, line: 55, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!308 = !DISubroutineType(types: !309)
!309 = !{!310, !310}
!310 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!311 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !312, file: !303, line: 225)
!312 = !DISubprogram(name: "acosh", linkageName: "_ZL5acoshf", scope: !303, file: !303, line: 57, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!313 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !314, file: !303, line: 226)
!314 = !DISubprogram(name: "asin", linkageName: "_ZL4asinf", scope: !303, file: !303, line: 59, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!315 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !316, file: !303, line: 227)
!316 = !DISubprogram(name: "asinh", linkageName: "_ZL5asinhf", scope: !303, file: !303, line: 61, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!317 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !318, file: !303, line: 228)
!318 = !DISubprogram(name: "atan", linkageName: "_ZL4atanf", scope: !303, file: !303, line: 65, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!319 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !320, file: !303, line: 229)
!320 = !DISubprogram(name: "atan2", linkageName: "_ZL5atan2ff", scope: !303, file: !303, line: 63, type: !321, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!321 = !DISubroutineType(types: !322)
!322 = !{!310, !310, !310}
!323 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !324, file: !303, line: 230)
!324 = !DISubprogram(name: "atanh", linkageName: "_ZL5atanhf", scope: !303, file: !303, line: 67, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!325 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !326, file: !303, line: 231)
!326 = !DISubprogram(name: "cbrt", linkageName: "_ZL4cbrtf", scope: !303, file: !303, line: 69, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!327 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !328, file: !303, line: 232)
!328 = !DISubprogram(name: "ceil", linkageName: "_ZL4ceilf", scope: !303, file: !303, line: 71, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!329 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !330, file: !303, line: 233)
!330 = !DISubprogram(name: "copysign", linkageName: "_ZL8copysignff", scope: !303, file: !303, line: 73, type: !321, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!331 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !332, file: !303, line: 234)
!332 = !DISubprogram(name: "cos", linkageName: "_ZL3cosf", scope: !303, file: !303, line: 75, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!333 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !334, file: !303, line: 235)
!334 = !DISubprogram(name: "cosh", linkageName: "_ZL4coshf", scope: !303, file: !303, line: 77, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!335 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !336, file: !303, line: 236)
!336 = !DISubprogram(name: "erf", linkageName: "_ZL3erff", scope: !303, file: !303, line: 81, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!337 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !338, file: !303, line: 237)
!338 = !DISubprogram(name: "erfc", linkageName: "_ZL4erfcf", scope: !303, file: !303, line: 79, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!339 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !340, file: !303, line: 238)
!340 = !DISubprogram(name: "exp", linkageName: "_ZL3expf", scope: !303, file: !303, line: 85, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!341 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !342, file: !303, line: 239)
!342 = !DISubprogram(name: "exp2", linkageName: "_ZL4exp2f", scope: !303, file: !303, line: 83, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!343 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !344, file: !303, line: 240)
!344 = !DISubprogram(name: "expm1", linkageName: "_ZL5expm1f", scope: !303, file: !303, line: 87, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!345 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !346, file: !303, line: 241)
!346 = !DISubprogram(name: "fabs", linkageName: "_ZL4fabsf", scope: !303, file: !303, line: 89, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!347 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !348, file: !303, line: 242)
!348 = !DISubprogram(name: "fdim", linkageName: "_ZL4fdimff", scope: !303, file: !303, line: 91, type: !321, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!349 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !350, file: !303, line: 243)
!350 = !DISubprogram(name: "floor", linkageName: "_ZL5floorf", scope: !303, file: !303, line: 93, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!351 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !352, file: !303, line: 244)
!352 = !DISubprogram(name: "fma", linkageName: "_ZL3fmafff", scope: !303, file: !303, line: 95, type: !353, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!353 = !DISubroutineType(types: !354)
!354 = !{!310, !310, !310, !310}
!355 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !356, file: !303, line: 245)
!356 = !DISubprogram(name: "fmax", linkageName: "_ZL4fmaxff", scope: !303, file: !303, line: 97, type: !321, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!357 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !358, file: !303, line: 246)
!358 = !DISubprogram(name: "fmin", linkageName: "_ZL4fminff", scope: !303, file: !303, line: 99, type: !321, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!359 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !360, file: !303, line: 247)
!360 = !DISubprogram(name: "fmod", linkageName: "_ZL4fmodff", scope: !303, file: !303, line: 101, type: !321, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!361 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !362, file: !303, line: 248)
!362 = !DISubprogram(name: "fpclassify", linkageName: "_ZL10fpclassifyf", scope: !303, file: !303, line: 103, type: !363, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!363 = !DISubroutineType(types: !364)
!364 = !{!97, !310}
!365 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !366, file: !303, line: 249)
!366 = !DISubprogram(name: "frexp", linkageName: "_ZL5frexpfPi", scope: !303, file: !303, line: 105, type: !367, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!367 = !DISubroutineType(types: !368)
!368 = !{!310, !310, !107}
!369 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !370, file: !303, line: 250)
!370 = !DISubprogram(name: "hypot", linkageName: "_ZL5hypotff", scope: !303, file: !303, line: 107, type: !321, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!371 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !372, file: !303, line: 251)
!372 = !DISubprogram(name: "ilogb", linkageName: "_ZL5ilogbf", scope: !303, file: !303, line: 109, type: !363, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!373 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !374, file: !303, line: 252)
!374 = !DISubprogram(name: "isfinite", linkageName: "_ZL8isfinitef", scope: !303, file: !303, line: 114, type: !375, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!375 = !DISubroutineType(types: !376)
!376 = !{!377, !310}
!377 = !DIBasicType(name: "bool", size: 8, encoding: DW_ATE_boolean)
!378 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !379, file: !303, line: 253)
!379 = !DISubprogram(name: "isgreater", linkageName: "_ZL9isgreaterff", scope: !303, file: !303, line: 118, type: !380, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!380 = !DISubroutineType(types: !381)
!381 = !{!377, !310, !310}
!382 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !383, file: !303, line: 254)
!383 = !DISubprogram(name: "isgreaterequal", linkageName: "_ZL14isgreaterequalff", scope: !303, file: !303, line: 117, type: !380, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!384 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !385, file: !303, line: 255)
!385 = !DISubprogram(name: "isinf", linkageName: "_ZL5isinff", scope: !303, file: !303, line: 123, type: !375, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!386 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !387, file: !303, line: 256)
!387 = !DISubprogram(name: "isless", linkageName: "_ZL6islessff", scope: !303, file: !303, line: 127, type: !380, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!388 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !389, file: !303, line: 257)
!389 = !DISubprogram(name: "islessequal", linkageName: "_ZL11islessequalff", scope: !303, file: !303, line: 126, type: !380, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!390 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !391, file: !303, line: 258)
!391 = !DISubprogram(name: "islessgreater", linkageName: "_ZL13islessgreaterff", scope: !303, file: !303, line: 129, type: !380, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!392 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !393, file: !303, line: 259)
!393 = !DISubprogram(name: "isnan", linkageName: "_ZL5isnanf", scope: !303, file: !303, line: 134, type: !375, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!394 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !395, file: !303, line: 260)
!395 = !DISubprogram(name: "isnormal", linkageName: "_ZL8isnormalf", scope: !303, file: !303, line: 136, type: !375, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!396 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !397, file: !303, line: 261)
!397 = !DISubprogram(name: "isunordered", linkageName: "_ZL11isunorderedff", scope: !303, file: !303, line: 138, type: !380, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!398 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !399, file: !303, line: 262)
!399 = !DISubprogram(name: "labs", linkageName: "_ZL4labsl", scope: !303, file: !303, line: 139, type: !400, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!400 = !DISubroutineType(types: !401)
!401 = !{!402, !402}
!402 = !DIBasicType(name: "long int", size: 64, encoding: DW_ATE_signed)
!403 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !404, file: !303, line: 263)
!404 = !DISubprogram(name: "ldexp", linkageName: "_ZL5ldexpfi", scope: !303, file: !303, line: 141, type: !405, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!405 = !DISubroutineType(types: !406)
!406 = !{!310, !310, !97}
!407 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !408, file: !303, line: 264)
!408 = !DISubprogram(name: "lgamma", linkageName: "_ZL6lgammaf", scope: !303, file: !303, line: 143, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!409 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !410, file: !303, line: 265)
!410 = !DISubprogram(name: "llabs", linkageName: "_ZL5llabsx", scope: !303, file: !303, line: 144, type: !411, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!411 = !DISubroutineType(types: !412)
!412 = !{!413, !413}
!413 = !DIBasicType(name: "long long int", size: 64, encoding: DW_ATE_signed)
!414 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !415, file: !303, line: 266)
!415 = !DISubprogram(name: "llrint", linkageName: "_ZL6llrintf", scope: !303, file: !303, line: 146, type: !416, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!416 = !DISubroutineType(types: !417)
!417 = !{!413, !310}
!418 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !419, file: !303, line: 267)
!419 = !DISubprogram(name: "log", linkageName: "_ZL3logf", scope: !303, file: !303, line: 159, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!420 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !421, file: !303, line: 268)
!421 = !DISubprogram(name: "log10", linkageName: "_ZL5log10f", scope: !303, file: !303, line: 148, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!422 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !423, file: !303, line: 269)
!423 = !DISubprogram(name: "log1p", linkageName: "_ZL5log1pf", scope: !303, file: !303, line: 150, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!424 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !425, file: !303, line: 270)
!425 = !DISubprogram(name: "log2", linkageName: "_ZL4log2f", scope: !303, file: !303, line: 152, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!426 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !427, file: !303, line: 271)
!427 = !DISubprogram(name: "logb", linkageName: "_ZL4logbf", scope: !303, file: !303, line: 154, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!428 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !429, file: !303, line: 272)
!429 = !DISubprogram(name: "lrint", linkageName: "_ZL5lrintf", scope: !303, file: !303, line: 161, type: !430, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!430 = !DISubroutineType(types: !431)
!431 = !{!402, !310}
!432 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !433, file: !303, line: 273)
!433 = !DISubprogram(name: "lround", linkageName: "_ZL6lroundf", scope: !303, file: !303, line: 163, type: !430, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!434 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !435, file: !303, line: 274)
!435 = !DISubprogram(name: "llround", linkageName: "_ZL7llroundf", scope: !303, file: !303, line: 164, type: !416, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!436 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !437, file: !303, line: 275)
!437 = !DISubprogram(name: "modf", linkageName: "_ZL4modffPf", scope: !303, file: !303, line: 166, type: !438, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!438 = !DISubroutineType(types: !439)
!439 = !{!310, !310, !440}
!440 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !310, size: 64)
!441 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !442, file: !303, line: 276)
!442 = !DISubprogram(name: "nan", linkageName: "_ZL3nanPKc", scope: !303, file: !303, line: 167, type: !443, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!443 = !DISubroutineType(types: !444)
!444 = !{!104, !445}
!445 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !446, size: 64)
!446 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !109)
!447 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !448, file: !303, line: 277)
!448 = !DISubprogram(name: "nanf", linkageName: "_ZL4nanfPKc", scope: !303, file: !303, line: 168, type: !449, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!449 = !DISubroutineType(types: !450)
!450 = !{!310, !445}
!451 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !452, file: !303, line: 278)
!452 = !DISubprogram(name: "nearbyint", linkageName: "_ZL9nearbyintf", scope: !303, file: !303, line: 170, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!453 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !454, file: !303, line: 279)
!454 = !DISubprogram(name: "nextafter", linkageName: "_ZL9nextafterff", scope: !303, file: !303, line: 172, type: !321, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!455 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !456, file: !303, line: 280)
!456 = !DISubprogram(name: "pow", linkageName: "_ZL3powfi", scope: !303, file: !303, line: 176, type: !405, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!457 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !458, file: !303, line: 281)
!458 = !DISubprogram(name: "remainder", linkageName: "_ZL9remainderff", scope: !303, file: !303, line: 178, type: !321, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!459 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !460, file: !303, line: 282)
!460 = !DISubprogram(name: "remquo", linkageName: "_ZL6remquoffPi", scope: !303, file: !303, line: 180, type: !461, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!461 = !DISubroutineType(types: !462)
!462 = !{!310, !310, !310, !107}
!463 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !464, file: !303, line: 283)
!464 = !DISubprogram(name: "rint", linkageName: "_ZL4rintf", scope: !303, file: !303, line: 182, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!465 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !466, file: !303, line: 284)
!466 = !DISubprogram(name: "round", linkageName: "_ZL5roundf", scope: !303, file: !303, line: 184, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!467 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !468, file: !303, line: 285)
!468 = !DISubprogram(name: "scalbln", linkageName: "_ZL7scalblnfl", scope: !303, file: !303, line: 186, type: !469, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!469 = !DISubroutineType(types: !470)
!470 = !{!310, !310, !402}
!471 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !472, file: !303, line: 286)
!472 = !DISubprogram(name: "scalbn", linkageName: "_ZL6scalbnfi", scope: !303, file: !303, line: 188, type: !405, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!473 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !474, file: !303, line: 287)
!474 = !DISubprogram(name: "signbit", linkageName: "_ZL7signbitf", scope: !303, file: !303, line: 190, type: !375, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!475 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !476, file: !303, line: 288)
!476 = !DISubprogram(name: "sin", linkageName: "_ZL3sinf", scope: !303, file: !303, line: 192, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!477 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !478, file: !303, line: 289)
!478 = !DISubprogram(name: "sinh", linkageName: "_ZL4sinhf", scope: !303, file: !303, line: 194, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!479 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !480, file: !303, line: 290)
!480 = !DISubprogram(name: "sqrt", linkageName: "_ZL4sqrtf", scope: !303, file: !303, line: 196, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!481 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !482, file: !303, line: 291)
!482 = !DISubprogram(name: "tan", linkageName: "_ZL3tanf", scope: !303, file: !303, line: 198, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!483 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !484, file: !303, line: 292)
!484 = !DISubprogram(name: "tanh", linkageName: "_ZL4tanhf", scope: !303, file: !303, line: 200, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!485 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !486, file: !303, line: 293)
!486 = !DISubprogram(name: "tgamma", linkageName: "_ZL6tgammaf", scope: !303, file: !303, line: 202, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!487 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !488, file: !303, line: 294)
!488 = !DISubprogram(name: "trunc", linkageName: "_ZL5truncf", scope: !303, file: !303, line: 204, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!489 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !490, file: !492, line: 52)
!490 = !DISubprogram(name: "abs", scope: !491, file: !491, line: 848, type: !304, flags: DIFlagPrototyped, spFlags: 0)
!491 = !DIFile(filename: "/usr/include/stdlib.h", directory: "")
!492 = !DIFile(filename: "/usr/lib/gcc/x86_64-linux-gnu/11/../../../../include/c++/11/bits/std_abs.h", directory: "")
!493 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !494, file: !498, line: 83)
!494 = !DISubprogram(name: "acos", scope: !495, file: !495, line: 53, type: !496, flags: DIFlagPrototyped, spFlags: 0)
!495 = !DIFile(filename: "/usr/include/x86_64-linux-gnu/bits/mathcalls.h", directory: "")
!496 = !DISubroutineType(types: !497)
!497 = !{!104, !104}
!498 = !DIFile(filename: "/usr/lib/gcc/x86_64-linux-gnu/11/../../../../include/c++/11/cmath", directory: "")
!499 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !500, file: !498, line: 102)
!500 = !DISubprogram(name: "asin", scope: !495, file: !495, line: 55, type: !496, flags: DIFlagPrototyped, spFlags: 0)
!501 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !502, file: !498, line: 121)
!502 = !DISubprogram(name: "atan", scope: !495, file: !495, line: 57, type: !496, flags: DIFlagPrototyped, spFlags: 0)
!503 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !504, file: !498, line: 140)
!504 = !DISubprogram(name: "atan2", scope: !495, file: !495, line: 59, type: !505, flags: DIFlagPrototyped, spFlags: 0)
!505 = !DISubroutineType(types: !506)
!506 = !{!104, !104, !104}
!507 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !508, file: !498, line: 161)
!508 = !DISubprogram(name: "ceil", scope: !495, file: !495, line: 159, type: !496, flags: DIFlagPrototyped, spFlags: 0)
!509 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !510, file: !498, line: 180)
!510 = !DISubprogram(name: "cos", scope: !495, file: !495, line: 62, type: !496, flags: DIFlagPrototyped, spFlags: 0)
!511 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !512, file: !498, line: 199)
!512 = !DISubprogram(name: "cosh", scope: !495, file: !495, line: 71, type: !496, flags: DIFlagPrototyped, spFlags: 0)
!513 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !514, file: !498, line: 218)
!514 = !DISubprogram(name: "exp", scope: !495, file: !495, line: 95, type: !496, flags: DIFlagPrototyped, spFlags: 0)
!515 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !516, file: !498, line: 237)
!516 = !DISubprogram(name: "fabs", scope: !495, file: !495, line: 162, type: !496, flags: DIFlagPrototyped, spFlags: 0)
!517 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !518, file: !498, line: 256)
!518 = !DISubprogram(name: "floor", scope: !495, file: !495, line: 165, type: !496, flags: DIFlagPrototyped, spFlags: 0)
!519 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !520, file: !498, line: 275)
!520 = !DISubprogram(name: "fmod", scope: !495, file: !495, line: 168, type: !505, flags: DIFlagPrototyped, spFlags: 0)
!521 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !522, file: !498, line: 296)
!522 = !DISubprogram(name: "frexp", scope: !495, file: !495, line: 98, type: !523, flags: DIFlagPrototyped, spFlags: 0)
!523 = !DISubroutineType(types: !524)
!524 = !{!104, !104, !107}
!525 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !526, file: !498, line: 315)
!526 = !DISubprogram(name: "ldexp", scope: !495, file: !495, line: 101, type: !527, flags: DIFlagPrototyped, spFlags: 0)
!527 = !DISubroutineType(types: !528)
!528 = !{!104, !104, !97}
!529 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !530, file: !498, line: 334)
!530 = !DISubprogram(name: "log", scope: !495, file: !495, line: 104, type: !496, flags: DIFlagPrototyped, spFlags: 0)
!531 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !532, file: !498, line: 353)
!532 = !DISubprogram(name: "log10", scope: !495, file: !495, line: 107, type: !496, flags: DIFlagPrototyped, spFlags: 0)
!533 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !534, file: !498, line: 372)
!534 = !DISubprogram(name: "modf", scope: !495, file: !495, line: 110, type: !535, flags: DIFlagPrototyped, spFlags: 0)
!535 = !DISubroutineType(types: !536)
!536 = !{!104, !104, !106}
!537 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !538, file: !498, line: 384)
!538 = !DISubprogram(name: "pow", scope: !495, file: !495, line: 140, type: !505, flags: DIFlagPrototyped, spFlags: 0)
!539 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !540, file: !498, line: 421)
!540 = !DISubprogram(name: "sin", scope: !495, file: !495, line: 64, type: !496, flags: DIFlagPrototyped, spFlags: 0)
!541 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !542, file: !498, line: 440)
!542 = !DISubprogram(name: "sinh", scope: !495, file: !495, line: 73, type: !496, flags: DIFlagPrototyped, spFlags: 0)
!543 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !544, file: !498, line: 459)
!544 = !DISubprogram(name: "sqrt", scope: !495, file: !495, line: 143, type: !496, flags: DIFlagPrototyped, spFlags: 0)
!545 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !546, file: !498, line: 478)
!546 = !DISubprogram(name: "tan", scope: !495, file: !495, line: 66, type: !496, flags: DIFlagPrototyped, spFlags: 0)
!547 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !548, file: !498, line: 497)
!548 = !DISubprogram(name: "tanh", scope: !495, file: !495, line: 75, type: !496, flags: DIFlagPrototyped, spFlags: 0)
!549 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !550, file: !552, line: 127)
!550 = !DIDerivedType(tag: DW_TAG_typedef, name: "div_t", file: !491, line: 63, baseType: !551)
!551 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !491, line: 59, flags: DIFlagFwdDecl, identifier: "_ZTS5div_t")
!552 = !DIFile(filename: "/usr/lib/gcc/x86_64-linux-gnu/11/../../../../include/c++/11/cstdlib", directory: "")
!553 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !554, file: !552, line: 128)
!554 = !DIDerivedType(tag: DW_TAG_typedef, name: "ldiv_t", file: !491, line: 71, baseType: !555)
!555 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !491, line: 67, size: 128, flags: DIFlagTypePassByValue, elements: !556, identifier: "_ZTS6ldiv_t")
!556 = !{!557, !558}
!557 = !DIDerivedType(tag: DW_TAG_member, name: "quot", scope: !555, file: !491, line: 69, baseType: !402, size: 64)
!558 = !DIDerivedType(tag: DW_TAG_member, name: "rem", scope: !555, file: !491, line: 70, baseType: !402, size: 64, offset: 64)
!559 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !560, file: !552, line: 130)
!560 = !DISubprogram(name: "abort", scope: !491, file: !491, line: 598, type: !561, flags: DIFlagPrototyped | DIFlagNoReturn, spFlags: 0)
!561 = !DISubroutineType(types: !562)
!562 = !{null}
!563 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !564, file: !552, line: 134)
!564 = !DISubprogram(name: "atexit", scope: !491, file: !491, line: 602, type: !565, flags: DIFlagPrototyped, spFlags: 0)
!565 = !DISubroutineType(types: !566)
!566 = !{!97, !567}
!567 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !561, size: 64)
!568 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !569, file: !552, line: 140)
!569 = !DISubprogram(name: "atof", scope: !491, file: !491, line: 102, type: !443, flags: DIFlagPrototyped, spFlags: 0)
!570 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !571, file: !552, line: 141)
!571 = !DISubprogram(name: "atoi", scope: !491, file: !491, line: 105, type: !572, flags: DIFlagPrototyped, spFlags: 0)
!572 = !DISubroutineType(types: !573)
!573 = !{!97, !445}
!574 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !575, file: !552, line: 142)
!575 = !DISubprogram(name: "atol", scope: !491, file: !491, line: 108, type: !576, flags: DIFlagPrototyped, spFlags: 0)
!576 = !DISubroutineType(types: !577)
!577 = !{!402, !445}
!578 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !579, file: !552, line: 143)
!579 = !DISubprogram(name: "bsearch", scope: !491, file: !491, line: 828, type: !580, flags: DIFlagPrototyped, spFlags: 0)
!580 = !DISubroutineType(types: !581)
!581 = !{!111, !582, !582, !131, !131, !584}
!582 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !583, size: 64)
!583 = !DIDerivedType(tag: DW_TAG_const_type, baseType: null)
!584 = !DIDerivedType(tag: DW_TAG_typedef, name: "__compar_fn_t", file: !491, line: 816, baseType: !585)
!585 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !586, size: 64)
!586 = !DISubroutineType(types: !587)
!587 = !{!97, !582, !582}
!588 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !589, file: !552, line: 144)
!589 = !DISubprogram(name: "calloc", scope: !491, file: !491, line: 543, type: !590, flags: DIFlagPrototyped, spFlags: 0)
!590 = !DISubroutineType(types: !591)
!591 = !{!111, !131, !131}
!592 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !593, file: !552, line: 145)
!593 = !DISubprogram(name: "div", scope: !491, file: !491, line: 860, type: !594, flags: DIFlagPrototyped, spFlags: 0)
!594 = !DISubroutineType(types: !595)
!595 = !{!550, !97, !97}
!596 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !597, file: !552, line: 146)
!597 = !DISubprogram(name: "exit", scope: !491, file: !491, line: 624, type: !598, flags: DIFlagPrototyped | DIFlagNoReturn, spFlags: 0)
!598 = !DISubroutineType(types: !599)
!599 = !{null, !97}
!600 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !601, file: !552, line: 147)
!601 = !DISubprogram(name: "free", scope: !491, file: !491, line: 555, type: !602, flags: DIFlagPrototyped, spFlags: 0)
!602 = !DISubroutineType(types: !603)
!603 = !{null, !111}
!604 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !605, file: !552, line: 148)
!605 = !DISubprogram(name: "getenv", scope: !491, file: !491, line: 641, type: !606, flags: DIFlagPrototyped, spFlags: 0)
!606 = !DISubroutineType(types: !607)
!607 = !{!108, !445}
!608 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !609, file: !552, line: 149)
!609 = !DISubprogram(name: "labs", scope: !491, file: !491, line: 849, type: !400, flags: DIFlagPrototyped, spFlags: 0)
!610 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !611, file: !552, line: 150)
!611 = !DISubprogram(name: "ldiv", scope: !491, file: !491, line: 862, type: !612, flags: DIFlagPrototyped, spFlags: 0)
!612 = !DISubroutineType(types: !613)
!613 = !{!554, !402, !402}
!614 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !615, file: !552, line: 151)
!615 = !DISubprogram(name: "malloc", scope: !491, file: !491, line: 540, type: !616, flags: DIFlagPrototyped, spFlags: 0)
!616 = !DISubroutineType(types: !617)
!617 = !{!111, !131}
!618 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !619, file: !552, line: 153)
!619 = !DISubprogram(name: "mblen", scope: !491, file: !491, line: 930, type: !620, flags: DIFlagPrototyped, spFlags: 0)
!620 = !DISubroutineType(types: !621)
!621 = !{!97, !445, !131}
!622 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !623, file: !552, line: 154)
!623 = !DISubprogram(name: "mbstowcs", scope: !491, file: !491, line: 941, type: !624, flags: DIFlagPrototyped, spFlags: 0)
!624 = !DISubroutineType(types: !625)
!625 = !{!131, !626, !629, !131}
!626 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !627)
!627 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !628, size: 64)
!628 = !DIBasicType(name: "wchar_t", size: 32, encoding: DW_ATE_signed)
!629 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !445)
!630 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !631, file: !552, line: 155)
!631 = !DISubprogram(name: "mbtowc", scope: !491, file: !491, line: 933, type: !632, flags: DIFlagPrototyped, spFlags: 0)
!632 = !DISubroutineType(types: !633)
!633 = !{!97, !626, !629, !131}
!634 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !635, file: !552, line: 157)
!635 = !DISubprogram(name: "qsort", scope: !491, file: !491, line: 838, type: !636, flags: DIFlagPrototyped, spFlags: 0)
!636 = !DISubroutineType(types: !637)
!637 = !{null, !111, !131, !131, !584}
!638 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !639, file: !552, line: 163)
!639 = !DISubprogram(name: "rand", scope: !491, file: !491, line: 454, type: !640, flags: DIFlagPrototyped, spFlags: 0)
!640 = !DISubroutineType(types: !641)
!641 = !{!97}
!642 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !643, file: !552, line: 164)
!643 = !DISubprogram(name: "realloc", scope: !491, file: !491, line: 551, type: !644, flags: DIFlagPrototyped, spFlags: 0)
!644 = !DISubroutineType(types: !645)
!645 = !{!111, !111, !131}
!646 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !647, file: !552, line: 165)
!647 = !DISubprogram(name: "srand", scope: !491, file: !491, line: 456, type: !648, flags: DIFlagPrototyped, spFlags: 0)
!648 = !DISubroutineType(types: !649)
!649 = !{null, !7}
!650 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !651, file: !552, line: 166)
!651 = !DISubprogram(name: "strtod", scope: !491, file: !491, line: 118, type: !652, flags: DIFlagPrototyped, spFlags: 0)
!652 = !DISubroutineType(types: !653)
!653 = !{!104, !629, !654}
!654 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !655)
!655 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !108, size: 64)
!656 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !657, file: !552, line: 167)
!657 = !DISubprogram(name: "strtol", scope: !491, file: !491, line: 177, type: !658, flags: DIFlagPrototyped, spFlags: 0)
!658 = !DISubroutineType(types: !659)
!659 = !{!402, !629, !654, !97}
!660 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !661, file: !552, line: 168)
!661 = !DISubprogram(name: "strtoul", scope: !491, file: !491, line: 181, type: !662, flags: DIFlagPrototyped, spFlags: 0)
!662 = !DISubroutineType(types: !663)
!663 = !{!133, !629, !654, !97}
!664 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !665, file: !552, line: 169)
!665 = !DISubprogram(name: "system", scope: !491, file: !491, line: 791, type: !572, flags: DIFlagPrototyped, spFlags: 0)
!666 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !667, file: !552, line: 171)
!667 = !DISubprogram(name: "wcstombs", scope: !491, file: !491, line: 945, type: !668, flags: DIFlagPrototyped, spFlags: 0)
!668 = !DISubroutineType(types: !669)
!669 = !{!131, !670, !671, !131}
!670 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !108)
!671 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !672)
!672 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !673, size: 64)
!673 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !628)
!674 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !675, file: !552, line: 172)
!675 = !DISubprogram(name: "wctomb", scope: !491, file: !491, line: 937, type: !676, flags: DIFlagPrototyped, spFlags: 0)
!676 = !DISubroutineType(types: !677)
!677 = !{!97, !108, !628}
!678 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !679, entity: !680, file: !552, line: 200)
!679 = !DINamespace(name: "__gnu_cxx", scope: null)
!680 = !DIDerivedType(tag: DW_TAG_typedef, name: "lldiv_t", file: !491, line: 81, baseType: !681)
!681 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !491, line: 77, size: 128, flags: DIFlagTypePassByValue, elements: !682, identifier: "_ZTS7lldiv_t")
!682 = !{!683, !684}
!683 = !DIDerivedType(tag: DW_TAG_member, name: "quot", scope: !681, file: !491, line: 79, baseType: !413, size: 64)
!684 = !DIDerivedType(tag: DW_TAG_member, name: "rem", scope: !681, file: !491, line: 80, baseType: !413, size: 64, offset: 64)
!685 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !679, entity: !686, file: !552, line: 206)
!686 = !DISubprogram(name: "_Exit", scope: !491, file: !491, line: 636, type: !598, flags: DIFlagPrototyped | DIFlagNoReturn, spFlags: 0)
!687 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !679, entity: !688, file: !552, line: 210)
!688 = !DISubprogram(name: "llabs", scope: !491, file: !491, line: 852, type: !411, flags: DIFlagPrototyped, spFlags: 0)
!689 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !679, entity: !690, file: !552, line: 216)
!690 = !DISubprogram(name: "lldiv", scope: !491, file: !491, line: 866, type: !691, flags: DIFlagPrototyped, spFlags: 0)
!691 = !DISubroutineType(types: !692)
!692 = !{!680, !413, !413}
!693 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !679, entity: !694, file: !552, line: 227)
!694 = !DISubprogram(name: "atoll", scope: !491, file: !491, line: 113, type: !695, flags: DIFlagPrototyped, spFlags: 0)
!695 = !DISubroutineType(types: !696)
!696 = !{!413, !445}
!697 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !679, entity: !698, file: !552, line: 228)
!698 = !DISubprogram(name: "strtoll", scope: !491, file: !491, line: 201, type: !699, flags: DIFlagPrototyped, spFlags: 0)
!699 = !DISubroutineType(types: !700)
!700 = !{!413, !629, !654, !97}
!701 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !679, entity: !702, file: !552, line: 229)
!702 = !DISubprogram(name: "strtoull", scope: !491, file: !491, line: 206, type: !703, flags: DIFlagPrototyped, spFlags: 0)
!703 = !DISubroutineType(types: !704)
!704 = !{!705, !629, !654, !97}
!705 = !DIBasicType(name: "long long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!706 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !679, entity: !707, file: !552, line: 231)
!707 = !DISubprogram(name: "strtof", scope: !491, file: !491, line: 124, type: !708, flags: DIFlagPrototyped, spFlags: 0)
!708 = !DISubroutineType(types: !709)
!709 = !{!310, !629, !654}
!710 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !679, entity: !711, file: !552, line: 232)
!711 = !DISubprogram(name: "strtold", scope: !491, file: !491, line: 127, type: !712, flags: DIFlagPrototyped, spFlags: 0)
!712 = !DISubroutineType(types: !713)
!713 = !{!714, !629, !654}
!714 = !DIBasicType(name: "long double", size: 128, encoding: DW_ATE_float)
!715 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !680, file: !552, line: 240)
!716 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !686, file: !552, line: 242)
!717 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !688, file: !552, line: 244)
!718 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !719, file: !552, line: 245)
!719 = !DISubprogram(name: "div", linkageName: "_ZN9__gnu_cxx3divExx", scope: !679, file: !552, line: 213, type: !691, flags: DIFlagPrototyped, spFlags: 0)
!720 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !690, file: !552, line: 246)
!721 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !694, file: !552, line: 248)
!722 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !707, file: !552, line: 249)
!723 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !698, file: !552, line: 250)
!724 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !702, file: !552, line: 251)
!725 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !711, file: !552, line: 252)
!726 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !560, file: !727, line: 38)
!727 = !DIFile(filename: "/usr/lib/gcc/x86_64-linux-gnu/11/../../../../include/c++/11/stdlib.h", directory: "")
!728 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !564, file: !727, line: 39)
!729 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !597, file: !727, line: 40)
!730 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !550, file: !727, line: 51)
!731 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !554, file: !727, line: 52)
!732 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !733, file: !727, line: 54)
!733 = !DISubprogram(name: "abs", linkageName: "_ZSt3abse", scope: !301, file: !492, line: 79, type: !734, flags: DIFlagPrototyped, spFlags: 0)
!734 = !DISubroutineType(types: !735)
!735 = !{!714, !714}
!736 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !569, file: !727, line: 55)
!737 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !571, file: !727, line: 56)
!738 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !575, file: !727, line: 57)
!739 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !579, file: !727, line: 58)
!740 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !589, file: !727, line: 59)
!741 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !719, file: !727, line: 60)
!742 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !601, file: !727, line: 61)
!743 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !605, file: !727, line: 62)
!744 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !609, file: !727, line: 63)
!745 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !611, file: !727, line: 64)
!746 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !615, file: !727, line: 65)
!747 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !619, file: !727, line: 67)
!748 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !623, file: !727, line: 68)
!749 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !631, file: !727, line: 69)
!750 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !635, file: !727, line: 71)
!751 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !639, file: !727, line: 72)
!752 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !643, file: !727, line: 73)
!753 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !647, file: !727, line: 74)
!754 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !651, file: !727, line: 75)
!755 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !657, file: !727, line: 76)
!756 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !661, file: !727, line: 77)
!757 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !665, file: !727, line: 78)
!758 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !667, file: !727, line: 80)
!759 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !675, file: !727, line: 81)
!760 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !761, file: !763, line: 414)
!761 = !DISubprogram(name: "acosf", linkageName: "_ZL5acosff", scope: !762, file: !762, line: 1126, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!762 = !DIFile(filename: "/usr/local/cuda/include/math_functions.hpp", directory: "")
!763 = !DIFile(filename: "/u/NAS_SCRATCH/ah7226/tulip/llvm-install-tulip/lib/clang/9.0.0/include/__clang_cuda_cmath.h", directory: "")
!764 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !765, file: !763, line: 415)
!765 = !DISubprogram(name: "acoshf", linkageName: "_ZL6acoshff", scope: !762, file: !762, line: 1154, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!766 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !767, file: !763, line: 416)
!767 = !DISubprogram(name: "asinf", linkageName: "_ZL5asinff", scope: !762, file: !762, line: 1121, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!768 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !769, file: !763, line: 417)
!769 = !DISubprogram(name: "asinhf", linkageName: "_ZL6asinhff", scope: !762, file: !762, line: 1159, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!770 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !771, file: !763, line: 418)
!771 = !DISubprogram(name: "atan2f", linkageName: "_ZL6atan2fff", scope: !762, file: !762, line: 1111, type: !321, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!772 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !773, file: !763, line: 419)
!773 = !DISubprogram(name: "atanf", linkageName: "_ZL5atanff", scope: !762, file: !762, line: 1116, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!774 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !775, file: !763, line: 420)
!775 = !DISubprogram(name: "atanhf", linkageName: "_ZL6atanhff", scope: !762, file: !762, line: 1164, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!776 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !777, file: !763, line: 421)
!777 = !DISubprogram(name: "cbrtf", linkageName: "_ZL5cbrtff", scope: !762, file: !762, line: 1199, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!778 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !779, file: !763, line: 422)
!779 = !DISubprogram(name: "ceilf", linkageName: "_ZL5ceilff", scope: !780, file: !780, line: 647, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!780 = !DIFile(filename: "/usr/local/cuda/include/device_functions.hpp", directory: "")
!781 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !782, file: !763, line: 423)
!782 = !DISubprogram(name: "copysignf", linkageName: "_ZL9copysignfff", scope: !762, file: !762, line: 973, type: !321, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!783 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !784, file: !763, line: 424)
!784 = !DISubprogram(name: "cosf", linkageName: "_ZL4cosff", scope: !762, file: !762, line: 1027, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!785 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !786, file: !763, line: 425)
!786 = !DISubprogram(name: "coshf", linkageName: "_ZL5coshff", scope: !762, file: !762, line: 1096, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!787 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !788, file: !763, line: 426)
!788 = !DISubprogram(name: "erfcf", linkageName: "_ZL5erfcff", scope: !762, file: !762, line: 1259, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!789 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !790, file: !763, line: 427)
!790 = !DISubprogram(name: "erff", linkageName: "_ZL4erfff", scope: !762, file: !762, line: 1249, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!791 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !792, file: !763, line: 428)
!792 = !DISubprogram(name: "exp2f", linkageName: "_ZL5exp2ff", scope: !780, file: !780, line: 637, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!793 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !794, file: !763, line: 429)
!794 = !DISubprogram(name: "expf", linkageName: "_ZL4expff", scope: !762, file: !762, line: 1078, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!795 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !796, file: !763, line: 430)
!796 = !DISubprogram(name: "expm1f", linkageName: "_ZL6expm1ff", scope: !762, file: !762, line: 1169, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!797 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !798, file: !763, line: 431)
!798 = !DISubprogram(name: "fabsf", linkageName: "_ZL5fabsff", scope: !780, file: !780, line: 582, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!799 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !800, file: !763, line: 432)
!800 = !DISubprogram(name: "fdimf", linkageName: "_ZL5fdimfff", scope: !762, file: !762, line: 1385, type: !321, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!801 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !802, file: !763, line: 433)
!802 = !DISubprogram(name: "floorf", linkageName: "_ZL6floorff", scope: !780, file: !780, line: 572, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!803 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !804, file: !763, line: 434)
!804 = !DISubprogram(name: "fmaf", linkageName: "_ZL4fmaffff", scope: !762, file: !762, line: 1337, type: !353, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!805 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !806, file: !763, line: 435)
!806 = !DISubprogram(name: "fmaxf", linkageName: "_ZL5fmaxfff", scope: !780, file: !780, line: 602, type: !321, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!807 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !808, file: !763, line: 436)
!808 = !DISubprogram(name: "fminf", linkageName: "_ZL5fminfff", scope: !780, file: !780, line: 597, type: !321, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!809 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !810, file: !763, line: 437)
!810 = !DISubprogram(name: "fmodf", linkageName: "_ZL5fmodfff", scope: !762, file: !762, line: 1322, type: !321, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!811 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !812, file: !763, line: 438)
!812 = !DISubprogram(name: "frexpf", linkageName: "_ZL6frexpffPi", scope: !762, file: !762, line: 1312, type: !367, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!813 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !814, file: !763, line: 439)
!814 = !DISubprogram(name: "hypotf", linkageName: "_ZL6hypotfff", scope: !762, file: !762, line: 1174, type: !321, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!815 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !816, file: !763, line: 440)
!816 = !DISubprogram(name: "ilogbf", linkageName: "_ZL6ilogbff", scope: !762, file: !762, line: 1390, type: !363, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!817 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !818, file: !763, line: 441)
!818 = !DISubprogram(name: "ldexpf", linkageName: "_ZL6ldexpffi", scope: !762, file: !762, line: 1289, type: !405, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!819 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !820, file: !763, line: 442)
!820 = !DISubprogram(name: "lgammaf", linkageName: "_ZL7lgammaff", scope: !762, file: !762, line: 1284, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!821 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !822, file: !763, line: 443)
!822 = !DISubprogram(name: "llrintf", linkageName: "_ZL7llrintff", scope: !762, file: !762, line: 933, type: !416, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!823 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !824, file: !763, line: 444)
!824 = !DISubprogram(name: "llroundf", linkageName: "_ZL8llroundff", scope: !762, file: !762, line: 1371, type: !416, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!825 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !826, file: !763, line: 445)
!826 = !DISubprogram(name: "log10f", linkageName: "_ZL6log10ff", scope: !762, file: !762, line: 1140, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!827 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !828, file: !763, line: 446)
!828 = !DISubprogram(name: "log1pf", linkageName: "_ZL6log1pff", scope: !762, file: !762, line: 1149, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!829 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !830, file: !763, line: 447)
!830 = !DISubprogram(name: "log2f", linkageName: "_ZL5log2ff", scope: !762, file: !762, line: 1069, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!831 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !832, file: !763, line: 448)
!832 = !DISubprogram(name: "logbf", linkageName: "_ZL5logbff", scope: !762, file: !762, line: 1395, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!833 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !834, file: !763, line: 449)
!834 = !DISubprogram(name: "logf", linkageName: "_ZL4logff", scope: !762, file: !762, line: 1131, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!835 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !836, file: !763, line: 450)
!836 = !DISubprogram(name: "lrintf", linkageName: "_ZL6lrintff", scope: !762, file: !762, line: 924, type: !430, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!837 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !838, file: !763, line: 451)
!838 = !DISubprogram(name: "lroundf", linkageName: "_ZL7lroundff", scope: !762, file: !762, line: 1376, type: !430, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!839 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !840, file: !763, line: 452)
!840 = !DISubprogram(name: "modff", linkageName: "_ZL5modfffPf", scope: !762, file: !762, line: 1317, type: !438, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!841 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !842, file: !763, line: 453)
!842 = !DISubprogram(name: "nearbyintf", linkageName: "_ZL10nearbyintff", scope: !762, file: !762, line: 938, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!843 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !844, file: !763, line: 454)
!844 = !DISubprogram(name: "nextafterf", linkageName: "_ZL10nextafterfff", scope: !762, file: !762, line: 1002, type: !321, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!845 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !846, file: !763, line: 455)
!846 = !DISubprogram(name: "powf", linkageName: "_ZL4powfff", scope: !762, file: !762, line: 1352, type: !321, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!847 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !848, file: !763, line: 456)
!848 = !DISubprogram(name: "remainderf", linkageName: "_ZL10remainderfff", scope: !762, file: !762, line: 1327, type: !321, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!849 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !850, file: !763, line: 457)
!850 = !DISubprogram(name: "remquof", linkageName: "_ZL7remquofffPi", scope: !762, file: !762, line: 1332, type: !461, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!851 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !852, file: !763, line: 458)
!852 = !DISubprogram(name: "rintf", linkageName: "_ZL5rintff", scope: !762, file: !762, line: 919, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!853 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !854, file: !763, line: 459)
!854 = !DISubprogram(name: "roundf", linkageName: "_ZL6roundff", scope: !762, file: !762, line: 1366, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!855 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !856, file: !763, line: 462)
!856 = !DISubprogram(name: "scalblnf", linkageName: "_ZL8scalblnffl", scope: !762, file: !762, line: 1299, type: !469, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!857 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !858, file: !763, line: 464)
!858 = !DISubprogram(name: "scalbnf", linkageName: "_ZL7scalbnffi", scope: !762, file: !762, line: 1294, type: !405, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!859 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !860, file: !763, line: 465)
!860 = !DISubprogram(name: "sinf", linkageName: "_ZL4sinff", scope: !762, file: !762, line: 1018, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!861 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !862, file: !763, line: 466)
!862 = !DISubprogram(name: "sinhf", linkageName: "_ZL5sinhff", scope: !762, file: !762, line: 1101, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!863 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !864, file: !763, line: 467)
!864 = !DISubprogram(name: "sqrtf", linkageName: "_ZL5sqrtff", scope: !780, file: !780, line: 887, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!865 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !866, file: !763, line: 468)
!866 = !DISubprogram(name: "tanf", linkageName: "_ZL4tanff", scope: !762, file: !762, line: 1060, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!867 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !868, file: !763, line: 469)
!868 = !DISubprogram(name: "tanhf", linkageName: "_ZL5tanhff", scope: !762, file: !762, line: 1106, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!869 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !870, file: !763, line: 470)
!870 = !DISubprogram(name: "tgammaf", linkageName: "_ZL7tgammaff", scope: !762, file: !762, line: 1361, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!871 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !872, file: !763, line: 471)
!872 = !DISubprogram(name: "truncf", linkageName: "_ZL6truncff", scope: !780, file: !780, line: 642, type: !308, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!873 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !874, file: !878, line: 98)
!874 = !DIDerivedType(tag: DW_TAG_typedef, name: "FILE", file: !875, line: 7, baseType: !876)
!875 = !DIFile(filename: "/usr/include/x86_64-linux-gnu/bits/types/FILE.h", directory: "")
!876 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "_IO_FILE", file: !877, line: 49, flags: DIFlagFwdDecl, identifier: "_ZTS8_IO_FILE")
!877 = !DIFile(filename: "/usr/include/x86_64-linux-gnu/bits/types/struct_FILE.h", directory: "")
!878 = !DIFile(filename: "/usr/lib/gcc/x86_64-linux-gnu/11/../../../../include/c++/11/cstdio", directory: "")
!879 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !880, file: !878, line: 99)
!880 = !DIDerivedType(tag: DW_TAG_typedef, name: "fpos_t", file: !881, line: 84, baseType: !882)
!881 = !DIFile(filename: "/usr/include/stdio.h", directory: "")
!882 = !DIDerivedType(tag: DW_TAG_typedef, name: "__fpos_t", file: !883, line: 14, baseType: !884)
!883 = !DIFile(filename: "/usr/include/x86_64-linux-gnu/bits/types/__fpos_t.h", directory: "")
!884 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "_G_fpos_t", file: !883, line: 10, flags: DIFlagFwdDecl, identifier: "_ZTS9_G_fpos_t")
!885 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !886, file: !878, line: 101)
!886 = !DISubprogram(name: "clearerr", scope: !881, file: !881, line: 786, type: !887, flags: DIFlagPrototyped, spFlags: 0)
!887 = !DISubroutineType(types: !888)
!888 = !{null, !889}
!889 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !874, size: 64)
!890 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !891, file: !878, line: 102)
!891 = !DISubprogram(name: "fclose", scope: !881, file: !881, line: 178, type: !892, flags: DIFlagPrototyped, spFlags: 0)
!892 = !DISubroutineType(types: !893)
!893 = !{!97, !889}
!894 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !895, file: !878, line: 103)
!895 = !DISubprogram(name: "feof", scope: !881, file: !881, line: 788, type: !892, flags: DIFlagPrototyped, spFlags: 0)
!896 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !897, file: !878, line: 104)
!897 = !DISubprogram(name: "ferror", scope: !881, file: !881, line: 790, type: !892, flags: DIFlagPrototyped, spFlags: 0)
!898 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !899, file: !878, line: 105)
!899 = !DISubprogram(name: "fflush", scope: !881, file: !881, line: 230, type: !892, flags: DIFlagPrototyped, spFlags: 0)
!900 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !901, file: !878, line: 106)
!901 = !DISubprogram(name: "fgetc", scope: !881, file: !881, line: 513, type: !892, flags: DIFlagPrototyped, spFlags: 0)
!902 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !903, file: !878, line: 107)
!903 = !DISubprogram(name: "fgetpos", scope: !881, file: !881, line: 760, type: !904, flags: DIFlagPrototyped, spFlags: 0)
!904 = !DISubroutineType(types: !905)
!905 = !{!97, !906, !907}
!906 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !889)
!907 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !908)
!908 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !880, size: 64)
!909 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !910, file: !878, line: 108)
!910 = !DISubprogram(name: "fgets", scope: !881, file: !881, line: 592, type: !911, flags: DIFlagPrototyped, spFlags: 0)
!911 = !DISubroutineType(types: !912)
!912 = !{!108, !670, !97, !906}
!913 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !914, file: !878, line: 109)
!914 = !DISubprogram(name: "fopen", scope: !881, file: !881, line: 258, type: !915, flags: DIFlagPrototyped, spFlags: 0)
!915 = !DISubroutineType(types: !916)
!916 = !{!889, !629, !629}
!917 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !918, file: !878, line: 110)
!918 = !DISubprogram(name: "fprintf", scope: !881, file: !881, line: 350, type: !919, flags: DIFlagPrototyped, spFlags: 0)
!919 = !DISubroutineType(types: !920)
!920 = !{!97, !906, !629, null}
!921 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !922, file: !878, line: 111)
!922 = !DISubprogram(name: "fputc", scope: !881, file: !881, line: 549, type: !923, flags: DIFlagPrototyped, spFlags: 0)
!923 = !DISubroutineType(types: !924)
!924 = !{!97, !97, !889}
!925 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !926, file: !878, line: 112)
!926 = !DISubprogram(name: "fputs", scope: !881, file: !881, line: 655, type: !927, flags: DIFlagPrototyped, spFlags: 0)
!927 = !DISubroutineType(types: !928)
!928 = !{!97, !629, !906}
!929 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !930, file: !878, line: 113)
!930 = !DISubprogram(name: "fread", scope: !881, file: !881, line: 675, type: !931, flags: DIFlagPrototyped, spFlags: 0)
!931 = !DISubroutineType(types: !932)
!932 = !{!131, !933, !131, !131, !906}
!933 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !111)
!934 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !935, file: !878, line: 114)
!935 = !DISubprogram(name: "freopen", scope: !881, file: !881, line: 265, type: !936, flags: DIFlagPrototyped, spFlags: 0)
!936 = !DISubroutineType(types: !937)
!937 = !{!889, !629, !629, !906}
!938 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !939, file: !878, line: 115)
!939 = !DISubprogram(name: "fscanf", scope: !881, file: !881, line: 415, type: !919, flags: DIFlagPrototyped, spFlags: 0)
!940 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !941, file: !878, line: 116)
!941 = !DISubprogram(name: "fseek", scope: !881, file: !881, line: 713, type: !942, flags: DIFlagPrototyped, spFlags: 0)
!942 = !DISubroutineType(types: !943)
!943 = !{!97, !889, !402, !97}
!944 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !945, file: !878, line: 117)
!945 = !DISubprogram(name: "fsetpos", scope: !881, file: !881, line: 765, type: !946, flags: DIFlagPrototyped, spFlags: 0)
!946 = !DISubroutineType(types: !947)
!947 = !{!97, !889, !948}
!948 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !949, size: 64)
!949 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !880)
!950 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !951, file: !878, line: 118)
!951 = !DISubprogram(name: "ftell", scope: !881, file: !881, line: 718, type: !952, flags: DIFlagPrototyped, spFlags: 0)
!952 = !DISubroutineType(types: !953)
!953 = !{!402, !889}
!954 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !955, file: !878, line: 119)
!955 = !DISubprogram(name: "fwrite", scope: !881, file: !881, line: 681, type: !956, flags: DIFlagPrototyped, spFlags: 0)
!956 = !DISubroutineType(types: !957)
!957 = !{!131, !958, !131, !131, !906}
!958 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !582)
!959 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !960, file: !878, line: 120)
!960 = !DISubprogram(name: "getc", scope: !881, file: !881, line: 514, type: !892, flags: DIFlagPrototyped, spFlags: 0)
!961 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !962, file: !878, line: 121)
!962 = !DISubprogram(name: "getchar", scope: !881, file: !881, line: 520, type: !640, flags: DIFlagPrototyped, spFlags: 0)
!963 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !964, file: !878, line: 124)
!964 = !DISubprogram(name: "gets", scope: !881, file: !881, line: 605, type: !965, flags: DIFlagPrototyped, spFlags: 0)
!965 = !DISubroutineType(types: !966)
!966 = !{!108, !108}
!967 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !968, file: !878, line: 126)
!968 = !DISubprogram(name: "perror", scope: !881, file: !881, line: 804, type: !969, flags: DIFlagPrototyped, spFlags: 0)
!969 = !DISubroutineType(types: !970)
!970 = !{null, !445}
!971 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !972, file: !878, line: 127)
!972 = !DISubprogram(name: "printf", scope: !881, file: !881, line: 356, type: !973, flags: DIFlagPrototyped, spFlags: 0)
!973 = !DISubroutineType(types: !974)
!974 = !{!97, !629, null}
!975 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !976, file: !878, line: 128)
!976 = !DISubprogram(name: "putc", scope: !881, file: !881, line: 550, type: !923, flags: DIFlagPrototyped, spFlags: 0)
!977 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !978, file: !878, line: 129)
!978 = !DISubprogram(name: "putchar", scope: !881, file: !881, line: 556, type: !304, flags: DIFlagPrototyped, spFlags: 0)
!979 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !980, file: !878, line: 130)
!980 = !DISubprogram(name: "puts", scope: !881, file: !881, line: 661, type: !572, flags: DIFlagPrototyped, spFlags: 0)
!981 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !982, file: !878, line: 131)
!982 = !DISubprogram(name: "remove", scope: !881, file: !881, line: 152, type: !572, flags: DIFlagPrototyped, spFlags: 0)
!983 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !984, file: !878, line: 132)
!984 = !DISubprogram(name: "rename", scope: !881, file: !881, line: 154, type: !985, flags: DIFlagPrototyped, spFlags: 0)
!985 = !DISubroutineType(types: !986)
!986 = !{!97, !445, !445}
!987 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !988, file: !878, line: 133)
!988 = !DISubprogram(name: "rewind", scope: !881, file: !881, line: 723, type: !887, flags: DIFlagPrototyped, spFlags: 0)
!989 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !990, file: !878, line: 134)
!990 = !DISubprogram(name: "scanf", scope: !881, file: !881, line: 421, type: !973, flags: DIFlagPrototyped, spFlags: 0)
!991 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !992, file: !878, line: 135)
!992 = !DISubprogram(name: "setbuf", scope: !881, file: !881, line: 328, type: !993, flags: DIFlagPrototyped, spFlags: 0)
!993 = !DISubroutineType(types: !994)
!994 = !{null, !906, !670}
!995 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !996, file: !878, line: 136)
!996 = !DISubprogram(name: "setvbuf", scope: !881, file: !881, line: 332, type: !997, flags: DIFlagPrototyped, spFlags: 0)
!997 = !DISubroutineType(types: !998)
!998 = !{!97, !906, !670, !97, !131}
!999 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !1000, file: !878, line: 137)
!1000 = !DISubprogram(name: "sprintf", scope: !881, file: !881, line: 358, type: !1001, flags: DIFlagPrototyped, spFlags: 0)
!1001 = !DISubroutineType(types: !1002)
!1002 = !{!97, !670, !629, null}
!1003 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !1004, file: !878, line: 138)
!1004 = !DISubprogram(name: "sscanf", scope: !881, file: !881, line: 423, type: !1005, flags: DIFlagPrototyped, spFlags: 0)
!1005 = !DISubroutineType(types: !1006)
!1006 = !{!97, !629, !629, null}
!1007 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !1008, file: !878, line: 139)
!1008 = !DISubprogram(name: "tmpfile", scope: !881, file: !881, line: 188, type: !1009, flags: DIFlagPrototyped, spFlags: 0)
!1009 = !DISubroutineType(types: !1010)
!1010 = !{!889}
!1011 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !1012, file: !878, line: 141)
!1012 = !DISubprogram(name: "tmpnam", scope: !881, file: !881, line: 205, type: !965, flags: DIFlagPrototyped, spFlags: 0)
!1013 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !1014, file: !878, line: 143)
!1014 = !DISubprogram(name: "ungetc", scope: !881, file: !881, line: 668, type: !923, flags: DIFlagPrototyped, spFlags: 0)
!1015 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !1016, file: !878, line: 144)
!1016 = !DISubprogram(name: "vfprintf", scope: !881, file: !881, line: 365, type: !1017, flags: DIFlagPrototyped, spFlags: 0)
!1017 = !DISubroutineType(types: !1018)
!1018 = !{!97, !906, !629, !1019}
!1019 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !1020, size: 64)
!1020 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "__va_list_tag", file: !3, flags: DIFlagFwdDecl, identifier: "_ZTS13__va_list_tag")
!1021 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !1022, file: !878, line: 145)
!1022 = !DISubprogram(name: "vprintf", scope: !881, file: !881, line: 371, type: !1023, flags: DIFlagPrototyped, spFlags: 0)
!1023 = !DISubroutineType(types: !1024)
!1024 = !{!97, !629, !1019}
!1025 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !1026, file: !878, line: 146)
!1026 = !DISubprogram(name: "vsprintf", scope: !881, file: !881, line: 373, type: !1027, flags: DIFlagPrototyped, spFlags: 0)
!1027 = !DISubroutineType(types: !1028)
!1028 = !{!97, !670, !629, !1019}
!1029 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !679, entity: !1030, file: !878, line: 175)
!1030 = !DISubprogram(name: "snprintf", scope: !881, file: !881, line: 378, type: !1031, flags: DIFlagPrototyped, spFlags: 0)
!1031 = !DISubroutineType(types: !1032)
!1032 = !{!97, !670, !131, !629, null}
!1033 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !679, entity: !1034, file: !878, line: 176)
!1034 = !DISubprogram(name: "vfscanf", scope: !881, file: !881, line: 459, type: !1017, flags: DIFlagPrototyped, spFlags: 0)
!1035 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !679, entity: !1036, file: !878, line: 177)
!1036 = !DISubprogram(name: "vscanf", scope: !881, file: !881, line: 467, type: !1023, flags: DIFlagPrototyped, spFlags: 0)
!1037 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !679, entity: !1038, file: !878, line: 178)
!1038 = !DISubprogram(name: "vsnprintf", scope: !881, file: !881, line: 382, type: !1039, flags: DIFlagPrototyped, spFlags: 0)
!1039 = !DISubroutineType(types: !1040)
!1040 = !{!97, !670, !131, !629, !1019}
!1041 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !679, entity: !1042, file: !878, line: 179)
!1042 = !DISubprogram(name: "vsscanf", scope: !881, file: !881, line: 471, type: !1043, flags: DIFlagPrototyped, spFlags: 0)
!1043 = !DISubroutineType(types: !1044)
!1044 = !{!97, !629, !629, !1019}
!1045 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !1030, file: !878, line: 185)
!1046 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !1034, file: !878, line: 186)
!1047 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !1036, file: !878, line: 187)
!1048 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !1038, file: !878, line: 188)
!1049 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !1042, file: !878, line: 189)
!1050 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 9.0.0 (https://github.com/yebinchon/llvm-project/ a1efa594106d738d0b74c9e4e2b9b779eb8b7d25)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !1051, retainedTypes: !1052, imports: !1056, nameTableKind: None)
!1051 = !{}
!1052 = !{!98, !104, !106, !97, !1053, !705, !1054, !413}
!1053 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !705, size: 64)
!1054 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !1055, size: 64)
!1055 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !413)
!1056 = !{!300, !306, !311, !313, !315, !317, !319, !323, !325, !327, !329, !331, !333, !335, !337, !339, !341, !343, !345, !347, !349, !351, !355, !357, !359, !361, !365, !369, !371, !373, !378, !382, !384, !386, !388, !390, !392, !394, !396, !398, !403, !407, !409, !414, !418, !420, !422, !424, !426, !428, !432, !434, !436, !441, !447, !451, !453, !455, !457, !459, !463, !465, !467, !471, !473, !475, !477, !479, !481, !483, !485, !487, !489, !493, !499, !501, !503, !507, !509, !511, !513, !515, !517, !519, !521, !525, !529, !531, !533, !537, !539, !541, !543, !545, !547, !549, !553, !559, !563, !568, !570, !574, !578, !588, !592, !596, !600, !604, !608, !610, !614, !618, !622, !630, !634, !638, !642, !646, !650, !656, !660, !664, !666, !674, !678, !685, !687, !689, !693, !697, !701, !706, !1057, !715, !716, !717, !718, !720, !721, !722, !723, !724, !1062, !1063, !1064, !1065, !1066, !1067, !1068, !1072, !1073, !1074, !1075, !1076, !1077, !1078, !1079, !1080, !1081, !1082, !1083, !1084, !1085, !1086, !1087, !1088, !1089, !1090, !1091, !1092, !1093, !1094, !1095, !760, !764, !766, !768, !770, !772, !774, !776, !778, !781, !783, !785, !787, !789, !791, !793, !795, !797, !799, !801, !803, !805, !807, !809, !811, !813, !815, !817, !819, !821, !823, !825, !827, !829, !831, !833, !835, !837, !839, !841, !843, !845, !847, !849, !851, !853, !855, !857, !859, !861, !863, !865, !867, !869, !871, !873, !879, !885, !890, !894, !896, !898, !900, !902, !909, !913, !917, !921, !925, !929, !934, !938, !940, !944, !950, !954, !959, !961, !963, !967, !971, !975, !977, !979, !981, !983, !987, !989, !991, !995, !999, !1003, !1007, !1011, !1013, !1096, !1103, !1107, !1029, !1111, !1113, !1115, !1119, !1045, !1123, !1124, !1125, !1126}
!1057 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !679, entity: !1058, file: !552, line: 232)
!1058 = !DISubprogram(name: "strtold", scope: !491, file: !491, line: 127, type: !1059, flags: DIFlagPrototyped, spFlags: 0)
!1059 = !DISubroutineType(types: !1060)
!1060 = !{!1061, !629, !654}
!1061 = !DIBasicType(name: "long double", size: 64, encoding: DW_ATE_float)
!1062 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !1058, file: !552, line: 252)
!1063 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !1050, entity: !560, file: !727, line: 38)
!1064 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !1050, entity: !564, file: !727, line: 39)
!1065 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !1050, entity: !597, file: !727, line: 40)
!1066 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !1050, entity: !550, file: !727, line: 51)
!1067 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !1050, entity: !554, file: !727, line: 52)
!1068 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !1050, entity: !1069, file: !727, line: 54)
!1069 = !DISubprogram(name: "abs", linkageName: "_ZSt3abse", scope: !301, file: !492, line: 79, type: !1070, flags: DIFlagPrototyped, spFlags: 0)
!1070 = !DISubroutineType(types: !1071)
!1071 = !{!1061, !1061}
!1072 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !1050, entity: !569, file: !727, line: 55)
!1073 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !1050, entity: !571, file: !727, line: 56)
!1074 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !1050, entity: !575, file: !727, line: 57)
!1075 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !1050, entity: !579, file: !727, line: 58)
!1076 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !1050, entity: !589, file: !727, line: 59)
!1077 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !1050, entity: !719, file: !727, line: 60)
!1078 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !1050, entity: !601, file: !727, line: 61)
!1079 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !1050, entity: !605, file: !727, line: 62)
!1080 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !1050, entity: !609, file: !727, line: 63)
!1081 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !1050, entity: !611, file: !727, line: 64)
!1082 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !1050, entity: !615, file: !727, line: 65)
!1083 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !1050, entity: !619, file: !727, line: 67)
!1084 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !1050, entity: !623, file: !727, line: 68)
!1085 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !1050, entity: !631, file: !727, line: 69)
!1086 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !1050, entity: !635, file: !727, line: 71)
!1087 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !1050, entity: !639, file: !727, line: 72)
!1088 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !1050, entity: !643, file: !727, line: 73)
!1089 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !1050, entity: !647, file: !727, line: 74)
!1090 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !1050, entity: !651, file: !727, line: 75)
!1091 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !1050, entity: !657, file: !727, line: 76)
!1092 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !1050, entity: !661, file: !727, line: 77)
!1093 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !1050, entity: !665, file: !727, line: 78)
!1094 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !1050, entity: !667, file: !727, line: 80)
!1095 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !1050, entity: !675, file: !727, line: 81)
!1096 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !1097, file: !878, line: 144)
!1097 = !DISubprogram(name: "vfprintf", scope: !881, file: !881, line: 365, type: !1098, flags: DIFlagPrototyped, spFlags: 0)
!1098 = !DISubroutineType(types: !1099)
!1099 = !{!97, !906, !629, !1100}
!1100 = !DIDerivedType(tag: DW_TAG_typedef, name: "__gnuc_va_list", file: !1101, line: 32, baseType: !1102)
!1101 = !DIFile(filename: "/u/NAS_SCRATCH/ah7226/tulip/llvm-install-tulip/lib/clang/9.0.0/include/stdarg.h", directory: "")
!1102 = !DIDerivedType(tag: DW_TAG_typedef, name: "__builtin_va_list", file: !3, baseType: !108)
!1103 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !1104, file: !878, line: 145)
!1104 = !DISubprogram(name: "vprintf", scope: !881, file: !881, line: 371, type: !1105, flags: DIFlagPrototyped, spFlags: 0)
!1105 = !DISubroutineType(types: !1106)
!1106 = !{!97, !629, !1100}
!1107 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !1108, file: !878, line: 146)
!1108 = !DISubprogram(name: "vsprintf", scope: !881, file: !881, line: 373, type: !1109, flags: DIFlagPrototyped, spFlags: 0)
!1109 = !DISubroutineType(types: !1110)
!1110 = !{!97, !670, !629, !1100}
!1111 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !679, entity: !1112, file: !878, line: 176)
!1112 = !DISubprogram(name: "vfscanf", scope: !881, file: !881, line: 459, type: !1098, flags: DIFlagPrototyped, spFlags: 0)
!1113 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !679, entity: !1114, file: !878, line: 177)
!1114 = !DISubprogram(name: "vscanf", scope: !881, file: !881, line: 467, type: !1105, flags: DIFlagPrototyped, spFlags: 0)
!1115 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !679, entity: !1116, file: !878, line: 178)
!1116 = !DISubprogram(name: "vsnprintf", scope: !881, file: !881, line: 382, type: !1117, flags: DIFlagPrototyped, spFlags: 0)
!1117 = !DISubroutineType(types: !1118)
!1118 = !{!97, !670, !131, !629, !1100}
!1119 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !679, entity: !1120, file: !878, line: 179)
!1120 = !DISubprogram(name: "vsscanf", scope: !881, file: !881, line: 471, type: !1121, flags: DIFlagPrototyped, spFlags: 0)
!1121 = !DISubroutineType(types: !1122)
!1122 = !{!97, !629, !629, !1100}
!1123 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !1112, file: !878, line: 186)
!1124 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !1114, file: !878, line: 187)
!1125 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !1116, file: !878, line: 188)
!1126 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !1120, file: !878, line: 189)
!1127 = !{void (%struct.dcomplex*, %struct.dcomplex*)* @_Z19cffts1_gpu_kernel_1P8dcomplexS0_, !"kernel", i32 1}
!1128 = !{void (i32, %struct.dcomplex*, %struct.dcomplex*, %struct.dcomplex*)* @_Z19cffts1_gpu_kernel_2iP8dcomplexS0_S0_, !"kernel", i32 1}
!1129 = !{void (%struct.dcomplex*, %struct.dcomplex*)* @_Z19cffts1_gpu_kernel_3P8dcomplexS0_, !"kernel", i32 1}
!1130 = !{void (%struct.dcomplex*, %struct.dcomplex*)* @_Z19cffts2_gpu_kernel_1P8dcomplexS0_, !"kernel", i32 1}
!1131 = !{void (i32, %struct.dcomplex*, %struct.dcomplex*, %struct.dcomplex*)* @_Z19cffts2_gpu_kernel_2iP8dcomplexS0_S0_, !"kernel", i32 1}
!1132 = !{void (%struct.dcomplex*, %struct.dcomplex*)* @_Z19cffts2_gpu_kernel_3P8dcomplexS0_, !"kernel", i32 1}
!1133 = !{void (%struct.dcomplex*, %struct.dcomplex*)* @_Z19cffts3_gpu_kernel_1P8dcomplexS0_, !"kernel", i32 1}
!1134 = !{void (i32, %struct.dcomplex*, %struct.dcomplex*, %struct.dcomplex*)* @_Z19cffts3_gpu_kernel_2iP8dcomplexS0_S0_, !"kernel", i32 1}
!1135 = !{void (%struct.dcomplex*, %struct.dcomplex*)* @_Z19cffts3_gpu_kernel_3P8dcomplexS0_, !"kernel", i32 1}
!1136 = !{void (i32, %struct.dcomplex*, %struct.dcomplex*)* @_Z19checksum_gpu_kerneliP8dcomplexS0_, !"kernel", i32 1}
!1137 = !{void (double*)* @_Z27compute_indexmap_gpu_kernelPd, !"kernel", i32 1}
!1138 = !{void (%struct.dcomplex*, double*)* @_Z37compute_initial_conditions_gpu_kernelP8dcomplexPd, !"kernel", i32 1}
!1139 = !{void (%struct.dcomplex*, %struct.dcomplex*, double*)* @_Z17evolve_gpu_kernelP8dcomplexS0_Pd, !"kernel", i32 1}
!1140 = !{void (%struct.dcomplex*, %struct.dcomplex*, double*)* @_Z18init_ui_gpu_kernelP8dcomplexS0_Pd, !"kernel", i32 1}
!1141 = !{null, !"align", i32 8}
!1142 = !{null, !"align", i32 8, !"align", i32 65544, !"align", i32 131080}
!1143 = !{null, !"align", i32 16}
!1144 = !{null, !"align", i32 16, !"align", i32 65552, !"align", i32 131088}
!1145 = !{!"clang version 9.0.0 (https://github.com/yebinchon/llvm-project/ a1efa594106d738d0b74c9e4e2b9b779eb8b7d25)"}
!1146 = !{i32 1, i32 2}
!1147 = !{i32 2, !"SDK Version", [2 x i32] [i32 7, i32 0]}
!1148 = !{i32 2, !"Dwarf Version", i32 2}
!1149 = !{i32 2, !"Debug Info Version", i32 3}
!1150 = !{i32 1, !"wchar_size", i32 4}
!1151 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!1152 = distinct !DISubprogram(name: "cffts1_gpu_kernel_1", linkageName: "_Z19cffts1_gpu_kernel_1P8dcomplexS0_", scope: !3, file: !3, line: 774, type: !1153, scopeLine: 775, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !1050, retainedNodes: !1051)
!1153 = !DISubroutineType(types: !1154)
!1154 = !{null, !98, !98}
!1155 = !DILocalVariable(name: "x_in", arg: 1, scope: !1152, file: !3, line: 774, type: !98)
!1156 = !DILocation(line: 774, column: 46, scope: !1152)
!1157 = !DILocalVariable(name: "y0", arg: 2, scope: !1152, file: !3, line: 775, type: !98)
!1158 = !DILocation(line: 775, column: 12, scope: !1152)
!1159 = !DILocalVariable(name: "x_y_z", scope: !1152, file: !3, line: 776, type: !97)
!1160 = !DILocation(line: 776, column: 6, scope: !1152)
!1161 = !DILocation(line: 64, column: 3, scope: !1162, inlinedAt: !1197)
!1162 = distinct !DISubprogram(name: "__fetch_builtin_x", linkageName: "_ZN25__cuda_builtin_blockIdx_t17__fetch_builtin_xEv", scope: !1164, file: !1163, line: 64, type: !1167, scopeLine: 64, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !1050, declaration: !1166, retainedNodes: !1051)
!1163 = !DIFile(filename: "/u/NAS_SCRATCH/ah7226/tulip/llvm-install-tulip/lib/clang/9.0.0/include/__clang_cuda_builtin_vars.h", directory: "")
!1164 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "__cuda_builtin_blockIdx_t", file: !1163, line: 63, size: 8, flags: DIFlagTypePassByReference | DIFlagNonTrivial, elements: !1165, identifier: "_ZTS25__cuda_builtin_blockIdx_t")
!1165 = !{!1166, !1169, !1170, !1171, !1182, !1186, !1190, !1193}
!1166 = !DISubprogram(name: "__fetch_builtin_x", linkageName: "_ZN25__cuda_builtin_blockIdx_t17__fetch_builtin_xEv", scope: !1164, file: !1163, line: 64, type: !1167, scopeLine: 64, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!1167 = !DISubroutineType(types: !1168)
!1168 = !{!7}
!1169 = !DISubprogram(name: "__fetch_builtin_y", linkageName: "_ZN25__cuda_builtin_blockIdx_t17__fetch_builtin_yEv", scope: !1164, file: !1163, line: 65, type: !1167, scopeLine: 65, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!1170 = !DISubprogram(name: "__fetch_builtin_z", linkageName: "_ZN25__cuda_builtin_blockIdx_t17__fetch_builtin_zEv", scope: !1164, file: !1163, line: 66, type: !1167, scopeLine: 66, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!1171 = !DISubprogram(name: "operator uint3", linkageName: "_ZNK25__cuda_builtin_blockIdx_tcv5uint3Ev", scope: !1164, file: !1163, line: 69, type: !1172, scopeLine: 69, flags: DIFlagPrototyped, spFlags: 0)
!1172 = !DISubroutineType(types: !1173)
!1173 = !{!1174, !1180}
!1174 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "uint3", file: !1175, line: 190, size: 96, flags: DIFlagTypePassByValue, elements: !1176, identifier: "_ZTS5uint3")
!1175 = !DIFile(filename: "/usr/local/cuda/include/vector_types.h", directory: "")
!1176 = !{!1177, !1178, !1179}
!1177 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !1174, file: !1175, line: 192, baseType: !7, size: 32)
!1178 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !1174, file: !1175, line: 192, baseType: !7, size: 32, offset: 32)
!1179 = !DIDerivedType(tag: DW_TAG_member, name: "z", scope: !1174, file: !1175, line: 192, baseType: !7, size: 32, offset: 64)
!1180 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !1181, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!1181 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !1164)
!1182 = !DISubprogram(name: "__cuda_builtin_blockIdx_t", scope: !1164, file: !1163, line: 71, type: !1183, scopeLine: 71, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!1183 = !DISubroutineType(types: !1184)
!1184 = !{null, !1185}
!1185 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !1164, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!1186 = !DISubprogram(name: "__cuda_builtin_blockIdx_t", scope: !1164, file: !1163, line: 71, type: !1187, scopeLine: 71, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!1187 = !DISubroutineType(types: !1188)
!1188 = !{null, !1185, !1189}
!1189 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !1181, size: 64)
!1190 = !DISubprogram(name: "operator=", linkageName: "_ZNK25__cuda_builtin_blockIdx_taSERKS_", scope: !1164, file: !1163, line: 71, type: !1191, scopeLine: 71, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!1191 = !DISubroutineType(types: !1192)
!1192 = !{null, !1180, !1189}
!1193 = !DISubprogram(name: "operator&", linkageName: "_ZNK25__cuda_builtin_blockIdx_tadEv", scope: !1164, file: !1163, line: 71, type: !1194, scopeLine: 71, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!1194 = !DISubroutineType(types: !1195)
!1195 = !{!1196, !1180}
!1196 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !1164, size: 64)
!1197 = distinct !DILocation(line: 776, column: 14, scope: !1152)
!1198 = !{i32 0, i32 65535}
!1199 = !DILocation(line: 75, column: 3, scope: !1200, inlinedAt: !1242)
!1200 = distinct !DISubprogram(name: "__fetch_builtin_x", linkageName: "_ZN25__cuda_builtin_blockDim_t17__fetch_builtin_xEv", scope: !1201, file: !1163, line: 75, type: !1167, scopeLine: 75, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !1050, declaration: !1203, retainedNodes: !1051)
!1201 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "__cuda_builtin_blockDim_t", file: !1163, line: 74, size: 8, flags: DIFlagTypePassByReference | DIFlagNonTrivial, elements: !1202, identifier: "_ZTS25__cuda_builtin_blockDim_t")
!1202 = !{!1203, !1204, !1205, !1206, !1227, !1231, !1235, !1238}
!1203 = !DISubprogram(name: "__fetch_builtin_x", linkageName: "_ZN25__cuda_builtin_blockDim_t17__fetch_builtin_xEv", scope: !1201, file: !1163, line: 75, type: !1167, scopeLine: 75, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!1204 = !DISubprogram(name: "__fetch_builtin_y", linkageName: "_ZN25__cuda_builtin_blockDim_t17__fetch_builtin_yEv", scope: !1201, file: !1163, line: 76, type: !1167, scopeLine: 76, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!1205 = !DISubprogram(name: "__fetch_builtin_z", linkageName: "_ZN25__cuda_builtin_blockDim_t17__fetch_builtin_zEv", scope: !1201, file: !1163, line: 77, type: !1167, scopeLine: 77, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!1206 = !DISubprogram(name: "operator dim3", linkageName: "_ZNK25__cuda_builtin_blockDim_tcv4dim3Ev", scope: !1201, file: !1163, line: 80, type: !1207, scopeLine: 80, flags: DIFlagPrototyped, spFlags: 0)
!1207 = !DISubroutineType(types: !1208)
!1208 = !{!1209, !1225}
!1209 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "dim3", file: !1175, line: 417, size: 96, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !1210, identifier: "_ZTS4dim3")
!1210 = !{!1211, !1212, !1213, !1214, !1218, !1222}
!1211 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !1209, file: !1175, line: 419, baseType: !7, size: 32)
!1212 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !1209, file: !1175, line: 419, baseType: !7, size: 32, offset: 32)
!1213 = !DIDerivedType(tag: DW_TAG_member, name: "z", scope: !1209, file: !1175, line: 419, baseType: !7, size: 32, offset: 64)
!1214 = !DISubprogram(name: "dim3", scope: !1209, file: !1175, line: 421, type: !1215, scopeLine: 421, flags: DIFlagPrototyped, spFlags: 0)
!1215 = !DISubroutineType(types: !1216)
!1216 = !{null, !1217, !7, !7, !7}
!1217 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !1209, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!1218 = !DISubprogram(name: "dim3", scope: !1209, file: !1175, line: 422, type: !1219, scopeLine: 422, flags: DIFlagPrototyped, spFlags: 0)
!1219 = !DISubroutineType(types: !1220)
!1220 = !{null, !1217, !1221}
!1221 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint3", file: !1175, line: 383, baseType: !1174)
!1222 = !DISubprogram(name: "operator uint3", linkageName: "_ZN4dim3cv5uint3Ev", scope: !1209, file: !1175, line: 423, type: !1223, scopeLine: 423, flags: DIFlagPrototyped, spFlags: 0)
!1223 = !DISubroutineType(types: !1224)
!1224 = !{!1221, !1217}
!1225 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !1226, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!1226 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !1201)
!1227 = !DISubprogram(name: "__cuda_builtin_blockDim_t", scope: !1201, file: !1163, line: 82, type: !1228, scopeLine: 82, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!1228 = !DISubroutineType(types: !1229)
!1229 = !{null, !1230}
!1230 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !1201, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!1231 = !DISubprogram(name: "__cuda_builtin_blockDim_t", scope: !1201, file: !1163, line: 82, type: !1232, scopeLine: 82, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!1232 = !DISubroutineType(types: !1233)
!1233 = !{null, !1230, !1234}
!1234 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !1226, size: 64)
!1235 = !DISubprogram(name: "operator=", linkageName: "_ZNK25__cuda_builtin_blockDim_taSERKS_", scope: !1201, file: !1163, line: 82, type: !1236, scopeLine: 82, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!1236 = !DISubroutineType(types: !1237)
!1237 = !{null, !1225, !1234}
!1238 = !DISubprogram(name: "operator&", linkageName: "_ZNK25__cuda_builtin_blockDim_tadEv", scope: !1201, file: !1163, line: 82, type: !1239, scopeLine: 82, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!1239 = !DISubroutineType(types: !1240)
!1240 = !{!1241, !1225}
!1241 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !1201, size: 64)
!1242 = distinct !DILocation(line: 776, column: 27, scope: !1152)
!1243 = !{i32 1, i32 1025}
!1244 = !DILocation(line: 776, column: 25, scope: !1152)
!1245 = !DILocation(line: 53, column: 3, scope: !1246, inlinedAt: !1272)
!1246 = distinct !DISubprogram(name: "__fetch_builtin_x", linkageName: "_ZN26__cuda_builtin_threadIdx_t17__fetch_builtin_xEv", scope: !1247, file: !1163, line: 53, type: !1167, scopeLine: 53, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !1050, declaration: !1249, retainedNodes: !1051)
!1247 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "__cuda_builtin_threadIdx_t", file: !1163, line: 52, size: 8, flags: DIFlagTypePassByReference | DIFlagNonTrivial, elements: !1248, identifier: "_ZTS26__cuda_builtin_threadIdx_t")
!1248 = !{!1249, !1250, !1251, !1252, !1257, !1261, !1265, !1268}
!1249 = !DISubprogram(name: "__fetch_builtin_x", linkageName: "_ZN26__cuda_builtin_threadIdx_t17__fetch_builtin_xEv", scope: !1247, file: !1163, line: 53, type: !1167, scopeLine: 53, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!1250 = !DISubprogram(name: "__fetch_builtin_y", linkageName: "_ZN26__cuda_builtin_threadIdx_t17__fetch_builtin_yEv", scope: !1247, file: !1163, line: 54, type: !1167, scopeLine: 54, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!1251 = !DISubprogram(name: "__fetch_builtin_z", linkageName: "_ZN26__cuda_builtin_threadIdx_t17__fetch_builtin_zEv", scope: !1247, file: !1163, line: 55, type: !1167, scopeLine: 55, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!1252 = !DISubprogram(name: "operator uint3", linkageName: "_ZNK26__cuda_builtin_threadIdx_tcv5uint3Ev", scope: !1247, file: !1163, line: 58, type: !1253, scopeLine: 58, flags: DIFlagPrototyped, spFlags: 0)
!1253 = !DISubroutineType(types: !1254)
!1254 = !{!1174, !1255}
!1255 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !1256, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!1256 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !1247)
!1257 = !DISubprogram(name: "__cuda_builtin_threadIdx_t", scope: !1247, file: !1163, line: 60, type: !1258, scopeLine: 60, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!1258 = !DISubroutineType(types: !1259)
!1259 = !{null, !1260}
!1260 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !1247, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!1261 = !DISubprogram(name: "__cuda_builtin_threadIdx_t", scope: !1247, file: !1163, line: 60, type: !1262, scopeLine: 60, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!1262 = !DISubroutineType(types: !1263)
!1263 = !{null, !1260, !1264}
!1264 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !1256, size: 64)
!1265 = !DISubprogram(name: "operator=", linkageName: "_ZNK26__cuda_builtin_threadIdx_taSERKS_", scope: !1247, file: !1163, line: 60, type: !1266, scopeLine: 60, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!1266 = !DISubroutineType(types: !1267)
!1267 = !{null, !1255, !1264}
!1268 = !DISubprogram(name: "operator&", linkageName: "_ZNK26__cuda_builtin_threadIdx_tadEv", scope: !1247, file: !1163, line: 60, type: !1269, scopeLine: 60, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!1269 = !DISubroutineType(types: !1270)
!1270 = !{!1271, !1255}
!1271 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !1247, size: 64)
!1272 = distinct !DILocation(line: 776, column: 40, scope: !1152)
!1273 = !{i32 0, i32 1024}
!1274 = !DILocation(line: 776, column: 38, scope: !1152)
!1275 = !DILocation(line: 777, column: 5, scope: !1276)
!1276 = distinct !DILexicalBlock(scope: !1152, file: !3, line: 777, column: 5)
!1277 = !DILocation(line: 777, column: 11, scope: !1276)
!1278 = !DILocation(line: 777, column: 5, scope: !1152)
!1279 = !DILocation(line: 778, column: 3, scope: !1280)
!1280 = distinct !DILexicalBlock(scope: !1276, file: !3, line: 777, column: 25)
!1281 = !DILocalVariable(name: "x", scope: !1152, file: !3, line: 780, type: !97)
!1282 = !DILocation(line: 780, column: 6, scope: !1152)
!1283 = !DILocation(line: 780, column: 10, scope: !1152)
!1284 = !DILocation(line: 780, column: 16, scope: !1152)
!1285 = !DILocalVariable(name: "y", scope: !1152, file: !3, line: 781, type: !97)
!1286 = !DILocation(line: 781, column: 6, scope: !1152)
!1287 = !DILocation(line: 781, column: 11, scope: !1152)
!1288 = !DILocation(line: 781, column: 17, scope: !1152)
!1289 = !DILocation(line: 781, column: 23, scope: !1152)
!1290 = !DILocalVariable(name: "z", scope: !1152, file: !3, line: 782, type: !97)
!1291 = !DILocation(line: 782, column: 6, scope: !1152)
!1292 = !DILocation(line: 782, column: 10, scope: !1152)
!1293 = !DILocation(line: 782, column: 16, scope: !1152)
!1294 = !DILocation(line: 783, column: 32, scope: !1152)
!1295 = !DILocation(line: 783, column: 37, scope: !1152)
!1296 = !DILocation(line: 783, column: 44, scope: !1152)
!1297 = !DILocation(line: 783, column: 2, scope: !1152)
!1298 = !DILocation(line: 783, column: 5, scope: !1152)
!1299 = !DILocation(line: 783, column: 8, scope: !1152)
!1300 = !DILocation(line: 783, column: 9, scope: !1152)
!1301 = !DILocation(line: 783, column: 6, scope: !1152)
!1302 = !DILocation(line: 783, column: 15, scope: !1152)
!1303 = !DILocation(line: 783, column: 16, scope: !1152)
!1304 = !DILocation(line: 783, column: 19, scope: !1152)
!1305 = !DILocation(line: 783, column: 13, scope: !1152)
!1306 = !DILocation(line: 783, column: 25, scope: !1152)
!1307 = !DILocation(line: 783, column: 30, scope: !1152)
!1308 = !DILocation(line: 784, column: 32, scope: !1152)
!1309 = !DILocation(line: 784, column: 37, scope: !1152)
!1310 = !DILocation(line: 784, column: 44, scope: !1152)
!1311 = !DILocation(line: 784, column: 2, scope: !1152)
!1312 = !DILocation(line: 784, column: 5, scope: !1152)
!1313 = !DILocation(line: 784, column: 8, scope: !1152)
!1314 = !DILocation(line: 784, column: 9, scope: !1152)
!1315 = !DILocation(line: 784, column: 6, scope: !1152)
!1316 = !DILocation(line: 784, column: 15, scope: !1152)
!1317 = !DILocation(line: 784, column: 16, scope: !1152)
!1318 = !DILocation(line: 784, column: 19, scope: !1152)
!1319 = !DILocation(line: 784, column: 13, scope: !1152)
!1320 = !DILocation(line: 784, column: 25, scope: !1152)
!1321 = !DILocation(line: 784, column: 30, scope: !1152)
!1322 = !DILocation(line: 785, column: 1, scope: !1152)
!1323 = distinct !DISubprogram(name: "cffts1_gpu_kernel_2", linkageName: "_Z19cffts1_gpu_kernel_2iP8dcomplexS0_S0_", scope: !3, file: !3, line: 792, type: !1324, scopeLine: 795, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !1050, retainedNodes: !1051)
!1324 = !DISubroutineType(types: !1325)
!1325 = !{null, !1326, !98, !98, !98}
!1326 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !97)
!1327 = !DILocalVariable(name: "is", arg: 1, scope: !1323, file: !3, line: 792, type: !1326)
!1328 = !DILocation(line: 792, column: 47, scope: !1323)
!1329 = !DILocalVariable(name: "gty1", arg: 2, scope: !1323, file: !3, line: 793, type: !98)
!1330 = !DILocation(line: 793, column: 12, scope: !1323)
!1331 = !DILocalVariable(name: "gty2", arg: 3, scope: !1323, file: !3, line: 794, type: !98)
!1332 = !DILocation(line: 794, column: 12, scope: !1323)
!1333 = !DILocalVariable(name: "u_device", arg: 4, scope: !1323, file: !3, line: 795, type: !98)
!1334 = !DILocation(line: 795, column: 12, scope: !1323)
!1335 = !DILocalVariable(name: "y_z", scope: !1323, file: !3, line: 796, type: !97)
!1336 = !DILocation(line: 796, column: 6, scope: !1323)
!1337 = !DILocation(line: 64, column: 3, scope: !1162, inlinedAt: !1338)
!1338 = distinct !DILocation(line: 796, column: 12, scope: !1323)
!1339 = !DILocation(line: 75, column: 3, scope: !1200, inlinedAt: !1340)
!1340 = distinct !DILocation(line: 796, column: 25, scope: !1323)
!1341 = !DILocation(line: 796, column: 23, scope: !1323)
!1342 = !DILocation(line: 53, column: 3, scope: !1246, inlinedAt: !1343)
!1343 = distinct !DILocation(line: 796, column: 38, scope: !1323)
!1344 = !DILocation(line: 796, column: 36, scope: !1323)
!1345 = !DILocation(line: 798, column: 5, scope: !1346)
!1346 = distinct !DILexicalBlock(scope: !1323, file: !3, line: 798, column: 5)
!1347 = !DILocation(line: 798, column: 9, scope: !1346)
!1348 = !DILocation(line: 798, column: 5, scope: !1323)
!1349 = !DILocation(line: 799, column: 3, scope: !1350)
!1350 = distinct !DILexicalBlock(scope: !1346, file: !3, line: 798, column: 20)
!1351 = !DILocalVariable(name: "j", scope: !1323, file: !3, line: 802, type: !97)
!1352 = !DILocation(line: 802, column: 6, scope: !1323)
!1353 = !DILocalVariable(name: "k", scope: !1323, file: !3, line: 802, type: !97)
!1354 = !DILocation(line: 802, column: 9, scope: !1323)
!1355 = !DILocalVariable(name: "l", scope: !1323, file: !3, line: 803, type: !97)
!1356 = !DILocation(line: 803, column: 6, scope: !1323)
!1357 = !DILocalVariable(name: "j1", scope: !1323, file: !3, line: 803, type: !97)
!1358 = !DILocation(line: 803, column: 9, scope: !1323)
!1359 = !DILocalVariable(name: "i1", scope: !1323, file: !3, line: 803, type: !97)
!1360 = !DILocation(line: 803, column: 13, scope: !1323)
!1361 = !DILocalVariable(name: "k1", scope: !1323, file: !3, line: 803, type: !97)
!1362 = !DILocation(line: 803, column: 17, scope: !1323)
!1363 = !DILocalVariable(name: "n1", scope: !1323, file: !3, line: 804, type: !97)
!1364 = !DILocation(line: 804, column: 6, scope: !1323)
!1365 = !DILocalVariable(name: "li", scope: !1323, file: !3, line: 804, type: !97)
!1366 = !DILocation(line: 804, column: 10, scope: !1323)
!1367 = !DILocalVariable(name: "lj", scope: !1323, file: !3, line: 804, type: !97)
!1368 = !DILocation(line: 804, column: 14, scope: !1323)
!1369 = !DILocalVariable(name: "lk", scope: !1323, file: !3, line: 804, type: !97)
!1370 = !DILocation(line: 804, column: 18, scope: !1323)
!1371 = !DILocalVariable(name: "ku", scope: !1323, file: !3, line: 804, type: !97)
!1372 = !DILocation(line: 804, column: 22, scope: !1323)
!1373 = !DILocalVariable(name: "i11", scope: !1323, file: !3, line: 804, type: !97)
!1374 = !DILocation(line: 804, column: 26, scope: !1323)
!1375 = !DILocalVariable(name: "i12", scope: !1323, file: !3, line: 804, type: !97)
!1376 = !DILocation(line: 804, column: 31, scope: !1323)
!1377 = !DILocalVariable(name: "i21", scope: !1323, file: !3, line: 804, type: !97)
!1378 = !DILocation(line: 804, column: 36, scope: !1323)
!1379 = !DILocalVariable(name: "i22", scope: !1323, file: !3, line: 804, type: !97)
!1380 = !DILocation(line: 804, column: 41, scope: !1323)
!1381 = !DILocation(line: 806, column: 6, scope: !1323)
!1382 = !DILocation(line: 806, column: 10, scope: !1323)
!1383 = !DILocation(line: 806, column: 4, scope: !1323)
!1384 = !DILocation(line: 807, column: 7, scope: !1323)
!1385 = !DILocation(line: 807, column: 11, scope: !1323)
!1386 = !DILocation(line: 807, column: 17, scope: !1323)
!1387 = !DILocation(line: 807, column: 4, scope: !1323)
!1388 = !DILocalVariable(name: "logd1", scope: !1323, file: !3, line: 809, type: !1326)
!1389 = !DILocation(line: 809, column: 12, scope: !1323)
!1390 = !DILocation(line: 809, column: 20, scope: !1323)
!1391 = !DILocalVariable(name: "uu1_real", scope: !1323, file: !3, line: 811, type: !104)
!1392 = !DILocation(line: 811, column: 9, scope: !1323)
!1393 = !DILocalVariable(name: "x11_real", scope: !1323, file: !3, line: 811, type: !104)
!1394 = !DILocation(line: 811, column: 19, scope: !1323)
!1395 = !DILocalVariable(name: "x21_real", scope: !1323, file: !3, line: 811, type: !104)
!1396 = !DILocation(line: 811, column: 29, scope: !1323)
!1397 = !DILocalVariable(name: "uu1_imag", scope: !1323, file: !3, line: 812, type: !104)
!1398 = !DILocation(line: 812, column: 9, scope: !1323)
!1399 = !DILocalVariable(name: "x11_imag", scope: !1323, file: !3, line: 812, type: !104)
!1400 = !DILocation(line: 812, column: 19, scope: !1323)
!1401 = !DILocalVariable(name: "x21_imag", scope: !1323, file: !3, line: 812, type: !104)
!1402 = !DILocation(line: 812, column: 29, scope: !1323)
!1403 = !DILocalVariable(name: "uu2_real", scope: !1323, file: !3, line: 813, type: !104)
!1404 = !DILocation(line: 813, column: 9, scope: !1323)
!1405 = !DILocalVariable(name: "x12_real", scope: !1323, file: !3, line: 813, type: !104)
!1406 = !DILocation(line: 813, column: 19, scope: !1323)
!1407 = !DILocalVariable(name: "x22_real", scope: !1323, file: !3, line: 813, type: !104)
!1408 = !DILocation(line: 813, column: 29, scope: !1323)
!1409 = !DILocalVariable(name: "uu2_imag", scope: !1323, file: !3, line: 814, type: !104)
!1410 = !DILocation(line: 814, column: 9, scope: !1323)
!1411 = !DILocalVariable(name: "x12_imag", scope: !1323, file: !3, line: 814, type: !104)
!1412 = !DILocation(line: 814, column: 19, scope: !1323)
!1413 = !DILocalVariable(name: "x22_imag", scope: !1323, file: !3, line: 814, type: !104)
!1414 = !DILocation(line: 814, column: 29, scope: !1323)
!1415 = !DILocalVariable(name: "temp_real", scope: !1323, file: !3, line: 815, type: !104)
!1416 = !DILocation(line: 815, column: 9, scope: !1323)
!1417 = !DILocalVariable(name: "temp2_real", scope: !1323, file: !3, line: 815, type: !104)
!1418 = !DILocation(line: 815, column: 20, scope: !1323)
!1419 = !DILocalVariable(name: "temp_imag", scope: !1323, file: !3, line: 816, type: !104)
!1420 = !DILocation(line: 816, column: 9, scope: !1323)
!1421 = !DILocalVariable(name: "temp2_imag", scope: !1323, file: !3, line: 816, type: !104)
!1422 = !DILocation(line: 816, column: 20, scope: !1323)
!1423 = !DILocation(line: 818, column: 7, scope: !1424)
!1424 = distinct !DILexicalBlock(scope: !1323, file: !3, line: 818, column: 2)
!1425 = !DILocation(line: 818, column: 6, scope: !1424)
!1426 = !DILocation(line: 818, column: 11, scope: !1427)
!1427 = distinct !DILexicalBlock(scope: !1424, file: !3, line: 818, column: 2)
!1428 = !DILocation(line: 818, column: 14, scope: !1427)
!1429 = !DILocation(line: 818, column: 12, scope: !1427)
!1430 = !DILocation(line: 818, column: 2, scope: !1424)
!1431 = !DILocation(line: 819, column: 6, scope: !1432)
!1432 = distinct !DILexicalBlock(scope: !1427, file: !3, line: 818, column: 26)
!1433 = !DILocation(line: 820, column: 14, scope: !1432)
!1434 = !DILocation(line: 820, column: 16, scope: !1432)
!1435 = !DILocation(line: 820, column: 10, scope: !1432)
!1436 = !DILocation(line: 820, column: 6, scope: !1432)
!1437 = !DILocation(line: 821, column: 14, scope: !1432)
!1438 = !DILocation(line: 821, column: 22, scope: !1432)
!1439 = !DILocation(line: 821, column: 20, scope: !1432)
!1440 = !DILocation(line: 821, column: 10, scope: !1432)
!1441 = !DILocation(line: 821, column: 6, scope: !1432)
!1442 = !DILocation(line: 822, column: 12, scope: !1432)
!1443 = !DILocation(line: 822, column: 10, scope: !1432)
!1444 = !DILocation(line: 822, column: 6, scope: !1432)
!1445 = !DILocation(line: 823, column: 8, scope: !1432)
!1446 = !DILocation(line: 823, column: 6, scope: !1432)
!1447 = !DILocation(line: 824, column: 9, scope: !1448)
!1448 = distinct !DILexicalBlock(scope: !1432, file: !3, line: 824, column: 3)
!1449 = !DILocation(line: 824, column: 7, scope: !1448)
!1450 = !DILocation(line: 824, column: 13, scope: !1451)
!1451 = distinct !DILexicalBlock(scope: !1448, file: !3, line: 824, column: 3)
!1452 = !DILocation(line: 824, column: 17, scope: !1451)
!1453 = !DILocation(line: 824, column: 19, scope: !1451)
!1454 = !DILocation(line: 824, column: 15, scope: !1451)
!1455 = !DILocation(line: 824, column: 3, scope: !1448)
!1456 = !DILocation(line: 825, column: 10, scope: !1457)
!1457 = distinct !DILexicalBlock(scope: !1458, file: !3, line: 825, column: 4)
!1458 = distinct !DILexicalBlock(scope: !1451, file: !3, line: 824, column: 28)
!1459 = !DILocation(line: 825, column: 8, scope: !1457)
!1460 = !DILocation(line: 825, column: 14, scope: !1461)
!1461 = distinct !DILexicalBlock(scope: !1457, file: !3, line: 825, column: 4)
!1462 = !DILocation(line: 825, column: 18, scope: !1461)
!1463 = !DILocation(line: 825, column: 20, scope: !1461)
!1464 = !DILocation(line: 825, column: 16, scope: !1461)
!1465 = !DILocation(line: 825, column: 4, scope: !1457)
!1466 = !DILocation(line: 826, column: 11, scope: !1467)
!1467 = distinct !DILexicalBlock(scope: !1461, file: !3, line: 825, column: 29)
!1468 = !DILocation(line: 826, column: 16, scope: !1467)
!1469 = !DILocation(line: 826, column: 14, scope: !1467)
!1470 = !DILocation(line: 826, column: 9, scope: !1467)
!1471 = !DILocation(line: 827, column: 11, scope: !1467)
!1472 = !DILocation(line: 827, column: 17, scope: !1467)
!1473 = !DILocation(line: 827, column: 15, scope: !1467)
!1474 = !DILocation(line: 827, column: 9, scope: !1467)
!1475 = !DILocation(line: 828, column: 11, scope: !1467)
!1476 = !DILocation(line: 828, column: 16, scope: !1467)
!1477 = !DILocation(line: 828, column: 14, scope: !1467)
!1478 = !DILocation(line: 828, column: 9, scope: !1467)
!1479 = !DILocation(line: 829, column: 11, scope: !1467)
!1480 = !DILocation(line: 829, column: 17, scope: !1467)
!1481 = !DILocation(line: 829, column: 15, scope: !1467)
!1482 = !DILocation(line: 829, column: 9, scope: !1467)
!1483 = !DILocation(line: 831, column: 16, scope: !1467)
!1484 = !DILocation(line: 831, column: 25, scope: !1467)
!1485 = !DILocation(line: 831, column: 28, scope: !1467)
!1486 = !DILocation(line: 831, column: 27, scope: !1467)
!1487 = !DILocation(line: 831, column: 32, scope: !1467)
!1488 = !DILocation(line: 831, column: 14, scope: !1467)
!1489 = !DILocation(line: 832, column: 16, scope: !1467)
!1490 = !DILocation(line: 832, column: 19, scope: !1467)
!1491 = !DILocation(line: 832, column: 28, scope: !1467)
!1492 = !DILocation(line: 832, column: 31, scope: !1467)
!1493 = !DILocation(line: 832, column: 30, scope: !1467)
!1494 = !DILocation(line: 832, column: 35, scope: !1467)
!1495 = !DILocation(line: 832, column: 18, scope: !1467)
!1496 = !DILocation(line: 832, column: 14, scope: !1467)
!1497 = !DILocation(line: 835, column: 16, scope: !1467)
!1498 = !DILocation(line: 835, column: 21, scope: !1467)
!1499 = !DILocation(line: 835, column: 26, scope: !1467)
!1500 = !DILocation(line: 835, column: 30, scope: !1467)
!1501 = !DILocation(line: 835, column: 29, scope: !1467)
!1502 = !DILocation(line: 835, column: 33, scope: !1467)
!1503 = !DILocation(line: 835, column: 23, scope: !1467)
!1504 = !DILocation(line: 835, column: 39, scope: !1467)
!1505 = !DILocation(line: 835, column: 40, scope: !1467)
!1506 = !DILocation(line: 835, column: 43, scope: !1467)
!1507 = !DILocation(line: 835, column: 37, scope: !1467)
!1508 = !DILocation(line: 835, column: 48, scope: !1467)
!1509 = !DILocation(line: 835, column: 14, scope: !1467)
!1510 = !DILocation(line: 836, column: 16, scope: !1467)
!1511 = !DILocation(line: 836, column: 21, scope: !1467)
!1512 = !DILocation(line: 836, column: 26, scope: !1467)
!1513 = !DILocation(line: 836, column: 30, scope: !1467)
!1514 = !DILocation(line: 836, column: 29, scope: !1467)
!1515 = !DILocation(line: 836, column: 33, scope: !1467)
!1516 = !DILocation(line: 836, column: 23, scope: !1467)
!1517 = !DILocation(line: 836, column: 39, scope: !1467)
!1518 = !DILocation(line: 836, column: 40, scope: !1467)
!1519 = !DILocation(line: 836, column: 43, scope: !1467)
!1520 = !DILocation(line: 836, column: 37, scope: !1467)
!1521 = !DILocation(line: 836, column: 48, scope: !1467)
!1522 = !DILocation(line: 836, column: 14, scope: !1467)
!1523 = !DILocation(line: 839, column: 16, scope: !1467)
!1524 = !DILocation(line: 839, column: 21, scope: !1467)
!1525 = !DILocation(line: 839, column: 26, scope: !1467)
!1526 = !DILocation(line: 839, column: 30, scope: !1467)
!1527 = !DILocation(line: 839, column: 29, scope: !1467)
!1528 = !DILocation(line: 839, column: 33, scope: !1467)
!1529 = !DILocation(line: 839, column: 23, scope: !1467)
!1530 = !DILocation(line: 839, column: 39, scope: !1467)
!1531 = !DILocation(line: 839, column: 40, scope: !1467)
!1532 = !DILocation(line: 839, column: 43, scope: !1467)
!1533 = !DILocation(line: 839, column: 37, scope: !1467)
!1534 = !DILocation(line: 839, column: 48, scope: !1467)
!1535 = !DILocation(line: 839, column: 14, scope: !1467)
!1536 = !DILocation(line: 840, column: 16, scope: !1467)
!1537 = !DILocation(line: 840, column: 21, scope: !1467)
!1538 = !DILocation(line: 840, column: 26, scope: !1467)
!1539 = !DILocation(line: 840, column: 30, scope: !1467)
!1540 = !DILocation(line: 840, column: 29, scope: !1467)
!1541 = !DILocation(line: 840, column: 33, scope: !1467)
!1542 = !DILocation(line: 840, column: 23, scope: !1467)
!1543 = !DILocation(line: 840, column: 39, scope: !1467)
!1544 = !DILocation(line: 840, column: 40, scope: !1467)
!1545 = !DILocation(line: 840, column: 43, scope: !1467)
!1546 = !DILocation(line: 840, column: 37, scope: !1467)
!1547 = !DILocation(line: 840, column: 48, scope: !1467)
!1548 = !DILocation(line: 840, column: 14, scope: !1467)
!1549 = !DILocation(line: 843, column: 44, scope: !1467)
!1550 = !DILocation(line: 843, column: 55, scope: !1467)
!1551 = !DILocation(line: 843, column: 53, scope: !1467)
!1552 = !DILocation(line: 843, column: 5, scope: !1467)
!1553 = !DILocation(line: 843, column: 10, scope: !1467)
!1554 = !DILocation(line: 843, column: 15, scope: !1467)
!1555 = !DILocation(line: 843, column: 19, scope: !1467)
!1556 = !DILocation(line: 843, column: 18, scope: !1467)
!1557 = !DILocation(line: 843, column: 22, scope: !1467)
!1558 = !DILocation(line: 843, column: 12, scope: !1467)
!1559 = !DILocation(line: 843, column: 28, scope: !1467)
!1560 = !DILocation(line: 843, column: 29, scope: !1467)
!1561 = !DILocation(line: 843, column: 32, scope: !1467)
!1562 = !DILocation(line: 843, column: 26, scope: !1467)
!1563 = !DILocation(line: 843, column: 37, scope: !1467)
!1564 = !DILocation(line: 843, column: 42, scope: !1467)
!1565 = !DILocation(line: 844, column: 44, scope: !1467)
!1566 = !DILocation(line: 844, column: 55, scope: !1467)
!1567 = !DILocation(line: 844, column: 53, scope: !1467)
!1568 = !DILocation(line: 844, column: 5, scope: !1467)
!1569 = !DILocation(line: 844, column: 10, scope: !1467)
!1570 = !DILocation(line: 844, column: 15, scope: !1467)
!1571 = !DILocation(line: 844, column: 19, scope: !1467)
!1572 = !DILocation(line: 844, column: 18, scope: !1467)
!1573 = !DILocation(line: 844, column: 22, scope: !1467)
!1574 = !DILocation(line: 844, column: 12, scope: !1467)
!1575 = !DILocation(line: 844, column: 28, scope: !1467)
!1576 = !DILocation(line: 844, column: 29, scope: !1467)
!1577 = !DILocation(line: 844, column: 32, scope: !1467)
!1578 = !DILocation(line: 844, column: 26, scope: !1467)
!1579 = !DILocation(line: 844, column: 37, scope: !1467)
!1580 = !DILocation(line: 844, column: 42, scope: !1467)
!1581 = !DILocation(line: 846, column: 17, scope: !1467)
!1582 = !DILocation(line: 846, column: 28, scope: !1467)
!1583 = !DILocation(line: 846, column: 26, scope: !1467)
!1584 = !DILocation(line: 846, column: 15, scope: !1467)
!1585 = !DILocation(line: 847, column: 17, scope: !1467)
!1586 = !DILocation(line: 847, column: 28, scope: !1467)
!1587 = !DILocation(line: 847, column: 26, scope: !1467)
!1588 = !DILocation(line: 847, column: 15, scope: !1467)
!1589 = !DILocation(line: 850, column: 45, scope: !1467)
!1590 = !DILocation(line: 850, column: 56, scope: !1467)
!1591 = !DILocation(line: 850, column: 54, scope: !1467)
!1592 = !DILocation(line: 850, column: 70, scope: !1467)
!1593 = !DILocation(line: 850, column: 81, scope: !1467)
!1594 = !DILocation(line: 850, column: 79, scope: !1467)
!1595 = !DILocation(line: 850, column: 67, scope: !1467)
!1596 = !DILocation(line: 850, column: 5, scope: !1467)
!1597 = !DILocation(line: 850, column: 10, scope: !1467)
!1598 = !DILocation(line: 850, column: 15, scope: !1467)
!1599 = !DILocation(line: 850, column: 19, scope: !1467)
!1600 = !DILocation(line: 850, column: 18, scope: !1467)
!1601 = !DILocation(line: 850, column: 22, scope: !1467)
!1602 = !DILocation(line: 850, column: 12, scope: !1467)
!1603 = !DILocation(line: 850, column: 28, scope: !1467)
!1604 = !DILocation(line: 850, column: 29, scope: !1467)
!1605 = !DILocation(line: 850, column: 32, scope: !1467)
!1606 = !DILocation(line: 850, column: 26, scope: !1467)
!1607 = !DILocation(line: 850, column: 37, scope: !1467)
!1608 = !DILocation(line: 850, column: 42, scope: !1467)
!1609 = !DILocation(line: 851, column: 45, scope: !1467)
!1610 = !DILocation(line: 851, column: 56, scope: !1467)
!1611 = !DILocation(line: 851, column: 54, scope: !1467)
!1612 = !DILocation(line: 851, column: 70, scope: !1467)
!1613 = !DILocation(line: 851, column: 81, scope: !1467)
!1614 = !DILocation(line: 851, column: 79, scope: !1467)
!1615 = !DILocation(line: 851, column: 67, scope: !1467)
!1616 = !DILocation(line: 851, column: 5, scope: !1467)
!1617 = !DILocation(line: 851, column: 10, scope: !1467)
!1618 = !DILocation(line: 851, column: 15, scope: !1467)
!1619 = !DILocation(line: 851, column: 19, scope: !1467)
!1620 = !DILocation(line: 851, column: 18, scope: !1467)
!1621 = !DILocation(line: 851, column: 22, scope: !1467)
!1622 = !DILocation(line: 851, column: 12, scope: !1467)
!1623 = !DILocation(line: 851, column: 28, scope: !1467)
!1624 = !DILocation(line: 851, column: 29, scope: !1467)
!1625 = !DILocation(line: 851, column: 32, scope: !1467)
!1626 = !DILocation(line: 851, column: 26, scope: !1467)
!1627 = !DILocation(line: 851, column: 37, scope: !1467)
!1628 = !DILocation(line: 851, column: 42, scope: !1467)
!1629 = !DILocation(line: 852, column: 4, scope: !1467)
!1630 = !DILocation(line: 825, column: 26, scope: !1461)
!1631 = !DILocation(line: 825, column: 4, scope: !1461)
!1632 = distinct !{!1632, !1465, !1633}
!1633 = !DILocation(line: 852, column: 4, scope: !1457)
!1634 = !DILocation(line: 853, column: 3, scope: !1458)
!1635 = !DILocation(line: 824, column: 25, scope: !1451)
!1636 = !DILocation(line: 824, column: 3, scope: !1451)
!1637 = distinct !{!1637, !1455, !1638}
!1638 = !DILocation(line: 853, column: 3, scope: !1448)
!1639 = !DILocation(line: 854, column: 6, scope: !1640)
!1640 = distinct !DILexicalBlock(scope: !1432, file: !3, line: 854, column: 6)
!1641 = !DILocation(line: 854, column: 9, scope: !1640)
!1642 = !DILocation(line: 854, column: 7, scope: !1640)
!1643 = !DILocation(line: 854, column: 6, scope: !1432)
!1644 = !DILocation(line: 855, column: 10, scope: !1645)
!1645 = distinct !DILexicalBlock(scope: !1646, file: !3, line: 855, column: 4)
!1646 = distinct !DILexicalBlock(scope: !1640, file: !3, line: 854, column: 15)
!1647 = !DILocation(line: 855, column: 8, scope: !1645)
!1648 = !DILocation(line: 855, column: 14, scope: !1649)
!1649 = distinct !DILexicalBlock(scope: !1645, file: !3, line: 855, column: 4)
!1650 = !DILocation(line: 855, column: 16, scope: !1649)
!1651 = !DILocation(line: 855, column: 4, scope: !1645)
!1652 = !DILocation(line: 857, column: 38, scope: !1653)
!1653 = distinct !DILexicalBlock(scope: !1649, file: !3, line: 855, column: 26)
!1654 = !DILocation(line: 857, column: 43, scope: !1653)
!1655 = !DILocation(line: 857, column: 47, scope: !1653)
!1656 = !DILocation(line: 857, column: 49, scope: !1653)
!1657 = !DILocation(line: 857, column: 45, scope: !1653)
!1658 = !DILocation(line: 857, column: 55, scope: !1653)
!1659 = !DILocation(line: 857, column: 56, scope: !1653)
!1660 = !DILocation(line: 857, column: 59, scope: !1653)
!1661 = !DILocation(line: 857, column: 53, scope: !1653)
!1662 = !DILocation(line: 857, column: 64, scope: !1653)
!1663 = !DILocation(line: 857, column: 5, scope: !1653)
!1664 = !DILocation(line: 857, column: 10, scope: !1653)
!1665 = !DILocation(line: 857, column: 14, scope: !1653)
!1666 = !DILocation(line: 857, column: 16, scope: !1653)
!1667 = !DILocation(line: 857, column: 12, scope: !1653)
!1668 = !DILocation(line: 857, column: 22, scope: !1653)
!1669 = !DILocation(line: 857, column: 23, scope: !1653)
!1670 = !DILocation(line: 857, column: 26, scope: !1653)
!1671 = !DILocation(line: 857, column: 20, scope: !1653)
!1672 = !DILocation(line: 857, column: 31, scope: !1653)
!1673 = !DILocation(line: 857, column: 36, scope: !1653)
!1674 = !DILocation(line: 858, column: 38, scope: !1653)
!1675 = !DILocation(line: 858, column: 43, scope: !1653)
!1676 = !DILocation(line: 858, column: 47, scope: !1653)
!1677 = !DILocation(line: 858, column: 49, scope: !1653)
!1678 = !DILocation(line: 858, column: 45, scope: !1653)
!1679 = !DILocation(line: 858, column: 55, scope: !1653)
!1680 = !DILocation(line: 858, column: 56, scope: !1653)
!1681 = !DILocation(line: 858, column: 59, scope: !1653)
!1682 = !DILocation(line: 858, column: 53, scope: !1653)
!1683 = !DILocation(line: 858, column: 64, scope: !1653)
!1684 = !DILocation(line: 858, column: 5, scope: !1653)
!1685 = !DILocation(line: 858, column: 10, scope: !1653)
!1686 = !DILocation(line: 858, column: 14, scope: !1653)
!1687 = !DILocation(line: 858, column: 16, scope: !1653)
!1688 = !DILocation(line: 858, column: 12, scope: !1653)
!1689 = !DILocation(line: 858, column: 22, scope: !1653)
!1690 = !DILocation(line: 858, column: 23, scope: !1653)
!1691 = !DILocation(line: 858, column: 26, scope: !1653)
!1692 = !DILocation(line: 858, column: 20, scope: !1653)
!1693 = !DILocation(line: 858, column: 31, scope: !1653)
!1694 = !DILocation(line: 858, column: 36, scope: !1653)
!1695 = !DILocation(line: 859, column: 4, scope: !1653)
!1696 = !DILocation(line: 855, column: 23, scope: !1649)
!1697 = !DILocation(line: 855, column: 4, scope: !1649)
!1698 = distinct !{!1698, !1651, !1699}
!1699 = !DILocation(line: 859, column: 4, scope: !1645)
!1700 = !DILocation(line: 860, column: 3, scope: !1646)
!1701 = !DILocation(line: 861, column: 7, scope: !1702)
!1702 = distinct !DILexicalBlock(scope: !1640, file: !3, line: 860, column: 8)
!1703 = !DILocation(line: 862, column: 15, scope: !1702)
!1704 = !DILocation(line: 862, column: 16, scope: !1702)
!1705 = !DILocation(line: 862, column: 19, scope: !1702)
!1706 = !DILocation(line: 862, column: 11, scope: !1702)
!1707 = !DILocation(line: 862, column: 7, scope: !1702)
!1708 = !DILocation(line: 863, column: 15, scope: !1702)
!1709 = !DILocation(line: 863, column: 24, scope: !1702)
!1710 = !DILocation(line: 863, column: 25, scope: !1702)
!1711 = !DILocation(line: 863, column: 21, scope: !1702)
!1712 = !DILocation(line: 863, column: 11, scope: !1702)
!1713 = !DILocation(line: 863, column: 7, scope: !1702)
!1714 = !DILocation(line: 864, column: 13, scope: !1702)
!1715 = !DILocation(line: 864, column: 11, scope: !1702)
!1716 = !DILocation(line: 864, column: 7, scope: !1702)
!1717 = !DILocation(line: 865, column: 9, scope: !1702)
!1718 = !DILocation(line: 865, column: 7, scope: !1702)
!1719 = !DILocation(line: 866, column: 10, scope: !1720)
!1720 = distinct !DILexicalBlock(scope: !1702, file: !3, line: 866, column: 4)
!1721 = !DILocation(line: 866, column: 8, scope: !1720)
!1722 = !DILocation(line: 866, column: 14, scope: !1723)
!1723 = distinct !DILexicalBlock(scope: !1720, file: !3, line: 866, column: 4)
!1724 = !DILocation(line: 866, column: 18, scope: !1723)
!1725 = !DILocation(line: 866, column: 20, scope: !1723)
!1726 = !DILocation(line: 866, column: 16, scope: !1723)
!1727 = !DILocation(line: 866, column: 4, scope: !1720)
!1728 = !DILocation(line: 867, column: 11, scope: !1729)
!1729 = distinct !DILexicalBlock(scope: !1730, file: !3, line: 867, column: 5)
!1730 = distinct !DILexicalBlock(scope: !1723, file: !3, line: 866, column: 29)
!1731 = !DILocation(line: 867, column: 9, scope: !1729)
!1732 = !DILocation(line: 867, column: 15, scope: !1733)
!1733 = distinct !DILexicalBlock(scope: !1729, file: !3, line: 867, column: 5)
!1734 = !DILocation(line: 867, column: 19, scope: !1733)
!1735 = !DILocation(line: 867, column: 21, scope: !1733)
!1736 = !DILocation(line: 867, column: 17, scope: !1733)
!1737 = !DILocation(line: 867, column: 5, scope: !1729)
!1738 = !DILocation(line: 868, column: 12, scope: !1739)
!1739 = distinct !DILexicalBlock(scope: !1733, file: !3, line: 867, column: 30)
!1740 = !DILocation(line: 868, column: 17, scope: !1739)
!1741 = !DILocation(line: 868, column: 15, scope: !1739)
!1742 = !DILocation(line: 868, column: 10, scope: !1739)
!1743 = !DILocation(line: 869, column: 12, scope: !1739)
!1744 = !DILocation(line: 869, column: 18, scope: !1739)
!1745 = !DILocation(line: 869, column: 16, scope: !1739)
!1746 = !DILocation(line: 869, column: 10, scope: !1739)
!1747 = !DILocation(line: 870, column: 12, scope: !1739)
!1748 = !DILocation(line: 870, column: 17, scope: !1739)
!1749 = !DILocation(line: 870, column: 15, scope: !1739)
!1750 = !DILocation(line: 870, column: 10, scope: !1739)
!1751 = !DILocation(line: 871, column: 12, scope: !1739)
!1752 = !DILocation(line: 871, column: 18, scope: !1739)
!1753 = !DILocation(line: 871, column: 16, scope: !1739)
!1754 = !DILocation(line: 871, column: 10, scope: !1739)
!1755 = !DILocation(line: 873, column: 17, scope: !1739)
!1756 = !DILocation(line: 873, column: 26, scope: !1739)
!1757 = !DILocation(line: 873, column: 29, scope: !1739)
!1758 = !DILocation(line: 873, column: 28, scope: !1739)
!1759 = !DILocation(line: 873, column: 33, scope: !1739)
!1760 = !DILocation(line: 873, column: 15, scope: !1739)
!1761 = !DILocation(line: 874, column: 17, scope: !1739)
!1762 = !DILocation(line: 874, column: 20, scope: !1739)
!1763 = !DILocation(line: 874, column: 29, scope: !1739)
!1764 = !DILocation(line: 874, column: 32, scope: !1739)
!1765 = !DILocation(line: 874, column: 31, scope: !1739)
!1766 = !DILocation(line: 874, column: 36, scope: !1739)
!1767 = !DILocation(line: 874, column: 19, scope: !1739)
!1768 = !DILocation(line: 874, column: 15, scope: !1739)
!1769 = !DILocation(line: 877, column: 17, scope: !1739)
!1770 = !DILocation(line: 877, column: 22, scope: !1739)
!1771 = !DILocation(line: 877, column: 27, scope: !1739)
!1772 = !DILocation(line: 877, column: 31, scope: !1739)
!1773 = !DILocation(line: 877, column: 30, scope: !1739)
!1774 = !DILocation(line: 877, column: 34, scope: !1739)
!1775 = !DILocation(line: 877, column: 24, scope: !1739)
!1776 = !DILocation(line: 877, column: 40, scope: !1739)
!1777 = !DILocation(line: 877, column: 41, scope: !1739)
!1778 = !DILocation(line: 877, column: 44, scope: !1739)
!1779 = !DILocation(line: 877, column: 38, scope: !1739)
!1780 = !DILocation(line: 877, column: 49, scope: !1739)
!1781 = !DILocation(line: 877, column: 15, scope: !1739)
!1782 = !DILocation(line: 878, column: 17, scope: !1739)
!1783 = !DILocation(line: 878, column: 22, scope: !1739)
!1784 = !DILocation(line: 878, column: 27, scope: !1739)
!1785 = !DILocation(line: 878, column: 31, scope: !1739)
!1786 = !DILocation(line: 878, column: 30, scope: !1739)
!1787 = !DILocation(line: 878, column: 34, scope: !1739)
!1788 = !DILocation(line: 878, column: 24, scope: !1739)
!1789 = !DILocation(line: 878, column: 40, scope: !1739)
!1790 = !DILocation(line: 878, column: 41, scope: !1739)
!1791 = !DILocation(line: 878, column: 44, scope: !1739)
!1792 = !DILocation(line: 878, column: 38, scope: !1739)
!1793 = !DILocation(line: 878, column: 49, scope: !1739)
!1794 = !DILocation(line: 878, column: 15, scope: !1739)
!1795 = !DILocation(line: 881, column: 17, scope: !1739)
!1796 = !DILocation(line: 881, column: 22, scope: !1739)
!1797 = !DILocation(line: 881, column: 27, scope: !1739)
!1798 = !DILocation(line: 881, column: 31, scope: !1739)
!1799 = !DILocation(line: 881, column: 30, scope: !1739)
!1800 = !DILocation(line: 881, column: 34, scope: !1739)
!1801 = !DILocation(line: 881, column: 24, scope: !1739)
!1802 = !DILocation(line: 881, column: 40, scope: !1739)
!1803 = !DILocation(line: 881, column: 41, scope: !1739)
!1804 = !DILocation(line: 881, column: 44, scope: !1739)
!1805 = !DILocation(line: 881, column: 38, scope: !1739)
!1806 = !DILocation(line: 881, column: 49, scope: !1739)
!1807 = !DILocation(line: 881, column: 15, scope: !1739)
!1808 = !DILocation(line: 882, column: 17, scope: !1739)
!1809 = !DILocation(line: 882, column: 22, scope: !1739)
!1810 = !DILocation(line: 882, column: 27, scope: !1739)
!1811 = !DILocation(line: 882, column: 31, scope: !1739)
!1812 = !DILocation(line: 882, column: 30, scope: !1739)
!1813 = !DILocation(line: 882, column: 34, scope: !1739)
!1814 = !DILocation(line: 882, column: 24, scope: !1739)
!1815 = !DILocation(line: 882, column: 40, scope: !1739)
!1816 = !DILocation(line: 882, column: 41, scope: !1739)
!1817 = !DILocation(line: 882, column: 44, scope: !1739)
!1818 = !DILocation(line: 882, column: 38, scope: !1739)
!1819 = !DILocation(line: 882, column: 49, scope: !1739)
!1820 = !DILocation(line: 882, column: 15, scope: !1739)
!1821 = !DILocation(line: 885, column: 45, scope: !1739)
!1822 = !DILocation(line: 885, column: 56, scope: !1739)
!1823 = !DILocation(line: 885, column: 54, scope: !1739)
!1824 = !DILocation(line: 885, column: 6, scope: !1739)
!1825 = !DILocation(line: 885, column: 11, scope: !1739)
!1826 = !DILocation(line: 885, column: 16, scope: !1739)
!1827 = !DILocation(line: 885, column: 20, scope: !1739)
!1828 = !DILocation(line: 885, column: 19, scope: !1739)
!1829 = !DILocation(line: 885, column: 23, scope: !1739)
!1830 = !DILocation(line: 885, column: 13, scope: !1739)
!1831 = !DILocation(line: 885, column: 29, scope: !1739)
!1832 = !DILocation(line: 885, column: 30, scope: !1739)
!1833 = !DILocation(line: 885, column: 33, scope: !1739)
!1834 = !DILocation(line: 885, column: 27, scope: !1739)
!1835 = !DILocation(line: 885, column: 38, scope: !1739)
!1836 = !DILocation(line: 885, column: 43, scope: !1739)
!1837 = !DILocation(line: 886, column: 45, scope: !1739)
!1838 = !DILocation(line: 886, column: 56, scope: !1739)
!1839 = !DILocation(line: 886, column: 54, scope: !1739)
!1840 = !DILocation(line: 886, column: 6, scope: !1739)
!1841 = !DILocation(line: 886, column: 11, scope: !1739)
!1842 = !DILocation(line: 886, column: 16, scope: !1739)
!1843 = !DILocation(line: 886, column: 20, scope: !1739)
!1844 = !DILocation(line: 886, column: 19, scope: !1739)
!1845 = !DILocation(line: 886, column: 23, scope: !1739)
!1846 = !DILocation(line: 886, column: 13, scope: !1739)
!1847 = !DILocation(line: 886, column: 29, scope: !1739)
!1848 = !DILocation(line: 886, column: 30, scope: !1739)
!1849 = !DILocation(line: 886, column: 33, scope: !1739)
!1850 = !DILocation(line: 886, column: 27, scope: !1739)
!1851 = !DILocation(line: 886, column: 38, scope: !1739)
!1852 = !DILocation(line: 886, column: 43, scope: !1739)
!1853 = !DILocation(line: 888, column: 19, scope: !1739)
!1854 = !DILocation(line: 888, column: 30, scope: !1739)
!1855 = !DILocation(line: 888, column: 28, scope: !1739)
!1856 = !DILocation(line: 888, column: 17, scope: !1739)
!1857 = !DILocation(line: 889, column: 19, scope: !1739)
!1858 = !DILocation(line: 889, column: 30, scope: !1739)
!1859 = !DILocation(line: 889, column: 28, scope: !1739)
!1860 = !DILocation(line: 889, column: 17, scope: !1739)
!1861 = !DILocation(line: 892, column: 46, scope: !1739)
!1862 = !DILocation(line: 892, column: 57, scope: !1739)
!1863 = !DILocation(line: 892, column: 55, scope: !1739)
!1864 = !DILocation(line: 892, column: 72, scope: !1739)
!1865 = !DILocation(line: 892, column: 83, scope: !1739)
!1866 = !DILocation(line: 892, column: 81, scope: !1739)
!1867 = !DILocation(line: 892, column: 69, scope: !1739)
!1868 = !DILocation(line: 892, column: 6, scope: !1739)
!1869 = !DILocation(line: 892, column: 11, scope: !1739)
!1870 = !DILocation(line: 892, column: 16, scope: !1739)
!1871 = !DILocation(line: 892, column: 20, scope: !1739)
!1872 = !DILocation(line: 892, column: 19, scope: !1739)
!1873 = !DILocation(line: 892, column: 23, scope: !1739)
!1874 = !DILocation(line: 892, column: 13, scope: !1739)
!1875 = !DILocation(line: 892, column: 29, scope: !1739)
!1876 = !DILocation(line: 892, column: 30, scope: !1739)
!1877 = !DILocation(line: 892, column: 33, scope: !1739)
!1878 = !DILocation(line: 892, column: 27, scope: !1739)
!1879 = !DILocation(line: 892, column: 38, scope: !1739)
!1880 = !DILocation(line: 892, column: 43, scope: !1739)
!1881 = !DILocation(line: 893, column: 46, scope: !1739)
!1882 = !DILocation(line: 893, column: 57, scope: !1739)
!1883 = !DILocation(line: 893, column: 55, scope: !1739)
!1884 = !DILocation(line: 893, column: 72, scope: !1739)
!1885 = !DILocation(line: 893, column: 83, scope: !1739)
!1886 = !DILocation(line: 893, column: 81, scope: !1739)
!1887 = !DILocation(line: 893, column: 69, scope: !1739)
!1888 = !DILocation(line: 893, column: 6, scope: !1739)
!1889 = !DILocation(line: 893, column: 11, scope: !1739)
!1890 = !DILocation(line: 893, column: 16, scope: !1739)
!1891 = !DILocation(line: 893, column: 20, scope: !1739)
!1892 = !DILocation(line: 893, column: 19, scope: !1739)
!1893 = !DILocation(line: 893, column: 23, scope: !1739)
!1894 = !DILocation(line: 893, column: 13, scope: !1739)
!1895 = !DILocation(line: 893, column: 29, scope: !1739)
!1896 = !DILocation(line: 893, column: 30, scope: !1739)
!1897 = !DILocation(line: 893, column: 33, scope: !1739)
!1898 = !DILocation(line: 893, column: 27, scope: !1739)
!1899 = !DILocation(line: 893, column: 38, scope: !1739)
!1900 = !DILocation(line: 893, column: 43, scope: !1739)
!1901 = !DILocation(line: 894, column: 5, scope: !1739)
!1902 = !DILocation(line: 867, column: 27, scope: !1733)
!1903 = !DILocation(line: 867, column: 5, scope: !1733)
!1904 = distinct !{!1904, !1737, !1905}
!1905 = !DILocation(line: 894, column: 5, scope: !1729)
!1906 = !DILocation(line: 895, column: 4, scope: !1730)
!1907 = !DILocation(line: 866, column: 26, scope: !1723)
!1908 = !DILocation(line: 866, column: 4, scope: !1723)
!1909 = distinct !{!1909, !1727, !1910}
!1910 = !DILocation(line: 895, column: 4, scope: !1720)
!1911 = !DILocation(line: 897, column: 2, scope: !1432)
!1912 = !DILocation(line: 818, column: 22, scope: !1427)
!1913 = !DILocation(line: 818, column: 2, scope: !1427)
!1914 = distinct !{!1914, !1430, !1915}
!1915 = !DILocation(line: 897, column: 2, scope: !1424)
!1916 = !DILocation(line: 898, column: 1, scope: !1323)
!1917 = distinct !DISubprogram(name: "ilog2_device", linkageName: "_Z12ilog2_devicei", scope: !3, file: !3, line: 1524, type: !304, scopeLine: 1524, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !1050, retainedNodes: !1051)
!1918 = !DILocalVariable(name: "n", arg: 1, scope: !1917, file: !3, line: 1524, type: !97)
!1919 = !DILocation(line: 1524, column: 33, scope: !1917)
!1920 = !DILocalVariable(name: "nn", scope: !1917, file: !3, line: 1525, type: !97)
!1921 = !DILocation(line: 1525, column: 6, scope: !1917)
!1922 = !DILocalVariable(name: "lg", scope: !1917, file: !3, line: 1525, type: !97)
!1923 = !DILocation(line: 1525, column: 10, scope: !1917)
!1924 = !DILocation(line: 1526, column: 5, scope: !1925)
!1925 = distinct !DILexicalBlock(scope: !1917, file: !3, line: 1526, column: 5)
!1926 = !DILocation(line: 1526, column: 6, scope: !1925)
!1927 = !DILocation(line: 1526, column: 5, scope: !1917)
!1928 = !DILocation(line: 1527, column: 3, scope: !1929)
!1929 = distinct !DILexicalBlock(scope: !1925, file: !3, line: 1526, column: 10)
!1930 = !DILocation(line: 1529, column: 5, scope: !1917)
!1931 = !DILocation(line: 1530, column: 5, scope: !1917)
!1932 = !DILocation(line: 1531, column: 2, scope: !1917)
!1933 = !DILocation(line: 1531, column: 8, scope: !1917)
!1934 = !DILocation(line: 1531, column: 11, scope: !1917)
!1935 = !DILocation(line: 1531, column: 10, scope: !1917)
!1936 = !DILocation(line: 1532, column: 8, scope: !1937)
!1937 = distinct !DILexicalBlock(scope: !1917, file: !3, line: 1531, column: 13)
!1938 = !DILocation(line: 1532, column: 11, scope: !1937)
!1939 = !DILocation(line: 1532, column: 6, scope: !1937)
!1940 = !DILocation(line: 1533, column: 5, scope: !1937)
!1941 = distinct !{!1941, !1932, !1942}
!1942 = !DILocation(line: 1534, column: 2, scope: !1917)
!1943 = !DILocation(line: 1535, column: 9, scope: !1917)
!1944 = !DILocation(line: 1535, column: 2, scope: !1917)
!1945 = !DILocation(line: 1536, column: 1, scope: !1917)
!1946 = distinct !DISubprogram(name: "cffts1_gpu_kernel_3", linkageName: "_Z19cffts1_gpu_kernel_3P8dcomplexS0_", scope: !3, file: !3, line: 907, type: !1153, scopeLine: 908, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !1050, retainedNodes: !1051)
!1947 = !DILocalVariable(name: "x_out", arg: 1, scope: !1946, file: !3, line: 907, type: !98)
!1948 = !DILocation(line: 907, column: 46, scope: !1946)
!1949 = !DILocalVariable(name: "y0", arg: 2, scope: !1946, file: !3, line: 908, type: !98)
!1950 = !DILocation(line: 908, column: 12, scope: !1946)
!1951 = !DILocalVariable(name: "x_y_z", scope: !1946, file: !3, line: 909, type: !97)
!1952 = !DILocation(line: 909, column: 6, scope: !1946)
!1953 = !DILocation(line: 64, column: 3, scope: !1162, inlinedAt: !1954)
!1954 = distinct !DILocation(line: 909, column: 14, scope: !1946)
!1955 = !DILocation(line: 75, column: 3, scope: !1200, inlinedAt: !1956)
!1956 = distinct !DILocation(line: 909, column: 27, scope: !1946)
!1957 = !DILocation(line: 909, column: 25, scope: !1946)
!1958 = !DILocation(line: 53, column: 3, scope: !1246, inlinedAt: !1959)
!1959 = distinct !DILocation(line: 909, column: 40, scope: !1946)
!1960 = !DILocation(line: 909, column: 38, scope: !1946)
!1961 = !DILocation(line: 910, column: 5, scope: !1962)
!1962 = distinct !DILexicalBlock(scope: !1946, file: !3, line: 910, column: 5)
!1963 = !DILocation(line: 910, column: 11, scope: !1962)
!1964 = !DILocation(line: 910, column: 5, scope: !1946)
!1965 = !DILocation(line: 911, column: 3, scope: !1966)
!1966 = distinct !DILexicalBlock(scope: !1962, file: !3, line: 910, column: 25)
!1967 = !DILocalVariable(name: "x", scope: !1946, file: !3, line: 913, type: !97)
!1968 = !DILocation(line: 913, column: 6, scope: !1946)
!1969 = !DILocation(line: 913, column: 10, scope: !1946)
!1970 = !DILocation(line: 913, column: 16, scope: !1946)
!1971 = !DILocalVariable(name: "y", scope: !1946, file: !3, line: 914, type: !97)
!1972 = !DILocation(line: 914, column: 6, scope: !1946)
!1973 = !DILocation(line: 914, column: 11, scope: !1946)
!1974 = !DILocation(line: 914, column: 17, scope: !1946)
!1975 = !DILocation(line: 914, column: 23, scope: !1946)
!1976 = !DILocalVariable(name: "z", scope: !1946, file: !3, line: 915, type: !97)
!1977 = !DILocation(line: 915, column: 6, scope: !1946)
!1978 = !DILocation(line: 915, column: 10, scope: !1946)
!1979 = !DILocation(line: 915, column: 16, scope: !1946)
!1980 = !DILocation(line: 916, column: 22, scope: !1946)
!1981 = !DILocation(line: 916, column: 25, scope: !1946)
!1982 = !DILocation(line: 916, column: 28, scope: !1946)
!1983 = !DILocation(line: 916, column: 29, scope: !1946)
!1984 = !DILocation(line: 916, column: 26, scope: !1946)
!1985 = !DILocation(line: 916, column: 35, scope: !1946)
!1986 = !DILocation(line: 916, column: 36, scope: !1946)
!1987 = !DILocation(line: 916, column: 39, scope: !1946)
!1988 = !DILocation(line: 916, column: 33, scope: !1946)
!1989 = !DILocation(line: 916, column: 45, scope: !1946)
!1990 = !DILocation(line: 916, column: 2, scope: !1946)
!1991 = !DILocation(line: 916, column: 8, scope: !1946)
!1992 = !DILocation(line: 916, column: 15, scope: !1946)
!1993 = !DILocation(line: 916, column: 20, scope: !1946)
!1994 = !DILocation(line: 917, column: 22, scope: !1946)
!1995 = !DILocation(line: 917, column: 25, scope: !1946)
!1996 = !DILocation(line: 917, column: 28, scope: !1946)
!1997 = !DILocation(line: 917, column: 29, scope: !1946)
!1998 = !DILocation(line: 917, column: 26, scope: !1946)
!1999 = !DILocation(line: 917, column: 35, scope: !1946)
!2000 = !DILocation(line: 917, column: 36, scope: !1946)
!2001 = !DILocation(line: 917, column: 39, scope: !1946)
!2002 = !DILocation(line: 917, column: 33, scope: !1946)
!2003 = !DILocation(line: 917, column: 45, scope: !1946)
!2004 = !DILocation(line: 917, column: 2, scope: !1946)
!2005 = !DILocation(line: 917, column: 8, scope: !1946)
!2006 = !DILocation(line: 917, column: 15, scope: !1946)
!2007 = !DILocation(line: 917, column: 20, scope: !1946)
!2008 = !DILocation(line: 918, column: 1, scope: !1946)
!2009 = distinct !DISubprogram(name: "cffts2_gpu_kernel_1", linkageName: "_Z19cffts2_gpu_kernel_1P8dcomplexS0_", scope: !3, file: !3, line: 969, type: !1153, scopeLine: 970, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !1050, retainedNodes: !1051)
!2010 = !DILocalVariable(name: "x_in", arg: 1, scope: !2009, file: !3, line: 969, type: !98)
!2011 = !DILocation(line: 969, column: 46, scope: !2009)
!2012 = !DILocalVariable(name: "y0", arg: 2, scope: !2009, file: !3, line: 970, type: !98)
!2013 = !DILocation(line: 970, column: 12, scope: !2009)
!2014 = !DILocalVariable(name: "x_y_z", scope: !2009, file: !3, line: 971, type: !97)
!2015 = !DILocation(line: 971, column: 6, scope: !2009)
!2016 = !DILocation(line: 64, column: 3, scope: !1162, inlinedAt: !2017)
!2017 = distinct !DILocation(line: 971, column: 14, scope: !2009)
!2018 = !DILocation(line: 75, column: 3, scope: !1200, inlinedAt: !2019)
!2019 = distinct !DILocation(line: 971, column: 27, scope: !2009)
!2020 = !DILocation(line: 971, column: 25, scope: !2009)
!2021 = !DILocation(line: 53, column: 3, scope: !1246, inlinedAt: !2022)
!2022 = distinct !DILocation(line: 971, column: 40, scope: !2009)
!2023 = !DILocation(line: 971, column: 38, scope: !2009)
!2024 = !DILocation(line: 972, column: 5, scope: !2025)
!2025 = distinct !DILexicalBlock(scope: !2009, file: !3, line: 972, column: 5)
!2026 = !DILocation(line: 972, column: 11, scope: !2025)
!2027 = !DILocation(line: 972, column: 5, scope: !2009)
!2028 = !DILocation(line: 973, column: 3, scope: !2029)
!2029 = distinct !DILexicalBlock(scope: !2025, file: !3, line: 972, column: 25)
!2030 = !DILocation(line: 975, column: 19, scope: !2009)
!2031 = !DILocation(line: 975, column: 24, scope: !2009)
!2032 = !DILocation(line: 975, column: 31, scope: !2009)
!2033 = !DILocation(line: 975, column: 2, scope: !2009)
!2034 = !DILocation(line: 975, column: 5, scope: !2009)
!2035 = !DILocation(line: 975, column: 12, scope: !2009)
!2036 = !DILocation(line: 975, column: 17, scope: !2009)
!2037 = !DILocation(line: 976, column: 19, scope: !2009)
!2038 = !DILocation(line: 976, column: 24, scope: !2009)
!2039 = !DILocation(line: 976, column: 31, scope: !2009)
!2040 = !DILocation(line: 976, column: 2, scope: !2009)
!2041 = !DILocation(line: 976, column: 5, scope: !2009)
!2042 = !DILocation(line: 976, column: 12, scope: !2009)
!2043 = !DILocation(line: 976, column: 17, scope: !2009)
!2044 = !DILocation(line: 977, column: 1, scope: !2009)
!2045 = distinct !DISubprogram(name: "cffts2_gpu_kernel_2", linkageName: "_Z19cffts2_gpu_kernel_2iP8dcomplexS0_S0_", scope: !3, file: !3, line: 984, type: !1324, scopeLine: 987, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !1050, retainedNodes: !1051)
!2046 = !DILocalVariable(name: "is", arg: 1, scope: !2045, file: !3, line: 984, type: !1326)
!2047 = !DILocation(line: 984, column: 47, scope: !2045)
!2048 = !DILocalVariable(name: "gty1", arg: 2, scope: !2045, file: !3, line: 985, type: !98)
!2049 = !DILocation(line: 985, column: 12, scope: !2045)
!2050 = !DILocalVariable(name: "gty2", arg: 3, scope: !2045, file: !3, line: 986, type: !98)
!2051 = !DILocation(line: 986, column: 12, scope: !2045)
!2052 = !DILocalVariable(name: "u_device", arg: 4, scope: !2045, file: !3, line: 987, type: !98)
!2053 = !DILocation(line: 987, column: 12, scope: !2045)
!2054 = !DILocalVariable(name: "x_z", scope: !2045, file: !3, line: 988, type: !97)
!2055 = !DILocation(line: 988, column: 6, scope: !2045)
!2056 = !DILocation(line: 64, column: 3, scope: !1162, inlinedAt: !2057)
!2057 = distinct !DILocation(line: 988, column: 12, scope: !2045)
!2058 = !DILocation(line: 75, column: 3, scope: !1200, inlinedAt: !2059)
!2059 = distinct !DILocation(line: 988, column: 25, scope: !2045)
!2060 = !DILocation(line: 988, column: 23, scope: !2045)
!2061 = !DILocation(line: 53, column: 3, scope: !1246, inlinedAt: !2062)
!2062 = distinct !DILocation(line: 988, column: 38, scope: !2045)
!2063 = !DILocation(line: 988, column: 36, scope: !2045)
!2064 = !DILocation(line: 990, column: 5, scope: !2065)
!2065 = distinct !DILexicalBlock(scope: !2045, file: !3, line: 990, column: 5)
!2066 = !DILocation(line: 990, column: 9, scope: !2065)
!2067 = !DILocation(line: 990, column: 5, scope: !2045)
!2068 = !DILocation(line: 991, column: 3, scope: !2069)
!2069 = distinct !DILexicalBlock(scope: !2065, file: !3, line: 990, column: 20)
!2070 = !DILocalVariable(name: "i", scope: !2045, file: !3, line: 994, type: !97)
!2071 = !DILocation(line: 994, column: 6, scope: !2045)
!2072 = !DILocalVariable(name: "k", scope: !2045, file: !3, line: 994, type: !97)
!2073 = !DILocation(line: 994, column: 9, scope: !2045)
!2074 = !DILocalVariable(name: "l", scope: !2045, file: !3, line: 995, type: !97)
!2075 = !DILocation(line: 995, column: 6, scope: !2045)
!2076 = !DILocalVariable(name: "j1", scope: !2045, file: !3, line: 995, type: !97)
!2077 = !DILocation(line: 995, column: 9, scope: !2045)
!2078 = !DILocalVariable(name: "i1", scope: !2045, file: !3, line: 995, type: !97)
!2079 = !DILocation(line: 995, column: 13, scope: !2045)
!2080 = !DILocalVariable(name: "k1", scope: !2045, file: !3, line: 995, type: !97)
!2081 = !DILocation(line: 995, column: 17, scope: !2045)
!2082 = !DILocalVariable(name: "n1", scope: !2045, file: !3, line: 996, type: !97)
!2083 = !DILocation(line: 996, column: 6, scope: !2045)
!2084 = !DILocalVariable(name: "li", scope: !2045, file: !3, line: 996, type: !97)
!2085 = !DILocation(line: 996, column: 10, scope: !2045)
!2086 = !DILocalVariable(name: "lj", scope: !2045, file: !3, line: 996, type: !97)
!2087 = !DILocation(line: 996, column: 14, scope: !2045)
!2088 = !DILocalVariable(name: "lk", scope: !2045, file: !3, line: 996, type: !97)
!2089 = !DILocation(line: 996, column: 18, scope: !2045)
!2090 = !DILocalVariable(name: "ku", scope: !2045, file: !3, line: 996, type: !97)
!2091 = !DILocation(line: 996, column: 22, scope: !2045)
!2092 = !DILocalVariable(name: "i11", scope: !2045, file: !3, line: 996, type: !97)
!2093 = !DILocation(line: 996, column: 26, scope: !2045)
!2094 = !DILocalVariable(name: "i12", scope: !2045, file: !3, line: 996, type: !97)
!2095 = !DILocation(line: 996, column: 31, scope: !2045)
!2096 = !DILocalVariable(name: "i21", scope: !2045, file: !3, line: 996, type: !97)
!2097 = !DILocation(line: 996, column: 36, scope: !2045)
!2098 = !DILocalVariable(name: "i22", scope: !2045, file: !3, line: 996, type: !97)
!2099 = !DILocation(line: 996, column: 41, scope: !2045)
!2100 = !DILocation(line: 998, column: 6, scope: !2045)
!2101 = !DILocation(line: 998, column: 10, scope: !2045)
!2102 = !DILocation(line: 998, column: 4, scope: !2045)
!2103 = !DILocation(line: 999, column: 7, scope: !2045)
!2104 = !DILocation(line: 999, column: 11, scope: !2045)
!2105 = !DILocation(line: 999, column: 17, scope: !2045)
!2106 = !DILocation(line: 999, column: 4, scope: !2045)
!2107 = !DILocalVariable(name: "logd2", scope: !2045, file: !3, line: 1001, type: !1326)
!2108 = !DILocation(line: 1001, column: 12, scope: !2045)
!2109 = !DILocation(line: 1001, column: 20, scope: !2045)
!2110 = !DILocalVariable(name: "uu1_real", scope: !2045, file: !3, line: 1003, type: !104)
!2111 = !DILocation(line: 1003, column: 9, scope: !2045)
!2112 = !DILocalVariable(name: "x11_real", scope: !2045, file: !3, line: 1003, type: !104)
!2113 = !DILocation(line: 1003, column: 19, scope: !2045)
!2114 = !DILocalVariable(name: "x21_real", scope: !2045, file: !3, line: 1003, type: !104)
!2115 = !DILocation(line: 1003, column: 29, scope: !2045)
!2116 = !DILocalVariable(name: "uu1_imag", scope: !2045, file: !3, line: 1004, type: !104)
!2117 = !DILocation(line: 1004, column: 9, scope: !2045)
!2118 = !DILocalVariable(name: "x11_imag", scope: !2045, file: !3, line: 1004, type: !104)
!2119 = !DILocation(line: 1004, column: 19, scope: !2045)
!2120 = !DILocalVariable(name: "x21_imag", scope: !2045, file: !3, line: 1004, type: !104)
!2121 = !DILocation(line: 1004, column: 29, scope: !2045)
!2122 = !DILocalVariable(name: "uu2_real", scope: !2045, file: !3, line: 1005, type: !104)
!2123 = !DILocation(line: 1005, column: 9, scope: !2045)
!2124 = !DILocalVariable(name: "x12_real", scope: !2045, file: !3, line: 1005, type: !104)
!2125 = !DILocation(line: 1005, column: 19, scope: !2045)
!2126 = !DILocalVariable(name: "x22_real", scope: !2045, file: !3, line: 1005, type: !104)
!2127 = !DILocation(line: 1005, column: 29, scope: !2045)
!2128 = !DILocalVariable(name: "uu2_imag", scope: !2045, file: !3, line: 1006, type: !104)
!2129 = !DILocation(line: 1006, column: 9, scope: !2045)
!2130 = !DILocalVariable(name: "x12_imag", scope: !2045, file: !3, line: 1006, type: !104)
!2131 = !DILocation(line: 1006, column: 19, scope: !2045)
!2132 = !DILocalVariable(name: "x22_imag", scope: !2045, file: !3, line: 1006, type: !104)
!2133 = !DILocation(line: 1006, column: 29, scope: !2045)
!2134 = !DILocalVariable(name: "temp_real", scope: !2045, file: !3, line: 1007, type: !104)
!2135 = !DILocation(line: 1007, column: 9, scope: !2045)
!2136 = !DILocalVariable(name: "temp2_real", scope: !2045, file: !3, line: 1007, type: !104)
!2137 = !DILocation(line: 1007, column: 20, scope: !2045)
!2138 = !DILocalVariable(name: "temp_imag", scope: !2045, file: !3, line: 1008, type: !104)
!2139 = !DILocation(line: 1008, column: 9, scope: !2045)
!2140 = !DILocalVariable(name: "temp2_imag", scope: !2045, file: !3, line: 1008, type: !104)
!2141 = !DILocation(line: 1008, column: 20, scope: !2045)
!2142 = !DILocation(line: 1010, column: 7, scope: !2143)
!2143 = distinct !DILexicalBlock(scope: !2045, file: !3, line: 1010, column: 2)
!2144 = !DILocation(line: 1010, column: 6, scope: !2143)
!2145 = !DILocation(line: 1010, column: 11, scope: !2146)
!2146 = distinct !DILexicalBlock(scope: !2143, file: !3, line: 1010, column: 2)
!2147 = !DILocation(line: 1010, column: 14, scope: !2146)
!2148 = !DILocation(line: 1010, column: 12, scope: !2146)
!2149 = !DILocation(line: 1010, column: 2, scope: !2143)
!2150 = !DILocation(line: 1011, column: 6, scope: !2151)
!2151 = distinct !DILexicalBlock(scope: !2146, file: !3, line: 1010, column: 26)
!2152 = !DILocation(line: 1012, column: 14, scope: !2151)
!2153 = !DILocation(line: 1012, column: 16, scope: !2151)
!2154 = !DILocation(line: 1012, column: 10, scope: !2151)
!2155 = !DILocation(line: 1012, column: 6, scope: !2151)
!2156 = !DILocation(line: 1013, column: 14, scope: !2151)
!2157 = !DILocation(line: 1013, column: 22, scope: !2151)
!2158 = !DILocation(line: 1013, column: 20, scope: !2151)
!2159 = !DILocation(line: 1013, column: 10, scope: !2151)
!2160 = !DILocation(line: 1013, column: 6, scope: !2151)
!2161 = !DILocation(line: 1014, column: 12, scope: !2151)
!2162 = !DILocation(line: 1014, column: 10, scope: !2151)
!2163 = !DILocation(line: 1014, column: 6, scope: !2151)
!2164 = !DILocation(line: 1015, column: 8, scope: !2151)
!2165 = !DILocation(line: 1015, column: 6, scope: !2151)
!2166 = !DILocation(line: 1016, column: 9, scope: !2167)
!2167 = distinct !DILexicalBlock(scope: !2151, file: !3, line: 1016, column: 3)
!2168 = !DILocation(line: 1016, column: 7, scope: !2167)
!2169 = !DILocation(line: 1016, column: 13, scope: !2170)
!2170 = distinct !DILexicalBlock(scope: !2167, file: !3, line: 1016, column: 3)
!2171 = !DILocation(line: 1016, column: 17, scope: !2170)
!2172 = !DILocation(line: 1016, column: 19, scope: !2170)
!2173 = !DILocation(line: 1016, column: 15, scope: !2170)
!2174 = !DILocation(line: 1016, column: 3, scope: !2167)
!2175 = !DILocation(line: 1017, column: 10, scope: !2176)
!2176 = distinct !DILexicalBlock(scope: !2177, file: !3, line: 1017, column: 4)
!2177 = distinct !DILexicalBlock(scope: !2170, file: !3, line: 1016, column: 28)
!2178 = !DILocation(line: 1017, column: 8, scope: !2176)
!2179 = !DILocation(line: 1017, column: 14, scope: !2180)
!2180 = distinct !DILexicalBlock(scope: !2176, file: !3, line: 1017, column: 4)
!2181 = !DILocation(line: 1017, column: 18, scope: !2180)
!2182 = !DILocation(line: 1017, column: 20, scope: !2180)
!2183 = !DILocation(line: 1017, column: 16, scope: !2180)
!2184 = !DILocation(line: 1017, column: 4, scope: !2176)
!2185 = !DILocation(line: 1018, column: 11, scope: !2186)
!2186 = distinct !DILexicalBlock(scope: !2180, file: !3, line: 1017, column: 29)
!2187 = !DILocation(line: 1018, column: 16, scope: !2186)
!2188 = !DILocation(line: 1018, column: 14, scope: !2186)
!2189 = !DILocation(line: 1018, column: 9, scope: !2186)
!2190 = !DILocation(line: 1019, column: 11, scope: !2186)
!2191 = !DILocation(line: 1019, column: 17, scope: !2186)
!2192 = !DILocation(line: 1019, column: 15, scope: !2186)
!2193 = !DILocation(line: 1019, column: 9, scope: !2186)
!2194 = !DILocation(line: 1020, column: 11, scope: !2186)
!2195 = !DILocation(line: 1020, column: 16, scope: !2186)
!2196 = !DILocation(line: 1020, column: 14, scope: !2186)
!2197 = !DILocation(line: 1020, column: 9, scope: !2186)
!2198 = !DILocation(line: 1021, column: 11, scope: !2186)
!2199 = !DILocation(line: 1021, column: 17, scope: !2186)
!2200 = !DILocation(line: 1021, column: 15, scope: !2186)
!2201 = !DILocation(line: 1021, column: 9, scope: !2186)
!2202 = !DILocation(line: 1023, column: 16, scope: !2186)
!2203 = !DILocation(line: 1023, column: 25, scope: !2186)
!2204 = !DILocation(line: 1023, column: 28, scope: !2186)
!2205 = !DILocation(line: 1023, column: 27, scope: !2186)
!2206 = !DILocation(line: 1023, column: 32, scope: !2186)
!2207 = !DILocation(line: 1023, column: 14, scope: !2186)
!2208 = !DILocation(line: 1024, column: 16, scope: !2186)
!2209 = !DILocation(line: 1024, column: 19, scope: !2186)
!2210 = !DILocation(line: 1024, column: 28, scope: !2186)
!2211 = !DILocation(line: 1024, column: 31, scope: !2186)
!2212 = !DILocation(line: 1024, column: 30, scope: !2186)
!2213 = !DILocation(line: 1024, column: 35, scope: !2186)
!2214 = !DILocation(line: 1024, column: 18, scope: !2186)
!2215 = !DILocation(line: 1024, column: 14, scope: !2186)
!2216 = !DILocation(line: 1027, column: 16, scope: !2186)
!2217 = !DILocation(line: 1027, column: 21, scope: !2186)
!2218 = !DILocation(line: 1027, column: 26, scope: !2186)
!2219 = !DILocation(line: 1027, column: 30, scope: !2186)
!2220 = !DILocation(line: 1027, column: 29, scope: !2186)
!2221 = !DILocation(line: 1027, column: 33, scope: !2186)
!2222 = !DILocation(line: 1027, column: 23, scope: !2186)
!2223 = !DILocation(line: 1027, column: 39, scope: !2186)
!2224 = !DILocation(line: 1027, column: 40, scope: !2186)
!2225 = !DILocation(line: 1027, column: 43, scope: !2186)
!2226 = !DILocation(line: 1027, column: 37, scope: !2186)
!2227 = !DILocation(line: 1027, column: 48, scope: !2186)
!2228 = !DILocation(line: 1027, column: 14, scope: !2186)
!2229 = !DILocation(line: 1028, column: 16, scope: !2186)
!2230 = !DILocation(line: 1028, column: 21, scope: !2186)
!2231 = !DILocation(line: 1028, column: 26, scope: !2186)
!2232 = !DILocation(line: 1028, column: 30, scope: !2186)
!2233 = !DILocation(line: 1028, column: 29, scope: !2186)
!2234 = !DILocation(line: 1028, column: 33, scope: !2186)
!2235 = !DILocation(line: 1028, column: 23, scope: !2186)
!2236 = !DILocation(line: 1028, column: 39, scope: !2186)
!2237 = !DILocation(line: 1028, column: 40, scope: !2186)
!2238 = !DILocation(line: 1028, column: 43, scope: !2186)
!2239 = !DILocation(line: 1028, column: 37, scope: !2186)
!2240 = !DILocation(line: 1028, column: 48, scope: !2186)
!2241 = !DILocation(line: 1028, column: 14, scope: !2186)
!2242 = !DILocation(line: 1031, column: 16, scope: !2186)
!2243 = !DILocation(line: 1031, column: 21, scope: !2186)
!2244 = !DILocation(line: 1031, column: 26, scope: !2186)
!2245 = !DILocation(line: 1031, column: 30, scope: !2186)
!2246 = !DILocation(line: 1031, column: 29, scope: !2186)
!2247 = !DILocation(line: 1031, column: 33, scope: !2186)
!2248 = !DILocation(line: 1031, column: 23, scope: !2186)
!2249 = !DILocation(line: 1031, column: 39, scope: !2186)
!2250 = !DILocation(line: 1031, column: 40, scope: !2186)
!2251 = !DILocation(line: 1031, column: 43, scope: !2186)
!2252 = !DILocation(line: 1031, column: 37, scope: !2186)
!2253 = !DILocation(line: 1031, column: 48, scope: !2186)
!2254 = !DILocation(line: 1031, column: 14, scope: !2186)
!2255 = !DILocation(line: 1032, column: 16, scope: !2186)
!2256 = !DILocation(line: 1032, column: 21, scope: !2186)
!2257 = !DILocation(line: 1032, column: 26, scope: !2186)
!2258 = !DILocation(line: 1032, column: 30, scope: !2186)
!2259 = !DILocation(line: 1032, column: 29, scope: !2186)
!2260 = !DILocation(line: 1032, column: 33, scope: !2186)
!2261 = !DILocation(line: 1032, column: 23, scope: !2186)
!2262 = !DILocation(line: 1032, column: 39, scope: !2186)
!2263 = !DILocation(line: 1032, column: 40, scope: !2186)
!2264 = !DILocation(line: 1032, column: 43, scope: !2186)
!2265 = !DILocation(line: 1032, column: 37, scope: !2186)
!2266 = !DILocation(line: 1032, column: 48, scope: !2186)
!2267 = !DILocation(line: 1032, column: 14, scope: !2186)
!2268 = !DILocation(line: 1035, column: 44, scope: !2186)
!2269 = !DILocation(line: 1035, column: 55, scope: !2186)
!2270 = !DILocation(line: 1035, column: 53, scope: !2186)
!2271 = !DILocation(line: 1035, column: 5, scope: !2186)
!2272 = !DILocation(line: 1035, column: 10, scope: !2186)
!2273 = !DILocation(line: 1035, column: 15, scope: !2186)
!2274 = !DILocation(line: 1035, column: 19, scope: !2186)
!2275 = !DILocation(line: 1035, column: 18, scope: !2186)
!2276 = !DILocation(line: 1035, column: 22, scope: !2186)
!2277 = !DILocation(line: 1035, column: 12, scope: !2186)
!2278 = !DILocation(line: 1035, column: 28, scope: !2186)
!2279 = !DILocation(line: 1035, column: 29, scope: !2186)
!2280 = !DILocation(line: 1035, column: 32, scope: !2186)
!2281 = !DILocation(line: 1035, column: 26, scope: !2186)
!2282 = !DILocation(line: 1035, column: 37, scope: !2186)
!2283 = !DILocation(line: 1035, column: 42, scope: !2186)
!2284 = !DILocation(line: 1036, column: 44, scope: !2186)
!2285 = !DILocation(line: 1036, column: 55, scope: !2186)
!2286 = !DILocation(line: 1036, column: 53, scope: !2186)
!2287 = !DILocation(line: 1036, column: 5, scope: !2186)
!2288 = !DILocation(line: 1036, column: 10, scope: !2186)
!2289 = !DILocation(line: 1036, column: 15, scope: !2186)
!2290 = !DILocation(line: 1036, column: 19, scope: !2186)
!2291 = !DILocation(line: 1036, column: 18, scope: !2186)
!2292 = !DILocation(line: 1036, column: 22, scope: !2186)
!2293 = !DILocation(line: 1036, column: 12, scope: !2186)
!2294 = !DILocation(line: 1036, column: 28, scope: !2186)
!2295 = !DILocation(line: 1036, column: 29, scope: !2186)
!2296 = !DILocation(line: 1036, column: 32, scope: !2186)
!2297 = !DILocation(line: 1036, column: 26, scope: !2186)
!2298 = !DILocation(line: 1036, column: 37, scope: !2186)
!2299 = !DILocation(line: 1036, column: 42, scope: !2186)
!2300 = !DILocation(line: 1038, column: 17, scope: !2186)
!2301 = !DILocation(line: 1038, column: 28, scope: !2186)
!2302 = !DILocation(line: 1038, column: 26, scope: !2186)
!2303 = !DILocation(line: 1038, column: 15, scope: !2186)
!2304 = !DILocation(line: 1039, column: 17, scope: !2186)
!2305 = !DILocation(line: 1039, column: 28, scope: !2186)
!2306 = !DILocation(line: 1039, column: 26, scope: !2186)
!2307 = !DILocation(line: 1039, column: 15, scope: !2186)
!2308 = !DILocation(line: 1042, column: 45, scope: !2186)
!2309 = !DILocation(line: 1042, column: 56, scope: !2186)
!2310 = !DILocation(line: 1042, column: 54, scope: !2186)
!2311 = !DILocation(line: 1042, column: 70, scope: !2186)
!2312 = !DILocation(line: 1042, column: 81, scope: !2186)
!2313 = !DILocation(line: 1042, column: 79, scope: !2186)
!2314 = !DILocation(line: 1042, column: 67, scope: !2186)
!2315 = !DILocation(line: 1042, column: 5, scope: !2186)
!2316 = !DILocation(line: 1042, column: 10, scope: !2186)
!2317 = !DILocation(line: 1042, column: 15, scope: !2186)
!2318 = !DILocation(line: 1042, column: 19, scope: !2186)
!2319 = !DILocation(line: 1042, column: 18, scope: !2186)
!2320 = !DILocation(line: 1042, column: 22, scope: !2186)
!2321 = !DILocation(line: 1042, column: 12, scope: !2186)
!2322 = !DILocation(line: 1042, column: 28, scope: !2186)
!2323 = !DILocation(line: 1042, column: 29, scope: !2186)
!2324 = !DILocation(line: 1042, column: 32, scope: !2186)
!2325 = !DILocation(line: 1042, column: 26, scope: !2186)
!2326 = !DILocation(line: 1042, column: 37, scope: !2186)
!2327 = !DILocation(line: 1042, column: 42, scope: !2186)
!2328 = !DILocation(line: 1043, column: 45, scope: !2186)
!2329 = !DILocation(line: 1043, column: 56, scope: !2186)
!2330 = !DILocation(line: 1043, column: 54, scope: !2186)
!2331 = !DILocation(line: 1043, column: 70, scope: !2186)
!2332 = !DILocation(line: 1043, column: 81, scope: !2186)
!2333 = !DILocation(line: 1043, column: 79, scope: !2186)
!2334 = !DILocation(line: 1043, column: 67, scope: !2186)
!2335 = !DILocation(line: 1043, column: 5, scope: !2186)
!2336 = !DILocation(line: 1043, column: 10, scope: !2186)
!2337 = !DILocation(line: 1043, column: 15, scope: !2186)
!2338 = !DILocation(line: 1043, column: 19, scope: !2186)
!2339 = !DILocation(line: 1043, column: 18, scope: !2186)
!2340 = !DILocation(line: 1043, column: 22, scope: !2186)
!2341 = !DILocation(line: 1043, column: 12, scope: !2186)
!2342 = !DILocation(line: 1043, column: 28, scope: !2186)
!2343 = !DILocation(line: 1043, column: 29, scope: !2186)
!2344 = !DILocation(line: 1043, column: 32, scope: !2186)
!2345 = !DILocation(line: 1043, column: 26, scope: !2186)
!2346 = !DILocation(line: 1043, column: 37, scope: !2186)
!2347 = !DILocation(line: 1043, column: 42, scope: !2186)
!2348 = !DILocation(line: 1045, column: 4, scope: !2186)
!2349 = !DILocation(line: 1017, column: 26, scope: !2180)
!2350 = !DILocation(line: 1017, column: 4, scope: !2180)
!2351 = distinct !{!2351, !2184, !2352}
!2352 = !DILocation(line: 1045, column: 4, scope: !2176)
!2353 = !DILocation(line: 1046, column: 3, scope: !2177)
!2354 = !DILocation(line: 1016, column: 25, scope: !2170)
!2355 = !DILocation(line: 1016, column: 3, scope: !2170)
!2356 = distinct !{!2356, !2174, !2357}
!2357 = !DILocation(line: 1046, column: 3, scope: !2167)
!2358 = !DILocation(line: 1047, column: 6, scope: !2359)
!2359 = distinct !DILexicalBlock(scope: !2151, file: !3, line: 1047, column: 6)
!2360 = !DILocation(line: 1047, column: 9, scope: !2359)
!2361 = !DILocation(line: 1047, column: 7, scope: !2359)
!2362 = !DILocation(line: 1047, column: 6, scope: !2151)
!2363 = !DILocation(line: 1048, column: 10, scope: !2364)
!2364 = distinct !DILexicalBlock(scope: !2365, file: !3, line: 1048, column: 4)
!2365 = distinct !DILexicalBlock(scope: !2359, file: !3, line: 1047, column: 15)
!2366 = !DILocation(line: 1048, column: 8, scope: !2364)
!2367 = !DILocation(line: 1048, column: 14, scope: !2368)
!2368 = distinct !DILexicalBlock(scope: !2364, file: !3, line: 1048, column: 4)
!2369 = !DILocation(line: 1048, column: 16, scope: !2368)
!2370 = !DILocation(line: 1048, column: 4, scope: !2364)
!2371 = !DILocation(line: 1050, column: 38, scope: !2372)
!2372 = distinct !DILexicalBlock(scope: !2368, file: !3, line: 1048, column: 26)
!2373 = !DILocation(line: 1050, column: 43, scope: !2372)
!2374 = !DILocation(line: 1050, column: 47, scope: !2372)
!2375 = !DILocation(line: 1050, column: 49, scope: !2372)
!2376 = !DILocation(line: 1050, column: 45, scope: !2372)
!2377 = !DILocation(line: 1050, column: 55, scope: !2372)
!2378 = !DILocation(line: 1050, column: 56, scope: !2372)
!2379 = !DILocation(line: 1050, column: 59, scope: !2372)
!2380 = !DILocation(line: 1050, column: 53, scope: !2372)
!2381 = !DILocation(line: 1050, column: 64, scope: !2372)
!2382 = !DILocation(line: 1050, column: 5, scope: !2372)
!2383 = !DILocation(line: 1050, column: 10, scope: !2372)
!2384 = !DILocation(line: 1050, column: 14, scope: !2372)
!2385 = !DILocation(line: 1050, column: 16, scope: !2372)
!2386 = !DILocation(line: 1050, column: 12, scope: !2372)
!2387 = !DILocation(line: 1050, column: 22, scope: !2372)
!2388 = !DILocation(line: 1050, column: 23, scope: !2372)
!2389 = !DILocation(line: 1050, column: 26, scope: !2372)
!2390 = !DILocation(line: 1050, column: 20, scope: !2372)
!2391 = !DILocation(line: 1050, column: 31, scope: !2372)
!2392 = !DILocation(line: 1050, column: 36, scope: !2372)
!2393 = !DILocation(line: 1051, column: 38, scope: !2372)
!2394 = !DILocation(line: 1051, column: 43, scope: !2372)
!2395 = !DILocation(line: 1051, column: 47, scope: !2372)
!2396 = !DILocation(line: 1051, column: 49, scope: !2372)
!2397 = !DILocation(line: 1051, column: 45, scope: !2372)
!2398 = !DILocation(line: 1051, column: 55, scope: !2372)
!2399 = !DILocation(line: 1051, column: 56, scope: !2372)
!2400 = !DILocation(line: 1051, column: 59, scope: !2372)
!2401 = !DILocation(line: 1051, column: 53, scope: !2372)
!2402 = !DILocation(line: 1051, column: 64, scope: !2372)
!2403 = !DILocation(line: 1051, column: 5, scope: !2372)
!2404 = !DILocation(line: 1051, column: 10, scope: !2372)
!2405 = !DILocation(line: 1051, column: 14, scope: !2372)
!2406 = !DILocation(line: 1051, column: 16, scope: !2372)
!2407 = !DILocation(line: 1051, column: 12, scope: !2372)
!2408 = !DILocation(line: 1051, column: 22, scope: !2372)
!2409 = !DILocation(line: 1051, column: 23, scope: !2372)
!2410 = !DILocation(line: 1051, column: 26, scope: !2372)
!2411 = !DILocation(line: 1051, column: 20, scope: !2372)
!2412 = !DILocation(line: 1051, column: 31, scope: !2372)
!2413 = !DILocation(line: 1051, column: 36, scope: !2372)
!2414 = !DILocation(line: 1052, column: 4, scope: !2372)
!2415 = !DILocation(line: 1048, column: 23, scope: !2368)
!2416 = !DILocation(line: 1048, column: 4, scope: !2368)
!2417 = distinct !{!2417, !2370, !2418}
!2418 = !DILocation(line: 1052, column: 4, scope: !2364)
!2419 = !DILocation(line: 1053, column: 3, scope: !2365)
!2420 = !DILocation(line: 1055, column: 7, scope: !2421)
!2421 = distinct !DILexicalBlock(scope: !2359, file: !3, line: 1054, column: 7)
!2422 = !DILocation(line: 1056, column: 15, scope: !2421)
!2423 = !DILocation(line: 1056, column: 16, scope: !2421)
!2424 = !DILocation(line: 1056, column: 19, scope: !2421)
!2425 = !DILocation(line: 1056, column: 11, scope: !2421)
!2426 = !DILocation(line: 1056, column: 7, scope: !2421)
!2427 = !DILocation(line: 1057, column: 15, scope: !2421)
!2428 = !DILocation(line: 1057, column: 24, scope: !2421)
!2429 = !DILocation(line: 1057, column: 25, scope: !2421)
!2430 = !DILocation(line: 1057, column: 21, scope: !2421)
!2431 = !DILocation(line: 1057, column: 11, scope: !2421)
!2432 = !DILocation(line: 1057, column: 7, scope: !2421)
!2433 = !DILocation(line: 1058, column: 13, scope: !2421)
!2434 = !DILocation(line: 1058, column: 11, scope: !2421)
!2435 = !DILocation(line: 1058, column: 7, scope: !2421)
!2436 = !DILocation(line: 1059, column: 9, scope: !2421)
!2437 = !DILocation(line: 1059, column: 7, scope: !2421)
!2438 = !DILocation(line: 1060, column: 10, scope: !2439)
!2439 = distinct !DILexicalBlock(scope: !2421, file: !3, line: 1060, column: 4)
!2440 = !DILocation(line: 1060, column: 8, scope: !2439)
!2441 = !DILocation(line: 1060, column: 14, scope: !2442)
!2442 = distinct !DILexicalBlock(scope: !2439, file: !3, line: 1060, column: 4)
!2443 = !DILocation(line: 1060, column: 18, scope: !2442)
!2444 = !DILocation(line: 1060, column: 20, scope: !2442)
!2445 = !DILocation(line: 1060, column: 16, scope: !2442)
!2446 = !DILocation(line: 1060, column: 4, scope: !2439)
!2447 = !DILocation(line: 1061, column: 11, scope: !2448)
!2448 = distinct !DILexicalBlock(scope: !2449, file: !3, line: 1061, column: 5)
!2449 = distinct !DILexicalBlock(scope: !2442, file: !3, line: 1060, column: 29)
!2450 = !DILocation(line: 1061, column: 9, scope: !2448)
!2451 = !DILocation(line: 1061, column: 15, scope: !2452)
!2452 = distinct !DILexicalBlock(scope: !2448, file: !3, line: 1061, column: 5)
!2453 = !DILocation(line: 1061, column: 19, scope: !2452)
!2454 = !DILocation(line: 1061, column: 21, scope: !2452)
!2455 = !DILocation(line: 1061, column: 17, scope: !2452)
!2456 = !DILocation(line: 1061, column: 5, scope: !2448)
!2457 = !DILocation(line: 1062, column: 12, scope: !2458)
!2458 = distinct !DILexicalBlock(scope: !2452, file: !3, line: 1061, column: 30)
!2459 = !DILocation(line: 1062, column: 17, scope: !2458)
!2460 = !DILocation(line: 1062, column: 15, scope: !2458)
!2461 = !DILocation(line: 1062, column: 10, scope: !2458)
!2462 = !DILocation(line: 1063, column: 12, scope: !2458)
!2463 = !DILocation(line: 1063, column: 18, scope: !2458)
!2464 = !DILocation(line: 1063, column: 16, scope: !2458)
!2465 = !DILocation(line: 1063, column: 10, scope: !2458)
!2466 = !DILocation(line: 1064, column: 12, scope: !2458)
!2467 = !DILocation(line: 1064, column: 17, scope: !2458)
!2468 = !DILocation(line: 1064, column: 15, scope: !2458)
!2469 = !DILocation(line: 1064, column: 10, scope: !2458)
!2470 = !DILocation(line: 1065, column: 12, scope: !2458)
!2471 = !DILocation(line: 1065, column: 18, scope: !2458)
!2472 = !DILocation(line: 1065, column: 16, scope: !2458)
!2473 = !DILocation(line: 1065, column: 10, scope: !2458)
!2474 = !DILocation(line: 1067, column: 17, scope: !2458)
!2475 = !DILocation(line: 1067, column: 26, scope: !2458)
!2476 = !DILocation(line: 1067, column: 29, scope: !2458)
!2477 = !DILocation(line: 1067, column: 28, scope: !2458)
!2478 = !DILocation(line: 1067, column: 33, scope: !2458)
!2479 = !DILocation(line: 1067, column: 15, scope: !2458)
!2480 = !DILocation(line: 1068, column: 17, scope: !2458)
!2481 = !DILocation(line: 1068, column: 20, scope: !2458)
!2482 = !DILocation(line: 1068, column: 29, scope: !2458)
!2483 = !DILocation(line: 1068, column: 32, scope: !2458)
!2484 = !DILocation(line: 1068, column: 31, scope: !2458)
!2485 = !DILocation(line: 1068, column: 36, scope: !2458)
!2486 = !DILocation(line: 1068, column: 19, scope: !2458)
!2487 = !DILocation(line: 1068, column: 15, scope: !2458)
!2488 = !DILocation(line: 1071, column: 17, scope: !2458)
!2489 = !DILocation(line: 1071, column: 22, scope: !2458)
!2490 = !DILocation(line: 1071, column: 27, scope: !2458)
!2491 = !DILocation(line: 1071, column: 31, scope: !2458)
!2492 = !DILocation(line: 1071, column: 30, scope: !2458)
!2493 = !DILocation(line: 1071, column: 34, scope: !2458)
!2494 = !DILocation(line: 1071, column: 24, scope: !2458)
!2495 = !DILocation(line: 1071, column: 40, scope: !2458)
!2496 = !DILocation(line: 1071, column: 41, scope: !2458)
!2497 = !DILocation(line: 1071, column: 44, scope: !2458)
!2498 = !DILocation(line: 1071, column: 38, scope: !2458)
!2499 = !DILocation(line: 1071, column: 49, scope: !2458)
!2500 = !DILocation(line: 1071, column: 15, scope: !2458)
!2501 = !DILocation(line: 1072, column: 17, scope: !2458)
!2502 = !DILocation(line: 1072, column: 22, scope: !2458)
!2503 = !DILocation(line: 1072, column: 27, scope: !2458)
!2504 = !DILocation(line: 1072, column: 31, scope: !2458)
!2505 = !DILocation(line: 1072, column: 30, scope: !2458)
!2506 = !DILocation(line: 1072, column: 34, scope: !2458)
!2507 = !DILocation(line: 1072, column: 24, scope: !2458)
!2508 = !DILocation(line: 1072, column: 40, scope: !2458)
!2509 = !DILocation(line: 1072, column: 41, scope: !2458)
!2510 = !DILocation(line: 1072, column: 44, scope: !2458)
!2511 = !DILocation(line: 1072, column: 38, scope: !2458)
!2512 = !DILocation(line: 1072, column: 49, scope: !2458)
!2513 = !DILocation(line: 1072, column: 15, scope: !2458)
!2514 = !DILocation(line: 1075, column: 17, scope: !2458)
!2515 = !DILocation(line: 1075, column: 22, scope: !2458)
!2516 = !DILocation(line: 1075, column: 27, scope: !2458)
!2517 = !DILocation(line: 1075, column: 31, scope: !2458)
!2518 = !DILocation(line: 1075, column: 30, scope: !2458)
!2519 = !DILocation(line: 1075, column: 34, scope: !2458)
!2520 = !DILocation(line: 1075, column: 24, scope: !2458)
!2521 = !DILocation(line: 1075, column: 40, scope: !2458)
!2522 = !DILocation(line: 1075, column: 41, scope: !2458)
!2523 = !DILocation(line: 1075, column: 44, scope: !2458)
!2524 = !DILocation(line: 1075, column: 38, scope: !2458)
!2525 = !DILocation(line: 1075, column: 49, scope: !2458)
!2526 = !DILocation(line: 1075, column: 15, scope: !2458)
!2527 = !DILocation(line: 1076, column: 17, scope: !2458)
!2528 = !DILocation(line: 1076, column: 22, scope: !2458)
!2529 = !DILocation(line: 1076, column: 27, scope: !2458)
!2530 = !DILocation(line: 1076, column: 31, scope: !2458)
!2531 = !DILocation(line: 1076, column: 30, scope: !2458)
!2532 = !DILocation(line: 1076, column: 34, scope: !2458)
!2533 = !DILocation(line: 1076, column: 24, scope: !2458)
!2534 = !DILocation(line: 1076, column: 40, scope: !2458)
!2535 = !DILocation(line: 1076, column: 41, scope: !2458)
!2536 = !DILocation(line: 1076, column: 44, scope: !2458)
!2537 = !DILocation(line: 1076, column: 38, scope: !2458)
!2538 = !DILocation(line: 1076, column: 49, scope: !2458)
!2539 = !DILocation(line: 1076, column: 15, scope: !2458)
!2540 = !DILocation(line: 1079, column: 45, scope: !2458)
!2541 = !DILocation(line: 1079, column: 56, scope: !2458)
!2542 = !DILocation(line: 1079, column: 54, scope: !2458)
!2543 = !DILocation(line: 1079, column: 6, scope: !2458)
!2544 = !DILocation(line: 1079, column: 11, scope: !2458)
!2545 = !DILocation(line: 1079, column: 16, scope: !2458)
!2546 = !DILocation(line: 1079, column: 20, scope: !2458)
!2547 = !DILocation(line: 1079, column: 19, scope: !2458)
!2548 = !DILocation(line: 1079, column: 23, scope: !2458)
!2549 = !DILocation(line: 1079, column: 13, scope: !2458)
!2550 = !DILocation(line: 1079, column: 29, scope: !2458)
!2551 = !DILocation(line: 1079, column: 30, scope: !2458)
!2552 = !DILocation(line: 1079, column: 33, scope: !2458)
!2553 = !DILocation(line: 1079, column: 27, scope: !2458)
!2554 = !DILocation(line: 1079, column: 38, scope: !2458)
!2555 = !DILocation(line: 1079, column: 43, scope: !2458)
!2556 = !DILocation(line: 1080, column: 45, scope: !2458)
!2557 = !DILocation(line: 1080, column: 56, scope: !2458)
!2558 = !DILocation(line: 1080, column: 54, scope: !2458)
!2559 = !DILocation(line: 1080, column: 6, scope: !2458)
!2560 = !DILocation(line: 1080, column: 11, scope: !2458)
!2561 = !DILocation(line: 1080, column: 16, scope: !2458)
!2562 = !DILocation(line: 1080, column: 20, scope: !2458)
!2563 = !DILocation(line: 1080, column: 19, scope: !2458)
!2564 = !DILocation(line: 1080, column: 23, scope: !2458)
!2565 = !DILocation(line: 1080, column: 13, scope: !2458)
!2566 = !DILocation(line: 1080, column: 29, scope: !2458)
!2567 = !DILocation(line: 1080, column: 30, scope: !2458)
!2568 = !DILocation(line: 1080, column: 33, scope: !2458)
!2569 = !DILocation(line: 1080, column: 27, scope: !2458)
!2570 = !DILocation(line: 1080, column: 38, scope: !2458)
!2571 = !DILocation(line: 1080, column: 43, scope: !2458)
!2572 = !DILocation(line: 1082, column: 19, scope: !2458)
!2573 = !DILocation(line: 1082, column: 30, scope: !2458)
!2574 = !DILocation(line: 1082, column: 28, scope: !2458)
!2575 = !DILocation(line: 1082, column: 17, scope: !2458)
!2576 = !DILocation(line: 1083, column: 19, scope: !2458)
!2577 = !DILocation(line: 1083, column: 30, scope: !2458)
!2578 = !DILocation(line: 1083, column: 28, scope: !2458)
!2579 = !DILocation(line: 1083, column: 17, scope: !2458)
!2580 = !DILocation(line: 1086, column: 46, scope: !2458)
!2581 = !DILocation(line: 1086, column: 57, scope: !2458)
!2582 = !DILocation(line: 1086, column: 55, scope: !2458)
!2583 = !DILocation(line: 1086, column: 72, scope: !2458)
!2584 = !DILocation(line: 1086, column: 83, scope: !2458)
!2585 = !DILocation(line: 1086, column: 81, scope: !2458)
!2586 = !DILocation(line: 1086, column: 69, scope: !2458)
!2587 = !DILocation(line: 1086, column: 6, scope: !2458)
!2588 = !DILocation(line: 1086, column: 11, scope: !2458)
!2589 = !DILocation(line: 1086, column: 16, scope: !2458)
!2590 = !DILocation(line: 1086, column: 20, scope: !2458)
!2591 = !DILocation(line: 1086, column: 19, scope: !2458)
!2592 = !DILocation(line: 1086, column: 23, scope: !2458)
!2593 = !DILocation(line: 1086, column: 13, scope: !2458)
!2594 = !DILocation(line: 1086, column: 29, scope: !2458)
!2595 = !DILocation(line: 1086, column: 30, scope: !2458)
!2596 = !DILocation(line: 1086, column: 33, scope: !2458)
!2597 = !DILocation(line: 1086, column: 27, scope: !2458)
!2598 = !DILocation(line: 1086, column: 38, scope: !2458)
!2599 = !DILocation(line: 1086, column: 43, scope: !2458)
!2600 = !DILocation(line: 1087, column: 46, scope: !2458)
!2601 = !DILocation(line: 1087, column: 57, scope: !2458)
!2602 = !DILocation(line: 1087, column: 55, scope: !2458)
!2603 = !DILocation(line: 1087, column: 72, scope: !2458)
!2604 = !DILocation(line: 1087, column: 83, scope: !2458)
!2605 = !DILocation(line: 1087, column: 81, scope: !2458)
!2606 = !DILocation(line: 1087, column: 69, scope: !2458)
!2607 = !DILocation(line: 1087, column: 6, scope: !2458)
!2608 = !DILocation(line: 1087, column: 11, scope: !2458)
!2609 = !DILocation(line: 1087, column: 16, scope: !2458)
!2610 = !DILocation(line: 1087, column: 20, scope: !2458)
!2611 = !DILocation(line: 1087, column: 19, scope: !2458)
!2612 = !DILocation(line: 1087, column: 23, scope: !2458)
!2613 = !DILocation(line: 1087, column: 13, scope: !2458)
!2614 = !DILocation(line: 1087, column: 29, scope: !2458)
!2615 = !DILocation(line: 1087, column: 30, scope: !2458)
!2616 = !DILocation(line: 1087, column: 33, scope: !2458)
!2617 = !DILocation(line: 1087, column: 27, scope: !2458)
!2618 = !DILocation(line: 1087, column: 38, scope: !2458)
!2619 = !DILocation(line: 1087, column: 43, scope: !2458)
!2620 = !DILocation(line: 1088, column: 5, scope: !2458)
!2621 = !DILocation(line: 1061, column: 27, scope: !2452)
!2622 = !DILocation(line: 1061, column: 5, scope: !2452)
!2623 = distinct !{!2623, !2456, !2624}
!2624 = !DILocation(line: 1088, column: 5, scope: !2448)
!2625 = !DILocation(line: 1089, column: 4, scope: !2449)
!2626 = !DILocation(line: 1060, column: 26, scope: !2442)
!2627 = !DILocation(line: 1060, column: 4, scope: !2442)
!2628 = distinct !{!2628, !2446, !2629}
!2629 = !DILocation(line: 1089, column: 4, scope: !2439)
!2630 = !DILocation(line: 1091, column: 2, scope: !2151)
!2631 = !DILocation(line: 1010, column: 22, scope: !2146)
!2632 = !DILocation(line: 1010, column: 2, scope: !2146)
!2633 = distinct !{!2633, !2149, !2634}
!2634 = !DILocation(line: 1091, column: 2, scope: !2143)
!2635 = !DILocation(line: 1092, column: 1, scope: !2045)
!2636 = distinct !DISubprogram(name: "cffts2_gpu_kernel_3", linkageName: "_Z19cffts2_gpu_kernel_3P8dcomplexS0_", scope: !3, file: !3, line: 1101, type: !1153, scopeLine: 1102, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !1050, retainedNodes: !1051)
!2637 = !DILocalVariable(name: "x_out", arg: 1, scope: !2636, file: !3, line: 1101, type: !98)
!2638 = !DILocation(line: 1101, column: 46, scope: !2636)
!2639 = !DILocalVariable(name: "y0", arg: 2, scope: !2636, file: !3, line: 1102, type: !98)
!2640 = !DILocation(line: 1102, column: 12, scope: !2636)
!2641 = !DILocalVariable(name: "x_y_z", scope: !2636, file: !3, line: 1103, type: !97)
!2642 = !DILocation(line: 1103, column: 6, scope: !2636)
!2643 = !DILocation(line: 64, column: 3, scope: !1162, inlinedAt: !2644)
!2644 = distinct !DILocation(line: 1103, column: 14, scope: !2636)
!2645 = !DILocation(line: 75, column: 3, scope: !1200, inlinedAt: !2646)
!2646 = distinct !DILocation(line: 1103, column: 27, scope: !2636)
!2647 = !DILocation(line: 1103, column: 25, scope: !2636)
!2648 = !DILocation(line: 53, column: 3, scope: !1246, inlinedAt: !2649)
!2649 = distinct !DILocation(line: 1103, column: 40, scope: !2636)
!2650 = !DILocation(line: 1103, column: 38, scope: !2636)
!2651 = !DILocation(line: 1104, column: 5, scope: !2652)
!2652 = distinct !DILexicalBlock(scope: !2636, file: !3, line: 1104, column: 5)
!2653 = !DILocation(line: 1104, column: 11, scope: !2652)
!2654 = !DILocation(line: 1104, column: 5, scope: !2636)
!2655 = !DILocation(line: 1105, column: 3, scope: !2656)
!2656 = distinct !DILexicalBlock(scope: !2652, file: !3, line: 1104, column: 25)
!2657 = !DILocation(line: 1107, column: 22, scope: !2636)
!2658 = !DILocation(line: 1107, column: 25, scope: !2636)
!2659 = !DILocation(line: 1107, column: 32, scope: !2636)
!2660 = !DILocation(line: 1107, column: 2, scope: !2636)
!2661 = !DILocation(line: 1107, column: 8, scope: !2636)
!2662 = !DILocation(line: 1107, column: 15, scope: !2636)
!2663 = !DILocation(line: 1107, column: 20, scope: !2636)
!2664 = !DILocation(line: 1108, column: 22, scope: !2636)
!2665 = !DILocation(line: 1108, column: 25, scope: !2636)
!2666 = !DILocation(line: 1108, column: 32, scope: !2636)
!2667 = !DILocation(line: 1108, column: 2, scope: !2636)
!2668 = !DILocation(line: 1108, column: 8, scope: !2636)
!2669 = !DILocation(line: 1108, column: 15, scope: !2636)
!2670 = !DILocation(line: 1108, column: 20, scope: !2636)
!2671 = !DILocation(line: 1109, column: 1, scope: !2636)
!2672 = distinct !DISubprogram(name: "cffts3_gpu_cfftz_device", linkageName: "_Z23cffts3_gpu_cfftz_deviceiiiP8dcomplexS0_S0_ii", scope: !3, file: !3, line: 1162, type: !2673, scopeLine: 1169, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !1050, retainedNodes: !1051)
!2673 = !DISubroutineType(types: !2674)
!2674 = !{null, !1326, !97, !97, !98, !98, !98, !97, !97}
!2675 = !DILocalVariable(name: "is", arg: 1, scope: !2672, file: !3, line: 1162, type: !1326)
!2676 = !DILocation(line: 1162, column: 51, scope: !2672)
!2677 = !DILocalVariable(name: "m", arg: 2, scope: !2672, file: !3, line: 1163, type: !97)
!2678 = !DILocation(line: 1163, column: 7, scope: !2672)
!2679 = !DILocalVariable(name: "n", arg: 3, scope: !2672, file: !3, line: 1164, type: !97)
!2680 = !DILocation(line: 1164, column: 7, scope: !2672)
!2681 = !DILocalVariable(name: "x", arg: 4, scope: !2672, file: !3, line: 1165, type: !98)
!2682 = !DILocation(line: 1165, column: 12, scope: !2672)
!2683 = !DILocalVariable(name: "y", arg: 5, scope: !2672, file: !3, line: 1166, type: !98)
!2684 = !DILocation(line: 1166, column: 12, scope: !2672)
!2685 = !DILocalVariable(name: "u_device", arg: 6, scope: !2672, file: !3, line: 1167, type: !98)
!2686 = !DILocation(line: 1167, column: 12, scope: !2672)
!2687 = !DILocalVariable(name: "index_arg", arg: 7, scope: !2672, file: !3, line: 1168, type: !97)
!2688 = !DILocation(line: 1168, column: 7, scope: !2672)
!2689 = !DILocalVariable(name: "size_arg", arg: 8, scope: !2672, file: !3, line: 1169, type: !97)
!2690 = !DILocation(line: 1169, column: 7, scope: !2672)
!2691 = !DILocalVariable(name: "j", scope: !2672, file: !3, line: 1170, type: !97)
!2692 = !DILocation(line: 1170, column: 6, scope: !2672)
!2693 = !DILocalVariable(name: "l", scope: !2672, file: !3, line: 1170, type: !97)
!2694 = !DILocation(line: 1170, column: 8, scope: !2672)
!2695 = !DILocation(line: 1176, column: 7, scope: !2696)
!2696 = distinct !DILexicalBlock(scope: !2672, file: !3, line: 1176, column: 2)
!2697 = !DILocation(line: 1176, column: 6, scope: !2696)
!2698 = !DILocation(line: 1176, column: 11, scope: !2699)
!2699 = distinct !DILexicalBlock(scope: !2696, file: !3, line: 1176, column: 2)
!2700 = !DILocation(line: 1176, column: 14, scope: !2699)
!2701 = !DILocation(line: 1176, column: 12, scope: !2699)
!2702 = !DILocation(line: 1176, column: 2, scope: !2696)
!2703 = !DILocation(line: 1177, column: 27, scope: !2704)
!2704 = distinct !DILexicalBlock(scope: !2699, file: !3, line: 1176, column: 22)
!2705 = !DILocation(line: 1177, column: 31, scope: !2704)
!2706 = !DILocation(line: 1177, column: 34, scope: !2704)
!2707 = !DILocation(line: 1177, column: 37, scope: !2704)
!2708 = !DILocation(line: 1177, column: 40, scope: !2704)
!2709 = !DILocation(line: 1177, column: 50, scope: !2704)
!2710 = !DILocation(line: 1177, column: 53, scope: !2704)
!2711 = !DILocation(line: 1177, column: 56, scope: !2704)
!2712 = !DILocation(line: 1177, column: 67, scope: !2704)
!2713 = !DILocation(line: 1177, column: 3, scope: !2704)
!2714 = !DILocation(line: 1178, column: 6, scope: !2715)
!2715 = distinct !DILexicalBlock(scope: !2704, file: !3, line: 1178, column: 6)
!2716 = !DILocation(line: 1178, column: 9, scope: !2715)
!2717 = !DILocation(line: 1178, column: 7, scope: !2715)
!2718 = !DILocation(line: 1178, column: 6, scope: !2704)
!2719 = !DILocation(line: 1178, column: 12, scope: !2720)
!2720 = distinct !DILexicalBlock(scope: !2715, file: !3, line: 1178, column: 11)
!2721 = !DILocation(line: 1179, column: 27, scope: !2704)
!2722 = !DILocation(line: 1179, column: 31, scope: !2704)
!2723 = !DILocation(line: 1179, column: 33, scope: !2704)
!2724 = !DILocation(line: 1179, column: 38, scope: !2704)
!2725 = !DILocation(line: 1179, column: 41, scope: !2704)
!2726 = !DILocation(line: 1179, column: 44, scope: !2704)
!2727 = !DILocation(line: 1179, column: 54, scope: !2704)
!2728 = !DILocation(line: 1179, column: 57, scope: !2704)
!2729 = !DILocation(line: 1179, column: 60, scope: !2704)
!2730 = !DILocation(line: 1179, column: 71, scope: !2704)
!2731 = !DILocation(line: 1179, column: 3, scope: !2704)
!2732 = !DILocation(line: 1180, column: 2, scope: !2704)
!2733 = !DILocation(line: 1176, column: 18, scope: !2699)
!2734 = !DILocation(line: 1176, column: 2, scope: !2699)
!2735 = distinct !{!2735, !2702, !2736}
!2736 = !DILocation(line: 1180, column: 2, scope: !2696)
!2737 = !DILocation(line: 1186, column: 5, scope: !2738)
!2738 = distinct !DILexicalBlock(scope: !2672, file: !3, line: 1186, column: 5)
!2739 = !DILocation(line: 1186, column: 6, scope: !2738)
!2740 = !DILocation(line: 1186, column: 8, scope: !2738)
!2741 = !DILocation(line: 1186, column: 5, scope: !2672)
!2742 = !DILocation(line: 1187, column: 8, scope: !2743)
!2743 = distinct !DILexicalBlock(scope: !2744, file: !3, line: 1187, column: 3)
!2744 = distinct !DILexicalBlock(scope: !2738, file: !3, line: 1186, column: 12)
!2745 = !DILocation(line: 1187, column: 7, scope: !2743)
!2746 = !DILocation(line: 1187, column: 12, scope: !2747)
!2747 = distinct !DILexicalBlock(scope: !2743, file: !3, line: 1187, column: 3)
!2748 = !DILocation(line: 1187, column: 14, scope: !2747)
!2749 = !DILocation(line: 1187, column: 13, scope: !2747)
!2750 = !DILocation(line: 1187, column: 3, scope: !2743)
!2751 = !DILocation(line: 1188, column: 35, scope: !2752)
!2752 = distinct !DILexicalBlock(scope: !2747, file: !3, line: 1187, column: 21)
!2753 = !DILocation(line: 1188, column: 37, scope: !2752)
!2754 = !DILocation(line: 1188, column: 39, scope: !2752)
!2755 = !DILocation(line: 1188, column: 38, scope: !2752)
!2756 = !DILocation(line: 1188, column: 48, scope: !2752)
!2757 = !DILocation(line: 1188, column: 47, scope: !2752)
!2758 = !DILocation(line: 1188, column: 59, scope: !2752)
!2759 = !DILocation(line: 1188, column: 4, scope: !2752)
!2760 = !DILocation(line: 1188, column: 6, scope: !2752)
!2761 = !DILocation(line: 1188, column: 8, scope: !2752)
!2762 = !DILocation(line: 1188, column: 7, scope: !2752)
!2763 = !DILocation(line: 1188, column: 17, scope: !2752)
!2764 = !DILocation(line: 1188, column: 16, scope: !2752)
!2765 = !DILocation(line: 1188, column: 28, scope: !2752)
!2766 = !DILocation(line: 1188, column: 33, scope: !2752)
!2767 = !DILocation(line: 1189, column: 35, scope: !2752)
!2768 = !DILocation(line: 1189, column: 37, scope: !2752)
!2769 = !DILocation(line: 1189, column: 39, scope: !2752)
!2770 = !DILocation(line: 1189, column: 38, scope: !2752)
!2771 = !DILocation(line: 1189, column: 48, scope: !2752)
!2772 = !DILocation(line: 1189, column: 47, scope: !2752)
!2773 = !DILocation(line: 1189, column: 59, scope: !2752)
!2774 = !DILocation(line: 1189, column: 4, scope: !2752)
!2775 = !DILocation(line: 1189, column: 6, scope: !2752)
!2776 = !DILocation(line: 1189, column: 8, scope: !2752)
!2777 = !DILocation(line: 1189, column: 7, scope: !2752)
!2778 = !DILocation(line: 1189, column: 17, scope: !2752)
!2779 = !DILocation(line: 1189, column: 16, scope: !2752)
!2780 = !DILocation(line: 1189, column: 28, scope: !2752)
!2781 = !DILocation(line: 1189, column: 33, scope: !2752)
!2782 = !DILocation(line: 1190, column: 3, scope: !2752)
!2783 = !DILocation(line: 1187, column: 18, scope: !2747)
!2784 = !DILocation(line: 1187, column: 3, scope: !2747)
!2785 = distinct !{!2785, !2750, !2786}
!2786 = !DILocation(line: 1190, column: 3, scope: !2743)
!2787 = !DILocation(line: 1191, column: 2, scope: !2744)
!2788 = !DILocation(line: 1192, column: 1, scope: !2672)
!2789 = distinct !DISubprogram(name: "cffts3_gpu_fftz2_device", linkageName: "_Z23cffts3_gpu_fftz2_deviceiiiiP8dcomplexS0_S0_ii", scope: !3, file: !3, line: 1203, type: !2790, scopeLine: 1211, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !1050, retainedNodes: !1051)
!2790 = !DISubroutineType(types: !2791)
!2791 = !{null, !1326, !97, !97, !97, !98, !98, !98, !97, !97}
!2792 = !DILocalVariable(name: "is", arg: 1, scope: !2789, file: !3, line: 1203, type: !1326)
!2793 = !DILocation(line: 1203, column: 51, scope: !2789)
!2794 = !DILocalVariable(name: "l", arg: 2, scope: !2789, file: !3, line: 1204, type: !97)
!2795 = !DILocation(line: 1204, column: 7, scope: !2789)
!2796 = !DILocalVariable(name: "m", arg: 3, scope: !2789, file: !3, line: 1205, type: !97)
!2797 = !DILocation(line: 1205, column: 7, scope: !2789)
!2798 = !DILocalVariable(name: "n", arg: 4, scope: !2789, file: !3, line: 1206, type: !97)
!2799 = !DILocation(line: 1206, column: 7, scope: !2789)
!2800 = !DILocalVariable(name: "u", arg: 5, scope: !2789, file: !3, line: 1207, type: !98)
!2801 = !DILocation(line: 1207, column: 12, scope: !2789)
!2802 = !DILocalVariable(name: "x", arg: 6, scope: !2789, file: !3, line: 1208, type: !98)
!2803 = !DILocation(line: 1208, column: 12, scope: !2789)
!2804 = !DILocalVariable(name: "y", arg: 7, scope: !2789, file: !3, line: 1209, type: !98)
!2805 = !DILocation(line: 1209, column: 12, scope: !2789)
!2806 = !DILocalVariable(name: "index_arg", arg: 8, scope: !2789, file: !3, line: 1210, type: !97)
!2807 = !DILocation(line: 1210, column: 7, scope: !2789)
!2808 = !DILocalVariable(name: "size_arg", arg: 9, scope: !2789, file: !3, line: 1211, type: !97)
!2809 = !DILocation(line: 1211, column: 7, scope: !2789)
!2810 = !DILocalVariable(name: "k", scope: !2789, file: !3, line: 1212, type: !97)
!2811 = !DILocation(line: 1212, column: 6, scope: !2789)
!2812 = !DILocalVariable(name: "n1", scope: !2789, file: !3, line: 1212, type: !97)
!2813 = !DILocation(line: 1212, column: 8, scope: !2789)
!2814 = !DILocalVariable(name: "li", scope: !2789, file: !3, line: 1212, type: !97)
!2815 = !DILocation(line: 1212, column: 11, scope: !2789)
!2816 = !DILocalVariable(name: "lj", scope: !2789, file: !3, line: 1212, type: !97)
!2817 = !DILocation(line: 1212, column: 14, scope: !2789)
!2818 = !DILocalVariable(name: "lk", scope: !2789, file: !3, line: 1212, type: !97)
!2819 = !DILocation(line: 1212, column: 17, scope: !2789)
!2820 = !DILocalVariable(name: "ku", scope: !2789, file: !3, line: 1212, type: !97)
!2821 = !DILocation(line: 1212, column: 20, scope: !2789)
!2822 = !DILocalVariable(name: "i", scope: !2789, file: !3, line: 1212, type: !97)
!2823 = !DILocation(line: 1212, column: 23, scope: !2789)
!2824 = !DILocalVariable(name: "i11", scope: !2789, file: !3, line: 1212, type: !97)
!2825 = !DILocation(line: 1212, column: 25, scope: !2789)
!2826 = !DILocalVariable(name: "i12", scope: !2789, file: !3, line: 1212, type: !97)
!2827 = !DILocation(line: 1212, column: 29, scope: !2789)
!2828 = !DILocalVariable(name: "i21", scope: !2789, file: !3, line: 1212, type: !97)
!2829 = !DILocation(line: 1212, column: 33, scope: !2789)
!2830 = !DILocalVariable(name: "i22", scope: !2789, file: !3, line: 1212, type: !97)
!2831 = !DILocation(line: 1212, column: 37, scope: !2789)
!2832 = !DILocalVariable(name: "x11real", scope: !2789, file: !3, line: 1213, type: !104)
!2833 = !DILocation(line: 1213, column: 9, scope: !2789)
!2834 = !DILocalVariable(name: "x11imag", scope: !2789, file: !3, line: 1213, type: !104)
!2835 = !DILocation(line: 1213, column: 18, scope: !2789)
!2836 = !DILocalVariable(name: "x21real", scope: !2789, file: !3, line: 1214, type: !104)
!2837 = !DILocation(line: 1214, column: 9, scope: !2789)
!2838 = !DILocalVariable(name: "x21imag", scope: !2789, file: !3, line: 1214, type: !104)
!2839 = !DILocation(line: 1214, column: 18, scope: !2789)
!2840 = !DILocalVariable(name: "u1", scope: !2789, file: !3, line: 1215, type: !99)
!2841 = !DILocation(line: 1215, column: 11, scope: !2789)
!2842 = !DILocation(line: 1221, column: 7, scope: !2789)
!2843 = !DILocation(line: 1221, column: 9, scope: !2789)
!2844 = !DILocation(line: 1221, column: 5, scope: !2789)
!2845 = !DILocation(line: 1222, column: 13, scope: !2789)
!2846 = !DILocation(line: 1222, column: 15, scope: !2789)
!2847 = !DILocation(line: 1222, column: 9, scope: !2789)
!2848 = !DILocation(line: 1222, column: 5, scope: !2789)
!2849 = !DILocation(line: 1223, column: 13, scope: !2789)
!2850 = !DILocation(line: 1223, column: 17, scope: !2789)
!2851 = !DILocation(line: 1223, column: 15, scope: !2789)
!2852 = !DILocation(line: 1223, column: 9, scope: !2789)
!2853 = !DILocation(line: 1223, column: 5, scope: !2789)
!2854 = !DILocation(line: 1224, column: 11, scope: !2789)
!2855 = !DILocation(line: 1224, column: 9, scope: !2789)
!2856 = !DILocation(line: 1224, column: 5, scope: !2789)
!2857 = !DILocation(line: 1225, column: 7, scope: !2789)
!2858 = !DILocation(line: 1225, column: 5, scope: !2789)
!2859 = !DILocation(line: 1226, column: 7, scope: !2860)
!2860 = distinct !DILexicalBlock(scope: !2789, file: !3, line: 1226, column: 2)
!2861 = !DILocation(line: 1226, column: 6, scope: !2860)
!2862 = !DILocation(line: 1226, column: 11, scope: !2863)
!2863 = distinct !DILexicalBlock(scope: !2860, file: !3, line: 1226, column: 2)
!2864 = !DILocation(line: 1226, column: 13, scope: !2863)
!2865 = !DILocation(line: 1226, column: 12, scope: !2863)
!2866 = !DILocation(line: 1226, column: 2, scope: !2860)
!2867 = !DILocation(line: 1227, column: 9, scope: !2868)
!2868 = distinct !DILexicalBlock(scope: !2863, file: !3, line: 1226, column: 21)
!2869 = !DILocation(line: 1227, column: 13, scope: !2868)
!2870 = !DILocation(line: 1227, column: 11, scope: !2868)
!2871 = !DILocation(line: 1227, column: 7, scope: !2868)
!2872 = !DILocation(line: 1228, column: 9, scope: !2868)
!2873 = !DILocation(line: 1228, column: 15, scope: !2868)
!2874 = !DILocation(line: 1228, column: 13, scope: !2868)
!2875 = !DILocation(line: 1228, column: 7, scope: !2868)
!2876 = !DILocation(line: 1229, column: 9, scope: !2868)
!2877 = !DILocation(line: 1229, column: 13, scope: !2868)
!2878 = !DILocation(line: 1229, column: 11, scope: !2868)
!2879 = !DILocation(line: 1229, column: 7, scope: !2868)
!2880 = !DILocation(line: 1230, column: 9, scope: !2868)
!2881 = !DILocation(line: 1230, column: 15, scope: !2868)
!2882 = !DILocation(line: 1230, column: 13, scope: !2868)
!2883 = !DILocation(line: 1230, column: 7, scope: !2868)
!2884 = !DILocation(line: 1231, column: 6, scope: !2885)
!2885 = distinct !DILexicalBlock(scope: !2868, file: !3, line: 1231, column: 6)
!2886 = !DILocation(line: 1231, column: 8, scope: !2885)
!2887 = !DILocation(line: 1231, column: 6, scope: !2868)
!2888 = !DILocation(line: 1232, column: 14, scope: !2889)
!2889 = distinct !DILexicalBlock(scope: !2885, file: !3, line: 1231, column: 12)
!2890 = !DILocation(line: 1232, column: 16, scope: !2889)
!2891 = !DILocation(line: 1232, column: 19, scope: !2889)
!2892 = !DILocation(line: 1232, column: 18, scope: !2889)
!2893 = !DILocation(line: 1232, column: 22, scope: !2889)
!2894 = !DILocation(line: 1232, column: 7, scope: !2889)
!2895 = !DILocation(line: 1232, column: 12, scope: !2889)
!2896 = !DILocation(line: 1233, column: 14, scope: !2889)
!2897 = !DILocation(line: 1233, column: 16, scope: !2889)
!2898 = !DILocation(line: 1233, column: 19, scope: !2889)
!2899 = !DILocation(line: 1233, column: 18, scope: !2889)
!2900 = !DILocation(line: 1233, column: 22, scope: !2889)
!2901 = !DILocation(line: 1233, column: 7, scope: !2889)
!2902 = !DILocation(line: 1233, column: 12, scope: !2889)
!2903 = !DILocation(line: 1234, column: 3, scope: !2889)
!2904 = !DILocation(line: 1235, column: 14, scope: !2905)
!2905 = distinct !DILexicalBlock(scope: !2885, file: !3, line: 1234, column: 8)
!2906 = !DILocation(line: 1235, column: 16, scope: !2905)
!2907 = !DILocation(line: 1235, column: 19, scope: !2905)
!2908 = !DILocation(line: 1235, column: 18, scope: !2905)
!2909 = !DILocation(line: 1235, column: 22, scope: !2905)
!2910 = !DILocation(line: 1235, column: 7, scope: !2905)
!2911 = !DILocation(line: 1235, column: 12, scope: !2905)
!2912 = !DILocation(line: 1236, column: 15, scope: !2905)
!2913 = !DILocation(line: 1236, column: 17, scope: !2905)
!2914 = !DILocation(line: 1236, column: 20, scope: !2905)
!2915 = !DILocation(line: 1236, column: 19, scope: !2905)
!2916 = !DILocation(line: 1236, column: 23, scope: !2905)
!2917 = !DILocation(line: 1236, column: 14, scope: !2905)
!2918 = !DILocation(line: 1236, column: 7, scope: !2905)
!2919 = !DILocation(line: 1236, column: 12, scope: !2905)
!2920 = !DILocation(line: 1238, column: 8, scope: !2921)
!2921 = distinct !DILexicalBlock(scope: !2868, file: !3, line: 1238, column: 3)
!2922 = !DILocation(line: 1238, column: 7, scope: !2921)
!2923 = !DILocation(line: 1238, column: 12, scope: !2924)
!2924 = distinct !DILexicalBlock(scope: !2921, file: !3, line: 1238, column: 3)
!2925 = !DILocation(line: 1238, column: 14, scope: !2924)
!2926 = !DILocation(line: 1238, column: 13, scope: !2924)
!2927 = !DILocation(line: 1238, column: 3, scope: !2921)
!2928 = !DILocation(line: 1239, column: 14, scope: !2929)
!2929 = distinct !DILexicalBlock(scope: !2924, file: !3, line: 1238, column: 22)
!2930 = !DILocation(line: 1239, column: 17, scope: !2929)
!2931 = !DILocation(line: 1239, column: 21, scope: !2929)
!2932 = !DILocation(line: 1239, column: 20, scope: !2929)
!2933 = !DILocation(line: 1239, column: 24, scope: !2929)
!2934 = !DILocation(line: 1239, column: 23, scope: !2929)
!2935 = !DILocation(line: 1239, column: 33, scope: !2929)
!2936 = !DILocation(line: 1239, column: 32, scope: !2929)
!2937 = !DILocation(line: 1239, column: 44, scope: !2929)
!2938 = !DILocation(line: 1239, column: 12, scope: !2929)
!2939 = !DILocation(line: 1240, column: 14, scope: !2929)
!2940 = !DILocation(line: 1240, column: 17, scope: !2929)
!2941 = !DILocation(line: 1240, column: 21, scope: !2929)
!2942 = !DILocation(line: 1240, column: 20, scope: !2929)
!2943 = !DILocation(line: 1240, column: 24, scope: !2929)
!2944 = !DILocation(line: 1240, column: 23, scope: !2929)
!2945 = !DILocation(line: 1240, column: 33, scope: !2929)
!2946 = !DILocation(line: 1240, column: 32, scope: !2929)
!2947 = !DILocation(line: 1240, column: 44, scope: !2929)
!2948 = !DILocation(line: 1240, column: 12, scope: !2929)
!2949 = !DILocation(line: 1241, column: 14, scope: !2929)
!2950 = !DILocation(line: 1241, column: 17, scope: !2929)
!2951 = !DILocation(line: 1241, column: 21, scope: !2929)
!2952 = !DILocation(line: 1241, column: 20, scope: !2929)
!2953 = !DILocation(line: 1241, column: 24, scope: !2929)
!2954 = !DILocation(line: 1241, column: 23, scope: !2929)
!2955 = !DILocation(line: 1241, column: 33, scope: !2929)
!2956 = !DILocation(line: 1241, column: 32, scope: !2929)
!2957 = !DILocation(line: 1241, column: 44, scope: !2929)
!2958 = !DILocation(line: 1241, column: 12, scope: !2929)
!2959 = !DILocation(line: 1242, column: 14, scope: !2929)
!2960 = !DILocation(line: 1242, column: 17, scope: !2929)
!2961 = !DILocation(line: 1242, column: 21, scope: !2929)
!2962 = !DILocation(line: 1242, column: 20, scope: !2929)
!2963 = !DILocation(line: 1242, column: 24, scope: !2929)
!2964 = !DILocation(line: 1242, column: 23, scope: !2929)
!2965 = !DILocation(line: 1242, column: 33, scope: !2929)
!2966 = !DILocation(line: 1242, column: 32, scope: !2929)
!2967 = !DILocation(line: 1242, column: 44, scope: !2929)
!2968 = !DILocation(line: 1242, column: 12, scope: !2929)
!2969 = !DILocation(line: 1243, column: 41, scope: !2929)
!2970 = !DILocation(line: 1243, column: 51, scope: !2929)
!2971 = !DILocation(line: 1243, column: 49, scope: !2929)
!2972 = !DILocation(line: 1243, column: 4, scope: !2929)
!2973 = !DILocation(line: 1243, column: 7, scope: !2929)
!2974 = !DILocation(line: 1243, column: 11, scope: !2929)
!2975 = !DILocation(line: 1243, column: 10, scope: !2929)
!2976 = !DILocation(line: 1243, column: 14, scope: !2929)
!2977 = !DILocation(line: 1243, column: 13, scope: !2929)
!2978 = !DILocation(line: 1243, column: 23, scope: !2929)
!2979 = !DILocation(line: 1243, column: 22, scope: !2929)
!2980 = !DILocation(line: 1243, column: 34, scope: !2929)
!2981 = !DILocation(line: 1243, column: 39, scope: !2929)
!2982 = !DILocation(line: 1244, column: 41, scope: !2929)
!2983 = !DILocation(line: 1244, column: 51, scope: !2929)
!2984 = !DILocation(line: 1244, column: 49, scope: !2929)
!2985 = !DILocation(line: 1244, column: 4, scope: !2929)
!2986 = !DILocation(line: 1244, column: 7, scope: !2929)
!2987 = !DILocation(line: 1244, column: 11, scope: !2929)
!2988 = !DILocation(line: 1244, column: 10, scope: !2929)
!2989 = !DILocation(line: 1244, column: 14, scope: !2929)
!2990 = !DILocation(line: 1244, column: 13, scope: !2929)
!2991 = !DILocation(line: 1244, column: 23, scope: !2929)
!2992 = !DILocation(line: 1244, column: 22, scope: !2929)
!2993 = !DILocation(line: 1244, column: 34, scope: !2929)
!2994 = !DILocation(line: 1244, column: 39, scope: !2929)
!2995 = !DILocation(line: 1245, column: 44, scope: !2929)
!2996 = !DILocation(line: 1245, column: 52, scope: !2929)
!2997 = !DILocation(line: 1245, column: 62, scope: !2929)
!2998 = !DILocation(line: 1245, column: 60, scope: !2929)
!2999 = !DILocation(line: 1245, column: 49, scope: !2929)
!3000 = !DILocation(line: 1245, column: 76, scope: !2929)
!3001 = !DILocation(line: 1245, column: 84, scope: !2929)
!3002 = !DILocation(line: 1245, column: 94, scope: !2929)
!3003 = !DILocation(line: 1245, column: 92, scope: !2929)
!3004 = !DILocation(line: 1245, column: 81, scope: !2929)
!3005 = !DILocation(line: 1245, column: 71, scope: !2929)
!3006 = !DILocation(line: 1245, column: 4, scope: !2929)
!3007 = !DILocation(line: 1245, column: 7, scope: !2929)
!3008 = !DILocation(line: 1245, column: 11, scope: !2929)
!3009 = !DILocation(line: 1245, column: 10, scope: !2929)
!3010 = !DILocation(line: 1245, column: 14, scope: !2929)
!3011 = !DILocation(line: 1245, column: 13, scope: !2929)
!3012 = !DILocation(line: 1245, column: 23, scope: !2929)
!3013 = !DILocation(line: 1245, column: 22, scope: !2929)
!3014 = !DILocation(line: 1245, column: 34, scope: !2929)
!3015 = !DILocation(line: 1245, column: 39, scope: !2929)
!3016 = !DILocation(line: 1246, column: 44, scope: !2929)
!3017 = !DILocation(line: 1246, column: 52, scope: !2929)
!3018 = !DILocation(line: 1246, column: 62, scope: !2929)
!3019 = !DILocation(line: 1246, column: 60, scope: !2929)
!3020 = !DILocation(line: 1246, column: 49, scope: !2929)
!3021 = !DILocation(line: 1246, column: 76, scope: !2929)
!3022 = !DILocation(line: 1246, column: 84, scope: !2929)
!3023 = !DILocation(line: 1246, column: 94, scope: !2929)
!3024 = !DILocation(line: 1246, column: 92, scope: !2929)
!3025 = !DILocation(line: 1246, column: 81, scope: !2929)
!3026 = !DILocation(line: 1246, column: 71, scope: !2929)
!3027 = !DILocation(line: 1246, column: 4, scope: !2929)
!3028 = !DILocation(line: 1246, column: 7, scope: !2929)
!3029 = !DILocation(line: 1246, column: 11, scope: !2929)
!3030 = !DILocation(line: 1246, column: 10, scope: !2929)
!3031 = !DILocation(line: 1246, column: 14, scope: !2929)
!3032 = !DILocation(line: 1246, column: 13, scope: !2929)
!3033 = !DILocation(line: 1246, column: 23, scope: !2929)
!3034 = !DILocation(line: 1246, column: 22, scope: !2929)
!3035 = !DILocation(line: 1246, column: 34, scope: !2929)
!3036 = !DILocation(line: 1246, column: 39, scope: !2929)
!3037 = !DILocation(line: 1247, column: 3, scope: !2929)
!3038 = !DILocation(line: 1238, column: 19, scope: !2924)
!3039 = !DILocation(line: 1238, column: 3, scope: !2924)
!3040 = distinct !{!3040, !2927, !3041}
!3041 = !DILocation(line: 1247, column: 3, scope: !2921)
!3042 = !DILocation(line: 1248, column: 2, scope: !2868)
!3043 = !DILocation(line: 1226, column: 18, scope: !2863)
!3044 = !DILocation(line: 1226, column: 2, scope: !2863)
!3045 = distinct !{!3045, !2866, !3046}
!3046 = !DILocation(line: 1248, column: 2, scope: !2860)
!3047 = !DILocation(line: 1249, column: 1, scope: !2789)
!3048 = distinct !DISubprogram(name: "cffts3_gpu_kernel_1", linkageName: "_Z19cffts3_gpu_kernel_1P8dcomplexS0_", scope: !3, file: !3, line: 1258, type: !1153, scopeLine: 1259, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !1050, retainedNodes: !1051)
!3049 = !DILocalVariable(name: "x_in", arg: 1, scope: !3048, file: !3, line: 1258, type: !98)
!3050 = !DILocation(line: 1258, column: 46, scope: !3048)
!3051 = !DILocalVariable(name: "y0", arg: 2, scope: !3048, file: !3, line: 1259, type: !98)
!3052 = !DILocation(line: 1259, column: 12, scope: !3048)
!3053 = !DILocalVariable(name: "x_y_z", scope: !3048, file: !3, line: 1260, type: !97)
!3054 = !DILocation(line: 1260, column: 6, scope: !3048)
!3055 = !DILocation(line: 64, column: 3, scope: !1162, inlinedAt: !3056)
!3056 = distinct !DILocation(line: 1260, column: 14, scope: !3048)
!3057 = !DILocation(line: 75, column: 3, scope: !1200, inlinedAt: !3058)
!3058 = distinct !DILocation(line: 1260, column: 27, scope: !3048)
!3059 = !DILocation(line: 1260, column: 25, scope: !3048)
!3060 = !DILocation(line: 53, column: 3, scope: !1246, inlinedAt: !3061)
!3061 = distinct !DILocation(line: 1260, column: 40, scope: !3048)
!3062 = !DILocation(line: 1260, column: 38, scope: !3048)
!3063 = !DILocation(line: 1261, column: 5, scope: !3064)
!3064 = distinct !DILexicalBlock(scope: !3048, file: !3, line: 1261, column: 5)
!3065 = !DILocation(line: 1261, column: 11, scope: !3064)
!3066 = !DILocation(line: 1261, column: 5, scope: !3048)
!3067 = !DILocation(line: 1262, column: 3, scope: !3068)
!3068 = distinct !DILexicalBlock(scope: !3064, file: !3, line: 1261, column: 25)
!3069 = !DILocation(line: 1264, column: 19, scope: !3048)
!3070 = !DILocation(line: 1264, column: 24, scope: !3048)
!3071 = !DILocation(line: 1264, column: 31, scope: !3048)
!3072 = !DILocation(line: 1264, column: 2, scope: !3048)
!3073 = !DILocation(line: 1264, column: 5, scope: !3048)
!3074 = !DILocation(line: 1264, column: 12, scope: !3048)
!3075 = !DILocation(line: 1264, column: 17, scope: !3048)
!3076 = !DILocation(line: 1265, column: 19, scope: !3048)
!3077 = !DILocation(line: 1265, column: 24, scope: !3048)
!3078 = !DILocation(line: 1265, column: 31, scope: !3048)
!3079 = !DILocation(line: 1265, column: 2, scope: !3048)
!3080 = !DILocation(line: 1265, column: 5, scope: !3048)
!3081 = !DILocation(line: 1265, column: 12, scope: !3048)
!3082 = !DILocation(line: 1265, column: 17, scope: !3048)
!3083 = !DILocation(line: 1266, column: 1, scope: !3048)
!3084 = distinct !DISubprogram(name: "cffts3_gpu_kernel_2", linkageName: "_Z19cffts3_gpu_kernel_2iP8dcomplexS0_S0_", scope: !3, file: !3, line: 1273, type: !1324, scopeLine: 1276, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !1050, retainedNodes: !1051)
!3085 = !DILocalVariable(name: "is", arg: 1, scope: !3084, file: !3, line: 1273, type: !1326)
!3086 = !DILocation(line: 1273, column: 47, scope: !3084)
!3087 = !DILocalVariable(name: "gty1", arg: 2, scope: !3084, file: !3, line: 1274, type: !98)
!3088 = !DILocation(line: 1274, column: 12, scope: !3084)
!3089 = !DILocalVariable(name: "gty2", arg: 3, scope: !3084, file: !3, line: 1275, type: !98)
!3090 = !DILocation(line: 1275, column: 12, scope: !3084)
!3091 = !DILocalVariable(name: "u_device", arg: 4, scope: !3084, file: !3, line: 1276, type: !98)
!3092 = !DILocation(line: 1276, column: 12, scope: !3084)
!3093 = !DILocalVariable(name: "x_y", scope: !3084, file: !3, line: 1277, type: !97)
!3094 = !DILocation(line: 1277, column: 6, scope: !3084)
!3095 = !DILocation(line: 64, column: 3, scope: !1162, inlinedAt: !3096)
!3096 = distinct !DILocation(line: 1277, column: 12, scope: !3084)
!3097 = !DILocation(line: 75, column: 3, scope: !1200, inlinedAt: !3098)
!3098 = distinct !DILocation(line: 1277, column: 25, scope: !3084)
!3099 = !DILocation(line: 1277, column: 23, scope: !3084)
!3100 = !DILocation(line: 53, column: 3, scope: !1246, inlinedAt: !3101)
!3101 = distinct !DILocation(line: 1277, column: 38, scope: !3084)
!3102 = !DILocation(line: 1277, column: 36, scope: !3084)
!3103 = !DILocation(line: 1278, column: 5, scope: !3104)
!3104 = distinct !DILexicalBlock(scope: !3084, file: !3, line: 1278, column: 5)
!3105 = !DILocation(line: 1278, column: 9, scope: !3104)
!3106 = !DILocation(line: 1278, column: 5, scope: !3084)
!3107 = !DILocation(line: 1279, column: 3, scope: !3108)
!3108 = distinct !DILexicalBlock(scope: !3104, file: !3, line: 1278, column: 20)
!3109 = !DILocation(line: 1281, column: 26, scope: !3084)
!3110 = !DILocation(line: 1282, column: 4, scope: !3084)
!3111 = !DILocation(line: 1284, column: 4, scope: !3084)
!3112 = !DILocation(line: 1285, column: 4, scope: !3084)
!3113 = !DILocation(line: 1286, column: 4, scope: !3084)
!3114 = !DILocation(line: 1287, column: 4, scope: !3084)
!3115 = !DILocation(line: 1281, column: 2, scope: !3084)
!3116 = !DILocation(line: 1289, column: 1, scope: !3084)
!3117 = distinct !DISubprogram(name: "cffts3_gpu_kernel_3", linkageName: "_Z19cffts3_gpu_kernel_3P8dcomplexS0_", scope: !3, file: !3, line: 1298, type: !1153, scopeLine: 1299, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !1050, retainedNodes: !1051)
!3118 = !DILocalVariable(name: "x_out", arg: 1, scope: !3117, file: !3, line: 1298, type: !98)
!3119 = !DILocation(line: 1298, column: 46, scope: !3117)
!3120 = !DILocalVariable(name: "y0", arg: 2, scope: !3117, file: !3, line: 1299, type: !98)
!3121 = !DILocation(line: 1299, column: 12, scope: !3117)
!3122 = !DILocalVariable(name: "x_y_z", scope: !3117, file: !3, line: 1300, type: !97)
!3123 = !DILocation(line: 1300, column: 6, scope: !3117)
!3124 = !DILocation(line: 64, column: 3, scope: !1162, inlinedAt: !3125)
!3125 = distinct !DILocation(line: 1300, column: 14, scope: !3117)
!3126 = !DILocation(line: 75, column: 3, scope: !1200, inlinedAt: !3127)
!3127 = distinct !DILocation(line: 1300, column: 27, scope: !3117)
!3128 = !DILocation(line: 1300, column: 25, scope: !3117)
!3129 = !DILocation(line: 53, column: 3, scope: !1246, inlinedAt: !3130)
!3130 = distinct !DILocation(line: 1300, column: 40, scope: !3117)
!3131 = !DILocation(line: 1300, column: 38, scope: !3117)
!3132 = !DILocation(line: 1301, column: 5, scope: !3133)
!3133 = distinct !DILexicalBlock(scope: !3117, file: !3, line: 1301, column: 5)
!3134 = !DILocation(line: 1301, column: 11, scope: !3133)
!3135 = !DILocation(line: 1301, column: 5, scope: !3117)
!3136 = !DILocation(line: 1302, column: 3, scope: !3137)
!3137 = distinct !DILexicalBlock(scope: !3133, file: !3, line: 1301, column: 25)
!3138 = !DILocation(line: 1304, column: 22, scope: !3117)
!3139 = !DILocation(line: 1304, column: 25, scope: !3117)
!3140 = !DILocation(line: 1304, column: 32, scope: !3117)
!3141 = !DILocation(line: 1304, column: 2, scope: !3117)
!3142 = !DILocation(line: 1304, column: 8, scope: !3117)
!3143 = !DILocation(line: 1304, column: 15, scope: !3117)
!3144 = !DILocation(line: 1304, column: 20, scope: !3117)
!3145 = !DILocation(line: 1305, column: 22, scope: !3117)
!3146 = !DILocation(line: 1305, column: 25, scope: !3117)
!3147 = !DILocation(line: 1305, column: 32, scope: !3117)
!3148 = !DILocation(line: 1305, column: 2, scope: !3117)
!3149 = !DILocation(line: 1305, column: 8, scope: !3117)
!3150 = !DILocation(line: 1305, column: 15, scope: !3117)
!3151 = !DILocation(line: 1305, column: 20, scope: !3117)
!3152 = !DILocation(line: 1306, column: 1, scope: !3117)
!3153 = distinct !DISubprogram(name: "checksum_gpu_kernel", linkageName: "_Z19checksum_gpu_kerneliP8dcomplexS0_", scope: !3, file: !3, line: 1323, type: !3154, scopeLine: 1325, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !1050, retainedNodes: !1051)
!3154 = !DISubroutineType(types: !3155)
!3155 = !{null, !97, !98, !98}
!3156 = !DILocalVariable(name: "iteration", arg: 1, scope: !3153, file: !3, line: 1323, type: !97)
!3157 = !DILocation(line: 1323, column: 41, scope: !3153)
!3158 = !DILocalVariable(name: "u1", arg: 2, scope: !3153, file: !3, line: 1324, type: !98)
!3159 = !DILocation(line: 1324, column: 12, scope: !3153)
!3160 = !DILocalVariable(name: "sums", arg: 3, scope: !3153, file: !3, line: 1325, type: !98)
!3161 = !DILocation(line: 1325, column: 12, scope: !3153)
!3162 = !DILocalVariable(name: "share_sums", scope: !3153, file: !3, line: 1326, type: !98)
!3163 = !DILocation(line: 1326, column: 12, scope: !3153)
!3164 = !DILocalVariable(name: "j", scope: !3153, file: !3, line: 1327, type: !97)
!3165 = !DILocation(line: 1327, column: 6, scope: !3153)
!3166 = !DILocation(line: 64, column: 3, scope: !1162, inlinedAt: !3167)
!3167 = distinct !DILocation(line: 1327, column: 11, scope: !3153)
!3168 = !DILocation(line: 75, column: 3, scope: !1200, inlinedAt: !3169)
!3169 = distinct !DILocation(line: 1327, column: 24, scope: !3153)
!3170 = !DILocation(line: 1327, column: 22, scope: !3153)
!3171 = !DILocation(line: 53, column: 3, scope: !1246, inlinedAt: !3172)
!3172 = distinct !DILocation(line: 1327, column: 37, scope: !3153)
!3173 = !DILocation(line: 1327, column: 35, scope: !3153)
!3174 = !DILocation(line: 1327, column: 50, scope: !3153)
!3175 = !DILocalVariable(name: "q", scope: !3153, file: !3, line: 1328, type: !97)
!3176 = !DILocation(line: 1328, column: 6, scope: !3153)
!3177 = !DILocalVariable(name: "r", scope: !3153, file: !3, line: 1328, type: !97)
!3178 = !DILocation(line: 1328, column: 9, scope: !3153)
!3179 = !DILocalVariable(name: "s", scope: !3153, file: !3, line: 1328, type: !97)
!3180 = !DILocation(line: 1328, column: 12, scope: !3153)
!3181 = !DILocation(line: 1330, column: 5, scope: !3182)
!3182 = distinct !DILexicalBlock(scope: !3153, file: !3, line: 1330, column: 5)
!3183 = !DILocation(line: 1330, column: 6, scope: !3182)
!3184 = !DILocation(line: 1330, column: 5, scope: !3153)
!3185 = !DILocation(line: 1331, column: 7, scope: !3186)
!3186 = distinct !DILexicalBlock(scope: !3182, file: !3, line: 1330, column: 23)
!3187 = !DILocation(line: 1331, column: 9, scope: !3186)
!3188 = !DILocation(line: 1331, column: 5, scope: !3186)
!3189 = !DILocation(line: 1332, column: 9, scope: !3186)
!3190 = !DILocation(line: 1332, column: 8, scope: !3186)
!3191 = !DILocation(line: 1332, column: 11, scope: !3186)
!3192 = !DILocation(line: 1332, column: 5, scope: !3186)
!3193 = !DILocation(line: 1333, column: 9, scope: !3186)
!3194 = !DILocation(line: 1333, column: 8, scope: !3186)
!3195 = !DILocation(line: 1333, column: 11, scope: !3186)
!3196 = !DILocation(line: 1333, column: 5, scope: !3186)
!3197 = !DILocation(line: 1334, column: 29, scope: !3186)
!3198 = !DILocation(line: 1334, column: 33, scope: !3186)
!3199 = !DILocation(line: 1334, column: 37, scope: !3186)
!3200 = !DILocation(line: 1334, column: 38, scope: !3186)
!3201 = !DILocation(line: 1334, column: 35, scope: !3186)
!3202 = !DILocation(line: 1334, column: 44, scope: !3186)
!3203 = !DILocation(line: 1334, column: 45, scope: !3186)
!3204 = !DILocation(line: 1334, column: 48, scope: !3186)
!3205 = !DILocation(line: 1334, column: 42, scope: !3186)
!3206 = !DILocation(line: 1334, column: 3, scope: !3186)
!3207 = !DILocation(line: 53, column: 3, scope: !1246, inlinedAt: !3208)
!3208 = distinct !DILocation(line: 1334, column: 14, scope: !3186)
!3209 = !DILocation(line: 1334, column: 27, scope: !3186)
!3210 = !DILocation(line: 1335, column: 2, scope: !3186)
!3211 = !DILocation(line: 1336, column: 29, scope: !3212)
!3212 = distinct !DILexicalBlock(scope: !3182, file: !3, line: 1335, column: 7)
!3213 = !DILocation(line: 1336, column: 3, scope: !3212)
!3214 = !DILocation(line: 53, column: 3, scope: !1246, inlinedAt: !3215)
!3215 = distinct !DILocation(line: 1336, column: 14, scope: !3212)
!3216 = !DILocation(line: 1336, column: 27, scope: !3212)
!3217 = !DILocation(line: 1340, column: 2, scope: !3153)
!3218 = !DILocation(line: 53, column: 3, scope: !1246, inlinedAt: !3219)
!3219 = distinct !DILocation(line: 1341, column: 5, scope: !3220)
!3220 = distinct !DILexicalBlock(scope: !3153, file: !3, line: 1341, column: 5)
!3221 = !DILocation(line: 1341, column: 16, scope: !3220)
!3222 = !DILocation(line: 1341, column: 5, scope: !3153)
!3223 = !DILocalVariable(name: "i", scope: !3224, file: !3, line: 1342, type: !97)
!3224 = distinct !DILexicalBlock(scope: !3225, file: !3, line: 1342, column: 3)
!3225 = distinct !DILexicalBlock(scope: !3220, file: !3, line: 1341, column: 20)
!3226 = !DILocation(line: 1342, column: 11, scope: !3224)
!3227 = !DILocation(line: 1342, column: 7, scope: !3224)
!3228 = !DILocation(line: 1342, column: 16, scope: !3229)
!3229 = distinct !DILexicalBlock(scope: !3224, file: !3, line: 1342, column: 3)
!3230 = !DILocation(line: 75, column: 3, scope: !1200, inlinedAt: !3231)
!3231 = distinct !DILocation(line: 1342, column: 18, scope: !3229)
!3232 = !DILocation(line: 1342, column: 17, scope: !3229)
!3233 = !DILocation(line: 1342, column: 3, scope: !3224)
!3234 = !DILocation(line: 1343, column: 20, scope: !3235)
!3235 = distinct !DILexicalBlock(scope: !3229, file: !3, line: 1342, column: 34)
!3236 = !DILocation(line: 1343, column: 4, scope: !3235)
!3237 = !DILocation(line: 1343, column: 18, scope: !3235)
!3238 = !DILocation(line: 1344, column: 3, scope: !3235)
!3239 = !DILocation(line: 1342, column: 30, scope: !3229)
!3240 = !DILocation(line: 1342, column: 3, scope: !3229)
!3241 = distinct !{!3241, !3233, !3242}
!3242 = !DILocation(line: 1344, column: 3, scope: !3224)
!3243 = !DILocation(line: 1345, column: 2, scope: !3225)
!3244 = !DILocation(line: 53, column: 3, scope: !1246, inlinedAt: !3245)
!3245 = distinct !DILocation(line: 1346, column: 5, scope: !3246)
!3246 = distinct !DILexicalBlock(scope: !3153, file: !3, line: 1346, column: 5)
!3247 = !DILocation(line: 1346, column: 16, scope: !3246)
!3248 = !DILocation(line: 1346, column: 5, scope: !3153)
!3249 = !DILocation(line: 1347, column: 24, scope: !3250)
!3250 = distinct !DILexicalBlock(scope: !3246, file: !3, line: 1346, column: 20)
!3251 = !DILocation(line: 1347, column: 38, scope: !3250)
!3252 = !DILocation(line: 1347, column: 42, scope: !3250)
!3253 = !DILocation(line: 1347, column: 3, scope: !3250)
!3254 = !DILocation(line: 1347, column: 17, scope: !3250)
!3255 = !DILocation(line: 1347, column: 22, scope: !3250)
!3256 = !DILocation(line: 1348, column: 14, scope: !3250)
!3257 = !DILocation(line: 1348, column: 19, scope: !3250)
!3258 = !DILocation(line: 1348, column: 30, scope: !3250)
!3259 = !DILocation(line: 1348, column: 35, scope: !3250)
!3260 = !DILocation(line: 1348, column: 49, scope: !3250)
!3261 = !DILocation(line: 1348, column: 3, scope: !3250)
!3262 = !DILocation(line: 1349, column: 24, scope: !3250)
!3263 = !DILocation(line: 1349, column: 38, scope: !3250)
!3264 = !DILocation(line: 1349, column: 42, scope: !3250)
!3265 = !DILocation(line: 1349, column: 3, scope: !3250)
!3266 = !DILocation(line: 1349, column: 17, scope: !3250)
!3267 = !DILocation(line: 1349, column: 22, scope: !3250)
!3268 = !DILocation(line: 1350, column: 14, scope: !3250)
!3269 = !DILocation(line: 1350, column: 19, scope: !3250)
!3270 = !DILocation(line: 1350, column: 30, scope: !3250)
!3271 = !DILocation(line: 1350, column: 35, scope: !3250)
!3272 = !DILocation(line: 1350, column: 49, scope: !3250)
!3273 = !DILocation(line: 1350, column: 3, scope: !3250)
!3274 = !DILocation(line: 1351, column: 2, scope: !3250)
!3275 = !DILocation(line: 1352, column: 1, scope: !3153)
!3276 = distinct !DISubprogram(name: "atomicAdd", linkageName: "_ZL9atomicAddPdd", scope: !100, file: !100, line: 54, type: !3277, scopeLine: 54, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !1050, retainedNodes: !1051)
!3277 = !DISubroutineType(types: !3278)
!3278 = !{!104, !106, !104}
!3279 = !DILocalVariable(name: "x", arg: 1, scope: !3280, file: !780, line: 1370, type: !104)
!3280 = distinct !DISubprogram(name: "__double_as_longlong", linkageName: "_ZL20__double_as_longlongd", scope: !780, file: !780, line: 1370, type: !3281, scopeLine: 1371, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !1050, retainedNodes: !1051)
!3281 = !DISubroutineType(types: !3282)
!3282 = !{!413, !104}
!3283 = !DILocation(line: 1370, column: 74, scope: !3280, inlinedAt: !3284)
!3284 = distinct !DILocation(line: 61, column: 44, scope: !3285)
!3285 = distinct !DILexicalBlock(scope: !3286, file: !100, line: 59, column: 35)
!3286 = distinct !DILexicalBlock(scope: !3287, file: !100, line: 59, column: 2)
!3287 = distinct !DILexicalBlock(scope: !3276, file: !100, line: 59, column: 2)
!3288 = !DILocalVariable(name: "x", arg: 1, scope: !3289, file: !780, line: 1365, type: !413)
!3289 = distinct !DISubprogram(name: "__longlong_as_double", linkageName: "_ZL20__longlong_as_doublex", scope: !780, file: !780, line: 1365, type: !3290, scopeLine: 1366, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !1050, retainedNodes: !1051)
!3290 = !DISubroutineType(types: !3291)
!3291 = !{!104, !413}
!3292 = !DILocation(line: 1365, column: 72, scope: !3289, inlinedAt: !3293)
!3293 = distinct !DILocation(line: 61, column: 70, scope: !3285)
!3294 = !DILocation(line: 1365, column: 72, scope: !3289, inlinedAt: !3295)
!3295 = distinct !DILocation(line: 62, column: 36, scope: !3296)
!3296 = distinct !DILexicalBlock(scope: !3285, file: !100, line: 62, column: 13)
!3297 = !DILocation(line: 1365, column: 72, scope: !3289, inlinedAt: !3298)
!3298 = distinct !DILocation(line: 64, column: 9, scope: !3276)
!3299 = !DILocation(line: 1365, column: 72, scope: !3289, inlinedAt: !3300)
!3300 = distinct !DILocation(line: 58, column: 10, scope: !3301)
!3301 = distinct !DILexicalBlock(scope: !3276, file: !100, line: 57, column: 6)
!3302 = !DILocalVariable(name: "address", arg: 1, scope: !3276, file: !100, line: 54, type: !106)
!3303 = !DILocation(line: 54, column: 55, scope: !3276)
!3304 = !DILocalVariable(name: "val", arg: 2, scope: !3276, file: !100, line: 54, type: !104)
!3305 = !DILocation(line: 54, column: 71, scope: !3276)
!3306 = !DILocalVariable(name: "address_as_ull", scope: !3276, file: !100, line: 55, type: !1053)
!3307 = !DILocation(line: 55, column: 26, scope: !3276)
!3308 = !DILocation(line: 55, column: 68, scope: !3276)
!3309 = !DILocation(line: 55, column: 43, scope: !3276)
!3310 = !DILocalVariable(name: "old", scope: !3276, file: !100, line: 56, type: !705)
!3311 = !DILocation(line: 56, column: 25, scope: !3276)
!3312 = !DILocation(line: 56, column: 32, scope: !3276)
!3313 = !DILocation(line: 56, column: 31, scope: !3276)
!3314 = !DILocalVariable(name: "assumed", scope: !3276, file: !100, line: 56, type: !705)
!3315 = !DILocation(line: 56, column: 48, scope: !3276)
!3316 = !DILocation(line: 57, column: 6, scope: !3301)
!3317 = !DILocation(line: 57, column: 9, scope: !3301)
!3318 = !DILocation(line: 57, column: 6, scope: !3276)
!3319 = !DILocation(line: 58, column: 31, scope: !3301)
!3320 = !DILocation(line: 1367, column: 34, scope: !3289, inlinedAt: !3300)
!3321 = !DILocation(line: 1367, column: 10, scope: !3289, inlinedAt: !3300)
!3322 = !DILocation(line: 58, column: 3, scope: !3301)
!3323 = !DILocalVariable(name: "i", scope: !3287, file: !100, line: 59, type: !97)
!3324 = !DILocation(line: 59, column: 11, scope: !3287)
!3325 = !DILocation(line: 59, column: 7, scope: !3287)
!3326 = !DILocation(line: 59, column: 18, scope: !3286)
!3327 = !DILocation(line: 59, column: 20, scope: !3286)
!3328 = !DILocation(line: 59, column: 2, scope: !3287)
!3329 = !DILocation(line: 60, column: 13, scope: !3285)
!3330 = !DILocation(line: 60, column: 11, scope: !3285)
!3331 = !DILocation(line: 61, column: 19, scope: !3285)
!3332 = !DILocation(line: 61, column: 35, scope: !3285)
!3333 = !DILocation(line: 61, column: 65, scope: !3285)
!3334 = !DILocation(line: 61, column: 91, scope: !3285)
!3335 = !DILocation(line: 1367, column: 34, scope: !3289, inlinedAt: !3293)
!3336 = !DILocation(line: 1367, column: 10, scope: !3289, inlinedAt: !3293)
!3337 = !DILocation(line: 61, column: 69, scope: !3285)
!3338 = !DILocation(line: 1372, column: 34, scope: !3280, inlinedAt: !3284)
!3339 = !DILocation(line: 1372, column: 10, scope: !3280, inlinedAt: !3284)
!3340 = !DILocation(line: 61, column: 9, scope: !3285)
!3341 = !DILocation(line: 61, column: 7, scope: !3285)
!3342 = !DILocation(line: 62, column: 13, scope: !3296)
!3343 = !DILocation(line: 62, column: 24, scope: !3296)
!3344 = !DILocation(line: 62, column: 21, scope: !3296)
!3345 = !DILocation(line: 62, column: 13, scope: !3285)
!3346 = !DILocation(line: 62, column: 57, scope: !3296)
!3347 = !DILocation(line: 1367, column: 34, scope: !3289, inlinedAt: !3295)
!3348 = !DILocation(line: 1367, column: 10, scope: !3289, inlinedAt: !3295)
!3349 = !DILocation(line: 62, column: 29, scope: !3296)
!3350 = !DILocation(line: 63, column: 2, scope: !3285)
!3351 = !DILocation(line: 59, column: 31, scope: !3286)
!3352 = !DILocation(line: 59, column: 2, scope: !3286)
!3353 = distinct !{!3353, !3328, !3354}
!3354 = !DILocation(line: 63, column: 2, scope: !3287)
!3355 = !DILocation(line: 64, column: 30, scope: !3276)
!3356 = !DILocation(line: 1367, column: 34, scope: !3289, inlinedAt: !3298)
!3357 = !DILocation(line: 1367, column: 10, scope: !3289, inlinedAt: !3298)
!3358 = !DILocation(line: 64, column: 2, scope: !3276)
!3359 = !DILocation(line: 65, column: 1, scope: !3276)
!3360 = distinct !DISubprogram(name: "atomicCAS", linkageName: "_ZL9atomicCASPyyy", scope: !3361, file: !3361, line: 211, type: !3362, scopeLine: 212, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !1050, retainedNodes: !1051)
!3361 = !DIFile(filename: "/usr/local/cuda/include/device_atomic_functions.hpp", directory: "")
!3362 = !DISubroutineType(types: !3363)
!3363 = !{!705, !1053, !705, !705}
!3364 = !DILocalVariable(name: "p", arg: 1, scope: !3365, file: !780, line: 1655, type: !1053)
!3365 = distinct !DISubprogram(name: "__ullAtomicCAS", linkageName: "_ZL14__ullAtomicCASPyyy", scope: !780, file: !780, line: 1655, type: !3362, scopeLine: 1658, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !1050, retainedNodes: !1051)
!3366 = !DILocation(line: 1655, column: 63, scope: !3365, inlinedAt: !3367)
!3367 = distinct !DILocation(line: 213, column: 10, scope: !3360)
!3368 = !DILocalVariable(name: "compare", arg: 2, scope: !3365, file: !780, line: 1656, type: !705)
!3369 = !DILocation(line: 1656, column: 62, scope: !3365, inlinedAt: !3367)
!3370 = !DILocalVariable(name: "val", arg: 3, scope: !3365, file: !780, line: 1657, type: !705)
!3371 = !DILocation(line: 1657, column: 62, scope: !3365, inlinedAt: !3367)
!3372 = !DILocalVariable(name: "address", arg: 1, scope: !3360, file: !3361, line: 211, type: !1053)
!3373 = !DILocation(line: 211, column: 91, scope: !3360)
!3374 = !DILocalVariable(name: "compare", arg: 2, scope: !3360, file: !3361, line: 211, type: !705)
!3375 = !DILocation(line: 211, column: 123, scope: !3360)
!3376 = !DILocalVariable(name: "val", arg: 3, scope: !3360, file: !3361, line: 211, type: !705)
!3377 = !DILocation(line: 211, column: 155, scope: !3360)
!3378 = !DILocation(line: 213, column: 25, scope: !3360)
!3379 = !DILocation(line: 213, column: 34, scope: !3360)
!3380 = !DILocation(line: 213, column: 43, scope: !3360)
!3381 = !DILocation(line: 1660, column: 78, scope: !3365, inlinedAt: !3367)
!3382 = !DILocation(line: 1661, column: 67, scope: !3365, inlinedAt: !3367)
!3383 = !DILocation(line: 1662, column: 67, scope: !3365, inlinedAt: !3367)
!3384 = !DILocation(line: 1660, column: 29, scope: !3365, inlinedAt: !3367)
!3385 = !DILocation(line: 213, column: 3, scope: !3360)
!3386 = distinct !DISubprogram(name: "compute_indexmap_gpu_kernel", linkageName: "_Z27compute_indexmap_gpu_kernelPd", scope: !3, file: !3, line: 1365, type: !3387, scopeLine: 1365, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !1050, retainedNodes: !1051)
!3387 = !DISubroutineType(types: !3388)
!3388 = !{null, !106}
!3389 = !DILocalVariable(name: "a", arg: 1, scope: !3390, file: !3391, line: 245, type: !104)
!3390 = distinct !DISubprogram(name: "exp", linkageName: "_ZL3expd", scope: !3391, file: !3391, line: 245, type: !496, scopeLine: 246, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !1050, retainedNodes: !1051)
!3391 = !DIFile(filename: "/usr/local/cuda/include/math_functions_dbl_ptx3.hpp", directory: "")
!3392 = !DILocation(line: 245, column: 52, scope: !3390, inlinedAt: !3393)
!3393 = distinct !DILocation(line: 1384, column: 23, scope: !3386)
!3394 = !DILocalVariable(name: "twiddle", arg: 1, scope: !3386, file: !3, line: 1365, type: !106)
!3395 = !DILocation(line: 1365, column: 52, scope: !3386)
!3396 = !DILocalVariable(name: "thread_id", scope: !3386, file: !3, line: 1366, type: !97)
!3397 = !DILocation(line: 1366, column: 6, scope: !3386)
!3398 = !DILocation(line: 64, column: 3, scope: !1162, inlinedAt: !3399)
!3399 = distinct !DILocation(line: 1366, column: 18, scope: !3386)
!3400 = !DILocation(line: 75, column: 3, scope: !1200, inlinedAt: !3401)
!3401 = distinct !DILocation(line: 1366, column: 31, scope: !3386)
!3402 = !DILocation(line: 1366, column: 29, scope: !3386)
!3403 = !DILocation(line: 53, column: 3, scope: !1246, inlinedAt: !3404)
!3404 = distinct !DILocation(line: 1366, column: 44, scope: !3386)
!3405 = !DILocation(line: 1366, column: 42, scope: !3386)
!3406 = !DILocation(line: 1368, column: 5, scope: !3407)
!3407 = distinct !DILexicalBlock(scope: !3386, file: !3, line: 1368, column: 5)
!3408 = !DILocation(line: 1368, column: 14, scope: !3407)
!3409 = !DILocation(line: 1368, column: 5, scope: !3386)
!3410 = !DILocation(line: 1369, column: 3, scope: !3411)
!3411 = distinct !DILexicalBlock(scope: !3407, file: !3, line: 1368, column: 23)
!3412 = !DILocalVariable(name: "i", scope: !3386, file: !3, line: 1372, type: !97)
!3413 = !DILocation(line: 1372, column: 6, scope: !3386)
!3414 = !DILocation(line: 1372, column: 10, scope: !3386)
!3415 = !DILocation(line: 1372, column: 20, scope: !3386)
!3416 = !DILocalVariable(name: "j", scope: !3386, file: !3, line: 1373, type: !97)
!3417 = !DILocation(line: 1373, column: 6, scope: !3386)
!3418 = !DILocation(line: 1373, column: 11, scope: !3386)
!3419 = !DILocation(line: 1373, column: 21, scope: !3386)
!3420 = !DILocation(line: 1373, column: 27, scope: !3386)
!3421 = !DILocalVariable(name: "k", scope: !3386, file: !3, line: 1374, type: !97)
!3422 = !DILocation(line: 1374, column: 6, scope: !3386)
!3423 = !DILocation(line: 1374, column: 10, scope: !3386)
!3424 = !DILocation(line: 1374, column: 20, scope: !3386)
!3425 = !DILocalVariable(name: "kk", scope: !3386, file: !3, line: 1376, type: !97)
!3426 = !DILocation(line: 1376, column: 6, scope: !3386)
!3427 = !DILocalVariable(name: "kk2", scope: !3386, file: !3, line: 1376, type: !97)
!3428 = !DILocation(line: 1376, column: 10, scope: !3386)
!3429 = !DILocalVariable(name: "jj", scope: !3386, file: !3, line: 1376, type: !97)
!3430 = !DILocation(line: 1376, column: 15, scope: !3386)
!3431 = !DILocalVariable(name: "kj2", scope: !3386, file: !3, line: 1376, type: !97)
!3432 = !DILocation(line: 1376, column: 19, scope: !3386)
!3433 = !DILocalVariable(name: "ii", scope: !3386, file: !3, line: 1376, type: !97)
!3434 = !DILocation(line: 1376, column: 24, scope: !3386)
!3435 = !DILocation(line: 1378, column: 9, scope: !3386)
!3436 = !DILocation(line: 1378, column: 10, scope: !3386)
!3437 = !DILocation(line: 1378, column: 17, scope: !3386)
!3438 = !DILocation(line: 1378, column: 23, scope: !3386)
!3439 = !DILocation(line: 1378, column: 5, scope: !3386)
!3440 = !DILocation(line: 1379, column: 8, scope: !3386)
!3441 = !DILocation(line: 1379, column: 11, scope: !3386)
!3442 = !DILocation(line: 1379, column: 10, scope: !3386)
!3443 = !DILocation(line: 1379, column: 6, scope: !3386)
!3444 = !DILocation(line: 1380, column: 9, scope: !3386)
!3445 = !DILocation(line: 1380, column: 10, scope: !3386)
!3446 = !DILocation(line: 1380, column: 17, scope: !3386)
!3447 = !DILocation(line: 1380, column: 23, scope: !3386)
!3448 = !DILocation(line: 1380, column: 5, scope: !3386)
!3449 = !DILocation(line: 1381, column: 8, scope: !3386)
!3450 = !DILocation(line: 1381, column: 11, scope: !3386)
!3451 = !DILocation(line: 1381, column: 10, scope: !3386)
!3452 = !DILocation(line: 1381, column: 14, scope: !3386)
!3453 = !DILocation(line: 1381, column: 13, scope: !3386)
!3454 = !DILocation(line: 1381, column: 6, scope: !3386)
!3455 = !DILocation(line: 1382, column: 9, scope: !3386)
!3456 = !DILocation(line: 1382, column: 10, scope: !3386)
!3457 = !DILocation(line: 1382, column: 17, scope: !3386)
!3458 = !DILocation(line: 1382, column: 23, scope: !3386)
!3459 = !DILocation(line: 1382, column: 5, scope: !3386)
!3460 = !DILocation(line: 1384, column: 39, scope: !3386)
!3461 = !DILocation(line: 1384, column: 42, scope: !3386)
!3462 = !DILocation(line: 1384, column: 41, scope: !3386)
!3463 = !DILocation(line: 1384, column: 45, scope: !3386)
!3464 = !DILocation(line: 1384, column: 44, scope: !3386)
!3465 = !DILocation(line: 1384, column: 38, scope: !3386)
!3466 = !DILocation(line: 1384, column: 29, scope: !3386)
!3467 = !DILocation(line: 247, column: 19, scope: !3390, inlinedAt: !3393)
!3468 = !DILocation(line: 247, column: 10, scope: !3390, inlinedAt: !3393)
!3469 = !DILocation(line: 1384, column: 2, scope: !3386)
!3470 = !DILocation(line: 1384, column: 10, scope: !3386)
!3471 = !DILocation(line: 1384, column: 21, scope: !3386)
!3472 = !DILocation(line: 1385, column: 1, scope: !3386)
!3473 = distinct !DISubprogram(name: "compute_initial_conditions_gpu_kernel", linkageName: "_Z37compute_initial_conditions_gpu_kernelP8dcomplexPd", scope: !3, file: !3, line: 1416, type: !3474, scopeLine: 1417, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !1050, retainedNodes: !1051)
!3474 = !DISubroutineType(types: !3475)
!3475 = !{null, !98, !106}
!3476 = !DILocalVariable(name: "u0", arg: 1, scope: !3473, file: !3, line: 1416, type: !98)
!3477 = !DILocation(line: 1416, column: 64, scope: !3473)
!3478 = !DILocalVariable(name: "starts", arg: 2, scope: !3473, file: !3, line: 1417, type: !106)
!3479 = !DILocation(line: 1417, column: 10, scope: !3473)
!3480 = !DILocalVariable(name: "z", scope: !3473, file: !3, line: 1418, type: !97)
!3481 = !DILocation(line: 1418, column: 6, scope: !3473)
!3482 = !DILocation(line: 64, column: 3, scope: !1162, inlinedAt: !3483)
!3483 = distinct !DILocation(line: 1418, column: 10, scope: !3473)
!3484 = !DILocation(line: 75, column: 3, scope: !1200, inlinedAt: !3485)
!3485 = distinct !DILocation(line: 1418, column: 23, scope: !3473)
!3486 = !DILocation(line: 1418, column: 21, scope: !3473)
!3487 = !DILocation(line: 53, column: 3, scope: !1246, inlinedAt: !3488)
!3488 = distinct !DILocation(line: 1418, column: 36, scope: !3473)
!3489 = !DILocation(line: 1418, column: 34, scope: !3473)
!3490 = !DILocation(line: 1420, column: 5, scope: !3491)
!3491 = distinct !DILexicalBlock(scope: !3473, file: !3, line: 1420, column: 5)
!3492 = !DILocation(line: 1420, column: 6, scope: !3491)
!3493 = !DILocation(line: 1420, column: 5, scope: !3473)
!3494 = !DILocation(line: 1420, column: 12, scope: !3495)
!3495 = distinct !DILexicalBlock(scope: !3491, file: !3, line: 1420, column: 11)
!3496 = !DILocalVariable(name: "x0", scope: !3473, file: !3, line: 1422, type: !104)
!3497 = !DILocation(line: 1422, column: 9, scope: !3473)
!3498 = !DILocation(line: 1422, column: 14, scope: !3473)
!3499 = !DILocation(line: 1422, column: 21, scope: !3473)
!3500 = !DILocalVariable(name: "y", scope: !3501, file: !3, line: 1423, type: !97)
!3501 = distinct !DILexicalBlock(scope: !3473, file: !3, line: 1423, column: 2)
!3502 = !DILocation(line: 1423, column: 10, scope: !3501)
!3503 = !DILocation(line: 1423, column: 6, scope: !3501)
!3504 = !DILocation(line: 1423, column: 15, scope: !3505)
!3505 = distinct !DILexicalBlock(scope: !3501, file: !3, line: 1423, column: 2)
!3506 = !DILocation(line: 1423, column: 16, scope: !3505)
!3507 = !DILocation(line: 1423, column: 2, scope: !3501)
!3508 = !DILocation(line: 1424, column: 41, scope: !3509)
!3509 = distinct !DILexicalBlock(scope: !3505, file: !3, line: 1423, column: 25)
!3510 = !DILocation(line: 1424, column: 49, scope: !3509)
!3511 = !DILocation(line: 1424, column: 50, scope: !3509)
!3512 = !DILocation(line: 1424, column: 47, scope: !3509)
!3513 = !DILocation(line: 1424, column: 56, scope: !3509)
!3514 = !DILocation(line: 1424, column: 57, scope: !3509)
!3515 = !DILocation(line: 1424, column: 60, scope: !3509)
!3516 = !DILocation(line: 1424, column: 54, scope: !3509)
!3517 = !DILocation(line: 1424, column: 31, scope: !3509)
!3518 = !DILocation(line: 1424, column: 3, scope: !3509)
!3519 = !DILocation(line: 1425, column: 2, scope: !3509)
!3520 = !DILocation(line: 1423, column: 22, scope: !3505)
!3521 = !DILocation(line: 1423, column: 2, scope: !3505)
!3522 = distinct !{!3522, !3507, !3523}
!3523 = !DILocation(line: 1425, column: 2, scope: !3501)
!3524 = !DILocation(line: 1426, column: 1, scope: !3473)
!3525 = distinct !DISubprogram(name: "vranlc_device", linkageName: "_Z13vranlc_deviceiPddS_", scope: !3, file: !3, line: 2034, type: !3526, scopeLine: 2037, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !1050, retainedNodes: !1051)
!3526 = !DISubroutineType(types: !3527)
!3527 = !{null, !97, !106, !104, !106}
!3528 = !DILocalVariable(name: "n", arg: 1, scope: !3525, file: !3, line: 2034, type: !97)
!3529 = !DILocation(line: 2034, column: 35, scope: !3525)
!3530 = !DILocalVariable(name: "x_seed", arg: 2, scope: !3525, file: !3, line: 2035, type: !106)
!3531 = !DILocation(line: 2035, column: 11, scope: !3525)
!3532 = !DILocalVariable(name: "a", arg: 3, scope: !3525, file: !3, line: 2036, type: !104)
!3533 = !DILocation(line: 2036, column: 10, scope: !3525)
!3534 = !DILocalVariable(name: "y", arg: 4, scope: !3525, file: !3, line: 2037, type: !106)
!3535 = !DILocation(line: 2037, column: 10, scope: !3525)
!3536 = !DILocalVariable(name: "i", scope: !3525, file: !3, line: 2038, type: !97)
!3537 = !DILocation(line: 2038, column: 6, scope: !3525)
!3538 = !DILocalVariable(name: "x", scope: !3525, file: !3, line: 2039, type: !104)
!3539 = !DILocation(line: 2039, column: 9, scope: !3525)
!3540 = !DILocalVariable(name: "t1", scope: !3525, file: !3, line: 2039, type: !104)
!3541 = !DILocation(line: 2039, column: 11, scope: !3525)
!3542 = !DILocalVariable(name: "t2", scope: !3525, file: !3, line: 2039, type: !104)
!3543 = !DILocation(line: 2039, column: 14, scope: !3525)
!3544 = !DILocalVariable(name: "t3", scope: !3525, file: !3, line: 2039, type: !104)
!3545 = !DILocation(line: 2039, column: 17, scope: !3525)
!3546 = !DILocalVariable(name: "t4", scope: !3525, file: !3, line: 2039, type: !104)
!3547 = !DILocation(line: 2039, column: 20, scope: !3525)
!3548 = !DILocalVariable(name: "a1", scope: !3525, file: !3, line: 2039, type: !104)
!3549 = !DILocation(line: 2039, column: 23, scope: !3525)
!3550 = !DILocalVariable(name: "a2", scope: !3525, file: !3, line: 2039, type: !104)
!3551 = !DILocation(line: 2039, column: 26, scope: !3525)
!3552 = !DILocalVariable(name: "x1", scope: !3525, file: !3, line: 2039, type: !104)
!3553 = !DILocation(line: 2039, column: 29, scope: !3525)
!3554 = !DILocalVariable(name: "x2", scope: !3525, file: !3, line: 2039, type: !104)
!3555 = !DILocation(line: 2039, column: 32, scope: !3525)
!3556 = !DILocalVariable(name: "z", scope: !3525, file: !3, line: 2039, type: !104)
!3557 = !DILocation(line: 2039, column: 35, scope: !3525)
!3558 = !DILocation(line: 2040, column: 13, scope: !3525)
!3559 = !DILocation(line: 2040, column: 11, scope: !3525)
!3560 = !DILocation(line: 2040, column: 5, scope: !3525)
!3561 = !DILocation(line: 2041, column: 12, scope: !3525)
!3562 = !DILocation(line: 2041, column: 7, scope: !3525)
!3563 = !DILocation(line: 2041, column: 5, scope: !3525)
!3564 = !DILocation(line: 2042, column: 7, scope: !3525)
!3565 = !DILocation(line: 2042, column: 17, scope: !3525)
!3566 = !DILocation(line: 2042, column: 15, scope: !3525)
!3567 = !DILocation(line: 2042, column: 9, scope: !3525)
!3568 = !DILocation(line: 2042, column: 5, scope: !3525)
!3569 = !DILocation(line: 2043, column: 7, scope: !3525)
!3570 = !DILocation(line: 2043, column: 6, scope: !3525)
!3571 = !DILocation(line: 2043, column: 4, scope: !3525)
!3572 = !DILocation(line: 2044, column: 7, scope: !3573)
!3573 = distinct !DILexicalBlock(scope: !3525, file: !3, line: 2044, column: 2)
!3574 = !DILocation(line: 2044, column: 6, scope: !3573)
!3575 = !DILocation(line: 2044, column: 11, scope: !3576)
!3576 = distinct !DILexicalBlock(scope: !3573, file: !3, line: 2044, column: 2)
!3577 = !DILocation(line: 2044, column: 13, scope: !3576)
!3578 = !DILocation(line: 2044, column: 12, scope: !3576)
!3579 = !DILocation(line: 2044, column: 2, scope: !3573)
!3580 = !DILocation(line: 2045, column: 14, scope: !3581)
!3581 = distinct !DILexicalBlock(scope: !3576, file: !3, line: 2044, column: 20)
!3582 = !DILocation(line: 2045, column: 12, scope: !3581)
!3583 = !DILocation(line: 2045, column: 6, scope: !3581)
!3584 = !DILocation(line: 2046, column: 13, scope: !3581)
!3585 = !DILocation(line: 2046, column: 8, scope: !3581)
!3586 = !DILocation(line: 2046, column: 6, scope: !3581)
!3587 = !DILocation(line: 2047, column: 8, scope: !3581)
!3588 = !DILocation(line: 2047, column: 18, scope: !3581)
!3589 = !DILocation(line: 2047, column: 16, scope: !3581)
!3590 = !DILocation(line: 2047, column: 10, scope: !3581)
!3591 = !DILocation(line: 2047, column: 6, scope: !3581)
!3592 = !DILocation(line: 2048, column: 8, scope: !3581)
!3593 = !DILocation(line: 2048, column: 13, scope: !3581)
!3594 = !DILocation(line: 2048, column: 11, scope: !3581)
!3595 = !DILocation(line: 2048, column: 18, scope: !3581)
!3596 = !DILocation(line: 2048, column: 23, scope: !3581)
!3597 = !DILocation(line: 2048, column: 21, scope: !3581)
!3598 = !DILocation(line: 2048, column: 16, scope: !3581)
!3599 = !DILocation(line: 2048, column: 6, scope: !3581)
!3600 = !DILocation(line: 2049, column: 20, scope: !3581)
!3601 = !DILocation(line: 2049, column: 18, scope: !3581)
!3602 = !DILocation(line: 2049, column: 13, scope: !3581)
!3603 = !DILocation(line: 2049, column: 8, scope: !3581)
!3604 = !DILocation(line: 2049, column: 6, scope: !3581)
!3605 = !DILocation(line: 2050, column: 7, scope: !3581)
!3606 = !DILocation(line: 2050, column: 18, scope: !3581)
!3607 = !DILocation(line: 2050, column: 16, scope: !3581)
!3608 = !DILocation(line: 2050, column: 10, scope: !3581)
!3609 = !DILocation(line: 2050, column: 5, scope: !3581)
!3610 = !DILocation(line: 2051, column: 14, scope: !3581)
!3611 = !DILocation(line: 2051, column: 12, scope: !3581)
!3612 = !DILocation(line: 2051, column: 18, scope: !3581)
!3613 = !DILocation(line: 2051, column: 23, scope: !3581)
!3614 = !DILocation(line: 2051, column: 21, scope: !3581)
!3615 = !DILocation(line: 2051, column: 16, scope: !3581)
!3616 = !DILocation(line: 2051, column: 6, scope: !3581)
!3617 = !DILocation(line: 2052, column: 20, scope: !3581)
!3618 = !DILocation(line: 2052, column: 18, scope: !3581)
!3619 = !DILocation(line: 2052, column: 13, scope: !3581)
!3620 = !DILocation(line: 2052, column: 8, scope: !3581)
!3621 = !DILocation(line: 2052, column: 6, scope: !3581)
!3622 = !DILocation(line: 2053, column: 7, scope: !3581)
!3623 = !DILocation(line: 2053, column: 18, scope: !3581)
!3624 = !DILocation(line: 2053, column: 16, scope: !3581)
!3625 = !DILocation(line: 2053, column: 10, scope: !3581)
!3626 = !DILocation(line: 2053, column: 5, scope: !3581)
!3627 = !DILocation(line: 2054, column: 16, scope: !3581)
!3628 = !DILocation(line: 2054, column: 14, scope: !3581)
!3629 = !DILocation(line: 2054, column: 3, scope: !3581)
!3630 = !DILocation(line: 2054, column: 5, scope: !3581)
!3631 = !DILocation(line: 2054, column: 8, scope: !3581)
!3632 = !DILocation(line: 2055, column: 2, scope: !3581)
!3633 = !DILocation(line: 2044, column: 17, scope: !3576)
!3634 = !DILocation(line: 2044, column: 2, scope: !3576)
!3635 = distinct !{!3635, !3579, !3636}
!3636 = !DILocation(line: 2055, column: 2, scope: !3573)
!3637 = !DILocation(line: 2056, column: 12, scope: !3525)
!3638 = !DILocation(line: 2056, column: 3, scope: !3525)
!3639 = !DILocation(line: 2056, column: 10, scope: !3525)
!3640 = !DILocation(line: 2057, column: 1, scope: !3525)
!3641 = distinct !DISubprogram(name: "evolve_gpu_kernel", linkageName: "_Z17evolve_gpu_kernelP8dcomplexS0_Pd", scope: !3, file: !3, line: 1444, type: !3642, scopeLine: 1446, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !1050, retainedNodes: !1051)
!3642 = !DISubroutineType(types: !3643)
!3643 = !{null, !98, !98, !106}
!3644 = !DILocalVariable(name: "u0", arg: 1, scope: !3641, file: !3, line: 1444, type: !98)
!3645 = !DILocation(line: 1444, column: 44, scope: !3641)
!3646 = !DILocalVariable(name: "u1", arg: 2, scope: !3641, file: !3, line: 1445, type: !98)
!3647 = !DILocation(line: 1445, column: 12, scope: !3641)
!3648 = !DILocalVariable(name: "twiddle", arg: 3, scope: !3641, file: !3, line: 1446, type: !106)
!3649 = !DILocation(line: 1446, column: 10, scope: !3641)
!3650 = !DILocalVariable(name: "thread_id", scope: !3641, file: !3, line: 1447, type: !97)
!3651 = !DILocation(line: 1447, column: 6, scope: !3641)
!3652 = !DILocation(line: 64, column: 3, scope: !1162, inlinedAt: !3653)
!3653 = distinct !DILocation(line: 1447, column: 18, scope: !3641)
!3654 = !DILocation(line: 75, column: 3, scope: !1200, inlinedAt: !3655)
!3655 = distinct !DILocation(line: 1447, column: 31, scope: !3641)
!3656 = !DILocation(line: 1447, column: 29, scope: !3641)
!3657 = !DILocation(line: 53, column: 3, scope: !1246, inlinedAt: !3658)
!3658 = distinct !DILocation(line: 1447, column: 44, scope: !3641)
!3659 = !DILocation(line: 1447, column: 42, scope: !3641)
!3660 = !DILocation(line: 1449, column: 5, scope: !3661)
!3661 = distinct !DILexicalBlock(scope: !3641, file: !3, line: 1449, column: 5)
!3662 = !DILocation(line: 1449, column: 14, scope: !3661)
!3663 = !DILocation(line: 1449, column: 5, scope: !3641)
!3664 = !DILocation(line: 1450, column: 3, scope: !3665)
!3665 = distinct !DILexicalBlock(scope: !3661, file: !3, line: 1449, column: 27)
!3666 = !DILocation(line: 1453, column: 18, scope: !3641)
!3667 = !DILocation(line: 1453, column: 2, scope: !3641)
!3668 = !DILocation(line: 1453, column: 5, scope: !3641)
!3669 = !DILocation(line: 1453, column: 16, scope: !3641)
!3670 = !DILocation(line: 1454, column: 18, scope: !3641)
!3671 = !DILocation(line: 1454, column: 21, scope: !3641)
!3672 = !DILocation(line: 1454, column: 2, scope: !3641)
!3673 = !DILocation(line: 1454, column: 5, scope: !3641)
!3674 = !DILocation(line: 1454, column: 16, scope: !3641)
!3675 = !DILocation(line: 1455, column: 1, scope: !3641)
!3676 = distinct !DISubprogram(name: "init_ui_gpu_kernel", linkageName: "_Z18init_ui_gpu_kernelP8dcomplexS0_Pd", scope: !3, file: !3, line: 1554, type: !3642, scopeLine: 1556, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !1050, retainedNodes: !1051)
!3677 = !DILocalVariable(name: "u0", arg: 1, scope: !3676, file: !3, line: 1554, type: !98)
!3678 = !DILocation(line: 1554, column: 45, scope: !3676)
!3679 = !DILocalVariable(name: "u1", arg: 2, scope: !3676, file: !3, line: 1555, type: !98)
!3680 = !DILocation(line: 1555, column: 12, scope: !3676)
!3681 = !DILocalVariable(name: "twiddle", arg: 3, scope: !3676, file: !3, line: 1556, type: !106)
!3682 = !DILocation(line: 1556, column: 10, scope: !3676)
!3683 = !DILocalVariable(name: "thread_id", scope: !3676, file: !3, line: 1557, type: !97)
!3684 = !DILocation(line: 1557, column: 6, scope: !3676)
!3685 = !DILocation(line: 64, column: 3, scope: !1162, inlinedAt: !3686)
!3686 = distinct !DILocation(line: 1557, column: 18, scope: !3676)
!3687 = !DILocation(line: 75, column: 3, scope: !1200, inlinedAt: !3688)
!3688 = distinct !DILocation(line: 1557, column: 31, scope: !3676)
!3689 = !DILocation(line: 1557, column: 29, scope: !3676)
!3690 = !DILocation(line: 53, column: 3, scope: !1246, inlinedAt: !3691)
!3691 = distinct !DILocation(line: 1557, column: 44, scope: !3676)
!3692 = !DILocation(line: 1557, column: 42, scope: !3676)
!3693 = !DILocation(line: 1559, column: 5, scope: !3694)
!3694 = distinct !DILexicalBlock(scope: !3676, file: !3, line: 1559, column: 5)
!3695 = !DILocation(line: 1559, column: 14, scope: !3694)
!3696 = !DILocation(line: 1559, column: 5, scope: !3676)
!3697 = !DILocation(line: 1560, column: 3, scope: !3698)
!3698 = distinct !DILexicalBlock(scope: !3694, file: !3, line: 1559, column: 23)
!3699 = !DILocation(line: 1563, column: 18, scope: !3676)
!3700 = !DILocation(line: 1563, column: 2, scope: !3676)
!3701 = !DILocation(line: 1563, column: 5, scope: !3676)
!3702 = !DILocation(line: 1563, column: 16, scope: !3676)
!3703 = !DILocation(line: 1564, column: 18, scope: !3676)
!3704 = !DILocation(line: 1564, column: 2, scope: !3676)
!3705 = !DILocation(line: 1564, column: 5, scope: !3676)
!3706 = !DILocation(line: 1564, column: 16, scope: !3676)
!3707 = !DILocation(line: 1565, column: 2, scope: !3676)
!3708 = !DILocation(line: 1565, column: 10, scope: !3676)
!3709 = !DILocation(line: 1565, column: 21, scope: !3676)
!3710 = !DILocation(line: 1566, column: 1, scope: !3676)
!3711 = distinct !DISubprogram(name: "ipow46_device", linkageName: "_Z13ipow46_devicediPd", scope: !3, file: !3, line: 1599, type: !3712, scopeLine: 1601, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !1050, retainedNodes: !1051)
!3712 = !DISubroutineType(types: !3713)
!3713 = !{null, !104, !97, !106}
!3714 = !DILocalVariable(name: "a", arg: 1, scope: !3711, file: !3, line: 1599, type: !104)
!3715 = !DILocation(line: 1599, column: 38, scope: !3711)
!3716 = !DILocalVariable(name: "exponent", arg: 2, scope: !3711, file: !3, line: 1600, type: !97)
!3717 = !DILocation(line: 1600, column: 7, scope: !3711)
!3718 = !DILocalVariable(name: "result", arg: 3, scope: !3711, file: !3, line: 1601, type: !106)
!3719 = !DILocation(line: 1601, column: 11, scope: !3711)
!3720 = !DILocalVariable(name: "q", scope: !3711, file: !3, line: 1602, type: !104)
!3721 = !DILocation(line: 1602, column: 9, scope: !3711)
!3722 = !DILocalVariable(name: "r", scope: !3711, file: !3, line: 1602, type: !104)
!3723 = !DILocation(line: 1602, column: 12, scope: !3711)
!3724 = !DILocalVariable(name: "n", scope: !3711, file: !3, line: 1603, type: !97)
!3725 = !DILocation(line: 1603, column: 6, scope: !3711)
!3726 = !DILocalVariable(name: "n2", scope: !3711, file: !3, line: 1603, type: !97)
!3727 = !DILocation(line: 1603, column: 9, scope: !3711)
!3728 = !DILocation(line: 1611, column: 3, scope: !3711)
!3729 = !DILocation(line: 1611, column: 10, scope: !3711)
!3730 = !DILocation(line: 1612, column: 5, scope: !3731)
!3731 = distinct !DILexicalBlock(scope: !3711, file: !3, line: 1612, column: 5)
!3732 = !DILocation(line: 1612, column: 13, scope: !3731)
!3733 = !DILocation(line: 1612, column: 5, scope: !3711)
!3734 = !DILocation(line: 1612, column: 18, scope: !3735)
!3735 = distinct !DILexicalBlock(scope: !3731, file: !3, line: 1612, column: 17)
!3736 = !DILocation(line: 1613, column: 6, scope: !3711)
!3737 = !DILocation(line: 1613, column: 4, scope: !3711)
!3738 = !DILocation(line: 1614, column: 4, scope: !3711)
!3739 = !DILocation(line: 1615, column: 6, scope: !3711)
!3740 = !DILocation(line: 1615, column: 4, scope: !3711)
!3741 = !DILocation(line: 1616, column: 2, scope: !3711)
!3742 = !DILocation(line: 1616, column: 8, scope: !3711)
!3743 = !DILocation(line: 1616, column: 9, scope: !3711)
!3744 = !DILocation(line: 1617, column: 8, scope: !3745)
!3745 = distinct !DILexicalBlock(scope: !3711, file: !3, line: 1616, column: 12)
!3746 = !DILocation(line: 1617, column: 9, scope: !3745)
!3747 = !DILocation(line: 1617, column: 6, scope: !3745)
!3748 = !DILocation(line: 1618, column: 6, scope: !3749)
!3749 = distinct !DILexicalBlock(scope: !3745, file: !3, line: 1618, column: 6)
!3750 = !DILocation(line: 1618, column: 8, scope: !3749)
!3751 = !DILocation(line: 1618, column: 12, scope: !3749)
!3752 = !DILocation(line: 1618, column: 10, scope: !3749)
!3753 = !DILocation(line: 1618, column: 6, scope: !3745)
!3754 = !DILocation(line: 1619, column: 22, scope: !3755)
!3755 = distinct !DILexicalBlock(scope: !3749, file: !3, line: 1618, column: 14)
!3756 = !DILocation(line: 1619, column: 4, scope: !3755)
!3757 = !DILocation(line: 1620, column: 8, scope: !3755)
!3758 = !DILocation(line: 1620, column: 6, scope: !3755)
!3759 = !DILocation(line: 1621, column: 3, scope: !3755)
!3760 = !DILocation(line: 1622, column: 22, scope: !3761)
!3761 = distinct !DILexicalBlock(scope: !3749, file: !3, line: 1621, column: 8)
!3762 = !DILocation(line: 1622, column: 4, scope: !3761)
!3763 = !DILocation(line: 1623, column: 8, scope: !3761)
!3764 = !DILocation(line: 1623, column: 9, scope: !3761)
!3765 = !DILocation(line: 1623, column: 6, scope: !3761)
!3766 = distinct !{!3766, !3741, !3767}
!3767 = !DILocation(line: 1625, column: 2, scope: !3711)
!3768 = !DILocation(line: 1626, column: 20, scope: !3711)
!3769 = !DILocation(line: 1626, column: 2, scope: !3711)
!3770 = !DILocation(line: 1627, column: 12, scope: !3711)
!3771 = !DILocation(line: 1627, column: 3, scope: !3711)
!3772 = !DILocation(line: 1627, column: 10, scope: !3711)
!3773 = !DILocation(line: 1628, column: 1, scope: !3711)
!3774 = distinct !DISubprogram(name: "randlc_device", linkageName: "_Z13randlc_devicePdd", scope: !3, file: !3, line: 1630, type: !3277, scopeLine: 1631, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !1050, retainedNodes: !1051)
!3775 = !DILocalVariable(name: "x", arg: 1, scope: !3774, file: !3, line: 1630, type: !106)
!3776 = !DILocation(line: 1630, column: 41, scope: !3774)
!3777 = !DILocalVariable(name: "a", arg: 2, scope: !3774, file: !3, line: 1631, type: !104)
!3778 = !DILocation(line: 1631, column: 10, scope: !3774)
!3779 = !DILocalVariable(name: "t1", scope: !3774, file: !3, line: 1632, type: !104)
!3780 = !DILocation(line: 1632, column: 9, scope: !3774)
!3781 = !DILocalVariable(name: "t2", scope: !3774, file: !3, line: 1632, type: !104)
!3782 = !DILocation(line: 1632, column: 12, scope: !3774)
!3783 = !DILocalVariable(name: "t3", scope: !3774, file: !3, line: 1632, type: !104)
!3784 = !DILocation(line: 1632, column: 15, scope: !3774)
!3785 = !DILocalVariable(name: "t4", scope: !3774, file: !3, line: 1632, type: !104)
!3786 = !DILocation(line: 1632, column: 18, scope: !3774)
!3787 = !DILocalVariable(name: "a1", scope: !3774, file: !3, line: 1632, type: !104)
!3788 = !DILocation(line: 1632, column: 21, scope: !3774)
!3789 = !DILocalVariable(name: "a2", scope: !3774, file: !3, line: 1632, type: !104)
!3790 = !DILocation(line: 1632, column: 24, scope: !3774)
!3791 = !DILocalVariable(name: "x1", scope: !3774, file: !3, line: 1632, type: !104)
!3792 = !DILocation(line: 1632, column: 27, scope: !3774)
!3793 = !DILocalVariable(name: "x2", scope: !3774, file: !3, line: 1632, type: !104)
!3794 = !DILocation(line: 1632, column: 30, scope: !3774)
!3795 = !DILocalVariable(name: "z", scope: !3774, file: !3, line: 1632, type: !104)
!3796 = !DILocation(line: 1632, column: 33, scope: !3774)
!3797 = !DILocation(line: 1633, column: 13, scope: !3774)
!3798 = !DILocation(line: 1633, column: 11, scope: !3774)
!3799 = !DILocation(line: 1633, column: 5, scope: !3774)
!3800 = !DILocation(line: 1634, column: 12, scope: !3774)
!3801 = !DILocation(line: 1634, column: 7, scope: !3774)
!3802 = !DILocation(line: 1634, column: 5, scope: !3774)
!3803 = !DILocation(line: 1635, column: 7, scope: !3774)
!3804 = !DILocation(line: 1635, column: 17, scope: !3774)
!3805 = !DILocation(line: 1635, column: 15, scope: !3774)
!3806 = !DILocation(line: 1635, column: 9, scope: !3774)
!3807 = !DILocation(line: 1635, column: 5, scope: !3774)
!3808 = !DILocation(line: 1636, column: 15, scope: !3774)
!3809 = !DILocation(line: 1636, column: 14, scope: !3774)
!3810 = !DILocation(line: 1636, column: 11, scope: !3774)
!3811 = !DILocation(line: 1636, column: 5, scope: !3774)
!3812 = !DILocation(line: 1637, column: 12, scope: !3774)
!3813 = !DILocation(line: 1637, column: 7, scope: !3774)
!3814 = !DILocation(line: 1637, column: 5, scope: !3774)
!3815 = !DILocation(line: 1638, column: 9, scope: !3774)
!3816 = !DILocation(line: 1638, column: 8, scope: !3774)
!3817 = !DILocation(line: 1638, column: 20, scope: !3774)
!3818 = !DILocation(line: 1638, column: 18, scope: !3774)
!3819 = !DILocation(line: 1638, column: 12, scope: !3774)
!3820 = !DILocation(line: 1638, column: 5, scope: !3774)
!3821 = !DILocation(line: 1639, column: 7, scope: !3774)
!3822 = !DILocation(line: 1639, column: 12, scope: !3774)
!3823 = !DILocation(line: 1639, column: 10, scope: !3774)
!3824 = !DILocation(line: 1639, column: 17, scope: !3774)
!3825 = !DILocation(line: 1639, column: 22, scope: !3774)
!3826 = !DILocation(line: 1639, column: 20, scope: !3774)
!3827 = !DILocation(line: 1639, column: 15, scope: !3774)
!3828 = !DILocation(line: 1639, column: 5, scope: !3774)
!3829 = !DILocation(line: 1640, column: 19, scope: !3774)
!3830 = !DILocation(line: 1640, column: 17, scope: !3774)
!3831 = !DILocation(line: 1640, column: 12, scope: !3774)
!3832 = !DILocation(line: 1640, column: 7, scope: !3774)
!3833 = !DILocation(line: 1640, column: 5, scope: !3774)
!3834 = !DILocation(line: 1641, column: 6, scope: !3774)
!3835 = !DILocation(line: 1641, column: 17, scope: !3774)
!3836 = !DILocation(line: 1641, column: 15, scope: !3774)
!3837 = !DILocation(line: 1641, column: 9, scope: !3774)
!3838 = !DILocation(line: 1641, column: 4, scope: !3774)
!3839 = !DILocation(line: 1642, column: 13, scope: !3774)
!3840 = !DILocation(line: 1642, column: 11, scope: !3774)
!3841 = !DILocation(line: 1642, column: 17, scope: !3774)
!3842 = !DILocation(line: 1642, column: 22, scope: !3774)
!3843 = !DILocation(line: 1642, column: 20, scope: !3774)
!3844 = !DILocation(line: 1642, column: 15, scope: !3774)
!3845 = !DILocation(line: 1642, column: 5, scope: !3774)
!3846 = !DILocation(line: 1643, column: 19, scope: !3774)
!3847 = !DILocation(line: 1643, column: 17, scope: !3774)
!3848 = !DILocation(line: 1643, column: 12, scope: !3774)
!3849 = !DILocation(line: 1643, column: 7, scope: !3774)
!3850 = !DILocation(line: 1643, column: 5, scope: !3774)
!3851 = !DILocation(line: 1644, column: 9, scope: !3774)
!3852 = !DILocation(line: 1644, column: 20, scope: !3774)
!3853 = !DILocation(line: 1644, column: 18, scope: !3774)
!3854 = !DILocation(line: 1644, column: 12, scope: !3774)
!3855 = !DILocation(line: 1644, column: 4, scope: !3774)
!3856 = !DILocation(line: 1644, column: 7, scope: !3774)
!3857 = !DILocation(line: 1645, column: 18, scope: !3774)
!3858 = !DILocation(line: 1645, column: 17, scope: !3774)
!3859 = !DILocation(line: 1645, column: 14, scope: !3774)
!3860 = !DILocation(line: 1645, column: 2, scope: !3774)
!3861 = distinct !DISubprogram(name: "randlc", linkageName: "_Z6randlcPdd", scope: !3, file: !3, line: 371, type: !3277, scopeLine: 371, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !1051)
!3862 = !DILocalVariable(name: "x", arg: 1, scope: !3861, file: !3, line: 371, type: !106)
!3863 = !DILocation(line: 371, column: 24, scope: !3861)
!3864 = !DILocalVariable(name: "a", arg: 2, scope: !3861, file: !3, line: 371, type: !104)
!3865 = !DILocation(line: 371, column: 34, scope: !3861)
!3866 = !DILocalVariable(name: "t1", scope: !3861, file: !3, line: 372, type: !104)
!3867 = !DILocation(line: 372, column: 10, scope: !3861)
!3868 = !DILocalVariable(name: "t2", scope: !3861, file: !3, line: 372, type: !104)
!3869 = !DILocation(line: 372, column: 13, scope: !3861)
!3870 = !DILocalVariable(name: "t3", scope: !3861, file: !3, line: 372, type: !104)
!3871 = !DILocation(line: 372, column: 16, scope: !3861)
!3872 = !DILocalVariable(name: "t4", scope: !3861, file: !3, line: 372, type: !104)
!3873 = !DILocation(line: 372, column: 19, scope: !3861)
!3874 = !DILocalVariable(name: "a1", scope: !3861, file: !3, line: 372, type: !104)
!3875 = !DILocation(line: 372, column: 22, scope: !3861)
!3876 = !DILocalVariable(name: "a2", scope: !3861, file: !3, line: 372, type: !104)
!3877 = !DILocation(line: 372, column: 25, scope: !3861)
!3878 = !DILocalVariable(name: "x1", scope: !3861, file: !3, line: 372, type: !104)
!3879 = !DILocation(line: 372, column: 28, scope: !3861)
!3880 = !DILocalVariable(name: "x2", scope: !3861, file: !3, line: 372, type: !104)
!3881 = !DILocation(line: 372, column: 31, scope: !3861)
!3882 = !DILocalVariable(name: "z", scope: !3861, file: !3, line: 372, type: !104)
!3883 = !DILocation(line: 372, column: 34, scope: !3861)
!3884 = !DILocation(line: 379, column: 14, scope: !3861)
!3885 = !DILocation(line: 379, column: 12, scope: !3861)
!3886 = !DILocation(line: 379, column: 6, scope: !3861)
!3887 = !DILocation(line: 380, column: 13, scope: !3861)
!3888 = !DILocation(line: 380, column: 8, scope: !3861)
!3889 = !DILocation(line: 380, column: 6, scope: !3861)
!3890 = !DILocation(line: 381, column: 8, scope: !3861)
!3891 = !DILocation(line: 381, column: 18, scope: !3861)
!3892 = !DILocation(line: 381, column: 16, scope: !3861)
!3893 = !DILocation(line: 381, column: 10, scope: !3861)
!3894 = !DILocation(line: 381, column: 6, scope: !3861)
!3895 = !DILocation(line: 390, column: 16, scope: !3861)
!3896 = !DILocation(line: 390, column: 15, scope: !3861)
!3897 = !DILocation(line: 390, column: 12, scope: !3861)
!3898 = !DILocation(line: 390, column: 6, scope: !3861)
!3899 = !DILocation(line: 391, column: 13, scope: !3861)
!3900 = !DILocation(line: 391, column: 8, scope: !3861)
!3901 = !DILocation(line: 391, column: 6, scope: !3861)
!3902 = !DILocation(line: 392, column: 10, scope: !3861)
!3903 = !DILocation(line: 392, column: 9, scope: !3861)
!3904 = !DILocation(line: 392, column: 21, scope: !3861)
!3905 = !DILocation(line: 392, column: 19, scope: !3861)
!3906 = !DILocation(line: 392, column: 13, scope: !3861)
!3907 = !DILocation(line: 392, column: 6, scope: !3861)
!3908 = !DILocation(line: 393, column: 8, scope: !3861)
!3909 = !DILocation(line: 393, column: 13, scope: !3861)
!3910 = !DILocation(line: 393, column: 11, scope: !3861)
!3911 = !DILocation(line: 393, column: 18, scope: !3861)
!3912 = !DILocation(line: 393, column: 23, scope: !3861)
!3913 = !DILocation(line: 393, column: 21, scope: !3861)
!3914 = !DILocation(line: 393, column: 16, scope: !3861)
!3915 = !DILocation(line: 393, column: 6, scope: !3861)
!3916 = !DILocation(line: 394, column: 20, scope: !3861)
!3917 = !DILocation(line: 394, column: 18, scope: !3861)
!3918 = !DILocation(line: 394, column: 13, scope: !3861)
!3919 = !DILocation(line: 394, column: 8, scope: !3861)
!3920 = !DILocation(line: 394, column: 6, scope: !3861)
!3921 = !DILocation(line: 395, column: 7, scope: !3861)
!3922 = !DILocation(line: 395, column: 18, scope: !3861)
!3923 = !DILocation(line: 395, column: 16, scope: !3861)
!3924 = !DILocation(line: 395, column: 10, scope: !3861)
!3925 = !DILocation(line: 395, column: 5, scope: !3861)
!3926 = !DILocation(line: 396, column: 14, scope: !3861)
!3927 = !DILocation(line: 396, column: 12, scope: !3861)
!3928 = !DILocation(line: 396, column: 18, scope: !3861)
!3929 = !DILocation(line: 396, column: 23, scope: !3861)
!3930 = !DILocation(line: 396, column: 21, scope: !3861)
!3931 = !DILocation(line: 396, column: 16, scope: !3861)
!3932 = !DILocation(line: 396, column: 6, scope: !3861)
!3933 = !DILocation(line: 397, column: 20, scope: !3861)
!3934 = !DILocation(line: 397, column: 18, scope: !3861)
!3935 = !DILocation(line: 397, column: 13, scope: !3861)
!3936 = !DILocation(line: 397, column: 8, scope: !3861)
!3937 = !DILocation(line: 397, column: 6, scope: !3861)
!3938 = !DILocation(line: 398, column: 10, scope: !3861)
!3939 = !DILocation(line: 398, column: 21, scope: !3861)
!3940 = !DILocation(line: 398, column: 19, scope: !3861)
!3941 = !DILocation(line: 398, column: 13, scope: !3861)
!3942 = !DILocation(line: 398, column: 5, scope: !3861)
!3943 = !DILocation(line: 398, column: 8, scope: !3861)
!3944 = !DILocation(line: 400, column: 19, scope: !3861)
!3945 = !DILocation(line: 400, column: 18, scope: !3861)
!3946 = !DILocation(line: 400, column: 15, scope: !3861)
!3947 = !DILocation(line: 400, column: 3, scope: !3861)
!3948 = distinct !DISubprogram(name: "c_print_results", linkageName: "_Z15c_print_resultsPcciiiiddS_iS_S_S_S_S_S_S_S_S_S_S_S_S_S_", scope: !3, file: !3, line: 406, type: !3949, scopeLine: 429, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !1051)
!3949 = !DISubroutineType(types: !3950)
!3950 = !{null, !108, !109, !97, !97, !97, !97, !104, !104, !108, !97, !108, !108, !108, !108, !108, !108, !108, !108, !108, !108, !108, !108, !108, !108}
!3951 = !DILocalVariable(name: "name", arg: 1, scope: !3948, file: !3, line: 406, type: !108)
!3952 = !DILocation(line: 406, column: 29, scope: !3948)
!3953 = !DILocalVariable(name: "class_npb", arg: 2, scope: !3948, file: !3, line: 407, type: !109)
!3954 = !DILocation(line: 407, column: 9, scope: !3948)
!3955 = !DILocalVariable(name: "n1", arg: 3, scope: !3948, file: !3, line: 408, type: !97)
!3956 = !DILocation(line: 408, column: 8, scope: !3948)
!3957 = !DILocalVariable(name: "n2", arg: 4, scope: !3948, file: !3, line: 409, type: !97)
!3958 = !DILocation(line: 409, column: 8, scope: !3948)
!3959 = !DILocalVariable(name: "n3", arg: 5, scope: !3948, file: !3, line: 410, type: !97)
!3960 = !DILocation(line: 410, column: 8, scope: !3948)
!3961 = !DILocalVariable(name: "niter", arg: 6, scope: !3948, file: !3, line: 411, type: !97)
!3962 = !DILocation(line: 411, column: 8, scope: !3948)
!3963 = !DILocalVariable(name: "t", arg: 7, scope: !3948, file: !3, line: 412, type: !104)
!3964 = !DILocation(line: 412, column: 11, scope: !3948)
!3965 = !DILocalVariable(name: "mops", arg: 8, scope: !3948, file: !3, line: 413, type: !104)
!3966 = !DILocation(line: 413, column: 11, scope: !3948)
!3967 = !DILocalVariable(name: "optype", arg: 9, scope: !3948, file: !3, line: 414, type: !108)
!3968 = !DILocation(line: 414, column: 10, scope: !3948)
!3969 = !DILocalVariable(name: "passed_verification", arg: 10, scope: !3948, file: !3, line: 415, type: !97)
!3970 = !DILocation(line: 415, column: 8, scope: !3948)
!3971 = !DILocalVariable(name: "npbversion", arg: 11, scope: !3948, file: !3, line: 416, type: !108)
!3972 = !DILocation(line: 416, column: 10, scope: !3948)
!3973 = !DILocalVariable(name: "compiletime", arg: 12, scope: !3948, file: !3, line: 417, type: !108)
!3974 = !DILocation(line: 417, column: 10, scope: !3948)
!3975 = !DILocalVariable(name: "compilerversion", arg: 13, scope: !3948, file: !3, line: 418, type: !108)
!3976 = !DILocation(line: 418, column: 10, scope: !3948)
!3977 = !DILocalVariable(name: "libversion", arg: 14, scope: !3948, file: !3, line: 419, type: !108)
!3978 = !DILocation(line: 419, column: 10, scope: !3948)
!3979 = !DILocalVariable(name: "cpu_device", arg: 15, scope: !3948, file: !3, line: 420, type: !108)
!3980 = !DILocation(line: 420, column: 10, scope: !3948)
!3981 = !DILocalVariable(name: "gpu_device", arg: 16, scope: !3948, file: !3, line: 421, type: !108)
!3982 = !DILocation(line: 421, column: 10, scope: !3948)
!3983 = !DILocalVariable(name: "gpu_config", arg: 17, scope: !3948, file: !3, line: 422, type: !108)
!3984 = !DILocation(line: 422, column: 10, scope: !3948)
!3985 = !DILocalVariable(name: "cc", arg: 18, scope: !3948, file: !3, line: 423, type: !108)
!3986 = !DILocation(line: 423, column: 10, scope: !3948)
!3987 = !DILocalVariable(name: "clink", arg: 19, scope: !3948, file: !3, line: 424, type: !108)
!3988 = !DILocation(line: 424, column: 10, scope: !3948)
!3989 = !DILocalVariable(name: "c_lib", arg: 20, scope: !3948, file: !3, line: 425, type: !108)
!3990 = !DILocation(line: 425, column: 10, scope: !3948)
!3991 = !DILocalVariable(name: "c_inc", arg: 21, scope: !3948, file: !3, line: 426, type: !108)
!3992 = !DILocation(line: 426, column: 10, scope: !3948)
!3993 = !DILocalVariable(name: "cflags", arg: 22, scope: !3948, file: !3, line: 427, type: !108)
!3994 = !DILocation(line: 427, column: 10, scope: !3948)
!3995 = !DILocalVariable(name: "clinkflags", arg: 23, scope: !3948, file: !3, line: 428, type: !108)
!3996 = !DILocation(line: 428, column: 10, scope: !3948)
!3997 = !DILocalVariable(name: "rand", arg: 24, scope: !3948, file: !3, line: 429, type: !108)
!3998 = !DILocation(line: 429, column: 10, scope: !3948)
!3999 = !DILocation(line: 430, column: 45, scope: !3948)
!4000 = !DILocation(line: 430, column: 5, scope: !3948)
!4001 = !DILocation(line: 431, column: 62, scope: !3948)
!4002 = !DILocation(line: 431, column: 5, scope: !3948)
!4003 = !DILocation(line: 432, column: 9, scope: !4004)
!4004 = distinct !DILexicalBlock(scope: !3948, file: !3, line: 432, column: 8)
!4005 = !DILocation(line: 432, column: 16, scope: !4004)
!4006 = !DILocation(line: 432, column: 22, scope: !4004)
!4007 = !DILocation(line: 432, column: 25, scope: !4004)
!4008 = !DILocation(line: 432, column: 32, scope: !4004)
!4009 = !DILocation(line: 432, column: 8, scope: !3948)
!4010 = !DILocation(line: 433, column: 9, scope: !4011)
!4011 = distinct !DILexicalBlock(scope: !4012, file: !3, line: 433, column: 9)
!4012 = distinct !DILexicalBlock(scope: !4004, file: !3, line: 432, column: 39)
!4013 = !DILocation(line: 433, column: 11, scope: !4011)
!4014 = !DILocation(line: 433, column: 9, scope: !4012)
!4015 = !DILocalVariable(name: "nn", scope: !4016, file: !3, line: 434, type: !402)
!4016 = distinct !DILexicalBlock(scope: !4011, file: !3, line: 433, column: 15)
!4017 = !DILocation(line: 434, column: 12, scope: !4016)
!4018 = !DILocation(line: 434, column: 17, scope: !4016)
!4019 = !DILocation(line: 435, column: 10, scope: !4020)
!4020 = distinct !DILexicalBlock(scope: !4016, file: !3, line: 435, column: 10)
!4021 = !DILocation(line: 435, column: 12, scope: !4020)
!4022 = !DILocation(line: 435, column: 10, scope: !4016)
!4023 = !DILocation(line: 435, column: 21, scope: !4024)
!4024 = distinct !DILexicalBlock(scope: !4020, file: !3, line: 435, column: 16)
!4025 = !DILocation(line: 435, column: 19, scope: !4024)
!4026 = !DILocation(line: 435, column: 24, scope: !4024)
!4027 = !DILocation(line: 436, column: 56, scope: !4016)
!4028 = !DILocation(line: 436, column: 7, scope: !4016)
!4029 = !DILocation(line: 437, column: 6, scope: !4016)
!4030 = !DILocation(line: 438, column: 62, scope: !4031)
!4031 = distinct !DILexicalBlock(scope: !4011, file: !3, line: 437, column: 11)
!4032 = !DILocation(line: 438, column: 65, scope: !4031)
!4033 = !DILocation(line: 438, column: 68, scope: !4031)
!4034 = !DILocation(line: 438, column: 7, scope: !4031)
!4035 = !DILocation(line: 440, column: 5, scope: !4012)
!4036 = !DILocalVariable(name: "size", scope: !4037, file: !3, line: 441, type: !4038)
!4037 = distinct !DILexicalBlock(scope: !4004, file: !3, line: 440, column: 10)
!4038 = !DICompositeType(tag: DW_TAG_array_type, baseType: !109, size: 128, elements: !4039)
!4039 = !{!4040}
!4040 = !DISubrange(count: 16)
!4041 = !DILocation(line: 441, column: 11, scope: !4037)
!4042 = !DILocalVariable(name: "j", scope: !4037, file: !3, line: 442, type: !97)
!4043 = !DILocation(line: 442, column: 10, scope: !4037)
!4044 = !DILocation(line: 443, column: 10, scope: !4045)
!4045 = distinct !DILexicalBlock(scope: !4037, file: !3, line: 443, column: 9)
!4046 = !DILocation(line: 443, column: 12, scope: !4045)
!4047 = !DILocation(line: 443, column: 17, scope: !4045)
!4048 = !DILocation(line: 443, column: 21, scope: !4045)
!4049 = !DILocation(line: 443, column: 23, scope: !4045)
!4050 = !DILocation(line: 443, column: 9, scope: !4037)
!4051 = !DILocation(line: 444, column: 11, scope: !4052)
!4052 = distinct !DILexicalBlock(scope: !4053, file: !3, line: 444, column: 10)
!4053 = distinct !DILexicalBlock(scope: !4045, file: !3, line: 443, column: 28)
!4054 = !DILocation(line: 444, column: 18, scope: !4052)
!4055 = !DILocation(line: 444, column: 24, scope: !4052)
!4056 = !DILocation(line: 444, column: 27, scope: !4052)
!4057 = !DILocation(line: 444, column: 34, scope: !4052)
!4058 = !DILocation(line: 444, column: 10, scope: !4053)
!4059 = !DILocation(line: 445, column: 16, scope: !4060)
!4060 = distinct !DILexicalBlock(scope: !4052, file: !3, line: 444, column: 41)
!4061 = !DILocation(line: 445, column: 42, scope: !4060)
!4062 = !DILocation(line: 445, column: 33, scope: !4060)
!4063 = !DILocation(line: 445, column: 8, scope: !4060)
!4064 = !DILocation(line: 446, column: 10, scope: !4060)
!4065 = !DILocation(line: 447, column: 16, scope: !4066)
!4066 = distinct !DILexicalBlock(scope: !4060, file: !3, line: 447, column: 11)
!4067 = !DILocation(line: 447, column: 11, scope: !4066)
!4068 = !DILocation(line: 447, column: 19, scope: !4066)
!4069 = !DILocation(line: 447, column: 11, scope: !4060)
!4070 = !DILocation(line: 448, column: 14, scope: !4071)
!4071 = distinct !DILexicalBlock(scope: !4066, file: !3, line: 447, column: 26)
!4072 = !DILocation(line: 448, column: 9, scope: !4071)
!4073 = !DILocation(line: 448, column: 17, scope: !4071)
!4074 = !DILocation(line: 449, column: 10, scope: !4071)
!4075 = !DILocation(line: 450, column: 8, scope: !4071)
!4076 = !DILocation(line: 451, column: 13, scope: !4060)
!4077 = !DILocation(line: 451, column: 14, scope: !4060)
!4078 = !DILocation(line: 451, column: 8, scope: !4060)
!4079 = !DILocation(line: 451, column: 18, scope: !4060)
!4080 = !DILocation(line: 452, column: 53, scope: !4060)
!4081 = !DILocation(line: 452, column: 8, scope: !4060)
!4082 = !DILocation(line: 453, column: 7, scope: !4060)
!4083 = !DILocation(line: 454, column: 56, scope: !4084)
!4084 = distinct !DILexicalBlock(scope: !4052, file: !3, line: 453, column: 12)
!4085 = !DILocation(line: 454, column: 8, scope: !4084)
!4086 = !DILocation(line: 456, column: 6, scope: !4053)
!4087 = !DILocation(line: 457, column: 60, scope: !4088)
!4088 = distinct !DILexicalBlock(scope: !4045, file: !3, line: 456, column: 11)
!4089 = !DILocation(line: 457, column: 64, scope: !4088)
!4090 = !DILocation(line: 457, column: 68, scope: !4088)
!4091 = !DILocation(line: 457, column: 7, scope: !4088)
!4092 = !DILocation(line: 460, column: 53, scope: !3948)
!4093 = !DILocation(line: 460, column: 5, scope: !3948)
!4094 = !DILocation(line: 461, column: 55, scope: !3948)
!4095 = !DILocation(line: 461, column: 5, scope: !3948)
!4096 = !DILocation(line: 462, column: 55, scope: !3948)
!4097 = !DILocation(line: 462, column: 5, scope: !3948)
!4098 = !DILocation(line: 463, column: 41, scope: !3948)
!4099 = !DILocation(line: 463, column: 5, scope: !3948)
!4100 = !DILocation(line: 464, column: 8, scope: !4101)
!4101 = distinct !DILexicalBlock(scope: !3948, file: !3, line: 464, column: 8)
!4102 = !DILocation(line: 464, column: 28, scope: !4101)
!4103 = !DILocation(line: 464, column: 8, scope: !3948)
!4104 = !DILocation(line: 465, column: 6, scope: !4105)
!4105 = distinct !DILexicalBlock(scope: !4101, file: !3, line: 464, column: 32)
!4106 = !DILocation(line: 466, column: 5, scope: !4105)
!4107 = !DILocation(line: 466, column: 14, scope: !4108)
!4108 = distinct !DILexicalBlock(scope: !4101, file: !3, line: 466, column: 14)
!4109 = !DILocation(line: 466, column: 14, scope: !4101)
!4110 = !DILocation(line: 467, column: 6, scope: !4111)
!4111 = distinct !DILexicalBlock(scope: !4108, file: !3, line: 466, column: 34)
!4112 = !DILocation(line: 468, column: 5, scope: !4111)
!4113 = !DILocation(line: 469, column: 6, scope: !4114)
!4114 = distinct !DILexicalBlock(scope: !4108, file: !3, line: 468, column: 10)
!4115 = !DILocation(line: 471, column: 53, scope: !3948)
!4116 = !DILocation(line: 471, column: 5, scope: !3948)
!4117 = !DILocation(line: 472, column: 53, scope: !3948)
!4118 = !DILocation(line: 472, column: 5, scope: !3948)
!4119 = !DILocation(line: 473, column: 53, scope: !3948)
!4120 = !DILocation(line: 473, column: 5, scope: !3948)
!4121 = !DILocation(line: 474, column: 53, scope: !3948)
!4122 = !DILocation(line: 474, column: 5, scope: !3948)
!4123 = !DILocation(line: 475, column: 5, scope: !3948)
!4124 = !DILocation(line: 476, column: 39, scope: !3948)
!4125 = !DILocation(line: 476, column: 5, scope: !3948)
!4126 = !DILocation(line: 477, column: 39, scope: !3948)
!4127 = !DILocation(line: 477, column: 5, scope: !3948)
!4128 = !DILocation(line: 478, column: 39, scope: !3948)
!4129 = !DILocation(line: 478, column: 5, scope: !3948)
!4130 = !DILocation(line: 479, column: 39, scope: !3948)
!4131 = !DILocation(line: 479, column: 5, scope: !3948)
!4132 = !DILocation(line: 480, column: 39, scope: !3948)
!4133 = !DILocation(line: 480, column: 5, scope: !3948)
!4134 = !DILocation(line: 481, column: 39, scope: !3948)
!4135 = !DILocation(line: 481, column: 5, scope: !3948)
!4136 = !DILocation(line: 482, column: 39, scope: !3948)
!4137 = !DILocation(line: 482, column: 5, scope: !3948)
!4138 = !DILocation(line: 483, column: 5, scope: !3948)
!4139 = !DILocation(line: 484, column: 39, scope: !3948)
!4140 = !DILocation(line: 484, column: 5, scope: !3948)
!4141 = !DILocation(line: 485, column: 39, scope: !3948)
!4142 = !DILocation(line: 485, column: 5, scope: !3948)
!4143 = !DILocation(line: 486, column: 5, scope: !3948)
!4144 = !DILocation(line: 487, column: 39, scope: !3948)
!4145 = !DILocation(line: 487, column: 5, scope: !3948)
!4146 = !DILocation(line: 502, column: 5, scope: !3948)
!4147 = !DILocation(line: 503, column: 5, scope: !3948)
!4148 = !DILocation(line: 504, column: 5, scope: !3948)
!4149 = !DILocation(line: 505, column: 5, scope: !3948)
!4150 = !DILocation(line: 506, column: 5, scope: !3948)
!4151 = !DILocation(line: 507, column: 5, scope: !3948)
!4152 = !DILocation(line: 508, column: 5, scope: !3948)
!4153 = !DILocation(line: 509, column: 5, scope: !3948)
!4154 = !DILocation(line: 510, column: 5, scope: !3948)
!4155 = !DILocation(line: 511, column: 5, scope: !3948)
!4156 = !DILocation(line: 512, column: 4, scope: !3948)
!4157 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 516, type: !4158, scopeLine: 516, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !1051)
!4158 = !DISubroutineType(types: !4159)
!4159 = !{!97, !97, !655}
!4160 = !DILocalVariable(name: "argc", arg: 1, scope: !4157, file: !3, line: 516, type: !97)
!4161 = !DILocation(line: 516, column: 14, scope: !4157)
!4162 = !DILocalVariable(name: "argv", arg: 2, scope: !4157, file: !3, line: 516, type: !655)
!4163 = !DILocation(line: 516, column: 27, scope: !4157)
!4164 = !DILocalVariable(name: "iter", scope: !4157, file: !3, line: 523, type: !97)
!4165 = !DILocation(line: 523, column: 6, scope: !4157)
!4166 = !DILocalVariable(name: "total_time", scope: !4157, file: !3, line: 524, type: !104)
!4167 = !DILocation(line: 524, column: 9, scope: !4157)
!4168 = !DILocalVariable(name: "mflops", scope: !4157, file: !3, line: 524, type: !104)
!4169 = !DILocation(line: 524, column: 21, scope: !4157)
!4170 = !DILocalVariable(name: "verified", scope: !4157, file: !3, line: 525, type: !4171)
!4171 = !DIDerivedType(tag: DW_TAG_typedef, name: "boolean", file: !100, line: 80, baseType: !97)
!4172 = !DILocation(line: 525, column: 10, scope: !4157)
!4173 = !DILocalVariable(name: "class_npb", scope: !4157, file: !3, line: 526, type: !109)
!4174 = !DILocation(line: 526, column: 7, scope: !4157)
!4175 = !DILocation(line: 529, column: 20, scope: !4157)
!4176 = !DILocation(line: 529, column: 9, scope: !4157)
!4177 = !DILocation(line: 529, column: 7, scope: !4157)
!4178 = !DILocation(line: 530, column: 21, scope: !4157)
!4179 = !DILocation(line: 530, column: 12, scope: !4157)
!4180 = !DILocation(line: 530, column: 10, scope: !4157)
!4181 = !DILocation(line: 531, column: 17, scope: !4157)
!4182 = !DILocation(line: 531, column: 6, scope: !4157)
!4183 = !DILocation(line: 531, column: 4, scope: !4157)
!4184 = !DILocation(line: 532, column: 18, scope: !4157)
!4185 = !DILocation(line: 532, column: 7, scope: !4157)
!4186 = !DILocation(line: 532, column: 5, scope: !4157)
!4187 = !DILocation(line: 533, column: 18, scope: !4157)
!4188 = !DILocation(line: 533, column: 7, scope: !4157)
!4189 = !DILocation(line: 533, column: 5, scope: !4157)
!4190 = !DILocation(line: 534, column: 15, scope: !4157)
!4191 = !DILocation(line: 534, column: 9, scope: !4157)
!4192 = !DILocation(line: 534, column: 7, scope: !4157)
!4193 = !DILocation(line: 544, column: 2, scope: !4157)
!4194 = !DILocation(line: 545, column: 2, scope: !4157)
!4195 = !DILocation(line: 546, column: 14, scope: !4157)
!4196 = !DILocation(line: 546, column: 25, scope: !4157)
!4197 = !DILocation(line: 546, column: 36, scope: !4157)
!4198 = !DILocation(line: 546, column: 2, scope: !4157)
!4199 = !DILocation(line: 549, column: 6, scope: !4200)
!4200 = distinct !DILexicalBlock(scope: !4201, file: !3, line: 549, column: 6)
!4201 = distinct !DILexicalBlock(scope: !4157, file: !3, line: 548, column: 2)
!4202 = !DILocation(line: 549, column: 26, scope: !4200)
!4203 = !DILocation(line: 549, column: 6, scope: !4201)
!4204 = !DILocation(line: 550, column: 25, scope: !4205)
!4205 = distinct !DILexicalBlock(scope: !4200, file: !3, line: 549, column: 42)
!4206 = !DILocation(line: 550, column: 4, scope: !4205)
!4207 = !DILocation(line: 551, column: 3, scope: !4205)
!4208 = !DILocation(line: 551, column: 12, scope: !4209)
!4209 = distinct !DILexicalBlock(scope: !4200, file: !3, line: 551, column: 12)
!4210 = !DILocation(line: 551, column: 32, scope: !4209)
!4211 = !DILocation(line: 551, column: 12, scope: !4200)
!4212 = !DILocation(line: 552, column: 35, scope: !4213)
!4213 = distinct !DILexicalBlock(scope: !4209, file: !3, line: 551, column: 58)
!4214 = !DILocation(line: 552, column: 4, scope: !4213)
!4215 = !DILocation(line: 553, column: 3, scope: !4213)
!4216 = !DILocation(line: 553, column: 12, scope: !4217)
!4217 = distinct !DILexicalBlock(scope: !4209, file: !3, line: 553, column: 12)
!4218 = !DILocation(line: 553, column: 32, scope: !4217)
!4219 = !DILocation(line: 553, column: 12, scope: !4209)
!4220 = !DILocation(line: 554, column: 4, scope: !4221)
!4221 = distinct !DILexicalBlock(scope: !4217, file: !3, line: 553, column: 47)
!4222 = !DILocation(line: 555, column: 3, scope: !4221)
!4223 = !DILocation(line: 556, column: 3, scope: !4157)
!4224 = !DILocation(line: 557, column: 13, scope: !4157)
!4225 = !DILocation(line: 557, column: 24, scope: !4157)
!4226 = !DILocation(line: 557, column: 2, scope: !4157)
!4227 = !DILocation(line: 586, column: 6, scope: !4228)
!4228 = distinct !DILexicalBlock(scope: !4229, file: !3, line: 586, column: 6)
!4229 = distinct !DILexicalBlock(scope: !4157, file: !3, line: 585, column: 2)
!4230 = !DILocation(line: 586, column: 26, scope: !4228)
!4231 = !DILocation(line: 586, column: 6, scope: !4229)
!4232 = !DILocation(line: 587, column: 25, scope: !4233)
!4233 = distinct !DILexicalBlock(scope: !4228, file: !3, line: 586, column: 42)
!4234 = !DILocation(line: 587, column: 4, scope: !4233)
!4235 = !DILocation(line: 588, column: 3, scope: !4233)
!4236 = !DILocation(line: 588, column: 12, scope: !4237)
!4237 = distinct !DILexicalBlock(scope: !4228, file: !3, line: 588, column: 12)
!4238 = !DILocation(line: 588, column: 32, scope: !4237)
!4239 = !DILocation(line: 588, column: 12, scope: !4228)
!4240 = !DILocation(line: 589, column: 35, scope: !4241)
!4241 = distinct !DILexicalBlock(scope: !4237, file: !3, line: 588, column: 58)
!4242 = !DILocation(line: 589, column: 4, scope: !4241)
!4243 = !DILocation(line: 590, column: 3, scope: !4241)
!4244 = !DILocation(line: 590, column: 12, scope: !4245)
!4245 = distinct !DILexicalBlock(scope: !4237, file: !3, line: 590, column: 12)
!4246 = !DILocation(line: 590, column: 32, scope: !4245)
!4247 = !DILocation(line: 590, column: 12, scope: !4237)
!4248 = !DILocation(line: 591, column: 4, scope: !4249)
!4249 = distinct !DILexicalBlock(scope: !4245, file: !3, line: 590, column: 47)
!4250 = !DILocation(line: 592, column: 3, scope: !4249)
!4251 = !DILocation(line: 593, column: 3, scope: !4157)
!4252 = !DILocation(line: 594, column: 13, scope: !4157)
!4253 = !DILocation(line: 594, column: 24, scope: !4157)
!4254 = !DILocation(line: 594, column: 2, scope: !4157)
!4255 = !DILocation(line: 595, column: 10, scope: !4256)
!4256 = distinct !DILexicalBlock(scope: !4157, file: !3, line: 595, column: 2)
!4257 = !DILocation(line: 595, column: 6, scope: !4256)
!4258 = !DILocation(line: 595, column: 14, scope: !4259)
!4259 = distinct !DILexicalBlock(scope: !4256, file: !3, line: 595, column: 2)
!4260 = !DILocation(line: 595, column: 20, scope: !4259)
!4261 = !DILocation(line: 595, column: 18, scope: !4259)
!4262 = !DILocation(line: 595, column: 2, scope: !4256)
!4263 = !DILocation(line: 596, column: 14, scope: !4264)
!4264 = distinct !DILexicalBlock(scope: !4259, file: !3, line: 595, column: 34)
!4265 = !DILocation(line: 596, column: 25, scope: !4264)
!4266 = !DILocation(line: 596, column: 36, scope: !4264)
!4267 = !DILocation(line: 596, column: 3, scope: !4264)
!4268 = !DILocation(line: 597, column: 15, scope: !4264)
!4269 = !DILocation(line: 597, column: 26, scope: !4264)
!4270 = !DILocation(line: 597, column: 3, scope: !4264)
!4271 = !DILocation(line: 598, column: 16, scope: !4264)
!4272 = !DILocation(line: 598, column: 22, scope: !4264)
!4273 = !DILocation(line: 598, column: 3, scope: !4264)
!4274 = !DILocation(line: 599, column: 2, scope: !4264)
!4275 = !DILocation(line: 595, column: 31, scope: !4259)
!4276 = !DILocation(line: 595, column: 2, scope: !4259)
!4277 = distinct !{!4277, !4262, !4278}
!4278 = !DILocation(line: 599, column: 2, scope: !4256)
!4279 = !DILocation(line: 601, column: 13, scope: !4157)
!4280 = !DILocation(line: 601, column: 19, scope: !4157)
!4281 = !DILocation(line: 601, column: 32, scope: !4157)
!4282 = !DILocation(line: 601, column: 2, scope: !4157)
!4283 = !DILocation(line: 602, column: 10, scope: !4284)
!4284 = distinct !DILexicalBlock(scope: !4157, file: !3, line: 602, column: 2)
!4285 = !DILocation(line: 602, column: 6, scope: !4284)
!4286 = !DILocation(line: 602, column: 14, scope: !4287)
!4287 = distinct !DILexicalBlock(scope: !4284, file: !3, line: 602, column: 2)
!4288 = !DILocation(line: 602, column: 20, scope: !4287)
!4289 = !DILocation(line: 602, column: 18, scope: !4287)
!4290 = !DILocation(line: 602, column: 2, scope: !4284)
!4291 = !DILocation(line: 603, column: 54, scope: !4292)
!4292 = distinct !DILexicalBlock(scope: !4287, file: !3, line: 602, column: 34)
!4293 = !DILocation(line: 603, column: 60, scope: !4292)
!4294 = !DILocation(line: 603, column: 65, scope: !4292)
!4295 = !DILocation(line: 603, column: 71, scope: !4292)
!4296 = !DILocation(line: 603, column: 77, scope: !4292)
!4297 = !DILocation(line: 603, column: 82, scope: !4292)
!4298 = !DILocation(line: 603, column: 88, scope: !4292)
!4299 = !DILocation(line: 603, column: 3, scope: !4292)
!4300 = !DILocation(line: 604, column: 2, scope: !4292)
!4301 = !DILocation(line: 602, column: 31, scope: !4287)
!4302 = !DILocation(line: 602, column: 2, scope: !4287)
!4303 = distinct !{!4303, !4290, !4304}
!4304 = !DILocation(line: 604, column: 2, scope: !4284)
!4305 = !DILocation(line: 606, column: 21, scope: !4157)
!4306 = !DILocation(line: 606, column: 2, scope: !4157)
!4307 = !DILocation(line: 610, column: 13, scope: !4157)
!4308 = !DILocation(line: 612, column: 5, scope: !4309)
!4309 = distinct !DILexicalBlock(scope: !4157, file: !3, line: 612, column: 5)
!4310 = !DILocation(line: 612, column: 16, scope: !4309)
!4311 = !DILocation(line: 612, column: 5, scope: !4157)
!4312 = !DILocation(line: 614, column: 25, scope: !4313)
!4313 = distinct !DILexicalBlock(scope: !4309, file: !3, line: 612, column: 23)
!4314 = !DILocation(line: 614, column: 23, scope: !4313)
!4315 = !DILocation(line: 614, column: 13, scope: !4313)
!4316 = !DILocation(line: 615, column: 28, scope: !4313)
!4317 = !DILocation(line: 615, column: 26, scope: !4313)
!4318 = !DILocation(line: 615, column: 16, scope: !4313)
!4319 = !DILocation(line: 615, column: 51, scope: !4313)
!4320 = !DILocation(line: 615, column: 50, scope: !4313)
!4321 = !DILocation(line: 615, column: 5, scope: !4313)
!4322 = !DILocation(line: 613, column: 40, scope: !4313)
!4323 = !DILocation(line: 616, column: 6, scope: !4313)
!4324 = !DILocation(line: 616, column: 4, scope: !4313)
!4325 = !DILocation(line: 613, column: 10, scope: !4313)
!4326 = !DILocation(line: 617, column: 2, scope: !4313)
!4327 = !DILocation(line: 618, column: 10, scope: !4328)
!4328 = distinct !DILexicalBlock(scope: !4309, file: !3, line: 617, column: 7)
!4329 = !DILocalVariable(name: "gpu_config", scope: !4157, file: !3, line: 621, type: !215)
!4330 = !DILocation(line: 621, column: 7, scope: !4157)
!4331 = !DILocalVariable(name: "gpu_config_string", scope: !4157, file: !3, line: 622, type: !4332)
!4332 = !DICompositeType(tag: DW_TAG_array_type, baseType: !109, size: 16384, elements: !4333)
!4333 = !{!4334}
!4334 = !DISubrange(count: 2048)
!4335 = !DILocation(line: 622, column: 7, scope: !4157)
!4336 = !DILocation(line: 655, column: 10, scope: !4157)
!4337 = !DILocation(line: 655, column: 2, scope: !4157)
!4338 = !DILocation(line: 656, column: 9, scope: !4157)
!4339 = !DILocation(line: 656, column: 28, scope: !4157)
!4340 = !DILocation(line: 656, column: 2, scope: !4157)
!4341 = !DILocation(line: 657, column: 10, scope: !4157)
!4342 = !DILocation(line: 657, column: 51, scope: !4157)
!4343 = !DILocation(line: 657, column: 2, scope: !4157)
!4344 = !DILocation(line: 658, column: 9, scope: !4157)
!4345 = !DILocation(line: 658, column: 28, scope: !4157)
!4346 = !DILocation(line: 658, column: 2, scope: !4157)
!4347 = !DILocation(line: 659, column: 10, scope: !4157)
!4348 = !DILocation(line: 659, column: 61, scope: !4157)
!4349 = !DILocation(line: 659, column: 2, scope: !4157)
!4350 = !DILocation(line: 660, column: 9, scope: !4157)
!4351 = !DILocation(line: 660, column: 28, scope: !4157)
!4352 = !DILocation(line: 660, column: 2, scope: !4157)
!4353 = !DILocation(line: 661, column: 10, scope: !4157)
!4354 = !DILocation(line: 661, column: 50, scope: !4157)
!4355 = !DILocation(line: 661, column: 2, scope: !4157)
!4356 = !DILocation(line: 662, column: 9, scope: !4157)
!4357 = !DILocation(line: 662, column: 28, scope: !4157)
!4358 = !DILocation(line: 662, column: 2, scope: !4157)
!4359 = !DILocation(line: 663, column: 10, scope: !4157)
!4360 = !DILocation(line: 663, column: 49, scope: !4157)
!4361 = !DILocation(line: 663, column: 2, scope: !4157)
!4362 = !DILocation(line: 664, column: 9, scope: !4157)
!4363 = !DILocation(line: 664, column: 28, scope: !4157)
!4364 = !DILocation(line: 664, column: 2, scope: !4157)
!4365 = !DILocation(line: 665, column: 10, scope: !4157)
!4366 = !DILocation(line: 665, column: 49, scope: !4157)
!4367 = !DILocation(line: 665, column: 2, scope: !4157)
!4368 = !DILocation(line: 666, column: 9, scope: !4157)
!4369 = !DILocation(line: 666, column: 28, scope: !4157)
!4370 = !DILocation(line: 666, column: 2, scope: !4157)
!4371 = !DILocation(line: 667, column: 10, scope: !4157)
!4372 = !DILocation(line: 667, column: 49, scope: !4157)
!4373 = !DILocation(line: 667, column: 2, scope: !4157)
!4374 = !DILocation(line: 668, column: 9, scope: !4157)
!4375 = !DILocation(line: 668, column: 28, scope: !4157)
!4376 = !DILocation(line: 668, column: 2, scope: !4157)
!4377 = !DILocation(line: 669, column: 10, scope: !4157)
!4378 = !DILocation(line: 669, column: 49, scope: !4157)
!4379 = !DILocation(line: 669, column: 2, scope: !4157)
!4380 = !DILocation(line: 670, column: 9, scope: !4157)
!4381 = !DILocation(line: 670, column: 28, scope: !4157)
!4382 = !DILocation(line: 670, column: 2, scope: !4157)
!4383 = !DILocation(line: 671, column: 10, scope: !4157)
!4384 = !DILocation(line: 671, column: 49, scope: !4157)
!4385 = !DILocation(line: 671, column: 2, scope: !4157)
!4386 = !DILocation(line: 672, column: 9, scope: !4157)
!4387 = !DILocation(line: 672, column: 28, scope: !4157)
!4388 = !DILocation(line: 672, column: 2, scope: !4157)
!4389 = !DILocation(line: 673, column: 10, scope: !4157)
!4390 = !DILocation(line: 673, column: 49, scope: !4157)
!4391 = !DILocation(line: 673, column: 2, scope: !4157)
!4392 = !DILocation(line: 674, column: 9, scope: !4157)
!4393 = !DILocation(line: 674, column: 28, scope: !4157)
!4394 = !DILocation(line: 674, column: 2, scope: !4157)
!4395 = !DILocation(line: 675, column: 10, scope: !4157)
!4396 = !DILocation(line: 675, column: 49, scope: !4157)
!4397 = !DILocation(line: 675, column: 2, scope: !4157)
!4398 = !DILocation(line: 676, column: 9, scope: !4157)
!4399 = !DILocation(line: 676, column: 28, scope: !4157)
!4400 = !DILocation(line: 676, column: 2, scope: !4157)
!4401 = !DILocation(line: 677, column: 10, scope: !4157)
!4402 = !DILocation(line: 677, column: 49, scope: !4157)
!4403 = !DILocation(line: 677, column: 2, scope: !4157)
!4404 = !DILocation(line: 678, column: 9, scope: !4157)
!4405 = !DILocation(line: 678, column: 28, scope: !4157)
!4406 = !DILocation(line: 678, column: 2, scope: !4157)
!4407 = !DILocation(line: 679, column: 10, scope: !4157)
!4408 = !DILocation(line: 679, column: 49, scope: !4157)
!4409 = !DILocation(line: 679, column: 2, scope: !4157)
!4410 = !DILocation(line: 680, column: 9, scope: !4157)
!4411 = !DILocation(line: 680, column: 28, scope: !4157)
!4412 = !DILocation(line: 680, column: 2, scope: !4157)
!4413 = !DILocation(line: 681, column: 10, scope: !4157)
!4414 = !DILocation(line: 681, column: 49, scope: !4157)
!4415 = !DILocation(line: 681, column: 2, scope: !4157)
!4416 = !DILocation(line: 682, column: 9, scope: !4157)
!4417 = !DILocation(line: 682, column: 28, scope: !4157)
!4418 = !DILocation(line: 682, column: 2, scope: !4157)
!4419 = !DILocation(line: 683, column: 10, scope: !4157)
!4420 = !DILocation(line: 683, column: 51, scope: !4157)
!4421 = !DILocation(line: 683, column: 2, scope: !4157)
!4422 = !DILocation(line: 684, column: 9, scope: !4157)
!4423 = !DILocation(line: 684, column: 28, scope: !4157)
!4424 = !DILocation(line: 684, column: 2, scope: !4157)
!4425 = !DILocation(line: 688, column: 4, scope: !4157)
!4426 = !DILocation(line: 692, column: 4, scope: !4157)
!4427 = !DILocation(line: 693, column: 4, scope: !4157)
!4428 = !DILocation(line: 694, column: 4, scope: !4157)
!4429 = !DILocation(line: 696, column: 4, scope: !4157)
!4430 = !DILocation(line: 703, column: 11, scope: !4157)
!4431 = !DILocation(line: 687, column: 2, scope: !4157)
!4432 = !DILocation(line: 712, column: 2, scope: !4157)
!4433 = !DILocation(line: 714, column: 7, scope: !4157)
!4434 = !DILocation(line: 714, column: 2, scope: !4157)
!4435 = !DILocation(line: 715, column: 7, scope: !4157)
!4436 = !DILocation(line: 715, column: 2, scope: !4157)
!4437 = !DILocation(line: 716, column: 7, scope: !4157)
!4438 = !DILocation(line: 716, column: 2, scope: !4157)
!4439 = !DILocation(line: 717, column: 7, scope: !4157)
!4440 = !DILocation(line: 717, column: 2, scope: !4157)
!4441 = !DILocation(line: 718, column: 7, scope: !4157)
!4442 = !DILocation(line: 718, column: 2, scope: !4157)
!4443 = !DILocation(line: 719, column: 7, scope: !4157)
!4444 = !DILocation(line: 719, column: 2, scope: !4157)
!4445 = !DILocation(line: 722, column: 2, scope: !4157)
!4446 = distinct !DISubprogram(name: "setup", linkageName: "_ZL5setupv", scope: !3, file: !3, line: 1659, type: !561, scopeLine: 1659, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1051)
!4447 = !DILocation(line: 1660, column: 8, scope: !4446)
!4448 = !DILocation(line: 1662, column: 2, scope: !4446)
!4449 = !DILocation(line: 1663, column: 2, scope: !4446)
!4450 = !DILocation(line: 1664, column: 48, scope: !4446)
!4451 = !DILocation(line: 1664, column: 2, scope: !4446)
!4452 = !DILocation(line: 1665, column: 2, scope: !4446)
!4453 = !DILocation(line: 1666, column: 1, scope: !4446)
!4454 = distinct !DISubprogram(name: "setup_gpu", linkageName: "_ZL9setup_gpuv", scope: !3, file: !3, line: 1668, type: !561, scopeLine: 1668, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1051)
!4455 = !DILocation(line: 1713, column: 33, scope: !4454)
!4456 = !DILocation(line: 1714, column: 43, scope: !4454)
!4457 = !DILocation(line: 1717, column: 69, scope: !4458)
!4458 = distinct !DILexicalBlock(scope: !4454, file: !3, line: 1716, column: 5)
!4459 = !DILocation(line: 1717, column: 45, scope: !4458)
!4460 = !DILocation(line: 1716, column: 5, scope: !4454)
!4461 = !DILocation(line: 1718, column: 41, scope: !4462)
!4462 = distinct !DILexicalBlock(scope: !4458, file: !3, line: 1717, column: 89)
!4463 = !DILocation(line: 1719, column: 2, scope: !4462)
!4464 = !DILocation(line: 1720, column: 65, scope: !4465)
!4465 = distinct !DILexicalBlock(scope: !4458, file: !3, line: 1719, column: 7)
!4466 = !DILocation(line: 1720, column: 41, scope: !4465)
!4467 = !DILocation(line: 1723, column: 79, scope: !4468)
!4468 = distinct !DILexicalBlock(scope: !4454, file: !3, line: 1722, column: 5)
!4469 = !DILocation(line: 1723, column: 55, scope: !4468)
!4470 = !DILocation(line: 1722, column: 5, scope: !4454)
!4471 = !DILocation(line: 1724, column: 51, scope: !4472)
!4472 = distinct !DILexicalBlock(scope: !4468, file: !3, line: 1723, column: 99)
!4473 = !DILocation(line: 1725, column: 2, scope: !4472)
!4474 = !DILocation(line: 1726, column: 75, scope: !4475)
!4475 = distinct !DILexicalBlock(scope: !4468, file: !3, line: 1725, column: 7)
!4476 = !DILocation(line: 1726, column: 51, scope: !4475)
!4477 = !DILocation(line: 1729, column: 60, scope: !4478)
!4478 = distinct !DILexicalBlock(scope: !4454, file: !3, line: 1728, column: 5)
!4479 = !DILocation(line: 1729, column: 36, scope: !4478)
!4480 = !DILocation(line: 1728, column: 5, scope: !4454)
!4481 = !DILocation(line: 1730, column: 32, scope: !4482)
!4482 = distinct !DILexicalBlock(scope: !4478, file: !3, line: 1729, column: 80)
!4483 = !DILocation(line: 1731, column: 2, scope: !4482)
!4484 = !DILocation(line: 1732, column: 54, scope: !4485)
!4485 = distinct !DILexicalBlock(scope: !4478, file: !3, line: 1731, column: 7)
!4486 = !DILocation(line: 1732, column: 31, scope: !4485)
!4487 = !DILocation(line: 1735, column: 59, scope: !4488)
!4488 = distinct !DILexicalBlock(scope: !4454, file: !3, line: 1734, column: 5)
!4489 = !DILocation(line: 1735, column: 35, scope: !4488)
!4490 = !DILocation(line: 1734, column: 5, scope: !4454)
!4491 = !DILocation(line: 1736, column: 31, scope: !4492)
!4492 = distinct !DILexicalBlock(scope: !4488, file: !3, line: 1735, column: 79)
!4493 = !DILocation(line: 1737, column: 2, scope: !4492)
!4494 = !DILocation(line: 1738, column: 53, scope: !4495)
!4495 = distinct !DILexicalBlock(scope: !4488, file: !3, line: 1737, column: 7)
!4496 = !DILocation(line: 1738, column: 30, scope: !4495)
!4497 = !DILocation(line: 1741, column: 59, scope: !4498)
!4498 = distinct !DILexicalBlock(scope: !4454, file: !3, line: 1740, column: 5)
!4499 = !DILocation(line: 1741, column: 35, scope: !4498)
!4500 = !DILocation(line: 1740, column: 5, scope: !4454)
!4501 = !DILocation(line: 1742, column: 31, scope: !4502)
!4502 = distinct !DILexicalBlock(scope: !4498, file: !3, line: 1741, column: 79)
!4503 = !DILocation(line: 1743, column: 2, scope: !4502)
!4504 = !DILocation(line: 1744, column: 55, scope: !4505)
!4505 = distinct !DILexicalBlock(scope: !4498, file: !3, line: 1743, column: 7)
!4506 = !DILocation(line: 1744, column: 31, scope: !4505)
!4507 = !DILocation(line: 1747, column: 59, scope: !4508)
!4508 = distinct !DILexicalBlock(scope: !4454, file: !3, line: 1746, column: 5)
!4509 = !DILocation(line: 1747, column: 35, scope: !4508)
!4510 = !DILocation(line: 1746, column: 5, scope: !4454)
!4511 = !DILocation(line: 1748, column: 31, scope: !4512)
!4512 = distinct !DILexicalBlock(scope: !4508, file: !3, line: 1747, column: 79)
!4513 = !DILocation(line: 1749, column: 2, scope: !4512)
!4514 = !DILocation(line: 1750, column: 55, scope: !4515)
!4515 = distinct !DILexicalBlock(scope: !4508, file: !3, line: 1749, column: 7)
!4516 = !DILocation(line: 1750, column: 31, scope: !4515)
!4517 = !DILocation(line: 1753, column: 59, scope: !4518)
!4518 = distinct !DILexicalBlock(scope: !4454, file: !3, line: 1752, column: 5)
!4519 = !DILocation(line: 1753, column: 35, scope: !4518)
!4520 = !DILocation(line: 1752, column: 5, scope: !4454)
!4521 = !DILocation(line: 1754, column: 31, scope: !4522)
!4522 = distinct !DILexicalBlock(scope: !4518, file: !3, line: 1753, column: 79)
!4523 = !DILocation(line: 1755, column: 2, scope: !4522)
!4524 = !DILocation(line: 1756, column: 55, scope: !4525)
!4525 = distinct !DILexicalBlock(scope: !4518, file: !3, line: 1755, column: 7)
!4526 = !DILocation(line: 1756, column: 31, scope: !4525)
!4527 = !DILocation(line: 1759, column: 59, scope: !4528)
!4528 = distinct !DILexicalBlock(scope: !4454, file: !3, line: 1758, column: 5)
!4529 = !DILocation(line: 1759, column: 35, scope: !4528)
!4530 = !DILocation(line: 1758, column: 5, scope: !4454)
!4531 = !DILocation(line: 1760, column: 31, scope: !4532)
!4532 = distinct !DILexicalBlock(scope: !4528, file: !3, line: 1759, column: 79)
!4533 = !DILocation(line: 1761, column: 2, scope: !4532)
!4534 = !DILocation(line: 1762, column: 55, scope: !4535)
!4535 = distinct !DILexicalBlock(scope: !4528, file: !3, line: 1761, column: 7)
!4536 = !DILocation(line: 1762, column: 31, scope: !4535)
!4537 = !DILocation(line: 1765, column: 59, scope: !4538)
!4538 = distinct !DILexicalBlock(scope: !4454, file: !3, line: 1764, column: 5)
!4539 = !DILocation(line: 1765, column: 35, scope: !4538)
!4540 = !DILocation(line: 1764, column: 5, scope: !4454)
!4541 = !DILocation(line: 1766, column: 31, scope: !4542)
!4542 = distinct !DILexicalBlock(scope: !4538, file: !3, line: 1765, column: 79)
!4543 = !DILocation(line: 1767, column: 2, scope: !4542)
!4544 = !DILocation(line: 1768, column: 55, scope: !4545)
!4545 = distinct !DILexicalBlock(scope: !4538, file: !3, line: 1767, column: 7)
!4546 = !DILocation(line: 1768, column: 31, scope: !4545)
!4547 = !DILocation(line: 1771, column: 59, scope: !4548)
!4548 = distinct !DILexicalBlock(scope: !4454, file: !3, line: 1770, column: 5)
!4549 = !DILocation(line: 1771, column: 35, scope: !4548)
!4550 = !DILocation(line: 1770, column: 5, scope: !4454)
!4551 = !DILocation(line: 1772, column: 31, scope: !4552)
!4552 = distinct !DILexicalBlock(scope: !4548, file: !3, line: 1771, column: 79)
!4553 = !DILocation(line: 1773, column: 2, scope: !4552)
!4554 = !DILocation(line: 1774, column: 55, scope: !4555)
!4555 = distinct !DILexicalBlock(scope: !4548, file: !3, line: 1773, column: 7)
!4556 = !DILocation(line: 1774, column: 31, scope: !4555)
!4557 = !DILocation(line: 1777, column: 59, scope: !4558)
!4558 = distinct !DILexicalBlock(scope: !4454, file: !3, line: 1776, column: 5)
!4559 = !DILocation(line: 1777, column: 35, scope: !4558)
!4560 = !DILocation(line: 1776, column: 5, scope: !4454)
!4561 = !DILocation(line: 1778, column: 31, scope: !4562)
!4562 = distinct !DILexicalBlock(scope: !4558, file: !3, line: 1777, column: 79)
!4563 = !DILocation(line: 1779, column: 2, scope: !4562)
!4564 = !DILocation(line: 1780, column: 55, scope: !4565)
!4565 = distinct !DILexicalBlock(scope: !4558, file: !3, line: 1779, column: 7)
!4566 = !DILocation(line: 1780, column: 31, scope: !4565)
!4567 = !DILocation(line: 1783, column: 59, scope: !4568)
!4568 = distinct !DILexicalBlock(scope: !4454, file: !3, line: 1782, column: 5)
!4569 = !DILocation(line: 1783, column: 35, scope: !4568)
!4570 = !DILocation(line: 1782, column: 5, scope: !4454)
!4571 = !DILocation(line: 1784, column: 31, scope: !4572)
!4572 = distinct !DILexicalBlock(scope: !4568, file: !3, line: 1783, column: 79)
!4573 = !DILocation(line: 1785, column: 2, scope: !4572)
!4574 = !DILocation(line: 1786, column: 55, scope: !4575)
!4575 = distinct !DILexicalBlock(scope: !4568, file: !3, line: 1785, column: 7)
!4576 = !DILocation(line: 1786, column: 31, scope: !4575)
!4577 = !DILocation(line: 1789, column: 59, scope: !4578)
!4578 = distinct !DILexicalBlock(scope: !4454, file: !3, line: 1788, column: 5)
!4579 = !DILocation(line: 1789, column: 35, scope: !4578)
!4580 = !DILocation(line: 1788, column: 5, scope: !4454)
!4581 = !DILocation(line: 1790, column: 31, scope: !4582)
!4582 = distinct !DILexicalBlock(scope: !4578, file: !3, line: 1789, column: 79)
!4583 = !DILocation(line: 1791, column: 2, scope: !4582)
!4584 = !DILocation(line: 1792, column: 55, scope: !4585)
!4585 = distinct !DILexicalBlock(scope: !4578, file: !3, line: 1791, column: 7)
!4586 = !DILocation(line: 1792, column: 31, scope: !4585)
!4587 = !DILocation(line: 1795, column: 61, scope: !4588)
!4588 = distinct !DILexicalBlock(scope: !4454, file: !3, line: 1794, column: 5)
!4589 = !DILocation(line: 1795, column: 37, scope: !4588)
!4590 = !DILocation(line: 1794, column: 5, scope: !4454)
!4591 = !DILocation(line: 1796, column: 33, scope: !4592)
!4592 = distinct !DILexicalBlock(scope: !4588, file: !3, line: 1795, column: 81)
!4593 = !DILocation(line: 1797, column: 2, scope: !4592)
!4594 = !DILocation(line: 1798, column: 57, scope: !4595)
!4595 = distinct !DILexicalBlock(scope: !4588, file: !3, line: 1797, column: 7)
!4596 = !DILocation(line: 1798, column: 33, scope: !4595)
!4597 = !DILocation(line: 1801, column: 65, scope: !4454)
!4598 = !DILocation(line: 1801, column: 57, scope: !4454)
!4599 = !DILocation(line: 1801, column: 38, scope: !4454)
!4600 = !DILocation(line: 1801, column: 37, scope: !4454)
!4601 = !DILocation(line: 1802, column: 71, scope: !4454)
!4602 = !DILocation(line: 1802, column: 63, scope: !4454)
!4603 = !DILocation(line: 1802, column: 48, scope: !4454)
!4604 = !DILocation(line: 1802, column: 47, scope: !4454)
!4605 = !DILocation(line: 1803, column: 56, scope: !4454)
!4606 = !DILocation(line: 1803, column: 48, scope: !4454)
!4607 = !DILocation(line: 1803, column: 29, scope: !4454)
!4608 = !DILocation(line: 1803, column: 28, scope: !4454)
!4609 = !DILocation(line: 1804, column: 55, scope: !4454)
!4610 = !DILocation(line: 1804, column: 47, scope: !4454)
!4611 = !DILocation(line: 1804, column: 28, scope: !4454)
!4612 = !DILocation(line: 1804, column: 27, scope: !4454)
!4613 = !DILocation(line: 1805, column: 57, scope: !4454)
!4614 = !DILocation(line: 1805, column: 49, scope: !4454)
!4615 = !DILocation(line: 1805, column: 28, scope: !4454)
!4616 = !DILocation(line: 1805, column: 27, scope: !4454)
!4617 = !DILocation(line: 1806, column: 54, scope: !4454)
!4618 = !DILocation(line: 1806, column: 46, scope: !4454)
!4619 = !DILocation(line: 1806, column: 28, scope: !4454)
!4620 = !DILocation(line: 1806, column: 27, scope: !4454)
!4621 = !DILocation(line: 1807, column: 57, scope: !4454)
!4622 = !DILocation(line: 1807, column: 49, scope: !4454)
!4623 = !DILocation(line: 1807, column: 28, scope: !4454)
!4624 = !DILocation(line: 1807, column: 27, scope: !4454)
!4625 = !DILocation(line: 1808, column: 57, scope: !4454)
!4626 = !DILocation(line: 1808, column: 49, scope: !4454)
!4627 = !DILocation(line: 1808, column: 28, scope: !4454)
!4628 = !DILocation(line: 1808, column: 27, scope: !4454)
!4629 = !DILocation(line: 1809, column: 54, scope: !4454)
!4630 = !DILocation(line: 1809, column: 46, scope: !4454)
!4631 = !DILocation(line: 1809, column: 28, scope: !4454)
!4632 = !DILocation(line: 1809, column: 27, scope: !4454)
!4633 = !DILocation(line: 1810, column: 57, scope: !4454)
!4634 = !DILocation(line: 1810, column: 49, scope: !4454)
!4635 = !DILocation(line: 1810, column: 28, scope: !4454)
!4636 = !DILocation(line: 1810, column: 27, scope: !4454)
!4637 = !DILocation(line: 1811, column: 57, scope: !4454)
!4638 = !DILocation(line: 1811, column: 49, scope: !4454)
!4639 = !DILocation(line: 1811, column: 28, scope: !4454)
!4640 = !DILocation(line: 1811, column: 27, scope: !4454)
!4641 = !DILocation(line: 1812, column: 54, scope: !4454)
!4642 = !DILocation(line: 1812, column: 46, scope: !4454)
!4643 = !DILocation(line: 1812, column: 28, scope: !4454)
!4644 = !DILocation(line: 1812, column: 27, scope: !4454)
!4645 = !DILocation(line: 1813, column: 57, scope: !4454)
!4646 = !DILocation(line: 1813, column: 49, scope: !4454)
!4647 = !DILocation(line: 1813, column: 28, scope: !4454)
!4648 = !DILocation(line: 1813, column: 27, scope: !4454)
!4649 = !DILocation(line: 1814, column: 65, scope: !4454)
!4650 = !DILocation(line: 1814, column: 57, scope: !4454)
!4651 = !DILocation(line: 1814, column: 30, scope: !4454)
!4652 = !DILocation(line: 1814, column: 29, scope: !4454)
!4653 = !DILocation(line: 1816, column: 18, scope: !4454)
!4654 = !DILocation(line: 1817, column: 20, scope: !4454)
!4655 = !DILocation(line: 1818, column: 21, scope: !4454)
!4656 = !DILocation(line: 1819, column: 15, scope: !4454)
!4657 = !DILocation(line: 1820, column: 16, scope: !4454)
!4658 = !DILocation(line: 1821, column: 16, scope: !4454)
!4659 = !DILocation(line: 1822, column: 16, scope: !4454)
!4660 = !DILocation(line: 1823, column: 16, scope: !4454)
!4661 = !DILocation(line: 1824, column: 19, scope: !4454)
!4662 = !DILocation(line: 1824, column: 48, scope: !4454)
!4663 = !DILocation(line: 1824, column: 18, scope: !4454)
!4664 = !DILocation(line: 1826, column: 27, scope: !4454)
!4665 = !DILocation(line: 1826, column: 2, scope: !4454)
!4666 = !DILocation(line: 1827, column: 29, scope: !4454)
!4667 = !DILocation(line: 1827, column: 2, scope: !4454)
!4668 = !DILocation(line: 1828, column: 30, scope: !4454)
!4669 = !DILocation(line: 1828, column: 2, scope: !4454)
!4670 = !DILocation(line: 1829, column: 24, scope: !4454)
!4671 = !DILocation(line: 1829, column: 2, scope: !4454)
!4672 = !DILocation(line: 1830, column: 25, scope: !4454)
!4673 = !DILocation(line: 1830, column: 2, scope: !4454)
!4674 = !DILocation(line: 1831, column: 25, scope: !4454)
!4675 = !DILocation(line: 1831, column: 2, scope: !4454)
!4676 = !DILocation(line: 1832, column: 25, scope: !4454)
!4677 = !DILocation(line: 1832, column: 2, scope: !4454)
!4678 = !DILocation(line: 1833, column: 25, scope: !4454)
!4679 = !DILocation(line: 1833, column: 2, scope: !4454)
!4680 = !DILocation(line: 1835, column: 2, scope: !4454)
!4681 = !DILocation(line: 1836, column: 1, scope: !4454)
!4682 = distinct !DISubprogram(name: "init_ui_gpu", linkageName: "_ZL11init_ui_gpuP8dcomplexS0_Pd", scope: !3, file: !3, line: 1538, type: !3642, scopeLine: 1540, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1051)
!4683 = !DILocalVariable(name: "u0", arg: 1, scope: !4682, file: !3, line: 1538, type: !98)
!4684 = !DILocation(line: 1538, column: 34, scope: !4682)
!4685 = !DILocalVariable(name: "u1", arg: 2, scope: !4682, file: !3, line: 1539, type: !98)
!4686 = !DILocation(line: 1539, column: 12, scope: !4682)
!4687 = !DILocalVariable(name: "twiddle", arg: 3, scope: !4682, file: !3, line: 1540, type: !106)
!4688 = !DILocation(line: 1540, column: 10, scope: !4682)
!4689 = !DILocation(line: 1544, column: 23, scope: !4682)
!4690 = !DILocation(line: 1545, column: 3, scope: !4682)
!4691 = !DILocation(line: 1544, column: 20, scope: !4682)
!4692 = !DILocation(line: 1544, column: 2, scope: !4682)
!4693 = !DILocation(line: 1545, column: 35, scope: !4682)
!4694 = !DILocation(line: 1546, column: 5, scope: !4682)
!4695 = !DILocation(line: 1547, column: 5, scope: !4682)
!4696 = !DILocation(line: 1552, column: 1, scope: !4682)
!4697 = distinct !DISubprogram(name: "compute_indexmap_gpu", linkageName: "_ZL20compute_indexmap_gpuPd", scope: !3, file: !3, line: 1354, type: !3387, scopeLine: 1354, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1051)
!4698 = !DILocalVariable(name: "twiddle", arg: 1, scope: !4697, file: !3, line: 1354, type: !106)
!4699 = !DILocation(line: 1354, column: 41, scope: !4697)
!4700 = !DILocation(line: 1358, column: 32, scope: !4697)
!4701 = !DILocation(line: 1359, column: 3, scope: !4697)
!4702 = !DILocation(line: 1358, column: 29, scope: !4697)
!4703 = !DILocation(line: 1358, column: 2, scope: !4697)
!4704 = !DILocation(line: 1359, column: 44, scope: !4697)
!4705 = !DILocation(line: 1363, column: 1, scope: !4697)
!4706 = distinct !DISubprogram(name: "compute_initial_conditions_gpu", linkageName: "_ZL30compute_initial_conditions_gpuP8dcomplex", scope: !3, file: !3, line: 1387, type: !4707, scopeLine: 1387, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1051)
!4707 = !DISubroutineType(types: !4708)
!4708 = !{null, !98}
!4709 = !DILocalVariable(name: "u0", arg: 1, scope: !4706, file: !3, line: 1387, type: !98)
!4710 = !DILocation(line: 1387, column: 53, scope: !4706)
!4711 = !DILocalVariable(name: "z", scope: !4706, file: !3, line: 1391, type: !97)
!4712 = !DILocation(line: 1391, column: 6, scope: !4706)
!4713 = !DILocalVariable(name: "start", scope: !4706, file: !3, line: 1392, type: !104)
!4714 = !DILocation(line: 1392, column: 9, scope: !4706)
!4715 = !DILocalVariable(name: "an", scope: !4706, file: !3, line: 1392, type: !104)
!4716 = !DILocation(line: 1392, column: 16, scope: !4706)
!4717 = !DILocalVariable(name: "starts", scope: !4706, file: !3, line: 1392, type: !4718)
!4718 = !DICompositeType(tag: DW_TAG_array_type, baseType: !104, size: 8192, elements: !4719)
!4719 = !{!4720}
!4720 = !DISubrange(count: 128)
!4721 = !DILocation(line: 1392, column: 20, scope: !4706)
!4722 = !DILocation(line: 1394, column: 8, scope: !4706)
!4723 = !DILocation(line: 1396, column: 2, scope: !4706)
!4724 = !DILocation(line: 1397, column: 17, scope: !4706)
!4725 = !DILocation(line: 1397, column: 2, scope: !4706)
!4726 = !DILocation(line: 1398, column: 2, scope: !4706)
!4727 = !DILocation(line: 1400, column: 14, scope: !4706)
!4728 = !DILocation(line: 1400, column: 2, scope: !4706)
!4729 = !DILocation(line: 1400, column: 12, scope: !4706)
!4730 = !DILocation(line: 1401, column: 7, scope: !4731)
!4731 = distinct !DILexicalBlock(scope: !4706, file: !3, line: 1401, column: 2)
!4732 = !DILocation(line: 1401, column: 6, scope: !4731)
!4733 = !DILocation(line: 1401, column: 11, scope: !4734)
!4734 = distinct !DILexicalBlock(scope: !4731, file: !3, line: 1401, column: 2)
!4735 = !DILocation(line: 1401, column: 12, scope: !4734)
!4736 = !DILocation(line: 1401, column: 2, scope: !4731)
!4737 = !DILocation(line: 1402, column: 18, scope: !4738)
!4738 = distinct !DILexicalBlock(scope: !4734, file: !3, line: 1401, column: 21)
!4739 = !DILocation(line: 1402, column: 3, scope: !4738)
!4740 = !DILocation(line: 1403, column: 15, scope: !4738)
!4741 = !DILocation(line: 1403, column: 10, scope: !4738)
!4742 = !DILocation(line: 1403, column: 3, scope: !4738)
!4743 = !DILocation(line: 1403, column: 13, scope: !4738)
!4744 = !DILocation(line: 1404, column: 2, scope: !4738)
!4745 = !DILocation(line: 1401, column: 18, scope: !4734)
!4746 = !DILocation(line: 1401, column: 2, scope: !4734)
!4747 = distinct !{!4747, !4736, !4748}
!4748 = !DILocation(line: 1404, column: 2, scope: !4731)
!4749 = !DILocation(line: 1406, column: 13, scope: !4706)
!4750 = !DILocation(line: 1406, column: 28, scope: !4706)
!4751 = !DILocation(line: 1406, column: 36, scope: !4706)
!4752 = !DILocation(line: 1406, column: 2, scope: !4706)
!4753 = !DILocation(line: 1408, column: 42, scope: !4706)
!4754 = !DILocation(line: 1409, column: 3, scope: !4706)
!4755 = !DILocation(line: 1408, column: 39, scope: !4706)
!4756 = !DILocation(line: 1408, column: 2, scope: !4706)
!4757 = !DILocation(line: 1409, column: 54, scope: !4706)
!4758 = !DILocation(line: 1410, column: 5, scope: !4706)
!4759 = !DILocation(line: 1414, column: 1, scope: !4706)
!4760 = distinct !DISubprogram(name: "fft_init_gpu", linkageName: "_ZL12fft_init_gpui", scope: !3, file: !3, line: 1479, type: !598, scopeLine: 1479, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1051)
!4761 = !DILocalVariable(name: "n", arg: 1, scope: !4760, file: !3, line: 1479, type: !97)
!4762 = !DILocation(line: 1479, column: 30, scope: !4760)
!4763 = !DILocalVariable(name: "m", scope: !4760, file: !3, line: 1483, type: !97)
!4764 = !DILocation(line: 1483, column: 6, scope: !4760)
!4765 = !DILocalVariable(name: "ku", scope: !4760, file: !3, line: 1483, type: !97)
!4766 = !DILocation(line: 1483, column: 8, scope: !4760)
!4767 = !DILocalVariable(name: "i", scope: !4760, file: !3, line: 1483, type: !97)
!4768 = !DILocation(line: 1483, column: 11, scope: !4760)
!4769 = !DILocalVariable(name: "j", scope: !4760, file: !3, line: 1483, type: !97)
!4770 = !DILocation(line: 1483, column: 13, scope: !4760)
!4771 = !DILocalVariable(name: "ln", scope: !4760, file: !3, line: 1483, type: !97)
!4772 = !DILocation(line: 1483, column: 15, scope: !4760)
!4773 = !DILocalVariable(name: "t", scope: !4760, file: !3, line: 1484, type: !104)
!4774 = !DILocation(line: 1484, column: 9, scope: !4760)
!4775 = !DILocalVariable(name: "ti", scope: !4760, file: !3, line: 1484, type: !104)
!4776 = !DILocation(line: 1484, column: 12, scope: !4760)
!4777 = !DILocation(line: 1491, column: 12, scope: !4760)
!4778 = !DILocation(line: 1491, column: 6, scope: !4760)
!4779 = !DILocation(line: 1491, column: 4, scope: !4760)
!4780 = !DILocation(line: 1492, column: 9, scope: !4760)
!4781 = !DILocation(line: 1492, column: 2, scope: !4760)
!4782 = !DILocation(line: 1492, column: 7, scope: !4760)
!4783 = !DILocation(line: 1493, column: 5, scope: !4760)
!4784 = !DILocation(line: 1494, column: 5, scope: !4760)
!4785 = !DILocation(line: 1495, column: 7, scope: !4786)
!4786 = distinct !DILexicalBlock(scope: !4760, file: !3, line: 1495, column: 2)
!4787 = !DILocation(line: 1495, column: 6, scope: !4786)
!4788 = !DILocation(line: 1495, column: 11, scope: !4789)
!4789 = distinct !DILexicalBlock(scope: !4786, file: !3, line: 1495, column: 2)
!4790 = !DILocation(line: 1495, column: 14, scope: !4789)
!4791 = !DILocation(line: 1495, column: 12, scope: !4789)
!4792 = !DILocation(line: 1495, column: 2, scope: !4786)
!4793 = !DILocation(line: 1496, column: 12, scope: !4794)
!4794 = distinct !DILexicalBlock(scope: !4789, file: !3, line: 1495, column: 21)
!4795 = !DILocation(line: 1496, column: 10, scope: !4794)
!4796 = !DILocation(line: 1496, column: 5, scope: !4794)
!4797 = !DILocation(line: 1497, column: 8, scope: !4798)
!4798 = distinct !DILexicalBlock(scope: !4794, file: !3, line: 1497, column: 3)
!4799 = !DILocation(line: 1497, column: 7, scope: !4798)
!4800 = !DILocation(line: 1497, column: 12, scope: !4801)
!4801 = distinct !DILexicalBlock(scope: !4798, file: !3, line: 1497, column: 3)
!4802 = !DILocation(line: 1497, column: 15, scope: !4801)
!4803 = !DILocation(line: 1497, column: 17, scope: !4801)
!4804 = !DILocation(line: 1497, column: 13, scope: !4801)
!4805 = !DILocation(line: 1497, column: 3, scope: !4798)
!4806 = !DILocation(line: 1498, column: 9, scope: !4807)
!4807 = distinct !DILexicalBlock(scope: !4801, file: !3, line: 1497, column: 25)
!4808 = !DILocation(line: 1498, column: 13, scope: !4807)
!4809 = !DILocation(line: 1498, column: 11, scope: !4807)
!4810 = !DILocation(line: 1498, column: 7, scope: !4807)
!4811 = !DILocation(line: 1499, column: 16, scope: !4807)
!4812 = !DILocation(line: 1499, column: 4, scope: !4807)
!4813 = !DILocation(line: 1499, column: 6, scope: !4807)
!4814 = !DILocation(line: 1499, column: 8, scope: !4807)
!4815 = !DILocation(line: 1499, column: 7, scope: !4807)
!4816 = !DILocation(line: 1499, column: 10, scope: !4807)
!4817 = !DILocation(line: 1499, column: 14, scope: !4807)
!4818 = !DILocation(line: 1500, column: 3, scope: !4807)
!4819 = !DILocation(line: 1497, column: 22, scope: !4801)
!4820 = !DILocation(line: 1497, column: 3, scope: !4801)
!4821 = distinct !{!4821, !4805, !4822}
!4822 = !DILocation(line: 1500, column: 3, scope: !4798)
!4823 = !DILocation(line: 1501, column: 8, scope: !4794)
!4824 = !DILocation(line: 1501, column: 13, scope: !4794)
!4825 = !DILocation(line: 1501, column: 11, scope: !4794)
!4826 = !DILocation(line: 1501, column: 6, scope: !4794)
!4827 = !DILocation(line: 1502, column: 12, scope: !4794)
!4828 = !DILocation(line: 1502, column: 10, scope: !4794)
!4829 = !DILocation(line: 1502, column: 6, scope: !4794)
!4830 = !DILocation(line: 1503, column: 2, scope: !4794)
!4831 = !DILocation(line: 1495, column: 18, scope: !4789)
!4832 = !DILocation(line: 1495, column: 2, scope: !4789)
!4833 = distinct !{!4833, !4792, !4834}
!4834 = !DILocation(line: 1503, column: 2, scope: !4786)
!4835 = !DILocation(line: 1504, column: 13, scope: !4760)
!4836 = !DILocation(line: 1504, column: 23, scope: !4760)
!4837 = !DILocation(line: 1504, column: 26, scope: !4760)
!4838 = !DILocation(line: 1504, column: 2, scope: !4760)
!4839 = !DILocation(line: 1508, column: 1, scope: !4760)
!4840 = distinct !DISubprogram(name: "fft_gpu", linkageName: "_ZL7fft_gpuiP8dcomplexS0_", scope: !3, file: !3, line: 1457, type: !3154, scopeLine: 1459, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1051)
!4841 = !DILocalVariable(name: "dir", arg: 1, scope: !4840, file: !3, line: 1457, type: !97)
!4842 = !DILocation(line: 1457, column: 25, scope: !4840)
!4843 = !DILocalVariable(name: "x1", arg: 2, scope: !4840, file: !3, line: 1458, type: !98)
!4844 = !DILocation(line: 1458, column: 12, scope: !4840)
!4845 = !DILocalVariable(name: "x2", arg: 3, scope: !4840, file: !3, line: 1459, type: !98)
!4846 = !DILocation(line: 1459, column: 12, scope: !4840)
!4847 = !DILocation(line: 1468, column: 5, scope: !4848)
!4848 = distinct !DILexicalBlock(scope: !4840, file: !3, line: 1468, column: 5)
!4849 = !DILocation(line: 1468, column: 8, scope: !4848)
!4850 = !DILocation(line: 1468, column: 5, scope: !4840)
!4851 = !DILocation(line: 1469, column: 17, scope: !4852)
!4852 = distinct !DILexicalBlock(scope: !4848, file: !3, line: 1468, column: 12)
!4853 = !DILocation(line: 1469, column: 27, scope: !4852)
!4854 = !DILocation(line: 1469, column: 31, scope: !4852)
!4855 = !DILocation(line: 1469, column: 35, scope: !4852)
!4856 = !DILocation(line: 1469, column: 46, scope: !4852)
!4857 = !DILocation(line: 1469, column: 3, scope: !4852)
!4858 = !DILocation(line: 1470, column: 17, scope: !4852)
!4859 = !DILocation(line: 1470, column: 27, scope: !4852)
!4860 = !DILocation(line: 1470, column: 31, scope: !4852)
!4861 = !DILocation(line: 1470, column: 35, scope: !4852)
!4862 = !DILocation(line: 1470, column: 46, scope: !4852)
!4863 = !DILocation(line: 1470, column: 3, scope: !4852)
!4864 = !DILocation(line: 1471, column: 17, scope: !4852)
!4865 = !DILocation(line: 1471, column: 27, scope: !4852)
!4866 = !DILocation(line: 1471, column: 31, scope: !4852)
!4867 = !DILocation(line: 1471, column: 35, scope: !4852)
!4868 = !DILocation(line: 1471, column: 46, scope: !4852)
!4869 = !DILocation(line: 1471, column: 3, scope: !4852)
!4870 = !DILocation(line: 1472, column: 2, scope: !4852)
!4871 = !DILocation(line: 1473, column: 18, scope: !4872)
!4872 = distinct !DILexicalBlock(scope: !4848, file: !3, line: 1472, column: 7)
!4873 = !DILocation(line: 1473, column: 28, scope: !4872)
!4874 = !DILocation(line: 1473, column: 32, scope: !4872)
!4875 = !DILocation(line: 1473, column: 36, scope: !4872)
!4876 = !DILocation(line: 1473, column: 47, scope: !4872)
!4877 = !DILocation(line: 1473, column: 3, scope: !4872)
!4878 = !DILocation(line: 1474, column: 18, scope: !4872)
!4879 = !DILocation(line: 1474, column: 28, scope: !4872)
!4880 = !DILocation(line: 1474, column: 32, scope: !4872)
!4881 = !DILocation(line: 1474, column: 36, scope: !4872)
!4882 = !DILocation(line: 1474, column: 47, scope: !4872)
!4883 = !DILocation(line: 1474, column: 3, scope: !4872)
!4884 = !DILocation(line: 1475, column: 18, scope: !4872)
!4885 = !DILocation(line: 1475, column: 28, scope: !4872)
!4886 = !DILocation(line: 1475, column: 32, scope: !4872)
!4887 = !DILocation(line: 1475, column: 36, scope: !4872)
!4888 = !DILocation(line: 1475, column: 47, scope: !4872)
!4889 = !DILocation(line: 1475, column: 3, scope: !4872)
!4890 = !DILocation(line: 1477, column: 1, scope: !4840)
!4891 = distinct !DISubprogram(name: "evolve_gpu", linkageName: "_ZL10evolve_gpuP8dcomplexS0_Pd", scope: !3, file: !3, line: 1428, type: !3642, scopeLine: 1430, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1051)
!4892 = !DILocalVariable(name: "u0", arg: 1, scope: !4891, file: !3, line: 1428, type: !98)
!4893 = !DILocation(line: 1428, column: 33, scope: !4891)
!4894 = !DILocalVariable(name: "u1", arg: 2, scope: !4891, file: !3, line: 1429, type: !98)
!4895 = !DILocation(line: 1429, column: 12, scope: !4891)
!4896 = !DILocalVariable(name: "twiddle", arg: 3, scope: !4891, file: !3, line: 1430, type: !106)
!4897 = !DILocation(line: 1430, column: 10, scope: !4891)
!4898 = !DILocation(line: 1434, column: 22, scope: !4891)
!4899 = !DILocation(line: 1435, column: 3, scope: !4891)
!4900 = !DILocation(line: 1434, column: 19, scope: !4891)
!4901 = !DILocation(line: 1434, column: 2, scope: !4891)
!4902 = !DILocation(line: 1435, column: 34, scope: !4891)
!4903 = !DILocation(line: 1436, column: 5, scope: !4891)
!4904 = !DILocation(line: 1437, column: 5, scope: !4891)
!4905 = !DILocation(line: 1438, column: 2, scope: !4891)
!4906 = !DILocation(line: 1442, column: 1, scope: !4891)
!4907 = distinct !DISubprogram(name: "checksum_gpu", linkageName: "_ZL12checksum_gpuiP8dcomplex", scope: !3, file: !3, line: 1308, type: !4908, scopeLine: 1309, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1051)
!4908 = !DISubroutineType(types: !4909)
!4909 = !{null, !97, !98}
!4910 = !DILocalVariable(name: "iteration", arg: 1, scope: !4907, file: !3, line: 1308, type: !97)
!4911 = !DILocation(line: 1308, column: 30, scope: !4907)
!4912 = !DILocalVariable(name: "u1", arg: 2, scope: !4907, file: !3, line: 1309, type: !98)
!4913 = !DILocation(line: 1309, column: 12, scope: !4907)
!4914 = !DILocation(line: 1313, column: 24, scope: !4907)
!4915 = !DILocation(line: 1314, column: 3, scope: !4907)
!4916 = !DILocation(line: 1315, column: 3, scope: !4907)
!4917 = !DILocation(line: 1313, column: 21, scope: !4907)
!4918 = !DILocation(line: 1313, column: 2, scope: !4907)
!4919 = !DILocation(line: 1315, column: 23, scope: !4907)
!4920 = !DILocation(line: 1316, column: 5, scope: !4907)
!4921 = !DILocation(line: 1317, column: 5, scope: !4907)
!4922 = !DILocation(line: 1321, column: 1, scope: !4907)
!4923 = distinct !DISubprogram(name: "verify", linkageName: "_ZL6verifyiiiiPiPc", scope: !3, file: !3, line: 1838, type: !4924, scopeLine: 1843, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1051)
!4924 = !DISubroutineType(types: !4925)
!4925 = !{null, !97, !97, !97, !97, !4926, !108}
!4926 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !4171, size: 64)
!4927 = !DILocalVariable(name: "d1", arg: 1, scope: !4923, file: !3, line: 1838, type: !97)
!4928 = !DILocation(line: 1838, column: 24, scope: !4923)
!4929 = !DILocalVariable(name: "d2", arg: 2, scope: !4923, file: !3, line: 1839, type: !97)
!4930 = !DILocation(line: 1839, column: 7, scope: !4923)
!4931 = !DILocalVariable(name: "d3", arg: 3, scope: !4923, file: !3, line: 1840, type: !97)
!4932 = !DILocation(line: 1840, column: 7, scope: !4923)
!4933 = !DILocalVariable(name: "nt", arg: 4, scope: !4923, file: !3, line: 1841, type: !97)
!4934 = !DILocation(line: 1841, column: 7, scope: !4923)
!4935 = !DILocalVariable(name: "verified", arg: 5, scope: !4923, file: !3, line: 1842, type: !4926)
!4936 = !DILocation(line: 1842, column: 12, scope: !4923)
!4937 = !DILocalVariable(name: "class_npb", arg: 6, scope: !4923, file: !3, line: 1843, type: !108)
!4938 = !DILocation(line: 1843, column: 9, scope: !4923)
!4939 = !DILocalVariable(name: "i", scope: !4923, file: !3, line: 1844, type: !97)
!4940 = !DILocation(line: 1844, column: 6, scope: !4923)
!4941 = !DILocalVariable(name: "err", scope: !4923, file: !3, line: 1845, type: !104)
!4942 = !DILocation(line: 1845, column: 9, scope: !4923)
!4943 = !DILocalVariable(name: "epsilon", scope: !4923, file: !3, line: 1845, type: !104)
!4944 = !DILocation(line: 1845, column: 14, scope: !4923)
!4945 = !DILocalVariable(name: "csum_ref", scope: !4923, file: !3, line: 1851, type: !4946)
!4946 = !DICompositeType(tag: DW_TAG_array_type, baseType: !99, size: 3328, elements: !4947)
!4947 = !{!4948}
!4948 = !DISubrange(count: 26)
!4949 = !DILocation(line: 1851, column: 11, scope: !4923)
!4950 = !DILocation(line: 1852, column: 3, scope: !4923)
!4951 = !DILocation(line: 1852, column: 13, scope: !4923)
!4952 = !DILocation(line: 1853, column: 10, scope: !4923)
!4953 = !DILocation(line: 1854, column: 3, scope: !4923)
!4954 = !DILocation(line: 1854, column: 12, scope: !4923)
!4955 = !DILocation(line: 1855, column: 5, scope: !4956)
!4956 = distinct !DILexicalBlock(scope: !4923, file: !3, line: 1855, column: 5)
!4957 = !DILocation(line: 1855, column: 8, scope: !4956)
!4958 = !DILocation(line: 1855, column: 14, scope: !4956)
!4959 = !DILocation(line: 1855, column: 17, scope: !4956)
!4960 = !DILocation(line: 1855, column: 20, scope: !4956)
!4961 = !DILocation(line: 1855, column: 26, scope: !4956)
!4962 = !DILocation(line: 1855, column: 29, scope: !4956)
!4963 = !DILocation(line: 1855, column: 32, scope: !4956)
!4964 = !DILocation(line: 1855, column: 38, scope: !4956)
!4965 = !DILocation(line: 1855, column: 41, scope: !4956)
!4966 = !DILocation(line: 1855, column: 44, scope: !4956)
!4967 = !DILocation(line: 1855, column: 5, scope: !4923)
!4968 = !DILocation(line: 1861, column: 4, scope: !4969)
!4969 = distinct !DILexicalBlock(scope: !4956, file: !3, line: 1855, column: 49)
!4970 = !DILocation(line: 1861, column: 14, scope: !4969)
!4971 = !DILocation(line: 1862, column: 17, scope: !4969)
!4972 = !DILocation(line: 1862, column: 3, scope: !4969)
!4973 = !DILocation(line: 1862, column: 15, scope: !4969)
!4974 = !DILocation(line: 1863, column: 17, scope: !4969)
!4975 = !DILocation(line: 1863, column: 3, scope: !4969)
!4976 = !DILocation(line: 1863, column: 15, scope: !4969)
!4977 = !DILocation(line: 1864, column: 17, scope: !4969)
!4978 = !DILocation(line: 1864, column: 3, scope: !4969)
!4979 = !DILocation(line: 1864, column: 15, scope: !4969)
!4980 = !DILocation(line: 1865, column: 17, scope: !4969)
!4981 = !DILocation(line: 1865, column: 3, scope: !4969)
!4982 = !DILocation(line: 1865, column: 15, scope: !4969)
!4983 = !DILocation(line: 1866, column: 17, scope: !4969)
!4984 = !DILocation(line: 1866, column: 3, scope: !4969)
!4985 = !DILocation(line: 1866, column: 15, scope: !4969)
!4986 = !DILocation(line: 1867, column: 17, scope: !4969)
!4987 = !DILocation(line: 1867, column: 3, scope: !4969)
!4988 = !DILocation(line: 1867, column: 15, scope: !4969)
!4989 = !DILocation(line: 1868, column: 2, scope: !4969)
!4990 = !DILocation(line: 1868, column: 11, scope: !4991)
!4991 = distinct !DILexicalBlock(scope: !4956, file: !3, line: 1868, column: 11)
!4992 = !DILocation(line: 1868, column: 14, scope: !4991)
!4993 = !DILocation(line: 1868, column: 21, scope: !4991)
!4994 = !DILocation(line: 1868, column: 24, scope: !4991)
!4995 = !DILocation(line: 1868, column: 27, scope: !4991)
!4996 = !DILocation(line: 1868, column: 34, scope: !4991)
!4997 = !DILocation(line: 1868, column: 37, scope: !4991)
!4998 = !DILocation(line: 1868, column: 40, scope: !4991)
!4999 = !DILocation(line: 1868, column: 46, scope: !4991)
!5000 = !DILocation(line: 1868, column: 49, scope: !4991)
!5001 = !DILocation(line: 1868, column: 52, scope: !4991)
!5002 = !DILocation(line: 1868, column: 11, scope: !4956)
!5003 = !DILocation(line: 1874, column: 4, scope: !5004)
!5004 = distinct !DILexicalBlock(scope: !4991, file: !3, line: 1868, column: 57)
!5005 = !DILocation(line: 1874, column: 14, scope: !5004)
!5006 = !DILocation(line: 1875, column: 17, scope: !5004)
!5007 = !DILocation(line: 1875, column: 3, scope: !5004)
!5008 = !DILocation(line: 1875, column: 15, scope: !5004)
!5009 = !DILocation(line: 1876, column: 17, scope: !5004)
!5010 = !DILocation(line: 1876, column: 3, scope: !5004)
!5011 = !DILocation(line: 1876, column: 15, scope: !5004)
!5012 = !DILocation(line: 1877, column: 17, scope: !5004)
!5013 = !DILocation(line: 1877, column: 3, scope: !5004)
!5014 = !DILocation(line: 1877, column: 15, scope: !5004)
!5015 = !DILocation(line: 1878, column: 17, scope: !5004)
!5016 = !DILocation(line: 1878, column: 3, scope: !5004)
!5017 = !DILocation(line: 1878, column: 15, scope: !5004)
!5018 = !DILocation(line: 1879, column: 17, scope: !5004)
!5019 = !DILocation(line: 1879, column: 3, scope: !5004)
!5020 = !DILocation(line: 1879, column: 15, scope: !5004)
!5021 = !DILocation(line: 1880, column: 17, scope: !5004)
!5022 = !DILocation(line: 1880, column: 3, scope: !5004)
!5023 = !DILocation(line: 1880, column: 15, scope: !5004)
!5024 = !DILocation(line: 1881, column: 2, scope: !5004)
!5025 = !DILocation(line: 1881, column: 11, scope: !5026)
!5026 = distinct !DILexicalBlock(scope: !4991, file: !3, line: 1881, column: 11)
!5027 = !DILocation(line: 1881, column: 14, scope: !5026)
!5028 = !DILocation(line: 1881, column: 21, scope: !5026)
!5029 = !DILocation(line: 1881, column: 24, scope: !5026)
!5030 = !DILocation(line: 1881, column: 27, scope: !5026)
!5031 = !DILocation(line: 1881, column: 34, scope: !5026)
!5032 = !DILocation(line: 1881, column: 37, scope: !5026)
!5033 = !DILocation(line: 1881, column: 40, scope: !5026)
!5034 = !DILocation(line: 1881, column: 47, scope: !5026)
!5035 = !DILocation(line: 1881, column: 50, scope: !5026)
!5036 = !DILocation(line: 1881, column: 53, scope: !5026)
!5037 = !DILocation(line: 1881, column: 11, scope: !4991)
!5038 = !DILocation(line: 1887, column: 4, scope: !5039)
!5039 = distinct !DILexicalBlock(scope: !5026, file: !3, line: 1881, column: 58)
!5040 = !DILocation(line: 1887, column: 14, scope: !5039)
!5041 = !DILocation(line: 1888, column: 17, scope: !5039)
!5042 = !DILocation(line: 1888, column: 3, scope: !5039)
!5043 = !DILocation(line: 1888, column: 15, scope: !5039)
!5044 = !DILocation(line: 1889, column: 17, scope: !5039)
!5045 = !DILocation(line: 1889, column: 3, scope: !5039)
!5046 = !DILocation(line: 1889, column: 15, scope: !5039)
!5047 = !DILocation(line: 1890, column: 17, scope: !5039)
!5048 = !DILocation(line: 1890, column: 3, scope: !5039)
!5049 = !DILocation(line: 1890, column: 15, scope: !5039)
!5050 = !DILocation(line: 1891, column: 17, scope: !5039)
!5051 = !DILocation(line: 1891, column: 3, scope: !5039)
!5052 = !DILocation(line: 1891, column: 15, scope: !5039)
!5053 = !DILocation(line: 1892, column: 17, scope: !5039)
!5054 = !DILocation(line: 1892, column: 3, scope: !5039)
!5055 = !DILocation(line: 1892, column: 15, scope: !5039)
!5056 = !DILocation(line: 1893, column: 17, scope: !5039)
!5057 = !DILocation(line: 1893, column: 3, scope: !5039)
!5058 = !DILocation(line: 1893, column: 15, scope: !5039)
!5059 = !DILocation(line: 1894, column: 2, scope: !5039)
!5060 = !DILocation(line: 1894, column: 11, scope: !5061)
!5061 = distinct !DILexicalBlock(scope: !5026, file: !3, line: 1894, column: 11)
!5062 = !DILocation(line: 1894, column: 14, scope: !5061)
!5063 = !DILocation(line: 1894, column: 21, scope: !5061)
!5064 = !DILocation(line: 1894, column: 24, scope: !5061)
!5065 = !DILocation(line: 1894, column: 27, scope: !5061)
!5066 = !DILocation(line: 1894, column: 34, scope: !5061)
!5067 = !DILocation(line: 1894, column: 37, scope: !5061)
!5068 = !DILocation(line: 1894, column: 40, scope: !5061)
!5069 = !DILocation(line: 1894, column: 47, scope: !5061)
!5070 = !DILocation(line: 1894, column: 50, scope: !5061)
!5071 = !DILocation(line: 1894, column: 53, scope: !5061)
!5072 = !DILocation(line: 1894, column: 11, scope: !5026)
!5073 = !DILocation(line: 1900, column: 4, scope: !5074)
!5074 = distinct !DILexicalBlock(scope: !5061, file: !3, line: 1894, column: 59)
!5075 = !DILocation(line: 1900, column: 14, scope: !5074)
!5076 = !DILocation(line: 1901, column: 18, scope: !5074)
!5077 = !DILocation(line: 1901, column: 3, scope: !5074)
!5078 = !DILocation(line: 1901, column: 16, scope: !5074)
!5079 = !DILocation(line: 1902, column: 18, scope: !5074)
!5080 = !DILocation(line: 1902, column: 3, scope: !5074)
!5081 = !DILocation(line: 1902, column: 16, scope: !5074)
!5082 = !DILocation(line: 1903, column: 18, scope: !5074)
!5083 = !DILocation(line: 1903, column: 3, scope: !5074)
!5084 = !DILocation(line: 1903, column: 16, scope: !5074)
!5085 = !DILocation(line: 1904, column: 18, scope: !5074)
!5086 = !DILocation(line: 1904, column: 3, scope: !5074)
!5087 = !DILocation(line: 1904, column: 16, scope: !5074)
!5088 = !DILocation(line: 1905, column: 18, scope: !5074)
!5089 = !DILocation(line: 1905, column: 3, scope: !5074)
!5090 = !DILocation(line: 1905, column: 16, scope: !5074)
!5091 = !DILocation(line: 1906, column: 18, scope: !5074)
!5092 = !DILocation(line: 1906, column: 3, scope: !5074)
!5093 = !DILocation(line: 1906, column: 16, scope: !5074)
!5094 = !DILocation(line: 1907, column: 18, scope: !5074)
!5095 = !DILocation(line: 1907, column: 3, scope: !5074)
!5096 = !DILocation(line: 1907, column: 16, scope: !5074)
!5097 = !DILocation(line: 1908, column: 18, scope: !5074)
!5098 = !DILocation(line: 1908, column: 3, scope: !5074)
!5099 = !DILocation(line: 1908, column: 16, scope: !5074)
!5100 = !DILocation(line: 1909, column: 18, scope: !5074)
!5101 = !DILocation(line: 1909, column: 3, scope: !5074)
!5102 = !DILocation(line: 1909, column: 16, scope: !5074)
!5103 = !DILocation(line: 1910, column: 18, scope: !5074)
!5104 = !DILocation(line: 1910, column: 3, scope: !5074)
!5105 = !DILocation(line: 1910, column: 16, scope: !5074)
!5106 = !DILocation(line: 1911, column: 18, scope: !5074)
!5107 = !DILocation(line: 1911, column: 3, scope: !5074)
!5108 = !DILocation(line: 1911, column: 16, scope: !5074)
!5109 = !DILocation(line: 1912, column: 18, scope: !5074)
!5110 = !DILocation(line: 1912, column: 3, scope: !5074)
!5111 = !DILocation(line: 1912, column: 16, scope: !5074)
!5112 = !DILocation(line: 1913, column: 18, scope: !5074)
!5113 = !DILocation(line: 1913, column: 3, scope: !5074)
!5114 = !DILocation(line: 1913, column: 16, scope: !5074)
!5115 = !DILocation(line: 1914, column: 18, scope: !5074)
!5116 = !DILocation(line: 1914, column: 3, scope: !5074)
!5117 = !DILocation(line: 1914, column: 16, scope: !5074)
!5118 = !DILocation(line: 1915, column: 18, scope: !5074)
!5119 = !DILocation(line: 1915, column: 3, scope: !5074)
!5120 = !DILocation(line: 1915, column: 16, scope: !5074)
!5121 = !DILocation(line: 1916, column: 18, scope: !5074)
!5122 = !DILocation(line: 1916, column: 3, scope: !5074)
!5123 = !DILocation(line: 1916, column: 16, scope: !5074)
!5124 = !DILocation(line: 1917, column: 18, scope: !5074)
!5125 = !DILocation(line: 1917, column: 3, scope: !5074)
!5126 = !DILocation(line: 1917, column: 16, scope: !5074)
!5127 = !DILocation(line: 1918, column: 18, scope: !5074)
!5128 = !DILocation(line: 1918, column: 3, scope: !5074)
!5129 = !DILocation(line: 1918, column: 16, scope: !5074)
!5130 = !DILocation(line: 1919, column: 18, scope: !5074)
!5131 = !DILocation(line: 1919, column: 3, scope: !5074)
!5132 = !DILocation(line: 1919, column: 16, scope: !5074)
!5133 = !DILocation(line: 1920, column: 18, scope: !5074)
!5134 = !DILocation(line: 1920, column: 3, scope: !5074)
!5135 = !DILocation(line: 1920, column: 16, scope: !5074)
!5136 = !DILocation(line: 1921, column: 2, scope: !5074)
!5137 = !DILocation(line: 1921, column: 11, scope: !5138)
!5138 = distinct !DILexicalBlock(scope: !5061, file: !3, line: 1921, column: 11)
!5139 = !DILocation(line: 1921, column: 14, scope: !5138)
!5140 = !DILocation(line: 1921, column: 21, scope: !5138)
!5141 = !DILocation(line: 1921, column: 24, scope: !5138)
!5142 = !DILocation(line: 1921, column: 27, scope: !5138)
!5143 = !DILocation(line: 1921, column: 34, scope: !5138)
!5144 = !DILocation(line: 1921, column: 37, scope: !5138)
!5145 = !DILocation(line: 1921, column: 40, scope: !5138)
!5146 = !DILocation(line: 1921, column: 47, scope: !5138)
!5147 = !DILocation(line: 1921, column: 50, scope: !5138)
!5148 = !DILocation(line: 1921, column: 53, scope: !5138)
!5149 = !DILocation(line: 1921, column: 11, scope: !5061)
!5150 = !DILocation(line: 1927, column: 4, scope: !5151)
!5151 = distinct !DILexicalBlock(scope: !5138, file: !3, line: 1921, column: 59)
!5152 = !DILocation(line: 1927, column: 14, scope: !5151)
!5153 = !DILocation(line: 1928, column: 18, scope: !5151)
!5154 = !DILocation(line: 1928, column: 3, scope: !5151)
!5155 = !DILocation(line: 1928, column: 16, scope: !5151)
!5156 = !DILocation(line: 1929, column: 18, scope: !5151)
!5157 = !DILocation(line: 1929, column: 3, scope: !5151)
!5158 = !DILocation(line: 1929, column: 16, scope: !5151)
!5159 = !DILocation(line: 1930, column: 18, scope: !5151)
!5160 = !DILocation(line: 1930, column: 3, scope: !5151)
!5161 = !DILocation(line: 1930, column: 16, scope: !5151)
!5162 = !DILocation(line: 1931, column: 18, scope: !5151)
!5163 = !DILocation(line: 1931, column: 3, scope: !5151)
!5164 = !DILocation(line: 1931, column: 16, scope: !5151)
!5165 = !DILocation(line: 1932, column: 18, scope: !5151)
!5166 = !DILocation(line: 1932, column: 3, scope: !5151)
!5167 = !DILocation(line: 1932, column: 16, scope: !5151)
!5168 = !DILocation(line: 1933, column: 18, scope: !5151)
!5169 = !DILocation(line: 1933, column: 3, scope: !5151)
!5170 = !DILocation(line: 1933, column: 16, scope: !5151)
!5171 = !DILocation(line: 1934, column: 18, scope: !5151)
!5172 = !DILocation(line: 1934, column: 3, scope: !5151)
!5173 = !DILocation(line: 1934, column: 16, scope: !5151)
!5174 = !DILocation(line: 1935, column: 18, scope: !5151)
!5175 = !DILocation(line: 1935, column: 3, scope: !5151)
!5176 = !DILocation(line: 1935, column: 16, scope: !5151)
!5177 = !DILocation(line: 1936, column: 18, scope: !5151)
!5178 = !DILocation(line: 1936, column: 3, scope: !5151)
!5179 = !DILocation(line: 1936, column: 16, scope: !5151)
!5180 = !DILocation(line: 1937, column: 18, scope: !5151)
!5181 = !DILocation(line: 1937, column: 3, scope: !5151)
!5182 = !DILocation(line: 1937, column: 16, scope: !5151)
!5183 = !DILocation(line: 1938, column: 18, scope: !5151)
!5184 = !DILocation(line: 1938, column: 3, scope: !5151)
!5185 = !DILocation(line: 1938, column: 16, scope: !5151)
!5186 = !DILocation(line: 1939, column: 18, scope: !5151)
!5187 = !DILocation(line: 1939, column: 3, scope: !5151)
!5188 = !DILocation(line: 1939, column: 16, scope: !5151)
!5189 = !DILocation(line: 1940, column: 18, scope: !5151)
!5190 = !DILocation(line: 1940, column: 3, scope: !5151)
!5191 = !DILocation(line: 1940, column: 16, scope: !5151)
!5192 = !DILocation(line: 1941, column: 18, scope: !5151)
!5193 = !DILocation(line: 1941, column: 3, scope: !5151)
!5194 = !DILocation(line: 1941, column: 16, scope: !5151)
!5195 = !DILocation(line: 1942, column: 18, scope: !5151)
!5196 = !DILocation(line: 1942, column: 3, scope: !5151)
!5197 = !DILocation(line: 1942, column: 16, scope: !5151)
!5198 = !DILocation(line: 1943, column: 18, scope: !5151)
!5199 = !DILocation(line: 1943, column: 3, scope: !5151)
!5200 = !DILocation(line: 1943, column: 16, scope: !5151)
!5201 = !DILocation(line: 1944, column: 18, scope: !5151)
!5202 = !DILocation(line: 1944, column: 3, scope: !5151)
!5203 = !DILocation(line: 1944, column: 16, scope: !5151)
!5204 = !DILocation(line: 1945, column: 18, scope: !5151)
!5205 = !DILocation(line: 1945, column: 3, scope: !5151)
!5206 = !DILocation(line: 1945, column: 16, scope: !5151)
!5207 = !DILocation(line: 1946, column: 18, scope: !5151)
!5208 = !DILocation(line: 1946, column: 3, scope: !5151)
!5209 = !DILocation(line: 1946, column: 16, scope: !5151)
!5210 = !DILocation(line: 1947, column: 18, scope: !5151)
!5211 = !DILocation(line: 1947, column: 3, scope: !5151)
!5212 = !DILocation(line: 1947, column: 16, scope: !5151)
!5213 = !DILocation(line: 1948, column: 2, scope: !5151)
!5214 = !DILocation(line: 1948, column: 11, scope: !5215)
!5215 = distinct !DILexicalBlock(scope: !5138, file: !3, line: 1948, column: 11)
!5216 = !DILocation(line: 1948, column: 14, scope: !5215)
!5217 = !DILocation(line: 1948, column: 22, scope: !5215)
!5218 = !DILocation(line: 1948, column: 25, scope: !5215)
!5219 = !DILocation(line: 1948, column: 28, scope: !5215)
!5220 = !DILocation(line: 1948, column: 36, scope: !5215)
!5221 = !DILocation(line: 1948, column: 39, scope: !5215)
!5222 = !DILocation(line: 1948, column: 42, scope: !5215)
!5223 = !DILocation(line: 1948, column: 50, scope: !5215)
!5224 = !DILocation(line: 1948, column: 53, scope: !5215)
!5225 = !DILocation(line: 1948, column: 56, scope: !5215)
!5226 = !DILocation(line: 1948, column: 11, scope: !5138)
!5227 = !DILocation(line: 1954, column: 4, scope: !5228)
!5228 = distinct !DILexicalBlock(scope: !5215, file: !3, line: 1948, column: 62)
!5229 = !DILocation(line: 1954, column: 14, scope: !5228)
!5230 = !DILocation(line: 1955, column: 18, scope: !5228)
!5231 = !DILocation(line: 1955, column: 3, scope: !5228)
!5232 = !DILocation(line: 1955, column: 16, scope: !5228)
!5233 = !DILocation(line: 1956, column: 18, scope: !5228)
!5234 = !DILocation(line: 1956, column: 3, scope: !5228)
!5235 = !DILocation(line: 1956, column: 16, scope: !5228)
!5236 = !DILocation(line: 1957, column: 18, scope: !5228)
!5237 = !DILocation(line: 1957, column: 3, scope: !5228)
!5238 = !DILocation(line: 1957, column: 16, scope: !5228)
!5239 = !DILocation(line: 1958, column: 18, scope: !5228)
!5240 = !DILocation(line: 1958, column: 3, scope: !5228)
!5241 = !DILocation(line: 1958, column: 16, scope: !5228)
!5242 = !DILocation(line: 1959, column: 18, scope: !5228)
!5243 = !DILocation(line: 1959, column: 3, scope: !5228)
!5244 = !DILocation(line: 1959, column: 16, scope: !5228)
!5245 = !DILocation(line: 1960, column: 18, scope: !5228)
!5246 = !DILocation(line: 1960, column: 3, scope: !5228)
!5247 = !DILocation(line: 1960, column: 16, scope: !5228)
!5248 = !DILocation(line: 1961, column: 18, scope: !5228)
!5249 = !DILocation(line: 1961, column: 3, scope: !5228)
!5250 = !DILocation(line: 1961, column: 16, scope: !5228)
!5251 = !DILocation(line: 1962, column: 18, scope: !5228)
!5252 = !DILocation(line: 1962, column: 3, scope: !5228)
!5253 = !DILocation(line: 1962, column: 16, scope: !5228)
!5254 = !DILocation(line: 1963, column: 18, scope: !5228)
!5255 = !DILocation(line: 1963, column: 3, scope: !5228)
!5256 = !DILocation(line: 1963, column: 16, scope: !5228)
!5257 = !DILocation(line: 1964, column: 18, scope: !5228)
!5258 = !DILocation(line: 1964, column: 3, scope: !5228)
!5259 = !DILocation(line: 1964, column: 16, scope: !5228)
!5260 = !DILocation(line: 1965, column: 18, scope: !5228)
!5261 = !DILocation(line: 1965, column: 3, scope: !5228)
!5262 = !DILocation(line: 1965, column: 16, scope: !5228)
!5263 = !DILocation(line: 1966, column: 18, scope: !5228)
!5264 = !DILocation(line: 1966, column: 3, scope: !5228)
!5265 = !DILocation(line: 1966, column: 16, scope: !5228)
!5266 = !DILocation(line: 1967, column: 18, scope: !5228)
!5267 = !DILocation(line: 1967, column: 3, scope: !5228)
!5268 = !DILocation(line: 1967, column: 16, scope: !5228)
!5269 = !DILocation(line: 1968, column: 18, scope: !5228)
!5270 = !DILocation(line: 1968, column: 3, scope: !5228)
!5271 = !DILocation(line: 1968, column: 16, scope: !5228)
!5272 = !DILocation(line: 1969, column: 18, scope: !5228)
!5273 = !DILocation(line: 1969, column: 3, scope: !5228)
!5274 = !DILocation(line: 1969, column: 16, scope: !5228)
!5275 = !DILocation(line: 1970, column: 18, scope: !5228)
!5276 = !DILocation(line: 1970, column: 3, scope: !5228)
!5277 = !DILocation(line: 1970, column: 16, scope: !5228)
!5278 = !DILocation(line: 1971, column: 18, scope: !5228)
!5279 = !DILocation(line: 1971, column: 3, scope: !5228)
!5280 = !DILocation(line: 1971, column: 16, scope: !5228)
!5281 = !DILocation(line: 1972, column: 18, scope: !5228)
!5282 = !DILocation(line: 1972, column: 3, scope: !5228)
!5283 = !DILocation(line: 1972, column: 16, scope: !5228)
!5284 = !DILocation(line: 1973, column: 18, scope: !5228)
!5285 = !DILocation(line: 1973, column: 3, scope: !5228)
!5286 = !DILocation(line: 1973, column: 16, scope: !5228)
!5287 = !DILocation(line: 1974, column: 18, scope: !5228)
!5288 = !DILocation(line: 1974, column: 3, scope: !5228)
!5289 = !DILocation(line: 1974, column: 16, scope: !5228)
!5290 = !DILocation(line: 1975, column: 18, scope: !5228)
!5291 = !DILocation(line: 1975, column: 3, scope: !5228)
!5292 = !DILocation(line: 1975, column: 16, scope: !5228)
!5293 = !DILocation(line: 1976, column: 18, scope: !5228)
!5294 = !DILocation(line: 1976, column: 3, scope: !5228)
!5295 = !DILocation(line: 1976, column: 16, scope: !5228)
!5296 = !DILocation(line: 1977, column: 18, scope: !5228)
!5297 = !DILocation(line: 1977, column: 3, scope: !5228)
!5298 = !DILocation(line: 1977, column: 16, scope: !5228)
!5299 = !DILocation(line: 1978, column: 18, scope: !5228)
!5300 = !DILocation(line: 1978, column: 3, scope: !5228)
!5301 = !DILocation(line: 1978, column: 16, scope: !5228)
!5302 = !DILocation(line: 1979, column: 18, scope: !5228)
!5303 = !DILocation(line: 1979, column: 3, scope: !5228)
!5304 = !DILocation(line: 1979, column: 16, scope: !5228)
!5305 = !DILocation(line: 1980, column: 2, scope: !5228)
!5306 = !DILocation(line: 1980, column: 11, scope: !5307)
!5307 = distinct !DILexicalBlock(scope: !5215, file: !3, line: 1980, column: 11)
!5308 = !DILocation(line: 1980, column: 14, scope: !5307)
!5309 = !DILocation(line: 1980, column: 22, scope: !5307)
!5310 = !DILocation(line: 1980, column: 25, scope: !5307)
!5311 = !DILocation(line: 1980, column: 28, scope: !5307)
!5312 = !DILocation(line: 1980, column: 36, scope: !5307)
!5313 = !DILocation(line: 1980, column: 39, scope: !5307)
!5314 = !DILocation(line: 1980, column: 42, scope: !5307)
!5315 = !DILocation(line: 1980, column: 50, scope: !5307)
!5316 = !DILocation(line: 1980, column: 53, scope: !5307)
!5317 = !DILocation(line: 1980, column: 56, scope: !5307)
!5318 = !DILocation(line: 1980, column: 11, scope: !5215)
!5319 = !DILocation(line: 1986, column: 4, scope: !5320)
!5320 = distinct !DILexicalBlock(scope: !5307, file: !3, line: 1980, column: 62)
!5321 = !DILocation(line: 1986, column: 14, scope: !5320)
!5322 = !DILocation(line: 1987, column: 18, scope: !5320)
!5323 = !DILocation(line: 1987, column: 3, scope: !5320)
!5324 = !DILocation(line: 1987, column: 16, scope: !5320)
!5325 = !DILocation(line: 1988, column: 18, scope: !5320)
!5326 = !DILocation(line: 1988, column: 3, scope: !5320)
!5327 = !DILocation(line: 1988, column: 16, scope: !5320)
!5328 = !DILocation(line: 1989, column: 18, scope: !5320)
!5329 = !DILocation(line: 1989, column: 3, scope: !5320)
!5330 = !DILocation(line: 1989, column: 16, scope: !5320)
!5331 = !DILocation(line: 1990, column: 18, scope: !5320)
!5332 = !DILocation(line: 1990, column: 3, scope: !5320)
!5333 = !DILocation(line: 1990, column: 16, scope: !5320)
!5334 = !DILocation(line: 1991, column: 18, scope: !5320)
!5335 = !DILocation(line: 1991, column: 3, scope: !5320)
!5336 = !DILocation(line: 1991, column: 16, scope: !5320)
!5337 = !DILocation(line: 1992, column: 18, scope: !5320)
!5338 = !DILocation(line: 1992, column: 3, scope: !5320)
!5339 = !DILocation(line: 1992, column: 16, scope: !5320)
!5340 = !DILocation(line: 1993, column: 18, scope: !5320)
!5341 = !DILocation(line: 1993, column: 3, scope: !5320)
!5342 = !DILocation(line: 1993, column: 16, scope: !5320)
!5343 = !DILocation(line: 1994, column: 18, scope: !5320)
!5344 = !DILocation(line: 1994, column: 3, scope: !5320)
!5345 = !DILocation(line: 1994, column: 16, scope: !5320)
!5346 = !DILocation(line: 1995, column: 18, scope: !5320)
!5347 = !DILocation(line: 1995, column: 3, scope: !5320)
!5348 = !DILocation(line: 1995, column: 16, scope: !5320)
!5349 = !DILocation(line: 1996, column: 18, scope: !5320)
!5350 = !DILocation(line: 1996, column: 3, scope: !5320)
!5351 = !DILocation(line: 1996, column: 16, scope: !5320)
!5352 = !DILocation(line: 1997, column: 18, scope: !5320)
!5353 = !DILocation(line: 1997, column: 3, scope: !5320)
!5354 = !DILocation(line: 1997, column: 16, scope: !5320)
!5355 = !DILocation(line: 1998, column: 18, scope: !5320)
!5356 = !DILocation(line: 1998, column: 3, scope: !5320)
!5357 = !DILocation(line: 1998, column: 16, scope: !5320)
!5358 = !DILocation(line: 1999, column: 18, scope: !5320)
!5359 = !DILocation(line: 1999, column: 3, scope: !5320)
!5360 = !DILocation(line: 1999, column: 16, scope: !5320)
!5361 = !DILocation(line: 2000, column: 18, scope: !5320)
!5362 = !DILocation(line: 2000, column: 3, scope: !5320)
!5363 = !DILocation(line: 2000, column: 16, scope: !5320)
!5364 = !DILocation(line: 2001, column: 18, scope: !5320)
!5365 = !DILocation(line: 2001, column: 3, scope: !5320)
!5366 = !DILocation(line: 2001, column: 16, scope: !5320)
!5367 = !DILocation(line: 2002, column: 18, scope: !5320)
!5368 = !DILocation(line: 2002, column: 3, scope: !5320)
!5369 = !DILocation(line: 2002, column: 16, scope: !5320)
!5370 = !DILocation(line: 2003, column: 18, scope: !5320)
!5371 = !DILocation(line: 2003, column: 3, scope: !5320)
!5372 = !DILocation(line: 2003, column: 16, scope: !5320)
!5373 = !DILocation(line: 2004, column: 18, scope: !5320)
!5374 = !DILocation(line: 2004, column: 3, scope: !5320)
!5375 = !DILocation(line: 2004, column: 16, scope: !5320)
!5376 = !DILocation(line: 2005, column: 18, scope: !5320)
!5377 = !DILocation(line: 2005, column: 3, scope: !5320)
!5378 = !DILocation(line: 2005, column: 16, scope: !5320)
!5379 = !DILocation(line: 2006, column: 18, scope: !5320)
!5380 = !DILocation(line: 2006, column: 3, scope: !5320)
!5381 = !DILocation(line: 2006, column: 16, scope: !5320)
!5382 = !DILocation(line: 2007, column: 18, scope: !5320)
!5383 = !DILocation(line: 2007, column: 3, scope: !5320)
!5384 = !DILocation(line: 2007, column: 16, scope: !5320)
!5385 = !DILocation(line: 2008, column: 18, scope: !5320)
!5386 = !DILocation(line: 2008, column: 3, scope: !5320)
!5387 = !DILocation(line: 2008, column: 16, scope: !5320)
!5388 = !DILocation(line: 2009, column: 18, scope: !5320)
!5389 = !DILocation(line: 2009, column: 3, scope: !5320)
!5390 = !DILocation(line: 2009, column: 16, scope: !5320)
!5391 = !DILocation(line: 2010, column: 18, scope: !5320)
!5392 = !DILocation(line: 2010, column: 3, scope: !5320)
!5393 = !DILocation(line: 2010, column: 16, scope: !5320)
!5394 = !DILocation(line: 2011, column: 18, scope: !5320)
!5395 = !DILocation(line: 2011, column: 3, scope: !5320)
!5396 = !DILocation(line: 2011, column: 16, scope: !5320)
!5397 = !DILocation(line: 2012, column: 2, scope: !5320)
!5398 = !DILocation(line: 2013, column: 6, scope: !5399)
!5399 = distinct !DILexicalBlock(scope: !4923, file: !3, line: 2013, column: 5)
!5400 = !DILocation(line: 2013, column: 5, scope: !5399)
!5401 = !DILocation(line: 2013, column: 16, scope: !5399)
!5402 = !DILocation(line: 2013, column: 5, scope: !4923)
!5403 = !DILocation(line: 2014, column: 4, scope: !5404)
!5404 = distinct !DILexicalBlock(scope: !5399, file: !3, line: 2013, column: 23)
!5405 = !DILocation(line: 2014, column: 13, scope: !5404)
!5406 = !DILocation(line: 2015, column: 9, scope: !5407)
!5407 = distinct !DILexicalBlock(scope: !5404, file: !3, line: 2015, column: 3)
!5408 = !DILocation(line: 2015, column: 7, scope: !5407)
!5409 = !DILocation(line: 2015, column: 14, scope: !5410)
!5410 = distinct !DILexicalBlock(scope: !5407, file: !3, line: 2015, column: 3)
!5411 = !DILocation(line: 2015, column: 19, scope: !5410)
!5412 = !DILocation(line: 2015, column: 16, scope: !5410)
!5413 = !DILocation(line: 2015, column: 3, scope: !5407)
!5414 = !DILocation(line: 2016, column: 10, scope: !5415)
!5415 = distinct !DILexicalBlock(scope: !5410, file: !3, line: 2015, column: 27)
!5416 = !DILocation(line: 2016, column: 8, scope: !5415)
!5417 = !DILocation(line: 2018, column: 9, scope: !5418)
!5418 = distinct !DILexicalBlock(scope: !5415, file: !3, line: 2018, column: 7)
!5419 = !DILocation(line: 2018, column: 16, scope: !5418)
!5420 = !DILocation(line: 2018, column: 13, scope: !5418)
!5421 = !DILocation(line: 2018, column: 7, scope: !5415)
!5422 = !DILocation(line: 2019, column: 6, scope: !5423)
!5423 = distinct !DILexicalBlock(scope: !5418, file: !3, line: 2018, column: 25)
!5424 = !DILocation(line: 2019, column: 15, scope: !5423)
!5425 = !DILocation(line: 2020, column: 5, scope: !5423)
!5426 = !DILocation(line: 2022, column: 3, scope: !5415)
!5427 = !DILocation(line: 2015, column: 24, scope: !5410)
!5428 = !DILocation(line: 2015, column: 3, scope: !5410)
!5429 = distinct !{!5429, !5413, !5430}
!5430 = !DILocation(line: 2022, column: 3, scope: !5407)
!5431 = !DILocation(line: 2023, column: 2, scope: !5404)
!5432 = !DILocation(line: 2024, column: 6, scope: !5433)
!5433 = distinct !DILexicalBlock(scope: !4923, file: !3, line: 2024, column: 5)
!5434 = !DILocation(line: 2024, column: 5, scope: !5433)
!5435 = !DILocation(line: 2024, column: 16, scope: !5433)
!5436 = !DILocation(line: 2024, column: 5, scope: !4923)
!5437 = !DILocation(line: 2025, column: 7, scope: !5438)
!5438 = distinct !DILexicalBlock(scope: !5439, file: !3, line: 2025, column: 6)
!5439 = distinct !DILexicalBlock(scope: !5433, file: !3, line: 2024, column: 23)
!5440 = !DILocation(line: 2025, column: 6, scope: !5438)
!5441 = !DILocation(line: 2025, column: 6, scope: !5439)
!5442 = !DILocation(line: 2026, column: 4, scope: !5443)
!5443 = distinct !DILexicalBlock(scope: !5438, file: !3, line: 2025, column: 16)
!5444 = !DILocation(line: 2027, column: 3, scope: !5443)
!5445 = !DILocation(line: 2028, column: 4, scope: !5446)
!5446 = distinct !DILexicalBlock(scope: !5438, file: !3, line: 2027, column: 8)
!5447 = !DILocation(line: 2030, column: 2, scope: !5439)
!5448 = !DILocation(line: 2031, column: 31, scope: !4923)
!5449 = !DILocation(line: 2031, column: 30, scope: !4923)
!5450 = !DILocation(line: 2031, column: 2, scope: !4923)
!5451 = !DILocation(line: 2032, column: 1, scope: !4923)
!5452 = distinct !DISubprogram(name: "release_gpu", linkageName: "_ZL11release_gpuv", scope: !3, file: !3, line: 1648, type: !561, scopeLine: 1648, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1051)
!5453 = !DILocation(line: 1649, column: 11, scope: !5452)
!5454 = !DILocation(line: 1649, column: 2, scope: !5452)
!5455 = !DILocation(line: 1650, column: 11, scope: !5452)
!5456 = !DILocation(line: 1650, column: 2, scope: !5452)
!5457 = !DILocation(line: 1651, column: 11, scope: !5452)
!5458 = !DILocation(line: 1651, column: 2, scope: !5452)
!5459 = !DILocation(line: 1652, column: 11, scope: !5452)
!5460 = !DILocation(line: 1652, column: 2, scope: !5452)
!5461 = !DILocation(line: 1653, column: 11, scope: !5452)
!5462 = !DILocation(line: 1653, column: 2, scope: !5452)
!5463 = !DILocation(line: 1654, column: 11, scope: !5452)
!5464 = !DILocation(line: 1654, column: 2, scope: !5452)
!5465 = !DILocation(line: 1655, column: 11, scope: !5452)
!5466 = !DILocation(line: 1655, column: 2, scope: !5452)
!5467 = !DILocation(line: 1656, column: 11, scope: !5452)
!5468 = !DILocation(line: 1656, column: 2, scope: !5452)
!5469 = !DILocation(line: 1657, column: 1, scope: !5452)
!5470 = distinct !DISubprogram(name: "dcomplex_div", linkageName: "_ZL12dcomplex_div8dcomplexS_", scope: !100, file: !100, line: 106, type: !5471, scopeLine: 106, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1051)
!5471 = !DISubroutineType(types: !5472)
!5472 = !{!99, !99, !99}
!5473 = !DILocalVariable(name: "z1", arg: 1, scope: !5470, file: !100, line: 106, type: !99)
!5474 = !DILocation(line: 106, column: 46, scope: !5470)
!5475 = !DILocalVariable(name: "z2", arg: 2, scope: !5470, file: !100, line: 106, type: !99)
!5476 = !DILocation(line: 106, column: 59, scope: !5470)
!5477 = !DILocalVariable(name: "a", scope: !5470, file: !100, line: 107, type: !104)
!5478 = !DILocation(line: 107, column: 9, scope: !5470)
!5479 = !DILocation(line: 107, column: 16, scope: !5470)
!5480 = !DILocalVariable(name: "b", scope: !5470, file: !100, line: 108, type: !104)
!5481 = !DILocation(line: 108, column: 9, scope: !5470)
!5482 = !DILocation(line: 108, column: 16, scope: !5470)
!5483 = !DILocalVariable(name: "c", scope: !5470, file: !100, line: 109, type: !104)
!5484 = !DILocation(line: 109, column: 9, scope: !5470)
!5485 = !DILocation(line: 109, column: 16, scope: !5470)
!5486 = !DILocalVariable(name: "d", scope: !5470, file: !100, line: 110, type: !104)
!5487 = !DILocation(line: 110, column: 9, scope: !5470)
!5488 = !DILocation(line: 110, column: 16, scope: !5470)
!5489 = !DILocalVariable(name: "divisor", scope: !5470, file: !100, line: 111, type: !104)
!5490 = !DILocation(line: 111, column: 9, scope: !5470)
!5491 = !DILocation(line: 111, column: 19, scope: !5470)
!5492 = !DILocation(line: 111, column: 21, scope: !5470)
!5493 = !DILocation(line: 111, column: 20, scope: !5470)
!5494 = !DILocation(line: 111, column: 25, scope: !5470)
!5495 = !DILocation(line: 111, column: 27, scope: !5470)
!5496 = !DILocation(line: 111, column: 26, scope: !5470)
!5497 = !DILocation(line: 111, column: 23, scope: !5470)
!5498 = !DILocalVariable(name: "real", scope: !5470, file: !100, line: 112, type: !104)
!5499 = !DILocation(line: 112, column: 9, scope: !5470)
!5500 = !DILocation(line: 112, column: 17, scope: !5470)
!5501 = !DILocation(line: 112, column: 19, scope: !5470)
!5502 = !DILocation(line: 112, column: 18, scope: !5470)
!5503 = !DILocation(line: 112, column: 23, scope: !5470)
!5504 = !DILocation(line: 112, column: 25, scope: !5470)
!5505 = !DILocation(line: 112, column: 24, scope: !5470)
!5506 = !DILocation(line: 112, column: 21, scope: !5470)
!5507 = !DILocation(line: 112, column: 30, scope: !5470)
!5508 = !DILocation(line: 112, column: 28, scope: !5470)
!5509 = !DILocalVariable(name: "imag", scope: !5470, file: !100, line: 113, type: !104)
!5510 = !DILocation(line: 113, column: 9, scope: !5470)
!5511 = !DILocation(line: 113, column: 17, scope: !5470)
!5512 = !DILocation(line: 113, column: 19, scope: !5470)
!5513 = !DILocation(line: 113, column: 18, scope: !5470)
!5514 = !DILocation(line: 113, column: 23, scope: !5470)
!5515 = !DILocation(line: 113, column: 25, scope: !5470)
!5516 = !DILocation(line: 113, column: 24, scope: !5470)
!5517 = !DILocation(line: 113, column: 21, scope: !5470)
!5518 = !DILocation(line: 113, column: 30, scope: !5470)
!5519 = !DILocation(line: 113, column: 28, scope: !5470)
!5520 = !DILocalVariable(name: "result", scope: !5470, file: !100, line: 114, type: !99)
!5521 = !DILocation(line: 114, column: 11, scope: !5470)
!5522 = !DILocation(line: 114, column: 30, scope: !5470)
!5523 = !DILocation(line: 114, column: 31, scope: !5470)
!5524 = !DILocation(line: 114, column: 37, scope: !5470)
!5525 = !DILocation(line: 115, column: 2, scope: !5470)
!5526 = distinct !DISubprogram(name: "dim3", linkageName: "_ZN4dim3C2Ejjj", scope: !1209, file: !1175, line: 421, type: !1215, scopeLine: 421, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, declaration: !1214, retainedNodes: !1051)
!5527 = !DILocalVariable(name: "this", arg: 1, scope: !5526, type: !5528, flags: DIFlagArtificial | DIFlagObjectPointer)
!5528 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !1209, size: 64)
!5529 = !DILocation(line: 0, scope: !5526)
!5530 = !DILocalVariable(name: "vx", arg: 2, scope: !5526, file: !1175, line: 421, type: !7)
!5531 = !DILocation(line: 421, column: 43, scope: !5526)
!5532 = !DILocalVariable(name: "vy", arg: 3, scope: !5526, file: !1175, line: 421, type: !7)
!5533 = !DILocation(line: 421, column: 64, scope: !5526)
!5534 = !DILocalVariable(name: "vz", arg: 4, scope: !5526, file: !1175, line: 421, type: !7)
!5535 = !DILocation(line: 421, column: 85, scope: !5526)
!5536 = !DILocation(line: 421, column: 95, scope: !5526)
!5537 = !DILocation(line: 421, column: 97, scope: !5526)
!5538 = !DILocation(line: 421, column: 102, scope: !5526)
!5539 = !DILocation(line: 421, column: 104, scope: !5526)
!5540 = !DILocation(line: 421, column: 109, scope: !5526)
!5541 = !DILocation(line: 421, column: 111, scope: !5526)
!5542 = !DILocation(line: 421, column: 116, scope: !5526)
!5543 = distinct !DISubprogram(name: "checksum_gpu_kernel", linkageName: "_Z19checksum_gpu_kerneliP8dcomplexS0_", scope: !3, file: !3, line: 1323, type: !3154, scopeLine: 1325, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !1051)
!5544 = !DILocalVariable(name: "iteration", arg: 1, scope: !5543, file: !3, line: 1323, type: !97)
!5545 = !DILocation(line: 1323, column: 41, scope: !5543)
!5546 = !DILocalVariable(name: "u1", arg: 2, scope: !5543, file: !3, line: 1324, type: !98)
!5547 = !DILocation(line: 1324, column: 12, scope: !5543)
!5548 = !DILocalVariable(name: "sums", arg: 3, scope: !5543, file: !3, line: 1325, type: !98)
!5549 = !DILocation(line: 1325, column: 12, scope: !5543)
!5550 = !DILocation(line: 1325, column: 19, scope: !5543)
!5551 = !DILocation(line: 1352, column: 1, scope: !5543)
!5552 = distinct !DISubprogram(name: "evolve_gpu_kernel", linkageName: "_Z17evolve_gpu_kernelP8dcomplexS0_Pd", scope: !3, file: !3, line: 1444, type: !3642, scopeLine: 1446, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !1051)
!5553 = !DILocalVariable(name: "u0", arg: 1, scope: !5552, file: !3, line: 1444, type: !98)
!5554 = !DILocation(line: 1444, column: 44, scope: !5552)
!5555 = !DILocalVariable(name: "u1", arg: 2, scope: !5552, file: !3, line: 1445, type: !98)
!5556 = !DILocation(line: 1445, column: 12, scope: !5552)
!5557 = !DILocalVariable(name: "twiddle", arg: 3, scope: !5552, file: !3, line: 1446, type: !106)
!5558 = !DILocation(line: 1446, column: 10, scope: !5552)
!5559 = !DILocation(line: 1446, column: 20, scope: !5552)
!5560 = !DILocation(line: 1455, column: 1, scope: !5552)
!5561 = distinct !DISubprogram(name: "cffts1_gpu", linkageName: "_ZL10cffts1_gpuiP8dcomplexS0_S0_S0_S0_", scope: !3, file: !3, line: 725, type: !5562, scopeLine: 730, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1051)
!5562 = !DISubroutineType(types: !5563)
!5563 = !{null, !1326, !98, !98, !98, !98, !98}
!5564 = !DILocalVariable(name: "is", arg: 1, scope: !5561, file: !3, line: 725, type: !1326)
!5565 = !DILocation(line: 725, column: 34, scope: !5561)
!5566 = !DILocalVariable(name: "u", arg: 2, scope: !5561, file: !3, line: 726, type: !98)
!5567 = !DILocation(line: 726, column: 12, scope: !5561)
!5568 = !DILocalVariable(name: "x_in", arg: 3, scope: !5561, file: !3, line: 727, type: !98)
!5569 = !DILocation(line: 727, column: 12, scope: !5561)
!5570 = !DILocalVariable(name: "x_out", arg: 4, scope: !5561, file: !3, line: 728, type: !98)
!5571 = !DILocation(line: 728, column: 12, scope: !5561)
!5572 = !DILocalVariable(name: "y0", arg: 5, scope: !5561, file: !3, line: 729, type: !98)
!5573 = !DILocation(line: 729, column: 12, scope: !5561)
!5574 = !DILocalVariable(name: "y1", arg: 6, scope: !5561, file: !3, line: 730, type: !98)
!5575 = !DILocation(line: 730, column: 12, scope: !5561)
!5576 = !DILocation(line: 734, column: 24, scope: !5561)
!5577 = !DILocation(line: 735, column: 3, scope: !5561)
!5578 = !DILocation(line: 734, column: 21, scope: !5561)
!5579 = !DILocation(line: 734, column: 2, scope: !5561)
!5580 = !DILocation(line: 735, column: 34, scope: !5561)
!5581 = !DILocation(line: 736, column: 5, scope: !5561)
!5582 = !DILocation(line: 737, column: 2, scope: !5561)
!5583 = !DILocation(line: 745, column: 24, scope: !5561)
!5584 = !DILocation(line: 746, column: 3, scope: !5561)
!5585 = !DILocation(line: 745, column: 21, scope: !5561)
!5586 = !DILocation(line: 745, column: 2, scope: !5561)
!5587 = !DILocation(line: 746, column: 34, scope: !5561)
!5588 = !DILocation(line: 747, column: 5, scope: !5561)
!5589 = !DILocation(line: 748, column: 5, scope: !5561)
!5590 = !DILocation(line: 749, column: 5, scope: !5561)
!5591 = !DILocation(line: 750, column: 2, scope: !5561)
!5592 = !DILocation(line: 758, column: 24, scope: !5561)
!5593 = !DILocation(line: 759, column: 3, scope: !5561)
!5594 = !DILocation(line: 758, column: 21, scope: !5561)
!5595 = !DILocation(line: 758, column: 2, scope: !5561)
!5596 = !DILocation(line: 759, column: 34, scope: !5561)
!5597 = !DILocation(line: 760, column: 5, scope: !5561)
!5598 = !DILocation(line: 761, column: 2, scope: !5561)
!5599 = !DILocation(line: 765, column: 1, scope: !5561)
!5600 = distinct !DISubprogram(name: "cffts2_gpu", linkageName: "_ZL10cffts2_gpuiP8dcomplexS0_S0_S0_S0_", scope: !3, file: !3, line: 920, type: !5601, scopeLine: 925, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1051)
!5601 = !DISubroutineType(types: !5602)
!5602 = !{null, !97, !98, !98, !98, !98, !98}
!5603 = !DILocalVariable(name: "is", arg: 1, scope: !5600, file: !3, line: 920, type: !97)
!5604 = !DILocation(line: 920, column: 28, scope: !5600)
!5605 = !DILocalVariable(name: "u", arg: 2, scope: !5600, file: !3, line: 921, type: !98)
!5606 = !DILocation(line: 921, column: 12, scope: !5600)
!5607 = !DILocalVariable(name: "x_in", arg: 3, scope: !5600, file: !3, line: 922, type: !98)
!5608 = !DILocation(line: 922, column: 12, scope: !5600)
!5609 = !DILocalVariable(name: "x_out", arg: 4, scope: !5600, file: !3, line: 923, type: !98)
!5610 = !DILocation(line: 923, column: 12, scope: !5600)
!5611 = !DILocalVariable(name: "y0", arg: 5, scope: !5600, file: !3, line: 924, type: !98)
!5612 = !DILocation(line: 924, column: 12, scope: !5600)
!5613 = !DILocalVariable(name: "y1", arg: 6, scope: !5600, file: !3, line: 925, type: !98)
!5614 = !DILocation(line: 925, column: 12, scope: !5600)
!5615 = !DILocation(line: 929, column: 24, scope: !5600)
!5616 = !DILocation(line: 930, column: 3, scope: !5600)
!5617 = !DILocation(line: 929, column: 21, scope: !5600)
!5618 = !DILocation(line: 929, column: 2, scope: !5600)
!5619 = !DILocation(line: 930, column: 34, scope: !5600)
!5620 = !DILocation(line: 931, column: 5, scope: !5600)
!5621 = !DILocation(line: 932, column: 2, scope: !5600)
!5622 = !DILocation(line: 940, column: 24, scope: !5600)
!5623 = !DILocation(line: 941, column: 3, scope: !5600)
!5624 = !DILocation(line: 940, column: 21, scope: !5600)
!5625 = !DILocation(line: 940, column: 2, scope: !5600)
!5626 = !DILocation(line: 941, column: 34, scope: !5600)
!5627 = !DILocation(line: 942, column: 5, scope: !5600)
!5628 = !DILocation(line: 943, column: 5, scope: !5600)
!5629 = !DILocation(line: 944, column: 5, scope: !5600)
!5630 = !DILocation(line: 945, column: 2, scope: !5600)
!5631 = !DILocation(line: 953, column: 24, scope: !5600)
!5632 = !DILocation(line: 954, column: 3, scope: !5600)
!5633 = !DILocation(line: 953, column: 21, scope: !5600)
!5634 = !DILocation(line: 953, column: 2, scope: !5600)
!5635 = !DILocation(line: 954, column: 34, scope: !5600)
!5636 = !DILocation(line: 955, column: 5, scope: !5600)
!5637 = !DILocation(line: 956, column: 2, scope: !5600)
!5638 = !DILocation(line: 960, column: 1, scope: !5600)
!5639 = distinct !DISubprogram(name: "cffts3_gpu", linkageName: "_ZL10cffts3_gpuiP8dcomplexS0_S0_S0_S0_", scope: !3, file: !3, line: 1111, type: !5601, scopeLine: 1116, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1051)
!5640 = !DILocalVariable(name: "is", arg: 1, scope: !5639, file: !3, line: 1111, type: !97)
!5641 = !DILocation(line: 1111, column: 28, scope: !5639)
!5642 = !DILocalVariable(name: "u", arg: 2, scope: !5639, file: !3, line: 1112, type: !98)
!5643 = !DILocation(line: 1112, column: 12, scope: !5639)
!5644 = !DILocalVariable(name: "x_in", arg: 3, scope: !5639, file: !3, line: 1113, type: !98)
!5645 = !DILocation(line: 1113, column: 12, scope: !5639)
!5646 = !DILocalVariable(name: "x_out", arg: 4, scope: !5639, file: !3, line: 1114, type: !98)
!5647 = !DILocation(line: 1114, column: 12, scope: !5639)
!5648 = !DILocalVariable(name: "y0", arg: 5, scope: !5639, file: !3, line: 1115, type: !98)
!5649 = !DILocation(line: 1115, column: 12, scope: !5639)
!5650 = !DILocalVariable(name: "y1", arg: 6, scope: !5639, file: !3, line: 1116, type: !98)
!5651 = !DILocation(line: 1116, column: 12, scope: !5639)
!5652 = !DILocation(line: 1120, column: 24, scope: !5639)
!5653 = !DILocation(line: 1121, column: 3, scope: !5639)
!5654 = !DILocation(line: 1120, column: 21, scope: !5639)
!5655 = !DILocation(line: 1120, column: 2, scope: !5639)
!5656 = !DILocation(line: 1121, column: 34, scope: !5639)
!5657 = !DILocation(line: 1122, column: 5, scope: !5639)
!5658 = !DILocation(line: 1123, column: 2, scope: !5639)
!5659 = !DILocation(line: 1131, column: 24, scope: !5639)
!5660 = !DILocation(line: 1132, column: 3, scope: !5639)
!5661 = !DILocation(line: 1131, column: 21, scope: !5639)
!5662 = !DILocation(line: 1131, column: 2, scope: !5639)
!5663 = !DILocation(line: 1132, column: 34, scope: !5639)
!5664 = !DILocation(line: 1133, column: 5, scope: !5639)
!5665 = !DILocation(line: 1134, column: 5, scope: !5639)
!5666 = !DILocation(line: 1135, column: 5, scope: !5639)
!5667 = !DILocation(line: 1136, column: 2, scope: !5639)
!5668 = !DILocation(line: 1144, column: 24, scope: !5639)
!5669 = !DILocation(line: 1145, column: 3, scope: !5639)
!5670 = !DILocation(line: 1144, column: 21, scope: !5639)
!5671 = !DILocation(line: 1144, column: 2, scope: !5639)
!5672 = !DILocation(line: 1145, column: 34, scope: !5639)
!5673 = !DILocation(line: 1146, column: 5, scope: !5639)
!5674 = !DILocation(line: 1147, column: 2, scope: !5639)
!5675 = !DILocation(line: 1151, column: 1, scope: !5639)
!5676 = distinct !DISubprogram(name: "cffts3_gpu_kernel_1", linkageName: "_Z19cffts3_gpu_kernel_1P8dcomplexS0_", scope: !3, file: !3, line: 1258, type: !1153, scopeLine: 1259, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !1051)
!5677 = !DILocalVariable(name: "x_in", arg: 1, scope: !5676, file: !3, line: 1258, type: !98)
!5678 = !DILocation(line: 1258, column: 46, scope: !5676)
!5679 = !DILocalVariable(name: "y0", arg: 2, scope: !5676, file: !3, line: 1259, type: !98)
!5680 = !DILocation(line: 1259, column: 12, scope: !5676)
!5681 = !DILocation(line: 1259, column: 17, scope: !5676)
!5682 = !DILocation(line: 1266, column: 1, scope: !5676)
!5683 = distinct !DISubprogram(name: "cffts3_gpu_kernel_2", linkageName: "_Z19cffts3_gpu_kernel_2iP8dcomplexS0_S0_", scope: !3, file: !3, line: 1273, type: !1324, scopeLine: 1276, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !1051)
!5684 = !DILocalVariable(name: "is", arg: 1, scope: !5683, file: !3, line: 1273, type: !1326)
!5685 = !DILocation(line: 1273, column: 47, scope: !5683)
!5686 = !DILocalVariable(name: "gty1", arg: 2, scope: !5683, file: !3, line: 1274, type: !98)
!5687 = !DILocation(line: 1274, column: 12, scope: !5683)
!5688 = !DILocalVariable(name: "gty2", arg: 3, scope: !5683, file: !3, line: 1275, type: !98)
!5689 = !DILocation(line: 1275, column: 12, scope: !5683)
!5690 = !DILocalVariable(name: "u_device", arg: 4, scope: !5683, file: !3, line: 1276, type: !98)
!5691 = !DILocation(line: 1276, column: 12, scope: !5683)
!5692 = !DILocation(line: 1276, column: 23, scope: !5683)
!5693 = !DILocation(line: 1289, column: 1, scope: !5683)
!5694 = distinct !DISubprogram(name: "cffts3_gpu_kernel_3", linkageName: "_Z19cffts3_gpu_kernel_3P8dcomplexS0_", scope: !3, file: !3, line: 1298, type: !1153, scopeLine: 1299, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !1051)
!5695 = !DILocalVariable(name: "x_out", arg: 1, scope: !5694, file: !3, line: 1298, type: !98)
!5696 = !DILocation(line: 1298, column: 46, scope: !5694)
!5697 = !DILocalVariable(name: "y0", arg: 2, scope: !5694, file: !3, line: 1299, type: !98)
!5698 = !DILocation(line: 1299, column: 12, scope: !5694)
!5699 = !DILocation(line: 1299, column: 17, scope: !5694)
!5700 = !DILocation(line: 1306, column: 1, scope: !5694)
!5701 = distinct !DISubprogram(name: "cffts2_gpu_kernel_1", linkageName: "_Z19cffts2_gpu_kernel_1P8dcomplexS0_", scope: !3, file: !3, line: 969, type: !1153, scopeLine: 970, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !1051)
!5702 = !DILocalVariable(name: "x_in", arg: 1, scope: !5701, file: !3, line: 969, type: !98)
!5703 = !DILocation(line: 969, column: 46, scope: !5701)
!5704 = !DILocalVariable(name: "y0", arg: 2, scope: !5701, file: !3, line: 970, type: !98)
!5705 = !DILocation(line: 970, column: 12, scope: !5701)
!5706 = !DILocation(line: 970, column: 17, scope: !5701)
!5707 = !DILocation(line: 977, column: 1, scope: !5701)
!5708 = distinct !DISubprogram(name: "cffts2_gpu_kernel_2", linkageName: "_Z19cffts2_gpu_kernel_2iP8dcomplexS0_S0_", scope: !3, file: !3, line: 984, type: !1324, scopeLine: 987, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !1051)
!5709 = !DILocalVariable(name: "is", arg: 1, scope: !5708, file: !3, line: 984, type: !1326)
!5710 = !DILocation(line: 984, column: 47, scope: !5708)
!5711 = !DILocalVariable(name: "gty1", arg: 2, scope: !5708, file: !3, line: 985, type: !98)
!5712 = !DILocation(line: 985, column: 12, scope: !5708)
!5713 = !DILocalVariable(name: "gty2", arg: 3, scope: !5708, file: !3, line: 986, type: !98)
!5714 = !DILocation(line: 986, column: 12, scope: !5708)
!5715 = !DILocalVariable(name: "u_device", arg: 4, scope: !5708, file: !3, line: 987, type: !98)
!5716 = !DILocation(line: 987, column: 12, scope: !5708)
!5717 = !DILocation(line: 987, column: 23, scope: !5708)
!5718 = !DILocation(line: 1092, column: 1, scope: !5708)
!5719 = distinct !DISubprogram(name: "cffts2_gpu_kernel_3", linkageName: "_Z19cffts2_gpu_kernel_3P8dcomplexS0_", scope: !3, file: !3, line: 1101, type: !1153, scopeLine: 1102, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !1051)
!5720 = !DILocalVariable(name: "x_out", arg: 1, scope: !5719, file: !3, line: 1101, type: !98)
!5721 = !DILocation(line: 1101, column: 46, scope: !5719)
!5722 = !DILocalVariable(name: "y0", arg: 2, scope: !5719, file: !3, line: 1102, type: !98)
!5723 = !DILocation(line: 1102, column: 12, scope: !5719)
!5724 = !DILocation(line: 1102, column: 17, scope: !5719)
!5725 = !DILocation(line: 1109, column: 1, scope: !5719)
!5726 = distinct !DISubprogram(name: "cffts1_gpu_kernel_1", linkageName: "_Z19cffts1_gpu_kernel_1P8dcomplexS0_", scope: !3, file: !3, line: 774, type: !1153, scopeLine: 775, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !1051)
!5727 = !DILocalVariable(name: "x_in", arg: 1, scope: !5726, file: !3, line: 774, type: !98)
!5728 = !DILocation(line: 774, column: 46, scope: !5726)
!5729 = !DILocalVariable(name: "y0", arg: 2, scope: !5726, file: !3, line: 775, type: !98)
!5730 = !DILocation(line: 775, column: 12, scope: !5726)
!5731 = !DILocation(line: 775, column: 17, scope: !5726)
!5732 = !DILocation(line: 785, column: 1, scope: !5726)
!5733 = distinct !DISubprogram(name: "cffts1_gpu_kernel_2", linkageName: "_Z19cffts1_gpu_kernel_2iP8dcomplexS0_S0_", scope: !3, file: !3, line: 792, type: !1324, scopeLine: 795, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !1051)
!5734 = !DILocalVariable(name: "is", arg: 1, scope: !5733, file: !3, line: 792, type: !1326)
!5735 = !DILocation(line: 792, column: 47, scope: !5733)
!5736 = !DILocalVariable(name: "gty1", arg: 2, scope: !5733, file: !3, line: 793, type: !98)
!5737 = !DILocation(line: 793, column: 12, scope: !5733)
!5738 = !DILocalVariable(name: "gty2", arg: 3, scope: !5733, file: !3, line: 794, type: !98)
!5739 = !DILocation(line: 794, column: 12, scope: !5733)
!5740 = !DILocalVariable(name: "u_device", arg: 4, scope: !5733, file: !3, line: 795, type: !98)
!5741 = !DILocation(line: 795, column: 12, scope: !5733)
!5742 = !DILocation(line: 795, column: 23, scope: !5733)
!5743 = !DILocation(line: 898, column: 1, scope: !5733)
!5744 = distinct !DISubprogram(name: "cffts1_gpu_kernel_3", linkageName: "_Z19cffts1_gpu_kernel_3P8dcomplexS0_", scope: !3, file: !3, line: 907, type: !1153, scopeLine: 908, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !1051)
!5745 = !DILocalVariable(name: "x_out", arg: 1, scope: !5744, file: !3, line: 907, type: !98)
!5746 = !DILocation(line: 907, column: 46, scope: !5744)
!5747 = !DILocalVariable(name: "y0", arg: 2, scope: !5744, file: !3, line: 908, type: !98)
!5748 = !DILocation(line: 908, column: 12, scope: !5744)
!5749 = !DILocation(line: 908, column: 17, scope: !5744)
!5750 = !DILocation(line: 918, column: 1, scope: !5744)
!5751 = distinct !DISubprogram(name: "ilog2", linkageName: "_ZL5ilog2i", scope: !3, file: !3, line: 1510, type: !304, scopeLine: 1510, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1051)
!5752 = !DILocalVariable(name: "n", arg: 1, scope: !5751, file: !3, line: 1510, type: !97)
!5753 = !DILocation(line: 1510, column: 22, scope: !5751)
!5754 = !DILocalVariable(name: "nn", scope: !5751, file: !3, line: 1511, type: !97)
!5755 = !DILocation(line: 1511, column: 6, scope: !5751)
!5756 = !DILocalVariable(name: "lg", scope: !5751, file: !3, line: 1511, type: !97)
!5757 = !DILocation(line: 1511, column: 10, scope: !5751)
!5758 = !DILocation(line: 1512, column: 5, scope: !5759)
!5759 = distinct !DILexicalBlock(scope: !5751, file: !3, line: 1512, column: 5)
!5760 = !DILocation(line: 1512, column: 6, scope: !5759)
!5761 = !DILocation(line: 1512, column: 5, scope: !5751)
!5762 = !DILocation(line: 1513, column: 3, scope: !5763)
!5763 = distinct !DILexicalBlock(scope: !5759, file: !3, line: 1512, column: 10)
!5764 = !DILocation(line: 1515, column: 5, scope: !5751)
!5765 = !DILocation(line: 1516, column: 5, scope: !5751)
!5766 = !DILocation(line: 1517, column: 2, scope: !5751)
!5767 = !DILocation(line: 1517, column: 8, scope: !5751)
!5768 = !DILocation(line: 1517, column: 11, scope: !5751)
!5769 = !DILocation(line: 1517, column: 10, scope: !5751)
!5770 = !DILocation(line: 1518, column: 8, scope: !5771)
!5771 = distinct !DILexicalBlock(scope: !5751, file: !3, line: 1517, column: 13)
!5772 = !DILocation(line: 1518, column: 11, scope: !5771)
!5773 = !DILocation(line: 1518, column: 6, scope: !5771)
!5774 = !DILocation(line: 1519, column: 5, scope: !5771)
!5775 = distinct !{!5775, !5766, !5776}
!5776 = !DILocation(line: 1520, column: 2, scope: !5751)
!5777 = !DILocation(line: 1521, column: 9, scope: !5751)
!5778 = !DILocation(line: 1521, column: 2, scope: !5751)
!5779 = !DILocation(line: 1522, column: 1, scope: !5751)
!5780 = distinct !DISubprogram(name: "ipow46", linkageName: "_ZL6ipow46diPd", scope: !3, file: !3, line: 1568, type: !3712, scopeLine: 1570, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1051)
!5781 = !DILocalVariable(name: "a", arg: 1, scope: !5780, file: !3, line: 1568, type: !104)
!5782 = !DILocation(line: 1568, column: 27, scope: !5780)
!5783 = !DILocalVariable(name: "exponent", arg: 2, scope: !5780, file: !3, line: 1569, type: !97)
!5784 = !DILocation(line: 1569, column: 7, scope: !5780)
!5785 = !DILocalVariable(name: "result", arg: 3, scope: !5780, file: !3, line: 1570, type: !106)
!5786 = !DILocation(line: 1570, column: 11, scope: !5780)
!5787 = !DILocalVariable(name: "q", scope: !5780, file: !3, line: 1571, type: !104)
!5788 = !DILocation(line: 1571, column: 9, scope: !5780)
!5789 = !DILocalVariable(name: "r", scope: !5780, file: !3, line: 1571, type: !104)
!5790 = !DILocation(line: 1571, column: 12, scope: !5780)
!5791 = !DILocalVariable(name: "n", scope: !5780, file: !3, line: 1572, type: !97)
!5792 = !DILocation(line: 1572, column: 6, scope: !5780)
!5793 = !DILocalVariable(name: "n2", scope: !5780, file: !3, line: 1572, type: !97)
!5794 = !DILocation(line: 1572, column: 9, scope: !5780)
!5795 = !DILocation(line: 1580, column: 3, scope: !5780)
!5796 = !DILocation(line: 1580, column: 10, scope: !5780)
!5797 = !DILocation(line: 1581, column: 5, scope: !5798)
!5798 = distinct !DILexicalBlock(scope: !5780, file: !3, line: 1581, column: 5)
!5799 = !DILocation(line: 1581, column: 13, scope: !5798)
!5800 = !DILocation(line: 1581, column: 5, scope: !5780)
!5801 = !DILocation(line: 1581, column: 18, scope: !5802)
!5802 = distinct !DILexicalBlock(scope: !5798, file: !3, line: 1581, column: 17)
!5803 = !DILocation(line: 1582, column: 6, scope: !5780)
!5804 = !DILocation(line: 1582, column: 4, scope: !5780)
!5805 = !DILocation(line: 1583, column: 4, scope: !5780)
!5806 = !DILocation(line: 1584, column: 6, scope: !5780)
!5807 = !DILocation(line: 1584, column: 4, scope: !5780)
!5808 = !DILocation(line: 1585, column: 2, scope: !5780)
!5809 = !DILocation(line: 1585, column: 8, scope: !5780)
!5810 = !DILocation(line: 1585, column: 9, scope: !5780)
!5811 = !DILocation(line: 1586, column: 8, scope: !5812)
!5812 = distinct !DILexicalBlock(scope: !5780, file: !3, line: 1585, column: 12)
!5813 = !DILocation(line: 1586, column: 9, scope: !5812)
!5814 = !DILocation(line: 1586, column: 6, scope: !5812)
!5815 = !DILocation(line: 1587, column: 6, scope: !5816)
!5816 = distinct !DILexicalBlock(scope: !5812, file: !3, line: 1587, column: 6)
!5817 = !DILocation(line: 1587, column: 8, scope: !5816)
!5818 = !DILocation(line: 1587, column: 12, scope: !5816)
!5819 = !DILocation(line: 1587, column: 10, scope: !5816)
!5820 = !DILocation(line: 1587, column: 6, scope: !5812)
!5821 = !DILocation(line: 1588, column: 15, scope: !5822)
!5822 = distinct !DILexicalBlock(scope: !5816, file: !3, line: 1587, column: 14)
!5823 = !DILocation(line: 1588, column: 4, scope: !5822)
!5824 = !DILocation(line: 1589, column: 8, scope: !5822)
!5825 = !DILocation(line: 1589, column: 6, scope: !5822)
!5826 = !DILocation(line: 1590, column: 3, scope: !5822)
!5827 = !DILocation(line: 1591, column: 15, scope: !5828)
!5828 = distinct !DILexicalBlock(scope: !5816, file: !3, line: 1590, column: 8)
!5829 = !DILocation(line: 1591, column: 4, scope: !5828)
!5830 = !DILocation(line: 1592, column: 8, scope: !5828)
!5831 = !DILocation(line: 1592, column: 9, scope: !5828)
!5832 = !DILocation(line: 1592, column: 6, scope: !5828)
!5833 = distinct !{!5833, !5808, !5834}
!5834 = !DILocation(line: 1594, column: 2, scope: !5780)
!5835 = !DILocation(line: 1595, column: 13, scope: !5780)
!5836 = !DILocation(line: 1595, column: 2, scope: !5780)
!5837 = !DILocation(line: 1596, column: 12, scope: !5780)
!5838 = !DILocation(line: 1596, column: 3, scope: !5780)
!5839 = !DILocation(line: 1596, column: 10, scope: !5780)
!5840 = !DILocation(line: 1597, column: 1, scope: !5780)
!5841 = distinct !DISubprogram(name: "compute_initial_conditions_gpu_kernel", linkageName: "_Z37compute_initial_conditions_gpu_kernelP8dcomplexPd", scope: !3, file: !3, line: 1416, type: !3474, scopeLine: 1417, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !1051)
!5842 = !DILocalVariable(name: "u0", arg: 1, scope: !5841, file: !3, line: 1416, type: !98)
!5843 = !DILocation(line: 1416, column: 64, scope: !5841)
!5844 = !DILocalVariable(name: "starts", arg: 2, scope: !5841, file: !3, line: 1417, type: !106)
!5845 = !DILocation(line: 1417, column: 10, scope: !5841)
!5846 = !DILocation(line: 1417, column: 19, scope: !5841)
!5847 = !DILocation(line: 1426, column: 1, scope: !5841)
!5848 = distinct !DISubprogram(name: "compute_indexmap_gpu_kernel", linkageName: "_Z27compute_indexmap_gpu_kernelPd", scope: !3, file: !3, line: 1365, type: !3387, scopeLine: 1365, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !1051)
!5849 = !DILocalVariable(name: "twiddle", arg: 1, scope: !5848, file: !3, line: 1365, type: !106)
!5850 = !DILocation(line: 1365, column: 52, scope: !5848)
!5851 = !DILocation(line: 1365, column: 62, scope: !5848)
!5852 = !DILocation(line: 1385, column: 1, scope: !5848)
!5853 = distinct !DISubprogram(name: "init_ui_gpu_kernel", linkageName: "_Z18init_ui_gpu_kernelP8dcomplexS0_Pd", scope: !3, file: !3, line: 1554, type: !3642, scopeLine: 1556, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !1051)
!5854 = !DILocalVariable(name: "u0", arg: 1, scope: !5853, file: !3, line: 1554, type: !98)
!5855 = !DILocation(line: 1554, column: 45, scope: !5853)
!5856 = !DILocalVariable(name: "u1", arg: 2, scope: !5853, file: !3, line: 1555, type: !98)
!5857 = !DILocation(line: 1555, column: 12, scope: !5853)
!5858 = !DILocalVariable(name: "twiddle", arg: 3, scope: !5853, file: !3, line: 1556, type: !106)
!5859 = !DILocation(line: 1556, column: 10, scope: !5853)
!5860 = !DILocation(line: 1556, column: 20, scope: !5853)
!5861 = !DILocation(line: 1566, column: 1, scope: !5853)
!5862 = distinct !DISubprogram(name: "cudaMalloc<dcomplex>", linkageName: "_ZL10cudaMallocI8dcomplexE9cudaErrorPPT_m", scope: !5863, file: !5863, line: 490, type: !5864, scopeLine: 494, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, templateParams: !5869, retainedNodes: !1051)
!5863 = !DIFile(filename: "/usr/local/cuda/include/cuda_runtime.h", directory: "")
!5864 = !DISubroutineType(types: !5865)
!5865 = !{!5866, !5867, !131}
!5866 = !DIDerivedType(tag: DW_TAG_typedef, name: "cudaError_t", file: !6, line: 1419, baseType: !14)
!5867 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !5868, size: 64)
!5868 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !101, size: 64)
!5869 = !{!5870}
!5870 = !DITemplateTypeParameter(name: "T", type: !101)
!5871 = !DILocalVariable(name: "devPtr", arg: 1, scope: !5862, file: !5863, line: 491, type: !5867)
!5872 = !DILocation(line: 491, column: 12, scope: !5862)
!5873 = !DILocalVariable(name: "size", arg: 2, scope: !5862, file: !5863, line: 492, type: !131)
!5874 = !DILocation(line: 492, column: 12, scope: !5862)
!5875 = !DILocation(line: 495, column: 38, scope: !5862)
!5876 = !DILocation(line: 495, column: 23, scope: !5862)
!5877 = !DILocation(line: 495, column: 46, scope: !5862)
!5878 = !DILocation(line: 495, column: 10, scope: !5862)
!5879 = !DILocation(line: 495, column: 3, scope: !5862)
!5880 = distinct !DISubprogram(name: "cudaMalloc<double>", linkageName: "_ZL10cudaMallocIdE9cudaErrorPPT_m", scope: !5863, file: !5863, line: 490, type: !5881, scopeLine: 494, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, templateParams: !5884, retainedNodes: !1051)
!5881 = !DISubroutineType(types: !5882)
!5882 = !{!5866, !5883, !131}
!5883 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !106, size: 64)
!5884 = !{!5885}
!5885 = !DITemplateTypeParameter(name: "T", type: !104)
!5886 = !DILocalVariable(name: "devPtr", arg: 1, scope: !5880, file: !5863, line: 491, type: !5883)
!5887 = !DILocation(line: 491, column: 12, scope: !5880)
!5888 = !DILocalVariable(name: "size", arg: 2, scope: !5880, file: !5863, line: 492, type: !131)
!5889 = !DILocation(line: 492, column: 12, scope: !5880)
!5890 = !DILocation(line: 495, column: 38, scope: !5880)
!5891 = !DILocation(line: 495, column: 23, scope: !5880)
!5892 = !DILocation(line: 495, column: 46, scope: !5880)
!5893 = !DILocation(line: 495, column: 10, scope: !5880)
!5894 = !DILocation(line: 495, column: 3, scope: !5880)
