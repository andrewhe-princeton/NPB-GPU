; ModuleID = 'ft_cpu.bc'
source_filename = "llvm-link-cudafe"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.dcomplex = type { double, double }
%struct.cudaDeviceProp = type { [256 x i8], i64, i64, i32, i32, i64, i32, [3 x i32], [3 x i32], i32, i64, i32, i32, i64, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, [2 x i32], [2 x i32], [3 x i32], [2 x i32], [3 x i32], [3 x i32], i32, [2 x i32], [3 x i32], [2 x i32], i32, [2 x i32], [3 x i32], [2 x i32], [3 x i32], i32, [2 x i32], i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i64, i32, i32, i32, i32 }
%struct.dim3 = type { i32, i32, i32 }

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
@extern_share_data_shared = internal thread_local global [1024 x double] zeroinitializer

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #0

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: convergent noinline nounwind
define dso_local i32 @_Z12ilog2_devicei(i32 %n) #2 !dbg !1152 {
entry:
  call void @llvm.dbg.value(metadata i32 %n, metadata !1153, metadata !DIExpression()), !dbg !1154
  %cmp = icmp eq i32 %n, 1, !dbg !1155
  br i1 %cmp, label %if.then, label %if.end, !dbg !1157

if.then:                                          ; preds = %entry
  br label %return, !dbg !1158

if.end:                                           ; preds = %entry
  call void @llvm.dbg.value(metadata i32 1, metadata !1160, metadata !DIExpression()), !dbg !1154
  call void @llvm.dbg.value(metadata i32 2, metadata !1161, metadata !DIExpression()), !dbg !1154
  br label %while.cond, !dbg !1162

while.cond:                                       ; preds = %while.body, %if.end
  %nn.0 = phi i32 [ 2, %if.end ], [ %shl, %while.body ], !dbg !1154
  %lg.0 = phi i32 [ 1, %if.end ], [ %inc, %while.body ], !dbg !1154
  call void @llvm.dbg.value(metadata i32 %lg.0, metadata !1160, metadata !DIExpression()), !dbg !1154
  call void @llvm.dbg.value(metadata i32 %nn.0, metadata !1161, metadata !DIExpression()), !dbg !1154
  %cmp1 = icmp slt i32 %nn.0, %n, !dbg !1163
  br i1 %cmp1, label %while.body, label %while.end, !dbg !1162

while.body:                                       ; preds = %while.cond
  %shl = shl i32 %nn.0, 1, !dbg !1164
  call void @llvm.dbg.value(metadata i32 %shl, metadata !1161, metadata !DIExpression()), !dbg !1154
  %inc = add nuw nsw i32 %lg.0, 1, !dbg !1166
  call void @llvm.dbg.value(metadata i32 %inc, metadata !1160, metadata !DIExpression()), !dbg !1154
  br label %while.cond, !dbg !1162, !llvm.loop !1167

while.end:                                        ; preds = %while.cond
  %lg.0.lcssa = phi i32 [ %lg.0, %while.cond ], !dbg !1154
  call void @llvm.dbg.value(metadata i32 %lg.0.lcssa, metadata !1160, metadata !DIExpression()), !dbg !1154
  br label %return, !dbg !1169

return:                                           ; preds = %while.end, %if.then
  %retval.0 = phi i32 [ 0, %if.then ], [ %lg.0.lcssa, %while.end ], !dbg !1154
  ret i32 %retval.0, !dbg !1170
}

; Function Attrs: convergent noinline nounwind
define dso_local void @_Z23cffts3_gpu_cfftz_deviceiiiP8dcomplexS0_S0_ii(i32 %is, i32 %m, i32 %n, %struct.dcomplex* %x, %struct.dcomplex* %y, %struct.dcomplex* %u_device, i32 %index_arg, i32 %size_arg) #2 !dbg !1171 {
entry:
  call void @llvm.dbg.value(metadata i32 %is, metadata !1175, metadata !DIExpression()), !dbg !1176
  call void @llvm.dbg.value(metadata i32 %m, metadata !1177, metadata !DIExpression()), !dbg !1176
  call void @llvm.dbg.value(metadata i32 %n, metadata !1178, metadata !DIExpression()), !dbg !1176
  call void @llvm.dbg.value(metadata %struct.dcomplex* %x, metadata !1179, metadata !DIExpression()), !dbg !1176
  call void @llvm.dbg.value(metadata %struct.dcomplex* %y, metadata !1180, metadata !DIExpression()), !dbg !1176
  call void @llvm.dbg.value(metadata %struct.dcomplex* %u_device, metadata !1181, metadata !DIExpression()), !dbg !1176
  call void @llvm.dbg.value(metadata i32 %index_arg, metadata !1182, metadata !DIExpression()), !dbg !1176
  call void @llvm.dbg.value(metadata i32 %size_arg, metadata !1183, metadata !DIExpression()), !dbg !1176
  call void @llvm.dbg.value(metadata i32 1, metadata !1184, metadata !DIExpression()), !dbg !1176
  br label %for.cond, !dbg !1185

for.cond:                                         ; preds = %for.inc, %entry
  %l.0 = phi i32 [ 1, %entry ], [ %add2, %for.inc ], !dbg !1187
  call void @llvm.dbg.value(metadata i32 %l.0, metadata !1184, metadata !DIExpression()), !dbg !1176
  %cmp = icmp sle i32 %l.0, %m, !dbg !1188
  br i1 %cmp, label %for.body, label %for.end.loopexit, !dbg !1190

for.body:                                         ; preds = %for.cond
  call void @_Z23cffts3_gpu_fftz2_deviceiiiiP8dcomplexS0_S0_ii(i32 %is, i32 %l.0, i32 %m, i32 %n, %struct.dcomplex* %u_device, %struct.dcomplex* %x, %struct.dcomplex* %y, i32 %index_arg, i32 %size_arg) #4, !dbg !1191
  %cmp1 = icmp eq i32 %l.0, %m, !dbg !1193
  br i1 %cmp1, label %if.then, label %if.end, !dbg !1195

if.then:                                          ; preds = %for.body
  br label %for.end, !dbg !1196

if.end:                                           ; preds = %for.body
  %add = add nuw nsw i32 %l.0, 1, !dbg !1198
  call void @_Z23cffts3_gpu_fftz2_deviceiiiiP8dcomplexS0_S0_ii(i32 %is, i32 %add, i32 %m, i32 %n, %struct.dcomplex* %u_device, %struct.dcomplex* %y, %struct.dcomplex* %x, i32 %index_arg, i32 %size_arg) #4, !dbg !1199
  br label %for.inc, !dbg !1200

for.inc:                                          ; preds = %if.end
  %add2 = add nuw nsw i32 %l.0, 2, !dbg !1201
  call void @llvm.dbg.value(metadata i32 %add2, metadata !1184, metadata !DIExpression()), !dbg !1176
  br label %for.cond, !dbg !1202, !llvm.loop !1203

for.end.loopexit:                                 ; preds = %for.cond
  br label %for.end, !dbg !1205

for.end:                                          ; preds = %for.end.loopexit, %if.then
  %rem = srem i32 %m, 2, !dbg !1205
  %cmp3 = icmp eq i32 %rem, 1, !dbg !1207
  br i1 %cmp3, label %if.then4, label %if.end25, !dbg !1208

if.then4:                                         ; preds = %for.end
  call void @llvm.dbg.value(metadata i32 0, metadata !1209, metadata !DIExpression()), !dbg !1176
  %0 = sext i32 %n to i64, !dbg !1210
  %1 = sext i32 %size_arg to i64, !dbg !1210
  %2 = sext i32 %index_arg to i64, !dbg !1210
  %3 = sext i32 %size_arg to i64, !dbg !1210
  %4 = sext i32 %index_arg to i64, !dbg !1210
  %5 = sext i32 %size_arg to i64, !dbg !1210
  %6 = sext i32 %index_arg to i64, !dbg !1210
  %7 = sext i32 %size_arg to i64, !dbg !1210
  %8 = sext i32 %index_arg to i64, !dbg !1210
  br label %for.cond5, !dbg !1210

for.cond5:                                        ; preds = %for.inc23, %if.then4
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc23 ], [ 0, %if.then4 ], !dbg !1213
  call void @llvm.dbg.value(metadata i64 %indvars.iv, metadata !1209, metadata !DIExpression()), !dbg !1176
  %cmp6 = icmp slt i64 %indvars.iv, %0, !dbg !1214
  br i1 %cmp6, label %for.body7, label %for.end24, !dbg !1216

for.body7:                                        ; preds = %for.cond5
  %9 = mul nsw i64 %indvars.iv, %1, !dbg !1217
  %10 = add nsw i64 %9, %2, !dbg !1219
  %arrayidx = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %y, i64 %10, !dbg !1220
  %real = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx, i32 0, i32 0, !dbg !1221
  %11 = load double, double* %real, align 8, !dbg !1221
  %12 = mul nsw i64 %indvars.iv, %3, !dbg !1222
  %13 = add nsw i64 %12, %4, !dbg !1223
  %arrayidx12 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %x, i64 %13, !dbg !1224
  %real13 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx12, i32 0, i32 0, !dbg !1225
  store double %11, double* %real13, align 8, !dbg !1226
  %14 = mul nsw i64 %indvars.iv, %5, !dbg !1227
  %15 = add nsw i64 %14, %6, !dbg !1228
  %arrayidx17 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %y, i64 %15, !dbg !1229
  %imag = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx17, i32 0, i32 1, !dbg !1230
  %16 = load double, double* %imag, align 8, !dbg !1230
  %17 = mul nsw i64 %indvars.iv, %7, !dbg !1231
  %18 = add nsw i64 %17, %8, !dbg !1232
  %arrayidx21 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %x, i64 %18, !dbg !1233
  %imag22 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx21, i32 0, i32 1, !dbg !1234
  store double %16, double* %imag22, align 8, !dbg !1235
  br label %for.inc23, !dbg !1236

for.inc23:                                        ; preds = %for.body7
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !1237
  call void @llvm.dbg.value(metadata i32 undef, metadata !1209, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !1176
  br label %for.cond5, !dbg !1238, !llvm.loop !1239

for.end24:                                        ; preds = %for.cond5
  br label %if.end25, !dbg !1241

if.end25:                                         ; preds = %for.end24, %for.end
  ret void, !dbg !1242
}

; Function Attrs: convergent noinline nounwind
define dso_local void @_Z23cffts3_gpu_fftz2_deviceiiiiP8dcomplexS0_S0_ii(i32 %is, i32 %l, i32 %m, i32 %n, %struct.dcomplex* %u, %struct.dcomplex* %x, %struct.dcomplex* %y, i32 %index_arg, i32 %size_arg) #2 !dbg !1243 {
entry:
  %u1 = alloca %struct.dcomplex, align 8
  call void @llvm.dbg.value(metadata i32 %is, metadata !1246, metadata !DIExpression()), !dbg !1247
  call void @llvm.dbg.value(metadata i32 %l, metadata !1248, metadata !DIExpression()), !dbg !1247
  call void @llvm.dbg.value(metadata i32 %m, metadata !1249, metadata !DIExpression()), !dbg !1247
  call void @llvm.dbg.value(metadata i32 %n, metadata !1250, metadata !DIExpression()), !dbg !1247
  call void @llvm.dbg.value(metadata %struct.dcomplex* %u, metadata !1251, metadata !DIExpression()), !dbg !1247
  call void @llvm.dbg.value(metadata %struct.dcomplex* %x, metadata !1252, metadata !DIExpression()), !dbg !1247
  call void @llvm.dbg.value(metadata %struct.dcomplex* %y, metadata !1253, metadata !DIExpression()), !dbg !1247
  call void @llvm.dbg.value(metadata i32 %index_arg, metadata !1254, metadata !DIExpression()), !dbg !1247
  call void @llvm.dbg.value(metadata i32 %size_arg, metadata !1255, metadata !DIExpression()), !dbg !1247
  call void @llvm.dbg.declare(metadata %struct.dcomplex* %u1, metadata !1256, metadata !DIExpression()), !dbg !1257
  %div = sdiv i32 %n, 2, !dbg !1258
  call void @llvm.dbg.value(metadata i32 %div, metadata !1259, metadata !DIExpression()), !dbg !1247
  %sub = sub nsw i32 %l, 1, !dbg !1260
  %shl = shl i32 1, %sub, !dbg !1261
  call void @llvm.dbg.value(metadata i32 %shl, metadata !1262, metadata !DIExpression()), !dbg !1247
  %sub1 = sub nsw i32 %m, %l, !dbg !1263
  %shl2 = shl i32 1, %sub1, !dbg !1264
  call void @llvm.dbg.value(metadata i32 %shl2, metadata !1265, metadata !DIExpression()), !dbg !1247
  %mul = mul nsw i32 2, %shl, !dbg !1266
  call void @llvm.dbg.value(metadata i32 %mul, metadata !1267, metadata !DIExpression()), !dbg !1247
  call void @llvm.dbg.value(metadata i32 %shl2, metadata !1268, metadata !DIExpression()), !dbg !1247
  call void @llvm.dbg.value(metadata i32 0, metadata !1269, metadata !DIExpression()), !dbg !1247
  %0 = sext i32 %shl to i64, !dbg !1270
  %1 = sext i32 %size_arg to i64, !dbg !1270
  %2 = sext i32 %index_arg to i64, !dbg !1270
  %3 = sext i32 %size_arg to i64, !dbg !1270
  %4 = sext i32 %index_arg to i64, !dbg !1270
  %5 = sext i32 %size_arg to i64, !dbg !1270
  %6 = sext i32 %index_arg to i64, !dbg !1270
  %7 = sext i32 %size_arg to i64, !dbg !1270
  %8 = sext i32 %index_arg to i64, !dbg !1270
  %9 = sext i32 %size_arg to i64, !dbg !1270
  %10 = sext i32 %index_arg to i64, !dbg !1270
  %11 = sext i32 %size_arg to i64, !dbg !1270
  %12 = sext i32 %index_arg to i64, !dbg !1270
  %13 = sext i32 %size_arg to i64, !dbg !1270
  %14 = sext i32 %index_arg to i64, !dbg !1270
  %15 = sext i32 %size_arg to i64, !dbg !1270
  %16 = sext i32 %index_arg to i64, !dbg !1270
  %17 = sext i32 %shl2 to i64, !dbg !1270
  %18 = sext i32 %shl to i64, !dbg !1270
  %19 = sext i32 %div to i64, !dbg !1270
  %20 = sext i32 %mul to i64, !dbg !1270
  %21 = sext i32 %shl to i64, !dbg !1270
  %22 = sext i32 %shl2 to i64, !dbg !1270
  %23 = sext i32 %shl2 to i64, !dbg !1270
  %24 = sext i32 %shl2 to i64, !dbg !1270
  %25 = sext i32 %shl2 to i64, !dbg !1270
  br label %for.cond, !dbg !1270

for.cond:                                         ; preds = %for.inc91, %entry
  %indvars.iv25 = phi i64 [ %indvars.iv.next26, %for.inc91 ], [ 0, %entry ], !dbg !1272
  call void @llvm.dbg.value(metadata i64 %indvars.iv25, metadata !1269, metadata !DIExpression()), !dbg !1247
  %cmp = icmp slt i64 %indvars.iv25, %17, !dbg !1273
  br i1 %cmp, label %for.body, label %for.end93, !dbg !1275

for.body:                                         ; preds = %for.cond
  %26 = mul nsw i64 %indvars.iv25, %18, !dbg !1276
  %27 = add nsw i64 %26, %19, !dbg !1278
  %28 = mul nsw i64 %indvars.iv25, %20, !dbg !1279
  %29 = add nsw i64 %28, %21, !dbg !1280
  %cmp6 = icmp sge i32 %is, 1, !dbg !1281
  br i1 %cmp6, label %if.then, label %if.else, !dbg !1283

if.then:                                          ; preds = %for.body
  %30 = add nsw i64 %22, %indvars.iv25, !dbg !1284
  %arrayidx = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %u, i64 %30, !dbg !1286
  %real = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx, i32 0, i32 0, !dbg !1287
  %31 = load double, double* %real, align 8, !dbg !1287
  %real8 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %u1, i32 0, i32 0, !dbg !1288
  store double %31, double* %real8, align 8, !dbg !1289
  %32 = add nsw i64 %23, %indvars.iv25, !dbg !1290
  %arrayidx11 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %u, i64 %32, !dbg !1291
  %imag = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx11, i32 0, i32 1, !dbg !1292
  %33 = load double, double* %imag, align 8, !dbg !1292
  %imag12 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %u1, i32 0, i32 1, !dbg !1293
  store double %33, double* %imag12, align 8, !dbg !1294
  br label %if.end, !dbg !1295

if.else:                                          ; preds = %for.body
  %34 = add nsw i64 %24, %indvars.iv25, !dbg !1296
  %arrayidx15 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %u, i64 %34, !dbg !1298
  %real16 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx15, i32 0, i32 0, !dbg !1299
  %35 = load double, double* %real16, align 8, !dbg !1299
  %real17 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %u1, i32 0, i32 0, !dbg !1300
  store double %35, double* %real17, align 8, !dbg !1301
  %36 = add nsw i64 %25, %indvars.iv25, !dbg !1302
  %arrayidx20 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %u, i64 %36, !dbg !1303
  %imag21 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx20, i32 0, i32 1, !dbg !1304
  %37 = load double, double* %imag21, align 8, !dbg !1304
  %sub22 = fsub double -0.000000e+00, %37, !dbg !1305
  %imag23 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %u1, i32 0, i32 1, !dbg !1306
  store double %sub22, double* %imag23, align 8, !dbg !1307
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  call void @llvm.dbg.value(metadata i32 0, metadata !1308, metadata !DIExpression()), !dbg !1247
  br label %for.cond24, !dbg !1309

for.cond24:                                       ; preds = %for.inc, %if.end
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %if.end ], !dbg !1311
  call void @llvm.dbg.value(metadata i64 %indvars.iv, metadata !1308, metadata !DIExpression()), !dbg !1247
  %cmp25 = icmp slt i64 %indvars.iv, %0, !dbg !1312
  br i1 %cmp25, label %for.body26, label %for.end, !dbg !1314

for.body26:                                       ; preds = %for.cond24
  %38 = add nsw i64 %26, %indvars.iv, !dbg !1315
  %39 = mul nsw i64 %38, %1, !dbg !1317
  %40 = add nsw i64 %39, %2, !dbg !1318
  %arrayidx31 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %x, i64 %40, !dbg !1319
  %real32 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx31, i32 0, i32 0, !dbg !1320
  %41 = load double, double* %real32, align 8, !dbg !1320
  call void @llvm.dbg.value(metadata double %41, metadata !1321, metadata !DIExpression()), !dbg !1247
  %42 = add nsw i64 %26, %indvars.iv, !dbg !1322
  %43 = mul nsw i64 %42, %3, !dbg !1323
  %44 = add nsw i64 %43, %4, !dbg !1324
  %arrayidx37 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %x, i64 %44, !dbg !1325
  %imag38 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx37, i32 0, i32 1, !dbg !1326
  %45 = load double, double* %imag38, align 8, !dbg !1326
  call void @llvm.dbg.value(metadata double %45, metadata !1327, metadata !DIExpression()), !dbg !1247
  %46 = add nsw i64 %27, %indvars.iv, !dbg !1328
  %47 = mul nsw i64 %46, %5, !dbg !1329
  %48 = add nsw i64 %47, %6, !dbg !1330
  %arrayidx43 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %x, i64 %48, !dbg !1331
  %real44 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx43, i32 0, i32 0, !dbg !1332
  %49 = load double, double* %real44, align 8, !dbg !1332
  call void @llvm.dbg.value(metadata double %49, metadata !1333, metadata !DIExpression()), !dbg !1247
  %50 = add nsw i64 %27, %indvars.iv, !dbg !1334
  %51 = mul nsw i64 %50, %7, !dbg !1335
  %52 = add nsw i64 %51, %8, !dbg !1336
  %arrayidx49 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %x, i64 %52, !dbg !1337
  %imag50 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx49, i32 0, i32 1, !dbg !1338
  %53 = load double, double* %imag50, align 8, !dbg !1338
  call void @llvm.dbg.value(metadata double %53, metadata !1339, metadata !DIExpression()), !dbg !1247
  %add51 = fadd contract double %41, %49, !dbg !1340
  %54 = add nsw i64 %28, %indvars.iv, !dbg !1341
  %55 = mul nsw i64 %54, %9, !dbg !1342
  %56 = add nsw i64 %55, %10, !dbg !1343
  %arrayidx56 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %y, i64 %56, !dbg !1344
  %real57 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx56, i32 0, i32 0, !dbg !1345
  store double %add51, double* %real57, align 8, !dbg !1346
  %add58 = fadd contract double %45, %53, !dbg !1347
  %57 = add nsw i64 %28, %indvars.iv, !dbg !1348
  %58 = mul nsw i64 %57, %11, !dbg !1349
  %59 = add nsw i64 %58, %12, !dbg !1350
  %arrayidx63 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %y, i64 %59, !dbg !1351
  %imag64 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx63, i32 0, i32 1, !dbg !1352
  store double %add58, double* %imag64, align 8, !dbg !1353
  %real65 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %u1, i32 0, i32 0, !dbg !1354
  %60 = load double, double* %real65, align 8, !dbg !1354
  %sub66 = fsub contract double %41, %49, !dbg !1355
  %mul67 = fmul contract double %60, %sub66, !dbg !1356
  %imag68 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %u1, i32 0, i32 1, !dbg !1357
  %61 = load double, double* %imag68, align 8, !dbg !1357
  %sub69 = fsub contract double %45, %53, !dbg !1358
  %mul70 = fmul contract double %61, %sub69, !dbg !1359
  %sub71 = fsub contract double %mul67, %mul70, !dbg !1360
  %62 = add nsw i64 %29, %indvars.iv, !dbg !1361
  %63 = mul nsw i64 %62, %13, !dbg !1362
  %64 = add nsw i64 %63, %14, !dbg !1363
  %arrayidx76 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %y, i64 %64, !dbg !1364
  %real77 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx76, i32 0, i32 0, !dbg !1365
  store double %sub71, double* %real77, align 8, !dbg !1366
  %real78 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %u1, i32 0, i32 0, !dbg !1367
  %65 = load double, double* %real78, align 8, !dbg !1367
  %sub79 = fsub contract double %45, %53, !dbg !1368
  %mul80 = fmul contract double %65, %sub79, !dbg !1369
  %imag81 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %u1, i32 0, i32 1, !dbg !1370
  %66 = load double, double* %imag81, align 8, !dbg !1370
  %sub82 = fsub contract double %41, %49, !dbg !1371
  %mul83 = fmul contract double %66, %sub82, !dbg !1372
  %add84 = fadd contract double %mul80, %mul83, !dbg !1373
  %67 = add nsw i64 %29, %indvars.iv, !dbg !1374
  %68 = mul nsw i64 %67, %15, !dbg !1375
  %69 = add nsw i64 %68, %16, !dbg !1376
  %arrayidx89 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %y, i64 %69, !dbg !1377
  %imag90 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx89, i32 0, i32 1, !dbg !1378
  store double %add84, double* %imag90, align 8, !dbg !1379
  br label %for.inc, !dbg !1380

for.inc:                                          ; preds = %for.body26
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !1381
  call void @llvm.dbg.value(metadata i32 undef, metadata !1308, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !1247
  br label %for.cond24, !dbg !1382, !llvm.loop !1383

for.end:                                          ; preds = %for.cond24
  br label %for.inc91, !dbg !1385

for.inc91:                                        ; preds = %for.end
  %indvars.iv.next26 = add nuw nsw i64 %indvars.iv25, 1, !dbg !1386
  call void @llvm.dbg.value(metadata i32 undef, metadata !1269, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !1247
  br label %for.cond, !dbg !1387, !llvm.loop !1388

for.end93:                                        ; preds = %for.cond
  ret void, !dbg !1390
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1 immarg) #3

; Function Attrs: convergent nounwind
declare void @llvm.nvvm.barrier0() #4

; Function Attrs: convergent noinline nounwind
declare dso_local double @_ZL9atomicAddPdd(double*, double) #5

; Function Attrs: convergent noinline nounwind
declare dso_local i64 @_ZL9atomicCASPyyy(i64*, i64, i64) #2

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.d2i.hi(double) #1

; Function Attrs: nounwind readnone
declare float @llvm.nvvm.fabs.f(float) #1

; Function Attrs: nounwind readnone
declare double @llvm.nvvm.mul.rn.d(double, double) #1

; Function Attrs: nounwind readnone
declare double @llvm.nvvm.add.rn.d(double, double) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.d2i.lo(double) #1

; Function Attrs: nounwind readnone
declare double @llvm.nvvm.fma.rn.d(double, double, double) #1

; Function Attrs: nounwind readnone
declare double @llvm.nvvm.lohi.i2d(i32, i32) #1

; Function Attrs: nounwind readnone
declare double @llvm.nvvm.fabs.d(double) #1

; Function Attrs: convergent noinline nounwind
define dso_local void @_Z13vranlc_deviceiPddS_(i32 %n, double* %x_seed, double %a, double* %y) #2 !dbg !1391 {
entry:
  call void @llvm.dbg.value(metadata i32 %n, metadata !1394, metadata !DIExpression()), !dbg !1395
  call void @llvm.dbg.value(metadata double* %x_seed, metadata !1396, metadata !DIExpression()), !dbg !1395
  call void @llvm.dbg.value(metadata double %a, metadata !1397, metadata !DIExpression()), !dbg !1395
  call void @llvm.dbg.value(metadata double* %y, metadata !1398, metadata !DIExpression()), !dbg !1395
  %mul = fmul contract double 0x3E80000000000000, %a, !dbg !1399
  call void @llvm.dbg.value(metadata double %mul, metadata !1400, metadata !DIExpression()), !dbg !1395
  %conv = fptosi double %mul to i32, !dbg !1401
  %conv1 = sitofp i32 %conv to double, !dbg !1402
  call void @llvm.dbg.value(metadata double %conv1, metadata !1403, metadata !DIExpression()), !dbg !1395
  %mul2 = fmul contract double 0x4160000000000000, %conv1, !dbg !1404
  %sub = fsub contract double %a, %mul2, !dbg !1405
  call void @llvm.dbg.value(metadata double %sub, metadata !1406, metadata !DIExpression()), !dbg !1395
  %0 = load double, double* %x_seed, align 8, !dbg !1407
  call void @llvm.dbg.value(metadata double %0, metadata !1408, metadata !DIExpression()), !dbg !1395
  call void @llvm.dbg.value(metadata i32 0, metadata !1409, metadata !DIExpression()), !dbg !1395
  %1 = sext i32 %n to i64, !dbg !1410
  br label %for.cond, !dbg !1410

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ], !dbg !1412
  %x.0 = phi double [ %0, %entry ], [ %sub22, %for.inc ], !dbg !1395
  call void @llvm.dbg.value(metadata double %x.0, metadata !1408, metadata !DIExpression()), !dbg !1395
  call void @llvm.dbg.value(metadata i64 %indvars.iv, metadata !1409, metadata !DIExpression()), !dbg !1395
  %cmp = icmp slt i64 %indvars.iv, %1, !dbg !1413
  br i1 %cmp, label %for.body, label %for.end, !dbg !1415

for.body:                                         ; preds = %for.cond
  %mul3 = fmul contract double 0x3E80000000000000, %x.0, !dbg !1416
  call void @llvm.dbg.value(metadata double %mul3, metadata !1400, metadata !DIExpression()), !dbg !1395
  %conv4 = fptosi double %mul3 to i32, !dbg !1418
  %conv5 = sitofp i32 %conv4 to double, !dbg !1419
  call void @llvm.dbg.value(metadata double %conv5, metadata !1420, metadata !DIExpression()), !dbg !1395
  %mul6 = fmul contract double 0x4160000000000000, %conv5, !dbg !1421
  %sub7 = fsub contract double %x.0, %mul6, !dbg !1422
  call void @llvm.dbg.value(metadata double %sub7, metadata !1423, metadata !DIExpression()), !dbg !1395
  %mul8 = fmul contract double %conv1, %sub7, !dbg !1424
  %mul9 = fmul contract double %sub, %conv5, !dbg !1425
  %add = fadd contract double %mul8, %mul9, !dbg !1426
  call void @llvm.dbg.value(metadata double %add, metadata !1400, metadata !DIExpression()), !dbg !1395
  %mul10 = fmul contract double 0x3E80000000000000, %add, !dbg !1427
  %conv11 = fptosi double %mul10 to i32, !dbg !1428
  %conv12 = sitofp i32 %conv11 to double, !dbg !1429
  call void @llvm.dbg.value(metadata double %conv12, metadata !1430, metadata !DIExpression()), !dbg !1395
  %mul13 = fmul contract double 0x4160000000000000, %conv12, !dbg !1431
  %sub14 = fsub contract double %add, %mul13, !dbg !1432
  call void @llvm.dbg.value(metadata double %sub14, metadata !1433, metadata !DIExpression()), !dbg !1395
  %mul15 = fmul contract double 0x4160000000000000, %sub14, !dbg !1434
  %mul16 = fmul contract double %sub, %sub7, !dbg !1435
  %add17 = fadd contract double %mul15, %mul16, !dbg !1436
  call void @llvm.dbg.value(metadata double %add17, metadata !1437, metadata !DIExpression()), !dbg !1395
  %mul18 = fmul contract double 0x3D10000000000000, %add17, !dbg !1438
  %conv19 = fptosi double %mul18 to i32, !dbg !1439
  %conv20 = sitofp i32 %conv19 to double, !dbg !1440
  call void @llvm.dbg.value(metadata double %conv20, metadata !1441, metadata !DIExpression()), !dbg !1395
  %mul21 = fmul contract double 0x42D0000000000000, %conv20, !dbg !1442
  %sub22 = fsub contract double %add17, %mul21, !dbg !1443
  call void @llvm.dbg.value(metadata double %sub22, metadata !1408, metadata !DIExpression()), !dbg !1395
  %mul23 = fmul contract double 0x3D10000000000000, %sub22, !dbg !1444
  %arrayidx = getelementptr inbounds double, double* %y, i64 %indvars.iv, !dbg !1445
  store double %mul23, double* %arrayidx, align 8, !dbg !1446
  br label %for.inc, !dbg !1447

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !1448
  call void @llvm.dbg.value(metadata i32 undef, metadata !1409, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !1395
  br label %for.cond, !dbg !1449, !llvm.loop !1450

for.end:                                          ; preds = %for.cond
  %x.0.lcssa = phi double [ %x.0, %for.cond ], !dbg !1395
  call void @llvm.dbg.value(metadata double %x.0.lcssa, metadata !1408, metadata !DIExpression()), !dbg !1395
  store double %x.0.lcssa, double* %x_seed, align 8, !dbg !1452
  ret void, !dbg !1453
}

; Function Attrs: convergent noinline nounwind
define dso_local void @_Z13ipow46_devicediPd(double %a, i32 %exponent, double* %result) #2 !dbg !1454 {
entry:
  %q = alloca double, align 8
  %r = alloca double, align 8
  call void @llvm.dbg.value(metadata double %a, metadata !1457, metadata !DIExpression()), !dbg !1458
  call void @llvm.dbg.value(metadata i32 %exponent, metadata !1459, metadata !DIExpression()), !dbg !1458
  call void @llvm.dbg.value(metadata double* %result, metadata !1460, metadata !DIExpression()), !dbg !1458
  call void @llvm.dbg.declare(metadata double* %q, metadata !1461, metadata !DIExpression()), !dbg !1462
  call void @llvm.dbg.declare(metadata double* %r, metadata !1463, metadata !DIExpression()), !dbg !1464
  store double 1.000000e+00, double* %result, align 8, !dbg !1465
  %cmp = icmp eq i32 %exponent, 0, !dbg !1466
  br i1 %cmp, label %if.then, label %if.end, !dbg !1468

if.then:                                          ; preds = %entry
  br label %return, !dbg !1469

if.end:                                           ; preds = %entry
  store double %a, double* %q, align 8, !dbg !1471
  store double 1.000000e+00, double* %r, align 8, !dbg !1472
  call void @llvm.dbg.value(metadata i32 %exponent, metadata !1473, metadata !DIExpression()), !dbg !1458
  br label %while.cond, !dbg !1474

while.cond:                                       ; preds = %if.end5, %if.end
  %n.0 = phi i32 [ %exponent, %if.end ], [ %n.1, %if.end5 ], !dbg !1458
  call void @llvm.dbg.value(metadata i32 %n.0, metadata !1473, metadata !DIExpression()), !dbg !1458
  %cmp1 = icmp sgt i32 %n.0, 1, !dbg !1475
  br i1 %cmp1, label %while.body, label %while.end, !dbg !1474

while.body:                                       ; preds = %while.cond
  %div = sdiv i32 %n.0, 2, !dbg !1476
  call void @llvm.dbg.value(metadata i32 %div, metadata !1478, metadata !DIExpression()), !dbg !1458
  %mul = mul nsw i32 %div, 2, !dbg !1479
  %cmp2 = icmp eq i32 %mul, %n.0, !dbg !1481
  br i1 %cmp2, label %if.then3, label %if.else, !dbg !1482

if.then3:                                         ; preds = %while.body
  %0 = load double, double* %q, align 8, !dbg !1483
  %call = call double @_Z13randlc_devicePdd(double* %q, double %0) #4, !dbg !1485
  call void @llvm.dbg.value(metadata i32 %div, metadata !1473, metadata !DIExpression()), !dbg !1458
  br label %if.end5, !dbg !1486

if.else:                                          ; preds = %while.body
  %1 = load double, double* %q, align 8, !dbg !1487
  %call4 = call double @_Z13randlc_devicePdd(double* %r, double %1) #4, !dbg !1489
  %sub = sub nsw i32 %n.0, 1, !dbg !1490
  call void @llvm.dbg.value(metadata i32 %sub, metadata !1473, metadata !DIExpression()), !dbg !1458
  br label %if.end5

if.end5:                                          ; preds = %if.else, %if.then3
  %n.1 = phi i32 [ %div, %if.then3 ], [ %sub, %if.else ], !dbg !1491
  call void @llvm.dbg.value(metadata i32 %n.1, metadata !1473, metadata !DIExpression()), !dbg !1458
  br label %while.cond, !dbg !1474, !llvm.loop !1492

while.end:                                        ; preds = %while.cond
  %2 = load double, double* %q, align 8, !dbg !1494
  %call6 = call double @_Z13randlc_devicePdd(double* %r, double %2) #4, !dbg !1495
  %3 = load double, double* %r, align 8, !dbg !1496
  store double %3, double* %result, align 8, !dbg !1497
  br label %return, !dbg !1498

return:                                           ; preds = %while.end, %if.then
  ret void, !dbg !1498
}

; Function Attrs: convergent noinline nounwind
define dso_local double @_Z13randlc_devicePdd(double* %x, double %a) #2 !dbg !1499 {
entry:
  call void @llvm.dbg.value(metadata double* %x, metadata !1502, metadata !DIExpression()), !dbg !1503
  call void @llvm.dbg.value(metadata double %a, metadata !1504, metadata !DIExpression()), !dbg !1503
  %mul = fmul contract double 0x3E80000000000000, %a, !dbg !1505
  call void @llvm.dbg.value(metadata double %mul, metadata !1506, metadata !DIExpression()), !dbg !1503
  %conv = fptosi double %mul to i32, !dbg !1507
  %conv1 = sitofp i32 %conv to double, !dbg !1508
  call void @llvm.dbg.value(metadata double %conv1, metadata !1509, metadata !DIExpression()), !dbg !1503
  %mul2 = fmul contract double 0x4160000000000000, %conv1, !dbg !1510
  %sub = fsub contract double %a, %mul2, !dbg !1511
  call void @llvm.dbg.value(metadata double %sub, metadata !1512, metadata !DIExpression()), !dbg !1503
  %0 = load double, double* %x, align 8, !dbg !1513
  %mul3 = fmul contract double 0x3E80000000000000, %0, !dbg !1514
  call void @llvm.dbg.value(metadata double %mul3, metadata !1506, metadata !DIExpression()), !dbg !1503
  %conv4 = fptosi double %mul3 to i32, !dbg !1515
  %conv5 = sitofp i32 %conv4 to double, !dbg !1516
  call void @llvm.dbg.value(metadata double %conv5, metadata !1517, metadata !DIExpression()), !dbg !1503
  %1 = load double, double* %x, align 8, !dbg !1518
  %mul6 = fmul contract double 0x4160000000000000, %conv5, !dbg !1519
  %sub7 = fsub contract double %1, %mul6, !dbg !1520
  call void @llvm.dbg.value(metadata double %sub7, metadata !1521, metadata !DIExpression()), !dbg !1503
  %mul8 = fmul contract double %conv1, %sub7, !dbg !1522
  %mul9 = fmul contract double %sub, %conv5, !dbg !1523
  %add = fadd contract double %mul8, %mul9, !dbg !1524
  call void @llvm.dbg.value(metadata double %add, metadata !1506, metadata !DIExpression()), !dbg !1503
  %mul10 = fmul contract double 0x3E80000000000000, %add, !dbg !1525
  %conv11 = fptosi double %mul10 to i32, !dbg !1526
  %conv12 = sitofp i32 %conv11 to double, !dbg !1527
  call void @llvm.dbg.value(metadata double %conv12, metadata !1528, metadata !DIExpression()), !dbg !1503
  %mul13 = fmul contract double 0x4160000000000000, %conv12, !dbg !1529
  %sub14 = fsub contract double %add, %mul13, !dbg !1530
  call void @llvm.dbg.value(metadata double %sub14, metadata !1531, metadata !DIExpression()), !dbg !1503
  %mul15 = fmul contract double 0x4160000000000000, %sub14, !dbg !1532
  %mul16 = fmul contract double %sub, %sub7, !dbg !1533
  %add17 = fadd contract double %mul15, %mul16, !dbg !1534
  call void @llvm.dbg.value(metadata double %add17, metadata !1535, metadata !DIExpression()), !dbg !1503
  %mul18 = fmul contract double 0x3D10000000000000, %add17, !dbg !1536
  %conv19 = fptosi double %mul18 to i32, !dbg !1537
  %conv20 = sitofp i32 %conv19 to double, !dbg !1538
  call void @llvm.dbg.value(metadata double %conv20, metadata !1539, metadata !DIExpression()), !dbg !1503
  %mul21 = fmul contract double 0x42D0000000000000, %conv20, !dbg !1540
  %sub22 = fsub contract double %add17, %mul21, !dbg !1541
  store double %sub22, double* %x, align 8, !dbg !1542
  %2 = load double, double* %x, align 8, !dbg !1543
  %mul23 = fmul contract double 0x3D10000000000000, %2, !dbg !1544
  ret double %mul23, !dbg !1545
}

; Function Attrs: noinline nounwind uwtable
define dso_local double @_Z6randlcPdd(double* %x, double %a) #6 !dbg !1546 {
entry:
  call void @llvm.dbg.value(metadata double* %x, metadata !1547, metadata !DIExpression()), !dbg !1548
  call void @llvm.dbg.value(metadata double %a, metadata !1549, metadata !DIExpression()), !dbg !1548
  %mul = fmul contract double 0x3E80000000000000, %a, !dbg !1550
  call void @llvm.dbg.value(metadata double %mul, metadata !1551, metadata !DIExpression()), !dbg !1548
  %conv = fptosi double %mul to i32, !dbg !1552
  %conv1 = sitofp i32 %conv to double, !dbg !1553
  call void @llvm.dbg.value(metadata double %conv1, metadata !1554, metadata !DIExpression()), !dbg !1548
  %mul2 = fmul contract double 0x4160000000000000, %conv1, !dbg !1555
  %sub = fsub contract double %a, %mul2, !dbg !1556
  call void @llvm.dbg.value(metadata double %sub, metadata !1557, metadata !DIExpression()), !dbg !1548
  %0 = load double, double* %x, align 8, !dbg !1558
  %mul3 = fmul contract double 0x3E80000000000000, %0, !dbg !1559
  call void @llvm.dbg.value(metadata double %mul3, metadata !1551, metadata !DIExpression()), !dbg !1548
  %conv4 = fptosi double %mul3 to i32, !dbg !1560
  %conv5 = sitofp i32 %conv4 to double, !dbg !1561
  call void @llvm.dbg.value(metadata double %conv5, metadata !1562, metadata !DIExpression()), !dbg !1548
  %1 = load double, double* %x, align 8, !dbg !1563
  %mul6 = fmul contract double 0x4160000000000000, %conv5, !dbg !1564
  %sub7 = fsub contract double %1, %mul6, !dbg !1565
  call void @llvm.dbg.value(metadata double %sub7, metadata !1566, metadata !DIExpression()), !dbg !1548
  %mul8 = fmul contract double %conv1, %sub7, !dbg !1567
  %mul9 = fmul contract double %sub, %conv5, !dbg !1568
  %add = fadd contract double %mul8, %mul9, !dbg !1569
  call void @llvm.dbg.value(metadata double %add, metadata !1551, metadata !DIExpression()), !dbg !1548
  %mul10 = fmul contract double 0x3E80000000000000, %add, !dbg !1570
  %conv11 = fptosi double %mul10 to i32, !dbg !1571
  %conv12 = sitofp i32 %conv11 to double, !dbg !1572
  call void @llvm.dbg.value(metadata double %conv12, metadata !1573, metadata !DIExpression()), !dbg !1548
  %mul13 = fmul contract double 0x4160000000000000, %conv12, !dbg !1574
  %sub14 = fsub contract double %add, %mul13, !dbg !1575
  call void @llvm.dbg.value(metadata double %sub14, metadata !1576, metadata !DIExpression()), !dbg !1548
  %mul15 = fmul contract double 0x4160000000000000, %sub14, !dbg !1577
  %mul16 = fmul contract double %sub, %sub7, !dbg !1578
  %add17 = fadd contract double %mul15, %mul16, !dbg !1579
  call void @llvm.dbg.value(metadata double %add17, metadata !1580, metadata !DIExpression()), !dbg !1548
  %mul18 = fmul contract double 0x3D10000000000000, %add17, !dbg !1581
  %conv19 = fptosi double %mul18 to i32, !dbg !1582
  %conv20 = sitofp i32 %conv19 to double, !dbg !1583
  call void @llvm.dbg.value(metadata double %conv20, metadata !1584, metadata !DIExpression()), !dbg !1548
  %mul21 = fmul contract double 0x42D0000000000000, %conv20, !dbg !1585
  %sub22 = fsub contract double %add17, %mul21, !dbg !1586
  store double %sub22, double* %x, align 8, !dbg !1587
  %2 = load double, double* %x, align 8, !dbg !1588
  %mul23 = fmul contract double 0x3D10000000000000, %2, !dbg !1589
  ret double %mul23, !dbg !1590
}

; Function Attrs: noinline uwtable
define dso_local void @_Z15c_print_resultsPcciiiiddS_iS_S_S_S_S_S_S_S_S_S_S_S_S_S_(i8* %name, i8 signext %class_npb, i32 %n1, i32 %n2, i32 %n3, i32 %niter, double %t, double %mops, i8* %optype, i32 %passed_verification, i8* %npbversion, i8* %compiletime, i8* %compilerversion, i8* %libversion, i8* %cpu_device, i8* %gpu_device, i8* %gpu_config, i8* %cc, i8* %clink, i8* %c_lib, i8* %c_inc, i8* %cflags, i8* %clinkflags, i8* %rand) #7 !dbg !1591 {
entry:
  %size = alloca [16 x i8], align 16
  call void @llvm.dbg.value(metadata i8* %name, metadata !1594, metadata !DIExpression()), !dbg !1595
  call void @llvm.dbg.value(metadata i8 %class_npb, metadata !1596, metadata !DIExpression()), !dbg !1595
  call void @llvm.dbg.value(metadata i32 %n1, metadata !1597, metadata !DIExpression()), !dbg !1595
  call void @llvm.dbg.value(metadata i32 %n2, metadata !1598, metadata !DIExpression()), !dbg !1595
  call void @llvm.dbg.value(metadata i32 %n3, metadata !1599, metadata !DIExpression()), !dbg !1595
  call void @llvm.dbg.value(metadata i32 %niter, metadata !1600, metadata !DIExpression()), !dbg !1595
  call void @llvm.dbg.value(metadata double %t, metadata !1601, metadata !DIExpression()), !dbg !1595
  call void @llvm.dbg.value(metadata double %mops, metadata !1602, metadata !DIExpression()), !dbg !1595
  call void @llvm.dbg.value(metadata i8* %optype, metadata !1603, metadata !DIExpression()), !dbg !1595
  call void @llvm.dbg.value(metadata i32 %passed_verification, metadata !1604, metadata !DIExpression()), !dbg !1595
  call void @llvm.dbg.value(metadata i8* %npbversion, metadata !1605, metadata !DIExpression()), !dbg !1595
  call void @llvm.dbg.value(metadata i8* %compiletime, metadata !1606, metadata !DIExpression()), !dbg !1595
  call void @llvm.dbg.value(metadata i8* %compilerversion, metadata !1607, metadata !DIExpression()), !dbg !1595
  call void @llvm.dbg.value(metadata i8* %libversion, metadata !1608, metadata !DIExpression()), !dbg !1595
  call void @llvm.dbg.value(metadata i8* %cpu_device, metadata !1609, metadata !DIExpression()), !dbg !1595
  call void @llvm.dbg.value(metadata i8* %gpu_device, metadata !1610, metadata !DIExpression()), !dbg !1595
  call void @llvm.dbg.value(metadata i8* %gpu_config, metadata !1611, metadata !DIExpression()), !dbg !1595
  call void @llvm.dbg.value(metadata i8* %cc, metadata !1612, metadata !DIExpression()), !dbg !1595
  call void @llvm.dbg.value(metadata i8* %clink, metadata !1613, metadata !DIExpression()), !dbg !1595
  call void @llvm.dbg.value(metadata i8* %c_lib, metadata !1614, metadata !DIExpression()), !dbg !1595
  call void @llvm.dbg.value(metadata i8* %c_inc, metadata !1615, metadata !DIExpression()), !dbg !1595
  call void @llvm.dbg.value(metadata i8* %cflags, metadata !1616, metadata !DIExpression()), !dbg !1595
  call void @llvm.dbg.value(metadata i8* %clinkflags, metadata !1617, metadata !DIExpression()), !dbg !1595
  call void @llvm.dbg.value(metadata i8* %rand, metadata !1618, metadata !DIExpression()), !dbg !1595
  %call = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([27 x i8], [27 x i8]* @.str, i64 0, i64 0), i8* %name), !dbg !1619
  %conv = sext i8 %class_npb to i32, !dbg !1620
  %call1 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([46 x i8], [46 x i8]* @.str.1, i64 0, i64 0), i32 %conv), !dbg !1621
  %arrayidx = getelementptr inbounds i8, i8* %name, i64 0, !dbg !1622
  %0 = load i8, i8* %arrayidx, align 1, !dbg !1622
  %conv2 = sext i8 %0 to i32, !dbg !1622
  %cmp = icmp eq i32 %conv2, 73, !dbg !1624
  br i1 %cmp, label %land.lhs.true, label %if.else15, !dbg !1625

land.lhs.true:                                    ; preds = %entry
  %arrayidx3 = getelementptr inbounds i8, i8* %name, i64 1, !dbg !1626
  %1 = load i8, i8* %arrayidx3, align 1, !dbg !1626
  %conv4 = sext i8 %1 to i32, !dbg !1626
  %cmp5 = icmp eq i32 %conv4, 83, !dbg !1627
  br i1 %cmp5, label %if.then, label %if.else15, !dbg !1628

if.then:                                          ; preds = %land.lhs.true
  %cmp6 = icmp eq i32 %n3, 0, !dbg !1629
  br i1 %cmp6, label %if.then7, label %if.else, !dbg !1632

if.then7:                                         ; preds = %if.then
  %conv8 = sext i32 %n1 to i64, !dbg !1633
  call void @llvm.dbg.value(metadata i64 %conv8, metadata !1635, metadata !DIExpression()), !dbg !1636
  %cmp9 = icmp ne i32 %n2, 0, !dbg !1637
  br i1 %cmp9, label %if.then10, label %if.end, !dbg !1639

if.then10:                                        ; preds = %if.then7
  %conv11 = sext i32 %n2 to i64, !dbg !1640
  %mul = mul nsw i64 %conv8, %conv11, !dbg !1642
  call void @llvm.dbg.value(metadata i64 %mul, metadata !1635, metadata !DIExpression()), !dbg !1636
  br label %if.end, !dbg !1643

if.end:                                           ; preds = %if.then10, %if.then7
  %nn.0 = phi i64 [ %mul, %if.then10 ], [ %conv8, %if.then7 ], !dbg !1636
  call void @llvm.dbg.value(metadata i64 %nn.0, metadata !1635, metadata !DIExpression()), !dbg !1636
  %call12 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([38 x i8], [38 x i8]* @.str.2, i64 0, i64 0), i64 %nn.0), !dbg !1644
  br label %if.end14, !dbg !1645

if.else:                                          ; preds = %if.then
  %call13 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([44 x i8], [44 x i8]* @.str.3, i64 0, i64 0), i32 %n1, i32 %n2, i32 %n3), !dbg !1646
  br label %if.end14

if.end14:                                         ; preds = %if.else, %if.end
  br label %if.end48, !dbg !1648

if.else15:                                        ; preds = %land.lhs.true, %entry
  call void @llvm.dbg.declare(metadata [16 x i8]* %size, metadata !1649, metadata !DIExpression()), !dbg !1654
  %cmp16 = icmp eq i32 %n2, 0, !dbg !1655
  br i1 %cmp16, label %land.lhs.true17, label %if.else45, !dbg !1657

land.lhs.true17:                                  ; preds = %if.else15
  %cmp18 = icmp eq i32 %n3, 0, !dbg !1658
  br i1 %cmp18, label %if.then19, label %if.else45, !dbg !1659

if.then19:                                        ; preds = %land.lhs.true17
  %arrayidx20 = getelementptr inbounds i8, i8* %name, i64 0, !dbg !1660
  %2 = load i8, i8* %arrayidx20, align 1, !dbg !1660
  %conv21 = sext i8 %2 to i32, !dbg !1660
  %cmp22 = icmp eq i32 %conv21, 69, !dbg !1663
  br i1 %cmp22, label %land.lhs.true23, label %if.else42, !dbg !1664

land.lhs.true23:                                  ; preds = %if.then19
  %arrayidx24 = getelementptr inbounds i8, i8* %name, i64 1, !dbg !1665
  %3 = load i8, i8* %arrayidx24, align 1, !dbg !1665
  %conv25 = sext i8 %3 to i32, !dbg !1665
  %cmp26 = icmp eq i32 %conv25, 80, !dbg !1666
  br i1 %cmp26, label %if.then27, label %if.else42, !dbg !1667

if.then27:                                        ; preds = %land.lhs.true23
  %arraydecay = getelementptr inbounds [16 x i8], [16 x i8]* %size, i64 0, i64 0, !dbg !1668
  %conv28 = sitofp i32 %n1 to double, !dbg !1670
  %call29 = call double @pow(double 2.000000e+00, double %conv28) #11, !dbg !1671
  %call30 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay, i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.4, i64 0, i64 0), double %call29) #11, !dbg !1672
  call void @llvm.dbg.value(metadata i32 14, metadata !1673, metadata !DIExpression()), !dbg !1674
  %idxprom = sext i32 14 to i64, !dbg !1675
  %arrayidx31 = getelementptr inbounds [16 x i8], [16 x i8]* %size, i64 0, i64 %idxprom, !dbg !1675
  %4 = load i8, i8* %arrayidx31, align 1, !dbg !1675
  %conv32 = sext i8 %4 to i32, !dbg !1675
  %cmp33 = icmp eq i32 %conv32, 46, !dbg !1677
  br i1 %cmp33, label %if.then34, label %if.end37, !dbg !1678

if.then34:                                        ; preds = %if.then27
  %idxprom35 = sext i32 14 to i64, !dbg !1679
  %arrayidx36 = getelementptr inbounds [16 x i8], [16 x i8]* %size, i64 0, i64 %idxprom35, !dbg !1679
  store i8 32, i8* %arrayidx36, align 1, !dbg !1681
  %dec = add nsw i32 14, -1, !dbg !1682
  call void @llvm.dbg.value(metadata i32 %dec, metadata !1673, metadata !DIExpression()), !dbg !1674
  br label %if.end37, !dbg !1683

if.end37:                                         ; preds = %if.then34, %if.then27
  %j.0 = phi i32 [ %dec, %if.then34 ], [ 14, %if.then27 ], !dbg !1684
  call void @llvm.dbg.value(metadata i32 %j.0, metadata !1673, metadata !DIExpression()), !dbg !1674
  %add = add nsw i32 %j.0, 1, !dbg !1685
  %idxprom38 = sext i32 %add to i64, !dbg !1686
  %arrayidx39 = getelementptr inbounds [16 x i8], [16 x i8]* %size, i64 0, i64 %idxprom38, !dbg !1686
  store i8 0, i8* %arrayidx39, align 1, !dbg !1687
  %arraydecay40 = getelementptr inbounds [16 x i8], [16 x i8]* %size, i64 0, i64 0, !dbg !1688
  %call41 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([34 x i8], [34 x i8]* @.str.5, i64 0, i64 0), i8* %arraydecay40), !dbg !1689
  br label %if.end44, !dbg !1690

if.else42:                                        ; preds = %land.lhs.true23, %if.then19
  %call43 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.6, i64 0, i64 0), i32 %n1), !dbg !1691
  br label %if.end44

if.end44:                                         ; preds = %if.else42, %if.end37
  br label %if.end47, !dbg !1693

if.else45:                                        ; preds = %land.lhs.true17, %if.else15
  %call46 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([42 x i8], [42 x i8]* @.str.7, i64 0, i64 0), i32 %n1, i32 %n2, i32 %n3), !dbg !1694
  br label %if.end47

if.end47:                                         ; preds = %if.else45, %if.end44
  br label %if.end48

if.end48:                                         ; preds = %if.end47, %if.end14
  %call49 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.8, i64 0, i64 0), i32 %niter), !dbg !1696
  %call50 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([39 x i8], [39 x i8]* @.str.9, i64 0, i64 0), double %t), !dbg !1697
  %call51 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([39 x i8], [39 x i8]* @.str.10, i64 0, i64 0), double %mops), !dbg !1698
  %call52 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([25 x i8], [25 x i8]* @.str.11, i64 0, i64 0), i8* %optype), !dbg !1699
  %cmp53 = icmp slt i32 %passed_verification, 0, !dbg !1700
  br i1 %cmp53, label %if.then54, label %if.else56, !dbg !1702

if.then54:                                        ; preds = %if.end48
  %call55 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([45 x i8], [45 x i8]* @.str.12, i64 0, i64 0)), !dbg !1703
  br label %if.end62, !dbg !1705

if.else56:                                        ; preds = %if.end48
  %tobool = icmp ne i32 %passed_verification, 0, !dbg !1706
  br i1 %tobool, label %if.then57, label %if.else59, !dbg !1708

if.then57:                                        ; preds = %if.else56
  %call58 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([45 x i8], [45 x i8]* @.str.13, i64 0, i64 0)), !dbg !1709
  br label %if.end61, !dbg !1711

if.else59:                                        ; preds = %if.else56
  %call60 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([45 x i8], [45 x i8]* @.str.14, i64 0, i64 0)), !dbg !1712
  br label %if.end61

if.end61:                                         ; preds = %if.else59, %if.then57
  br label %if.end62

if.end62:                                         ; preds = %if.end61, %if.then54
  %call63 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.15, i64 0, i64 0), i8* %npbversion), !dbg !1714
  %call64 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.16, i64 0, i64 0), i8* %compiletime), !dbg !1715
  %call65 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.17, i64 0, i64 0), i8* %compilerversion), !dbg !1716
  %call66 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.18, i64 0, i64 0), i8* %libversion), !dbg !1717
  %call67 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([20 x i8], [20 x i8]* @.str.19, i64 0, i64 0)), !dbg !1718
  %call68 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.20, i64 0, i64 0), i8* %cc), !dbg !1719
  %call69 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.21, i64 0, i64 0), i8* %clink), !dbg !1720
  %call70 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.22, i64 0, i64 0), i8* %c_lib), !dbg !1721
  %call71 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.23, i64 0, i64 0), i8* %c_inc), !dbg !1722
  %call72 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.24, i64 0, i64 0), i8* %cflags), !dbg !1723
  %call73 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.25, i64 0, i64 0), i8* %clinkflags), !dbg !1724
  %call74 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.26, i64 0, i64 0), i8* %rand), !dbg !1725
  %call75 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.27, i64 0, i64 0)), !dbg !1726
  %call76 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.28, i64 0, i64 0), i8* %cpu_device), !dbg !1727
  %call77 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.29, i64 0, i64 0), i8* %gpu_device), !dbg !1728
  %call78 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.30, i64 0, i64 0)), !dbg !1729
  %call79 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.31, i64 0, i64 0), i8* %gpu_config), !dbg !1730
  %call80 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.32, i64 0, i64 0)), !dbg !1731
  %call81 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([72 x i8], [72 x i8]* @.str.33, i64 0, i64 0)), !dbg !1732
  %call82 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([27 x i8], [27 x i8]* @.str.34, i64 0, i64 0)), !dbg !1733
  %call83 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([56 x i8], [56 x i8]* @.str.35, i64 0, i64 0)), !dbg !1734
  %call84 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([52 x i8], [52 x i8]* @.str.36, i64 0, i64 0)), !dbg !1735
  %call85 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([46 x i8], [46 x i8]* @.str.37, i64 0, i64 0)), !dbg !1736
  %call86 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.32, i64 0, i64 0)), !dbg !1737
  %call87 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([43 x i8], [43 x i8]* @.str.38, i64 0, i64 0)), !dbg !1738
  %call88 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([72 x i8], [72 x i8]* @.str.33, i64 0, i64 0)), !dbg !1739
  %call89 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.32, i64 0, i64 0)), !dbg !1740
  ret void, !dbg !1741
}

declare dso_local i32 @printf(i8*, ...) #8

; Function Attrs: nounwind
declare dso_local double @pow(double, double) #9

; Function Attrs: nounwind
declare dso_local i32 @sprintf(i8*, i8*, ...) #9

; Function Attrs: noinline norecurse uwtable
define dso_local i32 @main(i32 %argc, i8** %argv) #10 !dbg !1742 {
entry:
  %verified = alloca i32, align 4
  %class_npb = alloca i8, align 1
  %gpu_config = alloca [256 x i8], align 16
  %gpu_config_string = alloca [2048 x i8], align 16
  call void @llvm.dbg.value(metadata i32 %argc, metadata !1745, metadata !DIExpression()), !dbg !1746
  call void @llvm.dbg.value(metadata i8** %argv, metadata !1747, metadata !DIExpression()), !dbg !1746
  call void @llvm.dbg.value(metadata i32 0, metadata !1748, metadata !DIExpression()), !dbg !1746
  call void @llvm.dbg.declare(metadata i32* %verified, metadata !1749, metadata !DIExpression()), !dbg !1751
  call void @llvm.dbg.declare(metadata i8* %class_npb, metadata !1752, metadata !DIExpression()), !dbg !1753
  %call = call noalias i8* @malloc(i64 112) #11, !dbg !1754, !tulip.target.mapdata.from !1755
  %0 = bitcast i8* %call to %struct.dcomplex*, !dbg !1756
  store %struct.dcomplex* %0, %struct.dcomplex** @_ZL4sums, align 8, !dbg !1757
  %call1 = call noalias i8* @malloc(i64 67108864) #11, !dbg !1758
  %1 = bitcast i8* %call1 to double*, !dbg !1759
  store double* %1, double** @_ZL7twiddle, align 8, !dbg !1760
  %call2 = call noalias i8* @malloc(i64 4096) #11, !dbg !1761, !tulip.target.mapdata.to !1762
  %2 = bitcast i8* %call2 to %struct.dcomplex*, !dbg !1763
  store %struct.dcomplex* %2, %struct.dcomplex** @_ZL1u, align 8, !dbg !1764
  %call3 = call noalias i8* @malloc(i64 134217728) #11, !dbg !1765
  %3 = bitcast i8* %call3 to %struct.dcomplex*, !dbg !1766
  store %struct.dcomplex* %3, %struct.dcomplex** @_ZL2u0, align 8, !dbg !1767
  %call4 = call noalias i8* @malloc(i64 134217728) #11, !dbg !1768
  %4 = bitcast i8* %call4 to %struct.dcomplex*, !dbg !1769
  store %struct.dcomplex* %4, %struct.dcomplex** @_ZL2u1, align 8, !dbg !1770
  %call5 = call noalias i8* @malloc(i64 12) #11, !dbg !1771
  %5 = bitcast i8* %call5 to i32*, !dbg !1772
  store i32* %5, i32** @_ZL4dims, align 8, !dbg !1773
  call void @_ZL5setupv(), !dbg !1774
  call void @_ZL9setup_gpuv(), !dbg !1775
  %6 = load %struct.dcomplex*, %struct.dcomplex** @u0_device, align 8, !dbg !1776
  %7 = load %struct.dcomplex*, %struct.dcomplex** @u1_device, align 8, !dbg !1777
  %8 = load double*, double** @twiddle_device, align 8, !dbg !1778
  call void @_ZL11init_ui_gpuP8dcomplexS0_Pd(%struct.dcomplex* %6, %struct.dcomplex* %7, double* %8), !dbg !1779
  %call6 = call i32 @omp_get_thread_num(), !dbg !1780
  %cmp = icmp eq i32 %call6, 0, !dbg !1783
  br i1 %cmp, label %if.then, label %if.else, !dbg !1784

if.then:                                          ; preds = %entry
  %9 = load double*, double** @twiddle_device, align 8, !dbg !1785
  call void @_ZL20compute_indexmap_gpuPd(double* %9), !dbg !1787
  br label %if.end15, !dbg !1788

if.else:                                          ; preds = %entry
  %call7 = call i32 @omp_get_thread_num(), !dbg !1789
  %cmp8 = icmp eq i32 %call7, 1, !dbg !1791
  br i1 %cmp8, label %if.then9, label %if.else10, !dbg !1792

if.then9:                                         ; preds = %if.else
  %10 = load %struct.dcomplex*, %struct.dcomplex** @u1_device, align 8, !dbg !1793
  call void @_ZL30compute_initial_conditions_gpuP8dcomplex(%struct.dcomplex* %10), !dbg !1795
  br label %if.end14, !dbg !1796

if.else10:                                        ; preds = %if.else
  %call11 = call i32 @omp_get_thread_num(), !dbg !1797
  %cmp12 = icmp eq i32 %call11, 2, !dbg !1799
  br i1 %cmp12, label %if.then13, label %if.end, !dbg !1800

if.then13:                                        ; preds = %if.else10
  call void @_ZL12fft_init_gpui(i32 256), !dbg !1801
  br label %if.end, !dbg !1803

if.end:                                           ; preds = %if.then13, %if.else10
  br label %if.end14

if.end14:                                         ; preds = %if.end, %if.then9
  br label %if.end15

if.end15:                                         ; preds = %if.end14, %if.then
  %11 = load %struct.dcomplex*, %struct.dcomplex** @u1_device, align 8, !dbg !1804
  %12 = load %struct.dcomplex*, %struct.dcomplex** @u0_device, align 8, !dbg !1805
  call void @_ZL7fft_gpuiP8dcomplexS0_(i32 1, %struct.dcomplex* %11, %struct.dcomplex* %12), !dbg !1806
  %call17 = call i32 @omp_get_thread_num(), !dbg !1807
  %cmp18 = icmp eq i32 %call17, 0, !dbg !1810
  br i1 %cmp18, label %if.then19, label %if.else20, !dbg !1811

if.then19:                                        ; preds = %if.end15
  %13 = load double*, double** @twiddle_device, align 8, !dbg !1812
  call void @_ZL20compute_indexmap_gpuPd(double* %13), !dbg !1814
  br label %if.end30, !dbg !1815

if.else20:                                        ; preds = %if.end15
  %call21 = call i32 @omp_get_thread_num(), !dbg !1816
  %cmp22 = icmp eq i32 %call21, 1, !dbg !1818
  br i1 %cmp22, label %if.then23, label %if.else24, !dbg !1819

if.then23:                                        ; preds = %if.else20
  %14 = load %struct.dcomplex*, %struct.dcomplex** @u1_device, align 8, !dbg !1820
  call void @_ZL30compute_initial_conditions_gpuP8dcomplex(%struct.dcomplex* %14), !dbg !1822
  br label %if.end29, !dbg !1823

if.else24:                                        ; preds = %if.else20
  %call25 = call i32 @omp_get_thread_num(), !dbg !1824
  %cmp26 = icmp eq i32 %call25, 2, !dbg !1826
  br i1 %cmp26, label %if.then27, label %if.end28, !dbg !1827

if.then27:                                        ; preds = %if.else24
  call void @_ZL12fft_init_gpui(i32 256), !dbg !1828
  br label %if.end28, !dbg !1830

if.end28:                                         ; preds = %if.then27, %if.else24
  br label %if.end29

if.end29:                                         ; preds = %if.end28, %if.then23
  br label %if.end30

if.end30:                                         ; preds = %if.end29, %if.then19
  %15 = load %struct.dcomplex*, %struct.dcomplex** @u1_device, align 8, !dbg !1831
  %16 = load %struct.dcomplex*, %struct.dcomplex** @u0_device, align 8, !dbg !1832
  call void @_ZL7fft_gpuiP8dcomplexS0_(i32 1, %struct.dcomplex* %15, %struct.dcomplex* %16), !dbg !1833
  call void @llvm.dbg.value(metadata i32 1, metadata !1748, metadata !DIExpression()), !dbg !1746
  br label %for.cond, !dbg !1834

for.cond:                                         ; preds = %for.inc, %if.end30
  %iter.0 = phi i32 [ 1, %if.end30 ], [ %inc, %for.inc ], !dbg !1836
  call void @llvm.dbg.value(metadata i32 %iter.0, metadata !1748, metadata !DIExpression()), !dbg !1746
  %17 = load i32, i32* @_ZL5niter, align 4, !dbg !1837
  %cmp32 = icmp sle i32 %iter.0, %17, !dbg !1839
  br i1 %cmp32, label %for.body, label %for.end, !dbg !1840

for.body:                                         ; preds = %for.cond
  %18 = load %struct.dcomplex*, %struct.dcomplex** @u0_device, align 8, !dbg !1841
  %19 = load %struct.dcomplex*, %struct.dcomplex** @u1_device, align 8, !dbg !1843
  %20 = load double*, double** @twiddle_device, align 8, !dbg !1844
  call void @_ZL10evolve_gpuP8dcomplexS0_Pd(%struct.dcomplex* %18, %struct.dcomplex* %19, double* %20), !dbg !1845
  %21 = load %struct.dcomplex*, %struct.dcomplex** @u1_device, align 8, !dbg !1846
  %22 = load %struct.dcomplex*, %struct.dcomplex** @u1_device, align 8, !dbg !1847
  call void @_ZL7fft_gpuiP8dcomplexS0_(i32 -1, %struct.dcomplex* %21, %struct.dcomplex* %22), !dbg !1848
  %23 = load %struct.dcomplex*, %struct.dcomplex** @u1_device, align 8, !dbg !1849
  call void @_ZL12checksum_gpuiP8dcomplex(i32 %iter.0, %struct.dcomplex* %23), !dbg !1850
  br label %for.inc, !dbg !1851

for.inc:                                          ; preds = %for.body
  %inc = add nuw nsw i32 %iter.0, 1, !dbg !1852
  call void @llvm.dbg.value(metadata i32 %inc, metadata !1748, metadata !DIExpression()), !dbg !1746
  br label %for.cond, !dbg !1853, !llvm.loop !1854

for.end:                                          ; preds = %for.cond
  %24 = load %struct.dcomplex*, %struct.dcomplex** @_ZL4sums, align 8, !dbg !1856
  %25 = bitcast %struct.dcomplex* %24 to i8*, !dbg !1856
  %26 = load %struct.dcomplex*, %struct.dcomplex** @_ZL4sums, align 8, !dbg !1857
  %27 = bitcast %struct.dcomplex* %26 to i8*, !dbg !1857
  %28 = load i64, i64* @size_sums_device, align 8, !dbg !1858
  %call33 = call i32 @cudaMemcpy(i8* %25, i8* %27, i64 %28, i32 2), !dbg !1859, !tulip.target.end.of.map !1860
  call void @llvm.dbg.value(metadata i32 1, metadata !1748, metadata !DIExpression()), !dbg !1746
  br label %for.cond34, !dbg !1861

for.cond34:                                       ; preds = %for.inc40, %for.end
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc40 ], [ 1, %for.end ], !dbg !1863
  call void @llvm.dbg.value(metadata i64 %indvars.iv, metadata !1748, metadata !DIExpression()), !dbg !1746
  %29 = load i32, i32* @_ZL5niter, align 4, !dbg !1864
  %30 = sext i32 %29 to i64, !dbg !1866
  %cmp35 = icmp sle i64 %indvars.iv, %30, !dbg !1866
  br i1 %cmp35, label %for.body36, label %for.end42, !dbg !1867

for.body36:                                       ; preds = %for.cond34
  %31 = load %struct.dcomplex*, %struct.dcomplex** @_ZL4sums, align 8, !dbg !1868
  %arrayidx = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %31, i64 %indvars.iv, !dbg !1868
  %real = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx, i32 0, i32 0, !dbg !1870
  %32 = load double, double* %real, align 8, !dbg !1870
  %33 = load %struct.dcomplex*, %struct.dcomplex** @_ZL4sums, align 8, !dbg !1871
  %arrayidx38 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %33, i64 %indvars.iv, !dbg !1871
  %imag = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx38, i32 0, i32 1, !dbg !1872
  %34 = load double, double* %imag, align 8, !dbg !1872
  %35 = trunc i64 %indvars.iv to i32, !dbg !1873
  %call39 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([40 x i8], [40 x i8]* @.str.39, i64 0, i64 0), i32 %35, double %32, double %34), !dbg !1873
  br label %for.inc40, !dbg !1874

for.inc40:                                        ; preds = %for.body36
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !1875
  call void @llvm.dbg.value(metadata i32 undef, metadata !1748, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !1746
  br label %for.cond34, !dbg !1876, !llvm.loop !1877

for.end42:                                        ; preds = %for.cond34
  %36 = load i32, i32* @_ZL5niter, align 4, !dbg !1879
  call void @_ZL6verifyiiiiPiPc(i32 256, i32 256, i32 128, i32 %36, i32* %verified, i8* %class_npb), !dbg !1880
  call void @llvm.dbg.value(metadata double 0.000000e+00, metadata !1881, metadata !DIExpression()), !dbg !1746
  %cmp43 = fcmp une double 0.000000e+00, 0.000000e+00, !dbg !1882
  br i1 %cmp43, label %if.then44, label %if.else52, !dbg !1884

if.then44:                                        ; preds = %for.end42
  %call45 = call double @log(double 0x4160000000000000) #11, !dbg !1885
  %mul = fmul contract double 7.196410e+00, %call45, !dbg !1887
  %add = fadd contract double 1.481570e+01, %mul, !dbg !1888
  %call46 = call double @log(double 0x4160000000000000) #11, !dbg !1889
  %mul47 = fmul contract double 7.211130e+00, %call46, !dbg !1890
  %add48 = fadd contract double 5.235180e+00, %mul47, !dbg !1891
  %37 = load i32, i32* @_ZL5niter, align 4, !dbg !1892
  %conv = sitofp i32 %37 to double, !dbg !1892
  %mul49 = fmul contract double %add48, %conv, !dbg !1893
  %add50 = fadd contract double %add, %mul49, !dbg !1894
  %mul51 = fmul contract double 0x4020C6F7A0B5ED8D, %add50, !dbg !1895
  %div = fdiv double %mul51, 0.000000e+00, !dbg !1896
  call void @llvm.dbg.value(metadata double %div, metadata !1897, metadata !DIExpression()), !dbg !1746
  br label %if.end53, !dbg !1898

if.else52:                                        ; preds = %for.end42
  call void @llvm.dbg.value(metadata double 0.000000e+00, metadata !1897, metadata !DIExpression()), !dbg !1746
  br label %if.end53

if.end53:                                         ; preds = %if.else52, %if.then44
  %mflops.0 = phi double [ %div, %if.then44 ], [ 0.000000e+00, %if.else52 ], !dbg !1899
  call void @llvm.dbg.value(metadata double %mflops.0, metadata !1897, metadata !DIExpression()), !dbg !1746
  call void @llvm.dbg.declare(metadata [256 x i8]* %gpu_config, metadata !1900, metadata !DIExpression()), !dbg !1901
  call void @llvm.dbg.declare(metadata [2048 x i8]* %gpu_config_string, metadata !1902, metadata !DIExpression()), !dbg !1906
  %arraydecay = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1907
  %call54 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay, i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.40, i64 0, i64 0), i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.41, i64 0, i64 0), i8* getelementptr inbounds ([18 x i8], [18 x i8]* @.str.42, i64 0, i64 0)) #11, !dbg !1908
  %arraydecay55 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !1909
  %arraydecay56 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1910
  %call57 = call i8* @strcpy(i8* %arraydecay55, i8* %arraydecay56) #11, !dbg !1911
  %arraydecay58 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1912
  %38 = load i32, i32* @threads_per_block_on_compute_indexmap, align 4, !dbg !1913
  %call59 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay58, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.43, i64 0, i64 0), i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.44, i64 0, i64 0), i32 %38) #11, !dbg !1914
  %arraydecay60 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !1915
  %arraydecay61 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1916
  %call62 = call i8* @strcat(i8* %arraydecay60, i8* %arraydecay61) #11, !dbg !1917
  %arraydecay63 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1918
  %39 = load i32, i32* @threads_per_block_on_compute_initial_conditions, align 4, !dbg !1919
  %call64 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay63, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.43, i64 0, i64 0), i8* getelementptr inbounds ([20 x i8], [20 x i8]* @.str.45, i64 0, i64 0), i32 %39) #11, !dbg !1920
  %arraydecay65 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !1921
  %arraydecay66 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1922
  %call67 = call i8* @strcat(i8* %arraydecay65, i8* %arraydecay66) #11, !dbg !1923
  %arraydecay68 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1924
  %40 = load i32, i32* @threads_per_block_on_init_ui, align 4, !dbg !1925
  %call69 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay68, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.43, i64 0, i64 0), i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str.46, i64 0, i64 0), i32 %40) #11, !dbg !1926
  %arraydecay70 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !1927
  %arraydecay71 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1928
  %call72 = call i8* @strcat(i8* %arraydecay70, i8* %arraydecay71) #11, !dbg !1929
  %arraydecay73 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1930
  %41 = load i32, i32* @threads_per_block_on_evolve, align 4, !dbg !1931
  %call74 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay73, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.43, i64 0, i64 0), i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.47, i64 0, i64 0), i32 %41) #11, !dbg !1932
  %arraydecay75 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !1933
  %arraydecay76 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1934
  %call77 = call i8* @strcat(i8* %arraydecay75, i8* %arraydecay76) #11, !dbg !1935
  %arraydecay78 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1936
  %42 = load i32, i32* @threads_per_block_on_fftx_1, align 4, !dbg !1937
  %call79 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay78, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.43, i64 0, i64 0), i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.48, i64 0, i64 0), i32 %42) #11, !dbg !1938
  %arraydecay80 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !1939
  %arraydecay81 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1940
  %call82 = call i8* @strcat(i8* %arraydecay80, i8* %arraydecay81) #11, !dbg !1941
  %arraydecay83 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1942
  %43 = load i32, i32* @threads_per_block_on_fftx_2, align 4, !dbg !1943
  %call84 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay83, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.43, i64 0, i64 0), i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.49, i64 0, i64 0), i32 %43) #11, !dbg !1944
  %arraydecay85 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !1945
  %arraydecay86 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1946
  %call87 = call i8* @strcat(i8* %arraydecay85, i8* %arraydecay86) #11, !dbg !1947
  %arraydecay88 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1948
  %44 = load i32, i32* @threads_per_block_on_fftx_3, align 4, !dbg !1949
  %call89 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay88, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.43, i64 0, i64 0), i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.50, i64 0, i64 0), i32 %44) #11, !dbg !1950
  %arraydecay90 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !1951
  %arraydecay91 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1952
  %call92 = call i8* @strcat(i8* %arraydecay90, i8* %arraydecay91) #11, !dbg !1953
  %arraydecay93 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1954
  %45 = load i32, i32* @threads_per_block_on_ffty_1, align 4, !dbg !1955
  %call94 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay93, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.43, i64 0, i64 0), i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.51, i64 0, i64 0), i32 %45) #11, !dbg !1956
  %arraydecay95 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !1957
  %arraydecay96 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1958
  %call97 = call i8* @strcat(i8* %arraydecay95, i8* %arraydecay96) #11, !dbg !1959
  %arraydecay98 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1960
  %46 = load i32, i32* @threads_per_block_on_ffty_2, align 4, !dbg !1961
  %call99 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay98, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.43, i64 0, i64 0), i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.52, i64 0, i64 0), i32 %46) #11, !dbg !1962
  %arraydecay100 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !1963
  %arraydecay101 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1964
  %call102 = call i8* @strcat(i8* %arraydecay100, i8* %arraydecay101) #11, !dbg !1965
  %arraydecay103 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1966
  %47 = load i32, i32* @threads_per_block_on_ffty_3, align 4, !dbg !1967
  %call104 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay103, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.43, i64 0, i64 0), i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.53, i64 0, i64 0), i32 %47) #11, !dbg !1968
  %arraydecay105 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !1969
  %arraydecay106 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1970
  %call107 = call i8* @strcat(i8* %arraydecay105, i8* %arraydecay106) #11, !dbg !1971
  %arraydecay108 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1972
  %48 = load i32, i32* @threads_per_block_on_fftz_1, align 4, !dbg !1973
  %call109 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay108, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.43, i64 0, i64 0), i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.54, i64 0, i64 0), i32 %48) #11, !dbg !1974
  %arraydecay110 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !1975
  %arraydecay111 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1976
  %call112 = call i8* @strcat(i8* %arraydecay110, i8* %arraydecay111) #11, !dbg !1977
  %arraydecay113 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1978
  %49 = load i32, i32* @threads_per_block_on_fftz_2, align 4, !dbg !1979
  %call114 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay113, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.43, i64 0, i64 0), i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.55, i64 0, i64 0), i32 %49) #11, !dbg !1980
  %arraydecay115 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !1981
  %arraydecay116 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1982
  %call117 = call i8* @strcat(i8* %arraydecay115, i8* %arraydecay116) #11, !dbg !1983
  %arraydecay118 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1984
  %50 = load i32, i32* @threads_per_block_on_fftz_3, align 4, !dbg !1985
  %call119 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay118, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.43, i64 0, i64 0), i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.56, i64 0, i64 0), i32 %50) #11, !dbg !1986
  %arraydecay120 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !1987
  %arraydecay121 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1988
  %call122 = call i8* @strcat(i8* %arraydecay120, i8* %arraydecay121) #11, !dbg !1989
  %arraydecay123 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1990
  %51 = load i32, i32* @threads_per_block_on_checksum, align 4, !dbg !1991
  %call124 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay123, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.43, i64 0, i64 0), i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.57, i64 0, i64 0), i32 %51) #11, !dbg !1992
  %arraydecay125 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !1993
  %arraydecay126 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1994
  %call127 = call i8* @strcat(i8* %arraydecay125, i8* %arraydecay126) #11, !dbg !1995
  %52 = load i8, i8* %class_npb, align 1, !dbg !1996
  %53 = load i32, i32* @_ZL5niter, align 4, !dbg !1997
  %54 = load i32, i32* %verified, align 4, !dbg !1998
  %arraydecay128 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !1999
  call void @_Z15c_print_resultsPcciiiiddS_iS_S_S_S_S_S_S_S_S_S_S_S_S_S_(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.58, i64 0, i64 0), i8 signext %52, i32 256, i32 256, i32 128, i32 %53, double 0.000000e+00, double %mflops.0, i8* getelementptr inbounds ([25 x i8], [25 x i8]* @.str.59, i64 0, i64 0), i32 %54, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.60, i64 0, i64 0), i8* getelementptr inbounds ([12 x i8], [12 x i8]* @.str.61, i64 0, i64 0), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.62, i64 0, i64 0), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.62, i64 0, i64 0), i8* getelementptr inbounds ([42 x i8], [42 x i8]* @.str.63, i64 0, i64 0), i8* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 0, i64 0), i8* %arraydecay128, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.64, i64 0, i64 0), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.65, i64 0, i64 0), i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.66, i64 0, i64 0), i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.67, i64 0, i64 0), i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.68, i64 0, i64 0), i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.68, i64 0, i64 0), i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.69, i64 0, i64 0)), !dbg !2000
  call void @_ZL11release_gpuv(), !dbg !2001
  %55 = load %struct.dcomplex*, %struct.dcomplex** @_ZL4sums, align 8, !dbg !2002
  %56 = bitcast %struct.dcomplex* %55 to i8*, !dbg !2002
  call void @free(i8* %56) #11, !dbg !2003
  %57 = load double*, double** @_ZL7twiddle, align 8, !dbg !2004
  %58 = bitcast double* %57 to i8*, !dbg !2004
  call void @free(i8* %58) #11, !dbg !2005
  %59 = load %struct.dcomplex*, %struct.dcomplex** @_ZL1u, align 8, !dbg !2006
  %60 = bitcast %struct.dcomplex* %59 to i8*, !dbg !2006
  call void @free(i8* %60) #11, !dbg !2007
  %61 = load %struct.dcomplex*, %struct.dcomplex** @_ZL2u0, align 8, !dbg !2008
  %62 = bitcast %struct.dcomplex* %61 to i8*, !dbg !2008
  call void @free(i8* %62) #11, !dbg !2009
  %63 = load %struct.dcomplex*, %struct.dcomplex** @_ZL2u1, align 8, !dbg !2010
  %64 = bitcast %struct.dcomplex* %63 to i8*, !dbg !2010
  call void @free(i8* %64) #11, !dbg !2011
  %65 = load i32*, i32** @_ZL4dims, align 8, !dbg !2012
  %66 = bitcast i32* %65 to i8*, !dbg !2012
  call void @free(i8* %66) #11, !dbg !2013
  ret i32 0, !dbg !2014
}

; Function Attrs: nounwind
declare dso_local noalias i8* @malloc(i64) #9

; Function Attrs: noinline uwtable
define internal void @_ZL5setupv() #7 !dbg !2015 {
entry:
  store i32 6, i32* @_ZL5niter, align 4, !dbg !2016
  %call = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([65 x i8], [65 x i8]* @.str.70, i64 0, i64 0)), !dbg !2017
  %call1 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([36 x i8], [36 x i8]* @.str.71, i64 0, i64 0), i32 256, i32 256, i32 128), !dbg !2018
  %0 = load i32, i32* @_ZL5niter, align 4, !dbg !2019
  %call2 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([35 x i8], [35 x i8]* @.str.72, i64 0, i64 0), i32 %0), !dbg !2020
  %call3 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.32, i64 0, i64 0)), !dbg !2021
  ret void, !dbg !2022
}

; Function Attrs: noinline uwtable
define internal void @_ZL9setup_gpuv() #7 !dbg !2023 {
entry:
  store i32 32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !2024
  store i32 32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !2025
  %0 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !2026
  %cmp = icmp sle i32 32, %0, !dbg !2028
  br i1 %cmp, label %if.then, label %if.else, !dbg !2029

if.then:                                          ; preds = %entry
  store i32 32, i32* @threads_per_block_on_compute_indexmap, align 4, !dbg !2030
  br label %if.end, !dbg !2032

if.else:                                          ; preds = %entry
  %1 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !2033
  store i32 %1, i32* @threads_per_block_on_compute_indexmap, align 4, !dbg !2035
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %2 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !2036
  %cmp1 = icmp sle i32 32, %2, !dbg !2038
  br i1 %cmp1, label %if.then2, label %if.else3, !dbg !2039

if.then2:                                         ; preds = %if.end
  store i32 32, i32* @threads_per_block_on_compute_initial_conditions, align 4, !dbg !2040
  br label %if.end4, !dbg !2042

if.else3:                                         ; preds = %if.end
  %3 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !2043
  store i32 %3, i32* @threads_per_block_on_compute_initial_conditions, align 4, !dbg !2045
  br label %if.end4

if.end4:                                          ; preds = %if.else3, %if.then2
  %4 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !2046
  %cmp5 = icmp sle i32 32, %4, !dbg !2048
  br i1 %cmp5, label %if.then6, label %if.else7, !dbg !2049

if.then6:                                         ; preds = %if.end4
  store i32 32, i32* @threads_per_block_on_init_ui, align 4, !dbg !2050
  br label %if.end8, !dbg !2052

if.else7:                                         ; preds = %if.end4
  %5 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !2053
  store i32 %5, i32* @threads_per_block_on_init_ui, align 4, !dbg !2055
  br label %if.end8

if.end8:                                          ; preds = %if.else7, %if.then6
  %6 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !2056
  %cmp9 = icmp sle i32 32, %6, !dbg !2058
  br i1 %cmp9, label %if.then10, label %if.else11, !dbg !2059

if.then10:                                        ; preds = %if.end8
  store i32 32, i32* @threads_per_block_on_evolve, align 4, !dbg !2060
  br label %if.end12, !dbg !2062

if.else11:                                        ; preds = %if.end8
  %7 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !2063
  store i32 %7, i32* @threads_per_block_on_evolve, align 4, !dbg !2065
  br label %if.end12

if.end12:                                         ; preds = %if.else11, %if.then10
  %8 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !2066
  %cmp13 = icmp sle i32 1024, %8, !dbg !2068
  br i1 %cmp13, label %if.then14, label %if.else15, !dbg !2069

if.then14:                                        ; preds = %if.end12
  store i32 1024, i32* @threads_per_block_on_fftx_1, align 4, !dbg !2070
  br label %if.end16, !dbg !2072

if.else15:                                        ; preds = %if.end12
  %9 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !2073
  store i32 %9, i32* @threads_per_block_on_fftx_1, align 4, !dbg !2075
  br label %if.end16

if.end16:                                         ; preds = %if.else15, %if.then14
  %10 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !2076
  %cmp17 = icmp sle i32 32, %10, !dbg !2078
  br i1 %cmp17, label %if.then18, label %if.else19, !dbg !2079

if.then18:                                        ; preds = %if.end16
  store i32 32, i32* @threads_per_block_on_fftx_2, align 4, !dbg !2080
  br label %if.end20, !dbg !2082

if.else19:                                        ; preds = %if.end16
  %11 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !2083
  store i32 %11, i32* @threads_per_block_on_fftx_2, align 4, !dbg !2085
  br label %if.end20

if.end20:                                         ; preds = %if.else19, %if.then18
  %12 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !2086
  %cmp21 = icmp sle i32 256, %12, !dbg !2088
  br i1 %cmp21, label %if.then22, label %if.else23, !dbg !2089

if.then22:                                        ; preds = %if.end20
  store i32 256, i32* @threads_per_block_on_fftx_3, align 4, !dbg !2090
  br label %if.end24, !dbg !2092

if.else23:                                        ; preds = %if.end20
  %13 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !2093
  store i32 %13, i32* @threads_per_block_on_fftx_3, align 4, !dbg !2095
  br label %if.end24

if.end24:                                         ; preds = %if.else23, %if.then22
  %14 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !2096
  %cmp25 = icmp sle i32 32, %14, !dbg !2098
  br i1 %cmp25, label %if.then26, label %if.else27, !dbg !2099

if.then26:                                        ; preds = %if.end24
  store i32 32, i32* @threads_per_block_on_ffty_1, align 4, !dbg !2100
  br label %if.end28, !dbg !2102

if.else27:                                        ; preds = %if.end24
  %15 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !2103
  store i32 %15, i32* @threads_per_block_on_ffty_1, align 4, !dbg !2105
  br label %if.end28

if.end28:                                         ; preds = %if.else27, %if.then26
  %16 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !2106
  %cmp29 = icmp sle i32 32, %16, !dbg !2108
  br i1 %cmp29, label %if.then30, label %if.else31, !dbg !2109

if.then30:                                        ; preds = %if.end28
  store i32 32, i32* @threads_per_block_on_ffty_2, align 4, !dbg !2110
  br label %if.end32, !dbg !2112

if.else31:                                        ; preds = %if.end28
  %17 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !2113
  store i32 %17, i32* @threads_per_block_on_ffty_2, align 4, !dbg !2115
  br label %if.end32

if.end32:                                         ; preds = %if.else31, %if.then30
  %18 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !2116
  %cmp33 = icmp sle i32 32, %18, !dbg !2118
  br i1 %cmp33, label %if.then34, label %if.else35, !dbg !2119

if.then34:                                        ; preds = %if.end32
  store i32 32, i32* @threads_per_block_on_ffty_3, align 4, !dbg !2120
  br label %if.end36, !dbg !2122

if.else35:                                        ; preds = %if.end32
  %19 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !2123
  store i32 %19, i32* @threads_per_block_on_ffty_3, align 4, !dbg !2125
  br label %if.end36

if.end36:                                         ; preds = %if.else35, %if.then34
  %20 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !2126
  %cmp37 = icmp sle i32 32, %20, !dbg !2128
  br i1 %cmp37, label %if.then38, label %if.else39, !dbg !2129

if.then38:                                        ; preds = %if.end36
  store i32 32, i32* @threads_per_block_on_fftz_1, align 4, !dbg !2130
  br label %if.end40, !dbg !2132

if.else39:                                        ; preds = %if.end36
  %21 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !2133
  store i32 %21, i32* @threads_per_block_on_fftz_1, align 4, !dbg !2135
  br label %if.end40

if.end40:                                         ; preds = %if.else39, %if.then38
  %22 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !2136
  %cmp41 = icmp sle i32 32, %22, !dbg !2138
  br i1 %cmp41, label %if.then42, label %if.else43, !dbg !2139

if.then42:                                        ; preds = %if.end40
  store i32 32, i32* @threads_per_block_on_fftz_2, align 4, !dbg !2140
  br label %if.end44, !dbg !2142

if.else43:                                        ; preds = %if.end40
  %23 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !2143
  store i32 %23, i32* @threads_per_block_on_fftz_2, align 4, !dbg !2145
  br label %if.end44

if.end44:                                         ; preds = %if.else43, %if.then42
  %24 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !2146
  %cmp45 = icmp sle i32 32, %24, !dbg !2148
  br i1 %cmp45, label %if.then46, label %if.else47, !dbg !2149

if.then46:                                        ; preds = %if.end44
  store i32 32, i32* @threads_per_block_on_fftz_3, align 4, !dbg !2150
  br label %if.end48, !dbg !2152

if.else47:                                        ; preds = %if.end44
  %25 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !2153
  store i32 %25, i32* @threads_per_block_on_fftz_3, align 4, !dbg !2155
  br label %if.end48

if.end48:                                         ; preds = %if.else47, %if.then46
  %26 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !2156
  %cmp49 = icmp sle i32 32, %26, !dbg !2158
  br i1 %cmp49, label %if.then50, label %if.else51, !dbg !2159

if.then50:                                        ; preds = %if.end48
  store i32 32, i32* @threads_per_block_on_checksum, align 4, !dbg !2160
  br label %if.end52, !dbg !2162

if.else51:                                        ; preds = %if.end48
  %27 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !2163
  store i32 %27, i32* @threads_per_block_on_checksum, align 4, !dbg !2165
  br label %if.end52

if.end52:                                         ; preds = %if.else51, %if.then50
  %28 = load i32, i32* @threads_per_block_on_compute_indexmap, align 4, !dbg !2166
  %conv = sitofp i32 %28 to double, !dbg !2166
  %div = fdiv double 0x4160000000000000, %conv, !dbg !2167
  %29 = call double @llvm.ceil.f64(double %div), !dbg !2168
  %conv53 = fptosi double %29 to i32, !dbg !2168
  store i32 %conv53, i32* @blocks_per_grid_on_compute_indexmap, align 4, !dbg !2169
  %30 = load i32, i32* @threads_per_block_on_compute_initial_conditions, align 4, !dbg !2170
  %conv54 = sitofp i32 %30 to double, !dbg !2170
  %div55 = fdiv double 1.280000e+02, %conv54, !dbg !2171
  %31 = call double @llvm.ceil.f64(double %div55), !dbg !2172
  %conv56 = fptosi double %31 to i32, !dbg !2172
  store i32 %conv56, i32* @blocks_per_grid_on_compute_initial_conditions, align 4, !dbg !2173
  %32 = load i32, i32* @threads_per_block_on_init_ui, align 4, !dbg !2174
  %conv57 = sitofp i32 %32 to double, !dbg !2174
  %div58 = fdiv double 0x4160000000000000, %conv57, !dbg !2175
  %33 = call double @llvm.ceil.f64(double %div58), !dbg !2176
  %conv59 = fptosi double %33 to i32, !dbg !2176
  store i32 %conv59, i32* @blocks_per_grid_on_init_ui, align 4, !dbg !2177
  %34 = load i32, i32* @threads_per_block_on_evolve, align 4, !dbg !2178
  %conv60 = sitofp i32 %34 to double, !dbg !2178
  %div61 = fdiv double 0x4160000000000000, %conv60, !dbg !2179
  %35 = call double @llvm.ceil.f64(double %div61), !dbg !2180
  %conv62 = fptosi double %35 to i32, !dbg !2180
  store i32 %conv62, i32* @blocks_per_grid_on_evolve, align 4, !dbg !2181
  %36 = load i32, i32* @threads_per_block_on_fftx_1, align 4, !dbg !2182
  %conv63 = sitofp i32 %36 to double, !dbg !2182
  %div64 = fdiv double 0x4160000000000000, %conv63, !dbg !2183
  %37 = call double @llvm.ceil.f64(double %div64), !dbg !2184
  %conv65 = fptosi double %37 to i32, !dbg !2184
  store i32 %conv65, i32* @blocks_per_grid_on_fftx_1, align 4, !dbg !2185
  %38 = load i32, i32* @threads_per_block_on_fftx_2, align 4, !dbg !2186
  %conv66 = sitofp i32 %38 to double, !dbg !2186
  %div67 = fdiv double 3.276800e+04, %conv66, !dbg !2187
  %39 = call double @llvm.ceil.f64(double %div67), !dbg !2188
  %conv68 = fptosi double %39 to i32, !dbg !2188
  store i32 %conv68, i32* @blocks_per_grid_on_fftx_2, align 4, !dbg !2189
  %40 = load i32, i32* @threads_per_block_on_fftx_3, align 4, !dbg !2190
  %conv69 = sitofp i32 %40 to double, !dbg !2190
  %div70 = fdiv double 0x4160000000000000, %conv69, !dbg !2191
  %41 = call double @llvm.ceil.f64(double %div70), !dbg !2192
  %conv71 = fptosi double %41 to i32, !dbg !2192
  store i32 %conv71, i32* @blocks_per_grid_on_fftx_3, align 4, !dbg !2193
  %42 = load i32, i32* @threads_per_block_on_ffty_1, align 4, !dbg !2194
  %conv72 = sitofp i32 %42 to double, !dbg !2194
  %div73 = fdiv double 0x4160000000000000, %conv72, !dbg !2195
  %43 = call double @llvm.ceil.f64(double %div73), !dbg !2196
  %conv74 = fptosi double %43 to i32, !dbg !2196
  store i32 %conv74, i32* @blocks_per_grid_on_ffty_1, align 4, !dbg !2197
  %44 = load i32, i32* @threads_per_block_on_ffty_2, align 4, !dbg !2198
  %conv75 = sitofp i32 %44 to double, !dbg !2198
  %div76 = fdiv double 3.276800e+04, %conv75, !dbg !2199
  %45 = call double @llvm.ceil.f64(double %div76), !dbg !2200
  %conv77 = fptosi double %45 to i32, !dbg !2200
  store i32 %conv77, i32* @blocks_per_grid_on_ffty_2, align 4, !dbg !2201
  %46 = load i32, i32* @threads_per_block_on_ffty_3, align 4, !dbg !2202
  %conv78 = sitofp i32 %46 to double, !dbg !2202
  %div79 = fdiv double 0x4160000000000000, %conv78, !dbg !2203
  %47 = call double @llvm.ceil.f64(double %div79), !dbg !2204
  %conv80 = fptosi double %47 to i32, !dbg !2204
  store i32 %conv80, i32* @blocks_per_grid_on_ffty_3, align 4, !dbg !2205
  %48 = load i32, i32* @threads_per_block_on_fftz_1, align 4, !dbg !2206
  %conv81 = sitofp i32 %48 to double, !dbg !2206
  %div82 = fdiv double 0x4160000000000000, %conv81, !dbg !2207
  %49 = call double @llvm.ceil.f64(double %div82), !dbg !2208
  %conv83 = fptosi double %49 to i32, !dbg !2208
  store i32 %conv83, i32* @blocks_per_grid_on_fftz_1, align 4, !dbg !2209
  %50 = load i32, i32* @threads_per_block_on_fftz_2, align 4, !dbg !2210
  %conv84 = sitofp i32 %50 to double, !dbg !2210
  %div85 = fdiv double 6.553600e+04, %conv84, !dbg !2211
  %51 = call double @llvm.ceil.f64(double %div85), !dbg !2212
  %conv86 = fptosi double %51 to i32, !dbg !2212
  store i32 %conv86, i32* @blocks_per_grid_on_fftz_2, align 4, !dbg !2213
  %52 = load i32, i32* @threads_per_block_on_fftz_3, align 4, !dbg !2214
  %conv87 = sitofp i32 %52 to double, !dbg !2214
  %div88 = fdiv double 0x4160000000000000, %conv87, !dbg !2215
  %53 = call double @llvm.ceil.f64(double %div88), !dbg !2216
  %conv89 = fptosi double %53 to i32, !dbg !2216
  store i32 %conv89, i32* @blocks_per_grid_on_fftz_3, align 4, !dbg !2217
  %54 = load i32, i32* @threads_per_block_on_checksum, align 4, !dbg !2218
  %conv90 = sitofp i32 %54 to double, !dbg !2218
  %div91 = fdiv double 1.024000e+03, %conv90, !dbg !2219
  %55 = call double @llvm.ceil.f64(double %div91), !dbg !2220
  %conv92 = fptosi double %55 to i32, !dbg !2220
  store i32 %conv92, i32* @blocks_per_grid_on_checksum, align 4, !dbg !2221
  store i64 112, i64* @size_sums_device, align 8, !dbg !2222
  store i64 1024, i64* @size_starts_device, align 8, !dbg !2223
  store i64 67108864, i64* @size_twiddle_device, align 8, !dbg !2224
  store i64 4096, i64* @size_u_device, align 8, !dbg !2225
  store i64 134217728, i64* @size_u0_device, align 8, !dbg !2226
  store i64 134217728, i64* @size_u1_device, align 8, !dbg !2227
  store i64 134217728, i64* @size_y0_device, align 8, !dbg !2228
  store i64 134217728, i64* @size_y1_device, align 8, !dbg !2229
  %56 = load i32, i32* @threads_per_block_on_checksum, align 4, !dbg !2230
  %conv93 = sext i32 %56 to i64, !dbg !2230
  %mul = mul i64 %conv93, 16, !dbg !2231
  store i64 %mul, i64* @size_shared_data, align 8, !dbg !2232
  call void @omp_set_num_threads(i32 3), !dbg !2233
  ret void, !dbg !2234
}

; Function Attrs: noinline uwtable
define internal void @_ZL11init_ui_gpuP8dcomplexS0_Pd(%struct.dcomplex* %u0, %struct.dcomplex* %u1, double* %twiddle) #7 !dbg !2235 {
entry:
  %agg.tmp = alloca %struct.dim3, align 4
  %agg.tmp1 = alloca %struct.dim3, align 4
  %agg.tmp.coerce = alloca { i64, i32 }, align 4
  %agg.tmp1.coerce = alloca { i64, i32 }, align 4
  call void @llvm.dbg.value(metadata %struct.dcomplex* %u0, metadata !2238, metadata !DIExpression()), !dbg !2239
  call void @llvm.dbg.value(metadata %struct.dcomplex* %u1, metadata !2240, metadata !DIExpression()), !dbg !2239
  call void @llvm.dbg.value(metadata double* %twiddle, metadata !2241, metadata !DIExpression()), !dbg !2239
  %0 = load i32, i32* @blocks_per_grid_on_init_ui, align 4, !dbg !2242
  %dim3gep.0 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp, i32 0, i32 0
  store i32 %0, i32* %dim3gep.0
  %dim3gep.1 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp, i32 0, i32 1
  store i32 1, i32* %dim3gep.1
  %dim3gep.2 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp, i32 0, i32 2
  store i32 1, i32* %dim3gep.2
  %1 = load i32, i32* @threads_per_block_on_init_ui, align 4, !dbg !2243
  %dim3gep.01 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp1, i32 0, i32 0
  store i32 %1, i32* %dim3gep.01
  %dim3gep.12 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp1, i32 0, i32 1
  store i32 1, i32* %dim3gep.12
  %dim3gep.23 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp1, i32 0, i32 2
  store i32 1, i32* %dim3gep.23
  %2 = bitcast { i64, i32 }* %agg.tmp.coerce to i8*, !dbg !2244
  %3 = bitcast %struct.dim3* %agg.tmp to i8*, !dbg !2244
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %2, i8* align 4 %3, i64 12, i1 false), !dbg !2244
  %4 = bitcast { i64, i32 }* %agg.tmp1.coerce to i8*, !dbg !2244
  %5 = bitcast %struct.dim3* %agg.tmp1 to i8*, !dbg !2244
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %4, i8* align 4 %5, i64 12, i1 false), !dbg !2244
  br label %header.0

header.0:                                         ; preds = %latch.0, %entry
  %indvar.0 = phi i32 [ 0, %entry ], [ %indvar.next.0, %latch.0 ]
  %exitcond4 = icmp ne i32 %indvar.0, %0
  br i1 %exitcond4, label %header.1.preheader, label %kcall.end, !tulip.doall.loop.grid !1860

header.1.preheader:                               ; preds = %header.0
  br label %header.1

header.1:                                         ; preds = %header.1.preheader, %latch.1
  %indvar.1 = phi i32 [ %indvar.next.1, %latch.1 ], [ 0, %header.1.preheader ]
  %exitcond = icmp ne i32 %indvar.1, %1
  br i1 %exitcond, label %kcall.configok, label %latch.0, !tulip.doall.loop.block !1860

latch.1:                                          ; preds = %kcall.configok
  %indvar.next.1 = add i32 %indvar.1, 1
  br label %header.1

latch.0:                                          ; preds = %header.1
  %indvar.next.0 = add i32 %indvar.0, 1
  br label %header.0

kcall.configok:                                   ; preds = %header.1
  call void @init_ui_gpu_kernel(%struct.dcomplex* %u0, %struct.dcomplex* %u1, double* %twiddle, i32 %0, i32 1, i32 1, i32 %1, i32 1, i32 1, i32 %indvar.0, i32 0, i32 0, i32 %indvar.1, i32 0, i32 0)
  br label %latch.1

kcall.end:                                        ; preds = %header.0
  ret void, !dbg !2245
}

declare dso_local i32 @omp_get_thread_num() #8

; Function Attrs: noinline uwtable
define internal void @_ZL20compute_indexmap_gpuPd(double* %twiddle) #7 !dbg !2246 {
entry:
  %agg.tmp = alloca %struct.dim3, align 4
  %agg.tmp1 = alloca %struct.dim3, align 4
  %agg.tmp.coerce = alloca { i64, i32 }, align 4
  %agg.tmp1.coerce = alloca { i64, i32 }, align 4
  call void @llvm.dbg.value(metadata double* %twiddle, metadata !2249, metadata !DIExpression()), !dbg !2250
  %0 = load i32, i32* @blocks_per_grid_on_compute_indexmap, align 4, !dbg !2251
  %dim3gep.0 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp, i32 0, i32 0
  store i32 %0, i32* %dim3gep.0
  %dim3gep.1 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp, i32 0, i32 1
  store i32 1, i32* %dim3gep.1
  %dim3gep.2 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp, i32 0, i32 2
  store i32 1, i32* %dim3gep.2
  %1 = load i32, i32* @threads_per_block_on_compute_indexmap, align 4, !dbg !2252
  %dim3gep.01 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp1, i32 0, i32 0
  store i32 %1, i32* %dim3gep.01
  %dim3gep.12 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp1, i32 0, i32 1
  store i32 1, i32* %dim3gep.12
  %dim3gep.23 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp1, i32 0, i32 2
  store i32 1, i32* %dim3gep.23
  %2 = bitcast { i64, i32 }* %agg.tmp.coerce to i8*, !dbg !2253
  %3 = bitcast %struct.dim3* %agg.tmp to i8*, !dbg !2253
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %2, i8* align 4 %3, i64 12, i1 false), !dbg !2253
  %4 = bitcast { i64, i32 }* %agg.tmp1.coerce to i8*, !dbg !2253
  %5 = bitcast %struct.dim3* %agg.tmp1 to i8*, !dbg !2253
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %4, i8* align 4 %5, i64 12, i1 false), !dbg !2253
  br label %header.0

header.0:                                         ; preds = %latch.0, %entry
  %indvar.0 = phi i32 [ 0, %entry ], [ %indvar.next.0, %latch.0 ]
  %exitcond4 = icmp ne i32 %indvar.0, %0
  br i1 %exitcond4, label %header.1.preheader, label %kcall.end, !tulip.doall.loop.grid !1860

header.1.preheader:                               ; preds = %header.0
  br label %header.1

header.1:                                         ; preds = %header.1.preheader, %latch.1
  %indvar.1 = phi i32 [ %indvar.next.1, %latch.1 ], [ 0, %header.1.preheader ]
  %exitcond = icmp ne i32 %indvar.1, %1
  br i1 %exitcond, label %kcall.configok, label %latch.0, !tulip.doall.loop.block !1860

latch.1:                                          ; preds = %kcall.configok
  %indvar.next.1 = add i32 %indvar.1, 1
  br label %header.1

latch.0:                                          ; preds = %header.1
  %indvar.next.0 = add i32 %indvar.0, 1
  br label %header.0

kcall.configok:                                   ; preds = %header.1
  call void @compute_indexmap_gpu_kernel(double* %twiddle, i32 %0, i32 1, i32 1, i32 %1, i32 1, i32 1, i32 %indvar.0, i32 0, i32 0, i32 %indvar.1, i32 0, i32 0)
  br label %latch.1

kcall.end:                                        ; preds = %header.0
  ret void, !dbg !2254
}

; Function Attrs: noinline uwtable
define internal void @_ZL30compute_initial_conditions_gpuP8dcomplex(%struct.dcomplex* %u0) #7 !dbg !2255 {
entry:
  %start = alloca double, align 8
  %an = alloca double, align 8
  %starts = alloca [128 x double], align 16
  %agg.tmp = alloca %struct.dim3, align 4
  %agg.tmp4 = alloca %struct.dim3, align 4
  %agg.tmp.coerce = alloca { i64, i32 }, align 4
  %agg.tmp4.coerce = alloca { i64, i32 }, align 4
  call void @llvm.dbg.value(metadata %struct.dcomplex* %u0, metadata !2258, metadata !DIExpression()), !dbg !2259
  call void @llvm.dbg.declare(metadata double* %start, metadata !2260, metadata !DIExpression()), !dbg !2261
  call void @llvm.dbg.declare(metadata double* %an, metadata !2262, metadata !DIExpression()), !dbg !2263
  call void @llvm.dbg.declare(metadata [128 x double]* %starts, metadata !2264, metadata !DIExpression()), !dbg !2268
  store double 0x41B2B9B0A1000000, double* %start, align 8, !dbg !2269
  call void @_ZL6ipow46diPd(double 0x41D2309CE5400000, i32 0, double* %an), !dbg !2270
  %0 = load double, double* %an, align 8, !dbg !2271
  %call = call double @_Z6randlcPdd(double* %start, double %0), !dbg !2272
  call void @_ZL6ipow46diPd(double 0x41D2309CE5400000, i32 131072, double* %an), !dbg !2273
  %1 = load double, double* %start, align 8, !dbg !2274
  %arrayidx = getelementptr inbounds [128 x double], [128 x double]* %starts, i64 0, i64 0, !dbg !2275
  store double %1, double* %arrayidx, align 16, !dbg !2276
  call void @llvm.dbg.value(metadata i32 1, metadata !2277, metadata !DIExpression()), !dbg !2259
  br label %for.cond, !dbg !2278

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 1, %entry ], !dbg !2280
  call void @llvm.dbg.value(metadata i64 %indvars.iv, metadata !2277, metadata !DIExpression()), !dbg !2259
  %exitcond6 = icmp ne i64 %indvars.iv, 128, !dbg !2281
  br i1 %exitcond6, label %for.body, label %for.end, !dbg !2283

for.body:                                         ; preds = %for.cond
  %2 = load double, double* %an, align 8, !dbg !2284
  %call1 = call double @_Z6randlcPdd(double* %start, double %2), !dbg !2286
  %3 = load double, double* %start, align 8, !dbg !2287
  %arrayidx2 = getelementptr inbounds [128 x double], [128 x double]* %starts, i64 0, i64 %indvars.iv, !dbg !2288
  store double %3, double* %arrayidx2, align 8, !dbg !2289
  br label %for.inc, !dbg !2290

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !2291
  call void @llvm.dbg.value(metadata i32 undef, metadata !2277, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !2259
  br label %for.cond, !dbg !2292, !llvm.loop !2293

for.end:                                          ; preds = %for.cond
  %staticArrayPtr1 = getelementptr [128 x double], [128 x double]* %starts, i64 0, i64 0
  %4 = bitcast double* %staticArrayPtr1 to i8*, !dbg !2295
  %arraydecay = getelementptr inbounds [128 x double], [128 x double]* %starts, i64 0, i64 0, !dbg !2296
  %5 = bitcast double* %arraydecay to i8*, !dbg !2296
  %6 = load i64, i64* @size_starts_device, align 8, !dbg !2297
  %call3 = call i32 @cudaMemcpy(i8* %4, i8* %5, i64 %6, i32 1), !dbg !2298, !tulip.target.start.of.map !1860
  %7 = load i32, i32* @blocks_per_grid_on_compute_initial_conditions, align 4, !dbg !2299
  %dim3gep.0 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp, i32 0, i32 0
  store i32 %7, i32* %dim3gep.0
  %dim3gep.1 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp, i32 0, i32 1
  store i32 1, i32* %dim3gep.1
  %dim3gep.2 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp, i32 0, i32 2
  store i32 1, i32* %dim3gep.2
  %8 = load i32, i32* @threads_per_block_on_compute_initial_conditions, align 4, !dbg !2300
  %dim3gep.02 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp4, i32 0, i32 0
  store i32 %8, i32* %dim3gep.02
  %dim3gep.13 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp4, i32 0, i32 1
  store i32 1, i32* %dim3gep.13
  %dim3gep.24 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp4, i32 0, i32 2
  store i32 1, i32* %dim3gep.24
  %9 = bitcast { i64, i32 }* %agg.tmp.coerce to i8*, !dbg !2301
  %10 = bitcast %struct.dim3* %agg.tmp to i8*, !dbg !2301
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %9, i8* align 4 %10, i64 12, i1 false), !dbg !2301
  %11 = bitcast { i64, i32 }* %agg.tmp4.coerce to i8*, !dbg !2301
  %12 = bitcast %struct.dim3* %agg.tmp4 to i8*, !dbg !2301
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %11, i8* align 4 %12, i64 12, i1 false), !dbg !2301
  br label %header.0

header.0:                                         ; preds = %latch.0, %for.end
  %indvar.0 = phi i32 [ 0, %for.end ], [ %indvar.next.0, %latch.0 ]
  %exitcond5 = icmp ne i32 %indvar.0, %7
  br i1 %exitcond5, label %header.1.preheader, label %kcall.end, !tulip.doall.loop.grid !1860

header.1.preheader:                               ; preds = %header.0
  br label %header.1

header.1:                                         ; preds = %header.1.preheader, %latch.1
  %indvar.1 = phi i32 [ %indvar.next.1, %latch.1 ], [ 0, %header.1.preheader ]
  %exitcond = icmp ne i32 %indvar.1, %8
  br i1 %exitcond, label %kcall.configok, label %latch.0, !tulip.doall.loop.block !1860

latch.1:                                          ; preds = %kcall.configok
  %indvar.next.1 = add i32 %indvar.1, 1
  br label %header.1

latch.0:                                          ; preds = %header.1
  %indvar.next.0 = add i32 %indvar.0, 1
  br label %header.0

kcall.configok:                                   ; preds = %header.1
  %staticArrayPtr = getelementptr [128 x double], [128 x double]* %starts, i64 0, i64 0
  call void @compute_initial_conditions_gpu_kernel(%struct.dcomplex* %u0, double* %staticArrayPtr, i32 %7, i32 1, i32 1, i32 %8, i32 1, i32 1, i32 %indvar.0, i32 0, i32 0, i32 %indvar.1, i32 0, i32 0)
  br label %latch.1

kcall.end:                                        ; preds = %header.0
  ret void, !dbg !2302
}

; Function Attrs: noinline uwtable
define internal void @_ZL12fft_init_gpui(i32 %n) #7 !dbg !2303 {
entry:
  %ref.tmp = alloca %struct.dcomplex, align 8
  %ref.tmp6 = alloca %struct.dcomplex, align 8
  call void @llvm.dbg.value(metadata i32 %n, metadata !2304, metadata !DIExpression()), !dbg !2305
  %call = call i32 @_ZL5ilog2i(i32 %n), !dbg !2306
  call void @llvm.dbg.value(metadata i32 %call, metadata !2307, metadata !DIExpression()), !dbg !2305
  %real = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp, i32 0, i32 0, !dbg !2308
  %conv = sitofp i32 %call to double, !dbg !2308
  store double %conv, double* %real, align 8, !dbg !2308
  %imag = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp, i32 0, i32 1, !dbg !2308
  store double 0.000000e+00, double* %imag, align 8, !dbg !2308
  %0 = load %struct.dcomplex*, %struct.dcomplex** @_ZL1u, align 8, !dbg !2309
  %arrayidx = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %0, i64 0, !dbg !2309
  %1 = bitcast %struct.dcomplex* %arrayidx to i8*, !dbg !2310
  %2 = bitcast %struct.dcomplex* %ref.tmp to i8*, !dbg !2310
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %1, i8* align 8 %2, i64 16, i1 false), !dbg !2310
  call void @llvm.dbg.value(metadata i32 2, metadata !2311, metadata !DIExpression()), !dbg !2305
  call void @llvm.dbg.value(metadata i32 1, metadata !2312, metadata !DIExpression()), !dbg !2305
  call void @llvm.dbg.value(metadata i32 1, metadata !2313, metadata !DIExpression()), !dbg !2305
  br label %for.cond, !dbg !2314

for.cond:                                         ; preds = %for.inc15, %entry
  %ku.0 = phi i32 [ 2, %entry ], [ %add13, %for.inc15 ], !dbg !2305
  %j.0 = phi i32 [ 1, %entry ], [ %inc16, %for.inc15 ], !dbg !2316
  %ln.0 = phi i32 [ 1, %entry ], [ %mul14, %for.inc15 ], !dbg !2305
  call void @llvm.dbg.value(metadata i32 %ln.0, metadata !2312, metadata !DIExpression()), !dbg !2305
  call void @llvm.dbg.value(metadata i32 %j.0, metadata !2313, metadata !DIExpression()), !dbg !2305
  call void @llvm.dbg.value(metadata i32 %ku.0, metadata !2311, metadata !DIExpression()), !dbg !2305
  %cmp = icmp sle i32 %j.0, %call, !dbg !2317
  br i1 %cmp, label %for.body, label %for.end17, !dbg !2319

for.body:                                         ; preds = %for.cond
  %conv1 = sitofp i32 %ln.0 to double, !dbg !2320
  %div = fdiv double 0x400921FB54442D18, %conv1, !dbg !2322
  call void @llvm.dbg.value(metadata double %div, metadata !2323, metadata !DIExpression()), !dbg !2305
  call void @llvm.dbg.value(metadata i32 0, metadata !2324, metadata !DIExpression()), !dbg !2305
  %3 = sext i32 %ku.0 to i64, !dbg !2325
  br label %for.cond2, !dbg !2325

for.cond2:                                        ; preds = %for.inc, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %for.body ], !dbg !2327
  call void @llvm.dbg.value(metadata i64 %indvars.iv, metadata !2324, metadata !DIExpression()), !dbg !2305
  %sub = sub nsw i32 %ln.0, 1, !dbg !2328
  %4 = sext i32 %sub to i64, !dbg !2330
  %cmp3 = icmp sle i64 %indvars.iv, %4, !dbg !2330
  br i1 %cmp3, label %for.body4, label %for.end, !dbg !2331

for.body4:                                        ; preds = %for.cond2
  %5 = trunc i64 %indvars.iv to i32, !dbg !2332
  %conv5 = sitofp i32 %5 to double, !dbg !2332
  %mul = fmul contract double %conv5, %div, !dbg !2334
  call void @llvm.dbg.value(metadata double %mul, metadata !2335, metadata !DIExpression()), !dbg !2305
  %real7 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp6, i32 0, i32 0, !dbg !2336
  %call8 = call double @cos(double %mul) #11, !dbg !2336
  store double %call8, double* %real7, align 8, !dbg !2336
  %imag9 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp6, i32 0, i32 1, !dbg !2336
  %call10 = call double @sin(double %mul) #11, !dbg !2336
  store double %call10, double* %imag9, align 8, !dbg !2336
  %6 = load %struct.dcomplex*, %struct.dcomplex** @_ZL1u, align 8, !dbg !2337
  %7 = add nuw nsw i64 %indvars.iv, %3, !dbg !2338
  %8 = sub nsw i64 %7, 1, !dbg !2339
  %arrayidx12 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %6, i64 %8, !dbg !2337
  %9 = bitcast %struct.dcomplex* %arrayidx12 to i8*, !dbg !2340
  %10 = bitcast %struct.dcomplex* %ref.tmp6 to i8*, !dbg !2340
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %9, i8* align 8 %10, i64 16, i1 false), !dbg !2340
  br label %for.inc, !dbg !2341

for.inc:                                          ; preds = %for.body4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !2342
  call void @llvm.dbg.value(metadata i32 undef, metadata !2324, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !2305
  br label %for.cond2, !dbg !2343, !llvm.loop !2344

for.end:                                          ; preds = %for.cond2
  %add13 = add nsw i32 %ku.0, %ln.0, !dbg !2346
  call void @llvm.dbg.value(metadata i32 %add13, metadata !2311, metadata !DIExpression()), !dbg !2305
  %mul14 = mul nuw nsw i32 2, %ln.0, !dbg !2347
  call void @llvm.dbg.value(metadata i32 %mul14, metadata !2312, metadata !DIExpression()), !dbg !2305
  br label %for.inc15, !dbg !2348

for.inc15:                                        ; preds = %for.end
  %inc16 = add nuw nsw i32 %j.0, 1, !dbg !2349
  call void @llvm.dbg.value(metadata i32 %inc16, metadata !2313, metadata !DIExpression()), !dbg !2305
  br label %for.cond, !dbg !2350, !llvm.loop !2351

for.end17:                                        ; preds = %for.cond
  %11 = load %struct.dcomplex*, %struct.dcomplex** @_ZL1u, align 8, !dbg !2353
  %12 = bitcast %struct.dcomplex* %11 to i8*, !dbg !2353
  %13 = load %struct.dcomplex*, %struct.dcomplex** @_ZL1u, align 8, !dbg !2354
  %14 = bitcast %struct.dcomplex* %13 to i8*, !dbg !2354
  %15 = load i64, i64* @size_u_device, align 8, !dbg !2355
  %call18 = call i32 @cudaMemcpy(i8* %12, i8* %14, i64 %15, i32 1), !dbg !2356, !tulip.target.start.of.map !1860
  ret void, !dbg !2357
}

declare dso_local i32 @cudaDeviceSynchronize() #8

; Function Attrs: noinline uwtable
define internal void @_ZL7fft_gpuiP8dcomplexS0_(i32 %dir, %struct.dcomplex* %x1, %struct.dcomplex* %x2) #7 !dbg !2358 {
entry:
  call void @llvm.dbg.value(metadata i32 %dir, metadata !2361, metadata !DIExpression()), !dbg !2362
  call void @llvm.dbg.value(metadata %struct.dcomplex* %x1, metadata !2363, metadata !DIExpression()), !dbg !2362
  call void @llvm.dbg.value(metadata %struct.dcomplex* %x2, metadata !2364, metadata !DIExpression()), !dbg !2362
  %cmp = icmp eq i32 %dir, 1, !dbg !2365
  br i1 %cmp, label %if.then, label %if.else, !dbg !2367

if.then:                                          ; preds = %entry
  %0 = load %struct.dcomplex*, %struct.dcomplex** @_ZL1u, align 8, !dbg !2368
  %1 = load %struct.dcomplex*, %struct.dcomplex** @y0_device, align 8, !dbg !2370
  %2 = load %struct.dcomplex*, %struct.dcomplex** @y1_device, align 8, !dbg !2371
  call void @_ZL10cffts1_gpuiP8dcomplexS0_S0_S0_S0_(i32 1, %struct.dcomplex* %0, %struct.dcomplex* %x1, %struct.dcomplex* %x1, %struct.dcomplex* %1, %struct.dcomplex* %2), !dbg !2372
  %3 = load %struct.dcomplex*, %struct.dcomplex** @_ZL1u, align 8, !dbg !2373
  %4 = load %struct.dcomplex*, %struct.dcomplex** @y0_device, align 8, !dbg !2374
  %5 = load %struct.dcomplex*, %struct.dcomplex** @y1_device, align 8, !dbg !2375
  call void @_ZL10cffts2_gpuiP8dcomplexS0_S0_S0_S0_(i32 1, %struct.dcomplex* %3, %struct.dcomplex* %x1, %struct.dcomplex* %x1, %struct.dcomplex* %4, %struct.dcomplex* %5), !dbg !2376
  %6 = load %struct.dcomplex*, %struct.dcomplex** @_ZL1u, align 8, !dbg !2377
  %7 = load %struct.dcomplex*, %struct.dcomplex** @y0_device, align 8, !dbg !2378
  %8 = load %struct.dcomplex*, %struct.dcomplex** @y1_device, align 8, !dbg !2379
  call void @_ZL10cffts3_gpuiP8dcomplexS0_S0_S0_S0_(i32 1, %struct.dcomplex* %6, %struct.dcomplex* %x1, %struct.dcomplex* %x2, %struct.dcomplex* %7, %struct.dcomplex* %8), !dbg !2380
  br label %if.end, !dbg !2381

if.else:                                          ; preds = %entry
  %9 = load %struct.dcomplex*, %struct.dcomplex** @_ZL1u, align 8, !dbg !2382
  %10 = load %struct.dcomplex*, %struct.dcomplex** @y0_device, align 8, !dbg !2384
  %11 = load %struct.dcomplex*, %struct.dcomplex** @y1_device, align 8, !dbg !2385
  call void @_ZL10cffts3_gpuiP8dcomplexS0_S0_S0_S0_(i32 -1, %struct.dcomplex* %9, %struct.dcomplex* %x1, %struct.dcomplex* %x1, %struct.dcomplex* %10, %struct.dcomplex* %11), !dbg !2386
  %12 = load %struct.dcomplex*, %struct.dcomplex** @_ZL1u, align 8, !dbg !2387
  %13 = load %struct.dcomplex*, %struct.dcomplex** @y0_device, align 8, !dbg !2388
  %14 = load %struct.dcomplex*, %struct.dcomplex** @y1_device, align 8, !dbg !2389
  call void @_ZL10cffts2_gpuiP8dcomplexS0_S0_S0_S0_(i32 -1, %struct.dcomplex* %12, %struct.dcomplex* %x1, %struct.dcomplex* %x1, %struct.dcomplex* %13, %struct.dcomplex* %14), !dbg !2390
  %15 = load %struct.dcomplex*, %struct.dcomplex** @_ZL1u, align 8, !dbg !2391
  %16 = load %struct.dcomplex*, %struct.dcomplex** @y0_device, align 8, !dbg !2392
  %17 = load %struct.dcomplex*, %struct.dcomplex** @y1_device, align 8, !dbg !2393
  call void @_ZL10cffts1_gpuiP8dcomplexS0_S0_S0_S0_(i32 -1, %struct.dcomplex* %15, %struct.dcomplex* %x1, %struct.dcomplex* %x2, %struct.dcomplex* %16, %struct.dcomplex* %17), !dbg !2394
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void, !dbg !2395
}

; Function Attrs: noinline uwtable
define internal void @_ZL10evolve_gpuP8dcomplexS0_Pd(%struct.dcomplex* %u0, %struct.dcomplex* %u1, double* %twiddle) #7 !dbg !2396 {
entry:
  %agg.tmp = alloca %struct.dim3, align 4
  %agg.tmp1 = alloca %struct.dim3, align 4
  %agg.tmp.coerce = alloca { i64, i32 }, align 4
  %agg.tmp1.coerce = alloca { i64, i32 }, align 4
  call void @llvm.dbg.value(metadata %struct.dcomplex* %u0, metadata !2397, metadata !DIExpression()), !dbg !2398
  call void @llvm.dbg.value(metadata %struct.dcomplex* %u1, metadata !2399, metadata !DIExpression()), !dbg !2398
  call void @llvm.dbg.value(metadata double* %twiddle, metadata !2400, metadata !DIExpression()), !dbg !2398
  %0 = load i32, i32* @blocks_per_grid_on_evolve, align 4, !dbg !2401
  %dim3gep.0 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp, i32 0, i32 0
  store i32 %0, i32* %dim3gep.0
  %dim3gep.1 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp, i32 0, i32 1
  store i32 1, i32* %dim3gep.1
  %dim3gep.2 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp, i32 0, i32 2
  store i32 1, i32* %dim3gep.2
  %1 = load i32, i32* @threads_per_block_on_evolve, align 4, !dbg !2402
  %dim3gep.01 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp1, i32 0, i32 0
  store i32 %1, i32* %dim3gep.01
  %dim3gep.12 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp1, i32 0, i32 1
  store i32 1, i32* %dim3gep.12
  %dim3gep.23 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp1, i32 0, i32 2
  store i32 1, i32* %dim3gep.23
  %2 = bitcast { i64, i32 }* %agg.tmp.coerce to i8*, !dbg !2403
  %3 = bitcast %struct.dim3* %agg.tmp to i8*, !dbg !2403
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %2, i8* align 4 %3, i64 12, i1 false), !dbg !2403
  %4 = bitcast { i64, i32 }* %agg.tmp1.coerce to i8*, !dbg !2403
  %5 = bitcast %struct.dim3* %agg.tmp1 to i8*, !dbg !2403
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %4, i8* align 4 %5, i64 12, i1 false), !dbg !2403
  br label %header.0

header.0:                                         ; preds = %latch.0, %entry
  %indvar.0 = phi i32 [ 0, %entry ], [ %indvar.next.0, %latch.0 ]
  %exitcond4 = icmp ne i32 %indvar.0, %0
  br i1 %exitcond4, label %header.1.preheader, label %kcall.end, !tulip.doall.loop.grid !1860

header.1.preheader:                               ; preds = %header.0
  br label %header.1

header.1:                                         ; preds = %header.1.preheader, %latch.1
  %indvar.1 = phi i32 [ %indvar.next.1, %latch.1 ], [ 0, %header.1.preheader ]
  %exitcond = icmp ne i32 %indvar.1, %1
  br i1 %exitcond, label %kcall.configok, label %latch.0, !tulip.doall.loop.block !1860

latch.1:                                          ; preds = %kcall.configok
  %indvar.next.1 = add i32 %indvar.1, 1
  br label %header.1

latch.0:                                          ; preds = %header.1
  %indvar.next.0 = add i32 %indvar.0, 1
  br label %header.0

kcall.configok:                                   ; preds = %header.1
  call void @evolve_gpu_kernel(%struct.dcomplex* %u0, %struct.dcomplex* %u1, double* %twiddle, i32 %0, i32 1, i32 1, i32 %1, i32 1, i32 1, i32 %indvar.0, i32 0, i32 0, i32 %indvar.1, i32 0, i32 0)
  br label %latch.1

kcall.end:                                        ; preds = %header.0
  ret void, !dbg !2404
}

; Function Attrs: noinline uwtable
define internal void @_ZL12checksum_gpuiP8dcomplex(i32 %iteration, %struct.dcomplex* %u1) #7 !dbg !2405 {
entry:
  %agg.tmp = alloca %struct.dim3, align 4
  %agg.tmp1 = alloca %struct.dim3, align 4
  %agg.tmp.coerce = alloca { i64, i32 }, align 4
  %agg.tmp1.coerce = alloca { i64, i32 }, align 4
  call void @llvm.dbg.value(metadata i32 %iteration, metadata !2408, metadata !DIExpression()), !dbg !2409
  call void @llvm.dbg.value(metadata %struct.dcomplex* %u1, metadata !2410, metadata !DIExpression()), !dbg !2409
  %0 = load i32, i32* @blocks_per_grid_on_checksum, align 4, !dbg !2411
  %dim3gep.0 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp, i32 0, i32 0
  store i32 %0, i32* %dim3gep.0
  %dim3gep.1 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp, i32 0, i32 1
  store i32 1, i32* %dim3gep.1
  %dim3gep.2 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp, i32 0, i32 2
  store i32 1, i32* %dim3gep.2
  %1 = load i32, i32* @threads_per_block_on_checksum, align 4, !dbg !2412
  %dim3gep.01 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp1, i32 0, i32 0
  store i32 %1, i32* %dim3gep.01
  %dim3gep.12 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp1, i32 0, i32 1
  store i32 1, i32* %dim3gep.12
  %dim3gep.23 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp1, i32 0, i32 2
  store i32 1, i32* %dim3gep.23
  %2 = bitcast { i64, i32 }* %agg.tmp.coerce to i8*, !dbg !2413
  %3 = bitcast %struct.dim3* %agg.tmp to i8*, !dbg !2413
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %2, i8* align 4 %3, i64 12, i1 false), !dbg !2413
  %4 = bitcast { i64, i32 }* %agg.tmp1.coerce to i8*, !dbg !2413
  %5 = bitcast %struct.dim3* %agg.tmp1 to i8*, !dbg !2413
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %4, i8* align 4 %5, i64 12, i1 false), !dbg !2413
  br label %header.0

header.0:                                         ; preds = %latch.0, %entry
  %indvar.0 = phi i32 [ 0, %entry ], [ %indvar.next.0, %latch.0 ]
  %exitcond5 = icmp ne i32 %indvar.0, %0
  br i1 %exitcond5, label %header.1.preheader.clone0, label %kcall.end, !tulip.doall.loop.grid !1860

header.1.preheader.clone0:                        ; preds = %header.0
  br label %header.1.clone0

header.1.clone0:                                  ; preds = %latch.1.clone0, %header.1.preheader.clone0
  %indvar.1.clone0 = phi i32 [ %indvar.next.1.clone0, %latch.1.clone0 ], [ 0, %header.1.preheader.clone0 ]
  %exitcond = icmp ne i32 %indvar.1.clone0, %1
  br i1 %exitcond, label %kcall.configok.clone0, label %header.1.preheader

kcall.configok.clone0:                            ; preds = %header.1.clone0
  %6 = load %struct.dcomplex*, %struct.dcomplex** @_ZL4sums, align 8, !dbg !2414
  call void @checksum_gpu_kernel0(i32 %iteration, %struct.dcomplex* %u1, %struct.dcomplex* %6, i32 %0, i32 1, i32 1, i32 %1, i32 1, i32 1, i32 %indvar.0, i32 0, i32 0, i32 %indvar.1.clone0, i32 0, i32 0)
  br label %latch.1.clone0

latch.1.clone0:                                   ; preds = %kcall.configok.clone0
  %indvar.next.1.clone0 = add i32 %indvar.1.clone0, 1
  br label %header.1.clone0

header.1.preheader:                               ; preds = %header.1.clone0
  br label %header.1

header.1:                                         ; preds = %header.1.preheader, %latch.1
  %indvar.1 = phi i32 [ %indvar.next.1, %latch.1 ], [ 0, %header.1.preheader ]
  %exitcond4 = icmp ne i32 %indvar.1, %1
  br i1 %exitcond4, label %kcall.configok, label %latch.0

latch.1:                                          ; preds = %kcall.configok
  %indvar.next.1 = add i32 %indvar.1, 1
  br label %header.1

latch.0:                                          ; preds = %header.1
  %indvar.next.0 = add i32 %indvar.0, 1
  br label %header.0

kcall.configok:                                   ; preds = %header.1
  %7 = load %struct.dcomplex*, %struct.dcomplex** @_ZL4sums, align 8, !dbg !2414
  call void @checksum_gpu_kernel1(i32 %iteration, %struct.dcomplex* %u1, %struct.dcomplex* %7, i32 %0, i32 1, i32 1, i32 %1, i32 1, i32 1, i32 %indvar.0, i32 0, i32 0, i32 %indvar.1, i32 0, i32 0)
  br label %latch.1

kcall.end:                                        ; preds = %header.0
  ret void, !dbg !2415
}

declare dso_local i32 @cudaMemcpy(i8*, i8*, i64, i32) #8

; Function Attrs: noinline uwtable
define internal void @_ZL6verifyiiiiPiPc(i32 %d1, i32 %d2, i32 %d3, i32 %nt, i32* %verified, i8* %class_npb) #7 !dbg !2416 {
entry:
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
  call void @llvm.dbg.value(metadata i32 %d1, metadata !2420, metadata !DIExpression()), !dbg !2421
  call void @llvm.dbg.value(metadata i32 %d2, metadata !2422, metadata !DIExpression()), !dbg !2421
  call void @llvm.dbg.value(metadata i32 %d3, metadata !2423, metadata !DIExpression()), !dbg !2421
  call void @llvm.dbg.value(metadata i32 %nt, metadata !2424, metadata !DIExpression()), !dbg !2421
  call void @llvm.dbg.value(metadata i32* %verified, metadata !2425, metadata !DIExpression()), !dbg !2421
  call void @llvm.dbg.value(metadata i8* %class_npb, metadata !2426, metadata !DIExpression()), !dbg !2421
  call void @llvm.dbg.declare(metadata [26 x %struct.dcomplex]* %csum_ref, metadata !2427, metadata !DIExpression()), !dbg !2431
  store i8 85, i8* %class_npb, align 1, !dbg !2432
  call void @llvm.dbg.value(metadata double 0x3D719799812DEA11, metadata !2433, metadata !DIExpression()), !dbg !2421
  store i32 0, i32* %verified, align 4, !dbg !2434
  %cmp = icmp eq i32 %d1, 64, !dbg !2435
  br i1 %cmp, label %land.lhs.true, label %if.else, !dbg !2437

land.lhs.true:                                    ; preds = %entry
  %cmp1 = icmp eq i32 %d2, 64, !dbg !2438
  br i1 %cmp1, label %land.lhs.true2, label %if.else, !dbg !2439

land.lhs.true2:                                   ; preds = %land.lhs.true
  %cmp3 = icmp eq i32 %d3, 64, !dbg !2440
  br i1 %cmp3, label %land.lhs.true4, label %if.else, !dbg !2441

land.lhs.true4:                                   ; preds = %land.lhs.true2
  %cmp5 = icmp eq i32 %nt, 6, !dbg !2442
  br i1 %cmp5, label %if.then, label %if.else, !dbg !2443

if.then:                                          ; preds = %land.lhs.true4
  store i8 83, i8* %class_npb, align 1, !dbg !2444
  %real = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp, i32 0, i32 0, !dbg !2446
  store double 0x408154DE9E5DA8C7, double* %real, align 8, !dbg !2446
  %imag = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp, i32 0, i32 1, !dbg !2446
  store double 0x407E4894D21E84F6, double* %imag, align 8, !dbg !2446
  %arrayidx = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 1, !dbg !2447
  %0 = bitcast %struct.dcomplex* %arrayidx to i8*, !dbg !2448
  %1 = bitcast %struct.dcomplex* %ref.tmp to i8*, !dbg !2448
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %0, i8* align 8 %1, i64 16, i1 false), !dbg !2448
  %real7 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp6, i32 0, i32 0, !dbg !2449
  store double 0x4081551BBB575EAB, double* %real7, align 8, !dbg !2449
  %imag8 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp6, i32 0, i32 1, !dbg !2449
  store double 0x407E687CA0F87E44, double* %imag8, align 8, !dbg !2449
  %arrayidx9 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 2, !dbg !2450
  %2 = bitcast %struct.dcomplex* %arrayidx9 to i8*, !dbg !2451
  %3 = bitcast %struct.dcomplex* %ref.tmp6 to i8*, !dbg !2451
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %2, i8* align 8 %3, i64 16, i1 false), !dbg !2451
  %real11 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp10, i32 0, i32 0, !dbg !2452
  store double 0x408154EB318EB593, double* %real11, align 8, !dbg !2452
  %imag12 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp10, i32 0, i32 1, !dbg !2452
  store double 0x407E8641D4F55AF9, double* %imag12, align 8, !dbg !2452
  %arrayidx13 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 3, !dbg !2453
  %4 = bitcast %struct.dcomplex* %arrayidx13 to i8*, !dbg !2454
  %5 = bitcast %struct.dcomplex* %ref.tmp10 to i8*, !dbg !2454
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %4, i8* align 8 %5, i64 16, i1 false), !dbg !2454
  %real15 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp14, i32 0, i32 0, !dbg !2455
  store double 0x40815456C13A7B04, double* %real15, align 8, !dbg !2455
  %imag16 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp14, i32 0, i32 1, !dbg !2455
  store double 0x407EA2097D7357C2, double* %imag16, align 8, !dbg !2455
  %arrayidx17 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 4, !dbg !2456
  %6 = bitcast %struct.dcomplex* %arrayidx17 to i8*, !dbg !2457
  %7 = bitcast %struct.dcomplex* %ref.tmp14 to i8*, !dbg !2457
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %6, i8* align 8 %7, i64 16, i1 false), !dbg !2457
  %real19 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp18, i32 0, i32 0, !dbg !2458
  store double 0x408153676E9F169C, double* %real19, align 8, !dbg !2458
  %imag20 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp18, i32 0, i32 1, !dbg !2458
  store double 0x407EBBF61C86EF29, double* %imag20, align 8, !dbg !2458
  %arrayidx21 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 5, !dbg !2459
  %8 = bitcast %struct.dcomplex* %arrayidx21 to i8*, !dbg !2460
  %9 = bitcast %struct.dcomplex* %ref.tmp18 to i8*, !dbg !2460
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %8, i8* align 8 %9, i64 16, i1 false), !dbg !2460
  %real23 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp22, i32 0, i32 0, !dbg !2461
  store double 0x408152259010E0A1, double* %real23, align 8, !dbg !2461
  %imag24 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp22, i32 0, i32 1, !dbg !2461
  store double 0x407ED427D4DF0213, double* %imag24, align 8, !dbg !2461
  %arrayidx25 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 6, !dbg !2462
  %10 = bitcast %struct.dcomplex* %arrayidx25 to i8*, !dbg !2463
  %11 = bitcast %struct.dcomplex* %ref.tmp22 to i8*, !dbg !2463
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %10, i8* align 8 %11, i64 16, i1 false), !dbg !2463
  br label %if.end492, !dbg !2464

if.else:                                          ; preds = %land.lhs.true4, %land.lhs.true2, %land.lhs.true, %entry
  %cmp26 = icmp eq i32 %d1, 128, !dbg !2465
  br i1 %cmp26, label %land.lhs.true27, label %if.else58, !dbg !2467

land.lhs.true27:                                  ; preds = %if.else
  %cmp28 = icmp eq i32 %d2, 128, !dbg !2468
  br i1 %cmp28, label %land.lhs.true29, label %if.else58, !dbg !2469

land.lhs.true29:                                  ; preds = %land.lhs.true27
  %cmp30 = icmp eq i32 %d3, 32, !dbg !2470
  br i1 %cmp30, label %land.lhs.true31, label %if.else58, !dbg !2471

land.lhs.true31:                                  ; preds = %land.lhs.true29
  %cmp32 = icmp eq i32 %nt, 6, !dbg !2472
  br i1 %cmp32, label %if.then33, label %if.else58, !dbg !2473

if.then33:                                        ; preds = %land.lhs.true31
  store i8 87, i8* %class_npb, align 1, !dbg !2474
  %real35 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp34, i32 0, i32 0, !dbg !2476
  store double 0x4081BAE3C635196D, double* %real35, align 8, !dbg !2476
  %imag36 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp34, i32 0, i32 1, !dbg !2476
  store double 0x40808A98F467F156, double* %imag36, align 8, !dbg !2476
  %arrayidx37 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 1, !dbg !2477
  %12 = bitcast %struct.dcomplex* %arrayidx37 to i8*, !dbg !2478
  %13 = bitcast %struct.dcomplex* %ref.tmp34 to i8*, !dbg !2478
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %12, i8* align 8 %13, i64 16, i1 false), !dbg !2478
  %real39 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp38, i32 0, i32 0, !dbg !2479
  store double 0x40819926462BA5A4, double* %real39, align 8, !dbg !2479
  %imag40 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp38, i32 0, i32 1, !dbg !2479
  store double 0x408081B851380EB7, double* %imag40, align 8, !dbg !2479
  %arrayidx41 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 2, !dbg !2480
  %14 = bitcast %struct.dcomplex* %arrayidx41 to i8*, !dbg !2481
  %15 = bitcast %struct.dcomplex* %ref.tmp38 to i8*, !dbg !2481
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %14, i8* align 8 %15, i64 16, i1 false), !dbg !2481
  %real43 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp42, i32 0, i32 0, !dbg !2482
  store double 0x40817B3822354DD9, double* %real43, align 8, !dbg !2482
  %imag44 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp42, i32 0, i32 1, !dbg !2482
  store double 0x408078CC18578DFC, double* %imag44, align 8, !dbg !2482
  %arrayidx45 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 3, !dbg !2483
  %16 = bitcast %struct.dcomplex* %arrayidx45 to i8*, !dbg !2484
  %17 = bitcast %struct.dcomplex* %ref.tmp42 to i8*, !dbg !2484
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %16, i8* align 8 %17, i64 16, i1 false), !dbg !2484
  %real47 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp46, i32 0, i32 0, !dbg !2485
  store double 0x4081608EF5C48194, double* %real47, align 8, !dbg !2485
  %imag48 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp46, i32 0, i32 1, !dbg !2485
  store double 0x40807005B7059038, double* %imag48, align 8, !dbg !2485
  %arrayidx49 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 4, !dbg !2486
  %18 = bitcast %struct.dcomplex* %arrayidx49 to i8*, !dbg !2487
  %19 = bitcast %struct.dcomplex* %ref.tmp46 to i8*, !dbg !2487
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %18, i8* align 8 %19, i64 16, i1 false), !dbg !2487
  %real51 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp50, i32 0, i32 0, !dbg !2488
  store double 0x408148B81D084E83, double* %real51, align 8, !dbg !2488
  %imag52 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp50, i32 0, i32 1, !dbg !2488
  store double 0x408067854B0E36C9, double* %imag52, align 8, !dbg !2488
  %arrayidx53 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 5, !dbg !2489
  %20 = bitcast %struct.dcomplex* %arrayidx53 to i8*, !dbg !2490
  %21 = bitcast %struct.dcomplex* %ref.tmp50 to i8*, !dbg !2490
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %20, i8* align 8 %21, i64 16, i1 false), !dbg !2490
  %real55 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp54, i32 0, i32 0, !dbg !2491
  store double 0x40813353E9E3E09A, double* %real55, align 8, !dbg !2491
  %imag56 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp54, i32 0, i32 1, !dbg !2491
  store double 0x40805F5EAB0F5DA2, double* %imag56, align 8, !dbg !2491
  %arrayidx57 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 6, !dbg !2492
  %22 = bitcast %struct.dcomplex* %arrayidx57 to i8*, !dbg !2493
  %23 = bitcast %struct.dcomplex* %ref.tmp54 to i8*, !dbg !2493
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %22, i8* align 8 %23, i64 16, i1 false), !dbg !2493
  br label %if.end491, !dbg !2494

if.else58:                                        ; preds = %land.lhs.true31, %land.lhs.true29, %land.lhs.true27, %if.else
  %cmp59 = icmp eq i32 %d1, 256, !dbg !2495
  br i1 %cmp59, label %land.lhs.true60, label %if.else91, !dbg !2497

land.lhs.true60:                                  ; preds = %if.else58
  %cmp61 = icmp eq i32 %d2, 256, !dbg !2498
  br i1 %cmp61, label %land.lhs.true62, label %if.else91, !dbg !2499

land.lhs.true62:                                  ; preds = %land.lhs.true60
  %cmp63 = icmp eq i32 %d3, 128, !dbg !2500
  br i1 %cmp63, label %land.lhs.true64, label %if.else91, !dbg !2501

land.lhs.true64:                                  ; preds = %land.lhs.true62
  %cmp65 = icmp eq i32 %nt, 6, !dbg !2502
  br i1 %cmp65, label %if.then66, label %if.else91, !dbg !2503

if.then66:                                        ; preds = %land.lhs.true64
  store i8 65, i8* %class_npb, align 1, !dbg !2504
  %real68 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp67, i32 0, i32 0, !dbg !2506
  store double 0x407F8AC6A8CB8B90, double* %real68, align 8, !dbg !2506
  %imag69 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp67, i32 0, i32 1, !dbg !2506
  store double 0x407FF67A05A82466, double* %imag69, align 8, !dbg !2506
  %arrayidx70 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 1, !dbg !2507
  %24 = bitcast %struct.dcomplex* %arrayidx70 to i8*, !dbg !2508
  %25 = bitcast %struct.dcomplex* %ref.tmp67 to i8*, !dbg !2508
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %24, i8* align 8 %25, i64 16, i1 false), !dbg !2508
  %real72 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp71, i32 0, i32 0, !dbg !2509
  store double 0x407F9F0F4941FB3E, double* %real72, align 8, !dbg !2509
  %imag73 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp71, i32 0, i32 1, !dbg !2509
  store double 0x407FDE18707A9D72, double* %imag73, align 8, !dbg !2509
  %arrayidx74 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 2, !dbg !2510
  %26 = bitcast %struct.dcomplex* %arrayidx74 to i8*, !dbg !2511
  %27 = bitcast %struct.dcomplex* %ref.tmp71 to i8*, !dbg !2511
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %26, i8* align 8 %27, i64 16, i1 false), !dbg !2511
  %real76 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp75, i32 0, i32 0, !dbg !2512
  store double 0x407FAF00C6D7110A, double* %real76, align 8, !dbg !2512
  %imag77 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp75, i32 0, i32 1, !dbg !2512
  store double 0x407FDD07CCB88353, double* %imag77, align 8, !dbg !2512
  %arrayidx78 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 3, !dbg !2513
  %28 = bitcast %struct.dcomplex* %arrayidx78 to i8*, !dbg !2514
  %29 = bitcast %struct.dcomplex* %ref.tmp75 to i8*, !dbg !2514
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %28, i8* align 8 %29, i64 16, i1 false), !dbg !2514
  %real80 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp79, i32 0, i32 0, !dbg !2515
  store double 0x407FBCA0EB3ECBEF, double* %real80, align 8, !dbg !2515
  %imag81 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp79, i32 0, i32 1, !dbg !2515
  store double 0x407FE2234776F4EF, double* %imag81, align 8, !dbg !2515
  %arrayidx82 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 4, !dbg !2516
  %30 = bitcast %struct.dcomplex* %arrayidx82 to i8*, !dbg !2517
  %31 = bitcast %struct.dcomplex* %ref.tmp79 to i8*, !dbg !2517
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %30, i8* align 8 %31, i64 16, i1 false), !dbg !2517
  %real84 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp83, i32 0, i32 0, !dbg !2518
  store double 0x407FC85F79D2C1E9, double* %real84, align 8, !dbg !2518
  %imag85 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp83, i32 0, i32 1, !dbg !2518
  store double 0x407FE7DD0AF2CEF4, double* %imag85, align 8, !dbg !2518
  %arrayidx86 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 5, !dbg !2519
  %32 = bitcast %struct.dcomplex* %arrayidx86 to i8*, !dbg !2520
  %33 = bitcast %struct.dcomplex* %ref.tmp83 to i8*, !dbg !2520
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %32, i8* align 8 %33, i64 16, i1 false), !dbg !2520
  %real88 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp87, i32 0, i32 0, !dbg !2521
  store double 0x407FD2611DBB8FA9, double* %real88, align 8, !dbg !2521
  %imag89 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp87, i32 0, i32 1, !dbg !2521
  store double 0x407FECAB25FE5602, double* %imag89, align 8, !dbg !2521
  %arrayidx90 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 6, !dbg !2522
  %34 = bitcast %struct.dcomplex* %arrayidx90 to i8*, !dbg !2523
  %35 = bitcast %struct.dcomplex* %ref.tmp87 to i8*, !dbg !2523
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %34, i8* align 8 %35, i64 16, i1 false), !dbg !2523
  br label %if.end490, !dbg !2524

if.else91:                                        ; preds = %land.lhs.true64, %land.lhs.true62, %land.lhs.true60, %if.else58
  %cmp92 = icmp eq i32 %d1, 512, !dbg !2525
  br i1 %cmp92, label %land.lhs.true93, label %if.else180, !dbg !2527

land.lhs.true93:                                  ; preds = %if.else91
  %cmp94 = icmp eq i32 %d2, 256, !dbg !2528
  br i1 %cmp94, label %land.lhs.true95, label %if.else180, !dbg !2529

land.lhs.true95:                                  ; preds = %land.lhs.true93
  %cmp96 = icmp eq i32 %d3, 256, !dbg !2530
  br i1 %cmp96, label %land.lhs.true97, label %if.else180, !dbg !2531

land.lhs.true97:                                  ; preds = %land.lhs.true95
  %cmp98 = icmp eq i32 %nt, 20, !dbg !2532
  br i1 %cmp98, label %if.then99, label %if.else180, !dbg !2533

if.then99:                                        ; preds = %land.lhs.true97
  store i8 66, i8* %class_npb, align 1, !dbg !2534
  %real101 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp100, i32 0, i32 0, !dbg !2536
  store double 0x40802E1D67491D27, double* %real101, align 8, !dbg !2536
  %imag102 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp100, i32 0, i32 1, !dbg !2536
  store double 0x407FBC7C4BF0AFB0, double* %imag102, align 8, !dbg !2536
  %arrayidx103 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 1, !dbg !2537
  %36 = bitcast %struct.dcomplex* %arrayidx103 to i8*, !dbg !2538
  %37 = bitcast %struct.dcomplex* %ref.tmp100 to i8*, !dbg !2538
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %36, i8* align 8 %37, i64 16, i1 false), !dbg !2538
  %real105 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp104, i32 0, i32 0, !dbg !2539
  store double 0x40801B9DF5E01838, double* %real105, align 8, !dbg !2539
  %imag106 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp104, i32 0, i32 1, !dbg !2539
  store double 0x407FCD32F7994D45, double* %imag106, align 8, !dbg !2539
  %arrayidx107 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 2, !dbg !2540
  %38 = bitcast %struct.dcomplex* %arrayidx107 to i8*, !dbg !2541
  %39 = bitcast %struct.dcomplex* %ref.tmp104 to i8*, !dbg !2541
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %38, i8* align 8 %39, i64 16, i1 false), !dbg !2541
  %real109 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp108, i32 0, i32 0, !dbg !2542
  store double 0x408015209C2AC008, double* %real109, align 8, !dbg !2542
  %imag110 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp108, i32 0, i32 1, !dbg !2542
  store double 0x407FD9EF2BAE169A, double* %imag110, align 8, !dbg !2542
  %arrayidx111 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 3, !dbg !2543
  %40 = bitcast %struct.dcomplex* %arrayidx111 to i8*, !dbg !2544
  %41 = bitcast %struct.dcomplex* %ref.tmp108 to i8*, !dbg !2544
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %40, i8* align 8 %41, i64 16, i1 false), !dbg !2544
  %real113 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp112, i32 0, i32 0, !dbg !2545
  store double 0x408011E72B556FFE, double* %real113, align 8, !dbg !2545
  %imag114 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp112, i32 0, i32 1, !dbg !2545
  store double 0x407FE1A32DF83794, double* %imag114, align 8, !dbg !2545
  %arrayidx115 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 4, !dbg !2546
  %42 = bitcast %struct.dcomplex* %arrayidx115 to i8*, !dbg !2547
  %43 = bitcast %struct.dcomplex* %ref.tmp112 to i8*, !dbg !2547
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %42, i8* align 8 %43, i64 16, i1 false), !dbg !2547
  %real117 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp116, i32 0, i32 0, !dbg !2548
  store double 0x40800FB38AA32FE6, double* %real117, align 8, !dbg !2548
  %imag118 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp116, i32 0, i32 1, !dbg !2548
  store double 0x407FE65CD1D86E4E, double* %imag118, align 8, !dbg !2548
  %arrayidx119 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 5, !dbg !2549
  %44 = bitcast %struct.dcomplex* %arrayidx119 to i8*, !dbg !2550
  %45 = bitcast %struct.dcomplex* %ref.tmp116 to i8*, !dbg !2550
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %44, i8* align 8 %45, i64 16, i1 false), !dbg !2550
  %real121 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp120, i32 0, i32 0, !dbg !2551
  store double 0x40800DF0531A9C48, double* %real121, align 8, !dbg !2551
  %imag122 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp120, i32 0, i32 1, !dbg !2551
  store double 0x407FE9844F14C8E1, double* %imag122, align 8, !dbg !2551
  %arrayidx123 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 6, !dbg !2552
  %46 = bitcast %struct.dcomplex* %arrayidx123 to i8*, !dbg !2553
  %47 = bitcast %struct.dcomplex* %ref.tmp120 to i8*, !dbg !2553
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %46, i8* align 8 %47, i64 16, i1 false), !dbg !2553
  %real125 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp124, i32 0, i32 0, !dbg !2554
  store double 0x40800C700989200D, double* %real125, align 8, !dbg !2554
  %imag126 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp124, i32 0, i32 1, !dbg !2554
  store double 0x407FEBD8BF0DD370, double* %imag126, align 8, !dbg !2554
  %arrayidx127 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 7, !dbg !2555
  %48 = bitcast %struct.dcomplex* %arrayidx127 to i8*, !dbg !2556
  %49 = bitcast %struct.dcomplex* %ref.tmp124 to i8*, !dbg !2556
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %48, i8* align 8 %49, i64 16, i1 false), !dbg !2556
  %real129 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp128, i32 0, i32 0, !dbg !2557
  store double 0x40800B20F5210ADA, double* %real129, align 8, !dbg !2557
  %imag130 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp128, i32 0, i32 1, !dbg !2557
  store double 0x407FEDB8F6EE292B, double* %imag130, align 8, !dbg !2557
  %arrayidx131 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 8, !dbg !2558
  %50 = bitcast %struct.dcomplex* %arrayidx131 to i8*, !dbg !2559
  %51 = bitcast %struct.dcomplex* %ref.tmp128 to i8*, !dbg !2559
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %50, i8* align 8 %51, i64 16, i1 false), !dbg !2559
  %real133 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp132, i32 0, i32 0, !dbg !2560
  store double 0x408009FA001E667B, double* %real133, align 8, !dbg !2560
  %imag134 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp132, i32 0, i32 1, !dbg !2560
  store double 0x407FEF52DA70C18D, double* %imag134, align 8, !dbg !2560
  %arrayidx135 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 9, !dbg !2561
  %52 = bitcast %struct.dcomplex* %arrayidx135 to i8*, !dbg !2562
  %53 = bitcast %struct.dcomplex* %ref.tmp132 to i8*, !dbg !2562
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %52, i8* align 8 %53, i64 16, i1 false), !dbg !2562
  %real137 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp136, i32 0, i32 0, !dbg !2563
  store double 0x408008F54B8BB893, double* %real137, align 8, !dbg !2563
  %imag138 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp136, i32 0, i32 1, !dbg !2563
  store double 0x407FF0BC8A6C6119, double* %imag138, align 8, !dbg !2563
  %arrayidx139 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 10, !dbg !2564
  %54 = bitcast %struct.dcomplex* %arrayidx139 to i8*, !dbg !2565
  %55 = bitcast %struct.dcomplex* %ref.tmp136 to i8*, !dbg !2565
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %54, i8* align 8 %55, i64 16, i1 false), !dbg !2565
  %real141 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp140, i32 0, i32 0, !dbg !2566
  store double 0x4080080E66C1709C, double* %real141, align 8, !dbg !2566
  %imag142 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp140, i32 0, i32 1, !dbg !2566
  store double 0x407FF200FF33D23F, double* %imag142, align 8, !dbg !2566
  %arrayidx143 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 11, !dbg !2567
  %56 = bitcast %struct.dcomplex* %arrayidx143 to i8*, !dbg !2568
  %57 = bitcast %struct.dcomplex* %ref.tmp140 to i8*, !dbg !2568
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %56, i8* align 8 %57, i64 16, i1 false), !dbg !2568
  %real145 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp144, i32 0, i32 0, !dbg !2569
  store double 0x40800741A55F37AD, double* %real145, align 8, !dbg !2569
  %imag146 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp144, i32 0, i32 1, !dbg !2569
  store double 0x407FF3261FE7F7AD, double* %imag146, align 8, !dbg !2569
  %arrayidx147 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 12, !dbg !2570
  %58 = bitcast %struct.dcomplex* %arrayidx147 to i8*, !dbg !2571
  %59 = bitcast %struct.dcomplex* %ref.tmp144 to i8*, !dbg !2571
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %58, i8* align 8 %59, i64 16, i1 false), !dbg !2571
  %real149 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp148, i32 0, i32 0, !dbg !2572
  store double 0x4080068BDAC33674, double* %real149, align 8, !dbg !2572
  %imag150 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp148, i32 0, i32 1, !dbg !2572
  store double 0x407FF42F9BEB8DC0, double* %imag150, align 8, !dbg !2572
  %arrayidx151 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 13, !dbg !2573
  %60 = bitcast %struct.dcomplex* %arrayidx151 to i8*, !dbg !2574
  %61 = bitcast %struct.dcomplex* %ref.tmp148 to i8*, !dbg !2574
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %60, i8* align 8 %61, i64 16, i1 false), !dbg !2574
  %real153 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp152, i32 0, i32 0, !dbg !2575
  store double 0x408005EA3C919C43, double* %real153, align 8, !dbg !2575
  %imag154 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp152, i32 0, i32 1, !dbg !2575
  store double 0x407FF5203263B154, double* %imag154, align 8, !dbg !2575
  %arrayidx155 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 14, !dbg !2576
  %62 = bitcast %struct.dcomplex* %arrayidx155 to i8*, !dbg !2577
  %63 = bitcast %struct.dcomplex* %ref.tmp152 to i8*, !dbg !2577
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %62, i8* align 8 %63, i64 16, i1 false), !dbg !2577
  %real157 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp156, i32 0, i32 0, !dbg !2578
  store double 0x4080055A545A3920, double* %real157, align 8, !dbg !2578
  %imag158 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp156, i32 0, i32 1, !dbg !2578
  store double 0x407FF5FA3C741F6E, double* %imag158, align 8, !dbg !2578
  %arrayidx159 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 15, !dbg !2579
  %64 = bitcast %struct.dcomplex* %arrayidx159 to i8*, !dbg !2580
  %65 = bitcast %struct.dcomplex* %ref.tmp156 to i8*, !dbg !2580
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %64, i8* align 8 %65, i64 16, i1 false), !dbg !2580
  %real161 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp160, i32 0, i32 0, !dbg !2581
  store double 0x408004D9F6B6B8E1, double* %real161, align 8, !dbg !2581
  %imag162 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp160, i32 0, i32 1, !dbg !2581
  store double 0x407FF6BFE1A61501, double* %imag162, align 8, !dbg !2581
  %arrayidx163 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 16, !dbg !2582
  %66 = bitcast %struct.dcomplex* %arrayidx163 to i8*, !dbg !2583
  %67 = bitcast %struct.dcomplex* %ref.tmp160 to i8*, !dbg !2583
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %66, i8* align 8 %67, i64 16, i1 false), !dbg !2583
  %real165 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp164, i32 0, i32 0, !dbg !2584
  store double 0x408004673C213244, double* %real165, align 8, !dbg !2584
  %imag166 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp164, i32 0, i32 1, !dbg !2584
  store double 0x407FF77327A3F7B0, double* %imag166, align 8, !dbg !2584
  %arrayidx167 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 17, !dbg !2585
  %68 = bitcast %struct.dcomplex* %arrayidx167 to i8*, !dbg !2586
  %69 = bitcast %struct.dcomplex* %ref.tmp164 to i8*, !dbg !2586
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %68, i8* align 8 %69, i64 16, i1 false), !dbg !2586
  %real169 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp168, i32 0, i32 0, !dbg !2587
  store double 0x408004007A3FD0EA, double* %real169, align 8, !dbg !2587
  %imag170 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp168, i32 0, i32 1, !dbg !2587
  store double 0x407FF815F3F1C1DE, double* %imag170, align 8, !dbg !2587
  %arrayidx171 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 18, !dbg !2588
  %70 = bitcast %struct.dcomplex* %arrayidx171 to i8*, !dbg !2589
  %71 = bitcast %struct.dcomplex* %ref.tmp168 to i8*, !dbg !2589
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %70, i8* align 8 %71, i64 16, i1 false), !dbg !2589
  %real173 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp172, i32 0, i32 0, !dbg !2590
  store double 0x408003A43D5F793B, double* %real173, align 8, !dbg !2590
  %imag174 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp172, i32 0, i32 1, !dbg !2590
  store double 0x407FF8AA099402A0, double* %imag174, align 8, !dbg !2590
  %arrayidx175 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 19, !dbg !2591
  %72 = bitcast %struct.dcomplex* %arrayidx175 to i8*, !dbg !2592
  %73 = bitcast %struct.dcomplex* %ref.tmp172 to i8*, !dbg !2592
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %72, i8* align 8 %73, i64 16, i1 false), !dbg !2592
  %real177 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp176, i32 0, i32 0, !dbg !2593
  store double 0x40800351422D2EDF, double* %real177, align 8, !dbg !2593
  %imag178 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp176, i32 0, i32 1, !dbg !2593
  store double 0x407FF93106A352EE, double* %imag178, align 8, !dbg !2593
  %arrayidx179 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 20, !dbg !2594
  %74 = bitcast %struct.dcomplex* %arrayidx179 to i8*, !dbg !2595
  %75 = bitcast %struct.dcomplex* %ref.tmp176 to i8*, !dbg !2595
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %74, i8* align 8 %75, i64 16, i1 false), !dbg !2595
  br label %if.end489, !dbg !2596

if.else180:                                       ; preds = %land.lhs.true97, %land.lhs.true95, %land.lhs.true93, %if.else91
  %cmp181 = icmp eq i32 %d1, 512, !dbg !2597
  br i1 %cmp181, label %land.lhs.true182, label %if.else269, !dbg !2599

land.lhs.true182:                                 ; preds = %if.else180
  %cmp183 = icmp eq i32 %d2, 512, !dbg !2600
  br i1 %cmp183, label %land.lhs.true184, label %if.else269, !dbg !2601

land.lhs.true184:                                 ; preds = %land.lhs.true182
  %cmp185 = icmp eq i32 %d3, 512, !dbg !2602
  br i1 %cmp185, label %land.lhs.true186, label %if.else269, !dbg !2603

land.lhs.true186:                                 ; preds = %land.lhs.true184
  %cmp187 = icmp eq i32 %nt, 20, !dbg !2604
  br i1 %cmp187, label %if.then188, label %if.else269, !dbg !2605

if.then188:                                       ; preds = %land.lhs.true186
  store i8 67, i8* %class_npb, align 1, !dbg !2606
  %real190 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp189, i32 0, i32 0, !dbg !2608
  store double 0x40803C101E899B03, double* %real190, align 8, !dbg !2608
  %imag191 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp189, i32 0, i32 1, !dbg !2608
  store double 0x408017373C01E593, double* %imag191, align 8, !dbg !2608
  %arrayidx192 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 1, !dbg !2609
  %76 = bitcast %struct.dcomplex* %arrayidx192 to i8*, !dbg !2610
  %77 = bitcast %struct.dcomplex* %ref.tmp189 to i8*, !dbg !2610
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %76, i8* align 8 %77, i64 16, i1 false), !dbg !2610
  %real194 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp193, i32 0, i32 0, !dbg !2611
  store double 0x40801C5675ED0B14, double* %real194, align 8, !dbg !2611
  %imag195 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp193, i32 0, i32 1, !dbg !2611
  store double 0x4080061004096FAD, double* %imag195, align 8, !dbg !2611
  %arrayidx196 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 2, !dbg !2612
  %78 = bitcast %struct.dcomplex* %arrayidx196 to i8*, !dbg !2613
  %79 = bitcast %struct.dcomplex* %ref.tmp193 to i8*, !dbg !2613
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %78, i8* align 8 %79, i64 16, i1 false), !dbg !2613
  %real198 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp197, i32 0, i32 0, !dbg !2614
  store double 0x408013BE0F176AC3, double* %real198, align 8, !dbg !2614
  %imag199 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp197, i32 0, i32 1, !dbg !2614
  store double 0x408001CD2DA9B691, double* %imag199, align 8, !dbg !2614
  %arrayidx200 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 3, !dbg !2615
  %80 = bitcast %struct.dcomplex* %arrayidx200 to i8*, !dbg !2616
  %81 = bitcast %struct.dcomplex* %ref.tmp197 to i8*, !dbg !2616
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %80, i8* align 8 %81, i64 16, i1 false), !dbg !2616
  %real202 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp201, i32 0, i32 0, !dbg !2617
  store double 0x4080101ED77ADAFA, double* %real202, align 8, !dbg !2617
  %imag203 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp201, i32 0, i32 1, !dbg !2617
  store double 0x408000DF4A8B7C66, double* %imag203, align 8, !dbg !2617
  %arrayidx204 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 4, !dbg !2618
  %82 = bitcast %struct.dcomplex* %arrayidx204 to i8*, !dbg !2619
  %83 = bitcast %struct.dcomplex* %ref.tmp201 to i8*, !dbg !2619
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %82, i8* align 8 %83, i64 16, i1 false), !dbg !2619
  %real206 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp205, i32 0, i32 0, !dbg !2620
  store double 0x40800E0A53D12FD5, double* %real206, align 8, !dbg !2620
  %imag207 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp205, i32 0, i32 1, !dbg !2620
  store double 0x408000EA3A1348C8, double* %imag207, align 8, !dbg !2620
  %arrayidx208 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 5, !dbg !2621
  %84 = bitcast %struct.dcomplex* %arrayidx208 to i8*, !dbg !2622
  %85 = bitcast %struct.dcomplex* %ref.tmp205 to i8*, !dbg !2622
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %84, i8* align 8 %85, i64 16, i1 false), !dbg !2622
  %real210 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp209, i32 0, i32 0, !dbg !2623
  store double 0x40800CA61ABB2192, double* %real210, align 8, !dbg !2623
  %imag211 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp209, i32 0, i32 1, !dbg !2623
  store double 0x408001328991F77F, double* %imag211, align 8, !dbg !2623
  %arrayidx212 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 6, !dbg !2624
  %86 = bitcast %struct.dcomplex* %arrayidx212 to i8*, !dbg !2625
  %87 = bitcast %struct.dcomplex* %ref.tmp209 to i8*, !dbg !2625
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %86, i8* align 8 %87, i64 16, i1 false), !dbg !2625
  %real214 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp213, i32 0, i32 0, !dbg !2626
  store double 0x40800BA7CD2DCE4D, double* %real214, align 8, !dbg !2626
  %imag215 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp213, i32 0, i32 1, !dbg !2626
  store double 0x4080017F2A30930B, double* %imag215, align 8, !dbg !2626
  %arrayidx216 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 7, !dbg !2627
  %88 = bitcast %struct.dcomplex* %arrayidx216 to i8*, !dbg !2628
  %89 = bitcast %struct.dcomplex* %ref.tmp213 to i8*, !dbg !2628
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %88, i8* align 8 %89, i64 16, i1 false), !dbg !2628
  %real218 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp217, i32 0, i32 0, !dbg !2629
  store double 0x40800AEBECB397D4, double* %real218, align 8, !dbg !2629
  %imag219 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp217, i32 0, i32 1, !dbg !2629
  store double 0x408001C12D7B83F2, double* %imag219, align 8, !dbg !2629
  %arrayidx220 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 8, !dbg !2630
  %90 = bitcast %struct.dcomplex* %arrayidx220 to i8*, !dbg !2631
  %91 = bitcast %struct.dcomplex* %ref.tmp217 to i8*, !dbg !2631
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %90, i8* align 8 %91, i64 16, i1 false), !dbg !2631
  %real222 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp221, i32 0, i32 0, !dbg !2632
  store double 0x40800A5D393668AE, double* %real222, align 8, !dbg !2632
  %imag223 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp221, i32 0, i32 1, !dbg !2632
  store double 0x408001F6BADA1C71, double* %imag223, align 8, !dbg !2632
  %arrayidx224 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 9, !dbg !2633
  %92 = bitcast %struct.dcomplex* %arrayidx224 to i8*, !dbg !2634
  %93 = bitcast %struct.dcomplex* %ref.tmp221 to i8*, !dbg !2634
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %92, i8* align 8 %93, i64 16, i1 false), !dbg !2634
  %real226 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp225, i32 0, i32 0, !dbg !2635
  store double 0x408009EDAA24021D, double* %real226, align 8, !dbg !2635
  %imag227 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp225, i32 0, i32 1, !dbg !2635
  store double 0x4080022183F3CA50, double* %imag227, align 8, !dbg !2635
  %arrayidx228 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 10, !dbg !2636
  %94 = bitcast %struct.dcomplex* %arrayidx228 to i8*, !dbg !2637
  %95 = bitcast %struct.dcomplex* %ref.tmp225 to i8*, !dbg !2637
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %94, i8* align 8 %95, i64 16, i1 false), !dbg !2637
  %real230 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp229, i32 0, i32 0, !dbg !2638
  store double 0x40800993B097C5AC, double* %real230, align 8, !dbg !2638
  %imag231 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp229, i32 0, i32 1, !dbg !2638
  store double 0x40800243C3A1DCB2, double* %imag231, align 8, !dbg !2638
  %arrayidx232 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 11, !dbg !2639
  %96 = bitcast %struct.dcomplex* %arrayidx232 to i8*, !dbg !2640
  %97 = bitcast %struct.dcomplex* %ref.tmp229 to i8*, !dbg !2640
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %96, i8* align 8 %97, i64 16, i1 false), !dbg !2640
  %real234 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp233, i32 0, i32 0, !dbg !2641
  store double 0x40800948BF026ADC, double* %real234, align 8, !dbg !2641
  %imag235 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp233, i32 0, i32 1, !dbg !2641
  store double 0x4080025F68FD8268, double* %imag235, align 8, !dbg !2641
  %arrayidx236 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 12, !dbg !2642
  %98 = bitcast %struct.dcomplex* %arrayidx236 to i8*, !dbg !2643
  %99 = bitcast %struct.dcomplex* %ref.tmp233 to i8*, !dbg !2643
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %98, i8* align 8 %99, i64 16, i1 false), !dbg !2643
  %real238 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp237, i32 0, i32 0, !dbg !2644
  store double 0x4080090857A518D9, double* %real238, align 8, !dbg !2644
  %imag239 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp237, i32 0, i32 1, !dbg !2644
  store double 0x40800275F32F50EA, double* %imag239, align 8, !dbg !2644
  %arrayidx240 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 13, !dbg !2645
  %100 = bitcast %struct.dcomplex* %arrayidx240 to i8*, !dbg !2646
  %101 = bitcast %struct.dcomplex* %ref.tmp237 to i8*, !dbg !2646
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %100, i8* align 8 %101, i64 16, i1 false), !dbg !2646
  %real242 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp241, i32 0, i32 0, !dbg !2647
  store double 0x408008CF67B5F6E6, double* %real242, align 8, !dbg !2647
  %imag243 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp241, i32 0, i32 1, !dbg !2647
  store double 0x408002887F1716B0, double* %imag243, align 8, !dbg !2647
  %arrayidx244 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 14, !dbg !2648
  %102 = bitcast %struct.dcomplex* %arrayidx244 to i8*, !dbg !2649
  %103 = bitcast %struct.dcomplex* %ref.tmp241 to i8*, !dbg !2649
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %102, i8* align 8 %103, i64 16, i1 false), !dbg !2649
  %real246 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp245, i32 0, i32 0, !dbg !2650
  store double 0x4080089BD580EA3A, double* %real246, align 8, !dbg !2650
  %imag247 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp245, i32 0, i32 1, !dbg !2650
  store double 0x40800297DE24048E, double* %imag247, align 8, !dbg !2650
  %arrayidx248 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 15, !dbg !2651
  %104 = bitcast %struct.dcomplex* %arrayidx248 to i8*, !dbg !2652
  %105 = bitcast %struct.dcomplex* %ref.tmp245 to i8*, !dbg !2652
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %104, i8* align 8 %105, i64 16, i1 false), !dbg !2652
  %real250 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp249, i32 0, i32 0, !dbg !2653
  store double 0x4080086C31EBD984, double* %real250, align 8, !dbg !2653
  %imag251 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp249, i32 0, i32 1, !dbg !2653
  store double 0x408002A4AAB9F9F8, double* %imag251, align 8, !dbg !2653
  %arrayidx252 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 16, !dbg !2654
  %106 = bitcast %struct.dcomplex* %arrayidx252 to i8*, !dbg !2655
  %107 = bitcast %struct.dcomplex* %ref.tmp249 to i8*, !dbg !2655
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %106, i8* align 8 %107, i64 16, i1 false), !dbg !2655
  %real254 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp253, i32 0, i32 0, !dbg !2656
  store double 0x4080083F8294129E, double* %real254, align 8, !dbg !2656
  %imag255 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp253, i32 0, i32 1, !dbg !2656
  store double 0x408002AF57DC0D71, double* %imag255, align 8, !dbg !2656
  %arrayidx256 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 17, !dbg !2657
  %108 = bitcast %struct.dcomplex* %arrayidx256 to i8*, !dbg !2658
  %109 = bitcast %struct.dcomplex* %ref.tmp253 to i8*, !dbg !2658
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %108, i8* align 8 %109, i64 16, i1 false), !dbg !2658
  %real258 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp257, i32 0, i32 0, !dbg !2659
  store double 0x408008151CE457D2, double* %real258, align 8, !dbg !2659
  %imag259 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp257, i32 0, i32 1, !dbg !2659
  store double 0x408002B83C8A44C9, double* %imag259, align 8, !dbg !2659
  %arrayidx260 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 18, !dbg !2660
  %110 = bitcast %struct.dcomplex* %arrayidx260 to i8*, !dbg !2661
  %111 = bitcast %struct.dcomplex* %ref.tmp257 to i8*, !dbg !2661
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %110, i8* align 8 %111, i64 16, i1 false), !dbg !2661
  %real262 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp261, i32 0, i32 0, !dbg !2662
  store double 0x408007EC8CCD48ED, double* %real262, align 8, !dbg !2662
  %imag263 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp261, i32 0, i32 1, !dbg !2662
  store double 0x408002BF9BCECA75, double* %imag263, align 8, !dbg !2662
  %arrayidx264 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 19, !dbg !2663
  %112 = bitcast %struct.dcomplex* %arrayidx264 to i8*, !dbg !2664
  %113 = bitcast %struct.dcomplex* %ref.tmp261 to i8*, !dbg !2664
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %112, i8* align 8 %113, i64 16, i1 false), !dbg !2664
  %real266 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp265, i32 0, i32 0, !dbg !2665
  store double 0x408007C58371022F, double* %real266, align 8, !dbg !2665
  %imag267 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp265, i32 0, i32 1, !dbg !2665
  store double 0x408002C5AA6407B6, double* %imag267, align 8, !dbg !2665
  %arrayidx268 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 20, !dbg !2666
  %114 = bitcast %struct.dcomplex* %arrayidx268 to i8*, !dbg !2667
  %115 = bitcast %struct.dcomplex* %ref.tmp265 to i8*, !dbg !2667
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %114, i8* align 8 %115, i64 16, i1 false), !dbg !2667
  br label %if.end488, !dbg !2668

if.else269:                                       ; preds = %land.lhs.true186, %land.lhs.true184, %land.lhs.true182, %if.else180
  %cmp270 = icmp eq i32 %d1, 2048, !dbg !2669
  br i1 %cmp270, label %land.lhs.true271, label %if.else378, !dbg !2671

land.lhs.true271:                                 ; preds = %if.else269
  %cmp272 = icmp eq i32 %d2, 1024, !dbg !2672
  br i1 %cmp272, label %land.lhs.true273, label %if.else378, !dbg !2673

land.lhs.true273:                                 ; preds = %land.lhs.true271
  %cmp274 = icmp eq i32 %d3, 1024, !dbg !2674
  br i1 %cmp274, label %land.lhs.true275, label %if.else378, !dbg !2675

land.lhs.true275:                                 ; preds = %land.lhs.true273
  %cmp276 = icmp eq i32 %nt, 25, !dbg !2676
  br i1 %cmp276, label %if.then277, label %if.else378, !dbg !2677

if.then277:                                       ; preds = %land.lhs.true275
  store i8 68, i8* %class_npb, align 1, !dbg !2678
  %real279 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp278, i32 0, i32 0, !dbg !2680
  store double 0x408001C8B7A5243B, double* %real279, align 8, !dbg !2680
  %imag280 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp278, i32 0, i32 1, !dbg !2680
  store double 0x407FFDA78AA6499C, double* %imag280, align 8, !dbg !2680
  %arrayidx281 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 1, !dbg !2681
  %116 = bitcast %struct.dcomplex* %arrayidx281 to i8*, !dbg !2682
  %117 = bitcast %struct.dcomplex* %ref.tmp278 to i8*, !dbg !2682
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %116, i8* align 8 %117, i64 16, i1 false), !dbg !2682
  %real283 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp282, i32 0, i32 0, !dbg !2683
  store double 0x4080005F05B14D73, double* %real283, align 8, !dbg !2683
  %imag284 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp282, i32 0, i32 1, !dbg !2683
  store double 0x407FFB4C42805D51, double* %imag284, align 8, !dbg !2683
  %arrayidx285 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 2, !dbg !2684
  %118 = bitcast %struct.dcomplex* %arrayidx285 to i8*, !dbg !2685
  %119 = bitcast %struct.dcomplex* %ref.tmp282 to i8*, !dbg !2685
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %118, i8* align 8 %119, i64 16, i1 false), !dbg !2685
  %real287 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp286, i32 0, i32 0, !dbg !2686
  store double 0x407FFFC9049FE6AA, double* %real287, align 8, !dbg !2686
  %imag288 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp286, i32 0, i32 1, !dbg !2686
  store double 0x407FFB5AABC2C2DC, double* %imag288, align 8, !dbg !2686
  %arrayidx289 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 3, !dbg !2687
  %120 = bitcast %struct.dcomplex* %arrayidx289 to i8*, !dbg !2688
  %121 = bitcast %struct.dcomplex* %ref.tmp286 to i8*, !dbg !2688
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %120, i8* align 8 %121, i64 16, i1 false), !dbg !2688
  %real291 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp290, i32 0, i32 0, !dbg !2689
  store double 0x407FFF3AE6781D07, double* %real291, align 8, !dbg !2689
  %imag292 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp290, i32 0, i32 1, !dbg !2689
  store double 0x407FFBCC55AD30A5, double* %imag292, align 8, !dbg !2689
  %arrayidx293 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 4, !dbg !2690
  %122 = bitcast %struct.dcomplex* %arrayidx293 to i8*, !dbg !2691
  %123 = bitcast %struct.dcomplex* %ref.tmp290 to i8*, !dbg !2691
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %122, i8* align 8 %123, i64 16, i1 false), !dbg !2691
  %real295 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp294, i32 0, i32 0, !dbg !2692
  store double 0x407FFED49E586270, double* %real295, align 8, !dbg !2692
  %imag296 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp294, i32 0, i32 1, !dbg !2692
  store double 0x407FFC49DED1E229, double* %imag296, align 8, !dbg !2692
  %arrayidx297 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 5, !dbg !2693
  %124 = bitcast %struct.dcomplex* %arrayidx297 to i8*, !dbg !2694
  %125 = bitcast %struct.dcomplex* %ref.tmp294 to i8*, !dbg !2694
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %124, i8* align 8 %125, i64 16, i1 false), !dbg !2694
  %real299 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp298, i32 0, i32 0, !dbg !2695
  store double 0x407FFE88286F1600, double* %real299, align 8, !dbg !2695
  %imag300 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp298, i32 0, i32 1, !dbg !2695
  store double 0x407FFCBFA44E2DA9, double* %imag300, align 8, !dbg !2695
  %arrayidx301 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 6, !dbg !2696
  %126 = bitcast %struct.dcomplex* %arrayidx301 to i8*, !dbg !2697
  %127 = bitcast %struct.dcomplex* %ref.tmp298 to i8*, !dbg !2697
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %126, i8* align 8 %127, i64 16, i1 false), !dbg !2697
  %real303 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp302, i32 0, i32 0, !dbg !2698
  store double 0x407FFE4F62F012B7, double* %real303, align 8, !dbg !2698
  %imag304 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp302, i32 0, i32 1, !dbg !2698
  store double 0x407FFD2913502BF7, double* %imag304, align 8, !dbg !2698
  %arrayidx305 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 7, !dbg !2699
  %128 = bitcast %struct.dcomplex* %arrayidx305 to i8*, !dbg !2700
  %129 = bitcast %struct.dcomplex* %ref.tmp302 to i8*, !dbg !2700
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %128, i8* align 8 %129, i64 16, i1 false), !dbg !2700
  %real307 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp306, i32 0, i32 0, !dbg !2701
  store double 0x407FFE25D7467D87, double* %real307, align 8, !dbg !2701
  %imag308 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp306, i32 0, i32 1, !dbg !2701
  store double 0x407FFD85C991CC1E, double* %imag308, align 8, !dbg !2701
  %arrayidx309 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 8, !dbg !2702
  %130 = bitcast %struct.dcomplex* %arrayidx309 to i8*, !dbg !2703
  %131 = bitcast %struct.dcomplex* %ref.tmp306 to i8*, !dbg !2703
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %130, i8* align 8 %131, i64 16, i1 false), !dbg !2703
  %real311 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp310, i32 0, i32 0, !dbg !2704
  store double 0x407FFE07F5F9461B, double* %real311, align 8, !dbg !2704
  %imag312 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp310, i32 0, i32 1, !dbg !2704
  store double 0x407FFDD6ADE6AA2F, double* %imag312, align 8, !dbg !2704
  %arrayidx313 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 9, !dbg !2705
  %132 = bitcast %struct.dcomplex* %arrayidx313 to i8*, !dbg !2706
  %133 = bitcast %struct.dcomplex* %ref.tmp310 to i8*, !dbg !2706
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %132, i8* align 8 %133, i64 16, i1 false), !dbg !2706
  %real315 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp314, i32 0, i32 0, !dbg !2707
  store double 0x407FFDF2F9E3CE75, double* %real315, align 8, !dbg !2707
  %imag316 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp314, i32 0, i32 1, !dbg !2707
  store double 0x407FFE1D0052370F, double* %imag316, align 8, !dbg !2707
  %arrayidx317 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 10, !dbg !2708
  %134 = bitcast %struct.dcomplex* %arrayidx317 to i8*, !dbg !2709
  %135 = bitcast %struct.dcomplex* %ref.tmp314 to i8*, !dbg !2709
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %134, i8* align 8 %135, i64 16, i1 false), !dbg !2709
  %real319 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp318, i32 0, i32 0, !dbg !2710
  store double 0x407FFDE4CA360F49, double* %real319, align 8, !dbg !2710
  %imag320 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp318, i32 0, i32 1, !dbg !2710
  store double 0x407FFE5A05B5973E, double* %imag320, align 8, !dbg !2710
  %arrayidx321 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 11, !dbg !2711
  %136 = bitcast %struct.dcomplex* %arrayidx321 to i8*, !dbg !2712
  %137 = bitcast %struct.dcomplex* %ref.tmp318 to i8*, !dbg !2712
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %136, i8* align 8 %137, i64 16, i1 false), !dbg !2712
  %real323 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp322, i32 0, i32 0, !dbg !2713
  store double 0x407FFDDBD5F99711, double* %real323, align 8, !dbg !2713
  %imag324 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp322, i32 0, i32 1, !dbg !2713
  store double 0x407FFE8EEACAA874, double* %imag324, align 8, !dbg !2713
  %arrayidx325 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 12, !dbg !2714
  %138 = bitcast %struct.dcomplex* %arrayidx325 to i8*, !dbg !2715
  %139 = bitcast %struct.dcomplex* %ref.tmp322 to i8*, !dbg !2715
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %138, i8* align 8 %139, i64 16, i1 false), !dbg !2715
  %real327 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp326, i32 0, i32 0, !dbg !2716
  store double 0x407FFDD6F2033D21, double* %real327, align 8, !dbg !2716
  %imag328 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp326, i32 0, i32 1, !dbg !2716
  store double 0x407FFEBCBBFA2EBF, double* %imag328, align 8, !dbg !2716
  %arrayidx329 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 13, !dbg !2717
  %140 = bitcast %struct.dcomplex* %arrayidx329 to i8*, !dbg !2718
  %141 = bitcast %struct.dcomplex* %ref.tmp326 to i8*, !dbg !2718
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %140, i8* align 8 %141, i64 16, i1 false), !dbg !2718
  %real331 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp330, i32 0, i32 0, !dbg !2719
  store double 0x407FFDD53D74DC74, double* %real331, align 8, !dbg !2719
  %imag332 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp330, i32 0, i32 1, !dbg !2719
  store double 0x407FFEE46511649D, double* %imag332, align 8, !dbg !2719
  %arrayidx333 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 14, !dbg !2720
  %142 = bitcast %struct.dcomplex* %arrayidx333 to i8*, !dbg !2721
  %143 = bitcast %struct.dcomplex* %ref.tmp330 to i8*, !dbg !2721
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %142, i8* align 8 %143, i64 16, i1 false), !dbg !2721
  %real335 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp334, i32 0, i32 0, !dbg !2722
  store double 0x407FFDD60D2DB5D2, double* %real335, align 8, !dbg !2722
  %imag336 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp334, i32 0, i32 1, !dbg !2722
  store double 0x407FFF06B3C01AEA, double* %imag336, align 8, !dbg !2722
  %arrayidx337 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 15, !dbg !2723
  %144 = bitcast %struct.dcomplex* %arrayidx337 to i8*, !dbg !2724
  %145 = bitcast %struct.dcomplex* %ref.tmp334 to i8*, !dbg !2724
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %144, i8* align 8 %145, i64 16, i1 false), !dbg !2724
  %real339 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp338, i32 0, i32 0, !dbg !2725
  store double 0x407FFDD8DD056A7D, double* %real339, align 8, !dbg !2725
  %imag340 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp338, i32 0, i32 1, !dbg !2725
  store double 0x407FFF245ADF0BCE, double* %imag340, align 8, !dbg !2725
  %arrayidx341 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 16, !dbg !2726
  %146 = bitcast %struct.dcomplex* %arrayidx341 to i8*, !dbg !2727
  %147 = bitcast %struct.dcomplex* %ref.tmp338 to i8*, !dbg !2727
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %146, i8* align 8 %147, i64 16, i1 false), !dbg !2727
  %real343 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp342, i32 0, i32 0, !dbg !2728
  store double 0x407FFDDD45618FE6, double* %real343, align 8, !dbg !2728
  %imag344 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp342, i32 0, i32 1, !dbg !2728
  store double 0x407FFF3DF5BAB029, double* %imag344, align 8, !dbg !2728
  %arrayidx345 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 17, !dbg !2729
  %148 = bitcast %struct.dcomplex* %arrayidx345 to i8*, !dbg !2730
  %149 = bitcast %struct.dcomplex* %ref.tmp342 to i8*, !dbg !2730
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %148, i8* align 8 %149, i64 16, i1 false), !dbg !2730
  %real347 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp346, i32 0, i32 0, !dbg !2731
  store double 0x407FFDE2F3E650B3, double* %real347, align 8, !dbg !2731
  %imag348 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp346, i32 0, i32 1, !dbg !2731
  store double 0x407FFF540B1CF5A1, double* %imag348, align 8, !dbg !2731
  %arrayidx349 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 18, !dbg !2732
  %150 = bitcast %struct.dcomplex* %arrayidx349 to i8*, !dbg !2733
  %151 = bitcast %struct.dcomplex* %ref.tmp346 to i8*, !dbg !2733
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %150, i8* align 8 %151, i64 16, i1 false), !dbg !2733
  %real351 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp350, i32 0, i32 0, !dbg !2734
  store double 0x407FFDE9A64E1245, double* %real351, align 8, !dbg !2734
  %imag352 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp350, i32 0, i32 1, !dbg !2734
  store double 0x407FFF671002DAE5, double* %imag352, align 8, !dbg !2734
  %arrayidx353 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 19, !dbg !2735
  %152 = bitcast %struct.dcomplex* %arrayidx353 to i8*, !dbg !2736
  %153 = bitcast %struct.dcomplex* %ref.tmp350 to i8*, !dbg !2736
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %152, i8* align 8 %153, i64 16, i1 false), !dbg !2736
  %real355 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp354, i32 0, i32 0, !dbg !2737
  store double 0x407FFDF126BADF21, double* %real355, align 8, !dbg !2737
  %imag356 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp354, i32 0, i32 1, !dbg !2737
  store double 0x407FFF7769FD4D32, double* %imag356, align 8, !dbg !2737
  %arrayidx357 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 20, !dbg !2738
  %154 = bitcast %struct.dcomplex* %arrayidx357 to i8*, !dbg !2739
  %155 = bitcast %struct.dcomplex* %ref.tmp354 to i8*, !dbg !2739
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %154, i8* align 8 %155, i64 16, i1 false), !dbg !2739
  %real359 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp358, i32 0, i32 0, !dbg !2740
  store double 0x407FFDF94909BB13, double* %real359, align 8, !dbg !2740
  %imag360 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp358, i32 0, i32 1, !dbg !2740
  store double 0x407FFF85714411B2, double* %imag360, align 8, !dbg !2740
  %arrayidx361 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 21, !dbg !2741
  %156 = bitcast %struct.dcomplex* %arrayidx361 to i8*, !dbg !2742
  %157 = bitcast %struct.dcomplex* %ref.tmp358 to i8*, !dbg !2742
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %156, i8* align 8 %157, i64 16, i1 false), !dbg !2742
  %real363 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp362, i32 0, i32 0, !dbg !2743
  store double 0x407FFE01E8D7E962, double* %real363, align 8, !dbg !2743
  %imag364 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp362, i32 0, i32 1, !dbg !2743
  store double 0x407FFF9172826820, double* %imag364, align 8, !dbg !2743
  %arrayidx365 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 22, !dbg !2744
  %158 = bitcast %struct.dcomplex* %arrayidx365 to i8*, !dbg !2745
  %159 = bitcast %struct.dcomplex* %ref.tmp362 to i8*, !dbg !2745
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %158, i8* align 8 %159, i64 16, i1 false), !dbg !2745
  %real367 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp366, i32 0, i32 0, !dbg !2746
  store double 0x407FFE0AE8040E41, double* %real367, align 8, !dbg !2746
  %imag368 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp366, i32 0, i32 1, !dbg !2746
  store double 0x407FFF9BB06626E0, double* %imag368, align 8, !dbg !2746
  %arrayidx369 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 23, !dbg !2747
  %160 = bitcast %struct.dcomplex* %arrayidx369 to i8*, !dbg !2748
  %161 = bitcast %struct.dcomplex* %ref.tmp366 to i8*, !dbg !2748
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %160, i8* align 8 %161, i64 16, i1 false), !dbg !2748
  %real371 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp370, i32 0, i32 0, !dbg !2749
  store double 0x407FFE142D872C17, double* %real371, align 8, !dbg !2749
  %imag372 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp370, i32 0, i32 1, !dbg !2749
  store double 0x407FFFA464F89DCE, double* %imag372, align 8, !dbg !2749
  %arrayidx373 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 24, !dbg !2750
  %162 = bitcast %struct.dcomplex* %arrayidx373 to i8*, !dbg !2751
  %163 = bitcast %struct.dcomplex* %ref.tmp370 to i8*, !dbg !2751
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %162, i8* align 8 %163, i64 16, i1 false), !dbg !2751
  %real375 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp374, i32 0, i32 0, !dbg !2752
  store double 0x407FFE1DA48D386E, double* %real375, align 8, !dbg !2752
  %imag376 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp374, i32 0, i32 1, !dbg !2752
  store double 0x407FFFABC2C855DE, double* %imag376, align 8, !dbg !2752
  %arrayidx377 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 25, !dbg !2753
  %164 = bitcast %struct.dcomplex* %arrayidx377 to i8*, !dbg !2754
  %165 = bitcast %struct.dcomplex* %ref.tmp374 to i8*, !dbg !2754
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %164, i8* align 8 %165, i64 16, i1 false), !dbg !2754
  br label %if.end487, !dbg !2755

if.else378:                                       ; preds = %land.lhs.true275, %land.lhs.true273, %land.lhs.true271, %if.else269
  %cmp379 = icmp eq i32 %d1, 4096, !dbg !2756
  br i1 %cmp379, label %land.lhs.true380, label %if.end, !dbg !2758

land.lhs.true380:                                 ; preds = %if.else378
  %cmp381 = icmp eq i32 %d2, 2048, !dbg !2759
  br i1 %cmp381, label %land.lhs.true382, label %if.end, !dbg !2760

land.lhs.true382:                                 ; preds = %land.lhs.true380
  %cmp383 = icmp eq i32 %d3, 2048, !dbg !2761
  br i1 %cmp383, label %land.lhs.true384, label %if.end, !dbg !2762

land.lhs.true384:                                 ; preds = %land.lhs.true382
  %cmp385 = icmp eq i32 %nt, 25, !dbg !2763
  br i1 %cmp385, label %if.then386, label %if.end, !dbg !2764

if.then386:                                       ; preds = %land.lhs.true384
  store i8 69, i8* %class_npb, align 1, !dbg !2765
  %real388 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp387, i32 0, i32 0, !dbg !2767
  store double 0x40800147E4E2E063, double* %real388, align 8, !dbg !2767
  %imag389 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp387, i32 0, i32 1, !dbg !2767
  store double 0x407FFBD566A0B5FD, double* %imag389, align 8, !dbg !2767
  %arrayidx390 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 1, !dbg !2768
  %166 = bitcast %struct.dcomplex* %arrayidx390 to i8*, !dbg !2769
  %167 = bitcast %struct.dcomplex* %ref.tmp387 to i8*, !dbg !2769
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %166, i8* align 8 %167, i64 16, i1 false), !dbg !2769
  %real392 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp391, i32 0, i32 0, !dbg !2770
  store double 0x408000B96D3A755A, double* %real392, align 8, !dbg !2770
  %imag393 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp391, i32 0, i32 1, !dbg !2770
  store double 0x407FFDC89676A99F, double* %imag393, align 8, !dbg !2770
  %arrayidx394 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 2, !dbg !2771
  %168 = bitcast %struct.dcomplex* %arrayidx394 to i8*, !dbg !2772
  %169 = bitcast %struct.dcomplex* %ref.tmp391 to i8*, !dbg !2772
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %168, i8* align 8 %169, i64 16, i1 false), !dbg !2772
  %real396 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp395, i32 0, i32 0, !dbg !2773
  store double 0x4080007FA32A25BE, double* %real396, align 8, !dbg !2773
  %imag397 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp395, i32 0, i32 1, !dbg !2773
  store double 0x407FFE84CB3A10F8, double* %imag397, align 8, !dbg !2773
  %arrayidx398 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 3, !dbg !2774
  %170 = bitcast %struct.dcomplex* %arrayidx398 to i8*, !dbg !2775
  %171 = bitcast %struct.dcomplex* %ref.tmp395 to i8*, !dbg !2775
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %170, i8* align 8 %171, i64 16, i1 false), !dbg !2775
  %real400 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp399, i32 0, i32 0, !dbg !2776
  store double 0x40800059C9C82B40, double* %real400, align 8, !dbg !2776
  %imag401 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp399, i32 0, i32 1, !dbg !2776
  store double 0x407FFEF414B87FD6, double* %imag401, align 8, !dbg !2776
  %arrayidx402 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 4, !dbg !2777
  %172 = bitcast %struct.dcomplex* %arrayidx402 to i8*, !dbg !2778
  %173 = bitcast %struct.dcomplex* %ref.tmp399 to i8*, !dbg !2778
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %172, i8* align 8 %173, i64 16, i1 false), !dbg !2778
  %real404 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp403, i32 0, i32 0, !dbg !2779
  store double 0x4080003FCCB7C9C8, double* %real404, align 8, !dbg !2779
  %imag405 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp403, i32 0, i32 1, !dbg !2779
  store double 0x407FFF483912F11E, double* %imag405, align 8, !dbg !2779
  %arrayidx406 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 5, !dbg !2780
  %174 = bitcast %struct.dcomplex* %arrayidx406 to i8*, !dbg !2781
  %175 = bitcast %struct.dcomplex* %ref.tmp403 to i8*, !dbg !2781
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %174, i8* align 8 %175, i64 16, i1 false), !dbg !2781
  %real408 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp407, i32 0, i32 0, !dbg !2782
  store double 0x4080002E4D90A084, double* %real408, align 8, !dbg !2782
  %imag409 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp407, i32 0, i32 1, !dbg !2782
  store double 0x407FFF8D62BCE558, double* %imag409, align 8, !dbg !2782
  %arrayidx410 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 6, !dbg !2783
  %176 = bitcast %struct.dcomplex* %arrayidx410 to i8*, !dbg !2784
  %177 = bitcast %struct.dcomplex* %ref.tmp407 to i8*, !dbg !2784
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %176, i8* align 8 %177, i64 16, i1 false), !dbg !2784
  %real412 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp411, i32 0, i32 0, !dbg !2785
  store double 0x40800022AC039D7C, double* %real412, align 8, !dbg !2785
  %imag413 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp411, i32 0, i32 1, !dbg !2785
  store double 0x407FFFC737C3F7CD, double* %imag413, align 8, !dbg !2785
  %arrayidx414 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 7, !dbg !2786
  %178 = bitcast %struct.dcomplex* %arrayidx414 to i8*, !dbg !2787
  %179 = bitcast %struct.dcomplex* %ref.tmp411 to i8*, !dbg !2787
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %178, i8* align 8 %179, i64 16, i1 false), !dbg !2787
  %real416 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp415, i32 0, i32 0, !dbg !2788
  store double 0x4080001ADFFA71B9, double* %real416, align 8, !dbg !2788
  %imag417 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp415, i32 0, i32 1, !dbg !2788
  store double 0x407FFFF78C336255, double* %imag417, align 8, !dbg !2788
  %arrayidx418 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 8, !dbg !2789
  %180 = bitcast %struct.dcomplex* %arrayidx418 to i8*, !dbg !2790
  %181 = bitcast %struct.dcomplex* %ref.tmp415 to i8*, !dbg !2790
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %180, i8* align 8 %181, i64 16, i1 false), !dbg !2790
  %real420 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp419, i32 0, i32 0, !dbg !2791
  store double 0x4080001574D0520C, double* %real420, align 8, !dbg !2791
  %imag421 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp419, i32 0, i32 1, !dbg !2791
  store double 0x4080000FE85C03E9, double* %imag421, align 8, !dbg !2791
  %arrayidx422 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 9, !dbg !2792
  %182 = bitcast %struct.dcomplex* %arrayidx422 to i8*, !dbg !2793
  %183 = bitcast %struct.dcomplex* %ref.tmp419 to i8*, !dbg !2793
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %182, i8* align 8 %183, i64 16, i1 false), !dbg !2793
  %real424 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp423, i32 0, i32 0, !dbg !2794
  store double 0x408000116F284244, double* %real424, align 8, !dbg !2794
  %imag425 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp423, i32 0, i32 1, !dbg !2794
  store double 0x40800020A7695837, double* %imag425, align 8, !dbg !2794
  %arrayidx426 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 10, !dbg !2795
  %184 = bitcast %struct.dcomplex* %arrayidx426 to i8*, !dbg !2796
  %185 = bitcast %struct.dcomplex* %ref.tmp423 to i8*, !dbg !2796
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %184, i8* align 8 %185, i64 16, i1 false), !dbg !2796
  %real428 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp427, i32 0, i32 0, !dbg !2797
  store double 0x4080000E2D56813F, double* %real428, align 8, !dbg !2797
  %imag429 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp427, i32 0, i32 1, !dbg !2797
  store double 0x4080002E951F7B34, double* %imag429, align 8, !dbg !2797
  %arrayidx430 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 11, !dbg !2798
  %186 = bitcast %struct.dcomplex* %arrayidx430 to i8*, !dbg !2799
  %187 = bitcast %struct.dcomplex* %ref.tmp427 to i8*, !dbg !2799
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %186, i8* align 8 %187, i64 16, i1 false), !dbg !2799
  %real432 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp431, i32 0, i32 0, !dbg !2800
  store double 0x4080000B4BE05864, double* %real432, align 8, !dbg !2800
  %imag433 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp431, i32 0, i32 1, !dbg !2800
  store double 0x4080003A2ED08404, double* %imag433, align 8, !dbg !2800
  %arrayidx434 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 12, !dbg !2801
  %188 = bitcast %struct.dcomplex* %arrayidx434 to i8*, !dbg !2802
  %189 = bitcast %struct.dcomplex* %ref.tmp431 to i8*, !dbg !2802
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %188, i8* align 8 %189, i64 16, i1 false), !dbg !2802
  %real436 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp435, i32 0, i32 0, !dbg !2803
  store double 0x408000089094AC2D, double* %real436, align 8, !dbg !2803
  %imag437 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp435, i32 0, i32 1, !dbg !2803
  store double 0x40800043DD87C2F3, double* %imag437, align 8, !dbg !2803
  %arrayidx438 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 13, !dbg !2804
  %190 = bitcast %struct.dcomplex* %arrayidx438 to i8*, !dbg !2805
  %191 = bitcast %struct.dcomplex* %ref.tmp435 to i8*, !dbg !2805
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %190, i8* align 8 %191, i64 16, i1 false), !dbg !2805
  %real440 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp439, i32 0, i32 0, !dbg !2806
  store double 0x40800005DBBF34DD, double* %real440, align 8, !dbg !2806
  %imag441 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp439, i32 0, i32 1, !dbg !2806
  store double 0x4080004BF7DEAC1A, double* %imag441, align 8, !dbg !2806
  %arrayidx442 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 14, !dbg !2807
  %192 = bitcast %struct.dcomplex* %arrayidx442 to i8*, !dbg !2808
  %193 = bitcast %struct.dcomplex* %ref.tmp439 to i8*, !dbg !2808
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %192, i8* align 8 %193, i64 16, i1 false), !dbg !2808
  %real444 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp443, i32 0, i32 0, !dbg !2809
  store double 0x408000031E1FCB83, double* %real444, align 8, !dbg !2809
  %imag445 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp443, i32 0, i32 1, !dbg !2809
  store double 0x40800052C48391C0, double* %imag445, align 8, !dbg !2809
  %arrayidx446 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 15, !dbg !2810
  %194 = bitcast %struct.dcomplex* %arrayidx446 to i8*, !dbg !2811
  %195 = bitcast %struct.dcomplex* %ref.tmp443 to i8*, !dbg !2811
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %194, i8* align 8 %195, i64 16, i1 false), !dbg !2811
  %real448 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp447, i32 0, i32 0, !dbg !2812
  store double 0x4080000052507A84, double* %real448, align 8, !dbg !2812
  %imag449 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp447, i32 0, i32 1, !dbg !2812
  store double 0x408000587CD9C3A1, double* %imag449, align 8, !dbg !2812
  %arrayidx450 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 16, !dbg !2813
  %196 = bitcast %struct.dcomplex* %arrayidx450 to i8*, !dbg !2814
  %197 = bitcast %struct.dcomplex* %ref.tmp447 to i8*, !dbg !2814
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %196, i8* align 8 %197, i64 16, i1 false), !dbg !2814
  %real452 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp451, i32 0, i32 0, !dbg !2815
  store double 0x407FFFFAF1111C29, double* %real452, align 8, !dbg !2815
  %imag453 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp451, i32 0, i32 1, !dbg !2815
  store double 0x4080005D4F648E97, double* %imag453, align 8, !dbg !2815
  %arrayidx454 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 17, !dbg !2816
  %198 = bitcast %struct.dcomplex* %arrayidx454 to i8*, !dbg !2817
  %199 = bitcast %struct.dcomplex* %ref.tmp451 to i8*, !dbg !2817
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %198, i8* align 8 %199, i64 16, i1 false), !dbg !2817
  %real456 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp455, i32 0, i32 0, !dbg !2818
  store double 0x407FFFF527E792B0, double* %real456, align 8, !dbg !2818
  %imag457 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp455, i32 0, i32 1, !dbg !2818
  store double 0x4080006161DD7A20, double* %imag457, align 8, !dbg !2818
  %arrayidx458 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 18, !dbg !2819
  %200 = bitcast %struct.dcomplex* %arrayidx458 to i8*, !dbg !2820
  %201 = bitcast %struct.dcomplex* %ref.tmp455 to i8*, !dbg !2820
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %200, i8* align 8 %201, i64 16, i1 false), !dbg !2820
  %real460 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp459, i32 0, i32 0, !dbg !2821
  store double 0x407FFFEF5224A658, double* %real460, align 8, !dbg !2821
  %imag461 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp459, i32 0, i32 1, !dbg !2821
  store double 0x40800064D2F0E0FB, double* %imag461, align 8, !dbg !2821
  %arrayidx462 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 19, !dbg !2822
  %202 = bitcast %struct.dcomplex* %arrayidx462 to i8*, !dbg !2823
  %203 = bitcast %struct.dcomplex* %ref.tmp459 to i8*, !dbg !2823
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %202, i8* align 8 %203, i64 16, i1 false), !dbg !2823
  %real464 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp463, i32 0, i32 0, !dbg !2824
  store double 0x407FFFE97985082F, double* %real464, align 8, !dbg !2824
  %imag465 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp463, i32 0, i32 1, !dbg !2824
  store double 0x40800067BBA76761, double* %imag465, align 8, !dbg !2824
  %arrayidx466 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 20, !dbg !2825
  %204 = bitcast %struct.dcomplex* %arrayidx466 to i8*, !dbg !2826
  %205 = bitcast %struct.dcomplex* %ref.tmp463 to i8*, !dbg !2826
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %204, i8* align 8 %205, i64 16, i1 false), !dbg !2826
  %real468 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp467, i32 0, i32 0, !dbg !2827
  store double 0x407FFFE3A76CE198, double* %real468, align 8, !dbg !2827
  %imag469 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp467, i32 0, i32 1, !dbg !2827
  store double 0x4080006A3087F53C, double* %imag469, align 8, !dbg !2827
  %arrayidx470 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 21, !dbg !2828
  %206 = bitcast %struct.dcomplex* %arrayidx470 to i8*, !dbg !2829
  %207 = bitcast %struct.dcomplex* %ref.tmp467 to i8*, !dbg !2829
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %206, i8* align 8 %207, i64 16, i1 false), !dbg !2829
  %real472 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp471, i32 0, i32 0, !dbg !2830
  store double 0x407FFFDDE458AC2A, double* %real472, align 8, !dbg !2830
  %imag473 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp471, i32 0, i32 1, !dbg !2830
  store double 0x4080006C427E60CB, double* %imag473, align 8, !dbg !2830
  %arrayidx474 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 22, !dbg !2831
  %208 = bitcast %struct.dcomplex* %arrayidx474 to i8*, !dbg !2832
  %209 = bitcast %struct.dcomplex* %ref.tmp471 to i8*, !dbg !2832
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %208, i8* align 8 %209, i64 16, i1 false), !dbg !2832
  %real476 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp475, i32 0, i32 0, !dbg !2833
  store double 0x407FFFD8379EC190, double* %real476, align 8, !dbg !2833
  %imag477 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp475, i32 0, i32 1, !dbg !2833
  store double 0x4080006DFF9235BC, double* %imag477, align 8, !dbg !2833
  %arrayidx478 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 23, !dbg !2834
  %210 = bitcast %struct.dcomplex* %arrayidx478 to i8*, !dbg !2835
  %211 = bitcast %struct.dcomplex* %ref.tmp475 to i8*, !dbg !2835
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %210, i8* align 8 %211, i64 16, i1 false), !dbg !2835
  %real480 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp479, i32 0, i32 0, !dbg !2836
  store double 0x407FFFD2A76113A7, double* %real480, align 8, !dbg !2836
  %imag481 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp479, i32 0, i32 1, !dbg !2836
  store double 0x4080006F7377203C, double* %imag481, align 8, !dbg !2836
  %arrayidx482 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 24, !dbg !2837
  %212 = bitcast %struct.dcomplex* %arrayidx482 to i8*, !dbg !2838
  %213 = bitcast %struct.dcomplex* %ref.tmp479 to i8*, !dbg !2838
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %212, i8* align 8 %213, i64 16, i1 false), !dbg !2838
  %real484 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp483, i32 0, i32 0, !dbg !2839
  store double 0x407FFFCD389947BC, double* %real484, align 8, !dbg !2839
  %imag485 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp483, i32 0, i32 1, !dbg !2839
  store double 0x40800070A7FF2BFD, double* %imag485, align 8, !dbg !2839
  %arrayidx486 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 25, !dbg !2840
  %214 = bitcast %struct.dcomplex* %arrayidx486 to i8*, !dbg !2841
  %215 = bitcast %struct.dcomplex* %ref.tmp483 to i8*, !dbg !2841
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %214, i8* align 8 %215, i64 16, i1 false), !dbg !2841
  br label %if.end, !dbg !2842

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
  %216 = load i8, i8* %class_npb, align 1, !dbg !2843
  %conv = sext i8 %216 to i32, !dbg !2843
  %cmp493 = icmp ne i32 %conv, 85, !dbg !2845
  br i1 %cmp493, label %if.then494, label %if.end588, !dbg !2846

if.then494:                                       ; preds = %if.end492
  store i32 1, i32* %verified, align 4, !dbg !2847
  call void @llvm.dbg.value(metadata i32 1, metadata !2849, metadata !DIExpression()), !dbg !2421
  %217 = sext i32 %nt to i64, !dbg !2850
  br label %for.cond, !dbg !2850

for.cond:                                         ; preds = %for.inc, %if.then494
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 1, %if.then494 ], !dbg !2852
  call void @llvm.dbg.value(metadata i64 %indvars.iv, metadata !2849, metadata !DIExpression()), !dbg !2421
  %cmp495 = icmp sle i64 %indvars.iv, %217, !dbg !2853
  br i1 %cmp495, label %for.body, label %for.end.loopexit, !dbg !2855

for.body:                                         ; preds = %for.cond
  %real496 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %agg.tmp, i32 0, i32 0, !dbg !2856
  %218 = load %struct.dcomplex*, %struct.dcomplex** @_ZL4sums, align 8, !dbg !2856
  %arrayidx497 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %218, i64 %indvars.iv, !dbg !2856
  %real498 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx497, i32 0, i32 0, !dbg !2856
  %219 = load double, double* %real498, align 8, !dbg !2856
  %arrayidx500 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 %indvars.iv, !dbg !2856
  %real501 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx500, i32 0, i32 0, !dbg !2856
  %220 = load double, double* %real501, align 16, !dbg !2856
  %sub = fsub contract double %219, %220, !dbg !2856
  store double %sub, double* %real496, align 8, !dbg !2856
  %imag502 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %agg.tmp, i32 0, i32 1, !dbg !2856
  %221 = load %struct.dcomplex*, %struct.dcomplex** @_ZL4sums, align 8, !dbg !2856
  %arrayidx504 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %221, i64 %indvars.iv, !dbg !2856
  %imag505 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx504, i32 0, i32 1, !dbg !2856
  %222 = load double, double* %imag505, align 8, !dbg !2856
  %arrayidx507 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 %indvars.iv, !dbg !2856
  %imag508 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx507, i32 0, i32 1, !dbg !2856
  %223 = load double, double* %imag508, align 8, !dbg !2856
  %sub509 = fsub contract double %222, %223, !dbg !2856
  store double %sub509, double* %imag502, align 8, !dbg !2856
  %arrayidx512 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 %indvars.iv, !dbg !2856
  %224 = bitcast %struct.dcomplex* %agg.tmp510 to i8*, !dbg !2856
  %225 = bitcast %struct.dcomplex* %arrayidx512 to i8*, !dbg !2856
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %224, i8* align 16 %225, i64 16, i1 false), !dbg !2856
  %226 = bitcast %struct.dcomplex* %agg.tmp to { double, double }*, !dbg !2856
  %227 = getelementptr inbounds { double, double }, { double, double }* %226, i32 0, i32 0, !dbg !2856
  %228 = load double, double* %227, align 8, !dbg !2856
  %229 = getelementptr inbounds { double, double }, { double, double }* %226, i32 0, i32 1, !dbg !2856
  %230 = load double, double* %229, align 8, !dbg !2856
  %231 = bitcast %struct.dcomplex* %agg.tmp510 to { double, double }*, !dbg !2856
  %232 = getelementptr inbounds { double, double }, { double, double }* %231, i32 0, i32 0, !dbg !2856
  %233 = load double, double* %232, align 8, !dbg !2856
  %234 = getelementptr inbounds { double, double }, { double, double }* %231, i32 0, i32 1, !dbg !2856
  %235 = load double, double* %234, align 8, !dbg !2856
  %call = call { double, double } @_ZL12dcomplex_div8dcomplexS_(double %228, double %230, double %233, double %235), !dbg !2856
  %236 = bitcast %struct.dcomplex* %coerce to { double, double }*, !dbg !2856
  %237 = getelementptr inbounds { double, double }, { double, double }* %236, i32 0, i32 0, !dbg !2856
  %238 = extractvalue { double, double } %call, 0, !dbg !2856
  store double %238, double* %237, align 8, !dbg !2856
  %239 = getelementptr inbounds { double, double }, { double, double }* %236, i32 0, i32 1, !dbg !2856
  %240 = extractvalue { double, double } %call, 1, !dbg !2856
  store double %240, double* %239, align 8, !dbg !2856
  %real513 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %coerce, i32 0, i32 0, !dbg !2856
  %241 = load double, double* %real513, align 8, !dbg !2856
  %real515 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %agg.tmp514, i32 0, i32 0, !dbg !2856
  %242 = load %struct.dcomplex*, %struct.dcomplex** @_ZL4sums, align 8, !dbg !2856
  %arrayidx517 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %242, i64 %indvars.iv, !dbg !2856
  %real518 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx517, i32 0, i32 0, !dbg !2856
  %243 = load double, double* %real518, align 8, !dbg !2856
  %arrayidx520 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 %indvars.iv, !dbg !2856
  %real521 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx520, i32 0, i32 0, !dbg !2856
  %244 = load double, double* %real521, align 16, !dbg !2856
  %sub522 = fsub contract double %243, %244, !dbg !2856
  store double %sub522, double* %real515, align 8, !dbg !2856
  %imag523 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %agg.tmp514, i32 0, i32 1, !dbg !2856
  %245 = load %struct.dcomplex*, %struct.dcomplex** @_ZL4sums, align 8, !dbg !2856
  %arrayidx525 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %245, i64 %indvars.iv, !dbg !2856
  %imag526 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx525, i32 0, i32 1, !dbg !2856
  %246 = load double, double* %imag526, align 8, !dbg !2856
  %arrayidx528 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 %indvars.iv, !dbg !2856
  %imag529 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx528, i32 0, i32 1, !dbg !2856
  %247 = load double, double* %imag529, align 8, !dbg !2856
  %sub530 = fsub contract double %246, %247, !dbg !2856
  store double %sub530, double* %imag523, align 8, !dbg !2856
  %arrayidx533 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 %indvars.iv, !dbg !2856
  %248 = bitcast %struct.dcomplex* %agg.tmp531 to i8*, !dbg !2856
  %249 = bitcast %struct.dcomplex* %arrayidx533 to i8*, !dbg !2856
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %248, i8* align 16 %249, i64 16, i1 false), !dbg !2856
  %250 = bitcast %struct.dcomplex* %agg.tmp514 to { double, double }*, !dbg !2856
  %251 = getelementptr inbounds { double, double }, { double, double }* %250, i32 0, i32 0, !dbg !2856
  %252 = load double, double* %251, align 8, !dbg !2856
  %253 = getelementptr inbounds { double, double }, { double, double }* %250, i32 0, i32 1, !dbg !2856
  %254 = load double, double* %253, align 8, !dbg !2856
  %255 = bitcast %struct.dcomplex* %agg.tmp531 to { double, double }*, !dbg !2856
  %256 = getelementptr inbounds { double, double }, { double, double }* %255, i32 0, i32 0, !dbg !2856
  %257 = load double, double* %256, align 8, !dbg !2856
  %258 = getelementptr inbounds { double, double }, { double, double }* %255, i32 0, i32 1, !dbg !2856
  %259 = load double, double* %258, align 8, !dbg !2856
  %call534 = call { double, double } @_ZL12dcomplex_div8dcomplexS_(double %252, double %254, double %257, double %259), !dbg !2856
  %260 = bitcast %struct.dcomplex* %coerce535 to { double, double }*, !dbg !2856
  %261 = getelementptr inbounds { double, double }, { double, double }* %260, i32 0, i32 0, !dbg !2856
  %262 = extractvalue { double, double } %call534, 0, !dbg !2856
  store double %262, double* %261, align 8, !dbg !2856
  %263 = getelementptr inbounds { double, double }, { double, double }* %260, i32 0, i32 1, !dbg !2856
  %264 = extractvalue { double, double } %call534, 1, !dbg !2856
  store double %264, double* %263, align 8, !dbg !2856
  %real536 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %coerce535, i32 0, i32 0, !dbg !2856
  %265 = load double, double* %real536, align 8, !dbg !2856
  %mul = fmul contract double %241, %265, !dbg !2856
  %real538 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %agg.tmp537, i32 0, i32 0, !dbg !2856
  %266 = load %struct.dcomplex*, %struct.dcomplex** @_ZL4sums, align 8, !dbg !2856
  %arrayidx540 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %266, i64 %indvars.iv, !dbg !2856
  %real541 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx540, i32 0, i32 0, !dbg !2856
  %267 = load double, double* %real541, align 8, !dbg !2856
  %arrayidx543 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 %indvars.iv, !dbg !2856
  %real544 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx543, i32 0, i32 0, !dbg !2856
  %268 = load double, double* %real544, align 16, !dbg !2856
  %sub545 = fsub contract double %267, %268, !dbg !2856
  store double %sub545, double* %real538, align 8, !dbg !2856
  %imag546 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %agg.tmp537, i32 0, i32 1, !dbg !2856
  %269 = load %struct.dcomplex*, %struct.dcomplex** @_ZL4sums, align 8, !dbg !2856
  %arrayidx548 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %269, i64 %indvars.iv, !dbg !2856
  %imag549 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx548, i32 0, i32 1, !dbg !2856
  %270 = load double, double* %imag549, align 8, !dbg !2856
  %arrayidx551 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 %indvars.iv, !dbg !2856
  %imag552 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx551, i32 0, i32 1, !dbg !2856
  %271 = load double, double* %imag552, align 8, !dbg !2856
  %sub553 = fsub contract double %270, %271, !dbg !2856
  store double %sub553, double* %imag546, align 8, !dbg !2856
  %arrayidx556 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 %indvars.iv, !dbg !2856
  %272 = bitcast %struct.dcomplex* %agg.tmp554 to i8*, !dbg !2856
  %273 = bitcast %struct.dcomplex* %arrayidx556 to i8*, !dbg !2856
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %272, i8* align 16 %273, i64 16, i1 false), !dbg !2856
  %274 = bitcast %struct.dcomplex* %agg.tmp537 to { double, double }*, !dbg !2856
  %275 = getelementptr inbounds { double, double }, { double, double }* %274, i32 0, i32 0, !dbg !2856
  %276 = load double, double* %275, align 8, !dbg !2856
  %277 = getelementptr inbounds { double, double }, { double, double }* %274, i32 0, i32 1, !dbg !2856
  %278 = load double, double* %277, align 8, !dbg !2856
  %279 = bitcast %struct.dcomplex* %agg.tmp554 to { double, double }*, !dbg !2856
  %280 = getelementptr inbounds { double, double }, { double, double }* %279, i32 0, i32 0, !dbg !2856
  %281 = load double, double* %280, align 8, !dbg !2856
  %282 = getelementptr inbounds { double, double }, { double, double }* %279, i32 0, i32 1, !dbg !2856
  %283 = load double, double* %282, align 8, !dbg !2856
  %call557 = call { double, double } @_ZL12dcomplex_div8dcomplexS_(double %276, double %278, double %281, double %283), !dbg !2856
  %284 = bitcast %struct.dcomplex* %coerce558 to { double, double }*, !dbg !2856
  %285 = getelementptr inbounds { double, double }, { double, double }* %284, i32 0, i32 0, !dbg !2856
  %286 = extractvalue { double, double } %call557, 0, !dbg !2856
  store double %286, double* %285, align 8, !dbg !2856
  %287 = getelementptr inbounds { double, double }, { double, double }* %284, i32 0, i32 1, !dbg !2856
  %288 = extractvalue { double, double } %call557, 1, !dbg !2856
  store double %288, double* %287, align 8, !dbg !2856
  %imag559 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %coerce558, i32 0, i32 1, !dbg !2856
  %289 = load double, double* %imag559, align 8, !dbg !2856
  %real561 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %agg.tmp560, i32 0, i32 0, !dbg !2856
  %290 = load %struct.dcomplex*, %struct.dcomplex** @_ZL4sums, align 8, !dbg !2856
  %arrayidx563 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %290, i64 %indvars.iv, !dbg !2856
  %real564 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx563, i32 0, i32 0, !dbg !2856
  %291 = load double, double* %real564, align 8, !dbg !2856
  %arrayidx566 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 %indvars.iv, !dbg !2856
  %real567 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx566, i32 0, i32 0, !dbg !2856
  %292 = load double, double* %real567, align 16, !dbg !2856
  %sub568 = fsub contract double %291, %292, !dbg !2856
  store double %sub568, double* %real561, align 8, !dbg !2856
  %imag569 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %agg.tmp560, i32 0, i32 1, !dbg !2856
  %293 = load %struct.dcomplex*, %struct.dcomplex** @_ZL4sums, align 8, !dbg !2856
  %arrayidx571 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %293, i64 %indvars.iv, !dbg !2856
  %imag572 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx571, i32 0, i32 1, !dbg !2856
  %294 = load double, double* %imag572, align 8, !dbg !2856
  %arrayidx574 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 %indvars.iv, !dbg !2856
  %imag575 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx574, i32 0, i32 1, !dbg !2856
  %295 = load double, double* %imag575, align 8, !dbg !2856
  %sub576 = fsub contract double %294, %295, !dbg !2856
  store double %sub576, double* %imag569, align 8, !dbg !2856
  %arrayidx579 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 %indvars.iv, !dbg !2856
  %296 = bitcast %struct.dcomplex* %agg.tmp577 to i8*, !dbg !2856
  %297 = bitcast %struct.dcomplex* %arrayidx579 to i8*, !dbg !2856
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %296, i8* align 16 %297, i64 16, i1 false), !dbg !2856
  %298 = bitcast %struct.dcomplex* %agg.tmp560 to { double, double }*, !dbg !2856
  %299 = getelementptr inbounds { double, double }, { double, double }* %298, i32 0, i32 0, !dbg !2856
  %300 = load double, double* %299, align 8, !dbg !2856
  %301 = getelementptr inbounds { double, double }, { double, double }* %298, i32 0, i32 1, !dbg !2856
  %302 = load double, double* %301, align 8, !dbg !2856
  %303 = bitcast %struct.dcomplex* %agg.tmp577 to { double, double }*, !dbg !2856
  %304 = getelementptr inbounds { double, double }, { double, double }* %303, i32 0, i32 0, !dbg !2856
  %305 = load double, double* %304, align 8, !dbg !2856
  %306 = getelementptr inbounds { double, double }, { double, double }* %303, i32 0, i32 1, !dbg !2856
  %307 = load double, double* %306, align 8, !dbg !2856
  %call580 = call { double, double } @_ZL12dcomplex_div8dcomplexS_(double %300, double %302, double %305, double %307), !dbg !2856
  %308 = bitcast %struct.dcomplex* %coerce581 to { double, double }*, !dbg !2856
  %309 = getelementptr inbounds { double, double }, { double, double }* %308, i32 0, i32 0, !dbg !2856
  %310 = extractvalue { double, double } %call580, 0, !dbg !2856
  store double %310, double* %309, align 8, !dbg !2856
  %311 = getelementptr inbounds { double, double }, { double, double }* %308, i32 0, i32 1, !dbg !2856
  %312 = extractvalue { double, double } %call580, 1, !dbg !2856
  store double %312, double* %311, align 8, !dbg !2856
  %imag582 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %coerce581, i32 0, i32 1, !dbg !2856
  %313 = load double, double* %imag582, align 8, !dbg !2856
  %mul583 = fmul contract double %289, %313, !dbg !2856
  %add = fadd contract double %mul, %mul583, !dbg !2856
  %call584 = call double @sqrt(double %add) #11, !dbg !2856
  call void @llvm.dbg.value(metadata double %call584, metadata !2858, metadata !DIExpression()), !dbg !2421
  %cmp585 = fcmp ole double %call584, 0x3D719799812DEA11, !dbg !2859
  br i1 %cmp585, label %if.end587, label %if.then586, !dbg !2861

if.then586:                                       ; preds = %for.body
  store i32 0, i32* %verified, align 4, !dbg !2862
  br label %for.end, !dbg !2864

if.end587:                                        ; preds = %for.body
  br label %for.inc, !dbg !2865

for.inc:                                          ; preds = %if.end587
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !2866
  call void @llvm.dbg.value(metadata i32 undef, metadata !2849, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !2421
  br label %for.cond, !dbg !2867, !llvm.loop !2868

for.end.loopexit:                                 ; preds = %for.cond
  br label %for.end, !dbg !2870

for.end:                                          ; preds = %for.end.loopexit, %if.then586
  br label %if.end588, !dbg !2870

if.end588:                                        ; preds = %for.end, %if.end492
  %314 = load i8, i8* %class_npb, align 1, !dbg !2871
  %conv589 = sext i8 %314 to i32, !dbg !2871
  %cmp590 = icmp ne i32 %conv589, 85, !dbg !2873
  br i1 %cmp590, label %if.then591, label %if.end597, !dbg !2874

if.then591:                                       ; preds = %if.end588
  %315 = load i32, i32* %verified, align 4, !dbg !2875
  %tobool = icmp ne i32 %315, 0, !dbg !2875
  br i1 %tobool, label %if.then592, label %if.else594, !dbg !2878

if.then592:                                       ; preds = %if.then591
  %call593 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([33 x i8], [33 x i8]* @.str.73, i64 0, i64 0)), !dbg !2879
  br label %if.end596, !dbg !2881

if.else594:                                       ; preds = %if.then591
  %call595 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([29 x i8], [29 x i8]* @.str.74, i64 0, i64 0)), !dbg !2882
  br label %if.end596

if.end596:                                        ; preds = %if.else594, %if.then592
  br label %if.end597, !dbg !2884

if.end597:                                        ; preds = %if.end596, %if.end588
  %316 = load i8, i8* %class_npb, align 1, !dbg !2885
  %conv598 = sext i8 %316 to i32, !dbg !2885
  %call599 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([17 x i8], [17 x i8]* @.str.75, i64 0, i64 0), i32 %conv598), !dbg !2886
  ret void, !dbg !2887
}

; Function Attrs: nounwind
declare dso_local double @log(double) #9

; Function Attrs: nounwind
declare dso_local i8* @strcpy(i8*, i8*) #9

; Function Attrs: nounwind
declare dso_local i8* @strcat(i8*, i8*) #9

; Function Attrs: noinline uwtable
define internal void @_ZL11release_gpuv() #7 !dbg !2888 {
entry:
  ret void, !dbg !2889
}

; Function Attrs: nounwind
declare dso_local void @free(i8*) #9

; Function Attrs: noinline nounwind uwtable
define internal { double, double } @_ZL12dcomplex_div8dcomplexS_(double %z1.coerce0, double %z1.coerce1, double %z2.coerce0, double %z2.coerce1) #6 !dbg !2890 {
entry:
  %retval = alloca %struct.dcomplex, align 8
  %z1 = alloca %struct.dcomplex, align 8
  %z2 = alloca %struct.dcomplex, align 8
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
  call void @llvm.dbg.declare(metadata %struct.dcomplex* %z1, metadata !2893, metadata !DIExpression()), !dbg !2894
  call void @llvm.dbg.declare(metadata %struct.dcomplex* %z2, metadata !2895, metadata !DIExpression()), !dbg !2896
  %real = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %z1, i32 0, i32 0, !dbg !2897
  %6 = load double, double* %real, align 8, !dbg !2897
  call void @llvm.dbg.value(metadata double %6, metadata !2898, metadata !DIExpression()), !dbg !2899
  %imag = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %z1, i32 0, i32 1, !dbg !2900
  %7 = load double, double* %imag, align 8, !dbg !2900
  call void @llvm.dbg.value(metadata double %7, metadata !2901, metadata !DIExpression()), !dbg !2899
  %real1 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %z2, i32 0, i32 0, !dbg !2902
  %8 = load double, double* %real1, align 8, !dbg !2902
  call void @llvm.dbg.value(metadata double %8, metadata !2903, metadata !DIExpression()), !dbg !2899
  %imag2 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %z2, i32 0, i32 1, !dbg !2904
  %9 = load double, double* %imag2, align 8, !dbg !2904
  call void @llvm.dbg.value(metadata double %9, metadata !2905, metadata !DIExpression()), !dbg !2899
  %mul = fmul contract double %8, %8, !dbg !2906
  %mul3 = fmul contract double %9, %9, !dbg !2907
  %add = fadd contract double %mul, %mul3, !dbg !2908
  call void @llvm.dbg.value(metadata double %add, metadata !2909, metadata !DIExpression()), !dbg !2899
  %mul5 = fmul contract double %6, %8, !dbg !2910
  %mul6 = fmul contract double %7, %9, !dbg !2911
  %add7 = fadd contract double %mul5, %mul6, !dbg !2912
  %div = fdiv double %add7, %add, !dbg !2913
  call void @llvm.dbg.value(metadata double %div, metadata !2914, metadata !DIExpression()), !dbg !2899
  %mul9 = fmul contract double %7, %8, !dbg !2915
  %mul10 = fmul contract double %6, %9, !dbg !2916
  %sub = fsub contract double %mul9, %mul10, !dbg !2917
  %div11 = fdiv double %sub, %add, !dbg !2918
  call void @llvm.dbg.value(metadata double %div11, metadata !2919, metadata !DIExpression()), !dbg !2899
  call void @llvm.dbg.declare(metadata %struct.dcomplex* %retval, metadata !2920, metadata !DIExpression()), !dbg !2921
  %real12 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %retval, i32 0, i32 0, !dbg !2922
  store double %div, double* %real12, align 8, !dbg !2922
  %imag13 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %retval, i32 0, i32 1, !dbg !2922
  store double %div11, double* %imag13, align 8, !dbg !2922
  %10 = bitcast %struct.dcomplex* %retval to { double, double }*, !dbg !2923
  %11 = load { double, double }, { double, double }* %10, align 8, !dbg !2923
  ret { double, double } %11, !dbg !2923
}

; Function Attrs: nounwind
declare dso_local double @sqrt(double) #9

declare dso_local i32 @cudaSetupArgument(i8*, i64, i64)

declare dso_local i32 @cudaLaunch(i8*)

; Function Attrs: noinline uwtable
define internal void @_ZL10cffts1_gpuiP8dcomplexS0_S0_S0_S0_(i32 %is, %struct.dcomplex* %u, %struct.dcomplex* %x_in, %struct.dcomplex* %x_out, %struct.dcomplex* %y0, %struct.dcomplex* %y1) #7 !dbg !2924 {
entry:
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
  call void @llvm.dbg.value(metadata i32 %is, metadata !2927, metadata !DIExpression()), !dbg !2928
  call void @llvm.dbg.value(metadata %struct.dcomplex* %u, metadata !2929, metadata !DIExpression()), !dbg !2928
  call void @llvm.dbg.value(metadata %struct.dcomplex* %x_in, metadata !2930, metadata !DIExpression()), !dbg !2928
  call void @llvm.dbg.value(metadata %struct.dcomplex* %x_out, metadata !2931, metadata !DIExpression()), !dbg !2928
  call void @llvm.dbg.value(metadata %struct.dcomplex* %y0, metadata !2932, metadata !DIExpression()), !dbg !2928
  call void @llvm.dbg.value(metadata %struct.dcomplex* %y1, metadata !2933, metadata !DIExpression()), !dbg !2928
  %0 = load i32, i32* @blocks_per_grid_on_fftx_1, align 4, !dbg !2934
  %dim3gep.0 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp, i32 0, i32 0
  store i32 %0, i32* %dim3gep.0
  %dim3gep.1 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp, i32 0, i32 1
  store i32 1, i32* %dim3gep.1
  %dim3gep.2 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp, i32 0, i32 2
  store i32 1, i32* %dim3gep.2
  %1 = load i32, i32* @threads_per_block_on_fftx_1, align 4, !dbg !2935
  %dim3gep.01 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp1, i32 0, i32 0
  store i32 %1, i32* %dim3gep.01
  %dim3gep.12 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp1, i32 0, i32 1
  store i32 1, i32* %dim3gep.12
  %dim3gep.23 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp1, i32 0, i32 2
  store i32 1, i32* %dim3gep.23
  %2 = bitcast { i64, i32 }* %agg.tmp.coerce to i8*, !dbg !2936
  %3 = bitcast %struct.dim3* %agg.tmp to i8*, !dbg !2936
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %2, i8* align 4 %3, i64 12, i1 false), !dbg !2936
  %4 = bitcast { i64, i32 }* %agg.tmp1.coerce to i8*, !dbg !2936
  %5 = bitcast %struct.dim3* %agg.tmp1 to i8*, !dbg !2936
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %4, i8* align 4 %5, i64 12, i1 false), !dbg !2936
  br label %header.0

header.0:                                         ; preds = %latch.0, %entry
  %indvar.0 = phi i32 [ 0, %entry ], [ %indvar.next.0, %latch.0 ]
  %exitcond40 = icmp ne i32 %indvar.0, %0
  br i1 %exitcond40, label %header.1.preheader, label %kcall.end, !tulip.doall.loop.grid !1860

header.1.preheader:                               ; preds = %header.0
  br label %header.1

header.1:                                         ; preds = %header.1.preheader, %latch.1
  %indvar.1 = phi i32 [ %indvar.next.1, %latch.1 ], [ 0, %header.1.preheader ]
  %exitcond39 = icmp ne i32 %indvar.1, %1
  br i1 %exitcond39, label %kcall.configok, label %latch.0, !tulip.doall.loop.block !1860

latch.1:                                          ; preds = %kcall.configok
  %indvar.next.1 = add i32 %indvar.1, 1
  br label %header.1

latch.0:                                          ; preds = %header.1
  %indvar.next.0 = add i32 %indvar.0, 1
  br label %header.0

kcall.configok:                                   ; preds = %header.1
  call void @cffts1_gpu_kernel_1(%struct.dcomplex* %x_in, %struct.dcomplex* %y0, i32 %0, i32 1, i32 1, i32 %1, i32 1, i32 1, i32 %indvar.0, i32 0, i32 0, i32 %indvar.1, i32 0, i32 0)
  br label %latch.1

kcall.end:                                        ; preds = %header.0
  %6 = load i32, i32* @blocks_per_grid_on_fftx_2, align 4, !dbg !2937
  %dim3gep.04 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp3, i32 0, i32 0
  store i32 %6, i32* %dim3gep.04
  %dim3gep.15 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp3, i32 0, i32 1
  store i32 1, i32* %dim3gep.15
  %dim3gep.26 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp3, i32 0, i32 2
  store i32 1, i32* %dim3gep.26
  %7 = load i32, i32* @threads_per_block_on_fftx_2, align 4, !dbg !2938
  %dim3gep.07 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp4, i32 0, i32 0
  store i32 %7, i32* %dim3gep.07
  %dim3gep.18 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp4, i32 0, i32 1
  store i32 1, i32* %dim3gep.18
  %dim3gep.29 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp4, i32 0, i32 2
  store i32 1, i32* %dim3gep.29
  %8 = bitcast { i64, i32 }* %agg.tmp3.coerce to i8*, !dbg !2939
  %9 = bitcast %struct.dim3* %agg.tmp3 to i8*, !dbg !2939
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %8, i8* align 4 %9, i64 12, i1 false), !dbg !2939
  %10 = bitcast { i64, i32 }* %agg.tmp4.coerce to i8*, !dbg !2939
  %11 = bitcast %struct.dim3* %agg.tmp4 to i8*, !dbg !2939
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %10, i8* align 4 %11, i64 12, i1 false), !dbg !2939
  br label %header.016

header.016:                                       ; preds = %latch.019, %kcall.end
  %indvar.023 = phi i32 [ 0, %kcall.end ], [ %indvar.next.025, %latch.019 ]
  %exitcond38 = icmp ne i32 %indvar.023, %6
  br i1 %exitcond38, label %header.117.preheader, label %kcall.end8, !tulip.doall.loop.grid !1860

header.117.preheader:                             ; preds = %header.016
  br label %header.117

header.117:                                       ; preds = %header.117.preheader, %latch.118
  %indvar.120 = phi i32 [ %indvar.next.122, %latch.118 ], [ 0, %header.117.preheader ]
  %exitcond37 = icmp ne i32 %indvar.120, %7
  br i1 %exitcond37, label %kcall.configok7, label %latch.019, !tulip.doall.loop.block !1860

latch.118:                                        ; preds = %kcall.configok7
  %indvar.next.122 = add i32 %indvar.120, 1
  br label %header.117

latch.019:                                        ; preds = %header.117
  %indvar.next.025 = add i32 %indvar.023, 1
  br label %header.016

kcall.configok7:                                  ; preds = %header.117
  call void @cffts1_gpu_kernel_2(i32 %is, %struct.dcomplex* %y0, %struct.dcomplex* %y1, %struct.dcomplex* %u, i32 %6, i32 1, i32 1, i32 %7, i32 1, i32 1, i32 %indvar.023, i32 0, i32 0, i32 %indvar.120, i32 0, i32 0)
  br label %latch.118

kcall.end8:                                       ; preds = %header.016
  %12 = load i32, i32* @blocks_per_grid_on_fftx_3, align 4, !dbg !2940
  %dim3gep.010 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp10, i32 0, i32 0
  store i32 %12, i32* %dim3gep.010
  %dim3gep.111 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp10, i32 0, i32 1
  store i32 1, i32* %dim3gep.111
  %dim3gep.212 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp10, i32 0, i32 2
  store i32 1, i32* %dim3gep.212
  %13 = load i32, i32* @threads_per_block_on_fftx_3, align 4, !dbg !2941
  %dim3gep.013 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp11, i32 0, i32 0
  store i32 %13, i32* %dim3gep.013
  %dim3gep.114 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp11, i32 0, i32 1
  store i32 1, i32* %dim3gep.114
  %dim3gep.215 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp11, i32 0, i32 2
  store i32 1, i32* %dim3gep.215
  %14 = bitcast { i64, i32 }* %agg.tmp10.coerce to i8*, !dbg !2942
  %15 = bitcast %struct.dim3* %agg.tmp10 to i8*, !dbg !2942
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %14, i8* align 4 %15, i64 12, i1 false), !dbg !2942
  %16 = bitcast { i64, i32 }* %agg.tmp11.coerce to i8*, !dbg !2942
  %17 = bitcast %struct.dim3* %agg.tmp11 to i8*, !dbg !2942
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %16, i8* align 4 %17, i64 12, i1 false), !dbg !2942
  br label %header.026

header.026:                                       ; preds = %latch.029, %kcall.end8
  %indvar.033 = phi i32 [ 0, %kcall.end8 ], [ %indvar.next.035, %latch.029 ]
  %exitcond36 = icmp ne i32 %indvar.033, %12
  br i1 %exitcond36, label %header.127.preheader, label %kcall.end15, !tulip.doall.loop.grid !1860

header.127.preheader:                             ; preds = %header.026
  br label %header.127

header.127:                                       ; preds = %header.127.preheader, %latch.128
  %indvar.130 = phi i32 [ %indvar.next.132, %latch.128 ], [ 0, %header.127.preheader ]
  %exitcond = icmp ne i32 %indvar.130, %13
  br i1 %exitcond, label %kcall.configok14, label %latch.029, !tulip.doall.loop.block !1860

latch.128:                                        ; preds = %kcall.configok14
  %indvar.next.132 = add i32 %indvar.130, 1
  br label %header.127

latch.029:                                        ; preds = %header.127
  %indvar.next.035 = add i32 %indvar.033, 1
  br label %header.026

kcall.configok14:                                 ; preds = %header.127
  call void @cffts1_gpu_kernel_3(%struct.dcomplex* %x_out, %struct.dcomplex* %y0, i32 %12, i32 1, i32 1, i32 %13, i32 1, i32 1, i32 %indvar.033, i32 0, i32 0, i32 %indvar.130, i32 0, i32 0)
  br label %latch.128

kcall.end15:                                      ; preds = %header.026
  ret void, !dbg !2943
}

; Function Attrs: noinline uwtable
define internal void @_ZL10cffts2_gpuiP8dcomplexS0_S0_S0_S0_(i32 %is, %struct.dcomplex* %u, %struct.dcomplex* %x_in, %struct.dcomplex* %x_out, %struct.dcomplex* %y0, %struct.dcomplex* %y1) #7 !dbg !2944 {
entry:
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
  call void @llvm.dbg.value(metadata i32 %is, metadata !2947, metadata !DIExpression()), !dbg !2948
  call void @llvm.dbg.value(metadata %struct.dcomplex* %u, metadata !2949, metadata !DIExpression()), !dbg !2948
  call void @llvm.dbg.value(metadata %struct.dcomplex* %x_in, metadata !2950, metadata !DIExpression()), !dbg !2948
  call void @llvm.dbg.value(metadata %struct.dcomplex* %x_out, metadata !2951, metadata !DIExpression()), !dbg !2948
  call void @llvm.dbg.value(metadata %struct.dcomplex* %y0, metadata !2952, metadata !DIExpression()), !dbg !2948
  call void @llvm.dbg.value(metadata %struct.dcomplex* %y1, metadata !2953, metadata !DIExpression()), !dbg !2948
  %0 = load i32, i32* @blocks_per_grid_on_ffty_1, align 4, !dbg !2954
  %dim3gep.0 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp, i32 0, i32 0
  store i32 %0, i32* %dim3gep.0
  %dim3gep.1 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp, i32 0, i32 1
  store i32 1, i32* %dim3gep.1
  %dim3gep.2 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp, i32 0, i32 2
  store i32 1, i32* %dim3gep.2
  %1 = load i32, i32* @threads_per_block_on_ffty_1, align 4, !dbg !2955
  %dim3gep.01 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp1, i32 0, i32 0
  store i32 %1, i32* %dim3gep.01
  %dim3gep.12 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp1, i32 0, i32 1
  store i32 1, i32* %dim3gep.12
  %dim3gep.23 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp1, i32 0, i32 2
  store i32 1, i32* %dim3gep.23
  %2 = bitcast { i64, i32 }* %agg.tmp.coerce to i8*, !dbg !2956
  %3 = bitcast %struct.dim3* %agg.tmp to i8*, !dbg !2956
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %2, i8* align 4 %3, i64 12, i1 false), !dbg !2956
  %4 = bitcast { i64, i32 }* %agg.tmp1.coerce to i8*, !dbg !2956
  %5 = bitcast %struct.dim3* %agg.tmp1 to i8*, !dbg !2956
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %4, i8* align 4 %5, i64 12, i1 false), !dbg !2956
  br label %header.0

header.0:                                         ; preds = %latch.0, %entry
  %indvar.0 = phi i32 [ 0, %entry ], [ %indvar.next.0, %latch.0 ]
  %exitcond40 = icmp ne i32 %indvar.0, %0
  br i1 %exitcond40, label %header.1.preheader, label %kcall.end, !tulip.doall.loop.grid !1860

header.1.preheader:                               ; preds = %header.0
  br label %header.1

header.1:                                         ; preds = %header.1.preheader, %latch.1
  %indvar.1 = phi i32 [ %indvar.next.1, %latch.1 ], [ 0, %header.1.preheader ]
  %exitcond39 = icmp ne i32 %indvar.1, %1
  br i1 %exitcond39, label %kcall.configok, label %latch.0, !tulip.doall.loop.block !1860

latch.1:                                          ; preds = %kcall.configok
  %indvar.next.1 = add i32 %indvar.1, 1
  br label %header.1

latch.0:                                          ; preds = %header.1
  %indvar.next.0 = add i32 %indvar.0, 1
  br label %header.0

kcall.configok:                                   ; preds = %header.1
  call void @cffts2_gpu_kernel_1(%struct.dcomplex* %x_in, %struct.dcomplex* %y0, i32 %0, i32 1, i32 1, i32 %1, i32 1, i32 1, i32 %indvar.0, i32 0, i32 0, i32 %indvar.1, i32 0, i32 0)
  br label %latch.1

kcall.end:                                        ; preds = %header.0
  %6 = load i32, i32* @blocks_per_grid_on_ffty_2, align 4, !dbg !2957
  %dim3gep.04 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp3, i32 0, i32 0
  store i32 %6, i32* %dim3gep.04
  %dim3gep.15 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp3, i32 0, i32 1
  store i32 1, i32* %dim3gep.15
  %dim3gep.26 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp3, i32 0, i32 2
  store i32 1, i32* %dim3gep.26
  %7 = load i32, i32* @threads_per_block_on_ffty_2, align 4, !dbg !2958
  %dim3gep.07 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp4, i32 0, i32 0
  store i32 %7, i32* %dim3gep.07
  %dim3gep.18 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp4, i32 0, i32 1
  store i32 1, i32* %dim3gep.18
  %dim3gep.29 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp4, i32 0, i32 2
  store i32 1, i32* %dim3gep.29
  %8 = bitcast { i64, i32 }* %agg.tmp3.coerce to i8*, !dbg !2959
  %9 = bitcast %struct.dim3* %agg.tmp3 to i8*, !dbg !2959
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %8, i8* align 4 %9, i64 12, i1 false), !dbg !2959
  %10 = bitcast { i64, i32 }* %agg.tmp4.coerce to i8*, !dbg !2959
  %11 = bitcast %struct.dim3* %agg.tmp4 to i8*, !dbg !2959
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %10, i8* align 4 %11, i64 12, i1 false), !dbg !2959
  br label %header.016

header.016:                                       ; preds = %latch.019, %kcall.end
  %indvar.023 = phi i32 [ 0, %kcall.end ], [ %indvar.next.025, %latch.019 ]
  %exitcond38 = icmp ne i32 %indvar.023, %6
  br i1 %exitcond38, label %header.117.preheader, label %kcall.end8, !tulip.doall.loop.grid !1860

header.117.preheader:                             ; preds = %header.016
  br label %header.117

header.117:                                       ; preds = %header.117.preheader, %latch.118
  %indvar.120 = phi i32 [ %indvar.next.122, %latch.118 ], [ 0, %header.117.preheader ]
  %exitcond37 = icmp ne i32 %indvar.120, %7
  br i1 %exitcond37, label %kcall.configok7, label %latch.019, !tulip.doall.loop.block !1860

latch.118:                                        ; preds = %kcall.configok7
  %indvar.next.122 = add i32 %indvar.120, 1
  br label %header.117

latch.019:                                        ; preds = %header.117
  %indvar.next.025 = add i32 %indvar.023, 1
  br label %header.016

kcall.configok7:                                  ; preds = %header.117
  call void @cffts2_gpu_kernel_2(i32 %is, %struct.dcomplex* %y0, %struct.dcomplex* %y1, %struct.dcomplex* %u, i32 %6, i32 1, i32 1, i32 %7, i32 1, i32 1, i32 %indvar.023, i32 0, i32 0, i32 %indvar.120, i32 0, i32 0)
  br label %latch.118

kcall.end8:                                       ; preds = %header.016
  %12 = load i32, i32* @blocks_per_grid_on_ffty_3, align 4, !dbg !2960
  %dim3gep.010 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp10, i32 0, i32 0
  store i32 %12, i32* %dim3gep.010
  %dim3gep.111 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp10, i32 0, i32 1
  store i32 1, i32* %dim3gep.111
  %dim3gep.212 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp10, i32 0, i32 2
  store i32 1, i32* %dim3gep.212
  %13 = load i32, i32* @threads_per_block_on_ffty_3, align 4, !dbg !2961
  %dim3gep.013 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp11, i32 0, i32 0
  store i32 %13, i32* %dim3gep.013
  %dim3gep.114 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp11, i32 0, i32 1
  store i32 1, i32* %dim3gep.114
  %dim3gep.215 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp11, i32 0, i32 2
  store i32 1, i32* %dim3gep.215
  %14 = bitcast { i64, i32 }* %agg.tmp10.coerce to i8*, !dbg !2962
  %15 = bitcast %struct.dim3* %agg.tmp10 to i8*, !dbg !2962
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %14, i8* align 4 %15, i64 12, i1 false), !dbg !2962
  %16 = bitcast { i64, i32 }* %agg.tmp11.coerce to i8*, !dbg !2962
  %17 = bitcast %struct.dim3* %agg.tmp11 to i8*, !dbg !2962
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %16, i8* align 4 %17, i64 12, i1 false), !dbg !2962
  br label %header.026

header.026:                                       ; preds = %latch.029, %kcall.end8
  %indvar.033 = phi i32 [ 0, %kcall.end8 ], [ %indvar.next.035, %latch.029 ]
  %exitcond36 = icmp ne i32 %indvar.033, %12
  br i1 %exitcond36, label %header.127.preheader, label %kcall.end15, !tulip.doall.loop.grid !1860

header.127.preheader:                             ; preds = %header.026
  br label %header.127

header.127:                                       ; preds = %header.127.preheader, %latch.128
  %indvar.130 = phi i32 [ %indvar.next.132, %latch.128 ], [ 0, %header.127.preheader ]
  %exitcond = icmp ne i32 %indvar.130, %13
  br i1 %exitcond, label %kcall.configok14, label %latch.029, !tulip.doall.loop.block !1860

latch.128:                                        ; preds = %kcall.configok14
  %indvar.next.132 = add i32 %indvar.130, 1
  br label %header.127

latch.029:                                        ; preds = %header.127
  %indvar.next.035 = add i32 %indvar.033, 1
  br label %header.026

kcall.configok14:                                 ; preds = %header.127
  call void @cffts2_gpu_kernel_3(%struct.dcomplex* %x_out, %struct.dcomplex* %y0, i32 %12, i32 1, i32 1, i32 %13, i32 1, i32 1, i32 %indvar.033, i32 0, i32 0, i32 %indvar.130, i32 0, i32 0)
  br label %latch.128

kcall.end15:                                      ; preds = %header.026
  ret void, !dbg !2963
}

; Function Attrs: noinline uwtable
define internal void @_ZL10cffts3_gpuiP8dcomplexS0_S0_S0_S0_(i32 %is, %struct.dcomplex* %u, %struct.dcomplex* %x_in, %struct.dcomplex* %x_out, %struct.dcomplex* %y0, %struct.dcomplex* %y1) #7 !dbg !2964 {
entry:
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
  call void @llvm.dbg.value(metadata i32 %is, metadata !2965, metadata !DIExpression()), !dbg !2966
  call void @llvm.dbg.value(metadata %struct.dcomplex* %u, metadata !2967, metadata !DIExpression()), !dbg !2966
  call void @llvm.dbg.value(metadata %struct.dcomplex* %x_in, metadata !2968, metadata !DIExpression()), !dbg !2966
  call void @llvm.dbg.value(metadata %struct.dcomplex* %x_out, metadata !2969, metadata !DIExpression()), !dbg !2966
  call void @llvm.dbg.value(metadata %struct.dcomplex* %y0, metadata !2970, metadata !DIExpression()), !dbg !2966
  call void @llvm.dbg.value(metadata %struct.dcomplex* %y1, metadata !2971, metadata !DIExpression()), !dbg !2966
  %0 = load i32, i32* @blocks_per_grid_on_fftz_1, align 4, !dbg !2972
  %dim3gep.0 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp, i32 0, i32 0
  store i32 %0, i32* %dim3gep.0
  %dim3gep.1 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp, i32 0, i32 1
  store i32 1, i32* %dim3gep.1
  %dim3gep.2 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp, i32 0, i32 2
  store i32 1, i32* %dim3gep.2
  %1 = load i32, i32* @threads_per_block_on_fftz_1, align 4, !dbg !2973
  %dim3gep.01 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp1, i32 0, i32 0
  store i32 %1, i32* %dim3gep.01
  %dim3gep.12 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp1, i32 0, i32 1
  store i32 1, i32* %dim3gep.12
  %dim3gep.23 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp1, i32 0, i32 2
  store i32 1, i32* %dim3gep.23
  %2 = bitcast { i64, i32 }* %agg.tmp.coerce to i8*, !dbg !2974
  %3 = bitcast %struct.dim3* %agg.tmp to i8*, !dbg !2974
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %2, i8* align 4 %3, i64 12, i1 false), !dbg !2974
  %4 = bitcast { i64, i32 }* %agg.tmp1.coerce to i8*, !dbg !2974
  %5 = bitcast %struct.dim3* %agg.tmp1 to i8*, !dbg !2974
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %4, i8* align 4 %5, i64 12, i1 false), !dbg !2974
  br label %header.0

header.0:                                         ; preds = %latch.0, %entry
  %indvar.0 = phi i32 [ 0, %entry ], [ %indvar.next.0, %latch.0 ]
  %exitcond40 = icmp ne i32 %indvar.0, %0
  br i1 %exitcond40, label %header.1.preheader, label %kcall.end, !tulip.doall.loop.grid !1860

header.1.preheader:                               ; preds = %header.0
  br label %header.1

header.1:                                         ; preds = %header.1.preheader, %latch.1
  %indvar.1 = phi i32 [ %indvar.next.1, %latch.1 ], [ 0, %header.1.preheader ]
  %exitcond39 = icmp ne i32 %indvar.1, %1
  br i1 %exitcond39, label %kcall.configok, label %latch.0, !tulip.doall.loop.block !1860

latch.1:                                          ; preds = %kcall.configok
  %indvar.next.1 = add i32 %indvar.1, 1
  br label %header.1

latch.0:                                          ; preds = %header.1
  %indvar.next.0 = add i32 %indvar.0, 1
  br label %header.0

kcall.configok:                                   ; preds = %header.1
  call void @cffts3_gpu_kernel_1(%struct.dcomplex* %x_in, %struct.dcomplex* %y0, i32 %0, i32 1, i32 1, i32 %1, i32 1, i32 1, i32 %indvar.0, i32 0, i32 0, i32 %indvar.1, i32 0, i32 0)
  br label %latch.1

kcall.end:                                        ; preds = %header.0
  %6 = load i32, i32* @blocks_per_grid_on_fftz_2, align 4, !dbg !2975
  %dim3gep.04 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp3, i32 0, i32 0
  store i32 %6, i32* %dim3gep.04
  %dim3gep.15 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp3, i32 0, i32 1
  store i32 1, i32* %dim3gep.15
  %dim3gep.26 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp3, i32 0, i32 2
  store i32 1, i32* %dim3gep.26
  %7 = load i32, i32* @threads_per_block_on_fftz_2, align 4, !dbg !2976
  %dim3gep.07 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp4, i32 0, i32 0
  store i32 %7, i32* %dim3gep.07
  %dim3gep.18 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp4, i32 0, i32 1
  store i32 1, i32* %dim3gep.18
  %dim3gep.29 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp4, i32 0, i32 2
  store i32 1, i32* %dim3gep.29
  %8 = bitcast { i64, i32 }* %agg.tmp3.coerce to i8*, !dbg !2977
  %9 = bitcast %struct.dim3* %agg.tmp3 to i8*, !dbg !2977
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %8, i8* align 4 %9, i64 12, i1 false), !dbg !2977
  %10 = bitcast { i64, i32 }* %agg.tmp4.coerce to i8*, !dbg !2977
  %11 = bitcast %struct.dim3* %agg.tmp4 to i8*, !dbg !2977
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %10, i8* align 4 %11, i64 12, i1 false), !dbg !2977
  br label %header.016

header.016:                                       ; preds = %latch.019, %kcall.end
  %indvar.023 = phi i32 [ 0, %kcall.end ], [ %indvar.next.025, %latch.019 ]
  %exitcond38 = icmp ne i32 %indvar.023, %6
  br i1 %exitcond38, label %header.117.preheader, label %kcall.end8, !tulip.doall.loop.grid !1860

header.117.preheader:                             ; preds = %header.016
  br label %header.117

header.117:                                       ; preds = %header.117.preheader, %latch.118
  %indvar.120 = phi i32 [ %indvar.next.122, %latch.118 ], [ 0, %header.117.preheader ]
  %exitcond37 = icmp ne i32 %indvar.120, %7
  br i1 %exitcond37, label %kcall.configok7, label %latch.019, !tulip.doall.loop.block !1860

latch.118:                                        ; preds = %kcall.configok7
  %indvar.next.122 = add i32 %indvar.120, 1
  br label %header.117

latch.019:                                        ; preds = %header.117
  %indvar.next.025 = add i32 %indvar.023, 1
  br label %header.016

kcall.configok7:                                  ; preds = %header.117
  call void @cffts3_gpu_kernel_2(i32 %is, %struct.dcomplex* %y0, %struct.dcomplex* %y1, %struct.dcomplex* %u, i32 %6, i32 1, i32 1, i32 %7, i32 1, i32 1, i32 %indvar.023, i32 0, i32 0, i32 %indvar.120, i32 0, i32 0)
  br label %latch.118

kcall.end8:                                       ; preds = %header.016
  %12 = load i32, i32* @blocks_per_grid_on_fftz_3, align 4, !dbg !2978
  %dim3gep.010 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp10, i32 0, i32 0
  store i32 %12, i32* %dim3gep.010
  %dim3gep.111 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp10, i32 0, i32 1
  store i32 1, i32* %dim3gep.111
  %dim3gep.212 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp10, i32 0, i32 2
  store i32 1, i32* %dim3gep.212
  %13 = load i32, i32* @threads_per_block_on_fftz_3, align 4, !dbg !2979
  %dim3gep.013 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp11, i32 0, i32 0
  store i32 %13, i32* %dim3gep.013
  %dim3gep.114 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp11, i32 0, i32 1
  store i32 1, i32* %dim3gep.114
  %dim3gep.215 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp11, i32 0, i32 2
  store i32 1, i32* %dim3gep.215
  %14 = bitcast { i64, i32 }* %agg.tmp10.coerce to i8*, !dbg !2980
  %15 = bitcast %struct.dim3* %agg.tmp10 to i8*, !dbg !2980
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %14, i8* align 4 %15, i64 12, i1 false), !dbg !2980
  %16 = bitcast { i64, i32 }* %agg.tmp11.coerce to i8*, !dbg !2980
  %17 = bitcast %struct.dim3* %agg.tmp11 to i8*, !dbg !2980
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %16, i8* align 4 %17, i64 12, i1 false), !dbg !2980
  br label %header.026

header.026:                                       ; preds = %latch.029, %kcall.end8
  %indvar.033 = phi i32 [ 0, %kcall.end8 ], [ %indvar.next.035, %latch.029 ]
  %exitcond36 = icmp ne i32 %indvar.033, %12
  br i1 %exitcond36, label %header.127.preheader, label %kcall.end15, !tulip.doall.loop.grid !1860

header.127.preheader:                             ; preds = %header.026
  br label %header.127

header.127:                                       ; preds = %header.127.preheader, %latch.128
  %indvar.130 = phi i32 [ %indvar.next.132, %latch.128 ], [ 0, %header.127.preheader ]
  %exitcond = icmp ne i32 %indvar.130, %13
  br i1 %exitcond, label %kcall.configok14, label %latch.029, !tulip.doall.loop.block !1860

latch.128:                                        ; preds = %kcall.configok14
  %indvar.next.132 = add i32 %indvar.130, 1
  br label %header.127

latch.029:                                        ; preds = %header.127
  %indvar.next.035 = add i32 %indvar.033, 1
  br label %header.026

kcall.configok14:                                 ; preds = %header.127
  call void @cffts3_gpu_kernel_3(%struct.dcomplex* %x_out, %struct.dcomplex* %y0, i32 %12, i32 1, i32 1, i32 %13, i32 1, i32 1, i32 %indvar.033, i32 0, i32 0, i32 %indvar.130, i32 0, i32 0)
  br label %latch.128

kcall.end15:                                      ; preds = %header.026
  ret void, !dbg !2981
}

; Function Attrs: noinline nounwind uwtable
define internal i32 @_ZL5ilog2i(i32 %n) #6 !dbg !2982 {
entry:
  call void @llvm.dbg.value(metadata i32 %n, metadata !2983, metadata !DIExpression()), !dbg !2984
  %cmp = icmp eq i32 %n, 1, !dbg !2985
  br i1 %cmp, label %if.then, label %if.end, !dbg !2987

if.then:                                          ; preds = %entry
  br label %return, !dbg !2988

if.end:                                           ; preds = %entry
  call void @llvm.dbg.value(metadata i32 1, metadata !2990, metadata !DIExpression()), !dbg !2984
  call void @llvm.dbg.value(metadata i32 2, metadata !2991, metadata !DIExpression()), !dbg !2984
  br label %while.cond, !dbg !2992

while.cond:                                       ; preds = %while.body, %if.end
  %nn.0 = phi i32 [ 2, %if.end ], [ %shl, %while.body ], !dbg !2984
  %lg.0 = phi i32 [ 1, %if.end ], [ %inc, %while.body ], !dbg !2984
  call void @llvm.dbg.value(metadata i32 %lg.0, metadata !2990, metadata !DIExpression()), !dbg !2984
  call void @llvm.dbg.value(metadata i32 %nn.0, metadata !2991, metadata !DIExpression()), !dbg !2984
  %cmp1 = icmp slt i32 %nn.0, %n, !dbg !2993
  br i1 %cmp1, label %while.body, label %while.end, !dbg !2992

while.body:                                       ; preds = %while.cond
  %shl = shl i32 %nn.0, 1, !dbg !2994
  call void @llvm.dbg.value(metadata i32 %shl, metadata !2991, metadata !DIExpression()), !dbg !2984
  %inc = add nuw nsw i32 %lg.0, 1, !dbg !2996
  call void @llvm.dbg.value(metadata i32 %inc, metadata !2990, metadata !DIExpression()), !dbg !2984
  br label %while.cond, !dbg !2992, !llvm.loop !2997

while.end:                                        ; preds = %while.cond
  %lg.0.lcssa = phi i32 [ %lg.0, %while.cond ], !dbg !2984
  call void @llvm.dbg.value(metadata i32 %lg.0.lcssa, metadata !2990, metadata !DIExpression()), !dbg !2984
  br label %return, !dbg !2999

return:                                           ; preds = %while.end, %if.then
  %retval.0 = phi i32 [ 0, %if.then ], [ %lg.0.lcssa, %while.end ], !dbg !2984
  ret i32 %retval.0, !dbg !3000
}

; Function Attrs: nounwind
declare dso_local double @cos(double) #9

; Function Attrs: nounwind
declare dso_local double @sin(double) #9

; Function Attrs: noinline nounwind uwtable
define internal void @_ZL6ipow46diPd(double %a, i32 %exponent, double* %result) #6 !dbg !3001 {
entry:
  %q = alloca double, align 8
  %r = alloca double, align 8
  call void @llvm.dbg.value(metadata double %a, metadata !3002, metadata !DIExpression()), !dbg !3003
  call void @llvm.dbg.value(metadata i32 %exponent, metadata !3004, metadata !DIExpression()), !dbg !3003
  call void @llvm.dbg.value(metadata double* %result, metadata !3005, metadata !DIExpression()), !dbg !3003
  call void @llvm.dbg.declare(metadata double* %q, metadata !3006, metadata !DIExpression()), !dbg !3007
  call void @llvm.dbg.declare(metadata double* %r, metadata !3008, metadata !DIExpression()), !dbg !3009
  store double 1.000000e+00, double* %result, align 8, !dbg !3010
  %cmp = icmp eq i32 %exponent, 0, !dbg !3011
  br i1 %cmp, label %if.then, label %if.end, !dbg !3013

if.then:                                          ; preds = %entry
  br label %return, !dbg !3014

if.end:                                           ; preds = %entry
  store double %a, double* %q, align 8, !dbg !3016
  store double 1.000000e+00, double* %r, align 8, !dbg !3017
  call void @llvm.dbg.value(metadata i32 %exponent, metadata !3018, metadata !DIExpression()), !dbg !3003
  br label %while.cond, !dbg !3019

while.cond:                                       ; preds = %if.end5, %if.end
  %n.0 = phi i32 [ %exponent, %if.end ], [ %n.1, %if.end5 ], !dbg !3003
  call void @llvm.dbg.value(metadata i32 %n.0, metadata !3018, metadata !DIExpression()), !dbg !3003
  %cmp1 = icmp sgt i32 %n.0, 1, !dbg !3020
  br i1 %cmp1, label %while.body, label %while.end, !dbg !3019

while.body:                                       ; preds = %while.cond
  %div = sdiv i32 %n.0, 2, !dbg !3021
  call void @llvm.dbg.value(metadata i32 %div, metadata !3023, metadata !DIExpression()), !dbg !3003
  %mul = mul nsw i32 %div, 2, !dbg !3024
  %cmp2 = icmp eq i32 %mul, %n.0, !dbg !3026
  br i1 %cmp2, label %if.then3, label %if.else, !dbg !3027

if.then3:                                         ; preds = %while.body
  %0 = load double, double* %q, align 8, !dbg !3028
  %call = call double @_Z6randlcPdd(double* %q, double %0), !dbg !3030
  call void @llvm.dbg.value(metadata i32 %div, metadata !3018, metadata !DIExpression()), !dbg !3003
  br label %if.end5, !dbg !3031

if.else:                                          ; preds = %while.body
  %1 = load double, double* %q, align 8, !dbg !3032
  %call4 = call double @_Z6randlcPdd(double* %r, double %1), !dbg !3034
  %sub = sub nsw i32 %n.0, 1, !dbg !3035
  call void @llvm.dbg.value(metadata i32 %sub, metadata !3018, metadata !DIExpression()), !dbg !3003
  br label %if.end5

if.end5:                                          ; preds = %if.else, %if.then3
  %n.1 = phi i32 [ %div, %if.then3 ], [ %sub, %if.else ], !dbg !3036
  call void @llvm.dbg.value(metadata i32 %n.1, metadata !3018, metadata !DIExpression()), !dbg !3003
  br label %while.cond, !dbg !3019, !llvm.loop !3037

while.end:                                        ; preds = %while.cond
  %2 = load double, double* %q, align 8, !dbg !3039
  %call6 = call double @_Z6randlcPdd(double* %r, double %2), !dbg !3040
  %3 = load double, double* %r, align 8, !dbg !3041
  store double %3, double* %result, align 8, !dbg !3042
  br label %return, !dbg !3043

return:                                           ; preds = %while.end, %if.then
  ret void, !dbg !3043
}

; Function Attrs: nounwind readnone speculatable
declare double @llvm.ceil.f64(double) #0

declare dso_local void @omp_set_num_threads(i32) #8

declare double @exp(double)

; Function Attrs: convergent noinline nounwind
define dso_local void @init_ui_gpu_kernel(%struct.dcomplex* %u0, %struct.dcomplex* %u1, double* %twiddle, i32 %gridDim.x, i32 %gridDim.y, i32 %gridDim.z, i32 %blockDim.x, i32 %blockDim.y, i32 %blockDim.z, i32 %blockIdx.x, i32 %blockIdx.y, i32 %blockIdx.z, i32 %threadIdx.x, i32 %threadIdx.y, i32 %threadIdx.z) #2 {
entry:
  %ref.tmp = alloca %struct.dcomplex, align 8
  %ref.tmp3 = alloca %struct.dcomplex, align 8
  call void @llvm.dbg.value(metadata %struct.dcomplex* %u0, metadata !3044, metadata !DIExpression()), !dbg !3046
  call void @llvm.dbg.value(metadata %struct.dcomplex* %u1, metadata !3047, metadata !DIExpression()), !dbg !3046
  call void @llvm.dbg.value(metadata double* %twiddle, metadata !3048, metadata !DIExpression()), !dbg !3046
  %mul = mul i32 %blockIdx.x, %blockDim.x, !dbg !3049
  %add = add i32 %mul, %threadIdx.x, !dbg !3050
  call void @llvm.dbg.value(metadata i32 %add, metadata !3051, metadata !DIExpression()), !dbg !3046
  %cmp = icmp sge i32 %add, 8388608, !dbg !3052
  br i1 %cmp, label %if.then, label %if.end, !dbg !3054

if.then:                                          ; preds = %entry
  br label %return, !dbg !3055

if.end:                                           ; preds = %entry
  %real = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp, i32 0, i32 0, !dbg !3057
  store double 0.000000e+00, double* %real, align 8, !dbg !3057
  %imag = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp, i32 0, i32 1, !dbg !3057
  store double 0.000000e+00, double* %imag, align 8, !dbg !3057
  %idxprom = sext i32 %add to i64, !dbg !3058
  %arrayidx = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %u0, i64 %idxprom, !dbg !3058
  %0 = bitcast %struct.dcomplex* %arrayidx to i8*, !dbg !3059
  %1 = bitcast %struct.dcomplex* %ref.tmp to i8*, !dbg !3059
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %0, i8* align 8 %1, i64 16, i1 false), !dbg !3059
  %real4 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp3, i32 0, i32 0, !dbg !3060
  store double 0.000000e+00, double* %real4, align 8, !dbg !3060
  %imag5 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp3, i32 0, i32 1, !dbg !3060
  store double 0.000000e+00, double* %imag5, align 8, !dbg !3060
  %idxprom6 = sext i32 %add to i64, !dbg !3061
  %arrayidx7 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %u1, i64 %idxprom6, !dbg !3061
  %2 = bitcast %struct.dcomplex* %arrayidx7 to i8*, !dbg !3062
  %3 = bitcast %struct.dcomplex* %ref.tmp3 to i8*, !dbg !3062
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %2, i8* align 8 %3, i64 16, i1 false), !dbg !3062
  %idxprom8 = sext i32 %add to i64, !dbg !3063
  %arrayidx9 = getelementptr inbounds double, double* %twiddle, i64 %idxprom8, !dbg !3063
  store double 0.000000e+00, double* %arrayidx9, align 8, !dbg !3064
  br label %return, !dbg !3065

return:                                           ; preds = %if.end, %if.then
  ret void, !dbg !3065
}

; Function Attrs: convergent noinline nounwind
define dso_local void @compute_indexmap_gpu_kernel(double* %twiddle, i32 %gridDim.x, i32 %gridDim.y, i32 %gridDim.z, i32 %blockDim.x, i32 %blockDim.y, i32 %blockDim.z, i32 %blockIdx.x, i32 %blockIdx.y, i32 %blockIdx.z, i32 %threadIdx.x, i32 %threadIdx.y, i32 %threadIdx.z) #5 {
entry:
  call void @llvm.dbg.value(metadata double* %twiddle, metadata !3066, metadata !DIExpression()), !dbg !3068
  %mul = mul i32 %blockIdx.x, %blockDim.x, !dbg !3069
  %add = add i32 %mul, %threadIdx.x, !dbg !3070
  call void @llvm.dbg.value(metadata i32 %add, metadata !3071, metadata !DIExpression()), !dbg !3068
  %cmp = icmp sge i32 %add, 8388608, !dbg !3072
  br i1 %cmp, label %if.then, label %if.end, !dbg !3074

if.then:                                          ; preds = %entry
  br label %return, !dbg !3075

if.end:                                           ; preds = %entry
  %rem = srem i32 %add, 256, !dbg !3077
  call void @llvm.dbg.value(metadata i32 %rem, metadata !3078, metadata !DIExpression()), !dbg !3068
  %div = sdiv i32 %add, 256, !dbg !3079
  %rem3 = srem i32 %div, 256, !dbg !3080
  call void @llvm.dbg.value(metadata i32 %rem3, metadata !3081, metadata !DIExpression()), !dbg !3068
  %div4 = sdiv i32 %add, 65536, !dbg !3082
  call void @llvm.dbg.value(metadata i32 %div4, metadata !3083, metadata !DIExpression()), !dbg !3068
  %add5 = add nsw i32 %div4, 64, !dbg !3084
  %rem6 = srem i32 %add5, 128, !dbg !3085
  %sub = sub nsw i32 %rem6, 64, !dbg !3086
  call void @llvm.dbg.value(metadata i32 %sub, metadata !3087, metadata !DIExpression()), !dbg !3068
  %mul7 = mul nsw i32 %sub, %sub, !dbg !3088
  call void @llvm.dbg.value(metadata i32 %mul7, metadata !3089, metadata !DIExpression()), !dbg !3068
  %add8 = add nsw i32 %rem3, 128, !dbg !3090
  %rem9 = srem i32 %add8, 256, !dbg !3091
  %sub10 = sub nsw i32 %rem9, 128, !dbg !3092
  call void @llvm.dbg.value(metadata i32 %sub10, metadata !3093, metadata !DIExpression()), !dbg !3068
  %mul11 = mul nsw i32 %sub10, %sub10, !dbg !3094
  %add12 = add nsw i32 %mul11, %mul7, !dbg !3095
  call void @llvm.dbg.value(metadata i32 %add12, metadata !3096, metadata !DIExpression()), !dbg !3068
  %add13 = add nsw i32 %rem, 128, !dbg !3097
  %rem14 = srem i32 %add13, 256, !dbg !3098
  %sub15 = sub nsw i32 %rem14, 128, !dbg !3099
  call void @llvm.dbg.value(metadata i32 %sub15, metadata !3100, metadata !DIExpression()), !dbg !3068
  %mul16 = mul nsw i32 %sub15, %sub15, !dbg !3101
  %add17 = add nsw i32 %mul16, %add12, !dbg !3102
  %conv = sitofp i32 %add17 to double, !dbg !3103
  %mul18 = fmul contract double 0xBF04B2B4199E149A, %conv, !dbg !3104
  call void @llvm.dbg.value(metadata double %mul18, metadata !3105, metadata !DIExpression()), !dbg !3108
  %exp_result = call double @exp(double %mul18)
  br label %_ZL3expd.exit

_ZL3expd.exit:                                    ; preds = %if.end
  %idxprom = sext i32 %add to i64, !dbg !3110
  %arrayidx = getelementptr inbounds double, double* %twiddle, i64 %idxprom, !dbg !3110
  store double %exp_result, double* %arrayidx, align 8, !dbg !3111
  br label %return, !dbg !3112

return:                                           ; preds = %_ZL3expd.exit, %if.then
  ret void, !dbg !3112
}

; Function Attrs: convergent noinline nounwind
define dso_local void @compute_initial_conditions_gpu_kernel(%struct.dcomplex* %u0, double* %starts, i32 %gridDim.x, i32 %gridDim.y, i32 %gridDim.z, i32 %blockDim.x, i32 %blockDim.y, i32 %blockDim.z, i32 %blockIdx.x, i32 %blockIdx.y, i32 %blockIdx.z, i32 %threadIdx.x, i32 %threadIdx.y, i32 %threadIdx.z) #2 {
entry:
  %x0 = alloca double, align 8
  call void @llvm.dbg.value(metadata %struct.dcomplex* %u0, metadata !3113, metadata !DIExpression()), !dbg !3117
  call void @llvm.dbg.value(metadata double* %starts, metadata !3118, metadata !DIExpression()), !dbg !3117
  %mul = mul i32 %blockIdx.x, %blockDim.x, !dbg !3119
  %add = add i32 %mul, %threadIdx.x, !dbg !3120
  call void @llvm.dbg.value(metadata i32 %add, metadata !3121, metadata !DIExpression()), !dbg !3117
  %cmp = icmp sge i32 %add, 128, !dbg !3122
  br i1 %cmp, label %if.then, label %if.end, !dbg !3124

if.then:                                          ; preds = %entry
  br label %for.end, !dbg !3125

if.end:                                           ; preds = %entry
  call void @llvm.dbg.declare(metadata double* %x0, metadata !3127, metadata !DIExpression()), !dbg !3128
  %idxprom = sext i32 %add to i64, !dbg !3129
  %arrayidx = getelementptr inbounds double, double* %starts, i64 %idxprom, !dbg !3129
  %0 = load double, double* %arrayidx, align 8, !dbg !3129
  store double %0, double* %x0, align 8, !dbg !3128
  call void @llvm.dbg.value(metadata i32 0, metadata !3130, metadata !DIExpression()), !dbg !3132
  br label %for.cond, !dbg !3133

for.cond:                                         ; preds = %for.inc, %if.end
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %if.end ], !dbg !3132
  call void @llvm.dbg.value(metadata i64 %indvars.iv, metadata !3130, metadata !DIExpression()), !dbg !3132
  %exitcond = icmp ne i64 %indvars.iv, 256, !dbg !3134
  br i1 %exitcond, label %for.body, label %for.end.loopexit, !dbg !3136

for.body:                                         ; preds = %for.cond
  %1 = mul nuw nsw i64 %indvars.iv, 256, !dbg !3137
  %mul6 = mul nsw i32 %add, 256, !dbg !3139
  %mul7 = mul nsw i32 %mul6, 256, !dbg !3140
  %2 = sext i32 %mul7 to i64, !dbg !3141
  %3 = add nsw i64 %1, %2, !dbg !3141
  %arrayidx10 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %u0, i64 %3, !dbg !3142
  %4 = bitcast %struct.dcomplex* %arrayidx10 to double*, !dbg !3143
  call void @_Z13vranlc_deviceiPddS_(i32 512, double* %x0, double 0x41D2309CE5400000, double* %4) #4, !dbg !3144
  br label %for.inc, !dbg !3145

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !3146
  call void @llvm.dbg.value(metadata i32 undef, metadata !3130, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !3132
  br label %for.cond, !dbg !3147, !llvm.loop !3148

for.end.loopexit:                                 ; preds = %for.cond
  br label %for.end, !dbg !3150

for.end:                                          ; preds = %for.end.loopexit, %if.then
  ret void, !dbg !3150
}

; Function Attrs: convergent noinline nounwind
define dso_local void @evolve_gpu_kernel(%struct.dcomplex* %u0, %struct.dcomplex* %u1, double* %twiddle, i32 %gridDim.x, i32 %gridDim.y, i32 %gridDim.z, i32 %blockDim.x, i32 %blockDim.y, i32 %blockDim.z, i32 %blockIdx.x, i32 %blockIdx.y, i32 %blockIdx.z, i32 %threadIdx.x, i32 %threadIdx.y, i32 %threadIdx.z) #2 {
entry:
  %ref.tmp = alloca %struct.dcomplex, align 8
  call void @llvm.dbg.value(metadata %struct.dcomplex* %u0, metadata !3151, metadata !DIExpression()), !dbg !3153
  call void @llvm.dbg.value(metadata %struct.dcomplex* %u1, metadata !3154, metadata !DIExpression()), !dbg !3153
  call void @llvm.dbg.value(metadata double* %twiddle, metadata !3155, metadata !DIExpression()), !dbg !3153
  %mul = mul i32 %blockIdx.x, %blockDim.x, !dbg !3156
  %add = add i32 %mul, %threadIdx.x, !dbg !3157
  call void @llvm.dbg.value(metadata i32 %add, metadata !3158, metadata !DIExpression()), !dbg !3153
  %cmp = icmp sge i32 %add, 8388608, !dbg !3159
  br i1 %cmp, label %if.then, label %if.end, !dbg !3161

if.then:                                          ; preds = %entry
  br label %return, !dbg !3162

if.end:                                           ; preds = %entry
  %real = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp, i32 0, i32 0, !dbg !3164
  %idxprom = sext i32 %add to i64, !dbg !3164
  %arrayidx = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %u0, i64 %idxprom, !dbg !3164
  %real3 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx, i32 0, i32 0, !dbg !3164
  %0 = load double, double* %real3, align 8, !dbg !3164
  %idxprom4 = sext i32 %add to i64, !dbg !3164
  %arrayidx5 = getelementptr inbounds double, double* %twiddle, i64 %idxprom4, !dbg !3164
  %1 = load double, double* %arrayidx5, align 8, !dbg !3164
  %mul6 = fmul contract double %0, %1, !dbg !3164
  store double %mul6, double* %real, align 8, !dbg !3164
  %imag = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp, i32 0, i32 1, !dbg !3164
  %idxprom7 = sext i32 %add to i64, !dbg !3164
  %arrayidx8 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %u0, i64 %idxprom7, !dbg !3164
  %imag9 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx8, i32 0, i32 1, !dbg !3164
  %2 = load double, double* %imag9, align 8, !dbg !3164
  %idxprom10 = sext i32 %add to i64, !dbg !3164
  %arrayidx11 = getelementptr inbounds double, double* %twiddle, i64 %idxprom10, !dbg !3164
  %3 = load double, double* %arrayidx11, align 8, !dbg !3164
  %mul12 = fmul contract double %2, %3, !dbg !3164
  store double %mul12, double* %imag, align 8, !dbg !3164
  %idxprom13 = sext i32 %add to i64, !dbg !3165
  %arrayidx14 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %u0, i64 %idxprom13, !dbg !3165
  %4 = bitcast %struct.dcomplex* %arrayidx14 to i8*, !dbg !3166
  %5 = bitcast %struct.dcomplex* %ref.tmp to i8*, !dbg !3166
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %4, i8* align 8 %5, i64 16, i1 false), !dbg !3166
  %idxprom15 = sext i32 %add to i64, !dbg !3167
  %arrayidx16 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %u0, i64 %idxprom15, !dbg !3167
  %idxprom17 = sext i32 %add to i64, !dbg !3168
  %arrayidx18 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %u1, i64 %idxprom17, !dbg !3168
  %6 = bitcast %struct.dcomplex* %arrayidx18 to i8*, !dbg !3169
  %7 = bitcast %struct.dcomplex* %arrayidx16 to i8*, !dbg !3169
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %6, i8* align 8 %7, i64 16, i1 false), !dbg !3169
  br label %return, !dbg !3170

return:                                           ; preds = %if.end, %if.then
  ret void, !dbg !3170
}

; Function Attrs: convergent noinline nounwind
define dso_local void @checksum_gpu_kernel0(i32 %iteration, %struct.dcomplex* %u1, %struct.dcomplex* %sums, i32 %gridDim.x, i32 %gridDim.y, i32 %gridDim.z, i32 %blockDim.x, i32 %blockDim.y, i32 %blockDim.z, i32 %blockIdx.x, i32 %blockIdx.y, i32 %blockIdx.z, i32 %threadIdx.x, i32 %threadIdx.y, i32 %threadIdx.z) #2 {
entry:
  %ref.tmp = alloca %struct.dcomplex, align 8
  call void @llvm.dbg.value(metadata i32 %iteration, metadata !3171, metadata !DIExpression()), !dbg !3173
  call void @llvm.dbg.value(metadata %struct.dcomplex* %u1, metadata !3174, metadata !DIExpression()), !dbg !3173
  call void @llvm.dbg.value(metadata %struct.dcomplex* %sums, metadata !3175, metadata !DIExpression()), !dbg !3173
  call void @llvm.dbg.value(metadata %struct.dcomplex* bitcast ([1024 x double]* @extern_share_data_shared to %struct.dcomplex*), metadata !3176, metadata !DIExpression()), !dbg !3173
  %mul = mul i32 %blockIdx.x, %blockDim.x, !dbg !3177
  %add = add i32 %mul, %threadIdx.x, !dbg !3178
  %add3 = add i32 %add, 1, !dbg !3179
  call void @llvm.dbg.value(metadata i32 %add3, metadata !3180, metadata !DIExpression()), !dbg !3173
  %cmp = icmp sle i32 %add3, 1024, !dbg !3181
  br i1 %cmp, label %if.then, label %if.else, !dbg !3183

if.then:                                          ; preds = %entry
  %rem = srem i32 %add3, 256, !dbg !3184
  call void @llvm.dbg.value(metadata i32 %rem, metadata !3186, metadata !DIExpression()), !dbg !3173
  %mul4 = mul nsw i32 3, %add3, !dbg !3187
  %rem5 = srem i32 %mul4, 256, !dbg !3188
  call void @llvm.dbg.value(metadata i32 %rem5, metadata !3189, metadata !DIExpression()), !dbg !3173
  %mul6 = mul nsw i32 5, %add3, !dbg !3190
  %rem7 = srem i32 %mul6, 128, !dbg !3191
  call void @llvm.dbg.value(metadata i32 %rem7, metadata !3192, metadata !DIExpression()), !dbg !3173
  %mul8 = mul nsw i32 %rem5, 256, !dbg !3193
  %add9 = add nsw i32 %rem, %mul8, !dbg !3194
  %mul10 = mul nsw i32 %rem7, 256, !dbg !3195
  %mul11 = mul nsw i32 %mul10, 256, !dbg !3196
  %add12 = add nsw i32 %add9, %mul11, !dbg !3197
  %idxprom = sext i32 %add12 to i64, !dbg !3198
  %arrayidx = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %u1, i64 %idxprom, !dbg !3198
  %idxprom14 = zext i32 %threadIdx.x to i64, !dbg !3199
  %arrayidx15 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* bitcast ([1024 x double]* @extern_share_data_shared to %struct.dcomplex*), i64 %idxprom14, !dbg !3199
  %0 = bitcast %struct.dcomplex* %arrayidx15 to i8*, !dbg !3200
  %1 = bitcast %struct.dcomplex* %arrayidx to i8*, !dbg !3200
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %0, i8* align 8 %1, i64 16, i1 false), !dbg !3200
  br label %if.end, !dbg !3201

if.else:                                          ; preds = %entry
  %real = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp, i32 0, i32 0, !dbg !3202
  store double 0.000000e+00, double* %real, align 8, !dbg !3202
  %imag = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp, i32 0, i32 1, !dbg !3202
  store double 0.000000e+00, double* %imag, align 8, !dbg !3202
  %idxprom17 = zext i32 %threadIdx.x to i64, !dbg !3204
  %arrayidx18 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* bitcast ([1024 x double]* @extern_share_data_shared to %struct.dcomplex*), i64 %idxprom17, !dbg !3204
  %2 = bitcast %struct.dcomplex* %arrayidx18 to i8*, !dbg !3205
  %3 = bitcast %struct.dcomplex* %ref.tmp to i8*, !dbg !3205
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %2, i8* align 8 %3, i64 16, i1 false), !dbg !3205
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void, !dbg !3206
}

; Function Attrs: convergent noinline nounwind
define dso_local void @cffts1_gpu_kernel_1(%struct.dcomplex* %x_in, %struct.dcomplex* %y0, i32 %gridDim.x, i32 %gridDim.y, i32 %gridDim.z, i32 %blockDim.x, i32 %blockDim.y, i32 %blockDim.z, i32 %blockIdx.x, i32 %blockIdx.y, i32 %blockIdx.z, i32 %threadIdx.x, i32 %threadIdx.y, i32 %threadIdx.z) #2 {
entry:
  call void @llvm.dbg.value(metadata %struct.dcomplex* %x_in, metadata !3207, metadata !DIExpression()), !dbg !3211
  call void @llvm.dbg.value(metadata %struct.dcomplex* %y0, metadata !3212, metadata !DIExpression()), !dbg !3211
  %mul = mul i32 %blockIdx.x, %blockDim.x, !dbg !3213
  %add = add i32 %mul, %threadIdx.x, !dbg !3214
  call void @llvm.dbg.value(metadata i32 %add, metadata !3215, metadata !DIExpression()), !dbg !3211
  %cmp = icmp sge i32 %add, 8388608, !dbg !3216
  br i1 %cmp, label %if.then, label %if.end, !dbg !3218

if.then:                                          ; preds = %entry
  br label %return, !dbg !3219

if.end:                                           ; preds = %entry
  %rem = srem i32 %add, 256, !dbg !3221
  call void @llvm.dbg.value(metadata i32 %rem, metadata !3222, metadata !DIExpression()), !dbg !3211
  %div = sdiv i32 %add, 256, !dbg !3223
  %rem3 = srem i32 %div, 256, !dbg !3224
  call void @llvm.dbg.value(metadata i32 %rem3, metadata !3225, metadata !DIExpression()), !dbg !3211
  %div4 = sdiv i32 %add, 65536, !dbg !3226
  call void @llvm.dbg.value(metadata i32 %div4, metadata !3227, metadata !DIExpression()), !dbg !3211
  %idxprom = sext i32 %add to i64, !dbg !3228
  %arrayidx = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %x_in, i64 %idxprom, !dbg !3228
  %real = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx, i32 0, i32 0, !dbg !3229
  %0 = load double, double* %real, align 8, !dbg !3229
  %mul5 = mul nsw i32 %rem, 256, !dbg !3230
  %add6 = add nsw i32 %rem3, %mul5, !dbg !3231
  %mul7 = mul nsw i32 %div4, 256, !dbg !3232
  %mul8 = mul nsw i32 %mul7, 256, !dbg !3233
  %add9 = add nsw i32 %add6, %mul8, !dbg !3234
  %idxprom10 = sext i32 %add9 to i64, !dbg !3235
  %arrayidx11 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %y0, i64 %idxprom10, !dbg !3235
  %real12 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx11, i32 0, i32 0, !dbg !3236
  store double %0, double* %real12, align 8, !dbg !3237
  %idxprom13 = sext i32 %add to i64, !dbg !3238
  %arrayidx14 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %x_in, i64 %idxprom13, !dbg !3238
  %imag = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx14, i32 0, i32 1, !dbg !3239
  %1 = load double, double* %imag, align 8, !dbg !3239
  %mul15 = mul nsw i32 %rem, 256, !dbg !3240
  %add16 = add nsw i32 %rem3, %mul15, !dbg !3241
  %mul17 = mul nsw i32 %div4, 256, !dbg !3242
  %mul18 = mul nsw i32 %mul17, 256, !dbg !3243
  %add19 = add nsw i32 %add16, %mul18, !dbg !3244
  %idxprom20 = sext i32 %add19 to i64, !dbg !3245
  %arrayidx21 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %y0, i64 %idxprom20, !dbg !3245
  %imag22 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx21, i32 0, i32 1, !dbg !3246
  store double %1, double* %imag22, align 8, !dbg !3247
  br label %return, !dbg !3248

return:                                           ; preds = %if.end, %if.then
  ret void, !dbg !3248
}

; Function Attrs: convergent noinline nounwind
define dso_local void @cffts1_gpu_kernel_2(i32 %is, %struct.dcomplex* %gty1, %struct.dcomplex* %gty2, %struct.dcomplex* %u_device, i32 %gridDim.x, i32 %gridDim.y, i32 %gridDim.z, i32 %blockDim.x, i32 %blockDim.y, i32 %blockDim.z, i32 %blockIdx.x, i32 %blockIdx.y, i32 %blockIdx.z, i32 %threadIdx.x, i32 %threadIdx.y, i32 %threadIdx.z) #2 {
entry:
  call void @llvm.dbg.value(metadata i32 %is, metadata !3249, metadata !DIExpression()), !dbg !3253
  call void @llvm.dbg.value(metadata %struct.dcomplex* %gty1, metadata !3254, metadata !DIExpression()), !dbg !3253
  call void @llvm.dbg.value(metadata %struct.dcomplex* %gty2, metadata !3255, metadata !DIExpression()), !dbg !3253
  call void @llvm.dbg.value(metadata %struct.dcomplex* %u_device, metadata !3256, metadata !DIExpression()), !dbg !3253
  %mul = mul i32 %blockIdx.x, %blockDim.x, !dbg !3257
  %add = add i32 %mul, %threadIdx.x, !dbg !3258
  call void @llvm.dbg.value(metadata i32 %add, metadata !3259, metadata !DIExpression()), !dbg !3253
  %cmp = icmp sge i32 %add, 32768, !dbg !3260
  br i1 %cmp, label %if.then, label %if.end, !dbg !3262

if.then:                                          ; preds = %entry
  br label %for.end271, !dbg !3263

if.end:                                           ; preds = %entry
  %rem = srem i32 %add, 256, !dbg !3265
  call void @llvm.dbg.value(metadata i32 %rem, metadata !3266, metadata !DIExpression()), !dbg !3253
  %div = sdiv i32 %add, 256, !dbg !3267
  %rem3 = srem i32 %div, 128, !dbg !3268
  call void @llvm.dbg.value(metadata i32 %rem3, metadata !3269, metadata !DIExpression()), !dbg !3253
  %call4 = call i32 @_Z12ilog2_devicei(i32 256) #4, !dbg !3270
  call void @llvm.dbg.value(metadata i32 %call4, metadata !3271, metadata !DIExpression()), !dbg !3253
  call void @llvm.dbg.value(metadata i32 1, metadata !3272, metadata !DIExpression()), !dbg !3253
  %0 = sext i32 %rem to i64, !dbg !3273
  %1 = sext i32 %rem to i64, !dbg !3273
  %2 = sext i32 %rem to i64, !dbg !3273
  %3 = sext i32 %rem to i64, !dbg !3273
  %4 = sext i32 %rem to i64, !dbg !3273
  %5 = sext i32 %rem to i64, !dbg !3273
  %6 = sext i32 %rem to i64, !dbg !3273
  %7 = sext i32 %rem to i64, !dbg !3273
  %8 = sext i32 %rem to i64, !dbg !3273
  %9 = sext i32 %rem to i64, !dbg !3273
  %10 = sext i32 %rem to i64, !dbg !3273
  %11 = sext i32 %rem to i64, !dbg !3273
  %12 = sext i32 %rem to i64, !dbg !3273
  %13 = sext i32 %rem to i64, !dbg !3273
  %14 = sext i32 %rem to i64, !dbg !3273
  %15 = sext i32 %rem to i64, !dbg !3273
  %16 = sext i32 %rem to i64, !dbg !3273
  %17 = sext i32 %rem to i64, !dbg !3273
  %18 = sext i32 %rem to i64, !dbg !3273
  %19 = sext i32 %rem to i64, !dbg !3273
  br label %for.cond, !dbg !3273

for.cond:                                         ; preds = %for.inc269, %if.end
  %l.0 = phi i32 [ 1, %if.end ], [ %add270, %for.inc269 ], !dbg !3275
  call void @llvm.dbg.value(metadata i32 %l.0, metadata !3272, metadata !DIExpression()), !dbg !3253
  %cmp5 = icmp sle i32 %l.0, %call4, !dbg !3276
  br i1 %cmp5, label %for.body, label %for.end271.loopexit, !dbg !3278

for.body:                                         ; preds = %for.cond
  call void @llvm.dbg.value(metadata i32 128, metadata !3279, metadata !DIExpression()), !dbg !3253
  %sub = sub nuw nsw i32 %l.0, 1, !dbg !3280
  %shl = shl i32 1, %sub, !dbg !3282
  call void @llvm.dbg.value(metadata i32 %shl, metadata !3283, metadata !DIExpression()), !dbg !3253
  %sub6 = sub nsw i32 %call4, %l.0, !dbg !3284
  %shl7 = shl i32 1, %sub6, !dbg !3285
  call void @llvm.dbg.value(metadata i32 %shl7, metadata !3286, metadata !DIExpression()), !dbg !3253
  %mul8 = mul nsw i32 2, %shl, !dbg !3287
  call void @llvm.dbg.value(metadata i32 %mul8, metadata !3288, metadata !DIExpression()), !dbg !3253
  call void @llvm.dbg.value(metadata i32 %shl7, metadata !3289, metadata !DIExpression()), !dbg !3253
  call void @llvm.dbg.value(metadata i32 0, metadata !3290, metadata !DIExpression()), !dbg !3253
  %20 = sext i32 %shl to i64, !dbg !3291
  %21 = sext i32 %mul8 to i64, !dbg !3291
  %22 = sext i32 %shl to i64, !dbg !3291
  %23 = sext i32 %shl7 to i64, !dbg !3291
  %24 = sext i32 %shl7 to i64, !dbg !3291
  br label %for.cond9, !dbg !3291

for.cond9:                                        ; preds = %for.inc108, %for.body
  %indvars.iv33 = phi i64 [ %indvars.iv.next34, %for.inc108 ], [ 0, %for.body ], !dbg !3293
  call void @llvm.dbg.value(metadata i64 %indvars.iv33, metadata !3290, metadata !DIExpression()), !dbg !3253
  %sub10 = sub nsw i32 %shl7, 1, !dbg !3294
  %25 = sext i32 %sub10 to i64, !dbg !3296
  %cmp11 = icmp sle i64 %indvars.iv33, %25, !dbg !3296
  br i1 %cmp11, label %for.body12, label %for.end110, !dbg !3297

for.body12:                                       ; preds = %for.cond9
  call void @llvm.dbg.value(metadata i32 0, metadata !3298, metadata !DIExpression()), !dbg !3253
  br label %for.cond13, !dbg !3299

for.cond13:                                       ; preds = %for.inc, %for.body12
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %for.body12 ], !dbg !3302
  call void @llvm.dbg.value(metadata i64 %indvars.iv, metadata !3298, metadata !DIExpression()), !dbg !3253
  %sub14 = sub nsw i32 %shl, 1, !dbg !3303
  %26 = sext i32 %sub14 to i64, !dbg !3305
  %cmp15 = icmp sle i64 %indvars.iv, %26, !dbg !3305
  br i1 %cmp15, label %for.body16, label %for.end, !dbg !3306

for.body16:                                       ; preds = %for.cond13
  %27 = mul nsw i64 %indvars.iv33, %20, !dbg !3307
  %28 = add nsw i64 %27, 128, !dbg !3309
  %29 = mul nsw i64 %indvars.iv33, %21, !dbg !3310
  %30 = add nsw i64 %29, %22, !dbg !3311
  %31 = add nsw i64 %23, %indvars.iv33, !dbg !3312
  %arrayidx = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %u_device, i64 %31, !dbg !3313
  %real = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx, i32 0, i32 0, !dbg !3314
  %32 = load double, double* %real, align 8, !dbg !3314
  call void @llvm.dbg.value(metadata double %32, metadata !3315, metadata !DIExpression()), !dbg !3253
  %conv = sitofp i32 %is to double, !dbg !3316
  %33 = add nsw i64 %24, %indvars.iv33, !dbg !3317
  %arrayidx24 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %u_device, i64 %33, !dbg !3318
  %imag = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx24, i32 0, i32 1, !dbg !3319
  %34 = load double, double* %imag, align 8, !dbg !3319
  %mul25 = fmul contract double %conv, %34, !dbg !3320
  call void @llvm.dbg.value(metadata double %mul25, metadata !3321, metadata !DIExpression()), !dbg !3253
  %35 = add nsw i64 %27, %indvars.iv, !dbg !3322
  %36 = mul nsw i64 %35, 256, !dbg !3323
  %37 = add nsw i64 %0, %36, !dbg !3324
  %mul29 = mul nsw i32 %rem3, 256, !dbg !3325
  %mul30 = mul nsw i32 %mul29, 256, !dbg !3326
  %38 = sext i32 %mul30 to i64, !dbg !3327
  %39 = add nsw i64 %37, %38, !dbg !3327
  %arrayidx33 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %gty1, i64 %39, !dbg !3328
  %real34 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx33, i32 0, i32 0, !dbg !3329
  %40 = load double, double* %real34, align 8, !dbg !3329
  call void @llvm.dbg.value(metadata double %40, metadata !3330, metadata !DIExpression()), !dbg !3253
  %41 = add nsw i64 %27, %indvars.iv, !dbg !3331
  %42 = mul nsw i64 %41, 256, !dbg !3332
  %43 = add nsw i64 %1, %42, !dbg !3333
  %mul38 = mul nsw i32 %rem3, 256, !dbg !3334
  %mul39 = mul nsw i32 %mul38, 256, !dbg !3335
  %44 = sext i32 %mul39 to i64, !dbg !3336
  %45 = add nsw i64 %43, %44, !dbg !3336
  %arrayidx42 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %gty1, i64 %45, !dbg !3337
  %imag43 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx42, i32 0, i32 1, !dbg !3338
  %46 = load double, double* %imag43, align 8, !dbg !3338
  call void @llvm.dbg.value(metadata double %46, metadata !3339, metadata !DIExpression()), !dbg !3253
  %47 = add nsw i64 %28, %indvars.iv, !dbg !3340
  %48 = mul nsw i64 %47, 256, !dbg !3341
  %49 = add nsw i64 %2, %48, !dbg !3342
  %mul47 = mul nsw i32 %rem3, 256, !dbg !3343
  %mul48 = mul nsw i32 %mul47, 256, !dbg !3344
  %50 = sext i32 %mul48 to i64, !dbg !3345
  %51 = add nsw i64 %49, %50, !dbg !3345
  %arrayidx51 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %gty1, i64 %51, !dbg !3346
  %real52 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx51, i32 0, i32 0, !dbg !3347
  %52 = load double, double* %real52, align 8, !dbg !3347
  call void @llvm.dbg.value(metadata double %52, metadata !3348, metadata !DIExpression()), !dbg !3253
  %53 = add nsw i64 %28, %indvars.iv, !dbg !3349
  %54 = mul nsw i64 %53, 256, !dbg !3350
  %55 = add nsw i64 %3, %54, !dbg !3351
  %mul56 = mul nsw i32 %rem3, 256, !dbg !3352
  %mul57 = mul nsw i32 %mul56, 256, !dbg !3353
  %56 = sext i32 %mul57 to i64, !dbg !3354
  %57 = add nsw i64 %55, %56, !dbg !3354
  %arrayidx60 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %gty1, i64 %57, !dbg !3355
  %imag61 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx60, i32 0, i32 1, !dbg !3356
  %58 = load double, double* %imag61, align 8, !dbg !3356
  call void @llvm.dbg.value(metadata double %58, metadata !3357, metadata !DIExpression()), !dbg !3253
  %add62 = fadd contract double %40, %52, !dbg !3358
  %59 = add nsw i64 %29, %indvars.iv, !dbg !3359
  %60 = mul nsw i64 %59, 256, !dbg !3360
  %61 = add nsw i64 %4, %60, !dbg !3361
  %mul66 = mul nsw i32 %rem3, 256, !dbg !3362
  %mul67 = mul nsw i32 %mul66, 256, !dbg !3363
  %62 = sext i32 %mul67 to i64, !dbg !3364
  %63 = add nsw i64 %61, %62, !dbg !3364
  %arrayidx70 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %gty2, i64 %63, !dbg !3365
  %real71 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx70, i32 0, i32 0, !dbg !3366
  store double %add62, double* %real71, align 8, !dbg !3367
  %add72 = fadd contract double %46, %58, !dbg !3368
  %64 = add nsw i64 %29, %indvars.iv, !dbg !3369
  %65 = mul nsw i64 %64, 256, !dbg !3370
  %66 = add nsw i64 %5, %65, !dbg !3371
  %mul76 = mul nsw i32 %rem3, 256, !dbg !3372
  %mul77 = mul nsw i32 %mul76, 256, !dbg !3373
  %67 = sext i32 %mul77 to i64, !dbg !3374
  %68 = add nsw i64 %66, %67, !dbg !3374
  %arrayidx80 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %gty2, i64 %68, !dbg !3375
  %imag81 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx80, i32 0, i32 1, !dbg !3376
  store double %add72, double* %imag81, align 8, !dbg !3377
  %sub82 = fsub contract double %40, %52, !dbg !3378
  call void @llvm.dbg.value(metadata double %sub82, metadata !3379, metadata !DIExpression()), !dbg !3253
  %sub83 = fsub contract double %46, %58, !dbg !3380
  call void @llvm.dbg.value(metadata double %sub83, metadata !3381, metadata !DIExpression()), !dbg !3253
  %mul84 = fmul contract double %32, %sub82, !dbg !3382
  %mul85 = fmul contract double %mul25, %sub83, !dbg !3383
  %sub86 = fsub contract double %mul84, %mul85, !dbg !3384
  %69 = add nsw i64 %30, %indvars.iv, !dbg !3385
  %70 = mul nsw i64 %69, 256, !dbg !3386
  %71 = add nsw i64 %6, %70, !dbg !3387
  %mul90 = mul nsw i32 %rem3, 256, !dbg !3388
  %mul91 = mul nsw i32 %mul90, 256, !dbg !3389
  %72 = sext i32 %mul91 to i64, !dbg !3390
  %73 = add nsw i64 %71, %72, !dbg !3390
  %arrayidx94 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %gty2, i64 %73, !dbg !3391
  %real95 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx94, i32 0, i32 0, !dbg !3392
  store double %sub86, double* %real95, align 8, !dbg !3393
  %mul96 = fmul contract double %32, %sub83, !dbg !3394
  %mul97 = fmul contract double %mul25, %sub82, !dbg !3395
  %add98 = fadd contract double %mul96, %mul97, !dbg !3396
  %74 = add nsw i64 %30, %indvars.iv, !dbg !3397
  %75 = mul nsw i64 %74, 256, !dbg !3398
  %76 = add nsw i64 %7, %75, !dbg !3399
  %mul102 = mul nsw i32 %rem3, 256, !dbg !3400
  %mul103 = mul nsw i32 %mul102, 256, !dbg !3401
  %77 = sext i32 %mul103 to i64, !dbg !3402
  %78 = add nsw i64 %76, %77, !dbg !3402
  %arrayidx106 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %gty2, i64 %78, !dbg !3403
  %imag107 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx106, i32 0, i32 1, !dbg !3404
  store double %add98, double* %imag107, align 8, !dbg !3405
  br label %for.inc, !dbg !3406

for.inc:                                          ; preds = %for.body16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !3407
  call void @llvm.dbg.value(metadata i32 undef, metadata !3298, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !3253
  br label %for.cond13, !dbg !3408, !llvm.loop !3409

for.end:                                          ; preds = %for.cond13
  br label %for.inc108, !dbg !3411

for.inc108:                                       ; preds = %for.end
  %indvars.iv.next34 = add nuw nsw i64 %indvars.iv33, 1, !dbg !3412
  call void @llvm.dbg.value(metadata i32 undef, metadata !3290, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !3253
  br label %for.cond9, !dbg !3413, !llvm.loop !3414

for.end110:                                       ; preds = %for.cond9
  %cmp111 = icmp eq i32 %l.0, %call4, !dbg !3416
  br i1 %cmp111, label %if.then112, label %if.else, !dbg !3418

if.then112:                                       ; preds = %for.end110
  call void @llvm.dbg.value(metadata i32 0, metadata !3419, metadata !DIExpression()), !dbg !3253
  br label %for.cond113, !dbg !3420

for.cond113:                                      ; preds = %for.inc148, %if.then112
  %indvars.iv83 = phi i64 [ %indvars.iv.next84, %for.inc148 ], [ 0, %if.then112 ], !dbg !3423
  call void @llvm.dbg.value(metadata i64 %indvars.iv83, metadata !3419, metadata !DIExpression()), !dbg !3253
  %exitcond = icmp ne i64 %indvars.iv83, 256, !dbg !3424
  br i1 %exitcond, label %for.body115, label %for.end150, !dbg !3426

for.body115:                                      ; preds = %for.cond113
  %79 = mul nuw nsw i64 %indvars.iv83, 256, !dbg !3427
  %80 = add nsw i64 %16, %79, !dbg !3429
  %mul118 = mul nsw i32 %rem3, 256, !dbg !3430
  %mul119 = mul nsw i32 %mul118, 256, !dbg !3431
  %81 = sext i32 %mul119 to i64, !dbg !3432
  %82 = add nsw i64 %80, %81, !dbg !3432
  %arrayidx122 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %gty2, i64 %82, !dbg !3433
  %real123 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx122, i32 0, i32 0, !dbg !3434
  %83 = load double, double* %real123, align 8, !dbg !3434
  %84 = mul nuw nsw i64 %indvars.iv83, 256, !dbg !3435
  %85 = add nsw i64 %17, %84, !dbg !3436
  %mul126 = mul nsw i32 %rem3, 256, !dbg !3437
  %mul127 = mul nsw i32 %mul126, 256, !dbg !3438
  %86 = sext i32 %mul127 to i64, !dbg !3439
  %87 = add nsw i64 %85, %86, !dbg !3439
  %arrayidx130 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %gty1, i64 %87, !dbg !3440
  %real131 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx130, i32 0, i32 0, !dbg !3441
  store double %83, double* %real131, align 8, !dbg !3442
  %88 = mul nuw nsw i64 %indvars.iv83, 256, !dbg !3443
  %89 = add nsw i64 %18, %88, !dbg !3444
  %mul134 = mul nsw i32 %rem3, 256, !dbg !3445
  %mul135 = mul nsw i32 %mul134, 256, !dbg !3446
  %90 = sext i32 %mul135 to i64, !dbg !3447
  %91 = add nsw i64 %89, %90, !dbg !3447
  %arrayidx138 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %gty2, i64 %91, !dbg !3448
  %imag139 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx138, i32 0, i32 1, !dbg !3449
  %92 = load double, double* %imag139, align 8, !dbg !3449
  %93 = mul nuw nsw i64 %indvars.iv83, 256, !dbg !3450
  %94 = add nsw i64 %19, %93, !dbg !3451
  %mul142 = mul nsw i32 %rem3, 256, !dbg !3452
  %mul143 = mul nsw i32 %mul142, 256, !dbg !3453
  %95 = sext i32 %mul143 to i64, !dbg !3454
  %96 = add nsw i64 %94, %95, !dbg !3454
  %arrayidx146 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %gty1, i64 %96, !dbg !3455
  %imag147 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx146, i32 0, i32 1, !dbg !3456
  store double %92, double* %imag147, align 8, !dbg !3457
  br label %for.inc148, !dbg !3458

for.inc148:                                       ; preds = %for.body115
  %indvars.iv.next84 = add nuw nsw i64 %indvars.iv83, 1, !dbg !3459
  call void @llvm.dbg.value(metadata i32 undef, metadata !3419, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !3253
  br label %for.cond113, !dbg !3460, !llvm.loop !3461

for.end150:                                       ; preds = %for.cond113
  br label %if.end268, !dbg !3463

if.else:                                          ; preds = %for.end110
  call void @llvm.dbg.value(metadata i32 128, metadata !3279, metadata !DIExpression()), !dbg !3253
  %add151 = add nuw nsw i32 %l.0, 1, !dbg !3464
  %sub152 = sub nuw nsw i32 %add151, 1, !dbg !3466
  %shl153 = shl i32 1, %sub152, !dbg !3467
  call void @llvm.dbg.value(metadata i32 %shl153, metadata !3283, metadata !DIExpression()), !dbg !3253
  %add154 = add nuw nsw i32 %l.0, 1, !dbg !3468
  %sub155 = sub nsw i32 %call4, %add154, !dbg !3469
  %shl156 = shl i32 1, %sub155, !dbg !3470
  call void @llvm.dbg.value(metadata i32 %shl156, metadata !3286, metadata !DIExpression()), !dbg !3253
  %mul157 = mul nsw i32 2, %shl153, !dbg !3471
  call void @llvm.dbg.value(metadata i32 %mul157, metadata !3288, metadata !DIExpression()), !dbg !3253
  call void @llvm.dbg.value(metadata i32 %shl156, metadata !3289, metadata !DIExpression()), !dbg !3253
  call void @llvm.dbg.value(metadata i32 0, metadata !3290, metadata !DIExpression()), !dbg !3253
  %97 = sext i32 %shl153 to i64, !dbg !3472
  %98 = sext i32 %mul157 to i64, !dbg !3472
  %99 = sext i32 %shl153 to i64, !dbg !3472
  %100 = sext i32 %shl156 to i64, !dbg !3472
  %101 = sext i32 %shl156 to i64, !dbg !3472
  br label %for.cond158, !dbg !3472

for.cond158:                                      ; preds = %for.inc265, %if.else
  %indvars.iv75 = phi i64 [ %indvars.iv.next76, %for.inc265 ], [ 0, %if.else ], !dbg !3474
  call void @llvm.dbg.value(metadata i64 %indvars.iv75, metadata !3290, metadata !DIExpression()), !dbg !3253
  %sub159 = sub nsw i32 %shl156, 1, !dbg !3475
  %102 = sext i32 %sub159 to i64, !dbg !3477
  %cmp160 = icmp sle i64 %indvars.iv75, %102, !dbg !3477
  br i1 %cmp160, label %for.body161, label %for.end267, !dbg !3478

for.body161:                                      ; preds = %for.cond158
  call void @llvm.dbg.value(metadata i32 0, metadata !3298, metadata !DIExpression()), !dbg !3253
  br label %for.cond162, !dbg !3479

for.cond162:                                      ; preds = %for.inc262, %for.body161
  %indvars.iv41 = phi i64 [ %indvars.iv.next42, %for.inc262 ], [ 0, %for.body161 ], !dbg !3482
  call void @llvm.dbg.value(metadata i64 %indvars.iv41, metadata !3298, metadata !DIExpression()), !dbg !3253
  %sub163 = sub nsw i32 %shl153, 1, !dbg !3483
  %103 = sext i32 %sub163 to i64, !dbg !3485
  %cmp164 = icmp sle i64 %indvars.iv41, %103, !dbg !3485
  br i1 %cmp164, label %for.body165, label %for.end264, !dbg !3486

for.body165:                                      ; preds = %for.cond162
  %104 = mul nsw i64 %indvars.iv75, %97, !dbg !3487
  %105 = add nsw i64 %104, 128, !dbg !3489
  %106 = mul nsw i64 %indvars.iv75, %98, !dbg !3490
  %107 = add nsw i64 %106, %99, !dbg !3491
  %108 = add nsw i64 %100, %indvars.iv75, !dbg !3492
  %arrayidx172 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %u_device, i64 %108, !dbg !3493
  %real173 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx172, i32 0, i32 0, !dbg !3494
  %109 = load double, double* %real173, align 8, !dbg !3494
  call void @llvm.dbg.value(metadata double %109, metadata !3495, metadata !DIExpression()), !dbg !3253
  %conv174 = sitofp i32 %is to double, !dbg !3496
  %110 = add nsw i64 %101, %indvars.iv75, !dbg !3497
  %arrayidx177 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %u_device, i64 %110, !dbg !3498
  %imag178 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx177, i32 0, i32 1, !dbg !3499
  %111 = load double, double* %imag178, align 8, !dbg !3499
  %mul179 = fmul contract double %conv174, %111, !dbg !3500
  call void @llvm.dbg.value(metadata double %mul179, metadata !3501, metadata !DIExpression()), !dbg !3253
  %112 = add nsw i64 %104, %indvars.iv41, !dbg !3502
  %113 = mul nsw i64 %112, 256, !dbg !3503
  %114 = add nsw i64 %8, %113, !dbg !3504
  %mul183 = mul nsw i32 %rem3, 256, !dbg !3505
  %mul184 = mul nsw i32 %mul183, 256, !dbg !3506
  %115 = sext i32 %mul184 to i64, !dbg !3507
  %116 = add nsw i64 %114, %115, !dbg !3507
  %arrayidx187 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %gty2, i64 %116, !dbg !3508
  %real188 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx187, i32 0, i32 0, !dbg !3509
  %117 = load double, double* %real188, align 8, !dbg !3509
  call void @llvm.dbg.value(metadata double %117, metadata !3510, metadata !DIExpression()), !dbg !3253
  %118 = add nsw i64 %104, %indvars.iv41, !dbg !3511
  %119 = mul nsw i64 %118, 256, !dbg !3512
  %120 = add nsw i64 %9, %119, !dbg !3513
  %mul192 = mul nsw i32 %rem3, 256, !dbg !3514
  %mul193 = mul nsw i32 %mul192, 256, !dbg !3515
  %121 = sext i32 %mul193 to i64, !dbg !3516
  %122 = add nsw i64 %120, %121, !dbg !3516
  %arrayidx196 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %gty2, i64 %122, !dbg !3517
  %imag197 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx196, i32 0, i32 1, !dbg !3518
  %123 = load double, double* %imag197, align 8, !dbg !3518
  call void @llvm.dbg.value(metadata double %123, metadata !3519, metadata !DIExpression()), !dbg !3253
  %124 = add nsw i64 %105, %indvars.iv41, !dbg !3520
  %125 = mul nsw i64 %124, 256, !dbg !3521
  %126 = add nsw i64 %10, %125, !dbg !3522
  %mul201 = mul nsw i32 %rem3, 256, !dbg !3523
  %mul202 = mul nsw i32 %mul201, 256, !dbg !3524
  %127 = sext i32 %mul202 to i64, !dbg !3525
  %128 = add nsw i64 %126, %127, !dbg !3525
  %arrayidx205 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %gty2, i64 %128, !dbg !3526
  %real206 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx205, i32 0, i32 0, !dbg !3527
  %129 = load double, double* %real206, align 8, !dbg !3527
  call void @llvm.dbg.value(metadata double %129, metadata !3528, metadata !DIExpression()), !dbg !3253
  %130 = add nsw i64 %105, %indvars.iv41, !dbg !3529
  %131 = mul nsw i64 %130, 256, !dbg !3530
  %132 = add nsw i64 %11, %131, !dbg !3531
  %mul210 = mul nsw i32 %rem3, 256, !dbg !3532
  %mul211 = mul nsw i32 %mul210, 256, !dbg !3533
  %133 = sext i32 %mul211 to i64, !dbg !3534
  %134 = add nsw i64 %132, %133, !dbg !3534
  %arrayidx214 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %gty2, i64 %134, !dbg !3535
  %imag215 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx214, i32 0, i32 1, !dbg !3536
  %135 = load double, double* %imag215, align 8, !dbg !3536
  call void @llvm.dbg.value(metadata double %135, metadata !3537, metadata !DIExpression()), !dbg !3253
  %add216 = fadd contract double %117, %129, !dbg !3538
  %136 = add nsw i64 %106, %indvars.iv41, !dbg !3539
  %137 = mul nsw i64 %136, 256, !dbg !3540
  %138 = add nsw i64 %12, %137, !dbg !3541
  %mul220 = mul nsw i32 %rem3, 256, !dbg !3542
  %mul221 = mul nsw i32 %mul220, 256, !dbg !3543
  %139 = sext i32 %mul221 to i64, !dbg !3544
  %140 = add nsw i64 %138, %139, !dbg !3544
  %arrayidx224 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %gty1, i64 %140, !dbg !3545
  %real225 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx224, i32 0, i32 0, !dbg !3546
  store double %add216, double* %real225, align 8, !dbg !3547
  %add226 = fadd contract double %123, %135, !dbg !3548
  %141 = add nsw i64 %106, %indvars.iv41, !dbg !3549
  %142 = mul nsw i64 %141, 256, !dbg !3550
  %143 = add nsw i64 %13, %142, !dbg !3551
  %mul230 = mul nsw i32 %rem3, 256, !dbg !3552
  %mul231 = mul nsw i32 %mul230, 256, !dbg !3553
  %144 = sext i32 %mul231 to i64, !dbg !3554
  %145 = add nsw i64 %143, %144, !dbg !3554
  %arrayidx234 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %gty1, i64 %145, !dbg !3555
  %imag235 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx234, i32 0, i32 1, !dbg !3556
  store double %add226, double* %imag235, align 8, !dbg !3557
  %sub236 = fsub contract double %117, %129, !dbg !3558
  call void @llvm.dbg.value(metadata double %sub236, metadata !3559, metadata !DIExpression()), !dbg !3253
  %sub237 = fsub contract double %123, %135, !dbg !3560
  call void @llvm.dbg.value(metadata double %sub237, metadata !3561, metadata !DIExpression()), !dbg !3253
  %mul238 = fmul contract double %109, %sub236, !dbg !3562
  %mul239 = fmul contract double %mul179, %sub237, !dbg !3563
  %sub240 = fsub contract double %mul238, %mul239, !dbg !3564
  %146 = add nsw i64 %107, %indvars.iv41, !dbg !3565
  %147 = mul nsw i64 %146, 256, !dbg !3566
  %148 = add nsw i64 %14, %147, !dbg !3567
  %mul244 = mul nsw i32 %rem3, 256, !dbg !3568
  %mul245 = mul nsw i32 %mul244, 256, !dbg !3569
  %149 = sext i32 %mul245 to i64, !dbg !3570
  %150 = add nsw i64 %148, %149, !dbg !3570
  %arrayidx248 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %gty1, i64 %150, !dbg !3571
  %real249 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx248, i32 0, i32 0, !dbg !3572
  store double %sub240, double* %real249, align 8, !dbg !3573
  %mul250 = fmul contract double %109, %sub237, !dbg !3574
  %mul251 = fmul contract double %mul179, %sub236, !dbg !3575
  %add252 = fadd contract double %mul250, %mul251, !dbg !3576
  %151 = add nsw i64 %107, %indvars.iv41, !dbg !3577
  %152 = mul nsw i64 %151, 256, !dbg !3578
  %153 = add nsw i64 %15, %152, !dbg !3579
  %mul256 = mul nsw i32 %rem3, 256, !dbg !3580
  %mul257 = mul nsw i32 %mul256, 256, !dbg !3581
  %154 = sext i32 %mul257 to i64, !dbg !3582
  %155 = add nsw i64 %153, %154, !dbg !3582
  %arrayidx260 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %gty1, i64 %155, !dbg !3583
  %imag261 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx260, i32 0, i32 1, !dbg !3584
  store double %add252, double* %imag261, align 8, !dbg !3585
  br label %for.inc262, !dbg !3586

for.inc262:                                       ; preds = %for.body165
  %indvars.iv.next42 = add nuw nsw i64 %indvars.iv41, 1, !dbg !3587
  call void @llvm.dbg.value(metadata i32 undef, metadata !3298, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !3253
  br label %for.cond162, !dbg !3588, !llvm.loop !3589

for.end264:                                       ; preds = %for.cond162
  br label %for.inc265, !dbg !3591

for.inc265:                                       ; preds = %for.end264
  %indvars.iv.next76 = add nuw nsw i64 %indvars.iv75, 1, !dbg !3592
  call void @llvm.dbg.value(metadata i32 undef, metadata !3290, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !3253
  br label %for.cond158, !dbg !3593, !llvm.loop !3594

for.end267:                                       ; preds = %for.cond158
  br label %if.end268

if.end268:                                        ; preds = %for.end267, %for.end150
  br label %for.inc269, !dbg !3596

for.inc269:                                       ; preds = %if.end268
  %add270 = add nuw nsw i32 %l.0, 2, !dbg !3597
  call void @llvm.dbg.value(metadata i32 %add270, metadata !3272, metadata !DIExpression()), !dbg !3253
  br label %for.cond, !dbg !3598, !llvm.loop !3599

for.end271.loopexit:                              ; preds = %for.cond
  br label %for.end271, !dbg !3601

for.end271:                                       ; preds = %for.end271.loopexit, %if.then
  ret void, !dbg !3601
}

; Function Attrs: convergent noinline nounwind
define dso_local void @cffts1_gpu_kernel_3(%struct.dcomplex* %x_out, %struct.dcomplex* %y0, i32 %gridDim.x, i32 %gridDim.y, i32 %gridDim.z, i32 %blockDim.x, i32 %blockDim.y, i32 %blockDim.z, i32 %blockIdx.x, i32 %blockIdx.y, i32 %blockIdx.z, i32 %threadIdx.x, i32 %threadIdx.y, i32 %threadIdx.z) #2 {
entry:
  call void @llvm.dbg.value(metadata %struct.dcomplex* %x_out, metadata !3602, metadata !DIExpression()), !dbg !3604
  call void @llvm.dbg.value(metadata %struct.dcomplex* %y0, metadata !3605, metadata !DIExpression()), !dbg !3604
  %mul = mul i32 %blockIdx.x, %blockDim.x, !dbg !3606
  %add = add i32 %mul, %threadIdx.x, !dbg !3607
  call void @llvm.dbg.value(metadata i32 %add, metadata !3608, metadata !DIExpression()), !dbg !3604
  %cmp = icmp sge i32 %add, 8388608, !dbg !3609
  br i1 %cmp, label %if.then, label %if.end, !dbg !3611

if.then:                                          ; preds = %entry
  br label %return, !dbg !3612

if.end:                                           ; preds = %entry
  %rem = srem i32 %add, 256, !dbg !3614
  call void @llvm.dbg.value(metadata i32 %rem, metadata !3615, metadata !DIExpression()), !dbg !3604
  %div = sdiv i32 %add, 256, !dbg !3616
  %rem3 = srem i32 %div, 256, !dbg !3617
  call void @llvm.dbg.value(metadata i32 %rem3, metadata !3618, metadata !DIExpression()), !dbg !3604
  %div4 = sdiv i32 %add, 65536, !dbg !3619
  call void @llvm.dbg.value(metadata i32 %div4, metadata !3620, metadata !DIExpression()), !dbg !3604
  %mul5 = mul nsw i32 %rem, 256, !dbg !3621
  %add6 = add nsw i32 %rem3, %mul5, !dbg !3622
  %mul7 = mul nsw i32 %div4, 256, !dbg !3623
  %mul8 = mul nsw i32 %mul7, 256, !dbg !3624
  %add9 = add nsw i32 %add6, %mul8, !dbg !3625
  %idxprom = sext i32 %add9 to i64, !dbg !3626
  %arrayidx = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %y0, i64 %idxprom, !dbg !3626
  %real = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx, i32 0, i32 0, !dbg !3627
  %0 = load double, double* %real, align 8, !dbg !3627
  %idxprom10 = sext i32 %add to i64, !dbg !3628
  %arrayidx11 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %x_out, i64 %idxprom10, !dbg !3628
  %real12 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx11, i32 0, i32 0, !dbg !3629
  store double %0, double* %real12, align 8, !dbg !3630
  %mul13 = mul nsw i32 %rem, 256, !dbg !3631
  %add14 = add nsw i32 %rem3, %mul13, !dbg !3632
  %mul15 = mul nsw i32 %div4, 256, !dbg !3633
  %mul16 = mul nsw i32 %mul15, 256, !dbg !3634
  %add17 = add nsw i32 %add14, %mul16, !dbg !3635
  %idxprom18 = sext i32 %add17 to i64, !dbg !3636
  %arrayidx19 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %y0, i64 %idxprom18, !dbg !3636
  %imag = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx19, i32 0, i32 1, !dbg !3637
  %1 = load double, double* %imag, align 8, !dbg !3637
  %idxprom20 = sext i32 %add to i64, !dbg !3638
  %arrayidx21 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %x_out, i64 %idxprom20, !dbg !3638
  %imag22 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx21, i32 0, i32 1, !dbg !3639
  store double %1, double* %imag22, align 8, !dbg !3640
  br label %return, !dbg !3641

return:                                           ; preds = %if.end, %if.then
  ret void, !dbg !3641
}

; Function Attrs: convergent noinline nounwind
define dso_local void @cffts2_gpu_kernel_1(%struct.dcomplex* %x_in, %struct.dcomplex* %y0, i32 %gridDim.x, i32 %gridDim.y, i32 %gridDim.z, i32 %blockDim.x, i32 %blockDim.y, i32 %blockDim.z, i32 %blockIdx.x, i32 %blockIdx.y, i32 %blockIdx.z, i32 %threadIdx.x, i32 %threadIdx.y, i32 %threadIdx.z) #2 {
entry:
  call void @llvm.dbg.value(metadata %struct.dcomplex* %x_in, metadata !3642, metadata !DIExpression()), !dbg !3644
  call void @llvm.dbg.value(metadata %struct.dcomplex* %y0, metadata !3645, metadata !DIExpression()), !dbg !3644
  %mul = mul i32 %blockIdx.x, %blockDim.x, !dbg !3646
  %add = add i32 %mul, %threadIdx.x, !dbg !3647
  call void @llvm.dbg.value(metadata i32 %add, metadata !3648, metadata !DIExpression()), !dbg !3644
  %cmp = icmp sge i32 %add, 8388608, !dbg !3649
  br i1 %cmp, label %if.then, label %if.end, !dbg !3651

if.then:                                          ; preds = %entry
  br label %return, !dbg !3652

if.end:                                           ; preds = %entry
  %idxprom = sext i32 %add to i64, !dbg !3654
  %arrayidx = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %x_in, i64 %idxprom, !dbg !3654
  %real = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx, i32 0, i32 0, !dbg !3655
  %0 = load double, double* %real, align 8, !dbg !3655
  %idxprom3 = sext i32 %add to i64, !dbg !3656
  %arrayidx4 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %y0, i64 %idxprom3, !dbg !3656
  %real5 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx4, i32 0, i32 0, !dbg !3657
  store double %0, double* %real5, align 8, !dbg !3658
  %idxprom6 = sext i32 %add to i64, !dbg !3659
  %arrayidx7 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %x_in, i64 %idxprom6, !dbg !3659
  %imag = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx7, i32 0, i32 1, !dbg !3660
  %1 = load double, double* %imag, align 8, !dbg !3660
  %idxprom8 = sext i32 %add to i64, !dbg !3661
  %arrayidx9 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %y0, i64 %idxprom8, !dbg !3661
  %imag10 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx9, i32 0, i32 1, !dbg !3662
  store double %1, double* %imag10, align 8, !dbg !3663
  br label %return, !dbg !3664

return:                                           ; preds = %if.end, %if.then
  ret void, !dbg !3664
}

; Function Attrs: convergent noinline nounwind
define dso_local void @cffts2_gpu_kernel_2(i32 %is, %struct.dcomplex* %gty1, %struct.dcomplex* %gty2, %struct.dcomplex* %u_device, i32 %gridDim.x, i32 %gridDim.y, i32 %gridDim.z, i32 %blockDim.x, i32 %blockDim.y, i32 %blockDim.z, i32 %blockIdx.x, i32 %blockIdx.y, i32 %blockIdx.z, i32 %threadIdx.x, i32 %threadIdx.y, i32 %threadIdx.z) #2 {
entry:
  call void @llvm.dbg.value(metadata i32 %is, metadata !3665, metadata !DIExpression()), !dbg !3667
  call void @llvm.dbg.value(metadata %struct.dcomplex* %gty1, metadata !3668, metadata !DIExpression()), !dbg !3667
  call void @llvm.dbg.value(metadata %struct.dcomplex* %gty2, metadata !3669, metadata !DIExpression()), !dbg !3667
  call void @llvm.dbg.value(metadata %struct.dcomplex* %u_device, metadata !3670, metadata !DIExpression()), !dbg !3667
  %mul = mul i32 %blockIdx.x, %blockDim.x, !dbg !3671
  %add = add i32 %mul, %threadIdx.x, !dbg !3672
  call void @llvm.dbg.value(metadata i32 %add, metadata !3673, metadata !DIExpression()), !dbg !3667
  %cmp = icmp sge i32 %add, 32768, !dbg !3674
  br i1 %cmp, label %if.then, label %if.end, !dbg !3676

if.then:                                          ; preds = %entry
  br label %for.end271, !dbg !3677

if.end:                                           ; preds = %entry
  %rem = srem i32 %add, 256, !dbg !3679
  call void @llvm.dbg.value(metadata i32 %rem, metadata !3680, metadata !DIExpression()), !dbg !3667
  %div = sdiv i32 %add, 256, !dbg !3681
  %rem3 = srem i32 %div, 128, !dbg !3682
  call void @llvm.dbg.value(metadata i32 %rem3, metadata !3683, metadata !DIExpression()), !dbg !3667
  %call4 = call i32 @_Z12ilog2_devicei(i32 256) #4, !dbg !3684
  call void @llvm.dbg.value(metadata i32 %call4, metadata !3685, metadata !DIExpression()), !dbg !3667
  call void @llvm.dbg.value(metadata i32 1, metadata !3686, metadata !DIExpression()), !dbg !3667
  %0 = sext i32 %rem to i64, !dbg !3687
  %1 = sext i32 %rem to i64, !dbg !3687
  %2 = sext i32 %rem to i64, !dbg !3687
  %3 = sext i32 %rem to i64, !dbg !3687
  %4 = sext i32 %rem to i64, !dbg !3687
  %5 = sext i32 %rem to i64, !dbg !3687
  %6 = sext i32 %rem to i64, !dbg !3687
  %7 = sext i32 %rem to i64, !dbg !3687
  %8 = sext i32 %rem to i64, !dbg !3687
  %9 = sext i32 %rem to i64, !dbg !3687
  %10 = sext i32 %rem to i64, !dbg !3687
  %11 = sext i32 %rem to i64, !dbg !3687
  %12 = sext i32 %rem to i64, !dbg !3687
  %13 = sext i32 %rem to i64, !dbg !3687
  %14 = sext i32 %rem to i64, !dbg !3687
  %15 = sext i32 %rem to i64, !dbg !3687
  %16 = sext i32 %rem to i64, !dbg !3687
  %17 = sext i32 %rem to i64, !dbg !3687
  %18 = sext i32 %rem to i64, !dbg !3687
  %19 = sext i32 %rem to i64, !dbg !3687
  br label %for.cond, !dbg !3687

for.cond:                                         ; preds = %for.inc269, %if.end
  %l.0 = phi i32 [ 1, %if.end ], [ %add270, %for.inc269 ], !dbg !3689
  call void @llvm.dbg.value(metadata i32 %l.0, metadata !3686, metadata !DIExpression()), !dbg !3667
  %cmp5 = icmp sle i32 %l.0, %call4, !dbg !3690
  br i1 %cmp5, label %for.body, label %for.end271.loopexit, !dbg !3692

for.body:                                         ; preds = %for.cond
  call void @llvm.dbg.value(metadata i32 128, metadata !3693, metadata !DIExpression()), !dbg !3667
  %sub = sub nuw nsw i32 %l.0, 1, !dbg !3694
  %shl = shl i32 1, %sub, !dbg !3696
  call void @llvm.dbg.value(metadata i32 %shl, metadata !3697, metadata !DIExpression()), !dbg !3667
  %sub6 = sub nsw i32 %call4, %l.0, !dbg !3698
  %shl7 = shl i32 1, %sub6, !dbg !3699
  call void @llvm.dbg.value(metadata i32 %shl7, metadata !3700, metadata !DIExpression()), !dbg !3667
  %mul8 = mul nsw i32 2, %shl, !dbg !3701
  call void @llvm.dbg.value(metadata i32 %mul8, metadata !3702, metadata !DIExpression()), !dbg !3667
  call void @llvm.dbg.value(metadata i32 %shl7, metadata !3703, metadata !DIExpression()), !dbg !3667
  call void @llvm.dbg.value(metadata i32 0, metadata !3704, metadata !DIExpression()), !dbg !3667
  %20 = sext i32 %shl to i64, !dbg !3705
  %21 = sext i32 %mul8 to i64, !dbg !3705
  %22 = sext i32 %shl to i64, !dbg !3705
  %23 = sext i32 %shl7 to i64, !dbg !3705
  %24 = sext i32 %shl7 to i64, !dbg !3705
  br label %for.cond9, !dbg !3705

for.cond9:                                        ; preds = %for.inc108, %for.body
  %indvars.iv33 = phi i64 [ %indvars.iv.next34, %for.inc108 ], [ 0, %for.body ], !dbg !3707
  call void @llvm.dbg.value(metadata i64 %indvars.iv33, metadata !3704, metadata !DIExpression()), !dbg !3667
  %sub10 = sub nsw i32 %shl7, 1, !dbg !3708
  %25 = sext i32 %sub10 to i64, !dbg !3710
  %cmp11 = icmp sle i64 %indvars.iv33, %25, !dbg !3710
  br i1 %cmp11, label %for.body12, label %for.end110, !dbg !3711

for.body12:                                       ; preds = %for.cond9
  call void @llvm.dbg.value(metadata i32 0, metadata !3712, metadata !DIExpression()), !dbg !3667
  br label %for.cond13, !dbg !3713

for.cond13:                                       ; preds = %for.inc, %for.body12
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %for.body12 ], !dbg !3716
  call void @llvm.dbg.value(metadata i64 %indvars.iv, metadata !3712, metadata !DIExpression()), !dbg !3667
  %sub14 = sub nsw i32 %shl, 1, !dbg !3717
  %26 = sext i32 %sub14 to i64, !dbg !3719
  %cmp15 = icmp sle i64 %indvars.iv, %26, !dbg !3719
  br i1 %cmp15, label %for.body16, label %for.end, !dbg !3720

for.body16:                                       ; preds = %for.cond13
  %27 = mul nsw i64 %indvars.iv33, %20, !dbg !3721
  %28 = add nsw i64 %27, 128, !dbg !3723
  %29 = mul nsw i64 %indvars.iv33, %21, !dbg !3724
  %30 = add nsw i64 %29, %22, !dbg !3725
  %31 = add nsw i64 %23, %indvars.iv33, !dbg !3726
  %arrayidx = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %u_device, i64 %31, !dbg !3727
  %real = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx, i32 0, i32 0, !dbg !3728
  %32 = load double, double* %real, align 8, !dbg !3728
  call void @llvm.dbg.value(metadata double %32, metadata !3729, metadata !DIExpression()), !dbg !3667
  %conv = sitofp i32 %is to double, !dbg !3730
  %33 = add nsw i64 %24, %indvars.iv33, !dbg !3731
  %arrayidx24 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %u_device, i64 %33, !dbg !3732
  %imag = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx24, i32 0, i32 1, !dbg !3733
  %34 = load double, double* %imag, align 8, !dbg !3733
  %mul25 = fmul contract double %conv, %34, !dbg !3734
  call void @llvm.dbg.value(metadata double %mul25, metadata !3735, metadata !DIExpression()), !dbg !3667
  %35 = add nsw i64 %27, %indvars.iv, !dbg !3736
  %36 = mul nsw i64 %35, 256, !dbg !3737
  %37 = add nsw i64 %0, %36, !dbg !3738
  %mul29 = mul nsw i32 %rem3, 256, !dbg !3739
  %mul30 = mul nsw i32 %mul29, 256, !dbg !3740
  %38 = sext i32 %mul30 to i64, !dbg !3741
  %39 = add nsw i64 %37, %38, !dbg !3741
  %arrayidx33 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %gty1, i64 %39, !dbg !3742
  %real34 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx33, i32 0, i32 0, !dbg !3743
  %40 = load double, double* %real34, align 8, !dbg !3743
  call void @llvm.dbg.value(metadata double %40, metadata !3744, metadata !DIExpression()), !dbg !3667
  %41 = add nsw i64 %27, %indvars.iv, !dbg !3745
  %42 = mul nsw i64 %41, 256, !dbg !3746
  %43 = add nsw i64 %1, %42, !dbg !3747
  %mul38 = mul nsw i32 %rem3, 256, !dbg !3748
  %mul39 = mul nsw i32 %mul38, 256, !dbg !3749
  %44 = sext i32 %mul39 to i64, !dbg !3750
  %45 = add nsw i64 %43, %44, !dbg !3750
  %arrayidx42 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %gty1, i64 %45, !dbg !3751
  %imag43 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx42, i32 0, i32 1, !dbg !3752
  %46 = load double, double* %imag43, align 8, !dbg !3752
  call void @llvm.dbg.value(metadata double %46, metadata !3753, metadata !DIExpression()), !dbg !3667
  %47 = add nsw i64 %28, %indvars.iv, !dbg !3754
  %48 = mul nsw i64 %47, 256, !dbg !3755
  %49 = add nsw i64 %2, %48, !dbg !3756
  %mul47 = mul nsw i32 %rem3, 256, !dbg !3757
  %mul48 = mul nsw i32 %mul47, 256, !dbg !3758
  %50 = sext i32 %mul48 to i64, !dbg !3759
  %51 = add nsw i64 %49, %50, !dbg !3759
  %arrayidx51 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %gty1, i64 %51, !dbg !3760
  %real52 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx51, i32 0, i32 0, !dbg !3761
  %52 = load double, double* %real52, align 8, !dbg !3761
  call void @llvm.dbg.value(metadata double %52, metadata !3762, metadata !DIExpression()), !dbg !3667
  %53 = add nsw i64 %28, %indvars.iv, !dbg !3763
  %54 = mul nsw i64 %53, 256, !dbg !3764
  %55 = add nsw i64 %3, %54, !dbg !3765
  %mul56 = mul nsw i32 %rem3, 256, !dbg !3766
  %mul57 = mul nsw i32 %mul56, 256, !dbg !3767
  %56 = sext i32 %mul57 to i64, !dbg !3768
  %57 = add nsw i64 %55, %56, !dbg !3768
  %arrayidx60 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %gty1, i64 %57, !dbg !3769
  %imag61 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx60, i32 0, i32 1, !dbg !3770
  %58 = load double, double* %imag61, align 8, !dbg !3770
  call void @llvm.dbg.value(metadata double %58, metadata !3771, metadata !DIExpression()), !dbg !3667
  %add62 = fadd contract double %40, %52, !dbg !3772
  %59 = add nsw i64 %29, %indvars.iv, !dbg !3773
  %60 = mul nsw i64 %59, 256, !dbg !3774
  %61 = add nsw i64 %4, %60, !dbg !3775
  %mul66 = mul nsw i32 %rem3, 256, !dbg !3776
  %mul67 = mul nsw i32 %mul66, 256, !dbg !3777
  %62 = sext i32 %mul67 to i64, !dbg !3778
  %63 = add nsw i64 %61, %62, !dbg !3778
  %arrayidx70 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %gty2, i64 %63, !dbg !3779
  %real71 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx70, i32 0, i32 0, !dbg !3780
  store double %add62, double* %real71, align 8, !dbg !3781
  %add72 = fadd contract double %46, %58, !dbg !3782
  %64 = add nsw i64 %29, %indvars.iv, !dbg !3783
  %65 = mul nsw i64 %64, 256, !dbg !3784
  %66 = add nsw i64 %5, %65, !dbg !3785
  %mul76 = mul nsw i32 %rem3, 256, !dbg !3786
  %mul77 = mul nsw i32 %mul76, 256, !dbg !3787
  %67 = sext i32 %mul77 to i64, !dbg !3788
  %68 = add nsw i64 %66, %67, !dbg !3788
  %arrayidx80 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %gty2, i64 %68, !dbg !3789
  %imag81 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx80, i32 0, i32 1, !dbg !3790
  store double %add72, double* %imag81, align 8, !dbg !3791
  %sub82 = fsub contract double %40, %52, !dbg !3792
  call void @llvm.dbg.value(metadata double %sub82, metadata !3793, metadata !DIExpression()), !dbg !3667
  %sub83 = fsub contract double %46, %58, !dbg !3794
  call void @llvm.dbg.value(metadata double %sub83, metadata !3795, metadata !DIExpression()), !dbg !3667
  %mul84 = fmul contract double %32, %sub82, !dbg !3796
  %mul85 = fmul contract double %mul25, %sub83, !dbg !3797
  %sub86 = fsub contract double %mul84, %mul85, !dbg !3798
  %69 = add nsw i64 %30, %indvars.iv, !dbg !3799
  %70 = mul nsw i64 %69, 256, !dbg !3800
  %71 = add nsw i64 %6, %70, !dbg !3801
  %mul90 = mul nsw i32 %rem3, 256, !dbg !3802
  %mul91 = mul nsw i32 %mul90, 256, !dbg !3803
  %72 = sext i32 %mul91 to i64, !dbg !3804
  %73 = add nsw i64 %71, %72, !dbg !3804
  %arrayidx94 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %gty2, i64 %73, !dbg !3805
  %real95 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx94, i32 0, i32 0, !dbg !3806
  store double %sub86, double* %real95, align 8, !dbg !3807
  %mul96 = fmul contract double %32, %sub83, !dbg !3808
  %mul97 = fmul contract double %mul25, %sub82, !dbg !3809
  %add98 = fadd contract double %mul96, %mul97, !dbg !3810
  %74 = add nsw i64 %30, %indvars.iv, !dbg !3811
  %75 = mul nsw i64 %74, 256, !dbg !3812
  %76 = add nsw i64 %7, %75, !dbg !3813
  %mul102 = mul nsw i32 %rem3, 256, !dbg !3814
  %mul103 = mul nsw i32 %mul102, 256, !dbg !3815
  %77 = sext i32 %mul103 to i64, !dbg !3816
  %78 = add nsw i64 %76, %77, !dbg !3816
  %arrayidx106 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %gty2, i64 %78, !dbg !3817
  %imag107 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx106, i32 0, i32 1, !dbg !3818
  store double %add98, double* %imag107, align 8, !dbg !3819
  br label %for.inc, !dbg !3820

for.inc:                                          ; preds = %for.body16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !3821
  call void @llvm.dbg.value(metadata i32 undef, metadata !3712, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !3667
  br label %for.cond13, !dbg !3822, !llvm.loop !3823

for.end:                                          ; preds = %for.cond13
  br label %for.inc108, !dbg !3825

for.inc108:                                       ; preds = %for.end
  %indvars.iv.next34 = add nuw nsw i64 %indvars.iv33, 1, !dbg !3826
  call void @llvm.dbg.value(metadata i32 undef, metadata !3704, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !3667
  br label %for.cond9, !dbg !3827, !llvm.loop !3828

for.end110:                                       ; preds = %for.cond9
  %cmp111 = icmp eq i32 %l.0, %call4, !dbg !3830
  br i1 %cmp111, label %if.then112, label %if.else, !dbg !3832

if.then112:                                       ; preds = %for.end110
  call void @llvm.dbg.value(metadata i32 0, metadata !3833, metadata !DIExpression()), !dbg !3667
  br label %for.cond113, !dbg !3834

for.cond113:                                      ; preds = %for.inc148, %if.then112
  %indvars.iv83 = phi i64 [ %indvars.iv.next84, %for.inc148 ], [ 0, %if.then112 ], !dbg !3837
  call void @llvm.dbg.value(metadata i64 %indvars.iv83, metadata !3833, metadata !DIExpression()), !dbg !3667
  %exitcond = icmp ne i64 %indvars.iv83, 256, !dbg !3838
  br i1 %exitcond, label %for.body115, label %for.end150, !dbg !3840

for.body115:                                      ; preds = %for.cond113
  %79 = mul nuw nsw i64 %indvars.iv83, 256, !dbg !3841
  %80 = add nsw i64 %16, %79, !dbg !3843
  %mul118 = mul nsw i32 %rem3, 256, !dbg !3844
  %mul119 = mul nsw i32 %mul118, 256, !dbg !3845
  %81 = sext i32 %mul119 to i64, !dbg !3846
  %82 = add nsw i64 %80, %81, !dbg !3846
  %arrayidx122 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %gty2, i64 %82, !dbg !3847
  %real123 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx122, i32 0, i32 0, !dbg !3848
  %83 = load double, double* %real123, align 8, !dbg !3848
  %84 = mul nuw nsw i64 %indvars.iv83, 256, !dbg !3849
  %85 = add nsw i64 %17, %84, !dbg !3850
  %mul126 = mul nsw i32 %rem3, 256, !dbg !3851
  %mul127 = mul nsw i32 %mul126, 256, !dbg !3852
  %86 = sext i32 %mul127 to i64, !dbg !3853
  %87 = add nsw i64 %85, %86, !dbg !3853
  %arrayidx130 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %gty1, i64 %87, !dbg !3854
  %real131 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx130, i32 0, i32 0, !dbg !3855
  store double %83, double* %real131, align 8, !dbg !3856
  %88 = mul nuw nsw i64 %indvars.iv83, 256, !dbg !3857
  %89 = add nsw i64 %18, %88, !dbg !3858
  %mul134 = mul nsw i32 %rem3, 256, !dbg !3859
  %mul135 = mul nsw i32 %mul134, 256, !dbg !3860
  %90 = sext i32 %mul135 to i64, !dbg !3861
  %91 = add nsw i64 %89, %90, !dbg !3861
  %arrayidx138 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %gty2, i64 %91, !dbg !3862
  %imag139 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx138, i32 0, i32 1, !dbg !3863
  %92 = load double, double* %imag139, align 8, !dbg !3863
  %93 = mul nuw nsw i64 %indvars.iv83, 256, !dbg !3864
  %94 = add nsw i64 %19, %93, !dbg !3865
  %mul142 = mul nsw i32 %rem3, 256, !dbg !3866
  %mul143 = mul nsw i32 %mul142, 256, !dbg !3867
  %95 = sext i32 %mul143 to i64, !dbg !3868
  %96 = add nsw i64 %94, %95, !dbg !3868
  %arrayidx146 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %gty1, i64 %96, !dbg !3869
  %imag147 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx146, i32 0, i32 1, !dbg !3870
  store double %92, double* %imag147, align 8, !dbg !3871
  br label %for.inc148, !dbg !3872

for.inc148:                                       ; preds = %for.body115
  %indvars.iv.next84 = add nuw nsw i64 %indvars.iv83, 1, !dbg !3873
  call void @llvm.dbg.value(metadata i32 undef, metadata !3833, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !3667
  br label %for.cond113, !dbg !3874, !llvm.loop !3875

for.end150:                                       ; preds = %for.cond113
  br label %if.end268, !dbg !3877

if.else:                                          ; preds = %for.end110
  call void @llvm.dbg.value(metadata i32 128, metadata !3693, metadata !DIExpression()), !dbg !3667
  %add151 = add nuw nsw i32 %l.0, 1, !dbg !3878
  %sub152 = sub nuw nsw i32 %add151, 1, !dbg !3880
  %shl153 = shl i32 1, %sub152, !dbg !3881
  call void @llvm.dbg.value(metadata i32 %shl153, metadata !3697, metadata !DIExpression()), !dbg !3667
  %add154 = add nuw nsw i32 %l.0, 1, !dbg !3882
  %sub155 = sub nsw i32 %call4, %add154, !dbg !3883
  %shl156 = shl i32 1, %sub155, !dbg !3884
  call void @llvm.dbg.value(metadata i32 %shl156, metadata !3700, metadata !DIExpression()), !dbg !3667
  %mul157 = mul nsw i32 2, %shl153, !dbg !3885
  call void @llvm.dbg.value(metadata i32 %mul157, metadata !3702, metadata !DIExpression()), !dbg !3667
  call void @llvm.dbg.value(metadata i32 %shl156, metadata !3703, metadata !DIExpression()), !dbg !3667
  call void @llvm.dbg.value(metadata i32 0, metadata !3704, metadata !DIExpression()), !dbg !3667
  %97 = sext i32 %shl153 to i64, !dbg !3886
  %98 = sext i32 %mul157 to i64, !dbg !3886
  %99 = sext i32 %shl153 to i64, !dbg !3886
  %100 = sext i32 %shl156 to i64, !dbg !3886
  %101 = sext i32 %shl156 to i64, !dbg !3886
  br label %for.cond158, !dbg !3886

for.cond158:                                      ; preds = %for.inc265, %if.else
  %indvars.iv75 = phi i64 [ %indvars.iv.next76, %for.inc265 ], [ 0, %if.else ], !dbg !3888
  call void @llvm.dbg.value(metadata i64 %indvars.iv75, metadata !3704, metadata !DIExpression()), !dbg !3667
  %sub159 = sub nsw i32 %shl156, 1, !dbg !3889
  %102 = sext i32 %sub159 to i64, !dbg !3891
  %cmp160 = icmp sle i64 %indvars.iv75, %102, !dbg !3891
  br i1 %cmp160, label %for.body161, label %for.end267, !dbg !3892

for.body161:                                      ; preds = %for.cond158
  call void @llvm.dbg.value(metadata i32 0, metadata !3712, metadata !DIExpression()), !dbg !3667
  br label %for.cond162, !dbg !3893

for.cond162:                                      ; preds = %for.inc262, %for.body161
  %indvars.iv41 = phi i64 [ %indvars.iv.next42, %for.inc262 ], [ 0, %for.body161 ], !dbg !3896
  call void @llvm.dbg.value(metadata i64 %indvars.iv41, metadata !3712, metadata !DIExpression()), !dbg !3667
  %sub163 = sub nsw i32 %shl153, 1, !dbg !3897
  %103 = sext i32 %sub163 to i64, !dbg !3899
  %cmp164 = icmp sle i64 %indvars.iv41, %103, !dbg !3899
  br i1 %cmp164, label %for.body165, label %for.end264, !dbg !3900

for.body165:                                      ; preds = %for.cond162
  %104 = mul nsw i64 %indvars.iv75, %97, !dbg !3901
  %105 = add nsw i64 %104, 128, !dbg !3903
  %106 = mul nsw i64 %indvars.iv75, %98, !dbg !3904
  %107 = add nsw i64 %106, %99, !dbg !3905
  %108 = add nsw i64 %100, %indvars.iv75, !dbg !3906
  %arrayidx172 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %u_device, i64 %108, !dbg !3907
  %real173 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx172, i32 0, i32 0, !dbg !3908
  %109 = load double, double* %real173, align 8, !dbg !3908
  call void @llvm.dbg.value(metadata double %109, metadata !3909, metadata !DIExpression()), !dbg !3667
  %conv174 = sitofp i32 %is to double, !dbg !3910
  %110 = add nsw i64 %101, %indvars.iv75, !dbg !3911
  %arrayidx177 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %u_device, i64 %110, !dbg !3912
  %imag178 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx177, i32 0, i32 1, !dbg !3913
  %111 = load double, double* %imag178, align 8, !dbg !3913
  %mul179 = fmul contract double %conv174, %111, !dbg !3914
  call void @llvm.dbg.value(metadata double %mul179, metadata !3915, metadata !DIExpression()), !dbg !3667
  %112 = add nsw i64 %104, %indvars.iv41, !dbg !3916
  %113 = mul nsw i64 %112, 256, !dbg !3917
  %114 = add nsw i64 %8, %113, !dbg !3918
  %mul183 = mul nsw i32 %rem3, 256, !dbg !3919
  %mul184 = mul nsw i32 %mul183, 256, !dbg !3920
  %115 = sext i32 %mul184 to i64, !dbg !3921
  %116 = add nsw i64 %114, %115, !dbg !3921
  %arrayidx187 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %gty2, i64 %116, !dbg !3922
  %real188 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx187, i32 0, i32 0, !dbg !3923
  %117 = load double, double* %real188, align 8, !dbg !3923
  call void @llvm.dbg.value(metadata double %117, metadata !3924, metadata !DIExpression()), !dbg !3667
  %118 = add nsw i64 %104, %indvars.iv41, !dbg !3925
  %119 = mul nsw i64 %118, 256, !dbg !3926
  %120 = add nsw i64 %9, %119, !dbg !3927
  %mul192 = mul nsw i32 %rem3, 256, !dbg !3928
  %mul193 = mul nsw i32 %mul192, 256, !dbg !3929
  %121 = sext i32 %mul193 to i64, !dbg !3930
  %122 = add nsw i64 %120, %121, !dbg !3930
  %arrayidx196 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %gty2, i64 %122, !dbg !3931
  %imag197 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx196, i32 0, i32 1, !dbg !3932
  %123 = load double, double* %imag197, align 8, !dbg !3932
  call void @llvm.dbg.value(metadata double %123, metadata !3933, metadata !DIExpression()), !dbg !3667
  %124 = add nsw i64 %105, %indvars.iv41, !dbg !3934
  %125 = mul nsw i64 %124, 256, !dbg !3935
  %126 = add nsw i64 %10, %125, !dbg !3936
  %mul201 = mul nsw i32 %rem3, 256, !dbg !3937
  %mul202 = mul nsw i32 %mul201, 256, !dbg !3938
  %127 = sext i32 %mul202 to i64, !dbg !3939
  %128 = add nsw i64 %126, %127, !dbg !3939
  %arrayidx205 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %gty2, i64 %128, !dbg !3940
  %real206 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx205, i32 0, i32 0, !dbg !3941
  %129 = load double, double* %real206, align 8, !dbg !3941
  call void @llvm.dbg.value(metadata double %129, metadata !3942, metadata !DIExpression()), !dbg !3667
  %130 = add nsw i64 %105, %indvars.iv41, !dbg !3943
  %131 = mul nsw i64 %130, 256, !dbg !3944
  %132 = add nsw i64 %11, %131, !dbg !3945
  %mul210 = mul nsw i32 %rem3, 256, !dbg !3946
  %mul211 = mul nsw i32 %mul210, 256, !dbg !3947
  %133 = sext i32 %mul211 to i64, !dbg !3948
  %134 = add nsw i64 %132, %133, !dbg !3948
  %arrayidx214 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %gty2, i64 %134, !dbg !3949
  %imag215 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx214, i32 0, i32 1, !dbg !3950
  %135 = load double, double* %imag215, align 8, !dbg !3950
  call void @llvm.dbg.value(metadata double %135, metadata !3951, metadata !DIExpression()), !dbg !3667
  %add216 = fadd contract double %117, %129, !dbg !3952
  %136 = add nsw i64 %106, %indvars.iv41, !dbg !3953
  %137 = mul nsw i64 %136, 256, !dbg !3954
  %138 = add nsw i64 %12, %137, !dbg !3955
  %mul220 = mul nsw i32 %rem3, 256, !dbg !3956
  %mul221 = mul nsw i32 %mul220, 256, !dbg !3957
  %139 = sext i32 %mul221 to i64, !dbg !3958
  %140 = add nsw i64 %138, %139, !dbg !3958
  %arrayidx224 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %gty1, i64 %140, !dbg !3959
  %real225 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx224, i32 0, i32 0, !dbg !3960
  store double %add216, double* %real225, align 8, !dbg !3961
  %add226 = fadd contract double %123, %135, !dbg !3962
  %141 = add nsw i64 %106, %indvars.iv41, !dbg !3963
  %142 = mul nsw i64 %141, 256, !dbg !3964
  %143 = add nsw i64 %13, %142, !dbg !3965
  %mul230 = mul nsw i32 %rem3, 256, !dbg !3966
  %mul231 = mul nsw i32 %mul230, 256, !dbg !3967
  %144 = sext i32 %mul231 to i64, !dbg !3968
  %145 = add nsw i64 %143, %144, !dbg !3968
  %arrayidx234 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %gty1, i64 %145, !dbg !3969
  %imag235 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx234, i32 0, i32 1, !dbg !3970
  store double %add226, double* %imag235, align 8, !dbg !3971
  %sub236 = fsub contract double %117, %129, !dbg !3972
  call void @llvm.dbg.value(metadata double %sub236, metadata !3973, metadata !DIExpression()), !dbg !3667
  %sub237 = fsub contract double %123, %135, !dbg !3974
  call void @llvm.dbg.value(metadata double %sub237, metadata !3975, metadata !DIExpression()), !dbg !3667
  %mul238 = fmul contract double %109, %sub236, !dbg !3976
  %mul239 = fmul contract double %mul179, %sub237, !dbg !3977
  %sub240 = fsub contract double %mul238, %mul239, !dbg !3978
  %146 = add nsw i64 %107, %indvars.iv41, !dbg !3979
  %147 = mul nsw i64 %146, 256, !dbg !3980
  %148 = add nsw i64 %14, %147, !dbg !3981
  %mul244 = mul nsw i32 %rem3, 256, !dbg !3982
  %mul245 = mul nsw i32 %mul244, 256, !dbg !3983
  %149 = sext i32 %mul245 to i64, !dbg !3984
  %150 = add nsw i64 %148, %149, !dbg !3984
  %arrayidx248 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %gty1, i64 %150, !dbg !3985
  %real249 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx248, i32 0, i32 0, !dbg !3986
  store double %sub240, double* %real249, align 8, !dbg !3987
  %mul250 = fmul contract double %109, %sub237, !dbg !3988
  %mul251 = fmul contract double %mul179, %sub236, !dbg !3989
  %add252 = fadd contract double %mul250, %mul251, !dbg !3990
  %151 = add nsw i64 %107, %indvars.iv41, !dbg !3991
  %152 = mul nsw i64 %151, 256, !dbg !3992
  %153 = add nsw i64 %15, %152, !dbg !3993
  %mul256 = mul nsw i32 %rem3, 256, !dbg !3994
  %mul257 = mul nsw i32 %mul256, 256, !dbg !3995
  %154 = sext i32 %mul257 to i64, !dbg !3996
  %155 = add nsw i64 %153, %154, !dbg !3996
  %arrayidx260 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %gty1, i64 %155, !dbg !3997
  %imag261 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx260, i32 0, i32 1, !dbg !3998
  store double %add252, double* %imag261, align 8, !dbg !3999
  br label %for.inc262, !dbg !4000

for.inc262:                                       ; preds = %for.body165
  %indvars.iv.next42 = add nuw nsw i64 %indvars.iv41, 1, !dbg !4001
  call void @llvm.dbg.value(metadata i32 undef, metadata !3712, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !3667
  br label %for.cond162, !dbg !4002, !llvm.loop !4003

for.end264:                                       ; preds = %for.cond162
  br label %for.inc265, !dbg !4005

for.inc265:                                       ; preds = %for.end264
  %indvars.iv.next76 = add nuw nsw i64 %indvars.iv75, 1, !dbg !4006
  call void @llvm.dbg.value(metadata i32 undef, metadata !3704, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !3667
  br label %for.cond158, !dbg !4007, !llvm.loop !4008

for.end267:                                       ; preds = %for.cond158
  br label %if.end268

if.end268:                                        ; preds = %for.end267, %for.end150
  br label %for.inc269, !dbg !4010

for.inc269:                                       ; preds = %if.end268
  %add270 = add nuw nsw i32 %l.0, 2, !dbg !4011
  call void @llvm.dbg.value(metadata i32 %add270, metadata !3686, metadata !DIExpression()), !dbg !3667
  br label %for.cond, !dbg !4012, !llvm.loop !4013

for.end271.loopexit:                              ; preds = %for.cond
  br label %for.end271, !dbg !4015

for.end271:                                       ; preds = %for.end271.loopexit, %if.then
  ret void, !dbg !4015
}

; Function Attrs: convergent noinline nounwind
define dso_local void @cffts2_gpu_kernel_3(%struct.dcomplex* %x_out, %struct.dcomplex* %y0, i32 %gridDim.x, i32 %gridDim.y, i32 %gridDim.z, i32 %blockDim.x, i32 %blockDim.y, i32 %blockDim.z, i32 %blockIdx.x, i32 %blockIdx.y, i32 %blockIdx.z, i32 %threadIdx.x, i32 %threadIdx.y, i32 %threadIdx.z) #2 {
entry:
  call void @llvm.dbg.value(metadata %struct.dcomplex* %x_out, metadata !4016, metadata !DIExpression()), !dbg !4018
  call void @llvm.dbg.value(metadata %struct.dcomplex* %y0, metadata !4019, metadata !DIExpression()), !dbg !4018
  %mul = mul i32 %blockIdx.x, %blockDim.x, !dbg !4020
  %add = add i32 %mul, %threadIdx.x, !dbg !4021
  call void @llvm.dbg.value(metadata i32 %add, metadata !4022, metadata !DIExpression()), !dbg !4018
  %cmp = icmp sge i32 %add, 8388608, !dbg !4023
  br i1 %cmp, label %if.then, label %if.end, !dbg !4025

if.then:                                          ; preds = %entry
  br label %return, !dbg !4026

if.end:                                           ; preds = %entry
  %idxprom = sext i32 %add to i64, !dbg !4028
  %arrayidx = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %y0, i64 %idxprom, !dbg !4028
  %real = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx, i32 0, i32 0, !dbg !4029
  %0 = load double, double* %real, align 8, !dbg !4029
  %idxprom3 = sext i32 %add to i64, !dbg !4030
  %arrayidx4 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %x_out, i64 %idxprom3, !dbg !4030
  %real5 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx4, i32 0, i32 0, !dbg !4031
  store double %0, double* %real5, align 8, !dbg !4032
  %idxprom6 = sext i32 %add to i64, !dbg !4033
  %arrayidx7 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %y0, i64 %idxprom6, !dbg !4033
  %imag = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx7, i32 0, i32 1, !dbg !4034
  %1 = load double, double* %imag, align 8, !dbg !4034
  %idxprom8 = sext i32 %add to i64, !dbg !4035
  %arrayidx9 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %x_out, i64 %idxprom8, !dbg !4035
  %imag10 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx9, i32 0, i32 1, !dbg !4036
  store double %1, double* %imag10, align 8, !dbg !4037
  br label %return, !dbg !4038

return:                                           ; preds = %if.end, %if.then
  ret void, !dbg !4038
}

; Function Attrs: convergent noinline nounwind
define dso_local void @cffts3_gpu_kernel_1(%struct.dcomplex* %x_in, %struct.dcomplex* %y0, i32 %gridDim.x, i32 %gridDim.y, i32 %gridDim.z, i32 %blockDim.x, i32 %blockDim.y, i32 %blockDim.z, i32 %blockIdx.x, i32 %blockIdx.y, i32 %blockIdx.z, i32 %threadIdx.x, i32 %threadIdx.y, i32 %threadIdx.z) #2 {
entry:
  call void @llvm.dbg.value(metadata %struct.dcomplex* %x_in, metadata !4039, metadata !DIExpression()), !dbg !4041
  call void @llvm.dbg.value(metadata %struct.dcomplex* %y0, metadata !4042, metadata !DIExpression()), !dbg !4041
  %mul = mul i32 %blockIdx.x, %blockDim.x, !dbg !4043
  %add = add i32 %mul, %threadIdx.x, !dbg !4044
  call void @llvm.dbg.value(metadata i32 %add, metadata !4045, metadata !DIExpression()), !dbg !4041
  %cmp = icmp sge i32 %add, 8388608, !dbg !4046
  br i1 %cmp, label %if.then, label %if.end, !dbg !4048

if.then:                                          ; preds = %entry
  br label %return, !dbg !4049

if.end:                                           ; preds = %entry
  %idxprom = sext i32 %add to i64, !dbg !4051
  %arrayidx = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %x_in, i64 %idxprom, !dbg !4051
  %real = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx, i32 0, i32 0, !dbg !4052
  %0 = load double, double* %real, align 8, !dbg !4052
  %idxprom3 = sext i32 %add to i64, !dbg !4053
  %arrayidx4 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %y0, i64 %idxprom3, !dbg !4053
  %real5 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx4, i32 0, i32 0, !dbg !4054
  store double %0, double* %real5, align 8, !dbg !4055
  %idxprom6 = sext i32 %add to i64, !dbg !4056
  %arrayidx7 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %x_in, i64 %idxprom6, !dbg !4056
  %imag = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx7, i32 0, i32 1, !dbg !4057
  %1 = load double, double* %imag, align 8, !dbg !4057
  %idxprom8 = sext i32 %add to i64, !dbg !4058
  %arrayidx9 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %y0, i64 %idxprom8, !dbg !4058
  %imag10 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx9, i32 0, i32 1, !dbg !4059
  store double %1, double* %imag10, align 8, !dbg !4060
  br label %return, !dbg !4061

return:                                           ; preds = %if.end, %if.then
  ret void, !dbg !4061
}

; Function Attrs: convergent noinline nounwind
define dso_local void @cffts3_gpu_kernel_2(i32 %is, %struct.dcomplex* %gty1, %struct.dcomplex* %gty2, %struct.dcomplex* %u_device, i32 %gridDim.x, i32 %gridDim.y, i32 %gridDim.z, i32 %blockDim.x, i32 %blockDim.y, i32 %blockDim.z, i32 %blockIdx.x, i32 %blockIdx.y, i32 %blockIdx.z, i32 %threadIdx.x, i32 %threadIdx.y, i32 %threadIdx.z) #2 {
entry:
  call void @llvm.dbg.value(metadata i32 %is, metadata !4062, metadata !DIExpression()), !dbg !4064
  call void @llvm.dbg.value(metadata %struct.dcomplex* %gty1, metadata !4065, metadata !DIExpression()), !dbg !4064
  call void @llvm.dbg.value(metadata %struct.dcomplex* %gty2, metadata !4066, metadata !DIExpression()), !dbg !4064
  call void @llvm.dbg.value(metadata %struct.dcomplex* %u_device, metadata !4067, metadata !DIExpression()), !dbg !4064
  %mul = mul i32 %blockIdx.x, %blockDim.x, !dbg !4068
  %add = add i32 %mul, %threadIdx.x, !dbg !4069
  call void @llvm.dbg.value(metadata i32 %add, metadata !4070, metadata !DIExpression()), !dbg !4064
  %cmp = icmp sge i32 %add, 65536, !dbg !4071
  br i1 %cmp, label %if.then, label %if.end, !dbg !4073

if.then:                                          ; preds = %entry
  br label %return, !dbg !4074

if.end:                                           ; preds = %entry
  %call3 = call i32 @_Z12ilog2_devicei(i32 128) #4, !dbg !4076
  call void @_Z23cffts3_gpu_cfftz_deviceiiiP8dcomplexS0_S0_ii(i32 %is, i32 %call3, i32 128, %struct.dcomplex* %gty1, %struct.dcomplex* %gty2, %struct.dcomplex* %u_device, i32 %add, i32 65536) #4, !dbg !4077
  br label %return, !dbg !4078

return:                                           ; preds = %if.end, %if.then
  ret void, !dbg !4078
}

; Function Attrs: convergent noinline nounwind
define dso_local void @cffts3_gpu_kernel_3(%struct.dcomplex* %x_out, %struct.dcomplex* %y0, i32 %gridDim.x, i32 %gridDim.y, i32 %gridDim.z, i32 %blockDim.x, i32 %blockDim.y, i32 %blockDim.z, i32 %blockIdx.x, i32 %blockIdx.y, i32 %blockIdx.z, i32 %threadIdx.x, i32 %threadIdx.y, i32 %threadIdx.z) #2 {
entry:
  call void @llvm.dbg.value(metadata %struct.dcomplex* %x_out, metadata !4079, metadata !DIExpression()), !dbg !4081
  call void @llvm.dbg.value(metadata %struct.dcomplex* %y0, metadata !4082, metadata !DIExpression()), !dbg !4081
  %mul = mul i32 %blockIdx.x, %blockDim.x, !dbg !4083
  %add = add i32 %mul, %threadIdx.x, !dbg !4084
  call void @llvm.dbg.value(metadata i32 %add, metadata !4085, metadata !DIExpression()), !dbg !4081
  %cmp = icmp sge i32 %add, 8388608, !dbg !4086
  br i1 %cmp, label %if.then, label %if.end, !dbg !4088

if.then:                                          ; preds = %entry
  br label %return, !dbg !4089

if.end:                                           ; preds = %entry
  %idxprom = sext i32 %add to i64, !dbg !4091
  %arrayidx = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %y0, i64 %idxprom, !dbg !4091
  %real = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx, i32 0, i32 0, !dbg !4092
  %0 = load double, double* %real, align 8, !dbg !4092
  %idxprom3 = sext i32 %add to i64, !dbg !4093
  %arrayidx4 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %x_out, i64 %idxprom3, !dbg !4093
  %real5 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx4, i32 0, i32 0, !dbg !4094
  store double %0, double* %real5, align 8, !dbg !4095
  %idxprom6 = sext i32 %add to i64, !dbg !4096
  %arrayidx7 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %y0, i64 %idxprom6, !dbg !4096
  %imag = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx7, i32 0, i32 1, !dbg !4097
  %1 = load double, double* %imag, align 8, !dbg !4097
  %idxprom8 = sext i32 %add to i64, !dbg !4098
  %arrayidx9 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %x_out, i64 %idxprom8, !dbg !4098
  %imag10 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx9, i32 0, i32 1, !dbg !4099
  store double %1, double* %imag10, align 8, !dbg !4100
  br label %return, !dbg !4101

return:                                           ; preds = %if.end, %if.then
  ret void, !dbg !4101
}

; Function Attrs: convergent noinline nounwind
define dso_local void @checksum_gpu_kernel1(i32 %iteration, %struct.dcomplex* %u1, %struct.dcomplex* %sums, i32 %gridDim.x, i32 %gridDim.y, i32 %gridDim.z, i32 %blockDim.x, i32 %blockDim.y, i32 %blockDim.z, i32 %blockIdx.x, i32 %blockIdx.y, i32 %blockIdx.z, i32 %threadIdx.x, i32 %threadIdx.y, i32 %threadIdx.z) #2 {
entry:
  %ref.tmp24 = alloca %struct.dcomplex, align 8
  call void @llvm.dbg.value(metadata i32 %iteration, metadata !3171, metadata !DIExpression()), !dbg !3173
  call void @llvm.dbg.value(metadata %struct.dcomplex* %u1, metadata !3174, metadata !DIExpression()), !dbg !3173
  call void @llvm.dbg.value(metadata %struct.dcomplex* %sums, metadata !3175, metadata !DIExpression()), !dbg !3173
  call void @llvm.dbg.value(metadata %struct.dcomplex* bitcast ([1024 x double]* @extern_share_data_shared to %struct.dcomplex*), metadata !3176, metadata !DIExpression()), !dbg !3173
  call void @llvm.dbg.value(metadata !1051, metadata !3180, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !3173
  br label %syncpoint.1, !dbg !3183

syncpoint.1:                                      ; preds = %entry
  call void @llvm.nvvm.barrier0(), !dbg !3206
  %cmp20 = icmp eq i32 %threadIdx.x, 0, !dbg !4102
  br i1 %cmp20, label %if.then21, label %if.end40, !dbg !4104

if.then21:                                        ; preds = %syncpoint.1
  call void @llvm.dbg.value(metadata i32 1, metadata !4105, metadata !DIExpression()), !dbg !4108
  %0 = zext i32 %blockDim.x to i64, !dbg !4109
  br label %for.cond, !dbg !4109

for.cond:                                         ; preds = %for.inc, %if.then21
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 1, %if.then21 ], !dbg !4108
  call void @llvm.dbg.value(metadata i64 %indvars.iv, metadata !4105, metadata !DIExpression()), !dbg !4108
  %cmp23 = icmp ult i64 %indvars.iv, %0, !dbg !4110
  br i1 %cmp23, label %for.body, label %for.end, !dbg !4112

for.body:                                         ; preds = %for.cond
  %real25 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp24, i32 0, i32 0, !dbg !4113
  %arrayidx26 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* bitcast ([1024 x double]* @extern_share_data_shared to %struct.dcomplex*), i64 0, !dbg !4113
  %real27 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx26, i32 0, i32 0, !dbg !4113
  %1 = load double, double* %real27, align 8, !dbg !4113
  %arrayidx29 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* bitcast ([1024 x double]* @extern_share_data_shared to %struct.dcomplex*), i64 %indvars.iv, !dbg !4113
  %real30 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx29, i32 0, i32 0, !dbg !4113
  %2 = load double, double* %real30, align 8, !dbg !4113
  %add31 = fadd contract double %1, %2, !dbg !4113
  store double %add31, double* %real25, align 8, !dbg !4113
  %imag32 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp24, i32 0, i32 1, !dbg !4113
  %arrayidx33 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* bitcast ([1024 x double]* @extern_share_data_shared to %struct.dcomplex*), i64 0, !dbg !4113
  %imag34 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx33, i32 0, i32 1, !dbg !4113
  %3 = load double, double* %imag34, align 8, !dbg !4113
  %arrayidx36 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* bitcast ([1024 x double]* @extern_share_data_shared to %struct.dcomplex*), i64 %indvars.iv, !dbg !4113
  %imag37 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx36, i32 0, i32 1, !dbg !4113
  %4 = load double, double* %imag37, align 8, !dbg !4113
  %add38 = fadd contract double %3, %4, !dbg !4113
  store double %add38, double* %imag32, align 8, !dbg !4113
  %arrayidx39 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* bitcast ([1024 x double]* @extern_share_data_shared to %struct.dcomplex*), i64 0, !dbg !4115
  %5 = bitcast %struct.dcomplex* %arrayidx39 to i8*, !dbg !4116
  %6 = bitcast %struct.dcomplex* %ref.tmp24 to i8*, !dbg !4116
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %5, i8* align 8 %6, i64 16, i1 false), !dbg !4116
  br label %for.inc, !dbg !4117

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !4118
  call void @llvm.dbg.value(metadata i32 undef, metadata !4105, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !4108
  br label %for.cond, !dbg !4119, !llvm.loop !4120

for.end:                                          ; preds = %for.cond
  br label %if.end40, !dbg !4122

if.end40:                                         ; preds = %for.end, %syncpoint.1
  %cmp42 = icmp eq i32 %threadIdx.x, 0, !dbg !4123
  br i1 %cmp42, label %if.then43, label %if.end65, !dbg !4125

if.then43:                                        ; preds = %if.end40
  %arrayidx44 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* bitcast ([1024 x double]* @extern_share_data_shared to %struct.dcomplex*), i64 0, !dbg !4126
  %real45 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx44, i32 0, i32 0, !dbg !4128
  %7 = load double, double* %real45, align 8, !dbg !4128
  %div = fdiv double %7, 0x4160000000000000, !dbg !4129
  %arrayidx46 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* bitcast ([1024 x double]* @extern_share_data_shared to %struct.dcomplex*), i64 0, !dbg !4130
  %real47 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx46, i32 0, i32 0, !dbg !4131
  store double %div, double* %real47, align 8, !dbg !4132
  %idxprom48 = sext i32 %iteration to i64, !dbg !4133
  %arrayidx49 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %sums, i64 %idxprom48, !dbg !4133
  %real50 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx49, i32 0, i32 0, !dbg !4134
  %arrayidx51 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* bitcast ([1024 x double]* @extern_share_data_shared to %struct.dcomplex*), i64 0, !dbg !4135
  %real52 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx51, i32 0, i32 0, !dbg !4136
  %8 = load double, double* %real52, align 8, !dbg !4136
  %call53 = call double @_ZL9atomicAddPdd(double* %real50, double %8) #4, !dbg !4137, !tulip.atomic.add !1860
  %arrayidx54 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* bitcast ([1024 x double]* @extern_share_data_shared to %struct.dcomplex*), i64 0, !dbg !4138
  %imag55 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx54, i32 0, i32 1, !dbg !4139
  %9 = load double, double* %imag55, align 8, !dbg !4139
  %div56 = fdiv double %9, 0x4160000000000000, !dbg !4140
  %arrayidx57 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* bitcast ([1024 x double]* @extern_share_data_shared to %struct.dcomplex*), i64 0, !dbg !4141
  %imag58 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx57, i32 0, i32 1, !dbg !4142
  store double %div56, double* %imag58, align 8, !dbg !4143
  %idxprom59 = sext i32 %iteration to i64, !dbg !4144
  %arrayidx60 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %sums, i64 %idxprom59, !dbg !4144
  %imag61 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx60, i32 0, i32 1, !dbg !4145
  %arrayidx62 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* bitcast ([1024 x double]* @extern_share_data_shared to %struct.dcomplex*), i64 0, !dbg !4146
  %imag63 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx62, i32 0, i32 1, !dbg !4147
  %10 = load double, double* %imag63, align 8, !dbg !4147
  %call64 = call double @_ZL9atomicAddPdd(double* %imag61, double %10) #4, !dbg !4148, !tulip.atomic.add !1860
  br label %if.end65, !dbg !4149

if.end65:                                         ; preds = %if.then43, %if.end40
  ret void, !dbg !4150
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #0

attributes #0 = { nounwind readnone speculatable }
attributes #1 = { nounwind readnone }
attributes #2 = { convergent noinline nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { argmemonly nounwind }
attributes #4 = { convergent nounwind }
attributes #5 = { convergent noinline nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #7 = { noinline uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #8 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #9 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #10 = { noinline norecurse uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
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
!1127 = distinct !{null, !"kernel", i32 1}
!1128 = distinct !{null, !"kernel", i32 1}
!1129 = distinct !{null, !"kernel", i32 1}
!1130 = distinct !{null, !"kernel", i32 1}
!1131 = distinct !{null, !"kernel", i32 1}
!1132 = distinct !{null, !"kernel", i32 1}
!1133 = distinct !{null, !"kernel", i32 1}
!1134 = distinct !{null, !"kernel", i32 1}
!1135 = distinct !{null, !"kernel", i32 1}
!1136 = distinct !{null, !"kernel", i32 1}
!1137 = distinct !{null, !"kernel", i32 1}
!1138 = distinct !{null, !"kernel", i32 1}
!1139 = distinct !{null, !"kernel", i32 1}
!1140 = distinct !{null, !"kernel", i32 1}
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
!1152 = distinct !DISubprogram(name: "ilog2_device", linkageName: "_Z12ilog2_devicei", scope: !3, file: !3, line: 1524, type: !304, scopeLine: 1524, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !1050, retainedNodes: !1051)
!1153 = !DILocalVariable(name: "n", arg: 1, scope: !1152, file: !3, line: 1524, type: !97)
!1154 = !DILocation(line: 0, scope: !1152)
!1155 = !DILocation(line: 1526, column: 6, scope: !1156)
!1156 = distinct !DILexicalBlock(scope: !1152, file: !3, line: 1526, column: 5)
!1157 = !DILocation(line: 1526, column: 5, scope: !1152)
!1158 = !DILocation(line: 1527, column: 3, scope: !1159)
!1159 = distinct !DILexicalBlock(scope: !1156, file: !3, line: 1526, column: 10)
!1160 = !DILocalVariable(name: "lg", scope: !1152, file: !3, line: 1525, type: !97)
!1161 = !DILocalVariable(name: "nn", scope: !1152, file: !3, line: 1525, type: !97)
!1162 = !DILocation(line: 1531, column: 2, scope: !1152)
!1163 = !DILocation(line: 1531, column: 10, scope: !1152)
!1164 = !DILocation(line: 1532, column: 11, scope: !1165)
!1165 = distinct !DILexicalBlock(scope: !1152, file: !3, line: 1531, column: 13)
!1166 = !DILocation(line: 1533, column: 5, scope: !1165)
!1167 = distinct !{!1167, !1162, !1168}
!1168 = !DILocation(line: 1534, column: 2, scope: !1152)
!1169 = !DILocation(line: 1535, column: 2, scope: !1152)
!1170 = !DILocation(line: 1536, column: 1, scope: !1152)
!1171 = distinct !DISubprogram(name: "cffts3_gpu_cfftz_device", linkageName: "_Z23cffts3_gpu_cfftz_deviceiiiP8dcomplexS0_S0_ii", scope: !3, file: !3, line: 1162, type: !1172, scopeLine: 1169, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !1050, retainedNodes: !1051)
!1172 = !DISubroutineType(types: !1173)
!1173 = !{null, !1174, !97, !97, !98, !98, !98, !97, !97}
!1174 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !97)
!1175 = !DILocalVariable(name: "is", arg: 1, scope: !1171, file: !3, line: 1162, type: !1174)
!1176 = !DILocation(line: 0, scope: !1171)
!1177 = !DILocalVariable(name: "m", arg: 2, scope: !1171, file: !3, line: 1163, type: !97)
!1178 = !DILocalVariable(name: "n", arg: 3, scope: !1171, file: !3, line: 1164, type: !97)
!1179 = !DILocalVariable(name: "x", arg: 4, scope: !1171, file: !3, line: 1165, type: !98)
!1180 = !DILocalVariable(name: "y", arg: 5, scope: !1171, file: !3, line: 1166, type: !98)
!1181 = !DILocalVariable(name: "u_device", arg: 6, scope: !1171, file: !3, line: 1167, type: !98)
!1182 = !DILocalVariable(name: "index_arg", arg: 7, scope: !1171, file: !3, line: 1168, type: !97)
!1183 = !DILocalVariable(name: "size_arg", arg: 8, scope: !1171, file: !3, line: 1169, type: !97)
!1184 = !DILocalVariable(name: "l", scope: !1171, file: !3, line: 1170, type: !97)
!1185 = !DILocation(line: 1176, column: 6, scope: !1186)
!1186 = distinct !DILexicalBlock(scope: !1171, file: !3, line: 1176, column: 2)
!1187 = !DILocation(line: 0, scope: !1186)
!1188 = !DILocation(line: 1176, column: 12, scope: !1189)
!1189 = distinct !DILexicalBlock(scope: !1186, file: !3, line: 1176, column: 2)
!1190 = !DILocation(line: 1176, column: 2, scope: !1186)
!1191 = !DILocation(line: 1177, column: 3, scope: !1192)
!1192 = distinct !DILexicalBlock(scope: !1189, file: !3, line: 1176, column: 22)
!1193 = !DILocation(line: 1178, column: 7, scope: !1194)
!1194 = distinct !DILexicalBlock(scope: !1192, file: !3, line: 1178, column: 6)
!1195 = !DILocation(line: 1178, column: 6, scope: !1192)
!1196 = !DILocation(line: 1178, column: 12, scope: !1197)
!1197 = distinct !DILexicalBlock(scope: !1194, file: !3, line: 1178, column: 11)
!1198 = !DILocation(line: 1179, column: 33, scope: !1192)
!1199 = !DILocation(line: 1179, column: 3, scope: !1192)
!1200 = !DILocation(line: 1180, column: 2, scope: !1192)
!1201 = !DILocation(line: 1176, column: 18, scope: !1189)
!1202 = !DILocation(line: 1176, column: 2, scope: !1189)
!1203 = distinct !{!1203, !1190, !1204}
!1204 = !DILocation(line: 1180, column: 2, scope: !1186)
!1205 = !DILocation(line: 1186, column: 6, scope: !1206)
!1206 = distinct !DILexicalBlock(scope: !1171, file: !3, line: 1186, column: 5)
!1207 = !DILocation(line: 1186, column: 8, scope: !1206)
!1208 = !DILocation(line: 1186, column: 5, scope: !1171)
!1209 = !DILocalVariable(name: "j", scope: !1171, file: !3, line: 1170, type: !97)
!1210 = !DILocation(line: 1187, column: 7, scope: !1211)
!1211 = distinct !DILexicalBlock(scope: !1212, file: !3, line: 1187, column: 3)
!1212 = distinct !DILexicalBlock(scope: !1206, file: !3, line: 1186, column: 12)
!1213 = !DILocation(line: 0, scope: !1211)
!1214 = !DILocation(line: 1187, column: 13, scope: !1215)
!1215 = distinct !DILexicalBlock(scope: !1211, file: !3, line: 1187, column: 3)
!1216 = !DILocation(line: 1187, column: 3, scope: !1211)
!1217 = !DILocation(line: 1188, column: 38, scope: !1218)
!1218 = distinct !DILexicalBlock(scope: !1215, file: !3, line: 1187, column: 21)
!1219 = !DILocation(line: 1188, column: 47, scope: !1218)
!1220 = !DILocation(line: 1188, column: 35, scope: !1218)
!1221 = !DILocation(line: 1188, column: 59, scope: !1218)
!1222 = !DILocation(line: 1188, column: 7, scope: !1218)
!1223 = !DILocation(line: 1188, column: 16, scope: !1218)
!1224 = !DILocation(line: 1188, column: 4, scope: !1218)
!1225 = !DILocation(line: 1188, column: 28, scope: !1218)
!1226 = !DILocation(line: 1188, column: 33, scope: !1218)
!1227 = !DILocation(line: 1189, column: 38, scope: !1218)
!1228 = !DILocation(line: 1189, column: 47, scope: !1218)
!1229 = !DILocation(line: 1189, column: 35, scope: !1218)
!1230 = !DILocation(line: 1189, column: 59, scope: !1218)
!1231 = !DILocation(line: 1189, column: 7, scope: !1218)
!1232 = !DILocation(line: 1189, column: 16, scope: !1218)
!1233 = !DILocation(line: 1189, column: 4, scope: !1218)
!1234 = !DILocation(line: 1189, column: 28, scope: !1218)
!1235 = !DILocation(line: 1189, column: 33, scope: !1218)
!1236 = !DILocation(line: 1190, column: 3, scope: !1218)
!1237 = !DILocation(line: 1187, column: 18, scope: !1215)
!1238 = !DILocation(line: 1187, column: 3, scope: !1215)
!1239 = distinct !{!1239, !1216, !1240}
!1240 = !DILocation(line: 1190, column: 3, scope: !1211)
!1241 = !DILocation(line: 1191, column: 2, scope: !1212)
!1242 = !DILocation(line: 1192, column: 1, scope: !1171)
!1243 = distinct !DISubprogram(name: "cffts3_gpu_fftz2_device", linkageName: "_Z23cffts3_gpu_fftz2_deviceiiiiP8dcomplexS0_S0_ii", scope: !3, file: !3, line: 1203, type: !1244, scopeLine: 1211, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !1050, retainedNodes: !1051)
!1244 = !DISubroutineType(types: !1245)
!1245 = !{null, !1174, !97, !97, !97, !98, !98, !98, !97, !97}
!1246 = !DILocalVariable(name: "is", arg: 1, scope: !1243, file: !3, line: 1203, type: !1174)
!1247 = !DILocation(line: 0, scope: !1243)
!1248 = !DILocalVariable(name: "l", arg: 2, scope: !1243, file: !3, line: 1204, type: !97)
!1249 = !DILocalVariable(name: "m", arg: 3, scope: !1243, file: !3, line: 1205, type: !97)
!1250 = !DILocalVariable(name: "n", arg: 4, scope: !1243, file: !3, line: 1206, type: !97)
!1251 = !DILocalVariable(name: "u", arg: 5, scope: !1243, file: !3, line: 1207, type: !98)
!1252 = !DILocalVariable(name: "x", arg: 6, scope: !1243, file: !3, line: 1208, type: !98)
!1253 = !DILocalVariable(name: "y", arg: 7, scope: !1243, file: !3, line: 1209, type: !98)
!1254 = !DILocalVariable(name: "index_arg", arg: 8, scope: !1243, file: !3, line: 1210, type: !97)
!1255 = !DILocalVariable(name: "size_arg", arg: 9, scope: !1243, file: !3, line: 1211, type: !97)
!1256 = !DILocalVariable(name: "u1", scope: !1243, file: !3, line: 1215, type: !99)
!1257 = !DILocation(line: 1215, column: 11, scope: !1243)
!1258 = !DILocation(line: 1221, column: 9, scope: !1243)
!1259 = !DILocalVariable(name: "n1", scope: !1243, file: !3, line: 1212, type: !97)
!1260 = !DILocation(line: 1222, column: 15, scope: !1243)
!1261 = !DILocation(line: 1222, column: 9, scope: !1243)
!1262 = !DILocalVariable(name: "lk", scope: !1243, file: !3, line: 1212, type: !97)
!1263 = !DILocation(line: 1223, column: 15, scope: !1243)
!1264 = !DILocation(line: 1223, column: 9, scope: !1243)
!1265 = !DILocalVariable(name: "li", scope: !1243, file: !3, line: 1212, type: !97)
!1266 = !DILocation(line: 1224, column: 9, scope: !1243)
!1267 = !DILocalVariable(name: "lj", scope: !1243, file: !3, line: 1212, type: !97)
!1268 = !DILocalVariable(name: "ku", scope: !1243, file: !3, line: 1212, type: !97)
!1269 = !DILocalVariable(name: "i", scope: !1243, file: !3, line: 1212, type: !97)
!1270 = !DILocation(line: 1226, column: 6, scope: !1271)
!1271 = distinct !DILexicalBlock(scope: !1243, file: !3, line: 1226, column: 2)
!1272 = !DILocation(line: 0, scope: !1271)
!1273 = !DILocation(line: 1226, column: 12, scope: !1274)
!1274 = distinct !DILexicalBlock(scope: !1271, file: !3, line: 1226, column: 2)
!1275 = !DILocation(line: 1226, column: 2, scope: !1271)
!1276 = !DILocation(line: 1227, column: 11, scope: !1277)
!1277 = distinct !DILexicalBlock(scope: !1274, file: !3, line: 1226, column: 21)
!1278 = !DILocation(line: 1228, column: 13, scope: !1277)
!1279 = !DILocation(line: 1229, column: 11, scope: !1277)
!1280 = !DILocation(line: 1230, column: 13, scope: !1277)
!1281 = !DILocation(line: 1231, column: 8, scope: !1282)
!1282 = distinct !DILexicalBlock(scope: !1277, file: !3, line: 1231, column: 6)
!1283 = !DILocation(line: 1231, column: 6, scope: !1277)
!1284 = !DILocation(line: 1232, column: 18, scope: !1285)
!1285 = distinct !DILexicalBlock(scope: !1282, file: !3, line: 1231, column: 12)
!1286 = !DILocation(line: 1232, column: 14, scope: !1285)
!1287 = !DILocation(line: 1232, column: 22, scope: !1285)
!1288 = !DILocation(line: 1232, column: 7, scope: !1285)
!1289 = !DILocation(line: 1232, column: 12, scope: !1285)
!1290 = !DILocation(line: 1233, column: 18, scope: !1285)
!1291 = !DILocation(line: 1233, column: 14, scope: !1285)
!1292 = !DILocation(line: 1233, column: 22, scope: !1285)
!1293 = !DILocation(line: 1233, column: 7, scope: !1285)
!1294 = !DILocation(line: 1233, column: 12, scope: !1285)
!1295 = !DILocation(line: 1234, column: 3, scope: !1285)
!1296 = !DILocation(line: 1235, column: 18, scope: !1297)
!1297 = distinct !DILexicalBlock(scope: !1282, file: !3, line: 1234, column: 8)
!1298 = !DILocation(line: 1235, column: 14, scope: !1297)
!1299 = !DILocation(line: 1235, column: 22, scope: !1297)
!1300 = !DILocation(line: 1235, column: 7, scope: !1297)
!1301 = !DILocation(line: 1235, column: 12, scope: !1297)
!1302 = !DILocation(line: 1236, column: 19, scope: !1297)
!1303 = !DILocation(line: 1236, column: 15, scope: !1297)
!1304 = !DILocation(line: 1236, column: 23, scope: !1297)
!1305 = !DILocation(line: 1236, column: 14, scope: !1297)
!1306 = !DILocation(line: 1236, column: 7, scope: !1297)
!1307 = !DILocation(line: 1236, column: 12, scope: !1297)
!1308 = !DILocalVariable(name: "k", scope: !1243, file: !3, line: 1212, type: !97)
!1309 = !DILocation(line: 1238, column: 7, scope: !1310)
!1310 = distinct !DILexicalBlock(scope: !1277, file: !3, line: 1238, column: 3)
!1311 = !DILocation(line: 0, scope: !1310)
!1312 = !DILocation(line: 1238, column: 13, scope: !1313)
!1313 = distinct !DILexicalBlock(scope: !1310, file: !3, line: 1238, column: 3)
!1314 = !DILocation(line: 1238, column: 3, scope: !1310)
!1315 = !DILocation(line: 1239, column: 20, scope: !1316)
!1316 = distinct !DILexicalBlock(scope: !1313, file: !3, line: 1238, column: 22)
!1317 = !DILocation(line: 1239, column: 23, scope: !1316)
!1318 = !DILocation(line: 1239, column: 32, scope: !1316)
!1319 = !DILocation(line: 1239, column: 14, scope: !1316)
!1320 = !DILocation(line: 1239, column: 44, scope: !1316)
!1321 = !DILocalVariable(name: "x11real", scope: !1243, file: !3, line: 1213, type: !104)
!1322 = !DILocation(line: 1240, column: 20, scope: !1316)
!1323 = !DILocation(line: 1240, column: 23, scope: !1316)
!1324 = !DILocation(line: 1240, column: 32, scope: !1316)
!1325 = !DILocation(line: 1240, column: 14, scope: !1316)
!1326 = !DILocation(line: 1240, column: 44, scope: !1316)
!1327 = !DILocalVariable(name: "x11imag", scope: !1243, file: !3, line: 1213, type: !104)
!1328 = !DILocation(line: 1241, column: 20, scope: !1316)
!1329 = !DILocation(line: 1241, column: 23, scope: !1316)
!1330 = !DILocation(line: 1241, column: 32, scope: !1316)
!1331 = !DILocation(line: 1241, column: 14, scope: !1316)
!1332 = !DILocation(line: 1241, column: 44, scope: !1316)
!1333 = !DILocalVariable(name: "x21real", scope: !1243, file: !3, line: 1214, type: !104)
!1334 = !DILocation(line: 1242, column: 20, scope: !1316)
!1335 = !DILocation(line: 1242, column: 23, scope: !1316)
!1336 = !DILocation(line: 1242, column: 32, scope: !1316)
!1337 = !DILocation(line: 1242, column: 14, scope: !1316)
!1338 = !DILocation(line: 1242, column: 44, scope: !1316)
!1339 = !DILocalVariable(name: "x21imag", scope: !1243, file: !3, line: 1214, type: !104)
!1340 = !DILocation(line: 1243, column: 49, scope: !1316)
!1341 = !DILocation(line: 1243, column: 10, scope: !1316)
!1342 = !DILocation(line: 1243, column: 13, scope: !1316)
!1343 = !DILocation(line: 1243, column: 22, scope: !1316)
!1344 = !DILocation(line: 1243, column: 4, scope: !1316)
!1345 = !DILocation(line: 1243, column: 34, scope: !1316)
!1346 = !DILocation(line: 1243, column: 39, scope: !1316)
!1347 = !DILocation(line: 1244, column: 49, scope: !1316)
!1348 = !DILocation(line: 1244, column: 10, scope: !1316)
!1349 = !DILocation(line: 1244, column: 13, scope: !1316)
!1350 = !DILocation(line: 1244, column: 22, scope: !1316)
!1351 = !DILocation(line: 1244, column: 4, scope: !1316)
!1352 = !DILocation(line: 1244, column: 34, scope: !1316)
!1353 = !DILocation(line: 1244, column: 39, scope: !1316)
!1354 = !DILocation(line: 1245, column: 44, scope: !1316)
!1355 = !DILocation(line: 1245, column: 60, scope: !1316)
!1356 = !DILocation(line: 1245, column: 49, scope: !1316)
!1357 = !DILocation(line: 1245, column: 76, scope: !1316)
!1358 = !DILocation(line: 1245, column: 92, scope: !1316)
!1359 = !DILocation(line: 1245, column: 81, scope: !1316)
!1360 = !DILocation(line: 1245, column: 71, scope: !1316)
!1361 = !DILocation(line: 1245, column: 10, scope: !1316)
!1362 = !DILocation(line: 1245, column: 13, scope: !1316)
!1363 = !DILocation(line: 1245, column: 22, scope: !1316)
!1364 = !DILocation(line: 1245, column: 4, scope: !1316)
!1365 = !DILocation(line: 1245, column: 34, scope: !1316)
!1366 = !DILocation(line: 1245, column: 39, scope: !1316)
!1367 = !DILocation(line: 1246, column: 44, scope: !1316)
!1368 = !DILocation(line: 1246, column: 60, scope: !1316)
!1369 = !DILocation(line: 1246, column: 49, scope: !1316)
!1370 = !DILocation(line: 1246, column: 76, scope: !1316)
!1371 = !DILocation(line: 1246, column: 92, scope: !1316)
!1372 = !DILocation(line: 1246, column: 81, scope: !1316)
!1373 = !DILocation(line: 1246, column: 71, scope: !1316)
!1374 = !DILocation(line: 1246, column: 10, scope: !1316)
!1375 = !DILocation(line: 1246, column: 13, scope: !1316)
!1376 = !DILocation(line: 1246, column: 22, scope: !1316)
!1377 = !DILocation(line: 1246, column: 4, scope: !1316)
!1378 = !DILocation(line: 1246, column: 34, scope: !1316)
!1379 = !DILocation(line: 1246, column: 39, scope: !1316)
!1380 = !DILocation(line: 1247, column: 3, scope: !1316)
!1381 = !DILocation(line: 1238, column: 19, scope: !1313)
!1382 = !DILocation(line: 1238, column: 3, scope: !1313)
!1383 = distinct !{!1383, !1314, !1384}
!1384 = !DILocation(line: 1247, column: 3, scope: !1310)
!1385 = !DILocation(line: 1248, column: 2, scope: !1277)
!1386 = !DILocation(line: 1226, column: 18, scope: !1274)
!1387 = !DILocation(line: 1226, column: 2, scope: !1274)
!1388 = distinct !{!1388, !1275, !1389}
!1389 = !DILocation(line: 1248, column: 2, scope: !1271)
!1390 = !DILocation(line: 1249, column: 1, scope: !1243)
!1391 = distinct !DISubprogram(name: "vranlc_device", linkageName: "_Z13vranlc_deviceiPddS_", scope: !3, file: !3, line: 2034, type: !1392, scopeLine: 2037, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !1050, retainedNodes: !1051)
!1392 = !DISubroutineType(types: !1393)
!1393 = !{null, !97, !106, !104, !106}
!1394 = !DILocalVariable(name: "n", arg: 1, scope: !1391, file: !3, line: 2034, type: !97)
!1395 = !DILocation(line: 0, scope: !1391)
!1396 = !DILocalVariable(name: "x_seed", arg: 2, scope: !1391, file: !3, line: 2035, type: !106)
!1397 = !DILocalVariable(name: "a", arg: 3, scope: !1391, file: !3, line: 2036, type: !104)
!1398 = !DILocalVariable(name: "y", arg: 4, scope: !1391, file: !3, line: 2037, type: !106)
!1399 = !DILocation(line: 2040, column: 11, scope: !1391)
!1400 = !DILocalVariable(name: "t1", scope: !1391, file: !3, line: 2039, type: !104)
!1401 = !DILocation(line: 2041, column: 12, scope: !1391)
!1402 = !DILocation(line: 2041, column: 7, scope: !1391)
!1403 = !DILocalVariable(name: "a1", scope: !1391, file: !3, line: 2039, type: !104)
!1404 = !DILocation(line: 2042, column: 15, scope: !1391)
!1405 = !DILocation(line: 2042, column: 9, scope: !1391)
!1406 = !DILocalVariable(name: "a2", scope: !1391, file: !3, line: 2039, type: !104)
!1407 = !DILocation(line: 2043, column: 6, scope: !1391)
!1408 = !DILocalVariable(name: "x", scope: !1391, file: !3, line: 2039, type: !104)
!1409 = !DILocalVariable(name: "i", scope: !1391, file: !3, line: 2038, type: !97)
!1410 = !DILocation(line: 2044, column: 6, scope: !1411)
!1411 = distinct !DILexicalBlock(scope: !1391, file: !3, line: 2044, column: 2)
!1412 = !DILocation(line: 0, scope: !1411)
!1413 = !DILocation(line: 2044, column: 12, scope: !1414)
!1414 = distinct !DILexicalBlock(scope: !1411, file: !3, line: 2044, column: 2)
!1415 = !DILocation(line: 2044, column: 2, scope: !1411)
!1416 = !DILocation(line: 2045, column: 12, scope: !1417)
!1417 = distinct !DILexicalBlock(scope: !1414, file: !3, line: 2044, column: 20)
!1418 = !DILocation(line: 2046, column: 13, scope: !1417)
!1419 = !DILocation(line: 2046, column: 8, scope: !1417)
!1420 = !DILocalVariable(name: "x1", scope: !1391, file: !3, line: 2039, type: !104)
!1421 = !DILocation(line: 2047, column: 16, scope: !1417)
!1422 = !DILocation(line: 2047, column: 10, scope: !1417)
!1423 = !DILocalVariable(name: "x2", scope: !1391, file: !3, line: 2039, type: !104)
!1424 = !DILocation(line: 2048, column: 11, scope: !1417)
!1425 = !DILocation(line: 2048, column: 21, scope: !1417)
!1426 = !DILocation(line: 2048, column: 16, scope: !1417)
!1427 = !DILocation(line: 2049, column: 18, scope: !1417)
!1428 = !DILocation(line: 2049, column: 13, scope: !1417)
!1429 = !DILocation(line: 2049, column: 8, scope: !1417)
!1430 = !DILocalVariable(name: "t2", scope: !1391, file: !3, line: 2039, type: !104)
!1431 = !DILocation(line: 2050, column: 16, scope: !1417)
!1432 = !DILocation(line: 2050, column: 10, scope: !1417)
!1433 = !DILocalVariable(name: "z", scope: !1391, file: !3, line: 2039, type: !104)
!1434 = !DILocation(line: 2051, column: 12, scope: !1417)
!1435 = !DILocation(line: 2051, column: 21, scope: !1417)
!1436 = !DILocation(line: 2051, column: 16, scope: !1417)
!1437 = !DILocalVariable(name: "t3", scope: !1391, file: !3, line: 2039, type: !104)
!1438 = !DILocation(line: 2052, column: 18, scope: !1417)
!1439 = !DILocation(line: 2052, column: 13, scope: !1417)
!1440 = !DILocation(line: 2052, column: 8, scope: !1417)
!1441 = !DILocalVariable(name: "t4", scope: !1391, file: !3, line: 2039, type: !104)
!1442 = !DILocation(line: 2053, column: 16, scope: !1417)
!1443 = !DILocation(line: 2053, column: 10, scope: !1417)
!1444 = !DILocation(line: 2054, column: 14, scope: !1417)
!1445 = !DILocation(line: 2054, column: 3, scope: !1417)
!1446 = !DILocation(line: 2054, column: 8, scope: !1417)
!1447 = !DILocation(line: 2055, column: 2, scope: !1417)
!1448 = !DILocation(line: 2044, column: 17, scope: !1414)
!1449 = !DILocation(line: 2044, column: 2, scope: !1414)
!1450 = distinct !{!1450, !1415, !1451}
!1451 = !DILocation(line: 2055, column: 2, scope: !1411)
!1452 = !DILocation(line: 2056, column: 10, scope: !1391)
!1453 = !DILocation(line: 2057, column: 1, scope: !1391)
!1454 = distinct !DISubprogram(name: "ipow46_device", linkageName: "_Z13ipow46_devicediPd", scope: !3, file: !3, line: 1599, type: !1455, scopeLine: 1601, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !1050, retainedNodes: !1051)
!1455 = !DISubroutineType(types: !1456)
!1456 = !{null, !104, !97, !106}
!1457 = !DILocalVariable(name: "a", arg: 1, scope: !1454, file: !3, line: 1599, type: !104)
!1458 = !DILocation(line: 0, scope: !1454)
!1459 = !DILocalVariable(name: "exponent", arg: 2, scope: !1454, file: !3, line: 1600, type: !97)
!1460 = !DILocalVariable(name: "result", arg: 3, scope: !1454, file: !3, line: 1601, type: !106)
!1461 = !DILocalVariable(name: "q", scope: !1454, file: !3, line: 1602, type: !104)
!1462 = !DILocation(line: 1602, column: 9, scope: !1454)
!1463 = !DILocalVariable(name: "r", scope: !1454, file: !3, line: 1602, type: !104)
!1464 = !DILocation(line: 1602, column: 12, scope: !1454)
!1465 = !DILocation(line: 1611, column: 10, scope: !1454)
!1466 = !DILocation(line: 1612, column: 13, scope: !1467)
!1467 = distinct !DILexicalBlock(scope: !1454, file: !3, line: 1612, column: 5)
!1468 = !DILocation(line: 1612, column: 5, scope: !1454)
!1469 = !DILocation(line: 1612, column: 18, scope: !1470)
!1470 = distinct !DILexicalBlock(scope: !1467, file: !3, line: 1612, column: 17)
!1471 = !DILocation(line: 1613, column: 4, scope: !1454)
!1472 = !DILocation(line: 1614, column: 4, scope: !1454)
!1473 = !DILocalVariable(name: "n", scope: !1454, file: !3, line: 1603, type: !97)
!1474 = !DILocation(line: 1616, column: 2, scope: !1454)
!1475 = !DILocation(line: 1616, column: 9, scope: !1454)
!1476 = !DILocation(line: 1617, column: 9, scope: !1477)
!1477 = distinct !DILexicalBlock(scope: !1454, file: !3, line: 1616, column: 12)
!1478 = !DILocalVariable(name: "n2", scope: !1454, file: !3, line: 1603, type: !97)
!1479 = !DILocation(line: 1618, column: 8, scope: !1480)
!1480 = distinct !DILexicalBlock(scope: !1477, file: !3, line: 1618, column: 6)
!1481 = !DILocation(line: 1618, column: 10, scope: !1480)
!1482 = !DILocation(line: 1618, column: 6, scope: !1477)
!1483 = !DILocation(line: 1619, column: 22, scope: !1484)
!1484 = distinct !DILexicalBlock(scope: !1480, file: !3, line: 1618, column: 14)
!1485 = !DILocation(line: 1619, column: 4, scope: !1484)
!1486 = !DILocation(line: 1621, column: 3, scope: !1484)
!1487 = !DILocation(line: 1622, column: 22, scope: !1488)
!1488 = distinct !DILexicalBlock(scope: !1480, file: !3, line: 1621, column: 8)
!1489 = !DILocation(line: 1622, column: 4, scope: !1488)
!1490 = !DILocation(line: 1623, column: 9, scope: !1488)
!1491 = !DILocation(line: 0, scope: !1480)
!1492 = distinct !{!1492, !1474, !1493}
!1493 = !DILocation(line: 1625, column: 2, scope: !1454)
!1494 = !DILocation(line: 1626, column: 20, scope: !1454)
!1495 = !DILocation(line: 1626, column: 2, scope: !1454)
!1496 = !DILocation(line: 1627, column: 12, scope: !1454)
!1497 = !DILocation(line: 1627, column: 10, scope: !1454)
!1498 = !DILocation(line: 1628, column: 1, scope: !1454)
!1499 = distinct !DISubprogram(name: "randlc_device", linkageName: "_Z13randlc_devicePdd", scope: !3, file: !3, line: 1630, type: !1500, scopeLine: 1631, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !1050, retainedNodes: !1051)
!1500 = !DISubroutineType(types: !1501)
!1501 = !{!104, !106, !104}
!1502 = !DILocalVariable(name: "x", arg: 1, scope: !1499, file: !3, line: 1630, type: !106)
!1503 = !DILocation(line: 0, scope: !1499)
!1504 = !DILocalVariable(name: "a", arg: 2, scope: !1499, file: !3, line: 1631, type: !104)
!1505 = !DILocation(line: 1633, column: 11, scope: !1499)
!1506 = !DILocalVariable(name: "t1", scope: !1499, file: !3, line: 1632, type: !104)
!1507 = !DILocation(line: 1634, column: 12, scope: !1499)
!1508 = !DILocation(line: 1634, column: 7, scope: !1499)
!1509 = !DILocalVariable(name: "a1", scope: !1499, file: !3, line: 1632, type: !104)
!1510 = !DILocation(line: 1635, column: 15, scope: !1499)
!1511 = !DILocation(line: 1635, column: 9, scope: !1499)
!1512 = !DILocalVariable(name: "a2", scope: !1499, file: !3, line: 1632, type: !104)
!1513 = !DILocation(line: 1636, column: 14, scope: !1499)
!1514 = !DILocation(line: 1636, column: 11, scope: !1499)
!1515 = !DILocation(line: 1637, column: 12, scope: !1499)
!1516 = !DILocation(line: 1637, column: 7, scope: !1499)
!1517 = !DILocalVariable(name: "x1", scope: !1499, file: !3, line: 1632, type: !104)
!1518 = !DILocation(line: 1638, column: 8, scope: !1499)
!1519 = !DILocation(line: 1638, column: 18, scope: !1499)
!1520 = !DILocation(line: 1638, column: 12, scope: !1499)
!1521 = !DILocalVariable(name: "x2", scope: !1499, file: !3, line: 1632, type: !104)
!1522 = !DILocation(line: 1639, column: 10, scope: !1499)
!1523 = !DILocation(line: 1639, column: 20, scope: !1499)
!1524 = !DILocation(line: 1639, column: 15, scope: !1499)
!1525 = !DILocation(line: 1640, column: 17, scope: !1499)
!1526 = !DILocation(line: 1640, column: 12, scope: !1499)
!1527 = !DILocation(line: 1640, column: 7, scope: !1499)
!1528 = !DILocalVariable(name: "t2", scope: !1499, file: !3, line: 1632, type: !104)
!1529 = !DILocation(line: 1641, column: 15, scope: !1499)
!1530 = !DILocation(line: 1641, column: 9, scope: !1499)
!1531 = !DILocalVariable(name: "z", scope: !1499, file: !3, line: 1632, type: !104)
!1532 = !DILocation(line: 1642, column: 11, scope: !1499)
!1533 = !DILocation(line: 1642, column: 20, scope: !1499)
!1534 = !DILocation(line: 1642, column: 15, scope: !1499)
!1535 = !DILocalVariable(name: "t3", scope: !1499, file: !3, line: 1632, type: !104)
!1536 = !DILocation(line: 1643, column: 17, scope: !1499)
!1537 = !DILocation(line: 1643, column: 12, scope: !1499)
!1538 = !DILocation(line: 1643, column: 7, scope: !1499)
!1539 = !DILocalVariable(name: "t4", scope: !1499, file: !3, line: 1632, type: !104)
!1540 = !DILocation(line: 1644, column: 18, scope: !1499)
!1541 = !DILocation(line: 1644, column: 12, scope: !1499)
!1542 = !DILocation(line: 1644, column: 7, scope: !1499)
!1543 = !DILocation(line: 1645, column: 17, scope: !1499)
!1544 = !DILocation(line: 1645, column: 14, scope: !1499)
!1545 = !DILocation(line: 1645, column: 2, scope: !1499)
!1546 = distinct !DISubprogram(name: "randlc", linkageName: "_Z6randlcPdd", scope: !3, file: !3, line: 371, type: !1500, scopeLine: 371, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !1051)
!1547 = !DILocalVariable(name: "x", arg: 1, scope: !1546, file: !3, line: 371, type: !106)
!1548 = !DILocation(line: 0, scope: !1546)
!1549 = !DILocalVariable(name: "a", arg: 2, scope: !1546, file: !3, line: 371, type: !104)
!1550 = !DILocation(line: 379, column: 12, scope: !1546)
!1551 = !DILocalVariable(name: "t1", scope: !1546, file: !3, line: 372, type: !104)
!1552 = !DILocation(line: 380, column: 13, scope: !1546)
!1553 = !DILocation(line: 380, column: 8, scope: !1546)
!1554 = !DILocalVariable(name: "a1", scope: !1546, file: !3, line: 372, type: !104)
!1555 = !DILocation(line: 381, column: 16, scope: !1546)
!1556 = !DILocation(line: 381, column: 10, scope: !1546)
!1557 = !DILocalVariable(name: "a2", scope: !1546, file: !3, line: 372, type: !104)
!1558 = !DILocation(line: 390, column: 15, scope: !1546)
!1559 = !DILocation(line: 390, column: 12, scope: !1546)
!1560 = !DILocation(line: 391, column: 13, scope: !1546)
!1561 = !DILocation(line: 391, column: 8, scope: !1546)
!1562 = !DILocalVariable(name: "x1", scope: !1546, file: !3, line: 372, type: !104)
!1563 = !DILocation(line: 392, column: 9, scope: !1546)
!1564 = !DILocation(line: 392, column: 19, scope: !1546)
!1565 = !DILocation(line: 392, column: 13, scope: !1546)
!1566 = !DILocalVariable(name: "x2", scope: !1546, file: !3, line: 372, type: !104)
!1567 = !DILocation(line: 393, column: 11, scope: !1546)
!1568 = !DILocation(line: 393, column: 21, scope: !1546)
!1569 = !DILocation(line: 393, column: 16, scope: !1546)
!1570 = !DILocation(line: 394, column: 18, scope: !1546)
!1571 = !DILocation(line: 394, column: 13, scope: !1546)
!1572 = !DILocation(line: 394, column: 8, scope: !1546)
!1573 = !DILocalVariable(name: "t2", scope: !1546, file: !3, line: 372, type: !104)
!1574 = !DILocation(line: 395, column: 16, scope: !1546)
!1575 = !DILocation(line: 395, column: 10, scope: !1546)
!1576 = !DILocalVariable(name: "z", scope: !1546, file: !3, line: 372, type: !104)
!1577 = !DILocation(line: 396, column: 12, scope: !1546)
!1578 = !DILocation(line: 396, column: 21, scope: !1546)
!1579 = !DILocation(line: 396, column: 16, scope: !1546)
!1580 = !DILocalVariable(name: "t3", scope: !1546, file: !3, line: 372, type: !104)
!1581 = !DILocation(line: 397, column: 18, scope: !1546)
!1582 = !DILocation(line: 397, column: 13, scope: !1546)
!1583 = !DILocation(line: 397, column: 8, scope: !1546)
!1584 = !DILocalVariable(name: "t4", scope: !1546, file: !3, line: 372, type: !104)
!1585 = !DILocation(line: 398, column: 19, scope: !1546)
!1586 = !DILocation(line: 398, column: 13, scope: !1546)
!1587 = !DILocation(line: 398, column: 8, scope: !1546)
!1588 = !DILocation(line: 400, column: 18, scope: !1546)
!1589 = !DILocation(line: 400, column: 15, scope: !1546)
!1590 = !DILocation(line: 400, column: 3, scope: !1546)
!1591 = distinct !DISubprogram(name: "c_print_results", linkageName: "_Z15c_print_resultsPcciiiiddS_iS_S_S_S_S_S_S_S_S_S_S_S_S_S_", scope: !3, file: !3, line: 406, type: !1592, scopeLine: 429, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !1051)
!1592 = !DISubroutineType(types: !1593)
!1593 = !{null, !108, !109, !97, !97, !97, !97, !104, !104, !108, !97, !108, !108, !108, !108, !108, !108, !108, !108, !108, !108, !108, !108, !108, !108}
!1594 = !DILocalVariable(name: "name", arg: 1, scope: !1591, file: !3, line: 406, type: !108)
!1595 = !DILocation(line: 0, scope: !1591)
!1596 = !DILocalVariable(name: "class_npb", arg: 2, scope: !1591, file: !3, line: 407, type: !109)
!1597 = !DILocalVariable(name: "n1", arg: 3, scope: !1591, file: !3, line: 408, type: !97)
!1598 = !DILocalVariable(name: "n2", arg: 4, scope: !1591, file: !3, line: 409, type: !97)
!1599 = !DILocalVariable(name: "n3", arg: 5, scope: !1591, file: !3, line: 410, type: !97)
!1600 = !DILocalVariable(name: "niter", arg: 6, scope: !1591, file: !3, line: 411, type: !97)
!1601 = !DILocalVariable(name: "t", arg: 7, scope: !1591, file: !3, line: 412, type: !104)
!1602 = !DILocalVariable(name: "mops", arg: 8, scope: !1591, file: !3, line: 413, type: !104)
!1603 = !DILocalVariable(name: "optype", arg: 9, scope: !1591, file: !3, line: 414, type: !108)
!1604 = !DILocalVariable(name: "passed_verification", arg: 10, scope: !1591, file: !3, line: 415, type: !97)
!1605 = !DILocalVariable(name: "npbversion", arg: 11, scope: !1591, file: !3, line: 416, type: !108)
!1606 = !DILocalVariable(name: "compiletime", arg: 12, scope: !1591, file: !3, line: 417, type: !108)
!1607 = !DILocalVariable(name: "compilerversion", arg: 13, scope: !1591, file: !3, line: 418, type: !108)
!1608 = !DILocalVariable(name: "libversion", arg: 14, scope: !1591, file: !3, line: 419, type: !108)
!1609 = !DILocalVariable(name: "cpu_device", arg: 15, scope: !1591, file: !3, line: 420, type: !108)
!1610 = !DILocalVariable(name: "gpu_device", arg: 16, scope: !1591, file: !3, line: 421, type: !108)
!1611 = !DILocalVariable(name: "gpu_config", arg: 17, scope: !1591, file: !3, line: 422, type: !108)
!1612 = !DILocalVariable(name: "cc", arg: 18, scope: !1591, file: !3, line: 423, type: !108)
!1613 = !DILocalVariable(name: "clink", arg: 19, scope: !1591, file: !3, line: 424, type: !108)
!1614 = !DILocalVariable(name: "c_lib", arg: 20, scope: !1591, file: !3, line: 425, type: !108)
!1615 = !DILocalVariable(name: "c_inc", arg: 21, scope: !1591, file: !3, line: 426, type: !108)
!1616 = !DILocalVariable(name: "cflags", arg: 22, scope: !1591, file: !3, line: 427, type: !108)
!1617 = !DILocalVariable(name: "clinkflags", arg: 23, scope: !1591, file: !3, line: 428, type: !108)
!1618 = !DILocalVariable(name: "rand", arg: 24, scope: !1591, file: !3, line: 429, type: !108)
!1619 = !DILocation(line: 430, column: 5, scope: !1591)
!1620 = !DILocation(line: 431, column: 62, scope: !1591)
!1621 = !DILocation(line: 431, column: 5, scope: !1591)
!1622 = !DILocation(line: 432, column: 9, scope: !1623)
!1623 = distinct !DILexicalBlock(scope: !1591, file: !3, line: 432, column: 8)
!1624 = !DILocation(line: 432, column: 16, scope: !1623)
!1625 = !DILocation(line: 432, column: 22, scope: !1623)
!1626 = !DILocation(line: 432, column: 25, scope: !1623)
!1627 = !DILocation(line: 432, column: 32, scope: !1623)
!1628 = !DILocation(line: 432, column: 8, scope: !1591)
!1629 = !DILocation(line: 433, column: 11, scope: !1630)
!1630 = distinct !DILexicalBlock(scope: !1631, file: !3, line: 433, column: 9)
!1631 = distinct !DILexicalBlock(scope: !1623, file: !3, line: 432, column: 39)
!1632 = !DILocation(line: 433, column: 9, scope: !1631)
!1633 = !DILocation(line: 434, column: 17, scope: !1634)
!1634 = distinct !DILexicalBlock(scope: !1630, file: !3, line: 433, column: 15)
!1635 = !DILocalVariable(name: "nn", scope: !1634, file: !3, line: 434, type: !402)
!1636 = !DILocation(line: 0, scope: !1634)
!1637 = !DILocation(line: 435, column: 12, scope: !1638)
!1638 = distinct !DILexicalBlock(scope: !1634, file: !3, line: 435, column: 10)
!1639 = !DILocation(line: 435, column: 10, scope: !1634)
!1640 = !DILocation(line: 435, column: 21, scope: !1641)
!1641 = distinct !DILexicalBlock(scope: !1638, file: !3, line: 435, column: 16)
!1642 = !DILocation(line: 435, column: 19, scope: !1641)
!1643 = !DILocation(line: 435, column: 24, scope: !1641)
!1644 = !DILocation(line: 436, column: 7, scope: !1634)
!1645 = !DILocation(line: 437, column: 6, scope: !1634)
!1646 = !DILocation(line: 438, column: 7, scope: !1647)
!1647 = distinct !DILexicalBlock(scope: !1630, file: !3, line: 437, column: 11)
!1648 = !DILocation(line: 440, column: 5, scope: !1631)
!1649 = !DILocalVariable(name: "size", scope: !1650, file: !3, line: 441, type: !1651)
!1650 = distinct !DILexicalBlock(scope: !1623, file: !3, line: 440, column: 10)
!1651 = !DICompositeType(tag: DW_TAG_array_type, baseType: !109, size: 128, elements: !1652)
!1652 = !{!1653}
!1653 = !DISubrange(count: 16)
!1654 = !DILocation(line: 441, column: 11, scope: !1650)
!1655 = !DILocation(line: 443, column: 12, scope: !1656)
!1656 = distinct !DILexicalBlock(scope: !1650, file: !3, line: 443, column: 9)
!1657 = !DILocation(line: 443, column: 17, scope: !1656)
!1658 = !DILocation(line: 443, column: 23, scope: !1656)
!1659 = !DILocation(line: 443, column: 9, scope: !1650)
!1660 = !DILocation(line: 444, column: 11, scope: !1661)
!1661 = distinct !DILexicalBlock(scope: !1662, file: !3, line: 444, column: 10)
!1662 = distinct !DILexicalBlock(scope: !1656, file: !3, line: 443, column: 28)
!1663 = !DILocation(line: 444, column: 18, scope: !1661)
!1664 = !DILocation(line: 444, column: 24, scope: !1661)
!1665 = !DILocation(line: 444, column: 27, scope: !1661)
!1666 = !DILocation(line: 444, column: 34, scope: !1661)
!1667 = !DILocation(line: 444, column: 10, scope: !1662)
!1668 = !DILocation(line: 445, column: 16, scope: !1669)
!1669 = distinct !DILexicalBlock(scope: !1661, file: !3, line: 444, column: 41)
!1670 = !DILocation(line: 445, column: 42, scope: !1669)
!1671 = !DILocation(line: 445, column: 33, scope: !1669)
!1672 = !DILocation(line: 445, column: 8, scope: !1669)
!1673 = !DILocalVariable(name: "j", scope: !1650, file: !3, line: 442, type: !97)
!1674 = !DILocation(line: 0, scope: !1650)
!1675 = !DILocation(line: 447, column: 11, scope: !1676)
!1676 = distinct !DILexicalBlock(scope: !1669, file: !3, line: 447, column: 11)
!1677 = !DILocation(line: 447, column: 19, scope: !1676)
!1678 = !DILocation(line: 447, column: 11, scope: !1669)
!1679 = !DILocation(line: 448, column: 9, scope: !1680)
!1680 = distinct !DILexicalBlock(scope: !1676, file: !3, line: 447, column: 26)
!1681 = !DILocation(line: 448, column: 17, scope: !1680)
!1682 = !DILocation(line: 449, column: 10, scope: !1680)
!1683 = !DILocation(line: 450, column: 8, scope: !1680)
!1684 = !DILocation(line: 0, scope: !1669)
!1685 = !DILocation(line: 451, column: 14, scope: !1669)
!1686 = !DILocation(line: 451, column: 8, scope: !1669)
!1687 = !DILocation(line: 451, column: 18, scope: !1669)
!1688 = !DILocation(line: 452, column: 53, scope: !1669)
!1689 = !DILocation(line: 452, column: 8, scope: !1669)
!1690 = !DILocation(line: 453, column: 7, scope: !1669)
!1691 = !DILocation(line: 454, column: 8, scope: !1692)
!1692 = distinct !DILexicalBlock(scope: !1661, file: !3, line: 453, column: 12)
!1693 = !DILocation(line: 456, column: 6, scope: !1662)
!1694 = !DILocation(line: 457, column: 7, scope: !1695)
!1695 = distinct !DILexicalBlock(scope: !1656, file: !3, line: 456, column: 11)
!1696 = !DILocation(line: 460, column: 5, scope: !1591)
!1697 = !DILocation(line: 461, column: 5, scope: !1591)
!1698 = !DILocation(line: 462, column: 5, scope: !1591)
!1699 = !DILocation(line: 463, column: 5, scope: !1591)
!1700 = !DILocation(line: 464, column: 28, scope: !1701)
!1701 = distinct !DILexicalBlock(scope: !1591, file: !3, line: 464, column: 8)
!1702 = !DILocation(line: 464, column: 8, scope: !1591)
!1703 = !DILocation(line: 465, column: 6, scope: !1704)
!1704 = distinct !DILexicalBlock(scope: !1701, file: !3, line: 464, column: 32)
!1705 = !DILocation(line: 466, column: 5, scope: !1704)
!1706 = !DILocation(line: 466, column: 14, scope: !1707)
!1707 = distinct !DILexicalBlock(scope: !1701, file: !3, line: 466, column: 14)
!1708 = !DILocation(line: 466, column: 14, scope: !1701)
!1709 = !DILocation(line: 467, column: 6, scope: !1710)
!1710 = distinct !DILexicalBlock(scope: !1707, file: !3, line: 466, column: 34)
!1711 = !DILocation(line: 468, column: 5, scope: !1710)
!1712 = !DILocation(line: 469, column: 6, scope: !1713)
!1713 = distinct !DILexicalBlock(scope: !1707, file: !3, line: 468, column: 10)
!1714 = !DILocation(line: 471, column: 5, scope: !1591)
!1715 = !DILocation(line: 472, column: 5, scope: !1591)
!1716 = !DILocation(line: 473, column: 5, scope: !1591)
!1717 = !DILocation(line: 474, column: 5, scope: !1591)
!1718 = !DILocation(line: 475, column: 5, scope: !1591)
!1719 = !DILocation(line: 476, column: 5, scope: !1591)
!1720 = !DILocation(line: 477, column: 5, scope: !1591)
!1721 = !DILocation(line: 478, column: 5, scope: !1591)
!1722 = !DILocation(line: 479, column: 5, scope: !1591)
!1723 = !DILocation(line: 480, column: 5, scope: !1591)
!1724 = !DILocation(line: 481, column: 5, scope: !1591)
!1725 = !DILocation(line: 482, column: 5, scope: !1591)
!1726 = !DILocation(line: 483, column: 5, scope: !1591)
!1727 = !DILocation(line: 484, column: 5, scope: !1591)
!1728 = !DILocation(line: 485, column: 5, scope: !1591)
!1729 = !DILocation(line: 486, column: 5, scope: !1591)
!1730 = !DILocation(line: 487, column: 5, scope: !1591)
!1731 = !DILocation(line: 502, column: 5, scope: !1591)
!1732 = !DILocation(line: 503, column: 5, scope: !1591)
!1733 = !DILocation(line: 504, column: 5, scope: !1591)
!1734 = !DILocation(line: 505, column: 5, scope: !1591)
!1735 = !DILocation(line: 506, column: 5, scope: !1591)
!1736 = !DILocation(line: 507, column: 5, scope: !1591)
!1737 = !DILocation(line: 508, column: 5, scope: !1591)
!1738 = !DILocation(line: 509, column: 5, scope: !1591)
!1739 = !DILocation(line: 510, column: 5, scope: !1591)
!1740 = !DILocation(line: 511, column: 5, scope: !1591)
!1741 = !DILocation(line: 512, column: 4, scope: !1591)
!1742 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 516, type: !1743, scopeLine: 516, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !1051)
!1743 = !DISubroutineType(types: !1744)
!1744 = !{!97, !97, !655}
!1745 = !DILocalVariable(name: "argc", arg: 1, scope: !1742, file: !3, line: 516, type: !97)
!1746 = !DILocation(line: 0, scope: !1742)
!1747 = !DILocalVariable(name: "argv", arg: 2, scope: !1742, file: !3, line: 516, type: !655)
!1748 = !DILocalVariable(name: "iter", scope: !1742, file: !3, line: 523, type: !97)
!1749 = !DILocalVariable(name: "verified", scope: !1742, file: !3, line: 525, type: !1750)
!1750 = !DIDerivedType(tag: DW_TAG_typedef, name: "boolean", file: !100, line: 80, baseType: !97)
!1751 = !DILocation(line: 525, column: 10, scope: !1742)
!1752 = !DILocalVariable(name: "class_npb", scope: !1742, file: !3, line: 526, type: !109)
!1753 = !DILocation(line: 526, column: 7, scope: !1742)
!1754 = !DILocation(line: 529, column: 20, scope: !1742)
!1755 = !{i64 112}
!1756 = !DILocation(line: 529, column: 9, scope: !1742)
!1757 = !DILocation(line: 529, column: 7, scope: !1742)
!1758 = !DILocation(line: 530, column: 21, scope: !1742)
!1759 = !DILocation(line: 530, column: 12, scope: !1742)
!1760 = !DILocation(line: 530, column: 10, scope: !1742)
!1761 = !DILocation(line: 531, column: 17, scope: !1742)
!1762 = !{i64 4096}
!1763 = !DILocation(line: 531, column: 6, scope: !1742)
!1764 = !DILocation(line: 531, column: 4, scope: !1742)
!1765 = !DILocation(line: 532, column: 18, scope: !1742)
!1766 = !DILocation(line: 532, column: 7, scope: !1742)
!1767 = !DILocation(line: 532, column: 5, scope: !1742)
!1768 = !DILocation(line: 533, column: 18, scope: !1742)
!1769 = !DILocation(line: 533, column: 7, scope: !1742)
!1770 = !DILocation(line: 533, column: 5, scope: !1742)
!1771 = !DILocation(line: 534, column: 15, scope: !1742)
!1772 = !DILocation(line: 534, column: 9, scope: !1742)
!1773 = !DILocation(line: 534, column: 7, scope: !1742)
!1774 = !DILocation(line: 544, column: 2, scope: !1742)
!1775 = !DILocation(line: 545, column: 2, scope: !1742)
!1776 = !DILocation(line: 546, column: 14, scope: !1742)
!1777 = !DILocation(line: 546, column: 25, scope: !1742)
!1778 = !DILocation(line: 546, column: 36, scope: !1742)
!1779 = !DILocation(line: 546, column: 2, scope: !1742)
!1780 = !DILocation(line: 549, column: 6, scope: !1781)
!1781 = distinct !DILexicalBlock(scope: !1782, file: !3, line: 549, column: 6)
!1782 = distinct !DILexicalBlock(scope: !1742, file: !3, line: 548, column: 2)
!1783 = !DILocation(line: 549, column: 26, scope: !1781)
!1784 = !DILocation(line: 549, column: 6, scope: !1782)
!1785 = !DILocation(line: 550, column: 25, scope: !1786)
!1786 = distinct !DILexicalBlock(scope: !1781, file: !3, line: 549, column: 42)
!1787 = !DILocation(line: 550, column: 4, scope: !1786)
!1788 = !DILocation(line: 551, column: 3, scope: !1786)
!1789 = !DILocation(line: 551, column: 12, scope: !1790)
!1790 = distinct !DILexicalBlock(scope: !1781, file: !3, line: 551, column: 12)
!1791 = !DILocation(line: 551, column: 32, scope: !1790)
!1792 = !DILocation(line: 551, column: 12, scope: !1781)
!1793 = !DILocation(line: 552, column: 35, scope: !1794)
!1794 = distinct !DILexicalBlock(scope: !1790, file: !3, line: 551, column: 58)
!1795 = !DILocation(line: 552, column: 4, scope: !1794)
!1796 = !DILocation(line: 553, column: 3, scope: !1794)
!1797 = !DILocation(line: 553, column: 12, scope: !1798)
!1798 = distinct !DILexicalBlock(scope: !1790, file: !3, line: 553, column: 12)
!1799 = !DILocation(line: 553, column: 32, scope: !1798)
!1800 = !DILocation(line: 553, column: 12, scope: !1790)
!1801 = !DILocation(line: 554, column: 4, scope: !1802)
!1802 = distinct !DILexicalBlock(scope: !1798, file: !3, line: 553, column: 47)
!1803 = !DILocation(line: 555, column: 3, scope: !1802)
!1804 = !DILocation(line: 557, column: 13, scope: !1742)
!1805 = !DILocation(line: 557, column: 24, scope: !1742)
!1806 = !DILocation(line: 557, column: 2, scope: !1742)
!1807 = !DILocation(line: 586, column: 6, scope: !1808)
!1808 = distinct !DILexicalBlock(scope: !1809, file: !3, line: 586, column: 6)
!1809 = distinct !DILexicalBlock(scope: !1742, file: !3, line: 585, column: 2)
!1810 = !DILocation(line: 586, column: 26, scope: !1808)
!1811 = !DILocation(line: 586, column: 6, scope: !1809)
!1812 = !DILocation(line: 587, column: 25, scope: !1813)
!1813 = distinct !DILexicalBlock(scope: !1808, file: !3, line: 586, column: 42)
!1814 = !DILocation(line: 587, column: 4, scope: !1813)
!1815 = !DILocation(line: 588, column: 3, scope: !1813)
!1816 = !DILocation(line: 588, column: 12, scope: !1817)
!1817 = distinct !DILexicalBlock(scope: !1808, file: !3, line: 588, column: 12)
!1818 = !DILocation(line: 588, column: 32, scope: !1817)
!1819 = !DILocation(line: 588, column: 12, scope: !1808)
!1820 = !DILocation(line: 589, column: 35, scope: !1821)
!1821 = distinct !DILexicalBlock(scope: !1817, file: !3, line: 588, column: 58)
!1822 = !DILocation(line: 589, column: 4, scope: !1821)
!1823 = !DILocation(line: 590, column: 3, scope: !1821)
!1824 = !DILocation(line: 590, column: 12, scope: !1825)
!1825 = distinct !DILexicalBlock(scope: !1817, file: !3, line: 590, column: 12)
!1826 = !DILocation(line: 590, column: 32, scope: !1825)
!1827 = !DILocation(line: 590, column: 12, scope: !1817)
!1828 = !DILocation(line: 591, column: 4, scope: !1829)
!1829 = distinct !DILexicalBlock(scope: !1825, file: !3, line: 590, column: 47)
!1830 = !DILocation(line: 592, column: 3, scope: !1829)
!1831 = !DILocation(line: 594, column: 13, scope: !1742)
!1832 = !DILocation(line: 594, column: 24, scope: !1742)
!1833 = !DILocation(line: 594, column: 2, scope: !1742)
!1834 = !DILocation(line: 595, column: 6, scope: !1835)
!1835 = distinct !DILexicalBlock(scope: !1742, file: !3, line: 595, column: 2)
!1836 = !DILocation(line: 0, scope: !1835)
!1837 = !DILocation(line: 595, column: 20, scope: !1838)
!1838 = distinct !DILexicalBlock(scope: !1835, file: !3, line: 595, column: 2)
!1839 = !DILocation(line: 595, column: 18, scope: !1838)
!1840 = !DILocation(line: 595, column: 2, scope: !1835)
!1841 = !DILocation(line: 596, column: 14, scope: !1842)
!1842 = distinct !DILexicalBlock(scope: !1838, file: !3, line: 595, column: 34)
!1843 = !DILocation(line: 596, column: 25, scope: !1842)
!1844 = !DILocation(line: 596, column: 36, scope: !1842)
!1845 = !DILocation(line: 596, column: 3, scope: !1842)
!1846 = !DILocation(line: 597, column: 15, scope: !1842)
!1847 = !DILocation(line: 597, column: 26, scope: !1842)
!1848 = !DILocation(line: 597, column: 3, scope: !1842)
!1849 = !DILocation(line: 598, column: 22, scope: !1842)
!1850 = !DILocation(line: 598, column: 3, scope: !1842)
!1851 = !DILocation(line: 599, column: 2, scope: !1842)
!1852 = !DILocation(line: 595, column: 31, scope: !1838)
!1853 = !DILocation(line: 595, column: 2, scope: !1838)
!1854 = distinct !{!1854, !1840, !1855}
!1855 = !DILocation(line: 599, column: 2, scope: !1835)
!1856 = !DILocation(line: 601, column: 13, scope: !1742)
!1857 = !DILocation(line: 601, column: 19, scope: !1742)
!1858 = !DILocation(line: 601, column: 32, scope: !1742)
!1859 = !DILocation(line: 601, column: 2, scope: !1742)
!1860 = !{!""}
!1861 = !DILocation(line: 602, column: 6, scope: !1862)
!1862 = distinct !DILexicalBlock(scope: !1742, file: !3, line: 602, column: 2)
!1863 = !DILocation(line: 0, scope: !1862)
!1864 = !DILocation(line: 602, column: 20, scope: !1865)
!1865 = distinct !DILexicalBlock(scope: !1862, file: !3, line: 602, column: 2)
!1866 = !DILocation(line: 602, column: 18, scope: !1865)
!1867 = !DILocation(line: 602, column: 2, scope: !1862)
!1868 = !DILocation(line: 603, column: 60, scope: !1869)
!1869 = distinct !DILexicalBlock(scope: !1865, file: !3, line: 602, column: 34)
!1870 = !DILocation(line: 603, column: 71, scope: !1869)
!1871 = !DILocation(line: 603, column: 77, scope: !1869)
!1872 = !DILocation(line: 603, column: 88, scope: !1869)
!1873 = !DILocation(line: 603, column: 3, scope: !1869)
!1874 = !DILocation(line: 604, column: 2, scope: !1869)
!1875 = !DILocation(line: 602, column: 31, scope: !1865)
!1876 = !DILocation(line: 602, column: 2, scope: !1865)
!1877 = distinct !{!1877, !1867, !1878}
!1878 = !DILocation(line: 604, column: 2, scope: !1862)
!1879 = !DILocation(line: 606, column: 21, scope: !1742)
!1880 = !DILocation(line: 606, column: 2, scope: !1742)
!1881 = !DILocalVariable(name: "total_time", scope: !1742, file: !3, line: 524, type: !104)
!1882 = !DILocation(line: 612, column: 16, scope: !1883)
!1883 = distinct !DILexicalBlock(scope: !1742, file: !3, line: 612, column: 5)
!1884 = !DILocation(line: 612, column: 5, scope: !1742)
!1885 = !DILocation(line: 614, column: 25, scope: !1886)
!1886 = distinct !DILexicalBlock(scope: !1883, file: !3, line: 612, column: 23)
!1887 = !DILocation(line: 614, column: 23, scope: !1886)
!1888 = !DILocation(line: 614, column: 13, scope: !1886)
!1889 = !DILocation(line: 615, column: 28, scope: !1886)
!1890 = !DILocation(line: 615, column: 26, scope: !1886)
!1891 = !DILocation(line: 615, column: 16, scope: !1886)
!1892 = !DILocation(line: 615, column: 51, scope: !1886)
!1893 = !DILocation(line: 615, column: 50, scope: !1886)
!1894 = !DILocation(line: 615, column: 5, scope: !1886)
!1895 = !DILocation(line: 613, column: 40, scope: !1886)
!1896 = !DILocation(line: 616, column: 4, scope: !1886)
!1897 = !DILocalVariable(name: "mflops", scope: !1742, file: !3, line: 524, type: !104)
!1898 = !DILocation(line: 617, column: 2, scope: !1886)
!1899 = !DILocation(line: 0, scope: !1883)
!1900 = !DILocalVariable(name: "gpu_config", scope: !1742, file: !3, line: 621, type: !215)
!1901 = !DILocation(line: 621, column: 7, scope: !1742)
!1902 = !DILocalVariable(name: "gpu_config_string", scope: !1742, file: !3, line: 622, type: !1903)
!1903 = !DICompositeType(tag: DW_TAG_array_type, baseType: !109, size: 16384, elements: !1904)
!1904 = !{!1905}
!1905 = !DISubrange(count: 2048)
!1906 = !DILocation(line: 622, column: 7, scope: !1742)
!1907 = !DILocation(line: 655, column: 10, scope: !1742)
!1908 = !DILocation(line: 655, column: 2, scope: !1742)
!1909 = !DILocation(line: 656, column: 9, scope: !1742)
!1910 = !DILocation(line: 656, column: 28, scope: !1742)
!1911 = !DILocation(line: 656, column: 2, scope: !1742)
!1912 = !DILocation(line: 657, column: 10, scope: !1742)
!1913 = !DILocation(line: 657, column: 51, scope: !1742)
!1914 = !DILocation(line: 657, column: 2, scope: !1742)
!1915 = !DILocation(line: 658, column: 9, scope: !1742)
!1916 = !DILocation(line: 658, column: 28, scope: !1742)
!1917 = !DILocation(line: 658, column: 2, scope: !1742)
!1918 = !DILocation(line: 659, column: 10, scope: !1742)
!1919 = !DILocation(line: 659, column: 61, scope: !1742)
!1920 = !DILocation(line: 659, column: 2, scope: !1742)
!1921 = !DILocation(line: 660, column: 9, scope: !1742)
!1922 = !DILocation(line: 660, column: 28, scope: !1742)
!1923 = !DILocation(line: 660, column: 2, scope: !1742)
!1924 = !DILocation(line: 661, column: 10, scope: !1742)
!1925 = !DILocation(line: 661, column: 50, scope: !1742)
!1926 = !DILocation(line: 661, column: 2, scope: !1742)
!1927 = !DILocation(line: 662, column: 9, scope: !1742)
!1928 = !DILocation(line: 662, column: 28, scope: !1742)
!1929 = !DILocation(line: 662, column: 2, scope: !1742)
!1930 = !DILocation(line: 663, column: 10, scope: !1742)
!1931 = !DILocation(line: 663, column: 49, scope: !1742)
!1932 = !DILocation(line: 663, column: 2, scope: !1742)
!1933 = !DILocation(line: 664, column: 9, scope: !1742)
!1934 = !DILocation(line: 664, column: 28, scope: !1742)
!1935 = !DILocation(line: 664, column: 2, scope: !1742)
!1936 = !DILocation(line: 665, column: 10, scope: !1742)
!1937 = !DILocation(line: 665, column: 49, scope: !1742)
!1938 = !DILocation(line: 665, column: 2, scope: !1742)
!1939 = !DILocation(line: 666, column: 9, scope: !1742)
!1940 = !DILocation(line: 666, column: 28, scope: !1742)
!1941 = !DILocation(line: 666, column: 2, scope: !1742)
!1942 = !DILocation(line: 667, column: 10, scope: !1742)
!1943 = !DILocation(line: 667, column: 49, scope: !1742)
!1944 = !DILocation(line: 667, column: 2, scope: !1742)
!1945 = !DILocation(line: 668, column: 9, scope: !1742)
!1946 = !DILocation(line: 668, column: 28, scope: !1742)
!1947 = !DILocation(line: 668, column: 2, scope: !1742)
!1948 = !DILocation(line: 669, column: 10, scope: !1742)
!1949 = !DILocation(line: 669, column: 49, scope: !1742)
!1950 = !DILocation(line: 669, column: 2, scope: !1742)
!1951 = !DILocation(line: 670, column: 9, scope: !1742)
!1952 = !DILocation(line: 670, column: 28, scope: !1742)
!1953 = !DILocation(line: 670, column: 2, scope: !1742)
!1954 = !DILocation(line: 671, column: 10, scope: !1742)
!1955 = !DILocation(line: 671, column: 49, scope: !1742)
!1956 = !DILocation(line: 671, column: 2, scope: !1742)
!1957 = !DILocation(line: 672, column: 9, scope: !1742)
!1958 = !DILocation(line: 672, column: 28, scope: !1742)
!1959 = !DILocation(line: 672, column: 2, scope: !1742)
!1960 = !DILocation(line: 673, column: 10, scope: !1742)
!1961 = !DILocation(line: 673, column: 49, scope: !1742)
!1962 = !DILocation(line: 673, column: 2, scope: !1742)
!1963 = !DILocation(line: 674, column: 9, scope: !1742)
!1964 = !DILocation(line: 674, column: 28, scope: !1742)
!1965 = !DILocation(line: 674, column: 2, scope: !1742)
!1966 = !DILocation(line: 675, column: 10, scope: !1742)
!1967 = !DILocation(line: 675, column: 49, scope: !1742)
!1968 = !DILocation(line: 675, column: 2, scope: !1742)
!1969 = !DILocation(line: 676, column: 9, scope: !1742)
!1970 = !DILocation(line: 676, column: 28, scope: !1742)
!1971 = !DILocation(line: 676, column: 2, scope: !1742)
!1972 = !DILocation(line: 677, column: 10, scope: !1742)
!1973 = !DILocation(line: 677, column: 49, scope: !1742)
!1974 = !DILocation(line: 677, column: 2, scope: !1742)
!1975 = !DILocation(line: 678, column: 9, scope: !1742)
!1976 = !DILocation(line: 678, column: 28, scope: !1742)
!1977 = !DILocation(line: 678, column: 2, scope: !1742)
!1978 = !DILocation(line: 679, column: 10, scope: !1742)
!1979 = !DILocation(line: 679, column: 49, scope: !1742)
!1980 = !DILocation(line: 679, column: 2, scope: !1742)
!1981 = !DILocation(line: 680, column: 9, scope: !1742)
!1982 = !DILocation(line: 680, column: 28, scope: !1742)
!1983 = !DILocation(line: 680, column: 2, scope: !1742)
!1984 = !DILocation(line: 681, column: 10, scope: !1742)
!1985 = !DILocation(line: 681, column: 49, scope: !1742)
!1986 = !DILocation(line: 681, column: 2, scope: !1742)
!1987 = !DILocation(line: 682, column: 9, scope: !1742)
!1988 = !DILocation(line: 682, column: 28, scope: !1742)
!1989 = !DILocation(line: 682, column: 2, scope: !1742)
!1990 = !DILocation(line: 683, column: 10, scope: !1742)
!1991 = !DILocation(line: 683, column: 51, scope: !1742)
!1992 = !DILocation(line: 683, column: 2, scope: !1742)
!1993 = !DILocation(line: 684, column: 9, scope: !1742)
!1994 = !DILocation(line: 684, column: 28, scope: !1742)
!1995 = !DILocation(line: 684, column: 2, scope: !1742)
!1996 = !DILocation(line: 688, column: 4, scope: !1742)
!1997 = !DILocation(line: 692, column: 4, scope: !1742)
!1998 = !DILocation(line: 696, column: 4, scope: !1742)
!1999 = !DILocation(line: 703, column: 11, scope: !1742)
!2000 = !DILocation(line: 687, column: 2, scope: !1742)
!2001 = !DILocation(line: 712, column: 2, scope: !1742)
!2002 = !DILocation(line: 714, column: 7, scope: !1742)
!2003 = !DILocation(line: 714, column: 2, scope: !1742)
!2004 = !DILocation(line: 715, column: 7, scope: !1742)
!2005 = !DILocation(line: 715, column: 2, scope: !1742)
!2006 = !DILocation(line: 716, column: 7, scope: !1742)
!2007 = !DILocation(line: 716, column: 2, scope: !1742)
!2008 = !DILocation(line: 717, column: 7, scope: !1742)
!2009 = !DILocation(line: 717, column: 2, scope: !1742)
!2010 = !DILocation(line: 718, column: 7, scope: !1742)
!2011 = !DILocation(line: 718, column: 2, scope: !1742)
!2012 = !DILocation(line: 719, column: 7, scope: !1742)
!2013 = !DILocation(line: 719, column: 2, scope: !1742)
!2014 = !DILocation(line: 722, column: 2, scope: !1742)
!2015 = distinct !DISubprogram(name: "setup", linkageName: "_ZL5setupv", scope: !3, file: !3, line: 1659, type: !561, scopeLine: 1659, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1051)
!2016 = !DILocation(line: 1660, column: 8, scope: !2015)
!2017 = !DILocation(line: 1662, column: 2, scope: !2015)
!2018 = !DILocation(line: 1663, column: 2, scope: !2015)
!2019 = !DILocation(line: 1664, column: 48, scope: !2015)
!2020 = !DILocation(line: 1664, column: 2, scope: !2015)
!2021 = !DILocation(line: 1665, column: 2, scope: !2015)
!2022 = !DILocation(line: 1666, column: 1, scope: !2015)
!2023 = distinct !DISubprogram(name: "setup_gpu", linkageName: "_ZL9setup_gpuv", scope: !3, file: !3, line: 1668, type: !561, scopeLine: 1668, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1051)
!2024 = !DILocation(line: 1713, column: 33, scope: !2023)
!2025 = !DILocation(line: 1714, column: 43, scope: !2023)
!2026 = !DILocation(line: 1717, column: 69, scope: !2027)
!2027 = distinct !DILexicalBlock(scope: !2023, file: !3, line: 1716, column: 5)
!2028 = !DILocation(line: 1717, column: 45, scope: !2027)
!2029 = !DILocation(line: 1716, column: 5, scope: !2023)
!2030 = !DILocation(line: 1718, column: 41, scope: !2031)
!2031 = distinct !DILexicalBlock(scope: !2027, file: !3, line: 1717, column: 89)
!2032 = !DILocation(line: 1719, column: 2, scope: !2031)
!2033 = !DILocation(line: 1720, column: 65, scope: !2034)
!2034 = distinct !DILexicalBlock(scope: !2027, file: !3, line: 1719, column: 7)
!2035 = !DILocation(line: 1720, column: 41, scope: !2034)
!2036 = !DILocation(line: 1723, column: 79, scope: !2037)
!2037 = distinct !DILexicalBlock(scope: !2023, file: !3, line: 1722, column: 5)
!2038 = !DILocation(line: 1723, column: 55, scope: !2037)
!2039 = !DILocation(line: 1722, column: 5, scope: !2023)
!2040 = !DILocation(line: 1724, column: 51, scope: !2041)
!2041 = distinct !DILexicalBlock(scope: !2037, file: !3, line: 1723, column: 99)
!2042 = !DILocation(line: 1725, column: 2, scope: !2041)
!2043 = !DILocation(line: 1726, column: 75, scope: !2044)
!2044 = distinct !DILexicalBlock(scope: !2037, file: !3, line: 1725, column: 7)
!2045 = !DILocation(line: 1726, column: 51, scope: !2044)
!2046 = !DILocation(line: 1729, column: 60, scope: !2047)
!2047 = distinct !DILexicalBlock(scope: !2023, file: !3, line: 1728, column: 5)
!2048 = !DILocation(line: 1729, column: 36, scope: !2047)
!2049 = !DILocation(line: 1728, column: 5, scope: !2023)
!2050 = !DILocation(line: 1730, column: 32, scope: !2051)
!2051 = distinct !DILexicalBlock(scope: !2047, file: !3, line: 1729, column: 80)
!2052 = !DILocation(line: 1731, column: 2, scope: !2051)
!2053 = !DILocation(line: 1732, column: 54, scope: !2054)
!2054 = distinct !DILexicalBlock(scope: !2047, file: !3, line: 1731, column: 7)
!2055 = !DILocation(line: 1732, column: 31, scope: !2054)
!2056 = !DILocation(line: 1735, column: 59, scope: !2057)
!2057 = distinct !DILexicalBlock(scope: !2023, file: !3, line: 1734, column: 5)
!2058 = !DILocation(line: 1735, column: 35, scope: !2057)
!2059 = !DILocation(line: 1734, column: 5, scope: !2023)
!2060 = !DILocation(line: 1736, column: 31, scope: !2061)
!2061 = distinct !DILexicalBlock(scope: !2057, file: !3, line: 1735, column: 79)
!2062 = !DILocation(line: 1737, column: 2, scope: !2061)
!2063 = !DILocation(line: 1738, column: 53, scope: !2064)
!2064 = distinct !DILexicalBlock(scope: !2057, file: !3, line: 1737, column: 7)
!2065 = !DILocation(line: 1738, column: 30, scope: !2064)
!2066 = !DILocation(line: 1741, column: 59, scope: !2067)
!2067 = distinct !DILexicalBlock(scope: !2023, file: !3, line: 1740, column: 5)
!2068 = !DILocation(line: 1741, column: 35, scope: !2067)
!2069 = !DILocation(line: 1740, column: 5, scope: !2023)
!2070 = !DILocation(line: 1742, column: 31, scope: !2071)
!2071 = distinct !DILexicalBlock(scope: !2067, file: !3, line: 1741, column: 79)
!2072 = !DILocation(line: 1743, column: 2, scope: !2071)
!2073 = !DILocation(line: 1744, column: 55, scope: !2074)
!2074 = distinct !DILexicalBlock(scope: !2067, file: !3, line: 1743, column: 7)
!2075 = !DILocation(line: 1744, column: 31, scope: !2074)
!2076 = !DILocation(line: 1747, column: 59, scope: !2077)
!2077 = distinct !DILexicalBlock(scope: !2023, file: !3, line: 1746, column: 5)
!2078 = !DILocation(line: 1747, column: 35, scope: !2077)
!2079 = !DILocation(line: 1746, column: 5, scope: !2023)
!2080 = !DILocation(line: 1748, column: 31, scope: !2081)
!2081 = distinct !DILexicalBlock(scope: !2077, file: !3, line: 1747, column: 79)
!2082 = !DILocation(line: 1749, column: 2, scope: !2081)
!2083 = !DILocation(line: 1750, column: 55, scope: !2084)
!2084 = distinct !DILexicalBlock(scope: !2077, file: !3, line: 1749, column: 7)
!2085 = !DILocation(line: 1750, column: 31, scope: !2084)
!2086 = !DILocation(line: 1753, column: 59, scope: !2087)
!2087 = distinct !DILexicalBlock(scope: !2023, file: !3, line: 1752, column: 5)
!2088 = !DILocation(line: 1753, column: 35, scope: !2087)
!2089 = !DILocation(line: 1752, column: 5, scope: !2023)
!2090 = !DILocation(line: 1754, column: 31, scope: !2091)
!2091 = distinct !DILexicalBlock(scope: !2087, file: !3, line: 1753, column: 79)
!2092 = !DILocation(line: 1755, column: 2, scope: !2091)
!2093 = !DILocation(line: 1756, column: 55, scope: !2094)
!2094 = distinct !DILexicalBlock(scope: !2087, file: !3, line: 1755, column: 7)
!2095 = !DILocation(line: 1756, column: 31, scope: !2094)
!2096 = !DILocation(line: 1759, column: 59, scope: !2097)
!2097 = distinct !DILexicalBlock(scope: !2023, file: !3, line: 1758, column: 5)
!2098 = !DILocation(line: 1759, column: 35, scope: !2097)
!2099 = !DILocation(line: 1758, column: 5, scope: !2023)
!2100 = !DILocation(line: 1760, column: 31, scope: !2101)
!2101 = distinct !DILexicalBlock(scope: !2097, file: !3, line: 1759, column: 79)
!2102 = !DILocation(line: 1761, column: 2, scope: !2101)
!2103 = !DILocation(line: 1762, column: 55, scope: !2104)
!2104 = distinct !DILexicalBlock(scope: !2097, file: !3, line: 1761, column: 7)
!2105 = !DILocation(line: 1762, column: 31, scope: !2104)
!2106 = !DILocation(line: 1765, column: 59, scope: !2107)
!2107 = distinct !DILexicalBlock(scope: !2023, file: !3, line: 1764, column: 5)
!2108 = !DILocation(line: 1765, column: 35, scope: !2107)
!2109 = !DILocation(line: 1764, column: 5, scope: !2023)
!2110 = !DILocation(line: 1766, column: 31, scope: !2111)
!2111 = distinct !DILexicalBlock(scope: !2107, file: !3, line: 1765, column: 79)
!2112 = !DILocation(line: 1767, column: 2, scope: !2111)
!2113 = !DILocation(line: 1768, column: 55, scope: !2114)
!2114 = distinct !DILexicalBlock(scope: !2107, file: !3, line: 1767, column: 7)
!2115 = !DILocation(line: 1768, column: 31, scope: !2114)
!2116 = !DILocation(line: 1771, column: 59, scope: !2117)
!2117 = distinct !DILexicalBlock(scope: !2023, file: !3, line: 1770, column: 5)
!2118 = !DILocation(line: 1771, column: 35, scope: !2117)
!2119 = !DILocation(line: 1770, column: 5, scope: !2023)
!2120 = !DILocation(line: 1772, column: 31, scope: !2121)
!2121 = distinct !DILexicalBlock(scope: !2117, file: !3, line: 1771, column: 79)
!2122 = !DILocation(line: 1773, column: 2, scope: !2121)
!2123 = !DILocation(line: 1774, column: 55, scope: !2124)
!2124 = distinct !DILexicalBlock(scope: !2117, file: !3, line: 1773, column: 7)
!2125 = !DILocation(line: 1774, column: 31, scope: !2124)
!2126 = !DILocation(line: 1777, column: 59, scope: !2127)
!2127 = distinct !DILexicalBlock(scope: !2023, file: !3, line: 1776, column: 5)
!2128 = !DILocation(line: 1777, column: 35, scope: !2127)
!2129 = !DILocation(line: 1776, column: 5, scope: !2023)
!2130 = !DILocation(line: 1778, column: 31, scope: !2131)
!2131 = distinct !DILexicalBlock(scope: !2127, file: !3, line: 1777, column: 79)
!2132 = !DILocation(line: 1779, column: 2, scope: !2131)
!2133 = !DILocation(line: 1780, column: 55, scope: !2134)
!2134 = distinct !DILexicalBlock(scope: !2127, file: !3, line: 1779, column: 7)
!2135 = !DILocation(line: 1780, column: 31, scope: !2134)
!2136 = !DILocation(line: 1783, column: 59, scope: !2137)
!2137 = distinct !DILexicalBlock(scope: !2023, file: !3, line: 1782, column: 5)
!2138 = !DILocation(line: 1783, column: 35, scope: !2137)
!2139 = !DILocation(line: 1782, column: 5, scope: !2023)
!2140 = !DILocation(line: 1784, column: 31, scope: !2141)
!2141 = distinct !DILexicalBlock(scope: !2137, file: !3, line: 1783, column: 79)
!2142 = !DILocation(line: 1785, column: 2, scope: !2141)
!2143 = !DILocation(line: 1786, column: 55, scope: !2144)
!2144 = distinct !DILexicalBlock(scope: !2137, file: !3, line: 1785, column: 7)
!2145 = !DILocation(line: 1786, column: 31, scope: !2144)
!2146 = !DILocation(line: 1789, column: 59, scope: !2147)
!2147 = distinct !DILexicalBlock(scope: !2023, file: !3, line: 1788, column: 5)
!2148 = !DILocation(line: 1789, column: 35, scope: !2147)
!2149 = !DILocation(line: 1788, column: 5, scope: !2023)
!2150 = !DILocation(line: 1790, column: 31, scope: !2151)
!2151 = distinct !DILexicalBlock(scope: !2147, file: !3, line: 1789, column: 79)
!2152 = !DILocation(line: 1791, column: 2, scope: !2151)
!2153 = !DILocation(line: 1792, column: 55, scope: !2154)
!2154 = distinct !DILexicalBlock(scope: !2147, file: !3, line: 1791, column: 7)
!2155 = !DILocation(line: 1792, column: 31, scope: !2154)
!2156 = !DILocation(line: 1795, column: 61, scope: !2157)
!2157 = distinct !DILexicalBlock(scope: !2023, file: !3, line: 1794, column: 5)
!2158 = !DILocation(line: 1795, column: 37, scope: !2157)
!2159 = !DILocation(line: 1794, column: 5, scope: !2023)
!2160 = !DILocation(line: 1796, column: 33, scope: !2161)
!2161 = distinct !DILexicalBlock(scope: !2157, file: !3, line: 1795, column: 81)
!2162 = !DILocation(line: 1797, column: 2, scope: !2161)
!2163 = !DILocation(line: 1798, column: 57, scope: !2164)
!2164 = distinct !DILexicalBlock(scope: !2157, file: !3, line: 1797, column: 7)
!2165 = !DILocation(line: 1798, column: 33, scope: !2164)
!2166 = !DILocation(line: 1801, column: 65, scope: !2023)
!2167 = !DILocation(line: 1801, column: 57, scope: !2023)
!2168 = !DILocation(line: 1801, column: 38, scope: !2023)
!2169 = !DILocation(line: 1801, column: 37, scope: !2023)
!2170 = !DILocation(line: 1802, column: 71, scope: !2023)
!2171 = !DILocation(line: 1802, column: 63, scope: !2023)
!2172 = !DILocation(line: 1802, column: 48, scope: !2023)
!2173 = !DILocation(line: 1802, column: 47, scope: !2023)
!2174 = !DILocation(line: 1803, column: 56, scope: !2023)
!2175 = !DILocation(line: 1803, column: 48, scope: !2023)
!2176 = !DILocation(line: 1803, column: 29, scope: !2023)
!2177 = !DILocation(line: 1803, column: 28, scope: !2023)
!2178 = !DILocation(line: 1804, column: 55, scope: !2023)
!2179 = !DILocation(line: 1804, column: 47, scope: !2023)
!2180 = !DILocation(line: 1804, column: 28, scope: !2023)
!2181 = !DILocation(line: 1804, column: 27, scope: !2023)
!2182 = !DILocation(line: 1805, column: 57, scope: !2023)
!2183 = !DILocation(line: 1805, column: 49, scope: !2023)
!2184 = !DILocation(line: 1805, column: 28, scope: !2023)
!2185 = !DILocation(line: 1805, column: 27, scope: !2023)
!2186 = !DILocation(line: 1806, column: 54, scope: !2023)
!2187 = !DILocation(line: 1806, column: 46, scope: !2023)
!2188 = !DILocation(line: 1806, column: 28, scope: !2023)
!2189 = !DILocation(line: 1806, column: 27, scope: !2023)
!2190 = !DILocation(line: 1807, column: 57, scope: !2023)
!2191 = !DILocation(line: 1807, column: 49, scope: !2023)
!2192 = !DILocation(line: 1807, column: 28, scope: !2023)
!2193 = !DILocation(line: 1807, column: 27, scope: !2023)
!2194 = !DILocation(line: 1808, column: 57, scope: !2023)
!2195 = !DILocation(line: 1808, column: 49, scope: !2023)
!2196 = !DILocation(line: 1808, column: 28, scope: !2023)
!2197 = !DILocation(line: 1808, column: 27, scope: !2023)
!2198 = !DILocation(line: 1809, column: 54, scope: !2023)
!2199 = !DILocation(line: 1809, column: 46, scope: !2023)
!2200 = !DILocation(line: 1809, column: 28, scope: !2023)
!2201 = !DILocation(line: 1809, column: 27, scope: !2023)
!2202 = !DILocation(line: 1810, column: 57, scope: !2023)
!2203 = !DILocation(line: 1810, column: 49, scope: !2023)
!2204 = !DILocation(line: 1810, column: 28, scope: !2023)
!2205 = !DILocation(line: 1810, column: 27, scope: !2023)
!2206 = !DILocation(line: 1811, column: 57, scope: !2023)
!2207 = !DILocation(line: 1811, column: 49, scope: !2023)
!2208 = !DILocation(line: 1811, column: 28, scope: !2023)
!2209 = !DILocation(line: 1811, column: 27, scope: !2023)
!2210 = !DILocation(line: 1812, column: 54, scope: !2023)
!2211 = !DILocation(line: 1812, column: 46, scope: !2023)
!2212 = !DILocation(line: 1812, column: 28, scope: !2023)
!2213 = !DILocation(line: 1812, column: 27, scope: !2023)
!2214 = !DILocation(line: 1813, column: 57, scope: !2023)
!2215 = !DILocation(line: 1813, column: 49, scope: !2023)
!2216 = !DILocation(line: 1813, column: 28, scope: !2023)
!2217 = !DILocation(line: 1813, column: 27, scope: !2023)
!2218 = !DILocation(line: 1814, column: 65, scope: !2023)
!2219 = !DILocation(line: 1814, column: 57, scope: !2023)
!2220 = !DILocation(line: 1814, column: 30, scope: !2023)
!2221 = !DILocation(line: 1814, column: 29, scope: !2023)
!2222 = !DILocation(line: 1816, column: 18, scope: !2023)
!2223 = !DILocation(line: 1817, column: 20, scope: !2023)
!2224 = !DILocation(line: 1818, column: 21, scope: !2023)
!2225 = !DILocation(line: 1819, column: 15, scope: !2023)
!2226 = !DILocation(line: 1820, column: 16, scope: !2023)
!2227 = !DILocation(line: 1821, column: 16, scope: !2023)
!2228 = !DILocation(line: 1822, column: 16, scope: !2023)
!2229 = !DILocation(line: 1823, column: 16, scope: !2023)
!2230 = !DILocation(line: 1824, column: 19, scope: !2023)
!2231 = !DILocation(line: 1824, column: 48, scope: !2023)
!2232 = !DILocation(line: 1824, column: 18, scope: !2023)
!2233 = !DILocation(line: 1835, column: 2, scope: !2023)
!2234 = !DILocation(line: 1836, column: 1, scope: !2023)
!2235 = distinct !DISubprogram(name: "init_ui_gpu", linkageName: "_ZL11init_ui_gpuP8dcomplexS0_Pd", scope: !3, file: !3, line: 1538, type: !2236, scopeLine: 1540, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1051)
!2236 = !DISubroutineType(types: !2237)
!2237 = !{null, !98, !98, !106}
!2238 = !DILocalVariable(name: "u0", arg: 1, scope: !2235, file: !3, line: 1538, type: !98)
!2239 = !DILocation(line: 0, scope: !2235)
!2240 = !DILocalVariable(name: "u1", arg: 2, scope: !2235, file: !3, line: 1539, type: !98)
!2241 = !DILocalVariable(name: "twiddle", arg: 3, scope: !2235, file: !3, line: 1540, type: !106)
!2242 = !DILocation(line: 1544, column: 23, scope: !2235)
!2243 = !DILocation(line: 1545, column: 3, scope: !2235)
!2244 = !DILocation(line: 1544, column: 20, scope: !2235)
!2245 = !DILocation(line: 1552, column: 1, scope: !2235)
!2246 = distinct !DISubprogram(name: "compute_indexmap_gpu", linkageName: "_ZL20compute_indexmap_gpuPd", scope: !3, file: !3, line: 1354, type: !2247, scopeLine: 1354, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1051)
!2247 = !DISubroutineType(types: !2248)
!2248 = !{null, !106}
!2249 = !DILocalVariable(name: "twiddle", arg: 1, scope: !2246, file: !3, line: 1354, type: !106)
!2250 = !DILocation(line: 0, scope: !2246)
!2251 = !DILocation(line: 1358, column: 32, scope: !2246)
!2252 = !DILocation(line: 1359, column: 3, scope: !2246)
!2253 = !DILocation(line: 1358, column: 29, scope: !2246)
!2254 = !DILocation(line: 1363, column: 1, scope: !2246)
!2255 = distinct !DISubprogram(name: "compute_initial_conditions_gpu", linkageName: "_ZL30compute_initial_conditions_gpuP8dcomplex", scope: !3, file: !3, line: 1387, type: !2256, scopeLine: 1387, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1051)
!2256 = !DISubroutineType(types: !2257)
!2257 = !{null, !98}
!2258 = !DILocalVariable(name: "u0", arg: 1, scope: !2255, file: !3, line: 1387, type: !98)
!2259 = !DILocation(line: 0, scope: !2255)
!2260 = !DILocalVariable(name: "start", scope: !2255, file: !3, line: 1392, type: !104)
!2261 = !DILocation(line: 1392, column: 9, scope: !2255)
!2262 = !DILocalVariable(name: "an", scope: !2255, file: !3, line: 1392, type: !104)
!2263 = !DILocation(line: 1392, column: 16, scope: !2255)
!2264 = !DILocalVariable(name: "starts", scope: !2255, file: !3, line: 1392, type: !2265)
!2265 = !DICompositeType(tag: DW_TAG_array_type, baseType: !104, size: 8192, elements: !2266)
!2266 = !{!2267}
!2267 = !DISubrange(count: 128)
!2268 = !DILocation(line: 1392, column: 20, scope: !2255)
!2269 = !DILocation(line: 1394, column: 8, scope: !2255)
!2270 = !DILocation(line: 1396, column: 2, scope: !2255)
!2271 = !DILocation(line: 1397, column: 17, scope: !2255)
!2272 = !DILocation(line: 1397, column: 2, scope: !2255)
!2273 = !DILocation(line: 1398, column: 2, scope: !2255)
!2274 = !DILocation(line: 1400, column: 14, scope: !2255)
!2275 = !DILocation(line: 1400, column: 2, scope: !2255)
!2276 = !DILocation(line: 1400, column: 12, scope: !2255)
!2277 = !DILocalVariable(name: "z", scope: !2255, file: !3, line: 1391, type: !97)
!2278 = !DILocation(line: 1401, column: 6, scope: !2279)
!2279 = distinct !DILexicalBlock(scope: !2255, file: !3, line: 1401, column: 2)
!2280 = !DILocation(line: 0, scope: !2279)
!2281 = !DILocation(line: 1401, column: 12, scope: !2282)
!2282 = distinct !DILexicalBlock(scope: !2279, file: !3, line: 1401, column: 2)
!2283 = !DILocation(line: 1401, column: 2, scope: !2279)
!2284 = !DILocation(line: 1402, column: 18, scope: !2285)
!2285 = distinct !DILexicalBlock(scope: !2282, file: !3, line: 1401, column: 21)
!2286 = !DILocation(line: 1402, column: 3, scope: !2285)
!2287 = !DILocation(line: 1403, column: 15, scope: !2285)
!2288 = !DILocation(line: 1403, column: 3, scope: !2285)
!2289 = !DILocation(line: 1403, column: 13, scope: !2285)
!2290 = !DILocation(line: 1404, column: 2, scope: !2285)
!2291 = !DILocation(line: 1401, column: 18, scope: !2282)
!2292 = !DILocation(line: 1401, column: 2, scope: !2282)
!2293 = distinct !{!2293, !2283, !2294}
!2294 = !DILocation(line: 1404, column: 2, scope: !2279)
!2295 = !DILocation(line: 1406, column: 13, scope: !2255)
!2296 = !DILocation(line: 1406, column: 28, scope: !2255)
!2297 = !DILocation(line: 1406, column: 36, scope: !2255)
!2298 = !DILocation(line: 1406, column: 2, scope: !2255)
!2299 = !DILocation(line: 1408, column: 42, scope: !2255)
!2300 = !DILocation(line: 1409, column: 3, scope: !2255)
!2301 = !DILocation(line: 1408, column: 39, scope: !2255)
!2302 = !DILocation(line: 1414, column: 1, scope: !2255)
!2303 = distinct !DISubprogram(name: "fft_init_gpu", linkageName: "_ZL12fft_init_gpui", scope: !3, file: !3, line: 1479, type: !598, scopeLine: 1479, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1051)
!2304 = !DILocalVariable(name: "n", arg: 1, scope: !2303, file: !3, line: 1479, type: !97)
!2305 = !DILocation(line: 0, scope: !2303)
!2306 = !DILocation(line: 1491, column: 6, scope: !2303)
!2307 = !DILocalVariable(name: "m", scope: !2303, file: !3, line: 1483, type: !97)
!2308 = !DILocation(line: 1492, column: 9, scope: !2303)
!2309 = !DILocation(line: 1492, column: 2, scope: !2303)
!2310 = !DILocation(line: 1492, column: 7, scope: !2303)
!2311 = !DILocalVariable(name: "ku", scope: !2303, file: !3, line: 1483, type: !97)
!2312 = !DILocalVariable(name: "ln", scope: !2303, file: !3, line: 1483, type: !97)
!2313 = !DILocalVariable(name: "j", scope: !2303, file: !3, line: 1483, type: !97)
!2314 = !DILocation(line: 1495, column: 6, scope: !2315)
!2315 = distinct !DILexicalBlock(scope: !2303, file: !3, line: 1495, column: 2)
!2316 = !DILocation(line: 0, scope: !2315)
!2317 = !DILocation(line: 1495, column: 12, scope: !2318)
!2318 = distinct !DILexicalBlock(scope: !2315, file: !3, line: 1495, column: 2)
!2319 = !DILocation(line: 1495, column: 2, scope: !2315)
!2320 = !DILocation(line: 1496, column: 12, scope: !2321)
!2321 = distinct !DILexicalBlock(scope: !2318, file: !3, line: 1495, column: 21)
!2322 = !DILocation(line: 1496, column: 10, scope: !2321)
!2323 = !DILocalVariable(name: "t", scope: !2303, file: !3, line: 1484, type: !104)
!2324 = !DILocalVariable(name: "i", scope: !2303, file: !3, line: 1483, type: !97)
!2325 = !DILocation(line: 1497, column: 7, scope: !2326)
!2326 = distinct !DILexicalBlock(scope: !2321, file: !3, line: 1497, column: 3)
!2327 = !DILocation(line: 0, scope: !2326)
!2328 = !DILocation(line: 1497, column: 17, scope: !2329)
!2329 = distinct !DILexicalBlock(scope: !2326, file: !3, line: 1497, column: 3)
!2330 = !DILocation(line: 1497, column: 13, scope: !2329)
!2331 = !DILocation(line: 1497, column: 3, scope: !2326)
!2332 = !DILocation(line: 1498, column: 9, scope: !2333)
!2333 = distinct !DILexicalBlock(scope: !2329, file: !3, line: 1497, column: 25)
!2334 = !DILocation(line: 1498, column: 11, scope: !2333)
!2335 = !DILocalVariable(name: "ti", scope: !2303, file: !3, line: 1484, type: !104)
!2336 = !DILocation(line: 1499, column: 16, scope: !2333)
!2337 = !DILocation(line: 1499, column: 4, scope: !2333)
!2338 = !DILocation(line: 1499, column: 7, scope: !2333)
!2339 = !DILocation(line: 1499, column: 10, scope: !2333)
!2340 = !DILocation(line: 1499, column: 14, scope: !2333)
!2341 = !DILocation(line: 1500, column: 3, scope: !2333)
!2342 = !DILocation(line: 1497, column: 22, scope: !2329)
!2343 = !DILocation(line: 1497, column: 3, scope: !2329)
!2344 = distinct !{!2344, !2331, !2345}
!2345 = !DILocation(line: 1500, column: 3, scope: !2326)
!2346 = !DILocation(line: 1501, column: 11, scope: !2321)
!2347 = !DILocation(line: 1502, column: 10, scope: !2321)
!2348 = !DILocation(line: 1503, column: 2, scope: !2321)
!2349 = !DILocation(line: 1495, column: 18, scope: !2318)
!2350 = !DILocation(line: 1495, column: 2, scope: !2318)
!2351 = distinct !{!2351, !2319, !2352}
!2352 = !DILocation(line: 1503, column: 2, scope: !2315)
!2353 = !DILocation(line: 1504, column: 13, scope: !2303)
!2354 = !DILocation(line: 1504, column: 23, scope: !2303)
!2355 = !DILocation(line: 1504, column: 26, scope: !2303)
!2356 = !DILocation(line: 1504, column: 2, scope: !2303)
!2357 = !DILocation(line: 1508, column: 1, scope: !2303)
!2358 = distinct !DISubprogram(name: "fft_gpu", linkageName: "_ZL7fft_gpuiP8dcomplexS0_", scope: !3, file: !3, line: 1457, type: !2359, scopeLine: 1459, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1051)
!2359 = !DISubroutineType(types: !2360)
!2360 = !{null, !97, !98, !98}
!2361 = !DILocalVariable(name: "dir", arg: 1, scope: !2358, file: !3, line: 1457, type: !97)
!2362 = !DILocation(line: 0, scope: !2358)
!2363 = !DILocalVariable(name: "x1", arg: 2, scope: !2358, file: !3, line: 1458, type: !98)
!2364 = !DILocalVariable(name: "x2", arg: 3, scope: !2358, file: !3, line: 1459, type: !98)
!2365 = !DILocation(line: 1468, column: 8, scope: !2366)
!2366 = distinct !DILexicalBlock(scope: !2358, file: !3, line: 1468, column: 5)
!2367 = !DILocation(line: 1468, column: 5, scope: !2358)
!2368 = !DILocation(line: 1469, column: 17, scope: !2369)
!2369 = distinct !DILexicalBlock(scope: !2366, file: !3, line: 1468, column: 12)
!2370 = !DILocation(line: 1469, column: 35, scope: !2369)
!2371 = !DILocation(line: 1469, column: 46, scope: !2369)
!2372 = !DILocation(line: 1469, column: 3, scope: !2369)
!2373 = !DILocation(line: 1470, column: 17, scope: !2369)
!2374 = !DILocation(line: 1470, column: 35, scope: !2369)
!2375 = !DILocation(line: 1470, column: 46, scope: !2369)
!2376 = !DILocation(line: 1470, column: 3, scope: !2369)
!2377 = !DILocation(line: 1471, column: 17, scope: !2369)
!2378 = !DILocation(line: 1471, column: 35, scope: !2369)
!2379 = !DILocation(line: 1471, column: 46, scope: !2369)
!2380 = !DILocation(line: 1471, column: 3, scope: !2369)
!2381 = !DILocation(line: 1472, column: 2, scope: !2369)
!2382 = !DILocation(line: 1473, column: 18, scope: !2383)
!2383 = distinct !DILexicalBlock(scope: !2366, file: !3, line: 1472, column: 7)
!2384 = !DILocation(line: 1473, column: 36, scope: !2383)
!2385 = !DILocation(line: 1473, column: 47, scope: !2383)
!2386 = !DILocation(line: 1473, column: 3, scope: !2383)
!2387 = !DILocation(line: 1474, column: 18, scope: !2383)
!2388 = !DILocation(line: 1474, column: 36, scope: !2383)
!2389 = !DILocation(line: 1474, column: 47, scope: !2383)
!2390 = !DILocation(line: 1474, column: 3, scope: !2383)
!2391 = !DILocation(line: 1475, column: 18, scope: !2383)
!2392 = !DILocation(line: 1475, column: 36, scope: !2383)
!2393 = !DILocation(line: 1475, column: 47, scope: !2383)
!2394 = !DILocation(line: 1475, column: 3, scope: !2383)
!2395 = !DILocation(line: 1477, column: 1, scope: !2358)
!2396 = distinct !DISubprogram(name: "evolve_gpu", linkageName: "_ZL10evolve_gpuP8dcomplexS0_Pd", scope: !3, file: !3, line: 1428, type: !2236, scopeLine: 1430, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1051)
!2397 = !DILocalVariable(name: "u0", arg: 1, scope: !2396, file: !3, line: 1428, type: !98)
!2398 = !DILocation(line: 0, scope: !2396)
!2399 = !DILocalVariable(name: "u1", arg: 2, scope: !2396, file: !3, line: 1429, type: !98)
!2400 = !DILocalVariable(name: "twiddle", arg: 3, scope: !2396, file: !3, line: 1430, type: !106)
!2401 = !DILocation(line: 1434, column: 22, scope: !2396)
!2402 = !DILocation(line: 1435, column: 3, scope: !2396)
!2403 = !DILocation(line: 1434, column: 19, scope: !2396)
!2404 = !DILocation(line: 1442, column: 1, scope: !2396)
!2405 = distinct !DISubprogram(name: "checksum_gpu", linkageName: "_ZL12checksum_gpuiP8dcomplex", scope: !3, file: !3, line: 1308, type: !2406, scopeLine: 1309, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1051)
!2406 = !DISubroutineType(types: !2407)
!2407 = !{null, !97, !98}
!2408 = !DILocalVariable(name: "iteration", arg: 1, scope: !2405, file: !3, line: 1308, type: !97)
!2409 = !DILocation(line: 0, scope: !2405)
!2410 = !DILocalVariable(name: "u1", arg: 2, scope: !2405, file: !3, line: 1309, type: !98)
!2411 = !DILocation(line: 1313, column: 24, scope: !2405)
!2412 = !DILocation(line: 1314, column: 3, scope: !2405)
!2413 = !DILocation(line: 1313, column: 21, scope: !2405)
!2414 = !DILocation(line: 1317, column: 5, scope: !2405)
!2415 = !DILocation(line: 1321, column: 1, scope: !2405)
!2416 = distinct !DISubprogram(name: "verify", linkageName: "_ZL6verifyiiiiPiPc", scope: !3, file: !3, line: 1838, type: !2417, scopeLine: 1843, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1051)
!2417 = !DISubroutineType(types: !2418)
!2418 = !{null, !97, !97, !97, !97, !2419, !108}
!2419 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !1750, size: 64)
!2420 = !DILocalVariable(name: "d1", arg: 1, scope: !2416, file: !3, line: 1838, type: !97)
!2421 = !DILocation(line: 0, scope: !2416)
!2422 = !DILocalVariable(name: "d2", arg: 2, scope: !2416, file: !3, line: 1839, type: !97)
!2423 = !DILocalVariable(name: "d3", arg: 3, scope: !2416, file: !3, line: 1840, type: !97)
!2424 = !DILocalVariable(name: "nt", arg: 4, scope: !2416, file: !3, line: 1841, type: !97)
!2425 = !DILocalVariable(name: "verified", arg: 5, scope: !2416, file: !3, line: 1842, type: !2419)
!2426 = !DILocalVariable(name: "class_npb", arg: 6, scope: !2416, file: !3, line: 1843, type: !108)
!2427 = !DILocalVariable(name: "csum_ref", scope: !2416, file: !3, line: 1851, type: !2428)
!2428 = !DICompositeType(tag: DW_TAG_array_type, baseType: !99, size: 3328, elements: !2429)
!2429 = !{!2430}
!2430 = !DISubrange(count: 26)
!2431 = !DILocation(line: 1851, column: 11, scope: !2416)
!2432 = !DILocation(line: 1852, column: 13, scope: !2416)
!2433 = !DILocalVariable(name: "epsilon", scope: !2416, file: !3, line: 1845, type: !104)
!2434 = !DILocation(line: 1854, column: 12, scope: !2416)
!2435 = !DILocation(line: 1855, column: 8, scope: !2436)
!2436 = distinct !DILexicalBlock(scope: !2416, file: !3, line: 1855, column: 5)
!2437 = !DILocation(line: 1855, column: 14, scope: !2436)
!2438 = !DILocation(line: 1855, column: 20, scope: !2436)
!2439 = !DILocation(line: 1855, column: 26, scope: !2436)
!2440 = !DILocation(line: 1855, column: 32, scope: !2436)
!2441 = !DILocation(line: 1855, column: 38, scope: !2436)
!2442 = !DILocation(line: 1855, column: 44, scope: !2436)
!2443 = !DILocation(line: 1855, column: 5, scope: !2416)
!2444 = !DILocation(line: 1861, column: 14, scope: !2445)
!2445 = distinct !DILexicalBlock(scope: !2436, file: !3, line: 1855, column: 49)
!2446 = !DILocation(line: 1862, column: 17, scope: !2445)
!2447 = !DILocation(line: 1862, column: 3, scope: !2445)
!2448 = !DILocation(line: 1862, column: 15, scope: !2445)
!2449 = !DILocation(line: 1863, column: 17, scope: !2445)
!2450 = !DILocation(line: 1863, column: 3, scope: !2445)
!2451 = !DILocation(line: 1863, column: 15, scope: !2445)
!2452 = !DILocation(line: 1864, column: 17, scope: !2445)
!2453 = !DILocation(line: 1864, column: 3, scope: !2445)
!2454 = !DILocation(line: 1864, column: 15, scope: !2445)
!2455 = !DILocation(line: 1865, column: 17, scope: !2445)
!2456 = !DILocation(line: 1865, column: 3, scope: !2445)
!2457 = !DILocation(line: 1865, column: 15, scope: !2445)
!2458 = !DILocation(line: 1866, column: 17, scope: !2445)
!2459 = !DILocation(line: 1866, column: 3, scope: !2445)
!2460 = !DILocation(line: 1866, column: 15, scope: !2445)
!2461 = !DILocation(line: 1867, column: 17, scope: !2445)
!2462 = !DILocation(line: 1867, column: 3, scope: !2445)
!2463 = !DILocation(line: 1867, column: 15, scope: !2445)
!2464 = !DILocation(line: 1868, column: 2, scope: !2445)
!2465 = !DILocation(line: 1868, column: 14, scope: !2466)
!2466 = distinct !DILexicalBlock(scope: !2436, file: !3, line: 1868, column: 11)
!2467 = !DILocation(line: 1868, column: 21, scope: !2466)
!2468 = !DILocation(line: 1868, column: 27, scope: !2466)
!2469 = !DILocation(line: 1868, column: 34, scope: !2466)
!2470 = !DILocation(line: 1868, column: 40, scope: !2466)
!2471 = !DILocation(line: 1868, column: 46, scope: !2466)
!2472 = !DILocation(line: 1868, column: 52, scope: !2466)
!2473 = !DILocation(line: 1868, column: 11, scope: !2436)
!2474 = !DILocation(line: 1874, column: 14, scope: !2475)
!2475 = distinct !DILexicalBlock(scope: !2466, file: !3, line: 1868, column: 57)
!2476 = !DILocation(line: 1875, column: 17, scope: !2475)
!2477 = !DILocation(line: 1875, column: 3, scope: !2475)
!2478 = !DILocation(line: 1875, column: 15, scope: !2475)
!2479 = !DILocation(line: 1876, column: 17, scope: !2475)
!2480 = !DILocation(line: 1876, column: 3, scope: !2475)
!2481 = !DILocation(line: 1876, column: 15, scope: !2475)
!2482 = !DILocation(line: 1877, column: 17, scope: !2475)
!2483 = !DILocation(line: 1877, column: 3, scope: !2475)
!2484 = !DILocation(line: 1877, column: 15, scope: !2475)
!2485 = !DILocation(line: 1878, column: 17, scope: !2475)
!2486 = !DILocation(line: 1878, column: 3, scope: !2475)
!2487 = !DILocation(line: 1878, column: 15, scope: !2475)
!2488 = !DILocation(line: 1879, column: 17, scope: !2475)
!2489 = !DILocation(line: 1879, column: 3, scope: !2475)
!2490 = !DILocation(line: 1879, column: 15, scope: !2475)
!2491 = !DILocation(line: 1880, column: 17, scope: !2475)
!2492 = !DILocation(line: 1880, column: 3, scope: !2475)
!2493 = !DILocation(line: 1880, column: 15, scope: !2475)
!2494 = !DILocation(line: 1881, column: 2, scope: !2475)
!2495 = !DILocation(line: 1881, column: 14, scope: !2496)
!2496 = distinct !DILexicalBlock(scope: !2466, file: !3, line: 1881, column: 11)
!2497 = !DILocation(line: 1881, column: 21, scope: !2496)
!2498 = !DILocation(line: 1881, column: 27, scope: !2496)
!2499 = !DILocation(line: 1881, column: 34, scope: !2496)
!2500 = !DILocation(line: 1881, column: 40, scope: !2496)
!2501 = !DILocation(line: 1881, column: 47, scope: !2496)
!2502 = !DILocation(line: 1881, column: 53, scope: !2496)
!2503 = !DILocation(line: 1881, column: 11, scope: !2466)
!2504 = !DILocation(line: 1887, column: 14, scope: !2505)
!2505 = distinct !DILexicalBlock(scope: !2496, file: !3, line: 1881, column: 58)
!2506 = !DILocation(line: 1888, column: 17, scope: !2505)
!2507 = !DILocation(line: 1888, column: 3, scope: !2505)
!2508 = !DILocation(line: 1888, column: 15, scope: !2505)
!2509 = !DILocation(line: 1889, column: 17, scope: !2505)
!2510 = !DILocation(line: 1889, column: 3, scope: !2505)
!2511 = !DILocation(line: 1889, column: 15, scope: !2505)
!2512 = !DILocation(line: 1890, column: 17, scope: !2505)
!2513 = !DILocation(line: 1890, column: 3, scope: !2505)
!2514 = !DILocation(line: 1890, column: 15, scope: !2505)
!2515 = !DILocation(line: 1891, column: 17, scope: !2505)
!2516 = !DILocation(line: 1891, column: 3, scope: !2505)
!2517 = !DILocation(line: 1891, column: 15, scope: !2505)
!2518 = !DILocation(line: 1892, column: 17, scope: !2505)
!2519 = !DILocation(line: 1892, column: 3, scope: !2505)
!2520 = !DILocation(line: 1892, column: 15, scope: !2505)
!2521 = !DILocation(line: 1893, column: 17, scope: !2505)
!2522 = !DILocation(line: 1893, column: 3, scope: !2505)
!2523 = !DILocation(line: 1893, column: 15, scope: !2505)
!2524 = !DILocation(line: 1894, column: 2, scope: !2505)
!2525 = !DILocation(line: 1894, column: 14, scope: !2526)
!2526 = distinct !DILexicalBlock(scope: !2496, file: !3, line: 1894, column: 11)
!2527 = !DILocation(line: 1894, column: 21, scope: !2526)
!2528 = !DILocation(line: 1894, column: 27, scope: !2526)
!2529 = !DILocation(line: 1894, column: 34, scope: !2526)
!2530 = !DILocation(line: 1894, column: 40, scope: !2526)
!2531 = !DILocation(line: 1894, column: 47, scope: !2526)
!2532 = !DILocation(line: 1894, column: 53, scope: !2526)
!2533 = !DILocation(line: 1894, column: 11, scope: !2496)
!2534 = !DILocation(line: 1900, column: 14, scope: !2535)
!2535 = distinct !DILexicalBlock(scope: !2526, file: !3, line: 1894, column: 59)
!2536 = !DILocation(line: 1901, column: 18, scope: !2535)
!2537 = !DILocation(line: 1901, column: 3, scope: !2535)
!2538 = !DILocation(line: 1901, column: 16, scope: !2535)
!2539 = !DILocation(line: 1902, column: 18, scope: !2535)
!2540 = !DILocation(line: 1902, column: 3, scope: !2535)
!2541 = !DILocation(line: 1902, column: 16, scope: !2535)
!2542 = !DILocation(line: 1903, column: 18, scope: !2535)
!2543 = !DILocation(line: 1903, column: 3, scope: !2535)
!2544 = !DILocation(line: 1903, column: 16, scope: !2535)
!2545 = !DILocation(line: 1904, column: 18, scope: !2535)
!2546 = !DILocation(line: 1904, column: 3, scope: !2535)
!2547 = !DILocation(line: 1904, column: 16, scope: !2535)
!2548 = !DILocation(line: 1905, column: 18, scope: !2535)
!2549 = !DILocation(line: 1905, column: 3, scope: !2535)
!2550 = !DILocation(line: 1905, column: 16, scope: !2535)
!2551 = !DILocation(line: 1906, column: 18, scope: !2535)
!2552 = !DILocation(line: 1906, column: 3, scope: !2535)
!2553 = !DILocation(line: 1906, column: 16, scope: !2535)
!2554 = !DILocation(line: 1907, column: 18, scope: !2535)
!2555 = !DILocation(line: 1907, column: 3, scope: !2535)
!2556 = !DILocation(line: 1907, column: 16, scope: !2535)
!2557 = !DILocation(line: 1908, column: 18, scope: !2535)
!2558 = !DILocation(line: 1908, column: 3, scope: !2535)
!2559 = !DILocation(line: 1908, column: 16, scope: !2535)
!2560 = !DILocation(line: 1909, column: 18, scope: !2535)
!2561 = !DILocation(line: 1909, column: 3, scope: !2535)
!2562 = !DILocation(line: 1909, column: 16, scope: !2535)
!2563 = !DILocation(line: 1910, column: 18, scope: !2535)
!2564 = !DILocation(line: 1910, column: 3, scope: !2535)
!2565 = !DILocation(line: 1910, column: 16, scope: !2535)
!2566 = !DILocation(line: 1911, column: 18, scope: !2535)
!2567 = !DILocation(line: 1911, column: 3, scope: !2535)
!2568 = !DILocation(line: 1911, column: 16, scope: !2535)
!2569 = !DILocation(line: 1912, column: 18, scope: !2535)
!2570 = !DILocation(line: 1912, column: 3, scope: !2535)
!2571 = !DILocation(line: 1912, column: 16, scope: !2535)
!2572 = !DILocation(line: 1913, column: 18, scope: !2535)
!2573 = !DILocation(line: 1913, column: 3, scope: !2535)
!2574 = !DILocation(line: 1913, column: 16, scope: !2535)
!2575 = !DILocation(line: 1914, column: 18, scope: !2535)
!2576 = !DILocation(line: 1914, column: 3, scope: !2535)
!2577 = !DILocation(line: 1914, column: 16, scope: !2535)
!2578 = !DILocation(line: 1915, column: 18, scope: !2535)
!2579 = !DILocation(line: 1915, column: 3, scope: !2535)
!2580 = !DILocation(line: 1915, column: 16, scope: !2535)
!2581 = !DILocation(line: 1916, column: 18, scope: !2535)
!2582 = !DILocation(line: 1916, column: 3, scope: !2535)
!2583 = !DILocation(line: 1916, column: 16, scope: !2535)
!2584 = !DILocation(line: 1917, column: 18, scope: !2535)
!2585 = !DILocation(line: 1917, column: 3, scope: !2535)
!2586 = !DILocation(line: 1917, column: 16, scope: !2535)
!2587 = !DILocation(line: 1918, column: 18, scope: !2535)
!2588 = !DILocation(line: 1918, column: 3, scope: !2535)
!2589 = !DILocation(line: 1918, column: 16, scope: !2535)
!2590 = !DILocation(line: 1919, column: 18, scope: !2535)
!2591 = !DILocation(line: 1919, column: 3, scope: !2535)
!2592 = !DILocation(line: 1919, column: 16, scope: !2535)
!2593 = !DILocation(line: 1920, column: 18, scope: !2535)
!2594 = !DILocation(line: 1920, column: 3, scope: !2535)
!2595 = !DILocation(line: 1920, column: 16, scope: !2535)
!2596 = !DILocation(line: 1921, column: 2, scope: !2535)
!2597 = !DILocation(line: 1921, column: 14, scope: !2598)
!2598 = distinct !DILexicalBlock(scope: !2526, file: !3, line: 1921, column: 11)
!2599 = !DILocation(line: 1921, column: 21, scope: !2598)
!2600 = !DILocation(line: 1921, column: 27, scope: !2598)
!2601 = !DILocation(line: 1921, column: 34, scope: !2598)
!2602 = !DILocation(line: 1921, column: 40, scope: !2598)
!2603 = !DILocation(line: 1921, column: 47, scope: !2598)
!2604 = !DILocation(line: 1921, column: 53, scope: !2598)
!2605 = !DILocation(line: 1921, column: 11, scope: !2526)
!2606 = !DILocation(line: 1927, column: 14, scope: !2607)
!2607 = distinct !DILexicalBlock(scope: !2598, file: !3, line: 1921, column: 59)
!2608 = !DILocation(line: 1928, column: 18, scope: !2607)
!2609 = !DILocation(line: 1928, column: 3, scope: !2607)
!2610 = !DILocation(line: 1928, column: 16, scope: !2607)
!2611 = !DILocation(line: 1929, column: 18, scope: !2607)
!2612 = !DILocation(line: 1929, column: 3, scope: !2607)
!2613 = !DILocation(line: 1929, column: 16, scope: !2607)
!2614 = !DILocation(line: 1930, column: 18, scope: !2607)
!2615 = !DILocation(line: 1930, column: 3, scope: !2607)
!2616 = !DILocation(line: 1930, column: 16, scope: !2607)
!2617 = !DILocation(line: 1931, column: 18, scope: !2607)
!2618 = !DILocation(line: 1931, column: 3, scope: !2607)
!2619 = !DILocation(line: 1931, column: 16, scope: !2607)
!2620 = !DILocation(line: 1932, column: 18, scope: !2607)
!2621 = !DILocation(line: 1932, column: 3, scope: !2607)
!2622 = !DILocation(line: 1932, column: 16, scope: !2607)
!2623 = !DILocation(line: 1933, column: 18, scope: !2607)
!2624 = !DILocation(line: 1933, column: 3, scope: !2607)
!2625 = !DILocation(line: 1933, column: 16, scope: !2607)
!2626 = !DILocation(line: 1934, column: 18, scope: !2607)
!2627 = !DILocation(line: 1934, column: 3, scope: !2607)
!2628 = !DILocation(line: 1934, column: 16, scope: !2607)
!2629 = !DILocation(line: 1935, column: 18, scope: !2607)
!2630 = !DILocation(line: 1935, column: 3, scope: !2607)
!2631 = !DILocation(line: 1935, column: 16, scope: !2607)
!2632 = !DILocation(line: 1936, column: 18, scope: !2607)
!2633 = !DILocation(line: 1936, column: 3, scope: !2607)
!2634 = !DILocation(line: 1936, column: 16, scope: !2607)
!2635 = !DILocation(line: 1937, column: 18, scope: !2607)
!2636 = !DILocation(line: 1937, column: 3, scope: !2607)
!2637 = !DILocation(line: 1937, column: 16, scope: !2607)
!2638 = !DILocation(line: 1938, column: 18, scope: !2607)
!2639 = !DILocation(line: 1938, column: 3, scope: !2607)
!2640 = !DILocation(line: 1938, column: 16, scope: !2607)
!2641 = !DILocation(line: 1939, column: 18, scope: !2607)
!2642 = !DILocation(line: 1939, column: 3, scope: !2607)
!2643 = !DILocation(line: 1939, column: 16, scope: !2607)
!2644 = !DILocation(line: 1940, column: 18, scope: !2607)
!2645 = !DILocation(line: 1940, column: 3, scope: !2607)
!2646 = !DILocation(line: 1940, column: 16, scope: !2607)
!2647 = !DILocation(line: 1941, column: 18, scope: !2607)
!2648 = !DILocation(line: 1941, column: 3, scope: !2607)
!2649 = !DILocation(line: 1941, column: 16, scope: !2607)
!2650 = !DILocation(line: 1942, column: 18, scope: !2607)
!2651 = !DILocation(line: 1942, column: 3, scope: !2607)
!2652 = !DILocation(line: 1942, column: 16, scope: !2607)
!2653 = !DILocation(line: 1943, column: 18, scope: !2607)
!2654 = !DILocation(line: 1943, column: 3, scope: !2607)
!2655 = !DILocation(line: 1943, column: 16, scope: !2607)
!2656 = !DILocation(line: 1944, column: 18, scope: !2607)
!2657 = !DILocation(line: 1944, column: 3, scope: !2607)
!2658 = !DILocation(line: 1944, column: 16, scope: !2607)
!2659 = !DILocation(line: 1945, column: 18, scope: !2607)
!2660 = !DILocation(line: 1945, column: 3, scope: !2607)
!2661 = !DILocation(line: 1945, column: 16, scope: !2607)
!2662 = !DILocation(line: 1946, column: 18, scope: !2607)
!2663 = !DILocation(line: 1946, column: 3, scope: !2607)
!2664 = !DILocation(line: 1946, column: 16, scope: !2607)
!2665 = !DILocation(line: 1947, column: 18, scope: !2607)
!2666 = !DILocation(line: 1947, column: 3, scope: !2607)
!2667 = !DILocation(line: 1947, column: 16, scope: !2607)
!2668 = !DILocation(line: 1948, column: 2, scope: !2607)
!2669 = !DILocation(line: 1948, column: 14, scope: !2670)
!2670 = distinct !DILexicalBlock(scope: !2598, file: !3, line: 1948, column: 11)
!2671 = !DILocation(line: 1948, column: 22, scope: !2670)
!2672 = !DILocation(line: 1948, column: 28, scope: !2670)
!2673 = !DILocation(line: 1948, column: 36, scope: !2670)
!2674 = !DILocation(line: 1948, column: 42, scope: !2670)
!2675 = !DILocation(line: 1948, column: 50, scope: !2670)
!2676 = !DILocation(line: 1948, column: 56, scope: !2670)
!2677 = !DILocation(line: 1948, column: 11, scope: !2598)
!2678 = !DILocation(line: 1954, column: 14, scope: !2679)
!2679 = distinct !DILexicalBlock(scope: !2670, file: !3, line: 1948, column: 62)
!2680 = !DILocation(line: 1955, column: 18, scope: !2679)
!2681 = !DILocation(line: 1955, column: 3, scope: !2679)
!2682 = !DILocation(line: 1955, column: 16, scope: !2679)
!2683 = !DILocation(line: 1956, column: 18, scope: !2679)
!2684 = !DILocation(line: 1956, column: 3, scope: !2679)
!2685 = !DILocation(line: 1956, column: 16, scope: !2679)
!2686 = !DILocation(line: 1957, column: 18, scope: !2679)
!2687 = !DILocation(line: 1957, column: 3, scope: !2679)
!2688 = !DILocation(line: 1957, column: 16, scope: !2679)
!2689 = !DILocation(line: 1958, column: 18, scope: !2679)
!2690 = !DILocation(line: 1958, column: 3, scope: !2679)
!2691 = !DILocation(line: 1958, column: 16, scope: !2679)
!2692 = !DILocation(line: 1959, column: 18, scope: !2679)
!2693 = !DILocation(line: 1959, column: 3, scope: !2679)
!2694 = !DILocation(line: 1959, column: 16, scope: !2679)
!2695 = !DILocation(line: 1960, column: 18, scope: !2679)
!2696 = !DILocation(line: 1960, column: 3, scope: !2679)
!2697 = !DILocation(line: 1960, column: 16, scope: !2679)
!2698 = !DILocation(line: 1961, column: 18, scope: !2679)
!2699 = !DILocation(line: 1961, column: 3, scope: !2679)
!2700 = !DILocation(line: 1961, column: 16, scope: !2679)
!2701 = !DILocation(line: 1962, column: 18, scope: !2679)
!2702 = !DILocation(line: 1962, column: 3, scope: !2679)
!2703 = !DILocation(line: 1962, column: 16, scope: !2679)
!2704 = !DILocation(line: 1963, column: 18, scope: !2679)
!2705 = !DILocation(line: 1963, column: 3, scope: !2679)
!2706 = !DILocation(line: 1963, column: 16, scope: !2679)
!2707 = !DILocation(line: 1964, column: 18, scope: !2679)
!2708 = !DILocation(line: 1964, column: 3, scope: !2679)
!2709 = !DILocation(line: 1964, column: 16, scope: !2679)
!2710 = !DILocation(line: 1965, column: 18, scope: !2679)
!2711 = !DILocation(line: 1965, column: 3, scope: !2679)
!2712 = !DILocation(line: 1965, column: 16, scope: !2679)
!2713 = !DILocation(line: 1966, column: 18, scope: !2679)
!2714 = !DILocation(line: 1966, column: 3, scope: !2679)
!2715 = !DILocation(line: 1966, column: 16, scope: !2679)
!2716 = !DILocation(line: 1967, column: 18, scope: !2679)
!2717 = !DILocation(line: 1967, column: 3, scope: !2679)
!2718 = !DILocation(line: 1967, column: 16, scope: !2679)
!2719 = !DILocation(line: 1968, column: 18, scope: !2679)
!2720 = !DILocation(line: 1968, column: 3, scope: !2679)
!2721 = !DILocation(line: 1968, column: 16, scope: !2679)
!2722 = !DILocation(line: 1969, column: 18, scope: !2679)
!2723 = !DILocation(line: 1969, column: 3, scope: !2679)
!2724 = !DILocation(line: 1969, column: 16, scope: !2679)
!2725 = !DILocation(line: 1970, column: 18, scope: !2679)
!2726 = !DILocation(line: 1970, column: 3, scope: !2679)
!2727 = !DILocation(line: 1970, column: 16, scope: !2679)
!2728 = !DILocation(line: 1971, column: 18, scope: !2679)
!2729 = !DILocation(line: 1971, column: 3, scope: !2679)
!2730 = !DILocation(line: 1971, column: 16, scope: !2679)
!2731 = !DILocation(line: 1972, column: 18, scope: !2679)
!2732 = !DILocation(line: 1972, column: 3, scope: !2679)
!2733 = !DILocation(line: 1972, column: 16, scope: !2679)
!2734 = !DILocation(line: 1973, column: 18, scope: !2679)
!2735 = !DILocation(line: 1973, column: 3, scope: !2679)
!2736 = !DILocation(line: 1973, column: 16, scope: !2679)
!2737 = !DILocation(line: 1974, column: 18, scope: !2679)
!2738 = !DILocation(line: 1974, column: 3, scope: !2679)
!2739 = !DILocation(line: 1974, column: 16, scope: !2679)
!2740 = !DILocation(line: 1975, column: 18, scope: !2679)
!2741 = !DILocation(line: 1975, column: 3, scope: !2679)
!2742 = !DILocation(line: 1975, column: 16, scope: !2679)
!2743 = !DILocation(line: 1976, column: 18, scope: !2679)
!2744 = !DILocation(line: 1976, column: 3, scope: !2679)
!2745 = !DILocation(line: 1976, column: 16, scope: !2679)
!2746 = !DILocation(line: 1977, column: 18, scope: !2679)
!2747 = !DILocation(line: 1977, column: 3, scope: !2679)
!2748 = !DILocation(line: 1977, column: 16, scope: !2679)
!2749 = !DILocation(line: 1978, column: 18, scope: !2679)
!2750 = !DILocation(line: 1978, column: 3, scope: !2679)
!2751 = !DILocation(line: 1978, column: 16, scope: !2679)
!2752 = !DILocation(line: 1979, column: 18, scope: !2679)
!2753 = !DILocation(line: 1979, column: 3, scope: !2679)
!2754 = !DILocation(line: 1979, column: 16, scope: !2679)
!2755 = !DILocation(line: 1980, column: 2, scope: !2679)
!2756 = !DILocation(line: 1980, column: 14, scope: !2757)
!2757 = distinct !DILexicalBlock(scope: !2670, file: !3, line: 1980, column: 11)
!2758 = !DILocation(line: 1980, column: 22, scope: !2757)
!2759 = !DILocation(line: 1980, column: 28, scope: !2757)
!2760 = !DILocation(line: 1980, column: 36, scope: !2757)
!2761 = !DILocation(line: 1980, column: 42, scope: !2757)
!2762 = !DILocation(line: 1980, column: 50, scope: !2757)
!2763 = !DILocation(line: 1980, column: 56, scope: !2757)
!2764 = !DILocation(line: 1980, column: 11, scope: !2670)
!2765 = !DILocation(line: 1986, column: 14, scope: !2766)
!2766 = distinct !DILexicalBlock(scope: !2757, file: !3, line: 1980, column: 62)
!2767 = !DILocation(line: 1987, column: 18, scope: !2766)
!2768 = !DILocation(line: 1987, column: 3, scope: !2766)
!2769 = !DILocation(line: 1987, column: 16, scope: !2766)
!2770 = !DILocation(line: 1988, column: 18, scope: !2766)
!2771 = !DILocation(line: 1988, column: 3, scope: !2766)
!2772 = !DILocation(line: 1988, column: 16, scope: !2766)
!2773 = !DILocation(line: 1989, column: 18, scope: !2766)
!2774 = !DILocation(line: 1989, column: 3, scope: !2766)
!2775 = !DILocation(line: 1989, column: 16, scope: !2766)
!2776 = !DILocation(line: 1990, column: 18, scope: !2766)
!2777 = !DILocation(line: 1990, column: 3, scope: !2766)
!2778 = !DILocation(line: 1990, column: 16, scope: !2766)
!2779 = !DILocation(line: 1991, column: 18, scope: !2766)
!2780 = !DILocation(line: 1991, column: 3, scope: !2766)
!2781 = !DILocation(line: 1991, column: 16, scope: !2766)
!2782 = !DILocation(line: 1992, column: 18, scope: !2766)
!2783 = !DILocation(line: 1992, column: 3, scope: !2766)
!2784 = !DILocation(line: 1992, column: 16, scope: !2766)
!2785 = !DILocation(line: 1993, column: 18, scope: !2766)
!2786 = !DILocation(line: 1993, column: 3, scope: !2766)
!2787 = !DILocation(line: 1993, column: 16, scope: !2766)
!2788 = !DILocation(line: 1994, column: 18, scope: !2766)
!2789 = !DILocation(line: 1994, column: 3, scope: !2766)
!2790 = !DILocation(line: 1994, column: 16, scope: !2766)
!2791 = !DILocation(line: 1995, column: 18, scope: !2766)
!2792 = !DILocation(line: 1995, column: 3, scope: !2766)
!2793 = !DILocation(line: 1995, column: 16, scope: !2766)
!2794 = !DILocation(line: 1996, column: 18, scope: !2766)
!2795 = !DILocation(line: 1996, column: 3, scope: !2766)
!2796 = !DILocation(line: 1996, column: 16, scope: !2766)
!2797 = !DILocation(line: 1997, column: 18, scope: !2766)
!2798 = !DILocation(line: 1997, column: 3, scope: !2766)
!2799 = !DILocation(line: 1997, column: 16, scope: !2766)
!2800 = !DILocation(line: 1998, column: 18, scope: !2766)
!2801 = !DILocation(line: 1998, column: 3, scope: !2766)
!2802 = !DILocation(line: 1998, column: 16, scope: !2766)
!2803 = !DILocation(line: 1999, column: 18, scope: !2766)
!2804 = !DILocation(line: 1999, column: 3, scope: !2766)
!2805 = !DILocation(line: 1999, column: 16, scope: !2766)
!2806 = !DILocation(line: 2000, column: 18, scope: !2766)
!2807 = !DILocation(line: 2000, column: 3, scope: !2766)
!2808 = !DILocation(line: 2000, column: 16, scope: !2766)
!2809 = !DILocation(line: 2001, column: 18, scope: !2766)
!2810 = !DILocation(line: 2001, column: 3, scope: !2766)
!2811 = !DILocation(line: 2001, column: 16, scope: !2766)
!2812 = !DILocation(line: 2002, column: 18, scope: !2766)
!2813 = !DILocation(line: 2002, column: 3, scope: !2766)
!2814 = !DILocation(line: 2002, column: 16, scope: !2766)
!2815 = !DILocation(line: 2003, column: 18, scope: !2766)
!2816 = !DILocation(line: 2003, column: 3, scope: !2766)
!2817 = !DILocation(line: 2003, column: 16, scope: !2766)
!2818 = !DILocation(line: 2004, column: 18, scope: !2766)
!2819 = !DILocation(line: 2004, column: 3, scope: !2766)
!2820 = !DILocation(line: 2004, column: 16, scope: !2766)
!2821 = !DILocation(line: 2005, column: 18, scope: !2766)
!2822 = !DILocation(line: 2005, column: 3, scope: !2766)
!2823 = !DILocation(line: 2005, column: 16, scope: !2766)
!2824 = !DILocation(line: 2006, column: 18, scope: !2766)
!2825 = !DILocation(line: 2006, column: 3, scope: !2766)
!2826 = !DILocation(line: 2006, column: 16, scope: !2766)
!2827 = !DILocation(line: 2007, column: 18, scope: !2766)
!2828 = !DILocation(line: 2007, column: 3, scope: !2766)
!2829 = !DILocation(line: 2007, column: 16, scope: !2766)
!2830 = !DILocation(line: 2008, column: 18, scope: !2766)
!2831 = !DILocation(line: 2008, column: 3, scope: !2766)
!2832 = !DILocation(line: 2008, column: 16, scope: !2766)
!2833 = !DILocation(line: 2009, column: 18, scope: !2766)
!2834 = !DILocation(line: 2009, column: 3, scope: !2766)
!2835 = !DILocation(line: 2009, column: 16, scope: !2766)
!2836 = !DILocation(line: 2010, column: 18, scope: !2766)
!2837 = !DILocation(line: 2010, column: 3, scope: !2766)
!2838 = !DILocation(line: 2010, column: 16, scope: !2766)
!2839 = !DILocation(line: 2011, column: 18, scope: !2766)
!2840 = !DILocation(line: 2011, column: 3, scope: !2766)
!2841 = !DILocation(line: 2011, column: 16, scope: !2766)
!2842 = !DILocation(line: 2012, column: 2, scope: !2766)
!2843 = !DILocation(line: 2013, column: 5, scope: !2844)
!2844 = distinct !DILexicalBlock(scope: !2416, file: !3, line: 2013, column: 5)
!2845 = !DILocation(line: 2013, column: 16, scope: !2844)
!2846 = !DILocation(line: 2013, column: 5, scope: !2416)
!2847 = !DILocation(line: 2014, column: 13, scope: !2848)
!2848 = distinct !DILexicalBlock(scope: !2844, file: !3, line: 2013, column: 23)
!2849 = !DILocalVariable(name: "i", scope: !2416, file: !3, line: 1844, type: !97)
!2850 = !DILocation(line: 2015, column: 7, scope: !2851)
!2851 = distinct !DILexicalBlock(scope: !2848, file: !3, line: 2015, column: 3)
!2852 = !DILocation(line: 0, scope: !2851)
!2853 = !DILocation(line: 2015, column: 16, scope: !2854)
!2854 = distinct !DILexicalBlock(scope: !2851, file: !3, line: 2015, column: 3)
!2855 = !DILocation(line: 2015, column: 3, scope: !2851)
!2856 = !DILocation(line: 2016, column: 10, scope: !2857)
!2857 = distinct !DILexicalBlock(scope: !2854, file: !3, line: 2015, column: 27)
!2858 = !DILocalVariable(name: "err", scope: !2416, file: !3, line: 1845, type: !104)
!2859 = !DILocation(line: 2018, column: 13, scope: !2860)
!2860 = distinct !DILexicalBlock(scope: !2857, file: !3, line: 2018, column: 7)
!2861 = !DILocation(line: 2018, column: 7, scope: !2857)
!2862 = !DILocation(line: 2019, column: 15, scope: !2863)
!2863 = distinct !DILexicalBlock(scope: !2860, file: !3, line: 2018, column: 25)
!2864 = !DILocation(line: 2020, column: 5, scope: !2863)
!2865 = !DILocation(line: 2022, column: 3, scope: !2857)
!2866 = !DILocation(line: 2015, column: 24, scope: !2854)
!2867 = !DILocation(line: 2015, column: 3, scope: !2854)
!2868 = distinct !{!2868, !2855, !2869}
!2869 = !DILocation(line: 2022, column: 3, scope: !2851)
!2870 = !DILocation(line: 2023, column: 2, scope: !2848)
!2871 = !DILocation(line: 2024, column: 5, scope: !2872)
!2872 = distinct !DILexicalBlock(scope: !2416, file: !3, line: 2024, column: 5)
!2873 = !DILocation(line: 2024, column: 16, scope: !2872)
!2874 = !DILocation(line: 2024, column: 5, scope: !2416)
!2875 = !DILocation(line: 2025, column: 6, scope: !2876)
!2876 = distinct !DILexicalBlock(scope: !2877, file: !3, line: 2025, column: 6)
!2877 = distinct !DILexicalBlock(scope: !2872, file: !3, line: 2024, column: 23)
!2878 = !DILocation(line: 2025, column: 6, scope: !2877)
!2879 = !DILocation(line: 2026, column: 4, scope: !2880)
!2880 = distinct !DILexicalBlock(scope: !2876, file: !3, line: 2025, column: 16)
!2881 = !DILocation(line: 2027, column: 3, scope: !2880)
!2882 = !DILocation(line: 2028, column: 4, scope: !2883)
!2883 = distinct !DILexicalBlock(scope: !2876, file: !3, line: 2027, column: 8)
!2884 = !DILocation(line: 2030, column: 2, scope: !2877)
!2885 = !DILocation(line: 2031, column: 30, scope: !2416)
!2886 = !DILocation(line: 2031, column: 2, scope: !2416)
!2887 = !DILocation(line: 2032, column: 1, scope: !2416)
!2888 = distinct !DISubprogram(name: "release_gpu", linkageName: "_ZL11release_gpuv", scope: !3, file: !3, line: 1648, type: !561, scopeLine: 1648, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1051)
!2889 = !DILocation(line: 1657, column: 1, scope: !2888)
!2890 = distinct !DISubprogram(name: "dcomplex_div", linkageName: "_ZL12dcomplex_div8dcomplexS_", scope: !100, file: !100, line: 106, type: !2891, scopeLine: 106, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1051)
!2891 = !DISubroutineType(types: !2892)
!2892 = !{!99, !99, !99}
!2893 = !DILocalVariable(name: "z1", arg: 1, scope: !2890, file: !100, line: 106, type: !99)
!2894 = !DILocation(line: 106, column: 46, scope: !2890)
!2895 = !DILocalVariable(name: "z2", arg: 2, scope: !2890, file: !100, line: 106, type: !99)
!2896 = !DILocation(line: 106, column: 59, scope: !2890)
!2897 = !DILocation(line: 107, column: 16, scope: !2890)
!2898 = !DILocalVariable(name: "a", scope: !2890, file: !100, line: 107, type: !104)
!2899 = !DILocation(line: 0, scope: !2890)
!2900 = !DILocation(line: 108, column: 16, scope: !2890)
!2901 = !DILocalVariable(name: "b", scope: !2890, file: !100, line: 108, type: !104)
!2902 = !DILocation(line: 109, column: 16, scope: !2890)
!2903 = !DILocalVariable(name: "c", scope: !2890, file: !100, line: 109, type: !104)
!2904 = !DILocation(line: 110, column: 16, scope: !2890)
!2905 = !DILocalVariable(name: "d", scope: !2890, file: !100, line: 110, type: !104)
!2906 = !DILocation(line: 111, column: 20, scope: !2890)
!2907 = !DILocation(line: 111, column: 26, scope: !2890)
!2908 = !DILocation(line: 111, column: 23, scope: !2890)
!2909 = !DILocalVariable(name: "divisor", scope: !2890, file: !100, line: 111, type: !104)
!2910 = !DILocation(line: 112, column: 18, scope: !2890)
!2911 = !DILocation(line: 112, column: 24, scope: !2890)
!2912 = !DILocation(line: 112, column: 21, scope: !2890)
!2913 = !DILocation(line: 112, column: 28, scope: !2890)
!2914 = !DILocalVariable(name: "real", scope: !2890, file: !100, line: 112, type: !104)
!2915 = !DILocation(line: 113, column: 18, scope: !2890)
!2916 = !DILocation(line: 113, column: 24, scope: !2890)
!2917 = !DILocation(line: 113, column: 21, scope: !2890)
!2918 = !DILocation(line: 113, column: 28, scope: !2890)
!2919 = !DILocalVariable(name: "imag", scope: !2890, file: !100, line: 113, type: !104)
!2920 = !DILocalVariable(name: "result", scope: !2890, file: !100, line: 114, type: !99)
!2921 = !DILocation(line: 114, column: 11, scope: !2890)
!2922 = !DILocation(line: 114, column: 30, scope: !2890)
!2923 = !DILocation(line: 115, column: 2, scope: !2890)
!2924 = distinct !DISubprogram(name: "cffts1_gpu", linkageName: "_ZL10cffts1_gpuiP8dcomplexS0_S0_S0_S0_", scope: !3, file: !3, line: 725, type: !2925, scopeLine: 730, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1051)
!2925 = !DISubroutineType(types: !2926)
!2926 = !{null, !1174, !98, !98, !98, !98, !98}
!2927 = !DILocalVariable(name: "is", arg: 1, scope: !2924, file: !3, line: 725, type: !1174)
!2928 = !DILocation(line: 0, scope: !2924)
!2929 = !DILocalVariable(name: "u", arg: 2, scope: !2924, file: !3, line: 726, type: !98)
!2930 = !DILocalVariable(name: "x_in", arg: 3, scope: !2924, file: !3, line: 727, type: !98)
!2931 = !DILocalVariable(name: "x_out", arg: 4, scope: !2924, file: !3, line: 728, type: !98)
!2932 = !DILocalVariable(name: "y0", arg: 5, scope: !2924, file: !3, line: 729, type: !98)
!2933 = !DILocalVariable(name: "y1", arg: 6, scope: !2924, file: !3, line: 730, type: !98)
!2934 = !DILocation(line: 734, column: 24, scope: !2924)
!2935 = !DILocation(line: 735, column: 3, scope: !2924)
!2936 = !DILocation(line: 734, column: 21, scope: !2924)
!2937 = !DILocation(line: 745, column: 24, scope: !2924)
!2938 = !DILocation(line: 746, column: 3, scope: !2924)
!2939 = !DILocation(line: 745, column: 21, scope: !2924)
!2940 = !DILocation(line: 758, column: 24, scope: !2924)
!2941 = !DILocation(line: 759, column: 3, scope: !2924)
!2942 = !DILocation(line: 758, column: 21, scope: !2924)
!2943 = !DILocation(line: 765, column: 1, scope: !2924)
!2944 = distinct !DISubprogram(name: "cffts2_gpu", linkageName: "_ZL10cffts2_gpuiP8dcomplexS0_S0_S0_S0_", scope: !3, file: !3, line: 920, type: !2945, scopeLine: 925, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1051)
!2945 = !DISubroutineType(types: !2946)
!2946 = !{null, !97, !98, !98, !98, !98, !98}
!2947 = !DILocalVariable(name: "is", arg: 1, scope: !2944, file: !3, line: 920, type: !97)
!2948 = !DILocation(line: 0, scope: !2944)
!2949 = !DILocalVariable(name: "u", arg: 2, scope: !2944, file: !3, line: 921, type: !98)
!2950 = !DILocalVariable(name: "x_in", arg: 3, scope: !2944, file: !3, line: 922, type: !98)
!2951 = !DILocalVariable(name: "x_out", arg: 4, scope: !2944, file: !3, line: 923, type: !98)
!2952 = !DILocalVariable(name: "y0", arg: 5, scope: !2944, file: !3, line: 924, type: !98)
!2953 = !DILocalVariable(name: "y1", arg: 6, scope: !2944, file: !3, line: 925, type: !98)
!2954 = !DILocation(line: 929, column: 24, scope: !2944)
!2955 = !DILocation(line: 930, column: 3, scope: !2944)
!2956 = !DILocation(line: 929, column: 21, scope: !2944)
!2957 = !DILocation(line: 940, column: 24, scope: !2944)
!2958 = !DILocation(line: 941, column: 3, scope: !2944)
!2959 = !DILocation(line: 940, column: 21, scope: !2944)
!2960 = !DILocation(line: 953, column: 24, scope: !2944)
!2961 = !DILocation(line: 954, column: 3, scope: !2944)
!2962 = !DILocation(line: 953, column: 21, scope: !2944)
!2963 = !DILocation(line: 960, column: 1, scope: !2944)
!2964 = distinct !DISubprogram(name: "cffts3_gpu", linkageName: "_ZL10cffts3_gpuiP8dcomplexS0_S0_S0_S0_", scope: !3, file: !3, line: 1111, type: !2945, scopeLine: 1116, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1051)
!2965 = !DILocalVariable(name: "is", arg: 1, scope: !2964, file: !3, line: 1111, type: !97)
!2966 = !DILocation(line: 0, scope: !2964)
!2967 = !DILocalVariable(name: "u", arg: 2, scope: !2964, file: !3, line: 1112, type: !98)
!2968 = !DILocalVariable(name: "x_in", arg: 3, scope: !2964, file: !3, line: 1113, type: !98)
!2969 = !DILocalVariable(name: "x_out", arg: 4, scope: !2964, file: !3, line: 1114, type: !98)
!2970 = !DILocalVariable(name: "y0", arg: 5, scope: !2964, file: !3, line: 1115, type: !98)
!2971 = !DILocalVariable(name: "y1", arg: 6, scope: !2964, file: !3, line: 1116, type: !98)
!2972 = !DILocation(line: 1120, column: 24, scope: !2964)
!2973 = !DILocation(line: 1121, column: 3, scope: !2964)
!2974 = !DILocation(line: 1120, column: 21, scope: !2964)
!2975 = !DILocation(line: 1131, column: 24, scope: !2964)
!2976 = !DILocation(line: 1132, column: 3, scope: !2964)
!2977 = !DILocation(line: 1131, column: 21, scope: !2964)
!2978 = !DILocation(line: 1144, column: 24, scope: !2964)
!2979 = !DILocation(line: 1145, column: 3, scope: !2964)
!2980 = !DILocation(line: 1144, column: 21, scope: !2964)
!2981 = !DILocation(line: 1151, column: 1, scope: !2964)
!2982 = distinct !DISubprogram(name: "ilog2", linkageName: "_ZL5ilog2i", scope: !3, file: !3, line: 1510, type: !304, scopeLine: 1510, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1051)
!2983 = !DILocalVariable(name: "n", arg: 1, scope: !2982, file: !3, line: 1510, type: !97)
!2984 = !DILocation(line: 0, scope: !2982)
!2985 = !DILocation(line: 1512, column: 6, scope: !2986)
!2986 = distinct !DILexicalBlock(scope: !2982, file: !3, line: 1512, column: 5)
!2987 = !DILocation(line: 1512, column: 5, scope: !2982)
!2988 = !DILocation(line: 1513, column: 3, scope: !2989)
!2989 = distinct !DILexicalBlock(scope: !2986, file: !3, line: 1512, column: 10)
!2990 = !DILocalVariable(name: "lg", scope: !2982, file: !3, line: 1511, type: !97)
!2991 = !DILocalVariable(name: "nn", scope: !2982, file: !3, line: 1511, type: !97)
!2992 = !DILocation(line: 1517, column: 2, scope: !2982)
!2993 = !DILocation(line: 1517, column: 10, scope: !2982)
!2994 = !DILocation(line: 1518, column: 11, scope: !2995)
!2995 = distinct !DILexicalBlock(scope: !2982, file: !3, line: 1517, column: 13)
!2996 = !DILocation(line: 1519, column: 5, scope: !2995)
!2997 = distinct !{!2997, !2992, !2998}
!2998 = !DILocation(line: 1520, column: 2, scope: !2982)
!2999 = !DILocation(line: 1521, column: 2, scope: !2982)
!3000 = !DILocation(line: 1522, column: 1, scope: !2982)
!3001 = distinct !DISubprogram(name: "ipow46", linkageName: "_ZL6ipow46diPd", scope: !3, file: !3, line: 1568, type: !1455, scopeLine: 1570, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1051)
!3002 = !DILocalVariable(name: "a", arg: 1, scope: !3001, file: !3, line: 1568, type: !104)
!3003 = !DILocation(line: 0, scope: !3001)
!3004 = !DILocalVariable(name: "exponent", arg: 2, scope: !3001, file: !3, line: 1569, type: !97)
!3005 = !DILocalVariable(name: "result", arg: 3, scope: !3001, file: !3, line: 1570, type: !106)
!3006 = !DILocalVariable(name: "q", scope: !3001, file: !3, line: 1571, type: !104)
!3007 = !DILocation(line: 1571, column: 9, scope: !3001)
!3008 = !DILocalVariable(name: "r", scope: !3001, file: !3, line: 1571, type: !104)
!3009 = !DILocation(line: 1571, column: 12, scope: !3001)
!3010 = !DILocation(line: 1580, column: 10, scope: !3001)
!3011 = !DILocation(line: 1581, column: 13, scope: !3012)
!3012 = distinct !DILexicalBlock(scope: !3001, file: !3, line: 1581, column: 5)
!3013 = !DILocation(line: 1581, column: 5, scope: !3001)
!3014 = !DILocation(line: 1581, column: 18, scope: !3015)
!3015 = distinct !DILexicalBlock(scope: !3012, file: !3, line: 1581, column: 17)
!3016 = !DILocation(line: 1582, column: 4, scope: !3001)
!3017 = !DILocation(line: 1583, column: 4, scope: !3001)
!3018 = !DILocalVariable(name: "n", scope: !3001, file: !3, line: 1572, type: !97)
!3019 = !DILocation(line: 1585, column: 2, scope: !3001)
!3020 = !DILocation(line: 1585, column: 9, scope: !3001)
!3021 = !DILocation(line: 1586, column: 9, scope: !3022)
!3022 = distinct !DILexicalBlock(scope: !3001, file: !3, line: 1585, column: 12)
!3023 = !DILocalVariable(name: "n2", scope: !3001, file: !3, line: 1572, type: !97)
!3024 = !DILocation(line: 1587, column: 8, scope: !3025)
!3025 = distinct !DILexicalBlock(scope: !3022, file: !3, line: 1587, column: 6)
!3026 = !DILocation(line: 1587, column: 10, scope: !3025)
!3027 = !DILocation(line: 1587, column: 6, scope: !3022)
!3028 = !DILocation(line: 1588, column: 15, scope: !3029)
!3029 = distinct !DILexicalBlock(scope: !3025, file: !3, line: 1587, column: 14)
!3030 = !DILocation(line: 1588, column: 4, scope: !3029)
!3031 = !DILocation(line: 1590, column: 3, scope: !3029)
!3032 = !DILocation(line: 1591, column: 15, scope: !3033)
!3033 = distinct !DILexicalBlock(scope: !3025, file: !3, line: 1590, column: 8)
!3034 = !DILocation(line: 1591, column: 4, scope: !3033)
!3035 = !DILocation(line: 1592, column: 9, scope: !3033)
!3036 = !DILocation(line: 0, scope: !3025)
!3037 = distinct !{!3037, !3019, !3038}
!3038 = !DILocation(line: 1594, column: 2, scope: !3001)
!3039 = !DILocation(line: 1595, column: 13, scope: !3001)
!3040 = !DILocation(line: 1595, column: 2, scope: !3001)
!3041 = !DILocation(line: 1596, column: 12, scope: !3001)
!3042 = !DILocation(line: 1596, column: 10, scope: !3001)
!3043 = !DILocation(line: 1597, column: 1, scope: !3001)
!3044 = !DILocalVariable(name: "u0", arg: 1, scope: !3045, file: !3, line: 1554, type: !98)
!3045 = distinct !DISubprogram(name: "init_ui_gpu_kernel", linkageName: "_Z18init_ui_gpu_kernelP8dcomplexS0_Pd", scope: !3, file: !3, line: 1554, type: !2236, scopeLine: 1556, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !1050, retainedNodes: !1051)
!3046 = !DILocation(line: 0, scope: !3045)
!3047 = !DILocalVariable(name: "u1", arg: 2, scope: !3045, file: !3, line: 1555, type: !98)
!3048 = !DILocalVariable(name: "twiddle", arg: 3, scope: !3045, file: !3, line: 1556, type: !106)
!3049 = !DILocation(line: 1557, column: 29, scope: !3045)
!3050 = !DILocation(line: 1557, column: 42, scope: !3045)
!3051 = !DILocalVariable(name: "thread_id", scope: !3045, file: !3, line: 1557, type: !97)
!3052 = !DILocation(line: 1559, column: 14, scope: !3053)
!3053 = distinct !DILexicalBlock(scope: !3045, file: !3, line: 1559, column: 5)
!3054 = !DILocation(line: 1559, column: 5, scope: !3045)
!3055 = !DILocation(line: 1560, column: 3, scope: !3056)
!3056 = distinct !DILexicalBlock(scope: !3053, file: !3, line: 1559, column: 23)
!3057 = !DILocation(line: 1563, column: 18, scope: !3045)
!3058 = !DILocation(line: 1563, column: 2, scope: !3045)
!3059 = !DILocation(line: 1563, column: 16, scope: !3045)
!3060 = !DILocation(line: 1564, column: 18, scope: !3045)
!3061 = !DILocation(line: 1564, column: 2, scope: !3045)
!3062 = !DILocation(line: 1564, column: 16, scope: !3045)
!3063 = !DILocation(line: 1565, column: 2, scope: !3045)
!3064 = !DILocation(line: 1565, column: 21, scope: !3045)
!3065 = !DILocation(line: 1566, column: 1, scope: !3045)
!3066 = !DILocalVariable(name: "twiddle", arg: 1, scope: !3067, file: !3, line: 1365, type: !106)
!3067 = distinct !DISubprogram(name: "compute_indexmap_gpu_kernel", linkageName: "_Z27compute_indexmap_gpu_kernelPd", scope: !3, file: !3, line: 1365, type: !2247, scopeLine: 1365, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !1050, retainedNodes: !1051)
!3068 = !DILocation(line: 0, scope: !3067)
!3069 = !DILocation(line: 1366, column: 29, scope: !3067)
!3070 = !DILocation(line: 1366, column: 42, scope: !3067)
!3071 = !DILocalVariable(name: "thread_id", scope: !3067, file: !3, line: 1366, type: !97)
!3072 = !DILocation(line: 1368, column: 14, scope: !3073)
!3073 = distinct !DILexicalBlock(scope: !3067, file: !3, line: 1368, column: 5)
!3074 = !DILocation(line: 1368, column: 5, scope: !3067)
!3075 = !DILocation(line: 1369, column: 3, scope: !3076)
!3076 = distinct !DILexicalBlock(scope: !3073, file: !3, line: 1368, column: 23)
!3077 = !DILocation(line: 1372, column: 20, scope: !3067)
!3078 = !DILocalVariable(name: "i", scope: !3067, file: !3, line: 1372, type: !97)
!3079 = !DILocation(line: 1373, column: 21, scope: !3067)
!3080 = !DILocation(line: 1373, column: 27, scope: !3067)
!3081 = !DILocalVariable(name: "j", scope: !3067, file: !3, line: 1373, type: !97)
!3082 = !DILocation(line: 1374, column: 20, scope: !3067)
!3083 = !DILocalVariable(name: "k", scope: !3067, file: !3, line: 1374, type: !97)
!3084 = !DILocation(line: 1378, column: 10, scope: !3067)
!3085 = !DILocation(line: 1378, column: 17, scope: !3067)
!3086 = !DILocation(line: 1378, column: 23, scope: !3067)
!3087 = !DILocalVariable(name: "kk", scope: !3067, file: !3, line: 1376, type: !97)
!3088 = !DILocation(line: 1379, column: 10, scope: !3067)
!3089 = !DILocalVariable(name: "kk2", scope: !3067, file: !3, line: 1376, type: !97)
!3090 = !DILocation(line: 1380, column: 10, scope: !3067)
!3091 = !DILocation(line: 1380, column: 17, scope: !3067)
!3092 = !DILocation(line: 1380, column: 23, scope: !3067)
!3093 = !DILocalVariable(name: "jj", scope: !3067, file: !3, line: 1376, type: !97)
!3094 = !DILocation(line: 1381, column: 10, scope: !3067)
!3095 = !DILocation(line: 1381, column: 13, scope: !3067)
!3096 = !DILocalVariable(name: "kj2", scope: !3067, file: !3, line: 1376, type: !97)
!3097 = !DILocation(line: 1382, column: 10, scope: !3067)
!3098 = !DILocation(line: 1382, column: 17, scope: !3067)
!3099 = !DILocation(line: 1382, column: 23, scope: !3067)
!3100 = !DILocalVariable(name: "ii", scope: !3067, file: !3, line: 1376, type: !97)
!3101 = !DILocation(line: 1384, column: 41, scope: !3067)
!3102 = !DILocation(line: 1384, column: 44, scope: !3067)
!3103 = !DILocation(line: 1384, column: 38, scope: !3067)
!3104 = !DILocation(line: 1384, column: 29, scope: !3067)
!3105 = !DILocalVariable(name: "a", arg: 1, scope: !3106, file: !3107, line: 245, type: !104)
!3106 = distinct !DISubprogram(name: "exp", linkageName: "_ZL3expd", scope: !3107, file: !3107, line: 245, type: !496, scopeLine: 246, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !1050, retainedNodes: !1051)
!3107 = !DIFile(filename: "/usr/local/cuda/include/math_functions_dbl_ptx3.hpp", directory: "")
!3108 = !DILocation(line: 0, scope: !3106, inlinedAt: !3109)
!3109 = distinct !DILocation(line: 1384, column: 23, scope: !3067)
!3110 = !DILocation(line: 1384, column: 2, scope: !3067)
!3111 = !DILocation(line: 1384, column: 21, scope: !3067)
!3112 = !DILocation(line: 1385, column: 1, scope: !3067)
!3113 = !DILocalVariable(name: "u0", arg: 1, scope: !3114, file: !3, line: 1416, type: !98)
!3114 = distinct !DISubprogram(name: "compute_initial_conditions_gpu_kernel", linkageName: "_Z37compute_initial_conditions_gpu_kernelP8dcomplexPd", scope: !3, file: !3, line: 1416, type: !3115, scopeLine: 1417, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !1050, retainedNodes: !1051)
!3115 = !DISubroutineType(types: !3116)
!3116 = !{null, !98, !106}
!3117 = !DILocation(line: 0, scope: !3114)
!3118 = !DILocalVariable(name: "starts", arg: 2, scope: !3114, file: !3, line: 1417, type: !106)
!3119 = !DILocation(line: 1418, column: 21, scope: !3114)
!3120 = !DILocation(line: 1418, column: 34, scope: !3114)
!3121 = !DILocalVariable(name: "z", scope: !3114, file: !3, line: 1418, type: !97)
!3122 = !DILocation(line: 1420, column: 6, scope: !3123)
!3123 = distinct !DILexicalBlock(scope: !3114, file: !3, line: 1420, column: 5)
!3124 = !DILocation(line: 1420, column: 5, scope: !3114)
!3125 = !DILocation(line: 1420, column: 12, scope: !3126)
!3126 = distinct !DILexicalBlock(scope: !3123, file: !3, line: 1420, column: 11)
!3127 = !DILocalVariable(name: "x0", scope: !3114, file: !3, line: 1422, type: !104)
!3128 = !DILocation(line: 1422, column: 9, scope: !3114)
!3129 = !DILocation(line: 1422, column: 14, scope: !3114)
!3130 = !DILocalVariable(name: "y", scope: !3131, file: !3, line: 1423, type: !97)
!3131 = distinct !DILexicalBlock(scope: !3114, file: !3, line: 1423, column: 2)
!3132 = !DILocation(line: 0, scope: !3131)
!3133 = !DILocation(line: 1423, column: 6, scope: !3131)
!3134 = !DILocation(line: 1423, column: 16, scope: !3135)
!3135 = distinct !DILexicalBlock(scope: !3131, file: !3, line: 1423, column: 2)
!3136 = !DILocation(line: 1423, column: 2, scope: !3131)
!3137 = !DILocation(line: 1424, column: 50, scope: !3138)
!3138 = distinct !DILexicalBlock(scope: !3135, file: !3, line: 1423, column: 25)
!3139 = !DILocation(line: 1424, column: 57, scope: !3138)
!3140 = !DILocation(line: 1424, column: 60, scope: !3138)
!3141 = !DILocation(line: 1424, column: 54, scope: !3138)
!3142 = !DILocation(line: 1424, column: 41, scope: !3138)
!3143 = !DILocation(line: 1424, column: 31, scope: !3138)
!3144 = !DILocation(line: 1424, column: 3, scope: !3138)
!3145 = !DILocation(line: 1425, column: 2, scope: !3138)
!3146 = !DILocation(line: 1423, column: 22, scope: !3135)
!3147 = !DILocation(line: 1423, column: 2, scope: !3135)
!3148 = distinct !{!3148, !3136, !3149}
!3149 = !DILocation(line: 1425, column: 2, scope: !3131)
!3150 = !DILocation(line: 1426, column: 1, scope: !3114)
!3151 = !DILocalVariable(name: "u0", arg: 1, scope: !3152, file: !3, line: 1444, type: !98)
!3152 = distinct !DISubprogram(name: "evolve_gpu_kernel", linkageName: "_Z17evolve_gpu_kernelP8dcomplexS0_Pd", scope: !3, file: !3, line: 1444, type: !2236, scopeLine: 1446, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !1050, retainedNodes: !1051)
!3153 = !DILocation(line: 0, scope: !3152)
!3154 = !DILocalVariable(name: "u1", arg: 2, scope: !3152, file: !3, line: 1445, type: !98)
!3155 = !DILocalVariable(name: "twiddle", arg: 3, scope: !3152, file: !3, line: 1446, type: !106)
!3156 = !DILocation(line: 1447, column: 29, scope: !3152)
!3157 = !DILocation(line: 1447, column: 42, scope: !3152)
!3158 = !DILocalVariable(name: "thread_id", scope: !3152, file: !3, line: 1447, type: !97)
!3159 = !DILocation(line: 1449, column: 14, scope: !3160)
!3160 = distinct !DILexicalBlock(scope: !3152, file: !3, line: 1449, column: 5)
!3161 = !DILocation(line: 1449, column: 5, scope: !3152)
!3162 = !DILocation(line: 1450, column: 3, scope: !3163)
!3163 = distinct !DILexicalBlock(scope: !3160, file: !3, line: 1449, column: 27)
!3164 = !DILocation(line: 1453, column: 18, scope: !3152)
!3165 = !DILocation(line: 1453, column: 2, scope: !3152)
!3166 = !DILocation(line: 1453, column: 16, scope: !3152)
!3167 = !DILocation(line: 1454, column: 18, scope: !3152)
!3168 = !DILocation(line: 1454, column: 2, scope: !3152)
!3169 = !DILocation(line: 1454, column: 16, scope: !3152)
!3170 = !DILocation(line: 1455, column: 1, scope: !3152)
!3171 = !DILocalVariable(name: "iteration", arg: 1, scope: !3172, file: !3, line: 1323, type: !97)
!3172 = distinct !DISubprogram(name: "checksum_gpu_kernel", linkageName: "_Z19checksum_gpu_kerneliP8dcomplexS0_", scope: !3, file: !3, line: 1323, type: !2359, scopeLine: 1325, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !1050, retainedNodes: !1051)
!3173 = !DILocation(line: 0, scope: !3172)
!3174 = !DILocalVariable(name: "u1", arg: 2, scope: !3172, file: !3, line: 1324, type: !98)
!3175 = !DILocalVariable(name: "sums", arg: 3, scope: !3172, file: !3, line: 1325, type: !98)
!3176 = !DILocalVariable(name: "share_sums", scope: !3172, file: !3, line: 1326, type: !98)
!3177 = !DILocation(line: 1327, column: 22, scope: !3172)
!3178 = !DILocation(line: 1327, column: 35, scope: !3172)
!3179 = !DILocation(line: 1327, column: 50, scope: !3172)
!3180 = !DILocalVariable(name: "j", scope: !3172, file: !3, line: 1327, type: !97)
!3181 = !DILocation(line: 1330, column: 6, scope: !3182)
!3182 = distinct !DILexicalBlock(scope: !3172, file: !3, line: 1330, column: 5)
!3183 = !DILocation(line: 1330, column: 5, scope: !3172)
!3184 = !DILocation(line: 1331, column: 9, scope: !3185)
!3185 = distinct !DILexicalBlock(scope: !3182, file: !3, line: 1330, column: 23)
!3186 = !DILocalVariable(name: "q", scope: !3172, file: !3, line: 1328, type: !97)
!3187 = !DILocation(line: 1332, column: 8, scope: !3185)
!3188 = !DILocation(line: 1332, column: 11, scope: !3185)
!3189 = !DILocalVariable(name: "r", scope: !3172, file: !3, line: 1328, type: !97)
!3190 = !DILocation(line: 1333, column: 8, scope: !3185)
!3191 = !DILocation(line: 1333, column: 11, scope: !3185)
!3192 = !DILocalVariable(name: "s", scope: !3172, file: !3, line: 1328, type: !97)
!3193 = !DILocation(line: 1334, column: 38, scope: !3185)
!3194 = !DILocation(line: 1334, column: 35, scope: !3185)
!3195 = !DILocation(line: 1334, column: 45, scope: !3185)
!3196 = !DILocation(line: 1334, column: 48, scope: !3185)
!3197 = !DILocation(line: 1334, column: 42, scope: !3185)
!3198 = !DILocation(line: 1334, column: 29, scope: !3185)
!3199 = !DILocation(line: 1334, column: 3, scope: !3185)
!3200 = !DILocation(line: 1334, column: 27, scope: !3185)
!3201 = !DILocation(line: 1335, column: 2, scope: !3185)
!3202 = !DILocation(line: 1336, column: 29, scope: !3203)
!3203 = distinct !DILexicalBlock(scope: !3182, file: !3, line: 1335, column: 7)
!3204 = !DILocation(line: 1336, column: 3, scope: !3203)
!3205 = !DILocation(line: 1336, column: 27, scope: !3203)
!3206 = !DILocation(line: 1340, column: 2, scope: !3172)
!3207 = !DILocalVariable(name: "x_in", arg: 1, scope: !3208, file: !3, line: 774, type: !98)
!3208 = distinct !DISubprogram(name: "cffts1_gpu_kernel_1", linkageName: "_Z19cffts1_gpu_kernel_1P8dcomplexS0_", scope: !3, file: !3, line: 774, type: !3209, scopeLine: 775, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !1050, retainedNodes: !1051)
!3209 = !DISubroutineType(types: !3210)
!3210 = !{null, !98, !98}
!3211 = !DILocation(line: 0, scope: !3208)
!3212 = !DILocalVariable(name: "y0", arg: 2, scope: !3208, file: !3, line: 775, type: !98)
!3213 = !DILocation(line: 776, column: 25, scope: !3208)
!3214 = !DILocation(line: 776, column: 38, scope: !3208)
!3215 = !DILocalVariable(name: "x_y_z", scope: !3208, file: !3, line: 776, type: !97)
!3216 = !DILocation(line: 777, column: 11, scope: !3217)
!3217 = distinct !DILexicalBlock(scope: !3208, file: !3, line: 777, column: 5)
!3218 = !DILocation(line: 777, column: 5, scope: !3208)
!3219 = !DILocation(line: 778, column: 3, scope: !3220)
!3220 = distinct !DILexicalBlock(scope: !3217, file: !3, line: 777, column: 25)
!3221 = !DILocation(line: 780, column: 16, scope: !3208)
!3222 = !DILocalVariable(name: "x", scope: !3208, file: !3, line: 780, type: !97)
!3223 = !DILocation(line: 781, column: 17, scope: !3208)
!3224 = !DILocation(line: 781, column: 23, scope: !3208)
!3225 = !DILocalVariable(name: "y", scope: !3208, file: !3, line: 781, type: !97)
!3226 = !DILocation(line: 782, column: 16, scope: !3208)
!3227 = !DILocalVariable(name: "z", scope: !3208, file: !3, line: 782, type: !97)
!3228 = !DILocation(line: 783, column: 32, scope: !3208)
!3229 = !DILocation(line: 783, column: 44, scope: !3208)
!3230 = !DILocation(line: 783, column: 9, scope: !3208)
!3231 = !DILocation(line: 783, column: 6, scope: !3208)
!3232 = !DILocation(line: 783, column: 16, scope: !3208)
!3233 = !DILocation(line: 783, column: 19, scope: !3208)
!3234 = !DILocation(line: 783, column: 13, scope: !3208)
!3235 = !DILocation(line: 783, column: 2, scope: !3208)
!3236 = !DILocation(line: 783, column: 25, scope: !3208)
!3237 = !DILocation(line: 783, column: 30, scope: !3208)
!3238 = !DILocation(line: 784, column: 32, scope: !3208)
!3239 = !DILocation(line: 784, column: 44, scope: !3208)
!3240 = !DILocation(line: 784, column: 9, scope: !3208)
!3241 = !DILocation(line: 784, column: 6, scope: !3208)
!3242 = !DILocation(line: 784, column: 16, scope: !3208)
!3243 = !DILocation(line: 784, column: 19, scope: !3208)
!3244 = !DILocation(line: 784, column: 13, scope: !3208)
!3245 = !DILocation(line: 784, column: 2, scope: !3208)
!3246 = !DILocation(line: 784, column: 25, scope: !3208)
!3247 = !DILocation(line: 784, column: 30, scope: !3208)
!3248 = !DILocation(line: 785, column: 1, scope: !3208)
!3249 = !DILocalVariable(name: "is", arg: 1, scope: !3250, file: !3, line: 792, type: !1174)
!3250 = distinct !DISubprogram(name: "cffts1_gpu_kernel_2", linkageName: "_Z19cffts1_gpu_kernel_2iP8dcomplexS0_S0_", scope: !3, file: !3, line: 792, type: !3251, scopeLine: 795, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !1050, retainedNodes: !1051)
!3251 = !DISubroutineType(types: !3252)
!3252 = !{null, !1174, !98, !98, !98}
!3253 = !DILocation(line: 0, scope: !3250)
!3254 = !DILocalVariable(name: "gty1", arg: 2, scope: !3250, file: !3, line: 793, type: !98)
!3255 = !DILocalVariable(name: "gty2", arg: 3, scope: !3250, file: !3, line: 794, type: !98)
!3256 = !DILocalVariable(name: "u_device", arg: 4, scope: !3250, file: !3, line: 795, type: !98)
!3257 = !DILocation(line: 796, column: 23, scope: !3250)
!3258 = !DILocation(line: 796, column: 36, scope: !3250)
!3259 = !DILocalVariable(name: "y_z", scope: !3250, file: !3, line: 796, type: !97)
!3260 = !DILocation(line: 798, column: 9, scope: !3261)
!3261 = distinct !DILexicalBlock(scope: !3250, file: !3, line: 798, column: 5)
!3262 = !DILocation(line: 798, column: 5, scope: !3250)
!3263 = !DILocation(line: 799, column: 3, scope: !3264)
!3264 = distinct !DILexicalBlock(scope: !3261, file: !3, line: 798, column: 20)
!3265 = !DILocation(line: 806, column: 10, scope: !3250)
!3266 = !DILocalVariable(name: "j", scope: !3250, file: !3, line: 802, type: !97)
!3267 = !DILocation(line: 807, column: 11, scope: !3250)
!3268 = !DILocation(line: 807, column: 17, scope: !3250)
!3269 = !DILocalVariable(name: "k", scope: !3250, file: !3, line: 802, type: !97)
!3270 = !DILocation(line: 809, column: 20, scope: !3250)
!3271 = !DILocalVariable(name: "logd1", scope: !3250, file: !3, line: 809, type: !1174)
!3272 = !DILocalVariable(name: "l", scope: !3250, file: !3, line: 803, type: !97)
!3273 = !DILocation(line: 818, column: 6, scope: !3274)
!3274 = distinct !DILexicalBlock(scope: !3250, file: !3, line: 818, column: 2)
!3275 = !DILocation(line: 0, scope: !3274)
!3276 = !DILocation(line: 818, column: 12, scope: !3277)
!3277 = distinct !DILexicalBlock(scope: !3274, file: !3, line: 818, column: 2)
!3278 = !DILocation(line: 818, column: 2, scope: !3274)
!3279 = !DILocalVariable(name: "n1", scope: !3250, file: !3, line: 804, type: !97)
!3280 = !DILocation(line: 820, column: 16, scope: !3281)
!3281 = distinct !DILexicalBlock(scope: !3277, file: !3, line: 818, column: 26)
!3282 = !DILocation(line: 820, column: 10, scope: !3281)
!3283 = !DILocalVariable(name: "lk", scope: !3250, file: !3, line: 804, type: !97)
!3284 = !DILocation(line: 821, column: 20, scope: !3281)
!3285 = !DILocation(line: 821, column: 10, scope: !3281)
!3286 = !DILocalVariable(name: "li", scope: !3250, file: !3, line: 804, type: !97)
!3287 = !DILocation(line: 822, column: 10, scope: !3281)
!3288 = !DILocalVariable(name: "lj", scope: !3250, file: !3, line: 804, type: !97)
!3289 = !DILocalVariable(name: "ku", scope: !3250, file: !3, line: 804, type: !97)
!3290 = !DILocalVariable(name: "i1", scope: !3250, file: !3, line: 803, type: !97)
!3291 = !DILocation(line: 824, column: 7, scope: !3292)
!3292 = distinct !DILexicalBlock(scope: !3281, file: !3, line: 824, column: 3)
!3293 = !DILocation(line: 0, scope: !3292)
!3294 = !DILocation(line: 824, column: 19, scope: !3295)
!3295 = distinct !DILexicalBlock(scope: !3292, file: !3, line: 824, column: 3)
!3296 = !DILocation(line: 824, column: 15, scope: !3295)
!3297 = !DILocation(line: 824, column: 3, scope: !3292)
!3298 = !DILocalVariable(name: "k1", scope: !3250, file: !3, line: 803, type: !97)
!3299 = !DILocation(line: 825, column: 8, scope: !3300)
!3300 = distinct !DILexicalBlock(scope: !3301, file: !3, line: 825, column: 4)
!3301 = distinct !DILexicalBlock(scope: !3295, file: !3, line: 824, column: 28)
!3302 = !DILocation(line: 0, scope: !3300)
!3303 = !DILocation(line: 825, column: 20, scope: !3304)
!3304 = distinct !DILexicalBlock(scope: !3300, file: !3, line: 825, column: 4)
!3305 = !DILocation(line: 825, column: 16, scope: !3304)
!3306 = !DILocation(line: 825, column: 4, scope: !3300)
!3307 = !DILocation(line: 826, column: 14, scope: !3308)
!3308 = distinct !DILexicalBlock(scope: !3304, file: !3, line: 825, column: 29)
!3309 = !DILocation(line: 827, column: 15, scope: !3308)
!3310 = !DILocation(line: 828, column: 14, scope: !3308)
!3311 = !DILocation(line: 829, column: 15, scope: !3308)
!3312 = !DILocation(line: 831, column: 27, scope: !3308)
!3313 = !DILocation(line: 831, column: 16, scope: !3308)
!3314 = !DILocation(line: 831, column: 32, scope: !3308)
!3315 = !DILocalVariable(name: "uu1_real", scope: !3250, file: !3, line: 811, type: !104)
!3316 = !DILocation(line: 832, column: 16, scope: !3308)
!3317 = !DILocation(line: 832, column: 30, scope: !3308)
!3318 = !DILocation(line: 832, column: 19, scope: !3308)
!3319 = !DILocation(line: 832, column: 35, scope: !3308)
!3320 = !DILocation(line: 832, column: 18, scope: !3308)
!3321 = !DILocalVariable(name: "uu1_imag", scope: !3250, file: !3, line: 812, type: !104)
!3322 = !DILocation(line: 835, column: 29, scope: !3308)
!3323 = !DILocation(line: 835, column: 33, scope: !3308)
!3324 = !DILocation(line: 835, column: 23, scope: !3308)
!3325 = !DILocation(line: 835, column: 40, scope: !3308)
!3326 = !DILocation(line: 835, column: 43, scope: !3308)
!3327 = !DILocation(line: 835, column: 37, scope: !3308)
!3328 = !DILocation(line: 835, column: 16, scope: !3308)
!3329 = !DILocation(line: 835, column: 48, scope: !3308)
!3330 = !DILocalVariable(name: "x11_real", scope: !3250, file: !3, line: 811, type: !104)
!3331 = !DILocation(line: 836, column: 29, scope: !3308)
!3332 = !DILocation(line: 836, column: 33, scope: !3308)
!3333 = !DILocation(line: 836, column: 23, scope: !3308)
!3334 = !DILocation(line: 836, column: 40, scope: !3308)
!3335 = !DILocation(line: 836, column: 43, scope: !3308)
!3336 = !DILocation(line: 836, column: 37, scope: !3308)
!3337 = !DILocation(line: 836, column: 16, scope: !3308)
!3338 = !DILocation(line: 836, column: 48, scope: !3308)
!3339 = !DILocalVariable(name: "x11_imag", scope: !3250, file: !3, line: 812, type: !104)
!3340 = !DILocation(line: 839, column: 29, scope: !3308)
!3341 = !DILocation(line: 839, column: 33, scope: !3308)
!3342 = !DILocation(line: 839, column: 23, scope: !3308)
!3343 = !DILocation(line: 839, column: 40, scope: !3308)
!3344 = !DILocation(line: 839, column: 43, scope: !3308)
!3345 = !DILocation(line: 839, column: 37, scope: !3308)
!3346 = !DILocation(line: 839, column: 16, scope: !3308)
!3347 = !DILocation(line: 839, column: 48, scope: !3308)
!3348 = !DILocalVariable(name: "x21_real", scope: !3250, file: !3, line: 811, type: !104)
!3349 = !DILocation(line: 840, column: 29, scope: !3308)
!3350 = !DILocation(line: 840, column: 33, scope: !3308)
!3351 = !DILocation(line: 840, column: 23, scope: !3308)
!3352 = !DILocation(line: 840, column: 40, scope: !3308)
!3353 = !DILocation(line: 840, column: 43, scope: !3308)
!3354 = !DILocation(line: 840, column: 37, scope: !3308)
!3355 = !DILocation(line: 840, column: 16, scope: !3308)
!3356 = !DILocation(line: 840, column: 48, scope: !3308)
!3357 = !DILocalVariable(name: "x21_imag", scope: !3250, file: !3, line: 812, type: !104)
!3358 = !DILocation(line: 843, column: 53, scope: !3308)
!3359 = !DILocation(line: 843, column: 18, scope: !3308)
!3360 = !DILocation(line: 843, column: 22, scope: !3308)
!3361 = !DILocation(line: 843, column: 12, scope: !3308)
!3362 = !DILocation(line: 843, column: 29, scope: !3308)
!3363 = !DILocation(line: 843, column: 32, scope: !3308)
!3364 = !DILocation(line: 843, column: 26, scope: !3308)
!3365 = !DILocation(line: 843, column: 5, scope: !3308)
!3366 = !DILocation(line: 843, column: 37, scope: !3308)
!3367 = !DILocation(line: 843, column: 42, scope: !3308)
!3368 = !DILocation(line: 844, column: 53, scope: !3308)
!3369 = !DILocation(line: 844, column: 18, scope: !3308)
!3370 = !DILocation(line: 844, column: 22, scope: !3308)
!3371 = !DILocation(line: 844, column: 12, scope: !3308)
!3372 = !DILocation(line: 844, column: 29, scope: !3308)
!3373 = !DILocation(line: 844, column: 32, scope: !3308)
!3374 = !DILocation(line: 844, column: 26, scope: !3308)
!3375 = !DILocation(line: 844, column: 5, scope: !3308)
!3376 = !DILocation(line: 844, column: 37, scope: !3308)
!3377 = !DILocation(line: 844, column: 42, scope: !3308)
!3378 = !DILocation(line: 846, column: 26, scope: !3308)
!3379 = !DILocalVariable(name: "temp_real", scope: !3250, file: !3, line: 815, type: !104)
!3380 = !DILocation(line: 847, column: 26, scope: !3308)
!3381 = !DILocalVariable(name: "temp_imag", scope: !3250, file: !3, line: 816, type: !104)
!3382 = !DILocation(line: 850, column: 54, scope: !3308)
!3383 = !DILocation(line: 850, column: 79, scope: !3308)
!3384 = !DILocation(line: 850, column: 67, scope: !3308)
!3385 = !DILocation(line: 850, column: 18, scope: !3308)
!3386 = !DILocation(line: 850, column: 22, scope: !3308)
!3387 = !DILocation(line: 850, column: 12, scope: !3308)
!3388 = !DILocation(line: 850, column: 29, scope: !3308)
!3389 = !DILocation(line: 850, column: 32, scope: !3308)
!3390 = !DILocation(line: 850, column: 26, scope: !3308)
!3391 = !DILocation(line: 850, column: 5, scope: !3308)
!3392 = !DILocation(line: 850, column: 37, scope: !3308)
!3393 = !DILocation(line: 850, column: 42, scope: !3308)
!3394 = !DILocation(line: 851, column: 54, scope: !3308)
!3395 = !DILocation(line: 851, column: 79, scope: !3308)
!3396 = !DILocation(line: 851, column: 67, scope: !3308)
!3397 = !DILocation(line: 851, column: 18, scope: !3308)
!3398 = !DILocation(line: 851, column: 22, scope: !3308)
!3399 = !DILocation(line: 851, column: 12, scope: !3308)
!3400 = !DILocation(line: 851, column: 29, scope: !3308)
!3401 = !DILocation(line: 851, column: 32, scope: !3308)
!3402 = !DILocation(line: 851, column: 26, scope: !3308)
!3403 = !DILocation(line: 851, column: 5, scope: !3308)
!3404 = !DILocation(line: 851, column: 37, scope: !3308)
!3405 = !DILocation(line: 851, column: 42, scope: !3308)
!3406 = !DILocation(line: 852, column: 4, scope: !3308)
!3407 = !DILocation(line: 825, column: 26, scope: !3304)
!3408 = !DILocation(line: 825, column: 4, scope: !3304)
!3409 = distinct !{!3409, !3306, !3410}
!3410 = !DILocation(line: 852, column: 4, scope: !3300)
!3411 = !DILocation(line: 853, column: 3, scope: !3301)
!3412 = !DILocation(line: 824, column: 25, scope: !3295)
!3413 = !DILocation(line: 824, column: 3, scope: !3295)
!3414 = distinct !{!3414, !3297, !3415}
!3415 = !DILocation(line: 853, column: 3, scope: !3292)
!3416 = !DILocation(line: 854, column: 7, scope: !3417)
!3417 = distinct !DILexicalBlock(scope: !3281, file: !3, line: 854, column: 6)
!3418 = !DILocation(line: 854, column: 6, scope: !3281)
!3419 = !DILocalVariable(name: "j1", scope: !3250, file: !3, line: 803, type: !97)
!3420 = !DILocation(line: 855, column: 8, scope: !3421)
!3421 = distinct !DILexicalBlock(scope: !3422, file: !3, line: 855, column: 4)
!3422 = distinct !DILexicalBlock(scope: !3417, file: !3, line: 854, column: 15)
!3423 = !DILocation(line: 0, scope: !3421)
!3424 = !DILocation(line: 855, column: 16, scope: !3425)
!3425 = distinct !DILexicalBlock(scope: !3421, file: !3, line: 855, column: 4)
!3426 = !DILocation(line: 855, column: 4, scope: !3421)
!3427 = !DILocation(line: 857, column: 49, scope: !3428)
!3428 = distinct !DILexicalBlock(scope: !3425, file: !3, line: 855, column: 26)
!3429 = !DILocation(line: 857, column: 45, scope: !3428)
!3430 = !DILocation(line: 857, column: 56, scope: !3428)
!3431 = !DILocation(line: 857, column: 59, scope: !3428)
!3432 = !DILocation(line: 857, column: 53, scope: !3428)
!3433 = !DILocation(line: 857, column: 38, scope: !3428)
!3434 = !DILocation(line: 857, column: 64, scope: !3428)
!3435 = !DILocation(line: 857, column: 16, scope: !3428)
!3436 = !DILocation(line: 857, column: 12, scope: !3428)
!3437 = !DILocation(line: 857, column: 23, scope: !3428)
!3438 = !DILocation(line: 857, column: 26, scope: !3428)
!3439 = !DILocation(line: 857, column: 20, scope: !3428)
!3440 = !DILocation(line: 857, column: 5, scope: !3428)
!3441 = !DILocation(line: 857, column: 31, scope: !3428)
!3442 = !DILocation(line: 857, column: 36, scope: !3428)
!3443 = !DILocation(line: 858, column: 49, scope: !3428)
!3444 = !DILocation(line: 858, column: 45, scope: !3428)
!3445 = !DILocation(line: 858, column: 56, scope: !3428)
!3446 = !DILocation(line: 858, column: 59, scope: !3428)
!3447 = !DILocation(line: 858, column: 53, scope: !3428)
!3448 = !DILocation(line: 858, column: 38, scope: !3428)
!3449 = !DILocation(line: 858, column: 64, scope: !3428)
!3450 = !DILocation(line: 858, column: 16, scope: !3428)
!3451 = !DILocation(line: 858, column: 12, scope: !3428)
!3452 = !DILocation(line: 858, column: 23, scope: !3428)
!3453 = !DILocation(line: 858, column: 26, scope: !3428)
!3454 = !DILocation(line: 858, column: 20, scope: !3428)
!3455 = !DILocation(line: 858, column: 5, scope: !3428)
!3456 = !DILocation(line: 858, column: 31, scope: !3428)
!3457 = !DILocation(line: 858, column: 36, scope: !3428)
!3458 = !DILocation(line: 859, column: 4, scope: !3428)
!3459 = !DILocation(line: 855, column: 23, scope: !3425)
!3460 = !DILocation(line: 855, column: 4, scope: !3425)
!3461 = distinct !{!3461, !3426, !3462}
!3462 = !DILocation(line: 859, column: 4, scope: !3421)
!3463 = !DILocation(line: 860, column: 3, scope: !3422)
!3464 = !DILocation(line: 862, column: 16, scope: !3465)
!3465 = distinct !DILexicalBlock(scope: !3417, file: !3, line: 860, column: 8)
!3466 = !DILocation(line: 862, column: 19, scope: !3465)
!3467 = !DILocation(line: 862, column: 11, scope: !3465)
!3468 = !DILocation(line: 863, column: 25, scope: !3465)
!3469 = !DILocation(line: 863, column: 21, scope: !3465)
!3470 = !DILocation(line: 863, column: 11, scope: !3465)
!3471 = !DILocation(line: 864, column: 11, scope: !3465)
!3472 = !DILocation(line: 866, column: 8, scope: !3473)
!3473 = distinct !DILexicalBlock(scope: !3465, file: !3, line: 866, column: 4)
!3474 = !DILocation(line: 0, scope: !3473)
!3475 = !DILocation(line: 866, column: 20, scope: !3476)
!3476 = distinct !DILexicalBlock(scope: !3473, file: !3, line: 866, column: 4)
!3477 = !DILocation(line: 866, column: 16, scope: !3476)
!3478 = !DILocation(line: 866, column: 4, scope: !3473)
!3479 = !DILocation(line: 867, column: 9, scope: !3480)
!3480 = distinct !DILexicalBlock(scope: !3481, file: !3, line: 867, column: 5)
!3481 = distinct !DILexicalBlock(scope: !3476, file: !3, line: 866, column: 29)
!3482 = !DILocation(line: 0, scope: !3480)
!3483 = !DILocation(line: 867, column: 21, scope: !3484)
!3484 = distinct !DILexicalBlock(scope: !3480, file: !3, line: 867, column: 5)
!3485 = !DILocation(line: 867, column: 17, scope: !3484)
!3486 = !DILocation(line: 867, column: 5, scope: !3480)
!3487 = !DILocation(line: 868, column: 15, scope: !3488)
!3488 = distinct !DILexicalBlock(scope: !3484, file: !3, line: 867, column: 30)
!3489 = !DILocation(line: 869, column: 16, scope: !3488)
!3490 = !DILocation(line: 870, column: 15, scope: !3488)
!3491 = !DILocation(line: 871, column: 16, scope: !3488)
!3492 = !DILocation(line: 873, column: 28, scope: !3488)
!3493 = !DILocation(line: 873, column: 17, scope: !3488)
!3494 = !DILocation(line: 873, column: 33, scope: !3488)
!3495 = !DILocalVariable(name: "uu2_real", scope: !3250, file: !3, line: 813, type: !104)
!3496 = !DILocation(line: 874, column: 17, scope: !3488)
!3497 = !DILocation(line: 874, column: 31, scope: !3488)
!3498 = !DILocation(line: 874, column: 20, scope: !3488)
!3499 = !DILocation(line: 874, column: 36, scope: !3488)
!3500 = !DILocation(line: 874, column: 19, scope: !3488)
!3501 = !DILocalVariable(name: "uu2_imag", scope: !3250, file: !3, line: 814, type: !104)
!3502 = !DILocation(line: 877, column: 30, scope: !3488)
!3503 = !DILocation(line: 877, column: 34, scope: !3488)
!3504 = !DILocation(line: 877, column: 24, scope: !3488)
!3505 = !DILocation(line: 877, column: 41, scope: !3488)
!3506 = !DILocation(line: 877, column: 44, scope: !3488)
!3507 = !DILocation(line: 877, column: 38, scope: !3488)
!3508 = !DILocation(line: 877, column: 17, scope: !3488)
!3509 = !DILocation(line: 877, column: 49, scope: !3488)
!3510 = !DILocalVariable(name: "x12_real", scope: !3250, file: !3, line: 813, type: !104)
!3511 = !DILocation(line: 878, column: 30, scope: !3488)
!3512 = !DILocation(line: 878, column: 34, scope: !3488)
!3513 = !DILocation(line: 878, column: 24, scope: !3488)
!3514 = !DILocation(line: 878, column: 41, scope: !3488)
!3515 = !DILocation(line: 878, column: 44, scope: !3488)
!3516 = !DILocation(line: 878, column: 38, scope: !3488)
!3517 = !DILocation(line: 878, column: 17, scope: !3488)
!3518 = !DILocation(line: 878, column: 49, scope: !3488)
!3519 = !DILocalVariable(name: "x12_imag", scope: !3250, file: !3, line: 814, type: !104)
!3520 = !DILocation(line: 881, column: 30, scope: !3488)
!3521 = !DILocation(line: 881, column: 34, scope: !3488)
!3522 = !DILocation(line: 881, column: 24, scope: !3488)
!3523 = !DILocation(line: 881, column: 41, scope: !3488)
!3524 = !DILocation(line: 881, column: 44, scope: !3488)
!3525 = !DILocation(line: 881, column: 38, scope: !3488)
!3526 = !DILocation(line: 881, column: 17, scope: !3488)
!3527 = !DILocation(line: 881, column: 49, scope: !3488)
!3528 = !DILocalVariable(name: "x22_real", scope: !3250, file: !3, line: 813, type: !104)
!3529 = !DILocation(line: 882, column: 30, scope: !3488)
!3530 = !DILocation(line: 882, column: 34, scope: !3488)
!3531 = !DILocation(line: 882, column: 24, scope: !3488)
!3532 = !DILocation(line: 882, column: 41, scope: !3488)
!3533 = !DILocation(line: 882, column: 44, scope: !3488)
!3534 = !DILocation(line: 882, column: 38, scope: !3488)
!3535 = !DILocation(line: 882, column: 17, scope: !3488)
!3536 = !DILocation(line: 882, column: 49, scope: !3488)
!3537 = !DILocalVariable(name: "x22_imag", scope: !3250, file: !3, line: 814, type: !104)
!3538 = !DILocation(line: 885, column: 54, scope: !3488)
!3539 = !DILocation(line: 885, column: 19, scope: !3488)
!3540 = !DILocation(line: 885, column: 23, scope: !3488)
!3541 = !DILocation(line: 885, column: 13, scope: !3488)
!3542 = !DILocation(line: 885, column: 30, scope: !3488)
!3543 = !DILocation(line: 885, column: 33, scope: !3488)
!3544 = !DILocation(line: 885, column: 27, scope: !3488)
!3545 = !DILocation(line: 885, column: 6, scope: !3488)
!3546 = !DILocation(line: 885, column: 38, scope: !3488)
!3547 = !DILocation(line: 885, column: 43, scope: !3488)
!3548 = !DILocation(line: 886, column: 54, scope: !3488)
!3549 = !DILocation(line: 886, column: 19, scope: !3488)
!3550 = !DILocation(line: 886, column: 23, scope: !3488)
!3551 = !DILocation(line: 886, column: 13, scope: !3488)
!3552 = !DILocation(line: 886, column: 30, scope: !3488)
!3553 = !DILocation(line: 886, column: 33, scope: !3488)
!3554 = !DILocation(line: 886, column: 27, scope: !3488)
!3555 = !DILocation(line: 886, column: 6, scope: !3488)
!3556 = !DILocation(line: 886, column: 38, scope: !3488)
!3557 = !DILocation(line: 886, column: 43, scope: !3488)
!3558 = !DILocation(line: 888, column: 28, scope: !3488)
!3559 = !DILocalVariable(name: "temp2_real", scope: !3250, file: !3, line: 815, type: !104)
!3560 = !DILocation(line: 889, column: 28, scope: !3488)
!3561 = !DILocalVariable(name: "temp2_imag", scope: !3250, file: !3, line: 816, type: !104)
!3562 = !DILocation(line: 892, column: 55, scope: !3488)
!3563 = !DILocation(line: 892, column: 81, scope: !3488)
!3564 = !DILocation(line: 892, column: 69, scope: !3488)
!3565 = !DILocation(line: 892, column: 19, scope: !3488)
!3566 = !DILocation(line: 892, column: 23, scope: !3488)
!3567 = !DILocation(line: 892, column: 13, scope: !3488)
!3568 = !DILocation(line: 892, column: 30, scope: !3488)
!3569 = !DILocation(line: 892, column: 33, scope: !3488)
!3570 = !DILocation(line: 892, column: 27, scope: !3488)
!3571 = !DILocation(line: 892, column: 6, scope: !3488)
!3572 = !DILocation(line: 892, column: 38, scope: !3488)
!3573 = !DILocation(line: 892, column: 43, scope: !3488)
!3574 = !DILocation(line: 893, column: 55, scope: !3488)
!3575 = !DILocation(line: 893, column: 81, scope: !3488)
!3576 = !DILocation(line: 893, column: 69, scope: !3488)
!3577 = !DILocation(line: 893, column: 19, scope: !3488)
!3578 = !DILocation(line: 893, column: 23, scope: !3488)
!3579 = !DILocation(line: 893, column: 13, scope: !3488)
!3580 = !DILocation(line: 893, column: 30, scope: !3488)
!3581 = !DILocation(line: 893, column: 33, scope: !3488)
!3582 = !DILocation(line: 893, column: 27, scope: !3488)
!3583 = !DILocation(line: 893, column: 6, scope: !3488)
!3584 = !DILocation(line: 893, column: 38, scope: !3488)
!3585 = !DILocation(line: 893, column: 43, scope: !3488)
!3586 = !DILocation(line: 894, column: 5, scope: !3488)
!3587 = !DILocation(line: 867, column: 27, scope: !3484)
!3588 = !DILocation(line: 867, column: 5, scope: !3484)
!3589 = distinct !{!3589, !3486, !3590}
!3590 = !DILocation(line: 894, column: 5, scope: !3480)
!3591 = !DILocation(line: 895, column: 4, scope: !3481)
!3592 = !DILocation(line: 866, column: 26, scope: !3476)
!3593 = !DILocation(line: 866, column: 4, scope: !3476)
!3594 = distinct !{!3594, !3478, !3595}
!3595 = !DILocation(line: 895, column: 4, scope: !3473)
!3596 = !DILocation(line: 897, column: 2, scope: !3281)
!3597 = !DILocation(line: 818, column: 22, scope: !3277)
!3598 = !DILocation(line: 818, column: 2, scope: !3277)
!3599 = distinct !{!3599, !3278, !3600}
!3600 = !DILocation(line: 897, column: 2, scope: !3274)
!3601 = !DILocation(line: 898, column: 1, scope: !3250)
!3602 = !DILocalVariable(name: "x_out", arg: 1, scope: !3603, file: !3, line: 907, type: !98)
!3603 = distinct !DISubprogram(name: "cffts1_gpu_kernel_3", linkageName: "_Z19cffts1_gpu_kernel_3P8dcomplexS0_", scope: !3, file: !3, line: 907, type: !3209, scopeLine: 908, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !1050, retainedNodes: !1051)
!3604 = !DILocation(line: 0, scope: !3603)
!3605 = !DILocalVariable(name: "y0", arg: 2, scope: !3603, file: !3, line: 908, type: !98)
!3606 = !DILocation(line: 909, column: 25, scope: !3603)
!3607 = !DILocation(line: 909, column: 38, scope: !3603)
!3608 = !DILocalVariable(name: "x_y_z", scope: !3603, file: !3, line: 909, type: !97)
!3609 = !DILocation(line: 910, column: 11, scope: !3610)
!3610 = distinct !DILexicalBlock(scope: !3603, file: !3, line: 910, column: 5)
!3611 = !DILocation(line: 910, column: 5, scope: !3603)
!3612 = !DILocation(line: 911, column: 3, scope: !3613)
!3613 = distinct !DILexicalBlock(scope: !3610, file: !3, line: 910, column: 25)
!3614 = !DILocation(line: 913, column: 16, scope: !3603)
!3615 = !DILocalVariable(name: "x", scope: !3603, file: !3, line: 913, type: !97)
!3616 = !DILocation(line: 914, column: 17, scope: !3603)
!3617 = !DILocation(line: 914, column: 23, scope: !3603)
!3618 = !DILocalVariable(name: "y", scope: !3603, file: !3, line: 914, type: !97)
!3619 = !DILocation(line: 915, column: 16, scope: !3603)
!3620 = !DILocalVariable(name: "z", scope: !3603, file: !3, line: 915, type: !97)
!3621 = !DILocation(line: 916, column: 29, scope: !3603)
!3622 = !DILocation(line: 916, column: 26, scope: !3603)
!3623 = !DILocation(line: 916, column: 36, scope: !3603)
!3624 = !DILocation(line: 916, column: 39, scope: !3603)
!3625 = !DILocation(line: 916, column: 33, scope: !3603)
!3626 = !DILocation(line: 916, column: 22, scope: !3603)
!3627 = !DILocation(line: 916, column: 45, scope: !3603)
!3628 = !DILocation(line: 916, column: 2, scope: !3603)
!3629 = !DILocation(line: 916, column: 15, scope: !3603)
!3630 = !DILocation(line: 916, column: 20, scope: !3603)
!3631 = !DILocation(line: 917, column: 29, scope: !3603)
!3632 = !DILocation(line: 917, column: 26, scope: !3603)
!3633 = !DILocation(line: 917, column: 36, scope: !3603)
!3634 = !DILocation(line: 917, column: 39, scope: !3603)
!3635 = !DILocation(line: 917, column: 33, scope: !3603)
!3636 = !DILocation(line: 917, column: 22, scope: !3603)
!3637 = !DILocation(line: 917, column: 45, scope: !3603)
!3638 = !DILocation(line: 917, column: 2, scope: !3603)
!3639 = !DILocation(line: 917, column: 15, scope: !3603)
!3640 = !DILocation(line: 917, column: 20, scope: !3603)
!3641 = !DILocation(line: 918, column: 1, scope: !3603)
!3642 = !DILocalVariable(name: "x_in", arg: 1, scope: !3643, file: !3, line: 969, type: !98)
!3643 = distinct !DISubprogram(name: "cffts2_gpu_kernel_1", linkageName: "_Z19cffts2_gpu_kernel_1P8dcomplexS0_", scope: !3, file: !3, line: 969, type: !3209, scopeLine: 970, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !1050, retainedNodes: !1051)
!3644 = !DILocation(line: 0, scope: !3643)
!3645 = !DILocalVariable(name: "y0", arg: 2, scope: !3643, file: !3, line: 970, type: !98)
!3646 = !DILocation(line: 971, column: 25, scope: !3643)
!3647 = !DILocation(line: 971, column: 38, scope: !3643)
!3648 = !DILocalVariable(name: "x_y_z", scope: !3643, file: !3, line: 971, type: !97)
!3649 = !DILocation(line: 972, column: 11, scope: !3650)
!3650 = distinct !DILexicalBlock(scope: !3643, file: !3, line: 972, column: 5)
!3651 = !DILocation(line: 972, column: 5, scope: !3643)
!3652 = !DILocation(line: 973, column: 3, scope: !3653)
!3653 = distinct !DILexicalBlock(scope: !3650, file: !3, line: 972, column: 25)
!3654 = !DILocation(line: 975, column: 19, scope: !3643)
!3655 = !DILocation(line: 975, column: 31, scope: !3643)
!3656 = !DILocation(line: 975, column: 2, scope: !3643)
!3657 = !DILocation(line: 975, column: 12, scope: !3643)
!3658 = !DILocation(line: 975, column: 17, scope: !3643)
!3659 = !DILocation(line: 976, column: 19, scope: !3643)
!3660 = !DILocation(line: 976, column: 31, scope: !3643)
!3661 = !DILocation(line: 976, column: 2, scope: !3643)
!3662 = !DILocation(line: 976, column: 12, scope: !3643)
!3663 = !DILocation(line: 976, column: 17, scope: !3643)
!3664 = !DILocation(line: 977, column: 1, scope: !3643)
!3665 = !DILocalVariable(name: "is", arg: 1, scope: !3666, file: !3, line: 984, type: !1174)
!3666 = distinct !DISubprogram(name: "cffts2_gpu_kernel_2", linkageName: "_Z19cffts2_gpu_kernel_2iP8dcomplexS0_S0_", scope: !3, file: !3, line: 984, type: !3251, scopeLine: 987, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !1050, retainedNodes: !1051)
!3667 = !DILocation(line: 0, scope: !3666)
!3668 = !DILocalVariable(name: "gty1", arg: 2, scope: !3666, file: !3, line: 985, type: !98)
!3669 = !DILocalVariable(name: "gty2", arg: 3, scope: !3666, file: !3, line: 986, type: !98)
!3670 = !DILocalVariable(name: "u_device", arg: 4, scope: !3666, file: !3, line: 987, type: !98)
!3671 = !DILocation(line: 988, column: 23, scope: !3666)
!3672 = !DILocation(line: 988, column: 36, scope: !3666)
!3673 = !DILocalVariable(name: "x_z", scope: !3666, file: !3, line: 988, type: !97)
!3674 = !DILocation(line: 990, column: 9, scope: !3675)
!3675 = distinct !DILexicalBlock(scope: !3666, file: !3, line: 990, column: 5)
!3676 = !DILocation(line: 990, column: 5, scope: !3666)
!3677 = !DILocation(line: 991, column: 3, scope: !3678)
!3678 = distinct !DILexicalBlock(scope: !3675, file: !3, line: 990, column: 20)
!3679 = !DILocation(line: 998, column: 10, scope: !3666)
!3680 = !DILocalVariable(name: "i", scope: !3666, file: !3, line: 994, type: !97)
!3681 = !DILocation(line: 999, column: 11, scope: !3666)
!3682 = !DILocation(line: 999, column: 17, scope: !3666)
!3683 = !DILocalVariable(name: "k", scope: !3666, file: !3, line: 994, type: !97)
!3684 = !DILocation(line: 1001, column: 20, scope: !3666)
!3685 = !DILocalVariable(name: "logd2", scope: !3666, file: !3, line: 1001, type: !1174)
!3686 = !DILocalVariable(name: "l", scope: !3666, file: !3, line: 995, type: !97)
!3687 = !DILocation(line: 1010, column: 6, scope: !3688)
!3688 = distinct !DILexicalBlock(scope: !3666, file: !3, line: 1010, column: 2)
!3689 = !DILocation(line: 0, scope: !3688)
!3690 = !DILocation(line: 1010, column: 12, scope: !3691)
!3691 = distinct !DILexicalBlock(scope: !3688, file: !3, line: 1010, column: 2)
!3692 = !DILocation(line: 1010, column: 2, scope: !3688)
!3693 = !DILocalVariable(name: "n1", scope: !3666, file: !3, line: 996, type: !97)
!3694 = !DILocation(line: 1012, column: 16, scope: !3695)
!3695 = distinct !DILexicalBlock(scope: !3691, file: !3, line: 1010, column: 26)
!3696 = !DILocation(line: 1012, column: 10, scope: !3695)
!3697 = !DILocalVariable(name: "lk", scope: !3666, file: !3, line: 996, type: !97)
!3698 = !DILocation(line: 1013, column: 20, scope: !3695)
!3699 = !DILocation(line: 1013, column: 10, scope: !3695)
!3700 = !DILocalVariable(name: "li", scope: !3666, file: !3, line: 996, type: !97)
!3701 = !DILocation(line: 1014, column: 10, scope: !3695)
!3702 = !DILocalVariable(name: "lj", scope: !3666, file: !3, line: 996, type: !97)
!3703 = !DILocalVariable(name: "ku", scope: !3666, file: !3, line: 996, type: !97)
!3704 = !DILocalVariable(name: "i1", scope: !3666, file: !3, line: 995, type: !97)
!3705 = !DILocation(line: 1016, column: 7, scope: !3706)
!3706 = distinct !DILexicalBlock(scope: !3695, file: !3, line: 1016, column: 3)
!3707 = !DILocation(line: 0, scope: !3706)
!3708 = !DILocation(line: 1016, column: 19, scope: !3709)
!3709 = distinct !DILexicalBlock(scope: !3706, file: !3, line: 1016, column: 3)
!3710 = !DILocation(line: 1016, column: 15, scope: !3709)
!3711 = !DILocation(line: 1016, column: 3, scope: !3706)
!3712 = !DILocalVariable(name: "k1", scope: !3666, file: !3, line: 995, type: !97)
!3713 = !DILocation(line: 1017, column: 8, scope: !3714)
!3714 = distinct !DILexicalBlock(scope: !3715, file: !3, line: 1017, column: 4)
!3715 = distinct !DILexicalBlock(scope: !3709, file: !3, line: 1016, column: 28)
!3716 = !DILocation(line: 0, scope: !3714)
!3717 = !DILocation(line: 1017, column: 20, scope: !3718)
!3718 = distinct !DILexicalBlock(scope: !3714, file: !3, line: 1017, column: 4)
!3719 = !DILocation(line: 1017, column: 16, scope: !3718)
!3720 = !DILocation(line: 1017, column: 4, scope: !3714)
!3721 = !DILocation(line: 1018, column: 14, scope: !3722)
!3722 = distinct !DILexicalBlock(scope: !3718, file: !3, line: 1017, column: 29)
!3723 = !DILocation(line: 1019, column: 15, scope: !3722)
!3724 = !DILocation(line: 1020, column: 14, scope: !3722)
!3725 = !DILocation(line: 1021, column: 15, scope: !3722)
!3726 = !DILocation(line: 1023, column: 27, scope: !3722)
!3727 = !DILocation(line: 1023, column: 16, scope: !3722)
!3728 = !DILocation(line: 1023, column: 32, scope: !3722)
!3729 = !DILocalVariable(name: "uu1_real", scope: !3666, file: !3, line: 1003, type: !104)
!3730 = !DILocation(line: 1024, column: 16, scope: !3722)
!3731 = !DILocation(line: 1024, column: 30, scope: !3722)
!3732 = !DILocation(line: 1024, column: 19, scope: !3722)
!3733 = !DILocation(line: 1024, column: 35, scope: !3722)
!3734 = !DILocation(line: 1024, column: 18, scope: !3722)
!3735 = !DILocalVariable(name: "uu1_imag", scope: !3666, file: !3, line: 1004, type: !104)
!3736 = !DILocation(line: 1027, column: 29, scope: !3722)
!3737 = !DILocation(line: 1027, column: 33, scope: !3722)
!3738 = !DILocation(line: 1027, column: 23, scope: !3722)
!3739 = !DILocation(line: 1027, column: 40, scope: !3722)
!3740 = !DILocation(line: 1027, column: 43, scope: !3722)
!3741 = !DILocation(line: 1027, column: 37, scope: !3722)
!3742 = !DILocation(line: 1027, column: 16, scope: !3722)
!3743 = !DILocation(line: 1027, column: 48, scope: !3722)
!3744 = !DILocalVariable(name: "x11_real", scope: !3666, file: !3, line: 1003, type: !104)
!3745 = !DILocation(line: 1028, column: 29, scope: !3722)
!3746 = !DILocation(line: 1028, column: 33, scope: !3722)
!3747 = !DILocation(line: 1028, column: 23, scope: !3722)
!3748 = !DILocation(line: 1028, column: 40, scope: !3722)
!3749 = !DILocation(line: 1028, column: 43, scope: !3722)
!3750 = !DILocation(line: 1028, column: 37, scope: !3722)
!3751 = !DILocation(line: 1028, column: 16, scope: !3722)
!3752 = !DILocation(line: 1028, column: 48, scope: !3722)
!3753 = !DILocalVariable(name: "x11_imag", scope: !3666, file: !3, line: 1004, type: !104)
!3754 = !DILocation(line: 1031, column: 29, scope: !3722)
!3755 = !DILocation(line: 1031, column: 33, scope: !3722)
!3756 = !DILocation(line: 1031, column: 23, scope: !3722)
!3757 = !DILocation(line: 1031, column: 40, scope: !3722)
!3758 = !DILocation(line: 1031, column: 43, scope: !3722)
!3759 = !DILocation(line: 1031, column: 37, scope: !3722)
!3760 = !DILocation(line: 1031, column: 16, scope: !3722)
!3761 = !DILocation(line: 1031, column: 48, scope: !3722)
!3762 = !DILocalVariable(name: "x21_real", scope: !3666, file: !3, line: 1003, type: !104)
!3763 = !DILocation(line: 1032, column: 29, scope: !3722)
!3764 = !DILocation(line: 1032, column: 33, scope: !3722)
!3765 = !DILocation(line: 1032, column: 23, scope: !3722)
!3766 = !DILocation(line: 1032, column: 40, scope: !3722)
!3767 = !DILocation(line: 1032, column: 43, scope: !3722)
!3768 = !DILocation(line: 1032, column: 37, scope: !3722)
!3769 = !DILocation(line: 1032, column: 16, scope: !3722)
!3770 = !DILocation(line: 1032, column: 48, scope: !3722)
!3771 = !DILocalVariable(name: "x21_imag", scope: !3666, file: !3, line: 1004, type: !104)
!3772 = !DILocation(line: 1035, column: 53, scope: !3722)
!3773 = !DILocation(line: 1035, column: 18, scope: !3722)
!3774 = !DILocation(line: 1035, column: 22, scope: !3722)
!3775 = !DILocation(line: 1035, column: 12, scope: !3722)
!3776 = !DILocation(line: 1035, column: 29, scope: !3722)
!3777 = !DILocation(line: 1035, column: 32, scope: !3722)
!3778 = !DILocation(line: 1035, column: 26, scope: !3722)
!3779 = !DILocation(line: 1035, column: 5, scope: !3722)
!3780 = !DILocation(line: 1035, column: 37, scope: !3722)
!3781 = !DILocation(line: 1035, column: 42, scope: !3722)
!3782 = !DILocation(line: 1036, column: 53, scope: !3722)
!3783 = !DILocation(line: 1036, column: 18, scope: !3722)
!3784 = !DILocation(line: 1036, column: 22, scope: !3722)
!3785 = !DILocation(line: 1036, column: 12, scope: !3722)
!3786 = !DILocation(line: 1036, column: 29, scope: !3722)
!3787 = !DILocation(line: 1036, column: 32, scope: !3722)
!3788 = !DILocation(line: 1036, column: 26, scope: !3722)
!3789 = !DILocation(line: 1036, column: 5, scope: !3722)
!3790 = !DILocation(line: 1036, column: 37, scope: !3722)
!3791 = !DILocation(line: 1036, column: 42, scope: !3722)
!3792 = !DILocation(line: 1038, column: 26, scope: !3722)
!3793 = !DILocalVariable(name: "temp_real", scope: !3666, file: !3, line: 1007, type: !104)
!3794 = !DILocation(line: 1039, column: 26, scope: !3722)
!3795 = !DILocalVariable(name: "temp_imag", scope: !3666, file: !3, line: 1008, type: !104)
!3796 = !DILocation(line: 1042, column: 54, scope: !3722)
!3797 = !DILocation(line: 1042, column: 79, scope: !3722)
!3798 = !DILocation(line: 1042, column: 67, scope: !3722)
!3799 = !DILocation(line: 1042, column: 18, scope: !3722)
!3800 = !DILocation(line: 1042, column: 22, scope: !3722)
!3801 = !DILocation(line: 1042, column: 12, scope: !3722)
!3802 = !DILocation(line: 1042, column: 29, scope: !3722)
!3803 = !DILocation(line: 1042, column: 32, scope: !3722)
!3804 = !DILocation(line: 1042, column: 26, scope: !3722)
!3805 = !DILocation(line: 1042, column: 5, scope: !3722)
!3806 = !DILocation(line: 1042, column: 37, scope: !3722)
!3807 = !DILocation(line: 1042, column: 42, scope: !3722)
!3808 = !DILocation(line: 1043, column: 54, scope: !3722)
!3809 = !DILocation(line: 1043, column: 79, scope: !3722)
!3810 = !DILocation(line: 1043, column: 67, scope: !3722)
!3811 = !DILocation(line: 1043, column: 18, scope: !3722)
!3812 = !DILocation(line: 1043, column: 22, scope: !3722)
!3813 = !DILocation(line: 1043, column: 12, scope: !3722)
!3814 = !DILocation(line: 1043, column: 29, scope: !3722)
!3815 = !DILocation(line: 1043, column: 32, scope: !3722)
!3816 = !DILocation(line: 1043, column: 26, scope: !3722)
!3817 = !DILocation(line: 1043, column: 5, scope: !3722)
!3818 = !DILocation(line: 1043, column: 37, scope: !3722)
!3819 = !DILocation(line: 1043, column: 42, scope: !3722)
!3820 = !DILocation(line: 1045, column: 4, scope: !3722)
!3821 = !DILocation(line: 1017, column: 26, scope: !3718)
!3822 = !DILocation(line: 1017, column: 4, scope: !3718)
!3823 = distinct !{!3823, !3720, !3824}
!3824 = !DILocation(line: 1045, column: 4, scope: !3714)
!3825 = !DILocation(line: 1046, column: 3, scope: !3715)
!3826 = !DILocation(line: 1016, column: 25, scope: !3709)
!3827 = !DILocation(line: 1016, column: 3, scope: !3709)
!3828 = distinct !{!3828, !3711, !3829}
!3829 = !DILocation(line: 1046, column: 3, scope: !3706)
!3830 = !DILocation(line: 1047, column: 7, scope: !3831)
!3831 = distinct !DILexicalBlock(scope: !3695, file: !3, line: 1047, column: 6)
!3832 = !DILocation(line: 1047, column: 6, scope: !3695)
!3833 = !DILocalVariable(name: "j1", scope: !3666, file: !3, line: 995, type: !97)
!3834 = !DILocation(line: 1048, column: 8, scope: !3835)
!3835 = distinct !DILexicalBlock(scope: !3836, file: !3, line: 1048, column: 4)
!3836 = distinct !DILexicalBlock(scope: !3831, file: !3, line: 1047, column: 15)
!3837 = !DILocation(line: 0, scope: !3835)
!3838 = !DILocation(line: 1048, column: 16, scope: !3839)
!3839 = distinct !DILexicalBlock(scope: !3835, file: !3, line: 1048, column: 4)
!3840 = !DILocation(line: 1048, column: 4, scope: !3835)
!3841 = !DILocation(line: 1050, column: 49, scope: !3842)
!3842 = distinct !DILexicalBlock(scope: !3839, file: !3, line: 1048, column: 26)
!3843 = !DILocation(line: 1050, column: 45, scope: !3842)
!3844 = !DILocation(line: 1050, column: 56, scope: !3842)
!3845 = !DILocation(line: 1050, column: 59, scope: !3842)
!3846 = !DILocation(line: 1050, column: 53, scope: !3842)
!3847 = !DILocation(line: 1050, column: 38, scope: !3842)
!3848 = !DILocation(line: 1050, column: 64, scope: !3842)
!3849 = !DILocation(line: 1050, column: 16, scope: !3842)
!3850 = !DILocation(line: 1050, column: 12, scope: !3842)
!3851 = !DILocation(line: 1050, column: 23, scope: !3842)
!3852 = !DILocation(line: 1050, column: 26, scope: !3842)
!3853 = !DILocation(line: 1050, column: 20, scope: !3842)
!3854 = !DILocation(line: 1050, column: 5, scope: !3842)
!3855 = !DILocation(line: 1050, column: 31, scope: !3842)
!3856 = !DILocation(line: 1050, column: 36, scope: !3842)
!3857 = !DILocation(line: 1051, column: 49, scope: !3842)
!3858 = !DILocation(line: 1051, column: 45, scope: !3842)
!3859 = !DILocation(line: 1051, column: 56, scope: !3842)
!3860 = !DILocation(line: 1051, column: 59, scope: !3842)
!3861 = !DILocation(line: 1051, column: 53, scope: !3842)
!3862 = !DILocation(line: 1051, column: 38, scope: !3842)
!3863 = !DILocation(line: 1051, column: 64, scope: !3842)
!3864 = !DILocation(line: 1051, column: 16, scope: !3842)
!3865 = !DILocation(line: 1051, column: 12, scope: !3842)
!3866 = !DILocation(line: 1051, column: 23, scope: !3842)
!3867 = !DILocation(line: 1051, column: 26, scope: !3842)
!3868 = !DILocation(line: 1051, column: 20, scope: !3842)
!3869 = !DILocation(line: 1051, column: 5, scope: !3842)
!3870 = !DILocation(line: 1051, column: 31, scope: !3842)
!3871 = !DILocation(line: 1051, column: 36, scope: !3842)
!3872 = !DILocation(line: 1052, column: 4, scope: !3842)
!3873 = !DILocation(line: 1048, column: 23, scope: !3839)
!3874 = !DILocation(line: 1048, column: 4, scope: !3839)
!3875 = distinct !{!3875, !3840, !3876}
!3876 = !DILocation(line: 1052, column: 4, scope: !3835)
!3877 = !DILocation(line: 1053, column: 3, scope: !3836)
!3878 = !DILocation(line: 1056, column: 16, scope: !3879)
!3879 = distinct !DILexicalBlock(scope: !3831, file: !3, line: 1054, column: 7)
!3880 = !DILocation(line: 1056, column: 19, scope: !3879)
!3881 = !DILocation(line: 1056, column: 11, scope: !3879)
!3882 = !DILocation(line: 1057, column: 25, scope: !3879)
!3883 = !DILocation(line: 1057, column: 21, scope: !3879)
!3884 = !DILocation(line: 1057, column: 11, scope: !3879)
!3885 = !DILocation(line: 1058, column: 11, scope: !3879)
!3886 = !DILocation(line: 1060, column: 8, scope: !3887)
!3887 = distinct !DILexicalBlock(scope: !3879, file: !3, line: 1060, column: 4)
!3888 = !DILocation(line: 0, scope: !3887)
!3889 = !DILocation(line: 1060, column: 20, scope: !3890)
!3890 = distinct !DILexicalBlock(scope: !3887, file: !3, line: 1060, column: 4)
!3891 = !DILocation(line: 1060, column: 16, scope: !3890)
!3892 = !DILocation(line: 1060, column: 4, scope: !3887)
!3893 = !DILocation(line: 1061, column: 9, scope: !3894)
!3894 = distinct !DILexicalBlock(scope: !3895, file: !3, line: 1061, column: 5)
!3895 = distinct !DILexicalBlock(scope: !3890, file: !3, line: 1060, column: 29)
!3896 = !DILocation(line: 0, scope: !3894)
!3897 = !DILocation(line: 1061, column: 21, scope: !3898)
!3898 = distinct !DILexicalBlock(scope: !3894, file: !3, line: 1061, column: 5)
!3899 = !DILocation(line: 1061, column: 17, scope: !3898)
!3900 = !DILocation(line: 1061, column: 5, scope: !3894)
!3901 = !DILocation(line: 1062, column: 15, scope: !3902)
!3902 = distinct !DILexicalBlock(scope: !3898, file: !3, line: 1061, column: 30)
!3903 = !DILocation(line: 1063, column: 16, scope: !3902)
!3904 = !DILocation(line: 1064, column: 15, scope: !3902)
!3905 = !DILocation(line: 1065, column: 16, scope: !3902)
!3906 = !DILocation(line: 1067, column: 28, scope: !3902)
!3907 = !DILocation(line: 1067, column: 17, scope: !3902)
!3908 = !DILocation(line: 1067, column: 33, scope: !3902)
!3909 = !DILocalVariable(name: "uu2_real", scope: !3666, file: !3, line: 1005, type: !104)
!3910 = !DILocation(line: 1068, column: 17, scope: !3902)
!3911 = !DILocation(line: 1068, column: 31, scope: !3902)
!3912 = !DILocation(line: 1068, column: 20, scope: !3902)
!3913 = !DILocation(line: 1068, column: 36, scope: !3902)
!3914 = !DILocation(line: 1068, column: 19, scope: !3902)
!3915 = !DILocalVariable(name: "uu2_imag", scope: !3666, file: !3, line: 1006, type: !104)
!3916 = !DILocation(line: 1071, column: 30, scope: !3902)
!3917 = !DILocation(line: 1071, column: 34, scope: !3902)
!3918 = !DILocation(line: 1071, column: 24, scope: !3902)
!3919 = !DILocation(line: 1071, column: 41, scope: !3902)
!3920 = !DILocation(line: 1071, column: 44, scope: !3902)
!3921 = !DILocation(line: 1071, column: 38, scope: !3902)
!3922 = !DILocation(line: 1071, column: 17, scope: !3902)
!3923 = !DILocation(line: 1071, column: 49, scope: !3902)
!3924 = !DILocalVariable(name: "x12_real", scope: !3666, file: !3, line: 1005, type: !104)
!3925 = !DILocation(line: 1072, column: 30, scope: !3902)
!3926 = !DILocation(line: 1072, column: 34, scope: !3902)
!3927 = !DILocation(line: 1072, column: 24, scope: !3902)
!3928 = !DILocation(line: 1072, column: 41, scope: !3902)
!3929 = !DILocation(line: 1072, column: 44, scope: !3902)
!3930 = !DILocation(line: 1072, column: 38, scope: !3902)
!3931 = !DILocation(line: 1072, column: 17, scope: !3902)
!3932 = !DILocation(line: 1072, column: 49, scope: !3902)
!3933 = !DILocalVariable(name: "x12_imag", scope: !3666, file: !3, line: 1006, type: !104)
!3934 = !DILocation(line: 1075, column: 30, scope: !3902)
!3935 = !DILocation(line: 1075, column: 34, scope: !3902)
!3936 = !DILocation(line: 1075, column: 24, scope: !3902)
!3937 = !DILocation(line: 1075, column: 41, scope: !3902)
!3938 = !DILocation(line: 1075, column: 44, scope: !3902)
!3939 = !DILocation(line: 1075, column: 38, scope: !3902)
!3940 = !DILocation(line: 1075, column: 17, scope: !3902)
!3941 = !DILocation(line: 1075, column: 49, scope: !3902)
!3942 = !DILocalVariable(name: "x22_real", scope: !3666, file: !3, line: 1005, type: !104)
!3943 = !DILocation(line: 1076, column: 30, scope: !3902)
!3944 = !DILocation(line: 1076, column: 34, scope: !3902)
!3945 = !DILocation(line: 1076, column: 24, scope: !3902)
!3946 = !DILocation(line: 1076, column: 41, scope: !3902)
!3947 = !DILocation(line: 1076, column: 44, scope: !3902)
!3948 = !DILocation(line: 1076, column: 38, scope: !3902)
!3949 = !DILocation(line: 1076, column: 17, scope: !3902)
!3950 = !DILocation(line: 1076, column: 49, scope: !3902)
!3951 = !DILocalVariable(name: "x22_imag", scope: !3666, file: !3, line: 1006, type: !104)
!3952 = !DILocation(line: 1079, column: 54, scope: !3902)
!3953 = !DILocation(line: 1079, column: 19, scope: !3902)
!3954 = !DILocation(line: 1079, column: 23, scope: !3902)
!3955 = !DILocation(line: 1079, column: 13, scope: !3902)
!3956 = !DILocation(line: 1079, column: 30, scope: !3902)
!3957 = !DILocation(line: 1079, column: 33, scope: !3902)
!3958 = !DILocation(line: 1079, column: 27, scope: !3902)
!3959 = !DILocation(line: 1079, column: 6, scope: !3902)
!3960 = !DILocation(line: 1079, column: 38, scope: !3902)
!3961 = !DILocation(line: 1079, column: 43, scope: !3902)
!3962 = !DILocation(line: 1080, column: 54, scope: !3902)
!3963 = !DILocation(line: 1080, column: 19, scope: !3902)
!3964 = !DILocation(line: 1080, column: 23, scope: !3902)
!3965 = !DILocation(line: 1080, column: 13, scope: !3902)
!3966 = !DILocation(line: 1080, column: 30, scope: !3902)
!3967 = !DILocation(line: 1080, column: 33, scope: !3902)
!3968 = !DILocation(line: 1080, column: 27, scope: !3902)
!3969 = !DILocation(line: 1080, column: 6, scope: !3902)
!3970 = !DILocation(line: 1080, column: 38, scope: !3902)
!3971 = !DILocation(line: 1080, column: 43, scope: !3902)
!3972 = !DILocation(line: 1082, column: 28, scope: !3902)
!3973 = !DILocalVariable(name: "temp2_real", scope: !3666, file: !3, line: 1007, type: !104)
!3974 = !DILocation(line: 1083, column: 28, scope: !3902)
!3975 = !DILocalVariable(name: "temp2_imag", scope: !3666, file: !3, line: 1008, type: !104)
!3976 = !DILocation(line: 1086, column: 55, scope: !3902)
!3977 = !DILocation(line: 1086, column: 81, scope: !3902)
!3978 = !DILocation(line: 1086, column: 69, scope: !3902)
!3979 = !DILocation(line: 1086, column: 19, scope: !3902)
!3980 = !DILocation(line: 1086, column: 23, scope: !3902)
!3981 = !DILocation(line: 1086, column: 13, scope: !3902)
!3982 = !DILocation(line: 1086, column: 30, scope: !3902)
!3983 = !DILocation(line: 1086, column: 33, scope: !3902)
!3984 = !DILocation(line: 1086, column: 27, scope: !3902)
!3985 = !DILocation(line: 1086, column: 6, scope: !3902)
!3986 = !DILocation(line: 1086, column: 38, scope: !3902)
!3987 = !DILocation(line: 1086, column: 43, scope: !3902)
!3988 = !DILocation(line: 1087, column: 55, scope: !3902)
!3989 = !DILocation(line: 1087, column: 81, scope: !3902)
!3990 = !DILocation(line: 1087, column: 69, scope: !3902)
!3991 = !DILocation(line: 1087, column: 19, scope: !3902)
!3992 = !DILocation(line: 1087, column: 23, scope: !3902)
!3993 = !DILocation(line: 1087, column: 13, scope: !3902)
!3994 = !DILocation(line: 1087, column: 30, scope: !3902)
!3995 = !DILocation(line: 1087, column: 33, scope: !3902)
!3996 = !DILocation(line: 1087, column: 27, scope: !3902)
!3997 = !DILocation(line: 1087, column: 6, scope: !3902)
!3998 = !DILocation(line: 1087, column: 38, scope: !3902)
!3999 = !DILocation(line: 1087, column: 43, scope: !3902)
!4000 = !DILocation(line: 1088, column: 5, scope: !3902)
!4001 = !DILocation(line: 1061, column: 27, scope: !3898)
!4002 = !DILocation(line: 1061, column: 5, scope: !3898)
!4003 = distinct !{!4003, !3900, !4004}
!4004 = !DILocation(line: 1088, column: 5, scope: !3894)
!4005 = !DILocation(line: 1089, column: 4, scope: !3895)
!4006 = !DILocation(line: 1060, column: 26, scope: !3890)
!4007 = !DILocation(line: 1060, column: 4, scope: !3890)
!4008 = distinct !{!4008, !3892, !4009}
!4009 = !DILocation(line: 1089, column: 4, scope: !3887)
!4010 = !DILocation(line: 1091, column: 2, scope: !3695)
!4011 = !DILocation(line: 1010, column: 22, scope: !3691)
!4012 = !DILocation(line: 1010, column: 2, scope: !3691)
!4013 = distinct !{!4013, !3692, !4014}
!4014 = !DILocation(line: 1091, column: 2, scope: !3688)
!4015 = !DILocation(line: 1092, column: 1, scope: !3666)
!4016 = !DILocalVariable(name: "x_out", arg: 1, scope: !4017, file: !3, line: 1101, type: !98)
!4017 = distinct !DISubprogram(name: "cffts2_gpu_kernel_3", linkageName: "_Z19cffts2_gpu_kernel_3P8dcomplexS0_", scope: !3, file: !3, line: 1101, type: !3209, scopeLine: 1102, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !1050, retainedNodes: !1051)
!4018 = !DILocation(line: 0, scope: !4017)
!4019 = !DILocalVariable(name: "y0", arg: 2, scope: !4017, file: !3, line: 1102, type: !98)
!4020 = !DILocation(line: 1103, column: 25, scope: !4017)
!4021 = !DILocation(line: 1103, column: 38, scope: !4017)
!4022 = !DILocalVariable(name: "x_y_z", scope: !4017, file: !3, line: 1103, type: !97)
!4023 = !DILocation(line: 1104, column: 11, scope: !4024)
!4024 = distinct !DILexicalBlock(scope: !4017, file: !3, line: 1104, column: 5)
!4025 = !DILocation(line: 1104, column: 5, scope: !4017)
!4026 = !DILocation(line: 1105, column: 3, scope: !4027)
!4027 = distinct !DILexicalBlock(scope: !4024, file: !3, line: 1104, column: 25)
!4028 = !DILocation(line: 1107, column: 22, scope: !4017)
!4029 = !DILocation(line: 1107, column: 32, scope: !4017)
!4030 = !DILocation(line: 1107, column: 2, scope: !4017)
!4031 = !DILocation(line: 1107, column: 15, scope: !4017)
!4032 = !DILocation(line: 1107, column: 20, scope: !4017)
!4033 = !DILocation(line: 1108, column: 22, scope: !4017)
!4034 = !DILocation(line: 1108, column: 32, scope: !4017)
!4035 = !DILocation(line: 1108, column: 2, scope: !4017)
!4036 = !DILocation(line: 1108, column: 15, scope: !4017)
!4037 = !DILocation(line: 1108, column: 20, scope: !4017)
!4038 = !DILocation(line: 1109, column: 1, scope: !4017)
!4039 = !DILocalVariable(name: "x_in", arg: 1, scope: !4040, file: !3, line: 1258, type: !98)
!4040 = distinct !DISubprogram(name: "cffts3_gpu_kernel_1", linkageName: "_Z19cffts3_gpu_kernel_1P8dcomplexS0_", scope: !3, file: !3, line: 1258, type: !3209, scopeLine: 1259, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !1050, retainedNodes: !1051)
!4041 = !DILocation(line: 0, scope: !4040)
!4042 = !DILocalVariable(name: "y0", arg: 2, scope: !4040, file: !3, line: 1259, type: !98)
!4043 = !DILocation(line: 1260, column: 25, scope: !4040)
!4044 = !DILocation(line: 1260, column: 38, scope: !4040)
!4045 = !DILocalVariable(name: "x_y_z", scope: !4040, file: !3, line: 1260, type: !97)
!4046 = !DILocation(line: 1261, column: 11, scope: !4047)
!4047 = distinct !DILexicalBlock(scope: !4040, file: !3, line: 1261, column: 5)
!4048 = !DILocation(line: 1261, column: 5, scope: !4040)
!4049 = !DILocation(line: 1262, column: 3, scope: !4050)
!4050 = distinct !DILexicalBlock(scope: !4047, file: !3, line: 1261, column: 25)
!4051 = !DILocation(line: 1264, column: 19, scope: !4040)
!4052 = !DILocation(line: 1264, column: 31, scope: !4040)
!4053 = !DILocation(line: 1264, column: 2, scope: !4040)
!4054 = !DILocation(line: 1264, column: 12, scope: !4040)
!4055 = !DILocation(line: 1264, column: 17, scope: !4040)
!4056 = !DILocation(line: 1265, column: 19, scope: !4040)
!4057 = !DILocation(line: 1265, column: 31, scope: !4040)
!4058 = !DILocation(line: 1265, column: 2, scope: !4040)
!4059 = !DILocation(line: 1265, column: 12, scope: !4040)
!4060 = !DILocation(line: 1265, column: 17, scope: !4040)
!4061 = !DILocation(line: 1266, column: 1, scope: !4040)
!4062 = !DILocalVariable(name: "is", arg: 1, scope: !4063, file: !3, line: 1273, type: !1174)
!4063 = distinct !DISubprogram(name: "cffts3_gpu_kernel_2", linkageName: "_Z19cffts3_gpu_kernel_2iP8dcomplexS0_S0_", scope: !3, file: !3, line: 1273, type: !3251, scopeLine: 1276, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !1050, retainedNodes: !1051)
!4064 = !DILocation(line: 0, scope: !4063)
!4065 = !DILocalVariable(name: "gty1", arg: 2, scope: !4063, file: !3, line: 1274, type: !98)
!4066 = !DILocalVariable(name: "gty2", arg: 3, scope: !4063, file: !3, line: 1275, type: !98)
!4067 = !DILocalVariable(name: "u_device", arg: 4, scope: !4063, file: !3, line: 1276, type: !98)
!4068 = !DILocation(line: 1277, column: 23, scope: !4063)
!4069 = !DILocation(line: 1277, column: 36, scope: !4063)
!4070 = !DILocalVariable(name: "x_y", scope: !4063, file: !3, line: 1277, type: !97)
!4071 = !DILocation(line: 1278, column: 9, scope: !4072)
!4072 = distinct !DILexicalBlock(scope: !4063, file: !3, line: 1278, column: 5)
!4073 = !DILocation(line: 1278, column: 5, scope: !4063)
!4074 = !DILocation(line: 1279, column: 3, scope: !4075)
!4075 = distinct !DILexicalBlock(scope: !4072, file: !3, line: 1278, column: 20)
!4076 = !DILocation(line: 1282, column: 4, scope: !4063)
!4077 = !DILocation(line: 1281, column: 2, scope: !4063)
!4078 = !DILocation(line: 1289, column: 1, scope: !4063)
!4079 = !DILocalVariable(name: "x_out", arg: 1, scope: !4080, file: !3, line: 1298, type: !98)
!4080 = distinct !DISubprogram(name: "cffts3_gpu_kernel_3", linkageName: "_Z19cffts3_gpu_kernel_3P8dcomplexS0_", scope: !3, file: !3, line: 1298, type: !3209, scopeLine: 1299, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !1050, retainedNodes: !1051)
!4081 = !DILocation(line: 0, scope: !4080)
!4082 = !DILocalVariable(name: "y0", arg: 2, scope: !4080, file: !3, line: 1299, type: !98)
!4083 = !DILocation(line: 1300, column: 25, scope: !4080)
!4084 = !DILocation(line: 1300, column: 38, scope: !4080)
!4085 = !DILocalVariable(name: "x_y_z", scope: !4080, file: !3, line: 1300, type: !97)
!4086 = !DILocation(line: 1301, column: 11, scope: !4087)
!4087 = distinct !DILexicalBlock(scope: !4080, file: !3, line: 1301, column: 5)
!4088 = !DILocation(line: 1301, column: 5, scope: !4080)
!4089 = !DILocation(line: 1302, column: 3, scope: !4090)
!4090 = distinct !DILexicalBlock(scope: !4087, file: !3, line: 1301, column: 25)
!4091 = !DILocation(line: 1304, column: 22, scope: !4080)
!4092 = !DILocation(line: 1304, column: 32, scope: !4080)
!4093 = !DILocation(line: 1304, column: 2, scope: !4080)
!4094 = !DILocation(line: 1304, column: 15, scope: !4080)
!4095 = !DILocation(line: 1304, column: 20, scope: !4080)
!4096 = !DILocation(line: 1305, column: 22, scope: !4080)
!4097 = !DILocation(line: 1305, column: 32, scope: !4080)
!4098 = !DILocation(line: 1305, column: 2, scope: !4080)
!4099 = !DILocation(line: 1305, column: 15, scope: !4080)
!4100 = !DILocation(line: 1305, column: 20, scope: !4080)
!4101 = !DILocation(line: 1306, column: 1, scope: !4080)
!4102 = !DILocation(line: 1341, column: 16, scope: !4103)
!4103 = distinct !DILexicalBlock(scope: !3172, file: !3, line: 1341, column: 5)
!4104 = !DILocation(line: 1341, column: 5, scope: !3172)
!4105 = !DILocalVariable(name: "i", scope: !4106, file: !3, line: 1342, type: !97)
!4106 = distinct !DILexicalBlock(scope: !4107, file: !3, line: 1342, column: 3)
!4107 = distinct !DILexicalBlock(scope: !4103, file: !3, line: 1341, column: 20)
!4108 = !DILocation(line: 0, scope: !4106)
!4109 = !DILocation(line: 1342, column: 7, scope: !4106)
!4110 = !DILocation(line: 1342, column: 17, scope: !4111)
!4111 = distinct !DILexicalBlock(scope: !4106, file: !3, line: 1342, column: 3)
!4112 = !DILocation(line: 1342, column: 3, scope: !4106)
!4113 = !DILocation(line: 1343, column: 20, scope: !4114)
!4114 = distinct !DILexicalBlock(scope: !4111, file: !3, line: 1342, column: 34)
!4115 = !DILocation(line: 1343, column: 4, scope: !4114)
!4116 = !DILocation(line: 1343, column: 18, scope: !4114)
!4117 = !DILocation(line: 1344, column: 3, scope: !4114)
!4118 = !DILocation(line: 1342, column: 30, scope: !4111)
!4119 = !DILocation(line: 1342, column: 3, scope: !4111)
!4120 = distinct !{!4120, !4112, !4121}
!4121 = !DILocation(line: 1344, column: 3, scope: !4106)
!4122 = !DILocation(line: 1345, column: 2, scope: !4107)
!4123 = !DILocation(line: 1346, column: 16, scope: !4124)
!4124 = distinct !DILexicalBlock(scope: !3172, file: !3, line: 1346, column: 5)
!4125 = !DILocation(line: 1346, column: 5, scope: !3172)
!4126 = !DILocation(line: 1347, column: 24, scope: !4127)
!4127 = distinct !DILexicalBlock(scope: !4124, file: !3, line: 1346, column: 20)
!4128 = !DILocation(line: 1347, column: 38, scope: !4127)
!4129 = !DILocation(line: 1347, column: 42, scope: !4127)
!4130 = !DILocation(line: 1347, column: 3, scope: !4127)
!4131 = !DILocation(line: 1347, column: 17, scope: !4127)
!4132 = !DILocation(line: 1347, column: 22, scope: !4127)
!4133 = !DILocation(line: 1348, column: 14, scope: !4127)
!4134 = !DILocation(line: 1348, column: 30, scope: !4127)
!4135 = !DILocation(line: 1348, column: 35, scope: !4127)
!4136 = !DILocation(line: 1348, column: 49, scope: !4127)
!4137 = !DILocation(line: 1348, column: 3, scope: !4127)
!4138 = !DILocation(line: 1349, column: 24, scope: !4127)
!4139 = !DILocation(line: 1349, column: 38, scope: !4127)
!4140 = !DILocation(line: 1349, column: 42, scope: !4127)
!4141 = !DILocation(line: 1349, column: 3, scope: !4127)
!4142 = !DILocation(line: 1349, column: 17, scope: !4127)
!4143 = !DILocation(line: 1349, column: 22, scope: !4127)
!4144 = !DILocation(line: 1350, column: 14, scope: !4127)
!4145 = !DILocation(line: 1350, column: 30, scope: !4127)
!4146 = !DILocation(line: 1350, column: 35, scope: !4127)
!4147 = !DILocation(line: 1350, column: 49, scope: !4127)
!4148 = !DILocation(line: 1350, column: 3, scope: !4127)
!4149 = !DILocation(line: 1351, column: 2, scope: !4127)
!4150 = !DILocation(line: 1352, column: 1, scope: !3172)
