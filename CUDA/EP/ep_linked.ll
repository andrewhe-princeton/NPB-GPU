; ModuleID = 'ep_linked.bc'
source_filename = "llvm-link-cudafe"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.cudaDeviceProp = type { [256 x i8], i64, i64, i32, i32, i64, i32, [3 x i32], [3 x i32], i32, i64, i32, i32, i64, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, [2 x i32], [2 x i32], [3 x i32], [2 x i32], [3 x i32], [3 x i32], i32, [2 x i32], [3 x i32], [2 x i32], i32, [2 x i32], [3 x i32], [2 x i32], [3 x i32], i32, [2 x i32], i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i64, i32, i32, i32, i32 }
%struct.dim3 = type { i32, i32, i32 }
%struct.CUstream_st = type opaque

$_ZN4dim3C2Ejjj = comdat any

@q_host = dso_local global double* null, align 8, !dbg !0
@q_device = dso_local global double* null, align 8, !dbg !107
@sx_host = dso_local global double* null, align 8, !dbg !109
@sx_device = dso_local global double* null, align 8, !dbg !111
@sy_host = dso_local global double* null, align 8, !dbg !113
@sy_device = dso_local global double* null, align 8, !dbg !115
@threads_per_block = dso_local global i32 0, align 4, !dbg !117
@blocks_per_grid = dso_local global i32 0, align 4, !dbg !119
@size_q = dso_local global i64 0, align 8, !dbg !121
@size_sx = dso_local global i64 0, align 8, !dbg !126
@size_sy = dso_local global i64 0, align 8, !dbg !128
@gpu_device_id = dso_local global i32 0, align 4, !dbg !130
@total_devices = dso_local global i32 0, align 4, !dbg !132
@gpu_device_properties = dso_local global %struct.cudaDeviceProp zeroinitializer, align 8, !dbg !134
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_ep.cu, i8* null }]
@_ZL1q = internal global double* null, align 8, !dbg !105
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
@.str.39 = private unnamed_addr constant [7 x i8] c"%15.0f\00", align 1
@.str.40 = private unnamed_addr constant [65 x i8] c"\0A\0A NAS Parallel Benchmarks 4.1 CUDA C++ version - EP Benchmark\0A\0A\00", align 1
@.str.41 = private unnamed_addr constant [43 x i8] c" Number of random numbers generated: %15s\0A\00", align 1
@.str.42 = private unnamed_addr constant [26 x i8] c"\0A EP Benchmark Results:\0A\0A\00", align 1
@.str.43 = private unnamed_addr constant [19 x i8] c" CPU Time =%10.4f\0A\00", align 1
@.str.44 = private unnamed_addr constant [12 x i8] c" N = 2^%5d\0A\00", align 1
@.str.45 = private unnamed_addr constant [30 x i8] c" No. Gaussian Pairs = %15.0f\0A\00", align 1
@.str.46 = private unnamed_addr constant [25 x i8] c" Sums = %25.15e %25.15e\0A\00", align 1
@.str.47 = private unnamed_addr constant [11 x i8] c" Counts: \0A\00", align 1
@.str.48 = private unnamed_addr constant [11 x i8] c"%3d%15.0f\0A\00", align 1
@.str.49 = private unnamed_addr constant [10 x i8] c"%5s\09%25s\0A\00", align 1
@.str.50 = private unnamed_addr constant [11 x i8] c"GPU Kernel\00", align 1
@.str.51 = private unnamed_addr constant [18 x i8] c"Threads Per Block\00", align 1
@.str.52 = private unnamed_addr constant [11 x i8] c"%29s\09%25d\0A\00", align 1
@.str.53 = private unnamed_addr constant [4 x i8] c" ep\00", align 1
@.str.54 = private unnamed_addr constant [3 x i8] c"EP\00", align 1
@.str.55 = private unnamed_addr constant [25 x i8] c"Random numbers generated\00", align 1
@.str.56 = private unnamed_addr constant [4 x i8] c"4.1\00", align 1
@.str.57 = private unnamed_addr constant [12 x i8] c"04 Feb 2026\00", align 1
@.str.58 = private unnamed_addr constant [6 x i8] c"\93\E3\BC\FD\7F\00", align 1
@.str.59 = private unnamed_addr constant [42 x i8] c"Intel(R) Xeon(R) CPU E5-2697 v3 @ 2.60GHz\00", align 1
@.str.60 = private unnamed_addr constant [23 x i8] c"${NVCC} ${EXTRA_STUFF}\00", align 1
@.str.61 = private unnamed_addr constant [6 x i8] c"$(CC)\00", align 1
@.str.62 = private unnamed_addr constant [5 x i8] c"-lm \00", align 1
@.str.63 = private unnamed_addr constant [13 x i8] c"-I../common \00", align 1
@.str.64 = private unnamed_addr constant [4 x i8] c"-O3\00", align 1
@.str.65 = private unnamed_addr constant [7 x i8] c"randdp\00", align 1

; Function Attrs: convergent noinline nounwind
define dso_local void @_Z10gpu_kernelPdS_S_d(double* %q_global, double* %sx_global, double* %sy_global, double %an) #0 !dbg !1046 {
entry:
  %a.addr.i69 = alloca double, align 8
  call void @llvm.dbg.declare(metadata double* %a.addr.i69, metadata !1049, metadata !DIExpression()), !dbg !1052
  %x.addr.i = alloca double, align 8
  call void @llvm.dbg.declare(metadata double* %x.addr.i, metadata !1062, metadata !DIExpression()), !dbg !1064
  %a.addr.i = alloca double, align 8
  call void @llvm.dbg.declare(metadata double* %a.addr.i, metadata !1049, metadata !DIExpression()), !dbg !1066
  %f.addr.i68 = alloca double, align 8
  call void @llvm.dbg.declare(metadata double* %f.addr.i68, metadata !1068, metadata !DIExpression()), !dbg !1070
  %f.addr.i67 = alloca double, align 8
  call void @llvm.dbg.declare(metadata double* %f.addr.i67, metadata !1068, metadata !DIExpression()), !dbg !1072
  %f.addr.i66 = alloca double, align 8
  call void @llvm.dbg.declare(metadata double* %f.addr.i66, metadata !1068, metadata !DIExpression()), !dbg !1074
  %f.addr.i = alloca double, align 8
  call void @llvm.dbg.declare(metadata double* %f.addr.i, metadata !1068, metadata !DIExpression()), !dbg !1076
  %q_global.addr = alloca double*, align 8
  %sx_global.addr = alloca double*, align 8
  %sy_global.addr = alloca double*, align 8
  %an.addr = alloca double, align 8
  %x_local = alloca [256 x double], align 8
  %q_local = alloca [10 x double], align 8
  %sx_local = alloca double, align 8
  %sy_local = alloca double, align 8
  %t1 = alloca double, align 8
  %t2 = alloca double, align 8
  %t3 = alloca double, align 8
  %t4 = alloca double, align 8
  %x1 = alloca double, align 8
  %x2 = alloca double, align 8
  %seed = alloca double, align 8
  %i = alloca i32, align 4
  %ii = alloca i32, align 4
  %ik = alloca i32, align 4
  %kk = alloca i32, align 4
  %l = alloca i32, align 4
  store double* %q_global, double** %q_global.addr, align 8
  call void @llvm.dbg.declare(metadata double** %q_global.addr, metadata !1078, metadata !DIExpression()), !dbg !1079
  store double* %sx_global, double** %sx_global.addr, align 8
  call void @llvm.dbg.declare(metadata double** %sx_global.addr, metadata !1080, metadata !DIExpression()), !dbg !1081
  store double* %sy_global, double** %sy_global.addr, align 8
  call void @llvm.dbg.declare(metadata double** %sy_global.addr, metadata !1082, metadata !DIExpression()), !dbg !1083
  store double %an, double* %an.addr, align 8
  call void @llvm.dbg.declare(metadata double* %an.addr, metadata !1084, metadata !DIExpression()), !dbg !1085
  call void @llvm.dbg.declare(metadata [256 x double]* %x_local, metadata !1086, metadata !DIExpression()), !dbg !1088
  call void @llvm.dbg.declare(metadata [10 x double]* %q_local, metadata !1089, metadata !DIExpression()), !dbg !1093
  call void @llvm.dbg.declare(metadata double* %sx_local, metadata !1094, metadata !DIExpression()), !dbg !1095
  call void @llvm.dbg.declare(metadata double* %sy_local, metadata !1096, metadata !DIExpression()), !dbg !1097
  call void @llvm.dbg.declare(metadata double* %t1, metadata !1098, metadata !DIExpression()), !dbg !1099
  call void @llvm.dbg.declare(metadata double* %t2, metadata !1100, metadata !DIExpression()), !dbg !1101
  call void @llvm.dbg.declare(metadata double* %t3, metadata !1102, metadata !DIExpression()), !dbg !1103
  call void @llvm.dbg.declare(metadata double* %t4, metadata !1104, metadata !DIExpression()), !dbg !1105
  call void @llvm.dbg.declare(metadata double* %x1, metadata !1106, metadata !DIExpression()), !dbg !1107
  call void @llvm.dbg.declare(metadata double* %x2, metadata !1108, metadata !DIExpression()), !dbg !1109
  call void @llvm.dbg.declare(metadata double* %seed, metadata !1110, metadata !DIExpression()), !dbg !1111
  call void @llvm.dbg.declare(metadata i32* %i, metadata !1112, metadata !DIExpression()), !dbg !1113
  call void @llvm.dbg.declare(metadata i32* %ii, metadata !1114, metadata !DIExpression()), !dbg !1115
  call void @llvm.dbg.declare(metadata i32* %ik, metadata !1116, metadata !DIExpression()), !dbg !1117
  call void @llvm.dbg.declare(metadata i32* %kk, metadata !1118, metadata !DIExpression()), !dbg !1119
  call void @llvm.dbg.declare(metadata i32* %l, metadata !1120, metadata !DIExpression()), !dbg !1121
  %arrayidx = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 0, !dbg !1122
  store double 0.000000e+00, double* %arrayidx, align 8, !dbg !1123
  %arrayidx1 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 1, !dbg !1124
  store double 0.000000e+00, double* %arrayidx1, align 8, !dbg !1125
  %arrayidx2 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 2, !dbg !1126
  store double 0.000000e+00, double* %arrayidx2, align 8, !dbg !1127
  %arrayidx3 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 3, !dbg !1128
  store double 0.000000e+00, double* %arrayidx3, align 8, !dbg !1129
  %arrayidx4 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 4, !dbg !1130
  store double 0.000000e+00, double* %arrayidx4, align 8, !dbg !1131
  %arrayidx5 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 5, !dbg !1132
  store double 0.000000e+00, double* %arrayidx5, align 8, !dbg !1133
  %arrayidx6 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 6, !dbg !1134
  store double 0.000000e+00, double* %arrayidx6, align 8, !dbg !1135
  %arrayidx7 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 7, !dbg !1136
  store double 0.000000e+00, double* %arrayidx7, align 8, !dbg !1137
  %arrayidx8 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 8, !dbg !1138
  store double 0.000000e+00, double* %arrayidx8, align 8, !dbg !1139
  %arrayidx9 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 9, !dbg !1140
  store double 0.000000e+00, double* %arrayidx9, align 8, !dbg !1141
  store double 0.000000e+00, double* %sx_local, align 8, !dbg !1142
  store double 0.000000e+00, double* %sy_local, align 8, !dbg !1143
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #10, !dbg !1144, !range !1181
  %1 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #10, !dbg !1182, !range !1226
  %mul = mul i32 %0, %1, !dbg !1227
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #10, !dbg !1228, !range !1256
  %add = add i32 %mul, %2, !dbg !1257
  store i32 %add, i32* %kk, align 4, !dbg !1258
  %3 = load i32, i32* %kk, align 4, !dbg !1259
  %cmp = icmp sge i32 %3, 4096, !dbg !1261
  br i1 %cmp, label %if.then, label %if.end, !dbg !1262

if.then:                                          ; preds = %entry
  br label %for.end65, !dbg !1263

if.end:                                           ; preds = %entry
  store double 0x41B033C4D7000000, double* %t1, align 8, !dbg !1265
  %4 = load double, double* %an.addr, align 8, !dbg !1266
  store double %4, double* %t2, align 8, !dbg !1267
  store i32 1, i32* %i, align 4, !dbg !1268
  br label %for.cond, !dbg !1270

for.cond:                                         ; preds = %for.inc, %if.end
  %5 = load i32, i32* %i, align 4, !dbg !1271
  %cmp12 = icmp sle i32 %5, 100, !dbg !1273
  br i1 %cmp12, label %for.body, label %for.end, !dbg !1274

for.body:                                         ; preds = %for.cond
  %6 = load i32, i32* %kk, align 4, !dbg !1275
  %div = sdiv i32 %6, 2, !dbg !1277
  store i32 %div, i32* %ik, align 4, !dbg !1278
  %7 = load i32, i32* %ik, align 4, !dbg !1279
  %mul13 = mul nsw i32 2, %7, !dbg !1281
  %8 = load i32, i32* %kk, align 4, !dbg !1282
  %cmp14 = icmp ne i32 %mul13, %8, !dbg !1283
  br i1 %cmp14, label %if.then15, label %if.end17, !dbg !1284

if.then15:                                        ; preds = %for.body
  %9 = load double, double* %t2, align 8, !dbg !1285
  %call16 = call double @_Z13randlc_devicePdd(double* %t1, double %9) #11, !dbg !1287
  store double %call16, double* %t3, align 8, !dbg !1288
  br label %if.end17, !dbg !1289

if.end17:                                         ; preds = %if.then15, %for.body
  %10 = load i32, i32* %ik, align 4, !dbg !1290
  %cmp18 = icmp eq i32 %10, 0, !dbg !1292
  br i1 %cmp18, label %if.then19, label %if.end20, !dbg !1293

if.then19:                                        ; preds = %if.end17
  br label %for.end, !dbg !1294

if.end20:                                         ; preds = %if.end17
  %11 = load double, double* %t2, align 8, !dbg !1296
  %call21 = call double @_Z13randlc_devicePdd(double* %t2, double %11) #11, !dbg !1297
  store double %call21, double* %t3, align 8, !dbg !1298
  %12 = load i32, i32* %ik, align 4, !dbg !1299
  store i32 %12, i32* %kk, align 4, !dbg !1300
  br label %for.inc, !dbg !1301

for.inc:                                          ; preds = %if.end20
  %13 = load i32, i32* %i, align 4, !dbg !1302
  %inc = add nsw i32 %13, 1, !dbg !1302
  store i32 %inc, i32* %i, align 4, !dbg !1302
  br label %for.cond, !dbg !1303, !llvm.loop !1304

for.end:                                          ; preds = %if.then19, %for.cond
  %14 = load double, double* %t1, align 8, !dbg !1306
  store double %14, double* %seed, align 8, !dbg !1307
  store i32 0, i32* %ii, align 4, !dbg !1308
  br label %for.cond22, !dbg !1309

for.cond22:                                       ; preds = %for.inc63, %for.end
  %15 = load i32, i32* %ii, align 4, !dbg !1310
  %cmp23 = icmp slt i32 %15, 65536, !dbg !1311
  br i1 %cmp23, label %for.body24, label %for.end65, !dbg !1312

for.body24:                                       ; preds = %for.cond22
  %arraydecay = getelementptr inbounds [256 x double], [256 x double]* %x_local, i64 0, i64 0, !dbg !1313
  call void @_Z13vranlc_deviceiPddS_(i32 256, double* %seed, double 0x41D2309CE5400000, double* %arraydecay) #11, !dbg !1314
  store i32 0, i32* %i, align 4, !dbg !1315
  br label %for.cond25, !dbg !1316

for.cond25:                                       ; preds = %for.inc60, %for.body24
  %16 = load i32, i32* %i, align 4, !dbg !1317
  %cmp26 = icmp slt i32 %16, 128, !dbg !1318
  br i1 %cmp26, label %for.body27, label %for.end62, !dbg !1319

for.body27:                                       ; preds = %for.cond25
  %17 = load i32, i32* %i, align 4, !dbg !1320
  %mul28 = mul nsw i32 2, %17, !dbg !1321
  %idxprom = sext i32 %mul28 to i64, !dbg !1322
  %arrayidx29 = getelementptr inbounds [256 x double], [256 x double]* %x_local, i64 0, i64 %idxprom, !dbg !1322
  %18 = load double, double* %arrayidx29, align 8, !dbg !1322
  %mul30 = fmul contract double 2.000000e+00, %18, !dbg !1323
  %sub = fsub contract double %mul30, 1.000000e+00, !dbg !1324
  store double %sub, double* %x1, align 8, !dbg !1325
  %19 = load i32, i32* %i, align 4, !dbg !1326
  %mul31 = mul nsw i32 2, %19, !dbg !1327
  %add32 = add nsw i32 %mul31, 1, !dbg !1328
  %idxprom33 = sext i32 %add32 to i64, !dbg !1329
  %arrayidx34 = getelementptr inbounds [256 x double], [256 x double]* %x_local, i64 0, i64 %idxprom33, !dbg !1329
  %20 = load double, double* %arrayidx34, align 8, !dbg !1329
  %mul35 = fmul contract double 2.000000e+00, %20, !dbg !1330
  %sub36 = fsub contract double %mul35, 1.000000e+00, !dbg !1331
  store double %sub36, double* %x2, align 8, !dbg !1332
  %21 = load double, double* %x1, align 8, !dbg !1333
  %22 = load double, double* %x1, align 8, !dbg !1334
  %mul37 = fmul contract double %21, %22, !dbg !1335
  %23 = load double, double* %x2, align 8, !dbg !1336
  %24 = load double, double* %x2, align 8, !dbg !1337
  %mul38 = fmul contract double %23, %24, !dbg !1338
  %add39 = fadd contract double %mul37, %mul38, !dbg !1339
  store double %add39, double* %t1, align 8, !dbg !1340
  %25 = load double, double* %t1, align 8, !dbg !1341
  %cmp40 = fcmp ole double %25, 1.000000e+00, !dbg !1342
  br i1 %cmp40, label %if.then41, label %if.end59, !dbg !1343

if.then41:                                        ; preds = %for.body27
  %26 = load double, double* %t1, align 8, !dbg !1344
  store double %26, double* %a.addr.i69, align 8
  %27 = load double, double* %a.addr.i69, align 8, !dbg !1345
  %28 = call i32 @llvm.nvvm.d2i.hi(double %27) #10, !dbg !1346
  %29 = call i32 @llvm.nvvm.d2i.lo(double %27) #10, !dbg !1346
  %30 = fcmp ogt double %27, 0.000000e+00, !dbg !1346
  br i1 %30, label %31, label %33, !dbg !1346

31:                                               ; preds = %if.then41
  %32 = icmp slt i32 %28, 2146435072, !dbg !1346
  br label %33, !dbg !1346

33:                                               ; preds = %31, %if.then41
  %34 = phi i1 [ false, %if.then41 ], [ %32, %31 ], !dbg !1346
  br i1 %34, label %35, label %90, !dbg !1346

35:                                               ; preds = %33
  %36 = icmp slt i32 %28, 1048576, !dbg !1346
  br i1 %36, label %37, label %41, !dbg !1346

37:                                               ; preds = %35
  %38 = fmul double %27, 0x4350000000000000, !dbg !1346
  %39 = call i32 @llvm.nvvm.d2i.hi(double %38) #10, !dbg !1346
  %40 = call i32 @llvm.nvvm.d2i.lo(double %38) #10, !dbg !1346
  br label %41, !dbg !1346

41:                                               ; preds = %37, %35
  %ihi.0.i.i70 = phi i32 [ %39, %37 ], [ %28, %35 ], !dbg !1346
  %ilo.0.i.i71 = phi i32 [ %40, %37 ], [ %29, %35 ], !dbg !1346
  %e.0.i.i72 = phi i32 [ -1077, %37 ], [ -1023, %35 ], !dbg !1346
  %42 = lshr i32 %ihi.0.i.i70, 20, !dbg !1346
  %43 = add i32 %e.0.i.i72, %42, !dbg !1346
  %44 = and i32 %ihi.0.i.i70, -2146435073, !dbg !1346
  %45 = or i32 %44, 1072693248, !dbg !1346
  %46 = call double @llvm.nvvm.lohi.i2d(i32 %ilo.0.i.i71, i32 %45) #10, !dbg !1346
  %47 = icmp sgt i32 %45, 1073127582, !dbg !1346
  br i1 %47, label %48, label %54, !dbg !1346

48:                                               ; preds = %41
  %49 = call i32 @llvm.nvvm.d2i.lo(double %46) #10, !dbg !1346
  %50 = call i32 @llvm.nvvm.d2i.hi(double %46) #10, !dbg !1346
  %51 = add i32 -1048576, %50, !dbg !1346
  %52 = call double @llvm.nvvm.lohi.i2d(i32 %49, i32 %51) #10, !dbg !1346
  %53 = add nsw i32 %43, 1, !dbg !1346
  br label %54, !dbg !1346

54:                                               ; preds = %48, %41
  %m.0.i.i73 = phi double [ %52, %48 ], [ %46, %41 ], !dbg !1346
  %e.1.i.i74 = phi i32 [ %53, %48 ], [ %43, %41 ], !dbg !1346
  %55 = fsub double %m.0.i.i73, 1.000000e+00, !dbg !1346
  %56 = fadd double %m.0.i.i73, 1.000000e+00, !dbg !1346
  %57 = call double asm "rcp.approx.ftz.f64 $0,$1;", "=d,d"(double %56) #10, !dbg !1346
  %58 = fsub double -0.000000e+00, %56, !dbg !1346
  %59 = call double @llvm.nvvm.fma.rn.d(double %58, double %57, double 1.000000e+00) #10, !dbg !1346
  %60 = call double @llvm.nvvm.fma.rn.d(double %59, double %59, double %59) #10, !dbg !1346
  %61 = call double @llvm.nvvm.fma.rn.d(double %60, double %57, double %57) #10, !dbg !1346
  %62 = fmul double %55, %61, !dbg !1346
  %63 = fadd double %62, %62, !dbg !1346
  %64 = fmul double %63, %63, !dbg !1346
  %65 = call double @llvm.nvvm.fma.rn.d(double 0x3EB1380B3AE80F1E, double %64, double 0x3ED0EE258B7A8B04) #10, !dbg !1346
  %66 = call double @llvm.nvvm.fma.rn.d(double %65, double %64, double 0x3EF3B2669F02676F) #10, !dbg !1346
  %67 = call double @llvm.nvvm.fma.rn.d(double %66, double %64, double 0x3F1745CBA9AB0956) #10, !dbg !1346
  %68 = call double @llvm.nvvm.fma.rn.d(double %67, double %64, double 0x3F3C71C72D1B5154) #10, !dbg !1346
  %69 = call double @llvm.nvvm.fma.rn.d(double %68, double %64, double 0x3F624924923BE72D) #10, !dbg !1346
  %70 = call double @llvm.nvvm.fma.rn.d(double %69, double %64, double 0x3F8999999999A3C4) #10, !dbg !1346
  %71 = call double @llvm.nvvm.fma.rn.d(double %70, double %64, double 0x3FB5555555555554) #10, !dbg !1346
  %72 = fsub double %55, %63, !dbg !1346
  %73 = fmul double 2.000000e+00, %72, !dbg !1346
  %74 = fsub double -0.000000e+00, %63, !dbg !1346
  %75 = call double @llvm.nvvm.fma.rn.d(double %74, double %55, double %73) #10, !dbg !1346
  %76 = fmul double %61, %75, !dbg !1346
  %77 = fmul double %71, %64, !dbg !1346
  %78 = call double @llvm.nvvm.fma.rn.d(double %77, double %63, double %76) #10, !dbg !1346
  %79 = xor i32 -2147483648, %e.1.i.i74, !dbg !1346
  %80 = call double @llvm.nvvm.lohi.i2d(i32 %79, i32 1127219200) #10, !dbg !1346
  %81 = call double @llvm.nvvm.lohi.i2d(i32 -2147483648, i32 1127219200) #10, !dbg !1346
  %82 = fsub double %80, %81, !dbg !1346
  %83 = call double @llvm.nvvm.fma.rn.d(double %82, double 0x3FE62E42FEFA39EF, double %63) #10, !dbg !1346
  %84 = fsub double -0.000000e+00, %82, !dbg !1346
  %85 = call double @llvm.nvvm.fma.rn.d(double %84, double 0x3FE62E42FEFA39EF, double %83) #10, !dbg !1346
  %86 = fsub double %85, %63, !dbg !1346
  %87 = fsub double %78, %86, !dbg !1346
  %88 = call double @llvm.nvvm.fma.rn.d(double %82, double 0x3C7ABC9E3B39803F, double %87) #10, !dbg !1346
  %89 = fadd double %83, %88, !dbg !1346
  br label %_ZL3logd.exit79, !dbg !1346

90:                                               ; preds = %33
  %91 = call double @llvm.nvvm.fabs.d(double %27) #10, !dbg !1346
  %92 = fcmp ole double %91, 0x7FF0000000000000, !dbg !1346
  %93 = xor i1 %92, true, !dbg !1346
  %94 = zext i1 %93 to i32, !dbg !1346
  br i1 %93, label %95, label %97, !dbg !1346

95:                                               ; preds = %90
  %96 = fadd double %27, %27, !dbg !1346
  br label %106, !dbg !1346

97:                                               ; preds = %90
  %98 = fcmp oeq double %27, 0.000000e+00, !dbg !1346
  br i1 %98, label %99, label %100, !dbg !1346

99:                                               ; preds = %97
  br label %105, !dbg !1346

100:                                              ; preds = %97
  %101 = fcmp oeq double %27, 0x7FF0000000000000, !dbg !1346
  br i1 %101, label %102, label %103, !dbg !1346

102:                                              ; preds = %100
  br label %104, !dbg !1346

103:                                              ; preds = %100
  br label %104, !dbg !1346

104:                                              ; preds = %103, %102
  %q.0.i.i75 = phi double [ %27, %102 ], [ 0xFFF8000000000000, %103 ], !dbg !1346
  br label %105, !dbg !1346

105:                                              ; preds = %104, %99
  %q.1.i.i76 = phi double [ 0xFFF0000000000000, %99 ], [ %q.0.i.i75, %104 ], !dbg !1346
  br label %106, !dbg !1346

106:                                              ; preds = %105, %95
  %q.2.i.i77 = phi double [ %96, %95 ], [ %q.1.i.i76, %105 ], !dbg !1346
  br label %_ZL3logd.exit79, !dbg !1346

_ZL3logd.exit79:                                  ; preds = %106, %54
  %q.3.i.i78 = phi double [ %89, %54 ], [ %q.2.i.i77, %106 ], !dbg !1346
  %mul43 = fmul contract double -2.000000e+00, %q.3.i.i78, !dbg !1347
  %107 = load double, double* %t1, align 8, !dbg !1348
  %div44 = fdiv double %mul43, %107, !dbg !1349
  store double %div44, double* %x.addr.i, align 8
  %108 = load double, double* %x.addr.i, align 8, !dbg !1350
  %109 = call double @llvm.nvvm.sqrt.rn.d(double %108) #10, !dbg !1351
  store double %109, double* %t2, align 8, !dbg !1352
  %110 = load double, double* %x1, align 8, !dbg !1353
  %111 = load double, double* %t2, align 8, !dbg !1354
  %mul46 = fmul contract double %110, %111, !dbg !1355
  store double %mul46, double* %a.addr.i, align 8
  %112 = load double, double* %a.addr.i, align 8, !dbg !1356
  %113 = call i32 @llvm.nvvm.d2i.hi(double %112) #10, !dbg !1357
  %114 = call i32 @llvm.nvvm.d2i.lo(double %112) #10, !dbg !1357
  %115 = fcmp ogt double %112, 0.000000e+00, !dbg !1357
  br i1 %115, label %116, label %118, !dbg !1357

116:                                              ; preds = %_ZL3logd.exit79
  %117 = icmp slt i32 %113, 2146435072, !dbg !1357
  br label %118, !dbg !1357

118:                                              ; preds = %116, %_ZL3logd.exit79
  %119 = phi i1 [ false, %_ZL3logd.exit79 ], [ %117, %116 ], !dbg !1357
  br i1 %119, label %120, label %175, !dbg !1357

120:                                              ; preds = %118
  %121 = icmp slt i32 %113, 1048576, !dbg !1357
  br i1 %121, label %122, label %126, !dbg !1357

122:                                              ; preds = %120
  %123 = fmul double %112, 0x4350000000000000, !dbg !1357
  %124 = call i32 @llvm.nvvm.d2i.hi(double %123) #10, !dbg !1357
  %125 = call i32 @llvm.nvvm.d2i.lo(double %123) #10, !dbg !1357
  br label %126, !dbg !1357

126:                                              ; preds = %122, %120
  %ihi.0.i.i = phi i32 [ %124, %122 ], [ %113, %120 ], !dbg !1357
  %ilo.0.i.i = phi i32 [ %125, %122 ], [ %114, %120 ], !dbg !1357
  %e.0.i.i = phi i32 [ -1077, %122 ], [ -1023, %120 ], !dbg !1357
  %127 = lshr i32 %ihi.0.i.i, 20, !dbg !1357
  %128 = add i32 %e.0.i.i, %127, !dbg !1357
  %129 = and i32 %ihi.0.i.i, -2146435073, !dbg !1357
  %130 = or i32 %129, 1072693248, !dbg !1357
  %131 = call double @llvm.nvvm.lohi.i2d(i32 %ilo.0.i.i, i32 %130) #10, !dbg !1357
  %132 = icmp sgt i32 %130, 1073127582, !dbg !1357
  br i1 %132, label %133, label %139, !dbg !1357

133:                                              ; preds = %126
  %134 = call i32 @llvm.nvvm.d2i.lo(double %131) #10, !dbg !1357
  %135 = call i32 @llvm.nvvm.d2i.hi(double %131) #10, !dbg !1357
  %136 = add i32 -1048576, %135, !dbg !1357
  %137 = call double @llvm.nvvm.lohi.i2d(i32 %134, i32 %136) #10, !dbg !1357
  %138 = add nsw i32 %128, 1, !dbg !1357
  br label %139, !dbg !1357

139:                                              ; preds = %133, %126
  %m.0.i.i = phi double [ %137, %133 ], [ %131, %126 ], !dbg !1357
  %e.1.i.i = phi i32 [ %138, %133 ], [ %128, %126 ], !dbg !1357
  %140 = fsub double %m.0.i.i, 1.000000e+00, !dbg !1357
  %141 = fadd double %m.0.i.i, 1.000000e+00, !dbg !1357
  %142 = call double asm "rcp.approx.ftz.f64 $0,$1;", "=d,d"(double %141) #10, !dbg !1357
  %143 = fsub double -0.000000e+00, %141, !dbg !1357
  %144 = call double @llvm.nvvm.fma.rn.d(double %143, double %142, double 1.000000e+00) #10, !dbg !1357
  %145 = call double @llvm.nvvm.fma.rn.d(double %144, double %144, double %144) #10, !dbg !1357
  %146 = call double @llvm.nvvm.fma.rn.d(double %145, double %142, double %142) #10, !dbg !1357
  %147 = fmul double %140, %146, !dbg !1357
  %148 = fadd double %147, %147, !dbg !1357
  %149 = fmul double %148, %148, !dbg !1357
  %150 = call double @llvm.nvvm.fma.rn.d(double 0x3EB1380B3AE80F1E, double %149, double 0x3ED0EE258B7A8B04) #10, !dbg !1357
  %151 = call double @llvm.nvvm.fma.rn.d(double %150, double %149, double 0x3EF3B2669F02676F) #10, !dbg !1357
  %152 = call double @llvm.nvvm.fma.rn.d(double %151, double %149, double 0x3F1745CBA9AB0956) #10, !dbg !1357
  %153 = call double @llvm.nvvm.fma.rn.d(double %152, double %149, double 0x3F3C71C72D1B5154) #10, !dbg !1357
  %154 = call double @llvm.nvvm.fma.rn.d(double %153, double %149, double 0x3F624924923BE72D) #10, !dbg !1357
  %155 = call double @llvm.nvvm.fma.rn.d(double %154, double %149, double 0x3F8999999999A3C4) #10, !dbg !1357
  %156 = call double @llvm.nvvm.fma.rn.d(double %155, double %149, double 0x3FB5555555555554) #10, !dbg !1357
  %157 = fsub double %140, %148, !dbg !1357
  %158 = fmul double 2.000000e+00, %157, !dbg !1357
  %159 = fsub double -0.000000e+00, %148, !dbg !1357
  %160 = call double @llvm.nvvm.fma.rn.d(double %159, double %140, double %158) #10, !dbg !1357
  %161 = fmul double %146, %160, !dbg !1357
  %162 = fmul double %156, %149, !dbg !1357
  %163 = call double @llvm.nvvm.fma.rn.d(double %162, double %148, double %161) #10, !dbg !1357
  %164 = xor i32 -2147483648, %e.1.i.i, !dbg !1357
  %165 = call double @llvm.nvvm.lohi.i2d(i32 %164, i32 1127219200) #10, !dbg !1357
  %166 = call double @llvm.nvvm.lohi.i2d(i32 -2147483648, i32 1127219200) #10, !dbg !1357
  %167 = fsub double %165, %166, !dbg !1357
  %168 = call double @llvm.nvvm.fma.rn.d(double %167, double 0x3FE62E42FEFA39EF, double %148) #10, !dbg !1357
  %169 = fsub double -0.000000e+00, %167, !dbg !1357
  %170 = call double @llvm.nvvm.fma.rn.d(double %169, double 0x3FE62E42FEFA39EF, double %168) #10, !dbg !1357
  %171 = fsub double %170, %148, !dbg !1357
  %172 = fsub double %163, %171, !dbg !1357
  %173 = call double @llvm.nvvm.fma.rn.d(double %167, double 0x3C7ABC9E3B39803F, double %172) #10, !dbg !1357
  %174 = fadd double %168, %173, !dbg !1357
  br label %_ZL3logd.exit, !dbg !1357

175:                                              ; preds = %118
  %176 = call double @llvm.nvvm.fabs.d(double %112) #10, !dbg !1357
  %177 = fcmp ole double %176, 0x7FF0000000000000, !dbg !1357
  %178 = xor i1 %177, true, !dbg !1357
  %179 = zext i1 %178 to i32, !dbg !1357
  br i1 %178, label %180, label %182, !dbg !1357

180:                                              ; preds = %175
  %181 = fadd double %112, %112, !dbg !1357
  br label %191, !dbg !1357

182:                                              ; preds = %175
  %183 = fcmp oeq double %112, 0.000000e+00, !dbg !1357
  br i1 %183, label %184, label %185, !dbg !1357

184:                                              ; preds = %182
  br label %190, !dbg !1357

185:                                              ; preds = %182
  %186 = fcmp oeq double %112, 0x7FF0000000000000, !dbg !1357
  br i1 %186, label %187, label %188, !dbg !1357

187:                                              ; preds = %185
  br label %189, !dbg !1357

188:                                              ; preds = %185
  br label %189, !dbg !1357

189:                                              ; preds = %188, %187
  %q.0.i.i = phi double [ %112, %187 ], [ 0xFFF8000000000000, %188 ], !dbg !1357
  br label %190, !dbg !1357

190:                                              ; preds = %189, %184
  %q.1.i.i = phi double [ 0xFFF0000000000000, %184 ], [ %q.0.i.i, %189 ], !dbg !1357
  br label %191, !dbg !1357

191:                                              ; preds = %190, %180
  %q.2.i.i = phi double [ %181, %180 ], [ %q.1.i.i, %190 ], !dbg !1357
  br label %_ZL3logd.exit, !dbg !1357

_ZL3logd.exit:                                    ; preds = %191, %139
  %q.3.i.i = phi double [ %174, %139 ], [ %q.2.i.i, %191 ], !dbg !1357
  store double %q.3.i.i, double* %t3, align 8, !dbg !1358
  %192 = load double, double* %x2, align 8, !dbg !1359
  %193 = load double, double* %t2, align 8, !dbg !1360
  %mul48 = fmul contract double %192, %193, !dbg !1361
  store double %mul48, double* %t4, align 8, !dbg !1362
  %194 = load double, double* %t3, align 8, !dbg !1363
  store double %194, double* %f.addr.i68, align 8
  %195 = load double, double* %f.addr.i68, align 8, !dbg !1364
  %196 = call double @llvm.nvvm.fabs.d(double %195) #10, !dbg !1365
  %197 = load double, double* %t4, align 8, !dbg !1363
  store double %197, double* %f.addr.i67, align 8
  %198 = load double, double* %f.addr.i67, align 8, !dbg !1366
  %199 = call double @llvm.nvvm.fabs.d(double %198) #10, !dbg !1367
  %cmp51 = fcmp ogt double %196, %199, !dbg !1363
  br i1 %cmp51, label %cond.true, label %cond.false, !dbg !1363

cond.true:                                        ; preds = %_ZL3logd.exit
  %200 = load double, double* %t3, align 8, !dbg !1363
  store double %200, double* %f.addr.i66, align 8
  %201 = load double, double* %f.addr.i66, align 8, !dbg !1368
  %202 = call double @llvm.nvvm.fabs.d(double %201) #10, !dbg !1369
  br label %cond.end, !dbg !1363

cond.false:                                       ; preds = %_ZL3logd.exit
  %203 = load double, double* %t4, align 8, !dbg !1363
  store double %203, double* %f.addr.i, align 8
  %204 = load double, double* %f.addr.i, align 8, !dbg !1370
  %205 = call double @llvm.nvvm.fabs.d(double %204) #10, !dbg !1371
  br label %cond.end, !dbg !1363

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi double [ %202, %cond.true ], [ %205, %cond.false ], !dbg !1363
  %conv = fptosi double %cond to i32, !dbg !1363
  store i32 %conv, i32* %l, align 4, !dbg !1372
  %206 = load i32, i32* %l, align 4, !dbg !1373
  %idxprom54 = sext i32 %206 to i64, !dbg !1374
  %arrayidx55 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 %idxprom54, !dbg !1374
  %207 = load double, double* %arrayidx55, align 8, !dbg !1375
  %add56 = fadd contract double %207, 1.000000e+00, !dbg !1375
  store double %add56, double* %arrayidx55, align 8, !dbg !1375
  %208 = load double, double* %sx_local, align 8, !dbg !1376
  %209 = load double, double* %t3, align 8, !dbg !1377
  %add57 = fadd contract double %208, %209, !dbg !1378
  store double %add57, double* %sx_local, align 8, !dbg !1379
  %210 = load double, double* %t4, align 8, !dbg !1380
  %211 = load double, double* %sy_local, align 8, !dbg !1381
  %add58 = fadd contract double %211, %210, !dbg !1381
  store double %add58, double* %sy_local, align 8, !dbg !1381
  br label %if.end59, !dbg !1382

if.end59:                                         ; preds = %cond.end, %for.body27
  br label %for.inc60, !dbg !1383

for.inc60:                                        ; preds = %if.end59
  %212 = load i32, i32* %i, align 4, !dbg !1384
  %inc61 = add nsw i32 %212, 1, !dbg !1384
  store i32 %inc61, i32* %i, align 4, !dbg !1384
  br label %for.cond25, !dbg !1385, !llvm.loop !1386

for.end62:                                        ; preds = %for.cond25
  br label %for.inc63, !dbg !1388

for.inc63:                                        ; preds = %for.end62
  %213 = load i32, i32* %ii, align 4, !dbg !1389
  %add64 = add nsw i32 %213, 128, !dbg !1390
  store i32 %add64, i32* %ii, align 4, !dbg !1391
  br label %for.cond22, !dbg !1392, !llvm.loop !1393

for.end65:                                        ; preds = %for.cond22, %if.then
  ret void, !dbg !1395
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
define dso_local double @_Z13randlc_devicePdd(double* %x, double %a) #3 !dbg !1396 {
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
  call void @llvm.dbg.declare(metadata double** %x.addr, metadata !1399, metadata !DIExpression()), !dbg !1400
  store double %a, double* %a.addr, align 8
  call void @llvm.dbg.declare(metadata double* %a.addr, metadata !1401, metadata !DIExpression()), !dbg !1402
  call void @llvm.dbg.declare(metadata double* %t1, metadata !1403, metadata !DIExpression()), !dbg !1404
  call void @llvm.dbg.declare(metadata double* %t2, metadata !1405, metadata !DIExpression()), !dbg !1406
  call void @llvm.dbg.declare(metadata double* %t3, metadata !1407, metadata !DIExpression()), !dbg !1408
  call void @llvm.dbg.declare(metadata double* %t4, metadata !1409, metadata !DIExpression()), !dbg !1410
  call void @llvm.dbg.declare(metadata double* %a1, metadata !1411, metadata !DIExpression()), !dbg !1412
  call void @llvm.dbg.declare(metadata double* %a2, metadata !1413, metadata !DIExpression()), !dbg !1414
  call void @llvm.dbg.declare(metadata double* %x1, metadata !1415, metadata !DIExpression()), !dbg !1416
  call void @llvm.dbg.declare(metadata double* %x2, metadata !1417, metadata !DIExpression()), !dbg !1418
  call void @llvm.dbg.declare(metadata double* %z, metadata !1419, metadata !DIExpression()), !dbg !1420
  %0 = load double, double* %a.addr, align 8, !dbg !1421
  %mul = fmul contract double 0x3E80000000000000, %0, !dbg !1422
  store double %mul, double* %t1, align 8, !dbg !1423
  %1 = load double, double* %t1, align 8, !dbg !1424
  %conv = fptosi double %1 to i32, !dbg !1424
  %conv1 = sitofp i32 %conv to double, !dbg !1425
  store double %conv1, double* %a1, align 8, !dbg !1426
  %2 = load double, double* %a.addr, align 8, !dbg !1427
  %3 = load double, double* %a1, align 8, !dbg !1428
  %mul2 = fmul contract double 0x4160000000000000, %3, !dbg !1429
  %sub = fsub contract double %2, %mul2, !dbg !1430
  store double %sub, double* %a2, align 8, !dbg !1431
  %4 = load double*, double** %x.addr, align 8, !dbg !1432
  %5 = load double, double* %4, align 8, !dbg !1433
  %mul3 = fmul contract double 0x3E80000000000000, %5, !dbg !1434
  store double %mul3, double* %t1, align 8, !dbg !1435
  %6 = load double, double* %t1, align 8, !dbg !1436
  %conv4 = fptosi double %6 to i32, !dbg !1436
  %conv5 = sitofp i32 %conv4 to double, !dbg !1437
  store double %conv5, double* %x1, align 8, !dbg !1438
  %7 = load double*, double** %x.addr, align 8, !dbg !1439
  %8 = load double, double* %7, align 8, !dbg !1440
  %9 = load double, double* %x1, align 8, !dbg !1441
  %mul6 = fmul contract double 0x4160000000000000, %9, !dbg !1442
  %sub7 = fsub contract double %8, %mul6, !dbg !1443
  store double %sub7, double* %x2, align 8, !dbg !1444
  %10 = load double, double* %a1, align 8, !dbg !1445
  %11 = load double, double* %x2, align 8, !dbg !1446
  %mul8 = fmul contract double %10, %11, !dbg !1447
  %12 = load double, double* %a2, align 8, !dbg !1448
  %13 = load double, double* %x1, align 8, !dbg !1449
  %mul9 = fmul contract double %12, %13, !dbg !1450
  %add = fadd contract double %mul8, %mul9, !dbg !1451
  store double %add, double* %t1, align 8, !dbg !1452
  %14 = load double, double* %t1, align 8, !dbg !1453
  %mul10 = fmul contract double 0x3E80000000000000, %14, !dbg !1454
  %conv11 = fptosi double %mul10 to i32, !dbg !1455
  %conv12 = sitofp i32 %conv11 to double, !dbg !1456
  store double %conv12, double* %t2, align 8, !dbg !1457
  %15 = load double, double* %t1, align 8, !dbg !1458
  %16 = load double, double* %t2, align 8, !dbg !1459
  %mul13 = fmul contract double 0x4160000000000000, %16, !dbg !1460
  %sub14 = fsub contract double %15, %mul13, !dbg !1461
  store double %sub14, double* %z, align 8, !dbg !1462
  %17 = load double, double* %z, align 8, !dbg !1463
  %mul15 = fmul contract double 0x4160000000000000, %17, !dbg !1464
  %18 = load double, double* %a2, align 8, !dbg !1465
  %19 = load double, double* %x2, align 8, !dbg !1466
  %mul16 = fmul contract double %18, %19, !dbg !1467
  %add17 = fadd contract double %mul15, %mul16, !dbg !1468
  store double %add17, double* %t3, align 8, !dbg !1469
  %20 = load double, double* %t3, align 8, !dbg !1470
  %mul18 = fmul contract double 0x3D10000000000000, %20, !dbg !1471
  %conv19 = fptosi double %mul18 to i32, !dbg !1472
  %conv20 = sitofp i32 %conv19 to double, !dbg !1473
  store double %conv20, double* %t4, align 8, !dbg !1474
  %21 = load double, double* %t3, align 8, !dbg !1475
  %22 = load double, double* %t4, align 8, !dbg !1476
  %mul21 = fmul contract double 0x42D0000000000000, %22, !dbg !1477
  %sub22 = fsub contract double %21, %mul21, !dbg !1478
  %23 = load double*, double** %x.addr, align 8, !dbg !1479
  store double %sub22, double* %23, align 8, !dbg !1480
  %24 = load double*, double** %x.addr, align 8, !dbg !1481
  %25 = load double, double* %24, align 8, !dbg !1482
  %mul23 = fmul contract double 0x3D10000000000000, %25, !dbg !1483
  ret double %mul23, !dbg !1484
}

; Function Attrs: convergent noinline nounwind
define dso_local void @_Z13vranlc_deviceiPddS_(i32 %n, double* %x_seed, double %a, double* %y) #3 !dbg !1485 {
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
  call void @llvm.dbg.declare(metadata i32* %n.addr, metadata !1488, metadata !DIExpression()), !dbg !1489
  store double* %x_seed, double** %x_seed.addr, align 8
  call void @llvm.dbg.declare(metadata double** %x_seed.addr, metadata !1490, metadata !DIExpression()), !dbg !1491
  store double %a, double* %a.addr, align 8
  call void @llvm.dbg.declare(metadata double* %a.addr, metadata !1492, metadata !DIExpression()), !dbg !1493
  store double* %y, double** %y.addr, align 8
  call void @llvm.dbg.declare(metadata double** %y.addr, metadata !1494, metadata !DIExpression()), !dbg !1495
  call void @llvm.dbg.declare(metadata i32* %i, metadata !1496, metadata !DIExpression()), !dbg !1497
  call void @llvm.dbg.declare(metadata double* %x, metadata !1498, metadata !DIExpression()), !dbg !1499
  call void @llvm.dbg.declare(metadata double* %t1, metadata !1500, metadata !DIExpression()), !dbg !1501
  call void @llvm.dbg.declare(metadata double* %t2, metadata !1502, metadata !DIExpression()), !dbg !1503
  call void @llvm.dbg.declare(metadata double* %t3, metadata !1504, metadata !DIExpression()), !dbg !1505
  call void @llvm.dbg.declare(metadata double* %t4, metadata !1506, metadata !DIExpression()), !dbg !1507
  call void @llvm.dbg.declare(metadata double* %a1, metadata !1508, metadata !DIExpression()), !dbg !1509
  call void @llvm.dbg.declare(metadata double* %a2, metadata !1510, metadata !DIExpression()), !dbg !1511
  call void @llvm.dbg.declare(metadata double* %x1, metadata !1512, metadata !DIExpression()), !dbg !1513
  call void @llvm.dbg.declare(metadata double* %x2, metadata !1514, metadata !DIExpression()), !dbg !1515
  call void @llvm.dbg.declare(metadata double* %z, metadata !1516, metadata !DIExpression()), !dbg !1517
  %0 = load double, double* %a.addr, align 8, !dbg !1518
  %mul = fmul contract double 0x3E80000000000000, %0, !dbg !1519
  store double %mul, double* %t1, align 8, !dbg !1520
  %1 = load double, double* %t1, align 8, !dbg !1521
  %conv = fptosi double %1 to i32, !dbg !1521
  %conv1 = sitofp i32 %conv to double, !dbg !1522
  store double %conv1, double* %a1, align 8, !dbg !1523
  %2 = load double, double* %a.addr, align 8, !dbg !1524
  %3 = load double, double* %a1, align 8, !dbg !1525
  %mul2 = fmul contract double 0x4160000000000000, %3, !dbg !1526
  %sub = fsub contract double %2, %mul2, !dbg !1527
  store double %sub, double* %a2, align 8, !dbg !1528
  %4 = load double*, double** %x_seed.addr, align 8, !dbg !1529
  %5 = load double, double* %4, align 8, !dbg !1530
  store double %5, double* %x, align 8, !dbg !1531
  store i32 0, i32* %i, align 4, !dbg !1532
  br label %for.cond, !dbg !1534

for.cond:                                         ; preds = %for.inc, %entry
  %6 = load i32, i32* %i, align 4, !dbg !1535
  %7 = load i32, i32* %n.addr, align 4, !dbg !1537
  %cmp = icmp slt i32 %6, %7, !dbg !1538
  br i1 %cmp, label %for.body, label %for.end, !dbg !1539

for.body:                                         ; preds = %for.cond
  %8 = load double, double* %x, align 8, !dbg !1540
  %mul3 = fmul contract double 0x3E80000000000000, %8, !dbg !1542
  store double %mul3, double* %t1, align 8, !dbg !1543
  %9 = load double, double* %t1, align 8, !dbg !1544
  %conv4 = fptosi double %9 to i32, !dbg !1544
  %conv5 = sitofp i32 %conv4 to double, !dbg !1545
  store double %conv5, double* %x1, align 8, !dbg !1546
  %10 = load double, double* %x, align 8, !dbg !1547
  %11 = load double, double* %x1, align 8, !dbg !1548
  %mul6 = fmul contract double 0x4160000000000000, %11, !dbg !1549
  %sub7 = fsub contract double %10, %mul6, !dbg !1550
  store double %sub7, double* %x2, align 8, !dbg !1551
  %12 = load double, double* %a1, align 8, !dbg !1552
  %13 = load double, double* %x2, align 8, !dbg !1553
  %mul8 = fmul contract double %12, %13, !dbg !1554
  %14 = load double, double* %a2, align 8, !dbg !1555
  %15 = load double, double* %x1, align 8, !dbg !1556
  %mul9 = fmul contract double %14, %15, !dbg !1557
  %add = fadd contract double %mul8, %mul9, !dbg !1558
  store double %add, double* %t1, align 8, !dbg !1559
  %16 = load double, double* %t1, align 8, !dbg !1560
  %mul10 = fmul contract double 0x3E80000000000000, %16, !dbg !1561
  %conv11 = fptosi double %mul10 to i32, !dbg !1562
  %conv12 = sitofp i32 %conv11 to double, !dbg !1563
  store double %conv12, double* %t2, align 8, !dbg !1564
  %17 = load double, double* %t1, align 8, !dbg !1565
  %18 = load double, double* %t2, align 8, !dbg !1566
  %mul13 = fmul contract double 0x4160000000000000, %18, !dbg !1567
  %sub14 = fsub contract double %17, %mul13, !dbg !1568
  store double %sub14, double* %z, align 8, !dbg !1569
  %19 = load double, double* %z, align 8, !dbg !1570
  %mul15 = fmul contract double 0x4160000000000000, %19, !dbg !1571
  %20 = load double, double* %a2, align 8, !dbg !1572
  %21 = load double, double* %x2, align 8, !dbg !1573
  %mul16 = fmul contract double %20, %21, !dbg !1574
  %add17 = fadd contract double %mul15, %mul16, !dbg !1575
  store double %add17, double* %t3, align 8, !dbg !1576
  %22 = load double, double* %t3, align 8, !dbg !1577
  %mul18 = fmul contract double 0x3D10000000000000, %22, !dbg !1578
  %conv19 = fptosi double %mul18 to i32, !dbg !1579
  %conv20 = sitofp i32 %conv19 to double, !dbg !1580
  store double %conv20, double* %t4, align 8, !dbg !1581
  %23 = load double, double* %t3, align 8, !dbg !1582
  %24 = load double, double* %t4, align 8, !dbg !1583
  %mul21 = fmul contract double 0x42D0000000000000, %24, !dbg !1584
  %sub22 = fsub contract double %23, %mul21, !dbg !1585
  store double %sub22, double* %x, align 8, !dbg !1586
  %25 = load double, double* %x, align 8, !dbg !1587
  %mul23 = fmul contract double 0x3D10000000000000, %25, !dbg !1588
  %26 = load double*, double** %y.addr, align 8, !dbg !1589
  %27 = load i32, i32* %i, align 4, !dbg !1590
  %idxprom = sext i32 %27 to i64, !dbg !1589
  %arrayidx = getelementptr inbounds double, double* %26, i64 %idxprom, !dbg !1589
  store double %mul23, double* %arrayidx, align 8, !dbg !1591
  br label %for.inc, !dbg !1592

for.inc:                                          ; preds = %for.body
  %28 = load i32, i32* %i, align 4, !dbg !1593
  %inc = add nsw i32 %28, 1, !dbg !1593
  store i32 %inc, i32* %i, align 4, !dbg !1593
  br label %for.cond, !dbg !1594, !llvm.loop !1595

for.end:                                          ; preds = %for.cond
  %29 = load double, double* %x, align 8, !dbg !1597
  %30 = load double*, double** %x_seed.addr, align 8, !dbg !1598
  store double %29, double* %30, align 8, !dbg !1599
  ret void, !dbg !1600
}

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.d2i.hi(double) #2

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.d2i.lo(double) #2

; Function Attrs: nounwind readnone
declare double @llvm.nvvm.lohi.i2d(i32, i32) #2

; Function Attrs: nounwind readnone
declare double @llvm.nvvm.fma.rn.d(double, double, double) #2

; Function Attrs: nounwind readnone
declare double @llvm.nvvm.fabs.d(double) #2

; Function Attrs: nounwind readnone
declare double @llvm.nvvm.sqrt.rn.d(double) #2

; Function Attrs: noinline uwtable
define internal void @_GLOBAL__sub_I_ep.cu() #4 section ".text.startup" !dbg !1601 {
entry:
  call void @__cxx_global_var_init(), !dbg !1603
  ret void
}

; Function Attrs: noinline uwtable
define internal void @__cxx_global_var_init() #4 section ".text.startup" !dbg !1604 {
entry:
  %call = call noalias i8* @malloc(i64 80) #10, !dbg !1605
  %0 = bitcast i8* %call to double*, !dbg !1606
  store double* %0, double** @_ZL1q, align 8, !dbg !1606
  ret void, !dbg !1607
}

; Function Attrs: nounwind
declare dso_local noalias i8* @malloc(i64) #5

; Function Attrs: noinline nounwind uwtable
define dso_local double @_Z6randlcPdd(double* %x, double %a) #6 !dbg !1608 {
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
  call void @llvm.dbg.declare(metadata double** %x.addr, metadata !1609, metadata !DIExpression()), !dbg !1610
  store double %a, double* %a.addr, align 8
  call void @llvm.dbg.declare(metadata double* %a.addr, metadata !1611, metadata !DIExpression()), !dbg !1612
  call void @llvm.dbg.declare(metadata double* %t1, metadata !1613, metadata !DIExpression()), !dbg !1614
  call void @llvm.dbg.declare(metadata double* %t2, metadata !1615, metadata !DIExpression()), !dbg !1616
  call void @llvm.dbg.declare(metadata double* %t3, metadata !1617, metadata !DIExpression()), !dbg !1618
  call void @llvm.dbg.declare(metadata double* %t4, metadata !1619, metadata !DIExpression()), !dbg !1620
  call void @llvm.dbg.declare(metadata double* %a1, metadata !1621, metadata !DIExpression()), !dbg !1622
  call void @llvm.dbg.declare(metadata double* %a2, metadata !1623, metadata !DIExpression()), !dbg !1624
  call void @llvm.dbg.declare(metadata double* %x1, metadata !1625, metadata !DIExpression()), !dbg !1626
  call void @llvm.dbg.declare(metadata double* %x2, metadata !1627, metadata !DIExpression()), !dbg !1628
  call void @llvm.dbg.declare(metadata double* %z, metadata !1629, metadata !DIExpression()), !dbg !1630
  %0 = load double, double* %a.addr, align 8, !dbg !1631
  %mul = fmul contract double 0x3E80000000000000, %0, !dbg !1632
  store double %mul, double* %t1, align 8, !dbg !1633
  %1 = load double, double* %t1, align 8, !dbg !1634
  %conv = fptosi double %1 to i32, !dbg !1634
  %conv1 = sitofp i32 %conv to double, !dbg !1635
  store double %conv1, double* %a1, align 8, !dbg !1636
  %2 = load double, double* %a.addr, align 8, !dbg !1637
  %3 = load double, double* %a1, align 8, !dbg !1638
  %mul2 = fmul contract double 0x4160000000000000, %3, !dbg !1639
  %sub = fsub contract double %2, %mul2, !dbg !1640
  store double %sub, double* %a2, align 8, !dbg !1641
  %4 = load double*, double** %x.addr, align 8, !dbg !1642
  %5 = load double, double* %4, align 8, !dbg !1643
  %mul3 = fmul contract double 0x3E80000000000000, %5, !dbg !1644
  store double %mul3, double* %t1, align 8, !dbg !1645
  %6 = load double, double* %t1, align 8, !dbg !1646
  %conv4 = fptosi double %6 to i32, !dbg !1646
  %conv5 = sitofp i32 %conv4 to double, !dbg !1647
  store double %conv5, double* %x1, align 8, !dbg !1648
  %7 = load double*, double** %x.addr, align 8, !dbg !1649
  %8 = load double, double* %7, align 8, !dbg !1650
  %9 = load double, double* %x1, align 8, !dbg !1651
  %mul6 = fmul contract double 0x4160000000000000, %9, !dbg !1652
  %sub7 = fsub contract double %8, %mul6, !dbg !1653
  store double %sub7, double* %x2, align 8, !dbg !1654
  %10 = load double, double* %a1, align 8, !dbg !1655
  %11 = load double, double* %x2, align 8, !dbg !1656
  %mul8 = fmul contract double %10, %11, !dbg !1657
  %12 = load double, double* %a2, align 8, !dbg !1658
  %13 = load double, double* %x1, align 8, !dbg !1659
  %mul9 = fmul contract double %12, %13, !dbg !1660
  %add = fadd contract double %mul8, %mul9, !dbg !1661
  store double %add, double* %t1, align 8, !dbg !1662
  %14 = load double, double* %t1, align 8, !dbg !1663
  %mul10 = fmul contract double 0x3E80000000000000, %14, !dbg !1664
  %conv11 = fptosi double %mul10 to i32, !dbg !1665
  %conv12 = sitofp i32 %conv11 to double, !dbg !1666
  store double %conv12, double* %t2, align 8, !dbg !1667
  %15 = load double, double* %t1, align 8, !dbg !1668
  %16 = load double, double* %t2, align 8, !dbg !1669
  %mul13 = fmul contract double 0x4160000000000000, %16, !dbg !1670
  %sub14 = fsub contract double %15, %mul13, !dbg !1671
  store double %sub14, double* %z, align 8, !dbg !1672
  %17 = load double, double* %z, align 8, !dbg !1673
  %mul15 = fmul contract double 0x4160000000000000, %17, !dbg !1674
  %18 = load double, double* %a2, align 8, !dbg !1675
  %19 = load double, double* %x2, align 8, !dbg !1676
  %mul16 = fmul contract double %18, %19, !dbg !1677
  %add17 = fadd contract double %mul15, %mul16, !dbg !1678
  store double %add17, double* %t3, align 8, !dbg !1679
  %20 = load double, double* %t3, align 8, !dbg !1680
  %mul18 = fmul contract double 0x3D10000000000000, %20, !dbg !1681
  %conv19 = fptosi double %mul18 to i32, !dbg !1682
  %conv20 = sitofp i32 %conv19 to double, !dbg !1683
  store double %conv20, double* %t4, align 8, !dbg !1684
  %21 = load double, double* %t3, align 8, !dbg !1685
  %22 = load double, double* %t4, align 8, !dbg !1686
  %mul21 = fmul contract double 0x42D0000000000000, %22, !dbg !1687
  %sub22 = fsub contract double %21, %mul21, !dbg !1688
  %23 = load double*, double** %x.addr, align 8, !dbg !1689
  store double %sub22, double* %23, align 8, !dbg !1690
  %24 = load double*, double** %x.addr, align 8, !dbg !1691
  %25 = load double, double* %24, align 8, !dbg !1692
  %mul23 = fmul contract double 0x3D10000000000000, %25, !dbg !1693
  ret double %mul23, !dbg !1694
}

; Function Attrs: noinline uwtable
define dso_local void @_Z15c_print_resultsPcciiiiddS_iS_S_S_S_S_S_S_S_S_S_S_S_S_S_(i8* %name, i8 signext %class_npb, i32 %n1, i32 %n2, i32 %n3, i32 %niter, double %t, double %mops, i8* %optype, i32 %passed_verification, i8* %npbversion, i8* %compiletime, i8* %compilerversion, i8* %libversion, i8* %cpu_device, i8* %gpu_device, i8* %gpu_config, i8* %cc, i8* %clink, i8* %c_lib, i8* %c_inc, i8* %cflags, i8* %clinkflags, i8* %rand) #4 !dbg !1695 {
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
  call void @llvm.dbg.declare(metadata i8** %name.addr, metadata !1698, metadata !DIExpression()), !dbg !1699
  store i8 %class_npb, i8* %class_npb.addr, align 1
  call void @llvm.dbg.declare(metadata i8* %class_npb.addr, metadata !1700, metadata !DIExpression()), !dbg !1701
  store i32 %n1, i32* %n1.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %n1.addr, metadata !1702, metadata !DIExpression()), !dbg !1703
  store i32 %n2, i32* %n2.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %n2.addr, metadata !1704, metadata !DIExpression()), !dbg !1705
  store i32 %n3, i32* %n3.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %n3.addr, metadata !1706, metadata !DIExpression()), !dbg !1707
  store i32 %niter, i32* %niter.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %niter.addr, metadata !1708, metadata !DIExpression()), !dbg !1709
  store double %t, double* %t.addr, align 8
  call void @llvm.dbg.declare(metadata double* %t.addr, metadata !1710, metadata !DIExpression()), !dbg !1711
  store double %mops, double* %mops.addr, align 8
  call void @llvm.dbg.declare(metadata double* %mops.addr, metadata !1712, metadata !DIExpression()), !dbg !1713
  store i8* %optype, i8** %optype.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %optype.addr, metadata !1714, metadata !DIExpression()), !dbg !1715
  store i32 %passed_verification, i32* %passed_verification.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %passed_verification.addr, metadata !1716, metadata !DIExpression()), !dbg !1717
  store i8* %npbversion, i8** %npbversion.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %npbversion.addr, metadata !1718, metadata !DIExpression()), !dbg !1719
  store i8* %compiletime, i8** %compiletime.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %compiletime.addr, metadata !1720, metadata !DIExpression()), !dbg !1721
  store i8* %compilerversion, i8** %compilerversion.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %compilerversion.addr, metadata !1722, metadata !DIExpression()), !dbg !1723
  store i8* %libversion, i8** %libversion.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %libversion.addr, metadata !1724, metadata !DIExpression()), !dbg !1725
  store i8* %cpu_device, i8** %cpu_device.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %cpu_device.addr, metadata !1726, metadata !DIExpression()), !dbg !1727
  store i8* %gpu_device, i8** %gpu_device.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %gpu_device.addr, metadata !1728, metadata !DIExpression()), !dbg !1729
  store i8* %gpu_config, i8** %gpu_config.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %gpu_config.addr, metadata !1730, metadata !DIExpression()), !dbg !1731
  store i8* %cc, i8** %cc.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %cc.addr, metadata !1732, metadata !DIExpression()), !dbg !1733
  store i8* %clink, i8** %clink.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %clink.addr, metadata !1734, metadata !DIExpression()), !dbg !1735
  store i8* %c_lib, i8** %c_lib.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %c_lib.addr, metadata !1736, metadata !DIExpression()), !dbg !1737
  store i8* %c_inc, i8** %c_inc.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %c_inc.addr, metadata !1738, metadata !DIExpression()), !dbg !1739
  store i8* %cflags, i8** %cflags.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %cflags.addr, metadata !1740, metadata !DIExpression()), !dbg !1741
  store i8* %clinkflags, i8** %clinkflags.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %clinkflags.addr, metadata !1742, metadata !DIExpression()), !dbg !1743
  store i8* %rand, i8** %rand.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %rand.addr, metadata !1744, metadata !DIExpression()), !dbg !1745
  %0 = load i8*, i8** %name.addr, align 8, !dbg !1746
  %call = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([27 x i8], [27 x i8]* @.str, i64 0, i64 0), i8* %0), !dbg !1747
  %1 = load i8, i8* %class_npb.addr, align 1, !dbg !1748
  %conv = sext i8 %1 to i32, !dbg !1748
  %call1 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([46 x i8], [46 x i8]* @.str.1, i64 0, i64 0), i32 %conv), !dbg !1749
  %2 = load i8*, i8** %name.addr, align 8, !dbg !1750
  %arrayidx = getelementptr inbounds i8, i8* %2, i64 0, !dbg !1750
  %3 = load i8, i8* %arrayidx, align 1, !dbg !1750
  %conv2 = sext i8 %3 to i32, !dbg !1750
  %cmp = icmp eq i32 %conv2, 73, !dbg !1752
  br i1 %cmp, label %land.lhs.true, label %if.else15, !dbg !1753

land.lhs.true:                                    ; preds = %entry
  %4 = load i8*, i8** %name.addr, align 8, !dbg !1754
  %arrayidx3 = getelementptr inbounds i8, i8* %4, i64 1, !dbg !1754
  %5 = load i8, i8* %arrayidx3, align 1, !dbg !1754
  %conv4 = sext i8 %5 to i32, !dbg !1754
  %cmp5 = icmp eq i32 %conv4, 83, !dbg !1755
  br i1 %cmp5, label %if.then, label %if.else15, !dbg !1756

if.then:                                          ; preds = %land.lhs.true
  %6 = load i32, i32* %n3.addr, align 4, !dbg !1757
  %cmp6 = icmp eq i32 %6, 0, !dbg !1760
  br i1 %cmp6, label %if.then7, label %if.else, !dbg !1761

if.then7:                                         ; preds = %if.then
  call void @llvm.dbg.declare(metadata i64* %nn, metadata !1762, metadata !DIExpression()), !dbg !1764
  %7 = load i32, i32* %n1.addr, align 4, !dbg !1765
  %conv8 = sext i32 %7 to i64, !dbg !1765
  store i64 %conv8, i64* %nn, align 8, !dbg !1764
  %8 = load i32, i32* %n2.addr, align 4, !dbg !1766
  %cmp9 = icmp ne i32 %8, 0, !dbg !1768
  br i1 %cmp9, label %if.then10, label %if.end, !dbg !1769

if.then10:                                        ; preds = %if.then7
  %9 = load i32, i32* %n2.addr, align 4, !dbg !1770
  %conv11 = sext i32 %9 to i64, !dbg !1770
  %10 = load i64, i64* %nn, align 8, !dbg !1772
  %mul = mul nsw i64 %10, %conv11, !dbg !1772
  store i64 %mul, i64* %nn, align 8, !dbg !1772
  br label %if.end, !dbg !1773

if.end:                                           ; preds = %if.then10, %if.then7
  %11 = load i64, i64* %nn, align 8, !dbg !1774
  %call12 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([38 x i8], [38 x i8]* @.str.2, i64 0, i64 0), i64 %11), !dbg !1775
  br label %if.end14, !dbg !1776

if.else:                                          ; preds = %if.then
  %12 = load i32, i32* %n1.addr, align 4, !dbg !1777
  %13 = load i32, i32* %n2.addr, align 4, !dbg !1779
  %14 = load i32, i32* %n3.addr, align 4, !dbg !1780
  %call13 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([44 x i8], [44 x i8]* @.str.3, i64 0, i64 0), i32 %12, i32 %13, i32 %14), !dbg !1781
  br label %if.end14

if.end14:                                         ; preds = %if.else, %if.end
  br label %if.end48, !dbg !1782

if.else15:                                        ; preds = %land.lhs.true, %entry
  call void @llvm.dbg.declare(metadata [16 x i8]* %size, metadata !1783, metadata !DIExpression()), !dbg !1788
  call void @llvm.dbg.declare(metadata i32* %j, metadata !1789, metadata !DIExpression()), !dbg !1790
  %15 = load i32, i32* %n2.addr, align 4, !dbg !1791
  %cmp16 = icmp eq i32 %15, 0, !dbg !1793
  br i1 %cmp16, label %land.lhs.true17, label %if.else45, !dbg !1794

land.lhs.true17:                                  ; preds = %if.else15
  %16 = load i32, i32* %n3.addr, align 4, !dbg !1795
  %cmp18 = icmp eq i32 %16, 0, !dbg !1796
  br i1 %cmp18, label %if.then19, label %if.else45, !dbg !1797

if.then19:                                        ; preds = %land.lhs.true17
  %17 = load i8*, i8** %name.addr, align 8, !dbg !1798
  %arrayidx20 = getelementptr inbounds i8, i8* %17, i64 0, !dbg !1798
  %18 = load i8, i8* %arrayidx20, align 1, !dbg !1798
  %conv21 = sext i8 %18 to i32, !dbg !1798
  %cmp22 = icmp eq i32 %conv21, 69, !dbg !1801
  br i1 %cmp22, label %land.lhs.true23, label %if.else42, !dbg !1802

land.lhs.true23:                                  ; preds = %if.then19
  %19 = load i8*, i8** %name.addr, align 8, !dbg !1803
  %arrayidx24 = getelementptr inbounds i8, i8* %19, i64 1, !dbg !1803
  %20 = load i8, i8* %arrayidx24, align 1, !dbg !1803
  %conv25 = sext i8 %20 to i32, !dbg !1803
  %cmp26 = icmp eq i32 %conv25, 80, !dbg !1804
  br i1 %cmp26, label %if.then27, label %if.else42, !dbg !1805

if.then27:                                        ; preds = %land.lhs.true23
  %arraydecay = getelementptr inbounds [16 x i8], [16 x i8]* %size, i64 0, i64 0, !dbg !1806
  %21 = load i32, i32* %n1.addr, align 4, !dbg !1808
  %conv28 = sitofp i32 %21 to double, !dbg !1808
  %call29 = call double @pow(double 2.000000e+00, double %conv28) #10, !dbg !1809
  %call30 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay, i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.4, i64 0, i64 0), double %call29) #10, !dbg !1810
  store i32 14, i32* %j, align 4, !dbg !1811
  %22 = load i32, i32* %j, align 4, !dbg !1812
  %idxprom = sext i32 %22 to i64, !dbg !1814
  %arrayidx31 = getelementptr inbounds [16 x i8], [16 x i8]* %size, i64 0, i64 %idxprom, !dbg !1814
  %23 = load i8, i8* %arrayidx31, align 1, !dbg !1814
  %conv32 = sext i8 %23 to i32, !dbg !1814
  %cmp33 = icmp eq i32 %conv32, 46, !dbg !1815
  br i1 %cmp33, label %if.then34, label %if.end37, !dbg !1816

if.then34:                                        ; preds = %if.then27
  %24 = load i32, i32* %j, align 4, !dbg !1817
  %idxprom35 = sext i32 %24 to i64, !dbg !1819
  %arrayidx36 = getelementptr inbounds [16 x i8], [16 x i8]* %size, i64 0, i64 %idxprom35, !dbg !1819
  store i8 32, i8* %arrayidx36, align 1, !dbg !1820
  %25 = load i32, i32* %j, align 4, !dbg !1821
  %dec = add nsw i32 %25, -1, !dbg !1821
  store i32 %dec, i32* %j, align 4, !dbg !1821
  br label %if.end37, !dbg !1822

if.end37:                                         ; preds = %if.then34, %if.then27
  %26 = load i32, i32* %j, align 4, !dbg !1823
  %add = add nsw i32 %26, 1, !dbg !1824
  %idxprom38 = sext i32 %add to i64, !dbg !1825
  %arrayidx39 = getelementptr inbounds [16 x i8], [16 x i8]* %size, i64 0, i64 %idxprom38, !dbg !1825
  store i8 0, i8* %arrayidx39, align 1, !dbg !1826
  %arraydecay40 = getelementptr inbounds [16 x i8], [16 x i8]* %size, i64 0, i64 0, !dbg !1827
  %call41 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([34 x i8], [34 x i8]* @.str.5, i64 0, i64 0), i8* %arraydecay40), !dbg !1828
  br label %if.end44, !dbg !1829

if.else42:                                        ; preds = %land.lhs.true23, %if.then19
  %27 = load i32, i32* %n1.addr, align 4, !dbg !1830
  %call43 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.6, i64 0, i64 0), i32 %27), !dbg !1832
  br label %if.end44

if.end44:                                         ; preds = %if.else42, %if.end37
  br label %if.end47, !dbg !1833

if.else45:                                        ; preds = %land.lhs.true17, %if.else15
  %28 = load i32, i32* %n1.addr, align 4, !dbg !1834
  %29 = load i32, i32* %n2.addr, align 4, !dbg !1836
  %30 = load i32, i32* %n3.addr, align 4, !dbg !1837
  %call46 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([42 x i8], [42 x i8]* @.str.7, i64 0, i64 0), i32 %28, i32 %29, i32 %30), !dbg !1838
  br label %if.end47

if.end47:                                         ; preds = %if.else45, %if.end44
  br label %if.end48

if.end48:                                         ; preds = %if.end47, %if.end14
  %31 = load i32, i32* %niter.addr, align 4, !dbg !1839
  %call49 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.8, i64 0, i64 0), i32 %31), !dbg !1840
  %32 = load double, double* %t.addr, align 8, !dbg !1841
  %call50 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([39 x i8], [39 x i8]* @.str.9, i64 0, i64 0), double %32), !dbg !1842
  %33 = load double, double* %mops.addr, align 8, !dbg !1843
  %call51 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([39 x i8], [39 x i8]* @.str.10, i64 0, i64 0), double %33), !dbg !1844
  %34 = load i8*, i8** %optype.addr, align 8, !dbg !1845
  %call52 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([25 x i8], [25 x i8]* @.str.11, i64 0, i64 0), i8* %34), !dbg !1846
  %35 = load i32, i32* %passed_verification.addr, align 4, !dbg !1847
  %cmp53 = icmp slt i32 %35, 0, !dbg !1849
  br i1 %cmp53, label %if.then54, label %if.else56, !dbg !1850

if.then54:                                        ; preds = %if.end48
  %call55 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([45 x i8], [45 x i8]* @.str.12, i64 0, i64 0)), !dbg !1851
  br label %if.end62, !dbg !1853

if.else56:                                        ; preds = %if.end48
  %36 = load i32, i32* %passed_verification.addr, align 4, !dbg !1854
  %tobool = icmp ne i32 %36, 0, !dbg !1854
  br i1 %tobool, label %if.then57, label %if.else59, !dbg !1856

if.then57:                                        ; preds = %if.else56
  %call58 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([45 x i8], [45 x i8]* @.str.13, i64 0, i64 0)), !dbg !1857
  br label %if.end61, !dbg !1859

if.else59:                                        ; preds = %if.else56
  %call60 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([45 x i8], [45 x i8]* @.str.14, i64 0, i64 0)), !dbg !1860
  br label %if.end61

if.end61:                                         ; preds = %if.else59, %if.then57
  br label %if.end62

if.end62:                                         ; preds = %if.end61, %if.then54
  %37 = load i8*, i8** %npbversion.addr, align 8, !dbg !1862
  %call63 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.15, i64 0, i64 0), i8* %37), !dbg !1863
  %38 = load i8*, i8** %compiletime.addr, align 8, !dbg !1864
  %call64 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.16, i64 0, i64 0), i8* %38), !dbg !1865
  %39 = load i8*, i8** %compilerversion.addr, align 8, !dbg !1866
  %call65 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.17, i64 0, i64 0), i8* %39), !dbg !1867
  %40 = load i8*, i8** %libversion.addr, align 8, !dbg !1868
  %call66 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.18, i64 0, i64 0), i8* %40), !dbg !1869
  %call67 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([20 x i8], [20 x i8]* @.str.19, i64 0, i64 0)), !dbg !1870
  %41 = load i8*, i8** %cc.addr, align 8, !dbg !1871
  %call68 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.20, i64 0, i64 0), i8* %41), !dbg !1872
  %42 = load i8*, i8** %clink.addr, align 8, !dbg !1873
  %call69 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.21, i64 0, i64 0), i8* %42), !dbg !1874
  %43 = load i8*, i8** %c_lib.addr, align 8, !dbg !1875
  %call70 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.22, i64 0, i64 0), i8* %43), !dbg !1876
  %44 = load i8*, i8** %c_inc.addr, align 8, !dbg !1877
  %call71 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.23, i64 0, i64 0), i8* %44), !dbg !1878
  %45 = load i8*, i8** %cflags.addr, align 8, !dbg !1879
  %call72 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.24, i64 0, i64 0), i8* %45), !dbg !1880
  %46 = load i8*, i8** %clinkflags.addr, align 8, !dbg !1881
  %call73 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.25, i64 0, i64 0), i8* %46), !dbg !1882
  %47 = load i8*, i8** %rand.addr, align 8, !dbg !1883
  %call74 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.26, i64 0, i64 0), i8* %47), !dbg !1884
  %call75 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.27, i64 0, i64 0)), !dbg !1885
  %48 = load i8*, i8** %cpu_device.addr, align 8, !dbg !1886
  %call76 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.28, i64 0, i64 0), i8* %48), !dbg !1887
  %49 = load i8*, i8** %gpu_device.addr, align 8, !dbg !1888
  %call77 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.29, i64 0, i64 0), i8* %49), !dbg !1889
  %call78 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.30, i64 0, i64 0)), !dbg !1890
  %50 = load i8*, i8** %gpu_config.addr, align 8, !dbg !1891
  %call79 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.31, i64 0, i64 0), i8* %50), !dbg !1892
  %call80 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.32, i64 0, i64 0)), !dbg !1893
  %call81 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([72 x i8], [72 x i8]* @.str.33, i64 0, i64 0)), !dbg !1894
  %call82 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([27 x i8], [27 x i8]* @.str.34, i64 0, i64 0)), !dbg !1895
  %call83 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([56 x i8], [56 x i8]* @.str.35, i64 0, i64 0)), !dbg !1896
  %call84 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([52 x i8], [52 x i8]* @.str.36, i64 0, i64 0)), !dbg !1897
  %call85 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([46 x i8], [46 x i8]* @.str.37, i64 0, i64 0)), !dbg !1898
  %call86 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.32, i64 0, i64 0)), !dbg !1899
  %call87 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([43 x i8], [43 x i8]* @.str.38, i64 0, i64 0)), !dbg !1900
  %call88 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([72 x i8], [72 x i8]* @.str.33, i64 0, i64 0)), !dbg !1901
  %call89 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.32, i64 0, i64 0)), !dbg !1902
  ret void, !dbg !1903
}

declare dso_local i32 @printf(i8*, ...) #7

; Function Attrs: nounwind
declare dso_local double @pow(double, double) #5

; Function Attrs: nounwind
declare dso_local i32 @sprintf(i8*, i8*, ...) #5

; Function Attrs: noinline norecurse uwtable
define dso_local i32 @main(i32 %argc, i8** %argv) #8 !dbg !1904 {
entry:
  %retval = alloca i32, align 4
  %argc.addr = alloca i32, align 4
  %argv.addr = alloca i8**, align 8
  %Mops = alloca double, align 8
  %t1 = alloca double, align 8
  %sx = alloca double, align 8
  %sy = alloca double, align 8
  %tm = alloca double, align 8
  %an = alloca double, align 8
  %gc = alloca double, align 8
  %sx_verify_value = alloca double, align 8
  %sy_verify_value = alloca double, align 8
  %sx_err = alloca double, align 8
  %sy_err = alloca double, align 8
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  %nit = alloca i32, align 4
  %block = alloca i32, align 4
  %verified = alloca i32, align 4
  %size = alloca [16 x i8], align 16
  %agg.tmp = alloca %struct.dim3, align 4
  %agg.tmp17 = alloca %struct.dim3, align 4
  %agg.tmp.coerce = alloca { i64, i32 }, align 4
  %agg.tmp17.coerce = alloca { i64, i32 }, align 4
  %gpu_config = alloca [256 x i8], align 16
  %gpu_config_string = alloca [2048 x i8], align 16
  store i32 0, i32* %retval, align 4
  store i32 %argc, i32* %argc.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %argc.addr, metadata !1907, metadata !DIExpression()), !dbg !1908
  store i8** %argv, i8*** %argv.addr, align 8
  call void @llvm.dbg.declare(metadata i8*** %argv.addr, metadata !1909, metadata !DIExpression()), !dbg !1910
  call void @llvm.dbg.declare(metadata double* %Mops, metadata !1911, metadata !DIExpression()), !dbg !1912
  call void @llvm.dbg.declare(metadata double* %t1, metadata !1913, metadata !DIExpression()), !dbg !1914
  call void @llvm.dbg.declare(metadata double* %sx, metadata !1915, metadata !DIExpression()), !dbg !1916
  call void @llvm.dbg.declare(metadata double* %sy, metadata !1917, metadata !DIExpression()), !dbg !1918
  call void @llvm.dbg.declare(metadata double* %tm, metadata !1919, metadata !DIExpression()), !dbg !1920
  call void @llvm.dbg.declare(metadata double* %an, metadata !1921, metadata !DIExpression()), !dbg !1922
  call void @llvm.dbg.declare(metadata double* %gc, metadata !1923, metadata !DIExpression()), !dbg !1924
  call void @llvm.dbg.declare(metadata double* %sx_verify_value, metadata !1925, metadata !DIExpression()), !dbg !1926
  call void @llvm.dbg.declare(metadata double* %sy_verify_value, metadata !1927, metadata !DIExpression()), !dbg !1928
  call void @llvm.dbg.declare(metadata double* %sx_err, metadata !1929, metadata !DIExpression()), !dbg !1930
  call void @llvm.dbg.declare(metadata double* %sy_err, metadata !1931, metadata !DIExpression()), !dbg !1932
  call void @llvm.dbg.declare(metadata i32* %i, metadata !1933, metadata !DIExpression()), !dbg !1934
  call void @llvm.dbg.declare(metadata i32* %j, metadata !1935, metadata !DIExpression()), !dbg !1936
  call void @llvm.dbg.declare(metadata i32* %nit, metadata !1937, metadata !DIExpression()), !dbg !1938
  call void @llvm.dbg.declare(metadata i32* %block, metadata !1939, metadata !DIExpression()), !dbg !1940
  call void @llvm.dbg.declare(metadata i32* %verified, metadata !1941, metadata !DIExpression()), !dbg !1944
  call void @llvm.dbg.declare(metadata [16 x i8]* %size, metadata !1945, metadata !DIExpression()), !dbg !1946
  %arraydecay = getelementptr inbounds [16 x i8], [16 x i8]* %size, i64 0, i64 0, !dbg !1947
  %call = call double @pow(double 2.000000e+00, double 2.900000e+01) #10, !dbg !1948
  %call1 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay, i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.39, i64 0, i64 0), double %call) #10, !dbg !1949
  store i32 14, i32* %j, align 4, !dbg !1950
  %0 = load i32, i32* %j, align 4, !dbg !1951
  %idxprom = sext i32 %0 to i64, !dbg !1953
  %arrayidx = getelementptr inbounds [16 x i8], [16 x i8]* %size, i64 0, i64 %idxprom, !dbg !1953
  %1 = load i8, i8* %arrayidx, align 1, !dbg !1953
  %conv = sext i8 %1 to i32, !dbg !1953
  %cmp = icmp eq i32 %conv, 46, !dbg !1954
  br i1 %cmp, label %if.then, label %if.end, !dbg !1955

if.then:                                          ; preds = %entry
  %2 = load i32, i32* %j, align 4, !dbg !1956
  %dec = add nsw i32 %2, -1, !dbg !1956
  store i32 %dec, i32* %j, align 4, !dbg !1956
  br label %if.end, !dbg !1958

if.end:                                           ; preds = %if.then, %entry
  %3 = load i32, i32* %j, align 4, !dbg !1959
  %add = add nsw i32 %3, 1, !dbg !1960
  %idxprom2 = sext i32 %add to i64, !dbg !1961
  %arrayidx3 = getelementptr inbounds [16 x i8], [16 x i8]* %size, i64 0, i64 %idxprom2, !dbg !1961
  store i8 0, i8* %arrayidx3, align 1, !dbg !1962
  %call4 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([65 x i8], [65 x i8]* @.str.40, i64 0, i64 0)), !dbg !1963
  %arraydecay5 = getelementptr inbounds [16 x i8], [16 x i8]* %size, i64 0, i64 0, !dbg !1964
  %call6 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([43 x i8], [43 x i8]* @.str.41, i64 0, i64 0), i8* %arraydecay5), !dbg !1965
  store i32 0, i32* %verified, align 4, !dbg !1966
  store double 0x41D2309CE5400000, double* %t1, align 8, !dbg !1967
  store i32 0, i32* %i, align 4, !dbg !1968
  br label %for.cond, !dbg !1970

for.cond:                                         ; preds = %for.inc, %if.end
  %4 = load i32, i32* %i, align 4, !dbg !1971
  %cmp7 = icmp slt i32 %4, 17, !dbg !1973
  br i1 %cmp7, label %for.body, label %for.end, !dbg !1974

for.body:                                         ; preds = %for.cond
  %5 = load double, double* %t1, align 8, !dbg !1975
  %call8 = call double @_Z6randlcPdd(double* %t1, double %5), !dbg !1977
  br label %for.inc, !dbg !1978

for.inc:                                          ; preds = %for.body
  %6 = load i32, i32* %i, align 4, !dbg !1979
  %inc = add nsw i32 %6, 1, !dbg !1979
  store i32 %inc, i32* %i, align 4, !dbg !1979
  br label %for.cond, !dbg !1980, !llvm.loop !1981

for.end:                                          ; preds = %for.cond
  %7 = load double, double* %t1, align 8, !dbg !1983
  store double %7, double* %an, align 8, !dbg !1984
  store double 0.000000e+00, double* %gc, align 8, !dbg !1985
  store double 0.000000e+00, double* %sx, align 8, !dbg !1986
  store double 0.000000e+00, double* %sy, align 8, !dbg !1987
  store i32 0, i32* %i, align 4, !dbg !1988
  br label %for.cond9, !dbg !1990

for.cond9:                                        ; preds = %for.inc14, %for.end
  %8 = load i32, i32* %i, align 4, !dbg !1991
  %cmp10 = icmp slt i32 %8, 10, !dbg !1993
  br i1 %cmp10, label %for.body11, label %for.end16, !dbg !1994

for.body11:                                       ; preds = %for.cond9
  %9 = load double*, double** @_ZL1q, align 8, !dbg !1995
  %10 = load i32, i32* %i, align 4, !dbg !1997
  %idxprom12 = sext i32 %10 to i64, !dbg !1995
  %arrayidx13 = getelementptr inbounds double, double* %9, i64 %idxprom12, !dbg !1995
  store double 0.000000e+00, double* %arrayidx13, align 8, !dbg !1998
  br label %for.inc14, !dbg !1999

for.inc14:                                        ; preds = %for.body11
  %11 = load i32, i32* %i, align 4, !dbg !2000
  %inc15 = add nsw i32 %11, 1, !dbg !2000
  store i32 %inc15, i32* %i, align 4, !dbg !2000
  br label %for.cond9, !dbg !2001, !llvm.loop !2002

for.end16:                                        ; preds = %for.cond9
  call void @_ZL9setup_gpuv(), !dbg !2004
  %12 = load i32, i32* @blocks_per_grid, align 4, !dbg !2005
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp, i32 %12, i32 1, i32 1), !dbg !2005
  %13 = load i32, i32* @threads_per_block, align 4, !dbg !2006
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp17, i32 %13, i32 1, i32 1), !dbg !2006
  %14 = bitcast { i64, i32 }* %agg.tmp.coerce to i8*, !dbg !2007
  %15 = bitcast %struct.dim3* %agg.tmp to i8*, !dbg !2007
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %14, i8* align 4 %15, i64 12, i1 false), !dbg !2007
  %16 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 0, !dbg !2007
  %17 = load i64, i64* %16, align 4, !dbg !2007
  %18 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 1, !dbg !2007
  %19 = load i32, i32* %18, align 4, !dbg !2007
  %20 = bitcast { i64, i32 }* %agg.tmp17.coerce to i8*, !dbg !2007
  %21 = bitcast %struct.dim3* %agg.tmp17 to i8*, !dbg !2007
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %20, i8* align 4 %21, i64 12, i1 false), !dbg !2007
  %22 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp17.coerce, i32 0, i32 0, !dbg !2007
  %23 = load i64, i64* %22, align 4, !dbg !2007
  %24 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp17.coerce, i32 0, i32 1, !dbg !2007
  %25 = load i32, i32* %24, align 4, !dbg !2007
  %call18 = call i32 @cudaConfigureCall(i64 %17, i32 %19, i64 %23, i32 %25, i64 0, %struct.CUstream_st* null), !dbg !2007
  %tobool = icmp ne i32 %call18, 0, !dbg !2007
  br i1 %tobool, label %kcall.end, label %kcall.configok, !dbg !2008

kcall.configok:                                   ; preds = %for.end16
  %26 = load double*, double** @q_device, align 8, !dbg !2009
  %27 = load double*, double** @sx_device, align 8, !dbg !2010
  %28 = load double*, double** @sy_device, align 8, !dbg !2011
  %29 = load double, double* %an, align 8, !dbg !2012
  call void @ep.ll_CudaFE__Z10gpu_kernelPdS_S_d(double* %26, double* %27, double* %28, double %29), !dbg !2008
  br label %kcall.end, !dbg !2008

kcall.end:                                        ; preds = %kcall.configok, %for.end16
  %30 = load double*, double** @q_host, align 8, !dbg !2013
  %31 = bitcast double* %30 to i8*, !dbg !2013
  %32 = load double*, double** @q_device, align 8, !dbg !2014
  %33 = bitcast double* %32 to i8*, !dbg !2014
  %34 = load i64, i64* @size_q, align 8, !dbg !2015
  %call19 = call i32 @cudaMemcpy(i8* %31, i8* %33, i64 %34, i32 2), !dbg !2016
  %35 = load double*, double** @sx_host, align 8, !dbg !2017
  %36 = bitcast double* %35 to i8*, !dbg !2017
  %37 = load double*, double** @sx_device, align 8, !dbg !2018
  %38 = bitcast double* %37 to i8*, !dbg !2018
  %39 = load i64, i64* @size_sx, align 8, !dbg !2019
  %call20 = call i32 @cudaMemcpy(i8* %36, i8* %38, i64 %39, i32 2), !dbg !2020
  %40 = load double*, double** @sy_host, align 8, !dbg !2021
  %41 = bitcast double* %40 to i8*, !dbg !2021
  %42 = load double*, double** @sy_device, align 8, !dbg !2022
  %43 = bitcast double* %42 to i8*, !dbg !2022
  %44 = load i64, i64* @size_sy, align 8, !dbg !2023
  %call21 = call i32 @cudaMemcpy(i8* %41, i8* %43, i64 %44, i32 2), !dbg !2024
  store i32 0, i32* %block, align 4, !dbg !2025
  br label %for.cond22, !dbg !2027

for.cond22:                                       ; preds = %for.inc43, %kcall.end
  %45 = load i32, i32* %block, align 4, !dbg !2028
  %46 = load i32, i32* @blocks_per_grid, align 4, !dbg !2030
  %cmp23 = icmp slt i32 %45, %46, !dbg !2031
  br i1 %cmp23, label %for.body24, label %for.end45, !dbg !2032

for.body24:                                       ; preds = %for.cond22
  store i32 0, i32* %i, align 4, !dbg !2033
  br label %for.cond25, !dbg !2036

for.cond25:                                       ; preds = %for.inc34, %for.body24
  %47 = load i32, i32* %i, align 4, !dbg !2037
  %cmp26 = icmp slt i32 %47, 10, !dbg !2039
  br i1 %cmp26, label %for.body27, label %for.end36, !dbg !2040

for.body27:                                       ; preds = %for.cond25
  %48 = load double*, double** @q_host, align 8, !dbg !2041
  %49 = load i32, i32* %block, align 4, !dbg !2043
  %mul = mul nsw i32 %49, 10, !dbg !2044
  %50 = load i32, i32* %i, align 4, !dbg !2045
  %add28 = add nsw i32 %mul, %50, !dbg !2046
  %idxprom29 = sext i32 %add28 to i64, !dbg !2041
  %arrayidx30 = getelementptr inbounds double, double* %48, i64 %idxprom29, !dbg !2041
  %51 = load double, double* %arrayidx30, align 8, !dbg !2041
  %52 = load double*, double** @_ZL1q, align 8, !dbg !2047
  %53 = load i32, i32* %i, align 4, !dbg !2048
  %idxprom31 = sext i32 %53 to i64, !dbg !2047
  %arrayidx32 = getelementptr inbounds double, double* %52, i64 %idxprom31, !dbg !2047
  %54 = load double, double* %arrayidx32, align 8, !dbg !2049
  %add33 = fadd contract double %54, %51, !dbg !2049
  store double %add33, double* %arrayidx32, align 8, !dbg !2049
  br label %for.inc34, !dbg !2050

for.inc34:                                        ; preds = %for.body27
  %55 = load i32, i32* %i, align 4, !dbg !2051
  %inc35 = add nsw i32 %55, 1, !dbg !2051
  store i32 %inc35, i32* %i, align 4, !dbg !2051
  br label %for.cond25, !dbg !2052, !llvm.loop !2053

for.end36:                                        ; preds = %for.cond25
  %56 = load double*, double** @sx_host, align 8, !dbg !2055
  %57 = load i32, i32* %block, align 4, !dbg !2056
  %idxprom37 = sext i32 %57 to i64, !dbg !2055
  %arrayidx38 = getelementptr inbounds double, double* %56, i64 %idxprom37, !dbg !2055
  %58 = load double, double* %arrayidx38, align 8, !dbg !2055
  %59 = load double, double* %sx, align 8, !dbg !2057
  %add39 = fadd contract double %59, %58, !dbg !2057
  store double %add39, double* %sx, align 8, !dbg !2057
  %60 = load double*, double** @sy_host, align 8, !dbg !2058
  %61 = load i32, i32* %block, align 4, !dbg !2059
  %idxprom40 = sext i32 %61 to i64, !dbg !2058
  %arrayidx41 = getelementptr inbounds double, double* %60, i64 %idxprom40, !dbg !2058
  %62 = load double, double* %arrayidx41, align 8, !dbg !2058
  %63 = load double, double* %sy, align 8, !dbg !2060
  %add42 = fadd contract double %63, %62, !dbg !2060
  store double %add42, double* %sy, align 8, !dbg !2060
  br label %for.inc43, !dbg !2061

for.inc43:                                        ; preds = %for.end36
  %64 = load i32, i32* %block, align 4, !dbg !2062
  %inc44 = add nsw i32 %64, 1, !dbg !2062
  store i32 %inc44, i32* %block, align 4, !dbg !2062
  br label %for.cond22, !dbg !2063, !llvm.loop !2064

for.end45:                                        ; preds = %for.cond22
  store i32 0, i32* %i, align 4, !dbg !2066
  br label %for.cond46, !dbg !2068

for.cond46:                                       ; preds = %for.inc52, %for.end45
  %65 = load i32, i32* %i, align 4, !dbg !2069
  %cmp47 = icmp slt i32 %65, 10, !dbg !2071
  br i1 %cmp47, label %for.body48, label %for.end54, !dbg !2072

for.body48:                                       ; preds = %for.cond46
  %66 = load double*, double** @_ZL1q, align 8, !dbg !2073
  %67 = load i32, i32* %i, align 4, !dbg !2075
  %idxprom49 = sext i32 %67 to i64, !dbg !2073
  %arrayidx50 = getelementptr inbounds double, double* %66, i64 %idxprom49, !dbg !2073
  %68 = load double, double* %arrayidx50, align 8, !dbg !2073
  %69 = load double, double* %gc, align 8, !dbg !2076
  %add51 = fadd contract double %69, %68, !dbg !2076
  store double %add51, double* %gc, align 8, !dbg !2076
  br label %for.inc52, !dbg !2077

for.inc52:                                        ; preds = %for.body48
  %70 = load i32, i32* %i, align 4, !dbg !2078
  %inc53 = add nsw i32 %70, 1, !dbg !2078
  store i32 %inc53, i32* %i, align 4, !dbg !2078
  br label %for.cond46, !dbg !2079, !llvm.loop !2080

for.end54:                                        ; preds = %for.cond46
  store i32 0, i32* %nit, align 4, !dbg !2082
  store i32 1, i32* %verified, align 4, !dbg !2083
  store double 0xC0B0C7E00ADACEF8, double* %sx_verify_value, align 8, !dbg !2084
  store double 0xC0CEDFA9B1BE31DC, double* %sy_verify_value, align 8, !dbg !2089
  %71 = load i32, i32* %verified, align 4, !dbg !2090
  %tobool55 = icmp ne i32 %71, 0, !dbg !2090
  br i1 %tobool55, label %if.then56, label %if.end62, !dbg !2092

if.then56:                                        ; preds = %for.end54
  %72 = load double, double* %sx, align 8, !dbg !2093
  %73 = load double, double* %sx_verify_value, align 8, !dbg !2095
  %sub = fsub contract double %72, %73, !dbg !2096
  %74 = load double, double* %sx_verify_value, align 8, !dbg !2097
  %div = fdiv double %sub, %74, !dbg !2098
  %75 = call double @llvm.fabs.f64(double %div), !dbg !2099
  store double %75, double* %sx_err, align 8, !dbg !2100
  %76 = load double, double* %sy, align 8, !dbg !2101
  %77 = load double, double* %sy_verify_value, align 8, !dbg !2102
  %sub57 = fsub contract double %76, %77, !dbg !2103
  %78 = load double, double* %sy_verify_value, align 8, !dbg !2104
  %div58 = fdiv double %sub57, %78, !dbg !2105
  %79 = call double @llvm.fabs.f64(double %div58), !dbg !2106
  store double %79, double* %sy_err, align 8, !dbg !2107
  %80 = load double, double* %sx_err, align 8, !dbg !2108
  %cmp59 = fcmp ole double %80, 1.000000e-08, !dbg !2109
  br i1 %cmp59, label %land.rhs, label %land.end, !dbg !2110

land.rhs:                                         ; preds = %if.then56
  %81 = load double, double* %sy_err, align 8, !dbg !2111
  %cmp60 = fcmp ole double %81, 1.000000e-08, !dbg !2112
  br label %land.end

land.end:                                         ; preds = %land.rhs, %if.then56
  %82 = phi i1 [ false, %if.then56 ], [ %cmp60, %land.rhs ], !dbg !2113
  %conv61 = zext i1 %82 to i32, !dbg !2114
  store i32 %conv61, i32* %verified, align 4, !dbg !2115
  br label %if.end62, !dbg !2116

if.end62:                                         ; preds = %land.end, %for.end54
  %call63 = call double @pow(double 2.000000e+00, double 2.900000e+01) #10, !dbg !2117
  %83 = load double, double* %tm, align 8, !dbg !2118
  %div64 = fdiv double %call63, %83, !dbg !2119
  %div65 = fdiv double %div64, 1.000000e+06, !dbg !2120
  store double %div65, double* %Mops, align 8, !dbg !2121
  %call66 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([26 x i8], [26 x i8]* @.str.42, i64 0, i64 0)), !dbg !2122
  %84 = load double, double* %tm, align 8, !dbg !2123
  %call67 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([19 x i8], [19 x i8]* @.str.43, i64 0, i64 0), double %84), !dbg !2124
  %call68 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([12 x i8], [12 x i8]* @.str.44, i64 0, i64 0), i32 28), !dbg !2125
  %85 = load double, double* %gc, align 8, !dbg !2126
  %call69 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([30 x i8], [30 x i8]* @.str.45, i64 0, i64 0), double %85), !dbg !2127
  %86 = load double, double* %sx, align 8, !dbg !2128
  %87 = load double, double* %sy, align 8, !dbg !2129
  %call70 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([25 x i8], [25 x i8]* @.str.46, i64 0, i64 0), double %86, double %87), !dbg !2130
  %call71 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.47, i64 0, i64 0)), !dbg !2131
  store i32 0, i32* %i, align 4, !dbg !2132
  br label %for.cond72, !dbg !2134

for.cond72:                                       ; preds = %for.inc78, %if.end62
  %88 = load i32, i32* %i, align 4, !dbg !2135
  %cmp73 = icmp slt i32 %88, 10, !dbg !2137
  br i1 %cmp73, label %for.body74, label %for.end80, !dbg !2138

for.body74:                                       ; preds = %for.cond72
  %89 = load i32, i32* %i, align 4, !dbg !2139
  %90 = load double*, double** @_ZL1q, align 8, !dbg !2141
  %91 = load i32, i32* %i, align 4, !dbg !2142
  %idxprom75 = sext i32 %91 to i64, !dbg !2141
  %arrayidx76 = getelementptr inbounds double, double* %90, i64 %idxprom75, !dbg !2141
  %92 = load double, double* %arrayidx76, align 8, !dbg !2141
  %call77 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.48, i64 0, i64 0), i32 %89, double %92), !dbg !2143
  br label %for.inc78, !dbg !2144

for.inc78:                                        ; preds = %for.body74
  %93 = load i32, i32* %i, align 4, !dbg !2145
  %inc79 = add nsw i32 %93, 1, !dbg !2145
  store i32 %inc79, i32* %i, align 4, !dbg !2145
  br label %for.cond72, !dbg !2146, !llvm.loop !2147

for.end80:                                        ; preds = %for.cond72
  call void @llvm.dbg.declare(metadata [256 x i8]* %gpu_config, metadata !2149, metadata !DIExpression()), !dbg !2150
  call void @llvm.dbg.declare(metadata [2048 x i8]* %gpu_config_string, metadata !2151, metadata !DIExpression()), !dbg !2155
  %arraydecay81 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !2156
  %call82 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay81, i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.49, i64 0, i64 0), i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.50, i64 0, i64 0), i8* getelementptr inbounds ([18 x i8], [18 x i8]* @.str.51, i64 0, i64 0)) #10, !dbg !2157
  %arraydecay83 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !2158
  %arraydecay84 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !2159
  %call85 = call i8* @strcpy(i8* %arraydecay83, i8* %arraydecay84) #10, !dbg !2160
  %arraydecay86 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !2161
  %94 = load i32, i32* @threads_per_block, align 4, !dbg !2162
  %call87 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay86, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.52, i64 0, i64 0), i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.53, i64 0, i64 0), i32 %94) #10, !dbg !2163
  %arraydecay88 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !2164
  %arraydecay89 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !2165
  %call90 = call i8* @strcat(i8* %arraydecay88, i8* %arraydecay89) #10, !dbg !2166
  %95 = load i32, i32* %nit, align 4, !dbg !2167
  %96 = load double, double* %tm, align 8, !dbg !2168
  %97 = load double, double* %Mops, align 8, !dbg !2169
  %98 = load i32, i32* %verified, align 4, !dbg !2170
  %arraydecay91 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !2171
  call void @_Z15c_print_resultsPcciiiiddS_iS_S_S_S_S_S_S_S_S_S_S_S_S_S_(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.54, i64 0, i64 0), i8 signext 65, i32 29, i32 0, i32 0, i32 %95, double %96, double %97, i8* getelementptr inbounds ([25 x i8], [25 x i8]* @.str.55, i64 0, i64 0), i32 %98, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.56, i64 0, i64 0), i8* getelementptr inbounds ([12 x i8], [12 x i8]* @.str.57, i64 0, i64 0), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.58, i64 0, i64 0), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.58, i64 0, i64 0), i8* getelementptr inbounds ([42 x i8], [42 x i8]* @.str.59, i64 0, i64 0), i8* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 0, i64 0), i8* %arraydecay91, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.60, i64 0, i64 0), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.61, i64 0, i64 0), i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.62, i64 0, i64 0), i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.63, i64 0, i64 0), i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.64, i64 0, i64 0), i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.64, i64 0, i64 0), i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.65, i64 0, i64 0)), !dbg !2172
  call void @_ZL11release_gpuv(), !dbg !2173
  ret i32 0, !dbg !2174
}

; Function Attrs: noinline uwtable
define internal void @_ZL9setup_gpuv() #4 !dbg !2175 {
entry:
  %0 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !2176
  %cmp = icmp sle i32 32, %0, !dbg !2178
  br i1 %cmp, label %if.then, label %if.else, !dbg !2179

if.then:                                          ; preds = %entry
  store i32 32, i32* @threads_per_block, align 4, !dbg !2180
  br label %if.end, !dbg !2182

if.else:                                          ; preds = %entry
  %1 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !2183
  store i32 %1, i32* @threads_per_block, align 4, !dbg !2185
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %2 = load i32, i32* @threads_per_block, align 4, !dbg !2186
  %conv = sitofp i32 %2 to double, !dbg !2186
  %div = fdiv double 4.096000e+03, %conv, !dbg !2187
  %3 = call double @llvm.ceil.f64(double %div), !dbg !2188
  %conv1 = fptosi double %3 to i32, !dbg !2189
  store i32 %conv1, i32* @blocks_per_grid, align 4, !dbg !2190
  %4 = load i32, i32* @blocks_per_grid, align 4, !dbg !2191
  %mul = mul nsw i32 %4, 10, !dbg !2192
  %conv2 = sext i32 %mul to i64, !dbg !2191
  %mul3 = mul i64 %conv2, 8, !dbg !2193
  store i64 %mul3, i64* @size_q, align 8, !dbg !2194
  %5 = load i32, i32* @blocks_per_grid, align 4, !dbg !2195
  %conv4 = sext i32 %5 to i64, !dbg !2195
  %mul5 = mul i64 %conv4, 8, !dbg !2196
  store i64 %mul5, i64* @size_sx, align 8, !dbg !2197
  %6 = load i32, i32* @blocks_per_grid, align 4, !dbg !2198
  %conv6 = sext i32 %6 to i64, !dbg !2198
  %mul7 = mul i64 %conv6, 8, !dbg !2199
  store i64 %mul7, i64* @size_sy, align 8, !dbg !2200
  %7 = load i64, i64* @size_q, align 8, !dbg !2201
  %call = call noalias i8* @malloc(i64 %7) #10, !dbg !2202
  %8 = bitcast i8* %call to double*, !dbg !2203
  store double* %8, double** @q_host, align 8, !dbg !2204
  %9 = load i64, i64* @size_sx, align 8, !dbg !2205
  %call8 = call noalias i8* @malloc(i64 %9) #10, !dbg !2206
  %10 = bitcast i8* %call8 to double*, !dbg !2207
  store double* %10, double** @sx_host, align 8, !dbg !2208
  %11 = load i64, i64* @size_sy, align 8, !dbg !2209
  %call9 = call noalias i8* @malloc(i64 %11) #10, !dbg !2210
  %12 = bitcast i8* %call9 to double*, !dbg !2211
  store double* %12, double** @sy_host, align 8, !dbg !2212
  %13 = load i64, i64* @size_q, align 8, !dbg !2213
  %call10 = call i32 @_ZL10cudaMallocIdE9cudaErrorPPT_m(double** @q_device, i64 %13), !dbg !2214
  %14 = load i64, i64* @size_sx, align 8, !dbg !2215
  %call11 = call i32 @_ZL10cudaMallocIdE9cudaErrorPPT_m(double** @sx_device, i64 %14), !dbg !2216
  %15 = load i64, i64* @size_sy, align 8, !dbg !2217
  %call12 = call i32 @_ZL10cudaMallocIdE9cudaErrorPPT_m(double** @sy_device, i64 %15), !dbg !2218
  ret void, !dbg !2219
}

; Function Attrs: noinline nounwind uwtable
define linkonce_odr dso_local void @_ZN4dim3C2Ejjj(%struct.dim3* %this, i32 %vx, i32 %vy, i32 %vz) unnamed_addr #6 comdat align 2 !dbg !2220 {
entry:
  %this.addr = alloca %struct.dim3*, align 8
  %vx.addr = alloca i32, align 4
  %vy.addr = alloca i32, align 4
  %vz.addr = alloca i32, align 4
  store %struct.dim3* %this, %struct.dim3** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dim3** %this.addr, metadata !2221, metadata !DIExpression()), !dbg !2223
  store i32 %vx, i32* %vx.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %vx.addr, metadata !2224, metadata !DIExpression()), !dbg !2225
  store i32 %vy, i32* %vy.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %vy.addr, metadata !2226, metadata !DIExpression()), !dbg !2227
  store i32 %vz, i32* %vz.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %vz.addr, metadata !2228, metadata !DIExpression()), !dbg !2229
  %this1 = load %struct.dim3*, %struct.dim3** %this.addr, align 8
  %x = getelementptr inbounds %struct.dim3, %struct.dim3* %this1, i32 0, i32 0, !dbg !2230
  %0 = load i32, i32* %vx.addr, align 4, !dbg !2231
  store i32 %0, i32* %x, align 4, !dbg !2230
  %y = getelementptr inbounds %struct.dim3, %struct.dim3* %this1, i32 0, i32 1, !dbg !2232
  %1 = load i32, i32* %vy.addr, align 4, !dbg !2233
  store i32 %1, i32* %y, align 4, !dbg !2232
  %z = getelementptr inbounds %struct.dim3, %struct.dim3* %this1, i32 0, i32 2, !dbg !2234
  %2 = load i32, i32* %vz.addr, align 4, !dbg !2235
  store i32 %2, i32* %z, align 4, !dbg !2234
  ret void, !dbg !2236
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1 immarg) #9

declare dso_local i32 @cudaConfigureCall(i64, i32, i64, i32, i64, %struct.CUstream_st*) #7

; Function Attrs: noinline uwtable
define dso_local void @ep.ll_CudaFE__Z10gpu_kernelPdS_S_d(double* %q_global, double* %sx_global, double* %sy_global, double %an) #4 !dbg !2237 {
entry:
  %q_global.addr = alloca double*, align 8
  %sx_global.addr = alloca double*, align 8
  %sy_global.addr = alloca double*, align 8
  %an.addr = alloca double, align 8
  store double* %q_global, double** %q_global.addr, align 8
  call void @llvm.dbg.declare(metadata double** %q_global.addr, metadata !2238, metadata !DIExpression()), !dbg !2239
  store double* %sx_global, double** %sx_global.addr, align 8
  call void @llvm.dbg.declare(metadata double** %sx_global.addr, metadata !2240, metadata !DIExpression()), !dbg !2241
  store double* %sy_global, double** %sy_global.addr, align 8
  call void @llvm.dbg.declare(metadata double** %sy_global.addr, metadata !2242, metadata !DIExpression()), !dbg !2243
  store double %an, double* %an.addr, align 8
  call void @llvm.dbg.declare(metadata double* %an.addr, metadata !2244, metadata !DIExpression()), !dbg !2245
  %0 = bitcast double** %q_global.addr to i8*, !dbg !2246
  %1 = call i32 @cudaSetupArgument(i8* %0, i64 8, i64 0), !dbg !2246
  %2 = icmp eq i32 %1, 0, !dbg !2246
  br i1 %2, label %setup.next, label %setup.end, !dbg !2246

setup.next:                                       ; preds = %entry
  %3 = bitcast double** %sx_global.addr to i8*, !dbg !2246
  %4 = call i32 @cudaSetupArgument(i8* %3, i64 8, i64 8), !dbg !2246
  %5 = icmp eq i32 %4, 0, !dbg !2246
  br i1 %5, label %setup.next1, label %setup.end, !dbg !2246

setup.next1:                                      ; preds = %setup.next
  %6 = bitcast double** %sy_global.addr to i8*, !dbg !2246
  %7 = call i32 @cudaSetupArgument(i8* %6, i64 8, i64 16), !dbg !2246
  %8 = icmp eq i32 %7, 0, !dbg !2246
  br i1 %8, label %setup.next2, label %setup.end, !dbg !2246

setup.next2:                                      ; preds = %setup.next1
  %9 = bitcast double* %an.addr to i8*, !dbg !2246
  %10 = call i32 @cudaSetupArgument(i8* %9, i64 8, i64 24), !dbg !2246
  %11 = icmp eq i32 %10, 0, !dbg !2246
  br i1 %11, label %setup.next3, label %setup.end, !dbg !2246

setup.next3:                                      ; preds = %setup.next2
  %12 = call i32 @cudaLaunch(i8* bitcast (void (double*, double*, double*, double)* @ep.ll_CudaFE__Z10gpu_kernelPdS_S_d to i8*)), !dbg !2246
  br label %setup.end, !dbg !2246

setup.end:                                        ; preds = %setup.next3, %setup.next2, %setup.next1, %setup.next, %entry
  ret void, !dbg !2247
}

declare dso_local i32 @cudaMemcpy(i8*, i8*, i64, i32) #7

; Function Attrs: nounwind readnone speculatable
declare double @llvm.fabs.f64(double) #1

; Function Attrs: nounwind
declare dso_local i8* @strcpy(i8*, i8*) #5

; Function Attrs: nounwind
declare dso_local i8* @strcat(i8*, i8*) #5

; Function Attrs: noinline uwtable
define internal void @_ZL11release_gpuv() #4 !dbg !2248 {
entry:
  %0 = load double*, double** @q_device, align 8, !dbg !2249
  %1 = bitcast double* %0 to i8*, !dbg !2249
  %call = call i32 @cudaFree(i8* %1), !dbg !2250
  %2 = load double*, double** @sx_device, align 8, !dbg !2251
  %3 = bitcast double* %2 to i8*, !dbg !2251
  %call1 = call i32 @cudaFree(i8* %3), !dbg !2252
  %4 = load double*, double** @sy_device, align 8, !dbg !2253
  %5 = bitcast double* %4 to i8*, !dbg !2253
  %call2 = call i32 @cudaFree(i8* %5), !dbg !2254
  ret void, !dbg !2255
}

declare dso_local i32 @cudaFree(i8*) #7

declare dso_local i32 @cudaSetupArgument(i8*, i64, i64)

declare dso_local i32 @cudaLaunch(i8*)

; Function Attrs: nounwind readnone speculatable
declare double @llvm.ceil.f64(double) #1

; Function Attrs: noinline uwtable
define internal i32 @_ZL10cudaMallocIdE9cudaErrorPPT_m(double** %devPtr, i64 %size) #4 !dbg !2256 {
entry:
  %devPtr.addr = alloca double**, align 8
  %size.addr = alloca i64, align 8
  store double** %devPtr, double*** %devPtr.addr, align 8
  call void @llvm.dbg.declare(metadata double*** %devPtr.addr, metadata !2264, metadata !DIExpression()), !dbg !2265
  store i64 %size, i64* %size.addr, align 8
  call void @llvm.dbg.declare(metadata i64* %size.addr, metadata !2266, metadata !DIExpression()), !dbg !2267
  %0 = load double**, double*** %devPtr.addr, align 8, !dbg !2268
  %1 = bitcast double** %0 to i8*, !dbg !2268
  %2 = bitcast i8* %1 to i8**, !dbg !2269
  %3 = load i64, i64* %size.addr, align 8, !dbg !2270
  %call = call i32 @cudaMalloc(i8** %2, i64 %3), !dbg !2271
  ret i32 %call, !dbg !2272
}

declare dso_local i32 @cudaMalloc(i8**, i64) #7

attributes #0 = { convergent noinline nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="sm_20" "target-features"="+ptx42,+sm_20" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { nounwind readnone }
attributes #3 = { convergent noinline nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="sm_20" "target-features"="+ptx42,+sm_20" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { noinline uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #7 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #8 = { noinline norecurse uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #9 = { argmemonly nounwind }
attributes #10 = { nounwind }
attributes #11 = { convergent nounwind }

!llvm.dbg.cu = !{!961, !2}
!nvvm.annotations = !{!1034, !1035, !1036, !1035, !1037, !1037, !1037, !1037, !1038, !1038, !1037}
!llvm.ident = !{!1039, !1039}
!nvvm.internalize.after.link = !{}
!nvvmir.version = !{!1040}
!llvm.module.flags = !{!1041, !1042, !1043, !1044, !1045}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "q_host", scope: !2, file: !3, line: 84, type: !97, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 9.0.0 (https://github.com/yebinchon/llvm-project/ a1efa594106d738d0b74c9e4e2b9b779eb8b7d25)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !96, globals: !104, imports: !209, nameTableKind: None)
!3 = !DIFile(filename: "ep.cu", directory: "/scratch/ah7226/NPB-GPU/CUDA/EP")
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
!96 = !{!97, !99, !100, !98, !102, !103}
!97 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !98, size: 64)
!98 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
!99 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!100 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !101, size: 64)
!101 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!102 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !103, size: 64)
!103 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!104 = !{!105, !0, !107, !109, !111, !113, !115, !117, !119, !121, !126, !128, !130, !132, !134}
!105 = !DIGlobalVariableExpression(var: !106, expr: !DIExpression())
!106 = distinct !DIGlobalVariable(name: "q", linkageName: "_ZL1q", scope: !2, file: !3, line: 81, type: !97, isLocal: true, isDefinition: true)
!107 = !DIGlobalVariableExpression(var: !108, expr: !DIExpression())
!108 = distinct !DIGlobalVariable(name: "q_device", scope: !2, file: !3, line: 85, type: !97, isLocal: false, isDefinition: true)
!109 = !DIGlobalVariableExpression(var: !110, expr: !DIExpression())
!110 = distinct !DIGlobalVariable(name: "sx_host", scope: !2, file: !3, line: 86, type: !97, isLocal: false, isDefinition: true)
!111 = !DIGlobalVariableExpression(var: !112, expr: !DIExpression())
!112 = distinct !DIGlobalVariable(name: "sx_device", scope: !2, file: !3, line: 87, type: !97, isLocal: false, isDefinition: true)
!113 = !DIGlobalVariableExpression(var: !114, expr: !DIExpression())
!114 = distinct !DIGlobalVariable(name: "sy_host", scope: !2, file: !3, line: 88, type: !97, isLocal: false, isDefinition: true)
!115 = !DIGlobalVariableExpression(var: !116, expr: !DIExpression())
!116 = distinct !DIGlobalVariable(name: "sy_device", scope: !2, file: !3, line: 89, type: !97, isLocal: false, isDefinition: true)
!117 = !DIGlobalVariableExpression(var: !118, expr: !DIExpression())
!118 = distinct !DIGlobalVariable(name: "threads_per_block", scope: !2, file: !3, line: 90, type: !99, isLocal: false, isDefinition: true)
!119 = !DIGlobalVariableExpression(var: !120, expr: !DIExpression())
!120 = distinct !DIGlobalVariable(name: "blocks_per_grid", scope: !2, file: !3, line: 91, type: !99, isLocal: false, isDefinition: true)
!121 = !DIGlobalVariableExpression(var: !122, expr: !DIExpression())
!122 = distinct !DIGlobalVariable(name: "size_q", scope: !2, file: !3, line: 92, type: !123, isLocal: false, isDefinition: true)
!123 = !DIDerivedType(tag: DW_TAG_typedef, name: "size_t", file: !124, line: 46, baseType: !125)
!124 = !DIFile(filename: "/u/NAS_SCRATCH/ah7226/tulip/llvm-install-tulip/lib/clang/9.0.0/include/stddef.h", directory: "")
!125 = !DIBasicType(name: "long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!126 = !DIGlobalVariableExpression(var: !127, expr: !DIExpression())
!127 = distinct !DIGlobalVariable(name: "size_sx", scope: !2, file: !3, line: 93, type: !123, isLocal: false, isDefinition: true)
!128 = !DIGlobalVariableExpression(var: !129, expr: !DIExpression())
!129 = distinct !DIGlobalVariable(name: "size_sy", scope: !2, file: !3, line: 94, type: !123, isLocal: false, isDefinition: true)
!130 = !DIGlobalVariableExpression(var: !131, expr: !DIExpression())
!131 = distinct !DIGlobalVariable(name: "gpu_device_id", scope: !2, file: !3, line: 95, type: !99, isLocal: false, isDefinition: true)
!132 = !DIGlobalVariableExpression(var: !133, expr: !DIExpression())
!133 = distinct !DIGlobalVariable(name: "total_devices", scope: !2, file: !3, line: 96, type: !99, isLocal: false, isDefinition: true)
!134 = !DIGlobalVariableExpression(var: !135, expr: !DIExpression())
!135 = distinct !DIGlobalVariable(name: "gpu_device_properties", scope: !2, file: !3, line: 97, type: !136, isLocal: false, isDefinition: true)
!136 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "cudaDeviceProp", file: !6, line: 1257, size: 5056, flags: DIFlagTypePassByValue, elements: !137, identifier: "_ZTS14cudaDeviceProp")
!137 = !{!138, !142, !143, !144, !145, !146, !147, !148, !152, !153, !154, !155, !156, !157, !158, !159, !160, !161, !162, !163, !164, !165, !166, !167, !168, !172, !173, !174, !175, !176, !177, !178, !179, !180, !181, !182, !183, !184, !185, !186, !187, !188, !189, !190, !191, !192, !193, !194, !195, !196, !197, !198, !199, !200, !201, !202, !203, !204, !205, !206, !207, !208}
!138 = !DIDerivedType(tag: DW_TAG_member, name: "name", scope: !136, file: !6, line: 1259, baseType: !139, size: 2048)
!139 = !DICompositeType(tag: DW_TAG_array_type, baseType: !101, size: 2048, elements: !140)
!140 = !{!141}
!141 = !DISubrange(count: 256)
!142 = !DIDerivedType(tag: DW_TAG_member, name: "totalGlobalMem", scope: !136, file: !6, line: 1260, baseType: !123, size: 64, offset: 2048)
!143 = !DIDerivedType(tag: DW_TAG_member, name: "sharedMemPerBlock", scope: !136, file: !6, line: 1261, baseType: !123, size: 64, offset: 2112)
!144 = !DIDerivedType(tag: DW_TAG_member, name: "regsPerBlock", scope: !136, file: !6, line: 1262, baseType: !99, size: 32, offset: 2176)
!145 = !DIDerivedType(tag: DW_TAG_member, name: "warpSize", scope: !136, file: !6, line: 1263, baseType: !99, size: 32, offset: 2208)
!146 = !DIDerivedType(tag: DW_TAG_member, name: "memPitch", scope: !136, file: !6, line: 1264, baseType: !123, size: 64, offset: 2240)
!147 = !DIDerivedType(tag: DW_TAG_member, name: "maxThreadsPerBlock", scope: !136, file: !6, line: 1265, baseType: !99, size: 32, offset: 2304)
!148 = !DIDerivedType(tag: DW_TAG_member, name: "maxThreadsDim", scope: !136, file: !6, line: 1266, baseType: !149, size: 96, offset: 2336)
!149 = !DICompositeType(tag: DW_TAG_array_type, baseType: !99, size: 96, elements: !150)
!150 = !{!151}
!151 = !DISubrange(count: 3)
!152 = !DIDerivedType(tag: DW_TAG_member, name: "maxGridSize", scope: !136, file: !6, line: 1267, baseType: !149, size: 96, offset: 2432)
!153 = !DIDerivedType(tag: DW_TAG_member, name: "clockRate", scope: !136, file: !6, line: 1268, baseType: !99, size: 32, offset: 2528)
!154 = !DIDerivedType(tag: DW_TAG_member, name: "totalConstMem", scope: !136, file: !6, line: 1269, baseType: !123, size: 64, offset: 2560)
!155 = !DIDerivedType(tag: DW_TAG_member, name: "major", scope: !136, file: !6, line: 1270, baseType: !99, size: 32, offset: 2624)
!156 = !DIDerivedType(tag: DW_TAG_member, name: "minor", scope: !136, file: !6, line: 1271, baseType: !99, size: 32, offset: 2656)
!157 = !DIDerivedType(tag: DW_TAG_member, name: "textureAlignment", scope: !136, file: !6, line: 1272, baseType: !123, size: 64, offset: 2688)
!158 = !DIDerivedType(tag: DW_TAG_member, name: "texturePitchAlignment", scope: !136, file: !6, line: 1273, baseType: !123, size: 64, offset: 2752)
!159 = !DIDerivedType(tag: DW_TAG_member, name: "deviceOverlap", scope: !136, file: !6, line: 1274, baseType: !99, size: 32, offset: 2816)
!160 = !DIDerivedType(tag: DW_TAG_member, name: "multiProcessorCount", scope: !136, file: !6, line: 1275, baseType: !99, size: 32, offset: 2848)
!161 = !DIDerivedType(tag: DW_TAG_member, name: "kernelExecTimeoutEnabled", scope: !136, file: !6, line: 1276, baseType: !99, size: 32, offset: 2880)
!162 = !DIDerivedType(tag: DW_TAG_member, name: "integrated", scope: !136, file: !6, line: 1277, baseType: !99, size: 32, offset: 2912)
!163 = !DIDerivedType(tag: DW_TAG_member, name: "canMapHostMemory", scope: !136, file: !6, line: 1278, baseType: !99, size: 32, offset: 2944)
!164 = !DIDerivedType(tag: DW_TAG_member, name: "computeMode", scope: !136, file: !6, line: 1279, baseType: !99, size: 32, offset: 2976)
!165 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture1D", scope: !136, file: !6, line: 1280, baseType: !99, size: 32, offset: 3008)
!166 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture1DMipmap", scope: !136, file: !6, line: 1281, baseType: !99, size: 32, offset: 3040)
!167 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture1DLinear", scope: !136, file: !6, line: 1282, baseType: !99, size: 32, offset: 3072)
!168 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture2D", scope: !136, file: !6, line: 1283, baseType: !169, size: 64, offset: 3104)
!169 = !DICompositeType(tag: DW_TAG_array_type, baseType: !99, size: 64, elements: !170)
!170 = !{!171}
!171 = !DISubrange(count: 2)
!172 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture2DMipmap", scope: !136, file: !6, line: 1284, baseType: !169, size: 64, offset: 3168)
!173 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture2DLinear", scope: !136, file: !6, line: 1285, baseType: !149, size: 96, offset: 3232)
!174 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture2DGather", scope: !136, file: !6, line: 1286, baseType: !169, size: 64, offset: 3328)
!175 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture3D", scope: !136, file: !6, line: 1287, baseType: !149, size: 96, offset: 3392)
!176 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture3DAlt", scope: !136, file: !6, line: 1288, baseType: !149, size: 96, offset: 3488)
!177 = !DIDerivedType(tag: DW_TAG_member, name: "maxTextureCubemap", scope: !136, file: !6, line: 1289, baseType: !99, size: 32, offset: 3584)
!178 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture1DLayered", scope: !136, file: !6, line: 1290, baseType: !169, size: 64, offset: 3616)
!179 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture2DLayered", scope: !136, file: !6, line: 1291, baseType: !149, size: 96, offset: 3680)
!180 = !DIDerivedType(tag: DW_TAG_member, name: "maxTextureCubemapLayered", scope: !136, file: !6, line: 1292, baseType: !169, size: 64, offset: 3776)
!181 = !DIDerivedType(tag: DW_TAG_member, name: "maxSurface1D", scope: !136, file: !6, line: 1293, baseType: !99, size: 32, offset: 3840)
!182 = !DIDerivedType(tag: DW_TAG_member, name: "maxSurface2D", scope: !136, file: !6, line: 1294, baseType: !169, size: 64, offset: 3872)
!183 = !DIDerivedType(tag: DW_TAG_member, name: "maxSurface3D", scope: !136, file: !6, line: 1295, baseType: !149, size: 96, offset: 3936)
!184 = !DIDerivedType(tag: DW_TAG_member, name: "maxSurface1DLayered", scope: !136, file: !6, line: 1296, baseType: !169, size: 64, offset: 4032)
!185 = !DIDerivedType(tag: DW_TAG_member, name: "maxSurface2DLayered", scope: !136, file: !6, line: 1297, baseType: !149, size: 96, offset: 4096)
!186 = !DIDerivedType(tag: DW_TAG_member, name: "maxSurfaceCubemap", scope: !136, file: !6, line: 1298, baseType: !99, size: 32, offset: 4192)
!187 = !DIDerivedType(tag: DW_TAG_member, name: "maxSurfaceCubemapLayered", scope: !136, file: !6, line: 1299, baseType: !169, size: 64, offset: 4224)
!188 = !DIDerivedType(tag: DW_TAG_member, name: "surfaceAlignment", scope: !136, file: !6, line: 1300, baseType: !123, size: 64, offset: 4288)
!189 = !DIDerivedType(tag: DW_TAG_member, name: "concurrentKernels", scope: !136, file: !6, line: 1301, baseType: !99, size: 32, offset: 4352)
!190 = !DIDerivedType(tag: DW_TAG_member, name: "ECCEnabled", scope: !136, file: !6, line: 1302, baseType: !99, size: 32, offset: 4384)
!191 = !DIDerivedType(tag: DW_TAG_member, name: "pciBusID", scope: !136, file: !6, line: 1303, baseType: !99, size: 32, offset: 4416)
!192 = !DIDerivedType(tag: DW_TAG_member, name: "pciDeviceID", scope: !136, file: !6, line: 1304, baseType: !99, size: 32, offset: 4448)
!193 = !DIDerivedType(tag: DW_TAG_member, name: "pciDomainID", scope: !136, file: !6, line: 1305, baseType: !99, size: 32, offset: 4480)
!194 = !DIDerivedType(tag: DW_TAG_member, name: "tccDriver", scope: !136, file: !6, line: 1306, baseType: !99, size: 32, offset: 4512)
!195 = !DIDerivedType(tag: DW_TAG_member, name: "asyncEngineCount", scope: !136, file: !6, line: 1307, baseType: !99, size: 32, offset: 4544)
!196 = !DIDerivedType(tag: DW_TAG_member, name: "unifiedAddressing", scope: !136, file: !6, line: 1308, baseType: !99, size: 32, offset: 4576)
!197 = !DIDerivedType(tag: DW_TAG_member, name: "memoryClockRate", scope: !136, file: !6, line: 1309, baseType: !99, size: 32, offset: 4608)
!198 = !DIDerivedType(tag: DW_TAG_member, name: "memoryBusWidth", scope: !136, file: !6, line: 1310, baseType: !99, size: 32, offset: 4640)
!199 = !DIDerivedType(tag: DW_TAG_member, name: "l2CacheSize", scope: !136, file: !6, line: 1311, baseType: !99, size: 32, offset: 4672)
!200 = !DIDerivedType(tag: DW_TAG_member, name: "maxThreadsPerMultiProcessor", scope: !136, file: !6, line: 1312, baseType: !99, size: 32, offset: 4704)
!201 = !DIDerivedType(tag: DW_TAG_member, name: "streamPrioritiesSupported", scope: !136, file: !6, line: 1313, baseType: !99, size: 32, offset: 4736)
!202 = !DIDerivedType(tag: DW_TAG_member, name: "globalL1CacheSupported", scope: !136, file: !6, line: 1314, baseType: !99, size: 32, offset: 4768)
!203 = !DIDerivedType(tag: DW_TAG_member, name: "localL1CacheSupported", scope: !136, file: !6, line: 1315, baseType: !99, size: 32, offset: 4800)
!204 = !DIDerivedType(tag: DW_TAG_member, name: "sharedMemPerMultiprocessor", scope: !136, file: !6, line: 1316, baseType: !123, size: 64, offset: 4864)
!205 = !DIDerivedType(tag: DW_TAG_member, name: "regsPerMultiprocessor", scope: !136, file: !6, line: 1317, baseType: !99, size: 32, offset: 4928)
!206 = !DIDerivedType(tag: DW_TAG_member, name: "managedMemory", scope: !136, file: !6, line: 1318, baseType: !99, size: 32, offset: 4960)
!207 = !DIDerivedType(tag: DW_TAG_member, name: "isMultiGpuBoard", scope: !136, file: !6, line: 1319, baseType: !99, size: 32, offset: 4992)
!208 = !DIDerivedType(tag: DW_TAG_member, name: "multiGpuBoardGroupID", scope: !136, file: !6, line: 1320, baseType: !99, size: 32, offset: 5024)
!209 = !{!210, !216, !221, !223, !225, !227, !229, !233, !235, !237, !239, !241, !243, !245, !247, !249, !251, !253, !255, !257, !259, !261, !265, !267, !269, !271, !275, !280, !282, !284, !289, !293, !295, !297, !299, !301, !303, !305, !307, !309, !314, !318, !320, !325, !329, !331, !333, !335, !337, !339, !343, !345, !347, !352, !358, !362, !364, !366, !368, !370, !374, !376, !378, !382, !384, !386, !388, !390, !392, !394, !396, !398, !400, !404, !410, !412, !414, !418, !420, !422, !424, !426, !428, !430, !432, !436, !440, !442, !444, !448, !450, !452, !454, !456, !458, !460, !464, !470, !474, !479, !481, !485, !489, !499, !503, !507, !511, !515, !519, !521, !525, !529, !533, !541, !545, !549, !553, !557, !561, !567, !571, !575, !577, !585, !589, !596, !598, !600, !604, !608, !612, !617, !621, !626, !627, !628, !629, !631, !632, !633, !634, !635, !636, !637, !639, !640, !641, !642, !643, !647, !648, !649, !650, !651, !652, !653, !654, !655, !656, !657, !658, !659, !660, !661, !662, !663, !664, !665, !666, !667, !668, !669, !670, !671, !675, !677, !679, !681, !683, !685, !687, !689, !692, !694, !696, !698, !700, !702, !704, !706, !708, !710, !712, !714, !716, !718, !720, !722, !724, !726, !728, !730, !732, !734, !736, !738, !740, !742, !744, !746, !748, !750, !752, !754, !756, !758, !760, !762, !764, !766, !768, !770, !772, !774, !776, !778, !780, !782, !784, !790, !796, !801, !805, !807, !809, !811, !813, !820, !824, !828, !832, !836, !840, !845, !849, !851, !855, !861, !865, !870, !872, !874, !878, !882, !886, !888, !890, !892, !894, !898, !900, !902, !906, !910, !914, !918, !922, !924, !926, !932, !936, !940, !944, !946, !948, !952, !956, !957, !958, !959, !960}
!210 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !212, file: !213, line: 223)
!211 = !DINamespace(name: "std", scope: null)
!212 = !DISubprogram(name: "abs", linkageName: "_ZL3absi", scope: !213, file: !213, line: 53, type: !214, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!213 = !DIFile(filename: "/u/NAS_SCRATCH/ah7226/tulip/llvm-install-tulip/lib/clang/9.0.0/include/__clang_cuda_math_forward_declares.h", directory: "")
!214 = !DISubroutineType(types: !215)
!215 = !{!99, !99}
!216 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !217, file: !213, line: 224)
!217 = !DISubprogram(name: "acos", linkageName: "_ZL4acosf", scope: !213, file: !213, line: 55, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!218 = !DISubroutineType(types: !219)
!219 = !{!220, !220}
!220 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!221 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !222, file: !213, line: 225)
!222 = !DISubprogram(name: "acosh", linkageName: "_ZL5acoshf", scope: !213, file: !213, line: 57, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!223 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !224, file: !213, line: 226)
!224 = !DISubprogram(name: "asin", linkageName: "_ZL4asinf", scope: !213, file: !213, line: 59, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!225 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !226, file: !213, line: 227)
!226 = !DISubprogram(name: "asinh", linkageName: "_ZL5asinhf", scope: !213, file: !213, line: 61, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!227 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !228, file: !213, line: 228)
!228 = !DISubprogram(name: "atan", linkageName: "_ZL4atanf", scope: !213, file: !213, line: 65, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!229 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !230, file: !213, line: 229)
!230 = !DISubprogram(name: "atan2", linkageName: "_ZL5atan2ff", scope: !213, file: !213, line: 63, type: !231, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!231 = !DISubroutineType(types: !232)
!232 = !{!220, !220, !220}
!233 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !234, file: !213, line: 230)
!234 = !DISubprogram(name: "atanh", linkageName: "_ZL5atanhf", scope: !213, file: !213, line: 67, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!235 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !236, file: !213, line: 231)
!236 = !DISubprogram(name: "cbrt", linkageName: "_ZL4cbrtf", scope: !213, file: !213, line: 69, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!237 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !238, file: !213, line: 232)
!238 = !DISubprogram(name: "ceil", linkageName: "_ZL4ceilf", scope: !213, file: !213, line: 71, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!239 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !240, file: !213, line: 233)
!240 = !DISubprogram(name: "copysign", linkageName: "_ZL8copysignff", scope: !213, file: !213, line: 73, type: !231, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!241 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !242, file: !213, line: 234)
!242 = !DISubprogram(name: "cos", linkageName: "_ZL3cosf", scope: !213, file: !213, line: 75, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!243 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !244, file: !213, line: 235)
!244 = !DISubprogram(name: "cosh", linkageName: "_ZL4coshf", scope: !213, file: !213, line: 77, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!245 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !246, file: !213, line: 236)
!246 = !DISubprogram(name: "erf", linkageName: "_ZL3erff", scope: !213, file: !213, line: 81, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!247 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !248, file: !213, line: 237)
!248 = !DISubprogram(name: "erfc", linkageName: "_ZL4erfcf", scope: !213, file: !213, line: 79, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!249 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !250, file: !213, line: 238)
!250 = !DISubprogram(name: "exp", linkageName: "_ZL3expf", scope: !213, file: !213, line: 85, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!251 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !252, file: !213, line: 239)
!252 = !DISubprogram(name: "exp2", linkageName: "_ZL4exp2f", scope: !213, file: !213, line: 83, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!253 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !254, file: !213, line: 240)
!254 = !DISubprogram(name: "expm1", linkageName: "_ZL5expm1f", scope: !213, file: !213, line: 87, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!255 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !256, file: !213, line: 241)
!256 = !DISubprogram(name: "fabs", linkageName: "_ZL4fabsf", scope: !213, file: !213, line: 89, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!257 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !258, file: !213, line: 242)
!258 = !DISubprogram(name: "fdim", linkageName: "_ZL4fdimff", scope: !213, file: !213, line: 91, type: !231, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!259 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !260, file: !213, line: 243)
!260 = !DISubprogram(name: "floor", linkageName: "_ZL5floorf", scope: !213, file: !213, line: 93, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!261 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !262, file: !213, line: 244)
!262 = !DISubprogram(name: "fma", linkageName: "_ZL3fmafff", scope: !213, file: !213, line: 95, type: !263, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!263 = !DISubroutineType(types: !264)
!264 = !{!220, !220, !220, !220}
!265 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !266, file: !213, line: 245)
!266 = !DISubprogram(name: "fmax", linkageName: "_ZL4fmaxff", scope: !213, file: !213, line: 97, type: !231, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!267 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !268, file: !213, line: 246)
!268 = !DISubprogram(name: "fmin", linkageName: "_ZL4fminff", scope: !213, file: !213, line: 99, type: !231, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!269 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !270, file: !213, line: 247)
!270 = !DISubprogram(name: "fmod", linkageName: "_ZL4fmodff", scope: !213, file: !213, line: 101, type: !231, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!271 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !272, file: !213, line: 248)
!272 = !DISubprogram(name: "fpclassify", linkageName: "_ZL10fpclassifyf", scope: !213, file: !213, line: 103, type: !273, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!273 = !DISubroutineType(types: !274)
!274 = !{!99, !220}
!275 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !276, file: !213, line: 249)
!276 = !DISubprogram(name: "frexp", linkageName: "_ZL5frexpfPi", scope: !213, file: !213, line: 105, type: !277, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!277 = !DISubroutineType(types: !278)
!278 = !{!220, !220, !279}
!279 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !99, size: 64)
!280 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !281, file: !213, line: 250)
!281 = !DISubprogram(name: "hypot", linkageName: "_ZL5hypotff", scope: !213, file: !213, line: 107, type: !231, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!282 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !283, file: !213, line: 251)
!283 = !DISubprogram(name: "ilogb", linkageName: "_ZL5ilogbf", scope: !213, file: !213, line: 109, type: !273, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!284 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !285, file: !213, line: 252)
!285 = !DISubprogram(name: "isfinite", linkageName: "_ZL8isfinitef", scope: !213, file: !213, line: 114, type: !286, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!286 = !DISubroutineType(types: !287)
!287 = !{!288, !220}
!288 = !DIBasicType(name: "bool", size: 8, encoding: DW_ATE_boolean)
!289 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !290, file: !213, line: 253)
!290 = !DISubprogram(name: "isgreater", linkageName: "_ZL9isgreaterff", scope: !213, file: !213, line: 118, type: !291, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!291 = !DISubroutineType(types: !292)
!292 = !{!288, !220, !220}
!293 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !294, file: !213, line: 254)
!294 = !DISubprogram(name: "isgreaterequal", linkageName: "_ZL14isgreaterequalff", scope: !213, file: !213, line: 117, type: !291, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!295 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !296, file: !213, line: 255)
!296 = !DISubprogram(name: "isinf", linkageName: "_ZL5isinff", scope: !213, file: !213, line: 123, type: !286, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!297 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !298, file: !213, line: 256)
!298 = !DISubprogram(name: "isless", linkageName: "_ZL6islessff", scope: !213, file: !213, line: 127, type: !291, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!299 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !300, file: !213, line: 257)
!300 = !DISubprogram(name: "islessequal", linkageName: "_ZL11islessequalff", scope: !213, file: !213, line: 126, type: !291, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!301 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !302, file: !213, line: 258)
!302 = !DISubprogram(name: "islessgreater", linkageName: "_ZL13islessgreaterff", scope: !213, file: !213, line: 129, type: !291, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!303 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !304, file: !213, line: 259)
!304 = !DISubprogram(name: "isnan", linkageName: "_ZL5isnanf", scope: !213, file: !213, line: 134, type: !286, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!305 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !306, file: !213, line: 260)
!306 = !DISubprogram(name: "isnormal", linkageName: "_ZL8isnormalf", scope: !213, file: !213, line: 136, type: !286, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!307 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !308, file: !213, line: 261)
!308 = !DISubprogram(name: "isunordered", linkageName: "_ZL11isunorderedff", scope: !213, file: !213, line: 138, type: !291, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!309 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !310, file: !213, line: 262)
!310 = !DISubprogram(name: "labs", linkageName: "_ZL4labsl", scope: !213, file: !213, line: 139, type: !311, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!311 = !DISubroutineType(types: !312)
!312 = !{!313, !313}
!313 = !DIBasicType(name: "long int", size: 64, encoding: DW_ATE_signed)
!314 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !315, file: !213, line: 263)
!315 = !DISubprogram(name: "ldexp", linkageName: "_ZL5ldexpfi", scope: !213, file: !213, line: 141, type: !316, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!316 = !DISubroutineType(types: !317)
!317 = !{!220, !220, !99}
!318 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !319, file: !213, line: 264)
!319 = !DISubprogram(name: "lgamma", linkageName: "_ZL6lgammaf", scope: !213, file: !213, line: 143, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!320 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !321, file: !213, line: 265)
!321 = !DISubprogram(name: "llabs", linkageName: "_ZL5llabsx", scope: !213, file: !213, line: 144, type: !322, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!322 = !DISubroutineType(types: !323)
!323 = !{!324, !324}
!324 = !DIBasicType(name: "long long int", size: 64, encoding: DW_ATE_signed)
!325 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !326, file: !213, line: 266)
!326 = !DISubprogram(name: "llrint", linkageName: "_ZL6llrintf", scope: !213, file: !213, line: 146, type: !327, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!327 = !DISubroutineType(types: !328)
!328 = !{!324, !220}
!329 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !330, file: !213, line: 267)
!330 = !DISubprogram(name: "log", linkageName: "_ZL3logf", scope: !213, file: !213, line: 159, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!331 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !332, file: !213, line: 268)
!332 = !DISubprogram(name: "log10", linkageName: "_ZL5log10f", scope: !213, file: !213, line: 148, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!333 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !334, file: !213, line: 269)
!334 = !DISubprogram(name: "log1p", linkageName: "_ZL5log1pf", scope: !213, file: !213, line: 150, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!335 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !336, file: !213, line: 270)
!336 = !DISubprogram(name: "log2", linkageName: "_ZL4log2f", scope: !213, file: !213, line: 152, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!337 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !338, file: !213, line: 271)
!338 = !DISubprogram(name: "logb", linkageName: "_ZL4logbf", scope: !213, file: !213, line: 154, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!339 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !340, file: !213, line: 272)
!340 = !DISubprogram(name: "lrint", linkageName: "_ZL5lrintf", scope: !213, file: !213, line: 161, type: !341, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!341 = !DISubroutineType(types: !342)
!342 = !{!313, !220}
!343 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !344, file: !213, line: 273)
!344 = !DISubprogram(name: "lround", linkageName: "_ZL6lroundf", scope: !213, file: !213, line: 163, type: !341, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!345 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !346, file: !213, line: 274)
!346 = !DISubprogram(name: "llround", linkageName: "_ZL7llroundf", scope: !213, file: !213, line: 164, type: !327, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!347 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !348, file: !213, line: 275)
!348 = !DISubprogram(name: "modf", linkageName: "_ZL4modffPf", scope: !213, file: !213, line: 166, type: !349, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!349 = !DISubroutineType(types: !350)
!350 = !{!220, !220, !351}
!351 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !220, size: 64)
!352 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !353, file: !213, line: 276)
!353 = !DISubprogram(name: "nan", linkageName: "_ZL3nanPKc", scope: !213, file: !213, line: 167, type: !354, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!354 = !DISubroutineType(types: !355)
!355 = !{!98, !356}
!356 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !357, size: 64)
!357 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !101)
!358 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !359, file: !213, line: 277)
!359 = !DISubprogram(name: "nanf", linkageName: "_ZL4nanfPKc", scope: !213, file: !213, line: 168, type: !360, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!360 = !DISubroutineType(types: !361)
!361 = !{!220, !356}
!362 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !363, file: !213, line: 278)
!363 = !DISubprogram(name: "nearbyint", linkageName: "_ZL9nearbyintf", scope: !213, file: !213, line: 170, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!364 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !365, file: !213, line: 279)
!365 = !DISubprogram(name: "nextafter", linkageName: "_ZL9nextafterff", scope: !213, file: !213, line: 172, type: !231, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!366 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !367, file: !213, line: 280)
!367 = !DISubprogram(name: "pow", linkageName: "_ZL3powfi", scope: !213, file: !213, line: 176, type: !316, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!368 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !369, file: !213, line: 281)
!369 = !DISubprogram(name: "remainder", linkageName: "_ZL9remainderff", scope: !213, file: !213, line: 178, type: !231, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!370 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !371, file: !213, line: 282)
!371 = !DISubprogram(name: "remquo", linkageName: "_ZL6remquoffPi", scope: !213, file: !213, line: 180, type: !372, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!372 = !DISubroutineType(types: !373)
!373 = !{!220, !220, !220, !279}
!374 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !375, file: !213, line: 283)
!375 = !DISubprogram(name: "rint", linkageName: "_ZL4rintf", scope: !213, file: !213, line: 182, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!376 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !377, file: !213, line: 284)
!377 = !DISubprogram(name: "round", linkageName: "_ZL5roundf", scope: !213, file: !213, line: 184, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!378 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !379, file: !213, line: 285)
!379 = !DISubprogram(name: "scalbln", linkageName: "_ZL7scalblnfl", scope: !213, file: !213, line: 186, type: !380, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!380 = !DISubroutineType(types: !381)
!381 = !{!220, !220, !313}
!382 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !383, file: !213, line: 286)
!383 = !DISubprogram(name: "scalbn", linkageName: "_ZL6scalbnfi", scope: !213, file: !213, line: 188, type: !316, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!384 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !385, file: !213, line: 287)
!385 = !DISubprogram(name: "signbit", linkageName: "_ZL7signbitf", scope: !213, file: !213, line: 190, type: !286, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!386 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !387, file: !213, line: 288)
!387 = !DISubprogram(name: "sin", linkageName: "_ZL3sinf", scope: !213, file: !213, line: 192, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!388 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !389, file: !213, line: 289)
!389 = !DISubprogram(name: "sinh", linkageName: "_ZL4sinhf", scope: !213, file: !213, line: 194, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!390 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !391, file: !213, line: 290)
!391 = !DISubprogram(name: "sqrt", linkageName: "_ZL4sqrtf", scope: !213, file: !213, line: 196, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!392 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !393, file: !213, line: 291)
!393 = !DISubprogram(name: "tan", linkageName: "_ZL3tanf", scope: !213, file: !213, line: 198, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!394 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !395, file: !213, line: 292)
!395 = !DISubprogram(name: "tanh", linkageName: "_ZL4tanhf", scope: !213, file: !213, line: 200, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!396 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !397, file: !213, line: 293)
!397 = !DISubprogram(name: "tgamma", linkageName: "_ZL6tgammaf", scope: !213, file: !213, line: 202, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!398 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !399, file: !213, line: 294)
!399 = !DISubprogram(name: "trunc", linkageName: "_ZL5truncf", scope: !213, file: !213, line: 204, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!400 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !401, file: !403, line: 52)
!401 = !DISubprogram(name: "abs", scope: !402, file: !402, line: 848, type: !214, flags: DIFlagPrototyped, spFlags: 0)
!402 = !DIFile(filename: "/usr/include/stdlib.h", directory: "")
!403 = !DIFile(filename: "/usr/lib/gcc/x86_64-linux-gnu/11/../../../../include/c++/11/bits/std_abs.h", directory: "")
!404 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !405, file: !409, line: 83)
!405 = !DISubprogram(name: "acos", scope: !406, file: !406, line: 53, type: !407, flags: DIFlagPrototyped, spFlags: 0)
!406 = !DIFile(filename: "/usr/include/x86_64-linux-gnu/bits/mathcalls.h", directory: "")
!407 = !DISubroutineType(types: !408)
!408 = !{!98, !98}
!409 = !DIFile(filename: "/usr/lib/gcc/x86_64-linux-gnu/11/../../../../include/c++/11/cmath", directory: "")
!410 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !411, file: !409, line: 102)
!411 = !DISubprogram(name: "asin", scope: !406, file: !406, line: 55, type: !407, flags: DIFlagPrototyped, spFlags: 0)
!412 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !413, file: !409, line: 121)
!413 = !DISubprogram(name: "atan", scope: !406, file: !406, line: 57, type: !407, flags: DIFlagPrototyped, spFlags: 0)
!414 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !415, file: !409, line: 140)
!415 = !DISubprogram(name: "atan2", scope: !406, file: !406, line: 59, type: !416, flags: DIFlagPrototyped, spFlags: 0)
!416 = !DISubroutineType(types: !417)
!417 = !{!98, !98, !98}
!418 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !419, file: !409, line: 161)
!419 = !DISubprogram(name: "ceil", scope: !406, file: !406, line: 159, type: !407, flags: DIFlagPrototyped, spFlags: 0)
!420 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !421, file: !409, line: 180)
!421 = !DISubprogram(name: "cos", scope: !406, file: !406, line: 62, type: !407, flags: DIFlagPrototyped, spFlags: 0)
!422 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !423, file: !409, line: 199)
!423 = !DISubprogram(name: "cosh", scope: !406, file: !406, line: 71, type: !407, flags: DIFlagPrototyped, spFlags: 0)
!424 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !425, file: !409, line: 218)
!425 = !DISubprogram(name: "exp", scope: !406, file: !406, line: 95, type: !407, flags: DIFlagPrototyped, spFlags: 0)
!426 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !427, file: !409, line: 237)
!427 = !DISubprogram(name: "fabs", scope: !406, file: !406, line: 162, type: !407, flags: DIFlagPrototyped, spFlags: 0)
!428 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !429, file: !409, line: 256)
!429 = !DISubprogram(name: "floor", scope: !406, file: !406, line: 165, type: !407, flags: DIFlagPrototyped, spFlags: 0)
!430 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !431, file: !409, line: 275)
!431 = !DISubprogram(name: "fmod", scope: !406, file: !406, line: 168, type: !416, flags: DIFlagPrototyped, spFlags: 0)
!432 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !433, file: !409, line: 296)
!433 = !DISubprogram(name: "frexp", scope: !406, file: !406, line: 98, type: !434, flags: DIFlagPrototyped, spFlags: 0)
!434 = !DISubroutineType(types: !435)
!435 = !{!98, !98, !279}
!436 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !437, file: !409, line: 315)
!437 = !DISubprogram(name: "ldexp", scope: !406, file: !406, line: 101, type: !438, flags: DIFlagPrototyped, spFlags: 0)
!438 = !DISubroutineType(types: !439)
!439 = !{!98, !98, !99}
!440 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !441, file: !409, line: 334)
!441 = !DISubprogram(name: "log", scope: !406, file: !406, line: 104, type: !407, flags: DIFlagPrototyped, spFlags: 0)
!442 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !443, file: !409, line: 353)
!443 = !DISubprogram(name: "log10", scope: !406, file: !406, line: 107, type: !407, flags: DIFlagPrototyped, spFlags: 0)
!444 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !445, file: !409, line: 372)
!445 = !DISubprogram(name: "modf", scope: !406, file: !406, line: 110, type: !446, flags: DIFlagPrototyped, spFlags: 0)
!446 = !DISubroutineType(types: !447)
!447 = !{!98, !98, !97}
!448 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !449, file: !409, line: 384)
!449 = !DISubprogram(name: "pow", scope: !406, file: !406, line: 140, type: !416, flags: DIFlagPrototyped, spFlags: 0)
!450 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !451, file: !409, line: 421)
!451 = !DISubprogram(name: "sin", scope: !406, file: !406, line: 64, type: !407, flags: DIFlagPrototyped, spFlags: 0)
!452 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !453, file: !409, line: 440)
!453 = !DISubprogram(name: "sinh", scope: !406, file: !406, line: 73, type: !407, flags: DIFlagPrototyped, spFlags: 0)
!454 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !455, file: !409, line: 459)
!455 = !DISubprogram(name: "sqrt", scope: !406, file: !406, line: 143, type: !407, flags: DIFlagPrototyped, spFlags: 0)
!456 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !457, file: !409, line: 478)
!457 = !DISubprogram(name: "tan", scope: !406, file: !406, line: 66, type: !407, flags: DIFlagPrototyped, spFlags: 0)
!458 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !459, file: !409, line: 497)
!459 = !DISubprogram(name: "tanh", scope: !406, file: !406, line: 75, type: !407, flags: DIFlagPrototyped, spFlags: 0)
!460 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !461, file: !463, line: 127)
!461 = !DIDerivedType(tag: DW_TAG_typedef, name: "div_t", file: !402, line: 63, baseType: !462)
!462 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !402, line: 59, flags: DIFlagFwdDecl, identifier: "_ZTS5div_t")
!463 = !DIFile(filename: "/usr/lib/gcc/x86_64-linux-gnu/11/../../../../include/c++/11/cstdlib", directory: "")
!464 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !465, file: !463, line: 128)
!465 = !DIDerivedType(tag: DW_TAG_typedef, name: "ldiv_t", file: !402, line: 71, baseType: !466)
!466 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !402, line: 67, size: 128, flags: DIFlagTypePassByValue, elements: !467, identifier: "_ZTS6ldiv_t")
!467 = !{!468, !469}
!468 = !DIDerivedType(tag: DW_TAG_member, name: "quot", scope: !466, file: !402, line: 69, baseType: !313, size: 64)
!469 = !DIDerivedType(tag: DW_TAG_member, name: "rem", scope: !466, file: !402, line: 70, baseType: !313, size: 64, offset: 64)
!470 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !471, file: !463, line: 130)
!471 = !DISubprogram(name: "abort", scope: !402, file: !402, line: 598, type: !472, flags: DIFlagPrototyped | DIFlagNoReturn, spFlags: 0)
!472 = !DISubroutineType(types: !473)
!473 = !{null}
!474 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !475, file: !463, line: 134)
!475 = !DISubprogram(name: "atexit", scope: !402, file: !402, line: 602, type: !476, flags: DIFlagPrototyped, spFlags: 0)
!476 = !DISubroutineType(types: !477)
!477 = !{!99, !478}
!478 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !472, size: 64)
!479 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !480, file: !463, line: 140)
!480 = !DISubprogram(name: "atof", scope: !402, file: !402, line: 102, type: !354, flags: DIFlagPrototyped, spFlags: 0)
!481 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !482, file: !463, line: 141)
!482 = !DISubprogram(name: "atoi", scope: !402, file: !402, line: 105, type: !483, flags: DIFlagPrototyped, spFlags: 0)
!483 = !DISubroutineType(types: !484)
!484 = !{!99, !356}
!485 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !486, file: !463, line: 142)
!486 = !DISubprogram(name: "atol", scope: !402, file: !402, line: 108, type: !487, flags: DIFlagPrototyped, spFlags: 0)
!487 = !DISubroutineType(types: !488)
!488 = !{!313, !356}
!489 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !490, file: !463, line: 143)
!490 = !DISubprogram(name: "bsearch", scope: !402, file: !402, line: 828, type: !491, flags: DIFlagPrototyped, spFlags: 0)
!491 = !DISubroutineType(types: !492)
!492 = !{!103, !493, !493, !123, !123, !495}
!493 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !494, size: 64)
!494 = !DIDerivedType(tag: DW_TAG_const_type, baseType: null)
!495 = !DIDerivedType(tag: DW_TAG_typedef, name: "__compar_fn_t", file: !402, line: 816, baseType: !496)
!496 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !497, size: 64)
!497 = !DISubroutineType(types: !498)
!498 = !{!99, !493, !493}
!499 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !500, file: !463, line: 144)
!500 = !DISubprogram(name: "calloc", scope: !402, file: !402, line: 543, type: !501, flags: DIFlagPrototyped, spFlags: 0)
!501 = !DISubroutineType(types: !502)
!502 = !{!103, !123, !123}
!503 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !504, file: !463, line: 145)
!504 = !DISubprogram(name: "div", scope: !402, file: !402, line: 860, type: !505, flags: DIFlagPrototyped, spFlags: 0)
!505 = !DISubroutineType(types: !506)
!506 = !{!461, !99, !99}
!507 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !508, file: !463, line: 146)
!508 = !DISubprogram(name: "exit", scope: !402, file: !402, line: 624, type: !509, flags: DIFlagPrototyped | DIFlagNoReturn, spFlags: 0)
!509 = !DISubroutineType(types: !510)
!510 = !{null, !99}
!511 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !512, file: !463, line: 147)
!512 = !DISubprogram(name: "free", scope: !402, file: !402, line: 555, type: !513, flags: DIFlagPrototyped, spFlags: 0)
!513 = !DISubroutineType(types: !514)
!514 = !{null, !103}
!515 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !516, file: !463, line: 148)
!516 = !DISubprogram(name: "getenv", scope: !402, file: !402, line: 641, type: !517, flags: DIFlagPrototyped, spFlags: 0)
!517 = !DISubroutineType(types: !518)
!518 = !{!100, !356}
!519 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !520, file: !463, line: 149)
!520 = !DISubprogram(name: "labs", scope: !402, file: !402, line: 849, type: !311, flags: DIFlagPrototyped, spFlags: 0)
!521 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !522, file: !463, line: 150)
!522 = !DISubprogram(name: "ldiv", scope: !402, file: !402, line: 862, type: !523, flags: DIFlagPrototyped, spFlags: 0)
!523 = !DISubroutineType(types: !524)
!524 = !{!465, !313, !313}
!525 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !526, file: !463, line: 151)
!526 = !DISubprogram(name: "malloc", scope: !402, file: !402, line: 540, type: !527, flags: DIFlagPrototyped, spFlags: 0)
!527 = !DISubroutineType(types: !528)
!528 = !{!103, !123}
!529 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !530, file: !463, line: 153)
!530 = !DISubprogram(name: "mblen", scope: !402, file: !402, line: 930, type: !531, flags: DIFlagPrototyped, spFlags: 0)
!531 = !DISubroutineType(types: !532)
!532 = !{!99, !356, !123}
!533 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !534, file: !463, line: 154)
!534 = !DISubprogram(name: "mbstowcs", scope: !402, file: !402, line: 941, type: !535, flags: DIFlagPrototyped, spFlags: 0)
!535 = !DISubroutineType(types: !536)
!536 = !{!123, !537, !540, !123}
!537 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !538)
!538 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !539, size: 64)
!539 = !DIBasicType(name: "wchar_t", size: 32, encoding: DW_ATE_signed)
!540 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !356)
!541 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !542, file: !463, line: 155)
!542 = !DISubprogram(name: "mbtowc", scope: !402, file: !402, line: 933, type: !543, flags: DIFlagPrototyped, spFlags: 0)
!543 = !DISubroutineType(types: !544)
!544 = !{!99, !537, !540, !123}
!545 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !546, file: !463, line: 157)
!546 = !DISubprogram(name: "qsort", scope: !402, file: !402, line: 838, type: !547, flags: DIFlagPrototyped, spFlags: 0)
!547 = !DISubroutineType(types: !548)
!548 = !{null, !103, !123, !123, !495}
!549 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !550, file: !463, line: 163)
!550 = !DISubprogram(name: "rand", scope: !402, file: !402, line: 454, type: !551, flags: DIFlagPrototyped, spFlags: 0)
!551 = !DISubroutineType(types: !552)
!552 = !{!99}
!553 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !554, file: !463, line: 164)
!554 = !DISubprogram(name: "realloc", scope: !402, file: !402, line: 551, type: !555, flags: DIFlagPrototyped, spFlags: 0)
!555 = !DISubroutineType(types: !556)
!556 = !{!103, !103, !123}
!557 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !558, file: !463, line: 165)
!558 = !DISubprogram(name: "srand", scope: !402, file: !402, line: 456, type: !559, flags: DIFlagPrototyped, spFlags: 0)
!559 = !DISubroutineType(types: !560)
!560 = !{null, !7}
!561 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !562, file: !463, line: 166)
!562 = !DISubprogram(name: "strtod", scope: !402, file: !402, line: 118, type: !563, flags: DIFlagPrototyped, spFlags: 0)
!563 = !DISubroutineType(types: !564)
!564 = !{!98, !540, !565}
!565 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !566)
!566 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !100, size: 64)
!567 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !568, file: !463, line: 167)
!568 = !DISubprogram(name: "strtol", scope: !402, file: !402, line: 177, type: !569, flags: DIFlagPrototyped, spFlags: 0)
!569 = !DISubroutineType(types: !570)
!570 = !{!313, !540, !565, !99}
!571 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !572, file: !463, line: 168)
!572 = !DISubprogram(name: "strtoul", scope: !402, file: !402, line: 181, type: !573, flags: DIFlagPrototyped, spFlags: 0)
!573 = !DISubroutineType(types: !574)
!574 = !{!125, !540, !565, !99}
!575 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !576, file: !463, line: 169)
!576 = !DISubprogram(name: "system", scope: !402, file: !402, line: 791, type: !483, flags: DIFlagPrototyped, spFlags: 0)
!577 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !578, file: !463, line: 171)
!578 = !DISubprogram(name: "wcstombs", scope: !402, file: !402, line: 945, type: !579, flags: DIFlagPrototyped, spFlags: 0)
!579 = !DISubroutineType(types: !580)
!580 = !{!123, !581, !582, !123}
!581 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !100)
!582 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !583)
!583 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !584, size: 64)
!584 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !539)
!585 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !586, file: !463, line: 172)
!586 = !DISubprogram(name: "wctomb", scope: !402, file: !402, line: 937, type: !587, flags: DIFlagPrototyped, spFlags: 0)
!587 = !DISubroutineType(types: !588)
!588 = !{!99, !100, !539}
!589 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !590, entity: !591, file: !463, line: 200)
!590 = !DINamespace(name: "__gnu_cxx", scope: null)
!591 = !DIDerivedType(tag: DW_TAG_typedef, name: "lldiv_t", file: !402, line: 81, baseType: !592)
!592 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !402, line: 77, size: 128, flags: DIFlagTypePassByValue, elements: !593, identifier: "_ZTS7lldiv_t")
!593 = !{!594, !595}
!594 = !DIDerivedType(tag: DW_TAG_member, name: "quot", scope: !592, file: !402, line: 79, baseType: !324, size: 64)
!595 = !DIDerivedType(tag: DW_TAG_member, name: "rem", scope: !592, file: !402, line: 80, baseType: !324, size: 64, offset: 64)
!596 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !590, entity: !597, file: !463, line: 206)
!597 = !DISubprogram(name: "_Exit", scope: !402, file: !402, line: 636, type: !509, flags: DIFlagPrototyped | DIFlagNoReturn, spFlags: 0)
!598 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !590, entity: !599, file: !463, line: 210)
!599 = !DISubprogram(name: "llabs", scope: !402, file: !402, line: 852, type: !322, flags: DIFlagPrototyped, spFlags: 0)
!600 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !590, entity: !601, file: !463, line: 216)
!601 = !DISubprogram(name: "lldiv", scope: !402, file: !402, line: 866, type: !602, flags: DIFlagPrototyped, spFlags: 0)
!602 = !DISubroutineType(types: !603)
!603 = !{!591, !324, !324}
!604 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !590, entity: !605, file: !463, line: 227)
!605 = !DISubprogram(name: "atoll", scope: !402, file: !402, line: 113, type: !606, flags: DIFlagPrototyped, spFlags: 0)
!606 = !DISubroutineType(types: !607)
!607 = !{!324, !356}
!608 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !590, entity: !609, file: !463, line: 228)
!609 = !DISubprogram(name: "strtoll", scope: !402, file: !402, line: 201, type: !610, flags: DIFlagPrototyped, spFlags: 0)
!610 = !DISubroutineType(types: !611)
!611 = !{!324, !540, !565, !99}
!612 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !590, entity: !613, file: !463, line: 229)
!613 = !DISubprogram(name: "strtoull", scope: !402, file: !402, line: 206, type: !614, flags: DIFlagPrototyped, spFlags: 0)
!614 = !DISubroutineType(types: !615)
!615 = !{!616, !540, !565, !99}
!616 = !DIBasicType(name: "long long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!617 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !590, entity: !618, file: !463, line: 231)
!618 = !DISubprogram(name: "strtof", scope: !402, file: !402, line: 124, type: !619, flags: DIFlagPrototyped, spFlags: 0)
!619 = !DISubroutineType(types: !620)
!620 = !{!220, !540, !565}
!621 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !590, entity: !622, file: !463, line: 232)
!622 = !DISubprogram(name: "strtold", scope: !402, file: !402, line: 127, type: !623, flags: DIFlagPrototyped, spFlags: 0)
!623 = !DISubroutineType(types: !624)
!624 = !{!625, !540, !565}
!625 = !DIBasicType(name: "long double", size: 128, encoding: DW_ATE_float)
!626 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !591, file: !463, line: 240)
!627 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !597, file: !463, line: 242)
!628 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !599, file: !463, line: 244)
!629 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !630, file: !463, line: 245)
!630 = !DISubprogram(name: "div", linkageName: "_ZN9__gnu_cxx3divExx", scope: !590, file: !463, line: 213, type: !602, flags: DIFlagPrototyped, spFlags: 0)
!631 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !601, file: !463, line: 246)
!632 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !605, file: !463, line: 248)
!633 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !618, file: !463, line: 249)
!634 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !609, file: !463, line: 250)
!635 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !613, file: !463, line: 251)
!636 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !622, file: !463, line: 252)
!637 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !471, file: !638, line: 38)
!638 = !DIFile(filename: "/usr/lib/gcc/x86_64-linux-gnu/11/../../../../include/c++/11/stdlib.h", directory: "")
!639 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !475, file: !638, line: 39)
!640 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !508, file: !638, line: 40)
!641 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !461, file: !638, line: 51)
!642 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !465, file: !638, line: 52)
!643 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !644, file: !638, line: 54)
!644 = !DISubprogram(name: "abs", linkageName: "_ZSt3abse", scope: !211, file: !403, line: 79, type: !645, flags: DIFlagPrototyped, spFlags: 0)
!645 = !DISubroutineType(types: !646)
!646 = !{!625, !625}
!647 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !480, file: !638, line: 55)
!648 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !482, file: !638, line: 56)
!649 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !486, file: !638, line: 57)
!650 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !490, file: !638, line: 58)
!651 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !500, file: !638, line: 59)
!652 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !630, file: !638, line: 60)
!653 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !512, file: !638, line: 61)
!654 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !516, file: !638, line: 62)
!655 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !520, file: !638, line: 63)
!656 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !522, file: !638, line: 64)
!657 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !526, file: !638, line: 65)
!658 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !530, file: !638, line: 67)
!659 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !534, file: !638, line: 68)
!660 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !542, file: !638, line: 69)
!661 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !546, file: !638, line: 71)
!662 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !550, file: !638, line: 72)
!663 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !554, file: !638, line: 73)
!664 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !558, file: !638, line: 74)
!665 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !562, file: !638, line: 75)
!666 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !568, file: !638, line: 76)
!667 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !572, file: !638, line: 77)
!668 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !576, file: !638, line: 78)
!669 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !578, file: !638, line: 80)
!670 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !2, entity: !586, file: !638, line: 81)
!671 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !672, file: !674, line: 414)
!672 = !DISubprogram(name: "acosf", linkageName: "_ZL5acosff", scope: !673, file: !673, line: 1126, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!673 = !DIFile(filename: "/usr/local/cuda/include/math_functions.hpp", directory: "")
!674 = !DIFile(filename: "/u/NAS_SCRATCH/ah7226/tulip/llvm-install-tulip/lib/clang/9.0.0/include/__clang_cuda_cmath.h", directory: "")
!675 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !676, file: !674, line: 415)
!676 = !DISubprogram(name: "acoshf", linkageName: "_ZL6acoshff", scope: !673, file: !673, line: 1154, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!677 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !678, file: !674, line: 416)
!678 = !DISubprogram(name: "asinf", linkageName: "_ZL5asinff", scope: !673, file: !673, line: 1121, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!679 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !680, file: !674, line: 417)
!680 = !DISubprogram(name: "asinhf", linkageName: "_ZL6asinhff", scope: !673, file: !673, line: 1159, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!681 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !682, file: !674, line: 418)
!682 = !DISubprogram(name: "atan2f", linkageName: "_ZL6atan2fff", scope: !673, file: !673, line: 1111, type: !231, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!683 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !684, file: !674, line: 419)
!684 = !DISubprogram(name: "atanf", linkageName: "_ZL5atanff", scope: !673, file: !673, line: 1116, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!685 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !686, file: !674, line: 420)
!686 = !DISubprogram(name: "atanhf", linkageName: "_ZL6atanhff", scope: !673, file: !673, line: 1164, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!687 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !688, file: !674, line: 421)
!688 = !DISubprogram(name: "cbrtf", linkageName: "_ZL5cbrtff", scope: !673, file: !673, line: 1199, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!689 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !690, file: !674, line: 422)
!690 = !DISubprogram(name: "ceilf", linkageName: "_ZL5ceilff", scope: !691, file: !691, line: 647, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!691 = !DIFile(filename: "/usr/local/cuda/include/device_functions.hpp", directory: "")
!692 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !693, file: !674, line: 423)
!693 = !DISubprogram(name: "copysignf", linkageName: "_ZL9copysignfff", scope: !673, file: !673, line: 973, type: !231, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!694 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !695, file: !674, line: 424)
!695 = !DISubprogram(name: "cosf", linkageName: "_ZL4cosff", scope: !673, file: !673, line: 1027, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!696 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !697, file: !674, line: 425)
!697 = !DISubprogram(name: "coshf", linkageName: "_ZL5coshff", scope: !673, file: !673, line: 1096, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!698 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !699, file: !674, line: 426)
!699 = !DISubprogram(name: "erfcf", linkageName: "_ZL5erfcff", scope: !673, file: !673, line: 1259, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!700 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !701, file: !674, line: 427)
!701 = !DISubprogram(name: "erff", linkageName: "_ZL4erfff", scope: !673, file: !673, line: 1249, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!702 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !703, file: !674, line: 428)
!703 = !DISubprogram(name: "exp2f", linkageName: "_ZL5exp2ff", scope: !691, file: !691, line: 637, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!704 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !705, file: !674, line: 429)
!705 = !DISubprogram(name: "expf", linkageName: "_ZL4expff", scope: !673, file: !673, line: 1078, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!706 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !707, file: !674, line: 430)
!707 = !DISubprogram(name: "expm1f", linkageName: "_ZL6expm1ff", scope: !673, file: !673, line: 1169, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!708 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !709, file: !674, line: 431)
!709 = !DISubprogram(name: "fabsf", linkageName: "_ZL5fabsff", scope: !691, file: !691, line: 582, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!710 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !711, file: !674, line: 432)
!711 = !DISubprogram(name: "fdimf", linkageName: "_ZL5fdimfff", scope: !673, file: !673, line: 1385, type: !231, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!712 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !713, file: !674, line: 433)
!713 = !DISubprogram(name: "floorf", linkageName: "_ZL6floorff", scope: !691, file: !691, line: 572, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!714 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !715, file: !674, line: 434)
!715 = !DISubprogram(name: "fmaf", linkageName: "_ZL4fmaffff", scope: !673, file: !673, line: 1337, type: !263, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!716 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !717, file: !674, line: 435)
!717 = !DISubprogram(name: "fmaxf", linkageName: "_ZL5fmaxfff", scope: !691, file: !691, line: 602, type: !231, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!718 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !719, file: !674, line: 436)
!719 = !DISubprogram(name: "fminf", linkageName: "_ZL5fminfff", scope: !691, file: !691, line: 597, type: !231, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!720 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !721, file: !674, line: 437)
!721 = !DISubprogram(name: "fmodf", linkageName: "_ZL5fmodfff", scope: !673, file: !673, line: 1322, type: !231, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!722 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !723, file: !674, line: 438)
!723 = !DISubprogram(name: "frexpf", linkageName: "_ZL6frexpffPi", scope: !673, file: !673, line: 1312, type: !277, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!724 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !725, file: !674, line: 439)
!725 = !DISubprogram(name: "hypotf", linkageName: "_ZL6hypotfff", scope: !673, file: !673, line: 1174, type: !231, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!726 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !727, file: !674, line: 440)
!727 = !DISubprogram(name: "ilogbf", linkageName: "_ZL6ilogbff", scope: !673, file: !673, line: 1390, type: !273, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!728 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !729, file: !674, line: 441)
!729 = !DISubprogram(name: "ldexpf", linkageName: "_ZL6ldexpffi", scope: !673, file: !673, line: 1289, type: !316, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!730 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !731, file: !674, line: 442)
!731 = !DISubprogram(name: "lgammaf", linkageName: "_ZL7lgammaff", scope: !673, file: !673, line: 1284, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!732 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !733, file: !674, line: 443)
!733 = !DISubprogram(name: "llrintf", linkageName: "_ZL7llrintff", scope: !673, file: !673, line: 933, type: !327, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!734 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !735, file: !674, line: 444)
!735 = !DISubprogram(name: "llroundf", linkageName: "_ZL8llroundff", scope: !673, file: !673, line: 1371, type: !327, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!736 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !737, file: !674, line: 445)
!737 = !DISubprogram(name: "log10f", linkageName: "_ZL6log10ff", scope: !673, file: !673, line: 1140, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!738 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !739, file: !674, line: 446)
!739 = !DISubprogram(name: "log1pf", linkageName: "_ZL6log1pff", scope: !673, file: !673, line: 1149, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!740 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !741, file: !674, line: 447)
!741 = !DISubprogram(name: "log2f", linkageName: "_ZL5log2ff", scope: !673, file: !673, line: 1069, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!742 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !743, file: !674, line: 448)
!743 = !DISubprogram(name: "logbf", linkageName: "_ZL5logbff", scope: !673, file: !673, line: 1395, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!744 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !745, file: !674, line: 449)
!745 = !DISubprogram(name: "logf", linkageName: "_ZL4logff", scope: !673, file: !673, line: 1131, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!746 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !747, file: !674, line: 450)
!747 = !DISubprogram(name: "lrintf", linkageName: "_ZL6lrintff", scope: !673, file: !673, line: 924, type: !341, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!748 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !749, file: !674, line: 451)
!749 = !DISubprogram(name: "lroundf", linkageName: "_ZL7lroundff", scope: !673, file: !673, line: 1376, type: !341, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!750 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !751, file: !674, line: 452)
!751 = !DISubprogram(name: "modff", linkageName: "_ZL5modfffPf", scope: !673, file: !673, line: 1317, type: !349, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!752 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !753, file: !674, line: 453)
!753 = !DISubprogram(name: "nearbyintf", linkageName: "_ZL10nearbyintff", scope: !673, file: !673, line: 938, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!754 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !755, file: !674, line: 454)
!755 = !DISubprogram(name: "nextafterf", linkageName: "_ZL10nextafterfff", scope: !673, file: !673, line: 1002, type: !231, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!756 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !757, file: !674, line: 455)
!757 = !DISubprogram(name: "powf", linkageName: "_ZL4powfff", scope: !673, file: !673, line: 1352, type: !231, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!758 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !759, file: !674, line: 456)
!759 = !DISubprogram(name: "remainderf", linkageName: "_ZL10remainderfff", scope: !673, file: !673, line: 1327, type: !231, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!760 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !761, file: !674, line: 457)
!761 = !DISubprogram(name: "remquof", linkageName: "_ZL7remquofffPi", scope: !673, file: !673, line: 1332, type: !372, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!762 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !763, file: !674, line: 458)
!763 = !DISubprogram(name: "rintf", linkageName: "_ZL5rintff", scope: !673, file: !673, line: 919, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!764 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !765, file: !674, line: 459)
!765 = !DISubprogram(name: "roundf", linkageName: "_ZL6roundff", scope: !673, file: !673, line: 1366, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!766 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !767, file: !674, line: 462)
!767 = !DISubprogram(name: "scalblnf", linkageName: "_ZL8scalblnffl", scope: !673, file: !673, line: 1299, type: !380, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!768 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !769, file: !674, line: 464)
!769 = !DISubprogram(name: "scalbnf", linkageName: "_ZL7scalbnffi", scope: !673, file: !673, line: 1294, type: !316, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!770 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !771, file: !674, line: 465)
!771 = !DISubprogram(name: "sinf", linkageName: "_ZL4sinff", scope: !673, file: !673, line: 1018, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!772 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !773, file: !674, line: 466)
!773 = !DISubprogram(name: "sinhf", linkageName: "_ZL5sinhff", scope: !673, file: !673, line: 1101, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!774 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !775, file: !674, line: 467)
!775 = !DISubprogram(name: "sqrtf", linkageName: "_ZL5sqrtff", scope: !691, file: !691, line: 887, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!776 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !777, file: !674, line: 468)
!777 = !DISubprogram(name: "tanf", linkageName: "_ZL4tanff", scope: !673, file: !673, line: 1060, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!778 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !779, file: !674, line: 469)
!779 = !DISubprogram(name: "tanhf", linkageName: "_ZL5tanhff", scope: !673, file: !673, line: 1106, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!780 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !781, file: !674, line: 470)
!781 = !DISubprogram(name: "tgammaf", linkageName: "_ZL7tgammaff", scope: !673, file: !673, line: 1361, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!782 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !783, file: !674, line: 471)
!783 = !DISubprogram(name: "truncf", linkageName: "_ZL6truncff", scope: !691, file: !691, line: 642, type: !218, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!784 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !785, file: !789, line: 98)
!785 = !DIDerivedType(tag: DW_TAG_typedef, name: "FILE", file: !786, line: 7, baseType: !787)
!786 = !DIFile(filename: "/usr/include/x86_64-linux-gnu/bits/types/FILE.h", directory: "")
!787 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "_IO_FILE", file: !788, line: 49, flags: DIFlagFwdDecl, identifier: "_ZTS8_IO_FILE")
!788 = !DIFile(filename: "/usr/include/x86_64-linux-gnu/bits/types/struct_FILE.h", directory: "")
!789 = !DIFile(filename: "/usr/lib/gcc/x86_64-linux-gnu/11/../../../../include/c++/11/cstdio", directory: "")
!790 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !791, file: !789, line: 99)
!791 = !DIDerivedType(tag: DW_TAG_typedef, name: "fpos_t", file: !792, line: 84, baseType: !793)
!792 = !DIFile(filename: "/usr/include/stdio.h", directory: "")
!793 = !DIDerivedType(tag: DW_TAG_typedef, name: "__fpos_t", file: !794, line: 14, baseType: !795)
!794 = !DIFile(filename: "/usr/include/x86_64-linux-gnu/bits/types/__fpos_t.h", directory: "")
!795 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "_G_fpos_t", file: !794, line: 10, flags: DIFlagFwdDecl, identifier: "_ZTS9_G_fpos_t")
!796 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !797, file: !789, line: 101)
!797 = !DISubprogram(name: "clearerr", scope: !792, file: !792, line: 786, type: !798, flags: DIFlagPrototyped, spFlags: 0)
!798 = !DISubroutineType(types: !799)
!799 = !{null, !800}
!800 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !785, size: 64)
!801 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !802, file: !789, line: 102)
!802 = !DISubprogram(name: "fclose", scope: !792, file: !792, line: 178, type: !803, flags: DIFlagPrototyped, spFlags: 0)
!803 = !DISubroutineType(types: !804)
!804 = !{!99, !800}
!805 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !806, file: !789, line: 103)
!806 = !DISubprogram(name: "feof", scope: !792, file: !792, line: 788, type: !803, flags: DIFlagPrototyped, spFlags: 0)
!807 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !808, file: !789, line: 104)
!808 = !DISubprogram(name: "ferror", scope: !792, file: !792, line: 790, type: !803, flags: DIFlagPrototyped, spFlags: 0)
!809 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !810, file: !789, line: 105)
!810 = !DISubprogram(name: "fflush", scope: !792, file: !792, line: 230, type: !803, flags: DIFlagPrototyped, spFlags: 0)
!811 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !812, file: !789, line: 106)
!812 = !DISubprogram(name: "fgetc", scope: !792, file: !792, line: 513, type: !803, flags: DIFlagPrototyped, spFlags: 0)
!813 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !814, file: !789, line: 107)
!814 = !DISubprogram(name: "fgetpos", scope: !792, file: !792, line: 760, type: !815, flags: DIFlagPrototyped, spFlags: 0)
!815 = !DISubroutineType(types: !816)
!816 = !{!99, !817, !818}
!817 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !800)
!818 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !819)
!819 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !791, size: 64)
!820 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !821, file: !789, line: 108)
!821 = !DISubprogram(name: "fgets", scope: !792, file: !792, line: 592, type: !822, flags: DIFlagPrototyped, spFlags: 0)
!822 = !DISubroutineType(types: !823)
!823 = !{!100, !581, !99, !817}
!824 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !825, file: !789, line: 109)
!825 = !DISubprogram(name: "fopen", scope: !792, file: !792, line: 258, type: !826, flags: DIFlagPrototyped, spFlags: 0)
!826 = !DISubroutineType(types: !827)
!827 = !{!800, !540, !540}
!828 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !829, file: !789, line: 110)
!829 = !DISubprogram(name: "fprintf", scope: !792, file: !792, line: 350, type: !830, flags: DIFlagPrototyped, spFlags: 0)
!830 = !DISubroutineType(types: !831)
!831 = !{!99, !817, !540, null}
!832 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !833, file: !789, line: 111)
!833 = !DISubprogram(name: "fputc", scope: !792, file: !792, line: 549, type: !834, flags: DIFlagPrototyped, spFlags: 0)
!834 = !DISubroutineType(types: !835)
!835 = !{!99, !99, !800}
!836 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !837, file: !789, line: 112)
!837 = !DISubprogram(name: "fputs", scope: !792, file: !792, line: 655, type: !838, flags: DIFlagPrototyped, spFlags: 0)
!838 = !DISubroutineType(types: !839)
!839 = !{!99, !540, !817}
!840 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !841, file: !789, line: 113)
!841 = !DISubprogram(name: "fread", scope: !792, file: !792, line: 675, type: !842, flags: DIFlagPrototyped, spFlags: 0)
!842 = !DISubroutineType(types: !843)
!843 = !{!123, !844, !123, !123, !817}
!844 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !103)
!845 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !846, file: !789, line: 114)
!846 = !DISubprogram(name: "freopen", scope: !792, file: !792, line: 265, type: !847, flags: DIFlagPrototyped, spFlags: 0)
!847 = !DISubroutineType(types: !848)
!848 = !{!800, !540, !540, !817}
!849 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !850, file: !789, line: 115)
!850 = !DISubprogram(name: "fscanf", scope: !792, file: !792, line: 415, type: !830, flags: DIFlagPrototyped, spFlags: 0)
!851 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !852, file: !789, line: 116)
!852 = !DISubprogram(name: "fseek", scope: !792, file: !792, line: 713, type: !853, flags: DIFlagPrototyped, spFlags: 0)
!853 = !DISubroutineType(types: !854)
!854 = !{!99, !800, !313, !99}
!855 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !856, file: !789, line: 117)
!856 = !DISubprogram(name: "fsetpos", scope: !792, file: !792, line: 765, type: !857, flags: DIFlagPrototyped, spFlags: 0)
!857 = !DISubroutineType(types: !858)
!858 = !{!99, !800, !859}
!859 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !860, size: 64)
!860 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !791)
!861 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !862, file: !789, line: 118)
!862 = !DISubprogram(name: "ftell", scope: !792, file: !792, line: 718, type: !863, flags: DIFlagPrototyped, spFlags: 0)
!863 = !DISubroutineType(types: !864)
!864 = !{!313, !800}
!865 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !866, file: !789, line: 119)
!866 = !DISubprogram(name: "fwrite", scope: !792, file: !792, line: 681, type: !867, flags: DIFlagPrototyped, spFlags: 0)
!867 = !DISubroutineType(types: !868)
!868 = !{!123, !869, !123, !123, !817}
!869 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !493)
!870 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !871, file: !789, line: 120)
!871 = !DISubprogram(name: "getc", scope: !792, file: !792, line: 514, type: !803, flags: DIFlagPrototyped, spFlags: 0)
!872 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !873, file: !789, line: 121)
!873 = !DISubprogram(name: "getchar", scope: !792, file: !792, line: 520, type: !551, flags: DIFlagPrototyped, spFlags: 0)
!874 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !875, file: !789, line: 124)
!875 = !DISubprogram(name: "gets", scope: !792, file: !792, line: 605, type: !876, flags: DIFlagPrototyped, spFlags: 0)
!876 = !DISubroutineType(types: !877)
!877 = !{!100, !100}
!878 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !879, file: !789, line: 126)
!879 = !DISubprogram(name: "perror", scope: !792, file: !792, line: 804, type: !880, flags: DIFlagPrototyped, spFlags: 0)
!880 = !DISubroutineType(types: !881)
!881 = !{null, !356}
!882 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !883, file: !789, line: 127)
!883 = !DISubprogram(name: "printf", scope: !792, file: !792, line: 356, type: !884, flags: DIFlagPrototyped, spFlags: 0)
!884 = !DISubroutineType(types: !885)
!885 = !{!99, !540, null}
!886 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !887, file: !789, line: 128)
!887 = !DISubprogram(name: "putc", scope: !792, file: !792, line: 550, type: !834, flags: DIFlagPrototyped, spFlags: 0)
!888 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !889, file: !789, line: 129)
!889 = !DISubprogram(name: "putchar", scope: !792, file: !792, line: 556, type: !214, flags: DIFlagPrototyped, spFlags: 0)
!890 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !891, file: !789, line: 130)
!891 = !DISubprogram(name: "puts", scope: !792, file: !792, line: 661, type: !483, flags: DIFlagPrototyped, spFlags: 0)
!892 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !893, file: !789, line: 131)
!893 = !DISubprogram(name: "remove", scope: !792, file: !792, line: 152, type: !483, flags: DIFlagPrototyped, spFlags: 0)
!894 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !895, file: !789, line: 132)
!895 = !DISubprogram(name: "rename", scope: !792, file: !792, line: 154, type: !896, flags: DIFlagPrototyped, spFlags: 0)
!896 = !DISubroutineType(types: !897)
!897 = !{!99, !356, !356}
!898 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !899, file: !789, line: 133)
!899 = !DISubprogram(name: "rewind", scope: !792, file: !792, line: 723, type: !798, flags: DIFlagPrototyped, spFlags: 0)
!900 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !901, file: !789, line: 134)
!901 = !DISubprogram(name: "scanf", scope: !792, file: !792, line: 421, type: !884, flags: DIFlagPrototyped, spFlags: 0)
!902 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !903, file: !789, line: 135)
!903 = !DISubprogram(name: "setbuf", scope: !792, file: !792, line: 328, type: !904, flags: DIFlagPrototyped, spFlags: 0)
!904 = !DISubroutineType(types: !905)
!905 = !{null, !817, !581}
!906 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !907, file: !789, line: 136)
!907 = !DISubprogram(name: "setvbuf", scope: !792, file: !792, line: 332, type: !908, flags: DIFlagPrototyped, spFlags: 0)
!908 = !DISubroutineType(types: !909)
!909 = !{!99, !817, !581, !99, !123}
!910 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !911, file: !789, line: 137)
!911 = !DISubprogram(name: "sprintf", scope: !792, file: !792, line: 358, type: !912, flags: DIFlagPrototyped, spFlags: 0)
!912 = !DISubroutineType(types: !913)
!913 = !{!99, !581, !540, null}
!914 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !915, file: !789, line: 138)
!915 = !DISubprogram(name: "sscanf", scope: !792, file: !792, line: 423, type: !916, flags: DIFlagPrototyped, spFlags: 0)
!916 = !DISubroutineType(types: !917)
!917 = !{!99, !540, !540, null}
!918 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !919, file: !789, line: 139)
!919 = !DISubprogram(name: "tmpfile", scope: !792, file: !792, line: 188, type: !920, flags: DIFlagPrototyped, spFlags: 0)
!920 = !DISubroutineType(types: !921)
!921 = !{!800}
!922 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !923, file: !789, line: 141)
!923 = !DISubprogram(name: "tmpnam", scope: !792, file: !792, line: 205, type: !876, flags: DIFlagPrototyped, spFlags: 0)
!924 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !925, file: !789, line: 143)
!925 = !DISubprogram(name: "ungetc", scope: !792, file: !792, line: 668, type: !834, flags: DIFlagPrototyped, spFlags: 0)
!926 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !927, file: !789, line: 144)
!927 = !DISubprogram(name: "vfprintf", scope: !792, file: !792, line: 365, type: !928, flags: DIFlagPrototyped, spFlags: 0)
!928 = !DISubroutineType(types: !929)
!929 = !{!99, !817, !540, !930}
!930 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !931, size: 64)
!931 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "__va_list_tag", file: !3, flags: DIFlagFwdDecl, identifier: "_ZTS13__va_list_tag")
!932 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !933, file: !789, line: 145)
!933 = !DISubprogram(name: "vprintf", scope: !792, file: !792, line: 371, type: !934, flags: DIFlagPrototyped, spFlags: 0)
!934 = !DISubroutineType(types: !935)
!935 = !{!99, !540, !930}
!936 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !937, file: !789, line: 146)
!937 = !DISubprogram(name: "vsprintf", scope: !792, file: !792, line: 373, type: !938, flags: DIFlagPrototyped, spFlags: 0)
!938 = !DISubroutineType(types: !939)
!939 = !{!99, !581, !540, !930}
!940 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !590, entity: !941, file: !789, line: 175)
!941 = !DISubprogram(name: "snprintf", scope: !792, file: !792, line: 378, type: !942, flags: DIFlagPrototyped, spFlags: 0)
!942 = !DISubroutineType(types: !943)
!943 = !{!99, !581, !123, !540, null}
!944 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !590, entity: !945, file: !789, line: 176)
!945 = !DISubprogram(name: "vfscanf", scope: !792, file: !792, line: 459, type: !928, flags: DIFlagPrototyped, spFlags: 0)
!946 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !590, entity: !947, file: !789, line: 177)
!947 = !DISubprogram(name: "vscanf", scope: !792, file: !792, line: 467, type: !934, flags: DIFlagPrototyped, spFlags: 0)
!948 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !590, entity: !949, file: !789, line: 178)
!949 = !DISubprogram(name: "vsnprintf", scope: !792, file: !792, line: 382, type: !950, flags: DIFlagPrototyped, spFlags: 0)
!950 = !DISubroutineType(types: !951)
!951 = !{!99, !581, !123, !540, !930}
!952 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !590, entity: !953, file: !789, line: 179)
!953 = !DISubprogram(name: "vsscanf", scope: !792, file: !792, line: 471, type: !954, flags: DIFlagPrototyped, spFlags: 0)
!954 = !DISubroutineType(types: !955)
!955 = !{!99, !540, !540, !930}
!956 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !941, file: !789, line: 185)
!957 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !945, file: !789, line: 186)
!958 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !947, file: !789, line: 187)
!959 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !949, file: !789, line: 188)
!960 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !953, file: !789, line: 189)
!961 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 9.0.0 (https://github.com/yebinchon/llvm-project/ a1efa594106d738d0b74c9e4e2b9b779eb8b7d25)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !962, retainedTypes: !552, imports: !963, nameTableKind: None)
!962 = !{}
!963 = !{!210, !216, !221, !223, !225, !227, !229, !233, !235, !237, !239, !241, !243, !245, !247, !249, !251, !253, !255, !257, !259, !261, !265, !267, !269, !271, !275, !280, !282, !284, !289, !293, !295, !297, !299, !301, !303, !305, !307, !309, !314, !318, !320, !325, !329, !331, !333, !335, !337, !339, !343, !345, !347, !352, !358, !362, !364, !366, !368, !370, !374, !376, !378, !382, !384, !386, !388, !390, !392, !394, !396, !398, !400, !404, !410, !412, !414, !418, !420, !422, !424, !426, !428, !430, !432, !436, !440, !442, !444, !448, !450, !452, !454, !456, !458, !460, !464, !470, !474, !479, !481, !485, !489, !499, !503, !507, !511, !515, !519, !521, !525, !529, !533, !541, !545, !549, !553, !557, !561, !567, !571, !575, !577, !585, !589, !596, !598, !600, !604, !608, !612, !617, !964, !626, !627, !628, !629, !631, !632, !633, !634, !635, !969, !970, !971, !972, !973, !974, !975, !979, !980, !981, !982, !983, !984, !985, !986, !987, !988, !989, !990, !991, !992, !993, !994, !995, !996, !997, !998, !999, !1000, !1001, !1002, !671, !675, !677, !679, !681, !683, !685, !687, !689, !692, !694, !696, !698, !700, !702, !704, !706, !708, !710, !712, !714, !716, !718, !720, !722, !724, !726, !728, !730, !732, !734, !736, !738, !740, !742, !744, !746, !748, !750, !752, !754, !756, !758, !760, !762, !764, !766, !768, !770, !772, !774, !776, !778, !780, !782, !784, !790, !796, !801, !805, !807, !809, !811, !813, !820, !824, !828, !832, !836, !840, !845, !849, !851, !855, !861, !865, !870, !872, !874, !878, !882, !886, !888, !890, !892, !894, !898, !900, !902, !906, !910, !914, !918, !922, !924, !1003, !1010, !1014, !940, !1018, !1020, !1022, !1026, !956, !1030, !1031, !1032, !1033}
!964 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !590, entity: !965, file: !463, line: 232)
!965 = !DISubprogram(name: "strtold", scope: !402, file: !402, line: 127, type: !966, flags: DIFlagPrototyped, spFlags: 0)
!966 = !DISubroutineType(types: !967)
!967 = !{!968, !540, !565}
!968 = !DIBasicType(name: "long double", size: 64, encoding: DW_ATE_float)
!969 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !965, file: !463, line: 252)
!970 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !961, entity: !471, file: !638, line: 38)
!971 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !961, entity: !475, file: !638, line: 39)
!972 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !961, entity: !508, file: !638, line: 40)
!973 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !961, entity: !461, file: !638, line: 51)
!974 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !961, entity: !465, file: !638, line: 52)
!975 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !961, entity: !976, file: !638, line: 54)
!976 = !DISubprogram(name: "abs", linkageName: "_ZSt3abse", scope: !211, file: !403, line: 79, type: !977, flags: DIFlagPrototyped, spFlags: 0)
!977 = !DISubroutineType(types: !978)
!978 = !{!968, !968}
!979 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !961, entity: !480, file: !638, line: 55)
!980 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !961, entity: !482, file: !638, line: 56)
!981 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !961, entity: !486, file: !638, line: 57)
!982 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !961, entity: !490, file: !638, line: 58)
!983 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !961, entity: !500, file: !638, line: 59)
!984 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !961, entity: !630, file: !638, line: 60)
!985 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !961, entity: !512, file: !638, line: 61)
!986 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !961, entity: !516, file: !638, line: 62)
!987 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !961, entity: !520, file: !638, line: 63)
!988 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !961, entity: !522, file: !638, line: 64)
!989 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !961, entity: !526, file: !638, line: 65)
!990 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !961, entity: !530, file: !638, line: 67)
!991 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !961, entity: !534, file: !638, line: 68)
!992 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !961, entity: !542, file: !638, line: 69)
!993 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !961, entity: !546, file: !638, line: 71)
!994 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !961, entity: !550, file: !638, line: 72)
!995 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !961, entity: !554, file: !638, line: 73)
!996 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !961, entity: !558, file: !638, line: 74)
!997 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !961, entity: !562, file: !638, line: 75)
!998 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !961, entity: !568, file: !638, line: 76)
!999 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !961, entity: !572, file: !638, line: 77)
!1000 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !961, entity: !576, file: !638, line: 78)
!1001 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !961, entity: !578, file: !638, line: 80)
!1002 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !961, entity: !586, file: !638, line: 81)
!1003 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !1004, file: !789, line: 144)
!1004 = !DISubprogram(name: "vfprintf", scope: !792, file: !792, line: 365, type: !1005, flags: DIFlagPrototyped, spFlags: 0)
!1005 = !DISubroutineType(types: !1006)
!1006 = !{!99, !817, !540, !1007}
!1007 = !DIDerivedType(tag: DW_TAG_typedef, name: "__gnuc_va_list", file: !1008, line: 32, baseType: !1009)
!1008 = !DIFile(filename: "/u/NAS_SCRATCH/ah7226/tulip/llvm-install-tulip/lib/clang/9.0.0/include/stdarg.h", directory: "")
!1009 = !DIDerivedType(tag: DW_TAG_typedef, name: "__builtin_va_list", file: !3, baseType: !100)
!1010 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !1011, file: !789, line: 145)
!1011 = !DISubprogram(name: "vprintf", scope: !792, file: !792, line: 371, type: !1012, flags: DIFlagPrototyped, spFlags: 0)
!1012 = !DISubroutineType(types: !1013)
!1013 = !{!99, !540, !1007}
!1014 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !1015, file: !789, line: 146)
!1015 = !DISubprogram(name: "vsprintf", scope: !792, file: !792, line: 373, type: !1016, flags: DIFlagPrototyped, spFlags: 0)
!1016 = !DISubroutineType(types: !1017)
!1017 = !{!99, !581, !540, !1007}
!1018 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !590, entity: !1019, file: !789, line: 176)
!1019 = !DISubprogram(name: "vfscanf", scope: !792, file: !792, line: 459, type: !1005, flags: DIFlagPrototyped, spFlags: 0)
!1020 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !590, entity: !1021, file: !789, line: 177)
!1021 = !DISubprogram(name: "vscanf", scope: !792, file: !792, line: 467, type: !1012, flags: DIFlagPrototyped, spFlags: 0)
!1022 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !590, entity: !1023, file: !789, line: 178)
!1023 = !DISubprogram(name: "vsnprintf", scope: !792, file: !792, line: 382, type: !1024, flags: DIFlagPrototyped, spFlags: 0)
!1024 = !DISubroutineType(types: !1025)
!1025 = !{!99, !581, !123, !540, !1007}
!1026 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !590, entity: !1027, file: !789, line: 179)
!1027 = !DISubprogram(name: "vsscanf", scope: !792, file: !792, line: 471, type: !1028, flags: DIFlagPrototyped, spFlags: 0)
!1028 = !DISubroutineType(types: !1029)
!1029 = !{!99, !540, !540, !1007}
!1030 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !1019, file: !789, line: 186)
!1031 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !1021, file: !789, line: 187)
!1032 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !1023, file: !789, line: 188)
!1033 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !1027, file: !789, line: 189)
!1034 = !{void (double*, double*, double*, double)* @_Z10gpu_kernelPdS_S_d, !"kernel", i32 1}
!1035 = !{null, !"align", i32 8}
!1036 = !{null, !"align", i32 8, !"align", i32 65544, !"align", i32 131080}
!1037 = !{null, !"align", i32 16}
!1038 = !{null, !"align", i32 16, !"align", i32 65552, !"align", i32 131088}
!1039 = !{!"clang version 9.0.0 (https://github.com/yebinchon/llvm-project/ a1efa594106d738d0b74c9e4e2b9b779eb8b7d25)"}
!1040 = !{i32 1, i32 2}
!1041 = !{i32 2, !"SDK Version", [2 x i32] [i32 7, i32 0]}
!1042 = !{i32 2, !"Dwarf Version", i32 2}
!1043 = !{i32 2, !"Debug Info Version", i32 3}
!1044 = !{i32 1, !"wchar_size", i32 4}
!1045 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!1046 = distinct !DISubprogram(name: "gpu_kernel", linkageName: "_Z10gpu_kernelPdS_S_d", scope: !3, file: !3, line: 463, type: !1047, scopeLine: 466, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !961, retainedNodes: !962)
!1047 = !DISubroutineType(types: !1048)
!1048 = !{null, !97, !97, !97, !98}
!1049 = !DILocalVariable(name: "a", arg: 1, scope: !1050, file: !1051, line: 225, type: !98)
!1050 = distinct !DISubprogram(name: "log", linkageName: "_ZL3logd", scope: !1051, file: !1051, line: 225, type: !407, scopeLine: 226, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !961, retainedNodes: !962)
!1051 = !DIFile(filename: "/usr/local/cuda/include/math_functions_dbl_ptx3.hpp", directory: "")
!1052 = !DILocation(line: 225, column: 52, scope: !1050, inlinedAt: !1053)
!1053 = distinct !DILocation(line: 527, column: 18, scope: !1054)
!1054 = distinct !DILexicalBlock(scope: !1055, file: !3, line: 526, column: 15)
!1055 = distinct !DILexicalBlock(scope: !1056, file: !3, line: 526, column: 7)
!1056 = distinct !DILexicalBlock(scope: !1057, file: !3, line: 522, column: 33)
!1057 = distinct !DILexicalBlock(scope: !1058, file: !3, line: 522, column: 3)
!1058 = distinct !DILexicalBlock(scope: !1059, file: !3, line: 522, column: 3)
!1059 = distinct !DILexicalBlock(scope: !1060, file: !3, line: 513, column: 39)
!1060 = distinct !DILexicalBlock(scope: !1061, file: !3, line: 513, column: 2)
!1061 = distinct !DILexicalBlock(scope: !1046, file: !3, line: 513, column: 2)
!1062 = !DILocalVariable(name: "x", arg: 1, scope: !1063, file: !691, line: 892, type: !98)
!1063 = distinct !DISubprogram(name: "sqrt", linkageName: "_ZL4sqrtd", scope: !691, file: !691, line: 892, type: !407, scopeLine: 893, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !961, retainedNodes: !962)
!1064 = !DILocation(line: 892, column: 53, scope: !1063, inlinedAt: !1065)
!1065 = distinct !DILocation(line: 527, column: 8, scope: !1054)
!1066 = !DILocation(line: 225, column: 52, scope: !1050, inlinedAt: !1067)
!1067 = distinct !DILocation(line: 528, column: 8, scope: !1054)
!1068 = !DILocalVariable(name: "f", arg: 1, scope: !1069, file: !691, line: 587, type: !98)
!1069 = distinct !DISubprogram(name: "fabs", linkageName: "_ZL4fabsd", scope: !691, file: !691, line: 587, type: !407, scopeLine: 588, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !961, retainedNodes: !962)
!1070 = !DILocation(line: 587, column: 53, scope: !1069, inlinedAt: !1071)
!1071 = distinct !DILocation(line: 530, column: 7, scope: !1054)
!1072 = !DILocation(line: 587, column: 53, scope: !1069, inlinedAt: !1073)
!1073 = distinct !DILocation(line: 530, column: 7, scope: !1054)
!1074 = !DILocation(line: 587, column: 53, scope: !1069, inlinedAt: !1075)
!1075 = distinct !DILocation(line: 530, column: 7, scope: !1054)
!1076 = !DILocation(line: 587, column: 53, scope: !1069, inlinedAt: !1077)
!1077 = distinct !DILocation(line: 530, column: 7, scope: !1054)
!1078 = !DILocalVariable(name: "q_global", arg: 1, scope: !1046, file: !3, line: 463, type: !97)
!1079 = !DILocation(line: 463, column: 36, scope: !1046)
!1080 = !DILocalVariable(name: "sx_global", arg: 2, scope: !1046, file: !3, line: 464, type: !97)
!1081 = !DILocation(line: 464, column: 11, scope: !1046)
!1082 = !DILocalVariable(name: "sy_global", arg: 3, scope: !1046, file: !3, line: 465, type: !97)
!1083 = !DILocation(line: 465, column: 11, scope: !1046)
!1084 = !DILocalVariable(name: "an", arg: 4, scope: !1046, file: !3, line: 466, type: !98)
!1085 = !DILocation(line: 466, column: 10, scope: !1046)
!1086 = !DILocalVariable(name: "x_local", scope: !1046, file: !3, line: 467, type: !1087)
!1087 = !DICompositeType(tag: DW_TAG_array_type, baseType: !98, size: 16384, elements: !140)
!1088 = !DILocation(line: 467, column: 9, scope: !1046)
!1089 = !DILocalVariable(name: "q_local", scope: !1046, file: !3, line: 468, type: !1090)
!1090 = !DICompositeType(tag: DW_TAG_array_type, baseType: !98, size: 640, elements: !1091)
!1091 = !{!1092}
!1092 = !DISubrange(count: 10)
!1093 = !DILocation(line: 468, column: 9, scope: !1046)
!1094 = !DILocalVariable(name: "sx_local", scope: !1046, file: !3, line: 469, type: !98)
!1095 = !DILocation(line: 469, column: 9, scope: !1046)
!1096 = !DILocalVariable(name: "sy_local", scope: !1046, file: !3, line: 469, type: !98)
!1097 = !DILocation(line: 469, column: 19, scope: !1046)
!1098 = !DILocalVariable(name: "t1", scope: !1046, file: !3, line: 470, type: !98)
!1099 = !DILocation(line: 470, column: 9, scope: !1046)
!1100 = !DILocalVariable(name: "t2", scope: !1046, file: !3, line: 470, type: !98)
!1101 = !DILocation(line: 470, column: 13, scope: !1046)
!1102 = !DILocalVariable(name: "t3", scope: !1046, file: !3, line: 470, type: !98)
!1103 = !DILocation(line: 470, column: 17, scope: !1046)
!1104 = !DILocalVariable(name: "t4", scope: !1046, file: !3, line: 470, type: !98)
!1105 = !DILocation(line: 470, column: 21, scope: !1046)
!1106 = !DILocalVariable(name: "x1", scope: !1046, file: !3, line: 470, type: !98)
!1107 = !DILocation(line: 470, column: 25, scope: !1046)
!1108 = !DILocalVariable(name: "x2", scope: !1046, file: !3, line: 470, type: !98)
!1109 = !DILocation(line: 470, column: 29, scope: !1046)
!1110 = !DILocalVariable(name: "seed", scope: !1046, file: !3, line: 470, type: !98)
!1111 = !DILocation(line: 470, column: 33, scope: !1046)
!1112 = !DILocalVariable(name: "i", scope: !1046, file: !3, line: 471, type: !99)
!1113 = !DILocation(line: 471, column: 6, scope: !1046)
!1114 = !DILocalVariable(name: "ii", scope: !1046, file: !3, line: 471, type: !99)
!1115 = !DILocation(line: 471, column: 9, scope: !1046)
!1116 = !DILocalVariable(name: "ik", scope: !1046, file: !3, line: 471, type: !99)
!1117 = !DILocation(line: 471, column: 13, scope: !1046)
!1118 = !DILocalVariable(name: "kk", scope: !1046, file: !3, line: 471, type: !99)
!1119 = !DILocation(line: 471, column: 17, scope: !1046)
!1120 = !DILocalVariable(name: "l", scope: !1046, file: !3, line: 471, type: !99)
!1121 = !DILocation(line: 471, column: 21, scope: !1046)
!1122 = !DILocation(line: 473, column: 2, scope: !1046)
!1123 = !DILocation(line: 473, column: 12, scope: !1046)
!1124 = !DILocation(line: 474, column: 2, scope: !1046)
!1125 = !DILocation(line: 474, column: 12, scope: !1046)
!1126 = !DILocation(line: 475, column: 2, scope: !1046)
!1127 = !DILocation(line: 475, column: 12, scope: !1046)
!1128 = !DILocation(line: 476, column: 2, scope: !1046)
!1129 = !DILocation(line: 476, column: 12, scope: !1046)
!1130 = !DILocation(line: 477, column: 2, scope: !1046)
!1131 = !DILocation(line: 477, column: 12, scope: !1046)
!1132 = !DILocation(line: 478, column: 2, scope: !1046)
!1133 = !DILocation(line: 478, column: 12, scope: !1046)
!1134 = !DILocation(line: 479, column: 2, scope: !1046)
!1135 = !DILocation(line: 479, column: 12, scope: !1046)
!1136 = !DILocation(line: 480, column: 2, scope: !1046)
!1137 = !DILocation(line: 480, column: 12, scope: !1046)
!1138 = !DILocation(line: 481, column: 2, scope: !1046)
!1139 = !DILocation(line: 481, column: 12, scope: !1046)
!1140 = !DILocation(line: 482, column: 2, scope: !1046)
!1141 = !DILocation(line: 482, column: 12, scope: !1046)
!1142 = !DILocation(line: 483, column: 10, scope: !1046)
!1143 = !DILocation(line: 484, column: 10, scope: !1046)
!1144 = !DILocation(line: 64, column: 3, scope: !1145, inlinedAt: !1180)
!1145 = distinct !DISubprogram(name: "__fetch_builtin_x", linkageName: "_ZN25__cuda_builtin_blockIdx_t17__fetch_builtin_xEv", scope: !1147, file: !1146, line: 64, type: !1150, scopeLine: 64, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !961, declaration: !1149, retainedNodes: !962)
!1146 = !DIFile(filename: "/u/NAS_SCRATCH/ah7226/tulip/llvm-install-tulip/lib/clang/9.0.0/include/__clang_cuda_builtin_vars.h", directory: "")
!1147 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "__cuda_builtin_blockIdx_t", file: !1146, line: 63, size: 8, flags: DIFlagTypePassByReference | DIFlagNonTrivial, elements: !1148, identifier: "_ZTS25__cuda_builtin_blockIdx_t")
!1148 = !{!1149, !1152, !1153, !1154, !1165, !1169, !1173, !1176}
!1149 = !DISubprogram(name: "__fetch_builtin_x", linkageName: "_ZN25__cuda_builtin_blockIdx_t17__fetch_builtin_xEv", scope: !1147, file: !1146, line: 64, type: !1150, scopeLine: 64, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!1150 = !DISubroutineType(types: !1151)
!1151 = !{!7}
!1152 = !DISubprogram(name: "__fetch_builtin_y", linkageName: "_ZN25__cuda_builtin_blockIdx_t17__fetch_builtin_yEv", scope: !1147, file: !1146, line: 65, type: !1150, scopeLine: 65, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!1153 = !DISubprogram(name: "__fetch_builtin_z", linkageName: "_ZN25__cuda_builtin_blockIdx_t17__fetch_builtin_zEv", scope: !1147, file: !1146, line: 66, type: !1150, scopeLine: 66, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!1154 = !DISubprogram(name: "operator uint3", linkageName: "_ZNK25__cuda_builtin_blockIdx_tcv5uint3Ev", scope: !1147, file: !1146, line: 69, type: !1155, scopeLine: 69, flags: DIFlagPrototyped, spFlags: 0)
!1155 = !DISubroutineType(types: !1156)
!1156 = !{!1157, !1163}
!1157 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "uint3", file: !1158, line: 190, size: 96, flags: DIFlagTypePassByValue, elements: !1159, identifier: "_ZTS5uint3")
!1158 = !DIFile(filename: "/usr/local/cuda/include/vector_types.h", directory: "")
!1159 = !{!1160, !1161, !1162}
!1160 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !1157, file: !1158, line: 192, baseType: !7, size: 32)
!1161 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !1157, file: !1158, line: 192, baseType: !7, size: 32, offset: 32)
!1162 = !DIDerivedType(tag: DW_TAG_member, name: "z", scope: !1157, file: !1158, line: 192, baseType: !7, size: 32, offset: 64)
!1163 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !1164, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!1164 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !1147)
!1165 = !DISubprogram(name: "__cuda_builtin_blockIdx_t", scope: !1147, file: !1146, line: 71, type: !1166, scopeLine: 71, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!1166 = !DISubroutineType(types: !1167)
!1167 = !{null, !1168}
!1168 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !1147, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!1169 = !DISubprogram(name: "__cuda_builtin_blockIdx_t", scope: !1147, file: !1146, line: 71, type: !1170, scopeLine: 71, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!1170 = !DISubroutineType(types: !1171)
!1171 = !{null, !1168, !1172}
!1172 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !1164, size: 64)
!1173 = !DISubprogram(name: "operator=", linkageName: "_ZNK25__cuda_builtin_blockIdx_taSERKS_", scope: !1147, file: !1146, line: 71, type: !1174, scopeLine: 71, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!1174 = !DISubroutineType(types: !1175)
!1175 = !{null, !1163, !1172}
!1176 = !DISubprogram(name: "operator&", linkageName: "_ZNK25__cuda_builtin_blockIdx_tadEv", scope: !1147, file: !1146, line: 71, type: !1177, scopeLine: 71, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!1177 = !DISubroutineType(types: !1178)
!1178 = !{!1179, !1163}
!1179 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !1147, size: 64)
!1180 = distinct !DILocation(line: 486, column: 5, scope: !1046)
!1181 = !{i32 0, i32 65535}
!1182 = !DILocation(line: 75, column: 3, scope: !1183, inlinedAt: !1225)
!1183 = distinct !DISubprogram(name: "__fetch_builtin_x", linkageName: "_ZN25__cuda_builtin_blockDim_t17__fetch_builtin_xEv", scope: !1184, file: !1146, line: 75, type: !1150, scopeLine: 75, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !961, declaration: !1186, retainedNodes: !962)
!1184 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "__cuda_builtin_blockDim_t", file: !1146, line: 74, size: 8, flags: DIFlagTypePassByReference | DIFlagNonTrivial, elements: !1185, identifier: "_ZTS25__cuda_builtin_blockDim_t")
!1185 = !{!1186, !1187, !1188, !1189, !1210, !1214, !1218, !1221}
!1186 = !DISubprogram(name: "__fetch_builtin_x", linkageName: "_ZN25__cuda_builtin_blockDim_t17__fetch_builtin_xEv", scope: !1184, file: !1146, line: 75, type: !1150, scopeLine: 75, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!1187 = !DISubprogram(name: "__fetch_builtin_y", linkageName: "_ZN25__cuda_builtin_blockDim_t17__fetch_builtin_yEv", scope: !1184, file: !1146, line: 76, type: !1150, scopeLine: 76, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!1188 = !DISubprogram(name: "__fetch_builtin_z", linkageName: "_ZN25__cuda_builtin_blockDim_t17__fetch_builtin_zEv", scope: !1184, file: !1146, line: 77, type: !1150, scopeLine: 77, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!1189 = !DISubprogram(name: "operator dim3", linkageName: "_ZNK25__cuda_builtin_blockDim_tcv4dim3Ev", scope: !1184, file: !1146, line: 80, type: !1190, scopeLine: 80, flags: DIFlagPrototyped, spFlags: 0)
!1190 = !DISubroutineType(types: !1191)
!1191 = !{!1192, !1208}
!1192 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "dim3", file: !1158, line: 417, size: 96, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !1193, identifier: "_ZTS4dim3")
!1193 = !{!1194, !1195, !1196, !1197, !1201, !1205}
!1194 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !1192, file: !1158, line: 419, baseType: !7, size: 32)
!1195 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !1192, file: !1158, line: 419, baseType: !7, size: 32, offset: 32)
!1196 = !DIDerivedType(tag: DW_TAG_member, name: "z", scope: !1192, file: !1158, line: 419, baseType: !7, size: 32, offset: 64)
!1197 = !DISubprogram(name: "dim3", scope: !1192, file: !1158, line: 421, type: !1198, scopeLine: 421, flags: DIFlagPrototyped, spFlags: 0)
!1198 = !DISubroutineType(types: !1199)
!1199 = !{null, !1200, !7, !7, !7}
!1200 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !1192, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!1201 = !DISubprogram(name: "dim3", scope: !1192, file: !1158, line: 422, type: !1202, scopeLine: 422, flags: DIFlagPrototyped, spFlags: 0)
!1202 = !DISubroutineType(types: !1203)
!1203 = !{null, !1200, !1204}
!1204 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint3", file: !1158, line: 383, baseType: !1157)
!1205 = !DISubprogram(name: "operator uint3", linkageName: "_ZN4dim3cv5uint3Ev", scope: !1192, file: !1158, line: 423, type: !1206, scopeLine: 423, flags: DIFlagPrototyped, spFlags: 0)
!1206 = !DISubroutineType(types: !1207)
!1207 = !{!1204, !1200}
!1208 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !1209, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!1209 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !1184)
!1210 = !DISubprogram(name: "__cuda_builtin_blockDim_t", scope: !1184, file: !1146, line: 82, type: !1211, scopeLine: 82, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!1211 = !DISubroutineType(types: !1212)
!1212 = !{null, !1213}
!1213 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !1184, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!1214 = !DISubprogram(name: "__cuda_builtin_blockDim_t", scope: !1184, file: !1146, line: 82, type: !1215, scopeLine: 82, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!1215 = !DISubroutineType(types: !1216)
!1216 = !{null, !1213, !1217}
!1217 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !1209, size: 64)
!1218 = !DISubprogram(name: "operator=", linkageName: "_ZNK25__cuda_builtin_blockDim_taSERKS_", scope: !1184, file: !1146, line: 82, type: !1219, scopeLine: 82, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!1219 = !DISubroutineType(types: !1220)
!1220 = !{null, !1208, !1217}
!1221 = !DISubprogram(name: "operator&", linkageName: "_ZNK25__cuda_builtin_blockDim_tadEv", scope: !1184, file: !1146, line: 82, type: !1222, scopeLine: 82, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!1222 = !DISubroutineType(types: !1223)
!1223 = !{!1224, !1208}
!1224 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !1184, size: 64)
!1225 = distinct !DILocation(line: 486, column: 16, scope: !1046)
!1226 = !{i32 1, i32 1025}
!1227 = !DILocation(line: 486, column: 15, scope: !1046)
!1228 = !DILocation(line: 53, column: 3, scope: !1229, inlinedAt: !1255)
!1229 = distinct !DISubprogram(name: "__fetch_builtin_x", linkageName: "_ZN26__cuda_builtin_threadIdx_t17__fetch_builtin_xEv", scope: !1230, file: !1146, line: 53, type: !1150, scopeLine: 53, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !961, declaration: !1232, retainedNodes: !962)
!1230 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "__cuda_builtin_threadIdx_t", file: !1146, line: 52, size: 8, flags: DIFlagTypePassByReference | DIFlagNonTrivial, elements: !1231, identifier: "_ZTS26__cuda_builtin_threadIdx_t")
!1231 = !{!1232, !1233, !1234, !1235, !1240, !1244, !1248, !1251}
!1232 = !DISubprogram(name: "__fetch_builtin_x", linkageName: "_ZN26__cuda_builtin_threadIdx_t17__fetch_builtin_xEv", scope: !1230, file: !1146, line: 53, type: !1150, scopeLine: 53, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!1233 = !DISubprogram(name: "__fetch_builtin_y", linkageName: "_ZN26__cuda_builtin_threadIdx_t17__fetch_builtin_yEv", scope: !1230, file: !1146, line: 54, type: !1150, scopeLine: 54, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!1234 = !DISubprogram(name: "__fetch_builtin_z", linkageName: "_ZN26__cuda_builtin_threadIdx_t17__fetch_builtin_zEv", scope: !1230, file: !1146, line: 55, type: !1150, scopeLine: 55, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!1235 = !DISubprogram(name: "operator uint3", linkageName: "_ZNK26__cuda_builtin_threadIdx_tcv5uint3Ev", scope: !1230, file: !1146, line: 58, type: !1236, scopeLine: 58, flags: DIFlagPrototyped, spFlags: 0)
!1236 = !DISubroutineType(types: !1237)
!1237 = !{!1157, !1238}
!1238 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !1239, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!1239 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !1230)
!1240 = !DISubprogram(name: "__cuda_builtin_threadIdx_t", scope: !1230, file: !1146, line: 60, type: !1241, scopeLine: 60, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!1241 = !DISubroutineType(types: !1242)
!1242 = !{null, !1243}
!1243 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !1230, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!1244 = !DISubprogram(name: "__cuda_builtin_threadIdx_t", scope: !1230, file: !1146, line: 60, type: !1245, scopeLine: 60, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!1245 = !DISubroutineType(types: !1246)
!1246 = !{null, !1243, !1247}
!1247 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !1239, size: 64)
!1248 = !DISubprogram(name: "operator=", linkageName: "_ZNK26__cuda_builtin_threadIdx_taSERKS_", scope: !1230, file: !1146, line: 60, type: !1249, scopeLine: 60, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!1249 = !DISubroutineType(types: !1250)
!1250 = !{null, !1238, !1247}
!1251 = !DISubprogram(name: "operator&", linkageName: "_ZNK26__cuda_builtin_threadIdx_tadEv", scope: !1230, file: !1146, line: 60, type: !1252, scopeLine: 60, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!1252 = !DISubroutineType(types: !1253)
!1253 = !{!1254, !1238}
!1254 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !1230, size: 64)
!1255 = distinct !DILocation(line: 486, column: 27, scope: !1046)
!1256 = !{i32 0, i32 1024}
!1257 = !DILocation(line: 486, column: 26, scope: !1046)
!1258 = !DILocation(line: 486, column: 4, scope: !1046)
!1259 = !DILocation(line: 488, column: 5, scope: !1260)
!1260 = distinct !DILexicalBlock(scope: !1046, file: !3, line: 488, column: 5)
!1261 = !DILocation(line: 488, column: 7, scope: !1260)
!1262 = !DILocation(line: 488, column: 5, scope: !1046)
!1263 = !DILocation(line: 488, column: 13, scope: !1264)
!1264 = distinct !DILexicalBlock(scope: !1260, file: !3, line: 488, column: 12)
!1265 = !DILocation(line: 490, column: 4, scope: !1046)
!1266 = !DILocation(line: 491, column: 5, scope: !1046)
!1267 = !DILocation(line: 491, column: 4, scope: !1046)
!1268 = !DILocation(line: 494, column: 7, scope: !1269)
!1269 = distinct !DILexicalBlock(scope: !1046, file: !3, line: 494, column: 2)
!1270 = !DILocation(line: 494, column: 6, scope: !1269)
!1271 = !DILocation(line: 494, column: 11, scope: !1272)
!1272 = distinct !DILexicalBlock(scope: !1269, file: !3, line: 494, column: 2)
!1273 = !DILocation(line: 494, column: 12, scope: !1272)
!1274 = !DILocation(line: 494, column: 2, scope: !1269)
!1275 = !DILocation(line: 495, column: 6, scope: !1276)
!1276 = distinct !DILexicalBlock(scope: !1272, file: !3, line: 494, column: 23)
!1277 = !DILocation(line: 495, column: 8, scope: !1276)
!1278 = !DILocation(line: 495, column: 5, scope: !1276)
!1279 = !DILocation(line: 496, column: 9, scope: !1280)
!1280 = distinct !DILexicalBlock(scope: !1276, file: !3, line: 496, column: 6)
!1281 = !DILocation(line: 496, column: 8, scope: !1280)
!1282 = !DILocation(line: 496, column: 14, scope: !1280)
!1283 = !DILocation(line: 496, column: 12, scope: !1280)
!1284 = !DILocation(line: 496, column: 6, scope: !1276)
!1285 = !DILocation(line: 496, column: 40, scope: !1286)
!1286 = distinct !DILexicalBlock(scope: !1280, file: !3, line: 496, column: 17)
!1287 = !DILocation(line: 496, column: 21, scope: !1286)
!1288 = !DILocation(line: 496, column: 20, scope: !1286)
!1289 = !DILocation(line: 496, column: 44, scope: !1286)
!1290 = !DILocation(line: 497, column: 6, scope: !1291)
!1291 = distinct !DILexicalBlock(scope: !1276, file: !3, line: 497, column: 6)
!1292 = !DILocation(line: 497, column: 8, scope: !1291)
!1293 = !DILocation(line: 497, column: 6, scope: !1276)
!1294 = !DILocation(line: 497, column: 13, scope: !1295)
!1295 = distinct !DILexicalBlock(scope: !1291, file: !3, line: 497, column: 12)
!1296 = !DILocation(line: 498, column: 25, scope: !1276)
!1297 = !DILocation(line: 498, column: 6, scope: !1276)
!1298 = !DILocation(line: 498, column: 5, scope: !1276)
!1299 = !DILocation(line: 499, column: 6, scope: !1276)
!1300 = !DILocation(line: 499, column: 5, scope: !1276)
!1301 = !DILocation(line: 500, column: 2, scope: !1276)
!1302 = !DILocation(line: 494, column: 20, scope: !1272)
!1303 = !DILocation(line: 494, column: 2, scope: !1272)
!1304 = distinct !{!1304, !1274, !1305}
!1305 = !DILocation(line: 500, column: 2, scope: !1269)
!1306 = !DILocation(line: 512, column: 7, scope: !1046)
!1307 = !DILocation(line: 512, column: 6, scope: !1046)
!1308 = !DILocation(line: 513, column: 8, scope: !1061)
!1309 = !DILocation(line: 513, column: 6, scope: !1061)
!1310 = !DILocation(line: 513, column: 12, scope: !1060)
!1311 = !DILocation(line: 513, column: 14, scope: !1060)
!1312 = !DILocation(line: 513, column: 2, scope: !1061)
!1313 = !DILocation(line: 515, column: 44, scope: !1059)
!1314 = !DILocation(line: 515, column: 3, scope: !1059)
!1315 = !DILocation(line: 522, column: 8, scope: !1058)
!1316 = !DILocation(line: 522, column: 7, scope: !1058)
!1317 = !DILocation(line: 522, column: 12, scope: !1057)
!1318 = !DILocation(line: 522, column: 13, scope: !1057)
!1319 = !DILocation(line: 522, column: 3, scope: !1058)
!1320 = !DILocation(line: 523, column: 21, scope: !1056)
!1321 = !DILocation(line: 523, column: 20, scope: !1056)
!1322 = !DILocation(line: 523, column: 11, scope: !1056)
!1323 = !DILocation(line: 523, column: 10, scope: !1056)
!1324 = !DILocation(line: 523, column: 23, scope: !1056)
!1325 = !DILocation(line: 523, column: 6, scope: !1056)
!1326 = !DILocation(line: 524, column: 21, scope: !1056)
!1327 = !DILocation(line: 524, column: 20, scope: !1056)
!1328 = !DILocation(line: 524, column: 22, scope: !1056)
!1329 = !DILocation(line: 524, column: 11, scope: !1056)
!1330 = !DILocation(line: 524, column: 10, scope: !1056)
!1331 = !DILocation(line: 524, column: 25, scope: !1056)
!1332 = !DILocation(line: 524, column: 6, scope: !1056)
!1333 = !DILocation(line: 525, column: 7, scope: !1056)
!1334 = !DILocation(line: 525, column: 10, scope: !1056)
!1335 = !DILocation(line: 525, column: 9, scope: !1056)
!1336 = !DILocation(line: 525, column: 13, scope: !1056)
!1337 = !DILocation(line: 525, column: 16, scope: !1056)
!1338 = !DILocation(line: 525, column: 15, scope: !1056)
!1339 = !DILocation(line: 525, column: 12, scope: !1056)
!1340 = !DILocation(line: 525, column: 6, scope: !1056)
!1341 = !DILocation(line: 526, column: 7, scope: !1055)
!1342 = !DILocation(line: 526, column: 9, scope: !1055)
!1343 = !DILocation(line: 526, column: 7, scope: !1056)
!1344 = !DILocation(line: 527, column: 22, scope: !1054)
!1345 = !DILocation(line: 227, column: 19, scope: !1050, inlinedAt: !1053)
!1346 = !DILocation(line: 227, column: 10, scope: !1050, inlinedAt: !1053)
!1347 = !DILocation(line: 527, column: 17, scope: !1054)
!1348 = !DILocation(line: 527, column: 26, scope: !1054)
!1349 = !DILocation(line: 527, column: 25, scope: !1054)
!1350 = !DILocation(line: 894, column: 20, scope: !1063, inlinedAt: !1065)
!1351 = !DILocation(line: 894, column: 10, scope: !1063, inlinedAt: !1065)
!1352 = !DILocation(line: 527, column: 7, scope: !1054)
!1353 = !DILocation(line: 528, column: 12, scope: !1054)
!1354 = !DILocation(line: 528, column: 15, scope: !1054)
!1355 = !DILocation(line: 528, column: 14, scope: !1054)
!1356 = !DILocation(line: 227, column: 19, scope: !1050, inlinedAt: !1067)
!1357 = !DILocation(line: 227, column: 10, scope: !1050, inlinedAt: !1067)
!1358 = !DILocation(line: 528, column: 7, scope: !1054)
!1359 = !DILocation(line: 529, column: 9, scope: !1054)
!1360 = !DILocation(line: 529, column: 12, scope: !1054)
!1361 = !DILocation(line: 529, column: 11, scope: !1054)
!1362 = !DILocation(line: 529, column: 7, scope: !1054)
!1363 = !DILocation(line: 530, column: 7, scope: !1054)
!1364 = !DILocation(line: 589, column: 20, scope: !1069, inlinedAt: !1071)
!1365 = !DILocation(line: 589, column: 10, scope: !1069, inlinedAt: !1071)
!1366 = !DILocation(line: 589, column: 20, scope: !1069, inlinedAt: !1073)
!1367 = !DILocation(line: 589, column: 10, scope: !1069, inlinedAt: !1073)
!1368 = !DILocation(line: 589, column: 20, scope: !1069, inlinedAt: !1075)
!1369 = !DILocation(line: 589, column: 10, scope: !1069, inlinedAt: !1075)
!1370 = !DILocation(line: 589, column: 20, scope: !1069, inlinedAt: !1077)
!1371 = !DILocation(line: 589, column: 10, scope: !1069, inlinedAt: !1077)
!1372 = !DILocation(line: 530, column: 6, scope: !1054)
!1373 = !DILocation(line: 531, column: 13, scope: !1054)
!1374 = !DILocation(line: 531, column: 5, scope: !1054)
!1375 = !DILocation(line: 531, column: 15, scope: !1054)
!1376 = !DILocation(line: 532, column: 14, scope: !1054)
!1377 = !DILocation(line: 532, column: 23, scope: !1054)
!1378 = !DILocation(line: 532, column: 22, scope: !1054)
!1379 = !DILocation(line: 532, column: 13, scope: !1054)
!1380 = !DILocation(line: 533, column: 15, scope: !1054)
!1381 = !DILocation(line: 533, column: 13, scope: !1054)
!1382 = !DILocation(line: 534, column: 4, scope: !1054)
!1383 = !DILocation(line: 535, column: 3, scope: !1056)
!1384 = !DILocation(line: 522, column: 30, scope: !1057)
!1385 = !DILocation(line: 522, column: 3, scope: !1057)
!1386 = distinct !{!1386, !1319, !1387}
!1387 = !DILocation(line: 535, column: 3, scope: !1058)
!1388 = !DILocation(line: 536, column: 2, scope: !1059)
!1389 = !DILocation(line: 513, column: 22, scope: !1060)
!1390 = !DILocation(line: 513, column: 24, scope: !1060)
!1391 = !DILocation(line: 513, column: 21, scope: !1060)
!1392 = !DILocation(line: 513, column: 2, scope: !1060)
!1393 = distinct !{!1393, !1312, !1394}
!1394 = !DILocation(line: 536, column: 2, scope: !1061)
!1395 = !DILocation(line: 550, column: 1, scope: !1046)
!1396 = distinct !DISubprogram(name: "randlc_device", linkageName: "_Z13randlc_devicePdd", scope: !3, file: !3, line: 552, type: !1397, scopeLine: 553, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !961, retainedNodes: !962)
!1397 = !DISubroutineType(types: !1398)
!1398 = !{!98, !97, !98}
!1399 = !DILocalVariable(name: "x", arg: 1, scope: !1396, file: !3, line: 552, type: !97)
!1400 = !DILocation(line: 552, column: 41, scope: !1396)
!1401 = !DILocalVariable(name: "a", arg: 2, scope: !1396, file: !3, line: 553, type: !98)
!1402 = !DILocation(line: 553, column: 10, scope: !1396)
!1403 = !DILocalVariable(name: "t1", scope: !1396, file: !3, line: 554, type: !98)
!1404 = !DILocation(line: 554, column: 9, scope: !1396)
!1405 = !DILocalVariable(name: "t2", scope: !1396, file: !3, line: 554, type: !98)
!1406 = !DILocation(line: 554, column: 12, scope: !1396)
!1407 = !DILocalVariable(name: "t3", scope: !1396, file: !3, line: 554, type: !98)
!1408 = !DILocation(line: 554, column: 15, scope: !1396)
!1409 = !DILocalVariable(name: "t4", scope: !1396, file: !3, line: 554, type: !98)
!1410 = !DILocation(line: 554, column: 18, scope: !1396)
!1411 = !DILocalVariable(name: "a1", scope: !1396, file: !3, line: 554, type: !98)
!1412 = !DILocation(line: 554, column: 21, scope: !1396)
!1413 = !DILocalVariable(name: "a2", scope: !1396, file: !3, line: 554, type: !98)
!1414 = !DILocation(line: 554, column: 24, scope: !1396)
!1415 = !DILocalVariable(name: "x1", scope: !1396, file: !3, line: 554, type: !98)
!1416 = !DILocation(line: 554, column: 27, scope: !1396)
!1417 = !DILocalVariable(name: "x2", scope: !1396, file: !3, line: 554, type: !98)
!1418 = !DILocation(line: 554, column: 30, scope: !1396)
!1419 = !DILocalVariable(name: "z", scope: !1396, file: !3, line: 554, type: !98)
!1420 = !DILocation(line: 554, column: 33, scope: !1396)
!1421 = !DILocation(line: 555, column: 13, scope: !1396)
!1422 = !DILocation(line: 555, column: 11, scope: !1396)
!1423 = !DILocation(line: 555, column: 5, scope: !1396)
!1424 = !DILocation(line: 556, column: 12, scope: !1396)
!1425 = !DILocation(line: 556, column: 7, scope: !1396)
!1426 = !DILocation(line: 556, column: 5, scope: !1396)
!1427 = !DILocation(line: 557, column: 7, scope: !1396)
!1428 = !DILocation(line: 557, column: 17, scope: !1396)
!1429 = !DILocation(line: 557, column: 15, scope: !1396)
!1430 = !DILocation(line: 557, column: 9, scope: !1396)
!1431 = !DILocation(line: 557, column: 5, scope: !1396)
!1432 = !DILocation(line: 558, column: 15, scope: !1396)
!1433 = !DILocation(line: 558, column: 14, scope: !1396)
!1434 = !DILocation(line: 558, column: 11, scope: !1396)
!1435 = !DILocation(line: 558, column: 5, scope: !1396)
!1436 = !DILocation(line: 559, column: 12, scope: !1396)
!1437 = !DILocation(line: 559, column: 7, scope: !1396)
!1438 = !DILocation(line: 559, column: 5, scope: !1396)
!1439 = !DILocation(line: 560, column: 9, scope: !1396)
!1440 = !DILocation(line: 560, column: 8, scope: !1396)
!1441 = !DILocation(line: 560, column: 20, scope: !1396)
!1442 = !DILocation(line: 560, column: 18, scope: !1396)
!1443 = !DILocation(line: 560, column: 12, scope: !1396)
!1444 = !DILocation(line: 560, column: 5, scope: !1396)
!1445 = !DILocation(line: 561, column: 7, scope: !1396)
!1446 = !DILocation(line: 561, column: 12, scope: !1396)
!1447 = !DILocation(line: 561, column: 10, scope: !1396)
!1448 = !DILocation(line: 561, column: 17, scope: !1396)
!1449 = !DILocation(line: 561, column: 22, scope: !1396)
!1450 = !DILocation(line: 561, column: 20, scope: !1396)
!1451 = !DILocation(line: 561, column: 15, scope: !1396)
!1452 = !DILocation(line: 561, column: 5, scope: !1396)
!1453 = !DILocation(line: 562, column: 19, scope: !1396)
!1454 = !DILocation(line: 562, column: 17, scope: !1396)
!1455 = !DILocation(line: 562, column: 12, scope: !1396)
!1456 = !DILocation(line: 562, column: 7, scope: !1396)
!1457 = !DILocation(line: 562, column: 5, scope: !1396)
!1458 = !DILocation(line: 563, column: 6, scope: !1396)
!1459 = !DILocation(line: 563, column: 17, scope: !1396)
!1460 = !DILocation(line: 563, column: 15, scope: !1396)
!1461 = !DILocation(line: 563, column: 9, scope: !1396)
!1462 = !DILocation(line: 563, column: 4, scope: !1396)
!1463 = !DILocation(line: 564, column: 13, scope: !1396)
!1464 = !DILocation(line: 564, column: 11, scope: !1396)
!1465 = !DILocation(line: 564, column: 17, scope: !1396)
!1466 = !DILocation(line: 564, column: 22, scope: !1396)
!1467 = !DILocation(line: 564, column: 20, scope: !1396)
!1468 = !DILocation(line: 564, column: 15, scope: !1396)
!1469 = !DILocation(line: 564, column: 5, scope: !1396)
!1470 = !DILocation(line: 565, column: 19, scope: !1396)
!1471 = !DILocation(line: 565, column: 17, scope: !1396)
!1472 = !DILocation(line: 565, column: 12, scope: !1396)
!1473 = !DILocation(line: 565, column: 7, scope: !1396)
!1474 = !DILocation(line: 565, column: 5, scope: !1396)
!1475 = !DILocation(line: 566, column: 9, scope: !1396)
!1476 = !DILocation(line: 566, column: 20, scope: !1396)
!1477 = !DILocation(line: 566, column: 18, scope: !1396)
!1478 = !DILocation(line: 566, column: 12, scope: !1396)
!1479 = !DILocation(line: 566, column: 4, scope: !1396)
!1480 = !DILocation(line: 566, column: 7, scope: !1396)
!1481 = !DILocation(line: 567, column: 18, scope: !1396)
!1482 = !DILocation(line: 567, column: 17, scope: !1396)
!1483 = !DILocation(line: 567, column: 14, scope: !1396)
!1484 = !DILocation(line: 567, column: 2, scope: !1396)
!1485 = distinct !DISubprogram(name: "vranlc_device", linkageName: "_Z13vranlc_deviceiPddS_", scope: !3, file: !3, line: 646, type: !1486, scopeLine: 649, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !961, retainedNodes: !962)
!1486 = !DISubroutineType(types: !1487)
!1487 = !{null, !99, !97, !98, !97}
!1488 = !DILocalVariable(name: "n", arg: 1, scope: !1485, file: !3, line: 646, type: !99)
!1489 = !DILocation(line: 646, column: 35, scope: !1485)
!1490 = !DILocalVariable(name: "x_seed", arg: 2, scope: !1485, file: !3, line: 647, type: !97)
!1491 = !DILocation(line: 647, column: 11, scope: !1485)
!1492 = !DILocalVariable(name: "a", arg: 3, scope: !1485, file: !3, line: 648, type: !98)
!1493 = !DILocation(line: 648, column: 10, scope: !1485)
!1494 = !DILocalVariable(name: "y", arg: 4, scope: !1485, file: !3, line: 649, type: !97)
!1495 = !DILocation(line: 649, column: 11, scope: !1485)
!1496 = !DILocalVariable(name: "i", scope: !1485, file: !3, line: 650, type: !99)
!1497 = !DILocation(line: 650, column: 6, scope: !1485)
!1498 = !DILocalVariable(name: "x", scope: !1485, file: !3, line: 651, type: !98)
!1499 = !DILocation(line: 651, column: 9, scope: !1485)
!1500 = !DILocalVariable(name: "t1", scope: !1485, file: !3, line: 651, type: !98)
!1501 = !DILocation(line: 651, column: 11, scope: !1485)
!1502 = !DILocalVariable(name: "t2", scope: !1485, file: !3, line: 651, type: !98)
!1503 = !DILocation(line: 651, column: 14, scope: !1485)
!1504 = !DILocalVariable(name: "t3", scope: !1485, file: !3, line: 651, type: !98)
!1505 = !DILocation(line: 651, column: 17, scope: !1485)
!1506 = !DILocalVariable(name: "t4", scope: !1485, file: !3, line: 651, type: !98)
!1507 = !DILocation(line: 651, column: 20, scope: !1485)
!1508 = !DILocalVariable(name: "a1", scope: !1485, file: !3, line: 651, type: !98)
!1509 = !DILocation(line: 651, column: 23, scope: !1485)
!1510 = !DILocalVariable(name: "a2", scope: !1485, file: !3, line: 651, type: !98)
!1511 = !DILocation(line: 651, column: 26, scope: !1485)
!1512 = !DILocalVariable(name: "x1", scope: !1485, file: !3, line: 651, type: !98)
!1513 = !DILocation(line: 651, column: 29, scope: !1485)
!1514 = !DILocalVariable(name: "x2", scope: !1485, file: !3, line: 651, type: !98)
!1515 = !DILocation(line: 651, column: 32, scope: !1485)
!1516 = !DILocalVariable(name: "z", scope: !1485, file: !3, line: 651, type: !98)
!1517 = !DILocation(line: 651, column: 35, scope: !1485)
!1518 = !DILocation(line: 652, column: 13, scope: !1485)
!1519 = !DILocation(line: 652, column: 11, scope: !1485)
!1520 = !DILocation(line: 652, column: 5, scope: !1485)
!1521 = !DILocation(line: 653, column: 12, scope: !1485)
!1522 = !DILocation(line: 653, column: 7, scope: !1485)
!1523 = !DILocation(line: 653, column: 5, scope: !1485)
!1524 = !DILocation(line: 654, column: 7, scope: !1485)
!1525 = !DILocation(line: 654, column: 17, scope: !1485)
!1526 = !DILocation(line: 654, column: 15, scope: !1485)
!1527 = !DILocation(line: 654, column: 9, scope: !1485)
!1528 = !DILocation(line: 654, column: 5, scope: !1485)
!1529 = !DILocation(line: 655, column: 7, scope: !1485)
!1530 = !DILocation(line: 655, column: 6, scope: !1485)
!1531 = !DILocation(line: 655, column: 4, scope: !1485)
!1532 = !DILocation(line: 656, column: 7, scope: !1533)
!1533 = distinct !DILexicalBlock(scope: !1485, file: !3, line: 656, column: 2)
!1534 = !DILocation(line: 656, column: 6, scope: !1533)
!1535 = !DILocation(line: 656, column: 11, scope: !1536)
!1536 = distinct !DILexicalBlock(scope: !1533, file: !3, line: 656, column: 2)
!1537 = !DILocation(line: 656, column: 13, scope: !1536)
!1538 = !DILocation(line: 656, column: 12, scope: !1536)
!1539 = !DILocation(line: 656, column: 2, scope: !1533)
!1540 = !DILocation(line: 657, column: 14, scope: !1541)
!1541 = distinct !DILexicalBlock(scope: !1536, file: !3, line: 656, column: 20)
!1542 = !DILocation(line: 657, column: 12, scope: !1541)
!1543 = !DILocation(line: 657, column: 6, scope: !1541)
!1544 = !DILocation(line: 658, column: 13, scope: !1541)
!1545 = !DILocation(line: 658, column: 8, scope: !1541)
!1546 = !DILocation(line: 658, column: 6, scope: !1541)
!1547 = !DILocation(line: 659, column: 8, scope: !1541)
!1548 = !DILocation(line: 659, column: 18, scope: !1541)
!1549 = !DILocation(line: 659, column: 16, scope: !1541)
!1550 = !DILocation(line: 659, column: 10, scope: !1541)
!1551 = !DILocation(line: 659, column: 6, scope: !1541)
!1552 = !DILocation(line: 660, column: 8, scope: !1541)
!1553 = !DILocation(line: 660, column: 13, scope: !1541)
!1554 = !DILocation(line: 660, column: 11, scope: !1541)
!1555 = !DILocation(line: 660, column: 18, scope: !1541)
!1556 = !DILocation(line: 660, column: 23, scope: !1541)
!1557 = !DILocation(line: 660, column: 21, scope: !1541)
!1558 = !DILocation(line: 660, column: 16, scope: !1541)
!1559 = !DILocation(line: 660, column: 6, scope: !1541)
!1560 = !DILocation(line: 661, column: 20, scope: !1541)
!1561 = !DILocation(line: 661, column: 18, scope: !1541)
!1562 = !DILocation(line: 661, column: 13, scope: !1541)
!1563 = !DILocation(line: 661, column: 8, scope: !1541)
!1564 = !DILocation(line: 661, column: 6, scope: !1541)
!1565 = !DILocation(line: 662, column: 7, scope: !1541)
!1566 = !DILocation(line: 662, column: 18, scope: !1541)
!1567 = !DILocation(line: 662, column: 16, scope: !1541)
!1568 = !DILocation(line: 662, column: 10, scope: !1541)
!1569 = !DILocation(line: 662, column: 5, scope: !1541)
!1570 = !DILocation(line: 663, column: 14, scope: !1541)
!1571 = !DILocation(line: 663, column: 12, scope: !1541)
!1572 = !DILocation(line: 663, column: 18, scope: !1541)
!1573 = !DILocation(line: 663, column: 23, scope: !1541)
!1574 = !DILocation(line: 663, column: 21, scope: !1541)
!1575 = !DILocation(line: 663, column: 16, scope: !1541)
!1576 = !DILocation(line: 663, column: 6, scope: !1541)
!1577 = !DILocation(line: 664, column: 20, scope: !1541)
!1578 = !DILocation(line: 664, column: 18, scope: !1541)
!1579 = !DILocation(line: 664, column: 13, scope: !1541)
!1580 = !DILocation(line: 664, column: 8, scope: !1541)
!1581 = !DILocation(line: 664, column: 6, scope: !1541)
!1582 = !DILocation(line: 665, column: 7, scope: !1541)
!1583 = !DILocation(line: 665, column: 18, scope: !1541)
!1584 = !DILocation(line: 665, column: 16, scope: !1541)
!1585 = !DILocation(line: 665, column: 10, scope: !1541)
!1586 = !DILocation(line: 665, column: 5, scope: !1541)
!1587 = !DILocation(line: 666, column: 16, scope: !1541)
!1588 = !DILocation(line: 666, column: 14, scope: !1541)
!1589 = !DILocation(line: 666, column: 3, scope: !1541)
!1590 = !DILocation(line: 666, column: 5, scope: !1541)
!1591 = !DILocation(line: 666, column: 8, scope: !1541)
!1592 = !DILocation(line: 667, column: 2, scope: !1541)
!1593 = !DILocation(line: 656, column: 17, scope: !1536)
!1594 = !DILocation(line: 656, column: 2, scope: !1536)
!1595 = distinct !{!1595, !1539, !1596}
!1596 = !DILocation(line: 667, column: 2, scope: !1533)
!1597 = !DILocation(line: 668, column: 12, scope: !1485)
!1598 = !DILocation(line: 668, column: 3, scope: !1485)
!1599 = !DILocation(line: 668, column: 10, scope: !1485)
!1600 = !DILocation(line: 669, column: 1, scope: !1485)
!1601 = distinct !DISubprogram(linkageName: "_GLOBAL__sub_I_ep.cu", scope: !3, file: !3, type: !1602, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !962)
!1602 = !DISubroutineType(types: !962)
!1603 = !DILocation(line: 0, scope: !1601)
!1604 = distinct !DISubprogram(name: "__cxx_global_var_init", scope: !3, file: !3, line: 81, type: !472, scopeLine: 81, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !962)
!1605 = !DILocation(line: 81, column: 29, scope: !1604)
!1606 = !DILocation(line: 81, column: 20, scope: !1604)
!1607 = !DILocation(line: 81, column: 52, scope: !1604)
!1608 = distinct !DISubprogram(name: "randlc", linkageName: "_Z6randlcPdd", scope: !3, file: !3, line: 156, type: !1397, scopeLine: 156, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !962)
!1609 = !DILocalVariable(name: "x", arg: 1, scope: !1608, file: !3, line: 156, type: !97)
!1610 = !DILocation(line: 156, column: 23, scope: !1608)
!1611 = !DILocalVariable(name: "a", arg: 2, scope: !1608, file: !3, line: 156, type: !98)
!1612 = !DILocation(line: 156, column: 33, scope: !1608)
!1613 = !DILocalVariable(name: "t1", scope: !1608, file: !3, line: 157, type: !98)
!1614 = !DILocation(line: 157, column: 9, scope: !1608)
!1615 = !DILocalVariable(name: "t2", scope: !1608, file: !3, line: 157, type: !98)
!1616 = !DILocation(line: 157, column: 12, scope: !1608)
!1617 = !DILocalVariable(name: "t3", scope: !1608, file: !3, line: 157, type: !98)
!1618 = !DILocation(line: 157, column: 15, scope: !1608)
!1619 = !DILocalVariable(name: "t4", scope: !1608, file: !3, line: 157, type: !98)
!1620 = !DILocation(line: 157, column: 18, scope: !1608)
!1621 = !DILocalVariable(name: "a1", scope: !1608, file: !3, line: 157, type: !98)
!1622 = !DILocation(line: 157, column: 21, scope: !1608)
!1623 = !DILocalVariable(name: "a2", scope: !1608, file: !3, line: 157, type: !98)
!1624 = !DILocation(line: 157, column: 24, scope: !1608)
!1625 = !DILocalVariable(name: "x1", scope: !1608, file: !3, line: 157, type: !98)
!1626 = !DILocation(line: 157, column: 27, scope: !1608)
!1627 = !DILocalVariable(name: "x2", scope: !1608, file: !3, line: 157, type: !98)
!1628 = !DILocation(line: 157, column: 30, scope: !1608)
!1629 = !DILocalVariable(name: "z", scope: !1608, file: !3, line: 157, type: !98)
!1630 = !DILocation(line: 157, column: 33, scope: !1608)
!1631 = !DILocation(line: 164, column: 13, scope: !1608)
!1632 = !DILocation(line: 164, column: 11, scope: !1608)
!1633 = !DILocation(line: 164, column: 5, scope: !1608)
!1634 = !DILocation(line: 165, column: 12, scope: !1608)
!1635 = !DILocation(line: 165, column: 7, scope: !1608)
!1636 = !DILocation(line: 165, column: 5, scope: !1608)
!1637 = !DILocation(line: 166, column: 7, scope: !1608)
!1638 = !DILocation(line: 166, column: 17, scope: !1608)
!1639 = !DILocation(line: 166, column: 15, scope: !1608)
!1640 = !DILocation(line: 166, column: 9, scope: !1608)
!1641 = !DILocation(line: 166, column: 5, scope: !1608)
!1642 = !DILocation(line: 175, column: 15, scope: !1608)
!1643 = !DILocation(line: 175, column: 14, scope: !1608)
!1644 = !DILocation(line: 175, column: 11, scope: !1608)
!1645 = !DILocation(line: 175, column: 5, scope: !1608)
!1646 = !DILocation(line: 176, column: 12, scope: !1608)
!1647 = !DILocation(line: 176, column: 7, scope: !1608)
!1648 = !DILocation(line: 176, column: 5, scope: !1608)
!1649 = !DILocation(line: 177, column: 9, scope: !1608)
!1650 = !DILocation(line: 177, column: 8, scope: !1608)
!1651 = !DILocation(line: 177, column: 20, scope: !1608)
!1652 = !DILocation(line: 177, column: 18, scope: !1608)
!1653 = !DILocation(line: 177, column: 12, scope: !1608)
!1654 = !DILocation(line: 177, column: 5, scope: !1608)
!1655 = !DILocation(line: 178, column: 7, scope: !1608)
!1656 = !DILocation(line: 178, column: 12, scope: !1608)
!1657 = !DILocation(line: 178, column: 10, scope: !1608)
!1658 = !DILocation(line: 178, column: 17, scope: !1608)
!1659 = !DILocation(line: 178, column: 22, scope: !1608)
!1660 = !DILocation(line: 178, column: 20, scope: !1608)
!1661 = !DILocation(line: 178, column: 15, scope: !1608)
!1662 = !DILocation(line: 178, column: 5, scope: !1608)
!1663 = !DILocation(line: 179, column: 19, scope: !1608)
!1664 = !DILocation(line: 179, column: 17, scope: !1608)
!1665 = !DILocation(line: 179, column: 12, scope: !1608)
!1666 = !DILocation(line: 179, column: 7, scope: !1608)
!1667 = !DILocation(line: 179, column: 5, scope: !1608)
!1668 = !DILocation(line: 180, column: 6, scope: !1608)
!1669 = !DILocation(line: 180, column: 17, scope: !1608)
!1670 = !DILocation(line: 180, column: 15, scope: !1608)
!1671 = !DILocation(line: 180, column: 9, scope: !1608)
!1672 = !DILocation(line: 180, column: 4, scope: !1608)
!1673 = !DILocation(line: 181, column: 13, scope: !1608)
!1674 = !DILocation(line: 181, column: 11, scope: !1608)
!1675 = !DILocation(line: 181, column: 17, scope: !1608)
!1676 = !DILocation(line: 181, column: 22, scope: !1608)
!1677 = !DILocation(line: 181, column: 20, scope: !1608)
!1678 = !DILocation(line: 181, column: 15, scope: !1608)
!1679 = !DILocation(line: 181, column: 5, scope: !1608)
!1680 = !DILocation(line: 182, column: 19, scope: !1608)
!1681 = !DILocation(line: 182, column: 17, scope: !1608)
!1682 = !DILocation(line: 182, column: 12, scope: !1608)
!1683 = !DILocation(line: 182, column: 7, scope: !1608)
!1684 = !DILocation(line: 182, column: 5, scope: !1608)
!1685 = !DILocation(line: 183, column: 9, scope: !1608)
!1686 = !DILocation(line: 183, column: 20, scope: !1608)
!1687 = !DILocation(line: 183, column: 18, scope: !1608)
!1688 = !DILocation(line: 183, column: 12, scope: !1608)
!1689 = !DILocation(line: 183, column: 4, scope: !1608)
!1690 = !DILocation(line: 183, column: 7, scope: !1608)
!1691 = !DILocation(line: 185, column: 18, scope: !1608)
!1692 = !DILocation(line: 185, column: 17, scope: !1608)
!1693 = !DILocation(line: 185, column: 14, scope: !1608)
!1694 = !DILocation(line: 185, column: 2, scope: !1608)
!1695 = distinct !DISubprogram(name: "c_print_results", linkageName: "_Z15c_print_resultsPcciiiiddS_iS_S_S_S_S_S_S_S_S_S_S_S_S_S_", scope: !3, file: !3, line: 191, type: !1696, scopeLine: 214, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !962)
!1696 = !DISubroutineType(types: !1697)
!1697 = !{null, !100, !101, !99, !99, !99, !99, !98, !98, !100, !99, !100, !100, !100, !100, !100, !100, !100, !100, !100, !100, !100, !100, !100, !100}
!1698 = !DILocalVariable(name: "name", arg: 1, scope: !1695, file: !3, line: 191, type: !100)
!1699 = !DILocation(line: 191, column: 28, scope: !1695)
!1700 = !DILocalVariable(name: "class_npb", arg: 2, scope: !1695, file: !3, line: 192, type: !101)
!1701 = !DILocation(line: 192, column: 8, scope: !1695)
!1702 = !DILocalVariable(name: "n1", arg: 3, scope: !1695, file: !3, line: 193, type: !99)
!1703 = !DILocation(line: 193, column: 7, scope: !1695)
!1704 = !DILocalVariable(name: "n2", arg: 4, scope: !1695, file: !3, line: 194, type: !99)
!1705 = !DILocation(line: 194, column: 7, scope: !1695)
!1706 = !DILocalVariable(name: "n3", arg: 5, scope: !1695, file: !3, line: 195, type: !99)
!1707 = !DILocation(line: 195, column: 7, scope: !1695)
!1708 = !DILocalVariable(name: "niter", arg: 6, scope: !1695, file: !3, line: 196, type: !99)
!1709 = !DILocation(line: 196, column: 7, scope: !1695)
!1710 = !DILocalVariable(name: "t", arg: 7, scope: !1695, file: !3, line: 197, type: !98)
!1711 = !DILocation(line: 197, column: 10, scope: !1695)
!1712 = !DILocalVariable(name: "mops", arg: 8, scope: !1695, file: !3, line: 198, type: !98)
!1713 = !DILocation(line: 198, column: 10, scope: !1695)
!1714 = !DILocalVariable(name: "optype", arg: 9, scope: !1695, file: !3, line: 199, type: !100)
!1715 = !DILocation(line: 199, column: 9, scope: !1695)
!1716 = !DILocalVariable(name: "passed_verification", arg: 10, scope: !1695, file: !3, line: 200, type: !99)
!1717 = !DILocation(line: 200, column: 7, scope: !1695)
!1718 = !DILocalVariable(name: "npbversion", arg: 11, scope: !1695, file: !3, line: 201, type: !100)
!1719 = !DILocation(line: 201, column: 9, scope: !1695)
!1720 = !DILocalVariable(name: "compiletime", arg: 12, scope: !1695, file: !3, line: 202, type: !100)
!1721 = !DILocation(line: 202, column: 9, scope: !1695)
!1722 = !DILocalVariable(name: "compilerversion", arg: 13, scope: !1695, file: !3, line: 203, type: !100)
!1723 = !DILocation(line: 203, column: 9, scope: !1695)
!1724 = !DILocalVariable(name: "libversion", arg: 14, scope: !1695, file: !3, line: 204, type: !100)
!1725 = !DILocation(line: 204, column: 9, scope: !1695)
!1726 = !DILocalVariable(name: "cpu_device", arg: 15, scope: !1695, file: !3, line: 205, type: !100)
!1727 = !DILocation(line: 205, column: 9, scope: !1695)
!1728 = !DILocalVariable(name: "gpu_device", arg: 16, scope: !1695, file: !3, line: 206, type: !100)
!1729 = !DILocation(line: 206, column: 9, scope: !1695)
!1730 = !DILocalVariable(name: "gpu_config", arg: 17, scope: !1695, file: !3, line: 207, type: !100)
!1731 = !DILocation(line: 207, column: 9, scope: !1695)
!1732 = !DILocalVariable(name: "cc", arg: 18, scope: !1695, file: !3, line: 208, type: !100)
!1733 = !DILocation(line: 208, column: 9, scope: !1695)
!1734 = !DILocalVariable(name: "clink", arg: 19, scope: !1695, file: !3, line: 209, type: !100)
!1735 = !DILocation(line: 209, column: 9, scope: !1695)
!1736 = !DILocalVariable(name: "c_lib", arg: 20, scope: !1695, file: !3, line: 210, type: !100)
!1737 = !DILocation(line: 210, column: 9, scope: !1695)
!1738 = !DILocalVariable(name: "c_inc", arg: 21, scope: !1695, file: !3, line: 211, type: !100)
!1739 = !DILocation(line: 211, column: 9, scope: !1695)
!1740 = !DILocalVariable(name: "cflags", arg: 22, scope: !1695, file: !3, line: 212, type: !100)
!1741 = !DILocation(line: 212, column: 9, scope: !1695)
!1742 = !DILocalVariable(name: "clinkflags", arg: 23, scope: !1695, file: !3, line: 213, type: !100)
!1743 = !DILocation(line: 213, column: 9, scope: !1695)
!1744 = !DILocalVariable(name: "rand", arg: 24, scope: !1695, file: !3, line: 214, type: !100)
!1745 = !DILocation(line: 214, column: 9, scope: !1695)
!1746 = !DILocation(line: 215, column: 44, scope: !1695)
!1747 = !DILocation(line: 215, column: 4, scope: !1695)
!1748 = !DILocation(line: 216, column: 61, scope: !1695)
!1749 = !DILocation(line: 216, column: 4, scope: !1695)
!1750 = !DILocation(line: 217, column: 8, scope: !1751)
!1751 = distinct !DILexicalBlock(scope: !1695, file: !3, line: 217, column: 7)
!1752 = !DILocation(line: 217, column: 15, scope: !1751)
!1753 = !DILocation(line: 217, column: 21, scope: !1751)
!1754 = !DILocation(line: 217, column: 24, scope: !1751)
!1755 = !DILocation(line: 217, column: 31, scope: !1751)
!1756 = !DILocation(line: 217, column: 7, scope: !1695)
!1757 = !DILocation(line: 218, column: 8, scope: !1758)
!1758 = distinct !DILexicalBlock(scope: !1759, file: !3, line: 218, column: 8)
!1759 = distinct !DILexicalBlock(scope: !1751, file: !3, line: 217, column: 38)
!1760 = !DILocation(line: 218, column: 10, scope: !1758)
!1761 = !DILocation(line: 218, column: 8, scope: !1759)
!1762 = !DILocalVariable(name: "nn", scope: !1763, file: !3, line: 219, type: !313)
!1763 = distinct !DILexicalBlock(scope: !1758, file: !3, line: 218, column: 14)
!1764 = !DILocation(line: 219, column: 11, scope: !1763)
!1765 = !DILocation(line: 219, column: 16, scope: !1763)
!1766 = !DILocation(line: 220, column: 9, scope: !1767)
!1767 = distinct !DILexicalBlock(scope: !1763, file: !3, line: 220, column: 9)
!1768 = !DILocation(line: 220, column: 11, scope: !1767)
!1769 = !DILocation(line: 220, column: 9, scope: !1763)
!1770 = !DILocation(line: 220, column: 20, scope: !1771)
!1771 = distinct !DILexicalBlock(scope: !1767, file: !3, line: 220, column: 15)
!1772 = !DILocation(line: 220, column: 18, scope: !1771)
!1773 = !DILocation(line: 220, column: 23, scope: !1771)
!1774 = !DILocation(line: 221, column: 55, scope: !1763)
!1775 = !DILocation(line: 221, column: 6, scope: !1763)
!1776 = !DILocation(line: 222, column: 5, scope: !1763)
!1777 = !DILocation(line: 223, column: 61, scope: !1778)
!1778 = distinct !DILexicalBlock(scope: !1758, file: !3, line: 222, column: 10)
!1779 = !DILocation(line: 223, column: 64, scope: !1778)
!1780 = !DILocation(line: 223, column: 67, scope: !1778)
!1781 = !DILocation(line: 223, column: 6, scope: !1778)
!1782 = !DILocation(line: 225, column: 4, scope: !1759)
!1783 = !DILocalVariable(name: "size", scope: !1784, file: !3, line: 226, type: !1785)
!1784 = distinct !DILexicalBlock(scope: !1751, file: !3, line: 225, column: 9)
!1785 = !DICompositeType(tag: DW_TAG_array_type, baseType: !101, size: 128, elements: !1786)
!1786 = !{!1787}
!1787 = !DISubrange(count: 16)
!1788 = !DILocation(line: 226, column: 10, scope: !1784)
!1789 = !DILocalVariable(name: "j", scope: !1784, file: !3, line: 227, type: !99)
!1790 = !DILocation(line: 227, column: 9, scope: !1784)
!1791 = !DILocation(line: 228, column: 9, scope: !1792)
!1792 = distinct !DILexicalBlock(scope: !1784, file: !3, line: 228, column: 8)
!1793 = !DILocation(line: 228, column: 11, scope: !1792)
!1794 = !DILocation(line: 228, column: 16, scope: !1792)
!1795 = !DILocation(line: 228, column: 20, scope: !1792)
!1796 = !DILocation(line: 228, column: 22, scope: !1792)
!1797 = !DILocation(line: 228, column: 8, scope: !1784)
!1798 = !DILocation(line: 229, column: 10, scope: !1799)
!1799 = distinct !DILexicalBlock(scope: !1800, file: !3, line: 229, column: 9)
!1800 = distinct !DILexicalBlock(scope: !1792, file: !3, line: 228, column: 27)
!1801 = !DILocation(line: 229, column: 17, scope: !1799)
!1802 = !DILocation(line: 229, column: 23, scope: !1799)
!1803 = !DILocation(line: 229, column: 26, scope: !1799)
!1804 = !DILocation(line: 229, column: 33, scope: !1799)
!1805 = !DILocation(line: 229, column: 9, scope: !1800)
!1806 = !DILocation(line: 230, column: 15, scope: !1807)
!1807 = distinct !DILexicalBlock(scope: !1799, file: !3, line: 229, column: 40)
!1808 = !DILocation(line: 230, column: 41, scope: !1807)
!1809 = !DILocation(line: 230, column: 32, scope: !1807)
!1810 = !DILocation(line: 230, column: 7, scope: !1807)
!1811 = !DILocation(line: 231, column: 9, scope: !1807)
!1812 = !DILocation(line: 232, column: 15, scope: !1813)
!1813 = distinct !DILexicalBlock(scope: !1807, file: !3, line: 232, column: 10)
!1814 = !DILocation(line: 232, column: 10, scope: !1813)
!1815 = !DILocation(line: 232, column: 18, scope: !1813)
!1816 = !DILocation(line: 232, column: 10, scope: !1807)
!1817 = !DILocation(line: 233, column: 13, scope: !1818)
!1818 = distinct !DILexicalBlock(scope: !1813, file: !3, line: 232, column: 25)
!1819 = !DILocation(line: 233, column: 8, scope: !1818)
!1820 = !DILocation(line: 233, column: 16, scope: !1818)
!1821 = !DILocation(line: 234, column: 9, scope: !1818)
!1822 = !DILocation(line: 235, column: 7, scope: !1818)
!1823 = !DILocation(line: 236, column: 12, scope: !1807)
!1824 = !DILocation(line: 236, column: 13, scope: !1807)
!1825 = !DILocation(line: 236, column: 7, scope: !1807)
!1826 = !DILocation(line: 236, column: 17, scope: !1807)
!1827 = !DILocation(line: 237, column: 52, scope: !1807)
!1828 = !DILocation(line: 237, column: 7, scope: !1807)
!1829 = !DILocation(line: 238, column: 6, scope: !1807)
!1830 = !DILocation(line: 239, column: 55, scope: !1831)
!1831 = distinct !DILexicalBlock(scope: !1799, file: !3, line: 238, column: 11)
!1832 = !DILocation(line: 239, column: 7, scope: !1831)
!1833 = !DILocation(line: 241, column: 5, scope: !1800)
!1834 = !DILocation(line: 242, column: 59, scope: !1835)
!1835 = distinct !DILexicalBlock(scope: !1792, file: !3, line: 241, column: 10)
!1836 = !DILocation(line: 242, column: 63, scope: !1835)
!1837 = !DILocation(line: 242, column: 67, scope: !1835)
!1838 = !DILocation(line: 242, column: 6, scope: !1835)
!1839 = !DILocation(line: 245, column: 52, scope: !1695)
!1840 = !DILocation(line: 245, column: 4, scope: !1695)
!1841 = !DILocation(line: 246, column: 54, scope: !1695)
!1842 = !DILocation(line: 246, column: 4, scope: !1695)
!1843 = !DILocation(line: 247, column: 54, scope: !1695)
!1844 = !DILocation(line: 247, column: 4, scope: !1695)
!1845 = !DILocation(line: 248, column: 40, scope: !1695)
!1846 = !DILocation(line: 248, column: 4, scope: !1695)
!1847 = !DILocation(line: 249, column: 7, scope: !1848)
!1848 = distinct !DILexicalBlock(scope: !1695, file: !3, line: 249, column: 7)
!1849 = !DILocation(line: 249, column: 27, scope: !1848)
!1850 = !DILocation(line: 249, column: 7, scope: !1695)
!1851 = !DILocation(line: 250, column: 5, scope: !1852)
!1852 = distinct !DILexicalBlock(scope: !1848, file: !3, line: 249, column: 31)
!1853 = !DILocation(line: 251, column: 4, scope: !1852)
!1854 = !DILocation(line: 251, column: 13, scope: !1855)
!1855 = distinct !DILexicalBlock(scope: !1848, file: !3, line: 251, column: 13)
!1856 = !DILocation(line: 251, column: 13, scope: !1848)
!1857 = !DILocation(line: 252, column: 5, scope: !1858)
!1858 = distinct !DILexicalBlock(scope: !1855, file: !3, line: 251, column: 33)
!1859 = !DILocation(line: 253, column: 4, scope: !1858)
!1860 = !DILocation(line: 254, column: 5, scope: !1861)
!1861 = distinct !DILexicalBlock(scope: !1855, file: !3, line: 253, column: 9)
!1862 = !DILocation(line: 256, column: 52, scope: !1695)
!1863 = !DILocation(line: 256, column: 4, scope: !1695)
!1864 = !DILocation(line: 257, column: 52, scope: !1695)
!1865 = !DILocation(line: 257, column: 4, scope: !1695)
!1866 = !DILocation(line: 258, column: 52, scope: !1695)
!1867 = !DILocation(line: 258, column: 4, scope: !1695)
!1868 = !DILocation(line: 259, column: 52, scope: !1695)
!1869 = !DILocation(line: 259, column: 4, scope: !1695)
!1870 = !DILocation(line: 260, column: 4, scope: !1695)
!1871 = !DILocation(line: 261, column: 38, scope: !1695)
!1872 = !DILocation(line: 261, column: 4, scope: !1695)
!1873 = !DILocation(line: 262, column: 38, scope: !1695)
!1874 = !DILocation(line: 262, column: 4, scope: !1695)
!1875 = !DILocation(line: 263, column: 38, scope: !1695)
!1876 = !DILocation(line: 263, column: 4, scope: !1695)
!1877 = !DILocation(line: 264, column: 38, scope: !1695)
!1878 = !DILocation(line: 264, column: 4, scope: !1695)
!1879 = !DILocation(line: 265, column: 38, scope: !1695)
!1880 = !DILocation(line: 265, column: 4, scope: !1695)
!1881 = !DILocation(line: 266, column: 38, scope: !1695)
!1882 = !DILocation(line: 266, column: 4, scope: !1695)
!1883 = !DILocation(line: 267, column: 38, scope: !1695)
!1884 = !DILocation(line: 267, column: 4, scope: !1695)
!1885 = !DILocation(line: 268, column: 4, scope: !1695)
!1886 = !DILocation(line: 269, column: 38, scope: !1695)
!1887 = !DILocation(line: 269, column: 4, scope: !1695)
!1888 = !DILocation(line: 270, column: 38, scope: !1695)
!1889 = !DILocation(line: 270, column: 4, scope: !1695)
!1890 = !DILocation(line: 271, column: 4, scope: !1695)
!1891 = !DILocation(line: 272, column: 38, scope: !1695)
!1892 = !DILocation(line: 272, column: 4, scope: !1695)
!1893 = !DILocation(line: 287, column: 4, scope: !1695)
!1894 = !DILocation(line: 288, column: 4, scope: !1695)
!1895 = !DILocation(line: 289, column: 4, scope: !1695)
!1896 = !DILocation(line: 290, column: 4, scope: !1695)
!1897 = !DILocation(line: 291, column: 4, scope: !1695)
!1898 = !DILocation(line: 292, column: 4, scope: !1695)
!1899 = !DILocation(line: 293, column: 4, scope: !1695)
!1900 = !DILocation(line: 294, column: 4, scope: !1695)
!1901 = !DILocation(line: 295, column: 4, scope: !1695)
!1902 = !DILocation(line: 296, column: 4, scope: !1695)
!1903 = !DILocation(line: 297, column: 3, scope: !1695)
!1904 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 301, type: !1905, scopeLine: 301, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !962)
!1905 = !DISubroutineType(types: !1906)
!1906 = !{!99, !99, !566}
!1907 = !DILocalVariable(name: "argc", arg: 1, scope: !1904, file: !3, line: 301, type: !99)
!1908 = !DILocation(line: 301, column: 14, scope: !1904)
!1909 = !DILocalVariable(name: "argv", arg: 2, scope: !1904, file: !3, line: 301, type: !566)
!1910 = !DILocation(line: 301, column: 27, scope: !1904)
!1911 = !DILocalVariable(name: "Mops", scope: !1904, file: !3, line: 308, type: !98)
!1912 = !DILocation(line: 308, column: 9, scope: !1904)
!1913 = !DILocalVariable(name: "t1", scope: !1904, file: !3, line: 308, type: !98)
!1914 = !DILocation(line: 308, column: 15, scope: !1904)
!1915 = !DILocalVariable(name: "sx", scope: !1904, file: !3, line: 309, type: !98)
!1916 = !DILocation(line: 309, column: 9, scope: !1904)
!1917 = !DILocalVariable(name: "sy", scope: !1904, file: !3, line: 309, type: !98)
!1918 = !DILocation(line: 309, column: 13, scope: !1904)
!1919 = !DILocalVariable(name: "tm", scope: !1904, file: !3, line: 309, type: !98)
!1920 = !DILocation(line: 309, column: 17, scope: !1904)
!1921 = !DILocalVariable(name: "an", scope: !1904, file: !3, line: 309, type: !98)
!1922 = !DILocation(line: 309, column: 21, scope: !1904)
!1923 = !DILocalVariable(name: "gc", scope: !1904, file: !3, line: 309, type: !98)
!1924 = !DILocation(line: 309, column: 25, scope: !1904)
!1925 = !DILocalVariable(name: "sx_verify_value", scope: !1904, file: !3, line: 310, type: !98)
!1926 = !DILocation(line: 310, column: 9, scope: !1904)
!1927 = !DILocalVariable(name: "sy_verify_value", scope: !1904, file: !3, line: 310, type: !98)
!1928 = !DILocation(line: 310, column: 26, scope: !1904)
!1929 = !DILocalVariable(name: "sx_err", scope: !1904, file: !3, line: 310, type: !98)
!1930 = !DILocation(line: 310, column: 43, scope: !1904)
!1931 = !DILocalVariable(name: "sy_err", scope: !1904, file: !3, line: 310, type: !98)
!1932 = !DILocation(line: 310, column: 51, scope: !1904)
!1933 = !DILocalVariable(name: "i", scope: !1904, file: !3, line: 311, type: !99)
!1934 = !DILocation(line: 311, column: 6, scope: !1904)
!1935 = !DILocalVariable(name: "j", scope: !1904, file: !3, line: 311, type: !99)
!1936 = !DILocation(line: 311, column: 9, scope: !1904)
!1937 = !DILocalVariable(name: "nit", scope: !1904, file: !3, line: 311, type: !99)
!1938 = !DILocation(line: 311, column: 12, scope: !1904)
!1939 = !DILocalVariable(name: "block", scope: !1904, file: !3, line: 311, type: !99)
!1940 = !DILocation(line: 311, column: 17, scope: !1904)
!1941 = !DILocalVariable(name: "verified", scope: !1904, file: !3, line: 312, type: !1942)
!1942 = !DIDerivedType(tag: DW_TAG_typedef, name: "boolean", file: !1943, line: 80, baseType: !99)
!1943 = !DIFile(filename: "./../common/npb-CPP.hpp", directory: "/scratch/ah7226/NPB-GPU/CUDA/EP")
!1944 = !DILocation(line: 312, column: 10, scope: !1904)
!1945 = !DILocalVariable(name: "size", scope: !1904, file: !3, line: 313, type: !1785)
!1946 = !DILocation(line: 313, column: 7, scope: !1904)
!1947 = !DILocation(line: 323, column: 10, scope: !1904)
!1948 = !DILocation(line: 323, column: 26, scope: !1904)
!1949 = !DILocation(line: 323, column: 2, scope: !1904)
!1950 = !DILocation(line: 324, column: 4, scope: !1904)
!1951 = !DILocation(line: 325, column: 10, scope: !1952)
!1952 = distinct !DILexicalBlock(scope: !1904, file: !3, line: 325, column: 5)
!1953 = !DILocation(line: 325, column: 5, scope: !1952)
!1954 = !DILocation(line: 325, column: 12, scope: !1952)
!1955 = !DILocation(line: 325, column: 5, scope: !1904)
!1956 = !DILocation(line: 325, column: 20, scope: !1957)
!1957 = distinct !DILexicalBlock(scope: !1952, file: !3, line: 325, column: 18)
!1958 = !DILocation(line: 325, column: 23, scope: !1957)
!1959 = !DILocation(line: 326, column: 7, scope: !1904)
!1960 = !DILocation(line: 326, column: 8, scope: !1904)
!1961 = !DILocation(line: 326, column: 2, scope: !1904)
!1962 = !DILocation(line: 326, column: 12, scope: !1904)
!1963 = !DILocation(line: 327, column: 2, scope: !1904)
!1964 = !DILocation(line: 328, column: 56, scope: !1904)
!1965 = !DILocation(line: 328, column: 2, scope: !1904)
!1966 = !DILocation(line: 330, column: 11, scope: !1904)
!1967 = !DILocation(line: 332, column: 5, scope: !1904)
!1968 = !DILocation(line: 334, column: 7, scope: !1969)
!1969 = distinct !DILexicalBlock(scope: !1904, file: !3, line: 334, column: 2)
!1970 = !DILocation(line: 334, column: 6, scope: !1969)
!1971 = !DILocation(line: 334, column: 11, scope: !1972)
!1972 = distinct !DILexicalBlock(scope: !1969, file: !3, line: 334, column: 2)
!1973 = !DILocation(line: 334, column: 12, scope: !1972)
!1974 = !DILocation(line: 334, column: 2, scope: !1969)
!1975 = !DILocation(line: 335, column: 15, scope: !1976)
!1976 = distinct !DILexicalBlock(scope: !1972, file: !3, line: 334, column: 23)
!1977 = !DILocation(line: 335, column: 3, scope: !1976)
!1978 = !DILocation(line: 336, column: 2, scope: !1976)
!1979 = !DILocation(line: 334, column: 20, scope: !1972)
!1980 = !DILocation(line: 334, column: 2, scope: !1972)
!1981 = distinct !{!1981, !1974, !1982}
!1982 = !DILocation(line: 336, column: 2, scope: !1969)
!1983 = !DILocation(line: 338, column: 7, scope: !1904)
!1984 = !DILocation(line: 338, column: 5, scope: !1904)
!1985 = !DILocation(line: 339, column: 5, scope: !1904)
!1986 = !DILocation(line: 340, column: 5, scope: !1904)
!1987 = !DILocation(line: 341, column: 5, scope: !1904)
!1988 = !DILocation(line: 343, column: 7, scope: !1989)
!1989 = distinct !DILexicalBlock(scope: !1904, file: !3, line: 343, column: 2)
!1990 = !DILocation(line: 343, column: 6, scope: !1989)
!1991 = !DILocation(line: 343, column: 11, scope: !1992)
!1992 = distinct !DILexicalBlock(scope: !1989, file: !3, line: 343, column: 2)
!1993 = !DILocation(line: 343, column: 12, scope: !1992)
!1994 = !DILocation(line: 343, column: 2, scope: !1989)
!1995 = !DILocation(line: 344, column: 3, scope: !1996)
!1996 = distinct !DILexicalBlock(scope: !1992, file: !3, line: 343, column: 21)
!1997 = !DILocation(line: 344, column: 5, scope: !1996)
!1998 = !DILocation(line: 344, column: 8, scope: !1996)
!1999 = !DILocation(line: 345, column: 2, scope: !1996)
!2000 = !DILocation(line: 343, column: 18, scope: !1992)
!2001 = !DILocation(line: 343, column: 2, scope: !1992)
!2002 = distinct !{!2002, !1994, !2003}
!2003 = !DILocation(line: 345, column: 2, scope: !1989)
!2004 = !DILocation(line: 347, column: 2, scope: !1904)
!2005 = !DILocation(line: 352, column: 15, scope: !1904)
!2006 = !DILocation(line: 353, column: 3, scope: !1904)
!2007 = !DILocation(line: 352, column: 12, scope: !1904)
!2008 = !DILocation(line: 352, column: 2, scope: !1904)
!2009 = !DILocation(line: 353, column: 24, scope: !1904)
!2010 = !DILocation(line: 354, column: 5, scope: !1904)
!2011 = !DILocation(line: 355, column: 5, scope: !1904)
!2012 = !DILocation(line: 356, column: 5, scope: !1904)
!2013 = !DILocation(line: 361, column: 13, scope: !1904)
!2014 = !DILocation(line: 361, column: 21, scope: !1904)
!2015 = !DILocation(line: 361, column: 31, scope: !1904)
!2016 = !DILocation(line: 361, column: 2, scope: !1904)
!2017 = !DILocation(line: 362, column: 13, scope: !1904)
!2018 = !DILocation(line: 362, column: 22, scope: !1904)
!2019 = !DILocation(line: 362, column: 33, scope: !1904)
!2020 = !DILocation(line: 362, column: 2, scope: !1904)
!2021 = !DILocation(line: 363, column: 13, scope: !1904)
!2022 = !DILocation(line: 363, column: 22, scope: !1904)
!2023 = !DILocation(line: 363, column: 33, scope: !1904)
!2024 = !DILocation(line: 363, column: 2, scope: !1904)
!2025 = !DILocation(line: 365, column: 11, scope: !2026)
!2026 = distinct !DILexicalBlock(scope: !1904, file: !3, line: 365, column: 2)
!2027 = !DILocation(line: 365, column: 6, scope: !2026)
!2028 = !DILocation(line: 365, column: 15, scope: !2029)
!2029 = distinct !DILexicalBlock(scope: !2026, file: !3, line: 365, column: 2)
!2030 = !DILocation(line: 365, column: 21, scope: !2029)
!2031 = !DILocation(line: 365, column: 20, scope: !2029)
!2032 = !DILocation(line: 365, column: 2, scope: !2026)
!2033 = !DILocation(line: 366, column: 8, scope: !2034)
!2034 = distinct !DILexicalBlock(scope: !2035, file: !3, line: 366, column: 3)
!2035 = distinct !DILexicalBlock(scope: !2029, file: !3, line: 365, column: 46)
!2036 = !DILocation(line: 366, column: 7, scope: !2034)
!2037 = !DILocation(line: 366, column: 12, scope: !2038)
!2038 = distinct !DILexicalBlock(scope: !2034, file: !3, line: 366, column: 3)
!2039 = !DILocation(line: 366, column: 13, scope: !2038)
!2040 = !DILocation(line: 366, column: 3, scope: !2034)
!2041 = !DILocation(line: 367, column: 10, scope: !2042)
!2042 = distinct !DILexicalBlock(scope: !2038, file: !3, line: 366, column: 22)
!2043 = !DILocation(line: 367, column: 17, scope: !2042)
!2044 = !DILocation(line: 367, column: 22, scope: !2042)
!2045 = !DILocation(line: 367, column: 26, scope: !2042)
!2046 = !DILocation(line: 367, column: 25, scope: !2042)
!2047 = !DILocation(line: 367, column: 4, scope: !2042)
!2048 = !DILocation(line: 367, column: 6, scope: !2042)
!2049 = !DILocation(line: 367, column: 8, scope: !2042)
!2050 = !DILocation(line: 368, column: 3, scope: !2042)
!2051 = !DILocation(line: 366, column: 19, scope: !2038)
!2052 = !DILocation(line: 366, column: 3, scope: !2038)
!2053 = distinct !{!2053, !2040, !2054}
!2054 = !DILocation(line: 368, column: 3, scope: !2034)
!2055 = !DILocation(line: 369, column: 7, scope: !2035)
!2056 = !DILocation(line: 369, column: 15, scope: !2035)
!2057 = !DILocation(line: 369, column: 5, scope: !2035)
!2058 = !DILocation(line: 370, column: 7, scope: !2035)
!2059 = !DILocation(line: 370, column: 15, scope: !2035)
!2060 = !DILocation(line: 370, column: 5, scope: !2035)
!2061 = !DILocation(line: 371, column: 2, scope: !2035)
!2062 = !DILocation(line: 365, column: 43, scope: !2029)
!2063 = !DILocation(line: 365, column: 2, scope: !2029)
!2064 = distinct !{!2064, !2032, !2065}
!2065 = !DILocation(line: 371, column: 2, scope: !2026)
!2066 = !DILocation(line: 372, column: 7, scope: !2067)
!2067 = distinct !DILexicalBlock(scope: !1904, file: !3, line: 372, column: 2)
!2068 = !DILocation(line: 372, column: 6, scope: !2067)
!2069 = !DILocation(line: 372, column: 11, scope: !2070)
!2070 = distinct !DILexicalBlock(scope: !2067, file: !3, line: 372, column: 2)
!2071 = !DILocation(line: 372, column: 12, scope: !2070)
!2072 = !DILocation(line: 372, column: 2, scope: !2067)
!2073 = !DILocation(line: 373, column: 7, scope: !2074)
!2074 = distinct !DILexicalBlock(scope: !2070, file: !3, line: 372, column: 21)
!2075 = !DILocation(line: 373, column: 9, scope: !2074)
!2076 = !DILocation(line: 373, column: 5, scope: !2074)
!2077 = !DILocation(line: 374, column: 2, scope: !2074)
!2078 = !DILocation(line: 372, column: 18, scope: !2070)
!2079 = !DILocation(line: 372, column: 2, scope: !2070)
!2080 = distinct !{!2080, !2072, !2081}
!2081 = !DILocation(line: 374, column: 2, scope: !2067)
!2082 = !DILocation(line: 376, column: 6, scope: !1904)
!2083 = !DILocation(line: 377, column: 11, scope: !1904)
!2084 = !DILocation(line: 385, column: 19, scope: !2085)
!2085 = distinct !DILexicalBlock(scope: !2086, file: !3, line: 384, column: 19)
!2086 = distinct !DILexicalBlock(scope: !2087, file: !3, line: 384, column: 11)
!2087 = distinct !DILexicalBlock(scope: !2088, file: !3, line: 381, column: 11)
!2088 = distinct !DILexicalBlock(scope: !1904, file: !3, line: 378, column: 5)
!2089 = !DILocation(line: 386, column: 19, scope: !2085)
!2090 = !DILocation(line: 402, column: 5, scope: !2091)
!2091 = distinct !DILexicalBlock(scope: !1904, file: !3, line: 402, column: 5)
!2092 = !DILocation(line: 402, column: 5, scope: !1904)
!2093 = !DILocation(line: 403, column: 18, scope: !2094)
!2094 = distinct !DILexicalBlock(scope: !2091, file: !3, line: 402, column: 14)
!2095 = !DILocation(line: 403, column: 23, scope: !2094)
!2096 = !DILocation(line: 403, column: 21, scope: !2094)
!2097 = !DILocation(line: 403, column: 42, scope: !2094)
!2098 = !DILocation(line: 403, column: 40, scope: !2094)
!2099 = !DILocation(line: 403, column: 12, scope: !2094)
!2100 = !DILocation(line: 403, column: 10, scope: !2094)
!2101 = !DILocation(line: 404, column: 18, scope: !2094)
!2102 = !DILocation(line: 404, column: 23, scope: !2094)
!2103 = !DILocation(line: 404, column: 21, scope: !2094)
!2104 = !DILocation(line: 404, column: 42, scope: !2094)
!2105 = !DILocation(line: 404, column: 40, scope: !2094)
!2106 = !DILocation(line: 404, column: 12, scope: !2094)
!2107 = !DILocation(line: 404, column: 10, scope: !2094)
!2108 = !DILocation(line: 405, column: 16, scope: !2094)
!2109 = !DILocation(line: 405, column: 23, scope: !2094)
!2110 = !DILocation(line: 405, column: 35, scope: !2094)
!2111 = !DILocation(line: 405, column: 39, scope: !2094)
!2112 = !DILocation(line: 405, column: 46, scope: !2094)
!2113 = !DILocation(line: 0, scope: !2094)
!2114 = !DILocation(line: 405, column: 14, scope: !2094)
!2115 = !DILocation(line: 405, column: 12, scope: !2094)
!2116 = !DILocation(line: 406, column: 2, scope: !2094)
!2117 = !DILocation(line: 407, column: 9, scope: !1904)
!2118 = !DILocation(line: 407, column: 23, scope: !1904)
!2119 = !DILocation(line: 407, column: 22, scope: !1904)
!2120 = !DILocation(line: 407, column: 25, scope: !1904)
!2121 = !DILocation(line: 407, column: 7, scope: !1904)
!2122 = !DILocation(line: 409, column: 2, scope: !1904)
!2123 = !DILocation(line: 410, column: 32, scope: !1904)
!2124 = !DILocation(line: 410, column: 2, scope: !1904)
!2125 = !DILocation(line: 411, column: 2, scope: !1904)
!2126 = !DILocation(line: 412, column: 43, scope: !1904)
!2127 = !DILocation(line: 412, column: 2, scope: !1904)
!2128 = !DILocation(line: 413, column: 38, scope: !1904)
!2129 = !DILocation(line: 413, column: 42, scope: !1904)
!2130 = !DILocation(line: 413, column: 2, scope: !1904)
!2131 = !DILocation(line: 414, column: 2, scope: !1904)
!2132 = !DILocation(line: 415, column: 7, scope: !2133)
!2133 = distinct !DILexicalBlock(scope: !1904, file: !3, line: 415, column: 2)
!2134 = !DILocation(line: 415, column: 6, scope: !2133)
!2135 = !DILocation(line: 415, column: 11, scope: !2136)
!2136 = distinct !DILexicalBlock(scope: !2133, file: !3, line: 415, column: 2)
!2137 = !DILocation(line: 415, column: 12, scope: !2136)
!2138 = !DILocation(line: 415, column: 2, scope: !2133)
!2139 = !DILocation(line: 416, column: 25, scope: !2140)
!2140 = distinct !DILexicalBlock(scope: !2136, file: !3, line: 415, column: 21)
!2141 = !DILocation(line: 416, column: 28, scope: !2140)
!2142 = !DILocation(line: 416, column: 30, scope: !2140)
!2143 = !DILocation(line: 416, column: 3, scope: !2140)
!2144 = !DILocation(line: 417, column: 2, scope: !2140)
!2145 = !DILocation(line: 415, column: 18, scope: !2136)
!2146 = !DILocation(line: 415, column: 2, scope: !2136)
!2147 = distinct !{!2147, !2138, !2148}
!2148 = !DILocation(line: 417, column: 2, scope: !2133)
!2149 = !DILocalVariable(name: "gpu_config", scope: !1904, file: !3, line: 419, type: !139)
!2150 = !DILocation(line: 419, column: 7, scope: !1904)
!2151 = !DILocalVariable(name: "gpu_config_string", scope: !1904, file: !3, line: 420, type: !2152)
!2152 = !DICompositeType(tag: DW_TAG_array_type, baseType: !101, size: 16384, elements: !2153)
!2153 = !{!2154}
!2154 = !DISubrange(count: 2048)
!2155 = !DILocation(line: 420, column: 7, scope: !1904)
!2156 = !DILocation(line: 427, column: 10, scope: !1904)
!2157 = !DILocation(line: 427, column: 2, scope: !1904)
!2158 = !DILocation(line: 428, column: 9, scope: !1904)
!2159 = !DILocation(line: 428, column: 28, scope: !1904)
!2160 = !DILocation(line: 428, column: 2, scope: !1904)
!2161 = !DILocation(line: 429, column: 10, scope: !1904)
!2162 = !DILocation(line: 429, column: 45, scope: !1904)
!2163 = !DILocation(line: 429, column: 2, scope: !1904)
!2164 = !DILocation(line: 430, column: 9, scope: !1904)
!2165 = !DILocation(line: 430, column: 28, scope: !1904)
!2166 = !DILocation(line: 430, column: 2, scope: !1904)
!2167 = !DILocation(line: 438, column: 4, scope: !1904)
!2168 = !DILocation(line: 439, column: 4, scope: !1904)
!2169 = !DILocation(line: 440, column: 4, scope: !1904)
!2170 = !DILocation(line: 442, column: 4, scope: !1904)
!2171 = !DILocation(line: 449, column: 4, scope: !1904)
!2172 = !DILocation(line: 433, column: 2, scope: !1904)
!2173 = !DILocation(line: 458, column: 2, scope: !1904)
!2174 = !DILocation(line: 460, column: 2, scope: !1904)
!2175 = distinct !DISubprogram(name: "setup_gpu", linkageName: "_ZL9setup_gpuv", scope: !3, file: !3, line: 576, type: !472, scopeLine: 576, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !962)
!2176 = !DILocation(line: 625, column: 49, scope: !2177)
!2177 = distinct !DILexicalBlock(scope: !2175, file: !3, line: 624, column: 5)
!2178 = !DILocation(line: 625, column: 25, scope: !2177)
!2179 = !DILocation(line: 624, column: 5, scope: !2175)
!2180 = !DILocation(line: 626, column: 21, scope: !2181)
!2181 = distinct !DILexicalBlock(scope: !2177, file: !3, line: 625, column: 69)
!2182 = !DILocation(line: 627, column: 2, scope: !2181)
!2183 = !DILocation(line: 628, column: 45, scope: !2184)
!2184 = distinct !DILexicalBlock(scope: !2177, file: !3, line: 627, column: 7)
!2185 = !DILocation(line: 628, column: 21, scope: !2184)
!2186 = !DILocation(line: 631, column: 45, scope: !2175)
!2187 = !DILocation(line: 631, column: 36, scope: !2175)
!2188 = !DILocation(line: 631, column: 21, scope: !2175)
!2189 = !DILocation(line: 631, column: 20, scope: !2175)
!2190 = !DILocation(line: 631, column: 18, scope: !2175)
!2191 = !DILocation(line: 633, column: 11, scope: !2175)
!2192 = !DILocation(line: 633, column: 27, scope: !2175)
!2193 = !DILocation(line: 633, column: 32, scope: !2175)
!2194 = !DILocation(line: 633, column: 9, scope: !2175)
!2195 = !DILocation(line: 634, column: 12, scope: !2175)
!2196 = !DILocation(line: 634, column: 28, scope: !2175)
!2197 = !DILocation(line: 634, column: 10, scope: !2175)
!2198 = !DILocation(line: 635, column: 12, scope: !2175)
!2199 = !DILocation(line: 635, column: 28, scope: !2175)
!2200 = !DILocation(line: 635, column: 10, scope: !2175)
!2201 = !DILocation(line: 637, column: 25, scope: !2175)
!2202 = !DILocation(line: 637, column: 18, scope: !2175)
!2203 = !DILocation(line: 637, column: 9, scope: !2175)
!2204 = !DILocation(line: 637, column: 8, scope: !2175)
!2205 = !DILocation(line: 638, column: 26, scope: !2175)
!2206 = !DILocation(line: 638, column: 19, scope: !2175)
!2207 = !DILocation(line: 638, column: 10, scope: !2175)
!2208 = !DILocation(line: 638, column: 9, scope: !2175)
!2209 = !DILocation(line: 639, column: 26, scope: !2175)
!2210 = !DILocation(line: 639, column: 19, scope: !2175)
!2211 = !DILocation(line: 639, column: 10, scope: !2175)
!2212 = !DILocation(line: 639, column: 9, scope: !2175)
!2213 = !DILocation(line: 641, column: 24, scope: !2175)
!2214 = !DILocation(line: 641, column: 2, scope: !2175)
!2215 = !DILocation(line: 642, column: 25, scope: !2175)
!2216 = !DILocation(line: 642, column: 2, scope: !2175)
!2217 = !DILocation(line: 643, column: 25, scope: !2175)
!2218 = !DILocation(line: 643, column: 2, scope: !2175)
!2219 = !DILocation(line: 644, column: 1, scope: !2175)
!2220 = distinct !DISubprogram(name: "dim3", linkageName: "_ZN4dim3C2Ejjj", scope: !1192, file: !1158, line: 421, type: !1198, scopeLine: 421, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, declaration: !1197, retainedNodes: !962)
!2221 = !DILocalVariable(name: "this", arg: 1, scope: !2220, type: !2222, flags: DIFlagArtificial | DIFlagObjectPointer)
!2222 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !1192, size: 64)
!2223 = !DILocation(line: 0, scope: !2220)
!2224 = !DILocalVariable(name: "vx", arg: 2, scope: !2220, file: !1158, line: 421, type: !7)
!2225 = !DILocation(line: 421, column: 43, scope: !2220)
!2226 = !DILocalVariable(name: "vy", arg: 3, scope: !2220, file: !1158, line: 421, type: !7)
!2227 = !DILocation(line: 421, column: 64, scope: !2220)
!2228 = !DILocalVariable(name: "vz", arg: 4, scope: !2220, file: !1158, line: 421, type: !7)
!2229 = !DILocation(line: 421, column: 85, scope: !2220)
!2230 = !DILocation(line: 421, column: 95, scope: !2220)
!2231 = !DILocation(line: 421, column: 97, scope: !2220)
!2232 = !DILocation(line: 421, column: 102, scope: !2220)
!2233 = !DILocation(line: 421, column: 104, scope: !2220)
!2234 = !DILocation(line: 421, column: 109, scope: !2220)
!2235 = !DILocation(line: 421, column: 111, scope: !2220)
!2236 = !DILocation(line: 421, column: 116, scope: !2220)
!2237 = distinct !DISubprogram(name: "gpu_kernel", linkageName: "_Z10gpu_kernelPdS_S_d", scope: !3, file: !3, line: 463, type: !1047, scopeLine: 466, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !962)
!2238 = !DILocalVariable(name: "q_global", arg: 1, scope: !2237, file: !3, line: 463, type: !97)
!2239 = !DILocation(line: 463, column: 36, scope: !2237)
!2240 = !DILocalVariable(name: "sx_global", arg: 2, scope: !2237, file: !3, line: 464, type: !97)
!2241 = !DILocation(line: 464, column: 11, scope: !2237)
!2242 = !DILocalVariable(name: "sy_global", arg: 3, scope: !2237, file: !3, line: 465, type: !97)
!2243 = !DILocation(line: 465, column: 11, scope: !2237)
!2244 = !DILocalVariable(name: "an", arg: 4, scope: !2237, file: !3, line: 466, type: !98)
!2245 = !DILocation(line: 466, column: 10, scope: !2237)
!2246 = !DILocation(line: 466, column: 13, scope: !2237)
!2247 = !DILocation(line: 550, column: 1, scope: !2237)
!2248 = distinct !DISubprogram(name: "release_gpu", linkageName: "_ZL11release_gpuv", scope: !3, file: !3, line: 570, type: !472, scopeLine: 570, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !962)
!2249 = !DILocation(line: 571, column: 11, scope: !2248)
!2250 = !DILocation(line: 571, column: 2, scope: !2248)
!2251 = !DILocation(line: 572, column: 11, scope: !2248)
!2252 = !DILocation(line: 572, column: 2, scope: !2248)
!2253 = !DILocation(line: 573, column: 11, scope: !2248)
!2254 = !DILocation(line: 573, column: 2, scope: !2248)
!2255 = !DILocation(line: 574, column: 1, scope: !2248)
!2256 = distinct !DISubprogram(name: "cudaMalloc<double>", linkageName: "_ZL10cudaMallocIdE9cudaErrorPPT_m", scope: !2257, file: !2257, line: 490, type: !2258, scopeLine: 494, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, templateParams: !2262, retainedNodes: !962)
!2257 = !DIFile(filename: "/usr/local/cuda/include/cuda_runtime.h", directory: "")
!2258 = !DISubroutineType(types: !2259)
!2259 = !{!2260, !2261, !123}
!2260 = !DIDerivedType(tag: DW_TAG_typedef, name: "cudaError_t", file: !6, line: 1419, baseType: !14)
!2261 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !97, size: 64)
!2262 = !{!2263}
!2263 = !DITemplateTypeParameter(name: "T", type: !98)
!2264 = !DILocalVariable(name: "devPtr", arg: 1, scope: !2256, file: !2257, line: 491, type: !2261)
!2265 = !DILocation(line: 491, column: 12, scope: !2256)
!2266 = !DILocalVariable(name: "size", arg: 2, scope: !2256, file: !2257, line: 492, type: !123)
!2267 = !DILocation(line: 492, column: 12, scope: !2256)
!2268 = !DILocation(line: 495, column: 38, scope: !2256)
!2269 = !DILocation(line: 495, column: 23, scope: !2256)
!2270 = !DILocation(line: 495, column: 46, scope: !2256)
!2271 = !DILocation(line: 495, column: 10, scope: !2256)
!2272 = !DILocation(line: 495, column: 3, scope: !2256)
