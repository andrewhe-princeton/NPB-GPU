; ModuleID = 'ep_linked.bc'
source_filename = "llvm-link-cudafe"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.cudaDeviceProp = type { [256 x i8], i64, i64, i32, i32, i64, i32, [3 x i32], [3 x i32], i32, i64, i32, i32, i64, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, [2 x i32], [2 x i32], [3 x i32], [2 x i32], [3 x i32], [3 x i32], i32, [2 x i32], [3 x i32], [2 x i32], i32, [2 x i32], [3 x i32], [2 x i32], [3 x i32], i32, [2 x i32], i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i64, i32, i32, i32, i32 }
%struct.dim3 = type { i32, i32, i32 }
%struct.CUstream_st = type opaque

$_ZN4dim3C2Ejjj = comdat any

@q_host = dso_local global double* null, align 8, !dbg !0
@q_device = dso_local global double* null, align 8, !dbg !105
@sx_host = dso_local global double* null, align 8, !dbg !107
@sx_device = dso_local global double* null, align 8, !dbg !109
@sy_host = dso_local global double* null, align 8, !dbg !111
@sy_device = dso_local global double* null, align 8, !dbg !113
@threads_per_block = dso_local global i32 0, align 4, !dbg !115
@blocks_per_grid = dso_local global i32 0, align 4, !dbg !117
@size_q = dso_local global i64 0, align 8, !dbg !119
@size_sx = dso_local global i64 0, align 8, !dbg !124
@size_sy = dso_local global i64 0, align 8, !dbg !126
@gpu_device_id = dso_local global i32 0, align 4, !dbg !128
@total_devices = dso_local global i32 0, align 4, !dbg !130
@gpu_device_properties = dso_local global %struct.cudaDeviceProp zeroinitializer, align 8, !dbg !132
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
@_ZL1q = internal global double* null, align 8, !dbg !207
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
define dso_local void @_Z10gpu_kernelPdS_S_d(double* %q_global, double* %sx_global, double* %sy_global, double %an) #0 !dbg !1050 {
entry:
  %f.addr.i143 = alloca double, align 8
  call void @llvm.dbg.declare(metadata double* %f.addr.i143, metadata !1053, metadata !DIExpression()), !dbg !1055
  %f.addr.i142 = alloca double, align 8
  call void @llvm.dbg.declare(metadata double* %f.addr.i142, metadata !1053, metadata !DIExpression()), !dbg !1065
  %f.addr.i141 = alloca double, align 8
  call void @llvm.dbg.declare(metadata double* %f.addr.i141, metadata !1053, metadata !DIExpression()), !dbg !1067
  %f.addr.i = alloca double, align 8
  call void @llvm.dbg.declare(metadata double* %f.addr.i, metadata !1053, metadata !DIExpression()), !dbg !1069
  %x.addr.i = alloca double, align 8
  call void @llvm.dbg.declare(metadata double* %x.addr.i, metadata !1071, metadata !DIExpression()), !dbg !1073
  %a.addr.i = alloca double, align 8
  call void @llvm.dbg.declare(metadata double* %a.addr.i, metadata !1075, metadata !DIExpression()), !dbg !1078
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
  call void @llvm.dbg.declare(metadata double** %q_global.addr, metadata !1080, metadata !DIExpression()), !dbg !1081
  store double* %sx_global, double** %sx_global.addr, align 8
  call void @llvm.dbg.declare(metadata double** %sx_global.addr, metadata !1082, metadata !DIExpression()), !dbg !1083
  store double* %sy_global, double** %sy_global.addr, align 8
  call void @llvm.dbg.declare(metadata double** %sy_global.addr, metadata !1084, metadata !DIExpression()), !dbg !1085
  store double %an, double* %an.addr, align 8
  call void @llvm.dbg.declare(metadata double* %an.addr, metadata !1086, metadata !DIExpression()), !dbg !1087
  call void @llvm.dbg.declare(metadata [256 x double]* %x_local, metadata !1088, metadata !DIExpression()), !dbg !1090
  call void @llvm.dbg.declare(metadata [10 x double]* %q_local, metadata !1091, metadata !DIExpression()), !dbg !1095
  call void @llvm.dbg.declare(metadata double* %sx_local, metadata !1096, metadata !DIExpression()), !dbg !1097
  call void @llvm.dbg.declare(metadata double* %sy_local, metadata !1098, metadata !DIExpression()), !dbg !1099
  call void @llvm.dbg.declare(metadata double* %t1, metadata !1100, metadata !DIExpression()), !dbg !1101
  call void @llvm.dbg.declare(metadata double* %t2, metadata !1102, metadata !DIExpression()), !dbg !1103
  call void @llvm.dbg.declare(metadata double* %t3, metadata !1104, metadata !DIExpression()), !dbg !1105
  call void @llvm.dbg.declare(metadata double* %t4, metadata !1106, metadata !DIExpression()), !dbg !1107
  call void @llvm.dbg.declare(metadata double* %x1, metadata !1108, metadata !DIExpression()), !dbg !1109
  call void @llvm.dbg.declare(metadata double* %x2, metadata !1110, metadata !DIExpression()), !dbg !1111
  call void @llvm.dbg.declare(metadata double* %seed, metadata !1112, metadata !DIExpression()), !dbg !1113
  call void @llvm.dbg.declare(metadata i32* %i, metadata !1114, metadata !DIExpression()), !dbg !1115
  call void @llvm.dbg.declare(metadata i32* %ii, metadata !1116, metadata !DIExpression()), !dbg !1117
  call void @llvm.dbg.declare(metadata i32* %ik, metadata !1118, metadata !DIExpression()), !dbg !1119
  call void @llvm.dbg.declare(metadata i32* %kk, metadata !1120, metadata !DIExpression()), !dbg !1121
  call void @llvm.dbg.declare(metadata i32* %l, metadata !1122, metadata !DIExpression()), !dbg !1123
  %arrayidx = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 0, !dbg !1124
  store double 0.000000e+00, double* %arrayidx, align 8, !dbg !1125
  %arrayidx1 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 1, !dbg !1126
  store double 0.000000e+00, double* %arrayidx1, align 8, !dbg !1127
  %arrayidx2 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 2, !dbg !1128
  store double 0.000000e+00, double* %arrayidx2, align 8, !dbg !1129
  %arrayidx3 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 3, !dbg !1130
  store double 0.000000e+00, double* %arrayidx3, align 8, !dbg !1131
  %arrayidx4 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 4, !dbg !1132
  store double 0.000000e+00, double* %arrayidx4, align 8, !dbg !1133
  %arrayidx5 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 5, !dbg !1134
  store double 0.000000e+00, double* %arrayidx5, align 8, !dbg !1135
  %arrayidx6 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 6, !dbg !1136
  store double 0.000000e+00, double* %arrayidx6, align 8, !dbg !1137
  %arrayidx7 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 7, !dbg !1138
  store double 0.000000e+00, double* %arrayidx7, align 8, !dbg !1139
  %arrayidx8 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 8, !dbg !1140
  store double 0.000000e+00, double* %arrayidx8, align 8, !dbg !1141
  %arrayidx9 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 9, !dbg !1142
  store double 0.000000e+00, double* %arrayidx9, align 8, !dbg !1143
  store double 0.000000e+00, double* %sx_local, align 8, !dbg !1144
  store double 0.000000e+00, double* %sy_local, align 8, !dbg !1145
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #10, !dbg !1146, !range !1183
  %1 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #10, !dbg !1184, !range !1228
  %mul = mul i32 %0, %1, !dbg !1229
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #10, !dbg !1230, !range !1258
  %add = add i32 %mul, %2, !dbg !1259
  store i32 %add, i32* %kk, align 4, !dbg !1260
  %3 = load i32, i32* %kk, align 4, !dbg !1261
  %cmp = icmp sge i32 %3, 4096, !dbg !1263
  br i1 %cmp, label %if.then, label %if.end, !dbg !1264

if.then:                                          ; preds = %entry
  br label %return, !dbg !1265

if.end:                                           ; preds = %entry
  store double 0x41B033C4D7000000, double* %t1, align 8, !dbg !1267
  %4 = load double, double* %an.addr, align 8, !dbg !1268
  store double %4, double* %t2, align 8, !dbg !1269
  store i32 1, i32* %i, align 4, !dbg !1270
  br label %for.cond, !dbg !1272

for.cond:                                         ; preds = %for.inc, %if.end
  %5 = load i32, i32* %i, align 4, !dbg !1273
  %cmp12 = icmp sle i32 %5, 100, !dbg !1275
  br i1 %cmp12, label %for.body, label %for.end, !dbg !1276

for.body:                                         ; preds = %for.cond
  %6 = load i32, i32* %kk, align 4, !dbg !1277
  %div = sdiv i32 %6, 2, !dbg !1279
  store i32 %div, i32* %ik, align 4, !dbg !1280
  %7 = load i32, i32* %ik, align 4, !dbg !1281
  %mul13 = mul nsw i32 2, %7, !dbg !1283
  %8 = load i32, i32* %kk, align 4, !dbg !1284
  %cmp14 = icmp ne i32 %mul13, %8, !dbg !1285
  br i1 %cmp14, label %if.then15, label %if.end17, !dbg !1286

if.then15:                                        ; preds = %for.body
  %9 = load double, double* %t2, align 8, !dbg !1287
  %call16 = call double @_Z13randlc_devicePdd(double* %t1, double %9) #11, !dbg !1289
  store double %call16, double* %t3, align 8, !dbg !1290
  br label %if.end17, !dbg !1291

if.end17:                                         ; preds = %if.then15, %for.body
  %10 = load i32, i32* %ik, align 4, !dbg !1292
  %cmp18 = icmp eq i32 %10, 0, !dbg !1294
  br i1 %cmp18, label %if.then19, label %if.end20, !dbg !1295

if.then19:                                        ; preds = %if.end17
  br label %for.end, !dbg !1296

if.end20:                                         ; preds = %if.end17
  %11 = load double, double* %t2, align 8, !dbg !1298
  %call21 = call double @_Z13randlc_devicePdd(double* %t2, double %11) #11, !dbg !1299
  store double %call21, double* %t3, align 8, !dbg !1300
  %12 = load i32, i32* %ik, align 4, !dbg !1301
  store i32 %12, i32* %kk, align 4, !dbg !1302
  br label %for.inc, !dbg !1303

for.inc:                                          ; preds = %if.end20
  %13 = load i32, i32* %i, align 4, !dbg !1304
  %inc = add nsw i32 %13, 1, !dbg !1304
  store i32 %inc, i32* %i, align 4, !dbg !1304
  br label %for.cond, !dbg !1305, !llvm.loop !1306

for.end:                                          ; preds = %if.then19, %for.cond
  %14 = load double, double* %t1, align 8, !dbg !1308
  store double %14, double* %seed, align 8, !dbg !1309
  store i32 0, i32* %ii, align 4, !dbg !1310
  br label %for.cond22, !dbg !1311

for.cond22:                                       ; preds = %for.inc62, %for.end
  %15 = load i32, i32* %ii, align 4, !dbg !1312
  %cmp23 = icmp slt i32 %15, 65536, !dbg !1313
  br i1 %cmp23, label %for.body24, label %for.end64, !dbg !1314

for.body24:                                       ; preds = %for.cond22
  %arraydecay = getelementptr inbounds [256 x double], [256 x double]* %x_local, i64 0, i64 0, !dbg !1315
  call void @_Z13vranlc_deviceiPddS_(i32 256, double* %seed, double 0x41D2309CE5400000, double* %arraydecay) #11, !dbg !1316
  store i32 0, i32* %i, align 4, !dbg !1317
  br label %for.cond25, !dbg !1318

for.cond25:                                       ; preds = %for.inc59, %for.body24
  %16 = load i32, i32* %i, align 4, !dbg !1319
  %cmp26 = icmp slt i32 %16, 128, !dbg !1320
  br i1 %cmp26, label %for.body27, label %for.end61, !dbg !1321

for.body27:                                       ; preds = %for.cond25
  %17 = load i32, i32* %i, align 4, !dbg !1322
  %mul28 = mul nsw i32 2, %17, !dbg !1323
  %idxprom = sext i32 %mul28 to i64, !dbg !1324
  %arrayidx29 = getelementptr inbounds [256 x double], [256 x double]* %x_local, i64 0, i64 %idxprom, !dbg !1324
  %18 = load double, double* %arrayidx29, align 8, !dbg !1324
  %mul30 = fmul contract double 2.000000e+00, %18, !dbg !1325
  %sub = fsub contract double %mul30, 1.000000e+00, !dbg !1326
  store double %sub, double* %x1, align 8, !dbg !1327
  %19 = load i32, i32* %i, align 4, !dbg !1328
  %mul31 = mul nsw i32 2, %19, !dbg !1329
  %add32 = add nsw i32 %mul31, 1, !dbg !1330
  %idxprom33 = sext i32 %add32 to i64, !dbg !1331
  %arrayidx34 = getelementptr inbounds [256 x double], [256 x double]* %x_local, i64 0, i64 %idxprom33, !dbg !1331
  %20 = load double, double* %arrayidx34, align 8, !dbg !1331
  %mul35 = fmul contract double 2.000000e+00, %20, !dbg !1332
  %sub36 = fsub contract double %mul35, 1.000000e+00, !dbg !1333
  store double %sub36, double* %x2, align 8, !dbg !1334
  %21 = load double, double* %x1, align 8, !dbg !1335
  %22 = load double, double* %x1, align 8, !dbg !1336
  %mul37 = fmul contract double %21, %22, !dbg !1337
  %23 = load double, double* %x2, align 8, !dbg !1338
  %24 = load double, double* %x2, align 8, !dbg !1339
  %mul38 = fmul contract double %23, %24, !dbg !1340
  %add39 = fadd contract double %mul37, %mul38, !dbg !1341
  store double %add39, double* %t1, align 8, !dbg !1342
  %25 = load double, double* %t1, align 8, !dbg !1343
  %cmp40 = fcmp ole double %25, 1.000000e+00, !dbg !1344
  br i1 %cmp40, label %if.then41, label %if.end58, !dbg !1345

if.then41:                                        ; preds = %for.body27
  %26 = load double, double* %t1, align 8, !dbg !1346
  store double %26, double* %a.addr.i, align 8
  %27 = load double, double* %a.addr.i, align 8, !dbg !1347
  %28 = call i32 @llvm.nvvm.d2i.hi(double %27) #10, !dbg !1348
  %29 = call i32 @llvm.nvvm.d2i.lo(double %27) #10, !dbg !1348
  %30 = fcmp ogt double %27, 0.000000e+00, !dbg !1348
  br i1 %30, label %31, label %33, !dbg !1348

31:                                               ; preds = %if.then41
  %32 = icmp slt i32 %28, 2146435072, !dbg !1348
  br label %33, !dbg !1348

33:                                               ; preds = %31, %if.then41
  %34 = phi i1 [ false, %if.then41 ], [ %32, %31 ], !dbg !1348
  br i1 %34, label %35, label %90, !dbg !1348

35:                                               ; preds = %33
  %36 = icmp slt i32 %28, 1048576, !dbg !1348
  br i1 %36, label %37, label %41, !dbg !1348

37:                                               ; preds = %35
  %38 = fmul double %27, 0x4350000000000000, !dbg !1348
  %39 = call i32 @llvm.nvvm.d2i.hi(double %38) #10, !dbg !1348
  %40 = call i32 @llvm.nvvm.d2i.lo(double %38) #10, !dbg !1348
  br label %41, !dbg !1348

41:                                               ; preds = %37, %35
  %ihi.0.i.i = phi i32 [ %39, %37 ], [ %28, %35 ], !dbg !1348
  %ilo.0.i.i = phi i32 [ %40, %37 ], [ %29, %35 ], !dbg !1348
  %e.0.i.i = phi i32 [ -1077, %37 ], [ -1023, %35 ], !dbg !1348
  %42 = lshr i32 %ihi.0.i.i, 20, !dbg !1348
  %43 = add i32 %e.0.i.i, %42, !dbg !1348
  %44 = and i32 %ihi.0.i.i, -2146435073, !dbg !1348
  %45 = or i32 %44, 1072693248, !dbg !1348
  %46 = call double @llvm.nvvm.lohi.i2d(i32 %ilo.0.i.i, i32 %45) #10, !dbg !1348
  %47 = icmp sgt i32 %45, 1073127582, !dbg !1348
  br i1 %47, label %48, label %54, !dbg !1348

48:                                               ; preds = %41
  %49 = call i32 @llvm.nvvm.d2i.lo(double %46) #10, !dbg !1348
  %50 = call i32 @llvm.nvvm.d2i.hi(double %46) #10, !dbg !1348
  %51 = add i32 -1048576, %50, !dbg !1348
  %52 = call double @llvm.nvvm.lohi.i2d(i32 %49, i32 %51) #10, !dbg !1348
  %53 = add nsw i32 %43, 1, !dbg !1348
  br label %54, !dbg !1348

54:                                               ; preds = %48, %41
  %m.0.i.i = phi double [ %52, %48 ], [ %46, %41 ], !dbg !1348
  %e.1.i.i = phi i32 [ %53, %48 ], [ %43, %41 ], !dbg !1348
  %55 = fsub double %m.0.i.i, 1.000000e+00, !dbg !1348
  %56 = fadd double %m.0.i.i, 1.000000e+00, !dbg !1348
  %57 = call double asm "rcp.approx.ftz.f64 $0,$1;", "=d,d"(double %56) #10, !dbg !1348
  %58 = fsub double -0.000000e+00, %56, !dbg !1348
  %59 = call double @llvm.nvvm.fma.rn.d(double %58, double %57, double 1.000000e+00) #10, !dbg !1348
  %60 = call double @llvm.nvvm.fma.rn.d(double %59, double %59, double %59) #10, !dbg !1348
  %61 = call double @llvm.nvvm.fma.rn.d(double %60, double %57, double %57) #10, !dbg !1348
  %62 = fmul double %55, %61, !dbg !1348
  %63 = fadd double %62, %62, !dbg !1348
  %64 = fmul double %63, %63, !dbg !1348
  %65 = call double @llvm.nvvm.fma.rn.d(double 0x3EB1380B3AE80F1E, double %64, double 0x3ED0EE258B7A8B04) #10, !dbg !1348
  %66 = call double @llvm.nvvm.fma.rn.d(double %65, double %64, double 0x3EF3B2669F02676F) #10, !dbg !1348
  %67 = call double @llvm.nvvm.fma.rn.d(double %66, double %64, double 0x3F1745CBA9AB0956) #10, !dbg !1348
  %68 = call double @llvm.nvvm.fma.rn.d(double %67, double %64, double 0x3F3C71C72D1B5154) #10, !dbg !1348
  %69 = call double @llvm.nvvm.fma.rn.d(double %68, double %64, double 0x3F624924923BE72D) #10, !dbg !1348
  %70 = call double @llvm.nvvm.fma.rn.d(double %69, double %64, double 0x3F8999999999A3C4) #10, !dbg !1348
  %71 = call double @llvm.nvvm.fma.rn.d(double %70, double %64, double 0x3FB5555555555554) #10, !dbg !1348
  %72 = fsub double %55, %63, !dbg !1348
  %73 = fmul double 2.000000e+00, %72, !dbg !1348
  %74 = fsub double -0.000000e+00, %63, !dbg !1348
  %75 = call double @llvm.nvvm.fma.rn.d(double %74, double %55, double %73) #10, !dbg !1348
  %76 = fmul double %61, %75, !dbg !1348
  %77 = fmul double %71, %64, !dbg !1348
  %78 = call double @llvm.nvvm.fma.rn.d(double %77, double %63, double %76) #10, !dbg !1348
  %79 = xor i32 -2147483648, %e.1.i.i, !dbg !1348
  %80 = call double @llvm.nvvm.lohi.i2d(i32 %79, i32 1127219200) #10, !dbg !1348
  %81 = call double @llvm.nvvm.lohi.i2d(i32 -2147483648, i32 1127219200) #10, !dbg !1348
  %82 = fsub double %80, %81, !dbg !1348
  %83 = call double @llvm.nvvm.fma.rn.d(double %82, double 0x3FE62E42FEFA39EF, double %63) #10, !dbg !1348
  %84 = fsub double -0.000000e+00, %82, !dbg !1348
  %85 = call double @llvm.nvvm.fma.rn.d(double %84, double 0x3FE62E42FEFA39EF, double %83) #10, !dbg !1348
  %86 = fsub double %85, %63, !dbg !1348
  %87 = fsub double %78, %86, !dbg !1348
  %88 = call double @llvm.nvvm.fma.rn.d(double %82, double 0x3C7ABC9E3B39803F, double %87) #10, !dbg !1348
  %89 = fadd double %83, %88, !dbg !1348
  br label %_ZL3logd.exit, !dbg !1348

90:                                               ; preds = %33
  %91 = call double @llvm.nvvm.fabs.d(double %27) #10, !dbg !1348
  %92 = fcmp ole double %91, 0x7FF0000000000000, !dbg !1348
  %93 = xor i1 %92, true, !dbg !1348
  %94 = zext i1 %93 to i32, !dbg !1348
  br i1 %93, label %95, label %97, !dbg !1348

95:                                               ; preds = %90
  %96 = fadd double %27, %27, !dbg !1348
  br label %106, !dbg !1348

97:                                               ; preds = %90
  %98 = fcmp oeq double %27, 0.000000e+00, !dbg !1348
  br i1 %98, label %99, label %100, !dbg !1348

99:                                               ; preds = %97
  br label %105, !dbg !1348

100:                                              ; preds = %97
  %101 = fcmp oeq double %27, 0x7FF0000000000000, !dbg !1348
  br i1 %101, label %102, label %103, !dbg !1348

102:                                              ; preds = %100
  br label %104, !dbg !1348

103:                                              ; preds = %100
  br label %104, !dbg !1348

104:                                              ; preds = %103, %102
  %q.0.i.i = phi double [ %27, %102 ], [ 0xFFF8000000000000, %103 ], !dbg !1348
  br label %105, !dbg !1348

105:                                              ; preds = %104, %99
  %q.1.i.i = phi double [ 0xFFF0000000000000, %99 ], [ %q.0.i.i, %104 ], !dbg !1348
  br label %106, !dbg !1348

106:                                              ; preds = %105, %95
  %q.2.i.i = phi double [ %96, %95 ], [ %q.1.i.i, %105 ], !dbg !1348
  br label %_ZL3logd.exit, !dbg !1348

_ZL3logd.exit:                                    ; preds = %106, %54
  %q.3.i.i = phi double [ %89, %54 ], [ %q.2.i.i, %106 ], !dbg !1348
  %mul43 = fmul contract double -2.000000e+00, %q.3.i.i, !dbg !1349
  %107 = load double, double* %t1, align 8, !dbg !1350
  %div44 = fdiv double %mul43, %107, !dbg !1351
  store double %div44, double* %x.addr.i, align 8
  %108 = load double, double* %x.addr.i, align 8, !dbg !1352
  %109 = call double @llvm.nvvm.sqrt.rn.d(double %108) #10, !dbg !1353
  store double %109, double* %t2, align 8, !dbg !1354
  %110 = load double, double* %x1, align 8, !dbg !1355
  %111 = load double, double* %t2, align 8, !dbg !1356
  %mul46 = fmul contract double %110, %111, !dbg !1357
  store double %mul46, double* %t3, align 8, !dbg !1358
  %112 = load double, double* %x2, align 8, !dbg !1359
  %113 = load double, double* %t2, align 8, !dbg !1360
  %mul47 = fmul contract double %112, %113, !dbg !1361
  store double %mul47, double* %t4, align 8, !dbg !1362
  %114 = load double, double* %t3, align 8, !dbg !1363
  store double %114, double* %f.addr.i, align 8
  %115 = load double, double* %f.addr.i, align 8, !dbg !1364
  %116 = call double @llvm.nvvm.fabs.d(double %115) #10, !dbg !1365
  %117 = load double, double* %t4, align 8, !dbg !1363
  store double %117, double* %f.addr.i141, align 8
  %118 = load double, double* %f.addr.i141, align 8, !dbg !1366
  %119 = call double @llvm.nvvm.fabs.d(double %118) #10, !dbg !1367
  %cmp50 = fcmp ogt double %116, %119, !dbg !1363
  br i1 %cmp50, label %cond.true, label %cond.false, !dbg !1363

cond.true:                                        ; preds = %_ZL3logd.exit
  %120 = load double, double* %t3, align 8, !dbg !1363
  store double %120, double* %f.addr.i142, align 8
  %121 = load double, double* %f.addr.i142, align 8, !dbg !1368
  %122 = call double @llvm.nvvm.fabs.d(double %121) #10, !dbg !1369
  br label %cond.end, !dbg !1363

cond.false:                                       ; preds = %_ZL3logd.exit
  %123 = load double, double* %t4, align 8, !dbg !1363
  store double %123, double* %f.addr.i143, align 8
  %124 = load double, double* %f.addr.i143, align 8, !dbg !1370
  %125 = call double @llvm.nvvm.fabs.d(double %124) #10, !dbg !1371
  br label %cond.end, !dbg !1363

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi double [ %122, %cond.true ], [ %125, %cond.false ], !dbg !1363
  %conv = fptosi double %cond to i32, !dbg !1363
  store i32 %conv, i32* %l, align 4, !dbg !1372
  %126 = load i32, i32* %l, align 4, !dbg !1373
  %idxprom53 = sext i32 %126 to i64, !dbg !1374
  %arrayidx54 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 %idxprom53, !dbg !1374
  %127 = load double, double* %arrayidx54, align 8, !dbg !1375
  %add55 = fadd contract double %127, 1.000000e+00, !dbg !1375
  store double %add55, double* %arrayidx54, align 8, !dbg !1375
  %128 = load double, double* %sx_local, align 8, !dbg !1376
  %129 = load double, double* %t3, align 8, !dbg !1377
  %add56 = fadd contract double %128, %129, !dbg !1378
  store double %add56, double* %sx_local, align 8, !dbg !1379
  %130 = load double, double* %t4, align 8, !dbg !1380
  %131 = load double, double* %sy_local, align 8, !dbg !1381
  %add57 = fadd contract double %131, %130, !dbg !1381
  store double %add57, double* %sy_local, align 8, !dbg !1381
  br label %if.end58, !dbg !1382

if.end58:                                         ; preds = %cond.end, %for.body27
  br label %for.inc59, !dbg !1383

for.inc59:                                        ; preds = %if.end58
  %132 = load i32, i32* %i, align 4, !dbg !1384
  %inc60 = add nsw i32 %132, 1, !dbg !1384
  store i32 %inc60, i32* %i, align 4, !dbg !1384
  br label %for.cond25, !dbg !1385, !llvm.loop !1386

for.end61:                                        ; preds = %for.cond25
  br label %for.inc62, !dbg !1388

for.inc62:                                        ; preds = %for.end61
  %133 = load i32, i32* %ii, align 4, !dbg !1389
  %add63 = add nsw i32 %133, 128, !dbg !1390
  store i32 %add63, i32* %ii, align 4, !dbg !1391
  br label %for.cond22, !dbg !1392, !llvm.loop !1393

for.end64:                                        ; preds = %for.cond22
  %134 = load double*, double** %q_global.addr, align 8, !dbg !1395
  %135 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #10, !dbg !1396, !range !1183
  %mul66 = mul i32 %135, 10, !dbg !1398
  %idx.ext = zext i32 %mul66 to i64, !dbg !1399
  %add.ptr = getelementptr inbounds double, double* %134, i64 %idx.ext, !dbg !1399
  %add.ptr67 = getelementptr inbounds double, double* %add.ptr, i64 0, !dbg !1400
  %arrayidx68 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 0, !dbg !1401
  %136 = load double, double* %arrayidx68, align 8, !dbg !1401
  %call69 = call double @_ZL9atomicAddPdd(double* %add.ptr67, double %136) #11, !dbg !1402
  %137 = load double*, double** %q_global.addr, align 8, !dbg !1403
  %138 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #10, !dbg !1404, !range !1183
  %mul71 = mul i32 %138, 10, !dbg !1406
  %idx.ext72 = zext i32 %mul71 to i64, !dbg !1407
  %add.ptr73 = getelementptr inbounds double, double* %137, i64 %idx.ext72, !dbg !1407
  %add.ptr74 = getelementptr inbounds double, double* %add.ptr73, i64 1, !dbg !1408
  %arrayidx75 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 1, !dbg !1409
  %139 = load double, double* %arrayidx75, align 8, !dbg !1409
  %call76 = call double @_ZL9atomicAddPdd(double* %add.ptr74, double %139) #11, !dbg !1410
  %140 = load double*, double** %q_global.addr, align 8, !dbg !1411
  %141 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #10, !dbg !1412, !range !1183
  %mul78 = mul i32 %141, 10, !dbg !1414
  %idx.ext79 = zext i32 %mul78 to i64, !dbg !1415
  %add.ptr80 = getelementptr inbounds double, double* %140, i64 %idx.ext79, !dbg !1415
  %add.ptr81 = getelementptr inbounds double, double* %add.ptr80, i64 2, !dbg !1416
  %arrayidx82 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 2, !dbg !1417
  %142 = load double, double* %arrayidx82, align 8, !dbg !1417
  %call83 = call double @_ZL9atomicAddPdd(double* %add.ptr81, double %142) #11, !dbg !1418
  %143 = load double*, double** %q_global.addr, align 8, !dbg !1419
  %144 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #10, !dbg !1420, !range !1183
  %mul85 = mul i32 %144, 10, !dbg !1422
  %idx.ext86 = zext i32 %mul85 to i64, !dbg !1423
  %add.ptr87 = getelementptr inbounds double, double* %143, i64 %idx.ext86, !dbg !1423
  %add.ptr88 = getelementptr inbounds double, double* %add.ptr87, i64 3, !dbg !1424
  %arrayidx89 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 3, !dbg !1425
  %145 = load double, double* %arrayidx89, align 8, !dbg !1425
  %call90 = call double @_ZL9atomicAddPdd(double* %add.ptr88, double %145) #11, !dbg !1426
  %146 = load double*, double** %q_global.addr, align 8, !dbg !1427
  %147 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #10, !dbg !1428, !range !1183
  %mul92 = mul i32 %147, 10, !dbg !1430
  %idx.ext93 = zext i32 %mul92 to i64, !dbg !1431
  %add.ptr94 = getelementptr inbounds double, double* %146, i64 %idx.ext93, !dbg !1431
  %add.ptr95 = getelementptr inbounds double, double* %add.ptr94, i64 4, !dbg !1432
  %arrayidx96 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 4, !dbg !1433
  %148 = load double, double* %arrayidx96, align 8, !dbg !1433
  %call97 = call double @_ZL9atomicAddPdd(double* %add.ptr95, double %148) #11, !dbg !1434
  %149 = load double*, double** %q_global.addr, align 8, !dbg !1435
  %150 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #10, !dbg !1436, !range !1183
  %mul99 = mul i32 %150, 10, !dbg !1438
  %idx.ext100 = zext i32 %mul99 to i64, !dbg !1439
  %add.ptr101 = getelementptr inbounds double, double* %149, i64 %idx.ext100, !dbg !1439
  %add.ptr102 = getelementptr inbounds double, double* %add.ptr101, i64 5, !dbg !1440
  %arrayidx103 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 5, !dbg !1441
  %151 = load double, double* %arrayidx103, align 8, !dbg !1441
  %call104 = call double @_ZL9atomicAddPdd(double* %add.ptr102, double %151) #11, !dbg !1442
  %152 = load double*, double** %q_global.addr, align 8, !dbg !1443
  %153 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #10, !dbg !1444, !range !1183
  %mul106 = mul i32 %153, 10, !dbg !1446
  %idx.ext107 = zext i32 %mul106 to i64, !dbg !1447
  %add.ptr108 = getelementptr inbounds double, double* %152, i64 %idx.ext107, !dbg !1447
  %add.ptr109 = getelementptr inbounds double, double* %add.ptr108, i64 6, !dbg !1448
  %arrayidx110 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 6, !dbg !1449
  %154 = load double, double* %arrayidx110, align 8, !dbg !1449
  %call111 = call double @_ZL9atomicAddPdd(double* %add.ptr109, double %154) #11, !dbg !1450
  %155 = load double*, double** %q_global.addr, align 8, !dbg !1451
  %156 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #10, !dbg !1452, !range !1183
  %mul113 = mul i32 %156, 10, !dbg !1454
  %idx.ext114 = zext i32 %mul113 to i64, !dbg !1455
  %add.ptr115 = getelementptr inbounds double, double* %155, i64 %idx.ext114, !dbg !1455
  %add.ptr116 = getelementptr inbounds double, double* %add.ptr115, i64 7, !dbg !1456
  %arrayidx117 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 7, !dbg !1457
  %157 = load double, double* %arrayidx117, align 8, !dbg !1457
  %call118 = call double @_ZL9atomicAddPdd(double* %add.ptr116, double %157) #11, !dbg !1458
  %158 = load double*, double** %q_global.addr, align 8, !dbg !1459
  %159 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #10, !dbg !1460, !range !1183
  %mul120 = mul i32 %159, 10, !dbg !1462
  %idx.ext121 = zext i32 %mul120 to i64, !dbg !1463
  %add.ptr122 = getelementptr inbounds double, double* %158, i64 %idx.ext121, !dbg !1463
  %add.ptr123 = getelementptr inbounds double, double* %add.ptr122, i64 8, !dbg !1464
  %arrayidx124 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 8, !dbg !1465
  %160 = load double, double* %arrayidx124, align 8, !dbg !1465
  %call125 = call double @_ZL9atomicAddPdd(double* %add.ptr123, double %160) #11, !dbg !1466
  %161 = load double*, double** %q_global.addr, align 8, !dbg !1467
  %162 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #10, !dbg !1468, !range !1183
  %mul127 = mul i32 %162, 10, !dbg !1470
  %idx.ext128 = zext i32 %mul127 to i64, !dbg !1471
  %add.ptr129 = getelementptr inbounds double, double* %161, i64 %idx.ext128, !dbg !1471
  %add.ptr130 = getelementptr inbounds double, double* %add.ptr129, i64 9, !dbg !1472
  %arrayidx131 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 9, !dbg !1473
  %163 = load double, double* %arrayidx131, align 8, !dbg !1473
  %call132 = call double @_ZL9atomicAddPdd(double* %add.ptr130, double %163) #11, !dbg !1474
  %164 = load double*, double** %sx_global.addr, align 8, !dbg !1475
  %165 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #10, !dbg !1476, !range !1183
  %idx.ext134 = zext i32 %165 to i64, !dbg !1478
  %add.ptr135 = getelementptr inbounds double, double* %164, i64 %idx.ext134, !dbg !1478
  %166 = load double, double* %sx_local, align 8, !dbg !1479
  %call136 = call double @_ZL9atomicAddPdd(double* %add.ptr135, double %166) #11, !dbg !1480
  %167 = load double*, double** %sy_global.addr, align 8, !dbg !1481
  %168 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #10, !dbg !1482, !range !1183
  %idx.ext138 = zext i32 %168 to i64, !dbg !1484
  %add.ptr139 = getelementptr inbounds double, double* %167, i64 %idx.ext138, !dbg !1484
  %169 = load double, double* %sy_local, align 8, !dbg !1485
  %call140 = call double @_ZL9atomicAddPdd(double* %add.ptr139, double %169) #11, !dbg !1486
  br label %return, !dbg !1487

return:                                           ; preds = %for.end64, %if.then
  ret void, !dbg !1487
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
define dso_local double @_Z13randlc_devicePdd(double* %x, double %a) #3 !dbg !1488 {
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
  call void @llvm.dbg.declare(metadata double** %x.addr, metadata !1491, metadata !DIExpression()), !dbg !1492
  store double %a, double* %a.addr, align 8
  call void @llvm.dbg.declare(metadata double* %a.addr, metadata !1493, metadata !DIExpression()), !dbg !1494
  call void @llvm.dbg.declare(metadata double* %t1, metadata !1495, metadata !DIExpression()), !dbg !1496
  call void @llvm.dbg.declare(metadata double* %t2, metadata !1497, metadata !DIExpression()), !dbg !1498
  call void @llvm.dbg.declare(metadata double* %t3, metadata !1499, metadata !DIExpression()), !dbg !1500
  call void @llvm.dbg.declare(metadata double* %t4, metadata !1501, metadata !DIExpression()), !dbg !1502
  call void @llvm.dbg.declare(metadata double* %a1, metadata !1503, metadata !DIExpression()), !dbg !1504
  call void @llvm.dbg.declare(metadata double* %a2, metadata !1505, metadata !DIExpression()), !dbg !1506
  call void @llvm.dbg.declare(metadata double* %x1, metadata !1507, metadata !DIExpression()), !dbg !1508
  call void @llvm.dbg.declare(metadata double* %x2, metadata !1509, metadata !DIExpression()), !dbg !1510
  call void @llvm.dbg.declare(metadata double* %z, metadata !1511, metadata !DIExpression()), !dbg !1512
  %0 = load double, double* %a.addr, align 8, !dbg !1513
  %mul = fmul contract double 0x3E80000000000000, %0, !dbg !1514
  store double %mul, double* %t1, align 8, !dbg !1515
  %1 = load double, double* %t1, align 8, !dbg !1516
  %conv = fptosi double %1 to i32, !dbg !1516
  %conv1 = sitofp i32 %conv to double, !dbg !1517
  store double %conv1, double* %a1, align 8, !dbg !1518
  %2 = load double, double* %a.addr, align 8, !dbg !1519
  %3 = load double, double* %a1, align 8, !dbg !1520
  %mul2 = fmul contract double 0x4160000000000000, %3, !dbg !1521
  %sub = fsub contract double %2, %mul2, !dbg !1522
  store double %sub, double* %a2, align 8, !dbg !1523
  %4 = load double*, double** %x.addr, align 8, !dbg !1524
  %5 = load double, double* %4, align 8, !dbg !1525
  %mul3 = fmul contract double 0x3E80000000000000, %5, !dbg !1526
  store double %mul3, double* %t1, align 8, !dbg !1527
  %6 = load double, double* %t1, align 8, !dbg !1528
  %conv4 = fptosi double %6 to i32, !dbg !1528
  %conv5 = sitofp i32 %conv4 to double, !dbg !1529
  store double %conv5, double* %x1, align 8, !dbg !1530
  %7 = load double*, double** %x.addr, align 8, !dbg !1531
  %8 = load double, double* %7, align 8, !dbg !1532
  %9 = load double, double* %x1, align 8, !dbg !1533
  %mul6 = fmul contract double 0x4160000000000000, %9, !dbg !1534
  %sub7 = fsub contract double %8, %mul6, !dbg !1535
  store double %sub7, double* %x2, align 8, !dbg !1536
  %10 = load double, double* %a1, align 8, !dbg !1537
  %11 = load double, double* %x2, align 8, !dbg !1538
  %mul8 = fmul contract double %10, %11, !dbg !1539
  %12 = load double, double* %a2, align 8, !dbg !1540
  %13 = load double, double* %x1, align 8, !dbg !1541
  %mul9 = fmul contract double %12, %13, !dbg !1542
  %add = fadd contract double %mul8, %mul9, !dbg !1543
  store double %add, double* %t1, align 8, !dbg !1544
  %14 = load double, double* %t1, align 8, !dbg !1545
  %mul10 = fmul contract double 0x3E80000000000000, %14, !dbg !1546
  %conv11 = fptosi double %mul10 to i32, !dbg !1547
  %conv12 = sitofp i32 %conv11 to double, !dbg !1548
  store double %conv12, double* %t2, align 8, !dbg !1549
  %15 = load double, double* %t1, align 8, !dbg !1550
  %16 = load double, double* %t2, align 8, !dbg !1551
  %mul13 = fmul contract double 0x4160000000000000, %16, !dbg !1552
  %sub14 = fsub contract double %15, %mul13, !dbg !1553
  store double %sub14, double* %z, align 8, !dbg !1554
  %17 = load double, double* %z, align 8, !dbg !1555
  %mul15 = fmul contract double 0x4160000000000000, %17, !dbg !1556
  %18 = load double, double* %a2, align 8, !dbg !1557
  %19 = load double, double* %x2, align 8, !dbg !1558
  %mul16 = fmul contract double %18, %19, !dbg !1559
  %add17 = fadd contract double %mul15, %mul16, !dbg !1560
  store double %add17, double* %t3, align 8, !dbg !1561
  %20 = load double, double* %t3, align 8, !dbg !1562
  %mul18 = fmul contract double 0x3D10000000000000, %20, !dbg !1563
  %conv19 = fptosi double %mul18 to i32, !dbg !1564
  %conv20 = sitofp i32 %conv19 to double, !dbg !1565
  store double %conv20, double* %t4, align 8, !dbg !1566
  %21 = load double, double* %t3, align 8, !dbg !1567
  %22 = load double, double* %t4, align 8, !dbg !1568
  %mul21 = fmul contract double 0x42D0000000000000, %22, !dbg !1569
  %sub22 = fsub contract double %21, %mul21, !dbg !1570
  %23 = load double*, double** %x.addr, align 8, !dbg !1571
  store double %sub22, double* %23, align 8, !dbg !1572
  %24 = load double*, double** %x.addr, align 8, !dbg !1573
  %25 = load double, double* %24, align 8, !dbg !1574
  %mul23 = fmul contract double 0x3D10000000000000, %25, !dbg !1575
  ret double %mul23, !dbg !1576
}

; Function Attrs: convergent noinline nounwind
define dso_local void @_Z13vranlc_deviceiPddS_(i32 %n, double* %x_seed, double %a, double* %y) #3 !dbg !1577 {
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
  call void @llvm.dbg.declare(metadata i32* %n.addr, metadata !1580, metadata !DIExpression()), !dbg !1581
  store double* %x_seed, double** %x_seed.addr, align 8
  call void @llvm.dbg.declare(metadata double** %x_seed.addr, metadata !1582, metadata !DIExpression()), !dbg !1583
  store double %a, double* %a.addr, align 8
  call void @llvm.dbg.declare(metadata double* %a.addr, metadata !1584, metadata !DIExpression()), !dbg !1585
  store double* %y, double** %y.addr, align 8
  call void @llvm.dbg.declare(metadata double** %y.addr, metadata !1586, metadata !DIExpression()), !dbg !1587
  call void @llvm.dbg.declare(metadata i32* %i, metadata !1588, metadata !DIExpression()), !dbg !1589
  call void @llvm.dbg.declare(metadata double* %x, metadata !1590, metadata !DIExpression()), !dbg !1591
  call void @llvm.dbg.declare(metadata double* %t1, metadata !1592, metadata !DIExpression()), !dbg !1593
  call void @llvm.dbg.declare(metadata double* %t2, metadata !1594, metadata !DIExpression()), !dbg !1595
  call void @llvm.dbg.declare(metadata double* %t3, metadata !1596, metadata !DIExpression()), !dbg !1597
  call void @llvm.dbg.declare(metadata double* %t4, metadata !1598, metadata !DIExpression()), !dbg !1599
  call void @llvm.dbg.declare(metadata double* %a1, metadata !1600, metadata !DIExpression()), !dbg !1601
  call void @llvm.dbg.declare(metadata double* %a2, metadata !1602, metadata !DIExpression()), !dbg !1603
  call void @llvm.dbg.declare(metadata double* %x1, metadata !1604, metadata !DIExpression()), !dbg !1605
  call void @llvm.dbg.declare(metadata double* %x2, metadata !1606, metadata !DIExpression()), !dbg !1607
  call void @llvm.dbg.declare(metadata double* %z, metadata !1608, metadata !DIExpression()), !dbg !1609
  %0 = load double, double* %a.addr, align 8, !dbg !1610
  %mul = fmul contract double 0x3E80000000000000, %0, !dbg !1611
  store double %mul, double* %t1, align 8, !dbg !1612
  %1 = load double, double* %t1, align 8, !dbg !1613
  %conv = fptosi double %1 to i32, !dbg !1613
  %conv1 = sitofp i32 %conv to double, !dbg !1614
  store double %conv1, double* %a1, align 8, !dbg !1615
  %2 = load double, double* %a.addr, align 8, !dbg !1616
  %3 = load double, double* %a1, align 8, !dbg !1617
  %mul2 = fmul contract double 0x4160000000000000, %3, !dbg !1618
  %sub = fsub contract double %2, %mul2, !dbg !1619
  store double %sub, double* %a2, align 8, !dbg !1620
  %4 = load double*, double** %x_seed.addr, align 8, !dbg !1621
  %5 = load double, double* %4, align 8, !dbg !1622
  store double %5, double* %x, align 8, !dbg !1623
  store i32 0, i32* %i, align 4, !dbg !1624
  br label %for.cond, !dbg !1626

for.cond:                                         ; preds = %for.inc, %entry
  %6 = load i32, i32* %i, align 4, !dbg !1627
  %7 = load i32, i32* %n.addr, align 4, !dbg !1629
  %cmp = icmp slt i32 %6, %7, !dbg !1630
  br i1 %cmp, label %for.body, label %for.end, !dbg !1631

for.body:                                         ; preds = %for.cond
  %8 = load double, double* %x, align 8, !dbg !1632
  %mul3 = fmul contract double 0x3E80000000000000, %8, !dbg !1634
  store double %mul3, double* %t1, align 8, !dbg !1635
  %9 = load double, double* %t1, align 8, !dbg !1636
  %conv4 = fptosi double %9 to i32, !dbg !1636
  %conv5 = sitofp i32 %conv4 to double, !dbg !1637
  store double %conv5, double* %x1, align 8, !dbg !1638
  %10 = load double, double* %x, align 8, !dbg !1639
  %11 = load double, double* %x1, align 8, !dbg !1640
  %mul6 = fmul contract double 0x4160000000000000, %11, !dbg !1641
  %sub7 = fsub contract double %10, %mul6, !dbg !1642
  store double %sub7, double* %x2, align 8, !dbg !1643
  %12 = load double, double* %a1, align 8, !dbg !1644
  %13 = load double, double* %x2, align 8, !dbg !1645
  %mul8 = fmul contract double %12, %13, !dbg !1646
  %14 = load double, double* %a2, align 8, !dbg !1647
  %15 = load double, double* %x1, align 8, !dbg !1648
  %mul9 = fmul contract double %14, %15, !dbg !1649
  %add = fadd contract double %mul8, %mul9, !dbg !1650
  store double %add, double* %t1, align 8, !dbg !1651
  %16 = load double, double* %t1, align 8, !dbg !1652
  %mul10 = fmul contract double 0x3E80000000000000, %16, !dbg !1653
  %conv11 = fptosi double %mul10 to i32, !dbg !1654
  %conv12 = sitofp i32 %conv11 to double, !dbg !1655
  store double %conv12, double* %t2, align 8, !dbg !1656
  %17 = load double, double* %t1, align 8, !dbg !1657
  %18 = load double, double* %t2, align 8, !dbg !1658
  %mul13 = fmul contract double 0x4160000000000000, %18, !dbg !1659
  %sub14 = fsub contract double %17, %mul13, !dbg !1660
  store double %sub14, double* %z, align 8, !dbg !1661
  %19 = load double, double* %z, align 8, !dbg !1662
  %mul15 = fmul contract double 0x4160000000000000, %19, !dbg !1663
  %20 = load double, double* %a2, align 8, !dbg !1664
  %21 = load double, double* %x2, align 8, !dbg !1665
  %mul16 = fmul contract double %20, %21, !dbg !1666
  %add17 = fadd contract double %mul15, %mul16, !dbg !1667
  store double %add17, double* %t3, align 8, !dbg !1668
  %22 = load double, double* %t3, align 8, !dbg !1669
  %mul18 = fmul contract double 0x3D10000000000000, %22, !dbg !1670
  %conv19 = fptosi double %mul18 to i32, !dbg !1671
  %conv20 = sitofp i32 %conv19 to double, !dbg !1672
  store double %conv20, double* %t4, align 8, !dbg !1673
  %23 = load double, double* %t3, align 8, !dbg !1674
  %24 = load double, double* %t4, align 8, !dbg !1675
  %mul21 = fmul contract double 0x42D0000000000000, %24, !dbg !1676
  %sub22 = fsub contract double %23, %mul21, !dbg !1677
  store double %sub22, double* %x, align 8, !dbg !1678
  %25 = load double, double* %x, align 8, !dbg !1679
  %mul23 = fmul contract double 0x3D10000000000000, %25, !dbg !1680
  %26 = load double*, double** %y.addr, align 8, !dbg !1681
  %27 = load i32, i32* %i, align 4, !dbg !1682
  %idxprom = sext i32 %27 to i64, !dbg !1681
  %arrayidx = getelementptr inbounds double, double* %26, i64 %idxprom, !dbg !1681
  store double %mul23, double* %arrayidx, align 8, !dbg !1683
  br label %for.inc, !dbg !1684

for.inc:                                          ; preds = %for.body
  %28 = load i32, i32* %i, align 4, !dbg !1685
  %inc = add nsw i32 %28, 1, !dbg !1685
  store i32 %inc, i32* %i, align 4, !dbg !1685
  br label %for.cond, !dbg !1686, !llvm.loop !1687

for.end:                                          ; preds = %for.cond
  %29 = load double, double* %x, align 8, !dbg !1689
  %30 = load double*, double** %x_seed.addr, align 8, !dbg !1690
  store double %29, double* %30, align 8, !dbg !1691
  ret void, !dbg !1692
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

; Function Attrs: convergent noinline nounwind
define internal double @_ZL9atomicAddPdd(double* %address, double %val) #0 !dbg !1693 {
entry:
  %x.addr.i13 = alloca double, align 8
  call void @llvm.dbg.declare(metadata double* %x.addr.i13, metadata !1695, metadata !DIExpression()), !dbg !1699
  %x.addr.i12 = alloca i64, align 8
  call void @llvm.dbg.declare(metadata i64* %x.addr.i12, metadata !1704, metadata !DIExpression()), !dbg !1708
  %x.addr.i11 = alloca i64, align 8
  call void @llvm.dbg.declare(metadata i64* %x.addr.i11, metadata !1704, metadata !DIExpression()), !dbg !1710
  %x.addr.i10 = alloca i64, align 8
  call void @llvm.dbg.declare(metadata i64* %x.addr.i10, metadata !1704, metadata !DIExpression()), !dbg !1713
  %x.addr.i = alloca i64, align 8
  call void @llvm.dbg.declare(metadata i64* %x.addr.i, metadata !1704, metadata !DIExpression()), !dbg !1715
  %retval = alloca double, align 8
  %address.addr = alloca double*, align 8
  %val.addr = alloca double, align 8
  %address_as_ull = alloca i64*, align 8
  %old = alloca i64, align 8
  %assumed = alloca i64, align 8
  %i = alloca i32, align 4
  store double* %address, double** %address.addr, align 8
  call void @llvm.dbg.declare(metadata double** %address.addr, metadata !1718, metadata !DIExpression()), !dbg !1719
  store double %val, double* %val.addr, align 8
  call void @llvm.dbg.declare(metadata double* %val.addr, metadata !1720, metadata !DIExpression()), !dbg !1721
  call void @llvm.dbg.declare(metadata i64** %address_as_ull, metadata !1722, metadata !DIExpression()), !dbg !1723
  %0 = load double*, double** %address.addr, align 8, !dbg !1724
  %1 = bitcast double* %0 to i64*, !dbg !1725
  store i64* %1, i64** %address_as_ull, align 8, !dbg !1723
  call void @llvm.dbg.declare(metadata i64* %old, metadata !1726, metadata !DIExpression()), !dbg !1727
  %2 = load i64*, i64** %address_as_ull, align 8, !dbg !1728
  %3 = load i64, i64* %2, align 8, !dbg !1729
  store i64 %3, i64* %old, align 8, !dbg !1727
  call void @llvm.dbg.declare(metadata i64* %assumed, metadata !1730, metadata !DIExpression()), !dbg !1731
  %4 = load double, double* %val.addr, align 8, !dbg !1732
  %cmp = fcmp oeq double %4, 0.000000e+00, !dbg !1733
  br i1 %cmp, label %if.then, label %if.end, !dbg !1734

if.then:                                          ; preds = %entry
  %5 = load i64, i64* %old, align 8, !dbg !1735
  store i64 %5, i64* %x.addr.i, align 8
  %6 = load i64, i64* %x.addr.i, align 8, !dbg !1736
  %7 = bitcast i64 %6 to double, !dbg !1737
  store double %7, double* %retval, align 8, !dbg !1738
  br label %return, !dbg !1738

if.end:                                           ; preds = %entry
  call void @llvm.dbg.declare(metadata i32* %i, metadata !1739, metadata !DIExpression()), !dbg !1740
  store i32 0, i32* %i, align 4, !dbg !1740
  br label %for.cond, !dbg !1741

for.cond:                                         ; preds = %for.inc, %if.end
  %8 = load i32, i32* %i, align 4, !dbg !1742
  %cmp1 = icmp slt i32 %8, 100000, !dbg !1743
  br i1 %cmp1, label %for.body, label %for.end, !dbg !1744

for.body:                                         ; preds = %for.cond
  %9 = load i64, i64* %old, align 8, !dbg !1745
  store i64 %9, i64* %assumed, align 8, !dbg !1746
  %10 = load i64*, i64** %address_as_ull, align 8, !dbg !1747
  %11 = load i64, i64* %assumed, align 8, !dbg !1748
  %12 = load double, double* %val.addr, align 8, !dbg !1749
  %13 = load i64, i64* %assumed, align 8, !dbg !1750
  store i64 %13, i64* %x.addr.i12, align 8
  %14 = load i64, i64* %x.addr.i12, align 8, !dbg !1751
  %15 = bitcast i64 %14 to double, !dbg !1752
  %add = fadd contract double %12, %15, !dbg !1753
  store double %add, double* %x.addr.i13, align 8
  %16 = load double, double* %x.addr.i13, align 8, !dbg !1754
  %17 = bitcast double %16 to i64, !dbg !1755
  %call4 = call i64 @_ZL9atomicCASPyyy(i64* %10, i64 %11, i64 %17) #11, !dbg !1756
  store i64 %call4, i64* %old, align 8, !dbg !1757
  %18 = load i64, i64* %assumed, align 8, !dbg !1758
  %19 = load i64, i64* %old, align 8, !dbg !1759
  %cmp5 = icmp eq i64 %18, %19, !dbg !1760
  br i1 %cmp5, label %if.then6, label %if.end8, !dbg !1761

if.then6:                                         ; preds = %for.body
  %20 = load i64, i64* %old, align 8, !dbg !1762
  store i64 %20, i64* %x.addr.i11, align 8
  %21 = load i64, i64* %x.addr.i11, align 8, !dbg !1763
  %22 = bitcast i64 %21 to double, !dbg !1764
  store double %22, double* %retval, align 8, !dbg !1765
  br label %return, !dbg !1765

if.end8:                                          ; preds = %for.body
  br label %for.inc, !dbg !1766

for.inc:                                          ; preds = %if.end8
  %23 = load i32, i32* %i, align 4, !dbg !1767
  %inc = add nsw i32 %23, 1, !dbg !1767
  store i32 %inc, i32* %i, align 4, !dbg !1767
  br label %for.cond, !dbg !1768, !llvm.loop !1769

for.end:                                          ; preds = %for.cond
  %24 = load i64, i64* %old, align 8, !dbg !1771
  store i64 %24, i64* %x.addr.i10, align 8
  %25 = load i64, i64* %x.addr.i10, align 8, !dbg !1772
  %26 = bitcast i64 %25 to double, !dbg !1773
  store double %26, double* %retval, align 8, !dbg !1774
  br label %return, !dbg !1774

return:                                           ; preds = %for.end, %if.then6, %if.then
  %27 = load double, double* %retval, align 8, !dbg !1775
  ret double %27, !dbg !1775
}

; Function Attrs: convergent noinline nounwind
define internal i64 @_ZL9atomicCASPyyy(i64* %address, i64 %compare, i64 %val) #3 !dbg !1776 {
entry:
  %p.addr.i = alloca i64*, align 8
  call void @llvm.dbg.declare(metadata i64** %p.addr.i, metadata !1780, metadata !DIExpression()), !dbg !1782
  %compare.addr.i = alloca i64, align 8
  call void @llvm.dbg.declare(metadata i64* %compare.addr.i, metadata !1784, metadata !DIExpression()), !dbg !1785
  %val.addr.i = alloca i64, align 8
  call void @llvm.dbg.declare(metadata i64* %val.addr.i, metadata !1786, metadata !DIExpression()), !dbg !1787
  %address.addr = alloca i64*, align 8
  %compare.addr = alloca i64, align 8
  %val.addr = alloca i64, align 8
  store i64* %address, i64** %address.addr, align 8
  call void @llvm.dbg.declare(metadata i64** %address.addr, metadata !1788, metadata !DIExpression()), !dbg !1789
  store i64 %compare, i64* %compare.addr, align 8
  call void @llvm.dbg.declare(metadata i64* %compare.addr, metadata !1790, metadata !DIExpression()), !dbg !1791
  store i64 %val, i64* %val.addr, align 8
  call void @llvm.dbg.declare(metadata i64* %val.addr, metadata !1792, metadata !DIExpression()), !dbg !1793
  %0 = load i64*, i64** %address.addr, align 8, !dbg !1794
  %1 = load i64, i64* %compare.addr, align 8, !dbg !1795
  %2 = load i64, i64* %val.addr, align 8, !dbg !1796
  store i64* %0, i64** %p.addr.i, align 8
  store i64 %1, i64* %compare.addr.i, align 8
  store i64 %2, i64* %val.addr.i, align 8
  %3 = load i64*, i64** %p.addr.i, align 8, !dbg !1797
  %4 = load i64, i64* %compare.addr.i, align 8, !dbg !1798
  %5 = load i64, i64* %val.addr.i, align 8, !dbg !1799
  %6 = cmpxchg i64* %3, i64 %4, i64 %5 seq_cst seq_cst, !dbg !1800
  %7 = extractvalue { i64, i1 } %6, 0, !dbg !1800
  ret i64 %7, !dbg !1801
}

; Function Attrs: noinline nounwind uwtable
define dso_local double @_Z6randlcPdd(double* %x, double %a) #4 !dbg !1802 {
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
  call void @llvm.dbg.declare(metadata double** %x.addr, metadata !1803, metadata !DIExpression()), !dbg !1804
  store double %a, double* %a.addr, align 8
  call void @llvm.dbg.declare(metadata double* %a.addr, metadata !1805, metadata !DIExpression()), !dbg !1806
  call void @llvm.dbg.declare(metadata double* %t1, metadata !1807, metadata !DIExpression()), !dbg !1808
  call void @llvm.dbg.declare(metadata double* %t2, metadata !1809, metadata !DIExpression()), !dbg !1810
  call void @llvm.dbg.declare(metadata double* %t3, metadata !1811, metadata !DIExpression()), !dbg !1812
  call void @llvm.dbg.declare(metadata double* %t4, metadata !1813, metadata !DIExpression()), !dbg !1814
  call void @llvm.dbg.declare(metadata double* %a1, metadata !1815, metadata !DIExpression()), !dbg !1816
  call void @llvm.dbg.declare(metadata double* %a2, metadata !1817, metadata !DIExpression()), !dbg !1818
  call void @llvm.dbg.declare(metadata double* %x1, metadata !1819, metadata !DIExpression()), !dbg !1820
  call void @llvm.dbg.declare(metadata double* %x2, metadata !1821, metadata !DIExpression()), !dbg !1822
  call void @llvm.dbg.declare(metadata double* %z, metadata !1823, metadata !DIExpression()), !dbg !1824
  %0 = load double, double* %a.addr, align 8, !dbg !1825
  %mul = fmul contract double 0x3E80000000000000, %0, !dbg !1826
  store double %mul, double* %t1, align 8, !dbg !1827
  %1 = load double, double* %t1, align 8, !dbg !1828
  %conv = fptosi double %1 to i32, !dbg !1828
  %conv1 = sitofp i32 %conv to double, !dbg !1829
  store double %conv1, double* %a1, align 8, !dbg !1830
  %2 = load double, double* %a.addr, align 8, !dbg !1831
  %3 = load double, double* %a1, align 8, !dbg !1832
  %mul2 = fmul contract double 0x4160000000000000, %3, !dbg !1833
  %sub = fsub contract double %2, %mul2, !dbg !1834
  store double %sub, double* %a2, align 8, !dbg !1835
  %4 = load double*, double** %x.addr, align 8, !dbg !1836
  %5 = load double, double* %4, align 8, !dbg !1837
  %mul3 = fmul contract double 0x3E80000000000000, %5, !dbg !1838
  store double %mul3, double* %t1, align 8, !dbg !1839
  %6 = load double, double* %t1, align 8, !dbg !1840
  %conv4 = fptosi double %6 to i32, !dbg !1840
  %conv5 = sitofp i32 %conv4 to double, !dbg !1841
  store double %conv5, double* %x1, align 8, !dbg !1842
  %7 = load double*, double** %x.addr, align 8, !dbg !1843
  %8 = load double, double* %7, align 8, !dbg !1844
  %9 = load double, double* %x1, align 8, !dbg !1845
  %mul6 = fmul contract double 0x4160000000000000, %9, !dbg !1846
  %sub7 = fsub contract double %8, %mul6, !dbg !1847
  store double %sub7, double* %x2, align 8, !dbg !1848
  %10 = load double, double* %a1, align 8, !dbg !1849
  %11 = load double, double* %x2, align 8, !dbg !1850
  %mul8 = fmul contract double %10, %11, !dbg !1851
  %12 = load double, double* %a2, align 8, !dbg !1852
  %13 = load double, double* %x1, align 8, !dbg !1853
  %mul9 = fmul contract double %12, %13, !dbg !1854
  %add = fadd contract double %mul8, %mul9, !dbg !1855
  store double %add, double* %t1, align 8, !dbg !1856
  %14 = load double, double* %t1, align 8, !dbg !1857
  %mul10 = fmul contract double 0x3E80000000000000, %14, !dbg !1858
  %conv11 = fptosi double %mul10 to i32, !dbg !1859
  %conv12 = sitofp i32 %conv11 to double, !dbg !1860
  store double %conv12, double* %t2, align 8, !dbg !1861
  %15 = load double, double* %t1, align 8, !dbg !1862
  %16 = load double, double* %t2, align 8, !dbg !1863
  %mul13 = fmul contract double 0x4160000000000000, %16, !dbg !1864
  %sub14 = fsub contract double %15, %mul13, !dbg !1865
  store double %sub14, double* %z, align 8, !dbg !1866
  %17 = load double, double* %z, align 8, !dbg !1867
  %mul15 = fmul contract double 0x4160000000000000, %17, !dbg !1868
  %18 = load double, double* %a2, align 8, !dbg !1869
  %19 = load double, double* %x2, align 8, !dbg !1870
  %mul16 = fmul contract double %18, %19, !dbg !1871
  %add17 = fadd contract double %mul15, %mul16, !dbg !1872
  store double %add17, double* %t3, align 8, !dbg !1873
  %20 = load double, double* %t3, align 8, !dbg !1874
  %mul18 = fmul contract double 0x3D10000000000000, %20, !dbg !1875
  %conv19 = fptosi double %mul18 to i32, !dbg !1876
  %conv20 = sitofp i32 %conv19 to double, !dbg !1877
  store double %conv20, double* %t4, align 8, !dbg !1878
  %21 = load double, double* %t3, align 8, !dbg !1879
  %22 = load double, double* %t4, align 8, !dbg !1880
  %mul21 = fmul contract double 0x42D0000000000000, %22, !dbg !1881
  %sub22 = fsub contract double %21, %mul21, !dbg !1882
  %23 = load double*, double** %x.addr, align 8, !dbg !1883
  store double %sub22, double* %23, align 8, !dbg !1884
  %24 = load double*, double** %x.addr, align 8, !dbg !1885
  %25 = load double, double* %24, align 8, !dbg !1886
  %mul23 = fmul contract double 0x3D10000000000000, %25, !dbg !1887
  ret double %mul23, !dbg !1888
}

; Function Attrs: noinline uwtable
define dso_local void @_Z15c_print_resultsPcciiiiddS_iS_S_S_S_S_S_S_S_S_S_S_S_S_S_(i8* %name, i8 signext %class_npb, i32 %n1, i32 %n2, i32 %n3, i32 %niter, double %t, double %mops, i8* %optype, i32 %passed_verification, i8* %npbversion, i8* %compiletime, i8* %compilerversion, i8* %libversion, i8* %cpu_device, i8* %gpu_device, i8* %gpu_config, i8* %cc, i8* %clink, i8* %c_lib, i8* %c_inc, i8* %cflags, i8* %clinkflags, i8* %rand) #5 !dbg !1889 {
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
  call void @llvm.dbg.declare(metadata i8** %name.addr, metadata !1892, metadata !DIExpression()), !dbg !1893
  store i8 %class_npb, i8* %class_npb.addr, align 1
  call void @llvm.dbg.declare(metadata i8* %class_npb.addr, metadata !1894, metadata !DIExpression()), !dbg !1895
  store i32 %n1, i32* %n1.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %n1.addr, metadata !1896, metadata !DIExpression()), !dbg !1897
  store i32 %n2, i32* %n2.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %n2.addr, metadata !1898, metadata !DIExpression()), !dbg !1899
  store i32 %n3, i32* %n3.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %n3.addr, metadata !1900, metadata !DIExpression()), !dbg !1901
  store i32 %niter, i32* %niter.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %niter.addr, metadata !1902, metadata !DIExpression()), !dbg !1903
  store double %t, double* %t.addr, align 8
  call void @llvm.dbg.declare(metadata double* %t.addr, metadata !1904, metadata !DIExpression()), !dbg !1905
  store double %mops, double* %mops.addr, align 8
  call void @llvm.dbg.declare(metadata double* %mops.addr, metadata !1906, metadata !DIExpression()), !dbg !1907
  store i8* %optype, i8** %optype.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %optype.addr, metadata !1908, metadata !DIExpression()), !dbg !1909
  store i32 %passed_verification, i32* %passed_verification.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %passed_verification.addr, metadata !1910, metadata !DIExpression()), !dbg !1911
  store i8* %npbversion, i8** %npbversion.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %npbversion.addr, metadata !1912, metadata !DIExpression()), !dbg !1913
  store i8* %compiletime, i8** %compiletime.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %compiletime.addr, metadata !1914, metadata !DIExpression()), !dbg !1915
  store i8* %compilerversion, i8** %compilerversion.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %compilerversion.addr, metadata !1916, metadata !DIExpression()), !dbg !1917
  store i8* %libversion, i8** %libversion.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %libversion.addr, metadata !1918, metadata !DIExpression()), !dbg !1919
  store i8* %cpu_device, i8** %cpu_device.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %cpu_device.addr, metadata !1920, metadata !DIExpression()), !dbg !1921
  store i8* %gpu_device, i8** %gpu_device.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %gpu_device.addr, metadata !1922, metadata !DIExpression()), !dbg !1923
  store i8* %gpu_config, i8** %gpu_config.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %gpu_config.addr, metadata !1924, metadata !DIExpression()), !dbg !1925
  store i8* %cc, i8** %cc.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %cc.addr, metadata !1926, metadata !DIExpression()), !dbg !1927
  store i8* %clink, i8** %clink.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %clink.addr, metadata !1928, metadata !DIExpression()), !dbg !1929
  store i8* %c_lib, i8** %c_lib.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %c_lib.addr, metadata !1930, metadata !DIExpression()), !dbg !1931
  store i8* %c_inc, i8** %c_inc.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %c_inc.addr, metadata !1932, metadata !DIExpression()), !dbg !1933
  store i8* %cflags, i8** %cflags.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %cflags.addr, metadata !1934, metadata !DIExpression()), !dbg !1935
  store i8* %clinkflags, i8** %clinkflags.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %clinkflags.addr, metadata !1936, metadata !DIExpression()), !dbg !1937
  store i8* %rand, i8** %rand.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %rand.addr, metadata !1938, metadata !DIExpression()), !dbg !1939
  %0 = load i8*, i8** %name.addr, align 8, !dbg !1940
  %call = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([27 x i8], [27 x i8]* @.str, i64 0, i64 0), i8* %0), !dbg !1941
  %1 = load i8, i8* %class_npb.addr, align 1, !dbg !1942
  %conv = sext i8 %1 to i32, !dbg !1942
  %call1 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([46 x i8], [46 x i8]* @.str.1, i64 0, i64 0), i32 %conv), !dbg !1943
  %2 = load i8*, i8** %name.addr, align 8, !dbg !1944
  %arrayidx = getelementptr inbounds i8, i8* %2, i64 0, !dbg !1944
  %3 = load i8, i8* %arrayidx, align 1, !dbg !1944
  %conv2 = sext i8 %3 to i32, !dbg !1944
  %cmp = icmp eq i32 %conv2, 73, !dbg !1946
  br i1 %cmp, label %land.lhs.true, label %if.else15, !dbg !1947

land.lhs.true:                                    ; preds = %entry
  %4 = load i8*, i8** %name.addr, align 8, !dbg !1948
  %arrayidx3 = getelementptr inbounds i8, i8* %4, i64 1, !dbg !1948
  %5 = load i8, i8* %arrayidx3, align 1, !dbg !1948
  %conv4 = sext i8 %5 to i32, !dbg !1948
  %cmp5 = icmp eq i32 %conv4, 83, !dbg !1949
  br i1 %cmp5, label %if.then, label %if.else15, !dbg !1950

if.then:                                          ; preds = %land.lhs.true
  %6 = load i32, i32* %n3.addr, align 4, !dbg !1951
  %cmp6 = icmp eq i32 %6, 0, !dbg !1954
  br i1 %cmp6, label %if.then7, label %if.else, !dbg !1955

if.then7:                                         ; preds = %if.then
  call void @llvm.dbg.declare(metadata i64* %nn, metadata !1956, metadata !DIExpression()), !dbg !1958
  %7 = load i32, i32* %n1.addr, align 4, !dbg !1959
  %conv8 = sext i32 %7 to i64, !dbg !1959
  store i64 %conv8, i64* %nn, align 8, !dbg !1958
  %8 = load i32, i32* %n2.addr, align 4, !dbg !1960
  %cmp9 = icmp ne i32 %8, 0, !dbg !1962
  br i1 %cmp9, label %if.then10, label %if.end, !dbg !1963

if.then10:                                        ; preds = %if.then7
  %9 = load i32, i32* %n2.addr, align 4, !dbg !1964
  %conv11 = sext i32 %9 to i64, !dbg !1964
  %10 = load i64, i64* %nn, align 8, !dbg !1966
  %mul = mul nsw i64 %10, %conv11, !dbg !1966
  store i64 %mul, i64* %nn, align 8, !dbg !1966
  br label %if.end, !dbg !1967

if.end:                                           ; preds = %if.then10, %if.then7
  %11 = load i64, i64* %nn, align 8, !dbg !1968
  %call12 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([38 x i8], [38 x i8]* @.str.2, i64 0, i64 0), i64 %11), !dbg !1969
  br label %if.end14, !dbg !1970

if.else:                                          ; preds = %if.then
  %12 = load i32, i32* %n1.addr, align 4, !dbg !1971
  %13 = load i32, i32* %n2.addr, align 4, !dbg !1973
  %14 = load i32, i32* %n3.addr, align 4, !dbg !1974
  %call13 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([44 x i8], [44 x i8]* @.str.3, i64 0, i64 0), i32 %12, i32 %13, i32 %14), !dbg !1975
  br label %if.end14

if.end14:                                         ; preds = %if.else, %if.end
  br label %if.end48, !dbg !1976

if.else15:                                        ; preds = %land.lhs.true, %entry
  call void @llvm.dbg.declare(metadata [16 x i8]* %size, metadata !1977, metadata !DIExpression()), !dbg !1982
  call void @llvm.dbg.declare(metadata i32* %j, metadata !1983, metadata !DIExpression()), !dbg !1984
  %15 = load i32, i32* %n2.addr, align 4, !dbg !1985
  %cmp16 = icmp eq i32 %15, 0, !dbg !1987
  br i1 %cmp16, label %land.lhs.true17, label %if.else45, !dbg !1988

land.lhs.true17:                                  ; preds = %if.else15
  %16 = load i32, i32* %n3.addr, align 4, !dbg !1989
  %cmp18 = icmp eq i32 %16, 0, !dbg !1990
  br i1 %cmp18, label %if.then19, label %if.else45, !dbg !1991

if.then19:                                        ; preds = %land.lhs.true17
  %17 = load i8*, i8** %name.addr, align 8, !dbg !1992
  %arrayidx20 = getelementptr inbounds i8, i8* %17, i64 0, !dbg !1992
  %18 = load i8, i8* %arrayidx20, align 1, !dbg !1992
  %conv21 = sext i8 %18 to i32, !dbg !1992
  %cmp22 = icmp eq i32 %conv21, 69, !dbg !1995
  br i1 %cmp22, label %land.lhs.true23, label %if.else42, !dbg !1996

land.lhs.true23:                                  ; preds = %if.then19
  %19 = load i8*, i8** %name.addr, align 8, !dbg !1997
  %arrayidx24 = getelementptr inbounds i8, i8* %19, i64 1, !dbg !1997
  %20 = load i8, i8* %arrayidx24, align 1, !dbg !1997
  %conv25 = sext i8 %20 to i32, !dbg !1997
  %cmp26 = icmp eq i32 %conv25, 80, !dbg !1998
  br i1 %cmp26, label %if.then27, label %if.else42, !dbg !1999

if.then27:                                        ; preds = %land.lhs.true23
  %arraydecay = getelementptr inbounds [16 x i8], [16 x i8]* %size, i64 0, i64 0, !dbg !2000
  %21 = load i32, i32* %n1.addr, align 4, !dbg !2002
  %conv28 = sitofp i32 %21 to double, !dbg !2002
  %call29 = call double @pow(double 2.000000e+00, double %conv28) #10, !dbg !2003
  %call30 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay, i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.4, i64 0, i64 0), double %call29) #10, !dbg !2004
  store i32 14, i32* %j, align 4, !dbg !2005
  %22 = load i32, i32* %j, align 4, !dbg !2006
  %idxprom = sext i32 %22 to i64, !dbg !2008
  %arrayidx31 = getelementptr inbounds [16 x i8], [16 x i8]* %size, i64 0, i64 %idxprom, !dbg !2008
  %23 = load i8, i8* %arrayidx31, align 1, !dbg !2008
  %conv32 = sext i8 %23 to i32, !dbg !2008
  %cmp33 = icmp eq i32 %conv32, 46, !dbg !2009
  br i1 %cmp33, label %if.then34, label %if.end37, !dbg !2010

if.then34:                                        ; preds = %if.then27
  %24 = load i32, i32* %j, align 4, !dbg !2011
  %idxprom35 = sext i32 %24 to i64, !dbg !2013
  %arrayidx36 = getelementptr inbounds [16 x i8], [16 x i8]* %size, i64 0, i64 %idxprom35, !dbg !2013
  store i8 32, i8* %arrayidx36, align 1, !dbg !2014
  %25 = load i32, i32* %j, align 4, !dbg !2015
  %dec = add nsw i32 %25, -1, !dbg !2015
  store i32 %dec, i32* %j, align 4, !dbg !2015
  br label %if.end37, !dbg !2016

if.end37:                                         ; preds = %if.then34, %if.then27
  %26 = load i32, i32* %j, align 4, !dbg !2017
  %add = add nsw i32 %26, 1, !dbg !2018
  %idxprom38 = sext i32 %add to i64, !dbg !2019
  %arrayidx39 = getelementptr inbounds [16 x i8], [16 x i8]* %size, i64 0, i64 %idxprom38, !dbg !2019
  store i8 0, i8* %arrayidx39, align 1, !dbg !2020
  %arraydecay40 = getelementptr inbounds [16 x i8], [16 x i8]* %size, i64 0, i64 0, !dbg !2021
  %call41 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([34 x i8], [34 x i8]* @.str.5, i64 0, i64 0), i8* %arraydecay40), !dbg !2022
  br label %if.end44, !dbg !2023

if.else42:                                        ; preds = %land.lhs.true23, %if.then19
  %27 = load i32, i32* %n1.addr, align 4, !dbg !2024
  %call43 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.6, i64 0, i64 0), i32 %27), !dbg !2026
  br label %if.end44

if.end44:                                         ; preds = %if.else42, %if.end37
  br label %if.end47, !dbg !2027

if.else45:                                        ; preds = %land.lhs.true17, %if.else15
  %28 = load i32, i32* %n1.addr, align 4, !dbg !2028
  %29 = load i32, i32* %n2.addr, align 4, !dbg !2030
  %30 = load i32, i32* %n3.addr, align 4, !dbg !2031
  %call46 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([42 x i8], [42 x i8]* @.str.7, i64 0, i64 0), i32 %28, i32 %29, i32 %30), !dbg !2032
  br label %if.end47

if.end47:                                         ; preds = %if.else45, %if.end44
  br label %if.end48

if.end48:                                         ; preds = %if.end47, %if.end14
  %31 = load i32, i32* %niter.addr, align 4, !dbg !2033
  %call49 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.8, i64 0, i64 0), i32 %31), !dbg !2034
  %32 = load double, double* %t.addr, align 8, !dbg !2035
  %call50 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([39 x i8], [39 x i8]* @.str.9, i64 0, i64 0), double %32), !dbg !2036
  %33 = load double, double* %mops.addr, align 8, !dbg !2037
  %call51 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([39 x i8], [39 x i8]* @.str.10, i64 0, i64 0), double %33), !dbg !2038
  %34 = load i8*, i8** %optype.addr, align 8, !dbg !2039
  %call52 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([25 x i8], [25 x i8]* @.str.11, i64 0, i64 0), i8* %34), !dbg !2040
  %35 = load i32, i32* %passed_verification.addr, align 4, !dbg !2041
  %cmp53 = icmp slt i32 %35, 0, !dbg !2043
  br i1 %cmp53, label %if.then54, label %if.else56, !dbg !2044

if.then54:                                        ; preds = %if.end48
  %call55 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([45 x i8], [45 x i8]* @.str.12, i64 0, i64 0)), !dbg !2045
  br label %if.end62, !dbg !2047

if.else56:                                        ; preds = %if.end48
  %36 = load i32, i32* %passed_verification.addr, align 4, !dbg !2048
  %tobool = icmp ne i32 %36, 0, !dbg !2048
  br i1 %tobool, label %if.then57, label %if.else59, !dbg !2050

if.then57:                                        ; preds = %if.else56
  %call58 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([45 x i8], [45 x i8]* @.str.13, i64 0, i64 0)), !dbg !2051
  br label %if.end61, !dbg !2053

if.else59:                                        ; preds = %if.else56
  %call60 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([45 x i8], [45 x i8]* @.str.14, i64 0, i64 0)), !dbg !2054
  br label %if.end61

if.end61:                                         ; preds = %if.else59, %if.then57
  br label %if.end62

if.end62:                                         ; preds = %if.end61, %if.then54
  %37 = load i8*, i8** %npbversion.addr, align 8, !dbg !2056
  %call63 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.15, i64 0, i64 0), i8* %37), !dbg !2057
  %38 = load i8*, i8** %compiletime.addr, align 8, !dbg !2058
  %call64 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.16, i64 0, i64 0), i8* %38), !dbg !2059
  %39 = load i8*, i8** %compilerversion.addr, align 8, !dbg !2060
  %call65 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.17, i64 0, i64 0), i8* %39), !dbg !2061
  %40 = load i8*, i8** %libversion.addr, align 8, !dbg !2062
  %call66 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.18, i64 0, i64 0), i8* %40), !dbg !2063
  %call67 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([20 x i8], [20 x i8]* @.str.19, i64 0, i64 0)), !dbg !2064
  %41 = load i8*, i8** %cc.addr, align 8, !dbg !2065
  %call68 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.20, i64 0, i64 0), i8* %41), !dbg !2066
  %42 = load i8*, i8** %clink.addr, align 8, !dbg !2067
  %call69 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.21, i64 0, i64 0), i8* %42), !dbg !2068
  %43 = load i8*, i8** %c_lib.addr, align 8, !dbg !2069
  %call70 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.22, i64 0, i64 0), i8* %43), !dbg !2070
  %44 = load i8*, i8** %c_inc.addr, align 8, !dbg !2071
  %call71 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.23, i64 0, i64 0), i8* %44), !dbg !2072
  %45 = load i8*, i8** %cflags.addr, align 8, !dbg !2073
  %call72 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.24, i64 0, i64 0), i8* %45), !dbg !2074
  %46 = load i8*, i8** %clinkflags.addr, align 8, !dbg !2075
  %call73 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.25, i64 0, i64 0), i8* %46), !dbg !2076
  %47 = load i8*, i8** %rand.addr, align 8, !dbg !2077
  %call74 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.26, i64 0, i64 0), i8* %47), !dbg !2078
  %call75 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.27, i64 0, i64 0)), !dbg !2079
  %48 = load i8*, i8** %cpu_device.addr, align 8, !dbg !2080
  %call76 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.28, i64 0, i64 0), i8* %48), !dbg !2081
  %49 = load i8*, i8** %gpu_device.addr, align 8, !dbg !2082
  %call77 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.29, i64 0, i64 0), i8* %49), !dbg !2083
  %call78 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.30, i64 0, i64 0)), !dbg !2084
  %50 = load i8*, i8** %gpu_config.addr, align 8, !dbg !2085
  %call79 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.31, i64 0, i64 0), i8* %50), !dbg !2086
  %call80 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.32, i64 0, i64 0)), !dbg !2087
  %call81 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([72 x i8], [72 x i8]* @.str.33, i64 0, i64 0)), !dbg !2088
  %call82 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([27 x i8], [27 x i8]* @.str.34, i64 0, i64 0)), !dbg !2089
  %call83 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([56 x i8], [56 x i8]* @.str.35, i64 0, i64 0)), !dbg !2090
  %call84 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([52 x i8], [52 x i8]* @.str.36, i64 0, i64 0)), !dbg !2091
  %call85 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([46 x i8], [46 x i8]* @.str.37, i64 0, i64 0)), !dbg !2092
  %call86 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.32, i64 0, i64 0)), !dbg !2093
  %call87 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([43 x i8], [43 x i8]* @.str.38, i64 0, i64 0)), !dbg !2094
  %call88 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([72 x i8], [72 x i8]* @.str.33, i64 0, i64 0)), !dbg !2095
  %call89 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.32, i64 0, i64 0)), !dbg !2096
  ret void, !dbg !2097
}

declare dso_local i32 @printf(i8*, ...) #6

; Function Attrs: nounwind
declare dso_local double @pow(double, double) #7

; Function Attrs: nounwind
declare dso_local i32 @sprintf(i8*, i8*, ...) #7

; Function Attrs: noinline norecurse uwtable
define dso_local i32 @main(i32 %argc, i8** %argv) #8 !dbg !2098 {
entry:
  %retval = alloca i32, align 4
  %argc.addr = alloca i32, align 4
  %argv.addr = alloca i8**, align 8
  %Mops = alloca double, align 8
  %t1 = alloca double, align 8
  %sx = alloca double, align 8
  %sy = alloca double, align 8
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
  %agg.tmp18 = alloca %struct.dim3, align 4
  %agg.tmp.coerce = alloca { i64, i32 }, align 4
  %agg.tmp18.coerce = alloca { i64, i32 }, align 4
  %gpu_config = alloca [256 x i8], align 16
  %gpu_config_string = alloca [2048 x i8], align 16
  store i32 0, i32* %retval, align 4
  store i32 %argc, i32* %argc.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %argc.addr, metadata !2101, metadata !DIExpression()), !dbg !2102
  store i8** %argv, i8*** %argv.addr, align 8
  call void @llvm.dbg.declare(metadata i8*** %argv.addr, metadata !2103, metadata !DIExpression()), !dbg !2104
  %call = call noalias i8* @malloc(i64 80) #10, !dbg !2105
  %0 = bitcast i8* %call to double*, !dbg !2106
  store double* %0, double** @_ZL1q, align 8, !dbg !2107
  call void @llvm.dbg.declare(metadata double* %Mops, metadata !2108, metadata !DIExpression()), !dbg !2109
  call void @llvm.dbg.declare(metadata double* %t1, metadata !2110, metadata !DIExpression()), !dbg !2111
  call void @llvm.dbg.declare(metadata double* %sx, metadata !2112, metadata !DIExpression()), !dbg !2113
  call void @llvm.dbg.declare(metadata double* %sy, metadata !2114, metadata !DIExpression()), !dbg !2115
  call void @llvm.dbg.declare(metadata double* %an, metadata !2116, metadata !DIExpression()), !dbg !2117
  call void @llvm.dbg.declare(metadata double* %gc, metadata !2118, metadata !DIExpression()), !dbg !2119
  call void @llvm.dbg.declare(metadata double* %sx_verify_value, metadata !2120, metadata !DIExpression()), !dbg !2121
  call void @llvm.dbg.declare(metadata double* %sy_verify_value, metadata !2122, metadata !DIExpression()), !dbg !2123
  call void @llvm.dbg.declare(metadata double* %sx_err, metadata !2124, metadata !DIExpression()), !dbg !2125
  call void @llvm.dbg.declare(metadata double* %sy_err, metadata !2126, metadata !DIExpression()), !dbg !2127
  call void @llvm.dbg.declare(metadata i32* %i, metadata !2128, metadata !DIExpression()), !dbg !2129
  call void @llvm.dbg.declare(metadata i32* %j, metadata !2130, metadata !DIExpression()), !dbg !2131
  call void @llvm.dbg.declare(metadata i32* %nit, metadata !2132, metadata !DIExpression()), !dbg !2133
  call void @llvm.dbg.declare(metadata i32* %block, metadata !2134, metadata !DIExpression()), !dbg !2135
  call void @llvm.dbg.declare(metadata i32* %verified, metadata !2136, metadata !DIExpression()), !dbg !2138
  call void @llvm.dbg.declare(metadata [16 x i8]* %size, metadata !2139, metadata !DIExpression()), !dbg !2140
  %arraydecay = getelementptr inbounds [16 x i8], [16 x i8]* %size, i64 0, i64 0, !dbg !2141
  %call1 = call double @pow(double 2.000000e+00, double 2.900000e+01) #10, !dbg !2142
  %call2 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay, i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.39, i64 0, i64 0), double %call1) #10, !dbg !2143
  store i32 14, i32* %j, align 4, !dbg !2144
  %1 = load i32, i32* %j, align 4, !dbg !2145
  %idxprom = sext i32 %1 to i64, !dbg !2147
  %arrayidx = getelementptr inbounds [16 x i8], [16 x i8]* %size, i64 0, i64 %idxprom, !dbg !2147
  %2 = load i8, i8* %arrayidx, align 1, !dbg !2147
  %conv = sext i8 %2 to i32, !dbg !2147
  %cmp = icmp eq i32 %conv, 46, !dbg !2148
  br i1 %cmp, label %if.then, label %if.end, !dbg !2149

if.then:                                          ; preds = %entry
  %3 = load i32, i32* %j, align 4, !dbg !2150
  %dec = add nsw i32 %3, -1, !dbg !2150
  store i32 %dec, i32* %j, align 4, !dbg !2150
  br label %if.end, !dbg !2152

if.end:                                           ; preds = %if.then, %entry
  %4 = load i32, i32* %j, align 4, !dbg !2153
  %add = add nsw i32 %4, 1, !dbg !2154
  %idxprom3 = sext i32 %add to i64, !dbg !2155
  %arrayidx4 = getelementptr inbounds [16 x i8], [16 x i8]* %size, i64 0, i64 %idxprom3, !dbg !2155
  store i8 0, i8* %arrayidx4, align 1, !dbg !2156
  %call5 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([65 x i8], [65 x i8]* @.str.40, i64 0, i64 0)), !dbg !2157
  %arraydecay6 = getelementptr inbounds [16 x i8], [16 x i8]* %size, i64 0, i64 0, !dbg !2158
  %call7 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([43 x i8], [43 x i8]* @.str.41, i64 0, i64 0), i8* %arraydecay6), !dbg !2159
  store i32 0, i32* %verified, align 4, !dbg !2160
  store double 0x41D2309CE5400000, double* %t1, align 8, !dbg !2161
  store i32 0, i32* %i, align 4, !dbg !2162
  br label %for.cond, !dbg !2164

for.cond:                                         ; preds = %for.inc, %if.end
  %5 = load i32, i32* %i, align 4, !dbg !2165
  %cmp8 = icmp slt i32 %5, 17, !dbg !2167
  br i1 %cmp8, label %for.body, label %for.end, !dbg !2168

for.body:                                         ; preds = %for.cond
  %6 = load double, double* %t1, align 8, !dbg !2169
  %call9 = call double @_Z6randlcPdd(double* %t1, double %6), !dbg !2171
  br label %for.inc, !dbg !2172

for.inc:                                          ; preds = %for.body
  %7 = load i32, i32* %i, align 4, !dbg !2173
  %inc = add nsw i32 %7, 1, !dbg !2173
  store i32 %inc, i32* %i, align 4, !dbg !2173
  br label %for.cond, !dbg !2174, !llvm.loop !2175

for.end:                                          ; preds = %for.cond
  %8 = load double, double* %t1, align 8, !dbg !2177
  store double %8, double* %an, align 8, !dbg !2178
  store double 0.000000e+00, double* %gc, align 8, !dbg !2179
  store double 0.000000e+00, double* %sx, align 8, !dbg !2180
  store double 0.000000e+00, double* %sy, align 8, !dbg !2181
  store i32 0, i32* %i, align 4, !dbg !2182
  br label %for.cond10, !dbg !2184

for.cond10:                                       ; preds = %for.inc15, %for.end
  %9 = load i32, i32* %i, align 4, !dbg !2185
  %cmp11 = icmp slt i32 %9, 10, !dbg !2187
  br i1 %cmp11, label %for.body12, label %for.end17, !dbg !2188

for.body12:                                       ; preds = %for.cond10
  %10 = load double*, double** @_ZL1q, align 8, !dbg !2189
  %11 = load i32, i32* %i, align 4, !dbg !2191
  %idxprom13 = sext i32 %11 to i64, !dbg !2189
  %arrayidx14 = getelementptr inbounds double, double* %10, i64 %idxprom13, !dbg !2189
  store double 0.000000e+00, double* %arrayidx14, align 8, !dbg !2192
  br label %for.inc15, !dbg !2193

for.inc15:                                        ; preds = %for.body12
  %12 = load i32, i32* %i, align 4, !dbg !2194
  %inc16 = add nsw i32 %12, 1, !dbg !2194
  store i32 %inc16, i32* %i, align 4, !dbg !2194
  br label %for.cond10, !dbg !2195, !llvm.loop !2196

for.end17:                                        ; preds = %for.cond10
  call void @_ZL9setup_gpuv(), !dbg !2198
  %13 = load i32, i32* @blocks_per_grid, align 4, !dbg !2199
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp, i32 %13, i32 1, i32 1), !dbg !2199
  %14 = load i32, i32* @threads_per_block, align 4, !dbg !2200
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp18, i32 %14, i32 1, i32 1), !dbg !2200
  %15 = bitcast { i64, i32 }* %agg.tmp.coerce to i8*, !dbg !2201
  %16 = bitcast %struct.dim3* %agg.tmp to i8*, !dbg !2201
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %15, i8* align 4 %16, i64 12, i1 false), !dbg !2201
  %17 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 0, !dbg !2201
  %18 = load i64, i64* %17, align 4, !dbg !2201
  %19 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 1, !dbg !2201
  %20 = load i32, i32* %19, align 4, !dbg !2201
  %21 = bitcast { i64, i32 }* %agg.tmp18.coerce to i8*, !dbg !2201
  %22 = bitcast %struct.dim3* %agg.tmp18 to i8*, !dbg !2201
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %21, i8* align 4 %22, i64 12, i1 false), !dbg !2201
  %23 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp18.coerce, i32 0, i32 0, !dbg !2201
  %24 = load i64, i64* %23, align 4, !dbg !2201
  %25 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp18.coerce, i32 0, i32 1, !dbg !2201
  %26 = load i32, i32* %25, align 4, !dbg !2201
  %call19 = call i32 @cudaConfigureCall(i64 %18, i32 %20, i64 %24, i32 %26, i64 0, %struct.CUstream_st* null), !dbg !2201
  %tobool = icmp ne i32 %call19, 0, !dbg !2201
  br i1 %tobool, label %kcall.end, label %kcall.configok, !dbg !2202

kcall.configok:                                   ; preds = %for.end17
  %27 = load double*, double** @q_device, align 8, !dbg !2203
  %28 = load double*, double** @sx_device, align 8, !dbg !2204
  %29 = load double*, double** @sy_device, align 8, !dbg !2205
  %30 = load double, double* %an, align 8, !dbg !2206
  call void @ep.ll_CudaFE__Z10gpu_kernelPdS_S_d(double* %27, double* %28, double* %29, double %30), !dbg !2202
  br label %kcall.end, !dbg !2202

kcall.end:                                        ; preds = %kcall.configok, %for.end17
  %31 = load double*, double** @q_host, align 8, !dbg !2207
  %32 = bitcast double* %31 to i8*, !dbg !2207
  %33 = load double*, double** @q_device, align 8, !dbg !2208
  %34 = bitcast double* %33 to i8*, !dbg !2208
  %35 = load i64, i64* @size_q, align 8, !dbg !2209
  %call20 = call i32 @cudaMemcpy(i8* %32, i8* %34, i64 %35, i32 2), !dbg !2210
  %36 = load double*, double** @sx_host, align 8, !dbg !2211
  %37 = bitcast double* %36 to i8*, !dbg !2211
  %38 = load double*, double** @sx_device, align 8, !dbg !2212
  %39 = bitcast double* %38 to i8*, !dbg !2212
  %40 = load i64, i64* @size_sx, align 8, !dbg !2213
  %call21 = call i32 @cudaMemcpy(i8* %37, i8* %39, i64 %40, i32 2), !dbg !2214
  %41 = load double*, double** @sy_host, align 8, !dbg !2215
  %42 = bitcast double* %41 to i8*, !dbg !2215
  %43 = load double*, double** @sy_device, align 8, !dbg !2216
  %44 = bitcast double* %43 to i8*, !dbg !2216
  %45 = load i64, i64* @size_sy, align 8, !dbg !2217
  %call22 = call i32 @cudaMemcpy(i8* %42, i8* %44, i64 %45, i32 2), !dbg !2218
  store i32 0, i32* %block, align 4, !dbg !2219
  br label %for.cond23, !dbg !2221

for.cond23:                                       ; preds = %for.inc44, %kcall.end
  %46 = load i32, i32* %block, align 4, !dbg !2222
  %47 = load i32, i32* @blocks_per_grid, align 4, !dbg !2224
  %cmp24 = icmp slt i32 %46, %47, !dbg !2225
  br i1 %cmp24, label %for.body25, label %for.end46, !dbg !2226

for.body25:                                       ; preds = %for.cond23
  store i32 0, i32* %i, align 4, !dbg !2227
  br label %for.cond26, !dbg !2230

for.cond26:                                       ; preds = %for.inc35, %for.body25
  %48 = load i32, i32* %i, align 4, !dbg !2231
  %cmp27 = icmp slt i32 %48, 10, !dbg !2233
  br i1 %cmp27, label %for.body28, label %for.end37, !dbg !2234

for.body28:                                       ; preds = %for.cond26
  %49 = load double*, double** @q_host, align 8, !dbg !2235
  %50 = load i32, i32* %block, align 4, !dbg !2237
  %mul = mul nsw i32 %50, 10, !dbg !2238
  %51 = load i32, i32* %i, align 4, !dbg !2239
  %add29 = add nsw i32 %mul, %51, !dbg !2240
  %idxprom30 = sext i32 %add29 to i64, !dbg !2235
  %arrayidx31 = getelementptr inbounds double, double* %49, i64 %idxprom30, !dbg !2235
  %52 = load double, double* %arrayidx31, align 8, !dbg !2235
  %53 = load double*, double** @_ZL1q, align 8, !dbg !2241
  %54 = load i32, i32* %i, align 4, !dbg !2242
  %idxprom32 = sext i32 %54 to i64, !dbg !2241
  %arrayidx33 = getelementptr inbounds double, double* %53, i64 %idxprom32, !dbg !2241
  %55 = load double, double* %arrayidx33, align 8, !dbg !2243
  %add34 = fadd contract double %55, %52, !dbg !2243
  store double %add34, double* %arrayidx33, align 8, !dbg !2243
  br label %for.inc35, !dbg !2244

for.inc35:                                        ; preds = %for.body28
  %56 = load i32, i32* %i, align 4, !dbg !2245
  %inc36 = add nsw i32 %56, 1, !dbg !2245
  store i32 %inc36, i32* %i, align 4, !dbg !2245
  br label %for.cond26, !dbg !2246, !llvm.loop !2247

for.end37:                                        ; preds = %for.cond26
  %57 = load double*, double** @sx_host, align 8, !dbg !2249
  %58 = load i32, i32* %block, align 4, !dbg !2250
  %idxprom38 = sext i32 %58 to i64, !dbg !2249
  %arrayidx39 = getelementptr inbounds double, double* %57, i64 %idxprom38, !dbg !2249
  %59 = load double, double* %arrayidx39, align 8, !dbg !2249
  %60 = load double, double* %sx, align 8, !dbg !2251
  %add40 = fadd contract double %60, %59, !dbg !2251
  store double %add40, double* %sx, align 8, !dbg !2251
  %61 = load double*, double** @sy_host, align 8, !dbg !2252
  %62 = load i32, i32* %block, align 4, !dbg !2253
  %idxprom41 = sext i32 %62 to i64, !dbg !2252
  %arrayidx42 = getelementptr inbounds double, double* %61, i64 %idxprom41, !dbg !2252
  %63 = load double, double* %arrayidx42, align 8, !dbg !2252
  %64 = load double, double* %sy, align 8, !dbg !2254
  %add43 = fadd contract double %64, %63, !dbg !2254
  store double %add43, double* %sy, align 8, !dbg !2254
  br label %for.inc44, !dbg !2255

for.inc44:                                        ; preds = %for.end37
  %65 = load i32, i32* %block, align 4, !dbg !2256
  %inc45 = add nsw i32 %65, 1, !dbg !2256
  store i32 %inc45, i32* %block, align 4, !dbg !2256
  br label %for.cond23, !dbg !2257, !llvm.loop !2258

for.end46:                                        ; preds = %for.cond23
  store i32 0, i32* %i, align 4, !dbg !2260
  br label %for.cond47, !dbg !2262

for.cond47:                                       ; preds = %for.inc53, %for.end46
  %66 = load i32, i32* %i, align 4, !dbg !2263
  %cmp48 = icmp slt i32 %66, 10, !dbg !2265
  br i1 %cmp48, label %for.body49, label %for.end55, !dbg !2266

for.body49:                                       ; preds = %for.cond47
  %67 = load double*, double** @_ZL1q, align 8, !dbg !2267
  %68 = load i32, i32* %i, align 4, !dbg !2269
  %idxprom50 = sext i32 %68 to i64, !dbg !2267
  %arrayidx51 = getelementptr inbounds double, double* %67, i64 %idxprom50, !dbg !2267
  %69 = load double, double* %arrayidx51, align 8, !dbg !2267
  %70 = load double, double* %gc, align 8, !dbg !2270
  %add52 = fadd contract double %70, %69, !dbg !2270
  store double %add52, double* %gc, align 8, !dbg !2270
  br label %for.inc53, !dbg !2271

for.inc53:                                        ; preds = %for.body49
  %71 = load i32, i32* %i, align 4, !dbg !2272
  %inc54 = add nsw i32 %71, 1, !dbg !2272
  store i32 %inc54, i32* %i, align 4, !dbg !2272
  br label %for.cond47, !dbg !2273, !llvm.loop !2274

for.end55:                                        ; preds = %for.cond47
  store i32 0, i32* %nit, align 4, !dbg !2276
  store i32 1, i32* %verified, align 4, !dbg !2277
  store double 0xC0B0C7E00ADACEF8, double* %sx_verify_value, align 8, !dbg !2278
  store double 0xC0CEDFA9B1BE31DC, double* %sy_verify_value, align 8, !dbg !2283
  %72 = load i32, i32* %verified, align 4, !dbg !2284
  %tobool56 = icmp ne i32 %72, 0, !dbg !2284
  br i1 %tobool56, label %if.then57, label %if.end63, !dbg !2286

if.then57:                                        ; preds = %for.end55
  %73 = load double, double* %sx, align 8, !dbg !2287
  %74 = load double, double* %sx_verify_value, align 8, !dbg !2289
  %sub = fsub contract double %73, %74, !dbg !2290
  %75 = load double, double* %sx_verify_value, align 8, !dbg !2291
  %div = fdiv double %sub, %75, !dbg !2292
  %76 = call double @llvm.fabs.f64(double %div), !dbg !2293
  store double %76, double* %sx_err, align 8, !dbg !2294
  %77 = load double, double* %sy, align 8, !dbg !2295
  %78 = load double, double* %sy_verify_value, align 8, !dbg !2296
  %sub58 = fsub contract double %77, %78, !dbg !2297
  %79 = load double, double* %sy_verify_value, align 8, !dbg !2298
  %div59 = fdiv double %sub58, %79, !dbg !2299
  %80 = call double @llvm.fabs.f64(double %div59), !dbg !2300
  store double %80, double* %sy_err, align 8, !dbg !2301
  %81 = load double, double* %sx_err, align 8, !dbg !2302
  %cmp60 = fcmp ole double %81, 1.000000e-08, !dbg !2303
  br i1 %cmp60, label %land.rhs, label %land.end, !dbg !2304

land.rhs:                                         ; preds = %if.then57
  %82 = load double, double* %sy_err, align 8, !dbg !2305
  %cmp61 = fcmp ole double %82, 1.000000e-08, !dbg !2306
  br label %land.end

land.end:                                         ; preds = %land.rhs, %if.then57
  %83 = phi i1 [ false, %if.then57 ], [ %cmp61, %land.rhs ], !dbg !2307
  %conv62 = zext i1 %83 to i32, !dbg !2308
  store i32 %conv62, i32* %verified, align 4, !dbg !2309
  br label %if.end63, !dbg !2310

if.end63:                                         ; preds = %land.end, %for.end55
  %call64 = call double @pow(double 2.000000e+00, double 2.900000e+01) #10, !dbg !2311
  %div65 = fdiv double %call64, 1.000000e+06, !dbg !2312
  store double %div65, double* %Mops, align 8, !dbg !2313
  %call66 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([26 x i8], [26 x i8]* @.str.42, i64 0, i64 0)), !dbg !2314
  %call67 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([19 x i8], [19 x i8]* @.str.43, i64 0, i64 0)), !dbg !2315
  %call68 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([12 x i8], [12 x i8]* @.str.44, i64 0, i64 0), i32 28), !dbg !2316
  %84 = load double, double* %gc, align 8, !dbg !2317
  %call69 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([30 x i8], [30 x i8]* @.str.45, i64 0, i64 0), double %84), !dbg !2318
  %85 = load double, double* %sx, align 8, !dbg !2319
  %86 = load double, double* %sy, align 8, !dbg !2320
  %call70 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([25 x i8], [25 x i8]* @.str.46, i64 0, i64 0), double %85, double %86), !dbg !2321
  %call71 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.47, i64 0, i64 0)), !dbg !2322
  store i32 0, i32* %i, align 4, !dbg !2323
  br label %for.cond72, !dbg !2325

for.cond72:                                       ; preds = %for.inc78, %if.end63
  %87 = load i32, i32* %i, align 4, !dbg !2326
  %cmp73 = icmp slt i32 %87, 10, !dbg !2328
  br i1 %cmp73, label %for.body74, label %for.end80, !dbg !2329

for.body74:                                       ; preds = %for.cond72
  %88 = load i32, i32* %i, align 4, !dbg !2330
  %89 = load double*, double** @_ZL1q, align 8, !dbg !2332
  %90 = load i32, i32* %i, align 4, !dbg !2333
  %idxprom75 = sext i32 %90 to i64, !dbg !2332
  %arrayidx76 = getelementptr inbounds double, double* %89, i64 %idxprom75, !dbg !2332
  %91 = load double, double* %arrayidx76, align 8, !dbg !2332
  %call77 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.48, i64 0, i64 0), i32 %88, double %91), !dbg !2334
  br label %for.inc78, !dbg !2335

for.inc78:                                        ; preds = %for.body74
  %92 = load i32, i32* %i, align 4, !dbg !2336
  %inc79 = add nsw i32 %92, 1, !dbg !2336
  store i32 %inc79, i32* %i, align 4, !dbg !2336
  br label %for.cond72, !dbg !2337, !llvm.loop !2338

for.end80:                                        ; preds = %for.cond72
  call void @llvm.dbg.declare(metadata [256 x i8]* %gpu_config, metadata !2340, metadata !DIExpression()), !dbg !2341
  call void @llvm.dbg.declare(metadata [2048 x i8]* %gpu_config_string, metadata !2342, metadata !DIExpression()), !dbg !2346
  %arraydecay81 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !2347
  %call82 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay81, i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.49, i64 0, i64 0), i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.50, i64 0, i64 0), i8* getelementptr inbounds ([18 x i8], [18 x i8]* @.str.51, i64 0, i64 0)) #10, !dbg !2348
  %arraydecay83 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !2349
  %arraydecay84 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !2350
  %call85 = call i8* @strcpy(i8* %arraydecay83, i8* %arraydecay84) #10, !dbg !2351
  %arraydecay86 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !2352
  %93 = load i32, i32* @threads_per_block, align 4, !dbg !2353
  %call87 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay86, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.52, i64 0, i64 0), i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.53, i64 0, i64 0), i32 %93) #10, !dbg !2354
  %arraydecay88 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !2355
  %arraydecay89 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !2356
  %call90 = call i8* @strcat(i8* %arraydecay88, i8* %arraydecay89) #10, !dbg !2357
  %94 = load i32, i32* %nit, align 4, !dbg !2358
  %95 = load double, double* %Mops, align 8, !dbg !2359
  %96 = load i32, i32* %verified, align 4, !dbg !2360
  %arraydecay91 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !2361
  call void @_Z15c_print_resultsPcciiiiddS_iS_S_S_S_S_S_S_S_S_S_S_S_S_S_(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.54, i64 0, i64 0), i8 signext 65, i32 29, i32 0, i32 0, i32 %94, double 0.000000e+00, double %95, i8* getelementptr inbounds ([25 x i8], [25 x i8]* @.str.55, i64 0, i64 0), i32 %96, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.56, i64 0, i64 0), i8* getelementptr inbounds ([12 x i8], [12 x i8]* @.str.57, i64 0, i64 0), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.58, i64 0, i64 0), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.58, i64 0, i64 0), i8* getelementptr inbounds ([42 x i8], [42 x i8]* @.str.59, i64 0, i64 0), i8* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 0, i64 0), i8* %arraydecay91, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.60, i64 0, i64 0), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.61, i64 0, i64 0), i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.62, i64 0, i64 0), i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.63, i64 0, i64 0), i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.64, i64 0, i64 0), i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.64, i64 0, i64 0), i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.65, i64 0, i64 0)), !dbg !2362
  call void @_ZL11release_gpuv(), !dbg !2363
  ret i32 0, !dbg !2364
}

; Function Attrs: nounwind
declare dso_local noalias i8* @malloc(i64) #7

; Function Attrs: noinline uwtable
define internal void @_ZL9setup_gpuv() #5 !dbg !2365 {
entry:
  store i32 32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !2366
  store i32 32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !2367
  %0 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !2368
  %cmp = icmp sle i32 32, %0, !dbg !2370
  br i1 %cmp, label %if.then, label %if.else, !dbg !2371

if.then:                                          ; preds = %entry
  store i32 32, i32* @threads_per_block, align 4, !dbg !2372
  br label %if.end, !dbg !2374

if.else:                                          ; preds = %entry
  %1 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !2375
  store i32 %1, i32* @threads_per_block, align 4, !dbg !2377
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %2 = load i32, i32* @threads_per_block, align 4, !dbg !2378
  %conv = sitofp i32 %2 to double, !dbg !2378
  %div = fdiv double 4.096000e+03, %conv, !dbg !2379
  %3 = call double @llvm.ceil.f64(double %div), !dbg !2380
  %conv1 = fptosi double %3 to i32, !dbg !2381
  store i32 %conv1, i32* @blocks_per_grid, align 4, !dbg !2382
  %4 = load i32, i32* @blocks_per_grid, align 4, !dbg !2383
  %mul = mul nsw i32 %4, 10, !dbg !2384
  %conv2 = sext i32 %mul to i64, !dbg !2383
  %mul3 = mul i64 %conv2, 8, !dbg !2385
  store i64 %mul3, i64* @size_q, align 8, !dbg !2386
  %5 = load i32, i32* @blocks_per_grid, align 4, !dbg !2387
  %conv4 = sext i32 %5 to i64, !dbg !2387
  %mul5 = mul i64 %conv4, 8, !dbg !2388
  store i64 %mul5, i64* @size_sx, align 8, !dbg !2389
  %6 = load i32, i32* @blocks_per_grid, align 4, !dbg !2390
  %conv6 = sext i32 %6 to i64, !dbg !2390
  %mul7 = mul i64 %conv6, 8, !dbg !2391
  store i64 %mul7, i64* @size_sy, align 8, !dbg !2392
  %7 = load i64, i64* @size_q, align 8, !dbg !2393
  %call = call noalias i8* @malloc(i64 %7) #10, !dbg !2394
  %8 = bitcast i8* %call to double*, !dbg !2395
  store double* %8, double** @q_host, align 8, !dbg !2396
  %9 = load i64, i64* @size_sx, align 8, !dbg !2397
  %call8 = call noalias i8* @malloc(i64 %9) #10, !dbg !2398
  %10 = bitcast i8* %call8 to double*, !dbg !2399
  store double* %10, double** @sx_host, align 8, !dbg !2400
  %11 = load i64, i64* @size_sy, align 8, !dbg !2401
  %call9 = call noalias i8* @malloc(i64 %11) #10, !dbg !2402
  %12 = bitcast i8* %call9 to double*, !dbg !2403
  store double* %12, double** @sy_host, align 8, !dbg !2404
  %13 = load i64, i64* @size_q, align 8, !dbg !2405
  %call10 = call i32 @_ZL10cudaMallocIdE9cudaErrorPPT_m(double** @q_device, i64 %13), !dbg !2406
  %14 = load i64, i64* @size_sx, align 8, !dbg !2407
  %call11 = call i32 @_ZL10cudaMallocIdE9cudaErrorPPT_m(double** @sx_device, i64 %14), !dbg !2408
  %15 = load i64, i64* @size_sy, align 8, !dbg !2409
  %call12 = call i32 @_ZL10cudaMallocIdE9cudaErrorPPT_m(double** @sy_device, i64 %15), !dbg !2410
  ret void, !dbg !2411
}

; Function Attrs: noinline nounwind uwtable
define linkonce_odr dso_local void @_ZN4dim3C2Ejjj(%struct.dim3* %this, i32 %vx, i32 %vy, i32 %vz) unnamed_addr #4 comdat align 2 !dbg !2412 {
entry:
  %this.addr = alloca %struct.dim3*, align 8
  %vx.addr = alloca i32, align 4
  %vy.addr = alloca i32, align 4
  %vz.addr = alloca i32, align 4
  store %struct.dim3* %this, %struct.dim3** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dim3** %this.addr, metadata !2413, metadata !DIExpression()), !dbg !2415
  store i32 %vx, i32* %vx.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %vx.addr, metadata !2416, metadata !DIExpression()), !dbg !2417
  store i32 %vy, i32* %vy.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %vy.addr, metadata !2418, metadata !DIExpression()), !dbg !2419
  store i32 %vz, i32* %vz.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %vz.addr, metadata !2420, metadata !DIExpression()), !dbg !2421
  %this1 = load %struct.dim3*, %struct.dim3** %this.addr, align 8
  %x = getelementptr inbounds %struct.dim3, %struct.dim3* %this1, i32 0, i32 0, !dbg !2422
  %0 = load i32, i32* %vx.addr, align 4, !dbg !2423
  store i32 %0, i32* %x, align 4, !dbg !2422
  %y = getelementptr inbounds %struct.dim3, %struct.dim3* %this1, i32 0, i32 1, !dbg !2424
  %1 = load i32, i32* %vy.addr, align 4, !dbg !2425
  store i32 %1, i32* %y, align 4, !dbg !2424
  %z = getelementptr inbounds %struct.dim3, %struct.dim3* %this1, i32 0, i32 2, !dbg !2426
  %2 = load i32, i32* %vz.addr, align 4, !dbg !2427
  store i32 %2, i32* %z, align 4, !dbg !2426
  ret void, !dbg !2428
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1 immarg) #9

declare dso_local i32 @cudaConfigureCall(i64, i32, i64, i32, i64, %struct.CUstream_st*) #6

; Function Attrs: noinline uwtable
define dso_local void @ep.ll_CudaFE__Z10gpu_kernelPdS_S_d(double* %q_global, double* %sx_global, double* %sy_global, double %an) #5 !dbg !2429 {
entry:
  %q_global.addr = alloca double*, align 8
  %sx_global.addr = alloca double*, align 8
  %sy_global.addr = alloca double*, align 8
  %an.addr = alloca double, align 8
  store double* %q_global, double** %q_global.addr, align 8
  call void @llvm.dbg.declare(metadata double** %q_global.addr, metadata !2430, metadata !DIExpression()), !dbg !2431
  store double* %sx_global, double** %sx_global.addr, align 8
  call void @llvm.dbg.declare(metadata double** %sx_global.addr, metadata !2432, metadata !DIExpression()), !dbg !2433
  store double* %sy_global, double** %sy_global.addr, align 8
  call void @llvm.dbg.declare(metadata double** %sy_global.addr, metadata !2434, metadata !DIExpression()), !dbg !2435
  store double %an, double* %an.addr, align 8
  call void @llvm.dbg.declare(metadata double* %an.addr, metadata !2436, metadata !DIExpression()), !dbg !2437
  %0 = bitcast double** %q_global.addr to i8*, !dbg !2438
  %1 = call i32 @cudaSetupArgument(i8* %0, i64 8, i64 0), !dbg !2438
  %2 = icmp eq i32 %1, 0, !dbg !2438
  br i1 %2, label %setup.next, label %setup.end, !dbg !2438

setup.next:                                       ; preds = %entry
  %3 = bitcast double** %sx_global.addr to i8*, !dbg !2438
  %4 = call i32 @cudaSetupArgument(i8* %3, i64 8, i64 8), !dbg !2438
  %5 = icmp eq i32 %4, 0, !dbg !2438
  br i1 %5, label %setup.next1, label %setup.end, !dbg !2438

setup.next1:                                      ; preds = %setup.next
  %6 = bitcast double** %sy_global.addr to i8*, !dbg !2438
  %7 = call i32 @cudaSetupArgument(i8* %6, i64 8, i64 16), !dbg !2438
  %8 = icmp eq i32 %7, 0, !dbg !2438
  br i1 %8, label %setup.next2, label %setup.end, !dbg !2438

setup.next2:                                      ; preds = %setup.next1
  %9 = bitcast double* %an.addr to i8*, !dbg !2438
  %10 = call i32 @cudaSetupArgument(i8* %9, i64 8, i64 24), !dbg !2438
  %11 = icmp eq i32 %10, 0, !dbg !2438
  br i1 %11, label %setup.next3, label %setup.end, !dbg !2438

setup.next3:                                      ; preds = %setup.next2
  %12 = call i32 @cudaLaunch(i8* bitcast (void (double*, double*, double*, double)* @ep.ll_CudaFE__Z10gpu_kernelPdS_S_d to i8*)), !dbg !2438
  br label %setup.end, !dbg !2438

setup.end:                                        ; preds = %setup.next3, %setup.next2, %setup.next1, %setup.next, %entry
  ret void, !dbg !2439
}

declare dso_local i32 @cudaMemcpy(i8*, i8*, i64, i32) #6

; Function Attrs: nounwind readnone speculatable
declare double @llvm.fabs.f64(double) #1

; Function Attrs: nounwind
declare dso_local i8* @strcpy(i8*, i8*) #7

; Function Attrs: nounwind
declare dso_local i8* @strcat(i8*, i8*) #7

; Function Attrs: noinline uwtable
define internal void @_ZL11release_gpuv() #5 !dbg !2440 {
entry:
  %0 = load double*, double** @q_device, align 8, !dbg !2441
  %1 = bitcast double* %0 to i8*, !dbg !2441
  %call = call i32 @cudaFree(i8* %1), !dbg !2442
  %2 = load double*, double** @sx_device, align 8, !dbg !2443
  %3 = bitcast double* %2 to i8*, !dbg !2443
  %call1 = call i32 @cudaFree(i8* %3), !dbg !2444
  %4 = load double*, double** @sy_device, align 8, !dbg !2445
  %5 = bitcast double* %4 to i8*, !dbg !2445
  %call2 = call i32 @cudaFree(i8* %5), !dbg !2446
  ret void, !dbg !2447
}

declare dso_local i32 @cudaFree(i8*) #6

declare dso_local i32 @cudaSetupArgument(i8*, i64, i64)

declare dso_local i32 @cudaLaunch(i8*)

; Function Attrs: nounwind readnone speculatable
declare double @llvm.ceil.f64(double) #1

; Function Attrs: noinline uwtable
define internal i32 @_ZL10cudaMallocIdE9cudaErrorPPT_m(double** %devPtr, i64 %size) #5 !dbg !2448 {
entry:
  %devPtr.addr = alloca double**, align 8
  %size.addr = alloca i64, align 8
  store double** %devPtr, double*** %devPtr.addr, align 8
  call void @llvm.dbg.declare(metadata double*** %devPtr.addr, metadata !2456, metadata !DIExpression()), !dbg !2457
  store i64 %size, i64* %size.addr, align 8
  call void @llvm.dbg.declare(metadata i64* %size.addr, metadata !2458, metadata !DIExpression()), !dbg !2459
  %0 = load double**, double*** %devPtr.addr, align 8, !dbg !2460
  %1 = bitcast double** %0 to i8*, !dbg !2460
  %2 = bitcast i8* %1 to i8**, !dbg !2461
  %3 = load i64, i64* %size.addr, align 8, !dbg !2462
  %call = call i32 @cudaMalloc(i8** %2, i64 %3), !dbg !2463
  ret i32 %call, !dbg !2464
}

declare dso_local i32 @cudaMalloc(i8**, i64) #6

attributes #0 = { convergent noinline nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="sm_20" "target-features"="+ptx42,+sm_20" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { nounwind readnone }
attributes #3 = { convergent noinline nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="sm_20" "target-features"="+ptx42,+sm_20" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { noinline uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #7 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #8 = { noinline norecurse uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #9 = { argmemonly nounwind }
attributes #10 = { nounwind }
attributes #11 = { convergent nounwind }

!llvm.dbg.cu = !{!961, !2}
!nvvm.annotations = !{!1038, !1039, !1040, !1039, !1041, !1041, !1041, !1041, !1042, !1042, !1041}
!llvm.ident = !{!1043, !1043}
!nvvm.internalize.after.link = !{}
!nvvmir.version = !{!1044}
!llvm.module.flags = !{!1045, !1046, !1047, !1048, !1049}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "q_host", scope: !2, file: !3, line: 84, type: !98, isLocal: false, isDefinition: true)
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
!96 = !{!97, !98, !100, !99, !102, !103}
!97 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!98 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !99, size: 64)
!99 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
!100 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !101, size: 64)
!101 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!102 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !103, size: 64)
!103 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!104 = !{!0, !105, !107, !109, !111, !113, !115, !117, !119, !124, !126, !128, !130, !132, !207}
!105 = !DIGlobalVariableExpression(var: !106, expr: !DIExpression())
!106 = distinct !DIGlobalVariable(name: "q_device", scope: !2, file: !3, line: 85, type: !98, isLocal: false, isDefinition: true)
!107 = !DIGlobalVariableExpression(var: !108, expr: !DIExpression())
!108 = distinct !DIGlobalVariable(name: "sx_host", scope: !2, file: !3, line: 86, type: !98, isLocal: false, isDefinition: true)
!109 = !DIGlobalVariableExpression(var: !110, expr: !DIExpression())
!110 = distinct !DIGlobalVariable(name: "sx_device", scope: !2, file: !3, line: 87, type: !98, isLocal: false, isDefinition: true)
!111 = !DIGlobalVariableExpression(var: !112, expr: !DIExpression())
!112 = distinct !DIGlobalVariable(name: "sy_host", scope: !2, file: !3, line: 88, type: !98, isLocal: false, isDefinition: true)
!113 = !DIGlobalVariableExpression(var: !114, expr: !DIExpression())
!114 = distinct !DIGlobalVariable(name: "sy_device", scope: !2, file: !3, line: 89, type: !98, isLocal: false, isDefinition: true)
!115 = !DIGlobalVariableExpression(var: !116, expr: !DIExpression())
!116 = distinct !DIGlobalVariable(name: "threads_per_block", scope: !2, file: !3, line: 90, type: !97, isLocal: false, isDefinition: true)
!117 = !DIGlobalVariableExpression(var: !118, expr: !DIExpression())
!118 = distinct !DIGlobalVariable(name: "blocks_per_grid", scope: !2, file: !3, line: 91, type: !97, isLocal: false, isDefinition: true)
!119 = !DIGlobalVariableExpression(var: !120, expr: !DIExpression())
!120 = distinct !DIGlobalVariable(name: "size_q", scope: !2, file: !3, line: 92, type: !121, isLocal: false, isDefinition: true)
!121 = !DIDerivedType(tag: DW_TAG_typedef, name: "size_t", file: !122, line: 46, baseType: !123)
!122 = !DIFile(filename: "llvm-install-tulip/lib/clang/9.0.0/include/stddef.h", directory: "/scratch/ah7226")
!123 = !DIBasicType(name: "long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!124 = !DIGlobalVariableExpression(var: !125, expr: !DIExpression())
!125 = distinct !DIGlobalVariable(name: "size_sx", scope: !2, file: !3, line: 93, type: !121, isLocal: false, isDefinition: true)
!126 = !DIGlobalVariableExpression(var: !127, expr: !DIExpression())
!127 = distinct !DIGlobalVariable(name: "size_sy", scope: !2, file: !3, line: 94, type: !121, isLocal: false, isDefinition: true)
!128 = !DIGlobalVariableExpression(var: !129, expr: !DIExpression())
!129 = distinct !DIGlobalVariable(name: "gpu_device_id", scope: !2, file: !3, line: 95, type: !97, isLocal: false, isDefinition: true)
!130 = !DIGlobalVariableExpression(var: !131, expr: !DIExpression())
!131 = distinct !DIGlobalVariable(name: "total_devices", scope: !2, file: !3, line: 96, type: !97, isLocal: false, isDefinition: true)
!132 = !DIGlobalVariableExpression(var: !133, expr: !DIExpression())
!133 = distinct !DIGlobalVariable(name: "gpu_device_properties", scope: !2, file: !3, line: 97, type: !134, isLocal: false, isDefinition: true)
!134 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "cudaDeviceProp", file: !6, line: 1257, size: 5056, flags: DIFlagTypePassByValue, elements: !135, identifier: "_ZTS14cudaDeviceProp")
!135 = !{!136, !140, !141, !142, !143, !144, !145, !146, !150, !151, !152, !153, !154, !155, !156, !157, !158, !159, !160, !161, !162, !163, !164, !165, !166, !170, !171, !172, !173, !174, !175, !176, !177, !178, !179, !180, !181, !182, !183, !184, !185, !186, !187, !188, !189, !190, !191, !192, !193, !194, !195, !196, !197, !198, !199, !200, !201, !202, !203, !204, !205, !206}
!136 = !DIDerivedType(tag: DW_TAG_member, name: "name", scope: !134, file: !6, line: 1259, baseType: !137, size: 2048)
!137 = !DICompositeType(tag: DW_TAG_array_type, baseType: !101, size: 2048, elements: !138)
!138 = !{!139}
!139 = !DISubrange(count: 256)
!140 = !DIDerivedType(tag: DW_TAG_member, name: "totalGlobalMem", scope: !134, file: !6, line: 1260, baseType: !121, size: 64, offset: 2048)
!141 = !DIDerivedType(tag: DW_TAG_member, name: "sharedMemPerBlock", scope: !134, file: !6, line: 1261, baseType: !121, size: 64, offset: 2112)
!142 = !DIDerivedType(tag: DW_TAG_member, name: "regsPerBlock", scope: !134, file: !6, line: 1262, baseType: !97, size: 32, offset: 2176)
!143 = !DIDerivedType(tag: DW_TAG_member, name: "warpSize", scope: !134, file: !6, line: 1263, baseType: !97, size: 32, offset: 2208)
!144 = !DIDerivedType(tag: DW_TAG_member, name: "memPitch", scope: !134, file: !6, line: 1264, baseType: !121, size: 64, offset: 2240)
!145 = !DIDerivedType(tag: DW_TAG_member, name: "maxThreadsPerBlock", scope: !134, file: !6, line: 1265, baseType: !97, size: 32, offset: 2304)
!146 = !DIDerivedType(tag: DW_TAG_member, name: "maxThreadsDim", scope: !134, file: !6, line: 1266, baseType: !147, size: 96, offset: 2336)
!147 = !DICompositeType(tag: DW_TAG_array_type, baseType: !97, size: 96, elements: !148)
!148 = !{!149}
!149 = !DISubrange(count: 3)
!150 = !DIDerivedType(tag: DW_TAG_member, name: "maxGridSize", scope: !134, file: !6, line: 1267, baseType: !147, size: 96, offset: 2432)
!151 = !DIDerivedType(tag: DW_TAG_member, name: "clockRate", scope: !134, file: !6, line: 1268, baseType: !97, size: 32, offset: 2528)
!152 = !DIDerivedType(tag: DW_TAG_member, name: "totalConstMem", scope: !134, file: !6, line: 1269, baseType: !121, size: 64, offset: 2560)
!153 = !DIDerivedType(tag: DW_TAG_member, name: "major", scope: !134, file: !6, line: 1270, baseType: !97, size: 32, offset: 2624)
!154 = !DIDerivedType(tag: DW_TAG_member, name: "minor", scope: !134, file: !6, line: 1271, baseType: !97, size: 32, offset: 2656)
!155 = !DIDerivedType(tag: DW_TAG_member, name: "textureAlignment", scope: !134, file: !6, line: 1272, baseType: !121, size: 64, offset: 2688)
!156 = !DIDerivedType(tag: DW_TAG_member, name: "texturePitchAlignment", scope: !134, file: !6, line: 1273, baseType: !121, size: 64, offset: 2752)
!157 = !DIDerivedType(tag: DW_TAG_member, name: "deviceOverlap", scope: !134, file: !6, line: 1274, baseType: !97, size: 32, offset: 2816)
!158 = !DIDerivedType(tag: DW_TAG_member, name: "multiProcessorCount", scope: !134, file: !6, line: 1275, baseType: !97, size: 32, offset: 2848)
!159 = !DIDerivedType(tag: DW_TAG_member, name: "kernelExecTimeoutEnabled", scope: !134, file: !6, line: 1276, baseType: !97, size: 32, offset: 2880)
!160 = !DIDerivedType(tag: DW_TAG_member, name: "integrated", scope: !134, file: !6, line: 1277, baseType: !97, size: 32, offset: 2912)
!161 = !DIDerivedType(tag: DW_TAG_member, name: "canMapHostMemory", scope: !134, file: !6, line: 1278, baseType: !97, size: 32, offset: 2944)
!162 = !DIDerivedType(tag: DW_TAG_member, name: "computeMode", scope: !134, file: !6, line: 1279, baseType: !97, size: 32, offset: 2976)
!163 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture1D", scope: !134, file: !6, line: 1280, baseType: !97, size: 32, offset: 3008)
!164 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture1DMipmap", scope: !134, file: !6, line: 1281, baseType: !97, size: 32, offset: 3040)
!165 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture1DLinear", scope: !134, file: !6, line: 1282, baseType: !97, size: 32, offset: 3072)
!166 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture2D", scope: !134, file: !6, line: 1283, baseType: !167, size: 64, offset: 3104)
!167 = !DICompositeType(tag: DW_TAG_array_type, baseType: !97, size: 64, elements: !168)
!168 = !{!169}
!169 = !DISubrange(count: 2)
!170 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture2DMipmap", scope: !134, file: !6, line: 1284, baseType: !167, size: 64, offset: 3168)
!171 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture2DLinear", scope: !134, file: !6, line: 1285, baseType: !147, size: 96, offset: 3232)
!172 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture2DGather", scope: !134, file: !6, line: 1286, baseType: !167, size: 64, offset: 3328)
!173 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture3D", scope: !134, file: !6, line: 1287, baseType: !147, size: 96, offset: 3392)
!174 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture3DAlt", scope: !134, file: !6, line: 1288, baseType: !147, size: 96, offset: 3488)
!175 = !DIDerivedType(tag: DW_TAG_member, name: "maxTextureCubemap", scope: !134, file: !6, line: 1289, baseType: !97, size: 32, offset: 3584)
!176 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture1DLayered", scope: !134, file: !6, line: 1290, baseType: !167, size: 64, offset: 3616)
!177 = !DIDerivedType(tag: DW_TAG_member, name: "maxTexture2DLayered", scope: !134, file: !6, line: 1291, baseType: !147, size: 96, offset: 3680)
!178 = !DIDerivedType(tag: DW_TAG_member, name: "maxTextureCubemapLayered", scope: !134, file: !6, line: 1292, baseType: !167, size: 64, offset: 3776)
!179 = !DIDerivedType(tag: DW_TAG_member, name: "maxSurface1D", scope: !134, file: !6, line: 1293, baseType: !97, size: 32, offset: 3840)
!180 = !DIDerivedType(tag: DW_TAG_member, name: "maxSurface2D", scope: !134, file: !6, line: 1294, baseType: !167, size: 64, offset: 3872)
!181 = !DIDerivedType(tag: DW_TAG_member, name: "maxSurface3D", scope: !134, file: !6, line: 1295, baseType: !147, size: 96, offset: 3936)
!182 = !DIDerivedType(tag: DW_TAG_member, name: "maxSurface1DLayered", scope: !134, file: !6, line: 1296, baseType: !167, size: 64, offset: 4032)
!183 = !DIDerivedType(tag: DW_TAG_member, name: "maxSurface2DLayered", scope: !134, file: !6, line: 1297, baseType: !147, size: 96, offset: 4096)
!184 = !DIDerivedType(tag: DW_TAG_member, name: "maxSurfaceCubemap", scope: !134, file: !6, line: 1298, baseType: !97, size: 32, offset: 4192)
!185 = !DIDerivedType(tag: DW_TAG_member, name: "maxSurfaceCubemapLayered", scope: !134, file: !6, line: 1299, baseType: !167, size: 64, offset: 4224)
!186 = !DIDerivedType(tag: DW_TAG_member, name: "surfaceAlignment", scope: !134, file: !6, line: 1300, baseType: !121, size: 64, offset: 4288)
!187 = !DIDerivedType(tag: DW_TAG_member, name: "concurrentKernels", scope: !134, file: !6, line: 1301, baseType: !97, size: 32, offset: 4352)
!188 = !DIDerivedType(tag: DW_TAG_member, name: "ECCEnabled", scope: !134, file: !6, line: 1302, baseType: !97, size: 32, offset: 4384)
!189 = !DIDerivedType(tag: DW_TAG_member, name: "pciBusID", scope: !134, file: !6, line: 1303, baseType: !97, size: 32, offset: 4416)
!190 = !DIDerivedType(tag: DW_TAG_member, name: "pciDeviceID", scope: !134, file: !6, line: 1304, baseType: !97, size: 32, offset: 4448)
!191 = !DIDerivedType(tag: DW_TAG_member, name: "pciDomainID", scope: !134, file: !6, line: 1305, baseType: !97, size: 32, offset: 4480)
!192 = !DIDerivedType(tag: DW_TAG_member, name: "tccDriver", scope: !134, file: !6, line: 1306, baseType: !97, size: 32, offset: 4512)
!193 = !DIDerivedType(tag: DW_TAG_member, name: "asyncEngineCount", scope: !134, file: !6, line: 1307, baseType: !97, size: 32, offset: 4544)
!194 = !DIDerivedType(tag: DW_TAG_member, name: "unifiedAddressing", scope: !134, file: !6, line: 1308, baseType: !97, size: 32, offset: 4576)
!195 = !DIDerivedType(tag: DW_TAG_member, name: "memoryClockRate", scope: !134, file: !6, line: 1309, baseType: !97, size: 32, offset: 4608)
!196 = !DIDerivedType(tag: DW_TAG_member, name: "memoryBusWidth", scope: !134, file: !6, line: 1310, baseType: !97, size: 32, offset: 4640)
!197 = !DIDerivedType(tag: DW_TAG_member, name: "l2CacheSize", scope: !134, file: !6, line: 1311, baseType: !97, size: 32, offset: 4672)
!198 = !DIDerivedType(tag: DW_TAG_member, name: "maxThreadsPerMultiProcessor", scope: !134, file: !6, line: 1312, baseType: !97, size: 32, offset: 4704)
!199 = !DIDerivedType(tag: DW_TAG_member, name: "streamPrioritiesSupported", scope: !134, file: !6, line: 1313, baseType: !97, size: 32, offset: 4736)
!200 = !DIDerivedType(tag: DW_TAG_member, name: "globalL1CacheSupported", scope: !134, file: !6, line: 1314, baseType: !97, size: 32, offset: 4768)
!201 = !DIDerivedType(tag: DW_TAG_member, name: "localL1CacheSupported", scope: !134, file: !6, line: 1315, baseType: !97, size: 32, offset: 4800)
!202 = !DIDerivedType(tag: DW_TAG_member, name: "sharedMemPerMultiprocessor", scope: !134, file: !6, line: 1316, baseType: !121, size: 64, offset: 4864)
!203 = !DIDerivedType(tag: DW_TAG_member, name: "regsPerMultiprocessor", scope: !134, file: !6, line: 1317, baseType: !97, size: 32, offset: 4928)
!204 = !DIDerivedType(tag: DW_TAG_member, name: "managedMemory", scope: !134, file: !6, line: 1318, baseType: !97, size: 32, offset: 4960)
!205 = !DIDerivedType(tag: DW_TAG_member, name: "isMultiGpuBoard", scope: !134, file: !6, line: 1319, baseType: !97, size: 32, offset: 4992)
!206 = !DIDerivedType(tag: DW_TAG_member, name: "multiGpuBoardGroupID", scope: !134, file: !6, line: 1320, baseType: !97, size: 32, offset: 5024)
!207 = !DIGlobalVariableExpression(var: !208, expr: !DIExpression())
!208 = distinct !DIGlobalVariable(name: "q", linkageName: "_ZL1q", scope: !2, file: !3, line: 81, type: !98, isLocal: true, isDefinition: true)
!209 = !{!210, !216, !221, !223, !225, !227, !229, !233, !235, !237, !239, !241, !243, !245, !247, !249, !251, !253, !255, !257, !259, !261, !265, !267, !269, !271, !275, !280, !282, !284, !289, !293, !295, !297, !299, !301, !303, !305, !307, !309, !314, !318, !320, !325, !329, !331, !333, !335, !337, !339, !343, !345, !347, !352, !358, !362, !364, !366, !368, !370, !374, !376, !378, !382, !384, !386, !388, !390, !392, !394, !396, !398, !400, !404, !410, !412, !414, !418, !420, !422, !424, !426, !428, !430, !432, !436, !440, !442, !444, !448, !450, !452, !454, !456, !458, !460, !464, !470, !474, !479, !481, !485, !489, !499, !503, !507, !511, !515, !519, !521, !525, !529, !533, !541, !545, !549, !553, !557, !561, !567, !571, !575, !577, !585, !589, !596, !598, !600, !604, !608, !612, !617, !621, !626, !627, !628, !629, !631, !632, !633, !634, !635, !636, !637, !639, !640, !641, !642, !643, !647, !648, !649, !650, !651, !652, !653, !654, !655, !656, !657, !658, !659, !660, !661, !662, !663, !664, !665, !666, !667, !668, !669, !670, !671, !675, !677, !679, !681, !683, !685, !687, !689, !692, !694, !696, !698, !700, !702, !704, !706, !708, !710, !712, !714, !716, !718, !720, !722, !724, !726, !728, !730, !732, !734, !736, !738, !740, !742, !744, !746, !748, !750, !752, !754, !756, !758, !760, !762, !764, !766, !768, !770, !772, !774, !776, !778, !780, !782, !784, !790, !796, !801, !805, !807, !809, !811, !813, !820, !824, !828, !832, !836, !840, !845, !849, !851, !855, !861, !865, !870, !872, !874, !878, !882, !886, !888, !890, !892, !894, !898, !900, !902, !906, !910, !914, !918, !922, !924, !926, !932, !936, !940, !944, !946, !948, !952, !956, !957, !958, !959, !960}
!210 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !212, file: !213, line: 223)
!211 = !DINamespace(name: "std", scope: null)
!212 = !DISubprogram(name: "abs", linkageName: "_ZL3absi", scope: !213, file: !213, line: 53, type: !214, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!213 = !DIFile(filename: "llvm-install-tulip/lib/clang/9.0.0/include/__clang_cuda_math_forward_declares.h", directory: "/scratch/ah7226")
!214 = !DISubroutineType(types: !215)
!215 = !{!97, !97}
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
!274 = !{!97, !220}
!275 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !276, file: !213, line: 249)
!276 = !DISubprogram(name: "frexp", linkageName: "_ZL5frexpfPi", scope: !213, file: !213, line: 105, type: !277, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!277 = !DISubroutineType(types: !278)
!278 = !{!220, !220, !279}
!279 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !97, size: 64)
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
!317 = !{!220, !220, !97}
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
!355 = !{!99, !356}
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
!408 = !{!99, !99}
!409 = !DIFile(filename: "/usr/lib/gcc/x86_64-linux-gnu/11/../../../../include/c++/11/cmath", directory: "")
!410 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !411, file: !409, line: 102)
!411 = !DISubprogram(name: "asin", scope: !406, file: !406, line: 55, type: !407, flags: DIFlagPrototyped, spFlags: 0)
!412 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !413, file: !409, line: 121)
!413 = !DISubprogram(name: "atan", scope: !406, file: !406, line: 57, type: !407, flags: DIFlagPrototyped, spFlags: 0)
!414 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !415, file: !409, line: 140)
!415 = !DISubprogram(name: "atan2", scope: !406, file: !406, line: 59, type: !416, flags: DIFlagPrototyped, spFlags: 0)
!416 = !DISubroutineType(types: !417)
!417 = !{!99, !99, !99}
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
!435 = !{!99, !99, !279}
!436 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !437, file: !409, line: 315)
!437 = !DISubprogram(name: "ldexp", scope: !406, file: !406, line: 101, type: !438, flags: DIFlagPrototyped, spFlags: 0)
!438 = !DISubroutineType(types: !439)
!439 = !{!99, !99, !97}
!440 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !441, file: !409, line: 334)
!441 = !DISubprogram(name: "log", scope: !406, file: !406, line: 104, type: !407, flags: DIFlagPrototyped, spFlags: 0)
!442 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !443, file: !409, line: 353)
!443 = !DISubprogram(name: "log10", scope: !406, file: !406, line: 107, type: !407, flags: DIFlagPrototyped, spFlags: 0)
!444 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !445, file: !409, line: 372)
!445 = !DISubprogram(name: "modf", scope: !406, file: !406, line: 110, type: !446, flags: DIFlagPrototyped, spFlags: 0)
!446 = !DISubroutineType(types: !447)
!447 = !{!99, !99, !98}
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
!477 = !{!97, !478}
!478 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !472, size: 64)
!479 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !480, file: !463, line: 140)
!480 = !DISubprogram(name: "atof", scope: !402, file: !402, line: 102, type: !354, flags: DIFlagPrototyped, spFlags: 0)
!481 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !482, file: !463, line: 141)
!482 = !DISubprogram(name: "atoi", scope: !402, file: !402, line: 105, type: !483, flags: DIFlagPrototyped, spFlags: 0)
!483 = !DISubroutineType(types: !484)
!484 = !{!97, !356}
!485 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !486, file: !463, line: 142)
!486 = !DISubprogram(name: "atol", scope: !402, file: !402, line: 108, type: !487, flags: DIFlagPrototyped, spFlags: 0)
!487 = !DISubroutineType(types: !488)
!488 = !{!313, !356}
!489 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !490, file: !463, line: 143)
!490 = !DISubprogram(name: "bsearch", scope: !402, file: !402, line: 828, type: !491, flags: DIFlagPrototyped, spFlags: 0)
!491 = !DISubroutineType(types: !492)
!492 = !{!103, !493, !493, !121, !121, !495}
!493 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !494, size: 64)
!494 = !DIDerivedType(tag: DW_TAG_const_type, baseType: null)
!495 = !DIDerivedType(tag: DW_TAG_typedef, name: "__compar_fn_t", file: !402, line: 816, baseType: !496)
!496 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !497, size: 64)
!497 = !DISubroutineType(types: !498)
!498 = !{!97, !493, !493}
!499 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !500, file: !463, line: 144)
!500 = !DISubprogram(name: "calloc", scope: !402, file: !402, line: 543, type: !501, flags: DIFlagPrototyped, spFlags: 0)
!501 = !DISubroutineType(types: !502)
!502 = !{!103, !121, !121}
!503 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !504, file: !463, line: 145)
!504 = !DISubprogram(name: "div", scope: !402, file: !402, line: 860, type: !505, flags: DIFlagPrototyped, spFlags: 0)
!505 = !DISubroutineType(types: !506)
!506 = !{!461, !97, !97}
!507 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !508, file: !463, line: 146)
!508 = !DISubprogram(name: "exit", scope: !402, file: !402, line: 624, type: !509, flags: DIFlagPrototyped | DIFlagNoReturn, spFlags: 0)
!509 = !DISubroutineType(types: !510)
!510 = !{null, !97}
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
!528 = !{!103, !121}
!529 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !530, file: !463, line: 153)
!530 = !DISubprogram(name: "mblen", scope: !402, file: !402, line: 930, type: !531, flags: DIFlagPrototyped, spFlags: 0)
!531 = !DISubroutineType(types: !532)
!532 = !{!97, !356, !121}
!533 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !534, file: !463, line: 154)
!534 = !DISubprogram(name: "mbstowcs", scope: !402, file: !402, line: 941, type: !535, flags: DIFlagPrototyped, spFlags: 0)
!535 = !DISubroutineType(types: !536)
!536 = !{!121, !537, !540, !121}
!537 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !538)
!538 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !539, size: 64)
!539 = !DIBasicType(name: "wchar_t", size: 32, encoding: DW_ATE_signed)
!540 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !356)
!541 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !542, file: !463, line: 155)
!542 = !DISubprogram(name: "mbtowc", scope: !402, file: !402, line: 933, type: !543, flags: DIFlagPrototyped, spFlags: 0)
!543 = !DISubroutineType(types: !544)
!544 = !{!97, !537, !540, !121}
!545 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !546, file: !463, line: 157)
!546 = !DISubprogram(name: "qsort", scope: !402, file: !402, line: 838, type: !547, flags: DIFlagPrototyped, spFlags: 0)
!547 = !DISubroutineType(types: !548)
!548 = !{null, !103, !121, !121, !495}
!549 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !550, file: !463, line: 163)
!550 = !DISubprogram(name: "rand", scope: !402, file: !402, line: 454, type: !551, flags: DIFlagPrototyped, spFlags: 0)
!551 = !DISubroutineType(types: !552)
!552 = !{!97}
!553 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !554, file: !463, line: 164)
!554 = !DISubprogram(name: "realloc", scope: !402, file: !402, line: 551, type: !555, flags: DIFlagPrototyped, spFlags: 0)
!555 = !DISubroutineType(types: !556)
!556 = !{!103, !103, !121}
!557 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !558, file: !463, line: 165)
!558 = !DISubprogram(name: "srand", scope: !402, file: !402, line: 456, type: !559, flags: DIFlagPrototyped, spFlags: 0)
!559 = !DISubroutineType(types: !560)
!560 = !{null, !7}
!561 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !562, file: !463, line: 166)
!562 = !DISubprogram(name: "strtod", scope: !402, file: !402, line: 118, type: !563, flags: DIFlagPrototyped, spFlags: 0)
!563 = !DISubroutineType(types: !564)
!564 = !{!99, !540, !565}
!565 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !566)
!566 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !100, size: 64)
!567 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !568, file: !463, line: 167)
!568 = !DISubprogram(name: "strtol", scope: !402, file: !402, line: 177, type: !569, flags: DIFlagPrototyped, spFlags: 0)
!569 = !DISubroutineType(types: !570)
!570 = !{!313, !540, !565, !97}
!571 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !572, file: !463, line: 168)
!572 = !DISubprogram(name: "strtoul", scope: !402, file: !402, line: 181, type: !573, flags: DIFlagPrototyped, spFlags: 0)
!573 = !DISubroutineType(types: !574)
!574 = !{!123, !540, !565, !97}
!575 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !576, file: !463, line: 169)
!576 = !DISubprogram(name: "system", scope: !402, file: !402, line: 791, type: !483, flags: DIFlagPrototyped, spFlags: 0)
!577 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !578, file: !463, line: 171)
!578 = !DISubprogram(name: "wcstombs", scope: !402, file: !402, line: 945, type: !579, flags: DIFlagPrototyped, spFlags: 0)
!579 = !DISubroutineType(types: !580)
!580 = !{!121, !581, !582, !121}
!581 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !100)
!582 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !583)
!583 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !584, size: 64)
!584 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !539)
!585 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !586, file: !463, line: 172)
!586 = !DISubprogram(name: "wctomb", scope: !402, file: !402, line: 937, type: !587, flags: DIFlagPrototyped, spFlags: 0)
!587 = !DISubroutineType(types: !588)
!588 = !{!97, !100, !539}
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
!611 = !{!324, !540, !565, !97}
!612 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !590, entity: !613, file: !463, line: 229)
!613 = !DISubprogram(name: "strtoull", scope: !402, file: !402, line: 206, type: !614, flags: DIFlagPrototyped, spFlags: 0)
!614 = !DISubroutineType(types: !615)
!615 = !{!616, !540, !565, !97}
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
!674 = !DIFile(filename: "llvm-install-tulip/lib/clang/9.0.0/include/__clang_cuda_cmath.h", directory: "/scratch/ah7226")
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
!804 = !{!97, !800}
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
!816 = !{!97, !817, !818}
!817 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !800)
!818 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !819)
!819 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !791, size: 64)
!820 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !821, file: !789, line: 108)
!821 = !DISubprogram(name: "fgets", scope: !792, file: !792, line: 592, type: !822, flags: DIFlagPrototyped, spFlags: 0)
!822 = !DISubroutineType(types: !823)
!823 = !{!100, !581, !97, !817}
!824 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !825, file: !789, line: 109)
!825 = !DISubprogram(name: "fopen", scope: !792, file: !792, line: 258, type: !826, flags: DIFlagPrototyped, spFlags: 0)
!826 = !DISubroutineType(types: !827)
!827 = !{!800, !540, !540}
!828 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !829, file: !789, line: 110)
!829 = !DISubprogram(name: "fprintf", scope: !792, file: !792, line: 350, type: !830, flags: DIFlagPrototyped, spFlags: 0)
!830 = !DISubroutineType(types: !831)
!831 = !{!97, !817, !540, null}
!832 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !833, file: !789, line: 111)
!833 = !DISubprogram(name: "fputc", scope: !792, file: !792, line: 549, type: !834, flags: DIFlagPrototyped, spFlags: 0)
!834 = !DISubroutineType(types: !835)
!835 = !{!97, !97, !800}
!836 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !837, file: !789, line: 112)
!837 = !DISubprogram(name: "fputs", scope: !792, file: !792, line: 655, type: !838, flags: DIFlagPrototyped, spFlags: 0)
!838 = !DISubroutineType(types: !839)
!839 = !{!97, !540, !817}
!840 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !841, file: !789, line: 113)
!841 = !DISubprogram(name: "fread", scope: !792, file: !792, line: 675, type: !842, flags: DIFlagPrototyped, spFlags: 0)
!842 = !DISubroutineType(types: !843)
!843 = !{!121, !844, !121, !121, !817}
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
!854 = !{!97, !800, !313, !97}
!855 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !856, file: !789, line: 117)
!856 = !DISubprogram(name: "fsetpos", scope: !792, file: !792, line: 765, type: !857, flags: DIFlagPrototyped, spFlags: 0)
!857 = !DISubroutineType(types: !858)
!858 = !{!97, !800, !859}
!859 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !860, size: 64)
!860 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !791)
!861 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !862, file: !789, line: 118)
!862 = !DISubprogram(name: "ftell", scope: !792, file: !792, line: 718, type: !863, flags: DIFlagPrototyped, spFlags: 0)
!863 = !DISubroutineType(types: !864)
!864 = !{!313, !800}
!865 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !866, file: !789, line: 119)
!866 = !DISubprogram(name: "fwrite", scope: !792, file: !792, line: 681, type: !867, flags: DIFlagPrototyped, spFlags: 0)
!867 = !DISubroutineType(types: !868)
!868 = !{!121, !869, !121, !121, !817}
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
!885 = !{!97, !540, null}
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
!897 = !{!97, !356, !356}
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
!909 = !{!97, !817, !581, !97, !121}
!910 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !911, file: !789, line: 137)
!911 = !DISubprogram(name: "sprintf", scope: !792, file: !792, line: 358, type: !912, flags: DIFlagPrototyped, spFlags: 0)
!912 = !DISubroutineType(types: !913)
!913 = !{!97, !581, !540, null}
!914 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !915, file: !789, line: 138)
!915 = !DISubprogram(name: "sscanf", scope: !792, file: !792, line: 423, type: !916, flags: DIFlagPrototyped, spFlags: 0)
!916 = !DISubroutineType(types: !917)
!917 = !{!97, !540, !540, null}
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
!929 = !{!97, !817, !540, !930}
!930 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !931, size: 64)
!931 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "__va_list_tag", file: !3, flags: DIFlagFwdDecl, identifier: "_ZTS13__va_list_tag")
!932 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !933, file: !789, line: 145)
!933 = !DISubprogram(name: "vprintf", scope: !792, file: !792, line: 371, type: !934, flags: DIFlagPrototyped, spFlags: 0)
!934 = !DISubroutineType(types: !935)
!935 = !{!97, !540, !930}
!936 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !937, file: !789, line: 146)
!937 = !DISubprogram(name: "vsprintf", scope: !792, file: !792, line: 373, type: !938, flags: DIFlagPrototyped, spFlags: 0)
!938 = !DISubroutineType(types: !939)
!939 = !{!97, !581, !540, !930}
!940 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !590, entity: !941, file: !789, line: 175)
!941 = !DISubprogram(name: "snprintf", scope: !792, file: !792, line: 378, type: !942, flags: DIFlagPrototyped, spFlags: 0)
!942 = !DISubroutineType(types: !943)
!943 = !{!97, !581, !121, !540, null}
!944 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !590, entity: !945, file: !789, line: 176)
!945 = !DISubprogram(name: "vfscanf", scope: !792, file: !792, line: 459, type: !928, flags: DIFlagPrototyped, spFlags: 0)
!946 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !590, entity: !947, file: !789, line: 177)
!947 = !DISubprogram(name: "vscanf", scope: !792, file: !792, line: 467, type: !934, flags: DIFlagPrototyped, spFlags: 0)
!948 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !590, entity: !949, file: !789, line: 178)
!949 = !DISubprogram(name: "vsnprintf", scope: !792, file: !792, line: 382, type: !950, flags: DIFlagPrototyped, spFlags: 0)
!950 = !DISubroutineType(types: !951)
!951 = !{!97, !581, !121, !540, !930}
!952 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !590, entity: !953, file: !789, line: 179)
!953 = !DISubprogram(name: "vsscanf", scope: !792, file: !792, line: 471, type: !954, flags: DIFlagPrototyped, spFlags: 0)
!954 = !DISubroutineType(types: !955)
!955 = !{!97, !540, !540, !930}
!956 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !941, file: !789, line: 185)
!957 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !945, file: !789, line: 186)
!958 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !947, file: !789, line: 187)
!959 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !949, file: !789, line: 188)
!960 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !953, file: !789, line: 189)
!961 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 9.0.0 (https://github.com/yebinchon/llvm-project/ a1efa594106d738d0b74c9e4e2b9b779eb8b7d25)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !962, retainedTypes: !963, imports: !967, nameTableKind: None)
!962 = !{}
!963 = !{!97, !964, !616, !965, !324}
!964 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !616, size: 64)
!965 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !966, size: 64)
!966 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !324)
!967 = !{!210, !216, !221, !223, !225, !227, !229, !233, !235, !237, !239, !241, !243, !245, !247, !249, !251, !253, !255, !257, !259, !261, !265, !267, !269, !271, !275, !280, !282, !284, !289, !293, !295, !297, !299, !301, !303, !305, !307, !309, !314, !318, !320, !325, !329, !331, !333, !335, !337, !339, !343, !345, !347, !352, !358, !362, !364, !366, !368, !370, !374, !376, !378, !382, !384, !386, !388, !390, !392, !394, !396, !398, !400, !404, !410, !412, !414, !418, !420, !422, !424, !426, !428, !430, !432, !436, !440, !442, !444, !448, !450, !452, !454, !456, !458, !460, !464, !470, !474, !479, !481, !485, !489, !499, !503, !507, !511, !515, !519, !521, !525, !529, !533, !541, !545, !549, !553, !557, !561, !567, !571, !575, !577, !585, !589, !596, !598, !600, !604, !608, !612, !617, !968, !626, !627, !628, !629, !631, !632, !633, !634, !635, !973, !974, !975, !976, !977, !978, !979, !983, !984, !985, !986, !987, !988, !989, !990, !991, !992, !993, !994, !995, !996, !997, !998, !999, !1000, !1001, !1002, !1003, !1004, !1005, !1006, !671, !675, !677, !679, !681, !683, !685, !687, !689, !692, !694, !696, !698, !700, !702, !704, !706, !708, !710, !712, !714, !716, !718, !720, !722, !724, !726, !728, !730, !732, !734, !736, !738, !740, !742, !744, !746, !748, !750, !752, !754, !756, !758, !760, !762, !764, !766, !768, !770, !772, !774, !776, !778, !780, !782, !784, !790, !796, !801, !805, !807, !809, !811, !813, !820, !824, !828, !832, !836, !840, !845, !849, !851, !855, !861, !865, !870, !872, !874, !878, !882, !886, !888, !890, !892, !894, !898, !900, !902, !906, !910, !914, !918, !922, !924, !1007, !1014, !1018, !940, !1022, !1024, !1026, !1030, !956, !1034, !1035, !1036, !1037}
!968 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !590, entity: !969, file: !463, line: 232)
!969 = !DISubprogram(name: "strtold", scope: !402, file: !402, line: 127, type: !970, flags: DIFlagPrototyped, spFlags: 0)
!970 = !DISubroutineType(types: !971)
!971 = !{!972, !540, !565}
!972 = !DIBasicType(name: "long double", size: 64, encoding: DW_ATE_float)
!973 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !969, file: !463, line: 252)
!974 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !961, entity: !471, file: !638, line: 38)
!975 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !961, entity: !475, file: !638, line: 39)
!976 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !961, entity: !508, file: !638, line: 40)
!977 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !961, entity: !461, file: !638, line: 51)
!978 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !961, entity: !465, file: !638, line: 52)
!979 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !961, entity: !980, file: !638, line: 54)
!980 = !DISubprogram(name: "abs", linkageName: "_ZSt3abse", scope: !211, file: !403, line: 79, type: !981, flags: DIFlagPrototyped, spFlags: 0)
!981 = !DISubroutineType(types: !982)
!982 = !{!972, !972}
!983 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !961, entity: !480, file: !638, line: 55)
!984 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !961, entity: !482, file: !638, line: 56)
!985 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !961, entity: !486, file: !638, line: 57)
!986 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !961, entity: !490, file: !638, line: 58)
!987 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !961, entity: !500, file: !638, line: 59)
!988 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !961, entity: !630, file: !638, line: 60)
!989 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !961, entity: !512, file: !638, line: 61)
!990 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !961, entity: !516, file: !638, line: 62)
!991 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !961, entity: !520, file: !638, line: 63)
!992 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !961, entity: !522, file: !638, line: 64)
!993 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !961, entity: !526, file: !638, line: 65)
!994 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !961, entity: !530, file: !638, line: 67)
!995 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !961, entity: !534, file: !638, line: 68)
!996 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !961, entity: !542, file: !638, line: 69)
!997 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !961, entity: !546, file: !638, line: 71)
!998 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !961, entity: !550, file: !638, line: 72)
!999 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !961, entity: !554, file: !638, line: 73)
!1000 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !961, entity: !558, file: !638, line: 74)
!1001 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !961, entity: !562, file: !638, line: 75)
!1002 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !961, entity: !568, file: !638, line: 76)
!1003 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !961, entity: !572, file: !638, line: 77)
!1004 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !961, entity: !576, file: !638, line: 78)
!1005 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !961, entity: !578, file: !638, line: 80)
!1006 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !961, entity: !586, file: !638, line: 81)
!1007 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !1008, file: !789, line: 144)
!1008 = !DISubprogram(name: "vfprintf", scope: !792, file: !792, line: 365, type: !1009, flags: DIFlagPrototyped, spFlags: 0)
!1009 = !DISubroutineType(types: !1010)
!1010 = !{!97, !817, !540, !1011}
!1011 = !DIDerivedType(tag: DW_TAG_typedef, name: "__gnuc_va_list", file: !1012, line: 32, baseType: !1013)
!1012 = !DIFile(filename: "llvm-install-tulip/lib/clang/9.0.0/include/stdarg.h", directory: "/scratch/ah7226")
!1013 = !DIDerivedType(tag: DW_TAG_typedef, name: "__builtin_va_list", file: !3, baseType: !100)
!1014 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !1015, file: !789, line: 145)
!1015 = !DISubprogram(name: "vprintf", scope: !792, file: !792, line: 371, type: !1016, flags: DIFlagPrototyped, spFlags: 0)
!1016 = !DISubroutineType(types: !1017)
!1017 = !{!97, !540, !1011}
!1018 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !1019, file: !789, line: 146)
!1019 = !DISubprogram(name: "vsprintf", scope: !792, file: !792, line: 373, type: !1020, flags: DIFlagPrototyped, spFlags: 0)
!1020 = !DISubroutineType(types: !1021)
!1021 = !{!97, !581, !540, !1011}
!1022 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !590, entity: !1023, file: !789, line: 176)
!1023 = !DISubprogram(name: "vfscanf", scope: !792, file: !792, line: 459, type: !1009, flags: DIFlagPrototyped, spFlags: 0)
!1024 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !590, entity: !1025, file: !789, line: 177)
!1025 = !DISubprogram(name: "vscanf", scope: !792, file: !792, line: 467, type: !1016, flags: DIFlagPrototyped, spFlags: 0)
!1026 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !590, entity: !1027, file: !789, line: 178)
!1027 = !DISubprogram(name: "vsnprintf", scope: !792, file: !792, line: 382, type: !1028, flags: DIFlagPrototyped, spFlags: 0)
!1028 = !DISubroutineType(types: !1029)
!1029 = !{!97, !581, !121, !540, !1011}
!1030 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !590, entity: !1031, file: !789, line: 179)
!1031 = !DISubprogram(name: "vsscanf", scope: !792, file: !792, line: 471, type: !1032, flags: DIFlagPrototyped, spFlags: 0)
!1032 = !DISubroutineType(types: !1033)
!1033 = !{!97, !540, !540, !1011}
!1034 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !1023, file: !789, line: 186)
!1035 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !1025, file: !789, line: 187)
!1036 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !1027, file: !789, line: 188)
!1037 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !1031, file: !789, line: 189)
!1038 = !{void (double*, double*, double*, double)* @_Z10gpu_kernelPdS_S_d, !"kernel", i32 1}
!1039 = !{null, !"align", i32 8}
!1040 = !{null, !"align", i32 8, !"align", i32 65544, !"align", i32 131080}
!1041 = !{null, !"align", i32 16}
!1042 = !{null, !"align", i32 16, !"align", i32 65552, !"align", i32 131088}
!1043 = !{!"clang version 9.0.0 (https://github.com/yebinchon/llvm-project/ a1efa594106d738d0b74c9e4e2b9b779eb8b7d25)"}
!1044 = !{i32 1, i32 2}
!1045 = !{i32 2, !"SDK Version", [2 x i32] [i32 7, i32 0]}
!1046 = !{i32 2, !"Dwarf Version", i32 2}
!1047 = !{i32 2, !"Debug Info Version", i32 3}
!1048 = !{i32 1, !"wchar_size", i32 4}
!1049 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!1050 = distinct !DISubprogram(name: "gpu_kernel", linkageName: "_Z10gpu_kernelPdS_S_d", scope: !3, file: !3, line: 464, type: !1051, scopeLine: 467, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !961, retainedNodes: !962)
!1051 = !DISubroutineType(types: !1052)
!1052 = !{null, !98, !98, !98, !99}
!1053 = !DILocalVariable(name: "f", arg: 1, scope: !1054, file: !691, line: 587, type: !99)
!1054 = distinct !DISubprogram(name: "fabs", linkageName: "_ZL4fabsd", scope: !691, file: !691, line: 587, type: !407, scopeLine: 588, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !961, retainedNodes: !962)
!1055 = !DILocation(line: 587, column: 53, scope: !1054, inlinedAt: !1056)
!1056 = distinct !DILocation(line: 531, column: 7, scope: !1057)
!1057 = distinct !DILexicalBlock(scope: !1058, file: !3, line: 527, column: 15)
!1058 = distinct !DILexicalBlock(scope: !1059, file: !3, line: 527, column: 7)
!1059 = distinct !DILexicalBlock(scope: !1060, file: !3, line: 523, column: 33)
!1060 = distinct !DILexicalBlock(scope: !1061, file: !3, line: 523, column: 3)
!1061 = distinct !DILexicalBlock(scope: !1062, file: !3, line: 523, column: 3)
!1062 = distinct !DILexicalBlock(scope: !1063, file: !3, line: 514, column: 39)
!1063 = distinct !DILexicalBlock(scope: !1064, file: !3, line: 514, column: 2)
!1064 = distinct !DILexicalBlock(scope: !1050, file: !3, line: 514, column: 2)
!1065 = !DILocation(line: 587, column: 53, scope: !1054, inlinedAt: !1066)
!1066 = distinct !DILocation(line: 531, column: 7, scope: !1057)
!1067 = !DILocation(line: 587, column: 53, scope: !1054, inlinedAt: !1068)
!1068 = distinct !DILocation(line: 531, column: 7, scope: !1057)
!1069 = !DILocation(line: 587, column: 53, scope: !1054, inlinedAt: !1070)
!1070 = distinct !DILocation(line: 531, column: 7, scope: !1057)
!1071 = !DILocalVariable(name: "x", arg: 1, scope: !1072, file: !691, line: 892, type: !99)
!1072 = distinct !DISubprogram(name: "sqrt", linkageName: "_ZL4sqrtd", scope: !691, file: !691, line: 892, type: !407, scopeLine: 893, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !961, retainedNodes: !962)
!1073 = !DILocation(line: 892, column: 53, scope: !1072, inlinedAt: !1074)
!1074 = distinct !DILocation(line: 528, column: 8, scope: !1057)
!1075 = !DILocalVariable(name: "a", arg: 1, scope: !1076, file: !1077, line: 225, type: !99)
!1076 = distinct !DISubprogram(name: "log", linkageName: "_ZL3logd", scope: !1077, file: !1077, line: 225, type: !407, scopeLine: 226, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !961, retainedNodes: !962)
!1077 = !DIFile(filename: "/usr/local/cuda/include/math_functions_dbl_ptx3.hpp", directory: "")
!1078 = !DILocation(line: 225, column: 52, scope: !1076, inlinedAt: !1079)
!1079 = distinct !DILocation(line: 528, column: 19, scope: !1057)
!1080 = !DILocalVariable(name: "q_global", arg: 1, scope: !1050, file: !3, line: 464, type: !98)
!1081 = !DILocation(line: 464, column: 36, scope: !1050)
!1082 = !DILocalVariable(name: "sx_global", arg: 2, scope: !1050, file: !3, line: 465, type: !98)
!1083 = !DILocation(line: 465, column: 11, scope: !1050)
!1084 = !DILocalVariable(name: "sy_global", arg: 3, scope: !1050, file: !3, line: 466, type: !98)
!1085 = !DILocation(line: 466, column: 11, scope: !1050)
!1086 = !DILocalVariable(name: "an", arg: 4, scope: !1050, file: !3, line: 467, type: !99)
!1087 = !DILocation(line: 467, column: 10, scope: !1050)
!1088 = !DILocalVariable(name: "x_local", scope: !1050, file: !3, line: 468, type: !1089)
!1089 = !DICompositeType(tag: DW_TAG_array_type, baseType: !99, size: 16384, elements: !138)
!1090 = !DILocation(line: 468, column: 9, scope: !1050)
!1091 = !DILocalVariable(name: "q_local", scope: !1050, file: !3, line: 469, type: !1092)
!1092 = !DICompositeType(tag: DW_TAG_array_type, baseType: !99, size: 640, elements: !1093)
!1093 = !{!1094}
!1094 = !DISubrange(count: 10)
!1095 = !DILocation(line: 469, column: 9, scope: !1050)
!1096 = !DILocalVariable(name: "sx_local", scope: !1050, file: !3, line: 470, type: !99)
!1097 = !DILocation(line: 470, column: 9, scope: !1050)
!1098 = !DILocalVariable(name: "sy_local", scope: !1050, file: !3, line: 470, type: !99)
!1099 = !DILocation(line: 470, column: 19, scope: !1050)
!1100 = !DILocalVariable(name: "t1", scope: !1050, file: !3, line: 471, type: !99)
!1101 = !DILocation(line: 471, column: 9, scope: !1050)
!1102 = !DILocalVariable(name: "t2", scope: !1050, file: !3, line: 471, type: !99)
!1103 = !DILocation(line: 471, column: 13, scope: !1050)
!1104 = !DILocalVariable(name: "t3", scope: !1050, file: !3, line: 471, type: !99)
!1105 = !DILocation(line: 471, column: 17, scope: !1050)
!1106 = !DILocalVariable(name: "t4", scope: !1050, file: !3, line: 471, type: !99)
!1107 = !DILocation(line: 471, column: 21, scope: !1050)
!1108 = !DILocalVariable(name: "x1", scope: !1050, file: !3, line: 471, type: !99)
!1109 = !DILocation(line: 471, column: 25, scope: !1050)
!1110 = !DILocalVariable(name: "x2", scope: !1050, file: !3, line: 471, type: !99)
!1111 = !DILocation(line: 471, column: 29, scope: !1050)
!1112 = !DILocalVariable(name: "seed", scope: !1050, file: !3, line: 471, type: !99)
!1113 = !DILocation(line: 471, column: 33, scope: !1050)
!1114 = !DILocalVariable(name: "i", scope: !1050, file: !3, line: 472, type: !97)
!1115 = !DILocation(line: 472, column: 6, scope: !1050)
!1116 = !DILocalVariable(name: "ii", scope: !1050, file: !3, line: 472, type: !97)
!1117 = !DILocation(line: 472, column: 9, scope: !1050)
!1118 = !DILocalVariable(name: "ik", scope: !1050, file: !3, line: 472, type: !97)
!1119 = !DILocation(line: 472, column: 13, scope: !1050)
!1120 = !DILocalVariable(name: "kk", scope: !1050, file: !3, line: 472, type: !97)
!1121 = !DILocation(line: 472, column: 17, scope: !1050)
!1122 = !DILocalVariable(name: "l", scope: !1050, file: !3, line: 472, type: !97)
!1123 = !DILocation(line: 472, column: 21, scope: !1050)
!1124 = !DILocation(line: 474, column: 2, scope: !1050)
!1125 = !DILocation(line: 474, column: 12, scope: !1050)
!1126 = !DILocation(line: 475, column: 2, scope: !1050)
!1127 = !DILocation(line: 475, column: 12, scope: !1050)
!1128 = !DILocation(line: 476, column: 2, scope: !1050)
!1129 = !DILocation(line: 476, column: 12, scope: !1050)
!1130 = !DILocation(line: 477, column: 2, scope: !1050)
!1131 = !DILocation(line: 477, column: 12, scope: !1050)
!1132 = !DILocation(line: 478, column: 2, scope: !1050)
!1133 = !DILocation(line: 478, column: 12, scope: !1050)
!1134 = !DILocation(line: 479, column: 2, scope: !1050)
!1135 = !DILocation(line: 479, column: 12, scope: !1050)
!1136 = !DILocation(line: 480, column: 2, scope: !1050)
!1137 = !DILocation(line: 480, column: 12, scope: !1050)
!1138 = !DILocation(line: 481, column: 2, scope: !1050)
!1139 = !DILocation(line: 481, column: 12, scope: !1050)
!1140 = !DILocation(line: 482, column: 2, scope: !1050)
!1141 = !DILocation(line: 482, column: 12, scope: !1050)
!1142 = !DILocation(line: 483, column: 2, scope: !1050)
!1143 = !DILocation(line: 483, column: 12, scope: !1050)
!1144 = !DILocation(line: 484, column: 10, scope: !1050)
!1145 = !DILocation(line: 485, column: 10, scope: !1050)
!1146 = !DILocation(line: 64, column: 3, scope: !1147, inlinedAt: !1182)
!1147 = distinct !DISubprogram(name: "__fetch_builtin_x", linkageName: "_ZN25__cuda_builtin_blockIdx_t17__fetch_builtin_xEv", scope: !1149, file: !1148, line: 64, type: !1152, scopeLine: 64, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !961, declaration: !1151, retainedNodes: !962)
!1148 = !DIFile(filename: "llvm-install-tulip/lib/clang/9.0.0/include/__clang_cuda_builtin_vars.h", directory: "/scratch/ah7226")
!1149 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "__cuda_builtin_blockIdx_t", file: !1148, line: 63, size: 8, flags: DIFlagTypePassByReference | DIFlagNonTrivial, elements: !1150, identifier: "_ZTS25__cuda_builtin_blockIdx_t")
!1150 = !{!1151, !1154, !1155, !1156, !1167, !1171, !1175, !1178}
!1151 = !DISubprogram(name: "__fetch_builtin_x", linkageName: "_ZN25__cuda_builtin_blockIdx_t17__fetch_builtin_xEv", scope: !1149, file: !1148, line: 64, type: !1152, scopeLine: 64, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!1152 = !DISubroutineType(types: !1153)
!1153 = !{!7}
!1154 = !DISubprogram(name: "__fetch_builtin_y", linkageName: "_ZN25__cuda_builtin_blockIdx_t17__fetch_builtin_yEv", scope: !1149, file: !1148, line: 65, type: !1152, scopeLine: 65, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!1155 = !DISubprogram(name: "__fetch_builtin_z", linkageName: "_ZN25__cuda_builtin_blockIdx_t17__fetch_builtin_zEv", scope: !1149, file: !1148, line: 66, type: !1152, scopeLine: 66, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!1156 = !DISubprogram(name: "operator uint3", linkageName: "_ZNK25__cuda_builtin_blockIdx_tcv5uint3Ev", scope: !1149, file: !1148, line: 69, type: !1157, scopeLine: 69, flags: DIFlagPrototyped, spFlags: 0)
!1157 = !DISubroutineType(types: !1158)
!1158 = !{!1159, !1165}
!1159 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "uint3", file: !1160, line: 190, size: 96, flags: DIFlagTypePassByValue, elements: !1161, identifier: "_ZTS5uint3")
!1160 = !DIFile(filename: "/usr/local/cuda/include/vector_types.h", directory: "")
!1161 = !{!1162, !1163, !1164}
!1162 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !1159, file: !1160, line: 192, baseType: !7, size: 32)
!1163 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !1159, file: !1160, line: 192, baseType: !7, size: 32, offset: 32)
!1164 = !DIDerivedType(tag: DW_TAG_member, name: "z", scope: !1159, file: !1160, line: 192, baseType: !7, size: 32, offset: 64)
!1165 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !1166, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!1166 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !1149)
!1167 = !DISubprogram(name: "__cuda_builtin_blockIdx_t", scope: !1149, file: !1148, line: 71, type: !1168, scopeLine: 71, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!1168 = !DISubroutineType(types: !1169)
!1169 = !{null, !1170}
!1170 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !1149, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!1171 = !DISubprogram(name: "__cuda_builtin_blockIdx_t", scope: !1149, file: !1148, line: 71, type: !1172, scopeLine: 71, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!1172 = !DISubroutineType(types: !1173)
!1173 = !{null, !1170, !1174}
!1174 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !1166, size: 64)
!1175 = !DISubprogram(name: "operator=", linkageName: "_ZNK25__cuda_builtin_blockIdx_taSERKS_", scope: !1149, file: !1148, line: 71, type: !1176, scopeLine: 71, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!1176 = !DISubroutineType(types: !1177)
!1177 = !{null, !1165, !1174}
!1178 = !DISubprogram(name: "operator&", linkageName: "_ZNK25__cuda_builtin_blockIdx_tadEv", scope: !1149, file: !1148, line: 71, type: !1179, scopeLine: 71, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!1179 = !DISubroutineType(types: !1180)
!1180 = !{!1181, !1165}
!1181 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !1149, size: 64)
!1182 = distinct !DILocation(line: 487, column: 5, scope: !1050)
!1183 = !{i32 0, i32 65535}
!1184 = !DILocation(line: 75, column: 3, scope: !1185, inlinedAt: !1227)
!1185 = distinct !DISubprogram(name: "__fetch_builtin_x", linkageName: "_ZN25__cuda_builtin_blockDim_t17__fetch_builtin_xEv", scope: !1186, file: !1148, line: 75, type: !1152, scopeLine: 75, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !961, declaration: !1188, retainedNodes: !962)
!1186 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "__cuda_builtin_blockDim_t", file: !1148, line: 74, size: 8, flags: DIFlagTypePassByReference | DIFlagNonTrivial, elements: !1187, identifier: "_ZTS25__cuda_builtin_blockDim_t")
!1187 = !{!1188, !1189, !1190, !1191, !1212, !1216, !1220, !1223}
!1188 = !DISubprogram(name: "__fetch_builtin_x", linkageName: "_ZN25__cuda_builtin_blockDim_t17__fetch_builtin_xEv", scope: !1186, file: !1148, line: 75, type: !1152, scopeLine: 75, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!1189 = !DISubprogram(name: "__fetch_builtin_y", linkageName: "_ZN25__cuda_builtin_blockDim_t17__fetch_builtin_yEv", scope: !1186, file: !1148, line: 76, type: !1152, scopeLine: 76, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!1190 = !DISubprogram(name: "__fetch_builtin_z", linkageName: "_ZN25__cuda_builtin_blockDim_t17__fetch_builtin_zEv", scope: !1186, file: !1148, line: 77, type: !1152, scopeLine: 77, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!1191 = !DISubprogram(name: "operator dim3", linkageName: "_ZNK25__cuda_builtin_blockDim_tcv4dim3Ev", scope: !1186, file: !1148, line: 80, type: !1192, scopeLine: 80, flags: DIFlagPrototyped, spFlags: 0)
!1192 = !DISubroutineType(types: !1193)
!1193 = !{!1194, !1210}
!1194 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "dim3", file: !1160, line: 417, size: 96, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !1195, identifier: "_ZTS4dim3")
!1195 = !{!1196, !1197, !1198, !1199, !1203, !1207}
!1196 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !1194, file: !1160, line: 419, baseType: !7, size: 32)
!1197 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !1194, file: !1160, line: 419, baseType: !7, size: 32, offset: 32)
!1198 = !DIDerivedType(tag: DW_TAG_member, name: "z", scope: !1194, file: !1160, line: 419, baseType: !7, size: 32, offset: 64)
!1199 = !DISubprogram(name: "dim3", scope: !1194, file: !1160, line: 421, type: !1200, scopeLine: 421, flags: DIFlagPrototyped, spFlags: 0)
!1200 = !DISubroutineType(types: !1201)
!1201 = !{null, !1202, !7, !7, !7}
!1202 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !1194, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!1203 = !DISubprogram(name: "dim3", scope: !1194, file: !1160, line: 422, type: !1204, scopeLine: 422, flags: DIFlagPrototyped, spFlags: 0)
!1204 = !DISubroutineType(types: !1205)
!1205 = !{null, !1202, !1206}
!1206 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint3", file: !1160, line: 383, baseType: !1159)
!1207 = !DISubprogram(name: "operator uint3", linkageName: "_ZN4dim3cv5uint3Ev", scope: !1194, file: !1160, line: 423, type: !1208, scopeLine: 423, flags: DIFlagPrototyped, spFlags: 0)
!1208 = !DISubroutineType(types: !1209)
!1209 = !{!1206, !1202}
!1210 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !1211, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!1211 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !1186)
!1212 = !DISubprogram(name: "__cuda_builtin_blockDim_t", scope: !1186, file: !1148, line: 82, type: !1213, scopeLine: 82, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!1213 = !DISubroutineType(types: !1214)
!1214 = !{null, !1215}
!1215 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !1186, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!1216 = !DISubprogram(name: "__cuda_builtin_blockDim_t", scope: !1186, file: !1148, line: 82, type: !1217, scopeLine: 82, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!1217 = !DISubroutineType(types: !1218)
!1218 = !{null, !1215, !1219}
!1219 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !1211, size: 64)
!1220 = !DISubprogram(name: "operator=", linkageName: "_ZNK25__cuda_builtin_blockDim_taSERKS_", scope: !1186, file: !1148, line: 82, type: !1221, scopeLine: 82, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!1221 = !DISubroutineType(types: !1222)
!1222 = !{null, !1210, !1219}
!1223 = !DISubprogram(name: "operator&", linkageName: "_ZNK25__cuda_builtin_blockDim_tadEv", scope: !1186, file: !1148, line: 82, type: !1224, scopeLine: 82, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!1224 = !DISubroutineType(types: !1225)
!1225 = !{!1226, !1210}
!1226 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !1186, size: 64)
!1227 = distinct !DILocation(line: 487, column: 16, scope: !1050)
!1228 = !{i32 1, i32 1025}
!1229 = !DILocation(line: 487, column: 15, scope: !1050)
!1230 = !DILocation(line: 53, column: 3, scope: !1231, inlinedAt: !1257)
!1231 = distinct !DISubprogram(name: "__fetch_builtin_x", linkageName: "_ZN26__cuda_builtin_threadIdx_t17__fetch_builtin_xEv", scope: !1232, file: !1148, line: 53, type: !1152, scopeLine: 53, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !961, declaration: !1234, retainedNodes: !962)
!1232 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "__cuda_builtin_threadIdx_t", file: !1148, line: 52, size: 8, flags: DIFlagTypePassByReference | DIFlagNonTrivial, elements: !1233, identifier: "_ZTS26__cuda_builtin_threadIdx_t")
!1233 = !{!1234, !1235, !1236, !1237, !1242, !1246, !1250, !1253}
!1234 = !DISubprogram(name: "__fetch_builtin_x", linkageName: "_ZN26__cuda_builtin_threadIdx_t17__fetch_builtin_xEv", scope: !1232, file: !1148, line: 53, type: !1152, scopeLine: 53, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!1235 = !DISubprogram(name: "__fetch_builtin_y", linkageName: "_ZN26__cuda_builtin_threadIdx_t17__fetch_builtin_yEv", scope: !1232, file: !1148, line: 54, type: !1152, scopeLine: 54, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!1236 = !DISubprogram(name: "__fetch_builtin_z", linkageName: "_ZN26__cuda_builtin_threadIdx_t17__fetch_builtin_zEv", scope: !1232, file: !1148, line: 55, type: !1152, scopeLine: 55, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!1237 = !DISubprogram(name: "operator uint3", linkageName: "_ZNK26__cuda_builtin_threadIdx_tcv5uint3Ev", scope: !1232, file: !1148, line: 58, type: !1238, scopeLine: 58, flags: DIFlagPrototyped, spFlags: 0)
!1238 = !DISubroutineType(types: !1239)
!1239 = !{!1159, !1240}
!1240 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !1241, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!1241 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !1232)
!1242 = !DISubprogram(name: "__cuda_builtin_threadIdx_t", scope: !1232, file: !1148, line: 60, type: !1243, scopeLine: 60, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!1243 = !DISubroutineType(types: !1244)
!1244 = !{null, !1245}
!1245 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !1232, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!1246 = !DISubprogram(name: "__cuda_builtin_threadIdx_t", scope: !1232, file: !1148, line: 60, type: !1247, scopeLine: 60, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!1247 = !DISubroutineType(types: !1248)
!1248 = !{null, !1245, !1249}
!1249 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !1241, size: 64)
!1250 = !DISubprogram(name: "operator=", linkageName: "_ZNK26__cuda_builtin_threadIdx_taSERKS_", scope: !1232, file: !1148, line: 60, type: !1251, scopeLine: 60, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!1251 = !DISubroutineType(types: !1252)
!1252 = !{null, !1240, !1249}
!1253 = !DISubprogram(name: "operator&", linkageName: "_ZNK26__cuda_builtin_threadIdx_tadEv", scope: !1232, file: !1148, line: 60, type: !1254, scopeLine: 60, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!1254 = !DISubroutineType(types: !1255)
!1255 = !{!1256, !1240}
!1256 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !1232, size: 64)
!1257 = distinct !DILocation(line: 487, column: 27, scope: !1050)
!1258 = !{i32 0, i32 1024}
!1259 = !DILocation(line: 487, column: 26, scope: !1050)
!1260 = !DILocation(line: 487, column: 4, scope: !1050)
!1261 = !DILocation(line: 489, column: 5, scope: !1262)
!1262 = distinct !DILexicalBlock(scope: !1050, file: !3, line: 489, column: 5)
!1263 = !DILocation(line: 489, column: 7, scope: !1262)
!1264 = !DILocation(line: 489, column: 5, scope: !1050)
!1265 = !DILocation(line: 489, column: 13, scope: !1266)
!1266 = distinct !DILexicalBlock(scope: !1262, file: !3, line: 489, column: 12)
!1267 = !DILocation(line: 491, column: 4, scope: !1050)
!1268 = !DILocation(line: 492, column: 5, scope: !1050)
!1269 = !DILocation(line: 492, column: 4, scope: !1050)
!1270 = !DILocation(line: 495, column: 7, scope: !1271)
!1271 = distinct !DILexicalBlock(scope: !1050, file: !3, line: 495, column: 2)
!1272 = !DILocation(line: 495, column: 6, scope: !1271)
!1273 = !DILocation(line: 495, column: 11, scope: !1274)
!1274 = distinct !DILexicalBlock(scope: !1271, file: !3, line: 495, column: 2)
!1275 = !DILocation(line: 495, column: 12, scope: !1274)
!1276 = !DILocation(line: 495, column: 2, scope: !1271)
!1277 = !DILocation(line: 496, column: 6, scope: !1278)
!1278 = distinct !DILexicalBlock(scope: !1274, file: !3, line: 495, column: 23)
!1279 = !DILocation(line: 496, column: 8, scope: !1278)
!1280 = !DILocation(line: 496, column: 5, scope: !1278)
!1281 = !DILocation(line: 497, column: 9, scope: !1282)
!1282 = distinct !DILexicalBlock(scope: !1278, file: !3, line: 497, column: 6)
!1283 = !DILocation(line: 497, column: 8, scope: !1282)
!1284 = !DILocation(line: 497, column: 14, scope: !1282)
!1285 = !DILocation(line: 497, column: 12, scope: !1282)
!1286 = !DILocation(line: 497, column: 6, scope: !1278)
!1287 = !DILocation(line: 497, column: 40, scope: !1288)
!1288 = distinct !DILexicalBlock(scope: !1282, file: !3, line: 497, column: 17)
!1289 = !DILocation(line: 497, column: 21, scope: !1288)
!1290 = !DILocation(line: 497, column: 20, scope: !1288)
!1291 = !DILocation(line: 497, column: 44, scope: !1288)
!1292 = !DILocation(line: 498, column: 6, scope: !1293)
!1293 = distinct !DILexicalBlock(scope: !1278, file: !3, line: 498, column: 6)
!1294 = !DILocation(line: 498, column: 8, scope: !1293)
!1295 = !DILocation(line: 498, column: 6, scope: !1278)
!1296 = !DILocation(line: 498, column: 13, scope: !1297)
!1297 = distinct !DILexicalBlock(scope: !1293, file: !3, line: 498, column: 12)
!1298 = !DILocation(line: 499, column: 25, scope: !1278)
!1299 = !DILocation(line: 499, column: 6, scope: !1278)
!1300 = !DILocation(line: 499, column: 5, scope: !1278)
!1301 = !DILocation(line: 500, column: 6, scope: !1278)
!1302 = !DILocation(line: 500, column: 5, scope: !1278)
!1303 = !DILocation(line: 501, column: 2, scope: !1278)
!1304 = !DILocation(line: 495, column: 20, scope: !1274)
!1305 = !DILocation(line: 495, column: 2, scope: !1274)
!1306 = distinct !{!1306, !1276, !1307}
!1307 = !DILocation(line: 501, column: 2, scope: !1271)
!1308 = !DILocation(line: 513, column: 7, scope: !1050)
!1309 = !DILocation(line: 513, column: 6, scope: !1050)
!1310 = !DILocation(line: 514, column: 8, scope: !1064)
!1311 = !DILocation(line: 514, column: 6, scope: !1064)
!1312 = !DILocation(line: 514, column: 12, scope: !1063)
!1313 = !DILocation(line: 514, column: 14, scope: !1063)
!1314 = !DILocation(line: 514, column: 2, scope: !1064)
!1315 = !DILocation(line: 516, column: 44, scope: !1062)
!1316 = !DILocation(line: 516, column: 3, scope: !1062)
!1317 = !DILocation(line: 523, column: 8, scope: !1061)
!1318 = !DILocation(line: 523, column: 7, scope: !1061)
!1319 = !DILocation(line: 523, column: 12, scope: !1060)
!1320 = !DILocation(line: 523, column: 13, scope: !1060)
!1321 = !DILocation(line: 523, column: 3, scope: !1061)
!1322 = !DILocation(line: 524, column: 21, scope: !1059)
!1323 = !DILocation(line: 524, column: 20, scope: !1059)
!1324 = !DILocation(line: 524, column: 11, scope: !1059)
!1325 = !DILocation(line: 524, column: 10, scope: !1059)
!1326 = !DILocation(line: 524, column: 23, scope: !1059)
!1327 = !DILocation(line: 524, column: 6, scope: !1059)
!1328 = !DILocation(line: 525, column: 21, scope: !1059)
!1329 = !DILocation(line: 525, column: 20, scope: !1059)
!1330 = !DILocation(line: 525, column: 22, scope: !1059)
!1331 = !DILocation(line: 525, column: 11, scope: !1059)
!1332 = !DILocation(line: 525, column: 10, scope: !1059)
!1333 = !DILocation(line: 525, column: 25, scope: !1059)
!1334 = !DILocation(line: 525, column: 6, scope: !1059)
!1335 = !DILocation(line: 526, column: 7, scope: !1059)
!1336 = !DILocation(line: 526, column: 10, scope: !1059)
!1337 = !DILocation(line: 526, column: 9, scope: !1059)
!1338 = !DILocation(line: 526, column: 13, scope: !1059)
!1339 = !DILocation(line: 526, column: 16, scope: !1059)
!1340 = !DILocation(line: 526, column: 15, scope: !1059)
!1341 = !DILocation(line: 526, column: 12, scope: !1059)
!1342 = !DILocation(line: 526, column: 6, scope: !1059)
!1343 = !DILocation(line: 527, column: 7, scope: !1058)
!1344 = !DILocation(line: 527, column: 9, scope: !1058)
!1345 = !DILocation(line: 527, column: 7, scope: !1059)
!1346 = !DILocation(line: 528, column: 23, scope: !1057)
!1347 = !DILocation(line: 227, column: 19, scope: !1076, inlinedAt: !1079)
!1348 = !DILocation(line: 227, column: 10, scope: !1076, inlinedAt: !1079)
!1349 = !DILocation(line: 528, column: 17, scope: !1057)
!1350 = !DILocation(line: 528, column: 28, scope: !1057)
!1351 = !DILocation(line: 528, column: 27, scope: !1057)
!1352 = !DILocation(line: 894, column: 20, scope: !1072, inlinedAt: !1074)
!1353 = !DILocation(line: 894, column: 10, scope: !1072, inlinedAt: !1074)
!1354 = !DILocation(line: 528, column: 7, scope: !1057)
!1355 = !DILocation(line: 529, column: 9, scope: !1057)
!1356 = !DILocation(line: 529, column: 12, scope: !1057)
!1357 = !DILocation(line: 529, column: 11, scope: !1057)
!1358 = !DILocation(line: 529, column: 7, scope: !1057)
!1359 = !DILocation(line: 530, column: 9, scope: !1057)
!1360 = !DILocation(line: 530, column: 12, scope: !1057)
!1361 = !DILocation(line: 530, column: 11, scope: !1057)
!1362 = !DILocation(line: 530, column: 7, scope: !1057)
!1363 = !DILocation(line: 531, column: 7, scope: !1057)
!1364 = !DILocation(line: 589, column: 20, scope: !1054, inlinedAt: !1070)
!1365 = !DILocation(line: 589, column: 10, scope: !1054, inlinedAt: !1070)
!1366 = !DILocation(line: 589, column: 20, scope: !1054, inlinedAt: !1068)
!1367 = !DILocation(line: 589, column: 10, scope: !1054, inlinedAt: !1068)
!1368 = !DILocation(line: 589, column: 20, scope: !1054, inlinedAt: !1066)
!1369 = !DILocation(line: 589, column: 10, scope: !1054, inlinedAt: !1066)
!1370 = !DILocation(line: 589, column: 20, scope: !1054, inlinedAt: !1056)
!1371 = !DILocation(line: 589, column: 10, scope: !1054, inlinedAt: !1056)
!1372 = !DILocation(line: 531, column: 6, scope: !1057)
!1373 = !DILocation(line: 532, column: 13, scope: !1057)
!1374 = !DILocation(line: 532, column: 5, scope: !1057)
!1375 = !DILocation(line: 532, column: 15, scope: !1057)
!1376 = !DILocation(line: 533, column: 14, scope: !1057)
!1377 = !DILocation(line: 533, column: 23, scope: !1057)
!1378 = !DILocation(line: 533, column: 22, scope: !1057)
!1379 = !DILocation(line: 533, column: 13, scope: !1057)
!1380 = !DILocation(line: 534, column: 15, scope: !1057)
!1381 = !DILocation(line: 534, column: 13, scope: !1057)
!1382 = !DILocation(line: 535, column: 4, scope: !1057)
!1383 = !DILocation(line: 536, column: 3, scope: !1059)
!1384 = !DILocation(line: 523, column: 30, scope: !1060)
!1385 = !DILocation(line: 523, column: 3, scope: !1060)
!1386 = distinct !{!1386, !1321, !1387}
!1387 = !DILocation(line: 536, column: 3, scope: !1061)
!1388 = !DILocation(line: 537, column: 2, scope: !1062)
!1389 = !DILocation(line: 514, column: 22, scope: !1063)
!1390 = !DILocation(line: 514, column: 24, scope: !1063)
!1391 = !DILocation(line: 514, column: 21, scope: !1063)
!1392 = !DILocation(line: 514, column: 2, scope: !1063)
!1393 = distinct !{!1393, !1314, !1394}
!1394 = !DILocation(line: 537, column: 2, scope: !1064)
!1395 = !DILocation(line: 539, column: 12, scope: !1050)
!1396 = !DILocation(line: 64, column: 3, scope: !1147, inlinedAt: !1397)
!1397 = distinct !DILocation(line: 539, column: 21, scope: !1050)
!1398 = !DILocation(line: 539, column: 31, scope: !1050)
!1399 = !DILocation(line: 539, column: 20, scope: !1050)
!1400 = !DILocation(line: 539, column: 34, scope: !1050)
!1401 = !DILocation(line: 539, column: 38, scope: !1050)
!1402 = !DILocation(line: 539, column: 2, scope: !1050)
!1403 = !DILocation(line: 540, column: 12, scope: !1050)
!1404 = !DILocation(line: 64, column: 3, scope: !1147, inlinedAt: !1405)
!1405 = distinct !DILocation(line: 540, column: 21, scope: !1050)
!1406 = !DILocation(line: 540, column: 31, scope: !1050)
!1407 = !DILocation(line: 540, column: 20, scope: !1050)
!1408 = !DILocation(line: 540, column: 34, scope: !1050)
!1409 = !DILocation(line: 540, column: 38, scope: !1050)
!1410 = !DILocation(line: 540, column: 2, scope: !1050)
!1411 = !DILocation(line: 541, column: 12, scope: !1050)
!1412 = !DILocation(line: 64, column: 3, scope: !1147, inlinedAt: !1413)
!1413 = distinct !DILocation(line: 541, column: 21, scope: !1050)
!1414 = !DILocation(line: 541, column: 31, scope: !1050)
!1415 = !DILocation(line: 541, column: 20, scope: !1050)
!1416 = !DILocation(line: 541, column: 34, scope: !1050)
!1417 = !DILocation(line: 541, column: 38, scope: !1050)
!1418 = !DILocation(line: 541, column: 2, scope: !1050)
!1419 = !DILocation(line: 542, column: 12, scope: !1050)
!1420 = !DILocation(line: 64, column: 3, scope: !1147, inlinedAt: !1421)
!1421 = distinct !DILocation(line: 542, column: 21, scope: !1050)
!1422 = !DILocation(line: 542, column: 31, scope: !1050)
!1423 = !DILocation(line: 542, column: 20, scope: !1050)
!1424 = !DILocation(line: 542, column: 34, scope: !1050)
!1425 = !DILocation(line: 542, column: 38, scope: !1050)
!1426 = !DILocation(line: 542, column: 2, scope: !1050)
!1427 = !DILocation(line: 543, column: 12, scope: !1050)
!1428 = !DILocation(line: 64, column: 3, scope: !1147, inlinedAt: !1429)
!1429 = distinct !DILocation(line: 543, column: 21, scope: !1050)
!1430 = !DILocation(line: 543, column: 31, scope: !1050)
!1431 = !DILocation(line: 543, column: 20, scope: !1050)
!1432 = !DILocation(line: 543, column: 34, scope: !1050)
!1433 = !DILocation(line: 543, column: 38, scope: !1050)
!1434 = !DILocation(line: 543, column: 2, scope: !1050)
!1435 = !DILocation(line: 544, column: 12, scope: !1050)
!1436 = !DILocation(line: 64, column: 3, scope: !1147, inlinedAt: !1437)
!1437 = distinct !DILocation(line: 544, column: 21, scope: !1050)
!1438 = !DILocation(line: 544, column: 31, scope: !1050)
!1439 = !DILocation(line: 544, column: 20, scope: !1050)
!1440 = !DILocation(line: 544, column: 34, scope: !1050)
!1441 = !DILocation(line: 544, column: 38, scope: !1050)
!1442 = !DILocation(line: 544, column: 2, scope: !1050)
!1443 = !DILocation(line: 545, column: 12, scope: !1050)
!1444 = !DILocation(line: 64, column: 3, scope: !1147, inlinedAt: !1445)
!1445 = distinct !DILocation(line: 545, column: 21, scope: !1050)
!1446 = !DILocation(line: 545, column: 31, scope: !1050)
!1447 = !DILocation(line: 545, column: 20, scope: !1050)
!1448 = !DILocation(line: 545, column: 34, scope: !1050)
!1449 = !DILocation(line: 545, column: 38, scope: !1050)
!1450 = !DILocation(line: 545, column: 2, scope: !1050)
!1451 = !DILocation(line: 546, column: 12, scope: !1050)
!1452 = !DILocation(line: 64, column: 3, scope: !1147, inlinedAt: !1453)
!1453 = distinct !DILocation(line: 546, column: 21, scope: !1050)
!1454 = !DILocation(line: 546, column: 31, scope: !1050)
!1455 = !DILocation(line: 546, column: 20, scope: !1050)
!1456 = !DILocation(line: 546, column: 34, scope: !1050)
!1457 = !DILocation(line: 546, column: 38, scope: !1050)
!1458 = !DILocation(line: 546, column: 2, scope: !1050)
!1459 = !DILocation(line: 547, column: 12, scope: !1050)
!1460 = !DILocation(line: 64, column: 3, scope: !1147, inlinedAt: !1461)
!1461 = distinct !DILocation(line: 547, column: 21, scope: !1050)
!1462 = !DILocation(line: 547, column: 31, scope: !1050)
!1463 = !DILocation(line: 547, column: 20, scope: !1050)
!1464 = !DILocation(line: 547, column: 34, scope: !1050)
!1465 = !DILocation(line: 547, column: 38, scope: !1050)
!1466 = !DILocation(line: 547, column: 2, scope: !1050)
!1467 = !DILocation(line: 548, column: 12, scope: !1050)
!1468 = !DILocation(line: 64, column: 3, scope: !1147, inlinedAt: !1469)
!1469 = distinct !DILocation(line: 548, column: 21, scope: !1050)
!1470 = !DILocation(line: 548, column: 31, scope: !1050)
!1471 = !DILocation(line: 548, column: 20, scope: !1050)
!1472 = !DILocation(line: 548, column: 34, scope: !1050)
!1473 = !DILocation(line: 548, column: 38, scope: !1050)
!1474 = !DILocation(line: 548, column: 2, scope: !1050)
!1475 = !DILocation(line: 549, column: 12, scope: !1050)
!1476 = !DILocation(line: 64, column: 3, scope: !1147, inlinedAt: !1477)
!1477 = distinct !DILocation(line: 549, column: 22, scope: !1050)
!1478 = !DILocation(line: 549, column: 21, scope: !1050)
!1479 = !DILocation(line: 549, column: 34, scope: !1050)
!1480 = !DILocation(line: 549, column: 2, scope: !1050)
!1481 = !DILocation(line: 550, column: 12, scope: !1050)
!1482 = !DILocation(line: 64, column: 3, scope: !1147, inlinedAt: !1483)
!1483 = distinct !DILocation(line: 550, column: 22, scope: !1050)
!1484 = !DILocation(line: 550, column: 21, scope: !1050)
!1485 = !DILocation(line: 550, column: 34, scope: !1050)
!1486 = !DILocation(line: 550, column: 2, scope: !1050)
!1487 = !DILocation(line: 551, column: 1, scope: !1050)
!1488 = distinct !DISubprogram(name: "randlc_device", linkageName: "_Z13randlc_devicePdd", scope: !3, file: !3, line: 553, type: !1489, scopeLine: 554, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !961, retainedNodes: !962)
!1489 = !DISubroutineType(types: !1490)
!1490 = !{!99, !98, !99}
!1491 = !DILocalVariable(name: "x", arg: 1, scope: !1488, file: !3, line: 553, type: !98)
!1492 = !DILocation(line: 553, column: 41, scope: !1488)
!1493 = !DILocalVariable(name: "a", arg: 2, scope: !1488, file: !3, line: 554, type: !99)
!1494 = !DILocation(line: 554, column: 10, scope: !1488)
!1495 = !DILocalVariable(name: "t1", scope: !1488, file: !3, line: 555, type: !99)
!1496 = !DILocation(line: 555, column: 9, scope: !1488)
!1497 = !DILocalVariable(name: "t2", scope: !1488, file: !3, line: 555, type: !99)
!1498 = !DILocation(line: 555, column: 12, scope: !1488)
!1499 = !DILocalVariable(name: "t3", scope: !1488, file: !3, line: 555, type: !99)
!1500 = !DILocation(line: 555, column: 15, scope: !1488)
!1501 = !DILocalVariable(name: "t4", scope: !1488, file: !3, line: 555, type: !99)
!1502 = !DILocation(line: 555, column: 18, scope: !1488)
!1503 = !DILocalVariable(name: "a1", scope: !1488, file: !3, line: 555, type: !99)
!1504 = !DILocation(line: 555, column: 21, scope: !1488)
!1505 = !DILocalVariable(name: "a2", scope: !1488, file: !3, line: 555, type: !99)
!1506 = !DILocation(line: 555, column: 24, scope: !1488)
!1507 = !DILocalVariable(name: "x1", scope: !1488, file: !3, line: 555, type: !99)
!1508 = !DILocation(line: 555, column: 27, scope: !1488)
!1509 = !DILocalVariable(name: "x2", scope: !1488, file: !3, line: 555, type: !99)
!1510 = !DILocation(line: 555, column: 30, scope: !1488)
!1511 = !DILocalVariable(name: "z", scope: !1488, file: !3, line: 555, type: !99)
!1512 = !DILocation(line: 555, column: 33, scope: !1488)
!1513 = !DILocation(line: 556, column: 13, scope: !1488)
!1514 = !DILocation(line: 556, column: 11, scope: !1488)
!1515 = !DILocation(line: 556, column: 5, scope: !1488)
!1516 = !DILocation(line: 557, column: 12, scope: !1488)
!1517 = !DILocation(line: 557, column: 7, scope: !1488)
!1518 = !DILocation(line: 557, column: 5, scope: !1488)
!1519 = !DILocation(line: 558, column: 7, scope: !1488)
!1520 = !DILocation(line: 558, column: 17, scope: !1488)
!1521 = !DILocation(line: 558, column: 15, scope: !1488)
!1522 = !DILocation(line: 558, column: 9, scope: !1488)
!1523 = !DILocation(line: 558, column: 5, scope: !1488)
!1524 = !DILocation(line: 559, column: 15, scope: !1488)
!1525 = !DILocation(line: 559, column: 14, scope: !1488)
!1526 = !DILocation(line: 559, column: 11, scope: !1488)
!1527 = !DILocation(line: 559, column: 5, scope: !1488)
!1528 = !DILocation(line: 560, column: 12, scope: !1488)
!1529 = !DILocation(line: 560, column: 7, scope: !1488)
!1530 = !DILocation(line: 560, column: 5, scope: !1488)
!1531 = !DILocation(line: 561, column: 9, scope: !1488)
!1532 = !DILocation(line: 561, column: 8, scope: !1488)
!1533 = !DILocation(line: 561, column: 20, scope: !1488)
!1534 = !DILocation(line: 561, column: 18, scope: !1488)
!1535 = !DILocation(line: 561, column: 12, scope: !1488)
!1536 = !DILocation(line: 561, column: 5, scope: !1488)
!1537 = !DILocation(line: 562, column: 7, scope: !1488)
!1538 = !DILocation(line: 562, column: 12, scope: !1488)
!1539 = !DILocation(line: 562, column: 10, scope: !1488)
!1540 = !DILocation(line: 562, column: 17, scope: !1488)
!1541 = !DILocation(line: 562, column: 22, scope: !1488)
!1542 = !DILocation(line: 562, column: 20, scope: !1488)
!1543 = !DILocation(line: 562, column: 15, scope: !1488)
!1544 = !DILocation(line: 562, column: 5, scope: !1488)
!1545 = !DILocation(line: 563, column: 19, scope: !1488)
!1546 = !DILocation(line: 563, column: 17, scope: !1488)
!1547 = !DILocation(line: 563, column: 12, scope: !1488)
!1548 = !DILocation(line: 563, column: 7, scope: !1488)
!1549 = !DILocation(line: 563, column: 5, scope: !1488)
!1550 = !DILocation(line: 564, column: 6, scope: !1488)
!1551 = !DILocation(line: 564, column: 17, scope: !1488)
!1552 = !DILocation(line: 564, column: 15, scope: !1488)
!1553 = !DILocation(line: 564, column: 9, scope: !1488)
!1554 = !DILocation(line: 564, column: 4, scope: !1488)
!1555 = !DILocation(line: 565, column: 13, scope: !1488)
!1556 = !DILocation(line: 565, column: 11, scope: !1488)
!1557 = !DILocation(line: 565, column: 17, scope: !1488)
!1558 = !DILocation(line: 565, column: 22, scope: !1488)
!1559 = !DILocation(line: 565, column: 20, scope: !1488)
!1560 = !DILocation(line: 565, column: 15, scope: !1488)
!1561 = !DILocation(line: 565, column: 5, scope: !1488)
!1562 = !DILocation(line: 566, column: 19, scope: !1488)
!1563 = !DILocation(line: 566, column: 17, scope: !1488)
!1564 = !DILocation(line: 566, column: 12, scope: !1488)
!1565 = !DILocation(line: 566, column: 7, scope: !1488)
!1566 = !DILocation(line: 566, column: 5, scope: !1488)
!1567 = !DILocation(line: 567, column: 9, scope: !1488)
!1568 = !DILocation(line: 567, column: 20, scope: !1488)
!1569 = !DILocation(line: 567, column: 18, scope: !1488)
!1570 = !DILocation(line: 567, column: 12, scope: !1488)
!1571 = !DILocation(line: 567, column: 4, scope: !1488)
!1572 = !DILocation(line: 567, column: 7, scope: !1488)
!1573 = !DILocation(line: 568, column: 18, scope: !1488)
!1574 = !DILocation(line: 568, column: 17, scope: !1488)
!1575 = !DILocation(line: 568, column: 14, scope: !1488)
!1576 = !DILocation(line: 568, column: 2, scope: !1488)
!1577 = distinct !DISubprogram(name: "vranlc_device", linkageName: "_Z13vranlc_deviceiPddS_", scope: !3, file: !3, line: 649, type: !1578, scopeLine: 652, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !961, retainedNodes: !962)
!1578 = !DISubroutineType(types: !1579)
!1579 = !{null, !97, !98, !99, !98}
!1580 = !DILocalVariable(name: "n", arg: 1, scope: !1577, file: !3, line: 649, type: !97)
!1581 = !DILocation(line: 649, column: 35, scope: !1577)
!1582 = !DILocalVariable(name: "x_seed", arg: 2, scope: !1577, file: !3, line: 650, type: !98)
!1583 = !DILocation(line: 650, column: 11, scope: !1577)
!1584 = !DILocalVariable(name: "a", arg: 3, scope: !1577, file: !3, line: 651, type: !99)
!1585 = !DILocation(line: 651, column: 10, scope: !1577)
!1586 = !DILocalVariable(name: "y", arg: 4, scope: !1577, file: !3, line: 652, type: !98)
!1587 = !DILocation(line: 652, column: 11, scope: !1577)
!1588 = !DILocalVariable(name: "i", scope: !1577, file: !3, line: 653, type: !97)
!1589 = !DILocation(line: 653, column: 6, scope: !1577)
!1590 = !DILocalVariable(name: "x", scope: !1577, file: !3, line: 654, type: !99)
!1591 = !DILocation(line: 654, column: 9, scope: !1577)
!1592 = !DILocalVariable(name: "t1", scope: !1577, file: !3, line: 654, type: !99)
!1593 = !DILocation(line: 654, column: 11, scope: !1577)
!1594 = !DILocalVariable(name: "t2", scope: !1577, file: !3, line: 654, type: !99)
!1595 = !DILocation(line: 654, column: 14, scope: !1577)
!1596 = !DILocalVariable(name: "t3", scope: !1577, file: !3, line: 654, type: !99)
!1597 = !DILocation(line: 654, column: 17, scope: !1577)
!1598 = !DILocalVariable(name: "t4", scope: !1577, file: !3, line: 654, type: !99)
!1599 = !DILocation(line: 654, column: 20, scope: !1577)
!1600 = !DILocalVariable(name: "a1", scope: !1577, file: !3, line: 654, type: !99)
!1601 = !DILocation(line: 654, column: 23, scope: !1577)
!1602 = !DILocalVariable(name: "a2", scope: !1577, file: !3, line: 654, type: !99)
!1603 = !DILocation(line: 654, column: 26, scope: !1577)
!1604 = !DILocalVariable(name: "x1", scope: !1577, file: !3, line: 654, type: !99)
!1605 = !DILocation(line: 654, column: 29, scope: !1577)
!1606 = !DILocalVariable(name: "x2", scope: !1577, file: !3, line: 654, type: !99)
!1607 = !DILocation(line: 654, column: 32, scope: !1577)
!1608 = !DILocalVariable(name: "z", scope: !1577, file: !3, line: 654, type: !99)
!1609 = !DILocation(line: 654, column: 35, scope: !1577)
!1610 = !DILocation(line: 655, column: 13, scope: !1577)
!1611 = !DILocation(line: 655, column: 11, scope: !1577)
!1612 = !DILocation(line: 655, column: 5, scope: !1577)
!1613 = !DILocation(line: 656, column: 12, scope: !1577)
!1614 = !DILocation(line: 656, column: 7, scope: !1577)
!1615 = !DILocation(line: 656, column: 5, scope: !1577)
!1616 = !DILocation(line: 657, column: 7, scope: !1577)
!1617 = !DILocation(line: 657, column: 17, scope: !1577)
!1618 = !DILocation(line: 657, column: 15, scope: !1577)
!1619 = !DILocation(line: 657, column: 9, scope: !1577)
!1620 = !DILocation(line: 657, column: 5, scope: !1577)
!1621 = !DILocation(line: 658, column: 7, scope: !1577)
!1622 = !DILocation(line: 658, column: 6, scope: !1577)
!1623 = !DILocation(line: 658, column: 4, scope: !1577)
!1624 = !DILocation(line: 659, column: 7, scope: !1625)
!1625 = distinct !DILexicalBlock(scope: !1577, file: !3, line: 659, column: 2)
!1626 = !DILocation(line: 659, column: 6, scope: !1625)
!1627 = !DILocation(line: 659, column: 11, scope: !1628)
!1628 = distinct !DILexicalBlock(scope: !1625, file: !3, line: 659, column: 2)
!1629 = !DILocation(line: 659, column: 13, scope: !1628)
!1630 = !DILocation(line: 659, column: 12, scope: !1628)
!1631 = !DILocation(line: 659, column: 2, scope: !1625)
!1632 = !DILocation(line: 660, column: 14, scope: !1633)
!1633 = distinct !DILexicalBlock(scope: !1628, file: !3, line: 659, column: 20)
!1634 = !DILocation(line: 660, column: 12, scope: !1633)
!1635 = !DILocation(line: 660, column: 6, scope: !1633)
!1636 = !DILocation(line: 661, column: 13, scope: !1633)
!1637 = !DILocation(line: 661, column: 8, scope: !1633)
!1638 = !DILocation(line: 661, column: 6, scope: !1633)
!1639 = !DILocation(line: 662, column: 8, scope: !1633)
!1640 = !DILocation(line: 662, column: 18, scope: !1633)
!1641 = !DILocation(line: 662, column: 16, scope: !1633)
!1642 = !DILocation(line: 662, column: 10, scope: !1633)
!1643 = !DILocation(line: 662, column: 6, scope: !1633)
!1644 = !DILocation(line: 663, column: 8, scope: !1633)
!1645 = !DILocation(line: 663, column: 13, scope: !1633)
!1646 = !DILocation(line: 663, column: 11, scope: !1633)
!1647 = !DILocation(line: 663, column: 18, scope: !1633)
!1648 = !DILocation(line: 663, column: 23, scope: !1633)
!1649 = !DILocation(line: 663, column: 21, scope: !1633)
!1650 = !DILocation(line: 663, column: 16, scope: !1633)
!1651 = !DILocation(line: 663, column: 6, scope: !1633)
!1652 = !DILocation(line: 664, column: 20, scope: !1633)
!1653 = !DILocation(line: 664, column: 18, scope: !1633)
!1654 = !DILocation(line: 664, column: 13, scope: !1633)
!1655 = !DILocation(line: 664, column: 8, scope: !1633)
!1656 = !DILocation(line: 664, column: 6, scope: !1633)
!1657 = !DILocation(line: 665, column: 7, scope: !1633)
!1658 = !DILocation(line: 665, column: 18, scope: !1633)
!1659 = !DILocation(line: 665, column: 16, scope: !1633)
!1660 = !DILocation(line: 665, column: 10, scope: !1633)
!1661 = !DILocation(line: 665, column: 5, scope: !1633)
!1662 = !DILocation(line: 666, column: 14, scope: !1633)
!1663 = !DILocation(line: 666, column: 12, scope: !1633)
!1664 = !DILocation(line: 666, column: 18, scope: !1633)
!1665 = !DILocation(line: 666, column: 23, scope: !1633)
!1666 = !DILocation(line: 666, column: 21, scope: !1633)
!1667 = !DILocation(line: 666, column: 16, scope: !1633)
!1668 = !DILocation(line: 666, column: 6, scope: !1633)
!1669 = !DILocation(line: 667, column: 20, scope: !1633)
!1670 = !DILocation(line: 667, column: 18, scope: !1633)
!1671 = !DILocation(line: 667, column: 13, scope: !1633)
!1672 = !DILocation(line: 667, column: 8, scope: !1633)
!1673 = !DILocation(line: 667, column: 6, scope: !1633)
!1674 = !DILocation(line: 668, column: 7, scope: !1633)
!1675 = !DILocation(line: 668, column: 18, scope: !1633)
!1676 = !DILocation(line: 668, column: 16, scope: !1633)
!1677 = !DILocation(line: 668, column: 10, scope: !1633)
!1678 = !DILocation(line: 668, column: 5, scope: !1633)
!1679 = !DILocation(line: 669, column: 16, scope: !1633)
!1680 = !DILocation(line: 669, column: 14, scope: !1633)
!1681 = !DILocation(line: 669, column: 3, scope: !1633)
!1682 = !DILocation(line: 669, column: 5, scope: !1633)
!1683 = !DILocation(line: 669, column: 8, scope: !1633)
!1684 = !DILocation(line: 670, column: 2, scope: !1633)
!1685 = !DILocation(line: 659, column: 17, scope: !1628)
!1686 = !DILocation(line: 659, column: 2, scope: !1628)
!1687 = distinct !{!1687, !1631, !1688}
!1688 = !DILocation(line: 670, column: 2, scope: !1625)
!1689 = !DILocation(line: 671, column: 12, scope: !1577)
!1690 = !DILocation(line: 671, column: 3, scope: !1577)
!1691 = !DILocation(line: 671, column: 10, scope: !1577)
!1692 = !DILocation(line: 672, column: 1, scope: !1577)
!1693 = distinct !DISubprogram(name: "atomicAdd", linkageName: "_ZL9atomicAddPdd", scope: !1694, file: !1694, line: 54, type: !1489, scopeLine: 54, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !961, retainedNodes: !962)
!1694 = !DIFile(filename: "./../common/npb-CPP.hpp", directory: "/scratch/ah7226/NPB-GPU/CUDA/EP")
!1695 = !DILocalVariable(name: "x", arg: 1, scope: !1696, file: !691, line: 1370, type: !99)
!1696 = distinct !DISubprogram(name: "__double_as_longlong", linkageName: "_ZL20__double_as_longlongd", scope: !691, file: !691, line: 1370, type: !1697, scopeLine: 1371, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !961, retainedNodes: !962)
!1697 = !DISubroutineType(types: !1698)
!1698 = !{!324, !99}
!1699 = !DILocation(line: 1370, column: 74, scope: !1696, inlinedAt: !1700)
!1700 = distinct !DILocation(line: 61, column: 44, scope: !1701)
!1701 = distinct !DILexicalBlock(scope: !1702, file: !1694, line: 59, column: 35)
!1702 = distinct !DILexicalBlock(scope: !1703, file: !1694, line: 59, column: 2)
!1703 = distinct !DILexicalBlock(scope: !1693, file: !1694, line: 59, column: 2)
!1704 = !DILocalVariable(name: "x", arg: 1, scope: !1705, file: !691, line: 1365, type: !324)
!1705 = distinct !DISubprogram(name: "__longlong_as_double", linkageName: "_ZL20__longlong_as_doublex", scope: !691, file: !691, line: 1365, type: !1706, scopeLine: 1366, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !961, retainedNodes: !962)
!1706 = !DISubroutineType(types: !1707)
!1707 = !{!99, !324}
!1708 = !DILocation(line: 1365, column: 72, scope: !1705, inlinedAt: !1709)
!1709 = distinct !DILocation(line: 61, column: 70, scope: !1701)
!1710 = !DILocation(line: 1365, column: 72, scope: !1705, inlinedAt: !1711)
!1711 = distinct !DILocation(line: 62, column: 36, scope: !1712)
!1712 = distinct !DILexicalBlock(scope: !1701, file: !1694, line: 62, column: 13)
!1713 = !DILocation(line: 1365, column: 72, scope: !1705, inlinedAt: !1714)
!1714 = distinct !DILocation(line: 64, column: 9, scope: !1693)
!1715 = !DILocation(line: 1365, column: 72, scope: !1705, inlinedAt: !1716)
!1716 = distinct !DILocation(line: 58, column: 10, scope: !1717)
!1717 = distinct !DILexicalBlock(scope: !1693, file: !1694, line: 57, column: 6)
!1718 = !DILocalVariable(name: "address", arg: 1, scope: !1693, file: !1694, line: 54, type: !98)
!1719 = !DILocation(line: 54, column: 55, scope: !1693)
!1720 = !DILocalVariable(name: "val", arg: 2, scope: !1693, file: !1694, line: 54, type: !99)
!1721 = !DILocation(line: 54, column: 71, scope: !1693)
!1722 = !DILocalVariable(name: "address_as_ull", scope: !1693, file: !1694, line: 55, type: !964)
!1723 = !DILocation(line: 55, column: 26, scope: !1693)
!1724 = !DILocation(line: 55, column: 68, scope: !1693)
!1725 = !DILocation(line: 55, column: 43, scope: !1693)
!1726 = !DILocalVariable(name: "old", scope: !1693, file: !1694, line: 56, type: !616)
!1727 = !DILocation(line: 56, column: 25, scope: !1693)
!1728 = !DILocation(line: 56, column: 32, scope: !1693)
!1729 = !DILocation(line: 56, column: 31, scope: !1693)
!1730 = !DILocalVariable(name: "assumed", scope: !1693, file: !1694, line: 56, type: !616)
!1731 = !DILocation(line: 56, column: 48, scope: !1693)
!1732 = !DILocation(line: 57, column: 6, scope: !1717)
!1733 = !DILocation(line: 57, column: 9, scope: !1717)
!1734 = !DILocation(line: 57, column: 6, scope: !1693)
!1735 = !DILocation(line: 58, column: 31, scope: !1717)
!1736 = !DILocation(line: 1367, column: 34, scope: !1705, inlinedAt: !1716)
!1737 = !DILocation(line: 1367, column: 10, scope: !1705, inlinedAt: !1716)
!1738 = !DILocation(line: 58, column: 3, scope: !1717)
!1739 = !DILocalVariable(name: "i", scope: !1703, file: !1694, line: 59, type: !97)
!1740 = !DILocation(line: 59, column: 11, scope: !1703)
!1741 = !DILocation(line: 59, column: 7, scope: !1703)
!1742 = !DILocation(line: 59, column: 18, scope: !1702)
!1743 = !DILocation(line: 59, column: 20, scope: !1702)
!1744 = !DILocation(line: 59, column: 2, scope: !1703)
!1745 = !DILocation(line: 60, column: 13, scope: !1701)
!1746 = !DILocation(line: 60, column: 11, scope: !1701)
!1747 = !DILocation(line: 61, column: 19, scope: !1701)
!1748 = !DILocation(line: 61, column: 35, scope: !1701)
!1749 = !DILocation(line: 61, column: 65, scope: !1701)
!1750 = !DILocation(line: 61, column: 91, scope: !1701)
!1751 = !DILocation(line: 1367, column: 34, scope: !1705, inlinedAt: !1709)
!1752 = !DILocation(line: 1367, column: 10, scope: !1705, inlinedAt: !1709)
!1753 = !DILocation(line: 61, column: 69, scope: !1701)
!1754 = !DILocation(line: 1372, column: 34, scope: !1696, inlinedAt: !1700)
!1755 = !DILocation(line: 1372, column: 10, scope: !1696, inlinedAt: !1700)
!1756 = !DILocation(line: 61, column: 9, scope: !1701)
!1757 = !DILocation(line: 61, column: 7, scope: !1701)
!1758 = !DILocation(line: 62, column: 13, scope: !1712)
!1759 = !DILocation(line: 62, column: 24, scope: !1712)
!1760 = !DILocation(line: 62, column: 21, scope: !1712)
!1761 = !DILocation(line: 62, column: 13, scope: !1701)
!1762 = !DILocation(line: 62, column: 57, scope: !1712)
!1763 = !DILocation(line: 1367, column: 34, scope: !1705, inlinedAt: !1711)
!1764 = !DILocation(line: 1367, column: 10, scope: !1705, inlinedAt: !1711)
!1765 = !DILocation(line: 62, column: 29, scope: !1712)
!1766 = !DILocation(line: 63, column: 2, scope: !1701)
!1767 = !DILocation(line: 59, column: 31, scope: !1702)
!1768 = !DILocation(line: 59, column: 2, scope: !1702)
!1769 = distinct !{!1769, !1744, !1770}
!1770 = !DILocation(line: 63, column: 2, scope: !1703)
!1771 = !DILocation(line: 64, column: 30, scope: !1693)
!1772 = !DILocation(line: 1367, column: 34, scope: !1705, inlinedAt: !1714)
!1773 = !DILocation(line: 1367, column: 10, scope: !1705, inlinedAt: !1714)
!1774 = !DILocation(line: 64, column: 2, scope: !1693)
!1775 = !DILocation(line: 65, column: 1, scope: !1693)
!1776 = distinct !DISubprogram(name: "atomicCAS", linkageName: "_ZL9atomicCASPyyy", scope: !1777, file: !1777, line: 211, type: !1778, scopeLine: 212, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !961, retainedNodes: !962)
!1777 = !DIFile(filename: "/usr/local/cuda/include/device_atomic_functions.hpp", directory: "")
!1778 = !DISubroutineType(types: !1779)
!1779 = !{!616, !964, !616, !616}
!1780 = !DILocalVariable(name: "p", arg: 1, scope: !1781, file: !691, line: 1655, type: !964)
!1781 = distinct !DISubprogram(name: "__ullAtomicCAS", linkageName: "_ZL14__ullAtomicCASPyyy", scope: !691, file: !691, line: 1655, type: !1778, scopeLine: 1658, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !961, retainedNodes: !962)
!1782 = !DILocation(line: 1655, column: 63, scope: !1781, inlinedAt: !1783)
!1783 = distinct !DILocation(line: 213, column: 10, scope: !1776)
!1784 = !DILocalVariable(name: "compare", arg: 2, scope: !1781, file: !691, line: 1656, type: !616)
!1785 = !DILocation(line: 1656, column: 62, scope: !1781, inlinedAt: !1783)
!1786 = !DILocalVariable(name: "val", arg: 3, scope: !1781, file: !691, line: 1657, type: !616)
!1787 = !DILocation(line: 1657, column: 62, scope: !1781, inlinedAt: !1783)
!1788 = !DILocalVariable(name: "address", arg: 1, scope: !1776, file: !1777, line: 211, type: !964)
!1789 = !DILocation(line: 211, column: 91, scope: !1776)
!1790 = !DILocalVariable(name: "compare", arg: 2, scope: !1776, file: !1777, line: 211, type: !616)
!1791 = !DILocation(line: 211, column: 123, scope: !1776)
!1792 = !DILocalVariable(name: "val", arg: 3, scope: !1776, file: !1777, line: 211, type: !616)
!1793 = !DILocation(line: 211, column: 155, scope: !1776)
!1794 = !DILocation(line: 213, column: 25, scope: !1776)
!1795 = !DILocation(line: 213, column: 34, scope: !1776)
!1796 = !DILocation(line: 213, column: 43, scope: !1776)
!1797 = !DILocation(line: 1660, column: 78, scope: !1781, inlinedAt: !1783)
!1798 = !DILocation(line: 1661, column: 67, scope: !1781, inlinedAt: !1783)
!1799 = !DILocation(line: 1662, column: 67, scope: !1781, inlinedAt: !1783)
!1800 = !DILocation(line: 1660, column: 29, scope: !1781, inlinedAt: !1783)
!1801 = !DILocation(line: 213, column: 3, scope: !1776)
!1802 = distinct !DISubprogram(name: "randlc", linkageName: "_Z6randlcPdd", scope: !3, file: !3, line: 156, type: !1489, scopeLine: 156, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !962)
!1803 = !DILocalVariable(name: "x", arg: 1, scope: !1802, file: !3, line: 156, type: !98)
!1804 = !DILocation(line: 156, column: 23, scope: !1802)
!1805 = !DILocalVariable(name: "a", arg: 2, scope: !1802, file: !3, line: 156, type: !99)
!1806 = !DILocation(line: 156, column: 33, scope: !1802)
!1807 = !DILocalVariable(name: "t1", scope: !1802, file: !3, line: 157, type: !99)
!1808 = !DILocation(line: 157, column: 9, scope: !1802)
!1809 = !DILocalVariable(name: "t2", scope: !1802, file: !3, line: 157, type: !99)
!1810 = !DILocation(line: 157, column: 12, scope: !1802)
!1811 = !DILocalVariable(name: "t3", scope: !1802, file: !3, line: 157, type: !99)
!1812 = !DILocation(line: 157, column: 15, scope: !1802)
!1813 = !DILocalVariable(name: "t4", scope: !1802, file: !3, line: 157, type: !99)
!1814 = !DILocation(line: 157, column: 18, scope: !1802)
!1815 = !DILocalVariable(name: "a1", scope: !1802, file: !3, line: 157, type: !99)
!1816 = !DILocation(line: 157, column: 21, scope: !1802)
!1817 = !DILocalVariable(name: "a2", scope: !1802, file: !3, line: 157, type: !99)
!1818 = !DILocation(line: 157, column: 24, scope: !1802)
!1819 = !DILocalVariable(name: "x1", scope: !1802, file: !3, line: 157, type: !99)
!1820 = !DILocation(line: 157, column: 27, scope: !1802)
!1821 = !DILocalVariable(name: "x2", scope: !1802, file: !3, line: 157, type: !99)
!1822 = !DILocation(line: 157, column: 30, scope: !1802)
!1823 = !DILocalVariable(name: "z", scope: !1802, file: !3, line: 157, type: !99)
!1824 = !DILocation(line: 157, column: 33, scope: !1802)
!1825 = !DILocation(line: 164, column: 13, scope: !1802)
!1826 = !DILocation(line: 164, column: 11, scope: !1802)
!1827 = !DILocation(line: 164, column: 5, scope: !1802)
!1828 = !DILocation(line: 165, column: 12, scope: !1802)
!1829 = !DILocation(line: 165, column: 7, scope: !1802)
!1830 = !DILocation(line: 165, column: 5, scope: !1802)
!1831 = !DILocation(line: 166, column: 7, scope: !1802)
!1832 = !DILocation(line: 166, column: 17, scope: !1802)
!1833 = !DILocation(line: 166, column: 15, scope: !1802)
!1834 = !DILocation(line: 166, column: 9, scope: !1802)
!1835 = !DILocation(line: 166, column: 5, scope: !1802)
!1836 = !DILocation(line: 175, column: 15, scope: !1802)
!1837 = !DILocation(line: 175, column: 14, scope: !1802)
!1838 = !DILocation(line: 175, column: 11, scope: !1802)
!1839 = !DILocation(line: 175, column: 5, scope: !1802)
!1840 = !DILocation(line: 176, column: 12, scope: !1802)
!1841 = !DILocation(line: 176, column: 7, scope: !1802)
!1842 = !DILocation(line: 176, column: 5, scope: !1802)
!1843 = !DILocation(line: 177, column: 9, scope: !1802)
!1844 = !DILocation(line: 177, column: 8, scope: !1802)
!1845 = !DILocation(line: 177, column: 20, scope: !1802)
!1846 = !DILocation(line: 177, column: 18, scope: !1802)
!1847 = !DILocation(line: 177, column: 12, scope: !1802)
!1848 = !DILocation(line: 177, column: 5, scope: !1802)
!1849 = !DILocation(line: 178, column: 7, scope: !1802)
!1850 = !DILocation(line: 178, column: 12, scope: !1802)
!1851 = !DILocation(line: 178, column: 10, scope: !1802)
!1852 = !DILocation(line: 178, column: 17, scope: !1802)
!1853 = !DILocation(line: 178, column: 22, scope: !1802)
!1854 = !DILocation(line: 178, column: 20, scope: !1802)
!1855 = !DILocation(line: 178, column: 15, scope: !1802)
!1856 = !DILocation(line: 178, column: 5, scope: !1802)
!1857 = !DILocation(line: 179, column: 19, scope: !1802)
!1858 = !DILocation(line: 179, column: 17, scope: !1802)
!1859 = !DILocation(line: 179, column: 12, scope: !1802)
!1860 = !DILocation(line: 179, column: 7, scope: !1802)
!1861 = !DILocation(line: 179, column: 5, scope: !1802)
!1862 = !DILocation(line: 180, column: 6, scope: !1802)
!1863 = !DILocation(line: 180, column: 17, scope: !1802)
!1864 = !DILocation(line: 180, column: 15, scope: !1802)
!1865 = !DILocation(line: 180, column: 9, scope: !1802)
!1866 = !DILocation(line: 180, column: 4, scope: !1802)
!1867 = !DILocation(line: 181, column: 13, scope: !1802)
!1868 = !DILocation(line: 181, column: 11, scope: !1802)
!1869 = !DILocation(line: 181, column: 17, scope: !1802)
!1870 = !DILocation(line: 181, column: 22, scope: !1802)
!1871 = !DILocation(line: 181, column: 20, scope: !1802)
!1872 = !DILocation(line: 181, column: 15, scope: !1802)
!1873 = !DILocation(line: 181, column: 5, scope: !1802)
!1874 = !DILocation(line: 182, column: 19, scope: !1802)
!1875 = !DILocation(line: 182, column: 17, scope: !1802)
!1876 = !DILocation(line: 182, column: 12, scope: !1802)
!1877 = !DILocation(line: 182, column: 7, scope: !1802)
!1878 = !DILocation(line: 182, column: 5, scope: !1802)
!1879 = !DILocation(line: 183, column: 9, scope: !1802)
!1880 = !DILocation(line: 183, column: 20, scope: !1802)
!1881 = !DILocation(line: 183, column: 18, scope: !1802)
!1882 = !DILocation(line: 183, column: 12, scope: !1802)
!1883 = !DILocation(line: 183, column: 4, scope: !1802)
!1884 = !DILocation(line: 183, column: 7, scope: !1802)
!1885 = !DILocation(line: 185, column: 18, scope: !1802)
!1886 = !DILocation(line: 185, column: 17, scope: !1802)
!1887 = !DILocation(line: 185, column: 14, scope: !1802)
!1888 = !DILocation(line: 185, column: 2, scope: !1802)
!1889 = distinct !DISubprogram(name: "c_print_results", linkageName: "_Z15c_print_resultsPcciiiiddS_iS_S_S_S_S_S_S_S_S_S_S_S_S_S_", scope: !3, file: !3, line: 191, type: !1890, scopeLine: 214, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !962)
!1890 = !DISubroutineType(types: !1891)
!1891 = !{null, !100, !101, !97, !97, !97, !97, !99, !99, !100, !97, !100, !100, !100, !100, !100, !100, !100, !100, !100, !100, !100, !100, !100, !100}
!1892 = !DILocalVariable(name: "name", arg: 1, scope: !1889, file: !3, line: 191, type: !100)
!1893 = !DILocation(line: 191, column: 28, scope: !1889)
!1894 = !DILocalVariable(name: "class_npb", arg: 2, scope: !1889, file: !3, line: 192, type: !101)
!1895 = !DILocation(line: 192, column: 8, scope: !1889)
!1896 = !DILocalVariable(name: "n1", arg: 3, scope: !1889, file: !3, line: 193, type: !97)
!1897 = !DILocation(line: 193, column: 7, scope: !1889)
!1898 = !DILocalVariable(name: "n2", arg: 4, scope: !1889, file: !3, line: 194, type: !97)
!1899 = !DILocation(line: 194, column: 7, scope: !1889)
!1900 = !DILocalVariable(name: "n3", arg: 5, scope: !1889, file: !3, line: 195, type: !97)
!1901 = !DILocation(line: 195, column: 7, scope: !1889)
!1902 = !DILocalVariable(name: "niter", arg: 6, scope: !1889, file: !3, line: 196, type: !97)
!1903 = !DILocation(line: 196, column: 7, scope: !1889)
!1904 = !DILocalVariable(name: "t", arg: 7, scope: !1889, file: !3, line: 197, type: !99)
!1905 = !DILocation(line: 197, column: 10, scope: !1889)
!1906 = !DILocalVariable(name: "mops", arg: 8, scope: !1889, file: !3, line: 198, type: !99)
!1907 = !DILocation(line: 198, column: 10, scope: !1889)
!1908 = !DILocalVariable(name: "optype", arg: 9, scope: !1889, file: !3, line: 199, type: !100)
!1909 = !DILocation(line: 199, column: 9, scope: !1889)
!1910 = !DILocalVariable(name: "passed_verification", arg: 10, scope: !1889, file: !3, line: 200, type: !97)
!1911 = !DILocation(line: 200, column: 7, scope: !1889)
!1912 = !DILocalVariable(name: "npbversion", arg: 11, scope: !1889, file: !3, line: 201, type: !100)
!1913 = !DILocation(line: 201, column: 9, scope: !1889)
!1914 = !DILocalVariable(name: "compiletime", arg: 12, scope: !1889, file: !3, line: 202, type: !100)
!1915 = !DILocation(line: 202, column: 9, scope: !1889)
!1916 = !DILocalVariable(name: "compilerversion", arg: 13, scope: !1889, file: !3, line: 203, type: !100)
!1917 = !DILocation(line: 203, column: 9, scope: !1889)
!1918 = !DILocalVariable(name: "libversion", arg: 14, scope: !1889, file: !3, line: 204, type: !100)
!1919 = !DILocation(line: 204, column: 9, scope: !1889)
!1920 = !DILocalVariable(name: "cpu_device", arg: 15, scope: !1889, file: !3, line: 205, type: !100)
!1921 = !DILocation(line: 205, column: 9, scope: !1889)
!1922 = !DILocalVariable(name: "gpu_device", arg: 16, scope: !1889, file: !3, line: 206, type: !100)
!1923 = !DILocation(line: 206, column: 9, scope: !1889)
!1924 = !DILocalVariable(name: "gpu_config", arg: 17, scope: !1889, file: !3, line: 207, type: !100)
!1925 = !DILocation(line: 207, column: 9, scope: !1889)
!1926 = !DILocalVariable(name: "cc", arg: 18, scope: !1889, file: !3, line: 208, type: !100)
!1927 = !DILocation(line: 208, column: 9, scope: !1889)
!1928 = !DILocalVariable(name: "clink", arg: 19, scope: !1889, file: !3, line: 209, type: !100)
!1929 = !DILocation(line: 209, column: 9, scope: !1889)
!1930 = !DILocalVariable(name: "c_lib", arg: 20, scope: !1889, file: !3, line: 210, type: !100)
!1931 = !DILocation(line: 210, column: 9, scope: !1889)
!1932 = !DILocalVariable(name: "c_inc", arg: 21, scope: !1889, file: !3, line: 211, type: !100)
!1933 = !DILocation(line: 211, column: 9, scope: !1889)
!1934 = !DILocalVariable(name: "cflags", arg: 22, scope: !1889, file: !3, line: 212, type: !100)
!1935 = !DILocation(line: 212, column: 9, scope: !1889)
!1936 = !DILocalVariable(name: "clinkflags", arg: 23, scope: !1889, file: !3, line: 213, type: !100)
!1937 = !DILocation(line: 213, column: 9, scope: !1889)
!1938 = !DILocalVariable(name: "rand", arg: 24, scope: !1889, file: !3, line: 214, type: !100)
!1939 = !DILocation(line: 214, column: 9, scope: !1889)
!1940 = !DILocation(line: 215, column: 44, scope: !1889)
!1941 = !DILocation(line: 215, column: 4, scope: !1889)
!1942 = !DILocation(line: 216, column: 61, scope: !1889)
!1943 = !DILocation(line: 216, column: 4, scope: !1889)
!1944 = !DILocation(line: 217, column: 8, scope: !1945)
!1945 = distinct !DILexicalBlock(scope: !1889, file: !3, line: 217, column: 7)
!1946 = !DILocation(line: 217, column: 15, scope: !1945)
!1947 = !DILocation(line: 217, column: 21, scope: !1945)
!1948 = !DILocation(line: 217, column: 24, scope: !1945)
!1949 = !DILocation(line: 217, column: 31, scope: !1945)
!1950 = !DILocation(line: 217, column: 7, scope: !1889)
!1951 = !DILocation(line: 218, column: 8, scope: !1952)
!1952 = distinct !DILexicalBlock(scope: !1953, file: !3, line: 218, column: 8)
!1953 = distinct !DILexicalBlock(scope: !1945, file: !3, line: 217, column: 38)
!1954 = !DILocation(line: 218, column: 10, scope: !1952)
!1955 = !DILocation(line: 218, column: 8, scope: !1953)
!1956 = !DILocalVariable(name: "nn", scope: !1957, file: !3, line: 219, type: !313)
!1957 = distinct !DILexicalBlock(scope: !1952, file: !3, line: 218, column: 14)
!1958 = !DILocation(line: 219, column: 11, scope: !1957)
!1959 = !DILocation(line: 219, column: 16, scope: !1957)
!1960 = !DILocation(line: 220, column: 9, scope: !1961)
!1961 = distinct !DILexicalBlock(scope: !1957, file: !3, line: 220, column: 9)
!1962 = !DILocation(line: 220, column: 11, scope: !1961)
!1963 = !DILocation(line: 220, column: 9, scope: !1957)
!1964 = !DILocation(line: 220, column: 20, scope: !1965)
!1965 = distinct !DILexicalBlock(scope: !1961, file: !3, line: 220, column: 15)
!1966 = !DILocation(line: 220, column: 18, scope: !1965)
!1967 = !DILocation(line: 220, column: 23, scope: !1965)
!1968 = !DILocation(line: 221, column: 55, scope: !1957)
!1969 = !DILocation(line: 221, column: 6, scope: !1957)
!1970 = !DILocation(line: 222, column: 5, scope: !1957)
!1971 = !DILocation(line: 223, column: 61, scope: !1972)
!1972 = distinct !DILexicalBlock(scope: !1952, file: !3, line: 222, column: 10)
!1973 = !DILocation(line: 223, column: 64, scope: !1972)
!1974 = !DILocation(line: 223, column: 67, scope: !1972)
!1975 = !DILocation(line: 223, column: 6, scope: !1972)
!1976 = !DILocation(line: 225, column: 4, scope: !1953)
!1977 = !DILocalVariable(name: "size", scope: !1978, file: !3, line: 226, type: !1979)
!1978 = distinct !DILexicalBlock(scope: !1945, file: !3, line: 225, column: 9)
!1979 = !DICompositeType(tag: DW_TAG_array_type, baseType: !101, size: 128, elements: !1980)
!1980 = !{!1981}
!1981 = !DISubrange(count: 16)
!1982 = !DILocation(line: 226, column: 10, scope: !1978)
!1983 = !DILocalVariable(name: "j", scope: !1978, file: !3, line: 227, type: !97)
!1984 = !DILocation(line: 227, column: 9, scope: !1978)
!1985 = !DILocation(line: 228, column: 9, scope: !1986)
!1986 = distinct !DILexicalBlock(scope: !1978, file: !3, line: 228, column: 8)
!1987 = !DILocation(line: 228, column: 11, scope: !1986)
!1988 = !DILocation(line: 228, column: 16, scope: !1986)
!1989 = !DILocation(line: 228, column: 20, scope: !1986)
!1990 = !DILocation(line: 228, column: 22, scope: !1986)
!1991 = !DILocation(line: 228, column: 8, scope: !1978)
!1992 = !DILocation(line: 229, column: 10, scope: !1993)
!1993 = distinct !DILexicalBlock(scope: !1994, file: !3, line: 229, column: 9)
!1994 = distinct !DILexicalBlock(scope: !1986, file: !3, line: 228, column: 27)
!1995 = !DILocation(line: 229, column: 17, scope: !1993)
!1996 = !DILocation(line: 229, column: 23, scope: !1993)
!1997 = !DILocation(line: 229, column: 26, scope: !1993)
!1998 = !DILocation(line: 229, column: 33, scope: !1993)
!1999 = !DILocation(line: 229, column: 9, scope: !1994)
!2000 = !DILocation(line: 230, column: 15, scope: !2001)
!2001 = distinct !DILexicalBlock(scope: !1993, file: !3, line: 229, column: 40)
!2002 = !DILocation(line: 230, column: 41, scope: !2001)
!2003 = !DILocation(line: 230, column: 32, scope: !2001)
!2004 = !DILocation(line: 230, column: 7, scope: !2001)
!2005 = !DILocation(line: 231, column: 9, scope: !2001)
!2006 = !DILocation(line: 232, column: 15, scope: !2007)
!2007 = distinct !DILexicalBlock(scope: !2001, file: !3, line: 232, column: 10)
!2008 = !DILocation(line: 232, column: 10, scope: !2007)
!2009 = !DILocation(line: 232, column: 18, scope: !2007)
!2010 = !DILocation(line: 232, column: 10, scope: !2001)
!2011 = !DILocation(line: 233, column: 13, scope: !2012)
!2012 = distinct !DILexicalBlock(scope: !2007, file: !3, line: 232, column: 25)
!2013 = !DILocation(line: 233, column: 8, scope: !2012)
!2014 = !DILocation(line: 233, column: 16, scope: !2012)
!2015 = !DILocation(line: 234, column: 9, scope: !2012)
!2016 = !DILocation(line: 235, column: 7, scope: !2012)
!2017 = !DILocation(line: 236, column: 12, scope: !2001)
!2018 = !DILocation(line: 236, column: 13, scope: !2001)
!2019 = !DILocation(line: 236, column: 7, scope: !2001)
!2020 = !DILocation(line: 236, column: 17, scope: !2001)
!2021 = !DILocation(line: 237, column: 52, scope: !2001)
!2022 = !DILocation(line: 237, column: 7, scope: !2001)
!2023 = !DILocation(line: 238, column: 6, scope: !2001)
!2024 = !DILocation(line: 239, column: 55, scope: !2025)
!2025 = distinct !DILexicalBlock(scope: !1993, file: !3, line: 238, column: 11)
!2026 = !DILocation(line: 239, column: 7, scope: !2025)
!2027 = !DILocation(line: 241, column: 5, scope: !1994)
!2028 = !DILocation(line: 242, column: 59, scope: !2029)
!2029 = distinct !DILexicalBlock(scope: !1986, file: !3, line: 241, column: 10)
!2030 = !DILocation(line: 242, column: 63, scope: !2029)
!2031 = !DILocation(line: 242, column: 67, scope: !2029)
!2032 = !DILocation(line: 242, column: 6, scope: !2029)
!2033 = !DILocation(line: 245, column: 52, scope: !1889)
!2034 = !DILocation(line: 245, column: 4, scope: !1889)
!2035 = !DILocation(line: 246, column: 54, scope: !1889)
!2036 = !DILocation(line: 246, column: 4, scope: !1889)
!2037 = !DILocation(line: 247, column: 54, scope: !1889)
!2038 = !DILocation(line: 247, column: 4, scope: !1889)
!2039 = !DILocation(line: 248, column: 40, scope: !1889)
!2040 = !DILocation(line: 248, column: 4, scope: !1889)
!2041 = !DILocation(line: 249, column: 7, scope: !2042)
!2042 = distinct !DILexicalBlock(scope: !1889, file: !3, line: 249, column: 7)
!2043 = !DILocation(line: 249, column: 27, scope: !2042)
!2044 = !DILocation(line: 249, column: 7, scope: !1889)
!2045 = !DILocation(line: 250, column: 5, scope: !2046)
!2046 = distinct !DILexicalBlock(scope: !2042, file: !3, line: 249, column: 31)
!2047 = !DILocation(line: 251, column: 4, scope: !2046)
!2048 = !DILocation(line: 251, column: 13, scope: !2049)
!2049 = distinct !DILexicalBlock(scope: !2042, file: !3, line: 251, column: 13)
!2050 = !DILocation(line: 251, column: 13, scope: !2042)
!2051 = !DILocation(line: 252, column: 5, scope: !2052)
!2052 = distinct !DILexicalBlock(scope: !2049, file: !3, line: 251, column: 33)
!2053 = !DILocation(line: 253, column: 4, scope: !2052)
!2054 = !DILocation(line: 254, column: 5, scope: !2055)
!2055 = distinct !DILexicalBlock(scope: !2049, file: !3, line: 253, column: 9)
!2056 = !DILocation(line: 256, column: 52, scope: !1889)
!2057 = !DILocation(line: 256, column: 4, scope: !1889)
!2058 = !DILocation(line: 257, column: 52, scope: !1889)
!2059 = !DILocation(line: 257, column: 4, scope: !1889)
!2060 = !DILocation(line: 258, column: 52, scope: !1889)
!2061 = !DILocation(line: 258, column: 4, scope: !1889)
!2062 = !DILocation(line: 259, column: 52, scope: !1889)
!2063 = !DILocation(line: 259, column: 4, scope: !1889)
!2064 = !DILocation(line: 260, column: 4, scope: !1889)
!2065 = !DILocation(line: 261, column: 38, scope: !1889)
!2066 = !DILocation(line: 261, column: 4, scope: !1889)
!2067 = !DILocation(line: 262, column: 38, scope: !1889)
!2068 = !DILocation(line: 262, column: 4, scope: !1889)
!2069 = !DILocation(line: 263, column: 38, scope: !1889)
!2070 = !DILocation(line: 263, column: 4, scope: !1889)
!2071 = !DILocation(line: 264, column: 38, scope: !1889)
!2072 = !DILocation(line: 264, column: 4, scope: !1889)
!2073 = !DILocation(line: 265, column: 38, scope: !1889)
!2074 = !DILocation(line: 265, column: 4, scope: !1889)
!2075 = !DILocation(line: 266, column: 38, scope: !1889)
!2076 = !DILocation(line: 266, column: 4, scope: !1889)
!2077 = !DILocation(line: 267, column: 38, scope: !1889)
!2078 = !DILocation(line: 267, column: 4, scope: !1889)
!2079 = !DILocation(line: 268, column: 4, scope: !1889)
!2080 = !DILocation(line: 269, column: 38, scope: !1889)
!2081 = !DILocation(line: 269, column: 4, scope: !1889)
!2082 = !DILocation(line: 270, column: 38, scope: !1889)
!2083 = !DILocation(line: 270, column: 4, scope: !1889)
!2084 = !DILocation(line: 271, column: 4, scope: !1889)
!2085 = !DILocation(line: 272, column: 38, scope: !1889)
!2086 = !DILocation(line: 272, column: 4, scope: !1889)
!2087 = !DILocation(line: 287, column: 4, scope: !1889)
!2088 = !DILocation(line: 288, column: 4, scope: !1889)
!2089 = !DILocation(line: 289, column: 4, scope: !1889)
!2090 = !DILocation(line: 290, column: 4, scope: !1889)
!2091 = !DILocation(line: 291, column: 4, scope: !1889)
!2092 = !DILocation(line: 292, column: 4, scope: !1889)
!2093 = !DILocation(line: 293, column: 4, scope: !1889)
!2094 = !DILocation(line: 294, column: 4, scope: !1889)
!2095 = !DILocation(line: 295, column: 4, scope: !1889)
!2096 = !DILocation(line: 296, column: 4, scope: !1889)
!2097 = !DILocation(line: 297, column: 3, scope: !1889)
!2098 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 301, type: !2099, scopeLine: 301, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !962)
!2099 = !DISubroutineType(types: !2100)
!2100 = !{!97, !97, !566}
!2101 = !DILocalVariable(name: "argc", arg: 1, scope: !2098, file: !3, line: 301, type: !97)
!2102 = !DILocation(line: 301, column: 14, scope: !2098)
!2103 = !DILocalVariable(name: "argv", arg: 2, scope: !2098, file: !3, line: 301, type: !566)
!2104 = !DILocation(line: 301, column: 27, scope: !2098)
!2105 = !DILocation(line: 308, column: 15, scope: !2098)
!2106 = !DILocation(line: 308, column: 6, scope: !2098)
!2107 = !DILocation(line: 308, column: 4, scope: !2098)
!2108 = !DILocalVariable(name: "Mops", scope: !2098, file: !3, line: 309, type: !99)
!2109 = !DILocation(line: 309, column: 9, scope: !2098)
!2110 = !DILocalVariable(name: "t1", scope: !2098, file: !3, line: 309, type: !99)
!2111 = !DILocation(line: 309, column: 15, scope: !2098)
!2112 = !DILocalVariable(name: "sx", scope: !2098, file: !3, line: 310, type: !99)
!2113 = !DILocation(line: 310, column: 9, scope: !2098)
!2114 = !DILocalVariable(name: "sy", scope: !2098, file: !3, line: 310, type: !99)
!2115 = !DILocation(line: 310, column: 13, scope: !2098)
!2116 = !DILocalVariable(name: "an", scope: !2098, file: !3, line: 310, type: !99)
!2117 = !DILocation(line: 310, column: 17, scope: !2098)
!2118 = !DILocalVariable(name: "gc", scope: !2098, file: !3, line: 310, type: !99)
!2119 = !DILocation(line: 310, column: 21, scope: !2098)
!2120 = !DILocalVariable(name: "sx_verify_value", scope: !2098, file: !3, line: 311, type: !99)
!2121 = !DILocation(line: 311, column: 9, scope: !2098)
!2122 = !DILocalVariable(name: "sy_verify_value", scope: !2098, file: !3, line: 311, type: !99)
!2123 = !DILocation(line: 311, column: 26, scope: !2098)
!2124 = !DILocalVariable(name: "sx_err", scope: !2098, file: !3, line: 311, type: !99)
!2125 = !DILocation(line: 311, column: 43, scope: !2098)
!2126 = !DILocalVariable(name: "sy_err", scope: !2098, file: !3, line: 311, type: !99)
!2127 = !DILocation(line: 311, column: 51, scope: !2098)
!2128 = !DILocalVariable(name: "i", scope: !2098, file: !3, line: 312, type: !97)
!2129 = !DILocation(line: 312, column: 6, scope: !2098)
!2130 = !DILocalVariable(name: "j", scope: !2098, file: !3, line: 312, type: !97)
!2131 = !DILocation(line: 312, column: 9, scope: !2098)
!2132 = !DILocalVariable(name: "nit", scope: !2098, file: !3, line: 312, type: !97)
!2133 = !DILocation(line: 312, column: 12, scope: !2098)
!2134 = !DILocalVariable(name: "block", scope: !2098, file: !3, line: 312, type: !97)
!2135 = !DILocation(line: 312, column: 17, scope: !2098)
!2136 = !DILocalVariable(name: "verified", scope: !2098, file: !3, line: 313, type: !2137)
!2137 = !DIDerivedType(tag: DW_TAG_typedef, name: "boolean", file: !1694, line: 80, baseType: !97)
!2138 = !DILocation(line: 313, column: 10, scope: !2098)
!2139 = !DILocalVariable(name: "size", scope: !2098, file: !3, line: 314, type: !1979)
!2140 = !DILocation(line: 314, column: 7, scope: !2098)
!2141 = !DILocation(line: 324, column: 10, scope: !2098)
!2142 = !DILocation(line: 324, column: 26, scope: !2098)
!2143 = !DILocation(line: 324, column: 2, scope: !2098)
!2144 = !DILocation(line: 325, column: 4, scope: !2098)
!2145 = !DILocation(line: 326, column: 10, scope: !2146)
!2146 = distinct !DILexicalBlock(scope: !2098, file: !3, line: 326, column: 5)
!2147 = !DILocation(line: 326, column: 5, scope: !2146)
!2148 = !DILocation(line: 326, column: 12, scope: !2146)
!2149 = !DILocation(line: 326, column: 5, scope: !2098)
!2150 = !DILocation(line: 326, column: 20, scope: !2151)
!2151 = distinct !DILexicalBlock(scope: !2146, file: !3, line: 326, column: 18)
!2152 = !DILocation(line: 326, column: 23, scope: !2151)
!2153 = !DILocation(line: 327, column: 7, scope: !2098)
!2154 = !DILocation(line: 327, column: 8, scope: !2098)
!2155 = !DILocation(line: 327, column: 2, scope: !2098)
!2156 = !DILocation(line: 327, column: 12, scope: !2098)
!2157 = !DILocation(line: 328, column: 2, scope: !2098)
!2158 = !DILocation(line: 329, column: 56, scope: !2098)
!2159 = !DILocation(line: 329, column: 2, scope: !2098)
!2160 = !DILocation(line: 331, column: 11, scope: !2098)
!2161 = !DILocation(line: 333, column: 5, scope: !2098)
!2162 = !DILocation(line: 335, column: 7, scope: !2163)
!2163 = distinct !DILexicalBlock(scope: !2098, file: !3, line: 335, column: 2)
!2164 = !DILocation(line: 335, column: 6, scope: !2163)
!2165 = !DILocation(line: 335, column: 11, scope: !2166)
!2166 = distinct !DILexicalBlock(scope: !2163, file: !3, line: 335, column: 2)
!2167 = !DILocation(line: 335, column: 12, scope: !2166)
!2168 = !DILocation(line: 335, column: 2, scope: !2163)
!2169 = !DILocation(line: 336, column: 15, scope: !2170)
!2170 = distinct !DILexicalBlock(scope: !2166, file: !3, line: 335, column: 23)
!2171 = !DILocation(line: 336, column: 3, scope: !2170)
!2172 = !DILocation(line: 337, column: 2, scope: !2170)
!2173 = !DILocation(line: 335, column: 20, scope: !2166)
!2174 = !DILocation(line: 335, column: 2, scope: !2166)
!2175 = distinct !{!2175, !2168, !2176}
!2176 = !DILocation(line: 337, column: 2, scope: !2163)
!2177 = !DILocation(line: 339, column: 7, scope: !2098)
!2178 = !DILocation(line: 339, column: 5, scope: !2098)
!2179 = !DILocation(line: 340, column: 5, scope: !2098)
!2180 = !DILocation(line: 341, column: 5, scope: !2098)
!2181 = !DILocation(line: 342, column: 5, scope: !2098)
!2182 = !DILocation(line: 344, column: 7, scope: !2183)
!2183 = distinct !DILexicalBlock(scope: !2098, file: !3, line: 344, column: 2)
!2184 = !DILocation(line: 344, column: 6, scope: !2183)
!2185 = !DILocation(line: 344, column: 11, scope: !2186)
!2186 = distinct !DILexicalBlock(scope: !2183, file: !3, line: 344, column: 2)
!2187 = !DILocation(line: 344, column: 12, scope: !2186)
!2188 = !DILocation(line: 344, column: 2, scope: !2183)
!2189 = !DILocation(line: 345, column: 3, scope: !2190)
!2190 = distinct !DILexicalBlock(scope: !2186, file: !3, line: 344, column: 21)
!2191 = !DILocation(line: 345, column: 5, scope: !2190)
!2192 = !DILocation(line: 345, column: 8, scope: !2190)
!2193 = !DILocation(line: 346, column: 2, scope: !2190)
!2194 = !DILocation(line: 344, column: 18, scope: !2186)
!2195 = !DILocation(line: 344, column: 2, scope: !2186)
!2196 = distinct !{!2196, !2188, !2197}
!2197 = !DILocation(line: 346, column: 2, scope: !2183)
!2198 = !DILocation(line: 348, column: 2, scope: !2098)
!2199 = !DILocation(line: 353, column: 15, scope: !2098)
!2200 = !DILocation(line: 354, column: 3, scope: !2098)
!2201 = !DILocation(line: 353, column: 12, scope: !2098)
!2202 = !DILocation(line: 353, column: 2, scope: !2098)
!2203 = !DILocation(line: 354, column: 24, scope: !2098)
!2204 = !DILocation(line: 355, column: 5, scope: !2098)
!2205 = !DILocation(line: 356, column: 5, scope: !2098)
!2206 = !DILocation(line: 357, column: 5, scope: !2098)
!2207 = !DILocation(line: 362, column: 13, scope: !2098)
!2208 = !DILocation(line: 362, column: 21, scope: !2098)
!2209 = !DILocation(line: 362, column: 31, scope: !2098)
!2210 = !DILocation(line: 362, column: 2, scope: !2098)
!2211 = !DILocation(line: 363, column: 13, scope: !2098)
!2212 = !DILocation(line: 363, column: 22, scope: !2098)
!2213 = !DILocation(line: 363, column: 33, scope: !2098)
!2214 = !DILocation(line: 363, column: 2, scope: !2098)
!2215 = !DILocation(line: 364, column: 13, scope: !2098)
!2216 = !DILocation(line: 364, column: 22, scope: !2098)
!2217 = !DILocation(line: 364, column: 33, scope: !2098)
!2218 = !DILocation(line: 364, column: 2, scope: !2098)
!2219 = !DILocation(line: 366, column: 11, scope: !2220)
!2220 = distinct !DILexicalBlock(scope: !2098, file: !3, line: 366, column: 2)
!2221 = !DILocation(line: 366, column: 6, scope: !2220)
!2222 = !DILocation(line: 366, column: 15, scope: !2223)
!2223 = distinct !DILexicalBlock(scope: !2220, file: !3, line: 366, column: 2)
!2224 = !DILocation(line: 366, column: 21, scope: !2223)
!2225 = !DILocation(line: 366, column: 20, scope: !2223)
!2226 = !DILocation(line: 366, column: 2, scope: !2220)
!2227 = !DILocation(line: 367, column: 8, scope: !2228)
!2228 = distinct !DILexicalBlock(scope: !2229, file: !3, line: 367, column: 3)
!2229 = distinct !DILexicalBlock(scope: !2223, file: !3, line: 366, column: 46)
!2230 = !DILocation(line: 367, column: 7, scope: !2228)
!2231 = !DILocation(line: 367, column: 12, scope: !2232)
!2232 = distinct !DILexicalBlock(scope: !2228, file: !3, line: 367, column: 3)
!2233 = !DILocation(line: 367, column: 13, scope: !2232)
!2234 = !DILocation(line: 367, column: 3, scope: !2228)
!2235 = !DILocation(line: 368, column: 10, scope: !2236)
!2236 = distinct !DILexicalBlock(scope: !2232, file: !3, line: 367, column: 22)
!2237 = !DILocation(line: 368, column: 17, scope: !2236)
!2238 = !DILocation(line: 368, column: 22, scope: !2236)
!2239 = !DILocation(line: 368, column: 26, scope: !2236)
!2240 = !DILocation(line: 368, column: 25, scope: !2236)
!2241 = !DILocation(line: 368, column: 4, scope: !2236)
!2242 = !DILocation(line: 368, column: 6, scope: !2236)
!2243 = !DILocation(line: 368, column: 8, scope: !2236)
!2244 = !DILocation(line: 369, column: 3, scope: !2236)
!2245 = !DILocation(line: 367, column: 19, scope: !2232)
!2246 = !DILocation(line: 367, column: 3, scope: !2232)
!2247 = distinct !{!2247, !2234, !2248}
!2248 = !DILocation(line: 369, column: 3, scope: !2228)
!2249 = !DILocation(line: 370, column: 7, scope: !2229)
!2250 = !DILocation(line: 370, column: 15, scope: !2229)
!2251 = !DILocation(line: 370, column: 5, scope: !2229)
!2252 = !DILocation(line: 371, column: 7, scope: !2229)
!2253 = !DILocation(line: 371, column: 15, scope: !2229)
!2254 = !DILocation(line: 371, column: 5, scope: !2229)
!2255 = !DILocation(line: 372, column: 2, scope: !2229)
!2256 = !DILocation(line: 366, column: 43, scope: !2223)
!2257 = !DILocation(line: 366, column: 2, scope: !2223)
!2258 = distinct !{!2258, !2226, !2259}
!2259 = !DILocation(line: 372, column: 2, scope: !2220)
!2260 = !DILocation(line: 373, column: 7, scope: !2261)
!2261 = distinct !DILexicalBlock(scope: !2098, file: !3, line: 373, column: 2)
!2262 = !DILocation(line: 373, column: 6, scope: !2261)
!2263 = !DILocation(line: 373, column: 11, scope: !2264)
!2264 = distinct !DILexicalBlock(scope: !2261, file: !3, line: 373, column: 2)
!2265 = !DILocation(line: 373, column: 12, scope: !2264)
!2266 = !DILocation(line: 373, column: 2, scope: !2261)
!2267 = !DILocation(line: 374, column: 7, scope: !2268)
!2268 = distinct !DILexicalBlock(scope: !2264, file: !3, line: 373, column: 21)
!2269 = !DILocation(line: 374, column: 9, scope: !2268)
!2270 = !DILocation(line: 374, column: 5, scope: !2268)
!2271 = !DILocation(line: 375, column: 2, scope: !2268)
!2272 = !DILocation(line: 373, column: 18, scope: !2264)
!2273 = !DILocation(line: 373, column: 2, scope: !2264)
!2274 = distinct !{!2274, !2266, !2275}
!2275 = !DILocation(line: 375, column: 2, scope: !2261)
!2276 = !DILocation(line: 377, column: 6, scope: !2098)
!2277 = !DILocation(line: 378, column: 11, scope: !2098)
!2278 = !DILocation(line: 386, column: 19, scope: !2279)
!2279 = distinct !DILexicalBlock(scope: !2280, file: !3, line: 385, column: 19)
!2280 = distinct !DILexicalBlock(scope: !2281, file: !3, line: 385, column: 11)
!2281 = distinct !DILexicalBlock(scope: !2282, file: !3, line: 382, column: 11)
!2282 = distinct !DILexicalBlock(scope: !2098, file: !3, line: 379, column: 5)
!2283 = !DILocation(line: 387, column: 19, scope: !2279)
!2284 = !DILocation(line: 403, column: 5, scope: !2285)
!2285 = distinct !DILexicalBlock(scope: !2098, file: !3, line: 403, column: 5)
!2286 = !DILocation(line: 403, column: 5, scope: !2098)
!2287 = !DILocation(line: 404, column: 18, scope: !2288)
!2288 = distinct !DILexicalBlock(scope: !2285, file: !3, line: 403, column: 14)
!2289 = !DILocation(line: 404, column: 23, scope: !2288)
!2290 = !DILocation(line: 404, column: 21, scope: !2288)
!2291 = !DILocation(line: 404, column: 42, scope: !2288)
!2292 = !DILocation(line: 404, column: 40, scope: !2288)
!2293 = !DILocation(line: 404, column: 12, scope: !2288)
!2294 = !DILocation(line: 404, column: 10, scope: !2288)
!2295 = !DILocation(line: 405, column: 18, scope: !2288)
!2296 = !DILocation(line: 405, column: 23, scope: !2288)
!2297 = !DILocation(line: 405, column: 21, scope: !2288)
!2298 = !DILocation(line: 405, column: 42, scope: !2288)
!2299 = !DILocation(line: 405, column: 40, scope: !2288)
!2300 = !DILocation(line: 405, column: 12, scope: !2288)
!2301 = !DILocation(line: 405, column: 10, scope: !2288)
!2302 = !DILocation(line: 406, column: 16, scope: !2288)
!2303 = !DILocation(line: 406, column: 23, scope: !2288)
!2304 = !DILocation(line: 406, column: 35, scope: !2288)
!2305 = !DILocation(line: 406, column: 39, scope: !2288)
!2306 = !DILocation(line: 406, column: 46, scope: !2288)
!2307 = !DILocation(line: 0, scope: !2288)
!2308 = !DILocation(line: 406, column: 14, scope: !2288)
!2309 = !DILocation(line: 406, column: 12, scope: !2288)
!2310 = !DILocation(line: 407, column: 2, scope: !2288)
!2311 = !DILocation(line: 408, column: 9, scope: !2098)
!2312 = !DILocation(line: 408, column: 22, scope: !2098)
!2313 = !DILocation(line: 408, column: 7, scope: !2098)
!2314 = !DILocation(line: 410, column: 2, scope: !2098)
!2315 = !DILocation(line: 411, column: 2, scope: !2098)
!2316 = !DILocation(line: 412, column: 2, scope: !2098)
!2317 = !DILocation(line: 413, column: 43, scope: !2098)
!2318 = !DILocation(line: 413, column: 2, scope: !2098)
!2319 = !DILocation(line: 414, column: 38, scope: !2098)
!2320 = !DILocation(line: 414, column: 42, scope: !2098)
!2321 = !DILocation(line: 414, column: 2, scope: !2098)
!2322 = !DILocation(line: 415, column: 2, scope: !2098)
!2323 = !DILocation(line: 416, column: 7, scope: !2324)
!2324 = distinct !DILexicalBlock(scope: !2098, file: !3, line: 416, column: 2)
!2325 = !DILocation(line: 416, column: 6, scope: !2324)
!2326 = !DILocation(line: 416, column: 11, scope: !2327)
!2327 = distinct !DILexicalBlock(scope: !2324, file: !3, line: 416, column: 2)
!2328 = !DILocation(line: 416, column: 12, scope: !2327)
!2329 = !DILocation(line: 416, column: 2, scope: !2324)
!2330 = !DILocation(line: 417, column: 25, scope: !2331)
!2331 = distinct !DILexicalBlock(scope: !2327, file: !3, line: 416, column: 21)
!2332 = !DILocation(line: 417, column: 28, scope: !2331)
!2333 = !DILocation(line: 417, column: 30, scope: !2331)
!2334 = !DILocation(line: 417, column: 3, scope: !2331)
!2335 = !DILocation(line: 418, column: 2, scope: !2331)
!2336 = !DILocation(line: 416, column: 18, scope: !2327)
!2337 = !DILocation(line: 416, column: 2, scope: !2327)
!2338 = distinct !{!2338, !2329, !2339}
!2339 = !DILocation(line: 418, column: 2, scope: !2324)
!2340 = !DILocalVariable(name: "gpu_config", scope: !2098, file: !3, line: 420, type: !137)
!2341 = !DILocation(line: 420, column: 7, scope: !2098)
!2342 = !DILocalVariable(name: "gpu_config_string", scope: !2098, file: !3, line: 421, type: !2343)
!2343 = !DICompositeType(tag: DW_TAG_array_type, baseType: !101, size: 16384, elements: !2344)
!2344 = !{!2345}
!2345 = !DISubrange(count: 2048)
!2346 = !DILocation(line: 421, column: 7, scope: !2098)
!2347 = !DILocation(line: 428, column: 10, scope: !2098)
!2348 = !DILocation(line: 428, column: 2, scope: !2098)
!2349 = !DILocation(line: 429, column: 9, scope: !2098)
!2350 = !DILocation(line: 429, column: 28, scope: !2098)
!2351 = !DILocation(line: 429, column: 2, scope: !2098)
!2352 = !DILocation(line: 430, column: 10, scope: !2098)
!2353 = !DILocation(line: 430, column: 45, scope: !2098)
!2354 = !DILocation(line: 430, column: 2, scope: !2098)
!2355 = !DILocation(line: 431, column: 9, scope: !2098)
!2356 = !DILocation(line: 431, column: 28, scope: !2098)
!2357 = !DILocation(line: 431, column: 2, scope: !2098)
!2358 = !DILocation(line: 439, column: 4, scope: !2098)
!2359 = !DILocation(line: 441, column: 4, scope: !2098)
!2360 = !DILocation(line: 443, column: 4, scope: !2098)
!2361 = !DILocation(line: 450, column: 4, scope: !2098)
!2362 = !DILocation(line: 434, column: 2, scope: !2098)
!2363 = !DILocation(line: 459, column: 2, scope: !2098)
!2364 = !DILocation(line: 461, column: 2, scope: !2098)
!2365 = distinct !DISubprogram(name: "setup_gpu", linkageName: "_ZL9setup_gpuv", scope: !3, file: !3, line: 577, type: !472, scopeLine: 577, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !962)
!2366 = !DILocation(line: 624, column: 33, scope: !2365)
!2367 = !DILocation(line: 625, column: 43, scope: !2365)
!2368 = !DILocation(line: 628, column: 49, scope: !2369)
!2369 = distinct !DILexicalBlock(scope: !2365, file: !3, line: 627, column: 5)
!2370 = !DILocation(line: 628, column: 25, scope: !2369)
!2371 = !DILocation(line: 627, column: 5, scope: !2365)
!2372 = !DILocation(line: 629, column: 21, scope: !2373)
!2373 = distinct !DILexicalBlock(scope: !2369, file: !3, line: 628, column: 69)
!2374 = !DILocation(line: 630, column: 2, scope: !2373)
!2375 = !DILocation(line: 631, column: 45, scope: !2376)
!2376 = distinct !DILexicalBlock(scope: !2369, file: !3, line: 630, column: 7)
!2377 = !DILocation(line: 631, column: 21, scope: !2376)
!2378 = !DILocation(line: 634, column: 45, scope: !2365)
!2379 = !DILocation(line: 634, column: 36, scope: !2365)
!2380 = !DILocation(line: 634, column: 21, scope: !2365)
!2381 = !DILocation(line: 634, column: 20, scope: !2365)
!2382 = !DILocation(line: 634, column: 18, scope: !2365)
!2383 = !DILocation(line: 636, column: 11, scope: !2365)
!2384 = !DILocation(line: 636, column: 27, scope: !2365)
!2385 = !DILocation(line: 636, column: 32, scope: !2365)
!2386 = !DILocation(line: 636, column: 9, scope: !2365)
!2387 = !DILocation(line: 637, column: 12, scope: !2365)
!2388 = !DILocation(line: 637, column: 28, scope: !2365)
!2389 = !DILocation(line: 637, column: 10, scope: !2365)
!2390 = !DILocation(line: 638, column: 12, scope: !2365)
!2391 = !DILocation(line: 638, column: 28, scope: !2365)
!2392 = !DILocation(line: 638, column: 10, scope: !2365)
!2393 = !DILocation(line: 640, column: 25, scope: !2365)
!2394 = !DILocation(line: 640, column: 18, scope: !2365)
!2395 = !DILocation(line: 640, column: 9, scope: !2365)
!2396 = !DILocation(line: 640, column: 8, scope: !2365)
!2397 = !DILocation(line: 641, column: 26, scope: !2365)
!2398 = !DILocation(line: 641, column: 19, scope: !2365)
!2399 = !DILocation(line: 641, column: 10, scope: !2365)
!2400 = !DILocation(line: 641, column: 9, scope: !2365)
!2401 = !DILocation(line: 642, column: 26, scope: !2365)
!2402 = !DILocation(line: 642, column: 19, scope: !2365)
!2403 = !DILocation(line: 642, column: 10, scope: !2365)
!2404 = !DILocation(line: 642, column: 9, scope: !2365)
!2405 = !DILocation(line: 644, column: 24, scope: !2365)
!2406 = !DILocation(line: 644, column: 2, scope: !2365)
!2407 = !DILocation(line: 645, column: 25, scope: !2365)
!2408 = !DILocation(line: 645, column: 2, scope: !2365)
!2409 = !DILocation(line: 646, column: 25, scope: !2365)
!2410 = !DILocation(line: 646, column: 2, scope: !2365)
!2411 = !DILocation(line: 647, column: 1, scope: !2365)
!2412 = distinct !DISubprogram(name: "dim3", linkageName: "_ZN4dim3C2Ejjj", scope: !1194, file: !1160, line: 421, type: !1200, scopeLine: 421, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, declaration: !1199, retainedNodes: !962)
!2413 = !DILocalVariable(name: "this", arg: 1, scope: !2412, type: !2414, flags: DIFlagArtificial | DIFlagObjectPointer)
!2414 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !1194, size: 64)
!2415 = !DILocation(line: 0, scope: !2412)
!2416 = !DILocalVariable(name: "vx", arg: 2, scope: !2412, file: !1160, line: 421, type: !7)
!2417 = !DILocation(line: 421, column: 43, scope: !2412)
!2418 = !DILocalVariable(name: "vy", arg: 3, scope: !2412, file: !1160, line: 421, type: !7)
!2419 = !DILocation(line: 421, column: 64, scope: !2412)
!2420 = !DILocalVariable(name: "vz", arg: 4, scope: !2412, file: !1160, line: 421, type: !7)
!2421 = !DILocation(line: 421, column: 85, scope: !2412)
!2422 = !DILocation(line: 421, column: 95, scope: !2412)
!2423 = !DILocation(line: 421, column: 97, scope: !2412)
!2424 = !DILocation(line: 421, column: 102, scope: !2412)
!2425 = !DILocation(line: 421, column: 104, scope: !2412)
!2426 = !DILocation(line: 421, column: 109, scope: !2412)
!2427 = !DILocation(line: 421, column: 111, scope: !2412)
!2428 = !DILocation(line: 421, column: 116, scope: !2412)
!2429 = distinct !DISubprogram(name: "gpu_kernel", linkageName: "_Z10gpu_kernelPdS_S_d", scope: !3, file: !3, line: 464, type: !1051, scopeLine: 467, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !962)
!2430 = !DILocalVariable(name: "q_global", arg: 1, scope: !2429, file: !3, line: 464, type: !98)
!2431 = !DILocation(line: 464, column: 36, scope: !2429)
!2432 = !DILocalVariable(name: "sx_global", arg: 2, scope: !2429, file: !3, line: 465, type: !98)
!2433 = !DILocation(line: 465, column: 11, scope: !2429)
!2434 = !DILocalVariable(name: "sy_global", arg: 3, scope: !2429, file: !3, line: 466, type: !98)
!2435 = !DILocation(line: 466, column: 11, scope: !2429)
!2436 = !DILocalVariable(name: "an", arg: 4, scope: !2429, file: !3, line: 467, type: !99)
!2437 = !DILocation(line: 467, column: 10, scope: !2429)
!2438 = !DILocation(line: 467, column: 13, scope: !2429)
!2439 = !DILocation(line: 551, column: 1, scope: !2429)
!2440 = distinct !DISubprogram(name: "release_gpu", linkageName: "_ZL11release_gpuv", scope: !3, file: !3, line: 571, type: !472, scopeLine: 571, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !962)
!2441 = !DILocation(line: 572, column: 11, scope: !2440)
!2442 = !DILocation(line: 572, column: 2, scope: !2440)
!2443 = !DILocation(line: 573, column: 11, scope: !2440)
!2444 = !DILocation(line: 573, column: 2, scope: !2440)
!2445 = !DILocation(line: 574, column: 11, scope: !2440)
!2446 = !DILocation(line: 574, column: 2, scope: !2440)
!2447 = !DILocation(line: 575, column: 1, scope: !2440)
!2448 = distinct !DISubprogram(name: "cudaMalloc<double>", linkageName: "_ZL10cudaMallocIdE9cudaErrorPPT_m", scope: !2449, file: !2449, line: 490, type: !2450, scopeLine: 494, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, templateParams: !2454, retainedNodes: !962)
!2449 = !DIFile(filename: "/usr/local/cuda/include/cuda_runtime.h", directory: "")
!2450 = !DISubroutineType(types: !2451)
!2451 = !{!2452, !2453, !121}
!2452 = !DIDerivedType(tag: DW_TAG_typedef, name: "cudaError_t", file: !6, line: 1419, baseType: !14)
!2453 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !98, size: 64)
!2454 = !{!2455}
!2455 = !DITemplateTypeParameter(name: "T", type: !99)
!2456 = !DILocalVariable(name: "devPtr", arg: 1, scope: !2448, file: !2449, line: 491, type: !2453)
!2457 = !DILocation(line: 491, column: 12, scope: !2448)
!2458 = !DILocalVariable(name: "size", arg: 2, scope: !2448, file: !2449, line: 492, type: !121)
!2459 = !DILocation(line: 492, column: 12, scope: !2448)
!2460 = !DILocation(line: 495, column: 38, scope: !2448)
!2461 = !DILocation(line: 495, column: 23, scope: !2448)
!2462 = !DILocation(line: 495, column: 46, scope: !2448)
!2463 = !DILocation(line: 495, column: 10, scope: !2448)
!2464 = !DILocation(line: 495, column: 3, scope: !2448)
