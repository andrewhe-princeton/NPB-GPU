; ModuleID = 'ep.cu'
source_filename = "ep.cu"
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

; Function Attrs: noinline nounwind uwtable
define dso_local double @_Z6randlcPdd(double* %x, double %a) #0 !dbg !966 {
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
  call void @llvm.dbg.declare(metadata double** %x.addr, metadata !970, metadata !DIExpression()), !dbg !971
  store double %a, double* %a.addr, align 8
  call void @llvm.dbg.declare(metadata double* %a.addr, metadata !972, metadata !DIExpression()), !dbg !973
  call void @llvm.dbg.declare(metadata double* %t1, metadata !974, metadata !DIExpression()), !dbg !975
  call void @llvm.dbg.declare(metadata double* %t2, metadata !976, metadata !DIExpression()), !dbg !977
  call void @llvm.dbg.declare(metadata double* %t3, metadata !978, metadata !DIExpression()), !dbg !979
  call void @llvm.dbg.declare(metadata double* %t4, metadata !980, metadata !DIExpression()), !dbg !981
  call void @llvm.dbg.declare(metadata double* %a1, metadata !982, metadata !DIExpression()), !dbg !983
  call void @llvm.dbg.declare(metadata double* %a2, metadata !984, metadata !DIExpression()), !dbg !985
  call void @llvm.dbg.declare(metadata double* %x1, metadata !986, metadata !DIExpression()), !dbg !987
  call void @llvm.dbg.declare(metadata double* %x2, metadata !988, metadata !DIExpression()), !dbg !989
  call void @llvm.dbg.declare(metadata double* %z, metadata !990, metadata !DIExpression()), !dbg !991
  %0 = load double, double* %a.addr, align 8, !dbg !992
  %mul = fmul contract double 0x3E80000000000000, %0, !dbg !993
  store double %mul, double* %t1, align 8, !dbg !994
  %1 = load double, double* %t1, align 8, !dbg !995
  %conv = fptosi double %1 to i32, !dbg !995
  %conv1 = sitofp i32 %conv to double, !dbg !996
  store double %conv1, double* %a1, align 8, !dbg !997
  %2 = load double, double* %a.addr, align 8, !dbg !998
  %3 = load double, double* %a1, align 8, !dbg !999
  %mul2 = fmul contract double 0x4160000000000000, %3, !dbg !1000
  %sub = fsub contract double %2, %mul2, !dbg !1001
  store double %sub, double* %a2, align 8, !dbg !1002
  %4 = load double*, double** %x.addr, align 8, !dbg !1003
  %5 = load double, double* %4, align 8, !dbg !1004
  %mul3 = fmul contract double 0x3E80000000000000, %5, !dbg !1005
  store double %mul3, double* %t1, align 8, !dbg !1006
  %6 = load double, double* %t1, align 8, !dbg !1007
  %conv4 = fptosi double %6 to i32, !dbg !1007
  %conv5 = sitofp i32 %conv4 to double, !dbg !1008
  store double %conv5, double* %x1, align 8, !dbg !1009
  %7 = load double*, double** %x.addr, align 8, !dbg !1010
  %8 = load double, double* %7, align 8, !dbg !1011
  %9 = load double, double* %x1, align 8, !dbg !1012
  %mul6 = fmul contract double 0x4160000000000000, %9, !dbg !1013
  %sub7 = fsub contract double %8, %mul6, !dbg !1014
  store double %sub7, double* %x2, align 8, !dbg !1015
  %10 = load double, double* %a1, align 8, !dbg !1016
  %11 = load double, double* %x2, align 8, !dbg !1017
  %mul8 = fmul contract double %10, %11, !dbg !1018
  %12 = load double, double* %a2, align 8, !dbg !1019
  %13 = load double, double* %x1, align 8, !dbg !1020
  %mul9 = fmul contract double %12, %13, !dbg !1021
  %add = fadd contract double %mul8, %mul9, !dbg !1022
  store double %add, double* %t1, align 8, !dbg !1023
  %14 = load double, double* %t1, align 8, !dbg !1024
  %mul10 = fmul contract double 0x3E80000000000000, %14, !dbg !1025
  %conv11 = fptosi double %mul10 to i32, !dbg !1026
  %conv12 = sitofp i32 %conv11 to double, !dbg !1027
  store double %conv12, double* %t2, align 8, !dbg !1028
  %15 = load double, double* %t1, align 8, !dbg !1029
  %16 = load double, double* %t2, align 8, !dbg !1030
  %mul13 = fmul contract double 0x4160000000000000, %16, !dbg !1031
  %sub14 = fsub contract double %15, %mul13, !dbg !1032
  store double %sub14, double* %z, align 8, !dbg !1033
  %17 = load double, double* %z, align 8, !dbg !1034
  %mul15 = fmul contract double 0x4160000000000000, %17, !dbg !1035
  %18 = load double, double* %a2, align 8, !dbg !1036
  %19 = load double, double* %x2, align 8, !dbg !1037
  %mul16 = fmul contract double %18, %19, !dbg !1038
  %add17 = fadd contract double %mul15, %mul16, !dbg !1039
  store double %add17, double* %t3, align 8, !dbg !1040
  %20 = load double, double* %t3, align 8, !dbg !1041
  %mul18 = fmul contract double 0x3D10000000000000, %20, !dbg !1042
  %conv19 = fptosi double %mul18 to i32, !dbg !1043
  %conv20 = sitofp i32 %conv19 to double, !dbg !1044
  store double %conv20, double* %t4, align 8, !dbg !1045
  %21 = load double, double* %t3, align 8, !dbg !1046
  %22 = load double, double* %t4, align 8, !dbg !1047
  %mul21 = fmul contract double 0x42D0000000000000, %22, !dbg !1048
  %sub22 = fsub contract double %21, %mul21, !dbg !1049
  %23 = load double*, double** %x.addr, align 8, !dbg !1050
  store double %sub22, double* %23, align 8, !dbg !1051
  %24 = load double*, double** %x.addr, align 8, !dbg !1052
  %25 = load double, double* %24, align 8, !dbg !1053
  %mul23 = fmul contract double 0x3D10000000000000, %25, !dbg !1054
  ret double %mul23, !dbg !1055
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: noinline uwtable
define dso_local void @_Z15c_print_resultsPcciiiiddS_iS_S_S_S_S_S_S_S_S_S_S_S_S_S_(i8* %name, i8 signext %class_npb, i32 %n1, i32 %n2, i32 %n3, i32 %niter, double %t, double %mops, i8* %optype, i32 %passed_verification, i8* %npbversion, i8* %compiletime, i8* %compilerversion, i8* %libversion, i8* %cpu_device, i8* %gpu_device, i8* %gpu_config, i8* %cc, i8* %clink, i8* %c_lib, i8* %c_inc, i8* %cflags, i8* %clinkflags, i8* %rand) #2 !dbg !1056 {
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
  call void @llvm.dbg.declare(metadata i8** %name.addr, metadata !1059, metadata !DIExpression()), !dbg !1060
  store i8 %class_npb, i8* %class_npb.addr, align 1
  call void @llvm.dbg.declare(metadata i8* %class_npb.addr, metadata !1061, metadata !DIExpression()), !dbg !1062
  store i32 %n1, i32* %n1.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %n1.addr, metadata !1063, metadata !DIExpression()), !dbg !1064
  store i32 %n2, i32* %n2.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %n2.addr, metadata !1065, metadata !DIExpression()), !dbg !1066
  store i32 %n3, i32* %n3.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %n3.addr, metadata !1067, metadata !DIExpression()), !dbg !1068
  store i32 %niter, i32* %niter.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %niter.addr, metadata !1069, metadata !DIExpression()), !dbg !1070
  store double %t, double* %t.addr, align 8
  call void @llvm.dbg.declare(metadata double* %t.addr, metadata !1071, metadata !DIExpression()), !dbg !1072
  store double %mops, double* %mops.addr, align 8
  call void @llvm.dbg.declare(metadata double* %mops.addr, metadata !1073, metadata !DIExpression()), !dbg !1074
  store i8* %optype, i8** %optype.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %optype.addr, metadata !1075, metadata !DIExpression()), !dbg !1076
  store i32 %passed_verification, i32* %passed_verification.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %passed_verification.addr, metadata !1077, metadata !DIExpression()), !dbg !1078
  store i8* %npbversion, i8** %npbversion.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %npbversion.addr, metadata !1079, metadata !DIExpression()), !dbg !1080
  store i8* %compiletime, i8** %compiletime.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %compiletime.addr, metadata !1081, metadata !DIExpression()), !dbg !1082
  store i8* %compilerversion, i8** %compilerversion.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %compilerversion.addr, metadata !1083, metadata !DIExpression()), !dbg !1084
  store i8* %libversion, i8** %libversion.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %libversion.addr, metadata !1085, metadata !DIExpression()), !dbg !1086
  store i8* %cpu_device, i8** %cpu_device.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %cpu_device.addr, metadata !1087, metadata !DIExpression()), !dbg !1088
  store i8* %gpu_device, i8** %gpu_device.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %gpu_device.addr, metadata !1089, metadata !DIExpression()), !dbg !1090
  store i8* %gpu_config, i8** %gpu_config.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %gpu_config.addr, metadata !1091, metadata !DIExpression()), !dbg !1092
  store i8* %cc, i8** %cc.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %cc.addr, metadata !1093, metadata !DIExpression()), !dbg !1094
  store i8* %clink, i8** %clink.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %clink.addr, metadata !1095, metadata !DIExpression()), !dbg !1096
  store i8* %c_lib, i8** %c_lib.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %c_lib.addr, metadata !1097, metadata !DIExpression()), !dbg !1098
  store i8* %c_inc, i8** %c_inc.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %c_inc.addr, metadata !1099, metadata !DIExpression()), !dbg !1100
  store i8* %cflags, i8** %cflags.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %cflags.addr, metadata !1101, metadata !DIExpression()), !dbg !1102
  store i8* %clinkflags, i8** %clinkflags.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %clinkflags.addr, metadata !1103, metadata !DIExpression()), !dbg !1104
  store i8* %rand, i8** %rand.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %rand.addr, metadata !1105, metadata !DIExpression()), !dbg !1106
  %0 = load i8*, i8** %name.addr, align 8, !dbg !1107
  %call = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([27 x i8], [27 x i8]* @.str, i64 0, i64 0), i8* %0), !dbg !1108
  %1 = load i8, i8* %class_npb.addr, align 1, !dbg !1109
  %conv = sext i8 %1 to i32, !dbg !1109
  %call1 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([46 x i8], [46 x i8]* @.str.1, i64 0, i64 0), i32 %conv), !dbg !1110
  %2 = load i8*, i8** %name.addr, align 8, !dbg !1111
  %arrayidx = getelementptr inbounds i8, i8* %2, i64 0, !dbg !1111
  %3 = load i8, i8* %arrayidx, align 1, !dbg !1111
  %conv2 = sext i8 %3 to i32, !dbg !1111
  %cmp = icmp eq i32 %conv2, 73, !dbg !1113
  br i1 %cmp, label %land.lhs.true, label %if.else15, !dbg !1114

land.lhs.true:                                    ; preds = %entry
  %4 = load i8*, i8** %name.addr, align 8, !dbg !1115
  %arrayidx3 = getelementptr inbounds i8, i8* %4, i64 1, !dbg !1115
  %5 = load i8, i8* %arrayidx3, align 1, !dbg !1115
  %conv4 = sext i8 %5 to i32, !dbg !1115
  %cmp5 = icmp eq i32 %conv4, 83, !dbg !1116
  br i1 %cmp5, label %if.then, label %if.else15, !dbg !1117

if.then:                                          ; preds = %land.lhs.true
  %6 = load i32, i32* %n3.addr, align 4, !dbg !1118
  %cmp6 = icmp eq i32 %6, 0, !dbg !1121
  br i1 %cmp6, label %if.then7, label %if.else, !dbg !1122

if.then7:                                         ; preds = %if.then
  call void @llvm.dbg.declare(metadata i64* %nn, metadata !1123, metadata !DIExpression()), !dbg !1125
  %7 = load i32, i32* %n1.addr, align 4, !dbg !1126
  %conv8 = sext i32 %7 to i64, !dbg !1126
  store i64 %conv8, i64* %nn, align 8, !dbg !1125
  %8 = load i32, i32* %n2.addr, align 4, !dbg !1127
  %cmp9 = icmp ne i32 %8, 0, !dbg !1129
  br i1 %cmp9, label %if.then10, label %if.end, !dbg !1130

if.then10:                                        ; preds = %if.then7
  %9 = load i32, i32* %n2.addr, align 4, !dbg !1131
  %conv11 = sext i32 %9 to i64, !dbg !1131
  %10 = load i64, i64* %nn, align 8, !dbg !1133
  %mul = mul nsw i64 %10, %conv11, !dbg !1133
  store i64 %mul, i64* %nn, align 8, !dbg !1133
  br label %if.end, !dbg !1134

if.end:                                           ; preds = %if.then10, %if.then7
  %11 = load i64, i64* %nn, align 8, !dbg !1135
  %call12 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([38 x i8], [38 x i8]* @.str.2, i64 0, i64 0), i64 %11), !dbg !1136
  br label %if.end14, !dbg !1137

if.else:                                          ; preds = %if.then
  %12 = load i32, i32* %n1.addr, align 4, !dbg !1138
  %13 = load i32, i32* %n2.addr, align 4, !dbg !1140
  %14 = load i32, i32* %n3.addr, align 4, !dbg !1141
  %call13 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([44 x i8], [44 x i8]* @.str.3, i64 0, i64 0), i32 %12, i32 %13, i32 %14), !dbg !1142
  br label %if.end14

if.end14:                                         ; preds = %if.else, %if.end
  br label %if.end48, !dbg !1143

if.else15:                                        ; preds = %land.lhs.true, %entry
  call void @llvm.dbg.declare(metadata [16 x i8]* %size, metadata !1144, metadata !DIExpression()), !dbg !1149
  call void @llvm.dbg.declare(metadata i32* %j, metadata !1150, metadata !DIExpression()), !dbg !1151
  %15 = load i32, i32* %n2.addr, align 4, !dbg !1152
  %cmp16 = icmp eq i32 %15, 0, !dbg !1154
  br i1 %cmp16, label %land.lhs.true17, label %if.else45, !dbg !1155

land.lhs.true17:                                  ; preds = %if.else15
  %16 = load i32, i32* %n3.addr, align 4, !dbg !1156
  %cmp18 = icmp eq i32 %16, 0, !dbg !1157
  br i1 %cmp18, label %if.then19, label %if.else45, !dbg !1158

if.then19:                                        ; preds = %land.lhs.true17
  %17 = load i8*, i8** %name.addr, align 8, !dbg !1159
  %arrayidx20 = getelementptr inbounds i8, i8* %17, i64 0, !dbg !1159
  %18 = load i8, i8* %arrayidx20, align 1, !dbg !1159
  %conv21 = sext i8 %18 to i32, !dbg !1159
  %cmp22 = icmp eq i32 %conv21, 69, !dbg !1162
  br i1 %cmp22, label %land.lhs.true23, label %if.else42, !dbg !1163

land.lhs.true23:                                  ; preds = %if.then19
  %19 = load i8*, i8** %name.addr, align 8, !dbg !1164
  %arrayidx24 = getelementptr inbounds i8, i8* %19, i64 1, !dbg !1164
  %20 = load i8, i8* %arrayidx24, align 1, !dbg !1164
  %conv25 = sext i8 %20 to i32, !dbg !1164
  %cmp26 = icmp eq i32 %conv25, 80, !dbg !1165
  br i1 %cmp26, label %if.then27, label %if.else42, !dbg !1166

if.then27:                                        ; preds = %land.lhs.true23
  %arraydecay = getelementptr inbounds [16 x i8], [16 x i8]* %size, i64 0, i64 0, !dbg !1167
  %21 = load i32, i32* %n1.addr, align 4, !dbg !1169
  %conv28 = sitofp i32 %21 to double, !dbg !1169
  %call29 = call double @pow(double 2.000000e+00, double %conv28) #7, !dbg !1170
  %call30 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay, i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.4, i64 0, i64 0), double %call29) #7, !dbg !1171
  store i32 14, i32* %j, align 4, !dbg !1172
  %22 = load i32, i32* %j, align 4, !dbg !1173
  %idxprom = sext i32 %22 to i64, !dbg !1175
  %arrayidx31 = getelementptr inbounds [16 x i8], [16 x i8]* %size, i64 0, i64 %idxprom, !dbg !1175
  %23 = load i8, i8* %arrayidx31, align 1, !dbg !1175
  %conv32 = sext i8 %23 to i32, !dbg !1175
  %cmp33 = icmp eq i32 %conv32, 46, !dbg !1176
  br i1 %cmp33, label %if.then34, label %if.end37, !dbg !1177

if.then34:                                        ; preds = %if.then27
  %24 = load i32, i32* %j, align 4, !dbg !1178
  %idxprom35 = sext i32 %24 to i64, !dbg !1180
  %arrayidx36 = getelementptr inbounds [16 x i8], [16 x i8]* %size, i64 0, i64 %idxprom35, !dbg !1180
  store i8 32, i8* %arrayidx36, align 1, !dbg !1181
  %25 = load i32, i32* %j, align 4, !dbg !1182
  %dec = add nsw i32 %25, -1, !dbg !1182
  store i32 %dec, i32* %j, align 4, !dbg !1182
  br label %if.end37, !dbg !1183

if.end37:                                         ; preds = %if.then34, %if.then27
  %26 = load i32, i32* %j, align 4, !dbg !1184
  %add = add nsw i32 %26, 1, !dbg !1185
  %idxprom38 = sext i32 %add to i64, !dbg !1186
  %arrayidx39 = getelementptr inbounds [16 x i8], [16 x i8]* %size, i64 0, i64 %idxprom38, !dbg !1186
  store i8 0, i8* %arrayidx39, align 1, !dbg !1187
  %arraydecay40 = getelementptr inbounds [16 x i8], [16 x i8]* %size, i64 0, i64 0, !dbg !1188
  %call41 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([34 x i8], [34 x i8]* @.str.5, i64 0, i64 0), i8* %arraydecay40), !dbg !1189
  br label %if.end44, !dbg !1190

if.else42:                                        ; preds = %land.lhs.true23, %if.then19
  %27 = load i32, i32* %n1.addr, align 4, !dbg !1191
  %call43 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.6, i64 0, i64 0), i32 %27), !dbg !1193
  br label %if.end44

if.end44:                                         ; preds = %if.else42, %if.end37
  br label %if.end47, !dbg !1194

if.else45:                                        ; preds = %land.lhs.true17, %if.else15
  %28 = load i32, i32* %n1.addr, align 4, !dbg !1195
  %29 = load i32, i32* %n2.addr, align 4, !dbg !1197
  %30 = load i32, i32* %n3.addr, align 4, !dbg !1198
  %call46 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([42 x i8], [42 x i8]* @.str.7, i64 0, i64 0), i32 %28, i32 %29, i32 %30), !dbg !1199
  br label %if.end47

if.end47:                                         ; preds = %if.else45, %if.end44
  br label %if.end48

if.end48:                                         ; preds = %if.end47, %if.end14
  %31 = load i32, i32* %niter.addr, align 4, !dbg !1200
  %call49 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.8, i64 0, i64 0), i32 %31), !dbg !1201
  %32 = load double, double* %t.addr, align 8, !dbg !1202
  %call50 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([39 x i8], [39 x i8]* @.str.9, i64 0, i64 0), double %32), !dbg !1203
  %33 = load double, double* %mops.addr, align 8, !dbg !1204
  %call51 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([39 x i8], [39 x i8]* @.str.10, i64 0, i64 0), double %33), !dbg !1205
  %34 = load i8*, i8** %optype.addr, align 8, !dbg !1206
  %call52 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([25 x i8], [25 x i8]* @.str.11, i64 0, i64 0), i8* %34), !dbg !1207
  %35 = load i32, i32* %passed_verification.addr, align 4, !dbg !1208
  %cmp53 = icmp slt i32 %35, 0, !dbg !1210
  br i1 %cmp53, label %if.then54, label %if.else56, !dbg !1211

if.then54:                                        ; preds = %if.end48
  %call55 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([45 x i8], [45 x i8]* @.str.12, i64 0, i64 0)), !dbg !1212
  br label %if.end62, !dbg !1214

if.else56:                                        ; preds = %if.end48
  %36 = load i32, i32* %passed_verification.addr, align 4, !dbg !1215
  %tobool = icmp ne i32 %36, 0, !dbg !1215
  br i1 %tobool, label %if.then57, label %if.else59, !dbg !1217

if.then57:                                        ; preds = %if.else56
  %call58 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([45 x i8], [45 x i8]* @.str.13, i64 0, i64 0)), !dbg !1218
  br label %if.end61, !dbg !1220

if.else59:                                        ; preds = %if.else56
  %call60 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([45 x i8], [45 x i8]* @.str.14, i64 0, i64 0)), !dbg !1221
  br label %if.end61

if.end61:                                         ; preds = %if.else59, %if.then57
  br label %if.end62

if.end62:                                         ; preds = %if.end61, %if.then54
  %37 = load i8*, i8** %npbversion.addr, align 8, !dbg !1223
  %call63 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.15, i64 0, i64 0), i8* %37), !dbg !1224
  %38 = load i8*, i8** %compiletime.addr, align 8, !dbg !1225
  %call64 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.16, i64 0, i64 0), i8* %38), !dbg !1226
  %39 = load i8*, i8** %compilerversion.addr, align 8, !dbg !1227
  %call65 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.17, i64 0, i64 0), i8* %39), !dbg !1228
  %40 = load i8*, i8** %libversion.addr, align 8, !dbg !1229
  %call66 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.18, i64 0, i64 0), i8* %40), !dbg !1230
  %call67 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([20 x i8], [20 x i8]* @.str.19, i64 0, i64 0)), !dbg !1231
  %41 = load i8*, i8** %cc.addr, align 8, !dbg !1232
  %call68 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.20, i64 0, i64 0), i8* %41), !dbg !1233
  %42 = load i8*, i8** %clink.addr, align 8, !dbg !1234
  %call69 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.21, i64 0, i64 0), i8* %42), !dbg !1235
  %43 = load i8*, i8** %c_lib.addr, align 8, !dbg !1236
  %call70 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.22, i64 0, i64 0), i8* %43), !dbg !1237
  %44 = load i8*, i8** %c_inc.addr, align 8, !dbg !1238
  %call71 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.23, i64 0, i64 0), i8* %44), !dbg !1239
  %45 = load i8*, i8** %cflags.addr, align 8, !dbg !1240
  %call72 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.24, i64 0, i64 0), i8* %45), !dbg !1241
  %46 = load i8*, i8** %clinkflags.addr, align 8, !dbg !1242
  %call73 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.25, i64 0, i64 0), i8* %46), !dbg !1243
  %47 = load i8*, i8** %rand.addr, align 8, !dbg !1244
  %call74 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.26, i64 0, i64 0), i8* %47), !dbg !1245
  %call75 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.27, i64 0, i64 0)), !dbg !1246
  %48 = load i8*, i8** %cpu_device.addr, align 8, !dbg !1247
  %call76 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.28, i64 0, i64 0), i8* %48), !dbg !1248
  %49 = load i8*, i8** %gpu_device.addr, align 8, !dbg !1249
  %call77 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.29, i64 0, i64 0), i8* %49), !dbg !1250
  %call78 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.30, i64 0, i64 0)), !dbg !1251
  %50 = load i8*, i8** %gpu_config.addr, align 8, !dbg !1252
  %call79 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.31, i64 0, i64 0), i8* %50), !dbg !1253
  %call80 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.32, i64 0, i64 0)), !dbg !1254
  %call81 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([72 x i8], [72 x i8]* @.str.33, i64 0, i64 0)), !dbg !1255
  %call82 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([27 x i8], [27 x i8]* @.str.34, i64 0, i64 0)), !dbg !1256
  %call83 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([56 x i8], [56 x i8]* @.str.35, i64 0, i64 0)), !dbg !1257
  %call84 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([52 x i8], [52 x i8]* @.str.36, i64 0, i64 0)), !dbg !1258
  %call85 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([46 x i8], [46 x i8]* @.str.37, i64 0, i64 0)), !dbg !1259
  %call86 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.32, i64 0, i64 0)), !dbg !1260
  %call87 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([43 x i8], [43 x i8]* @.str.38, i64 0, i64 0)), !dbg !1261
  %call88 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([72 x i8], [72 x i8]* @.str.33, i64 0, i64 0)), !dbg !1262
  %call89 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.32, i64 0, i64 0)), !dbg !1263
  ret void, !dbg !1264
}

declare dso_local i32 @printf(i8*, ...) #3

; Function Attrs: nounwind
declare dso_local i32 @sprintf(i8*, i8*, ...) #4

; Function Attrs: nounwind
declare dso_local double @pow(double, double) #4

; Function Attrs: noinline norecurse uwtable
define dso_local i32 @main(i32 %argc, i8** %argv) #5 !dbg !1265 {
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
  call void @llvm.dbg.declare(metadata i32* %argc.addr, metadata !1268, metadata !DIExpression()), !dbg !1269
  store i8** %argv, i8*** %argv.addr, align 8
  call void @llvm.dbg.declare(metadata i8*** %argv.addr, metadata !1270, metadata !DIExpression()), !dbg !1271
  %call = call noalias i8* @malloc(i64 80) #7, !dbg !1272
  %0 = bitcast i8* %call to double*, !dbg !1273
  store double* %0, double** @_ZL1q, align 8, !dbg !1274
  call void @llvm.dbg.declare(metadata double* %Mops, metadata !1275, metadata !DIExpression()), !dbg !1276
  call void @llvm.dbg.declare(metadata double* %t1, metadata !1277, metadata !DIExpression()), !dbg !1278
  call void @llvm.dbg.declare(metadata double* %sx, metadata !1279, metadata !DIExpression()), !dbg !1280
  call void @llvm.dbg.declare(metadata double* %sy, metadata !1281, metadata !DIExpression()), !dbg !1282
  call void @llvm.dbg.declare(metadata double* %an, metadata !1283, metadata !DIExpression()), !dbg !1284
  call void @llvm.dbg.declare(metadata double* %gc, metadata !1285, metadata !DIExpression()), !dbg !1286
  call void @llvm.dbg.declare(metadata double* %sx_verify_value, metadata !1287, metadata !DIExpression()), !dbg !1288
  call void @llvm.dbg.declare(metadata double* %sy_verify_value, metadata !1289, metadata !DIExpression()), !dbg !1290
  call void @llvm.dbg.declare(metadata double* %sx_err, metadata !1291, metadata !DIExpression()), !dbg !1292
  call void @llvm.dbg.declare(metadata double* %sy_err, metadata !1293, metadata !DIExpression()), !dbg !1294
  call void @llvm.dbg.declare(metadata i32* %i, metadata !1295, metadata !DIExpression()), !dbg !1296
  call void @llvm.dbg.declare(metadata i32* %j, metadata !1297, metadata !DIExpression()), !dbg !1298
  call void @llvm.dbg.declare(metadata i32* %nit, metadata !1299, metadata !DIExpression()), !dbg !1300
  call void @llvm.dbg.declare(metadata i32* %block, metadata !1301, metadata !DIExpression()), !dbg !1302
  call void @llvm.dbg.declare(metadata i32* %verified, metadata !1303, metadata !DIExpression()), !dbg !1306
  call void @llvm.dbg.declare(metadata [16 x i8]* %size, metadata !1307, metadata !DIExpression()), !dbg !1308
  %arraydecay = getelementptr inbounds [16 x i8], [16 x i8]* %size, i64 0, i64 0, !dbg !1309
  %call1 = call double @pow(double 2.000000e+00, double 2.900000e+01) #7, !dbg !1310
  %call2 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay, i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.39, i64 0, i64 0), double %call1) #7, !dbg !1311
  store i32 14, i32* %j, align 4, !dbg !1312
  %1 = load i32, i32* %j, align 4, !dbg !1313
  %idxprom = sext i32 %1 to i64, !dbg !1315
  %arrayidx = getelementptr inbounds [16 x i8], [16 x i8]* %size, i64 0, i64 %idxprom, !dbg !1315
  %2 = load i8, i8* %arrayidx, align 1, !dbg !1315
  %conv = sext i8 %2 to i32, !dbg !1315
  %cmp = icmp eq i32 %conv, 46, !dbg !1316
  br i1 %cmp, label %if.then, label %if.end, !dbg !1317

if.then:                                          ; preds = %entry
  %3 = load i32, i32* %j, align 4, !dbg !1318
  %dec = add nsw i32 %3, -1, !dbg !1318
  store i32 %dec, i32* %j, align 4, !dbg !1318
  br label %if.end, !dbg !1320

if.end:                                           ; preds = %if.then, %entry
  %4 = load i32, i32* %j, align 4, !dbg !1321
  %add = add nsw i32 %4, 1, !dbg !1322
  %idxprom3 = sext i32 %add to i64, !dbg !1323
  %arrayidx4 = getelementptr inbounds [16 x i8], [16 x i8]* %size, i64 0, i64 %idxprom3, !dbg !1323
  store i8 0, i8* %arrayidx4, align 1, !dbg !1324
  %call5 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([65 x i8], [65 x i8]* @.str.40, i64 0, i64 0)), !dbg !1325
  %arraydecay6 = getelementptr inbounds [16 x i8], [16 x i8]* %size, i64 0, i64 0, !dbg !1326
  %call7 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([43 x i8], [43 x i8]* @.str.41, i64 0, i64 0), i8* %arraydecay6), !dbg !1327
  store i32 0, i32* %verified, align 4, !dbg !1328
  store double 0x41D2309CE5400000, double* %t1, align 8, !dbg !1329
  store i32 0, i32* %i, align 4, !dbg !1330
  br label %for.cond, !dbg !1332

for.cond:                                         ; preds = %for.inc, %if.end
  %5 = load i32, i32* %i, align 4, !dbg !1333
  %cmp8 = icmp slt i32 %5, 17, !dbg !1335
  br i1 %cmp8, label %for.body, label %for.end, !dbg !1336

for.body:                                         ; preds = %for.cond
  %6 = load double, double* %t1, align 8, !dbg !1337
  %call9 = call double @_Z6randlcPdd(double* %t1, double %6), !dbg !1339
  br label %for.inc, !dbg !1340

for.inc:                                          ; preds = %for.body
  %7 = load i32, i32* %i, align 4, !dbg !1341
  %inc = add nsw i32 %7, 1, !dbg !1341
  store i32 %inc, i32* %i, align 4, !dbg !1341
  br label %for.cond, !dbg !1342, !llvm.loop !1343

for.end:                                          ; preds = %for.cond
  %8 = load double, double* %t1, align 8, !dbg !1345
  store double %8, double* %an, align 8, !dbg !1346
  store double 0.000000e+00, double* %gc, align 8, !dbg !1347
  store double 0.000000e+00, double* %sx, align 8, !dbg !1348
  store double 0.000000e+00, double* %sy, align 8, !dbg !1349
  store i32 0, i32* %i, align 4, !dbg !1350
  br label %for.cond10, !dbg !1352

for.cond10:                                       ; preds = %for.inc15, %for.end
  %9 = load i32, i32* %i, align 4, !dbg !1353
  %cmp11 = icmp slt i32 %9, 10, !dbg !1355
  br i1 %cmp11, label %for.body12, label %for.end17, !dbg !1356

for.body12:                                       ; preds = %for.cond10
  %10 = load double*, double** @_ZL1q, align 8, !dbg !1357
  %11 = load i32, i32* %i, align 4, !dbg !1359
  %idxprom13 = sext i32 %11 to i64, !dbg !1357
  %arrayidx14 = getelementptr inbounds double, double* %10, i64 %idxprom13, !dbg !1357
  store double 0.000000e+00, double* %arrayidx14, align 8, !dbg !1360
  br label %for.inc15, !dbg !1361

for.inc15:                                        ; preds = %for.body12
  %12 = load i32, i32* %i, align 4, !dbg !1362
  %inc16 = add nsw i32 %12, 1, !dbg !1362
  store i32 %inc16, i32* %i, align 4, !dbg !1362
  br label %for.cond10, !dbg !1363, !llvm.loop !1364

for.end17:                                        ; preds = %for.cond10
  call void @_ZL9setup_gpuv(), !dbg !1366
  %13 = load i32, i32* @blocks_per_grid, align 4, !dbg !1367
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp, i32 %13, i32 1, i32 1), !dbg !1367
  %14 = load i32, i32* @threads_per_block, align 4, !dbg !1368
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp18, i32 %14, i32 1, i32 1), !dbg !1368
  %15 = bitcast { i64, i32 }* %agg.tmp.coerce to i8*, !dbg !1369
  %16 = bitcast %struct.dim3* %agg.tmp to i8*, !dbg !1369
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %15, i8* align 4 %16, i64 12, i1 false), !dbg !1369
  %17 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 0, !dbg !1369
  %18 = load i64, i64* %17, align 4, !dbg !1369
  %19 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 1, !dbg !1369
  %20 = load i32, i32* %19, align 4, !dbg !1369
  %21 = bitcast { i64, i32 }* %agg.tmp18.coerce to i8*, !dbg !1369
  %22 = bitcast %struct.dim3* %agg.tmp18 to i8*, !dbg !1369
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %21, i8* align 4 %22, i64 12, i1 false), !dbg !1369
  %23 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp18.coerce, i32 0, i32 0, !dbg !1369
  %24 = load i64, i64* %23, align 4, !dbg !1369
  %25 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp18.coerce, i32 0, i32 1, !dbg !1369
  %26 = load i32, i32* %25, align 4, !dbg !1369
  %call19 = call i32 @cudaConfigureCall(i64 %18, i32 %20, i64 %24, i32 %26, i64 0, %struct.CUstream_st* null), !dbg !1369
  %tobool = icmp ne i32 %call19, 0, !dbg !1369
  br i1 %tobool, label %kcall.end, label %kcall.configok, !dbg !1370

kcall.configok:                                   ; preds = %for.end17
  %27 = load double*, double** @q_device, align 8, !dbg !1371
  %28 = load double*, double** @sx_device, align 8, !dbg !1372
  %29 = load double*, double** @sy_device, align 8, !dbg !1373
  %30 = load double, double* %an, align 8, !dbg !1374
  call void @_Z10gpu_kernelPdS_S_d(double* %27, double* %28, double* %29, double %30), !dbg !1370
  br label %kcall.end, !dbg !1370

kcall.end:                                        ; preds = %kcall.configok, %for.end17
  %31 = load double*, double** @q_host, align 8, !dbg !1375
  %32 = bitcast double* %31 to i8*, !dbg !1375
  %33 = load double*, double** @q_device, align 8, !dbg !1376
  %34 = bitcast double* %33 to i8*, !dbg !1376
  %35 = load i64, i64* @size_q, align 8, !dbg !1377
  %call20 = call i32 @cudaMemcpy(i8* %32, i8* %34, i64 %35, i32 2), !dbg !1378
  %36 = load double*, double** @sx_host, align 8, !dbg !1379
  %37 = bitcast double* %36 to i8*, !dbg !1379
  %38 = load double*, double** @sx_device, align 8, !dbg !1380
  %39 = bitcast double* %38 to i8*, !dbg !1380
  %40 = load i64, i64* @size_sx, align 8, !dbg !1381
  %call21 = call i32 @cudaMemcpy(i8* %37, i8* %39, i64 %40, i32 2), !dbg !1382
  %41 = load double*, double** @sy_host, align 8, !dbg !1383
  %42 = bitcast double* %41 to i8*, !dbg !1383
  %43 = load double*, double** @sy_device, align 8, !dbg !1384
  %44 = bitcast double* %43 to i8*, !dbg !1384
  %45 = load i64, i64* @size_sy, align 8, !dbg !1385
  %call22 = call i32 @cudaMemcpy(i8* %42, i8* %44, i64 %45, i32 2), !dbg !1386
  store i32 0, i32* %block, align 4, !dbg !1387
  br label %for.cond23, !dbg !1389

for.cond23:                                       ; preds = %for.inc44, %kcall.end
  %46 = load i32, i32* %block, align 4, !dbg !1390
  %47 = load i32, i32* @blocks_per_grid, align 4, !dbg !1392
  %cmp24 = icmp slt i32 %46, %47, !dbg !1393
  br i1 %cmp24, label %for.body25, label %for.end46, !dbg !1394

for.body25:                                       ; preds = %for.cond23
  store i32 0, i32* %i, align 4, !dbg !1395
  br label %for.cond26, !dbg !1398

for.cond26:                                       ; preds = %for.inc35, %for.body25
  %48 = load i32, i32* %i, align 4, !dbg !1399
  %cmp27 = icmp slt i32 %48, 10, !dbg !1401
  br i1 %cmp27, label %for.body28, label %for.end37, !dbg !1402

for.body28:                                       ; preds = %for.cond26
  %49 = load double*, double** @q_host, align 8, !dbg !1403
  %50 = load i32, i32* %block, align 4, !dbg !1405
  %mul = mul nsw i32 %50, 10, !dbg !1406
  %51 = load i32, i32* %i, align 4, !dbg !1407
  %add29 = add nsw i32 %mul, %51, !dbg !1408
  %idxprom30 = sext i32 %add29 to i64, !dbg !1403
  %arrayidx31 = getelementptr inbounds double, double* %49, i64 %idxprom30, !dbg !1403
  %52 = load double, double* %arrayidx31, align 8, !dbg !1403
  %53 = load double*, double** @_ZL1q, align 8, !dbg !1409
  %54 = load i32, i32* %i, align 4, !dbg !1410
  %idxprom32 = sext i32 %54 to i64, !dbg !1409
  %arrayidx33 = getelementptr inbounds double, double* %53, i64 %idxprom32, !dbg !1409
  %55 = load double, double* %arrayidx33, align 8, !dbg !1411
  %add34 = fadd contract double %55, %52, !dbg !1411
  store double %add34, double* %arrayidx33, align 8, !dbg !1411
  br label %for.inc35, !dbg !1412

for.inc35:                                        ; preds = %for.body28
  %56 = load i32, i32* %i, align 4, !dbg !1413
  %inc36 = add nsw i32 %56, 1, !dbg !1413
  store i32 %inc36, i32* %i, align 4, !dbg !1413
  br label %for.cond26, !dbg !1414, !llvm.loop !1415

for.end37:                                        ; preds = %for.cond26
  %57 = load double*, double** @sx_host, align 8, !dbg !1417
  %58 = load i32, i32* %block, align 4, !dbg !1418
  %idxprom38 = sext i32 %58 to i64, !dbg !1417
  %arrayidx39 = getelementptr inbounds double, double* %57, i64 %idxprom38, !dbg !1417
  %59 = load double, double* %arrayidx39, align 8, !dbg !1417
  %60 = load double, double* %sx, align 8, !dbg !1419
  %add40 = fadd contract double %60, %59, !dbg !1419
  store double %add40, double* %sx, align 8, !dbg !1419
  %61 = load double*, double** @sy_host, align 8, !dbg !1420
  %62 = load i32, i32* %block, align 4, !dbg !1421
  %idxprom41 = sext i32 %62 to i64, !dbg !1420
  %arrayidx42 = getelementptr inbounds double, double* %61, i64 %idxprom41, !dbg !1420
  %63 = load double, double* %arrayidx42, align 8, !dbg !1420
  %64 = load double, double* %sy, align 8, !dbg !1422
  %add43 = fadd contract double %64, %63, !dbg !1422
  store double %add43, double* %sy, align 8, !dbg !1422
  br label %for.inc44, !dbg !1423

for.inc44:                                        ; preds = %for.end37
  %65 = load i32, i32* %block, align 4, !dbg !1424
  %inc45 = add nsw i32 %65, 1, !dbg !1424
  store i32 %inc45, i32* %block, align 4, !dbg !1424
  br label %for.cond23, !dbg !1425, !llvm.loop !1426

for.end46:                                        ; preds = %for.cond23
  store i32 0, i32* %i, align 4, !dbg !1428
  br label %for.cond47, !dbg !1430

for.cond47:                                       ; preds = %for.inc53, %for.end46
  %66 = load i32, i32* %i, align 4, !dbg !1431
  %cmp48 = icmp slt i32 %66, 10, !dbg !1433
  br i1 %cmp48, label %for.body49, label %for.end55, !dbg !1434

for.body49:                                       ; preds = %for.cond47
  %67 = load double*, double** @_ZL1q, align 8, !dbg !1435
  %68 = load i32, i32* %i, align 4, !dbg !1437
  %idxprom50 = sext i32 %68 to i64, !dbg !1435
  %arrayidx51 = getelementptr inbounds double, double* %67, i64 %idxprom50, !dbg !1435
  %69 = load double, double* %arrayidx51, align 8, !dbg !1435
  %70 = load double, double* %gc, align 8, !dbg !1438
  %add52 = fadd contract double %70, %69, !dbg !1438
  store double %add52, double* %gc, align 8, !dbg !1438
  br label %for.inc53, !dbg !1439

for.inc53:                                        ; preds = %for.body49
  %71 = load i32, i32* %i, align 4, !dbg !1440
  %inc54 = add nsw i32 %71, 1, !dbg !1440
  store i32 %inc54, i32* %i, align 4, !dbg !1440
  br label %for.cond47, !dbg !1441, !llvm.loop !1442

for.end55:                                        ; preds = %for.cond47
  store i32 0, i32* %nit, align 4, !dbg !1444
  store i32 1, i32* %verified, align 4, !dbg !1445
  store double 0xC0B0C7E00ADACEF8, double* %sx_verify_value, align 8, !dbg !1446
  store double 0xC0CEDFA9B1BE31DC, double* %sy_verify_value, align 8, !dbg !1451
  %72 = load i32, i32* %verified, align 4, !dbg !1452
  %tobool56 = icmp ne i32 %72, 0, !dbg !1452
  br i1 %tobool56, label %if.then57, label %if.end63, !dbg !1454

if.then57:                                        ; preds = %for.end55
  %73 = load double, double* %sx, align 8, !dbg !1455
  %74 = load double, double* %sx_verify_value, align 8, !dbg !1457
  %sub = fsub contract double %73, %74, !dbg !1458
  %75 = load double, double* %sx_verify_value, align 8, !dbg !1459
  %div = fdiv double %sub, %75, !dbg !1460
  %76 = call double @llvm.fabs.f64(double %div), !dbg !1461
  store double %76, double* %sx_err, align 8, !dbg !1462
  %77 = load double, double* %sy, align 8, !dbg !1463
  %78 = load double, double* %sy_verify_value, align 8, !dbg !1464
  %sub58 = fsub contract double %77, %78, !dbg !1465
  %79 = load double, double* %sy_verify_value, align 8, !dbg !1466
  %div59 = fdiv double %sub58, %79, !dbg !1467
  %80 = call double @llvm.fabs.f64(double %div59), !dbg !1468
  store double %80, double* %sy_err, align 8, !dbg !1469
  %81 = load double, double* %sx_err, align 8, !dbg !1470
  %cmp60 = fcmp ole double %81, 1.000000e-08, !dbg !1471
  br i1 %cmp60, label %land.rhs, label %land.end, !dbg !1472

land.rhs:                                         ; preds = %if.then57
  %82 = load double, double* %sy_err, align 8, !dbg !1473
  %cmp61 = fcmp ole double %82, 1.000000e-08, !dbg !1474
  br label %land.end

land.end:                                         ; preds = %land.rhs, %if.then57
  %83 = phi i1 [ false, %if.then57 ], [ %cmp61, %land.rhs ], !dbg !1475
  %conv62 = zext i1 %83 to i32, !dbg !1476
  store i32 %conv62, i32* %verified, align 4, !dbg !1477
  br label %if.end63, !dbg !1478

if.end63:                                         ; preds = %land.end, %for.end55
  %call64 = call double @pow(double 2.000000e+00, double 2.900000e+01) #7, !dbg !1479
  %div65 = fdiv double %call64, 1.000000e+06, !dbg !1480
  store double %div65, double* %Mops, align 8, !dbg !1481
  %call66 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([26 x i8], [26 x i8]* @.str.42, i64 0, i64 0)), !dbg !1482
  %call67 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([19 x i8], [19 x i8]* @.str.43, i64 0, i64 0)), !dbg !1483
  %call68 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([12 x i8], [12 x i8]* @.str.44, i64 0, i64 0), i32 28), !dbg !1484
  %84 = load double, double* %gc, align 8, !dbg !1485
  %call69 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([30 x i8], [30 x i8]* @.str.45, i64 0, i64 0), double %84), !dbg !1486
  %85 = load double, double* %sx, align 8, !dbg !1487
  %86 = load double, double* %sy, align 8, !dbg !1488
  %call70 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([25 x i8], [25 x i8]* @.str.46, i64 0, i64 0), double %85, double %86), !dbg !1489
  %call71 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.47, i64 0, i64 0)), !dbg !1490
  store i32 0, i32* %i, align 4, !dbg !1491
  br label %for.cond72, !dbg !1493

for.cond72:                                       ; preds = %for.inc78, %if.end63
  %87 = load i32, i32* %i, align 4, !dbg !1494
  %cmp73 = icmp slt i32 %87, 10, !dbg !1496
  br i1 %cmp73, label %for.body74, label %for.end80, !dbg !1497

for.body74:                                       ; preds = %for.cond72
  %88 = load i32, i32* %i, align 4, !dbg !1498
  %89 = load double*, double** @_ZL1q, align 8, !dbg !1500
  %90 = load i32, i32* %i, align 4, !dbg !1501
  %idxprom75 = sext i32 %90 to i64, !dbg !1500
  %arrayidx76 = getelementptr inbounds double, double* %89, i64 %idxprom75, !dbg !1500
  %91 = load double, double* %arrayidx76, align 8, !dbg !1500
  %call77 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.48, i64 0, i64 0), i32 %88, double %91), !dbg !1502
  br label %for.inc78, !dbg !1503

for.inc78:                                        ; preds = %for.body74
  %92 = load i32, i32* %i, align 4, !dbg !1504
  %inc79 = add nsw i32 %92, 1, !dbg !1504
  store i32 %inc79, i32* %i, align 4, !dbg !1504
  br label %for.cond72, !dbg !1505, !llvm.loop !1506

for.end80:                                        ; preds = %for.cond72
  call void @llvm.dbg.declare(metadata [256 x i8]* %gpu_config, metadata !1508, metadata !DIExpression()), !dbg !1509
  call void @llvm.dbg.declare(metadata [2048 x i8]* %gpu_config_string, metadata !1510, metadata !DIExpression()), !dbg !1514
  %arraydecay81 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1515
  %call82 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay81, i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.49, i64 0, i64 0), i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.50, i64 0, i64 0), i8* getelementptr inbounds ([18 x i8], [18 x i8]* @.str.51, i64 0, i64 0)) #7, !dbg !1516
  %arraydecay83 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !1517
  %arraydecay84 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1518
  %call85 = call i8* @strcpy(i8* %arraydecay83, i8* %arraydecay84) #7, !dbg !1519
  %arraydecay86 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1520
  %93 = load i32, i32* @threads_per_block, align 4, !dbg !1521
  %call87 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay86, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.52, i64 0, i64 0), i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.53, i64 0, i64 0), i32 %93) #7, !dbg !1522
  %arraydecay88 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !1523
  %arraydecay89 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1524
  %call90 = call i8* @strcat(i8* %arraydecay88, i8* %arraydecay89) #7, !dbg !1525
  %94 = load i32, i32* %nit, align 4, !dbg !1526
  %95 = load double, double* %Mops, align 8, !dbg !1527
  %96 = load i32, i32* %verified, align 4, !dbg !1528
  %arraydecay91 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !1529
  call void @_Z15c_print_resultsPcciiiiddS_iS_S_S_S_S_S_S_S_S_S_S_S_S_S_(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.54, i64 0, i64 0), i8 signext 65, i32 29, i32 0, i32 0, i32 %94, double 0.000000e+00, double %95, i8* getelementptr inbounds ([25 x i8], [25 x i8]* @.str.55, i64 0, i64 0), i32 %96, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.56, i64 0, i64 0), i8* getelementptr inbounds ([12 x i8], [12 x i8]* @.str.57, i64 0, i64 0), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.58, i64 0, i64 0), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.58, i64 0, i64 0), i8* getelementptr inbounds ([42 x i8], [42 x i8]* @.str.59, i64 0, i64 0), i8* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 0, i64 0), i8* %arraydecay91, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.60, i64 0, i64 0), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.61, i64 0, i64 0), i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.62, i64 0, i64 0), i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.63, i64 0, i64 0), i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.64, i64 0, i64 0), i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.64, i64 0, i64 0), i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.65, i64 0, i64 0)), !dbg !1530
  call void @_ZL11release_gpuv(), !dbg !1531
  ret i32 0, !dbg !1532
}

; Function Attrs: nounwind
declare dso_local noalias i8* @malloc(i64) #4

; Function Attrs: noinline uwtable
define internal void @_ZL9setup_gpuv() #2 !dbg !1533 {
entry:
  store i32 32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !1534
  store i32 32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !1535
  %0 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !1536
  %cmp = icmp sle i32 32, %0, !dbg !1538
  br i1 %cmp, label %if.then, label %if.else, !dbg !1539

if.then:                                          ; preds = %entry
  store i32 32, i32* @threads_per_block, align 4, !dbg !1540
  br label %if.end, !dbg !1542

if.else:                                          ; preds = %entry
  %1 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !1543
  store i32 %1, i32* @threads_per_block, align 4, !dbg !1545
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %2 = load i32, i32* @threads_per_block, align 4, !dbg !1546
  %conv = sitofp i32 %2 to double, !dbg !1546
  %div = fdiv double 4.096000e+03, %conv, !dbg !1547
  %3 = call double @llvm.ceil.f64(double %div), !dbg !1548
  %conv1 = fptosi double %3 to i32, !dbg !1549
  store i32 %conv1, i32* @blocks_per_grid, align 4, !dbg !1550
  %4 = load i32, i32* @blocks_per_grid, align 4, !dbg !1551
  %mul = mul nsw i32 %4, 10, !dbg !1552
  %conv2 = sext i32 %mul to i64, !dbg !1551
  %mul3 = mul i64 %conv2, 8, !dbg !1553
  store i64 %mul3, i64* @size_q, align 8, !dbg !1554
  %5 = load i32, i32* @blocks_per_grid, align 4, !dbg !1555
  %conv4 = sext i32 %5 to i64, !dbg !1555
  %mul5 = mul i64 %conv4, 8, !dbg !1556
  store i64 %mul5, i64* @size_sx, align 8, !dbg !1557
  %6 = load i32, i32* @blocks_per_grid, align 4, !dbg !1558
  %conv6 = sext i32 %6 to i64, !dbg !1558
  %mul7 = mul i64 %conv6, 8, !dbg !1559
  store i64 %mul7, i64* @size_sy, align 8, !dbg !1560
  %7 = load i64, i64* @size_q, align 8, !dbg !1561
  %call = call noalias i8* @malloc(i64 %7) #7, !dbg !1562
  %8 = bitcast i8* %call to double*, !dbg !1563
  store double* %8, double** @q_host, align 8, !dbg !1564
  %9 = load i64, i64* @size_sx, align 8, !dbg !1565
  %call8 = call noalias i8* @malloc(i64 %9) #7, !dbg !1566
  %10 = bitcast i8* %call8 to double*, !dbg !1567
  store double* %10, double** @sx_host, align 8, !dbg !1568
  %11 = load i64, i64* @size_sy, align 8, !dbg !1569
  %call9 = call noalias i8* @malloc(i64 %11) #7, !dbg !1570
  %12 = bitcast i8* %call9 to double*, !dbg !1571
  store double* %12, double** @sy_host, align 8, !dbg !1572
  %13 = load i64, i64* @size_q, align 8, !dbg !1573
  %call10 = call i32 @_ZL10cudaMallocIdE9cudaErrorPPT_m(double** @q_device, i64 %13), !dbg !1574
  %14 = load i64, i64* @size_sx, align 8, !dbg !1575
  %call11 = call i32 @_ZL10cudaMallocIdE9cudaErrorPPT_m(double** @sx_device, i64 %14), !dbg !1576
  %15 = load i64, i64* @size_sy, align 8, !dbg !1577
  %call12 = call i32 @_ZL10cudaMallocIdE9cudaErrorPPT_m(double** @sy_device, i64 %15), !dbg !1578
  ret void, !dbg !1579
}

declare dso_local i32 @cudaConfigureCall(i64, i32, i64, i32, i64, %struct.CUstream_st*) #3

; Function Attrs: noinline nounwind uwtable
define linkonce_odr dso_local void @_ZN4dim3C2Ejjj(%struct.dim3* %this, i32 %vx, i32 %vy, i32 %vz) unnamed_addr #0 comdat align 2 !dbg !1580 {
entry:
  %this.addr = alloca %struct.dim3*, align 8
  %vx.addr = alloca i32, align 4
  %vy.addr = alloca i32, align 4
  %vz.addr = alloca i32, align 4
  store %struct.dim3* %this, %struct.dim3** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dim3** %this.addr, metadata !1603, metadata !DIExpression()), !dbg !1605
  store i32 %vx, i32* %vx.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %vx.addr, metadata !1606, metadata !DIExpression()), !dbg !1607
  store i32 %vy, i32* %vy.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %vy.addr, metadata !1608, metadata !DIExpression()), !dbg !1609
  store i32 %vz, i32* %vz.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %vz.addr, metadata !1610, metadata !DIExpression()), !dbg !1611
  %this1 = load %struct.dim3*, %struct.dim3** %this.addr, align 8
  %x = getelementptr inbounds %struct.dim3, %struct.dim3* %this1, i32 0, i32 0, !dbg !1612
  %0 = load i32, i32* %vx.addr, align 4, !dbg !1613
  store i32 %0, i32* %x, align 4, !dbg !1612
  %y = getelementptr inbounds %struct.dim3, %struct.dim3* %this1, i32 0, i32 1, !dbg !1614
  %1 = load i32, i32* %vy.addr, align 4, !dbg !1615
  store i32 %1, i32* %y, align 4, !dbg !1614
  %z = getelementptr inbounds %struct.dim3, %struct.dim3* %this1, i32 0, i32 2, !dbg !1616
  %2 = load i32, i32* %vz.addr, align 4, !dbg !1617
  store i32 %2, i32* %z, align 4, !dbg !1616
  ret void, !dbg !1618
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1 immarg) #6

; Function Attrs: noinline uwtable
define dso_local void @_Z10gpu_kernelPdS_S_d(double* %q_global, double* %sx_global, double* %sy_global, double %an) #2 !dbg !1619 {
entry:
  %q_global.addr = alloca double*, align 8
  %sx_global.addr = alloca double*, align 8
  %sy_global.addr = alloca double*, align 8
  %an.addr = alloca double, align 8
  store double* %q_global, double** %q_global.addr, align 8
  call void @llvm.dbg.declare(metadata double** %q_global.addr, metadata !1622, metadata !DIExpression()), !dbg !1623
  store double* %sx_global, double** %sx_global.addr, align 8
  call void @llvm.dbg.declare(metadata double** %sx_global.addr, metadata !1624, metadata !DIExpression()), !dbg !1625
  store double* %sy_global, double** %sy_global.addr, align 8
  call void @llvm.dbg.declare(metadata double** %sy_global.addr, metadata !1626, metadata !DIExpression()), !dbg !1627
  store double %an, double* %an.addr, align 8
  call void @llvm.dbg.declare(metadata double* %an.addr, metadata !1628, metadata !DIExpression()), !dbg !1629
  %0 = bitcast double** %q_global.addr to i8*, !dbg !1630
  %1 = call i32 @cudaSetupArgument(i8* %0, i64 8, i64 0), !dbg !1630
  %2 = icmp eq i32 %1, 0, !dbg !1630
  br i1 %2, label %setup.next, label %setup.end, !dbg !1630

setup.next:                                       ; preds = %entry
  %3 = bitcast double** %sx_global.addr to i8*, !dbg !1630
  %4 = call i32 @cudaSetupArgument(i8* %3, i64 8, i64 8), !dbg !1630
  %5 = icmp eq i32 %4, 0, !dbg !1630
  br i1 %5, label %setup.next1, label %setup.end, !dbg !1630

setup.next1:                                      ; preds = %setup.next
  %6 = bitcast double** %sy_global.addr to i8*, !dbg !1630
  %7 = call i32 @cudaSetupArgument(i8* %6, i64 8, i64 16), !dbg !1630
  %8 = icmp eq i32 %7, 0, !dbg !1630
  br i1 %8, label %setup.next2, label %setup.end, !dbg !1630

setup.next2:                                      ; preds = %setup.next1
  %9 = bitcast double* %an.addr to i8*, !dbg !1630
  %10 = call i32 @cudaSetupArgument(i8* %9, i64 8, i64 24), !dbg !1630
  %11 = icmp eq i32 %10, 0, !dbg !1630
  br i1 %11, label %setup.next3, label %setup.end, !dbg !1630

setup.next3:                                      ; preds = %setup.next2
  %12 = call i32 @cudaLaunch(i8* bitcast (void (double*, double*, double*, double)* @_Z10gpu_kernelPdS_S_d to i8*)), !dbg !1630
  br label %setup.end, !dbg !1630

setup.end:                                        ; preds = %setup.next3, %setup.next2, %setup.next1, %setup.next, %entry
  ret void, !dbg !1631
}

declare dso_local i32 @cudaMemcpy(i8*, i8*, i64, i32) #3

; Function Attrs: nounwind readnone speculatable
declare double @llvm.fabs.f64(double) #1

; Function Attrs: nounwind
declare dso_local i8* @strcpy(i8*, i8*) #4

; Function Attrs: nounwind
declare dso_local i8* @strcat(i8*, i8*) #4

; Function Attrs: noinline uwtable
define internal void @_ZL11release_gpuv() #2 !dbg !1632 {
entry:
  %0 = load double*, double** @q_device, align 8, !dbg !1633
  %1 = bitcast double* %0 to i8*, !dbg !1633
  %call = call i32 @cudaFree(i8* %1), !dbg !1634
  %2 = load double*, double** @sx_device, align 8, !dbg !1635
  %3 = bitcast double* %2 to i8*, !dbg !1635
  %call1 = call i32 @cudaFree(i8* %3), !dbg !1636
  %4 = load double*, double** @sy_device, align 8, !dbg !1637
  %5 = bitcast double* %4 to i8*, !dbg !1637
  %call2 = call i32 @cudaFree(i8* %5), !dbg !1638
  ret void, !dbg !1639
}

declare dso_local i32 @cudaSetupArgument(i8*, i64, i64)

declare dso_local i32 @cudaLaunch(i8*)

declare dso_local i32 @cudaFree(i8*) #3

; Function Attrs: nounwind readnone speculatable
declare double @llvm.ceil.f64(double) #1

; Function Attrs: noinline uwtable
define internal i32 @_ZL10cudaMallocIdE9cudaErrorPPT_m(double** %devPtr, i64 %size) #2 !dbg !1640 {
entry:
  %devPtr.addr = alloca double**, align 8
  %size.addr = alloca i64, align 8
  store double** %devPtr, double*** %devPtr.addr, align 8
  call void @llvm.dbg.declare(metadata double*** %devPtr.addr, metadata !1648, metadata !DIExpression()), !dbg !1649
  store i64 %size, i64* %size.addr, align 8
  call void @llvm.dbg.declare(metadata i64* %size.addr, metadata !1650, metadata !DIExpression()), !dbg !1651
  %0 = load double**, double*** %devPtr.addr, align 8, !dbg !1652
  %1 = bitcast double** %0 to i8*, !dbg !1652
  %2 = bitcast i8* %1 to i8**, !dbg !1653
  %3 = load i64, i64* %size.addr, align 8, !dbg !1654
  %call = call i32 @cudaMalloc(i8** %2, i64 %3), !dbg !1655
  ret i32 %call, !dbg !1656
}

declare dso_local i32 @cudaMalloc(i8**, i64) #3

attributes #0 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { noinline uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { noinline norecurse uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { argmemonly nounwind }
attributes #7 = { nounwind }

!llvm.module.flags = !{!961, !962, !963, !964}
!llvm.dbg.cu = !{!2}
!llvm.ident = !{!965}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "q_host", scope: !2, file: !3, line: 84, type: !98, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 9.0.0 (https://github.com/yebinchon/llvm-project/ a1efa594106d738d0b74c9e4e2b9b779eb8b7d25)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !96, globals: !104, imports: !209, nameTableKind: None)
!3 = !DIFile(filename: "ep.cu", directory: "/scratch/ah7226/NPB-GPU/CUDA/EP")
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
!122 = !DIFile(filename: "/u/NAS_SCRATCH/ah7226/tulip/llvm-install-tulip/lib/clang/9.0.0/include/stddef.h", directory: "")
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
!213 = !DIFile(filename: "/u/NAS_SCRATCH/ah7226/tulip/llvm-install-tulip/lib/clang/9.0.0/include/__clang_cuda_math_forward_declares.h", directory: "")
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
!462 = !DICompositeType(tag: DW_TAG_structure_type, file: !402, line: 59, flags: DIFlagFwdDecl, identifier: "_ZTS5div_t")
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
!787 = !DICompositeType(tag: DW_TAG_structure_type, name: "_IO_FILE", file: !788, line: 49, flags: DIFlagFwdDecl, identifier: "_ZTS8_IO_FILE")
!788 = !DIFile(filename: "/usr/include/x86_64-linux-gnu/bits/types/struct_FILE.h", directory: "")
!789 = !DIFile(filename: "/usr/lib/gcc/x86_64-linux-gnu/11/../../../../include/c++/11/cstdio", directory: "")
!790 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !211, entity: !791, file: !789, line: 99)
!791 = !DIDerivedType(tag: DW_TAG_typedef, name: "fpos_t", file: !792, line: 84, baseType: !793)
!792 = !DIFile(filename: "/usr/include/stdio.h", directory: "")
!793 = !DIDerivedType(tag: DW_TAG_typedef, name: "__fpos_t", file: !794, line: 14, baseType: !795)
!794 = !DIFile(filename: "/usr/include/x86_64-linux-gnu/bits/types/__fpos_t.h", directory: "")
!795 = !DICompositeType(tag: DW_TAG_structure_type, name: "_G_fpos_t", file: !794, line: 10, flags: DIFlagFwdDecl, identifier: "_ZTS9_G_fpos_t")
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
!931 = !DICompositeType(tag: DW_TAG_structure_type, name: "__va_list_tag", file: !3, flags: DIFlagFwdDecl, identifier: "_ZTS13__va_list_tag")
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
!961 = !{i32 2, !"SDK Version", [2 x i32] [i32 7, i32 0]}
!962 = !{i32 2, !"Dwarf Version", i32 4}
!963 = !{i32 2, !"Debug Info Version", i32 3}
!964 = !{i32 1, !"wchar_size", i32 4}
!965 = !{!"clang version 9.0.0 (https://github.com/yebinchon/llvm-project/ a1efa594106d738d0b74c9e4e2b9b779eb8b7d25)"}
!966 = distinct !DISubprogram(name: "randlc", linkageName: "_Z6randlcPdd", scope: !3, file: !3, line: 156, type: !967, scopeLine: 156, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !969)
!967 = !DISubroutineType(types: !968)
!968 = !{!99, !98, !99}
!969 = !{}
!970 = !DILocalVariable(name: "x", arg: 1, scope: !966, file: !3, line: 156, type: !98)
!971 = !DILocation(line: 156, column: 23, scope: !966)
!972 = !DILocalVariable(name: "a", arg: 2, scope: !966, file: !3, line: 156, type: !99)
!973 = !DILocation(line: 156, column: 33, scope: !966)
!974 = !DILocalVariable(name: "t1", scope: !966, file: !3, line: 157, type: !99)
!975 = !DILocation(line: 157, column: 9, scope: !966)
!976 = !DILocalVariable(name: "t2", scope: !966, file: !3, line: 157, type: !99)
!977 = !DILocation(line: 157, column: 12, scope: !966)
!978 = !DILocalVariable(name: "t3", scope: !966, file: !3, line: 157, type: !99)
!979 = !DILocation(line: 157, column: 15, scope: !966)
!980 = !DILocalVariable(name: "t4", scope: !966, file: !3, line: 157, type: !99)
!981 = !DILocation(line: 157, column: 18, scope: !966)
!982 = !DILocalVariable(name: "a1", scope: !966, file: !3, line: 157, type: !99)
!983 = !DILocation(line: 157, column: 21, scope: !966)
!984 = !DILocalVariable(name: "a2", scope: !966, file: !3, line: 157, type: !99)
!985 = !DILocation(line: 157, column: 24, scope: !966)
!986 = !DILocalVariable(name: "x1", scope: !966, file: !3, line: 157, type: !99)
!987 = !DILocation(line: 157, column: 27, scope: !966)
!988 = !DILocalVariable(name: "x2", scope: !966, file: !3, line: 157, type: !99)
!989 = !DILocation(line: 157, column: 30, scope: !966)
!990 = !DILocalVariable(name: "z", scope: !966, file: !3, line: 157, type: !99)
!991 = !DILocation(line: 157, column: 33, scope: !966)
!992 = !DILocation(line: 164, column: 13, scope: !966)
!993 = !DILocation(line: 164, column: 11, scope: !966)
!994 = !DILocation(line: 164, column: 5, scope: !966)
!995 = !DILocation(line: 165, column: 12, scope: !966)
!996 = !DILocation(line: 165, column: 7, scope: !966)
!997 = !DILocation(line: 165, column: 5, scope: !966)
!998 = !DILocation(line: 166, column: 7, scope: !966)
!999 = !DILocation(line: 166, column: 17, scope: !966)
!1000 = !DILocation(line: 166, column: 15, scope: !966)
!1001 = !DILocation(line: 166, column: 9, scope: !966)
!1002 = !DILocation(line: 166, column: 5, scope: !966)
!1003 = !DILocation(line: 175, column: 15, scope: !966)
!1004 = !DILocation(line: 175, column: 14, scope: !966)
!1005 = !DILocation(line: 175, column: 11, scope: !966)
!1006 = !DILocation(line: 175, column: 5, scope: !966)
!1007 = !DILocation(line: 176, column: 12, scope: !966)
!1008 = !DILocation(line: 176, column: 7, scope: !966)
!1009 = !DILocation(line: 176, column: 5, scope: !966)
!1010 = !DILocation(line: 177, column: 9, scope: !966)
!1011 = !DILocation(line: 177, column: 8, scope: !966)
!1012 = !DILocation(line: 177, column: 20, scope: !966)
!1013 = !DILocation(line: 177, column: 18, scope: !966)
!1014 = !DILocation(line: 177, column: 12, scope: !966)
!1015 = !DILocation(line: 177, column: 5, scope: !966)
!1016 = !DILocation(line: 178, column: 7, scope: !966)
!1017 = !DILocation(line: 178, column: 12, scope: !966)
!1018 = !DILocation(line: 178, column: 10, scope: !966)
!1019 = !DILocation(line: 178, column: 17, scope: !966)
!1020 = !DILocation(line: 178, column: 22, scope: !966)
!1021 = !DILocation(line: 178, column: 20, scope: !966)
!1022 = !DILocation(line: 178, column: 15, scope: !966)
!1023 = !DILocation(line: 178, column: 5, scope: !966)
!1024 = !DILocation(line: 179, column: 19, scope: !966)
!1025 = !DILocation(line: 179, column: 17, scope: !966)
!1026 = !DILocation(line: 179, column: 12, scope: !966)
!1027 = !DILocation(line: 179, column: 7, scope: !966)
!1028 = !DILocation(line: 179, column: 5, scope: !966)
!1029 = !DILocation(line: 180, column: 6, scope: !966)
!1030 = !DILocation(line: 180, column: 17, scope: !966)
!1031 = !DILocation(line: 180, column: 15, scope: !966)
!1032 = !DILocation(line: 180, column: 9, scope: !966)
!1033 = !DILocation(line: 180, column: 4, scope: !966)
!1034 = !DILocation(line: 181, column: 13, scope: !966)
!1035 = !DILocation(line: 181, column: 11, scope: !966)
!1036 = !DILocation(line: 181, column: 17, scope: !966)
!1037 = !DILocation(line: 181, column: 22, scope: !966)
!1038 = !DILocation(line: 181, column: 20, scope: !966)
!1039 = !DILocation(line: 181, column: 15, scope: !966)
!1040 = !DILocation(line: 181, column: 5, scope: !966)
!1041 = !DILocation(line: 182, column: 19, scope: !966)
!1042 = !DILocation(line: 182, column: 17, scope: !966)
!1043 = !DILocation(line: 182, column: 12, scope: !966)
!1044 = !DILocation(line: 182, column: 7, scope: !966)
!1045 = !DILocation(line: 182, column: 5, scope: !966)
!1046 = !DILocation(line: 183, column: 9, scope: !966)
!1047 = !DILocation(line: 183, column: 20, scope: !966)
!1048 = !DILocation(line: 183, column: 18, scope: !966)
!1049 = !DILocation(line: 183, column: 12, scope: !966)
!1050 = !DILocation(line: 183, column: 4, scope: !966)
!1051 = !DILocation(line: 183, column: 7, scope: !966)
!1052 = !DILocation(line: 185, column: 18, scope: !966)
!1053 = !DILocation(line: 185, column: 17, scope: !966)
!1054 = !DILocation(line: 185, column: 14, scope: !966)
!1055 = !DILocation(line: 185, column: 2, scope: !966)
!1056 = distinct !DISubprogram(name: "c_print_results", linkageName: "_Z15c_print_resultsPcciiiiddS_iS_S_S_S_S_S_S_S_S_S_S_S_S_S_", scope: !3, file: !3, line: 191, type: !1057, scopeLine: 214, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !969)
!1057 = !DISubroutineType(types: !1058)
!1058 = !{null, !100, !101, !97, !97, !97, !97, !99, !99, !100, !97, !100, !100, !100, !100, !100, !100, !100, !100, !100, !100, !100, !100, !100, !100}
!1059 = !DILocalVariable(name: "name", arg: 1, scope: !1056, file: !3, line: 191, type: !100)
!1060 = !DILocation(line: 191, column: 28, scope: !1056)
!1061 = !DILocalVariable(name: "class_npb", arg: 2, scope: !1056, file: !3, line: 192, type: !101)
!1062 = !DILocation(line: 192, column: 8, scope: !1056)
!1063 = !DILocalVariable(name: "n1", arg: 3, scope: !1056, file: !3, line: 193, type: !97)
!1064 = !DILocation(line: 193, column: 7, scope: !1056)
!1065 = !DILocalVariable(name: "n2", arg: 4, scope: !1056, file: !3, line: 194, type: !97)
!1066 = !DILocation(line: 194, column: 7, scope: !1056)
!1067 = !DILocalVariable(name: "n3", arg: 5, scope: !1056, file: !3, line: 195, type: !97)
!1068 = !DILocation(line: 195, column: 7, scope: !1056)
!1069 = !DILocalVariable(name: "niter", arg: 6, scope: !1056, file: !3, line: 196, type: !97)
!1070 = !DILocation(line: 196, column: 7, scope: !1056)
!1071 = !DILocalVariable(name: "t", arg: 7, scope: !1056, file: !3, line: 197, type: !99)
!1072 = !DILocation(line: 197, column: 10, scope: !1056)
!1073 = !DILocalVariable(name: "mops", arg: 8, scope: !1056, file: !3, line: 198, type: !99)
!1074 = !DILocation(line: 198, column: 10, scope: !1056)
!1075 = !DILocalVariable(name: "optype", arg: 9, scope: !1056, file: !3, line: 199, type: !100)
!1076 = !DILocation(line: 199, column: 9, scope: !1056)
!1077 = !DILocalVariable(name: "passed_verification", arg: 10, scope: !1056, file: !3, line: 200, type: !97)
!1078 = !DILocation(line: 200, column: 7, scope: !1056)
!1079 = !DILocalVariable(name: "npbversion", arg: 11, scope: !1056, file: !3, line: 201, type: !100)
!1080 = !DILocation(line: 201, column: 9, scope: !1056)
!1081 = !DILocalVariable(name: "compiletime", arg: 12, scope: !1056, file: !3, line: 202, type: !100)
!1082 = !DILocation(line: 202, column: 9, scope: !1056)
!1083 = !DILocalVariable(name: "compilerversion", arg: 13, scope: !1056, file: !3, line: 203, type: !100)
!1084 = !DILocation(line: 203, column: 9, scope: !1056)
!1085 = !DILocalVariable(name: "libversion", arg: 14, scope: !1056, file: !3, line: 204, type: !100)
!1086 = !DILocation(line: 204, column: 9, scope: !1056)
!1087 = !DILocalVariable(name: "cpu_device", arg: 15, scope: !1056, file: !3, line: 205, type: !100)
!1088 = !DILocation(line: 205, column: 9, scope: !1056)
!1089 = !DILocalVariable(name: "gpu_device", arg: 16, scope: !1056, file: !3, line: 206, type: !100)
!1090 = !DILocation(line: 206, column: 9, scope: !1056)
!1091 = !DILocalVariable(name: "gpu_config", arg: 17, scope: !1056, file: !3, line: 207, type: !100)
!1092 = !DILocation(line: 207, column: 9, scope: !1056)
!1093 = !DILocalVariable(name: "cc", arg: 18, scope: !1056, file: !3, line: 208, type: !100)
!1094 = !DILocation(line: 208, column: 9, scope: !1056)
!1095 = !DILocalVariable(name: "clink", arg: 19, scope: !1056, file: !3, line: 209, type: !100)
!1096 = !DILocation(line: 209, column: 9, scope: !1056)
!1097 = !DILocalVariable(name: "c_lib", arg: 20, scope: !1056, file: !3, line: 210, type: !100)
!1098 = !DILocation(line: 210, column: 9, scope: !1056)
!1099 = !DILocalVariable(name: "c_inc", arg: 21, scope: !1056, file: !3, line: 211, type: !100)
!1100 = !DILocation(line: 211, column: 9, scope: !1056)
!1101 = !DILocalVariable(name: "cflags", arg: 22, scope: !1056, file: !3, line: 212, type: !100)
!1102 = !DILocation(line: 212, column: 9, scope: !1056)
!1103 = !DILocalVariable(name: "clinkflags", arg: 23, scope: !1056, file: !3, line: 213, type: !100)
!1104 = !DILocation(line: 213, column: 9, scope: !1056)
!1105 = !DILocalVariable(name: "rand", arg: 24, scope: !1056, file: !3, line: 214, type: !100)
!1106 = !DILocation(line: 214, column: 9, scope: !1056)
!1107 = !DILocation(line: 215, column: 44, scope: !1056)
!1108 = !DILocation(line: 215, column: 4, scope: !1056)
!1109 = !DILocation(line: 216, column: 61, scope: !1056)
!1110 = !DILocation(line: 216, column: 4, scope: !1056)
!1111 = !DILocation(line: 217, column: 8, scope: !1112)
!1112 = distinct !DILexicalBlock(scope: !1056, file: !3, line: 217, column: 7)
!1113 = !DILocation(line: 217, column: 15, scope: !1112)
!1114 = !DILocation(line: 217, column: 21, scope: !1112)
!1115 = !DILocation(line: 217, column: 24, scope: !1112)
!1116 = !DILocation(line: 217, column: 31, scope: !1112)
!1117 = !DILocation(line: 217, column: 7, scope: !1056)
!1118 = !DILocation(line: 218, column: 8, scope: !1119)
!1119 = distinct !DILexicalBlock(scope: !1120, file: !3, line: 218, column: 8)
!1120 = distinct !DILexicalBlock(scope: !1112, file: !3, line: 217, column: 38)
!1121 = !DILocation(line: 218, column: 10, scope: !1119)
!1122 = !DILocation(line: 218, column: 8, scope: !1120)
!1123 = !DILocalVariable(name: "nn", scope: !1124, file: !3, line: 219, type: !313)
!1124 = distinct !DILexicalBlock(scope: !1119, file: !3, line: 218, column: 14)
!1125 = !DILocation(line: 219, column: 11, scope: !1124)
!1126 = !DILocation(line: 219, column: 16, scope: !1124)
!1127 = !DILocation(line: 220, column: 9, scope: !1128)
!1128 = distinct !DILexicalBlock(scope: !1124, file: !3, line: 220, column: 9)
!1129 = !DILocation(line: 220, column: 11, scope: !1128)
!1130 = !DILocation(line: 220, column: 9, scope: !1124)
!1131 = !DILocation(line: 220, column: 20, scope: !1132)
!1132 = distinct !DILexicalBlock(scope: !1128, file: !3, line: 220, column: 15)
!1133 = !DILocation(line: 220, column: 18, scope: !1132)
!1134 = !DILocation(line: 220, column: 23, scope: !1132)
!1135 = !DILocation(line: 221, column: 55, scope: !1124)
!1136 = !DILocation(line: 221, column: 6, scope: !1124)
!1137 = !DILocation(line: 222, column: 5, scope: !1124)
!1138 = !DILocation(line: 223, column: 61, scope: !1139)
!1139 = distinct !DILexicalBlock(scope: !1119, file: !3, line: 222, column: 10)
!1140 = !DILocation(line: 223, column: 64, scope: !1139)
!1141 = !DILocation(line: 223, column: 67, scope: !1139)
!1142 = !DILocation(line: 223, column: 6, scope: !1139)
!1143 = !DILocation(line: 225, column: 4, scope: !1120)
!1144 = !DILocalVariable(name: "size", scope: !1145, file: !3, line: 226, type: !1146)
!1145 = distinct !DILexicalBlock(scope: !1112, file: !3, line: 225, column: 9)
!1146 = !DICompositeType(tag: DW_TAG_array_type, baseType: !101, size: 128, elements: !1147)
!1147 = !{!1148}
!1148 = !DISubrange(count: 16)
!1149 = !DILocation(line: 226, column: 10, scope: !1145)
!1150 = !DILocalVariable(name: "j", scope: !1145, file: !3, line: 227, type: !97)
!1151 = !DILocation(line: 227, column: 9, scope: !1145)
!1152 = !DILocation(line: 228, column: 9, scope: !1153)
!1153 = distinct !DILexicalBlock(scope: !1145, file: !3, line: 228, column: 8)
!1154 = !DILocation(line: 228, column: 11, scope: !1153)
!1155 = !DILocation(line: 228, column: 16, scope: !1153)
!1156 = !DILocation(line: 228, column: 20, scope: !1153)
!1157 = !DILocation(line: 228, column: 22, scope: !1153)
!1158 = !DILocation(line: 228, column: 8, scope: !1145)
!1159 = !DILocation(line: 229, column: 10, scope: !1160)
!1160 = distinct !DILexicalBlock(scope: !1161, file: !3, line: 229, column: 9)
!1161 = distinct !DILexicalBlock(scope: !1153, file: !3, line: 228, column: 27)
!1162 = !DILocation(line: 229, column: 17, scope: !1160)
!1163 = !DILocation(line: 229, column: 23, scope: !1160)
!1164 = !DILocation(line: 229, column: 26, scope: !1160)
!1165 = !DILocation(line: 229, column: 33, scope: !1160)
!1166 = !DILocation(line: 229, column: 9, scope: !1161)
!1167 = !DILocation(line: 230, column: 15, scope: !1168)
!1168 = distinct !DILexicalBlock(scope: !1160, file: !3, line: 229, column: 40)
!1169 = !DILocation(line: 230, column: 41, scope: !1168)
!1170 = !DILocation(line: 230, column: 32, scope: !1168)
!1171 = !DILocation(line: 230, column: 7, scope: !1168)
!1172 = !DILocation(line: 231, column: 9, scope: !1168)
!1173 = !DILocation(line: 232, column: 15, scope: !1174)
!1174 = distinct !DILexicalBlock(scope: !1168, file: !3, line: 232, column: 10)
!1175 = !DILocation(line: 232, column: 10, scope: !1174)
!1176 = !DILocation(line: 232, column: 18, scope: !1174)
!1177 = !DILocation(line: 232, column: 10, scope: !1168)
!1178 = !DILocation(line: 233, column: 13, scope: !1179)
!1179 = distinct !DILexicalBlock(scope: !1174, file: !3, line: 232, column: 25)
!1180 = !DILocation(line: 233, column: 8, scope: !1179)
!1181 = !DILocation(line: 233, column: 16, scope: !1179)
!1182 = !DILocation(line: 234, column: 9, scope: !1179)
!1183 = !DILocation(line: 235, column: 7, scope: !1179)
!1184 = !DILocation(line: 236, column: 12, scope: !1168)
!1185 = !DILocation(line: 236, column: 13, scope: !1168)
!1186 = !DILocation(line: 236, column: 7, scope: !1168)
!1187 = !DILocation(line: 236, column: 17, scope: !1168)
!1188 = !DILocation(line: 237, column: 52, scope: !1168)
!1189 = !DILocation(line: 237, column: 7, scope: !1168)
!1190 = !DILocation(line: 238, column: 6, scope: !1168)
!1191 = !DILocation(line: 239, column: 55, scope: !1192)
!1192 = distinct !DILexicalBlock(scope: !1160, file: !3, line: 238, column: 11)
!1193 = !DILocation(line: 239, column: 7, scope: !1192)
!1194 = !DILocation(line: 241, column: 5, scope: !1161)
!1195 = !DILocation(line: 242, column: 59, scope: !1196)
!1196 = distinct !DILexicalBlock(scope: !1153, file: !3, line: 241, column: 10)
!1197 = !DILocation(line: 242, column: 63, scope: !1196)
!1198 = !DILocation(line: 242, column: 67, scope: !1196)
!1199 = !DILocation(line: 242, column: 6, scope: !1196)
!1200 = !DILocation(line: 245, column: 52, scope: !1056)
!1201 = !DILocation(line: 245, column: 4, scope: !1056)
!1202 = !DILocation(line: 246, column: 54, scope: !1056)
!1203 = !DILocation(line: 246, column: 4, scope: !1056)
!1204 = !DILocation(line: 247, column: 54, scope: !1056)
!1205 = !DILocation(line: 247, column: 4, scope: !1056)
!1206 = !DILocation(line: 248, column: 40, scope: !1056)
!1207 = !DILocation(line: 248, column: 4, scope: !1056)
!1208 = !DILocation(line: 249, column: 7, scope: !1209)
!1209 = distinct !DILexicalBlock(scope: !1056, file: !3, line: 249, column: 7)
!1210 = !DILocation(line: 249, column: 27, scope: !1209)
!1211 = !DILocation(line: 249, column: 7, scope: !1056)
!1212 = !DILocation(line: 250, column: 5, scope: !1213)
!1213 = distinct !DILexicalBlock(scope: !1209, file: !3, line: 249, column: 31)
!1214 = !DILocation(line: 251, column: 4, scope: !1213)
!1215 = !DILocation(line: 251, column: 13, scope: !1216)
!1216 = distinct !DILexicalBlock(scope: !1209, file: !3, line: 251, column: 13)
!1217 = !DILocation(line: 251, column: 13, scope: !1209)
!1218 = !DILocation(line: 252, column: 5, scope: !1219)
!1219 = distinct !DILexicalBlock(scope: !1216, file: !3, line: 251, column: 33)
!1220 = !DILocation(line: 253, column: 4, scope: !1219)
!1221 = !DILocation(line: 254, column: 5, scope: !1222)
!1222 = distinct !DILexicalBlock(scope: !1216, file: !3, line: 253, column: 9)
!1223 = !DILocation(line: 256, column: 52, scope: !1056)
!1224 = !DILocation(line: 256, column: 4, scope: !1056)
!1225 = !DILocation(line: 257, column: 52, scope: !1056)
!1226 = !DILocation(line: 257, column: 4, scope: !1056)
!1227 = !DILocation(line: 258, column: 52, scope: !1056)
!1228 = !DILocation(line: 258, column: 4, scope: !1056)
!1229 = !DILocation(line: 259, column: 52, scope: !1056)
!1230 = !DILocation(line: 259, column: 4, scope: !1056)
!1231 = !DILocation(line: 260, column: 4, scope: !1056)
!1232 = !DILocation(line: 261, column: 38, scope: !1056)
!1233 = !DILocation(line: 261, column: 4, scope: !1056)
!1234 = !DILocation(line: 262, column: 38, scope: !1056)
!1235 = !DILocation(line: 262, column: 4, scope: !1056)
!1236 = !DILocation(line: 263, column: 38, scope: !1056)
!1237 = !DILocation(line: 263, column: 4, scope: !1056)
!1238 = !DILocation(line: 264, column: 38, scope: !1056)
!1239 = !DILocation(line: 264, column: 4, scope: !1056)
!1240 = !DILocation(line: 265, column: 38, scope: !1056)
!1241 = !DILocation(line: 265, column: 4, scope: !1056)
!1242 = !DILocation(line: 266, column: 38, scope: !1056)
!1243 = !DILocation(line: 266, column: 4, scope: !1056)
!1244 = !DILocation(line: 267, column: 38, scope: !1056)
!1245 = !DILocation(line: 267, column: 4, scope: !1056)
!1246 = !DILocation(line: 268, column: 4, scope: !1056)
!1247 = !DILocation(line: 269, column: 38, scope: !1056)
!1248 = !DILocation(line: 269, column: 4, scope: !1056)
!1249 = !DILocation(line: 270, column: 38, scope: !1056)
!1250 = !DILocation(line: 270, column: 4, scope: !1056)
!1251 = !DILocation(line: 271, column: 4, scope: !1056)
!1252 = !DILocation(line: 272, column: 38, scope: !1056)
!1253 = !DILocation(line: 272, column: 4, scope: !1056)
!1254 = !DILocation(line: 287, column: 4, scope: !1056)
!1255 = !DILocation(line: 288, column: 4, scope: !1056)
!1256 = !DILocation(line: 289, column: 4, scope: !1056)
!1257 = !DILocation(line: 290, column: 4, scope: !1056)
!1258 = !DILocation(line: 291, column: 4, scope: !1056)
!1259 = !DILocation(line: 292, column: 4, scope: !1056)
!1260 = !DILocation(line: 293, column: 4, scope: !1056)
!1261 = !DILocation(line: 294, column: 4, scope: !1056)
!1262 = !DILocation(line: 295, column: 4, scope: !1056)
!1263 = !DILocation(line: 296, column: 4, scope: !1056)
!1264 = !DILocation(line: 297, column: 3, scope: !1056)
!1265 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 301, type: !1266, scopeLine: 301, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !969)
!1266 = !DISubroutineType(types: !1267)
!1267 = !{!97, !97, !566}
!1268 = !DILocalVariable(name: "argc", arg: 1, scope: !1265, file: !3, line: 301, type: !97)
!1269 = !DILocation(line: 301, column: 14, scope: !1265)
!1270 = !DILocalVariable(name: "argv", arg: 2, scope: !1265, file: !3, line: 301, type: !566)
!1271 = !DILocation(line: 301, column: 27, scope: !1265)
!1272 = !DILocation(line: 308, column: 15, scope: !1265)
!1273 = !DILocation(line: 308, column: 6, scope: !1265)
!1274 = !DILocation(line: 308, column: 4, scope: !1265)
!1275 = !DILocalVariable(name: "Mops", scope: !1265, file: !3, line: 309, type: !99)
!1276 = !DILocation(line: 309, column: 9, scope: !1265)
!1277 = !DILocalVariable(name: "t1", scope: !1265, file: !3, line: 309, type: !99)
!1278 = !DILocation(line: 309, column: 15, scope: !1265)
!1279 = !DILocalVariable(name: "sx", scope: !1265, file: !3, line: 310, type: !99)
!1280 = !DILocation(line: 310, column: 9, scope: !1265)
!1281 = !DILocalVariable(name: "sy", scope: !1265, file: !3, line: 310, type: !99)
!1282 = !DILocation(line: 310, column: 13, scope: !1265)
!1283 = !DILocalVariable(name: "an", scope: !1265, file: !3, line: 310, type: !99)
!1284 = !DILocation(line: 310, column: 17, scope: !1265)
!1285 = !DILocalVariable(name: "gc", scope: !1265, file: !3, line: 310, type: !99)
!1286 = !DILocation(line: 310, column: 21, scope: !1265)
!1287 = !DILocalVariable(name: "sx_verify_value", scope: !1265, file: !3, line: 311, type: !99)
!1288 = !DILocation(line: 311, column: 9, scope: !1265)
!1289 = !DILocalVariable(name: "sy_verify_value", scope: !1265, file: !3, line: 311, type: !99)
!1290 = !DILocation(line: 311, column: 26, scope: !1265)
!1291 = !DILocalVariable(name: "sx_err", scope: !1265, file: !3, line: 311, type: !99)
!1292 = !DILocation(line: 311, column: 43, scope: !1265)
!1293 = !DILocalVariable(name: "sy_err", scope: !1265, file: !3, line: 311, type: !99)
!1294 = !DILocation(line: 311, column: 51, scope: !1265)
!1295 = !DILocalVariable(name: "i", scope: !1265, file: !3, line: 312, type: !97)
!1296 = !DILocation(line: 312, column: 6, scope: !1265)
!1297 = !DILocalVariable(name: "j", scope: !1265, file: !3, line: 312, type: !97)
!1298 = !DILocation(line: 312, column: 9, scope: !1265)
!1299 = !DILocalVariable(name: "nit", scope: !1265, file: !3, line: 312, type: !97)
!1300 = !DILocation(line: 312, column: 12, scope: !1265)
!1301 = !DILocalVariable(name: "block", scope: !1265, file: !3, line: 312, type: !97)
!1302 = !DILocation(line: 312, column: 17, scope: !1265)
!1303 = !DILocalVariable(name: "verified", scope: !1265, file: !3, line: 313, type: !1304)
!1304 = !DIDerivedType(tag: DW_TAG_typedef, name: "boolean", file: !1305, line: 80, baseType: !97)
!1305 = !DIFile(filename: "./../common/npb-CPP.hpp", directory: "/scratch/ah7226/NPB-GPU/CUDA/EP")
!1306 = !DILocation(line: 313, column: 10, scope: !1265)
!1307 = !DILocalVariable(name: "size", scope: !1265, file: !3, line: 314, type: !1146)
!1308 = !DILocation(line: 314, column: 7, scope: !1265)
!1309 = !DILocation(line: 324, column: 10, scope: !1265)
!1310 = !DILocation(line: 324, column: 26, scope: !1265)
!1311 = !DILocation(line: 324, column: 2, scope: !1265)
!1312 = !DILocation(line: 325, column: 4, scope: !1265)
!1313 = !DILocation(line: 326, column: 10, scope: !1314)
!1314 = distinct !DILexicalBlock(scope: !1265, file: !3, line: 326, column: 5)
!1315 = !DILocation(line: 326, column: 5, scope: !1314)
!1316 = !DILocation(line: 326, column: 12, scope: !1314)
!1317 = !DILocation(line: 326, column: 5, scope: !1265)
!1318 = !DILocation(line: 326, column: 20, scope: !1319)
!1319 = distinct !DILexicalBlock(scope: !1314, file: !3, line: 326, column: 18)
!1320 = !DILocation(line: 326, column: 23, scope: !1319)
!1321 = !DILocation(line: 327, column: 7, scope: !1265)
!1322 = !DILocation(line: 327, column: 8, scope: !1265)
!1323 = !DILocation(line: 327, column: 2, scope: !1265)
!1324 = !DILocation(line: 327, column: 12, scope: !1265)
!1325 = !DILocation(line: 328, column: 2, scope: !1265)
!1326 = !DILocation(line: 329, column: 56, scope: !1265)
!1327 = !DILocation(line: 329, column: 2, scope: !1265)
!1328 = !DILocation(line: 331, column: 11, scope: !1265)
!1329 = !DILocation(line: 333, column: 5, scope: !1265)
!1330 = !DILocation(line: 335, column: 7, scope: !1331)
!1331 = distinct !DILexicalBlock(scope: !1265, file: !3, line: 335, column: 2)
!1332 = !DILocation(line: 335, column: 6, scope: !1331)
!1333 = !DILocation(line: 335, column: 11, scope: !1334)
!1334 = distinct !DILexicalBlock(scope: !1331, file: !3, line: 335, column: 2)
!1335 = !DILocation(line: 335, column: 12, scope: !1334)
!1336 = !DILocation(line: 335, column: 2, scope: !1331)
!1337 = !DILocation(line: 336, column: 15, scope: !1338)
!1338 = distinct !DILexicalBlock(scope: !1334, file: !3, line: 335, column: 23)
!1339 = !DILocation(line: 336, column: 3, scope: !1338)
!1340 = !DILocation(line: 337, column: 2, scope: !1338)
!1341 = !DILocation(line: 335, column: 20, scope: !1334)
!1342 = !DILocation(line: 335, column: 2, scope: !1334)
!1343 = distinct !{!1343, !1336, !1344}
!1344 = !DILocation(line: 337, column: 2, scope: !1331)
!1345 = !DILocation(line: 339, column: 7, scope: !1265)
!1346 = !DILocation(line: 339, column: 5, scope: !1265)
!1347 = !DILocation(line: 340, column: 5, scope: !1265)
!1348 = !DILocation(line: 341, column: 5, scope: !1265)
!1349 = !DILocation(line: 342, column: 5, scope: !1265)
!1350 = !DILocation(line: 344, column: 7, scope: !1351)
!1351 = distinct !DILexicalBlock(scope: !1265, file: !3, line: 344, column: 2)
!1352 = !DILocation(line: 344, column: 6, scope: !1351)
!1353 = !DILocation(line: 344, column: 11, scope: !1354)
!1354 = distinct !DILexicalBlock(scope: !1351, file: !3, line: 344, column: 2)
!1355 = !DILocation(line: 344, column: 12, scope: !1354)
!1356 = !DILocation(line: 344, column: 2, scope: !1351)
!1357 = !DILocation(line: 345, column: 3, scope: !1358)
!1358 = distinct !DILexicalBlock(scope: !1354, file: !3, line: 344, column: 21)
!1359 = !DILocation(line: 345, column: 5, scope: !1358)
!1360 = !DILocation(line: 345, column: 8, scope: !1358)
!1361 = !DILocation(line: 346, column: 2, scope: !1358)
!1362 = !DILocation(line: 344, column: 18, scope: !1354)
!1363 = !DILocation(line: 344, column: 2, scope: !1354)
!1364 = distinct !{!1364, !1356, !1365}
!1365 = !DILocation(line: 346, column: 2, scope: !1351)
!1366 = !DILocation(line: 348, column: 2, scope: !1265)
!1367 = !DILocation(line: 353, column: 15, scope: !1265)
!1368 = !DILocation(line: 354, column: 3, scope: !1265)
!1369 = !DILocation(line: 353, column: 12, scope: !1265)
!1370 = !DILocation(line: 353, column: 2, scope: !1265)
!1371 = !DILocation(line: 354, column: 24, scope: !1265)
!1372 = !DILocation(line: 355, column: 5, scope: !1265)
!1373 = !DILocation(line: 356, column: 5, scope: !1265)
!1374 = !DILocation(line: 357, column: 5, scope: !1265)
!1375 = !DILocation(line: 362, column: 13, scope: !1265)
!1376 = !DILocation(line: 362, column: 21, scope: !1265)
!1377 = !DILocation(line: 362, column: 31, scope: !1265)
!1378 = !DILocation(line: 362, column: 2, scope: !1265)
!1379 = !DILocation(line: 363, column: 13, scope: !1265)
!1380 = !DILocation(line: 363, column: 22, scope: !1265)
!1381 = !DILocation(line: 363, column: 33, scope: !1265)
!1382 = !DILocation(line: 363, column: 2, scope: !1265)
!1383 = !DILocation(line: 364, column: 13, scope: !1265)
!1384 = !DILocation(line: 364, column: 22, scope: !1265)
!1385 = !DILocation(line: 364, column: 33, scope: !1265)
!1386 = !DILocation(line: 364, column: 2, scope: !1265)
!1387 = !DILocation(line: 366, column: 11, scope: !1388)
!1388 = distinct !DILexicalBlock(scope: !1265, file: !3, line: 366, column: 2)
!1389 = !DILocation(line: 366, column: 6, scope: !1388)
!1390 = !DILocation(line: 366, column: 15, scope: !1391)
!1391 = distinct !DILexicalBlock(scope: !1388, file: !3, line: 366, column: 2)
!1392 = !DILocation(line: 366, column: 21, scope: !1391)
!1393 = !DILocation(line: 366, column: 20, scope: !1391)
!1394 = !DILocation(line: 366, column: 2, scope: !1388)
!1395 = !DILocation(line: 367, column: 8, scope: !1396)
!1396 = distinct !DILexicalBlock(scope: !1397, file: !3, line: 367, column: 3)
!1397 = distinct !DILexicalBlock(scope: !1391, file: !3, line: 366, column: 46)
!1398 = !DILocation(line: 367, column: 7, scope: !1396)
!1399 = !DILocation(line: 367, column: 12, scope: !1400)
!1400 = distinct !DILexicalBlock(scope: !1396, file: !3, line: 367, column: 3)
!1401 = !DILocation(line: 367, column: 13, scope: !1400)
!1402 = !DILocation(line: 367, column: 3, scope: !1396)
!1403 = !DILocation(line: 368, column: 10, scope: !1404)
!1404 = distinct !DILexicalBlock(scope: !1400, file: !3, line: 367, column: 22)
!1405 = !DILocation(line: 368, column: 17, scope: !1404)
!1406 = !DILocation(line: 368, column: 22, scope: !1404)
!1407 = !DILocation(line: 368, column: 26, scope: !1404)
!1408 = !DILocation(line: 368, column: 25, scope: !1404)
!1409 = !DILocation(line: 368, column: 4, scope: !1404)
!1410 = !DILocation(line: 368, column: 6, scope: !1404)
!1411 = !DILocation(line: 368, column: 8, scope: !1404)
!1412 = !DILocation(line: 369, column: 3, scope: !1404)
!1413 = !DILocation(line: 367, column: 19, scope: !1400)
!1414 = !DILocation(line: 367, column: 3, scope: !1400)
!1415 = distinct !{!1415, !1402, !1416}
!1416 = !DILocation(line: 369, column: 3, scope: !1396)
!1417 = !DILocation(line: 370, column: 7, scope: !1397)
!1418 = !DILocation(line: 370, column: 15, scope: !1397)
!1419 = !DILocation(line: 370, column: 5, scope: !1397)
!1420 = !DILocation(line: 371, column: 7, scope: !1397)
!1421 = !DILocation(line: 371, column: 15, scope: !1397)
!1422 = !DILocation(line: 371, column: 5, scope: !1397)
!1423 = !DILocation(line: 372, column: 2, scope: !1397)
!1424 = !DILocation(line: 366, column: 43, scope: !1391)
!1425 = !DILocation(line: 366, column: 2, scope: !1391)
!1426 = distinct !{!1426, !1394, !1427}
!1427 = !DILocation(line: 372, column: 2, scope: !1388)
!1428 = !DILocation(line: 373, column: 7, scope: !1429)
!1429 = distinct !DILexicalBlock(scope: !1265, file: !3, line: 373, column: 2)
!1430 = !DILocation(line: 373, column: 6, scope: !1429)
!1431 = !DILocation(line: 373, column: 11, scope: !1432)
!1432 = distinct !DILexicalBlock(scope: !1429, file: !3, line: 373, column: 2)
!1433 = !DILocation(line: 373, column: 12, scope: !1432)
!1434 = !DILocation(line: 373, column: 2, scope: !1429)
!1435 = !DILocation(line: 374, column: 7, scope: !1436)
!1436 = distinct !DILexicalBlock(scope: !1432, file: !3, line: 373, column: 21)
!1437 = !DILocation(line: 374, column: 9, scope: !1436)
!1438 = !DILocation(line: 374, column: 5, scope: !1436)
!1439 = !DILocation(line: 375, column: 2, scope: !1436)
!1440 = !DILocation(line: 373, column: 18, scope: !1432)
!1441 = !DILocation(line: 373, column: 2, scope: !1432)
!1442 = distinct !{!1442, !1434, !1443}
!1443 = !DILocation(line: 375, column: 2, scope: !1429)
!1444 = !DILocation(line: 377, column: 6, scope: !1265)
!1445 = !DILocation(line: 378, column: 11, scope: !1265)
!1446 = !DILocation(line: 386, column: 19, scope: !1447)
!1447 = distinct !DILexicalBlock(scope: !1448, file: !3, line: 385, column: 19)
!1448 = distinct !DILexicalBlock(scope: !1449, file: !3, line: 385, column: 11)
!1449 = distinct !DILexicalBlock(scope: !1450, file: !3, line: 382, column: 11)
!1450 = distinct !DILexicalBlock(scope: !1265, file: !3, line: 379, column: 5)
!1451 = !DILocation(line: 387, column: 19, scope: !1447)
!1452 = !DILocation(line: 403, column: 5, scope: !1453)
!1453 = distinct !DILexicalBlock(scope: !1265, file: !3, line: 403, column: 5)
!1454 = !DILocation(line: 403, column: 5, scope: !1265)
!1455 = !DILocation(line: 404, column: 18, scope: !1456)
!1456 = distinct !DILexicalBlock(scope: !1453, file: !3, line: 403, column: 14)
!1457 = !DILocation(line: 404, column: 23, scope: !1456)
!1458 = !DILocation(line: 404, column: 21, scope: !1456)
!1459 = !DILocation(line: 404, column: 42, scope: !1456)
!1460 = !DILocation(line: 404, column: 40, scope: !1456)
!1461 = !DILocation(line: 404, column: 12, scope: !1456)
!1462 = !DILocation(line: 404, column: 10, scope: !1456)
!1463 = !DILocation(line: 405, column: 18, scope: !1456)
!1464 = !DILocation(line: 405, column: 23, scope: !1456)
!1465 = !DILocation(line: 405, column: 21, scope: !1456)
!1466 = !DILocation(line: 405, column: 42, scope: !1456)
!1467 = !DILocation(line: 405, column: 40, scope: !1456)
!1468 = !DILocation(line: 405, column: 12, scope: !1456)
!1469 = !DILocation(line: 405, column: 10, scope: !1456)
!1470 = !DILocation(line: 406, column: 16, scope: !1456)
!1471 = !DILocation(line: 406, column: 23, scope: !1456)
!1472 = !DILocation(line: 406, column: 35, scope: !1456)
!1473 = !DILocation(line: 406, column: 39, scope: !1456)
!1474 = !DILocation(line: 406, column: 46, scope: !1456)
!1475 = !DILocation(line: 0, scope: !1456)
!1476 = !DILocation(line: 406, column: 14, scope: !1456)
!1477 = !DILocation(line: 406, column: 12, scope: !1456)
!1478 = !DILocation(line: 407, column: 2, scope: !1456)
!1479 = !DILocation(line: 408, column: 9, scope: !1265)
!1480 = !DILocation(line: 408, column: 22, scope: !1265)
!1481 = !DILocation(line: 408, column: 7, scope: !1265)
!1482 = !DILocation(line: 410, column: 2, scope: !1265)
!1483 = !DILocation(line: 411, column: 2, scope: !1265)
!1484 = !DILocation(line: 412, column: 2, scope: !1265)
!1485 = !DILocation(line: 413, column: 43, scope: !1265)
!1486 = !DILocation(line: 413, column: 2, scope: !1265)
!1487 = !DILocation(line: 414, column: 38, scope: !1265)
!1488 = !DILocation(line: 414, column: 42, scope: !1265)
!1489 = !DILocation(line: 414, column: 2, scope: !1265)
!1490 = !DILocation(line: 415, column: 2, scope: !1265)
!1491 = !DILocation(line: 416, column: 7, scope: !1492)
!1492 = distinct !DILexicalBlock(scope: !1265, file: !3, line: 416, column: 2)
!1493 = !DILocation(line: 416, column: 6, scope: !1492)
!1494 = !DILocation(line: 416, column: 11, scope: !1495)
!1495 = distinct !DILexicalBlock(scope: !1492, file: !3, line: 416, column: 2)
!1496 = !DILocation(line: 416, column: 12, scope: !1495)
!1497 = !DILocation(line: 416, column: 2, scope: !1492)
!1498 = !DILocation(line: 417, column: 25, scope: !1499)
!1499 = distinct !DILexicalBlock(scope: !1495, file: !3, line: 416, column: 21)
!1500 = !DILocation(line: 417, column: 28, scope: !1499)
!1501 = !DILocation(line: 417, column: 30, scope: !1499)
!1502 = !DILocation(line: 417, column: 3, scope: !1499)
!1503 = !DILocation(line: 418, column: 2, scope: !1499)
!1504 = !DILocation(line: 416, column: 18, scope: !1495)
!1505 = !DILocation(line: 416, column: 2, scope: !1495)
!1506 = distinct !{!1506, !1497, !1507}
!1507 = !DILocation(line: 418, column: 2, scope: !1492)
!1508 = !DILocalVariable(name: "gpu_config", scope: !1265, file: !3, line: 420, type: !137)
!1509 = !DILocation(line: 420, column: 7, scope: !1265)
!1510 = !DILocalVariable(name: "gpu_config_string", scope: !1265, file: !3, line: 421, type: !1511)
!1511 = !DICompositeType(tag: DW_TAG_array_type, baseType: !101, size: 16384, elements: !1512)
!1512 = !{!1513}
!1513 = !DISubrange(count: 2048)
!1514 = !DILocation(line: 421, column: 7, scope: !1265)
!1515 = !DILocation(line: 428, column: 10, scope: !1265)
!1516 = !DILocation(line: 428, column: 2, scope: !1265)
!1517 = !DILocation(line: 429, column: 9, scope: !1265)
!1518 = !DILocation(line: 429, column: 28, scope: !1265)
!1519 = !DILocation(line: 429, column: 2, scope: !1265)
!1520 = !DILocation(line: 430, column: 10, scope: !1265)
!1521 = !DILocation(line: 430, column: 45, scope: !1265)
!1522 = !DILocation(line: 430, column: 2, scope: !1265)
!1523 = !DILocation(line: 431, column: 9, scope: !1265)
!1524 = !DILocation(line: 431, column: 28, scope: !1265)
!1525 = !DILocation(line: 431, column: 2, scope: !1265)
!1526 = !DILocation(line: 439, column: 4, scope: !1265)
!1527 = !DILocation(line: 441, column: 4, scope: !1265)
!1528 = !DILocation(line: 443, column: 4, scope: !1265)
!1529 = !DILocation(line: 450, column: 4, scope: !1265)
!1530 = !DILocation(line: 434, column: 2, scope: !1265)
!1531 = !DILocation(line: 459, column: 2, scope: !1265)
!1532 = !DILocation(line: 461, column: 2, scope: !1265)
!1533 = distinct !DISubprogram(name: "setup_gpu", linkageName: "_ZL9setup_gpuv", scope: !3, file: !3, line: 577, type: !472, scopeLine: 577, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !969)
!1534 = !DILocation(line: 624, column: 33, scope: !1533)
!1535 = !DILocation(line: 625, column: 43, scope: !1533)
!1536 = !DILocation(line: 628, column: 49, scope: !1537)
!1537 = distinct !DILexicalBlock(scope: !1533, file: !3, line: 627, column: 5)
!1538 = !DILocation(line: 628, column: 25, scope: !1537)
!1539 = !DILocation(line: 627, column: 5, scope: !1533)
!1540 = !DILocation(line: 629, column: 21, scope: !1541)
!1541 = distinct !DILexicalBlock(scope: !1537, file: !3, line: 628, column: 69)
!1542 = !DILocation(line: 630, column: 2, scope: !1541)
!1543 = !DILocation(line: 631, column: 45, scope: !1544)
!1544 = distinct !DILexicalBlock(scope: !1537, file: !3, line: 630, column: 7)
!1545 = !DILocation(line: 631, column: 21, scope: !1544)
!1546 = !DILocation(line: 634, column: 45, scope: !1533)
!1547 = !DILocation(line: 634, column: 36, scope: !1533)
!1548 = !DILocation(line: 634, column: 21, scope: !1533)
!1549 = !DILocation(line: 634, column: 20, scope: !1533)
!1550 = !DILocation(line: 634, column: 18, scope: !1533)
!1551 = !DILocation(line: 636, column: 11, scope: !1533)
!1552 = !DILocation(line: 636, column: 27, scope: !1533)
!1553 = !DILocation(line: 636, column: 32, scope: !1533)
!1554 = !DILocation(line: 636, column: 9, scope: !1533)
!1555 = !DILocation(line: 637, column: 12, scope: !1533)
!1556 = !DILocation(line: 637, column: 28, scope: !1533)
!1557 = !DILocation(line: 637, column: 10, scope: !1533)
!1558 = !DILocation(line: 638, column: 12, scope: !1533)
!1559 = !DILocation(line: 638, column: 28, scope: !1533)
!1560 = !DILocation(line: 638, column: 10, scope: !1533)
!1561 = !DILocation(line: 640, column: 25, scope: !1533)
!1562 = !DILocation(line: 640, column: 18, scope: !1533)
!1563 = !DILocation(line: 640, column: 9, scope: !1533)
!1564 = !DILocation(line: 640, column: 8, scope: !1533)
!1565 = !DILocation(line: 641, column: 26, scope: !1533)
!1566 = !DILocation(line: 641, column: 19, scope: !1533)
!1567 = !DILocation(line: 641, column: 10, scope: !1533)
!1568 = !DILocation(line: 641, column: 9, scope: !1533)
!1569 = !DILocation(line: 642, column: 26, scope: !1533)
!1570 = !DILocation(line: 642, column: 19, scope: !1533)
!1571 = !DILocation(line: 642, column: 10, scope: !1533)
!1572 = !DILocation(line: 642, column: 9, scope: !1533)
!1573 = !DILocation(line: 644, column: 24, scope: !1533)
!1574 = !DILocation(line: 644, column: 2, scope: !1533)
!1575 = !DILocation(line: 645, column: 25, scope: !1533)
!1576 = !DILocation(line: 645, column: 2, scope: !1533)
!1577 = !DILocation(line: 646, column: 25, scope: !1533)
!1578 = !DILocation(line: 646, column: 2, scope: !1533)
!1579 = !DILocation(line: 647, column: 1, scope: !1533)
!1580 = distinct !DISubprogram(name: "dim3", linkageName: "_ZN4dim3C2Ejjj", scope: !1582, file: !1581, line: 421, type: !1588, scopeLine: 421, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, declaration: !1587, retainedNodes: !969)
!1581 = !DIFile(filename: "/usr/local/cuda/include/vector_types.h", directory: "")
!1582 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "dim3", file: !1581, line: 417, size: 96, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !1583, identifier: "_ZTS4dim3")
!1583 = !{!1584, !1585, !1586, !1587, !1591, !1600}
!1584 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !1582, file: !1581, line: 419, baseType: !7, size: 32)
!1585 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !1582, file: !1581, line: 419, baseType: !7, size: 32, offset: 32)
!1586 = !DIDerivedType(tag: DW_TAG_member, name: "z", scope: !1582, file: !1581, line: 419, baseType: !7, size: 32, offset: 64)
!1587 = !DISubprogram(name: "dim3", scope: !1582, file: !1581, line: 421, type: !1588, scopeLine: 421, flags: DIFlagPrototyped, spFlags: 0)
!1588 = !DISubroutineType(types: !1589)
!1589 = !{null, !1590, !7, !7, !7}
!1590 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !1582, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!1591 = !DISubprogram(name: "dim3", scope: !1582, file: !1581, line: 422, type: !1592, scopeLine: 422, flags: DIFlagPrototyped, spFlags: 0)
!1592 = !DISubroutineType(types: !1593)
!1593 = !{null, !1590, !1594}
!1594 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint3", file: !1581, line: 383, baseType: !1595)
!1595 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "uint3", file: !1581, line: 190, size: 96, flags: DIFlagTypePassByValue, elements: !1596, identifier: "_ZTS5uint3")
!1596 = !{!1597, !1598, !1599}
!1597 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !1595, file: !1581, line: 192, baseType: !7, size: 32)
!1598 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !1595, file: !1581, line: 192, baseType: !7, size: 32, offset: 32)
!1599 = !DIDerivedType(tag: DW_TAG_member, name: "z", scope: !1595, file: !1581, line: 192, baseType: !7, size: 32, offset: 64)
!1600 = !DISubprogram(name: "operator uint3", linkageName: "_ZN4dim3cv5uint3Ev", scope: !1582, file: !1581, line: 423, type: !1601, scopeLine: 423, flags: DIFlagPrototyped, spFlags: 0)
!1601 = !DISubroutineType(types: !1602)
!1602 = !{!1594, !1590}
!1603 = !DILocalVariable(name: "this", arg: 1, scope: !1580, type: !1604, flags: DIFlagArtificial | DIFlagObjectPointer)
!1604 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !1582, size: 64)
!1605 = !DILocation(line: 0, scope: !1580)
!1606 = !DILocalVariable(name: "vx", arg: 2, scope: !1580, file: !1581, line: 421, type: !7)
!1607 = !DILocation(line: 421, column: 43, scope: !1580)
!1608 = !DILocalVariable(name: "vy", arg: 3, scope: !1580, file: !1581, line: 421, type: !7)
!1609 = !DILocation(line: 421, column: 64, scope: !1580)
!1610 = !DILocalVariable(name: "vz", arg: 4, scope: !1580, file: !1581, line: 421, type: !7)
!1611 = !DILocation(line: 421, column: 85, scope: !1580)
!1612 = !DILocation(line: 421, column: 95, scope: !1580)
!1613 = !DILocation(line: 421, column: 97, scope: !1580)
!1614 = !DILocation(line: 421, column: 102, scope: !1580)
!1615 = !DILocation(line: 421, column: 104, scope: !1580)
!1616 = !DILocation(line: 421, column: 109, scope: !1580)
!1617 = !DILocation(line: 421, column: 111, scope: !1580)
!1618 = !DILocation(line: 421, column: 116, scope: !1580)
!1619 = distinct !DISubprogram(name: "gpu_kernel", linkageName: "_Z10gpu_kernelPdS_S_d", scope: !3, file: !3, line: 464, type: !1620, scopeLine: 467, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !969)
!1620 = !DISubroutineType(types: !1621)
!1621 = !{null, !98, !98, !98, !99}
!1622 = !DILocalVariable(name: "q_global", arg: 1, scope: !1619, file: !3, line: 464, type: !98)
!1623 = !DILocation(line: 464, column: 36, scope: !1619)
!1624 = !DILocalVariable(name: "sx_global", arg: 2, scope: !1619, file: !3, line: 465, type: !98)
!1625 = !DILocation(line: 465, column: 11, scope: !1619)
!1626 = !DILocalVariable(name: "sy_global", arg: 3, scope: !1619, file: !3, line: 466, type: !98)
!1627 = !DILocation(line: 466, column: 11, scope: !1619)
!1628 = !DILocalVariable(name: "an", arg: 4, scope: !1619, file: !3, line: 467, type: !99)
!1629 = !DILocation(line: 467, column: 10, scope: !1619)
!1630 = !DILocation(line: 467, column: 13, scope: !1619)
!1631 = !DILocation(line: 551, column: 1, scope: !1619)
!1632 = distinct !DISubprogram(name: "release_gpu", linkageName: "_ZL11release_gpuv", scope: !3, file: !3, line: 571, type: !472, scopeLine: 571, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !969)
!1633 = !DILocation(line: 572, column: 11, scope: !1632)
!1634 = !DILocation(line: 572, column: 2, scope: !1632)
!1635 = !DILocation(line: 573, column: 11, scope: !1632)
!1636 = !DILocation(line: 573, column: 2, scope: !1632)
!1637 = !DILocation(line: 574, column: 11, scope: !1632)
!1638 = !DILocation(line: 574, column: 2, scope: !1632)
!1639 = !DILocation(line: 575, column: 1, scope: !1632)
!1640 = distinct !DISubprogram(name: "cudaMalloc<double>", linkageName: "_ZL10cudaMallocIdE9cudaErrorPPT_m", scope: !1641, file: !1641, line: 490, type: !1642, scopeLine: 494, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, templateParams: !1646, retainedNodes: !969)
!1641 = !DIFile(filename: "/usr/local/cuda/include/cuda_runtime.h", directory: "")
!1642 = !DISubroutineType(types: !1643)
!1643 = !{!1644, !1645, !121}
!1644 = !DIDerivedType(tag: DW_TAG_typedef, name: "cudaError_t", file: !6, line: 1419, baseType: !14)
!1645 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !98, size: 64)
!1646 = !{!1647}
!1647 = !DITemplateTypeParameter(name: "T", type: !99)
!1648 = !DILocalVariable(name: "devPtr", arg: 1, scope: !1640, file: !1641, line: 491, type: !1645)
!1649 = !DILocation(line: 491, column: 12, scope: !1640)
!1650 = !DILocalVariable(name: "size", arg: 2, scope: !1640, file: !1641, line: 492, type: !121)
!1651 = !DILocation(line: 492, column: 12, scope: !1640)
!1652 = !DILocation(line: 495, column: 38, scope: !1640)
!1653 = !DILocation(line: 495, column: 23, scope: !1640)
!1654 = !DILocation(line: 495, column: 46, scope: !1640)
!1655 = !DILocation(line: 495, column: 10, scope: !1640)
!1656 = !DILocation(line: 495, column: 3, scope: !1640)
