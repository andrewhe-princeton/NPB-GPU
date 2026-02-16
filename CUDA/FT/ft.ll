; ModuleID = 'ft.cu'
source_filename = "ft.cu"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.dcomplex = type { double, double }
%struct.cudaDeviceProp = type { [256 x i8], i64, i64, i32, i32, i64, i32, [3 x i32], [3 x i32], i32, i64, i32, i32, i64, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, [2 x i32], [2 x i32], [3 x i32], [2 x i32], [3 x i32], [3 x i32], i32, [2 x i32], [3 x i32], [2 x i32], i32, [2 x i32], [3 x i32], [2 x i32], [3 x i32], i32, [2 x i32], i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i64, i32, i32, i32, i32 }
%struct.dim3 = type { i32, i32, i32 }
%struct.CUstream_st = type opaque

$_ZN4dim3C2Ejjj = comdat any

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
@.str.61 = private unnamed_addr constant [12 x i8] c"16 Feb 2026\00", align 1
@.str.62 = private unnamed_addr constant [6 x i8] c"\DC\7FR\FE\7F\00", align 1
@.str.63 = private unnamed_addr constant [42 x i8] c"Intel(R) Xeon(R) CPU E5-2697 v3 @ 2.60GHz\00", align 1
@.str.64 = private unnamed_addr constant [23 x i8] c"${NVCC} ${EXTRA_STUFF}\00", align 1
@.str.65 = private unnamed_addr constant [6 x i8] c"$(CC)\00", align 1
@.str.66 = private unnamed_addr constant [5 x i8] c"-lm \00", align 1
@.str.67 = private unnamed_addr constant [13 x i8] c"-I../common \00", align 1
@.str.68 = private unnamed_addr constant [4 x i8] c"-O3\00", align 1
@.str.69 = private unnamed_addr constant [7 x i8] c"randdp\00", align 1
@.str.70 = private unnamed_addr constant [65 x i8] c"\0A\0A NAS Parallel Benchmarks 4.1 CUDA C++ version - FT Benchmark\0A\0A\00", align 1
@.str.71 = private unnamed_addr constant [36 x i8] c" Size                : %4dx%4dx%4d\0A\00", align 1
@.str.72 = private unnamed_addr constant [35 x i8] c" Iterations                  :%7d\0A\00", align 1
@.str.73 = private unnamed_addr constant [33 x i8] c" Result verification successful\0A\00", align 1
@.str.74 = private unnamed_addr constant [29 x i8] c" Result verification failed\0A\00", align 1
@.str.75 = private unnamed_addr constant [17 x i8] c" class_npb = %c\0A\00", align 1

; Function Attrs: noinline nounwind uwtable
define dso_local double @_Z6randlcPdd(double* %x, double %a) #0 !dbg !1055 {
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
  call void @llvm.dbg.declare(metadata double** %x.addr, metadata !1059, metadata !DIExpression()), !dbg !1060
  store double %a, double* %a.addr, align 8
  call void @llvm.dbg.declare(metadata double* %a.addr, metadata !1061, metadata !DIExpression()), !dbg !1062
  call void @llvm.dbg.declare(metadata double* %t1, metadata !1063, metadata !DIExpression()), !dbg !1064
  call void @llvm.dbg.declare(metadata double* %t2, metadata !1065, metadata !DIExpression()), !dbg !1066
  call void @llvm.dbg.declare(metadata double* %t3, metadata !1067, metadata !DIExpression()), !dbg !1068
  call void @llvm.dbg.declare(metadata double* %t4, metadata !1069, metadata !DIExpression()), !dbg !1070
  call void @llvm.dbg.declare(metadata double* %a1, metadata !1071, metadata !DIExpression()), !dbg !1072
  call void @llvm.dbg.declare(metadata double* %a2, metadata !1073, metadata !DIExpression()), !dbg !1074
  call void @llvm.dbg.declare(metadata double* %x1, metadata !1075, metadata !DIExpression()), !dbg !1076
  call void @llvm.dbg.declare(metadata double* %x2, metadata !1077, metadata !DIExpression()), !dbg !1078
  call void @llvm.dbg.declare(metadata double* %z, metadata !1079, metadata !DIExpression()), !dbg !1080
  %0 = load double, double* %a.addr, align 8, !dbg !1081
  %mul = fmul contract double 0x3E80000000000000, %0, !dbg !1082
  store double %mul, double* %t1, align 8, !dbg !1083
  %1 = load double, double* %t1, align 8, !dbg !1084
  %conv = fptosi double %1 to i32, !dbg !1084
  %conv1 = sitofp i32 %conv to double, !dbg !1085
  store double %conv1, double* %a1, align 8, !dbg !1086
  %2 = load double, double* %a.addr, align 8, !dbg !1087
  %3 = load double, double* %a1, align 8, !dbg !1088
  %mul2 = fmul contract double 0x4160000000000000, %3, !dbg !1089
  %sub = fsub contract double %2, %mul2, !dbg !1090
  store double %sub, double* %a2, align 8, !dbg !1091
  %4 = load double*, double** %x.addr, align 8, !dbg !1092
  %5 = load double, double* %4, align 8, !dbg !1093
  %mul3 = fmul contract double 0x3E80000000000000, %5, !dbg !1094
  store double %mul3, double* %t1, align 8, !dbg !1095
  %6 = load double, double* %t1, align 8, !dbg !1096
  %conv4 = fptosi double %6 to i32, !dbg !1096
  %conv5 = sitofp i32 %conv4 to double, !dbg !1097
  store double %conv5, double* %x1, align 8, !dbg !1098
  %7 = load double*, double** %x.addr, align 8, !dbg !1099
  %8 = load double, double* %7, align 8, !dbg !1100
  %9 = load double, double* %x1, align 8, !dbg !1101
  %mul6 = fmul contract double 0x4160000000000000, %9, !dbg !1102
  %sub7 = fsub contract double %8, %mul6, !dbg !1103
  store double %sub7, double* %x2, align 8, !dbg !1104
  %10 = load double, double* %a1, align 8, !dbg !1105
  %11 = load double, double* %x2, align 8, !dbg !1106
  %mul8 = fmul contract double %10, %11, !dbg !1107
  %12 = load double, double* %a2, align 8, !dbg !1108
  %13 = load double, double* %x1, align 8, !dbg !1109
  %mul9 = fmul contract double %12, %13, !dbg !1110
  %add = fadd contract double %mul8, %mul9, !dbg !1111
  store double %add, double* %t1, align 8, !dbg !1112
  %14 = load double, double* %t1, align 8, !dbg !1113
  %mul10 = fmul contract double 0x3E80000000000000, %14, !dbg !1114
  %conv11 = fptosi double %mul10 to i32, !dbg !1115
  %conv12 = sitofp i32 %conv11 to double, !dbg !1116
  store double %conv12, double* %t2, align 8, !dbg !1117
  %15 = load double, double* %t1, align 8, !dbg !1118
  %16 = load double, double* %t2, align 8, !dbg !1119
  %mul13 = fmul contract double 0x4160000000000000, %16, !dbg !1120
  %sub14 = fsub contract double %15, %mul13, !dbg !1121
  store double %sub14, double* %z, align 8, !dbg !1122
  %17 = load double, double* %z, align 8, !dbg !1123
  %mul15 = fmul contract double 0x4160000000000000, %17, !dbg !1124
  %18 = load double, double* %a2, align 8, !dbg !1125
  %19 = load double, double* %x2, align 8, !dbg !1126
  %mul16 = fmul contract double %18, %19, !dbg !1127
  %add17 = fadd contract double %mul15, %mul16, !dbg !1128
  store double %add17, double* %t3, align 8, !dbg !1129
  %20 = load double, double* %t3, align 8, !dbg !1130
  %mul18 = fmul contract double 0x3D10000000000000, %20, !dbg !1131
  %conv19 = fptosi double %mul18 to i32, !dbg !1132
  %conv20 = sitofp i32 %conv19 to double, !dbg !1133
  store double %conv20, double* %t4, align 8, !dbg !1134
  %21 = load double, double* %t3, align 8, !dbg !1135
  %22 = load double, double* %t4, align 8, !dbg !1136
  %mul21 = fmul contract double 0x42D0000000000000, %22, !dbg !1137
  %sub22 = fsub contract double %21, %mul21, !dbg !1138
  %23 = load double*, double** %x.addr, align 8, !dbg !1139
  store double %sub22, double* %23, align 8, !dbg !1140
  %24 = load double*, double** %x.addr, align 8, !dbg !1141
  %25 = load double, double* %24, align 8, !dbg !1142
  %mul23 = fmul contract double 0x3D10000000000000, %25, !dbg !1143
  ret double %mul23, !dbg !1144
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: noinline uwtable
define dso_local void @_Z15c_print_resultsPcciiiiddS_iS_S_S_S_S_S_S_S_S_S_S_S_S_S_(i8* %name, i8 signext %class_npb, i32 %n1, i32 %n2, i32 %n3, i32 %niter, double %t, double %mops, i8* %optype, i32 %passed_verification, i8* %npbversion, i8* %compiletime, i8* %compilerversion, i8* %libversion, i8* %cpu_device, i8* %gpu_device, i8* %gpu_config, i8* %cc, i8* %clink, i8* %c_lib, i8* %c_inc, i8* %cflags, i8* %clinkflags, i8* %rand) #2 !dbg !1145 {
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
  call void @llvm.dbg.declare(metadata i8** %name.addr, metadata !1148, metadata !DIExpression()), !dbg !1149
  store i8 %class_npb, i8* %class_npb.addr, align 1
  call void @llvm.dbg.declare(metadata i8* %class_npb.addr, metadata !1150, metadata !DIExpression()), !dbg !1151
  store i32 %n1, i32* %n1.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %n1.addr, metadata !1152, metadata !DIExpression()), !dbg !1153
  store i32 %n2, i32* %n2.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %n2.addr, metadata !1154, metadata !DIExpression()), !dbg !1155
  store i32 %n3, i32* %n3.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %n3.addr, metadata !1156, metadata !DIExpression()), !dbg !1157
  store i32 %niter, i32* %niter.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %niter.addr, metadata !1158, metadata !DIExpression()), !dbg !1159
  store double %t, double* %t.addr, align 8
  call void @llvm.dbg.declare(metadata double* %t.addr, metadata !1160, metadata !DIExpression()), !dbg !1161
  store double %mops, double* %mops.addr, align 8
  call void @llvm.dbg.declare(metadata double* %mops.addr, metadata !1162, metadata !DIExpression()), !dbg !1163
  store i8* %optype, i8** %optype.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %optype.addr, metadata !1164, metadata !DIExpression()), !dbg !1165
  store i32 %passed_verification, i32* %passed_verification.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %passed_verification.addr, metadata !1166, metadata !DIExpression()), !dbg !1167
  store i8* %npbversion, i8** %npbversion.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %npbversion.addr, metadata !1168, metadata !DIExpression()), !dbg !1169
  store i8* %compiletime, i8** %compiletime.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %compiletime.addr, metadata !1170, metadata !DIExpression()), !dbg !1171
  store i8* %compilerversion, i8** %compilerversion.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %compilerversion.addr, metadata !1172, metadata !DIExpression()), !dbg !1173
  store i8* %libversion, i8** %libversion.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %libversion.addr, metadata !1174, metadata !DIExpression()), !dbg !1175
  store i8* %cpu_device, i8** %cpu_device.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %cpu_device.addr, metadata !1176, metadata !DIExpression()), !dbg !1177
  store i8* %gpu_device, i8** %gpu_device.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %gpu_device.addr, metadata !1178, metadata !DIExpression()), !dbg !1179
  store i8* %gpu_config, i8** %gpu_config.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %gpu_config.addr, metadata !1180, metadata !DIExpression()), !dbg !1181
  store i8* %cc, i8** %cc.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %cc.addr, metadata !1182, metadata !DIExpression()), !dbg !1183
  store i8* %clink, i8** %clink.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %clink.addr, metadata !1184, metadata !DIExpression()), !dbg !1185
  store i8* %c_lib, i8** %c_lib.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %c_lib.addr, metadata !1186, metadata !DIExpression()), !dbg !1187
  store i8* %c_inc, i8** %c_inc.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %c_inc.addr, metadata !1188, metadata !DIExpression()), !dbg !1189
  store i8* %cflags, i8** %cflags.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %cflags.addr, metadata !1190, metadata !DIExpression()), !dbg !1191
  store i8* %clinkflags, i8** %clinkflags.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %clinkflags.addr, metadata !1192, metadata !DIExpression()), !dbg !1193
  store i8* %rand, i8** %rand.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %rand.addr, metadata !1194, metadata !DIExpression()), !dbg !1195
  %0 = load i8*, i8** %name.addr, align 8, !dbg !1196
  %call = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([27 x i8], [27 x i8]* @.str, i64 0, i64 0), i8* %0), !dbg !1197
  %1 = load i8, i8* %class_npb.addr, align 1, !dbg !1198
  %conv = sext i8 %1 to i32, !dbg !1198
  %call1 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([46 x i8], [46 x i8]* @.str.1, i64 0, i64 0), i32 %conv), !dbg !1199
  %2 = load i8*, i8** %name.addr, align 8, !dbg !1200
  %arrayidx = getelementptr inbounds i8, i8* %2, i64 0, !dbg !1200
  %3 = load i8, i8* %arrayidx, align 1, !dbg !1200
  %conv2 = sext i8 %3 to i32, !dbg !1200
  %cmp = icmp eq i32 %conv2, 73, !dbg !1202
  br i1 %cmp, label %land.lhs.true, label %if.else15, !dbg !1203

land.lhs.true:                                    ; preds = %entry
  %4 = load i8*, i8** %name.addr, align 8, !dbg !1204
  %arrayidx3 = getelementptr inbounds i8, i8* %4, i64 1, !dbg !1204
  %5 = load i8, i8* %arrayidx3, align 1, !dbg !1204
  %conv4 = sext i8 %5 to i32, !dbg !1204
  %cmp5 = icmp eq i32 %conv4, 83, !dbg !1205
  br i1 %cmp5, label %if.then, label %if.else15, !dbg !1206

if.then:                                          ; preds = %land.lhs.true
  %6 = load i32, i32* %n3.addr, align 4, !dbg !1207
  %cmp6 = icmp eq i32 %6, 0, !dbg !1210
  br i1 %cmp6, label %if.then7, label %if.else, !dbg !1211

if.then7:                                         ; preds = %if.then
  call void @llvm.dbg.declare(metadata i64* %nn, metadata !1212, metadata !DIExpression()), !dbg !1214
  %7 = load i32, i32* %n1.addr, align 4, !dbg !1215
  %conv8 = sext i32 %7 to i64, !dbg !1215
  store i64 %conv8, i64* %nn, align 8, !dbg !1214
  %8 = load i32, i32* %n2.addr, align 4, !dbg !1216
  %cmp9 = icmp ne i32 %8, 0, !dbg !1218
  br i1 %cmp9, label %if.then10, label %if.end, !dbg !1219

if.then10:                                        ; preds = %if.then7
  %9 = load i32, i32* %n2.addr, align 4, !dbg !1220
  %conv11 = sext i32 %9 to i64, !dbg !1220
  %10 = load i64, i64* %nn, align 8, !dbg !1222
  %mul = mul nsw i64 %10, %conv11, !dbg !1222
  store i64 %mul, i64* %nn, align 8, !dbg !1222
  br label %if.end, !dbg !1223

if.end:                                           ; preds = %if.then10, %if.then7
  %11 = load i64, i64* %nn, align 8, !dbg !1224
  %call12 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([38 x i8], [38 x i8]* @.str.2, i64 0, i64 0), i64 %11), !dbg !1225
  br label %if.end14, !dbg !1226

if.else:                                          ; preds = %if.then
  %12 = load i32, i32* %n1.addr, align 4, !dbg !1227
  %13 = load i32, i32* %n2.addr, align 4, !dbg !1229
  %14 = load i32, i32* %n3.addr, align 4, !dbg !1230
  %call13 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([44 x i8], [44 x i8]* @.str.3, i64 0, i64 0), i32 %12, i32 %13, i32 %14), !dbg !1231
  br label %if.end14

if.end14:                                         ; preds = %if.else, %if.end
  br label %if.end48, !dbg !1232

if.else15:                                        ; preds = %land.lhs.true, %entry
  call void @llvm.dbg.declare(metadata [16 x i8]* %size, metadata !1233, metadata !DIExpression()), !dbg !1238
  call void @llvm.dbg.declare(metadata i32* %j, metadata !1239, metadata !DIExpression()), !dbg !1240
  %15 = load i32, i32* %n2.addr, align 4, !dbg !1241
  %cmp16 = icmp eq i32 %15, 0, !dbg !1243
  br i1 %cmp16, label %land.lhs.true17, label %if.else45, !dbg !1244

land.lhs.true17:                                  ; preds = %if.else15
  %16 = load i32, i32* %n3.addr, align 4, !dbg !1245
  %cmp18 = icmp eq i32 %16, 0, !dbg !1246
  br i1 %cmp18, label %if.then19, label %if.else45, !dbg !1247

if.then19:                                        ; preds = %land.lhs.true17
  %17 = load i8*, i8** %name.addr, align 8, !dbg !1248
  %arrayidx20 = getelementptr inbounds i8, i8* %17, i64 0, !dbg !1248
  %18 = load i8, i8* %arrayidx20, align 1, !dbg !1248
  %conv21 = sext i8 %18 to i32, !dbg !1248
  %cmp22 = icmp eq i32 %conv21, 69, !dbg !1251
  br i1 %cmp22, label %land.lhs.true23, label %if.else42, !dbg !1252

land.lhs.true23:                                  ; preds = %if.then19
  %19 = load i8*, i8** %name.addr, align 8, !dbg !1253
  %arrayidx24 = getelementptr inbounds i8, i8* %19, i64 1, !dbg !1253
  %20 = load i8, i8* %arrayidx24, align 1, !dbg !1253
  %conv25 = sext i8 %20 to i32, !dbg !1253
  %cmp26 = icmp eq i32 %conv25, 80, !dbg !1254
  br i1 %cmp26, label %if.then27, label %if.else42, !dbg !1255

if.then27:                                        ; preds = %land.lhs.true23
  %arraydecay = getelementptr inbounds [16 x i8], [16 x i8]* %size, i64 0, i64 0, !dbg !1256
  %21 = load i32, i32* %n1.addr, align 4, !dbg !1258
  %conv28 = sitofp i32 %21 to double, !dbg !1258
  %call29 = call double @pow(double 2.000000e+00, double %conv28) #7, !dbg !1259
  %call30 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay, i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.4, i64 0, i64 0), double %call29) #7, !dbg !1260
  store i32 14, i32* %j, align 4, !dbg !1261
  %22 = load i32, i32* %j, align 4, !dbg !1262
  %idxprom = sext i32 %22 to i64, !dbg !1264
  %arrayidx31 = getelementptr inbounds [16 x i8], [16 x i8]* %size, i64 0, i64 %idxprom, !dbg !1264
  %23 = load i8, i8* %arrayidx31, align 1, !dbg !1264
  %conv32 = sext i8 %23 to i32, !dbg !1264
  %cmp33 = icmp eq i32 %conv32, 46, !dbg !1265
  br i1 %cmp33, label %if.then34, label %if.end37, !dbg !1266

if.then34:                                        ; preds = %if.then27
  %24 = load i32, i32* %j, align 4, !dbg !1267
  %idxprom35 = sext i32 %24 to i64, !dbg !1269
  %arrayidx36 = getelementptr inbounds [16 x i8], [16 x i8]* %size, i64 0, i64 %idxprom35, !dbg !1269
  store i8 32, i8* %arrayidx36, align 1, !dbg !1270
  %25 = load i32, i32* %j, align 4, !dbg !1271
  %dec = add nsw i32 %25, -1, !dbg !1271
  store i32 %dec, i32* %j, align 4, !dbg !1271
  br label %if.end37, !dbg !1272

if.end37:                                         ; preds = %if.then34, %if.then27
  %26 = load i32, i32* %j, align 4, !dbg !1273
  %add = add nsw i32 %26, 1, !dbg !1274
  %idxprom38 = sext i32 %add to i64, !dbg !1275
  %arrayidx39 = getelementptr inbounds [16 x i8], [16 x i8]* %size, i64 0, i64 %idxprom38, !dbg !1275
  store i8 0, i8* %arrayidx39, align 1, !dbg !1276
  %arraydecay40 = getelementptr inbounds [16 x i8], [16 x i8]* %size, i64 0, i64 0, !dbg !1277
  %call41 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([34 x i8], [34 x i8]* @.str.5, i64 0, i64 0), i8* %arraydecay40), !dbg !1278
  br label %if.end44, !dbg !1279

if.else42:                                        ; preds = %land.lhs.true23, %if.then19
  %27 = load i32, i32* %n1.addr, align 4, !dbg !1280
  %call43 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.6, i64 0, i64 0), i32 %27), !dbg !1282
  br label %if.end44

if.end44:                                         ; preds = %if.else42, %if.end37
  br label %if.end47, !dbg !1283

if.else45:                                        ; preds = %land.lhs.true17, %if.else15
  %28 = load i32, i32* %n1.addr, align 4, !dbg !1284
  %29 = load i32, i32* %n2.addr, align 4, !dbg !1286
  %30 = load i32, i32* %n3.addr, align 4, !dbg !1287
  %call46 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([42 x i8], [42 x i8]* @.str.7, i64 0, i64 0), i32 %28, i32 %29, i32 %30), !dbg !1288
  br label %if.end47

if.end47:                                         ; preds = %if.else45, %if.end44
  br label %if.end48

if.end48:                                         ; preds = %if.end47, %if.end14
  %31 = load i32, i32* %niter.addr, align 4, !dbg !1289
  %call49 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.8, i64 0, i64 0), i32 %31), !dbg !1290
  %32 = load double, double* %t.addr, align 8, !dbg !1291
  %call50 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([39 x i8], [39 x i8]* @.str.9, i64 0, i64 0), double %32), !dbg !1292
  %33 = load double, double* %mops.addr, align 8, !dbg !1293
  %call51 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([39 x i8], [39 x i8]* @.str.10, i64 0, i64 0), double %33), !dbg !1294
  %34 = load i8*, i8** %optype.addr, align 8, !dbg !1295
  %call52 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([25 x i8], [25 x i8]* @.str.11, i64 0, i64 0), i8* %34), !dbg !1296
  %35 = load i32, i32* %passed_verification.addr, align 4, !dbg !1297
  %cmp53 = icmp slt i32 %35, 0, !dbg !1299
  br i1 %cmp53, label %if.then54, label %if.else56, !dbg !1300

if.then54:                                        ; preds = %if.end48
  %call55 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([45 x i8], [45 x i8]* @.str.12, i64 0, i64 0)), !dbg !1301
  br label %if.end62, !dbg !1303

if.else56:                                        ; preds = %if.end48
  %36 = load i32, i32* %passed_verification.addr, align 4, !dbg !1304
  %tobool = icmp ne i32 %36, 0, !dbg !1304
  br i1 %tobool, label %if.then57, label %if.else59, !dbg !1306

if.then57:                                        ; preds = %if.else56
  %call58 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([45 x i8], [45 x i8]* @.str.13, i64 0, i64 0)), !dbg !1307
  br label %if.end61, !dbg !1309

if.else59:                                        ; preds = %if.else56
  %call60 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([45 x i8], [45 x i8]* @.str.14, i64 0, i64 0)), !dbg !1310
  br label %if.end61

if.end61:                                         ; preds = %if.else59, %if.then57
  br label %if.end62

if.end62:                                         ; preds = %if.end61, %if.then54
  %37 = load i8*, i8** %npbversion.addr, align 8, !dbg !1312
  %call63 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.15, i64 0, i64 0), i8* %37), !dbg !1313
  %38 = load i8*, i8** %compiletime.addr, align 8, !dbg !1314
  %call64 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.16, i64 0, i64 0), i8* %38), !dbg !1315
  %39 = load i8*, i8** %compilerversion.addr, align 8, !dbg !1316
  %call65 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.17, i64 0, i64 0), i8* %39), !dbg !1317
  %40 = load i8*, i8** %libversion.addr, align 8, !dbg !1318
  %call66 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.18, i64 0, i64 0), i8* %40), !dbg !1319
  %call67 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([20 x i8], [20 x i8]* @.str.19, i64 0, i64 0)), !dbg !1320
  %41 = load i8*, i8** %cc.addr, align 8, !dbg !1321
  %call68 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.20, i64 0, i64 0), i8* %41), !dbg !1322
  %42 = load i8*, i8** %clink.addr, align 8, !dbg !1323
  %call69 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.21, i64 0, i64 0), i8* %42), !dbg !1324
  %43 = load i8*, i8** %c_lib.addr, align 8, !dbg !1325
  %call70 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.22, i64 0, i64 0), i8* %43), !dbg !1326
  %44 = load i8*, i8** %c_inc.addr, align 8, !dbg !1327
  %call71 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.23, i64 0, i64 0), i8* %44), !dbg !1328
  %45 = load i8*, i8** %cflags.addr, align 8, !dbg !1329
  %call72 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.24, i64 0, i64 0), i8* %45), !dbg !1330
  %46 = load i8*, i8** %clinkflags.addr, align 8, !dbg !1331
  %call73 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.25, i64 0, i64 0), i8* %46), !dbg !1332
  %47 = load i8*, i8** %rand.addr, align 8, !dbg !1333
  %call74 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.26, i64 0, i64 0), i8* %47), !dbg !1334
  %call75 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.27, i64 0, i64 0)), !dbg !1335
  %48 = load i8*, i8** %cpu_device.addr, align 8, !dbg !1336
  %call76 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.28, i64 0, i64 0), i8* %48), !dbg !1337
  %49 = load i8*, i8** %gpu_device.addr, align 8, !dbg !1338
  %call77 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.29, i64 0, i64 0), i8* %49), !dbg !1339
  %call78 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.30, i64 0, i64 0)), !dbg !1340
  %50 = load i8*, i8** %gpu_config.addr, align 8, !dbg !1341
  %call79 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.31, i64 0, i64 0), i8* %50), !dbg !1342
  %call80 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.32, i64 0, i64 0)), !dbg !1343
  %call81 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([72 x i8], [72 x i8]* @.str.33, i64 0, i64 0)), !dbg !1344
  %call82 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([27 x i8], [27 x i8]* @.str.34, i64 0, i64 0)), !dbg !1345
  %call83 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([56 x i8], [56 x i8]* @.str.35, i64 0, i64 0)), !dbg !1346
  %call84 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([52 x i8], [52 x i8]* @.str.36, i64 0, i64 0)), !dbg !1347
  %call85 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([46 x i8], [46 x i8]* @.str.37, i64 0, i64 0)), !dbg !1348
  %call86 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.32, i64 0, i64 0)), !dbg !1349
  %call87 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([43 x i8], [43 x i8]* @.str.38, i64 0, i64 0)), !dbg !1350
  %call88 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([72 x i8], [72 x i8]* @.str.33, i64 0, i64 0)), !dbg !1351
  %call89 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.32, i64 0, i64 0)), !dbg !1352
  ret void, !dbg !1353
}

declare dso_local i32 @printf(i8*, ...) #3

; Function Attrs: nounwind
declare dso_local i32 @sprintf(i8*, i8*, ...) #4

; Function Attrs: nounwind
declare dso_local double @pow(double, double) #4

; Function Attrs: noinline norecurse uwtable
define dso_local i32 @main(i32 %argc, i8** %argv) #5 !dbg !1354 {
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
  call void @llvm.dbg.declare(metadata i32* %argc.addr, metadata !1357, metadata !DIExpression()), !dbg !1358
  store i8** %argv, i8*** %argv.addr, align 8
  call void @llvm.dbg.declare(metadata i8*** %argv.addr, metadata !1359, metadata !DIExpression()), !dbg !1360
  call void @llvm.dbg.declare(metadata i32* %iter, metadata !1361, metadata !DIExpression()), !dbg !1362
  store i32 0, i32* %iter, align 4, !dbg !1362
  call void @llvm.dbg.declare(metadata double* %total_time, metadata !1363, metadata !DIExpression()), !dbg !1364
  call void @llvm.dbg.declare(metadata double* %mflops, metadata !1365, metadata !DIExpression()), !dbg !1366
  call void @llvm.dbg.declare(metadata i32* %verified, metadata !1367, metadata !DIExpression()), !dbg !1369
  call void @llvm.dbg.declare(metadata i8* %class_npb, metadata !1370, metadata !DIExpression()), !dbg !1371
  %call = call noalias i8* @malloc(i64 112) #7, !dbg !1372
  %0 = bitcast i8* %call to %struct.dcomplex*, !dbg !1373
  store %struct.dcomplex* %0, %struct.dcomplex** @_ZL4sums, align 8, !dbg !1374
  %call1 = call noalias i8* @malloc(i64 67108864) #7, !dbg !1375
  %1 = bitcast i8* %call1 to double*, !dbg !1376
  store double* %1, double** @_ZL7twiddle, align 8, !dbg !1377
  %call2 = call noalias i8* @malloc(i64 4096) #7, !dbg !1378
  %2 = bitcast i8* %call2 to %struct.dcomplex*, !dbg !1379
  store %struct.dcomplex* %2, %struct.dcomplex** @_ZL1u, align 8, !dbg !1380
  %call3 = call noalias i8* @malloc(i64 134217728) #7, !dbg !1381
  %3 = bitcast i8* %call3 to %struct.dcomplex*, !dbg !1382
  store %struct.dcomplex* %3, %struct.dcomplex** @_ZL2u0, align 8, !dbg !1383
  %call4 = call noalias i8* @malloc(i64 134217728) #7, !dbg !1384
  %4 = bitcast i8* %call4 to %struct.dcomplex*, !dbg !1385
  store %struct.dcomplex* %4, %struct.dcomplex** @_ZL2u1, align 8, !dbg !1386
  %call5 = call noalias i8* @malloc(i64 12) #7, !dbg !1387
  %5 = bitcast i8* %call5 to i32*, !dbg !1388
  store i32* %5, i32** @_ZL4dims, align 8, !dbg !1389
  call void @_ZL5setupv(), !dbg !1390
  call void @_ZL9setup_gpuv(), !dbg !1391
  %6 = load %struct.dcomplex*, %struct.dcomplex** @u0_device, align 8, !dbg !1392
  %7 = load %struct.dcomplex*, %struct.dcomplex** @u1_device, align 8, !dbg !1393
  %8 = load double*, double** @twiddle_device, align 8, !dbg !1394
  call void @_ZL11init_ui_gpuP8dcomplexS0_Pd(%struct.dcomplex* %6, %struct.dcomplex* %7, double* %8), !dbg !1395
  %9 = load double*, double** @twiddle_device, align 8, !dbg !1396
  call void @_ZL20compute_indexmap_gpuPd(double* %9), !dbg !1397
  %10 = load %struct.dcomplex*, %struct.dcomplex** @u1_device, align 8, !dbg !1398
  call void @_ZL30compute_initial_conditions_gpuP8dcomplex(%struct.dcomplex* %10), !dbg !1399
  call void @_ZL12fft_init_gpui(i32 256), !dbg !1400
  %call6 = call i32 @cudaDeviceSynchronize(), !dbg !1401
  %11 = load %struct.dcomplex*, %struct.dcomplex** @u1_device, align 8, !dbg !1402
  %12 = load %struct.dcomplex*, %struct.dcomplex** @u0_device, align 8, !dbg !1403
  call void @_ZL7fft_gpuiP8dcomplexS0_(i32 1, %struct.dcomplex* %11, %struct.dcomplex* %12), !dbg !1404
  %13 = load double*, double** @twiddle_device, align 8, !dbg !1405
  call void @_ZL20compute_indexmap_gpuPd(double* %13), !dbg !1406
  %14 = load %struct.dcomplex*, %struct.dcomplex** @u1_device, align 8, !dbg !1407
  call void @_ZL30compute_initial_conditions_gpuP8dcomplex(%struct.dcomplex* %14), !dbg !1408
  call void @_ZL12fft_init_gpui(i32 256), !dbg !1409
  %call7 = call i32 @cudaDeviceSynchronize(), !dbg !1410
  %15 = load %struct.dcomplex*, %struct.dcomplex** @u1_device, align 8, !dbg !1411
  %16 = load %struct.dcomplex*, %struct.dcomplex** @u0_device, align 8, !dbg !1412
  call void @_ZL7fft_gpuiP8dcomplexS0_(i32 1, %struct.dcomplex* %15, %struct.dcomplex* %16), !dbg !1413
  store i32 1, i32* %iter, align 4, !dbg !1414
  br label %for.cond, !dbg !1416

for.cond:                                         ; preds = %for.inc, %entry
  %17 = load i32, i32* %iter, align 4, !dbg !1417
  %18 = load i32, i32* @_ZL5niter, align 4, !dbg !1419
  %cmp = icmp sle i32 %17, %18, !dbg !1420
  br i1 %cmp, label %for.body, label %for.end, !dbg !1421

for.body:                                         ; preds = %for.cond
  %19 = load %struct.dcomplex*, %struct.dcomplex** @u0_device, align 8, !dbg !1422
  %20 = load %struct.dcomplex*, %struct.dcomplex** @u1_device, align 8, !dbg !1424
  %21 = load double*, double** @twiddle_device, align 8, !dbg !1425
  call void @_ZL10evolve_gpuP8dcomplexS0_Pd(%struct.dcomplex* %19, %struct.dcomplex* %20, double* %21), !dbg !1426
  %22 = load %struct.dcomplex*, %struct.dcomplex** @u1_device, align 8, !dbg !1427
  %23 = load %struct.dcomplex*, %struct.dcomplex** @u1_device, align 8, !dbg !1428
  call void @_ZL7fft_gpuiP8dcomplexS0_(i32 -1, %struct.dcomplex* %22, %struct.dcomplex* %23), !dbg !1429
  %24 = load i32, i32* %iter, align 4, !dbg !1430
  %25 = load %struct.dcomplex*, %struct.dcomplex** @u1_device, align 8, !dbg !1431
  call void @_ZL12checksum_gpuiP8dcomplex(i32 %24, %struct.dcomplex* %25), !dbg !1432
  br label %for.inc, !dbg !1433

for.inc:                                          ; preds = %for.body
  %26 = load i32, i32* %iter, align 4, !dbg !1434
  %inc = add nsw i32 %26, 1, !dbg !1434
  store i32 %inc, i32* %iter, align 4, !dbg !1434
  br label %for.cond, !dbg !1435, !llvm.loop !1436

for.end:                                          ; preds = %for.cond
  %27 = load %struct.dcomplex*, %struct.dcomplex** @_ZL4sums, align 8, !dbg !1438
  %28 = bitcast %struct.dcomplex* %27 to i8*, !dbg !1438
  %29 = load %struct.dcomplex*, %struct.dcomplex** @sums_device, align 8, !dbg !1439
  %30 = bitcast %struct.dcomplex* %29 to i8*, !dbg !1439
  %31 = load i64, i64* @size_sums_device, align 8, !dbg !1440
  %call8 = call i32 @cudaMemcpy(i8* %28, i8* %30, i64 %31, i32 2), !dbg !1441
  store i32 1, i32* %iter, align 4, !dbg !1442
  br label %for.cond9, !dbg !1444

for.cond9:                                        ; preds = %for.inc15, %for.end
  %32 = load i32, i32* %iter, align 4, !dbg !1445
  %33 = load i32, i32* @_ZL5niter, align 4, !dbg !1447
  %cmp10 = icmp sle i32 %32, %33, !dbg !1448
  br i1 %cmp10, label %for.body11, label %for.end17, !dbg !1449

for.body11:                                       ; preds = %for.cond9
  %34 = load i32, i32* %iter, align 4, !dbg !1450
  %35 = load %struct.dcomplex*, %struct.dcomplex** @_ZL4sums, align 8, !dbg !1452
  %36 = load i32, i32* %iter, align 4, !dbg !1453
  %idxprom = sext i32 %36 to i64, !dbg !1452
  %arrayidx = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %35, i64 %idxprom, !dbg !1452
  %real = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx, i32 0, i32 0, !dbg !1454
  %37 = load double, double* %real, align 8, !dbg !1454
  %38 = load %struct.dcomplex*, %struct.dcomplex** @_ZL4sums, align 8, !dbg !1455
  %39 = load i32, i32* %iter, align 4, !dbg !1456
  %idxprom12 = sext i32 %39 to i64, !dbg !1455
  %arrayidx13 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %38, i64 %idxprom12, !dbg !1455
  %imag = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx13, i32 0, i32 1, !dbg !1457
  %40 = load double, double* %imag, align 8, !dbg !1457
  %call14 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([40 x i8], [40 x i8]* @.str.39, i64 0, i64 0), i32 %34, double %37, double %40), !dbg !1458
  br label %for.inc15, !dbg !1459

for.inc15:                                        ; preds = %for.body11
  %41 = load i32, i32* %iter, align 4, !dbg !1460
  %inc16 = add nsw i32 %41, 1, !dbg !1460
  store i32 %inc16, i32* %iter, align 4, !dbg !1460
  br label %for.cond9, !dbg !1461, !llvm.loop !1462

for.end17:                                        ; preds = %for.cond9
  %42 = load i32, i32* @_ZL5niter, align 4, !dbg !1464
  call void @_ZL6verifyiiiiPiPc(i32 256, i32 256, i32 128, i32 %42, i32* %verified, i8* %class_npb), !dbg !1465
  store double 0.000000e+00, double* %total_time, align 8, !dbg !1466
  %43 = load double, double* %total_time, align 8, !dbg !1467
  %cmp18 = fcmp une double %43, 0.000000e+00, !dbg !1469
  br i1 %cmp18, label %if.then, label %if.else, !dbg !1470

if.then:                                          ; preds = %for.end17
  %call19 = call double @log(double 0x4160000000000000) #7, !dbg !1471
  %mul = fmul contract double 7.196410e+00, %call19, !dbg !1473
  %add = fadd contract double 1.481570e+01, %mul, !dbg !1474
  %call20 = call double @log(double 0x4160000000000000) #7, !dbg !1475
  %mul21 = fmul contract double 7.211130e+00, %call20, !dbg !1476
  %add22 = fadd contract double 5.235180e+00, %mul21, !dbg !1477
  %44 = load i32, i32* @_ZL5niter, align 4, !dbg !1478
  %conv = sitofp i32 %44 to double, !dbg !1478
  %mul23 = fmul contract double %add22, %conv, !dbg !1479
  %add24 = fadd contract double %add, %mul23, !dbg !1480
  %mul25 = fmul contract double 0x4020C6F7A0B5ED8D, %add24, !dbg !1481
  %45 = load double, double* %total_time, align 8, !dbg !1482
  %div = fdiv double %mul25, %45, !dbg !1483
  store double %div, double* %mflops, align 8, !dbg !1484
  br label %if.end, !dbg !1485

if.else:                                          ; preds = %for.end17
  store double 0.000000e+00, double* %mflops, align 8, !dbg !1486
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  call void @llvm.dbg.declare(metadata [256 x i8]* %gpu_config, metadata !1488, metadata !DIExpression()), !dbg !1489
  call void @llvm.dbg.declare(metadata [2048 x i8]* %gpu_config_string, metadata !1490, metadata !DIExpression()), !dbg !1494
  %arraydecay = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1495
  %call26 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay, i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.40, i64 0, i64 0), i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.41, i64 0, i64 0), i8* getelementptr inbounds ([18 x i8], [18 x i8]* @.str.42, i64 0, i64 0)) #7, !dbg !1496
  %arraydecay27 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !1497
  %arraydecay28 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1498
  %call29 = call i8* @strcpy(i8* %arraydecay27, i8* %arraydecay28) #7, !dbg !1499
  %arraydecay30 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1500
  %46 = load i32, i32* @threads_per_block_on_compute_indexmap, align 4, !dbg !1501
  %call31 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay30, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.43, i64 0, i64 0), i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.44, i64 0, i64 0), i32 %46) #7, !dbg !1502
  %arraydecay32 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !1503
  %arraydecay33 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1504
  %call34 = call i8* @strcat(i8* %arraydecay32, i8* %arraydecay33) #7, !dbg !1505
  %arraydecay35 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1506
  %47 = load i32, i32* @threads_per_block_on_compute_initial_conditions, align 4, !dbg !1507
  %call36 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay35, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.43, i64 0, i64 0), i8* getelementptr inbounds ([20 x i8], [20 x i8]* @.str.45, i64 0, i64 0), i32 %47) #7, !dbg !1508
  %arraydecay37 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !1509
  %arraydecay38 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1510
  %call39 = call i8* @strcat(i8* %arraydecay37, i8* %arraydecay38) #7, !dbg !1511
  %arraydecay40 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1512
  %48 = load i32, i32* @threads_per_block_on_init_ui, align 4, !dbg !1513
  %call41 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay40, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.43, i64 0, i64 0), i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str.46, i64 0, i64 0), i32 %48) #7, !dbg !1514
  %arraydecay42 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !1515
  %arraydecay43 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1516
  %call44 = call i8* @strcat(i8* %arraydecay42, i8* %arraydecay43) #7, !dbg !1517
  %arraydecay45 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1518
  %49 = load i32, i32* @threads_per_block_on_evolve, align 4, !dbg !1519
  %call46 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay45, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.43, i64 0, i64 0), i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.47, i64 0, i64 0), i32 %49) #7, !dbg !1520
  %arraydecay47 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !1521
  %arraydecay48 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1522
  %call49 = call i8* @strcat(i8* %arraydecay47, i8* %arraydecay48) #7, !dbg !1523
  %arraydecay50 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1524
  %50 = load i32, i32* @threads_per_block_on_fftx_1, align 4, !dbg !1525
  %call51 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay50, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.43, i64 0, i64 0), i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.48, i64 0, i64 0), i32 %50) #7, !dbg !1526
  %arraydecay52 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !1527
  %arraydecay53 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1528
  %call54 = call i8* @strcat(i8* %arraydecay52, i8* %arraydecay53) #7, !dbg !1529
  %arraydecay55 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1530
  %51 = load i32, i32* @threads_per_block_on_fftx_2, align 4, !dbg !1531
  %call56 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay55, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.43, i64 0, i64 0), i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.49, i64 0, i64 0), i32 %51) #7, !dbg !1532
  %arraydecay57 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !1533
  %arraydecay58 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1534
  %call59 = call i8* @strcat(i8* %arraydecay57, i8* %arraydecay58) #7, !dbg !1535
  %arraydecay60 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1536
  %52 = load i32, i32* @threads_per_block_on_fftx_3, align 4, !dbg !1537
  %call61 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay60, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.43, i64 0, i64 0), i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.50, i64 0, i64 0), i32 %52) #7, !dbg !1538
  %arraydecay62 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !1539
  %arraydecay63 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1540
  %call64 = call i8* @strcat(i8* %arraydecay62, i8* %arraydecay63) #7, !dbg !1541
  %arraydecay65 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1542
  %53 = load i32, i32* @threads_per_block_on_ffty_1, align 4, !dbg !1543
  %call66 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay65, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.43, i64 0, i64 0), i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.51, i64 0, i64 0), i32 %53) #7, !dbg !1544
  %arraydecay67 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !1545
  %arraydecay68 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1546
  %call69 = call i8* @strcat(i8* %arraydecay67, i8* %arraydecay68) #7, !dbg !1547
  %arraydecay70 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1548
  %54 = load i32, i32* @threads_per_block_on_ffty_2, align 4, !dbg !1549
  %call71 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay70, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.43, i64 0, i64 0), i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.52, i64 0, i64 0), i32 %54) #7, !dbg !1550
  %arraydecay72 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !1551
  %arraydecay73 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1552
  %call74 = call i8* @strcat(i8* %arraydecay72, i8* %arraydecay73) #7, !dbg !1553
  %arraydecay75 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1554
  %55 = load i32, i32* @threads_per_block_on_ffty_3, align 4, !dbg !1555
  %call76 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay75, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.43, i64 0, i64 0), i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.53, i64 0, i64 0), i32 %55) #7, !dbg !1556
  %arraydecay77 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !1557
  %arraydecay78 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1558
  %call79 = call i8* @strcat(i8* %arraydecay77, i8* %arraydecay78) #7, !dbg !1559
  %arraydecay80 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1560
  %56 = load i32, i32* @threads_per_block_on_fftz_1, align 4, !dbg !1561
  %call81 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay80, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.43, i64 0, i64 0), i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.54, i64 0, i64 0), i32 %56) #7, !dbg !1562
  %arraydecay82 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !1563
  %arraydecay83 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1564
  %call84 = call i8* @strcat(i8* %arraydecay82, i8* %arraydecay83) #7, !dbg !1565
  %arraydecay85 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1566
  %57 = load i32, i32* @threads_per_block_on_fftz_2, align 4, !dbg !1567
  %call86 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay85, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.43, i64 0, i64 0), i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.55, i64 0, i64 0), i32 %57) #7, !dbg !1568
  %arraydecay87 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !1569
  %arraydecay88 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1570
  %call89 = call i8* @strcat(i8* %arraydecay87, i8* %arraydecay88) #7, !dbg !1571
  %arraydecay90 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1572
  %58 = load i32, i32* @threads_per_block_on_fftz_3, align 4, !dbg !1573
  %call91 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay90, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.43, i64 0, i64 0), i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.56, i64 0, i64 0), i32 %58) #7, !dbg !1574
  %arraydecay92 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !1575
  %arraydecay93 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1576
  %call94 = call i8* @strcat(i8* %arraydecay92, i8* %arraydecay93) #7, !dbg !1577
  %arraydecay95 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1578
  %59 = load i32, i32* @threads_per_block_on_checksum, align 4, !dbg !1579
  %call96 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay95, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.43, i64 0, i64 0), i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.57, i64 0, i64 0), i32 %59) #7, !dbg !1580
  %arraydecay97 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !1581
  %arraydecay98 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1582
  %call99 = call i8* @strcat(i8* %arraydecay97, i8* %arraydecay98) #7, !dbg !1583
  %60 = load i8, i8* %class_npb, align 1, !dbg !1584
  %61 = load i32, i32* @_ZL5niter, align 4, !dbg !1585
  %62 = load double, double* %total_time, align 8, !dbg !1586
  %63 = load double, double* %mflops, align 8, !dbg !1587
  %64 = load i32, i32* %verified, align 4, !dbg !1588
  %arraydecay100 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !1589
  call void @_Z15c_print_resultsPcciiiiddS_iS_S_S_S_S_S_S_S_S_S_S_S_S_S_(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.58, i64 0, i64 0), i8 signext %60, i32 256, i32 256, i32 128, i32 %61, double %62, double %63, i8* getelementptr inbounds ([25 x i8], [25 x i8]* @.str.59, i64 0, i64 0), i32 %64, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.60, i64 0, i64 0), i8* getelementptr inbounds ([12 x i8], [12 x i8]* @.str.61, i64 0, i64 0), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.62, i64 0, i64 0), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.62, i64 0, i64 0), i8* getelementptr inbounds ([42 x i8], [42 x i8]* @.str.63, i64 0, i64 0), i8* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 0, i64 0), i8* %arraydecay100, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.64, i64 0, i64 0), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.65, i64 0, i64 0), i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.66, i64 0, i64 0), i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.67, i64 0, i64 0), i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.68, i64 0, i64 0), i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.68, i64 0, i64 0), i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.69, i64 0, i64 0)), !dbg !1590
  call void @_ZL11release_gpuv(), !dbg !1591
  %65 = load %struct.dcomplex*, %struct.dcomplex** @_ZL4sums, align 8, !dbg !1592
  %66 = bitcast %struct.dcomplex* %65 to i8*, !dbg !1592
  call void @free(i8* %66) #7, !dbg !1593
  %67 = load double*, double** @_ZL7twiddle, align 8, !dbg !1594
  %68 = bitcast double* %67 to i8*, !dbg !1594
  call void @free(i8* %68) #7, !dbg !1595
  %69 = load %struct.dcomplex*, %struct.dcomplex** @_ZL1u, align 8, !dbg !1596
  %70 = bitcast %struct.dcomplex* %69 to i8*, !dbg !1596
  call void @free(i8* %70) #7, !dbg !1597
  %71 = load %struct.dcomplex*, %struct.dcomplex** @_ZL2u0, align 8, !dbg !1598
  %72 = bitcast %struct.dcomplex* %71 to i8*, !dbg !1598
  call void @free(i8* %72) #7, !dbg !1599
  %73 = load %struct.dcomplex*, %struct.dcomplex** @_ZL2u1, align 8, !dbg !1600
  %74 = bitcast %struct.dcomplex* %73 to i8*, !dbg !1600
  call void @free(i8* %74) #7, !dbg !1601
  %75 = load i32*, i32** @_ZL4dims, align 8, !dbg !1602
  %76 = bitcast i32* %75 to i8*, !dbg !1602
  call void @free(i8* %76) #7, !dbg !1603
  ret i32 0, !dbg !1604
}

; Function Attrs: nounwind
declare dso_local noalias i8* @malloc(i64) #4

; Function Attrs: noinline uwtable
define internal void @_ZL5setupv() #2 !dbg !1605 {
entry:
  store i32 6, i32* @_ZL5niter, align 4, !dbg !1606
  %call = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([65 x i8], [65 x i8]* @.str.70, i64 0, i64 0)), !dbg !1607
  %call1 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([36 x i8], [36 x i8]* @.str.71, i64 0, i64 0), i32 256, i32 256, i32 128), !dbg !1608
  %0 = load i32, i32* @_ZL5niter, align 4, !dbg !1609
  %call2 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([35 x i8], [35 x i8]* @.str.72, i64 0, i64 0), i32 %0), !dbg !1610
  %call3 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.32, i64 0, i64 0)), !dbg !1611
  ret void, !dbg !1612
}

; Function Attrs: noinline uwtable
define internal void @_ZL9setup_gpuv() #2 !dbg !1613 {
entry:
  store i32 32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !1614
  store i32 32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !1615
  %0 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !1616
  %cmp = icmp sle i32 32, %0, !dbg !1618
  br i1 %cmp, label %if.then, label %if.else, !dbg !1619

if.then:                                          ; preds = %entry
  store i32 32, i32* @threads_per_block_on_compute_indexmap, align 4, !dbg !1620
  br label %if.end, !dbg !1622

if.else:                                          ; preds = %entry
  %1 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !1623
  store i32 %1, i32* @threads_per_block_on_compute_indexmap, align 4, !dbg !1625
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %2 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !1626
  %cmp1 = icmp sle i32 32, %2, !dbg !1628
  br i1 %cmp1, label %if.then2, label %if.else3, !dbg !1629

if.then2:                                         ; preds = %if.end
  store i32 32, i32* @threads_per_block_on_compute_initial_conditions, align 4, !dbg !1630
  br label %if.end4, !dbg !1632

if.else3:                                         ; preds = %if.end
  %3 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !1633
  store i32 %3, i32* @threads_per_block_on_compute_initial_conditions, align 4, !dbg !1635
  br label %if.end4

if.end4:                                          ; preds = %if.else3, %if.then2
  %4 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !1636
  %cmp5 = icmp sle i32 32, %4, !dbg !1638
  br i1 %cmp5, label %if.then6, label %if.else7, !dbg !1639

if.then6:                                         ; preds = %if.end4
  store i32 32, i32* @threads_per_block_on_init_ui, align 4, !dbg !1640
  br label %if.end8, !dbg !1642

if.else7:                                         ; preds = %if.end4
  %5 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !1643
  store i32 %5, i32* @threads_per_block_on_init_ui, align 4, !dbg !1645
  br label %if.end8

if.end8:                                          ; preds = %if.else7, %if.then6
  %6 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !1646
  %cmp9 = icmp sle i32 32, %6, !dbg !1648
  br i1 %cmp9, label %if.then10, label %if.else11, !dbg !1649

if.then10:                                        ; preds = %if.end8
  store i32 32, i32* @threads_per_block_on_evolve, align 4, !dbg !1650
  br label %if.end12, !dbg !1652

if.else11:                                        ; preds = %if.end8
  %7 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !1653
  store i32 %7, i32* @threads_per_block_on_evolve, align 4, !dbg !1655
  br label %if.end12

if.end12:                                         ; preds = %if.else11, %if.then10
  %8 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !1656
  %cmp13 = icmp sle i32 1024, %8, !dbg !1658
  br i1 %cmp13, label %if.then14, label %if.else15, !dbg !1659

if.then14:                                        ; preds = %if.end12
  store i32 1024, i32* @threads_per_block_on_fftx_1, align 4, !dbg !1660
  br label %if.end16, !dbg !1662

if.else15:                                        ; preds = %if.end12
  %9 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !1663
  store i32 %9, i32* @threads_per_block_on_fftx_1, align 4, !dbg !1665
  br label %if.end16

if.end16:                                         ; preds = %if.else15, %if.then14
  %10 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !1666
  %cmp17 = icmp sle i32 32, %10, !dbg !1668
  br i1 %cmp17, label %if.then18, label %if.else19, !dbg !1669

if.then18:                                        ; preds = %if.end16
  store i32 32, i32* @threads_per_block_on_fftx_2, align 4, !dbg !1670
  br label %if.end20, !dbg !1672

if.else19:                                        ; preds = %if.end16
  %11 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !1673
  store i32 %11, i32* @threads_per_block_on_fftx_2, align 4, !dbg !1675
  br label %if.end20

if.end20:                                         ; preds = %if.else19, %if.then18
  %12 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !1676
  %cmp21 = icmp sle i32 256, %12, !dbg !1678
  br i1 %cmp21, label %if.then22, label %if.else23, !dbg !1679

if.then22:                                        ; preds = %if.end20
  store i32 256, i32* @threads_per_block_on_fftx_3, align 4, !dbg !1680
  br label %if.end24, !dbg !1682

if.else23:                                        ; preds = %if.end20
  %13 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !1683
  store i32 %13, i32* @threads_per_block_on_fftx_3, align 4, !dbg !1685
  br label %if.end24

if.end24:                                         ; preds = %if.else23, %if.then22
  %14 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !1686
  %cmp25 = icmp sle i32 32, %14, !dbg !1688
  br i1 %cmp25, label %if.then26, label %if.else27, !dbg !1689

if.then26:                                        ; preds = %if.end24
  store i32 32, i32* @threads_per_block_on_ffty_1, align 4, !dbg !1690
  br label %if.end28, !dbg !1692

if.else27:                                        ; preds = %if.end24
  %15 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !1693
  store i32 %15, i32* @threads_per_block_on_ffty_1, align 4, !dbg !1695
  br label %if.end28

if.end28:                                         ; preds = %if.else27, %if.then26
  %16 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !1696
  %cmp29 = icmp sle i32 32, %16, !dbg !1698
  br i1 %cmp29, label %if.then30, label %if.else31, !dbg !1699

if.then30:                                        ; preds = %if.end28
  store i32 32, i32* @threads_per_block_on_ffty_2, align 4, !dbg !1700
  br label %if.end32, !dbg !1702

if.else31:                                        ; preds = %if.end28
  %17 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !1703
  store i32 %17, i32* @threads_per_block_on_ffty_2, align 4, !dbg !1705
  br label %if.end32

if.end32:                                         ; preds = %if.else31, %if.then30
  %18 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !1706
  %cmp33 = icmp sle i32 32, %18, !dbg !1708
  br i1 %cmp33, label %if.then34, label %if.else35, !dbg !1709

if.then34:                                        ; preds = %if.end32
  store i32 32, i32* @threads_per_block_on_ffty_3, align 4, !dbg !1710
  br label %if.end36, !dbg !1712

if.else35:                                        ; preds = %if.end32
  %19 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !1713
  store i32 %19, i32* @threads_per_block_on_ffty_3, align 4, !dbg !1715
  br label %if.end36

if.end36:                                         ; preds = %if.else35, %if.then34
  %20 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !1716
  %cmp37 = icmp sle i32 32, %20, !dbg !1718
  br i1 %cmp37, label %if.then38, label %if.else39, !dbg !1719

if.then38:                                        ; preds = %if.end36
  store i32 32, i32* @threads_per_block_on_fftz_1, align 4, !dbg !1720
  br label %if.end40, !dbg !1722

if.else39:                                        ; preds = %if.end36
  %21 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !1723
  store i32 %21, i32* @threads_per_block_on_fftz_1, align 4, !dbg !1725
  br label %if.end40

if.end40:                                         ; preds = %if.else39, %if.then38
  %22 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !1726
  %cmp41 = icmp sle i32 32, %22, !dbg !1728
  br i1 %cmp41, label %if.then42, label %if.else43, !dbg !1729

if.then42:                                        ; preds = %if.end40
  store i32 32, i32* @threads_per_block_on_fftz_2, align 4, !dbg !1730
  br label %if.end44, !dbg !1732

if.else43:                                        ; preds = %if.end40
  %23 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !1733
  store i32 %23, i32* @threads_per_block_on_fftz_2, align 4, !dbg !1735
  br label %if.end44

if.end44:                                         ; preds = %if.else43, %if.then42
  %24 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !1736
  %cmp45 = icmp sle i32 32, %24, !dbg !1738
  br i1 %cmp45, label %if.then46, label %if.else47, !dbg !1739

if.then46:                                        ; preds = %if.end44
  store i32 32, i32* @threads_per_block_on_fftz_3, align 4, !dbg !1740
  br label %if.end48, !dbg !1742

if.else47:                                        ; preds = %if.end44
  %25 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !1743
  store i32 %25, i32* @threads_per_block_on_fftz_3, align 4, !dbg !1745
  br label %if.end48

if.end48:                                         ; preds = %if.else47, %if.then46
  %26 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !1746
  %cmp49 = icmp sle i32 32, %26, !dbg !1748
  br i1 %cmp49, label %if.then50, label %if.else51, !dbg !1749

if.then50:                                        ; preds = %if.end48
  store i32 32, i32* @threads_per_block_on_checksum, align 4, !dbg !1750
  br label %if.end52, !dbg !1752

if.else51:                                        ; preds = %if.end48
  %27 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !1753
  store i32 %27, i32* @threads_per_block_on_checksum, align 4, !dbg !1755
  br label %if.end52

if.end52:                                         ; preds = %if.else51, %if.then50
  %28 = load i32, i32* @threads_per_block_on_compute_indexmap, align 4, !dbg !1756
  %conv = sitofp i32 %28 to double, !dbg !1756
  %div = fdiv double 0x4160000000000000, %conv, !dbg !1757
  %29 = call double @llvm.ceil.f64(double %div), !dbg !1758
  %conv53 = fptosi double %29 to i32, !dbg !1758
  store i32 %conv53, i32* @blocks_per_grid_on_compute_indexmap, align 4, !dbg !1759
  %30 = load i32, i32* @threads_per_block_on_compute_initial_conditions, align 4, !dbg !1760
  %conv54 = sitofp i32 %30 to double, !dbg !1760
  %div55 = fdiv double 1.280000e+02, %conv54, !dbg !1761
  %31 = call double @llvm.ceil.f64(double %div55), !dbg !1762
  %conv56 = fptosi double %31 to i32, !dbg !1762
  store i32 %conv56, i32* @blocks_per_grid_on_compute_initial_conditions, align 4, !dbg !1763
  %32 = load i32, i32* @threads_per_block_on_init_ui, align 4, !dbg !1764
  %conv57 = sitofp i32 %32 to double, !dbg !1764
  %div58 = fdiv double 0x4160000000000000, %conv57, !dbg !1765
  %33 = call double @llvm.ceil.f64(double %div58), !dbg !1766
  %conv59 = fptosi double %33 to i32, !dbg !1766
  store i32 %conv59, i32* @blocks_per_grid_on_init_ui, align 4, !dbg !1767
  %34 = load i32, i32* @threads_per_block_on_evolve, align 4, !dbg !1768
  %conv60 = sitofp i32 %34 to double, !dbg !1768
  %div61 = fdiv double 0x4160000000000000, %conv60, !dbg !1769
  %35 = call double @llvm.ceil.f64(double %div61), !dbg !1770
  %conv62 = fptosi double %35 to i32, !dbg !1770
  store i32 %conv62, i32* @blocks_per_grid_on_evolve, align 4, !dbg !1771
  %36 = load i32, i32* @threads_per_block_on_fftx_1, align 4, !dbg !1772
  %conv63 = sitofp i32 %36 to double, !dbg !1772
  %div64 = fdiv double 0x4160000000000000, %conv63, !dbg !1773
  %37 = call double @llvm.ceil.f64(double %div64), !dbg !1774
  %conv65 = fptosi double %37 to i32, !dbg !1774
  store i32 %conv65, i32* @blocks_per_grid_on_fftx_1, align 4, !dbg !1775
  %38 = load i32, i32* @threads_per_block_on_fftx_2, align 4, !dbg !1776
  %conv66 = sitofp i32 %38 to double, !dbg !1776
  %div67 = fdiv double 3.276800e+04, %conv66, !dbg !1777
  %39 = call double @llvm.ceil.f64(double %div67), !dbg !1778
  %conv68 = fptosi double %39 to i32, !dbg !1778
  store i32 %conv68, i32* @blocks_per_grid_on_fftx_2, align 4, !dbg !1779
  %40 = load i32, i32* @threads_per_block_on_fftx_3, align 4, !dbg !1780
  %conv69 = sitofp i32 %40 to double, !dbg !1780
  %div70 = fdiv double 0x4160000000000000, %conv69, !dbg !1781
  %41 = call double @llvm.ceil.f64(double %div70), !dbg !1782
  %conv71 = fptosi double %41 to i32, !dbg !1782
  store i32 %conv71, i32* @blocks_per_grid_on_fftx_3, align 4, !dbg !1783
  %42 = load i32, i32* @threads_per_block_on_ffty_1, align 4, !dbg !1784
  %conv72 = sitofp i32 %42 to double, !dbg !1784
  %div73 = fdiv double 0x4160000000000000, %conv72, !dbg !1785
  %43 = call double @llvm.ceil.f64(double %div73), !dbg !1786
  %conv74 = fptosi double %43 to i32, !dbg !1786
  store i32 %conv74, i32* @blocks_per_grid_on_ffty_1, align 4, !dbg !1787
  %44 = load i32, i32* @threads_per_block_on_ffty_2, align 4, !dbg !1788
  %conv75 = sitofp i32 %44 to double, !dbg !1788
  %div76 = fdiv double 3.276800e+04, %conv75, !dbg !1789
  %45 = call double @llvm.ceil.f64(double %div76), !dbg !1790
  %conv77 = fptosi double %45 to i32, !dbg !1790
  store i32 %conv77, i32* @blocks_per_grid_on_ffty_2, align 4, !dbg !1791
  %46 = load i32, i32* @threads_per_block_on_ffty_3, align 4, !dbg !1792
  %conv78 = sitofp i32 %46 to double, !dbg !1792
  %div79 = fdiv double 0x4160000000000000, %conv78, !dbg !1793
  %47 = call double @llvm.ceil.f64(double %div79), !dbg !1794
  %conv80 = fptosi double %47 to i32, !dbg !1794
  store i32 %conv80, i32* @blocks_per_grid_on_ffty_3, align 4, !dbg !1795
  %48 = load i32, i32* @threads_per_block_on_fftz_1, align 4, !dbg !1796
  %conv81 = sitofp i32 %48 to double, !dbg !1796
  %div82 = fdiv double 0x4160000000000000, %conv81, !dbg !1797
  %49 = call double @llvm.ceil.f64(double %div82), !dbg !1798
  %conv83 = fptosi double %49 to i32, !dbg !1798
  store i32 %conv83, i32* @blocks_per_grid_on_fftz_1, align 4, !dbg !1799
  %50 = load i32, i32* @threads_per_block_on_fftz_2, align 4, !dbg !1800
  %conv84 = sitofp i32 %50 to double, !dbg !1800
  %div85 = fdiv double 6.553600e+04, %conv84, !dbg !1801
  %51 = call double @llvm.ceil.f64(double %div85), !dbg !1802
  %conv86 = fptosi double %51 to i32, !dbg !1802
  store i32 %conv86, i32* @blocks_per_grid_on_fftz_2, align 4, !dbg !1803
  %52 = load i32, i32* @threads_per_block_on_fftz_3, align 4, !dbg !1804
  %conv87 = sitofp i32 %52 to double, !dbg !1804
  %div88 = fdiv double 0x4160000000000000, %conv87, !dbg !1805
  %53 = call double @llvm.ceil.f64(double %div88), !dbg !1806
  %conv89 = fptosi double %53 to i32, !dbg !1806
  store i32 %conv89, i32* @blocks_per_grid_on_fftz_3, align 4, !dbg !1807
  %54 = load i32, i32* @threads_per_block_on_checksum, align 4, !dbg !1808
  %conv90 = sitofp i32 %54 to double, !dbg !1808
  %div91 = fdiv double 1.024000e+03, %conv90, !dbg !1809
  %55 = call double @llvm.ceil.f64(double %div91), !dbg !1810
  %conv92 = fptosi double %55 to i32, !dbg !1810
  store i32 %conv92, i32* @blocks_per_grid_on_checksum, align 4, !dbg !1811
  store i64 112, i64* @size_sums_device, align 8, !dbg !1812
  store i64 1024, i64* @size_starts_device, align 8, !dbg !1813
  store i64 67108864, i64* @size_twiddle_device, align 8, !dbg !1814
  store i64 4096, i64* @size_u_device, align 8, !dbg !1815
  store i64 134217728, i64* @size_u0_device, align 8, !dbg !1816
  store i64 134217728, i64* @size_u1_device, align 8, !dbg !1817
  store i64 134217728, i64* @size_y0_device, align 8, !dbg !1818
  store i64 134217728, i64* @size_y1_device, align 8, !dbg !1819
  %56 = load i32, i32* @threads_per_block_on_checksum, align 4, !dbg !1820
  %conv93 = sext i32 %56 to i64, !dbg !1820
  %mul = mul i64 %conv93, 16, !dbg !1821
  store i64 %mul, i64* @size_shared_data, align 8, !dbg !1822
  %57 = load i64, i64* @size_sums_device, align 8, !dbg !1823
  %call = call i32 @_ZL10cudaMallocI8dcomplexE9cudaErrorPPT_m(%struct.dcomplex** @sums_device, i64 %57), !dbg !1824
  %58 = load i64, i64* @size_starts_device, align 8, !dbg !1825
  %call94 = call i32 @_ZL10cudaMallocIdE9cudaErrorPPT_m(double** @starts_device, i64 %58), !dbg !1826
  %59 = load i64, i64* @size_twiddle_device, align 8, !dbg !1827
  %call95 = call i32 @_ZL10cudaMallocIdE9cudaErrorPPT_m(double** @twiddle_device, i64 %59), !dbg !1828
  %60 = load i64, i64* @size_u_device, align 8, !dbg !1829
  %call96 = call i32 @_ZL10cudaMallocI8dcomplexE9cudaErrorPPT_m(%struct.dcomplex** @u_device, i64 %60), !dbg !1830
  %61 = load i64, i64* @size_u0_device, align 8, !dbg !1831
  %call97 = call i32 @_ZL10cudaMallocI8dcomplexE9cudaErrorPPT_m(%struct.dcomplex** @u0_device, i64 %61), !dbg !1832
  %62 = load i64, i64* @size_u1_device, align 8, !dbg !1833
  %call98 = call i32 @_ZL10cudaMallocI8dcomplexE9cudaErrorPPT_m(%struct.dcomplex** @u1_device, i64 %62), !dbg !1834
  %63 = load i64, i64* @size_y0_device, align 8, !dbg !1835
  %call99 = call i32 @_ZL10cudaMallocI8dcomplexE9cudaErrorPPT_m(%struct.dcomplex** @y0_device, i64 %63), !dbg !1836
  %64 = load i64, i64* @size_y1_device, align 8, !dbg !1837
  %call100 = call i32 @_ZL10cudaMallocI8dcomplexE9cudaErrorPPT_m(%struct.dcomplex** @y1_device, i64 %64), !dbg !1838
  call void @omp_set_num_threads(i32 3), !dbg !1839
  ret void, !dbg !1840
}

; Function Attrs: noinline uwtable
define internal void @_ZL11init_ui_gpuP8dcomplexS0_Pd(%struct.dcomplex* %u0, %struct.dcomplex* %u1, double* %twiddle) #2 !dbg !1841 {
entry:
  %u0.addr = alloca %struct.dcomplex*, align 8
  %u1.addr = alloca %struct.dcomplex*, align 8
  %twiddle.addr = alloca double*, align 8
  %agg.tmp = alloca %struct.dim3, align 4
  %agg.tmp1 = alloca %struct.dim3, align 4
  %agg.tmp.coerce = alloca { i64, i32 }, align 4
  %agg.tmp1.coerce = alloca { i64, i32 }, align 4
  store %struct.dcomplex* %u0, %struct.dcomplex** %u0.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %u0.addr, metadata !1844, metadata !DIExpression()), !dbg !1845
  store %struct.dcomplex* %u1, %struct.dcomplex** %u1.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %u1.addr, metadata !1846, metadata !DIExpression()), !dbg !1847
  store double* %twiddle, double** %twiddle.addr, align 8
  call void @llvm.dbg.declare(metadata double** %twiddle.addr, metadata !1848, metadata !DIExpression()), !dbg !1849
  %0 = load i32, i32* @blocks_per_grid_on_init_ui, align 4, !dbg !1850
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp, i32 %0, i32 1, i32 1), !dbg !1850
  %1 = load i32, i32* @threads_per_block_on_init_ui, align 4, !dbg !1851
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp1, i32 %1, i32 1, i32 1), !dbg !1851
  %2 = bitcast { i64, i32 }* %agg.tmp.coerce to i8*, !dbg !1852
  %3 = bitcast %struct.dim3* %agg.tmp to i8*, !dbg !1852
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %2, i8* align 4 %3, i64 12, i1 false), !dbg !1852
  %4 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 0, !dbg !1852
  %5 = load i64, i64* %4, align 4, !dbg !1852
  %6 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 1, !dbg !1852
  %7 = load i32, i32* %6, align 4, !dbg !1852
  %8 = bitcast { i64, i32 }* %agg.tmp1.coerce to i8*, !dbg !1852
  %9 = bitcast %struct.dim3* %agg.tmp1 to i8*, !dbg !1852
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %8, i8* align 4 %9, i64 12, i1 false), !dbg !1852
  %10 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp1.coerce, i32 0, i32 0, !dbg !1852
  %11 = load i64, i64* %10, align 4, !dbg !1852
  %12 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp1.coerce, i32 0, i32 1, !dbg !1852
  %13 = load i32, i32* %12, align 4, !dbg !1852
  %call = call i32 @cudaConfigureCall(i64 %5, i32 %7, i64 %11, i32 %13, i64 0, %struct.CUstream_st* null), !dbg !1852
  %tobool = icmp ne i32 %call, 0, !dbg !1852
  br i1 %tobool, label %kcall.end, label %kcall.configok, !dbg !1853

kcall.configok:                                   ; preds = %entry
  %14 = load %struct.dcomplex*, %struct.dcomplex** %u0.addr, align 8, !dbg !1854
  %15 = load %struct.dcomplex*, %struct.dcomplex** %u1.addr, align 8, !dbg !1855
  %16 = load double*, double** %twiddle.addr, align 8, !dbg !1856
  call void @_Z18init_ui_gpu_kernelP8dcomplexS0_Pd(%struct.dcomplex* %14, %struct.dcomplex* %15, double* %16), !dbg !1853
  br label %kcall.end, !dbg !1853

kcall.end:                                        ; preds = %kcall.configok, %entry
  ret void, !dbg !1857
}

; Function Attrs: noinline uwtable
define internal void @_ZL20compute_indexmap_gpuPd(double* %twiddle) #2 !dbg !1858 {
entry:
  %twiddle.addr = alloca double*, align 8
  %agg.tmp = alloca %struct.dim3, align 4
  %agg.tmp1 = alloca %struct.dim3, align 4
  %agg.tmp.coerce = alloca { i64, i32 }, align 4
  %agg.tmp1.coerce = alloca { i64, i32 }, align 4
  store double* %twiddle, double** %twiddle.addr, align 8
  call void @llvm.dbg.declare(metadata double** %twiddle.addr, metadata !1861, metadata !DIExpression()), !dbg !1862
  %0 = load i32, i32* @blocks_per_grid_on_compute_indexmap, align 4, !dbg !1863
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp, i32 %0, i32 1, i32 1), !dbg !1863
  %1 = load i32, i32* @threads_per_block_on_compute_indexmap, align 4, !dbg !1864
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp1, i32 %1, i32 1, i32 1), !dbg !1864
  %2 = bitcast { i64, i32 }* %agg.tmp.coerce to i8*, !dbg !1865
  %3 = bitcast %struct.dim3* %agg.tmp to i8*, !dbg !1865
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %2, i8* align 4 %3, i64 12, i1 false), !dbg !1865
  %4 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 0, !dbg !1865
  %5 = load i64, i64* %4, align 4, !dbg !1865
  %6 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 1, !dbg !1865
  %7 = load i32, i32* %6, align 4, !dbg !1865
  %8 = bitcast { i64, i32 }* %agg.tmp1.coerce to i8*, !dbg !1865
  %9 = bitcast %struct.dim3* %agg.tmp1 to i8*, !dbg !1865
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %8, i8* align 4 %9, i64 12, i1 false), !dbg !1865
  %10 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp1.coerce, i32 0, i32 0, !dbg !1865
  %11 = load i64, i64* %10, align 4, !dbg !1865
  %12 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp1.coerce, i32 0, i32 1, !dbg !1865
  %13 = load i32, i32* %12, align 4, !dbg !1865
  %call = call i32 @cudaConfigureCall(i64 %5, i32 %7, i64 %11, i32 %13, i64 0, %struct.CUstream_st* null), !dbg !1865
  %tobool = icmp ne i32 %call, 0, !dbg !1865
  br i1 %tobool, label %kcall.end, label %kcall.configok, !dbg !1866

kcall.configok:                                   ; preds = %entry
  %14 = load double*, double** %twiddle.addr, align 8, !dbg !1867
  call void @_Z27compute_indexmap_gpu_kernelPd(double* %14), !dbg !1866
  br label %kcall.end, !dbg !1866

kcall.end:                                        ; preds = %kcall.configok, %entry
  ret void, !dbg !1868
}

; Function Attrs: noinline uwtable
define internal void @_ZL30compute_initial_conditions_gpuP8dcomplex(%struct.dcomplex* %u0) #2 !dbg !1869 {
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
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %u0.addr, metadata !1872, metadata !DIExpression()), !dbg !1873
  call void @llvm.dbg.declare(metadata i32* %z, metadata !1874, metadata !DIExpression()), !dbg !1875
  call void @llvm.dbg.declare(metadata double* %start, metadata !1876, metadata !DIExpression()), !dbg !1877
  call void @llvm.dbg.declare(metadata double* %an, metadata !1878, metadata !DIExpression()), !dbg !1879
  call void @llvm.dbg.declare(metadata [128 x double]* %starts, metadata !1880, metadata !DIExpression()), !dbg !1884
  store double 0x41B2B9B0A1000000, double* %start, align 8, !dbg !1885
  call void @_ZL6ipow46diPd(double 0x41D2309CE5400000, i32 0, double* %an), !dbg !1886
  %0 = load double, double* %an, align 8, !dbg !1887
  %call = call double @_Z6randlcPdd(double* %start, double %0), !dbg !1888
  call void @_ZL6ipow46diPd(double 0x41D2309CE5400000, i32 131072, double* %an), !dbg !1889
  %1 = load double, double* %start, align 8, !dbg !1890
  %arrayidx = getelementptr inbounds [128 x double], [128 x double]* %starts, i64 0, i64 0, !dbg !1891
  store double %1, double* %arrayidx, align 16, !dbg !1892
  store i32 1, i32* %z, align 4, !dbg !1893
  br label %for.cond, !dbg !1895

for.cond:                                         ; preds = %for.inc, %entry
  %2 = load i32, i32* %z, align 4, !dbg !1896
  %cmp = icmp slt i32 %2, 128, !dbg !1898
  br i1 %cmp, label %for.body, label %for.end, !dbg !1899

for.body:                                         ; preds = %for.cond
  %3 = load double, double* %an, align 8, !dbg !1900
  %call1 = call double @_Z6randlcPdd(double* %start, double %3), !dbg !1902
  %4 = load double, double* %start, align 8, !dbg !1903
  %5 = load i32, i32* %z, align 4, !dbg !1904
  %idxprom = sext i32 %5 to i64, !dbg !1905
  %arrayidx2 = getelementptr inbounds [128 x double], [128 x double]* %starts, i64 0, i64 %idxprom, !dbg !1905
  store double %4, double* %arrayidx2, align 8, !dbg !1906
  br label %for.inc, !dbg !1907

for.inc:                                          ; preds = %for.body
  %6 = load i32, i32* %z, align 4, !dbg !1908
  %inc = add nsw i32 %6, 1, !dbg !1908
  store i32 %inc, i32* %z, align 4, !dbg !1908
  br label %for.cond, !dbg !1909, !llvm.loop !1910

for.end:                                          ; preds = %for.cond
  %7 = load double*, double** @starts_device, align 8, !dbg !1912
  %8 = bitcast double* %7 to i8*, !dbg !1912
  %arraydecay = getelementptr inbounds [128 x double], [128 x double]* %starts, i64 0, i64 0, !dbg !1913
  %9 = bitcast double* %arraydecay to i8*, !dbg !1913
  %10 = load i64, i64* @size_starts_device, align 8, !dbg !1914
  %call3 = call i32 @cudaMemcpy(i8* %8, i8* %9, i64 %10, i32 1), !dbg !1915
  %11 = load i32, i32* @blocks_per_grid_on_compute_initial_conditions, align 4, !dbg !1916
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp, i32 %11, i32 1, i32 1), !dbg !1916
  %12 = load i32, i32* @threads_per_block_on_compute_initial_conditions, align 4, !dbg !1917
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp4, i32 %12, i32 1, i32 1), !dbg !1917
  %13 = bitcast { i64, i32 }* %agg.tmp.coerce to i8*, !dbg !1918
  %14 = bitcast %struct.dim3* %agg.tmp to i8*, !dbg !1918
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %13, i8* align 4 %14, i64 12, i1 false), !dbg !1918
  %15 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 0, !dbg !1918
  %16 = load i64, i64* %15, align 4, !dbg !1918
  %17 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 1, !dbg !1918
  %18 = load i32, i32* %17, align 4, !dbg !1918
  %19 = bitcast { i64, i32 }* %agg.tmp4.coerce to i8*, !dbg !1918
  %20 = bitcast %struct.dim3* %agg.tmp4 to i8*, !dbg !1918
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %19, i8* align 4 %20, i64 12, i1 false), !dbg !1918
  %21 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp4.coerce, i32 0, i32 0, !dbg !1918
  %22 = load i64, i64* %21, align 4, !dbg !1918
  %23 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp4.coerce, i32 0, i32 1, !dbg !1918
  %24 = load i32, i32* %23, align 4, !dbg !1918
  %call5 = call i32 @cudaConfigureCall(i64 %16, i32 %18, i64 %22, i32 %24, i64 0, %struct.CUstream_st* null), !dbg !1918
  %tobool = icmp ne i32 %call5, 0, !dbg !1918
  br i1 %tobool, label %kcall.end, label %kcall.configok, !dbg !1919

kcall.configok:                                   ; preds = %for.end
  %25 = load %struct.dcomplex*, %struct.dcomplex** %u0.addr, align 8, !dbg !1920
  %26 = load double*, double** @starts_device, align 8, !dbg !1921
  call void @_Z37compute_initial_conditions_gpu_kernelP8dcomplexPd(%struct.dcomplex* %25, double* %26), !dbg !1919
  br label %kcall.end, !dbg !1919

kcall.end:                                        ; preds = %kcall.configok, %for.end
  ret void, !dbg !1922
}

; Function Attrs: noinline uwtable
define internal void @_ZL12fft_init_gpui(i32 %n) #2 !dbg !1923 {
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
  call void @llvm.dbg.declare(metadata i32* %n.addr, metadata !1924, metadata !DIExpression()), !dbg !1925
  call void @llvm.dbg.declare(metadata i32* %m, metadata !1926, metadata !DIExpression()), !dbg !1927
  call void @llvm.dbg.declare(metadata i32* %ku, metadata !1928, metadata !DIExpression()), !dbg !1929
  call void @llvm.dbg.declare(metadata i32* %i, metadata !1930, metadata !DIExpression()), !dbg !1931
  call void @llvm.dbg.declare(metadata i32* %j, metadata !1932, metadata !DIExpression()), !dbg !1933
  call void @llvm.dbg.declare(metadata i32* %ln, metadata !1934, metadata !DIExpression()), !dbg !1935
  call void @llvm.dbg.declare(metadata double* %t, metadata !1936, metadata !DIExpression()), !dbg !1937
  call void @llvm.dbg.declare(metadata double* %ti, metadata !1938, metadata !DIExpression()), !dbg !1939
  %0 = load i32, i32* %n.addr, align 4, !dbg !1940
  %call = call i32 @_ZL5ilog2i(i32 %0), !dbg !1941
  store i32 %call, i32* %m, align 4, !dbg !1942
  %real = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp, i32 0, i32 0, !dbg !1943
  %1 = load i32, i32* %m, align 4, !dbg !1943
  %conv = sitofp i32 %1 to double, !dbg !1943
  store double %conv, double* %real, align 8, !dbg !1943
  %imag = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp, i32 0, i32 1, !dbg !1943
  store double 0.000000e+00, double* %imag, align 8, !dbg !1943
  %2 = load %struct.dcomplex*, %struct.dcomplex** @_ZL1u, align 8, !dbg !1944
  %arrayidx = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %2, i64 0, !dbg !1944
  %3 = bitcast %struct.dcomplex* %arrayidx to i8*, !dbg !1945
  %4 = bitcast %struct.dcomplex* %ref.tmp to i8*, !dbg !1945
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %3, i8* align 8 %4, i64 16, i1 false), !dbg !1945
  store i32 2, i32* %ku, align 4, !dbg !1946
  store i32 1, i32* %ln, align 4, !dbg !1947
  store i32 1, i32* %j, align 4, !dbg !1948
  br label %for.cond, !dbg !1950

for.cond:                                         ; preds = %for.inc15, %entry
  %5 = load i32, i32* %j, align 4, !dbg !1951
  %6 = load i32, i32* %m, align 4, !dbg !1953
  %cmp = icmp sle i32 %5, %6, !dbg !1954
  br i1 %cmp, label %for.body, label %for.end17, !dbg !1955

for.body:                                         ; preds = %for.cond
  %7 = load i32, i32* %ln, align 4, !dbg !1956
  %conv1 = sitofp i32 %7 to double, !dbg !1956
  %div = fdiv double 0x400921FB54442D18, %conv1, !dbg !1958
  store double %div, double* %t, align 8, !dbg !1959
  store i32 0, i32* %i, align 4, !dbg !1960
  br label %for.cond2, !dbg !1962

for.cond2:                                        ; preds = %for.inc, %for.body
  %8 = load i32, i32* %i, align 4, !dbg !1963
  %9 = load i32, i32* %ln, align 4, !dbg !1965
  %sub = sub nsw i32 %9, 1, !dbg !1966
  %cmp3 = icmp sle i32 %8, %sub, !dbg !1967
  br i1 %cmp3, label %for.body4, label %for.end, !dbg !1968

for.body4:                                        ; preds = %for.cond2
  %10 = load i32, i32* %i, align 4, !dbg !1969
  %conv5 = sitofp i32 %10 to double, !dbg !1969
  %11 = load double, double* %t, align 8, !dbg !1971
  %mul = fmul contract double %conv5, %11, !dbg !1972
  store double %mul, double* %ti, align 8, !dbg !1973
  %real7 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp6, i32 0, i32 0, !dbg !1974
  %12 = load double, double* %ti, align 8, !dbg !1974
  %call8 = call double @cos(double %12) #7, !dbg !1974
  store double %call8, double* %real7, align 8, !dbg !1974
  %imag9 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp6, i32 0, i32 1, !dbg !1974
  %13 = load double, double* %ti, align 8, !dbg !1974
  %call10 = call double @sin(double %13) #7, !dbg !1974
  store double %call10, double* %imag9, align 8, !dbg !1974
  %14 = load %struct.dcomplex*, %struct.dcomplex** @_ZL1u, align 8, !dbg !1975
  %15 = load i32, i32* %i, align 4, !dbg !1976
  %16 = load i32, i32* %ku, align 4, !dbg !1977
  %add = add nsw i32 %15, %16, !dbg !1978
  %sub11 = sub nsw i32 %add, 1, !dbg !1979
  %idxprom = sext i32 %sub11 to i64, !dbg !1975
  %arrayidx12 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %14, i64 %idxprom, !dbg !1975
  %17 = bitcast %struct.dcomplex* %arrayidx12 to i8*, !dbg !1980
  %18 = bitcast %struct.dcomplex* %ref.tmp6 to i8*, !dbg !1980
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %17, i8* align 8 %18, i64 16, i1 false), !dbg !1980
  br label %for.inc, !dbg !1981

for.inc:                                          ; preds = %for.body4
  %19 = load i32, i32* %i, align 4, !dbg !1982
  %inc = add nsw i32 %19, 1, !dbg !1982
  store i32 %inc, i32* %i, align 4, !dbg !1982
  br label %for.cond2, !dbg !1983, !llvm.loop !1984

for.end:                                          ; preds = %for.cond2
  %20 = load i32, i32* %ku, align 4, !dbg !1986
  %21 = load i32, i32* %ln, align 4, !dbg !1987
  %add13 = add nsw i32 %20, %21, !dbg !1988
  store i32 %add13, i32* %ku, align 4, !dbg !1989
  %22 = load i32, i32* %ln, align 4, !dbg !1990
  %mul14 = mul nsw i32 2, %22, !dbg !1991
  store i32 %mul14, i32* %ln, align 4, !dbg !1992
  br label %for.inc15, !dbg !1993

for.inc15:                                        ; preds = %for.end
  %23 = load i32, i32* %j, align 4, !dbg !1994
  %inc16 = add nsw i32 %23, 1, !dbg !1994
  store i32 %inc16, i32* %j, align 4, !dbg !1994
  br label %for.cond, !dbg !1995, !llvm.loop !1996

for.end17:                                        ; preds = %for.cond
  %24 = load %struct.dcomplex*, %struct.dcomplex** @u_device, align 8, !dbg !1998
  %25 = bitcast %struct.dcomplex* %24 to i8*, !dbg !1998
  %26 = load %struct.dcomplex*, %struct.dcomplex** @_ZL1u, align 8, !dbg !1999
  %27 = bitcast %struct.dcomplex* %26 to i8*, !dbg !1999
  %28 = load i64, i64* @size_u_device, align 8, !dbg !2000
  %call18 = call i32 @cudaMemcpy(i8* %25, i8* %27, i64 %28, i32 1), !dbg !2001
  ret void, !dbg !2002
}

declare dso_local i32 @cudaDeviceSynchronize() #3

; Function Attrs: noinline uwtable
define internal void @_ZL7fft_gpuiP8dcomplexS0_(i32 %dir, %struct.dcomplex* %x1, %struct.dcomplex* %x2) #2 !dbg !2003 {
entry:
  %dir.addr = alloca i32, align 4
  %x1.addr = alloca %struct.dcomplex*, align 8
  %x2.addr = alloca %struct.dcomplex*, align 8
  store i32 %dir, i32* %dir.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %dir.addr, metadata !2006, metadata !DIExpression()), !dbg !2007
  store %struct.dcomplex* %x1, %struct.dcomplex** %x1.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %x1.addr, metadata !2008, metadata !DIExpression()), !dbg !2009
  store %struct.dcomplex* %x2, %struct.dcomplex** %x2.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %x2.addr, metadata !2010, metadata !DIExpression()), !dbg !2011
  %0 = load i32, i32* %dir.addr, align 4, !dbg !2012
  %cmp = icmp eq i32 %0, 1, !dbg !2014
  br i1 %cmp, label %if.then, label %if.else, !dbg !2015

if.then:                                          ; preds = %entry
  %1 = load %struct.dcomplex*, %struct.dcomplex** @u_device, align 8, !dbg !2016
  %2 = load %struct.dcomplex*, %struct.dcomplex** %x1.addr, align 8, !dbg !2018
  %3 = load %struct.dcomplex*, %struct.dcomplex** %x1.addr, align 8, !dbg !2019
  %4 = load %struct.dcomplex*, %struct.dcomplex** @y0_device, align 8, !dbg !2020
  %5 = load %struct.dcomplex*, %struct.dcomplex** @y1_device, align 8, !dbg !2021
  call void @_ZL10cffts1_gpuiP8dcomplexS0_S0_S0_S0_(i32 1, %struct.dcomplex* %1, %struct.dcomplex* %2, %struct.dcomplex* %3, %struct.dcomplex* %4, %struct.dcomplex* %5), !dbg !2022
  %6 = load %struct.dcomplex*, %struct.dcomplex** @u_device, align 8, !dbg !2023
  %7 = load %struct.dcomplex*, %struct.dcomplex** %x1.addr, align 8, !dbg !2024
  %8 = load %struct.dcomplex*, %struct.dcomplex** %x1.addr, align 8, !dbg !2025
  %9 = load %struct.dcomplex*, %struct.dcomplex** @y0_device, align 8, !dbg !2026
  %10 = load %struct.dcomplex*, %struct.dcomplex** @y1_device, align 8, !dbg !2027
  call void @_ZL10cffts2_gpuiP8dcomplexS0_S0_S0_S0_(i32 1, %struct.dcomplex* %6, %struct.dcomplex* %7, %struct.dcomplex* %8, %struct.dcomplex* %9, %struct.dcomplex* %10), !dbg !2028
  %11 = load %struct.dcomplex*, %struct.dcomplex** @u_device, align 8, !dbg !2029
  %12 = load %struct.dcomplex*, %struct.dcomplex** %x1.addr, align 8, !dbg !2030
  %13 = load %struct.dcomplex*, %struct.dcomplex** %x2.addr, align 8, !dbg !2031
  %14 = load %struct.dcomplex*, %struct.dcomplex** @y0_device, align 8, !dbg !2032
  %15 = load %struct.dcomplex*, %struct.dcomplex** @y1_device, align 8, !dbg !2033
  call void @_ZL10cffts3_gpuiP8dcomplexS0_S0_S0_S0_(i32 1, %struct.dcomplex* %11, %struct.dcomplex* %12, %struct.dcomplex* %13, %struct.dcomplex* %14, %struct.dcomplex* %15), !dbg !2034
  br label %if.end, !dbg !2035

if.else:                                          ; preds = %entry
  %16 = load %struct.dcomplex*, %struct.dcomplex** @u_device, align 8, !dbg !2036
  %17 = load %struct.dcomplex*, %struct.dcomplex** %x1.addr, align 8, !dbg !2038
  %18 = load %struct.dcomplex*, %struct.dcomplex** %x1.addr, align 8, !dbg !2039
  %19 = load %struct.dcomplex*, %struct.dcomplex** @y0_device, align 8, !dbg !2040
  %20 = load %struct.dcomplex*, %struct.dcomplex** @y1_device, align 8, !dbg !2041
  call void @_ZL10cffts3_gpuiP8dcomplexS0_S0_S0_S0_(i32 -1, %struct.dcomplex* %16, %struct.dcomplex* %17, %struct.dcomplex* %18, %struct.dcomplex* %19, %struct.dcomplex* %20), !dbg !2042
  %21 = load %struct.dcomplex*, %struct.dcomplex** @u_device, align 8, !dbg !2043
  %22 = load %struct.dcomplex*, %struct.dcomplex** %x1.addr, align 8, !dbg !2044
  %23 = load %struct.dcomplex*, %struct.dcomplex** %x1.addr, align 8, !dbg !2045
  %24 = load %struct.dcomplex*, %struct.dcomplex** @y0_device, align 8, !dbg !2046
  %25 = load %struct.dcomplex*, %struct.dcomplex** @y1_device, align 8, !dbg !2047
  call void @_ZL10cffts2_gpuiP8dcomplexS0_S0_S0_S0_(i32 -1, %struct.dcomplex* %21, %struct.dcomplex* %22, %struct.dcomplex* %23, %struct.dcomplex* %24, %struct.dcomplex* %25), !dbg !2048
  %26 = load %struct.dcomplex*, %struct.dcomplex** @u_device, align 8, !dbg !2049
  %27 = load %struct.dcomplex*, %struct.dcomplex** %x1.addr, align 8, !dbg !2050
  %28 = load %struct.dcomplex*, %struct.dcomplex** %x2.addr, align 8, !dbg !2051
  %29 = load %struct.dcomplex*, %struct.dcomplex** @y0_device, align 8, !dbg !2052
  %30 = load %struct.dcomplex*, %struct.dcomplex** @y1_device, align 8, !dbg !2053
  call void @_ZL10cffts1_gpuiP8dcomplexS0_S0_S0_S0_(i32 -1, %struct.dcomplex* %26, %struct.dcomplex* %27, %struct.dcomplex* %28, %struct.dcomplex* %29, %struct.dcomplex* %30), !dbg !2054
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void, !dbg !2055
}

; Function Attrs: noinline uwtable
define internal void @_ZL10evolve_gpuP8dcomplexS0_Pd(%struct.dcomplex* %u0, %struct.dcomplex* %u1, double* %twiddle) #2 !dbg !2056 {
entry:
  %u0.addr = alloca %struct.dcomplex*, align 8
  %u1.addr = alloca %struct.dcomplex*, align 8
  %twiddle.addr = alloca double*, align 8
  %agg.tmp = alloca %struct.dim3, align 4
  %agg.tmp1 = alloca %struct.dim3, align 4
  %agg.tmp.coerce = alloca { i64, i32 }, align 4
  %agg.tmp1.coerce = alloca { i64, i32 }, align 4
  store %struct.dcomplex* %u0, %struct.dcomplex** %u0.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %u0.addr, metadata !2057, metadata !DIExpression()), !dbg !2058
  store %struct.dcomplex* %u1, %struct.dcomplex** %u1.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %u1.addr, metadata !2059, metadata !DIExpression()), !dbg !2060
  store double* %twiddle, double** %twiddle.addr, align 8
  call void @llvm.dbg.declare(metadata double** %twiddle.addr, metadata !2061, metadata !DIExpression()), !dbg !2062
  %0 = load i32, i32* @blocks_per_grid_on_evolve, align 4, !dbg !2063
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp, i32 %0, i32 1, i32 1), !dbg !2063
  %1 = load i32, i32* @threads_per_block_on_evolve, align 4, !dbg !2064
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp1, i32 %1, i32 1, i32 1), !dbg !2064
  %2 = bitcast { i64, i32 }* %agg.tmp.coerce to i8*, !dbg !2065
  %3 = bitcast %struct.dim3* %agg.tmp to i8*, !dbg !2065
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %2, i8* align 4 %3, i64 12, i1 false), !dbg !2065
  %4 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 0, !dbg !2065
  %5 = load i64, i64* %4, align 4, !dbg !2065
  %6 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 1, !dbg !2065
  %7 = load i32, i32* %6, align 4, !dbg !2065
  %8 = bitcast { i64, i32 }* %agg.tmp1.coerce to i8*, !dbg !2065
  %9 = bitcast %struct.dim3* %agg.tmp1 to i8*, !dbg !2065
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %8, i8* align 4 %9, i64 12, i1 false), !dbg !2065
  %10 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp1.coerce, i32 0, i32 0, !dbg !2065
  %11 = load i64, i64* %10, align 4, !dbg !2065
  %12 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp1.coerce, i32 0, i32 1, !dbg !2065
  %13 = load i32, i32* %12, align 4, !dbg !2065
  %call = call i32 @cudaConfigureCall(i64 %5, i32 %7, i64 %11, i32 %13, i64 0, %struct.CUstream_st* null), !dbg !2065
  %tobool = icmp ne i32 %call, 0, !dbg !2065
  br i1 %tobool, label %kcall.end, label %kcall.configok, !dbg !2066

kcall.configok:                                   ; preds = %entry
  %14 = load %struct.dcomplex*, %struct.dcomplex** %u0.addr, align 8, !dbg !2067
  %15 = load %struct.dcomplex*, %struct.dcomplex** %u1.addr, align 8, !dbg !2068
  %16 = load double*, double** %twiddle.addr, align 8, !dbg !2069
  call void @_Z17evolve_gpu_kernelP8dcomplexS0_Pd(%struct.dcomplex* %14, %struct.dcomplex* %15, double* %16), !dbg !2066
  br label %kcall.end, !dbg !2066

kcall.end:                                        ; preds = %kcall.configok, %entry
  %call2 = call i32 @cudaDeviceSynchronize(), !dbg !2070
  ret void, !dbg !2071
}

; Function Attrs: noinline uwtable
define internal void @_ZL12checksum_gpuiP8dcomplex(i32 %iteration, %struct.dcomplex* %u1) #2 !dbg !2072 {
entry:
  %iteration.addr = alloca i32, align 4
  %u1.addr = alloca %struct.dcomplex*, align 8
  %agg.tmp = alloca %struct.dim3, align 4
  %agg.tmp1 = alloca %struct.dim3, align 4
  %agg.tmp.coerce = alloca { i64, i32 }, align 4
  %agg.tmp1.coerce = alloca { i64, i32 }, align 4
  store i32 %iteration, i32* %iteration.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %iteration.addr, metadata !2075, metadata !DIExpression()), !dbg !2076
  store %struct.dcomplex* %u1, %struct.dcomplex** %u1.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %u1.addr, metadata !2077, metadata !DIExpression()), !dbg !2078
  %0 = load i32, i32* @blocks_per_grid_on_checksum, align 4, !dbg !2079
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp, i32 %0, i32 1, i32 1), !dbg !2079
  %1 = load i32, i32* @threads_per_block_on_checksum, align 4, !dbg !2080
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp1, i32 %1, i32 1, i32 1), !dbg !2080
  %2 = load i64, i64* @size_shared_data, align 8, !dbg !2081
  %3 = bitcast { i64, i32 }* %agg.tmp.coerce to i8*, !dbg !2082
  %4 = bitcast %struct.dim3* %agg.tmp to i8*, !dbg !2082
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %3, i8* align 4 %4, i64 12, i1 false), !dbg !2082
  %5 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 0, !dbg !2082
  %6 = load i64, i64* %5, align 4, !dbg !2082
  %7 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 1, !dbg !2082
  %8 = load i32, i32* %7, align 4, !dbg !2082
  %9 = bitcast { i64, i32 }* %agg.tmp1.coerce to i8*, !dbg !2082
  %10 = bitcast %struct.dim3* %agg.tmp1 to i8*, !dbg !2082
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %9, i8* align 4 %10, i64 12, i1 false), !dbg !2082
  %11 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp1.coerce, i32 0, i32 0, !dbg !2082
  %12 = load i64, i64* %11, align 4, !dbg !2082
  %13 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp1.coerce, i32 0, i32 1, !dbg !2082
  %14 = load i32, i32* %13, align 4, !dbg !2082
  %call = call i32 @cudaConfigureCall(i64 %6, i32 %8, i64 %12, i32 %14, i64 %2, %struct.CUstream_st* null), !dbg !2082
  %tobool = icmp ne i32 %call, 0, !dbg !2082
  br i1 %tobool, label %kcall.end, label %kcall.configok, !dbg !2083

kcall.configok:                                   ; preds = %entry
  %15 = load i32, i32* %iteration.addr, align 4, !dbg !2084
  %16 = load %struct.dcomplex*, %struct.dcomplex** %u1.addr, align 8, !dbg !2085
  %17 = load %struct.dcomplex*, %struct.dcomplex** @sums_device, align 8, !dbg !2086
  call void @_Z19checksum_gpu_kerneliP8dcomplexS0_(i32 %15, %struct.dcomplex* %16, %struct.dcomplex* %17), !dbg !2083
  br label %kcall.end, !dbg !2083

kcall.end:                                        ; preds = %kcall.configok, %entry
  ret void, !dbg !2087
}

declare dso_local i32 @cudaMemcpy(i8*, i8*, i64, i32) #3

; Function Attrs: noinline uwtable
define internal void @_ZL6verifyiiiiPiPc(i32 %d1, i32 %d2, i32 %d3, i32 %nt, i32* %verified, i8* %class_npb) #2 !dbg !2088 {
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
  call void @llvm.dbg.declare(metadata i32* %d1.addr, metadata !2092, metadata !DIExpression()), !dbg !2093
  store i32 %d2, i32* %d2.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %d2.addr, metadata !2094, metadata !DIExpression()), !dbg !2095
  store i32 %d3, i32* %d3.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %d3.addr, metadata !2096, metadata !DIExpression()), !dbg !2097
  store i32 %nt, i32* %nt.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %nt.addr, metadata !2098, metadata !DIExpression()), !dbg !2099
  store i32* %verified, i32** %verified.addr, align 8
  call void @llvm.dbg.declare(metadata i32** %verified.addr, metadata !2100, metadata !DIExpression()), !dbg !2101
  store i8* %class_npb, i8** %class_npb.addr, align 8
  call void @llvm.dbg.declare(metadata i8** %class_npb.addr, metadata !2102, metadata !DIExpression()), !dbg !2103
  call void @llvm.dbg.declare(metadata i32* %i, metadata !2104, metadata !DIExpression()), !dbg !2105
  call void @llvm.dbg.declare(metadata double* %err, metadata !2106, metadata !DIExpression()), !dbg !2107
  call void @llvm.dbg.declare(metadata double* %epsilon, metadata !2108, metadata !DIExpression()), !dbg !2109
  call void @llvm.dbg.declare(metadata [26 x %struct.dcomplex]* %csum_ref, metadata !2110, metadata !DIExpression()), !dbg !2114
  %0 = load i8*, i8** %class_npb.addr, align 8, !dbg !2115
  store i8 85, i8* %0, align 1, !dbg !2116
  store double 0x3D719799812DEA11, double* %epsilon, align 8, !dbg !2117
  %1 = load i32*, i32** %verified.addr, align 8, !dbg !2118
  store i32 0, i32* %1, align 4, !dbg !2119
  %2 = load i32, i32* %d1.addr, align 4, !dbg !2120
  %cmp = icmp eq i32 %2, 64, !dbg !2122
  br i1 %cmp, label %land.lhs.true, label %if.else, !dbg !2123

land.lhs.true:                                    ; preds = %entry
  %3 = load i32, i32* %d2.addr, align 4, !dbg !2124
  %cmp1 = icmp eq i32 %3, 64, !dbg !2125
  br i1 %cmp1, label %land.lhs.true2, label %if.else, !dbg !2126

land.lhs.true2:                                   ; preds = %land.lhs.true
  %4 = load i32, i32* %d3.addr, align 4, !dbg !2127
  %cmp3 = icmp eq i32 %4, 64, !dbg !2128
  br i1 %cmp3, label %land.lhs.true4, label %if.else, !dbg !2129

land.lhs.true4:                                   ; preds = %land.lhs.true2
  %5 = load i32, i32* %nt.addr, align 4, !dbg !2130
  %cmp5 = icmp eq i32 %5, 6, !dbg !2131
  br i1 %cmp5, label %if.then, label %if.else, !dbg !2132

if.then:                                          ; preds = %land.lhs.true4
  %6 = load i8*, i8** %class_npb.addr, align 8, !dbg !2133
  store i8 83, i8* %6, align 1, !dbg !2135
  %real = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp, i32 0, i32 0, !dbg !2136
  store double 0x408154DE9E5DA8C7, double* %real, align 8, !dbg !2136
  %imag = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp, i32 0, i32 1, !dbg !2136
  store double 0x407E4894D21E84F6, double* %imag, align 8, !dbg !2136
  %arrayidx = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 1, !dbg !2137
  %7 = bitcast %struct.dcomplex* %arrayidx to i8*, !dbg !2138
  %8 = bitcast %struct.dcomplex* %ref.tmp to i8*, !dbg !2138
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %7, i8* align 8 %8, i64 16, i1 false), !dbg !2138
  %real7 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp6, i32 0, i32 0, !dbg !2139
  store double 0x4081551BBB575EAB, double* %real7, align 8, !dbg !2139
  %imag8 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp6, i32 0, i32 1, !dbg !2139
  store double 0x407E687CA0F87E44, double* %imag8, align 8, !dbg !2139
  %arrayidx9 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 2, !dbg !2140
  %9 = bitcast %struct.dcomplex* %arrayidx9 to i8*, !dbg !2141
  %10 = bitcast %struct.dcomplex* %ref.tmp6 to i8*, !dbg !2141
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %9, i8* align 8 %10, i64 16, i1 false), !dbg !2141
  %real11 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp10, i32 0, i32 0, !dbg !2142
  store double 0x408154EB318EB593, double* %real11, align 8, !dbg !2142
  %imag12 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp10, i32 0, i32 1, !dbg !2142
  store double 0x407E8641D4F55AF9, double* %imag12, align 8, !dbg !2142
  %arrayidx13 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 3, !dbg !2143
  %11 = bitcast %struct.dcomplex* %arrayidx13 to i8*, !dbg !2144
  %12 = bitcast %struct.dcomplex* %ref.tmp10 to i8*, !dbg !2144
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %11, i8* align 8 %12, i64 16, i1 false), !dbg !2144
  %real15 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp14, i32 0, i32 0, !dbg !2145
  store double 0x40815456C13A7B04, double* %real15, align 8, !dbg !2145
  %imag16 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp14, i32 0, i32 1, !dbg !2145
  store double 0x407EA2097D7357C2, double* %imag16, align 8, !dbg !2145
  %arrayidx17 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 4, !dbg !2146
  %13 = bitcast %struct.dcomplex* %arrayidx17 to i8*, !dbg !2147
  %14 = bitcast %struct.dcomplex* %ref.tmp14 to i8*, !dbg !2147
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %13, i8* align 8 %14, i64 16, i1 false), !dbg !2147
  %real19 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp18, i32 0, i32 0, !dbg !2148
  store double 0x408153676E9F169C, double* %real19, align 8, !dbg !2148
  %imag20 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp18, i32 0, i32 1, !dbg !2148
  store double 0x407EBBF61C86EF29, double* %imag20, align 8, !dbg !2148
  %arrayidx21 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 5, !dbg !2149
  %15 = bitcast %struct.dcomplex* %arrayidx21 to i8*, !dbg !2150
  %16 = bitcast %struct.dcomplex* %ref.tmp18 to i8*, !dbg !2150
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %15, i8* align 8 %16, i64 16, i1 false), !dbg !2150
  %real23 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp22, i32 0, i32 0, !dbg !2151
  store double 0x408152259010E0A1, double* %real23, align 8, !dbg !2151
  %imag24 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp22, i32 0, i32 1, !dbg !2151
  store double 0x407ED427D4DF0213, double* %imag24, align 8, !dbg !2151
  %arrayidx25 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 6, !dbg !2152
  %17 = bitcast %struct.dcomplex* %arrayidx25 to i8*, !dbg !2153
  %18 = bitcast %struct.dcomplex* %ref.tmp22 to i8*, !dbg !2153
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %17, i8* align 8 %18, i64 16, i1 false), !dbg !2153
  br label %if.end492, !dbg !2154

if.else:                                          ; preds = %land.lhs.true4, %land.lhs.true2, %land.lhs.true, %entry
  %19 = load i32, i32* %d1.addr, align 4, !dbg !2155
  %cmp26 = icmp eq i32 %19, 128, !dbg !2157
  br i1 %cmp26, label %land.lhs.true27, label %if.else58, !dbg !2158

land.lhs.true27:                                  ; preds = %if.else
  %20 = load i32, i32* %d2.addr, align 4, !dbg !2159
  %cmp28 = icmp eq i32 %20, 128, !dbg !2160
  br i1 %cmp28, label %land.lhs.true29, label %if.else58, !dbg !2161

land.lhs.true29:                                  ; preds = %land.lhs.true27
  %21 = load i32, i32* %d3.addr, align 4, !dbg !2162
  %cmp30 = icmp eq i32 %21, 32, !dbg !2163
  br i1 %cmp30, label %land.lhs.true31, label %if.else58, !dbg !2164

land.lhs.true31:                                  ; preds = %land.lhs.true29
  %22 = load i32, i32* %nt.addr, align 4, !dbg !2165
  %cmp32 = icmp eq i32 %22, 6, !dbg !2166
  br i1 %cmp32, label %if.then33, label %if.else58, !dbg !2167

if.then33:                                        ; preds = %land.lhs.true31
  %23 = load i8*, i8** %class_npb.addr, align 8, !dbg !2168
  store i8 87, i8* %23, align 1, !dbg !2170
  %real35 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp34, i32 0, i32 0, !dbg !2171
  store double 0x4081BAE3C635196D, double* %real35, align 8, !dbg !2171
  %imag36 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp34, i32 0, i32 1, !dbg !2171
  store double 0x40808A98F467F156, double* %imag36, align 8, !dbg !2171
  %arrayidx37 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 1, !dbg !2172
  %24 = bitcast %struct.dcomplex* %arrayidx37 to i8*, !dbg !2173
  %25 = bitcast %struct.dcomplex* %ref.tmp34 to i8*, !dbg !2173
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %24, i8* align 8 %25, i64 16, i1 false), !dbg !2173
  %real39 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp38, i32 0, i32 0, !dbg !2174
  store double 0x40819926462BA5A4, double* %real39, align 8, !dbg !2174
  %imag40 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp38, i32 0, i32 1, !dbg !2174
  store double 0x408081B851380EB7, double* %imag40, align 8, !dbg !2174
  %arrayidx41 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 2, !dbg !2175
  %26 = bitcast %struct.dcomplex* %arrayidx41 to i8*, !dbg !2176
  %27 = bitcast %struct.dcomplex* %ref.tmp38 to i8*, !dbg !2176
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %26, i8* align 8 %27, i64 16, i1 false), !dbg !2176
  %real43 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp42, i32 0, i32 0, !dbg !2177
  store double 0x40817B3822354DD9, double* %real43, align 8, !dbg !2177
  %imag44 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp42, i32 0, i32 1, !dbg !2177
  store double 0x408078CC18578DFC, double* %imag44, align 8, !dbg !2177
  %arrayidx45 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 3, !dbg !2178
  %28 = bitcast %struct.dcomplex* %arrayidx45 to i8*, !dbg !2179
  %29 = bitcast %struct.dcomplex* %ref.tmp42 to i8*, !dbg !2179
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %28, i8* align 8 %29, i64 16, i1 false), !dbg !2179
  %real47 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp46, i32 0, i32 0, !dbg !2180
  store double 0x4081608EF5C48194, double* %real47, align 8, !dbg !2180
  %imag48 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp46, i32 0, i32 1, !dbg !2180
  store double 0x40807005B7059038, double* %imag48, align 8, !dbg !2180
  %arrayidx49 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 4, !dbg !2181
  %30 = bitcast %struct.dcomplex* %arrayidx49 to i8*, !dbg !2182
  %31 = bitcast %struct.dcomplex* %ref.tmp46 to i8*, !dbg !2182
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %30, i8* align 8 %31, i64 16, i1 false), !dbg !2182
  %real51 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp50, i32 0, i32 0, !dbg !2183
  store double 0x408148B81D084E83, double* %real51, align 8, !dbg !2183
  %imag52 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp50, i32 0, i32 1, !dbg !2183
  store double 0x408067854B0E36C9, double* %imag52, align 8, !dbg !2183
  %arrayidx53 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 5, !dbg !2184
  %32 = bitcast %struct.dcomplex* %arrayidx53 to i8*, !dbg !2185
  %33 = bitcast %struct.dcomplex* %ref.tmp50 to i8*, !dbg !2185
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %32, i8* align 8 %33, i64 16, i1 false), !dbg !2185
  %real55 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp54, i32 0, i32 0, !dbg !2186
  store double 0x40813353E9E3E09A, double* %real55, align 8, !dbg !2186
  %imag56 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp54, i32 0, i32 1, !dbg !2186
  store double 0x40805F5EAB0F5DA2, double* %imag56, align 8, !dbg !2186
  %arrayidx57 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 6, !dbg !2187
  %34 = bitcast %struct.dcomplex* %arrayidx57 to i8*, !dbg !2188
  %35 = bitcast %struct.dcomplex* %ref.tmp54 to i8*, !dbg !2188
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %34, i8* align 8 %35, i64 16, i1 false), !dbg !2188
  br label %if.end491, !dbg !2189

if.else58:                                        ; preds = %land.lhs.true31, %land.lhs.true29, %land.lhs.true27, %if.else
  %36 = load i32, i32* %d1.addr, align 4, !dbg !2190
  %cmp59 = icmp eq i32 %36, 256, !dbg !2192
  br i1 %cmp59, label %land.lhs.true60, label %if.else91, !dbg !2193

land.lhs.true60:                                  ; preds = %if.else58
  %37 = load i32, i32* %d2.addr, align 4, !dbg !2194
  %cmp61 = icmp eq i32 %37, 256, !dbg !2195
  br i1 %cmp61, label %land.lhs.true62, label %if.else91, !dbg !2196

land.lhs.true62:                                  ; preds = %land.lhs.true60
  %38 = load i32, i32* %d3.addr, align 4, !dbg !2197
  %cmp63 = icmp eq i32 %38, 128, !dbg !2198
  br i1 %cmp63, label %land.lhs.true64, label %if.else91, !dbg !2199

land.lhs.true64:                                  ; preds = %land.lhs.true62
  %39 = load i32, i32* %nt.addr, align 4, !dbg !2200
  %cmp65 = icmp eq i32 %39, 6, !dbg !2201
  br i1 %cmp65, label %if.then66, label %if.else91, !dbg !2202

if.then66:                                        ; preds = %land.lhs.true64
  %40 = load i8*, i8** %class_npb.addr, align 8, !dbg !2203
  store i8 65, i8* %40, align 1, !dbg !2205
  %real68 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp67, i32 0, i32 0, !dbg !2206
  store double 0x407F8AC6A8CB8B90, double* %real68, align 8, !dbg !2206
  %imag69 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp67, i32 0, i32 1, !dbg !2206
  store double 0x407FF67A05A82466, double* %imag69, align 8, !dbg !2206
  %arrayidx70 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 1, !dbg !2207
  %41 = bitcast %struct.dcomplex* %arrayidx70 to i8*, !dbg !2208
  %42 = bitcast %struct.dcomplex* %ref.tmp67 to i8*, !dbg !2208
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %41, i8* align 8 %42, i64 16, i1 false), !dbg !2208
  %real72 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp71, i32 0, i32 0, !dbg !2209
  store double 0x407F9F0F4941FB3E, double* %real72, align 8, !dbg !2209
  %imag73 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp71, i32 0, i32 1, !dbg !2209
  store double 0x407FDE18707A9D72, double* %imag73, align 8, !dbg !2209
  %arrayidx74 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 2, !dbg !2210
  %43 = bitcast %struct.dcomplex* %arrayidx74 to i8*, !dbg !2211
  %44 = bitcast %struct.dcomplex* %ref.tmp71 to i8*, !dbg !2211
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %43, i8* align 8 %44, i64 16, i1 false), !dbg !2211
  %real76 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp75, i32 0, i32 0, !dbg !2212
  store double 0x407FAF00C6D7110A, double* %real76, align 8, !dbg !2212
  %imag77 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp75, i32 0, i32 1, !dbg !2212
  store double 0x407FDD07CCB88353, double* %imag77, align 8, !dbg !2212
  %arrayidx78 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 3, !dbg !2213
  %45 = bitcast %struct.dcomplex* %arrayidx78 to i8*, !dbg !2214
  %46 = bitcast %struct.dcomplex* %ref.tmp75 to i8*, !dbg !2214
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %45, i8* align 8 %46, i64 16, i1 false), !dbg !2214
  %real80 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp79, i32 0, i32 0, !dbg !2215
  store double 0x407FBCA0EB3ECBEF, double* %real80, align 8, !dbg !2215
  %imag81 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp79, i32 0, i32 1, !dbg !2215
  store double 0x407FE2234776F4EF, double* %imag81, align 8, !dbg !2215
  %arrayidx82 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 4, !dbg !2216
  %47 = bitcast %struct.dcomplex* %arrayidx82 to i8*, !dbg !2217
  %48 = bitcast %struct.dcomplex* %ref.tmp79 to i8*, !dbg !2217
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %47, i8* align 8 %48, i64 16, i1 false), !dbg !2217
  %real84 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp83, i32 0, i32 0, !dbg !2218
  store double 0x407FC85F79D2C1E9, double* %real84, align 8, !dbg !2218
  %imag85 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp83, i32 0, i32 1, !dbg !2218
  store double 0x407FE7DD0AF2CEF4, double* %imag85, align 8, !dbg !2218
  %arrayidx86 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 5, !dbg !2219
  %49 = bitcast %struct.dcomplex* %arrayidx86 to i8*, !dbg !2220
  %50 = bitcast %struct.dcomplex* %ref.tmp83 to i8*, !dbg !2220
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %49, i8* align 8 %50, i64 16, i1 false), !dbg !2220
  %real88 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp87, i32 0, i32 0, !dbg !2221
  store double 0x407FD2611DBB8FA9, double* %real88, align 8, !dbg !2221
  %imag89 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp87, i32 0, i32 1, !dbg !2221
  store double 0x407FECAB25FE5602, double* %imag89, align 8, !dbg !2221
  %arrayidx90 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 6, !dbg !2222
  %51 = bitcast %struct.dcomplex* %arrayidx90 to i8*, !dbg !2223
  %52 = bitcast %struct.dcomplex* %ref.tmp87 to i8*, !dbg !2223
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %51, i8* align 8 %52, i64 16, i1 false), !dbg !2223
  br label %if.end490, !dbg !2224

if.else91:                                        ; preds = %land.lhs.true64, %land.lhs.true62, %land.lhs.true60, %if.else58
  %53 = load i32, i32* %d1.addr, align 4, !dbg !2225
  %cmp92 = icmp eq i32 %53, 512, !dbg !2227
  br i1 %cmp92, label %land.lhs.true93, label %if.else180, !dbg !2228

land.lhs.true93:                                  ; preds = %if.else91
  %54 = load i32, i32* %d2.addr, align 4, !dbg !2229
  %cmp94 = icmp eq i32 %54, 256, !dbg !2230
  br i1 %cmp94, label %land.lhs.true95, label %if.else180, !dbg !2231

land.lhs.true95:                                  ; preds = %land.lhs.true93
  %55 = load i32, i32* %d3.addr, align 4, !dbg !2232
  %cmp96 = icmp eq i32 %55, 256, !dbg !2233
  br i1 %cmp96, label %land.lhs.true97, label %if.else180, !dbg !2234

land.lhs.true97:                                  ; preds = %land.lhs.true95
  %56 = load i32, i32* %nt.addr, align 4, !dbg !2235
  %cmp98 = icmp eq i32 %56, 20, !dbg !2236
  br i1 %cmp98, label %if.then99, label %if.else180, !dbg !2237

if.then99:                                        ; preds = %land.lhs.true97
  %57 = load i8*, i8** %class_npb.addr, align 8, !dbg !2238
  store i8 66, i8* %57, align 1, !dbg !2240
  %real101 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp100, i32 0, i32 0, !dbg !2241
  store double 0x40802E1D67491D27, double* %real101, align 8, !dbg !2241
  %imag102 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp100, i32 0, i32 1, !dbg !2241
  store double 0x407FBC7C4BF0AFB0, double* %imag102, align 8, !dbg !2241
  %arrayidx103 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 1, !dbg !2242
  %58 = bitcast %struct.dcomplex* %arrayidx103 to i8*, !dbg !2243
  %59 = bitcast %struct.dcomplex* %ref.tmp100 to i8*, !dbg !2243
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %58, i8* align 8 %59, i64 16, i1 false), !dbg !2243
  %real105 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp104, i32 0, i32 0, !dbg !2244
  store double 0x40801B9DF5E01838, double* %real105, align 8, !dbg !2244
  %imag106 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp104, i32 0, i32 1, !dbg !2244
  store double 0x407FCD32F7994D45, double* %imag106, align 8, !dbg !2244
  %arrayidx107 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 2, !dbg !2245
  %60 = bitcast %struct.dcomplex* %arrayidx107 to i8*, !dbg !2246
  %61 = bitcast %struct.dcomplex* %ref.tmp104 to i8*, !dbg !2246
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %60, i8* align 8 %61, i64 16, i1 false), !dbg !2246
  %real109 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp108, i32 0, i32 0, !dbg !2247
  store double 0x408015209C2AC008, double* %real109, align 8, !dbg !2247
  %imag110 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp108, i32 0, i32 1, !dbg !2247
  store double 0x407FD9EF2BAE169A, double* %imag110, align 8, !dbg !2247
  %arrayidx111 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 3, !dbg !2248
  %62 = bitcast %struct.dcomplex* %arrayidx111 to i8*, !dbg !2249
  %63 = bitcast %struct.dcomplex* %ref.tmp108 to i8*, !dbg !2249
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %62, i8* align 8 %63, i64 16, i1 false), !dbg !2249
  %real113 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp112, i32 0, i32 0, !dbg !2250
  store double 0x408011E72B556FFE, double* %real113, align 8, !dbg !2250
  %imag114 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp112, i32 0, i32 1, !dbg !2250
  store double 0x407FE1A32DF83794, double* %imag114, align 8, !dbg !2250
  %arrayidx115 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 4, !dbg !2251
  %64 = bitcast %struct.dcomplex* %arrayidx115 to i8*, !dbg !2252
  %65 = bitcast %struct.dcomplex* %ref.tmp112 to i8*, !dbg !2252
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %64, i8* align 8 %65, i64 16, i1 false), !dbg !2252
  %real117 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp116, i32 0, i32 0, !dbg !2253
  store double 0x40800FB38AA32FE6, double* %real117, align 8, !dbg !2253
  %imag118 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp116, i32 0, i32 1, !dbg !2253
  store double 0x407FE65CD1D86E4E, double* %imag118, align 8, !dbg !2253
  %arrayidx119 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 5, !dbg !2254
  %66 = bitcast %struct.dcomplex* %arrayidx119 to i8*, !dbg !2255
  %67 = bitcast %struct.dcomplex* %ref.tmp116 to i8*, !dbg !2255
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %66, i8* align 8 %67, i64 16, i1 false), !dbg !2255
  %real121 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp120, i32 0, i32 0, !dbg !2256
  store double 0x40800DF0531A9C48, double* %real121, align 8, !dbg !2256
  %imag122 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp120, i32 0, i32 1, !dbg !2256
  store double 0x407FE9844F14C8E1, double* %imag122, align 8, !dbg !2256
  %arrayidx123 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 6, !dbg !2257
  %68 = bitcast %struct.dcomplex* %arrayidx123 to i8*, !dbg !2258
  %69 = bitcast %struct.dcomplex* %ref.tmp120 to i8*, !dbg !2258
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %68, i8* align 8 %69, i64 16, i1 false), !dbg !2258
  %real125 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp124, i32 0, i32 0, !dbg !2259
  store double 0x40800C700989200D, double* %real125, align 8, !dbg !2259
  %imag126 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp124, i32 0, i32 1, !dbg !2259
  store double 0x407FEBD8BF0DD370, double* %imag126, align 8, !dbg !2259
  %arrayidx127 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 7, !dbg !2260
  %70 = bitcast %struct.dcomplex* %arrayidx127 to i8*, !dbg !2261
  %71 = bitcast %struct.dcomplex* %ref.tmp124 to i8*, !dbg !2261
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %70, i8* align 8 %71, i64 16, i1 false), !dbg !2261
  %real129 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp128, i32 0, i32 0, !dbg !2262
  store double 0x40800B20F5210ADA, double* %real129, align 8, !dbg !2262
  %imag130 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp128, i32 0, i32 1, !dbg !2262
  store double 0x407FEDB8F6EE292B, double* %imag130, align 8, !dbg !2262
  %arrayidx131 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 8, !dbg !2263
  %72 = bitcast %struct.dcomplex* %arrayidx131 to i8*, !dbg !2264
  %73 = bitcast %struct.dcomplex* %ref.tmp128 to i8*, !dbg !2264
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %72, i8* align 8 %73, i64 16, i1 false), !dbg !2264
  %real133 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp132, i32 0, i32 0, !dbg !2265
  store double 0x408009FA001E667B, double* %real133, align 8, !dbg !2265
  %imag134 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp132, i32 0, i32 1, !dbg !2265
  store double 0x407FEF52DA70C18D, double* %imag134, align 8, !dbg !2265
  %arrayidx135 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 9, !dbg !2266
  %74 = bitcast %struct.dcomplex* %arrayidx135 to i8*, !dbg !2267
  %75 = bitcast %struct.dcomplex* %ref.tmp132 to i8*, !dbg !2267
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %74, i8* align 8 %75, i64 16, i1 false), !dbg !2267
  %real137 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp136, i32 0, i32 0, !dbg !2268
  store double 0x408008F54B8BB893, double* %real137, align 8, !dbg !2268
  %imag138 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp136, i32 0, i32 1, !dbg !2268
  store double 0x407FF0BC8A6C6119, double* %imag138, align 8, !dbg !2268
  %arrayidx139 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 10, !dbg !2269
  %76 = bitcast %struct.dcomplex* %arrayidx139 to i8*, !dbg !2270
  %77 = bitcast %struct.dcomplex* %ref.tmp136 to i8*, !dbg !2270
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %76, i8* align 8 %77, i64 16, i1 false), !dbg !2270
  %real141 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp140, i32 0, i32 0, !dbg !2271
  store double 0x4080080E66C1709C, double* %real141, align 8, !dbg !2271
  %imag142 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp140, i32 0, i32 1, !dbg !2271
  store double 0x407FF200FF33D23F, double* %imag142, align 8, !dbg !2271
  %arrayidx143 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 11, !dbg !2272
  %78 = bitcast %struct.dcomplex* %arrayidx143 to i8*, !dbg !2273
  %79 = bitcast %struct.dcomplex* %ref.tmp140 to i8*, !dbg !2273
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %78, i8* align 8 %79, i64 16, i1 false), !dbg !2273
  %real145 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp144, i32 0, i32 0, !dbg !2274
  store double 0x40800741A55F37AD, double* %real145, align 8, !dbg !2274
  %imag146 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp144, i32 0, i32 1, !dbg !2274
  store double 0x407FF3261FE7F7AD, double* %imag146, align 8, !dbg !2274
  %arrayidx147 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 12, !dbg !2275
  %80 = bitcast %struct.dcomplex* %arrayidx147 to i8*, !dbg !2276
  %81 = bitcast %struct.dcomplex* %ref.tmp144 to i8*, !dbg !2276
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %80, i8* align 8 %81, i64 16, i1 false), !dbg !2276
  %real149 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp148, i32 0, i32 0, !dbg !2277
  store double 0x4080068BDAC33674, double* %real149, align 8, !dbg !2277
  %imag150 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp148, i32 0, i32 1, !dbg !2277
  store double 0x407FF42F9BEB8DC0, double* %imag150, align 8, !dbg !2277
  %arrayidx151 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 13, !dbg !2278
  %82 = bitcast %struct.dcomplex* %arrayidx151 to i8*, !dbg !2279
  %83 = bitcast %struct.dcomplex* %ref.tmp148 to i8*, !dbg !2279
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %82, i8* align 8 %83, i64 16, i1 false), !dbg !2279
  %real153 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp152, i32 0, i32 0, !dbg !2280
  store double 0x408005EA3C919C43, double* %real153, align 8, !dbg !2280
  %imag154 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp152, i32 0, i32 1, !dbg !2280
  store double 0x407FF5203263B154, double* %imag154, align 8, !dbg !2280
  %arrayidx155 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 14, !dbg !2281
  %84 = bitcast %struct.dcomplex* %arrayidx155 to i8*, !dbg !2282
  %85 = bitcast %struct.dcomplex* %ref.tmp152 to i8*, !dbg !2282
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %84, i8* align 8 %85, i64 16, i1 false), !dbg !2282
  %real157 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp156, i32 0, i32 0, !dbg !2283
  store double 0x4080055A545A3920, double* %real157, align 8, !dbg !2283
  %imag158 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp156, i32 0, i32 1, !dbg !2283
  store double 0x407FF5FA3C741F6E, double* %imag158, align 8, !dbg !2283
  %arrayidx159 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 15, !dbg !2284
  %86 = bitcast %struct.dcomplex* %arrayidx159 to i8*, !dbg !2285
  %87 = bitcast %struct.dcomplex* %ref.tmp156 to i8*, !dbg !2285
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %86, i8* align 8 %87, i64 16, i1 false), !dbg !2285
  %real161 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp160, i32 0, i32 0, !dbg !2286
  store double 0x408004D9F6B6B8E1, double* %real161, align 8, !dbg !2286
  %imag162 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp160, i32 0, i32 1, !dbg !2286
  store double 0x407FF6BFE1A61501, double* %imag162, align 8, !dbg !2286
  %arrayidx163 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 16, !dbg !2287
  %88 = bitcast %struct.dcomplex* %arrayidx163 to i8*, !dbg !2288
  %89 = bitcast %struct.dcomplex* %ref.tmp160 to i8*, !dbg !2288
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %88, i8* align 8 %89, i64 16, i1 false), !dbg !2288
  %real165 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp164, i32 0, i32 0, !dbg !2289
  store double 0x408004673C213244, double* %real165, align 8, !dbg !2289
  %imag166 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp164, i32 0, i32 1, !dbg !2289
  store double 0x407FF77327A3F7B0, double* %imag166, align 8, !dbg !2289
  %arrayidx167 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 17, !dbg !2290
  %90 = bitcast %struct.dcomplex* %arrayidx167 to i8*, !dbg !2291
  %91 = bitcast %struct.dcomplex* %ref.tmp164 to i8*, !dbg !2291
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %90, i8* align 8 %91, i64 16, i1 false), !dbg !2291
  %real169 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp168, i32 0, i32 0, !dbg !2292
  store double 0x408004007A3FD0EA, double* %real169, align 8, !dbg !2292
  %imag170 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp168, i32 0, i32 1, !dbg !2292
  store double 0x407FF815F3F1C1DE, double* %imag170, align 8, !dbg !2292
  %arrayidx171 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 18, !dbg !2293
  %92 = bitcast %struct.dcomplex* %arrayidx171 to i8*, !dbg !2294
  %93 = bitcast %struct.dcomplex* %ref.tmp168 to i8*, !dbg !2294
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %92, i8* align 8 %93, i64 16, i1 false), !dbg !2294
  %real173 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp172, i32 0, i32 0, !dbg !2295
  store double 0x408003A43D5F793B, double* %real173, align 8, !dbg !2295
  %imag174 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp172, i32 0, i32 1, !dbg !2295
  store double 0x407FF8AA099402A0, double* %imag174, align 8, !dbg !2295
  %arrayidx175 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 19, !dbg !2296
  %94 = bitcast %struct.dcomplex* %arrayidx175 to i8*, !dbg !2297
  %95 = bitcast %struct.dcomplex* %ref.tmp172 to i8*, !dbg !2297
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %94, i8* align 8 %95, i64 16, i1 false), !dbg !2297
  %real177 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp176, i32 0, i32 0, !dbg !2298
  store double 0x40800351422D2EDF, double* %real177, align 8, !dbg !2298
  %imag178 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp176, i32 0, i32 1, !dbg !2298
  store double 0x407FF93106A352EE, double* %imag178, align 8, !dbg !2298
  %arrayidx179 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 20, !dbg !2299
  %96 = bitcast %struct.dcomplex* %arrayidx179 to i8*, !dbg !2300
  %97 = bitcast %struct.dcomplex* %ref.tmp176 to i8*, !dbg !2300
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %96, i8* align 8 %97, i64 16, i1 false), !dbg !2300
  br label %if.end489, !dbg !2301

if.else180:                                       ; preds = %land.lhs.true97, %land.lhs.true95, %land.lhs.true93, %if.else91
  %98 = load i32, i32* %d1.addr, align 4, !dbg !2302
  %cmp181 = icmp eq i32 %98, 512, !dbg !2304
  br i1 %cmp181, label %land.lhs.true182, label %if.else269, !dbg !2305

land.lhs.true182:                                 ; preds = %if.else180
  %99 = load i32, i32* %d2.addr, align 4, !dbg !2306
  %cmp183 = icmp eq i32 %99, 512, !dbg !2307
  br i1 %cmp183, label %land.lhs.true184, label %if.else269, !dbg !2308

land.lhs.true184:                                 ; preds = %land.lhs.true182
  %100 = load i32, i32* %d3.addr, align 4, !dbg !2309
  %cmp185 = icmp eq i32 %100, 512, !dbg !2310
  br i1 %cmp185, label %land.lhs.true186, label %if.else269, !dbg !2311

land.lhs.true186:                                 ; preds = %land.lhs.true184
  %101 = load i32, i32* %nt.addr, align 4, !dbg !2312
  %cmp187 = icmp eq i32 %101, 20, !dbg !2313
  br i1 %cmp187, label %if.then188, label %if.else269, !dbg !2314

if.then188:                                       ; preds = %land.lhs.true186
  %102 = load i8*, i8** %class_npb.addr, align 8, !dbg !2315
  store i8 67, i8* %102, align 1, !dbg !2317
  %real190 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp189, i32 0, i32 0, !dbg !2318
  store double 0x40803C101E899B03, double* %real190, align 8, !dbg !2318
  %imag191 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp189, i32 0, i32 1, !dbg !2318
  store double 0x408017373C01E593, double* %imag191, align 8, !dbg !2318
  %arrayidx192 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 1, !dbg !2319
  %103 = bitcast %struct.dcomplex* %arrayidx192 to i8*, !dbg !2320
  %104 = bitcast %struct.dcomplex* %ref.tmp189 to i8*, !dbg !2320
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %103, i8* align 8 %104, i64 16, i1 false), !dbg !2320
  %real194 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp193, i32 0, i32 0, !dbg !2321
  store double 0x40801C5675ED0B14, double* %real194, align 8, !dbg !2321
  %imag195 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp193, i32 0, i32 1, !dbg !2321
  store double 0x4080061004096FAD, double* %imag195, align 8, !dbg !2321
  %arrayidx196 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 2, !dbg !2322
  %105 = bitcast %struct.dcomplex* %arrayidx196 to i8*, !dbg !2323
  %106 = bitcast %struct.dcomplex* %ref.tmp193 to i8*, !dbg !2323
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %105, i8* align 8 %106, i64 16, i1 false), !dbg !2323
  %real198 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp197, i32 0, i32 0, !dbg !2324
  store double 0x408013BE0F176AC3, double* %real198, align 8, !dbg !2324
  %imag199 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp197, i32 0, i32 1, !dbg !2324
  store double 0x408001CD2DA9B691, double* %imag199, align 8, !dbg !2324
  %arrayidx200 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 3, !dbg !2325
  %107 = bitcast %struct.dcomplex* %arrayidx200 to i8*, !dbg !2326
  %108 = bitcast %struct.dcomplex* %ref.tmp197 to i8*, !dbg !2326
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %107, i8* align 8 %108, i64 16, i1 false), !dbg !2326
  %real202 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp201, i32 0, i32 0, !dbg !2327
  store double 0x4080101ED77ADAFA, double* %real202, align 8, !dbg !2327
  %imag203 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp201, i32 0, i32 1, !dbg !2327
  store double 0x408000DF4A8B7C66, double* %imag203, align 8, !dbg !2327
  %arrayidx204 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 4, !dbg !2328
  %109 = bitcast %struct.dcomplex* %arrayidx204 to i8*, !dbg !2329
  %110 = bitcast %struct.dcomplex* %ref.tmp201 to i8*, !dbg !2329
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %109, i8* align 8 %110, i64 16, i1 false), !dbg !2329
  %real206 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp205, i32 0, i32 0, !dbg !2330
  store double 0x40800E0A53D12FD5, double* %real206, align 8, !dbg !2330
  %imag207 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp205, i32 0, i32 1, !dbg !2330
  store double 0x408000EA3A1348C8, double* %imag207, align 8, !dbg !2330
  %arrayidx208 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 5, !dbg !2331
  %111 = bitcast %struct.dcomplex* %arrayidx208 to i8*, !dbg !2332
  %112 = bitcast %struct.dcomplex* %ref.tmp205 to i8*, !dbg !2332
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %111, i8* align 8 %112, i64 16, i1 false), !dbg !2332
  %real210 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp209, i32 0, i32 0, !dbg !2333
  store double 0x40800CA61ABB2192, double* %real210, align 8, !dbg !2333
  %imag211 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp209, i32 0, i32 1, !dbg !2333
  store double 0x408001328991F77F, double* %imag211, align 8, !dbg !2333
  %arrayidx212 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 6, !dbg !2334
  %113 = bitcast %struct.dcomplex* %arrayidx212 to i8*, !dbg !2335
  %114 = bitcast %struct.dcomplex* %ref.tmp209 to i8*, !dbg !2335
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %113, i8* align 8 %114, i64 16, i1 false), !dbg !2335
  %real214 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp213, i32 0, i32 0, !dbg !2336
  store double 0x40800BA7CD2DCE4D, double* %real214, align 8, !dbg !2336
  %imag215 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp213, i32 0, i32 1, !dbg !2336
  store double 0x4080017F2A30930B, double* %imag215, align 8, !dbg !2336
  %arrayidx216 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 7, !dbg !2337
  %115 = bitcast %struct.dcomplex* %arrayidx216 to i8*, !dbg !2338
  %116 = bitcast %struct.dcomplex* %ref.tmp213 to i8*, !dbg !2338
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %115, i8* align 8 %116, i64 16, i1 false), !dbg !2338
  %real218 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp217, i32 0, i32 0, !dbg !2339
  store double 0x40800AEBECB397D4, double* %real218, align 8, !dbg !2339
  %imag219 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp217, i32 0, i32 1, !dbg !2339
  store double 0x408001C12D7B83F2, double* %imag219, align 8, !dbg !2339
  %arrayidx220 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 8, !dbg !2340
  %117 = bitcast %struct.dcomplex* %arrayidx220 to i8*, !dbg !2341
  %118 = bitcast %struct.dcomplex* %ref.tmp217 to i8*, !dbg !2341
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %117, i8* align 8 %118, i64 16, i1 false), !dbg !2341
  %real222 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp221, i32 0, i32 0, !dbg !2342
  store double 0x40800A5D393668AE, double* %real222, align 8, !dbg !2342
  %imag223 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp221, i32 0, i32 1, !dbg !2342
  store double 0x408001F6BADA1C71, double* %imag223, align 8, !dbg !2342
  %arrayidx224 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 9, !dbg !2343
  %119 = bitcast %struct.dcomplex* %arrayidx224 to i8*, !dbg !2344
  %120 = bitcast %struct.dcomplex* %ref.tmp221 to i8*, !dbg !2344
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %119, i8* align 8 %120, i64 16, i1 false), !dbg !2344
  %real226 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp225, i32 0, i32 0, !dbg !2345
  store double 0x408009EDAA24021D, double* %real226, align 8, !dbg !2345
  %imag227 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp225, i32 0, i32 1, !dbg !2345
  store double 0x4080022183F3CA50, double* %imag227, align 8, !dbg !2345
  %arrayidx228 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 10, !dbg !2346
  %121 = bitcast %struct.dcomplex* %arrayidx228 to i8*, !dbg !2347
  %122 = bitcast %struct.dcomplex* %ref.tmp225 to i8*, !dbg !2347
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %121, i8* align 8 %122, i64 16, i1 false), !dbg !2347
  %real230 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp229, i32 0, i32 0, !dbg !2348
  store double 0x40800993B097C5AC, double* %real230, align 8, !dbg !2348
  %imag231 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp229, i32 0, i32 1, !dbg !2348
  store double 0x40800243C3A1DCB2, double* %imag231, align 8, !dbg !2348
  %arrayidx232 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 11, !dbg !2349
  %123 = bitcast %struct.dcomplex* %arrayidx232 to i8*, !dbg !2350
  %124 = bitcast %struct.dcomplex* %ref.tmp229 to i8*, !dbg !2350
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %123, i8* align 8 %124, i64 16, i1 false), !dbg !2350
  %real234 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp233, i32 0, i32 0, !dbg !2351
  store double 0x40800948BF026ADC, double* %real234, align 8, !dbg !2351
  %imag235 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp233, i32 0, i32 1, !dbg !2351
  store double 0x4080025F68FD8268, double* %imag235, align 8, !dbg !2351
  %arrayidx236 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 12, !dbg !2352
  %125 = bitcast %struct.dcomplex* %arrayidx236 to i8*, !dbg !2353
  %126 = bitcast %struct.dcomplex* %ref.tmp233 to i8*, !dbg !2353
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %125, i8* align 8 %126, i64 16, i1 false), !dbg !2353
  %real238 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp237, i32 0, i32 0, !dbg !2354
  store double 0x4080090857A518D9, double* %real238, align 8, !dbg !2354
  %imag239 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp237, i32 0, i32 1, !dbg !2354
  store double 0x40800275F32F50EA, double* %imag239, align 8, !dbg !2354
  %arrayidx240 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 13, !dbg !2355
  %127 = bitcast %struct.dcomplex* %arrayidx240 to i8*, !dbg !2356
  %128 = bitcast %struct.dcomplex* %ref.tmp237 to i8*, !dbg !2356
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %127, i8* align 8 %128, i64 16, i1 false), !dbg !2356
  %real242 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp241, i32 0, i32 0, !dbg !2357
  store double 0x408008CF67B5F6E6, double* %real242, align 8, !dbg !2357
  %imag243 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp241, i32 0, i32 1, !dbg !2357
  store double 0x408002887F1716B0, double* %imag243, align 8, !dbg !2357
  %arrayidx244 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 14, !dbg !2358
  %129 = bitcast %struct.dcomplex* %arrayidx244 to i8*, !dbg !2359
  %130 = bitcast %struct.dcomplex* %ref.tmp241 to i8*, !dbg !2359
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %129, i8* align 8 %130, i64 16, i1 false), !dbg !2359
  %real246 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp245, i32 0, i32 0, !dbg !2360
  store double 0x4080089BD580EA3A, double* %real246, align 8, !dbg !2360
  %imag247 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp245, i32 0, i32 1, !dbg !2360
  store double 0x40800297DE24048E, double* %imag247, align 8, !dbg !2360
  %arrayidx248 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 15, !dbg !2361
  %131 = bitcast %struct.dcomplex* %arrayidx248 to i8*, !dbg !2362
  %132 = bitcast %struct.dcomplex* %ref.tmp245 to i8*, !dbg !2362
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %131, i8* align 8 %132, i64 16, i1 false), !dbg !2362
  %real250 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp249, i32 0, i32 0, !dbg !2363
  store double 0x4080086C31EBD984, double* %real250, align 8, !dbg !2363
  %imag251 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp249, i32 0, i32 1, !dbg !2363
  store double 0x408002A4AAB9F9F8, double* %imag251, align 8, !dbg !2363
  %arrayidx252 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 16, !dbg !2364
  %133 = bitcast %struct.dcomplex* %arrayidx252 to i8*, !dbg !2365
  %134 = bitcast %struct.dcomplex* %ref.tmp249 to i8*, !dbg !2365
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %133, i8* align 8 %134, i64 16, i1 false), !dbg !2365
  %real254 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp253, i32 0, i32 0, !dbg !2366
  store double 0x4080083F8294129E, double* %real254, align 8, !dbg !2366
  %imag255 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp253, i32 0, i32 1, !dbg !2366
  store double 0x408002AF57DC0D71, double* %imag255, align 8, !dbg !2366
  %arrayidx256 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 17, !dbg !2367
  %135 = bitcast %struct.dcomplex* %arrayidx256 to i8*, !dbg !2368
  %136 = bitcast %struct.dcomplex* %ref.tmp253 to i8*, !dbg !2368
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %135, i8* align 8 %136, i64 16, i1 false), !dbg !2368
  %real258 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp257, i32 0, i32 0, !dbg !2369
  store double 0x408008151CE457D2, double* %real258, align 8, !dbg !2369
  %imag259 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp257, i32 0, i32 1, !dbg !2369
  store double 0x408002B83C8A44C9, double* %imag259, align 8, !dbg !2369
  %arrayidx260 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 18, !dbg !2370
  %137 = bitcast %struct.dcomplex* %arrayidx260 to i8*, !dbg !2371
  %138 = bitcast %struct.dcomplex* %ref.tmp257 to i8*, !dbg !2371
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %137, i8* align 8 %138, i64 16, i1 false), !dbg !2371
  %real262 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp261, i32 0, i32 0, !dbg !2372
  store double 0x408007EC8CCD48ED, double* %real262, align 8, !dbg !2372
  %imag263 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp261, i32 0, i32 1, !dbg !2372
  store double 0x408002BF9BCECA75, double* %imag263, align 8, !dbg !2372
  %arrayidx264 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 19, !dbg !2373
  %139 = bitcast %struct.dcomplex* %arrayidx264 to i8*, !dbg !2374
  %140 = bitcast %struct.dcomplex* %ref.tmp261 to i8*, !dbg !2374
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %139, i8* align 8 %140, i64 16, i1 false), !dbg !2374
  %real266 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp265, i32 0, i32 0, !dbg !2375
  store double 0x408007C58371022F, double* %real266, align 8, !dbg !2375
  %imag267 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp265, i32 0, i32 1, !dbg !2375
  store double 0x408002C5AA6407B6, double* %imag267, align 8, !dbg !2375
  %arrayidx268 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 20, !dbg !2376
  %141 = bitcast %struct.dcomplex* %arrayidx268 to i8*, !dbg !2377
  %142 = bitcast %struct.dcomplex* %ref.tmp265 to i8*, !dbg !2377
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %141, i8* align 8 %142, i64 16, i1 false), !dbg !2377
  br label %if.end488, !dbg !2378

if.else269:                                       ; preds = %land.lhs.true186, %land.lhs.true184, %land.lhs.true182, %if.else180
  %143 = load i32, i32* %d1.addr, align 4, !dbg !2379
  %cmp270 = icmp eq i32 %143, 2048, !dbg !2381
  br i1 %cmp270, label %land.lhs.true271, label %if.else378, !dbg !2382

land.lhs.true271:                                 ; preds = %if.else269
  %144 = load i32, i32* %d2.addr, align 4, !dbg !2383
  %cmp272 = icmp eq i32 %144, 1024, !dbg !2384
  br i1 %cmp272, label %land.lhs.true273, label %if.else378, !dbg !2385

land.lhs.true273:                                 ; preds = %land.lhs.true271
  %145 = load i32, i32* %d3.addr, align 4, !dbg !2386
  %cmp274 = icmp eq i32 %145, 1024, !dbg !2387
  br i1 %cmp274, label %land.lhs.true275, label %if.else378, !dbg !2388

land.lhs.true275:                                 ; preds = %land.lhs.true273
  %146 = load i32, i32* %nt.addr, align 4, !dbg !2389
  %cmp276 = icmp eq i32 %146, 25, !dbg !2390
  br i1 %cmp276, label %if.then277, label %if.else378, !dbg !2391

if.then277:                                       ; preds = %land.lhs.true275
  %147 = load i8*, i8** %class_npb.addr, align 8, !dbg !2392
  store i8 68, i8* %147, align 1, !dbg !2394
  %real279 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp278, i32 0, i32 0, !dbg !2395
  store double 0x408001C8B7A5243B, double* %real279, align 8, !dbg !2395
  %imag280 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp278, i32 0, i32 1, !dbg !2395
  store double 0x407FFDA78AA6499C, double* %imag280, align 8, !dbg !2395
  %arrayidx281 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 1, !dbg !2396
  %148 = bitcast %struct.dcomplex* %arrayidx281 to i8*, !dbg !2397
  %149 = bitcast %struct.dcomplex* %ref.tmp278 to i8*, !dbg !2397
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %148, i8* align 8 %149, i64 16, i1 false), !dbg !2397
  %real283 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp282, i32 0, i32 0, !dbg !2398
  store double 0x4080005F05B14D73, double* %real283, align 8, !dbg !2398
  %imag284 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp282, i32 0, i32 1, !dbg !2398
  store double 0x407FFB4C42805D51, double* %imag284, align 8, !dbg !2398
  %arrayidx285 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 2, !dbg !2399
  %150 = bitcast %struct.dcomplex* %arrayidx285 to i8*, !dbg !2400
  %151 = bitcast %struct.dcomplex* %ref.tmp282 to i8*, !dbg !2400
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %150, i8* align 8 %151, i64 16, i1 false), !dbg !2400
  %real287 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp286, i32 0, i32 0, !dbg !2401
  store double 0x407FFFC9049FE6AA, double* %real287, align 8, !dbg !2401
  %imag288 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp286, i32 0, i32 1, !dbg !2401
  store double 0x407FFB5AABC2C2DC, double* %imag288, align 8, !dbg !2401
  %arrayidx289 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 3, !dbg !2402
  %152 = bitcast %struct.dcomplex* %arrayidx289 to i8*, !dbg !2403
  %153 = bitcast %struct.dcomplex* %ref.tmp286 to i8*, !dbg !2403
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %152, i8* align 8 %153, i64 16, i1 false), !dbg !2403
  %real291 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp290, i32 0, i32 0, !dbg !2404
  store double 0x407FFF3AE6781D07, double* %real291, align 8, !dbg !2404
  %imag292 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp290, i32 0, i32 1, !dbg !2404
  store double 0x407FFBCC55AD30A5, double* %imag292, align 8, !dbg !2404
  %arrayidx293 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 4, !dbg !2405
  %154 = bitcast %struct.dcomplex* %arrayidx293 to i8*, !dbg !2406
  %155 = bitcast %struct.dcomplex* %ref.tmp290 to i8*, !dbg !2406
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %154, i8* align 8 %155, i64 16, i1 false), !dbg !2406
  %real295 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp294, i32 0, i32 0, !dbg !2407
  store double 0x407FFED49E586270, double* %real295, align 8, !dbg !2407
  %imag296 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp294, i32 0, i32 1, !dbg !2407
  store double 0x407FFC49DED1E229, double* %imag296, align 8, !dbg !2407
  %arrayidx297 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 5, !dbg !2408
  %156 = bitcast %struct.dcomplex* %arrayidx297 to i8*, !dbg !2409
  %157 = bitcast %struct.dcomplex* %ref.tmp294 to i8*, !dbg !2409
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %156, i8* align 8 %157, i64 16, i1 false), !dbg !2409
  %real299 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp298, i32 0, i32 0, !dbg !2410
  store double 0x407FFE88286F1600, double* %real299, align 8, !dbg !2410
  %imag300 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp298, i32 0, i32 1, !dbg !2410
  store double 0x407FFCBFA44E2DA9, double* %imag300, align 8, !dbg !2410
  %arrayidx301 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 6, !dbg !2411
  %158 = bitcast %struct.dcomplex* %arrayidx301 to i8*, !dbg !2412
  %159 = bitcast %struct.dcomplex* %ref.tmp298 to i8*, !dbg !2412
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %158, i8* align 8 %159, i64 16, i1 false), !dbg !2412
  %real303 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp302, i32 0, i32 0, !dbg !2413
  store double 0x407FFE4F62F012B7, double* %real303, align 8, !dbg !2413
  %imag304 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp302, i32 0, i32 1, !dbg !2413
  store double 0x407FFD2913502BF7, double* %imag304, align 8, !dbg !2413
  %arrayidx305 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 7, !dbg !2414
  %160 = bitcast %struct.dcomplex* %arrayidx305 to i8*, !dbg !2415
  %161 = bitcast %struct.dcomplex* %ref.tmp302 to i8*, !dbg !2415
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %160, i8* align 8 %161, i64 16, i1 false), !dbg !2415
  %real307 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp306, i32 0, i32 0, !dbg !2416
  store double 0x407FFE25D7467D87, double* %real307, align 8, !dbg !2416
  %imag308 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp306, i32 0, i32 1, !dbg !2416
  store double 0x407FFD85C991CC1E, double* %imag308, align 8, !dbg !2416
  %arrayidx309 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 8, !dbg !2417
  %162 = bitcast %struct.dcomplex* %arrayidx309 to i8*, !dbg !2418
  %163 = bitcast %struct.dcomplex* %ref.tmp306 to i8*, !dbg !2418
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %162, i8* align 8 %163, i64 16, i1 false), !dbg !2418
  %real311 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp310, i32 0, i32 0, !dbg !2419
  store double 0x407FFE07F5F9461B, double* %real311, align 8, !dbg !2419
  %imag312 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp310, i32 0, i32 1, !dbg !2419
  store double 0x407FFDD6ADE6AA2F, double* %imag312, align 8, !dbg !2419
  %arrayidx313 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 9, !dbg !2420
  %164 = bitcast %struct.dcomplex* %arrayidx313 to i8*, !dbg !2421
  %165 = bitcast %struct.dcomplex* %ref.tmp310 to i8*, !dbg !2421
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %164, i8* align 8 %165, i64 16, i1 false), !dbg !2421
  %real315 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp314, i32 0, i32 0, !dbg !2422
  store double 0x407FFDF2F9E3CE75, double* %real315, align 8, !dbg !2422
  %imag316 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp314, i32 0, i32 1, !dbg !2422
  store double 0x407FFE1D0052370F, double* %imag316, align 8, !dbg !2422
  %arrayidx317 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 10, !dbg !2423
  %166 = bitcast %struct.dcomplex* %arrayidx317 to i8*, !dbg !2424
  %167 = bitcast %struct.dcomplex* %ref.tmp314 to i8*, !dbg !2424
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %166, i8* align 8 %167, i64 16, i1 false), !dbg !2424
  %real319 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp318, i32 0, i32 0, !dbg !2425
  store double 0x407FFDE4CA360F49, double* %real319, align 8, !dbg !2425
  %imag320 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp318, i32 0, i32 1, !dbg !2425
  store double 0x407FFE5A05B5973E, double* %imag320, align 8, !dbg !2425
  %arrayidx321 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 11, !dbg !2426
  %168 = bitcast %struct.dcomplex* %arrayidx321 to i8*, !dbg !2427
  %169 = bitcast %struct.dcomplex* %ref.tmp318 to i8*, !dbg !2427
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %168, i8* align 8 %169, i64 16, i1 false), !dbg !2427
  %real323 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp322, i32 0, i32 0, !dbg !2428
  store double 0x407FFDDBD5F99711, double* %real323, align 8, !dbg !2428
  %imag324 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp322, i32 0, i32 1, !dbg !2428
  store double 0x407FFE8EEACAA874, double* %imag324, align 8, !dbg !2428
  %arrayidx325 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 12, !dbg !2429
  %170 = bitcast %struct.dcomplex* %arrayidx325 to i8*, !dbg !2430
  %171 = bitcast %struct.dcomplex* %ref.tmp322 to i8*, !dbg !2430
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %170, i8* align 8 %171, i64 16, i1 false), !dbg !2430
  %real327 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp326, i32 0, i32 0, !dbg !2431
  store double 0x407FFDD6F2033D21, double* %real327, align 8, !dbg !2431
  %imag328 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp326, i32 0, i32 1, !dbg !2431
  store double 0x407FFEBCBBFA2EBF, double* %imag328, align 8, !dbg !2431
  %arrayidx329 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 13, !dbg !2432
  %172 = bitcast %struct.dcomplex* %arrayidx329 to i8*, !dbg !2433
  %173 = bitcast %struct.dcomplex* %ref.tmp326 to i8*, !dbg !2433
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %172, i8* align 8 %173, i64 16, i1 false), !dbg !2433
  %real331 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp330, i32 0, i32 0, !dbg !2434
  store double 0x407FFDD53D74DC74, double* %real331, align 8, !dbg !2434
  %imag332 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp330, i32 0, i32 1, !dbg !2434
  store double 0x407FFEE46511649D, double* %imag332, align 8, !dbg !2434
  %arrayidx333 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 14, !dbg !2435
  %174 = bitcast %struct.dcomplex* %arrayidx333 to i8*, !dbg !2436
  %175 = bitcast %struct.dcomplex* %ref.tmp330 to i8*, !dbg !2436
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %174, i8* align 8 %175, i64 16, i1 false), !dbg !2436
  %real335 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp334, i32 0, i32 0, !dbg !2437
  store double 0x407FFDD60D2DB5D2, double* %real335, align 8, !dbg !2437
  %imag336 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp334, i32 0, i32 1, !dbg !2437
  store double 0x407FFF06B3C01AEA, double* %imag336, align 8, !dbg !2437
  %arrayidx337 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 15, !dbg !2438
  %176 = bitcast %struct.dcomplex* %arrayidx337 to i8*, !dbg !2439
  %177 = bitcast %struct.dcomplex* %ref.tmp334 to i8*, !dbg !2439
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %176, i8* align 8 %177, i64 16, i1 false), !dbg !2439
  %real339 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp338, i32 0, i32 0, !dbg !2440
  store double 0x407FFDD8DD056A7D, double* %real339, align 8, !dbg !2440
  %imag340 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp338, i32 0, i32 1, !dbg !2440
  store double 0x407FFF245ADF0BCE, double* %imag340, align 8, !dbg !2440
  %arrayidx341 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 16, !dbg !2441
  %178 = bitcast %struct.dcomplex* %arrayidx341 to i8*, !dbg !2442
  %179 = bitcast %struct.dcomplex* %ref.tmp338 to i8*, !dbg !2442
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %178, i8* align 8 %179, i64 16, i1 false), !dbg !2442
  %real343 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp342, i32 0, i32 0, !dbg !2443
  store double 0x407FFDDD45618FE6, double* %real343, align 8, !dbg !2443
  %imag344 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp342, i32 0, i32 1, !dbg !2443
  store double 0x407FFF3DF5BAB029, double* %imag344, align 8, !dbg !2443
  %arrayidx345 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 17, !dbg !2444
  %180 = bitcast %struct.dcomplex* %arrayidx345 to i8*, !dbg !2445
  %181 = bitcast %struct.dcomplex* %ref.tmp342 to i8*, !dbg !2445
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %180, i8* align 8 %181, i64 16, i1 false), !dbg !2445
  %real347 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp346, i32 0, i32 0, !dbg !2446
  store double 0x407FFDE2F3E650B3, double* %real347, align 8, !dbg !2446
  %imag348 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp346, i32 0, i32 1, !dbg !2446
  store double 0x407FFF540B1CF5A1, double* %imag348, align 8, !dbg !2446
  %arrayidx349 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 18, !dbg !2447
  %182 = bitcast %struct.dcomplex* %arrayidx349 to i8*, !dbg !2448
  %183 = bitcast %struct.dcomplex* %ref.tmp346 to i8*, !dbg !2448
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %182, i8* align 8 %183, i64 16, i1 false), !dbg !2448
  %real351 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp350, i32 0, i32 0, !dbg !2449
  store double 0x407FFDE9A64E1245, double* %real351, align 8, !dbg !2449
  %imag352 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp350, i32 0, i32 1, !dbg !2449
  store double 0x407FFF671002DAE5, double* %imag352, align 8, !dbg !2449
  %arrayidx353 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 19, !dbg !2450
  %184 = bitcast %struct.dcomplex* %arrayidx353 to i8*, !dbg !2451
  %185 = bitcast %struct.dcomplex* %ref.tmp350 to i8*, !dbg !2451
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %184, i8* align 8 %185, i64 16, i1 false), !dbg !2451
  %real355 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp354, i32 0, i32 0, !dbg !2452
  store double 0x407FFDF126BADF21, double* %real355, align 8, !dbg !2452
  %imag356 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp354, i32 0, i32 1, !dbg !2452
  store double 0x407FFF7769FD4D32, double* %imag356, align 8, !dbg !2452
  %arrayidx357 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 20, !dbg !2453
  %186 = bitcast %struct.dcomplex* %arrayidx357 to i8*, !dbg !2454
  %187 = bitcast %struct.dcomplex* %ref.tmp354 to i8*, !dbg !2454
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %186, i8* align 8 %187, i64 16, i1 false), !dbg !2454
  %real359 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp358, i32 0, i32 0, !dbg !2455
  store double 0x407FFDF94909BB13, double* %real359, align 8, !dbg !2455
  %imag360 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp358, i32 0, i32 1, !dbg !2455
  store double 0x407FFF85714411B2, double* %imag360, align 8, !dbg !2455
  %arrayidx361 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 21, !dbg !2456
  %188 = bitcast %struct.dcomplex* %arrayidx361 to i8*, !dbg !2457
  %189 = bitcast %struct.dcomplex* %ref.tmp358 to i8*, !dbg !2457
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %188, i8* align 8 %189, i64 16, i1 false), !dbg !2457
  %real363 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp362, i32 0, i32 0, !dbg !2458
  store double 0x407FFE01E8D7E962, double* %real363, align 8, !dbg !2458
  %imag364 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp362, i32 0, i32 1, !dbg !2458
  store double 0x407FFF9172826820, double* %imag364, align 8, !dbg !2458
  %arrayidx365 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 22, !dbg !2459
  %190 = bitcast %struct.dcomplex* %arrayidx365 to i8*, !dbg !2460
  %191 = bitcast %struct.dcomplex* %ref.tmp362 to i8*, !dbg !2460
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %190, i8* align 8 %191, i64 16, i1 false), !dbg !2460
  %real367 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp366, i32 0, i32 0, !dbg !2461
  store double 0x407FFE0AE8040E41, double* %real367, align 8, !dbg !2461
  %imag368 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp366, i32 0, i32 1, !dbg !2461
  store double 0x407FFF9BB06626E0, double* %imag368, align 8, !dbg !2461
  %arrayidx369 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 23, !dbg !2462
  %192 = bitcast %struct.dcomplex* %arrayidx369 to i8*, !dbg !2463
  %193 = bitcast %struct.dcomplex* %ref.tmp366 to i8*, !dbg !2463
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %192, i8* align 8 %193, i64 16, i1 false), !dbg !2463
  %real371 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp370, i32 0, i32 0, !dbg !2464
  store double 0x407FFE142D872C17, double* %real371, align 8, !dbg !2464
  %imag372 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp370, i32 0, i32 1, !dbg !2464
  store double 0x407FFFA464F89DCE, double* %imag372, align 8, !dbg !2464
  %arrayidx373 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 24, !dbg !2465
  %194 = bitcast %struct.dcomplex* %arrayidx373 to i8*, !dbg !2466
  %195 = bitcast %struct.dcomplex* %ref.tmp370 to i8*, !dbg !2466
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %194, i8* align 8 %195, i64 16, i1 false), !dbg !2466
  %real375 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp374, i32 0, i32 0, !dbg !2467
  store double 0x407FFE1DA48D386E, double* %real375, align 8, !dbg !2467
  %imag376 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp374, i32 0, i32 1, !dbg !2467
  store double 0x407FFFABC2C855DE, double* %imag376, align 8, !dbg !2467
  %arrayidx377 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 25, !dbg !2468
  %196 = bitcast %struct.dcomplex* %arrayidx377 to i8*, !dbg !2469
  %197 = bitcast %struct.dcomplex* %ref.tmp374 to i8*, !dbg !2469
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %196, i8* align 8 %197, i64 16, i1 false), !dbg !2469
  br label %if.end487, !dbg !2470

if.else378:                                       ; preds = %land.lhs.true275, %land.lhs.true273, %land.lhs.true271, %if.else269
  %198 = load i32, i32* %d1.addr, align 4, !dbg !2471
  %cmp379 = icmp eq i32 %198, 4096, !dbg !2473
  br i1 %cmp379, label %land.lhs.true380, label %if.end, !dbg !2474

land.lhs.true380:                                 ; preds = %if.else378
  %199 = load i32, i32* %d2.addr, align 4, !dbg !2475
  %cmp381 = icmp eq i32 %199, 2048, !dbg !2476
  br i1 %cmp381, label %land.lhs.true382, label %if.end, !dbg !2477

land.lhs.true382:                                 ; preds = %land.lhs.true380
  %200 = load i32, i32* %d3.addr, align 4, !dbg !2478
  %cmp383 = icmp eq i32 %200, 2048, !dbg !2479
  br i1 %cmp383, label %land.lhs.true384, label %if.end, !dbg !2480

land.lhs.true384:                                 ; preds = %land.lhs.true382
  %201 = load i32, i32* %nt.addr, align 4, !dbg !2481
  %cmp385 = icmp eq i32 %201, 25, !dbg !2482
  br i1 %cmp385, label %if.then386, label %if.end, !dbg !2483

if.then386:                                       ; preds = %land.lhs.true384
  %202 = load i8*, i8** %class_npb.addr, align 8, !dbg !2484
  store i8 69, i8* %202, align 1, !dbg !2486
  %real388 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp387, i32 0, i32 0, !dbg !2487
  store double 0x40800147E4E2E063, double* %real388, align 8, !dbg !2487
  %imag389 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp387, i32 0, i32 1, !dbg !2487
  store double 0x407FFBD566A0B5FD, double* %imag389, align 8, !dbg !2487
  %arrayidx390 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 1, !dbg !2488
  %203 = bitcast %struct.dcomplex* %arrayidx390 to i8*, !dbg !2489
  %204 = bitcast %struct.dcomplex* %ref.tmp387 to i8*, !dbg !2489
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %203, i8* align 8 %204, i64 16, i1 false), !dbg !2489
  %real392 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp391, i32 0, i32 0, !dbg !2490
  store double 0x408000B96D3A755A, double* %real392, align 8, !dbg !2490
  %imag393 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp391, i32 0, i32 1, !dbg !2490
  store double 0x407FFDC89676A99F, double* %imag393, align 8, !dbg !2490
  %arrayidx394 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 2, !dbg !2491
  %205 = bitcast %struct.dcomplex* %arrayidx394 to i8*, !dbg !2492
  %206 = bitcast %struct.dcomplex* %ref.tmp391 to i8*, !dbg !2492
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %205, i8* align 8 %206, i64 16, i1 false), !dbg !2492
  %real396 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp395, i32 0, i32 0, !dbg !2493
  store double 0x4080007FA32A25BE, double* %real396, align 8, !dbg !2493
  %imag397 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp395, i32 0, i32 1, !dbg !2493
  store double 0x407FFE84CB3A10F8, double* %imag397, align 8, !dbg !2493
  %arrayidx398 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 3, !dbg !2494
  %207 = bitcast %struct.dcomplex* %arrayidx398 to i8*, !dbg !2495
  %208 = bitcast %struct.dcomplex* %ref.tmp395 to i8*, !dbg !2495
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %207, i8* align 8 %208, i64 16, i1 false), !dbg !2495
  %real400 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp399, i32 0, i32 0, !dbg !2496
  store double 0x40800059C9C82B40, double* %real400, align 8, !dbg !2496
  %imag401 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp399, i32 0, i32 1, !dbg !2496
  store double 0x407FFEF414B87FD6, double* %imag401, align 8, !dbg !2496
  %arrayidx402 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 4, !dbg !2497
  %209 = bitcast %struct.dcomplex* %arrayidx402 to i8*, !dbg !2498
  %210 = bitcast %struct.dcomplex* %ref.tmp399 to i8*, !dbg !2498
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %209, i8* align 8 %210, i64 16, i1 false), !dbg !2498
  %real404 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp403, i32 0, i32 0, !dbg !2499
  store double 0x4080003FCCB7C9C8, double* %real404, align 8, !dbg !2499
  %imag405 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp403, i32 0, i32 1, !dbg !2499
  store double 0x407FFF483912F11E, double* %imag405, align 8, !dbg !2499
  %arrayidx406 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 5, !dbg !2500
  %211 = bitcast %struct.dcomplex* %arrayidx406 to i8*, !dbg !2501
  %212 = bitcast %struct.dcomplex* %ref.tmp403 to i8*, !dbg !2501
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %211, i8* align 8 %212, i64 16, i1 false), !dbg !2501
  %real408 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp407, i32 0, i32 0, !dbg !2502
  store double 0x4080002E4D90A084, double* %real408, align 8, !dbg !2502
  %imag409 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp407, i32 0, i32 1, !dbg !2502
  store double 0x407FFF8D62BCE558, double* %imag409, align 8, !dbg !2502
  %arrayidx410 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 6, !dbg !2503
  %213 = bitcast %struct.dcomplex* %arrayidx410 to i8*, !dbg !2504
  %214 = bitcast %struct.dcomplex* %ref.tmp407 to i8*, !dbg !2504
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %213, i8* align 8 %214, i64 16, i1 false), !dbg !2504
  %real412 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp411, i32 0, i32 0, !dbg !2505
  store double 0x40800022AC039D7C, double* %real412, align 8, !dbg !2505
  %imag413 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp411, i32 0, i32 1, !dbg !2505
  store double 0x407FFFC737C3F7CD, double* %imag413, align 8, !dbg !2505
  %arrayidx414 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 7, !dbg !2506
  %215 = bitcast %struct.dcomplex* %arrayidx414 to i8*, !dbg !2507
  %216 = bitcast %struct.dcomplex* %ref.tmp411 to i8*, !dbg !2507
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %215, i8* align 8 %216, i64 16, i1 false), !dbg !2507
  %real416 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp415, i32 0, i32 0, !dbg !2508
  store double 0x4080001ADFFA71B9, double* %real416, align 8, !dbg !2508
  %imag417 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp415, i32 0, i32 1, !dbg !2508
  store double 0x407FFFF78C336255, double* %imag417, align 8, !dbg !2508
  %arrayidx418 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 8, !dbg !2509
  %217 = bitcast %struct.dcomplex* %arrayidx418 to i8*, !dbg !2510
  %218 = bitcast %struct.dcomplex* %ref.tmp415 to i8*, !dbg !2510
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %217, i8* align 8 %218, i64 16, i1 false), !dbg !2510
  %real420 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp419, i32 0, i32 0, !dbg !2511
  store double 0x4080001574D0520C, double* %real420, align 8, !dbg !2511
  %imag421 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp419, i32 0, i32 1, !dbg !2511
  store double 0x4080000FE85C03E9, double* %imag421, align 8, !dbg !2511
  %arrayidx422 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 9, !dbg !2512
  %219 = bitcast %struct.dcomplex* %arrayidx422 to i8*, !dbg !2513
  %220 = bitcast %struct.dcomplex* %ref.tmp419 to i8*, !dbg !2513
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %219, i8* align 8 %220, i64 16, i1 false), !dbg !2513
  %real424 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp423, i32 0, i32 0, !dbg !2514
  store double 0x408000116F284244, double* %real424, align 8, !dbg !2514
  %imag425 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp423, i32 0, i32 1, !dbg !2514
  store double 0x40800020A7695837, double* %imag425, align 8, !dbg !2514
  %arrayidx426 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 10, !dbg !2515
  %221 = bitcast %struct.dcomplex* %arrayidx426 to i8*, !dbg !2516
  %222 = bitcast %struct.dcomplex* %ref.tmp423 to i8*, !dbg !2516
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %221, i8* align 8 %222, i64 16, i1 false), !dbg !2516
  %real428 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp427, i32 0, i32 0, !dbg !2517
  store double 0x4080000E2D56813F, double* %real428, align 8, !dbg !2517
  %imag429 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp427, i32 0, i32 1, !dbg !2517
  store double 0x4080002E951F7B34, double* %imag429, align 8, !dbg !2517
  %arrayidx430 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 11, !dbg !2518
  %223 = bitcast %struct.dcomplex* %arrayidx430 to i8*, !dbg !2519
  %224 = bitcast %struct.dcomplex* %ref.tmp427 to i8*, !dbg !2519
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %223, i8* align 8 %224, i64 16, i1 false), !dbg !2519
  %real432 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp431, i32 0, i32 0, !dbg !2520
  store double 0x4080000B4BE05864, double* %real432, align 8, !dbg !2520
  %imag433 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp431, i32 0, i32 1, !dbg !2520
  store double 0x4080003A2ED08404, double* %imag433, align 8, !dbg !2520
  %arrayidx434 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 12, !dbg !2521
  %225 = bitcast %struct.dcomplex* %arrayidx434 to i8*, !dbg !2522
  %226 = bitcast %struct.dcomplex* %ref.tmp431 to i8*, !dbg !2522
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %225, i8* align 8 %226, i64 16, i1 false), !dbg !2522
  %real436 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp435, i32 0, i32 0, !dbg !2523
  store double 0x408000089094AC2D, double* %real436, align 8, !dbg !2523
  %imag437 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp435, i32 0, i32 1, !dbg !2523
  store double 0x40800043DD87C2F3, double* %imag437, align 8, !dbg !2523
  %arrayidx438 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 13, !dbg !2524
  %227 = bitcast %struct.dcomplex* %arrayidx438 to i8*, !dbg !2525
  %228 = bitcast %struct.dcomplex* %ref.tmp435 to i8*, !dbg !2525
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %227, i8* align 8 %228, i64 16, i1 false), !dbg !2525
  %real440 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp439, i32 0, i32 0, !dbg !2526
  store double 0x40800005DBBF34DD, double* %real440, align 8, !dbg !2526
  %imag441 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp439, i32 0, i32 1, !dbg !2526
  store double 0x4080004BF7DEAC1A, double* %imag441, align 8, !dbg !2526
  %arrayidx442 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 14, !dbg !2527
  %229 = bitcast %struct.dcomplex* %arrayidx442 to i8*, !dbg !2528
  %230 = bitcast %struct.dcomplex* %ref.tmp439 to i8*, !dbg !2528
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %229, i8* align 8 %230, i64 16, i1 false), !dbg !2528
  %real444 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp443, i32 0, i32 0, !dbg !2529
  store double 0x408000031E1FCB83, double* %real444, align 8, !dbg !2529
  %imag445 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp443, i32 0, i32 1, !dbg !2529
  store double 0x40800052C48391C0, double* %imag445, align 8, !dbg !2529
  %arrayidx446 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 15, !dbg !2530
  %231 = bitcast %struct.dcomplex* %arrayidx446 to i8*, !dbg !2531
  %232 = bitcast %struct.dcomplex* %ref.tmp443 to i8*, !dbg !2531
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %231, i8* align 8 %232, i64 16, i1 false), !dbg !2531
  %real448 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp447, i32 0, i32 0, !dbg !2532
  store double 0x4080000052507A84, double* %real448, align 8, !dbg !2532
  %imag449 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp447, i32 0, i32 1, !dbg !2532
  store double 0x408000587CD9C3A1, double* %imag449, align 8, !dbg !2532
  %arrayidx450 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 16, !dbg !2533
  %233 = bitcast %struct.dcomplex* %arrayidx450 to i8*, !dbg !2534
  %234 = bitcast %struct.dcomplex* %ref.tmp447 to i8*, !dbg !2534
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %233, i8* align 8 %234, i64 16, i1 false), !dbg !2534
  %real452 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp451, i32 0, i32 0, !dbg !2535
  store double 0x407FFFFAF1111C29, double* %real452, align 8, !dbg !2535
  %imag453 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp451, i32 0, i32 1, !dbg !2535
  store double 0x4080005D4F648E97, double* %imag453, align 8, !dbg !2535
  %arrayidx454 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 17, !dbg !2536
  %235 = bitcast %struct.dcomplex* %arrayidx454 to i8*, !dbg !2537
  %236 = bitcast %struct.dcomplex* %ref.tmp451 to i8*, !dbg !2537
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %235, i8* align 8 %236, i64 16, i1 false), !dbg !2537
  %real456 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp455, i32 0, i32 0, !dbg !2538
  store double 0x407FFFF527E792B0, double* %real456, align 8, !dbg !2538
  %imag457 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp455, i32 0, i32 1, !dbg !2538
  store double 0x4080006161DD7A20, double* %imag457, align 8, !dbg !2538
  %arrayidx458 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 18, !dbg !2539
  %237 = bitcast %struct.dcomplex* %arrayidx458 to i8*, !dbg !2540
  %238 = bitcast %struct.dcomplex* %ref.tmp455 to i8*, !dbg !2540
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %237, i8* align 8 %238, i64 16, i1 false), !dbg !2540
  %real460 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp459, i32 0, i32 0, !dbg !2541
  store double 0x407FFFEF5224A658, double* %real460, align 8, !dbg !2541
  %imag461 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp459, i32 0, i32 1, !dbg !2541
  store double 0x40800064D2F0E0FB, double* %imag461, align 8, !dbg !2541
  %arrayidx462 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 19, !dbg !2542
  %239 = bitcast %struct.dcomplex* %arrayidx462 to i8*, !dbg !2543
  %240 = bitcast %struct.dcomplex* %ref.tmp459 to i8*, !dbg !2543
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %239, i8* align 8 %240, i64 16, i1 false), !dbg !2543
  %real464 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp463, i32 0, i32 0, !dbg !2544
  store double 0x407FFFE97985082F, double* %real464, align 8, !dbg !2544
  %imag465 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp463, i32 0, i32 1, !dbg !2544
  store double 0x40800067BBA76761, double* %imag465, align 8, !dbg !2544
  %arrayidx466 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 20, !dbg !2545
  %241 = bitcast %struct.dcomplex* %arrayidx466 to i8*, !dbg !2546
  %242 = bitcast %struct.dcomplex* %ref.tmp463 to i8*, !dbg !2546
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %241, i8* align 8 %242, i64 16, i1 false), !dbg !2546
  %real468 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp467, i32 0, i32 0, !dbg !2547
  store double 0x407FFFE3A76CE198, double* %real468, align 8, !dbg !2547
  %imag469 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp467, i32 0, i32 1, !dbg !2547
  store double 0x4080006A3087F53C, double* %imag469, align 8, !dbg !2547
  %arrayidx470 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 21, !dbg !2548
  %243 = bitcast %struct.dcomplex* %arrayidx470 to i8*, !dbg !2549
  %244 = bitcast %struct.dcomplex* %ref.tmp467 to i8*, !dbg !2549
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %243, i8* align 8 %244, i64 16, i1 false), !dbg !2549
  %real472 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp471, i32 0, i32 0, !dbg !2550
  store double 0x407FFFDDE458AC2A, double* %real472, align 8, !dbg !2550
  %imag473 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp471, i32 0, i32 1, !dbg !2550
  store double 0x4080006C427E60CB, double* %imag473, align 8, !dbg !2550
  %arrayidx474 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 22, !dbg !2551
  %245 = bitcast %struct.dcomplex* %arrayidx474 to i8*, !dbg !2552
  %246 = bitcast %struct.dcomplex* %ref.tmp471 to i8*, !dbg !2552
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %245, i8* align 8 %246, i64 16, i1 false), !dbg !2552
  %real476 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp475, i32 0, i32 0, !dbg !2553
  store double 0x407FFFD8379EC190, double* %real476, align 8, !dbg !2553
  %imag477 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp475, i32 0, i32 1, !dbg !2553
  store double 0x4080006DFF9235BC, double* %imag477, align 8, !dbg !2553
  %arrayidx478 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 23, !dbg !2554
  %247 = bitcast %struct.dcomplex* %arrayidx478 to i8*, !dbg !2555
  %248 = bitcast %struct.dcomplex* %ref.tmp475 to i8*, !dbg !2555
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %247, i8* align 8 %248, i64 16, i1 false), !dbg !2555
  %real480 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp479, i32 0, i32 0, !dbg !2556
  store double 0x407FFFD2A76113A7, double* %real480, align 8, !dbg !2556
  %imag481 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp479, i32 0, i32 1, !dbg !2556
  store double 0x4080006F7377203C, double* %imag481, align 8, !dbg !2556
  %arrayidx482 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 24, !dbg !2557
  %249 = bitcast %struct.dcomplex* %arrayidx482 to i8*, !dbg !2558
  %250 = bitcast %struct.dcomplex* %ref.tmp479 to i8*, !dbg !2558
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %249, i8* align 8 %250, i64 16, i1 false), !dbg !2558
  %real484 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp483, i32 0, i32 0, !dbg !2559
  store double 0x407FFFCD389947BC, double* %real484, align 8, !dbg !2559
  %imag485 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp483, i32 0, i32 1, !dbg !2559
  store double 0x40800070A7FF2BFD, double* %imag485, align 8, !dbg !2559
  %arrayidx486 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 25, !dbg !2560
  %251 = bitcast %struct.dcomplex* %arrayidx486 to i8*, !dbg !2561
  %252 = bitcast %struct.dcomplex* %ref.tmp483 to i8*, !dbg !2561
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %251, i8* align 8 %252, i64 16, i1 false), !dbg !2561
  br label %if.end, !dbg !2562

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
  %253 = load i8*, i8** %class_npb.addr, align 8, !dbg !2563
  %254 = load i8, i8* %253, align 1, !dbg !2565
  %conv = sext i8 %254 to i32, !dbg !2565
  %cmp493 = icmp ne i32 %conv, 85, !dbg !2566
  br i1 %cmp493, label %if.then494, label %if.end588, !dbg !2567

if.then494:                                       ; preds = %if.end492
  %255 = load i32*, i32** %verified.addr, align 8, !dbg !2568
  store i32 1, i32* %255, align 4, !dbg !2570
  store i32 1, i32* %i, align 4, !dbg !2571
  br label %for.cond, !dbg !2573

for.cond:                                         ; preds = %for.inc, %if.then494
  %256 = load i32, i32* %i, align 4, !dbg !2574
  %257 = load i32, i32* %nt.addr, align 4, !dbg !2576
  %cmp495 = icmp sle i32 %256, %257, !dbg !2577
  br i1 %cmp495, label %for.body, label %for.end, !dbg !2578

for.body:                                         ; preds = %for.cond
  %real496 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %agg.tmp, i32 0, i32 0, !dbg !2579
  %258 = load %struct.dcomplex*, %struct.dcomplex** @_ZL4sums, align 8, !dbg !2579
  %259 = load i32, i32* %i, align 4, !dbg !2579
  %idxprom = sext i32 %259 to i64, !dbg !2579
  %arrayidx497 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %258, i64 %idxprom, !dbg !2579
  %real498 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx497, i32 0, i32 0, !dbg !2579
  %260 = load double, double* %real498, align 8, !dbg !2579
  %261 = load i32, i32* %i, align 4, !dbg !2579
  %idxprom499 = sext i32 %261 to i64, !dbg !2579
  %arrayidx500 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 %idxprom499, !dbg !2579
  %real501 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx500, i32 0, i32 0, !dbg !2579
  %262 = load double, double* %real501, align 16, !dbg !2579
  %sub = fsub contract double %260, %262, !dbg !2579
  store double %sub, double* %real496, align 8, !dbg !2579
  %imag502 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %agg.tmp, i32 0, i32 1, !dbg !2579
  %263 = load %struct.dcomplex*, %struct.dcomplex** @_ZL4sums, align 8, !dbg !2579
  %264 = load i32, i32* %i, align 4, !dbg !2579
  %idxprom503 = sext i32 %264 to i64, !dbg !2579
  %arrayidx504 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %263, i64 %idxprom503, !dbg !2579
  %imag505 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx504, i32 0, i32 1, !dbg !2579
  %265 = load double, double* %imag505, align 8, !dbg !2579
  %266 = load i32, i32* %i, align 4, !dbg !2579
  %idxprom506 = sext i32 %266 to i64, !dbg !2579
  %arrayidx507 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 %idxprom506, !dbg !2579
  %imag508 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx507, i32 0, i32 1, !dbg !2579
  %267 = load double, double* %imag508, align 8, !dbg !2579
  %sub509 = fsub contract double %265, %267, !dbg !2579
  store double %sub509, double* %imag502, align 8, !dbg !2579
  %268 = load i32, i32* %i, align 4, !dbg !2579
  %idxprom511 = sext i32 %268 to i64, !dbg !2579
  %arrayidx512 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 %idxprom511, !dbg !2579
  %269 = bitcast %struct.dcomplex* %agg.tmp510 to i8*, !dbg !2579
  %270 = bitcast %struct.dcomplex* %arrayidx512 to i8*, !dbg !2579
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %269, i8* align 16 %270, i64 16, i1 false), !dbg !2579
  %271 = bitcast %struct.dcomplex* %agg.tmp to { double, double }*, !dbg !2579
  %272 = getelementptr inbounds { double, double }, { double, double }* %271, i32 0, i32 0, !dbg !2579
  %273 = load double, double* %272, align 8, !dbg !2579
  %274 = getelementptr inbounds { double, double }, { double, double }* %271, i32 0, i32 1, !dbg !2579
  %275 = load double, double* %274, align 8, !dbg !2579
  %276 = bitcast %struct.dcomplex* %agg.tmp510 to { double, double }*, !dbg !2579
  %277 = getelementptr inbounds { double, double }, { double, double }* %276, i32 0, i32 0, !dbg !2579
  %278 = load double, double* %277, align 8, !dbg !2579
  %279 = getelementptr inbounds { double, double }, { double, double }* %276, i32 0, i32 1, !dbg !2579
  %280 = load double, double* %279, align 8, !dbg !2579
  %call = call { double, double } @_ZL12dcomplex_div8dcomplexS_(double %273, double %275, double %278, double %280), !dbg !2579
  %281 = bitcast %struct.dcomplex* %coerce to { double, double }*, !dbg !2579
  %282 = getelementptr inbounds { double, double }, { double, double }* %281, i32 0, i32 0, !dbg !2579
  %283 = extractvalue { double, double } %call, 0, !dbg !2579
  store double %283, double* %282, align 8, !dbg !2579
  %284 = getelementptr inbounds { double, double }, { double, double }* %281, i32 0, i32 1, !dbg !2579
  %285 = extractvalue { double, double } %call, 1, !dbg !2579
  store double %285, double* %284, align 8, !dbg !2579
  %real513 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %coerce, i32 0, i32 0, !dbg !2579
  %286 = load double, double* %real513, align 8, !dbg !2579
  %real515 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %agg.tmp514, i32 0, i32 0, !dbg !2579
  %287 = load %struct.dcomplex*, %struct.dcomplex** @_ZL4sums, align 8, !dbg !2579
  %288 = load i32, i32* %i, align 4, !dbg !2579
  %idxprom516 = sext i32 %288 to i64, !dbg !2579
  %arrayidx517 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %287, i64 %idxprom516, !dbg !2579
  %real518 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx517, i32 0, i32 0, !dbg !2579
  %289 = load double, double* %real518, align 8, !dbg !2579
  %290 = load i32, i32* %i, align 4, !dbg !2579
  %idxprom519 = sext i32 %290 to i64, !dbg !2579
  %arrayidx520 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 %idxprom519, !dbg !2579
  %real521 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx520, i32 0, i32 0, !dbg !2579
  %291 = load double, double* %real521, align 16, !dbg !2579
  %sub522 = fsub contract double %289, %291, !dbg !2579
  store double %sub522, double* %real515, align 8, !dbg !2579
  %imag523 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %agg.tmp514, i32 0, i32 1, !dbg !2579
  %292 = load %struct.dcomplex*, %struct.dcomplex** @_ZL4sums, align 8, !dbg !2579
  %293 = load i32, i32* %i, align 4, !dbg !2579
  %idxprom524 = sext i32 %293 to i64, !dbg !2579
  %arrayidx525 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %292, i64 %idxprom524, !dbg !2579
  %imag526 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx525, i32 0, i32 1, !dbg !2579
  %294 = load double, double* %imag526, align 8, !dbg !2579
  %295 = load i32, i32* %i, align 4, !dbg !2579
  %idxprom527 = sext i32 %295 to i64, !dbg !2579
  %arrayidx528 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 %idxprom527, !dbg !2579
  %imag529 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx528, i32 0, i32 1, !dbg !2579
  %296 = load double, double* %imag529, align 8, !dbg !2579
  %sub530 = fsub contract double %294, %296, !dbg !2579
  store double %sub530, double* %imag523, align 8, !dbg !2579
  %297 = load i32, i32* %i, align 4, !dbg !2579
  %idxprom532 = sext i32 %297 to i64, !dbg !2579
  %arrayidx533 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 %idxprom532, !dbg !2579
  %298 = bitcast %struct.dcomplex* %agg.tmp531 to i8*, !dbg !2579
  %299 = bitcast %struct.dcomplex* %arrayidx533 to i8*, !dbg !2579
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %298, i8* align 16 %299, i64 16, i1 false), !dbg !2579
  %300 = bitcast %struct.dcomplex* %agg.tmp514 to { double, double }*, !dbg !2579
  %301 = getelementptr inbounds { double, double }, { double, double }* %300, i32 0, i32 0, !dbg !2579
  %302 = load double, double* %301, align 8, !dbg !2579
  %303 = getelementptr inbounds { double, double }, { double, double }* %300, i32 0, i32 1, !dbg !2579
  %304 = load double, double* %303, align 8, !dbg !2579
  %305 = bitcast %struct.dcomplex* %agg.tmp531 to { double, double }*, !dbg !2579
  %306 = getelementptr inbounds { double, double }, { double, double }* %305, i32 0, i32 0, !dbg !2579
  %307 = load double, double* %306, align 8, !dbg !2579
  %308 = getelementptr inbounds { double, double }, { double, double }* %305, i32 0, i32 1, !dbg !2579
  %309 = load double, double* %308, align 8, !dbg !2579
  %call534 = call { double, double } @_ZL12dcomplex_div8dcomplexS_(double %302, double %304, double %307, double %309), !dbg !2579
  %310 = bitcast %struct.dcomplex* %coerce535 to { double, double }*, !dbg !2579
  %311 = getelementptr inbounds { double, double }, { double, double }* %310, i32 0, i32 0, !dbg !2579
  %312 = extractvalue { double, double } %call534, 0, !dbg !2579
  store double %312, double* %311, align 8, !dbg !2579
  %313 = getelementptr inbounds { double, double }, { double, double }* %310, i32 0, i32 1, !dbg !2579
  %314 = extractvalue { double, double } %call534, 1, !dbg !2579
  store double %314, double* %313, align 8, !dbg !2579
  %real536 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %coerce535, i32 0, i32 0, !dbg !2579
  %315 = load double, double* %real536, align 8, !dbg !2579
  %mul = fmul contract double %286, %315, !dbg !2579
  %real538 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %agg.tmp537, i32 0, i32 0, !dbg !2579
  %316 = load %struct.dcomplex*, %struct.dcomplex** @_ZL4sums, align 8, !dbg !2579
  %317 = load i32, i32* %i, align 4, !dbg !2579
  %idxprom539 = sext i32 %317 to i64, !dbg !2579
  %arrayidx540 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %316, i64 %idxprom539, !dbg !2579
  %real541 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx540, i32 0, i32 0, !dbg !2579
  %318 = load double, double* %real541, align 8, !dbg !2579
  %319 = load i32, i32* %i, align 4, !dbg !2579
  %idxprom542 = sext i32 %319 to i64, !dbg !2579
  %arrayidx543 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 %idxprom542, !dbg !2579
  %real544 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx543, i32 0, i32 0, !dbg !2579
  %320 = load double, double* %real544, align 16, !dbg !2579
  %sub545 = fsub contract double %318, %320, !dbg !2579
  store double %sub545, double* %real538, align 8, !dbg !2579
  %imag546 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %agg.tmp537, i32 0, i32 1, !dbg !2579
  %321 = load %struct.dcomplex*, %struct.dcomplex** @_ZL4sums, align 8, !dbg !2579
  %322 = load i32, i32* %i, align 4, !dbg !2579
  %idxprom547 = sext i32 %322 to i64, !dbg !2579
  %arrayidx548 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %321, i64 %idxprom547, !dbg !2579
  %imag549 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx548, i32 0, i32 1, !dbg !2579
  %323 = load double, double* %imag549, align 8, !dbg !2579
  %324 = load i32, i32* %i, align 4, !dbg !2579
  %idxprom550 = sext i32 %324 to i64, !dbg !2579
  %arrayidx551 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 %idxprom550, !dbg !2579
  %imag552 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx551, i32 0, i32 1, !dbg !2579
  %325 = load double, double* %imag552, align 8, !dbg !2579
  %sub553 = fsub contract double %323, %325, !dbg !2579
  store double %sub553, double* %imag546, align 8, !dbg !2579
  %326 = load i32, i32* %i, align 4, !dbg !2579
  %idxprom555 = sext i32 %326 to i64, !dbg !2579
  %arrayidx556 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 %idxprom555, !dbg !2579
  %327 = bitcast %struct.dcomplex* %agg.tmp554 to i8*, !dbg !2579
  %328 = bitcast %struct.dcomplex* %arrayidx556 to i8*, !dbg !2579
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %327, i8* align 16 %328, i64 16, i1 false), !dbg !2579
  %329 = bitcast %struct.dcomplex* %agg.tmp537 to { double, double }*, !dbg !2579
  %330 = getelementptr inbounds { double, double }, { double, double }* %329, i32 0, i32 0, !dbg !2579
  %331 = load double, double* %330, align 8, !dbg !2579
  %332 = getelementptr inbounds { double, double }, { double, double }* %329, i32 0, i32 1, !dbg !2579
  %333 = load double, double* %332, align 8, !dbg !2579
  %334 = bitcast %struct.dcomplex* %agg.tmp554 to { double, double }*, !dbg !2579
  %335 = getelementptr inbounds { double, double }, { double, double }* %334, i32 0, i32 0, !dbg !2579
  %336 = load double, double* %335, align 8, !dbg !2579
  %337 = getelementptr inbounds { double, double }, { double, double }* %334, i32 0, i32 1, !dbg !2579
  %338 = load double, double* %337, align 8, !dbg !2579
  %call557 = call { double, double } @_ZL12dcomplex_div8dcomplexS_(double %331, double %333, double %336, double %338), !dbg !2579
  %339 = bitcast %struct.dcomplex* %coerce558 to { double, double }*, !dbg !2579
  %340 = getelementptr inbounds { double, double }, { double, double }* %339, i32 0, i32 0, !dbg !2579
  %341 = extractvalue { double, double } %call557, 0, !dbg !2579
  store double %341, double* %340, align 8, !dbg !2579
  %342 = getelementptr inbounds { double, double }, { double, double }* %339, i32 0, i32 1, !dbg !2579
  %343 = extractvalue { double, double } %call557, 1, !dbg !2579
  store double %343, double* %342, align 8, !dbg !2579
  %imag559 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %coerce558, i32 0, i32 1, !dbg !2579
  %344 = load double, double* %imag559, align 8, !dbg !2579
  %real561 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %agg.tmp560, i32 0, i32 0, !dbg !2579
  %345 = load %struct.dcomplex*, %struct.dcomplex** @_ZL4sums, align 8, !dbg !2579
  %346 = load i32, i32* %i, align 4, !dbg !2579
  %idxprom562 = sext i32 %346 to i64, !dbg !2579
  %arrayidx563 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %345, i64 %idxprom562, !dbg !2579
  %real564 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx563, i32 0, i32 0, !dbg !2579
  %347 = load double, double* %real564, align 8, !dbg !2579
  %348 = load i32, i32* %i, align 4, !dbg !2579
  %idxprom565 = sext i32 %348 to i64, !dbg !2579
  %arrayidx566 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 %idxprom565, !dbg !2579
  %real567 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx566, i32 0, i32 0, !dbg !2579
  %349 = load double, double* %real567, align 16, !dbg !2579
  %sub568 = fsub contract double %347, %349, !dbg !2579
  store double %sub568, double* %real561, align 8, !dbg !2579
  %imag569 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %agg.tmp560, i32 0, i32 1, !dbg !2579
  %350 = load %struct.dcomplex*, %struct.dcomplex** @_ZL4sums, align 8, !dbg !2579
  %351 = load i32, i32* %i, align 4, !dbg !2579
  %idxprom570 = sext i32 %351 to i64, !dbg !2579
  %arrayidx571 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %350, i64 %idxprom570, !dbg !2579
  %imag572 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx571, i32 0, i32 1, !dbg !2579
  %352 = load double, double* %imag572, align 8, !dbg !2579
  %353 = load i32, i32* %i, align 4, !dbg !2579
  %idxprom573 = sext i32 %353 to i64, !dbg !2579
  %arrayidx574 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 %idxprom573, !dbg !2579
  %imag575 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx574, i32 0, i32 1, !dbg !2579
  %354 = load double, double* %imag575, align 8, !dbg !2579
  %sub576 = fsub contract double %352, %354, !dbg !2579
  store double %sub576, double* %imag569, align 8, !dbg !2579
  %355 = load i32, i32* %i, align 4, !dbg !2579
  %idxprom578 = sext i32 %355 to i64, !dbg !2579
  %arrayidx579 = getelementptr inbounds [26 x %struct.dcomplex], [26 x %struct.dcomplex]* %csum_ref, i64 0, i64 %idxprom578, !dbg !2579
  %356 = bitcast %struct.dcomplex* %agg.tmp577 to i8*, !dbg !2579
  %357 = bitcast %struct.dcomplex* %arrayidx579 to i8*, !dbg !2579
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %356, i8* align 16 %357, i64 16, i1 false), !dbg !2579
  %358 = bitcast %struct.dcomplex* %agg.tmp560 to { double, double }*, !dbg !2579
  %359 = getelementptr inbounds { double, double }, { double, double }* %358, i32 0, i32 0, !dbg !2579
  %360 = load double, double* %359, align 8, !dbg !2579
  %361 = getelementptr inbounds { double, double }, { double, double }* %358, i32 0, i32 1, !dbg !2579
  %362 = load double, double* %361, align 8, !dbg !2579
  %363 = bitcast %struct.dcomplex* %agg.tmp577 to { double, double }*, !dbg !2579
  %364 = getelementptr inbounds { double, double }, { double, double }* %363, i32 0, i32 0, !dbg !2579
  %365 = load double, double* %364, align 8, !dbg !2579
  %366 = getelementptr inbounds { double, double }, { double, double }* %363, i32 0, i32 1, !dbg !2579
  %367 = load double, double* %366, align 8, !dbg !2579
  %call580 = call { double, double } @_ZL12dcomplex_div8dcomplexS_(double %360, double %362, double %365, double %367), !dbg !2579
  %368 = bitcast %struct.dcomplex* %coerce581 to { double, double }*, !dbg !2579
  %369 = getelementptr inbounds { double, double }, { double, double }* %368, i32 0, i32 0, !dbg !2579
  %370 = extractvalue { double, double } %call580, 0, !dbg !2579
  store double %370, double* %369, align 8, !dbg !2579
  %371 = getelementptr inbounds { double, double }, { double, double }* %368, i32 0, i32 1, !dbg !2579
  %372 = extractvalue { double, double } %call580, 1, !dbg !2579
  store double %372, double* %371, align 8, !dbg !2579
  %imag582 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %coerce581, i32 0, i32 1, !dbg !2579
  %373 = load double, double* %imag582, align 8, !dbg !2579
  %mul583 = fmul contract double %344, %373, !dbg !2579
  %add = fadd contract double %mul, %mul583, !dbg !2579
  %call584 = call double @sqrt(double %add) #7, !dbg !2579
  store double %call584, double* %err, align 8, !dbg !2581
  %374 = load double, double* %err, align 8, !dbg !2582
  %375 = load double, double* %epsilon, align 8, !dbg !2584
  %cmp585 = fcmp ole double %374, %375, !dbg !2585
  br i1 %cmp585, label %if.end587, label %if.then586, !dbg !2586

if.then586:                                       ; preds = %for.body
  %376 = load i32*, i32** %verified.addr, align 8, !dbg !2587
  store i32 0, i32* %376, align 4, !dbg !2589
  br label %for.end, !dbg !2590

if.end587:                                        ; preds = %for.body
  br label %for.inc, !dbg !2591

for.inc:                                          ; preds = %if.end587
  %377 = load i32, i32* %i, align 4, !dbg !2592
  %inc = add nsw i32 %377, 1, !dbg !2592
  store i32 %inc, i32* %i, align 4, !dbg !2592
  br label %for.cond, !dbg !2593, !llvm.loop !2594

for.end:                                          ; preds = %if.then586, %for.cond
  br label %if.end588, !dbg !2596

if.end588:                                        ; preds = %for.end, %if.end492
  %378 = load i8*, i8** %class_npb.addr, align 8, !dbg !2597
  %379 = load i8, i8* %378, align 1, !dbg !2599
  %conv589 = sext i8 %379 to i32, !dbg !2599
  %cmp590 = icmp ne i32 %conv589, 85, !dbg !2600
  br i1 %cmp590, label %if.then591, label %if.end597, !dbg !2601

if.then591:                                       ; preds = %if.end588
  %380 = load i32*, i32** %verified.addr, align 8, !dbg !2602
  %381 = load i32, i32* %380, align 4, !dbg !2605
  %tobool = icmp ne i32 %381, 0, !dbg !2605
  br i1 %tobool, label %if.then592, label %if.else594, !dbg !2606

if.then592:                                       ; preds = %if.then591
  %call593 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([33 x i8], [33 x i8]* @.str.73, i64 0, i64 0)), !dbg !2607
  br label %if.end596, !dbg !2609

if.else594:                                       ; preds = %if.then591
  %call595 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([29 x i8], [29 x i8]* @.str.74, i64 0, i64 0)), !dbg !2610
  br label %if.end596

if.end596:                                        ; preds = %if.else594, %if.then592
  br label %if.end597, !dbg !2612

if.end597:                                        ; preds = %if.end596, %if.end588
  %382 = load i8*, i8** %class_npb.addr, align 8, !dbg !2613
  %383 = load i8, i8* %382, align 1, !dbg !2614
  %conv598 = sext i8 %383 to i32, !dbg !2614
  %call599 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([17 x i8], [17 x i8]* @.str.75, i64 0, i64 0), i32 %conv598), !dbg !2615
  ret void, !dbg !2616
}

; Function Attrs: nounwind
declare dso_local double @log(double) #4

; Function Attrs: nounwind
declare dso_local i8* @strcpy(i8*, i8*) #4

; Function Attrs: nounwind
declare dso_local i8* @strcat(i8*, i8*) #4

; Function Attrs: noinline uwtable
define internal void @_ZL11release_gpuv() #2 !dbg !2617 {
entry:
  %0 = load %struct.dcomplex*, %struct.dcomplex** @sums_device, align 8, !dbg !2618
  %1 = bitcast %struct.dcomplex* %0 to i8*, !dbg !2618
  %call = call i32 @cudaFree(i8* %1), !dbg !2619
  %2 = load double*, double** @starts_device, align 8, !dbg !2620
  %3 = bitcast double* %2 to i8*, !dbg !2620
  %call1 = call i32 @cudaFree(i8* %3), !dbg !2621
  %4 = load double*, double** @twiddle_device, align 8, !dbg !2622
  %5 = bitcast double* %4 to i8*, !dbg !2622
  %call2 = call i32 @cudaFree(i8* %5), !dbg !2623
  %6 = load %struct.dcomplex*, %struct.dcomplex** @u_device, align 8, !dbg !2624
  %7 = bitcast %struct.dcomplex* %6 to i8*, !dbg !2624
  %call3 = call i32 @cudaFree(i8* %7), !dbg !2625
  %8 = load %struct.dcomplex*, %struct.dcomplex** @u0_device, align 8, !dbg !2626
  %9 = bitcast %struct.dcomplex* %8 to i8*, !dbg !2626
  %call4 = call i32 @cudaFree(i8* %9), !dbg !2627
  %10 = load %struct.dcomplex*, %struct.dcomplex** @u1_device, align 8, !dbg !2628
  %11 = bitcast %struct.dcomplex* %10 to i8*, !dbg !2628
  %call5 = call i32 @cudaFree(i8* %11), !dbg !2629
  %12 = load %struct.dcomplex*, %struct.dcomplex** @y0_device, align 8, !dbg !2630
  %13 = bitcast %struct.dcomplex* %12 to i8*, !dbg !2630
  %call6 = call i32 @cudaFree(i8* %13), !dbg !2631
  %14 = load %struct.dcomplex*, %struct.dcomplex** @y1_device, align 8, !dbg !2632
  %15 = bitcast %struct.dcomplex* %14 to i8*, !dbg !2632
  %call7 = call i32 @cudaFree(i8* %15), !dbg !2633
  ret void, !dbg !2634
}

; Function Attrs: nounwind
declare dso_local void @free(i8*) #4

; Function Attrs: noinline uwtable
define dso_local void @_Z19cffts1_gpu_kernel_1P8dcomplexS0_(%struct.dcomplex* %x_in, %struct.dcomplex* %y0) #2 !dbg !2635 {
entry:
  %x_in.addr = alloca %struct.dcomplex*, align 8
  %y0.addr = alloca %struct.dcomplex*, align 8
  store %struct.dcomplex* %x_in, %struct.dcomplex** %x_in.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %x_in.addr, metadata !2638, metadata !DIExpression()), !dbg !2639
  store %struct.dcomplex* %y0, %struct.dcomplex** %y0.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %y0.addr, metadata !2640, metadata !DIExpression()), !dbg !2641
  %0 = bitcast %struct.dcomplex** %x_in.addr to i8*, !dbg !2642
  %1 = call i32 @cudaSetupArgument(i8* %0, i64 8, i64 0), !dbg !2642
  %2 = icmp eq i32 %1, 0, !dbg !2642
  br i1 %2, label %setup.next, label %setup.end, !dbg !2642

setup.next:                                       ; preds = %entry
  %3 = bitcast %struct.dcomplex** %y0.addr to i8*, !dbg !2642
  %4 = call i32 @cudaSetupArgument(i8* %3, i64 8, i64 8), !dbg !2642
  %5 = icmp eq i32 %4, 0, !dbg !2642
  br i1 %5, label %setup.next1, label %setup.end, !dbg !2642

setup.next1:                                      ; preds = %setup.next
  %6 = call i32 @cudaLaunch(i8* bitcast (void (%struct.dcomplex*, %struct.dcomplex*)* @_Z19cffts1_gpu_kernel_1P8dcomplexS0_ to i8*)), !dbg !2642
  br label %setup.end, !dbg !2642

setup.end:                                        ; preds = %setup.next1, %setup.next, %entry
  ret void, !dbg !2643
}

declare dso_local i32 @cudaSetupArgument(i8*, i64, i64)

declare dso_local i32 @cudaLaunch(i8*)

; Function Attrs: noinline uwtable
define dso_local void @_Z19cffts1_gpu_kernel_2iP8dcomplexS0_S0_(i32 %is, %struct.dcomplex* %gty1, %struct.dcomplex* %gty2, %struct.dcomplex* %u_device) #2 !dbg !2644 {
entry:
  %is.addr = alloca i32, align 4
  %gty1.addr = alloca %struct.dcomplex*, align 8
  %gty2.addr = alloca %struct.dcomplex*, align 8
  %u_device.addr = alloca %struct.dcomplex*, align 8
  store i32 %is, i32* %is.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %is.addr, metadata !2648, metadata !DIExpression()), !dbg !2649
  store %struct.dcomplex* %gty1, %struct.dcomplex** %gty1.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %gty1.addr, metadata !2650, metadata !DIExpression()), !dbg !2651
  store %struct.dcomplex* %gty2, %struct.dcomplex** %gty2.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %gty2.addr, metadata !2652, metadata !DIExpression()), !dbg !2653
  store %struct.dcomplex* %u_device, %struct.dcomplex** %u_device.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %u_device.addr, metadata !2654, metadata !DIExpression()), !dbg !2655
  %0 = bitcast i32* %is.addr to i8*, !dbg !2656
  %1 = call i32 @cudaSetupArgument(i8* %0, i64 4, i64 0), !dbg !2656
  %2 = icmp eq i32 %1, 0, !dbg !2656
  br i1 %2, label %setup.next, label %setup.end, !dbg !2656

setup.next:                                       ; preds = %entry
  %3 = bitcast %struct.dcomplex** %gty1.addr to i8*, !dbg !2656
  %4 = call i32 @cudaSetupArgument(i8* %3, i64 8, i64 8), !dbg !2656
  %5 = icmp eq i32 %4, 0, !dbg !2656
  br i1 %5, label %setup.next1, label %setup.end, !dbg !2656

setup.next1:                                      ; preds = %setup.next
  %6 = bitcast %struct.dcomplex** %gty2.addr to i8*, !dbg !2656
  %7 = call i32 @cudaSetupArgument(i8* %6, i64 8, i64 16), !dbg !2656
  %8 = icmp eq i32 %7, 0, !dbg !2656
  br i1 %8, label %setup.next2, label %setup.end, !dbg !2656

setup.next2:                                      ; preds = %setup.next1
  %9 = bitcast %struct.dcomplex** %u_device.addr to i8*, !dbg !2656
  %10 = call i32 @cudaSetupArgument(i8* %9, i64 8, i64 24), !dbg !2656
  %11 = icmp eq i32 %10, 0, !dbg !2656
  br i1 %11, label %setup.next3, label %setup.end, !dbg !2656

setup.next3:                                      ; preds = %setup.next2
  %12 = call i32 @cudaLaunch(i8* bitcast (void (i32, %struct.dcomplex*, %struct.dcomplex*, %struct.dcomplex*)* @_Z19cffts1_gpu_kernel_2iP8dcomplexS0_S0_ to i8*)), !dbg !2656
  br label %setup.end, !dbg !2656

setup.end:                                        ; preds = %setup.next3, %setup.next2, %setup.next1, %setup.next, %entry
  ret void, !dbg !2657
}

; Function Attrs: noinline uwtable
define dso_local void @_Z19cffts1_gpu_kernel_3P8dcomplexS0_(%struct.dcomplex* %x_out, %struct.dcomplex* %y0) #2 !dbg !2658 {
entry:
  %x_out.addr = alloca %struct.dcomplex*, align 8
  %y0.addr = alloca %struct.dcomplex*, align 8
  store %struct.dcomplex* %x_out, %struct.dcomplex** %x_out.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %x_out.addr, metadata !2659, metadata !DIExpression()), !dbg !2660
  store %struct.dcomplex* %y0, %struct.dcomplex** %y0.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %y0.addr, metadata !2661, metadata !DIExpression()), !dbg !2662
  %0 = bitcast %struct.dcomplex** %x_out.addr to i8*, !dbg !2663
  %1 = call i32 @cudaSetupArgument(i8* %0, i64 8, i64 0), !dbg !2663
  %2 = icmp eq i32 %1, 0, !dbg !2663
  br i1 %2, label %setup.next, label %setup.end, !dbg !2663

setup.next:                                       ; preds = %entry
  %3 = bitcast %struct.dcomplex** %y0.addr to i8*, !dbg !2663
  %4 = call i32 @cudaSetupArgument(i8* %3, i64 8, i64 8), !dbg !2663
  %5 = icmp eq i32 %4, 0, !dbg !2663
  br i1 %5, label %setup.next1, label %setup.end, !dbg !2663

setup.next1:                                      ; preds = %setup.next
  %6 = call i32 @cudaLaunch(i8* bitcast (void (%struct.dcomplex*, %struct.dcomplex*)* @_Z19cffts1_gpu_kernel_3P8dcomplexS0_ to i8*)), !dbg !2663
  br label %setup.end, !dbg !2663

setup.end:                                        ; preds = %setup.next1, %setup.next, %entry
  ret void, !dbg !2664
}

; Function Attrs: noinline uwtable
define dso_local void @_Z19cffts2_gpu_kernel_1P8dcomplexS0_(%struct.dcomplex* %x_in, %struct.dcomplex* %y0) #2 !dbg !2665 {
entry:
  %x_in.addr = alloca %struct.dcomplex*, align 8
  %y0.addr = alloca %struct.dcomplex*, align 8
  store %struct.dcomplex* %x_in, %struct.dcomplex** %x_in.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %x_in.addr, metadata !2666, metadata !DIExpression()), !dbg !2667
  store %struct.dcomplex* %y0, %struct.dcomplex** %y0.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %y0.addr, metadata !2668, metadata !DIExpression()), !dbg !2669
  %0 = bitcast %struct.dcomplex** %x_in.addr to i8*, !dbg !2670
  %1 = call i32 @cudaSetupArgument(i8* %0, i64 8, i64 0), !dbg !2670
  %2 = icmp eq i32 %1, 0, !dbg !2670
  br i1 %2, label %setup.next, label %setup.end, !dbg !2670

setup.next:                                       ; preds = %entry
  %3 = bitcast %struct.dcomplex** %y0.addr to i8*, !dbg !2670
  %4 = call i32 @cudaSetupArgument(i8* %3, i64 8, i64 8), !dbg !2670
  %5 = icmp eq i32 %4, 0, !dbg !2670
  br i1 %5, label %setup.next1, label %setup.end, !dbg !2670

setup.next1:                                      ; preds = %setup.next
  %6 = call i32 @cudaLaunch(i8* bitcast (void (%struct.dcomplex*, %struct.dcomplex*)* @_Z19cffts2_gpu_kernel_1P8dcomplexS0_ to i8*)), !dbg !2670
  br label %setup.end, !dbg !2670

setup.end:                                        ; preds = %setup.next1, %setup.next, %entry
  ret void, !dbg !2671
}

; Function Attrs: noinline uwtable
define dso_local void @_Z19cffts2_gpu_kernel_2iP8dcomplexS0_S0_(i32 %is, %struct.dcomplex* %gty1, %struct.dcomplex* %gty2, %struct.dcomplex* %u_device) #2 !dbg !2672 {
entry:
  %is.addr = alloca i32, align 4
  %gty1.addr = alloca %struct.dcomplex*, align 8
  %gty2.addr = alloca %struct.dcomplex*, align 8
  %u_device.addr = alloca %struct.dcomplex*, align 8
  store i32 %is, i32* %is.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %is.addr, metadata !2673, metadata !DIExpression()), !dbg !2674
  store %struct.dcomplex* %gty1, %struct.dcomplex** %gty1.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %gty1.addr, metadata !2675, metadata !DIExpression()), !dbg !2676
  store %struct.dcomplex* %gty2, %struct.dcomplex** %gty2.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %gty2.addr, metadata !2677, metadata !DIExpression()), !dbg !2678
  store %struct.dcomplex* %u_device, %struct.dcomplex** %u_device.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %u_device.addr, metadata !2679, metadata !DIExpression()), !dbg !2680
  %0 = bitcast i32* %is.addr to i8*, !dbg !2681
  %1 = call i32 @cudaSetupArgument(i8* %0, i64 4, i64 0), !dbg !2681
  %2 = icmp eq i32 %1, 0, !dbg !2681
  br i1 %2, label %setup.next, label %setup.end, !dbg !2681

setup.next:                                       ; preds = %entry
  %3 = bitcast %struct.dcomplex** %gty1.addr to i8*, !dbg !2681
  %4 = call i32 @cudaSetupArgument(i8* %3, i64 8, i64 8), !dbg !2681
  %5 = icmp eq i32 %4, 0, !dbg !2681
  br i1 %5, label %setup.next1, label %setup.end, !dbg !2681

setup.next1:                                      ; preds = %setup.next
  %6 = bitcast %struct.dcomplex** %gty2.addr to i8*, !dbg !2681
  %7 = call i32 @cudaSetupArgument(i8* %6, i64 8, i64 16), !dbg !2681
  %8 = icmp eq i32 %7, 0, !dbg !2681
  br i1 %8, label %setup.next2, label %setup.end, !dbg !2681

setup.next2:                                      ; preds = %setup.next1
  %9 = bitcast %struct.dcomplex** %u_device.addr to i8*, !dbg !2681
  %10 = call i32 @cudaSetupArgument(i8* %9, i64 8, i64 24), !dbg !2681
  %11 = icmp eq i32 %10, 0, !dbg !2681
  br i1 %11, label %setup.next3, label %setup.end, !dbg !2681

setup.next3:                                      ; preds = %setup.next2
  %12 = call i32 @cudaLaunch(i8* bitcast (void (i32, %struct.dcomplex*, %struct.dcomplex*, %struct.dcomplex*)* @_Z19cffts2_gpu_kernel_2iP8dcomplexS0_S0_ to i8*)), !dbg !2681
  br label %setup.end, !dbg !2681

setup.end:                                        ; preds = %setup.next3, %setup.next2, %setup.next1, %setup.next, %entry
  ret void, !dbg !2682
}

; Function Attrs: noinline uwtable
define dso_local void @_Z19cffts2_gpu_kernel_3P8dcomplexS0_(%struct.dcomplex* %x_out, %struct.dcomplex* %y0) #2 !dbg !2683 {
entry:
  %x_out.addr = alloca %struct.dcomplex*, align 8
  %y0.addr = alloca %struct.dcomplex*, align 8
  store %struct.dcomplex* %x_out, %struct.dcomplex** %x_out.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %x_out.addr, metadata !2684, metadata !DIExpression()), !dbg !2685
  store %struct.dcomplex* %y0, %struct.dcomplex** %y0.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %y0.addr, metadata !2686, metadata !DIExpression()), !dbg !2687
  %0 = bitcast %struct.dcomplex** %x_out.addr to i8*, !dbg !2688
  %1 = call i32 @cudaSetupArgument(i8* %0, i64 8, i64 0), !dbg !2688
  %2 = icmp eq i32 %1, 0, !dbg !2688
  br i1 %2, label %setup.next, label %setup.end, !dbg !2688

setup.next:                                       ; preds = %entry
  %3 = bitcast %struct.dcomplex** %y0.addr to i8*, !dbg !2688
  %4 = call i32 @cudaSetupArgument(i8* %3, i64 8, i64 8), !dbg !2688
  %5 = icmp eq i32 %4, 0, !dbg !2688
  br i1 %5, label %setup.next1, label %setup.end, !dbg !2688

setup.next1:                                      ; preds = %setup.next
  %6 = call i32 @cudaLaunch(i8* bitcast (void (%struct.dcomplex*, %struct.dcomplex*)* @_Z19cffts2_gpu_kernel_3P8dcomplexS0_ to i8*)), !dbg !2688
  br label %setup.end, !dbg !2688

setup.end:                                        ; preds = %setup.next1, %setup.next, %entry
  ret void, !dbg !2689
}

; Function Attrs: noinline uwtable
define dso_local void @_Z19cffts3_gpu_kernel_1P8dcomplexS0_(%struct.dcomplex* %x_in, %struct.dcomplex* %y0) #2 !dbg !2690 {
entry:
  %x_in.addr = alloca %struct.dcomplex*, align 8
  %y0.addr = alloca %struct.dcomplex*, align 8
  store %struct.dcomplex* %x_in, %struct.dcomplex** %x_in.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %x_in.addr, metadata !2691, metadata !DIExpression()), !dbg !2692
  store %struct.dcomplex* %y0, %struct.dcomplex** %y0.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %y0.addr, metadata !2693, metadata !DIExpression()), !dbg !2694
  %0 = bitcast %struct.dcomplex** %x_in.addr to i8*, !dbg !2695
  %1 = call i32 @cudaSetupArgument(i8* %0, i64 8, i64 0), !dbg !2695
  %2 = icmp eq i32 %1, 0, !dbg !2695
  br i1 %2, label %setup.next, label %setup.end, !dbg !2695

setup.next:                                       ; preds = %entry
  %3 = bitcast %struct.dcomplex** %y0.addr to i8*, !dbg !2695
  %4 = call i32 @cudaSetupArgument(i8* %3, i64 8, i64 8), !dbg !2695
  %5 = icmp eq i32 %4, 0, !dbg !2695
  br i1 %5, label %setup.next1, label %setup.end, !dbg !2695

setup.next1:                                      ; preds = %setup.next
  %6 = call i32 @cudaLaunch(i8* bitcast (void (%struct.dcomplex*, %struct.dcomplex*)* @_Z19cffts3_gpu_kernel_1P8dcomplexS0_ to i8*)), !dbg !2695
  br label %setup.end, !dbg !2695

setup.end:                                        ; preds = %setup.next1, %setup.next, %entry
  ret void, !dbg !2696
}

; Function Attrs: noinline uwtable
define dso_local void @_Z19cffts3_gpu_kernel_2iP8dcomplexS0_S0_(i32 %is, %struct.dcomplex* %gty1, %struct.dcomplex* %gty2, %struct.dcomplex* %u_device) #2 !dbg !2697 {
entry:
  %is.addr = alloca i32, align 4
  %gty1.addr = alloca %struct.dcomplex*, align 8
  %gty2.addr = alloca %struct.dcomplex*, align 8
  %u_device.addr = alloca %struct.dcomplex*, align 8
  store i32 %is, i32* %is.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %is.addr, metadata !2698, metadata !DIExpression()), !dbg !2699
  store %struct.dcomplex* %gty1, %struct.dcomplex** %gty1.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %gty1.addr, metadata !2700, metadata !DIExpression()), !dbg !2701
  store %struct.dcomplex* %gty2, %struct.dcomplex** %gty2.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %gty2.addr, metadata !2702, metadata !DIExpression()), !dbg !2703
  store %struct.dcomplex* %u_device, %struct.dcomplex** %u_device.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %u_device.addr, metadata !2704, metadata !DIExpression()), !dbg !2705
  %0 = bitcast i32* %is.addr to i8*, !dbg !2706
  %1 = call i32 @cudaSetupArgument(i8* %0, i64 4, i64 0), !dbg !2706
  %2 = icmp eq i32 %1, 0, !dbg !2706
  br i1 %2, label %setup.next, label %setup.end, !dbg !2706

setup.next:                                       ; preds = %entry
  %3 = bitcast %struct.dcomplex** %gty1.addr to i8*, !dbg !2706
  %4 = call i32 @cudaSetupArgument(i8* %3, i64 8, i64 8), !dbg !2706
  %5 = icmp eq i32 %4, 0, !dbg !2706
  br i1 %5, label %setup.next1, label %setup.end, !dbg !2706

setup.next1:                                      ; preds = %setup.next
  %6 = bitcast %struct.dcomplex** %gty2.addr to i8*, !dbg !2706
  %7 = call i32 @cudaSetupArgument(i8* %6, i64 8, i64 16), !dbg !2706
  %8 = icmp eq i32 %7, 0, !dbg !2706
  br i1 %8, label %setup.next2, label %setup.end, !dbg !2706

setup.next2:                                      ; preds = %setup.next1
  %9 = bitcast %struct.dcomplex** %u_device.addr to i8*, !dbg !2706
  %10 = call i32 @cudaSetupArgument(i8* %9, i64 8, i64 24), !dbg !2706
  %11 = icmp eq i32 %10, 0, !dbg !2706
  br i1 %11, label %setup.next3, label %setup.end, !dbg !2706

setup.next3:                                      ; preds = %setup.next2
  %12 = call i32 @cudaLaunch(i8* bitcast (void (i32, %struct.dcomplex*, %struct.dcomplex*, %struct.dcomplex*)* @_Z19cffts3_gpu_kernel_2iP8dcomplexS0_S0_ to i8*)), !dbg !2706
  br label %setup.end, !dbg !2706

setup.end:                                        ; preds = %setup.next3, %setup.next2, %setup.next1, %setup.next, %entry
  ret void, !dbg !2707
}

; Function Attrs: noinline uwtable
define dso_local void @_Z19cffts3_gpu_kernel_3P8dcomplexS0_(%struct.dcomplex* %x_out, %struct.dcomplex* %y0) #2 !dbg !2708 {
entry:
  %x_out.addr = alloca %struct.dcomplex*, align 8
  %y0.addr = alloca %struct.dcomplex*, align 8
  store %struct.dcomplex* %x_out, %struct.dcomplex** %x_out.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %x_out.addr, metadata !2709, metadata !DIExpression()), !dbg !2710
  store %struct.dcomplex* %y0, %struct.dcomplex** %y0.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %y0.addr, metadata !2711, metadata !DIExpression()), !dbg !2712
  %0 = bitcast %struct.dcomplex** %x_out.addr to i8*, !dbg !2713
  %1 = call i32 @cudaSetupArgument(i8* %0, i64 8, i64 0), !dbg !2713
  %2 = icmp eq i32 %1, 0, !dbg !2713
  br i1 %2, label %setup.next, label %setup.end, !dbg !2713

setup.next:                                       ; preds = %entry
  %3 = bitcast %struct.dcomplex** %y0.addr to i8*, !dbg !2713
  %4 = call i32 @cudaSetupArgument(i8* %3, i64 8, i64 8), !dbg !2713
  %5 = icmp eq i32 %4, 0, !dbg !2713
  br i1 %5, label %setup.next1, label %setup.end, !dbg !2713

setup.next1:                                      ; preds = %setup.next
  %6 = call i32 @cudaLaunch(i8* bitcast (void (%struct.dcomplex*, %struct.dcomplex*)* @_Z19cffts3_gpu_kernel_3P8dcomplexS0_ to i8*)), !dbg !2713
  br label %setup.end, !dbg !2713

setup.end:                                        ; preds = %setup.next1, %setup.next, %entry
  ret void, !dbg !2714
}

; Function Attrs: noinline uwtable
define dso_local void @_Z19checksum_gpu_kerneliP8dcomplexS0_(i32 %iteration, %struct.dcomplex* %u1, %struct.dcomplex* %sums) #2 !dbg !2715 {
entry:
  %iteration.addr = alloca i32, align 4
  %u1.addr = alloca %struct.dcomplex*, align 8
  %sums.addr = alloca %struct.dcomplex*, align 8
  store i32 %iteration, i32* %iteration.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %iteration.addr, metadata !2716, metadata !DIExpression()), !dbg !2717
  store %struct.dcomplex* %u1, %struct.dcomplex** %u1.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %u1.addr, metadata !2718, metadata !DIExpression()), !dbg !2719
  store %struct.dcomplex* %sums, %struct.dcomplex** %sums.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %sums.addr, metadata !2720, metadata !DIExpression()), !dbg !2721
  %0 = bitcast i32* %iteration.addr to i8*, !dbg !2722
  %1 = call i32 @cudaSetupArgument(i8* %0, i64 4, i64 0), !dbg !2722
  %2 = icmp eq i32 %1, 0, !dbg !2722
  br i1 %2, label %setup.next, label %setup.end, !dbg !2722

setup.next:                                       ; preds = %entry
  %3 = bitcast %struct.dcomplex** %u1.addr to i8*, !dbg !2722
  %4 = call i32 @cudaSetupArgument(i8* %3, i64 8, i64 8), !dbg !2722
  %5 = icmp eq i32 %4, 0, !dbg !2722
  br i1 %5, label %setup.next1, label %setup.end, !dbg !2722

setup.next1:                                      ; preds = %setup.next
  %6 = bitcast %struct.dcomplex** %sums.addr to i8*, !dbg !2722
  %7 = call i32 @cudaSetupArgument(i8* %6, i64 8, i64 16), !dbg !2722
  %8 = icmp eq i32 %7, 0, !dbg !2722
  br i1 %8, label %setup.next2, label %setup.end, !dbg !2722

setup.next2:                                      ; preds = %setup.next1
  %9 = call i32 @cudaLaunch(i8* bitcast (void (i32, %struct.dcomplex*, %struct.dcomplex*)* @_Z19checksum_gpu_kerneliP8dcomplexS0_ to i8*)), !dbg !2722
  br label %setup.end, !dbg !2722

setup.end:                                        ; preds = %setup.next2, %setup.next1, %setup.next, %entry
  ret void, !dbg !2723
}

; Function Attrs: noinline uwtable
define dso_local void @_Z27compute_indexmap_gpu_kernelPd(double* %twiddle) #2 !dbg !2724 {
entry:
  %twiddle.addr = alloca double*, align 8
  store double* %twiddle, double** %twiddle.addr, align 8
  call void @llvm.dbg.declare(metadata double** %twiddle.addr, metadata !2725, metadata !DIExpression()), !dbg !2726
  %0 = bitcast double** %twiddle.addr to i8*, !dbg !2727
  %1 = call i32 @cudaSetupArgument(i8* %0, i64 8, i64 0), !dbg !2727
  %2 = icmp eq i32 %1, 0, !dbg !2727
  br i1 %2, label %setup.next, label %setup.end, !dbg !2727

setup.next:                                       ; preds = %entry
  %3 = call i32 @cudaLaunch(i8* bitcast (void (double*)* @_Z27compute_indexmap_gpu_kernelPd to i8*)), !dbg !2727
  br label %setup.end, !dbg !2727

setup.end:                                        ; preds = %setup.next, %entry
  ret void, !dbg !2728
}

; Function Attrs: noinline uwtable
define dso_local void @_Z37compute_initial_conditions_gpu_kernelP8dcomplexPd(%struct.dcomplex* %u0, double* %starts) #2 !dbg !2729 {
entry:
  %u0.addr = alloca %struct.dcomplex*, align 8
  %starts.addr = alloca double*, align 8
  store %struct.dcomplex* %u0, %struct.dcomplex** %u0.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %u0.addr, metadata !2732, metadata !DIExpression()), !dbg !2733
  store double* %starts, double** %starts.addr, align 8
  call void @llvm.dbg.declare(metadata double** %starts.addr, metadata !2734, metadata !DIExpression()), !dbg !2735
  %0 = bitcast %struct.dcomplex** %u0.addr to i8*, !dbg !2736
  %1 = call i32 @cudaSetupArgument(i8* %0, i64 8, i64 0), !dbg !2736
  %2 = icmp eq i32 %1, 0, !dbg !2736
  br i1 %2, label %setup.next, label %setup.end, !dbg !2736

setup.next:                                       ; preds = %entry
  %3 = bitcast double** %starts.addr to i8*, !dbg !2736
  %4 = call i32 @cudaSetupArgument(i8* %3, i64 8, i64 8), !dbg !2736
  %5 = icmp eq i32 %4, 0, !dbg !2736
  br i1 %5, label %setup.next1, label %setup.end, !dbg !2736

setup.next1:                                      ; preds = %setup.next
  %6 = call i32 @cudaLaunch(i8* bitcast (void (%struct.dcomplex*, double*)* @_Z37compute_initial_conditions_gpu_kernelP8dcomplexPd to i8*)), !dbg !2736
  br label %setup.end, !dbg !2736

setup.end:                                        ; preds = %setup.next1, %setup.next, %entry
  ret void, !dbg !2737
}

; Function Attrs: noinline uwtable
define dso_local void @_Z17evolve_gpu_kernelP8dcomplexS0_Pd(%struct.dcomplex* %u0, %struct.dcomplex* %u1, double* %twiddle) #2 !dbg !2738 {
entry:
  %u0.addr = alloca %struct.dcomplex*, align 8
  %u1.addr = alloca %struct.dcomplex*, align 8
  %twiddle.addr = alloca double*, align 8
  store %struct.dcomplex* %u0, %struct.dcomplex** %u0.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %u0.addr, metadata !2739, metadata !DIExpression()), !dbg !2740
  store %struct.dcomplex* %u1, %struct.dcomplex** %u1.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %u1.addr, metadata !2741, metadata !DIExpression()), !dbg !2742
  store double* %twiddle, double** %twiddle.addr, align 8
  call void @llvm.dbg.declare(metadata double** %twiddle.addr, metadata !2743, metadata !DIExpression()), !dbg !2744
  %0 = bitcast %struct.dcomplex** %u0.addr to i8*, !dbg !2745
  %1 = call i32 @cudaSetupArgument(i8* %0, i64 8, i64 0), !dbg !2745
  %2 = icmp eq i32 %1, 0, !dbg !2745
  br i1 %2, label %setup.next, label %setup.end, !dbg !2745

setup.next:                                       ; preds = %entry
  %3 = bitcast %struct.dcomplex** %u1.addr to i8*, !dbg !2745
  %4 = call i32 @cudaSetupArgument(i8* %3, i64 8, i64 8), !dbg !2745
  %5 = icmp eq i32 %4, 0, !dbg !2745
  br i1 %5, label %setup.next1, label %setup.end, !dbg !2745

setup.next1:                                      ; preds = %setup.next
  %6 = bitcast double** %twiddle.addr to i8*, !dbg !2745
  %7 = call i32 @cudaSetupArgument(i8* %6, i64 8, i64 16), !dbg !2745
  %8 = icmp eq i32 %7, 0, !dbg !2745
  br i1 %8, label %setup.next2, label %setup.end, !dbg !2745

setup.next2:                                      ; preds = %setup.next1
  %9 = call i32 @cudaLaunch(i8* bitcast (void (%struct.dcomplex*, %struct.dcomplex*, double*)* @_Z17evolve_gpu_kernelP8dcomplexS0_Pd to i8*)), !dbg !2745
  br label %setup.end, !dbg !2745

setup.end:                                        ; preds = %setup.next2, %setup.next1, %setup.next, %entry
  ret void, !dbg !2746
}

; Function Attrs: noinline uwtable
define dso_local void @_Z18init_ui_gpu_kernelP8dcomplexS0_Pd(%struct.dcomplex* %u0, %struct.dcomplex* %u1, double* %twiddle) #2 !dbg !2747 {
entry:
  %u0.addr = alloca %struct.dcomplex*, align 8
  %u1.addr = alloca %struct.dcomplex*, align 8
  %twiddle.addr = alloca double*, align 8
  store %struct.dcomplex* %u0, %struct.dcomplex** %u0.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %u0.addr, metadata !2748, metadata !DIExpression()), !dbg !2749
  store %struct.dcomplex* %u1, %struct.dcomplex** %u1.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %u1.addr, metadata !2750, metadata !DIExpression()), !dbg !2751
  store double* %twiddle, double** %twiddle.addr, align 8
  call void @llvm.dbg.declare(metadata double** %twiddle.addr, metadata !2752, metadata !DIExpression()), !dbg !2753
  %0 = bitcast %struct.dcomplex** %u0.addr to i8*, !dbg !2754
  %1 = call i32 @cudaSetupArgument(i8* %0, i64 8, i64 0), !dbg !2754
  %2 = icmp eq i32 %1, 0, !dbg !2754
  br i1 %2, label %setup.next, label %setup.end, !dbg !2754

setup.next:                                       ; preds = %entry
  %3 = bitcast %struct.dcomplex** %u1.addr to i8*, !dbg !2754
  %4 = call i32 @cudaSetupArgument(i8* %3, i64 8, i64 8), !dbg !2754
  %5 = icmp eq i32 %4, 0, !dbg !2754
  br i1 %5, label %setup.next1, label %setup.end, !dbg !2754

setup.next1:                                      ; preds = %setup.next
  %6 = bitcast double** %twiddle.addr to i8*, !dbg !2754
  %7 = call i32 @cudaSetupArgument(i8* %6, i64 8, i64 16), !dbg !2754
  %8 = icmp eq i32 %7, 0, !dbg !2754
  br i1 %8, label %setup.next2, label %setup.end, !dbg !2754

setup.next2:                                      ; preds = %setup.next1
  %9 = call i32 @cudaLaunch(i8* bitcast (void (%struct.dcomplex*, %struct.dcomplex*, double*)* @_Z18init_ui_gpu_kernelP8dcomplexS0_Pd to i8*)), !dbg !2754
  br label %setup.end, !dbg !2754

setup.end:                                        ; preds = %setup.next2, %setup.next1, %setup.next, %entry
  ret void, !dbg !2755
}

declare dso_local i32 @cudaConfigureCall(i64, i32, i64, i32, i64, %struct.CUstream_st*) #3

; Function Attrs: noinline nounwind uwtable
define linkonce_odr dso_local void @_ZN4dim3C2Ejjj(%struct.dim3* %this, i32 %vx, i32 %vy, i32 %vz) unnamed_addr #0 comdat align 2 !dbg !2756 {
entry:
  %this.addr = alloca %struct.dim3*, align 8
  %vx.addr = alloca i32, align 4
  %vy.addr = alloca i32, align 4
  %vz.addr = alloca i32, align 4
  store %struct.dim3* %this, %struct.dim3** %this.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dim3** %this.addr, metadata !2779, metadata !DIExpression()), !dbg !2781
  store i32 %vx, i32* %vx.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %vx.addr, metadata !2782, metadata !DIExpression()), !dbg !2783
  store i32 %vy, i32* %vy.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %vy.addr, metadata !2784, metadata !DIExpression()), !dbg !2785
  store i32 %vz, i32* %vz.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %vz.addr, metadata !2786, metadata !DIExpression()), !dbg !2787
  %this1 = load %struct.dim3*, %struct.dim3** %this.addr, align 8
  %x = getelementptr inbounds %struct.dim3, %struct.dim3* %this1, i32 0, i32 0, !dbg !2788
  %0 = load i32, i32* %vx.addr, align 4, !dbg !2789
  store i32 %0, i32* %x, align 4, !dbg !2788
  %y = getelementptr inbounds %struct.dim3, %struct.dim3* %this1, i32 0, i32 1, !dbg !2790
  %1 = load i32, i32* %vy.addr, align 4, !dbg !2791
  store i32 %1, i32* %y, align 4, !dbg !2790
  %z = getelementptr inbounds %struct.dim3, %struct.dim3* %this1, i32 0, i32 2, !dbg !2792
  %2 = load i32, i32* %vz.addr, align 4, !dbg !2793
  store i32 %2, i32* %z, align 4, !dbg !2792
  ret void, !dbg !2794
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1 immarg) #6

; Function Attrs: noinline nounwind uwtable
define internal void @_ZL6ipow46diPd(double %a, i32 %exponent, double* %result) #0 !dbg !2795 {
entry:
  %a.addr = alloca double, align 8
  %exponent.addr = alloca i32, align 4
  %result.addr = alloca double*, align 8
  %q = alloca double, align 8
  %r = alloca double, align 8
  %n = alloca i32, align 4
  %n2 = alloca i32, align 4
  store double %a, double* %a.addr, align 8
  call void @llvm.dbg.declare(metadata double* %a.addr, metadata !2798, metadata !DIExpression()), !dbg !2799
  store i32 %exponent, i32* %exponent.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %exponent.addr, metadata !2800, metadata !DIExpression()), !dbg !2801
  store double* %result, double** %result.addr, align 8
  call void @llvm.dbg.declare(metadata double** %result.addr, metadata !2802, metadata !DIExpression()), !dbg !2803
  call void @llvm.dbg.declare(metadata double* %q, metadata !2804, metadata !DIExpression()), !dbg !2805
  call void @llvm.dbg.declare(metadata double* %r, metadata !2806, metadata !DIExpression()), !dbg !2807
  call void @llvm.dbg.declare(metadata i32* %n, metadata !2808, metadata !DIExpression()), !dbg !2809
  call void @llvm.dbg.declare(metadata i32* %n2, metadata !2810, metadata !DIExpression()), !dbg !2811
  %0 = load double*, double** %result.addr, align 8, !dbg !2812
  store double 1.000000e+00, double* %0, align 8, !dbg !2813
  %1 = load i32, i32* %exponent.addr, align 4, !dbg !2814
  %cmp = icmp eq i32 %1, 0, !dbg !2816
  br i1 %cmp, label %if.then, label %if.end, !dbg !2817

if.then:                                          ; preds = %entry
  br label %return, !dbg !2818

if.end:                                           ; preds = %entry
  %2 = load double, double* %a.addr, align 8, !dbg !2820
  store double %2, double* %q, align 8, !dbg !2821
  store double 1.000000e+00, double* %r, align 8, !dbg !2822
  %3 = load i32, i32* %exponent.addr, align 4, !dbg !2823
  store i32 %3, i32* %n, align 4, !dbg !2824
  br label %while.cond, !dbg !2825

while.cond:                                       ; preds = %if.end5, %if.end
  %4 = load i32, i32* %n, align 4, !dbg !2826
  %cmp1 = icmp sgt i32 %4, 1, !dbg !2827
  br i1 %cmp1, label %while.body, label %while.end, !dbg !2825

while.body:                                       ; preds = %while.cond
  %5 = load i32, i32* %n, align 4, !dbg !2828
  %div = sdiv i32 %5, 2, !dbg !2830
  store i32 %div, i32* %n2, align 4, !dbg !2831
  %6 = load i32, i32* %n2, align 4, !dbg !2832
  %mul = mul nsw i32 %6, 2, !dbg !2834
  %7 = load i32, i32* %n, align 4, !dbg !2835
  %cmp2 = icmp eq i32 %mul, %7, !dbg !2836
  br i1 %cmp2, label %if.then3, label %if.else, !dbg !2837

if.then3:                                         ; preds = %while.body
  %8 = load double, double* %q, align 8, !dbg !2838
  %call = call double @_Z6randlcPdd(double* %q, double %8), !dbg !2840
  %9 = load i32, i32* %n2, align 4, !dbg !2841
  store i32 %9, i32* %n, align 4, !dbg !2842
  br label %if.end5, !dbg !2843

if.else:                                          ; preds = %while.body
  %10 = load double, double* %q, align 8, !dbg !2844
  %call4 = call double @_Z6randlcPdd(double* %r, double %10), !dbg !2846
  %11 = load i32, i32* %n, align 4, !dbg !2847
  %sub = sub nsw i32 %11, 1, !dbg !2848
  store i32 %sub, i32* %n, align 4, !dbg !2849
  br label %if.end5

if.end5:                                          ; preds = %if.else, %if.then3
  br label %while.cond, !dbg !2825, !llvm.loop !2850

while.end:                                        ; preds = %while.cond
  %12 = load double, double* %q, align 8, !dbg !2852
  %call6 = call double @_Z6randlcPdd(double* %r, double %12), !dbg !2853
  %13 = load double, double* %r, align 8, !dbg !2854
  %14 = load double*, double** %result.addr, align 8, !dbg !2855
  store double %13, double* %14, align 8, !dbg !2856
  br label %return, !dbg !2857

return:                                           ; preds = %while.end, %if.then
  ret void, !dbg !2857
}

; Function Attrs: noinline uwtable
define internal void @_ZL10cffts1_gpuiP8dcomplexS0_S0_S0_S0_(i32 %is, %struct.dcomplex* %u, %struct.dcomplex* %x_in, %struct.dcomplex* %x_out, %struct.dcomplex* %y0, %struct.dcomplex* %y1) #2 !dbg !2858 {
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
  call void @llvm.dbg.declare(metadata i32* %is.addr, metadata !2861, metadata !DIExpression()), !dbg !2862
  store %struct.dcomplex* %u, %struct.dcomplex** %u.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %u.addr, metadata !2863, metadata !DIExpression()), !dbg !2864
  store %struct.dcomplex* %x_in, %struct.dcomplex** %x_in.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %x_in.addr, metadata !2865, metadata !DIExpression()), !dbg !2866
  store %struct.dcomplex* %x_out, %struct.dcomplex** %x_out.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %x_out.addr, metadata !2867, metadata !DIExpression()), !dbg !2868
  store %struct.dcomplex* %y0, %struct.dcomplex** %y0.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %y0.addr, metadata !2869, metadata !DIExpression()), !dbg !2870
  store %struct.dcomplex* %y1, %struct.dcomplex** %y1.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %y1.addr, metadata !2871, metadata !DIExpression()), !dbg !2872
  %0 = load i32, i32* @blocks_per_grid_on_fftx_1, align 4, !dbg !2873
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp, i32 %0, i32 1, i32 1), !dbg !2873
  %1 = load i32, i32* @threads_per_block_on_fftx_1, align 4, !dbg !2874
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp1, i32 %1, i32 1, i32 1), !dbg !2874
  %2 = bitcast { i64, i32 }* %agg.tmp.coerce to i8*, !dbg !2875
  %3 = bitcast %struct.dim3* %agg.tmp to i8*, !dbg !2875
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %2, i8* align 4 %3, i64 12, i1 false), !dbg !2875
  %4 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 0, !dbg !2875
  %5 = load i64, i64* %4, align 4, !dbg !2875
  %6 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 1, !dbg !2875
  %7 = load i32, i32* %6, align 4, !dbg !2875
  %8 = bitcast { i64, i32 }* %agg.tmp1.coerce to i8*, !dbg !2875
  %9 = bitcast %struct.dim3* %agg.tmp1 to i8*, !dbg !2875
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %8, i8* align 4 %9, i64 12, i1 false), !dbg !2875
  %10 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp1.coerce, i32 0, i32 0, !dbg !2875
  %11 = load i64, i64* %10, align 4, !dbg !2875
  %12 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp1.coerce, i32 0, i32 1, !dbg !2875
  %13 = load i32, i32* %12, align 4, !dbg !2875
  %call = call i32 @cudaConfigureCall(i64 %5, i32 %7, i64 %11, i32 %13, i64 0, %struct.CUstream_st* null), !dbg !2875
  %tobool = icmp ne i32 %call, 0, !dbg !2875
  br i1 %tobool, label %kcall.end, label %kcall.configok, !dbg !2876

kcall.configok:                                   ; preds = %entry
  %14 = load %struct.dcomplex*, %struct.dcomplex** %x_in.addr, align 8, !dbg !2877
  %15 = load %struct.dcomplex*, %struct.dcomplex** %y0.addr, align 8, !dbg !2878
  call void @_Z19cffts1_gpu_kernel_1P8dcomplexS0_(%struct.dcomplex* %14, %struct.dcomplex* %15), !dbg !2876
  br label %kcall.end, !dbg !2876

kcall.end:                                        ; preds = %kcall.configok, %entry
  %call2 = call i32 @cudaDeviceSynchronize(), !dbg !2879
  %16 = load i32, i32* @blocks_per_grid_on_fftx_2, align 4, !dbg !2880
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp3, i32 %16, i32 1, i32 1), !dbg !2880
  %17 = load i32, i32* @threads_per_block_on_fftx_2, align 4, !dbg !2881
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp4, i32 %17, i32 1, i32 1), !dbg !2881
  %18 = bitcast { i64, i32 }* %agg.tmp3.coerce to i8*, !dbg !2882
  %19 = bitcast %struct.dim3* %agg.tmp3 to i8*, !dbg !2882
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %18, i8* align 4 %19, i64 12, i1 false), !dbg !2882
  %20 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp3.coerce, i32 0, i32 0, !dbg !2882
  %21 = load i64, i64* %20, align 4, !dbg !2882
  %22 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp3.coerce, i32 0, i32 1, !dbg !2882
  %23 = load i32, i32* %22, align 4, !dbg !2882
  %24 = bitcast { i64, i32 }* %agg.tmp4.coerce to i8*, !dbg !2882
  %25 = bitcast %struct.dim3* %agg.tmp4 to i8*, !dbg !2882
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %24, i8* align 4 %25, i64 12, i1 false), !dbg !2882
  %26 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp4.coerce, i32 0, i32 0, !dbg !2882
  %27 = load i64, i64* %26, align 4, !dbg !2882
  %28 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp4.coerce, i32 0, i32 1, !dbg !2882
  %29 = load i32, i32* %28, align 4, !dbg !2882
  %call5 = call i32 @cudaConfigureCall(i64 %21, i32 %23, i64 %27, i32 %29, i64 0, %struct.CUstream_st* null), !dbg !2882
  %tobool6 = icmp ne i32 %call5, 0, !dbg !2882
  br i1 %tobool6, label %kcall.end8, label %kcall.configok7, !dbg !2883

kcall.configok7:                                  ; preds = %kcall.end
  %30 = load i32, i32* %is.addr, align 4, !dbg !2884
  %31 = load %struct.dcomplex*, %struct.dcomplex** %y0.addr, align 8, !dbg !2885
  %32 = load %struct.dcomplex*, %struct.dcomplex** %y1.addr, align 8, !dbg !2886
  %33 = load %struct.dcomplex*, %struct.dcomplex** %u.addr, align 8, !dbg !2887
  call void @_Z19cffts1_gpu_kernel_2iP8dcomplexS0_S0_(i32 %30, %struct.dcomplex* %31, %struct.dcomplex* %32, %struct.dcomplex* %33), !dbg !2883
  br label %kcall.end8, !dbg !2883

kcall.end8:                                       ; preds = %kcall.configok7, %kcall.end
  %call9 = call i32 @cudaDeviceSynchronize(), !dbg !2888
  %34 = load i32, i32* @blocks_per_grid_on_fftx_3, align 4, !dbg !2889
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp10, i32 %34, i32 1, i32 1), !dbg !2889
  %35 = load i32, i32* @threads_per_block_on_fftx_3, align 4, !dbg !2890
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp11, i32 %35, i32 1, i32 1), !dbg !2890
  %36 = bitcast { i64, i32 }* %agg.tmp10.coerce to i8*, !dbg !2891
  %37 = bitcast %struct.dim3* %agg.tmp10 to i8*, !dbg !2891
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %36, i8* align 4 %37, i64 12, i1 false), !dbg !2891
  %38 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp10.coerce, i32 0, i32 0, !dbg !2891
  %39 = load i64, i64* %38, align 4, !dbg !2891
  %40 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp10.coerce, i32 0, i32 1, !dbg !2891
  %41 = load i32, i32* %40, align 4, !dbg !2891
  %42 = bitcast { i64, i32 }* %agg.tmp11.coerce to i8*, !dbg !2891
  %43 = bitcast %struct.dim3* %agg.tmp11 to i8*, !dbg !2891
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %42, i8* align 4 %43, i64 12, i1 false), !dbg !2891
  %44 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp11.coerce, i32 0, i32 0, !dbg !2891
  %45 = load i64, i64* %44, align 4, !dbg !2891
  %46 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp11.coerce, i32 0, i32 1, !dbg !2891
  %47 = load i32, i32* %46, align 4, !dbg !2891
  %call12 = call i32 @cudaConfigureCall(i64 %39, i32 %41, i64 %45, i32 %47, i64 0, %struct.CUstream_st* null), !dbg !2891
  %tobool13 = icmp ne i32 %call12, 0, !dbg !2891
  br i1 %tobool13, label %kcall.end15, label %kcall.configok14, !dbg !2892

kcall.configok14:                                 ; preds = %kcall.end8
  %48 = load %struct.dcomplex*, %struct.dcomplex** %x_out.addr, align 8, !dbg !2893
  %49 = load %struct.dcomplex*, %struct.dcomplex** %y0.addr, align 8, !dbg !2894
  call void @_Z19cffts1_gpu_kernel_3P8dcomplexS0_(%struct.dcomplex* %48, %struct.dcomplex* %49), !dbg !2892
  br label %kcall.end15, !dbg !2892

kcall.end15:                                      ; preds = %kcall.configok14, %kcall.end8
  %call16 = call i32 @cudaDeviceSynchronize(), !dbg !2895
  ret void, !dbg !2896
}

; Function Attrs: noinline uwtable
define internal void @_ZL10cffts2_gpuiP8dcomplexS0_S0_S0_S0_(i32 %is, %struct.dcomplex* %u, %struct.dcomplex* %x_in, %struct.dcomplex* %x_out, %struct.dcomplex* %y0, %struct.dcomplex* %y1) #2 !dbg !2897 {
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
  call void @llvm.dbg.declare(metadata i32* %is.addr, metadata !2900, metadata !DIExpression()), !dbg !2901
  store %struct.dcomplex* %u, %struct.dcomplex** %u.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %u.addr, metadata !2902, metadata !DIExpression()), !dbg !2903
  store %struct.dcomplex* %x_in, %struct.dcomplex** %x_in.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %x_in.addr, metadata !2904, metadata !DIExpression()), !dbg !2905
  store %struct.dcomplex* %x_out, %struct.dcomplex** %x_out.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %x_out.addr, metadata !2906, metadata !DIExpression()), !dbg !2907
  store %struct.dcomplex* %y0, %struct.dcomplex** %y0.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %y0.addr, metadata !2908, metadata !DIExpression()), !dbg !2909
  store %struct.dcomplex* %y1, %struct.dcomplex** %y1.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %y1.addr, metadata !2910, metadata !DIExpression()), !dbg !2911
  %0 = load i32, i32* @blocks_per_grid_on_ffty_1, align 4, !dbg !2912
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp, i32 %0, i32 1, i32 1), !dbg !2912
  %1 = load i32, i32* @threads_per_block_on_ffty_1, align 4, !dbg !2913
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp1, i32 %1, i32 1, i32 1), !dbg !2913
  %2 = bitcast { i64, i32 }* %agg.tmp.coerce to i8*, !dbg !2914
  %3 = bitcast %struct.dim3* %agg.tmp to i8*, !dbg !2914
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %2, i8* align 4 %3, i64 12, i1 false), !dbg !2914
  %4 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 0, !dbg !2914
  %5 = load i64, i64* %4, align 4, !dbg !2914
  %6 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 1, !dbg !2914
  %7 = load i32, i32* %6, align 4, !dbg !2914
  %8 = bitcast { i64, i32 }* %agg.tmp1.coerce to i8*, !dbg !2914
  %9 = bitcast %struct.dim3* %agg.tmp1 to i8*, !dbg !2914
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %8, i8* align 4 %9, i64 12, i1 false), !dbg !2914
  %10 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp1.coerce, i32 0, i32 0, !dbg !2914
  %11 = load i64, i64* %10, align 4, !dbg !2914
  %12 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp1.coerce, i32 0, i32 1, !dbg !2914
  %13 = load i32, i32* %12, align 4, !dbg !2914
  %call = call i32 @cudaConfigureCall(i64 %5, i32 %7, i64 %11, i32 %13, i64 0, %struct.CUstream_st* null), !dbg !2914
  %tobool = icmp ne i32 %call, 0, !dbg !2914
  br i1 %tobool, label %kcall.end, label %kcall.configok, !dbg !2915

kcall.configok:                                   ; preds = %entry
  %14 = load %struct.dcomplex*, %struct.dcomplex** %x_in.addr, align 8, !dbg !2916
  %15 = load %struct.dcomplex*, %struct.dcomplex** %y0.addr, align 8, !dbg !2917
  call void @_Z19cffts2_gpu_kernel_1P8dcomplexS0_(%struct.dcomplex* %14, %struct.dcomplex* %15), !dbg !2915
  br label %kcall.end, !dbg !2915

kcall.end:                                        ; preds = %kcall.configok, %entry
  %call2 = call i32 @cudaDeviceSynchronize(), !dbg !2918
  %16 = load i32, i32* @blocks_per_grid_on_ffty_2, align 4, !dbg !2919
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp3, i32 %16, i32 1, i32 1), !dbg !2919
  %17 = load i32, i32* @threads_per_block_on_ffty_2, align 4, !dbg !2920
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp4, i32 %17, i32 1, i32 1), !dbg !2920
  %18 = bitcast { i64, i32 }* %agg.tmp3.coerce to i8*, !dbg !2921
  %19 = bitcast %struct.dim3* %agg.tmp3 to i8*, !dbg !2921
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %18, i8* align 4 %19, i64 12, i1 false), !dbg !2921
  %20 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp3.coerce, i32 0, i32 0, !dbg !2921
  %21 = load i64, i64* %20, align 4, !dbg !2921
  %22 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp3.coerce, i32 0, i32 1, !dbg !2921
  %23 = load i32, i32* %22, align 4, !dbg !2921
  %24 = bitcast { i64, i32 }* %agg.tmp4.coerce to i8*, !dbg !2921
  %25 = bitcast %struct.dim3* %agg.tmp4 to i8*, !dbg !2921
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %24, i8* align 4 %25, i64 12, i1 false), !dbg !2921
  %26 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp4.coerce, i32 0, i32 0, !dbg !2921
  %27 = load i64, i64* %26, align 4, !dbg !2921
  %28 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp4.coerce, i32 0, i32 1, !dbg !2921
  %29 = load i32, i32* %28, align 4, !dbg !2921
  %call5 = call i32 @cudaConfigureCall(i64 %21, i32 %23, i64 %27, i32 %29, i64 0, %struct.CUstream_st* null), !dbg !2921
  %tobool6 = icmp ne i32 %call5, 0, !dbg !2921
  br i1 %tobool6, label %kcall.end8, label %kcall.configok7, !dbg !2922

kcall.configok7:                                  ; preds = %kcall.end
  %30 = load i32, i32* %is.addr, align 4, !dbg !2923
  %31 = load %struct.dcomplex*, %struct.dcomplex** %y0.addr, align 8, !dbg !2924
  %32 = load %struct.dcomplex*, %struct.dcomplex** %y1.addr, align 8, !dbg !2925
  %33 = load %struct.dcomplex*, %struct.dcomplex** %u.addr, align 8, !dbg !2926
  call void @_Z19cffts2_gpu_kernel_2iP8dcomplexS0_S0_(i32 %30, %struct.dcomplex* %31, %struct.dcomplex* %32, %struct.dcomplex* %33), !dbg !2922
  br label %kcall.end8, !dbg !2922

kcall.end8:                                       ; preds = %kcall.configok7, %kcall.end
  %call9 = call i32 @cudaDeviceSynchronize(), !dbg !2927
  %34 = load i32, i32* @blocks_per_grid_on_ffty_3, align 4, !dbg !2928
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp10, i32 %34, i32 1, i32 1), !dbg !2928
  %35 = load i32, i32* @threads_per_block_on_ffty_3, align 4, !dbg !2929
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp11, i32 %35, i32 1, i32 1), !dbg !2929
  %36 = bitcast { i64, i32 }* %agg.tmp10.coerce to i8*, !dbg !2930
  %37 = bitcast %struct.dim3* %agg.tmp10 to i8*, !dbg !2930
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %36, i8* align 4 %37, i64 12, i1 false), !dbg !2930
  %38 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp10.coerce, i32 0, i32 0, !dbg !2930
  %39 = load i64, i64* %38, align 4, !dbg !2930
  %40 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp10.coerce, i32 0, i32 1, !dbg !2930
  %41 = load i32, i32* %40, align 4, !dbg !2930
  %42 = bitcast { i64, i32 }* %agg.tmp11.coerce to i8*, !dbg !2930
  %43 = bitcast %struct.dim3* %agg.tmp11 to i8*, !dbg !2930
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %42, i8* align 4 %43, i64 12, i1 false), !dbg !2930
  %44 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp11.coerce, i32 0, i32 0, !dbg !2930
  %45 = load i64, i64* %44, align 4, !dbg !2930
  %46 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp11.coerce, i32 0, i32 1, !dbg !2930
  %47 = load i32, i32* %46, align 4, !dbg !2930
  %call12 = call i32 @cudaConfigureCall(i64 %39, i32 %41, i64 %45, i32 %47, i64 0, %struct.CUstream_st* null), !dbg !2930
  %tobool13 = icmp ne i32 %call12, 0, !dbg !2930
  br i1 %tobool13, label %kcall.end15, label %kcall.configok14, !dbg !2931

kcall.configok14:                                 ; preds = %kcall.end8
  %48 = load %struct.dcomplex*, %struct.dcomplex** %x_out.addr, align 8, !dbg !2932
  %49 = load %struct.dcomplex*, %struct.dcomplex** %y0.addr, align 8, !dbg !2933
  call void @_Z19cffts2_gpu_kernel_3P8dcomplexS0_(%struct.dcomplex* %48, %struct.dcomplex* %49), !dbg !2931
  br label %kcall.end15, !dbg !2931

kcall.end15:                                      ; preds = %kcall.configok14, %kcall.end8
  %call16 = call i32 @cudaDeviceSynchronize(), !dbg !2934
  ret void, !dbg !2935
}

; Function Attrs: noinline uwtable
define internal void @_ZL10cffts3_gpuiP8dcomplexS0_S0_S0_S0_(i32 %is, %struct.dcomplex* %u, %struct.dcomplex* %x_in, %struct.dcomplex* %x_out, %struct.dcomplex* %y0, %struct.dcomplex* %y1) #2 !dbg !2936 {
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
  call void @llvm.dbg.declare(metadata i32* %is.addr, metadata !2937, metadata !DIExpression()), !dbg !2938
  store %struct.dcomplex* %u, %struct.dcomplex** %u.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %u.addr, metadata !2939, metadata !DIExpression()), !dbg !2940
  store %struct.dcomplex* %x_in, %struct.dcomplex** %x_in.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %x_in.addr, metadata !2941, metadata !DIExpression()), !dbg !2942
  store %struct.dcomplex* %x_out, %struct.dcomplex** %x_out.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %x_out.addr, metadata !2943, metadata !DIExpression()), !dbg !2944
  store %struct.dcomplex* %y0, %struct.dcomplex** %y0.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %y0.addr, metadata !2945, metadata !DIExpression()), !dbg !2946
  store %struct.dcomplex* %y1, %struct.dcomplex** %y1.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %y1.addr, metadata !2947, metadata !DIExpression()), !dbg !2948
  %0 = load i32, i32* @blocks_per_grid_on_fftz_1, align 4, !dbg !2949
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp, i32 %0, i32 1, i32 1), !dbg !2949
  %1 = load i32, i32* @threads_per_block_on_fftz_1, align 4, !dbg !2950
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp1, i32 %1, i32 1, i32 1), !dbg !2950
  %2 = bitcast { i64, i32 }* %agg.tmp.coerce to i8*, !dbg !2951
  %3 = bitcast %struct.dim3* %agg.tmp to i8*, !dbg !2951
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %2, i8* align 4 %3, i64 12, i1 false), !dbg !2951
  %4 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 0, !dbg !2951
  %5 = load i64, i64* %4, align 4, !dbg !2951
  %6 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp.coerce, i32 0, i32 1, !dbg !2951
  %7 = load i32, i32* %6, align 4, !dbg !2951
  %8 = bitcast { i64, i32 }* %agg.tmp1.coerce to i8*, !dbg !2951
  %9 = bitcast %struct.dim3* %agg.tmp1 to i8*, !dbg !2951
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %8, i8* align 4 %9, i64 12, i1 false), !dbg !2951
  %10 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp1.coerce, i32 0, i32 0, !dbg !2951
  %11 = load i64, i64* %10, align 4, !dbg !2951
  %12 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp1.coerce, i32 0, i32 1, !dbg !2951
  %13 = load i32, i32* %12, align 4, !dbg !2951
  %call = call i32 @cudaConfigureCall(i64 %5, i32 %7, i64 %11, i32 %13, i64 0, %struct.CUstream_st* null), !dbg !2951
  %tobool = icmp ne i32 %call, 0, !dbg !2951
  br i1 %tobool, label %kcall.end, label %kcall.configok, !dbg !2952

kcall.configok:                                   ; preds = %entry
  %14 = load %struct.dcomplex*, %struct.dcomplex** %x_in.addr, align 8, !dbg !2953
  %15 = load %struct.dcomplex*, %struct.dcomplex** %y0.addr, align 8, !dbg !2954
  call void @_Z19cffts3_gpu_kernel_1P8dcomplexS0_(%struct.dcomplex* %14, %struct.dcomplex* %15), !dbg !2952
  br label %kcall.end, !dbg !2952

kcall.end:                                        ; preds = %kcall.configok, %entry
  %call2 = call i32 @cudaDeviceSynchronize(), !dbg !2955
  %16 = load i32, i32* @blocks_per_grid_on_fftz_2, align 4, !dbg !2956
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp3, i32 %16, i32 1, i32 1), !dbg !2956
  %17 = load i32, i32* @threads_per_block_on_fftz_2, align 4, !dbg !2957
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp4, i32 %17, i32 1, i32 1), !dbg !2957
  %18 = bitcast { i64, i32 }* %agg.tmp3.coerce to i8*, !dbg !2958
  %19 = bitcast %struct.dim3* %agg.tmp3 to i8*, !dbg !2958
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %18, i8* align 4 %19, i64 12, i1 false), !dbg !2958
  %20 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp3.coerce, i32 0, i32 0, !dbg !2958
  %21 = load i64, i64* %20, align 4, !dbg !2958
  %22 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp3.coerce, i32 0, i32 1, !dbg !2958
  %23 = load i32, i32* %22, align 4, !dbg !2958
  %24 = bitcast { i64, i32 }* %agg.tmp4.coerce to i8*, !dbg !2958
  %25 = bitcast %struct.dim3* %agg.tmp4 to i8*, !dbg !2958
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %24, i8* align 4 %25, i64 12, i1 false), !dbg !2958
  %26 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp4.coerce, i32 0, i32 0, !dbg !2958
  %27 = load i64, i64* %26, align 4, !dbg !2958
  %28 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp4.coerce, i32 0, i32 1, !dbg !2958
  %29 = load i32, i32* %28, align 4, !dbg !2958
  %call5 = call i32 @cudaConfigureCall(i64 %21, i32 %23, i64 %27, i32 %29, i64 0, %struct.CUstream_st* null), !dbg !2958
  %tobool6 = icmp ne i32 %call5, 0, !dbg !2958
  br i1 %tobool6, label %kcall.end8, label %kcall.configok7, !dbg !2959

kcall.configok7:                                  ; preds = %kcall.end
  %30 = load i32, i32* %is.addr, align 4, !dbg !2960
  %31 = load %struct.dcomplex*, %struct.dcomplex** %y0.addr, align 8, !dbg !2961
  %32 = load %struct.dcomplex*, %struct.dcomplex** %y1.addr, align 8, !dbg !2962
  %33 = load %struct.dcomplex*, %struct.dcomplex** %u.addr, align 8, !dbg !2963
  call void @_Z19cffts3_gpu_kernel_2iP8dcomplexS0_S0_(i32 %30, %struct.dcomplex* %31, %struct.dcomplex* %32, %struct.dcomplex* %33), !dbg !2959
  br label %kcall.end8, !dbg !2959

kcall.end8:                                       ; preds = %kcall.configok7, %kcall.end
  %call9 = call i32 @cudaDeviceSynchronize(), !dbg !2964
  %34 = load i32, i32* @blocks_per_grid_on_fftz_3, align 4, !dbg !2965
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp10, i32 %34, i32 1, i32 1), !dbg !2965
  %35 = load i32, i32* @threads_per_block_on_fftz_3, align 4, !dbg !2966
  call void @_ZN4dim3C2Ejjj(%struct.dim3* %agg.tmp11, i32 %35, i32 1, i32 1), !dbg !2966
  %36 = bitcast { i64, i32 }* %agg.tmp10.coerce to i8*, !dbg !2967
  %37 = bitcast %struct.dim3* %agg.tmp10 to i8*, !dbg !2967
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %36, i8* align 4 %37, i64 12, i1 false), !dbg !2967
  %38 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp10.coerce, i32 0, i32 0, !dbg !2967
  %39 = load i64, i64* %38, align 4, !dbg !2967
  %40 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp10.coerce, i32 0, i32 1, !dbg !2967
  %41 = load i32, i32* %40, align 4, !dbg !2967
  %42 = bitcast { i64, i32 }* %agg.tmp11.coerce to i8*, !dbg !2967
  %43 = bitcast %struct.dim3* %agg.tmp11 to i8*, !dbg !2967
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %42, i8* align 4 %43, i64 12, i1 false), !dbg !2967
  %44 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp11.coerce, i32 0, i32 0, !dbg !2967
  %45 = load i64, i64* %44, align 4, !dbg !2967
  %46 = getelementptr inbounds { i64, i32 }, { i64, i32 }* %agg.tmp11.coerce, i32 0, i32 1, !dbg !2967
  %47 = load i32, i32* %46, align 4, !dbg !2967
  %call12 = call i32 @cudaConfigureCall(i64 %39, i32 %41, i64 %45, i32 %47, i64 0, %struct.CUstream_st* null), !dbg !2967
  %tobool13 = icmp ne i32 %call12, 0, !dbg !2967
  br i1 %tobool13, label %kcall.end15, label %kcall.configok14, !dbg !2968

kcall.configok14:                                 ; preds = %kcall.end8
  %48 = load %struct.dcomplex*, %struct.dcomplex** %x_out.addr, align 8, !dbg !2969
  %49 = load %struct.dcomplex*, %struct.dcomplex** %y0.addr, align 8, !dbg !2970
  call void @_Z19cffts3_gpu_kernel_3P8dcomplexS0_(%struct.dcomplex* %48, %struct.dcomplex* %49), !dbg !2968
  br label %kcall.end15, !dbg !2968

kcall.end15:                                      ; preds = %kcall.configok14, %kcall.end8
  %call16 = call i32 @cudaDeviceSynchronize(), !dbg !2971
  ret void, !dbg !2972
}

; Function Attrs: noinline nounwind uwtable
define internal i32 @_ZL5ilog2i(i32 %n) #0 !dbg !2973 {
entry:
  %retval = alloca i32, align 4
  %n.addr = alloca i32, align 4
  %nn = alloca i32, align 4
  %lg = alloca i32, align 4
  store i32 %n, i32* %n.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %n.addr, metadata !2974, metadata !DIExpression()), !dbg !2975
  call void @llvm.dbg.declare(metadata i32* %nn, metadata !2976, metadata !DIExpression()), !dbg !2977
  call void @llvm.dbg.declare(metadata i32* %lg, metadata !2978, metadata !DIExpression()), !dbg !2979
  %0 = load i32, i32* %n.addr, align 4, !dbg !2980
  %cmp = icmp eq i32 %0, 1, !dbg !2982
  br i1 %cmp, label %if.then, label %if.end, !dbg !2983

if.then:                                          ; preds = %entry
  store i32 0, i32* %retval, align 4, !dbg !2984
  br label %return, !dbg !2984

if.end:                                           ; preds = %entry
  store i32 1, i32* %lg, align 4, !dbg !2986
  store i32 2, i32* %nn, align 4, !dbg !2987
  br label %while.cond, !dbg !2988

while.cond:                                       ; preds = %while.body, %if.end
  %1 = load i32, i32* %nn, align 4, !dbg !2989
  %2 = load i32, i32* %n.addr, align 4, !dbg !2990
  %cmp1 = icmp slt i32 %1, %2, !dbg !2991
  br i1 %cmp1, label %while.body, label %while.end, !dbg !2988

while.body:                                       ; preds = %while.cond
  %3 = load i32, i32* %nn, align 4, !dbg !2992
  %shl = shl i32 %3, 1, !dbg !2994
  store i32 %shl, i32* %nn, align 4, !dbg !2995
  %4 = load i32, i32* %lg, align 4, !dbg !2996
  %inc = add nsw i32 %4, 1, !dbg !2996
  store i32 %inc, i32* %lg, align 4, !dbg !2996
  br label %while.cond, !dbg !2988, !llvm.loop !2997

while.end:                                        ; preds = %while.cond
  %5 = load i32, i32* %lg, align 4, !dbg !2999
  store i32 %5, i32* %retval, align 4, !dbg !3000
  br label %return, !dbg !3000

return:                                           ; preds = %while.end, %if.then
  %6 = load i32, i32* %retval, align 4, !dbg !3001
  ret i32 %6, !dbg !3001
}

; Function Attrs: nounwind
declare dso_local double @cos(double) #4

; Function Attrs: nounwind
declare dso_local double @sin(double) #4

declare dso_local i32 @cudaFree(i8*) #3

; Function Attrs: nounwind readnone speculatable
declare double @llvm.ceil.f64(double) #1

; Function Attrs: noinline uwtable
define internal i32 @_ZL10cudaMallocI8dcomplexE9cudaErrorPPT_m(%struct.dcomplex** %devPtr, i64 %size) #2 !dbg !3002 {
entry:
  %devPtr.addr = alloca %struct.dcomplex**, align 8
  %size.addr = alloca i64, align 8
  store %struct.dcomplex** %devPtr, %struct.dcomplex*** %devPtr.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex*** %devPtr.addr, metadata !3011, metadata !DIExpression()), !dbg !3012
  store i64 %size, i64* %size.addr, align 8
  call void @llvm.dbg.declare(metadata i64* %size.addr, metadata !3013, metadata !DIExpression()), !dbg !3014
  %0 = load %struct.dcomplex**, %struct.dcomplex*** %devPtr.addr, align 8, !dbg !3015
  %1 = bitcast %struct.dcomplex** %0 to i8*, !dbg !3015
  %2 = bitcast i8* %1 to i8**, !dbg !3016
  %3 = load i64, i64* %size.addr, align 8, !dbg !3017
  %call = call i32 @cudaMalloc(i8** %2, i64 %3), !dbg !3018
  ret i32 %call, !dbg !3019
}

; Function Attrs: noinline uwtable
define internal i32 @_ZL10cudaMallocIdE9cudaErrorPPT_m(double** %devPtr, i64 %size) #2 !dbg !3020 {
entry:
  %devPtr.addr = alloca double**, align 8
  %size.addr = alloca i64, align 8
  store double** %devPtr, double*** %devPtr.addr, align 8
  call void @llvm.dbg.declare(metadata double*** %devPtr.addr, metadata !3026, metadata !DIExpression()), !dbg !3027
  store i64 %size, i64* %size.addr, align 8
  call void @llvm.dbg.declare(metadata i64* %size.addr, metadata !3028, metadata !DIExpression()), !dbg !3029
  %0 = load double**, double*** %devPtr.addr, align 8, !dbg !3030
  %1 = bitcast double** %0 to i8*, !dbg !3030
  %2 = bitcast i8* %1 to i8**, !dbg !3031
  %3 = load i64, i64* %size.addr, align 8, !dbg !3032
  %call = call i32 @cudaMalloc(i8** %2, i64 %3), !dbg !3033
  ret i32 %call, !dbg !3034
}

declare dso_local void @omp_set_num_threads(i32) #3

declare dso_local i32 @cudaMalloc(i8**, i64) #3

; Function Attrs: nounwind
declare dso_local double @sqrt(double) #4

; Function Attrs: noinline nounwind uwtable
define internal { double, double } @_ZL12dcomplex_div8dcomplexS_(double %z1.coerce0, double %z1.coerce1, double %z2.coerce0, double %z2.coerce1) #0 !dbg !3035 {
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
  call void @llvm.dbg.declare(metadata %struct.dcomplex* %z1, metadata !3038, metadata !DIExpression()), !dbg !3039
  call void @llvm.dbg.declare(metadata %struct.dcomplex* %z2, metadata !3040, metadata !DIExpression()), !dbg !3041
  call void @llvm.dbg.declare(metadata double* %a, metadata !3042, metadata !DIExpression()), !dbg !3043
  %real = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %z1, i32 0, i32 0, !dbg !3044
  %6 = load double, double* %real, align 8, !dbg !3044
  store double %6, double* %a, align 8, !dbg !3043
  call void @llvm.dbg.declare(metadata double* %b, metadata !3045, metadata !DIExpression()), !dbg !3046
  %imag = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %z1, i32 0, i32 1, !dbg !3047
  %7 = load double, double* %imag, align 8, !dbg !3047
  store double %7, double* %b, align 8, !dbg !3046
  call void @llvm.dbg.declare(metadata double* %c, metadata !3048, metadata !DIExpression()), !dbg !3049
  %real1 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %z2, i32 0, i32 0, !dbg !3050
  %8 = load double, double* %real1, align 8, !dbg !3050
  store double %8, double* %c, align 8, !dbg !3049
  call void @llvm.dbg.declare(metadata double* %d, metadata !3051, metadata !DIExpression()), !dbg !3052
  %imag2 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %z2, i32 0, i32 1, !dbg !3053
  %9 = load double, double* %imag2, align 8, !dbg !3053
  store double %9, double* %d, align 8, !dbg !3052
  call void @llvm.dbg.declare(metadata double* %divisor, metadata !3054, metadata !DIExpression()), !dbg !3055
  %10 = load double, double* %c, align 8, !dbg !3056
  %11 = load double, double* %c, align 8, !dbg !3057
  %mul = fmul contract double %10, %11, !dbg !3058
  %12 = load double, double* %d, align 8, !dbg !3059
  %13 = load double, double* %d, align 8, !dbg !3060
  %mul3 = fmul contract double %12, %13, !dbg !3061
  %add = fadd contract double %mul, %mul3, !dbg !3062
  store double %add, double* %divisor, align 8, !dbg !3055
  call void @llvm.dbg.declare(metadata double* %real4, metadata !3063, metadata !DIExpression()), !dbg !3064
  %14 = load double, double* %a, align 8, !dbg !3065
  %15 = load double, double* %c, align 8, !dbg !3066
  %mul5 = fmul contract double %14, %15, !dbg !3067
  %16 = load double, double* %b, align 8, !dbg !3068
  %17 = load double, double* %d, align 8, !dbg !3069
  %mul6 = fmul contract double %16, %17, !dbg !3070
  %add7 = fadd contract double %mul5, %mul6, !dbg !3071
  %18 = load double, double* %divisor, align 8, !dbg !3072
  %div = fdiv double %add7, %18, !dbg !3073
  store double %div, double* %real4, align 8, !dbg !3064
  call void @llvm.dbg.declare(metadata double* %imag8, metadata !3074, metadata !DIExpression()), !dbg !3075
  %19 = load double, double* %b, align 8, !dbg !3076
  %20 = load double, double* %c, align 8, !dbg !3077
  %mul9 = fmul contract double %19, %20, !dbg !3078
  %21 = load double, double* %a, align 8, !dbg !3079
  %22 = load double, double* %d, align 8, !dbg !3080
  %mul10 = fmul contract double %21, %22, !dbg !3081
  %sub = fsub contract double %mul9, %mul10, !dbg !3082
  %23 = load double, double* %divisor, align 8, !dbg !3083
  %div11 = fdiv double %sub, %23, !dbg !3084
  store double %div11, double* %imag8, align 8, !dbg !3075
  call void @llvm.dbg.declare(metadata %struct.dcomplex* %retval, metadata !3085, metadata !DIExpression()), !dbg !3086
  %real12 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %retval, i32 0, i32 0, !dbg !3087
  %24 = load double, double* %real4, align 8, !dbg !3088
  store double %24, double* %real12, align 8, !dbg !3087
  %imag13 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %retval, i32 0, i32 1, !dbg !3087
  %25 = load double, double* %imag8, align 8, !dbg !3089
  store double %25, double* %imag13, align 8, !dbg !3087
  %26 = bitcast %struct.dcomplex* %retval to { double, double }*, !dbg !3090
  %27 = load { double, double }, { double, double }* %26, align 8, !dbg !3090
  ret { double, double } %27, !dbg !3090
}

attributes #0 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { noinline uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { noinline norecurse uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { argmemonly nounwind }
attributes #7 = { nounwind }

!llvm.module.flags = !{!1050, !1051, !1052, !1053}
!llvm.dbg.cu = !{!2}
!llvm.ident = !{!1054}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "starts_device", scope: !2, file: !3, line: 173, type: !106, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 9.0.0 (https://github.com/yebinchon/llvm-project/ a1efa594106d738d0b74c9e4e2b9b779eb8b7d25)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !96, globals: !112, imports: !299, nameTableKind: None)
!3 = !DIFile(filename: "ft.cu", directory: "/scratch/ah7226/NPB-GPU/CUDA/FT")
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
!132 = !DIFile(filename: "llvm-install-tulip/lib/clang/9.0.0/include/stddef.h", directory: "/scratch/ah7226")
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
!303 = !DIFile(filename: "llvm-install-tulip/lib/clang/9.0.0/include/__clang_cuda_math_forward_declares.h", directory: "/scratch/ah7226")
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
!551 = !DICompositeType(tag: DW_TAG_structure_type, file: !491, line: 59, flags: DIFlagFwdDecl, identifier: "_ZTS5div_t")
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
!763 = !DIFile(filename: "llvm-install-tulip/lib/clang/9.0.0/include/__clang_cuda_cmath.h", directory: "/scratch/ah7226")
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
!876 = !DICompositeType(tag: DW_TAG_structure_type, name: "_IO_FILE", file: !877, line: 49, flags: DIFlagFwdDecl, identifier: "_ZTS8_IO_FILE")
!877 = !DIFile(filename: "/usr/include/x86_64-linux-gnu/bits/types/struct_FILE.h", directory: "")
!878 = !DIFile(filename: "/usr/lib/gcc/x86_64-linux-gnu/11/../../../../include/c++/11/cstdio", directory: "")
!879 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !301, entity: !880, file: !878, line: 99)
!880 = !DIDerivedType(tag: DW_TAG_typedef, name: "fpos_t", file: !881, line: 84, baseType: !882)
!881 = !DIFile(filename: "/usr/include/stdio.h", directory: "")
!882 = !DIDerivedType(tag: DW_TAG_typedef, name: "__fpos_t", file: !883, line: 14, baseType: !884)
!883 = !DIFile(filename: "/usr/include/x86_64-linux-gnu/bits/types/__fpos_t.h", directory: "")
!884 = !DICompositeType(tag: DW_TAG_structure_type, name: "_G_fpos_t", file: !883, line: 10, flags: DIFlagFwdDecl, identifier: "_ZTS9_G_fpos_t")
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
!1020 = !DICompositeType(tag: DW_TAG_structure_type, name: "__va_list_tag", file: !3, flags: DIFlagFwdDecl, identifier: "_ZTS13__va_list_tag")
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
!1050 = !{i32 2, !"SDK Version", [2 x i32] [i32 7, i32 0]}
!1051 = !{i32 2, !"Dwarf Version", i32 4}
!1052 = !{i32 2, !"Debug Info Version", i32 3}
!1053 = !{i32 1, !"wchar_size", i32 4}
!1054 = !{!"clang version 9.0.0 (https://github.com/yebinchon/llvm-project/ a1efa594106d738d0b74c9e4e2b9b779eb8b7d25)"}
!1055 = distinct !DISubprogram(name: "randlc", linkageName: "_Z6randlcPdd", scope: !3, file: !3, line: 371, type: !1056, scopeLine: 371, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !1058)
!1056 = !DISubroutineType(types: !1057)
!1057 = !{!104, !106, !104}
!1058 = !{}
!1059 = !DILocalVariable(name: "x", arg: 1, scope: !1055, file: !3, line: 371, type: !106)
!1060 = !DILocation(line: 371, column: 24, scope: !1055)
!1061 = !DILocalVariable(name: "a", arg: 2, scope: !1055, file: !3, line: 371, type: !104)
!1062 = !DILocation(line: 371, column: 34, scope: !1055)
!1063 = !DILocalVariable(name: "t1", scope: !1055, file: !3, line: 372, type: !104)
!1064 = !DILocation(line: 372, column: 10, scope: !1055)
!1065 = !DILocalVariable(name: "t2", scope: !1055, file: !3, line: 372, type: !104)
!1066 = !DILocation(line: 372, column: 13, scope: !1055)
!1067 = !DILocalVariable(name: "t3", scope: !1055, file: !3, line: 372, type: !104)
!1068 = !DILocation(line: 372, column: 16, scope: !1055)
!1069 = !DILocalVariable(name: "t4", scope: !1055, file: !3, line: 372, type: !104)
!1070 = !DILocation(line: 372, column: 19, scope: !1055)
!1071 = !DILocalVariable(name: "a1", scope: !1055, file: !3, line: 372, type: !104)
!1072 = !DILocation(line: 372, column: 22, scope: !1055)
!1073 = !DILocalVariable(name: "a2", scope: !1055, file: !3, line: 372, type: !104)
!1074 = !DILocation(line: 372, column: 25, scope: !1055)
!1075 = !DILocalVariable(name: "x1", scope: !1055, file: !3, line: 372, type: !104)
!1076 = !DILocation(line: 372, column: 28, scope: !1055)
!1077 = !DILocalVariable(name: "x2", scope: !1055, file: !3, line: 372, type: !104)
!1078 = !DILocation(line: 372, column: 31, scope: !1055)
!1079 = !DILocalVariable(name: "z", scope: !1055, file: !3, line: 372, type: !104)
!1080 = !DILocation(line: 372, column: 34, scope: !1055)
!1081 = !DILocation(line: 379, column: 14, scope: !1055)
!1082 = !DILocation(line: 379, column: 12, scope: !1055)
!1083 = !DILocation(line: 379, column: 6, scope: !1055)
!1084 = !DILocation(line: 380, column: 13, scope: !1055)
!1085 = !DILocation(line: 380, column: 8, scope: !1055)
!1086 = !DILocation(line: 380, column: 6, scope: !1055)
!1087 = !DILocation(line: 381, column: 8, scope: !1055)
!1088 = !DILocation(line: 381, column: 18, scope: !1055)
!1089 = !DILocation(line: 381, column: 16, scope: !1055)
!1090 = !DILocation(line: 381, column: 10, scope: !1055)
!1091 = !DILocation(line: 381, column: 6, scope: !1055)
!1092 = !DILocation(line: 390, column: 16, scope: !1055)
!1093 = !DILocation(line: 390, column: 15, scope: !1055)
!1094 = !DILocation(line: 390, column: 12, scope: !1055)
!1095 = !DILocation(line: 390, column: 6, scope: !1055)
!1096 = !DILocation(line: 391, column: 13, scope: !1055)
!1097 = !DILocation(line: 391, column: 8, scope: !1055)
!1098 = !DILocation(line: 391, column: 6, scope: !1055)
!1099 = !DILocation(line: 392, column: 10, scope: !1055)
!1100 = !DILocation(line: 392, column: 9, scope: !1055)
!1101 = !DILocation(line: 392, column: 21, scope: !1055)
!1102 = !DILocation(line: 392, column: 19, scope: !1055)
!1103 = !DILocation(line: 392, column: 13, scope: !1055)
!1104 = !DILocation(line: 392, column: 6, scope: !1055)
!1105 = !DILocation(line: 393, column: 8, scope: !1055)
!1106 = !DILocation(line: 393, column: 13, scope: !1055)
!1107 = !DILocation(line: 393, column: 11, scope: !1055)
!1108 = !DILocation(line: 393, column: 18, scope: !1055)
!1109 = !DILocation(line: 393, column: 23, scope: !1055)
!1110 = !DILocation(line: 393, column: 21, scope: !1055)
!1111 = !DILocation(line: 393, column: 16, scope: !1055)
!1112 = !DILocation(line: 393, column: 6, scope: !1055)
!1113 = !DILocation(line: 394, column: 20, scope: !1055)
!1114 = !DILocation(line: 394, column: 18, scope: !1055)
!1115 = !DILocation(line: 394, column: 13, scope: !1055)
!1116 = !DILocation(line: 394, column: 8, scope: !1055)
!1117 = !DILocation(line: 394, column: 6, scope: !1055)
!1118 = !DILocation(line: 395, column: 7, scope: !1055)
!1119 = !DILocation(line: 395, column: 18, scope: !1055)
!1120 = !DILocation(line: 395, column: 16, scope: !1055)
!1121 = !DILocation(line: 395, column: 10, scope: !1055)
!1122 = !DILocation(line: 395, column: 5, scope: !1055)
!1123 = !DILocation(line: 396, column: 14, scope: !1055)
!1124 = !DILocation(line: 396, column: 12, scope: !1055)
!1125 = !DILocation(line: 396, column: 18, scope: !1055)
!1126 = !DILocation(line: 396, column: 23, scope: !1055)
!1127 = !DILocation(line: 396, column: 21, scope: !1055)
!1128 = !DILocation(line: 396, column: 16, scope: !1055)
!1129 = !DILocation(line: 396, column: 6, scope: !1055)
!1130 = !DILocation(line: 397, column: 20, scope: !1055)
!1131 = !DILocation(line: 397, column: 18, scope: !1055)
!1132 = !DILocation(line: 397, column: 13, scope: !1055)
!1133 = !DILocation(line: 397, column: 8, scope: !1055)
!1134 = !DILocation(line: 397, column: 6, scope: !1055)
!1135 = !DILocation(line: 398, column: 10, scope: !1055)
!1136 = !DILocation(line: 398, column: 21, scope: !1055)
!1137 = !DILocation(line: 398, column: 19, scope: !1055)
!1138 = !DILocation(line: 398, column: 13, scope: !1055)
!1139 = !DILocation(line: 398, column: 5, scope: !1055)
!1140 = !DILocation(line: 398, column: 8, scope: !1055)
!1141 = !DILocation(line: 400, column: 19, scope: !1055)
!1142 = !DILocation(line: 400, column: 18, scope: !1055)
!1143 = !DILocation(line: 400, column: 15, scope: !1055)
!1144 = !DILocation(line: 400, column: 3, scope: !1055)
!1145 = distinct !DISubprogram(name: "c_print_results", linkageName: "_Z15c_print_resultsPcciiiiddS_iS_S_S_S_S_S_S_S_S_S_S_S_S_S_", scope: !3, file: !3, line: 406, type: !1146, scopeLine: 429, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !1058)
!1146 = !DISubroutineType(types: !1147)
!1147 = !{null, !108, !109, !97, !97, !97, !97, !104, !104, !108, !97, !108, !108, !108, !108, !108, !108, !108, !108, !108, !108, !108, !108, !108, !108}
!1148 = !DILocalVariable(name: "name", arg: 1, scope: !1145, file: !3, line: 406, type: !108)
!1149 = !DILocation(line: 406, column: 29, scope: !1145)
!1150 = !DILocalVariable(name: "class_npb", arg: 2, scope: !1145, file: !3, line: 407, type: !109)
!1151 = !DILocation(line: 407, column: 9, scope: !1145)
!1152 = !DILocalVariable(name: "n1", arg: 3, scope: !1145, file: !3, line: 408, type: !97)
!1153 = !DILocation(line: 408, column: 8, scope: !1145)
!1154 = !DILocalVariable(name: "n2", arg: 4, scope: !1145, file: !3, line: 409, type: !97)
!1155 = !DILocation(line: 409, column: 8, scope: !1145)
!1156 = !DILocalVariable(name: "n3", arg: 5, scope: !1145, file: !3, line: 410, type: !97)
!1157 = !DILocation(line: 410, column: 8, scope: !1145)
!1158 = !DILocalVariable(name: "niter", arg: 6, scope: !1145, file: !3, line: 411, type: !97)
!1159 = !DILocation(line: 411, column: 8, scope: !1145)
!1160 = !DILocalVariable(name: "t", arg: 7, scope: !1145, file: !3, line: 412, type: !104)
!1161 = !DILocation(line: 412, column: 11, scope: !1145)
!1162 = !DILocalVariable(name: "mops", arg: 8, scope: !1145, file: !3, line: 413, type: !104)
!1163 = !DILocation(line: 413, column: 11, scope: !1145)
!1164 = !DILocalVariable(name: "optype", arg: 9, scope: !1145, file: !3, line: 414, type: !108)
!1165 = !DILocation(line: 414, column: 10, scope: !1145)
!1166 = !DILocalVariable(name: "passed_verification", arg: 10, scope: !1145, file: !3, line: 415, type: !97)
!1167 = !DILocation(line: 415, column: 8, scope: !1145)
!1168 = !DILocalVariable(name: "npbversion", arg: 11, scope: !1145, file: !3, line: 416, type: !108)
!1169 = !DILocation(line: 416, column: 10, scope: !1145)
!1170 = !DILocalVariable(name: "compiletime", arg: 12, scope: !1145, file: !3, line: 417, type: !108)
!1171 = !DILocation(line: 417, column: 10, scope: !1145)
!1172 = !DILocalVariable(name: "compilerversion", arg: 13, scope: !1145, file: !3, line: 418, type: !108)
!1173 = !DILocation(line: 418, column: 10, scope: !1145)
!1174 = !DILocalVariable(name: "libversion", arg: 14, scope: !1145, file: !3, line: 419, type: !108)
!1175 = !DILocation(line: 419, column: 10, scope: !1145)
!1176 = !DILocalVariable(name: "cpu_device", arg: 15, scope: !1145, file: !3, line: 420, type: !108)
!1177 = !DILocation(line: 420, column: 10, scope: !1145)
!1178 = !DILocalVariable(name: "gpu_device", arg: 16, scope: !1145, file: !3, line: 421, type: !108)
!1179 = !DILocation(line: 421, column: 10, scope: !1145)
!1180 = !DILocalVariable(name: "gpu_config", arg: 17, scope: !1145, file: !3, line: 422, type: !108)
!1181 = !DILocation(line: 422, column: 10, scope: !1145)
!1182 = !DILocalVariable(name: "cc", arg: 18, scope: !1145, file: !3, line: 423, type: !108)
!1183 = !DILocation(line: 423, column: 10, scope: !1145)
!1184 = !DILocalVariable(name: "clink", arg: 19, scope: !1145, file: !3, line: 424, type: !108)
!1185 = !DILocation(line: 424, column: 10, scope: !1145)
!1186 = !DILocalVariable(name: "c_lib", arg: 20, scope: !1145, file: !3, line: 425, type: !108)
!1187 = !DILocation(line: 425, column: 10, scope: !1145)
!1188 = !DILocalVariable(name: "c_inc", arg: 21, scope: !1145, file: !3, line: 426, type: !108)
!1189 = !DILocation(line: 426, column: 10, scope: !1145)
!1190 = !DILocalVariable(name: "cflags", arg: 22, scope: !1145, file: !3, line: 427, type: !108)
!1191 = !DILocation(line: 427, column: 10, scope: !1145)
!1192 = !DILocalVariable(name: "clinkflags", arg: 23, scope: !1145, file: !3, line: 428, type: !108)
!1193 = !DILocation(line: 428, column: 10, scope: !1145)
!1194 = !DILocalVariable(name: "rand", arg: 24, scope: !1145, file: !3, line: 429, type: !108)
!1195 = !DILocation(line: 429, column: 10, scope: !1145)
!1196 = !DILocation(line: 430, column: 45, scope: !1145)
!1197 = !DILocation(line: 430, column: 5, scope: !1145)
!1198 = !DILocation(line: 431, column: 62, scope: !1145)
!1199 = !DILocation(line: 431, column: 5, scope: !1145)
!1200 = !DILocation(line: 432, column: 9, scope: !1201)
!1201 = distinct !DILexicalBlock(scope: !1145, file: !3, line: 432, column: 8)
!1202 = !DILocation(line: 432, column: 16, scope: !1201)
!1203 = !DILocation(line: 432, column: 22, scope: !1201)
!1204 = !DILocation(line: 432, column: 25, scope: !1201)
!1205 = !DILocation(line: 432, column: 32, scope: !1201)
!1206 = !DILocation(line: 432, column: 8, scope: !1145)
!1207 = !DILocation(line: 433, column: 9, scope: !1208)
!1208 = distinct !DILexicalBlock(scope: !1209, file: !3, line: 433, column: 9)
!1209 = distinct !DILexicalBlock(scope: !1201, file: !3, line: 432, column: 39)
!1210 = !DILocation(line: 433, column: 11, scope: !1208)
!1211 = !DILocation(line: 433, column: 9, scope: !1209)
!1212 = !DILocalVariable(name: "nn", scope: !1213, file: !3, line: 434, type: !402)
!1213 = distinct !DILexicalBlock(scope: !1208, file: !3, line: 433, column: 15)
!1214 = !DILocation(line: 434, column: 12, scope: !1213)
!1215 = !DILocation(line: 434, column: 17, scope: !1213)
!1216 = !DILocation(line: 435, column: 10, scope: !1217)
!1217 = distinct !DILexicalBlock(scope: !1213, file: !3, line: 435, column: 10)
!1218 = !DILocation(line: 435, column: 12, scope: !1217)
!1219 = !DILocation(line: 435, column: 10, scope: !1213)
!1220 = !DILocation(line: 435, column: 21, scope: !1221)
!1221 = distinct !DILexicalBlock(scope: !1217, file: !3, line: 435, column: 16)
!1222 = !DILocation(line: 435, column: 19, scope: !1221)
!1223 = !DILocation(line: 435, column: 24, scope: !1221)
!1224 = !DILocation(line: 436, column: 56, scope: !1213)
!1225 = !DILocation(line: 436, column: 7, scope: !1213)
!1226 = !DILocation(line: 437, column: 6, scope: !1213)
!1227 = !DILocation(line: 438, column: 62, scope: !1228)
!1228 = distinct !DILexicalBlock(scope: !1208, file: !3, line: 437, column: 11)
!1229 = !DILocation(line: 438, column: 65, scope: !1228)
!1230 = !DILocation(line: 438, column: 68, scope: !1228)
!1231 = !DILocation(line: 438, column: 7, scope: !1228)
!1232 = !DILocation(line: 440, column: 5, scope: !1209)
!1233 = !DILocalVariable(name: "size", scope: !1234, file: !3, line: 441, type: !1235)
!1234 = distinct !DILexicalBlock(scope: !1201, file: !3, line: 440, column: 10)
!1235 = !DICompositeType(tag: DW_TAG_array_type, baseType: !109, size: 128, elements: !1236)
!1236 = !{!1237}
!1237 = !DISubrange(count: 16)
!1238 = !DILocation(line: 441, column: 11, scope: !1234)
!1239 = !DILocalVariable(name: "j", scope: !1234, file: !3, line: 442, type: !97)
!1240 = !DILocation(line: 442, column: 10, scope: !1234)
!1241 = !DILocation(line: 443, column: 10, scope: !1242)
!1242 = distinct !DILexicalBlock(scope: !1234, file: !3, line: 443, column: 9)
!1243 = !DILocation(line: 443, column: 12, scope: !1242)
!1244 = !DILocation(line: 443, column: 17, scope: !1242)
!1245 = !DILocation(line: 443, column: 21, scope: !1242)
!1246 = !DILocation(line: 443, column: 23, scope: !1242)
!1247 = !DILocation(line: 443, column: 9, scope: !1234)
!1248 = !DILocation(line: 444, column: 11, scope: !1249)
!1249 = distinct !DILexicalBlock(scope: !1250, file: !3, line: 444, column: 10)
!1250 = distinct !DILexicalBlock(scope: !1242, file: !3, line: 443, column: 28)
!1251 = !DILocation(line: 444, column: 18, scope: !1249)
!1252 = !DILocation(line: 444, column: 24, scope: !1249)
!1253 = !DILocation(line: 444, column: 27, scope: !1249)
!1254 = !DILocation(line: 444, column: 34, scope: !1249)
!1255 = !DILocation(line: 444, column: 10, scope: !1250)
!1256 = !DILocation(line: 445, column: 16, scope: !1257)
!1257 = distinct !DILexicalBlock(scope: !1249, file: !3, line: 444, column: 41)
!1258 = !DILocation(line: 445, column: 42, scope: !1257)
!1259 = !DILocation(line: 445, column: 33, scope: !1257)
!1260 = !DILocation(line: 445, column: 8, scope: !1257)
!1261 = !DILocation(line: 446, column: 10, scope: !1257)
!1262 = !DILocation(line: 447, column: 16, scope: !1263)
!1263 = distinct !DILexicalBlock(scope: !1257, file: !3, line: 447, column: 11)
!1264 = !DILocation(line: 447, column: 11, scope: !1263)
!1265 = !DILocation(line: 447, column: 19, scope: !1263)
!1266 = !DILocation(line: 447, column: 11, scope: !1257)
!1267 = !DILocation(line: 448, column: 14, scope: !1268)
!1268 = distinct !DILexicalBlock(scope: !1263, file: !3, line: 447, column: 26)
!1269 = !DILocation(line: 448, column: 9, scope: !1268)
!1270 = !DILocation(line: 448, column: 17, scope: !1268)
!1271 = !DILocation(line: 449, column: 10, scope: !1268)
!1272 = !DILocation(line: 450, column: 8, scope: !1268)
!1273 = !DILocation(line: 451, column: 13, scope: !1257)
!1274 = !DILocation(line: 451, column: 14, scope: !1257)
!1275 = !DILocation(line: 451, column: 8, scope: !1257)
!1276 = !DILocation(line: 451, column: 18, scope: !1257)
!1277 = !DILocation(line: 452, column: 53, scope: !1257)
!1278 = !DILocation(line: 452, column: 8, scope: !1257)
!1279 = !DILocation(line: 453, column: 7, scope: !1257)
!1280 = !DILocation(line: 454, column: 56, scope: !1281)
!1281 = distinct !DILexicalBlock(scope: !1249, file: !3, line: 453, column: 12)
!1282 = !DILocation(line: 454, column: 8, scope: !1281)
!1283 = !DILocation(line: 456, column: 6, scope: !1250)
!1284 = !DILocation(line: 457, column: 60, scope: !1285)
!1285 = distinct !DILexicalBlock(scope: !1242, file: !3, line: 456, column: 11)
!1286 = !DILocation(line: 457, column: 64, scope: !1285)
!1287 = !DILocation(line: 457, column: 68, scope: !1285)
!1288 = !DILocation(line: 457, column: 7, scope: !1285)
!1289 = !DILocation(line: 460, column: 53, scope: !1145)
!1290 = !DILocation(line: 460, column: 5, scope: !1145)
!1291 = !DILocation(line: 461, column: 55, scope: !1145)
!1292 = !DILocation(line: 461, column: 5, scope: !1145)
!1293 = !DILocation(line: 462, column: 55, scope: !1145)
!1294 = !DILocation(line: 462, column: 5, scope: !1145)
!1295 = !DILocation(line: 463, column: 41, scope: !1145)
!1296 = !DILocation(line: 463, column: 5, scope: !1145)
!1297 = !DILocation(line: 464, column: 8, scope: !1298)
!1298 = distinct !DILexicalBlock(scope: !1145, file: !3, line: 464, column: 8)
!1299 = !DILocation(line: 464, column: 28, scope: !1298)
!1300 = !DILocation(line: 464, column: 8, scope: !1145)
!1301 = !DILocation(line: 465, column: 6, scope: !1302)
!1302 = distinct !DILexicalBlock(scope: !1298, file: !3, line: 464, column: 32)
!1303 = !DILocation(line: 466, column: 5, scope: !1302)
!1304 = !DILocation(line: 466, column: 14, scope: !1305)
!1305 = distinct !DILexicalBlock(scope: !1298, file: !3, line: 466, column: 14)
!1306 = !DILocation(line: 466, column: 14, scope: !1298)
!1307 = !DILocation(line: 467, column: 6, scope: !1308)
!1308 = distinct !DILexicalBlock(scope: !1305, file: !3, line: 466, column: 34)
!1309 = !DILocation(line: 468, column: 5, scope: !1308)
!1310 = !DILocation(line: 469, column: 6, scope: !1311)
!1311 = distinct !DILexicalBlock(scope: !1305, file: !3, line: 468, column: 10)
!1312 = !DILocation(line: 471, column: 53, scope: !1145)
!1313 = !DILocation(line: 471, column: 5, scope: !1145)
!1314 = !DILocation(line: 472, column: 53, scope: !1145)
!1315 = !DILocation(line: 472, column: 5, scope: !1145)
!1316 = !DILocation(line: 473, column: 53, scope: !1145)
!1317 = !DILocation(line: 473, column: 5, scope: !1145)
!1318 = !DILocation(line: 474, column: 53, scope: !1145)
!1319 = !DILocation(line: 474, column: 5, scope: !1145)
!1320 = !DILocation(line: 475, column: 5, scope: !1145)
!1321 = !DILocation(line: 476, column: 39, scope: !1145)
!1322 = !DILocation(line: 476, column: 5, scope: !1145)
!1323 = !DILocation(line: 477, column: 39, scope: !1145)
!1324 = !DILocation(line: 477, column: 5, scope: !1145)
!1325 = !DILocation(line: 478, column: 39, scope: !1145)
!1326 = !DILocation(line: 478, column: 5, scope: !1145)
!1327 = !DILocation(line: 479, column: 39, scope: !1145)
!1328 = !DILocation(line: 479, column: 5, scope: !1145)
!1329 = !DILocation(line: 480, column: 39, scope: !1145)
!1330 = !DILocation(line: 480, column: 5, scope: !1145)
!1331 = !DILocation(line: 481, column: 39, scope: !1145)
!1332 = !DILocation(line: 481, column: 5, scope: !1145)
!1333 = !DILocation(line: 482, column: 39, scope: !1145)
!1334 = !DILocation(line: 482, column: 5, scope: !1145)
!1335 = !DILocation(line: 483, column: 5, scope: !1145)
!1336 = !DILocation(line: 484, column: 39, scope: !1145)
!1337 = !DILocation(line: 484, column: 5, scope: !1145)
!1338 = !DILocation(line: 485, column: 39, scope: !1145)
!1339 = !DILocation(line: 485, column: 5, scope: !1145)
!1340 = !DILocation(line: 486, column: 5, scope: !1145)
!1341 = !DILocation(line: 487, column: 39, scope: !1145)
!1342 = !DILocation(line: 487, column: 5, scope: !1145)
!1343 = !DILocation(line: 502, column: 5, scope: !1145)
!1344 = !DILocation(line: 503, column: 5, scope: !1145)
!1345 = !DILocation(line: 504, column: 5, scope: !1145)
!1346 = !DILocation(line: 505, column: 5, scope: !1145)
!1347 = !DILocation(line: 506, column: 5, scope: !1145)
!1348 = !DILocation(line: 507, column: 5, scope: !1145)
!1349 = !DILocation(line: 508, column: 5, scope: !1145)
!1350 = !DILocation(line: 509, column: 5, scope: !1145)
!1351 = !DILocation(line: 510, column: 5, scope: !1145)
!1352 = !DILocation(line: 511, column: 5, scope: !1145)
!1353 = !DILocation(line: 512, column: 4, scope: !1145)
!1354 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 516, type: !1355, scopeLine: 516, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !1058)
!1355 = !DISubroutineType(types: !1356)
!1356 = !{!97, !97, !655}
!1357 = !DILocalVariable(name: "argc", arg: 1, scope: !1354, file: !3, line: 516, type: !97)
!1358 = !DILocation(line: 516, column: 14, scope: !1354)
!1359 = !DILocalVariable(name: "argv", arg: 2, scope: !1354, file: !3, line: 516, type: !655)
!1360 = !DILocation(line: 516, column: 27, scope: !1354)
!1361 = !DILocalVariable(name: "iter", scope: !1354, file: !3, line: 523, type: !97)
!1362 = !DILocation(line: 523, column: 6, scope: !1354)
!1363 = !DILocalVariable(name: "total_time", scope: !1354, file: !3, line: 524, type: !104)
!1364 = !DILocation(line: 524, column: 9, scope: !1354)
!1365 = !DILocalVariable(name: "mflops", scope: !1354, file: !3, line: 524, type: !104)
!1366 = !DILocation(line: 524, column: 21, scope: !1354)
!1367 = !DILocalVariable(name: "verified", scope: !1354, file: !3, line: 525, type: !1368)
!1368 = !DIDerivedType(tag: DW_TAG_typedef, name: "boolean", file: !100, line: 80, baseType: !97)
!1369 = !DILocation(line: 525, column: 10, scope: !1354)
!1370 = !DILocalVariable(name: "class_npb", scope: !1354, file: !3, line: 526, type: !109)
!1371 = !DILocation(line: 526, column: 7, scope: !1354)
!1372 = !DILocation(line: 529, column: 20, scope: !1354)
!1373 = !DILocation(line: 529, column: 9, scope: !1354)
!1374 = !DILocation(line: 529, column: 7, scope: !1354)
!1375 = !DILocation(line: 530, column: 21, scope: !1354)
!1376 = !DILocation(line: 530, column: 12, scope: !1354)
!1377 = !DILocation(line: 530, column: 10, scope: !1354)
!1378 = !DILocation(line: 531, column: 17, scope: !1354)
!1379 = !DILocation(line: 531, column: 6, scope: !1354)
!1380 = !DILocation(line: 531, column: 4, scope: !1354)
!1381 = !DILocation(line: 532, column: 18, scope: !1354)
!1382 = !DILocation(line: 532, column: 7, scope: !1354)
!1383 = !DILocation(line: 532, column: 5, scope: !1354)
!1384 = !DILocation(line: 533, column: 18, scope: !1354)
!1385 = !DILocation(line: 533, column: 7, scope: !1354)
!1386 = !DILocation(line: 533, column: 5, scope: !1354)
!1387 = !DILocation(line: 534, column: 15, scope: !1354)
!1388 = !DILocation(line: 534, column: 9, scope: !1354)
!1389 = !DILocation(line: 534, column: 7, scope: !1354)
!1390 = !DILocation(line: 544, column: 2, scope: !1354)
!1391 = !DILocation(line: 545, column: 2, scope: !1354)
!1392 = !DILocation(line: 546, column: 14, scope: !1354)
!1393 = !DILocation(line: 546, column: 25, scope: !1354)
!1394 = !DILocation(line: 546, column: 36, scope: !1354)
!1395 = !DILocation(line: 546, column: 2, scope: !1354)
!1396 = !DILocation(line: 547, column: 23, scope: !1354)
!1397 = !DILocation(line: 547, column: 2, scope: !1354)
!1398 = !DILocation(line: 548, column: 33, scope: !1354)
!1399 = !DILocation(line: 548, column: 2, scope: !1354)
!1400 = !DILocation(line: 549, column: 2, scope: !1354)
!1401 = !DILocation(line: 550, column: 2, scope: !1354)
!1402 = !DILocation(line: 551, column: 13, scope: !1354)
!1403 = !DILocation(line: 551, column: 24, scope: !1354)
!1404 = !DILocation(line: 551, column: 2, scope: !1354)
!1405 = !DILocation(line: 578, column: 23, scope: !1354)
!1406 = !DILocation(line: 578, column: 2, scope: !1354)
!1407 = !DILocation(line: 579, column: 33, scope: !1354)
!1408 = !DILocation(line: 579, column: 2, scope: !1354)
!1409 = !DILocation(line: 580, column: 2, scope: !1354)
!1410 = !DILocation(line: 581, column: 2, scope: !1354)
!1411 = !DILocation(line: 582, column: 13, scope: !1354)
!1412 = !DILocation(line: 582, column: 24, scope: !1354)
!1413 = !DILocation(line: 582, column: 2, scope: !1354)
!1414 = !DILocation(line: 583, column: 10, scope: !1415)
!1415 = distinct !DILexicalBlock(scope: !1354, file: !3, line: 583, column: 2)
!1416 = !DILocation(line: 583, column: 6, scope: !1415)
!1417 = !DILocation(line: 583, column: 14, scope: !1418)
!1418 = distinct !DILexicalBlock(scope: !1415, file: !3, line: 583, column: 2)
!1419 = !DILocation(line: 583, column: 20, scope: !1418)
!1420 = !DILocation(line: 583, column: 18, scope: !1418)
!1421 = !DILocation(line: 583, column: 2, scope: !1415)
!1422 = !DILocation(line: 584, column: 14, scope: !1423)
!1423 = distinct !DILexicalBlock(scope: !1418, file: !3, line: 583, column: 34)
!1424 = !DILocation(line: 584, column: 25, scope: !1423)
!1425 = !DILocation(line: 584, column: 36, scope: !1423)
!1426 = !DILocation(line: 584, column: 3, scope: !1423)
!1427 = !DILocation(line: 585, column: 15, scope: !1423)
!1428 = !DILocation(line: 585, column: 26, scope: !1423)
!1429 = !DILocation(line: 585, column: 3, scope: !1423)
!1430 = !DILocation(line: 586, column: 16, scope: !1423)
!1431 = !DILocation(line: 586, column: 22, scope: !1423)
!1432 = !DILocation(line: 586, column: 3, scope: !1423)
!1433 = !DILocation(line: 587, column: 2, scope: !1423)
!1434 = !DILocation(line: 583, column: 31, scope: !1418)
!1435 = !DILocation(line: 583, column: 2, scope: !1418)
!1436 = distinct !{!1436, !1421, !1437}
!1437 = !DILocation(line: 587, column: 2, scope: !1415)
!1438 = !DILocation(line: 589, column: 13, scope: !1354)
!1439 = !DILocation(line: 589, column: 19, scope: !1354)
!1440 = !DILocation(line: 589, column: 32, scope: !1354)
!1441 = !DILocation(line: 589, column: 2, scope: !1354)
!1442 = !DILocation(line: 590, column: 10, scope: !1443)
!1443 = distinct !DILexicalBlock(scope: !1354, file: !3, line: 590, column: 2)
!1444 = !DILocation(line: 590, column: 6, scope: !1443)
!1445 = !DILocation(line: 590, column: 14, scope: !1446)
!1446 = distinct !DILexicalBlock(scope: !1443, file: !3, line: 590, column: 2)
!1447 = !DILocation(line: 590, column: 20, scope: !1446)
!1448 = !DILocation(line: 590, column: 18, scope: !1446)
!1449 = !DILocation(line: 590, column: 2, scope: !1443)
!1450 = !DILocation(line: 591, column: 54, scope: !1451)
!1451 = distinct !DILexicalBlock(scope: !1446, file: !3, line: 590, column: 34)
!1452 = !DILocation(line: 591, column: 60, scope: !1451)
!1453 = !DILocation(line: 591, column: 65, scope: !1451)
!1454 = !DILocation(line: 591, column: 71, scope: !1451)
!1455 = !DILocation(line: 591, column: 77, scope: !1451)
!1456 = !DILocation(line: 591, column: 82, scope: !1451)
!1457 = !DILocation(line: 591, column: 88, scope: !1451)
!1458 = !DILocation(line: 591, column: 3, scope: !1451)
!1459 = !DILocation(line: 592, column: 2, scope: !1451)
!1460 = !DILocation(line: 590, column: 31, scope: !1446)
!1461 = !DILocation(line: 590, column: 2, scope: !1446)
!1462 = distinct !{!1462, !1449, !1463}
!1463 = !DILocation(line: 592, column: 2, scope: !1443)
!1464 = !DILocation(line: 594, column: 21, scope: !1354)
!1465 = !DILocation(line: 594, column: 2, scope: !1354)
!1466 = !DILocation(line: 598, column: 13, scope: !1354)
!1467 = !DILocation(line: 600, column: 5, scope: !1468)
!1468 = distinct !DILexicalBlock(scope: !1354, file: !3, line: 600, column: 5)
!1469 = !DILocation(line: 600, column: 16, scope: !1468)
!1470 = !DILocation(line: 600, column: 5, scope: !1354)
!1471 = !DILocation(line: 602, column: 25, scope: !1472)
!1472 = distinct !DILexicalBlock(scope: !1468, file: !3, line: 600, column: 23)
!1473 = !DILocation(line: 602, column: 23, scope: !1472)
!1474 = !DILocation(line: 602, column: 13, scope: !1472)
!1475 = !DILocation(line: 603, column: 28, scope: !1472)
!1476 = !DILocation(line: 603, column: 26, scope: !1472)
!1477 = !DILocation(line: 603, column: 16, scope: !1472)
!1478 = !DILocation(line: 603, column: 51, scope: !1472)
!1479 = !DILocation(line: 603, column: 50, scope: !1472)
!1480 = !DILocation(line: 603, column: 5, scope: !1472)
!1481 = !DILocation(line: 601, column: 40, scope: !1472)
!1482 = !DILocation(line: 604, column: 6, scope: !1472)
!1483 = !DILocation(line: 604, column: 4, scope: !1472)
!1484 = !DILocation(line: 601, column: 10, scope: !1472)
!1485 = !DILocation(line: 605, column: 2, scope: !1472)
!1486 = !DILocation(line: 606, column: 10, scope: !1487)
!1487 = distinct !DILexicalBlock(scope: !1468, file: !3, line: 605, column: 7)
!1488 = !DILocalVariable(name: "gpu_config", scope: !1354, file: !3, line: 609, type: !215)
!1489 = !DILocation(line: 609, column: 7, scope: !1354)
!1490 = !DILocalVariable(name: "gpu_config_string", scope: !1354, file: !3, line: 610, type: !1491)
!1491 = !DICompositeType(tag: DW_TAG_array_type, baseType: !109, size: 16384, elements: !1492)
!1492 = !{!1493}
!1493 = !DISubrange(count: 2048)
!1494 = !DILocation(line: 610, column: 7, scope: !1354)
!1495 = !DILocation(line: 643, column: 10, scope: !1354)
!1496 = !DILocation(line: 643, column: 2, scope: !1354)
!1497 = !DILocation(line: 644, column: 9, scope: !1354)
!1498 = !DILocation(line: 644, column: 28, scope: !1354)
!1499 = !DILocation(line: 644, column: 2, scope: !1354)
!1500 = !DILocation(line: 645, column: 10, scope: !1354)
!1501 = !DILocation(line: 645, column: 51, scope: !1354)
!1502 = !DILocation(line: 645, column: 2, scope: !1354)
!1503 = !DILocation(line: 646, column: 9, scope: !1354)
!1504 = !DILocation(line: 646, column: 28, scope: !1354)
!1505 = !DILocation(line: 646, column: 2, scope: !1354)
!1506 = !DILocation(line: 647, column: 10, scope: !1354)
!1507 = !DILocation(line: 647, column: 61, scope: !1354)
!1508 = !DILocation(line: 647, column: 2, scope: !1354)
!1509 = !DILocation(line: 648, column: 9, scope: !1354)
!1510 = !DILocation(line: 648, column: 28, scope: !1354)
!1511 = !DILocation(line: 648, column: 2, scope: !1354)
!1512 = !DILocation(line: 649, column: 10, scope: !1354)
!1513 = !DILocation(line: 649, column: 50, scope: !1354)
!1514 = !DILocation(line: 649, column: 2, scope: !1354)
!1515 = !DILocation(line: 650, column: 9, scope: !1354)
!1516 = !DILocation(line: 650, column: 28, scope: !1354)
!1517 = !DILocation(line: 650, column: 2, scope: !1354)
!1518 = !DILocation(line: 651, column: 10, scope: !1354)
!1519 = !DILocation(line: 651, column: 49, scope: !1354)
!1520 = !DILocation(line: 651, column: 2, scope: !1354)
!1521 = !DILocation(line: 652, column: 9, scope: !1354)
!1522 = !DILocation(line: 652, column: 28, scope: !1354)
!1523 = !DILocation(line: 652, column: 2, scope: !1354)
!1524 = !DILocation(line: 653, column: 10, scope: !1354)
!1525 = !DILocation(line: 653, column: 49, scope: !1354)
!1526 = !DILocation(line: 653, column: 2, scope: !1354)
!1527 = !DILocation(line: 654, column: 9, scope: !1354)
!1528 = !DILocation(line: 654, column: 28, scope: !1354)
!1529 = !DILocation(line: 654, column: 2, scope: !1354)
!1530 = !DILocation(line: 655, column: 10, scope: !1354)
!1531 = !DILocation(line: 655, column: 49, scope: !1354)
!1532 = !DILocation(line: 655, column: 2, scope: !1354)
!1533 = !DILocation(line: 656, column: 9, scope: !1354)
!1534 = !DILocation(line: 656, column: 28, scope: !1354)
!1535 = !DILocation(line: 656, column: 2, scope: !1354)
!1536 = !DILocation(line: 657, column: 10, scope: !1354)
!1537 = !DILocation(line: 657, column: 49, scope: !1354)
!1538 = !DILocation(line: 657, column: 2, scope: !1354)
!1539 = !DILocation(line: 658, column: 9, scope: !1354)
!1540 = !DILocation(line: 658, column: 28, scope: !1354)
!1541 = !DILocation(line: 658, column: 2, scope: !1354)
!1542 = !DILocation(line: 659, column: 10, scope: !1354)
!1543 = !DILocation(line: 659, column: 49, scope: !1354)
!1544 = !DILocation(line: 659, column: 2, scope: !1354)
!1545 = !DILocation(line: 660, column: 9, scope: !1354)
!1546 = !DILocation(line: 660, column: 28, scope: !1354)
!1547 = !DILocation(line: 660, column: 2, scope: !1354)
!1548 = !DILocation(line: 661, column: 10, scope: !1354)
!1549 = !DILocation(line: 661, column: 49, scope: !1354)
!1550 = !DILocation(line: 661, column: 2, scope: !1354)
!1551 = !DILocation(line: 662, column: 9, scope: !1354)
!1552 = !DILocation(line: 662, column: 28, scope: !1354)
!1553 = !DILocation(line: 662, column: 2, scope: !1354)
!1554 = !DILocation(line: 663, column: 10, scope: !1354)
!1555 = !DILocation(line: 663, column: 49, scope: !1354)
!1556 = !DILocation(line: 663, column: 2, scope: !1354)
!1557 = !DILocation(line: 664, column: 9, scope: !1354)
!1558 = !DILocation(line: 664, column: 28, scope: !1354)
!1559 = !DILocation(line: 664, column: 2, scope: !1354)
!1560 = !DILocation(line: 665, column: 10, scope: !1354)
!1561 = !DILocation(line: 665, column: 49, scope: !1354)
!1562 = !DILocation(line: 665, column: 2, scope: !1354)
!1563 = !DILocation(line: 666, column: 9, scope: !1354)
!1564 = !DILocation(line: 666, column: 28, scope: !1354)
!1565 = !DILocation(line: 666, column: 2, scope: !1354)
!1566 = !DILocation(line: 667, column: 10, scope: !1354)
!1567 = !DILocation(line: 667, column: 49, scope: !1354)
!1568 = !DILocation(line: 667, column: 2, scope: !1354)
!1569 = !DILocation(line: 668, column: 9, scope: !1354)
!1570 = !DILocation(line: 668, column: 28, scope: !1354)
!1571 = !DILocation(line: 668, column: 2, scope: !1354)
!1572 = !DILocation(line: 669, column: 10, scope: !1354)
!1573 = !DILocation(line: 669, column: 49, scope: !1354)
!1574 = !DILocation(line: 669, column: 2, scope: !1354)
!1575 = !DILocation(line: 670, column: 9, scope: !1354)
!1576 = !DILocation(line: 670, column: 28, scope: !1354)
!1577 = !DILocation(line: 670, column: 2, scope: !1354)
!1578 = !DILocation(line: 671, column: 10, scope: !1354)
!1579 = !DILocation(line: 671, column: 51, scope: !1354)
!1580 = !DILocation(line: 671, column: 2, scope: !1354)
!1581 = !DILocation(line: 672, column: 9, scope: !1354)
!1582 = !DILocation(line: 672, column: 28, scope: !1354)
!1583 = !DILocation(line: 672, column: 2, scope: !1354)
!1584 = !DILocation(line: 676, column: 4, scope: !1354)
!1585 = !DILocation(line: 680, column: 4, scope: !1354)
!1586 = !DILocation(line: 681, column: 4, scope: !1354)
!1587 = !DILocation(line: 682, column: 4, scope: !1354)
!1588 = !DILocation(line: 684, column: 4, scope: !1354)
!1589 = !DILocation(line: 691, column: 11, scope: !1354)
!1590 = !DILocation(line: 675, column: 2, scope: !1354)
!1591 = !DILocation(line: 700, column: 2, scope: !1354)
!1592 = !DILocation(line: 702, column: 7, scope: !1354)
!1593 = !DILocation(line: 702, column: 2, scope: !1354)
!1594 = !DILocation(line: 703, column: 7, scope: !1354)
!1595 = !DILocation(line: 703, column: 2, scope: !1354)
!1596 = !DILocation(line: 704, column: 7, scope: !1354)
!1597 = !DILocation(line: 704, column: 2, scope: !1354)
!1598 = !DILocation(line: 705, column: 7, scope: !1354)
!1599 = !DILocation(line: 705, column: 2, scope: !1354)
!1600 = !DILocation(line: 706, column: 7, scope: !1354)
!1601 = !DILocation(line: 706, column: 2, scope: !1354)
!1602 = !DILocation(line: 707, column: 7, scope: !1354)
!1603 = !DILocation(line: 707, column: 2, scope: !1354)
!1604 = !DILocation(line: 710, column: 2, scope: !1354)
!1605 = distinct !DISubprogram(name: "setup", linkageName: "_ZL5setupv", scope: !3, file: !3, line: 1647, type: !561, scopeLine: 1647, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1058)
!1606 = !DILocation(line: 1648, column: 8, scope: !1605)
!1607 = !DILocation(line: 1650, column: 2, scope: !1605)
!1608 = !DILocation(line: 1651, column: 2, scope: !1605)
!1609 = !DILocation(line: 1652, column: 48, scope: !1605)
!1610 = !DILocation(line: 1652, column: 2, scope: !1605)
!1611 = !DILocation(line: 1653, column: 2, scope: !1605)
!1612 = !DILocation(line: 1654, column: 1, scope: !1605)
!1613 = distinct !DISubprogram(name: "setup_gpu", linkageName: "_ZL9setup_gpuv", scope: !3, file: !3, line: 1656, type: !561, scopeLine: 1656, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1058)
!1614 = !DILocation(line: 1701, column: 33, scope: !1613)
!1615 = !DILocation(line: 1702, column: 43, scope: !1613)
!1616 = !DILocation(line: 1705, column: 69, scope: !1617)
!1617 = distinct !DILexicalBlock(scope: !1613, file: !3, line: 1704, column: 5)
!1618 = !DILocation(line: 1705, column: 45, scope: !1617)
!1619 = !DILocation(line: 1704, column: 5, scope: !1613)
!1620 = !DILocation(line: 1706, column: 41, scope: !1621)
!1621 = distinct !DILexicalBlock(scope: !1617, file: !3, line: 1705, column: 89)
!1622 = !DILocation(line: 1707, column: 2, scope: !1621)
!1623 = !DILocation(line: 1708, column: 65, scope: !1624)
!1624 = distinct !DILexicalBlock(scope: !1617, file: !3, line: 1707, column: 7)
!1625 = !DILocation(line: 1708, column: 41, scope: !1624)
!1626 = !DILocation(line: 1711, column: 79, scope: !1627)
!1627 = distinct !DILexicalBlock(scope: !1613, file: !3, line: 1710, column: 5)
!1628 = !DILocation(line: 1711, column: 55, scope: !1627)
!1629 = !DILocation(line: 1710, column: 5, scope: !1613)
!1630 = !DILocation(line: 1712, column: 51, scope: !1631)
!1631 = distinct !DILexicalBlock(scope: !1627, file: !3, line: 1711, column: 99)
!1632 = !DILocation(line: 1713, column: 2, scope: !1631)
!1633 = !DILocation(line: 1714, column: 75, scope: !1634)
!1634 = distinct !DILexicalBlock(scope: !1627, file: !3, line: 1713, column: 7)
!1635 = !DILocation(line: 1714, column: 51, scope: !1634)
!1636 = !DILocation(line: 1717, column: 60, scope: !1637)
!1637 = distinct !DILexicalBlock(scope: !1613, file: !3, line: 1716, column: 5)
!1638 = !DILocation(line: 1717, column: 36, scope: !1637)
!1639 = !DILocation(line: 1716, column: 5, scope: !1613)
!1640 = !DILocation(line: 1718, column: 32, scope: !1641)
!1641 = distinct !DILexicalBlock(scope: !1637, file: !3, line: 1717, column: 80)
!1642 = !DILocation(line: 1719, column: 2, scope: !1641)
!1643 = !DILocation(line: 1720, column: 54, scope: !1644)
!1644 = distinct !DILexicalBlock(scope: !1637, file: !3, line: 1719, column: 7)
!1645 = !DILocation(line: 1720, column: 31, scope: !1644)
!1646 = !DILocation(line: 1723, column: 59, scope: !1647)
!1647 = distinct !DILexicalBlock(scope: !1613, file: !3, line: 1722, column: 5)
!1648 = !DILocation(line: 1723, column: 35, scope: !1647)
!1649 = !DILocation(line: 1722, column: 5, scope: !1613)
!1650 = !DILocation(line: 1724, column: 31, scope: !1651)
!1651 = distinct !DILexicalBlock(scope: !1647, file: !3, line: 1723, column: 79)
!1652 = !DILocation(line: 1725, column: 2, scope: !1651)
!1653 = !DILocation(line: 1726, column: 53, scope: !1654)
!1654 = distinct !DILexicalBlock(scope: !1647, file: !3, line: 1725, column: 7)
!1655 = !DILocation(line: 1726, column: 30, scope: !1654)
!1656 = !DILocation(line: 1729, column: 59, scope: !1657)
!1657 = distinct !DILexicalBlock(scope: !1613, file: !3, line: 1728, column: 5)
!1658 = !DILocation(line: 1729, column: 35, scope: !1657)
!1659 = !DILocation(line: 1728, column: 5, scope: !1613)
!1660 = !DILocation(line: 1730, column: 31, scope: !1661)
!1661 = distinct !DILexicalBlock(scope: !1657, file: !3, line: 1729, column: 79)
!1662 = !DILocation(line: 1731, column: 2, scope: !1661)
!1663 = !DILocation(line: 1732, column: 55, scope: !1664)
!1664 = distinct !DILexicalBlock(scope: !1657, file: !3, line: 1731, column: 7)
!1665 = !DILocation(line: 1732, column: 31, scope: !1664)
!1666 = !DILocation(line: 1735, column: 59, scope: !1667)
!1667 = distinct !DILexicalBlock(scope: !1613, file: !3, line: 1734, column: 5)
!1668 = !DILocation(line: 1735, column: 35, scope: !1667)
!1669 = !DILocation(line: 1734, column: 5, scope: !1613)
!1670 = !DILocation(line: 1736, column: 31, scope: !1671)
!1671 = distinct !DILexicalBlock(scope: !1667, file: !3, line: 1735, column: 79)
!1672 = !DILocation(line: 1737, column: 2, scope: !1671)
!1673 = !DILocation(line: 1738, column: 55, scope: !1674)
!1674 = distinct !DILexicalBlock(scope: !1667, file: !3, line: 1737, column: 7)
!1675 = !DILocation(line: 1738, column: 31, scope: !1674)
!1676 = !DILocation(line: 1741, column: 59, scope: !1677)
!1677 = distinct !DILexicalBlock(scope: !1613, file: !3, line: 1740, column: 5)
!1678 = !DILocation(line: 1741, column: 35, scope: !1677)
!1679 = !DILocation(line: 1740, column: 5, scope: !1613)
!1680 = !DILocation(line: 1742, column: 31, scope: !1681)
!1681 = distinct !DILexicalBlock(scope: !1677, file: !3, line: 1741, column: 79)
!1682 = !DILocation(line: 1743, column: 2, scope: !1681)
!1683 = !DILocation(line: 1744, column: 55, scope: !1684)
!1684 = distinct !DILexicalBlock(scope: !1677, file: !3, line: 1743, column: 7)
!1685 = !DILocation(line: 1744, column: 31, scope: !1684)
!1686 = !DILocation(line: 1747, column: 59, scope: !1687)
!1687 = distinct !DILexicalBlock(scope: !1613, file: !3, line: 1746, column: 5)
!1688 = !DILocation(line: 1747, column: 35, scope: !1687)
!1689 = !DILocation(line: 1746, column: 5, scope: !1613)
!1690 = !DILocation(line: 1748, column: 31, scope: !1691)
!1691 = distinct !DILexicalBlock(scope: !1687, file: !3, line: 1747, column: 79)
!1692 = !DILocation(line: 1749, column: 2, scope: !1691)
!1693 = !DILocation(line: 1750, column: 55, scope: !1694)
!1694 = distinct !DILexicalBlock(scope: !1687, file: !3, line: 1749, column: 7)
!1695 = !DILocation(line: 1750, column: 31, scope: !1694)
!1696 = !DILocation(line: 1753, column: 59, scope: !1697)
!1697 = distinct !DILexicalBlock(scope: !1613, file: !3, line: 1752, column: 5)
!1698 = !DILocation(line: 1753, column: 35, scope: !1697)
!1699 = !DILocation(line: 1752, column: 5, scope: !1613)
!1700 = !DILocation(line: 1754, column: 31, scope: !1701)
!1701 = distinct !DILexicalBlock(scope: !1697, file: !3, line: 1753, column: 79)
!1702 = !DILocation(line: 1755, column: 2, scope: !1701)
!1703 = !DILocation(line: 1756, column: 55, scope: !1704)
!1704 = distinct !DILexicalBlock(scope: !1697, file: !3, line: 1755, column: 7)
!1705 = !DILocation(line: 1756, column: 31, scope: !1704)
!1706 = !DILocation(line: 1759, column: 59, scope: !1707)
!1707 = distinct !DILexicalBlock(scope: !1613, file: !3, line: 1758, column: 5)
!1708 = !DILocation(line: 1759, column: 35, scope: !1707)
!1709 = !DILocation(line: 1758, column: 5, scope: !1613)
!1710 = !DILocation(line: 1760, column: 31, scope: !1711)
!1711 = distinct !DILexicalBlock(scope: !1707, file: !3, line: 1759, column: 79)
!1712 = !DILocation(line: 1761, column: 2, scope: !1711)
!1713 = !DILocation(line: 1762, column: 55, scope: !1714)
!1714 = distinct !DILexicalBlock(scope: !1707, file: !3, line: 1761, column: 7)
!1715 = !DILocation(line: 1762, column: 31, scope: !1714)
!1716 = !DILocation(line: 1765, column: 59, scope: !1717)
!1717 = distinct !DILexicalBlock(scope: !1613, file: !3, line: 1764, column: 5)
!1718 = !DILocation(line: 1765, column: 35, scope: !1717)
!1719 = !DILocation(line: 1764, column: 5, scope: !1613)
!1720 = !DILocation(line: 1766, column: 31, scope: !1721)
!1721 = distinct !DILexicalBlock(scope: !1717, file: !3, line: 1765, column: 79)
!1722 = !DILocation(line: 1767, column: 2, scope: !1721)
!1723 = !DILocation(line: 1768, column: 55, scope: !1724)
!1724 = distinct !DILexicalBlock(scope: !1717, file: !3, line: 1767, column: 7)
!1725 = !DILocation(line: 1768, column: 31, scope: !1724)
!1726 = !DILocation(line: 1771, column: 59, scope: !1727)
!1727 = distinct !DILexicalBlock(scope: !1613, file: !3, line: 1770, column: 5)
!1728 = !DILocation(line: 1771, column: 35, scope: !1727)
!1729 = !DILocation(line: 1770, column: 5, scope: !1613)
!1730 = !DILocation(line: 1772, column: 31, scope: !1731)
!1731 = distinct !DILexicalBlock(scope: !1727, file: !3, line: 1771, column: 79)
!1732 = !DILocation(line: 1773, column: 2, scope: !1731)
!1733 = !DILocation(line: 1774, column: 55, scope: !1734)
!1734 = distinct !DILexicalBlock(scope: !1727, file: !3, line: 1773, column: 7)
!1735 = !DILocation(line: 1774, column: 31, scope: !1734)
!1736 = !DILocation(line: 1777, column: 59, scope: !1737)
!1737 = distinct !DILexicalBlock(scope: !1613, file: !3, line: 1776, column: 5)
!1738 = !DILocation(line: 1777, column: 35, scope: !1737)
!1739 = !DILocation(line: 1776, column: 5, scope: !1613)
!1740 = !DILocation(line: 1778, column: 31, scope: !1741)
!1741 = distinct !DILexicalBlock(scope: !1737, file: !3, line: 1777, column: 79)
!1742 = !DILocation(line: 1779, column: 2, scope: !1741)
!1743 = !DILocation(line: 1780, column: 55, scope: !1744)
!1744 = distinct !DILexicalBlock(scope: !1737, file: !3, line: 1779, column: 7)
!1745 = !DILocation(line: 1780, column: 31, scope: !1744)
!1746 = !DILocation(line: 1783, column: 61, scope: !1747)
!1747 = distinct !DILexicalBlock(scope: !1613, file: !3, line: 1782, column: 5)
!1748 = !DILocation(line: 1783, column: 37, scope: !1747)
!1749 = !DILocation(line: 1782, column: 5, scope: !1613)
!1750 = !DILocation(line: 1784, column: 33, scope: !1751)
!1751 = distinct !DILexicalBlock(scope: !1747, file: !3, line: 1783, column: 81)
!1752 = !DILocation(line: 1785, column: 2, scope: !1751)
!1753 = !DILocation(line: 1786, column: 57, scope: !1754)
!1754 = distinct !DILexicalBlock(scope: !1747, file: !3, line: 1785, column: 7)
!1755 = !DILocation(line: 1786, column: 33, scope: !1754)
!1756 = !DILocation(line: 1789, column: 65, scope: !1613)
!1757 = !DILocation(line: 1789, column: 57, scope: !1613)
!1758 = !DILocation(line: 1789, column: 38, scope: !1613)
!1759 = !DILocation(line: 1789, column: 37, scope: !1613)
!1760 = !DILocation(line: 1790, column: 71, scope: !1613)
!1761 = !DILocation(line: 1790, column: 63, scope: !1613)
!1762 = !DILocation(line: 1790, column: 48, scope: !1613)
!1763 = !DILocation(line: 1790, column: 47, scope: !1613)
!1764 = !DILocation(line: 1791, column: 56, scope: !1613)
!1765 = !DILocation(line: 1791, column: 48, scope: !1613)
!1766 = !DILocation(line: 1791, column: 29, scope: !1613)
!1767 = !DILocation(line: 1791, column: 28, scope: !1613)
!1768 = !DILocation(line: 1792, column: 55, scope: !1613)
!1769 = !DILocation(line: 1792, column: 47, scope: !1613)
!1770 = !DILocation(line: 1792, column: 28, scope: !1613)
!1771 = !DILocation(line: 1792, column: 27, scope: !1613)
!1772 = !DILocation(line: 1793, column: 57, scope: !1613)
!1773 = !DILocation(line: 1793, column: 49, scope: !1613)
!1774 = !DILocation(line: 1793, column: 28, scope: !1613)
!1775 = !DILocation(line: 1793, column: 27, scope: !1613)
!1776 = !DILocation(line: 1794, column: 54, scope: !1613)
!1777 = !DILocation(line: 1794, column: 46, scope: !1613)
!1778 = !DILocation(line: 1794, column: 28, scope: !1613)
!1779 = !DILocation(line: 1794, column: 27, scope: !1613)
!1780 = !DILocation(line: 1795, column: 57, scope: !1613)
!1781 = !DILocation(line: 1795, column: 49, scope: !1613)
!1782 = !DILocation(line: 1795, column: 28, scope: !1613)
!1783 = !DILocation(line: 1795, column: 27, scope: !1613)
!1784 = !DILocation(line: 1796, column: 57, scope: !1613)
!1785 = !DILocation(line: 1796, column: 49, scope: !1613)
!1786 = !DILocation(line: 1796, column: 28, scope: !1613)
!1787 = !DILocation(line: 1796, column: 27, scope: !1613)
!1788 = !DILocation(line: 1797, column: 54, scope: !1613)
!1789 = !DILocation(line: 1797, column: 46, scope: !1613)
!1790 = !DILocation(line: 1797, column: 28, scope: !1613)
!1791 = !DILocation(line: 1797, column: 27, scope: !1613)
!1792 = !DILocation(line: 1798, column: 57, scope: !1613)
!1793 = !DILocation(line: 1798, column: 49, scope: !1613)
!1794 = !DILocation(line: 1798, column: 28, scope: !1613)
!1795 = !DILocation(line: 1798, column: 27, scope: !1613)
!1796 = !DILocation(line: 1799, column: 57, scope: !1613)
!1797 = !DILocation(line: 1799, column: 49, scope: !1613)
!1798 = !DILocation(line: 1799, column: 28, scope: !1613)
!1799 = !DILocation(line: 1799, column: 27, scope: !1613)
!1800 = !DILocation(line: 1800, column: 54, scope: !1613)
!1801 = !DILocation(line: 1800, column: 46, scope: !1613)
!1802 = !DILocation(line: 1800, column: 28, scope: !1613)
!1803 = !DILocation(line: 1800, column: 27, scope: !1613)
!1804 = !DILocation(line: 1801, column: 57, scope: !1613)
!1805 = !DILocation(line: 1801, column: 49, scope: !1613)
!1806 = !DILocation(line: 1801, column: 28, scope: !1613)
!1807 = !DILocation(line: 1801, column: 27, scope: !1613)
!1808 = !DILocation(line: 1802, column: 65, scope: !1613)
!1809 = !DILocation(line: 1802, column: 57, scope: !1613)
!1810 = !DILocation(line: 1802, column: 30, scope: !1613)
!1811 = !DILocation(line: 1802, column: 29, scope: !1613)
!1812 = !DILocation(line: 1804, column: 18, scope: !1613)
!1813 = !DILocation(line: 1805, column: 20, scope: !1613)
!1814 = !DILocation(line: 1806, column: 21, scope: !1613)
!1815 = !DILocation(line: 1807, column: 15, scope: !1613)
!1816 = !DILocation(line: 1808, column: 16, scope: !1613)
!1817 = !DILocation(line: 1809, column: 16, scope: !1613)
!1818 = !DILocation(line: 1810, column: 16, scope: !1613)
!1819 = !DILocation(line: 1811, column: 16, scope: !1613)
!1820 = !DILocation(line: 1812, column: 19, scope: !1613)
!1821 = !DILocation(line: 1812, column: 48, scope: !1613)
!1822 = !DILocation(line: 1812, column: 18, scope: !1613)
!1823 = !DILocation(line: 1814, column: 27, scope: !1613)
!1824 = !DILocation(line: 1814, column: 2, scope: !1613)
!1825 = !DILocation(line: 1815, column: 29, scope: !1613)
!1826 = !DILocation(line: 1815, column: 2, scope: !1613)
!1827 = !DILocation(line: 1816, column: 30, scope: !1613)
!1828 = !DILocation(line: 1816, column: 2, scope: !1613)
!1829 = !DILocation(line: 1817, column: 24, scope: !1613)
!1830 = !DILocation(line: 1817, column: 2, scope: !1613)
!1831 = !DILocation(line: 1818, column: 25, scope: !1613)
!1832 = !DILocation(line: 1818, column: 2, scope: !1613)
!1833 = !DILocation(line: 1819, column: 25, scope: !1613)
!1834 = !DILocation(line: 1819, column: 2, scope: !1613)
!1835 = !DILocation(line: 1820, column: 25, scope: !1613)
!1836 = !DILocation(line: 1820, column: 2, scope: !1613)
!1837 = !DILocation(line: 1821, column: 25, scope: !1613)
!1838 = !DILocation(line: 1821, column: 2, scope: !1613)
!1839 = !DILocation(line: 1823, column: 2, scope: !1613)
!1840 = !DILocation(line: 1824, column: 1, scope: !1613)
!1841 = distinct !DISubprogram(name: "init_ui_gpu", linkageName: "_ZL11init_ui_gpuP8dcomplexS0_Pd", scope: !3, file: !3, line: 1526, type: !1842, scopeLine: 1528, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1058)
!1842 = !DISubroutineType(types: !1843)
!1843 = !{null, !98, !98, !106}
!1844 = !DILocalVariable(name: "u0", arg: 1, scope: !1841, file: !3, line: 1526, type: !98)
!1845 = !DILocation(line: 1526, column: 34, scope: !1841)
!1846 = !DILocalVariable(name: "u1", arg: 2, scope: !1841, file: !3, line: 1527, type: !98)
!1847 = !DILocation(line: 1527, column: 12, scope: !1841)
!1848 = !DILocalVariable(name: "twiddle", arg: 3, scope: !1841, file: !3, line: 1528, type: !106)
!1849 = !DILocation(line: 1528, column: 10, scope: !1841)
!1850 = !DILocation(line: 1532, column: 23, scope: !1841)
!1851 = !DILocation(line: 1533, column: 3, scope: !1841)
!1852 = !DILocation(line: 1532, column: 20, scope: !1841)
!1853 = !DILocation(line: 1532, column: 2, scope: !1841)
!1854 = !DILocation(line: 1533, column: 35, scope: !1841)
!1855 = !DILocation(line: 1534, column: 5, scope: !1841)
!1856 = !DILocation(line: 1535, column: 5, scope: !1841)
!1857 = !DILocation(line: 1540, column: 1, scope: !1841)
!1858 = distinct !DISubprogram(name: "compute_indexmap_gpu", linkageName: "_ZL20compute_indexmap_gpuPd", scope: !3, file: !3, line: 1342, type: !1859, scopeLine: 1342, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1058)
!1859 = !DISubroutineType(types: !1860)
!1860 = !{null, !106}
!1861 = !DILocalVariable(name: "twiddle", arg: 1, scope: !1858, file: !3, line: 1342, type: !106)
!1862 = !DILocation(line: 1342, column: 41, scope: !1858)
!1863 = !DILocation(line: 1346, column: 32, scope: !1858)
!1864 = !DILocation(line: 1347, column: 3, scope: !1858)
!1865 = !DILocation(line: 1346, column: 29, scope: !1858)
!1866 = !DILocation(line: 1346, column: 2, scope: !1858)
!1867 = !DILocation(line: 1347, column: 44, scope: !1858)
!1868 = !DILocation(line: 1351, column: 1, scope: !1858)
!1869 = distinct !DISubprogram(name: "compute_initial_conditions_gpu", linkageName: "_ZL30compute_initial_conditions_gpuP8dcomplex", scope: !3, file: !3, line: 1375, type: !1870, scopeLine: 1375, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1058)
!1870 = !DISubroutineType(types: !1871)
!1871 = !{null, !98}
!1872 = !DILocalVariable(name: "u0", arg: 1, scope: !1869, file: !3, line: 1375, type: !98)
!1873 = !DILocation(line: 1375, column: 53, scope: !1869)
!1874 = !DILocalVariable(name: "z", scope: !1869, file: !3, line: 1379, type: !97)
!1875 = !DILocation(line: 1379, column: 6, scope: !1869)
!1876 = !DILocalVariable(name: "start", scope: !1869, file: !3, line: 1380, type: !104)
!1877 = !DILocation(line: 1380, column: 9, scope: !1869)
!1878 = !DILocalVariable(name: "an", scope: !1869, file: !3, line: 1380, type: !104)
!1879 = !DILocation(line: 1380, column: 16, scope: !1869)
!1880 = !DILocalVariable(name: "starts", scope: !1869, file: !3, line: 1380, type: !1881)
!1881 = !DICompositeType(tag: DW_TAG_array_type, baseType: !104, size: 8192, elements: !1882)
!1882 = !{!1883}
!1883 = !DISubrange(count: 128)
!1884 = !DILocation(line: 1380, column: 20, scope: !1869)
!1885 = !DILocation(line: 1382, column: 8, scope: !1869)
!1886 = !DILocation(line: 1384, column: 2, scope: !1869)
!1887 = !DILocation(line: 1385, column: 17, scope: !1869)
!1888 = !DILocation(line: 1385, column: 2, scope: !1869)
!1889 = !DILocation(line: 1386, column: 2, scope: !1869)
!1890 = !DILocation(line: 1388, column: 14, scope: !1869)
!1891 = !DILocation(line: 1388, column: 2, scope: !1869)
!1892 = !DILocation(line: 1388, column: 12, scope: !1869)
!1893 = !DILocation(line: 1389, column: 7, scope: !1894)
!1894 = distinct !DILexicalBlock(scope: !1869, file: !3, line: 1389, column: 2)
!1895 = !DILocation(line: 1389, column: 6, scope: !1894)
!1896 = !DILocation(line: 1389, column: 11, scope: !1897)
!1897 = distinct !DILexicalBlock(scope: !1894, file: !3, line: 1389, column: 2)
!1898 = !DILocation(line: 1389, column: 12, scope: !1897)
!1899 = !DILocation(line: 1389, column: 2, scope: !1894)
!1900 = !DILocation(line: 1390, column: 18, scope: !1901)
!1901 = distinct !DILexicalBlock(scope: !1897, file: !3, line: 1389, column: 21)
!1902 = !DILocation(line: 1390, column: 3, scope: !1901)
!1903 = !DILocation(line: 1391, column: 15, scope: !1901)
!1904 = !DILocation(line: 1391, column: 10, scope: !1901)
!1905 = !DILocation(line: 1391, column: 3, scope: !1901)
!1906 = !DILocation(line: 1391, column: 13, scope: !1901)
!1907 = !DILocation(line: 1392, column: 2, scope: !1901)
!1908 = !DILocation(line: 1389, column: 18, scope: !1897)
!1909 = !DILocation(line: 1389, column: 2, scope: !1897)
!1910 = distinct !{!1910, !1899, !1911}
!1911 = !DILocation(line: 1392, column: 2, scope: !1894)
!1912 = !DILocation(line: 1394, column: 13, scope: !1869)
!1913 = !DILocation(line: 1394, column: 28, scope: !1869)
!1914 = !DILocation(line: 1394, column: 36, scope: !1869)
!1915 = !DILocation(line: 1394, column: 2, scope: !1869)
!1916 = !DILocation(line: 1396, column: 42, scope: !1869)
!1917 = !DILocation(line: 1397, column: 3, scope: !1869)
!1918 = !DILocation(line: 1396, column: 39, scope: !1869)
!1919 = !DILocation(line: 1396, column: 2, scope: !1869)
!1920 = !DILocation(line: 1397, column: 54, scope: !1869)
!1921 = !DILocation(line: 1398, column: 5, scope: !1869)
!1922 = !DILocation(line: 1402, column: 1, scope: !1869)
!1923 = distinct !DISubprogram(name: "fft_init_gpu", linkageName: "_ZL12fft_init_gpui", scope: !3, file: !3, line: 1467, type: !598, scopeLine: 1467, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1058)
!1924 = !DILocalVariable(name: "n", arg: 1, scope: !1923, file: !3, line: 1467, type: !97)
!1925 = !DILocation(line: 1467, column: 30, scope: !1923)
!1926 = !DILocalVariable(name: "m", scope: !1923, file: !3, line: 1471, type: !97)
!1927 = !DILocation(line: 1471, column: 6, scope: !1923)
!1928 = !DILocalVariable(name: "ku", scope: !1923, file: !3, line: 1471, type: !97)
!1929 = !DILocation(line: 1471, column: 8, scope: !1923)
!1930 = !DILocalVariable(name: "i", scope: !1923, file: !3, line: 1471, type: !97)
!1931 = !DILocation(line: 1471, column: 11, scope: !1923)
!1932 = !DILocalVariable(name: "j", scope: !1923, file: !3, line: 1471, type: !97)
!1933 = !DILocation(line: 1471, column: 13, scope: !1923)
!1934 = !DILocalVariable(name: "ln", scope: !1923, file: !3, line: 1471, type: !97)
!1935 = !DILocation(line: 1471, column: 15, scope: !1923)
!1936 = !DILocalVariable(name: "t", scope: !1923, file: !3, line: 1472, type: !104)
!1937 = !DILocation(line: 1472, column: 9, scope: !1923)
!1938 = !DILocalVariable(name: "ti", scope: !1923, file: !3, line: 1472, type: !104)
!1939 = !DILocation(line: 1472, column: 12, scope: !1923)
!1940 = !DILocation(line: 1479, column: 12, scope: !1923)
!1941 = !DILocation(line: 1479, column: 6, scope: !1923)
!1942 = !DILocation(line: 1479, column: 4, scope: !1923)
!1943 = !DILocation(line: 1480, column: 9, scope: !1923)
!1944 = !DILocation(line: 1480, column: 2, scope: !1923)
!1945 = !DILocation(line: 1480, column: 7, scope: !1923)
!1946 = !DILocation(line: 1481, column: 5, scope: !1923)
!1947 = !DILocation(line: 1482, column: 5, scope: !1923)
!1948 = !DILocation(line: 1483, column: 7, scope: !1949)
!1949 = distinct !DILexicalBlock(scope: !1923, file: !3, line: 1483, column: 2)
!1950 = !DILocation(line: 1483, column: 6, scope: !1949)
!1951 = !DILocation(line: 1483, column: 11, scope: !1952)
!1952 = distinct !DILexicalBlock(scope: !1949, file: !3, line: 1483, column: 2)
!1953 = !DILocation(line: 1483, column: 14, scope: !1952)
!1954 = !DILocation(line: 1483, column: 12, scope: !1952)
!1955 = !DILocation(line: 1483, column: 2, scope: !1949)
!1956 = !DILocation(line: 1484, column: 12, scope: !1957)
!1957 = distinct !DILexicalBlock(scope: !1952, file: !3, line: 1483, column: 21)
!1958 = !DILocation(line: 1484, column: 10, scope: !1957)
!1959 = !DILocation(line: 1484, column: 5, scope: !1957)
!1960 = !DILocation(line: 1485, column: 8, scope: !1961)
!1961 = distinct !DILexicalBlock(scope: !1957, file: !3, line: 1485, column: 3)
!1962 = !DILocation(line: 1485, column: 7, scope: !1961)
!1963 = !DILocation(line: 1485, column: 12, scope: !1964)
!1964 = distinct !DILexicalBlock(scope: !1961, file: !3, line: 1485, column: 3)
!1965 = !DILocation(line: 1485, column: 15, scope: !1964)
!1966 = !DILocation(line: 1485, column: 17, scope: !1964)
!1967 = !DILocation(line: 1485, column: 13, scope: !1964)
!1968 = !DILocation(line: 1485, column: 3, scope: !1961)
!1969 = !DILocation(line: 1486, column: 9, scope: !1970)
!1970 = distinct !DILexicalBlock(scope: !1964, file: !3, line: 1485, column: 25)
!1971 = !DILocation(line: 1486, column: 13, scope: !1970)
!1972 = !DILocation(line: 1486, column: 11, scope: !1970)
!1973 = !DILocation(line: 1486, column: 7, scope: !1970)
!1974 = !DILocation(line: 1487, column: 16, scope: !1970)
!1975 = !DILocation(line: 1487, column: 4, scope: !1970)
!1976 = !DILocation(line: 1487, column: 6, scope: !1970)
!1977 = !DILocation(line: 1487, column: 8, scope: !1970)
!1978 = !DILocation(line: 1487, column: 7, scope: !1970)
!1979 = !DILocation(line: 1487, column: 10, scope: !1970)
!1980 = !DILocation(line: 1487, column: 14, scope: !1970)
!1981 = !DILocation(line: 1488, column: 3, scope: !1970)
!1982 = !DILocation(line: 1485, column: 22, scope: !1964)
!1983 = !DILocation(line: 1485, column: 3, scope: !1964)
!1984 = distinct !{!1984, !1968, !1985}
!1985 = !DILocation(line: 1488, column: 3, scope: !1961)
!1986 = !DILocation(line: 1489, column: 8, scope: !1957)
!1987 = !DILocation(line: 1489, column: 13, scope: !1957)
!1988 = !DILocation(line: 1489, column: 11, scope: !1957)
!1989 = !DILocation(line: 1489, column: 6, scope: !1957)
!1990 = !DILocation(line: 1490, column: 12, scope: !1957)
!1991 = !DILocation(line: 1490, column: 10, scope: !1957)
!1992 = !DILocation(line: 1490, column: 6, scope: !1957)
!1993 = !DILocation(line: 1491, column: 2, scope: !1957)
!1994 = !DILocation(line: 1483, column: 18, scope: !1952)
!1995 = !DILocation(line: 1483, column: 2, scope: !1952)
!1996 = distinct !{!1996, !1955, !1997}
!1997 = !DILocation(line: 1491, column: 2, scope: !1949)
!1998 = !DILocation(line: 1492, column: 13, scope: !1923)
!1999 = !DILocation(line: 1492, column: 23, scope: !1923)
!2000 = !DILocation(line: 1492, column: 26, scope: !1923)
!2001 = !DILocation(line: 1492, column: 2, scope: !1923)
!2002 = !DILocation(line: 1496, column: 1, scope: !1923)
!2003 = distinct !DISubprogram(name: "fft_gpu", linkageName: "_ZL7fft_gpuiP8dcomplexS0_", scope: !3, file: !3, line: 1445, type: !2004, scopeLine: 1447, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1058)
!2004 = !DISubroutineType(types: !2005)
!2005 = !{null, !97, !98, !98}
!2006 = !DILocalVariable(name: "dir", arg: 1, scope: !2003, file: !3, line: 1445, type: !97)
!2007 = !DILocation(line: 1445, column: 25, scope: !2003)
!2008 = !DILocalVariable(name: "x1", arg: 2, scope: !2003, file: !3, line: 1446, type: !98)
!2009 = !DILocation(line: 1446, column: 12, scope: !2003)
!2010 = !DILocalVariable(name: "x2", arg: 3, scope: !2003, file: !3, line: 1447, type: !98)
!2011 = !DILocation(line: 1447, column: 12, scope: !2003)
!2012 = !DILocation(line: 1456, column: 5, scope: !2013)
!2013 = distinct !DILexicalBlock(scope: !2003, file: !3, line: 1456, column: 5)
!2014 = !DILocation(line: 1456, column: 8, scope: !2013)
!2015 = !DILocation(line: 1456, column: 5, scope: !2003)
!2016 = !DILocation(line: 1457, column: 17, scope: !2017)
!2017 = distinct !DILexicalBlock(scope: !2013, file: !3, line: 1456, column: 12)
!2018 = !DILocation(line: 1457, column: 27, scope: !2017)
!2019 = !DILocation(line: 1457, column: 31, scope: !2017)
!2020 = !DILocation(line: 1457, column: 35, scope: !2017)
!2021 = !DILocation(line: 1457, column: 46, scope: !2017)
!2022 = !DILocation(line: 1457, column: 3, scope: !2017)
!2023 = !DILocation(line: 1458, column: 17, scope: !2017)
!2024 = !DILocation(line: 1458, column: 27, scope: !2017)
!2025 = !DILocation(line: 1458, column: 31, scope: !2017)
!2026 = !DILocation(line: 1458, column: 35, scope: !2017)
!2027 = !DILocation(line: 1458, column: 46, scope: !2017)
!2028 = !DILocation(line: 1458, column: 3, scope: !2017)
!2029 = !DILocation(line: 1459, column: 17, scope: !2017)
!2030 = !DILocation(line: 1459, column: 27, scope: !2017)
!2031 = !DILocation(line: 1459, column: 31, scope: !2017)
!2032 = !DILocation(line: 1459, column: 35, scope: !2017)
!2033 = !DILocation(line: 1459, column: 46, scope: !2017)
!2034 = !DILocation(line: 1459, column: 3, scope: !2017)
!2035 = !DILocation(line: 1460, column: 2, scope: !2017)
!2036 = !DILocation(line: 1461, column: 18, scope: !2037)
!2037 = distinct !DILexicalBlock(scope: !2013, file: !3, line: 1460, column: 7)
!2038 = !DILocation(line: 1461, column: 28, scope: !2037)
!2039 = !DILocation(line: 1461, column: 32, scope: !2037)
!2040 = !DILocation(line: 1461, column: 36, scope: !2037)
!2041 = !DILocation(line: 1461, column: 47, scope: !2037)
!2042 = !DILocation(line: 1461, column: 3, scope: !2037)
!2043 = !DILocation(line: 1462, column: 18, scope: !2037)
!2044 = !DILocation(line: 1462, column: 28, scope: !2037)
!2045 = !DILocation(line: 1462, column: 32, scope: !2037)
!2046 = !DILocation(line: 1462, column: 36, scope: !2037)
!2047 = !DILocation(line: 1462, column: 47, scope: !2037)
!2048 = !DILocation(line: 1462, column: 3, scope: !2037)
!2049 = !DILocation(line: 1463, column: 18, scope: !2037)
!2050 = !DILocation(line: 1463, column: 28, scope: !2037)
!2051 = !DILocation(line: 1463, column: 32, scope: !2037)
!2052 = !DILocation(line: 1463, column: 36, scope: !2037)
!2053 = !DILocation(line: 1463, column: 47, scope: !2037)
!2054 = !DILocation(line: 1463, column: 3, scope: !2037)
!2055 = !DILocation(line: 1465, column: 1, scope: !2003)
!2056 = distinct !DISubprogram(name: "evolve_gpu", linkageName: "_ZL10evolve_gpuP8dcomplexS0_Pd", scope: !3, file: !3, line: 1416, type: !1842, scopeLine: 1418, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1058)
!2057 = !DILocalVariable(name: "u0", arg: 1, scope: !2056, file: !3, line: 1416, type: !98)
!2058 = !DILocation(line: 1416, column: 33, scope: !2056)
!2059 = !DILocalVariable(name: "u1", arg: 2, scope: !2056, file: !3, line: 1417, type: !98)
!2060 = !DILocation(line: 1417, column: 12, scope: !2056)
!2061 = !DILocalVariable(name: "twiddle", arg: 3, scope: !2056, file: !3, line: 1418, type: !106)
!2062 = !DILocation(line: 1418, column: 10, scope: !2056)
!2063 = !DILocation(line: 1422, column: 22, scope: !2056)
!2064 = !DILocation(line: 1423, column: 3, scope: !2056)
!2065 = !DILocation(line: 1422, column: 19, scope: !2056)
!2066 = !DILocation(line: 1422, column: 2, scope: !2056)
!2067 = !DILocation(line: 1423, column: 34, scope: !2056)
!2068 = !DILocation(line: 1424, column: 5, scope: !2056)
!2069 = !DILocation(line: 1425, column: 5, scope: !2056)
!2070 = !DILocation(line: 1426, column: 2, scope: !2056)
!2071 = !DILocation(line: 1430, column: 1, scope: !2056)
!2072 = distinct !DISubprogram(name: "checksum_gpu", linkageName: "_ZL12checksum_gpuiP8dcomplex", scope: !3, file: !3, line: 1296, type: !2073, scopeLine: 1297, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1058)
!2073 = !DISubroutineType(types: !2074)
!2074 = !{null, !97, !98}
!2075 = !DILocalVariable(name: "iteration", arg: 1, scope: !2072, file: !3, line: 1296, type: !97)
!2076 = !DILocation(line: 1296, column: 30, scope: !2072)
!2077 = !DILocalVariable(name: "u1", arg: 2, scope: !2072, file: !3, line: 1297, type: !98)
!2078 = !DILocation(line: 1297, column: 12, scope: !2072)
!2079 = !DILocation(line: 1301, column: 24, scope: !2072)
!2080 = !DILocation(line: 1302, column: 3, scope: !2072)
!2081 = !DILocation(line: 1303, column: 3, scope: !2072)
!2082 = !DILocation(line: 1301, column: 21, scope: !2072)
!2083 = !DILocation(line: 1301, column: 2, scope: !2072)
!2084 = !DILocation(line: 1303, column: 23, scope: !2072)
!2085 = !DILocation(line: 1304, column: 5, scope: !2072)
!2086 = !DILocation(line: 1305, column: 5, scope: !2072)
!2087 = !DILocation(line: 1309, column: 1, scope: !2072)
!2088 = distinct !DISubprogram(name: "verify", linkageName: "_ZL6verifyiiiiPiPc", scope: !3, file: !3, line: 1826, type: !2089, scopeLine: 1831, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1058)
!2089 = !DISubroutineType(types: !2090)
!2090 = !{null, !97, !97, !97, !97, !2091, !108}
!2091 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !1368, size: 64)
!2092 = !DILocalVariable(name: "d1", arg: 1, scope: !2088, file: !3, line: 1826, type: !97)
!2093 = !DILocation(line: 1826, column: 24, scope: !2088)
!2094 = !DILocalVariable(name: "d2", arg: 2, scope: !2088, file: !3, line: 1827, type: !97)
!2095 = !DILocation(line: 1827, column: 7, scope: !2088)
!2096 = !DILocalVariable(name: "d3", arg: 3, scope: !2088, file: !3, line: 1828, type: !97)
!2097 = !DILocation(line: 1828, column: 7, scope: !2088)
!2098 = !DILocalVariable(name: "nt", arg: 4, scope: !2088, file: !3, line: 1829, type: !97)
!2099 = !DILocation(line: 1829, column: 7, scope: !2088)
!2100 = !DILocalVariable(name: "verified", arg: 5, scope: !2088, file: !3, line: 1830, type: !2091)
!2101 = !DILocation(line: 1830, column: 12, scope: !2088)
!2102 = !DILocalVariable(name: "class_npb", arg: 6, scope: !2088, file: !3, line: 1831, type: !108)
!2103 = !DILocation(line: 1831, column: 9, scope: !2088)
!2104 = !DILocalVariable(name: "i", scope: !2088, file: !3, line: 1832, type: !97)
!2105 = !DILocation(line: 1832, column: 6, scope: !2088)
!2106 = !DILocalVariable(name: "err", scope: !2088, file: !3, line: 1833, type: !104)
!2107 = !DILocation(line: 1833, column: 9, scope: !2088)
!2108 = !DILocalVariable(name: "epsilon", scope: !2088, file: !3, line: 1833, type: !104)
!2109 = !DILocation(line: 1833, column: 14, scope: !2088)
!2110 = !DILocalVariable(name: "csum_ref", scope: !2088, file: !3, line: 1839, type: !2111)
!2111 = !DICompositeType(tag: DW_TAG_array_type, baseType: !99, size: 3328, elements: !2112)
!2112 = !{!2113}
!2113 = !DISubrange(count: 26)
!2114 = !DILocation(line: 1839, column: 11, scope: !2088)
!2115 = !DILocation(line: 1840, column: 3, scope: !2088)
!2116 = !DILocation(line: 1840, column: 13, scope: !2088)
!2117 = !DILocation(line: 1841, column: 10, scope: !2088)
!2118 = !DILocation(line: 1842, column: 3, scope: !2088)
!2119 = !DILocation(line: 1842, column: 12, scope: !2088)
!2120 = !DILocation(line: 1843, column: 5, scope: !2121)
!2121 = distinct !DILexicalBlock(scope: !2088, file: !3, line: 1843, column: 5)
!2122 = !DILocation(line: 1843, column: 8, scope: !2121)
!2123 = !DILocation(line: 1843, column: 14, scope: !2121)
!2124 = !DILocation(line: 1843, column: 17, scope: !2121)
!2125 = !DILocation(line: 1843, column: 20, scope: !2121)
!2126 = !DILocation(line: 1843, column: 26, scope: !2121)
!2127 = !DILocation(line: 1843, column: 29, scope: !2121)
!2128 = !DILocation(line: 1843, column: 32, scope: !2121)
!2129 = !DILocation(line: 1843, column: 38, scope: !2121)
!2130 = !DILocation(line: 1843, column: 41, scope: !2121)
!2131 = !DILocation(line: 1843, column: 44, scope: !2121)
!2132 = !DILocation(line: 1843, column: 5, scope: !2088)
!2133 = !DILocation(line: 1849, column: 4, scope: !2134)
!2134 = distinct !DILexicalBlock(scope: !2121, file: !3, line: 1843, column: 49)
!2135 = !DILocation(line: 1849, column: 14, scope: !2134)
!2136 = !DILocation(line: 1850, column: 17, scope: !2134)
!2137 = !DILocation(line: 1850, column: 3, scope: !2134)
!2138 = !DILocation(line: 1850, column: 15, scope: !2134)
!2139 = !DILocation(line: 1851, column: 17, scope: !2134)
!2140 = !DILocation(line: 1851, column: 3, scope: !2134)
!2141 = !DILocation(line: 1851, column: 15, scope: !2134)
!2142 = !DILocation(line: 1852, column: 17, scope: !2134)
!2143 = !DILocation(line: 1852, column: 3, scope: !2134)
!2144 = !DILocation(line: 1852, column: 15, scope: !2134)
!2145 = !DILocation(line: 1853, column: 17, scope: !2134)
!2146 = !DILocation(line: 1853, column: 3, scope: !2134)
!2147 = !DILocation(line: 1853, column: 15, scope: !2134)
!2148 = !DILocation(line: 1854, column: 17, scope: !2134)
!2149 = !DILocation(line: 1854, column: 3, scope: !2134)
!2150 = !DILocation(line: 1854, column: 15, scope: !2134)
!2151 = !DILocation(line: 1855, column: 17, scope: !2134)
!2152 = !DILocation(line: 1855, column: 3, scope: !2134)
!2153 = !DILocation(line: 1855, column: 15, scope: !2134)
!2154 = !DILocation(line: 1856, column: 2, scope: !2134)
!2155 = !DILocation(line: 1856, column: 11, scope: !2156)
!2156 = distinct !DILexicalBlock(scope: !2121, file: !3, line: 1856, column: 11)
!2157 = !DILocation(line: 1856, column: 14, scope: !2156)
!2158 = !DILocation(line: 1856, column: 21, scope: !2156)
!2159 = !DILocation(line: 1856, column: 24, scope: !2156)
!2160 = !DILocation(line: 1856, column: 27, scope: !2156)
!2161 = !DILocation(line: 1856, column: 34, scope: !2156)
!2162 = !DILocation(line: 1856, column: 37, scope: !2156)
!2163 = !DILocation(line: 1856, column: 40, scope: !2156)
!2164 = !DILocation(line: 1856, column: 46, scope: !2156)
!2165 = !DILocation(line: 1856, column: 49, scope: !2156)
!2166 = !DILocation(line: 1856, column: 52, scope: !2156)
!2167 = !DILocation(line: 1856, column: 11, scope: !2121)
!2168 = !DILocation(line: 1862, column: 4, scope: !2169)
!2169 = distinct !DILexicalBlock(scope: !2156, file: !3, line: 1856, column: 57)
!2170 = !DILocation(line: 1862, column: 14, scope: !2169)
!2171 = !DILocation(line: 1863, column: 17, scope: !2169)
!2172 = !DILocation(line: 1863, column: 3, scope: !2169)
!2173 = !DILocation(line: 1863, column: 15, scope: !2169)
!2174 = !DILocation(line: 1864, column: 17, scope: !2169)
!2175 = !DILocation(line: 1864, column: 3, scope: !2169)
!2176 = !DILocation(line: 1864, column: 15, scope: !2169)
!2177 = !DILocation(line: 1865, column: 17, scope: !2169)
!2178 = !DILocation(line: 1865, column: 3, scope: !2169)
!2179 = !DILocation(line: 1865, column: 15, scope: !2169)
!2180 = !DILocation(line: 1866, column: 17, scope: !2169)
!2181 = !DILocation(line: 1866, column: 3, scope: !2169)
!2182 = !DILocation(line: 1866, column: 15, scope: !2169)
!2183 = !DILocation(line: 1867, column: 17, scope: !2169)
!2184 = !DILocation(line: 1867, column: 3, scope: !2169)
!2185 = !DILocation(line: 1867, column: 15, scope: !2169)
!2186 = !DILocation(line: 1868, column: 17, scope: !2169)
!2187 = !DILocation(line: 1868, column: 3, scope: !2169)
!2188 = !DILocation(line: 1868, column: 15, scope: !2169)
!2189 = !DILocation(line: 1869, column: 2, scope: !2169)
!2190 = !DILocation(line: 1869, column: 11, scope: !2191)
!2191 = distinct !DILexicalBlock(scope: !2156, file: !3, line: 1869, column: 11)
!2192 = !DILocation(line: 1869, column: 14, scope: !2191)
!2193 = !DILocation(line: 1869, column: 21, scope: !2191)
!2194 = !DILocation(line: 1869, column: 24, scope: !2191)
!2195 = !DILocation(line: 1869, column: 27, scope: !2191)
!2196 = !DILocation(line: 1869, column: 34, scope: !2191)
!2197 = !DILocation(line: 1869, column: 37, scope: !2191)
!2198 = !DILocation(line: 1869, column: 40, scope: !2191)
!2199 = !DILocation(line: 1869, column: 47, scope: !2191)
!2200 = !DILocation(line: 1869, column: 50, scope: !2191)
!2201 = !DILocation(line: 1869, column: 53, scope: !2191)
!2202 = !DILocation(line: 1869, column: 11, scope: !2156)
!2203 = !DILocation(line: 1875, column: 4, scope: !2204)
!2204 = distinct !DILexicalBlock(scope: !2191, file: !3, line: 1869, column: 58)
!2205 = !DILocation(line: 1875, column: 14, scope: !2204)
!2206 = !DILocation(line: 1876, column: 17, scope: !2204)
!2207 = !DILocation(line: 1876, column: 3, scope: !2204)
!2208 = !DILocation(line: 1876, column: 15, scope: !2204)
!2209 = !DILocation(line: 1877, column: 17, scope: !2204)
!2210 = !DILocation(line: 1877, column: 3, scope: !2204)
!2211 = !DILocation(line: 1877, column: 15, scope: !2204)
!2212 = !DILocation(line: 1878, column: 17, scope: !2204)
!2213 = !DILocation(line: 1878, column: 3, scope: !2204)
!2214 = !DILocation(line: 1878, column: 15, scope: !2204)
!2215 = !DILocation(line: 1879, column: 17, scope: !2204)
!2216 = !DILocation(line: 1879, column: 3, scope: !2204)
!2217 = !DILocation(line: 1879, column: 15, scope: !2204)
!2218 = !DILocation(line: 1880, column: 17, scope: !2204)
!2219 = !DILocation(line: 1880, column: 3, scope: !2204)
!2220 = !DILocation(line: 1880, column: 15, scope: !2204)
!2221 = !DILocation(line: 1881, column: 17, scope: !2204)
!2222 = !DILocation(line: 1881, column: 3, scope: !2204)
!2223 = !DILocation(line: 1881, column: 15, scope: !2204)
!2224 = !DILocation(line: 1882, column: 2, scope: !2204)
!2225 = !DILocation(line: 1882, column: 11, scope: !2226)
!2226 = distinct !DILexicalBlock(scope: !2191, file: !3, line: 1882, column: 11)
!2227 = !DILocation(line: 1882, column: 14, scope: !2226)
!2228 = !DILocation(line: 1882, column: 21, scope: !2226)
!2229 = !DILocation(line: 1882, column: 24, scope: !2226)
!2230 = !DILocation(line: 1882, column: 27, scope: !2226)
!2231 = !DILocation(line: 1882, column: 34, scope: !2226)
!2232 = !DILocation(line: 1882, column: 37, scope: !2226)
!2233 = !DILocation(line: 1882, column: 40, scope: !2226)
!2234 = !DILocation(line: 1882, column: 47, scope: !2226)
!2235 = !DILocation(line: 1882, column: 50, scope: !2226)
!2236 = !DILocation(line: 1882, column: 53, scope: !2226)
!2237 = !DILocation(line: 1882, column: 11, scope: !2191)
!2238 = !DILocation(line: 1888, column: 4, scope: !2239)
!2239 = distinct !DILexicalBlock(scope: !2226, file: !3, line: 1882, column: 59)
!2240 = !DILocation(line: 1888, column: 14, scope: !2239)
!2241 = !DILocation(line: 1889, column: 18, scope: !2239)
!2242 = !DILocation(line: 1889, column: 3, scope: !2239)
!2243 = !DILocation(line: 1889, column: 16, scope: !2239)
!2244 = !DILocation(line: 1890, column: 18, scope: !2239)
!2245 = !DILocation(line: 1890, column: 3, scope: !2239)
!2246 = !DILocation(line: 1890, column: 16, scope: !2239)
!2247 = !DILocation(line: 1891, column: 18, scope: !2239)
!2248 = !DILocation(line: 1891, column: 3, scope: !2239)
!2249 = !DILocation(line: 1891, column: 16, scope: !2239)
!2250 = !DILocation(line: 1892, column: 18, scope: !2239)
!2251 = !DILocation(line: 1892, column: 3, scope: !2239)
!2252 = !DILocation(line: 1892, column: 16, scope: !2239)
!2253 = !DILocation(line: 1893, column: 18, scope: !2239)
!2254 = !DILocation(line: 1893, column: 3, scope: !2239)
!2255 = !DILocation(line: 1893, column: 16, scope: !2239)
!2256 = !DILocation(line: 1894, column: 18, scope: !2239)
!2257 = !DILocation(line: 1894, column: 3, scope: !2239)
!2258 = !DILocation(line: 1894, column: 16, scope: !2239)
!2259 = !DILocation(line: 1895, column: 18, scope: !2239)
!2260 = !DILocation(line: 1895, column: 3, scope: !2239)
!2261 = !DILocation(line: 1895, column: 16, scope: !2239)
!2262 = !DILocation(line: 1896, column: 18, scope: !2239)
!2263 = !DILocation(line: 1896, column: 3, scope: !2239)
!2264 = !DILocation(line: 1896, column: 16, scope: !2239)
!2265 = !DILocation(line: 1897, column: 18, scope: !2239)
!2266 = !DILocation(line: 1897, column: 3, scope: !2239)
!2267 = !DILocation(line: 1897, column: 16, scope: !2239)
!2268 = !DILocation(line: 1898, column: 18, scope: !2239)
!2269 = !DILocation(line: 1898, column: 3, scope: !2239)
!2270 = !DILocation(line: 1898, column: 16, scope: !2239)
!2271 = !DILocation(line: 1899, column: 18, scope: !2239)
!2272 = !DILocation(line: 1899, column: 3, scope: !2239)
!2273 = !DILocation(line: 1899, column: 16, scope: !2239)
!2274 = !DILocation(line: 1900, column: 18, scope: !2239)
!2275 = !DILocation(line: 1900, column: 3, scope: !2239)
!2276 = !DILocation(line: 1900, column: 16, scope: !2239)
!2277 = !DILocation(line: 1901, column: 18, scope: !2239)
!2278 = !DILocation(line: 1901, column: 3, scope: !2239)
!2279 = !DILocation(line: 1901, column: 16, scope: !2239)
!2280 = !DILocation(line: 1902, column: 18, scope: !2239)
!2281 = !DILocation(line: 1902, column: 3, scope: !2239)
!2282 = !DILocation(line: 1902, column: 16, scope: !2239)
!2283 = !DILocation(line: 1903, column: 18, scope: !2239)
!2284 = !DILocation(line: 1903, column: 3, scope: !2239)
!2285 = !DILocation(line: 1903, column: 16, scope: !2239)
!2286 = !DILocation(line: 1904, column: 18, scope: !2239)
!2287 = !DILocation(line: 1904, column: 3, scope: !2239)
!2288 = !DILocation(line: 1904, column: 16, scope: !2239)
!2289 = !DILocation(line: 1905, column: 18, scope: !2239)
!2290 = !DILocation(line: 1905, column: 3, scope: !2239)
!2291 = !DILocation(line: 1905, column: 16, scope: !2239)
!2292 = !DILocation(line: 1906, column: 18, scope: !2239)
!2293 = !DILocation(line: 1906, column: 3, scope: !2239)
!2294 = !DILocation(line: 1906, column: 16, scope: !2239)
!2295 = !DILocation(line: 1907, column: 18, scope: !2239)
!2296 = !DILocation(line: 1907, column: 3, scope: !2239)
!2297 = !DILocation(line: 1907, column: 16, scope: !2239)
!2298 = !DILocation(line: 1908, column: 18, scope: !2239)
!2299 = !DILocation(line: 1908, column: 3, scope: !2239)
!2300 = !DILocation(line: 1908, column: 16, scope: !2239)
!2301 = !DILocation(line: 1909, column: 2, scope: !2239)
!2302 = !DILocation(line: 1909, column: 11, scope: !2303)
!2303 = distinct !DILexicalBlock(scope: !2226, file: !3, line: 1909, column: 11)
!2304 = !DILocation(line: 1909, column: 14, scope: !2303)
!2305 = !DILocation(line: 1909, column: 21, scope: !2303)
!2306 = !DILocation(line: 1909, column: 24, scope: !2303)
!2307 = !DILocation(line: 1909, column: 27, scope: !2303)
!2308 = !DILocation(line: 1909, column: 34, scope: !2303)
!2309 = !DILocation(line: 1909, column: 37, scope: !2303)
!2310 = !DILocation(line: 1909, column: 40, scope: !2303)
!2311 = !DILocation(line: 1909, column: 47, scope: !2303)
!2312 = !DILocation(line: 1909, column: 50, scope: !2303)
!2313 = !DILocation(line: 1909, column: 53, scope: !2303)
!2314 = !DILocation(line: 1909, column: 11, scope: !2226)
!2315 = !DILocation(line: 1915, column: 4, scope: !2316)
!2316 = distinct !DILexicalBlock(scope: !2303, file: !3, line: 1909, column: 59)
!2317 = !DILocation(line: 1915, column: 14, scope: !2316)
!2318 = !DILocation(line: 1916, column: 18, scope: !2316)
!2319 = !DILocation(line: 1916, column: 3, scope: !2316)
!2320 = !DILocation(line: 1916, column: 16, scope: !2316)
!2321 = !DILocation(line: 1917, column: 18, scope: !2316)
!2322 = !DILocation(line: 1917, column: 3, scope: !2316)
!2323 = !DILocation(line: 1917, column: 16, scope: !2316)
!2324 = !DILocation(line: 1918, column: 18, scope: !2316)
!2325 = !DILocation(line: 1918, column: 3, scope: !2316)
!2326 = !DILocation(line: 1918, column: 16, scope: !2316)
!2327 = !DILocation(line: 1919, column: 18, scope: !2316)
!2328 = !DILocation(line: 1919, column: 3, scope: !2316)
!2329 = !DILocation(line: 1919, column: 16, scope: !2316)
!2330 = !DILocation(line: 1920, column: 18, scope: !2316)
!2331 = !DILocation(line: 1920, column: 3, scope: !2316)
!2332 = !DILocation(line: 1920, column: 16, scope: !2316)
!2333 = !DILocation(line: 1921, column: 18, scope: !2316)
!2334 = !DILocation(line: 1921, column: 3, scope: !2316)
!2335 = !DILocation(line: 1921, column: 16, scope: !2316)
!2336 = !DILocation(line: 1922, column: 18, scope: !2316)
!2337 = !DILocation(line: 1922, column: 3, scope: !2316)
!2338 = !DILocation(line: 1922, column: 16, scope: !2316)
!2339 = !DILocation(line: 1923, column: 18, scope: !2316)
!2340 = !DILocation(line: 1923, column: 3, scope: !2316)
!2341 = !DILocation(line: 1923, column: 16, scope: !2316)
!2342 = !DILocation(line: 1924, column: 18, scope: !2316)
!2343 = !DILocation(line: 1924, column: 3, scope: !2316)
!2344 = !DILocation(line: 1924, column: 16, scope: !2316)
!2345 = !DILocation(line: 1925, column: 18, scope: !2316)
!2346 = !DILocation(line: 1925, column: 3, scope: !2316)
!2347 = !DILocation(line: 1925, column: 16, scope: !2316)
!2348 = !DILocation(line: 1926, column: 18, scope: !2316)
!2349 = !DILocation(line: 1926, column: 3, scope: !2316)
!2350 = !DILocation(line: 1926, column: 16, scope: !2316)
!2351 = !DILocation(line: 1927, column: 18, scope: !2316)
!2352 = !DILocation(line: 1927, column: 3, scope: !2316)
!2353 = !DILocation(line: 1927, column: 16, scope: !2316)
!2354 = !DILocation(line: 1928, column: 18, scope: !2316)
!2355 = !DILocation(line: 1928, column: 3, scope: !2316)
!2356 = !DILocation(line: 1928, column: 16, scope: !2316)
!2357 = !DILocation(line: 1929, column: 18, scope: !2316)
!2358 = !DILocation(line: 1929, column: 3, scope: !2316)
!2359 = !DILocation(line: 1929, column: 16, scope: !2316)
!2360 = !DILocation(line: 1930, column: 18, scope: !2316)
!2361 = !DILocation(line: 1930, column: 3, scope: !2316)
!2362 = !DILocation(line: 1930, column: 16, scope: !2316)
!2363 = !DILocation(line: 1931, column: 18, scope: !2316)
!2364 = !DILocation(line: 1931, column: 3, scope: !2316)
!2365 = !DILocation(line: 1931, column: 16, scope: !2316)
!2366 = !DILocation(line: 1932, column: 18, scope: !2316)
!2367 = !DILocation(line: 1932, column: 3, scope: !2316)
!2368 = !DILocation(line: 1932, column: 16, scope: !2316)
!2369 = !DILocation(line: 1933, column: 18, scope: !2316)
!2370 = !DILocation(line: 1933, column: 3, scope: !2316)
!2371 = !DILocation(line: 1933, column: 16, scope: !2316)
!2372 = !DILocation(line: 1934, column: 18, scope: !2316)
!2373 = !DILocation(line: 1934, column: 3, scope: !2316)
!2374 = !DILocation(line: 1934, column: 16, scope: !2316)
!2375 = !DILocation(line: 1935, column: 18, scope: !2316)
!2376 = !DILocation(line: 1935, column: 3, scope: !2316)
!2377 = !DILocation(line: 1935, column: 16, scope: !2316)
!2378 = !DILocation(line: 1936, column: 2, scope: !2316)
!2379 = !DILocation(line: 1936, column: 11, scope: !2380)
!2380 = distinct !DILexicalBlock(scope: !2303, file: !3, line: 1936, column: 11)
!2381 = !DILocation(line: 1936, column: 14, scope: !2380)
!2382 = !DILocation(line: 1936, column: 22, scope: !2380)
!2383 = !DILocation(line: 1936, column: 25, scope: !2380)
!2384 = !DILocation(line: 1936, column: 28, scope: !2380)
!2385 = !DILocation(line: 1936, column: 36, scope: !2380)
!2386 = !DILocation(line: 1936, column: 39, scope: !2380)
!2387 = !DILocation(line: 1936, column: 42, scope: !2380)
!2388 = !DILocation(line: 1936, column: 50, scope: !2380)
!2389 = !DILocation(line: 1936, column: 53, scope: !2380)
!2390 = !DILocation(line: 1936, column: 56, scope: !2380)
!2391 = !DILocation(line: 1936, column: 11, scope: !2303)
!2392 = !DILocation(line: 1942, column: 4, scope: !2393)
!2393 = distinct !DILexicalBlock(scope: !2380, file: !3, line: 1936, column: 62)
!2394 = !DILocation(line: 1942, column: 14, scope: !2393)
!2395 = !DILocation(line: 1943, column: 18, scope: !2393)
!2396 = !DILocation(line: 1943, column: 3, scope: !2393)
!2397 = !DILocation(line: 1943, column: 16, scope: !2393)
!2398 = !DILocation(line: 1944, column: 18, scope: !2393)
!2399 = !DILocation(line: 1944, column: 3, scope: !2393)
!2400 = !DILocation(line: 1944, column: 16, scope: !2393)
!2401 = !DILocation(line: 1945, column: 18, scope: !2393)
!2402 = !DILocation(line: 1945, column: 3, scope: !2393)
!2403 = !DILocation(line: 1945, column: 16, scope: !2393)
!2404 = !DILocation(line: 1946, column: 18, scope: !2393)
!2405 = !DILocation(line: 1946, column: 3, scope: !2393)
!2406 = !DILocation(line: 1946, column: 16, scope: !2393)
!2407 = !DILocation(line: 1947, column: 18, scope: !2393)
!2408 = !DILocation(line: 1947, column: 3, scope: !2393)
!2409 = !DILocation(line: 1947, column: 16, scope: !2393)
!2410 = !DILocation(line: 1948, column: 18, scope: !2393)
!2411 = !DILocation(line: 1948, column: 3, scope: !2393)
!2412 = !DILocation(line: 1948, column: 16, scope: !2393)
!2413 = !DILocation(line: 1949, column: 18, scope: !2393)
!2414 = !DILocation(line: 1949, column: 3, scope: !2393)
!2415 = !DILocation(line: 1949, column: 16, scope: !2393)
!2416 = !DILocation(line: 1950, column: 18, scope: !2393)
!2417 = !DILocation(line: 1950, column: 3, scope: !2393)
!2418 = !DILocation(line: 1950, column: 16, scope: !2393)
!2419 = !DILocation(line: 1951, column: 18, scope: !2393)
!2420 = !DILocation(line: 1951, column: 3, scope: !2393)
!2421 = !DILocation(line: 1951, column: 16, scope: !2393)
!2422 = !DILocation(line: 1952, column: 18, scope: !2393)
!2423 = !DILocation(line: 1952, column: 3, scope: !2393)
!2424 = !DILocation(line: 1952, column: 16, scope: !2393)
!2425 = !DILocation(line: 1953, column: 18, scope: !2393)
!2426 = !DILocation(line: 1953, column: 3, scope: !2393)
!2427 = !DILocation(line: 1953, column: 16, scope: !2393)
!2428 = !DILocation(line: 1954, column: 18, scope: !2393)
!2429 = !DILocation(line: 1954, column: 3, scope: !2393)
!2430 = !DILocation(line: 1954, column: 16, scope: !2393)
!2431 = !DILocation(line: 1955, column: 18, scope: !2393)
!2432 = !DILocation(line: 1955, column: 3, scope: !2393)
!2433 = !DILocation(line: 1955, column: 16, scope: !2393)
!2434 = !DILocation(line: 1956, column: 18, scope: !2393)
!2435 = !DILocation(line: 1956, column: 3, scope: !2393)
!2436 = !DILocation(line: 1956, column: 16, scope: !2393)
!2437 = !DILocation(line: 1957, column: 18, scope: !2393)
!2438 = !DILocation(line: 1957, column: 3, scope: !2393)
!2439 = !DILocation(line: 1957, column: 16, scope: !2393)
!2440 = !DILocation(line: 1958, column: 18, scope: !2393)
!2441 = !DILocation(line: 1958, column: 3, scope: !2393)
!2442 = !DILocation(line: 1958, column: 16, scope: !2393)
!2443 = !DILocation(line: 1959, column: 18, scope: !2393)
!2444 = !DILocation(line: 1959, column: 3, scope: !2393)
!2445 = !DILocation(line: 1959, column: 16, scope: !2393)
!2446 = !DILocation(line: 1960, column: 18, scope: !2393)
!2447 = !DILocation(line: 1960, column: 3, scope: !2393)
!2448 = !DILocation(line: 1960, column: 16, scope: !2393)
!2449 = !DILocation(line: 1961, column: 18, scope: !2393)
!2450 = !DILocation(line: 1961, column: 3, scope: !2393)
!2451 = !DILocation(line: 1961, column: 16, scope: !2393)
!2452 = !DILocation(line: 1962, column: 18, scope: !2393)
!2453 = !DILocation(line: 1962, column: 3, scope: !2393)
!2454 = !DILocation(line: 1962, column: 16, scope: !2393)
!2455 = !DILocation(line: 1963, column: 18, scope: !2393)
!2456 = !DILocation(line: 1963, column: 3, scope: !2393)
!2457 = !DILocation(line: 1963, column: 16, scope: !2393)
!2458 = !DILocation(line: 1964, column: 18, scope: !2393)
!2459 = !DILocation(line: 1964, column: 3, scope: !2393)
!2460 = !DILocation(line: 1964, column: 16, scope: !2393)
!2461 = !DILocation(line: 1965, column: 18, scope: !2393)
!2462 = !DILocation(line: 1965, column: 3, scope: !2393)
!2463 = !DILocation(line: 1965, column: 16, scope: !2393)
!2464 = !DILocation(line: 1966, column: 18, scope: !2393)
!2465 = !DILocation(line: 1966, column: 3, scope: !2393)
!2466 = !DILocation(line: 1966, column: 16, scope: !2393)
!2467 = !DILocation(line: 1967, column: 18, scope: !2393)
!2468 = !DILocation(line: 1967, column: 3, scope: !2393)
!2469 = !DILocation(line: 1967, column: 16, scope: !2393)
!2470 = !DILocation(line: 1968, column: 2, scope: !2393)
!2471 = !DILocation(line: 1968, column: 11, scope: !2472)
!2472 = distinct !DILexicalBlock(scope: !2380, file: !3, line: 1968, column: 11)
!2473 = !DILocation(line: 1968, column: 14, scope: !2472)
!2474 = !DILocation(line: 1968, column: 22, scope: !2472)
!2475 = !DILocation(line: 1968, column: 25, scope: !2472)
!2476 = !DILocation(line: 1968, column: 28, scope: !2472)
!2477 = !DILocation(line: 1968, column: 36, scope: !2472)
!2478 = !DILocation(line: 1968, column: 39, scope: !2472)
!2479 = !DILocation(line: 1968, column: 42, scope: !2472)
!2480 = !DILocation(line: 1968, column: 50, scope: !2472)
!2481 = !DILocation(line: 1968, column: 53, scope: !2472)
!2482 = !DILocation(line: 1968, column: 56, scope: !2472)
!2483 = !DILocation(line: 1968, column: 11, scope: !2380)
!2484 = !DILocation(line: 1974, column: 4, scope: !2485)
!2485 = distinct !DILexicalBlock(scope: !2472, file: !3, line: 1968, column: 62)
!2486 = !DILocation(line: 1974, column: 14, scope: !2485)
!2487 = !DILocation(line: 1975, column: 18, scope: !2485)
!2488 = !DILocation(line: 1975, column: 3, scope: !2485)
!2489 = !DILocation(line: 1975, column: 16, scope: !2485)
!2490 = !DILocation(line: 1976, column: 18, scope: !2485)
!2491 = !DILocation(line: 1976, column: 3, scope: !2485)
!2492 = !DILocation(line: 1976, column: 16, scope: !2485)
!2493 = !DILocation(line: 1977, column: 18, scope: !2485)
!2494 = !DILocation(line: 1977, column: 3, scope: !2485)
!2495 = !DILocation(line: 1977, column: 16, scope: !2485)
!2496 = !DILocation(line: 1978, column: 18, scope: !2485)
!2497 = !DILocation(line: 1978, column: 3, scope: !2485)
!2498 = !DILocation(line: 1978, column: 16, scope: !2485)
!2499 = !DILocation(line: 1979, column: 18, scope: !2485)
!2500 = !DILocation(line: 1979, column: 3, scope: !2485)
!2501 = !DILocation(line: 1979, column: 16, scope: !2485)
!2502 = !DILocation(line: 1980, column: 18, scope: !2485)
!2503 = !DILocation(line: 1980, column: 3, scope: !2485)
!2504 = !DILocation(line: 1980, column: 16, scope: !2485)
!2505 = !DILocation(line: 1981, column: 18, scope: !2485)
!2506 = !DILocation(line: 1981, column: 3, scope: !2485)
!2507 = !DILocation(line: 1981, column: 16, scope: !2485)
!2508 = !DILocation(line: 1982, column: 18, scope: !2485)
!2509 = !DILocation(line: 1982, column: 3, scope: !2485)
!2510 = !DILocation(line: 1982, column: 16, scope: !2485)
!2511 = !DILocation(line: 1983, column: 18, scope: !2485)
!2512 = !DILocation(line: 1983, column: 3, scope: !2485)
!2513 = !DILocation(line: 1983, column: 16, scope: !2485)
!2514 = !DILocation(line: 1984, column: 18, scope: !2485)
!2515 = !DILocation(line: 1984, column: 3, scope: !2485)
!2516 = !DILocation(line: 1984, column: 16, scope: !2485)
!2517 = !DILocation(line: 1985, column: 18, scope: !2485)
!2518 = !DILocation(line: 1985, column: 3, scope: !2485)
!2519 = !DILocation(line: 1985, column: 16, scope: !2485)
!2520 = !DILocation(line: 1986, column: 18, scope: !2485)
!2521 = !DILocation(line: 1986, column: 3, scope: !2485)
!2522 = !DILocation(line: 1986, column: 16, scope: !2485)
!2523 = !DILocation(line: 1987, column: 18, scope: !2485)
!2524 = !DILocation(line: 1987, column: 3, scope: !2485)
!2525 = !DILocation(line: 1987, column: 16, scope: !2485)
!2526 = !DILocation(line: 1988, column: 18, scope: !2485)
!2527 = !DILocation(line: 1988, column: 3, scope: !2485)
!2528 = !DILocation(line: 1988, column: 16, scope: !2485)
!2529 = !DILocation(line: 1989, column: 18, scope: !2485)
!2530 = !DILocation(line: 1989, column: 3, scope: !2485)
!2531 = !DILocation(line: 1989, column: 16, scope: !2485)
!2532 = !DILocation(line: 1990, column: 18, scope: !2485)
!2533 = !DILocation(line: 1990, column: 3, scope: !2485)
!2534 = !DILocation(line: 1990, column: 16, scope: !2485)
!2535 = !DILocation(line: 1991, column: 18, scope: !2485)
!2536 = !DILocation(line: 1991, column: 3, scope: !2485)
!2537 = !DILocation(line: 1991, column: 16, scope: !2485)
!2538 = !DILocation(line: 1992, column: 18, scope: !2485)
!2539 = !DILocation(line: 1992, column: 3, scope: !2485)
!2540 = !DILocation(line: 1992, column: 16, scope: !2485)
!2541 = !DILocation(line: 1993, column: 18, scope: !2485)
!2542 = !DILocation(line: 1993, column: 3, scope: !2485)
!2543 = !DILocation(line: 1993, column: 16, scope: !2485)
!2544 = !DILocation(line: 1994, column: 18, scope: !2485)
!2545 = !DILocation(line: 1994, column: 3, scope: !2485)
!2546 = !DILocation(line: 1994, column: 16, scope: !2485)
!2547 = !DILocation(line: 1995, column: 18, scope: !2485)
!2548 = !DILocation(line: 1995, column: 3, scope: !2485)
!2549 = !DILocation(line: 1995, column: 16, scope: !2485)
!2550 = !DILocation(line: 1996, column: 18, scope: !2485)
!2551 = !DILocation(line: 1996, column: 3, scope: !2485)
!2552 = !DILocation(line: 1996, column: 16, scope: !2485)
!2553 = !DILocation(line: 1997, column: 18, scope: !2485)
!2554 = !DILocation(line: 1997, column: 3, scope: !2485)
!2555 = !DILocation(line: 1997, column: 16, scope: !2485)
!2556 = !DILocation(line: 1998, column: 18, scope: !2485)
!2557 = !DILocation(line: 1998, column: 3, scope: !2485)
!2558 = !DILocation(line: 1998, column: 16, scope: !2485)
!2559 = !DILocation(line: 1999, column: 18, scope: !2485)
!2560 = !DILocation(line: 1999, column: 3, scope: !2485)
!2561 = !DILocation(line: 1999, column: 16, scope: !2485)
!2562 = !DILocation(line: 2000, column: 2, scope: !2485)
!2563 = !DILocation(line: 2001, column: 6, scope: !2564)
!2564 = distinct !DILexicalBlock(scope: !2088, file: !3, line: 2001, column: 5)
!2565 = !DILocation(line: 2001, column: 5, scope: !2564)
!2566 = !DILocation(line: 2001, column: 16, scope: !2564)
!2567 = !DILocation(line: 2001, column: 5, scope: !2088)
!2568 = !DILocation(line: 2002, column: 4, scope: !2569)
!2569 = distinct !DILexicalBlock(scope: !2564, file: !3, line: 2001, column: 23)
!2570 = !DILocation(line: 2002, column: 13, scope: !2569)
!2571 = !DILocation(line: 2003, column: 9, scope: !2572)
!2572 = distinct !DILexicalBlock(scope: !2569, file: !3, line: 2003, column: 3)
!2573 = !DILocation(line: 2003, column: 7, scope: !2572)
!2574 = !DILocation(line: 2003, column: 14, scope: !2575)
!2575 = distinct !DILexicalBlock(scope: !2572, file: !3, line: 2003, column: 3)
!2576 = !DILocation(line: 2003, column: 19, scope: !2575)
!2577 = !DILocation(line: 2003, column: 16, scope: !2575)
!2578 = !DILocation(line: 2003, column: 3, scope: !2572)
!2579 = !DILocation(line: 2004, column: 10, scope: !2580)
!2580 = distinct !DILexicalBlock(scope: !2575, file: !3, line: 2003, column: 27)
!2581 = !DILocation(line: 2004, column: 8, scope: !2580)
!2582 = !DILocation(line: 2006, column: 9, scope: !2583)
!2583 = distinct !DILexicalBlock(scope: !2580, file: !3, line: 2006, column: 7)
!2584 = !DILocation(line: 2006, column: 16, scope: !2583)
!2585 = !DILocation(line: 2006, column: 13, scope: !2583)
!2586 = !DILocation(line: 2006, column: 7, scope: !2580)
!2587 = !DILocation(line: 2007, column: 6, scope: !2588)
!2588 = distinct !DILexicalBlock(scope: !2583, file: !3, line: 2006, column: 25)
!2589 = !DILocation(line: 2007, column: 15, scope: !2588)
!2590 = !DILocation(line: 2008, column: 5, scope: !2588)
!2591 = !DILocation(line: 2010, column: 3, scope: !2580)
!2592 = !DILocation(line: 2003, column: 24, scope: !2575)
!2593 = !DILocation(line: 2003, column: 3, scope: !2575)
!2594 = distinct !{!2594, !2578, !2595}
!2595 = !DILocation(line: 2010, column: 3, scope: !2572)
!2596 = !DILocation(line: 2011, column: 2, scope: !2569)
!2597 = !DILocation(line: 2012, column: 6, scope: !2598)
!2598 = distinct !DILexicalBlock(scope: !2088, file: !3, line: 2012, column: 5)
!2599 = !DILocation(line: 2012, column: 5, scope: !2598)
!2600 = !DILocation(line: 2012, column: 16, scope: !2598)
!2601 = !DILocation(line: 2012, column: 5, scope: !2088)
!2602 = !DILocation(line: 2013, column: 7, scope: !2603)
!2603 = distinct !DILexicalBlock(scope: !2604, file: !3, line: 2013, column: 6)
!2604 = distinct !DILexicalBlock(scope: !2598, file: !3, line: 2012, column: 23)
!2605 = !DILocation(line: 2013, column: 6, scope: !2603)
!2606 = !DILocation(line: 2013, column: 6, scope: !2604)
!2607 = !DILocation(line: 2014, column: 4, scope: !2608)
!2608 = distinct !DILexicalBlock(scope: !2603, file: !3, line: 2013, column: 16)
!2609 = !DILocation(line: 2015, column: 3, scope: !2608)
!2610 = !DILocation(line: 2016, column: 4, scope: !2611)
!2611 = distinct !DILexicalBlock(scope: !2603, file: !3, line: 2015, column: 8)
!2612 = !DILocation(line: 2018, column: 2, scope: !2604)
!2613 = !DILocation(line: 2019, column: 31, scope: !2088)
!2614 = !DILocation(line: 2019, column: 30, scope: !2088)
!2615 = !DILocation(line: 2019, column: 2, scope: !2088)
!2616 = !DILocation(line: 2020, column: 1, scope: !2088)
!2617 = distinct !DISubprogram(name: "release_gpu", linkageName: "_ZL11release_gpuv", scope: !3, file: !3, line: 1636, type: !561, scopeLine: 1636, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1058)
!2618 = !DILocation(line: 1637, column: 11, scope: !2617)
!2619 = !DILocation(line: 1637, column: 2, scope: !2617)
!2620 = !DILocation(line: 1638, column: 11, scope: !2617)
!2621 = !DILocation(line: 1638, column: 2, scope: !2617)
!2622 = !DILocation(line: 1639, column: 11, scope: !2617)
!2623 = !DILocation(line: 1639, column: 2, scope: !2617)
!2624 = !DILocation(line: 1640, column: 11, scope: !2617)
!2625 = !DILocation(line: 1640, column: 2, scope: !2617)
!2626 = !DILocation(line: 1641, column: 11, scope: !2617)
!2627 = !DILocation(line: 1641, column: 2, scope: !2617)
!2628 = !DILocation(line: 1642, column: 11, scope: !2617)
!2629 = !DILocation(line: 1642, column: 2, scope: !2617)
!2630 = !DILocation(line: 1643, column: 11, scope: !2617)
!2631 = !DILocation(line: 1643, column: 2, scope: !2617)
!2632 = !DILocation(line: 1644, column: 11, scope: !2617)
!2633 = !DILocation(line: 1644, column: 2, scope: !2617)
!2634 = !DILocation(line: 1645, column: 1, scope: !2617)
!2635 = distinct !DISubprogram(name: "cffts1_gpu_kernel_1", linkageName: "_Z19cffts1_gpu_kernel_1P8dcomplexS0_", scope: !3, file: !3, line: 762, type: !2636, scopeLine: 763, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !1058)
!2636 = !DISubroutineType(types: !2637)
!2637 = !{null, !98, !98}
!2638 = !DILocalVariable(name: "x_in", arg: 1, scope: !2635, file: !3, line: 762, type: !98)
!2639 = !DILocation(line: 762, column: 46, scope: !2635)
!2640 = !DILocalVariable(name: "y0", arg: 2, scope: !2635, file: !3, line: 763, type: !98)
!2641 = !DILocation(line: 763, column: 12, scope: !2635)
!2642 = !DILocation(line: 763, column: 17, scope: !2635)
!2643 = !DILocation(line: 773, column: 1, scope: !2635)
!2644 = distinct !DISubprogram(name: "cffts1_gpu_kernel_2", linkageName: "_Z19cffts1_gpu_kernel_2iP8dcomplexS0_S0_", scope: !3, file: !3, line: 780, type: !2645, scopeLine: 783, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !1058)
!2645 = !DISubroutineType(types: !2646)
!2646 = !{null, !2647, !98, !98, !98}
!2647 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !97)
!2648 = !DILocalVariable(name: "is", arg: 1, scope: !2644, file: !3, line: 780, type: !2647)
!2649 = !DILocation(line: 780, column: 47, scope: !2644)
!2650 = !DILocalVariable(name: "gty1", arg: 2, scope: !2644, file: !3, line: 781, type: !98)
!2651 = !DILocation(line: 781, column: 12, scope: !2644)
!2652 = !DILocalVariable(name: "gty2", arg: 3, scope: !2644, file: !3, line: 782, type: !98)
!2653 = !DILocation(line: 782, column: 12, scope: !2644)
!2654 = !DILocalVariable(name: "u_device", arg: 4, scope: !2644, file: !3, line: 783, type: !98)
!2655 = !DILocation(line: 783, column: 12, scope: !2644)
!2656 = !DILocation(line: 783, column: 23, scope: !2644)
!2657 = !DILocation(line: 886, column: 1, scope: !2644)
!2658 = distinct !DISubprogram(name: "cffts1_gpu_kernel_3", linkageName: "_Z19cffts1_gpu_kernel_3P8dcomplexS0_", scope: !3, file: !3, line: 895, type: !2636, scopeLine: 896, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !1058)
!2659 = !DILocalVariable(name: "x_out", arg: 1, scope: !2658, file: !3, line: 895, type: !98)
!2660 = !DILocation(line: 895, column: 46, scope: !2658)
!2661 = !DILocalVariable(name: "y0", arg: 2, scope: !2658, file: !3, line: 896, type: !98)
!2662 = !DILocation(line: 896, column: 12, scope: !2658)
!2663 = !DILocation(line: 896, column: 17, scope: !2658)
!2664 = !DILocation(line: 906, column: 1, scope: !2658)
!2665 = distinct !DISubprogram(name: "cffts2_gpu_kernel_1", linkageName: "_Z19cffts2_gpu_kernel_1P8dcomplexS0_", scope: !3, file: !3, line: 957, type: !2636, scopeLine: 958, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !1058)
!2666 = !DILocalVariable(name: "x_in", arg: 1, scope: !2665, file: !3, line: 957, type: !98)
!2667 = !DILocation(line: 957, column: 46, scope: !2665)
!2668 = !DILocalVariable(name: "y0", arg: 2, scope: !2665, file: !3, line: 958, type: !98)
!2669 = !DILocation(line: 958, column: 12, scope: !2665)
!2670 = !DILocation(line: 958, column: 17, scope: !2665)
!2671 = !DILocation(line: 965, column: 1, scope: !2665)
!2672 = distinct !DISubprogram(name: "cffts2_gpu_kernel_2", linkageName: "_Z19cffts2_gpu_kernel_2iP8dcomplexS0_S0_", scope: !3, file: !3, line: 972, type: !2645, scopeLine: 975, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !1058)
!2673 = !DILocalVariable(name: "is", arg: 1, scope: !2672, file: !3, line: 972, type: !2647)
!2674 = !DILocation(line: 972, column: 47, scope: !2672)
!2675 = !DILocalVariable(name: "gty1", arg: 2, scope: !2672, file: !3, line: 973, type: !98)
!2676 = !DILocation(line: 973, column: 12, scope: !2672)
!2677 = !DILocalVariable(name: "gty2", arg: 3, scope: !2672, file: !3, line: 974, type: !98)
!2678 = !DILocation(line: 974, column: 12, scope: !2672)
!2679 = !DILocalVariable(name: "u_device", arg: 4, scope: !2672, file: !3, line: 975, type: !98)
!2680 = !DILocation(line: 975, column: 12, scope: !2672)
!2681 = !DILocation(line: 975, column: 23, scope: !2672)
!2682 = !DILocation(line: 1080, column: 1, scope: !2672)
!2683 = distinct !DISubprogram(name: "cffts2_gpu_kernel_3", linkageName: "_Z19cffts2_gpu_kernel_3P8dcomplexS0_", scope: !3, file: !3, line: 1089, type: !2636, scopeLine: 1090, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !1058)
!2684 = !DILocalVariable(name: "x_out", arg: 1, scope: !2683, file: !3, line: 1089, type: !98)
!2685 = !DILocation(line: 1089, column: 46, scope: !2683)
!2686 = !DILocalVariable(name: "y0", arg: 2, scope: !2683, file: !3, line: 1090, type: !98)
!2687 = !DILocation(line: 1090, column: 12, scope: !2683)
!2688 = !DILocation(line: 1090, column: 17, scope: !2683)
!2689 = !DILocation(line: 1097, column: 1, scope: !2683)
!2690 = distinct !DISubprogram(name: "cffts3_gpu_kernel_1", linkageName: "_Z19cffts3_gpu_kernel_1P8dcomplexS0_", scope: !3, file: !3, line: 1246, type: !2636, scopeLine: 1247, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !1058)
!2691 = !DILocalVariable(name: "x_in", arg: 1, scope: !2690, file: !3, line: 1246, type: !98)
!2692 = !DILocation(line: 1246, column: 46, scope: !2690)
!2693 = !DILocalVariable(name: "y0", arg: 2, scope: !2690, file: !3, line: 1247, type: !98)
!2694 = !DILocation(line: 1247, column: 12, scope: !2690)
!2695 = !DILocation(line: 1247, column: 17, scope: !2690)
!2696 = !DILocation(line: 1254, column: 1, scope: !2690)
!2697 = distinct !DISubprogram(name: "cffts3_gpu_kernel_2", linkageName: "_Z19cffts3_gpu_kernel_2iP8dcomplexS0_S0_", scope: !3, file: !3, line: 1261, type: !2645, scopeLine: 1264, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !1058)
!2698 = !DILocalVariable(name: "is", arg: 1, scope: !2697, file: !3, line: 1261, type: !2647)
!2699 = !DILocation(line: 1261, column: 47, scope: !2697)
!2700 = !DILocalVariable(name: "gty1", arg: 2, scope: !2697, file: !3, line: 1262, type: !98)
!2701 = !DILocation(line: 1262, column: 12, scope: !2697)
!2702 = !DILocalVariable(name: "gty2", arg: 3, scope: !2697, file: !3, line: 1263, type: !98)
!2703 = !DILocation(line: 1263, column: 12, scope: !2697)
!2704 = !DILocalVariable(name: "u_device", arg: 4, scope: !2697, file: !3, line: 1264, type: !98)
!2705 = !DILocation(line: 1264, column: 12, scope: !2697)
!2706 = !DILocation(line: 1264, column: 23, scope: !2697)
!2707 = !DILocation(line: 1277, column: 1, scope: !2697)
!2708 = distinct !DISubprogram(name: "cffts3_gpu_kernel_3", linkageName: "_Z19cffts3_gpu_kernel_3P8dcomplexS0_", scope: !3, file: !3, line: 1286, type: !2636, scopeLine: 1287, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !1058)
!2709 = !DILocalVariable(name: "x_out", arg: 1, scope: !2708, file: !3, line: 1286, type: !98)
!2710 = !DILocation(line: 1286, column: 46, scope: !2708)
!2711 = !DILocalVariable(name: "y0", arg: 2, scope: !2708, file: !3, line: 1287, type: !98)
!2712 = !DILocation(line: 1287, column: 12, scope: !2708)
!2713 = !DILocation(line: 1287, column: 17, scope: !2708)
!2714 = !DILocation(line: 1294, column: 1, scope: !2708)
!2715 = distinct !DISubprogram(name: "checksum_gpu_kernel", linkageName: "_Z19checksum_gpu_kerneliP8dcomplexS0_", scope: !3, file: !3, line: 1311, type: !2004, scopeLine: 1313, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !1058)
!2716 = !DILocalVariable(name: "iteration", arg: 1, scope: !2715, file: !3, line: 1311, type: !97)
!2717 = !DILocation(line: 1311, column: 41, scope: !2715)
!2718 = !DILocalVariable(name: "u1", arg: 2, scope: !2715, file: !3, line: 1312, type: !98)
!2719 = !DILocation(line: 1312, column: 12, scope: !2715)
!2720 = !DILocalVariable(name: "sums", arg: 3, scope: !2715, file: !3, line: 1313, type: !98)
!2721 = !DILocation(line: 1313, column: 12, scope: !2715)
!2722 = !DILocation(line: 1313, column: 19, scope: !2715)
!2723 = !DILocation(line: 1340, column: 1, scope: !2715)
!2724 = distinct !DISubprogram(name: "compute_indexmap_gpu_kernel", linkageName: "_Z27compute_indexmap_gpu_kernelPd", scope: !3, file: !3, line: 1353, type: !1859, scopeLine: 1353, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !1058)
!2725 = !DILocalVariable(name: "twiddle", arg: 1, scope: !2724, file: !3, line: 1353, type: !106)
!2726 = !DILocation(line: 1353, column: 52, scope: !2724)
!2727 = !DILocation(line: 1353, column: 62, scope: !2724)
!2728 = !DILocation(line: 1373, column: 1, scope: !2724)
!2729 = distinct !DISubprogram(name: "compute_initial_conditions_gpu_kernel", linkageName: "_Z37compute_initial_conditions_gpu_kernelP8dcomplexPd", scope: !3, file: !3, line: 1404, type: !2730, scopeLine: 1405, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !1058)
!2730 = !DISubroutineType(types: !2731)
!2731 = !{null, !98, !106}
!2732 = !DILocalVariable(name: "u0", arg: 1, scope: !2729, file: !3, line: 1404, type: !98)
!2733 = !DILocation(line: 1404, column: 64, scope: !2729)
!2734 = !DILocalVariable(name: "starts", arg: 2, scope: !2729, file: !3, line: 1405, type: !106)
!2735 = !DILocation(line: 1405, column: 10, scope: !2729)
!2736 = !DILocation(line: 1405, column: 19, scope: !2729)
!2737 = !DILocation(line: 1414, column: 1, scope: !2729)
!2738 = distinct !DISubprogram(name: "evolve_gpu_kernel", linkageName: "_Z17evolve_gpu_kernelP8dcomplexS0_Pd", scope: !3, file: !3, line: 1432, type: !1842, scopeLine: 1434, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !1058)
!2739 = !DILocalVariable(name: "u0", arg: 1, scope: !2738, file: !3, line: 1432, type: !98)
!2740 = !DILocation(line: 1432, column: 44, scope: !2738)
!2741 = !DILocalVariable(name: "u1", arg: 2, scope: !2738, file: !3, line: 1433, type: !98)
!2742 = !DILocation(line: 1433, column: 12, scope: !2738)
!2743 = !DILocalVariable(name: "twiddle", arg: 3, scope: !2738, file: !3, line: 1434, type: !106)
!2744 = !DILocation(line: 1434, column: 10, scope: !2738)
!2745 = !DILocation(line: 1434, column: 20, scope: !2738)
!2746 = !DILocation(line: 1443, column: 1, scope: !2738)
!2747 = distinct !DISubprogram(name: "init_ui_gpu_kernel", linkageName: "_Z18init_ui_gpu_kernelP8dcomplexS0_Pd", scope: !3, file: !3, line: 1542, type: !1842, scopeLine: 1544, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !1058)
!2748 = !DILocalVariable(name: "u0", arg: 1, scope: !2747, file: !3, line: 1542, type: !98)
!2749 = !DILocation(line: 1542, column: 45, scope: !2747)
!2750 = !DILocalVariable(name: "u1", arg: 2, scope: !2747, file: !3, line: 1543, type: !98)
!2751 = !DILocation(line: 1543, column: 12, scope: !2747)
!2752 = !DILocalVariable(name: "twiddle", arg: 3, scope: !2747, file: !3, line: 1544, type: !106)
!2753 = !DILocation(line: 1544, column: 10, scope: !2747)
!2754 = !DILocation(line: 1544, column: 20, scope: !2747)
!2755 = !DILocation(line: 1554, column: 1, scope: !2747)
!2756 = distinct !DISubprogram(name: "dim3", linkageName: "_ZN4dim3C2Ejjj", scope: !2758, file: !2757, line: 421, type: !2764, scopeLine: 421, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, declaration: !2763, retainedNodes: !1058)
!2757 = !DIFile(filename: "/usr/local/cuda/include/vector_types.h", directory: "")
!2758 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "dim3", file: !2757, line: 417, size: 96, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !2759, identifier: "_ZTS4dim3")
!2759 = !{!2760, !2761, !2762, !2763, !2767, !2776}
!2760 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !2758, file: !2757, line: 419, baseType: !7, size: 32)
!2761 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !2758, file: !2757, line: 419, baseType: !7, size: 32, offset: 32)
!2762 = !DIDerivedType(tag: DW_TAG_member, name: "z", scope: !2758, file: !2757, line: 419, baseType: !7, size: 32, offset: 64)
!2763 = !DISubprogram(name: "dim3", scope: !2758, file: !2757, line: 421, type: !2764, scopeLine: 421, flags: DIFlagPrototyped, spFlags: 0)
!2764 = !DISubroutineType(types: !2765)
!2765 = !{null, !2766, !7, !7, !7}
!2766 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !2758, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!2767 = !DISubprogram(name: "dim3", scope: !2758, file: !2757, line: 422, type: !2768, scopeLine: 422, flags: DIFlagPrototyped, spFlags: 0)
!2768 = !DISubroutineType(types: !2769)
!2769 = !{null, !2766, !2770}
!2770 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint3", file: !2757, line: 383, baseType: !2771)
!2771 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "uint3", file: !2757, line: 190, size: 96, flags: DIFlagTypePassByValue, elements: !2772, identifier: "_ZTS5uint3")
!2772 = !{!2773, !2774, !2775}
!2773 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !2771, file: !2757, line: 192, baseType: !7, size: 32)
!2774 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !2771, file: !2757, line: 192, baseType: !7, size: 32, offset: 32)
!2775 = !DIDerivedType(tag: DW_TAG_member, name: "z", scope: !2771, file: !2757, line: 192, baseType: !7, size: 32, offset: 64)
!2776 = !DISubprogram(name: "operator uint3", linkageName: "_ZN4dim3cv5uint3Ev", scope: !2758, file: !2757, line: 423, type: !2777, scopeLine: 423, flags: DIFlagPrototyped, spFlags: 0)
!2777 = !DISubroutineType(types: !2778)
!2778 = !{!2770, !2766}
!2779 = !DILocalVariable(name: "this", arg: 1, scope: !2756, type: !2780, flags: DIFlagArtificial | DIFlagObjectPointer)
!2780 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !2758, size: 64)
!2781 = !DILocation(line: 0, scope: !2756)
!2782 = !DILocalVariable(name: "vx", arg: 2, scope: !2756, file: !2757, line: 421, type: !7)
!2783 = !DILocation(line: 421, column: 43, scope: !2756)
!2784 = !DILocalVariable(name: "vy", arg: 3, scope: !2756, file: !2757, line: 421, type: !7)
!2785 = !DILocation(line: 421, column: 64, scope: !2756)
!2786 = !DILocalVariable(name: "vz", arg: 4, scope: !2756, file: !2757, line: 421, type: !7)
!2787 = !DILocation(line: 421, column: 85, scope: !2756)
!2788 = !DILocation(line: 421, column: 95, scope: !2756)
!2789 = !DILocation(line: 421, column: 97, scope: !2756)
!2790 = !DILocation(line: 421, column: 102, scope: !2756)
!2791 = !DILocation(line: 421, column: 104, scope: !2756)
!2792 = !DILocation(line: 421, column: 109, scope: !2756)
!2793 = !DILocation(line: 421, column: 111, scope: !2756)
!2794 = !DILocation(line: 421, column: 116, scope: !2756)
!2795 = distinct !DISubprogram(name: "ipow46", linkageName: "_ZL6ipow46diPd", scope: !3, file: !3, line: 1556, type: !2796, scopeLine: 1558, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1058)
!2796 = !DISubroutineType(types: !2797)
!2797 = !{null, !104, !97, !106}
!2798 = !DILocalVariable(name: "a", arg: 1, scope: !2795, file: !3, line: 1556, type: !104)
!2799 = !DILocation(line: 1556, column: 27, scope: !2795)
!2800 = !DILocalVariable(name: "exponent", arg: 2, scope: !2795, file: !3, line: 1557, type: !97)
!2801 = !DILocation(line: 1557, column: 7, scope: !2795)
!2802 = !DILocalVariable(name: "result", arg: 3, scope: !2795, file: !3, line: 1558, type: !106)
!2803 = !DILocation(line: 1558, column: 11, scope: !2795)
!2804 = !DILocalVariable(name: "q", scope: !2795, file: !3, line: 1559, type: !104)
!2805 = !DILocation(line: 1559, column: 9, scope: !2795)
!2806 = !DILocalVariable(name: "r", scope: !2795, file: !3, line: 1559, type: !104)
!2807 = !DILocation(line: 1559, column: 12, scope: !2795)
!2808 = !DILocalVariable(name: "n", scope: !2795, file: !3, line: 1560, type: !97)
!2809 = !DILocation(line: 1560, column: 6, scope: !2795)
!2810 = !DILocalVariable(name: "n2", scope: !2795, file: !3, line: 1560, type: !97)
!2811 = !DILocation(line: 1560, column: 9, scope: !2795)
!2812 = !DILocation(line: 1568, column: 3, scope: !2795)
!2813 = !DILocation(line: 1568, column: 10, scope: !2795)
!2814 = !DILocation(line: 1569, column: 5, scope: !2815)
!2815 = distinct !DILexicalBlock(scope: !2795, file: !3, line: 1569, column: 5)
!2816 = !DILocation(line: 1569, column: 13, scope: !2815)
!2817 = !DILocation(line: 1569, column: 5, scope: !2795)
!2818 = !DILocation(line: 1569, column: 18, scope: !2819)
!2819 = distinct !DILexicalBlock(scope: !2815, file: !3, line: 1569, column: 17)
!2820 = !DILocation(line: 1570, column: 6, scope: !2795)
!2821 = !DILocation(line: 1570, column: 4, scope: !2795)
!2822 = !DILocation(line: 1571, column: 4, scope: !2795)
!2823 = !DILocation(line: 1572, column: 6, scope: !2795)
!2824 = !DILocation(line: 1572, column: 4, scope: !2795)
!2825 = !DILocation(line: 1573, column: 2, scope: !2795)
!2826 = !DILocation(line: 1573, column: 8, scope: !2795)
!2827 = !DILocation(line: 1573, column: 9, scope: !2795)
!2828 = !DILocation(line: 1574, column: 8, scope: !2829)
!2829 = distinct !DILexicalBlock(scope: !2795, file: !3, line: 1573, column: 12)
!2830 = !DILocation(line: 1574, column: 9, scope: !2829)
!2831 = !DILocation(line: 1574, column: 6, scope: !2829)
!2832 = !DILocation(line: 1575, column: 6, scope: !2833)
!2833 = distinct !DILexicalBlock(scope: !2829, file: !3, line: 1575, column: 6)
!2834 = !DILocation(line: 1575, column: 8, scope: !2833)
!2835 = !DILocation(line: 1575, column: 12, scope: !2833)
!2836 = !DILocation(line: 1575, column: 10, scope: !2833)
!2837 = !DILocation(line: 1575, column: 6, scope: !2829)
!2838 = !DILocation(line: 1576, column: 15, scope: !2839)
!2839 = distinct !DILexicalBlock(scope: !2833, file: !3, line: 1575, column: 14)
!2840 = !DILocation(line: 1576, column: 4, scope: !2839)
!2841 = !DILocation(line: 1577, column: 8, scope: !2839)
!2842 = !DILocation(line: 1577, column: 6, scope: !2839)
!2843 = !DILocation(line: 1578, column: 3, scope: !2839)
!2844 = !DILocation(line: 1579, column: 15, scope: !2845)
!2845 = distinct !DILexicalBlock(scope: !2833, file: !3, line: 1578, column: 8)
!2846 = !DILocation(line: 1579, column: 4, scope: !2845)
!2847 = !DILocation(line: 1580, column: 8, scope: !2845)
!2848 = !DILocation(line: 1580, column: 9, scope: !2845)
!2849 = !DILocation(line: 1580, column: 6, scope: !2845)
!2850 = distinct !{!2850, !2825, !2851}
!2851 = !DILocation(line: 1582, column: 2, scope: !2795)
!2852 = !DILocation(line: 1583, column: 13, scope: !2795)
!2853 = !DILocation(line: 1583, column: 2, scope: !2795)
!2854 = !DILocation(line: 1584, column: 12, scope: !2795)
!2855 = !DILocation(line: 1584, column: 3, scope: !2795)
!2856 = !DILocation(line: 1584, column: 10, scope: !2795)
!2857 = !DILocation(line: 1585, column: 1, scope: !2795)
!2858 = distinct !DISubprogram(name: "cffts1_gpu", linkageName: "_ZL10cffts1_gpuiP8dcomplexS0_S0_S0_S0_", scope: !3, file: !3, line: 713, type: !2859, scopeLine: 718, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1058)
!2859 = !DISubroutineType(types: !2860)
!2860 = !{null, !2647, !98, !98, !98, !98, !98}
!2861 = !DILocalVariable(name: "is", arg: 1, scope: !2858, file: !3, line: 713, type: !2647)
!2862 = !DILocation(line: 713, column: 34, scope: !2858)
!2863 = !DILocalVariable(name: "u", arg: 2, scope: !2858, file: !3, line: 714, type: !98)
!2864 = !DILocation(line: 714, column: 12, scope: !2858)
!2865 = !DILocalVariable(name: "x_in", arg: 3, scope: !2858, file: !3, line: 715, type: !98)
!2866 = !DILocation(line: 715, column: 12, scope: !2858)
!2867 = !DILocalVariable(name: "x_out", arg: 4, scope: !2858, file: !3, line: 716, type: !98)
!2868 = !DILocation(line: 716, column: 12, scope: !2858)
!2869 = !DILocalVariable(name: "y0", arg: 5, scope: !2858, file: !3, line: 717, type: !98)
!2870 = !DILocation(line: 717, column: 12, scope: !2858)
!2871 = !DILocalVariable(name: "y1", arg: 6, scope: !2858, file: !3, line: 718, type: !98)
!2872 = !DILocation(line: 718, column: 12, scope: !2858)
!2873 = !DILocation(line: 722, column: 24, scope: !2858)
!2874 = !DILocation(line: 723, column: 3, scope: !2858)
!2875 = !DILocation(line: 722, column: 21, scope: !2858)
!2876 = !DILocation(line: 722, column: 2, scope: !2858)
!2877 = !DILocation(line: 723, column: 34, scope: !2858)
!2878 = !DILocation(line: 724, column: 5, scope: !2858)
!2879 = !DILocation(line: 725, column: 2, scope: !2858)
!2880 = !DILocation(line: 733, column: 24, scope: !2858)
!2881 = !DILocation(line: 734, column: 3, scope: !2858)
!2882 = !DILocation(line: 733, column: 21, scope: !2858)
!2883 = !DILocation(line: 733, column: 2, scope: !2858)
!2884 = !DILocation(line: 734, column: 34, scope: !2858)
!2885 = !DILocation(line: 735, column: 5, scope: !2858)
!2886 = !DILocation(line: 736, column: 5, scope: !2858)
!2887 = !DILocation(line: 737, column: 5, scope: !2858)
!2888 = !DILocation(line: 738, column: 2, scope: !2858)
!2889 = !DILocation(line: 746, column: 24, scope: !2858)
!2890 = !DILocation(line: 747, column: 3, scope: !2858)
!2891 = !DILocation(line: 746, column: 21, scope: !2858)
!2892 = !DILocation(line: 746, column: 2, scope: !2858)
!2893 = !DILocation(line: 747, column: 34, scope: !2858)
!2894 = !DILocation(line: 748, column: 5, scope: !2858)
!2895 = !DILocation(line: 749, column: 2, scope: !2858)
!2896 = !DILocation(line: 753, column: 1, scope: !2858)
!2897 = distinct !DISubprogram(name: "cffts2_gpu", linkageName: "_ZL10cffts2_gpuiP8dcomplexS0_S0_S0_S0_", scope: !3, file: !3, line: 908, type: !2898, scopeLine: 913, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1058)
!2898 = !DISubroutineType(types: !2899)
!2899 = !{null, !97, !98, !98, !98, !98, !98}
!2900 = !DILocalVariable(name: "is", arg: 1, scope: !2897, file: !3, line: 908, type: !97)
!2901 = !DILocation(line: 908, column: 28, scope: !2897)
!2902 = !DILocalVariable(name: "u", arg: 2, scope: !2897, file: !3, line: 909, type: !98)
!2903 = !DILocation(line: 909, column: 12, scope: !2897)
!2904 = !DILocalVariable(name: "x_in", arg: 3, scope: !2897, file: !3, line: 910, type: !98)
!2905 = !DILocation(line: 910, column: 12, scope: !2897)
!2906 = !DILocalVariable(name: "x_out", arg: 4, scope: !2897, file: !3, line: 911, type: !98)
!2907 = !DILocation(line: 911, column: 12, scope: !2897)
!2908 = !DILocalVariable(name: "y0", arg: 5, scope: !2897, file: !3, line: 912, type: !98)
!2909 = !DILocation(line: 912, column: 12, scope: !2897)
!2910 = !DILocalVariable(name: "y1", arg: 6, scope: !2897, file: !3, line: 913, type: !98)
!2911 = !DILocation(line: 913, column: 12, scope: !2897)
!2912 = !DILocation(line: 917, column: 24, scope: !2897)
!2913 = !DILocation(line: 918, column: 3, scope: !2897)
!2914 = !DILocation(line: 917, column: 21, scope: !2897)
!2915 = !DILocation(line: 917, column: 2, scope: !2897)
!2916 = !DILocation(line: 918, column: 34, scope: !2897)
!2917 = !DILocation(line: 919, column: 5, scope: !2897)
!2918 = !DILocation(line: 920, column: 2, scope: !2897)
!2919 = !DILocation(line: 928, column: 24, scope: !2897)
!2920 = !DILocation(line: 929, column: 3, scope: !2897)
!2921 = !DILocation(line: 928, column: 21, scope: !2897)
!2922 = !DILocation(line: 928, column: 2, scope: !2897)
!2923 = !DILocation(line: 929, column: 34, scope: !2897)
!2924 = !DILocation(line: 930, column: 5, scope: !2897)
!2925 = !DILocation(line: 931, column: 5, scope: !2897)
!2926 = !DILocation(line: 932, column: 5, scope: !2897)
!2927 = !DILocation(line: 933, column: 2, scope: !2897)
!2928 = !DILocation(line: 941, column: 24, scope: !2897)
!2929 = !DILocation(line: 942, column: 3, scope: !2897)
!2930 = !DILocation(line: 941, column: 21, scope: !2897)
!2931 = !DILocation(line: 941, column: 2, scope: !2897)
!2932 = !DILocation(line: 942, column: 34, scope: !2897)
!2933 = !DILocation(line: 943, column: 5, scope: !2897)
!2934 = !DILocation(line: 944, column: 2, scope: !2897)
!2935 = !DILocation(line: 948, column: 1, scope: !2897)
!2936 = distinct !DISubprogram(name: "cffts3_gpu", linkageName: "_ZL10cffts3_gpuiP8dcomplexS0_S0_S0_S0_", scope: !3, file: !3, line: 1099, type: !2898, scopeLine: 1104, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1058)
!2937 = !DILocalVariable(name: "is", arg: 1, scope: !2936, file: !3, line: 1099, type: !97)
!2938 = !DILocation(line: 1099, column: 28, scope: !2936)
!2939 = !DILocalVariable(name: "u", arg: 2, scope: !2936, file: !3, line: 1100, type: !98)
!2940 = !DILocation(line: 1100, column: 12, scope: !2936)
!2941 = !DILocalVariable(name: "x_in", arg: 3, scope: !2936, file: !3, line: 1101, type: !98)
!2942 = !DILocation(line: 1101, column: 12, scope: !2936)
!2943 = !DILocalVariable(name: "x_out", arg: 4, scope: !2936, file: !3, line: 1102, type: !98)
!2944 = !DILocation(line: 1102, column: 12, scope: !2936)
!2945 = !DILocalVariable(name: "y0", arg: 5, scope: !2936, file: !3, line: 1103, type: !98)
!2946 = !DILocation(line: 1103, column: 12, scope: !2936)
!2947 = !DILocalVariable(name: "y1", arg: 6, scope: !2936, file: !3, line: 1104, type: !98)
!2948 = !DILocation(line: 1104, column: 12, scope: !2936)
!2949 = !DILocation(line: 1108, column: 24, scope: !2936)
!2950 = !DILocation(line: 1109, column: 3, scope: !2936)
!2951 = !DILocation(line: 1108, column: 21, scope: !2936)
!2952 = !DILocation(line: 1108, column: 2, scope: !2936)
!2953 = !DILocation(line: 1109, column: 34, scope: !2936)
!2954 = !DILocation(line: 1110, column: 5, scope: !2936)
!2955 = !DILocation(line: 1111, column: 2, scope: !2936)
!2956 = !DILocation(line: 1119, column: 24, scope: !2936)
!2957 = !DILocation(line: 1120, column: 3, scope: !2936)
!2958 = !DILocation(line: 1119, column: 21, scope: !2936)
!2959 = !DILocation(line: 1119, column: 2, scope: !2936)
!2960 = !DILocation(line: 1120, column: 34, scope: !2936)
!2961 = !DILocation(line: 1121, column: 5, scope: !2936)
!2962 = !DILocation(line: 1122, column: 5, scope: !2936)
!2963 = !DILocation(line: 1123, column: 5, scope: !2936)
!2964 = !DILocation(line: 1124, column: 2, scope: !2936)
!2965 = !DILocation(line: 1132, column: 24, scope: !2936)
!2966 = !DILocation(line: 1133, column: 3, scope: !2936)
!2967 = !DILocation(line: 1132, column: 21, scope: !2936)
!2968 = !DILocation(line: 1132, column: 2, scope: !2936)
!2969 = !DILocation(line: 1133, column: 34, scope: !2936)
!2970 = !DILocation(line: 1134, column: 5, scope: !2936)
!2971 = !DILocation(line: 1135, column: 2, scope: !2936)
!2972 = !DILocation(line: 1139, column: 1, scope: !2936)
!2973 = distinct !DISubprogram(name: "ilog2", linkageName: "_ZL5ilog2i", scope: !3, file: !3, line: 1498, type: !304, scopeLine: 1498, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1058)
!2974 = !DILocalVariable(name: "n", arg: 1, scope: !2973, file: !3, line: 1498, type: !97)
!2975 = !DILocation(line: 1498, column: 22, scope: !2973)
!2976 = !DILocalVariable(name: "nn", scope: !2973, file: !3, line: 1499, type: !97)
!2977 = !DILocation(line: 1499, column: 6, scope: !2973)
!2978 = !DILocalVariable(name: "lg", scope: !2973, file: !3, line: 1499, type: !97)
!2979 = !DILocation(line: 1499, column: 10, scope: !2973)
!2980 = !DILocation(line: 1500, column: 5, scope: !2981)
!2981 = distinct !DILexicalBlock(scope: !2973, file: !3, line: 1500, column: 5)
!2982 = !DILocation(line: 1500, column: 6, scope: !2981)
!2983 = !DILocation(line: 1500, column: 5, scope: !2973)
!2984 = !DILocation(line: 1501, column: 3, scope: !2985)
!2985 = distinct !DILexicalBlock(scope: !2981, file: !3, line: 1500, column: 10)
!2986 = !DILocation(line: 1503, column: 5, scope: !2973)
!2987 = !DILocation(line: 1504, column: 5, scope: !2973)
!2988 = !DILocation(line: 1505, column: 2, scope: !2973)
!2989 = !DILocation(line: 1505, column: 8, scope: !2973)
!2990 = !DILocation(line: 1505, column: 11, scope: !2973)
!2991 = !DILocation(line: 1505, column: 10, scope: !2973)
!2992 = !DILocation(line: 1506, column: 8, scope: !2993)
!2993 = distinct !DILexicalBlock(scope: !2973, file: !3, line: 1505, column: 13)
!2994 = !DILocation(line: 1506, column: 11, scope: !2993)
!2995 = !DILocation(line: 1506, column: 6, scope: !2993)
!2996 = !DILocation(line: 1507, column: 5, scope: !2993)
!2997 = distinct !{!2997, !2988, !2998}
!2998 = !DILocation(line: 1508, column: 2, scope: !2973)
!2999 = !DILocation(line: 1509, column: 9, scope: !2973)
!3000 = !DILocation(line: 1509, column: 2, scope: !2973)
!3001 = !DILocation(line: 1510, column: 1, scope: !2973)
!3002 = distinct !DISubprogram(name: "cudaMalloc<dcomplex>", linkageName: "_ZL10cudaMallocI8dcomplexE9cudaErrorPPT_m", scope: !3003, file: !3003, line: 490, type: !3004, scopeLine: 494, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, templateParams: !3009, retainedNodes: !1058)
!3003 = !DIFile(filename: "/usr/local/cuda/include/cuda_runtime.h", directory: "")
!3004 = !DISubroutineType(types: !3005)
!3005 = !{!3006, !3007, !131}
!3006 = !DIDerivedType(tag: DW_TAG_typedef, name: "cudaError_t", file: !6, line: 1419, baseType: !14)
!3007 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !3008, size: 64)
!3008 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !101, size: 64)
!3009 = !{!3010}
!3010 = !DITemplateTypeParameter(name: "T", type: !101)
!3011 = !DILocalVariable(name: "devPtr", arg: 1, scope: !3002, file: !3003, line: 491, type: !3007)
!3012 = !DILocation(line: 491, column: 12, scope: !3002)
!3013 = !DILocalVariable(name: "size", arg: 2, scope: !3002, file: !3003, line: 492, type: !131)
!3014 = !DILocation(line: 492, column: 12, scope: !3002)
!3015 = !DILocation(line: 495, column: 38, scope: !3002)
!3016 = !DILocation(line: 495, column: 23, scope: !3002)
!3017 = !DILocation(line: 495, column: 46, scope: !3002)
!3018 = !DILocation(line: 495, column: 10, scope: !3002)
!3019 = !DILocation(line: 495, column: 3, scope: !3002)
!3020 = distinct !DISubprogram(name: "cudaMalloc<double>", linkageName: "_ZL10cudaMallocIdE9cudaErrorPPT_m", scope: !3003, file: !3003, line: 490, type: !3021, scopeLine: 494, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, templateParams: !3024, retainedNodes: !1058)
!3021 = !DISubroutineType(types: !3022)
!3022 = !{!3006, !3023, !131}
!3023 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !106, size: 64)
!3024 = !{!3025}
!3025 = !DITemplateTypeParameter(name: "T", type: !104)
!3026 = !DILocalVariable(name: "devPtr", arg: 1, scope: !3020, file: !3003, line: 491, type: !3023)
!3027 = !DILocation(line: 491, column: 12, scope: !3020)
!3028 = !DILocalVariable(name: "size", arg: 2, scope: !3020, file: !3003, line: 492, type: !131)
!3029 = !DILocation(line: 492, column: 12, scope: !3020)
!3030 = !DILocation(line: 495, column: 38, scope: !3020)
!3031 = !DILocation(line: 495, column: 23, scope: !3020)
!3032 = !DILocation(line: 495, column: 46, scope: !3020)
!3033 = !DILocation(line: 495, column: 10, scope: !3020)
!3034 = !DILocation(line: 495, column: 3, scope: !3020)
!3035 = distinct !DISubprogram(name: "dcomplex_div", linkageName: "_ZL12dcomplex_div8dcomplexS_", scope: !100, file: !100, line: 106, type: !3036, scopeLine: 106, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !1058)
!3036 = !DISubroutineType(types: !3037)
!3037 = !{!99, !99, !99}
!3038 = !DILocalVariable(name: "z1", arg: 1, scope: !3035, file: !100, line: 106, type: !99)
!3039 = !DILocation(line: 106, column: 46, scope: !3035)
!3040 = !DILocalVariable(name: "z2", arg: 2, scope: !3035, file: !100, line: 106, type: !99)
!3041 = !DILocation(line: 106, column: 59, scope: !3035)
!3042 = !DILocalVariable(name: "a", scope: !3035, file: !100, line: 107, type: !104)
!3043 = !DILocation(line: 107, column: 9, scope: !3035)
!3044 = !DILocation(line: 107, column: 16, scope: !3035)
!3045 = !DILocalVariable(name: "b", scope: !3035, file: !100, line: 108, type: !104)
!3046 = !DILocation(line: 108, column: 9, scope: !3035)
!3047 = !DILocation(line: 108, column: 16, scope: !3035)
!3048 = !DILocalVariable(name: "c", scope: !3035, file: !100, line: 109, type: !104)
!3049 = !DILocation(line: 109, column: 9, scope: !3035)
!3050 = !DILocation(line: 109, column: 16, scope: !3035)
!3051 = !DILocalVariable(name: "d", scope: !3035, file: !100, line: 110, type: !104)
!3052 = !DILocation(line: 110, column: 9, scope: !3035)
!3053 = !DILocation(line: 110, column: 16, scope: !3035)
!3054 = !DILocalVariable(name: "divisor", scope: !3035, file: !100, line: 111, type: !104)
!3055 = !DILocation(line: 111, column: 9, scope: !3035)
!3056 = !DILocation(line: 111, column: 19, scope: !3035)
!3057 = !DILocation(line: 111, column: 21, scope: !3035)
!3058 = !DILocation(line: 111, column: 20, scope: !3035)
!3059 = !DILocation(line: 111, column: 25, scope: !3035)
!3060 = !DILocation(line: 111, column: 27, scope: !3035)
!3061 = !DILocation(line: 111, column: 26, scope: !3035)
!3062 = !DILocation(line: 111, column: 23, scope: !3035)
!3063 = !DILocalVariable(name: "real", scope: !3035, file: !100, line: 112, type: !104)
!3064 = !DILocation(line: 112, column: 9, scope: !3035)
!3065 = !DILocation(line: 112, column: 17, scope: !3035)
!3066 = !DILocation(line: 112, column: 19, scope: !3035)
!3067 = !DILocation(line: 112, column: 18, scope: !3035)
!3068 = !DILocation(line: 112, column: 23, scope: !3035)
!3069 = !DILocation(line: 112, column: 25, scope: !3035)
!3070 = !DILocation(line: 112, column: 24, scope: !3035)
!3071 = !DILocation(line: 112, column: 21, scope: !3035)
!3072 = !DILocation(line: 112, column: 30, scope: !3035)
!3073 = !DILocation(line: 112, column: 28, scope: !3035)
!3074 = !DILocalVariable(name: "imag", scope: !3035, file: !100, line: 113, type: !104)
!3075 = !DILocation(line: 113, column: 9, scope: !3035)
!3076 = !DILocation(line: 113, column: 17, scope: !3035)
!3077 = !DILocation(line: 113, column: 19, scope: !3035)
!3078 = !DILocation(line: 113, column: 18, scope: !3035)
!3079 = !DILocation(line: 113, column: 23, scope: !3035)
!3080 = !DILocation(line: 113, column: 25, scope: !3035)
!3081 = !DILocation(line: 113, column: 24, scope: !3035)
!3082 = !DILocation(line: 113, column: 21, scope: !3035)
!3083 = !DILocation(line: 113, column: 30, scope: !3035)
!3084 = !DILocation(line: 113, column: 28, scope: !3035)
!3085 = !DILocalVariable(name: "result", scope: !3035, file: !100, line: 114, type: !99)
!3086 = !DILocation(line: 114, column: 11, scope: !3035)
!3087 = !DILocation(line: 114, column: 30, scope: !3035)
!3088 = !DILocation(line: 114, column: 31, scope: !3035)
!3089 = !DILocation(line: 114, column: 37, scope: !3035)
!3090 = !DILocation(line: 115, column: 2, scope: !3035)
