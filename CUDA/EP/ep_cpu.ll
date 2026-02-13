; ModuleID = 'ep_cpu.bc'
source_filename = "llvm-link-cudafe"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.cudaDeviceProp = type { [256 x i8], i64, i64, i32, i32, i64, i32, [3 x i32], [3 x i32], i32, i64, i32, i32, i64, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, [2 x i32], [2 x i32], [3 x i32], [2 x i32], [3 x i32], [3 x i32], i32, [2 x i32], [3 x i32], [2 x i32], i32, [2 x i32], [3 x i32], [2 x i32], [3 x i32], i32, [2 x i32], i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i64, i32, i32, i32, i32 }
%struct.dim3 = type { i32, i32, i32 }

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

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #0

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: convergent noinline nounwind
define dso_local double @_Z13randlc_devicePdd(double* %x, double %a) #2 !dbg !1046 {
entry:
  call void @llvm.dbg.value(metadata double* %x, metadata !1049, metadata !DIExpression()), !dbg !1050
  call void @llvm.dbg.value(metadata double %a, metadata !1051, metadata !DIExpression()), !dbg !1050
  %mul = fmul contract double 0x3E80000000000000, %a, !dbg !1052
  call void @llvm.dbg.value(metadata double %mul, metadata !1053, metadata !DIExpression()), !dbg !1050
  %conv = fptosi double %mul to i32, !dbg !1054
  %conv1 = sitofp i32 %conv to double, !dbg !1055
  call void @llvm.dbg.value(metadata double %conv1, metadata !1056, metadata !DIExpression()), !dbg !1050
  %mul2 = fmul contract double 0x4160000000000000, %conv1, !dbg !1057
  %sub = fsub contract double %a, %mul2, !dbg !1058
  call void @llvm.dbg.value(metadata double %sub, metadata !1059, metadata !DIExpression()), !dbg !1050
  %0 = load double, double* %x, align 8, !dbg !1060
  %mul3 = fmul contract double 0x3E80000000000000, %0, !dbg !1061
  call void @llvm.dbg.value(metadata double %mul3, metadata !1053, metadata !DIExpression()), !dbg !1050
  %conv4 = fptosi double %mul3 to i32, !dbg !1062
  %conv5 = sitofp i32 %conv4 to double, !dbg !1063
  call void @llvm.dbg.value(metadata double %conv5, metadata !1064, metadata !DIExpression()), !dbg !1050
  %1 = load double, double* %x, align 8, !dbg !1065
  %mul6 = fmul contract double 0x4160000000000000, %conv5, !dbg !1066
  %sub7 = fsub contract double %1, %mul6, !dbg !1067
  call void @llvm.dbg.value(metadata double %sub7, metadata !1068, metadata !DIExpression()), !dbg !1050
  %mul8 = fmul contract double %conv1, %sub7, !dbg !1069
  %mul9 = fmul contract double %sub, %conv5, !dbg !1070
  %add = fadd contract double %mul8, %mul9, !dbg !1071
  call void @llvm.dbg.value(metadata double %add, metadata !1053, metadata !DIExpression()), !dbg !1050
  %mul10 = fmul contract double 0x3E80000000000000, %add, !dbg !1072
  %conv11 = fptosi double %mul10 to i32, !dbg !1073
  %conv12 = sitofp i32 %conv11 to double, !dbg !1074
  call void @llvm.dbg.value(metadata double %conv12, metadata !1075, metadata !DIExpression()), !dbg !1050
  %mul13 = fmul contract double 0x4160000000000000, %conv12, !dbg !1076
  %sub14 = fsub contract double %add, %mul13, !dbg !1077
  call void @llvm.dbg.value(metadata double %sub14, metadata !1078, metadata !DIExpression()), !dbg !1050
  %mul15 = fmul contract double 0x4160000000000000, %sub14, !dbg !1079
  %mul16 = fmul contract double %sub, %sub7, !dbg !1080
  %add17 = fadd contract double %mul15, %mul16, !dbg !1081
  call void @llvm.dbg.value(metadata double %add17, metadata !1082, metadata !DIExpression()), !dbg !1050
  %mul18 = fmul contract double 0x3D10000000000000, %add17, !dbg !1083
  %conv19 = fptosi double %mul18 to i32, !dbg !1084
  %conv20 = sitofp i32 %conv19 to double, !dbg !1085
  call void @llvm.dbg.value(metadata double %conv20, metadata !1086, metadata !DIExpression()), !dbg !1050
  %mul21 = fmul contract double 0x42D0000000000000, %conv20, !dbg !1087
  %sub22 = fsub contract double %add17, %mul21, !dbg !1088
  store double %sub22, double* %x, align 8, !dbg !1089
  %2 = load double, double* %x, align 8, !dbg !1090
  %mul23 = fmul contract double 0x3D10000000000000, %2, !dbg !1091
  ret double %mul23, !dbg !1092
}

; Function Attrs: convergent noinline nounwind
define dso_local void @_Z13vranlc_deviceiPddS_(i32 %n, double* %x_seed, double %a, double* %y) #2 !dbg !1093 {
entry:
  call void @llvm.dbg.value(metadata i32 %n, metadata !1096, metadata !DIExpression()), !dbg !1097
  call void @llvm.dbg.value(metadata double* %x_seed, metadata !1098, metadata !DIExpression()), !dbg !1097
  call void @llvm.dbg.value(metadata double %a, metadata !1099, metadata !DIExpression()), !dbg !1097
  call void @llvm.dbg.value(metadata double* %y, metadata !1100, metadata !DIExpression()), !dbg !1097
  %mul = fmul contract double 0x3E80000000000000, %a, !dbg !1101
  call void @llvm.dbg.value(metadata double %mul, metadata !1102, metadata !DIExpression()), !dbg !1097
  %conv = fptosi double %mul to i32, !dbg !1103
  %conv1 = sitofp i32 %conv to double, !dbg !1104
  call void @llvm.dbg.value(metadata double %conv1, metadata !1105, metadata !DIExpression()), !dbg !1097
  %mul2 = fmul contract double 0x4160000000000000, %conv1, !dbg !1106
  %sub = fsub contract double %a, %mul2, !dbg !1107
  call void @llvm.dbg.value(metadata double %sub, metadata !1108, metadata !DIExpression()), !dbg !1097
  %0 = load double, double* %x_seed, align 8, !dbg !1109
  call void @llvm.dbg.value(metadata double %0, metadata !1110, metadata !DIExpression()), !dbg !1097
  call void @llvm.dbg.value(metadata i32 0, metadata !1111, metadata !DIExpression()), !dbg !1097
  %1 = sext i32 %n to i64, !dbg !1112
  br label %for.cond, !dbg !1112

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ], !dbg !1114
  %x.0 = phi double [ %0, %entry ], [ %sub22, %for.inc ], !dbg !1097
  call void @llvm.dbg.value(metadata double %x.0, metadata !1110, metadata !DIExpression()), !dbg !1097
  call void @llvm.dbg.value(metadata i64 %indvars.iv, metadata !1111, metadata !DIExpression()), !dbg !1097
  %cmp = icmp slt i64 %indvars.iv, %1, !dbg !1115
  br i1 %cmp, label %for.body, label %for.end, !dbg !1117

for.body:                                         ; preds = %for.cond
  %mul3 = fmul contract double 0x3E80000000000000, %x.0, !dbg !1118
  call void @llvm.dbg.value(metadata double %mul3, metadata !1102, metadata !DIExpression()), !dbg !1097
  %conv4 = fptosi double %mul3 to i32, !dbg !1120
  %conv5 = sitofp i32 %conv4 to double, !dbg !1121
  call void @llvm.dbg.value(metadata double %conv5, metadata !1122, metadata !DIExpression()), !dbg !1097
  %mul6 = fmul contract double 0x4160000000000000, %conv5, !dbg !1123
  %sub7 = fsub contract double %x.0, %mul6, !dbg !1124
  call void @llvm.dbg.value(metadata double %sub7, metadata !1125, metadata !DIExpression()), !dbg !1097
  %mul8 = fmul contract double %conv1, %sub7, !dbg !1126
  %mul9 = fmul contract double %sub, %conv5, !dbg !1127
  %add = fadd contract double %mul8, %mul9, !dbg !1128
  call void @llvm.dbg.value(metadata double %add, metadata !1102, metadata !DIExpression()), !dbg !1097
  %mul10 = fmul contract double 0x3E80000000000000, %add, !dbg !1129
  %conv11 = fptosi double %mul10 to i32, !dbg !1130
  %conv12 = sitofp i32 %conv11 to double, !dbg !1131
  call void @llvm.dbg.value(metadata double %conv12, metadata !1132, metadata !DIExpression()), !dbg !1097
  %mul13 = fmul contract double 0x4160000000000000, %conv12, !dbg !1133
  %sub14 = fsub contract double %add, %mul13, !dbg !1134
  call void @llvm.dbg.value(metadata double %sub14, metadata !1135, metadata !DIExpression()), !dbg !1097
  %mul15 = fmul contract double 0x4160000000000000, %sub14, !dbg !1136
  %mul16 = fmul contract double %sub, %sub7, !dbg !1137
  %add17 = fadd contract double %mul15, %mul16, !dbg !1138
  call void @llvm.dbg.value(metadata double %add17, metadata !1139, metadata !DIExpression()), !dbg !1097
  %mul18 = fmul contract double 0x3D10000000000000, %add17, !dbg !1140
  %conv19 = fptosi double %mul18 to i32, !dbg !1141
  %conv20 = sitofp i32 %conv19 to double, !dbg !1142
  call void @llvm.dbg.value(metadata double %conv20, metadata !1143, metadata !DIExpression()), !dbg !1097
  %mul21 = fmul contract double 0x42D0000000000000, %conv20, !dbg !1144
  %sub22 = fsub contract double %add17, %mul21, !dbg !1145
  call void @llvm.dbg.value(metadata double %sub22, metadata !1110, metadata !DIExpression()), !dbg !1097
  %mul23 = fmul contract double 0x3D10000000000000, %sub22, !dbg !1146
  %arrayidx = getelementptr inbounds double, double* %y, i64 %indvars.iv, !dbg !1147
  store double %mul23, double* %arrayidx, align 8, !dbg !1148
  br label %for.inc, !dbg !1149

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !1150
  call void @llvm.dbg.value(metadata i32 undef, metadata !1111, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !1097
  br label %for.cond, !dbg !1151, !llvm.loop !1152

for.end:                                          ; preds = %for.cond
  %x.0.lcssa = phi double [ %x.0, %for.cond ], !dbg !1097
  call void @llvm.dbg.value(metadata double %x.0.lcssa, metadata !1110, metadata !DIExpression()), !dbg !1097
  store double %x.0.lcssa, double* %x_seed, align 8, !dbg !1154
  ret void, !dbg !1155
}

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.d2i.hi(double) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.d2i.lo(double) #1

; Function Attrs: nounwind readnone
declare double @llvm.nvvm.lohi.i2d(i32, i32) #1

; Function Attrs: nounwind readnone
declare double @llvm.nvvm.fma.rn.d(double, double, double) #1

; Function Attrs: nounwind readnone
declare double @llvm.nvvm.fabs.d(double) #1

; Function Attrs: nounwind readnone
declare double @llvm.nvvm.sqrt.rn.d(double) #1

; Function Attrs: noinline uwtable
define internal void @_GLOBAL__sub_I_ep.cu() #3 section ".text.startup" !dbg !1156 {
entry:
  call void @__cxx_global_var_init(), !dbg !1158
  ret void
}

; Function Attrs: noinline uwtable
define internal void @__cxx_global_var_init() #3 section ".text.startup" !dbg !1159 {
entry:
  %call = call noalias i8* @malloc(i64 80) #10, !dbg !1160
  %0 = bitcast i8* %call to double*, !dbg !1161
  store double* %0, double** @_ZL1q, align 8, !dbg !1161
  ret void, !dbg !1162
}

; Function Attrs: nounwind
declare dso_local noalias i8* @malloc(i64) #4

; Function Attrs: noinline nounwind uwtable
define dso_local double @_Z6randlcPdd(double* %x, double %a) #5 !dbg !1163 {
entry:
  call void @llvm.dbg.value(metadata double* %x, metadata !1164, metadata !DIExpression()), !dbg !1165
  call void @llvm.dbg.value(metadata double %a, metadata !1166, metadata !DIExpression()), !dbg !1165
  %mul = fmul contract double 0x3E80000000000000, %a, !dbg !1167
  call void @llvm.dbg.value(metadata double %mul, metadata !1168, metadata !DIExpression()), !dbg !1165
  %conv = fptosi double %mul to i32, !dbg !1169
  %conv1 = sitofp i32 %conv to double, !dbg !1170
  call void @llvm.dbg.value(metadata double %conv1, metadata !1171, metadata !DIExpression()), !dbg !1165
  %mul2 = fmul contract double 0x4160000000000000, %conv1, !dbg !1172
  %sub = fsub contract double %a, %mul2, !dbg !1173
  call void @llvm.dbg.value(metadata double %sub, metadata !1174, metadata !DIExpression()), !dbg !1165
  %0 = load double, double* %x, align 8, !dbg !1175
  %mul3 = fmul contract double 0x3E80000000000000, %0, !dbg !1176
  call void @llvm.dbg.value(metadata double %mul3, metadata !1168, metadata !DIExpression()), !dbg !1165
  %conv4 = fptosi double %mul3 to i32, !dbg !1177
  %conv5 = sitofp i32 %conv4 to double, !dbg !1178
  call void @llvm.dbg.value(metadata double %conv5, metadata !1179, metadata !DIExpression()), !dbg !1165
  %1 = load double, double* %x, align 8, !dbg !1180
  %mul6 = fmul contract double 0x4160000000000000, %conv5, !dbg !1181
  %sub7 = fsub contract double %1, %mul6, !dbg !1182
  call void @llvm.dbg.value(metadata double %sub7, metadata !1183, metadata !DIExpression()), !dbg !1165
  %mul8 = fmul contract double %conv1, %sub7, !dbg !1184
  %mul9 = fmul contract double %sub, %conv5, !dbg !1185
  %add = fadd contract double %mul8, %mul9, !dbg !1186
  call void @llvm.dbg.value(metadata double %add, metadata !1168, metadata !DIExpression()), !dbg !1165
  %mul10 = fmul contract double 0x3E80000000000000, %add, !dbg !1187
  %conv11 = fptosi double %mul10 to i32, !dbg !1188
  %conv12 = sitofp i32 %conv11 to double, !dbg !1189
  call void @llvm.dbg.value(metadata double %conv12, metadata !1190, metadata !DIExpression()), !dbg !1165
  %mul13 = fmul contract double 0x4160000000000000, %conv12, !dbg !1191
  %sub14 = fsub contract double %add, %mul13, !dbg !1192
  call void @llvm.dbg.value(metadata double %sub14, metadata !1193, metadata !DIExpression()), !dbg !1165
  %mul15 = fmul contract double 0x4160000000000000, %sub14, !dbg !1194
  %mul16 = fmul contract double %sub, %sub7, !dbg !1195
  %add17 = fadd contract double %mul15, %mul16, !dbg !1196
  call void @llvm.dbg.value(metadata double %add17, metadata !1197, metadata !DIExpression()), !dbg !1165
  %mul18 = fmul contract double 0x3D10000000000000, %add17, !dbg !1198
  %conv19 = fptosi double %mul18 to i32, !dbg !1199
  %conv20 = sitofp i32 %conv19 to double, !dbg !1200
  call void @llvm.dbg.value(metadata double %conv20, metadata !1201, metadata !DIExpression()), !dbg !1165
  %mul21 = fmul contract double 0x42D0000000000000, %conv20, !dbg !1202
  %sub22 = fsub contract double %add17, %mul21, !dbg !1203
  store double %sub22, double* %x, align 8, !dbg !1204
  %2 = load double, double* %x, align 8, !dbg !1205
  %mul23 = fmul contract double 0x3D10000000000000, %2, !dbg !1206
  ret double %mul23, !dbg !1207
}

; Function Attrs: noinline uwtable
define dso_local void @_Z15c_print_resultsPcciiiiddS_iS_S_S_S_S_S_S_S_S_S_S_S_S_S_(i8* %name, i8 signext %class_npb, i32 %n1, i32 %n2, i32 %n3, i32 %niter, double %t, double %mops, i8* %optype, i32 %passed_verification, i8* %npbversion, i8* %compiletime, i8* %compilerversion, i8* %libversion, i8* %cpu_device, i8* %gpu_device, i8* %gpu_config, i8* %cc, i8* %clink, i8* %c_lib, i8* %c_inc, i8* %cflags, i8* %clinkflags, i8* %rand) #3 !dbg !1208 {
entry:
  %size = alloca [16 x i8], align 16
  call void @llvm.dbg.value(metadata i8* %name, metadata !1211, metadata !DIExpression()), !dbg !1212
  call void @llvm.dbg.value(metadata i8 %class_npb, metadata !1213, metadata !DIExpression()), !dbg !1212
  call void @llvm.dbg.value(metadata i32 %n1, metadata !1214, metadata !DIExpression()), !dbg !1212
  call void @llvm.dbg.value(metadata i32 %n2, metadata !1215, metadata !DIExpression()), !dbg !1212
  call void @llvm.dbg.value(metadata i32 %n3, metadata !1216, metadata !DIExpression()), !dbg !1212
  call void @llvm.dbg.value(metadata i32 %niter, metadata !1217, metadata !DIExpression()), !dbg !1212
  call void @llvm.dbg.value(metadata double %t, metadata !1218, metadata !DIExpression()), !dbg !1212
  call void @llvm.dbg.value(metadata double %mops, metadata !1219, metadata !DIExpression()), !dbg !1212
  call void @llvm.dbg.value(metadata i8* %optype, metadata !1220, metadata !DIExpression()), !dbg !1212
  call void @llvm.dbg.value(metadata i32 %passed_verification, metadata !1221, metadata !DIExpression()), !dbg !1212
  call void @llvm.dbg.value(metadata i8* %npbversion, metadata !1222, metadata !DIExpression()), !dbg !1212
  call void @llvm.dbg.value(metadata i8* %compiletime, metadata !1223, metadata !DIExpression()), !dbg !1212
  call void @llvm.dbg.value(metadata i8* %compilerversion, metadata !1224, metadata !DIExpression()), !dbg !1212
  call void @llvm.dbg.value(metadata i8* %libversion, metadata !1225, metadata !DIExpression()), !dbg !1212
  call void @llvm.dbg.value(metadata i8* %cpu_device, metadata !1226, metadata !DIExpression()), !dbg !1212
  call void @llvm.dbg.value(metadata i8* %gpu_device, metadata !1227, metadata !DIExpression()), !dbg !1212
  call void @llvm.dbg.value(metadata i8* %gpu_config, metadata !1228, metadata !DIExpression()), !dbg !1212
  call void @llvm.dbg.value(metadata i8* %cc, metadata !1229, metadata !DIExpression()), !dbg !1212
  call void @llvm.dbg.value(metadata i8* %clink, metadata !1230, metadata !DIExpression()), !dbg !1212
  call void @llvm.dbg.value(metadata i8* %c_lib, metadata !1231, metadata !DIExpression()), !dbg !1212
  call void @llvm.dbg.value(metadata i8* %c_inc, metadata !1232, metadata !DIExpression()), !dbg !1212
  call void @llvm.dbg.value(metadata i8* %cflags, metadata !1233, metadata !DIExpression()), !dbg !1212
  call void @llvm.dbg.value(metadata i8* %clinkflags, metadata !1234, metadata !DIExpression()), !dbg !1212
  call void @llvm.dbg.value(metadata i8* %rand, metadata !1235, metadata !DIExpression()), !dbg !1212
  %call = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([27 x i8], [27 x i8]* @.str, i64 0, i64 0), i8* %name), !dbg !1236
  %conv = sext i8 %class_npb to i32, !dbg !1237
  %call1 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([46 x i8], [46 x i8]* @.str.1, i64 0, i64 0), i32 %conv), !dbg !1238
  %arrayidx = getelementptr inbounds i8, i8* %name, i64 0, !dbg !1239
  %0 = load i8, i8* %arrayidx, align 1, !dbg !1239
  %conv2 = sext i8 %0 to i32, !dbg !1239
  %cmp = icmp eq i32 %conv2, 73, !dbg !1241
  br i1 %cmp, label %land.lhs.true, label %if.else15, !dbg !1242

land.lhs.true:                                    ; preds = %entry
  %arrayidx3 = getelementptr inbounds i8, i8* %name, i64 1, !dbg !1243
  %1 = load i8, i8* %arrayidx3, align 1, !dbg !1243
  %conv4 = sext i8 %1 to i32, !dbg !1243
  %cmp5 = icmp eq i32 %conv4, 83, !dbg !1244
  br i1 %cmp5, label %if.then, label %if.else15, !dbg !1245

if.then:                                          ; preds = %land.lhs.true
  %cmp6 = icmp eq i32 %n3, 0, !dbg !1246
  br i1 %cmp6, label %if.then7, label %if.else, !dbg !1249

if.then7:                                         ; preds = %if.then
  %conv8 = sext i32 %n1 to i64, !dbg !1250
  call void @llvm.dbg.value(metadata i64 %conv8, metadata !1252, metadata !DIExpression()), !dbg !1253
  %cmp9 = icmp ne i32 %n2, 0, !dbg !1254
  br i1 %cmp9, label %if.then10, label %if.end, !dbg !1256

if.then10:                                        ; preds = %if.then7
  %conv11 = sext i32 %n2 to i64, !dbg !1257
  %mul = mul nsw i64 %conv8, %conv11, !dbg !1259
  call void @llvm.dbg.value(metadata i64 %mul, metadata !1252, metadata !DIExpression()), !dbg !1253
  br label %if.end, !dbg !1260

if.end:                                           ; preds = %if.then10, %if.then7
  %nn.0 = phi i64 [ %mul, %if.then10 ], [ %conv8, %if.then7 ], !dbg !1253
  call void @llvm.dbg.value(metadata i64 %nn.0, metadata !1252, metadata !DIExpression()), !dbg !1253
  %call12 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([38 x i8], [38 x i8]* @.str.2, i64 0, i64 0), i64 %nn.0), !dbg !1261
  br label %if.end14, !dbg !1262

if.else:                                          ; preds = %if.then
  %call13 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([44 x i8], [44 x i8]* @.str.3, i64 0, i64 0), i32 %n1, i32 %n2, i32 %n3), !dbg !1263
  br label %if.end14

if.end14:                                         ; preds = %if.else, %if.end
  br label %if.end48, !dbg !1265

if.else15:                                        ; preds = %land.lhs.true, %entry
  call void @llvm.dbg.declare(metadata [16 x i8]* %size, metadata !1266, metadata !DIExpression()), !dbg !1271
  %cmp16 = icmp eq i32 %n2, 0, !dbg !1272
  br i1 %cmp16, label %land.lhs.true17, label %if.else45, !dbg !1274

land.lhs.true17:                                  ; preds = %if.else15
  %cmp18 = icmp eq i32 %n3, 0, !dbg !1275
  br i1 %cmp18, label %if.then19, label %if.else45, !dbg !1276

if.then19:                                        ; preds = %land.lhs.true17
  %arrayidx20 = getelementptr inbounds i8, i8* %name, i64 0, !dbg !1277
  %2 = load i8, i8* %arrayidx20, align 1, !dbg !1277
  %conv21 = sext i8 %2 to i32, !dbg !1277
  %cmp22 = icmp eq i32 %conv21, 69, !dbg !1280
  br i1 %cmp22, label %land.lhs.true23, label %if.else42, !dbg !1281

land.lhs.true23:                                  ; preds = %if.then19
  %arrayidx24 = getelementptr inbounds i8, i8* %name, i64 1, !dbg !1282
  %3 = load i8, i8* %arrayidx24, align 1, !dbg !1282
  %conv25 = sext i8 %3 to i32, !dbg !1282
  %cmp26 = icmp eq i32 %conv25, 80, !dbg !1283
  br i1 %cmp26, label %if.then27, label %if.else42, !dbg !1284

if.then27:                                        ; preds = %land.lhs.true23
  %arraydecay = getelementptr inbounds [16 x i8], [16 x i8]* %size, i64 0, i64 0, !dbg !1285
  %conv28 = sitofp i32 %n1 to double, !dbg !1287
  %call29 = call double @pow(double 2.000000e+00, double %conv28) #10, !dbg !1288
  %call30 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay, i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.4, i64 0, i64 0), double %call29) #10, !dbg !1289
  call void @llvm.dbg.value(metadata i32 14, metadata !1290, metadata !DIExpression()), !dbg !1291
  %idxprom = sext i32 14 to i64, !dbg !1292
  %arrayidx31 = getelementptr inbounds [16 x i8], [16 x i8]* %size, i64 0, i64 %idxprom, !dbg !1292
  %4 = load i8, i8* %arrayidx31, align 1, !dbg !1292
  %conv32 = sext i8 %4 to i32, !dbg !1292
  %cmp33 = icmp eq i32 %conv32, 46, !dbg !1294
  br i1 %cmp33, label %if.then34, label %if.end37, !dbg !1295

if.then34:                                        ; preds = %if.then27
  %idxprom35 = sext i32 14 to i64, !dbg !1296
  %arrayidx36 = getelementptr inbounds [16 x i8], [16 x i8]* %size, i64 0, i64 %idxprom35, !dbg !1296
  store i8 32, i8* %arrayidx36, align 1, !dbg !1298
  %dec = add nsw i32 14, -1, !dbg !1299
  call void @llvm.dbg.value(metadata i32 %dec, metadata !1290, metadata !DIExpression()), !dbg !1291
  br label %if.end37, !dbg !1300

if.end37:                                         ; preds = %if.then34, %if.then27
  %j.0 = phi i32 [ %dec, %if.then34 ], [ 14, %if.then27 ], !dbg !1301
  call void @llvm.dbg.value(metadata i32 %j.0, metadata !1290, metadata !DIExpression()), !dbg !1291
  %add = add nsw i32 %j.0, 1, !dbg !1302
  %idxprom38 = sext i32 %add to i64, !dbg !1303
  %arrayidx39 = getelementptr inbounds [16 x i8], [16 x i8]* %size, i64 0, i64 %idxprom38, !dbg !1303
  store i8 0, i8* %arrayidx39, align 1, !dbg !1304
  %arraydecay40 = getelementptr inbounds [16 x i8], [16 x i8]* %size, i64 0, i64 0, !dbg !1305
  %call41 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([34 x i8], [34 x i8]* @.str.5, i64 0, i64 0), i8* %arraydecay40), !dbg !1306
  br label %if.end44, !dbg !1307

if.else42:                                        ; preds = %land.lhs.true23, %if.then19
  %call43 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.6, i64 0, i64 0), i32 %n1), !dbg !1308
  br label %if.end44

if.end44:                                         ; preds = %if.else42, %if.end37
  br label %if.end47, !dbg !1310

if.else45:                                        ; preds = %land.lhs.true17, %if.else15
  %call46 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([42 x i8], [42 x i8]* @.str.7, i64 0, i64 0), i32 %n1, i32 %n2, i32 %n3), !dbg !1311
  br label %if.end47

if.end47:                                         ; preds = %if.else45, %if.end44
  br label %if.end48

if.end48:                                         ; preds = %if.end47, %if.end14
  %call49 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.8, i64 0, i64 0), i32 %niter), !dbg !1313
  %call50 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([39 x i8], [39 x i8]* @.str.9, i64 0, i64 0), double %t), !dbg !1314
  %call51 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([39 x i8], [39 x i8]* @.str.10, i64 0, i64 0), double %mops), !dbg !1315
  %call52 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([25 x i8], [25 x i8]* @.str.11, i64 0, i64 0), i8* %optype), !dbg !1316
  %cmp53 = icmp slt i32 %passed_verification, 0, !dbg !1317
  br i1 %cmp53, label %if.then54, label %if.else56, !dbg !1319

if.then54:                                        ; preds = %if.end48
  %call55 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([45 x i8], [45 x i8]* @.str.12, i64 0, i64 0)), !dbg !1320
  br label %if.end62, !dbg !1322

if.else56:                                        ; preds = %if.end48
  %tobool = icmp ne i32 %passed_verification, 0, !dbg !1323
  br i1 %tobool, label %if.then57, label %if.else59, !dbg !1325

if.then57:                                        ; preds = %if.else56
  %call58 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([45 x i8], [45 x i8]* @.str.13, i64 0, i64 0)), !dbg !1326
  br label %if.end61, !dbg !1328

if.else59:                                        ; preds = %if.else56
  %call60 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([45 x i8], [45 x i8]* @.str.14, i64 0, i64 0)), !dbg !1329
  br label %if.end61

if.end61:                                         ; preds = %if.else59, %if.then57
  br label %if.end62

if.end62:                                         ; preds = %if.end61, %if.then54
  %call63 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.15, i64 0, i64 0), i8* %npbversion), !dbg !1331
  %call64 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.16, i64 0, i64 0), i8* %compiletime), !dbg !1332
  %call65 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.17, i64 0, i64 0), i8* %compilerversion), !dbg !1333
  %call66 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.18, i64 0, i64 0), i8* %libversion), !dbg !1334
  %call67 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([20 x i8], [20 x i8]* @.str.19, i64 0, i64 0)), !dbg !1335
  %call68 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.20, i64 0, i64 0), i8* %cc), !dbg !1336
  %call69 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.21, i64 0, i64 0), i8* %clink), !dbg !1337
  %call70 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.22, i64 0, i64 0), i8* %c_lib), !dbg !1338
  %call71 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.23, i64 0, i64 0), i8* %c_inc), !dbg !1339
  %call72 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.24, i64 0, i64 0), i8* %cflags), !dbg !1340
  %call73 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.25, i64 0, i64 0), i8* %clinkflags), !dbg !1341
  %call74 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.26, i64 0, i64 0), i8* %rand), !dbg !1342
  %call75 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.27, i64 0, i64 0)), !dbg !1343
  %call76 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.28, i64 0, i64 0), i8* %cpu_device), !dbg !1344
  %call77 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.29, i64 0, i64 0), i8* %gpu_device), !dbg !1345
  %call78 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.30, i64 0, i64 0)), !dbg !1346
  %call79 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.31, i64 0, i64 0), i8* %gpu_config), !dbg !1347
  %call80 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.32, i64 0, i64 0)), !dbg !1348
  %call81 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([72 x i8], [72 x i8]* @.str.33, i64 0, i64 0)), !dbg !1349
  %call82 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([27 x i8], [27 x i8]* @.str.34, i64 0, i64 0)), !dbg !1350
  %call83 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([56 x i8], [56 x i8]* @.str.35, i64 0, i64 0)), !dbg !1351
  %call84 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([52 x i8], [52 x i8]* @.str.36, i64 0, i64 0)), !dbg !1352
  %call85 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([46 x i8], [46 x i8]* @.str.37, i64 0, i64 0)), !dbg !1353
  %call86 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.32, i64 0, i64 0)), !dbg !1354
  %call87 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([43 x i8], [43 x i8]* @.str.38, i64 0, i64 0)), !dbg !1355
  %call88 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([72 x i8], [72 x i8]* @.str.33, i64 0, i64 0)), !dbg !1356
  %call89 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.32, i64 0, i64 0)), !dbg !1357
  ret void, !dbg !1358
}

declare dso_local i32 @printf(i8*, ...) #6

; Function Attrs: nounwind
declare dso_local double @pow(double, double) #4

; Function Attrs: nounwind
declare dso_local i32 @sprintf(i8*, i8*, ...) #4

; Function Attrs: noinline norecurse uwtable
define dso_local i32 @main(i32 %argc, i8** %argv) #7 !dbg !1359 {
entry:
  %t1 = alloca double, align 8
  %size = alloca [16 x i8], align 16
  %agg.tmp = alloca %struct.dim3, align 4
  %agg.tmp17 = alloca %struct.dim3, align 4
  %agg.tmp.coerce = alloca { i64, i32 }, align 4
  %agg.tmp17.coerce = alloca { i64, i32 }, align 4
  %gpu_config = alloca [256 x i8], align 16
  %gpu_config_string = alloca [2048 x i8], align 16
  call void @llvm.dbg.value(metadata i32 %argc, metadata !1362, metadata !DIExpression()), !dbg !1363
  call void @llvm.dbg.value(metadata i8** %argv, metadata !1364, metadata !DIExpression()), !dbg !1363
  call void @llvm.dbg.declare(metadata double* %t1, metadata !1365, metadata !DIExpression()), !dbg !1366
  call void @llvm.dbg.declare(metadata [16 x i8]* %size, metadata !1367, metadata !DIExpression()), !dbg !1368
  %arraydecay = getelementptr inbounds [16 x i8], [16 x i8]* %size, i64 0, i64 0, !dbg !1369
  %call = call double @pow(double 2.000000e+00, double 2.900000e+01) #10, !dbg !1370
  %call1 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay, i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.39, i64 0, i64 0), double %call) #10, !dbg !1371
  call void @llvm.dbg.value(metadata i32 14, metadata !1372, metadata !DIExpression()), !dbg !1363
  %idxprom = sext i32 14 to i64, !dbg !1373
  %arrayidx = getelementptr inbounds [16 x i8], [16 x i8]* %size, i64 0, i64 %idxprom, !dbg !1373
  %0 = load i8, i8* %arrayidx, align 1, !dbg !1373
  %conv = sext i8 %0 to i32, !dbg !1373
  %cmp = icmp eq i32 %conv, 46, !dbg !1375
  br i1 %cmp, label %if.then, label %if.end, !dbg !1376

if.then:                                          ; preds = %entry
  %dec = add nsw i32 14, -1, !dbg !1377
  call void @llvm.dbg.value(metadata i32 %dec, metadata !1372, metadata !DIExpression()), !dbg !1363
  br label %if.end, !dbg !1379

if.end:                                           ; preds = %if.then, %entry
  %j.0 = phi i32 [ %dec, %if.then ], [ 14, %entry ], !dbg !1363
  call void @llvm.dbg.value(metadata i32 %j.0, metadata !1372, metadata !DIExpression()), !dbg !1363
  %add = add nsw i32 %j.0, 1, !dbg !1380
  %idxprom2 = sext i32 %add to i64, !dbg !1381
  %arrayidx3 = getelementptr inbounds [16 x i8], [16 x i8]* %size, i64 0, i64 %idxprom2, !dbg !1381
  store i8 0, i8* %arrayidx3, align 1, !dbg !1382
  %call4 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([65 x i8], [65 x i8]* @.str.40, i64 0, i64 0)), !dbg !1383
  %arraydecay5 = getelementptr inbounds [16 x i8], [16 x i8]* %size, i64 0, i64 0, !dbg !1384
  %call6 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([43 x i8], [43 x i8]* @.str.41, i64 0, i64 0), i8* %arraydecay5), !dbg !1385
  call void @llvm.dbg.value(metadata i32 0, metadata !1386, metadata !DIExpression()), !dbg !1363
  store double 0x41D2309CE5400000, double* %t1, align 8, !dbg !1389
  call void @llvm.dbg.value(metadata i32 0, metadata !1390, metadata !DIExpression()), !dbg !1363
  br label %for.cond, !dbg !1391

for.cond:                                         ; preds = %for.inc, %if.end
  %i.0 = phi i32 [ 0, %if.end ], [ %inc, %for.inc ], !dbg !1393
  call void @llvm.dbg.value(metadata i32 %i.0, metadata !1390, metadata !DIExpression()), !dbg !1363
  %exitcond19 = icmp ne i32 %i.0, 17, !dbg !1394
  br i1 %exitcond19, label %for.body, label %for.end, !dbg !1396

for.body:                                         ; preds = %for.cond
  %1 = load double, double* %t1, align 8, !dbg !1397
  %call8 = call double @_Z6randlcPdd(double* %t1, double %1), !dbg !1399
  br label %for.inc, !dbg !1400

for.inc:                                          ; preds = %for.body
  %inc = add nuw nsw i32 %i.0, 1, !dbg !1401
  call void @llvm.dbg.value(metadata i32 %inc, metadata !1390, metadata !DIExpression()), !dbg !1363
  br label %for.cond, !dbg !1402, !llvm.loop !1403

for.end:                                          ; preds = %for.cond
  %2 = load double, double* %t1, align 8, !dbg !1405
  call void @llvm.dbg.value(metadata double %2, metadata !1406, metadata !DIExpression()), !dbg !1363
  call void @llvm.dbg.value(metadata double 0.000000e+00, metadata !1407, metadata !DIExpression()), !dbg !1363
  call void @llvm.dbg.value(metadata double 0.000000e+00, metadata !1408, metadata !DIExpression()), !dbg !1363
  call void @llvm.dbg.value(metadata double 0.000000e+00, metadata !1409, metadata !DIExpression()), !dbg !1363
  call void @llvm.dbg.value(metadata i32 0, metadata !1390, metadata !DIExpression()), !dbg !1363
  br label %for.cond9, !dbg !1410

for.cond9:                                        ; preds = %for.inc14, %for.end
  %indvars.iv16 = phi i64 [ %indvars.iv.next17, %for.inc14 ], [ 0, %for.end ], !dbg !1412
  call void @llvm.dbg.value(metadata i64 %indvars.iv16, metadata !1390, metadata !DIExpression()), !dbg !1363
  %exitcond18 = icmp ne i64 %indvars.iv16, 10, !dbg !1413
  br i1 %exitcond18, label %for.body11, label %for.end16, !dbg !1415

for.body11:                                       ; preds = %for.cond9
  %3 = load double*, double** @_ZL1q, align 8, !dbg !1416
  %arrayidx13 = getelementptr inbounds double, double* %3, i64 %indvars.iv16, !dbg !1416
  store double 0.000000e+00, double* %arrayidx13, align 8, !dbg !1418
  br label %for.inc14, !dbg !1419

for.inc14:                                        ; preds = %for.body11
  %indvars.iv.next17 = add nuw nsw i64 %indvars.iv16, 1, !dbg !1420
  call void @llvm.dbg.value(metadata i32 undef, metadata !1390, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !1363
  br label %for.cond9, !dbg !1421, !llvm.loop !1422

for.end16:                                        ; preds = %for.cond9
  call void @_ZL9setup_gpuv(), !dbg !1424
  %4 = load i32, i32* @blocks_per_grid, align 4, !dbg !1425
  %dim3gep.0 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp, i32 0, i32 0
  store i32 %4, i32* %dim3gep.0
  %dim3gep.1 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp, i32 0, i32 1
  store i32 1, i32* %dim3gep.1
  %dim3gep.2 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp, i32 0, i32 2
  store i32 1, i32* %dim3gep.2
  %5 = load i32, i32* @threads_per_block, align 4, !dbg !1426
  %dim3gep.01 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp17, i32 0, i32 0
  store i32 %5, i32* %dim3gep.01
  %dim3gep.12 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp17, i32 0, i32 1
  store i32 1, i32* %dim3gep.12
  %dim3gep.23 = getelementptr %struct.dim3, %struct.dim3* %agg.tmp17, i32 0, i32 2
  store i32 1, i32* %dim3gep.23
  %6 = bitcast { i64, i32 }* %agg.tmp.coerce to i8*, !dbg !1427
  %7 = bitcast %struct.dim3* %agg.tmp to i8*, !dbg !1427
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %6, i8* align 4 %7, i64 12, i1 false), !dbg !1427
  %8 = bitcast { i64, i32 }* %agg.tmp17.coerce to i8*, !dbg !1427
  %9 = bitcast %struct.dim3* %agg.tmp17 to i8*, !dbg !1427
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %8, i8* align 4 %9, i64 12, i1 false), !dbg !1427
  br label %header.0

header.0:                                         ; preds = %latch.0, %for.end16
  %indvar.0 = phi i32 [ 0, %for.end16 ], [ %indvar.next.0, %latch.0 ]
  %exitcond15 = icmp ne i32 %indvar.0, %4
  br i1 %exitcond15, label %header.1.preheader, label %kcall.end, !tulip.doall.loop.grid !1428

header.1.preheader:                               ; preds = %header.0
  br label %header.1

header.1:                                         ; preds = %header.1.preheader, %latch.1
  %indvar.1 = phi i32 [ %indvar.next.1, %latch.1 ], [ 0, %header.1.preheader ]
  %exitcond14 = icmp ne i32 %indvar.1, %5
  br i1 %exitcond14, label %kcall.configok, label %latch.0, !tulip.doall.loop.block !1428

latch.1:                                          ; preds = %kcall.configok
  %indvar.next.1 = add i32 %indvar.1, 1
  br label %header.1

latch.0:                                          ; preds = %header.1
  %indvar.next.0 = add i32 %indvar.0, 1
  br label %header.0

kcall.configok:                                   ; preds = %header.1
  %10 = load double*, double** @q_host, align 8, !dbg !1429
  %11 = load double*, double** @sx_host, align 8, !dbg !1430
  %12 = load double*, double** @sy_host, align 8, !dbg !1431
  call void @gpu_kernel(double* %10, double* %11, double* %12, double %2, i32 %4, i32 1, i32 1, i32 %5, i32 1, i32 1, i32 %indvar.0, i32 0, i32 0, i32 %indvar.1, i32 0, i32 0)
  br label %latch.1

kcall.end:                                        ; preds = %header.0
  %13 = load double*, double** @q_host, align 8, !dbg !1432
  %14 = bitcast double* %13 to i8*, !dbg !1432
  %15 = load double*, double** @q_host, align 8, !dbg !1433
  %16 = bitcast double* %15 to i8*, !dbg !1433
  %17 = load i64, i64* @size_q, align 8, !dbg !1434
  %call19 = call i32 @cudaMemcpy(i8* %14, i8* %16, i64 %17, i32 2), !dbg !1435, !tulip.target.end.of.map !1428
  call void @llvm.dbg.value(metadata i32 0, metadata !1436, metadata !DIExpression()), !dbg !1363
  br label %for.cond22, !dbg !1437

for.cond22:                                       ; preds = %for.inc43, %kcall.end
  %indvars.iv11 = phi i64 [ %indvars.iv.next12, %for.inc43 ], [ 0, %kcall.end ], !dbg !1363
  %sy.0 = phi double [ 0.000000e+00, %kcall.end ], [ %add42, %for.inc43 ], !dbg !1363
  %sx.0 = phi double [ 0.000000e+00, %kcall.end ], [ %add39, %for.inc43 ], !dbg !1363
  call void @llvm.dbg.value(metadata i64 %indvars.iv11, metadata !1436, metadata !DIExpression()), !dbg !1363
  call void @llvm.dbg.value(metadata double %sx.0, metadata !1408, metadata !DIExpression()), !dbg !1363
  call void @llvm.dbg.value(metadata double %sy.0, metadata !1409, metadata !DIExpression()), !dbg !1363
  %18 = load i32, i32* @blocks_per_grid, align 4, !dbg !1439
  %19 = sext i32 %18 to i64, !dbg !1441
  %cmp23 = icmp slt i64 %indvars.iv11, %19, !dbg !1441
  br i1 %cmp23, label %for.body24, label %for.end45, !dbg !1442

for.body24:                                       ; preds = %for.cond22
  call void @llvm.dbg.value(metadata i32 0, metadata !1390, metadata !DIExpression()), !dbg !1363
  br label %for.cond25, !dbg !1443

for.cond25:                                       ; preds = %for.inc34, %for.body24
  %indvars.iv7 = phi i64 [ %indvars.iv.next8, %for.inc34 ], [ 0, %for.body24 ], !dbg !1446
  call void @llvm.dbg.value(metadata i64 %indvars.iv7, metadata !1390, metadata !DIExpression()), !dbg !1363
  %exitcond10 = icmp ne i64 %indvars.iv7, 10, !dbg !1447
  br i1 %exitcond10, label %for.body27, label %for.end36, !dbg !1449

for.body27:                                       ; preds = %for.cond25
  %20 = load double*, double** @q_host, align 8, !dbg !1450
  %21 = mul nuw nsw i64 %indvars.iv11, 10, !dbg !1452
  %22 = add nuw nsw i64 %21, %indvars.iv7, !dbg !1453
  %arrayidx30 = getelementptr inbounds double, double* %20, i64 %22, !dbg !1450
  %23 = load double, double* %arrayidx30, align 8, !dbg !1450
  %24 = load double*, double** @_ZL1q, align 8, !dbg !1454
  %arrayidx32 = getelementptr inbounds double, double* %24, i64 %indvars.iv7, !dbg !1454
  %25 = load double, double* %arrayidx32, align 8, !dbg !1455
  %add33 = fadd contract double %25, %23, !dbg !1455
  store double %add33, double* %arrayidx32, align 8, !dbg !1455
  br label %for.inc34, !dbg !1456

for.inc34:                                        ; preds = %for.body27
  %indvars.iv.next8 = add nuw nsw i64 %indvars.iv7, 1, !dbg !1457
  call void @llvm.dbg.value(metadata i32 undef, metadata !1390, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !1363
  br label %for.cond25, !dbg !1458, !llvm.loop !1459

for.end36:                                        ; preds = %for.cond25
  %26 = load double*, double** @sx_host, align 8, !dbg !1461
  %arrayidx38 = getelementptr inbounds double, double* %26, i64 %indvars.iv11, !dbg !1461
  %27 = load double, double* %arrayidx38, align 8, !dbg !1461
  %add39 = fadd contract double %sx.0, %27, !dbg !1462
  call void @llvm.dbg.value(metadata double %add39, metadata !1408, metadata !DIExpression()), !dbg !1363
  %28 = load double*, double** @sy_host, align 8, !dbg !1463
  %arrayidx41 = getelementptr inbounds double, double* %28, i64 %indvars.iv11, !dbg !1463
  %29 = load double, double* %arrayidx41, align 8, !dbg !1463
  %add42 = fadd contract double %sy.0, %29, !dbg !1464
  call void @llvm.dbg.value(metadata double %add42, metadata !1409, metadata !DIExpression()), !dbg !1363
  br label %for.inc43, !dbg !1465

for.inc43:                                        ; preds = %for.end36
  %indvars.iv.next12 = add nuw nsw i64 %indvars.iv11, 1, !dbg !1466
  call void @llvm.dbg.value(metadata i32 undef, metadata !1436, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !1363
  br label %for.cond22, !dbg !1467, !llvm.loop !1468

for.end45:                                        ; preds = %for.cond22
  %sy.0.lcssa = phi double [ %sy.0, %for.cond22 ], !dbg !1363
  %sx.0.lcssa = phi double [ %sx.0, %for.cond22 ], !dbg !1363
  call void @llvm.dbg.value(metadata double %sy.0.lcssa, metadata !1409, metadata !DIExpression()), !dbg !1363
  call void @llvm.dbg.value(metadata double %sx.0.lcssa, metadata !1408, metadata !DIExpression()), !dbg !1363
  call void @llvm.dbg.value(metadata i32 0, metadata !1390, metadata !DIExpression()), !dbg !1363
  br label %for.cond46, !dbg !1470

for.cond46:                                       ; preds = %for.inc52, %for.end45
  %indvars.iv4 = phi i64 [ %indvars.iv.next5, %for.inc52 ], [ 0, %for.end45 ], !dbg !1363
  %gc.0 = phi double [ 0.000000e+00, %for.end45 ], [ %add51, %for.inc52 ], !dbg !1363
  call void @llvm.dbg.value(metadata i64 %indvars.iv4, metadata !1390, metadata !DIExpression()), !dbg !1363
  call void @llvm.dbg.value(metadata double %gc.0, metadata !1407, metadata !DIExpression()), !dbg !1363
  %exitcond6 = icmp ne i64 %indvars.iv4, 10, !dbg !1472
  br i1 %exitcond6, label %for.body48, label %for.end54, !dbg !1474

for.body48:                                       ; preds = %for.cond46
  %30 = load double*, double** @_ZL1q, align 8, !dbg !1475
  %arrayidx50 = getelementptr inbounds double, double* %30, i64 %indvars.iv4, !dbg !1475
  %31 = load double, double* %arrayidx50, align 8, !dbg !1475
  %add51 = fadd contract double %gc.0, %31, !dbg !1477
  call void @llvm.dbg.value(metadata double %add51, metadata !1407, metadata !DIExpression()), !dbg !1363
  br label %for.inc52, !dbg !1478

for.inc52:                                        ; preds = %for.body48
  %indvars.iv.next5 = add nuw nsw i64 %indvars.iv4, 1, !dbg !1479
  call void @llvm.dbg.value(metadata i32 undef, metadata !1390, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !1363
  br label %for.cond46, !dbg !1480, !llvm.loop !1481

for.end54:                                        ; preds = %for.cond46
  %gc.0.lcssa = phi double [ %gc.0, %for.cond46 ], !dbg !1363
  call void @llvm.dbg.value(metadata double %gc.0.lcssa, metadata !1407, metadata !DIExpression()), !dbg !1363
  call void @llvm.dbg.value(metadata i32 0, metadata !1483, metadata !DIExpression()), !dbg !1363
  call void @llvm.dbg.value(metadata i32 1, metadata !1386, metadata !DIExpression()), !dbg !1363
  call void @llvm.dbg.value(metadata double 0xC0B0C7E00ADACEF8, metadata !1484, metadata !DIExpression()), !dbg !1363
  call void @llvm.dbg.value(metadata double 0xC0CEDFA9B1BE31DC, metadata !1485, metadata !DIExpression()), !dbg !1363
  %tobool55 = icmp ne i32 1, 0, !dbg !1486
  br i1 %tobool55, label %if.then56, label %if.end62, !dbg !1488

if.then56:                                        ; preds = %for.end54
  %sub = fsub contract double %sx.0.lcssa, 0xC0B0C7E00ADACEF8, !dbg !1489
  %div = fdiv double %sub, 0xC0B0C7E00ADACEF8, !dbg !1491
  %32 = call double @llvm.fabs.f64(double %div), !dbg !1492
  call void @llvm.dbg.value(metadata double %32, metadata !1493, metadata !DIExpression()), !dbg !1363
  %sub57 = fsub contract double %sy.0.lcssa, 0xC0CEDFA9B1BE31DC, !dbg !1494
  %div58 = fdiv double %sub57, 0xC0CEDFA9B1BE31DC, !dbg !1495
  %33 = call double @llvm.fabs.f64(double %div58), !dbg !1496
  call void @llvm.dbg.value(metadata double %33, metadata !1497, metadata !DIExpression()), !dbg !1363
  %cmp59 = fcmp ole double %32, 1.000000e-08, !dbg !1498
  br i1 %cmp59, label %land.rhs, label %land.end, !dbg !1499

land.rhs:                                         ; preds = %if.then56
  %cmp60 = fcmp ole double %33, 1.000000e-08, !dbg !1500
  br label %land.end

land.end:                                         ; preds = %land.rhs, %if.then56
  %34 = phi i1 [ false, %if.then56 ], [ %cmp60, %land.rhs ], !dbg !1501
  %conv61 = zext i1 %34 to i32, !dbg !1502
  call void @llvm.dbg.value(metadata i32 %conv61, metadata !1386, metadata !DIExpression()), !dbg !1363
  br label %if.end62, !dbg !1503

if.end62:                                         ; preds = %land.end, %for.end54
  %verified.0 = phi i32 [ %conv61, %land.end ], [ 1, %for.end54 ], !dbg !1363
  call void @llvm.dbg.value(metadata i32 %verified.0, metadata !1386, metadata !DIExpression()), !dbg !1363
  %call63 = call double @pow(double 2.000000e+00, double 2.900000e+01) #10, !dbg !1504
  %div64 = fdiv double %call63, undef, !dbg !1505
  call void @llvm.dbg.value(metadata double %div65, metadata !1506, metadata !DIExpression()), !dbg !1363
  %call66 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([26 x i8], [26 x i8]* @.str.42, i64 0, i64 0)), !dbg !1507
  %call67 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([19 x i8], [19 x i8]* @.str.43, i64 0, i64 0), double undef), !dbg !1508
  %call68 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([12 x i8], [12 x i8]* @.str.44, i64 0, i64 0), i32 28), !dbg !1509
  %call69 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([30 x i8], [30 x i8]* @.str.45, i64 0, i64 0), double %gc.0.lcssa), !dbg !1510
  %call70 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([25 x i8], [25 x i8]* @.str.46, i64 0, i64 0), double %sx.0.lcssa, double %sy.0.lcssa), !dbg !1511
  %call71 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.47, i64 0, i64 0)), !dbg !1512
  call void @llvm.dbg.value(metadata i32 0, metadata !1390, metadata !DIExpression()), !dbg !1363
  br label %for.cond72, !dbg !1513

for.cond72:                                       ; preds = %for.inc78, %if.end62
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc78 ], [ 0, %if.end62 ], !dbg !1515
  call void @llvm.dbg.value(metadata i64 %indvars.iv, metadata !1390, metadata !DIExpression()), !dbg !1363
  %exitcond = icmp ne i64 %indvars.iv, 10, !dbg !1516
  br i1 %exitcond, label %for.body74, label %for.end80, !dbg !1518

for.body74:                                       ; preds = %for.cond72
  %35 = load double*, double** @_ZL1q, align 8, !dbg !1519
  %arrayidx76 = getelementptr inbounds double, double* %35, i64 %indvars.iv, !dbg !1519
  %36 = load double, double* %arrayidx76, align 8, !dbg !1519
  %37 = trunc i64 %indvars.iv to i32, !dbg !1521
  %call77 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.48, i64 0, i64 0), i32 %37, double %36), !dbg !1521
  br label %for.inc78, !dbg !1522

for.inc78:                                        ; preds = %for.body74
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !1523
  call void @llvm.dbg.value(metadata i32 undef, metadata !1390, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !1363
  br label %for.cond72, !dbg !1524, !llvm.loop !1525

for.end80:                                        ; preds = %for.cond72
  %div65 = fdiv double %div64, 1.000000e+06, !dbg !1527
  call void @llvm.dbg.declare(metadata [256 x i8]* %gpu_config, metadata !1528, metadata !DIExpression()), !dbg !1529
  call void @llvm.dbg.declare(metadata [2048 x i8]* %gpu_config_string, metadata !1530, metadata !DIExpression()), !dbg !1534
  %arraydecay81 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1535
  %call82 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay81, i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.49, i64 0, i64 0), i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.50, i64 0, i64 0), i8* getelementptr inbounds ([18 x i8], [18 x i8]* @.str.51, i64 0, i64 0)) #10, !dbg !1536
  %arraydecay83 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !1537
  %arraydecay84 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1538
  %call85 = call i8* @strcpy(i8* %arraydecay83, i8* %arraydecay84) #10, !dbg !1539
  %arraydecay86 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1540
  %38 = load i32, i32* @threads_per_block, align 4, !dbg !1541
  %call87 = call i32 (i8*, i8*, ...) @sprintf(i8* %arraydecay86, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.52, i64 0, i64 0), i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.53, i64 0, i64 0), i32 %38) #10, !dbg !1542
  %arraydecay88 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !1543
  %arraydecay89 = getelementptr inbounds [256 x i8], [256 x i8]* %gpu_config, i64 0, i64 0, !dbg !1544
  %call90 = call i8* @strcat(i8* %arraydecay88, i8* %arraydecay89) #10, !dbg !1545
  %arraydecay91 = getelementptr inbounds [2048 x i8], [2048 x i8]* %gpu_config_string, i64 0, i64 0, !dbg !1546
  call void @_Z15c_print_resultsPcciiiiddS_iS_S_S_S_S_S_S_S_S_S_S_S_S_S_(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.54, i64 0, i64 0), i8 signext 65, i32 29, i32 0, i32 0, i32 0, double undef, double %div65, i8* getelementptr inbounds ([25 x i8], [25 x i8]* @.str.55, i64 0, i64 0), i32 %verified.0, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.56, i64 0, i64 0), i8* getelementptr inbounds ([12 x i8], [12 x i8]* @.str.57, i64 0, i64 0), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.58, i64 0, i64 0), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.58, i64 0, i64 0), i8* getelementptr inbounds ([42 x i8], [42 x i8]* @.str.59, i64 0, i64 0), i8* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 0, i64 0), i8* %arraydecay91, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.60, i64 0, i64 0), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.61, i64 0, i64 0), i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.62, i64 0, i64 0), i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.63, i64 0, i64 0), i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.64, i64 0, i64 0), i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.64, i64 0, i64 0), i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.65, i64 0, i64 0)), !dbg !1547
  call void @_ZL11release_gpuv(), !dbg !1548
  ret i32 0, !dbg !1549
}

; Function Attrs: noinline uwtable
define internal void @_ZL9setup_gpuv() #3 !dbg !1550 {
entry:
  %0 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 6), align 8, !dbg !1551
  %cmp = icmp sle i32 32, %0, !dbg !1553
  br i1 %cmp, label %if.then, label %if.else, !dbg !1554

if.then:                                          ; preds = %entry
  store i32 32, i32* @threads_per_block, align 4, !dbg !1555
  br label %if.end, !dbg !1557

if.else:                                          ; preds = %entry
  %1 = load i32, i32* getelementptr inbounds (%struct.cudaDeviceProp, %struct.cudaDeviceProp* @gpu_device_properties, i32 0, i32 4), align 4, !dbg !1558
  store i32 %1, i32* @threads_per_block, align 4, !dbg !1560
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %2 = load i32, i32* @threads_per_block, align 4, !dbg !1561
  %conv = sitofp i32 %2 to double, !dbg !1561
  %div = fdiv double 4.096000e+03, %conv, !dbg !1562
  %3 = call double @llvm.ceil.f64(double %div), !dbg !1563
  %conv1 = fptosi double %3 to i32, !dbg !1564
  store i32 %conv1, i32* @blocks_per_grid, align 4, !dbg !1565
  %4 = load i32, i32* @blocks_per_grid, align 4, !dbg !1566
  %mul = mul nsw i32 %4, 10, !dbg !1567
  %conv2 = sext i32 %mul to i64, !dbg !1566
  %mul3 = mul i64 %conv2, 8, !dbg !1568
  store i64 %mul3, i64* @size_q, align 8, !dbg !1569
  %5 = load i32, i32* @blocks_per_grid, align 4, !dbg !1570
  %conv4 = sext i32 %5 to i64, !dbg !1570
  %mul5 = mul i64 %conv4, 8, !dbg !1571
  store i64 %mul5, i64* @size_sx, align 8, !dbg !1572
  %6 = load i32, i32* @blocks_per_grid, align 4, !dbg !1573
  %conv6 = sext i32 %6 to i64, !dbg !1573
  %mul7 = mul i64 %conv6, 8, !dbg !1574
  store i64 %mul7, i64* @size_sy, align 8, !dbg !1575
  %7 = load i64, i64* @size_q, align 8, !dbg !1576, !tulip.target.datasize !1577
  %call = call noalias i8* @malloc(i64 %7) #10, !dbg !1578, !tulip.target.mapdata.from !1579
  %8 = bitcast i8* %call to double*, !dbg !1580
  store double* %8, double** @q_host, align 8, !dbg !1581
  %9 = load i64, i64* @size_sx, align 8, !dbg !1582, !tulip.target.datasize !1583
  %call8 = call noalias i8* @malloc(i64 %9) #10, !dbg !1584, !tulip.target.mapdata.from !1585
  %10 = bitcast i8* %call8 to double*, !dbg !1586
  store double* %10, double** @sx_host, align 8, !dbg !1587
  %11 = load i64, i64* @size_sy, align 8, !dbg !1588, !tulip.target.datasize !1589
  %call9 = call noalias i8* @malloc(i64 %11) #10, !dbg !1590, !tulip.target.mapdata.from !1591
  %12 = bitcast i8* %call9 to double*, !dbg !1592
  store double* %12, double** @sy_host, align 8, !dbg !1593
  ret void, !dbg !1594
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1 immarg) #8

declare dso_local i32 @cudaMemcpy(i8*, i8*, i64, i32) #6

; Function Attrs: nounwind readnone speculatable
declare double @llvm.fabs.f64(double) #0

; Function Attrs: nounwind
declare dso_local i8* @strcpy(i8*, i8*) #4

; Function Attrs: nounwind
declare dso_local i8* @strcat(i8*, i8*) #4

; Function Attrs: noinline uwtable
define internal void @_ZL11release_gpuv() #3 !dbg !1595 {
entry:
  ret void, !dbg !1596
}

declare dso_local i32 @cudaSetupArgument(i8*, i64, i64)

declare dso_local i32 @cudaLaunch(i8*)

; Function Attrs: nounwind readnone speculatable
declare double @llvm.ceil.f64(double) #0

declare double @log(double)

declare double @fabs(double)

; Function Attrs: convergent noinline nounwind
define dso_local void @gpu_kernel(double* %q_global, double* %sx_global, double* %sy_global, double %an, i32 %gridDim.x, i32 %gridDim.y, i32 %gridDim.z, i32 %blockDim.x, i32 %blockDim.y, i32 %blockDim.z, i32 %blockIdx.x, i32 %blockIdx.y, i32 %blockIdx.z, i32 %threadIdx.x, i32 %threadIdx.y, i32 %threadIdx.z) #9 {
entry:
  %x_local = alloca [256 x double], align 8
  %q_local = alloca [10 x double], align 8
  %t1 = alloca double, align 8
  %t2 = alloca double, align 8
  %seed = alloca double, align 8
  call void @llvm.dbg.value(metadata double* %q_global, metadata !1597, metadata !DIExpression()), !dbg !1601
  call void @llvm.dbg.value(metadata double* %sx_global, metadata !1602, metadata !DIExpression()), !dbg !1601
  call void @llvm.dbg.value(metadata double* %sy_global, metadata !1603, metadata !DIExpression()), !dbg !1601
  call void @llvm.dbg.value(metadata double %an, metadata !1604, metadata !DIExpression()), !dbg !1601
  call void @llvm.dbg.declare(metadata [256 x double]* %x_local, metadata !1605, metadata !DIExpression()), !dbg !1607
  call void @llvm.dbg.declare(metadata [10 x double]* %q_local, metadata !1608, metadata !DIExpression()), !dbg !1612
  call void @llvm.dbg.declare(metadata double* %t1, metadata !1613, metadata !DIExpression()), !dbg !1614
  call void @llvm.dbg.declare(metadata double* %t2, metadata !1615, metadata !DIExpression()), !dbg !1616
  call void @llvm.dbg.declare(metadata double* %seed, metadata !1617, metadata !DIExpression()), !dbg !1618
  %arrayidx = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 0, !dbg !1619
  store double 0.000000e+00, double* %arrayidx, align 8, !dbg !1620
  %arrayidx1 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 1, !dbg !1621
  store double 0.000000e+00, double* %arrayidx1, align 8, !dbg !1622
  %arrayidx2 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 2, !dbg !1623
  store double 0.000000e+00, double* %arrayidx2, align 8, !dbg !1624
  %arrayidx3 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 3, !dbg !1625
  store double 0.000000e+00, double* %arrayidx3, align 8, !dbg !1626
  %arrayidx4 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 4, !dbg !1627
  store double 0.000000e+00, double* %arrayidx4, align 8, !dbg !1628
  %arrayidx5 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 5, !dbg !1629
  store double 0.000000e+00, double* %arrayidx5, align 8, !dbg !1630
  %arrayidx6 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 6, !dbg !1631
  store double 0.000000e+00, double* %arrayidx6, align 8, !dbg !1632
  %arrayidx7 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 7, !dbg !1633
  store double 0.000000e+00, double* %arrayidx7, align 8, !dbg !1634
  %arrayidx8 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 8, !dbg !1635
  store double 0.000000e+00, double* %arrayidx8, align 8, !dbg !1636
  %arrayidx9 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 9, !dbg !1637
  store double 0.000000e+00, double* %arrayidx9, align 8, !dbg !1638
  call void @llvm.dbg.value(metadata double 0.000000e+00, metadata !1639, metadata !DIExpression()), !dbg !1601
  call void @llvm.dbg.value(metadata double 0.000000e+00, metadata !1640, metadata !DIExpression()), !dbg !1601
  %mul = mul i32 %blockIdx.x, %blockDim.x, !dbg !1641
  %add = add i32 %mul, %threadIdx.x, !dbg !1642
  call void @llvm.dbg.value(metadata i32 %add, metadata !1643, metadata !DIExpression()), !dbg !1601
  %cmp = icmp sge i32 %add, 4096, !dbg !1644
  br i1 %cmp, label %if.then, label %if.end, !dbg !1646

if.then:                                          ; preds = %entry
  br label %for.end65, !dbg !1647

if.end:                                           ; preds = %entry
  store double 0x41B033C4D7000000, double* %t1, align 8, !dbg !1649
  store double %an, double* %t2, align 8, !dbg !1650
  call void @llvm.dbg.value(metadata i32 1, metadata !1651, metadata !DIExpression()), !dbg !1601
  br label %for.cond, !dbg !1652

for.cond:                                         ; preds = %for.inc, %if.end
  %i.0 = phi i32 [ 1, %if.end ], [ %inc, %for.inc ], !dbg !1654
  %kk.0 = phi i32 [ %add, %if.end ], [ %div, %for.inc ], !dbg !1601
  call void @llvm.dbg.value(metadata i32 %kk.0, metadata !1643, metadata !DIExpression()), !dbg !1601
  call void @llvm.dbg.value(metadata i32 %i.0, metadata !1651, metadata !DIExpression()), !dbg !1601
  %exitcond4 = icmp ne i32 %i.0, 101, !dbg !1655
  br i1 %exitcond4, label %for.body, label %for.end.loopexit, !dbg !1657

for.body:                                         ; preds = %for.cond
  %div = sdiv i32 %kk.0, 2, !dbg !1658
  call void @llvm.dbg.value(metadata i32 %div, metadata !1660, metadata !DIExpression()), !dbg !1601
  %mul13 = mul nsw i32 2, %div, !dbg !1661
  %cmp14 = icmp ne i32 %mul13, %kk.0, !dbg !1663
  br i1 %cmp14, label %if.then15, label %if.end17, !dbg !1664

if.then15:                                        ; preds = %for.body
  %0 = load double, double* %t2, align 8, !dbg !1665
  %call16 = call double @_Z13randlc_devicePdd(double* %t1, double %0) #11, !dbg !1667
  call void @llvm.dbg.value(metadata double %call16, metadata !1668, metadata !DIExpression()), !dbg !1601
  br label %if.end17, !dbg !1669

if.end17:                                         ; preds = %if.then15, %for.body
  %cmp18 = icmp eq i32 %div, 0, !dbg !1670
  br i1 %cmp18, label %if.then19, label %if.end20, !dbg !1672

if.then19:                                        ; preds = %if.end17
  br label %for.end, !dbg !1673

if.end20:                                         ; preds = %if.end17
  %1 = load double, double* %t2, align 8, !dbg !1675
  %call21 = call double @_Z13randlc_devicePdd(double* %t2, double %1) #11, !dbg !1676
  call void @llvm.dbg.value(metadata double %call21, metadata !1668, metadata !DIExpression()), !dbg !1601
  call void @llvm.dbg.value(metadata i32 %div, metadata !1643, metadata !DIExpression()), !dbg !1601
  br label %for.inc, !dbg !1677

for.inc:                                          ; preds = %if.end20
  %inc = add nuw nsw i32 %i.0, 1, !dbg !1678
  call void @llvm.dbg.value(metadata i32 %inc, metadata !1651, metadata !DIExpression()), !dbg !1601
  br label %for.cond, !dbg !1679, !llvm.loop !1680

for.end.loopexit:                                 ; preds = %for.cond
  br label %for.end, !dbg !1682

for.end:                                          ; preds = %for.end.loopexit, %if.then19
  %2 = load double, double* %t1, align 8, !dbg !1682
  store double %2, double* %seed, align 8, !dbg !1683
  call void @llvm.dbg.value(metadata i32 0, metadata !1684, metadata !DIExpression()), !dbg !1601
  br label %for.cond22, !dbg !1685

for.cond22:                                       ; preds = %for.inc63, %for.end
  %sx_local.0 = phi double [ 0.000000e+00, %for.end ], [ %sx_local.1.lcssa, %for.inc63 ], !dbg !1687
  %sy_local.0 = phi double [ 0.000000e+00, %for.end ], [ %sy_local.1.lcssa, %for.inc63 ], !dbg !1688
  %ii.0 = phi i32 [ 0, %for.end ], [ %add64, %for.inc63 ], !dbg !1689
  call void @llvm.dbg.value(metadata i32 %ii.0, metadata !1684, metadata !DIExpression()), !dbg !1601
  call void @llvm.dbg.value(metadata double %sy_local.0, metadata !1640, metadata !DIExpression()), !dbg !1601
  call void @llvm.dbg.value(metadata double %sx_local.0, metadata !1639, metadata !DIExpression()), !dbg !1601
  %cmp23 = icmp ult i32 %ii.0, 65536, !dbg !1690
  br i1 %cmp23, label %for.body24, label %for.end65.loopexit, !dbg !1692

for.body24:                                       ; preds = %for.cond22
  %arraydecay = getelementptr inbounds [256 x double], [256 x double]* %x_local, i64 0, i64 0, !dbg !1693
  call void @_Z13vranlc_deviceiPddS_(i32 256, double* %seed, double 0x41D2309CE5400000, double* %arraydecay) #11, !dbg !1695
  call void @llvm.dbg.value(metadata i32 0, metadata !1651, metadata !DIExpression()), !dbg !1601
  br label %for.cond25, !dbg !1696

for.cond25:                                       ; preds = %for.inc60, %for.body24
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc60 ], [ 0, %for.body24 ], !dbg !1601
  %sx_local.1 = phi double [ %sx_local.0, %for.body24 ], [ %sx_local.2, %for.inc60 ], !dbg !1601
  %sy_local.1 = phi double [ %sy_local.0, %for.body24 ], [ %sy_local.2, %for.inc60 ], !dbg !1601
  call void @llvm.dbg.value(metadata i64 %indvars.iv, metadata !1651, metadata !DIExpression()), !dbg !1601
  call void @llvm.dbg.value(metadata double %sy_local.1, metadata !1640, metadata !DIExpression()), !dbg !1601
  call void @llvm.dbg.value(metadata double %sx_local.1, metadata !1639, metadata !DIExpression()), !dbg !1601
  %exitcond = icmp ne i64 %indvars.iv, 128, !dbg !1698
  br i1 %exitcond, label %for.body27, label %for.end62, !dbg !1700

for.body27:                                       ; preds = %for.cond25
  %3 = mul nuw nsw i64 2, %indvars.iv, !dbg !1701
  %arrayidx29 = getelementptr inbounds [256 x double], [256 x double]* %x_local, i64 0, i64 %3, !dbg !1703
  %4 = load double, double* %arrayidx29, align 8, !dbg !1703
  %mul30 = fmul contract double 2.000000e+00, %4, !dbg !1704
  %sub = fsub contract double %mul30, 1.000000e+00, !dbg !1705
  call void @llvm.dbg.value(metadata double %sub, metadata !1706, metadata !DIExpression()), !dbg !1601
  %5 = mul nuw nsw i64 2, %indvars.iv, !dbg !1707
  %6 = add nuw nsw i64 %5, 1, !dbg !1708
  %arrayidx34 = getelementptr inbounds [256 x double], [256 x double]* %x_local, i64 0, i64 %6, !dbg !1709
  %7 = load double, double* %arrayidx34, align 8, !dbg !1709
  %mul35 = fmul contract double 2.000000e+00, %7, !dbg !1710
  %sub36 = fsub contract double %mul35, 1.000000e+00, !dbg !1711
  call void @llvm.dbg.value(metadata double %sub36, metadata !1712, metadata !DIExpression()), !dbg !1601
  %mul37 = fmul contract double %sub, %sub, !dbg !1713
  %mul38 = fmul contract double %sub36, %sub36, !dbg !1714
  %add39 = fadd contract double %mul37, %mul38, !dbg !1715
  store double %add39, double* %t1, align 8, !dbg !1716
  %8 = load double, double* %t1, align 8, !dbg !1717
  %cmp40 = fcmp ole double %8, 1.000000e+00, !dbg !1719
  br i1 %cmp40, label %if.then41, label %if.end59, !dbg !1720

if.then41:                                        ; preds = %for.body27
  %9 = load double, double* %t1, align 8, !dbg !1721
  call void @llvm.dbg.value(metadata double %9, metadata !1723, metadata !DIExpression()), !dbg !1726
  %log_result = call double @log(double %9)
  br label %_ZL3logd.exit

_ZL3logd.exit:                                    ; preds = %if.then41
  call void @llvm.dbg.value(metadata double %log_result, metadata !1668, metadata !DIExpression()), !dbg !1601
  %10 = load double, double* %t2, align 8, !dbg !1728
  %mul48 = fmul contract double %sub36, %10, !dbg !1729
  call void @llvm.dbg.value(metadata double %mul48, metadata !1730, metadata !DIExpression()), !dbg !1601
  call void @llvm.dbg.value(metadata double %log_result, metadata !1731, metadata !DIExpression()), !dbg !1733
  %11 = call double @fabs(double %log_result)
  call void @llvm.dbg.value(metadata double %mul48, metadata !1731, metadata !DIExpression()), !dbg !1735
  %12 = call double @fabs(double %mul48)
  %cmp51 = fcmp ogt double %11, %12, !dbg !1737
  br i1 %cmp51, label %cond.true, label %cond.false, !dbg !1737

cond.true:                                        ; preds = %_ZL3logd.exit
  call void @llvm.dbg.value(metadata double %log_result, metadata !1731, metadata !DIExpression()), !dbg !1738
  %13 = call double @fabs(double %log_result)
  br label %cond.end, !dbg !1737

cond.false:                                       ; preds = %_ZL3logd.exit
  call void @llvm.dbg.value(metadata double %mul48, metadata !1731, metadata !DIExpression()), !dbg !1740
  %14 = call double @fabs(double %mul48)
  br label %cond.end, !dbg !1737

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi double [ %13, %cond.true ], [ %14, %cond.false ], !dbg !1737
  %conv = fptosi double %cond to i32, !dbg !1737
  call void @llvm.dbg.value(metadata i32 %conv, metadata !1742, metadata !DIExpression()), !dbg !1601
  %idxprom54 = sext i32 %conv to i64, !dbg !1743
  %arrayidx55 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 %idxprom54, !dbg !1743
  %15 = load double, double* %arrayidx55, align 8, !dbg !1744
  %add56 = fadd contract double %15, 1.000000e+00, !dbg !1744
  store double %add56, double* %arrayidx55, align 8, !dbg !1744
  %add57 = fadd contract double %sx_local.1, %log_result, !dbg !1745
  call void @llvm.dbg.value(metadata double %add57, metadata !1639, metadata !DIExpression()), !dbg !1601
  %add58 = fadd contract double %sy_local.1, %mul48, !dbg !1746
  call void @llvm.dbg.value(metadata double %add58, metadata !1640, metadata !DIExpression()), !dbg !1601
  br label %if.end59, !dbg !1747

if.end59:                                         ; preds = %cond.end, %for.body27
  %sx_local.2 = phi double [ %add57, %cond.end ], [ %sx_local.1, %for.body27 ], !dbg !1601
  %sy_local.2 = phi double [ %add58, %cond.end ], [ %sy_local.1, %for.body27 ], !dbg !1601
  call void @llvm.dbg.value(metadata double %sy_local.2, metadata !1640, metadata !DIExpression()), !dbg !1601
  call void @llvm.dbg.value(metadata double %sx_local.2, metadata !1639, metadata !DIExpression()), !dbg !1601
  br label %for.inc60, !dbg !1748

for.inc60:                                        ; preds = %if.end59
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !1749
  call void @llvm.dbg.value(metadata i32 undef, metadata !1651, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !1601
  br label %for.cond25, !dbg !1750, !llvm.loop !1751

for.end62:                                        ; preds = %for.cond25
  %sx_local.1.lcssa = phi double [ %sx_local.1, %for.cond25 ], !dbg !1601
  %sy_local.1.lcssa = phi double [ %sy_local.1, %for.cond25 ], !dbg !1601
  call void @llvm.dbg.value(metadata double %sx_local.1.lcssa, metadata !1639, metadata !DIExpression()), !dbg !1601
  call void @llvm.dbg.value(metadata double %sy_local.1.lcssa, metadata !1640, metadata !DIExpression()), !dbg !1601
  br label %for.inc63, !dbg !1753

for.inc63:                                        ; preds = %for.end62
  %add64 = add nuw nsw i32 %ii.0, 128, !dbg !1754
  call void @llvm.dbg.value(metadata i32 %add64, metadata !1684, metadata !DIExpression()), !dbg !1601
  br label %for.cond22, !dbg !1755, !llvm.loop !1756

for.end65.loopexit:                               ; preds = %for.cond22
  br label %for.end65, !dbg !1758

for.end65:                                        ; preds = %for.end65.loopexit, %if.then
  ret void, !dbg !1758
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #0

attributes #0 = { nounwind readnone speculatable }
attributes #1 = { nounwind readnone }
attributes #2 = { convergent noinline nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { noinline uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #7 = { noinline norecurse uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #8 = { argmemonly nounwind }
attributes #9 = { convergent noinline nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
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
!1034 = distinct !{null, !"kernel", i32 1}
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
!1046 = distinct !DISubprogram(name: "randlc_device", linkageName: "_Z13randlc_devicePdd", scope: !3, file: !3, line: 552, type: !1047, scopeLine: 553, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !961, retainedNodes: !962)
!1047 = !DISubroutineType(types: !1048)
!1048 = !{!98, !97, !98}
!1049 = !DILocalVariable(name: "x", arg: 1, scope: !1046, file: !3, line: 552, type: !97)
!1050 = !DILocation(line: 0, scope: !1046)
!1051 = !DILocalVariable(name: "a", arg: 2, scope: !1046, file: !3, line: 553, type: !98)
!1052 = !DILocation(line: 555, column: 11, scope: !1046)
!1053 = !DILocalVariable(name: "t1", scope: !1046, file: !3, line: 554, type: !98)
!1054 = !DILocation(line: 556, column: 12, scope: !1046)
!1055 = !DILocation(line: 556, column: 7, scope: !1046)
!1056 = !DILocalVariable(name: "a1", scope: !1046, file: !3, line: 554, type: !98)
!1057 = !DILocation(line: 557, column: 15, scope: !1046)
!1058 = !DILocation(line: 557, column: 9, scope: !1046)
!1059 = !DILocalVariable(name: "a2", scope: !1046, file: !3, line: 554, type: !98)
!1060 = !DILocation(line: 558, column: 14, scope: !1046)
!1061 = !DILocation(line: 558, column: 11, scope: !1046)
!1062 = !DILocation(line: 559, column: 12, scope: !1046)
!1063 = !DILocation(line: 559, column: 7, scope: !1046)
!1064 = !DILocalVariable(name: "x1", scope: !1046, file: !3, line: 554, type: !98)
!1065 = !DILocation(line: 560, column: 8, scope: !1046)
!1066 = !DILocation(line: 560, column: 18, scope: !1046)
!1067 = !DILocation(line: 560, column: 12, scope: !1046)
!1068 = !DILocalVariable(name: "x2", scope: !1046, file: !3, line: 554, type: !98)
!1069 = !DILocation(line: 561, column: 10, scope: !1046)
!1070 = !DILocation(line: 561, column: 20, scope: !1046)
!1071 = !DILocation(line: 561, column: 15, scope: !1046)
!1072 = !DILocation(line: 562, column: 17, scope: !1046)
!1073 = !DILocation(line: 562, column: 12, scope: !1046)
!1074 = !DILocation(line: 562, column: 7, scope: !1046)
!1075 = !DILocalVariable(name: "t2", scope: !1046, file: !3, line: 554, type: !98)
!1076 = !DILocation(line: 563, column: 15, scope: !1046)
!1077 = !DILocation(line: 563, column: 9, scope: !1046)
!1078 = !DILocalVariable(name: "z", scope: !1046, file: !3, line: 554, type: !98)
!1079 = !DILocation(line: 564, column: 11, scope: !1046)
!1080 = !DILocation(line: 564, column: 20, scope: !1046)
!1081 = !DILocation(line: 564, column: 15, scope: !1046)
!1082 = !DILocalVariable(name: "t3", scope: !1046, file: !3, line: 554, type: !98)
!1083 = !DILocation(line: 565, column: 17, scope: !1046)
!1084 = !DILocation(line: 565, column: 12, scope: !1046)
!1085 = !DILocation(line: 565, column: 7, scope: !1046)
!1086 = !DILocalVariable(name: "t4", scope: !1046, file: !3, line: 554, type: !98)
!1087 = !DILocation(line: 566, column: 18, scope: !1046)
!1088 = !DILocation(line: 566, column: 12, scope: !1046)
!1089 = !DILocation(line: 566, column: 7, scope: !1046)
!1090 = !DILocation(line: 567, column: 17, scope: !1046)
!1091 = !DILocation(line: 567, column: 14, scope: !1046)
!1092 = !DILocation(line: 567, column: 2, scope: !1046)
!1093 = distinct !DISubprogram(name: "vranlc_device", linkageName: "_Z13vranlc_deviceiPddS_", scope: !3, file: !3, line: 646, type: !1094, scopeLine: 649, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !961, retainedNodes: !962)
!1094 = !DISubroutineType(types: !1095)
!1095 = !{null, !99, !97, !98, !97}
!1096 = !DILocalVariable(name: "n", arg: 1, scope: !1093, file: !3, line: 646, type: !99)
!1097 = !DILocation(line: 0, scope: !1093)
!1098 = !DILocalVariable(name: "x_seed", arg: 2, scope: !1093, file: !3, line: 647, type: !97)
!1099 = !DILocalVariable(name: "a", arg: 3, scope: !1093, file: !3, line: 648, type: !98)
!1100 = !DILocalVariable(name: "y", arg: 4, scope: !1093, file: !3, line: 649, type: !97)
!1101 = !DILocation(line: 652, column: 11, scope: !1093)
!1102 = !DILocalVariable(name: "t1", scope: !1093, file: !3, line: 651, type: !98)
!1103 = !DILocation(line: 653, column: 12, scope: !1093)
!1104 = !DILocation(line: 653, column: 7, scope: !1093)
!1105 = !DILocalVariable(name: "a1", scope: !1093, file: !3, line: 651, type: !98)
!1106 = !DILocation(line: 654, column: 15, scope: !1093)
!1107 = !DILocation(line: 654, column: 9, scope: !1093)
!1108 = !DILocalVariable(name: "a2", scope: !1093, file: !3, line: 651, type: !98)
!1109 = !DILocation(line: 655, column: 6, scope: !1093)
!1110 = !DILocalVariable(name: "x", scope: !1093, file: !3, line: 651, type: !98)
!1111 = !DILocalVariable(name: "i", scope: !1093, file: !3, line: 650, type: !99)
!1112 = !DILocation(line: 656, column: 6, scope: !1113)
!1113 = distinct !DILexicalBlock(scope: !1093, file: !3, line: 656, column: 2)
!1114 = !DILocation(line: 0, scope: !1113)
!1115 = !DILocation(line: 656, column: 12, scope: !1116)
!1116 = distinct !DILexicalBlock(scope: !1113, file: !3, line: 656, column: 2)
!1117 = !DILocation(line: 656, column: 2, scope: !1113)
!1118 = !DILocation(line: 657, column: 12, scope: !1119)
!1119 = distinct !DILexicalBlock(scope: !1116, file: !3, line: 656, column: 20)
!1120 = !DILocation(line: 658, column: 13, scope: !1119)
!1121 = !DILocation(line: 658, column: 8, scope: !1119)
!1122 = !DILocalVariable(name: "x1", scope: !1093, file: !3, line: 651, type: !98)
!1123 = !DILocation(line: 659, column: 16, scope: !1119)
!1124 = !DILocation(line: 659, column: 10, scope: !1119)
!1125 = !DILocalVariable(name: "x2", scope: !1093, file: !3, line: 651, type: !98)
!1126 = !DILocation(line: 660, column: 11, scope: !1119)
!1127 = !DILocation(line: 660, column: 21, scope: !1119)
!1128 = !DILocation(line: 660, column: 16, scope: !1119)
!1129 = !DILocation(line: 661, column: 18, scope: !1119)
!1130 = !DILocation(line: 661, column: 13, scope: !1119)
!1131 = !DILocation(line: 661, column: 8, scope: !1119)
!1132 = !DILocalVariable(name: "t2", scope: !1093, file: !3, line: 651, type: !98)
!1133 = !DILocation(line: 662, column: 16, scope: !1119)
!1134 = !DILocation(line: 662, column: 10, scope: !1119)
!1135 = !DILocalVariable(name: "z", scope: !1093, file: !3, line: 651, type: !98)
!1136 = !DILocation(line: 663, column: 12, scope: !1119)
!1137 = !DILocation(line: 663, column: 21, scope: !1119)
!1138 = !DILocation(line: 663, column: 16, scope: !1119)
!1139 = !DILocalVariable(name: "t3", scope: !1093, file: !3, line: 651, type: !98)
!1140 = !DILocation(line: 664, column: 18, scope: !1119)
!1141 = !DILocation(line: 664, column: 13, scope: !1119)
!1142 = !DILocation(line: 664, column: 8, scope: !1119)
!1143 = !DILocalVariable(name: "t4", scope: !1093, file: !3, line: 651, type: !98)
!1144 = !DILocation(line: 665, column: 16, scope: !1119)
!1145 = !DILocation(line: 665, column: 10, scope: !1119)
!1146 = !DILocation(line: 666, column: 14, scope: !1119)
!1147 = !DILocation(line: 666, column: 3, scope: !1119)
!1148 = !DILocation(line: 666, column: 8, scope: !1119)
!1149 = !DILocation(line: 667, column: 2, scope: !1119)
!1150 = !DILocation(line: 656, column: 17, scope: !1116)
!1151 = !DILocation(line: 656, column: 2, scope: !1116)
!1152 = distinct !{!1152, !1117, !1153}
!1153 = !DILocation(line: 667, column: 2, scope: !1113)
!1154 = !DILocation(line: 668, column: 10, scope: !1093)
!1155 = !DILocation(line: 669, column: 1, scope: !1093)
!1156 = distinct !DISubprogram(linkageName: "_GLOBAL__sub_I_ep.cu", scope: !3, file: !3, type: !1157, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !962)
!1157 = !DISubroutineType(types: !962)
!1158 = !DILocation(line: 0, scope: !1156)
!1159 = distinct !DISubprogram(name: "__cxx_global_var_init", scope: !3, file: !3, line: 81, type: !472, scopeLine: 81, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !962)
!1160 = !DILocation(line: 81, column: 29, scope: !1159)
!1161 = !DILocation(line: 81, column: 20, scope: !1159)
!1162 = !DILocation(line: 81, column: 52, scope: !1159)
!1163 = distinct !DISubprogram(name: "randlc", linkageName: "_Z6randlcPdd", scope: !3, file: !3, line: 156, type: !1047, scopeLine: 156, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !962)
!1164 = !DILocalVariable(name: "x", arg: 1, scope: !1163, file: !3, line: 156, type: !97)
!1165 = !DILocation(line: 0, scope: !1163)
!1166 = !DILocalVariable(name: "a", arg: 2, scope: !1163, file: !3, line: 156, type: !98)
!1167 = !DILocation(line: 164, column: 11, scope: !1163)
!1168 = !DILocalVariable(name: "t1", scope: !1163, file: !3, line: 157, type: !98)
!1169 = !DILocation(line: 165, column: 12, scope: !1163)
!1170 = !DILocation(line: 165, column: 7, scope: !1163)
!1171 = !DILocalVariable(name: "a1", scope: !1163, file: !3, line: 157, type: !98)
!1172 = !DILocation(line: 166, column: 15, scope: !1163)
!1173 = !DILocation(line: 166, column: 9, scope: !1163)
!1174 = !DILocalVariable(name: "a2", scope: !1163, file: !3, line: 157, type: !98)
!1175 = !DILocation(line: 175, column: 14, scope: !1163)
!1176 = !DILocation(line: 175, column: 11, scope: !1163)
!1177 = !DILocation(line: 176, column: 12, scope: !1163)
!1178 = !DILocation(line: 176, column: 7, scope: !1163)
!1179 = !DILocalVariable(name: "x1", scope: !1163, file: !3, line: 157, type: !98)
!1180 = !DILocation(line: 177, column: 8, scope: !1163)
!1181 = !DILocation(line: 177, column: 18, scope: !1163)
!1182 = !DILocation(line: 177, column: 12, scope: !1163)
!1183 = !DILocalVariable(name: "x2", scope: !1163, file: !3, line: 157, type: !98)
!1184 = !DILocation(line: 178, column: 10, scope: !1163)
!1185 = !DILocation(line: 178, column: 20, scope: !1163)
!1186 = !DILocation(line: 178, column: 15, scope: !1163)
!1187 = !DILocation(line: 179, column: 17, scope: !1163)
!1188 = !DILocation(line: 179, column: 12, scope: !1163)
!1189 = !DILocation(line: 179, column: 7, scope: !1163)
!1190 = !DILocalVariable(name: "t2", scope: !1163, file: !3, line: 157, type: !98)
!1191 = !DILocation(line: 180, column: 15, scope: !1163)
!1192 = !DILocation(line: 180, column: 9, scope: !1163)
!1193 = !DILocalVariable(name: "z", scope: !1163, file: !3, line: 157, type: !98)
!1194 = !DILocation(line: 181, column: 11, scope: !1163)
!1195 = !DILocation(line: 181, column: 20, scope: !1163)
!1196 = !DILocation(line: 181, column: 15, scope: !1163)
!1197 = !DILocalVariable(name: "t3", scope: !1163, file: !3, line: 157, type: !98)
!1198 = !DILocation(line: 182, column: 17, scope: !1163)
!1199 = !DILocation(line: 182, column: 12, scope: !1163)
!1200 = !DILocation(line: 182, column: 7, scope: !1163)
!1201 = !DILocalVariable(name: "t4", scope: !1163, file: !3, line: 157, type: !98)
!1202 = !DILocation(line: 183, column: 18, scope: !1163)
!1203 = !DILocation(line: 183, column: 12, scope: !1163)
!1204 = !DILocation(line: 183, column: 7, scope: !1163)
!1205 = !DILocation(line: 185, column: 17, scope: !1163)
!1206 = !DILocation(line: 185, column: 14, scope: !1163)
!1207 = !DILocation(line: 185, column: 2, scope: !1163)
!1208 = distinct !DISubprogram(name: "c_print_results", linkageName: "_Z15c_print_resultsPcciiiiddS_iS_S_S_S_S_S_S_S_S_S_S_S_S_S_", scope: !3, file: !3, line: 191, type: !1209, scopeLine: 214, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !962)
!1209 = !DISubroutineType(types: !1210)
!1210 = !{null, !100, !101, !99, !99, !99, !99, !98, !98, !100, !99, !100, !100, !100, !100, !100, !100, !100, !100, !100, !100, !100, !100, !100, !100}
!1211 = !DILocalVariable(name: "name", arg: 1, scope: !1208, file: !3, line: 191, type: !100)
!1212 = !DILocation(line: 0, scope: !1208)
!1213 = !DILocalVariable(name: "class_npb", arg: 2, scope: !1208, file: !3, line: 192, type: !101)
!1214 = !DILocalVariable(name: "n1", arg: 3, scope: !1208, file: !3, line: 193, type: !99)
!1215 = !DILocalVariable(name: "n2", arg: 4, scope: !1208, file: !3, line: 194, type: !99)
!1216 = !DILocalVariable(name: "n3", arg: 5, scope: !1208, file: !3, line: 195, type: !99)
!1217 = !DILocalVariable(name: "niter", arg: 6, scope: !1208, file: !3, line: 196, type: !99)
!1218 = !DILocalVariable(name: "t", arg: 7, scope: !1208, file: !3, line: 197, type: !98)
!1219 = !DILocalVariable(name: "mops", arg: 8, scope: !1208, file: !3, line: 198, type: !98)
!1220 = !DILocalVariable(name: "optype", arg: 9, scope: !1208, file: !3, line: 199, type: !100)
!1221 = !DILocalVariable(name: "passed_verification", arg: 10, scope: !1208, file: !3, line: 200, type: !99)
!1222 = !DILocalVariable(name: "npbversion", arg: 11, scope: !1208, file: !3, line: 201, type: !100)
!1223 = !DILocalVariable(name: "compiletime", arg: 12, scope: !1208, file: !3, line: 202, type: !100)
!1224 = !DILocalVariable(name: "compilerversion", arg: 13, scope: !1208, file: !3, line: 203, type: !100)
!1225 = !DILocalVariable(name: "libversion", arg: 14, scope: !1208, file: !3, line: 204, type: !100)
!1226 = !DILocalVariable(name: "cpu_device", arg: 15, scope: !1208, file: !3, line: 205, type: !100)
!1227 = !DILocalVariable(name: "gpu_device", arg: 16, scope: !1208, file: !3, line: 206, type: !100)
!1228 = !DILocalVariable(name: "gpu_config", arg: 17, scope: !1208, file: !3, line: 207, type: !100)
!1229 = !DILocalVariable(name: "cc", arg: 18, scope: !1208, file: !3, line: 208, type: !100)
!1230 = !DILocalVariable(name: "clink", arg: 19, scope: !1208, file: !3, line: 209, type: !100)
!1231 = !DILocalVariable(name: "c_lib", arg: 20, scope: !1208, file: !3, line: 210, type: !100)
!1232 = !DILocalVariable(name: "c_inc", arg: 21, scope: !1208, file: !3, line: 211, type: !100)
!1233 = !DILocalVariable(name: "cflags", arg: 22, scope: !1208, file: !3, line: 212, type: !100)
!1234 = !DILocalVariable(name: "clinkflags", arg: 23, scope: !1208, file: !3, line: 213, type: !100)
!1235 = !DILocalVariable(name: "rand", arg: 24, scope: !1208, file: !3, line: 214, type: !100)
!1236 = !DILocation(line: 215, column: 4, scope: !1208)
!1237 = !DILocation(line: 216, column: 61, scope: !1208)
!1238 = !DILocation(line: 216, column: 4, scope: !1208)
!1239 = !DILocation(line: 217, column: 8, scope: !1240)
!1240 = distinct !DILexicalBlock(scope: !1208, file: !3, line: 217, column: 7)
!1241 = !DILocation(line: 217, column: 15, scope: !1240)
!1242 = !DILocation(line: 217, column: 21, scope: !1240)
!1243 = !DILocation(line: 217, column: 24, scope: !1240)
!1244 = !DILocation(line: 217, column: 31, scope: !1240)
!1245 = !DILocation(line: 217, column: 7, scope: !1208)
!1246 = !DILocation(line: 218, column: 10, scope: !1247)
!1247 = distinct !DILexicalBlock(scope: !1248, file: !3, line: 218, column: 8)
!1248 = distinct !DILexicalBlock(scope: !1240, file: !3, line: 217, column: 38)
!1249 = !DILocation(line: 218, column: 8, scope: !1248)
!1250 = !DILocation(line: 219, column: 16, scope: !1251)
!1251 = distinct !DILexicalBlock(scope: !1247, file: !3, line: 218, column: 14)
!1252 = !DILocalVariable(name: "nn", scope: !1251, file: !3, line: 219, type: !313)
!1253 = !DILocation(line: 0, scope: !1251)
!1254 = !DILocation(line: 220, column: 11, scope: !1255)
!1255 = distinct !DILexicalBlock(scope: !1251, file: !3, line: 220, column: 9)
!1256 = !DILocation(line: 220, column: 9, scope: !1251)
!1257 = !DILocation(line: 220, column: 20, scope: !1258)
!1258 = distinct !DILexicalBlock(scope: !1255, file: !3, line: 220, column: 15)
!1259 = !DILocation(line: 220, column: 18, scope: !1258)
!1260 = !DILocation(line: 220, column: 23, scope: !1258)
!1261 = !DILocation(line: 221, column: 6, scope: !1251)
!1262 = !DILocation(line: 222, column: 5, scope: !1251)
!1263 = !DILocation(line: 223, column: 6, scope: !1264)
!1264 = distinct !DILexicalBlock(scope: !1247, file: !3, line: 222, column: 10)
!1265 = !DILocation(line: 225, column: 4, scope: !1248)
!1266 = !DILocalVariable(name: "size", scope: !1267, file: !3, line: 226, type: !1268)
!1267 = distinct !DILexicalBlock(scope: !1240, file: !3, line: 225, column: 9)
!1268 = !DICompositeType(tag: DW_TAG_array_type, baseType: !101, size: 128, elements: !1269)
!1269 = !{!1270}
!1270 = !DISubrange(count: 16)
!1271 = !DILocation(line: 226, column: 10, scope: !1267)
!1272 = !DILocation(line: 228, column: 11, scope: !1273)
!1273 = distinct !DILexicalBlock(scope: !1267, file: !3, line: 228, column: 8)
!1274 = !DILocation(line: 228, column: 16, scope: !1273)
!1275 = !DILocation(line: 228, column: 22, scope: !1273)
!1276 = !DILocation(line: 228, column: 8, scope: !1267)
!1277 = !DILocation(line: 229, column: 10, scope: !1278)
!1278 = distinct !DILexicalBlock(scope: !1279, file: !3, line: 229, column: 9)
!1279 = distinct !DILexicalBlock(scope: !1273, file: !3, line: 228, column: 27)
!1280 = !DILocation(line: 229, column: 17, scope: !1278)
!1281 = !DILocation(line: 229, column: 23, scope: !1278)
!1282 = !DILocation(line: 229, column: 26, scope: !1278)
!1283 = !DILocation(line: 229, column: 33, scope: !1278)
!1284 = !DILocation(line: 229, column: 9, scope: !1279)
!1285 = !DILocation(line: 230, column: 15, scope: !1286)
!1286 = distinct !DILexicalBlock(scope: !1278, file: !3, line: 229, column: 40)
!1287 = !DILocation(line: 230, column: 41, scope: !1286)
!1288 = !DILocation(line: 230, column: 32, scope: !1286)
!1289 = !DILocation(line: 230, column: 7, scope: !1286)
!1290 = !DILocalVariable(name: "j", scope: !1267, file: !3, line: 227, type: !99)
!1291 = !DILocation(line: 0, scope: !1267)
!1292 = !DILocation(line: 232, column: 10, scope: !1293)
!1293 = distinct !DILexicalBlock(scope: !1286, file: !3, line: 232, column: 10)
!1294 = !DILocation(line: 232, column: 18, scope: !1293)
!1295 = !DILocation(line: 232, column: 10, scope: !1286)
!1296 = !DILocation(line: 233, column: 8, scope: !1297)
!1297 = distinct !DILexicalBlock(scope: !1293, file: !3, line: 232, column: 25)
!1298 = !DILocation(line: 233, column: 16, scope: !1297)
!1299 = !DILocation(line: 234, column: 9, scope: !1297)
!1300 = !DILocation(line: 235, column: 7, scope: !1297)
!1301 = !DILocation(line: 0, scope: !1286)
!1302 = !DILocation(line: 236, column: 13, scope: !1286)
!1303 = !DILocation(line: 236, column: 7, scope: !1286)
!1304 = !DILocation(line: 236, column: 17, scope: !1286)
!1305 = !DILocation(line: 237, column: 52, scope: !1286)
!1306 = !DILocation(line: 237, column: 7, scope: !1286)
!1307 = !DILocation(line: 238, column: 6, scope: !1286)
!1308 = !DILocation(line: 239, column: 7, scope: !1309)
!1309 = distinct !DILexicalBlock(scope: !1278, file: !3, line: 238, column: 11)
!1310 = !DILocation(line: 241, column: 5, scope: !1279)
!1311 = !DILocation(line: 242, column: 6, scope: !1312)
!1312 = distinct !DILexicalBlock(scope: !1273, file: !3, line: 241, column: 10)
!1313 = !DILocation(line: 245, column: 4, scope: !1208)
!1314 = !DILocation(line: 246, column: 4, scope: !1208)
!1315 = !DILocation(line: 247, column: 4, scope: !1208)
!1316 = !DILocation(line: 248, column: 4, scope: !1208)
!1317 = !DILocation(line: 249, column: 27, scope: !1318)
!1318 = distinct !DILexicalBlock(scope: !1208, file: !3, line: 249, column: 7)
!1319 = !DILocation(line: 249, column: 7, scope: !1208)
!1320 = !DILocation(line: 250, column: 5, scope: !1321)
!1321 = distinct !DILexicalBlock(scope: !1318, file: !3, line: 249, column: 31)
!1322 = !DILocation(line: 251, column: 4, scope: !1321)
!1323 = !DILocation(line: 251, column: 13, scope: !1324)
!1324 = distinct !DILexicalBlock(scope: !1318, file: !3, line: 251, column: 13)
!1325 = !DILocation(line: 251, column: 13, scope: !1318)
!1326 = !DILocation(line: 252, column: 5, scope: !1327)
!1327 = distinct !DILexicalBlock(scope: !1324, file: !3, line: 251, column: 33)
!1328 = !DILocation(line: 253, column: 4, scope: !1327)
!1329 = !DILocation(line: 254, column: 5, scope: !1330)
!1330 = distinct !DILexicalBlock(scope: !1324, file: !3, line: 253, column: 9)
!1331 = !DILocation(line: 256, column: 4, scope: !1208)
!1332 = !DILocation(line: 257, column: 4, scope: !1208)
!1333 = !DILocation(line: 258, column: 4, scope: !1208)
!1334 = !DILocation(line: 259, column: 4, scope: !1208)
!1335 = !DILocation(line: 260, column: 4, scope: !1208)
!1336 = !DILocation(line: 261, column: 4, scope: !1208)
!1337 = !DILocation(line: 262, column: 4, scope: !1208)
!1338 = !DILocation(line: 263, column: 4, scope: !1208)
!1339 = !DILocation(line: 264, column: 4, scope: !1208)
!1340 = !DILocation(line: 265, column: 4, scope: !1208)
!1341 = !DILocation(line: 266, column: 4, scope: !1208)
!1342 = !DILocation(line: 267, column: 4, scope: !1208)
!1343 = !DILocation(line: 268, column: 4, scope: !1208)
!1344 = !DILocation(line: 269, column: 4, scope: !1208)
!1345 = !DILocation(line: 270, column: 4, scope: !1208)
!1346 = !DILocation(line: 271, column: 4, scope: !1208)
!1347 = !DILocation(line: 272, column: 4, scope: !1208)
!1348 = !DILocation(line: 287, column: 4, scope: !1208)
!1349 = !DILocation(line: 288, column: 4, scope: !1208)
!1350 = !DILocation(line: 289, column: 4, scope: !1208)
!1351 = !DILocation(line: 290, column: 4, scope: !1208)
!1352 = !DILocation(line: 291, column: 4, scope: !1208)
!1353 = !DILocation(line: 292, column: 4, scope: !1208)
!1354 = !DILocation(line: 293, column: 4, scope: !1208)
!1355 = !DILocation(line: 294, column: 4, scope: !1208)
!1356 = !DILocation(line: 295, column: 4, scope: !1208)
!1357 = !DILocation(line: 296, column: 4, scope: !1208)
!1358 = !DILocation(line: 297, column: 3, scope: !1208)
!1359 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 301, type: !1360, scopeLine: 301, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !962)
!1360 = !DISubroutineType(types: !1361)
!1361 = !{!99, !99, !566}
!1362 = !DILocalVariable(name: "argc", arg: 1, scope: !1359, file: !3, line: 301, type: !99)
!1363 = !DILocation(line: 0, scope: !1359)
!1364 = !DILocalVariable(name: "argv", arg: 2, scope: !1359, file: !3, line: 301, type: !566)
!1365 = !DILocalVariable(name: "t1", scope: !1359, file: !3, line: 308, type: !98)
!1366 = !DILocation(line: 308, column: 15, scope: !1359)
!1367 = !DILocalVariable(name: "size", scope: !1359, file: !3, line: 313, type: !1268)
!1368 = !DILocation(line: 313, column: 7, scope: !1359)
!1369 = !DILocation(line: 323, column: 10, scope: !1359)
!1370 = !DILocation(line: 323, column: 26, scope: !1359)
!1371 = !DILocation(line: 323, column: 2, scope: !1359)
!1372 = !DILocalVariable(name: "j", scope: !1359, file: !3, line: 311, type: !99)
!1373 = !DILocation(line: 325, column: 5, scope: !1374)
!1374 = distinct !DILexicalBlock(scope: !1359, file: !3, line: 325, column: 5)
!1375 = !DILocation(line: 325, column: 12, scope: !1374)
!1376 = !DILocation(line: 325, column: 5, scope: !1359)
!1377 = !DILocation(line: 325, column: 20, scope: !1378)
!1378 = distinct !DILexicalBlock(scope: !1374, file: !3, line: 325, column: 18)
!1379 = !DILocation(line: 325, column: 23, scope: !1378)
!1380 = !DILocation(line: 326, column: 8, scope: !1359)
!1381 = !DILocation(line: 326, column: 2, scope: !1359)
!1382 = !DILocation(line: 326, column: 12, scope: !1359)
!1383 = !DILocation(line: 327, column: 2, scope: !1359)
!1384 = !DILocation(line: 328, column: 56, scope: !1359)
!1385 = !DILocation(line: 328, column: 2, scope: !1359)
!1386 = !DILocalVariable(name: "verified", scope: !1359, file: !3, line: 312, type: !1387)
!1387 = !DIDerivedType(tag: DW_TAG_typedef, name: "boolean", file: !1388, line: 80, baseType: !99)
!1388 = !DIFile(filename: "./../common/npb-CPP.hpp", directory: "/scratch/ah7226/NPB-GPU/CUDA/EP")
!1389 = !DILocation(line: 332, column: 5, scope: !1359)
!1390 = !DILocalVariable(name: "i", scope: !1359, file: !3, line: 311, type: !99)
!1391 = !DILocation(line: 334, column: 6, scope: !1392)
!1392 = distinct !DILexicalBlock(scope: !1359, file: !3, line: 334, column: 2)
!1393 = !DILocation(line: 0, scope: !1392)
!1394 = !DILocation(line: 334, column: 12, scope: !1395)
!1395 = distinct !DILexicalBlock(scope: !1392, file: !3, line: 334, column: 2)
!1396 = !DILocation(line: 334, column: 2, scope: !1392)
!1397 = !DILocation(line: 335, column: 15, scope: !1398)
!1398 = distinct !DILexicalBlock(scope: !1395, file: !3, line: 334, column: 23)
!1399 = !DILocation(line: 335, column: 3, scope: !1398)
!1400 = !DILocation(line: 336, column: 2, scope: !1398)
!1401 = !DILocation(line: 334, column: 20, scope: !1395)
!1402 = !DILocation(line: 334, column: 2, scope: !1395)
!1403 = distinct !{!1403, !1396, !1404}
!1404 = !DILocation(line: 336, column: 2, scope: !1392)
!1405 = !DILocation(line: 338, column: 7, scope: !1359)
!1406 = !DILocalVariable(name: "an", scope: !1359, file: !3, line: 309, type: !98)
!1407 = !DILocalVariable(name: "gc", scope: !1359, file: !3, line: 309, type: !98)
!1408 = !DILocalVariable(name: "sx", scope: !1359, file: !3, line: 309, type: !98)
!1409 = !DILocalVariable(name: "sy", scope: !1359, file: !3, line: 309, type: !98)
!1410 = !DILocation(line: 343, column: 6, scope: !1411)
!1411 = distinct !DILexicalBlock(scope: !1359, file: !3, line: 343, column: 2)
!1412 = !DILocation(line: 0, scope: !1411)
!1413 = !DILocation(line: 343, column: 12, scope: !1414)
!1414 = distinct !DILexicalBlock(scope: !1411, file: !3, line: 343, column: 2)
!1415 = !DILocation(line: 343, column: 2, scope: !1411)
!1416 = !DILocation(line: 344, column: 3, scope: !1417)
!1417 = distinct !DILexicalBlock(scope: !1414, file: !3, line: 343, column: 21)
!1418 = !DILocation(line: 344, column: 8, scope: !1417)
!1419 = !DILocation(line: 345, column: 2, scope: !1417)
!1420 = !DILocation(line: 343, column: 18, scope: !1414)
!1421 = !DILocation(line: 343, column: 2, scope: !1414)
!1422 = distinct !{!1422, !1415, !1423}
!1423 = !DILocation(line: 345, column: 2, scope: !1411)
!1424 = !DILocation(line: 347, column: 2, scope: !1359)
!1425 = !DILocation(line: 352, column: 15, scope: !1359)
!1426 = !DILocation(line: 353, column: 3, scope: !1359)
!1427 = !DILocation(line: 352, column: 12, scope: !1359)
!1428 = !{!""}
!1429 = !DILocation(line: 353, column: 24, scope: !1359)
!1430 = !DILocation(line: 354, column: 5, scope: !1359)
!1431 = !DILocation(line: 355, column: 5, scope: !1359)
!1432 = !DILocation(line: 361, column: 13, scope: !1359)
!1433 = !DILocation(line: 361, column: 21, scope: !1359)
!1434 = !DILocation(line: 361, column: 31, scope: !1359)
!1435 = !DILocation(line: 361, column: 2, scope: !1359)
!1436 = !DILocalVariable(name: "block", scope: !1359, file: !3, line: 311, type: !99)
!1437 = !DILocation(line: 365, column: 6, scope: !1438)
!1438 = distinct !DILexicalBlock(scope: !1359, file: !3, line: 365, column: 2)
!1439 = !DILocation(line: 365, column: 21, scope: !1440)
!1440 = distinct !DILexicalBlock(scope: !1438, file: !3, line: 365, column: 2)
!1441 = !DILocation(line: 365, column: 20, scope: !1440)
!1442 = !DILocation(line: 365, column: 2, scope: !1438)
!1443 = !DILocation(line: 366, column: 7, scope: !1444)
!1444 = distinct !DILexicalBlock(scope: !1445, file: !3, line: 366, column: 3)
!1445 = distinct !DILexicalBlock(scope: !1440, file: !3, line: 365, column: 46)
!1446 = !DILocation(line: 0, scope: !1444)
!1447 = !DILocation(line: 366, column: 13, scope: !1448)
!1448 = distinct !DILexicalBlock(scope: !1444, file: !3, line: 366, column: 3)
!1449 = !DILocation(line: 366, column: 3, scope: !1444)
!1450 = !DILocation(line: 367, column: 10, scope: !1451)
!1451 = distinct !DILexicalBlock(scope: !1448, file: !3, line: 366, column: 22)
!1452 = !DILocation(line: 367, column: 22, scope: !1451)
!1453 = !DILocation(line: 367, column: 25, scope: !1451)
!1454 = !DILocation(line: 367, column: 4, scope: !1451)
!1455 = !DILocation(line: 367, column: 8, scope: !1451)
!1456 = !DILocation(line: 368, column: 3, scope: !1451)
!1457 = !DILocation(line: 366, column: 19, scope: !1448)
!1458 = !DILocation(line: 366, column: 3, scope: !1448)
!1459 = distinct !{!1459, !1449, !1460}
!1460 = !DILocation(line: 368, column: 3, scope: !1444)
!1461 = !DILocation(line: 369, column: 7, scope: !1445)
!1462 = !DILocation(line: 369, column: 5, scope: !1445)
!1463 = !DILocation(line: 370, column: 7, scope: !1445)
!1464 = !DILocation(line: 370, column: 5, scope: !1445)
!1465 = !DILocation(line: 371, column: 2, scope: !1445)
!1466 = !DILocation(line: 365, column: 43, scope: !1440)
!1467 = !DILocation(line: 365, column: 2, scope: !1440)
!1468 = distinct !{!1468, !1442, !1469}
!1469 = !DILocation(line: 371, column: 2, scope: !1438)
!1470 = !DILocation(line: 372, column: 6, scope: !1471)
!1471 = distinct !DILexicalBlock(scope: !1359, file: !3, line: 372, column: 2)
!1472 = !DILocation(line: 372, column: 12, scope: !1473)
!1473 = distinct !DILexicalBlock(scope: !1471, file: !3, line: 372, column: 2)
!1474 = !DILocation(line: 372, column: 2, scope: !1471)
!1475 = !DILocation(line: 373, column: 7, scope: !1476)
!1476 = distinct !DILexicalBlock(scope: !1473, file: !3, line: 372, column: 21)
!1477 = !DILocation(line: 373, column: 5, scope: !1476)
!1478 = !DILocation(line: 374, column: 2, scope: !1476)
!1479 = !DILocation(line: 372, column: 18, scope: !1473)
!1480 = !DILocation(line: 372, column: 2, scope: !1473)
!1481 = distinct !{!1481, !1474, !1482}
!1482 = !DILocation(line: 374, column: 2, scope: !1471)
!1483 = !DILocalVariable(name: "nit", scope: !1359, file: !3, line: 311, type: !99)
!1484 = !DILocalVariable(name: "sx_verify_value", scope: !1359, file: !3, line: 310, type: !98)
!1485 = !DILocalVariable(name: "sy_verify_value", scope: !1359, file: !3, line: 310, type: !98)
!1486 = !DILocation(line: 402, column: 5, scope: !1487)
!1487 = distinct !DILexicalBlock(scope: !1359, file: !3, line: 402, column: 5)
!1488 = !DILocation(line: 402, column: 5, scope: !1359)
!1489 = !DILocation(line: 403, column: 21, scope: !1490)
!1490 = distinct !DILexicalBlock(scope: !1487, file: !3, line: 402, column: 14)
!1491 = !DILocation(line: 403, column: 40, scope: !1490)
!1492 = !DILocation(line: 403, column: 12, scope: !1490)
!1493 = !DILocalVariable(name: "sx_err", scope: !1359, file: !3, line: 310, type: !98)
!1494 = !DILocation(line: 404, column: 21, scope: !1490)
!1495 = !DILocation(line: 404, column: 40, scope: !1490)
!1496 = !DILocation(line: 404, column: 12, scope: !1490)
!1497 = !DILocalVariable(name: "sy_err", scope: !1359, file: !3, line: 310, type: !98)
!1498 = !DILocation(line: 405, column: 23, scope: !1490)
!1499 = !DILocation(line: 405, column: 35, scope: !1490)
!1500 = !DILocation(line: 405, column: 46, scope: !1490)
!1501 = !DILocation(line: 0, scope: !1490)
!1502 = !DILocation(line: 405, column: 14, scope: !1490)
!1503 = !DILocation(line: 406, column: 2, scope: !1490)
!1504 = !DILocation(line: 407, column: 9, scope: !1359)
!1505 = !DILocation(line: 407, column: 22, scope: !1359)
!1506 = !DILocalVariable(name: "Mops", scope: !1359, file: !3, line: 308, type: !98)
!1507 = !DILocation(line: 409, column: 2, scope: !1359)
!1508 = !DILocation(line: 410, column: 2, scope: !1359)
!1509 = !DILocation(line: 411, column: 2, scope: !1359)
!1510 = !DILocation(line: 412, column: 2, scope: !1359)
!1511 = !DILocation(line: 413, column: 2, scope: !1359)
!1512 = !DILocation(line: 414, column: 2, scope: !1359)
!1513 = !DILocation(line: 415, column: 6, scope: !1514)
!1514 = distinct !DILexicalBlock(scope: !1359, file: !3, line: 415, column: 2)
!1515 = !DILocation(line: 0, scope: !1514)
!1516 = !DILocation(line: 415, column: 12, scope: !1517)
!1517 = distinct !DILexicalBlock(scope: !1514, file: !3, line: 415, column: 2)
!1518 = !DILocation(line: 415, column: 2, scope: !1514)
!1519 = !DILocation(line: 416, column: 28, scope: !1520)
!1520 = distinct !DILexicalBlock(scope: !1517, file: !3, line: 415, column: 21)
!1521 = !DILocation(line: 416, column: 3, scope: !1520)
!1522 = !DILocation(line: 417, column: 2, scope: !1520)
!1523 = !DILocation(line: 415, column: 18, scope: !1517)
!1524 = !DILocation(line: 415, column: 2, scope: !1517)
!1525 = distinct !{!1525, !1518, !1526}
!1526 = !DILocation(line: 417, column: 2, scope: !1514)
!1527 = !DILocation(line: 407, column: 25, scope: !1359)
!1528 = !DILocalVariable(name: "gpu_config", scope: !1359, file: !3, line: 419, type: !139)
!1529 = !DILocation(line: 419, column: 7, scope: !1359)
!1530 = !DILocalVariable(name: "gpu_config_string", scope: !1359, file: !3, line: 420, type: !1531)
!1531 = !DICompositeType(tag: DW_TAG_array_type, baseType: !101, size: 16384, elements: !1532)
!1532 = !{!1533}
!1533 = !DISubrange(count: 2048)
!1534 = !DILocation(line: 420, column: 7, scope: !1359)
!1535 = !DILocation(line: 427, column: 10, scope: !1359)
!1536 = !DILocation(line: 427, column: 2, scope: !1359)
!1537 = !DILocation(line: 428, column: 9, scope: !1359)
!1538 = !DILocation(line: 428, column: 28, scope: !1359)
!1539 = !DILocation(line: 428, column: 2, scope: !1359)
!1540 = !DILocation(line: 429, column: 10, scope: !1359)
!1541 = !DILocation(line: 429, column: 45, scope: !1359)
!1542 = !DILocation(line: 429, column: 2, scope: !1359)
!1543 = !DILocation(line: 430, column: 9, scope: !1359)
!1544 = !DILocation(line: 430, column: 28, scope: !1359)
!1545 = !DILocation(line: 430, column: 2, scope: !1359)
!1546 = !DILocation(line: 449, column: 4, scope: !1359)
!1547 = !DILocation(line: 433, column: 2, scope: !1359)
!1548 = !DILocation(line: 458, column: 2, scope: !1359)
!1549 = !DILocation(line: 460, column: 2, scope: !1359)
!1550 = distinct !DISubprogram(name: "setup_gpu", linkageName: "_ZL9setup_gpuv", scope: !3, file: !3, line: 576, type: !472, scopeLine: 576, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !962)
!1551 = !DILocation(line: 625, column: 49, scope: !1552)
!1552 = distinct !DILexicalBlock(scope: !1550, file: !3, line: 624, column: 5)
!1553 = !DILocation(line: 625, column: 25, scope: !1552)
!1554 = !DILocation(line: 624, column: 5, scope: !1550)
!1555 = !DILocation(line: 626, column: 21, scope: !1556)
!1556 = distinct !DILexicalBlock(scope: !1552, file: !3, line: 625, column: 69)
!1557 = !DILocation(line: 627, column: 2, scope: !1556)
!1558 = !DILocation(line: 628, column: 45, scope: !1559)
!1559 = distinct !DILexicalBlock(scope: !1552, file: !3, line: 627, column: 7)
!1560 = !DILocation(line: 628, column: 21, scope: !1559)
!1561 = !DILocation(line: 631, column: 45, scope: !1550)
!1562 = !DILocation(line: 631, column: 36, scope: !1550)
!1563 = !DILocation(line: 631, column: 21, scope: !1550)
!1564 = !DILocation(line: 631, column: 20, scope: !1550)
!1565 = !DILocation(line: 631, column: 18, scope: !1550)
!1566 = !DILocation(line: 633, column: 11, scope: !1550)
!1567 = !DILocation(line: 633, column: 27, scope: !1550)
!1568 = !DILocation(line: 633, column: 32, scope: !1550)
!1569 = !DILocation(line: 633, column: 9, scope: !1550)
!1570 = !DILocation(line: 634, column: 12, scope: !1550)
!1571 = !DILocation(line: 634, column: 28, scope: !1550)
!1572 = !DILocation(line: 634, column: 10, scope: !1550)
!1573 = !DILocation(line: 635, column: 12, scope: !1550)
!1574 = !DILocation(line: 635, column: 28, scope: !1550)
!1575 = !DILocation(line: 635, column: 10, scope: !1550)
!1576 = !DILocation(line: 637, column: 25, scope: !1550)
!1577 = !{!"0"}
!1578 = !DILocation(line: 637, column: 18, scope: !1550)
!1579 = !{!1577}
!1580 = !DILocation(line: 637, column: 9, scope: !1550)
!1581 = !DILocation(line: 637, column: 8, scope: !1550)
!1582 = !DILocation(line: 638, column: 26, scope: !1550)
!1583 = !{!"1"}
!1584 = !DILocation(line: 638, column: 19, scope: !1550)
!1585 = !{!1583}
!1586 = !DILocation(line: 638, column: 10, scope: !1550)
!1587 = !DILocation(line: 638, column: 9, scope: !1550)
!1588 = !DILocation(line: 639, column: 26, scope: !1550)
!1589 = !{!"2"}
!1590 = !DILocation(line: 639, column: 19, scope: !1550)
!1591 = !{!1589}
!1592 = !DILocation(line: 639, column: 10, scope: !1550)
!1593 = !DILocation(line: 639, column: 9, scope: !1550)
!1594 = !DILocation(line: 644, column: 1, scope: !1550)
!1595 = distinct !DISubprogram(name: "release_gpu", linkageName: "_ZL11release_gpuv", scope: !3, file: !3, line: 570, type: !472, scopeLine: 570, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !962)
!1596 = !DILocation(line: 574, column: 1, scope: !1595)
!1597 = !DILocalVariable(name: "q_global", arg: 1, scope: !1598, file: !3, line: 463, type: !97)
!1598 = distinct !DISubprogram(name: "gpu_kernel", linkageName: "_Z10gpu_kernelPdS_S_d", scope: !3, file: !3, line: 463, type: !1599, scopeLine: 466, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !961, retainedNodes: !962)
!1599 = !DISubroutineType(types: !1600)
!1600 = !{null, !97, !97, !97, !98}
!1601 = !DILocation(line: 0, scope: !1598)
!1602 = !DILocalVariable(name: "sx_global", arg: 2, scope: !1598, file: !3, line: 464, type: !97)
!1603 = !DILocalVariable(name: "sy_global", arg: 3, scope: !1598, file: !3, line: 465, type: !97)
!1604 = !DILocalVariable(name: "an", arg: 4, scope: !1598, file: !3, line: 466, type: !98)
!1605 = !DILocalVariable(name: "x_local", scope: !1598, file: !3, line: 467, type: !1606)
!1606 = !DICompositeType(tag: DW_TAG_array_type, baseType: !98, size: 16384, elements: !140)
!1607 = !DILocation(line: 467, column: 9, scope: !1598)
!1608 = !DILocalVariable(name: "q_local", scope: !1598, file: !3, line: 468, type: !1609)
!1609 = !DICompositeType(tag: DW_TAG_array_type, baseType: !98, size: 640, elements: !1610)
!1610 = !{!1611}
!1611 = !DISubrange(count: 10)
!1612 = !DILocation(line: 468, column: 9, scope: !1598)
!1613 = !DILocalVariable(name: "t1", scope: !1598, file: !3, line: 470, type: !98)
!1614 = !DILocation(line: 470, column: 9, scope: !1598)
!1615 = !DILocalVariable(name: "t2", scope: !1598, file: !3, line: 470, type: !98)
!1616 = !DILocation(line: 470, column: 13, scope: !1598)
!1617 = !DILocalVariable(name: "seed", scope: !1598, file: !3, line: 470, type: !98)
!1618 = !DILocation(line: 470, column: 33, scope: !1598)
!1619 = !DILocation(line: 473, column: 2, scope: !1598)
!1620 = !DILocation(line: 473, column: 12, scope: !1598)
!1621 = !DILocation(line: 474, column: 2, scope: !1598)
!1622 = !DILocation(line: 474, column: 12, scope: !1598)
!1623 = !DILocation(line: 475, column: 2, scope: !1598)
!1624 = !DILocation(line: 475, column: 12, scope: !1598)
!1625 = !DILocation(line: 476, column: 2, scope: !1598)
!1626 = !DILocation(line: 476, column: 12, scope: !1598)
!1627 = !DILocation(line: 477, column: 2, scope: !1598)
!1628 = !DILocation(line: 477, column: 12, scope: !1598)
!1629 = !DILocation(line: 478, column: 2, scope: !1598)
!1630 = !DILocation(line: 478, column: 12, scope: !1598)
!1631 = !DILocation(line: 479, column: 2, scope: !1598)
!1632 = !DILocation(line: 479, column: 12, scope: !1598)
!1633 = !DILocation(line: 480, column: 2, scope: !1598)
!1634 = !DILocation(line: 480, column: 12, scope: !1598)
!1635 = !DILocation(line: 481, column: 2, scope: !1598)
!1636 = !DILocation(line: 481, column: 12, scope: !1598)
!1637 = !DILocation(line: 482, column: 2, scope: !1598)
!1638 = !DILocation(line: 482, column: 12, scope: !1598)
!1639 = !DILocalVariable(name: "sx_local", scope: !1598, file: !3, line: 469, type: !98)
!1640 = !DILocalVariable(name: "sy_local", scope: !1598, file: !3, line: 469, type: !98)
!1641 = !DILocation(line: 486, column: 15, scope: !1598)
!1642 = !DILocation(line: 486, column: 26, scope: !1598)
!1643 = !DILocalVariable(name: "kk", scope: !1598, file: !3, line: 471, type: !99)
!1644 = !DILocation(line: 488, column: 7, scope: !1645)
!1645 = distinct !DILexicalBlock(scope: !1598, file: !3, line: 488, column: 5)
!1646 = !DILocation(line: 488, column: 5, scope: !1598)
!1647 = !DILocation(line: 488, column: 13, scope: !1648)
!1648 = distinct !DILexicalBlock(scope: !1645, file: !3, line: 488, column: 12)
!1649 = !DILocation(line: 490, column: 4, scope: !1598)
!1650 = !DILocation(line: 491, column: 4, scope: !1598)
!1651 = !DILocalVariable(name: "i", scope: !1598, file: !3, line: 471, type: !99)
!1652 = !DILocation(line: 494, column: 6, scope: !1653)
!1653 = distinct !DILexicalBlock(scope: !1598, file: !3, line: 494, column: 2)
!1654 = !DILocation(line: 0, scope: !1653)
!1655 = !DILocation(line: 494, column: 12, scope: !1656)
!1656 = distinct !DILexicalBlock(scope: !1653, file: !3, line: 494, column: 2)
!1657 = !DILocation(line: 494, column: 2, scope: !1653)
!1658 = !DILocation(line: 495, column: 8, scope: !1659)
!1659 = distinct !DILexicalBlock(scope: !1656, file: !3, line: 494, column: 23)
!1660 = !DILocalVariable(name: "ik", scope: !1598, file: !3, line: 471, type: !99)
!1661 = !DILocation(line: 496, column: 8, scope: !1662)
!1662 = distinct !DILexicalBlock(scope: !1659, file: !3, line: 496, column: 6)
!1663 = !DILocation(line: 496, column: 12, scope: !1662)
!1664 = !DILocation(line: 496, column: 6, scope: !1659)
!1665 = !DILocation(line: 496, column: 40, scope: !1666)
!1666 = distinct !DILexicalBlock(scope: !1662, file: !3, line: 496, column: 17)
!1667 = !DILocation(line: 496, column: 21, scope: !1666)
!1668 = !DILocalVariable(name: "t3", scope: !1598, file: !3, line: 470, type: !98)
!1669 = !DILocation(line: 496, column: 44, scope: !1666)
!1670 = !DILocation(line: 497, column: 8, scope: !1671)
!1671 = distinct !DILexicalBlock(scope: !1659, file: !3, line: 497, column: 6)
!1672 = !DILocation(line: 497, column: 6, scope: !1659)
!1673 = !DILocation(line: 497, column: 13, scope: !1674)
!1674 = distinct !DILexicalBlock(scope: !1671, file: !3, line: 497, column: 12)
!1675 = !DILocation(line: 498, column: 25, scope: !1659)
!1676 = !DILocation(line: 498, column: 6, scope: !1659)
!1677 = !DILocation(line: 500, column: 2, scope: !1659)
!1678 = !DILocation(line: 494, column: 20, scope: !1656)
!1679 = !DILocation(line: 494, column: 2, scope: !1656)
!1680 = distinct !{!1680, !1657, !1681}
!1681 = !DILocation(line: 500, column: 2, scope: !1653)
!1682 = !DILocation(line: 512, column: 7, scope: !1598)
!1683 = !DILocation(line: 512, column: 6, scope: !1598)
!1684 = !DILocalVariable(name: "ii", scope: !1598, file: !3, line: 471, type: !99)
!1685 = !DILocation(line: 513, column: 6, scope: !1686)
!1686 = distinct !DILexicalBlock(scope: !1598, file: !3, line: 513, column: 2)
!1687 = !DILocation(line: 483, column: 10, scope: !1598)
!1688 = !DILocation(line: 484, column: 10, scope: !1598)
!1689 = !DILocation(line: 0, scope: !1686)
!1690 = !DILocation(line: 513, column: 14, scope: !1691)
!1691 = distinct !DILexicalBlock(scope: !1686, file: !3, line: 513, column: 2)
!1692 = !DILocation(line: 513, column: 2, scope: !1686)
!1693 = !DILocation(line: 515, column: 44, scope: !1694)
!1694 = distinct !DILexicalBlock(scope: !1691, file: !3, line: 513, column: 39)
!1695 = !DILocation(line: 515, column: 3, scope: !1694)
!1696 = !DILocation(line: 522, column: 7, scope: !1697)
!1697 = distinct !DILexicalBlock(scope: !1694, file: !3, line: 522, column: 3)
!1698 = !DILocation(line: 522, column: 13, scope: !1699)
!1699 = distinct !DILexicalBlock(scope: !1697, file: !3, line: 522, column: 3)
!1700 = !DILocation(line: 522, column: 3, scope: !1697)
!1701 = !DILocation(line: 523, column: 20, scope: !1702)
!1702 = distinct !DILexicalBlock(scope: !1699, file: !3, line: 522, column: 33)
!1703 = !DILocation(line: 523, column: 11, scope: !1702)
!1704 = !DILocation(line: 523, column: 10, scope: !1702)
!1705 = !DILocation(line: 523, column: 23, scope: !1702)
!1706 = !DILocalVariable(name: "x1", scope: !1598, file: !3, line: 470, type: !98)
!1707 = !DILocation(line: 524, column: 20, scope: !1702)
!1708 = !DILocation(line: 524, column: 22, scope: !1702)
!1709 = !DILocation(line: 524, column: 11, scope: !1702)
!1710 = !DILocation(line: 524, column: 10, scope: !1702)
!1711 = !DILocation(line: 524, column: 25, scope: !1702)
!1712 = !DILocalVariable(name: "x2", scope: !1598, file: !3, line: 470, type: !98)
!1713 = !DILocation(line: 525, column: 9, scope: !1702)
!1714 = !DILocation(line: 525, column: 15, scope: !1702)
!1715 = !DILocation(line: 525, column: 12, scope: !1702)
!1716 = !DILocation(line: 525, column: 6, scope: !1702)
!1717 = !DILocation(line: 526, column: 7, scope: !1718)
!1718 = distinct !DILexicalBlock(scope: !1702, file: !3, line: 526, column: 7)
!1719 = !DILocation(line: 526, column: 9, scope: !1718)
!1720 = !DILocation(line: 526, column: 7, scope: !1702)
!1721 = !DILocation(line: 527, column: 22, scope: !1722)
!1722 = distinct !DILexicalBlock(scope: !1718, file: !3, line: 526, column: 15)
!1723 = !DILocalVariable(name: "a", arg: 1, scope: !1724, file: !1725, line: 225, type: !98)
!1724 = distinct !DISubprogram(name: "log", linkageName: "_ZL3logd", scope: !1725, file: !1725, line: 225, type: !407, scopeLine: 226, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !961, retainedNodes: !962)
!1725 = !DIFile(filename: "/usr/local/cuda/include/math_functions_dbl_ptx3.hpp", directory: "")
!1726 = !DILocation(line: 0, scope: !1724, inlinedAt: !1727)
!1727 = distinct !DILocation(line: 527, column: 18, scope: !1722)
!1728 = !DILocation(line: 529, column: 12, scope: !1722)
!1729 = !DILocation(line: 529, column: 11, scope: !1722)
!1730 = !DILocalVariable(name: "t4", scope: !1598, file: !3, line: 470, type: !98)
!1731 = !DILocalVariable(name: "f", arg: 1, scope: !1732, file: !691, line: 587, type: !98)
!1732 = distinct !DISubprogram(name: "fabs", linkageName: "_ZL4fabsd", scope: !691, file: !691, line: 587, type: !407, scopeLine: 588, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !961, retainedNodes: !962)
!1733 = !DILocation(line: 0, scope: !1732, inlinedAt: !1734)
!1734 = distinct !DILocation(line: 530, column: 7, scope: !1722)
!1735 = !DILocation(line: 0, scope: !1732, inlinedAt: !1736)
!1736 = distinct !DILocation(line: 530, column: 7, scope: !1722)
!1737 = !DILocation(line: 530, column: 7, scope: !1722)
!1738 = !DILocation(line: 0, scope: !1732, inlinedAt: !1739)
!1739 = distinct !DILocation(line: 530, column: 7, scope: !1722)
!1740 = !DILocation(line: 0, scope: !1732, inlinedAt: !1741)
!1741 = distinct !DILocation(line: 530, column: 7, scope: !1722)
!1742 = !DILocalVariable(name: "l", scope: !1598, file: !3, line: 471, type: !99)
!1743 = !DILocation(line: 531, column: 5, scope: !1722)
!1744 = !DILocation(line: 531, column: 15, scope: !1722)
!1745 = !DILocation(line: 532, column: 22, scope: !1722)
!1746 = !DILocation(line: 533, column: 13, scope: !1722)
!1747 = !DILocation(line: 534, column: 4, scope: !1722)
!1748 = !DILocation(line: 535, column: 3, scope: !1702)
!1749 = !DILocation(line: 522, column: 30, scope: !1699)
!1750 = !DILocation(line: 522, column: 3, scope: !1699)
!1751 = distinct !{!1751, !1700, !1752}
!1752 = !DILocation(line: 535, column: 3, scope: !1697)
!1753 = !DILocation(line: 536, column: 2, scope: !1694)
!1754 = !DILocation(line: 513, column: 24, scope: !1691)
!1755 = !DILocation(line: 513, column: 2, scope: !1691)
!1756 = distinct !{!1756, !1692, !1757}
!1757 = !DILocation(line: 536, column: 2, scope: !1686)
!1758 = !DILocation(line: 550, column: 1, scope: !1598)
