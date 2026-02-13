; ModuleID = 'cg.cu'
source_filename = "cg.cu"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

%struct.__cuda_builtin_blockIdx_t = type { i8 }
%struct.__cuda_builtin_blockDim_t = type { i8 }
%struct.__cuda_builtin_threadIdx_t = type { i8 }

@blockIdx = extern_weak dso_local addrspace(1) global %struct.__cuda_builtin_blockIdx_t, align 1
@blockDim = extern_weak dso_local addrspace(1) global %struct.__cuda_builtin_blockDim_t, align 1
@threadIdx = extern_weak dso_local addrspace(1) global %struct.__cuda_builtin_threadIdx_t, align 1
@extern_share_data = external dso_local addrspace(3) global [0 x double], align 8

; Function Attrs: convergent noinline nounwind
define dso_local void @_Z21gpu_kernel_one_devicePdS_S_S_S_(double* %p, double* %q, double* %r, double* %x, double* %z) #0 !dbg !791 {
entry:
  %p.addr = alloca double*, align 8
  %q.addr = alloca double*, align 8
  %r.addr = alloca double*, align 8
  %x.addr = alloca double*, align 8
  %z.addr = alloca double*, align 8
  %thread_id = alloca i32, align 4
  %x_value = alloca double, align 8
  store double* %p, double** %p.addr, align 8
  call void @llvm.dbg.declare(metadata double** %p.addr, metadata !794, metadata !DIExpression()), !dbg !795
  store double* %q, double** %q.addr, align 8
  call void @llvm.dbg.declare(metadata double** %q.addr, metadata !796, metadata !DIExpression()), !dbg !797
  store double* %r, double** %r.addr, align 8
  call void @llvm.dbg.declare(metadata double** %r.addr, metadata !798, metadata !DIExpression()), !dbg !799
  store double* %x, double** %x.addr, align 8
  call void @llvm.dbg.declare(metadata double** %x.addr, metadata !800, metadata !DIExpression()), !dbg !801
  store double* %z, double** %z.addr, align 8
  call void @llvm.dbg.declare(metadata double** %z.addr, metadata !802, metadata !DIExpression()), !dbg !803
  call void @llvm.dbg.declare(metadata i32* %thread_id, metadata !804, metadata !DIExpression()), !dbg !805
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #4, !dbg !806, !range !843
  %1 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #4, !dbg !844, !range !888
  %mul = mul i32 %0, %1, !dbg !889
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4, !dbg !890, !range !918
  %add = add i32 %mul, %2, !dbg !919
  store i32 %add, i32* %thread_id, align 4, !dbg !805
  %3 = load i32, i32* %thread_id, align 4, !dbg !920
  %cmp = icmp sge i32 %3, 14000, !dbg !922
  br i1 %cmp, label %if.then, label %if.end, !dbg !923

if.then:                                          ; preds = %entry
  br label %return, !dbg !924

if.end:                                           ; preds = %entry
  %4 = load double*, double** %q.addr, align 8, !dbg !926
  %5 = load i32, i32* %thread_id, align 4, !dbg !927
  %idxprom = sext i32 %5 to i64, !dbg !926
  %arrayidx = getelementptr inbounds double, double* %4, i64 %idxprom, !dbg !926
  store double 0.000000e+00, double* %arrayidx, align 8, !dbg !928
  %6 = load double*, double** %z.addr, align 8, !dbg !929
  %7 = load i32, i32* %thread_id, align 4, !dbg !930
  %idxprom3 = sext i32 %7 to i64, !dbg !929
  %arrayidx4 = getelementptr inbounds double, double* %6, i64 %idxprom3, !dbg !929
  store double 0.000000e+00, double* %arrayidx4, align 8, !dbg !931
  call void @llvm.dbg.declare(metadata double* %x_value, metadata !932, metadata !DIExpression()), !dbg !933
  %8 = load double*, double** %x.addr, align 8, !dbg !934
  %9 = load i32, i32* %thread_id, align 4, !dbg !935
  %idxprom5 = sext i32 %9 to i64, !dbg !934
  %arrayidx6 = getelementptr inbounds double, double* %8, i64 %idxprom5, !dbg !934
  %10 = load double, double* %arrayidx6, align 8, !dbg !934
  store double %10, double* %x_value, align 8, !dbg !933
  %11 = load double, double* %x_value, align 8, !dbg !936
  %12 = load double*, double** %r.addr, align 8, !dbg !937
  %13 = load i32, i32* %thread_id, align 4, !dbg !938
  %idxprom7 = sext i32 %13 to i64, !dbg !937
  %arrayidx8 = getelementptr inbounds double, double* %12, i64 %idxprom7, !dbg !937
  store double %11, double* %arrayidx8, align 8, !dbg !939
  %14 = load double, double* %x_value, align 8, !dbg !940
  %15 = load double*, double** %p.addr, align 8, !dbg !941
  %16 = load i32, i32* %thread_id, align 4, !dbg !942
  %idxprom9 = sext i32 %16 to i64, !dbg !941
  %arrayidx10 = getelementptr inbounds double, double* %15, i64 %idxprom9, !dbg !941
  store double %14, double* %arrayidx10, align 8, !dbg !943
  br label %return, !dbg !944

return:                                           ; preds = %if.end, %if.then
  ret void, !dbg !944
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: convergent noinline nounwind
define dso_local void @_Z21gpu_kernel_two_devicePdS_S_(double* %r, double* %rho, double* %global_data) #0 !dbg !945 {
entry:
  %r.addr = alloca double*, align 8
  %rho.addr = alloca double*, align 8
  %global_data.addr = alloca double*, align 8
  %share_data = alloca double*, align 8
  %thread_id = alloca i32, align 4
  %local_id = alloca i32, align 4
  %r_value = alloca double, align 8
  %i = alloca i32, align 4
  store double* %r, double** %r.addr, align 8
  call void @llvm.dbg.declare(metadata double** %r.addr, metadata !948, metadata !DIExpression()), !dbg !949
  store double* %rho, double** %rho.addr, align 8
  call void @llvm.dbg.declare(metadata double** %rho.addr, metadata !950, metadata !DIExpression()), !dbg !951
  store double* %global_data, double** %global_data.addr, align 8
  call void @llvm.dbg.declare(metadata double** %global_data.addr, metadata !952, metadata !DIExpression()), !dbg !953
  call void @llvm.dbg.declare(metadata double** %share_data, metadata !954, metadata !DIExpression()), !dbg !955
  store double* getelementptr inbounds ([0 x double], [0 x double]* addrspacecast ([0 x double] addrspace(3)* @extern_share_data to [0 x double]*), i64 0, i64 0), double** %share_data, align 8, !dbg !955
  call void @llvm.dbg.declare(metadata i32* %thread_id, metadata !956, metadata !DIExpression()), !dbg !957
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #4, !dbg !958, !range !843
  %1 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #4, !dbg !960, !range !888
  %mul = mul i32 %0, %1, !dbg !962
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4, !dbg !963, !range !918
  %add = add i32 %mul, %2, !dbg !965
  store i32 %add, i32* %thread_id, align 4, !dbg !957
  call void @llvm.dbg.declare(metadata i32* %local_id, metadata !966, metadata !DIExpression()), !dbg !967
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4, !dbg !968, !range !918
  store i32 %3, i32* %local_id, align 4, !dbg !967
  %4 = load double*, double** %share_data, align 8, !dbg !970
  %5 = load i32, i32* %local_id, align 4, !dbg !971
  %idxprom = sext i32 %5 to i64, !dbg !970
  %arrayidx = getelementptr inbounds double, double* %4, i64 %idxprom, !dbg !970
  store double 0.000000e+00, double* %arrayidx, align 8, !dbg !972
  %6 = load i32, i32* %thread_id, align 4, !dbg !973
  %cmp = icmp slt i32 %6, 14000, !dbg !975
  br i1 %cmp, label %if.then, label %if.end, !dbg !976

if.then:                                          ; preds = %entry
  call void @llvm.dbg.declare(metadata double* %r_value, metadata !977, metadata !DIExpression()), !dbg !979
  %7 = load double*, double** %r.addr, align 8, !dbg !980
  %8 = load i32, i32* %thread_id, align 4, !dbg !981
  %idxprom4 = sext i32 %8 to i64, !dbg !980
  %arrayidx5 = getelementptr inbounds double, double* %7, i64 %idxprom4, !dbg !980
  %9 = load double, double* %arrayidx5, align 8, !dbg !980
  store double %9, double* %r_value, align 8, !dbg !979
  %10 = load double, double* %r_value, align 8, !dbg !982
  %11 = load double, double* %r_value, align 8, !dbg !983
  %mul6 = fmul contract double %10, %11, !dbg !984
  %12 = load double*, double** %share_data, align 8, !dbg !985
  %13 = load i32, i32* %local_id, align 4, !dbg !986
  %idxprom7 = sext i32 %13 to i64, !dbg !985
  %arrayidx8 = getelementptr inbounds double, double* %12, i64 %idxprom7, !dbg !985
  store double %mul6, double* %arrayidx8, align 8, !dbg !987
  br label %if.end, !dbg !988

if.end:                                           ; preds = %if.then, %entry
  call void @llvm.nvvm.barrier0(), !dbg !989
  %14 = load i32, i32* %local_id, align 4, !dbg !990
  %cmp9 = icmp eq i32 %14, 0, !dbg !992
  br i1 %cmp9, label %if.then10, label %if.end21, !dbg !993

if.then10:                                        ; preds = %if.end
  call void @llvm.dbg.declare(metadata i32* %i, metadata !994, metadata !DIExpression()), !dbg !997
  store i32 1, i32* %i, align 4, !dbg !997
  br label %for.cond, !dbg !998

for.cond:                                         ; preds = %for.inc, %if.then10
  %15 = load i32, i32* %i, align 4, !dbg !999
  %16 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #4, !dbg !1001, !range !888
  %cmp12 = icmp ult i32 %15, %16, !dbg !1003
  br i1 %cmp12, label %for.body, label %for.end, !dbg !1004

for.body:                                         ; preds = %for.cond
  %17 = load double*, double** %share_data, align 8, !dbg !1005
  %18 = load i32, i32* %i, align 4, !dbg !1007
  %idxprom13 = sext i32 %18 to i64, !dbg !1005
  %arrayidx14 = getelementptr inbounds double, double* %17, i64 %idxprom13, !dbg !1005
  %19 = load double, double* %arrayidx14, align 8, !dbg !1005
  %20 = load double*, double** %share_data, align 8, !dbg !1008
  %arrayidx15 = getelementptr inbounds double, double* %20, i64 0, !dbg !1008
  %21 = load double, double* %arrayidx15, align 8, !dbg !1009
  %add16 = fadd contract double %21, %19, !dbg !1009
  store double %add16, double* %arrayidx15, align 8, !dbg !1009
  br label %for.inc, !dbg !1010

for.inc:                                          ; preds = %for.body
  %22 = load i32, i32* %i, align 4, !dbg !1011
  %inc = add nsw i32 %22, 1, !dbg !1011
  store i32 %inc, i32* %i, align 4, !dbg !1011
  br label %for.cond, !dbg !1012, !llvm.loop !1013

for.end:                                          ; preds = %for.cond
  %23 = load double*, double** %share_data, align 8, !dbg !1015
  %arrayidx17 = getelementptr inbounds double, double* %23, i64 0, !dbg !1015
  %24 = load double, double* %arrayidx17, align 8, !dbg !1015
  %25 = load double*, double** %global_data.addr, align 8, !dbg !1016
  %26 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #4, !dbg !1017, !range !843
  %idxprom19 = zext i32 %26 to i64, !dbg !1016
  %arrayidx20 = getelementptr inbounds double, double* %25, i64 %idxprom19, !dbg !1016
  store double %24, double* %arrayidx20, align 8, !dbg !1019
  br label %if.end21, !dbg !1020

if.end21:                                         ; preds = %for.end, %if.end
  ret void, !dbg !1021
}

; Function Attrs: convergent nounwind
declare void @llvm.nvvm.barrier0() #2

; Function Attrs: convergent noinline nounwind
define dso_local void @_Z23gpu_kernel_three_devicePiS_PdS0_S0_(i32* %colidx, i32* %rowstr, double* %a, double* %p, double* %q) #0 !dbg !1022 {
entry:
  %colidx.addr = alloca i32*, align 8
  %rowstr.addr = alloca i32*, align 8
  %a.addr = alloca double*, align 8
  %p.addr = alloca double*, align 8
  %q.addr = alloca double*, align 8
  %share_data = alloca double*, align 8
  %j = alloca i32, align 4
  %local_id = alloca i32, align 4
  %begin = alloca i32, align 4
  %end = alloca i32, align 4
  %sum = alloca double, align 8
  %k = alloca i32, align 4
  %i = alloca i32, align 4
  store i32* %colidx, i32** %colidx.addr, align 8
  call void @llvm.dbg.declare(metadata i32** %colidx.addr, metadata !1025, metadata !DIExpression()), !dbg !1026
  store i32* %rowstr, i32** %rowstr.addr, align 8
  call void @llvm.dbg.declare(metadata i32** %rowstr.addr, metadata !1027, metadata !DIExpression()), !dbg !1028
  store double* %a, double** %a.addr, align 8
  call void @llvm.dbg.declare(metadata double** %a.addr, metadata !1029, metadata !DIExpression()), !dbg !1030
  store double* %p, double** %p.addr, align 8
  call void @llvm.dbg.declare(metadata double** %p.addr, metadata !1031, metadata !DIExpression()), !dbg !1032
  store double* %q, double** %q.addr, align 8
  call void @llvm.dbg.declare(metadata double** %q.addr, metadata !1033, metadata !DIExpression()), !dbg !1034
  call void @llvm.dbg.declare(metadata double** %share_data, metadata !1035, metadata !DIExpression()), !dbg !1036
  store double* getelementptr inbounds ([0 x double], [0 x double]* addrspacecast ([0 x double] addrspace(3)* @extern_share_data to [0 x double]*), i64 0, i64 0), double** %share_data, align 8, !dbg !1036
  call void @llvm.dbg.declare(metadata i32* %j, metadata !1037, metadata !DIExpression()), !dbg !1038
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #4, !dbg !1039, !range !843
  %1 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #4, !dbg !1041, !range !888
  %mul = mul i32 %0, %1, !dbg !1043
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4, !dbg !1044, !range !918
  %add = add i32 %mul, %2, !dbg !1046
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #4, !dbg !1047, !range !888
  %div = udiv i32 %add, %3, !dbg !1049
  store i32 %div, i32* %j, align 4, !dbg !1038
  call void @llvm.dbg.declare(metadata i32* %local_id, metadata !1050, metadata !DIExpression()), !dbg !1051
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4, !dbg !1052, !range !918
  store i32 %4, i32* %local_id, align 4, !dbg !1051
  call void @llvm.dbg.declare(metadata i32* %begin, metadata !1054, metadata !DIExpression()), !dbg !1055
  %5 = load i32*, i32** %rowstr.addr, align 8, !dbg !1056
  %6 = load i32, i32* %j, align 4, !dbg !1057
  %idxprom = sext i32 %6 to i64, !dbg !1056
  %arrayidx = getelementptr inbounds i32, i32* %5, i64 %idxprom, !dbg !1056
  %7 = load i32, i32* %arrayidx, align 4, !dbg !1056
  store i32 %7, i32* %begin, align 4, !dbg !1055
  call void @llvm.dbg.declare(metadata i32* %end, metadata !1058, metadata !DIExpression()), !dbg !1059
  %8 = load i32*, i32** %rowstr.addr, align 8, !dbg !1060
  %9 = load i32, i32* %j, align 4, !dbg !1061
  %add5 = add nsw i32 %9, 1, !dbg !1062
  %idxprom6 = sext i32 %add5 to i64, !dbg !1060
  %arrayidx7 = getelementptr inbounds i32, i32* %8, i64 %idxprom6, !dbg !1060
  %10 = load i32, i32* %arrayidx7, align 4, !dbg !1060
  store i32 %10, i32* %end, align 4, !dbg !1059
  call void @llvm.dbg.declare(metadata double* %sum, metadata !1063, metadata !DIExpression()), !dbg !1064
  store double 0.000000e+00, double* %sum, align 8, !dbg !1064
  call void @llvm.dbg.declare(metadata i32* %k, metadata !1065, metadata !DIExpression()), !dbg !1067
  %11 = load i32, i32* %begin, align 4, !dbg !1068
  %12 = load i32, i32* %local_id, align 4, !dbg !1069
  %add8 = add nsw i32 %11, %12, !dbg !1070
  store i32 %add8, i32* %k, align 4, !dbg !1067
  br label %for.cond, !dbg !1071

for.cond:                                         ; preds = %for.inc, %entry
  %13 = load i32, i32* %k, align 4, !dbg !1072
  %14 = load i32, i32* %end, align 4, !dbg !1074
  %cmp = icmp slt i32 %13, %14, !dbg !1075
  br i1 %cmp, label %for.body, label %for.end, !dbg !1076

for.body:                                         ; preds = %for.cond
  %15 = load double, double* %sum, align 8, !dbg !1077
  %16 = load double*, double** %a.addr, align 8, !dbg !1079
  %17 = load i32, i32* %k, align 4, !dbg !1080
  %idxprom9 = sext i32 %17 to i64, !dbg !1079
  %arrayidx10 = getelementptr inbounds double, double* %16, i64 %idxprom9, !dbg !1079
  %18 = load double, double* %arrayidx10, align 8, !dbg !1079
  %19 = load double*, double** %p.addr, align 8, !dbg !1081
  %20 = load i32*, i32** %colidx.addr, align 8, !dbg !1082
  %21 = load i32, i32* %k, align 4, !dbg !1083
  %idxprom11 = sext i32 %21 to i64, !dbg !1082
  %arrayidx12 = getelementptr inbounds i32, i32* %20, i64 %idxprom11, !dbg !1082
  %22 = load i32, i32* %arrayidx12, align 4, !dbg !1082
  %idxprom13 = sext i32 %22 to i64, !dbg !1081
  %arrayidx14 = getelementptr inbounds double, double* %19, i64 %idxprom13, !dbg !1081
  %23 = load double, double* %arrayidx14, align 8, !dbg !1081
  %mul15 = fmul contract double %18, %23, !dbg !1084
  %add16 = fadd contract double %15, %mul15, !dbg !1085
  store double %add16, double* %sum, align 8, !dbg !1086
  br label %for.inc, !dbg !1087

for.inc:                                          ; preds = %for.body
  %24 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #4, !dbg !1088, !range !888
  %25 = load i32, i32* %k, align 4, !dbg !1090
  %add18 = add i32 %25, %24, !dbg !1090
  store i32 %add18, i32* %k, align 4, !dbg !1090
  br label %for.cond, !dbg !1091, !llvm.loop !1092

for.end:                                          ; preds = %for.cond
  %26 = load double, double* %sum, align 8, !dbg !1094
  %27 = load double*, double** %share_data, align 8, !dbg !1095
  %28 = load i32, i32* %local_id, align 4, !dbg !1096
  %idxprom19 = sext i32 %28 to i64, !dbg !1095
  %arrayidx20 = getelementptr inbounds double, double* %27, i64 %idxprom19, !dbg !1095
  store double %26, double* %arrayidx20, align 8, !dbg !1097
  call void @llvm.nvvm.barrier0(), !dbg !1098
  %29 = load i32, i32* %local_id, align 4, !dbg !1099
  %cmp21 = icmp eq i32 %29, 0, !dbg !1101
  br i1 %cmp21, label %if.then, label %if.end, !dbg !1102

if.then:                                          ; preds = %for.end
  call void @llvm.dbg.declare(metadata i32* %i, metadata !1103, metadata !DIExpression()), !dbg !1106
  store i32 1, i32* %i, align 4, !dbg !1106
  br label %for.cond22, !dbg !1107

for.cond22:                                       ; preds = %for.inc30, %if.then
  %30 = load i32, i32* %i, align 4, !dbg !1108
  %31 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #4, !dbg !1110, !range !888
  %cmp24 = icmp ult i32 %30, %31, !dbg !1112
  br i1 %cmp24, label %for.body25, label %for.end31, !dbg !1113

for.body25:                                       ; preds = %for.cond22
  %32 = load double*, double** %share_data, align 8, !dbg !1114
  %33 = load i32, i32* %i, align 4, !dbg !1116
  %idxprom26 = sext i32 %33 to i64, !dbg !1114
  %arrayidx27 = getelementptr inbounds double, double* %32, i64 %idxprom26, !dbg !1114
  %34 = load double, double* %arrayidx27, align 8, !dbg !1114
  %35 = load double*, double** %share_data, align 8, !dbg !1117
  %arrayidx28 = getelementptr inbounds double, double* %35, i64 0, !dbg !1117
  %36 = load double, double* %arrayidx28, align 8, !dbg !1118
  %add29 = fadd contract double %36, %34, !dbg !1118
  store double %add29, double* %arrayidx28, align 8, !dbg !1118
  br label %for.inc30, !dbg !1119

for.inc30:                                        ; preds = %for.body25
  %37 = load i32, i32* %i, align 4, !dbg !1120
  %inc = add nsw i32 %37, 1, !dbg !1120
  store i32 %inc, i32* %i, align 4, !dbg !1120
  br label %for.cond22, !dbg !1121, !llvm.loop !1122

for.end31:                                        ; preds = %for.cond22
  %38 = load double*, double** %share_data, align 8, !dbg !1124
  %arrayidx32 = getelementptr inbounds double, double* %38, i64 0, !dbg !1124
  %39 = load double, double* %arrayidx32, align 8, !dbg !1124
  %40 = load double*, double** %q.addr, align 8, !dbg !1125
  %41 = load i32, i32* %j, align 4, !dbg !1126
  %idxprom33 = sext i32 %41 to i64, !dbg !1125
  %arrayidx34 = getelementptr inbounds double, double* %40, i64 %idxprom33, !dbg !1125
  store double %39, double* %arrayidx34, align 8, !dbg !1127
  br label %if.end, !dbg !1128

if.end:                                           ; preds = %for.end31, %for.end
  ret void, !dbg !1129
}

; Function Attrs: convergent noinline nounwind
define dso_local void @_Z22gpu_kernel_four_devicePdS_S_S_(double* %d, double* %p, double* %q, double* %global_data) #0 !dbg !1130 {
entry:
  %d.addr = alloca double*, align 8
  %p.addr = alloca double*, align 8
  %q.addr = alloca double*, align 8
  %global_data.addr = alloca double*, align 8
  %share_data = alloca double*, align 8
  %thread_id = alloca i32, align 4
  %local_id = alloca i32, align 4
  %i = alloca i32, align 4
  store double* %d, double** %d.addr, align 8
  call void @llvm.dbg.declare(metadata double** %d.addr, metadata !1133, metadata !DIExpression()), !dbg !1134
  store double* %p, double** %p.addr, align 8
  call void @llvm.dbg.declare(metadata double** %p.addr, metadata !1135, metadata !DIExpression()), !dbg !1136
  store double* %q, double** %q.addr, align 8
  call void @llvm.dbg.declare(metadata double** %q.addr, metadata !1137, metadata !DIExpression()), !dbg !1138
  store double* %global_data, double** %global_data.addr, align 8
  call void @llvm.dbg.declare(metadata double** %global_data.addr, metadata !1139, metadata !DIExpression()), !dbg !1140
  call void @llvm.dbg.declare(metadata double** %share_data, metadata !1141, metadata !DIExpression()), !dbg !1142
  store double* getelementptr inbounds ([0 x double], [0 x double]* addrspacecast ([0 x double] addrspace(3)* @extern_share_data to [0 x double]*), i64 0, i64 0), double** %share_data, align 8, !dbg !1142
  call void @llvm.dbg.declare(metadata i32* %thread_id, metadata !1143, metadata !DIExpression()), !dbg !1144
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #4, !dbg !1145, !range !843
  %1 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #4, !dbg !1147, !range !888
  %mul = mul i32 %0, %1, !dbg !1149
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4, !dbg !1150, !range !918
  %add = add i32 %mul, %2, !dbg !1152
  store i32 %add, i32* %thread_id, align 4, !dbg !1144
  call void @llvm.dbg.declare(metadata i32* %local_id, metadata !1153, metadata !DIExpression()), !dbg !1154
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4, !dbg !1155, !range !918
  store i32 %3, i32* %local_id, align 4, !dbg !1154
  %4 = load double*, double** %share_data, align 8, !dbg !1157
  %5 = load i32, i32* %local_id, align 4, !dbg !1158
  %idxprom = sext i32 %5 to i64, !dbg !1157
  %arrayidx = getelementptr inbounds double, double* %4, i64 %idxprom, !dbg !1157
  store double 0.000000e+00, double* %arrayidx, align 8, !dbg !1159
  %6 = load double*, double** %share_data, align 8, !dbg !1160
  %7 = load i32, i32* %local_id, align 4, !dbg !1161
  %idxprom4 = sext i32 %7 to i64, !dbg !1160
  %arrayidx5 = getelementptr inbounds double, double* %6, i64 %idxprom4, !dbg !1160
  store double 0.000000e+00, double* %arrayidx5, align 8, !dbg !1162
  %8 = load i32, i32* %thread_id, align 4, !dbg !1163
  %cmp = icmp slt i32 %8, 14000, !dbg !1165
  br i1 %cmp, label %if.then, label %if.end, !dbg !1166

if.then:                                          ; preds = %entry
  %9 = load double*, double** %p.addr, align 8, !dbg !1167
  %10 = load i32, i32* %thread_id, align 4, !dbg !1169
  %idxprom6 = sext i32 %10 to i64, !dbg !1167
  %arrayidx7 = getelementptr inbounds double, double* %9, i64 %idxprom6, !dbg !1167
  %11 = load double, double* %arrayidx7, align 8, !dbg !1167
  %12 = load double*, double** %q.addr, align 8, !dbg !1170
  %13 = load i32, i32* %thread_id, align 4, !dbg !1171
  %idxprom8 = sext i32 %13 to i64, !dbg !1170
  %arrayidx9 = getelementptr inbounds double, double* %12, i64 %idxprom8, !dbg !1170
  %14 = load double, double* %arrayidx9, align 8, !dbg !1170
  %mul10 = fmul contract double %11, %14, !dbg !1172
  %15 = load double*, double** %share_data, align 8, !dbg !1173
  %16 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4, !dbg !1174, !range !918
  %idxprom12 = zext i32 %16 to i64, !dbg !1173
  %arrayidx13 = getelementptr inbounds double, double* %15, i64 %idxprom12, !dbg !1173
  store double %mul10, double* %arrayidx13, align 8, !dbg !1176
  br label %if.end, !dbg !1177

if.end:                                           ; preds = %if.then, %entry
  call void @llvm.nvvm.barrier0(), !dbg !1178
  %17 = load i32, i32* %local_id, align 4, !dbg !1179
  %cmp14 = icmp eq i32 %17, 0, !dbg !1181
  br i1 %cmp14, label %if.then15, label %if.end26, !dbg !1182

if.then15:                                        ; preds = %if.end
  call void @llvm.dbg.declare(metadata i32* %i, metadata !1183, metadata !DIExpression()), !dbg !1186
  store i32 1, i32* %i, align 4, !dbg !1186
  br label %for.cond, !dbg !1187

for.cond:                                         ; preds = %for.inc, %if.then15
  %18 = load i32, i32* %i, align 4, !dbg !1188
  %19 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #4, !dbg !1190, !range !888
  %cmp17 = icmp ult i32 %18, %19, !dbg !1192
  br i1 %cmp17, label %for.body, label %for.end, !dbg !1193

for.body:                                         ; preds = %for.cond
  %20 = load double*, double** %share_data, align 8, !dbg !1194
  %21 = load i32, i32* %i, align 4, !dbg !1196
  %idxprom18 = sext i32 %21 to i64, !dbg !1194
  %arrayidx19 = getelementptr inbounds double, double* %20, i64 %idxprom18, !dbg !1194
  %22 = load double, double* %arrayidx19, align 8, !dbg !1194
  %23 = load double*, double** %share_data, align 8, !dbg !1197
  %arrayidx20 = getelementptr inbounds double, double* %23, i64 0, !dbg !1197
  %24 = load double, double* %arrayidx20, align 8, !dbg !1198
  %add21 = fadd contract double %24, %22, !dbg !1198
  store double %add21, double* %arrayidx20, align 8, !dbg !1198
  br label %for.inc, !dbg !1199

for.inc:                                          ; preds = %for.body
  %25 = load i32, i32* %i, align 4, !dbg !1200
  %inc = add nsw i32 %25, 1, !dbg !1200
  store i32 %inc, i32* %i, align 4, !dbg !1200
  br label %for.cond, !dbg !1201, !llvm.loop !1202

for.end:                                          ; preds = %for.cond
  %26 = load double*, double** %share_data, align 8, !dbg !1204
  %arrayidx22 = getelementptr inbounds double, double* %26, i64 0, !dbg !1204
  %27 = load double, double* %arrayidx22, align 8, !dbg !1204
  %28 = load double*, double** %global_data.addr, align 8, !dbg !1205
  %29 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #4, !dbg !1206, !range !843
  %idxprom24 = zext i32 %29 to i64, !dbg !1205
  %arrayidx25 = getelementptr inbounds double, double* %28, i64 %idxprom24, !dbg !1205
  store double %27, double* %arrayidx25, align 8, !dbg !1208
  br label %if.end26, !dbg !1209

if.end26:                                         ; preds = %for.end, %if.end
  ret void, !dbg !1210
}

; Function Attrs: convergent noinline nounwind
define dso_local void @_Z17gpu_kernel_five_1dPdS_(double %alpha, double* %p, double* %z) #0 !dbg !1211 {
entry:
  %alpha.addr = alloca double, align 8
  %p.addr = alloca double*, align 8
  %z.addr = alloca double*, align 8
  %j = alloca i32, align 4
  store double %alpha, double* %alpha.addr, align 8
  call void @llvm.dbg.declare(metadata double* %alpha.addr, metadata !1214, metadata !DIExpression()), !dbg !1215
  store double* %p, double** %p.addr, align 8
  call void @llvm.dbg.declare(metadata double** %p.addr, metadata !1216, metadata !DIExpression()), !dbg !1217
  store double* %z, double** %z.addr, align 8
  call void @llvm.dbg.declare(metadata double** %z.addr, metadata !1218, metadata !DIExpression()), !dbg !1219
  call void @llvm.dbg.declare(metadata i32* %j, metadata !1220, metadata !DIExpression()), !dbg !1221
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #4, !dbg !1222, !range !843
  %1 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #4, !dbg !1224, !range !888
  %mul = mul i32 %0, %1, !dbg !1226
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4, !dbg !1227, !range !918
  %add = add i32 %mul, %2, !dbg !1229
  store i32 %add, i32* %j, align 4, !dbg !1221
  %3 = load i32, i32* %j, align 4, !dbg !1230
  %cmp = icmp sge i32 %3, 14000, !dbg !1232
  br i1 %cmp, label %if.then, label %if.end, !dbg !1233

if.then:                                          ; preds = %entry
  br label %return, !dbg !1234

if.end:                                           ; preds = %entry
  %4 = load double, double* %alpha.addr, align 8, !dbg !1236
  %5 = load double*, double** %p.addr, align 8, !dbg !1237
  %6 = load i32, i32* %j, align 4, !dbg !1238
  %idxprom = sext i32 %6 to i64, !dbg !1237
  %arrayidx = getelementptr inbounds double, double* %5, i64 %idxprom, !dbg !1237
  %7 = load double, double* %arrayidx, align 8, !dbg !1237
  %mul3 = fmul contract double %4, %7, !dbg !1239
  %8 = load double*, double** %z.addr, align 8, !dbg !1240
  %9 = load i32, i32* %j, align 4, !dbg !1241
  %idxprom4 = sext i32 %9 to i64, !dbg !1240
  %arrayidx5 = getelementptr inbounds double, double* %8, i64 %idxprom4, !dbg !1240
  %10 = load double, double* %arrayidx5, align 8, !dbg !1242
  %add6 = fadd contract double %10, %mul3, !dbg !1242
  store double %add6, double* %arrayidx5, align 8, !dbg !1242
  br label %return, !dbg !1243

return:                                           ; preds = %if.end, %if.then
  ret void, !dbg !1243
}

; Function Attrs: convergent noinline nounwind
define dso_local void @_Z17gpu_kernel_five_2dPdS_(double %alpha, double* %q, double* %r) #0 !dbg !1244 {
entry:
  %alpha.addr = alloca double, align 8
  %q.addr = alloca double*, align 8
  %r.addr = alloca double*, align 8
  %j = alloca i32, align 4
  store double %alpha, double* %alpha.addr, align 8
  call void @llvm.dbg.declare(metadata double* %alpha.addr, metadata !1245, metadata !DIExpression()), !dbg !1246
  store double* %q, double** %q.addr, align 8
  call void @llvm.dbg.declare(metadata double** %q.addr, metadata !1247, metadata !DIExpression()), !dbg !1248
  store double* %r, double** %r.addr, align 8
  call void @llvm.dbg.declare(metadata double** %r.addr, metadata !1249, metadata !DIExpression()), !dbg !1250
  call void @llvm.dbg.declare(metadata i32* %j, metadata !1251, metadata !DIExpression()), !dbg !1252
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #4, !dbg !1253, !range !843
  %1 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #4, !dbg !1255, !range !888
  %mul = mul i32 %0, %1, !dbg !1257
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4, !dbg !1258, !range !918
  %add = add i32 %mul, %2, !dbg !1260
  store i32 %add, i32* %j, align 4, !dbg !1252
  %3 = load i32, i32* %j, align 4, !dbg !1261
  %cmp = icmp sge i32 %3, 14000, !dbg !1263
  br i1 %cmp, label %if.then, label %if.end, !dbg !1264

if.then:                                          ; preds = %entry
  br label %return, !dbg !1265

if.end:                                           ; preds = %entry
  %4 = load double, double* %alpha.addr, align 8, !dbg !1267
  %5 = load double*, double** %q.addr, align 8, !dbg !1268
  %6 = load i32, i32* %j, align 4, !dbg !1269
  %idxprom = sext i32 %6 to i64, !dbg !1268
  %arrayidx = getelementptr inbounds double, double* %5, i64 %idxprom, !dbg !1268
  %7 = load double, double* %arrayidx, align 8, !dbg !1268
  %mul3 = fmul contract double %4, %7, !dbg !1270
  %8 = load double*, double** %r.addr, align 8, !dbg !1271
  %9 = load i32, i32* %j, align 4, !dbg !1272
  %idxprom4 = sext i32 %9 to i64, !dbg !1271
  %arrayidx5 = getelementptr inbounds double, double* %8, i64 %idxprom4, !dbg !1271
  %10 = load double, double* %arrayidx5, align 8, !dbg !1273
  %sub = fsub contract double %10, %mul3, !dbg !1273
  store double %sub, double* %arrayidx5, align 8, !dbg !1273
  br label %return, !dbg !1274

return:                                           ; preds = %if.end, %if.then
  ret void, !dbg !1274
}

; Function Attrs: convergent noinline nounwind
define dso_local void @_Z21gpu_kernel_six_devicePdS_(double* %r, double* %global_data) #0 !dbg !1275 {
entry:
  %r.addr = alloca double*, align 8
  %global_data.addr = alloca double*, align 8
  %share_data = alloca double*, align 8
  %thread_id = alloca i32, align 4
  %local_id = alloca i32, align 4
  %r_value = alloca double, align 8
  %i = alloca i32, align 4
  store double* %r, double** %r.addr, align 8
  call void @llvm.dbg.declare(metadata double** %r.addr, metadata !1278, metadata !DIExpression()), !dbg !1279
  store double* %global_data, double** %global_data.addr, align 8
  call void @llvm.dbg.declare(metadata double** %global_data.addr, metadata !1280, metadata !DIExpression()), !dbg !1281
  call void @llvm.dbg.declare(metadata double** %share_data, metadata !1282, metadata !DIExpression()), !dbg !1283
  store double* getelementptr inbounds ([0 x double], [0 x double]* addrspacecast ([0 x double] addrspace(3)* @extern_share_data to [0 x double]*), i64 0, i64 0), double** %share_data, align 8, !dbg !1283
  call void @llvm.dbg.declare(metadata i32* %thread_id, metadata !1284, metadata !DIExpression()), !dbg !1285
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #4, !dbg !1286, !range !843
  %1 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #4, !dbg !1288, !range !888
  %mul = mul i32 %0, %1, !dbg !1290
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4, !dbg !1291, !range !918
  %add = add i32 %mul, %2, !dbg !1293
  store i32 %add, i32* %thread_id, align 4, !dbg !1285
  call void @llvm.dbg.declare(metadata i32* %local_id, metadata !1294, metadata !DIExpression()), !dbg !1295
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4, !dbg !1296, !range !918
  store i32 %3, i32* %local_id, align 4, !dbg !1295
  %4 = load double*, double** %share_data, align 8, !dbg !1298
  %5 = load i32, i32* %local_id, align 4, !dbg !1299
  %idxprom = sext i32 %5 to i64, !dbg !1298
  %arrayidx = getelementptr inbounds double, double* %4, i64 %idxprom, !dbg !1298
  store double 0.000000e+00, double* %arrayidx, align 8, !dbg !1300
  %6 = load i32, i32* %thread_id, align 4, !dbg !1301
  %cmp = icmp slt i32 %6, 14000, !dbg !1303
  br i1 %cmp, label %if.then, label %if.end, !dbg !1304

if.then:                                          ; preds = %entry
  call void @llvm.dbg.declare(metadata double* %r_value, metadata !1305, metadata !DIExpression()), !dbg !1307
  %7 = load double*, double** %r.addr, align 8, !dbg !1308
  %8 = load i32, i32* %thread_id, align 4, !dbg !1309
  %idxprom4 = sext i32 %8 to i64, !dbg !1308
  %arrayidx5 = getelementptr inbounds double, double* %7, i64 %idxprom4, !dbg !1308
  %9 = load double, double* %arrayidx5, align 8, !dbg !1308
  store double %9, double* %r_value, align 8, !dbg !1307
  %10 = load double, double* %r_value, align 8, !dbg !1310
  %11 = load double, double* %r_value, align 8, !dbg !1311
  %mul6 = fmul contract double %10, %11, !dbg !1312
  %12 = load double*, double** %share_data, align 8, !dbg !1313
  %13 = load i32, i32* %local_id, align 4, !dbg !1314
  %idxprom7 = sext i32 %13 to i64, !dbg !1313
  %arrayidx8 = getelementptr inbounds double, double* %12, i64 %idxprom7, !dbg !1313
  store double %mul6, double* %arrayidx8, align 8, !dbg !1315
  br label %if.end, !dbg !1316

if.end:                                           ; preds = %if.then, %entry
  call void @llvm.nvvm.barrier0(), !dbg !1317
  %14 = load i32, i32* %local_id, align 4, !dbg !1318
  %cmp9 = icmp eq i32 %14, 0, !dbg !1320
  br i1 %cmp9, label %if.then10, label %if.end21, !dbg !1321

if.then10:                                        ; preds = %if.end
  call void @llvm.dbg.declare(metadata i32* %i, metadata !1322, metadata !DIExpression()), !dbg !1325
  store i32 1, i32* %i, align 4, !dbg !1325
  br label %for.cond, !dbg !1326

for.cond:                                         ; preds = %for.inc, %if.then10
  %15 = load i32, i32* %i, align 4, !dbg !1327
  %16 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #4, !dbg !1329, !range !888
  %cmp12 = icmp ult i32 %15, %16, !dbg !1331
  br i1 %cmp12, label %for.body, label %for.end, !dbg !1332

for.body:                                         ; preds = %for.cond
  %17 = load double*, double** %share_data, align 8, !dbg !1333
  %18 = load i32, i32* %i, align 4, !dbg !1335
  %idxprom13 = sext i32 %18 to i64, !dbg !1333
  %arrayidx14 = getelementptr inbounds double, double* %17, i64 %idxprom13, !dbg !1333
  %19 = load double, double* %arrayidx14, align 8, !dbg !1333
  %20 = load double*, double** %share_data, align 8, !dbg !1336
  %arrayidx15 = getelementptr inbounds double, double* %20, i64 0, !dbg !1336
  %21 = load double, double* %arrayidx15, align 8, !dbg !1337
  %add16 = fadd contract double %21, %19, !dbg !1337
  store double %add16, double* %arrayidx15, align 8, !dbg !1337
  br label %for.inc, !dbg !1338

for.inc:                                          ; preds = %for.body
  %22 = load i32, i32* %i, align 4, !dbg !1339
  %inc = add nsw i32 %22, 1, !dbg !1339
  store i32 %inc, i32* %i, align 4, !dbg !1339
  br label %for.cond, !dbg !1340, !llvm.loop !1341

for.end:                                          ; preds = %for.cond
  %23 = load double*, double** %share_data, align 8, !dbg !1343
  %arrayidx17 = getelementptr inbounds double, double* %23, i64 0, !dbg !1343
  %24 = load double, double* %arrayidx17, align 8, !dbg !1343
  %25 = load double*, double** %global_data.addr, align 8, !dbg !1344
  %26 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #4, !dbg !1345, !range !843
  %idxprom19 = zext i32 %26 to i64, !dbg !1344
  %arrayidx20 = getelementptr inbounds double, double* %25, i64 %idxprom19, !dbg !1344
  store double %24, double* %arrayidx20, align 8, !dbg !1347
  br label %if.end21, !dbg !1348

if.end21:                                         ; preds = %for.end, %if.end
  ret void, !dbg !1349
}

; Function Attrs: convergent noinline nounwind
define dso_local void @_Z23gpu_kernel_seven_devicedPdS_(double %beta, double* %p, double* %r) #0 !dbg !1350 {
entry:
  %beta.addr = alloca double, align 8
  %p.addr = alloca double*, align 8
  %r.addr = alloca double*, align 8
  %j = alloca i32, align 4
  store double %beta, double* %beta.addr, align 8
  call void @llvm.dbg.declare(metadata double* %beta.addr, metadata !1351, metadata !DIExpression()), !dbg !1352
  store double* %p, double** %p.addr, align 8
  call void @llvm.dbg.declare(metadata double** %p.addr, metadata !1353, metadata !DIExpression()), !dbg !1354
  store double* %r, double** %r.addr, align 8
  call void @llvm.dbg.declare(metadata double** %r.addr, metadata !1355, metadata !DIExpression()), !dbg !1356
  call void @llvm.dbg.declare(metadata i32* %j, metadata !1357, metadata !DIExpression()), !dbg !1358
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #4, !dbg !1359, !range !843
  %1 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #4, !dbg !1361, !range !888
  %mul = mul i32 %0, %1, !dbg !1363
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4, !dbg !1364, !range !918
  %add = add i32 %mul, %2, !dbg !1366
  store i32 %add, i32* %j, align 4, !dbg !1358
  %3 = load i32, i32* %j, align 4, !dbg !1367
  %cmp = icmp sge i32 %3, 14000, !dbg !1369
  br i1 %cmp, label %if.then, label %if.end, !dbg !1370

if.then:                                          ; preds = %entry
  br label %return, !dbg !1371

if.end:                                           ; preds = %entry
  %4 = load double*, double** %r.addr, align 8, !dbg !1373
  %5 = load i32, i32* %j, align 4, !dbg !1374
  %idxprom = sext i32 %5 to i64, !dbg !1373
  %arrayidx = getelementptr inbounds double, double* %4, i64 %idxprom, !dbg !1373
  %6 = load double, double* %arrayidx, align 8, !dbg !1373
  %7 = load double, double* %beta.addr, align 8, !dbg !1375
  %8 = load double*, double** %p.addr, align 8, !dbg !1376
  %9 = load i32, i32* %j, align 4, !dbg !1377
  %idxprom3 = sext i32 %9 to i64, !dbg !1376
  %arrayidx4 = getelementptr inbounds double, double* %8, i64 %idxprom3, !dbg !1376
  %10 = load double, double* %arrayidx4, align 8, !dbg !1376
  %mul5 = fmul contract double %7, %10, !dbg !1378
  %add6 = fadd contract double %6, %mul5, !dbg !1379
  %11 = load double*, double** %p.addr, align 8, !dbg !1380
  %12 = load i32, i32* %j, align 4, !dbg !1381
  %idxprom7 = sext i32 %12 to i64, !dbg !1380
  %arrayidx8 = getelementptr inbounds double, double* %11, i64 %idxprom7, !dbg !1380
  store double %add6, double* %arrayidx8, align 8, !dbg !1382
  br label %return, !dbg !1383

return:                                           ; preds = %if.end, %if.then
  ret void, !dbg !1383
}

; Function Attrs: convergent noinline nounwind
define dso_local void @_Z23gpu_kernel_eight_devicePiS_PdS0_S0_(i32* %colidx, i32* %rowstr, double* %a, double* %r, double* %z) #0 !dbg !1384 {
entry:
  %colidx.addr = alloca i32*, align 8
  %rowstr.addr = alloca i32*, align 8
  %a.addr = alloca double*, align 8
  %r.addr = alloca double*, align 8
  %z.addr = alloca double*, align 8
  %share_data = alloca double*, align 8
  %j = alloca i32, align 4
  %local_id = alloca i32, align 4
  %begin = alloca i32, align 4
  %end = alloca i32, align 4
  %sum = alloca double, align 8
  %k = alloca i32, align 4
  %i = alloca i32, align 4
  store i32* %colidx, i32** %colidx.addr, align 8
  call void @llvm.dbg.declare(metadata i32** %colidx.addr, metadata !1385, metadata !DIExpression()), !dbg !1386
  store i32* %rowstr, i32** %rowstr.addr, align 8
  call void @llvm.dbg.declare(metadata i32** %rowstr.addr, metadata !1387, metadata !DIExpression()), !dbg !1388
  store double* %a, double** %a.addr, align 8
  call void @llvm.dbg.declare(metadata double** %a.addr, metadata !1389, metadata !DIExpression()), !dbg !1390
  store double* %r, double** %r.addr, align 8
  call void @llvm.dbg.declare(metadata double** %r.addr, metadata !1391, metadata !DIExpression()), !dbg !1392
  store double* %z, double** %z.addr, align 8
  call void @llvm.dbg.declare(metadata double** %z.addr, metadata !1393, metadata !DIExpression()), !dbg !1394
  call void @llvm.dbg.declare(metadata double** %share_data, metadata !1395, metadata !DIExpression()), !dbg !1396
  store double* getelementptr inbounds ([0 x double], [0 x double]* addrspacecast ([0 x double] addrspace(3)* @extern_share_data to [0 x double]*), i64 0, i64 0), double** %share_data, align 8, !dbg !1396
  call void @llvm.dbg.declare(metadata i32* %j, metadata !1397, metadata !DIExpression()), !dbg !1398
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #4, !dbg !1399, !range !843
  %1 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #4, !dbg !1401, !range !888
  %mul = mul i32 %0, %1, !dbg !1403
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4, !dbg !1404, !range !918
  %add = add i32 %mul, %2, !dbg !1406
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #4, !dbg !1407, !range !888
  %div = udiv i32 %add, %3, !dbg !1409
  store i32 %div, i32* %j, align 4, !dbg !1398
  call void @llvm.dbg.declare(metadata i32* %local_id, metadata !1410, metadata !DIExpression()), !dbg !1411
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4, !dbg !1412, !range !918
  store i32 %4, i32* %local_id, align 4, !dbg !1411
  call void @llvm.dbg.declare(metadata i32* %begin, metadata !1414, metadata !DIExpression()), !dbg !1415
  %5 = load i32*, i32** %rowstr.addr, align 8, !dbg !1416
  %6 = load i32, i32* %j, align 4, !dbg !1417
  %idxprom = sext i32 %6 to i64, !dbg !1416
  %arrayidx = getelementptr inbounds i32, i32* %5, i64 %idxprom, !dbg !1416
  %7 = load i32, i32* %arrayidx, align 4, !dbg !1416
  store i32 %7, i32* %begin, align 4, !dbg !1415
  call void @llvm.dbg.declare(metadata i32* %end, metadata !1418, metadata !DIExpression()), !dbg !1419
  %8 = load i32*, i32** %rowstr.addr, align 8, !dbg !1420
  %9 = load i32, i32* %j, align 4, !dbg !1421
  %add5 = add nsw i32 %9, 1, !dbg !1422
  %idxprom6 = sext i32 %add5 to i64, !dbg !1420
  %arrayidx7 = getelementptr inbounds i32, i32* %8, i64 %idxprom6, !dbg !1420
  %10 = load i32, i32* %arrayidx7, align 4, !dbg !1420
  store i32 %10, i32* %end, align 4, !dbg !1419
  call void @llvm.dbg.declare(metadata double* %sum, metadata !1423, metadata !DIExpression()), !dbg !1424
  store double 0.000000e+00, double* %sum, align 8, !dbg !1424
  call void @llvm.dbg.declare(metadata i32* %k, metadata !1425, metadata !DIExpression()), !dbg !1427
  %11 = load i32, i32* %begin, align 4, !dbg !1428
  %12 = load i32, i32* %local_id, align 4, !dbg !1429
  %add8 = add nsw i32 %11, %12, !dbg !1430
  store i32 %add8, i32* %k, align 4, !dbg !1427
  br label %for.cond, !dbg !1431

for.cond:                                         ; preds = %for.inc, %entry
  %13 = load i32, i32* %k, align 4, !dbg !1432
  %14 = load i32, i32* %end, align 4, !dbg !1434
  %cmp = icmp slt i32 %13, %14, !dbg !1435
  br i1 %cmp, label %for.body, label %for.end, !dbg !1436

for.body:                                         ; preds = %for.cond
  %15 = load double, double* %sum, align 8, !dbg !1437
  %16 = load double*, double** %a.addr, align 8, !dbg !1439
  %17 = load i32, i32* %k, align 4, !dbg !1440
  %idxprom9 = sext i32 %17 to i64, !dbg !1439
  %arrayidx10 = getelementptr inbounds double, double* %16, i64 %idxprom9, !dbg !1439
  %18 = load double, double* %arrayidx10, align 8, !dbg !1439
  %19 = load double*, double** %z.addr, align 8, !dbg !1441
  %20 = load i32*, i32** %colidx.addr, align 8, !dbg !1442
  %21 = load i32, i32* %k, align 4, !dbg !1443
  %idxprom11 = sext i32 %21 to i64, !dbg !1442
  %arrayidx12 = getelementptr inbounds i32, i32* %20, i64 %idxprom11, !dbg !1442
  %22 = load i32, i32* %arrayidx12, align 4, !dbg !1442
  %idxprom13 = sext i32 %22 to i64, !dbg !1441
  %arrayidx14 = getelementptr inbounds double, double* %19, i64 %idxprom13, !dbg !1441
  %23 = load double, double* %arrayidx14, align 8, !dbg !1441
  %mul15 = fmul contract double %18, %23, !dbg !1444
  %add16 = fadd contract double %15, %mul15, !dbg !1445
  store double %add16, double* %sum, align 8, !dbg !1446
  br label %for.inc, !dbg !1447

for.inc:                                          ; preds = %for.body
  %24 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #4, !dbg !1448, !range !888
  %25 = load i32, i32* %k, align 4, !dbg !1450
  %add18 = add i32 %25, %24, !dbg !1450
  store i32 %add18, i32* %k, align 4, !dbg !1450
  br label %for.cond, !dbg !1451, !llvm.loop !1452

for.end:                                          ; preds = %for.cond
  %26 = load double, double* %sum, align 8, !dbg !1454
  %27 = load double*, double** %share_data, align 8, !dbg !1455
  %28 = load i32, i32* %local_id, align 4, !dbg !1456
  %idxprom19 = sext i32 %28 to i64, !dbg !1455
  %arrayidx20 = getelementptr inbounds double, double* %27, i64 %idxprom19, !dbg !1455
  store double %26, double* %arrayidx20, align 8, !dbg !1457
  call void @llvm.nvvm.barrier0(), !dbg !1458
  %29 = load i32, i32* %local_id, align 4, !dbg !1459
  %cmp21 = icmp eq i32 %29, 0, !dbg !1461
  br i1 %cmp21, label %if.then, label %if.end, !dbg !1462

if.then:                                          ; preds = %for.end
  call void @llvm.dbg.declare(metadata i32* %i, metadata !1463, metadata !DIExpression()), !dbg !1466
  store i32 1, i32* %i, align 4, !dbg !1466
  br label %for.cond22, !dbg !1467

for.cond22:                                       ; preds = %for.inc30, %if.then
  %30 = load i32, i32* %i, align 4, !dbg !1468
  %31 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #4, !dbg !1470, !range !888
  %cmp24 = icmp ult i32 %30, %31, !dbg !1472
  br i1 %cmp24, label %for.body25, label %for.end31, !dbg !1473

for.body25:                                       ; preds = %for.cond22
  %32 = load double*, double** %share_data, align 8, !dbg !1474
  %33 = load i32, i32* %i, align 4, !dbg !1476
  %idxprom26 = sext i32 %33 to i64, !dbg !1474
  %arrayidx27 = getelementptr inbounds double, double* %32, i64 %idxprom26, !dbg !1474
  %34 = load double, double* %arrayidx27, align 8, !dbg !1474
  %35 = load double*, double** %share_data, align 8, !dbg !1477
  %arrayidx28 = getelementptr inbounds double, double* %35, i64 0, !dbg !1477
  %36 = load double, double* %arrayidx28, align 8, !dbg !1478
  %add29 = fadd contract double %36, %34, !dbg !1478
  store double %add29, double* %arrayidx28, align 8, !dbg !1478
  br label %for.inc30, !dbg !1479

for.inc30:                                        ; preds = %for.body25
  %37 = load i32, i32* %i, align 4, !dbg !1480
  %inc = add nsw i32 %37, 1, !dbg !1480
  store i32 %inc, i32* %i, align 4, !dbg !1480
  br label %for.cond22, !dbg !1481, !llvm.loop !1482

for.end31:                                        ; preds = %for.cond22
  %38 = load double*, double** %share_data, align 8, !dbg !1484
  %arrayidx32 = getelementptr inbounds double, double* %38, i64 0, !dbg !1484
  %39 = load double, double* %arrayidx32, align 8, !dbg !1484
  %40 = load double*, double** %r.addr, align 8, !dbg !1485
  %41 = load i32, i32* %j, align 4, !dbg !1486
  %idxprom33 = sext i32 %41 to i64, !dbg !1485
  %arrayidx34 = getelementptr inbounds double, double* %40, i64 %idxprom33, !dbg !1485
  store double %39, double* %arrayidx34, align 8, !dbg !1487
  br label %if.end, !dbg !1488

if.end:                                           ; preds = %for.end31, %for.end
  ret void, !dbg !1489
}

; Function Attrs: convergent noinline nounwind
define dso_local void @_Z22gpu_kernel_nine_devicePdS_S_S_(double* %r, double* %x, double* %sum, double* %global_data) #0 !dbg !1490 {
entry:
  %r.addr = alloca double*, align 8
  %x.addr = alloca double*, align 8
  %sum.addr = alloca double*, align 8
  %global_data.addr = alloca double*, align 8
  %share_data = alloca double*, align 8
  %thread_id = alloca i32, align 4
  %local_id = alloca i32, align 4
  %i = alloca i32, align 4
  store double* %r, double** %r.addr, align 8
  call void @llvm.dbg.declare(metadata double** %r.addr, metadata !1491, metadata !DIExpression()), !dbg !1492
  store double* %x, double** %x.addr, align 8
  call void @llvm.dbg.declare(metadata double** %x.addr, metadata !1493, metadata !DIExpression()), !dbg !1494
  store double* %sum, double** %sum.addr, align 8
  call void @llvm.dbg.declare(metadata double** %sum.addr, metadata !1495, metadata !DIExpression()), !dbg !1496
  store double* %global_data, double** %global_data.addr, align 8
  call void @llvm.dbg.declare(metadata double** %global_data.addr, metadata !1497, metadata !DIExpression()), !dbg !1498
  call void @llvm.dbg.declare(metadata double** %share_data, metadata !1499, metadata !DIExpression()), !dbg !1500
  store double* getelementptr inbounds ([0 x double], [0 x double]* addrspacecast ([0 x double] addrspace(3)* @extern_share_data to [0 x double]*), i64 0, i64 0), double** %share_data, align 8, !dbg !1500
  call void @llvm.dbg.declare(metadata i32* %thread_id, metadata !1501, metadata !DIExpression()), !dbg !1502
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #4, !dbg !1503, !range !843
  %1 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #4, !dbg !1505, !range !888
  %mul = mul i32 %0, %1, !dbg !1507
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4, !dbg !1508, !range !918
  %add = add i32 %mul, %2, !dbg !1510
  store i32 %add, i32* %thread_id, align 4, !dbg !1502
  call void @llvm.dbg.declare(metadata i32* %local_id, metadata !1511, metadata !DIExpression()), !dbg !1512
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4, !dbg !1513, !range !918
  store i32 %3, i32* %local_id, align 4, !dbg !1512
  %4 = load double*, double** %share_data, align 8, !dbg !1515
  %5 = load i32, i32* %local_id, align 4, !dbg !1516
  %idxprom = sext i32 %5 to i64, !dbg !1515
  %arrayidx = getelementptr inbounds double, double* %4, i64 %idxprom, !dbg !1515
  store double 0.000000e+00, double* %arrayidx, align 8, !dbg !1517
  %6 = load i32, i32* %thread_id, align 4, !dbg !1518
  %cmp = icmp slt i32 %6, 14000, !dbg !1520
  br i1 %cmp, label %if.then, label %if.end, !dbg !1521

if.then:                                          ; preds = %entry
  %7 = load double*, double** %x.addr, align 8, !dbg !1522
  %8 = load i32, i32* %thread_id, align 4, !dbg !1524
  %idxprom4 = sext i32 %8 to i64, !dbg !1522
  %arrayidx5 = getelementptr inbounds double, double* %7, i64 %idxprom4, !dbg !1522
  %9 = load double, double* %arrayidx5, align 8, !dbg !1522
  %10 = load double*, double** %r.addr, align 8, !dbg !1525
  %11 = load i32, i32* %thread_id, align 4, !dbg !1526
  %idxprom6 = sext i32 %11 to i64, !dbg !1525
  %arrayidx7 = getelementptr inbounds double, double* %10, i64 %idxprom6, !dbg !1525
  %12 = load double, double* %arrayidx7, align 8, !dbg !1525
  %sub = fsub contract double %9, %12, !dbg !1527
  %13 = load double*, double** %share_data, align 8, !dbg !1528
  %14 = load i32, i32* %local_id, align 4, !dbg !1529
  %idxprom8 = sext i32 %14 to i64, !dbg !1528
  %arrayidx9 = getelementptr inbounds double, double* %13, i64 %idxprom8, !dbg !1528
  store double %sub, double* %arrayidx9, align 8, !dbg !1530
  %15 = load double*, double** %share_data, align 8, !dbg !1531
  %16 = load i32, i32* %local_id, align 4, !dbg !1532
  %idxprom10 = sext i32 %16 to i64, !dbg !1531
  %arrayidx11 = getelementptr inbounds double, double* %15, i64 %idxprom10, !dbg !1531
  %17 = load double, double* %arrayidx11, align 8, !dbg !1531
  %18 = load double*, double** %share_data, align 8, !dbg !1533
  %19 = load i32, i32* %local_id, align 4, !dbg !1534
  %idxprom12 = sext i32 %19 to i64, !dbg !1533
  %arrayidx13 = getelementptr inbounds double, double* %18, i64 %idxprom12, !dbg !1533
  %20 = load double, double* %arrayidx13, align 8, !dbg !1533
  %mul14 = fmul contract double %17, %20, !dbg !1535
  %21 = load double*, double** %share_data, align 8, !dbg !1536
  %22 = load i32, i32* %local_id, align 4, !dbg !1537
  %idxprom15 = sext i32 %22 to i64, !dbg !1536
  %arrayidx16 = getelementptr inbounds double, double* %21, i64 %idxprom15, !dbg !1536
  store double %mul14, double* %arrayidx16, align 8, !dbg !1538
  br label %if.end, !dbg !1539

if.end:                                           ; preds = %if.then, %entry
  call void @llvm.nvvm.barrier0(), !dbg !1540
  %23 = load i32, i32* %local_id, align 4, !dbg !1541
  %cmp17 = icmp eq i32 %23, 0, !dbg !1543
  br i1 %cmp17, label %if.then18, label %if.end29, !dbg !1544

if.then18:                                        ; preds = %if.end
  call void @llvm.dbg.declare(metadata i32* %i, metadata !1545, metadata !DIExpression()), !dbg !1548
  store i32 1, i32* %i, align 4, !dbg !1548
  br label %for.cond, !dbg !1549

for.cond:                                         ; preds = %for.inc, %if.then18
  %24 = load i32, i32* %i, align 4, !dbg !1550
  %25 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #4, !dbg !1552, !range !888
  %cmp20 = icmp ult i32 %24, %25, !dbg !1554
  br i1 %cmp20, label %for.body, label %for.end, !dbg !1555

for.body:                                         ; preds = %for.cond
  %26 = load double*, double** %share_data, align 8, !dbg !1556
  %27 = load i32, i32* %i, align 4, !dbg !1558
  %idxprom21 = sext i32 %27 to i64, !dbg !1556
  %arrayidx22 = getelementptr inbounds double, double* %26, i64 %idxprom21, !dbg !1556
  %28 = load double, double* %arrayidx22, align 8, !dbg !1556
  %29 = load double*, double** %share_data, align 8, !dbg !1559
  %arrayidx23 = getelementptr inbounds double, double* %29, i64 0, !dbg !1559
  %30 = load double, double* %arrayidx23, align 8, !dbg !1560
  %add24 = fadd contract double %30, %28, !dbg !1560
  store double %add24, double* %arrayidx23, align 8, !dbg !1560
  br label %for.inc, !dbg !1561

for.inc:                                          ; preds = %for.body
  %31 = load i32, i32* %i, align 4, !dbg !1562
  %inc = add nsw i32 %31, 1, !dbg !1562
  store i32 %inc, i32* %i, align 4, !dbg !1562
  br label %for.cond, !dbg !1563, !llvm.loop !1564

for.end:                                          ; preds = %for.cond
  %32 = load double*, double** %share_data, align 8, !dbg !1566
  %arrayidx25 = getelementptr inbounds double, double* %32, i64 0, !dbg !1566
  %33 = load double, double* %arrayidx25, align 8, !dbg !1566
  %34 = load double*, double** %global_data.addr, align 8, !dbg !1567
  %35 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #4, !dbg !1568, !range !843
  %idxprom27 = zext i32 %35 to i64, !dbg !1567
  %arrayidx28 = getelementptr inbounds double, double* %34, i64 %idxprom27, !dbg !1567
  store double %33, double* %arrayidx28, align 8, !dbg !1570
  br label %if.end29, !dbg !1571

if.end29:                                         ; preds = %for.end, %if.end
  ret void, !dbg !1572
}

; Function Attrs: convergent noinline nounwind
define dso_local void @_Z16gpu_kernel_ten_1PdS_S_(double* %norm_temp, double* %x, double* %z) #0 !dbg !1573 {
entry:
  %norm_temp.addr = alloca double*, align 8
  %x.addr = alloca double*, align 8
  %z.addr = alloca double*, align 8
  %share_data = alloca double*, align 8
  %thread_id = alloca i32, align 4
  %local_id = alloca i32, align 4
  %i = alloca i32, align 4
  store double* %norm_temp, double** %norm_temp.addr, align 8
  call void @llvm.dbg.declare(metadata double** %norm_temp.addr, metadata !1574, metadata !DIExpression()), !dbg !1575
  store double* %x, double** %x.addr, align 8
  call void @llvm.dbg.declare(metadata double** %x.addr, metadata !1576, metadata !DIExpression()), !dbg !1577
  store double* %z, double** %z.addr, align 8
  call void @llvm.dbg.declare(metadata double** %z.addr, metadata !1578, metadata !DIExpression()), !dbg !1579
  call void @llvm.dbg.declare(metadata double** %share_data, metadata !1580, metadata !DIExpression()), !dbg !1581
  store double* getelementptr inbounds ([0 x double], [0 x double]* addrspacecast ([0 x double] addrspace(3)* @extern_share_data to [0 x double]*), i64 0, i64 0), double** %share_data, align 8, !dbg !1581
  call void @llvm.dbg.declare(metadata i32* %thread_id, metadata !1582, metadata !DIExpression()), !dbg !1583
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #4, !dbg !1584, !range !843
  %1 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #4, !dbg !1586, !range !888
  %mul = mul i32 %0, %1, !dbg !1588
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4, !dbg !1589, !range !918
  %add = add i32 %mul, %2, !dbg !1591
  store i32 %add, i32* %thread_id, align 4, !dbg !1583
  call void @llvm.dbg.declare(metadata i32* %local_id, metadata !1592, metadata !DIExpression()), !dbg !1593
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4, !dbg !1594, !range !918
  store i32 %3, i32* %local_id, align 4, !dbg !1593
  %4 = load double*, double** %share_data, align 8, !dbg !1596
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4, !dbg !1597, !range !918
  %idxprom = zext i32 %5 to i64, !dbg !1596
  %arrayidx = getelementptr inbounds double, double* %4, i64 %idxprom, !dbg !1596
  store double 0.000000e+00, double* %arrayidx, align 8, !dbg !1599
  %6 = load i32, i32* %thread_id, align 4, !dbg !1600
  %cmp = icmp slt i32 %6, 14000, !dbg !1602
  br i1 %cmp, label %if.then, label %if.end, !dbg !1603

if.then:                                          ; preds = %entry
  %7 = load double*, double** %x.addr, align 8, !dbg !1604
  %8 = load i32, i32* %thread_id, align 4, !dbg !1606
  %idxprom5 = sext i32 %8 to i64, !dbg !1604
  %arrayidx6 = getelementptr inbounds double, double* %7, i64 %idxprom5, !dbg !1604
  %9 = load double, double* %arrayidx6, align 8, !dbg !1604
  %10 = load double*, double** %z.addr, align 8, !dbg !1607
  %11 = load i32, i32* %thread_id, align 4, !dbg !1608
  %idxprom7 = sext i32 %11 to i64, !dbg !1607
  %arrayidx8 = getelementptr inbounds double, double* %10, i64 %idxprom7, !dbg !1607
  %12 = load double, double* %arrayidx8, align 8, !dbg !1607
  %mul9 = fmul contract double %9, %12, !dbg !1609
  %13 = load double*, double** %share_data, align 8, !dbg !1610
  %14 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4, !dbg !1611, !range !918
  %idxprom11 = zext i32 %14 to i64, !dbg !1610
  %arrayidx12 = getelementptr inbounds double, double* %13, i64 %idxprom11, !dbg !1610
  store double %mul9, double* %arrayidx12, align 8, !dbg !1613
  br label %if.end, !dbg !1614

if.end:                                           ; preds = %if.then, %entry
  call void @llvm.nvvm.barrier0(), !dbg !1615
  %15 = load i32, i32* %local_id, align 4, !dbg !1616
  %cmp13 = icmp eq i32 %15, 0, !dbg !1618
  br i1 %cmp13, label %if.then14, label %if.end25, !dbg !1619

if.then14:                                        ; preds = %if.end
  call void @llvm.dbg.declare(metadata i32* %i, metadata !1620, metadata !DIExpression()), !dbg !1623
  store i32 1, i32* %i, align 4, !dbg !1623
  br label %for.cond, !dbg !1624

for.cond:                                         ; preds = %for.inc, %if.then14
  %16 = load i32, i32* %i, align 4, !dbg !1625
  %17 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #4, !dbg !1627, !range !888
  %cmp16 = icmp ult i32 %16, %17, !dbg !1629
  br i1 %cmp16, label %for.body, label %for.end, !dbg !1630

for.body:                                         ; preds = %for.cond
  %18 = load double*, double** %share_data, align 8, !dbg !1631
  %19 = load i32, i32* %i, align 4, !dbg !1633
  %idxprom17 = sext i32 %19 to i64, !dbg !1631
  %arrayidx18 = getelementptr inbounds double, double* %18, i64 %idxprom17, !dbg !1631
  %20 = load double, double* %arrayidx18, align 8, !dbg !1631
  %21 = load double*, double** %share_data, align 8, !dbg !1634
  %arrayidx19 = getelementptr inbounds double, double* %21, i64 0, !dbg !1634
  %22 = load double, double* %arrayidx19, align 8, !dbg !1635
  %add20 = fadd contract double %22, %20, !dbg !1635
  store double %add20, double* %arrayidx19, align 8, !dbg !1635
  br label %for.inc, !dbg !1636

for.inc:                                          ; preds = %for.body
  %23 = load i32, i32* %i, align 4, !dbg !1637
  %inc = add nsw i32 %23, 1, !dbg !1637
  store i32 %inc, i32* %i, align 4, !dbg !1637
  br label %for.cond, !dbg !1638, !llvm.loop !1639

for.end:                                          ; preds = %for.cond
  %24 = load double*, double** %share_data, align 8, !dbg !1641
  %arrayidx21 = getelementptr inbounds double, double* %24, i64 0, !dbg !1641
  %25 = load double, double* %arrayidx21, align 8, !dbg !1641
  %26 = load double*, double** %norm_temp.addr, align 8, !dbg !1642
  %27 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #4, !dbg !1643, !range !843
  %idxprom23 = zext i32 %27 to i64, !dbg !1642
  %arrayidx24 = getelementptr inbounds double, double* %26, i64 %idxprom23, !dbg !1642
  store double %25, double* %arrayidx24, align 8, !dbg !1645
  br label %if.end25, !dbg !1646

if.end25:                                         ; preds = %for.end, %if.end
  ret void, !dbg !1647
}

; Function Attrs: convergent noinline nounwind
define dso_local void @_Z16gpu_kernel_ten_2PdS_S_(double* %norm_temp, double* %x, double* %z) #0 !dbg !1648 {
entry:
  %norm_temp.addr = alloca double*, align 8
  %x.addr = alloca double*, align 8
  %z.addr = alloca double*, align 8
  %share_data = alloca double*, align 8
  %thread_id = alloca i32, align 4
  %local_id = alloca i32, align 4
  %i = alloca i32, align 4
  store double* %norm_temp, double** %norm_temp.addr, align 8
  call void @llvm.dbg.declare(metadata double** %norm_temp.addr, metadata !1649, metadata !DIExpression()), !dbg !1650
  store double* %x, double** %x.addr, align 8
  call void @llvm.dbg.declare(metadata double** %x.addr, metadata !1651, metadata !DIExpression()), !dbg !1652
  store double* %z, double** %z.addr, align 8
  call void @llvm.dbg.declare(metadata double** %z.addr, metadata !1653, metadata !DIExpression()), !dbg !1654
  call void @llvm.dbg.declare(metadata double** %share_data, metadata !1655, metadata !DIExpression()), !dbg !1656
  store double* getelementptr inbounds ([0 x double], [0 x double]* addrspacecast ([0 x double] addrspace(3)* @extern_share_data to [0 x double]*), i64 0, i64 0), double** %share_data, align 8, !dbg !1656
  call void @llvm.dbg.declare(metadata i32* %thread_id, metadata !1657, metadata !DIExpression()), !dbg !1658
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #4, !dbg !1659, !range !843
  %1 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #4, !dbg !1661, !range !888
  %mul = mul i32 %0, %1, !dbg !1663
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4, !dbg !1664, !range !918
  %add = add i32 %mul, %2, !dbg !1666
  store i32 %add, i32* %thread_id, align 4, !dbg !1658
  call void @llvm.dbg.declare(metadata i32* %local_id, metadata !1667, metadata !DIExpression()), !dbg !1668
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4, !dbg !1669, !range !918
  store i32 %3, i32* %local_id, align 4, !dbg !1668
  %4 = load double*, double** %share_data, align 8, !dbg !1671
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4, !dbg !1672, !range !918
  %idxprom = zext i32 %5 to i64, !dbg !1671
  %arrayidx = getelementptr inbounds double, double* %4, i64 %idxprom, !dbg !1671
  store double 0.000000e+00, double* %arrayidx, align 8, !dbg !1674
  %6 = load i32, i32* %thread_id, align 4, !dbg !1675
  %cmp = icmp slt i32 %6, 14000, !dbg !1677
  br i1 %cmp, label %if.then, label %if.end, !dbg !1678

if.then:                                          ; preds = %entry
  %7 = load double*, double** %z.addr, align 8, !dbg !1679
  %8 = load i32, i32* %thread_id, align 4, !dbg !1681
  %idxprom5 = sext i32 %8 to i64, !dbg !1679
  %arrayidx6 = getelementptr inbounds double, double* %7, i64 %idxprom5, !dbg !1679
  %9 = load double, double* %arrayidx6, align 8, !dbg !1679
  %10 = load double*, double** %z.addr, align 8, !dbg !1682
  %11 = load i32, i32* %thread_id, align 4, !dbg !1683
  %idxprom7 = sext i32 %11 to i64, !dbg !1682
  %arrayidx8 = getelementptr inbounds double, double* %10, i64 %idxprom7, !dbg !1682
  %12 = load double, double* %arrayidx8, align 8, !dbg !1682
  %mul9 = fmul contract double %9, %12, !dbg !1684
  %13 = load double*, double** %share_data, align 8, !dbg !1685
  %14 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4, !dbg !1686, !range !918
  %idxprom11 = zext i32 %14 to i64, !dbg !1685
  %arrayidx12 = getelementptr inbounds double, double* %13, i64 %idxprom11, !dbg !1685
  store double %mul9, double* %arrayidx12, align 8, !dbg !1688
  br label %if.end, !dbg !1689

if.end:                                           ; preds = %if.then, %entry
  call void @llvm.nvvm.barrier0(), !dbg !1690
  %15 = load i32, i32* %local_id, align 4, !dbg !1691
  %cmp13 = icmp eq i32 %15, 0, !dbg !1693
  br i1 %cmp13, label %if.then14, label %if.end25, !dbg !1694

if.then14:                                        ; preds = %if.end
  call void @llvm.dbg.declare(metadata i32* %i, metadata !1695, metadata !DIExpression()), !dbg !1698
  store i32 1, i32* %i, align 4, !dbg !1698
  br label %for.cond, !dbg !1699

for.cond:                                         ; preds = %for.inc, %if.then14
  %16 = load i32, i32* %i, align 4, !dbg !1700
  %17 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #4, !dbg !1702, !range !888
  %cmp16 = icmp ult i32 %16, %17, !dbg !1704
  br i1 %cmp16, label %for.body, label %for.end, !dbg !1705

for.body:                                         ; preds = %for.cond
  %18 = load double*, double** %share_data, align 8, !dbg !1706
  %19 = load i32, i32* %i, align 4, !dbg !1708
  %idxprom17 = sext i32 %19 to i64, !dbg !1706
  %arrayidx18 = getelementptr inbounds double, double* %18, i64 %idxprom17, !dbg !1706
  %20 = load double, double* %arrayidx18, align 8, !dbg !1706
  %21 = load double*, double** %share_data, align 8, !dbg !1709
  %arrayidx19 = getelementptr inbounds double, double* %21, i64 0, !dbg !1709
  %22 = load double, double* %arrayidx19, align 8, !dbg !1710
  %add20 = fadd contract double %22, %20, !dbg !1710
  store double %add20, double* %arrayidx19, align 8, !dbg !1710
  br label %for.inc, !dbg !1711

for.inc:                                          ; preds = %for.body
  %23 = load i32, i32* %i, align 4, !dbg !1712
  %inc = add nsw i32 %23, 1, !dbg !1712
  store i32 %inc, i32* %i, align 4, !dbg !1712
  br label %for.cond, !dbg !1713, !llvm.loop !1714

for.end:                                          ; preds = %for.cond
  %24 = load double*, double** %share_data, align 8, !dbg !1716
  %arrayidx21 = getelementptr inbounds double, double* %24, i64 0, !dbg !1716
  %25 = load double, double* %arrayidx21, align 8, !dbg !1716
  %26 = load double*, double** %norm_temp.addr, align 8, !dbg !1717
  %27 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #4, !dbg !1718, !range !843
  %idxprom23 = zext i32 %27 to i64, !dbg !1717
  %arrayidx24 = getelementptr inbounds double, double* %26, i64 %idxprom23, !dbg !1717
  store double %25, double* %arrayidx24, align 8, !dbg !1720
  br label %if.end25, !dbg !1721

if.end25:                                         ; preds = %for.end, %if.end
  ret void, !dbg !1722
}

; Function Attrs: convergent noinline nounwind
define dso_local void @_Z24gpu_kernel_eleven_devicedPdS_(double %norm_temp2, double* %x, double* %z) #0 !dbg !1723 {
entry:
  %norm_temp2.addr = alloca double, align 8
  %x.addr = alloca double*, align 8
  %z.addr = alloca double*, align 8
  %j = alloca i32, align 4
  store double %norm_temp2, double* %norm_temp2.addr, align 8
  call void @llvm.dbg.declare(metadata double* %norm_temp2.addr, metadata !1724, metadata !DIExpression()), !dbg !1725
  store double* %x, double** %x.addr, align 8
  call void @llvm.dbg.declare(metadata double** %x.addr, metadata !1726, metadata !DIExpression()), !dbg !1727
  store double* %z, double** %z.addr, align 8
  call void @llvm.dbg.declare(metadata double** %z.addr, metadata !1728, metadata !DIExpression()), !dbg !1729
  call void @llvm.dbg.declare(metadata i32* %j, metadata !1730, metadata !DIExpression()), !dbg !1731
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #4, !dbg !1732, !range !843
  %1 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #4, !dbg !1734, !range !888
  %mul = mul i32 %0, %1, !dbg !1736
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4, !dbg !1737, !range !918
  %add = add i32 %mul, %2, !dbg !1739
  store i32 %add, i32* %j, align 4, !dbg !1731
  %3 = load i32, i32* %j, align 4, !dbg !1740
  %cmp = icmp sge i32 %3, 14000, !dbg !1742
  br i1 %cmp, label %if.then, label %if.end, !dbg !1743

if.then:                                          ; preds = %entry
  br label %return, !dbg !1744

if.end:                                           ; preds = %entry
  %4 = load double, double* %norm_temp2.addr, align 8, !dbg !1746
  %5 = load double*, double** %z.addr, align 8, !dbg !1747
  %6 = load i32, i32* %j, align 4, !dbg !1748
  %idxprom = sext i32 %6 to i64, !dbg !1747
  %arrayidx = getelementptr inbounds double, double* %5, i64 %idxprom, !dbg !1747
  %7 = load double, double* %arrayidx, align 8, !dbg !1747
  %mul3 = fmul contract double %4, %7, !dbg !1749
  %8 = load double*, double** %x.addr, align 8, !dbg !1750
  %9 = load i32, i32* %j, align 4, !dbg !1751
  %idxprom4 = sext i32 %9 to i64, !dbg !1750
  %arrayidx5 = getelementptr inbounds double, double* %8, i64 %idxprom4, !dbg !1750
  store double %mul3, double* %arrayidx5, align 8, !dbg !1752
  br label %return, !dbg !1753

return:                                           ; preds = %if.end, %if.then
  ret void, !dbg !1753
}

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #3

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #3

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3

attributes #0 = { convergent noinline nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="sm_20" "target-features"="+ptx42,+sm_20" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { convergent nounwind }
attributes #3 = { nounwind readnone }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.dbg.cu = !{!5}
!nvvm.annotations = !{!772, !773, !774, !775, !776, !777, !778, !779, !780, !781, !782, !783, !784, !785, !786, !785, !787, !787, !787, !787, !788, !788, !787}
!llvm.ident = !{!789}
!nvvm.internalize.after.link = !{}
!nvvmir.version = !{!790}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 7, i32 0]}
!1 = !{i32 2, !"Dwarf Version", i32 2}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !6, producer: "clang version 9.0.0 (https://github.com/yebinchon/llvm-project/ a1efa594106d738d0b74c9e4e2b9b779eb8b7d25)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !7, retainedTypes: !8, imports: !12, nameTableKind: None)
!6 = !DIFile(filename: "cg.cu", directory: "/scratch/ah7226/NPB-GPU/CUDA/CG")
!7 = !{}
!8 = !{!9, !11}
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64)
!10 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !{!13, !19, !24, !26, !28, !30, !32, !36, !38, !40, !42, !44, !46, !48, !50, !52, !54, !56, !58, !60, !62, !64, !68, !70, !72, !74, !78, !83, !85, !87, !92, !96, !98, !100, !102, !104, !106, !108, !110, !112, !117, !121, !123, !128, !132, !134, !136, !138, !140, !142, !146, !148, !150, !155, !162, !166, !168, !170, !172, !174, !178, !180, !182, !186, !188, !190, !192, !194, !196, !198, !200, !202, !204, !208, !214, !216, !218, !222, !224, !226, !228, !230, !232, !234, !236, !240, !244, !246, !248, !252, !254, !256, !258, !260, !262, !264, !268, !274, !278, !283, !285, !289, !293, !307, !311, !315, !319, !323, !328, !330, !334, !338, !342, !350, !354, !358, !362, !366, !371, !377, !381, !385, !387, !395, !399, !406, !408, !410, !414, !418, !422, !427, !431, !436, !437, !438, !439, !441, !442, !443, !444, !445, !446, !447, !449, !450, !451, !452, !453, !457, !458, !459, !460, !461, !462, !463, !464, !465, !466, !467, !468, !469, !470, !471, !472, !473, !474, !475, !476, !477, !478, !479, !480, !481, !485, !487, !489, !491, !493, !495, !497, !499, !502, !504, !506, !508, !510, !512, !514, !516, !518, !520, !522, !524, !526, !528, !530, !532, !534, !536, !538, !540, !542, !544, !546, !548, !550, !552, !554, !556, !558, !560, !562, !564, !566, !568, !570, !572, !574, !576, !578, !580, !582, !584, !586, !588, !590, !592, !594, !600, !606, !611, !615, !617, !619, !621, !623, !630, !634, !638, !642, !646, !650, !655, !659, !661, !665, !671, !675, !680, !682, !684, !688, !692, !696, !698, !700, !702, !704, !708, !710, !712, !716, !720, !724, !728, !732, !734, !736, !743, !747, !751, !755, !757, !759, !763, !767, !768, !769, !770, !771}
!13 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !15, file: !16, line: 223)
!14 = !DINamespace(name: "std", scope: null)
!15 = !DISubprogram(name: "abs", linkageName: "_ZL3absi", scope: !16, file: !16, line: 53, type: !17, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!16 = !DIFile(filename: "/u/NAS_SCRATCH/ah7226/tulip/llvm-install-tulip/lib/clang/9.0.0/include/__clang_cuda_math_forward_declares.h", directory: "")
!17 = !DISubroutineType(types: !18)
!18 = !{!11, !11}
!19 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !20, file: !16, line: 224)
!20 = !DISubprogram(name: "acos", linkageName: "_ZL4acosf", scope: !16, file: !16, line: 55, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!21 = !DISubroutineType(types: !22)
!22 = !{!23, !23}
!23 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!24 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !25, file: !16, line: 225)
!25 = !DISubprogram(name: "acosh", linkageName: "_ZL5acoshf", scope: !16, file: !16, line: 57, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!26 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !27, file: !16, line: 226)
!27 = !DISubprogram(name: "asin", linkageName: "_ZL4asinf", scope: !16, file: !16, line: 59, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!28 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !29, file: !16, line: 227)
!29 = !DISubprogram(name: "asinh", linkageName: "_ZL5asinhf", scope: !16, file: !16, line: 61, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!30 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !31, file: !16, line: 228)
!31 = !DISubprogram(name: "atan", linkageName: "_ZL4atanf", scope: !16, file: !16, line: 65, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!32 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !33, file: !16, line: 229)
!33 = !DISubprogram(name: "atan2", linkageName: "_ZL5atan2ff", scope: !16, file: !16, line: 63, type: !34, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!34 = !DISubroutineType(types: !35)
!35 = !{!23, !23, !23}
!36 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !37, file: !16, line: 230)
!37 = !DISubprogram(name: "atanh", linkageName: "_ZL5atanhf", scope: !16, file: !16, line: 67, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!38 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !39, file: !16, line: 231)
!39 = !DISubprogram(name: "cbrt", linkageName: "_ZL4cbrtf", scope: !16, file: !16, line: 69, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!40 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !41, file: !16, line: 232)
!41 = !DISubprogram(name: "ceil", linkageName: "_ZL4ceilf", scope: !16, file: !16, line: 71, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!42 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !43, file: !16, line: 233)
!43 = !DISubprogram(name: "copysign", linkageName: "_ZL8copysignff", scope: !16, file: !16, line: 73, type: !34, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!44 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !45, file: !16, line: 234)
!45 = !DISubprogram(name: "cos", linkageName: "_ZL3cosf", scope: !16, file: !16, line: 75, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!46 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !47, file: !16, line: 235)
!47 = !DISubprogram(name: "cosh", linkageName: "_ZL4coshf", scope: !16, file: !16, line: 77, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!48 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !49, file: !16, line: 236)
!49 = !DISubprogram(name: "erf", linkageName: "_ZL3erff", scope: !16, file: !16, line: 81, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!50 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !51, file: !16, line: 237)
!51 = !DISubprogram(name: "erfc", linkageName: "_ZL4erfcf", scope: !16, file: !16, line: 79, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!52 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !53, file: !16, line: 238)
!53 = !DISubprogram(name: "exp", linkageName: "_ZL3expf", scope: !16, file: !16, line: 85, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!54 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !55, file: !16, line: 239)
!55 = !DISubprogram(name: "exp2", linkageName: "_ZL4exp2f", scope: !16, file: !16, line: 83, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!56 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !57, file: !16, line: 240)
!57 = !DISubprogram(name: "expm1", linkageName: "_ZL5expm1f", scope: !16, file: !16, line: 87, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!58 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !59, file: !16, line: 241)
!59 = !DISubprogram(name: "fabs", linkageName: "_ZL4fabsf", scope: !16, file: !16, line: 89, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!60 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !61, file: !16, line: 242)
!61 = !DISubprogram(name: "fdim", linkageName: "_ZL4fdimff", scope: !16, file: !16, line: 91, type: !34, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!62 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !63, file: !16, line: 243)
!63 = !DISubprogram(name: "floor", linkageName: "_ZL5floorf", scope: !16, file: !16, line: 93, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!64 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !65, file: !16, line: 244)
!65 = !DISubprogram(name: "fma", linkageName: "_ZL3fmafff", scope: !16, file: !16, line: 95, type: !66, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!66 = !DISubroutineType(types: !67)
!67 = !{!23, !23, !23, !23}
!68 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !69, file: !16, line: 245)
!69 = !DISubprogram(name: "fmax", linkageName: "_ZL4fmaxff", scope: !16, file: !16, line: 97, type: !34, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!70 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !71, file: !16, line: 246)
!71 = !DISubprogram(name: "fmin", linkageName: "_ZL4fminff", scope: !16, file: !16, line: 99, type: !34, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!72 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !73, file: !16, line: 247)
!73 = !DISubprogram(name: "fmod", linkageName: "_ZL4fmodff", scope: !16, file: !16, line: 101, type: !34, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!74 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !75, file: !16, line: 248)
!75 = !DISubprogram(name: "fpclassify", linkageName: "_ZL10fpclassifyf", scope: !16, file: !16, line: 103, type: !76, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!76 = !DISubroutineType(types: !77)
!77 = !{!11, !23}
!78 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !79, file: !16, line: 249)
!79 = !DISubprogram(name: "frexp", linkageName: "_ZL5frexpfPi", scope: !16, file: !16, line: 105, type: !80, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!80 = !DISubroutineType(types: !81)
!81 = !{!23, !23, !82}
!82 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11, size: 64)
!83 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !84, file: !16, line: 250)
!84 = !DISubprogram(name: "hypot", linkageName: "_ZL5hypotff", scope: !16, file: !16, line: 107, type: !34, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!85 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !86, file: !16, line: 251)
!86 = !DISubprogram(name: "ilogb", linkageName: "_ZL5ilogbf", scope: !16, file: !16, line: 109, type: !76, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!87 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !88, file: !16, line: 252)
!88 = !DISubprogram(name: "isfinite", linkageName: "_ZL8isfinitef", scope: !16, file: !16, line: 114, type: !89, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!89 = !DISubroutineType(types: !90)
!90 = !{!91, !23}
!91 = !DIBasicType(name: "bool", size: 8, encoding: DW_ATE_boolean)
!92 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !93, file: !16, line: 253)
!93 = !DISubprogram(name: "isgreater", linkageName: "_ZL9isgreaterff", scope: !16, file: !16, line: 118, type: !94, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!94 = !DISubroutineType(types: !95)
!95 = !{!91, !23, !23}
!96 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !97, file: !16, line: 254)
!97 = !DISubprogram(name: "isgreaterequal", linkageName: "_ZL14isgreaterequalff", scope: !16, file: !16, line: 117, type: !94, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!98 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !99, file: !16, line: 255)
!99 = !DISubprogram(name: "isinf", linkageName: "_ZL5isinff", scope: !16, file: !16, line: 123, type: !89, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!100 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !101, file: !16, line: 256)
!101 = !DISubprogram(name: "isless", linkageName: "_ZL6islessff", scope: !16, file: !16, line: 127, type: !94, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!102 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !103, file: !16, line: 257)
!103 = !DISubprogram(name: "islessequal", linkageName: "_ZL11islessequalff", scope: !16, file: !16, line: 126, type: !94, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!104 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !105, file: !16, line: 258)
!105 = !DISubprogram(name: "islessgreater", linkageName: "_ZL13islessgreaterff", scope: !16, file: !16, line: 129, type: !94, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!106 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !107, file: !16, line: 259)
!107 = !DISubprogram(name: "isnan", linkageName: "_ZL5isnanf", scope: !16, file: !16, line: 134, type: !89, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!108 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !109, file: !16, line: 260)
!109 = !DISubprogram(name: "isnormal", linkageName: "_ZL8isnormalf", scope: !16, file: !16, line: 136, type: !89, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!110 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !111, file: !16, line: 261)
!111 = !DISubprogram(name: "isunordered", linkageName: "_ZL11isunorderedff", scope: !16, file: !16, line: 138, type: !94, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!112 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !113, file: !16, line: 262)
!113 = !DISubprogram(name: "labs", linkageName: "_ZL4labsl", scope: !16, file: !16, line: 139, type: !114, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!114 = !DISubroutineType(types: !115)
!115 = !{!116, !116}
!116 = !DIBasicType(name: "long int", size: 64, encoding: DW_ATE_signed)
!117 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !118, file: !16, line: 263)
!118 = !DISubprogram(name: "ldexp", linkageName: "_ZL5ldexpfi", scope: !16, file: !16, line: 141, type: !119, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!119 = !DISubroutineType(types: !120)
!120 = !{!23, !23, !11}
!121 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !122, file: !16, line: 264)
!122 = !DISubprogram(name: "lgamma", linkageName: "_ZL6lgammaf", scope: !16, file: !16, line: 143, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!123 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !124, file: !16, line: 265)
!124 = !DISubprogram(name: "llabs", linkageName: "_ZL5llabsx", scope: !16, file: !16, line: 144, type: !125, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!125 = !DISubroutineType(types: !126)
!126 = !{!127, !127}
!127 = !DIBasicType(name: "long long int", size: 64, encoding: DW_ATE_signed)
!128 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !129, file: !16, line: 266)
!129 = !DISubprogram(name: "llrint", linkageName: "_ZL6llrintf", scope: !16, file: !16, line: 146, type: !130, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!130 = !DISubroutineType(types: !131)
!131 = !{!127, !23}
!132 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !133, file: !16, line: 267)
!133 = !DISubprogram(name: "log", linkageName: "_ZL3logf", scope: !16, file: !16, line: 159, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!134 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !135, file: !16, line: 268)
!135 = !DISubprogram(name: "log10", linkageName: "_ZL5log10f", scope: !16, file: !16, line: 148, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!136 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !137, file: !16, line: 269)
!137 = !DISubprogram(name: "log1p", linkageName: "_ZL5log1pf", scope: !16, file: !16, line: 150, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!138 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !139, file: !16, line: 270)
!139 = !DISubprogram(name: "log2", linkageName: "_ZL4log2f", scope: !16, file: !16, line: 152, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!140 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !141, file: !16, line: 271)
!141 = !DISubprogram(name: "logb", linkageName: "_ZL4logbf", scope: !16, file: !16, line: 154, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!142 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !143, file: !16, line: 272)
!143 = !DISubprogram(name: "lrint", linkageName: "_ZL5lrintf", scope: !16, file: !16, line: 161, type: !144, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!144 = !DISubroutineType(types: !145)
!145 = !{!116, !23}
!146 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !147, file: !16, line: 273)
!147 = !DISubprogram(name: "lround", linkageName: "_ZL6lroundf", scope: !16, file: !16, line: 163, type: !144, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!148 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !149, file: !16, line: 274)
!149 = !DISubprogram(name: "llround", linkageName: "_ZL7llroundf", scope: !16, file: !16, line: 164, type: !130, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!150 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !151, file: !16, line: 275)
!151 = !DISubprogram(name: "modf", linkageName: "_ZL4modffPf", scope: !16, file: !16, line: 166, type: !152, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!152 = !DISubroutineType(types: !153)
!153 = !{!23, !23, !154}
!154 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !23, size: 64)
!155 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !156, file: !16, line: 276)
!156 = !DISubprogram(name: "nan", linkageName: "_ZL3nanPKc", scope: !16, file: !16, line: 167, type: !157, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!157 = !DISubroutineType(types: !158)
!158 = !{!10, !159}
!159 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !160, size: 64)
!160 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !161)
!161 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!162 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !163, file: !16, line: 277)
!163 = !DISubprogram(name: "nanf", linkageName: "_ZL4nanfPKc", scope: !16, file: !16, line: 168, type: !164, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!164 = !DISubroutineType(types: !165)
!165 = !{!23, !159}
!166 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !167, file: !16, line: 278)
!167 = !DISubprogram(name: "nearbyint", linkageName: "_ZL9nearbyintf", scope: !16, file: !16, line: 170, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!168 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !169, file: !16, line: 279)
!169 = !DISubprogram(name: "nextafter", linkageName: "_ZL9nextafterff", scope: !16, file: !16, line: 172, type: !34, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!170 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !171, file: !16, line: 280)
!171 = !DISubprogram(name: "pow", linkageName: "_ZL3powfi", scope: !16, file: !16, line: 176, type: !119, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!172 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !173, file: !16, line: 281)
!173 = !DISubprogram(name: "remainder", linkageName: "_ZL9remainderff", scope: !16, file: !16, line: 178, type: !34, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!174 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !175, file: !16, line: 282)
!175 = !DISubprogram(name: "remquo", linkageName: "_ZL6remquoffPi", scope: !16, file: !16, line: 180, type: !176, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!176 = !DISubroutineType(types: !177)
!177 = !{!23, !23, !23, !82}
!178 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !179, file: !16, line: 283)
!179 = !DISubprogram(name: "rint", linkageName: "_ZL4rintf", scope: !16, file: !16, line: 182, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!180 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !181, file: !16, line: 284)
!181 = !DISubprogram(name: "round", linkageName: "_ZL5roundf", scope: !16, file: !16, line: 184, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!182 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !183, file: !16, line: 285)
!183 = !DISubprogram(name: "scalbln", linkageName: "_ZL7scalblnfl", scope: !16, file: !16, line: 186, type: !184, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!184 = !DISubroutineType(types: !185)
!185 = !{!23, !23, !116}
!186 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !187, file: !16, line: 286)
!187 = !DISubprogram(name: "scalbn", linkageName: "_ZL6scalbnfi", scope: !16, file: !16, line: 188, type: !119, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!188 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !189, file: !16, line: 287)
!189 = !DISubprogram(name: "signbit", linkageName: "_ZL7signbitf", scope: !16, file: !16, line: 190, type: !89, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!190 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !191, file: !16, line: 288)
!191 = !DISubprogram(name: "sin", linkageName: "_ZL3sinf", scope: !16, file: !16, line: 192, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!192 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !193, file: !16, line: 289)
!193 = !DISubprogram(name: "sinh", linkageName: "_ZL4sinhf", scope: !16, file: !16, line: 194, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!194 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !195, file: !16, line: 290)
!195 = !DISubprogram(name: "sqrt", linkageName: "_ZL4sqrtf", scope: !16, file: !16, line: 196, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!196 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !197, file: !16, line: 291)
!197 = !DISubprogram(name: "tan", linkageName: "_ZL3tanf", scope: !16, file: !16, line: 198, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!198 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !199, file: !16, line: 292)
!199 = !DISubprogram(name: "tanh", linkageName: "_ZL4tanhf", scope: !16, file: !16, line: 200, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!200 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !201, file: !16, line: 293)
!201 = !DISubprogram(name: "tgamma", linkageName: "_ZL6tgammaf", scope: !16, file: !16, line: 202, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!202 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !203, file: !16, line: 294)
!203 = !DISubprogram(name: "trunc", linkageName: "_ZL5truncf", scope: !16, file: !16, line: 204, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!204 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !205, file: !207, line: 52)
!205 = !DISubprogram(name: "abs", scope: !206, file: !206, line: 848, type: !17, flags: DIFlagPrototyped, spFlags: 0)
!206 = !DIFile(filename: "/usr/include/stdlib.h", directory: "")
!207 = !DIFile(filename: "/usr/lib/gcc/x86_64-linux-gnu/11/../../../../include/c++/11/bits/std_abs.h", directory: "")
!208 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !209, file: !213, line: 83)
!209 = !DISubprogram(name: "acos", scope: !210, file: !210, line: 53, type: !211, flags: DIFlagPrototyped, spFlags: 0)
!210 = !DIFile(filename: "/usr/include/x86_64-linux-gnu/bits/mathcalls.h", directory: "")
!211 = !DISubroutineType(types: !212)
!212 = !{!10, !10}
!213 = !DIFile(filename: "/usr/lib/gcc/x86_64-linux-gnu/11/../../../../include/c++/11/cmath", directory: "")
!214 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !215, file: !213, line: 102)
!215 = !DISubprogram(name: "asin", scope: !210, file: !210, line: 55, type: !211, flags: DIFlagPrototyped, spFlags: 0)
!216 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !217, file: !213, line: 121)
!217 = !DISubprogram(name: "atan", scope: !210, file: !210, line: 57, type: !211, flags: DIFlagPrototyped, spFlags: 0)
!218 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !219, file: !213, line: 140)
!219 = !DISubprogram(name: "atan2", scope: !210, file: !210, line: 59, type: !220, flags: DIFlagPrototyped, spFlags: 0)
!220 = !DISubroutineType(types: !221)
!221 = !{!10, !10, !10}
!222 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !223, file: !213, line: 161)
!223 = !DISubprogram(name: "ceil", scope: !210, file: !210, line: 159, type: !211, flags: DIFlagPrototyped, spFlags: 0)
!224 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !225, file: !213, line: 180)
!225 = !DISubprogram(name: "cos", scope: !210, file: !210, line: 62, type: !211, flags: DIFlagPrototyped, spFlags: 0)
!226 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !227, file: !213, line: 199)
!227 = !DISubprogram(name: "cosh", scope: !210, file: !210, line: 71, type: !211, flags: DIFlagPrototyped, spFlags: 0)
!228 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !229, file: !213, line: 218)
!229 = !DISubprogram(name: "exp", scope: !210, file: !210, line: 95, type: !211, flags: DIFlagPrototyped, spFlags: 0)
!230 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !231, file: !213, line: 237)
!231 = !DISubprogram(name: "fabs", scope: !210, file: !210, line: 162, type: !211, flags: DIFlagPrototyped, spFlags: 0)
!232 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !233, file: !213, line: 256)
!233 = !DISubprogram(name: "floor", scope: !210, file: !210, line: 165, type: !211, flags: DIFlagPrototyped, spFlags: 0)
!234 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !235, file: !213, line: 275)
!235 = !DISubprogram(name: "fmod", scope: !210, file: !210, line: 168, type: !220, flags: DIFlagPrototyped, spFlags: 0)
!236 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !237, file: !213, line: 296)
!237 = !DISubprogram(name: "frexp", scope: !210, file: !210, line: 98, type: !238, flags: DIFlagPrototyped, spFlags: 0)
!238 = !DISubroutineType(types: !239)
!239 = !{!10, !10, !82}
!240 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !241, file: !213, line: 315)
!241 = !DISubprogram(name: "ldexp", scope: !210, file: !210, line: 101, type: !242, flags: DIFlagPrototyped, spFlags: 0)
!242 = !DISubroutineType(types: !243)
!243 = !{!10, !10, !11}
!244 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !245, file: !213, line: 334)
!245 = !DISubprogram(name: "log", scope: !210, file: !210, line: 104, type: !211, flags: DIFlagPrototyped, spFlags: 0)
!246 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !247, file: !213, line: 353)
!247 = !DISubprogram(name: "log10", scope: !210, file: !210, line: 107, type: !211, flags: DIFlagPrototyped, spFlags: 0)
!248 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !249, file: !213, line: 372)
!249 = !DISubprogram(name: "modf", scope: !210, file: !210, line: 110, type: !250, flags: DIFlagPrototyped, spFlags: 0)
!250 = !DISubroutineType(types: !251)
!251 = !{!10, !10, !9}
!252 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !253, file: !213, line: 384)
!253 = !DISubprogram(name: "pow", scope: !210, file: !210, line: 140, type: !220, flags: DIFlagPrototyped, spFlags: 0)
!254 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !255, file: !213, line: 421)
!255 = !DISubprogram(name: "sin", scope: !210, file: !210, line: 64, type: !211, flags: DIFlagPrototyped, spFlags: 0)
!256 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !257, file: !213, line: 440)
!257 = !DISubprogram(name: "sinh", scope: !210, file: !210, line: 73, type: !211, flags: DIFlagPrototyped, spFlags: 0)
!258 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !259, file: !213, line: 459)
!259 = !DISubprogram(name: "sqrt", scope: !210, file: !210, line: 143, type: !211, flags: DIFlagPrototyped, spFlags: 0)
!260 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !261, file: !213, line: 478)
!261 = !DISubprogram(name: "tan", scope: !210, file: !210, line: 66, type: !211, flags: DIFlagPrototyped, spFlags: 0)
!262 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !263, file: !213, line: 497)
!263 = !DISubprogram(name: "tanh", scope: !210, file: !210, line: 75, type: !211, flags: DIFlagPrototyped, spFlags: 0)
!264 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !265, file: !267, line: 127)
!265 = !DIDerivedType(tag: DW_TAG_typedef, name: "div_t", file: !206, line: 63, baseType: !266)
!266 = !DICompositeType(tag: DW_TAG_structure_type, file: !206, line: 59, flags: DIFlagFwdDecl, identifier: "_ZTS5div_t")
!267 = !DIFile(filename: "/usr/lib/gcc/x86_64-linux-gnu/11/../../../../include/c++/11/cstdlib", directory: "")
!268 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !269, file: !267, line: 128)
!269 = !DIDerivedType(tag: DW_TAG_typedef, name: "ldiv_t", file: !206, line: 71, baseType: !270)
!270 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !206, line: 67, size: 128, flags: DIFlagTypePassByValue, elements: !271, identifier: "_ZTS6ldiv_t")
!271 = !{!272, !273}
!272 = !DIDerivedType(tag: DW_TAG_member, name: "quot", scope: !270, file: !206, line: 69, baseType: !116, size: 64)
!273 = !DIDerivedType(tag: DW_TAG_member, name: "rem", scope: !270, file: !206, line: 70, baseType: !116, size: 64, offset: 64)
!274 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !275, file: !267, line: 130)
!275 = !DISubprogram(name: "abort", scope: !206, file: !206, line: 598, type: !276, flags: DIFlagPrototyped | DIFlagNoReturn, spFlags: 0)
!276 = !DISubroutineType(types: !277)
!277 = !{null}
!278 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !279, file: !267, line: 134)
!279 = !DISubprogram(name: "atexit", scope: !206, file: !206, line: 602, type: !280, flags: DIFlagPrototyped, spFlags: 0)
!280 = !DISubroutineType(types: !281)
!281 = !{!11, !282}
!282 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !276, size: 64)
!283 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !284, file: !267, line: 140)
!284 = !DISubprogram(name: "atof", scope: !206, file: !206, line: 102, type: !157, flags: DIFlagPrototyped, spFlags: 0)
!285 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !286, file: !267, line: 141)
!286 = !DISubprogram(name: "atoi", scope: !206, file: !206, line: 105, type: !287, flags: DIFlagPrototyped, spFlags: 0)
!287 = !DISubroutineType(types: !288)
!288 = !{!11, !159}
!289 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !290, file: !267, line: 142)
!290 = !DISubprogram(name: "atol", scope: !206, file: !206, line: 108, type: !291, flags: DIFlagPrototyped, spFlags: 0)
!291 = !DISubroutineType(types: !292)
!292 = !{!116, !159}
!293 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !294, file: !267, line: 143)
!294 = !DISubprogram(name: "bsearch", scope: !206, file: !206, line: 828, type: !295, flags: DIFlagPrototyped, spFlags: 0)
!295 = !DISubroutineType(types: !296)
!296 = !{!297, !298, !298, !300, !300, !303}
!297 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!298 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !299, size: 64)
!299 = !DIDerivedType(tag: DW_TAG_const_type, baseType: null)
!300 = !DIDerivedType(tag: DW_TAG_typedef, name: "size_t", file: !301, line: 46, baseType: !302)
!301 = !DIFile(filename: "/u/NAS_SCRATCH/ah7226/tulip/llvm-install-tulip/lib/clang/9.0.0/include/stddef.h", directory: "")
!302 = !DIBasicType(name: "long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!303 = !DIDerivedType(tag: DW_TAG_typedef, name: "__compar_fn_t", file: !206, line: 816, baseType: !304)
!304 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !305, size: 64)
!305 = !DISubroutineType(types: !306)
!306 = !{!11, !298, !298}
!307 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !308, file: !267, line: 144)
!308 = !DISubprogram(name: "calloc", scope: !206, file: !206, line: 543, type: !309, flags: DIFlagPrototyped, spFlags: 0)
!309 = !DISubroutineType(types: !310)
!310 = !{!297, !300, !300}
!311 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !312, file: !267, line: 145)
!312 = !DISubprogram(name: "div", scope: !206, file: !206, line: 860, type: !313, flags: DIFlagPrototyped, spFlags: 0)
!313 = !DISubroutineType(types: !314)
!314 = !{!265, !11, !11}
!315 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !316, file: !267, line: 146)
!316 = !DISubprogram(name: "exit", scope: !206, file: !206, line: 624, type: !317, flags: DIFlagPrototyped | DIFlagNoReturn, spFlags: 0)
!317 = !DISubroutineType(types: !318)
!318 = !{null, !11}
!319 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !320, file: !267, line: 147)
!320 = !DISubprogram(name: "free", scope: !206, file: !206, line: 555, type: !321, flags: DIFlagPrototyped, spFlags: 0)
!321 = !DISubroutineType(types: !322)
!322 = !{null, !297}
!323 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !324, file: !267, line: 148)
!324 = !DISubprogram(name: "getenv", scope: !206, file: !206, line: 641, type: !325, flags: DIFlagPrototyped, spFlags: 0)
!325 = !DISubroutineType(types: !326)
!326 = !{!327, !159}
!327 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !161, size: 64)
!328 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !329, file: !267, line: 149)
!329 = !DISubprogram(name: "labs", scope: !206, file: !206, line: 849, type: !114, flags: DIFlagPrototyped, spFlags: 0)
!330 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !331, file: !267, line: 150)
!331 = !DISubprogram(name: "ldiv", scope: !206, file: !206, line: 862, type: !332, flags: DIFlagPrototyped, spFlags: 0)
!332 = !DISubroutineType(types: !333)
!333 = !{!269, !116, !116}
!334 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !335, file: !267, line: 151)
!335 = !DISubprogram(name: "malloc", scope: !206, file: !206, line: 540, type: !336, flags: DIFlagPrototyped, spFlags: 0)
!336 = !DISubroutineType(types: !337)
!337 = !{!297, !300}
!338 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !339, file: !267, line: 153)
!339 = !DISubprogram(name: "mblen", scope: !206, file: !206, line: 930, type: !340, flags: DIFlagPrototyped, spFlags: 0)
!340 = !DISubroutineType(types: !341)
!341 = !{!11, !159, !300}
!342 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !343, file: !267, line: 154)
!343 = !DISubprogram(name: "mbstowcs", scope: !206, file: !206, line: 941, type: !344, flags: DIFlagPrototyped, spFlags: 0)
!344 = !DISubroutineType(types: !345)
!345 = !{!300, !346, !349, !300}
!346 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !347)
!347 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !348, size: 64)
!348 = !DIBasicType(name: "wchar_t", size: 32, encoding: DW_ATE_signed)
!349 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !159)
!350 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !351, file: !267, line: 155)
!351 = !DISubprogram(name: "mbtowc", scope: !206, file: !206, line: 933, type: !352, flags: DIFlagPrototyped, spFlags: 0)
!352 = !DISubroutineType(types: !353)
!353 = !{!11, !346, !349, !300}
!354 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !355, file: !267, line: 157)
!355 = !DISubprogram(name: "qsort", scope: !206, file: !206, line: 838, type: !356, flags: DIFlagPrototyped, spFlags: 0)
!356 = !DISubroutineType(types: !357)
!357 = !{null, !297, !300, !300, !303}
!358 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !359, file: !267, line: 163)
!359 = !DISubprogram(name: "rand", scope: !206, file: !206, line: 454, type: !360, flags: DIFlagPrototyped, spFlags: 0)
!360 = !DISubroutineType(types: !361)
!361 = !{!11}
!362 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !363, file: !267, line: 164)
!363 = !DISubprogram(name: "realloc", scope: !206, file: !206, line: 551, type: !364, flags: DIFlagPrototyped, spFlags: 0)
!364 = !DISubroutineType(types: !365)
!365 = !{!297, !297, !300}
!366 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !367, file: !267, line: 165)
!367 = !DISubprogram(name: "srand", scope: !206, file: !206, line: 456, type: !368, flags: DIFlagPrototyped, spFlags: 0)
!368 = !DISubroutineType(types: !369)
!369 = !{null, !370}
!370 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!371 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !372, file: !267, line: 166)
!372 = !DISubprogram(name: "strtod", scope: !206, file: !206, line: 118, type: !373, flags: DIFlagPrototyped, spFlags: 0)
!373 = !DISubroutineType(types: !374)
!374 = !{!10, !349, !375}
!375 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !376)
!376 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !327, size: 64)
!377 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !378, file: !267, line: 167)
!378 = !DISubprogram(name: "strtol", scope: !206, file: !206, line: 177, type: !379, flags: DIFlagPrototyped, spFlags: 0)
!379 = !DISubroutineType(types: !380)
!380 = !{!116, !349, !375, !11}
!381 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !382, file: !267, line: 168)
!382 = !DISubprogram(name: "strtoul", scope: !206, file: !206, line: 181, type: !383, flags: DIFlagPrototyped, spFlags: 0)
!383 = !DISubroutineType(types: !384)
!384 = !{!302, !349, !375, !11}
!385 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !386, file: !267, line: 169)
!386 = !DISubprogram(name: "system", scope: !206, file: !206, line: 791, type: !287, flags: DIFlagPrototyped, spFlags: 0)
!387 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !388, file: !267, line: 171)
!388 = !DISubprogram(name: "wcstombs", scope: !206, file: !206, line: 945, type: !389, flags: DIFlagPrototyped, spFlags: 0)
!389 = !DISubroutineType(types: !390)
!390 = !{!300, !391, !392, !300}
!391 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !327)
!392 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !393)
!393 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !394, size: 64)
!394 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !348)
!395 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !396, file: !267, line: 172)
!396 = !DISubprogram(name: "wctomb", scope: !206, file: !206, line: 937, type: !397, flags: DIFlagPrototyped, spFlags: 0)
!397 = !DISubroutineType(types: !398)
!398 = !{!11, !327, !348}
!399 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !400, entity: !401, file: !267, line: 200)
!400 = !DINamespace(name: "__gnu_cxx", scope: null)
!401 = !DIDerivedType(tag: DW_TAG_typedef, name: "lldiv_t", file: !206, line: 81, baseType: !402)
!402 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !206, line: 77, size: 128, flags: DIFlagTypePassByValue, elements: !403, identifier: "_ZTS7lldiv_t")
!403 = !{!404, !405}
!404 = !DIDerivedType(tag: DW_TAG_member, name: "quot", scope: !402, file: !206, line: 79, baseType: !127, size: 64)
!405 = !DIDerivedType(tag: DW_TAG_member, name: "rem", scope: !402, file: !206, line: 80, baseType: !127, size: 64, offset: 64)
!406 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !400, entity: !407, file: !267, line: 206)
!407 = !DISubprogram(name: "_Exit", scope: !206, file: !206, line: 636, type: !317, flags: DIFlagPrototyped | DIFlagNoReturn, spFlags: 0)
!408 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !400, entity: !409, file: !267, line: 210)
!409 = !DISubprogram(name: "llabs", scope: !206, file: !206, line: 852, type: !125, flags: DIFlagPrototyped, spFlags: 0)
!410 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !400, entity: !411, file: !267, line: 216)
!411 = !DISubprogram(name: "lldiv", scope: !206, file: !206, line: 866, type: !412, flags: DIFlagPrototyped, spFlags: 0)
!412 = !DISubroutineType(types: !413)
!413 = !{!401, !127, !127}
!414 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !400, entity: !415, file: !267, line: 227)
!415 = !DISubprogram(name: "atoll", scope: !206, file: !206, line: 113, type: !416, flags: DIFlagPrototyped, spFlags: 0)
!416 = !DISubroutineType(types: !417)
!417 = !{!127, !159}
!418 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !400, entity: !419, file: !267, line: 228)
!419 = !DISubprogram(name: "strtoll", scope: !206, file: !206, line: 201, type: !420, flags: DIFlagPrototyped, spFlags: 0)
!420 = !DISubroutineType(types: !421)
!421 = !{!127, !349, !375, !11}
!422 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !400, entity: !423, file: !267, line: 229)
!423 = !DISubprogram(name: "strtoull", scope: !206, file: !206, line: 206, type: !424, flags: DIFlagPrototyped, spFlags: 0)
!424 = !DISubroutineType(types: !425)
!425 = !{!426, !349, !375, !11}
!426 = !DIBasicType(name: "long long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!427 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !400, entity: !428, file: !267, line: 231)
!428 = !DISubprogram(name: "strtof", scope: !206, file: !206, line: 124, type: !429, flags: DIFlagPrototyped, spFlags: 0)
!429 = !DISubroutineType(types: !430)
!430 = !{!23, !349, !375}
!431 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !400, entity: !432, file: !267, line: 232)
!432 = !DISubprogram(name: "strtold", scope: !206, file: !206, line: 127, type: !433, flags: DIFlagPrototyped, spFlags: 0)
!433 = !DISubroutineType(types: !434)
!434 = !{!435, !349, !375}
!435 = !DIBasicType(name: "long double", size: 64, encoding: DW_ATE_float)
!436 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !401, file: !267, line: 240)
!437 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !407, file: !267, line: 242)
!438 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !409, file: !267, line: 244)
!439 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !440, file: !267, line: 245)
!440 = !DISubprogram(name: "div", linkageName: "_ZN9__gnu_cxx3divExx", scope: !400, file: !267, line: 213, type: !412, flags: DIFlagPrototyped, spFlags: 0)
!441 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !411, file: !267, line: 246)
!442 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !415, file: !267, line: 248)
!443 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !428, file: !267, line: 249)
!444 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !419, file: !267, line: 250)
!445 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !423, file: !267, line: 251)
!446 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !432, file: !267, line: 252)
!447 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !275, file: !448, line: 38)
!448 = !DIFile(filename: "/usr/lib/gcc/x86_64-linux-gnu/11/../../../../include/c++/11/stdlib.h", directory: "")
!449 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !279, file: !448, line: 39)
!450 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !316, file: !448, line: 40)
!451 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !265, file: !448, line: 51)
!452 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !269, file: !448, line: 52)
!453 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !454, file: !448, line: 54)
!454 = !DISubprogram(name: "abs", linkageName: "_ZSt3abse", scope: !14, file: !207, line: 79, type: !455, flags: DIFlagPrototyped, spFlags: 0)
!455 = !DISubroutineType(types: !456)
!456 = !{!435, !435}
!457 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !284, file: !448, line: 55)
!458 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !286, file: !448, line: 56)
!459 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !290, file: !448, line: 57)
!460 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !294, file: !448, line: 58)
!461 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !308, file: !448, line: 59)
!462 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !440, file: !448, line: 60)
!463 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !320, file: !448, line: 61)
!464 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !324, file: !448, line: 62)
!465 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !329, file: !448, line: 63)
!466 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !331, file: !448, line: 64)
!467 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !335, file: !448, line: 65)
!468 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !339, file: !448, line: 67)
!469 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !343, file: !448, line: 68)
!470 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !351, file: !448, line: 69)
!471 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !355, file: !448, line: 71)
!472 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !359, file: !448, line: 72)
!473 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !363, file: !448, line: 73)
!474 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !367, file: !448, line: 74)
!475 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !372, file: !448, line: 75)
!476 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !378, file: !448, line: 76)
!477 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !382, file: !448, line: 77)
!478 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !386, file: !448, line: 78)
!479 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !388, file: !448, line: 80)
!480 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !396, file: !448, line: 81)
!481 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !482, file: !484, line: 414)
!482 = !DISubprogram(name: "acosf", linkageName: "_ZL5acosff", scope: !483, file: !483, line: 1126, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!483 = !DIFile(filename: "/usr/local/cuda/include/math_functions.hpp", directory: "")
!484 = !DIFile(filename: "/u/NAS_SCRATCH/ah7226/tulip/llvm-install-tulip/lib/clang/9.0.0/include/__clang_cuda_cmath.h", directory: "")
!485 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !486, file: !484, line: 415)
!486 = !DISubprogram(name: "acoshf", linkageName: "_ZL6acoshff", scope: !483, file: !483, line: 1154, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!487 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !488, file: !484, line: 416)
!488 = !DISubprogram(name: "asinf", linkageName: "_ZL5asinff", scope: !483, file: !483, line: 1121, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!489 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !490, file: !484, line: 417)
!490 = !DISubprogram(name: "asinhf", linkageName: "_ZL6asinhff", scope: !483, file: !483, line: 1159, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!491 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !492, file: !484, line: 418)
!492 = !DISubprogram(name: "atan2f", linkageName: "_ZL6atan2fff", scope: !483, file: !483, line: 1111, type: !34, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!493 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !494, file: !484, line: 419)
!494 = !DISubprogram(name: "atanf", linkageName: "_ZL5atanff", scope: !483, file: !483, line: 1116, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!495 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !496, file: !484, line: 420)
!496 = !DISubprogram(name: "atanhf", linkageName: "_ZL6atanhff", scope: !483, file: !483, line: 1164, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!497 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !498, file: !484, line: 421)
!498 = !DISubprogram(name: "cbrtf", linkageName: "_ZL5cbrtff", scope: !483, file: !483, line: 1199, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!499 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !500, file: !484, line: 422)
!500 = !DISubprogram(name: "ceilf", linkageName: "_ZL5ceilff", scope: !501, file: !501, line: 647, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!501 = !DIFile(filename: "/usr/local/cuda/include/device_functions.hpp", directory: "")
!502 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !503, file: !484, line: 423)
!503 = !DISubprogram(name: "copysignf", linkageName: "_ZL9copysignfff", scope: !483, file: !483, line: 973, type: !34, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!504 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !505, file: !484, line: 424)
!505 = !DISubprogram(name: "cosf", linkageName: "_ZL4cosff", scope: !483, file: !483, line: 1027, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!506 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !507, file: !484, line: 425)
!507 = !DISubprogram(name: "coshf", linkageName: "_ZL5coshff", scope: !483, file: !483, line: 1096, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!508 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !509, file: !484, line: 426)
!509 = !DISubprogram(name: "erfcf", linkageName: "_ZL5erfcff", scope: !483, file: !483, line: 1259, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!510 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !511, file: !484, line: 427)
!511 = !DISubprogram(name: "erff", linkageName: "_ZL4erfff", scope: !483, file: !483, line: 1249, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!512 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !513, file: !484, line: 428)
!513 = !DISubprogram(name: "exp2f", linkageName: "_ZL5exp2ff", scope: !501, file: !501, line: 637, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!514 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !515, file: !484, line: 429)
!515 = !DISubprogram(name: "expf", linkageName: "_ZL4expff", scope: !483, file: !483, line: 1078, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!516 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !517, file: !484, line: 430)
!517 = !DISubprogram(name: "expm1f", linkageName: "_ZL6expm1ff", scope: !483, file: !483, line: 1169, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!518 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !519, file: !484, line: 431)
!519 = !DISubprogram(name: "fabsf", linkageName: "_ZL5fabsff", scope: !501, file: !501, line: 582, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!520 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !521, file: !484, line: 432)
!521 = !DISubprogram(name: "fdimf", linkageName: "_ZL5fdimfff", scope: !483, file: !483, line: 1385, type: !34, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!522 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !523, file: !484, line: 433)
!523 = !DISubprogram(name: "floorf", linkageName: "_ZL6floorff", scope: !501, file: !501, line: 572, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!524 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !525, file: !484, line: 434)
!525 = !DISubprogram(name: "fmaf", linkageName: "_ZL4fmaffff", scope: !483, file: !483, line: 1337, type: !66, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!526 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !527, file: !484, line: 435)
!527 = !DISubprogram(name: "fmaxf", linkageName: "_ZL5fmaxfff", scope: !501, file: !501, line: 602, type: !34, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!528 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !529, file: !484, line: 436)
!529 = !DISubprogram(name: "fminf", linkageName: "_ZL5fminfff", scope: !501, file: !501, line: 597, type: !34, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!530 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !531, file: !484, line: 437)
!531 = !DISubprogram(name: "fmodf", linkageName: "_ZL5fmodfff", scope: !483, file: !483, line: 1322, type: !34, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!532 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !533, file: !484, line: 438)
!533 = !DISubprogram(name: "frexpf", linkageName: "_ZL6frexpffPi", scope: !483, file: !483, line: 1312, type: !80, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!534 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !535, file: !484, line: 439)
!535 = !DISubprogram(name: "hypotf", linkageName: "_ZL6hypotfff", scope: !483, file: !483, line: 1174, type: !34, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!536 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !537, file: !484, line: 440)
!537 = !DISubprogram(name: "ilogbf", linkageName: "_ZL6ilogbff", scope: !483, file: !483, line: 1390, type: !76, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!538 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !539, file: !484, line: 441)
!539 = !DISubprogram(name: "ldexpf", linkageName: "_ZL6ldexpffi", scope: !483, file: !483, line: 1289, type: !119, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!540 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !541, file: !484, line: 442)
!541 = !DISubprogram(name: "lgammaf", linkageName: "_ZL7lgammaff", scope: !483, file: !483, line: 1284, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!542 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !543, file: !484, line: 443)
!543 = !DISubprogram(name: "llrintf", linkageName: "_ZL7llrintff", scope: !483, file: !483, line: 933, type: !130, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!544 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !545, file: !484, line: 444)
!545 = !DISubprogram(name: "llroundf", linkageName: "_ZL8llroundff", scope: !483, file: !483, line: 1371, type: !130, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!546 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !547, file: !484, line: 445)
!547 = !DISubprogram(name: "log10f", linkageName: "_ZL6log10ff", scope: !483, file: !483, line: 1140, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!548 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !549, file: !484, line: 446)
!549 = !DISubprogram(name: "log1pf", linkageName: "_ZL6log1pff", scope: !483, file: !483, line: 1149, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!550 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !551, file: !484, line: 447)
!551 = !DISubprogram(name: "log2f", linkageName: "_ZL5log2ff", scope: !483, file: !483, line: 1069, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!552 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !553, file: !484, line: 448)
!553 = !DISubprogram(name: "logbf", linkageName: "_ZL5logbff", scope: !483, file: !483, line: 1395, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!554 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !555, file: !484, line: 449)
!555 = !DISubprogram(name: "logf", linkageName: "_ZL4logff", scope: !483, file: !483, line: 1131, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!556 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !557, file: !484, line: 450)
!557 = !DISubprogram(name: "lrintf", linkageName: "_ZL6lrintff", scope: !483, file: !483, line: 924, type: !144, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!558 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !559, file: !484, line: 451)
!559 = !DISubprogram(name: "lroundf", linkageName: "_ZL7lroundff", scope: !483, file: !483, line: 1376, type: !144, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!560 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !561, file: !484, line: 452)
!561 = !DISubprogram(name: "modff", linkageName: "_ZL5modfffPf", scope: !483, file: !483, line: 1317, type: !152, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!562 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !563, file: !484, line: 453)
!563 = !DISubprogram(name: "nearbyintf", linkageName: "_ZL10nearbyintff", scope: !483, file: !483, line: 938, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!564 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !565, file: !484, line: 454)
!565 = !DISubprogram(name: "nextafterf", linkageName: "_ZL10nextafterfff", scope: !483, file: !483, line: 1002, type: !34, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!566 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !567, file: !484, line: 455)
!567 = !DISubprogram(name: "powf", linkageName: "_ZL4powfff", scope: !483, file: !483, line: 1352, type: !34, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!568 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !569, file: !484, line: 456)
!569 = !DISubprogram(name: "remainderf", linkageName: "_ZL10remainderfff", scope: !483, file: !483, line: 1327, type: !34, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!570 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !571, file: !484, line: 457)
!571 = !DISubprogram(name: "remquof", linkageName: "_ZL7remquofffPi", scope: !483, file: !483, line: 1332, type: !176, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!572 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !573, file: !484, line: 458)
!573 = !DISubprogram(name: "rintf", linkageName: "_ZL5rintff", scope: !483, file: !483, line: 919, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!574 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !575, file: !484, line: 459)
!575 = !DISubprogram(name: "roundf", linkageName: "_ZL6roundff", scope: !483, file: !483, line: 1366, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!576 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !577, file: !484, line: 462)
!577 = !DISubprogram(name: "scalblnf", linkageName: "_ZL8scalblnffl", scope: !483, file: !483, line: 1299, type: !184, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!578 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !579, file: !484, line: 464)
!579 = !DISubprogram(name: "scalbnf", linkageName: "_ZL7scalbnffi", scope: !483, file: !483, line: 1294, type: !119, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!580 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !581, file: !484, line: 465)
!581 = !DISubprogram(name: "sinf", linkageName: "_ZL4sinff", scope: !483, file: !483, line: 1018, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!582 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !583, file: !484, line: 466)
!583 = !DISubprogram(name: "sinhf", linkageName: "_ZL5sinhff", scope: !483, file: !483, line: 1101, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!584 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !585, file: !484, line: 467)
!585 = !DISubprogram(name: "sqrtf", linkageName: "_ZL5sqrtff", scope: !501, file: !501, line: 887, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!586 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !587, file: !484, line: 468)
!587 = !DISubprogram(name: "tanf", linkageName: "_ZL4tanff", scope: !483, file: !483, line: 1060, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!588 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !589, file: !484, line: 469)
!589 = !DISubprogram(name: "tanhf", linkageName: "_ZL5tanhff", scope: !483, file: !483, line: 1106, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!590 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !591, file: !484, line: 470)
!591 = !DISubprogram(name: "tgammaf", linkageName: "_ZL7tgammaff", scope: !483, file: !483, line: 1361, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!592 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !593, file: !484, line: 471)
!593 = !DISubprogram(name: "truncf", linkageName: "_ZL6truncff", scope: !501, file: !501, line: 642, type: !21, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!594 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !595, file: !599, line: 98)
!595 = !DIDerivedType(tag: DW_TAG_typedef, name: "FILE", file: !596, line: 7, baseType: !597)
!596 = !DIFile(filename: "/usr/include/x86_64-linux-gnu/bits/types/FILE.h", directory: "")
!597 = !DICompositeType(tag: DW_TAG_structure_type, name: "_IO_FILE", file: !598, line: 49, flags: DIFlagFwdDecl, identifier: "_ZTS8_IO_FILE")
!598 = !DIFile(filename: "/usr/include/x86_64-linux-gnu/bits/types/struct_FILE.h", directory: "")
!599 = !DIFile(filename: "/usr/lib/gcc/x86_64-linux-gnu/11/../../../../include/c++/11/cstdio", directory: "")
!600 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !601, file: !599, line: 99)
!601 = !DIDerivedType(tag: DW_TAG_typedef, name: "fpos_t", file: !602, line: 84, baseType: !603)
!602 = !DIFile(filename: "/usr/include/stdio.h", directory: "")
!603 = !DIDerivedType(tag: DW_TAG_typedef, name: "__fpos_t", file: !604, line: 14, baseType: !605)
!604 = !DIFile(filename: "/usr/include/x86_64-linux-gnu/bits/types/__fpos_t.h", directory: "")
!605 = !DICompositeType(tag: DW_TAG_structure_type, name: "_G_fpos_t", file: !604, line: 10, flags: DIFlagFwdDecl, identifier: "_ZTS9_G_fpos_t")
!606 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !607, file: !599, line: 101)
!607 = !DISubprogram(name: "clearerr", scope: !602, file: !602, line: 786, type: !608, flags: DIFlagPrototyped, spFlags: 0)
!608 = !DISubroutineType(types: !609)
!609 = !{null, !610}
!610 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !595, size: 64)
!611 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !612, file: !599, line: 102)
!612 = !DISubprogram(name: "fclose", scope: !602, file: !602, line: 178, type: !613, flags: DIFlagPrototyped, spFlags: 0)
!613 = !DISubroutineType(types: !614)
!614 = !{!11, !610}
!615 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !616, file: !599, line: 103)
!616 = !DISubprogram(name: "feof", scope: !602, file: !602, line: 788, type: !613, flags: DIFlagPrototyped, spFlags: 0)
!617 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !618, file: !599, line: 104)
!618 = !DISubprogram(name: "ferror", scope: !602, file: !602, line: 790, type: !613, flags: DIFlagPrototyped, spFlags: 0)
!619 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !620, file: !599, line: 105)
!620 = !DISubprogram(name: "fflush", scope: !602, file: !602, line: 230, type: !613, flags: DIFlagPrototyped, spFlags: 0)
!621 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !622, file: !599, line: 106)
!622 = !DISubprogram(name: "fgetc", scope: !602, file: !602, line: 513, type: !613, flags: DIFlagPrototyped, spFlags: 0)
!623 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !624, file: !599, line: 107)
!624 = !DISubprogram(name: "fgetpos", scope: !602, file: !602, line: 760, type: !625, flags: DIFlagPrototyped, spFlags: 0)
!625 = !DISubroutineType(types: !626)
!626 = !{!11, !627, !628}
!627 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !610)
!628 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !629)
!629 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !601, size: 64)
!630 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !631, file: !599, line: 108)
!631 = !DISubprogram(name: "fgets", scope: !602, file: !602, line: 592, type: !632, flags: DIFlagPrototyped, spFlags: 0)
!632 = !DISubroutineType(types: !633)
!633 = !{!327, !391, !11, !627}
!634 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !635, file: !599, line: 109)
!635 = !DISubprogram(name: "fopen", scope: !602, file: !602, line: 258, type: !636, flags: DIFlagPrototyped, spFlags: 0)
!636 = !DISubroutineType(types: !637)
!637 = !{!610, !349, !349}
!638 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !639, file: !599, line: 110)
!639 = !DISubprogram(name: "fprintf", scope: !602, file: !602, line: 350, type: !640, flags: DIFlagPrototyped, spFlags: 0)
!640 = !DISubroutineType(types: !641)
!641 = !{!11, !627, !349, null}
!642 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !643, file: !599, line: 111)
!643 = !DISubprogram(name: "fputc", scope: !602, file: !602, line: 549, type: !644, flags: DIFlagPrototyped, spFlags: 0)
!644 = !DISubroutineType(types: !645)
!645 = !{!11, !11, !610}
!646 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !647, file: !599, line: 112)
!647 = !DISubprogram(name: "fputs", scope: !602, file: !602, line: 655, type: !648, flags: DIFlagPrototyped, spFlags: 0)
!648 = !DISubroutineType(types: !649)
!649 = !{!11, !349, !627}
!650 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !651, file: !599, line: 113)
!651 = !DISubprogram(name: "fread", scope: !602, file: !602, line: 675, type: !652, flags: DIFlagPrototyped, spFlags: 0)
!652 = !DISubroutineType(types: !653)
!653 = !{!300, !654, !300, !300, !627}
!654 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !297)
!655 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !656, file: !599, line: 114)
!656 = !DISubprogram(name: "freopen", scope: !602, file: !602, line: 265, type: !657, flags: DIFlagPrototyped, spFlags: 0)
!657 = !DISubroutineType(types: !658)
!658 = !{!610, !349, !349, !627}
!659 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !660, file: !599, line: 115)
!660 = !DISubprogram(name: "fscanf", scope: !602, file: !602, line: 415, type: !640, flags: DIFlagPrototyped, spFlags: 0)
!661 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !662, file: !599, line: 116)
!662 = !DISubprogram(name: "fseek", scope: !602, file: !602, line: 713, type: !663, flags: DIFlagPrototyped, spFlags: 0)
!663 = !DISubroutineType(types: !664)
!664 = !{!11, !610, !116, !11}
!665 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !666, file: !599, line: 117)
!666 = !DISubprogram(name: "fsetpos", scope: !602, file: !602, line: 765, type: !667, flags: DIFlagPrototyped, spFlags: 0)
!667 = !DISubroutineType(types: !668)
!668 = !{!11, !610, !669}
!669 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !670, size: 64)
!670 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !601)
!671 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !672, file: !599, line: 118)
!672 = !DISubprogram(name: "ftell", scope: !602, file: !602, line: 718, type: !673, flags: DIFlagPrototyped, spFlags: 0)
!673 = !DISubroutineType(types: !674)
!674 = !{!116, !610}
!675 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !676, file: !599, line: 119)
!676 = !DISubprogram(name: "fwrite", scope: !602, file: !602, line: 681, type: !677, flags: DIFlagPrototyped, spFlags: 0)
!677 = !DISubroutineType(types: !678)
!678 = !{!300, !679, !300, !300, !627}
!679 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !298)
!680 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !681, file: !599, line: 120)
!681 = !DISubprogram(name: "getc", scope: !602, file: !602, line: 514, type: !613, flags: DIFlagPrototyped, spFlags: 0)
!682 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !683, file: !599, line: 121)
!683 = !DISubprogram(name: "getchar", scope: !602, file: !602, line: 520, type: !360, flags: DIFlagPrototyped, spFlags: 0)
!684 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !685, file: !599, line: 124)
!685 = !DISubprogram(name: "gets", scope: !602, file: !602, line: 605, type: !686, flags: DIFlagPrototyped, spFlags: 0)
!686 = !DISubroutineType(types: !687)
!687 = !{!327, !327}
!688 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !689, file: !599, line: 126)
!689 = !DISubprogram(name: "perror", scope: !602, file: !602, line: 804, type: !690, flags: DIFlagPrototyped, spFlags: 0)
!690 = !DISubroutineType(types: !691)
!691 = !{null, !159}
!692 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !693, file: !599, line: 127)
!693 = !DISubprogram(name: "printf", scope: !602, file: !602, line: 356, type: !694, flags: DIFlagPrototyped, spFlags: 0)
!694 = !DISubroutineType(types: !695)
!695 = !{!11, !349, null}
!696 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !697, file: !599, line: 128)
!697 = !DISubprogram(name: "putc", scope: !602, file: !602, line: 550, type: !644, flags: DIFlagPrototyped, spFlags: 0)
!698 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !699, file: !599, line: 129)
!699 = !DISubprogram(name: "putchar", scope: !602, file: !602, line: 556, type: !17, flags: DIFlagPrototyped, spFlags: 0)
!700 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !701, file: !599, line: 130)
!701 = !DISubprogram(name: "puts", scope: !602, file: !602, line: 661, type: !287, flags: DIFlagPrototyped, spFlags: 0)
!702 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !703, file: !599, line: 131)
!703 = !DISubprogram(name: "remove", scope: !602, file: !602, line: 152, type: !287, flags: DIFlagPrototyped, spFlags: 0)
!704 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !705, file: !599, line: 132)
!705 = !DISubprogram(name: "rename", scope: !602, file: !602, line: 154, type: !706, flags: DIFlagPrototyped, spFlags: 0)
!706 = !DISubroutineType(types: !707)
!707 = !{!11, !159, !159}
!708 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !709, file: !599, line: 133)
!709 = !DISubprogram(name: "rewind", scope: !602, file: !602, line: 723, type: !608, flags: DIFlagPrototyped, spFlags: 0)
!710 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !711, file: !599, line: 134)
!711 = !DISubprogram(name: "scanf", scope: !602, file: !602, line: 421, type: !694, flags: DIFlagPrototyped, spFlags: 0)
!712 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !713, file: !599, line: 135)
!713 = !DISubprogram(name: "setbuf", scope: !602, file: !602, line: 328, type: !714, flags: DIFlagPrototyped, spFlags: 0)
!714 = !DISubroutineType(types: !715)
!715 = !{null, !627, !391}
!716 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !717, file: !599, line: 136)
!717 = !DISubprogram(name: "setvbuf", scope: !602, file: !602, line: 332, type: !718, flags: DIFlagPrototyped, spFlags: 0)
!718 = !DISubroutineType(types: !719)
!719 = !{!11, !627, !391, !11, !300}
!720 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !721, file: !599, line: 137)
!721 = !DISubprogram(name: "sprintf", scope: !602, file: !602, line: 358, type: !722, flags: DIFlagPrototyped, spFlags: 0)
!722 = !DISubroutineType(types: !723)
!723 = !{!11, !391, !349, null}
!724 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !725, file: !599, line: 138)
!725 = !DISubprogram(name: "sscanf", scope: !602, file: !602, line: 423, type: !726, flags: DIFlagPrototyped, spFlags: 0)
!726 = !DISubroutineType(types: !727)
!727 = !{!11, !349, !349, null}
!728 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !729, file: !599, line: 139)
!729 = !DISubprogram(name: "tmpfile", scope: !602, file: !602, line: 188, type: !730, flags: DIFlagPrototyped, spFlags: 0)
!730 = !DISubroutineType(types: !731)
!731 = !{!610}
!732 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !733, file: !599, line: 141)
!733 = !DISubprogram(name: "tmpnam", scope: !602, file: !602, line: 205, type: !686, flags: DIFlagPrototyped, spFlags: 0)
!734 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !735, file: !599, line: 143)
!735 = !DISubprogram(name: "ungetc", scope: !602, file: !602, line: 668, type: !644, flags: DIFlagPrototyped, spFlags: 0)
!736 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !737, file: !599, line: 144)
!737 = !DISubprogram(name: "vfprintf", scope: !602, file: !602, line: 365, type: !738, flags: DIFlagPrototyped, spFlags: 0)
!738 = !DISubroutineType(types: !739)
!739 = !{!11, !627, !349, !740}
!740 = !DIDerivedType(tag: DW_TAG_typedef, name: "__gnuc_va_list", file: !741, line: 32, baseType: !742)
!741 = !DIFile(filename: "/u/NAS_SCRATCH/ah7226/tulip/llvm-install-tulip/lib/clang/9.0.0/include/stdarg.h", directory: "")
!742 = !DIDerivedType(tag: DW_TAG_typedef, name: "__builtin_va_list", file: !6, baseType: !327)
!743 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !744, file: !599, line: 145)
!744 = !DISubprogram(name: "vprintf", scope: !602, file: !602, line: 371, type: !745, flags: DIFlagPrototyped, spFlags: 0)
!745 = !DISubroutineType(types: !746)
!746 = !{!11, !349, !740}
!747 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !748, file: !599, line: 146)
!748 = !DISubprogram(name: "vsprintf", scope: !602, file: !602, line: 373, type: !749, flags: DIFlagPrototyped, spFlags: 0)
!749 = !DISubroutineType(types: !750)
!750 = !{!11, !391, !349, !740}
!751 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !400, entity: !752, file: !599, line: 175)
!752 = !DISubprogram(name: "snprintf", scope: !602, file: !602, line: 378, type: !753, flags: DIFlagPrototyped, spFlags: 0)
!753 = !DISubroutineType(types: !754)
!754 = !{!11, !391, !300, !349, null}
!755 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !400, entity: !756, file: !599, line: 176)
!756 = !DISubprogram(name: "vfscanf", scope: !602, file: !602, line: 459, type: !738, flags: DIFlagPrototyped, spFlags: 0)
!757 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !400, entity: !758, file: !599, line: 177)
!758 = !DISubprogram(name: "vscanf", scope: !602, file: !602, line: 467, type: !745, flags: DIFlagPrototyped, spFlags: 0)
!759 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !400, entity: !760, file: !599, line: 178)
!760 = !DISubprogram(name: "vsnprintf", scope: !602, file: !602, line: 382, type: !761, flags: DIFlagPrototyped, spFlags: 0)
!761 = !DISubroutineType(types: !762)
!762 = !{!11, !391, !300, !349, !740}
!763 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !400, entity: !764, file: !599, line: 179)
!764 = !DISubprogram(name: "vsscanf", scope: !602, file: !602, line: 471, type: !765, flags: DIFlagPrototyped, spFlags: 0)
!765 = !DISubroutineType(types: !766)
!766 = !{!11, !349, !349, !740}
!767 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !752, file: !599, line: 185)
!768 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !756, file: !599, line: 186)
!769 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !758, file: !599, line: 187)
!770 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !760, file: !599, line: 188)
!771 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !14, entity: !764, file: !599, line: 189)
!772 = !{void (double*, double*, double*, double*, double*)* @_Z21gpu_kernel_one_devicePdS_S_S_S_, !"kernel", i32 1}
!773 = !{void (double*, double*, double*)* @_Z21gpu_kernel_two_devicePdS_S_, !"kernel", i32 1}
!774 = !{void (i32*, i32*, double*, double*, double*)* @_Z23gpu_kernel_three_devicePiS_PdS0_S0_, !"kernel", i32 1}
!775 = !{void (double*, double*, double*, double*)* @_Z22gpu_kernel_four_devicePdS_S_S_, !"kernel", i32 1}
!776 = !{void (double, double*, double*)* @_Z17gpu_kernel_five_1dPdS_, !"kernel", i32 1}
!777 = !{void (double, double*, double*)* @_Z17gpu_kernel_five_2dPdS_, !"kernel", i32 1}
!778 = !{void (double*, double*)* @_Z21gpu_kernel_six_devicePdS_, !"kernel", i32 1}
!779 = !{void (double, double*, double*)* @_Z23gpu_kernel_seven_devicedPdS_, !"kernel", i32 1}
!780 = !{void (i32*, i32*, double*, double*, double*)* @_Z23gpu_kernel_eight_devicePiS_PdS0_S0_, !"kernel", i32 1}
!781 = !{void (double*, double*, double*, double*)* @_Z22gpu_kernel_nine_devicePdS_S_S_, !"kernel", i32 1}
!782 = !{void (double*, double*, double*)* @_Z16gpu_kernel_ten_1PdS_S_, !"kernel", i32 1}
!783 = !{void (double*, double*, double*)* @_Z16gpu_kernel_ten_2PdS_S_, !"kernel", i32 1}
!784 = !{void (double, double*, double*)* @_Z24gpu_kernel_eleven_devicedPdS_, !"kernel", i32 1}
!785 = !{null, !"align", i32 8}
!786 = !{null, !"align", i32 8, !"align", i32 65544, !"align", i32 131080}
!787 = !{null, !"align", i32 16}
!788 = !{null, !"align", i32 16, !"align", i32 65552, !"align", i32 131088}
!789 = !{!"clang version 9.0.0 (https://github.com/yebinchon/llvm-project/ a1efa594106d738d0b74c9e4e2b9b779eb8b7d25)"}
!790 = !{i32 1, i32 2}
!791 = distinct !DISubprogram(name: "gpu_kernel_one_device", linkageName: "_Z21gpu_kernel_one_devicePdS_S_S_S_", scope: !6, file: !6, line: 1051, type: !792, scopeLine: 1055, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !7)
!792 = !DISubroutineType(types: !793)
!793 = !{null, !9, !9, !9, !9, !9}
!794 = !DILocalVariable(name: "p", arg: 1, scope: !791, file: !6, line: 1051, type: !9)
!795 = !DILocation(line: 1051, column: 46, scope: !791)
!796 = !DILocalVariable(name: "q", arg: 2, scope: !791, file: !6, line: 1052, type: !9)
!797 = !DILocation(line: 1052, column: 10, scope: !791)
!798 = !DILocalVariable(name: "r", arg: 3, scope: !791, file: !6, line: 1053, type: !9)
!799 = !DILocation(line: 1053, column: 10, scope: !791)
!800 = !DILocalVariable(name: "x", arg: 4, scope: !791, file: !6, line: 1054, type: !9)
!801 = !DILocation(line: 1054, column: 10, scope: !791)
!802 = !DILocalVariable(name: "z", arg: 5, scope: !791, file: !6, line: 1055, type: !9)
!803 = !DILocation(line: 1055, column: 10, scope: !791)
!804 = !DILocalVariable(name: "thread_id", scope: !791, file: !6, line: 1056, type: !11)
!805 = !DILocation(line: 1056, column: 6, scope: !791)
!806 = !DILocation(line: 64, column: 3, scope: !807, inlinedAt: !842)
!807 = distinct !DISubprogram(name: "__fetch_builtin_x", linkageName: "_ZN25__cuda_builtin_blockIdx_t17__fetch_builtin_xEv", scope: !809, file: !808, line: 64, type: !812, scopeLine: 64, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !5, declaration: !811, retainedNodes: !7)
!808 = !DIFile(filename: "/u/NAS_SCRATCH/ah7226/tulip/llvm-install-tulip/lib/clang/9.0.0/include/__clang_cuda_builtin_vars.h", directory: "")
!809 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "__cuda_builtin_blockIdx_t", file: !808, line: 63, size: 8, flags: DIFlagTypePassByReference | DIFlagNonTrivial, elements: !810, identifier: "_ZTS25__cuda_builtin_blockIdx_t")
!810 = !{!811, !814, !815, !816, !827, !831, !835, !838}
!811 = !DISubprogram(name: "__fetch_builtin_x", linkageName: "_ZN25__cuda_builtin_blockIdx_t17__fetch_builtin_xEv", scope: !809, file: !808, line: 64, type: !812, scopeLine: 64, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!812 = !DISubroutineType(types: !813)
!813 = !{!370}
!814 = !DISubprogram(name: "__fetch_builtin_y", linkageName: "_ZN25__cuda_builtin_blockIdx_t17__fetch_builtin_yEv", scope: !809, file: !808, line: 65, type: !812, scopeLine: 65, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!815 = !DISubprogram(name: "__fetch_builtin_z", linkageName: "_ZN25__cuda_builtin_blockIdx_t17__fetch_builtin_zEv", scope: !809, file: !808, line: 66, type: !812, scopeLine: 66, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!816 = !DISubprogram(name: "operator uint3", linkageName: "_ZNK25__cuda_builtin_blockIdx_tcv5uint3Ev", scope: !809, file: !808, line: 69, type: !817, scopeLine: 69, flags: DIFlagPrototyped, spFlags: 0)
!817 = !DISubroutineType(types: !818)
!818 = !{!819, !825}
!819 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "uint3", file: !820, line: 190, size: 96, flags: DIFlagTypePassByValue, elements: !821, identifier: "_ZTS5uint3")
!820 = !DIFile(filename: "/usr/local/cuda/include/vector_types.h", directory: "")
!821 = !{!822, !823, !824}
!822 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !819, file: !820, line: 192, baseType: !370, size: 32)
!823 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !819, file: !820, line: 192, baseType: !370, size: 32, offset: 32)
!824 = !DIDerivedType(tag: DW_TAG_member, name: "z", scope: !819, file: !820, line: 192, baseType: !370, size: 32, offset: 64)
!825 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !826, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!826 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !809)
!827 = !DISubprogram(name: "__cuda_builtin_blockIdx_t", scope: !809, file: !808, line: 71, type: !828, scopeLine: 71, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!828 = !DISubroutineType(types: !829)
!829 = !{null, !830}
!830 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !809, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!831 = !DISubprogram(name: "__cuda_builtin_blockIdx_t", scope: !809, file: !808, line: 71, type: !832, scopeLine: 71, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!832 = !DISubroutineType(types: !833)
!833 = !{null, !830, !834}
!834 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !826, size: 64)
!835 = !DISubprogram(name: "operator=", linkageName: "_ZNK25__cuda_builtin_blockIdx_taSERKS_", scope: !809, file: !808, line: 71, type: !836, scopeLine: 71, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!836 = !DISubroutineType(types: !837)
!837 = !{null, !825, !834}
!838 = !DISubprogram(name: "operator&", linkageName: "_ZNK25__cuda_builtin_blockIdx_tadEv", scope: !809, file: !808, line: 71, type: !839, scopeLine: 71, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!839 = !DISubroutineType(types: !840)
!840 = !{!841, !825}
!841 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !809, size: 64)
!842 = distinct !DILocation(line: 1056, column: 18, scope: !791)
!843 = !{i32 0, i32 65535}
!844 = !DILocation(line: 75, column: 3, scope: !845, inlinedAt: !887)
!845 = distinct !DISubprogram(name: "__fetch_builtin_x", linkageName: "_ZN25__cuda_builtin_blockDim_t17__fetch_builtin_xEv", scope: !846, file: !808, line: 75, type: !812, scopeLine: 75, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !5, declaration: !848, retainedNodes: !7)
!846 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "__cuda_builtin_blockDim_t", file: !808, line: 74, size: 8, flags: DIFlagTypePassByReference | DIFlagNonTrivial, elements: !847, identifier: "_ZTS25__cuda_builtin_blockDim_t")
!847 = !{!848, !849, !850, !851, !872, !876, !880, !883}
!848 = !DISubprogram(name: "__fetch_builtin_x", linkageName: "_ZN25__cuda_builtin_blockDim_t17__fetch_builtin_xEv", scope: !846, file: !808, line: 75, type: !812, scopeLine: 75, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!849 = !DISubprogram(name: "__fetch_builtin_y", linkageName: "_ZN25__cuda_builtin_blockDim_t17__fetch_builtin_yEv", scope: !846, file: !808, line: 76, type: !812, scopeLine: 76, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!850 = !DISubprogram(name: "__fetch_builtin_z", linkageName: "_ZN25__cuda_builtin_blockDim_t17__fetch_builtin_zEv", scope: !846, file: !808, line: 77, type: !812, scopeLine: 77, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!851 = !DISubprogram(name: "operator dim3", linkageName: "_ZNK25__cuda_builtin_blockDim_tcv4dim3Ev", scope: !846, file: !808, line: 80, type: !852, scopeLine: 80, flags: DIFlagPrototyped, spFlags: 0)
!852 = !DISubroutineType(types: !853)
!853 = !{!854, !870}
!854 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "dim3", file: !820, line: 417, size: 96, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !855, identifier: "_ZTS4dim3")
!855 = !{!856, !857, !858, !859, !863, !867}
!856 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !854, file: !820, line: 419, baseType: !370, size: 32)
!857 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !854, file: !820, line: 419, baseType: !370, size: 32, offset: 32)
!858 = !DIDerivedType(tag: DW_TAG_member, name: "z", scope: !854, file: !820, line: 419, baseType: !370, size: 32, offset: 64)
!859 = !DISubprogram(name: "dim3", scope: !854, file: !820, line: 421, type: !860, scopeLine: 421, flags: DIFlagPrototyped, spFlags: 0)
!860 = !DISubroutineType(types: !861)
!861 = !{null, !862, !370, !370, !370}
!862 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !854, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!863 = !DISubprogram(name: "dim3", scope: !854, file: !820, line: 422, type: !864, scopeLine: 422, flags: DIFlagPrototyped, spFlags: 0)
!864 = !DISubroutineType(types: !865)
!865 = !{null, !862, !866}
!866 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint3", file: !820, line: 383, baseType: !819)
!867 = !DISubprogram(name: "operator uint3", linkageName: "_ZN4dim3cv5uint3Ev", scope: !854, file: !820, line: 423, type: !868, scopeLine: 423, flags: DIFlagPrototyped, spFlags: 0)
!868 = !DISubroutineType(types: !869)
!869 = !{!866, !862}
!870 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !871, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!871 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !846)
!872 = !DISubprogram(name: "__cuda_builtin_blockDim_t", scope: !846, file: !808, line: 82, type: !873, scopeLine: 82, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!873 = !DISubroutineType(types: !874)
!874 = !{null, !875}
!875 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !846, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!876 = !DISubprogram(name: "__cuda_builtin_blockDim_t", scope: !846, file: !808, line: 82, type: !877, scopeLine: 82, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!877 = !DISubroutineType(types: !878)
!878 = !{null, !875, !879}
!879 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !871, size: 64)
!880 = !DISubprogram(name: "operator=", linkageName: "_ZNK25__cuda_builtin_blockDim_taSERKS_", scope: !846, file: !808, line: 82, type: !881, scopeLine: 82, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!881 = !DISubroutineType(types: !882)
!882 = !{null, !870, !879}
!883 = !DISubprogram(name: "operator&", linkageName: "_ZNK25__cuda_builtin_blockDim_tadEv", scope: !846, file: !808, line: 82, type: !884, scopeLine: 82, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!884 = !DISubroutineType(types: !885)
!885 = !{!886, !870}
!886 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !846, size: 64)
!887 = distinct !DILocation(line: 1056, column: 31, scope: !791)
!888 = !{i32 1, i32 1025}
!889 = !DILocation(line: 1056, column: 29, scope: !791)
!890 = !DILocation(line: 53, column: 3, scope: !891, inlinedAt: !917)
!891 = distinct !DISubprogram(name: "__fetch_builtin_x", linkageName: "_ZN26__cuda_builtin_threadIdx_t17__fetch_builtin_xEv", scope: !892, file: !808, line: 53, type: !812, scopeLine: 53, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !5, declaration: !894, retainedNodes: !7)
!892 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "__cuda_builtin_threadIdx_t", file: !808, line: 52, size: 8, flags: DIFlagTypePassByReference | DIFlagNonTrivial, elements: !893, identifier: "_ZTS26__cuda_builtin_threadIdx_t")
!893 = !{!894, !895, !896, !897, !902, !906, !910, !913}
!894 = !DISubprogram(name: "__fetch_builtin_x", linkageName: "_ZN26__cuda_builtin_threadIdx_t17__fetch_builtin_xEv", scope: !892, file: !808, line: 53, type: !812, scopeLine: 53, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!895 = !DISubprogram(name: "__fetch_builtin_y", linkageName: "_ZN26__cuda_builtin_threadIdx_t17__fetch_builtin_yEv", scope: !892, file: !808, line: 54, type: !812, scopeLine: 54, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!896 = !DISubprogram(name: "__fetch_builtin_z", linkageName: "_ZN26__cuda_builtin_threadIdx_t17__fetch_builtin_zEv", scope: !892, file: !808, line: 55, type: !812, scopeLine: 55, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!897 = !DISubprogram(name: "operator uint3", linkageName: "_ZNK26__cuda_builtin_threadIdx_tcv5uint3Ev", scope: !892, file: !808, line: 58, type: !898, scopeLine: 58, flags: DIFlagPrototyped, spFlags: 0)
!898 = !DISubroutineType(types: !899)
!899 = !{!819, !900}
!900 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !901, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!901 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !892)
!902 = !DISubprogram(name: "__cuda_builtin_threadIdx_t", scope: !892, file: !808, line: 60, type: !903, scopeLine: 60, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!903 = !DISubroutineType(types: !904)
!904 = !{null, !905}
!905 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !892, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!906 = !DISubprogram(name: "__cuda_builtin_threadIdx_t", scope: !892, file: !808, line: 60, type: !907, scopeLine: 60, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!907 = !DISubroutineType(types: !908)
!908 = !{null, !905, !909}
!909 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !901, size: 64)
!910 = !DISubprogram(name: "operator=", linkageName: "_ZNK26__cuda_builtin_threadIdx_taSERKS_", scope: !892, file: !808, line: 60, type: !911, scopeLine: 60, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!911 = !DISubroutineType(types: !912)
!912 = !{null, !900, !909}
!913 = !DISubprogram(name: "operator&", linkageName: "_ZNK26__cuda_builtin_threadIdx_tadEv", scope: !892, file: !808, line: 60, type: !914, scopeLine: 60, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!914 = !DISubroutineType(types: !915)
!915 = !{!916, !900}
!916 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !892, size: 64)
!917 = distinct !DILocation(line: 1056, column: 44, scope: !791)
!918 = !{i32 0, i32 1024}
!919 = !DILocation(line: 1056, column: 42, scope: !791)
!920 = !DILocation(line: 1057, column: 5, scope: !921)
!921 = distinct !DILexicalBlock(scope: !791, file: !6, line: 1057, column: 5)
!922 = !DILocation(line: 1057, column: 15, scope: !921)
!923 = !DILocation(line: 1057, column: 5, scope: !791)
!924 = !DILocation(line: 1057, column: 22, scope: !925)
!925 = distinct !DILexicalBlock(scope: !921, file: !6, line: 1057, column: 21)
!926 = !DILocation(line: 1058, column: 2, scope: !791)
!927 = !DILocation(line: 1058, column: 4, scope: !791)
!928 = !DILocation(line: 1058, column: 15, scope: !791)
!929 = !DILocation(line: 1059, column: 2, scope: !791)
!930 = !DILocation(line: 1059, column: 4, scope: !791)
!931 = !DILocation(line: 1059, column: 15, scope: !791)
!932 = !DILocalVariable(name: "x_value", scope: !791, file: !6, line: 1060, type: !10)
!933 = !DILocation(line: 1060, column: 9, scope: !791)
!934 = !DILocation(line: 1060, column: 19, scope: !791)
!935 = !DILocation(line: 1060, column: 21, scope: !791)
!936 = !DILocation(line: 1061, column: 17, scope: !791)
!937 = !DILocation(line: 1061, column: 2, scope: !791)
!938 = !DILocation(line: 1061, column: 4, scope: !791)
!939 = !DILocation(line: 1061, column: 15, scope: !791)
!940 = !DILocation(line: 1062, column: 17, scope: !791)
!941 = !DILocation(line: 1062, column: 2, scope: !791)
!942 = !DILocation(line: 1062, column: 4, scope: !791)
!943 = !DILocation(line: 1062, column: 15, scope: !791)
!944 = !DILocation(line: 1063, column: 1, scope: !791)
!945 = distinct !DISubprogram(name: "gpu_kernel_two_device", linkageName: "_Z21gpu_kernel_two_devicePdS_S_", scope: !6, file: !6, line: 1084, type: !946, scopeLine: 1086, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !7)
!946 = !DISubroutineType(types: !947)
!947 = !{null, !9, !9, !9}
!948 = !DILocalVariable(name: "r", arg: 1, scope: !945, file: !6, line: 1084, type: !9)
!949 = !DILocation(line: 1084, column: 46, scope: !945)
!950 = !DILocalVariable(name: "rho", arg: 2, scope: !945, file: !6, line: 1085, type: !9)
!951 = !DILocation(line: 1085, column: 11, scope: !945)
!952 = !DILocalVariable(name: "global_data", arg: 3, scope: !945, file: !6, line: 1086, type: !9)
!953 = !DILocation(line: 1086, column: 10, scope: !945)
!954 = !DILocalVariable(name: "share_data", scope: !945, file: !6, line: 1087, type: !9)
!955 = !DILocation(line: 1087, column: 10, scope: !945)
!956 = !DILocalVariable(name: "thread_id", scope: !945, file: !6, line: 1089, type: !11)
!957 = !DILocation(line: 1089, column: 6, scope: !945)
!958 = !DILocation(line: 64, column: 3, scope: !807, inlinedAt: !959)
!959 = distinct !DILocation(line: 1089, column: 18, scope: !945)
!960 = !DILocation(line: 75, column: 3, scope: !845, inlinedAt: !961)
!961 = distinct !DILocation(line: 1089, column: 31, scope: !945)
!962 = !DILocation(line: 1089, column: 29, scope: !945)
!963 = !DILocation(line: 53, column: 3, scope: !891, inlinedAt: !964)
!964 = distinct !DILocation(line: 1089, column: 44, scope: !945)
!965 = !DILocation(line: 1089, column: 42, scope: !945)
!966 = !DILocalVariable(name: "local_id", scope: !945, file: !6, line: 1090, type: !11)
!967 = !DILocation(line: 1090, column: 6, scope: !945)
!968 = !DILocation(line: 53, column: 3, scope: !891, inlinedAt: !969)
!969 = distinct !DILocation(line: 1090, column: 17, scope: !945)
!970 = !DILocation(line: 1092, column: 2, scope: !945)
!971 = !DILocation(line: 1092, column: 13, scope: !945)
!972 = !DILocation(line: 1092, column: 23, scope: !945)
!973 = !DILocation(line: 1096, column: 5, scope: !974)
!974 = distinct !DILexicalBlock(scope: !945, file: !6, line: 1096, column: 5)
!975 = !DILocation(line: 1096, column: 15, scope: !974)
!976 = !DILocation(line: 1096, column: 5, scope: !945)
!977 = !DILocalVariable(name: "r_value", scope: !978, file: !6, line: 1097, type: !10)
!978 = distinct !DILexicalBlock(scope: !974, file: !6, line: 1096, column: 20)
!979 = !DILocation(line: 1097, column: 16, scope: !978)
!980 = !DILocation(line: 1097, column: 26, scope: !978)
!981 = !DILocation(line: 1097, column: 28, scope: !978)
!982 = !DILocation(line: 1098, column: 32, scope: !978)
!983 = !DILocation(line: 1098, column: 42, scope: !978)
!984 = !DILocation(line: 1098, column: 40, scope: !978)
!985 = !DILocation(line: 1098, column: 9, scope: !978)
!986 = !DILocation(line: 1098, column: 20, scope: !978)
!987 = !DILocation(line: 1098, column: 30, scope: !978)
!988 = !DILocation(line: 1099, column: 5, scope: !978)
!989 = !DILocation(line: 1109, column: 2, scope: !945)
!990 = !DILocation(line: 1110, column: 5, scope: !991)
!991 = distinct !DILexicalBlock(scope: !945, file: !6, line: 1110, column: 5)
!992 = !DILocation(line: 1110, column: 13, scope: !991)
!993 = !DILocation(line: 1110, column: 5, scope: !945)
!994 = !DILocalVariable(name: "i", scope: !995, file: !6, line: 1111, type: !11)
!995 = distinct !DILexicalBlock(scope: !996, file: !6, line: 1111, column: 3)
!996 = distinct !DILexicalBlock(scope: !991, file: !6, line: 1110, column: 17)
!997 = !DILocation(line: 1111, column: 11, scope: !995)
!998 = !DILocation(line: 1111, column: 7, scope: !995)
!999 = !DILocation(line: 1111, column: 16, scope: !1000)
!1000 = distinct !DILexicalBlock(scope: !995, file: !6, line: 1111, column: 3)
!1001 = !DILocation(line: 75, column: 3, scope: !845, inlinedAt: !1002)
!1002 = distinct !DILocation(line: 1111, column: 18, scope: !1000)
!1003 = !DILocation(line: 1111, column: 17, scope: !1000)
!1004 = !DILocation(line: 1111, column: 3, scope: !995)
!1005 = !DILocation(line: 1112, column: 19, scope: !1006)
!1006 = distinct !DILexicalBlock(scope: !1000, file: !6, line: 1111, column: 34)
!1007 = !DILocation(line: 1112, column: 30, scope: !1006)
!1008 = !DILocation(line: 1112, column: 4, scope: !1006)
!1009 = !DILocation(line: 1112, column: 17, scope: !1006)
!1010 = !DILocation(line: 1113, column: 3, scope: !1006)
!1011 = !DILocation(line: 1111, column: 31, scope: !1000)
!1012 = !DILocation(line: 1111, column: 3, scope: !1000)
!1013 = distinct !{!1013, !1004, !1014}
!1014 = !DILocation(line: 1113, column: 3, scope: !995)
!1015 = !DILocation(line: 1114, column: 27, scope: !996)
!1016 = !DILocation(line: 1114, column: 3, scope: !996)
!1017 = !DILocation(line: 64, column: 3, scope: !807, inlinedAt: !1018)
!1018 = distinct !DILocation(line: 1114, column: 15, scope: !996)
!1019 = !DILocation(line: 1114, column: 26, scope: !996)
!1020 = !DILocation(line: 1115, column: 2, scope: !996)
!1021 = !DILocation(line: 1116, column: 1, scope: !945)
!1022 = distinct !DISubprogram(name: "gpu_kernel_three_device", linkageName: "_Z23gpu_kernel_three_devicePiS_PdS0_S0_", scope: !6, file: !6, line: 1135, type: !1023, scopeLine: 1139, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !7)
!1023 = !DISubroutineType(types: !1024)
!1024 = !{null, !82, !82, !9, !9, !9}
!1025 = !DILocalVariable(name: "colidx", arg: 1, scope: !1022, file: !6, line: 1135, type: !82)
!1026 = !DILocation(line: 1135, column: 45, scope: !1022)
!1027 = !DILocalVariable(name: "rowstr", arg: 2, scope: !1022, file: !6, line: 1136, type: !82)
!1028 = !DILocation(line: 1136, column: 7, scope: !1022)
!1029 = !DILocalVariable(name: "a", arg: 3, scope: !1022, file: !6, line: 1137, type: !9)
!1030 = !DILocation(line: 1137, column: 10, scope: !1022)
!1031 = !DILocalVariable(name: "p", arg: 4, scope: !1022, file: !6, line: 1138, type: !9)
!1032 = !DILocation(line: 1138, column: 10, scope: !1022)
!1033 = !DILocalVariable(name: "q", arg: 5, scope: !1022, file: !6, line: 1139, type: !9)
!1034 = !DILocation(line: 1139, column: 10, scope: !1022)
!1035 = !DILocalVariable(name: "share_data", scope: !1022, file: !6, line: 1140, type: !9)
!1036 = !DILocation(line: 1140, column: 10, scope: !1022)
!1037 = !DILocalVariable(name: "j", scope: !1022, file: !6, line: 1142, type: !11)
!1038 = !DILocation(line: 1142, column: 6, scope: !1022)
!1039 = !DILocation(line: 64, column: 3, scope: !807, inlinedAt: !1040)
!1040 = distinct !DILocation(line: 1142, column: 18, scope: !1022)
!1041 = !DILocation(line: 75, column: 3, scope: !845, inlinedAt: !1042)
!1042 = distinct !DILocation(line: 1142, column: 29, scope: !1022)
!1043 = !DILocation(line: 1142, column: 28, scope: !1022)
!1044 = !DILocation(line: 53, column: 3, scope: !891, inlinedAt: !1045)
!1045 = distinct !DILocation(line: 1142, column: 40, scope: !1022)
!1046 = !DILocation(line: 1142, column: 39, scope: !1022)
!1047 = !DILocation(line: 75, column: 3, scope: !845, inlinedAt: !1048)
!1048 = distinct !DILocation(line: 1142, column: 55, scope: !1022)
!1049 = !DILocation(line: 1142, column: 53, scope: !1022)
!1050 = !DILocalVariable(name: "local_id", scope: !1022, file: !6, line: 1143, type: !11)
!1051 = !DILocation(line: 1143, column: 6, scope: !1022)
!1052 = !DILocation(line: 53, column: 3, scope: !891, inlinedAt: !1053)
!1053 = distinct !DILocation(line: 1143, column: 17, scope: !1022)
!1054 = !DILocalVariable(name: "begin", scope: !1022, file: !6, line: 1145, type: !11)
!1055 = !DILocation(line: 1145, column: 6, scope: !1022)
!1056 = !DILocation(line: 1145, column: 14, scope: !1022)
!1057 = !DILocation(line: 1145, column: 21, scope: !1022)
!1058 = !DILocalVariable(name: "end", scope: !1022, file: !6, line: 1146, type: !11)
!1059 = !DILocation(line: 1146, column: 6, scope: !1022)
!1060 = !DILocation(line: 1146, column: 12, scope: !1022)
!1061 = !DILocation(line: 1146, column: 19, scope: !1022)
!1062 = !DILocation(line: 1146, column: 20, scope: !1022)
!1063 = !DILocalVariable(name: "sum", scope: !1022, file: !6, line: 1147, type: !10)
!1064 = !DILocation(line: 1147, column: 9, scope: !1022)
!1065 = !DILocalVariable(name: "k", scope: !1066, file: !6, line: 1148, type: !11)
!1066 = distinct !DILexicalBlock(scope: !1022, file: !6, line: 1148, column: 2)
!1067 = !DILocation(line: 1148, column: 10, scope: !1066)
!1068 = !DILocation(line: 1148, column: 12, scope: !1066)
!1069 = !DILocation(line: 1148, column: 18, scope: !1066)
!1070 = !DILocation(line: 1148, column: 17, scope: !1066)
!1071 = !DILocation(line: 1148, column: 6, scope: !1066)
!1072 = !DILocation(line: 1148, column: 28, scope: !1073)
!1073 = distinct !DILexicalBlock(scope: !1066, file: !6, line: 1148, column: 2)
!1074 = !DILocation(line: 1148, column: 30, scope: !1073)
!1075 = !DILocation(line: 1148, column: 29, scope: !1073)
!1076 = !DILocation(line: 1148, column: 2, scope: !1066)
!1077 = !DILocation(line: 1149, column: 9, scope: !1078)
!1078 = distinct !DILexicalBlock(scope: !1073, file: !6, line: 1148, column: 49)
!1079 = !DILocation(line: 1149, column: 15, scope: !1078)
!1080 = !DILocation(line: 1149, column: 17, scope: !1078)
!1081 = !DILocation(line: 1149, column: 20, scope: !1078)
!1082 = !DILocation(line: 1149, column: 22, scope: !1078)
!1083 = !DILocation(line: 1149, column: 29, scope: !1078)
!1084 = !DILocation(line: 1149, column: 19, scope: !1078)
!1085 = !DILocation(line: 1149, column: 13, scope: !1078)
!1086 = !DILocation(line: 1149, column: 7, scope: !1078)
!1087 = !DILocation(line: 1150, column: 2, scope: !1078)
!1088 = !DILocation(line: 75, column: 3, scope: !845, inlinedAt: !1089)
!1089 = distinct !DILocation(line: 1148, column: 38, scope: !1073)
!1090 = !DILocation(line: 1148, column: 36, scope: !1073)
!1091 = !DILocation(line: 1148, column: 2, scope: !1073)
!1092 = distinct !{!1092, !1076, !1093}
!1093 = !DILocation(line: 1150, column: 2, scope: !1066)
!1094 = !DILocation(line: 1151, column: 25, scope: !1022)
!1095 = !DILocation(line: 1151, column: 2, scope: !1022)
!1096 = !DILocation(line: 1151, column: 13, scope: !1022)
!1097 = !DILocation(line: 1151, column: 23, scope: !1022)
!1098 = !DILocation(line: 1161, column: 2, scope: !1022)
!1099 = !DILocation(line: 1162, column: 5, scope: !1100)
!1100 = distinct !DILexicalBlock(scope: !1022, file: !6, line: 1162, column: 5)
!1101 = !DILocation(line: 1162, column: 13, scope: !1100)
!1102 = !DILocation(line: 1162, column: 5, scope: !1022)
!1103 = !DILocalVariable(name: "i", scope: !1104, file: !6, line: 1163, type: !11)
!1104 = distinct !DILexicalBlock(scope: !1105, file: !6, line: 1163, column: 3)
!1105 = distinct !DILexicalBlock(scope: !1100, file: !6, line: 1162, column: 17)
!1106 = !DILocation(line: 1163, column: 11, scope: !1104)
!1107 = !DILocation(line: 1163, column: 7, scope: !1104)
!1108 = !DILocation(line: 1163, column: 16, scope: !1109)
!1109 = distinct !DILexicalBlock(scope: !1104, file: !6, line: 1163, column: 3)
!1110 = !DILocation(line: 75, column: 3, scope: !845, inlinedAt: !1111)
!1111 = distinct !DILocation(line: 1163, column: 18, scope: !1109)
!1112 = !DILocation(line: 1163, column: 17, scope: !1109)
!1113 = !DILocation(line: 1163, column: 3, scope: !1104)
!1114 = !DILocation(line: 1164, column: 19, scope: !1115)
!1115 = distinct !DILexicalBlock(scope: !1109, file: !6, line: 1163, column: 34)
!1116 = !DILocation(line: 1164, column: 30, scope: !1115)
!1117 = !DILocation(line: 1164, column: 4, scope: !1115)
!1118 = !DILocation(line: 1164, column: 17, scope: !1115)
!1119 = !DILocation(line: 1165, column: 3, scope: !1115)
!1120 = !DILocation(line: 1163, column: 31, scope: !1109)
!1121 = !DILocation(line: 1163, column: 3, scope: !1109)
!1122 = distinct !{!1122, !1113, !1123}
!1123 = !DILocation(line: 1165, column: 3, scope: !1104)
!1124 = !DILocation(line: 1166, column: 8, scope: !1105)
!1125 = !DILocation(line: 1166, column: 3, scope: !1105)
!1126 = !DILocation(line: 1166, column: 5, scope: !1105)
!1127 = !DILocation(line: 1166, column: 7, scope: !1105)
!1128 = !DILocation(line: 1167, column: 2, scope: !1105)
!1129 = !DILocation(line: 1168, column: 1, scope: !1022)
!1130 = distinct !DISubprogram(name: "gpu_kernel_four_device", linkageName: "_Z22gpu_kernel_four_devicePdS_S_S_", scope: !6, file: !6, line: 1190, type: !1131, scopeLine: 1193, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !7)
!1131 = !DISubroutineType(types: !1132)
!1132 = !{null, !9, !9, !9, !9}
!1133 = !DILocalVariable(name: "d", arg: 1, scope: !1130, file: !6, line: 1190, type: !9)
!1134 = !DILocation(line: 1190, column: 48, scope: !1130)
!1135 = !DILocalVariable(name: "p", arg: 2, scope: !1130, file: !6, line: 1191, type: !9)
!1136 = !DILocation(line: 1191, column: 11, scope: !1130)
!1137 = !DILocalVariable(name: "q", arg: 3, scope: !1130, file: !6, line: 1192, type: !9)
!1138 = !DILocation(line: 1192, column: 11, scope: !1130)
!1139 = !DILocalVariable(name: "global_data", arg: 4, scope: !1130, file: !6, line: 1193, type: !9)
!1140 = !DILocation(line: 1193, column: 10, scope: !1130)
!1141 = !DILocalVariable(name: "share_data", scope: !1130, file: !6, line: 1194, type: !9)
!1142 = !DILocation(line: 1194, column: 10, scope: !1130)
!1143 = !DILocalVariable(name: "thread_id", scope: !1130, file: !6, line: 1196, type: !11)
!1144 = !DILocation(line: 1196, column: 6, scope: !1130)
!1145 = !DILocation(line: 64, column: 3, scope: !807, inlinedAt: !1146)
!1146 = distinct !DILocation(line: 1196, column: 18, scope: !1130)
!1147 = !DILocation(line: 75, column: 3, scope: !845, inlinedAt: !1148)
!1148 = distinct !DILocation(line: 1196, column: 31, scope: !1130)
!1149 = !DILocation(line: 1196, column: 29, scope: !1130)
!1150 = !DILocation(line: 53, column: 3, scope: !891, inlinedAt: !1151)
!1151 = distinct !DILocation(line: 1196, column: 44, scope: !1130)
!1152 = !DILocation(line: 1196, column: 42, scope: !1130)
!1153 = !DILocalVariable(name: "local_id", scope: !1130, file: !6, line: 1197, type: !11)
!1154 = !DILocation(line: 1197, column: 6, scope: !1130)
!1155 = !DILocation(line: 53, column: 3, scope: !891, inlinedAt: !1156)
!1156 = distinct !DILocation(line: 1197, column: 17, scope: !1130)
!1157 = !DILocation(line: 1199, column: 2, scope: !1130)
!1158 = !DILocation(line: 1199, column: 13, scope: !1130)
!1159 = !DILocation(line: 1199, column: 23, scope: !1130)
!1160 = !DILocation(line: 1203, column: 2, scope: !1130)
!1161 = !DILocation(line: 1203, column: 13, scope: !1130)
!1162 = !DILocation(line: 1203, column: 23, scope: !1130)
!1163 = !DILocation(line: 1205, column: 5, scope: !1164)
!1164 = distinct !DILexicalBlock(scope: !1130, file: !6, line: 1205, column: 5)
!1165 = !DILocation(line: 1205, column: 15, scope: !1164)
!1166 = !DILocation(line: 1205, column: 5, scope: !1130)
!1167 = !DILocation(line: 1206, column: 35, scope: !1168)
!1168 = distinct !DILexicalBlock(scope: !1164, file: !6, line: 1205, column: 20)
!1169 = !DILocation(line: 1206, column: 37, scope: !1168)
!1170 = !DILocation(line: 1206, column: 50, scope: !1168)
!1171 = !DILocation(line: 1206, column: 52, scope: !1168)
!1172 = !DILocation(line: 1206, column: 48, scope: !1168)
!1173 = !DILocation(line: 1206, column: 9, scope: !1168)
!1174 = !DILocation(line: 53, column: 3, scope: !891, inlinedAt: !1175)
!1175 = distinct !DILocation(line: 1206, column: 20, scope: !1168)
!1176 = !DILocation(line: 1206, column: 33, scope: !1168)
!1177 = !DILocation(line: 1207, column: 5, scope: !1168)
!1178 = !DILocation(line: 1217, column: 2, scope: !1130)
!1179 = !DILocation(line: 1218, column: 5, scope: !1180)
!1180 = distinct !DILexicalBlock(scope: !1130, file: !6, line: 1218, column: 5)
!1181 = !DILocation(line: 1218, column: 13, scope: !1180)
!1182 = !DILocation(line: 1218, column: 5, scope: !1130)
!1183 = !DILocalVariable(name: "i", scope: !1184, file: !6, line: 1219, type: !11)
!1184 = distinct !DILexicalBlock(scope: !1185, file: !6, line: 1219, column: 3)
!1185 = distinct !DILexicalBlock(scope: !1180, file: !6, line: 1218, column: 17)
!1186 = !DILocation(line: 1219, column: 11, scope: !1184)
!1187 = !DILocation(line: 1219, column: 7, scope: !1184)
!1188 = !DILocation(line: 1219, column: 16, scope: !1189)
!1189 = distinct !DILexicalBlock(scope: !1184, file: !6, line: 1219, column: 3)
!1190 = !DILocation(line: 75, column: 3, scope: !845, inlinedAt: !1191)
!1191 = distinct !DILocation(line: 1219, column: 18, scope: !1189)
!1192 = !DILocation(line: 1219, column: 17, scope: !1189)
!1193 = !DILocation(line: 1219, column: 3, scope: !1184)
!1194 = !DILocation(line: 1220, column: 19, scope: !1195)
!1195 = distinct !DILexicalBlock(scope: !1189, file: !6, line: 1219, column: 34)
!1196 = !DILocation(line: 1220, column: 30, scope: !1195)
!1197 = !DILocation(line: 1220, column: 4, scope: !1195)
!1198 = !DILocation(line: 1220, column: 17, scope: !1195)
!1199 = !DILocation(line: 1221, column: 3, scope: !1195)
!1200 = !DILocation(line: 1219, column: 31, scope: !1189)
!1201 = !DILocation(line: 1219, column: 3, scope: !1189)
!1202 = distinct !{!1202, !1193, !1203}
!1203 = !DILocation(line: 1221, column: 3, scope: !1184)
!1204 = !DILocation(line: 1222, column: 27, scope: !1185)
!1205 = !DILocation(line: 1222, column: 3, scope: !1185)
!1206 = !DILocation(line: 64, column: 3, scope: !807, inlinedAt: !1207)
!1207 = distinct !DILocation(line: 1222, column: 15, scope: !1185)
!1208 = !DILocation(line: 1222, column: 26, scope: !1185)
!1209 = !DILocation(line: 1223, column: 2, scope: !1185)
!1210 = !DILocation(line: 1224, column: 1, scope: !1130)
!1211 = distinct !DISubprogram(name: "gpu_kernel_five_1", linkageName: "_Z17gpu_kernel_five_1dPdS_", scope: !6, file: !6, line: 1245, type: !1212, scopeLine: 1247, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !7)
!1212 = !DISubroutineType(types: !1213)
!1213 = !{null, !10, !9, !9}
!1214 = !DILocalVariable(name: "alpha", arg: 1, scope: !1211, file: !6, line: 1245, type: !10)
!1215 = !DILocation(line: 1245, column: 42, scope: !1211)
!1216 = !DILocalVariable(name: "p", arg: 2, scope: !1211, file: !6, line: 1246, type: !9)
!1217 = !DILocation(line: 1246, column: 11, scope: !1211)
!1218 = !DILocalVariable(name: "z", arg: 3, scope: !1211, file: !6, line: 1247, type: !9)
!1219 = !DILocation(line: 1247, column: 11, scope: !1211)
!1220 = !DILocalVariable(name: "j", scope: !1211, file: !6, line: 1248, type: !11)
!1221 = !DILocation(line: 1248, column: 6, scope: !1211)
!1222 = !DILocation(line: 64, column: 3, scope: !807, inlinedAt: !1223)
!1223 = distinct !DILocation(line: 1248, column: 10, scope: !1211)
!1224 = !DILocation(line: 75, column: 3, scope: !845, inlinedAt: !1225)
!1225 = distinct !DILocation(line: 1248, column: 23, scope: !1211)
!1226 = !DILocation(line: 1248, column: 21, scope: !1211)
!1227 = !DILocation(line: 53, column: 3, scope: !891, inlinedAt: !1228)
!1228 = distinct !DILocation(line: 1248, column: 36, scope: !1211)
!1229 = !DILocation(line: 1248, column: 34, scope: !1211)
!1230 = !DILocation(line: 1249, column: 5, scope: !1231)
!1231 = distinct !DILexicalBlock(scope: !1211, file: !6, line: 1249, column: 5)
!1232 = !DILocation(line: 1249, column: 7, scope: !1231)
!1233 = !DILocation(line: 1249, column: 5, scope: !1211)
!1234 = !DILocation(line: 1249, column: 14, scope: !1235)
!1235 = distinct !DILexicalBlock(scope: !1231, file: !6, line: 1249, column: 13)
!1236 = !DILocation(line: 1250, column: 10, scope: !1211)
!1237 = !DILocation(line: 1250, column: 18, scope: !1211)
!1238 = !DILocation(line: 1250, column: 20, scope: !1211)
!1239 = !DILocation(line: 1250, column: 16, scope: !1211)
!1240 = !DILocation(line: 1250, column: 2, scope: !1211)
!1241 = !DILocation(line: 1250, column: 4, scope: !1211)
!1242 = !DILocation(line: 1250, column: 7, scope: !1211)
!1243 = !DILocation(line: 1251, column: 1, scope: !1211)
!1244 = distinct !DISubprogram(name: "gpu_kernel_five_2", linkageName: "_Z17gpu_kernel_five_2dPdS_", scope: !6, file: !6, line: 1253, type: !1212, scopeLine: 1255, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !7)
!1245 = !DILocalVariable(name: "alpha", arg: 1, scope: !1244, file: !6, line: 1253, type: !10)
!1246 = !DILocation(line: 1253, column: 42, scope: !1244)
!1247 = !DILocalVariable(name: "q", arg: 2, scope: !1244, file: !6, line: 1254, type: !9)
!1248 = !DILocation(line: 1254, column: 11, scope: !1244)
!1249 = !DILocalVariable(name: "r", arg: 3, scope: !1244, file: !6, line: 1255, type: !9)
!1250 = !DILocation(line: 1255, column: 11, scope: !1244)
!1251 = !DILocalVariable(name: "j", scope: !1244, file: !6, line: 1256, type: !11)
!1252 = !DILocation(line: 1256, column: 6, scope: !1244)
!1253 = !DILocation(line: 64, column: 3, scope: !807, inlinedAt: !1254)
!1254 = distinct !DILocation(line: 1256, column: 10, scope: !1244)
!1255 = !DILocation(line: 75, column: 3, scope: !845, inlinedAt: !1256)
!1256 = distinct !DILocation(line: 1256, column: 23, scope: !1244)
!1257 = !DILocation(line: 1256, column: 21, scope: !1244)
!1258 = !DILocation(line: 53, column: 3, scope: !891, inlinedAt: !1259)
!1259 = distinct !DILocation(line: 1256, column: 36, scope: !1244)
!1260 = !DILocation(line: 1256, column: 34, scope: !1244)
!1261 = !DILocation(line: 1257, column: 5, scope: !1262)
!1262 = distinct !DILexicalBlock(scope: !1244, file: !6, line: 1257, column: 5)
!1263 = !DILocation(line: 1257, column: 7, scope: !1262)
!1264 = !DILocation(line: 1257, column: 5, scope: !1244)
!1265 = !DILocation(line: 1257, column: 14, scope: !1266)
!1266 = distinct !DILexicalBlock(scope: !1262, file: !6, line: 1257, column: 13)
!1267 = !DILocation(line: 1258, column: 10, scope: !1244)
!1268 = !DILocation(line: 1258, column: 18, scope: !1244)
!1269 = !DILocation(line: 1258, column: 20, scope: !1244)
!1270 = !DILocation(line: 1258, column: 16, scope: !1244)
!1271 = !DILocation(line: 1258, column: 2, scope: !1244)
!1272 = !DILocation(line: 1258, column: 4, scope: !1244)
!1273 = !DILocation(line: 1258, column: 7, scope: !1244)
!1274 = !DILocation(line: 1259, column: 1, scope: !1244)
!1275 = distinct !DISubprogram(name: "gpu_kernel_six_device", linkageName: "_Z21gpu_kernel_six_devicePdS_", scope: !6, file: !6, line: 1279, type: !1276, scopeLine: 1280, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !7)
!1276 = !DISubroutineType(types: !1277)
!1277 = !{null, !9, !9}
!1278 = !DILocalVariable(name: "r", arg: 1, scope: !1275, file: !6, line: 1279, type: !9)
!1279 = !DILocation(line: 1279, column: 46, scope: !1275)
!1280 = !DILocalVariable(name: "global_data", arg: 2, scope: !1275, file: !6, line: 1280, type: !9)
!1281 = !DILocation(line: 1280, column: 10, scope: !1275)
!1282 = !DILocalVariable(name: "share_data", scope: !1275, file: !6, line: 1281, type: !9)
!1283 = !DILocation(line: 1281, column: 10, scope: !1275)
!1284 = !DILocalVariable(name: "thread_id", scope: !1275, file: !6, line: 1282, type: !11)
!1285 = !DILocation(line: 1282, column: 6, scope: !1275)
!1286 = !DILocation(line: 64, column: 3, scope: !807, inlinedAt: !1287)
!1287 = distinct !DILocation(line: 1282, column: 18, scope: !1275)
!1288 = !DILocation(line: 75, column: 3, scope: !845, inlinedAt: !1289)
!1289 = distinct !DILocation(line: 1282, column: 31, scope: !1275)
!1290 = !DILocation(line: 1282, column: 29, scope: !1275)
!1291 = !DILocation(line: 53, column: 3, scope: !891, inlinedAt: !1292)
!1292 = distinct !DILocation(line: 1282, column: 44, scope: !1275)
!1293 = !DILocation(line: 1282, column: 42, scope: !1275)
!1294 = !DILocalVariable(name: "local_id", scope: !1275, file: !6, line: 1283, type: !11)
!1295 = !DILocation(line: 1283, column: 6, scope: !1275)
!1296 = !DILocation(line: 53, column: 3, scope: !891, inlinedAt: !1297)
!1297 = distinct !DILocation(line: 1283, column: 17, scope: !1275)
!1298 = !DILocation(line: 1284, column: 2, scope: !1275)
!1299 = !DILocation(line: 1284, column: 13, scope: !1275)
!1300 = !DILocation(line: 1284, column: 23, scope: !1275)
!1301 = !DILocation(line: 1286, column: 5, scope: !1302)
!1302 = distinct !DILexicalBlock(scope: !1275, file: !6, line: 1286, column: 5)
!1303 = !DILocation(line: 1286, column: 15, scope: !1302)
!1304 = !DILocation(line: 1286, column: 5, scope: !1275)
!1305 = !DILocalVariable(name: "r_value", scope: !1306, file: !6, line: 1287, type: !10)
!1306 = distinct !DILexicalBlock(scope: !1302, file: !6, line: 1286, column: 20)
!1307 = !DILocation(line: 1287, column: 16, scope: !1306)
!1308 = !DILocation(line: 1287, column: 26, scope: !1306)
!1309 = !DILocation(line: 1287, column: 28, scope: !1306)
!1310 = !DILocation(line: 1288, column: 32, scope: !1306)
!1311 = !DILocation(line: 1288, column: 42, scope: !1306)
!1312 = !DILocation(line: 1288, column: 40, scope: !1306)
!1313 = !DILocation(line: 1288, column: 9, scope: !1306)
!1314 = !DILocation(line: 1288, column: 20, scope: !1306)
!1315 = !DILocation(line: 1288, column: 30, scope: !1306)
!1316 = !DILocation(line: 1289, column: 5, scope: !1306)
!1317 = !DILocation(line: 1298, column: 2, scope: !1275)
!1318 = !DILocation(line: 1299, column: 5, scope: !1319)
!1319 = distinct !DILexicalBlock(scope: !1275, file: !6, line: 1299, column: 5)
!1320 = !DILocation(line: 1299, column: 13, scope: !1319)
!1321 = !DILocation(line: 1299, column: 5, scope: !1275)
!1322 = !DILocalVariable(name: "i", scope: !1323, file: !6, line: 1300, type: !11)
!1323 = distinct !DILexicalBlock(scope: !1324, file: !6, line: 1300, column: 3)
!1324 = distinct !DILexicalBlock(scope: !1319, file: !6, line: 1299, column: 17)
!1325 = !DILocation(line: 1300, column: 11, scope: !1323)
!1326 = !DILocation(line: 1300, column: 7, scope: !1323)
!1327 = !DILocation(line: 1300, column: 16, scope: !1328)
!1328 = distinct !DILexicalBlock(scope: !1323, file: !6, line: 1300, column: 3)
!1329 = !DILocation(line: 75, column: 3, scope: !845, inlinedAt: !1330)
!1330 = distinct !DILocation(line: 1300, column: 18, scope: !1328)
!1331 = !DILocation(line: 1300, column: 17, scope: !1328)
!1332 = !DILocation(line: 1300, column: 3, scope: !1323)
!1333 = !DILocation(line: 1301, column: 19, scope: !1334)
!1334 = distinct !DILexicalBlock(scope: !1328, file: !6, line: 1300, column: 34)
!1335 = !DILocation(line: 1301, column: 30, scope: !1334)
!1336 = !DILocation(line: 1301, column: 4, scope: !1334)
!1337 = !DILocation(line: 1301, column: 17, scope: !1334)
!1338 = !DILocation(line: 1302, column: 3, scope: !1334)
!1339 = !DILocation(line: 1300, column: 31, scope: !1328)
!1340 = !DILocation(line: 1300, column: 3, scope: !1328)
!1341 = distinct !{!1341, !1332, !1342}
!1342 = !DILocation(line: 1302, column: 3, scope: !1323)
!1343 = !DILocation(line: 1303, column: 27, scope: !1324)
!1344 = !DILocation(line: 1303, column: 3, scope: !1324)
!1345 = !DILocation(line: 64, column: 3, scope: !807, inlinedAt: !1346)
!1346 = distinct !DILocation(line: 1303, column: 15, scope: !1324)
!1347 = !DILocation(line: 1303, column: 26, scope: !1324)
!1348 = !DILocation(line: 1304, column: 2, scope: !1324)
!1349 = !DILocation(line: 1305, column: 1, scope: !1275)
!1350 = distinct !DISubprogram(name: "gpu_kernel_seven_device", linkageName: "_Z23gpu_kernel_seven_devicedPdS_", scope: !6, file: !6, line: 1321, type: !1212, scopeLine: 1323, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !7)
!1351 = !DILocalVariable(name: "beta", arg: 1, scope: !1350, file: !6, line: 1321, type: !10)
!1352 = !DILocation(line: 1321, column: 48, scope: !1350)
!1353 = !DILocalVariable(name: "p", arg: 2, scope: !1350, file: !6, line: 1322, type: !9)
!1354 = !DILocation(line: 1322, column: 11, scope: !1350)
!1355 = !DILocalVariable(name: "r", arg: 3, scope: !1350, file: !6, line: 1323, type: !9)
!1356 = !DILocation(line: 1323, column: 11, scope: !1350)
!1357 = !DILocalVariable(name: "j", scope: !1350, file: !6, line: 1324, type: !11)
!1358 = !DILocation(line: 1324, column: 6, scope: !1350)
!1359 = !DILocation(line: 64, column: 3, scope: !807, inlinedAt: !1360)
!1360 = distinct !DILocation(line: 1324, column: 10, scope: !1350)
!1361 = !DILocation(line: 75, column: 3, scope: !845, inlinedAt: !1362)
!1362 = distinct !DILocation(line: 1324, column: 23, scope: !1350)
!1363 = !DILocation(line: 1324, column: 21, scope: !1350)
!1364 = !DILocation(line: 53, column: 3, scope: !891, inlinedAt: !1365)
!1365 = distinct !DILocation(line: 1324, column: 36, scope: !1350)
!1366 = !DILocation(line: 1324, column: 34, scope: !1350)
!1367 = !DILocation(line: 1325, column: 5, scope: !1368)
!1368 = distinct !DILexicalBlock(scope: !1350, file: !6, line: 1325, column: 5)
!1369 = !DILocation(line: 1325, column: 7, scope: !1368)
!1370 = !DILocation(line: 1325, column: 5, scope: !1350)
!1371 = !DILocation(line: 1325, column: 14, scope: !1372)
!1372 = distinct !DILexicalBlock(scope: !1368, file: !6, line: 1325, column: 13)
!1373 = !DILocation(line: 1326, column: 9, scope: !1350)
!1374 = !DILocation(line: 1326, column: 11, scope: !1350)
!1375 = !DILocation(line: 1326, column: 16, scope: !1350)
!1376 = !DILocation(line: 1326, column: 21, scope: !1350)
!1377 = !DILocation(line: 1326, column: 23, scope: !1350)
!1378 = !DILocation(line: 1326, column: 20, scope: !1350)
!1379 = !DILocation(line: 1326, column: 14, scope: !1350)
!1380 = !DILocation(line: 1326, column: 2, scope: !1350)
!1381 = !DILocation(line: 1326, column: 4, scope: !1350)
!1382 = !DILocation(line: 1326, column: 7, scope: !1350)
!1383 = !DILocation(line: 1327, column: 1, scope: !1350)
!1384 = distinct !DISubprogram(name: "gpu_kernel_eight_device", linkageName: "_Z23gpu_kernel_eight_devicePiS_PdS0_S0_", scope: !6, file: !6, line: 1346, type: !1023, scopeLine: 1350, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !7)
!1385 = !DILocalVariable(name: "colidx", arg: 1, scope: !1384, file: !6, line: 1346, type: !82)
!1386 = !DILocation(line: 1346, column: 45, scope: !1384)
!1387 = !DILocalVariable(name: "rowstr", arg: 2, scope: !1384, file: !6, line: 1347, type: !82)
!1388 = !DILocation(line: 1347, column: 7, scope: !1384)
!1389 = !DILocalVariable(name: "a", arg: 3, scope: !1384, file: !6, line: 1348, type: !9)
!1390 = !DILocation(line: 1348, column: 10, scope: !1384)
!1391 = !DILocalVariable(name: "r", arg: 4, scope: !1384, file: !6, line: 1349, type: !9)
!1392 = !DILocation(line: 1349, column: 10, scope: !1384)
!1393 = !DILocalVariable(name: "z", arg: 5, scope: !1384, file: !6, line: 1350, type: !9)
!1394 = !DILocation(line: 1350, column: 11, scope: !1384)
!1395 = !DILocalVariable(name: "share_data", scope: !1384, file: !6, line: 1351, type: !9)
!1396 = !DILocation(line: 1351, column: 10, scope: !1384)
!1397 = !DILocalVariable(name: "j", scope: !1384, file: !6, line: 1353, type: !11)
!1398 = !DILocation(line: 1353, column: 6, scope: !1384)
!1399 = !DILocation(line: 64, column: 3, scope: !807, inlinedAt: !1400)
!1400 = distinct !DILocation(line: 1353, column: 18, scope: !1384)
!1401 = !DILocation(line: 75, column: 3, scope: !845, inlinedAt: !1402)
!1402 = distinct !DILocation(line: 1353, column: 29, scope: !1384)
!1403 = !DILocation(line: 1353, column: 28, scope: !1384)
!1404 = !DILocation(line: 53, column: 3, scope: !891, inlinedAt: !1405)
!1405 = distinct !DILocation(line: 1353, column: 40, scope: !1384)
!1406 = !DILocation(line: 1353, column: 39, scope: !1384)
!1407 = !DILocation(line: 75, column: 3, scope: !845, inlinedAt: !1408)
!1408 = distinct !DILocation(line: 1353, column: 55, scope: !1384)
!1409 = !DILocation(line: 1353, column: 53, scope: !1384)
!1410 = !DILocalVariable(name: "local_id", scope: !1384, file: !6, line: 1354, type: !11)
!1411 = !DILocation(line: 1354, column: 6, scope: !1384)
!1412 = !DILocation(line: 53, column: 3, scope: !891, inlinedAt: !1413)
!1413 = distinct !DILocation(line: 1354, column: 17, scope: !1384)
!1414 = !DILocalVariable(name: "begin", scope: !1384, file: !6, line: 1356, type: !11)
!1415 = !DILocation(line: 1356, column: 6, scope: !1384)
!1416 = !DILocation(line: 1356, column: 14, scope: !1384)
!1417 = !DILocation(line: 1356, column: 21, scope: !1384)
!1418 = !DILocalVariable(name: "end", scope: !1384, file: !6, line: 1357, type: !11)
!1419 = !DILocation(line: 1357, column: 6, scope: !1384)
!1420 = !DILocation(line: 1357, column: 12, scope: !1384)
!1421 = !DILocation(line: 1357, column: 19, scope: !1384)
!1422 = !DILocation(line: 1357, column: 20, scope: !1384)
!1423 = !DILocalVariable(name: "sum", scope: !1384, file: !6, line: 1358, type: !10)
!1424 = !DILocation(line: 1358, column: 9, scope: !1384)
!1425 = !DILocalVariable(name: "k", scope: !1426, file: !6, line: 1359, type: !11)
!1426 = distinct !DILexicalBlock(scope: !1384, file: !6, line: 1359, column: 2)
!1427 = !DILocation(line: 1359, column: 10, scope: !1426)
!1428 = !DILocation(line: 1359, column: 12, scope: !1426)
!1429 = !DILocation(line: 1359, column: 18, scope: !1426)
!1430 = !DILocation(line: 1359, column: 17, scope: !1426)
!1431 = !DILocation(line: 1359, column: 6, scope: !1426)
!1432 = !DILocation(line: 1359, column: 28, scope: !1433)
!1433 = distinct !DILexicalBlock(scope: !1426, file: !6, line: 1359, column: 2)
!1434 = !DILocation(line: 1359, column: 30, scope: !1433)
!1435 = !DILocation(line: 1359, column: 29, scope: !1433)
!1436 = !DILocation(line: 1359, column: 2, scope: !1426)
!1437 = !DILocation(line: 1360, column: 9, scope: !1438)
!1438 = distinct !DILexicalBlock(scope: !1433, file: !6, line: 1359, column: 49)
!1439 = !DILocation(line: 1360, column: 15, scope: !1438)
!1440 = !DILocation(line: 1360, column: 17, scope: !1438)
!1441 = !DILocation(line: 1360, column: 20, scope: !1438)
!1442 = !DILocation(line: 1360, column: 22, scope: !1438)
!1443 = !DILocation(line: 1360, column: 29, scope: !1438)
!1444 = !DILocation(line: 1360, column: 19, scope: !1438)
!1445 = !DILocation(line: 1360, column: 13, scope: !1438)
!1446 = !DILocation(line: 1360, column: 7, scope: !1438)
!1447 = !DILocation(line: 1361, column: 2, scope: !1438)
!1448 = !DILocation(line: 75, column: 3, scope: !845, inlinedAt: !1449)
!1449 = distinct !DILocation(line: 1359, column: 38, scope: !1433)
!1450 = !DILocation(line: 1359, column: 36, scope: !1433)
!1451 = !DILocation(line: 1359, column: 2, scope: !1433)
!1452 = distinct !{!1452, !1436, !1453}
!1453 = !DILocation(line: 1361, column: 2, scope: !1426)
!1454 = !DILocation(line: 1362, column: 25, scope: !1384)
!1455 = !DILocation(line: 1362, column: 2, scope: !1384)
!1456 = !DILocation(line: 1362, column: 13, scope: !1384)
!1457 = !DILocation(line: 1362, column: 23, scope: !1384)
!1458 = !DILocation(line: 1372, column: 2, scope: !1384)
!1459 = !DILocation(line: 1373, column: 5, scope: !1460)
!1460 = distinct !DILexicalBlock(scope: !1384, file: !6, line: 1373, column: 5)
!1461 = !DILocation(line: 1373, column: 13, scope: !1460)
!1462 = !DILocation(line: 1373, column: 5, scope: !1384)
!1463 = !DILocalVariable(name: "i", scope: !1464, file: !6, line: 1374, type: !11)
!1464 = distinct !DILexicalBlock(scope: !1465, file: !6, line: 1374, column: 3)
!1465 = distinct !DILexicalBlock(scope: !1460, file: !6, line: 1373, column: 17)
!1466 = !DILocation(line: 1374, column: 11, scope: !1464)
!1467 = !DILocation(line: 1374, column: 7, scope: !1464)
!1468 = !DILocation(line: 1374, column: 16, scope: !1469)
!1469 = distinct !DILexicalBlock(scope: !1464, file: !6, line: 1374, column: 3)
!1470 = !DILocation(line: 75, column: 3, scope: !845, inlinedAt: !1471)
!1471 = distinct !DILocation(line: 1374, column: 18, scope: !1469)
!1472 = !DILocation(line: 1374, column: 17, scope: !1469)
!1473 = !DILocation(line: 1374, column: 3, scope: !1464)
!1474 = !DILocation(line: 1375, column: 19, scope: !1475)
!1475 = distinct !DILexicalBlock(scope: !1469, file: !6, line: 1374, column: 34)
!1476 = !DILocation(line: 1375, column: 30, scope: !1475)
!1477 = !DILocation(line: 1375, column: 4, scope: !1475)
!1478 = !DILocation(line: 1375, column: 17, scope: !1475)
!1479 = !DILocation(line: 1376, column: 3, scope: !1475)
!1480 = !DILocation(line: 1374, column: 31, scope: !1469)
!1481 = !DILocation(line: 1374, column: 3, scope: !1469)
!1482 = distinct !{!1482, !1473, !1483}
!1483 = !DILocation(line: 1376, column: 3, scope: !1464)
!1484 = !DILocation(line: 1377, column: 8, scope: !1465)
!1485 = !DILocation(line: 1377, column: 3, scope: !1465)
!1486 = !DILocation(line: 1377, column: 5, scope: !1465)
!1487 = !DILocation(line: 1377, column: 7, scope: !1465)
!1488 = !DILocation(line: 1378, column: 2, scope: !1465)
!1489 = !DILocation(line: 1379, column: 1, scope: !1384)
!1490 = distinct !DISubprogram(name: "gpu_kernel_nine_device", linkageName: "_Z22gpu_kernel_nine_devicePdS_S_S_", scope: !6, file: !6, line: 1401, type: !1131, scopeLine: 1401, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !7)
!1491 = !DILocalVariable(name: "r", arg: 1, scope: !1490, file: !6, line: 1401, type: !9)
!1492 = !DILocation(line: 1401, column: 47, scope: !1490)
!1493 = !DILocalVariable(name: "x", arg: 2, scope: !1490, file: !6, line: 1401, type: !9)
!1494 = !DILocation(line: 1401, column: 59, scope: !1490)
!1495 = !DILocalVariable(name: "sum", arg: 3, scope: !1490, file: !6, line: 1401, type: !9)
!1496 = !DILocation(line: 1401, column: 72, scope: !1490)
!1497 = !DILocalVariable(name: "global_data", arg: 4, scope: !1490, file: !6, line: 1401, type: !9)
!1498 = !DILocation(line: 1401, column: 84, scope: !1490)
!1499 = !DILocalVariable(name: "share_data", scope: !1490, file: !6, line: 1402, type: !9)
!1500 = !DILocation(line: 1402, column: 10, scope: !1490)
!1501 = !DILocalVariable(name: "thread_id", scope: !1490, file: !6, line: 1404, type: !11)
!1502 = !DILocation(line: 1404, column: 6, scope: !1490)
!1503 = !DILocation(line: 64, column: 3, scope: !807, inlinedAt: !1504)
!1504 = distinct !DILocation(line: 1404, column: 18, scope: !1490)
!1505 = !DILocation(line: 75, column: 3, scope: !845, inlinedAt: !1506)
!1506 = distinct !DILocation(line: 1404, column: 31, scope: !1490)
!1507 = !DILocation(line: 1404, column: 29, scope: !1490)
!1508 = !DILocation(line: 53, column: 3, scope: !891, inlinedAt: !1509)
!1509 = distinct !DILocation(line: 1404, column: 44, scope: !1490)
!1510 = !DILocation(line: 1404, column: 42, scope: !1490)
!1511 = !DILocalVariable(name: "local_id", scope: !1490, file: !6, line: 1405, type: !11)
!1512 = !DILocation(line: 1405, column: 6, scope: !1490)
!1513 = !DILocation(line: 53, column: 3, scope: !891, inlinedAt: !1514)
!1514 = distinct !DILocation(line: 1405, column: 17, scope: !1490)
!1515 = !DILocation(line: 1407, column: 2, scope: !1490)
!1516 = !DILocation(line: 1407, column: 13, scope: !1490)
!1517 = !DILocation(line: 1407, column: 23, scope: !1490)
!1518 = !DILocation(line: 1411, column: 5, scope: !1519)
!1519 = distinct !DILexicalBlock(scope: !1490, file: !6, line: 1411, column: 5)
!1520 = !DILocation(line: 1411, column: 15, scope: !1519)
!1521 = !DILocation(line: 1411, column: 5, scope: !1490)
!1522 = !DILocation(line: 1412, column: 32, scope: !1523)
!1523 = distinct !DILexicalBlock(scope: !1519, file: !6, line: 1411, column: 20)
!1524 = !DILocation(line: 1412, column: 34, scope: !1523)
!1525 = !DILocation(line: 1412, column: 47, scope: !1523)
!1526 = !DILocation(line: 1412, column: 49, scope: !1523)
!1527 = !DILocation(line: 1412, column: 45, scope: !1523)
!1528 = !DILocation(line: 1412, column: 9, scope: !1523)
!1529 = !DILocation(line: 1412, column: 20, scope: !1523)
!1530 = !DILocation(line: 1412, column: 30, scope: !1523)
!1531 = !DILocation(line: 1413, column: 32, scope: !1523)
!1532 = !DILocation(line: 1413, column: 43, scope: !1523)
!1533 = !DILocation(line: 1413, column: 55, scope: !1523)
!1534 = !DILocation(line: 1413, column: 66, scope: !1523)
!1535 = !DILocation(line: 1413, column: 53, scope: !1523)
!1536 = !DILocation(line: 1413, column: 9, scope: !1523)
!1537 = !DILocation(line: 1413, column: 20, scope: !1523)
!1538 = !DILocation(line: 1413, column: 30, scope: !1523)
!1539 = !DILocation(line: 1414, column: 5, scope: !1523)
!1540 = !DILocation(line: 1424, column: 2, scope: !1490)
!1541 = !DILocation(line: 1425, column: 5, scope: !1542)
!1542 = distinct !DILexicalBlock(scope: !1490, file: !6, line: 1425, column: 5)
!1543 = !DILocation(line: 1425, column: 13, scope: !1542)
!1544 = !DILocation(line: 1425, column: 5, scope: !1490)
!1545 = !DILocalVariable(name: "i", scope: !1546, file: !6, line: 1426, type: !11)
!1546 = distinct !DILexicalBlock(scope: !1547, file: !6, line: 1426, column: 3)
!1547 = distinct !DILexicalBlock(scope: !1542, file: !6, line: 1425, column: 17)
!1548 = !DILocation(line: 1426, column: 11, scope: !1546)
!1549 = !DILocation(line: 1426, column: 7, scope: !1546)
!1550 = !DILocation(line: 1426, column: 16, scope: !1551)
!1551 = distinct !DILexicalBlock(scope: !1546, file: !6, line: 1426, column: 3)
!1552 = !DILocation(line: 75, column: 3, scope: !845, inlinedAt: !1553)
!1553 = distinct !DILocation(line: 1426, column: 18, scope: !1551)
!1554 = !DILocation(line: 1426, column: 17, scope: !1551)
!1555 = !DILocation(line: 1426, column: 3, scope: !1546)
!1556 = !DILocation(line: 1427, column: 19, scope: !1557)
!1557 = distinct !DILexicalBlock(scope: !1551, file: !6, line: 1426, column: 34)
!1558 = !DILocation(line: 1427, column: 30, scope: !1557)
!1559 = !DILocation(line: 1427, column: 4, scope: !1557)
!1560 = !DILocation(line: 1427, column: 17, scope: !1557)
!1561 = !DILocation(line: 1428, column: 3, scope: !1557)
!1562 = !DILocation(line: 1426, column: 31, scope: !1551)
!1563 = !DILocation(line: 1426, column: 3, scope: !1551)
!1564 = distinct !{!1564, !1555, !1565}
!1565 = !DILocation(line: 1428, column: 3, scope: !1546)
!1566 = !DILocation(line: 1429, column: 27, scope: !1547)
!1567 = !DILocation(line: 1429, column: 3, scope: !1547)
!1568 = !DILocation(line: 64, column: 3, scope: !807, inlinedAt: !1569)
!1569 = distinct !DILocation(line: 1429, column: 15, scope: !1547)
!1570 = !DILocation(line: 1429, column: 26, scope: !1547)
!1571 = !DILocation(line: 1430, column: 2, scope: !1547)
!1572 = !DILocation(line: 1431, column: 1, scope: !1490)
!1573 = distinct !DISubprogram(name: "gpu_kernel_ten_1", linkageName: "_Z16gpu_kernel_ten_1PdS_S_", scope: !6, file: !6, line: 1454, type: !946, scopeLine: 1456, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !7)
!1574 = !DILocalVariable(name: "norm_temp", arg: 1, scope: !1573, file: !6, line: 1454, type: !9)
!1575 = !DILocation(line: 1454, column: 42, scope: !1573)
!1576 = !DILocalVariable(name: "x", arg: 2, scope: !1573, file: !6, line: 1455, type: !9)
!1577 = !DILocation(line: 1455, column: 10, scope: !1573)
!1578 = !DILocalVariable(name: "z", arg: 3, scope: !1573, file: !6, line: 1456, type: !9)
!1579 = !DILocation(line: 1456, column: 10, scope: !1573)
!1580 = !DILocalVariable(name: "share_data", scope: !1573, file: !6, line: 1457, type: !9)
!1581 = !DILocation(line: 1457, column: 10, scope: !1573)
!1582 = !DILocalVariable(name: "thread_id", scope: !1573, file: !6, line: 1459, type: !11)
!1583 = !DILocation(line: 1459, column: 6, scope: !1573)
!1584 = !DILocation(line: 64, column: 3, scope: !807, inlinedAt: !1585)
!1585 = distinct !DILocation(line: 1459, column: 18, scope: !1573)
!1586 = !DILocation(line: 75, column: 3, scope: !845, inlinedAt: !1587)
!1587 = distinct !DILocation(line: 1459, column: 31, scope: !1573)
!1588 = !DILocation(line: 1459, column: 29, scope: !1573)
!1589 = !DILocation(line: 53, column: 3, scope: !891, inlinedAt: !1590)
!1590 = distinct !DILocation(line: 1459, column: 44, scope: !1573)
!1591 = !DILocation(line: 1459, column: 42, scope: !1573)
!1592 = !DILocalVariable(name: "local_id", scope: !1573, file: !6, line: 1460, type: !11)
!1593 = !DILocation(line: 1460, column: 6, scope: !1573)
!1594 = !DILocation(line: 53, column: 3, scope: !891, inlinedAt: !1595)
!1595 = distinct !DILocation(line: 1460, column: 17, scope: !1573)
!1596 = !DILocation(line: 1462, column: 2, scope: !1573)
!1597 = !DILocation(line: 53, column: 3, scope: !891, inlinedAt: !1598)
!1598 = distinct !DILocation(line: 1462, column: 13, scope: !1573)
!1599 = !DILocation(line: 1462, column: 26, scope: !1573)
!1600 = !DILocation(line: 1466, column: 5, scope: !1601)
!1601 = distinct !DILexicalBlock(scope: !1573, file: !6, line: 1466, column: 5)
!1602 = !DILocation(line: 1466, column: 15, scope: !1601)
!1603 = !DILocation(line: 1466, column: 5, scope: !1573)
!1604 = !DILocation(line: 1467, column: 35, scope: !1605)
!1605 = distinct !DILexicalBlock(scope: !1601, file: !6, line: 1466, column: 20)
!1606 = !DILocation(line: 1467, column: 37, scope: !1605)
!1607 = !DILocation(line: 1467, column: 48, scope: !1605)
!1608 = !DILocation(line: 1467, column: 50, scope: !1605)
!1609 = !DILocation(line: 1467, column: 47, scope: !1605)
!1610 = !DILocation(line: 1467, column: 9, scope: !1605)
!1611 = !DILocation(line: 53, column: 3, scope: !891, inlinedAt: !1612)
!1612 = distinct !DILocation(line: 1467, column: 20, scope: !1605)
!1613 = !DILocation(line: 1467, column: 33, scope: !1605)
!1614 = !DILocation(line: 1468, column: 5, scope: !1605)
!1615 = !DILocation(line: 1478, column: 2, scope: !1573)
!1616 = !DILocation(line: 1479, column: 5, scope: !1617)
!1617 = distinct !DILexicalBlock(scope: !1573, file: !6, line: 1479, column: 5)
!1618 = !DILocation(line: 1479, column: 13, scope: !1617)
!1619 = !DILocation(line: 1479, column: 5, scope: !1573)
!1620 = !DILocalVariable(name: "i", scope: !1621, file: !6, line: 1480, type: !11)
!1621 = distinct !DILexicalBlock(scope: !1622, file: !6, line: 1480, column: 3)
!1622 = distinct !DILexicalBlock(scope: !1617, file: !6, line: 1479, column: 17)
!1623 = !DILocation(line: 1480, column: 11, scope: !1621)
!1624 = !DILocation(line: 1480, column: 7, scope: !1621)
!1625 = !DILocation(line: 1480, column: 16, scope: !1626)
!1626 = distinct !DILexicalBlock(scope: !1621, file: !6, line: 1480, column: 3)
!1627 = !DILocation(line: 75, column: 3, scope: !845, inlinedAt: !1628)
!1628 = distinct !DILocation(line: 1480, column: 18, scope: !1626)
!1629 = !DILocation(line: 1480, column: 17, scope: !1626)
!1630 = !DILocation(line: 1480, column: 3, scope: !1621)
!1631 = !DILocation(line: 1481, column: 19, scope: !1632)
!1632 = distinct !DILexicalBlock(scope: !1626, file: !6, line: 1480, column: 34)
!1633 = !DILocation(line: 1481, column: 30, scope: !1632)
!1634 = !DILocation(line: 1481, column: 4, scope: !1632)
!1635 = !DILocation(line: 1481, column: 17, scope: !1632)
!1636 = !DILocation(line: 1482, column: 3, scope: !1632)
!1637 = !DILocation(line: 1480, column: 31, scope: !1626)
!1638 = !DILocation(line: 1480, column: 3, scope: !1626)
!1639 = distinct !{!1639, !1630, !1640}
!1640 = !DILocation(line: 1482, column: 3, scope: !1621)
!1641 = !DILocation(line: 1483, column: 25, scope: !1622)
!1642 = !DILocation(line: 1483, column: 3, scope: !1622)
!1643 = !DILocation(line: 64, column: 3, scope: !807, inlinedAt: !1644)
!1644 = distinct !DILocation(line: 1483, column: 13, scope: !1622)
!1645 = !DILocation(line: 1483, column: 24, scope: !1622)
!1646 = !DILocation(line: 1484, column: 2, scope: !1622)
!1647 = !DILocation(line: 1485, column: 1, scope: !1573)
!1648 = distinct !DISubprogram(name: "gpu_kernel_ten_2", linkageName: "_Z16gpu_kernel_ten_2PdS_S_", scope: !6, file: !6, line: 1487, type: !946, scopeLine: 1489, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !7)
!1649 = !DILocalVariable(name: "norm_temp", arg: 1, scope: !1648, file: !6, line: 1487, type: !9)
!1650 = !DILocation(line: 1487, column: 42, scope: !1648)
!1651 = !DILocalVariable(name: "x", arg: 2, scope: !1648, file: !6, line: 1488, type: !9)
!1652 = !DILocation(line: 1488, column: 10, scope: !1648)
!1653 = !DILocalVariable(name: "z", arg: 3, scope: !1648, file: !6, line: 1489, type: !9)
!1654 = !DILocation(line: 1489, column: 10, scope: !1648)
!1655 = !DILocalVariable(name: "share_data", scope: !1648, file: !6, line: 1490, type: !9)
!1656 = !DILocation(line: 1490, column: 10, scope: !1648)
!1657 = !DILocalVariable(name: "thread_id", scope: !1648, file: !6, line: 1492, type: !11)
!1658 = !DILocation(line: 1492, column: 6, scope: !1648)
!1659 = !DILocation(line: 64, column: 3, scope: !807, inlinedAt: !1660)
!1660 = distinct !DILocation(line: 1492, column: 18, scope: !1648)
!1661 = !DILocation(line: 75, column: 3, scope: !845, inlinedAt: !1662)
!1662 = distinct !DILocation(line: 1492, column: 31, scope: !1648)
!1663 = !DILocation(line: 1492, column: 29, scope: !1648)
!1664 = !DILocation(line: 53, column: 3, scope: !891, inlinedAt: !1665)
!1665 = distinct !DILocation(line: 1492, column: 44, scope: !1648)
!1666 = !DILocation(line: 1492, column: 42, scope: !1648)
!1667 = !DILocalVariable(name: "local_id", scope: !1648, file: !6, line: 1493, type: !11)
!1668 = !DILocation(line: 1493, column: 6, scope: !1648)
!1669 = !DILocation(line: 53, column: 3, scope: !891, inlinedAt: !1670)
!1670 = distinct !DILocation(line: 1493, column: 17, scope: !1648)
!1671 = !DILocation(line: 1495, column: 2, scope: !1648)
!1672 = !DILocation(line: 53, column: 3, scope: !891, inlinedAt: !1673)
!1673 = distinct !DILocation(line: 1495, column: 13, scope: !1648)
!1674 = !DILocation(line: 1495, column: 26, scope: !1648)
!1675 = !DILocation(line: 1499, column: 5, scope: !1676)
!1676 = distinct !DILexicalBlock(scope: !1648, file: !6, line: 1499, column: 5)
!1677 = !DILocation(line: 1499, column: 15, scope: !1676)
!1678 = !DILocation(line: 1499, column: 5, scope: !1648)
!1679 = !DILocation(line: 1500, column: 35, scope: !1680)
!1680 = distinct !DILexicalBlock(scope: !1676, file: !6, line: 1499, column: 20)
!1681 = !DILocation(line: 1500, column: 37, scope: !1680)
!1682 = !DILocation(line: 1500, column: 48, scope: !1680)
!1683 = !DILocation(line: 1500, column: 50, scope: !1680)
!1684 = !DILocation(line: 1500, column: 47, scope: !1680)
!1685 = !DILocation(line: 1500, column: 9, scope: !1680)
!1686 = !DILocation(line: 53, column: 3, scope: !891, inlinedAt: !1687)
!1687 = distinct !DILocation(line: 1500, column: 20, scope: !1680)
!1688 = !DILocation(line: 1500, column: 33, scope: !1680)
!1689 = !DILocation(line: 1501, column: 5, scope: !1680)
!1690 = !DILocation(line: 1511, column: 2, scope: !1648)
!1691 = !DILocation(line: 1512, column: 5, scope: !1692)
!1692 = distinct !DILexicalBlock(scope: !1648, file: !6, line: 1512, column: 5)
!1693 = !DILocation(line: 1512, column: 13, scope: !1692)
!1694 = !DILocation(line: 1512, column: 5, scope: !1648)
!1695 = !DILocalVariable(name: "i", scope: !1696, file: !6, line: 1513, type: !11)
!1696 = distinct !DILexicalBlock(scope: !1697, file: !6, line: 1513, column: 3)
!1697 = distinct !DILexicalBlock(scope: !1692, file: !6, line: 1512, column: 17)
!1698 = !DILocation(line: 1513, column: 11, scope: !1696)
!1699 = !DILocation(line: 1513, column: 7, scope: !1696)
!1700 = !DILocation(line: 1513, column: 16, scope: !1701)
!1701 = distinct !DILexicalBlock(scope: !1696, file: !6, line: 1513, column: 3)
!1702 = !DILocation(line: 75, column: 3, scope: !845, inlinedAt: !1703)
!1703 = distinct !DILocation(line: 1513, column: 18, scope: !1701)
!1704 = !DILocation(line: 1513, column: 17, scope: !1701)
!1705 = !DILocation(line: 1513, column: 3, scope: !1696)
!1706 = !DILocation(line: 1514, column: 19, scope: !1707)
!1707 = distinct !DILexicalBlock(scope: !1701, file: !6, line: 1513, column: 34)
!1708 = !DILocation(line: 1514, column: 30, scope: !1707)
!1709 = !DILocation(line: 1514, column: 4, scope: !1707)
!1710 = !DILocation(line: 1514, column: 17, scope: !1707)
!1711 = !DILocation(line: 1515, column: 3, scope: !1707)
!1712 = !DILocation(line: 1513, column: 31, scope: !1701)
!1713 = !DILocation(line: 1513, column: 3, scope: !1701)
!1714 = distinct !{!1714, !1705, !1715}
!1715 = !DILocation(line: 1515, column: 3, scope: !1696)
!1716 = !DILocation(line: 1516, column: 25, scope: !1697)
!1717 = !DILocation(line: 1516, column: 3, scope: !1697)
!1718 = !DILocation(line: 64, column: 3, scope: !807, inlinedAt: !1719)
!1719 = distinct !DILocation(line: 1516, column: 13, scope: !1697)
!1720 = !DILocation(line: 1516, column: 24, scope: !1697)
!1721 = !DILocation(line: 1517, column: 2, scope: !1697)
!1722 = !DILocation(line: 1518, column: 1, scope: !1648)
!1723 = distinct !DISubprogram(name: "gpu_kernel_eleven_device", linkageName: "_Z24gpu_kernel_eleven_devicedPdS_", scope: !6, file: !6, line: 1534, type: !1212, scopeLine: 1534, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !7)
!1724 = !DILocalVariable(name: "norm_temp2", arg: 1, scope: !1723, file: !6, line: 1534, type: !10)
!1725 = !DILocation(line: 1534, column: 49, scope: !1723)
!1726 = !DILocalVariable(name: "x", arg: 2, scope: !1723, file: !6, line: 1534, type: !9)
!1727 = !DILocation(line: 1534, column: 68, scope: !1723)
!1728 = !DILocalVariable(name: "z", arg: 3, scope: !1723, file: !6, line: 1534, type: !9)
!1729 = !DILocation(line: 1534, column: 80, scope: !1723)
!1730 = !DILocalVariable(name: "j", scope: !1723, file: !6, line: 1535, type: !11)
!1731 = !DILocation(line: 1535, column: 6, scope: !1723)
!1732 = !DILocation(line: 64, column: 3, scope: !807, inlinedAt: !1733)
!1733 = distinct !DILocation(line: 1535, column: 10, scope: !1723)
!1734 = !DILocation(line: 75, column: 3, scope: !845, inlinedAt: !1735)
!1735 = distinct !DILocation(line: 1535, column: 23, scope: !1723)
!1736 = !DILocation(line: 1535, column: 21, scope: !1723)
!1737 = !DILocation(line: 53, column: 3, scope: !891, inlinedAt: !1738)
!1738 = distinct !DILocation(line: 1535, column: 36, scope: !1723)
!1739 = !DILocation(line: 1535, column: 34, scope: !1723)
!1740 = !DILocation(line: 1536, column: 5, scope: !1741)
!1741 = distinct !DILexicalBlock(scope: !1723, file: !6, line: 1536, column: 5)
!1742 = !DILocation(line: 1536, column: 7, scope: !1741)
!1743 = !DILocation(line: 1536, column: 5, scope: !1723)
!1744 = !DILocation(line: 1536, column: 14, scope: !1745)
!1745 = distinct !DILexicalBlock(scope: !1741, file: !6, line: 1536, column: 13)
!1746 = !DILocation(line: 1537, column: 7, scope: !1723)
!1747 = !DILocation(line: 1537, column: 18, scope: !1723)
!1748 = !DILocation(line: 1537, column: 20, scope: !1723)
!1749 = !DILocation(line: 1537, column: 17, scope: !1723)
!1750 = !DILocation(line: 1537, column: 2, scope: !1723)
!1751 = !DILocation(line: 1537, column: 4, scope: !1723)
!1752 = !DILocation(line: 1537, column: 6, scope: !1723)
!1753 = !DILocation(line: 1538, column: 1, scope: !1723)
