; ModuleID = 'ft.cu'
source_filename = "ft.cu"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

%struct.__cuda_builtin_blockIdx_t = type { i8 }
%struct.__cuda_builtin_blockDim_t = type { i8 }
%struct.__cuda_builtin_threadIdx_t = type { i8 }
%struct.dcomplex = type { double, double }

@blockIdx = extern_weak dso_local addrspace(1) global %struct.__cuda_builtin_blockIdx_t, align 1
@blockDim = extern_weak dso_local addrspace(1) global %struct.__cuda_builtin_blockDim_t, align 1
@threadIdx = extern_weak dso_local addrspace(1) global %struct.__cuda_builtin_threadIdx_t, align 1
@extern_share_data = external dso_local addrspace(3) global [0 x double], align 8
@"$str" = private addrspace(1) constant [11 x i8] c"__CUDA_FTZ\00"

; Function Attrs: convergent noinline nounwind
define dso_local void @_Z19cffts1_gpu_kernel_1P8dcomplexS0_(%struct.dcomplex* %x_in, %struct.dcomplex* %y0) #0 !dbg !802 {
entry:
  %x_in.addr = alloca %struct.dcomplex*, align 8
  %y0.addr = alloca %struct.dcomplex*, align 8
  %x_y_z = alloca i32, align 4
  %x = alloca i32, align 4
  %y = alloca i32, align 4
  %z = alloca i32, align 4
  store %struct.dcomplex* %x_in, %struct.dcomplex** %x_in.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %x_in.addr, metadata !805, metadata !DIExpression()), !dbg !806
  store %struct.dcomplex* %y0, %struct.dcomplex** %y0.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %y0.addr, metadata !807, metadata !DIExpression()), !dbg !808
  call void @llvm.dbg.declare(metadata i32* %x_y_z, metadata !809, metadata !DIExpression()), !dbg !810
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #8, !dbg !811, !range !848
  %1 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #8, !dbg !849, !range !893
  %mul = mul i32 %0, %1, !dbg !894
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #8, !dbg !895, !range !923
  %add = add i32 %mul, %2, !dbg !924
  store i32 %add, i32* %x_y_z, align 4, !dbg !810
  %3 = load i32, i32* %x_y_z, align 4, !dbg !925
  %cmp = icmp sge i32 %3, 8388608, !dbg !927
  br i1 %cmp, label %if.then, label %if.end, !dbg !928

if.then:                                          ; preds = %entry
  br label %return, !dbg !929

if.end:                                           ; preds = %entry
  call void @llvm.dbg.declare(metadata i32* %x, metadata !931, metadata !DIExpression()), !dbg !932
  %4 = load i32, i32* %x_y_z, align 4, !dbg !933
  %rem = srem i32 %4, 256, !dbg !934
  store i32 %rem, i32* %x, align 4, !dbg !932
  call void @llvm.dbg.declare(metadata i32* %y, metadata !935, metadata !DIExpression()), !dbg !936
  %5 = load i32, i32* %x_y_z, align 4, !dbg !937
  %div = sdiv i32 %5, 256, !dbg !938
  %rem3 = srem i32 %div, 256, !dbg !939
  store i32 %rem3, i32* %y, align 4, !dbg !936
  call void @llvm.dbg.declare(metadata i32* %z, metadata !940, metadata !DIExpression()), !dbg !941
  %6 = load i32, i32* %x_y_z, align 4, !dbg !942
  %div4 = sdiv i32 %6, 65536, !dbg !943
  store i32 %div4, i32* %z, align 4, !dbg !941
  %7 = load %struct.dcomplex*, %struct.dcomplex** %x_in.addr, align 8, !dbg !944
  %8 = load i32, i32* %x_y_z, align 4, !dbg !945
  %idxprom = sext i32 %8 to i64, !dbg !944
  %arrayidx = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %7, i64 %idxprom, !dbg !944
  %real = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx, i32 0, i32 0, !dbg !946
  %9 = load double, double* %real, align 8, !dbg !946
  %10 = load %struct.dcomplex*, %struct.dcomplex** %y0.addr, align 8, !dbg !947
  %11 = load i32, i32* %y, align 4, !dbg !948
  %12 = load i32, i32* %x, align 4, !dbg !949
  %mul5 = mul nsw i32 %12, 256, !dbg !950
  %add6 = add nsw i32 %11, %mul5, !dbg !951
  %13 = load i32, i32* %z, align 4, !dbg !952
  %mul7 = mul nsw i32 %13, 256, !dbg !953
  %mul8 = mul nsw i32 %mul7, 256, !dbg !954
  %add9 = add nsw i32 %add6, %mul8, !dbg !955
  %idxprom10 = sext i32 %add9 to i64, !dbg !947
  %arrayidx11 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %10, i64 %idxprom10, !dbg !947
  %real12 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx11, i32 0, i32 0, !dbg !956
  store double %9, double* %real12, align 8, !dbg !957
  %14 = load %struct.dcomplex*, %struct.dcomplex** %x_in.addr, align 8, !dbg !958
  %15 = load i32, i32* %x_y_z, align 4, !dbg !959
  %idxprom13 = sext i32 %15 to i64, !dbg !958
  %arrayidx14 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %14, i64 %idxprom13, !dbg !958
  %imag = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx14, i32 0, i32 1, !dbg !960
  %16 = load double, double* %imag, align 8, !dbg !960
  %17 = load %struct.dcomplex*, %struct.dcomplex** %y0.addr, align 8, !dbg !961
  %18 = load i32, i32* %y, align 4, !dbg !962
  %19 = load i32, i32* %x, align 4, !dbg !963
  %mul15 = mul nsw i32 %19, 256, !dbg !964
  %add16 = add nsw i32 %18, %mul15, !dbg !965
  %20 = load i32, i32* %z, align 4, !dbg !966
  %mul17 = mul nsw i32 %20, 256, !dbg !967
  %mul18 = mul nsw i32 %mul17, 256, !dbg !968
  %add19 = add nsw i32 %add16, %mul18, !dbg !969
  %idxprom20 = sext i32 %add19 to i64, !dbg !961
  %arrayidx21 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %17, i64 %idxprom20, !dbg !961
  %imag22 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx21, i32 0, i32 1, !dbg !970
  store double %16, double* %imag22, align 8, !dbg !971
  br label %return, !dbg !972

return:                                           ; preds = %if.end, %if.then
  ret void, !dbg !972
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: convergent noinline nounwind
define dso_local void @_Z19cffts1_gpu_kernel_2iP8dcomplexS0_S0_(i32 %is, %struct.dcomplex* %gty1, %struct.dcomplex* %gty2, %struct.dcomplex* %u_device) #0 !dbg !973 {
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
  call void @llvm.dbg.declare(metadata i32* %is.addr, metadata !977, metadata !DIExpression()), !dbg !978
  store %struct.dcomplex* %gty1, %struct.dcomplex** %gty1.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %gty1.addr, metadata !979, metadata !DIExpression()), !dbg !980
  store %struct.dcomplex* %gty2, %struct.dcomplex** %gty2.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %gty2.addr, metadata !981, metadata !DIExpression()), !dbg !982
  store %struct.dcomplex* %u_device, %struct.dcomplex** %u_device.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %u_device.addr, metadata !983, metadata !DIExpression()), !dbg !984
  call void @llvm.dbg.declare(metadata i32* %y_z, metadata !985, metadata !DIExpression()), !dbg !986
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #8, !dbg !987, !range !848
  %1 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #8, !dbg !989, !range !893
  %mul = mul i32 %0, %1, !dbg !991
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #8, !dbg !992, !range !923
  %add = add i32 %mul, %2, !dbg !994
  store i32 %add, i32* %y_z, align 4, !dbg !986
  %3 = load i32, i32* %y_z, align 4, !dbg !995
  %cmp = icmp sge i32 %3, 32768, !dbg !997
  br i1 %cmp, label %if.then, label %if.end, !dbg !998

if.then:                                          ; preds = %entry
  br label %for.end271, !dbg !999

if.end:                                           ; preds = %entry
  call void @llvm.dbg.declare(metadata i32* %j, metadata !1001, metadata !DIExpression()), !dbg !1002
  call void @llvm.dbg.declare(metadata i32* %k, metadata !1003, metadata !DIExpression()), !dbg !1004
  call void @llvm.dbg.declare(metadata i32* %l, metadata !1005, metadata !DIExpression()), !dbg !1006
  call void @llvm.dbg.declare(metadata i32* %j1, metadata !1007, metadata !DIExpression()), !dbg !1008
  call void @llvm.dbg.declare(metadata i32* %i1, metadata !1009, metadata !DIExpression()), !dbg !1010
  call void @llvm.dbg.declare(metadata i32* %k1, metadata !1011, metadata !DIExpression()), !dbg !1012
  call void @llvm.dbg.declare(metadata i32* %n1, metadata !1013, metadata !DIExpression()), !dbg !1014
  call void @llvm.dbg.declare(metadata i32* %li, metadata !1015, metadata !DIExpression()), !dbg !1016
  call void @llvm.dbg.declare(metadata i32* %lj, metadata !1017, metadata !DIExpression()), !dbg !1018
  call void @llvm.dbg.declare(metadata i32* %lk, metadata !1019, metadata !DIExpression()), !dbg !1020
  call void @llvm.dbg.declare(metadata i32* %ku, metadata !1021, metadata !DIExpression()), !dbg !1022
  call void @llvm.dbg.declare(metadata i32* %i11, metadata !1023, metadata !DIExpression()), !dbg !1024
  call void @llvm.dbg.declare(metadata i32* %i12, metadata !1025, metadata !DIExpression()), !dbg !1026
  call void @llvm.dbg.declare(metadata i32* %i21, metadata !1027, metadata !DIExpression()), !dbg !1028
  call void @llvm.dbg.declare(metadata i32* %i22, metadata !1029, metadata !DIExpression()), !dbg !1030
  %4 = load i32, i32* %y_z, align 4, !dbg !1031
  %rem = srem i32 %4, 256, !dbg !1032
  store i32 %rem, i32* %j, align 4, !dbg !1033
  %5 = load i32, i32* %y_z, align 4, !dbg !1034
  %div = sdiv i32 %5, 256, !dbg !1035
  %rem3 = srem i32 %div, 128, !dbg !1036
  store i32 %rem3, i32* %k, align 4, !dbg !1037
  call void @llvm.dbg.declare(metadata i32* %logd1, metadata !1038, metadata !DIExpression()), !dbg !1039
  %call4 = call i32 @_Z12ilog2_devicei(i32 256) #3, !dbg !1040
  store i32 %call4, i32* %logd1, align 4, !dbg !1039
  call void @llvm.dbg.declare(metadata double* %uu1_real, metadata !1041, metadata !DIExpression()), !dbg !1042
  call void @llvm.dbg.declare(metadata double* %x11_real, metadata !1043, metadata !DIExpression()), !dbg !1044
  call void @llvm.dbg.declare(metadata double* %x21_real, metadata !1045, metadata !DIExpression()), !dbg !1046
  call void @llvm.dbg.declare(metadata double* %uu1_imag, metadata !1047, metadata !DIExpression()), !dbg !1048
  call void @llvm.dbg.declare(metadata double* %x11_imag, metadata !1049, metadata !DIExpression()), !dbg !1050
  call void @llvm.dbg.declare(metadata double* %x21_imag, metadata !1051, metadata !DIExpression()), !dbg !1052
  call void @llvm.dbg.declare(metadata double* %uu2_real, metadata !1053, metadata !DIExpression()), !dbg !1054
  call void @llvm.dbg.declare(metadata double* %x12_real, metadata !1055, metadata !DIExpression()), !dbg !1056
  call void @llvm.dbg.declare(metadata double* %x22_real, metadata !1057, metadata !DIExpression()), !dbg !1058
  call void @llvm.dbg.declare(metadata double* %uu2_imag, metadata !1059, metadata !DIExpression()), !dbg !1060
  call void @llvm.dbg.declare(metadata double* %x12_imag, metadata !1061, metadata !DIExpression()), !dbg !1062
  call void @llvm.dbg.declare(metadata double* %x22_imag, metadata !1063, metadata !DIExpression()), !dbg !1064
  call void @llvm.dbg.declare(metadata double* %temp_real, metadata !1065, metadata !DIExpression()), !dbg !1066
  call void @llvm.dbg.declare(metadata double* %temp2_real, metadata !1067, metadata !DIExpression()), !dbg !1068
  call void @llvm.dbg.declare(metadata double* %temp_imag, metadata !1069, metadata !DIExpression()), !dbg !1070
  call void @llvm.dbg.declare(metadata double* %temp2_imag, metadata !1071, metadata !DIExpression()), !dbg !1072
  store i32 1, i32* %l, align 4, !dbg !1073
  br label %for.cond, !dbg !1075

for.cond:                                         ; preds = %for.inc269, %if.end
  %6 = load i32, i32* %l, align 4, !dbg !1076
  %7 = load i32, i32* %logd1, align 4, !dbg !1078
  %cmp5 = icmp sle i32 %6, %7, !dbg !1079
  br i1 %cmp5, label %for.body, label %for.end271, !dbg !1080

for.body:                                         ; preds = %for.cond
  store i32 128, i32* %n1, align 4, !dbg !1081
  %8 = load i32, i32* %l, align 4, !dbg !1083
  %sub = sub nsw i32 %8, 1, !dbg !1084
  %shl = shl i32 1, %sub, !dbg !1085
  store i32 %shl, i32* %lk, align 4, !dbg !1086
  %9 = load i32, i32* %logd1, align 4, !dbg !1087
  %10 = load i32, i32* %l, align 4, !dbg !1088
  %sub6 = sub nsw i32 %9, %10, !dbg !1089
  %shl7 = shl i32 1, %sub6, !dbg !1090
  store i32 %shl7, i32* %li, align 4, !dbg !1091
  %11 = load i32, i32* %lk, align 4, !dbg !1092
  %mul8 = mul nsw i32 2, %11, !dbg !1093
  store i32 %mul8, i32* %lj, align 4, !dbg !1094
  %12 = load i32, i32* %li, align 4, !dbg !1095
  store i32 %12, i32* %ku, align 4, !dbg !1096
  store i32 0, i32* %i1, align 4, !dbg !1097
  br label %for.cond9, !dbg !1099

for.cond9:                                        ; preds = %for.inc108, %for.body
  %13 = load i32, i32* %i1, align 4, !dbg !1100
  %14 = load i32, i32* %li, align 4, !dbg !1102
  %sub10 = sub nsw i32 %14, 1, !dbg !1103
  %cmp11 = icmp sle i32 %13, %sub10, !dbg !1104
  br i1 %cmp11, label %for.body12, label %for.end110, !dbg !1105

for.body12:                                       ; preds = %for.cond9
  store i32 0, i32* %k1, align 4, !dbg !1106
  br label %for.cond13, !dbg !1109

for.cond13:                                       ; preds = %for.inc, %for.body12
  %15 = load i32, i32* %k1, align 4, !dbg !1110
  %16 = load i32, i32* %lk, align 4, !dbg !1112
  %sub14 = sub nsw i32 %16, 1, !dbg !1113
  %cmp15 = icmp sle i32 %15, %sub14, !dbg !1114
  br i1 %cmp15, label %for.body16, label %for.end, !dbg !1115

for.body16:                                       ; preds = %for.cond13
  %17 = load i32, i32* %i1, align 4, !dbg !1116
  %18 = load i32, i32* %lk, align 4, !dbg !1118
  %mul17 = mul nsw i32 %17, %18, !dbg !1119
  store i32 %mul17, i32* %i11, align 4, !dbg !1120
  %19 = load i32, i32* %i11, align 4, !dbg !1121
  %20 = load i32, i32* %n1, align 4, !dbg !1122
  %add18 = add nsw i32 %19, %20, !dbg !1123
  store i32 %add18, i32* %i12, align 4, !dbg !1124
  %21 = load i32, i32* %i1, align 4, !dbg !1125
  %22 = load i32, i32* %lj, align 4, !dbg !1126
  %mul19 = mul nsw i32 %21, %22, !dbg !1127
  store i32 %mul19, i32* %i21, align 4, !dbg !1128
  %23 = load i32, i32* %i21, align 4, !dbg !1129
  %24 = load i32, i32* %lk, align 4, !dbg !1130
  %add20 = add nsw i32 %23, %24, !dbg !1131
  store i32 %add20, i32* %i22, align 4, !dbg !1132
  %25 = load %struct.dcomplex*, %struct.dcomplex** %u_device.addr, align 8, !dbg !1133
  %26 = load i32, i32* %ku, align 4, !dbg !1134
  %27 = load i32, i32* %i1, align 4, !dbg !1135
  %add21 = add nsw i32 %26, %27, !dbg !1136
  %idxprom = sext i32 %add21 to i64, !dbg !1133
  %arrayidx = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %25, i64 %idxprom, !dbg !1133
  %real = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx, i32 0, i32 0, !dbg !1137
  %28 = load double, double* %real, align 8, !dbg !1137
  store double %28, double* %uu1_real, align 8, !dbg !1138
  %29 = load i32, i32* %is.addr, align 4, !dbg !1139
  %conv = sitofp i32 %29 to double, !dbg !1139
  %30 = load %struct.dcomplex*, %struct.dcomplex** %u_device.addr, align 8, !dbg !1140
  %31 = load i32, i32* %ku, align 4, !dbg !1141
  %32 = load i32, i32* %i1, align 4, !dbg !1142
  %add22 = add nsw i32 %31, %32, !dbg !1143
  %idxprom23 = sext i32 %add22 to i64, !dbg !1140
  %arrayidx24 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %30, i64 %idxprom23, !dbg !1140
  %imag = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx24, i32 0, i32 1, !dbg !1144
  %33 = load double, double* %imag, align 8, !dbg !1144
  %mul25 = fmul contract double %conv, %33, !dbg !1145
  store double %mul25, double* %uu1_imag, align 8, !dbg !1146
  %34 = load %struct.dcomplex*, %struct.dcomplex** %gty1.addr, align 8, !dbg !1147
  %35 = load i32, i32* %j, align 4, !dbg !1148
  %36 = load i32, i32* %i11, align 4, !dbg !1149
  %37 = load i32, i32* %k1, align 4, !dbg !1150
  %add26 = add nsw i32 %36, %37, !dbg !1151
  %mul27 = mul nsw i32 %add26, 256, !dbg !1152
  %add28 = add nsw i32 %35, %mul27, !dbg !1153
  %38 = load i32, i32* %k, align 4, !dbg !1154
  %mul29 = mul nsw i32 %38, 256, !dbg !1155
  %mul30 = mul nsw i32 %mul29, 256, !dbg !1156
  %add31 = add nsw i32 %add28, %mul30, !dbg !1157
  %idxprom32 = sext i32 %add31 to i64, !dbg !1147
  %arrayidx33 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %34, i64 %idxprom32, !dbg !1147
  %real34 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx33, i32 0, i32 0, !dbg !1158
  %39 = load double, double* %real34, align 8, !dbg !1158
  store double %39, double* %x11_real, align 8, !dbg !1159
  %40 = load %struct.dcomplex*, %struct.dcomplex** %gty1.addr, align 8, !dbg !1160
  %41 = load i32, i32* %j, align 4, !dbg !1161
  %42 = load i32, i32* %i11, align 4, !dbg !1162
  %43 = load i32, i32* %k1, align 4, !dbg !1163
  %add35 = add nsw i32 %42, %43, !dbg !1164
  %mul36 = mul nsw i32 %add35, 256, !dbg !1165
  %add37 = add nsw i32 %41, %mul36, !dbg !1166
  %44 = load i32, i32* %k, align 4, !dbg !1167
  %mul38 = mul nsw i32 %44, 256, !dbg !1168
  %mul39 = mul nsw i32 %mul38, 256, !dbg !1169
  %add40 = add nsw i32 %add37, %mul39, !dbg !1170
  %idxprom41 = sext i32 %add40 to i64, !dbg !1160
  %arrayidx42 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %40, i64 %idxprom41, !dbg !1160
  %imag43 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx42, i32 0, i32 1, !dbg !1171
  %45 = load double, double* %imag43, align 8, !dbg !1171
  store double %45, double* %x11_imag, align 8, !dbg !1172
  %46 = load %struct.dcomplex*, %struct.dcomplex** %gty1.addr, align 8, !dbg !1173
  %47 = load i32, i32* %j, align 4, !dbg !1174
  %48 = load i32, i32* %i12, align 4, !dbg !1175
  %49 = load i32, i32* %k1, align 4, !dbg !1176
  %add44 = add nsw i32 %48, %49, !dbg !1177
  %mul45 = mul nsw i32 %add44, 256, !dbg !1178
  %add46 = add nsw i32 %47, %mul45, !dbg !1179
  %50 = load i32, i32* %k, align 4, !dbg !1180
  %mul47 = mul nsw i32 %50, 256, !dbg !1181
  %mul48 = mul nsw i32 %mul47, 256, !dbg !1182
  %add49 = add nsw i32 %add46, %mul48, !dbg !1183
  %idxprom50 = sext i32 %add49 to i64, !dbg !1173
  %arrayidx51 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %46, i64 %idxprom50, !dbg !1173
  %real52 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx51, i32 0, i32 0, !dbg !1184
  %51 = load double, double* %real52, align 8, !dbg !1184
  store double %51, double* %x21_real, align 8, !dbg !1185
  %52 = load %struct.dcomplex*, %struct.dcomplex** %gty1.addr, align 8, !dbg !1186
  %53 = load i32, i32* %j, align 4, !dbg !1187
  %54 = load i32, i32* %i12, align 4, !dbg !1188
  %55 = load i32, i32* %k1, align 4, !dbg !1189
  %add53 = add nsw i32 %54, %55, !dbg !1190
  %mul54 = mul nsw i32 %add53, 256, !dbg !1191
  %add55 = add nsw i32 %53, %mul54, !dbg !1192
  %56 = load i32, i32* %k, align 4, !dbg !1193
  %mul56 = mul nsw i32 %56, 256, !dbg !1194
  %mul57 = mul nsw i32 %mul56, 256, !dbg !1195
  %add58 = add nsw i32 %add55, %mul57, !dbg !1196
  %idxprom59 = sext i32 %add58 to i64, !dbg !1186
  %arrayidx60 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %52, i64 %idxprom59, !dbg !1186
  %imag61 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx60, i32 0, i32 1, !dbg !1197
  %57 = load double, double* %imag61, align 8, !dbg !1197
  store double %57, double* %x21_imag, align 8, !dbg !1198
  %58 = load double, double* %x11_real, align 8, !dbg !1199
  %59 = load double, double* %x21_real, align 8, !dbg !1200
  %add62 = fadd contract double %58, %59, !dbg !1201
  %60 = load %struct.dcomplex*, %struct.dcomplex** %gty2.addr, align 8, !dbg !1202
  %61 = load i32, i32* %j, align 4, !dbg !1203
  %62 = load i32, i32* %i21, align 4, !dbg !1204
  %63 = load i32, i32* %k1, align 4, !dbg !1205
  %add63 = add nsw i32 %62, %63, !dbg !1206
  %mul64 = mul nsw i32 %add63, 256, !dbg !1207
  %add65 = add nsw i32 %61, %mul64, !dbg !1208
  %64 = load i32, i32* %k, align 4, !dbg !1209
  %mul66 = mul nsw i32 %64, 256, !dbg !1210
  %mul67 = mul nsw i32 %mul66, 256, !dbg !1211
  %add68 = add nsw i32 %add65, %mul67, !dbg !1212
  %idxprom69 = sext i32 %add68 to i64, !dbg !1202
  %arrayidx70 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %60, i64 %idxprom69, !dbg !1202
  %real71 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx70, i32 0, i32 0, !dbg !1213
  store double %add62, double* %real71, align 8, !dbg !1214
  %65 = load double, double* %x11_imag, align 8, !dbg !1215
  %66 = load double, double* %x21_imag, align 8, !dbg !1216
  %add72 = fadd contract double %65, %66, !dbg !1217
  %67 = load %struct.dcomplex*, %struct.dcomplex** %gty2.addr, align 8, !dbg !1218
  %68 = load i32, i32* %j, align 4, !dbg !1219
  %69 = load i32, i32* %i21, align 4, !dbg !1220
  %70 = load i32, i32* %k1, align 4, !dbg !1221
  %add73 = add nsw i32 %69, %70, !dbg !1222
  %mul74 = mul nsw i32 %add73, 256, !dbg !1223
  %add75 = add nsw i32 %68, %mul74, !dbg !1224
  %71 = load i32, i32* %k, align 4, !dbg !1225
  %mul76 = mul nsw i32 %71, 256, !dbg !1226
  %mul77 = mul nsw i32 %mul76, 256, !dbg !1227
  %add78 = add nsw i32 %add75, %mul77, !dbg !1228
  %idxprom79 = sext i32 %add78 to i64, !dbg !1218
  %arrayidx80 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %67, i64 %idxprom79, !dbg !1218
  %imag81 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx80, i32 0, i32 1, !dbg !1229
  store double %add72, double* %imag81, align 8, !dbg !1230
  %72 = load double, double* %x11_real, align 8, !dbg !1231
  %73 = load double, double* %x21_real, align 8, !dbg !1232
  %sub82 = fsub contract double %72, %73, !dbg !1233
  store double %sub82, double* %temp_real, align 8, !dbg !1234
  %74 = load double, double* %x11_imag, align 8, !dbg !1235
  %75 = load double, double* %x21_imag, align 8, !dbg !1236
  %sub83 = fsub contract double %74, %75, !dbg !1237
  store double %sub83, double* %temp_imag, align 8, !dbg !1238
  %76 = load double, double* %uu1_real, align 8, !dbg !1239
  %77 = load double, double* %temp_real, align 8, !dbg !1240
  %mul84 = fmul contract double %76, %77, !dbg !1241
  %78 = load double, double* %uu1_imag, align 8, !dbg !1242
  %79 = load double, double* %temp_imag, align 8, !dbg !1243
  %mul85 = fmul contract double %78, %79, !dbg !1244
  %sub86 = fsub contract double %mul84, %mul85, !dbg !1245
  %80 = load %struct.dcomplex*, %struct.dcomplex** %gty2.addr, align 8, !dbg !1246
  %81 = load i32, i32* %j, align 4, !dbg !1247
  %82 = load i32, i32* %i22, align 4, !dbg !1248
  %83 = load i32, i32* %k1, align 4, !dbg !1249
  %add87 = add nsw i32 %82, %83, !dbg !1250
  %mul88 = mul nsw i32 %add87, 256, !dbg !1251
  %add89 = add nsw i32 %81, %mul88, !dbg !1252
  %84 = load i32, i32* %k, align 4, !dbg !1253
  %mul90 = mul nsw i32 %84, 256, !dbg !1254
  %mul91 = mul nsw i32 %mul90, 256, !dbg !1255
  %add92 = add nsw i32 %add89, %mul91, !dbg !1256
  %idxprom93 = sext i32 %add92 to i64, !dbg !1246
  %arrayidx94 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %80, i64 %idxprom93, !dbg !1246
  %real95 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx94, i32 0, i32 0, !dbg !1257
  store double %sub86, double* %real95, align 8, !dbg !1258
  %85 = load double, double* %uu1_real, align 8, !dbg !1259
  %86 = load double, double* %temp_imag, align 8, !dbg !1260
  %mul96 = fmul contract double %85, %86, !dbg !1261
  %87 = load double, double* %uu1_imag, align 8, !dbg !1262
  %88 = load double, double* %temp_real, align 8, !dbg !1263
  %mul97 = fmul contract double %87, %88, !dbg !1264
  %add98 = fadd contract double %mul96, %mul97, !dbg !1265
  %89 = load %struct.dcomplex*, %struct.dcomplex** %gty2.addr, align 8, !dbg !1266
  %90 = load i32, i32* %j, align 4, !dbg !1267
  %91 = load i32, i32* %i22, align 4, !dbg !1268
  %92 = load i32, i32* %k1, align 4, !dbg !1269
  %add99 = add nsw i32 %91, %92, !dbg !1270
  %mul100 = mul nsw i32 %add99, 256, !dbg !1271
  %add101 = add nsw i32 %90, %mul100, !dbg !1272
  %93 = load i32, i32* %k, align 4, !dbg !1273
  %mul102 = mul nsw i32 %93, 256, !dbg !1274
  %mul103 = mul nsw i32 %mul102, 256, !dbg !1275
  %add104 = add nsw i32 %add101, %mul103, !dbg !1276
  %idxprom105 = sext i32 %add104 to i64, !dbg !1266
  %arrayidx106 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %89, i64 %idxprom105, !dbg !1266
  %imag107 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx106, i32 0, i32 1, !dbg !1277
  store double %add98, double* %imag107, align 8, !dbg !1278
  br label %for.inc, !dbg !1279

for.inc:                                          ; preds = %for.body16
  %94 = load i32, i32* %k1, align 4, !dbg !1280
  %inc = add nsw i32 %94, 1, !dbg !1280
  store i32 %inc, i32* %k1, align 4, !dbg !1280
  br label %for.cond13, !dbg !1281, !llvm.loop !1282

for.end:                                          ; preds = %for.cond13
  br label %for.inc108, !dbg !1284

for.inc108:                                       ; preds = %for.end
  %95 = load i32, i32* %i1, align 4, !dbg !1285
  %inc109 = add nsw i32 %95, 1, !dbg !1285
  store i32 %inc109, i32* %i1, align 4, !dbg !1285
  br label %for.cond9, !dbg !1286, !llvm.loop !1287

for.end110:                                       ; preds = %for.cond9
  %96 = load i32, i32* %l, align 4, !dbg !1289
  %97 = load i32, i32* %logd1, align 4, !dbg !1291
  %cmp111 = icmp eq i32 %96, %97, !dbg !1292
  br i1 %cmp111, label %if.then112, label %if.else, !dbg !1293

if.then112:                                       ; preds = %for.end110
  store i32 0, i32* %j1, align 4, !dbg !1294
  br label %for.cond113, !dbg !1297

for.cond113:                                      ; preds = %for.inc148, %if.then112
  %98 = load i32, i32* %j1, align 4, !dbg !1298
  %cmp114 = icmp slt i32 %98, 256, !dbg !1300
  br i1 %cmp114, label %for.body115, label %for.end150, !dbg !1301

for.body115:                                      ; preds = %for.cond113
  %99 = load %struct.dcomplex*, %struct.dcomplex** %gty2.addr, align 8, !dbg !1302
  %100 = load i32, i32* %j, align 4, !dbg !1304
  %101 = load i32, i32* %j1, align 4, !dbg !1305
  %mul116 = mul nsw i32 %101, 256, !dbg !1306
  %add117 = add nsw i32 %100, %mul116, !dbg !1307
  %102 = load i32, i32* %k, align 4, !dbg !1308
  %mul118 = mul nsw i32 %102, 256, !dbg !1309
  %mul119 = mul nsw i32 %mul118, 256, !dbg !1310
  %add120 = add nsw i32 %add117, %mul119, !dbg !1311
  %idxprom121 = sext i32 %add120 to i64, !dbg !1302
  %arrayidx122 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %99, i64 %idxprom121, !dbg !1302
  %real123 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx122, i32 0, i32 0, !dbg !1312
  %103 = load double, double* %real123, align 8, !dbg !1312
  %104 = load %struct.dcomplex*, %struct.dcomplex** %gty1.addr, align 8, !dbg !1313
  %105 = load i32, i32* %j, align 4, !dbg !1314
  %106 = load i32, i32* %j1, align 4, !dbg !1315
  %mul124 = mul nsw i32 %106, 256, !dbg !1316
  %add125 = add nsw i32 %105, %mul124, !dbg !1317
  %107 = load i32, i32* %k, align 4, !dbg !1318
  %mul126 = mul nsw i32 %107, 256, !dbg !1319
  %mul127 = mul nsw i32 %mul126, 256, !dbg !1320
  %add128 = add nsw i32 %add125, %mul127, !dbg !1321
  %idxprom129 = sext i32 %add128 to i64, !dbg !1313
  %arrayidx130 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %104, i64 %idxprom129, !dbg !1313
  %real131 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx130, i32 0, i32 0, !dbg !1322
  store double %103, double* %real131, align 8, !dbg !1323
  %108 = load %struct.dcomplex*, %struct.dcomplex** %gty2.addr, align 8, !dbg !1324
  %109 = load i32, i32* %j, align 4, !dbg !1325
  %110 = load i32, i32* %j1, align 4, !dbg !1326
  %mul132 = mul nsw i32 %110, 256, !dbg !1327
  %add133 = add nsw i32 %109, %mul132, !dbg !1328
  %111 = load i32, i32* %k, align 4, !dbg !1329
  %mul134 = mul nsw i32 %111, 256, !dbg !1330
  %mul135 = mul nsw i32 %mul134, 256, !dbg !1331
  %add136 = add nsw i32 %add133, %mul135, !dbg !1332
  %idxprom137 = sext i32 %add136 to i64, !dbg !1324
  %arrayidx138 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %108, i64 %idxprom137, !dbg !1324
  %imag139 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx138, i32 0, i32 1, !dbg !1333
  %112 = load double, double* %imag139, align 8, !dbg !1333
  %113 = load %struct.dcomplex*, %struct.dcomplex** %gty1.addr, align 8, !dbg !1334
  %114 = load i32, i32* %j, align 4, !dbg !1335
  %115 = load i32, i32* %j1, align 4, !dbg !1336
  %mul140 = mul nsw i32 %115, 256, !dbg !1337
  %add141 = add nsw i32 %114, %mul140, !dbg !1338
  %116 = load i32, i32* %k, align 4, !dbg !1339
  %mul142 = mul nsw i32 %116, 256, !dbg !1340
  %mul143 = mul nsw i32 %mul142, 256, !dbg !1341
  %add144 = add nsw i32 %add141, %mul143, !dbg !1342
  %idxprom145 = sext i32 %add144 to i64, !dbg !1334
  %arrayidx146 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %113, i64 %idxprom145, !dbg !1334
  %imag147 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx146, i32 0, i32 1, !dbg !1343
  store double %112, double* %imag147, align 8, !dbg !1344
  br label %for.inc148, !dbg !1345

for.inc148:                                       ; preds = %for.body115
  %117 = load i32, i32* %j1, align 4, !dbg !1346
  %inc149 = add nsw i32 %117, 1, !dbg !1346
  store i32 %inc149, i32* %j1, align 4, !dbg !1346
  br label %for.cond113, !dbg !1347, !llvm.loop !1348

for.end150:                                       ; preds = %for.cond113
  br label %if.end268, !dbg !1350

if.else:                                          ; preds = %for.end110
  store i32 128, i32* %n1, align 4, !dbg !1351
  %118 = load i32, i32* %l, align 4, !dbg !1353
  %add151 = add nsw i32 %118, 1, !dbg !1354
  %sub152 = sub nsw i32 %add151, 1, !dbg !1355
  %shl153 = shl i32 1, %sub152, !dbg !1356
  store i32 %shl153, i32* %lk, align 4, !dbg !1357
  %119 = load i32, i32* %logd1, align 4, !dbg !1358
  %120 = load i32, i32* %l, align 4, !dbg !1359
  %add154 = add nsw i32 %120, 1, !dbg !1360
  %sub155 = sub nsw i32 %119, %add154, !dbg !1361
  %shl156 = shl i32 1, %sub155, !dbg !1362
  store i32 %shl156, i32* %li, align 4, !dbg !1363
  %121 = load i32, i32* %lk, align 4, !dbg !1364
  %mul157 = mul nsw i32 2, %121, !dbg !1365
  store i32 %mul157, i32* %lj, align 4, !dbg !1366
  %122 = load i32, i32* %li, align 4, !dbg !1367
  store i32 %122, i32* %ku, align 4, !dbg !1368
  store i32 0, i32* %i1, align 4, !dbg !1369
  br label %for.cond158, !dbg !1371

for.cond158:                                      ; preds = %for.inc265, %if.else
  %123 = load i32, i32* %i1, align 4, !dbg !1372
  %124 = load i32, i32* %li, align 4, !dbg !1374
  %sub159 = sub nsw i32 %124, 1, !dbg !1375
  %cmp160 = icmp sle i32 %123, %sub159, !dbg !1376
  br i1 %cmp160, label %for.body161, label %for.end267, !dbg !1377

for.body161:                                      ; preds = %for.cond158
  store i32 0, i32* %k1, align 4, !dbg !1378
  br label %for.cond162, !dbg !1381

for.cond162:                                      ; preds = %for.inc262, %for.body161
  %125 = load i32, i32* %k1, align 4, !dbg !1382
  %126 = load i32, i32* %lk, align 4, !dbg !1384
  %sub163 = sub nsw i32 %126, 1, !dbg !1385
  %cmp164 = icmp sle i32 %125, %sub163, !dbg !1386
  br i1 %cmp164, label %for.body165, label %for.end264, !dbg !1387

for.body165:                                      ; preds = %for.cond162
  %127 = load i32, i32* %i1, align 4, !dbg !1388
  %128 = load i32, i32* %lk, align 4, !dbg !1390
  %mul166 = mul nsw i32 %127, %128, !dbg !1391
  store i32 %mul166, i32* %i11, align 4, !dbg !1392
  %129 = load i32, i32* %i11, align 4, !dbg !1393
  %130 = load i32, i32* %n1, align 4, !dbg !1394
  %add167 = add nsw i32 %129, %130, !dbg !1395
  store i32 %add167, i32* %i12, align 4, !dbg !1396
  %131 = load i32, i32* %i1, align 4, !dbg !1397
  %132 = load i32, i32* %lj, align 4, !dbg !1398
  %mul168 = mul nsw i32 %131, %132, !dbg !1399
  store i32 %mul168, i32* %i21, align 4, !dbg !1400
  %133 = load i32, i32* %i21, align 4, !dbg !1401
  %134 = load i32, i32* %lk, align 4, !dbg !1402
  %add169 = add nsw i32 %133, %134, !dbg !1403
  store i32 %add169, i32* %i22, align 4, !dbg !1404
  %135 = load %struct.dcomplex*, %struct.dcomplex** %u_device.addr, align 8, !dbg !1405
  %136 = load i32, i32* %ku, align 4, !dbg !1406
  %137 = load i32, i32* %i1, align 4, !dbg !1407
  %add170 = add nsw i32 %136, %137, !dbg !1408
  %idxprom171 = sext i32 %add170 to i64, !dbg !1405
  %arrayidx172 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %135, i64 %idxprom171, !dbg !1405
  %real173 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx172, i32 0, i32 0, !dbg !1409
  %138 = load double, double* %real173, align 8, !dbg !1409
  store double %138, double* %uu2_real, align 8, !dbg !1410
  %139 = load i32, i32* %is.addr, align 4, !dbg !1411
  %conv174 = sitofp i32 %139 to double, !dbg !1411
  %140 = load %struct.dcomplex*, %struct.dcomplex** %u_device.addr, align 8, !dbg !1412
  %141 = load i32, i32* %ku, align 4, !dbg !1413
  %142 = load i32, i32* %i1, align 4, !dbg !1414
  %add175 = add nsw i32 %141, %142, !dbg !1415
  %idxprom176 = sext i32 %add175 to i64, !dbg !1412
  %arrayidx177 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %140, i64 %idxprom176, !dbg !1412
  %imag178 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx177, i32 0, i32 1, !dbg !1416
  %143 = load double, double* %imag178, align 8, !dbg !1416
  %mul179 = fmul contract double %conv174, %143, !dbg !1417
  store double %mul179, double* %uu2_imag, align 8, !dbg !1418
  %144 = load %struct.dcomplex*, %struct.dcomplex** %gty2.addr, align 8, !dbg !1419
  %145 = load i32, i32* %j, align 4, !dbg !1420
  %146 = load i32, i32* %i11, align 4, !dbg !1421
  %147 = load i32, i32* %k1, align 4, !dbg !1422
  %add180 = add nsw i32 %146, %147, !dbg !1423
  %mul181 = mul nsw i32 %add180, 256, !dbg !1424
  %add182 = add nsw i32 %145, %mul181, !dbg !1425
  %148 = load i32, i32* %k, align 4, !dbg !1426
  %mul183 = mul nsw i32 %148, 256, !dbg !1427
  %mul184 = mul nsw i32 %mul183, 256, !dbg !1428
  %add185 = add nsw i32 %add182, %mul184, !dbg !1429
  %idxprom186 = sext i32 %add185 to i64, !dbg !1419
  %arrayidx187 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %144, i64 %idxprom186, !dbg !1419
  %real188 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx187, i32 0, i32 0, !dbg !1430
  %149 = load double, double* %real188, align 8, !dbg !1430
  store double %149, double* %x12_real, align 8, !dbg !1431
  %150 = load %struct.dcomplex*, %struct.dcomplex** %gty2.addr, align 8, !dbg !1432
  %151 = load i32, i32* %j, align 4, !dbg !1433
  %152 = load i32, i32* %i11, align 4, !dbg !1434
  %153 = load i32, i32* %k1, align 4, !dbg !1435
  %add189 = add nsw i32 %152, %153, !dbg !1436
  %mul190 = mul nsw i32 %add189, 256, !dbg !1437
  %add191 = add nsw i32 %151, %mul190, !dbg !1438
  %154 = load i32, i32* %k, align 4, !dbg !1439
  %mul192 = mul nsw i32 %154, 256, !dbg !1440
  %mul193 = mul nsw i32 %mul192, 256, !dbg !1441
  %add194 = add nsw i32 %add191, %mul193, !dbg !1442
  %idxprom195 = sext i32 %add194 to i64, !dbg !1432
  %arrayidx196 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %150, i64 %idxprom195, !dbg !1432
  %imag197 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx196, i32 0, i32 1, !dbg !1443
  %155 = load double, double* %imag197, align 8, !dbg !1443
  store double %155, double* %x12_imag, align 8, !dbg !1444
  %156 = load %struct.dcomplex*, %struct.dcomplex** %gty2.addr, align 8, !dbg !1445
  %157 = load i32, i32* %j, align 4, !dbg !1446
  %158 = load i32, i32* %i12, align 4, !dbg !1447
  %159 = load i32, i32* %k1, align 4, !dbg !1448
  %add198 = add nsw i32 %158, %159, !dbg !1449
  %mul199 = mul nsw i32 %add198, 256, !dbg !1450
  %add200 = add nsw i32 %157, %mul199, !dbg !1451
  %160 = load i32, i32* %k, align 4, !dbg !1452
  %mul201 = mul nsw i32 %160, 256, !dbg !1453
  %mul202 = mul nsw i32 %mul201, 256, !dbg !1454
  %add203 = add nsw i32 %add200, %mul202, !dbg !1455
  %idxprom204 = sext i32 %add203 to i64, !dbg !1445
  %arrayidx205 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %156, i64 %idxprom204, !dbg !1445
  %real206 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx205, i32 0, i32 0, !dbg !1456
  %161 = load double, double* %real206, align 8, !dbg !1456
  store double %161, double* %x22_real, align 8, !dbg !1457
  %162 = load %struct.dcomplex*, %struct.dcomplex** %gty2.addr, align 8, !dbg !1458
  %163 = load i32, i32* %j, align 4, !dbg !1459
  %164 = load i32, i32* %i12, align 4, !dbg !1460
  %165 = load i32, i32* %k1, align 4, !dbg !1461
  %add207 = add nsw i32 %164, %165, !dbg !1462
  %mul208 = mul nsw i32 %add207, 256, !dbg !1463
  %add209 = add nsw i32 %163, %mul208, !dbg !1464
  %166 = load i32, i32* %k, align 4, !dbg !1465
  %mul210 = mul nsw i32 %166, 256, !dbg !1466
  %mul211 = mul nsw i32 %mul210, 256, !dbg !1467
  %add212 = add nsw i32 %add209, %mul211, !dbg !1468
  %idxprom213 = sext i32 %add212 to i64, !dbg !1458
  %arrayidx214 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %162, i64 %idxprom213, !dbg !1458
  %imag215 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx214, i32 0, i32 1, !dbg !1469
  %167 = load double, double* %imag215, align 8, !dbg !1469
  store double %167, double* %x22_imag, align 8, !dbg !1470
  %168 = load double, double* %x12_real, align 8, !dbg !1471
  %169 = load double, double* %x22_real, align 8, !dbg !1472
  %add216 = fadd contract double %168, %169, !dbg !1473
  %170 = load %struct.dcomplex*, %struct.dcomplex** %gty1.addr, align 8, !dbg !1474
  %171 = load i32, i32* %j, align 4, !dbg !1475
  %172 = load i32, i32* %i21, align 4, !dbg !1476
  %173 = load i32, i32* %k1, align 4, !dbg !1477
  %add217 = add nsw i32 %172, %173, !dbg !1478
  %mul218 = mul nsw i32 %add217, 256, !dbg !1479
  %add219 = add nsw i32 %171, %mul218, !dbg !1480
  %174 = load i32, i32* %k, align 4, !dbg !1481
  %mul220 = mul nsw i32 %174, 256, !dbg !1482
  %mul221 = mul nsw i32 %mul220, 256, !dbg !1483
  %add222 = add nsw i32 %add219, %mul221, !dbg !1484
  %idxprom223 = sext i32 %add222 to i64, !dbg !1474
  %arrayidx224 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %170, i64 %idxprom223, !dbg !1474
  %real225 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx224, i32 0, i32 0, !dbg !1485
  store double %add216, double* %real225, align 8, !dbg !1486
  %175 = load double, double* %x12_imag, align 8, !dbg !1487
  %176 = load double, double* %x22_imag, align 8, !dbg !1488
  %add226 = fadd contract double %175, %176, !dbg !1489
  %177 = load %struct.dcomplex*, %struct.dcomplex** %gty1.addr, align 8, !dbg !1490
  %178 = load i32, i32* %j, align 4, !dbg !1491
  %179 = load i32, i32* %i21, align 4, !dbg !1492
  %180 = load i32, i32* %k1, align 4, !dbg !1493
  %add227 = add nsw i32 %179, %180, !dbg !1494
  %mul228 = mul nsw i32 %add227, 256, !dbg !1495
  %add229 = add nsw i32 %178, %mul228, !dbg !1496
  %181 = load i32, i32* %k, align 4, !dbg !1497
  %mul230 = mul nsw i32 %181, 256, !dbg !1498
  %mul231 = mul nsw i32 %mul230, 256, !dbg !1499
  %add232 = add nsw i32 %add229, %mul231, !dbg !1500
  %idxprom233 = sext i32 %add232 to i64, !dbg !1490
  %arrayidx234 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %177, i64 %idxprom233, !dbg !1490
  %imag235 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx234, i32 0, i32 1, !dbg !1501
  store double %add226, double* %imag235, align 8, !dbg !1502
  %182 = load double, double* %x12_real, align 8, !dbg !1503
  %183 = load double, double* %x22_real, align 8, !dbg !1504
  %sub236 = fsub contract double %182, %183, !dbg !1505
  store double %sub236, double* %temp2_real, align 8, !dbg !1506
  %184 = load double, double* %x12_imag, align 8, !dbg !1507
  %185 = load double, double* %x22_imag, align 8, !dbg !1508
  %sub237 = fsub contract double %184, %185, !dbg !1509
  store double %sub237, double* %temp2_imag, align 8, !dbg !1510
  %186 = load double, double* %uu2_real, align 8, !dbg !1511
  %187 = load double, double* %temp2_real, align 8, !dbg !1512
  %mul238 = fmul contract double %186, %187, !dbg !1513
  %188 = load double, double* %uu2_imag, align 8, !dbg !1514
  %189 = load double, double* %temp2_imag, align 8, !dbg !1515
  %mul239 = fmul contract double %188, %189, !dbg !1516
  %sub240 = fsub contract double %mul238, %mul239, !dbg !1517
  %190 = load %struct.dcomplex*, %struct.dcomplex** %gty1.addr, align 8, !dbg !1518
  %191 = load i32, i32* %j, align 4, !dbg !1519
  %192 = load i32, i32* %i22, align 4, !dbg !1520
  %193 = load i32, i32* %k1, align 4, !dbg !1521
  %add241 = add nsw i32 %192, %193, !dbg !1522
  %mul242 = mul nsw i32 %add241, 256, !dbg !1523
  %add243 = add nsw i32 %191, %mul242, !dbg !1524
  %194 = load i32, i32* %k, align 4, !dbg !1525
  %mul244 = mul nsw i32 %194, 256, !dbg !1526
  %mul245 = mul nsw i32 %mul244, 256, !dbg !1527
  %add246 = add nsw i32 %add243, %mul245, !dbg !1528
  %idxprom247 = sext i32 %add246 to i64, !dbg !1518
  %arrayidx248 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %190, i64 %idxprom247, !dbg !1518
  %real249 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx248, i32 0, i32 0, !dbg !1529
  store double %sub240, double* %real249, align 8, !dbg !1530
  %195 = load double, double* %uu2_real, align 8, !dbg !1531
  %196 = load double, double* %temp2_imag, align 8, !dbg !1532
  %mul250 = fmul contract double %195, %196, !dbg !1533
  %197 = load double, double* %uu2_imag, align 8, !dbg !1534
  %198 = load double, double* %temp2_real, align 8, !dbg !1535
  %mul251 = fmul contract double %197, %198, !dbg !1536
  %add252 = fadd contract double %mul250, %mul251, !dbg !1537
  %199 = load %struct.dcomplex*, %struct.dcomplex** %gty1.addr, align 8, !dbg !1538
  %200 = load i32, i32* %j, align 4, !dbg !1539
  %201 = load i32, i32* %i22, align 4, !dbg !1540
  %202 = load i32, i32* %k1, align 4, !dbg !1541
  %add253 = add nsw i32 %201, %202, !dbg !1542
  %mul254 = mul nsw i32 %add253, 256, !dbg !1543
  %add255 = add nsw i32 %200, %mul254, !dbg !1544
  %203 = load i32, i32* %k, align 4, !dbg !1545
  %mul256 = mul nsw i32 %203, 256, !dbg !1546
  %mul257 = mul nsw i32 %mul256, 256, !dbg !1547
  %add258 = add nsw i32 %add255, %mul257, !dbg !1548
  %idxprom259 = sext i32 %add258 to i64, !dbg !1538
  %arrayidx260 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %199, i64 %idxprom259, !dbg !1538
  %imag261 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx260, i32 0, i32 1, !dbg !1549
  store double %add252, double* %imag261, align 8, !dbg !1550
  br label %for.inc262, !dbg !1551

for.inc262:                                       ; preds = %for.body165
  %204 = load i32, i32* %k1, align 4, !dbg !1552
  %inc263 = add nsw i32 %204, 1, !dbg !1552
  store i32 %inc263, i32* %k1, align 4, !dbg !1552
  br label %for.cond162, !dbg !1553, !llvm.loop !1554

for.end264:                                       ; preds = %for.cond162
  br label %for.inc265, !dbg !1556

for.inc265:                                       ; preds = %for.end264
  %205 = load i32, i32* %i1, align 4, !dbg !1557
  %inc266 = add nsw i32 %205, 1, !dbg !1557
  store i32 %inc266, i32* %i1, align 4, !dbg !1557
  br label %for.cond158, !dbg !1558, !llvm.loop !1559

for.end267:                                       ; preds = %for.cond158
  br label %if.end268

if.end268:                                        ; preds = %for.end267, %for.end150
  br label %for.inc269, !dbg !1561

for.inc269:                                       ; preds = %if.end268
  %206 = load i32, i32* %l, align 4, !dbg !1562
  %add270 = add nsw i32 %206, 2, !dbg !1562
  store i32 %add270, i32* %l, align 4, !dbg !1562
  br label %for.cond, !dbg !1563, !llvm.loop !1564

for.end271:                                       ; preds = %if.then, %for.cond
  ret void, !dbg !1566
}

; Function Attrs: convergent noinline nounwind
define dso_local i32 @_Z12ilog2_devicei(i32 %n) #0 !dbg !1567 {
entry:
  %retval = alloca i32, align 4
  %n.addr = alloca i32, align 4
  %nn = alloca i32, align 4
  %lg = alloca i32, align 4
  store i32 %n, i32* %n.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %n.addr, metadata !1568, metadata !DIExpression()), !dbg !1569
  call void @llvm.dbg.declare(metadata i32* %nn, metadata !1570, metadata !DIExpression()), !dbg !1571
  call void @llvm.dbg.declare(metadata i32* %lg, metadata !1572, metadata !DIExpression()), !dbg !1573
  %0 = load i32, i32* %n.addr, align 4, !dbg !1574
  %cmp = icmp eq i32 %0, 1, !dbg !1576
  br i1 %cmp, label %if.then, label %if.end, !dbg !1577

if.then:                                          ; preds = %entry
  store i32 0, i32* %retval, align 4, !dbg !1578
  br label %return, !dbg !1578

if.end:                                           ; preds = %entry
  store i32 1, i32* %lg, align 4, !dbg !1580
  store i32 2, i32* %nn, align 4, !dbg !1581
  br label %while.cond, !dbg !1582

while.cond:                                       ; preds = %while.body, %if.end
  %1 = load i32, i32* %nn, align 4, !dbg !1583
  %2 = load i32, i32* %n.addr, align 4, !dbg !1584
  %cmp1 = icmp slt i32 %1, %2, !dbg !1585
  br i1 %cmp1, label %while.body, label %while.end, !dbg !1582

while.body:                                       ; preds = %while.cond
  %3 = load i32, i32* %nn, align 4, !dbg !1586
  %shl = shl i32 %3, 1, !dbg !1588
  store i32 %shl, i32* %nn, align 4, !dbg !1589
  %4 = load i32, i32* %lg, align 4, !dbg !1590
  %inc = add nsw i32 %4, 1, !dbg !1590
  store i32 %inc, i32* %lg, align 4, !dbg !1590
  br label %while.cond, !dbg !1582, !llvm.loop !1591

while.end:                                        ; preds = %while.cond
  %5 = load i32, i32* %lg, align 4, !dbg !1593
  store i32 %5, i32* %retval, align 4, !dbg !1594
  br label %return, !dbg !1594

return:                                           ; preds = %while.end, %if.then
  %6 = load i32, i32* %retval, align 4, !dbg !1595
  ret i32 %6, !dbg !1595
}

; Function Attrs: convergent noinline nounwind
define dso_local void @_Z19cffts1_gpu_kernel_3P8dcomplexS0_(%struct.dcomplex* %x_out, %struct.dcomplex* %y0) #0 !dbg !1596 {
entry:
  %x_out.addr = alloca %struct.dcomplex*, align 8
  %y0.addr = alloca %struct.dcomplex*, align 8
  %x_y_z = alloca i32, align 4
  %x = alloca i32, align 4
  %y = alloca i32, align 4
  %z = alloca i32, align 4
  store %struct.dcomplex* %x_out, %struct.dcomplex** %x_out.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %x_out.addr, metadata !1597, metadata !DIExpression()), !dbg !1598
  store %struct.dcomplex* %y0, %struct.dcomplex** %y0.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %y0.addr, metadata !1599, metadata !DIExpression()), !dbg !1600
  call void @llvm.dbg.declare(metadata i32* %x_y_z, metadata !1601, metadata !DIExpression()), !dbg !1602
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #8, !dbg !1603, !range !848
  %1 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #8, !dbg !1605, !range !893
  %mul = mul i32 %0, %1, !dbg !1607
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #8, !dbg !1608, !range !923
  %add = add i32 %mul, %2, !dbg !1610
  store i32 %add, i32* %x_y_z, align 4, !dbg !1602
  %3 = load i32, i32* %x_y_z, align 4, !dbg !1611
  %cmp = icmp sge i32 %3, 8388608, !dbg !1613
  br i1 %cmp, label %if.then, label %if.end, !dbg !1614

if.then:                                          ; preds = %entry
  br label %return, !dbg !1615

if.end:                                           ; preds = %entry
  call void @llvm.dbg.declare(metadata i32* %x, metadata !1617, metadata !DIExpression()), !dbg !1618
  %4 = load i32, i32* %x_y_z, align 4, !dbg !1619
  %rem = srem i32 %4, 256, !dbg !1620
  store i32 %rem, i32* %x, align 4, !dbg !1618
  call void @llvm.dbg.declare(metadata i32* %y, metadata !1621, metadata !DIExpression()), !dbg !1622
  %5 = load i32, i32* %x_y_z, align 4, !dbg !1623
  %div = sdiv i32 %5, 256, !dbg !1624
  %rem3 = srem i32 %div, 256, !dbg !1625
  store i32 %rem3, i32* %y, align 4, !dbg !1622
  call void @llvm.dbg.declare(metadata i32* %z, metadata !1626, metadata !DIExpression()), !dbg !1627
  %6 = load i32, i32* %x_y_z, align 4, !dbg !1628
  %div4 = sdiv i32 %6, 65536, !dbg !1629
  store i32 %div4, i32* %z, align 4, !dbg !1627
  %7 = load %struct.dcomplex*, %struct.dcomplex** %y0.addr, align 8, !dbg !1630
  %8 = load i32, i32* %y, align 4, !dbg !1631
  %9 = load i32, i32* %x, align 4, !dbg !1632
  %mul5 = mul nsw i32 %9, 256, !dbg !1633
  %add6 = add nsw i32 %8, %mul5, !dbg !1634
  %10 = load i32, i32* %z, align 4, !dbg !1635
  %mul7 = mul nsw i32 %10, 256, !dbg !1636
  %mul8 = mul nsw i32 %mul7, 256, !dbg !1637
  %add9 = add nsw i32 %add6, %mul8, !dbg !1638
  %idxprom = sext i32 %add9 to i64, !dbg !1630
  %arrayidx = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %7, i64 %idxprom, !dbg !1630
  %real = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx, i32 0, i32 0, !dbg !1639
  %11 = load double, double* %real, align 8, !dbg !1639
  %12 = load %struct.dcomplex*, %struct.dcomplex** %x_out.addr, align 8, !dbg !1640
  %13 = load i32, i32* %x_y_z, align 4, !dbg !1641
  %idxprom10 = sext i32 %13 to i64, !dbg !1640
  %arrayidx11 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %12, i64 %idxprom10, !dbg !1640
  %real12 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx11, i32 0, i32 0, !dbg !1642
  store double %11, double* %real12, align 8, !dbg !1643
  %14 = load %struct.dcomplex*, %struct.dcomplex** %y0.addr, align 8, !dbg !1644
  %15 = load i32, i32* %y, align 4, !dbg !1645
  %16 = load i32, i32* %x, align 4, !dbg !1646
  %mul13 = mul nsw i32 %16, 256, !dbg !1647
  %add14 = add nsw i32 %15, %mul13, !dbg !1648
  %17 = load i32, i32* %z, align 4, !dbg !1649
  %mul15 = mul nsw i32 %17, 256, !dbg !1650
  %mul16 = mul nsw i32 %mul15, 256, !dbg !1651
  %add17 = add nsw i32 %add14, %mul16, !dbg !1652
  %idxprom18 = sext i32 %add17 to i64, !dbg !1644
  %arrayidx19 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %14, i64 %idxprom18, !dbg !1644
  %imag = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx19, i32 0, i32 1, !dbg !1653
  %18 = load double, double* %imag, align 8, !dbg !1653
  %19 = load %struct.dcomplex*, %struct.dcomplex** %x_out.addr, align 8, !dbg !1654
  %20 = load i32, i32* %x_y_z, align 4, !dbg !1655
  %idxprom20 = sext i32 %20 to i64, !dbg !1654
  %arrayidx21 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %19, i64 %idxprom20, !dbg !1654
  %imag22 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx21, i32 0, i32 1, !dbg !1656
  store double %18, double* %imag22, align 8, !dbg !1657
  br label %return, !dbg !1658

return:                                           ; preds = %if.end, %if.then
  ret void, !dbg !1658
}

; Function Attrs: convergent noinline nounwind
define dso_local void @_Z19cffts2_gpu_kernel_1P8dcomplexS0_(%struct.dcomplex* %x_in, %struct.dcomplex* %y0) #0 !dbg !1659 {
entry:
  %x_in.addr = alloca %struct.dcomplex*, align 8
  %y0.addr = alloca %struct.dcomplex*, align 8
  %x_y_z = alloca i32, align 4
  store %struct.dcomplex* %x_in, %struct.dcomplex** %x_in.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %x_in.addr, metadata !1660, metadata !DIExpression()), !dbg !1661
  store %struct.dcomplex* %y0, %struct.dcomplex** %y0.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %y0.addr, metadata !1662, metadata !DIExpression()), !dbg !1663
  call void @llvm.dbg.declare(metadata i32* %x_y_z, metadata !1664, metadata !DIExpression()), !dbg !1665
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #8, !dbg !1666, !range !848
  %1 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #8, !dbg !1668, !range !893
  %mul = mul i32 %0, %1, !dbg !1670
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #8, !dbg !1671, !range !923
  %add = add i32 %mul, %2, !dbg !1673
  store i32 %add, i32* %x_y_z, align 4, !dbg !1665
  %3 = load i32, i32* %x_y_z, align 4, !dbg !1674
  %cmp = icmp sge i32 %3, 8388608, !dbg !1676
  br i1 %cmp, label %if.then, label %if.end, !dbg !1677

if.then:                                          ; preds = %entry
  br label %return, !dbg !1678

if.end:                                           ; preds = %entry
  %4 = load %struct.dcomplex*, %struct.dcomplex** %x_in.addr, align 8, !dbg !1680
  %5 = load i32, i32* %x_y_z, align 4, !dbg !1681
  %idxprom = sext i32 %5 to i64, !dbg !1680
  %arrayidx = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %4, i64 %idxprom, !dbg !1680
  %real = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx, i32 0, i32 0, !dbg !1682
  %6 = load double, double* %real, align 8, !dbg !1682
  %7 = load %struct.dcomplex*, %struct.dcomplex** %y0.addr, align 8, !dbg !1683
  %8 = load i32, i32* %x_y_z, align 4, !dbg !1684
  %idxprom3 = sext i32 %8 to i64, !dbg !1683
  %arrayidx4 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %7, i64 %idxprom3, !dbg !1683
  %real5 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx4, i32 0, i32 0, !dbg !1685
  store double %6, double* %real5, align 8, !dbg !1686
  %9 = load %struct.dcomplex*, %struct.dcomplex** %x_in.addr, align 8, !dbg !1687
  %10 = load i32, i32* %x_y_z, align 4, !dbg !1688
  %idxprom6 = sext i32 %10 to i64, !dbg !1687
  %arrayidx7 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %9, i64 %idxprom6, !dbg !1687
  %imag = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx7, i32 0, i32 1, !dbg !1689
  %11 = load double, double* %imag, align 8, !dbg !1689
  %12 = load %struct.dcomplex*, %struct.dcomplex** %y0.addr, align 8, !dbg !1690
  %13 = load i32, i32* %x_y_z, align 4, !dbg !1691
  %idxprom8 = sext i32 %13 to i64, !dbg !1690
  %arrayidx9 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %12, i64 %idxprom8, !dbg !1690
  %imag10 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx9, i32 0, i32 1, !dbg !1692
  store double %11, double* %imag10, align 8, !dbg !1693
  br label %return, !dbg !1694

return:                                           ; preds = %if.end, %if.then
  ret void, !dbg !1694
}

; Function Attrs: convergent noinline nounwind
define dso_local void @_Z19cffts2_gpu_kernel_2iP8dcomplexS0_S0_(i32 %is, %struct.dcomplex* %gty1, %struct.dcomplex* %gty2, %struct.dcomplex* %u_device) #0 !dbg !1695 {
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
  call void @llvm.dbg.declare(metadata i32* %is.addr, metadata !1696, metadata !DIExpression()), !dbg !1697
  store %struct.dcomplex* %gty1, %struct.dcomplex** %gty1.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %gty1.addr, metadata !1698, metadata !DIExpression()), !dbg !1699
  store %struct.dcomplex* %gty2, %struct.dcomplex** %gty2.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %gty2.addr, metadata !1700, metadata !DIExpression()), !dbg !1701
  store %struct.dcomplex* %u_device, %struct.dcomplex** %u_device.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %u_device.addr, metadata !1702, metadata !DIExpression()), !dbg !1703
  call void @llvm.dbg.declare(metadata i32* %x_z, metadata !1704, metadata !DIExpression()), !dbg !1705
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #8, !dbg !1706, !range !848
  %1 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #8, !dbg !1708, !range !893
  %mul = mul i32 %0, %1, !dbg !1710
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #8, !dbg !1711, !range !923
  %add = add i32 %mul, %2, !dbg !1713
  store i32 %add, i32* %x_z, align 4, !dbg !1705
  %3 = load i32, i32* %x_z, align 4, !dbg !1714
  %cmp = icmp sge i32 %3, 32768, !dbg !1716
  br i1 %cmp, label %if.then, label %if.end, !dbg !1717

if.then:                                          ; preds = %entry
  br label %for.end271, !dbg !1718

if.end:                                           ; preds = %entry
  call void @llvm.dbg.declare(metadata i32* %i, metadata !1720, metadata !DIExpression()), !dbg !1721
  call void @llvm.dbg.declare(metadata i32* %k, metadata !1722, metadata !DIExpression()), !dbg !1723
  call void @llvm.dbg.declare(metadata i32* %l, metadata !1724, metadata !DIExpression()), !dbg !1725
  call void @llvm.dbg.declare(metadata i32* %j1, metadata !1726, metadata !DIExpression()), !dbg !1727
  call void @llvm.dbg.declare(metadata i32* %i1, metadata !1728, metadata !DIExpression()), !dbg !1729
  call void @llvm.dbg.declare(metadata i32* %k1, metadata !1730, metadata !DIExpression()), !dbg !1731
  call void @llvm.dbg.declare(metadata i32* %n1, metadata !1732, metadata !DIExpression()), !dbg !1733
  call void @llvm.dbg.declare(metadata i32* %li, metadata !1734, metadata !DIExpression()), !dbg !1735
  call void @llvm.dbg.declare(metadata i32* %lj, metadata !1736, metadata !DIExpression()), !dbg !1737
  call void @llvm.dbg.declare(metadata i32* %lk, metadata !1738, metadata !DIExpression()), !dbg !1739
  call void @llvm.dbg.declare(metadata i32* %ku, metadata !1740, metadata !DIExpression()), !dbg !1741
  call void @llvm.dbg.declare(metadata i32* %i11, metadata !1742, metadata !DIExpression()), !dbg !1743
  call void @llvm.dbg.declare(metadata i32* %i12, metadata !1744, metadata !DIExpression()), !dbg !1745
  call void @llvm.dbg.declare(metadata i32* %i21, metadata !1746, metadata !DIExpression()), !dbg !1747
  call void @llvm.dbg.declare(metadata i32* %i22, metadata !1748, metadata !DIExpression()), !dbg !1749
  %4 = load i32, i32* %x_z, align 4, !dbg !1750
  %rem = srem i32 %4, 256, !dbg !1751
  store i32 %rem, i32* %i, align 4, !dbg !1752
  %5 = load i32, i32* %x_z, align 4, !dbg !1753
  %div = sdiv i32 %5, 256, !dbg !1754
  %rem3 = srem i32 %div, 128, !dbg !1755
  store i32 %rem3, i32* %k, align 4, !dbg !1756
  call void @llvm.dbg.declare(metadata i32* %logd2, metadata !1757, metadata !DIExpression()), !dbg !1758
  %call4 = call i32 @_Z12ilog2_devicei(i32 256) #3, !dbg !1759
  store i32 %call4, i32* %logd2, align 4, !dbg !1758
  call void @llvm.dbg.declare(metadata double* %uu1_real, metadata !1760, metadata !DIExpression()), !dbg !1761
  call void @llvm.dbg.declare(metadata double* %x11_real, metadata !1762, metadata !DIExpression()), !dbg !1763
  call void @llvm.dbg.declare(metadata double* %x21_real, metadata !1764, metadata !DIExpression()), !dbg !1765
  call void @llvm.dbg.declare(metadata double* %uu1_imag, metadata !1766, metadata !DIExpression()), !dbg !1767
  call void @llvm.dbg.declare(metadata double* %x11_imag, metadata !1768, metadata !DIExpression()), !dbg !1769
  call void @llvm.dbg.declare(metadata double* %x21_imag, metadata !1770, metadata !DIExpression()), !dbg !1771
  call void @llvm.dbg.declare(metadata double* %uu2_real, metadata !1772, metadata !DIExpression()), !dbg !1773
  call void @llvm.dbg.declare(metadata double* %x12_real, metadata !1774, metadata !DIExpression()), !dbg !1775
  call void @llvm.dbg.declare(metadata double* %x22_real, metadata !1776, metadata !DIExpression()), !dbg !1777
  call void @llvm.dbg.declare(metadata double* %uu2_imag, metadata !1778, metadata !DIExpression()), !dbg !1779
  call void @llvm.dbg.declare(metadata double* %x12_imag, metadata !1780, metadata !DIExpression()), !dbg !1781
  call void @llvm.dbg.declare(metadata double* %x22_imag, metadata !1782, metadata !DIExpression()), !dbg !1783
  call void @llvm.dbg.declare(metadata double* %temp_real, metadata !1784, metadata !DIExpression()), !dbg !1785
  call void @llvm.dbg.declare(metadata double* %temp2_real, metadata !1786, metadata !DIExpression()), !dbg !1787
  call void @llvm.dbg.declare(metadata double* %temp_imag, metadata !1788, metadata !DIExpression()), !dbg !1789
  call void @llvm.dbg.declare(metadata double* %temp2_imag, metadata !1790, metadata !DIExpression()), !dbg !1791
  store i32 1, i32* %l, align 4, !dbg !1792
  br label %for.cond, !dbg !1794

for.cond:                                         ; preds = %for.inc269, %if.end
  %6 = load i32, i32* %l, align 4, !dbg !1795
  %7 = load i32, i32* %logd2, align 4, !dbg !1797
  %cmp5 = icmp sle i32 %6, %7, !dbg !1798
  br i1 %cmp5, label %for.body, label %for.end271, !dbg !1799

for.body:                                         ; preds = %for.cond
  store i32 128, i32* %n1, align 4, !dbg !1800
  %8 = load i32, i32* %l, align 4, !dbg !1802
  %sub = sub nsw i32 %8, 1, !dbg !1803
  %shl = shl i32 1, %sub, !dbg !1804
  store i32 %shl, i32* %lk, align 4, !dbg !1805
  %9 = load i32, i32* %logd2, align 4, !dbg !1806
  %10 = load i32, i32* %l, align 4, !dbg !1807
  %sub6 = sub nsw i32 %9, %10, !dbg !1808
  %shl7 = shl i32 1, %sub6, !dbg !1809
  store i32 %shl7, i32* %li, align 4, !dbg !1810
  %11 = load i32, i32* %lk, align 4, !dbg !1811
  %mul8 = mul nsw i32 2, %11, !dbg !1812
  store i32 %mul8, i32* %lj, align 4, !dbg !1813
  %12 = load i32, i32* %li, align 4, !dbg !1814
  store i32 %12, i32* %ku, align 4, !dbg !1815
  store i32 0, i32* %i1, align 4, !dbg !1816
  br label %for.cond9, !dbg !1818

for.cond9:                                        ; preds = %for.inc108, %for.body
  %13 = load i32, i32* %i1, align 4, !dbg !1819
  %14 = load i32, i32* %li, align 4, !dbg !1821
  %sub10 = sub nsw i32 %14, 1, !dbg !1822
  %cmp11 = icmp sle i32 %13, %sub10, !dbg !1823
  br i1 %cmp11, label %for.body12, label %for.end110, !dbg !1824

for.body12:                                       ; preds = %for.cond9
  store i32 0, i32* %k1, align 4, !dbg !1825
  br label %for.cond13, !dbg !1828

for.cond13:                                       ; preds = %for.inc, %for.body12
  %15 = load i32, i32* %k1, align 4, !dbg !1829
  %16 = load i32, i32* %lk, align 4, !dbg !1831
  %sub14 = sub nsw i32 %16, 1, !dbg !1832
  %cmp15 = icmp sle i32 %15, %sub14, !dbg !1833
  br i1 %cmp15, label %for.body16, label %for.end, !dbg !1834

for.body16:                                       ; preds = %for.cond13
  %17 = load i32, i32* %i1, align 4, !dbg !1835
  %18 = load i32, i32* %lk, align 4, !dbg !1837
  %mul17 = mul nsw i32 %17, %18, !dbg !1838
  store i32 %mul17, i32* %i11, align 4, !dbg !1839
  %19 = load i32, i32* %i11, align 4, !dbg !1840
  %20 = load i32, i32* %n1, align 4, !dbg !1841
  %add18 = add nsw i32 %19, %20, !dbg !1842
  store i32 %add18, i32* %i12, align 4, !dbg !1843
  %21 = load i32, i32* %i1, align 4, !dbg !1844
  %22 = load i32, i32* %lj, align 4, !dbg !1845
  %mul19 = mul nsw i32 %21, %22, !dbg !1846
  store i32 %mul19, i32* %i21, align 4, !dbg !1847
  %23 = load i32, i32* %i21, align 4, !dbg !1848
  %24 = load i32, i32* %lk, align 4, !dbg !1849
  %add20 = add nsw i32 %23, %24, !dbg !1850
  store i32 %add20, i32* %i22, align 4, !dbg !1851
  %25 = load %struct.dcomplex*, %struct.dcomplex** %u_device.addr, align 8, !dbg !1852
  %26 = load i32, i32* %ku, align 4, !dbg !1853
  %27 = load i32, i32* %i1, align 4, !dbg !1854
  %add21 = add nsw i32 %26, %27, !dbg !1855
  %idxprom = sext i32 %add21 to i64, !dbg !1852
  %arrayidx = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %25, i64 %idxprom, !dbg !1852
  %real = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx, i32 0, i32 0, !dbg !1856
  %28 = load double, double* %real, align 8, !dbg !1856
  store double %28, double* %uu1_real, align 8, !dbg !1857
  %29 = load i32, i32* %is.addr, align 4, !dbg !1858
  %conv = sitofp i32 %29 to double, !dbg !1858
  %30 = load %struct.dcomplex*, %struct.dcomplex** %u_device.addr, align 8, !dbg !1859
  %31 = load i32, i32* %ku, align 4, !dbg !1860
  %32 = load i32, i32* %i1, align 4, !dbg !1861
  %add22 = add nsw i32 %31, %32, !dbg !1862
  %idxprom23 = sext i32 %add22 to i64, !dbg !1859
  %arrayidx24 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %30, i64 %idxprom23, !dbg !1859
  %imag = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx24, i32 0, i32 1, !dbg !1863
  %33 = load double, double* %imag, align 8, !dbg !1863
  %mul25 = fmul contract double %conv, %33, !dbg !1864
  store double %mul25, double* %uu1_imag, align 8, !dbg !1865
  %34 = load %struct.dcomplex*, %struct.dcomplex** %gty1.addr, align 8, !dbg !1866
  %35 = load i32, i32* %i, align 4, !dbg !1867
  %36 = load i32, i32* %i11, align 4, !dbg !1868
  %37 = load i32, i32* %k1, align 4, !dbg !1869
  %add26 = add nsw i32 %36, %37, !dbg !1870
  %mul27 = mul nsw i32 %add26, 256, !dbg !1871
  %add28 = add nsw i32 %35, %mul27, !dbg !1872
  %38 = load i32, i32* %k, align 4, !dbg !1873
  %mul29 = mul nsw i32 %38, 256, !dbg !1874
  %mul30 = mul nsw i32 %mul29, 256, !dbg !1875
  %add31 = add nsw i32 %add28, %mul30, !dbg !1876
  %idxprom32 = sext i32 %add31 to i64, !dbg !1866
  %arrayidx33 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %34, i64 %idxprom32, !dbg !1866
  %real34 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx33, i32 0, i32 0, !dbg !1877
  %39 = load double, double* %real34, align 8, !dbg !1877
  store double %39, double* %x11_real, align 8, !dbg !1878
  %40 = load %struct.dcomplex*, %struct.dcomplex** %gty1.addr, align 8, !dbg !1879
  %41 = load i32, i32* %i, align 4, !dbg !1880
  %42 = load i32, i32* %i11, align 4, !dbg !1881
  %43 = load i32, i32* %k1, align 4, !dbg !1882
  %add35 = add nsw i32 %42, %43, !dbg !1883
  %mul36 = mul nsw i32 %add35, 256, !dbg !1884
  %add37 = add nsw i32 %41, %mul36, !dbg !1885
  %44 = load i32, i32* %k, align 4, !dbg !1886
  %mul38 = mul nsw i32 %44, 256, !dbg !1887
  %mul39 = mul nsw i32 %mul38, 256, !dbg !1888
  %add40 = add nsw i32 %add37, %mul39, !dbg !1889
  %idxprom41 = sext i32 %add40 to i64, !dbg !1879
  %arrayidx42 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %40, i64 %idxprom41, !dbg !1879
  %imag43 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx42, i32 0, i32 1, !dbg !1890
  %45 = load double, double* %imag43, align 8, !dbg !1890
  store double %45, double* %x11_imag, align 8, !dbg !1891
  %46 = load %struct.dcomplex*, %struct.dcomplex** %gty1.addr, align 8, !dbg !1892
  %47 = load i32, i32* %i, align 4, !dbg !1893
  %48 = load i32, i32* %i12, align 4, !dbg !1894
  %49 = load i32, i32* %k1, align 4, !dbg !1895
  %add44 = add nsw i32 %48, %49, !dbg !1896
  %mul45 = mul nsw i32 %add44, 256, !dbg !1897
  %add46 = add nsw i32 %47, %mul45, !dbg !1898
  %50 = load i32, i32* %k, align 4, !dbg !1899
  %mul47 = mul nsw i32 %50, 256, !dbg !1900
  %mul48 = mul nsw i32 %mul47, 256, !dbg !1901
  %add49 = add nsw i32 %add46, %mul48, !dbg !1902
  %idxprom50 = sext i32 %add49 to i64, !dbg !1892
  %arrayidx51 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %46, i64 %idxprom50, !dbg !1892
  %real52 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx51, i32 0, i32 0, !dbg !1903
  %51 = load double, double* %real52, align 8, !dbg !1903
  store double %51, double* %x21_real, align 8, !dbg !1904
  %52 = load %struct.dcomplex*, %struct.dcomplex** %gty1.addr, align 8, !dbg !1905
  %53 = load i32, i32* %i, align 4, !dbg !1906
  %54 = load i32, i32* %i12, align 4, !dbg !1907
  %55 = load i32, i32* %k1, align 4, !dbg !1908
  %add53 = add nsw i32 %54, %55, !dbg !1909
  %mul54 = mul nsw i32 %add53, 256, !dbg !1910
  %add55 = add nsw i32 %53, %mul54, !dbg !1911
  %56 = load i32, i32* %k, align 4, !dbg !1912
  %mul56 = mul nsw i32 %56, 256, !dbg !1913
  %mul57 = mul nsw i32 %mul56, 256, !dbg !1914
  %add58 = add nsw i32 %add55, %mul57, !dbg !1915
  %idxprom59 = sext i32 %add58 to i64, !dbg !1905
  %arrayidx60 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %52, i64 %idxprom59, !dbg !1905
  %imag61 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx60, i32 0, i32 1, !dbg !1916
  %57 = load double, double* %imag61, align 8, !dbg !1916
  store double %57, double* %x21_imag, align 8, !dbg !1917
  %58 = load double, double* %x11_real, align 8, !dbg !1918
  %59 = load double, double* %x21_real, align 8, !dbg !1919
  %add62 = fadd contract double %58, %59, !dbg !1920
  %60 = load %struct.dcomplex*, %struct.dcomplex** %gty2.addr, align 8, !dbg !1921
  %61 = load i32, i32* %i, align 4, !dbg !1922
  %62 = load i32, i32* %i21, align 4, !dbg !1923
  %63 = load i32, i32* %k1, align 4, !dbg !1924
  %add63 = add nsw i32 %62, %63, !dbg !1925
  %mul64 = mul nsw i32 %add63, 256, !dbg !1926
  %add65 = add nsw i32 %61, %mul64, !dbg !1927
  %64 = load i32, i32* %k, align 4, !dbg !1928
  %mul66 = mul nsw i32 %64, 256, !dbg !1929
  %mul67 = mul nsw i32 %mul66, 256, !dbg !1930
  %add68 = add nsw i32 %add65, %mul67, !dbg !1931
  %idxprom69 = sext i32 %add68 to i64, !dbg !1921
  %arrayidx70 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %60, i64 %idxprom69, !dbg !1921
  %real71 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx70, i32 0, i32 0, !dbg !1932
  store double %add62, double* %real71, align 8, !dbg !1933
  %65 = load double, double* %x11_imag, align 8, !dbg !1934
  %66 = load double, double* %x21_imag, align 8, !dbg !1935
  %add72 = fadd contract double %65, %66, !dbg !1936
  %67 = load %struct.dcomplex*, %struct.dcomplex** %gty2.addr, align 8, !dbg !1937
  %68 = load i32, i32* %i, align 4, !dbg !1938
  %69 = load i32, i32* %i21, align 4, !dbg !1939
  %70 = load i32, i32* %k1, align 4, !dbg !1940
  %add73 = add nsw i32 %69, %70, !dbg !1941
  %mul74 = mul nsw i32 %add73, 256, !dbg !1942
  %add75 = add nsw i32 %68, %mul74, !dbg !1943
  %71 = load i32, i32* %k, align 4, !dbg !1944
  %mul76 = mul nsw i32 %71, 256, !dbg !1945
  %mul77 = mul nsw i32 %mul76, 256, !dbg !1946
  %add78 = add nsw i32 %add75, %mul77, !dbg !1947
  %idxprom79 = sext i32 %add78 to i64, !dbg !1937
  %arrayidx80 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %67, i64 %idxprom79, !dbg !1937
  %imag81 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx80, i32 0, i32 1, !dbg !1948
  store double %add72, double* %imag81, align 8, !dbg !1949
  %72 = load double, double* %x11_real, align 8, !dbg !1950
  %73 = load double, double* %x21_real, align 8, !dbg !1951
  %sub82 = fsub contract double %72, %73, !dbg !1952
  store double %sub82, double* %temp_real, align 8, !dbg !1953
  %74 = load double, double* %x11_imag, align 8, !dbg !1954
  %75 = load double, double* %x21_imag, align 8, !dbg !1955
  %sub83 = fsub contract double %74, %75, !dbg !1956
  store double %sub83, double* %temp_imag, align 8, !dbg !1957
  %76 = load double, double* %uu1_real, align 8, !dbg !1958
  %77 = load double, double* %temp_real, align 8, !dbg !1959
  %mul84 = fmul contract double %76, %77, !dbg !1960
  %78 = load double, double* %uu1_imag, align 8, !dbg !1961
  %79 = load double, double* %temp_imag, align 8, !dbg !1962
  %mul85 = fmul contract double %78, %79, !dbg !1963
  %sub86 = fsub contract double %mul84, %mul85, !dbg !1964
  %80 = load %struct.dcomplex*, %struct.dcomplex** %gty2.addr, align 8, !dbg !1965
  %81 = load i32, i32* %i, align 4, !dbg !1966
  %82 = load i32, i32* %i22, align 4, !dbg !1967
  %83 = load i32, i32* %k1, align 4, !dbg !1968
  %add87 = add nsw i32 %82, %83, !dbg !1969
  %mul88 = mul nsw i32 %add87, 256, !dbg !1970
  %add89 = add nsw i32 %81, %mul88, !dbg !1971
  %84 = load i32, i32* %k, align 4, !dbg !1972
  %mul90 = mul nsw i32 %84, 256, !dbg !1973
  %mul91 = mul nsw i32 %mul90, 256, !dbg !1974
  %add92 = add nsw i32 %add89, %mul91, !dbg !1975
  %idxprom93 = sext i32 %add92 to i64, !dbg !1965
  %arrayidx94 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %80, i64 %idxprom93, !dbg !1965
  %real95 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx94, i32 0, i32 0, !dbg !1976
  store double %sub86, double* %real95, align 8, !dbg !1977
  %85 = load double, double* %uu1_real, align 8, !dbg !1978
  %86 = load double, double* %temp_imag, align 8, !dbg !1979
  %mul96 = fmul contract double %85, %86, !dbg !1980
  %87 = load double, double* %uu1_imag, align 8, !dbg !1981
  %88 = load double, double* %temp_real, align 8, !dbg !1982
  %mul97 = fmul contract double %87, %88, !dbg !1983
  %add98 = fadd contract double %mul96, %mul97, !dbg !1984
  %89 = load %struct.dcomplex*, %struct.dcomplex** %gty2.addr, align 8, !dbg !1985
  %90 = load i32, i32* %i, align 4, !dbg !1986
  %91 = load i32, i32* %i22, align 4, !dbg !1987
  %92 = load i32, i32* %k1, align 4, !dbg !1988
  %add99 = add nsw i32 %91, %92, !dbg !1989
  %mul100 = mul nsw i32 %add99, 256, !dbg !1990
  %add101 = add nsw i32 %90, %mul100, !dbg !1991
  %93 = load i32, i32* %k, align 4, !dbg !1992
  %mul102 = mul nsw i32 %93, 256, !dbg !1993
  %mul103 = mul nsw i32 %mul102, 256, !dbg !1994
  %add104 = add nsw i32 %add101, %mul103, !dbg !1995
  %idxprom105 = sext i32 %add104 to i64, !dbg !1985
  %arrayidx106 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %89, i64 %idxprom105, !dbg !1985
  %imag107 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx106, i32 0, i32 1, !dbg !1996
  store double %add98, double* %imag107, align 8, !dbg !1997
  br label %for.inc, !dbg !1998

for.inc:                                          ; preds = %for.body16
  %94 = load i32, i32* %k1, align 4, !dbg !1999
  %inc = add nsw i32 %94, 1, !dbg !1999
  store i32 %inc, i32* %k1, align 4, !dbg !1999
  br label %for.cond13, !dbg !2000, !llvm.loop !2001

for.end:                                          ; preds = %for.cond13
  br label %for.inc108, !dbg !2003

for.inc108:                                       ; preds = %for.end
  %95 = load i32, i32* %i1, align 4, !dbg !2004
  %inc109 = add nsw i32 %95, 1, !dbg !2004
  store i32 %inc109, i32* %i1, align 4, !dbg !2004
  br label %for.cond9, !dbg !2005, !llvm.loop !2006

for.end110:                                       ; preds = %for.cond9
  %96 = load i32, i32* %l, align 4, !dbg !2008
  %97 = load i32, i32* %logd2, align 4, !dbg !2010
  %cmp111 = icmp eq i32 %96, %97, !dbg !2011
  br i1 %cmp111, label %if.then112, label %if.else, !dbg !2012

if.then112:                                       ; preds = %for.end110
  store i32 0, i32* %j1, align 4, !dbg !2013
  br label %for.cond113, !dbg !2016

for.cond113:                                      ; preds = %for.inc148, %if.then112
  %98 = load i32, i32* %j1, align 4, !dbg !2017
  %cmp114 = icmp slt i32 %98, 256, !dbg !2019
  br i1 %cmp114, label %for.body115, label %for.end150, !dbg !2020

for.body115:                                      ; preds = %for.cond113
  %99 = load %struct.dcomplex*, %struct.dcomplex** %gty2.addr, align 8, !dbg !2021
  %100 = load i32, i32* %i, align 4, !dbg !2023
  %101 = load i32, i32* %j1, align 4, !dbg !2024
  %mul116 = mul nsw i32 %101, 256, !dbg !2025
  %add117 = add nsw i32 %100, %mul116, !dbg !2026
  %102 = load i32, i32* %k, align 4, !dbg !2027
  %mul118 = mul nsw i32 %102, 256, !dbg !2028
  %mul119 = mul nsw i32 %mul118, 256, !dbg !2029
  %add120 = add nsw i32 %add117, %mul119, !dbg !2030
  %idxprom121 = sext i32 %add120 to i64, !dbg !2021
  %arrayidx122 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %99, i64 %idxprom121, !dbg !2021
  %real123 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx122, i32 0, i32 0, !dbg !2031
  %103 = load double, double* %real123, align 8, !dbg !2031
  %104 = load %struct.dcomplex*, %struct.dcomplex** %gty1.addr, align 8, !dbg !2032
  %105 = load i32, i32* %i, align 4, !dbg !2033
  %106 = load i32, i32* %j1, align 4, !dbg !2034
  %mul124 = mul nsw i32 %106, 256, !dbg !2035
  %add125 = add nsw i32 %105, %mul124, !dbg !2036
  %107 = load i32, i32* %k, align 4, !dbg !2037
  %mul126 = mul nsw i32 %107, 256, !dbg !2038
  %mul127 = mul nsw i32 %mul126, 256, !dbg !2039
  %add128 = add nsw i32 %add125, %mul127, !dbg !2040
  %idxprom129 = sext i32 %add128 to i64, !dbg !2032
  %arrayidx130 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %104, i64 %idxprom129, !dbg !2032
  %real131 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx130, i32 0, i32 0, !dbg !2041
  store double %103, double* %real131, align 8, !dbg !2042
  %108 = load %struct.dcomplex*, %struct.dcomplex** %gty2.addr, align 8, !dbg !2043
  %109 = load i32, i32* %i, align 4, !dbg !2044
  %110 = load i32, i32* %j1, align 4, !dbg !2045
  %mul132 = mul nsw i32 %110, 256, !dbg !2046
  %add133 = add nsw i32 %109, %mul132, !dbg !2047
  %111 = load i32, i32* %k, align 4, !dbg !2048
  %mul134 = mul nsw i32 %111, 256, !dbg !2049
  %mul135 = mul nsw i32 %mul134, 256, !dbg !2050
  %add136 = add nsw i32 %add133, %mul135, !dbg !2051
  %idxprom137 = sext i32 %add136 to i64, !dbg !2043
  %arrayidx138 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %108, i64 %idxprom137, !dbg !2043
  %imag139 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx138, i32 0, i32 1, !dbg !2052
  %112 = load double, double* %imag139, align 8, !dbg !2052
  %113 = load %struct.dcomplex*, %struct.dcomplex** %gty1.addr, align 8, !dbg !2053
  %114 = load i32, i32* %i, align 4, !dbg !2054
  %115 = load i32, i32* %j1, align 4, !dbg !2055
  %mul140 = mul nsw i32 %115, 256, !dbg !2056
  %add141 = add nsw i32 %114, %mul140, !dbg !2057
  %116 = load i32, i32* %k, align 4, !dbg !2058
  %mul142 = mul nsw i32 %116, 256, !dbg !2059
  %mul143 = mul nsw i32 %mul142, 256, !dbg !2060
  %add144 = add nsw i32 %add141, %mul143, !dbg !2061
  %idxprom145 = sext i32 %add144 to i64, !dbg !2053
  %arrayidx146 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %113, i64 %idxprom145, !dbg !2053
  %imag147 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx146, i32 0, i32 1, !dbg !2062
  store double %112, double* %imag147, align 8, !dbg !2063
  br label %for.inc148, !dbg !2064

for.inc148:                                       ; preds = %for.body115
  %117 = load i32, i32* %j1, align 4, !dbg !2065
  %inc149 = add nsw i32 %117, 1, !dbg !2065
  store i32 %inc149, i32* %j1, align 4, !dbg !2065
  br label %for.cond113, !dbg !2066, !llvm.loop !2067

for.end150:                                       ; preds = %for.cond113
  br label %if.end268, !dbg !2069

if.else:                                          ; preds = %for.end110
  store i32 128, i32* %n1, align 4, !dbg !2070
  %118 = load i32, i32* %l, align 4, !dbg !2072
  %add151 = add nsw i32 %118, 1, !dbg !2073
  %sub152 = sub nsw i32 %add151, 1, !dbg !2074
  %shl153 = shl i32 1, %sub152, !dbg !2075
  store i32 %shl153, i32* %lk, align 4, !dbg !2076
  %119 = load i32, i32* %logd2, align 4, !dbg !2077
  %120 = load i32, i32* %l, align 4, !dbg !2078
  %add154 = add nsw i32 %120, 1, !dbg !2079
  %sub155 = sub nsw i32 %119, %add154, !dbg !2080
  %shl156 = shl i32 1, %sub155, !dbg !2081
  store i32 %shl156, i32* %li, align 4, !dbg !2082
  %121 = load i32, i32* %lk, align 4, !dbg !2083
  %mul157 = mul nsw i32 2, %121, !dbg !2084
  store i32 %mul157, i32* %lj, align 4, !dbg !2085
  %122 = load i32, i32* %li, align 4, !dbg !2086
  store i32 %122, i32* %ku, align 4, !dbg !2087
  store i32 0, i32* %i1, align 4, !dbg !2088
  br label %for.cond158, !dbg !2090

for.cond158:                                      ; preds = %for.inc265, %if.else
  %123 = load i32, i32* %i1, align 4, !dbg !2091
  %124 = load i32, i32* %li, align 4, !dbg !2093
  %sub159 = sub nsw i32 %124, 1, !dbg !2094
  %cmp160 = icmp sle i32 %123, %sub159, !dbg !2095
  br i1 %cmp160, label %for.body161, label %for.end267, !dbg !2096

for.body161:                                      ; preds = %for.cond158
  store i32 0, i32* %k1, align 4, !dbg !2097
  br label %for.cond162, !dbg !2100

for.cond162:                                      ; preds = %for.inc262, %for.body161
  %125 = load i32, i32* %k1, align 4, !dbg !2101
  %126 = load i32, i32* %lk, align 4, !dbg !2103
  %sub163 = sub nsw i32 %126, 1, !dbg !2104
  %cmp164 = icmp sle i32 %125, %sub163, !dbg !2105
  br i1 %cmp164, label %for.body165, label %for.end264, !dbg !2106

for.body165:                                      ; preds = %for.cond162
  %127 = load i32, i32* %i1, align 4, !dbg !2107
  %128 = load i32, i32* %lk, align 4, !dbg !2109
  %mul166 = mul nsw i32 %127, %128, !dbg !2110
  store i32 %mul166, i32* %i11, align 4, !dbg !2111
  %129 = load i32, i32* %i11, align 4, !dbg !2112
  %130 = load i32, i32* %n1, align 4, !dbg !2113
  %add167 = add nsw i32 %129, %130, !dbg !2114
  store i32 %add167, i32* %i12, align 4, !dbg !2115
  %131 = load i32, i32* %i1, align 4, !dbg !2116
  %132 = load i32, i32* %lj, align 4, !dbg !2117
  %mul168 = mul nsw i32 %131, %132, !dbg !2118
  store i32 %mul168, i32* %i21, align 4, !dbg !2119
  %133 = load i32, i32* %i21, align 4, !dbg !2120
  %134 = load i32, i32* %lk, align 4, !dbg !2121
  %add169 = add nsw i32 %133, %134, !dbg !2122
  store i32 %add169, i32* %i22, align 4, !dbg !2123
  %135 = load %struct.dcomplex*, %struct.dcomplex** %u_device.addr, align 8, !dbg !2124
  %136 = load i32, i32* %ku, align 4, !dbg !2125
  %137 = load i32, i32* %i1, align 4, !dbg !2126
  %add170 = add nsw i32 %136, %137, !dbg !2127
  %idxprom171 = sext i32 %add170 to i64, !dbg !2124
  %arrayidx172 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %135, i64 %idxprom171, !dbg !2124
  %real173 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx172, i32 0, i32 0, !dbg !2128
  %138 = load double, double* %real173, align 8, !dbg !2128
  store double %138, double* %uu2_real, align 8, !dbg !2129
  %139 = load i32, i32* %is.addr, align 4, !dbg !2130
  %conv174 = sitofp i32 %139 to double, !dbg !2130
  %140 = load %struct.dcomplex*, %struct.dcomplex** %u_device.addr, align 8, !dbg !2131
  %141 = load i32, i32* %ku, align 4, !dbg !2132
  %142 = load i32, i32* %i1, align 4, !dbg !2133
  %add175 = add nsw i32 %141, %142, !dbg !2134
  %idxprom176 = sext i32 %add175 to i64, !dbg !2131
  %arrayidx177 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %140, i64 %idxprom176, !dbg !2131
  %imag178 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx177, i32 0, i32 1, !dbg !2135
  %143 = load double, double* %imag178, align 8, !dbg !2135
  %mul179 = fmul contract double %conv174, %143, !dbg !2136
  store double %mul179, double* %uu2_imag, align 8, !dbg !2137
  %144 = load %struct.dcomplex*, %struct.dcomplex** %gty2.addr, align 8, !dbg !2138
  %145 = load i32, i32* %i, align 4, !dbg !2139
  %146 = load i32, i32* %i11, align 4, !dbg !2140
  %147 = load i32, i32* %k1, align 4, !dbg !2141
  %add180 = add nsw i32 %146, %147, !dbg !2142
  %mul181 = mul nsw i32 %add180, 256, !dbg !2143
  %add182 = add nsw i32 %145, %mul181, !dbg !2144
  %148 = load i32, i32* %k, align 4, !dbg !2145
  %mul183 = mul nsw i32 %148, 256, !dbg !2146
  %mul184 = mul nsw i32 %mul183, 256, !dbg !2147
  %add185 = add nsw i32 %add182, %mul184, !dbg !2148
  %idxprom186 = sext i32 %add185 to i64, !dbg !2138
  %arrayidx187 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %144, i64 %idxprom186, !dbg !2138
  %real188 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx187, i32 0, i32 0, !dbg !2149
  %149 = load double, double* %real188, align 8, !dbg !2149
  store double %149, double* %x12_real, align 8, !dbg !2150
  %150 = load %struct.dcomplex*, %struct.dcomplex** %gty2.addr, align 8, !dbg !2151
  %151 = load i32, i32* %i, align 4, !dbg !2152
  %152 = load i32, i32* %i11, align 4, !dbg !2153
  %153 = load i32, i32* %k1, align 4, !dbg !2154
  %add189 = add nsw i32 %152, %153, !dbg !2155
  %mul190 = mul nsw i32 %add189, 256, !dbg !2156
  %add191 = add nsw i32 %151, %mul190, !dbg !2157
  %154 = load i32, i32* %k, align 4, !dbg !2158
  %mul192 = mul nsw i32 %154, 256, !dbg !2159
  %mul193 = mul nsw i32 %mul192, 256, !dbg !2160
  %add194 = add nsw i32 %add191, %mul193, !dbg !2161
  %idxprom195 = sext i32 %add194 to i64, !dbg !2151
  %arrayidx196 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %150, i64 %idxprom195, !dbg !2151
  %imag197 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx196, i32 0, i32 1, !dbg !2162
  %155 = load double, double* %imag197, align 8, !dbg !2162
  store double %155, double* %x12_imag, align 8, !dbg !2163
  %156 = load %struct.dcomplex*, %struct.dcomplex** %gty2.addr, align 8, !dbg !2164
  %157 = load i32, i32* %i, align 4, !dbg !2165
  %158 = load i32, i32* %i12, align 4, !dbg !2166
  %159 = load i32, i32* %k1, align 4, !dbg !2167
  %add198 = add nsw i32 %158, %159, !dbg !2168
  %mul199 = mul nsw i32 %add198, 256, !dbg !2169
  %add200 = add nsw i32 %157, %mul199, !dbg !2170
  %160 = load i32, i32* %k, align 4, !dbg !2171
  %mul201 = mul nsw i32 %160, 256, !dbg !2172
  %mul202 = mul nsw i32 %mul201, 256, !dbg !2173
  %add203 = add nsw i32 %add200, %mul202, !dbg !2174
  %idxprom204 = sext i32 %add203 to i64, !dbg !2164
  %arrayidx205 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %156, i64 %idxprom204, !dbg !2164
  %real206 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx205, i32 0, i32 0, !dbg !2175
  %161 = load double, double* %real206, align 8, !dbg !2175
  store double %161, double* %x22_real, align 8, !dbg !2176
  %162 = load %struct.dcomplex*, %struct.dcomplex** %gty2.addr, align 8, !dbg !2177
  %163 = load i32, i32* %i, align 4, !dbg !2178
  %164 = load i32, i32* %i12, align 4, !dbg !2179
  %165 = load i32, i32* %k1, align 4, !dbg !2180
  %add207 = add nsw i32 %164, %165, !dbg !2181
  %mul208 = mul nsw i32 %add207, 256, !dbg !2182
  %add209 = add nsw i32 %163, %mul208, !dbg !2183
  %166 = load i32, i32* %k, align 4, !dbg !2184
  %mul210 = mul nsw i32 %166, 256, !dbg !2185
  %mul211 = mul nsw i32 %mul210, 256, !dbg !2186
  %add212 = add nsw i32 %add209, %mul211, !dbg !2187
  %idxprom213 = sext i32 %add212 to i64, !dbg !2177
  %arrayidx214 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %162, i64 %idxprom213, !dbg !2177
  %imag215 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx214, i32 0, i32 1, !dbg !2188
  %167 = load double, double* %imag215, align 8, !dbg !2188
  store double %167, double* %x22_imag, align 8, !dbg !2189
  %168 = load double, double* %x12_real, align 8, !dbg !2190
  %169 = load double, double* %x22_real, align 8, !dbg !2191
  %add216 = fadd contract double %168, %169, !dbg !2192
  %170 = load %struct.dcomplex*, %struct.dcomplex** %gty1.addr, align 8, !dbg !2193
  %171 = load i32, i32* %i, align 4, !dbg !2194
  %172 = load i32, i32* %i21, align 4, !dbg !2195
  %173 = load i32, i32* %k1, align 4, !dbg !2196
  %add217 = add nsw i32 %172, %173, !dbg !2197
  %mul218 = mul nsw i32 %add217, 256, !dbg !2198
  %add219 = add nsw i32 %171, %mul218, !dbg !2199
  %174 = load i32, i32* %k, align 4, !dbg !2200
  %mul220 = mul nsw i32 %174, 256, !dbg !2201
  %mul221 = mul nsw i32 %mul220, 256, !dbg !2202
  %add222 = add nsw i32 %add219, %mul221, !dbg !2203
  %idxprom223 = sext i32 %add222 to i64, !dbg !2193
  %arrayidx224 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %170, i64 %idxprom223, !dbg !2193
  %real225 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx224, i32 0, i32 0, !dbg !2204
  store double %add216, double* %real225, align 8, !dbg !2205
  %175 = load double, double* %x12_imag, align 8, !dbg !2206
  %176 = load double, double* %x22_imag, align 8, !dbg !2207
  %add226 = fadd contract double %175, %176, !dbg !2208
  %177 = load %struct.dcomplex*, %struct.dcomplex** %gty1.addr, align 8, !dbg !2209
  %178 = load i32, i32* %i, align 4, !dbg !2210
  %179 = load i32, i32* %i21, align 4, !dbg !2211
  %180 = load i32, i32* %k1, align 4, !dbg !2212
  %add227 = add nsw i32 %179, %180, !dbg !2213
  %mul228 = mul nsw i32 %add227, 256, !dbg !2214
  %add229 = add nsw i32 %178, %mul228, !dbg !2215
  %181 = load i32, i32* %k, align 4, !dbg !2216
  %mul230 = mul nsw i32 %181, 256, !dbg !2217
  %mul231 = mul nsw i32 %mul230, 256, !dbg !2218
  %add232 = add nsw i32 %add229, %mul231, !dbg !2219
  %idxprom233 = sext i32 %add232 to i64, !dbg !2209
  %arrayidx234 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %177, i64 %idxprom233, !dbg !2209
  %imag235 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx234, i32 0, i32 1, !dbg !2220
  store double %add226, double* %imag235, align 8, !dbg !2221
  %182 = load double, double* %x12_real, align 8, !dbg !2222
  %183 = load double, double* %x22_real, align 8, !dbg !2223
  %sub236 = fsub contract double %182, %183, !dbg !2224
  store double %sub236, double* %temp2_real, align 8, !dbg !2225
  %184 = load double, double* %x12_imag, align 8, !dbg !2226
  %185 = load double, double* %x22_imag, align 8, !dbg !2227
  %sub237 = fsub contract double %184, %185, !dbg !2228
  store double %sub237, double* %temp2_imag, align 8, !dbg !2229
  %186 = load double, double* %uu2_real, align 8, !dbg !2230
  %187 = load double, double* %temp2_real, align 8, !dbg !2231
  %mul238 = fmul contract double %186, %187, !dbg !2232
  %188 = load double, double* %uu2_imag, align 8, !dbg !2233
  %189 = load double, double* %temp2_imag, align 8, !dbg !2234
  %mul239 = fmul contract double %188, %189, !dbg !2235
  %sub240 = fsub contract double %mul238, %mul239, !dbg !2236
  %190 = load %struct.dcomplex*, %struct.dcomplex** %gty1.addr, align 8, !dbg !2237
  %191 = load i32, i32* %i, align 4, !dbg !2238
  %192 = load i32, i32* %i22, align 4, !dbg !2239
  %193 = load i32, i32* %k1, align 4, !dbg !2240
  %add241 = add nsw i32 %192, %193, !dbg !2241
  %mul242 = mul nsw i32 %add241, 256, !dbg !2242
  %add243 = add nsw i32 %191, %mul242, !dbg !2243
  %194 = load i32, i32* %k, align 4, !dbg !2244
  %mul244 = mul nsw i32 %194, 256, !dbg !2245
  %mul245 = mul nsw i32 %mul244, 256, !dbg !2246
  %add246 = add nsw i32 %add243, %mul245, !dbg !2247
  %idxprom247 = sext i32 %add246 to i64, !dbg !2237
  %arrayidx248 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %190, i64 %idxprom247, !dbg !2237
  %real249 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx248, i32 0, i32 0, !dbg !2248
  store double %sub240, double* %real249, align 8, !dbg !2249
  %195 = load double, double* %uu2_real, align 8, !dbg !2250
  %196 = load double, double* %temp2_imag, align 8, !dbg !2251
  %mul250 = fmul contract double %195, %196, !dbg !2252
  %197 = load double, double* %uu2_imag, align 8, !dbg !2253
  %198 = load double, double* %temp2_real, align 8, !dbg !2254
  %mul251 = fmul contract double %197, %198, !dbg !2255
  %add252 = fadd contract double %mul250, %mul251, !dbg !2256
  %199 = load %struct.dcomplex*, %struct.dcomplex** %gty1.addr, align 8, !dbg !2257
  %200 = load i32, i32* %i, align 4, !dbg !2258
  %201 = load i32, i32* %i22, align 4, !dbg !2259
  %202 = load i32, i32* %k1, align 4, !dbg !2260
  %add253 = add nsw i32 %201, %202, !dbg !2261
  %mul254 = mul nsw i32 %add253, 256, !dbg !2262
  %add255 = add nsw i32 %200, %mul254, !dbg !2263
  %203 = load i32, i32* %k, align 4, !dbg !2264
  %mul256 = mul nsw i32 %203, 256, !dbg !2265
  %mul257 = mul nsw i32 %mul256, 256, !dbg !2266
  %add258 = add nsw i32 %add255, %mul257, !dbg !2267
  %idxprom259 = sext i32 %add258 to i64, !dbg !2257
  %arrayidx260 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %199, i64 %idxprom259, !dbg !2257
  %imag261 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx260, i32 0, i32 1, !dbg !2268
  store double %add252, double* %imag261, align 8, !dbg !2269
  br label %for.inc262, !dbg !2270

for.inc262:                                       ; preds = %for.body165
  %204 = load i32, i32* %k1, align 4, !dbg !2271
  %inc263 = add nsw i32 %204, 1, !dbg !2271
  store i32 %inc263, i32* %k1, align 4, !dbg !2271
  br label %for.cond162, !dbg !2272, !llvm.loop !2273

for.end264:                                       ; preds = %for.cond162
  br label %for.inc265, !dbg !2275

for.inc265:                                       ; preds = %for.end264
  %205 = load i32, i32* %i1, align 4, !dbg !2276
  %inc266 = add nsw i32 %205, 1, !dbg !2276
  store i32 %inc266, i32* %i1, align 4, !dbg !2276
  br label %for.cond158, !dbg !2277, !llvm.loop !2278

for.end267:                                       ; preds = %for.cond158
  br label %if.end268

if.end268:                                        ; preds = %for.end267, %for.end150
  br label %for.inc269, !dbg !2280

for.inc269:                                       ; preds = %if.end268
  %206 = load i32, i32* %l, align 4, !dbg !2281
  %add270 = add nsw i32 %206, 2, !dbg !2281
  store i32 %add270, i32* %l, align 4, !dbg !2281
  br label %for.cond, !dbg !2282, !llvm.loop !2283

for.end271:                                       ; preds = %if.then, %for.cond
  ret void, !dbg !2285
}

; Function Attrs: convergent noinline nounwind
define dso_local void @_Z19cffts2_gpu_kernel_3P8dcomplexS0_(%struct.dcomplex* %x_out, %struct.dcomplex* %y0) #0 !dbg !2286 {
entry:
  %x_out.addr = alloca %struct.dcomplex*, align 8
  %y0.addr = alloca %struct.dcomplex*, align 8
  %x_y_z = alloca i32, align 4
  store %struct.dcomplex* %x_out, %struct.dcomplex** %x_out.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %x_out.addr, metadata !2287, metadata !DIExpression()), !dbg !2288
  store %struct.dcomplex* %y0, %struct.dcomplex** %y0.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %y0.addr, metadata !2289, metadata !DIExpression()), !dbg !2290
  call void @llvm.dbg.declare(metadata i32* %x_y_z, metadata !2291, metadata !DIExpression()), !dbg !2292
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #8, !dbg !2293, !range !848
  %1 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #8, !dbg !2295, !range !893
  %mul = mul i32 %0, %1, !dbg !2297
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #8, !dbg !2298, !range !923
  %add = add i32 %mul, %2, !dbg !2300
  store i32 %add, i32* %x_y_z, align 4, !dbg !2292
  %3 = load i32, i32* %x_y_z, align 4, !dbg !2301
  %cmp = icmp sge i32 %3, 8388608, !dbg !2303
  br i1 %cmp, label %if.then, label %if.end, !dbg !2304

if.then:                                          ; preds = %entry
  br label %return, !dbg !2305

if.end:                                           ; preds = %entry
  %4 = load %struct.dcomplex*, %struct.dcomplex** %y0.addr, align 8, !dbg !2307
  %5 = load i32, i32* %x_y_z, align 4, !dbg !2308
  %idxprom = sext i32 %5 to i64, !dbg !2307
  %arrayidx = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %4, i64 %idxprom, !dbg !2307
  %real = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx, i32 0, i32 0, !dbg !2309
  %6 = load double, double* %real, align 8, !dbg !2309
  %7 = load %struct.dcomplex*, %struct.dcomplex** %x_out.addr, align 8, !dbg !2310
  %8 = load i32, i32* %x_y_z, align 4, !dbg !2311
  %idxprom3 = sext i32 %8 to i64, !dbg !2310
  %arrayidx4 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %7, i64 %idxprom3, !dbg !2310
  %real5 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx4, i32 0, i32 0, !dbg !2312
  store double %6, double* %real5, align 8, !dbg !2313
  %9 = load %struct.dcomplex*, %struct.dcomplex** %y0.addr, align 8, !dbg !2314
  %10 = load i32, i32* %x_y_z, align 4, !dbg !2315
  %idxprom6 = sext i32 %10 to i64, !dbg !2314
  %arrayidx7 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %9, i64 %idxprom6, !dbg !2314
  %imag = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx7, i32 0, i32 1, !dbg !2316
  %11 = load double, double* %imag, align 8, !dbg !2316
  %12 = load %struct.dcomplex*, %struct.dcomplex** %x_out.addr, align 8, !dbg !2317
  %13 = load i32, i32* %x_y_z, align 4, !dbg !2318
  %idxprom8 = sext i32 %13 to i64, !dbg !2317
  %arrayidx9 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %12, i64 %idxprom8, !dbg !2317
  %imag10 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx9, i32 0, i32 1, !dbg !2319
  store double %11, double* %imag10, align 8, !dbg !2320
  br label %return, !dbg !2321

return:                                           ; preds = %if.end, %if.then
  ret void, !dbg !2321
}

; Function Attrs: convergent noinline nounwind
define dso_local void @_Z23cffts3_gpu_cfftz_deviceiiiP8dcomplexS0_S0_ii(i32 %is, i32 %m, i32 %n, %struct.dcomplex* %x, %struct.dcomplex* %y, %struct.dcomplex* %u_device, i32 %index_arg, i32 %size_arg) #0 !dbg !2322 {
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
  call void @llvm.dbg.declare(metadata i32* %is.addr, metadata !2325, metadata !DIExpression()), !dbg !2326
  store i32 %m, i32* %m.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %m.addr, metadata !2327, metadata !DIExpression()), !dbg !2328
  store i32 %n, i32* %n.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %n.addr, metadata !2329, metadata !DIExpression()), !dbg !2330
  store %struct.dcomplex* %x, %struct.dcomplex** %x.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %x.addr, metadata !2331, metadata !DIExpression()), !dbg !2332
  store %struct.dcomplex* %y, %struct.dcomplex** %y.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %y.addr, metadata !2333, metadata !DIExpression()), !dbg !2334
  store %struct.dcomplex* %u_device, %struct.dcomplex** %u_device.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %u_device.addr, metadata !2335, metadata !DIExpression()), !dbg !2336
  store i32 %index_arg, i32* %index_arg.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %index_arg.addr, metadata !2337, metadata !DIExpression()), !dbg !2338
  store i32 %size_arg, i32* %size_arg.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %size_arg.addr, metadata !2339, metadata !DIExpression()), !dbg !2340
  call void @llvm.dbg.declare(metadata i32* %j, metadata !2341, metadata !DIExpression()), !dbg !2342
  call void @llvm.dbg.declare(metadata i32* %l, metadata !2343, metadata !DIExpression()), !dbg !2344
  store i32 1, i32* %l, align 4, !dbg !2345
  br label %for.cond, !dbg !2347

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, i32* %l, align 4, !dbg !2348
  %1 = load i32, i32* %m.addr, align 4, !dbg !2350
  %cmp = icmp sle i32 %0, %1, !dbg !2351
  br i1 %cmp, label %for.body, label %for.end, !dbg !2352

for.body:                                         ; preds = %for.cond
  %2 = load i32, i32* %is.addr, align 4, !dbg !2353
  %3 = load i32, i32* %l, align 4, !dbg !2355
  %4 = load i32, i32* %m.addr, align 4, !dbg !2356
  %5 = load i32, i32* %n.addr, align 4, !dbg !2357
  %6 = load %struct.dcomplex*, %struct.dcomplex** %u_device.addr, align 8, !dbg !2358
  %7 = load %struct.dcomplex*, %struct.dcomplex** %x.addr, align 8, !dbg !2359
  %8 = load %struct.dcomplex*, %struct.dcomplex** %y.addr, align 8, !dbg !2360
  %9 = load i32, i32* %index_arg.addr, align 4, !dbg !2361
  %10 = load i32, i32* %size_arg.addr, align 4, !dbg !2362
  call void @_Z23cffts3_gpu_fftz2_deviceiiiiP8dcomplexS0_S0_ii(i32 %2, i32 %3, i32 %4, i32 %5, %struct.dcomplex* %6, %struct.dcomplex* %7, %struct.dcomplex* %8, i32 %9, i32 %10) #3, !dbg !2363
  %11 = load i32, i32* %l, align 4, !dbg !2364
  %12 = load i32, i32* %m.addr, align 4, !dbg !2366
  %cmp1 = icmp eq i32 %11, %12, !dbg !2367
  br i1 %cmp1, label %if.then, label %if.end, !dbg !2368

if.then:                                          ; preds = %for.body
  br label %for.end, !dbg !2369

if.end:                                           ; preds = %for.body
  %13 = load i32, i32* %is.addr, align 4, !dbg !2371
  %14 = load i32, i32* %l, align 4, !dbg !2372
  %add = add nsw i32 %14, 1, !dbg !2373
  %15 = load i32, i32* %m.addr, align 4, !dbg !2374
  %16 = load i32, i32* %n.addr, align 4, !dbg !2375
  %17 = load %struct.dcomplex*, %struct.dcomplex** %u_device.addr, align 8, !dbg !2376
  %18 = load %struct.dcomplex*, %struct.dcomplex** %y.addr, align 8, !dbg !2377
  %19 = load %struct.dcomplex*, %struct.dcomplex** %x.addr, align 8, !dbg !2378
  %20 = load i32, i32* %index_arg.addr, align 4, !dbg !2379
  %21 = load i32, i32* %size_arg.addr, align 4, !dbg !2380
  call void @_Z23cffts3_gpu_fftz2_deviceiiiiP8dcomplexS0_S0_ii(i32 %13, i32 %add, i32 %15, i32 %16, %struct.dcomplex* %17, %struct.dcomplex* %18, %struct.dcomplex* %19, i32 %20, i32 %21) #3, !dbg !2381
  br label %for.inc, !dbg !2382

for.inc:                                          ; preds = %if.end
  %22 = load i32, i32* %l, align 4, !dbg !2383
  %add2 = add nsw i32 %22, 2, !dbg !2383
  store i32 %add2, i32* %l, align 4, !dbg !2383
  br label %for.cond, !dbg !2384, !llvm.loop !2385

for.end:                                          ; preds = %if.then, %for.cond
  %23 = load i32, i32* %m.addr, align 4, !dbg !2387
  %rem = srem i32 %23, 2, !dbg !2389
  %cmp3 = icmp eq i32 %rem, 1, !dbg !2390
  br i1 %cmp3, label %if.then4, label %if.end25, !dbg !2391

if.then4:                                         ; preds = %for.end
  store i32 0, i32* %j, align 4, !dbg !2392
  br label %for.cond5, !dbg !2395

for.cond5:                                        ; preds = %for.inc23, %if.then4
  %24 = load i32, i32* %j, align 4, !dbg !2396
  %25 = load i32, i32* %n.addr, align 4, !dbg !2398
  %cmp6 = icmp slt i32 %24, %25, !dbg !2399
  br i1 %cmp6, label %for.body7, label %for.end24, !dbg !2400

for.body7:                                        ; preds = %for.cond5
  %26 = load %struct.dcomplex*, %struct.dcomplex** %y.addr, align 8, !dbg !2401
  %27 = load i32, i32* %j, align 4, !dbg !2403
  %28 = load i32, i32* %size_arg.addr, align 4, !dbg !2404
  %mul = mul nsw i32 %27, %28, !dbg !2405
  %29 = load i32, i32* %index_arg.addr, align 4, !dbg !2406
  %add8 = add nsw i32 %mul, %29, !dbg !2407
  %idxprom = sext i32 %add8 to i64, !dbg !2401
  %arrayidx = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %26, i64 %idxprom, !dbg !2401
  %real = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx, i32 0, i32 0, !dbg !2408
  %30 = load double, double* %real, align 8, !dbg !2408
  %31 = load %struct.dcomplex*, %struct.dcomplex** %x.addr, align 8, !dbg !2409
  %32 = load i32, i32* %j, align 4, !dbg !2410
  %33 = load i32, i32* %size_arg.addr, align 4, !dbg !2411
  %mul9 = mul nsw i32 %32, %33, !dbg !2412
  %34 = load i32, i32* %index_arg.addr, align 4, !dbg !2413
  %add10 = add nsw i32 %mul9, %34, !dbg !2414
  %idxprom11 = sext i32 %add10 to i64, !dbg !2409
  %arrayidx12 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %31, i64 %idxprom11, !dbg !2409
  %real13 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx12, i32 0, i32 0, !dbg !2415
  store double %30, double* %real13, align 8, !dbg !2416
  %35 = load %struct.dcomplex*, %struct.dcomplex** %y.addr, align 8, !dbg !2417
  %36 = load i32, i32* %j, align 4, !dbg !2418
  %37 = load i32, i32* %size_arg.addr, align 4, !dbg !2419
  %mul14 = mul nsw i32 %36, %37, !dbg !2420
  %38 = load i32, i32* %index_arg.addr, align 4, !dbg !2421
  %add15 = add nsw i32 %mul14, %38, !dbg !2422
  %idxprom16 = sext i32 %add15 to i64, !dbg !2417
  %arrayidx17 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %35, i64 %idxprom16, !dbg !2417
  %imag = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx17, i32 0, i32 1, !dbg !2423
  %39 = load double, double* %imag, align 8, !dbg !2423
  %40 = load %struct.dcomplex*, %struct.dcomplex** %x.addr, align 8, !dbg !2424
  %41 = load i32, i32* %j, align 4, !dbg !2425
  %42 = load i32, i32* %size_arg.addr, align 4, !dbg !2426
  %mul18 = mul nsw i32 %41, %42, !dbg !2427
  %43 = load i32, i32* %index_arg.addr, align 4, !dbg !2428
  %add19 = add nsw i32 %mul18, %43, !dbg !2429
  %idxprom20 = sext i32 %add19 to i64, !dbg !2424
  %arrayidx21 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %40, i64 %idxprom20, !dbg !2424
  %imag22 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx21, i32 0, i32 1, !dbg !2430
  store double %39, double* %imag22, align 8, !dbg !2431
  br label %for.inc23, !dbg !2432

for.inc23:                                        ; preds = %for.body7
  %44 = load i32, i32* %j, align 4, !dbg !2433
  %inc = add nsw i32 %44, 1, !dbg !2433
  store i32 %inc, i32* %j, align 4, !dbg !2433
  br label %for.cond5, !dbg !2434, !llvm.loop !2435

for.end24:                                        ; preds = %for.cond5
  br label %if.end25, !dbg !2437

if.end25:                                         ; preds = %for.end24, %for.end
  ret void, !dbg !2438
}

; Function Attrs: convergent noinline nounwind
define dso_local void @_Z23cffts3_gpu_fftz2_deviceiiiiP8dcomplexS0_S0_ii(i32 %is, i32 %l, i32 %m, i32 %n, %struct.dcomplex* %u, %struct.dcomplex* %x, %struct.dcomplex* %y, i32 %index_arg, i32 %size_arg) #0 !dbg !2439 {
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
  call void @llvm.dbg.declare(metadata i32* %is.addr, metadata !2442, metadata !DIExpression()), !dbg !2443
  store i32 %l, i32* %l.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %l.addr, metadata !2444, metadata !DIExpression()), !dbg !2445
  store i32 %m, i32* %m.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %m.addr, metadata !2446, metadata !DIExpression()), !dbg !2447
  store i32 %n, i32* %n.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %n.addr, metadata !2448, metadata !DIExpression()), !dbg !2449
  store %struct.dcomplex* %u, %struct.dcomplex** %u.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %u.addr, metadata !2450, metadata !DIExpression()), !dbg !2451
  store %struct.dcomplex* %x, %struct.dcomplex** %x.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %x.addr, metadata !2452, metadata !DIExpression()), !dbg !2453
  store %struct.dcomplex* %y, %struct.dcomplex** %y.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %y.addr, metadata !2454, metadata !DIExpression()), !dbg !2455
  store i32 %index_arg, i32* %index_arg.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %index_arg.addr, metadata !2456, metadata !DIExpression()), !dbg !2457
  store i32 %size_arg, i32* %size_arg.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %size_arg.addr, metadata !2458, metadata !DIExpression()), !dbg !2459
  call void @llvm.dbg.declare(metadata i32* %k, metadata !2460, metadata !DIExpression()), !dbg !2461
  call void @llvm.dbg.declare(metadata i32* %n1, metadata !2462, metadata !DIExpression()), !dbg !2463
  call void @llvm.dbg.declare(metadata i32* %li, metadata !2464, metadata !DIExpression()), !dbg !2465
  call void @llvm.dbg.declare(metadata i32* %lj, metadata !2466, metadata !DIExpression()), !dbg !2467
  call void @llvm.dbg.declare(metadata i32* %lk, metadata !2468, metadata !DIExpression()), !dbg !2469
  call void @llvm.dbg.declare(metadata i32* %ku, metadata !2470, metadata !DIExpression()), !dbg !2471
  call void @llvm.dbg.declare(metadata i32* %i, metadata !2472, metadata !DIExpression()), !dbg !2473
  call void @llvm.dbg.declare(metadata i32* %i11, metadata !2474, metadata !DIExpression()), !dbg !2475
  call void @llvm.dbg.declare(metadata i32* %i12, metadata !2476, metadata !DIExpression()), !dbg !2477
  call void @llvm.dbg.declare(metadata i32* %i21, metadata !2478, metadata !DIExpression()), !dbg !2479
  call void @llvm.dbg.declare(metadata i32* %i22, metadata !2480, metadata !DIExpression()), !dbg !2481
  call void @llvm.dbg.declare(metadata double* %x11real, metadata !2482, metadata !DIExpression()), !dbg !2483
  call void @llvm.dbg.declare(metadata double* %x11imag, metadata !2484, metadata !DIExpression()), !dbg !2485
  call void @llvm.dbg.declare(metadata double* %x21real, metadata !2486, metadata !DIExpression()), !dbg !2487
  call void @llvm.dbg.declare(metadata double* %x21imag, metadata !2488, metadata !DIExpression()), !dbg !2489
  call void @llvm.dbg.declare(metadata %struct.dcomplex* %u1, metadata !2490, metadata !DIExpression()), !dbg !2491
  %0 = load i32, i32* %n.addr, align 4, !dbg !2492
  %div = sdiv i32 %0, 2, !dbg !2493
  store i32 %div, i32* %n1, align 4, !dbg !2494
  %1 = load i32, i32* %l.addr, align 4, !dbg !2495
  %sub = sub nsw i32 %1, 1, !dbg !2496
  %shl = shl i32 1, %sub, !dbg !2497
  store i32 %shl, i32* %lk, align 4, !dbg !2498
  %2 = load i32, i32* %m.addr, align 4, !dbg !2499
  %3 = load i32, i32* %l.addr, align 4, !dbg !2500
  %sub1 = sub nsw i32 %2, %3, !dbg !2501
  %shl2 = shl i32 1, %sub1, !dbg !2502
  store i32 %shl2, i32* %li, align 4, !dbg !2503
  %4 = load i32, i32* %lk, align 4, !dbg !2504
  %mul = mul nsw i32 2, %4, !dbg !2505
  store i32 %mul, i32* %lj, align 4, !dbg !2506
  %5 = load i32, i32* %li, align 4, !dbg !2507
  store i32 %5, i32* %ku, align 4, !dbg !2508
  store i32 0, i32* %i, align 4, !dbg !2509
  br label %for.cond, !dbg !2511

for.cond:                                         ; preds = %for.inc91, %entry
  %6 = load i32, i32* %i, align 4, !dbg !2512
  %7 = load i32, i32* %li, align 4, !dbg !2514
  %cmp = icmp slt i32 %6, %7, !dbg !2515
  br i1 %cmp, label %for.body, label %for.end93, !dbg !2516

for.body:                                         ; preds = %for.cond
  %8 = load i32, i32* %i, align 4, !dbg !2517
  %9 = load i32, i32* %lk, align 4, !dbg !2519
  %mul3 = mul nsw i32 %8, %9, !dbg !2520
  store i32 %mul3, i32* %i11, align 4, !dbg !2521
  %10 = load i32, i32* %i11, align 4, !dbg !2522
  %11 = load i32, i32* %n1, align 4, !dbg !2523
  %add = add nsw i32 %10, %11, !dbg !2524
  store i32 %add, i32* %i12, align 4, !dbg !2525
  %12 = load i32, i32* %i, align 4, !dbg !2526
  %13 = load i32, i32* %lj, align 4, !dbg !2527
  %mul4 = mul nsw i32 %12, %13, !dbg !2528
  store i32 %mul4, i32* %i21, align 4, !dbg !2529
  %14 = load i32, i32* %i21, align 4, !dbg !2530
  %15 = load i32, i32* %lk, align 4, !dbg !2531
  %add5 = add nsw i32 %14, %15, !dbg !2532
  store i32 %add5, i32* %i22, align 4, !dbg !2533
  %16 = load i32, i32* %is.addr, align 4, !dbg !2534
  %cmp6 = icmp sge i32 %16, 1, !dbg !2536
  br i1 %cmp6, label %if.then, label %if.else, !dbg !2537

if.then:                                          ; preds = %for.body
  %17 = load %struct.dcomplex*, %struct.dcomplex** %u.addr, align 8, !dbg !2538
  %18 = load i32, i32* %ku, align 4, !dbg !2540
  %19 = load i32, i32* %i, align 4, !dbg !2541
  %add7 = add nsw i32 %18, %19, !dbg !2542
  %idxprom = sext i32 %add7 to i64, !dbg !2538
  %arrayidx = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %17, i64 %idxprom, !dbg !2538
  %real = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx, i32 0, i32 0, !dbg !2543
  %20 = load double, double* %real, align 8, !dbg !2543
  %real8 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %u1, i32 0, i32 0, !dbg !2544
  store double %20, double* %real8, align 8, !dbg !2545
  %21 = load %struct.dcomplex*, %struct.dcomplex** %u.addr, align 8, !dbg !2546
  %22 = load i32, i32* %ku, align 4, !dbg !2547
  %23 = load i32, i32* %i, align 4, !dbg !2548
  %add9 = add nsw i32 %22, %23, !dbg !2549
  %idxprom10 = sext i32 %add9 to i64, !dbg !2546
  %arrayidx11 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %21, i64 %idxprom10, !dbg !2546
  %imag = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx11, i32 0, i32 1, !dbg !2550
  %24 = load double, double* %imag, align 8, !dbg !2550
  %imag12 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %u1, i32 0, i32 1, !dbg !2551
  store double %24, double* %imag12, align 8, !dbg !2552
  br label %if.end, !dbg !2553

if.else:                                          ; preds = %for.body
  %25 = load %struct.dcomplex*, %struct.dcomplex** %u.addr, align 8, !dbg !2554
  %26 = load i32, i32* %ku, align 4, !dbg !2556
  %27 = load i32, i32* %i, align 4, !dbg !2557
  %add13 = add nsw i32 %26, %27, !dbg !2558
  %idxprom14 = sext i32 %add13 to i64, !dbg !2554
  %arrayidx15 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %25, i64 %idxprom14, !dbg !2554
  %real16 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx15, i32 0, i32 0, !dbg !2559
  %28 = load double, double* %real16, align 8, !dbg !2559
  %real17 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %u1, i32 0, i32 0, !dbg !2560
  store double %28, double* %real17, align 8, !dbg !2561
  %29 = load %struct.dcomplex*, %struct.dcomplex** %u.addr, align 8, !dbg !2562
  %30 = load i32, i32* %ku, align 4, !dbg !2563
  %31 = load i32, i32* %i, align 4, !dbg !2564
  %add18 = add nsw i32 %30, %31, !dbg !2565
  %idxprom19 = sext i32 %add18 to i64, !dbg !2562
  %arrayidx20 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %29, i64 %idxprom19, !dbg !2562
  %imag21 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx20, i32 0, i32 1, !dbg !2566
  %32 = load double, double* %imag21, align 8, !dbg !2566
  %sub22 = fsub double -0.000000e+00, %32, !dbg !2567
  %imag23 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %u1, i32 0, i32 1, !dbg !2568
  store double %sub22, double* %imag23, align 8, !dbg !2569
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  store i32 0, i32* %k, align 4, !dbg !2570
  br label %for.cond24, !dbg !2572

for.cond24:                                       ; preds = %for.inc, %if.end
  %33 = load i32, i32* %k, align 4, !dbg !2573
  %34 = load i32, i32* %lk, align 4, !dbg !2575
  %cmp25 = icmp slt i32 %33, %34, !dbg !2576
  br i1 %cmp25, label %for.body26, label %for.end, !dbg !2577

for.body26:                                       ; preds = %for.cond24
  %35 = load %struct.dcomplex*, %struct.dcomplex** %x.addr, align 8, !dbg !2578
  %36 = load i32, i32* %i11, align 4, !dbg !2580
  %37 = load i32, i32* %k, align 4, !dbg !2581
  %add27 = add nsw i32 %36, %37, !dbg !2582
  %38 = load i32, i32* %size_arg.addr, align 4, !dbg !2583
  %mul28 = mul nsw i32 %add27, %38, !dbg !2584
  %39 = load i32, i32* %index_arg.addr, align 4, !dbg !2585
  %add29 = add nsw i32 %mul28, %39, !dbg !2586
  %idxprom30 = sext i32 %add29 to i64, !dbg !2578
  %arrayidx31 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %35, i64 %idxprom30, !dbg !2578
  %real32 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx31, i32 0, i32 0, !dbg !2587
  %40 = load double, double* %real32, align 8, !dbg !2587
  store double %40, double* %x11real, align 8, !dbg !2588
  %41 = load %struct.dcomplex*, %struct.dcomplex** %x.addr, align 8, !dbg !2589
  %42 = load i32, i32* %i11, align 4, !dbg !2590
  %43 = load i32, i32* %k, align 4, !dbg !2591
  %add33 = add nsw i32 %42, %43, !dbg !2592
  %44 = load i32, i32* %size_arg.addr, align 4, !dbg !2593
  %mul34 = mul nsw i32 %add33, %44, !dbg !2594
  %45 = load i32, i32* %index_arg.addr, align 4, !dbg !2595
  %add35 = add nsw i32 %mul34, %45, !dbg !2596
  %idxprom36 = sext i32 %add35 to i64, !dbg !2589
  %arrayidx37 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %41, i64 %idxprom36, !dbg !2589
  %imag38 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx37, i32 0, i32 1, !dbg !2597
  %46 = load double, double* %imag38, align 8, !dbg !2597
  store double %46, double* %x11imag, align 8, !dbg !2598
  %47 = load %struct.dcomplex*, %struct.dcomplex** %x.addr, align 8, !dbg !2599
  %48 = load i32, i32* %i12, align 4, !dbg !2600
  %49 = load i32, i32* %k, align 4, !dbg !2601
  %add39 = add nsw i32 %48, %49, !dbg !2602
  %50 = load i32, i32* %size_arg.addr, align 4, !dbg !2603
  %mul40 = mul nsw i32 %add39, %50, !dbg !2604
  %51 = load i32, i32* %index_arg.addr, align 4, !dbg !2605
  %add41 = add nsw i32 %mul40, %51, !dbg !2606
  %idxprom42 = sext i32 %add41 to i64, !dbg !2599
  %arrayidx43 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %47, i64 %idxprom42, !dbg !2599
  %real44 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx43, i32 0, i32 0, !dbg !2607
  %52 = load double, double* %real44, align 8, !dbg !2607
  store double %52, double* %x21real, align 8, !dbg !2608
  %53 = load %struct.dcomplex*, %struct.dcomplex** %x.addr, align 8, !dbg !2609
  %54 = load i32, i32* %i12, align 4, !dbg !2610
  %55 = load i32, i32* %k, align 4, !dbg !2611
  %add45 = add nsw i32 %54, %55, !dbg !2612
  %56 = load i32, i32* %size_arg.addr, align 4, !dbg !2613
  %mul46 = mul nsw i32 %add45, %56, !dbg !2614
  %57 = load i32, i32* %index_arg.addr, align 4, !dbg !2615
  %add47 = add nsw i32 %mul46, %57, !dbg !2616
  %idxprom48 = sext i32 %add47 to i64, !dbg !2609
  %arrayidx49 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %53, i64 %idxprom48, !dbg !2609
  %imag50 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx49, i32 0, i32 1, !dbg !2617
  %58 = load double, double* %imag50, align 8, !dbg !2617
  store double %58, double* %x21imag, align 8, !dbg !2618
  %59 = load double, double* %x11real, align 8, !dbg !2619
  %60 = load double, double* %x21real, align 8, !dbg !2620
  %add51 = fadd contract double %59, %60, !dbg !2621
  %61 = load %struct.dcomplex*, %struct.dcomplex** %y.addr, align 8, !dbg !2622
  %62 = load i32, i32* %i21, align 4, !dbg !2623
  %63 = load i32, i32* %k, align 4, !dbg !2624
  %add52 = add nsw i32 %62, %63, !dbg !2625
  %64 = load i32, i32* %size_arg.addr, align 4, !dbg !2626
  %mul53 = mul nsw i32 %add52, %64, !dbg !2627
  %65 = load i32, i32* %index_arg.addr, align 4, !dbg !2628
  %add54 = add nsw i32 %mul53, %65, !dbg !2629
  %idxprom55 = sext i32 %add54 to i64, !dbg !2622
  %arrayidx56 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %61, i64 %idxprom55, !dbg !2622
  %real57 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx56, i32 0, i32 0, !dbg !2630
  store double %add51, double* %real57, align 8, !dbg !2631
  %66 = load double, double* %x11imag, align 8, !dbg !2632
  %67 = load double, double* %x21imag, align 8, !dbg !2633
  %add58 = fadd contract double %66, %67, !dbg !2634
  %68 = load %struct.dcomplex*, %struct.dcomplex** %y.addr, align 8, !dbg !2635
  %69 = load i32, i32* %i21, align 4, !dbg !2636
  %70 = load i32, i32* %k, align 4, !dbg !2637
  %add59 = add nsw i32 %69, %70, !dbg !2638
  %71 = load i32, i32* %size_arg.addr, align 4, !dbg !2639
  %mul60 = mul nsw i32 %add59, %71, !dbg !2640
  %72 = load i32, i32* %index_arg.addr, align 4, !dbg !2641
  %add61 = add nsw i32 %mul60, %72, !dbg !2642
  %idxprom62 = sext i32 %add61 to i64, !dbg !2635
  %arrayidx63 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %68, i64 %idxprom62, !dbg !2635
  %imag64 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx63, i32 0, i32 1, !dbg !2643
  store double %add58, double* %imag64, align 8, !dbg !2644
  %real65 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %u1, i32 0, i32 0, !dbg !2645
  %73 = load double, double* %real65, align 8, !dbg !2645
  %74 = load double, double* %x11real, align 8, !dbg !2646
  %75 = load double, double* %x21real, align 8, !dbg !2647
  %sub66 = fsub contract double %74, %75, !dbg !2648
  %mul67 = fmul contract double %73, %sub66, !dbg !2649
  %imag68 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %u1, i32 0, i32 1, !dbg !2650
  %76 = load double, double* %imag68, align 8, !dbg !2650
  %77 = load double, double* %x11imag, align 8, !dbg !2651
  %78 = load double, double* %x21imag, align 8, !dbg !2652
  %sub69 = fsub contract double %77, %78, !dbg !2653
  %mul70 = fmul contract double %76, %sub69, !dbg !2654
  %sub71 = fsub contract double %mul67, %mul70, !dbg !2655
  %79 = load %struct.dcomplex*, %struct.dcomplex** %y.addr, align 8, !dbg !2656
  %80 = load i32, i32* %i22, align 4, !dbg !2657
  %81 = load i32, i32* %k, align 4, !dbg !2658
  %add72 = add nsw i32 %80, %81, !dbg !2659
  %82 = load i32, i32* %size_arg.addr, align 4, !dbg !2660
  %mul73 = mul nsw i32 %add72, %82, !dbg !2661
  %83 = load i32, i32* %index_arg.addr, align 4, !dbg !2662
  %add74 = add nsw i32 %mul73, %83, !dbg !2663
  %idxprom75 = sext i32 %add74 to i64, !dbg !2656
  %arrayidx76 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %79, i64 %idxprom75, !dbg !2656
  %real77 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx76, i32 0, i32 0, !dbg !2664
  store double %sub71, double* %real77, align 8, !dbg !2665
  %real78 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %u1, i32 0, i32 0, !dbg !2666
  %84 = load double, double* %real78, align 8, !dbg !2666
  %85 = load double, double* %x11imag, align 8, !dbg !2667
  %86 = load double, double* %x21imag, align 8, !dbg !2668
  %sub79 = fsub contract double %85, %86, !dbg !2669
  %mul80 = fmul contract double %84, %sub79, !dbg !2670
  %imag81 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %u1, i32 0, i32 1, !dbg !2671
  %87 = load double, double* %imag81, align 8, !dbg !2671
  %88 = load double, double* %x11real, align 8, !dbg !2672
  %89 = load double, double* %x21real, align 8, !dbg !2673
  %sub82 = fsub contract double %88, %89, !dbg !2674
  %mul83 = fmul contract double %87, %sub82, !dbg !2675
  %add84 = fadd contract double %mul80, %mul83, !dbg !2676
  %90 = load %struct.dcomplex*, %struct.dcomplex** %y.addr, align 8, !dbg !2677
  %91 = load i32, i32* %i22, align 4, !dbg !2678
  %92 = load i32, i32* %k, align 4, !dbg !2679
  %add85 = add nsw i32 %91, %92, !dbg !2680
  %93 = load i32, i32* %size_arg.addr, align 4, !dbg !2681
  %mul86 = mul nsw i32 %add85, %93, !dbg !2682
  %94 = load i32, i32* %index_arg.addr, align 4, !dbg !2683
  %add87 = add nsw i32 %mul86, %94, !dbg !2684
  %idxprom88 = sext i32 %add87 to i64, !dbg !2677
  %arrayidx89 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %90, i64 %idxprom88, !dbg !2677
  %imag90 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx89, i32 0, i32 1, !dbg !2685
  store double %add84, double* %imag90, align 8, !dbg !2686
  br label %for.inc, !dbg !2687

for.inc:                                          ; preds = %for.body26
  %95 = load i32, i32* %k, align 4, !dbg !2688
  %inc = add nsw i32 %95, 1, !dbg !2688
  store i32 %inc, i32* %k, align 4, !dbg !2688
  br label %for.cond24, !dbg !2689, !llvm.loop !2690

for.end:                                          ; preds = %for.cond24
  br label %for.inc91, !dbg !2692

for.inc91:                                        ; preds = %for.end
  %96 = load i32, i32* %i, align 4, !dbg !2693
  %inc92 = add nsw i32 %96, 1, !dbg !2693
  store i32 %inc92, i32* %i, align 4, !dbg !2693
  br label %for.cond, !dbg !2694, !llvm.loop !2695

for.end93:                                        ; preds = %for.cond
  ret void, !dbg !2697
}

; Function Attrs: convergent noinline nounwind
define dso_local void @_Z19cffts3_gpu_kernel_1P8dcomplexS0_(%struct.dcomplex* %x_in, %struct.dcomplex* %y0) #0 !dbg !2698 {
entry:
  %x_in.addr = alloca %struct.dcomplex*, align 8
  %y0.addr = alloca %struct.dcomplex*, align 8
  %x_y_z = alloca i32, align 4
  store %struct.dcomplex* %x_in, %struct.dcomplex** %x_in.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %x_in.addr, metadata !2699, metadata !DIExpression()), !dbg !2700
  store %struct.dcomplex* %y0, %struct.dcomplex** %y0.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %y0.addr, metadata !2701, metadata !DIExpression()), !dbg !2702
  call void @llvm.dbg.declare(metadata i32* %x_y_z, metadata !2703, metadata !DIExpression()), !dbg !2704
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #8, !dbg !2705, !range !848
  %1 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #8, !dbg !2707, !range !893
  %mul = mul i32 %0, %1, !dbg !2709
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #8, !dbg !2710, !range !923
  %add = add i32 %mul, %2, !dbg !2712
  store i32 %add, i32* %x_y_z, align 4, !dbg !2704
  %3 = load i32, i32* %x_y_z, align 4, !dbg !2713
  %cmp = icmp sge i32 %3, 8388608, !dbg !2715
  br i1 %cmp, label %if.then, label %if.end, !dbg !2716

if.then:                                          ; preds = %entry
  br label %return, !dbg !2717

if.end:                                           ; preds = %entry
  %4 = load %struct.dcomplex*, %struct.dcomplex** %x_in.addr, align 8, !dbg !2719
  %5 = load i32, i32* %x_y_z, align 4, !dbg !2720
  %idxprom = sext i32 %5 to i64, !dbg !2719
  %arrayidx = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %4, i64 %idxprom, !dbg !2719
  %real = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx, i32 0, i32 0, !dbg !2721
  %6 = load double, double* %real, align 8, !dbg !2721
  %7 = load %struct.dcomplex*, %struct.dcomplex** %y0.addr, align 8, !dbg !2722
  %8 = load i32, i32* %x_y_z, align 4, !dbg !2723
  %idxprom3 = sext i32 %8 to i64, !dbg !2722
  %arrayidx4 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %7, i64 %idxprom3, !dbg !2722
  %real5 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx4, i32 0, i32 0, !dbg !2724
  store double %6, double* %real5, align 8, !dbg !2725
  %9 = load %struct.dcomplex*, %struct.dcomplex** %x_in.addr, align 8, !dbg !2726
  %10 = load i32, i32* %x_y_z, align 4, !dbg !2727
  %idxprom6 = sext i32 %10 to i64, !dbg !2726
  %arrayidx7 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %9, i64 %idxprom6, !dbg !2726
  %imag = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx7, i32 0, i32 1, !dbg !2728
  %11 = load double, double* %imag, align 8, !dbg !2728
  %12 = load %struct.dcomplex*, %struct.dcomplex** %y0.addr, align 8, !dbg !2729
  %13 = load i32, i32* %x_y_z, align 4, !dbg !2730
  %idxprom8 = sext i32 %13 to i64, !dbg !2729
  %arrayidx9 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %12, i64 %idxprom8, !dbg !2729
  %imag10 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx9, i32 0, i32 1, !dbg !2731
  store double %11, double* %imag10, align 8, !dbg !2732
  br label %return, !dbg !2733

return:                                           ; preds = %if.end, %if.then
  ret void, !dbg !2733
}

; Function Attrs: convergent noinline nounwind
define dso_local void @_Z19cffts3_gpu_kernel_2iP8dcomplexS0_S0_(i32 %is, %struct.dcomplex* %gty1, %struct.dcomplex* %gty2, %struct.dcomplex* %u_device) #0 !dbg !2734 {
entry:
  %is.addr = alloca i32, align 4
  %gty1.addr = alloca %struct.dcomplex*, align 8
  %gty2.addr = alloca %struct.dcomplex*, align 8
  %u_device.addr = alloca %struct.dcomplex*, align 8
  %x_y = alloca i32, align 4
  store i32 %is, i32* %is.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %is.addr, metadata !2735, metadata !DIExpression()), !dbg !2736
  store %struct.dcomplex* %gty1, %struct.dcomplex** %gty1.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %gty1.addr, metadata !2737, metadata !DIExpression()), !dbg !2738
  store %struct.dcomplex* %gty2, %struct.dcomplex** %gty2.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %gty2.addr, metadata !2739, metadata !DIExpression()), !dbg !2740
  store %struct.dcomplex* %u_device, %struct.dcomplex** %u_device.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %u_device.addr, metadata !2741, metadata !DIExpression()), !dbg !2742
  call void @llvm.dbg.declare(metadata i32* %x_y, metadata !2743, metadata !DIExpression()), !dbg !2744
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #8, !dbg !2745, !range !848
  %1 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #8, !dbg !2747, !range !893
  %mul = mul i32 %0, %1, !dbg !2749
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #8, !dbg !2750, !range !923
  %add = add i32 %mul, %2, !dbg !2752
  store i32 %add, i32* %x_y, align 4, !dbg !2744
  %3 = load i32, i32* %x_y, align 4, !dbg !2753
  %cmp = icmp sge i32 %3, 65536, !dbg !2755
  br i1 %cmp, label %if.then, label %if.end, !dbg !2756

if.then:                                          ; preds = %entry
  br label %return, !dbg !2757

if.end:                                           ; preds = %entry
  %4 = load i32, i32* %is.addr, align 4, !dbg !2759
  %call3 = call i32 @_Z12ilog2_devicei(i32 128) #3, !dbg !2760
  %5 = load %struct.dcomplex*, %struct.dcomplex** %gty1.addr, align 8, !dbg !2761
  %6 = load %struct.dcomplex*, %struct.dcomplex** %gty2.addr, align 8, !dbg !2762
  %7 = load %struct.dcomplex*, %struct.dcomplex** %u_device.addr, align 8, !dbg !2763
  %8 = load i32, i32* %x_y, align 4, !dbg !2764
  call void @_Z23cffts3_gpu_cfftz_deviceiiiP8dcomplexS0_S0_ii(i32 %4, i32 %call3, i32 128, %struct.dcomplex* %5, %struct.dcomplex* %6, %struct.dcomplex* %7, i32 %8, i32 65536) #3, !dbg !2765
  br label %return, !dbg !2766

return:                                           ; preds = %if.end, %if.then
  ret void, !dbg !2766
}

; Function Attrs: convergent noinline nounwind
define dso_local void @_Z19cffts3_gpu_kernel_3P8dcomplexS0_(%struct.dcomplex* %x_out, %struct.dcomplex* %y0) #0 !dbg !2767 {
entry:
  %x_out.addr = alloca %struct.dcomplex*, align 8
  %y0.addr = alloca %struct.dcomplex*, align 8
  %x_y_z = alloca i32, align 4
  store %struct.dcomplex* %x_out, %struct.dcomplex** %x_out.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %x_out.addr, metadata !2768, metadata !DIExpression()), !dbg !2769
  store %struct.dcomplex* %y0, %struct.dcomplex** %y0.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %y0.addr, metadata !2770, metadata !DIExpression()), !dbg !2771
  call void @llvm.dbg.declare(metadata i32* %x_y_z, metadata !2772, metadata !DIExpression()), !dbg !2773
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #8, !dbg !2774, !range !848
  %1 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #8, !dbg !2776, !range !893
  %mul = mul i32 %0, %1, !dbg !2778
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #8, !dbg !2779, !range !923
  %add = add i32 %mul, %2, !dbg !2781
  store i32 %add, i32* %x_y_z, align 4, !dbg !2773
  %3 = load i32, i32* %x_y_z, align 4, !dbg !2782
  %cmp = icmp sge i32 %3, 8388608, !dbg !2784
  br i1 %cmp, label %if.then, label %if.end, !dbg !2785

if.then:                                          ; preds = %entry
  br label %return, !dbg !2786

if.end:                                           ; preds = %entry
  %4 = load %struct.dcomplex*, %struct.dcomplex** %y0.addr, align 8, !dbg !2788
  %5 = load i32, i32* %x_y_z, align 4, !dbg !2789
  %idxprom = sext i32 %5 to i64, !dbg !2788
  %arrayidx = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %4, i64 %idxprom, !dbg !2788
  %real = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx, i32 0, i32 0, !dbg !2790
  %6 = load double, double* %real, align 8, !dbg !2790
  %7 = load %struct.dcomplex*, %struct.dcomplex** %x_out.addr, align 8, !dbg !2791
  %8 = load i32, i32* %x_y_z, align 4, !dbg !2792
  %idxprom3 = sext i32 %8 to i64, !dbg !2791
  %arrayidx4 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %7, i64 %idxprom3, !dbg !2791
  %real5 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx4, i32 0, i32 0, !dbg !2793
  store double %6, double* %real5, align 8, !dbg !2794
  %9 = load %struct.dcomplex*, %struct.dcomplex** %y0.addr, align 8, !dbg !2795
  %10 = load i32, i32* %x_y_z, align 4, !dbg !2796
  %idxprom6 = sext i32 %10 to i64, !dbg !2795
  %arrayidx7 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %9, i64 %idxprom6, !dbg !2795
  %imag = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx7, i32 0, i32 1, !dbg !2797
  %11 = load double, double* %imag, align 8, !dbg !2797
  %12 = load %struct.dcomplex*, %struct.dcomplex** %x_out.addr, align 8, !dbg !2798
  %13 = load i32, i32* %x_y_z, align 4, !dbg !2799
  %idxprom8 = sext i32 %13 to i64, !dbg !2798
  %arrayidx9 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %12, i64 %idxprom8, !dbg !2798
  %imag10 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx9, i32 0, i32 1, !dbg !2800
  store double %11, double* %imag10, align 8, !dbg !2801
  br label %return, !dbg !2802

return:                                           ; preds = %if.end, %if.then
  ret void, !dbg !2802
}

; Function Attrs: convergent noinline nounwind
define dso_local void @_Z19checksum_gpu_kerneliP8dcomplexS0_(i32 %iteration, %struct.dcomplex* %u1, %struct.dcomplex* %sums) #0 !dbg !2803 {
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
  call void @llvm.dbg.declare(metadata i32* %iteration.addr, metadata !2806, metadata !DIExpression()), !dbg !2807
  store %struct.dcomplex* %u1, %struct.dcomplex** %u1.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %u1.addr, metadata !2808, metadata !DIExpression()), !dbg !2809
  store %struct.dcomplex* %sums, %struct.dcomplex** %sums.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %sums.addr, metadata !2810, metadata !DIExpression()), !dbg !2811
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %share_sums, metadata !2812, metadata !DIExpression()), !dbg !2813
  store %struct.dcomplex* addrspacecast (%struct.dcomplex addrspace(3)* bitcast ([0 x double] addrspace(3)* @extern_share_data to %struct.dcomplex addrspace(3)*) to %struct.dcomplex*), %struct.dcomplex** %share_sums, align 8, !dbg !2813
  call void @llvm.dbg.declare(metadata i32* %j, metadata !2814, metadata !DIExpression()), !dbg !2815
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #8, !dbg !2816, !range !848
  %1 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #8, !dbg !2818, !range !893
  %mul = mul i32 %0, %1, !dbg !2820
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #8, !dbg !2821, !range !923
  %add = add i32 %mul, %2, !dbg !2823
  %add3 = add i32 %add, 1, !dbg !2824
  store i32 %add3, i32* %j, align 4, !dbg !2815
  call void @llvm.dbg.declare(metadata i32* %q, metadata !2825, metadata !DIExpression()), !dbg !2826
  call void @llvm.dbg.declare(metadata i32* %r, metadata !2827, metadata !DIExpression()), !dbg !2828
  call void @llvm.dbg.declare(metadata i32* %s, metadata !2829, metadata !DIExpression()), !dbg !2830
  %3 = load i32, i32* %j, align 4, !dbg !2831
  %cmp = icmp sle i32 %3, 1024, !dbg !2833
  br i1 %cmp, label %if.then, label %if.else, !dbg !2834

if.then:                                          ; preds = %entry
  %4 = load i32, i32* %j, align 4, !dbg !2835
  %rem = srem i32 %4, 256, !dbg !2837
  store i32 %rem, i32* %q, align 4, !dbg !2838
  %5 = load i32, i32* %j, align 4, !dbg !2839
  %mul4 = mul nsw i32 3, %5, !dbg !2840
  %rem5 = srem i32 %mul4, 256, !dbg !2841
  store i32 %rem5, i32* %r, align 4, !dbg !2842
  %6 = load i32, i32* %j, align 4, !dbg !2843
  %mul6 = mul nsw i32 5, %6, !dbg !2844
  %rem7 = srem i32 %mul6, 128, !dbg !2845
  store i32 %rem7, i32* %s, align 4, !dbg !2846
  %7 = load %struct.dcomplex*, %struct.dcomplex** %u1.addr, align 8, !dbg !2847
  %8 = load i32, i32* %q, align 4, !dbg !2848
  %9 = load i32, i32* %r, align 4, !dbg !2849
  %mul8 = mul nsw i32 %9, 256, !dbg !2850
  %add9 = add nsw i32 %8, %mul8, !dbg !2851
  %10 = load i32, i32* %s, align 4, !dbg !2852
  %mul10 = mul nsw i32 %10, 256, !dbg !2853
  %mul11 = mul nsw i32 %mul10, 256, !dbg !2854
  %add12 = add nsw i32 %add9, %mul11, !dbg !2855
  %idxprom = sext i32 %add12 to i64, !dbg !2847
  %arrayidx = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %7, i64 %idxprom, !dbg !2847
  %11 = load %struct.dcomplex*, %struct.dcomplex** %share_sums, align 8, !dbg !2856
  %12 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #8, !dbg !2857, !range !923
  %idxprom14 = zext i32 %12 to i64, !dbg !2856
  %arrayidx15 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %11, i64 %idxprom14, !dbg !2856
  %13 = bitcast %struct.dcomplex* %arrayidx15 to i8*, !dbg !2859
  %14 = bitcast %struct.dcomplex* %arrayidx to i8*, !dbg !2859
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %13, i8* align 8 %14, i64 16, i1 false), !dbg !2859
  br label %if.end, !dbg !2860

if.else:                                          ; preds = %entry
  %real = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp, i32 0, i32 0, !dbg !2861
  store double 0.000000e+00, double* %real, align 8, !dbg !2861
  %imag = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp, i32 0, i32 1, !dbg !2861
  store double 0.000000e+00, double* %imag, align 8, !dbg !2861
  %15 = load %struct.dcomplex*, %struct.dcomplex** %share_sums, align 8, !dbg !2863
  %16 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #8, !dbg !2864, !range !923
  %idxprom17 = zext i32 %16 to i64, !dbg !2863
  %arrayidx18 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %15, i64 %idxprom17, !dbg !2863
  %17 = bitcast %struct.dcomplex* %arrayidx18 to i8*, !dbg !2866
  %18 = bitcast %struct.dcomplex* %ref.tmp to i8*, !dbg !2866
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %17, i8* align 8 %18, i64 16, i1 false), !dbg !2866
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  call void @llvm.nvvm.barrier0(), !dbg !2867
  %19 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #8, !dbg !2868, !range !923
  %cmp20 = icmp eq i32 %19, 0, !dbg !2871
  br i1 %cmp20, label %if.then21, label %if.end40, !dbg !2872

if.then21:                                        ; preds = %if.end
  call void @llvm.dbg.declare(metadata i32* %i, metadata !2873, metadata !DIExpression()), !dbg !2876
  store i32 1, i32* %i, align 4, !dbg !2876
  br label %for.cond, !dbg !2877

for.cond:                                         ; preds = %for.inc, %if.then21
  %20 = load i32, i32* %i, align 4, !dbg !2878
  %21 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #8, !dbg !2880, !range !893
  %cmp23 = icmp ult i32 %20, %21, !dbg !2882
  br i1 %cmp23, label %for.body, label %for.end, !dbg !2883

for.body:                                         ; preds = %for.cond
  %real25 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp24, i32 0, i32 0, !dbg !2884
  %22 = load %struct.dcomplex*, %struct.dcomplex** %share_sums, align 8, !dbg !2884
  %arrayidx26 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %22, i64 0, !dbg !2884
  %real27 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx26, i32 0, i32 0, !dbg !2884
  %23 = load double, double* %real27, align 8, !dbg !2884
  %24 = load %struct.dcomplex*, %struct.dcomplex** %share_sums, align 8, !dbg !2884
  %25 = load i32, i32* %i, align 4, !dbg !2884
  %idxprom28 = sext i32 %25 to i64, !dbg !2884
  %arrayidx29 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %24, i64 %idxprom28, !dbg !2884
  %real30 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx29, i32 0, i32 0, !dbg !2884
  %26 = load double, double* %real30, align 8, !dbg !2884
  %add31 = fadd contract double %23, %26, !dbg !2884
  store double %add31, double* %real25, align 8, !dbg !2884
  %imag32 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp24, i32 0, i32 1, !dbg !2884
  %27 = load %struct.dcomplex*, %struct.dcomplex** %share_sums, align 8, !dbg !2884
  %arrayidx33 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %27, i64 0, !dbg !2884
  %imag34 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx33, i32 0, i32 1, !dbg !2884
  %28 = load double, double* %imag34, align 8, !dbg !2884
  %29 = load %struct.dcomplex*, %struct.dcomplex** %share_sums, align 8, !dbg !2884
  %30 = load i32, i32* %i, align 4, !dbg !2884
  %idxprom35 = sext i32 %30 to i64, !dbg !2884
  %arrayidx36 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %29, i64 %idxprom35, !dbg !2884
  %imag37 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx36, i32 0, i32 1, !dbg !2884
  %31 = load double, double* %imag37, align 8, !dbg !2884
  %add38 = fadd contract double %28, %31, !dbg !2884
  store double %add38, double* %imag32, align 8, !dbg !2884
  %32 = load %struct.dcomplex*, %struct.dcomplex** %share_sums, align 8, !dbg !2886
  %arrayidx39 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %32, i64 0, !dbg !2886
  %33 = bitcast %struct.dcomplex* %arrayidx39 to i8*, !dbg !2887
  %34 = bitcast %struct.dcomplex* %ref.tmp24 to i8*, !dbg !2887
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %33, i8* align 8 %34, i64 16, i1 false), !dbg !2887
  br label %for.inc, !dbg !2888

for.inc:                                          ; preds = %for.body
  %35 = load i32, i32* %i, align 4, !dbg !2889
  %inc = add nsw i32 %35, 1, !dbg !2889
  store i32 %inc, i32* %i, align 4, !dbg !2889
  br label %for.cond, !dbg !2890, !llvm.loop !2891

for.end:                                          ; preds = %for.cond
  br label %if.end40, !dbg !2893

if.end40:                                         ; preds = %for.end, %if.end
  %36 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #8, !dbg !2894, !range !923
  %cmp42 = icmp eq i32 %36, 0, !dbg !2897
  br i1 %cmp42, label %if.then43, label %if.end65, !dbg !2898

if.then43:                                        ; preds = %if.end40
  %37 = load %struct.dcomplex*, %struct.dcomplex** %share_sums, align 8, !dbg !2899
  %arrayidx44 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %37, i64 0, !dbg !2899
  %real45 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx44, i32 0, i32 0, !dbg !2901
  %38 = load double, double* %real45, align 8, !dbg !2901
  %div = fdiv double %38, 0x4160000000000000, !dbg !2902
  %39 = load %struct.dcomplex*, %struct.dcomplex** %share_sums, align 8, !dbg !2903
  %arrayidx46 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %39, i64 0, !dbg !2903
  %real47 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx46, i32 0, i32 0, !dbg !2904
  store double %div, double* %real47, align 8, !dbg !2905
  %40 = load %struct.dcomplex*, %struct.dcomplex** %sums.addr, align 8, !dbg !2906
  %41 = load i32, i32* %iteration.addr, align 4, !dbg !2907
  %idxprom48 = sext i32 %41 to i64, !dbg !2906
  %arrayidx49 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %40, i64 %idxprom48, !dbg !2906
  %real50 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx49, i32 0, i32 0, !dbg !2908
  %42 = load %struct.dcomplex*, %struct.dcomplex** %share_sums, align 8, !dbg !2909
  %arrayidx51 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %42, i64 0, !dbg !2909
  %real52 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx51, i32 0, i32 0, !dbg !2910
  %43 = load double, double* %real52, align 8, !dbg !2910
  %call53 = call double @_ZL9atomicAddPdd(double* %real50, double %43) #3, !dbg !2911
  %44 = load %struct.dcomplex*, %struct.dcomplex** %share_sums, align 8, !dbg !2912
  %arrayidx54 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %44, i64 0, !dbg !2912
  %imag55 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx54, i32 0, i32 1, !dbg !2913
  %45 = load double, double* %imag55, align 8, !dbg !2913
  %div56 = fdiv double %45, 0x4160000000000000, !dbg !2914
  %46 = load %struct.dcomplex*, %struct.dcomplex** %share_sums, align 8, !dbg !2915
  %arrayidx57 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %46, i64 0, !dbg !2915
  %imag58 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx57, i32 0, i32 1, !dbg !2916
  store double %div56, double* %imag58, align 8, !dbg !2917
  %47 = load %struct.dcomplex*, %struct.dcomplex** %sums.addr, align 8, !dbg !2918
  %48 = load i32, i32* %iteration.addr, align 4, !dbg !2919
  %idxprom59 = sext i32 %48 to i64, !dbg !2918
  %arrayidx60 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %47, i64 %idxprom59, !dbg !2918
  %imag61 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx60, i32 0, i32 1, !dbg !2920
  %49 = load %struct.dcomplex*, %struct.dcomplex** %share_sums, align 8, !dbg !2921
  %arrayidx62 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %49, i64 0, !dbg !2921
  %imag63 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx62, i32 0, i32 1, !dbg !2922
  %50 = load double, double* %imag63, align 8, !dbg !2922
  %call64 = call double @_ZL9atomicAddPdd(double* %imag61, double %50) #3, !dbg !2923
  br label %if.end65, !dbg !2924

if.end65:                                         ; preds = %if.then43, %if.end40
  ret void, !dbg !2925
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1 immarg) #2

; Function Attrs: convergent nounwind
declare void @llvm.nvvm.barrier0() #3

; Function Attrs: convergent noinline nounwind
define internal double @_ZL9atomicAddPdd(double* %address, double %val) #4 !dbg !2926 {
entry:
  %x.addr.i13 = alloca double, align 8
  call void @llvm.dbg.declare(metadata double* %x.addr.i13, metadata !2929, metadata !DIExpression()), !dbg !2933
  %x.addr.i12 = alloca i64, align 8
  call void @llvm.dbg.declare(metadata i64* %x.addr.i12, metadata !2938, metadata !DIExpression()), !dbg !2942
  %x.addr.i11 = alloca i64, align 8
  call void @llvm.dbg.declare(metadata i64* %x.addr.i11, metadata !2938, metadata !DIExpression()), !dbg !2944
  %x.addr.i10 = alloca i64, align 8
  call void @llvm.dbg.declare(metadata i64* %x.addr.i10, metadata !2938, metadata !DIExpression()), !dbg !2947
  %x.addr.i = alloca i64, align 8
  call void @llvm.dbg.declare(metadata i64* %x.addr.i, metadata !2938, metadata !DIExpression()), !dbg !2949
  %retval = alloca double, align 8
  %address.addr = alloca double*, align 8
  %val.addr = alloca double, align 8
  %address_as_ull = alloca i64*, align 8
  %old = alloca i64, align 8
  %assumed = alloca i64, align 8
  %i = alloca i32, align 4
  store double* %address, double** %address.addr, align 8
  call void @llvm.dbg.declare(metadata double** %address.addr, metadata !2952, metadata !DIExpression()), !dbg !2953
  store double %val, double* %val.addr, align 8
  call void @llvm.dbg.declare(metadata double* %val.addr, metadata !2954, metadata !DIExpression()), !dbg !2955
  call void @llvm.dbg.declare(metadata i64** %address_as_ull, metadata !2956, metadata !DIExpression()), !dbg !2957
  %0 = load double*, double** %address.addr, align 8, !dbg !2958
  %1 = bitcast double* %0 to i64*, !dbg !2959
  store i64* %1, i64** %address_as_ull, align 8, !dbg !2957
  call void @llvm.dbg.declare(metadata i64* %old, metadata !2960, metadata !DIExpression()), !dbg !2961
  %2 = load i64*, i64** %address_as_ull, align 8, !dbg !2962
  %3 = load i64, i64* %2, align 8, !dbg !2963
  store i64 %3, i64* %old, align 8, !dbg !2961
  call void @llvm.dbg.declare(metadata i64* %assumed, metadata !2964, metadata !DIExpression()), !dbg !2965
  %4 = load double, double* %val.addr, align 8, !dbg !2966
  %cmp = fcmp oeq double %4, 0.000000e+00, !dbg !2967
  br i1 %cmp, label %if.then, label %if.end, !dbg !2968

if.then:                                          ; preds = %entry
  %5 = load i64, i64* %old, align 8, !dbg !2969
  store i64 %5, i64* %x.addr.i, align 8
  %6 = load i64, i64* %x.addr.i, align 8, !dbg !2970
  %7 = bitcast i64 %6 to double, !dbg !2971
  store double %7, double* %retval, align 8, !dbg !2972
  br label %return, !dbg !2972

if.end:                                           ; preds = %entry
  call void @llvm.dbg.declare(metadata i32* %i, metadata !2973, metadata !DIExpression()), !dbg !2974
  store i32 0, i32* %i, align 4, !dbg !2974
  br label %for.cond, !dbg !2975

for.cond:                                         ; preds = %for.inc, %if.end
  %8 = load i32, i32* %i, align 4, !dbg !2976
  %cmp1 = icmp slt i32 %8, 100000, !dbg !2977
  br i1 %cmp1, label %for.body, label %for.end, !dbg !2978

for.body:                                         ; preds = %for.cond
  %9 = load i64, i64* %old, align 8, !dbg !2979
  store i64 %9, i64* %assumed, align 8, !dbg !2980
  %10 = load i64*, i64** %address_as_ull, align 8, !dbg !2981
  %11 = load i64, i64* %assumed, align 8, !dbg !2982
  %12 = load double, double* %val.addr, align 8, !dbg !2983
  %13 = load i64, i64* %assumed, align 8, !dbg !2984
  store i64 %13, i64* %x.addr.i12, align 8
  %14 = load i64, i64* %x.addr.i12, align 8, !dbg !2985
  %15 = bitcast i64 %14 to double, !dbg !2986
  %add = fadd contract double %12, %15, !dbg !2987
  store double %add, double* %x.addr.i13, align 8
  %16 = load double, double* %x.addr.i13, align 8, !dbg !2988
  %17 = bitcast double %16 to i64, !dbg !2989
  %call4 = call i64 @_ZL9atomicCASPyyy(i64* %10, i64 %11, i64 %17) #3, !dbg !2990
  store i64 %call4, i64* %old, align 8, !dbg !2991
  %18 = load i64, i64* %assumed, align 8, !dbg !2992
  %19 = load i64, i64* %old, align 8, !dbg !2993
  %cmp5 = icmp eq i64 %18, %19, !dbg !2994
  br i1 %cmp5, label %if.then6, label %if.end8, !dbg !2995

if.then6:                                         ; preds = %for.body
  %20 = load i64, i64* %old, align 8, !dbg !2996
  store i64 %20, i64* %x.addr.i11, align 8
  %21 = load i64, i64* %x.addr.i11, align 8, !dbg !2997
  %22 = bitcast i64 %21 to double, !dbg !2998
  store double %22, double* %retval, align 8, !dbg !2999
  br label %return, !dbg !2999

if.end8:                                          ; preds = %for.body
  br label %for.inc, !dbg !3000

for.inc:                                          ; preds = %if.end8
  %23 = load i32, i32* %i, align 4, !dbg !3001
  %inc = add nsw i32 %23, 1, !dbg !3001
  store i32 %inc, i32* %i, align 4, !dbg !3001
  br label %for.cond, !dbg !3002, !llvm.loop !3003

for.end:                                          ; preds = %for.cond
  %24 = load i64, i64* %old, align 8, !dbg !3005
  store i64 %24, i64* %x.addr.i10, align 8
  %25 = load i64, i64* %x.addr.i10, align 8, !dbg !3006
  %26 = bitcast i64 %25 to double, !dbg !3007
  store double %26, double* %retval, align 8, !dbg !3008
  br label %return, !dbg !3008

return:                                           ; preds = %for.end, %if.then6, %if.then
  %27 = load double, double* %retval, align 8, !dbg !3009
  ret double %27, !dbg !3009
}

; Function Attrs: convergent noinline nounwind
define dso_local void @_Z27compute_indexmap_gpu_kernelPd(double* %twiddle) #4 !dbg !3010 {
entry:
  %a.addr.i = alloca double, align 8
  call void @llvm.dbg.declare(metadata double* %a.addr.i, metadata !3013, metadata !DIExpression()), !dbg !3016
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
  call void @llvm.dbg.declare(metadata double** %twiddle.addr, metadata !3018, metadata !DIExpression()), !dbg !3019
  call void @llvm.dbg.declare(metadata i32* %thread_id, metadata !3020, metadata !DIExpression()), !dbg !3021
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #8, !dbg !3022, !range !848
  %1 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #8, !dbg !3024, !range !893
  %mul = mul i32 %0, %1, !dbg !3026
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #8, !dbg !3027, !range !923
  %add = add i32 %mul, %2, !dbg !3029
  store i32 %add, i32* %thread_id, align 4, !dbg !3021
  %3 = load i32, i32* %thread_id, align 4, !dbg !3030
  %cmp = icmp sge i32 %3, 8388608, !dbg !3032
  br i1 %cmp, label %if.then, label %if.end, !dbg !3033

if.then:                                          ; preds = %entry
  br label %return, !dbg !3034

if.end:                                           ; preds = %entry
  call void @llvm.dbg.declare(metadata i32* %i, metadata !3036, metadata !DIExpression()), !dbg !3037
  %4 = load i32, i32* %thread_id, align 4, !dbg !3038
  %rem = srem i32 %4, 256, !dbg !3039
  store i32 %rem, i32* %i, align 4, !dbg !3037
  call void @llvm.dbg.declare(metadata i32* %j, metadata !3040, metadata !DIExpression()), !dbg !3041
  %5 = load i32, i32* %thread_id, align 4, !dbg !3042
  %div = sdiv i32 %5, 256, !dbg !3043
  %rem3 = srem i32 %div, 256, !dbg !3044
  store i32 %rem3, i32* %j, align 4, !dbg !3041
  call void @llvm.dbg.declare(metadata i32* %k, metadata !3045, metadata !DIExpression()), !dbg !3046
  %6 = load i32, i32* %thread_id, align 4, !dbg !3047
  %div4 = sdiv i32 %6, 65536, !dbg !3048
  store i32 %div4, i32* %k, align 4, !dbg !3046
  call void @llvm.dbg.declare(metadata i32* %kk, metadata !3049, metadata !DIExpression()), !dbg !3050
  call void @llvm.dbg.declare(metadata i32* %kk2, metadata !3051, metadata !DIExpression()), !dbg !3052
  call void @llvm.dbg.declare(metadata i32* %jj, metadata !3053, metadata !DIExpression()), !dbg !3054
  call void @llvm.dbg.declare(metadata i32* %kj2, metadata !3055, metadata !DIExpression()), !dbg !3056
  call void @llvm.dbg.declare(metadata i32* %ii, metadata !3057, metadata !DIExpression()), !dbg !3058
  %7 = load i32, i32* %k, align 4, !dbg !3059
  %add5 = add nsw i32 %7, 64, !dbg !3060
  %rem6 = srem i32 %add5, 128, !dbg !3061
  %sub = sub nsw i32 %rem6, 64, !dbg !3062
  store i32 %sub, i32* %kk, align 4, !dbg !3063
  %8 = load i32, i32* %kk, align 4, !dbg !3064
  %9 = load i32, i32* %kk, align 4, !dbg !3065
  %mul7 = mul nsw i32 %8, %9, !dbg !3066
  store i32 %mul7, i32* %kk2, align 4, !dbg !3067
  %10 = load i32, i32* %j, align 4, !dbg !3068
  %add8 = add nsw i32 %10, 128, !dbg !3069
  %rem9 = srem i32 %add8, 256, !dbg !3070
  %sub10 = sub nsw i32 %rem9, 128, !dbg !3071
  store i32 %sub10, i32* %jj, align 4, !dbg !3072
  %11 = load i32, i32* %jj, align 4, !dbg !3073
  %12 = load i32, i32* %jj, align 4, !dbg !3074
  %mul11 = mul nsw i32 %11, %12, !dbg !3075
  %13 = load i32, i32* %kk2, align 4, !dbg !3076
  %add12 = add nsw i32 %mul11, %13, !dbg !3077
  store i32 %add12, i32* %kj2, align 4, !dbg !3078
  %14 = load i32, i32* %i, align 4, !dbg !3079
  %add13 = add nsw i32 %14, 128, !dbg !3080
  %rem14 = srem i32 %add13, 256, !dbg !3081
  %sub15 = sub nsw i32 %rem14, 128, !dbg !3082
  store i32 %sub15, i32* %ii, align 4, !dbg !3083
  %15 = load i32, i32* %ii, align 4, !dbg !3084
  %16 = load i32, i32* %ii, align 4, !dbg !3085
  %mul16 = mul nsw i32 %15, %16, !dbg !3086
  %17 = load i32, i32* %kj2, align 4, !dbg !3087
  %add17 = add nsw i32 %mul16, %17, !dbg !3088
  %conv = sitofp i32 %add17 to double, !dbg !3089
  %mul18 = fmul contract double 0xBF04B2B4199E149A, %conv, !dbg !3090
  store double %mul18, double* %a.addr.i, align 8
  %18 = load double, double* %a.addr.i, align 8, !dbg !3091
  %19 = call i32 @llvm.nvvm.d2i.hi(double %18) #8, !dbg !3092
  %20 = bitcast i32 %19 to float, !dbg !3092
  %21 = call float @llvm.nvvm.fabs.f(float %20) #8, !dbg !3092
  %22 = fcmp olt float %21, 0x4010E92220000000, !dbg !3092
  %23 = zext i1 %22 to i32, !dbg !3092
  br i1 %22, label %24, label %57, !dbg !3092

24:                                               ; preds = %if.end
  %25 = call double @llvm.nvvm.mul.rn.d(double %18, double 0x3FF71547652B82FE) #8, !dbg !3092
  %26 = call double @llvm.nvvm.add.rn.d(double %25, double 0x4338000000000000) #8, !dbg !3092
  %27 = call i32 @llvm.nvvm.d2i.lo(double %26) #8, !dbg !3092
  %28 = call double @llvm.nvvm.add.rn.d(double %25, double 0x4338000000000000) #8, !dbg !3092
  %29 = call double @llvm.nvvm.add.rn.d(double %28, double 0xC338000000000000) #8, !dbg !3092
  %30 = call double @llvm.nvvm.fma.rn.d(double %29, double 0xBFE62E42FEFA39EF, double %18) #8, !dbg !3092
  %31 = call double @llvm.nvvm.fma.rn.d(double %29, double 0xBC7ABC9E3B39803F, double %30) #8, !dbg !3092
  %32 = call double @llvm.nvvm.fma.rn.d(double 0x3E5ADE1569CE2BDF, double %31, double 0x3E928AF3FCA213EA) #8, !dbg !3092
  %33 = call double @llvm.nvvm.fma.rn.d(double %32, double %31, double 0x3EC71DEE62401315) #8, !dbg !3092
  %34 = call double @llvm.nvvm.fma.rn.d(double %33, double %31, double 0x3EFA01997C89EB71) #8, !dbg !3092
  %35 = call double @llvm.nvvm.fma.rn.d(double %34, double %31, double 0x3F2A01A014761F65) #8, !dbg !3092
  %36 = call double @llvm.nvvm.fma.rn.d(double %35, double %31, double 0x3F56C16C1852B7AF) #8, !dbg !3092
  %37 = call double @llvm.nvvm.fma.rn.d(double %36, double %31, double 0x3F81111111122322) #8, !dbg !3092
  %38 = call double @llvm.nvvm.fma.rn.d(double %37, double %31, double 0x3FA55555555502A1) #8, !dbg !3092
  %39 = call double @llvm.nvvm.fma.rn.d(double %38, double %31, double 0x3FC5555555555511) #8, !dbg !3092
  %40 = call double @llvm.nvvm.fma.rn.d(double %39, double %31, double 0x3FE000000000000B) #8, !dbg !3092
  %41 = call double @llvm.nvvm.fma.rn.d(double %40, double %31, double 1.000000e+00) #8, !dbg !3092
  %42 = call double @llvm.nvvm.fma.rn.d(double %41, double %31, double 1.000000e+00) #8, !dbg !3092
  %neg.i.i = sub i32 0, %27, !dbg !3092
  %abs.cond.i.i = icmp sge i32 %27, 0, !dbg !3092
  %abs.i.i = select i1 %abs.cond.i.i, i32 %27, i32 %neg.i.i, !dbg !3092
  %43 = icmp slt i32 %abs.i.i, 1023, !dbg !3092
  br i1 %43, label %44, label %47, !dbg !3092

44:                                               ; preds = %24
  %45 = shl i32 %27, 20, !dbg !3092
  %46 = add nsw i32 %45, 1072693248, !dbg !3092
  br label %__internal_exp_kernel.exit.i.i, !dbg !3092

47:                                               ; preds = %24
  %48 = add nsw i32 %27, 2046, !dbg !3092
  %49 = udiv i32 %48, 2, !dbg !3092
  %50 = shl i32 %49, 20, !dbg !3092
  %51 = shl i32 %48, 20, !dbg !3092
  %52 = sub i32 %51, %50, !dbg !3092
  %53 = call double @llvm.nvvm.lohi.i2d(i32 0, i32 %50) #8, !dbg !3092
  %54 = fmul double %42, %53, !dbg !3092
  br label %__internal_exp_kernel.exit.i.i, !dbg !3092

__internal_exp_kernel.exit.i.i:                   ; preds = %47, %44
  %a.addr.0.i.i.i.i = phi double [ %42, %44 ], [ %54, %47 ], !dbg !3092
  %k.0.i.i.i.i = phi i32 [ %46, %44 ], [ %52, %47 ], !dbg !3092
  %55 = call double @llvm.nvvm.lohi.i2d(i32 0, i32 %k.0.i.i.i.i) #8, !dbg !3092
  %56 = fmul double %a.addr.0.i.i.i.i, %55, !dbg !3092
  br label %_ZL3expd.exit, !dbg !3092

57:                                               ; preds = %if.end
  %58 = icmp slt i32 %19, 0, !dbg !3092
  br i1 %58, label %59, label %60, !dbg !3092

59:                                               ; preds = %57
  br label %61, !dbg !3092

60:                                               ; preds = %57
  br label %61, !dbg !3092

61:                                               ; preds = %60, %59
  %62 = phi double [ 0.000000e+00, %59 ], [ 0x7FF0000000000000, %60 ], !dbg !3092
  %63 = call double @llvm.nvvm.fabs.d(double %18) #8, !dbg !3092
  %64 = fcmp ole double %63, 0x7FF0000000000000, !dbg !3092
  %65 = xor i1 %64, true, !dbg !3092
  %66 = zext i1 %65 to i32, !dbg !3092
  br i1 %65, label %67, label %69, !dbg !3092

67:                                               ; preds = %61
  %68 = fadd double %18, %18, !dbg !3092
  br label %69, !dbg !3092

69:                                               ; preds = %67, %61
  %t.0.i.i = phi double [ %68, %67 ], [ %62, %61 ], !dbg !3092
  br label %_ZL3expd.exit, !dbg !3092

_ZL3expd.exit:                                    ; preds = %__internal_exp_kernel.exit.i.i, %69
  %t.1.i.i = phi double [ %56, %__internal_exp_kernel.exit.i.i ], [ %t.0.i.i, %69 ], !dbg !3092
  %70 = load double*, double** %twiddle.addr, align 8, !dbg !3093
  %71 = load i32, i32* %thread_id, align 4, !dbg !3094
  %idxprom = sext i32 %71 to i64, !dbg !3093
  %arrayidx = getelementptr inbounds double, double* %70, i64 %idxprom, !dbg !3093
  store double %t.1.i.i, double* %arrayidx, align 8, !dbg !3095
  br label %return, !dbg !3096

return:                                           ; preds = %_ZL3expd.exit, %if.then
  ret void, !dbg !3096
}

; Function Attrs: convergent noinline nounwind
define dso_local void @_Z37compute_initial_conditions_gpu_kernelP8dcomplexPd(%struct.dcomplex* %u0, double* %starts) #0 !dbg !3097 {
entry:
  %u0.addr = alloca %struct.dcomplex*, align 8
  %starts.addr = alloca double*, align 8
  %z = alloca i32, align 4
  %x0 = alloca double, align 8
  %y = alloca i32, align 4
  store %struct.dcomplex* %u0, %struct.dcomplex** %u0.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %u0.addr, metadata !3100, metadata !DIExpression()), !dbg !3101
  store double* %starts, double** %starts.addr, align 8
  call void @llvm.dbg.declare(metadata double** %starts.addr, metadata !3102, metadata !DIExpression()), !dbg !3103
  call void @llvm.dbg.declare(metadata i32* %z, metadata !3104, metadata !DIExpression()), !dbg !3105
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #8, !dbg !3106, !range !848
  %1 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #8, !dbg !3108, !range !893
  %mul = mul i32 %0, %1, !dbg !3110
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #8, !dbg !3111, !range !923
  %add = add i32 %mul, %2, !dbg !3113
  store i32 %add, i32* %z, align 4, !dbg !3105
  %3 = load i32, i32* %z, align 4, !dbg !3114
  %cmp = icmp sge i32 %3, 128, !dbg !3116
  br i1 %cmp, label %if.then, label %if.end, !dbg !3117

if.then:                                          ; preds = %entry
  br label %for.end, !dbg !3118

if.end:                                           ; preds = %entry
  call void @llvm.dbg.declare(metadata double* %x0, metadata !3120, metadata !DIExpression()), !dbg !3121
  %4 = load double*, double** %starts.addr, align 8, !dbg !3122
  %5 = load i32, i32* %z, align 4, !dbg !3123
  %idxprom = sext i32 %5 to i64, !dbg !3122
  %arrayidx = getelementptr inbounds double, double* %4, i64 %idxprom, !dbg !3122
  %6 = load double, double* %arrayidx, align 8, !dbg !3122
  store double %6, double* %x0, align 8, !dbg !3121
  call void @llvm.dbg.declare(metadata i32* %y, metadata !3124, metadata !DIExpression()), !dbg !3126
  store i32 0, i32* %y, align 4, !dbg !3126
  br label %for.cond, !dbg !3127

for.cond:                                         ; preds = %for.inc, %if.end
  %7 = load i32, i32* %y, align 4, !dbg !3128
  %cmp3 = icmp slt i32 %7, 256, !dbg !3130
  br i1 %cmp3, label %for.body, label %for.end, !dbg !3131

for.body:                                         ; preds = %for.cond
  %8 = load %struct.dcomplex*, %struct.dcomplex** %u0.addr, align 8, !dbg !3132
  %9 = load i32, i32* %y, align 4, !dbg !3134
  %mul4 = mul nsw i32 %9, 256, !dbg !3135
  %add5 = add nsw i32 0, %mul4, !dbg !3136
  %10 = load i32, i32* %z, align 4, !dbg !3137
  %mul6 = mul nsw i32 %10, 256, !dbg !3138
  %mul7 = mul nsw i32 %mul6, 256, !dbg !3139
  %add8 = add nsw i32 %add5, %mul7, !dbg !3140
  %idxprom9 = sext i32 %add8 to i64, !dbg !3132
  %arrayidx10 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %8, i64 %idxprom9, !dbg !3132
  %11 = bitcast %struct.dcomplex* %arrayidx10 to double*, !dbg !3141
  call void @_Z13vranlc_deviceiPddS_(i32 512, double* %x0, double 0x41D2309CE5400000, double* %11) #3, !dbg !3142
  br label %for.inc, !dbg !3143

for.inc:                                          ; preds = %for.body
  %12 = load i32, i32* %y, align 4, !dbg !3144
  %inc = add nsw i32 %12, 1, !dbg !3144
  store i32 %inc, i32* %y, align 4, !dbg !3144
  br label %for.cond, !dbg !3145, !llvm.loop !3146

for.end:                                          ; preds = %if.then, %for.cond
  ret void, !dbg !3148
}

; Function Attrs: convergent noinline nounwind
define dso_local void @_Z13vranlc_deviceiPddS_(i32 %n, double* %x_seed, double %a, double* %y) #0 !dbg !3149 {
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
  call void @llvm.dbg.declare(metadata i32* %n.addr, metadata !3152, metadata !DIExpression()), !dbg !3153
  store double* %x_seed, double** %x_seed.addr, align 8
  call void @llvm.dbg.declare(metadata double** %x_seed.addr, metadata !3154, metadata !DIExpression()), !dbg !3155
  store double %a, double* %a.addr, align 8
  call void @llvm.dbg.declare(metadata double* %a.addr, metadata !3156, metadata !DIExpression()), !dbg !3157
  store double* %y, double** %y.addr, align 8
  call void @llvm.dbg.declare(metadata double** %y.addr, metadata !3158, metadata !DIExpression()), !dbg !3159
  call void @llvm.dbg.declare(metadata i32* %i, metadata !3160, metadata !DIExpression()), !dbg !3161
  call void @llvm.dbg.declare(metadata double* %x, metadata !3162, metadata !DIExpression()), !dbg !3163
  call void @llvm.dbg.declare(metadata double* %t1, metadata !3164, metadata !DIExpression()), !dbg !3165
  call void @llvm.dbg.declare(metadata double* %t2, metadata !3166, metadata !DIExpression()), !dbg !3167
  call void @llvm.dbg.declare(metadata double* %t3, metadata !3168, metadata !DIExpression()), !dbg !3169
  call void @llvm.dbg.declare(metadata double* %t4, metadata !3170, metadata !DIExpression()), !dbg !3171
  call void @llvm.dbg.declare(metadata double* %a1, metadata !3172, metadata !DIExpression()), !dbg !3173
  call void @llvm.dbg.declare(metadata double* %a2, metadata !3174, metadata !DIExpression()), !dbg !3175
  call void @llvm.dbg.declare(metadata double* %x1, metadata !3176, metadata !DIExpression()), !dbg !3177
  call void @llvm.dbg.declare(metadata double* %x2, metadata !3178, metadata !DIExpression()), !dbg !3179
  call void @llvm.dbg.declare(metadata double* %z, metadata !3180, metadata !DIExpression()), !dbg !3181
  %0 = load double, double* %a.addr, align 8, !dbg !3182
  %mul = fmul contract double 0x3E80000000000000, %0, !dbg !3183
  store double %mul, double* %t1, align 8, !dbg !3184
  %1 = load double, double* %t1, align 8, !dbg !3185
  %conv = fptosi double %1 to i32, !dbg !3185
  %conv1 = sitofp i32 %conv to double, !dbg !3186
  store double %conv1, double* %a1, align 8, !dbg !3187
  %2 = load double, double* %a.addr, align 8, !dbg !3188
  %3 = load double, double* %a1, align 8, !dbg !3189
  %mul2 = fmul contract double 0x4160000000000000, %3, !dbg !3190
  %sub = fsub contract double %2, %mul2, !dbg !3191
  store double %sub, double* %a2, align 8, !dbg !3192
  %4 = load double*, double** %x_seed.addr, align 8, !dbg !3193
  %5 = load double, double* %4, align 8, !dbg !3194
  store double %5, double* %x, align 8, !dbg !3195
  store i32 0, i32* %i, align 4, !dbg !3196
  br label %for.cond, !dbg !3198

for.cond:                                         ; preds = %for.inc, %entry
  %6 = load i32, i32* %i, align 4, !dbg !3199
  %7 = load i32, i32* %n.addr, align 4, !dbg !3201
  %cmp = icmp slt i32 %6, %7, !dbg !3202
  br i1 %cmp, label %for.body, label %for.end, !dbg !3203

for.body:                                         ; preds = %for.cond
  %8 = load double, double* %x, align 8, !dbg !3204
  %mul3 = fmul contract double 0x3E80000000000000, %8, !dbg !3206
  store double %mul3, double* %t1, align 8, !dbg !3207
  %9 = load double, double* %t1, align 8, !dbg !3208
  %conv4 = fptosi double %9 to i32, !dbg !3208
  %conv5 = sitofp i32 %conv4 to double, !dbg !3209
  store double %conv5, double* %x1, align 8, !dbg !3210
  %10 = load double, double* %x, align 8, !dbg !3211
  %11 = load double, double* %x1, align 8, !dbg !3212
  %mul6 = fmul contract double 0x4160000000000000, %11, !dbg !3213
  %sub7 = fsub contract double %10, %mul6, !dbg !3214
  store double %sub7, double* %x2, align 8, !dbg !3215
  %12 = load double, double* %a1, align 8, !dbg !3216
  %13 = load double, double* %x2, align 8, !dbg !3217
  %mul8 = fmul contract double %12, %13, !dbg !3218
  %14 = load double, double* %a2, align 8, !dbg !3219
  %15 = load double, double* %x1, align 8, !dbg !3220
  %mul9 = fmul contract double %14, %15, !dbg !3221
  %add = fadd contract double %mul8, %mul9, !dbg !3222
  store double %add, double* %t1, align 8, !dbg !3223
  %16 = load double, double* %t1, align 8, !dbg !3224
  %mul10 = fmul contract double 0x3E80000000000000, %16, !dbg !3225
  %conv11 = fptosi double %mul10 to i32, !dbg !3226
  %conv12 = sitofp i32 %conv11 to double, !dbg !3227
  store double %conv12, double* %t2, align 8, !dbg !3228
  %17 = load double, double* %t1, align 8, !dbg !3229
  %18 = load double, double* %t2, align 8, !dbg !3230
  %mul13 = fmul contract double 0x4160000000000000, %18, !dbg !3231
  %sub14 = fsub contract double %17, %mul13, !dbg !3232
  store double %sub14, double* %z, align 8, !dbg !3233
  %19 = load double, double* %z, align 8, !dbg !3234
  %mul15 = fmul contract double 0x4160000000000000, %19, !dbg !3235
  %20 = load double, double* %a2, align 8, !dbg !3236
  %21 = load double, double* %x2, align 8, !dbg !3237
  %mul16 = fmul contract double %20, %21, !dbg !3238
  %add17 = fadd contract double %mul15, %mul16, !dbg !3239
  store double %add17, double* %t3, align 8, !dbg !3240
  %22 = load double, double* %t3, align 8, !dbg !3241
  %mul18 = fmul contract double 0x3D10000000000000, %22, !dbg !3242
  %conv19 = fptosi double %mul18 to i32, !dbg !3243
  %conv20 = sitofp i32 %conv19 to double, !dbg !3244
  store double %conv20, double* %t4, align 8, !dbg !3245
  %23 = load double, double* %t3, align 8, !dbg !3246
  %24 = load double, double* %t4, align 8, !dbg !3247
  %mul21 = fmul contract double 0x42D0000000000000, %24, !dbg !3248
  %sub22 = fsub contract double %23, %mul21, !dbg !3249
  store double %sub22, double* %x, align 8, !dbg !3250
  %25 = load double, double* %x, align 8, !dbg !3251
  %mul23 = fmul contract double 0x3D10000000000000, %25, !dbg !3252
  %26 = load double*, double** %y.addr, align 8, !dbg !3253
  %27 = load i32, i32* %i, align 4, !dbg !3254
  %idxprom = sext i32 %27 to i64, !dbg !3253
  %arrayidx = getelementptr inbounds double, double* %26, i64 %idxprom, !dbg !3253
  store double %mul23, double* %arrayidx, align 8, !dbg !3255
  br label %for.inc, !dbg !3256

for.inc:                                          ; preds = %for.body
  %28 = load i32, i32* %i, align 4, !dbg !3257
  %inc = add nsw i32 %28, 1, !dbg !3257
  store i32 %inc, i32* %i, align 4, !dbg !3257
  br label %for.cond, !dbg !3258, !llvm.loop !3259

for.end:                                          ; preds = %for.cond
  %29 = load double, double* %x, align 8, !dbg !3261
  %30 = load double*, double** %x_seed.addr, align 8, !dbg !3262
  store double %29, double* %30, align 8, !dbg !3263
  ret void, !dbg !3264
}

; Function Attrs: convergent noinline nounwind
define dso_local void @_Z17evolve_gpu_kernelP8dcomplexS0_Pd(%struct.dcomplex* %u0, %struct.dcomplex* %u1, double* %twiddle) #0 !dbg !3265 {
entry:
  %u0.addr = alloca %struct.dcomplex*, align 8
  %u1.addr = alloca %struct.dcomplex*, align 8
  %twiddle.addr = alloca double*, align 8
  %thread_id = alloca i32, align 4
  %ref.tmp = alloca %struct.dcomplex, align 8
  store %struct.dcomplex* %u0, %struct.dcomplex** %u0.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %u0.addr, metadata !3268, metadata !DIExpression()), !dbg !3269
  store %struct.dcomplex* %u1, %struct.dcomplex** %u1.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %u1.addr, metadata !3270, metadata !DIExpression()), !dbg !3271
  store double* %twiddle, double** %twiddle.addr, align 8
  call void @llvm.dbg.declare(metadata double** %twiddle.addr, metadata !3272, metadata !DIExpression()), !dbg !3273
  call void @llvm.dbg.declare(metadata i32* %thread_id, metadata !3274, metadata !DIExpression()), !dbg !3275
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #8, !dbg !3276, !range !848
  %1 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #8, !dbg !3278, !range !893
  %mul = mul i32 %0, %1, !dbg !3280
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #8, !dbg !3281, !range !923
  %add = add i32 %mul, %2, !dbg !3283
  store i32 %add, i32* %thread_id, align 4, !dbg !3275
  %3 = load i32, i32* %thread_id, align 4, !dbg !3284
  %cmp = icmp sge i32 %3, 8388608, !dbg !3286
  br i1 %cmp, label %if.then, label %if.end, !dbg !3287

if.then:                                          ; preds = %entry
  br label %return, !dbg !3288

if.end:                                           ; preds = %entry
  %real = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp, i32 0, i32 0, !dbg !3290
  %4 = load %struct.dcomplex*, %struct.dcomplex** %u0.addr, align 8, !dbg !3290
  %5 = load i32, i32* %thread_id, align 4, !dbg !3290
  %idxprom = sext i32 %5 to i64, !dbg !3290
  %arrayidx = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %4, i64 %idxprom, !dbg !3290
  %real3 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx, i32 0, i32 0, !dbg !3290
  %6 = load double, double* %real3, align 8, !dbg !3290
  %7 = load double*, double** %twiddle.addr, align 8, !dbg !3290
  %8 = load i32, i32* %thread_id, align 4, !dbg !3290
  %idxprom4 = sext i32 %8 to i64, !dbg !3290
  %arrayidx5 = getelementptr inbounds double, double* %7, i64 %idxprom4, !dbg !3290
  %9 = load double, double* %arrayidx5, align 8, !dbg !3290
  %mul6 = fmul contract double %6, %9, !dbg !3290
  store double %mul6, double* %real, align 8, !dbg !3290
  %imag = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp, i32 0, i32 1, !dbg !3290
  %10 = load %struct.dcomplex*, %struct.dcomplex** %u0.addr, align 8, !dbg !3290
  %11 = load i32, i32* %thread_id, align 4, !dbg !3290
  %idxprom7 = sext i32 %11 to i64, !dbg !3290
  %arrayidx8 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %10, i64 %idxprom7, !dbg !3290
  %imag9 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %arrayidx8, i32 0, i32 1, !dbg !3290
  %12 = load double, double* %imag9, align 8, !dbg !3290
  %13 = load double*, double** %twiddle.addr, align 8, !dbg !3290
  %14 = load i32, i32* %thread_id, align 4, !dbg !3290
  %idxprom10 = sext i32 %14 to i64, !dbg !3290
  %arrayidx11 = getelementptr inbounds double, double* %13, i64 %idxprom10, !dbg !3290
  %15 = load double, double* %arrayidx11, align 8, !dbg !3290
  %mul12 = fmul contract double %12, %15, !dbg !3290
  store double %mul12, double* %imag, align 8, !dbg !3290
  %16 = load %struct.dcomplex*, %struct.dcomplex** %u0.addr, align 8, !dbg !3291
  %17 = load i32, i32* %thread_id, align 4, !dbg !3292
  %idxprom13 = sext i32 %17 to i64, !dbg !3291
  %arrayidx14 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %16, i64 %idxprom13, !dbg !3291
  %18 = bitcast %struct.dcomplex* %arrayidx14 to i8*, !dbg !3293
  %19 = bitcast %struct.dcomplex* %ref.tmp to i8*, !dbg !3293
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %18, i8* align 8 %19, i64 16, i1 false), !dbg !3293
  %20 = load %struct.dcomplex*, %struct.dcomplex** %u0.addr, align 8, !dbg !3294
  %21 = load i32, i32* %thread_id, align 4, !dbg !3295
  %idxprom15 = sext i32 %21 to i64, !dbg !3294
  %arrayidx16 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %20, i64 %idxprom15, !dbg !3294
  %22 = load %struct.dcomplex*, %struct.dcomplex** %u1.addr, align 8, !dbg !3296
  %23 = load i32, i32* %thread_id, align 4, !dbg !3297
  %idxprom17 = sext i32 %23 to i64, !dbg !3296
  %arrayidx18 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %22, i64 %idxprom17, !dbg !3296
  %24 = bitcast %struct.dcomplex* %arrayidx18 to i8*, !dbg !3298
  %25 = bitcast %struct.dcomplex* %arrayidx16 to i8*, !dbg !3298
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %24, i8* align 8 %25, i64 16, i1 false), !dbg !3298
  br label %return, !dbg !3299

return:                                           ; preds = %if.end, %if.then
  ret void, !dbg !3299
}

; Function Attrs: convergent noinline nounwind
define dso_local void @_Z18init_ui_gpu_kernelP8dcomplexS0_Pd(%struct.dcomplex* %u0, %struct.dcomplex* %u1, double* %twiddle) #0 !dbg !3300 {
entry:
  %u0.addr = alloca %struct.dcomplex*, align 8
  %u1.addr = alloca %struct.dcomplex*, align 8
  %twiddle.addr = alloca double*, align 8
  %thread_id = alloca i32, align 4
  %ref.tmp = alloca %struct.dcomplex, align 8
  %ref.tmp3 = alloca %struct.dcomplex, align 8
  store %struct.dcomplex* %u0, %struct.dcomplex** %u0.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %u0.addr, metadata !3301, metadata !DIExpression()), !dbg !3302
  store %struct.dcomplex* %u1, %struct.dcomplex** %u1.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.dcomplex** %u1.addr, metadata !3303, metadata !DIExpression()), !dbg !3304
  store double* %twiddle, double** %twiddle.addr, align 8
  call void @llvm.dbg.declare(metadata double** %twiddle.addr, metadata !3305, metadata !DIExpression()), !dbg !3306
  call void @llvm.dbg.declare(metadata i32* %thread_id, metadata !3307, metadata !DIExpression()), !dbg !3308
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #8, !dbg !3309, !range !848
  %1 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #8, !dbg !3311, !range !893
  %mul = mul i32 %0, %1, !dbg !3313
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #8, !dbg !3314, !range !923
  %add = add i32 %mul, %2, !dbg !3316
  store i32 %add, i32* %thread_id, align 4, !dbg !3308
  %3 = load i32, i32* %thread_id, align 4, !dbg !3317
  %cmp = icmp sge i32 %3, 8388608, !dbg !3319
  br i1 %cmp, label %if.then, label %if.end, !dbg !3320

if.then:                                          ; preds = %entry
  br label %return, !dbg !3321

if.end:                                           ; preds = %entry
  %real = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp, i32 0, i32 0, !dbg !3323
  store double 0.000000e+00, double* %real, align 8, !dbg !3323
  %imag = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp, i32 0, i32 1, !dbg !3323
  store double 0.000000e+00, double* %imag, align 8, !dbg !3323
  %4 = load %struct.dcomplex*, %struct.dcomplex** %u0.addr, align 8, !dbg !3324
  %5 = load i32, i32* %thread_id, align 4, !dbg !3325
  %idxprom = sext i32 %5 to i64, !dbg !3324
  %arrayidx = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %4, i64 %idxprom, !dbg !3324
  %6 = bitcast %struct.dcomplex* %arrayidx to i8*, !dbg !3326
  %7 = bitcast %struct.dcomplex* %ref.tmp to i8*, !dbg !3326
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %6, i8* align 8 %7, i64 16, i1 false), !dbg !3326
  %real4 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp3, i32 0, i32 0, !dbg !3327
  store double 0.000000e+00, double* %real4, align 8, !dbg !3327
  %imag5 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %ref.tmp3, i32 0, i32 1, !dbg !3327
  store double 0.000000e+00, double* %imag5, align 8, !dbg !3327
  %8 = load %struct.dcomplex*, %struct.dcomplex** %u1.addr, align 8, !dbg !3328
  %9 = load i32, i32* %thread_id, align 4, !dbg !3329
  %idxprom6 = sext i32 %9 to i64, !dbg !3328
  %arrayidx7 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %8, i64 %idxprom6, !dbg !3328
  %10 = bitcast %struct.dcomplex* %arrayidx7 to i8*, !dbg !3330
  %11 = bitcast %struct.dcomplex* %ref.tmp3 to i8*, !dbg !3330
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %10, i8* align 8 %11, i64 16, i1 false), !dbg !3330
  %12 = load double*, double** %twiddle.addr, align 8, !dbg !3331
  %13 = load i32, i32* %thread_id, align 4, !dbg !3332
  %idxprom8 = sext i32 %13 to i64, !dbg !3331
  %arrayidx9 = getelementptr inbounds double, double* %12, i64 %idxprom8, !dbg !3331
  store double 0.000000e+00, double* %arrayidx9, align 8, !dbg !3333
  br label %return, !dbg !3334

return:                                           ; preds = %if.end, %if.then
  ret void, !dbg !3334
}

; Function Attrs: convergent noinline nounwind
define dso_local void @_Z13ipow46_devicediPd(double %a, i32 %exponent, double* %result) #0 !dbg !3335 {
entry:
  %a.addr = alloca double, align 8
  %exponent.addr = alloca i32, align 4
  %result.addr = alloca double*, align 8
  %q = alloca double, align 8
  %r = alloca double, align 8
  %n = alloca i32, align 4
  %n2 = alloca i32, align 4
  store double %a, double* %a.addr, align 8
  call void @llvm.dbg.declare(metadata double* %a.addr, metadata !3338, metadata !DIExpression()), !dbg !3339
  store i32 %exponent, i32* %exponent.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %exponent.addr, metadata !3340, metadata !DIExpression()), !dbg !3341
  store double* %result, double** %result.addr, align 8
  call void @llvm.dbg.declare(metadata double** %result.addr, metadata !3342, metadata !DIExpression()), !dbg !3343
  call void @llvm.dbg.declare(metadata double* %q, metadata !3344, metadata !DIExpression()), !dbg !3345
  call void @llvm.dbg.declare(metadata double* %r, metadata !3346, metadata !DIExpression()), !dbg !3347
  call void @llvm.dbg.declare(metadata i32* %n, metadata !3348, metadata !DIExpression()), !dbg !3349
  call void @llvm.dbg.declare(metadata i32* %n2, metadata !3350, metadata !DIExpression()), !dbg !3351
  %0 = load double*, double** %result.addr, align 8, !dbg !3352
  store double 1.000000e+00, double* %0, align 8, !dbg !3353
  %1 = load i32, i32* %exponent.addr, align 4, !dbg !3354
  %cmp = icmp eq i32 %1, 0, !dbg !3356
  br i1 %cmp, label %if.then, label %if.end, !dbg !3357

if.then:                                          ; preds = %entry
  br label %return, !dbg !3358

if.end:                                           ; preds = %entry
  %2 = load double, double* %a.addr, align 8, !dbg !3360
  store double %2, double* %q, align 8, !dbg !3361
  store double 1.000000e+00, double* %r, align 8, !dbg !3362
  %3 = load i32, i32* %exponent.addr, align 4, !dbg !3363
  store i32 %3, i32* %n, align 4, !dbg !3364
  br label %while.cond, !dbg !3365

while.cond:                                       ; preds = %if.end5, %if.end
  %4 = load i32, i32* %n, align 4, !dbg !3366
  %cmp1 = icmp sgt i32 %4, 1, !dbg !3367
  br i1 %cmp1, label %while.body, label %while.end, !dbg !3365

while.body:                                       ; preds = %while.cond
  %5 = load i32, i32* %n, align 4, !dbg !3368
  %div = sdiv i32 %5, 2, !dbg !3370
  store i32 %div, i32* %n2, align 4, !dbg !3371
  %6 = load i32, i32* %n2, align 4, !dbg !3372
  %mul = mul nsw i32 %6, 2, !dbg !3374
  %7 = load i32, i32* %n, align 4, !dbg !3375
  %cmp2 = icmp eq i32 %mul, %7, !dbg !3376
  br i1 %cmp2, label %if.then3, label %if.else, !dbg !3377

if.then3:                                         ; preds = %while.body
  %8 = load double, double* %q, align 8, !dbg !3378
  %call = call double @_Z13randlc_devicePdd(double* %q, double %8) #3, !dbg !3380
  %9 = load i32, i32* %n2, align 4, !dbg !3381
  store i32 %9, i32* %n, align 4, !dbg !3382
  br label %if.end5, !dbg !3383

if.else:                                          ; preds = %while.body
  %10 = load double, double* %q, align 8, !dbg !3384
  %call4 = call double @_Z13randlc_devicePdd(double* %r, double %10) #3, !dbg !3386
  %11 = load i32, i32* %n, align 4, !dbg !3387
  %sub = sub nsw i32 %11, 1, !dbg !3388
  store i32 %sub, i32* %n, align 4, !dbg !3389
  br label %if.end5

if.end5:                                          ; preds = %if.else, %if.then3
  br label %while.cond, !dbg !3365, !llvm.loop !3390

while.end:                                        ; preds = %while.cond
  %12 = load double, double* %q, align 8, !dbg !3392
  %call6 = call double @_Z13randlc_devicePdd(double* %r, double %12) #3, !dbg !3393
  %13 = load double, double* %r, align 8, !dbg !3394
  %14 = load double*, double** %result.addr, align 8, !dbg !3395
  store double %13, double* %14, align 8, !dbg !3396
  br label %return, !dbg !3397

return:                                           ; preds = %while.end, %if.then
  ret void, !dbg !3397
}

; Function Attrs: convergent noinline nounwind
define dso_local double @_Z13randlc_devicePdd(double* %x, double %a) #0 !dbg !3398 {
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
  call void @llvm.dbg.declare(metadata double** %x.addr, metadata !3399, metadata !DIExpression()), !dbg !3400
  store double %a, double* %a.addr, align 8
  call void @llvm.dbg.declare(metadata double* %a.addr, metadata !3401, metadata !DIExpression()), !dbg !3402
  call void @llvm.dbg.declare(metadata double* %t1, metadata !3403, metadata !DIExpression()), !dbg !3404
  call void @llvm.dbg.declare(metadata double* %t2, metadata !3405, metadata !DIExpression()), !dbg !3406
  call void @llvm.dbg.declare(metadata double* %t3, metadata !3407, metadata !DIExpression()), !dbg !3408
  call void @llvm.dbg.declare(metadata double* %t4, metadata !3409, metadata !DIExpression()), !dbg !3410
  call void @llvm.dbg.declare(metadata double* %a1, metadata !3411, metadata !DIExpression()), !dbg !3412
  call void @llvm.dbg.declare(metadata double* %a2, metadata !3413, metadata !DIExpression()), !dbg !3414
  call void @llvm.dbg.declare(metadata double* %x1, metadata !3415, metadata !DIExpression()), !dbg !3416
  call void @llvm.dbg.declare(metadata double* %x2, metadata !3417, metadata !DIExpression()), !dbg !3418
  call void @llvm.dbg.declare(metadata double* %z, metadata !3419, metadata !DIExpression()), !dbg !3420
  %0 = load double, double* %a.addr, align 8, !dbg !3421
  %mul = fmul contract double 0x3E80000000000000, %0, !dbg !3422
  store double %mul, double* %t1, align 8, !dbg !3423
  %1 = load double, double* %t1, align 8, !dbg !3424
  %conv = fptosi double %1 to i32, !dbg !3424
  %conv1 = sitofp i32 %conv to double, !dbg !3425
  store double %conv1, double* %a1, align 8, !dbg !3426
  %2 = load double, double* %a.addr, align 8, !dbg !3427
  %3 = load double, double* %a1, align 8, !dbg !3428
  %mul2 = fmul contract double 0x4160000000000000, %3, !dbg !3429
  %sub = fsub contract double %2, %mul2, !dbg !3430
  store double %sub, double* %a2, align 8, !dbg !3431
  %4 = load double*, double** %x.addr, align 8, !dbg !3432
  %5 = load double, double* %4, align 8, !dbg !3433
  %mul3 = fmul contract double 0x3E80000000000000, %5, !dbg !3434
  store double %mul3, double* %t1, align 8, !dbg !3435
  %6 = load double, double* %t1, align 8, !dbg !3436
  %conv4 = fptosi double %6 to i32, !dbg !3436
  %conv5 = sitofp i32 %conv4 to double, !dbg !3437
  store double %conv5, double* %x1, align 8, !dbg !3438
  %7 = load double*, double** %x.addr, align 8, !dbg !3439
  %8 = load double, double* %7, align 8, !dbg !3440
  %9 = load double, double* %x1, align 8, !dbg !3441
  %mul6 = fmul contract double 0x4160000000000000, %9, !dbg !3442
  %sub7 = fsub contract double %8, %mul6, !dbg !3443
  store double %sub7, double* %x2, align 8, !dbg !3444
  %10 = load double, double* %a1, align 8, !dbg !3445
  %11 = load double, double* %x2, align 8, !dbg !3446
  %mul8 = fmul contract double %10, %11, !dbg !3447
  %12 = load double, double* %a2, align 8, !dbg !3448
  %13 = load double, double* %x1, align 8, !dbg !3449
  %mul9 = fmul contract double %12, %13, !dbg !3450
  %add = fadd contract double %mul8, %mul9, !dbg !3451
  store double %add, double* %t1, align 8, !dbg !3452
  %14 = load double, double* %t1, align 8, !dbg !3453
  %mul10 = fmul contract double 0x3E80000000000000, %14, !dbg !3454
  %conv11 = fptosi double %mul10 to i32, !dbg !3455
  %conv12 = sitofp i32 %conv11 to double, !dbg !3456
  store double %conv12, double* %t2, align 8, !dbg !3457
  %15 = load double, double* %t1, align 8, !dbg !3458
  %16 = load double, double* %t2, align 8, !dbg !3459
  %mul13 = fmul contract double 0x4160000000000000, %16, !dbg !3460
  %sub14 = fsub contract double %15, %mul13, !dbg !3461
  store double %sub14, double* %z, align 8, !dbg !3462
  %17 = load double, double* %z, align 8, !dbg !3463
  %mul15 = fmul contract double 0x4160000000000000, %17, !dbg !3464
  %18 = load double, double* %a2, align 8, !dbg !3465
  %19 = load double, double* %x2, align 8, !dbg !3466
  %mul16 = fmul contract double %18, %19, !dbg !3467
  %add17 = fadd contract double %mul15, %mul16, !dbg !3468
  store double %add17, double* %t3, align 8, !dbg !3469
  %20 = load double, double* %t3, align 8, !dbg !3470
  %mul18 = fmul contract double 0x3D10000000000000, %20, !dbg !3471
  %conv19 = fptosi double %mul18 to i32, !dbg !3472
  %conv20 = sitofp i32 %conv19 to double, !dbg !3473
  store double %conv20, double* %t4, align 8, !dbg !3474
  %21 = load double, double* %t3, align 8, !dbg !3475
  %22 = load double, double* %t4, align 8, !dbg !3476
  %mul21 = fmul contract double 0x42D0000000000000, %22, !dbg !3477
  %sub22 = fsub contract double %21, %mul21, !dbg !3478
  %23 = load double*, double** %x.addr, align 8, !dbg !3479
  store double %sub22, double* %23, align 8, !dbg !3480
  %24 = load double*, double** %x.addr, align 8, !dbg !3481
  %25 = load double, double* %24, align 8, !dbg !3482
  %mul23 = fmul contract double 0x3D10000000000000, %25, !dbg !3483
  ret double %mul23, !dbg !3484
}

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #5

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #5

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() #5

; Function Attrs: convergent noinline nounwind
define internal i64 @_ZL9atomicCASPyyy(i64* %address, i64 %compare, i64 %val) #0 !dbg !3485 {
entry:
  %p.addr.i = alloca i64*, align 8
  call void @llvm.dbg.declare(metadata i64** %p.addr.i, metadata !3489, metadata !DIExpression()), !dbg !3491
  %compare.addr.i = alloca i64, align 8
  call void @llvm.dbg.declare(metadata i64* %compare.addr.i, metadata !3493, metadata !DIExpression()), !dbg !3494
  %val.addr.i = alloca i64, align 8
  call void @llvm.dbg.declare(metadata i64* %val.addr.i, metadata !3495, metadata !DIExpression()), !dbg !3496
  %address.addr = alloca i64*, align 8
  %compare.addr = alloca i64, align 8
  %val.addr = alloca i64, align 8
  store i64* %address, i64** %address.addr, align 8
  call void @llvm.dbg.declare(metadata i64** %address.addr, metadata !3497, metadata !DIExpression()), !dbg !3498
  store i64 %compare, i64* %compare.addr, align 8
  call void @llvm.dbg.declare(metadata i64* %compare.addr, metadata !3499, metadata !DIExpression()), !dbg !3500
  store i64 %val, i64* %val.addr, align 8
  call void @llvm.dbg.declare(metadata i64* %val.addr, metadata !3501, metadata !DIExpression()), !dbg !3502
  %0 = load i64*, i64** %address.addr, align 8, !dbg !3503
  %1 = load i64, i64* %compare.addr, align 8, !dbg !3504
  %2 = load i64, i64* %val.addr, align 8, !dbg !3505
  store i64* %0, i64** %p.addr.i, align 8
  store i64 %1, i64* %compare.addr.i, align 8
  store i64 %2, i64* %val.addr.i, align 8
  %3 = load i64*, i64** %p.addr.i, align 8, !dbg !3506
  %4 = load i64, i64* %compare.addr.i, align 8, !dbg !3507
  %5 = load i64, i64* %val.addr.i, align 8, !dbg !3508
  %6 = cmpxchg i64* %3, i64 %4, i64 %5 seq_cst seq_cst, !dbg !3509
  %7 = extractvalue { i64, i1 } %6, 0, !dbg !3509
  ret i64 %7, !dbg !3510
}

; Function Attrs: convergent nounwind readnone
declare i32 @llvm.nvvm.d2i.hi(double) #6

; Function Attrs: convergent nounwind
declare i32 @__nvvm_reflect(i8*) #7

; Function Attrs: convergent nounwind readnone
declare float @llvm.nvvm.fabs.ftz.f(float) #6

; Function Attrs: convergent nounwind readnone
declare float @llvm.nvvm.fabs.f(float) #6

; Function Attrs: convergent nounwind readnone
declare double @llvm.nvvm.mul.rn.d(double, double) #6

; Function Attrs: convergent nounwind readnone
declare double @llvm.nvvm.add.rn.d(double, double) #6

; Function Attrs: convergent nounwind readnone
declare i32 @llvm.nvvm.d2i.lo(double) #6

; Function Attrs: convergent nounwind readnone
declare double @llvm.nvvm.fma.rn.d(double, double, double) #6

; Function Attrs: convergent nounwind readnone
declare double @llvm.nvvm.lohi.i2d(i32, i32) #6

; Function Attrs: convergent nounwind readnone
declare double @llvm.nvvm.fabs.d(double) #6

attributes #0 = { convergent noinline nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="sm_20" "target-features"="+ptx42,+sm_20" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { argmemonly nounwind }
attributes #3 = { convergent nounwind }
attributes #4 = { convergent noinline nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="sm_20" "target-features"="+ptx42,+sm_20" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { nounwind readnone }
attributes #6 = { convergent nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #7 = { convergent nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #8 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.dbg.cu = !{!5}
!nvvm.annotations = !{!782, !783, !784, !785, !786, !787, !788, !789, !790, !791, !792, !793, !794, !795, !796, !797, !796, !798, !798, !798, !798, !799, !799, !798}
!llvm.ident = !{!800}
!nvvm.internalize.after.link = !{}
!nvvmir.version = !{!801}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 7, i32 0]}
!1 = !{i32 2, !"Dwarf Version", i32 2}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !6, producer: "clang version 9.0.0 (https://github.com/yebinchon/llvm-project/ a1efa594106d738d0b74c9e4e2b9b779eb8b7d25)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !7, retainedTypes: !8, imports: !24, nameTableKind: None)
!6 = !DIFile(filename: "ft.cu", directory: "/scratch/ah7226/NPB-GPU/CUDA/FT")
!7 = !{}
!8 = !{!9, !15, !17, !18, !19, !20, !21, !23}
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64)
!10 = !DIDerivedType(tag: DW_TAG_typedef, name: "dcomplex", file: !11, line: 81, baseType: !12)
!11 = !DIFile(filename: "./../common/npb-CPP.hpp", directory: "/scratch/ah7226/NPB-GPU/CUDA/FT")
!12 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !11, line: 81, size: 128, flags: DIFlagTypePassByValue, elements: !13, identifier: "_ZTS8dcomplex")
!13 = !{!14, !16}
!14 = !DIDerivedType(tag: DW_TAG_member, name: "real", scope: !12, file: !11, line: 81, baseType: !15, size: 64)
!15 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
!16 = !DIDerivedType(tag: DW_TAG_member, name: "imag", scope: !12, file: !11, line: 81, baseType: !15, size: 64, offset: 64)
!17 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !15, size: 64)
!18 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!19 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !20, size: 64)
!20 = !DIBasicType(name: "long long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !22, size: 64)
!22 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !23)
!23 = !DIBasicType(name: "long long int", size: 64, encoding: DW_ATE_signed)
!24 = !{!25, !31, !36, !38, !40, !42, !44, !48, !50, !52, !54, !56, !58, !60, !62, !64, !66, !68, !70, !72, !74, !76, !80, !82, !84, !86, !90, !95, !97, !99, !104, !108, !110, !112, !114, !116, !118, !120, !122, !124, !129, !133, !135, !139, !143, !145, !147, !149, !151, !153, !157, !159, !161, !166, !173, !177, !179, !181, !183, !185, !189, !191, !193, !197, !199, !201, !203, !205, !207, !209, !211, !213, !215, !219, !225, !227, !229, !233, !235, !237, !239, !241, !243, !245, !247, !251, !255, !257, !259, !263, !265, !267, !269, !271, !273, !275, !279, !285, !289, !294, !296, !300, !304, !318, !322, !326, !330, !334, !339, !341, !345, !349, !353, !361, !365, !369, !373, !377, !382, !388, !392, !396, !398, !406, !410, !417, !419, !421, !425, !429, !433, !437, !441, !446, !447, !448, !449, !451, !452, !453, !454, !455, !456, !457, !459, !460, !461, !462, !463, !467, !468, !469, !470, !471, !472, !473, !474, !475, !476, !477, !478, !479, !480, !481, !482, !483, !484, !485, !486, !487, !488, !489, !490, !491, !495, !497, !499, !501, !503, !505, !507, !509, !512, !514, !516, !518, !520, !522, !524, !526, !528, !530, !532, !534, !536, !538, !540, !542, !544, !546, !548, !550, !552, !554, !556, !558, !560, !562, !564, !566, !568, !570, !572, !574, !576, !578, !580, !582, !584, !586, !588, !590, !592, !594, !596, !598, !600, !602, !604, !610, !616, !621, !625, !627, !629, !631, !633, !640, !644, !648, !652, !656, !660, !665, !669, !671, !675, !681, !685, !690, !692, !694, !698, !702, !706, !708, !710, !712, !714, !718, !720, !722, !726, !730, !734, !738, !742, !744, !746, !753, !757, !761, !765, !767, !769, !773, !777, !778, !779, !780, !781}
!25 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !27, file: !28, line: 223)
!26 = !DINamespace(name: "std", scope: null)
!27 = !DISubprogram(name: "abs", linkageName: "_ZL3absi", scope: !28, file: !28, line: 53, type: !29, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!28 = !DIFile(filename: "llvm-install-tulip/lib/clang/9.0.0/include/__clang_cuda_math_forward_declares.h", directory: "/scratch/ah7226")
!29 = !DISubroutineType(types: !30)
!30 = !{!18, !18}
!31 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !32, file: !28, line: 224)
!32 = !DISubprogram(name: "acos", linkageName: "_ZL4acosf", scope: !28, file: !28, line: 55, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!33 = !DISubroutineType(types: !34)
!34 = !{!35, !35}
!35 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!36 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !37, file: !28, line: 225)
!37 = !DISubprogram(name: "acosh", linkageName: "_ZL5acoshf", scope: !28, file: !28, line: 57, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!38 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !39, file: !28, line: 226)
!39 = !DISubprogram(name: "asin", linkageName: "_ZL4asinf", scope: !28, file: !28, line: 59, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!40 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !41, file: !28, line: 227)
!41 = !DISubprogram(name: "asinh", linkageName: "_ZL5asinhf", scope: !28, file: !28, line: 61, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!42 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !43, file: !28, line: 228)
!43 = !DISubprogram(name: "atan", linkageName: "_ZL4atanf", scope: !28, file: !28, line: 65, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!44 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !45, file: !28, line: 229)
!45 = !DISubprogram(name: "atan2", linkageName: "_ZL5atan2ff", scope: !28, file: !28, line: 63, type: !46, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!46 = !DISubroutineType(types: !47)
!47 = !{!35, !35, !35}
!48 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !49, file: !28, line: 230)
!49 = !DISubprogram(name: "atanh", linkageName: "_ZL5atanhf", scope: !28, file: !28, line: 67, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!50 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !51, file: !28, line: 231)
!51 = !DISubprogram(name: "cbrt", linkageName: "_ZL4cbrtf", scope: !28, file: !28, line: 69, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!52 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !53, file: !28, line: 232)
!53 = !DISubprogram(name: "ceil", linkageName: "_ZL4ceilf", scope: !28, file: !28, line: 71, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!54 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !55, file: !28, line: 233)
!55 = !DISubprogram(name: "copysign", linkageName: "_ZL8copysignff", scope: !28, file: !28, line: 73, type: !46, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!56 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !57, file: !28, line: 234)
!57 = !DISubprogram(name: "cos", linkageName: "_ZL3cosf", scope: !28, file: !28, line: 75, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!58 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !59, file: !28, line: 235)
!59 = !DISubprogram(name: "cosh", linkageName: "_ZL4coshf", scope: !28, file: !28, line: 77, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!60 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !61, file: !28, line: 236)
!61 = !DISubprogram(name: "erf", linkageName: "_ZL3erff", scope: !28, file: !28, line: 81, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!62 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !63, file: !28, line: 237)
!63 = !DISubprogram(name: "erfc", linkageName: "_ZL4erfcf", scope: !28, file: !28, line: 79, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!64 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !65, file: !28, line: 238)
!65 = !DISubprogram(name: "exp", linkageName: "_ZL3expf", scope: !28, file: !28, line: 85, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!66 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !67, file: !28, line: 239)
!67 = !DISubprogram(name: "exp2", linkageName: "_ZL4exp2f", scope: !28, file: !28, line: 83, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!68 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !69, file: !28, line: 240)
!69 = !DISubprogram(name: "expm1", linkageName: "_ZL5expm1f", scope: !28, file: !28, line: 87, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!70 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !71, file: !28, line: 241)
!71 = !DISubprogram(name: "fabs", linkageName: "_ZL4fabsf", scope: !28, file: !28, line: 89, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!72 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !73, file: !28, line: 242)
!73 = !DISubprogram(name: "fdim", linkageName: "_ZL4fdimff", scope: !28, file: !28, line: 91, type: !46, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!74 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !75, file: !28, line: 243)
!75 = !DISubprogram(name: "floor", linkageName: "_ZL5floorf", scope: !28, file: !28, line: 93, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!76 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !77, file: !28, line: 244)
!77 = !DISubprogram(name: "fma", linkageName: "_ZL3fmafff", scope: !28, file: !28, line: 95, type: !78, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!78 = !DISubroutineType(types: !79)
!79 = !{!35, !35, !35, !35}
!80 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !81, file: !28, line: 245)
!81 = !DISubprogram(name: "fmax", linkageName: "_ZL4fmaxff", scope: !28, file: !28, line: 97, type: !46, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!82 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !83, file: !28, line: 246)
!83 = !DISubprogram(name: "fmin", linkageName: "_ZL4fminff", scope: !28, file: !28, line: 99, type: !46, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!84 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !85, file: !28, line: 247)
!85 = !DISubprogram(name: "fmod", linkageName: "_ZL4fmodff", scope: !28, file: !28, line: 101, type: !46, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!86 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !87, file: !28, line: 248)
!87 = !DISubprogram(name: "fpclassify", linkageName: "_ZL10fpclassifyf", scope: !28, file: !28, line: 103, type: !88, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!88 = !DISubroutineType(types: !89)
!89 = !{!18, !35}
!90 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !91, file: !28, line: 249)
!91 = !DISubprogram(name: "frexp", linkageName: "_ZL5frexpfPi", scope: !28, file: !28, line: 105, type: !92, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!92 = !DISubroutineType(types: !93)
!93 = !{!35, !35, !94}
!94 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !18, size: 64)
!95 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !96, file: !28, line: 250)
!96 = !DISubprogram(name: "hypot", linkageName: "_ZL5hypotff", scope: !28, file: !28, line: 107, type: !46, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!97 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !98, file: !28, line: 251)
!98 = !DISubprogram(name: "ilogb", linkageName: "_ZL5ilogbf", scope: !28, file: !28, line: 109, type: !88, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!99 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !100, file: !28, line: 252)
!100 = !DISubprogram(name: "isfinite", linkageName: "_ZL8isfinitef", scope: !28, file: !28, line: 114, type: !101, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!101 = !DISubroutineType(types: !102)
!102 = !{!103, !35}
!103 = !DIBasicType(name: "bool", size: 8, encoding: DW_ATE_boolean)
!104 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !105, file: !28, line: 253)
!105 = !DISubprogram(name: "isgreater", linkageName: "_ZL9isgreaterff", scope: !28, file: !28, line: 118, type: !106, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!106 = !DISubroutineType(types: !107)
!107 = !{!103, !35, !35}
!108 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !109, file: !28, line: 254)
!109 = !DISubprogram(name: "isgreaterequal", linkageName: "_ZL14isgreaterequalff", scope: !28, file: !28, line: 117, type: !106, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!110 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !111, file: !28, line: 255)
!111 = !DISubprogram(name: "isinf", linkageName: "_ZL5isinff", scope: !28, file: !28, line: 123, type: !101, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!112 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !113, file: !28, line: 256)
!113 = !DISubprogram(name: "isless", linkageName: "_ZL6islessff", scope: !28, file: !28, line: 127, type: !106, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!114 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !115, file: !28, line: 257)
!115 = !DISubprogram(name: "islessequal", linkageName: "_ZL11islessequalff", scope: !28, file: !28, line: 126, type: !106, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!116 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !117, file: !28, line: 258)
!117 = !DISubprogram(name: "islessgreater", linkageName: "_ZL13islessgreaterff", scope: !28, file: !28, line: 129, type: !106, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!118 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !119, file: !28, line: 259)
!119 = !DISubprogram(name: "isnan", linkageName: "_ZL5isnanf", scope: !28, file: !28, line: 134, type: !101, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!120 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !121, file: !28, line: 260)
!121 = !DISubprogram(name: "isnormal", linkageName: "_ZL8isnormalf", scope: !28, file: !28, line: 136, type: !101, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!122 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !123, file: !28, line: 261)
!123 = !DISubprogram(name: "isunordered", linkageName: "_ZL11isunorderedff", scope: !28, file: !28, line: 138, type: !106, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!124 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !125, file: !28, line: 262)
!125 = !DISubprogram(name: "labs", linkageName: "_ZL4labsl", scope: !28, file: !28, line: 139, type: !126, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!126 = !DISubroutineType(types: !127)
!127 = !{!128, !128}
!128 = !DIBasicType(name: "long int", size: 64, encoding: DW_ATE_signed)
!129 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !130, file: !28, line: 263)
!130 = !DISubprogram(name: "ldexp", linkageName: "_ZL5ldexpfi", scope: !28, file: !28, line: 141, type: !131, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!131 = !DISubroutineType(types: !132)
!132 = !{!35, !35, !18}
!133 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !134, file: !28, line: 264)
!134 = !DISubprogram(name: "lgamma", linkageName: "_ZL6lgammaf", scope: !28, file: !28, line: 143, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!135 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !136, file: !28, line: 265)
!136 = !DISubprogram(name: "llabs", linkageName: "_ZL5llabsx", scope: !28, file: !28, line: 144, type: !137, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!137 = !DISubroutineType(types: !138)
!138 = !{!23, !23}
!139 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !140, file: !28, line: 266)
!140 = !DISubprogram(name: "llrint", linkageName: "_ZL6llrintf", scope: !28, file: !28, line: 146, type: !141, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!141 = !DISubroutineType(types: !142)
!142 = !{!23, !35}
!143 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !144, file: !28, line: 267)
!144 = !DISubprogram(name: "log", linkageName: "_ZL3logf", scope: !28, file: !28, line: 159, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!145 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !146, file: !28, line: 268)
!146 = !DISubprogram(name: "log10", linkageName: "_ZL5log10f", scope: !28, file: !28, line: 148, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!147 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !148, file: !28, line: 269)
!148 = !DISubprogram(name: "log1p", linkageName: "_ZL5log1pf", scope: !28, file: !28, line: 150, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!149 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !150, file: !28, line: 270)
!150 = !DISubprogram(name: "log2", linkageName: "_ZL4log2f", scope: !28, file: !28, line: 152, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!151 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !152, file: !28, line: 271)
!152 = !DISubprogram(name: "logb", linkageName: "_ZL4logbf", scope: !28, file: !28, line: 154, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!153 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !154, file: !28, line: 272)
!154 = !DISubprogram(name: "lrint", linkageName: "_ZL5lrintf", scope: !28, file: !28, line: 161, type: !155, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!155 = !DISubroutineType(types: !156)
!156 = !{!128, !35}
!157 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !158, file: !28, line: 273)
!158 = !DISubprogram(name: "lround", linkageName: "_ZL6lroundf", scope: !28, file: !28, line: 163, type: !155, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!159 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !160, file: !28, line: 274)
!160 = !DISubprogram(name: "llround", linkageName: "_ZL7llroundf", scope: !28, file: !28, line: 164, type: !141, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!161 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !162, file: !28, line: 275)
!162 = !DISubprogram(name: "modf", linkageName: "_ZL4modffPf", scope: !28, file: !28, line: 166, type: !163, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!163 = !DISubroutineType(types: !164)
!164 = !{!35, !35, !165}
!165 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !35, size: 64)
!166 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !167, file: !28, line: 276)
!167 = !DISubprogram(name: "nan", linkageName: "_ZL3nanPKc", scope: !28, file: !28, line: 167, type: !168, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!168 = !DISubroutineType(types: !169)
!169 = !{!15, !170}
!170 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !171, size: 64)
!171 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !172)
!172 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!173 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !174, file: !28, line: 277)
!174 = !DISubprogram(name: "nanf", linkageName: "_ZL4nanfPKc", scope: !28, file: !28, line: 168, type: !175, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!175 = !DISubroutineType(types: !176)
!176 = !{!35, !170}
!177 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !178, file: !28, line: 278)
!178 = !DISubprogram(name: "nearbyint", linkageName: "_ZL9nearbyintf", scope: !28, file: !28, line: 170, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!179 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !180, file: !28, line: 279)
!180 = !DISubprogram(name: "nextafter", linkageName: "_ZL9nextafterff", scope: !28, file: !28, line: 172, type: !46, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!181 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !182, file: !28, line: 280)
!182 = !DISubprogram(name: "pow", linkageName: "_ZL3powfi", scope: !28, file: !28, line: 176, type: !131, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!183 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !184, file: !28, line: 281)
!184 = !DISubprogram(name: "remainder", linkageName: "_ZL9remainderff", scope: !28, file: !28, line: 178, type: !46, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!185 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !186, file: !28, line: 282)
!186 = !DISubprogram(name: "remquo", linkageName: "_ZL6remquoffPi", scope: !28, file: !28, line: 180, type: !187, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!187 = !DISubroutineType(types: !188)
!188 = !{!35, !35, !35, !94}
!189 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !190, file: !28, line: 283)
!190 = !DISubprogram(name: "rint", linkageName: "_ZL4rintf", scope: !28, file: !28, line: 182, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!191 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !192, file: !28, line: 284)
!192 = !DISubprogram(name: "round", linkageName: "_ZL5roundf", scope: !28, file: !28, line: 184, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!193 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !194, file: !28, line: 285)
!194 = !DISubprogram(name: "scalbln", linkageName: "_ZL7scalblnfl", scope: !28, file: !28, line: 186, type: !195, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!195 = !DISubroutineType(types: !196)
!196 = !{!35, !35, !128}
!197 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !198, file: !28, line: 286)
!198 = !DISubprogram(name: "scalbn", linkageName: "_ZL6scalbnfi", scope: !28, file: !28, line: 188, type: !131, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!199 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !200, file: !28, line: 287)
!200 = !DISubprogram(name: "signbit", linkageName: "_ZL7signbitf", scope: !28, file: !28, line: 190, type: !101, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!201 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !202, file: !28, line: 288)
!202 = !DISubprogram(name: "sin", linkageName: "_ZL3sinf", scope: !28, file: !28, line: 192, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!203 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !204, file: !28, line: 289)
!204 = !DISubprogram(name: "sinh", linkageName: "_ZL4sinhf", scope: !28, file: !28, line: 194, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!205 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !206, file: !28, line: 290)
!206 = !DISubprogram(name: "sqrt", linkageName: "_ZL4sqrtf", scope: !28, file: !28, line: 196, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!207 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !208, file: !28, line: 291)
!208 = !DISubprogram(name: "tan", linkageName: "_ZL3tanf", scope: !28, file: !28, line: 198, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!209 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !210, file: !28, line: 292)
!210 = !DISubprogram(name: "tanh", linkageName: "_ZL4tanhf", scope: !28, file: !28, line: 200, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!211 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !212, file: !28, line: 293)
!212 = !DISubprogram(name: "tgamma", linkageName: "_ZL6tgammaf", scope: !28, file: !28, line: 202, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!213 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !214, file: !28, line: 294)
!214 = !DISubprogram(name: "trunc", linkageName: "_ZL5truncf", scope: !28, file: !28, line: 204, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!215 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !216, file: !218, line: 52)
!216 = !DISubprogram(name: "abs", scope: !217, file: !217, line: 848, type: !29, flags: DIFlagPrototyped, spFlags: 0)
!217 = !DIFile(filename: "/usr/include/stdlib.h", directory: "")
!218 = !DIFile(filename: "/usr/lib/gcc/x86_64-linux-gnu/11/../../../../include/c++/11/bits/std_abs.h", directory: "")
!219 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !220, file: !224, line: 83)
!220 = !DISubprogram(name: "acos", scope: !221, file: !221, line: 53, type: !222, flags: DIFlagPrototyped, spFlags: 0)
!221 = !DIFile(filename: "/usr/include/x86_64-linux-gnu/bits/mathcalls.h", directory: "")
!222 = !DISubroutineType(types: !223)
!223 = !{!15, !15}
!224 = !DIFile(filename: "/usr/lib/gcc/x86_64-linux-gnu/11/../../../../include/c++/11/cmath", directory: "")
!225 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !226, file: !224, line: 102)
!226 = !DISubprogram(name: "asin", scope: !221, file: !221, line: 55, type: !222, flags: DIFlagPrototyped, spFlags: 0)
!227 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !228, file: !224, line: 121)
!228 = !DISubprogram(name: "atan", scope: !221, file: !221, line: 57, type: !222, flags: DIFlagPrototyped, spFlags: 0)
!229 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !230, file: !224, line: 140)
!230 = !DISubprogram(name: "atan2", scope: !221, file: !221, line: 59, type: !231, flags: DIFlagPrototyped, spFlags: 0)
!231 = !DISubroutineType(types: !232)
!232 = !{!15, !15, !15}
!233 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !234, file: !224, line: 161)
!234 = !DISubprogram(name: "ceil", scope: !221, file: !221, line: 159, type: !222, flags: DIFlagPrototyped, spFlags: 0)
!235 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !236, file: !224, line: 180)
!236 = !DISubprogram(name: "cos", scope: !221, file: !221, line: 62, type: !222, flags: DIFlagPrototyped, spFlags: 0)
!237 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !238, file: !224, line: 199)
!238 = !DISubprogram(name: "cosh", scope: !221, file: !221, line: 71, type: !222, flags: DIFlagPrototyped, spFlags: 0)
!239 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !240, file: !224, line: 218)
!240 = !DISubprogram(name: "exp", scope: !221, file: !221, line: 95, type: !222, flags: DIFlagPrototyped, spFlags: 0)
!241 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !242, file: !224, line: 237)
!242 = !DISubprogram(name: "fabs", scope: !221, file: !221, line: 162, type: !222, flags: DIFlagPrototyped, spFlags: 0)
!243 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !244, file: !224, line: 256)
!244 = !DISubprogram(name: "floor", scope: !221, file: !221, line: 165, type: !222, flags: DIFlagPrototyped, spFlags: 0)
!245 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !246, file: !224, line: 275)
!246 = !DISubprogram(name: "fmod", scope: !221, file: !221, line: 168, type: !231, flags: DIFlagPrototyped, spFlags: 0)
!247 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !248, file: !224, line: 296)
!248 = !DISubprogram(name: "frexp", scope: !221, file: !221, line: 98, type: !249, flags: DIFlagPrototyped, spFlags: 0)
!249 = !DISubroutineType(types: !250)
!250 = !{!15, !15, !94}
!251 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !252, file: !224, line: 315)
!252 = !DISubprogram(name: "ldexp", scope: !221, file: !221, line: 101, type: !253, flags: DIFlagPrototyped, spFlags: 0)
!253 = !DISubroutineType(types: !254)
!254 = !{!15, !15, !18}
!255 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !256, file: !224, line: 334)
!256 = !DISubprogram(name: "log", scope: !221, file: !221, line: 104, type: !222, flags: DIFlagPrototyped, spFlags: 0)
!257 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !258, file: !224, line: 353)
!258 = !DISubprogram(name: "log10", scope: !221, file: !221, line: 107, type: !222, flags: DIFlagPrototyped, spFlags: 0)
!259 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !260, file: !224, line: 372)
!260 = !DISubprogram(name: "modf", scope: !221, file: !221, line: 110, type: !261, flags: DIFlagPrototyped, spFlags: 0)
!261 = !DISubroutineType(types: !262)
!262 = !{!15, !15, !17}
!263 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !264, file: !224, line: 384)
!264 = !DISubprogram(name: "pow", scope: !221, file: !221, line: 140, type: !231, flags: DIFlagPrototyped, spFlags: 0)
!265 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !266, file: !224, line: 421)
!266 = !DISubprogram(name: "sin", scope: !221, file: !221, line: 64, type: !222, flags: DIFlagPrototyped, spFlags: 0)
!267 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !268, file: !224, line: 440)
!268 = !DISubprogram(name: "sinh", scope: !221, file: !221, line: 73, type: !222, flags: DIFlagPrototyped, spFlags: 0)
!269 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !270, file: !224, line: 459)
!270 = !DISubprogram(name: "sqrt", scope: !221, file: !221, line: 143, type: !222, flags: DIFlagPrototyped, spFlags: 0)
!271 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !272, file: !224, line: 478)
!272 = !DISubprogram(name: "tan", scope: !221, file: !221, line: 66, type: !222, flags: DIFlagPrototyped, spFlags: 0)
!273 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !274, file: !224, line: 497)
!274 = !DISubprogram(name: "tanh", scope: !221, file: !221, line: 75, type: !222, flags: DIFlagPrototyped, spFlags: 0)
!275 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !276, file: !278, line: 127)
!276 = !DIDerivedType(tag: DW_TAG_typedef, name: "div_t", file: !217, line: 63, baseType: !277)
!277 = !DICompositeType(tag: DW_TAG_structure_type, file: !217, line: 59, flags: DIFlagFwdDecl, identifier: "_ZTS5div_t")
!278 = !DIFile(filename: "/usr/lib/gcc/x86_64-linux-gnu/11/../../../../include/c++/11/cstdlib", directory: "")
!279 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !280, file: !278, line: 128)
!280 = !DIDerivedType(tag: DW_TAG_typedef, name: "ldiv_t", file: !217, line: 71, baseType: !281)
!281 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !217, line: 67, size: 128, flags: DIFlagTypePassByValue, elements: !282, identifier: "_ZTS6ldiv_t")
!282 = !{!283, !284}
!283 = !DIDerivedType(tag: DW_TAG_member, name: "quot", scope: !281, file: !217, line: 69, baseType: !128, size: 64)
!284 = !DIDerivedType(tag: DW_TAG_member, name: "rem", scope: !281, file: !217, line: 70, baseType: !128, size: 64, offset: 64)
!285 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !286, file: !278, line: 130)
!286 = !DISubprogram(name: "abort", scope: !217, file: !217, line: 598, type: !287, flags: DIFlagPrototyped | DIFlagNoReturn, spFlags: 0)
!287 = !DISubroutineType(types: !288)
!288 = !{null}
!289 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !290, file: !278, line: 134)
!290 = !DISubprogram(name: "atexit", scope: !217, file: !217, line: 602, type: !291, flags: DIFlagPrototyped, spFlags: 0)
!291 = !DISubroutineType(types: !292)
!292 = !{!18, !293}
!293 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !287, size: 64)
!294 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !295, file: !278, line: 140)
!295 = !DISubprogram(name: "atof", scope: !217, file: !217, line: 102, type: !168, flags: DIFlagPrototyped, spFlags: 0)
!296 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !297, file: !278, line: 141)
!297 = !DISubprogram(name: "atoi", scope: !217, file: !217, line: 105, type: !298, flags: DIFlagPrototyped, spFlags: 0)
!298 = !DISubroutineType(types: !299)
!299 = !{!18, !170}
!300 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !301, file: !278, line: 142)
!301 = !DISubprogram(name: "atol", scope: !217, file: !217, line: 108, type: !302, flags: DIFlagPrototyped, spFlags: 0)
!302 = !DISubroutineType(types: !303)
!303 = !{!128, !170}
!304 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !305, file: !278, line: 143)
!305 = !DISubprogram(name: "bsearch", scope: !217, file: !217, line: 828, type: !306, flags: DIFlagPrototyped, spFlags: 0)
!306 = !DISubroutineType(types: !307)
!307 = !{!308, !309, !309, !311, !311, !314}
!308 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!309 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !310, size: 64)
!310 = !DIDerivedType(tag: DW_TAG_const_type, baseType: null)
!311 = !DIDerivedType(tag: DW_TAG_typedef, name: "size_t", file: !312, line: 46, baseType: !313)
!312 = !DIFile(filename: "llvm-install-tulip/lib/clang/9.0.0/include/stddef.h", directory: "/scratch/ah7226")
!313 = !DIBasicType(name: "long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!314 = !DIDerivedType(tag: DW_TAG_typedef, name: "__compar_fn_t", file: !217, line: 816, baseType: !315)
!315 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !316, size: 64)
!316 = !DISubroutineType(types: !317)
!317 = !{!18, !309, !309}
!318 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !319, file: !278, line: 144)
!319 = !DISubprogram(name: "calloc", scope: !217, file: !217, line: 543, type: !320, flags: DIFlagPrototyped, spFlags: 0)
!320 = !DISubroutineType(types: !321)
!321 = !{!308, !311, !311}
!322 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !323, file: !278, line: 145)
!323 = !DISubprogram(name: "div", scope: !217, file: !217, line: 860, type: !324, flags: DIFlagPrototyped, spFlags: 0)
!324 = !DISubroutineType(types: !325)
!325 = !{!276, !18, !18}
!326 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !327, file: !278, line: 146)
!327 = !DISubprogram(name: "exit", scope: !217, file: !217, line: 624, type: !328, flags: DIFlagPrototyped | DIFlagNoReturn, spFlags: 0)
!328 = !DISubroutineType(types: !329)
!329 = !{null, !18}
!330 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !331, file: !278, line: 147)
!331 = !DISubprogram(name: "free", scope: !217, file: !217, line: 555, type: !332, flags: DIFlagPrototyped, spFlags: 0)
!332 = !DISubroutineType(types: !333)
!333 = !{null, !308}
!334 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !335, file: !278, line: 148)
!335 = !DISubprogram(name: "getenv", scope: !217, file: !217, line: 641, type: !336, flags: DIFlagPrototyped, spFlags: 0)
!336 = !DISubroutineType(types: !337)
!337 = !{!338, !170}
!338 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !172, size: 64)
!339 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !340, file: !278, line: 149)
!340 = !DISubprogram(name: "labs", scope: !217, file: !217, line: 849, type: !126, flags: DIFlagPrototyped, spFlags: 0)
!341 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !342, file: !278, line: 150)
!342 = !DISubprogram(name: "ldiv", scope: !217, file: !217, line: 862, type: !343, flags: DIFlagPrototyped, spFlags: 0)
!343 = !DISubroutineType(types: !344)
!344 = !{!280, !128, !128}
!345 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !346, file: !278, line: 151)
!346 = !DISubprogram(name: "malloc", scope: !217, file: !217, line: 540, type: !347, flags: DIFlagPrototyped, spFlags: 0)
!347 = !DISubroutineType(types: !348)
!348 = !{!308, !311}
!349 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !350, file: !278, line: 153)
!350 = !DISubprogram(name: "mblen", scope: !217, file: !217, line: 930, type: !351, flags: DIFlagPrototyped, spFlags: 0)
!351 = !DISubroutineType(types: !352)
!352 = !{!18, !170, !311}
!353 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !354, file: !278, line: 154)
!354 = !DISubprogram(name: "mbstowcs", scope: !217, file: !217, line: 941, type: !355, flags: DIFlagPrototyped, spFlags: 0)
!355 = !DISubroutineType(types: !356)
!356 = !{!311, !357, !360, !311}
!357 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !358)
!358 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !359, size: 64)
!359 = !DIBasicType(name: "wchar_t", size: 32, encoding: DW_ATE_signed)
!360 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !170)
!361 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !362, file: !278, line: 155)
!362 = !DISubprogram(name: "mbtowc", scope: !217, file: !217, line: 933, type: !363, flags: DIFlagPrototyped, spFlags: 0)
!363 = !DISubroutineType(types: !364)
!364 = !{!18, !357, !360, !311}
!365 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !366, file: !278, line: 157)
!366 = !DISubprogram(name: "qsort", scope: !217, file: !217, line: 838, type: !367, flags: DIFlagPrototyped, spFlags: 0)
!367 = !DISubroutineType(types: !368)
!368 = !{null, !308, !311, !311, !314}
!369 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !370, file: !278, line: 163)
!370 = !DISubprogram(name: "rand", scope: !217, file: !217, line: 454, type: !371, flags: DIFlagPrototyped, spFlags: 0)
!371 = !DISubroutineType(types: !372)
!372 = !{!18}
!373 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !374, file: !278, line: 164)
!374 = !DISubprogram(name: "realloc", scope: !217, file: !217, line: 551, type: !375, flags: DIFlagPrototyped, spFlags: 0)
!375 = !DISubroutineType(types: !376)
!376 = !{!308, !308, !311}
!377 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !378, file: !278, line: 165)
!378 = !DISubprogram(name: "srand", scope: !217, file: !217, line: 456, type: !379, flags: DIFlagPrototyped, spFlags: 0)
!379 = !DISubroutineType(types: !380)
!380 = !{null, !381}
!381 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!382 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !383, file: !278, line: 166)
!383 = !DISubprogram(name: "strtod", scope: !217, file: !217, line: 118, type: !384, flags: DIFlagPrototyped, spFlags: 0)
!384 = !DISubroutineType(types: !385)
!385 = !{!15, !360, !386}
!386 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !387)
!387 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !338, size: 64)
!388 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !389, file: !278, line: 167)
!389 = !DISubprogram(name: "strtol", scope: !217, file: !217, line: 177, type: !390, flags: DIFlagPrototyped, spFlags: 0)
!390 = !DISubroutineType(types: !391)
!391 = !{!128, !360, !386, !18}
!392 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !393, file: !278, line: 168)
!393 = !DISubprogram(name: "strtoul", scope: !217, file: !217, line: 181, type: !394, flags: DIFlagPrototyped, spFlags: 0)
!394 = !DISubroutineType(types: !395)
!395 = !{!313, !360, !386, !18}
!396 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !397, file: !278, line: 169)
!397 = !DISubprogram(name: "system", scope: !217, file: !217, line: 791, type: !298, flags: DIFlagPrototyped, spFlags: 0)
!398 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !399, file: !278, line: 171)
!399 = !DISubprogram(name: "wcstombs", scope: !217, file: !217, line: 945, type: !400, flags: DIFlagPrototyped, spFlags: 0)
!400 = !DISubroutineType(types: !401)
!401 = !{!311, !402, !403, !311}
!402 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !338)
!403 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !404)
!404 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !405, size: 64)
!405 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !359)
!406 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !407, file: !278, line: 172)
!407 = !DISubprogram(name: "wctomb", scope: !217, file: !217, line: 937, type: !408, flags: DIFlagPrototyped, spFlags: 0)
!408 = !DISubroutineType(types: !409)
!409 = !{!18, !338, !359}
!410 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !411, entity: !412, file: !278, line: 200)
!411 = !DINamespace(name: "__gnu_cxx", scope: null)
!412 = !DIDerivedType(tag: DW_TAG_typedef, name: "lldiv_t", file: !217, line: 81, baseType: !413)
!413 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !217, line: 77, size: 128, flags: DIFlagTypePassByValue, elements: !414, identifier: "_ZTS7lldiv_t")
!414 = !{!415, !416}
!415 = !DIDerivedType(tag: DW_TAG_member, name: "quot", scope: !413, file: !217, line: 79, baseType: !23, size: 64)
!416 = !DIDerivedType(tag: DW_TAG_member, name: "rem", scope: !413, file: !217, line: 80, baseType: !23, size: 64, offset: 64)
!417 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !411, entity: !418, file: !278, line: 206)
!418 = !DISubprogram(name: "_Exit", scope: !217, file: !217, line: 636, type: !328, flags: DIFlagPrototyped | DIFlagNoReturn, spFlags: 0)
!419 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !411, entity: !420, file: !278, line: 210)
!420 = !DISubprogram(name: "llabs", scope: !217, file: !217, line: 852, type: !137, flags: DIFlagPrototyped, spFlags: 0)
!421 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !411, entity: !422, file: !278, line: 216)
!422 = !DISubprogram(name: "lldiv", scope: !217, file: !217, line: 866, type: !423, flags: DIFlagPrototyped, spFlags: 0)
!423 = !DISubroutineType(types: !424)
!424 = !{!412, !23, !23}
!425 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !411, entity: !426, file: !278, line: 227)
!426 = !DISubprogram(name: "atoll", scope: !217, file: !217, line: 113, type: !427, flags: DIFlagPrototyped, spFlags: 0)
!427 = !DISubroutineType(types: !428)
!428 = !{!23, !170}
!429 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !411, entity: !430, file: !278, line: 228)
!430 = !DISubprogram(name: "strtoll", scope: !217, file: !217, line: 201, type: !431, flags: DIFlagPrototyped, spFlags: 0)
!431 = !DISubroutineType(types: !432)
!432 = !{!23, !360, !386, !18}
!433 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !411, entity: !434, file: !278, line: 229)
!434 = !DISubprogram(name: "strtoull", scope: !217, file: !217, line: 206, type: !435, flags: DIFlagPrototyped, spFlags: 0)
!435 = !DISubroutineType(types: !436)
!436 = !{!20, !360, !386, !18}
!437 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !411, entity: !438, file: !278, line: 231)
!438 = !DISubprogram(name: "strtof", scope: !217, file: !217, line: 124, type: !439, flags: DIFlagPrototyped, spFlags: 0)
!439 = !DISubroutineType(types: !440)
!440 = !{!35, !360, !386}
!441 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !411, entity: !442, file: !278, line: 232)
!442 = !DISubprogram(name: "strtold", scope: !217, file: !217, line: 127, type: !443, flags: DIFlagPrototyped, spFlags: 0)
!443 = !DISubroutineType(types: !444)
!444 = !{!445, !360, !386}
!445 = !DIBasicType(name: "long double", size: 64, encoding: DW_ATE_float)
!446 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !412, file: !278, line: 240)
!447 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !418, file: !278, line: 242)
!448 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !420, file: !278, line: 244)
!449 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !450, file: !278, line: 245)
!450 = !DISubprogram(name: "div", linkageName: "_ZN9__gnu_cxx3divExx", scope: !411, file: !278, line: 213, type: !423, flags: DIFlagPrototyped, spFlags: 0)
!451 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !422, file: !278, line: 246)
!452 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !426, file: !278, line: 248)
!453 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !438, file: !278, line: 249)
!454 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !430, file: !278, line: 250)
!455 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !434, file: !278, line: 251)
!456 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !442, file: !278, line: 252)
!457 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !286, file: !458, line: 38)
!458 = !DIFile(filename: "/usr/lib/gcc/x86_64-linux-gnu/11/../../../../include/c++/11/stdlib.h", directory: "")
!459 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !290, file: !458, line: 39)
!460 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !327, file: !458, line: 40)
!461 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !276, file: !458, line: 51)
!462 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !280, file: !458, line: 52)
!463 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !464, file: !458, line: 54)
!464 = !DISubprogram(name: "abs", linkageName: "_ZSt3abse", scope: !26, file: !218, line: 79, type: !465, flags: DIFlagPrototyped, spFlags: 0)
!465 = !DISubroutineType(types: !466)
!466 = !{!445, !445}
!467 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !295, file: !458, line: 55)
!468 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !297, file: !458, line: 56)
!469 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !301, file: !458, line: 57)
!470 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !305, file: !458, line: 58)
!471 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !319, file: !458, line: 59)
!472 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !450, file: !458, line: 60)
!473 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !331, file: !458, line: 61)
!474 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !335, file: !458, line: 62)
!475 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !340, file: !458, line: 63)
!476 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !342, file: !458, line: 64)
!477 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !346, file: !458, line: 65)
!478 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !350, file: !458, line: 67)
!479 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !354, file: !458, line: 68)
!480 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !362, file: !458, line: 69)
!481 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !366, file: !458, line: 71)
!482 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !370, file: !458, line: 72)
!483 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !374, file: !458, line: 73)
!484 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !378, file: !458, line: 74)
!485 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !383, file: !458, line: 75)
!486 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !389, file: !458, line: 76)
!487 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !393, file: !458, line: 77)
!488 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !397, file: !458, line: 78)
!489 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !399, file: !458, line: 80)
!490 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !407, file: !458, line: 81)
!491 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !492, file: !494, line: 414)
!492 = !DISubprogram(name: "acosf", linkageName: "_ZL5acosff", scope: !493, file: !493, line: 1126, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!493 = !DIFile(filename: "/usr/local/cuda/include/math_functions.hpp", directory: "")
!494 = !DIFile(filename: "llvm-install-tulip/lib/clang/9.0.0/include/__clang_cuda_cmath.h", directory: "/scratch/ah7226")
!495 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !496, file: !494, line: 415)
!496 = !DISubprogram(name: "acoshf", linkageName: "_ZL6acoshff", scope: !493, file: !493, line: 1154, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!497 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !498, file: !494, line: 416)
!498 = !DISubprogram(name: "asinf", linkageName: "_ZL5asinff", scope: !493, file: !493, line: 1121, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!499 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !500, file: !494, line: 417)
!500 = !DISubprogram(name: "asinhf", linkageName: "_ZL6asinhff", scope: !493, file: !493, line: 1159, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!501 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !502, file: !494, line: 418)
!502 = !DISubprogram(name: "atan2f", linkageName: "_ZL6atan2fff", scope: !493, file: !493, line: 1111, type: !46, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!503 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !504, file: !494, line: 419)
!504 = !DISubprogram(name: "atanf", linkageName: "_ZL5atanff", scope: !493, file: !493, line: 1116, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!505 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !506, file: !494, line: 420)
!506 = !DISubprogram(name: "atanhf", linkageName: "_ZL6atanhff", scope: !493, file: !493, line: 1164, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!507 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !508, file: !494, line: 421)
!508 = !DISubprogram(name: "cbrtf", linkageName: "_ZL5cbrtff", scope: !493, file: !493, line: 1199, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!509 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !510, file: !494, line: 422)
!510 = !DISubprogram(name: "ceilf", linkageName: "_ZL5ceilff", scope: !511, file: !511, line: 647, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!511 = !DIFile(filename: "/usr/local/cuda/include/device_functions.hpp", directory: "")
!512 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !513, file: !494, line: 423)
!513 = !DISubprogram(name: "copysignf", linkageName: "_ZL9copysignfff", scope: !493, file: !493, line: 973, type: !46, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!514 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !515, file: !494, line: 424)
!515 = !DISubprogram(name: "cosf", linkageName: "_ZL4cosff", scope: !493, file: !493, line: 1027, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!516 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !517, file: !494, line: 425)
!517 = !DISubprogram(name: "coshf", linkageName: "_ZL5coshff", scope: !493, file: !493, line: 1096, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!518 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !519, file: !494, line: 426)
!519 = !DISubprogram(name: "erfcf", linkageName: "_ZL5erfcff", scope: !493, file: !493, line: 1259, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!520 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !521, file: !494, line: 427)
!521 = !DISubprogram(name: "erff", linkageName: "_ZL4erfff", scope: !493, file: !493, line: 1249, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!522 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !523, file: !494, line: 428)
!523 = !DISubprogram(name: "exp2f", linkageName: "_ZL5exp2ff", scope: !511, file: !511, line: 637, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!524 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !525, file: !494, line: 429)
!525 = !DISubprogram(name: "expf", linkageName: "_ZL4expff", scope: !493, file: !493, line: 1078, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!526 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !527, file: !494, line: 430)
!527 = !DISubprogram(name: "expm1f", linkageName: "_ZL6expm1ff", scope: !493, file: !493, line: 1169, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!528 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !529, file: !494, line: 431)
!529 = !DISubprogram(name: "fabsf", linkageName: "_ZL5fabsff", scope: !511, file: !511, line: 582, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!530 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !531, file: !494, line: 432)
!531 = !DISubprogram(name: "fdimf", linkageName: "_ZL5fdimfff", scope: !493, file: !493, line: 1385, type: !46, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!532 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !533, file: !494, line: 433)
!533 = !DISubprogram(name: "floorf", linkageName: "_ZL6floorff", scope: !511, file: !511, line: 572, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!534 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !535, file: !494, line: 434)
!535 = !DISubprogram(name: "fmaf", linkageName: "_ZL4fmaffff", scope: !493, file: !493, line: 1337, type: !78, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!536 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !537, file: !494, line: 435)
!537 = !DISubprogram(name: "fmaxf", linkageName: "_ZL5fmaxfff", scope: !511, file: !511, line: 602, type: !46, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!538 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !539, file: !494, line: 436)
!539 = !DISubprogram(name: "fminf", linkageName: "_ZL5fminfff", scope: !511, file: !511, line: 597, type: !46, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!540 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !541, file: !494, line: 437)
!541 = !DISubprogram(name: "fmodf", linkageName: "_ZL5fmodfff", scope: !493, file: !493, line: 1322, type: !46, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!542 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !543, file: !494, line: 438)
!543 = !DISubprogram(name: "frexpf", linkageName: "_ZL6frexpffPi", scope: !493, file: !493, line: 1312, type: !92, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!544 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !545, file: !494, line: 439)
!545 = !DISubprogram(name: "hypotf", linkageName: "_ZL6hypotfff", scope: !493, file: !493, line: 1174, type: !46, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!546 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !547, file: !494, line: 440)
!547 = !DISubprogram(name: "ilogbf", linkageName: "_ZL6ilogbff", scope: !493, file: !493, line: 1390, type: !88, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!548 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !549, file: !494, line: 441)
!549 = !DISubprogram(name: "ldexpf", linkageName: "_ZL6ldexpffi", scope: !493, file: !493, line: 1289, type: !131, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!550 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !551, file: !494, line: 442)
!551 = !DISubprogram(name: "lgammaf", linkageName: "_ZL7lgammaff", scope: !493, file: !493, line: 1284, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!552 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !553, file: !494, line: 443)
!553 = !DISubprogram(name: "llrintf", linkageName: "_ZL7llrintff", scope: !493, file: !493, line: 933, type: !141, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!554 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !555, file: !494, line: 444)
!555 = !DISubprogram(name: "llroundf", linkageName: "_ZL8llroundff", scope: !493, file: !493, line: 1371, type: !141, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!556 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !557, file: !494, line: 445)
!557 = !DISubprogram(name: "log10f", linkageName: "_ZL6log10ff", scope: !493, file: !493, line: 1140, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!558 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !559, file: !494, line: 446)
!559 = !DISubprogram(name: "log1pf", linkageName: "_ZL6log1pff", scope: !493, file: !493, line: 1149, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!560 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !561, file: !494, line: 447)
!561 = !DISubprogram(name: "log2f", linkageName: "_ZL5log2ff", scope: !493, file: !493, line: 1069, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!562 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !563, file: !494, line: 448)
!563 = !DISubprogram(name: "logbf", linkageName: "_ZL5logbff", scope: !493, file: !493, line: 1395, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!564 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !565, file: !494, line: 449)
!565 = !DISubprogram(name: "logf", linkageName: "_ZL4logff", scope: !493, file: !493, line: 1131, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!566 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !567, file: !494, line: 450)
!567 = !DISubprogram(name: "lrintf", linkageName: "_ZL6lrintff", scope: !493, file: !493, line: 924, type: !155, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!568 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !569, file: !494, line: 451)
!569 = !DISubprogram(name: "lroundf", linkageName: "_ZL7lroundff", scope: !493, file: !493, line: 1376, type: !155, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!570 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !571, file: !494, line: 452)
!571 = !DISubprogram(name: "modff", linkageName: "_ZL5modfffPf", scope: !493, file: !493, line: 1317, type: !163, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!572 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !573, file: !494, line: 453)
!573 = !DISubprogram(name: "nearbyintf", linkageName: "_ZL10nearbyintff", scope: !493, file: !493, line: 938, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!574 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !575, file: !494, line: 454)
!575 = !DISubprogram(name: "nextafterf", linkageName: "_ZL10nextafterfff", scope: !493, file: !493, line: 1002, type: !46, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!576 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !577, file: !494, line: 455)
!577 = !DISubprogram(name: "powf", linkageName: "_ZL4powfff", scope: !493, file: !493, line: 1352, type: !46, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!578 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !579, file: !494, line: 456)
!579 = !DISubprogram(name: "remainderf", linkageName: "_ZL10remainderfff", scope: !493, file: !493, line: 1327, type: !46, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!580 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !581, file: !494, line: 457)
!581 = !DISubprogram(name: "remquof", linkageName: "_ZL7remquofffPi", scope: !493, file: !493, line: 1332, type: !187, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!582 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !583, file: !494, line: 458)
!583 = !DISubprogram(name: "rintf", linkageName: "_ZL5rintff", scope: !493, file: !493, line: 919, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!584 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !585, file: !494, line: 459)
!585 = !DISubprogram(name: "roundf", linkageName: "_ZL6roundff", scope: !493, file: !493, line: 1366, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!586 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !587, file: !494, line: 462)
!587 = !DISubprogram(name: "scalblnf", linkageName: "_ZL8scalblnffl", scope: !493, file: !493, line: 1299, type: !195, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!588 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !589, file: !494, line: 464)
!589 = !DISubprogram(name: "scalbnf", linkageName: "_ZL7scalbnffi", scope: !493, file: !493, line: 1294, type: !131, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!590 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !591, file: !494, line: 465)
!591 = !DISubprogram(name: "sinf", linkageName: "_ZL4sinff", scope: !493, file: !493, line: 1018, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!592 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !593, file: !494, line: 466)
!593 = !DISubprogram(name: "sinhf", linkageName: "_ZL5sinhff", scope: !493, file: !493, line: 1101, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!594 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !595, file: !494, line: 467)
!595 = !DISubprogram(name: "sqrtf", linkageName: "_ZL5sqrtff", scope: !511, file: !511, line: 887, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!596 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !597, file: !494, line: 468)
!597 = !DISubprogram(name: "tanf", linkageName: "_ZL4tanff", scope: !493, file: !493, line: 1060, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!598 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !599, file: !494, line: 469)
!599 = !DISubprogram(name: "tanhf", linkageName: "_ZL5tanhff", scope: !493, file: !493, line: 1106, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!600 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !601, file: !494, line: 470)
!601 = !DISubprogram(name: "tgammaf", linkageName: "_ZL7tgammaff", scope: !493, file: !493, line: 1361, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!602 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !603, file: !494, line: 471)
!603 = !DISubprogram(name: "truncf", linkageName: "_ZL6truncff", scope: !511, file: !511, line: 642, type: !33, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!604 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !605, file: !609, line: 98)
!605 = !DIDerivedType(tag: DW_TAG_typedef, name: "FILE", file: !606, line: 7, baseType: !607)
!606 = !DIFile(filename: "/usr/include/x86_64-linux-gnu/bits/types/FILE.h", directory: "")
!607 = !DICompositeType(tag: DW_TAG_structure_type, name: "_IO_FILE", file: !608, line: 49, flags: DIFlagFwdDecl, identifier: "_ZTS8_IO_FILE")
!608 = !DIFile(filename: "/usr/include/x86_64-linux-gnu/bits/types/struct_FILE.h", directory: "")
!609 = !DIFile(filename: "/usr/lib/gcc/x86_64-linux-gnu/11/../../../../include/c++/11/cstdio", directory: "")
!610 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !611, file: !609, line: 99)
!611 = !DIDerivedType(tag: DW_TAG_typedef, name: "fpos_t", file: !612, line: 84, baseType: !613)
!612 = !DIFile(filename: "/usr/include/stdio.h", directory: "")
!613 = !DIDerivedType(tag: DW_TAG_typedef, name: "__fpos_t", file: !614, line: 14, baseType: !615)
!614 = !DIFile(filename: "/usr/include/x86_64-linux-gnu/bits/types/__fpos_t.h", directory: "")
!615 = !DICompositeType(tag: DW_TAG_structure_type, name: "_G_fpos_t", file: !614, line: 10, flags: DIFlagFwdDecl, identifier: "_ZTS9_G_fpos_t")
!616 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !617, file: !609, line: 101)
!617 = !DISubprogram(name: "clearerr", scope: !612, file: !612, line: 786, type: !618, flags: DIFlagPrototyped, spFlags: 0)
!618 = !DISubroutineType(types: !619)
!619 = !{null, !620}
!620 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !605, size: 64)
!621 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !622, file: !609, line: 102)
!622 = !DISubprogram(name: "fclose", scope: !612, file: !612, line: 178, type: !623, flags: DIFlagPrototyped, spFlags: 0)
!623 = !DISubroutineType(types: !624)
!624 = !{!18, !620}
!625 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !626, file: !609, line: 103)
!626 = !DISubprogram(name: "feof", scope: !612, file: !612, line: 788, type: !623, flags: DIFlagPrototyped, spFlags: 0)
!627 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !628, file: !609, line: 104)
!628 = !DISubprogram(name: "ferror", scope: !612, file: !612, line: 790, type: !623, flags: DIFlagPrototyped, spFlags: 0)
!629 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !630, file: !609, line: 105)
!630 = !DISubprogram(name: "fflush", scope: !612, file: !612, line: 230, type: !623, flags: DIFlagPrototyped, spFlags: 0)
!631 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !632, file: !609, line: 106)
!632 = !DISubprogram(name: "fgetc", scope: !612, file: !612, line: 513, type: !623, flags: DIFlagPrototyped, spFlags: 0)
!633 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !634, file: !609, line: 107)
!634 = !DISubprogram(name: "fgetpos", scope: !612, file: !612, line: 760, type: !635, flags: DIFlagPrototyped, spFlags: 0)
!635 = !DISubroutineType(types: !636)
!636 = !{!18, !637, !638}
!637 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !620)
!638 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !639)
!639 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !611, size: 64)
!640 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !641, file: !609, line: 108)
!641 = !DISubprogram(name: "fgets", scope: !612, file: !612, line: 592, type: !642, flags: DIFlagPrototyped, spFlags: 0)
!642 = !DISubroutineType(types: !643)
!643 = !{!338, !402, !18, !637}
!644 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !645, file: !609, line: 109)
!645 = !DISubprogram(name: "fopen", scope: !612, file: !612, line: 258, type: !646, flags: DIFlagPrototyped, spFlags: 0)
!646 = !DISubroutineType(types: !647)
!647 = !{!620, !360, !360}
!648 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !649, file: !609, line: 110)
!649 = !DISubprogram(name: "fprintf", scope: !612, file: !612, line: 350, type: !650, flags: DIFlagPrototyped, spFlags: 0)
!650 = !DISubroutineType(types: !651)
!651 = !{!18, !637, !360, null}
!652 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !653, file: !609, line: 111)
!653 = !DISubprogram(name: "fputc", scope: !612, file: !612, line: 549, type: !654, flags: DIFlagPrototyped, spFlags: 0)
!654 = !DISubroutineType(types: !655)
!655 = !{!18, !18, !620}
!656 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !657, file: !609, line: 112)
!657 = !DISubprogram(name: "fputs", scope: !612, file: !612, line: 655, type: !658, flags: DIFlagPrototyped, spFlags: 0)
!658 = !DISubroutineType(types: !659)
!659 = !{!18, !360, !637}
!660 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !661, file: !609, line: 113)
!661 = !DISubprogram(name: "fread", scope: !612, file: !612, line: 675, type: !662, flags: DIFlagPrototyped, spFlags: 0)
!662 = !DISubroutineType(types: !663)
!663 = !{!311, !664, !311, !311, !637}
!664 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !308)
!665 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !666, file: !609, line: 114)
!666 = !DISubprogram(name: "freopen", scope: !612, file: !612, line: 265, type: !667, flags: DIFlagPrototyped, spFlags: 0)
!667 = !DISubroutineType(types: !668)
!668 = !{!620, !360, !360, !637}
!669 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !670, file: !609, line: 115)
!670 = !DISubprogram(name: "fscanf", scope: !612, file: !612, line: 415, type: !650, flags: DIFlagPrototyped, spFlags: 0)
!671 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !672, file: !609, line: 116)
!672 = !DISubprogram(name: "fseek", scope: !612, file: !612, line: 713, type: !673, flags: DIFlagPrototyped, spFlags: 0)
!673 = !DISubroutineType(types: !674)
!674 = !{!18, !620, !128, !18}
!675 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !676, file: !609, line: 117)
!676 = !DISubprogram(name: "fsetpos", scope: !612, file: !612, line: 765, type: !677, flags: DIFlagPrototyped, spFlags: 0)
!677 = !DISubroutineType(types: !678)
!678 = !{!18, !620, !679}
!679 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !680, size: 64)
!680 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !611)
!681 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !682, file: !609, line: 118)
!682 = !DISubprogram(name: "ftell", scope: !612, file: !612, line: 718, type: !683, flags: DIFlagPrototyped, spFlags: 0)
!683 = !DISubroutineType(types: !684)
!684 = !{!128, !620}
!685 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !686, file: !609, line: 119)
!686 = !DISubprogram(name: "fwrite", scope: !612, file: !612, line: 681, type: !687, flags: DIFlagPrototyped, spFlags: 0)
!687 = !DISubroutineType(types: !688)
!688 = !{!311, !689, !311, !311, !637}
!689 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !309)
!690 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !691, file: !609, line: 120)
!691 = !DISubprogram(name: "getc", scope: !612, file: !612, line: 514, type: !623, flags: DIFlagPrototyped, spFlags: 0)
!692 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !693, file: !609, line: 121)
!693 = !DISubprogram(name: "getchar", scope: !612, file: !612, line: 520, type: !371, flags: DIFlagPrototyped, spFlags: 0)
!694 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !695, file: !609, line: 124)
!695 = !DISubprogram(name: "gets", scope: !612, file: !612, line: 605, type: !696, flags: DIFlagPrototyped, spFlags: 0)
!696 = !DISubroutineType(types: !697)
!697 = !{!338, !338}
!698 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !699, file: !609, line: 126)
!699 = !DISubprogram(name: "perror", scope: !612, file: !612, line: 804, type: !700, flags: DIFlagPrototyped, spFlags: 0)
!700 = !DISubroutineType(types: !701)
!701 = !{null, !170}
!702 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !703, file: !609, line: 127)
!703 = !DISubprogram(name: "printf", scope: !612, file: !612, line: 356, type: !704, flags: DIFlagPrototyped, spFlags: 0)
!704 = !DISubroutineType(types: !705)
!705 = !{!18, !360, null}
!706 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !707, file: !609, line: 128)
!707 = !DISubprogram(name: "putc", scope: !612, file: !612, line: 550, type: !654, flags: DIFlagPrototyped, spFlags: 0)
!708 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !709, file: !609, line: 129)
!709 = !DISubprogram(name: "putchar", scope: !612, file: !612, line: 556, type: !29, flags: DIFlagPrototyped, spFlags: 0)
!710 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !711, file: !609, line: 130)
!711 = !DISubprogram(name: "puts", scope: !612, file: !612, line: 661, type: !298, flags: DIFlagPrototyped, spFlags: 0)
!712 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !713, file: !609, line: 131)
!713 = !DISubprogram(name: "remove", scope: !612, file: !612, line: 152, type: !298, flags: DIFlagPrototyped, spFlags: 0)
!714 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !715, file: !609, line: 132)
!715 = !DISubprogram(name: "rename", scope: !612, file: !612, line: 154, type: !716, flags: DIFlagPrototyped, spFlags: 0)
!716 = !DISubroutineType(types: !717)
!717 = !{!18, !170, !170}
!718 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !719, file: !609, line: 133)
!719 = !DISubprogram(name: "rewind", scope: !612, file: !612, line: 723, type: !618, flags: DIFlagPrototyped, spFlags: 0)
!720 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !721, file: !609, line: 134)
!721 = !DISubprogram(name: "scanf", scope: !612, file: !612, line: 421, type: !704, flags: DIFlagPrototyped, spFlags: 0)
!722 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !723, file: !609, line: 135)
!723 = !DISubprogram(name: "setbuf", scope: !612, file: !612, line: 328, type: !724, flags: DIFlagPrototyped, spFlags: 0)
!724 = !DISubroutineType(types: !725)
!725 = !{null, !637, !402}
!726 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !727, file: !609, line: 136)
!727 = !DISubprogram(name: "setvbuf", scope: !612, file: !612, line: 332, type: !728, flags: DIFlagPrototyped, spFlags: 0)
!728 = !DISubroutineType(types: !729)
!729 = !{!18, !637, !402, !18, !311}
!730 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !731, file: !609, line: 137)
!731 = !DISubprogram(name: "sprintf", scope: !612, file: !612, line: 358, type: !732, flags: DIFlagPrototyped, spFlags: 0)
!732 = !DISubroutineType(types: !733)
!733 = !{!18, !402, !360, null}
!734 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !735, file: !609, line: 138)
!735 = !DISubprogram(name: "sscanf", scope: !612, file: !612, line: 423, type: !736, flags: DIFlagPrototyped, spFlags: 0)
!736 = !DISubroutineType(types: !737)
!737 = !{!18, !360, !360, null}
!738 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !739, file: !609, line: 139)
!739 = !DISubprogram(name: "tmpfile", scope: !612, file: !612, line: 188, type: !740, flags: DIFlagPrototyped, spFlags: 0)
!740 = !DISubroutineType(types: !741)
!741 = !{!620}
!742 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !743, file: !609, line: 141)
!743 = !DISubprogram(name: "tmpnam", scope: !612, file: !612, line: 205, type: !696, flags: DIFlagPrototyped, spFlags: 0)
!744 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !745, file: !609, line: 143)
!745 = !DISubprogram(name: "ungetc", scope: !612, file: !612, line: 668, type: !654, flags: DIFlagPrototyped, spFlags: 0)
!746 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !747, file: !609, line: 144)
!747 = !DISubprogram(name: "vfprintf", scope: !612, file: !612, line: 365, type: !748, flags: DIFlagPrototyped, spFlags: 0)
!748 = !DISubroutineType(types: !749)
!749 = !{!18, !637, !360, !750}
!750 = !DIDerivedType(tag: DW_TAG_typedef, name: "__gnuc_va_list", file: !751, line: 32, baseType: !752)
!751 = !DIFile(filename: "llvm-install-tulip/lib/clang/9.0.0/include/stdarg.h", directory: "/scratch/ah7226")
!752 = !DIDerivedType(tag: DW_TAG_typedef, name: "__builtin_va_list", file: !6, baseType: !338)
!753 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !754, file: !609, line: 145)
!754 = !DISubprogram(name: "vprintf", scope: !612, file: !612, line: 371, type: !755, flags: DIFlagPrototyped, spFlags: 0)
!755 = !DISubroutineType(types: !756)
!756 = !{!18, !360, !750}
!757 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !758, file: !609, line: 146)
!758 = !DISubprogram(name: "vsprintf", scope: !612, file: !612, line: 373, type: !759, flags: DIFlagPrototyped, spFlags: 0)
!759 = !DISubroutineType(types: !760)
!760 = !{!18, !402, !360, !750}
!761 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !411, entity: !762, file: !609, line: 175)
!762 = !DISubprogram(name: "snprintf", scope: !612, file: !612, line: 378, type: !763, flags: DIFlagPrototyped, spFlags: 0)
!763 = !DISubroutineType(types: !764)
!764 = !{!18, !402, !311, !360, null}
!765 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !411, entity: !766, file: !609, line: 176)
!766 = !DISubprogram(name: "vfscanf", scope: !612, file: !612, line: 459, type: !748, flags: DIFlagPrototyped, spFlags: 0)
!767 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !411, entity: !768, file: !609, line: 177)
!768 = !DISubprogram(name: "vscanf", scope: !612, file: !612, line: 467, type: !755, flags: DIFlagPrototyped, spFlags: 0)
!769 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !411, entity: !770, file: !609, line: 178)
!770 = !DISubprogram(name: "vsnprintf", scope: !612, file: !612, line: 382, type: !771, flags: DIFlagPrototyped, spFlags: 0)
!771 = !DISubroutineType(types: !772)
!772 = !{!18, !402, !311, !360, !750}
!773 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !411, entity: !774, file: !609, line: 179)
!774 = !DISubprogram(name: "vsscanf", scope: !612, file: !612, line: 471, type: !775, flags: DIFlagPrototyped, spFlags: 0)
!775 = !DISubroutineType(types: !776)
!776 = !{!18, !360, !360, !750}
!777 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !762, file: !609, line: 185)
!778 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !766, file: !609, line: 186)
!779 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !768, file: !609, line: 187)
!780 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !770, file: !609, line: 188)
!781 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !26, entity: !774, file: !609, line: 189)
!782 = !{void (%struct.dcomplex*, %struct.dcomplex*)* @_Z19cffts1_gpu_kernel_1P8dcomplexS0_, !"kernel", i32 1}
!783 = !{void (i32, %struct.dcomplex*, %struct.dcomplex*, %struct.dcomplex*)* @_Z19cffts1_gpu_kernel_2iP8dcomplexS0_S0_, !"kernel", i32 1}
!784 = !{void (%struct.dcomplex*, %struct.dcomplex*)* @_Z19cffts1_gpu_kernel_3P8dcomplexS0_, !"kernel", i32 1}
!785 = !{void (%struct.dcomplex*, %struct.dcomplex*)* @_Z19cffts2_gpu_kernel_1P8dcomplexS0_, !"kernel", i32 1}
!786 = !{void (i32, %struct.dcomplex*, %struct.dcomplex*, %struct.dcomplex*)* @_Z19cffts2_gpu_kernel_2iP8dcomplexS0_S0_, !"kernel", i32 1}
!787 = !{void (%struct.dcomplex*, %struct.dcomplex*)* @_Z19cffts2_gpu_kernel_3P8dcomplexS0_, !"kernel", i32 1}
!788 = !{void (%struct.dcomplex*, %struct.dcomplex*)* @_Z19cffts3_gpu_kernel_1P8dcomplexS0_, !"kernel", i32 1}
!789 = !{void (i32, %struct.dcomplex*, %struct.dcomplex*, %struct.dcomplex*)* @_Z19cffts3_gpu_kernel_2iP8dcomplexS0_S0_, !"kernel", i32 1}
!790 = !{void (%struct.dcomplex*, %struct.dcomplex*)* @_Z19cffts3_gpu_kernel_3P8dcomplexS0_, !"kernel", i32 1}
!791 = !{void (i32, %struct.dcomplex*, %struct.dcomplex*)* @_Z19checksum_gpu_kerneliP8dcomplexS0_, !"kernel", i32 1}
!792 = !{void (double*)* @_Z27compute_indexmap_gpu_kernelPd, !"kernel", i32 1}
!793 = !{void (%struct.dcomplex*, double*)* @_Z37compute_initial_conditions_gpu_kernelP8dcomplexPd, !"kernel", i32 1}
!794 = !{void (%struct.dcomplex*, %struct.dcomplex*, double*)* @_Z17evolve_gpu_kernelP8dcomplexS0_Pd, !"kernel", i32 1}
!795 = !{void (%struct.dcomplex*, %struct.dcomplex*, double*)* @_Z18init_ui_gpu_kernelP8dcomplexS0_Pd, !"kernel", i32 1}
!796 = !{null, !"align", i32 8}
!797 = !{null, !"align", i32 8, !"align", i32 65544, !"align", i32 131080}
!798 = !{null, !"align", i32 16}
!799 = !{null, !"align", i32 16, !"align", i32 65552, !"align", i32 131088}
!800 = !{!"clang version 9.0.0 (https://github.com/yebinchon/llvm-project/ a1efa594106d738d0b74c9e4e2b9b779eb8b7d25)"}
!801 = !{i32 1, i32 2}
!802 = distinct !DISubprogram(name: "cffts1_gpu_kernel_1", linkageName: "_Z19cffts1_gpu_kernel_1P8dcomplexS0_", scope: !6, file: !6, line: 762, type: !803, scopeLine: 763, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !7)
!803 = !DISubroutineType(types: !804)
!804 = !{null, !9, !9}
!805 = !DILocalVariable(name: "x_in", arg: 1, scope: !802, file: !6, line: 762, type: !9)
!806 = !DILocation(line: 762, column: 46, scope: !802)
!807 = !DILocalVariable(name: "y0", arg: 2, scope: !802, file: !6, line: 763, type: !9)
!808 = !DILocation(line: 763, column: 12, scope: !802)
!809 = !DILocalVariable(name: "x_y_z", scope: !802, file: !6, line: 764, type: !18)
!810 = !DILocation(line: 764, column: 6, scope: !802)
!811 = !DILocation(line: 64, column: 3, scope: !812, inlinedAt: !847)
!812 = distinct !DISubprogram(name: "__fetch_builtin_x", linkageName: "_ZN25__cuda_builtin_blockIdx_t17__fetch_builtin_xEv", scope: !814, file: !813, line: 64, type: !817, scopeLine: 64, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !5, declaration: !816, retainedNodes: !7)
!813 = !DIFile(filename: "llvm-install-tulip/lib/clang/9.0.0/include/__clang_cuda_builtin_vars.h", directory: "/scratch/ah7226")
!814 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "__cuda_builtin_blockIdx_t", file: !813, line: 63, size: 8, flags: DIFlagTypePassByReference | DIFlagNonTrivial, elements: !815, identifier: "_ZTS25__cuda_builtin_blockIdx_t")
!815 = !{!816, !819, !820, !821, !832, !836, !840, !843}
!816 = !DISubprogram(name: "__fetch_builtin_x", linkageName: "_ZN25__cuda_builtin_blockIdx_t17__fetch_builtin_xEv", scope: !814, file: !813, line: 64, type: !817, scopeLine: 64, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!817 = !DISubroutineType(types: !818)
!818 = !{!381}
!819 = !DISubprogram(name: "__fetch_builtin_y", linkageName: "_ZN25__cuda_builtin_blockIdx_t17__fetch_builtin_yEv", scope: !814, file: !813, line: 65, type: !817, scopeLine: 65, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!820 = !DISubprogram(name: "__fetch_builtin_z", linkageName: "_ZN25__cuda_builtin_blockIdx_t17__fetch_builtin_zEv", scope: !814, file: !813, line: 66, type: !817, scopeLine: 66, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!821 = !DISubprogram(name: "operator uint3", linkageName: "_ZNK25__cuda_builtin_blockIdx_tcv5uint3Ev", scope: !814, file: !813, line: 69, type: !822, scopeLine: 69, flags: DIFlagPrototyped, spFlags: 0)
!822 = !DISubroutineType(types: !823)
!823 = !{!824, !830}
!824 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "uint3", file: !825, line: 190, size: 96, flags: DIFlagTypePassByValue, elements: !826, identifier: "_ZTS5uint3")
!825 = !DIFile(filename: "/usr/local/cuda/include/vector_types.h", directory: "")
!826 = !{!827, !828, !829}
!827 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !824, file: !825, line: 192, baseType: !381, size: 32)
!828 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !824, file: !825, line: 192, baseType: !381, size: 32, offset: 32)
!829 = !DIDerivedType(tag: DW_TAG_member, name: "z", scope: !824, file: !825, line: 192, baseType: !381, size: 32, offset: 64)
!830 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !831, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!831 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !814)
!832 = !DISubprogram(name: "__cuda_builtin_blockIdx_t", scope: !814, file: !813, line: 71, type: !833, scopeLine: 71, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!833 = !DISubroutineType(types: !834)
!834 = !{null, !835}
!835 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !814, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!836 = !DISubprogram(name: "__cuda_builtin_blockIdx_t", scope: !814, file: !813, line: 71, type: !837, scopeLine: 71, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!837 = !DISubroutineType(types: !838)
!838 = !{null, !835, !839}
!839 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !831, size: 64)
!840 = !DISubprogram(name: "operator=", linkageName: "_ZNK25__cuda_builtin_blockIdx_taSERKS_", scope: !814, file: !813, line: 71, type: !841, scopeLine: 71, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!841 = !DISubroutineType(types: !842)
!842 = !{null, !830, !839}
!843 = !DISubprogram(name: "operator&", linkageName: "_ZNK25__cuda_builtin_blockIdx_tadEv", scope: !814, file: !813, line: 71, type: !844, scopeLine: 71, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!844 = !DISubroutineType(types: !845)
!845 = !{!846, !830}
!846 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !814, size: 64)
!847 = distinct !DILocation(line: 764, column: 14, scope: !802)
!848 = !{i32 0, i32 65535}
!849 = !DILocation(line: 75, column: 3, scope: !850, inlinedAt: !892)
!850 = distinct !DISubprogram(name: "__fetch_builtin_x", linkageName: "_ZN25__cuda_builtin_blockDim_t17__fetch_builtin_xEv", scope: !851, file: !813, line: 75, type: !817, scopeLine: 75, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !5, declaration: !853, retainedNodes: !7)
!851 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "__cuda_builtin_blockDim_t", file: !813, line: 74, size: 8, flags: DIFlagTypePassByReference | DIFlagNonTrivial, elements: !852, identifier: "_ZTS25__cuda_builtin_blockDim_t")
!852 = !{!853, !854, !855, !856, !877, !881, !885, !888}
!853 = !DISubprogram(name: "__fetch_builtin_x", linkageName: "_ZN25__cuda_builtin_blockDim_t17__fetch_builtin_xEv", scope: !851, file: !813, line: 75, type: !817, scopeLine: 75, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!854 = !DISubprogram(name: "__fetch_builtin_y", linkageName: "_ZN25__cuda_builtin_blockDim_t17__fetch_builtin_yEv", scope: !851, file: !813, line: 76, type: !817, scopeLine: 76, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!855 = !DISubprogram(name: "__fetch_builtin_z", linkageName: "_ZN25__cuda_builtin_blockDim_t17__fetch_builtin_zEv", scope: !851, file: !813, line: 77, type: !817, scopeLine: 77, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!856 = !DISubprogram(name: "operator dim3", linkageName: "_ZNK25__cuda_builtin_blockDim_tcv4dim3Ev", scope: !851, file: !813, line: 80, type: !857, scopeLine: 80, flags: DIFlagPrototyped, spFlags: 0)
!857 = !DISubroutineType(types: !858)
!858 = !{!859, !875}
!859 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "dim3", file: !825, line: 417, size: 96, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !860, identifier: "_ZTS4dim3")
!860 = !{!861, !862, !863, !864, !868, !872}
!861 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !859, file: !825, line: 419, baseType: !381, size: 32)
!862 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !859, file: !825, line: 419, baseType: !381, size: 32, offset: 32)
!863 = !DIDerivedType(tag: DW_TAG_member, name: "z", scope: !859, file: !825, line: 419, baseType: !381, size: 32, offset: 64)
!864 = !DISubprogram(name: "dim3", scope: !859, file: !825, line: 421, type: !865, scopeLine: 421, flags: DIFlagPrototyped, spFlags: 0)
!865 = !DISubroutineType(types: !866)
!866 = !{null, !867, !381, !381, !381}
!867 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !859, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!868 = !DISubprogram(name: "dim3", scope: !859, file: !825, line: 422, type: !869, scopeLine: 422, flags: DIFlagPrototyped, spFlags: 0)
!869 = !DISubroutineType(types: !870)
!870 = !{null, !867, !871}
!871 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint3", file: !825, line: 383, baseType: !824)
!872 = !DISubprogram(name: "operator uint3", linkageName: "_ZN4dim3cv5uint3Ev", scope: !859, file: !825, line: 423, type: !873, scopeLine: 423, flags: DIFlagPrototyped, spFlags: 0)
!873 = !DISubroutineType(types: !874)
!874 = !{!871, !867}
!875 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !876, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!876 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !851)
!877 = !DISubprogram(name: "__cuda_builtin_blockDim_t", scope: !851, file: !813, line: 82, type: !878, scopeLine: 82, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!878 = !DISubroutineType(types: !879)
!879 = !{null, !880}
!880 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !851, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!881 = !DISubprogram(name: "__cuda_builtin_blockDim_t", scope: !851, file: !813, line: 82, type: !882, scopeLine: 82, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!882 = !DISubroutineType(types: !883)
!883 = !{null, !880, !884}
!884 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !876, size: 64)
!885 = !DISubprogram(name: "operator=", linkageName: "_ZNK25__cuda_builtin_blockDim_taSERKS_", scope: !851, file: !813, line: 82, type: !886, scopeLine: 82, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!886 = !DISubroutineType(types: !887)
!887 = !{null, !875, !884}
!888 = !DISubprogram(name: "operator&", linkageName: "_ZNK25__cuda_builtin_blockDim_tadEv", scope: !851, file: !813, line: 82, type: !889, scopeLine: 82, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!889 = !DISubroutineType(types: !890)
!890 = !{!891, !875}
!891 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !851, size: 64)
!892 = distinct !DILocation(line: 764, column: 27, scope: !802)
!893 = !{i32 1, i32 1025}
!894 = !DILocation(line: 764, column: 25, scope: !802)
!895 = !DILocation(line: 53, column: 3, scope: !896, inlinedAt: !922)
!896 = distinct !DISubprogram(name: "__fetch_builtin_x", linkageName: "_ZN26__cuda_builtin_threadIdx_t17__fetch_builtin_xEv", scope: !897, file: !813, line: 53, type: !817, scopeLine: 53, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !5, declaration: !899, retainedNodes: !7)
!897 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "__cuda_builtin_threadIdx_t", file: !813, line: 52, size: 8, flags: DIFlagTypePassByReference | DIFlagNonTrivial, elements: !898, identifier: "_ZTS26__cuda_builtin_threadIdx_t")
!898 = !{!899, !900, !901, !902, !907, !911, !915, !918}
!899 = !DISubprogram(name: "__fetch_builtin_x", linkageName: "_ZN26__cuda_builtin_threadIdx_t17__fetch_builtin_xEv", scope: !897, file: !813, line: 53, type: !817, scopeLine: 53, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!900 = !DISubprogram(name: "__fetch_builtin_y", linkageName: "_ZN26__cuda_builtin_threadIdx_t17__fetch_builtin_yEv", scope: !897, file: !813, line: 54, type: !817, scopeLine: 54, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!901 = !DISubprogram(name: "__fetch_builtin_z", linkageName: "_ZN26__cuda_builtin_threadIdx_t17__fetch_builtin_zEv", scope: !897, file: !813, line: 55, type: !817, scopeLine: 55, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!902 = !DISubprogram(name: "operator uint3", linkageName: "_ZNK26__cuda_builtin_threadIdx_tcv5uint3Ev", scope: !897, file: !813, line: 58, type: !903, scopeLine: 58, flags: DIFlagPrototyped, spFlags: 0)
!903 = !DISubroutineType(types: !904)
!904 = !{!824, !905}
!905 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !906, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!906 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !897)
!907 = !DISubprogram(name: "__cuda_builtin_threadIdx_t", scope: !897, file: !813, line: 60, type: !908, scopeLine: 60, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!908 = !DISubroutineType(types: !909)
!909 = !{null, !910}
!910 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !897, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!911 = !DISubprogram(name: "__cuda_builtin_threadIdx_t", scope: !897, file: !813, line: 60, type: !912, scopeLine: 60, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!912 = !DISubroutineType(types: !913)
!913 = !{null, !910, !914}
!914 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !906, size: 64)
!915 = !DISubprogram(name: "operator=", linkageName: "_ZNK26__cuda_builtin_threadIdx_taSERKS_", scope: !897, file: !813, line: 60, type: !916, scopeLine: 60, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!916 = !DISubroutineType(types: !917)
!917 = !{null, !905, !914}
!918 = !DISubprogram(name: "operator&", linkageName: "_ZNK26__cuda_builtin_threadIdx_tadEv", scope: !897, file: !813, line: 60, type: !919, scopeLine: 60, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!919 = !DISubroutineType(types: !920)
!920 = !{!921, !905}
!921 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !897, size: 64)
!922 = distinct !DILocation(line: 764, column: 40, scope: !802)
!923 = !{i32 0, i32 1024}
!924 = !DILocation(line: 764, column: 38, scope: !802)
!925 = !DILocation(line: 765, column: 5, scope: !926)
!926 = distinct !DILexicalBlock(scope: !802, file: !6, line: 765, column: 5)
!927 = !DILocation(line: 765, column: 11, scope: !926)
!928 = !DILocation(line: 765, column: 5, scope: !802)
!929 = !DILocation(line: 766, column: 3, scope: !930)
!930 = distinct !DILexicalBlock(scope: !926, file: !6, line: 765, column: 25)
!931 = !DILocalVariable(name: "x", scope: !802, file: !6, line: 768, type: !18)
!932 = !DILocation(line: 768, column: 6, scope: !802)
!933 = !DILocation(line: 768, column: 10, scope: !802)
!934 = !DILocation(line: 768, column: 16, scope: !802)
!935 = !DILocalVariable(name: "y", scope: !802, file: !6, line: 769, type: !18)
!936 = !DILocation(line: 769, column: 6, scope: !802)
!937 = !DILocation(line: 769, column: 11, scope: !802)
!938 = !DILocation(line: 769, column: 17, scope: !802)
!939 = !DILocation(line: 769, column: 23, scope: !802)
!940 = !DILocalVariable(name: "z", scope: !802, file: !6, line: 770, type: !18)
!941 = !DILocation(line: 770, column: 6, scope: !802)
!942 = !DILocation(line: 770, column: 10, scope: !802)
!943 = !DILocation(line: 770, column: 16, scope: !802)
!944 = !DILocation(line: 771, column: 32, scope: !802)
!945 = !DILocation(line: 771, column: 37, scope: !802)
!946 = !DILocation(line: 771, column: 44, scope: !802)
!947 = !DILocation(line: 771, column: 2, scope: !802)
!948 = !DILocation(line: 771, column: 5, scope: !802)
!949 = !DILocation(line: 771, column: 8, scope: !802)
!950 = !DILocation(line: 771, column: 9, scope: !802)
!951 = !DILocation(line: 771, column: 6, scope: !802)
!952 = !DILocation(line: 771, column: 15, scope: !802)
!953 = !DILocation(line: 771, column: 16, scope: !802)
!954 = !DILocation(line: 771, column: 19, scope: !802)
!955 = !DILocation(line: 771, column: 13, scope: !802)
!956 = !DILocation(line: 771, column: 25, scope: !802)
!957 = !DILocation(line: 771, column: 30, scope: !802)
!958 = !DILocation(line: 772, column: 32, scope: !802)
!959 = !DILocation(line: 772, column: 37, scope: !802)
!960 = !DILocation(line: 772, column: 44, scope: !802)
!961 = !DILocation(line: 772, column: 2, scope: !802)
!962 = !DILocation(line: 772, column: 5, scope: !802)
!963 = !DILocation(line: 772, column: 8, scope: !802)
!964 = !DILocation(line: 772, column: 9, scope: !802)
!965 = !DILocation(line: 772, column: 6, scope: !802)
!966 = !DILocation(line: 772, column: 15, scope: !802)
!967 = !DILocation(line: 772, column: 16, scope: !802)
!968 = !DILocation(line: 772, column: 19, scope: !802)
!969 = !DILocation(line: 772, column: 13, scope: !802)
!970 = !DILocation(line: 772, column: 25, scope: !802)
!971 = !DILocation(line: 772, column: 30, scope: !802)
!972 = !DILocation(line: 773, column: 1, scope: !802)
!973 = distinct !DISubprogram(name: "cffts1_gpu_kernel_2", linkageName: "_Z19cffts1_gpu_kernel_2iP8dcomplexS0_S0_", scope: !6, file: !6, line: 780, type: !974, scopeLine: 783, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !7)
!974 = !DISubroutineType(types: !975)
!975 = !{null, !976, !9, !9, !9}
!976 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !18)
!977 = !DILocalVariable(name: "is", arg: 1, scope: !973, file: !6, line: 780, type: !976)
!978 = !DILocation(line: 780, column: 47, scope: !973)
!979 = !DILocalVariable(name: "gty1", arg: 2, scope: !973, file: !6, line: 781, type: !9)
!980 = !DILocation(line: 781, column: 12, scope: !973)
!981 = !DILocalVariable(name: "gty2", arg: 3, scope: !973, file: !6, line: 782, type: !9)
!982 = !DILocation(line: 782, column: 12, scope: !973)
!983 = !DILocalVariable(name: "u_device", arg: 4, scope: !973, file: !6, line: 783, type: !9)
!984 = !DILocation(line: 783, column: 12, scope: !973)
!985 = !DILocalVariable(name: "y_z", scope: !973, file: !6, line: 784, type: !18)
!986 = !DILocation(line: 784, column: 6, scope: !973)
!987 = !DILocation(line: 64, column: 3, scope: !812, inlinedAt: !988)
!988 = distinct !DILocation(line: 784, column: 12, scope: !973)
!989 = !DILocation(line: 75, column: 3, scope: !850, inlinedAt: !990)
!990 = distinct !DILocation(line: 784, column: 25, scope: !973)
!991 = !DILocation(line: 784, column: 23, scope: !973)
!992 = !DILocation(line: 53, column: 3, scope: !896, inlinedAt: !993)
!993 = distinct !DILocation(line: 784, column: 38, scope: !973)
!994 = !DILocation(line: 784, column: 36, scope: !973)
!995 = !DILocation(line: 786, column: 5, scope: !996)
!996 = distinct !DILexicalBlock(scope: !973, file: !6, line: 786, column: 5)
!997 = !DILocation(line: 786, column: 9, scope: !996)
!998 = !DILocation(line: 786, column: 5, scope: !973)
!999 = !DILocation(line: 787, column: 3, scope: !1000)
!1000 = distinct !DILexicalBlock(scope: !996, file: !6, line: 786, column: 20)
!1001 = !DILocalVariable(name: "j", scope: !973, file: !6, line: 790, type: !18)
!1002 = !DILocation(line: 790, column: 6, scope: !973)
!1003 = !DILocalVariable(name: "k", scope: !973, file: !6, line: 790, type: !18)
!1004 = !DILocation(line: 790, column: 9, scope: !973)
!1005 = !DILocalVariable(name: "l", scope: !973, file: !6, line: 791, type: !18)
!1006 = !DILocation(line: 791, column: 6, scope: !973)
!1007 = !DILocalVariable(name: "j1", scope: !973, file: !6, line: 791, type: !18)
!1008 = !DILocation(line: 791, column: 9, scope: !973)
!1009 = !DILocalVariable(name: "i1", scope: !973, file: !6, line: 791, type: !18)
!1010 = !DILocation(line: 791, column: 13, scope: !973)
!1011 = !DILocalVariable(name: "k1", scope: !973, file: !6, line: 791, type: !18)
!1012 = !DILocation(line: 791, column: 17, scope: !973)
!1013 = !DILocalVariable(name: "n1", scope: !973, file: !6, line: 792, type: !18)
!1014 = !DILocation(line: 792, column: 6, scope: !973)
!1015 = !DILocalVariable(name: "li", scope: !973, file: !6, line: 792, type: !18)
!1016 = !DILocation(line: 792, column: 10, scope: !973)
!1017 = !DILocalVariable(name: "lj", scope: !973, file: !6, line: 792, type: !18)
!1018 = !DILocation(line: 792, column: 14, scope: !973)
!1019 = !DILocalVariable(name: "lk", scope: !973, file: !6, line: 792, type: !18)
!1020 = !DILocation(line: 792, column: 18, scope: !973)
!1021 = !DILocalVariable(name: "ku", scope: !973, file: !6, line: 792, type: !18)
!1022 = !DILocation(line: 792, column: 22, scope: !973)
!1023 = !DILocalVariable(name: "i11", scope: !973, file: !6, line: 792, type: !18)
!1024 = !DILocation(line: 792, column: 26, scope: !973)
!1025 = !DILocalVariable(name: "i12", scope: !973, file: !6, line: 792, type: !18)
!1026 = !DILocation(line: 792, column: 31, scope: !973)
!1027 = !DILocalVariable(name: "i21", scope: !973, file: !6, line: 792, type: !18)
!1028 = !DILocation(line: 792, column: 36, scope: !973)
!1029 = !DILocalVariable(name: "i22", scope: !973, file: !6, line: 792, type: !18)
!1030 = !DILocation(line: 792, column: 41, scope: !973)
!1031 = !DILocation(line: 794, column: 6, scope: !973)
!1032 = !DILocation(line: 794, column: 10, scope: !973)
!1033 = !DILocation(line: 794, column: 4, scope: !973)
!1034 = !DILocation(line: 795, column: 7, scope: !973)
!1035 = !DILocation(line: 795, column: 11, scope: !973)
!1036 = !DILocation(line: 795, column: 17, scope: !973)
!1037 = !DILocation(line: 795, column: 4, scope: !973)
!1038 = !DILocalVariable(name: "logd1", scope: !973, file: !6, line: 797, type: !976)
!1039 = !DILocation(line: 797, column: 12, scope: !973)
!1040 = !DILocation(line: 797, column: 20, scope: !973)
!1041 = !DILocalVariable(name: "uu1_real", scope: !973, file: !6, line: 799, type: !15)
!1042 = !DILocation(line: 799, column: 9, scope: !973)
!1043 = !DILocalVariable(name: "x11_real", scope: !973, file: !6, line: 799, type: !15)
!1044 = !DILocation(line: 799, column: 19, scope: !973)
!1045 = !DILocalVariable(name: "x21_real", scope: !973, file: !6, line: 799, type: !15)
!1046 = !DILocation(line: 799, column: 29, scope: !973)
!1047 = !DILocalVariable(name: "uu1_imag", scope: !973, file: !6, line: 800, type: !15)
!1048 = !DILocation(line: 800, column: 9, scope: !973)
!1049 = !DILocalVariable(name: "x11_imag", scope: !973, file: !6, line: 800, type: !15)
!1050 = !DILocation(line: 800, column: 19, scope: !973)
!1051 = !DILocalVariable(name: "x21_imag", scope: !973, file: !6, line: 800, type: !15)
!1052 = !DILocation(line: 800, column: 29, scope: !973)
!1053 = !DILocalVariable(name: "uu2_real", scope: !973, file: !6, line: 801, type: !15)
!1054 = !DILocation(line: 801, column: 9, scope: !973)
!1055 = !DILocalVariable(name: "x12_real", scope: !973, file: !6, line: 801, type: !15)
!1056 = !DILocation(line: 801, column: 19, scope: !973)
!1057 = !DILocalVariable(name: "x22_real", scope: !973, file: !6, line: 801, type: !15)
!1058 = !DILocation(line: 801, column: 29, scope: !973)
!1059 = !DILocalVariable(name: "uu2_imag", scope: !973, file: !6, line: 802, type: !15)
!1060 = !DILocation(line: 802, column: 9, scope: !973)
!1061 = !DILocalVariable(name: "x12_imag", scope: !973, file: !6, line: 802, type: !15)
!1062 = !DILocation(line: 802, column: 19, scope: !973)
!1063 = !DILocalVariable(name: "x22_imag", scope: !973, file: !6, line: 802, type: !15)
!1064 = !DILocation(line: 802, column: 29, scope: !973)
!1065 = !DILocalVariable(name: "temp_real", scope: !973, file: !6, line: 803, type: !15)
!1066 = !DILocation(line: 803, column: 9, scope: !973)
!1067 = !DILocalVariable(name: "temp2_real", scope: !973, file: !6, line: 803, type: !15)
!1068 = !DILocation(line: 803, column: 20, scope: !973)
!1069 = !DILocalVariable(name: "temp_imag", scope: !973, file: !6, line: 804, type: !15)
!1070 = !DILocation(line: 804, column: 9, scope: !973)
!1071 = !DILocalVariable(name: "temp2_imag", scope: !973, file: !6, line: 804, type: !15)
!1072 = !DILocation(line: 804, column: 20, scope: !973)
!1073 = !DILocation(line: 806, column: 7, scope: !1074)
!1074 = distinct !DILexicalBlock(scope: !973, file: !6, line: 806, column: 2)
!1075 = !DILocation(line: 806, column: 6, scope: !1074)
!1076 = !DILocation(line: 806, column: 11, scope: !1077)
!1077 = distinct !DILexicalBlock(scope: !1074, file: !6, line: 806, column: 2)
!1078 = !DILocation(line: 806, column: 14, scope: !1077)
!1079 = !DILocation(line: 806, column: 12, scope: !1077)
!1080 = !DILocation(line: 806, column: 2, scope: !1074)
!1081 = !DILocation(line: 807, column: 6, scope: !1082)
!1082 = distinct !DILexicalBlock(scope: !1077, file: !6, line: 806, column: 26)
!1083 = !DILocation(line: 808, column: 14, scope: !1082)
!1084 = !DILocation(line: 808, column: 16, scope: !1082)
!1085 = !DILocation(line: 808, column: 10, scope: !1082)
!1086 = !DILocation(line: 808, column: 6, scope: !1082)
!1087 = !DILocation(line: 809, column: 14, scope: !1082)
!1088 = !DILocation(line: 809, column: 22, scope: !1082)
!1089 = !DILocation(line: 809, column: 20, scope: !1082)
!1090 = !DILocation(line: 809, column: 10, scope: !1082)
!1091 = !DILocation(line: 809, column: 6, scope: !1082)
!1092 = !DILocation(line: 810, column: 12, scope: !1082)
!1093 = !DILocation(line: 810, column: 10, scope: !1082)
!1094 = !DILocation(line: 810, column: 6, scope: !1082)
!1095 = !DILocation(line: 811, column: 8, scope: !1082)
!1096 = !DILocation(line: 811, column: 6, scope: !1082)
!1097 = !DILocation(line: 812, column: 9, scope: !1098)
!1098 = distinct !DILexicalBlock(scope: !1082, file: !6, line: 812, column: 3)
!1099 = !DILocation(line: 812, column: 7, scope: !1098)
!1100 = !DILocation(line: 812, column: 13, scope: !1101)
!1101 = distinct !DILexicalBlock(scope: !1098, file: !6, line: 812, column: 3)
!1102 = !DILocation(line: 812, column: 17, scope: !1101)
!1103 = !DILocation(line: 812, column: 19, scope: !1101)
!1104 = !DILocation(line: 812, column: 15, scope: !1101)
!1105 = !DILocation(line: 812, column: 3, scope: !1098)
!1106 = !DILocation(line: 813, column: 10, scope: !1107)
!1107 = distinct !DILexicalBlock(scope: !1108, file: !6, line: 813, column: 4)
!1108 = distinct !DILexicalBlock(scope: !1101, file: !6, line: 812, column: 28)
!1109 = !DILocation(line: 813, column: 8, scope: !1107)
!1110 = !DILocation(line: 813, column: 14, scope: !1111)
!1111 = distinct !DILexicalBlock(scope: !1107, file: !6, line: 813, column: 4)
!1112 = !DILocation(line: 813, column: 18, scope: !1111)
!1113 = !DILocation(line: 813, column: 20, scope: !1111)
!1114 = !DILocation(line: 813, column: 16, scope: !1111)
!1115 = !DILocation(line: 813, column: 4, scope: !1107)
!1116 = !DILocation(line: 814, column: 11, scope: !1117)
!1117 = distinct !DILexicalBlock(scope: !1111, file: !6, line: 813, column: 29)
!1118 = !DILocation(line: 814, column: 16, scope: !1117)
!1119 = !DILocation(line: 814, column: 14, scope: !1117)
!1120 = !DILocation(line: 814, column: 9, scope: !1117)
!1121 = !DILocation(line: 815, column: 11, scope: !1117)
!1122 = !DILocation(line: 815, column: 17, scope: !1117)
!1123 = !DILocation(line: 815, column: 15, scope: !1117)
!1124 = !DILocation(line: 815, column: 9, scope: !1117)
!1125 = !DILocation(line: 816, column: 11, scope: !1117)
!1126 = !DILocation(line: 816, column: 16, scope: !1117)
!1127 = !DILocation(line: 816, column: 14, scope: !1117)
!1128 = !DILocation(line: 816, column: 9, scope: !1117)
!1129 = !DILocation(line: 817, column: 11, scope: !1117)
!1130 = !DILocation(line: 817, column: 17, scope: !1117)
!1131 = !DILocation(line: 817, column: 15, scope: !1117)
!1132 = !DILocation(line: 817, column: 9, scope: !1117)
!1133 = !DILocation(line: 819, column: 16, scope: !1117)
!1134 = !DILocation(line: 819, column: 25, scope: !1117)
!1135 = !DILocation(line: 819, column: 28, scope: !1117)
!1136 = !DILocation(line: 819, column: 27, scope: !1117)
!1137 = !DILocation(line: 819, column: 32, scope: !1117)
!1138 = !DILocation(line: 819, column: 14, scope: !1117)
!1139 = !DILocation(line: 820, column: 16, scope: !1117)
!1140 = !DILocation(line: 820, column: 19, scope: !1117)
!1141 = !DILocation(line: 820, column: 28, scope: !1117)
!1142 = !DILocation(line: 820, column: 31, scope: !1117)
!1143 = !DILocation(line: 820, column: 30, scope: !1117)
!1144 = !DILocation(line: 820, column: 35, scope: !1117)
!1145 = !DILocation(line: 820, column: 18, scope: !1117)
!1146 = !DILocation(line: 820, column: 14, scope: !1117)
!1147 = !DILocation(line: 823, column: 16, scope: !1117)
!1148 = !DILocation(line: 823, column: 21, scope: !1117)
!1149 = !DILocation(line: 823, column: 26, scope: !1117)
!1150 = !DILocation(line: 823, column: 30, scope: !1117)
!1151 = !DILocation(line: 823, column: 29, scope: !1117)
!1152 = !DILocation(line: 823, column: 33, scope: !1117)
!1153 = !DILocation(line: 823, column: 23, scope: !1117)
!1154 = !DILocation(line: 823, column: 39, scope: !1117)
!1155 = !DILocation(line: 823, column: 40, scope: !1117)
!1156 = !DILocation(line: 823, column: 43, scope: !1117)
!1157 = !DILocation(line: 823, column: 37, scope: !1117)
!1158 = !DILocation(line: 823, column: 48, scope: !1117)
!1159 = !DILocation(line: 823, column: 14, scope: !1117)
!1160 = !DILocation(line: 824, column: 16, scope: !1117)
!1161 = !DILocation(line: 824, column: 21, scope: !1117)
!1162 = !DILocation(line: 824, column: 26, scope: !1117)
!1163 = !DILocation(line: 824, column: 30, scope: !1117)
!1164 = !DILocation(line: 824, column: 29, scope: !1117)
!1165 = !DILocation(line: 824, column: 33, scope: !1117)
!1166 = !DILocation(line: 824, column: 23, scope: !1117)
!1167 = !DILocation(line: 824, column: 39, scope: !1117)
!1168 = !DILocation(line: 824, column: 40, scope: !1117)
!1169 = !DILocation(line: 824, column: 43, scope: !1117)
!1170 = !DILocation(line: 824, column: 37, scope: !1117)
!1171 = !DILocation(line: 824, column: 48, scope: !1117)
!1172 = !DILocation(line: 824, column: 14, scope: !1117)
!1173 = !DILocation(line: 827, column: 16, scope: !1117)
!1174 = !DILocation(line: 827, column: 21, scope: !1117)
!1175 = !DILocation(line: 827, column: 26, scope: !1117)
!1176 = !DILocation(line: 827, column: 30, scope: !1117)
!1177 = !DILocation(line: 827, column: 29, scope: !1117)
!1178 = !DILocation(line: 827, column: 33, scope: !1117)
!1179 = !DILocation(line: 827, column: 23, scope: !1117)
!1180 = !DILocation(line: 827, column: 39, scope: !1117)
!1181 = !DILocation(line: 827, column: 40, scope: !1117)
!1182 = !DILocation(line: 827, column: 43, scope: !1117)
!1183 = !DILocation(line: 827, column: 37, scope: !1117)
!1184 = !DILocation(line: 827, column: 48, scope: !1117)
!1185 = !DILocation(line: 827, column: 14, scope: !1117)
!1186 = !DILocation(line: 828, column: 16, scope: !1117)
!1187 = !DILocation(line: 828, column: 21, scope: !1117)
!1188 = !DILocation(line: 828, column: 26, scope: !1117)
!1189 = !DILocation(line: 828, column: 30, scope: !1117)
!1190 = !DILocation(line: 828, column: 29, scope: !1117)
!1191 = !DILocation(line: 828, column: 33, scope: !1117)
!1192 = !DILocation(line: 828, column: 23, scope: !1117)
!1193 = !DILocation(line: 828, column: 39, scope: !1117)
!1194 = !DILocation(line: 828, column: 40, scope: !1117)
!1195 = !DILocation(line: 828, column: 43, scope: !1117)
!1196 = !DILocation(line: 828, column: 37, scope: !1117)
!1197 = !DILocation(line: 828, column: 48, scope: !1117)
!1198 = !DILocation(line: 828, column: 14, scope: !1117)
!1199 = !DILocation(line: 831, column: 44, scope: !1117)
!1200 = !DILocation(line: 831, column: 55, scope: !1117)
!1201 = !DILocation(line: 831, column: 53, scope: !1117)
!1202 = !DILocation(line: 831, column: 5, scope: !1117)
!1203 = !DILocation(line: 831, column: 10, scope: !1117)
!1204 = !DILocation(line: 831, column: 15, scope: !1117)
!1205 = !DILocation(line: 831, column: 19, scope: !1117)
!1206 = !DILocation(line: 831, column: 18, scope: !1117)
!1207 = !DILocation(line: 831, column: 22, scope: !1117)
!1208 = !DILocation(line: 831, column: 12, scope: !1117)
!1209 = !DILocation(line: 831, column: 28, scope: !1117)
!1210 = !DILocation(line: 831, column: 29, scope: !1117)
!1211 = !DILocation(line: 831, column: 32, scope: !1117)
!1212 = !DILocation(line: 831, column: 26, scope: !1117)
!1213 = !DILocation(line: 831, column: 37, scope: !1117)
!1214 = !DILocation(line: 831, column: 42, scope: !1117)
!1215 = !DILocation(line: 832, column: 44, scope: !1117)
!1216 = !DILocation(line: 832, column: 55, scope: !1117)
!1217 = !DILocation(line: 832, column: 53, scope: !1117)
!1218 = !DILocation(line: 832, column: 5, scope: !1117)
!1219 = !DILocation(line: 832, column: 10, scope: !1117)
!1220 = !DILocation(line: 832, column: 15, scope: !1117)
!1221 = !DILocation(line: 832, column: 19, scope: !1117)
!1222 = !DILocation(line: 832, column: 18, scope: !1117)
!1223 = !DILocation(line: 832, column: 22, scope: !1117)
!1224 = !DILocation(line: 832, column: 12, scope: !1117)
!1225 = !DILocation(line: 832, column: 28, scope: !1117)
!1226 = !DILocation(line: 832, column: 29, scope: !1117)
!1227 = !DILocation(line: 832, column: 32, scope: !1117)
!1228 = !DILocation(line: 832, column: 26, scope: !1117)
!1229 = !DILocation(line: 832, column: 37, scope: !1117)
!1230 = !DILocation(line: 832, column: 42, scope: !1117)
!1231 = !DILocation(line: 834, column: 17, scope: !1117)
!1232 = !DILocation(line: 834, column: 28, scope: !1117)
!1233 = !DILocation(line: 834, column: 26, scope: !1117)
!1234 = !DILocation(line: 834, column: 15, scope: !1117)
!1235 = !DILocation(line: 835, column: 17, scope: !1117)
!1236 = !DILocation(line: 835, column: 28, scope: !1117)
!1237 = !DILocation(line: 835, column: 26, scope: !1117)
!1238 = !DILocation(line: 835, column: 15, scope: !1117)
!1239 = !DILocation(line: 838, column: 45, scope: !1117)
!1240 = !DILocation(line: 838, column: 56, scope: !1117)
!1241 = !DILocation(line: 838, column: 54, scope: !1117)
!1242 = !DILocation(line: 838, column: 70, scope: !1117)
!1243 = !DILocation(line: 838, column: 81, scope: !1117)
!1244 = !DILocation(line: 838, column: 79, scope: !1117)
!1245 = !DILocation(line: 838, column: 67, scope: !1117)
!1246 = !DILocation(line: 838, column: 5, scope: !1117)
!1247 = !DILocation(line: 838, column: 10, scope: !1117)
!1248 = !DILocation(line: 838, column: 15, scope: !1117)
!1249 = !DILocation(line: 838, column: 19, scope: !1117)
!1250 = !DILocation(line: 838, column: 18, scope: !1117)
!1251 = !DILocation(line: 838, column: 22, scope: !1117)
!1252 = !DILocation(line: 838, column: 12, scope: !1117)
!1253 = !DILocation(line: 838, column: 28, scope: !1117)
!1254 = !DILocation(line: 838, column: 29, scope: !1117)
!1255 = !DILocation(line: 838, column: 32, scope: !1117)
!1256 = !DILocation(line: 838, column: 26, scope: !1117)
!1257 = !DILocation(line: 838, column: 37, scope: !1117)
!1258 = !DILocation(line: 838, column: 42, scope: !1117)
!1259 = !DILocation(line: 839, column: 45, scope: !1117)
!1260 = !DILocation(line: 839, column: 56, scope: !1117)
!1261 = !DILocation(line: 839, column: 54, scope: !1117)
!1262 = !DILocation(line: 839, column: 70, scope: !1117)
!1263 = !DILocation(line: 839, column: 81, scope: !1117)
!1264 = !DILocation(line: 839, column: 79, scope: !1117)
!1265 = !DILocation(line: 839, column: 67, scope: !1117)
!1266 = !DILocation(line: 839, column: 5, scope: !1117)
!1267 = !DILocation(line: 839, column: 10, scope: !1117)
!1268 = !DILocation(line: 839, column: 15, scope: !1117)
!1269 = !DILocation(line: 839, column: 19, scope: !1117)
!1270 = !DILocation(line: 839, column: 18, scope: !1117)
!1271 = !DILocation(line: 839, column: 22, scope: !1117)
!1272 = !DILocation(line: 839, column: 12, scope: !1117)
!1273 = !DILocation(line: 839, column: 28, scope: !1117)
!1274 = !DILocation(line: 839, column: 29, scope: !1117)
!1275 = !DILocation(line: 839, column: 32, scope: !1117)
!1276 = !DILocation(line: 839, column: 26, scope: !1117)
!1277 = !DILocation(line: 839, column: 37, scope: !1117)
!1278 = !DILocation(line: 839, column: 42, scope: !1117)
!1279 = !DILocation(line: 840, column: 4, scope: !1117)
!1280 = !DILocation(line: 813, column: 26, scope: !1111)
!1281 = !DILocation(line: 813, column: 4, scope: !1111)
!1282 = distinct !{!1282, !1115, !1283}
!1283 = !DILocation(line: 840, column: 4, scope: !1107)
!1284 = !DILocation(line: 841, column: 3, scope: !1108)
!1285 = !DILocation(line: 812, column: 25, scope: !1101)
!1286 = !DILocation(line: 812, column: 3, scope: !1101)
!1287 = distinct !{!1287, !1105, !1288}
!1288 = !DILocation(line: 841, column: 3, scope: !1098)
!1289 = !DILocation(line: 842, column: 6, scope: !1290)
!1290 = distinct !DILexicalBlock(scope: !1082, file: !6, line: 842, column: 6)
!1291 = !DILocation(line: 842, column: 9, scope: !1290)
!1292 = !DILocation(line: 842, column: 7, scope: !1290)
!1293 = !DILocation(line: 842, column: 6, scope: !1082)
!1294 = !DILocation(line: 843, column: 10, scope: !1295)
!1295 = distinct !DILexicalBlock(scope: !1296, file: !6, line: 843, column: 4)
!1296 = distinct !DILexicalBlock(scope: !1290, file: !6, line: 842, column: 15)
!1297 = !DILocation(line: 843, column: 8, scope: !1295)
!1298 = !DILocation(line: 843, column: 14, scope: !1299)
!1299 = distinct !DILexicalBlock(scope: !1295, file: !6, line: 843, column: 4)
!1300 = !DILocation(line: 843, column: 16, scope: !1299)
!1301 = !DILocation(line: 843, column: 4, scope: !1295)
!1302 = !DILocation(line: 845, column: 38, scope: !1303)
!1303 = distinct !DILexicalBlock(scope: !1299, file: !6, line: 843, column: 26)
!1304 = !DILocation(line: 845, column: 43, scope: !1303)
!1305 = !DILocation(line: 845, column: 47, scope: !1303)
!1306 = !DILocation(line: 845, column: 49, scope: !1303)
!1307 = !DILocation(line: 845, column: 45, scope: !1303)
!1308 = !DILocation(line: 845, column: 55, scope: !1303)
!1309 = !DILocation(line: 845, column: 56, scope: !1303)
!1310 = !DILocation(line: 845, column: 59, scope: !1303)
!1311 = !DILocation(line: 845, column: 53, scope: !1303)
!1312 = !DILocation(line: 845, column: 64, scope: !1303)
!1313 = !DILocation(line: 845, column: 5, scope: !1303)
!1314 = !DILocation(line: 845, column: 10, scope: !1303)
!1315 = !DILocation(line: 845, column: 14, scope: !1303)
!1316 = !DILocation(line: 845, column: 16, scope: !1303)
!1317 = !DILocation(line: 845, column: 12, scope: !1303)
!1318 = !DILocation(line: 845, column: 22, scope: !1303)
!1319 = !DILocation(line: 845, column: 23, scope: !1303)
!1320 = !DILocation(line: 845, column: 26, scope: !1303)
!1321 = !DILocation(line: 845, column: 20, scope: !1303)
!1322 = !DILocation(line: 845, column: 31, scope: !1303)
!1323 = !DILocation(line: 845, column: 36, scope: !1303)
!1324 = !DILocation(line: 846, column: 38, scope: !1303)
!1325 = !DILocation(line: 846, column: 43, scope: !1303)
!1326 = !DILocation(line: 846, column: 47, scope: !1303)
!1327 = !DILocation(line: 846, column: 49, scope: !1303)
!1328 = !DILocation(line: 846, column: 45, scope: !1303)
!1329 = !DILocation(line: 846, column: 55, scope: !1303)
!1330 = !DILocation(line: 846, column: 56, scope: !1303)
!1331 = !DILocation(line: 846, column: 59, scope: !1303)
!1332 = !DILocation(line: 846, column: 53, scope: !1303)
!1333 = !DILocation(line: 846, column: 64, scope: !1303)
!1334 = !DILocation(line: 846, column: 5, scope: !1303)
!1335 = !DILocation(line: 846, column: 10, scope: !1303)
!1336 = !DILocation(line: 846, column: 14, scope: !1303)
!1337 = !DILocation(line: 846, column: 16, scope: !1303)
!1338 = !DILocation(line: 846, column: 12, scope: !1303)
!1339 = !DILocation(line: 846, column: 22, scope: !1303)
!1340 = !DILocation(line: 846, column: 23, scope: !1303)
!1341 = !DILocation(line: 846, column: 26, scope: !1303)
!1342 = !DILocation(line: 846, column: 20, scope: !1303)
!1343 = !DILocation(line: 846, column: 31, scope: !1303)
!1344 = !DILocation(line: 846, column: 36, scope: !1303)
!1345 = !DILocation(line: 847, column: 4, scope: !1303)
!1346 = !DILocation(line: 843, column: 23, scope: !1299)
!1347 = !DILocation(line: 843, column: 4, scope: !1299)
!1348 = distinct !{!1348, !1301, !1349}
!1349 = !DILocation(line: 847, column: 4, scope: !1295)
!1350 = !DILocation(line: 848, column: 3, scope: !1296)
!1351 = !DILocation(line: 849, column: 7, scope: !1352)
!1352 = distinct !DILexicalBlock(scope: !1290, file: !6, line: 848, column: 8)
!1353 = !DILocation(line: 850, column: 15, scope: !1352)
!1354 = !DILocation(line: 850, column: 16, scope: !1352)
!1355 = !DILocation(line: 850, column: 19, scope: !1352)
!1356 = !DILocation(line: 850, column: 11, scope: !1352)
!1357 = !DILocation(line: 850, column: 7, scope: !1352)
!1358 = !DILocation(line: 851, column: 15, scope: !1352)
!1359 = !DILocation(line: 851, column: 24, scope: !1352)
!1360 = !DILocation(line: 851, column: 25, scope: !1352)
!1361 = !DILocation(line: 851, column: 21, scope: !1352)
!1362 = !DILocation(line: 851, column: 11, scope: !1352)
!1363 = !DILocation(line: 851, column: 7, scope: !1352)
!1364 = !DILocation(line: 852, column: 13, scope: !1352)
!1365 = !DILocation(line: 852, column: 11, scope: !1352)
!1366 = !DILocation(line: 852, column: 7, scope: !1352)
!1367 = !DILocation(line: 853, column: 9, scope: !1352)
!1368 = !DILocation(line: 853, column: 7, scope: !1352)
!1369 = !DILocation(line: 854, column: 10, scope: !1370)
!1370 = distinct !DILexicalBlock(scope: !1352, file: !6, line: 854, column: 4)
!1371 = !DILocation(line: 854, column: 8, scope: !1370)
!1372 = !DILocation(line: 854, column: 14, scope: !1373)
!1373 = distinct !DILexicalBlock(scope: !1370, file: !6, line: 854, column: 4)
!1374 = !DILocation(line: 854, column: 18, scope: !1373)
!1375 = !DILocation(line: 854, column: 20, scope: !1373)
!1376 = !DILocation(line: 854, column: 16, scope: !1373)
!1377 = !DILocation(line: 854, column: 4, scope: !1370)
!1378 = !DILocation(line: 855, column: 11, scope: !1379)
!1379 = distinct !DILexicalBlock(scope: !1380, file: !6, line: 855, column: 5)
!1380 = distinct !DILexicalBlock(scope: !1373, file: !6, line: 854, column: 29)
!1381 = !DILocation(line: 855, column: 9, scope: !1379)
!1382 = !DILocation(line: 855, column: 15, scope: !1383)
!1383 = distinct !DILexicalBlock(scope: !1379, file: !6, line: 855, column: 5)
!1384 = !DILocation(line: 855, column: 19, scope: !1383)
!1385 = !DILocation(line: 855, column: 21, scope: !1383)
!1386 = !DILocation(line: 855, column: 17, scope: !1383)
!1387 = !DILocation(line: 855, column: 5, scope: !1379)
!1388 = !DILocation(line: 856, column: 12, scope: !1389)
!1389 = distinct !DILexicalBlock(scope: !1383, file: !6, line: 855, column: 30)
!1390 = !DILocation(line: 856, column: 17, scope: !1389)
!1391 = !DILocation(line: 856, column: 15, scope: !1389)
!1392 = !DILocation(line: 856, column: 10, scope: !1389)
!1393 = !DILocation(line: 857, column: 12, scope: !1389)
!1394 = !DILocation(line: 857, column: 18, scope: !1389)
!1395 = !DILocation(line: 857, column: 16, scope: !1389)
!1396 = !DILocation(line: 857, column: 10, scope: !1389)
!1397 = !DILocation(line: 858, column: 12, scope: !1389)
!1398 = !DILocation(line: 858, column: 17, scope: !1389)
!1399 = !DILocation(line: 858, column: 15, scope: !1389)
!1400 = !DILocation(line: 858, column: 10, scope: !1389)
!1401 = !DILocation(line: 859, column: 12, scope: !1389)
!1402 = !DILocation(line: 859, column: 18, scope: !1389)
!1403 = !DILocation(line: 859, column: 16, scope: !1389)
!1404 = !DILocation(line: 859, column: 10, scope: !1389)
!1405 = !DILocation(line: 861, column: 17, scope: !1389)
!1406 = !DILocation(line: 861, column: 26, scope: !1389)
!1407 = !DILocation(line: 861, column: 29, scope: !1389)
!1408 = !DILocation(line: 861, column: 28, scope: !1389)
!1409 = !DILocation(line: 861, column: 33, scope: !1389)
!1410 = !DILocation(line: 861, column: 15, scope: !1389)
!1411 = !DILocation(line: 862, column: 17, scope: !1389)
!1412 = !DILocation(line: 862, column: 20, scope: !1389)
!1413 = !DILocation(line: 862, column: 29, scope: !1389)
!1414 = !DILocation(line: 862, column: 32, scope: !1389)
!1415 = !DILocation(line: 862, column: 31, scope: !1389)
!1416 = !DILocation(line: 862, column: 36, scope: !1389)
!1417 = !DILocation(line: 862, column: 19, scope: !1389)
!1418 = !DILocation(line: 862, column: 15, scope: !1389)
!1419 = !DILocation(line: 865, column: 17, scope: !1389)
!1420 = !DILocation(line: 865, column: 22, scope: !1389)
!1421 = !DILocation(line: 865, column: 27, scope: !1389)
!1422 = !DILocation(line: 865, column: 31, scope: !1389)
!1423 = !DILocation(line: 865, column: 30, scope: !1389)
!1424 = !DILocation(line: 865, column: 34, scope: !1389)
!1425 = !DILocation(line: 865, column: 24, scope: !1389)
!1426 = !DILocation(line: 865, column: 40, scope: !1389)
!1427 = !DILocation(line: 865, column: 41, scope: !1389)
!1428 = !DILocation(line: 865, column: 44, scope: !1389)
!1429 = !DILocation(line: 865, column: 38, scope: !1389)
!1430 = !DILocation(line: 865, column: 49, scope: !1389)
!1431 = !DILocation(line: 865, column: 15, scope: !1389)
!1432 = !DILocation(line: 866, column: 17, scope: !1389)
!1433 = !DILocation(line: 866, column: 22, scope: !1389)
!1434 = !DILocation(line: 866, column: 27, scope: !1389)
!1435 = !DILocation(line: 866, column: 31, scope: !1389)
!1436 = !DILocation(line: 866, column: 30, scope: !1389)
!1437 = !DILocation(line: 866, column: 34, scope: !1389)
!1438 = !DILocation(line: 866, column: 24, scope: !1389)
!1439 = !DILocation(line: 866, column: 40, scope: !1389)
!1440 = !DILocation(line: 866, column: 41, scope: !1389)
!1441 = !DILocation(line: 866, column: 44, scope: !1389)
!1442 = !DILocation(line: 866, column: 38, scope: !1389)
!1443 = !DILocation(line: 866, column: 49, scope: !1389)
!1444 = !DILocation(line: 866, column: 15, scope: !1389)
!1445 = !DILocation(line: 869, column: 17, scope: !1389)
!1446 = !DILocation(line: 869, column: 22, scope: !1389)
!1447 = !DILocation(line: 869, column: 27, scope: !1389)
!1448 = !DILocation(line: 869, column: 31, scope: !1389)
!1449 = !DILocation(line: 869, column: 30, scope: !1389)
!1450 = !DILocation(line: 869, column: 34, scope: !1389)
!1451 = !DILocation(line: 869, column: 24, scope: !1389)
!1452 = !DILocation(line: 869, column: 40, scope: !1389)
!1453 = !DILocation(line: 869, column: 41, scope: !1389)
!1454 = !DILocation(line: 869, column: 44, scope: !1389)
!1455 = !DILocation(line: 869, column: 38, scope: !1389)
!1456 = !DILocation(line: 869, column: 49, scope: !1389)
!1457 = !DILocation(line: 869, column: 15, scope: !1389)
!1458 = !DILocation(line: 870, column: 17, scope: !1389)
!1459 = !DILocation(line: 870, column: 22, scope: !1389)
!1460 = !DILocation(line: 870, column: 27, scope: !1389)
!1461 = !DILocation(line: 870, column: 31, scope: !1389)
!1462 = !DILocation(line: 870, column: 30, scope: !1389)
!1463 = !DILocation(line: 870, column: 34, scope: !1389)
!1464 = !DILocation(line: 870, column: 24, scope: !1389)
!1465 = !DILocation(line: 870, column: 40, scope: !1389)
!1466 = !DILocation(line: 870, column: 41, scope: !1389)
!1467 = !DILocation(line: 870, column: 44, scope: !1389)
!1468 = !DILocation(line: 870, column: 38, scope: !1389)
!1469 = !DILocation(line: 870, column: 49, scope: !1389)
!1470 = !DILocation(line: 870, column: 15, scope: !1389)
!1471 = !DILocation(line: 873, column: 45, scope: !1389)
!1472 = !DILocation(line: 873, column: 56, scope: !1389)
!1473 = !DILocation(line: 873, column: 54, scope: !1389)
!1474 = !DILocation(line: 873, column: 6, scope: !1389)
!1475 = !DILocation(line: 873, column: 11, scope: !1389)
!1476 = !DILocation(line: 873, column: 16, scope: !1389)
!1477 = !DILocation(line: 873, column: 20, scope: !1389)
!1478 = !DILocation(line: 873, column: 19, scope: !1389)
!1479 = !DILocation(line: 873, column: 23, scope: !1389)
!1480 = !DILocation(line: 873, column: 13, scope: !1389)
!1481 = !DILocation(line: 873, column: 29, scope: !1389)
!1482 = !DILocation(line: 873, column: 30, scope: !1389)
!1483 = !DILocation(line: 873, column: 33, scope: !1389)
!1484 = !DILocation(line: 873, column: 27, scope: !1389)
!1485 = !DILocation(line: 873, column: 38, scope: !1389)
!1486 = !DILocation(line: 873, column: 43, scope: !1389)
!1487 = !DILocation(line: 874, column: 45, scope: !1389)
!1488 = !DILocation(line: 874, column: 56, scope: !1389)
!1489 = !DILocation(line: 874, column: 54, scope: !1389)
!1490 = !DILocation(line: 874, column: 6, scope: !1389)
!1491 = !DILocation(line: 874, column: 11, scope: !1389)
!1492 = !DILocation(line: 874, column: 16, scope: !1389)
!1493 = !DILocation(line: 874, column: 20, scope: !1389)
!1494 = !DILocation(line: 874, column: 19, scope: !1389)
!1495 = !DILocation(line: 874, column: 23, scope: !1389)
!1496 = !DILocation(line: 874, column: 13, scope: !1389)
!1497 = !DILocation(line: 874, column: 29, scope: !1389)
!1498 = !DILocation(line: 874, column: 30, scope: !1389)
!1499 = !DILocation(line: 874, column: 33, scope: !1389)
!1500 = !DILocation(line: 874, column: 27, scope: !1389)
!1501 = !DILocation(line: 874, column: 38, scope: !1389)
!1502 = !DILocation(line: 874, column: 43, scope: !1389)
!1503 = !DILocation(line: 876, column: 19, scope: !1389)
!1504 = !DILocation(line: 876, column: 30, scope: !1389)
!1505 = !DILocation(line: 876, column: 28, scope: !1389)
!1506 = !DILocation(line: 876, column: 17, scope: !1389)
!1507 = !DILocation(line: 877, column: 19, scope: !1389)
!1508 = !DILocation(line: 877, column: 30, scope: !1389)
!1509 = !DILocation(line: 877, column: 28, scope: !1389)
!1510 = !DILocation(line: 877, column: 17, scope: !1389)
!1511 = !DILocation(line: 880, column: 46, scope: !1389)
!1512 = !DILocation(line: 880, column: 57, scope: !1389)
!1513 = !DILocation(line: 880, column: 55, scope: !1389)
!1514 = !DILocation(line: 880, column: 72, scope: !1389)
!1515 = !DILocation(line: 880, column: 83, scope: !1389)
!1516 = !DILocation(line: 880, column: 81, scope: !1389)
!1517 = !DILocation(line: 880, column: 69, scope: !1389)
!1518 = !DILocation(line: 880, column: 6, scope: !1389)
!1519 = !DILocation(line: 880, column: 11, scope: !1389)
!1520 = !DILocation(line: 880, column: 16, scope: !1389)
!1521 = !DILocation(line: 880, column: 20, scope: !1389)
!1522 = !DILocation(line: 880, column: 19, scope: !1389)
!1523 = !DILocation(line: 880, column: 23, scope: !1389)
!1524 = !DILocation(line: 880, column: 13, scope: !1389)
!1525 = !DILocation(line: 880, column: 29, scope: !1389)
!1526 = !DILocation(line: 880, column: 30, scope: !1389)
!1527 = !DILocation(line: 880, column: 33, scope: !1389)
!1528 = !DILocation(line: 880, column: 27, scope: !1389)
!1529 = !DILocation(line: 880, column: 38, scope: !1389)
!1530 = !DILocation(line: 880, column: 43, scope: !1389)
!1531 = !DILocation(line: 881, column: 46, scope: !1389)
!1532 = !DILocation(line: 881, column: 57, scope: !1389)
!1533 = !DILocation(line: 881, column: 55, scope: !1389)
!1534 = !DILocation(line: 881, column: 72, scope: !1389)
!1535 = !DILocation(line: 881, column: 83, scope: !1389)
!1536 = !DILocation(line: 881, column: 81, scope: !1389)
!1537 = !DILocation(line: 881, column: 69, scope: !1389)
!1538 = !DILocation(line: 881, column: 6, scope: !1389)
!1539 = !DILocation(line: 881, column: 11, scope: !1389)
!1540 = !DILocation(line: 881, column: 16, scope: !1389)
!1541 = !DILocation(line: 881, column: 20, scope: !1389)
!1542 = !DILocation(line: 881, column: 19, scope: !1389)
!1543 = !DILocation(line: 881, column: 23, scope: !1389)
!1544 = !DILocation(line: 881, column: 13, scope: !1389)
!1545 = !DILocation(line: 881, column: 29, scope: !1389)
!1546 = !DILocation(line: 881, column: 30, scope: !1389)
!1547 = !DILocation(line: 881, column: 33, scope: !1389)
!1548 = !DILocation(line: 881, column: 27, scope: !1389)
!1549 = !DILocation(line: 881, column: 38, scope: !1389)
!1550 = !DILocation(line: 881, column: 43, scope: !1389)
!1551 = !DILocation(line: 882, column: 5, scope: !1389)
!1552 = !DILocation(line: 855, column: 27, scope: !1383)
!1553 = !DILocation(line: 855, column: 5, scope: !1383)
!1554 = distinct !{!1554, !1387, !1555}
!1555 = !DILocation(line: 882, column: 5, scope: !1379)
!1556 = !DILocation(line: 883, column: 4, scope: !1380)
!1557 = !DILocation(line: 854, column: 26, scope: !1373)
!1558 = !DILocation(line: 854, column: 4, scope: !1373)
!1559 = distinct !{!1559, !1377, !1560}
!1560 = !DILocation(line: 883, column: 4, scope: !1370)
!1561 = !DILocation(line: 885, column: 2, scope: !1082)
!1562 = !DILocation(line: 806, column: 22, scope: !1077)
!1563 = !DILocation(line: 806, column: 2, scope: !1077)
!1564 = distinct !{!1564, !1080, !1565}
!1565 = !DILocation(line: 885, column: 2, scope: !1074)
!1566 = !DILocation(line: 886, column: 1, scope: !973)
!1567 = distinct !DISubprogram(name: "ilog2_device", linkageName: "_Z12ilog2_devicei", scope: !6, file: !6, line: 1512, type: !29, scopeLine: 1512, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !7)
!1568 = !DILocalVariable(name: "n", arg: 1, scope: !1567, file: !6, line: 1512, type: !18)
!1569 = !DILocation(line: 1512, column: 33, scope: !1567)
!1570 = !DILocalVariable(name: "nn", scope: !1567, file: !6, line: 1513, type: !18)
!1571 = !DILocation(line: 1513, column: 6, scope: !1567)
!1572 = !DILocalVariable(name: "lg", scope: !1567, file: !6, line: 1513, type: !18)
!1573 = !DILocation(line: 1513, column: 10, scope: !1567)
!1574 = !DILocation(line: 1514, column: 5, scope: !1575)
!1575 = distinct !DILexicalBlock(scope: !1567, file: !6, line: 1514, column: 5)
!1576 = !DILocation(line: 1514, column: 6, scope: !1575)
!1577 = !DILocation(line: 1514, column: 5, scope: !1567)
!1578 = !DILocation(line: 1515, column: 3, scope: !1579)
!1579 = distinct !DILexicalBlock(scope: !1575, file: !6, line: 1514, column: 10)
!1580 = !DILocation(line: 1517, column: 5, scope: !1567)
!1581 = !DILocation(line: 1518, column: 5, scope: !1567)
!1582 = !DILocation(line: 1519, column: 2, scope: !1567)
!1583 = !DILocation(line: 1519, column: 8, scope: !1567)
!1584 = !DILocation(line: 1519, column: 11, scope: !1567)
!1585 = !DILocation(line: 1519, column: 10, scope: !1567)
!1586 = !DILocation(line: 1520, column: 8, scope: !1587)
!1587 = distinct !DILexicalBlock(scope: !1567, file: !6, line: 1519, column: 13)
!1588 = !DILocation(line: 1520, column: 11, scope: !1587)
!1589 = !DILocation(line: 1520, column: 6, scope: !1587)
!1590 = !DILocation(line: 1521, column: 5, scope: !1587)
!1591 = distinct !{!1591, !1582, !1592}
!1592 = !DILocation(line: 1522, column: 2, scope: !1567)
!1593 = !DILocation(line: 1523, column: 9, scope: !1567)
!1594 = !DILocation(line: 1523, column: 2, scope: !1567)
!1595 = !DILocation(line: 1524, column: 1, scope: !1567)
!1596 = distinct !DISubprogram(name: "cffts1_gpu_kernel_3", linkageName: "_Z19cffts1_gpu_kernel_3P8dcomplexS0_", scope: !6, file: !6, line: 895, type: !803, scopeLine: 896, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !7)
!1597 = !DILocalVariable(name: "x_out", arg: 1, scope: !1596, file: !6, line: 895, type: !9)
!1598 = !DILocation(line: 895, column: 46, scope: !1596)
!1599 = !DILocalVariable(name: "y0", arg: 2, scope: !1596, file: !6, line: 896, type: !9)
!1600 = !DILocation(line: 896, column: 12, scope: !1596)
!1601 = !DILocalVariable(name: "x_y_z", scope: !1596, file: !6, line: 897, type: !18)
!1602 = !DILocation(line: 897, column: 6, scope: !1596)
!1603 = !DILocation(line: 64, column: 3, scope: !812, inlinedAt: !1604)
!1604 = distinct !DILocation(line: 897, column: 14, scope: !1596)
!1605 = !DILocation(line: 75, column: 3, scope: !850, inlinedAt: !1606)
!1606 = distinct !DILocation(line: 897, column: 27, scope: !1596)
!1607 = !DILocation(line: 897, column: 25, scope: !1596)
!1608 = !DILocation(line: 53, column: 3, scope: !896, inlinedAt: !1609)
!1609 = distinct !DILocation(line: 897, column: 40, scope: !1596)
!1610 = !DILocation(line: 897, column: 38, scope: !1596)
!1611 = !DILocation(line: 898, column: 5, scope: !1612)
!1612 = distinct !DILexicalBlock(scope: !1596, file: !6, line: 898, column: 5)
!1613 = !DILocation(line: 898, column: 11, scope: !1612)
!1614 = !DILocation(line: 898, column: 5, scope: !1596)
!1615 = !DILocation(line: 899, column: 3, scope: !1616)
!1616 = distinct !DILexicalBlock(scope: !1612, file: !6, line: 898, column: 25)
!1617 = !DILocalVariable(name: "x", scope: !1596, file: !6, line: 901, type: !18)
!1618 = !DILocation(line: 901, column: 6, scope: !1596)
!1619 = !DILocation(line: 901, column: 10, scope: !1596)
!1620 = !DILocation(line: 901, column: 16, scope: !1596)
!1621 = !DILocalVariable(name: "y", scope: !1596, file: !6, line: 902, type: !18)
!1622 = !DILocation(line: 902, column: 6, scope: !1596)
!1623 = !DILocation(line: 902, column: 11, scope: !1596)
!1624 = !DILocation(line: 902, column: 17, scope: !1596)
!1625 = !DILocation(line: 902, column: 23, scope: !1596)
!1626 = !DILocalVariable(name: "z", scope: !1596, file: !6, line: 903, type: !18)
!1627 = !DILocation(line: 903, column: 6, scope: !1596)
!1628 = !DILocation(line: 903, column: 10, scope: !1596)
!1629 = !DILocation(line: 903, column: 16, scope: !1596)
!1630 = !DILocation(line: 904, column: 22, scope: !1596)
!1631 = !DILocation(line: 904, column: 25, scope: !1596)
!1632 = !DILocation(line: 904, column: 28, scope: !1596)
!1633 = !DILocation(line: 904, column: 29, scope: !1596)
!1634 = !DILocation(line: 904, column: 26, scope: !1596)
!1635 = !DILocation(line: 904, column: 35, scope: !1596)
!1636 = !DILocation(line: 904, column: 36, scope: !1596)
!1637 = !DILocation(line: 904, column: 39, scope: !1596)
!1638 = !DILocation(line: 904, column: 33, scope: !1596)
!1639 = !DILocation(line: 904, column: 45, scope: !1596)
!1640 = !DILocation(line: 904, column: 2, scope: !1596)
!1641 = !DILocation(line: 904, column: 8, scope: !1596)
!1642 = !DILocation(line: 904, column: 15, scope: !1596)
!1643 = !DILocation(line: 904, column: 20, scope: !1596)
!1644 = !DILocation(line: 905, column: 22, scope: !1596)
!1645 = !DILocation(line: 905, column: 25, scope: !1596)
!1646 = !DILocation(line: 905, column: 28, scope: !1596)
!1647 = !DILocation(line: 905, column: 29, scope: !1596)
!1648 = !DILocation(line: 905, column: 26, scope: !1596)
!1649 = !DILocation(line: 905, column: 35, scope: !1596)
!1650 = !DILocation(line: 905, column: 36, scope: !1596)
!1651 = !DILocation(line: 905, column: 39, scope: !1596)
!1652 = !DILocation(line: 905, column: 33, scope: !1596)
!1653 = !DILocation(line: 905, column: 45, scope: !1596)
!1654 = !DILocation(line: 905, column: 2, scope: !1596)
!1655 = !DILocation(line: 905, column: 8, scope: !1596)
!1656 = !DILocation(line: 905, column: 15, scope: !1596)
!1657 = !DILocation(line: 905, column: 20, scope: !1596)
!1658 = !DILocation(line: 906, column: 1, scope: !1596)
!1659 = distinct !DISubprogram(name: "cffts2_gpu_kernel_1", linkageName: "_Z19cffts2_gpu_kernel_1P8dcomplexS0_", scope: !6, file: !6, line: 957, type: !803, scopeLine: 958, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !7)
!1660 = !DILocalVariable(name: "x_in", arg: 1, scope: !1659, file: !6, line: 957, type: !9)
!1661 = !DILocation(line: 957, column: 46, scope: !1659)
!1662 = !DILocalVariable(name: "y0", arg: 2, scope: !1659, file: !6, line: 958, type: !9)
!1663 = !DILocation(line: 958, column: 12, scope: !1659)
!1664 = !DILocalVariable(name: "x_y_z", scope: !1659, file: !6, line: 959, type: !18)
!1665 = !DILocation(line: 959, column: 6, scope: !1659)
!1666 = !DILocation(line: 64, column: 3, scope: !812, inlinedAt: !1667)
!1667 = distinct !DILocation(line: 959, column: 14, scope: !1659)
!1668 = !DILocation(line: 75, column: 3, scope: !850, inlinedAt: !1669)
!1669 = distinct !DILocation(line: 959, column: 27, scope: !1659)
!1670 = !DILocation(line: 959, column: 25, scope: !1659)
!1671 = !DILocation(line: 53, column: 3, scope: !896, inlinedAt: !1672)
!1672 = distinct !DILocation(line: 959, column: 40, scope: !1659)
!1673 = !DILocation(line: 959, column: 38, scope: !1659)
!1674 = !DILocation(line: 960, column: 5, scope: !1675)
!1675 = distinct !DILexicalBlock(scope: !1659, file: !6, line: 960, column: 5)
!1676 = !DILocation(line: 960, column: 11, scope: !1675)
!1677 = !DILocation(line: 960, column: 5, scope: !1659)
!1678 = !DILocation(line: 961, column: 3, scope: !1679)
!1679 = distinct !DILexicalBlock(scope: !1675, file: !6, line: 960, column: 25)
!1680 = !DILocation(line: 963, column: 19, scope: !1659)
!1681 = !DILocation(line: 963, column: 24, scope: !1659)
!1682 = !DILocation(line: 963, column: 31, scope: !1659)
!1683 = !DILocation(line: 963, column: 2, scope: !1659)
!1684 = !DILocation(line: 963, column: 5, scope: !1659)
!1685 = !DILocation(line: 963, column: 12, scope: !1659)
!1686 = !DILocation(line: 963, column: 17, scope: !1659)
!1687 = !DILocation(line: 964, column: 19, scope: !1659)
!1688 = !DILocation(line: 964, column: 24, scope: !1659)
!1689 = !DILocation(line: 964, column: 31, scope: !1659)
!1690 = !DILocation(line: 964, column: 2, scope: !1659)
!1691 = !DILocation(line: 964, column: 5, scope: !1659)
!1692 = !DILocation(line: 964, column: 12, scope: !1659)
!1693 = !DILocation(line: 964, column: 17, scope: !1659)
!1694 = !DILocation(line: 965, column: 1, scope: !1659)
!1695 = distinct !DISubprogram(name: "cffts2_gpu_kernel_2", linkageName: "_Z19cffts2_gpu_kernel_2iP8dcomplexS0_S0_", scope: !6, file: !6, line: 972, type: !974, scopeLine: 975, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !7)
!1696 = !DILocalVariable(name: "is", arg: 1, scope: !1695, file: !6, line: 972, type: !976)
!1697 = !DILocation(line: 972, column: 47, scope: !1695)
!1698 = !DILocalVariable(name: "gty1", arg: 2, scope: !1695, file: !6, line: 973, type: !9)
!1699 = !DILocation(line: 973, column: 12, scope: !1695)
!1700 = !DILocalVariable(name: "gty2", arg: 3, scope: !1695, file: !6, line: 974, type: !9)
!1701 = !DILocation(line: 974, column: 12, scope: !1695)
!1702 = !DILocalVariable(name: "u_device", arg: 4, scope: !1695, file: !6, line: 975, type: !9)
!1703 = !DILocation(line: 975, column: 12, scope: !1695)
!1704 = !DILocalVariable(name: "x_z", scope: !1695, file: !6, line: 976, type: !18)
!1705 = !DILocation(line: 976, column: 6, scope: !1695)
!1706 = !DILocation(line: 64, column: 3, scope: !812, inlinedAt: !1707)
!1707 = distinct !DILocation(line: 976, column: 12, scope: !1695)
!1708 = !DILocation(line: 75, column: 3, scope: !850, inlinedAt: !1709)
!1709 = distinct !DILocation(line: 976, column: 25, scope: !1695)
!1710 = !DILocation(line: 976, column: 23, scope: !1695)
!1711 = !DILocation(line: 53, column: 3, scope: !896, inlinedAt: !1712)
!1712 = distinct !DILocation(line: 976, column: 38, scope: !1695)
!1713 = !DILocation(line: 976, column: 36, scope: !1695)
!1714 = !DILocation(line: 978, column: 5, scope: !1715)
!1715 = distinct !DILexicalBlock(scope: !1695, file: !6, line: 978, column: 5)
!1716 = !DILocation(line: 978, column: 9, scope: !1715)
!1717 = !DILocation(line: 978, column: 5, scope: !1695)
!1718 = !DILocation(line: 979, column: 3, scope: !1719)
!1719 = distinct !DILexicalBlock(scope: !1715, file: !6, line: 978, column: 20)
!1720 = !DILocalVariable(name: "i", scope: !1695, file: !6, line: 982, type: !18)
!1721 = !DILocation(line: 982, column: 6, scope: !1695)
!1722 = !DILocalVariable(name: "k", scope: !1695, file: !6, line: 982, type: !18)
!1723 = !DILocation(line: 982, column: 9, scope: !1695)
!1724 = !DILocalVariable(name: "l", scope: !1695, file: !6, line: 983, type: !18)
!1725 = !DILocation(line: 983, column: 6, scope: !1695)
!1726 = !DILocalVariable(name: "j1", scope: !1695, file: !6, line: 983, type: !18)
!1727 = !DILocation(line: 983, column: 9, scope: !1695)
!1728 = !DILocalVariable(name: "i1", scope: !1695, file: !6, line: 983, type: !18)
!1729 = !DILocation(line: 983, column: 13, scope: !1695)
!1730 = !DILocalVariable(name: "k1", scope: !1695, file: !6, line: 983, type: !18)
!1731 = !DILocation(line: 983, column: 17, scope: !1695)
!1732 = !DILocalVariable(name: "n1", scope: !1695, file: !6, line: 984, type: !18)
!1733 = !DILocation(line: 984, column: 6, scope: !1695)
!1734 = !DILocalVariable(name: "li", scope: !1695, file: !6, line: 984, type: !18)
!1735 = !DILocation(line: 984, column: 10, scope: !1695)
!1736 = !DILocalVariable(name: "lj", scope: !1695, file: !6, line: 984, type: !18)
!1737 = !DILocation(line: 984, column: 14, scope: !1695)
!1738 = !DILocalVariable(name: "lk", scope: !1695, file: !6, line: 984, type: !18)
!1739 = !DILocation(line: 984, column: 18, scope: !1695)
!1740 = !DILocalVariable(name: "ku", scope: !1695, file: !6, line: 984, type: !18)
!1741 = !DILocation(line: 984, column: 22, scope: !1695)
!1742 = !DILocalVariable(name: "i11", scope: !1695, file: !6, line: 984, type: !18)
!1743 = !DILocation(line: 984, column: 26, scope: !1695)
!1744 = !DILocalVariable(name: "i12", scope: !1695, file: !6, line: 984, type: !18)
!1745 = !DILocation(line: 984, column: 31, scope: !1695)
!1746 = !DILocalVariable(name: "i21", scope: !1695, file: !6, line: 984, type: !18)
!1747 = !DILocation(line: 984, column: 36, scope: !1695)
!1748 = !DILocalVariable(name: "i22", scope: !1695, file: !6, line: 984, type: !18)
!1749 = !DILocation(line: 984, column: 41, scope: !1695)
!1750 = !DILocation(line: 986, column: 6, scope: !1695)
!1751 = !DILocation(line: 986, column: 10, scope: !1695)
!1752 = !DILocation(line: 986, column: 4, scope: !1695)
!1753 = !DILocation(line: 987, column: 7, scope: !1695)
!1754 = !DILocation(line: 987, column: 11, scope: !1695)
!1755 = !DILocation(line: 987, column: 17, scope: !1695)
!1756 = !DILocation(line: 987, column: 4, scope: !1695)
!1757 = !DILocalVariable(name: "logd2", scope: !1695, file: !6, line: 989, type: !976)
!1758 = !DILocation(line: 989, column: 12, scope: !1695)
!1759 = !DILocation(line: 989, column: 20, scope: !1695)
!1760 = !DILocalVariable(name: "uu1_real", scope: !1695, file: !6, line: 991, type: !15)
!1761 = !DILocation(line: 991, column: 9, scope: !1695)
!1762 = !DILocalVariable(name: "x11_real", scope: !1695, file: !6, line: 991, type: !15)
!1763 = !DILocation(line: 991, column: 19, scope: !1695)
!1764 = !DILocalVariable(name: "x21_real", scope: !1695, file: !6, line: 991, type: !15)
!1765 = !DILocation(line: 991, column: 29, scope: !1695)
!1766 = !DILocalVariable(name: "uu1_imag", scope: !1695, file: !6, line: 992, type: !15)
!1767 = !DILocation(line: 992, column: 9, scope: !1695)
!1768 = !DILocalVariable(name: "x11_imag", scope: !1695, file: !6, line: 992, type: !15)
!1769 = !DILocation(line: 992, column: 19, scope: !1695)
!1770 = !DILocalVariable(name: "x21_imag", scope: !1695, file: !6, line: 992, type: !15)
!1771 = !DILocation(line: 992, column: 29, scope: !1695)
!1772 = !DILocalVariable(name: "uu2_real", scope: !1695, file: !6, line: 993, type: !15)
!1773 = !DILocation(line: 993, column: 9, scope: !1695)
!1774 = !DILocalVariable(name: "x12_real", scope: !1695, file: !6, line: 993, type: !15)
!1775 = !DILocation(line: 993, column: 19, scope: !1695)
!1776 = !DILocalVariable(name: "x22_real", scope: !1695, file: !6, line: 993, type: !15)
!1777 = !DILocation(line: 993, column: 29, scope: !1695)
!1778 = !DILocalVariable(name: "uu2_imag", scope: !1695, file: !6, line: 994, type: !15)
!1779 = !DILocation(line: 994, column: 9, scope: !1695)
!1780 = !DILocalVariable(name: "x12_imag", scope: !1695, file: !6, line: 994, type: !15)
!1781 = !DILocation(line: 994, column: 19, scope: !1695)
!1782 = !DILocalVariable(name: "x22_imag", scope: !1695, file: !6, line: 994, type: !15)
!1783 = !DILocation(line: 994, column: 29, scope: !1695)
!1784 = !DILocalVariable(name: "temp_real", scope: !1695, file: !6, line: 995, type: !15)
!1785 = !DILocation(line: 995, column: 9, scope: !1695)
!1786 = !DILocalVariable(name: "temp2_real", scope: !1695, file: !6, line: 995, type: !15)
!1787 = !DILocation(line: 995, column: 20, scope: !1695)
!1788 = !DILocalVariable(name: "temp_imag", scope: !1695, file: !6, line: 996, type: !15)
!1789 = !DILocation(line: 996, column: 9, scope: !1695)
!1790 = !DILocalVariable(name: "temp2_imag", scope: !1695, file: !6, line: 996, type: !15)
!1791 = !DILocation(line: 996, column: 20, scope: !1695)
!1792 = !DILocation(line: 998, column: 7, scope: !1793)
!1793 = distinct !DILexicalBlock(scope: !1695, file: !6, line: 998, column: 2)
!1794 = !DILocation(line: 998, column: 6, scope: !1793)
!1795 = !DILocation(line: 998, column: 11, scope: !1796)
!1796 = distinct !DILexicalBlock(scope: !1793, file: !6, line: 998, column: 2)
!1797 = !DILocation(line: 998, column: 14, scope: !1796)
!1798 = !DILocation(line: 998, column: 12, scope: !1796)
!1799 = !DILocation(line: 998, column: 2, scope: !1793)
!1800 = !DILocation(line: 999, column: 6, scope: !1801)
!1801 = distinct !DILexicalBlock(scope: !1796, file: !6, line: 998, column: 26)
!1802 = !DILocation(line: 1000, column: 14, scope: !1801)
!1803 = !DILocation(line: 1000, column: 16, scope: !1801)
!1804 = !DILocation(line: 1000, column: 10, scope: !1801)
!1805 = !DILocation(line: 1000, column: 6, scope: !1801)
!1806 = !DILocation(line: 1001, column: 14, scope: !1801)
!1807 = !DILocation(line: 1001, column: 22, scope: !1801)
!1808 = !DILocation(line: 1001, column: 20, scope: !1801)
!1809 = !DILocation(line: 1001, column: 10, scope: !1801)
!1810 = !DILocation(line: 1001, column: 6, scope: !1801)
!1811 = !DILocation(line: 1002, column: 12, scope: !1801)
!1812 = !DILocation(line: 1002, column: 10, scope: !1801)
!1813 = !DILocation(line: 1002, column: 6, scope: !1801)
!1814 = !DILocation(line: 1003, column: 8, scope: !1801)
!1815 = !DILocation(line: 1003, column: 6, scope: !1801)
!1816 = !DILocation(line: 1004, column: 9, scope: !1817)
!1817 = distinct !DILexicalBlock(scope: !1801, file: !6, line: 1004, column: 3)
!1818 = !DILocation(line: 1004, column: 7, scope: !1817)
!1819 = !DILocation(line: 1004, column: 13, scope: !1820)
!1820 = distinct !DILexicalBlock(scope: !1817, file: !6, line: 1004, column: 3)
!1821 = !DILocation(line: 1004, column: 17, scope: !1820)
!1822 = !DILocation(line: 1004, column: 19, scope: !1820)
!1823 = !DILocation(line: 1004, column: 15, scope: !1820)
!1824 = !DILocation(line: 1004, column: 3, scope: !1817)
!1825 = !DILocation(line: 1005, column: 10, scope: !1826)
!1826 = distinct !DILexicalBlock(scope: !1827, file: !6, line: 1005, column: 4)
!1827 = distinct !DILexicalBlock(scope: !1820, file: !6, line: 1004, column: 28)
!1828 = !DILocation(line: 1005, column: 8, scope: !1826)
!1829 = !DILocation(line: 1005, column: 14, scope: !1830)
!1830 = distinct !DILexicalBlock(scope: !1826, file: !6, line: 1005, column: 4)
!1831 = !DILocation(line: 1005, column: 18, scope: !1830)
!1832 = !DILocation(line: 1005, column: 20, scope: !1830)
!1833 = !DILocation(line: 1005, column: 16, scope: !1830)
!1834 = !DILocation(line: 1005, column: 4, scope: !1826)
!1835 = !DILocation(line: 1006, column: 11, scope: !1836)
!1836 = distinct !DILexicalBlock(scope: !1830, file: !6, line: 1005, column: 29)
!1837 = !DILocation(line: 1006, column: 16, scope: !1836)
!1838 = !DILocation(line: 1006, column: 14, scope: !1836)
!1839 = !DILocation(line: 1006, column: 9, scope: !1836)
!1840 = !DILocation(line: 1007, column: 11, scope: !1836)
!1841 = !DILocation(line: 1007, column: 17, scope: !1836)
!1842 = !DILocation(line: 1007, column: 15, scope: !1836)
!1843 = !DILocation(line: 1007, column: 9, scope: !1836)
!1844 = !DILocation(line: 1008, column: 11, scope: !1836)
!1845 = !DILocation(line: 1008, column: 16, scope: !1836)
!1846 = !DILocation(line: 1008, column: 14, scope: !1836)
!1847 = !DILocation(line: 1008, column: 9, scope: !1836)
!1848 = !DILocation(line: 1009, column: 11, scope: !1836)
!1849 = !DILocation(line: 1009, column: 17, scope: !1836)
!1850 = !DILocation(line: 1009, column: 15, scope: !1836)
!1851 = !DILocation(line: 1009, column: 9, scope: !1836)
!1852 = !DILocation(line: 1011, column: 16, scope: !1836)
!1853 = !DILocation(line: 1011, column: 25, scope: !1836)
!1854 = !DILocation(line: 1011, column: 28, scope: !1836)
!1855 = !DILocation(line: 1011, column: 27, scope: !1836)
!1856 = !DILocation(line: 1011, column: 32, scope: !1836)
!1857 = !DILocation(line: 1011, column: 14, scope: !1836)
!1858 = !DILocation(line: 1012, column: 16, scope: !1836)
!1859 = !DILocation(line: 1012, column: 19, scope: !1836)
!1860 = !DILocation(line: 1012, column: 28, scope: !1836)
!1861 = !DILocation(line: 1012, column: 31, scope: !1836)
!1862 = !DILocation(line: 1012, column: 30, scope: !1836)
!1863 = !DILocation(line: 1012, column: 35, scope: !1836)
!1864 = !DILocation(line: 1012, column: 18, scope: !1836)
!1865 = !DILocation(line: 1012, column: 14, scope: !1836)
!1866 = !DILocation(line: 1015, column: 16, scope: !1836)
!1867 = !DILocation(line: 1015, column: 21, scope: !1836)
!1868 = !DILocation(line: 1015, column: 26, scope: !1836)
!1869 = !DILocation(line: 1015, column: 30, scope: !1836)
!1870 = !DILocation(line: 1015, column: 29, scope: !1836)
!1871 = !DILocation(line: 1015, column: 33, scope: !1836)
!1872 = !DILocation(line: 1015, column: 23, scope: !1836)
!1873 = !DILocation(line: 1015, column: 39, scope: !1836)
!1874 = !DILocation(line: 1015, column: 40, scope: !1836)
!1875 = !DILocation(line: 1015, column: 43, scope: !1836)
!1876 = !DILocation(line: 1015, column: 37, scope: !1836)
!1877 = !DILocation(line: 1015, column: 48, scope: !1836)
!1878 = !DILocation(line: 1015, column: 14, scope: !1836)
!1879 = !DILocation(line: 1016, column: 16, scope: !1836)
!1880 = !DILocation(line: 1016, column: 21, scope: !1836)
!1881 = !DILocation(line: 1016, column: 26, scope: !1836)
!1882 = !DILocation(line: 1016, column: 30, scope: !1836)
!1883 = !DILocation(line: 1016, column: 29, scope: !1836)
!1884 = !DILocation(line: 1016, column: 33, scope: !1836)
!1885 = !DILocation(line: 1016, column: 23, scope: !1836)
!1886 = !DILocation(line: 1016, column: 39, scope: !1836)
!1887 = !DILocation(line: 1016, column: 40, scope: !1836)
!1888 = !DILocation(line: 1016, column: 43, scope: !1836)
!1889 = !DILocation(line: 1016, column: 37, scope: !1836)
!1890 = !DILocation(line: 1016, column: 48, scope: !1836)
!1891 = !DILocation(line: 1016, column: 14, scope: !1836)
!1892 = !DILocation(line: 1019, column: 16, scope: !1836)
!1893 = !DILocation(line: 1019, column: 21, scope: !1836)
!1894 = !DILocation(line: 1019, column: 26, scope: !1836)
!1895 = !DILocation(line: 1019, column: 30, scope: !1836)
!1896 = !DILocation(line: 1019, column: 29, scope: !1836)
!1897 = !DILocation(line: 1019, column: 33, scope: !1836)
!1898 = !DILocation(line: 1019, column: 23, scope: !1836)
!1899 = !DILocation(line: 1019, column: 39, scope: !1836)
!1900 = !DILocation(line: 1019, column: 40, scope: !1836)
!1901 = !DILocation(line: 1019, column: 43, scope: !1836)
!1902 = !DILocation(line: 1019, column: 37, scope: !1836)
!1903 = !DILocation(line: 1019, column: 48, scope: !1836)
!1904 = !DILocation(line: 1019, column: 14, scope: !1836)
!1905 = !DILocation(line: 1020, column: 16, scope: !1836)
!1906 = !DILocation(line: 1020, column: 21, scope: !1836)
!1907 = !DILocation(line: 1020, column: 26, scope: !1836)
!1908 = !DILocation(line: 1020, column: 30, scope: !1836)
!1909 = !DILocation(line: 1020, column: 29, scope: !1836)
!1910 = !DILocation(line: 1020, column: 33, scope: !1836)
!1911 = !DILocation(line: 1020, column: 23, scope: !1836)
!1912 = !DILocation(line: 1020, column: 39, scope: !1836)
!1913 = !DILocation(line: 1020, column: 40, scope: !1836)
!1914 = !DILocation(line: 1020, column: 43, scope: !1836)
!1915 = !DILocation(line: 1020, column: 37, scope: !1836)
!1916 = !DILocation(line: 1020, column: 48, scope: !1836)
!1917 = !DILocation(line: 1020, column: 14, scope: !1836)
!1918 = !DILocation(line: 1023, column: 44, scope: !1836)
!1919 = !DILocation(line: 1023, column: 55, scope: !1836)
!1920 = !DILocation(line: 1023, column: 53, scope: !1836)
!1921 = !DILocation(line: 1023, column: 5, scope: !1836)
!1922 = !DILocation(line: 1023, column: 10, scope: !1836)
!1923 = !DILocation(line: 1023, column: 15, scope: !1836)
!1924 = !DILocation(line: 1023, column: 19, scope: !1836)
!1925 = !DILocation(line: 1023, column: 18, scope: !1836)
!1926 = !DILocation(line: 1023, column: 22, scope: !1836)
!1927 = !DILocation(line: 1023, column: 12, scope: !1836)
!1928 = !DILocation(line: 1023, column: 28, scope: !1836)
!1929 = !DILocation(line: 1023, column: 29, scope: !1836)
!1930 = !DILocation(line: 1023, column: 32, scope: !1836)
!1931 = !DILocation(line: 1023, column: 26, scope: !1836)
!1932 = !DILocation(line: 1023, column: 37, scope: !1836)
!1933 = !DILocation(line: 1023, column: 42, scope: !1836)
!1934 = !DILocation(line: 1024, column: 44, scope: !1836)
!1935 = !DILocation(line: 1024, column: 55, scope: !1836)
!1936 = !DILocation(line: 1024, column: 53, scope: !1836)
!1937 = !DILocation(line: 1024, column: 5, scope: !1836)
!1938 = !DILocation(line: 1024, column: 10, scope: !1836)
!1939 = !DILocation(line: 1024, column: 15, scope: !1836)
!1940 = !DILocation(line: 1024, column: 19, scope: !1836)
!1941 = !DILocation(line: 1024, column: 18, scope: !1836)
!1942 = !DILocation(line: 1024, column: 22, scope: !1836)
!1943 = !DILocation(line: 1024, column: 12, scope: !1836)
!1944 = !DILocation(line: 1024, column: 28, scope: !1836)
!1945 = !DILocation(line: 1024, column: 29, scope: !1836)
!1946 = !DILocation(line: 1024, column: 32, scope: !1836)
!1947 = !DILocation(line: 1024, column: 26, scope: !1836)
!1948 = !DILocation(line: 1024, column: 37, scope: !1836)
!1949 = !DILocation(line: 1024, column: 42, scope: !1836)
!1950 = !DILocation(line: 1026, column: 17, scope: !1836)
!1951 = !DILocation(line: 1026, column: 28, scope: !1836)
!1952 = !DILocation(line: 1026, column: 26, scope: !1836)
!1953 = !DILocation(line: 1026, column: 15, scope: !1836)
!1954 = !DILocation(line: 1027, column: 17, scope: !1836)
!1955 = !DILocation(line: 1027, column: 28, scope: !1836)
!1956 = !DILocation(line: 1027, column: 26, scope: !1836)
!1957 = !DILocation(line: 1027, column: 15, scope: !1836)
!1958 = !DILocation(line: 1030, column: 45, scope: !1836)
!1959 = !DILocation(line: 1030, column: 56, scope: !1836)
!1960 = !DILocation(line: 1030, column: 54, scope: !1836)
!1961 = !DILocation(line: 1030, column: 70, scope: !1836)
!1962 = !DILocation(line: 1030, column: 81, scope: !1836)
!1963 = !DILocation(line: 1030, column: 79, scope: !1836)
!1964 = !DILocation(line: 1030, column: 67, scope: !1836)
!1965 = !DILocation(line: 1030, column: 5, scope: !1836)
!1966 = !DILocation(line: 1030, column: 10, scope: !1836)
!1967 = !DILocation(line: 1030, column: 15, scope: !1836)
!1968 = !DILocation(line: 1030, column: 19, scope: !1836)
!1969 = !DILocation(line: 1030, column: 18, scope: !1836)
!1970 = !DILocation(line: 1030, column: 22, scope: !1836)
!1971 = !DILocation(line: 1030, column: 12, scope: !1836)
!1972 = !DILocation(line: 1030, column: 28, scope: !1836)
!1973 = !DILocation(line: 1030, column: 29, scope: !1836)
!1974 = !DILocation(line: 1030, column: 32, scope: !1836)
!1975 = !DILocation(line: 1030, column: 26, scope: !1836)
!1976 = !DILocation(line: 1030, column: 37, scope: !1836)
!1977 = !DILocation(line: 1030, column: 42, scope: !1836)
!1978 = !DILocation(line: 1031, column: 45, scope: !1836)
!1979 = !DILocation(line: 1031, column: 56, scope: !1836)
!1980 = !DILocation(line: 1031, column: 54, scope: !1836)
!1981 = !DILocation(line: 1031, column: 70, scope: !1836)
!1982 = !DILocation(line: 1031, column: 81, scope: !1836)
!1983 = !DILocation(line: 1031, column: 79, scope: !1836)
!1984 = !DILocation(line: 1031, column: 67, scope: !1836)
!1985 = !DILocation(line: 1031, column: 5, scope: !1836)
!1986 = !DILocation(line: 1031, column: 10, scope: !1836)
!1987 = !DILocation(line: 1031, column: 15, scope: !1836)
!1988 = !DILocation(line: 1031, column: 19, scope: !1836)
!1989 = !DILocation(line: 1031, column: 18, scope: !1836)
!1990 = !DILocation(line: 1031, column: 22, scope: !1836)
!1991 = !DILocation(line: 1031, column: 12, scope: !1836)
!1992 = !DILocation(line: 1031, column: 28, scope: !1836)
!1993 = !DILocation(line: 1031, column: 29, scope: !1836)
!1994 = !DILocation(line: 1031, column: 32, scope: !1836)
!1995 = !DILocation(line: 1031, column: 26, scope: !1836)
!1996 = !DILocation(line: 1031, column: 37, scope: !1836)
!1997 = !DILocation(line: 1031, column: 42, scope: !1836)
!1998 = !DILocation(line: 1033, column: 4, scope: !1836)
!1999 = !DILocation(line: 1005, column: 26, scope: !1830)
!2000 = !DILocation(line: 1005, column: 4, scope: !1830)
!2001 = distinct !{!2001, !1834, !2002}
!2002 = !DILocation(line: 1033, column: 4, scope: !1826)
!2003 = !DILocation(line: 1034, column: 3, scope: !1827)
!2004 = !DILocation(line: 1004, column: 25, scope: !1820)
!2005 = !DILocation(line: 1004, column: 3, scope: !1820)
!2006 = distinct !{!2006, !1824, !2007}
!2007 = !DILocation(line: 1034, column: 3, scope: !1817)
!2008 = !DILocation(line: 1035, column: 6, scope: !2009)
!2009 = distinct !DILexicalBlock(scope: !1801, file: !6, line: 1035, column: 6)
!2010 = !DILocation(line: 1035, column: 9, scope: !2009)
!2011 = !DILocation(line: 1035, column: 7, scope: !2009)
!2012 = !DILocation(line: 1035, column: 6, scope: !1801)
!2013 = !DILocation(line: 1036, column: 10, scope: !2014)
!2014 = distinct !DILexicalBlock(scope: !2015, file: !6, line: 1036, column: 4)
!2015 = distinct !DILexicalBlock(scope: !2009, file: !6, line: 1035, column: 15)
!2016 = !DILocation(line: 1036, column: 8, scope: !2014)
!2017 = !DILocation(line: 1036, column: 14, scope: !2018)
!2018 = distinct !DILexicalBlock(scope: !2014, file: !6, line: 1036, column: 4)
!2019 = !DILocation(line: 1036, column: 16, scope: !2018)
!2020 = !DILocation(line: 1036, column: 4, scope: !2014)
!2021 = !DILocation(line: 1038, column: 38, scope: !2022)
!2022 = distinct !DILexicalBlock(scope: !2018, file: !6, line: 1036, column: 26)
!2023 = !DILocation(line: 1038, column: 43, scope: !2022)
!2024 = !DILocation(line: 1038, column: 47, scope: !2022)
!2025 = !DILocation(line: 1038, column: 49, scope: !2022)
!2026 = !DILocation(line: 1038, column: 45, scope: !2022)
!2027 = !DILocation(line: 1038, column: 55, scope: !2022)
!2028 = !DILocation(line: 1038, column: 56, scope: !2022)
!2029 = !DILocation(line: 1038, column: 59, scope: !2022)
!2030 = !DILocation(line: 1038, column: 53, scope: !2022)
!2031 = !DILocation(line: 1038, column: 64, scope: !2022)
!2032 = !DILocation(line: 1038, column: 5, scope: !2022)
!2033 = !DILocation(line: 1038, column: 10, scope: !2022)
!2034 = !DILocation(line: 1038, column: 14, scope: !2022)
!2035 = !DILocation(line: 1038, column: 16, scope: !2022)
!2036 = !DILocation(line: 1038, column: 12, scope: !2022)
!2037 = !DILocation(line: 1038, column: 22, scope: !2022)
!2038 = !DILocation(line: 1038, column: 23, scope: !2022)
!2039 = !DILocation(line: 1038, column: 26, scope: !2022)
!2040 = !DILocation(line: 1038, column: 20, scope: !2022)
!2041 = !DILocation(line: 1038, column: 31, scope: !2022)
!2042 = !DILocation(line: 1038, column: 36, scope: !2022)
!2043 = !DILocation(line: 1039, column: 38, scope: !2022)
!2044 = !DILocation(line: 1039, column: 43, scope: !2022)
!2045 = !DILocation(line: 1039, column: 47, scope: !2022)
!2046 = !DILocation(line: 1039, column: 49, scope: !2022)
!2047 = !DILocation(line: 1039, column: 45, scope: !2022)
!2048 = !DILocation(line: 1039, column: 55, scope: !2022)
!2049 = !DILocation(line: 1039, column: 56, scope: !2022)
!2050 = !DILocation(line: 1039, column: 59, scope: !2022)
!2051 = !DILocation(line: 1039, column: 53, scope: !2022)
!2052 = !DILocation(line: 1039, column: 64, scope: !2022)
!2053 = !DILocation(line: 1039, column: 5, scope: !2022)
!2054 = !DILocation(line: 1039, column: 10, scope: !2022)
!2055 = !DILocation(line: 1039, column: 14, scope: !2022)
!2056 = !DILocation(line: 1039, column: 16, scope: !2022)
!2057 = !DILocation(line: 1039, column: 12, scope: !2022)
!2058 = !DILocation(line: 1039, column: 22, scope: !2022)
!2059 = !DILocation(line: 1039, column: 23, scope: !2022)
!2060 = !DILocation(line: 1039, column: 26, scope: !2022)
!2061 = !DILocation(line: 1039, column: 20, scope: !2022)
!2062 = !DILocation(line: 1039, column: 31, scope: !2022)
!2063 = !DILocation(line: 1039, column: 36, scope: !2022)
!2064 = !DILocation(line: 1040, column: 4, scope: !2022)
!2065 = !DILocation(line: 1036, column: 23, scope: !2018)
!2066 = !DILocation(line: 1036, column: 4, scope: !2018)
!2067 = distinct !{!2067, !2020, !2068}
!2068 = !DILocation(line: 1040, column: 4, scope: !2014)
!2069 = !DILocation(line: 1041, column: 3, scope: !2015)
!2070 = !DILocation(line: 1043, column: 7, scope: !2071)
!2071 = distinct !DILexicalBlock(scope: !2009, file: !6, line: 1042, column: 7)
!2072 = !DILocation(line: 1044, column: 15, scope: !2071)
!2073 = !DILocation(line: 1044, column: 16, scope: !2071)
!2074 = !DILocation(line: 1044, column: 19, scope: !2071)
!2075 = !DILocation(line: 1044, column: 11, scope: !2071)
!2076 = !DILocation(line: 1044, column: 7, scope: !2071)
!2077 = !DILocation(line: 1045, column: 15, scope: !2071)
!2078 = !DILocation(line: 1045, column: 24, scope: !2071)
!2079 = !DILocation(line: 1045, column: 25, scope: !2071)
!2080 = !DILocation(line: 1045, column: 21, scope: !2071)
!2081 = !DILocation(line: 1045, column: 11, scope: !2071)
!2082 = !DILocation(line: 1045, column: 7, scope: !2071)
!2083 = !DILocation(line: 1046, column: 13, scope: !2071)
!2084 = !DILocation(line: 1046, column: 11, scope: !2071)
!2085 = !DILocation(line: 1046, column: 7, scope: !2071)
!2086 = !DILocation(line: 1047, column: 9, scope: !2071)
!2087 = !DILocation(line: 1047, column: 7, scope: !2071)
!2088 = !DILocation(line: 1048, column: 10, scope: !2089)
!2089 = distinct !DILexicalBlock(scope: !2071, file: !6, line: 1048, column: 4)
!2090 = !DILocation(line: 1048, column: 8, scope: !2089)
!2091 = !DILocation(line: 1048, column: 14, scope: !2092)
!2092 = distinct !DILexicalBlock(scope: !2089, file: !6, line: 1048, column: 4)
!2093 = !DILocation(line: 1048, column: 18, scope: !2092)
!2094 = !DILocation(line: 1048, column: 20, scope: !2092)
!2095 = !DILocation(line: 1048, column: 16, scope: !2092)
!2096 = !DILocation(line: 1048, column: 4, scope: !2089)
!2097 = !DILocation(line: 1049, column: 11, scope: !2098)
!2098 = distinct !DILexicalBlock(scope: !2099, file: !6, line: 1049, column: 5)
!2099 = distinct !DILexicalBlock(scope: !2092, file: !6, line: 1048, column: 29)
!2100 = !DILocation(line: 1049, column: 9, scope: !2098)
!2101 = !DILocation(line: 1049, column: 15, scope: !2102)
!2102 = distinct !DILexicalBlock(scope: !2098, file: !6, line: 1049, column: 5)
!2103 = !DILocation(line: 1049, column: 19, scope: !2102)
!2104 = !DILocation(line: 1049, column: 21, scope: !2102)
!2105 = !DILocation(line: 1049, column: 17, scope: !2102)
!2106 = !DILocation(line: 1049, column: 5, scope: !2098)
!2107 = !DILocation(line: 1050, column: 12, scope: !2108)
!2108 = distinct !DILexicalBlock(scope: !2102, file: !6, line: 1049, column: 30)
!2109 = !DILocation(line: 1050, column: 17, scope: !2108)
!2110 = !DILocation(line: 1050, column: 15, scope: !2108)
!2111 = !DILocation(line: 1050, column: 10, scope: !2108)
!2112 = !DILocation(line: 1051, column: 12, scope: !2108)
!2113 = !DILocation(line: 1051, column: 18, scope: !2108)
!2114 = !DILocation(line: 1051, column: 16, scope: !2108)
!2115 = !DILocation(line: 1051, column: 10, scope: !2108)
!2116 = !DILocation(line: 1052, column: 12, scope: !2108)
!2117 = !DILocation(line: 1052, column: 17, scope: !2108)
!2118 = !DILocation(line: 1052, column: 15, scope: !2108)
!2119 = !DILocation(line: 1052, column: 10, scope: !2108)
!2120 = !DILocation(line: 1053, column: 12, scope: !2108)
!2121 = !DILocation(line: 1053, column: 18, scope: !2108)
!2122 = !DILocation(line: 1053, column: 16, scope: !2108)
!2123 = !DILocation(line: 1053, column: 10, scope: !2108)
!2124 = !DILocation(line: 1055, column: 17, scope: !2108)
!2125 = !DILocation(line: 1055, column: 26, scope: !2108)
!2126 = !DILocation(line: 1055, column: 29, scope: !2108)
!2127 = !DILocation(line: 1055, column: 28, scope: !2108)
!2128 = !DILocation(line: 1055, column: 33, scope: !2108)
!2129 = !DILocation(line: 1055, column: 15, scope: !2108)
!2130 = !DILocation(line: 1056, column: 17, scope: !2108)
!2131 = !DILocation(line: 1056, column: 20, scope: !2108)
!2132 = !DILocation(line: 1056, column: 29, scope: !2108)
!2133 = !DILocation(line: 1056, column: 32, scope: !2108)
!2134 = !DILocation(line: 1056, column: 31, scope: !2108)
!2135 = !DILocation(line: 1056, column: 36, scope: !2108)
!2136 = !DILocation(line: 1056, column: 19, scope: !2108)
!2137 = !DILocation(line: 1056, column: 15, scope: !2108)
!2138 = !DILocation(line: 1059, column: 17, scope: !2108)
!2139 = !DILocation(line: 1059, column: 22, scope: !2108)
!2140 = !DILocation(line: 1059, column: 27, scope: !2108)
!2141 = !DILocation(line: 1059, column: 31, scope: !2108)
!2142 = !DILocation(line: 1059, column: 30, scope: !2108)
!2143 = !DILocation(line: 1059, column: 34, scope: !2108)
!2144 = !DILocation(line: 1059, column: 24, scope: !2108)
!2145 = !DILocation(line: 1059, column: 40, scope: !2108)
!2146 = !DILocation(line: 1059, column: 41, scope: !2108)
!2147 = !DILocation(line: 1059, column: 44, scope: !2108)
!2148 = !DILocation(line: 1059, column: 38, scope: !2108)
!2149 = !DILocation(line: 1059, column: 49, scope: !2108)
!2150 = !DILocation(line: 1059, column: 15, scope: !2108)
!2151 = !DILocation(line: 1060, column: 17, scope: !2108)
!2152 = !DILocation(line: 1060, column: 22, scope: !2108)
!2153 = !DILocation(line: 1060, column: 27, scope: !2108)
!2154 = !DILocation(line: 1060, column: 31, scope: !2108)
!2155 = !DILocation(line: 1060, column: 30, scope: !2108)
!2156 = !DILocation(line: 1060, column: 34, scope: !2108)
!2157 = !DILocation(line: 1060, column: 24, scope: !2108)
!2158 = !DILocation(line: 1060, column: 40, scope: !2108)
!2159 = !DILocation(line: 1060, column: 41, scope: !2108)
!2160 = !DILocation(line: 1060, column: 44, scope: !2108)
!2161 = !DILocation(line: 1060, column: 38, scope: !2108)
!2162 = !DILocation(line: 1060, column: 49, scope: !2108)
!2163 = !DILocation(line: 1060, column: 15, scope: !2108)
!2164 = !DILocation(line: 1063, column: 17, scope: !2108)
!2165 = !DILocation(line: 1063, column: 22, scope: !2108)
!2166 = !DILocation(line: 1063, column: 27, scope: !2108)
!2167 = !DILocation(line: 1063, column: 31, scope: !2108)
!2168 = !DILocation(line: 1063, column: 30, scope: !2108)
!2169 = !DILocation(line: 1063, column: 34, scope: !2108)
!2170 = !DILocation(line: 1063, column: 24, scope: !2108)
!2171 = !DILocation(line: 1063, column: 40, scope: !2108)
!2172 = !DILocation(line: 1063, column: 41, scope: !2108)
!2173 = !DILocation(line: 1063, column: 44, scope: !2108)
!2174 = !DILocation(line: 1063, column: 38, scope: !2108)
!2175 = !DILocation(line: 1063, column: 49, scope: !2108)
!2176 = !DILocation(line: 1063, column: 15, scope: !2108)
!2177 = !DILocation(line: 1064, column: 17, scope: !2108)
!2178 = !DILocation(line: 1064, column: 22, scope: !2108)
!2179 = !DILocation(line: 1064, column: 27, scope: !2108)
!2180 = !DILocation(line: 1064, column: 31, scope: !2108)
!2181 = !DILocation(line: 1064, column: 30, scope: !2108)
!2182 = !DILocation(line: 1064, column: 34, scope: !2108)
!2183 = !DILocation(line: 1064, column: 24, scope: !2108)
!2184 = !DILocation(line: 1064, column: 40, scope: !2108)
!2185 = !DILocation(line: 1064, column: 41, scope: !2108)
!2186 = !DILocation(line: 1064, column: 44, scope: !2108)
!2187 = !DILocation(line: 1064, column: 38, scope: !2108)
!2188 = !DILocation(line: 1064, column: 49, scope: !2108)
!2189 = !DILocation(line: 1064, column: 15, scope: !2108)
!2190 = !DILocation(line: 1067, column: 45, scope: !2108)
!2191 = !DILocation(line: 1067, column: 56, scope: !2108)
!2192 = !DILocation(line: 1067, column: 54, scope: !2108)
!2193 = !DILocation(line: 1067, column: 6, scope: !2108)
!2194 = !DILocation(line: 1067, column: 11, scope: !2108)
!2195 = !DILocation(line: 1067, column: 16, scope: !2108)
!2196 = !DILocation(line: 1067, column: 20, scope: !2108)
!2197 = !DILocation(line: 1067, column: 19, scope: !2108)
!2198 = !DILocation(line: 1067, column: 23, scope: !2108)
!2199 = !DILocation(line: 1067, column: 13, scope: !2108)
!2200 = !DILocation(line: 1067, column: 29, scope: !2108)
!2201 = !DILocation(line: 1067, column: 30, scope: !2108)
!2202 = !DILocation(line: 1067, column: 33, scope: !2108)
!2203 = !DILocation(line: 1067, column: 27, scope: !2108)
!2204 = !DILocation(line: 1067, column: 38, scope: !2108)
!2205 = !DILocation(line: 1067, column: 43, scope: !2108)
!2206 = !DILocation(line: 1068, column: 45, scope: !2108)
!2207 = !DILocation(line: 1068, column: 56, scope: !2108)
!2208 = !DILocation(line: 1068, column: 54, scope: !2108)
!2209 = !DILocation(line: 1068, column: 6, scope: !2108)
!2210 = !DILocation(line: 1068, column: 11, scope: !2108)
!2211 = !DILocation(line: 1068, column: 16, scope: !2108)
!2212 = !DILocation(line: 1068, column: 20, scope: !2108)
!2213 = !DILocation(line: 1068, column: 19, scope: !2108)
!2214 = !DILocation(line: 1068, column: 23, scope: !2108)
!2215 = !DILocation(line: 1068, column: 13, scope: !2108)
!2216 = !DILocation(line: 1068, column: 29, scope: !2108)
!2217 = !DILocation(line: 1068, column: 30, scope: !2108)
!2218 = !DILocation(line: 1068, column: 33, scope: !2108)
!2219 = !DILocation(line: 1068, column: 27, scope: !2108)
!2220 = !DILocation(line: 1068, column: 38, scope: !2108)
!2221 = !DILocation(line: 1068, column: 43, scope: !2108)
!2222 = !DILocation(line: 1070, column: 19, scope: !2108)
!2223 = !DILocation(line: 1070, column: 30, scope: !2108)
!2224 = !DILocation(line: 1070, column: 28, scope: !2108)
!2225 = !DILocation(line: 1070, column: 17, scope: !2108)
!2226 = !DILocation(line: 1071, column: 19, scope: !2108)
!2227 = !DILocation(line: 1071, column: 30, scope: !2108)
!2228 = !DILocation(line: 1071, column: 28, scope: !2108)
!2229 = !DILocation(line: 1071, column: 17, scope: !2108)
!2230 = !DILocation(line: 1074, column: 46, scope: !2108)
!2231 = !DILocation(line: 1074, column: 57, scope: !2108)
!2232 = !DILocation(line: 1074, column: 55, scope: !2108)
!2233 = !DILocation(line: 1074, column: 72, scope: !2108)
!2234 = !DILocation(line: 1074, column: 83, scope: !2108)
!2235 = !DILocation(line: 1074, column: 81, scope: !2108)
!2236 = !DILocation(line: 1074, column: 69, scope: !2108)
!2237 = !DILocation(line: 1074, column: 6, scope: !2108)
!2238 = !DILocation(line: 1074, column: 11, scope: !2108)
!2239 = !DILocation(line: 1074, column: 16, scope: !2108)
!2240 = !DILocation(line: 1074, column: 20, scope: !2108)
!2241 = !DILocation(line: 1074, column: 19, scope: !2108)
!2242 = !DILocation(line: 1074, column: 23, scope: !2108)
!2243 = !DILocation(line: 1074, column: 13, scope: !2108)
!2244 = !DILocation(line: 1074, column: 29, scope: !2108)
!2245 = !DILocation(line: 1074, column: 30, scope: !2108)
!2246 = !DILocation(line: 1074, column: 33, scope: !2108)
!2247 = !DILocation(line: 1074, column: 27, scope: !2108)
!2248 = !DILocation(line: 1074, column: 38, scope: !2108)
!2249 = !DILocation(line: 1074, column: 43, scope: !2108)
!2250 = !DILocation(line: 1075, column: 46, scope: !2108)
!2251 = !DILocation(line: 1075, column: 57, scope: !2108)
!2252 = !DILocation(line: 1075, column: 55, scope: !2108)
!2253 = !DILocation(line: 1075, column: 72, scope: !2108)
!2254 = !DILocation(line: 1075, column: 83, scope: !2108)
!2255 = !DILocation(line: 1075, column: 81, scope: !2108)
!2256 = !DILocation(line: 1075, column: 69, scope: !2108)
!2257 = !DILocation(line: 1075, column: 6, scope: !2108)
!2258 = !DILocation(line: 1075, column: 11, scope: !2108)
!2259 = !DILocation(line: 1075, column: 16, scope: !2108)
!2260 = !DILocation(line: 1075, column: 20, scope: !2108)
!2261 = !DILocation(line: 1075, column: 19, scope: !2108)
!2262 = !DILocation(line: 1075, column: 23, scope: !2108)
!2263 = !DILocation(line: 1075, column: 13, scope: !2108)
!2264 = !DILocation(line: 1075, column: 29, scope: !2108)
!2265 = !DILocation(line: 1075, column: 30, scope: !2108)
!2266 = !DILocation(line: 1075, column: 33, scope: !2108)
!2267 = !DILocation(line: 1075, column: 27, scope: !2108)
!2268 = !DILocation(line: 1075, column: 38, scope: !2108)
!2269 = !DILocation(line: 1075, column: 43, scope: !2108)
!2270 = !DILocation(line: 1076, column: 5, scope: !2108)
!2271 = !DILocation(line: 1049, column: 27, scope: !2102)
!2272 = !DILocation(line: 1049, column: 5, scope: !2102)
!2273 = distinct !{!2273, !2106, !2274}
!2274 = !DILocation(line: 1076, column: 5, scope: !2098)
!2275 = !DILocation(line: 1077, column: 4, scope: !2099)
!2276 = !DILocation(line: 1048, column: 26, scope: !2092)
!2277 = !DILocation(line: 1048, column: 4, scope: !2092)
!2278 = distinct !{!2278, !2096, !2279}
!2279 = !DILocation(line: 1077, column: 4, scope: !2089)
!2280 = !DILocation(line: 1079, column: 2, scope: !1801)
!2281 = !DILocation(line: 998, column: 22, scope: !1796)
!2282 = !DILocation(line: 998, column: 2, scope: !1796)
!2283 = distinct !{!2283, !1799, !2284}
!2284 = !DILocation(line: 1079, column: 2, scope: !1793)
!2285 = !DILocation(line: 1080, column: 1, scope: !1695)
!2286 = distinct !DISubprogram(name: "cffts2_gpu_kernel_3", linkageName: "_Z19cffts2_gpu_kernel_3P8dcomplexS0_", scope: !6, file: !6, line: 1089, type: !803, scopeLine: 1090, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !7)
!2287 = !DILocalVariable(name: "x_out", arg: 1, scope: !2286, file: !6, line: 1089, type: !9)
!2288 = !DILocation(line: 1089, column: 46, scope: !2286)
!2289 = !DILocalVariable(name: "y0", arg: 2, scope: !2286, file: !6, line: 1090, type: !9)
!2290 = !DILocation(line: 1090, column: 12, scope: !2286)
!2291 = !DILocalVariable(name: "x_y_z", scope: !2286, file: !6, line: 1091, type: !18)
!2292 = !DILocation(line: 1091, column: 6, scope: !2286)
!2293 = !DILocation(line: 64, column: 3, scope: !812, inlinedAt: !2294)
!2294 = distinct !DILocation(line: 1091, column: 14, scope: !2286)
!2295 = !DILocation(line: 75, column: 3, scope: !850, inlinedAt: !2296)
!2296 = distinct !DILocation(line: 1091, column: 27, scope: !2286)
!2297 = !DILocation(line: 1091, column: 25, scope: !2286)
!2298 = !DILocation(line: 53, column: 3, scope: !896, inlinedAt: !2299)
!2299 = distinct !DILocation(line: 1091, column: 40, scope: !2286)
!2300 = !DILocation(line: 1091, column: 38, scope: !2286)
!2301 = !DILocation(line: 1092, column: 5, scope: !2302)
!2302 = distinct !DILexicalBlock(scope: !2286, file: !6, line: 1092, column: 5)
!2303 = !DILocation(line: 1092, column: 11, scope: !2302)
!2304 = !DILocation(line: 1092, column: 5, scope: !2286)
!2305 = !DILocation(line: 1093, column: 3, scope: !2306)
!2306 = distinct !DILexicalBlock(scope: !2302, file: !6, line: 1092, column: 25)
!2307 = !DILocation(line: 1095, column: 22, scope: !2286)
!2308 = !DILocation(line: 1095, column: 25, scope: !2286)
!2309 = !DILocation(line: 1095, column: 32, scope: !2286)
!2310 = !DILocation(line: 1095, column: 2, scope: !2286)
!2311 = !DILocation(line: 1095, column: 8, scope: !2286)
!2312 = !DILocation(line: 1095, column: 15, scope: !2286)
!2313 = !DILocation(line: 1095, column: 20, scope: !2286)
!2314 = !DILocation(line: 1096, column: 22, scope: !2286)
!2315 = !DILocation(line: 1096, column: 25, scope: !2286)
!2316 = !DILocation(line: 1096, column: 32, scope: !2286)
!2317 = !DILocation(line: 1096, column: 2, scope: !2286)
!2318 = !DILocation(line: 1096, column: 8, scope: !2286)
!2319 = !DILocation(line: 1096, column: 15, scope: !2286)
!2320 = !DILocation(line: 1096, column: 20, scope: !2286)
!2321 = !DILocation(line: 1097, column: 1, scope: !2286)
!2322 = distinct !DISubprogram(name: "cffts3_gpu_cfftz_device", linkageName: "_Z23cffts3_gpu_cfftz_deviceiiiP8dcomplexS0_S0_ii", scope: !6, file: !6, line: 1150, type: !2323, scopeLine: 1157, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !7)
!2323 = !DISubroutineType(types: !2324)
!2324 = !{null, !976, !18, !18, !9, !9, !9, !18, !18}
!2325 = !DILocalVariable(name: "is", arg: 1, scope: !2322, file: !6, line: 1150, type: !976)
!2326 = !DILocation(line: 1150, column: 51, scope: !2322)
!2327 = !DILocalVariable(name: "m", arg: 2, scope: !2322, file: !6, line: 1151, type: !18)
!2328 = !DILocation(line: 1151, column: 7, scope: !2322)
!2329 = !DILocalVariable(name: "n", arg: 3, scope: !2322, file: !6, line: 1152, type: !18)
!2330 = !DILocation(line: 1152, column: 7, scope: !2322)
!2331 = !DILocalVariable(name: "x", arg: 4, scope: !2322, file: !6, line: 1153, type: !9)
!2332 = !DILocation(line: 1153, column: 12, scope: !2322)
!2333 = !DILocalVariable(name: "y", arg: 5, scope: !2322, file: !6, line: 1154, type: !9)
!2334 = !DILocation(line: 1154, column: 12, scope: !2322)
!2335 = !DILocalVariable(name: "u_device", arg: 6, scope: !2322, file: !6, line: 1155, type: !9)
!2336 = !DILocation(line: 1155, column: 12, scope: !2322)
!2337 = !DILocalVariable(name: "index_arg", arg: 7, scope: !2322, file: !6, line: 1156, type: !18)
!2338 = !DILocation(line: 1156, column: 7, scope: !2322)
!2339 = !DILocalVariable(name: "size_arg", arg: 8, scope: !2322, file: !6, line: 1157, type: !18)
!2340 = !DILocation(line: 1157, column: 7, scope: !2322)
!2341 = !DILocalVariable(name: "j", scope: !2322, file: !6, line: 1158, type: !18)
!2342 = !DILocation(line: 1158, column: 6, scope: !2322)
!2343 = !DILocalVariable(name: "l", scope: !2322, file: !6, line: 1158, type: !18)
!2344 = !DILocation(line: 1158, column: 8, scope: !2322)
!2345 = !DILocation(line: 1164, column: 7, scope: !2346)
!2346 = distinct !DILexicalBlock(scope: !2322, file: !6, line: 1164, column: 2)
!2347 = !DILocation(line: 1164, column: 6, scope: !2346)
!2348 = !DILocation(line: 1164, column: 11, scope: !2349)
!2349 = distinct !DILexicalBlock(scope: !2346, file: !6, line: 1164, column: 2)
!2350 = !DILocation(line: 1164, column: 14, scope: !2349)
!2351 = !DILocation(line: 1164, column: 12, scope: !2349)
!2352 = !DILocation(line: 1164, column: 2, scope: !2346)
!2353 = !DILocation(line: 1165, column: 27, scope: !2354)
!2354 = distinct !DILexicalBlock(scope: !2349, file: !6, line: 1164, column: 22)
!2355 = !DILocation(line: 1165, column: 31, scope: !2354)
!2356 = !DILocation(line: 1165, column: 34, scope: !2354)
!2357 = !DILocation(line: 1165, column: 37, scope: !2354)
!2358 = !DILocation(line: 1165, column: 40, scope: !2354)
!2359 = !DILocation(line: 1165, column: 50, scope: !2354)
!2360 = !DILocation(line: 1165, column: 53, scope: !2354)
!2361 = !DILocation(line: 1165, column: 56, scope: !2354)
!2362 = !DILocation(line: 1165, column: 67, scope: !2354)
!2363 = !DILocation(line: 1165, column: 3, scope: !2354)
!2364 = !DILocation(line: 1166, column: 6, scope: !2365)
!2365 = distinct !DILexicalBlock(scope: !2354, file: !6, line: 1166, column: 6)
!2366 = !DILocation(line: 1166, column: 9, scope: !2365)
!2367 = !DILocation(line: 1166, column: 7, scope: !2365)
!2368 = !DILocation(line: 1166, column: 6, scope: !2354)
!2369 = !DILocation(line: 1166, column: 12, scope: !2370)
!2370 = distinct !DILexicalBlock(scope: !2365, file: !6, line: 1166, column: 11)
!2371 = !DILocation(line: 1167, column: 27, scope: !2354)
!2372 = !DILocation(line: 1167, column: 31, scope: !2354)
!2373 = !DILocation(line: 1167, column: 33, scope: !2354)
!2374 = !DILocation(line: 1167, column: 38, scope: !2354)
!2375 = !DILocation(line: 1167, column: 41, scope: !2354)
!2376 = !DILocation(line: 1167, column: 44, scope: !2354)
!2377 = !DILocation(line: 1167, column: 54, scope: !2354)
!2378 = !DILocation(line: 1167, column: 57, scope: !2354)
!2379 = !DILocation(line: 1167, column: 60, scope: !2354)
!2380 = !DILocation(line: 1167, column: 71, scope: !2354)
!2381 = !DILocation(line: 1167, column: 3, scope: !2354)
!2382 = !DILocation(line: 1168, column: 2, scope: !2354)
!2383 = !DILocation(line: 1164, column: 18, scope: !2349)
!2384 = !DILocation(line: 1164, column: 2, scope: !2349)
!2385 = distinct !{!2385, !2352, !2386}
!2386 = !DILocation(line: 1168, column: 2, scope: !2346)
!2387 = !DILocation(line: 1174, column: 5, scope: !2388)
!2388 = distinct !DILexicalBlock(scope: !2322, file: !6, line: 1174, column: 5)
!2389 = !DILocation(line: 1174, column: 6, scope: !2388)
!2390 = !DILocation(line: 1174, column: 8, scope: !2388)
!2391 = !DILocation(line: 1174, column: 5, scope: !2322)
!2392 = !DILocation(line: 1175, column: 8, scope: !2393)
!2393 = distinct !DILexicalBlock(scope: !2394, file: !6, line: 1175, column: 3)
!2394 = distinct !DILexicalBlock(scope: !2388, file: !6, line: 1174, column: 12)
!2395 = !DILocation(line: 1175, column: 7, scope: !2393)
!2396 = !DILocation(line: 1175, column: 12, scope: !2397)
!2397 = distinct !DILexicalBlock(scope: !2393, file: !6, line: 1175, column: 3)
!2398 = !DILocation(line: 1175, column: 14, scope: !2397)
!2399 = !DILocation(line: 1175, column: 13, scope: !2397)
!2400 = !DILocation(line: 1175, column: 3, scope: !2393)
!2401 = !DILocation(line: 1176, column: 35, scope: !2402)
!2402 = distinct !DILexicalBlock(scope: !2397, file: !6, line: 1175, column: 21)
!2403 = !DILocation(line: 1176, column: 37, scope: !2402)
!2404 = !DILocation(line: 1176, column: 39, scope: !2402)
!2405 = !DILocation(line: 1176, column: 38, scope: !2402)
!2406 = !DILocation(line: 1176, column: 48, scope: !2402)
!2407 = !DILocation(line: 1176, column: 47, scope: !2402)
!2408 = !DILocation(line: 1176, column: 59, scope: !2402)
!2409 = !DILocation(line: 1176, column: 4, scope: !2402)
!2410 = !DILocation(line: 1176, column: 6, scope: !2402)
!2411 = !DILocation(line: 1176, column: 8, scope: !2402)
!2412 = !DILocation(line: 1176, column: 7, scope: !2402)
!2413 = !DILocation(line: 1176, column: 17, scope: !2402)
!2414 = !DILocation(line: 1176, column: 16, scope: !2402)
!2415 = !DILocation(line: 1176, column: 28, scope: !2402)
!2416 = !DILocation(line: 1176, column: 33, scope: !2402)
!2417 = !DILocation(line: 1177, column: 35, scope: !2402)
!2418 = !DILocation(line: 1177, column: 37, scope: !2402)
!2419 = !DILocation(line: 1177, column: 39, scope: !2402)
!2420 = !DILocation(line: 1177, column: 38, scope: !2402)
!2421 = !DILocation(line: 1177, column: 48, scope: !2402)
!2422 = !DILocation(line: 1177, column: 47, scope: !2402)
!2423 = !DILocation(line: 1177, column: 59, scope: !2402)
!2424 = !DILocation(line: 1177, column: 4, scope: !2402)
!2425 = !DILocation(line: 1177, column: 6, scope: !2402)
!2426 = !DILocation(line: 1177, column: 8, scope: !2402)
!2427 = !DILocation(line: 1177, column: 7, scope: !2402)
!2428 = !DILocation(line: 1177, column: 17, scope: !2402)
!2429 = !DILocation(line: 1177, column: 16, scope: !2402)
!2430 = !DILocation(line: 1177, column: 28, scope: !2402)
!2431 = !DILocation(line: 1177, column: 33, scope: !2402)
!2432 = !DILocation(line: 1178, column: 3, scope: !2402)
!2433 = !DILocation(line: 1175, column: 18, scope: !2397)
!2434 = !DILocation(line: 1175, column: 3, scope: !2397)
!2435 = distinct !{!2435, !2400, !2436}
!2436 = !DILocation(line: 1178, column: 3, scope: !2393)
!2437 = !DILocation(line: 1179, column: 2, scope: !2394)
!2438 = !DILocation(line: 1180, column: 1, scope: !2322)
!2439 = distinct !DISubprogram(name: "cffts3_gpu_fftz2_device", linkageName: "_Z23cffts3_gpu_fftz2_deviceiiiiP8dcomplexS0_S0_ii", scope: !6, file: !6, line: 1191, type: !2440, scopeLine: 1199, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !7)
!2440 = !DISubroutineType(types: !2441)
!2441 = !{null, !976, !18, !18, !18, !9, !9, !9, !18, !18}
!2442 = !DILocalVariable(name: "is", arg: 1, scope: !2439, file: !6, line: 1191, type: !976)
!2443 = !DILocation(line: 1191, column: 51, scope: !2439)
!2444 = !DILocalVariable(name: "l", arg: 2, scope: !2439, file: !6, line: 1192, type: !18)
!2445 = !DILocation(line: 1192, column: 7, scope: !2439)
!2446 = !DILocalVariable(name: "m", arg: 3, scope: !2439, file: !6, line: 1193, type: !18)
!2447 = !DILocation(line: 1193, column: 7, scope: !2439)
!2448 = !DILocalVariable(name: "n", arg: 4, scope: !2439, file: !6, line: 1194, type: !18)
!2449 = !DILocation(line: 1194, column: 7, scope: !2439)
!2450 = !DILocalVariable(name: "u", arg: 5, scope: !2439, file: !6, line: 1195, type: !9)
!2451 = !DILocation(line: 1195, column: 12, scope: !2439)
!2452 = !DILocalVariable(name: "x", arg: 6, scope: !2439, file: !6, line: 1196, type: !9)
!2453 = !DILocation(line: 1196, column: 12, scope: !2439)
!2454 = !DILocalVariable(name: "y", arg: 7, scope: !2439, file: !6, line: 1197, type: !9)
!2455 = !DILocation(line: 1197, column: 12, scope: !2439)
!2456 = !DILocalVariable(name: "index_arg", arg: 8, scope: !2439, file: !6, line: 1198, type: !18)
!2457 = !DILocation(line: 1198, column: 7, scope: !2439)
!2458 = !DILocalVariable(name: "size_arg", arg: 9, scope: !2439, file: !6, line: 1199, type: !18)
!2459 = !DILocation(line: 1199, column: 7, scope: !2439)
!2460 = !DILocalVariable(name: "k", scope: !2439, file: !6, line: 1200, type: !18)
!2461 = !DILocation(line: 1200, column: 6, scope: !2439)
!2462 = !DILocalVariable(name: "n1", scope: !2439, file: !6, line: 1200, type: !18)
!2463 = !DILocation(line: 1200, column: 8, scope: !2439)
!2464 = !DILocalVariable(name: "li", scope: !2439, file: !6, line: 1200, type: !18)
!2465 = !DILocation(line: 1200, column: 11, scope: !2439)
!2466 = !DILocalVariable(name: "lj", scope: !2439, file: !6, line: 1200, type: !18)
!2467 = !DILocation(line: 1200, column: 14, scope: !2439)
!2468 = !DILocalVariable(name: "lk", scope: !2439, file: !6, line: 1200, type: !18)
!2469 = !DILocation(line: 1200, column: 17, scope: !2439)
!2470 = !DILocalVariable(name: "ku", scope: !2439, file: !6, line: 1200, type: !18)
!2471 = !DILocation(line: 1200, column: 20, scope: !2439)
!2472 = !DILocalVariable(name: "i", scope: !2439, file: !6, line: 1200, type: !18)
!2473 = !DILocation(line: 1200, column: 23, scope: !2439)
!2474 = !DILocalVariable(name: "i11", scope: !2439, file: !6, line: 1200, type: !18)
!2475 = !DILocation(line: 1200, column: 25, scope: !2439)
!2476 = !DILocalVariable(name: "i12", scope: !2439, file: !6, line: 1200, type: !18)
!2477 = !DILocation(line: 1200, column: 29, scope: !2439)
!2478 = !DILocalVariable(name: "i21", scope: !2439, file: !6, line: 1200, type: !18)
!2479 = !DILocation(line: 1200, column: 33, scope: !2439)
!2480 = !DILocalVariable(name: "i22", scope: !2439, file: !6, line: 1200, type: !18)
!2481 = !DILocation(line: 1200, column: 37, scope: !2439)
!2482 = !DILocalVariable(name: "x11real", scope: !2439, file: !6, line: 1201, type: !15)
!2483 = !DILocation(line: 1201, column: 9, scope: !2439)
!2484 = !DILocalVariable(name: "x11imag", scope: !2439, file: !6, line: 1201, type: !15)
!2485 = !DILocation(line: 1201, column: 18, scope: !2439)
!2486 = !DILocalVariable(name: "x21real", scope: !2439, file: !6, line: 1202, type: !15)
!2487 = !DILocation(line: 1202, column: 9, scope: !2439)
!2488 = !DILocalVariable(name: "x21imag", scope: !2439, file: !6, line: 1202, type: !15)
!2489 = !DILocation(line: 1202, column: 18, scope: !2439)
!2490 = !DILocalVariable(name: "u1", scope: !2439, file: !6, line: 1203, type: !10)
!2491 = !DILocation(line: 1203, column: 11, scope: !2439)
!2492 = !DILocation(line: 1209, column: 7, scope: !2439)
!2493 = !DILocation(line: 1209, column: 9, scope: !2439)
!2494 = !DILocation(line: 1209, column: 5, scope: !2439)
!2495 = !DILocation(line: 1210, column: 13, scope: !2439)
!2496 = !DILocation(line: 1210, column: 15, scope: !2439)
!2497 = !DILocation(line: 1210, column: 9, scope: !2439)
!2498 = !DILocation(line: 1210, column: 5, scope: !2439)
!2499 = !DILocation(line: 1211, column: 13, scope: !2439)
!2500 = !DILocation(line: 1211, column: 17, scope: !2439)
!2501 = !DILocation(line: 1211, column: 15, scope: !2439)
!2502 = !DILocation(line: 1211, column: 9, scope: !2439)
!2503 = !DILocation(line: 1211, column: 5, scope: !2439)
!2504 = !DILocation(line: 1212, column: 11, scope: !2439)
!2505 = !DILocation(line: 1212, column: 9, scope: !2439)
!2506 = !DILocation(line: 1212, column: 5, scope: !2439)
!2507 = !DILocation(line: 1213, column: 7, scope: !2439)
!2508 = !DILocation(line: 1213, column: 5, scope: !2439)
!2509 = !DILocation(line: 1214, column: 7, scope: !2510)
!2510 = distinct !DILexicalBlock(scope: !2439, file: !6, line: 1214, column: 2)
!2511 = !DILocation(line: 1214, column: 6, scope: !2510)
!2512 = !DILocation(line: 1214, column: 11, scope: !2513)
!2513 = distinct !DILexicalBlock(scope: !2510, file: !6, line: 1214, column: 2)
!2514 = !DILocation(line: 1214, column: 13, scope: !2513)
!2515 = !DILocation(line: 1214, column: 12, scope: !2513)
!2516 = !DILocation(line: 1214, column: 2, scope: !2510)
!2517 = !DILocation(line: 1215, column: 9, scope: !2518)
!2518 = distinct !DILexicalBlock(scope: !2513, file: !6, line: 1214, column: 21)
!2519 = !DILocation(line: 1215, column: 13, scope: !2518)
!2520 = !DILocation(line: 1215, column: 11, scope: !2518)
!2521 = !DILocation(line: 1215, column: 7, scope: !2518)
!2522 = !DILocation(line: 1216, column: 9, scope: !2518)
!2523 = !DILocation(line: 1216, column: 15, scope: !2518)
!2524 = !DILocation(line: 1216, column: 13, scope: !2518)
!2525 = !DILocation(line: 1216, column: 7, scope: !2518)
!2526 = !DILocation(line: 1217, column: 9, scope: !2518)
!2527 = !DILocation(line: 1217, column: 13, scope: !2518)
!2528 = !DILocation(line: 1217, column: 11, scope: !2518)
!2529 = !DILocation(line: 1217, column: 7, scope: !2518)
!2530 = !DILocation(line: 1218, column: 9, scope: !2518)
!2531 = !DILocation(line: 1218, column: 15, scope: !2518)
!2532 = !DILocation(line: 1218, column: 13, scope: !2518)
!2533 = !DILocation(line: 1218, column: 7, scope: !2518)
!2534 = !DILocation(line: 1219, column: 6, scope: !2535)
!2535 = distinct !DILexicalBlock(scope: !2518, file: !6, line: 1219, column: 6)
!2536 = !DILocation(line: 1219, column: 8, scope: !2535)
!2537 = !DILocation(line: 1219, column: 6, scope: !2518)
!2538 = !DILocation(line: 1220, column: 14, scope: !2539)
!2539 = distinct !DILexicalBlock(scope: !2535, file: !6, line: 1219, column: 12)
!2540 = !DILocation(line: 1220, column: 16, scope: !2539)
!2541 = !DILocation(line: 1220, column: 19, scope: !2539)
!2542 = !DILocation(line: 1220, column: 18, scope: !2539)
!2543 = !DILocation(line: 1220, column: 22, scope: !2539)
!2544 = !DILocation(line: 1220, column: 7, scope: !2539)
!2545 = !DILocation(line: 1220, column: 12, scope: !2539)
!2546 = !DILocation(line: 1221, column: 14, scope: !2539)
!2547 = !DILocation(line: 1221, column: 16, scope: !2539)
!2548 = !DILocation(line: 1221, column: 19, scope: !2539)
!2549 = !DILocation(line: 1221, column: 18, scope: !2539)
!2550 = !DILocation(line: 1221, column: 22, scope: !2539)
!2551 = !DILocation(line: 1221, column: 7, scope: !2539)
!2552 = !DILocation(line: 1221, column: 12, scope: !2539)
!2553 = !DILocation(line: 1222, column: 3, scope: !2539)
!2554 = !DILocation(line: 1223, column: 14, scope: !2555)
!2555 = distinct !DILexicalBlock(scope: !2535, file: !6, line: 1222, column: 8)
!2556 = !DILocation(line: 1223, column: 16, scope: !2555)
!2557 = !DILocation(line: 1223, column: 19, scope: !2555)
!2558 = !DILocation(line: 1223, column: 18, scope: !2555)
!2559 = !DILocation(line: 1223, column: 22, scope: !2555)
!2560 = !DILocation(line: 1223, column: 7, scope: !2555)
!2561 = !DILocation(line: 1223, column: 12, scope: !2555)
!2562 = !DILocation(line: 1224, column: 15, scope: !2555)
!2563 = !DILocation(line: 1224, column: 17, scope: !2555)
!2564 = !DILocation(line: 1224, column: 20, scope: !2555)
!2565 = !DILocation(line: 1224, column: 19, scope: !2555)
!2566 = !DILocation(line: 1224, column: 23, scope: !2555)
!2567 = !DILocation(line: 1224, column: 14, scope: !2555)
!2568 = !DILocation(line: 1224, column: 7, scope: !2555)
!2569 = !DILocation(line: 1224, column: 12, scope: !2555)
!2570 = !DILocation(line: 1226, column: 8, scope: !2571)
!2571 = distinct !DILexicalBlock(scope: !2518, file: !6, line: 1226, column: 3)
!2572 = !DILocation(line: 1226, column: 7, scope: !2571)
!2573 = !DILocation(line: 1226, column: 12, scope: !2574)
!2574 = distinct !DILexicalBlock(scope: !2571, file: !6, line: 1226, column: 3)
!2575 = !DILocation(line: 1226, column: 14, scope: !2574)
!2576 = !DILocation(line: 1226, column: 13, scope: !2574)
!2577 = !DILocation(line: 1226, column: 3, scope: !2571)
!2578 = !DILocation(line: 1227, column: 14, scope: !2579)
!2579 = distinct !DILexicalBlock(scope: !2574, file: !6, line: 1226, column: 22)
!2580 = !DILocation(line: 1227, column: 17, scope: !2579)
!2581 = !DILocation(line: 1227, column: 21, scope: !2579)
!2582 = !DILocation(line: 1227, column: 20, scope: !2579)
!2583 = !DILocation(line: 1227, column: 24, scope: !2579)
!2584 = !DILocation(line: 1227, column: 23, scope: !2579)
!2585 = !DILocation(line: 1227, column: 33, scope: !2579)
!2586 = !DILocation(line: 1227, column: 32, scope: !2579)
!2587 = !DILocation(line: 1227, column: 44, scope: !2579)
!2588 = !DILocation(line: 1227, column: 12, scope: !2579)
!2589 = !DILocation(line: 1228, column: 14, scope: !2579)
!2590 = !DILocation(line: 1228, column: 17, scope: !2579)
!2591 = !DILocation(line: 1228, column: 21, scope: !2579)
!2592 = !DILocation(line: 1228, column: 20, scope: !2579)
!2593 = !DILocation(line: 1228, column: 24, scope: !2579)
!2594 = !DILocation(line: 1228, column: 23, scope: !2579)
!2595 = !DILocation(line: 1228, column: 33, scope: !2579)
!2596 = !DILocation(line: 1228, column: 32, scope: !2579)
!2597 = !DILocation(line: 1228, column: 44, scope: !2579)
!2598 = !DILocation(line: 1228, column: 12, scope: !2579)
!2599 = !DILocation(line: 1229, column: 14, scope: !2579)
!2600 = !DILocation(line: 1229, column: 17, scope: !2579)
!2601 = !DILocation(line: 1229, column: 21, scope: !2579)
!2602 = !DILocation(line: 1229, column: 20, scope: !2579)
!2603 = !DILocation(line: 1229, column: 24, scope: !2579)
!2604 = !DILocation(line: 1229, column: 23, scope: !2579)
!2605 = !DILocation(line: 1229, column: 33, scope: !2579)
!2606 = !DILocation(line: 1229, column: 32, scope: !2579)
!2607 = !DILocation(line: 1229, column: 44, scope: !2579)
!2608 = !DILocation(line: 1229, column: 12, scope: !2579)
!2609 = !DILocation(line: 1230, column: 14, scope: !2579)
!2610 = !DILocation(line: 1230, column: 17, scope: !2579)
!2611 = !DILocation(line: 1230, column: 21, scope: !2579)
!2612 = !DILocation(line: 1230, column: 20, scope: !2579)
!2613 = !DILocation(line: 1230, column: 24, scope: !2579)
!2614 = !DILocation(line: 1230, column: 23, scope: !2579)
!2615 = !DILocation(line: 1230, column: 33, scope: !2579)
!2616 = !DILocation(line: 1230, column: 32, scope: !2579)
!2617 = !DILocation(line: 1230, column: 44, scope: !2579)
!2618 = !DILocation(line: 1230, column: 12, scope: !2579)
!2619 = !DILocation(line: 1231, column: 41, scope: !2579)
!2620 = !DILocation(line: 1231, column: 51, scope: !2579)
!2621 = !DILocation(line: 1231, column: 49, scope: !2579)
!2622 = !DILocation(line: 1231, column: 4, scope: !2579)
!2623 = !DILocation(line: 1231, column: 7, scope: !2579)
!2624 = !DILocation(line: 1231, column: 11, scope: !2579)
!2625 = !DILocation(line: 1231, column: 10, scope: !2579)
!2626 = !DILocation(line: 1231, column: 14, scope: !2579)
!2627 = !DILocation(line: 1231, column: 13, scope: !2579)
!2628 = !DILocation(line: 1231, column: 23, scope: !2579)
!2629 = !DILocation(line: 1231, column: 22, scope: !2579)
!2630 = !DILocation(line: 1231, column: 34, scope: !2579)
!2631 = !DILocation(line: 1231, column: 39, scope: !2579)
!2632 = !DILocation(line: 1232, column: 41, scope: !2579)
!2633 = !DILocation(line: 1232, column: 51, scope: !2579)
!2634 = !DILocation(line: 1232, column: 49, scope: !2579)
!2635 = !DILocation(line: 1232, column: 4, scope: !2579)
!2636 = !DILocation(line: 1232, column: 7, scope: !2579)
!2637 = !DILocation(line: 1232, column: 11, scope: !2579)
!2638 = !DILocation(line: 1232, column: 10, scope: !2579)
!2639 = !DILocation(line: 1232, column: 14, scope: !2579)
!2640 = !DILocation(line: 1232, column: 13, scope: !2579)
!2641 = !DILocation(line: 1232, column: 23, scope: !2579)
!2642 = !DILocation(line: 1232, column: 22, scope: !2579)
!2643 = !DILocation(line: 1232, column: 34, scope: !2579)
!2644 = !DILocation(line: 1232, column: 39, scope: !2579)
!2645 = !DILocation(line: 1233, column: 44, scope: !2579)
!2646 = !DILocation(line: 1233, column: 52, scope: !2579)
!2647 = !DILocation(line: 1233, column: 62, scope: !2579)
!2648 = !DILocation(line: 1233, column: 60, scope: !2579)
!2649 = !DILocation(line: 1233, column: 49, scope: !2579)
!2650 = !DILocation(line: 1233, column: 76, scope: !2579)
!2651 = !DILocation(line: 1233, column: 84, scope: !2579)
!2652 = !DILocation(line: 1233, column: 94, scope: !2579)
!2653 = !DILocation(line: 1233, column: 92, scope: !2579)
!2654 = !DILocation(line: 1233, column: 81, scope: !2579)
!2655 = !DILocation(line: 1233, column: 71, scope: !2579)
!2656 = !DILocation(line: 1233, column: 4, scope: !2579)
!2657 = !DILocation(line: 1233, column: 7, scope: !2579)
!2658 = !DILocation(line: 1233, column: 11, scope: !2579)
!2659 = !DILocation(line: 1233, column: 10, scope: !2579)
!2660 = !DILocation(line: 1233, column: 14, scope: !2579)
!2661 = !DILocation(line: 1233, column: 13, scope: !2579)
!2662 = !DILocation(line: 1233, column: 23, scope: !2579)
!2663 = !DILocation(line: 1233, column: 22, scope: !2579)
!2664 = !DILocation(line: 1233, column: 34, scope: !2579)
!2665 = !DILocation(line: 1233, column: 39, scope: !2579)
!2666 = !DILocation(line: 1234, column: 44, scope: !2579)
!2667 = !DILocation(line: 1234, column: 52, scope: !2579)
!2668 = !DILocation(line: 1234, column: 62, scope: !2579)
!2669 = !DILocation(line: 1234, column: 60, scope: !2579)
!2670 = !DILocation(line: 1234, column: 49, scope: !2579)
!2671 = !DILocation(line: 1234, column: 76, scope: !2579)
!2672 = !DILocation(line: 1234, column: 84, scope: !2579)
!2673 = !DILocation(line: 1234, column: 94, scope: !2579)
!2674 = !DILocation(line: 1234, column: 92, scope: !2579)
!2675 = !DILocation(line: 1234, column: 81, scope: !2579)
!2676 = !DILocation(line: 1234, column: 71, scope: !2579)
!2677 = !DILocation(line: 1234, column: 4, scope: !2579)
!2678 = !DILocation(line: 1234, column: 7, scope: !2579)
!2679 = !DILocation(line: 1234, column: 11, scope: !2579)
!2680 = !DILocation(line: 1234, column: 10, scope: !2579)
!2681 = !DILocation(line: 1234, column: 14, scope: !2579)
!2682 = !DILocation(line: 1234, column: 13, scope: !2579)
!2683 = !DILocation(line: 1234, column: 23, scope: !2579)
!2684 = !DILocation(line: 1234, column: 22, scope: !2579)
!2685 = !DILocation(line: 1234, column: 34, scope: !2579)
!2686 = !DILocation(line: 1234, column: 39, scope: !2579)
!2687 = !DILocation(line: 1235, column: 3, scope: !2579)
!2688 = !DILocation(line: 1226, column: 19, scope: !2574)
!2689 = !DILocation(line: 1226, column: 3, scope: !2574)
!2690 = distinct !{!2690, !2577, !2691}
!2691 = !DILocation(line: 1235, column: 3, scope: !2571)
!2692 = !DILocation(line: 1236, column: 2, scope: !2518)
!2693 = !DILocation(line: 1214, column: 18, scope: !2513)
!2694 = !DILocation(line: 1214, column: 2, scope: !2513)
!2695 = distinct !{!2695, !2516, !2696}
!2696 = !DILocation(line: 1236, column: 2, scope: !2510)
!2697 = !DILocation(line: 1237, column: 1, scope: !2439)
!2698 = distinct !DISubprogram(name: "cffts3_gpu_kernel_1", linkageName: "_Z19cffts3_gpu_kernel_1P8dcomplexS0_", scope: !6, file: !6, line: 1246, type: !803, scopeLine: 1247, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !7)
!2699 = !DILocalVariable(name: "x_in", arg: 1, scope: !2698, file: !6, line: 1246, type: !9)
!2700 = !DILocation(line: 1246, column: 46, scope: !2698)
!2701 = !DILocalVariable(name: "y0", arg: 2, scope: !2698, file: !6, line: 1247, type: !9)
!2702 = !DILocation(line: 1247, column: 12, scope: !2698)
!2703 = !DILocalVariable(name: "x_y_z", scope: !2698, file: !6, line: 1248, type: !18)
!2704 = !DILocation(line: 1248, column: 6, scope: !2698)
!2705 = !DILocation(line: 64, column: 3, scope: !812, inlinedAt: !2706)
!2706 = distinct !DILocation(line: 1248, column: 14, scope: !2698)
!2707 = !DILocation(line: 75, column: 3, scope: !850, inlinedAt: !2708)
!2708 = distinct !DILocation(line: 1248, column: 27, scope: !2698)
!2709 = !DILocation(line: 1248, column: 25, scope: !2698)
!2710 = !DILocation(line: 53, column: 3, scope: !896, inlinedAt: !2711)
!2711 = distinct !DILocation(line: 1248, column: 40, scope: !2698)
!2712 = !DILocation(line: 1248, column: 38, scope: !2698)
!2713 = !DILocation(line: 1249, column: 5, scope: !2714)
!2714 = distinct !DILexicalBlock(scope: !2698, file: !6, line: 1249, column: 5)
!2715 = !DILocation(line: 1249, column: 11, scope: !2714)
!2716 = !DILocation(line: 1249, column: 5, scope: !2698)
!2717 = !DILocation(line: 1250, column: 3, scope: !2718)
!2718 = distinct !DILexicalBlock(scope: !2714, file: !6, line: 1249, column: 25)
!2719 = !DILocation(line: 1252, column: 19, scope: !2698)
!2720 = !DILocation(line: 1252, column: 24, scope: !2698)
!2721 = !DILocation(line: 1252, column: 31, scope: !2698)
!2722 = !DILocation(line: 1252, column: 2, scope: !2698)
!2723 = !DILocation(line: 1252, column: 5, scope: !2698)
!2724 = !DILocation(line: 1252, column: 12, scope: !2698)
!2725 = !DILocation(line: 1252, column: 17, scope: !2698)
!2726 = !DILocation(line: 1253, column: 19, scope: !2698)
!2727 = !DILocation(line: 1253, column: 24, scope: !2698)
!2728 = !DILocation(line: 1253, column: 31, scope: !2698)
!2729 = !DILocation(line: 1253, column: 2, scope: !2698)
!2730 = !DILocation(line: 1253, column: 5, scope: !2698)
!2731 = !DILocation(line: 1253, column: 12, scope: !2698)
!2732 = !DILocation(line: 1253, column: 17, scope: !2698)
!2733 = !DILocation(line: 1254, column: 1, scope: !2698)
!2734 = distinct !DISubprogram(name: "cffts3_gpu_kernel_2", linkageName: "_Z19cffts3_gpu_kernel_2iP8dcomplexS0_S0_", scope: !6, file: !6, line: 1261, type: !974, scopeLine: 1264, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !7)
!2735 = !DILocalVariable(name: "is", arg: 1, scope: !2734, file: !6, line: 1261, type: !976)
!2736 = !DILocation(line: 1261, column: 47, scope: !2734)
!2737 = !DILocalVariable(name: "gty1", arg: 2, scope: !2734, file: !6, line: 1262, type: !9)
!2738 = !DILocation(line: 1262, column: 12, scope: !2734)
!2739 = !DILocalVariable(name: "gty2", arg: 3, scope: !2734, file: !6, line: 1263, type: !9)
!2740 = !DILocation(line: 1263, column: 12, scope: !2734)
!2741 = !DILocalVariable(name: "u_device", arg: 4, scope: !2734, file: !6, line: 1264, type: !9)
!2742 = !DILocation(line: 1264, column: 12, scope: !2734)
!2743 = !DILocalVariable(name: "x_y", scope: !2734, file: !6, line: 1265, type: !18)
!2744 = !DILocation(line: 1265, column: 6, scope: !2734)
!2745 = !DILocation(line: 64, column: 3, scope: !812, inlinedAt: !2746)
!2746 = distinct !DILocation(line: 1265, column: 12, scope: !2734)
!2747 = !DILocation(line: 75, column: 3, scope: !850, inlinedAt: !2748)
!2748 = distinct !DILocation(line: 1265, column: 25, scope: !2734)
!2749 = !DILocation(line: 1265, column: 23, scope: !2734)
!2750 = !DILocation(line: 53, column: 3, scope: !896, inlinedAt: !2751)
!2751 = distinct !DILocation(line: 1265, column: 38, scope: !2734)
!2752 = !DILocation(line: 1265, column: 36, scope: !2734)
!2753 = !DILocation(line: 1266, column: 5, scope: !2754)
!2754 = distinct !DILexicalBlock(scope: !2734, file: !6, line: 1266, column: 5)
!2755 = !DILocation(line: 1266, column: 9, scope: !2754)
!2756 = !DILocation(line: 1266, column: 5, scope: !2734)
!2757 = !DILocation(line: 1267, column: 3, scope: !2758)
!2758 = distinct !DILexicalBlock(scope: !2754, file: !6, line: 1266, column: 20)
!2759 = !DILocation(line: 1269, column: 26, scope: !2734)
!2760 = !DILocation(line: 1270, column: 4, scope: !2734)
!2761 = !DILocation(line: 1272, column: 4, scope: !2734)
!2762 = !DILocation(line: 1273, column: 4, scope: !2734)
!2763 = !DILocation(line: 1274, column: 4, scope: !2734)
!2764 = !DILocation(line: 1275, column: 4, scope: !2734)
!2765 = !DILocation(line: 1269, column: 2, scope: !2734)
!2766 = !DILocation(line: 1277, column: 1, scope: !2734)
!2767 = distinct !DISubprogram(name: "cffts3_gpu_kernel_3", linkageName: "_Z19cffts3_gpu_kernel_3P8dcomplexS0_", scope: !6, file: !6, line: 1286, type: !803, scopeLine: 1287, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !7)
!2768 = !DILocalVariable(name: "x_out", arg: 1, scope: !2767, file: !6, line: 1286, type: !9)
!2769 = !DILocation(line: 1286, column: 46, scope: !2767)
!2770 = !DILocalVariable(name: "y0", arg: 2, scope: !2767, file: !6, line: 1287, type: !9)
!2771 = !DILocation(line: 1287, column: 12, scope: !2767)
!2772 = !DILocalVariable(name: "x_y_z", scope: !2767, file: !6, line: 1288, type: !18)
!2773 = !DILocation(line: 1288, column: 6, scope: !2767)
!2774 = !DILocation(line: 64, column: 3, scope: !812, inlinedAt: !2775)
!2775 = distinct !DILocation(line: 1288, column: 14, scope: !2767)
!2776 = !DILocation(line: 75, column: 3, scope: !850, inlinedAt: !2777)
!2777 = distinct !DILocation(line: 1288, column: 27, scope: !2767)
!2778 = !DILocation(line: 1288, column: 25, scope: !2767)
!2779 = !DILocation(line: 53, column: 3, scope: !896, inlinedAt: !2780)
!2780 = distinct !DILocation(line: 1288, column: 40, scope: !2767)
!2781 = !DILocation(line: 1288, column: 38, scope: !2767)
!2782 = !DILocation(line: 1289, column: 5, scope: !2783)
!2783 = distinct !DILexicalBlock(scope: !2767, file: !6, line: 1289, column: 5)
!2784 = !DILocation(line: 1289, column: 11, scope: !2783)
!2785 = !DILocation(line: 1289, column: 5, scope: !2767)
!2786 = !DILocation(line: 1290, column: 3, scope: !2787)
!2787 = distinct !DILexicalBlock(scope: !2783, file: !6, line: 1289, column: 25)
!2788 = !DILocation(line: 1292, column: 22, scope: !2767)
!2789 = !DILocation(line: 1292, column: 25, scope: !2767)
!2790 = !DILocation(line: 1292, column: 32, scope: !2767)
!2791 = !DILocation(line: 1292, column: 2, scope: !2767)
!2792 = !DILocation(line: 1292, column: 8, scope: !2767)
!2793 = !DILocation(line: 1292, column: 15, scope: !2767)
!2794 = !DILocation(line: 1292, column: 20, scope: !2767)
!2795 = !DILocation(line: 1293, column: 22, scope: !2767)
!2796 = !DILocation(line: 1293, column: 25, scope: !2767)
!2797 = !DILocation(line: 1293, column: 32, scope: !2767)
!2798 = !DILocation(line: 1293, column: 2, scope: !2767)
!2799 = !DILocation(line: 1293, column: 8, scope: !2767)
!2800 = !DILocation(line: 1293, column: 15, scope: !2767)
!2801 = !DILocation(line: 1293, column: 20, scope: !2767)
!2802 = !DILocation(line: 1294, column: 1, scope: !2767)
!2803 = distinct !DISubprogram(name: "checksum_gpu_kernel", linkageName: "_Z19checksum_gpu_kerneliP8dcomplexS0_", scope: !6, file: !6, line: 1311, type: !2804, scopeLine: 1313, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !7)
!2804 = !DISubroutineType(types: !2805)
!2805 = !{null, !18, !9, !9}
!2806 = !DILocalVariable(name: "iteration", arg: 1, scope: !2803, file: !6, line: 1311, type: !18)
!2807 = !DILocation(line: 1311, column: 41, scope: !2803)
!2808 = !DILocalVariable(name: "u1", arg: 2, scope: !2803, file: !6, line: 1312, type: !9)
!2809 = !DILocation(line: 1312, column: 12, scope: !2803)
!2810 = !DILocalVariable(name: "sums", arg: 3, scope: !2803, file: !6, line: 1313, type: !9)
!2811 = !DILocation(line: 1313, column: 12, scope: !2803)
!2812 = !DILocalVariable(name: "share_sums", scope: !2803, file: !6, line: 1314, type: !9)
!2813 = !DILocation(line: 1314, column: 12, scope: !2803)
!2814 = !DILocalVariable(name: "j", scope: !2803, file: !6, line: 1315, type: !18)
!2815 = !DILocation(line: 1315, column: 6, scope: !2803)
!2816 = !DILocation(line: 64, column: 3, scope: !812, inlinedAt: !2817)
!2817 = distinct !DILocation(line: 1315, column: 11, scope: !2803)
!2818 = !DILocation(line: 75, column: 3, scope: !850, inlinedAt: !2819)
!2819 = distinct !DILocation(line: 1315, column: 24, scope: !2803)
!2820 = !DILocation(line: 1315, column: 22, scope: !2803)
!2821 = !DILocation(line: 53, column: 3, scope: !896, inlinedAt: !2822)
!2822 = distinct !DILocation(line: 1315, column: 37, scope: !2803)
!2823 = !DILocation(line: 1315, column: 35, scope: !2803)
!2824 = !DILocation(line: 1315, column: 50, scope: !2803)
!2825 = !DILocalVariable(name: "q", scope: !2803, file: !6, line: 1316, type: !18)
!2826 = !DILocation(line: 1316, column: 6, scope: !2803)
!2827 = !DILocalVariable(name: "r", scope: !2803, file: !6, line: 1316, type: !18)
!2828 = !DILocation(line: 1316, column: 9, scope: !2803)
!2829 = !DILocalVariable(name: "s", scope: !2803, file: !6, line: 1316, type: !18)
!2830 = !DILocation(line: 1316, column: 12, scope: !2803)
!2831 = !DILocation(line: 1318, column: 5, scope: !2832)
!2832 = distinct !DILexicalBlock(scope: !2803, file: !6, line: 1318, column: 5)
!2833 = !DILocation(line: 1318, column: 6, scope: !2832)
!2834 = !DILocation(line: 1318, column: 5, scope: !2803)
!2835 = !DILocation(line: 1319, column: 7, scope: !2836)
!2836 = distinct !DILexicalBlock(scope: !2832, file: !6, line: 1318, column: 23)
!2837 = !DILocation(line: 1319, column: 9, scope: !2836)
!2838 = !DILocation(line: 1319, column: 5, scope: !2836)
!2839 = !DILocation(line: 1320, column: 9, scope: !2836)
!2840 = !DILocation(line: 1320, column: 8, scope: !2836)
!2841 = !DILocation(line: 1320, column: 11, scope: !2836)
!2842 = !DILocation(line: 1320, column: 5, scope: !2836)
!2843 = !DILocation(line: 1321, column: 9, scope: !2836)
!2844 = !DILocation(line: 1321, column: 8, scope: !2836)
!2845 = !DILocation(line: 1321, column: 11, scope: !2836)
!2846 = !DILocation(line: 1321, column: 5, scope: !2836)
!2847 = !DILocation(line: 1322, column: 29, scope: !2836)
!2848 = !DILocation(line: 1322, column: 33, scope: !2836)
!2849 = !DILocation(line: 1322, column: 37, scope: !2836)
!2850 = !DILocation(line: 1322, column: 38, scope: !2836)
!2851 = !DILocation(line: 1322, column: 35, scope: !2836)
!2852 = !DILocation(line: 1322, column: 44, scope: !2836)
!2853 = !DILocation(line: 1322, column: 45, scope: !2836)
!2854 = !DILocation(line: 1322, column: 48, scope: !2836)
!2855 = !DILocation(line: 1322, column: 42, scope: !2836)
!2856 = !DILocation(line: 1322, column: 3, scope: !2836)
!2857 = !DILocation(line: 53, column: 3, scope: !896, inlinedAt: !2858)
!2858 = distinct !DILocation(line: 1322, column: 14, scope: !2836)
!2859 = !DILocation(line: 1322, column: 27, scope: !2836)
!2860 = !DILocation(line: 1323, column: 2, scope: !2836)
!2861 = !DILocation(line: 1324, column: 29, scope: !2862)
!2862 = distinct !DILexicalBlock(scope: !2832, file: !6, line: 1323, column: 7)
!2863 = !DILocation(line: 1324, column: 3, scope: !2862)
!2864 = !DILocation(line: 53, column: 3, scope: !896, inlinedAt: !2865)
!2865 = distinct !DILocation(line: 1324, column: 14, scope: !2862)
!2866 = !DILocation(line: 1324, column: 27, scope: !2862)
!2867 = !DILocation(line: 1328, column: 2, scope: !2803)
!2868 = !DILocation(line: 53, column: 3, scope: !896, inlinedAt: !2869)
!2869 = distinct !DILocation(line: 1329, column: 5, scope: !2870)
!2870 = distinct !DILexicalBlock(scope: !2803, file: !6, line: 1329, column: 5)
!2871 = !DILocation(line: 1329, column: 16, scope: !2870)
!2872 = !DILocation(line: 1329, column: 5, scope: !2803)
!2873 = !DILocalVariable(name: "i", scope: !2874, file: !6, line: 1330, type: !18)
!2874 = distinct !DILexicalBlock(scope: !2875, file: !6, line: 1330, column: 3)
!2875 = distinct !DILexicalBlock(scope: !2870, file: !6, line: 1329, column: 20)
!2876 = !DILocation(line: 1330, column: 11, scope: !2874)
!2877 = !DILocation(line: 1330, column: 7, scope: !2874)
!2878 = !DILocation(line: 1330, column: 16, scope: !2879)
!2879 = distinct !DILexicalBlock(scope: !2874, file: !6, line: 1330, column: 3)
!2880 = !DILocation(line: 75, column: 3, scope: !850, inlinedAt: !2881)
!2881 = distinct !DILocation(line: 1330, column: 18, scope: !2879)
!2882 = !DILocation(line: 1330, column: 17, scope: !2879)
!2883 = !DILocation(line: 1330, column: 3, scope: !2874)
!2884 = !DILocation(line: 1331, column: 20, scope: !2885)
!2885 = distinct !DILexicalBlock(scope: !2879, file: !6, line: 1330, column: 34)
!2886 = !DILocation(line: 1331, column: 4, scope: !2885)
!2887 = !DILocation(line: 1331, column: 18, scope: !2885)
!2888 = !DILocation(line: 1332, column: 3, scope: !2885)
!2889 = !DILocation(line: 1330, column: 30, scope: !2879)
!2890 = !DILocation(line: 1330, column: 3, scope: !2879)
!2891 = distinct !{!2891, !2883, !2892}
!2892 = !DILocation(line: 1332, column: 3, scope: !2874)
!2893 = !DILocation(line: 1333, column: 2, scope: !2875)
!2894 = !DILocation(line: 53, column: 3, scope: !896, inlinedAt: !2895)
!2895 = distinct !DILocation(line: 1334, column: 5, scope: !2896)
!2896 = distinct !DILexicalBlock(scope: !2803, file: !6, line: 1334, column: 5)
!2897 = !DILocation(line: 1334, column: 16, scope: !2896)
!2898 = !DILocation(line: 1334, column: 5, scope: !2803)
!2899 = !DILocation(line: 1335, column: 24, scope: !2900)
!2900 = distinct !DILexicalBlock(scope: !2896, file: !6, line: 1334, column: 20)
!2901 = !DILocation(line: 1335, column: 38, scope: !2900)
!2902 = !DILocation(line: 1335, column: 42, scope: !2900)
!2903 = !DILocation(line: 1335, column: 3, scope: !2900)
!2904 = !DILocation(line: 1335, column: 17, scope: !2900)
!2905 = !DILocation(line: 1335, column: 22, scope: !2900)
!2906 = !DILocation(line: 1336, column: 14, scope: !2900)
!2907 = !DILocation(line: 1336, column: 19, scope: !2900)
!2908 = !DILocation(line: 1336, column: 30, scope: !2900)
!2909 = !DILocation(line: 1336, column: 35, scope: !2900)
!2910 = !DILocation(line: 1336, column: 49, scope: !2900)
!2911 = !DILocation(line: 1336, column: 3, scope: !2900)
!2912 = !DILocation(line: 1337, column: 24, scope: !2900)
!2913 = !DILocation(line: 1337, column: 38, scope: !2900)
!2914 = !DILocation(line: 1337, column: 42, scope: !2900)
!2915 = !DILocation(line: 1337, column: 3, scope: !2900)
!2916 = !DILocation(line: 1337, column: 17, scope: !2900)
!2917 = !DILocation(line: 1337, column: 22, scope: !2900)
!2918 = !DILocation(line: 1338, column: 14, scope: !2900)
!2919 = !DILocation(line: 1338, column: 19, scope: !2900)
!2920 = !DILocation(line: 1338, column: 30, scope: !2900)
!2921 = !DILocation(line: 1338, column: 35, scope: !2900)
!2922 = !DILocation(line: 1338, column: 49, scope: !2900)
!2923 = !DILocation(line: 1338, column: 3, scope: !2900)
!2924 = !DILocation(line: 1339, column: 2, scope: !2900)
!2925 = !DILocation(line: 1340, column: 1, scope: !2803)
!2926 = distinct !DISubprogram(name: "atomicAdd", linkageName: "_ZL9atomicAddPdd", scope: !11, file: !11, line: 54, type: !2927, scopeLine: 54, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !5, retainedNodes: !7)
!2927 = !DISubroutineType(types: !2928)
!2928 = !{!15, !17, !15}
!2929 = !DILocalVariable(name: "x", arg: 1, scope: !2930, file: !511, line: 1370, type: !15)
!2930 = distinct !DISubprogram(name: "__double_as_longlong", linkageName: "_ZL20__double_as_longlongd", scope: !511, file: !511, line: 1370, type: !2931, scopeLine: 1371, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !5, retainedNodes: !7)
!2931 = !DISubroutineType(types: !2932)
!2932 = !{!23, !15}
!2933 = !DILocation(line: 1370, column: 74, scope: !2930, inlinedAt: !2934)
!2934 = distinct !DILocation(line: 61, column: 44, scope: !2935)
!2935 = distinct !DILexicalBlock(scope: !2936, file: !11, line: 59, column: 35)
!2936 = distinct !DILexicalBlock(scope: !2937, file: !11, line: 59, column: 2)
!2937 = distinct !DILexicalBlock(scope: !2926, file: !11, line: 59, column: 2)
!2938 = !DILocalVariable(name: "x", arg: 1, scope: !2939, file: !511, line: 1365, type: !23)
!2939 = distinct !DISubprogram(name: "__longlong_as_double", linkageName: "_ZL20__longlong_as_doublex", scope: !511, file: !511, line: 1365, type: !2940, scopeLine: 1366, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !5, retainedNodes: !7)
!2940 = !DISubroutineType(types: !2941)
!2941 = !{!15, !23}
!2942 = !DILocation(line: 1365, column: 72, scope: !2939, inlinedAt: !2943)
!2943 = distinct !DILocation(line: 61, column: 70, scope: !2935)
!2944 = !DILocation(line: 1365, column: 72, scope: !2939, inlinedAt: !2945)
!2945 = distinct !DILocation(line: 62, column: 36, scope: !2946)
!2946 = distinct !DILexicalBlock(scope: !2935, file: !11, line: 62, column: 13)
!2947 = !DILocation(line: 1365, column: 72, scope: !2939, inlinedAt: !2948)
!2948 = distinct !DILocation(line: 64, column: 9, scope: !2926)
!2949 = !DILocation(line: 1365, column: 72, scope: !2939, inlinedAt: !2950)
!2950 = distinct !DILocation(line: 58, column: 10, scope: !2951)
!2951 = distinct !DILexicalBlock(scope: !2926, file: !11, line: 57, column: 6)
!2952 = !DILocalVariable(name: "address", arg: 1, scope: !2926, file: !11, line: 54, type: !17)
!2953 = !DILocation(line: 54, column: 55, scope: !2926)
!2954 = !DILocalVariable(name: "val", arg: 2, scope: !2926, file: !11, line: 54, type: !15)
!2955 = !DILocation(line: 54, column: 71, scope: !2926)
!2956 = !DILocalVariable(name: "address_as_ull", scope: !2926, file: !11, line: 55, type: !19)
!2957 = !DILocation(line: 55, column: 26, scope: !2926)
!2958 = !DILocation(line: 55, column: 68, scope: !2926)
!2959 = !DILocation(line: 55, column: 43, scope: !2926)
!2960 = !DILocalVariable(name: "old", scope: !2926, file: !11, line: 56, type: !20)
!2961 = !DILocation(line: 56, column: 25, scope: !2926)
!2962 = !DILocation(line: 56, column: 32, scope: !2926)
!2963 = !DILocation(line: 56, column: 31, scope: !2926)
!2964 = !DILocalVariable(name: "assumed", scope: !2926, file: !11, line: 56, type: !20)
!2965 = !DILocation(line: 56, column: 48, scope: !2926)
!2966 = !DILocation(line: 57, column: 6, scope: !2951)
!2967 = !DILocation(line: 57, column: 9, scope: !2951)
!2968 = !DILocation(line: 57, column: 6, scope: !2926)
!2969 = !DILocation(line: 58, column: 31, scope: !2951)
!2970 = !DILocation(line: 1367, column: 34, scope: !2939, inlinedAt: !2950)
!2971 = !DILocation(line: 1367, column: 10, scope: !2939, inlinedAt: !2950)
!2972 = !DILocation(line: 58, column: 3, scope: !2951)
!2973 = !DILocalVariable(name: "i", scope: !2937, file: !11, line: 59, type: !18)
!2974 = !DILocation(line: 59, column: 11, scope: !2937)
!2975 = !DILocation(line: 59, column: 7, scope: !2937)
!2976 = !DILocation(line: 59, column: 18, scope: !2936)
!2977 = !DILocation(line: 59, column: 20, scope: !2936)
!2978 = !DILocation(line: 59, column: 2, scope: !2937)
!2979 = !DILocation(line: 60, column: 13, scope: !2935)
!2980 = !DILocation(line: 60, column: 11, scope: !2935)
!2981 = !DILocation(line: 61, column: 19, scope: !2935)
!2982 = !DILocation(line: 61, column: 35, scope: !2935)
!2983 = !DILocation(line: 61, column: 65, scope: !2935)
!2984 = !DILocation(line: 61, column: 91, scope: !2935)
!2985 = !DILocation(line: 1367, column: 34, scope: !2939, inlinedAt: !2943)
!2986 = !DILocation(line: 1367, column: 10, scope: !2939, inlinedAt: !2943)
!2987 = !DILocation(line: 61, column: 69, scope: !2935)
!2988 = !DILocation(line: 1372, column: 34, scope: !2930, inlinedAt: !2934)
!2989 = !DILocation(line: 1372, column: 10, scope: !2930, inlinedAt: !2934)
!2990 = !DILocation(line: 61, column: 9, scope: !2935)
!2991 = !DILocation(line: 61, column: 7, scope: !2935)
!2992 = !DILocation(line: 62, column: 13, scope: !2946)
!2993 = !DILocation(line: 62, column: 24, scope: !2946)
!2994 = !DILocation(line: 62, column: 21, scope: !2946)
!2995 = !DILocation(line: 62, column: 13, scope: !2935)
!2996 = !DILocation(line: 62, column: 57, scope: !2946)
!2997 = !DILocation(line: 1367, column: 34, scope: !2939, inlinedAt: !2945)
!2998 = !DILocation(line: 1367, column: 10, scope: !2939, inlinedAt: !2945)
!2999 = !DILocation(line: 62, column: 29, scope: !2946)
!3000 = !DILocation(line: 63, column: 2, scope: !2935)
!3001 = !DILocation(line: 59, column: 31, scope: !2936)
!3002 = !DILocation(line: 59, column: 2, scope: !2936)
!3003 = distinct !{!3003, !2978, !3004}
!3004 = !DILocation(line: 63, column: 2, scope: !2937)
!3005 = !DILocation(line: 64, column: 30, scope: !2926)
!3006 = !DILocation(line: 1367, column: 34, scope: !2939, inlinedAt: !2948)
!3007 = !DILocation(line: 1367, column: 10, scope: !2939, inlinedAt: !2948)
!3008 = !DILocation(line: 64, column: 2, scope: !2926)
!3009 = !DILocation(line: 65, column: 1, scope: !2926)
!3010 = distinct !DISubprogram(name: "compute_indexmap_gpu_kernel", linkageName: "_Z27compute_indexmap_gpu_kernelPd", scope: !6, file: !6, line: 1353, type: !3011, scopeLine: 1353, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !7)
!3011 = !DISubroutineType(types: !3012)
!3012 = !{null, !17}
!3013 = !DILocalVariable(name: "a", arg: 1, scope: !3014, file: !3015, line: 245, type: !15)
!3014 = distinct !DISubprogram(name: "exp", linkageName: "_ZL3expd", scope: !3015, file: !3015, line: 245, type: !222, scopeLine: 246, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !5, retainedNodes: !7)
!3015 = !DIFile(filename: "/usr/local/cuda/include/math_functions_dbl_ptx3.hpp", directory: "")
!3016 = !DILocation(line: 245, column: 52, scope: !3014, inlinedAt: !3017)
!3017 = distinct !DILocation(line: 1372, column: 23, scope: !3010)
!3018 = !DILocalVariable(name: "twiddle", arg: 1, scope: !3010, file: !6, line: 1353, type: !17)
!3019 = !DILocation(line: 1353, column: 52, scope: !3010)
!3020 = !DILocalVariable(name: "thread_id", scope: !3010, file: !6, line: 1354, type: !18)
!3021 = !DILocation(line: 1354, column: 6, scope: !3010)
!3022 = !DILocation(line: 64, column: 3, scope: !812, inlinedAt: !3023)
!3023 = distinct !DILocation(line: 1354, column: 18, scope: !3010)
!3024 = !DILocation(line: 75, column: 3, scope: !850, inlinedAt: !3025)
!3025 = distinct !DILocation(line: 1354, column: 31, scope: !3010)
!3026 = !DILocation(line: 1354, column: 29, scope: !3010)
!3027 = !DILocation(line: 53, column: 3, scope: !896, inlinedAt: !3028)
!3028 = distinct !DILocation(line: 1354, column: 44, scope: !3010)
!3029 = !DILocation(line: 1354, column: 42, scope: !3010)
!3030 = !DILocation(line: 1356, column: 5, scope: !3031)
!3031 = distinct !DILexicalBlock(scope: !3010, file: !6, line: 1356, column: 5)
!3032 = !DILocation(line: 1356, column: 14, scope: !3031)
!3033 = !DILocation(line: 1356, column: 5, scope: !3010)
!3034 = !DILocation(line: 1357, column: 3, scope: !3035)
!3035 = distinct !DILexicalBlock(scope: !3031, file: !6, line: 1356, column: 23)
!3036 = !DILocalVariable(name: "i", scope: !3010, file: !6, line: 1360, type: !18)
!3037 = !DILocation(line: 1360, column: 6, scope: !3010)
!3038 = !DILocation(line: 1360, column: 10, scope: !3010)
!3039 = !DILocation(line: 1360, column: 20, scope: !3010)
!3040 = !DILocalVariable(name: "j", scope: !3010, file: !6, line: 1361, type: !18)
!3041 = !DILocation(line: 1361, column: 6, scope: !3010)
!3042 = !DILocation(line: 1361, column: 11, scope: !3010)
!3043 = !DILocation(line: 1361, column: 21, scope: !3010)
!3044 = !DILocation(line: 1361, column: 27, scope: !3010)
!3045 = !DILocalVariable(name: "k", scope: !3010, file: !6, line: 1362, type: !18)
!3046 = !DILocation(line: 1362, column: 6, scope: !3010)
!3047 = !DILocation(line: 1362, column: 10, scope: !3010)
!3048 = !DILocation(line: 1362, column: 20, scope: !3010)
!3049 = !DILocalVariable(name: "kk", scope: !3010, file: !6, line: 1364, type: !18)
!3050 = !DILocation(line: 1364, column: 6, scope: !3010)
!3051 = !DILocalVariable(name: "kk2", scope: !3010, file: !6, line: 1364, type: !18)
!3052 = !DILocation(line: 1364, column: 10, scope: !3010)
!3053 = !DILocalVariable(name: "jj", scope: !3010, file: !6, line: 1364, type: !18)
!3054 = !DILocation(line: 1364, column: 15, scope: !3010)
!3055 = !DILocalVariable(name: "kj2", scope: !3010, file: !6, line: 1364, type: !18)
!3056 = !DILocation(line: 1364, column: 19, scope: !3010)
!3057 = !DILocalVariable(name: "ii", scope: !3010, file: !6, line: 1364, type: !18)
!3058 = !DILocation(line: 1364, column: 24, scope: !3010)
!3059 = !DILocation(line: 1366, column: 9, scope: !3010)
!3060 = !DILocation(line: 1366, column: 10, scope: !3010)
!3061 = !DILocation(line: 1366, column: 17, scope: !3010)
!3062 = !DILocation(line: 1366, column: 23, scope: !3010)
!3063 = !DILocation(line: 1366, column: 5, scope: !3010)
!3064 = !DILocation(line: 1367, column: 8, scope: !3010)
!3065 = !DILocation(line: 1367, column: 11, scope: !3010)
!3066 = !DILocation(line: 1367, column: 10, scope: !3010)
!3067 = !DILocation(line: 1367, column: 6, scope: !3010)
!3068 = !DILocation(line: 1368, column: 9, scope: !3010)
!3069 = !DILocation(line: 1368, column: 10, scope: !3010)
!3070 = !DILocation(line: 1368, column: 17, scope: !3010)
!3071 = !DILocation(line: 1368, column: 23, scope: !3010)
!3072 = !DILocation(line: 1368, column: 5, scope: !3010)
!3073 = !DILocation(line: 1369, column: 8, scope: !3010)
!3074 = !DILocation(line: 1369, column: 11, scope: !3010)
!3075 = !DILocation(line: 1369, column: 10, scope: !3010)
!3076 = !DILocation(line: 1369, column: 14, scope: !3010)
!3077 = !DILocation(line: 1369, column: 13, scope: !3010)
!3078 = !DILocation(line: 1369, column: 6, scope: !3010)
!3079 = !DILocation(line: 1370, column: 9, scope: !3010)
!3080 = !DILocation(line: 1370, column: 10, scope: !3010)
!3081 = !DILocation(line: 1370, column: 17, scope: !3010)
!3082 = !DILocation(line: 1370, column: 23, scope: !3010)
!3083 = !DILocation(line: 1370, column: 5, scope: !3010)
!3084 = !DILocation(line: 1372, column: 39, scope: !3010)
!3085 = !DILocation(line: 1372, column: 42, scope: !3010)
!3086 = !DILocation(line: 1372, column: 41, scope: !3010)
!3087 = !DILocation(line: 1372, column: 45, scope: !3010)
!3088 = !DILocation(line: 1372, column: 44, scope: !3010)
!3089 = !DILocation(line: 1372, column: 38, scope: !3010)
!3090 = !DILocation(line: 1372, column: 29, scope: !3010)
!3091 = !DILocation(line: 247, column: 19, scope: !3014, inlinedAt: !3017)
!3092 = !DILocation(line: 247, column: 10, scope: !3014, inlinedAt: !3017)
!3093 = !DILocation(line: 1372, column: 2, scope: !3010)
!3094 = !DILocation(line: 1372, column: 10, scope: !3010)
!3095 = !DILocation(line: 1372, column: 21, scope: !3010)
!3096 = !DILocation(line: 1373, column: 1, scope: !3010)
!3097 = distinct !DISubprogram(name: "compute_initial_conditions_gpu_kernel", linkageName: "_Z37compute_initial_conditions_gpu_kernelP8dcomplexPd", scope: !6, file: !6, line: 1404, type: !3098, scopeLine: 1405, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !7)
!3098 = !DISubroutineType(types: !3099)
!3099 = !{null, !9, !17}
!3100 = !DILocalVariable(name: "u0", arg: 1, scope: !3097, file: !6, line: 1404, type: !9)
!3101 = !DILocation(line: 1404, column: 64, scope: !3097)
!3102 = !DILocalVariable(name: "starts", arg: 2, scope: !3097, file: !6, line: 1405, type: !17)
!3103 = !DILocation(line: 1405, column: 10, scope: !3097)
!3104 = !DILocalVariable(name: "z", scope: !3097, file: !6, line: 1406, type: !18)
!3105 = !DILocation(line: 1406, column: 6, scope: !3097)
!3106 = !DILocation(line: 64, column: 3, scope: !812, inlinedAt: !3107)
!3107 = distinct !DILocation(line: 1406, column: 10, scope: !3097)
!3108 = !DILocation(line: 75, column: 3, scope: !850, inlinedAt: !3109)
!3109 = distinct !DILocation(line: 1406, column: 23, scope: !3097)
!3110 = !DILocation(line: 1406, column: 21, scope: !3097)
!3111 = !DILocation(line: 53, column: 3, scope: !896, inlinedAt: !3112)
!3112 = distinct !DILocation(line: 1406, column: 36, scope: !3097)
!3113 = !DILocation(line: 1406, column: 34, scope: !3097)
!3114 = !DILocation(line: 1408, column: 5, scope: !3115)
!3115 = distinct !DILexicalBlock(scope: !3097, file: !6, line: 1408, column: 5)
!3116 = !DILocation(line: 1408, column: 6, scope: !3115)
!3117 = !DILocation(line: 1408, column: 5, scope: !3097)
!3118 = !DILocation(line: 1408, column: 12, scope: !3119)
!3119 = distinct !DILexicalBlock(scope: !3115, file: !6, line: 1408, column: 11)
!3120 = !DILocalVariable(name: "x0", scope: !3097, file: !6, line: 1410, type: !15)
!3121 = !DILocation(line: 1410, column: 9, scope: !3097)
!3122 = !DILocation(line: 1410, column: 14, scope: !3097)
!3123 = !DILocation(line: 1410, column: 21, scope: !3097)
!3124 = !DILocalVariable(name: "y", scope: !3125, file: !6, line: 1411, type: !18)
!3125 = distinct !DILexicalBlock(scope: !3097, file: !6, line: 1411, column: 2)
!3126 = !DILocation(line: 1411, column: 10, scope: !3125)
!3127 = !DILocation(line: 1411, column: 6, scope: !3125)
!3128 = !DILocation(line: 1411, column: 15, scope: !3129)
!3129 = distinct !DILexicalBlock(scope: !3125, file: !6, line: 1411, column: 2)
!3130 = !DILocation(line: 1411, column: 16, scope: !3129)
!3131 = !DILocation(line: 1411, column: 2, scope: !3125)
!3132 = !DILocation(line: 1412, column: 41, scope: !3133)
!3133 = distinct !DILexicalBlock(scope: !3129, file: !6, line: 1411, column: 25)
!3134 = !DILocation(line: 1412, column: 49, scope: !3133)
!3135 = !DILocation(line: 1412, column: 50, scope: !3133)
!3136 = !DILocation(line: 1412, column: 47, scope: !3133)
!3137 = !DILocation(line: 1412, column: 56, scope: !3133)
!3138 = !DILocation(line: 1412, column: 57, scope: !3133)
!3139 = !DILocation(line: 1412, column: 60, scope: !3133)
!3140 = !DILocation(line: 1412, column: 54, scope: !3133)
!3141 = !DILocation(line: 1412, column: 31, scope: !3133)
!3142 = !DILocation(line: 1412, column: 3, scope: !3133)
!3143 = !DILocation(line: 1413, column: 2, scope: !3133)
!3144 = !DILocation(line: 1411, column: 22, scope: !3129)
!3145 = !DILocation(line: 1411, column: 2, scope: !3129)
!3146 = distinct !{!3146, !3131, !3147}
!3147 = !DILocation(line: 1413, column: 2, scope: !3125)
!3148 = !DILocation(line: 1414, column: 1, scope: !3097)
!3149 = distinct !DISubprogram(name: "vranlc_device", linkageName: "_Z13vranlc_deviceiPddS_", scope: !6, file: !6, line: 2022, type: !3150, scopeLine: 2025, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !7)
!3150 = !DISubroutineType(types: !3151)
!3151 = !{null, !18, !17, !15, !17}
!3152 = !DILocalVariable(name: "n", arg: 1, scope: !3149, file: !6, line: 2022, type: !18)
!3153 = !DILocation(line: 2022, column: 35, scope: !3149)
!3154 = !DILocalVariable(name: "x_seed", arg: 2, scope: !3149, file: !6, line: 2023, type: !17)
!3155 = !DILocation(line: 2023, column: 11, scope: !3149)
!3156 = !DILocalVariable(name: "a", arg: 3, scope: !3149, file: !6, line: 2024, type: !15)
!3157 = !DILocation(line: 2024, column: 10, scope: !3149)
!3158 = !DILocalVariable(name: "y", arg: 4, scope: !3149, file: !6, line: 2025, type: !17)
!3159 = !DILocation(line: 2025, column: 10, scope: !3149)
!3160 = !DILocalVariable(name: "i", scope: !3149, file: !6, line: 2026, type: !18)
!3161 = !DILocation(line: 2026, column: 6, scope: !3149)
!3162 = !DILocalVariable(name: "x", scope: !3149, file: !6, line: 2027, type: !15)
!3163 = !DILocation(line: 2027, column: 9, scope: !3149)
!3164 = !DILocalVariable(name: "t1", scope: !3149, file: !6, line: 2027, type: !15)
!3165 = !DILocation(line: 2027, column: 11, scope: !3149)
!3166 = !DILocalVariable(name: "t2", scope: !3149, file: !6, line: 2027, type: !15)
!3167 = !DILocation(line: 2027, column: 14, scope: !3149)
!3168 = !DILocalVariable(name: "t3", scope: !3149, file: !6, line: 2027, type: !15)
!3169 = !DILocation(line: 2027, column: 17, scope: !3149)
!3170 = !DILocalVariable(name: "t4", scope: !3149, file: !6, line: 2027, type: !15)
!3171 = !DILocation(line: 2027, column: 20, scope: !3149)
!3172 = !DILocalVariable(name: "a1", scope: !3149, file: !6, line: 2027, type: !15)
!3173 = !DILocation(line: 2027, column: 23, scope: !3149)
!3174 = !DILocalVariable(name: "a2", scope: !3149, file: !6, line: 2027, type: !15)
!3175 = !DILocation(line: 2027, column: 26, scope: !3149)
!3176 = !DILocalVariable(name: "x1", scope: !3149, file: !6, line: 2027, type: !15)
!3177 = !DILocation(line: 2027, column: 29, scope: !3149)
!3178 = !DILocalVariable(name: "x2", scope: !3149, file: !6, line: 2027, type: !15)
!3179 = !DILocation(line: 2027, column: 32, scope: !3149)
!3180 = !DILocalVariable(name: "z", scope: !3149, file: !6, line: 2027, type: !15)
!3181 = !DILocation(line: 2027, column: 35, scope: !3149)
!3182 = !DILocation(line: 2028, column: 13, scope: !3149)
!3183 = !DILocation(line: 2028, column: 11, scope: !3149)
!3184 = !DILocation(line: 2028, column: 5, scope: !3149)
!3185 = !DILocation(line: 2029, column: 12, scope: !3149)
!3186 = !DILocation(line: 2029, column: 7, scope: !3149)
!3187 = !DILocation(line: 2029, column: 5, scope: !3149)
!3188 = !DILocation(line: 2030, column: 7, scope: !3149)
!3189 = !DILocation(line: 2030, column: 17, scope: !3149)
!3190 = !DILocation(line: 2030, column: 15, scope: !3149)
!3191 = !DILocation(line: 2030, column: 9, scope: !3149)
!3192 = !DILocation(line: 2030, column: 5, scope: !3149)
!3193 = !DILocation(line: 2031, column: 7, scope: !3149)
!3194 = !DILocation(line: 2031, column: 6, scope: !3149)
!3195 = !DILocation(line: 2031, column: 4, scope: !3149)
!3196 = !DILocation(line: 2032, column: 7, scope: !3197)
!3197 = distinct !DILexicalBlock(scope: !3149, file: !6, line: 2032, column: 2)
!3198 = !DILocation(line: 2032, column: 6, scope: !3197)
!3199 = !DILocation(line: 2032, column: 11, scope: !3200)
!3200 = distinct !DILexicalBlock(scope: !3197, file: !6, line: 2032, column: 2)
!3201 = !DILocation(line: 2032, column: 13, scope: !3200)
!3202 = !DILocation(line: 2032, column: 12, scope: !3200)
!3203 = !DILocation(line: 2032, column: 2, scope: !3197)
!3204 = !DILocation(line: 2033, column: 14, scope: !3205)
!3205 = distinct !DILexicalBlock(scope: !3200, file: !6, line: 2032, column: 20)
!3206 = !DILocation(line: 2033, column: 12, scope: !3205)
!3207 = !DILocation(line: 2033, column: 6, scope: !3205)
!3208 = !DILocation(line: 2034, column: 13, scope: !3205)
!3209 = !DILocation(line: 2034, column: 8, scope: !3205)
!3210 = !DILocation(line: 2034, column: 6, scope: !3205)
!3211 = !DILocation(line: 2035, column: 8, scope: !3205)
!3212 = !DILocation(line: 2035, column: 18, scope: !3205)
!3213 = !DILocation(line: 2035, column: 16, scope: !3205)
!3214 = !DILocation(line: 2035, column: 10, scope: !3205)
!3215 = !DILocation(line: 2035, column: 6, scope: !3205)
!3216 = !DILocation(line: 2036, column: 8, scope: !3205)
!3217 = !DILocation(line: 2036, column: 13, scope: !3205)
!3218 = !DILocation(line: 2036, column: 11, scope: !3205)
!3219 = !DILocation(line: 2036, column: 18, scope: !3205)
!3220 = !DILocation(line: 2036, column: 23, scope: !3205)
!3221 = !DILocation(line: 2036, column: 21, scope: !3205)
!3222 = !DILocation(line: 2036, column: 16, scope: !3205)
!3223 = !DILocation(line: 2036, column: 6, scope: !3205)
!3224 = !DILocation(line: 2037, column: 20, scope: !3205)
!3225 = !DILocation(line: 2037, column: 18, scope: !3205)
!3226 = !DILocation(line: 2037, column: 13, scope: !3205)
!3227 = !DILocation(line: 2037, column: 8, scope: !3205)
!3228 = !DILocation(line: 2037, column: 6, scope: !3205)
!3229 = !DILocation(line: 2038, column: 7, scope: !3205)
!3230 = !DILocation(line: 2038, column: 18, scope: !3205)
!3231 = !DILocation(line: 2038, column: 16, scope: !3205)
!3232 = !DILocation(line: 2038, column: 10, scope: !3205)
!3233 = !DILocation(line: 2038, column: 5, scope: !3205)
!3234 = !DILocation(line: 2039, column: 14, scope: !3205)
!3235 = !DILocation(line: 2039, column: 12, scope: !3205)
!3236 = !DILocation(line: 2039, column: 18, scope: !3205)
!3237 = !DILocation(line: 2039, column: 23, scope: !3205)
!3238 = !DILocation(line: 2039, column: 21, scope: !3205)
!3239 = !DILocation(line: 2039, column: 16, scope: !3205)
!3240 = !DILocation(line: 2039, column: 6, scope: !3205)
!3241 = !DILocation(line: 2040, column: 20, scope: !3205)
!3242 = !DILocation(line: 2040, column: 18, scope: !3205)
!3243 = !DILocation(line: 2040, column: 13, scope: !3205)
!3244 = !DILocation(line: 2040, column: 8, scope: !3205)
!3245 = !DILocation(line: 2040, column: 6, scope: !3205)
!3246 = !DILocation(line: 2041, column: 7, scope: !3205)
!3247 = !DILocation(line: 2041, column: 18, scope: !3205)
!3248 = !DILocation(line: 2041, column: 16, scope: !3205)
!3249 = !DILocation(line: 2041, column: 10, scope: !3205)
!3250 = !DILocation(line: 2041, column: 5, scope: !3205)
!3251 = !DILocation(line: 2042, column: 16, scope: !3205)
!3252 = !DILocation(line: 2042, column: 14, scope: !3205)
!3253 = !DILocation(line: 2042, column: 3, scope: !3205)
!3254 = !DILocation(line: 2042, column: 5, scope: !3205)
!3255 = !DILocation(line: 2042, column: 8, scope: !3205)
!3256 = !DILocation(line: 2043, column: 2, scope: !3205)
!3257 = !DILocation(line: 2032, column: 17, scope: !3200)
!3258 = !DILocation(line: 2032, column: 2, scope: !3200)
!3259 = distinct !{!3259, !3203, !3260}
!3260 = !DILocation(line: 2043, column: 2, scope: !3197)
!3261 = !DILocation(line: 2044, column: 12, scope: !3149)
!3262 = !DILocation(line: 2044, column: 3, scope: !3149)
!3263 = !DILocation(line: 2044, column: 10, scope: !3149)
!3264 = !DILocation(line: 2045, column: 1, scope: !3149)
!3265 = distinct !DISubprogram(name: "evolve_gpu_kernel", linkageName: "_Z17evolve_gpu_kernelP8dcomplexS0_Pd", scope: !6, file: !6, line: 1432, type: !3266, scopeLine: 1434, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !7)
!3266 = !DISubroutineType(types: !3267)
!3267 = !{null, !9, !9, !17}
!3268 = !DILocalVariable(name: "u0", arg: 1, scope: !3265, file: !6, line: 1432, type: !9)
!3269 = !DILocation(line: 1432, column: 44, scope: !3265)
!3270 = !DILocalVariable(name: "u1", arg: 2, scope: !3265, file: !6, line: 1433, type: !9)
!3271 = !DILocation(line: 1433, column: 12, scope: !3265)
!3272 = !DILocalVariable(name: "twiddle", arg: 3, scope: !3265, file: !6, line: 1434, type: !17)
!3273 = !DILocation(line: 1434, column: 10, scope: !3265)
!3274 = !DILocalVariable(name: "thread_id", scope: !3265, file: !6, line: 1435, type: !18)
!3275 = !DILocation(line: 1435, column: 6, scope: !3265)
!3276 = !DILocation(line: 64, column: 3, scope: !812, inlinedAt: !3277)
!3277 = distinct !DILocation(line: 1435, column: 18, scope: !3265)
!3278 = !DILocation(line: 75, column: 3, scope: !850, inlinedAt: !3279)
!3279 = distinct !DILocation(line: 1435, column: 31, scope: !3265)
!3280 = !DILocation(line: 1435, column: 29, scope: !3265)
!3281 = !DILocation(line: 53, column: 3, scope: !896, inlinedAt: !3282)
!3282 = distinct !DILocation(line: 1435, column: 44, scope: !3265)
!3283 = !DILocation(line: 1435, column: 42, scope: !3265)
!3284 = !DILocation(line: 1437, column: 5, scope: !3285)
!3285 = distinct !DILexicalBlock(scope: !3265, file: !6, line: 1437, column: 5)
!3286 = !DILocation(line: 1437, column: 14, scope: !3285)
!3287 = !DILocation(line: 1437, column: 5, scope: !3265)
!3288 = !DILocation(line: 1438, column: 3, scope: !3289)
!3289 = distinct !DILexicalBlock(scope: !3285, file: !6, line: 1437, column: 27)
!3290 = !DILocation(line: 1441, column: 18, scope: !3265)
!3291 = !DILocation(line: 1441, column: 2, scope: !3265)
!3292 = !DILocation(line: 1441, column: 5, scope: !3265)
!3293 = !DILocation(line: 1441, column: 16, scope: !3265)
!3294 = !DILocation(line: 1442, column: 18, scope: !3265)
!3295 = !DILocation(line: 1442, column: 21, scope: !3265)
!3296 = !DILocation(line: 1442, column: 2, scope: !3265)
!3297 = !DILocation(line: 1442, column: 5, scope: !3265)
!3298 = !DILocation(line: 1442, column: 16, scope: !3265)
!3299 = !DILocation(line: 1443, column: 1, scope: !3265)
!3300 = distinct !DISubprogram(name: "init_ui_gpu_kernel", linkageName: "_Z18init_ui_gpu_kernelP8dcomplexS0_Pd", scope: !6, file: !6, line: 1542, type: !3266, scopeLine: 1544, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !7)
!3301 = !DILocalVariable(name: "u0", arg: 1, scope: !3300, file: !6, line: 1542, type: !9)
!3302 = !DILocation(line: 1542, column: 45, scope: !3300)
!3303 = !DILocalVariable(name: "u1", arg: 2, scope: !3300, file: !6, line: 1543, type: !9)
!3304 = !DILocation(line: 1543, column: 12, scope: !3300)
!3305 = !DILocalVariable(name: "twiddle", arg: 3, scope: !3300, file: !6, line: 1544, type: !17)
!3306 = !DILocation(line: 1544, column: 10, scope: !3300)
!3307 = !DILocalVariable(name: "thread_id", scope: !3300, file: !6, line: 1545, type: !18)
!3308 = !DILocation(line: 1545, column: 6, scope: !3300)
!3309 = !DILocation(line: 64, column: 3, scope: !812, inlinedAt: !3310)
!3310 = distinct !DILocation(line: 1545, column: 18, scope: !3300)
!3311 = !DILocation(line: 75, column: 3, scope: !850, inlinedAt: !3312)
!3312 = distinct !DILocation(line: 1545, column: 31, scope: !3300)
!3313 = !DILocation(line: 1545, column: 29, scope: !3300)
!3314 = !DILocation(line: 53, column: 3, scope: !896, inlinedAt: !3315)
!3315 = distinct !DILocation(line: 1545, column: 44, scope: !3300)
!3316 = !DILocation(line: 1545, column: 42, scope: !3300)
!3317 = !DILocation(line: 1547, column: 5, scope: !3318)
!3318 = distinct !DILexicalBlock(scope: !3300, file: !6, line: 1547, column: 5)
!3319 = !DILocation(line: 1547, column: 14, scope: !3318)
!3320 = !DILocation(line: 1547, column: 5, scope: !3300)
!3321 = !DILocation(line: 1548, column: 3, scope: !3322)
!3322 = distinct !DILexicalBlock(scope: !3318, file: !6, line: 1547, column: 23)
!3323 = !DILocation(line: 1551, column: 18, scope: !3300)
!3324 = !DILocation(line: 1551, column: 2, scope: !3300)
!3325 = !DILocation(line: 1551, column: 5, scope: !3300)
!3326 = !DILocation(line: 1551, column: 16, scope: !3300)
!3327 = !DILocation(line: 1552, column: 18, scope: !3300)
!3328 = !DILocation(line: 1552, column: 2, scope: !3300)
!3329 = !DILocation(line: 1552, column: 5, scope: !3300)
!3330 = !DILocation(line: 1552, column: 16, scope: !3300)
!3331 = !DILocation(line: 1553, column: 2, scope: !3300)
!3332 = !DILocation(line: 1553, column: 10, scope: !3300)
!3333 = !DILocation(line: 1553, column: 21, scope: !3300)
!3334 = !DILocation(line: 1554, column: 1, scope: !3300)
!3335 = distinct !DISubprogram(name: "ipow46_device", linkageName: "_Z13ipow46_devicediPd", scope: !6, file: !6, line: 1587, type: !3336, scopeLine: 1589, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !7)
!3336 = !DISubroutineType(types: !3337)
!3337 = !{null, !15, !18, !17}
!3338 = !DILocalVariable(name: "a", arg: 1, scope: !3335, file: !6, line: 1587, type: !15)
!3339 = !DILocation(line: 1587, column: 38, scope: !3335)
!3340 = !DILocalVariable(name: "exponent", arg: 2, scope: !3335, file: !6, line: 1588, type: !18)
!3341 = !DILocation(line: 1588, column: 7, scope: !3335)
!3342 = !DILocalVariable(name: "result", arg: 3, scope: !3335, file: !6, line: 1589, type: !17)
!3343 = !DILocation(line: 1589, column: 11, scope: !3335)
!3344 = !DILocalVariable(name: "q", scope: !3335, file: !6, line: 1590, type: !15)
!3345 = !DILocation(line: 1590, column: 9, scope: !3335)
!3346 = !DILocalVariable(name: "r", scope: !3335, file: !6, line: 1590, type: !15)
!3347 = !DILocation(line: 1590, column: 12, scope: !3335)
!3348 = !DILocalVariable(name: "n", scope: !3335, file: !6, line: 1591, type: !18)
!3349 = !DILocation(line: 1591, column: 6, scope: !3335)
!3350 = !DILocalVariable(name: "n2", scope: !3335, file: !6, line: 1591, type: !18)
!3351 = !DILocation(line: 1591, column: 9, scope: !3335)
!3352 = !DILocation(line: 1599, column: 3, scope: !3335)
!3353 = !DILocation(line: 1599, column: 10, scope: !3335)
!3354 = !DILocation(line: 1600, column: 5, scope: !3355)
!3355 = distinct !DILexicalBlock(scope: !3335, file: !6, line: 1600, column: 5)
!3356 = !DILocation(line: 1600, column: 13, scope: !3355)
!3357 = !DILocation(line: 1600, column: 5, scope: !3335)
!3358 = !DILocation(line: 1600, column: 18, scope: !3359)
!3359 = distinct !DILexicalBlock(scope: !3355, file: !6, line: 1600, column: 17)
!3360 = !DILocation(line: 1601, column: 6, scope: !3335)
!3361 = !DILocation(line: 1601, column: 4, scope: !3335)
!3362 = !DILocation(line: 1602, column: 4, scope: !3335)
!3363 = !DILocation(line: 1603, column: 6, scope: !3335)
!3364 = !DILocation(line: 1603, column: 4, scope: !3335)
!3365 = !DILocation(line: 1604, column: 2, scope: !3335)
!3366 = !DILocation(line: 1604, column: 8, scope: !3335)
!3367 = !DILocation(line: 1604, column: 9, scope: !3335)
!3368 = !DILocation(line: 1605, column: 8, scope: !3369)
!3369 = distinct !DILexicalBlock(scope: !3335, file: !6, line: 1604, column: 12)
!3370 = !DILocation(line: 1605, column: 9, scope: !3369)
!3371 = !DILocation(line: 1605, column: 6, scope: !3369)
!3372 = !DILocation(line: 1606, column: 6, scope: !3373)
!3373 = distinct !DILexicalBlock(scope: !3369, file: !6, line: 1606, column: 6)
!3374 = !DILocation(line: 1606, column: 8, scope: !3373)
!3375 = !DILocation(line: 1606, column: 12, scope: !3373)
!3376 = !DILocation(line: 1606, column: 10, scope: !3373)
!3377 = !DILocation(line: 1606, column: 6, scope: !3369)
!3378 = !DILocation(line: 1607, column: 22, scope: !3379)
!3379 = distinct !DILexicalBlock(scope: !3373, file: !6, line: 1606, column: 14)
!3380 = !DILocation(line: 1607, column: 4, scope: !3379)
!3381 = !DILocation(line: 1608, column: 8, scope: !3379)
!3382 = !DILocation(line: 1608, column: 6, scope: !3379)
!3383 = !DILocation(line: 1609, column: 3, scope: !3379)
!3384 = !DILocation(line: 1610, column: 22, scope: !3385)
!3385 = distinct !DILexicalBlock(scope: !3373, file: !6, line: 1609, column: 8)
!3386 = !DILocation(line: 1610, column: 4, scope: !3385)
!3387 = !DILocation(line: 1611, column: 8, scope: !3385)
!3388 = !DILocation(line: 1611, column: 9, scope: !3385)
!3389 = !DILocation(line: 1611, column: 6, scope: !3385)
!3390 = distinct !{!3390, !3365, !3391}
!3391 = !DILocation(line: 1613, column: 2, scope: !3335)
!3392 = !DILocation(line: 1614, column: 20, scope: !3335)
!3393 = !DILocation(line: 1614, column: 2, scope: !3335)
!3394 = !DILocation(line: 1615, column: 12, scope: !3335)
!3395 = !DILocation(line: 1615, column: 3, scope: !3335)
!3396 = !DILocation(line: 1615, column: 10, scope: !3335)
!3397 = !DILocation(line: 1616, column: 1, scope: !3335)
!3398 = distinct !DISubprogram(name: "randlc_device", linkageName: "_Z13randlc_devicePdd", scope: !6, file: !6, line: 1618, type: !2927, scopeLine: 1619, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !7)
!3399 = !DILocalVariable(name: "x", arg: 1, scope: !3398, file: !6, line: 1618, type: !17)
!3400 = !DILocation(line: 1618, column: 41, scope: !3398)
!3401 = !DILocalVariable(name: "a", arg: 2, scope: !3398, file: !6, line: 1619, type: !15)
!3402 = !DILocation(line: 1619, column: 10, scope: !3398)
!3403 = !DILocalVariable(name: "t1", scope: !3398, file: !6, line: 1620, type: !15)
!3404 = !DILocation(line: 1620, column: 9, scope: !3398)
!3405 = !DILocalVariable(name: "t2", scope: !3398, file: !6, line: 1620, type: !15)
!3406 = !DILocation(line: 1620, column: 12, scope: !3398)
!3407 = !DILocalVariable(name: "t3", scope: !3398, file: !6, line: 1620, type: !15)
!3408 = !DILocation(line: 1620, column: 15, scope: !3398)
!3409 = !DILocalVariable(name: "t4", scope: !3398, file: !6, line: 1620, type: !15)
!3410 = !DILocation(line: 1620, column: 18, scope: !3398)
!3411 = !DILocalVariable(name: "a1", scope: !3398, file: !6, line: 1620, type: !15)
!3412 = !DILocation(line: 1620, column: 21, scope: !3398)
!3413 = !DILocalVariable(name: "a2", scope: !3398, file: !6, line: 1620, type: !15)
!3414 = !DILocation(line: 1620, column: 24, scope: !3398)
!3415 = !DILocalVariable(name: "x1", scope: !3398, file: !6, line: 1620, type: !15)
!3416 = !DILocation(line: 1620, column: 27, scope: !3398)
!3417 = !DILocalVariable(name: "x2", scope: !3398, file: !6, line: 1620, type: !15)
!3418 = !DILocation(line: 1620, column: 30, scope: !3398)
!3419 = !DILocalVariable(name: "z", scope: !3398, file: !6, line: 1620, type: !15)
!3420 = !DILocation(line: 1620, column: 33, scope: !3398)
!3421 = !DILocation(line: 1621, column: 13, scope: !3398)
!3422 = !DILocation(line: 1621, column: 11, scope: !3398)
!3423 = !DILocation(line: 1621, column: 5, scope: !3398)
!3424 = !DILocation(line: 1622, column: 12, scope: !3398)
!3425 = !DILocation(line: 1622, column: 7, scope: !3398)
!3426 = !DILocation(line: 1622, column: 5, scope: !3398)
!3427 = !DILocation(line: 1623, column: 7, scope: !3398)
!3428 = !DILocation(line: 1623, column: 17, scope: !3398)
!3429 = !DILocation(line: 1623, column: 15, scope: !3398)
!3430 = !DILocation(line: 1623, column: 9, scope: !3398)
!3431 = !DILocation(line: 1623, column: 5, scope: !3398)
!3432 = !DILocation(line: 1624, column: 15, scope: !3398)
!3433 = !DILocation(line: 1624, column: 14, scope: !3398)
!3434 = !DILocation(line: 1624, column: 11, scope: !3398)
!3435 = !DILocation(line: 1624, column: 5, scope: !3398)
!3436 = !DILocation(line: 1625, column: 12, scope: !3398)
!3437 = !DILocation(line: 1625, column: 7, scope: !3398)
!3438 = !DILocation(line: 1625, column: 5, scope: !3398)
!3439 = !DILocation(line: 1626, column: 9, scope: !3398)
!3440 = !DILocation(line: 1626, column: 8, scope: !3398)
!3441 = !DILocation(line: 1626, column: 20, scope: !3398)
!3442 = !DILocation(line: 1626, column: 18, scope: !3398)
!3443 = !DILocation(line: 1626, column: 12, scope: !3398)
!3444 = !DILocation(line: 1626, column: 5, scope: !3398)
!3445 = !DILocation(line: 1627, column: 7, scope: !3398)
!3446 = !DILocation(line: 1627, column: 12, scope: !3398)
!3447 = !DILocation(line: 1627, column: 10, scope: !3398)
!3448 = !DILocation(line: 1627, column: 17, scope: !3398)
!3449 = !DILocation(line: 1627, column: 22, scope: !3398)
!3450 = !DILocation(line: 1627, column: 20, scope: !3398)
!3451 = !DILocation(line: 1627, column: 15, scope: !3398)
!3452 = !DILocation(line: 1627, column: 5, scope: !3398)
!3453 = !DILocation(line: 1628, column: 19, scope: !3398)
!3454 = !DILocation(line: 1628, column: 17, scope: !3398)
!3455 = !DILocation(line: 1628, column: 12, scope: !3398)
!3456 = !DILocation(line: 1628, column: 7, scope: !3398)
!3457 = !DILocation(line: 1628, column: 5, scope: !3398)
!3458 = !DILocation(line: 1629, column: 6, scope: !3398)
!3459 = !DILocation(line: 1629, column: 17, scope: !3398)
!3460 = !DILocation(line: 1629, column: 15, scope: !3398)
!3461 = !DILocation(line: 1629, column: 9, scope: !3398)
!3462 = !DILocation(line: 1629, column: 4, scope: !3398)
!3463 = !DILocation(line: 1630, column: 13, scope: !3398)
!3464 = !DILocation(line: 1630, column: 11, scope: !3398)
!3465 = !DILocation(line: 1630, column: 17, scope: !3398)
!3466 = !DILocation(line: 1630, column: 22, scope: !3398)
!3467 = !DILocation(line: 1630, column: 20, scope: !3398)
!3468 = !DILocation(line: 1630, column: 15, scope: !3398)
!3469 = !DILocation(line: 1630, column: 5, scope: !3398)
!3470 = !DILocation(line: 1631, column: 19, scope: !3398)
!3471 = !DILocation(line: 1631, column: 17, scope: !3398)
!3472 = !DILocation(line: 1631, column: 12, scope: !3398)
!3473 = !DILocation(line: 1631, column: 7, scope: !3398)
!3474 = !DILocation(line: 1631, column: 5, scope: !3398)
!3475 = !DILocation(line: 1632, column: 9, scope: !3398)
!3476 = !DILocation(line: 1632, column: 20, scope: !3398)
!3477 = !DILocation(line: 1632, column: 18, scope: !3398)
!3478 = !DILocation(line: 1632, column: 12, scope: !3398)
!3479 = !DILocation(line: 1632, column: 4, scope: !3398)
!3480 = !DILocation(line: 1632, column: 7, scope: !3398)
!3481 = !DILocation(line: 1633, column: 18, scope: !3398)
!3482 = !DILocation(line: 1633, column: 17, scope: !3398)
!3483 = !DILocation(line: 1633, column: 14, scope: !3398)
!3484 = !DILocation(line: 1633, column: 2, scope: !3398)
!3485 = distinct !DISubprogram(name: "atomicCAS", linkageName: "_ZL9atomicCASPyyy", scope: !3486, file: !3486, line: 211, type: !3487, scopeLine: 212, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !5, retainedNodes: !7)
!3486 = !DIFile(filename: "/usr/local/cuda/include/device_atomic_functions.hpp", directory: "")
!3487 = !DISubroutineType(types: !3488)
!3488 = !{!20, !19, !20, !20}
!3489 = !DILocalVariable(name: "p", arg: 1, scope: !3490, file: !511, line: 1655, type: !19)
!3490 = distinct !DISubprogram(name: "__ullAtomicCAS", linkageName: "_ZL14__ullAtomicCASPyyy", scope: !511, file: !511, line: 1655, type: !3487, scopeLine: 1658, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !5, retainedNodes: !7)
!3491 = !DILocation(line: 1655, column: 63, scope: !3490, inlinedAt: !3492)
!3492 = distinct !DILocation(line: 213, column: 10, scope: !3485)
!3493 = !DILocalVariable(name: "compare", arg: 2, scope: !3490, file: !511, line: 1656, type: !20)
!3494 = !DILocation(line: 1656, column: 62, scope: !3490, inlinedAt: !3492)
!3495 = !DILocalVariable(name: "val", arg: 3, scope: !3490, file: !511, line: 1657, type: !20)
!3496 = !DILocation(line: 1657, column: 62, scope: !3490, inlinedAt: !3492)
!3497 = !DILocalVariable(name: "address", arg: 1, scope: !3485, file: !3486, line: 211, type: !19)
!3498 = !DILocation(line: 211, column: 91, scope: !3485)
!3499 = !DILocalVariable(name: "compare", arg: 2, scope: !3485, file: !3486, line: 211, type: !20)
!3500 = !DILocation(line: 211, column: 123, scope: !3485)
!3501 = !DILocalVariable(name: "val", arg: 3, scope: !3485, file: !3486, line: 211, type: !20)
!3502 = !DILocation(line: 211, column: 155, scope: !3485)
!3503 = !DILocation(line: 213, column: 25, scope: !3485)
!3504 = !DILocation(line: 213, column: 34, scope: !3485)
!3505 = !DILocation(line: 213, column: 43, scope: !3485)
!3506 = !DILocation(line: 1660, column: 78, scope: !3490, inlinedAt: !3492)
!3507 = !DILocation(line: 1661, column: 67, scope: !3490, inlinedAt: !3492)
!3508 = !DILocation(line: 1662, column: 67, scope: !3490, inlinedAt: !3492)
!3509 = !DILocation(line: 1660, column: 29, scope: !3490, inlinedAt: !3492)
!3510 = !DILocation(line: 213, column: 3, scope: !3485)
