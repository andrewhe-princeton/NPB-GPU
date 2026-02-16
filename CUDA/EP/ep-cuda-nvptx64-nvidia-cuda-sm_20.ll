; ModuleID = 'ep.cu'
source_filename = "ep.cu"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

%struct.__cuda_builtin_blockIdx_t = type { i8 }
%struct.__cuda_builtin_blockDim_t = type { i8 }
%struct.__cuda_builtin_threadIdx_t = type { i8 }

@blockIdx = extern_weak dso_local addrspace(1) global %struct.__cuda_builtin_blockIdx_t, align 1
@blockDim = extern_weak dso_local addrspace(1) global %struct.__cuda_builtin_blockDim_t, align 1
@threadIdx = extern_weak dso_local addrspace(1) global %struct.__cuda_builtin_threadIdx_t, align 1

; Function Attrs: convergent noinline nounwind
define dso_local void @_Z10gpu_kernelPdS_S_d(double* %q_global, double* %sx_global, double* %sy_global, double %an) #0 !dbg !782 {
entry:
  %f.addr.i143 = alloca double, align 8
  call void @llvm.dbg.declare(metadata double* %f.addr.i143, metadata !785, metadata !DIExpression()), !dbg !787
  %f.addr.i142 = alloca double, align 8
  call void @llvm.dbg.declare(metadata double* %f.addr.i142, metadata !785, metadata !DIExpression()), !dbg !797
  %f.addr.i141 = alloca double, align 8
  call void @llvm.dbg.declare(metadata double* %f.addr.i141, metadata !785, metadata !DIExpression()), !dbg !799
  %f.addr.i = alloca double, align 8
  call void @llvm.dbg.declare(metadata double* %f.addr.i, metadata !785, metadata !DIExpression()), !dbg !801
  %x.addr.i = alloca double, align 8
  call void @llvm.dbg.declare(metadata double* %x.addr.i, metadata !803, metadata !DIExpression()), !dbg !805
  %a.addr.i = alloca double, align 8
  call void @llvm.dbg.declare(metadata double* %a.addr.i, metadata !807, metadata !DIExpression()), !dbg !810
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
  call void @llvm.dbg.declare(metadata double** %q_global.addr, metadata !812, metadata !DIExpression()), !dbg !813
  store double* %sx_global, double** %sx_global.addr, align 8
  call void @llvm.dbg.declare(metadata double** %sx_global.addr, metadata !814, metadata !DIExpression()), !dbg !815
  store double* %sy_global, double** %sy_global.addr, align 8
  call void @llvm.dbg.declare(metadata double** %sy_global.addr, metadata !816, metadata !DIExpression()), !dbg !817
  store double %an, double* %an.addr, align 8
  call void @llvm.dbg.declare(metadata double* %an.addr, metadata !818, metadata !DIExpression()), !dbg !819
  call void @llvm.dbg.declare(metadata [256 x double]* %x_local, metadata !820, metadata !DIExpression()), !dbg !824
  call void @llvm.dbg.declare(metadata [10 x double]* %q_local, metadata !825, metadata !DIExpression()), !dbg !829
  call void @llvm.dbg.declare(metadata double* %sx_local, metadata !830, metadata !DIExpression()), !dbg !831
  call void @llvm.dbg.declare(metadata double* %sy_local, metadata !832, metadata !DIExpression()), !dbg !833
  call void @llvm.dbg.declare(metadata double* %t1, metadata !834, metadata !DIExpression()), !dbg !835
  call void @llvm.dbg.declare(metadata double* %t2, metadata !836, metadata !DIExpression()), !dbg !837
  call void @llvm.dbg.declare(metadata double* %t3, metadata !838, metadata !DIExpression()), !dbg !839
  call void @llvm.dbg.declare(metadata double* %t4, metadata !840, metadata !DIExpression()), !dbg !841
  call void @llvm.dbg.declare(metadata double* %x1, metadata !842, metadata !DIExpression()), !dbg !843
  call void @llvm.dbg.declare(metadata double* %x2, metadata !844, metadata !DIExpression()), !dbg !845
  call void @llvm.dbg.declare(metadata double* %seed, metadata !846, metadata !DIExpression()), !dbg !847
  call void @llvm.dbg.declare(metadata i32* %i, metadata !848, metadata !DIExpression()), !dbg !849
  call void @llvm.dbg.declare(metadata i32* %ii, metadata !850, metadata !DIExpression()), !dbg !851
  call void @llvm.dbg.declare(metadata i32* %ik, metadata !852, metadata !DIExpression()), !dbg !853
  call void @llvm.dbg.declare(metadata i32* %kk, metadata !854, metadata !DIExpression()), !dbg !855
  call void @llvm.dbg.declare(metadata i32* %l, metadata !856, metadata !DIExpression()), !dbg !857
  %arrayidx = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 0, !dbg !858
  store double 0.000000e+00, double* %arrayidx, align 8, !dbg !859
  %arrayidx1 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 1, !dbg !860
  store double 0.000000e+00, double* %arrayidx1, align 8, !dbg !861
  %arrayidx2 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 2, !dbg !862
  store double 0.000000e+00, double* %arrayidx2, align 8, !dbg !863
  %arrayidx3 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 3, !dbg !864
  store double 0.000000e+00, double* %arrayidx3, align 8, !dbg !865
  %arrayidx4 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 4, !dbg !866
  store double 0.000000e+00, double* %arrayidx4, align 8, !dbg !867
  %arrayidx5 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 5, !dbg !868
  store double 0.000000e+00, double* %arrayidx5, align 8, !dbg !869
  %arrayidx6 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 6, !dbg !870
  store double 0.000000e+00, double* %arrayidx6, align 8, !dbg !871
  %arrayidx7 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 7, !dbg !872
  store double 0.000000e+00, double* %arrayidx7, align 8, !dbg !873
  %arrayidx8 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 8, !dbg !874
  store double 0.000000e+00, double* %arrayidx8, align 8, !dbg !875
  %arrayidx9 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 9, !dbg !876
  store double 0.000000e+00, double* %arrayidx9, align 8, !dbg !877
  store double 0.000000e+00, double* %sx_local, align 8, !dbg !878
  store double 0.000000e+00, double* %sy_local, align 8, !dbg !879
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #5, !dbg !880, !range !917
  %1 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #5, !dbg !918, !range !962
  %mul = mul i32 %0, %1, !dbg !963
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #5, !dbg !964, !range !992
  %add = add i32 %mul, %2, !dbg !993
  store i32 %add, i32* %kk, align 4, !dbg !994
  %3 = load i32, i32* %kk, align 4, !dbg !995
  %cmp = icmp sge i32 %3, 4096, !dbg !997
  br i1 %cmp, label %if.then, label %if.end, !dbg !998

if.then:                                          ; preds = %entry
  br label %return, !dbg !999

if.end:                                           ; preds = %entry
  store double 0x41B033C4D7000000, double* %t1, align 8, !dbg !1001
  %4 = load double, double* %an.addr, align 8, !dbg !1002
  store double %4, double* %t2, align 8, !dbg !1003
  store i32 1, i32* %i, align 4, !dbg !1004
  br label %for.cond, !dbg !1006

for.cond:                                         ; preds = %for.inc, %if.end
  %5 = load i32, i32* %i, align 4, !dbg !1007
  %cmp12 = icmp sle i32 %5, 100, !dbg !1009
  br i1 %cmp12, label %for.body, label %for.end, !dbg !1010

for.body:                                         ; preds = %for.cond
  %6 = load i32, i32* %kk, align 4, !dbg !1011
  %div = sdiv i32 %6, 2, !dbg !1013
  store i32 %div, i32* %ik, align 4, !dbg !1014
  %7 = load i32, i32* %ik, align 4, !dbg !1015
  %mul13 = mul nsw i32 2, %7, !dbg !1017
  %8 = load i32, i32* %kk, align 4, !dbg !1018
  %cmp14 = icmp ne i32 %mul13, %8, !dbg !1019
  br i1 %cmp14, label %if.then15, label %if.end17, !dbg !1020

if.then15:                                        ; preds = %for.body
  %9 = load double, double* %t2, align 8, !dbg !1021
  %call16 = call double @_Z13randlc_devicePdd(double* %t1, double %9) #6, !dbg !1023
  store double %call16, double* %t3, align 8, !dbg !1024
  br label %if.end17, !dbg !1025

if.end17:                                         ; preds = %if.then15, %for.body
  %10 = load i32, i32* %ik, align 4, !dbg !1026
  %cmp18 = icmp eq i32 %10, 0, !dbg !1028
  br i1 %cmp18, label %if.then19, label %if.end20, !dbg !1029

if.then19:                                        ; preds = %if.end17
  br label %for.end, !dbg !1030

if.end20:                                         ; preds = %if.end17
  %11 = load double, double* %t2, align 8, !dbg !1032
  %call21 = call double @_Z13randlc_devicePdd(double* %t2, double %11) #6, !dbg !1033
  store double %call21, double* %t3, align 8, !dbg !1034
  %12 = load i32, i32* %ik, align 4, !dbg !1035
  store i32 %12, i32* %kk, align 4, !dbg !1036
  br label %for.inc, !dbg !1037

for.inc:                                          ; preds = %if.end20
  %13 = load i32, i32* %i, align 4, !dbg !1038
  %inc = add nsw i32 %13, 1, !dbg !1038
  store i32 %inc, i32* %i, align 4, !dbg !1038
  br label %for.cond, !dbg !1039, !llvm.loop !1040

for.end:                                          ; preds = %if.then19, %for.cond
  %14 = load double, double* %t1, align 8, !dbg !1042
  store double %14, double* %seed, align 8, !dbg !1043
  store i32 0, i32* %ii, align 4, !dbg !1044
  br label %for.cond22, !dbg !1045

for.cond22:                                       ; preds = %for.inc62, %for.end
  %15 = load i32, i32* %ii, align 4, !dbg !1046
  %cmp23 = icmp slt i32 %15, 65536, !dbg !1047
  br i1 %cmp23, label %for.body24, label %for.end64, !dbg !1048

for.body24:                                       ; preds = %for.cond22
  %arraydecay = getelementptr inbounds [256 x double], [256 x double]* %x_local, i64 0, i64 0, !dbg !1049
  call void @_Z13vranlc_deviceiPddS_(i32 256, double* %seed, double 0x41D2309CE5400000, double* %arraydecay) #6, !dbg !1050
  store i32 0, i32* %i, align 4, !dbg !1051
  br label %for.cond25, !dbg !1052

for.cond25:                                       ; preds = %for.inc59, %for.body24
  %16 = load i32, i32* %i, align 4, !dbg !1053
  %cmp26 = icmp slt i32 %16, 128, !dbg !1054
  br i1 %cmp26, label %for.body27, label %for.end61, !dbg !1055

for.body27:                                       ; preds = %for.cond25
  %17 = load i32, i32* %i, align 4, !dbg !1056
  %mul28 = mul nsw i32 2, %17, !dbg !1057
  %idxprom = sext i32 %mul28 to i64, !dbg !1058
  %arrayidx29 = getelementptr inbounds [256 x double], [256 x double]* %x_local, i64 0, i64 %idxprom, !dbg !1058
  %18 = load double, double* %arrayidx29, align 8, !dbg !1058
  %mul30 = fmul contract double 2.000000e+00, %18, !dbg !1059
  %sub = fsub contract double %mul30, 1.000000e+00, !dbg !1060
  store double %sub, double* %x1, align 8, !dbg !1061
  %19 = load i32, i32* %i, align 4, !dbg !1062
  %mul31 = mul nsw i32 2, %19, !dbg !1063
  %add32 = add nsw i32 %mul31, 1, !dbg !1064
  %idxprom33 = sext i32 %add32 to i64, !dbg !1065
  %arrayidx34 = getelementptr inbounds [256 x double], [256 x double]* %x_local, i64 0, i64 %idxprom33, !dbg !1065
  %20 = load double, double* %arrayidx34, align 8, !dbg !1065
  %mul35 = fmul contract double 2.000000e+00, %20, !dbg !1066
  %sub36 = fsub contract double %mul35, 1.000000e+00, !dbg !1067
  store double %sub36, double* %x2, align 8, !dbg !1068
  %21 = load double, double* %x1, align 8, !dbg !1069
  %22 = load double, double* %x1, align 8, !dbg !1070
  %mul37 = fmul contract double %21, %22, !dbg !1071
  %23 = load double, double* %x2, align 8, !dbg !1072
  %24 = load double, double* %x2, align 8, !dbg !1073
  %mul38 = fmul contract double %23, %24, !dbg !1074
  %add39 = fadd contract double %mul37, %mul38, !dbg !1075
  store double %add39, double* %t1, align 8, !dbg !1076
  %25 = load double, double* %t1, align 8, !dbg !1077
  %cmp40 = fcmp ole double %25, 1.000000e+00, !dbg !1078
  br i1 %cmp40, label %if.then41, label %if.end58, !dbg !1079

if.then41:                                        ; preds = %for.body27
  %26 = load double, double* %t1, align 8, !dbg !1080
  store double %26, double* %a.addr.i, align 8
  %27 = load double, double* %a.addr.i, align 8, !dbg !1081
  %28 = call i32 @llvm.nvvm.d2i.hi(double %27) #5, !dbg !1082
  %29 = call i32 @llvm.nvvm.d2i.lo(double %27) #5, !dbg !1082
  %30 = fcmp ogt double %27, 0.000000e+00, !dbg !1082
  br i1 %30, label %31, label %33, !dbg !1082

31:                                               ; preds = %if.then41
  %32 = icmp slt i32 %28, 2146435072, !dbg !1082
  br label %33, !dbg !1082

33:                                               ; preds = %31, %if.then41
  %34 = phi i1 [ false, %if.then41 ], [ %32, %31 ], !dbg !1082
  br i1 %34, label %35, label %90, !dbg !1082

35:                                               ; preds = %33
  %36 = icmp slt i32 %28, 1048576, !dbg !1082
  br i1 %36, label %37, label %41, !dbg !1082

37:                                               ; preds = %35
  %38 = fmul double %27, 0x4350000000000000, !dbg !1082
  %39 = call i32 @llvm.nvvm.d2i.hi(double %38) #5, !dbg !1082
  %40 = call i32 @llvm.nvvm.d2i.lo(double %38) #5, !dbg !1082
  br label %41, !dbg !1082

41:                                               ; preds = %37, %35
  %ihi.0.i.i = phi i32 [ %39, %37 ], [ %28, %35 ], !dbg !1082
  %ilo.0.i.i = phi i32 [ %40, %37 ], [ %29, %35 ], !dbg !1082
  %e.0.i.i = phi i32 [ -1077, %37 ], [ -1023, %35 ], !dbg !1082
  %42 = lshr i32 %ihi.0.i.i, 20, !dbg !1082
  %43 = add i32 %e.0.i.i, %42, !dbg !1082
  %44 = and i32 %ihi.0.i.i, -2146435073, !dbg !1082
  %45 = or i32 %44, 1072693248, !dbg !1082
  %46 = call double @llvm.nvvm.lohi.i2d(i32 %ilo.0.i.i, i32 %45) #5, !dbg !1082
  %47 = icmp sgt i32 %45, 1073127582, !dbg !1082
  br i1 %47, label %48, label %54, !dbg !1082

48:                                               ; preds = %41
  %49 = call i32 @llvm.nvvm.d2i.lo(double %46) #5, !dbg !1082
  %50 = call i32 @llvm.nvvm.d2i.hi(double %46) #5, !dbg !1082
  %51 = add i32 -1048576, %50, !dbg !1082
  %52 = call double @llvm.nvvm.lohi.i2d(i32 %49, i32 %51) #5, !dbg !1082
  %53 = add nsw i32 %43, 1, !dbg !1082
  br label %54, !dbg !1082

54:                                               ; preds = %48, %41
  %m.0.i.i = phi double [ %52, %48 ], [ %46, %41 ], !dbg !1082
  %e.1.i.i = phi i32 [ %53, %48 ], [ %43, %41 ], !dbg !1082
  %55 = fsub double %m.0.i.i, 1.000000e+00, !dbg !1082
  %56 = fadd double %m.0.i.i, 1.000000e+00, !dbg !1082
  %57 = call double asm "rcp.approx.ftz.f64 $0,$1;", "=d,d"(double %56) #5, !dbg !1082
  %58 = fsub double -0.000000e+00, %56, !dbg !1082
  %59 = call double @llvm.nvvm.fma.rn.d(double %58, double %57, double 1.000000e+00) #5, !dbg !1082
  %60 = call double @llvm.nvvm.fma.rn.d(double %59, double %59, double %59) #5, !dbg !1082
  %61 = call double @llvm.nvvm.fma.rn.d(double %60, double %57, double %57) #5, !dbg !1082
  %62 = fmul double %55, %61, !dbg !1082
  %63 = fadd double %62, %62, !dbg !1082
  %64 = fmul double %63, %63, !dbg !1082
  %65 = call double @llvm.nvvm.fma.rn.d(double 0x3EB1380B3AE80F1E, double %64, double 0x3ED0EE258B7A8B04) #5, !dbg !1082
  %66 = call double @llvm.nvvm.fma.rn.d(double %65, double %64, double 0x3EF3B2669F02676F) #5, !dbg !1082
  %67 = call double @llvm.nvvm.fma.rn.d(double %66, double %64, double 0x3F1745CBA9AB0956) #5, !dbg !1082
  %68 = call double @llvm.nvvm.fma.rn.d(double %67, double %64, double 0x3F3C71C72D1B5154) #5, !dbg !1082
  %69 = call double @llvm.nvvm.fma.rn.d(double %68, double %64, double 0x3F624924923BE72D) #5, !dbg !1082
  %70 = call double @llvm.nvvm.fma.rn.d(double %69, double %64, double 0x3F8999999999A3C4) #5, !dbg !1082
  %71 = call double @llvm.nvvm.fma.rn.d(double %70, double %64, double 0x3FB5555555555554) #5, !dbg !1082
  %72 = fsub double %55, %63, !dbg !1082
  %73 = fmul double 2.000000e+00, %72, !dbg !1082
  %74 = fsub double -0.000000e+00, %63, !dbg !1082
  %75 = call double @llvm.nvvm.fma.rn.d(double %74, double %55, double %73) #5, !dbg !1082
  %76 = fmul double %61, %75, !dbg !1082
  %77 = fmul double %71, %64, !dbg !1082
  %78 = call double @llvm.nvvm.fma.rn.d(double %77, double %63, double %76) #5, !dbg !1082
  %79 = xor i32 -2147483648, %e.1.i.i, !dbg !1082
  %80 = call double @llvm.nvvm.lohi.i2d(i32 %79, i32 1127219200) #5, !dbg !1082
  %81 = call double @llvm.nvvm.lohi.i2d(i32 -2147483648, i32 1127219200) #5, !dbg !1082
  %82 = fsub double %80, %81, !dbg !1082
  %83 = call double @llvm.nvvm.fma.rn.d(double %82, double 0x3FE62E42FEFA39EF, double %63) #5, !dbg !1082
  %84 = fsub double -0.000000e+00, %82, !dbg !1082
  %85 = call double @llvm.nvvm.fma.rn.d(double %84, double 0x3FE62E42FEFA39EF, double %83) #5, !dbg !1082
  %86 = fsub double %85, %63, !dbg !1082
  %87 = fsub double %78, %86, !dbg !1082
  %88 = call double @llvm.nvvm.fma.rn.d(double %82, double 0x3C7ABC9E3B39803F, double %87) #5, !dbg !1082
  %89 = fadd double %83, %88, !dbg !1082
  br label %_ZL3logd.exit, !dbg !1082

90:                                               ; preds = %33
  %91 = call double @llvm.nvvm.fabs.d(double %27) #5, !dbg !1082
  %92 = fcmp ole double %91, 0x7FF0000000000000, !dbg !1082
  %93 = xor i1 %92, true, !dbg !1082
  %94 = zext i1 %93 to i32, !dbg !1082
  br i1 %93, label %95, label %97, !dbg !1082

95:                                               ; preds = %90
  %96 = fadd double %27, %27, !dbg !1082
  br label %106, !dbg !1082

97:                                               ; preds = %90
  %98 = fcmp oeq double %27, 0.000000e+00, !dbg !1082
  br i1 %98, label %99, label %100, !dbg !1082

99:                                               ; preds = %97
  br label %105, !dbg !1082

100:                                              ; preds = %97
  %101 = fcmp oeq double %27, 0x7FF0000000000000, !dbg !1082
  br i1 %101, label %102, label %103, !dbg !1082

102:                                              ; preds = %100
  br label %104, !dbg !1082

103:                                              ; preds = %100
  br label %104, !dbg !1082

104:                                              ; preds = %103, %102
  %q.0.i.i = phi double [ %27, %102 ], [ 0xFFF8000000000000, %103 ], !dbg !1082
  br label %105, !dbg !1082

105:                                              ; preds = %104, %99
  %q.1.i.i = phi double [ 0xFFF0000000000000, %99 ], [ %q.0.i.i, %104 ], !dbg !1082
  br label %106, !dbg !1082

106:                                              ; preds = %105, %95
  %q.2.i.i = phi double [ %96, %95 ], [ %q.1.i.i, %105 ], !dbg !1082
  br label %_ZL3logd.exit, !dbg !1082

_ZL3logd.exit:                                    ; preds = %54, %106
  %q.3.i.i = phi double [ %89, %54 ], [ %q.2.i.i, %106 ], !dbg !1082
  %mul43 = fmul contract double -2.000000e+00, %q.3.i.i, !dbg !1083
  %107 = load double, double* %t1, align 8, !dbg !1084
  %div44 = fdiv double %mul43, %107, !dbg !1085
  store double %div44, double* %x.addr.i, align 8
  %108 = load double, double* %x.addr.i, align 8, !dbg !1086
  %109 = call double @llvm.nvvm.sqrt.rn.d(double %108) #5, !dbg !1087
  store double %109, double* %t2, align 8, !dbg !1088
  %110 = load double, double* %x1, align 8, !dbg !1089
  %111 = load double, double* %t2, align 8, !dbg !1090
  %mul46 = fmul contract double %110, %111, !dbg !1091
  store double %mul46, double* %t3, align 8, !dbg !1092
  %112 = load double, double* %x2, align 8, !dbg !1093
  %113 = load double, double* %t2, align 8, !dbg !1094
  %mul47 = fmul contract double %112, %113, !dbg !1095
  store double %mul47, double* %t4, align 8, !dbg !1096
  %114 = load double, double* %t3, align 8, !dbg !1097
  store double %114, double* %f.addr.i, align 8
  %115 = load double, double* %f.addr.i, align 8, !dbg !1098
  %116 = call double @llvm.nvvm.fabs.d(double %115) #5, !dbg !1099
  %117 = load double, double* %t4, align 8, !dbg !1097
  store double %117, double* %f.addr.i141, align 8
  %118 = load double, double* %f.addr.i141, align 8, !dbg !1100
  %119 = call double @llvm.nvvm.fabs.d(double %118) #5, !dbg !1101
  %cmp50 = fcmp ogt double %116, %119, !dbg !1097
  br i1 %cmp50, label %cond.true, label %cond.false, !dbg !1097

cond.true:                                        ; preds = %_ZL3logd.exit
  %120 = load double, double* %t3, align 8, !dbg !1097
  store double %120, double* %f.addr.i142, align 8
  %121 = load double, double* %f.addr.i142, align 8, !dbg !1102
  %122 = call double @llvm.nvvm.fabs.d(double %121) #5, !dbg !1103
  br label %cond.end, !dbg !1097

cond.false:                                       ; preds = %_ZL3logd.exit
  %123 = load double, double* %t4, align 8, !dbg !1097
  store double %123, double* %f.addr.i143, align 8
  %124 = load double, double* %f.addr.i143, align 8, !dbg !1104
  %125 = call double @llvm.nvvm.fabs.d(double %124) #5, !dbg !1105
  br label %cond.end, !dbg !1097

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi double [ %122, %cond.true ], [ %125, %cond.false ], !dbg !1097
  %conv = fptosi double %cond to i32, !dbg !1097
  store i32 %conv, i32* %l, align 4, !dbg !1106
  %126 = load i32, i32* %l, align 4, !dbg !1107
  %idxprom53 = sext i32 %126 to i64, !dbg !1108
  %arrayidx54 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 %idxprom53, !dbg !1108
  %127 = load double, double* %arrayidx54, align 8, !dbg !1109
  %add55 = fadd contract double %127, 1.000000e+00, !dbg !1109
  store double %add55, double* %arrayidx54, align 8, !dbg !1109
  %128 = load double, double* %sx_local, align 8, !dbg !1110
  %129 = load double, double* %t3, align 8, !dbg !1111
  %add56 = fadd contract double %128, %129, !dbg !1112
  store double %add56, double* %sx_local, align 8, !dbg !1113
  %130 = load double, double* %t4, align 8, !dbg !1114
  %131 = load double, double* %sy_local, align 8, !dbg !1115
  %add57 = fadd contract double %131, %130, !dbg !1115
  store double %add57, double* %sy_local, align 8, !dbg !1115
  br label %if.end58, !dbg !1116

if.end58:                                         ; preds = %cond.end, %for.body27
  br label %for.inc59, !dbg !1117

for.inc59:                                        ; preds = %if.end58
  %132 = load i32, i32* %i, align 4, !dbg !1118
  %inc60 = add nsw i32 %132, 1, !dbg !1118
  store i32 %inc60, i32* %i, align 4, !dbg !1118
  br label %for.cond25, !dbg !1119, !llvm.loop !1120

for.end61:                                        ; preds = %for.cond25
  br label %for.inc62, !dbg !1122

for.inc62:                                        ; preds = %for.end61
  %133 = load i32, i32* %ii, align 4, !dbg !1123
  %add63 = add nsw i32 %133, 128, !dbg !1124
  store i32 %add63, i32* %ii, align 4, !dbg !1125
  br label %for.cond22, !dbg !1126, !llvm.loop !1127

for.end64:                                        ; preds = %for.cond22
  %134 = load double*, double** %q_global.addr, align 8, !dbg !1129
  %135 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #5, !dbg !1130, !range !917
  %mul66 = mul i32 %135, 10, !dbg !1132
  %idx.ext = zext i32 %mul66 to i64, !dbg !1133
  %add.ptr = getelementptr inbounds double, double* %134, i64 %idx.ext, !dbg !1133
  %add.ptr67 = getelementptr inbounds double, double* %add.ptr, i64 0, !dbg !1134
  %arrayidx68 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 0, !dbg !1135
  %136 = load double, double* %arrayidx68, align 8, !dbg !1135
  %call69 = call double @_ZL9atomicAddPdd(double* %add.ptr67, double %136) #6, !dbg !1136
  %137 = load double*, double** %q_global.addr, align 8, !dbg !1137
  %138 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #5, !dbg !1138, !range !917
  %mul71 = mul i32 %138, 10, !dbg !1140
  %idx.ext72 = zext i32 %mul71 to i64, !dbg !1141
  %add.ptr73 = getelementptr inbounds double, double* %137, i64 %idx.ext72, !dbg !1141
  %add.ptr74 = getelementptr inbounds double, double* %add.ptr73, i64 1, !dbg !1142
  %arrayidx75 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 1, !dbg !1143
  %139 = load double, double* %arrayidx75, align 8, !dbg !1143
  %call76 = call double @_ZL9atomicAddPdd(double* %add.ptr74, double %139) #6, !dbg !1144
  %140 = load double*, double** %q_global.addr, align 8, !dbg !1145
  %141 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #5, !dbg !1146, !range !917
  %mul78 = mul i32 %141, 10, !dbg !1148
  %idx.ext79 = zext i32 %mul78 to i64, !dbg !1149
  %add.ptr80 = getelementptr inbounds double, double* %140, i64 %idx.ext79, !dbg !1149
  %add.ptr81 = getelementptr inbounds double, double* %add.ptr80, i64 2, !dbg !1150
  %arrayidx82 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 2, !dbg !1151
  %142 = load double, double* %arrayidx82, align 8, !dbg !1151
  %call83 = call double @_ZL9atomicAddPdd(double* %add.ptr81, double %142) #6, !dbg !1152
  %143 = load double*, double** %q_global.addr, align 8, !dbg !1153
  %144 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #5, !dbg !1154, !range !917
  %mul85 = mul i32 %144, 10, !dbg !1156
  %idx.ext86 = zext i32 %mul85 to i64, !dbg !1157
  %add.ptr87 = getelementptr inbounds double, double* %143, i64 %idx.ext86, !dbg !1157
  %add.ptr88 = getelementptr inbounds double, double* %add.ptr87, i64 3, !dbg !1158
  %arrayidx89 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 3, !dbg !1159
  %145 = load double, double* %arrayidx89, align 8, !dbg !1159
  %call90 = call double @_ZL9atomicAddPdd(double* %add.ptr88, double %145) #6, !dbg !1160
  %146 = load double*, double** %q_global.addr, align 8, !dbg !1161
  %147 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #5, !dbg !1162, !range !917
  %mul92 = mul i32 %147, 10, !dbg !1164
  %idx.ext93 = zext i32 %mul92 to i64, !dbg !1165
  %add.ptr94 = getelementptr inbounds double, double* %146, i64 %idx.ext93, !dbg !1165
  %add.ptr95 = getelementptr inbounds double, double* %add.ptr94, i64 4, !dbg !1166
  %arrayidx96 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 4, !dbg !1167
  %148 = load double, double* %arrayidx96, align 8, !dbg !1167
  %call97 = call double @_ZL9atomicAddPdd(double* %add.ptr95, double %148) #6, !dbg !1168
  %149 = load double*, double** %q_global.addr, align 8, !dbg !1169
  %150 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #5, !dbg !1170, !range !917
  %mul99 = mul i32 %150, 10, !dbg !1172
  %idx.ext100 = zext i32 %mul99 to i64, !dbg !1173
  %add.ptr101 = getelementptr inbounds double, double* %149, i64 %idx.ext100, !dbg !1173
  %add.ptr102 = getelementptr inbounds double, double* %add.ptr101, i64 5, !dbg !1174
  %arrayidx103 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 5, !dbg !1175
  %151 = load double, double* %arrayidx103, align 8, !dbg !1175
  %call104 = call double @_ZL9atomicAddPdd(double* %add.ptr102, double %151) #6, !dbg !1176
  %152 = load double*, double** %q_global.addr, align 8, !dbg !1177
  %153 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #5, !dbg !1178, !range !917
  %mul106 = mul i32 %153, 10, !dbg !1180
  %idx.ext107 = zext i32 %mul106 to i64, !dbg !1181
  %add.ptr108 = getelementptr inbounds double, double* %152, i64 %idx.ext107, !dbg !1181
  %add.ptr109 = getelementptr inbounds double, double* %add.ptr108, i64 6, !dbg !1182
  %arrayidx110 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 6, !dbg !1183
  %154 = load double, double* %arrayidx110, align 8, !dbg !1183
  %call111 = call double @_ZL9atomicAddPdd(double* %add.ptr109, double %154) #6, !dbg !1184
  %155 = load double*, double** %q_global.addr, align 8, !dbg !1185
  %156 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #5, !dbg !1186, !range !917
  %mul113 = mul i32 %156, 10, !dbg !1188
  %idx.ext114 = zext i32 %mul113 to i64, !dbg !1189
  %add.ptr115 = getelementptr inbounds double, double* %155, i64 %idx.ext114, !dbg !1189
  %add.ptr116 = getelementptr inbounds double, double* %add.ptr115, i64 7, !dbg !1190
  %arrayidx117 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 7, !dbg !1191
  %157 = load double, double* %arrayidx117, align 8, !dbg !1191
  %call118 = call double @_ZL9atomicAddPdd(double* %add.ptr116, double %157) #6, !dbg !1192
  %158 = load double*, double** %q_global.addr, align 8, !dbg !1193
  %159 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #5, !dbg !1194, !range !917
  %mul120 = mul i32 %159, 10, !dbg !1196
  %idx.ext121 = zext i32 %mul120 to i64, !dbg !1197
  %add.ptr122 = getelementptr inbounds double, double* %158, i64 %idx.ext121, !dbg !1197
  %add.ptr123 = getelementptr inbounds double, double* %add.ptr122, i64 8, !dbg !1198
  %arrayidx124 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 8, !dbg !1199
  %160 = load double, double* %arrayidx124, align 8, !dbg !1199
  %call125 = call double @_ZL9atomicAddPdd(double* %add.ptr123, double %160) #6, !dbg !1200
  %161 = load double*, double** %q_global.addr, align 8, !dbg !1201
  %162 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #5, !dbg !1202, !range !917
  %mul127 = mul i32 %162, 10, !dbg !1204
  %idx.ext128 = zext i32 %mul127 to i64, !dbg !1205
  %add.ptr129 = getelementptr inbounds double, double* %161, i64 %idx.ext128, !dbg !1205
  %add.ptr130 = getelementptr inbounds double, double* %add.ptr129, i64 9, !dbg !1206
  %arrayidx131 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 9, !dbg !1207
  %163 = load double, double* %arrayidx131, align 8, !dbg !1207
  %call132 = call double @_ZL9atomicAddPdd(double* %add.ptr130, double %163) #6, !dbg !1208
  %164 = load double*, double** %sx_global.addr, align 8, !dbg !1209
  %165 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #5, !dbg !1210, !range !917
  %idx.ext134 = zext i32 %165 to i64, !dbg !1212
  %add.ptr135 = getelementptr inbounds double, double* %164, i64 %idx.ext134, !dbg !1212
  %166 = load double, double* %sx_local, align 8, !dbg !1213
  %call136 = call double @_ZL9atomicAddPdd(double* %add.ptr135, double %166) #6, !dbg !1214
  %167 = load double*, double** %sy_global.addr, align 8, !dbg !1215
  %168 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #5, !dbg !1216, !range !917
  %idx.ext138 = zext i32 %168 to i64, !dbg !1218
  %add.ptr139 = getelementptr inbounds double, double* %167, i64 %idx.ext138, !dbg !1218
  %169 = load double, double* %sy_local, align 8, !dbg !1219
  %call140 = call double @_ZL9atomicAddPdd(double* %add.ptr139, double %169) #6, !dbg !1220
  br label %return, !dbg !1221

return:                                           ; preds = %for.end64, %if.then
  ret void, !dbg !1221
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: convergent noinline nounwind
define dso_local double @_Z13randlc_devicePdd(double* %x, double %a) #2 !dbg !1222 {
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
  call void @llvm.dbg.declare(metadata double** %x.addr, metadata !1225, metadata !DIExpression()), !dbg !1226
  store double %a, double* %a.addr, align 8
  call void @llvm.dbg.declare(metadata double* %a.addr, metadata !1227, metadata !DIExpression()), !dbg !1228
  call void @llvm.dbg.declare(metadata double* %t1, metadata !1229, metadata !DIExpression()), !dbg !1230
  call void @llvm.dbg.declare(metadata double* %t2, metadata !1231, metadata !DIExpression()), !dbg !1232
  call void @llvm.dbg.declare(metadata double* %t3, metadata !1233, metadata !DIExpression()), !dbg !1234
  call void @llvm.dbg.declare(metadata double* %t4, metadata !1235, metadata !DIExpression()), !dbg !1236
  call void @llvm.dbg.declare(metadata double* %a1, metadata !1237, metadata !DIExpression()), !dbg !1238
  call void @llvm.dbg.declare(metadata double* %a2, metadata !1239, metadata !DIExpression()), !dbg !1240
  call void @llvm.dbg.declare(metadata double* %x1, metadata !1241, metadata !DIExpression()), !dbg !1242
  call void @llvm.dbg.declare(metadata double* %x2, metadata !1243, metadata !DIExpression()), !dbg !1244
  call void @llvm.dbg.declare(metadata double* %z, metadata !1245, metadata !DIExpression()), !dbg !1246
  %0 = load double, double* %a.addr, align 8, !dbg !1247
  %mul = fmul contract double 0x3E80000000000000, %0, !dbg !1248
  store double %mul, double* %t1, align 8, !dbg !1249
  %1 = load double, double* %t1, align 8, !dbg !1250
  %conv = fptosi double %1 to i32, !dbg !1250
  %conv1 = sitofp i32 %conv to double, !dbg !1251
  store double %conv1, double* %a1, align 8, !dbg !1252
  %2 = load double, double* %a.addr, align 8, !dbg !1253
  %3 = load double, double* %a1, align 8, !dbg !1254
  %mul2 = fmul contract double 0x4160000000000000, %3, !dbg !1255
  %sub = fsub contract double %2, %mul2, !dbg !1256
  store double %sub, double* %a2, align 8, !dbg !1257
  %4 = load double*, double** %x.addr, align 8, !dbg !1258
  %5 = load double, double* %4, align 8, !dbg !1259
  %mul3 = fmul contract double 0x3E80000000000000, %5, !dbg !1260
  store double %mul3, double* %t1, align 8, !dbg !1261
  %6 = load double, double* %t1, align 8, !dbg !1262
  %conv4 = fptosi double %6 to i32, !dbg !1262
  %conv5 = sitofp i32 %conv4 to double, !dbg !1263
  store double %conv5, double* %x1, align 8, !dbg !1264
  %7 = load double*, double** %x.addr, align 8, !dbg !1265
  %8 = load double, double* %7, align 8, !dbg !1266
  %9 = load double, double* %x1, align 8, !dbg !1267
  %mul6 = fmul contract double 0x4160000000000000, %9, !dbg !1268
  %sub7 = fsub contract double %8, %mul6, !dbg !1269
  store double %sub7, double* %x2, align 8, !dbg !1270
  %10 = load double, double* %a1, align 8, !dbg !1271
  %11 = load double, double* %x2, align 8, !dbg !1272
  %mul8 = fmul contract double %10, %11, !dbg !1273
  %12 = load double, double* %a2, align 8, !dbg !1274
  %13 = load double, double* %x1, align 8, !dbg !1275
  %mul9 = fmul contract double %12, %13, !dbg !1276
  %add = fadd contract double %mul8, %mul9, !dbg !1277
  store double %add, double* %t1, align 8, !dbg !1278
  %14 = load double, double* %t1, align 8, !dbg !1279
  %mul10 = fmul contract double 0x3E80000000000000, %14, !dbg !1280
  %conv11 = fptosi double %mul10 to i32, !dbg !1281
  %conv12 = sitofp i32 %conv11 to double, !dbg !1282
  store double %conv12, double* %t2, align 8, !dbg !1283
  %15 = load double, double* %t1, align 8, !dbg !1284
  %16 = load double, double* %t2, align 8, !dbg !1285
  %mul13 = fmul contract double 0x4160000000000000, %16, !dbg !1286
  %sub14 = fsub contract double %15, %mul13, !dbg !1287
  store double %sub14, double* %z, align 8, !dbg !1288
  %17 = load double, double* %z, align 8, !dbg !1289
  %mul15 = fmul contract double 0x4160000000000000, %17, !dbg !1290
  %18 = load double, double* %a2, align 8, !dbg !1291
  %19 = load double, double* %x2, align 8, !dbg !1292
  %mul16 = fmul contract double %18, %19, !dbg !1293
  %add17 = fadd contract double %mul15, %mul16, !dbg !1294
  store double %add17, double* %t3, align 8, !dbg !1295
  %20 = load double, double* %t3, align 8, !dbg !1296
  %mul18 = fmul contract double 0x3D10000000000000, %20, !dbg !1297
  %conv19 = fptosi double %mul18 to i32, !dbg !1298
  %conv20 = sitofp i32 %conv19 to double, !dbg !1299
  store double %conv20, double* %t4, align 8, !dbg !1300
  %21 = load double, double* %t3, align 8, !dbg !1301
  %22 = load double, double* %t4, align 8, !dbg !1302
  %mul21 = fmul contract double 0x42D0000000000000, %22, !dbg !1303
  %sub22 = fsub contract double %21, %mul21, !dbg !1304
  %23 = load double*, double** %x.addr, align 8, !dbg !1305
  store double %sub22, double* %23, align 8, !dbg !1306
  %24 = load double*, double** %x.addr, align 8, !dbg !1307
  %25 = load double, double* %24, align 8, !dbg !1308
  %mul23 = fmul contract double 0x3D10000000000000, %25, !dbg !1309
  ret double %mul23, !dbg !1310
}

; Function Attrs: convergent noinline nounwind
define dso_local void @_Z13vranlc_deviceiPddS_(i32 %n, double* %x_seed, double %a, double* %y) #2 !dbg !1311 {
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
  call void @llvm.dbg.declare(metadata i32* %n.addr, metadata !1314, metadata !DIExpression()), !dbg !1315
  store double* %x_seed, double** %x_seed.addr, align 8
  call void @llvm.dbg.declare(metadata double** %x_seed.addr, metadata !1316, metadata !DIExpression()), !dbg !1317
  store double %a, double* %a.addr, align 8
  call void @llvm.dbg.declare(metadata double* %a.addr, metadata !1318, metadata !DIExpression()), !dbg !1319
  store double* %y, double** %y.addr, align 8
  call void @llvm.dbg.declare(metadata double** %y.addr, metadata !1320, metadata !DIExpression()), !dbg !1321
  call void @llvm.dbg.declare(metadata i32* %i, metadata !1322, metadata !DIExpression()), !dbg !1323
  call void @llvm.dbg.declare(metadata double* %x, metadata !1324, metadata !DIExpression()), !dbg !1325
  call void @llvm.dbg.declare(metadata double* %t1, metadata !1326, metadata !DIExpression()), !dbg !1327
  call void @llvm.dbg.declare(metadata double* %t2, metadata !1328, metadata !DIExpression()), !dbg !1329
  call void @llvm.dbg.declare(metadata double* %t3, metadata !1330, metadata !DIExpression()), !dbg !1331
  call void @llvm.dbg.declare(metadata double* %t4, metadata !1332, metadata !DIExpression()), !dbg !1333
  call void @llvm.dbg.declare(metadata double* %a1, metadata !1334, metadata !DIExpression()), !dbg !1335
  call void @llvm.dbg.declare(metadata double* %a2, metadata !1336, metadata !DIExpression()), !dbg !1337
  call void @llvm.dbg.declare(metadata double* %x1, metadata !1338, metadata !DIExpression()), !dbg !1339
  call void @llvm.dbg.declare(metadata double* %x2, metadata !1340, metadata !DIExpression()), !dbg !1341
  call void @llvm.dbg.declare(metadata double* %z, metadata !1342, metadata !DIExpression()), !dbg !1343
  %0 = load double, double* %a.addr, align 8, !dbg !1344
  %mul = fmul contract double 0x3E80000000000000, %0, !dbg !1345
  store double %mul, double* %t1, align 8, !dbg !1346
  %1 = load double, double* %t1, align 8, !dbg !1347
  %conv = fptosi double %1 to i32, !dbg !1347
  %conv1 = sitofp i32 %conv to double, !dbg !1348
  store double %conv1, double* %a1, align 8, !dbg !1349
  %2 = load double, double* %a.addr, align 8, !dbg !1350
  %3 = load double, double* %a1, align 8, !dbg !1351
  %mul2 = fmul contract double 0x4160000000000000, %3, !dbg !1352
  %sub = fsub contract double %2, %mul2, !dbg !1353
  store double %sub, double* %a2, align 8, !dbg !1354
  %4 = load double*, double** %x_seed.addr, align 8, !dbg !1355
  %5 = load double, double* %4, align 8, !dbg !1356
  store double %5, double* %x, align 8, !dbg !1357
  store i32 0, i32* %i, align 4, !dbg !1358
  br label %for.cond, !dbg !1360

for.cond:                                         ; preds = %for.inc, %entry
  %6 = load i32, i32* %i, align 4, !dbg !1361
  %7 = load i32, i32* %n.addr, align 4, !dbg !1363
  %cmp = icmp slt i32 %6, %7, !dbg !1364
  br i1 %cmp, label %for.body, label %for.end, !dbg !1365

for.body:                                         ; preds = %for.cond
  %8 = load double, double* %x, align 8, !dbg !1366
  %mul3 = fmul contract double 0x3E80000000000000, %8, !dbg !1368
  store double %mul3, double* %t1, align 8, !dbg !1369
  %9 = load double, double* %t1, align 8, !dbg !1370
  %conv4 = fptosi double %9 to i32, !dbg !1370
  %conv5 = sitofp i32 %conv4 to double, !dbg !1371
  store double %conv5, double* %x1, align 8, !dbg !1372
  %10 = load double, double* %x, align 8, !dbg !1373
  %11 = load double, double* %x1, align 8, !dbg !1374
  %mul6 = fmul contract double 0x4160000000000000, %11, !dbg !1375
  %sub7 = fsub contract double %10, %mul6, !dbg !1376
  store double %sub7, double* %x2, align 8, !dbg !1377
  %12 = load double, double* %a1, align 8, !dbg !1378
  %13 = load double, double* %x2, align 8, !dbg !1379
  %mul8 = fmul contract double %12, %13, !dbg !1380
  %14 = load double, double* %a2, align 8, !dbg !1381
  %15 = load double, double* %x1, align 8, !dbg !1382
  %mul9 = fmul contract double %14, %15, !dbg !1383
  %add = fadd contract double %mul8, %mul9, !dbg !1384
  store double %add, double* %t1, align 8, !dbg !1385
  %16 = load double, double* %t1, align 8, !dbg !1386
  %mul10 = fmul contract double 0x3E80000000000000, %16, !dbg !1387
  %conv11 = fptosi double %mul10 to i32, !dbg !1388
  %conv12 = sitofp i32 %conv11 to double, !dbg !1389
  store double %conv12, double* %t2, align 8, !dbg !1390
  %17 = load double, double* %t1, align 8, !dbg !1391
  %18 = load double, double* %t2, align 8, !dbg !1392
  %mul13 = fmul contract double 0x4160000000000000, %18, !dbg !1393
  %sub14 = fsub contract double %17, %mul13, !dbg !1394
  store double %sub14, double* %z, align 8, !dbg !1395
  %19 = load double, double* %z, align 8, !dbg !1396
  %mul15 = fmul contract double 0x4160000000000000, %19, !dbg !1397
  %20 = load double, double* %a2, align 8, !dbg !1398
  %21 = load double, double* %x2, align 8, !dbg !1399
  %mul16 = fmul contract double %20, %21, !dbg !1400
  %add17 = fadd contract double %mul15, %mul16, !dbg !1401
  store double %add17, double* %t3, align 8, !dbg !1402
  %22 = load double, double* %t3, align 8, !dbg !1403
  %mul18 = fmul contract double 0x3D10000000000000, %22, !dbg !1404
  %conv19 = fptosi double %mul18 to i32, !dbg !1405
  %conv20 = sitofp i32 %conv19 to double, !dbg !1406
  store double %conv20, double* %t4, align 8, !dbg !1407
  %23 = load double, double* %t3, align 8, !dbg !1408
  %24 = load double, double* %t4, align 8, !dbg !1409
  %mul21 = fmul contract double 0x42D0000000000000, %24, !dbg !1410
  %sub22 = fsub contract double %23, %mul21, !dbg !1411
  store double %sub22, double* %x, align 8, !dbg !1412
  %25 = load double, double* %x, align 8, !dbg !1413
  %mul23 = fmul contract double 0x3D10000000000000, %25, !dbg !1414
  %26 = load double*, double** %y.addr, align 8, !dbg !1415
  %27 = load i32, i32* %i, align 4, !dbg !1416
  %idxprom = sext i32 %27 to i64, !dbg !1415
  %arrayidx = getelementptr inbounds double, double* %26, i64 %idxprom, !dbg !1415
  store double %mul23, double* %arrayidx, align 8, !dbg !1417
  br label %for.inc, !dbg !1418

for.inc:                                          ; preds = %for.body
  %28 = load i32, i32* %i, align 4, !dbg !1419
  %inc = add nsw i32 %28, 1, !dbg !1419
  store i32 %inc, i32* %i, align 4, !dbg !1419
  br label %for.cond, !dbg !1420, !llvm.loop !1421

for.end:                                          ; preds = %for.cond
  %29 = load double, double* %x, align 8, !dbg !1423
  %30 = load double*, double** %x_seed.addr, align 8, !dbg !1424
  store double %29, double* %30, align 8, !dbg !1425
  ret void, !dbg !1426
}

; Function Attrs: convergent noinline nounwind
define internal double @_ZL9atomicAddPdd(double* %address, double %val) #0 !dbg !1427 {
entry:
  %x.addr.i13 = alloca double, align 8
  call void @llvm.dbg.declare(metadata double* %x.addr.i13, metadata !1429, metadata !DIExpression()), !dbg !1433
  %x.addr.i12 = alloca i64, align 8
  call void @llvm.dbg.declare(metadata i64* %x.addr.i12, metadata !1438, metadata !DIExpression()), !dbg !1442
  %x.addr.i11 = alloca i64, align 8
  call void @llvm.dbg.declare(metadata i64* %x.addr.i11, metadata !1438, metadata !DIExpression()), !dbg !1444
  %x.addr.i10 = alloca i64, align 8
  call void @llvm.dbg.declare(metadata i64* %x.addr.i10, metadata !1438, metadata !DIExpression()), !dbg !1447
  %x.addr.i = alloca i64, align 8
  call void @llvm.dbg.declare(metadata i64* %x.addr.i, metadata !1438, metadata !DIExpression()), !dbg !1449
  %retval = alloca double, align 8
  %address.addr = alloca double*, align 8
  %val.addr = alloca double, align 8
  %address_as_ull = alloca i64*, align 8
  %old = alloca i64, align 8
  %assumed = alloca i64, align 8
  %i = alloca i32, align 4
  store double* %address, double** %address.addr, align 8
  call void @llvm.dbg.declare(metadata double** %address.addr, metadata !1452, metadata !DIExpression()), !dbg !1453
  store double %val, double* %val.addr, align 8
  call void @llvm.dbg.declare(metadata double* %val.addr, metadata !1454, metadata !DIExpression()), !dbg !1455
  call void @llvm.dbg.declare(metadata i64** %address_as_ull, metadata !1456, metadata !DIExpression()), !dbg !1457
  %0 = load double*, double** %address.addr, align 8, !dbg !1458
  %1 = bitcast double* %0 to i64*, !dbg !1459
  store i64* %1, i64** %address_as_ull, align 8, !dbg !1457
  call void @llvm.dbg.declare(metadata i64* %old, metadata !1460, metadata !DIExpression()), !dbg !1461
  %2 = load i64*, i64** %address_as_ull, align 8, !dbg !1462
  %3 = load i64, i64* %2, align 8, !dbg !1463
  store i64 %3, i64* %old, align 8, !dbg !1461
  call void @llvm.dbg.declare(metadata i64* %assumed, metadata !1464, metadata !DIExpression()), !dbg !1465
  %4 = load double, double* %val.addr, align 8, !dbg !1466
  %cmp = fcmp oeq double %4, 0.000000e+00, !dbg !1467
  br i1 %cmp, label %if.then, label %if.end, !dbg !1468

if.then:                                          ; preds = %entry
  %5 = load i64, i64* %old, align 8, !dbg !1469
  store i64 %5, i64* %x.addr.i, align 8
  %6 = load i64, i64* %x.addr.i, align 8, !dbg !1470
  %7 = bitcast i64 %6 to double, !dbg !1471
  store double %7, double* %retval, align 8, !dbg !1472
  br label %return, !dbg !1472

if.end:                                           ; preds = %entry
  call void @llvm.dbg.declare(metadata i32* %i, metadata !1473, metadata !DIExpression()), !dbg !1474
  store i32 0, i32* %i, align 4, !dbg !1474
  br label %for.cond, !dbg !1475

for.cond:                                         ; preds = %for.inc, %if.end
  %8 = load i32, i32* %i, align 4, !dbg !1476
  %cmp1 = icmp slt i32 %8, 100000, !dbg !1477
  br i1 %cmp1, label %for.body, label %for.end, !dbg !1478

for.body:                                         ; preds = %for.cond
  %9 = load i64, i64* %old, align 8, !dbg !1479
  store i64 %9, i64* %assumed, align 8, !dbg !1480
  %10 = load i64*, i64** %address_as_ull, align 8, !dbg !1481
  %11 = load i64, i64* %assumed, align 8, !dbg !1482
  %12 = load double, double* %val.addr, align 8, !dbg !1483
  %13 = load i64, i64* %assumed, align 8, !dbg !1484
  store i64 %13, i64* %x.addr.i12, align 8
  %14 = load i64, i64* %x.addr.i12, align 8, !dbg !1485
  %15 = bitcast i64 %14 to double, !dbg !1486
  %add = fadd contract double %12, %15, !dbg !1487
  store double %add, double* %x.addr.i13, align 8
  %16 = load double, double* %x.addr.i13, align 8, !dbg !1488
  %17 = bitcast double %16 to i64, !dbg !1489
  %call4 = call i64 @_ZL9atomicCASPyyy(i64* %10, i64 %11, i64 %17) #6, !dbg !1490
  store i64 %call4, i64* %old, align 8, !dbg !1491
  %18 = load i64, i64* %assumed, align 8, !dbg !1492
  %19 = load i64, i64* %old, align 8, !dbg !1493
  %cmp5 = icmp eq i64 %18, %19, !dbg !1494
  br i1 %cmp5, label %if.then6, label %if.end8, !dbg !1495

if.then6:                                         ; preds = %for.body
  %20 = load i64, i64* %old, align 8, !dbg !1496
  store i64 %20, i64* %x.addr.i11, align 8
  %21 = load i64, i64* %x.addr.i11, align 8, !dbg !1497
  %22 = bitcast i64 %21 to double, !dbg !1498
  store double %22, double* %retval, align 8, !dbg !1499
  br label %return, !dbg !1499

if.end8:                                          ; preds = %for.body
  br label %for.inc, !dbg !1500

for.inc:                                          ; preds = %if.end8
  %23 = load i32, i32* %i, align 4, !dbg !1501
  %inc = add nsw i32 %23, 1, !dbg !1501
  store i32 %inc, i32* %i, align 4, !dbg !1501
  br label %for.cond, !dbg !1502, !llvm.loop !1503

for.end:                                          ; preds = %for.cond
  %24 = load i64, i64* %old, align 8, !dbg !1505
  store i64 %24, i64* %x.addr.i10, align 8
  %25 = load i64, i64* %x.addr.i10, align 8, !dbg !1506
  %26 = bitcast i64 %25 to double, !dbg !1507
  store double %26, double* %retval, align 8, !dbg !1508
  br label %return, !dbg !1508

return:                                           ; preds = %for.end, %if.then6, %if.then
  %27 = load double, double* %retval, align 8, !dbg !1509
  ret double %27, !dbg !1509
}

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #3

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #3

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3

; Function Attrs: convergent noinline nounwind
define internal i64 @_ZL9atomicCASPyyy(i64* %address, i64 %compare, i64 %val) #2 !dbg !1510 {
entry:
  %p.addr.i = alloca i64*, align 8
  call void @llvm.dbg.declare(metadata i64** %p.addr.i, metadata !1514, metadata !DIExpression()), !dbg !1516
  %compare.addr.i = alloca i64, align 8
  call void @llvm.dbg.declare(metadata i64* %compare.addr.i, metadata !1518, metadata !DIExpression()), !dbg !1519
  %val.addr.i = alloca i64, align 8
  call void @llvm.dbg.declare(metadata i64* %val.addr.i, metadata !1520, metadata !DIExpression()), !dbg !1521
  %address.addr = alloca i64*, align 8
  %compare.addr = alloca i64, align 8
  %val.addr = alloca i64, align 8
  store i64* %address, i64** %address.addr, align 8
  call void @llvm.dbg.declare(metadata i64** %address.addr, metadata !1522, metadata !DIExpression()), !dbg !1523
  store i64 %compare, i64* %compare.addr, align 8
  call void @llvm.dbg.declare(metadata i64* %compare.addr, metadata !1524, metadata !DIExpression()), !dbg !1525
  store i64 %val, i64* %val.addr, align 8
  call void @llvm.dbg.declare(metadata i64* %val.addr, metadata !1526, metadata !DIExpression()), !dbg !1527
  %0 = load i64*, i64** %address.addr, align 8, !dbg !1528
  %1 = load i64, i64* %compare.addr, align 8, !dbg !1529
  %2 = load i64, i64* %val.addr, align 8, !dbg !1530
  store i64* %0, i64** %p.addr.i, align 8
  store i64 %1, i64* %compare.addr.i, align 8
  store i64 %2, i64* %val.addr.i, align 8
  %3 = load i64*, i64** %p.addr.i, align 8, !dbg !1531
  %4 = load i64, i64* %compare.addr.i, align 8, !dbg !1532
  %5 = load i64, i64* %val.addr.i, align 8, !dbg !1533
  %6 = cmpxchg i64* %3, i64 %4, i64 %5 seq_cst seq_cst, !dbg !1534
  %7 = extractvalue { i64, i1 } %6, 0, !dbg !1534
  ret i64 %7, !dbg !1535
}

; Function Attrs: convergent nounwind readnone
declare double @llvm.nvvm.fabs.d(double) #4

; Function Attrs: convergent nounwind readnone
declare double @llvm.nvvm.sqrt.rn.d(double) #4

; Function Attrs: convergent nounwind readnone
declare i32 @llvm.nvvm.d2i.hi(double) #4

; Function Attrs: convergent nounwind readnone
declare i32 @llvm.nvvm.d2i.lo(double) #4

; Function Attrs: convergent nounwind readnone
declare double @llvm.nvvm.lohi.i2d(i32, i32) #4

; Function Attrs: convergent nounwind readnone
declare double @llvm.nvvm.fma.rn.d(double, double, double) #4

attributes #0 = { convergent noinline nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="sm_20" "target-features"="+ptx42,+sm_20" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { convergent noinline nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="sm_20" "target-features"="+ptx42,+sm_20" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind readnone }
attributes #4 = { convergent nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { nounwind }
attributes #6 = { convergent nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.dbg.cu = !{!5}
!nvvm.annotations = !{!775, !776, !777, !776, !778, !778, !778, !778, !779, !779, !778}
!llvm.ident = !{!780}
!nvvm.internalize.after.link = !{}
!nvvmir.version = !{!781}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 7, i32 0]}
!1 = !{i32 2, !"Dwarf Version", i32 2}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !6, producer: "clang version 9.0.0 (https://github.com/yebinchon/llvm-project/ a1efa594106d738d0b74c9e4e2b9b779eb8b7d25)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !7, retainedTypes: !8, imports: !15, nameTableKind: None)
!6 = !DIFile(filename: "ep.cu", directory: "/scratch/ah7226/NPB-GPU/CUDA/EP")
!7 = !{}
!8 = !{!9, !10, !11, !12, !14}
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11, size: 64)
!11 = !DIBasicType(name: "long long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 64)
!13 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !14)
!14 = !DIBasicType(name: "long long int", size: 64, encoding: DW_ATE_signed)
!15 = !{!16, !22, !27, !29, !31, !33, !35, !39, !41, !43, !45, !47, !49, !51, !53, !55, !57, !59, !61, !63, !65, !67, !71, !73, !75, !77, !81, !86, !88, !90, !95, !99, !101, !103, !105, !107, !109, !111, !113, !115, !120, !124, !126, !130, !134, !136, !138, !140, !142, !144, !148, !150, !152, !157, !165, !169, !171, !173, !175, !177, !181, !183, !185, !189, !191, !193, !195, !197, !199, !201, !203, !205, !207, !211, !217, !219, !221, !225, !227, !229, !231, !233, !235, !237, !239, !243, !247, !249, !251, !256, !258, !260, !262, !264, !266, !268, !272, !278, !282, !287, !289, !293, !297, !311, !315, !319, !323, !327, !332, !334, !338, !342, !346, !354, !358, !362, !366, !370, !375, !381, !385, !389, !391, !399, !403, !410, !412, !414, !418, !422, !426, !430, !434, !439, !440, !441, !442, !444, !445, !446, !447, !448, !449, !450, !452, !453, !454, !455, !456, !460, !461, !462, !463, !464, !465, !466, !467, !468, !469, !470, !471, !472, !473, !474, !475, !476, !477, !478, !479, !480, !481, !482, !483, !484, !488, !490, !492, !494, !496, !498, !500, !502, !505, !507, !509, !511, !513, !515, !517, !519, !521, !523, !525, !527, !529, !531, !533, !535, !537, !539, !541, !543, !545, !547, !549, !551, !553, !555, !557, !559, !561, !563, !565, !567, !569, !571, !573, !575, !577, !579, !581, !583, !585, !587, !589, !591, !593, !595, !597, !603, !609, !614, !618, !620, !622, !624, !626, !633, !637, !641, !645, !649, !653, !658, !662, !664, !668, !674, !678, !683, !685, !687, !691, !695, !699, !701, !703, !705, !707, !711, !713, !715, !719, !723, !727, !731, !735, !737, !739, !746, !750, !754, !758, !760, !762, !766, !770, !771, !772, !773, !774}
!16 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !18, file: !19, line: 223)
!17 = !DINamespace(name: "std", scope: null)
!18 = !DISubprogram(name: "abs", linkageName: "_ZL3absi", scope: !19, file: !19, line: 53, type: !20, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!19 = !DIFile(filename: "llvm-install-tulip/lib/clang/9.0.0/include/__clang_cuda_math_forward_declares.h", directory: "/scratch/ah7226")
!20 = !DISubroutineType(types: !21)
!21 = !{!9, !9}
!22 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !23, file: !19, line: 224)
!23 = !DISubprogram(name: "acos", linkageName: "_ZL4acosf", scope: !19, file: !19, line: 55, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!24 = !DISubroutineType(types: !25)
!25 = !{!26, !26}
!26 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!27 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !28, file: !19, line: 225)
!28 = !DISubprogram(name: "acosh", linkageName: "_ZL5acoshf", scope: !19, file: !19, line: 57, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!29 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !30, file: !19, line: 226)
!30 = !DISubprogram(name: "asin", linkageName: "_ZL4asinf", scope: !19, file: !19, line: 59, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!31 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !32, file: !19, line: 227)
!32 = !DISubprogram(name: "asinh", linkageName: "_ZL5asinhf", scope: !19, file: !19, line: 61, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!33 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !34, file: !19, line: 228)
!34 = !DISubprogram(name: "atan", linkageName: "_ZL4atanf", scope: !19, file: !19, line: 65, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!35 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !36, file: !19, line: 229)
!36 = !DISubprogram(name: "atan2", linkageName: "_ZL5atan2ff", scope: !19, file: !19, line: 63, type: !37, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!37 = !DISubroutineType(types: !38)
!38 = !{!26, !26, !26}
!39 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !40, file: !19, line: 230)
!40 = !DISubprogram(name: "atanh", linkageName: "_ZL5atanhf", scope: !19, file: !19, line: 67, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!41 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !42, file: !19, line: 231)
!42 = !DISubprogram(name: "cbrt", linkageName: "_ZL4cbrtf", scope: !19, file: !19, line: 69, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!43 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !44, file: !19, line: 232)
!44 = !DISubprogram(name: "ceil", linkageName: "_ZL4ceilf", scope: !19, file: !19, line: 71, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!45 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !46, file: !19, line: 233)
!46 = !DISubprogram(name: "copysign", linkageName: "_ZL8copysignff", scope: !19, file: !19, line: 73, type: !37, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!47 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !48, file: !19, line: 234)
!48 = !DISubprogram(name: "cos", linkageName: "_ZL3cosf", scope: !19, file: !19, line: 75, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!49 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !50, file: !19, line: 235)
!50 = !DISubprogram(name: "cosh", linkageName: "_ZL4coshf", scope: !19, file: !19, line: 77, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!51 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !52, file: !19, line: 236)
!52 = !DISubprogram(name: "erf", linkageName: "_ZL3erff", scope: !19, file: !19, line: 81, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!53 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !54, file: !19, line: 237)
!54 = !DISubprogram(name: "erfc", linkageName: "_ZL4erfcf", scope: !19, file: !19, line: 79, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!55 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !56, file: !19, line: 238)
!56 = !DISubprogram(name: "exp", linkageName: "_ZL3expf", scope: !19, file: !19, line: 85, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!57 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !58, file: !19, line: 239)
!58 = !DISubprogram(name: "exp2", linkageName: "_ZL4exp2f", scope: !19, file: !19, line: 83, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!59 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !60, file: !19, line: 240)
!60 = !DISubprogram(name: "expm1", linkageName: "_ZL5expm1f", scope: !19, file: !19, line: 87, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!61 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !62, file: !19, line: 241)
!62 = !DISubprogram(name: "fabs", linkageName: "_ZL4fabsf", scope: !19, file: !19, line: 89, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!63 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !64, file: !19, line: 242)
!64 = !DISubprogram(name: "fdim", linkageName: "_ZL4fdimff", scope: !19, file: !19, line: 91, type: !37, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!65 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !66, file: !19, line: 243)
!66 = !DISubprogram(name: "floor", linkageName: "_ZL5floorf", scope: !19, file: !19, line: 93, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!67 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !68, file: !19, line: 244)
!68 = !DISubprogram(name: "fma", linkageName: "_ZL3fmafff", scope: !19, file: !19, line: 95, type: !69, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!69 = !DISubroutineType(types: !70)
!70 = !{!26, !26, !26, !26}
!71 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !72, file: !19, line: 245)
!72 = !DISubprogram(name: "fmax", linkageName: "_ZL4fmaxff", scope: !19, file: !19, line: 97, type: !37, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!73 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !74, file: !19, line: 246)
!74 = !DISubprogram(name: "fmin", linkageName: "_ZL4fminff", scope: !19, file: !19, line: 99, type: !37, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!75 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !76, file: !19, line: 247)
!76 = !DISubprogram(name: "fmod", linkageName: "_ZL4fmodff", scope: !19, file: !19, line: 101, type: !37, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!77 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !78, file: !19, line: 248)
!78 = !DISubprogram(name: "fpclassify", linkageName: "_ZL10fpclassifyf", scope: !19, file: !19, line: 103, type: !79, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!79 = !DISubroutineType(types: !80)
!80 = !{!9, !26}
!81 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !82, file: !19, line: 249)
!82 = !DISubprogram(name: "frexp", linkageName: "_ZL5frexpfPi", scope: !19, file: !19, line: 105, type: !83, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!83 = !DISubroutineType(types: !84)
!84 = !{!26, !26, !85}
!85 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 64)
!86 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !87, file: !19, line: 250)
!87 = !DISubprogram(name: "hypot", linkageName: "_ZL5hypotff", scope: !19, file: !19, line: 107, type: !37, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!88 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !89, file: !19, line: 251)
!89 = !DISubprogram(name: "ilogb", linkageName: "_ZL5ilogbf", scope: !19, file: !19, line: 109, type: !79, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!90 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !91, file: !19, line: 252)
!91 = !DISubprogram(name: "isfinite", linkageName: "_ZL8isfinitef", scope: !19, file: !19, line: 114, type: !92, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!92 = !DISubroutineType(types: !93)
!93 = !{!94, !26}
!94 = !DIBasicType(name: "bool", size: 8, encoding: DW_ATE_boolean)
!95 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !96, file: !19, line: 253)
!96 = !DISubprogram(name: "isgreater", linkageName: "_ZL9isgreaterff", scope: !19, file: !19, line: 118, type: !97, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!97 = !DISubroutineType(types: !98)
!98 = !{!94, !26, !26}
!99 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !100, file: !19, line: 254)
!100 = !DISubprogram(name: "isgreaterequal", linkageName: "_ZL14isgreaterequalff", scope: !19, file: !19, line: 117, type: !97, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!101 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !102, file: !19, line: 255)
!102 = !DISubprogram(name: "isinf", linkageName: "_ZL5isinff", scope: !19, file: !19, line: 123, type: !92, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!103 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !104, file: !19, line: 256)
!104 = !DISubprogram(name: "isless", linkageName: "_ZL6islessff", scope: !19, file: !19, line: 127, type: !97, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!105 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !106, file: !19, line: 257)
!106 = !DISubprogram(name: "islessequal", linkageName: "_ZL11islessequalff", scope: !19, file: !19, line: 126, type: !97, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!107 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !108, file: !19, line: 258)
!108 = !DISubprogram(name: "islessgreater", linkageName: "_ZL13islessgreaterff", scope: !19, file: !19, line: 129, type: !97, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!109 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !110, file: !19, line: 259)
!110 = !DISubprogram(name: "isnan", linkageName: "_ZL5isnanf", scope: !19, file: !19, line: 134, type: !92, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!111 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !112, file: !19, line: 260)
!112 = !DISubprogram(name: "isnormal", linkageName: "_ZL8isnormalf", scope: !19, file: !19, line: 136, type: !92, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!113 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !114, file: !19, line: 261)
!114 = !DISubprogram(name: "isunordered", linkageName: "_ZL11isunorderedff", scope: !19, file: !19, line: 138, type: !97, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!115 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !116, file: !19, line: 262)
!116 = !DISubprogram(name: "labs", linkageName: "_ZL4labsl", scope: !19, file: !19, line: 139, type: !117, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!117 = !DISubroutineType(types: !118)
!118 = !{!119, !119}
!119 = !DIBasicType(name: "long int", size: 64, encoding: DW_ATE_signed)
!120 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !121, file: !19, line: 263)
!121 = !DISubprogram(name: "ldexp", linkageName: "_ZL5ldexpfi", scope: !19, file: !19, line: 141, type: !122, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!122 = !DISubroutineType(types: !123)
!123 = !{!26, !26, !9}
!124 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !125, file: !19, line: 264)
!125 = !DISubprogram(name: "lgamma", linkageName: "_ZL6lgammaf", scope: !19, file: !19, line: 143, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!126 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !127, file: !19, line: 265)
!127 = !DISubprogram(name: "llabs", linkageName: "_ZL5llabsx", scope: !19, file: !19, line: 144, type: !128, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!128 = !DISubroutineType(types: !129)
!129 = !{!14, !14}
!130 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !131, file: !19, line: 266)
!131 = !DISubprogram(name: "llrint", linkageName: "_ZL6llrintf", scope: !19, file: !19, line: 146, type: !132, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!132 = !DISubroutineType(types: !133)
!133 = !{!14, !26}
!134 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !135, file: !19, line: 267)
!135 = !DISubprogram(name: "log", linkageName: "_ZL3logf", scope: !19, file: !19, line: 159, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!136 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !137, file: !19, line: 268)
!137 = !DISubprogram(name: "log10", linkageName: "_ZL5log10f", scope: !19, file: !19, line: 148, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!138 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !139, file: !19, line: 269)
!139 = !DISubprogram(name: "log1p", linkageName: "_ZL5log1pf", scope: !19, file: !19, line: 150, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!140 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !141, file: !19, line: 270)
!141 = !DISubprogram(name: "log2", linkageName: "_ZL4log2f", scope: !19, file: !19, line: 152, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!142 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !143, file: !19, line: 271)
!143 = !DISubprogram(name: "logb", linkageName: "_ZL4logbf", scope: !19, file: !19, line: 154, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!144 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !145, file: !19, line: 272)
!145 = !DISubprogram(name: "lrint", linkageName: "_ZL5lrintf", scope: !19, file: !19, line: 161, type: !146, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!146 = !DISubroutineType(types: !147)
!147 = !{!119, !26}
!148 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !149, file: !19, line: 273)
!149 = !DISubprogram(name: "lround", linkageName: "_ZL6lroundf", scope: !19, file: !19, line: 163, type: !146, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!150 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !151, file: !19, line: 274)
!151 = !DISubprogram(name: "llround", linkageName: "_ZL7llroundf", scope: !19, file: !19, line: 164, type: !132, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!152 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !153, file: !19, line: 275)
!153 = !DISubprogram(name: "modf", linkageName: "_ZL4modffPf", scope: !19, file: !19, line: 166, type: !154, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!154 = !DISubroutineType(types: !155)
!155 = !{!26, !26, !156}
!156 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !26, size: 64)
!157 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !158, file: !19, line: 276)
!158 = !DISubprogram(name: "nan", linkageName: "_ZL3nanPKc", scope: !19, file: !19, line: 167, type: !159, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!159 = !DISubroutineType(types: !160)
!160 = !{!161, !162}
!161 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
!162 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !163, size: 64)
!163 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !164)
!164 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!165 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !166, file: !19, line: 277)
!166 = !DISubprogram(name: "nanf", linkageName: "_ZL4nanfPKc", scope: !19, file: !19, line: 168, type: !167, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!167 = !DISubroutineType(types: !168)
!168 = !{!26, !162}
!169 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !170, file: !19, line: 278)
!170 = !DISubprogram(name: "nearbyint", linkageName: "_ZL9nearbyintf", scope: !19, file: !19, line: 170, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!171 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !172, file: !19, line: 279)
!172 = !DISubprogram(name: "nextafter", linkageName: "_ZL9nextafterff", scope: !19, file: !19, line: 172, type: !37, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!173 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !174, file: !19, line: 280)
!174 = !DISubprogram(name: "pow", linkageName: "_ZL3powfi", scope: !19, file: !19, line: 176, type: !122, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!175 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !176, file: !19, line: 281)
!176 = !DISubprogram(name: "remainder", linkageName: "_ZL9remainderff", scope: !19, file: !19, line: 178, type: !37, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!177 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !178, file: !19, line: 282)
!178 = !DISubprogram(name: "remquo", linkageName: "_ZL6remquoffPi", scope: !19, file: !19, line: 180, type: !179, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!179 = !DISubroutineType(types: !180)
!180 = !{!26, !26, !26, !85}
!181 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !182, file: !19, line: 283)
!182 = !DISubprogram(name: "rint", linkageName: "_ZL4rintf", scope: !19, file: !19, line: 182, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!183 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !184, file: !19, line: 284)
!184 = !DISubprogram(name: "round", linkageName: "_ZL5roundf", scope: !19, file: !19, line: 184, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!185 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !186, file: !19, line: 285)
!186 = !DISubprogram(name: "scalbln", linkageName: "_ZL7scalblnfl", scope: !19, file: !19, line: 186, type: !187, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!187 = !DISubroutineType(types: !188)
!188 = !{!26, !26, !119}
!189 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !190, file: !19, line: 286)
!190 = !DISubprogram(name: "scalbn", linkageName: "_ZL6scalbnfi", scope: !19, file: !19, line: 188, type: !122, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!191 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !192, file: !19, line: 287)
!192 = !DISubprogram(name: "signbit", linkageName: "_ZL7signbitf", scope: !19, file: !19, line: 190, type: !92, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!193 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !194, file: !19, line: 288)
!194 = !DISubprogram(name: "sin", linkageName: "_ZL3sinf", scope: !19, file: !19, line: 192, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!195 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !196, file: !19, line: 289)
!196 = !DISubprogram(name: "sinh", linkageName: "_ZL4sinhf", scope: !19, file: !19, line: 194, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!197 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !198, file: !19, line: 290)
!198 = !DISubprogram(name: "sqrt", linkageName: "_ZL4sqrtf", scope: !19, file: !19, line: 196, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!199 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !200, file: !19, line: 291)
!200 = !DISubprogram(name: "tan", linkageName: "_ZL3tanf", scope: !19, file: !19, line: 198, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!201 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !202, file: !19, line: 292)
!202 = !DISubprogram(name: "tanh", linkageName: "_ZL4tanhf", scope: !19, file: !19, line: 200, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!203 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !204, file: !19, line: 293)
!204 = !DISubprogram(name: "tgamma", linkageName: "_ZL6tgammaf", scope: !19, file: !19, line: 202, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!205 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !206, file: !19, line: 294)
!206 = !DISubprogram(name: "trunc", linkageName: "_ZL5truncf", scope: !19, file: !19, line: 204, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!207 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !208, file: !210, line: 52)
!208 = !DISubprogram(name: "abs", scope: !209, file: !209, line: 848, type: !20, flags: DIFlagPrototyped, spFlags: 0)
!209 = !DIFile(filename: "/usr/include/stdlib.h", directory: "")
!210 = !DIFile(filename: "/usr/lib/gcc/x86_64-linux-gnu/11/../../../../include/c++/11/bits/std_abs.h", directory: "")
!211 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !212, file: !216, line: 83)
!212 = !DISubprogram(name: "acos", scope: !213, file: !213, line: 53, type: !214, flags: DIFlagPrototyped, spFlags: 0)
!213 = !DIFile(filename: "/usr/include/x86_64-linux-gnu/bits/mathcalls.h", directory: "")
!214 = !DISubroutineType(types: !215)
!215 = !{!161, !161}
!216 = !DIFile(filename: "/usr/lib/gcc/x86_64-linux-gnu/11/../../../../include/c++/11/cmath", directory: "")
!217 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !218, file: !216, line: 102)
!218 = !DISubprogram(name: "asin", scope: !213, file: !213, line: 55, type: !214, flags: DIFlagPrototyped, spFlags: 0)
!219 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !220, file: !216, line: 121)
!220 = !DISubprogram(name: "atan", scope: !213, file: !213, line: 57, type: !214, flags: DIFlagPrototyped, spFlags: 0)
!221 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !222, file: !216, line: 140)
!222 = !DISubprogram(name: "atan2", scope: !213, file: !213, line: 59, type: !223, flags: DIFlagPrototyped, spFlags: 0)
!223 = !DISubroutineType(types: !224)
!224 = !{!161, !161, !161}
!225 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !226, file: !216, line: 161)
!226 = !DISubprogram(name: "ceil", scope: !213, file: !213, line: 159, type: !214, flags: DIFlagPrototyped, spFlags: 0)
!227 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !228, file: !216, line: 180)
!228 = !DISubprogram(name: "cos", scope: !213, file: !213, line: 62, type: !214, flags: DIFlagPrototyped, spFlags: 0)
!229 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !230, file: !216, line: 199)
!230 = !DISubprogram(name: "cosh", scope: !213, file: !213, line: 71, type: !214, flags: DIFlagPrototyped, spFlags: 0)
!231 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !232, file: !216, line: 218)
!232 = !DISubprogram(name: "exp", scope: !213, file: !213, line: 95, type: !214, flags: DIFlagPrototyped, spFlags: 0)
!233 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !234, file: !216, line: 237)
!234 = !DISubprogram(name: "fabs", scope: !213, file: !213, line: 162, type: !214, flags: DIFlagPrototyped, spFlags: 0)
!235 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !236, file: !216, line: 256)
!236 = !DISubprogram(name: "floor", scope: !213, file: !213, line: 165, type: !214, flags: DIFlagPrototyped, spFlags: 0)
!237 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !238, file: !216, line: 275)
!238 = !DISubprogram(name: "fmod", scope: !213, file: !213, line: 168, type: !223, flags: DIFlagPrototyped, spFlags: 0)
!239 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !240, file: !216, line: 296)
!240 = !DISubprogram(name: "frexp", scope: !213, file: !213, line: 98, type: !241, flags: DIFlagPrototyped, spFlags: 0)
!241 = !DISubroutineType(types: !242)
!242 = !{!161, !161, !85}
!243 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !244, file: !216, line: 315)
!244 = !DISubprogram(name: "ldexp", scope: !213, file: !213, line: 101, type: !245, flags: DIFlagPrototyped, spFlags: 0)
!245 = !DISubroutineType(types: !246)
!246 = !{!161, !161, !9}
!247 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !248, file: !216, line: 334)
!248 = !DISubprogram(name: "log", scope: !213, file: !213, line: 104, type: !214, flags: DIFlagPrototyped, spFlags: 0)
!249 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !250, file: !216, line: 353)
!250 = !DISubprogram(name: "log10", scope: !213, file: !213, line: 107, type: !214, flags: DIFlagPrototyped, spFlags: 0)
!251 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !252, file: !216, line: 372)
!252 = !DISubprogram(name: "modf", scope: !213, file: !213, line: 110, type: !253, flags: DIFlagPrototyped, spFlags: 0)
!253 = !DISubroutineType(types: !254)
!254 = !{!161, !161, !255}
!255 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !161, size: 64)
!256 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !257, file: !216, line: 384)
!257 = !DISubprogram(name: "pow", scope: !213, file: !213, line: 140, type: !223, flags: DIFlagPrototyped, spFlags: 0)
!258 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !259, file: !216, line: 421)
!259 = !DISubprogram(name: "sin", scope: !213, file: !213, line: 64, type: !214, flags: DIFlagPrototyped, spFlags: 0)
!260 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !261, file: !216, line: 440)
!261 = !DISubprogram(name: "sinh", scope: !213, file: !213, line: 73, type: !214, flags: DIFlagPrototyped, spFlags: 0)
!262 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !263, file: !216, line: 459)
!263 = !DISubprogram(name: "sqrt", scope: !213, file: !213, line: 143, type: !214, flags: DIFlagPrototyped, spFlags: 0)
!264 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !265, file: !216, line: 478)
!265 = !DISubprogram(name: "tan", scope: !213, file: !213, line: 66, type: !214, flags: DIFlagPrototyped, spFlags: 0)
!266 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !267, file: !216, line: 497)
!267 = !DISubprogram(name: "tanh", scope: !213, file: !213, line: 75, type: !214, flags: DIFlagPrototyped, spFlags: 0)
!268 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !269, file: !271, line: 127)
!269 = !DIDerivedType(tag: DW_TAG_typedef, name: "div_t", file: !209, line: 63, baseType: !270)
!270 = !DICompositeType(tag: DW_TAG_structure_type, file: !209, line: 59, flags: DIFlagFwdDecl, identifier: "_ZTS5div_t")
!271 = !DIFile(filename: "/usr/lib/gcc/x86_64-linux-gnu/11/../../../../include/c++/11/cstdlib", directory: "")
!272 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !273, file: !271, line: 128)
!273 = !DIDerivedType(tag: DW_TAG_typedef, name: "ldiv_t", file: !209, line: 71, baseType: !274)
!274 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !209, line: 67, size: 128, flags: DIFlagTypePassByValue, elements: !275, identifier: "_ZTS6ldiv_t")
!275 = !{!276, !277}
!276 = !DIDerivedType(tag: DW_TAG_member, name: "quot", scope: !274, file: !209, line: 69, baseType: !119, size: 64)
!277 = !DIDerivedType(tag: DW_TAG_member, name: "rem", scope: !274, file: !209, line: 70, baseType: !119, size: 64, offset: 64)
!278 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !279, file: !271, line: 130)
!279 = !DISubprogram(name: "abort", scope: !209, file: !209, line: 598, type: !280, flags: DIFlagPrototyped | DIFlagNoReturn, spFlags: 0)
!280 = !DISubroutineType(types: !281)
!281 = !{null}
!282 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !283, file: !271, line: 134)
!283 = !DISubprogram(name: "atexit", scope: !209, file: !209, line: 602, type: !284, flags: DIFlagPrototyped, spFlags: 0)
!284 = !DISubroutineType(types: !285)
!285 = !{!9, !286}
!286 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !280, size: 64)
!287 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !288, file: !271, line: 140)
!288 = !DISubprogram(name: "atof", scope: !209, file: !209, line: 102, type: !159, flags: DIFlagPrototyped, spFlags: 0)
!289 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !290, file: !271, line: 141)
!290 = !DISubprogram(name: "atoi", scope: !209, file: !209, line: 105, type: !291, flags: DIFlagPrototyped, spFlags: 0)
!291 = !DISubroutineType(types: !292)
!292 = !{!9, !162}
!293 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !294, file: !271, line: 142)
!294 = !DISubprogram(name: "atol", scope: !209, file: !209, line: 108, type: !295, flags: DIFlagPrototyped, spFlags: 0)
!295 = !DISubroutineType(types: !296)
!296 = !{!119, !162}
!297 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !298, file: !271, line: 143)
!298 = !DISubprogram(name: "bsearch", scope: !209, file: !209, line: 828, type: !299, flags: DIFlagPrototyped, spFlags: 0)
!299 = !DISubroutineType(types: !300)
!300 = !{!301, !302, !302, !304, !304, !307}
!301 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!302 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !303, size: 64)
!303 = !DIDerivedType(tag: DW_TAG_const_type, baseType: null)
!304 = !DIDerivedType(tag: DW_TAG_typedef, name: "size_t", file: !305, line: 46, baseType: !306)
!305 = !DIFile(filename: "llvm-install-tulip/lib/clang/9.0.0/include/stddef.h", directory: "/scratch/ah7226")
!306 = !DIBasicType(name: "long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!307 = !DIDerivedType(tag: DW_TAG_typedef, name: "__compar_fn_t", file: !209, line: 816, baseType: !308)
!308 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !309, size: 64)
!309 = !DISubroutineType(types: !310)
!310 = !{!9, !302, !302}
!311 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !312, file: !271, line: 144)
!312 = !DISubprogram(name: "calloc", scope: !209, file: !209, line: 543, type: !313, flags: DIFlagPrototyped, spFlags: 0)
!313 = !DISubroutineType(types: !314)
!314 = !{!301, !304, !304}
!315 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !316, file: !271, line: 145)
!316 = !DISubprogram(name: "div", scope: !209, file: !209, line: 860, type: !317, flags: DIFlagPrototyped, spFlags: 0)
!317 = !DISubroutineType(types: !318)
!318 = !{!269, !9, !9}
!319 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !320, file: !271, line: 146)
!320 = !DISubprogram(name: "exit", scope: !209, file: !209, line: 624, type: !321, flags: DIFlagPrototyped | DIFlagNoReturn, spFlags: 0)
!321 = !DISubroutineType(types: !322)
!322 = !{null, !9}
!323 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !324, file: !271, line: 147)
!324 = !DISubprogram(name: "free", scope: !209, file: !209, line: 555, type: !325, flags: DIFlagPrototyped, spFlags: 0)
!325 = !DISubroutineType(types: !326)
!326 = !{null, !301}
!327 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !328, file: !271, line: 148)
!328 = !DISubprogram(name: "getenv", scope: !209, file: !209, line: 641, type: !329, flags: DIFlagPrototyped, spFlags: 0)
!329 = !DISubroutineType(types: !330)
!330 = !{!331, !162}
!331 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !164, size: 64)
!332 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !333, file: !271, line: 149)
!333 = !DISubprogram(name: "labs", scope: !209, file: !209, line: 849, type: !117, flags: DIFlagPrototyped, spFlags: 0)
!334 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !335, file: !271, line: 150)
!335 = !DISubprogram(name: "ldiv", scope: !209, file: !209, line: 862, type: !336, flags: DIFlagPrototyped, spFlags: 0)
!336 = !DISubroutineType(types: !337)
!337 = !{!273, !119, !119}
!338 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !339, file: !271, line: 151)
!339 = !DISubprogram(name: "malloc", scope: !209, file: !209, line: 540, type: !340, flags: DIFlagPrototyped, spFlags: 0)
!340 = !DISubroutineType(types: !341)
!341 = !{!301, !304}
!342 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !343, file: !271, line: 153)
!343 = !DISubprogram(name: "mblen", scope: !209, file: !209, line: 930, type: !344, flags: DIFlagPrototyped, spFlags: 0)
!344 = !DISubroutineType(types: !345)
!345 = !{!9, !162, !304}
!346 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !347, file: !271, line: 154)
!347 = !DISubprogram(name: "mbstowcs", scope: !209, file: !209, line: 941, type: !348, flags: DIFlagPrototyped, spFlags: 0)
!348 = !DISubroutineType(types: !349)
!349 = !{!304, !350, !353, !304}
!350 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !351)
!351 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !352, size: 64)
!352 = !DIBasicType(name: "wchar_t", size: 32, encoding: DW_ATE_signed)
!353 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !162)
!354 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !355, file: !271, line: 155)
!355 = !DISubprogram(name: "mbtowc", scope: !209, file: !209, line: 933, type: !356, flags: DIFlagPrototyped, spFlags: 0)
!356 = !DISubroutineType(types: !357)
!357 = !{!9, !350, !353, !304}
!358 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !359, file: !271, line: 157)
!359 = !DISubprogram(name: "qsort", scope: !209, file: !209, line: 838, type: !360, flags: DIFlagPrototyped, spFlags: 0)
!360 = !DISubroutineType(types: !361)
!361 = !{null, !301, !304, !304, !307}
!362 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !363, file: !271, line: 163)
!363 = !DISubprogram(name: "rand", scope: !209, file: !209, line: 454, type: !364, flags: DIFlagPrototyped, spFlags: 0)
!364 = !DISubroutineType(types: !365)
!365 = !{!9}
!366 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !367, file: !271, line: 164)
!367 = !DISubprogram(name: "realloc", scope: !209, file: !209, line: 551, type: !368, flags: DIFlagPrototyped, spFlags: 0)
!368 = !DISubroutineType(types: !369)
!369 = !{!301, !301, !304}
!370 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !371, file: !271, line: 165)
!371 = !DISubprogram(name: "srand", scope: !209, file: !209, line: 456, type: !372, flags: DIFlagPrototyped, spFlags: 0)
!372 = !DISubroutineType(types: !373)
!373 = !{null, !374}
!374 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!375 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !376, file: !271, line: 166)
!376 = !DISubprogram(name: "strtod", scope: !209, file: !209, line: 118, type: !377, flags: DIFlagPrototyped, spFlags: 0)
!377 = !DISubroutineType(types: !378)
!378 = !{!161, !353, !379}
!379 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !380)
!380 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !331, size: 64)
!381 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !382, file: !271, line: 167)
!382 = !DISubprogram(name: "strtol", scope: !209, file: !209, line: 177, type: !383, flags: DIFlagPrototyped, spFlags: 0)
!383 = !DISubroutineType(types: !384)
!384 = !{!119, !353, !379, !9}
!385 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !386, file: !271, line: 168)
!386 = !DISubprogram(name: "strtoul", scope: !209, file: !209, line: 181, type: !387, flags: DIFlagPrototyped, spFlags: 0)
!387 = !DISubroutineType(types: !388)
!388 = !{!306, !353, !379, !9}
!389 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !390, file: !271, line: 169)
!390 = !DISubprogram(name: "system", scope: !209, file: !209, line: 791, type: !291, flags: DIFlagPrototyped, spFlags: 0)
!391 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !392, file: !271, line: 171)
!392 = !DISubprogram(name: "wcstombs", scope: !209, file: !209, line: 945, type: !393, flags: DIFlagPrototyped, spFlags: 0)
!393 = !DISubroutineType(types: !394)
!394 = !{!304, !395, !396, !304}
!395 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !331)
!396 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !397)
!397 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !398, size: 64)
!398 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !352)
!399 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !400, file: !271, line: 172)
!400 = !DISubprogram(name: "wctomb", scope: !209, file: !209, line: 937, type: !401, flags: DIFlagPrototyped, spFlags: 0)
!401 = !DISubroutineType(types: !402)
!402 = !{!9, !331, !352}
!403 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !404, entity: !405, file: !271, line: 200)
!404 = !DINamespace(name: "__gnu_cxx", scope: null)
!405 = !DIDerivedType(tag: DW_TAG_typedef, name: "lldiv_t", file: !209, line: 81, baseType: !406)
!406 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !209, line: 77, size: 128, flags: DIFlagTypePassByValue, elements: !407, identifier: "_ZTS7lldiv_t")
!407 = !{!408, !409}
!408 = !DIDerivedType(tag: DW_TAG_member, name: "quot", scope: !406, file: !209, line: 79, baseType: !14, size: 64)
!409 = !DIDerivedType(tag: DW_TAG_member, name: "rem", scope: !406, file: !209, line: 80, baseType: !14, size: 64, offset: 64)
!410 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !404, entity: !411, file: !271, line: 206)
!411 = !DISubprogram(name: "_Exit", scope: !209, file: !209, line: 636, type: !321, flags: DIFlagPrototyped | DIFlagNoReturn, spFlags: 0)
!412 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !404, entity: !413, file: !271, line: 210)
!413 = !DISubprogram(name: "llabs", scope: !209, file: !209, line: 852, type: !128, flags: DIFlagPrototyped, spFlags: 0)
!414 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !404, entity: !415, file: !271, line: 216)
!415 = !DISubprogram(name: "lldiv", scope: !209, file: !209, line: 866, type: !416, flags: DIFlagPrototyped, spFlags: 0)
!416 = !DISubroutineType(types: !417)
!417 = !{!405, !14, !14}
!418 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !404, entity: !419, file: !271, line: 227)
!419 = !DISubprogram(name: "atoll", scope: !209, file: !209, line: 113, type: !420, flags: DIFlagPrototyped, spFlags: 0)
!420 = !DISubroutineType(types: !421)
!421 = !{!14, !162}
!422 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !404, entity: !423, file: !271, line: 228)
!423 = !DISubprogram(name: "strtoll", scope: !209, file: !209, line: 201, type: !424, flags: DIFlagPrototyped, spFlags: 0)
!424 = !DISubroutineType(types: !425)
!425 = !{!14, !353, !379, !9}
!426 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !404, entity: !427, file: !271, line: 229)
!427 = !DISubprogram(name: "strtoull", scope: !209, file: !209, line: 206, type: !428, flags: DIFlagPrototyped, spFlags: 0)
!428 = !DISubroutineType(types: !429)
!429 = !{!11, !353, !379, !9}
!430 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !404, entity: !431, file: !271, line: 231)
!431 = !DISubprogram(name: "strtof", scope: !209, file: !209, line: 124, type: !432, flags: DIFlagPrototyped, spFlags: 0)
!432 = !DISubroutineType(types: !433)
!433 = !{!26, !353, !379}
!434 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !404, entity: !435, file: !271, line: 232)
!435 = !DISubprogram(name: "strtold", scope: !209, file: !209, line: 127, type: !436, flags: DIFlagPrototyped, spFlags: 0)
!436 = !DISubroutineType(types: !437)
!437 = !{!438, !353, !379}
!438 = !DIBasicType(name: "long double", size: 64, encoding: DW_ATE_float)
!439 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !405, file: !271, line: 240)
!440 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !411, file: !271, line: 242)
!441 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !413, file: !271, line: 244)
!442 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !443, file: !271, line: 245)
!443 = !DISubprogram(name: "div", linkageName: "_ZN9__gnu_cxx3divExx", scope: !404, file: !271, line: 213, type: !416, flags: DIFlagPrototyped, spFlags: 0)
!444 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !415, file: !271, line: 246)
!445 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !419, file: !271, line: 248)
!446 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !431, file: !271, line: 249)
!447 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !423, file: !271, line: 250)
!448 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !427, file: !271, line: 251)
!449 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !435, file: !271, line: 252)
!450 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !279, file: !451, line: 38)
!451 = !DIFile(filename: "/usr/lib/gcc/x86_64-linux-gnu/11/../../../../include/c++/11/stdlib.h", directory: "")
!452 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !283, file: !451, line: 39)
!453 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !320, file: !451, line: 40)
!454 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !269, file: !451, line: 51)
!455 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !273, file: !451, line: 52)
!456 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !457, file: !451, line: 54)
!457 = !DISubprogram(name: "abs", linkageName: "_ZSt3abse", scope: !17, file: !210, line: 79, type: !458, flags: DIFlagPrototyped, spFlags: 0)
!458 = !DISubroutineType(types: !459)
!459 = !{!438, !438}
!460 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !288, file: !451, line: 55)
!461 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !290, file: !451, line: 56)
!462 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !294, file: !451, line: 57)
!463 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !298, file: !451, line: 58)
!464 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !312, file: !451, line: 59)
!465 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !443, file: !451, line: 60)
!466 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !324, file: !451, line: 61)
!467 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !328, file: !451, line: 62)
!468 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !333, file: !451, line: 63)
!469 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !335, file: !451, line: 64)
!470 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !339, file: !451, line: 65)
!471 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !343, file: !451, line: 67)
!472 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !347, file: !451, line: 68)
!473 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !355, file: !451, line: 69)
!474 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !359, file: !451, line: 71)
!475 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !363, file: !451, line: 72)
!476 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !367, file: !451, line: 73)
!477 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !371, file: !451, line: 74)
!478 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !376, file: !451, line: 75)
!479 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !382, file: !451, line: 76)
!480 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !386, file: !451, line: 77)
!481 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !390, file: !451, line: 78)
!482 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !392, file: !451, line: 80)
!483 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !400, file: !451, line: 81)
!484 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !485, file: !487, line: 414)
!485 = !DISubprogram(name: "acosf", linkageName: "_ZL5acosff", scope: !486, file: !486, line: 1126, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!486 = !DIFile(filename: "/usr/local/cuda/include/math_functions.hpp", directory: "")
!487 = !DIFile(filename: "llvm-install-tulip/lib/clang/9.0.0/include/__clang_cuda_cmath.h", directory: "/scratch/ah7226")
!488 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !489, file: !487, line: 415)
!489 = !DISubprogram(name: "acoshf", linkageName: "_ZL6acoshff", scope: !486, file: !486, line: 1154, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!490 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !491, file: !487, line: 416)
!491 = !DISubprogram(name: "asinf", linkageName: "_ZL5asinff", scope: !486, file: !486, line: 1121, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!492 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !493, file: !487, line: 417)
!493 = !DISubprogram(name: "asinhf", linkageName: "_ZL6asinhff", scope: !486, file: !486, line: 1159, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!494 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !495, file: !487, line: 418)
!495 = !DISubprogram(name: "atan2f", linkageName: "_ZL6atan2fff", scope: !486, file: !486, line: 1111, type: !37, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!496 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !497, file: !487, line: 419)
!497 = !DISubprogram(name: "atanf", linkageName: "_ZL5atanff", scope: !486, file: !486, line: 1116, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!498 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !499, file: !487, line: 420)
!499 = !DISubprogram(name: "atanhf", linkageName: "_ZL6atanhff", scope: !486, file: !486, line: 1164, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!500 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !501, file: !487, line: 421)
!501 = !DISubprogram(name: "cbrtf", linkageName: "_ZL5cbrtff", scope: !486, file: !486, line: 1199, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!502 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !503, file: !487, line: 422)
!503 = !DISubprogram(name: "ceilf", linkageName: "_ZL5ceilff", scope: !504, file: !504, line: 647, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!504 = !DIFile(filename: "/usr/local/cuda/include/device_functions.hpp", directory: "")
!505 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !506, file: !487, line: 423)
!506 = !DISubprogram(name: "copysignf", linkageName: "_ZL9copysignfff", scope: !486, file: !486, line: 973, type: !37, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!507 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !508, file: !487, line: 424)
!508 = !DISubprogram(name: "cosf", linkageName: "_ZL4cosff", scope: !486, file: !486, line: 1027, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!509 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !510, file: !487, line: 425)
!510 = !DISubprogram(name: "coshf", linkageName: "_ZL5coshff", scope: !486, file: !486, line: 1096, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!511 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !512, file: !487, line: 426)
!512 = !DISubprogram(name: "erfcf", linkageName: "_ZL5erfcff", scope: !486, file: !486, line: 1259, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!513 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !514, file: !487, line: 427)
!514 = !DISubprogram(name: "erff", linkageName: "_ZL4erfff", scope: !486, file: !486, line: 1249, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!515 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !516, file: !487, line: 428)
!516 = !DISubprogram(name: "exp2f", linkageName: "_ZL5exp2ff", scope: !504, file: !504, line: 637, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!517 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !518, file: !487, line: 429)
!518 = !DISubprogram(name: "expf", linkageName: "_ZL4expff", scope: !486, file: !486, line: 1078, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!519 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !520, file: !487, line: 430)
!520 = !DISubprogram(name: "expm1f", linkageName: "_ZL6expm1ff", scope: !486, file: !486, line: 1169, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!521 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !522, file: !487, line: 431)
!522 = !DISubprogram(name: "fabsf", linkageName: "_ZL5fabsff", scope: !504, file: !504, line: 582, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!523 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !524, file: !487, line: 432)
!524 = !DISubprogram(name: "fdimf", linkageName: "_ZL5fdimfff", scope: !486, file: !486, line: 1385, type: !37, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!525 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !526, file: !487, line: 433)
!526 = !DISubprogram(name: "floorf", linkageName: "_ZL6floorff", scope: !504, file: !504, line: 572, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!527 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !528, file: !487, line: 434)
!528 = !DISubprogram(name: "fmaf", linkageName: "_ZL4fmaffff", scope: !486, file: !486, line: 1337, type: !69, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!529 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !530, file: !487, line: 435)
!530 = !DISubprogram(name: "fmaxf", linkageName: "_ZL5fmaxfff", scope: !504, file: !504, line: 602, type: !37, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!531 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !532, file: !487, line: 436)
!532 = !DISubprogram(name: "fminf", linkageName: "_ZL5fminfff", scope: !504, file: !504, line: 597, type: !37, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!533 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !534, file: !487, line: 437)
!534 = !DISubprogram(name: "fmodf", linkageName: "_ZL5fmodfff", scope: !486, file: !486, line: 1322, type: !37, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!535 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !536, file: !487, line: 438)
!536 = !DISubprogram(name: "frexpf", linkageName: "_ZL6frexpffPi", scope: !486, file: !486, line: 1312, type: !83, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!537 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !538, file: !487, line: 439)
!538 = !DISubprogram(name: "hypotf", linkageName: "_ZL6hypotfff", scope: !486, file: !486, line: 1174, type: !37, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!539 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !540, file: !487, line: 440)
!540 = !DISubprogram(name: "ilogbf", linkageName: "_ZL6ilogbff", scope: !486, file: !486, line: 1390, type: !79, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!541 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !542, file: !487, line: 441)
!542 = !DISubprogram(name: "ldexpf", linkageName: "_ZL6ldexpffi", scope: !486, file: !486, line: 1289, type: !122, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!543 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !544, file: !487, line: 442)
!544 = !DISubprogram(name: "lgammaf", linkageName: "_ZL7lgammaff", scope: !486, file: !486, line: 1284, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!545 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !546, file: !487, line: 443)
!546 = !DISubprogram(name: "llrintf", linkageName: "_ZL7llrintff", scope: !486, file: !486, line: 933, type: !132, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!547 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !548, file: !487, line: 444)
!548 = !DISubprogram(name: "llroundf", linkageName: "_ZL8llroundff", scope: !486, file: !486, line: 1371, type: !132, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!549 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !550, file: !487, line: 445)
!550 = !DISubprogram(name: "log10f", linkageName: "_ZL6log10ff", scope: !486, file: !486, line: 1140, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!551 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !552, file: !487, line: 446)
!552 = !DISubprogram(name: "log1pf", linkageName: "_ZL6log1pff", scope: !486, file: !486, line: 1149, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!553 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !554, file: !487, line: 447)
!554 = !DISubprogram(name: "log2f", linkageName: "_ZL5log2ff", scope: !486, file: !486, line: 1069, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!555 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !556, file: !487, line: 448)
!556 = !DISubprogram(name: "logbf", linkageName: "_ZL5logbff", scope: !486, file: !486, line: 1395, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!557 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !558, file: !487, line: 449)
!558 = !DISubprogram(name: "logf", linkageName: "_ZL4logff", scope: !486, file: !486, line: 1131, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!559 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !560, file: !487, line: 450)
!560 = !DISubprogram(name: "lrintf", linkageName: "_ZL6lrintff", scope: !486, file: !486, line: 924, type: !146, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!561 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !562, file: !487, line: 451)
!562 = !DISubprogram(name: "lroundf", linkageName: "_ZL7lroundff", scope: !486, file: !486, line: 1376, type: !146, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!563 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !564, file: !487, line: 452)
!564 = !DISubprogram(name: "modff", linkageName: "_ZL5modfffPf", scope: !486, file: !486, line: 1317, type: !154, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!565 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !566, file: !487, line: 453)
!566 = !DISubprogram(name: "nearbyintf", linkageName: "_ZL10nearbyintff", scope: !486, file: !486, line: 938, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!567 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !568, file: !487, line: 454)
!568 = !DISubprogram(name: "nextafterf", linkageName: "_ZL10nextafterfff", scope: !486, file: !486, line: 1002, type: !37, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!569 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !570, file: !487, line: 455)
!570 = !DISubprogram(name: "powf", linkageName: "_ZL4powfff", scope: !486, file: !486, line: 1352, type: !37, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!571 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !572, file: !487, line: 456)
!572 = !DISubprogram(name: "remainderf", linkageName: "_ZL10remainderfff", scope: !486, file: !486, line: 1327, type: !37, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!573 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !574, file: !487, line: 457)
!574 = !DISubprogram(name: "remquof", linkageName: "_ZL7remquofffPi", scope: !486, file: !486, line: 1332, type: !179, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!575 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !576, file: !487, line: 458)
!576 = !DISubprogram(name: "rintf", linkageName: "_ZL5rintff", scope: !486, file: !486, line: 919, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!577 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !578, file: !487, line: 459)
!578 = !DISubprogram(name: "roundf", linkageName: "_ZL6roundff", scope: !486, file: !486, line: 1366, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!579 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !580, file: !487, line: 462)
!580 = !DISubprogram(name: "scalblnf", linkageName: "_ZL8scalblnffl", scope: !486, file: !486, line: 1299, type: !187, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!581 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !582, file: !487, line: 464)
!582 = !DISubprogram(name: "scalbnf", linkageName: "_ZL7scalbnffi", scope: !486, file: !486, line: 1294, type: !122, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!583 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !584, file: !487, line: 465)
!584 = !DISubprogram(name: "sinf", linkageName: "_ZL4sinff", scope: !486, file: !486, line: 1018, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!585 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !586, file: !487, line: 466)
!586 = !DISubprogram(name: "sinhf", linkageName: "_ZL5sinhff", scope: !486, file: !486, line: 1101, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!587 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !588, file: !487, line: 467)
!588 = !DISubprogram(name: "sqrtf", linkageName: "_ZL5sqrtff", scope: !504, file: !504, line: 887, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!589 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !590, file: !487, line: 468)
!590 = !DISubprogram(name: "tanf", linkageName: "_ZL4tanff", scope: !486, file: !486, line: 1060, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!591 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !592, file: !487, line: 469)
!592 = !DISubprogram(name: "tanhf", linkageName: "_ZL5tanhff", scope: !486, file: !486, line: 1106, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!593 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !594, file: !487, line: 470)
!594 = !DISubprogram(name: "tgammaf", linkageName: "_ZL7tgammaff", scope: !486, file: !486, line: 1361, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!595 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !596, file: !487, line: 471)
!596 = !DISubprogram(name: "truncf", linkageName: "_ZL6truncff", scope: !504, file: !504, line: 642, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!597 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !598, file: !602, line: 98)
!598 = !DIDerivedType(tag: DW_TAG_typedef, name: "FILE", file: !599, line: 7, baseType: !600)
!599 = !DIFile(filename: "/usr/include/x86_64-linux-gnu/bits/types/FILE.h", directory: "")
!600 = !DICompositeType(tag: DW_TAG_structure_type, name: "_IO_FILE", file: !601, line: 49, flags: DIFlagFwdDecl, identifier: "_ZTS8_IO_FILE")
!601 = !DIFile(filename: "/usr/include/x86_64-linux-gnu/bits/types/struct_FILE.h", directory: "")
!602 = !DIFile(filename: "/usr/lib/gcc/x86_64-linux-gnu/11/../../../../include/c++/11/cstdio", directory: "")
!603 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !604, file: !602, line: 99)
!604 = !DIDerivedType(tag: DW_TAG_typedef, name: "fpos_t", file: !605, line: 84, baseType: !606)
!605 = !DIFile(filename: "/usr/include/stdio.h", directory: "")
!606 = !DIDerivedType(tag: DW_TAG_typedef, name: "__fpos_t", file: !607, line: 14, baseType: !608)
!607 = !DIFile(filename: "/usr/include/x86_64-linux-gnu/bits/types/__fpos_t.h", directory: "")
!608 = !DICompositeType(tag: DW_TAG_structure_type, name: "_G_fpos_t", file: !607, line: 10, flags: DIFlagFwdDecl, identifier: "_ZTS9_G_fpos_t")
!609 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !610, file: !602, line: 101)
!610 = !DISubprogram(name: "clearerr", scope: !605, file: !605, line: 786, type: !611, flags: DIFlagPrototyped, spFlags: 0)
!611 = !DISubroutineType(types: !612)
!612 = !{null, !613}
!613 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !598, size: 64)
!614 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !615, file: !602, line: 102)
!615 = !DISubprogram(name: "fclose", scope: !605, file: !605, line: 178, type: !616, flags: DIFlagPrototyped, spFlags: 0)
!616 = !DISubroutineType(types: !617)
!617 = !{!9, !613}
!618 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !619, file: !602, line: 103)
!619 = !DISubprogram(name: "feof", scope: !605, file: !605, line: 788, type: !616, flags: DIFlagPrototyped, spFlags: 0)
!620 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !621, file: !602, line: 104)
!621 = !DISubprogram(name: "ferror", scope: !605, file: !605, line: 790, type: !616, flags: DIFlagPrototyped, spFlags: 0)
!622 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !623, file: !602, line: 105)
!623 = !DISubprogram(name: "fflush", scope: !605, file: !605, line: 230, type: !616, flags: DIFlagPrototyped, spFlags: 0)
!624 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !625, file: !602, line: 106)
!625 = !DISubprogram(name: "fgetc", scope: !605, file: !605, line: 513, type: !616, flags: DIFlagPrototyped, spFlags: 0)
!626 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !627, file: !602, line: 107)
!627 = !DISubprogram(name: "fgetpos", scope: !605, file: !605, line: 760, type: !628, flags: DIFlagPrototyped, spFlags: 0)
!628 = !DISubroutineType(types: !629)
!629 = !{!9, !630, !631}
!630 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !613)
!631 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !632)
!632 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !604, size: 64)
!633 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !634, file: !602, line: 108)
!634 = !DISubprogram(name: "fgets", scope: !605, file: !605, line: 592, type: !635, flags: DIFlagPrototyped, spFlags: 0)
!635 = !DISubroutineType(types: !636)
!636 = !{!331, !395, !9, !630}
!637 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !638, file: !602, line: 109)
!638 = !DISubprogram(name: "fopen", scope: !605, file: !605, line: 258, type: !639, flags: DIFlagPrototyped, spFlags: 0)
!639 = !DISubroutineType(types: !640)
!640 = !{!613, !353, !353}
!641 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !642, file: !602, line: 110)
!642 = !DISubprogram(name: "fprintf", scope: !605, file: !605, line: 350, type: !643, flags: DIFlagPrototyped, spFlags: 0)
!643 = !DISubroutineType(types: !644)
!644 = !{!9, !630, !353, null}
!645 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !646, file: !602, line: 111)
!646 = !DISubprogram(name: "fputc", scope: !605, file: !605, line: 549, type: !647, flags: DIFlagPrototyped, spFlags: 0)
!647 = !DISubroutineType(types: !648)
!648 = !{!9, !9, !613}
!649 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !650, file: !602, line: 112)
!650 = !DISubprogram(name: "fputs", scope: !605, file: !605, line: 655, type: !651, flags: DIFlagPrototyped, spFlags: 0)
!651 = !DISubroutineType(types: !652)
!652 = !{!9, !353, !630}
!653 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !654, file: !602, line: 113)
!654 = !DISubprogram(name: "fread", scope: !605, file: !605, line: 675, type: !655, flags: DIFlagPrototyped, spFlags: 0)
!655 = !DISubroutineType(types: !656)
!656 = !{!304, !657, !304, !304, !630}
!657 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !301)
!658 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !659, file: !602, line: 114)
!659 = !DISubprogram(name: "freopen", scope: !605, file: !605, line: 265, type: !660, flags: DIFlagPrototyped, spFlags: 0)
!660 = !DISubroutineType(types: !661)
!661 = !{!613, !353, !353, !630}
!662 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !663, file: !602, line: 115)
!663 = !DISubprogram(name: "fscanf", scope: !605, file: !605, line: 415, type: !643, flags: DIFlagPrototyped, spFlags: 0)
!664 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !665, file: !602, line: 116)
!665 = !DISubprogram(name: "fseek", scope: !605, file: !605, line: 713, type: !666, flags: DIFlagPrototyped, spFlags: 0)
!666 = !DISubroutineType(types: !667)
!667 = !{!9, !613, !119, !9}
!668 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !669, file: !602, line: 117)
!669 = !DISubprogram(name: "fsetpos", scope: !605, file: !605, line: 765, type: !670, flags: DIFlagPrototyped, spFlags: 0)
!670 = !DISubroutineType(types: !671)
!671 = !{!9, !613, !672}
!672 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !673, size: 64)
!673 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !604)
!674 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !675, file: !602, line: 118)
!675 = !DISubprogram(name: "ftell", scope: !605, file: !605, line: 718, type: !676, flags: DIFlagPrototyped, spFlags: 0)
!676 = !DISubroutineType(types: !677)
!677 = !{!119, !613}
!678 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !679, file: !602, line: 119)
!679 = !DISubprogram(name: "fwrite", scope: !605, file: !605, line: 681, type: !680, flags: DIFlagPrototyped, spFlags: 0)
!680 = !DISubroutineType(types: !681)
!681 = !{!304, !682, !304, !304, !630}
!682 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !302)
!683 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !684, file: !602, line: 120)
!684 = !DISubprogram(name: "getc", scope: !605, file: !605, line: 514, type: !616, flags: DIFlagPrototyped, spFlags: 0)
!685 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !686, file: !602, line: 121)
!686 = !DISubprogram(name: "getchar", scope: !605, file: !605, line: 520, type: !364, flags: DIFlagPrototyped, spFlags: 0)
!687 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !688, file: !602, line: 124)
!688 = !DISubprogram(name: "gets", scope: !605, file: !605, line: 605, type: !689, flags: DIFlagPrototyped, spFlags: 0)
!689 = !DISubroutineType(types: !690)
!690 = !{!331, !331}
!691 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !692, file: !602, line: 126)
!692 = !DISubprogram(name: "perror", scope: !605, file: !605, line: 804, type: !693, flags: DIFlagPrototyped, spFlags: 0)
!693 = !DISubroutineType(types: !694)
!694 = !{null, !162}
!695 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !696, file: !602, line: 127)
!696 = !DISubprogram(name: "printf", scope: !605, file: !605, line: 356, type: !697, flags: DIFlagPrototyped, spFlags: 0)
!697 = !DISubroutineType(types: !698)
!698 = !{!9, !353, null}
!699 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !700, file: !602, line: 128)
!700 = !DISubprogram(name: "putc", scope: !605, file: !605, line: 550, type: !647, flags: DIFlagPrototyped, spFlags: 0)
!701 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !702, file: !602, line: 129)
!702 = !DISubprogram(name: "putchar", scope: !605, file: !605, line: 556, type: !20, flags: DIFlagPrototyped, spFlags: 0)
!703 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !704, file: !602, line: 130)
!704 = !DISubprogram(name: "puts", scope: !605, file: !605, line: 661, type: !291, flags: DIFlagPrototyped, spFlags: 0)
!705 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !706, file: !602, line: 131)
!706 = !DISubprogram(name: "remove", scope: !605, file: !605, line: 152, type: !291, flags: DIFlagPrototyped, spFlags: 0)
!707 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !708, file: !602, line: 132)
!708 = !DISubprogram(name: "rename", scope: !605, file: !605, line: 154, type: !709, flags: DIFlagPrototyped, spFlags: 0)
!709 = !DISubroutineType(types: !710)
!710 = !{!9, !162, !162}
!711 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !712, file: !602, line: 133)
!712 = !DISubprogram(name: "rewind", scope: !605, file: !605, line: 723, type: !611, flags: DIFlagPrototyped, spFlags: 0)
!713 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !714, file: !602, line: 134)
!714 = !DISubprogram(name: "scanf", scope: !605, file: !605, line: 421, type: !697, flags: DIFlagPrototyped, spFlags: 0)
!715 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !716, file: !602, line: 135)
!716 = !DISubprogram(name: "setbuf", scope: !605, file: !605, line: 328, type: !717, flags: DIFlagPrototyped, spFlags: 0)
!717 = !DISubroutineType(types: !718)
!718 = !{null, !630, !395}
!719 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !720, file: !602, line: 136)
!720 = !DISubprogram(name: "setvbuf", scope: !605, file: !605, line: 332, type: !721, flags: DIFlagPrototyped, spFlags: 0)
!721 = !DISubroutineType(types: !722)
!722 = !{!9, !630, !395, !9, !304}
!723 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !724, file: !602, line: 137)
!724 = !DISubprogram(name: "sprintf", scope: !605, file: !605, line: 358, type: !725, flags: DIFlagPrototyped, spFlags: 0)
!725 = !DISubroutineType(types: !726)
!726 = !{!9, !395, !353, null}
!727 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !728, file: !602, line: 138)
!728 = !DISubprogram(name: "sscanf", scope: !605, file: !605, line: 423, type: !729, flags: DIFlagPrototyped, spFlags: 0)
!729 = !DISubroutineType(types: !730)
!730 = !{!9, !353, !353, null}
!731 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !732, file: !602, line: 139)
!732 = !DISubprogram(name: "tmpfile", scope: !605, file: !605, line: 188, type: !733, flags: DIFlagPrototyped, spFlags: 0)
!733 = !DISubroutineType(types: !734)
!734 = !{!613}
!735 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !736, file: !602, line: 141)
!736 = !DISubprogram(name: "tmpnam", scope: !605, file: !605, line: 205, type: !689, flags: DIFlagPrototyped, spFlags: 0)
!737 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !738, file: !602, line: 143)
!738 = !DISubprogram(name: "ungetc", scope: !605, file: !605, line: 668, type: !647, flags: DIFlagPrototyped, spFlags: 0)
!739 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !740, file: !602, line: 144)
!740 = !DISubprogram(name: "vfprintf", scope: !605, file: !605, line: 365, type: !741, flags: DIFlagPrototyped, spFlags: 0)
!741 = !DISubroutineType(types: !742)
!742 = !{!9, !630, !353, !743}
!743 = !DIDerivedType(tag: DW_TAG_typedef, name: "__gnuc_va_list", file: !744, line: 32, baseType: !745)
!744 = !DIFile(filename: "llvm-install-tulip/lib/clang/9.0.0/include/stdarg.h", directory: "/scratch/ah7226")
!745 = !DIDerivedType(tag: DW_TAG_typedef, name: "__builtin_va_list", file: !6, baseType: !331)
!746 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !747, file: !602, line: 145)
!747 = !DISubprogram(name: "vprintf", scope: !605, file: !605, line: 371, type: !748, flags: DIFlagPrototyped, spFlags: 0)
!748 = !DISubroutineType(types: !749)
!749 = !{!9, !353, !743}
!750 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !751, file: !602, line: 146)
!751 = !DISubprogram(name: "vsprintf", scope: !605, file: !605, line: 373, type: !752, flags: DIFlagPrototyped, spFlags: 0)
!752 = !DISubroutineType(types: !753)
!753 = !{!9, !395, !353, !743}
!754 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !404, entity: !755, file: !602, line: 175)
!755 = !DISubprogram(name: "snprintf", scope: !605, file: !605, line: 378, type: !756, flags: DIFlagPrototyped, spFlags: 0)
!756 = !DISubroutineType(types: !757)
!757 = !{!9, !395, !304, !353, null}
!758 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !404, entity: !759, file: !602, line: 176)
!759 = !DISubprogram(name: "vfscanf", scope: !605, file: !605, line: 459, type: !741, flags: DIFlagPrototyped, spFlags: 0)
!760 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !404, entity: !761, file: !602, line: 177)
!761 = !DISubprogram(name: "vscanf", scope: !605, file: !605, line: 467, type: !748, flags: DIFlagPrototyped, spFlags: 0)
!762 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !404, entity: !763, file: !602, line: 178)
!763 = !DISubprogram(name: "vsnprintf", scope: !605, file: !605, line: 382, type: !764, flags: DIFlagPrototyped, spFlags: 0)
!764 = !DISubroutineType(types: !765)
!765 = !{!9, !395, !304, !353, !743}
!766 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !404, entity: !767, file: !602, line: 179)
!767 = !DISubprogram(name: "vsscanf", scope: !605, file: !605, line: 471, type: !768, flags: DIFlagPrototyped, spFlags: 0)
!768 = !DISubroutineType(types: !769)
!769 = !{!9, !353, !353, !743}
!770 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !755, file: !602, line: 185)
!771 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !759, file: !602, line: 186)
!772 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !761, file: !602, line: 187)
!773 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !763, file: !602, line: 188)
!774 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !17, entity: !767, file: !602, line: 189)
!775 = !{void (double*, double*, double*, double)* @_Z10gpu_kernelPdS_S_d, !"kernel", i32 1}
!776 = !{null, !"align", i32 8}
!777 = !{null, !"align", i32 8, !"align", i32 65544, !"align", i32 131080}
!778 = !{null, !"align", i32 16}
!779 = !{null, !"align", i32 16, !"align", i32 65552, !"align", i32 131088}
!780 = !{!"clang version 9.0.0 (https://github.com/yebinchon/llvm-project/ a1efa594106d738d0b74c9e4e2b9b779eb8b7d25)"}
!781 = !{i32 1, i32 2}
!782 = distinct !DISubprogram(name: "gpu_kernel", linkageName: "_Z10gpu_kernelPdS_S_d", scope: !6, file: !6, line: 464, type: !783, scopeLine: 467, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !7)
!783 = !DISubroutineType(types: !784)
!784 = !{null, !255, !255, !255, !161}
!785 = !DILocalVariable(name: "f", arg: 1, scope: !786, file: !504, line: 587, type: !161)
!786 = distinct !DISubprogram(name: "fabs", linkageName: "_ZL4fabsd", scope: !504, file: !504, line: 587, type: !214, scopeLine: 588, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !5, retainedNodes: !7)
!787 = !DILocation(line: 587, column: 53, scope: !786, inlinedAt: !788)
!788 = distinct !DILocation(line: 531, column: 7, scope: !789)
!789 = distinct !DILexicalBlock(scope: !790, file: !6, line: 527, column: 15)
!790 = distinct !DILexicalBlock(scope: !791, file: !6, line: 527, column: 7)
!791 = distinct !DILexicalBlock(scope: !792, file: !6, line: 523, column: 33)
!792 = distinct !DILexicalBlock(scope: !793, file: !6, line: 523, column: 3)
!793 = distinct !DILexicalBlock(scope: !794, file: !6, line: 523, column: 3)
!794 = distinct !DILexicalBlock(scope: !795, file: !6, line: 514, column: 39)
!795 = distinct !DILexicalBlock(scope: !796, file: !6, line: 514, column: 2)
!796 = distinct !DILexicalBlock(scope: !782, file: !6, line: 514, column: 2)
!797 = !DILocation(line: 587, column: 53, scope: !786, inlinedAt: !798)
!798 = distinct !DILocation(line: 531, column: 7, scope: !789)
!799 = !DILocation(line: 587, column: 53, scope: !786, inlinedAt: !800)
!800 = distinct !DILocation(line: 531, column: 7, scope: !789)
!801 = !DILocation(line: 587, column: 53, scope: !786, inlinedAt: !802)
!802 = distinct !DILocation(line: 531, column: 7, scope: !789)
!803 = !DILocalVariable(name: "x", arg: 1, scope: !804, file: !504, line: 892, type: !161)
!804 = distinct !DISubprogram(name: "sqrt", linkageName: "_ZL4sqrtd", scope: !504, file: !504, line: 892, type: !214, scopeLine: 893, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !5, retainedNodes: !7)
!805 = !DILocation(line: 892, column: 53, scope: !804, inlinedAt: !806)
!806 = distinct !DILocation(line: 528, column: 8, scope: !789)
!807 = !DILocalVariable(name: "a", arg: 1, scope: !808, file: !809, line: 225, type: !161)
!808 = distinct !DISubprogram(name: "log", linkageName: "_ZL3logd", scope: !809, file: !809, line: 225, type: !214, scopeLine: 226, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !5, retainedNodes: !7)
!809 = !DIFile(filename: "/usr/local/cuda/include/math_functions_dbl_ptx3.hpp", directory: "")
!810 = !DILocation(line: 225, column: 52, scope: !808, inlinedAt: !811)
!811 = distinct !DILocation(line: 528, column: 19, scope: !789)
!812 = !DILocalVariable(name: "q_global", arg: 1, scope: !782, file: !6, line: 464, type: !255)
!813 = !DILocation(line: 464, column: 36, scope: !782)
!814 = !DILocalVariable(name: "sx_global", arg: 2, scope: !782, file: !6, line: 465, type: !255)
!815 = !DILocation(line: 465, column: 11, scope: !782)
!816 = !DILocalVariable(name: "sy_global", arg: 3, scope: !782, file: !6, line: 466, type: !255)
!817 = !DILocation(line: 466, column: 11, scope: !782)
!818 = !DILocalVariable(name: "an", arg: 4, scope: !782, file: !6, line: 467, type: !161)
!819 = !DILocation(line: 467, column: 10, scope: !782)
!820 = !DILocalVariable(name: "x_local", scope: !782, file: !6, line: 468, type: !821)
!821 = !DICompositeType(tag: DW_TAG_array_type, baseType: !161, size: 16384, elements: !822)
!822 = !{!823}
!823 = !DISubrange(count: 256)
!824 = !DILocation(line: 468, column: 9, scope: !782)
!825 = !DILocalVariable(name: "q_local", scope: !782, file: !6, line: 469, type: !826)
!826 = !DICompositeType(tag: DW_TAG_array_type, baseType: !161, size: 640, elements: !827)
!827 = !{!828}
!828 = !DISubrange(count: 10)
!829 = !DILocation(line: 469, column: 9, scope: !782)
!830 = !DILocalVariable(name: "sx_local", scope: !782, file: !6, line: 470, type: !161)
!831 = !DILocation(line: 470, column: 9, scope: !782)
!832 = !DILocalVariable(name: "sy_local", scope: !782, file: !6, line: 470, type: !161)
!833 = !DILocation(line: 470, column: 19, scope: !782)
!834 = !DILocalVariable(name: "t1", scope: !782, file: !6, line: 471, type: !161)
!835 = !DILocation(line: 471, column: 9, scope: !782)
!836 = !DILocalVariable(name: "t2", scope: !782, file: !6, line: 471, type: !161)
!837 = !DILocation(line: 471, column: 13, scope: !782)
!838 = !DILocalVariable(name: "t3", scope: !782, file: !6, line: 471, type: !161)
!839 = !DILocation(line: 471, column: 17, scope: !782)
!840 = !DILocalVariable(name: "t4", scope: !782, file: !6, line: 471, type: !161)
!841 = !DILocation(line: 471, column: 21, scope: !782)
!842 = !DILocalVariable(name: "x1", scope: !782, file: !6, line: 471, type: !161)
!843 = !DILocation(line: 471, column: 25, scope: !782)
!844 = !DILocalVariable(name: "x2", scope: !782, file: !6, line: 471, type: !161)
!845 = !DILocation(line: 471, column: 29, scope: !782)
!846 = !DILocalVariable(name: "seed", scope: !782, file: !6, line: 471, type: !161)
!847 = !DILocation(line: 471, column: 33, scope: !782)
!848 = !DILocalVariable(name: "i", scope: !782, file: !6, line: 472, type: !9)
!849 = !DILocation(line: 472, column: 6, scope: !782)
!850 = !DILocalVariable(name: "ii", scope: !782, file: !6, line: 472, type: !9)
!851 = !DILocation(line: 472, column: 9, scope: !782)
!852 = !DILocalVariable(name: "ik", scope: !782, file: !6, line: 472, type: !9)
!853 = !DILocation(line: 472, column: 13, scope: !782)
!854 = !DILocalVariable(name: "kk", scope: !782, file: !6, line: 472, type: !9)
!855 = !DILocation(line: 472, column: 17, scope: !782)
!856 = !DILocalVariable(name: "l", scope: !782, file: !6, line: 472, type: !9)
!857 = !DILocation(line: 472, column: 21, scope: !782)
!858 = !DILocation(line: 474, column: 2, scope: !782)
!859 = !DILocation(line: 474, column: 12, scope: !782)
!860 = !DILocation(line: 475, column: 2, scope: !782)
!861 = !DILocation(line: 475, column: 12, scope: !782)
!862 = !DILocation(line: 476, column: 2, scope: !782)
!863 = !DILocation(line: 476, column: 12, scope: !782)
!864 = !DILocation(line: 477, column: 2, scope: !782)
!865 = !DILocation(line: 477, column: 12, scope: !782)
!866 = !DILocation(line: 478, column: 2, scope: !782)
!867 = !DILocation(line: 478, column: 12, scope: !782)
!868 = !DILocation(line: 479, column: 2, scope: !782)
!869 = !DILocation(line: 479, column: 12, scope: !782)
!870 = !DILocation(line: 480, column: 2, scope: !782)
!871 = !DILocation(line: 480, column: 12, scope: !782)
!872 = !DILocation(line: 481, column: 2, scope: !782)
!873 = !DILocation(line: 481, column: 12, scope: !782)
!874 = !DILocation(line: 482, column: 2, scope: !782)
!875 = !DILocation(line: 482, column: 12, scope: !782)
!876 = !DILocation(line: 483, column: 2, scope: !782)
!877 = !DILocation(line: 483, column: 12, scope: !782)
!878 = !DILocation(line: 484, column: 10, scope: !782)
!879 = !DILocation(line: 485, column: 10, scope: !782)
!880 = !DILocation(line: 64, column: 3, scope: !881, inlinedAt: !916)
!881 = distinct !DISubprogram(name: "__fetch_builtin_x", linkageName: "_ZN25__cuda_builtin_blockIdx_t17__fetch_builtin_xEv", scope: !883, file: !882, line: 64, type: !886, scopeLine: 64, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !5, declaration: !885, retainedNodes: !7)
!882 = !DIFile(filename: "llvm-install-tulip/lib/clang/9.0.0/include/__clang_cuda_builtin_vars.h", directory: "/scratch/ah7226")
!883 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "__cuda_builtin_blockIdx_t", file: !882, line: 63, size: 8, flags: DIFlagTypePassByReference | DIFlagNonTrivial, elements: !884, identifier: "_ZTS25__cuda_builtin_blockIdx_t")
!884 = !{!885, !888, !889, !890, !901, !905, !909, !912}
!885 = !DISubprogram(name: "__fetch_builtin_x", linkageName: "_ZN25__cuda_builtin_blockIdx_t17__fetch_builtin_xEv", scope: !883, file: !882, line: 64, type: !886, scopeLine: 64, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!886 = !DISubroutineType(types: !887)
!887 = !{!374}
!888 = !DISubprogram(name: "__fetch_builtin_y", linkageName: "_ZN25__cuda_builtin_blockIdx_t17__fetch_builtin_yEv", scope: !883, file: !882, line: 65, type: !886, scopeLine: 65, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!889 = !DISubprogram(name: "__fetch_builtin_z", linkageName: "_ZN25__cuda_builtin_blockIdx_t17__fetch_builtin_zEv", scope: !883, file: !882, line: 66, type: !886, scopeLine: 66, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!890 = !DISubprogram(name: "operator uint3", linkageName: "_ZNK25__cuda_builtin_blockIdx_tcv5uint3Ev", scope: !883, file: !882, line: 69, type: !891, scopeLine: 69, flags: DIFlagPrototyped, spFlags: 0)
!891 = !DISubroutineType(types: !892)
!892 = !{!893, !899}
!893 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "uint3", file: !894, line: 190, size: 96, flags: DIFlagTypePassByValue, elements: !895, identifier: "_ZTS5uint3")
!894 = !DIFile(filename: "/usr/local/cuda/include/vector_types.h", directory: "")
!895 = !{!896, !897, !898}
!896 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !893, file: !894, line: 192, baseType: !374, size: 32)
!897 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !893, file: !894, line: 192, baseType: !374, size: 32, offset: 32)
!898 = !DIDerivedType(tag: DW_TAG_member, name: "z", scope: !893, file: !894, line: 192, baseType: !374, size: 32, offset: 64)
!899 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !900, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!900 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !883)
!901 = !DISubprogram(name: "__cuda_builtin_blockIdx_t", scope: !883, file: !882, line: 71, type: !902, scopeLine: 71, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!902 = !DISubroutineType(types: !903)
!903 = !{null, !904}
!904 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !883, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!905 = !DISubprogram(name: "__cuda_builtin_blockIdx_t", scope: !883, file: !882, line: 71, type: !906, scopeLine: 71, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!906 = !DISubroutineType(types: !907)
!907 = !{null, !904, !908}
!908 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !900, size: 64)
!909 = !DISubprogram(name: "operator=", linkageName: "_ZNK25__cuda_builtin_blockIdx_taSERKS_", scope: !883, file: !882, line: 71, type: !910, scopeLine: 71, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!910 = !DISubroutineType(types: !911)
!911 = !{null, !899, !908}
!912 = !DISubprogram(name: "operator&", linkageName: "_ZNK25__cuda_builtin_blockIdx_tadEv", scope: !883, file: !882, line: 71, type: !913, scopeLine: 71, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!913 = !DISubroutineType(types: !914)
!914 = !{!915, !899}
!915 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !883, size: 64)
!916 = distinct !DILocation(line: 487, column: 5, scope: !782)
!917 = !{i32 0, i32 65535}
!918 = !DILocation(line: 75, column: 3, scope: !919, inlinedAt: !961)
!919 = distinct !DISubprogram(name: "__fetch_builtin_x", linkageName: "_ZN25__cuda_builtin_blockDim_t17__fetch_builtin_xEv", scope: !920, file: !882, line: 75, type: !886, scopeLine: 75, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !5, declaration: !922, retainedNodes: !7)
!920 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "__cuda_builtin_blockDim_t", file: !882, line: 74, size: 8, flags: DIFlagTypePassByReference | DIFlagNonTrivial, elements: !921, identifier: "_ZTS25__cuda_builtin_blockDim_t")
!921 = !{!922, !923, !924, !925, !946, !950, !954, !957}
!922 = !DISubprogram(name: "__fetch_builtin_x", linkageName: "_ZN25__cuda_builtin_blockDim_t17__fetch_builtin_xEv", scope: !920, file: !882, line: 75, type: !886, scopeLine: 75, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!923 = !DISubprogram(name: "__fetch_builtin_y", linkageName: "_ZN25__cuda_builtin_blockDim_t17__fetch_builtin_yEv", scope: !920, file: !882, line: 76, type: !886, scopeLine: 76, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!924 = !DISubprogram(name: "__fetch_builtin_z", linkageName: "_ZN25__cuda_builtin_blockDim_t17__fetch_builtin_zEv", scope: !920, file: !882, line: 77, type: !886, scopeLine: 77, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!925 = !DISubprogram(name: "operator dim3", linkageName: "_ZNK25__cuda_builtin_blockDim_tcv4dim3Ev", scope: !920, file: !882, line: 80, type: !926, scopeLine: 80, flags: DIFlagPrototyped, spFlags: 0)
!926 = !DISubroutineType(types: !927)
!927 = !{!928, !944}
!928 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "dim3", file: !894, line: 417, size: 96, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !929, identifier: "_ZTS4dim3")
!929 = !{!930, !931, !932, !933, !937, !941}
!930 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !928, file: !894, line: 419, baseType: !374, size: 32)
!931 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !928, file: !894, line: 419, baseType: !374, size: 32, offset: 32)
!932 = !DIDerivedType(tag: DW_TAG_member, name: "z", scope: !928, file: !894, line: 419, baseType: !374, size: 32, offset: 64)
!933 = !DISubprogram(name: "dim3", scope: !928, file: !894, line: 421, type: !934, scopeLine: 421, flags: DIFlagPrototyped, spFlags: 0)
!934 = !DISubroutineType(types: !935)
!935 = !{null, !936, !374, !374, !374}
!936 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !928, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!937 = !DISubprogram(name: "dim3", scope: !928, file: !894, line: 422, type: !938, scopeLine: 422, flags: DIFlagPrototyped, spFlags: 0)
!938 = !DISubroutineType(types: !939)
!939 = !{null, !936, !940}
!940 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint3", file: !894, line: 383, baseType: !893)
!941 = !DISubprogram(name: "operator uint3", linkageName: "_ZN4dim3cv5uint3Ev", scope: !928, file: !894, line: 423, type: !942, scopeLine: 423, flags: DIFlagPrototyped, spFlags: 0)
!942 = !DISubroutineType(types: !943)
!943 = !{!940, !936}
!944 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !945, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!945 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !920)
!946 = !DISubprogram(name: "__cuda_builtin_blockDim_t", scope: !920, file: !882, line: 82, type: !947, scopeLine: 82, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!947 = !DISubroutineType(types: !948)
!948 = !{null, !949}
!949 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !920, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!950 = !DISubprogram(name: "__cuda_builtin_blockDim_t", scope: !920, file: !882, line: 82, type: !951, scopeLine: 82, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!951 = !DISubroutineType(types: !952)
!952 = !{null, !949, !953}
!953 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !945, size: 64)
!954 = !DISubprogram(name: "operator=", linkageName: "_ZNK25__cuda_builtin_blockDim_taSERKS_", scope: !920, file: !882, line: 82, type: !955, scopeLine: 82, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!955 = !DISubroutineType(types: !956)
!956 = !{null, !944, !953}
!957 = !DISubprogram(name: "operator&", linkageName: "_ZNK25__cuda_builtin_blockDim_tadEv", scope: !920, file: !882, line: 82, type: !958, scopeLine: 82, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!958 = !DISubroutineType(types: !959)
!959 = !{!960, !944}
!960 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !920, size: 64)
!961 = distinct !DILocation(line: 487, column: 16, scope: !782)
!962 = !{i32 1, i32 1025}
!963 = !DILocation(line: 487, column: 15, scope: !782)
!964 = !DILocation(line: 53, column: 3, scope: !965, inlinedAt: !991)
!965 = distinct !DISubprogram(name: "__fetch_builtin_x", linkageName: "_ZN26__cuda_builtin_threadIdx_t17__fetch_builtin_xEv", scope: !966, file: !882, line: 53, type: !886, scopeLine: 53, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !5, declaration: !968, retainedNodes: !7)
!966 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "__cuda_builtin_threadIdx_t", file: !882, line: 52, size: 8, flags: DIFlagTypePassByReference | DIFlagNonTrivial, elements: !967, identifier: "_ZTS26__cuda_builtin_threadIdx_t")
!967 = !{!968, !969, !970, !971, !976, !980, !984, !987}
!968 = !DISubprogram(name: "__fetch_builtin_x", linkageName: "_ZN26__cuda_builtin_threadIdx_t17__fetch_builtin_xEv", scope: !966, file: !882, line: 53, type: !886, scopeLine: 53, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!969 = !DISubprogram(name: "__fetch_builtin_y", linkageName: "_ZN26__cuda_builtin_threadIdx_t17__fetch_builtin_yEv", scope: !966, file: !882, line: 54, type: !886, scopeLine: 54, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!970 = !DISubprogram(name: "__fetch_builtin_z", linkageName: "_ZN26__cuda_builtin_threadIdx_t17__fetch_builtin_zEv", scope: !966, file: !882, line: 55, type: !886, scopeLine: 55, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!971 = !DISubprogram(name: "operator uint3", linkageName: "_ZNK26__cuda_builtin_threadIdx_tcv5uint3Ev", scope: !966, file: !882, line: 58, type: !972, scopeLine: 58, flags: DIFlagPrototyped, spFlags: 0)
!972 = !DISubroutineType(types: !973)
!973 = !{!893, !974}
!974 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !975, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!975 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !966)
!976 = !DISubprogram(name: "__cuda_builtin_threadIdx_t", scope: !966, file: !882, line: 60, type: !977, scopeLine: 60, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!977 = !DISubroutineType(types: !978)
!978 = !{null, !979}
!979 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !966, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!980 = !DISubprogram(name: "__cuda_builtin_threadIdx_t", scope: !966, file: !882, line: 60, type: !981, scopeLine: 60, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!981 = !DISubroutineType(types: !982)
!982 = !{null, !979, !983}
!983 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !975, size: 64)
!984 = !DISubprogram(name: "operator=", linkageName: "_ZNK26__cuda_builtin_threadIdx_taSERKS_", scope: !966, file: !882, line: 60, type: !985, scopeLine: 60, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!985 = !DISubroutineType(types: !986)
!986 = !{null, !974, !983}
!987 = !DISubprogram(name: "operator&", linkageName: "_ZNK26__cuda_builtin_threadIdx_tadEv", scope: !966, file: !882, line: 60, type: !988, scopeLine: 60, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!988 = !DISubroutineType(types: !989)
!989 = !{!990, !974}
!990 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !966, size: 64)
!991 = distinct !DILocation(line: 487, column: 27, scope: !782)
!992 = !{i32 0, i32 1024}
!993 = !DILocation(line: 487, column: 26, scope: !782)
!994 = !DILocation(line: 487, column: 4, scope: !782)
!995 = !DILocation(line: 489, column: 5, scope: !996)
!996 = distinct !DILexicalBlock(scope: !782, file: !6, line: 489, column: 5)
!997 = !DILocation(line: 489, column: 7, scope: !996)
!998 = !DILocation(line: 489, column: 5, scope: !782)
!999 = !DILocation(line: 489, column: 13, scope: !1000)
!1000 = distinct !DILexicalBlock(scope: !996, file: !6, line: 489, column: 12)
!1001 = !DILocation(line: 491, column: 4, scope: !782)
!1002 = !DILocation(line: 492, column: 5, scope: !782)
!1003 = !DILocation(line: 492, column: 4, scope: !782)
!1004 = !DILocation(line: 495, column: 7, scope: !1005)
!1005 = distinct !DILexicalBlock(scope: !782, file: !6, line: 495, column: 2)
!1006 = !DILocation(line: 495, column: 6, scope: !1005)
!1007 = !DILocation(line: 495, column: 11, scope: !1008)
!1008 = distinct !DILexicalBlock(scope: !1005, file: !6, line: 495, column: 2)
!1009 = !DILocation(line: 495, column: 12, scope: !1008)
!1010 = !DILocation(line: 495, column: 2, scope: !1005)
!1011 = !DILocation(line: 496, column: 6, scope: !1012)
!1012 = distinct !DILexicalBlock(scope: !1008, file: !6, line: 495, column: 23)
!1013 = !DILocation(line: 496, column: 8, scope: !1012)
!1014 = !DILocation(line: 496, column: 5, scope: !1012)
!1015 = !DILocation(line: 497, column: 9, scope: !1016)
!1016 = distinct !DILexicalBlock(scope: !1012, file: !6, line: 497, column: 6)
!1017 = !DILocation(line: 497, column: 8, scope: !1016)
!1018 = !DILocation(line: 497, column: 14, scope: !1016)
!1019 = !DILocation(line: 497, column: 12, scope: !1016)
!1020 = !DILocation(line: 497, column: 6, scope: !1012)
!1021 = !DILocation(line: 497, column: 40, scope: !1022)
!1022 = distinct !DILexicalBlock(scope: !1016, file: !6, line: 497, column: 17)
!1023 = !DILocation(line: 497, column: 21, scope: !1022)
!1024 = !DILocation(line: 497, column: 20, scope: !1022)
!1025 = !DILocation(line: 497, column: 44, scope: !1022)
!1026 = !DILocation(line: 498, column: 6, scope: !1027)
!1027 = distinct !DILexicalBlock(scope: !1012, file: !6, line: 498, column: 6)
!1028 = !DILocation(line: 498, column: 8, scope: !1027)
!1029 = !DILocation(line: 498, column: 6, scope: !1012)
!1030 = !DILocation(line: 498, column: 13, scope: !1031)
!1031 = distinct !DILexicalBlock(scope: !1027, file: !6, line: 498, column: 12)
!1032 = !DILocation(line: 499, column: 25, scope: !1012)
!1033 = !DILocation(line: 499, column: 6, scope: !1012)
!1034 = !DILocation(line: 499, column: 5, scope: !1012)
!1035 = !DILocation(line: 500, column: 6, scope: !1012)
!1036 = !DILocation(line: 500, column: 5, scope: !1012)
!1037 = !DILocation(line: 501, column: 2, scope: !1012)
!1038 = !DILocation(line: 495, column: 20, scope: !1008)
!1039 = !DILocation(line: 495, column: 2, scope: !1008)
!1040 = distinct !{!1040, !1010, !1041}
!1041 = !DILocation(line: 501, column: 2, scope: !1005)
!1042 = !DILocation(line: 513, column: 7, scope: !782)
!1043 = !DILocation(line: 513, column: 6, scope: !782)
!1044 = !DILocation(line: 514, column: 8, scope: !796)
!1045 = !DILocation(line: 514, column: 6, scope: !796)
!1046 = !DILocation(line: 514, column: 12, scope: !795)
!1047 = !DILocation(line: 514, column: 14, scope: !795)
!1048 = !DILocation(line: 514, column: 2, scope: !796)
!1049 = !DILocation(line: 516, column: 44, scope: !794)
!1050 = !DILocation(line: 516, column: 3, scope: !794)
!1051 = !DILocation(line: 523, column: 8, scope: !793)
!1052 = !DILocation(line: 523, column: 7, scope: !793)
!1053 = !DILocation(line: 523, column: 12, scope: !792)
!1054 = !DILocation(line: 523, column: 13, scope: !792)
!1055 = !DILocation(line: 523, column: 3, scope: !793)
!1056 = !DILocation(line: 524, column: 21, scope: !791)
!1057 = !DILocation(line: 524, column: 20, scope: !791)
!1058 = !DILocation(line: 524, column: 11, scope: !791)
!1059 = !DILocation(line: 524, column: 10, scope: !791)
!1060 = !DILocation(line: 524, column: 23, scope: !791)
!1061 = !DILocation(line: 524, column: 6, scope: !791)
!1062 = !DILocation(line: 525, column: 21, scope: !791)
!1063 = !DILocation(line: 525, column: 20, scope: !791)
!1064 = !DILocation(line: 525, column: 22, scope: !791)
!1065 = !DILocation(line: 525, column: 11, scope: !791)
!1066 = !DILocation(line: 525, column: 10, scope: !791)
!1067 = !DILocation(line: 525, column: 25, scope: !791)
!1068 = !DILocation(line: 525, column: 6, scope: !791)
!1069 = !DILocation(line: 526, column: 7, scope: !791)
!1070 = !DILocation(line: 526, column: 10, scope: !791)
!1071 = !DILocation(line: 526, column: 9, scope: !791)
!1072 = !DILocation(line: 526, column: 13, scope: !791)
!1073 = !DILocation(line: 526, column: 16, scope: !791)
!1074 = !DILocation(line: 526, column: 15, scope: !791)
!1075 = !DILocation(line: 526, column: 12, scope: !791)
!1076 = !DILocation(line: 526, column: 6, scope: !791)
!1077 = !DILocation(line: 527, column: 7, scope: !790)
!1078 = !DILocation(line: 527, column: 9, scope: !790)
!1079 = !DILocation(line: 527, column: 7, scope: !791)
!1080 = !DILocation(line: 528, column: 23, scope: !789)
!1081 = !DILocation(line: 227, column: 19, scope: !808, inlinedAt: !811)
!1082 = !DILocation(line: 227, column: 10, scope: !808, inlinedAt: !811)
!1083 = !DILocation(line: 528, column: 17, scope: !789)
!1084 = !DILocation(line: 528, column: 28, scope: !789)
!1085 = !DILocation(line: 528, column: 27, scope: !789)
!1086 = !DILocation(line: 894, column: 20, scope: !804, inlinedAt: !806)
!1087 = !DILocation(line: 894, column: 10, scope: !804, inlinedAt: !806)
!1088 = !DILocation(line: 528, column: 7, scope: !789)
!1089 = !DILocation(line: 529, column: 9, scope: !789)
!1090 = !DILocation(line: 529, column: 12, scope: !789)
!1091 = !DILocation(line: 529, column: 11, scope: !789)
!1092 = !DILocation(line: 529, column: 7, scope: !789)
!1093 = !DILocation(line: 530, column: 9, scope: !789)
!1094 = !DILocation(line: 530, column: 12, scope: !789)
!1095 = !DILocation(line: 530, column: 11, scope: !789)
!1096 = !DILocation(line: 530, column: 7, scope: !789)
!1097 = !DILocation(line: 531, column: 7, scope: !789)
!1098 = !DILocation(line: 589, column: 20, scope: !786, inlinedAt: !802)
!1099 = !DILocation(line: 589, column: 10, scope: !786, inlinedAt: !802)
!1100 = !DILocation(line: 589, column: 20, scope: !786, inlinedAt: !800)
!1101 = !DILocation(line: 589, column: 10, scope: !786, inlinedAt: !800)
!1102 = !DILocation(line: 589, column: 20, scope: !786, inlinedAt: !798)
!1103 = !DILocation(line: 589, column: 10, scope: !786, inlinedAt: !798)
!1104 = !DILocation(line: 589, column: 20, scope: !786, inlinedAt: !788)
!1105 = !DILocation(line: 589, column: 10, scope: !786, inlinedAt: !788)
!1106 = !DILocation(line: 531, column: 6, scope: !789)
!1107 = !DILocation(line: 532, column: 13, scope: !789)
!1108 = !DILocation(line: 532, column: 5, scope: !789)
!1109 = !DILocation(line: 532, column: 15, scope: !789)
!1110 = !DILocation(line: 533, column: 14, scope: !789)
!1111 = !DILocation(line: 533, column: 23, scope: !789)
!1112 = !DILocation(line: 533, column: 22, scope: !789)
!1113 = !DILocation(line: 533, column: 13, scope: !789)
!1114 = !DILocation(line: 534, column: 15, scope: !789)
!1115 = !DILocation(line: 534, column: 13, scope: !789)
!1116 = !DILocation(line: 535, column: 4, scope: !789)
!1117 = !DILocation(line: 536, column: 3, scope: !791)
!1118 = !DILocation(line: 523, column: 30, scope: !792)
!1119 = !DILocation(line: 523, column: 3, scope: !792)
!1120 = distinct !{!1120, !1055, !1121}
!1121 = !DILocation(line: 536, column: 3, scope: !793)
!1122 = !DILocation(line: 537, column: 2, scope: !794)
!1123 = !DILocation(line: 514, column: 22, scope: !795)
!1124 = !DILocation(line: 514, column: 24, scope: !795)
!1125 = !DILocation(line: 514, column: 21, scope: !795)
!1126 = !DILocation(line: 514, column: 2, scope: !795)
!1127 = distinct !{!1127, !1048, !1128}
!1128 = !DILocation(line: 537, column: 2, scope: !796)
!1129 = !DILocation(line: 539, column: 12, scope: !782)
!1130 = !DILocation(line: 64, column: 3, scope: !881, inlinedAt: !1131)
!1131 = distinct !DILocation(line: 539, column: 21, scope: !782)
!1132 = !DILocation(line: 539, column: 31, scope: !782)
!1133 = !DILocation(line: 539, column: 20, scope: !782)
!1134 = !DILocation(line: 539, column: 34, scope: !782)
!1135 = !DILocation(line: 539, column: 38, scope: !782)
!1136 = !DILocation(line: 539, column: 2, scope: !782)
!1137 = !DILocation(line: 540, column: 12, scope: !782)
!1138 = !DILocation(line: 64, column: 3, scope: !881, inlinedAt: !1139)
!1139 = distinct !DILocation(line: 540, column: 21, scope: !782)
!1140 = !DILocation(line: 540, column: 31, scope: !782)
!1141 = !DILocation(line: 540, column: 20, scope: !782)
!1142 = !DILocation(line: 540, column: 34, scope: !782)
!1143 = !DILocation(line: 540, column: 38, scope: !782)
!1144 = !DILocation(line: 540, column: 2, scope: !782)
!1145 = !DILocation(line: 541, column: 12, scope: !782)
!1146 = !DILocation(line: 64, column: 3, scope: !881, inlinedAt: !1147)
!1147 = distinct !DILocation(line: 541, column: 21, scope: !782)
!1148 = !DILocation(line: 541, column: 31, scope: !782)
!1149 = !DILocation(line: 541, column: 20, scope: !782)
!1150 = !DILocation(line: 541, column: 34, scope: !782)
!1151 = !DILocation(line: 541, column: 38, scope: !782)
!1152 = !DILocation(line: 541, column: 2, scope: !782)
!1153 = !DILocation(line: 542, column: 12, scope: !782)
!1154 = !DILocation(line: 64, column: 3, scope: !881, inlinedAt: !1155)
!1155 = distinct !DILocation(line: 542, column: 21, scope: !782)
!1156 = !DILocation(line: 542, column: 31, scope: !782)
!1157 = !DILocation(line: 542, column: 20, scope: !782)
!1158 = !DILocation(line: 542, column: 34, scope: !782)
!1159 = !DILocation(line: 542, column: 38, scope: !782)
!1160 = !DILocation(line: 542, column: 2, scope: !782)
!1161 = !DILocation(line: 543, column: 12, scope: !782)
!1162 = !DILocation(line: 64, column: 3, scope: !881, inlinedAt: !1163)
!1163 = distinct !DILocation(line: 543, column: 21, scope: !782)
!1164 = !DILocation(line: 543, column: 31, scope: !782)
!1165 = !DILocation(line: 543, column: 20, scope: !782)
!1166 = !DILocation(line: 543, column: 34, scope: !782)
!1167 = !DILocation(line: 543, column: 38, scope: !782)
!1168 = !DILocation(line: 543, column: 2, scope: !782)
!1169 = !DILocation(line: 544, column: 12, scope: !782)
!1170 = !DILocation(line: 64, column: 3, scope: !881, inlinedAt: !1171)
!1171 = distinct !DILocation(line: 544, column: 21, scope: !782)
!1172 = !DILocation(line: 544, column: 31, scope: !782)
!1173 = !DILocation(line: 544, column: 20, scope: !782)
!1174 = !DILocation(line: 544, column: 34, scope: !782)
!1175 = !DILocation(line: 544, column: 38, scope: !782)
!1176 = !DILocation(line: 544, column: 2, scope: !782)
!1177 = !DILocation(line: 545, column: 12, scope: !782)
!1178 = !DILocation(line: 64, column: 3, scope: !881, inlinedAt: !1179)
!1179 = distinct !DILocation(line: 545, column: 21, scope: !782)
!1180 = !DILocation(line: 545, column: 31, scope: !782)
!1181 = !DILocation(line: 545, column: 20, scope: !782)
!1182 = !DILocation(line: 545, column: 34, scope: !782)
!1183 = !DILocation(line: 545, column: 38, scope: !782)
!1184 = !DILocation(line: 545, column: 2, scope: !782)
!1185 = !DILocation(line: 546, column: 12, scope: !782)
!1186 = !DILocation(line: 64, column: 3, scope: !881, inlinedAt: !1187)
!1187 = distinct !DILocation(line: 546, column: 21, scope: !782)
!1188 = !DILocation(line: 546, column: 31, scope: !782)
!1189 = !DILocation(line: 546, column: 20, scope: !782)
!1190 = !DILocation(line: 546, column: 34, scope: !782)
!1191 = !DILocation(line: 546, column: 38, scope: !782)
!1192 = !DILocation(line: 546, column: 2, scope: !782)
!1193 = !DILocation(line: 547, column: 12, scope: !782)
!1194 = !DILocation(line: 64, column: 3, scope: !881, inlinedAt: !1195)
!1195 = distinct !DILocation(line: 547, column: 21, scope: !782)
!1196 = !DILocation(line: 547, column: 31, scope: !782)
!1197 = !DILocation(line: 547, column: 20, scope: !782)
!1198 = !DILocation(line: 547, column: 34, scope: !782)
!1199 = !DILocation(line: 547, column: 38, scope: !782)
!1200 = !DILocation(line: 547, column: 2, scope: !782)
!1201 = !DILocation(line: 548, column: 12, scope: !782)
!1202 = !DILocation(line: 64, column: 3, scope: !881, inlinedAt: !1203)
!1203 = distinct !DILocation(line: 548, column: 21, scope: !782)
!1204 = !DILocation(line: 548, column: 31, scope: !782)
!1205 = !DILocation(line: 548, column: 20, scope: !782)
!1206 = !DILocation(line: 548, column: 34, scope: !782)
!1207 = !DILocation(line: 548, column: 38, scope: !782)
!1208 = !DILocation(line: 548, column: 2, scope: !782)
!1209 = !DILocation(line: 549, column: 12, scope: !782)
!1210 = !DILocation(line: 64, column: 3, scope: !881, inlinedAt: !1211)
!1211 = distinct !DILocation(line: 549, column: 22, scope: !782)
!1212 = !DILocation(line: 549, column: 21, scope: !782)
!1213 = !DILocation(line: 549, column: 34, scope: !782)
!1214 = !DILocation(line: 549, column: 2, scope: !782)
!1215 = !DILocation(line: 550, column: 12, scope: !782)
!1216 = !DILocation(line: 64, column: 3, scope: !881, inlinedAt: !1217)
!1217 = distinct !DILocation(line: 550, column: 22, scope: !782)
!1218 = !DILocation(line: 550, column: 21, scope: !782)
!1219 = !DILocation(line: 550, column: 34, scope: !782)
!1220 = !DILocation(line: 550, column: 2, scope: !782)
!1221 = !DILocation(line: 551, column: 1, scope: !782)
!1222 = distinct !DISubprogram(name: "randlc_device", linkageName: "_Z13randlc_devicePdd", scope: !6, file: !6, line: 553, type: !1223, scopeLine: 554, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !7)
!1223 = !DISubroutineType(types: !1224)
!1224 = !{!161, !255, !161}
!1225 = !DILocalVariable(name: "x", arg: 1, scope: !1222, file: !6, line: 553, type: !255)
!1226 = !DILocation(line: 553, column: 41, scope: !1222)
!1227 = !DILocalVariable(name: "a", arg: 2, scope: !1222, file: !6, line: 554, type: !161)
!1228 = !DILocation(line: 554, column: 10, scope: !1222)
!1229 = !DILocalVariable(name: "t1", scope: !1222, file: !6, line: 555, type: !161)
!1230 = !DILocation(line: 555, column: 9, scope: !1222)
!1231 = !DILocalVariable(name: "t2", scope: !1222, file: !6, line: 555, type: !161)
!1232 = !DILocation(line: 555, column: 12, scope: !1222)
!1233 = !DILocalVariable(name: "t3", scope: !1222, file: !6, line: 555, type: !161)
!1234 = !DILocation(line: 555, column: 15, scope: !1222)
!1235 = !DILocalVariable(name: "t4", scope: !1222, file: !6, line: 555, type: !161)
!1236 = !DILocation(line: 555, column: 18, scope: !1222)
!1237 = !DILocalVariable(name: "a1", scope: !1222, file: !6, line: 555, type: !161)
!1238 = !DILocation(line: 555, column: 21, scope: !1222)
!1239 = !DILocalVariable(name: "a2", scope: !1222, file: !6, line: 555, type: !161)
!1240 = !DILocation(line: 555, column: 24, scope: !1222)
!1241 = !DILocalVariable(name: "x1", scope: !1222, file: !6, line: 555, type: !161)
!1242 = !DILocation(line: 555, column: 27, scope: !1222)
!1243 = !DILocalVariable(name: "x2", scope: !1222, file: !6, line: 555, type: !161)
!1244 = !DILocation(line: 555, column: 30, scope: !1222)
!1245 = !DILocalVariable(name: "z", scope: !1222, file: !6, line: 555, type: !161)
!1246 = !DILocation(line: 555, column: 33, scope: !1222)
!1247 = !DILocation(line: 556, column: 13, scope: !1222)
!1248 = !DILocation(line: 556, column: 11, scope: !1222)
!1249 = !DILocation(line: 556, column: 5, scope: !1222)
!1250 = !DILocation(line: 557, column: 12, scope: !1222)
!1251 = !DILocation(line: 557, column: 7, scope: !1222)
!1252 = !DILocation(line: 557, column: 5, scope: !1222)
!1253 = !DILocation(line: 558, column: 7, scope: !1222)
!1254 = !DILocation(line: 558, column: 17, scope: !1222)
!1255 = !DILocation(line: 558, column: 15, scope: !1222)
!1256 = !DILocation(line: 558, column: 9, scope: !1222)
!1257 = !DILocation(line: 558, column: 5, scope: !1222)
!1258 = !DILocation(line: 559, column: 15, scope: !1222)
!1259 = !DILocation(line: 559, column: 14, scope: !1222)
!1260 = !DILocation(line: 559, column: 11, scope: !1222)
!1261 = !DILocation(line: 559, column: 5, scope: !1222)
!1262 = !DILocation(line: 560, column: 12, scope: !1222)
!1263 = !DILocation(line: 560, column: 7, scope: !1222)
!1264 = !DILocation(line: 560, column: 5, scope: !1222)
!1265 = !DILocation(line: 561, column: 9, scope: !1222)
!1266 = !DILocation(line: 561, column: 8, scope: !1222)
!1267 = !DILocation(line: 561, column: 20, scope: !1222)
!1268 = !DILocation(line: 561, column: 18, scope: !1222)
!1269 = !DILocation(line: 561, column: 12, scope: !1222)
!1270 = !DILocation(line: 561, column: 5, scope: !1222)
!1271 = !DILocation(line: 562, column: 7, scope: !1222)
!1272 = !DILocation(line: 562, column: 12, scope: !1222)
!1273 = !DILocation(line: 562, column: 10, scope: !1222)
!1274 = !DILocation(line: 562, column: 17, scope: !1222)
!1275 = !DILocation(line: 562, column: 22, scope: !1222)
!1276 = !DILocation(line: 562, column: 20, scope: !1222)
!1277 = !DILocation(line: 562, column: 15, scope: !1222)
!1278 = !DILocation(line: 562, column: 5, scope: !1222)
!1279 = !DILocation(line: 563, column: 19, scope: !1222)
!1280 = !DILocation(line: 563, column: 17, scope: !1222)
!1281 = !DILocation(line: 563, column: 12, scope: !1222)
!1282 = !DILocation(line: 563, column: 7, scope: !1222)
!1283 = !DILocation(line: 563, column: 5, scope: !1222)
!1284 = !DILocation(line: 564, column: 6, scope: !1222)
!1285 = !DILocation(line: 564, column: 17, scope: !1222)
!1286 = !DILocation(line: 564, column: 15, scope: !1222)
!1287 = !DILocation(line: 564, column: 9, scope: !1222)
!1288 = !DILocation(line: 564, column: 4, scope: !1222)
!1289 = !DILocation(line: 565, column: 13, scope: !1222)
!1290 = !DILocation(line: 565, column: 11, scope: !1222)
!1291 = !DILocation(line: 565, column: 17, scope: !1222)
!1292 = !DILocation(line: 565, column: 22, scope: !1222)
!1293 = !DILocation(line: 565, column: 20, scope: !1222)
!1294 = !DILocation(line: 565, column: 15, scope: !1222)
!1295 = !DILocation(line: 565, column: 5, scope: !1222)
!1296 = !DILocation(line: 566, column: 19, scope: !1222)
!1297 = !DILocation(line: 566, column: 17, scope: !1222)
!1298 = !DILocation(line: 566, column: 12, scope: !1222)
!1299 = !DILocation(line: 566, column: 7, scope: !1222)
!1300 = !DILocation(line: 566, column: 5, scope: !1222)
!1301 = !DILocation(line: 567, column: 9, scope: !1222)
!1302 = !DILocation(line: 567, column: 20, scope: !1222)
!1303 = !DILocation(line: 567, column: 18, scope: !1222)
!1304 = !DILocation(line: 567, column: 12, scope: !1222)
!1305 = !DILocation(line: 567, column: 4, scope: !1222)
!1306 = !DILocation(line: 567, column: 7, scope: !1222)
!1307 = !DILocation(line: 568, column: 18, scope: !1222)
!1308 = !DILocation(line: 568, column: 17, scope: !1222)
!1309 = !DILocation(line: 568, column: 14, scope: !1222)
!1310 = !DILocation(line: 568, column: 2, scope: !1222)
!1311 = distinct !DISubprogram(name: "vranlc_device", linkageName: "_Z13vranlc_deviceiPddS_", scope: !6, file: !6, line: 649, type: !1312, scopeLine: 652, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !7)
!1312 = !DISubroutineType(types: !1313)
!1313 = !{null, !9, !255, !161, !255}
!1314 = !DILocalVariable(name: "n", arg: 1, scope: !1311, file: !6, line: 649, type: !9)
!1315 = !DILocation(line: 649, column: 35, scope: !1311)
!1316 = !DILocalVariable(name: "x_seed", arg: 2, scope: !1311, file: !6, line: 650, type: !255)
!1317 = !DILocation(line: 650, column: 11, scope: !1311)
!1318 = !DILocalVariable(name: "a", arg: 3, scope: !1311, file: !6, line: 651, type: !161)
!1319 = !DILocation(line: 651, column: 10, scope: !1311)
!1320 = !DILocalVariable(name: "y", arg: 4, scope: !1311, file: !6, line: 652, type: !255)
!1321 = !DILocation(line: 652, column: 11, scope: !1311)
!1322 = !DILocalVariable(name: "i", scope: !1311, file: !6, line: 653, type: !9)
!1323 = !DILocation(line: 653, column: 6, scope: !1311)
!1324 = !DILocalVariable(name: "x", scope: !1311, file: !6, line: 654, type: !161)
!1325 = !DILocation(line: 654, column: 9, scope: !1311)
!1326 = !DILocalVariable(name: "t1", scope: !1311, file: !6, line: 654, type: !161)
!1327 = !DILocation(line: 654, column: 11, scope: !1311)
!1328 = !DILocalVariable(name: "t2", scope: !1311, file: !6, line: 654, type: !161)
!1329 = !DILocation(line: 654, column: 14, scope: !1311)
!1330 = !DILocalVariable(name: "t3", scope: !1311, file: !6, line: 654, type: !161)
!1331 = !DILocation(line: 654, column: 17, scope: !1311)
!1332 = !DILocalVariable(name: "t4", scope: !1311, file: !6, line: 654, type: !161)
!1333 = !DILocation(line: 654, column: 20, scope: !1311)
!1334 = !DILocalVariable(name: "a1", scope: !1311, file: !6, line: 654, type: !161)
!1335 = !DILocation(line: 654, column: 23, scope: !1311)
!1336 = !DILocalVariable(name: "a2", scope: !1311, file: !6, line: 654, type: !161)
!1337 = !DILocation(line: 654, column: 26, scope: !1311)
!1338 = !DILocalVariable(name: "x1", scope: !1311, file: !6, line: 654, type: !161)
!1339 = !DILocation(line: 654, column: 29, scope: !1311)
!1340 = !DILocalVariable(name: "x2", scope: !1311, file: !6, line: 654, type: !161)
!1341 = !DILocation(line: 654, column: 32, scope: !1311)
!1342 = !DILocalVariable(name: "z", scope: !1311, file: !6, line: 654, type: !161)
!1343 = !DILocation(line: 654, column: 35, scope: !1311)
!1344 = !DILocation(line: 655, column: 13, scope: !1311)
!1345 = !DILocation(line: 655, column: 11, scope: !1311)
!1346 = !DILocation(line: 655, column: 5, scope: !1311)
!1347 = !DILocation(line: 656, column: 12, scope: !1311)
!1348 = !DILocation(line: 656, column: 7, scope: !1311)
!1349 = !DILocation(line: 656, column: 5, scope: !1311)
!1350 = !DILocation(line: 657, column: 7, scope: !1311)
!1351 = !DILocation(line: 657, column: 17, scope: !1311)
!1352 = !DILocation(line: 657, column: 15, scope: !1311)
!1353 = !DILocation(line: 657, column: 9, scope: !1311)
!1354 = !DILocation(line: 657, column: 5, scope: !1311)
!1355 = !DILocation(line: 658, column: 7, scope: !1311)
!1356 = !DILocation(line: 658, column: 6, scope: !1311)
!1357 = !DILocation(line: 658, column: 4, scope: !1311)
!1358 = !DILocation(line: 659, column: 7, scope: !1359)
!1359 = distinct !DILexicalBlock(scope: !1311, file: !6, line: 659, column: 2)
!1360 = !DILocation(line: 659, column: 6, scope: !1359)
!1361 = !DILocation(line: 659, column: 11, scope: !1362)
!1362 = distinct !DILexicalBlock(scope: !1359, file: !6, line: 659, column: 2)
!1363 = !DILocation(line: 659, column: 13, scope: !1362)
!1364 = !DILocation(line: 659, column: 12, scope: !1362)
!1365 = !DILocation(line: 659, column: 2, scope: !1359)
!1366 = !DILocation(line: 660, column: 14, scope: !1367)
!1367 = distinct !DILexicalBlock(scope: !1362, file: !6, line: 659, column: 20)
!1368 = !DILocation(line: 660, column: 12, scope: !1367)
!1369 = !DILocation(line: 660, column: 6, scope: !1367)
!1370 = !DILocation(line: 661, column: 13, scope: !1367)
!1371 = !DILocation(line: 661, column: 8, scope: !1367)
!1372 = !DILocation(line: 661, column: 6, scope: !1367)
!1373 = !DILocation(line: 662, column: 8, scope: !1367)
!1374 = !DILocation(line: 662, column: 18, scope: !1367)
!1375 = !DILocation(line: 662, column: 16, scope: !1367)
!1376 = !DILocation(line: 662, column: 10, scope: !1367)
!1377 = !DILocation(line: 662, column: 6, scope: !1367)
!1378 = !DILocation(line: 663, column: 8, scope: !1367)
!1379 = !DILocation(line: 663, column: 13, scope: !1367)
!1380 = !DILocation(line: 663, column: 11, scope: !1367)
!1381 = !DILocation(line: 663, column: 18, scope: !1367)
!1382 = !DILocation(line: 663, column: 23, scope: !1367)
!1383 = !DILocation(line: 663, column: 21, scope: !1367)
!1384 = !DILocation(line: 663, column: 16, scope: !1367)
!1385 = !DILocation(line: 663, column: 6, scope: !1367)
!1386 = !DILocation(line: 664, column: 20, scope: !1367)
!1387 = !DILocation(line: 664, column: 18, scope: !1367)
!1388 = !DILocation(line: 664, column: 13, scope: !1367)
!1389 = !DILocation(line: 664, column: 8, scope: !1367)
!1390 = !DILocation(line: 664, column: 6, scope: !1367)
!1391 = !DILocation(line: 665, column: 7, scope: !1367)
!1392 = !DILocation(line: 665, column: 18, scope: !1367)
!1393 = !DILocation(line: 665, column: 16, scope: !1367)
!1394 = !DILocation(line: 665, column: 10, scope: !1367)
!1395 = !DILocation(line: 665, column: 5, scope: !1367)
!1396 = !DILocation(line: 666, column: 14, scope: !1367)
!1397 = !DILocation(line: 666, column: 12, scope: !1367)
!1398 = !DILocation(line: 666, column: 18, scope: !1367)
!1399 = !DILocation(line: 666, column: 23, scope: !1367)
!1400 = !DILocation(line: 666, column: 21, scope: !1367)
!1401 = !DILocation(line: 666, column: 16, scope: !1367)
!1402 = !DILocation(line: 666, column: 6, scope: !1367)
!1403 = !DILocation(line: 667, column: 20, scope: !1367)
!1404 = !DILocation(line: 667, column: 18, scope: !1367)
!1405 = !DILocation(line: 667, column: 13, scope: !1367)
!1406 = !DILocation(line: 667, column: 8, scope: !1367)
!1407 = !DILocation(line: 667, column: 6, scope: !1367)
!1408 = !DILocation(line: 668, column: 7, scope: !1367)
!1409 = !DILocation(line: 668, column: 18, scope: !1367)
!1410 = !DILocation(line: 668, column: 16, scope: !1367)
!1411 = !DILocation(line: 668, column: 10, scope: !1367)
!1412 = !DILocation(line: 668, column: 5, scope: !1367)
!1413 = !DILocation(line: 669, column: 16, scope: !1367)
!1414 = !DILocation(line: 669, column: 14, scope: !1367)
!1415 = !DILocation(line: 669, column: 3, scope: !1367)
!1416 = !DILocation(line: 669, column: 5, scope: !1367)
!1417 = !DILocation(line: 669, column: 8, scope: !1367)
!1418 = !DILocation(line: 670, column: 2, scope: !1367)
!1419 = !DILocation(line: 659, column: 17, scope: !1362)
!1420 = !DILocation(line: 659, column: 2, scope: !1362)
!1421 = distinct !{!1421, !1365, !1422}
!1422 = !DILocation(line: 670, column: 2, scope: !1359)
!1423 = !DILocation(line: 671, column: 12, scope: !1311)
!1424 = !DILocation(line: 671, column: 3, scope: !1311)
!1425 = !DILocation(line: 671, column: 10, scope: !1311)
!1426 = !DILocation(line: 672, column: 1, scope: !1311)
!1427 = distinct !DISubprogram(name: "atomicAdd", linkageName: "_ZL9atomicAddPdd", scope: !1428, file: !1428, line: 54, type: !1223, scopeLine: 54, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !5, retainedNodes: !7)
!1428 = !DIFile(filename: "./../common/npb-CPP.hpp", directory: "/scratch/ah7226/NPB-GPU/CUDA/EP")
!1429 = !DILocalVariable(name: "x", arg: 1, scope: !1430, file: !504, line: 1370, type: !161)
!1430 = distinct !DISubprogram(name: "__double_as_longlong", linkageName: "_ZL20__double_as_longlongd", scope: !504, file: !504, line: 1370, type: !1431, scopeLine: 1371, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !5, retainedNodes: !7)
!1431 = !DISubroutineType(types: !1432)
!1432 = !{!14, !161}
!1433 = !DILocation(line: 1370, column: 74, scope: !1430, inlinedAt: !1434)
!1434 = distinct !DILocation(line: 61, column: 44, scope: !1435)
!1435 = distinct !DILexicalBlock(scope: !1436, file: !1428, line: 59, column: 35)
!1436 = distinct !DILexicalBlock(scope: !1437, file: !1428, line: 59, column: 2)
!1437 = distinct !DILexicalBlock(scope: !1427, file: !1428, line: 59, column: 2)
!1438 = !DILocalVariable(name: "x", arg: 1, scope: !1439, file: !504, line: 1365, type: !14)
!1439 = distinct !DISubprogram(name: "__longlong_as_double", linkageName: "_ZL20__longlong_as_doublex", scope: !504, file: !504, line: 1365, type: !1440, scopeLine: 1366, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !5, retainedNodes: !7)
!1440 = !DISubroutineType(types: !1441)
!1441 = !{!161, !14}
!1442 = !DILocation(line: 1365, column: 72, scope: !1439, inlinedAt: !1443)
!1443 = distinct !DILocation(line: 61, column: 70, scope: !1435)
!1444 = !DILocation(line: 1365, column: 72, scope: !1439, inlinedAt: !1445)
!1445 = distinct !DILocation(line: 62, column: 36, scope: !1446)
!1446 = distinct !DILexicalBlock(scope: !1435, file: !1428, line: 62, column: 13)
!1447 = !DILocation(line: 1365, column: 72, scope: !1439, inlinedAt: !1448)
!1448 = distinct !DILocation(line: 64, column: 9, scope: !1427)
!1449 = !DILocation(line: 1365, column: 72, scope: !1439, inlinedAt: !1450)
!1450 = distinct !DILocation(line: 58, column: 10, scope: !1451)
!1451 = distinct !DILexicalBlock(scope: !1427, file: !1428, line: 57, column: 6)
!1452 = !DILocalVariable(name: "address", arg: 1, scope: !1427, file: !1428, line: 54, type: !255)
!1453 = !DILocation(line: 54, column: 55, scope: !1427)
!1454 = !DILocalVariable(name: "val", arg: 2, scope: !1427, file: !1428, line: 54, type: !161)
!1455 = !DILocation(line: 54, column: 71, scope: !1427)
!1456 = !DILocalVariable(name: "address_as_ull", scope: !1427, file: !1428, line: 55, type: !10)
!1457 = !DILocation(line: 55, column: 26, scope: !1427)
!1458 = !DILocation(line: 55, column: 68, scope: !1427)
!1459 = !DILocation(line: 55, column: 43, scope: !1427)
!1460 = !DILocalVariable(name: "old", scope: !1427, file: !1428, line: 56, type: !11)
!1461 = !DILocation(line: 56, column: 25, scope: !1427)
!1462 = !DILocation(line: 56, column: 32, scope: !1427)
!1463 = !DILocation(line: 56, column: 31, scope: !1427)
!1464 = !DILocalVariable(name: "assumed", scope: !1427, file: !1428, line: 56, type: !11)
!1465 = !DILocation(line: 56, column: 48, scope: !1427)
!1466 = !DILocation(line: 57, column: 6, scope: !1451)
!1467 = !DILocation(line: 57, column: 9, scope: !1451)
!1468 = !DILocation(line: 57, column: 6, scope: !1427)
!1469 = !DILocation(line: 58, column: 31, scope: !1451)
!1470 = !DILocation(line: 1367, column: 34, scope: !1439, inlinedAt: !1450)
!1471 = !DILocation(line: 1367, column: 10, scope: !1439, inlinedAt: !1450)
!1472 = !DILocation(line: 58, column: 3, scope: !1451)
!1473 = !DILocalVariable(name: "i", scope: !1437, file: !1428, line: 59, type: !9)
!1474 = !DILocation(line: 59, column: 11, scope: !1437)
!1475 = !DILocation(line: 59, column: 7, scope: !1437)
!1476 = !DILocation(line: 59, column: 18, scope: !1436)
!1477 = !DILocation(line: 59, column: 20, scope: !1436)
!1478 = !DILocation(line: 59, column: 2, scope: !1437)
!1479 = !DILocation(line: 60, column: 13, scope: !1435)
!1480 = !DILocation(line: 60, column: 11, scope: !1435)
!1481 = !DILocation(line: 61, column: 19, scope: !1435)
!1482 = !DILocation(line: 61, column: 35, scope: !1435)
!1483 = !DILocation(line: 61, column: 65, scope: !1435)
!1484 = !DILocation(line: 61, column: 91, scope: !1435)
!1485 = !DILocation(line: 1367, column: 34, scope: !1439, inlinedAt: !1443)
!1486 = !DILocation(line: 1367, column: 10, scope: !1439, inlinedAt: !1443)
!1487 = !DILocation(line: 61, column: 69, scope: !1435)
!1488 = !DILocation(line: 1372, column: 34, scope: !1430, inlinedAt: !1434)
!1489 = !DILocation(line: 1372, column: 10, scope: !1430, inlinedAt: !1434)
!1490 = !DILocation(line: 61, column: 9, scope: !1435)
!1491 = !DILocation(line: 61, column: 7, scope: !1435)
!1492 = !DILocation(line: 62, column: 13, scope: !1446)
!1493 = !DILocation(line: 62, column: 24, scope: !1446)
!1494 = !DILocation(line: 62, column: 21, scope: !1446)
!1495 = !DILocation(line: 62, column: 13, scope: !1435)
!1496 = !DILocation(line: 62, column: 57, scope: !1446)
!1497 = !DILocation(line: 1367, column: 34, scope: !1439, inlinedAt: !1445)
!1498 = !DILocation(line: 1367, column: 10, scope: !1439, inlinedAt: !1445)
!1499 = !DILocation(line: 62, column: 29, scope: !1446)
!1500 = !DILocation(line: 63, column: 2, scope: !1435)
!1501 = !DILocation(line: 59, column: 31, scope: !1436)
!1502 = !DILocation(line: 59, column: 2, scope: !1436)
!1503 = distinct !{!1503, !1478, !1504}
!1504 = !DILocation(line: 63, column: 2, scope: !1437)
!1505 = !DILocation(line: 64, column: 30, scope: !1427)
!1506 = !DILocation(line: 1367, column: 34, scope: !1439, inlinedAt: !1448)
!1507 = !DILocation(line: 1367, column: 10, scope: !1439, inlinedAt: !1448)
!1508 = !DILocation(line: 64, column: 2, scope: !1427)
!1509 = !DILocation(line: 65, column: 1, scope: !1427)
!1510 = distinct !DISubprogram(name: "atomicCAS", linkageName: "_ZL9atomicCASPyyy", scope: !1511, file: !1511, line: 211, type: !1512, scopeLine: 212, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !5, retainedNodes: !7)
!1511 = !DIFile(filename: "/usr/local/cuda/include/device_atomic_functions.hpp", directory: "")
!1512 = !DISubroutineType(types: !1513)
!1513 = !{!11, !10, !11, !11}
!1514 = !DILocalVariable(name: "p", arg: 1, scope: !1515, file: !504, line: 1655, type: !10)
!1515 = distinct !DISubprogram(name: "__ullAtomicCAS", linkageName: "_ZL14__ullAtomicCASPyyy", scope: !504, file: !504, line: 1655, type: !1512, scopeLine: 1658, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !5, retainedNodes: !7)
!1516 = !DILocation(line: 1655, column: 63, scope: !1515, inlinedAt: !1517)
!1517 = distinct !DILocation(line: 213, column: 10, scope: !1510)
!1518 = !DILocalVariable(name: "compare", arg: 2, scope: !1515, file: !504, line: 1656, type: !11)
!1519 = !DILocation(line: 1656, column: 62, scope: !1515, inlinedAt: !1517)
!1520 = !DILocalVariable(name: "val", arg: 3, scope: !1515, file: !504, line: 1657, type: !11)
!1521 = !DILocation(line: 1657, column: 62, scope: !1515, inlinedAt: !1517)
!1522 = !DILocalVariable(name: "address", arg: 1, scope: !1510, file: !1511, line: 211, type: !10)
!1523 = !DILocation(line: 211, column: 91, scope: !1510)
!1524 = !DILocalVariable(name: "compare", arg: 2, scope: !1510, file: !1511, line: 211, type: !11)
!1525 = !DILocation(line: 211, column: 123, scope: !1510)
!1526 = !DILocalVariable(name: "val", arg: 3, scope: !1510, file: !1511, line: 211, type: !11)
!1527 = !DILocation(line: 211, column: 155, scope: !1510)
!1528 = !DILocation(line: 213, column: 25, scope: !1510)
!1529 = !DILocation(line: 213, column: 34, scope: !1510)
!1530 = !DILocation(line: 213, column: 43, scope: !1510)
!1531 = !DILocation(line: 1660, column: 78, scope: !1515, inlinedAt: !1517)
!1532 = !DILocation(line: 1661, column: 67, scope: !1515, inlinedAt: !1517)
!1533 = !DILocation(line: 1662, column: 67, scope: !1515, inlinedAt: !1517)
!1534 = !DILocation(line: 1660, column: 29, scope: !1515, inlinedAt: !1517)
!1535 = !DILocation(line: 213, column: 3, scope: !1510)
