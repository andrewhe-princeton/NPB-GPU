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
define dso_local void @_Z10gpu_kernelPdS_S_d(double* %q_global, double* %sx_global, double* %sy_global, double %an) #0 !dbg !778 {
entry:
  %a.addr.i = alloca double, align 8
  call void @llvm.dbg.declare(metadata double* %a.addr.i, metadata !781, metadata !DIExpression()), !dbg !784
  %x.addr.i = alloca double, align 8
  call void @llvm.dbg.declare(metadata double* %x.addr.i, metadata !794, metadata !DIExpression()), !dbg !796
  %f.addr.i67 = alloca double, align 8
  call void @llvm.dbg.declare(metadata double* %f.addr.i67, metadata !798, metadata !DIExpression()), !dbg !800
  %f.addr.i66 = alloca double, align 8
  call void @llvm.dbg.declare(metadata double* %f.addr.i66, metadata !798, metadata !DIExpression()), !dbg !802
  %f.addr.i65 = alloca double, align 8
  call void @llvm.dbg.declare(metadata double* %f.addr.i65, metadata !798, metadata !DIExpression()), !dbg !804
  %f.addr.i = alloca double, align 8
  call void @llvm.dbg.declare(metadata double* %f.addr.i, metadata !798, metadata !DIExpression()), !dbg !806
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
  call void @llvm.dbg.declare(metadata double** %q_global.addr, metadata !808, metadata !DIExpression()), !dbg !809
  store double* %sx_global, double** %sx_global.addr, align 8
  call void @llvm.dbg.declare(metadata double** %sx_global.addr, metadata !810, metadata !DIExpression()), !dbg !811
  store double* %sy_global, double** %sy_global.addr, align 8
  call void @llvm.dbg.declare(metadata double** %sy_global.addr, metadata !812, metadata !DIExpression()), !dbg !813
  store double %an, double* %an.addr, align 8
  call void @llvm.dbg.declare(metadata double* %an.addr, metadata !814, metadata !DIExpression()), !dbg !815
  call void @llvm.dbg.declare(metadata [256 x double]* %x_local, metadata !816, metadata !DIExpression()), !dbg !820
  call void @llvm.dbg.declare(metadata [10 x double]* %q_local, metadata !821, metadata !DIExpression()), !dbg !825
  call void @llvm.dbg.declare(metadata double* %sx_local, metadata !826, metadata !DIExpression()), !dbg !827
  call void @llvm.dbg.declare(metadata double* %sy_local, metadata !828, metadata !DIExpression()), !dbg !829
  call void @llvm.dbg.declare(metadata double* %t1, metadata !830, metadata !DIExpression()), !dbg !831
  call void @llvm.dbg.declare(metadata double* %t2, metadata !832, metadata !DIExpression()), !dbg !833
  call void @llvm.dbg.declare(metadata double* %t3, metadata !834, metadata !DIExpression()), !dbg !835
  call void @llvm.dbg.declare(metadata double* %t4, metadata !836, metadata !DIExpression()), !dbg !837
  call void @llvm.dbg.declare(metadata double* %x1, metadata !838, metadata !DIExpression()), !dbg !839
  call void @llvm.dbg.declare(metadata double* %x2, metadata !840, metadata !DIExpression()), !dbg !841
  call void @llvm.dbg.declare(metadata double* %seed, metadata !842, metadata !DIExpression()), !dbg !843
  call void @llvm.dbg.declare(metadata i32* %i, metadata !844, metadata !DIExpression()), !dbg !845
  call void @llvm.dbg.declare(metadata i32* %ii, metadata !846, metadata !DIExpression()), !dbg !847
  call void @llvm.dbg.declare(metadata i32* %ik, metadata !848, metadata !DIExpression()), !dbg !849
  call void @llvm.dbg.declare(metadata i32* %kk, metadata !850, metadata !DIExpression()), !dbg !851
  call void @llvm.dbg.declare(metadata i32* %l, metadata !852, metadata !DIExpression()), !dbg !853
  %arrayidx = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 0, !dbg !854
  store double 0.000000e+00, double* %arrayidx, align 8, !dbg !855
  %arrayidx1 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 1, !dbg !856
  store double 0.000000e+00, double* %arrayidx1, align 8, !dbg !857
  %arrayidx2 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 2, !dbg !858
  store double 0.000000e+00, double* %arrayidx2, align 8, !dbg !859
  %arrayidx3 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 3, !dbg !860
  store double 0.000000e+00, double* %arrayidx3, align 8, !dbg !861
  %arrayidx4 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 4, !dbg !862
  store double 0.000000e+00, double* %arrayidx4, align 8, !dbg !863
  %arrayidx5 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 5, !dbg !864
  store double 0.000000e+00, double* %arrayidx5, align 8, !dbg !865
  %arrayidx6 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 6, !dbg !866
  store double 0.000000e+00, double* %arrayidx6, align 8, !dbg !867
  %arrayidx7 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 7, !dbg !868
  store double 0.000000e+00, double* %arrayidx7, align 8, !dbg !869
  %arrayidx8 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 8, !dbg !870
  store double 0.000000e+00, double* %arrayidx8, align 8, !dbg !871
  %arrayidx9 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 9, !dbg !872
  store double 0.000000e+00, double* %arrayidx9, align 8, !dbg !873
  store double 0.000000e+00, double* %sx_local, align 8, !dbg !874
  store double 0.000000e+00, double* %sy_local, align 8, !dbg !875
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #5, !dbg !876, !range !913
  %1 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #5, !dbg !914, !range !958
  %mul = mul i32 %0, %1, !dbg !959
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #5, !dbg !960, !range !988
  %add = add i32 %mul, %2, !dbg !989
  store i32 %add, i32* %kk, align 4, !dbg !990
  %3 = load i32, i32* %kk, align 4, !dbg !991
  %cmp = icmp sge i32 %3, 4096, !dbg !993
  br i1 %cmp, label %if.then, label %if.end, !dbg !994

if.then:                                          ; preds = %entry
  br label %for.end64, !dbg !995

if.end:                                           ; preds = %entry
  store double 0x41B033C4D7000000, double* %t1, align 8, !dbg !997
  %4 = load double, double* %an.addr, align 8, !dbg !998
  store double %4, double* %t2, align 8, !dbg !999
  store i32 1, i32* %i, align 4, !dbg !1000
  br label %for.cond, !dbg !1002

for.cond:                                         ; preds = %for.inc, %if.end
  %5 = load i32, i32* %i, align 4, !dbg !1003
  %cmp12 = icmp sle i32 %5, 100, !dbg !1005
  br i1 %cmp12, label %for.body, label %for.end, !dbg !1006

for.body:                                         ; preds = %for.cond
  %6 = load i32, i32* %kk, align 4, !dbg !1007
  %div = sdiv i32 %6, 2, !dbg !1009
  store i32 %div, i32* %ik, align 4, !dbg !1010
  %7 = load i32, i32* %ik, align 4, !dbg !1011
  %mul13 = mul nsw i32 2, %7, !dbg !1013
  %8 = load i32, i32* %kk, align 4, !dbg !1014
  %cmp14 = icmp ne i32 %mul13, %8, !dbg !1015
  br i1 %cmp14, label %if.then15, label %if.end17, !dbg !1016

if.then15:                                        ; preds = %for.body
  %9 = load double, double* %t2, align 8, !dbg !1017
  %call16 = call double @_Z13randlc_devicePdd(double* %t1, double %9) #6, !dbg !1019
  store double %call16, double* %t3, align 8, !dbg !1020
  br label %if.end17, !dbg !1021

if.end17:                                         ; preds = %if.then15, %for.body
  %10 = load i32, i32* %ik, align 4, !dbg !1022
  %cmp18 = icmp eq i32 %10, 0, !dbg !1024
  br i1 %cmp18, label %if.then19, label %if.end20, !dbg !1025

if.then19:                                        ; preds = %if.end17
  br label %for.end, !dbg !1026

if.end20:                                         ; preds = %if.end17
  %11 = load double, double* %t2, align 8, !dbg !1028
  %call21 = call double @_Z13randlc_devicePdd(double* %t2, double %11) #6, !dbg !1029
  store double %call21, double* %t3, align 8, !dbg !1030
  %12 = load i32, i32* %ik, align 4, !dbg !1031
  store i32 %12, i32* %kk, align 4, !dbg !1032
  br label %for.inc, !dbg !1033

for.inc:                                          ; preds = %if.end20
  %13 = load i32, i32* %i, align 4, !dbg !1034
  %inc = add nsw i32 %13, 1, !dbg !1034
  store i32 %inc, i32* %i, align 4, !dbg !1034
  br label %for.cond, !dbg !1035, !llvm.loop !1036

for.end:                                          ; preds = %if.then19, %for.cond
  %14 = load double, double* %t1, align 8, !dbg !1038
  store double %14, double* %seed, align 8, !dbg !1039
  store i32 0, i32* %ii, align 4, !dbg !1040
  br label %for.cond22, !dbg !1041

for.cond22:                                       ; preds = %for.inc62, %for.end
  %15 = load i32, i32* %ii, align 4, !dbg !1042
  %cmp23 = icmp slt i32 %15, 65536, !dbg !1043
  br i1 %cmp23, label %for.body24, label %for.end64, !dbg !1044

for.body24:                                       ; preds = %for.cond22
  %arraydecay = getelementptr inbounds [256 x double], [256 x double]* %x_local, i64 0, i64 0, !dbg !1045
  call void @_Z13vranlc_deviceiPddS_(i32 256, double* %seed, double 0x41D2309CE5400000, double* %arraydecay) #6, !dbg !1046
  store i32 0, i32* %i, align 4, !dbg !1047
  br label %for.cond25, !dbg !1048

for.cond25:                                       ; preds = %for.inc59, %for.body24
  %16 = load i32, i32* %i, align 4, !dbg !1049
  %cmp26 = icmp slt i32 %16, 128, !dbg !1050
  br i1 %cmp26, label %for.body27, label %for.end61, !dbg !1051

for.body27:                                       ; preds = %for.cond25
  %17 = load i32, i32* %i, align 4, !dbg !1052
  %mul28 = mul nsw i32 2, %17, !dbg !1053
  %idxprom = sext i32 %mul28 to i64, !dbg !1054
  %arrayidx29 = getelementptr inbounds [256 x double], [256 x double]* %x_local, i64 0, i64 %idxprom, !dbg !1054
  %18 = load double, double* %arrayidx29, align 8, !dbg !1054
  %mul30 = fmul contract double 2.000000e+00, %18, !dbg !1055
  %sub = fsub contract double %mul30, 1.000000e+00, !dbg !1056
  store double %sub, double* %x1, align 8, !dbg !1057
  %19 = load i32, i32* %i, align 4, !dbg !1058
  %mul31 = mul nsw i32 2, %19, !dbg !1059
  %add32 = add nsw i32 %mul31, 1, !dbg !1060
  %idxprom33 = sext i32 %add32 to i64, !dbg !1061
  %arrayidx34 = getelementptr inbounds [256 x double], [256 x double]* %x_local, i64 0, i64 %idxprom33, !dbg !1061
  %20 = load double, double* %arrayidx34, align 8, !dbg !1061
  %mul35 = fmul contract double 2.000000e+00, %20, !dbg !1062
  %sub36 = fsub contract double %mul35, 1.000000e+00, !dbg !1063
  store double %sub36, double* %x2, align 8, !dbg !1064
  %21 = load double, double* %x1, align 8, !dbg !1065
  %22 = load double, double* %x1, align 8, !dbg !1066
  %mul37 = fmul contract double %21, %22, !dbg !1067
  %23 = load double, double* %x2, align 8, !dbg !1068
  %24 = load double, double* %x2, align 8, !dbg !1069
  %mul38 = fmul contract double %23, %24, !dbg !1070
  %add39 = fadd contract double %mul37, %mul38, !dbg !1071
  store double %add39, double* %t1, align 8, !dbg !1072
  %25 = load double, double* %t1, align 8, !dbg !1073
  %cmp40 = fcmp ole double %25, 1.000000e+00, !dbg !1074
  br i1 %cmp40, label %if.then41, label %if.end58, !dbg !1075

if.then41:                                        ; preds = %for.body27
  %26 = load double, double* %t1, align 8, !dbg !1076
  store double %26, double* %a.addr.i, align 8
  %27 = load double, double* %a.addr.i, align 8, !dbg !1077
  %28 = call i32 @llvm.nvvm.d2i.hi(double %27) #5, !dbg !1078
  %29 = call i32 @llvm.nvvm.d2i.lo(double %27) #5, !dbg !1078
  %30 = fcmp ogt double %27, 0.000000e+00, !dbg !1078
  br i1 %30, label %31, label %33, !dbg !1078

31:                                               ; preds = %if.then41
  %32 = icmp slt i32 %28, 2146435072, !dbg !1078
  br label %33, !dbg !1078

33:                                               ; preds = %31, %if.then41
  %34 = phi i1 [ false, %if.then41 ], [ %32, %31 ], !dbg !1078
  br i1 %34, label %35, label %90, !dbg !1078

35:                                               ; preds = %33
  %36 = icmp slt i32 %28, 1048576, !dbg !1078
  br i1 %36, label %37, label %41, !dbg !1078

37:                                               ; preds = %35
  %38 = fmul double %27, 0x4350000000000000, !dbg !1078
  %39 = call i32 @llvm.nvvm.d2i.hi(double %38) #5, !dbg !1078
  %40 = call i32 @llvm.nvvm.d2i.lo(double %38) #5, !dbg !1078
  br label %41, !dbg !1078

41:                                               ; preds = %37, %35
  %ihi.0.i.i = phi i32 [ %39, %37 ], [ %28, %35 ], !dbg !1078
  %ilo.0.i.i = phi i32 [ %40, %37 ], [ %29, %35 ], !dbg !1078
  %e.0.i.i = phi i32 [ -1077, %37 ], [ -1023, %35 ], !dbg !1078
  %42 = lshr i32 %ihi.0.i.i, 20, !dbg !1078
  %43 = add i32 %e.0.i.i, %42, !dbg !1078
  %44 = and i32 %ihi.0.i.i, -2146435073, !dbg !1078
  %45 = or i32 %44, 1072693248, !dbg !1078
  %46 = call double @llvm.nvvm.lohi.i2d(i32 %ilo.0.i.i, i32 %45) #5, !dbg !1078
  %47 = icmp sgt i32 %45, 1073127582, !dbg !1078
  br i1 %47, label %48, label %54, !dbg !1078

48:                                               ; preds = %41
  %49 = call i32 @llvm.nvvm.d2i.lo(double %46) #5, !dbg !1078
  %50 = call i32 @llvm.nvvm.d2i.hi(double %46) #5, !dbg !1078
  %51 = add i32 -1048576, %50, !dbg !1078
  %52 = call double @llvm.nvvm.lohi.i2d(i32 %49, i32 %51) #5, !dbg !1078
  %53 = add nsw i32 %43, 1, !dbg !1078
  br label %54, !dbg !1078

54:                                               ; preds = %48, %41
  %m.0.i.i = phi double [ %52, %48 ], [ %46, %41 ], !dbg !1078
  %e.1.i.i = phi i32 [ %53, %48 ], [ %43, %41 ], !dbg !1078
  %55 = fsub double %m.0.i.i, 1.000000e+00, !dbg !1078
  %56 = fadd double %m.0.i.i, 1.000000e+00, !dbg !1078
  %57 = call double asm "rcp.approx.ftz.f64 $0,$1;", "=d,d"(double %56) #5, !dbg !1078
  %58 = fsub double -0.000000e+00, %56, !dbg !1078
  %59 = call double @llvm.nvvm.fma.rn.d(double %58, double %57, double 1.000000e+00) #5, !dbg !1078
  %60 = call double @llvm.nvvm.fma.rn.d(double %59, double %59, double %59) #5, !dbg !1078
  %61 = call double @llvm.nvvm.fma.rn.d(double %60, double %57, double %57) #5, !dbg !1078
  %62 = fmul double %55, %61, !dbg !1078
  %63 = fadd double %62, %62, !dbg !1078
  %64 = fmul double %63, %63, !dbg !1078
  %65 = call double @llvm.nvvm.fma.rn.d(double 0x3EB1380B3AE80F1E, double %64, double 0x3ED0EE258B7A8B04) #5, !dbg !1078
  %66 = call double @llvm.nvvm.fma.rn.d(double %65, double %64, double 0x3EF3B2669F02676F) #5, !dbg !1078
  %67 = call double @llvm.nvvm.fma.rn.d(double %66, double %64, double 0x3F1745CBA9AB0956) #5, !dbg !1078
  %68 = call double @llvm.nvvm.fma.rn.d(double %67, double %64, double 0x3F3C71C72D1B5154) #5, !dbg !1078
  %69 = call double @llvm.nvvm.fma.rn.d(double %68, double %64, double 0x3F624924923BE72D) #5, !dbg !1078
  %70 = call double @llvm.nvvm.fma.rn.d(double %69, double %64, double 0x3F8999999999A3C4) #5, !dbg !1078
  %71 = call double @llvm.nvvm.fma.rn.d(double %70, double %64, double 0x3FB5555555555554) #5, !dbg !1078
  %72 = fsub double %55, %63, !dbg !1078
  %73 = fmul double 2.000000e+00, %72, !dbg !1078
  %74 = fsub double -0.000000e+00, %63, !dbg !1078
  %75 = call double @llvm.nvvm.fma.rn.d(double %74, double %55, double %73) #5, !dbg !1078
  %76 = fmul double %61, %75, !dbg !1078
  %77 = fmul double %71, %64, !dbg !1078
  %78 = call double @llvm.nvvm.fma.rn.d(double %77, double %63, double %76) #5, !dbg !1078
  %79 = xor i32 -2147483648, %e.1.i.i, !dbg !1078
  %80 = call double @llvm.nvvm.lohi.i2d(i32 %79, i32 1127219200) #5, !dbg !1078
  %81 = call double @llvm.nvvm.lohi.i2d(i32 -2147483648, i32 1127219200) #5, !dbg !1078
  %82 = fsub double %80, %81, !dbg !1078
  %83 = call double @llvm.nvvm.fma.rn.d(double %82, double 0x3FE62E42FEFA39EF, double %63) #5, !dbg !1078
  %84 = fsub double -0.000000e+00, %82, !dbg !1078
  %85 = call double @llvm.nvvm.fma.rn.d(double %84, double 0x3FE62E42FEFA39EF, double %83) #5, !dbg !1078
  %86 = fsub double %85, %63, !dbg !1078
  %87 = fsub double %78, %86, !dbg !1078
  %88 = call double @llvm.nvvm.fma.rn.d(double %82, double 0x3C7ABC9E3B39803F, double %87) #5, !dbg !1078
  %89 = fadd double %83, %88, !dbg !1078
  br label %_ZL3logd.exit, !dbg !1078

90:                                               ; preds = %33
  %91 = call double @llvm.nvvm.fabs.d(double %27) #5, !dbg !1078
  %92 = fcmp ole double %91, 0x7FF0000000000000, !dbg !1078
  %93 = xor i1 %92, true, !dbg !1078
  %94 = zext i1 %93 to i32, !dbg !1078
  br i1 %93, label %95, label %97, !dbg !1078

95:                                               ; preds = %90
  %96 = fadd double %27, %27, !dbg !1078
  br label %106, !dbg !1078

97:                                               ; preds = %90
  %98 = fcmp oeq double %27, 0.000000e+00, !dbg !1078
  br i1 %98, label %99, label %100, !dbg !1078

99:                                               ; preds = %97
  br label %105, !dbg !1078

100:                                              ; preds = %97
  %101 = fcmp oeq double %27, 0x7FF0000000000000, !dbg !1078
  br i1 %101, label %102, label %103, !dbg !1078

102:                                              ; preds = %100
  br label %104, !dbg !1078

103:                                              ; preds = %100
  br label %104, !dbg !1078

104:                                              ; preds = %103, %102
  %q.0.i.i = phi double [ %27, %102 ], [ 0xFFF8000000000000, %103 ], !dbg !1078
  br label %105, !dbg !1078

105:                                              ; preds = %104, %99
  %q.1.i.i = phi double [ 0xFFF0000000000000, %99 ], [ %q.0.i.i, %104 ], !dbg !1078
  br label %106, !dbg !1078

106:                                              ; preds = %105, %95
  %q.2.i.i = phi double [ %96, %95 ], [ %q.1.i.i, %105 ], !dbg !1078
  br label %_ZL3logd.exit, !dbg !1078

_ZL3logd.exit:                                    ; preds = %54, %106
  %q.3.i.i = phi double [ %89, %54 ], [ %q.2.i.i, %106 ], !dbg !1078
  %mul43 = fmul contract double -2.000000e+00, %q.3.i.i, !dbg !1079
  %107 = load double, double* %t1, align 8, !dbg !1080
  %div44 = fdiv double %mul43, %107, !dbg !1081
  store double %div44, double* %x.addr.i, align 8
  %108 = load double, double* %x.addr.i, align 8, !dbg !1082
  %109 = call double @llvm.nvvm.sqrt.rn.d(double %108) #5, !dbg !1083
  store double %109, double* %t2, align 8, !dbg !1084
  %110 = load double, double* %x1, align 8, !dbg !1085
  %111 = load double, double* %t2, align 8, !dbg !1086
  %mul46 = fmul contract double %110, %111, !dbg !1087
  store double %mul46, double* %t3, align 8, !dbg !1088
  %112 = load double, double* %x2, align 8, !dbg !1089
  %113 = load double, double* %t2, align 8, !dbg !1090
  %mul47 = fmul contract double %112, %113, !dbg !1091
  store double %mul47, double* %t4, align 8, !dbg !1092
  %114 = load double, double* %t3, align 8, !dbg !1093
  store double %114, double* %f.addr.i67, align 8
  %115 = load double, double* %f.addr.i67, align 8, !dbg !1094
  %116 = call double @llvm.nvvm.fabs.d(double %115) #5, !dbg !1095
  %117 = load double, double* %t4, align 8, !dbg !1093
  store double %117, double* %f.addr.i66, align 8
  %118 = load double, double* %f.addr.i66, align 8, !dbg !1096
  %119 = call double @llvm.nvvm.fabs.d(double %118) #5, !dbg !1097
  %cmp50 = fcmp ogt double %116, %119, !dbg !1093
  br i1 %cmp50, label %cond.true, label %cond.false, !dbg !1093

cond.true:                                        ; preds = %_ZL3logd.exit
  %120 = load double, double* %t3, align 8, !dbg !1093
  store double %120, double* %f.addr.i65, align 8
  %121 = load double, double* %f.addr.i65, align 8, !dbg !1098
  %122 = call double @llvm.nvvm.fabs.d(double %121) #5, !dbg !1099
  br label %cond.end, !dbg !1093

cond.false:                                       ; preds = %_ZL3logd.exit
  %123 = load double, double* %t4, align 8, !dbg !1093
  store double %123, double* %f.addr.i, align 8
  %124 = load double, double* %f.addr.i, align 8, !dbg !1100
  %125 = call double @llvm.nvvm.fabs.d(double %124) #5, !dbg !1101
  br label %cond.end, !dbg !1093

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi double [ %122, %cond.true ], [ %125, %cond.false ], !dbg !1093
  %conv = fptosi double %cond to i32, !dbg !1093
  store i32 %conv, i32* %l, align 4, !dbg !1102
  %126 = load i32, i32* %l, align 4, !dbg !1103
  %idxprom53 = sext i32 %126 to i64, !dbg !1104
  %arrayidx54 = getelementptr inbounds [10 x double], [10 x double]* %q_local, i64 0, i64 %idxprom53, !dbg !1104
  %127 = load double, double* %arrayidx54, align 8, !dbg !1105
  %add55 = fadd contract double %127, 1.000000e+00, !dbg !1105
  store double %add55, double* %arrayidx54, align 8, !dbg !1105
  %128 = load double, double* %sx_local, align 8, !dbg !1106
  %129 = load double, double* %t3, align 8, !dbg !1107
  %add56 = fadd contract double %128, %129, !dbg !1108
  store double %add56, double* %sx_local, align 8, !dbg !1109
  %130 = load double, double* %t4, align 8, !dbg !1110
  %131 = load double, double* %sy_local, align 8, !dbg !1111
  %add57 = fadd contract double %131, %130, !dbg !1111
  store double %add57, double* %sy_local, align 8, !dbg !1111
  br label %if.end58, !dbg !1112

if.end58:                                         ; preds = %cond.end, %for.body27
  br label %for.inc59, !dbg !1113

for.inc59:                                        ; preds = %if.end58
  %132 = load i32, i32* %i, align 4, !dbg !1114
  %inc60 = add nsw i32 %132, 1, !dbg !1114
  store i32 %inc60, i32* %i, align 4, !dbg !1114
  br label %for.cond25, !dbg !1115, !llvm.loop !1116

for.end61:                                        ; preds = %for.cond25
  br label %for.inc62, !dbg !1118

for.inc62:                                        ; preds = %for.end61
  %133 = load i32, i32* %ii, align 4, !dbg !1119
  %add63 = add nsw i32 %133, 128, !dbg !1120
  store i32 %add63, i32* %ii, align 4, !dbg !1121
  br label %for.cond22, !dbg !1122, !llvm.loop !1123

for.end64:                                        ; preds = %if.then, %for.cond22
  ret void, !dbg !1125
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: convergent noinline nounwind
define dso_local double @_Z13randlc_devicePdd(double* %x, double %a) #2 !dbg !1126 {
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
  call void @llvm.dbg.declare(metadata double** %x.addr, metadata !1129, metadata !DIExpression()), !dbg !1130
  store double %a, double* %a.addr, align 8
  call void @llvm.dbg.declare(metadata double* %a.addr, metadata !1131, metadata !DIExpression()), !dbg !1132
  call void @llvm.dbg.declare(metadata double* %t1, metadata !1133, metadata !DIExpression()), !dbg !1134
  call void @llvm.dbg.declare(metadata double* %t2, metadata !1135, metadata !DIExpression()), !dbg !1136
  call void @llvm.dbg.declare(metadata double* %t3, metadata !1137, metadata !DIExpression()), !dbg !1138
  call void @llvm.dbg.declare(metadata double* %t4, metadata !1139, metadata !DIExpression()), !dbg !1140
  call void @llvm.dbg.declare(metadata double* %a1, metadata !1141, metadata !DIExpression()), !dbg !1142
  call void @llvm.dbg.declare(metadata double* %a2, metadata !1143, metadata !DIExpression()), !dbg !1144
  call void @llvm.dbg.declare(metadata double* %x1, metadata !1145, metadata !DIExpression()), !dbg !1146
  call void @llvm.dbg.declare(metadata double* %x2, metadata !1147, metadata !DIExpression()), !dbg !1148
  call void @llvm.dbg.declare(metadata double* %z, metadata !1149, metadata !DIExpression()), !dbg !1150
  %0 = load double, double* %a.addr, align 8, !dbg !1151
  %mul = fmul contract double 0x3E80000000000000, %0, !dbg !1152
  store double %mul, double* %t1, align 8, !dbg !1153
  %1 = load double, double* %t1, align 8, !dbg !1154
  %conv = fptosi double %1 to i32, !dbg !1154
  %conv1 = sitofp i32 %conv to double, !dbg !1155
  store double %conv1, double* %a1, align 8, !dbg !1156
  %2 = load double, double* %a.addr, align 8, !dbg !1157
  %3 = load double, double* %a1, align 8, !dbg !1158
  %mul2 = fmul contract double 0x4160000000000000, %3, !dbg !1159
  %sub = fsub contract double %2, %mul2, !dbg !1160
  store double %sub, double* %a2, align 8, !dbg !1161
  %4 = load double*, double** %x.addr, align 8, !dbg !1162
  %5 = load double, double* %4, align 8, !dbg !1163
  %mul3 = fmul contract double 0x3E80000000000000, %5, !dbg !1164
  store double %mul3, double* %t1, align 8, !dbg !1165
  %6 = load double, double* %t1, align 8, !dbg !1166
  %conv4 = fptosi double %6 to i32, !dbg !1166
  %conv5 = sitofp i32 %conv4 to double, !dbg !1167
  store double %conv5, double* %x1, align 8, !dbg !1168
  %7 = load double*, double** %x.addr, align 8, !dbg !1169
  %8 = load double, double* %7, align 8, !dbg !1170
  %9 = load double, double* %x1, align 8, !dbg !1171
  %mul6 = fmul contract double 0x4160000000000000, %9, !dbg !1172
  %sub7 = fsub contract double %8, %mul6, !dbg !1173
  store double %sub7, double* %x2, align 8, !dbg !1174
  %10 = load double, double* %a1, align 8, !dbg !1175
  %11 = load double, double* %x2, align 8, !dbg !1176
  %mul8 = fmul contract double %10, %11, !dbg !1177
  %12 = load double, double* %a2, align 8, !dbg !1178
  %13 = load double, double* %x1, align 8, !dbg !1179
  %mul9 = fmul contract double %12, %13, !dbg !1180
  %add = fadd contract double %mul8, %mul9, !dbg !1181
  store double %add, double* %t1, align 8, !dbg !1182
  %14 = load double, double* %t1, align 8, !dbg !1183
  %mul10 = fmul contract double 0x3E80000000000000, %14, !dbg !1184
  %conv11 = fptosi double %mul10 to i32, !dbg !1185
  %conv12 = sitofp i32 %conv11 to double, !dbg !1186
  store double %conv12, double* %t2, align 8, !dbg !1187
  %15 = load double, double* %t1, align 8, !dbg !1188
  %16 = load double, double* %t2, align 8, !dbg !1189
  %mul13 = fmul contract double 0x4160000000000000, %16, !dbg !1190
  %sub14 = fsub contract double %15, %mul13, !dbg !1191
  store double %sub14, double* %z, align 8, !dbg !1192
  %17 = load double, double* %z, align 8, !dbg !1193
  %mul15 = fmul contract double 0x4160000000000000, %17, !dbg !1194
  %18 = load double, double* %a2, align 8, !dbg !1195
  %19 = load double, double* %x2, align 8, !dbg !1196
  %mul16 = fmul contract double %18, %19, !dbg !1197
  %add17 = fadd contract double %mul15, %mul16, !dbg !1198
  store double %add17, double* %t3, align 8, !dbg !1199
  %20 = load double, double* %t3, align 8, !dbg !1200
  %mul18 = fmul contract double 0x3D10000000000000, %20, !dbg !1201
  %conv19 = fptosi double %mul18 to i32, !dbg !1202
  %conv20 = sitofp i32 %conv19 to double, !dbg !1203
  store double %conv20, double* %t4, align 8, !dbg !1204
  %21 = load double, double* %t3, align 8, !dbg !1205
  %22 = load double, double* %t4, align 8, !dbg !1206
  %mul21 = fmul contract double 0x42D0000000000000, %22, !dbg !1207
  %sub22 = fsub contract double %21, %mul21, !dbg !1208
  %23 = load double*, double** %x.addr, align 8, !dbg !1209
  store double %sub22, double* %23, align 8, !dbg !1210
  %24 = load double*, double** %x.addr, align 8, !dbg !1211
  %25 = load double, double* %24, align 8, !dbg !1212
  %mul23 = fmul contract double 0x3D10000000000000, %25, !dbg !1213
  ret double %mul23, !dbg !1214
}

; Function Attrs: convergent noinline nounwind
define dso_local void @_Z13vranlc_deviceiPddS_(i32 %n, double* %x_seed, double %a, double* %y) #2 !dbg !1215 {
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
  call void @llvm.dbg.declare(metadata i32* %n.addr, metadata !1218, metadata !DIExpression()), !dbg !1219
  store double* %x_seed, double** %x_seed.addr, align 8
  call void @llvm.dbg.declare(metadata double** %x_seed.addr, metadata !1220, metadata !DIExpression()), !dbg !1221
  store double %a, double* %a.addr, align 8
  call void @llvm.dbg.declare(metadata double* %a.addr, metadata !1222, metadata !DIExpression()), !dbg !1223
  store double* %y, double** %y.addr, align 8
  call void @llvm.dbg.declare(metadata double** %y.addr, metadata !1224, metadata !DIExpression()), !dbg !1225
  call void @llvm.dbg.declare(metadata i32* %i, metadata !1226, metadata !DIExpression()), !dbg !1227
  call void @llvm.dbg.declare(metadata double* %x, metadata !1228, metadata !DIExpression()), !dbg !1229
  call void @llvm.dbg.declare(metadata double* %t1, metadata !1230, metadata !DIExpression()), !dbg !1231
  call void @llvm.dbg.declare(metadata double* %t2, metadata !1232, metadata !DIExpression()), !dbg !1233
  call void @llvm.dbg.declare(metadata double* %t3, metadata !1234, metadata !DIExpression()), !dbg !1235
  call void @llvm.dbg.declare(metadata double* %t4, metadata !1236, metadata !DIExpression()), !dbg !1237
  call void @llvm.dbg.declare(metadata double* %a1, metadata !1238, metadata !DIExpression()), !dbg !1239
  call void @llvm.dbg.declare(metadata double* %a2, metadata !1240, metadata !DIExpression()), !dbg !1241
  call void @llvm.dbg.declare(metadata double* %x1, metadata !1242, metadata !DIExpression()), !dbg !1243
  call void @llvm.dbg.declare(metadata double* %x2, metadata !1244, metadata !DIExpression()), !dbg !1245
  call void @llvm.dbg.declare(metadata double* %z, metadata !1246, metadata !DIExpression()), !dbg !1247
  %0 = load double, double* %a.addr, align 8, !dbg !1248
  %mul = fmul contract double 0x3E80000000000000, %0, !dbg !1249
  store double %mul, double* %t1, align 8, !dbg !1250
  %1 = load double, double* %t1, align 8, !dbg !1251
  %conv = fptosi double %1 to i32, !dbg !1251
  %conv1 = sitofp i32 %conv to double, !dbg !1252
  store double %conv1, double* %a1, align 8, !dbg !1253
  %2 = load double, double* %a.addr, align 8, !dbg !1254
  %3 = load double, double* %a1, align 8, !dbg !1255
  %mul2 = fmul contract double 0x4160000000000000, %3, !dbg !1256
  %sub = fsub contract double %2, %mul2, !dbg !1257
  store double %sub, double* %a2, align 8, !dbg !1258
  %4 = load double*, double** %x_seed.addr, align 8, !dbg !1259
  %5 = load double, double* %4, align 8, !dbg !1260
  store double %5, double* %x, align 8, !dbg !1261
  store i32 0, i32* %i, align 4, !dbg !1262
  br label %for.cond, !dbg !1264

for.cond:                                         ; preds = %for.inc, %entry
  %6 = load i32, i32* %i, align 4, !dbg !1265
  %7 = load i32, i32* %n.addr, align 4, !dbg !1267
  %cmp = icmp slt i32 %6, %7, !dbg !1268
  br i1 %cmp, label %for.body, label %for.end, !dbg !1269

for.body:                                         ; preds = %for.cond
  %8 = load double, double* %x, align 8, !dbg !1270
  %mul3 = fmul contract double 0x3E80000000000000, %8, !dbg !1272
  store double %mul3, double* %t1, align 8, !dbg !1273
  %9 = load double, double* %t1, align 8, !dbg !1274
  %conv4 = fptosi double %9 to i32, !dbg !1274
  %conv5 = sitofp i32 %conv4 to double, !dbg !1275
  store double %conv5, double* %x1, align 8, !dbg !1276
  %10 = load double, double* %x, align 8, !dbg !1277
  %11 = load double, double* %x1, align 8, !dbg !1278
  %mul6 = fmul contract double 0x4160000000000000, %11, !dbg !1279
  %sub7 = fsub contract double %10, %mul6, !dbg !1280
  store double %sub7, double* %x2, align 8, !dbg !1281
  %12 = load double, double* %a1, align 8, !dbg !1282
  %13 = load double, double* %x2, align 8, !dbg !1283
  %mul8 = fmul contract double %12, %13, !dbg !1284
  %14 = load double, double* %a2, align 8, !dbg !1285
  %15 = load double, double* %x1, align 8, !dbg !1286
  %mul9 = fmul contract double %14, %15, !dbg !1287
  %add = fadd contract double %mul8, %mul9, !dbg !1288
  store double %add, double* %t1, align 8, !dbg !1289
  %16 = load double, double* %t1, align 8, !dbg !1290
  %mul10 = fmul contract double 0x3E80000000000000, %16, !dbg !1291
  %conv11 = fptosi double %mul10 to i32, !dbg !1292
  %conv12 = sitofp i32 %conv11 to double, !dbg !1293
  store double %conv12, double* %t2, align 8, !dbg !1294
  %17 = load double, double* %t1, align 8, !dbg !1295
  %18 = load double, double* %t2, align 8, !dbg !1296
  %mul13 = fmul contract double 0x4160000000000000, %18, !dbg !1297
  %sub14 = fsub contract double %17, %mul13, !dbg !1298
  store double %sub14, double* %z, align 8, !dbg !1299
  %19 = load double, double* %z, align 8, !dbg !1300
  %mul15 = fmul contract double 0x4160000000000000, %19, !dbg !1301
  %20 = load double, double* %a2, align 8, !dbg !1302
  %21 = load double, double* %x2, align 8, !dbg !1303
  %mul16 = fmul contract double %20, %21, !dbg !1304
  %add17 = fadd contract double %mul15, %mul16, !dbg !1305
  store double %add17, double* %t3, align 8, !dbg !1306
  %22 = load double, double* %t3, align 8, !dbg !1307
  %mul18 = fmul contract double 0x3D10000000000000, %22, !dbg !1308
  %conv19 = fptosi double %mul18 to i32, !dbg !1309
  %conv20 = sitofp i32 %conv19 to double, !dbg !1310
  store double %conv20, double* %t4, align 8, !dbg !1311
  %23 = load double, double* %t3, align 8, !dbg !1312
  %24 = load double, double* %t4, align 8, !dbg !1313
  %mul21 = fmul contract double 0x42D0000000000000, %24, !dbg !1314
  %sub22 = fsub contract double %23, %mul21, !dbg !1315
  store double %sub22, double* %x, align 8, !dbg !1316
  %25 = load double, double* %x, align 8, !dbg !1317
  %mul23 = fmul contract double 0x3D10000000000000, %25, !dbg !1318
  %26 = load double*, double** %y.addr, align 8, !dbg !1319
  %27 = load i32, i32* %i, align 4, !dbg !1320
  %idxprom = sext i32 %27 to i64, !dbg !1319
  %arrayidx = getelementptr inbounds double, double* %26, i64 %idxprom, !dbg !1319
  store double %mul23, double* %arrayidx, align 8, !dbg !1321
  br label %for.inc, !dbg !1322

for.inc:                                          ; preds = %for.body
  %28 = load i32, i32* %i, align 4, !dbg !1323
  %inc = add nsw i32 %28, 1, !dbg !1323
  store i32 %inc, i32* %i, align 4, !dbg !1323
  br label %for.cond, !dbg !1324, !llvm.loop !1325

for.end:                                          ; preds = %for.cond
  %29 = load double, double* %x, align 8, !dbg !1327
  %30 = load double*, double** %x_seed.addr, align 8, !dbg !1328
  store double %29, double* %30, align 8, !dbg !1329
  ret void, !dbg !1330
}

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #3

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #3

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3

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
!nvvm.annotations = !{!771, !772, !773, !772, !774, !774, !774, !774, !775, !775, !774}
!llvm.ident = !{!776}
!nvvm.internalize.after.link = !{}
!nvvmir.version = !{!777}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 7, i32 0]}
!1 = !{i32 2, !"Dwarf Version", i32 2}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !6, producer: "clang version 9.0.0 (https://github.com/yebinchon/llvm-project/ a1efa594106d738d0b74c9e4e2b9b779eb8b7d25)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !7, retainedTypes: !8, imports: !10, nameTableKind: None)
!6 = !DIFile(filename: "ep.cu", directory: "/scratch/ah7226/NPB-GPU/CUDA/EP")
!7 = !{}
!8 = !{!9}
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !{!11, !17, !22, !24, !26, !28, !30, !34, !36, !38, !40, !42, !44, !46, !48, !50, !52, !54, !56, !58, !60, !62, !66, !68, !70, !72, !76, !81, !83, !85, !90, !94, !96, !98, !100, !102, !104, !106, !108, !110, !115, !119, !121, !126, !130, !132, !134, !136, !138, !140, !144, !146, !148, !153, !161, !165, !167, !169, !171, !173, !177, !179, !181, !185, !187, !189, !191, !193, !195, !197, !199, !201, !203, !207, !213, !215, !217, !221, !223, !225, !227, !229, !231, !233, !235, !239, !243, !245, !247, !252, !254, !256, !258, !260, !262, !264, !268, !274, !278, !283, !285, !289, !293, !307, !311, !315, !319, !323, !328, !330, !334, !338, !342, !350, !354, !358, !361, !365, !370, !376, !380, !384, !386, !394, !398, !405, !407, !409, !413, !417, !421, !426, !430, !435, !436, !437, !438, !440, !441, !442, !443, !444, !445, !446, !448, !449, !450, !451, !452, !456, !457, !458, !459, !460, !461, !462, !463, !464, !465, !466, !467, !468, !469, !470, !471, !472, !473, !474, !475, !476, !477, !478, !479, !480, !484, !486, !488, !490, !492, !494, !496, !498, !501, !503, !505, !507, !509, !511, !513, !515, !517, !519, !521, !523, !525, !527, !529, !531, !533, !535, !537, !539, !541, !543, !545, !547, !549, !551, !553, !555, !557, !559, !561, !563, !565, !567, !569, !571, !573, !575, !577, !579, !581, !583, !585, !587, !589, !591, !593, !599, !605, !610, !614, !616, !618, !620, !622, !629, !633, !637, !641, !645, !649, !654, !658, !660, !664, !670, !674, !679, !681, !683, !687, !691, !695, !697, !699, !701, !703, !707, !709, !711, !715, !719, !723, !727, !731, !733, !735, !742, !746, !750, !754, !756, !758, !762, !766, !767, !768, !769, !770}
!11 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !13, file: !14, line: 223)
!12 = !DINamespace(name: "std", scope: null)
!13 = !DISubprogram(name: "abs", linkageName: "_ZL3absi", scope: !14, file: !14, line: 53, type: !15, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!14 = !DIFile(filename: "llvm-install-tulip/lib/clang/9.0.0/include/__clang_cuda_math_forward_declares.h", directory: "/scratch/ah7226")
!15 = !DISubroutineType(types: !16)
!16 = !{!9, !9}
!17 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !18, file: !14, line: 224)
!18 = !DISubprogram(name: "acos", linkageName: "_ZL4acosf", scope: !14, file: !14, line: 55, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!19 = !DISubroutineType(types: !20)
!20 = !{!21, !21}
!21 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!22 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !23, file: !14, line: 225)
!23 = !DISubprogram(name: "acosh", linkageName: "_ZL5acoshf", scope: !14, file: !14, line: 57, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!24 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !25, file: !14, line: 226)
!25 = !DISubprogram(name: "asin", linkageName: "_ZL4asinf", scope: !14, file: !14, line: 59, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!26 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !27, file: !14, line: 227)
!27 = !DISubprogram(name: "asinh", linkageName: "_ZL5asinhf", scope: !14, file: !14, line: 61, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!28 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !29, file: !14, line: 228)
!29 = !DISubprogram(name: "atan", linkageName: "_ZL4atanf", scope: !14, file: !14, line: 65, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!30 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !31, file: !14, line: 229)
!31 = !DISubprogram(name: "atan2", linkageName: "_ZL5atan2ff", scope: !14, file: !14, line: 63, type: !32, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!32 = !DISubroutineType(types: !33)
!33 = !{!21, !21, !21}
!34 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !35, file: !14, line: 230)
!35 = !DISubprogram(name: "atanh", linkageName: "_ZL5atanhf", scope: !14, file: !14, line: 67, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!36 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !37, file: !14, line: 231)
!37 = !DISubprogram(name: "cbrt", linkageName: "_ZL4cbrtf", scope: !14, file: !14, line: 69, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!38 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !39, file: !14, line: 232)
!39 = !DISubprogram(name: "ceil", linkageName: "_ZL4ceilf", scope: !14, file: !14, line: 71, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!40 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !41, file: !14, line: 233)
!41 = !DISubprogram(name: "copysign", linkageName: "_ZL8copysignff", scope: !14, file: !14, line: 73, type: !32, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!42 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !43, file: !14, line: 234)
!43 = !DISubprogram(name: "cos", linkageName: "_ZL3cosf", scope: !14, file: !14, line: 75, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!44 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !45, file: !14, line: 235)
!45 = !DISubprogram(name: "cosh", linkageName: "_ZL4coshf", scope: !14, file: !14, line: 77, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!46 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !47, file: !14, line: 236)
!47 = !DISubprogram(name: "erf", linkageName: "_ZL3erff", scope: !14, file: !14, line: 81, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!48 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !49, file: !14, line: 237)
!49 = !DISubprogram(name: "erfc", linkageName: "_ZL4erfcf", scope: !14, file: !14, line: 79, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!50 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !51, file: !14, line: 238)
!51 = !DISubprogram(name: "exp", linkageName: "_ZL3expf", scope: !14, file: !14, line: 85, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!52 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !53, file: !14, line: 239)
!53 = !DISubprogram(name: "exp2", linkageName: "_ZL4exp2f", scope: !14, file: !14, line: 83, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!54 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !55, file: !14, line: 240)
!55 = !DISubprogram(name: "expm1", linkageName: "_ZL5expm1f", scope: !14, file: !14, line: 87, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!56 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !57, file: !14, line: 241)
!57 = !DISubprogram(name: "fabs", linkageName: "_ZL4fabsf", scope: !14, file: !14, line: 89, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!58 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !59, file: !14, line: 242)
!59 = !DISubprogram(name: "fdim", linkageName: "_ZL4fdimff", scope: !14, file: !14, line: 91, type: !32, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!60 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !61, file: !14, line: 243)
!61 = !DISubprogram(name: "floor", linkageName: "_ZL5floorf", scope: !14, file: !14, line: 93, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!62 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !63, file: !14, line: 244)
!63 = !DISubprogram(name: "fma", linkageName: "_ZL3fmafff", scope: !14, file: !14, line: 95, type: !64, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!64 = !DISubroutineType(types: !65)
!65 = !{!21, !21, !21, !21}
!66 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !67, file: !14, line: 245)
!67 = !DISubprogram(name: "fmax", linkageName: "_ZL4fmaxff", scope: !14, file: !14, line: 97, type: !32, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!68 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !69, file: !14, line: 246)
!69 = !DISubprogram(name: "fmin", linkageName: "_ZL4fminff", scope: !14, file: !14, line: 99, type: !32, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!70 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !71, file: !14, line: 247)
!71 = !DISubprogram(name: "fmod", linkageName: "_ZL4fmodff", scope: !14, file: !14, line: 101, type: !32, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!72 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !73, file: !14, line: 248)
!73 = !DISubprogram(name: "fpclassify", linkageName: "_ZL10fpclassifyf", scope: !14, file: !14, line: 103, type: !74, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!74 = !DISubroutineType(types: !75)
!75 = !{!9, !21}
!76 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !77, file: !14, line: 249)
!77 = !DISubprogram(name: "frexp", linkageName: "_ZL5frexpfPi", scope: !14, file: !14, line: 105, type: !78, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!78 = !DISubroutineType(types: !79)
!79 = !{!21, !21, !80}
!80 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 64)
!81 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !82, file: !14, line: 250)
!82 = !DISubprogram(name: "hypot", linkageName: "_ZL5hypotff", scope: !14, file: !14, line: 107, type: !32, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!83 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !84, file: !14, line: 251)
!84 = !DISubprogram(name: "ilogb", linkageName: "_ZL5ilogbf", scope: !14, file: !14, line: 109, type: !74, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!85 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !86, file: !14, line: 252)
!86 = !DISubprogram(name: "isfinite", linkageName: "_ZL8isfinitef", scope: !14, file: !14, line: 114, type: !87, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!87 = !DISubroutineType(types: !88)
!88 = !{!89, !21}
!89 = !DIBasicType(name: "bool", size: 8, encoding: DW_ATE_boolean)
!90 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !91, file: !14, line: 253)
!91 = !DISubprogram(name: "isgreater", linkageName: "_ZL9isgreaterff", scope: !14, file: !14, line: 118, type: !92, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!92 = !DISubroutineType(types: !93)
!93 = !{!89, !21, !21}
!94 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !95, file: !14, line: 254)
!95 = !DISubprogram(name: "isgreaterequal", linkageName: "_ZL14isgreaterequalff", scope: !14, file: !14, line: 117, type: !92, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!96 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !97, file: !14, line: 255)
!97 = !DISubprogram(name: "isinf", linkageName: "_ZL5isinff", scope: !14, file: !14, line: 123, type: !87, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!98 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !99, file: !14, line: 256)
!99 = !DISubprogram(name: "isless", linkageName: "_ZL6islessff", scope: !14, file: !14, line: 127, type: !92, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!100 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !101, file: !14, line: 257)
!101 = !DISubprogram(name: "islessequal", linkageName: "_ZL11islessequalff", scope: !14, file: !14, line: 126, type: !92, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!102 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !103, file: !14, line: 258)
!103 = !DISubprogram(name: "islessgreater", linkageName: "_ZL13islessgreaterff", scope: !14, file: !14, line: 129, type: !92, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!104 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !105, file: !14, line: 259)
!105 = !DISubprogram(name: "isnan", linkageName: "_ZL5isnanf", scope: !14, file: !14, line: 134, type: !87, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!106 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !107, file: !14, line: 260)
!107 = !DISubprogram(name: "isnormal", linkageName: "_ZL8isnormalf", scope: !14, file: !14, line: 136, type: !87, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!108 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !109, file: !14, line: 261)
!109 = !DISubprogram(name: "isunordered", linkageName: "_ZL11isunorderedff", scope: !14, file: !14, line: 138, type: !92, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!110 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !111, file: !14, line: 262)
!111 = !DISubprogram(name: "labs", linkageName: "_ZL4labsl", scope: !14, file: !14, line: 139, type: !112, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!112 = !DISubroutineType(types: !113)
!113 = !{!114, !114}
!114 = !DIBasicType(name: "long int", size: 64, encoding: DW_ATE_signed)
!115 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !116, file: !14, line: 263)
!116 = !DISubprogram(name: "ldexp", linkageName: "_ZL5ldexpfi", scope: !14, file: !14, line: 141, type: !117, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!117 = !DISubroutineType(types: !118)
!118 = !{!21, !21, !9}
!119 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !120, file: !14, line: 264)
!120 = !DISubprogram(name: "lgamma", linkageName: "_ZL6lgammaf", scope: !14, file: !14, line: 143, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!121 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !122, file: !14, line: 265)
!122 = !DISubprogram(name: "llabs", linkageName: "_ZL5llabsx", scope: !14, file: !14, line: 144, type: !123, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!123 = !DISubroutineType(types: !124)
!124 = !{!125, !125}
!125 = !DIBasicType(name: "long long int", size: 64, encoding: DW_ATE_signed)
!126 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !127, file: !14, line: 266)
!127 = !DISubprogram(name: "llrint", linkageName: "_ZL6llrintf", scope: !14, file: !14, line: 146, type: !128, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!128 = !DISubroutineType(types: !129)
!129 = !{!125, !21}
!130 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !131, file: !14, line: 267)
!131 = !DISubprogram(name: "log", linkageName: "_ZL3logf", scope: !14, file: !14, line: 159, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!132 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !133, file: !14, line: 268)
!133 = !DISubprogram(name: "log10", linkageName: "_ZL5log10f", scope: !14, file: !14, line: 148, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!134 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !135, file: !14, line: 269)
!135 = !DISubprogram(name: "log1p", linkageName: "_ZL5log1pf", scope: !14, file: !14, line: 150, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!136 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !137, file: !14, line: 270)
!137 = !DISubprogram(name: "log2", linkageName: "_ZL4log2f", scope: !14, file: !14, line: 152, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!138 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !139, file: !14, line: 271)
!139 = !DISubprogram(name: "logb", linkageName: "_ZL4logbf", scope: !14, file: !14, line: 154, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!140 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !141, file: !14, line: 272)
!141 = !DISubprogram(name: "lrint", linkageName: "_ZL5lrintf", scope: !14, file: !14, line: 161, type: !142, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!142 = !DISubroutineType(types: !143)
!143 = !{!114, !21}
!144 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !145, file: !14, line: 273)
!145 = !DISubprogram(name: "lround", linkageName: "_ZL6lroundf", scope: !14, file: !14, line: 163, type: !142, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!146 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !147, file: !14, line: 274)
!147 = !DISubprogram(name: "llround", linkageName: "_ZL7llroundf", scope: !14, file: !14, line: 164, type: !128, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!148 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !149, file: !14, line: 275)
!149 = !DISubprogram(name: "modf", linkageName: "_ZL4modffPf", scope: !14, file: !14, line: 166, type: !150, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!150 = !DISubroutineType(types: !151)
!151 = !{!21, !21, !152}
!152 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !21, size: 64)
!153 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !154, file: !14, line: 276)
!154 = !DISubprogram(name: "nan", linkageName: "_ZL3nanPKc", scope: !14, file: !14, line: 167, type: !155, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!155 = !DISubroutineType(types: !156)
!156 = !{!157, !158}
!157 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
!158 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !159, size: 64)
!159 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !160)
!160 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!161 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !162, file: !14, line: 277)
!162 = !DISubprogram(name: "nanf", linkageName: "_ZL4nanfPKc", scope: !14, file: !14, line: 168, type: !163, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!163 = !DISubroutineType(types: !164)
!164 = !{!21, !158}
!165 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !166, file: !14, line: 278)
!166 = !DISubprogram(name: "nearbyint", linkageName: "_ZL9nearbyintf", scope: !14, file: !14, line: 170, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!167 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !168, file: !14, line: 279)
!168 = !DISubprogram(name: "nextafter", linkageName: "_ZL9nextafterff", scope: !14, file: !14, line: 172, type: !32, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!169 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !170, file: !14, line: 280)
!170 = !DISubprogram(name: "pow", linkageName: "_ZL3powfi", scope: !14, file: !14, line: 176, type: !117, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!171 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !172, file: !14, line: 281)
!172 = !DISubprogram(name: "remainder", linkageName: "_ZL9remainderff", scope: !14, file: !14, line: 178, type: !32, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!173 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !174, file: !14, line: 282)
!174 = !DISubprogram(name: "remquo", linkageName: "_ZL6remquoffPi", scope: !14, file: !14, line: 180, type: !175, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!175 = !DISubroutineType(types: !176)
!176 = !{!21, !21, !21, !80}
!177 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !178, file: !14, line: 283)
!178 = !DISubprogram(name: "rint", linkageName: "_ZL4rintf", scope: !14, file: !14, line: 182, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!179 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !180, file: !14, line: 284)
!180 = !DISubprogram(name: "round", linkageName: "_ZL5roundf", scope: !14, file: !14, line: 184, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!181 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !182, file: !14, line: 285)
!182 = !DISubprogram(name: "scalbln", linkageName: "_ZL7scalblnfl", scope: !14, file: !14, line: 186, type: !183, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!183 = !DISubroutineType(types: !184)
!184 = !{!21, !21, !114}
!185 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !186, file: !14, line: 286)
!186 = !DISubprogram(name: "scalbn", linkageName: "_ZL6scalbnfi", scope: !14, file: !14, line: 188, type: !117, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!187 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !188, file: !14, line: 287)
!188 = !DISubprogram(name: "signbit", linkageName: "_ZL7signbitf", scope: !14, file: !14, line: 190, type: !87, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!189 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !190, file: !14, line: 288)
!190 = !DISubprogram(name: "sin", linkageName: "_ZL3sinf", scope: !14, file: !14, line: 192, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!191 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !192, file: !14, line: 289)
!192 = !DISubprogram(name: "sinh", linkageName: "_ZL4sinhf", scope: !14, file: !14, line: 194, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!193 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !194, file: !14, line: 290)
!194 = !DISubprogram(name: "sqrt", linkageName: "_ZL4sqrtf", scope: !14, file: !14, line: 196, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!195 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !196, file: !14, line: 291)
!196 = !DISubprogram(name: "tan", linkageName: "_ZL3tanf", scope: !14, file: !14, line: 198, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!197 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !198, file: !14, line: 292)
!198 = !DISubprogram(name: "tanh", linkageName: "_ZL4tanhf", scope: !14, file: !14, line: 200, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!199 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !200, file: !14, line: 293)
!200 = !DISubprogram(name: "tgamma", linkageName: "_ZL6tgammaf", scope: !14, file: !14, line: 202, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!201 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !202, file: !14, line: 294)
!202 = !DISubprogram(name: "trunc", linkageName: "_ZL5truncf", scope: !14, file: !14, line: 204, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!203 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !204, file: !206, line: 52)
!204 = !DISubprogram(name: "abs", scope: !205, file: !205, line: 848, type: !15, flags: DIFlagPrototyped, spFlags: 0)
!205 = !DIFile(filename: "/usr/include/stdlib.h", directory: "")
!206 = !DIFile(filename: "/usr/lib/gcc/x86_64-linux-gnu/11/../../../../include/c++/11/bits/std_abs.h", directory: "")
!207 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !208, file: !212, line: 83)
!208 = !DISubprogram(name: "acos", scope: !209, file: !209, line: 53, type: !210, flags: DIFlagPrototyped, spFlags: 0)
!209 = !DIFile(filename: "/usr/include/x86_64-linux-gnu/bits/mathcalls.h", directory: "")
!210 = !DISubroutineType(types: !211)
!211 = !{!157, !157}
!212 = !DIFile(filename: "/usr/lib/gcc/x86_64-linux-gnu/11/../../../../include/c++/11/cmath", directory: "")
!213 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !214, file: !212, line: 102)
!214 = !DISubprogram(name: "asin", scope: !209, file: !209, line: 55, type: !210, flags: DIFlagPrototyped, spFlags: 0)
!215 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !216, file: !212, line: 121)
!216 = !DISubprogram(name: "atan", scope: !209, file: !209, line: 57, type: !210, flags: DIFlagPrototyped, spFlags: 0)
!217 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !218, file: !212, line: 140)
!218 = !DISubprogram(name: "atan2", scope: !209, file: !209, line: 59, type: !219, flags: DIFlagPrototyped, spFlags: 0)
!219 = !DISubroutineType(types: !220)
!220 = !{!157, !157, !157}
!221 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !222, file: !212, line: 161)
!222 = !DISubprogram(name: "ceil", scope: !209, file: !209, line: 159, type: !210, flags: DIFlagPrototyped, spFlags: 0)
!223 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !224, file: !212, line: 180)
!224 = !DISubprogram(name: "cos", scope: !209, file: !209, line: 62, type: !210, flags: DIFlagPrototyped, spFlags: 0)
!225 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !226, file: !212, line: 199)
!226 = !DISubprogram(name: "cosh", scope: !209, file: !209, line: 71, type: !210, flags: DIFlagPrototyped, spFlags: 0)
!227 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !228, file: !212, line: 218)
!228 = !DISubprogram(name: "exp", scope: !209, file: !209, line: 95, type: !210, flags: DIFlagPrototyped, spFlags: 0)
!229 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !230, file: !212, line: 237)
!230 = !DISubprogram(name: "fabs", scope: !209, file: !209, line: 162, type: !210, flags: DIFlagPrototyped, spFlags: 0)
!231 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !232, file: !212, line: 256)
!232 = !DISubprogram(name: "floor", scope: !209, file: !209, line: 165, type: !210, flags: DIFlagPrototyped, spFlags: 0)
!233 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !234, file: !212, line: 275)
!234 = !DISubprogram(name: "fmod", scope: !209, file: !209, line: 168, type: !219, flags: DIFlagPrototyped, spFlags: 0)
!235 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !236, file: !212, line: 296)
!236 = !DISubprogram(name: "frexp", scope: !209, file: !209, line: 98, type: !237, flags: DIFlagPrototyped, spFlags: 0)
!237 = !DISubroutineType(types: !238)
!238 = !{!157, !157, !80}
!239 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !240, file: !212, line: 315)
!240 = !DISubprogram(name: "ldexp", scope: !209, file: !209, line: 101, type: !241, flags: DIFlagPrototyped, spFlags: 0)
!241 = !DISubroutineType(types: !242)
!242 = !{!157, !157, !9}
!243 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !244, file: !212, line: 334)
!244 = !DISubprogram(name: "log", scope: !209, file: !209, line: 104, type: !210, flags: DIFlagPrototyped, spFlags: 0)
!245 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !246, file: !212, line: 353)
!246 = !DISubprogram(name: "log10", scope: !209, file: !209, line: 107, type: !210, flags: DIFlagPrototyped, spFlags: 0)
!247 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !248, file: !212, line: 372)
!248 = !DISubprogram(name: "modf", scope: !209, file: !209, line: 110, type: !249, flags: DIFlagPrototyped, spFlags: 0)
!249 = !DISubroutineType(types: !250)
!250 = !{!157, !157, !251}
!251 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !157, size: 64)
!252 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !253, file: !212, line: 384)
!253 = !DISubprogram(name: "pow", scope: !209, file: !209, line: 140, type: !219, flags: DIFlagPrototyped, spFlags: 0)
!254 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !255, file: !212, line: 421)
!255 = !DISubprogram(name: "sin", scope: !209, file: !209, line: 64, type: !210, flags: DIFlagPrototyped, spFlags: 0)
!256 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !257, file: !212, line: 440)
!257 = !DISubprogram(name: "sinh", scope: !209, file: !209, line: 73, type: !210, flags: DIFlagPrototyped, spFlags: 0)
!258 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !259, file: !212, line: 459)
!259 = !DISubprogram(name: "sqrt", scope: !209, file: !209, line: 143, type: !210, flags: DIFlagPrototyped, spFlags: 0)
!260 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !261, file: !212, line: 478)
!261 = !DISubprogram(name: "tan", scope: !209, file: !209, line: 66, type: !210, flags: DIFlagPrototyped, spFlags: 0)
!262 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !263, file: !212, line: 497)
!263 = !DISubprogram(name: "tanh", scope: !209, file: !209, line: 75, type: !210, flags: DIFlagPrototyped, spFlags: 0)
!264 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !265, file: !267, line: 127)
!265 = !DIDerivedType(tag: DW_TAG_typedef, name: "div_t", file: !205, line: 63, baseType: !266)
!266 = !DICompositeType(tag: DW_TAG_structure_type, file: !205, line: 59, flags: DIFlagFwdDecl, identifier: "_ZTS5div_t")
!267 = !DIFile(filename: "/usr/lib/gcc/x86_64-linux-gnu/11/../../../../include/c++/11/cstdlib", directory: "")
!268 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !269, file: !267, line: 128)
!269 = !DIDerivedType(tag: DW_TAG_typedef, name: "ldiv_t", file: !205, line: 71, baseType: !270)
!270 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !205, line: 67, size: 128, flags: DIFlagTypePassByValue, elements: !271, identifier: "_ZTS6ldiv_t")
!271 = !{!272, !273}
!272 = !DIDerivedType(tag: DW_TAG_member, name: "quot", scope: !270, file: !205, line: 69, baseType: !114, size: 64)
!273 = !DIDerivedType(tag: DW_TAG_member, name: "rem", scope: !270, file: !205, line: 70, baseType: !114, size: 64, offset: 64)
!274 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !275, file: !267, line: 130)
!275 = !DISubprogram(name: "abort", scope: !205, file: !205, line: 598, type: !276, flags: DIFlagPrototyped | DIFlagNoReturn, spFlags: 0)
!276 = !DISubroutineType(types: !277)
!277 = !{null}
!278 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !279, file: !267, line: 134)
!279 = !DISubprogram(name: "atexit", scope: !205, file: !205, line: 602, type: !280, flags: DIFlagPrototyped, spFlags: 0)
!280 = !DISubroutineType(types: !281)
!281 = !{!9, !282}
!282 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !276, size: 64)
!283 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !284, file: !267, line: 140)
!284 = !DISubprogram(name: "atof", scope: !205, file: !205, line: 102, type: !155, flags: DIFlagPrototyped, spFlags: 0)
!285 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !286, file: !267, line: 141)
!286 = !DISubprogram(name: "atoi", scope: !205, file: !205, line: 105, type: !287, flags: DIFlagPrototyped, spFlags: 0)
!287 = !DISubroutineType(types: !288)
!288 = !{!9, !158}
!289 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !290, file: !267, line: 142)
!290 = !DISubprogram(name: "atol", scope: !205, file: !205, line: 108, type: !291, flags: DIFlagPrototyped, spFlags: 0)
!291 = !DISubroutineType(types: !292)
!292 = !{!114, !158}
!293 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !294, file: !267, line: 143)
!294 = !DISubprogram(name: "bsearch", scope: !205, file: !205, line: 828, type: !295, flags: DIFlagPrototyped, spFlags: 0)
!295 = !DISubroutineType(types: !296)
!296 = !{!297, !298, !298, !300, !300, !303}
!297 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!298 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !299, size: 64)
!299 = !DIDerivedType(tag: DW_TAG_const_type, baseType: null)
!300 = !DIDerivedType(tag: DW_TAG_typedef, name: "size_t", file: !301, line: 46, baseType: !302)
!301 = !DIFile(filename: "llvm-install-tulip/lib/clang/9.0.0/include/stddef.h", directory: "/scratch/ah7226")
!302 = !DIBasicType(name: "long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!303 = !DIDerivedType(tag: DW_TAG_typedef, name: "__compar_fn_t", file: !205, line: 816, baseType: !304)
!304 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !305, size: 64)
!305 = !DISubroutineType(types: !306)
!306 = !{!9, !298, !298}
!307 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !308, file: !267, line: 144)
!308 = !DISubprogram(name: "calloc", scope: !205, file: !205, line: 543, type: !309, flags: DIFlagPrototyped, spFlags: 0)
!309 = !DISubroutineType(types: !310)
!310 = !{!297, !300, !300}
!311 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !312, file: !267, line: 145)
!312 = !DISubprogram(name: "div", scope: !205, file: !205, line: 860, type: !313, flags: DIFlagPrototyped, spFlags: 0)
!313 = !DISubroutineType(types: !314)
!314 = !{!265, !9, !9}
!315 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !316, file: !267, line: 146)
!316 = !DISubprogram(name: "exit", scope: !205, file: !205, line: 624, type: !317, flags: DIFlagPrototyped | DIFlagNoReturn, spFlags: 0)
!317 = !DISubroutineType(types: !318)
!318 = !{null, !9}
!319 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !320, file: !267, line: 147)
!320 = !DISubprogram(name: "free", scope: !205, file: !205, line: 555, type: !321, flags: DIFlagPrototyped, spFlags: 0)
!321 = !DISubroutineType(types: !322)
!322 = !{null, !297}
!323 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !324, file: !267, line: 148)
!324 = !DISubprogram(name: "getenv", scope: !205, file: !205, line: 641, type: !325, flags: DIFlagPrototyped, spFlags: 0)
!325 = !DISubroutineType(types: !326)
!326 = !{!327, !158}
!327 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !160, size: 64)
!328 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !329, file: !267, line: 149)
!329 = !DISubprogram(name: "labs", scope: !205, file: !205, line: 849, type: !112, flags: DIFlagPrototyped, spFlags: 0)
!330 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !331, file: !267, line: 150)
!331 = !DISubprogram(name: "ldiv", scope: !205, file: !205, line: 862, type: !332, flags: DIFlagPrototyped, spFlags: 0)
!332 = !DISubroutineType(types: !333)
!333 = !{!269, !114, !114}
!334 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !335, file: !267, line: 151)
!335 = !DISubprogram(name: "malloc", scope: !205, file: !205, line: 540, type: !336, flags: DIFlagPrototyped, spFlags: 0)
!336 = !DISubroutineType(types: !337)
!337 = !{!297, !300}
!338 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !339, file: !267, line: 153)
!339 = !DISubprogram(name: "mblen", scope: !205, file: !205, line: 930, type: !340, flags: DIFlagPrototyped, spFlags: 0)
!340 = !DISubroutineType(types: !341)
!341 = !{!9, !158, !300}
!342 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !343, file: !267, line: 154)
!343 = !DISubprogram(name: "mbstowcs", scope: !205, file: !205, line: 941, type: !344, flags: DIFlagPrototyped, spFlags: 0)
!344 = !DISubroutineType(types: !345)
!345 = !{!300, !346, !349, !300}
!346 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !347)
!347 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !348, size: 64)
!348 = !DIBasicType(name: "wchar_t", size: 32, encoding: DW_ATE_signed)
!349 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !158)
!350 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !351, file: !267, line: 155)
!351 = !DISubprogram(name: "mbtowc", scope: !205, file: !205, line: 933, type: !352, flags: DIFlagPrototyped, spFlags: 0)
!352 = !DISubroutineType(types: !353)
!353 = !{!9, !346, !349, !300}
!354 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !355, file: !267, line: 157)
!355 = !DISubprogram(name: "qsort", scope: !205, file: !205, line: 838, type: !356, flags: DIFlagPrototyped, spFlags: 0)
!356 = !DISubroutineType(types: !357)
!357 = !{null, !297, !300, !300, !303}
!358 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !359, file: !267, line: 163)
!359 = !DISubprogram(name: "rand", scope: !205, file: !205, line: 454, type: !360, flags: DIFlagPrototyped, spFlags: 0)
!360 = !DISubroutineType(types: !8)
!361 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !362, file: !267, line: 164)
!362 = !DISubprogram(name: "realloc", scope: !205, file: !205, line: 551, type: !363, flags: DIFlagPrototyped, spFlags: 0)
!363 = !DISubroutineType(types: !364)
!364 = !{!297, !297, !300}
!365 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !366, file: !267, line: 165)
!366 = !DISubprogram(name: "srand", scope: !205, file: !205, line: 456, type: !367, flags: DIFlagPrototyped, spFlags: 0)
!367 = !DISubroutineType(types: !368)
!368 = !{null, !369}
!369 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!370 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !371, file: !267, line: 166)
!371 = !DISubprogram(name: "strtod", scope: !205, file: !205, line: 118, type: !372, flags: DIFlagPrototyped, spFlags: 0)
!372 = !DISubroutineType(types: !373)
!373 = !{!157, !349, !374}
!374 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !375)
!375 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !327, size: 64)
!376 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !377, file: !267, line: 167)
!377 = !DISubprogram(name: "strtol", scope: !205, file: !205, line: 177, type: !378, flags: DIFlagPrototyped, spFlags: 0)
!378 = !DISubroutineType(types: !379)
!379 = !{!114, !349, !374, !9}
!380 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !381, file: !267, line: 168)
!381 = !DISubprogram(name: "strtoul", scope: !205, file: !205, line: 181, type: !382, flags: DIFlagPrototyped, spFlags: 0)
!382 = !DISubroutineType(types: !383)
!383 = !{!302, !349, !374, !9}
!384 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !385, file: !267, line: 169)
!385 = !DISubprogram(name: "system", scope: !205, file: !205, line: 791, type: !287, flags: DIFlagPrototyped, spFlags: 0)
!386 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !387, file: !267, line: 171)
!387 = !DISubprogram(name: "wcstombs", scope: !205, file: !205, line: 945, type: !388, flags: DIFlagPrototyped, spFlags: 0)
!388 = !DISubroutineType(types: !389)
!389 = !{!300, !390, !391, !300}
!390 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !327)
!391 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !392)
!392 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !393, size: 64)
!393 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !348)
!394 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !395, file: !267, line: 172)
!395 = !DISubprogram(name: "wctomb", scope: !205, file: !205, line: 937, type: !396, flags: DIFlagPrototyped, spFlags: 0)
!396 = !DISubroutineType(types: !397)
!397 = !{!9, !327, !348}
!398 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !399, entity: !400, file: !267, line: 200)
!399 = !DINamespace(name: "__gnu_cxx", scope: null)
!400 = !DIDerivedType(tag: DW_TAG_typedef, name: "lldiv_t", file: !205, line: 81, baseType: !401)
!401 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !205, line: 77, size: 128, flags: DIFlagTypePassByValue, elements: !402, identifier: "_ZTS7lldiv_t")
!402 = !{!403, !404}
!403 = !DIDerivedType(tag: DW_TAG_member, name: "quot", scope: !401, file: !205, line: 79, baseType: !125, size: 64)
!404 = !DIDerivedType(tag: DW_TAG_member, name: "rem", scope: !401, file: !205, line: 80, baseType: !125, size: 64, offset: 64)
!405 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !399, entity: !406, file: !267, line: 206)
!406 = !DISubprogram(name: "_Exit", scope: !205, file: !205, line: 636, type: !317, flags: DIFlagPrototyped | DIFlagNoReturn, spFlags: 0)
!407 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !399, entity: !408, file: !267, line: 210)
!408 = !DISubprogram(name: "llabs", scope: !205, file: !205, line: 852, type: !123, flags: DIFlagPrototyped, spFlags: 0)
!409 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !399, entity: !410, file: !267, line: 216)
!410 = !DISubprogram(name: "lldiv", scope: !205, file: !205, line: 866, type: !411, flags: DIFlagPrototyped, spFlags: 0)
!411 = !DISubroutineType(types: !412)
!412 = !{!400, !125, !125}
!413 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !399, entity: !414, file: !267, line: 227)
!414 = !DISubprogram(name: "atoll", scope: !205, file: !205, line: 113, type: !415, flags: DIFlagPrototyped, spFlags: 0)
!415 = !DISubroutineType(types: !416)
!416 = !{!125, !158}
!417 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !399, entity: !418, file: !267, line: 228)
!418 = !DISubprogram(name: "strtoll", scope: !205, file: !205, line: 201, type: !419, flags: DIFlagPrototyped, spFlags: 0)
!419 = !DISubroutineType(types: !420)
!420 = !{!125, !349, !374, !9}
!421 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !399, entity: !422, file: !267, line: 229)
!422 = !DISubprogram(name: "strtoull", scope: !205, file: !205, line: 206, type: !423, flags: DIFlagPrototyped, spFlags: 0)
!423 = !DISubroutineType(types: !424)
!424 = !{!425, !349, !374, !9}
!425 = !DIBasicType(name: "long long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!426 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !399, entity: !427, file: !267, line: 231)
!427 = !DISubprogram(name: "strtof", scope: !205, file: !205, line: 124, type: !428, flags: DIFlagPrototyped, spFlags: 0)
!428 = !DISubroutineType(types: !429)
!429 = !{!21, !349, !374}
!430 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !399, entity: !431, file: !267, line: 232)
!431 = !DISubprogram(name: "strtold", scope: !205, file: !205, line: 127, type: !432, flags: DIFlagPrototyped, spFlags: 0)
!432 = !DISubroutineType(types: !433)
!433 = !{!434, !349, !374}
!434 = !DIBasicType(name: "long double", size: 64, encoding: DW_ATE_float)
!435 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !400, file: !267, line: 240)
!436 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !406, file: !267, line: 242)
!437 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !408, file: !267, line: 244)
!438 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !439, file: !267, line: 245)
!439 = !DISubprogram(name: "div", linkageName: "_ZN9__gnu_cxx3divExx", scope: !399, file: !267, line: 213, type: !411, flags: DIFlagPrototyped, spFlags: 0)
!440 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !410, file: !267, line: 246)
!441 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !414, file: !267, line: 248)
!442 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !427, file: !267, line: 249)
!443 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !418, file: !267, line: 250)
!444 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !422, file: !267, line: 251)
!445 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !431, file: !267, line: 252)
!446 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !275, file: !447, line: 38)
!447 = !DIFile(filename: "/usr/lib/gcc/x86_64-linux-gnu/11/../../../../include/c++/11/stdlib.h", directory: "")
!448 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !279, file: !447, line: 39)
!449 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !316, file: !447, line: 40)
!450 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !265, file: !447, line: 51)
!451 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !269, file: !447, line: 52)
!452 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !453, file: !447, line: 54)
!453 = !DISubprogram(name: "abs", linkageName: "_ZSt3abse", scope: !12, file: !206, line: 79, type: !454, flags: DIFlagPrototyped, spFlags: 0)
!454 = !DISubroutineType(types: !455)
!455 = !{!434, !434}
!456 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !284, file: !447, line: 55)
!457 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !286, file: !447, line: 56)
!458 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !290, file: !447, line: 57)
!459 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !294, file: !447, line: 58)
!460 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !308, file: !447, line: 59)
!461 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !439, file: !447, line: 60)
!462 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !320, file: !447, line: 61)
!463 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !324, file: !447, line: 62)
!464 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !329, file: !447, line: 63)
!465 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !331, file: !447, line: 64)
!466 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !335, file: !447, line: 65)
!467 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !339, file: !447, line: 67)
!468 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !343, file: !447, line: 68)
!469 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !351, file: !447, line: 69)
!470 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !355, file: !447, line: 71)
!471 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !359, file: !447, line: 72)
!472 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !362, file: !447, line: 73)
!473 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !366, file: !447, line: 74)
!474 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !371, file: !447, line: 75)
!475 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !377, file: !447, line: 76)
!476 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !381, file: !447, line: 77)
!477 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !385, file: !447, line: 78)
!478 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !387, file: !447, line: 80)
!479 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !5, entity: !395, file: !447, line: 81)
!480 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !481, file: !483, line: 414)
!481 = !DISubprogram(name: "acosf", linkageName: "_ZL5acosff", scope: !482, file: !482, line: 1126, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!482 = !DIFile(filename: "/usr/local/cuda/include/math_functions.hpp", directory: "")
!483 = !DIFile(filename: "llvm-install-tulip/lib/clang/9.0.0/include/__clang_cuda_cmath.h", directory: "/scratch/ah7226")
!484 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !485, file: !483, line: 415)
!485 = !DISubprogram(name: "acoshf", linkageName: "_ZL6acoshff", scope: !482, file: !482, line: 1154, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!486 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !487, file: !483, line: 416)
!487 = !DISubprogram(name: "asinf", linkageName: "_ZL5asinff", scope: !482, file: !482, line: 1121, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!488 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !489, file: !483, line: 417)
!489 = !DISubprogram(name: "asinhf", linkageName: "_ZL6asinhff", scope: !482, file: !482, line: 1159, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!490 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !491, file: !483, line: 418)
!491 = !DISubprogram(name: "atan2f", linkageName: "_ZL6atan2fff", scope: !482, file: !482, line: 1111, type: !32, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!492 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !493, file: !483, line: 419)
!493 = !DISubprogram(name: "atanf", linkageName: "_ZL5atanff", scope: !482, file: !482, line: 1116, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!494 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !495, file: !483, line: 420)
!495 = !DISubprogram(name: "atanhf", linkageName: "_ZL6atanhff", scope: !482, file: !482, line: 1164, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!496 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !497, file: !483, line: 421)
!497 = !DISubprogram(name: "cbrtf", linkageName: "_ZL5cbrtff", scope: !482, file: !482, line: 1199, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!498 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !499, file: !483, line: 422)
!499 = !DISubprogram(name: "ceilf", linkageName: "_ZL5ceilff", scope: !500, file: !500, line: 647, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!500 = !DIFile(filename: "/usr/local/cuda/include/device_functions.hpp", directory: "")
!501 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !502, file: !483, line: 423)
!502 = !DISubprogram(name: "copysignf", linkageName: "_ZL9copysignfff", scope: !482, file: !482, line: 973, type: !32, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!503 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !504, file: !483, line: 424)
!504 = !DISubprogram(name: "cosf", linkageName: "_ZL4cosff", scope: !482, file: !482, line: 1027, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!505 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !506, file: !483, line: 425)
!506 = !DISubprogram(name: "coshf", linkageName: "_ZL5coshff", scope: !482, file: !482, line: 1096, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!507 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !508, file: !483, line: 426)
!508 = !DISubprogram(name: "erfcf", linkageName: "_ZL5erfcff", scope: !482, file: !482, line: 1259, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!509 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !510, file: !483, line: 427)
!510 = !DISubprogram(name: "erff", linkageName: "_ZL4erfff", scope: !482, file: !482, line: 1249, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!511 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !512, file: !483, line: 428)
!512 = !DISubprogram(name: "exp2f", linkageName: "_ZL5exp2ff", scope: !500, file: !500, line: 637, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!513 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !514, file: !483, line: 429)
!514 = !DISubprogram(name: "expf", linkageName: "_ZL4expff", scope: !482, file: !482, line: 1078, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!515 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !516, file: !483, line: 430)
!516 = !DISubprogram(name: "expm1f", linkageName: "_ZL6expm1ff", scope: !482, file: !482, line: 1169, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!517 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !518, file: !483, line: 431)
!518 = !DISubprogram(name: "fabsf", linkageName: "_ZL5fabsff", scope: !500, file: !500, line: 582, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!519 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !520, file: !483, line: 432)
!520 = !DISubprogram(name: "fdimf", linkageName: "_ZL5fdimfff", scope: !482, file: !482, line: 1385, type: !32, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!521 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !522, file: !483, line: 433)
!522 = !DISubprogram(name: "floorf", linkageName: "_ZL6floorff", scope: !500, file: !500, line: 572, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!523 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !524, file: !483, line: 434)
!524 = !DISubprogram(name: "fmaf", linkageName: "_ZL4fmaffff", scope: !482, file: !482, line: 1337, type: !64, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!525 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !526, file: !483, line: 435)
!526 = !DISubprogram(name: "fmaxf", linkageName: "_ZL5fmaxfff", scope: !500, file: !500, line: 602, type: !32, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!527 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !528, file: !483, line: 436)
!528 = !DISubprogram(name: "fminf", linkageName: "_ZL5fminfff", scope: !500, file: !500, line: 597, type: !32, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!529 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !530, file: !483, line: 437)
!530 = !DISubprogram(name: "fmodf", linkageName: "_ZL5fmodfff", scope: !482, file: !482, line: 1322, type: !32, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!531 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !532, file: !483, line: 438)
!532 = !DISubprogram(name: "frexpf", linkageName: "_ZL6frexpffPi", scope: !482, file: !482, line: 1312, type: !78, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!533 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !534, file: !483, line: 439)
!534 = !DISubprogram(name: "hypotf", linkageName: "_ZL6hypotfff", scope: !482, file: !482, line: 1174, type: !32, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!535 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !536, file: !483, line: 440)
!536 = !DISubprogram(name: "ilogbf", linkageName: "_ZL6ilogbff", scope: !482, file: !482, line: 1390, type: !74, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!537 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !538, file: !483, line: 441)
!538 = !DISubprogram(name: "ldexpf", linkageName: "_ZL6ldexpffi", scope: !482, file: !482, line: 1289, type: !117, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!539 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !540, file: !483, line: 442)
!540 = !DISubprogram(name: "lgammaf", linkageName: "_ZL7lgammaff", scope: !482, file: !482, line: 1284, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!541 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !542, file: !483, line: 443)
!542 = !DISubprogram(name: "llrintf", linkageName: "_ZL7llrintff", scope: !482, file: !482, line: 933, type: !128, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!543 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !544, file: !483, line: 444)
!544 = !DISubprogram(name: "llroundf", linkageName: "_ZL8llroundff", scope: !482, file: !482, line: 1371, type: !128, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!545 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !546, file: !483, line: 445)
!546 = !DISubprogram(name: "log10f", linkageName: "_ZL6log10ff", scope: !482, file: !482, line: 1140, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!547 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !548, file: !483, line: 446)
!548 = !DISubprogram(name: "log1pf", linkageName: "_ZL6log1pff", scope: !482, file: !482, line: 1149, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!549 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !550, file: !483, line: 447)
!550 = !DISubprogram(name: "log2f", linkageName: "_ZL5log2ff", scope: !482, file: !482, line: 1069, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!551 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !552, file: !483, line: 448)
!552 = !DISubprogram(name: "logbf", linkageName: "_ZL5logbff", scope: !482, file: !482, line: 1395, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!553 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !554, file: !483, line: 449)
!554 = !DISubprogram(name: "logf", linkageName: "_ZL4logff", scope: !482, file: !482, line: 1131, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!555 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !556, file: !483, line: 450)
!556 = !DISubprogram(name: "lrintf", linkageName: "_ZL6lrintff", scope: !482, file: !482, line: 924, type: !142, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!557 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !558, file: !483, line: 451)
!558 = !DISubprogram(name: "lroundf", linkageName: "_ZL7lroundff", scope: !482, file: !482, line: 1376, type: !142, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!559 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !560, file: !483, line: 452)
!560 = !DISubprogram(name: "modff", linkageName: "_ZL5modfffPf", scope: !482, file: !482, line: 1317, type: !150, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!561 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !562, file: !483, line: 453)
!562 = !DISubprogram(name: "nearbyintf", linkageName: "_ZL10nearbyintff", scope: !482, file: !482, line: 938, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!563 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !564, file: !483, line: 454)
!564 = !DISubprogram(name: "nextafterf", linkageName: "_ZL10nextafterfff", scope: !482, file: !482, line: 1002, type: !32, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!565 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !566, file: !483, line: 455)
!566 = !DISubprogram(name: "powf", linkageName: "_ZL4powfff", scope: !482, file: !482, line: 1352, type: !32, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!567 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !568, file: !483, line: 456)
!568 = !DISubprogram(name: "remainderf", linkageName: "_ZL10remainderfff", scope: !482, file: !482, line: 1327, type: !32, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!569 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !570, file: !483, line: 457)
!570 = !DISubprogram(name: "remquof", linkageName: "_ZL7remquofffPi", scope: !482, file: !482, line: 1332, type: !175, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!571 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !572, file: !483, line: 458)
!572 = !DISubprogram(name: "rintf", linkageName: "_ZL5rintff", scope: !482, file: !482, line: 919, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!573 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !574, file: !483, line: 459)
!574 = !DISubprogram(name: "roundf", linkageName: "_ZL6roundff", scope: !482, file: !482, line: 1366, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!575 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !576, file: !483, line: 462)
!576 = !DISubprogram(name: "scalblnf", linkageName: "_ZL8scalblnffl", scope: !482, file: !482, line: 1299, type: !183, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!577 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !578, file: !483, line: 464)
!578 = !DISubprogram(name: "scalbnf", linkageName: "_ZL7scalbnffi", scope: !482, file: !482, line: 1294, type: !117, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!579 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !580, file: !483, line: 465)
!580 = !DISubprogram(name: "sinf", linkageName: "_ZL4sinff", scope: !482, file: !482, line: 1018, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!581 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !582, file: !483, line: 466)
!582 = !DISubprogram(name: "sinhf", linkageName: "_ZL5sinhff", scope: !482, file: !482, line: 1101, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!583 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !584, file: !483, line: 467)
!584 = !DISubprogram(name: "sqrtf", linkageName: "_ZL5sqrtff", scope: !500, file: !500, line: 887, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!585 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !586, file: !483, line: 468)
!586 = !DISubprogram(name: "tanf", linkageName: "_ZL4tanff", scope: !482, file: !482, line: 1060, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!587 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !588, file: !483, line: 469)
!588 = !DISubprogram(name: "tanhf", linkageName: "_ZL5tanhff", scope: !482, file: !482, line: 1106, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!589 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !590, file: !483, line: 470)
!590 = !DISubprogram(name: "tgammaf", linkageName: "_ZL7tgammaff", scope: !482, file: !482, line: 1361, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!591 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !592, file: !483, line: 471)
!592 = !DISubprogram(name: "truncf", linkageName: "_ZL6truncff", scope: !500, file: !500, line: 642, type: !19, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit)
!593 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !594, file: !598, line: 98)
!594 = !DIDerivedType(tag: DW_TAG_typedef, name: "FILE", file: !595, line: 7, baseType: !596)
!595 = !DIFile(filename: "/usr/include/x86_64-linux-gnu/bits/types/FILE.h", directory: "")
!596 = !DICompositeType(tag: DW_TAG_structure_type, name: "_IO_FILE", file: !597, line: 49, flags: DIFlagFwdDecl, identifier: "_ZTS8_IO_FILE")
!597 = !DIFile(filename: "/usr/include/x86_64-linux-gnu/bits/types/struct_FILE.h", directory: "")
!598 = !DIFile(filename: "/usr/lib/gcc/x86_64-linux-gnu/11/../../../../include/c++/11/cstdio", directory: "")
!599 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !600, file: !598, line: 99)
!600 = !DIDerivedType(tag: DW_TAG_typedef, name: "fpos_t", file: !601, line: 84, baseType: !602)
!601 = !DIFile(filename: "/usr/include/stdio.h", directory: "")
!602 = !DIDerivedType(tag: DW_TAG_typedef, name: "__fpos_t", file: !603, line: 14, baseType: !604)
!603 = !DIFile(filename: "/usr/include/x86_64-linux-gnu/bits/types/__fpos_t.h", directory: "")
!604 = !DICompositeType(tag: DW_TAG_structure_type, name: "_G_fpos_t", file: !603, line: 10, flags: DIFlagFwdDecl, identifier: "_ZTS9_G_fpos_t")
!605 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !606, file: !598, line: 101)
!606 = !DISubprogram(name: "clearerr", scope: !601, file: !601, line: 786, type: !607, flags: DIFlagPrototyped, spFlags: 0)
!607 = !DISubroutineType(types: !608)
!608 = !{null, !609}
!609 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !594, size: 64)
!610 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !611, file: !598, line: 102)
!611 = !DISubprogram(name: "fclose", scope: !601, file: !601, line: 178, type: !612, flags: DIFlagPrototyped, spFlags: 0)
!612 = !DISubroutineType(types: !613)
!613 = !{!9, !609}
!614 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !615, file: !598, line: 103)
!615 = !DISubprogram(name: "feof", scope: !601, file: !601, line: 788, type: !612, flags: DIFlagPrototyped, spFlags: 0)
!616 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !617, file: !598, line: 104)
!617 = !DISubprogram(name: "ferror", scope: !601, file: !601, line: 790, type: !612, flags: DIFlagPrototyped, spFlags: 0)
!618 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !619, file: !598, line: 105)
!619 = !DISubprogram(name: "fflush", scope: !601, file: !601, line: 230, type: !612, flags: DIFlagPrototyped, spFlags: 0)
!620 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !621, file: !598, line: 106)
!621 = !DISubprogram(name: "fgetc", scope: !601, file: !601, line: 513, type: !612, flags: DIFlagPrototyped, spFlags: 0)
!622 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !623, file: !598, line: 107)
!623 = !DISubprogram(name: "fgetpos", scope: !601, file: !601, line: 760, type: !624, flags: DIFlagPrototyped, spFlags: 0)
!624 = !DISubroutineType(types: !625)
!625 = !{!9, !626, !627}
!626 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !609)
!627 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !628)
!628 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !600, size: 64)
!629 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !630, file: !598, line: 108)
!630 = !DISubprogram(name: "fgets", scope: !601, file: !601, line: 592, type: !631, flags: DIFlagPrototyped, spFlags: 0)
!631 = !DISubroutineType(types: !632)
!632 = !{!327, !390, !9, !626}
!633 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !634, file: !598, line: 109)
!634 = !DISubprogram(name: "fopen", scope: !601, file: !601, line: 258, type: !635, flags: DIFlagPrototyped, spFlags: 0)
!635 = !DISubroutineType(types: !636)
!636 = !{!609, !349, !349}
!637 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !638, file: !598, line: 110)
!638 = !DISubprogram(name: "fprintf", scope: !601, file: !601, line: 350, type: !639, flags: DIFlagPrototyped, spFlags: 0)
!639 = !DISubroutineType(types: !640)
!640 = !{!9, !626, !349, null}
!641 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !642, file: !598, line: 111)
!642 = !DISubprogram(name: "fputc", scope: !601, file: !601, line: 549, type: !643, flags: DIFlagPrototyped, spFlags: 0)
!643 = !DISubroutineType(types: !644)
!644 = !{!9, !9, !609}
!645 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !646, file: !598, line: 112)
!646 = !DISubprogram(name: "fputs", scope: !601, file: !601, line: 655, type: !647, flags: DIFlagPrototyped, spFlags: 0)
!647 = !DISubroutineType(types: !648)
!648 = !{!9, !349, !626}
!649 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !650, file: !598, line: 113)
!650 = !DISubprogram(name: "fread", scope: !601, file: !601, line: 675, type: !651, flags: DIFlagPrototyped, spFlags: 0)
!651 = !DISubroutineType(types: !652)
!652 = !{!300, !653, !300, !300, !626}
!653 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !297)
!654 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !655, file: !598, line: 114)
!655 = !DISubprogram(name: "freopen", scope: !601, file: !601, line: 265, type: !656, flags: DIFlagPrototyped, spFlags: 0)
!656 = !DISubroutineType(types: !657)
!657 = !{!609, !349, !349, !626}
!658 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !659, file: !598, line: 115)
!659 = !DISubprogram(name: "fscanf", scope: !601, file: !601, line: 415, type: !639, flags: DIFlagPrototyped, spFlags: 0)
!660 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !661, file: !598, line: 116)
!661 = !DISubprogram(name: "fseek", scope: !601, file: !601, line: 713, type: !662, flags: DIFlagPrototyped, spFlags: 0)
!662 = !DISubroutineType(types: !663)
!663 = !{!9, !609, !114, !9}
!664 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !665, file: !598, line: 117)
!665 = !DISubprogram(name: "fsetpos", scope: !601, file: !601, line: 765, type: !666, flags: DIFlagPrototyped, spFlags: 0)
!666 = !DISubroutineType(types: !667)
!667 = !{!9, !609, !668}
!668 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !669, size: 64)
!669 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !600)
!670 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !671, file: !598, line: 118)
!671 = !DISubprogram(name: "ftell", scope: !601, file: !601, line: 718, type: !672, flags: DIFlagPrototyped, spFlags: 0)
!672 = !DISubroutineType(types: !673)
!673 = !{!114, !609}
!674 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !675, file: !598, line: 119)
!675 = !DISubprogram(name: "fwrite", scope: !601, file: !601, line: 681, type: !676, flags: DIFlagPrototyped, spFlags: 0)
!676 = !DISubroutineType(types: !677)
!677 = !{!300, !678, !300, !300, !626}
!678 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !298)
!679 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !680, file: !598, line: 120)
!680 = !DISubprogram(name: "getc", scope: !601, file: !601, line: 514, type: !612, flags: DIFlagPrototyped, spFlags: 0)
!681 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !682, file: !598, line: 121)
!682 = !DISubprogram(name: "getchar", scope: !601, file: !601, line: 520, type: !360, flags: DIFlagPrototyped, spFlags: 0)
!683 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !684, file: !598, line: 124)
!684 = !DISubprogram(name: "gets", scope: !601, file: !601, line: 605, type: !685, flags: DIFlagPrototyped, spFlags: 0)
!685 = !DISubroutineType(types: !686)
!686 = !{!327, !327}
!687 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !688, file: !598, line: 126)
!688 = !DISubprogram(name: "perror", scope: !601, file: !601, line: 804, type: !689, flags: DIFlagPrototyped, spFlags: 0)
!689 = !DISubroutineType(types: !690)
!690 = !{null, !158}
!691 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !692, file: !598, line: 127)
!692 = !DISubprogram(name: "printf", scope: !601, file: !601, line: 356, type: !693, flags: DIFlagPrototyped, spFlags: 0)
!693 = !DISubroutineType(types: !694)
!694 = !{!9, !349, null}
!695 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !696, file: !598, line: 128)
!696 = !DISubprogram(name: "putc", scope: !601, file: !601, line: 550, type: !643, flags: DIFlagPrototyped, spFlags: 0)
!697 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !698, file: !598, line: 129)
!698 = !DISubprogram(name: "putchar", scope: !601, file: !601, line: 556, type: !15, flags: DIFlagPrototyped, spFlags: 0)
!699 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !700, file: !598, line: 130)
!700 = !DISubprogram(name: "puts", scope: !601, file: !601, line: 661, type: !287, flags: DIFlagPrototyped, spFlags: 0)
!701 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !702, file: !598, line: 131)
!702 = !DISubprogram(name: "remove", scope: !601, file: !601, line: 152, type: !287, flags: DIFlagPrototyped, spFlags: 0)
!703 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !704, file: !598, line: 132)
!704 = !DISubprogram(name: "rename", scope: !601, file: !601, line: 154, type: !705, flags: DIFlagPrototyped, spFlags: 0)
!705 = !DISubroutineType(types: !706)
!706 = !{!9, !158, !158}
!707 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !708, file: !598, line: 133)
!708 = !DISubprogram(name: "rewind", scope: !601, file: !601, line: 723, type: !607, flags: DIFlagPrototyped, spFlags: 0)
!709 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !710, file: !598, line: 134)
!710 = !DISubprogram(name: "scanf", scope: !601, file: !601, line: 421, type: !693, flags: DIFlagPrototyped, spFlags: 0)
!711 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !712, file: !598, line: 135)
!712 = !DISubprogram(name: "setbuf", scope: !601, file: !601, line: 328, type: !713, flags: DIFlagPrototyped, spFlags: 0)
!713 = !DISubroutineType(types: !714)
!714 = !{null, !626, !390}
!715 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !716, file: !598, line: 136)
!716 = !DISubprogram(name: "setvbuf", scope: !601, file: !601, line: 332, type: !717, flags: DIFlagPrototyped, spFlags: 0)
!717 = !DISubroutineType(types: !718)
!718 = !{!9, !626, !390, !9, !300}
!719 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !720, file: !598, line: 137)
!720 = !DISubprogram(name: "sprintf", scope: !601, file: !601, line: 358, type: !721, flags: DIFlagPrototyped, spFlags: 0)
!721 = !DISubroutineType(types: !722)
!722 = !{!9, !390, !349, null}
!723 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !724, file: !598, line: 138)
!724 = !DISubprogram(name: "sscanf", scope: !601, file: !601, line: 423, type: !725, flags: DIFlagPrototyped, spFlags: 0)
!725 = !DISubroutineType(types: !726)
!726 = !{!9, !349, !349, null}
!727 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !728, file: !598, line: 139)
!728 = !DISubprogram(name: "tmpfile", scope: !601, file: !601, line: 188, type: !729, flags: DIFlagPrototyped, spFlags: 0)
!729 = !DISubroutineType(types: !730)
!730 = !{!609}
!731 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !732, file: !598, line: 141)
!732 = !DISubprogram(name: "tmpnam", scope: !601, file: !601, line: 205, type: !685, flags: DIFlagPrototyped, spFlags: 0)
!733 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !734, file: !598, line: 143)
!734 = !DISubprogram(name: "ungetc", scope: !601, file: !601, line: 668, type: !643, flags: DIFlagPrototyped, spFlags: 0)
!735 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !736, file: !598, line: 144)
!736 = !DISubprogram(name: "vfprintf", scope: !601, file: !601, line: 365, type: !737, flags: DIFlagPrototyped, spFlags: 0)
!737 = !DISubroutineType(types: !738)
!738 = !{!9, !626, !349, !739}
!739 = !DIDerivedType(tag: DW_TAG_typedef, name: "__gnuc_va_list", file: !740, line: 32, baseType: !741)
!740 = !DIFile(filename: "llvm-install-tulip/lib/clang/9.0.0/include/stdarg.h", directory: "/scratch/ah7226")
!741 = !DIDerivedType(tag: DW_TAG_typedef, name: "__builtin_va_list", file: !6, baseType: !327)
!742 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !743, file: !598, line: 145)
!743 = !DISubprogram(name: "vprintf", scope: !601, file: !601, line: 371, type: !744, flags: DIFlagPrototyped, spFlags: 0)
!744 = !DISubroutineType(types: !745)
!745 = !{!9, !349, !739}
!746 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !747, file: !598, line: 146)
!747 = !DISubprogram(name: "vsprintf", scope: !601, file: !601, line: 373, type: !748, flags: DIFlagPrototyped, spFlags: 0)
!748 = !DISubroutineType(types: !749)
!749 = !{!9, !390, !349, !739}
!750 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !399, entity: !751, file: !598, line: 175)
!751 = !DISubprogram(name: "snprintf", scope: !601, file: !601, line: 378, type: !752, flags: DIFlagPrototyped, spFlags: 0)
!752 = !DISubroutineType(types: !753)
!753 = !{!9, !390, !300, !349, null}
!754 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !399, entity: !755, file: !598, line: 176)
!755 = !DISubprogram(name: "vfscanf", scope: !601, file: !601, line: 459, type: !737, flags: DIFlagPrototyped, spFlags: 0)
!756 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !399, entity: !757, file: !598, line: 177)
!757 = !DISubprogram(name: "vscanf", scope: !601, file: !601, line: 467, type: !744, flags: DIFlagPrototyped, spFlags: 0)
!758 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !399, entity: !759, file: !598, line: 178)
!759 = !DISubprogram(name: "vsnprintf", scope: !601, file: !601, line: 382, type: !760, flags: DIFlagPrototyped, spFlags: 0)
!760 = !DISubroutineType(types: !761)
!761 = !{!9, !390, !300, !349, !739}
!762 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !399, entity: !763, file: !598, line: 179)
!763 = !DISubprogram(name: "vsscanf", scope: !601, file: !601, line: 471, type: !764, flags: DIFlagPrototyped, spFlags: 0)
!764 = !DISubroutineType(types: !765)
!765 = !{!9, !349, !349, !739}
!766 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !751, file: !598, line: 185)
!767 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !755, file: !598, line: 186)
!768 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !757, file: !598, line: 187)
!769 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !759, file: !598, line: 188)
!770 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !12, entity: !763, file: !598, line: 189)
!771 = !{void (double*, double*, double*, double)* @_Z10gpu_kernelPdS_S_d, !"kernel", i32 1}
!772 = !{null, !"align", i32 8}
!773 = !{null, !"align", i32 8, !"align", i32 65544, !"align", i32 131080}
!774 = !{null, !"align", i32 16}
!775 = !{null, !"align", i32 16, !"align", i32 65552, !"align", i32 131088}
!776 = !{!"clang version 9.0.0 (https://github.com/yebinchon/llvm-project/ a1efa594106d738d0b74c9e4e2b9b779eb8b7d25)"}
!777 = !{i32 1, i32 2}
!778 = distinct !DISubprogram(name: "gpu_kernel", linkageName: "_Z10gpu_kernelPdS_S_d", scope: !6, file: !6, line: 464, type: !779, scopeLine: 467, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !7)
!779 = !DISubroutineType(types: !780)
!780 = !{null, !251, !251, !251, !157}
!781 = !DILocalVariable(name: "a", arg: 1, scope: !782, file: !783, line: 225, type: !157)
!782 = distinct !DISubprogram(name: "log", linkageName: "_ZL3logd", scope: !783, file: !783, line: 225, type: !210, scopeLine: 226, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !5, retainedNodes: !7)
!783 = !DIFile(filename: "/usr/local/cuda/include/math_functions_dbl_ptx3.hpp", directory: "")
!784 = !DILocation(line: 225, column: 52, scope: !782, inlinedAt: !785)
!785 = distinct !DILocation(line: 528, column: 19, scope: !786)
!786 = distinct !DILexicalBlock(scope: !787, file: !6, line: 527, column: 15)
!787 = distinct !DILexicalBlock(scope: !788, file: !6, line: 527, column: 7)
!788 = distinct !DILexicalBlock(scope: !789, file: !6, line: 523, column: 33)
!789 = distinct !DILexicalBlock(scope: !790, file: !6, line: 523, column: 3)
!790 = distinct !DILexicalBlock(scope: !791, file: !6, line: 523, column: 3)
!791 = distinct !DILexicalBlock(scope: !792, file: !6, line: 514, column: 39)
!792 = distinct !DILexicalBlock(scope: !793, file: !6, line: 514, column: 2)
!793 = distinct !DILexicalBlock(scope: !778, file: !6, line: 514, column: 2)
!794 = !DILocalVariable(name: "x", arg: 1, scope: !795, file: !500, line: 892, type: !157)
!795 = distinct !DISubprogram(name: "sqrt", linkageName: "_ZL4sqrtd", scope: !500, file: !500, line: 892, type: !210, scopeLine: 893, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !5, retainedNodes: !7)
!796 = !DILocation(line: 892, column: 53, scope: !795, inlinedAt: !797)
!797 = distinct !DILocation(line: 528, column: 8, scope: !786)
!798 = !DILocalVariable(name: "f", arg: 1, scope: !799, file: !500, line: 587, type: !157)
!799 = distinct !DISubprogram(name: "fabs", linkageName: "_ZL4fabsd", scope: !500, file: !500, line: 587, type: !210, scopeLine: 588, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !5, retainedNodes: !7)
!800 = !DILocation(line: 587, column: 53, scope: !799, inlinedAt: !801)
!801 = distinct !DILocation(line: 531, column: 7, scope: !786)
!802 = !DILocation(line: 587, column: 53, scope: !799, inlinedAt: !803)
!803 = distinct !DILocation(line: 531, column: 7, scope: !786)
!804 = !DILocation(line: 587, column: 53, scope: !799, inlinedAt: !805)
!805 = distinct !DILocation(line: 531, column: 7, scope: !786)
!806 = !DILocation(line: 587, column: 53, scope: !799, inlinedAt: !807)
!807 = distinct !DILocation(line: 531, column: 7, scope: !786)
!808 = !DILocalVariable(name: "q_global", arg: 1, scope: !778, file: !6, line: 464, type: !251)
!809 = !DILocation(line: 464, column: 36, scope: !778)
!810 = !DILocalVariable(name: "sx_global", arg: 2, scope: !778, file: !6, line: 465, type: !251)
!811 = !DILocation(line: 465, column: 11, scope: !778)
!812 = !DILocalVariable(name: "sy_global", arg: 3, scope: !778, file: !6, line: 466, type: !251)
!813 = !DILocation(line: 466, column: 11, scope: !778)
!814 = !DILocalVariable(name: "an", arg: 4, scope: !778, file: !6, line: 467, type: !157)
!815 = !DILocation(line: 467, column: 10, scope: !778)
!816 = !DILocalVariable(name: "x_local", scope: !778, file: !6, line: 468, type: !817)
!817 = !DICompositeType(tag: DW_TAG_array_type, baseType: !157, size: 16384, elements: !818)
!818 = !{!819}
!819 = !DISubrange(count: 256)
!820 = !DILocation(line: 468, column: 9, scope: !778)
!821 = !DILocalVariable(name: "q_local", scope: !778, file: !6, line: 469, type: !822)
!822 = !DICompositeType(tag: DW_TAG_array_type, baseType: !157, size: 640, elements: !823)
!823 = !{!824}
!824 = !DISubrange(count: 10)
!825 = !DILocation(line: 469, column: 9, scope: !778)
!826 = !DILocalVariable(name: "sx_local", scope: !778, file: !6, line: 470, type: !157)
!827 = !DILocation(line: 470, column: 9, scope: !778)
!828 = !DILocalVariable(name: "sy_local", scope: !778, file: !6, line: 470, type: !157)
!829 = !DILocation(line: 470, column: 19, scope: !778)
!830 = !DILocalVariable(name: "t1", scope: !778, file: !6, line: 471, type: !157)
!831 = !DILocation(line: 471, column: 9, scope: !778)
!832 = !DILocalVariable(name: "t2", scope: !778, file: !6, line: 471, type: !157)
!833 = !DILocation(line: 471, column: 13, scope: !778)
!834 = !DILocalVariable(name: "t3", scope: !778, file: !6, line: 471, type: !157)
!835 = !DILocation(line: 471, column: 17, scope: !778)
!836 = !DILocalVariable(name: "t4", scope: !778, file: !6, line: 471, type: !157)
!837 = !DILocation(line: 471, column: 21, scope: !778)
!838 = !DILocalVariable(name: "x1", scope: !778, file: !6, line: 471, type: !157)
!839 = !DILocation(line: 471, column: 25, scope: !778)
!840 = !DILocalVariable(name: "x2", scope: !778, file: !6, line: 471, type: !157)
!841 = !DILocation(line: 471, column: 29, scope: !778)
!842 = !DILocalVariable(name: "seed", scope: !778, file: !6, line: 471, type: !157)
!843 = !DILocation(line: 471, column: 33, scope: !778)
!844 = !DILocalVariable(name: "i", scope: !778, file: !6, line: 472, type: !9)
!845 = !DILocation(line: 472, column: 6, scope: !778)
!846 = !DILocalVariable(name: "ii", scope: !778, file: !6, line: 472, type: !9)
!847 = !DILocation(line: 472, column: 9, scope: !778)
!848 = !DILocalVariable(name: "ik", scope: !778, file: !6, line: 472, type: !9)
!849 = !DILocation(line: 472, column: 13, scope: !778)
!850 = !DILocalVariable(name: "kk", scope: !778, file: !6, line: 472, type: !9)
!851 = !DILocation(line: 472, column: 17, scope: !778)
!852 = !DILocalVariable(name: "l", scope: !778, file: !6, line: 472, type: !9)
!853 = !DILocation(line: 472, column: 21, scope: !778)
!854 = !DILocation(line: 474, column: 2, scope: !778)
!855 = !DILocation(line: 474, column: 12, scope: !778)
!856 = !DILocation(line: 475, column: 2, scope: !778)
!857 = !DILocation(line: 475, column: 12, scope: !778)
!858 = !DILocation(line: 476, column: 2, scope: !778)
!859 = !DILocation(line: 476, column: 12, scope: !778)
!860 = !DILocation(line: 477, column: 2, scope: !778)
!861 = !DILocation(line: 477, column: 12, scope: !778)
!862 = !DILocation(line: 478, column: 2, scope: !778)
!863 = !DILocation(line: 478, column: 12, scope: !778)
!864 = !DILocation(line: 479, column: 2, scope: !778)
!865 = !DILocation(line: 479, column: 12, scope: !778)
!866 = !DILocation(line: 480, column: 2, scope: !778)
!867 = !DILocation(line: 480, column: 12, scope: !778)
!868 = !DILocation(line: 481, column: 2, scope: !778)
!869 = !DILocation(line: 481, column: 12, scope: !778)
!870 = !DILocation(line: 482, column: 2, scope: !778)
!871 = !DILocation(line: 482, column: 12, scope: !778)
!872 = !DILocation(line: 483, column: 2, scope: !778)
!873 = !DILocation(line: 483, column: 12, scope: !778)
!874 = !DILocation(line: 484, column: 10, scope: !778)
!875 = !DILocation(line: 485, column: 10, scope: !778)
!876 = !DILocation(line: 64, column: 3, scope: !877, inlinedAt: !912)
!877 = distinct !DISubprogram(name: "__fetch_builtin_x", linkageName: "_ZN25__cuda_builtin_blockIdx_t17__fetch_builtin_xEv", scope: !879, file: !878, line: 64, type: !882, scopeLine: 64, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !5, declaration: !881, retainedNodes: !7)
!878 = !DIFile(filename: "llvm-install-tulip/lib/clang/9.0.0/include/__clang_cuda_builtin_vars.h", directory: "/scratch/ah7226")
!879 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "__cuda_builtin_blockIdx_t", file: !878, line: 63, size: 8, flags: DIFlagTypePassByReference | DIFlagNonTrivial, elements: !880, identifier: "_ZTS25__cuda_builtin_blockIdx_t")
!880 = !{!881, !884, !885, !886, !897, !901, !905, !908}
!881 = !DISubprogram(name: "__fetch_builtin_x", linkageName: "_ZN25__cuda_builtin_blockIdx_t17__fetch_builtin_xEv", scope: !879, file: !878, line: 64, type: !882, scopeLine: 64, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!882 = !DISubroutineType(types: !883)
!883 = !{!369}
!884 = !DISubprogram(name: "__fetch_builtin_y", linkageName: "_ZN25__cuda_builtin_blockIdx_t17__fetch_builtin_yEv", scope: !879, file: !878, line: 65, type: !882, scopeLine: 65, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!885 = !DISubprogram(name: "__fetch_builtin_z", linkageName: "_ZN25__cuda_builtin_blockIdx_t17__fetch_builtin_zEv", scope: !879, file: !878, line: 66, type: !882, scopeLine: 66, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!886 = !DISubprogram(name: "operator uint3", linkageName: "_ZNK25__cuda_builtin_blockIdx_tcv5uint3Ev", scope: !879, file: !878, line: 69, type: !887, scopeLine: 69, flags: DIFlagPrototyped, spFlags: 0)
!887 = !DISubroutineType(types: !888)
!888 = !{!889, !895}
!889 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "uint3", file: !890, line: 190, size: 96, flags: DIFlagTypePassByValue, elements: !891, identifier: "_ZTS5uint3")
!890 = !DIFile(filename: "/usr/local/cuda/include/vector_types.h", directory: "")
!891 = !{!892, !893, !894}
!892 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !889, file: !890, line: 192, baseType: !369, size: 32)
!893 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !889, file: !890, line: 192, baseType: !369, size: 32, offset: 32)
!894 = !DIDerivedType(tag: DW_TAG_member, name: "z", scope: !889, file: !890, line: 192, baseType: !369, size: 32, offset: 64)
!895 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !896, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!896 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !879)
!897 = !DISubprogram(name: "__cuda_builtin_blockIdx_t", scope: !879, file: !878, line: 71, type: !898, scopeLine: 71, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!898 = !DISubroutineType(types: !899)
!899 = !{null, !900}
!900 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !879, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!901 = !DISubprogram(name: "__cuda_builtin_blockIdx_t", scope: !879, file: !878, line: 71, type: !902, scopeLine: 71, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!902 = !DISubroutineType(types: !903)
!903 = !{null, !900, !904}
!904 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !896, size: 64)
!905 = !DISubprogram(name: "operator=", linkageName: "_ZNK25__cuda_builtin_blockIdx_taSERKS_", scope: !879, file: !878, line: 71, type: !906, scopeLine: 71, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!906 = !DISubroutineType(types: !907)
!907 = !{null, !895, !904}
!908 = !DISubprogram(name: "operator&", linkageName: "_ZNK25__cuda_builtin_blockIdx_tadEv", scope: !879, file: !878, line: 71, type: !909, scopeLine: 71, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!909 = !DISubroutineType(types: !910)
!910 = !{!911, !895}
!911 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !879, size: 64)
!912 = distinct !DILocation(line: 487, column: 5, scope: !778)
!913 = !{i32 0, i32 65535}
!914 = !DILocation(line: 75, column: 3, scope: !915, inlinedAt: !957)
!915 = distinct !DISubprogram(name: "__fetch_builtin_x", linkageName: "_ZN25__cuda_builtin_blockDim_t17__fetch_builtin_xEv", scope: !916, file: !878, line: 75, type: !882, scopeLine: 75, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !5, declaration: !918, retainedNodes: !7)
!916 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "__cuda_builtin_blockDim_t", file: !878, line: 74, size: 8, flags: DIFlagTypePassByReference | DIFlagNonTrivial, elements: !917, identifier: "_ZTS25__cuda_builtin_blockDim_t")
!917 = !{!918, !919, !920, !921, !942, !946, !950, !953}
!918 = !DISubprogram(name: "__fetch_builtin_x", linkageName: "_ZN25__cuda_builtin_blockDim_t17__fetch_builtin_xEv", scope: !916, file: !878, line: 75, type: !882, scopeLine: 75, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!919 = !DISubprogram(name: "__fetch_builtin_y", linkageName: "_ZN25__cuda_builtin_blockDim_t17__fetch_builtin_yEv", scope: !916, file: !878, line: 76, type: !882, scopeLine: 76, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!920 = !DISubprogram(name: "__fetch_builtin_z", linkageName: "_ZN25__cuda_builtin_blockDim_t17__fetch_builtin_zEv", scope: !916, file: !878, line: 77, type: !882, scopeLine: 77, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!921 = !DISubprogram(name: "operator dim3", linkageName: "_ZNK25__cuda_builtin_blockDim_tcv4dim3Ev", scope: !916, file: !878, line: 80, type: !922, scopeLine: 80, flags: DIFlagPrototyped, spFlags: 0)
!922 = !DISubroutineType(types: !923)
!923 = !{!924, !940}
!924 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "dim3", file: !890, line: 417, size: 96, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !925, identifier: "_ZTS4dim3")
!925 = !{!926, !927, !928, !929, !933, !937}
!926 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !924, file: !890, line: 419, baseType: !369, size: 32)
!927 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !924, file: !890, line: 419, baseType: !369, size: 32, offset: 32)
!928 = !DIDerivedType(tag: DW_TAG_member, name: "z", scope: !924, file: !890, line: 419, baseType: !369, size: 32, offset: 64)
!929 = !DISubprogram(name: "dim3", scope: !924, file: !890, line: 421, type: !930, scopeLine: 421, flags: DIFlagPrototyped, spFlags: 0)
!930 = !DISubroutineType(types: !931)
!931 = !{null, !932, !369, !369, !369}
!932 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !924, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!933 = !DISubprogram(name: "dim3", scope: !924, file: !890, line: 422, type: !934, scopeLine: 422, flags: DIFlagPrototyped, spFlags: 0)
!934 = !DISubroutineType(types: !935)
!935 = !{null, !932, !936}
!936 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint3", file: !890, line: 383, baseType: !889)
!937 = !DISubprogram(name: "operator uint3", linkageName: "_ZN4dim3cv5uint3Ev", scope: !924, file: !890, line: 423, type: !938, scopeLine: 423, flags: DIFlagPrototyped, spFlags: 0)
!938 = !DISubroutineType(types: !939)
!939 = !{!936, !932}
!940 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !941, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!941 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !916)
!942 = !DISubprogram(name: "__cuda_builtin_blockDim_t", scope: !916, file: !878, line: 82, type: !943, scopeLine: 82, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!943 = !DISubroutineType(types: !944)
!944 = !{null, !945}
!945 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !916, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!946 = !DISubprogram(name: "__cuda_builtin_blockDim_t", scope: !916, file: !878, line: 82, type: !947, scopeLine: 82, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!947 = !DISubroutineType(types: !948)
!948 = !{null, !945, !949}
!949 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !941, size: 64)
!950 = !DISubprogram(name: "operator=", linkageName: "_ZNK25__cuda_builtin_blockDim_taSERKS_", scope: !916, file: !878, line: 82, type: !951, scopeLine: 82, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!951 = !DISubroutineType(types: !952)
!952 = !{null, !940, !949}
!953 = !DISubprogram(name: "operator&", linkageName: "_ZNK25__cuda_builtin_blockDim_tadEv", scope: !916, file: !878, line: 82, type: !954, scopeLine: 82, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!954 = !DISubroutineType(types: !955)
!955 = !{!956, !940}
!956 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !916, size: 64)
!957 = distinct !DILocation(line: 487, column: 16, scope: !778)
!958 = !{i32 1, i32 1025}
!959 = !DILocation(line: 487, column: 15, scope: !778)
!960 = !DILocation(line: 53, column: 3, scope: !961, inlinedAt: !987)
!961 = distinct !DISubprogram(name: "__fetch_builtin_x", linkageName: "_ZN26__cuda_builtin_threadIdx_t17__fetch_builtin_xEv", scope: !962, file: !878, line: 53, type: !882, scopeLine: 53, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !5, declaration: !964, retainedNodes: !7)
!962 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "__cuda_builtin_threadIdx_t", file: !878, line: 52, size: 8, flags: DIFlagTypePassByReference | DIFlagNonTrivial, elements: !963, identifier: "_ZTS26__cuda_builtin_threadIdx_t")
!963 = !{!964, !965, !966, !967, !972, !976, !980, !983}
!964 = !DISubprogram(name: "__fetch_builtin_x", linkageName: "_ZN26__cuda_builtin_threadIdx_t17__fetch_builtin_xEv", scope: !962, file: !878, line: 53, type: !882, scopeLine: 53, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!965 = !DISubprogram(name: "__fetch_builtin_y", linkageName: "_ZN26__cuda_builtin_threadIdx_t17__fetch_builtin_yEv", scope: !962, file: !878, line: 54, type: !882, scopeLine: 54, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!966 = !DISubprogram(name: "__fetch_builtin_z", linkageName: "_ZN26__cuda_builtin_threadIdx_t17__fetch_builtin_zEv", scope: !962, file: !878, line: 55, type: !882, scopeLine: 55, flags: DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!967 = !DISubprogram(name: "operator uint3", linkageName: "_ZNK26__cuda_builtin_threadIdx_tcv5uint3Ev", scope: !962, file: !878, line: 58, type: !968, scopeLine: 58, flags: DIFlagPrototyped, spFlags: 0)
!968 = !DISubroutineType(types: !969)
!969 = !{!889, !970}
!970 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !971, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!971 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !962)
!972 = !DISubprogram(name: "__cuda_builtin_threadIdx_t", scope: !962, file: !878, line: 60, type: !973, scopeLine: 60, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!973 = !DISubroutineType(types: !974)
!974 = !{null, !975}
!975 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !962, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!976 = !DISubprogram(name: "__cuda_builtin_threadIdx_t", scope: !962, file: !878, line: 60, type: !977, scopeLine: 60, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!977 = !DISubroutineType(types: !978)
!978 = !{null, !975, !979}
!979 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !971, size: 64)
!980 = !DISubprogram(name: "operator=", linkageName: "_ZNK26__cuda_builtin_threadIdx_taSERKS_", scope: !962, file: !878, line: 60, type: !981, scopeLine: 60, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!981 = !DISubroutineType(types: !982)
!982 = !{null, !970, !979}
!983 = !DISubprogram(name: "operator&", linkageName: "_ZNK26__cuda_builtin_threadIdx_tadEv", scope: !962, file: !878, line: 60, type: !984, scopeLine: 60, flags: DIFlagPrivate | DIFlagPrototyped, spFlags: 0)
!984 = !DISubroutineType(types: !985)
!985 = !{!986, !970}
!986 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !962, size: 64)
!987 = distinct !DILocation(line: 487, column: 27, scope: !778)
!988 = !{i32 0, i32 1024}
!989 = !DILocation(line: 487, column: 26, scope: !778)
!990 = !DILocation(line: 487, column: 4, scope: !778)
!991 = !DILocation(line: 489, column: 5, scope: !992)
!992 = distinct !DILexicalBlock(scope: !778, file: !6, line: 489, column: 5)
!993 = !DILocation(line: 489, column: 7, scope: !992)
!994 = !DILocation(line: 489, column: 5, scope: !778)
!995 = !DILocation(line: 489, column: 13, scope: !996)
!996 = distinct !DILexicalBlock(scope: !992, file: !6, line: 489, column: 12)
!997 = !DILocation(line: 491, column: 4, scope: !778)
!998 = !DILocation(line: 492, column: 5, scope: !778)
!999 = !DILocation(line: 492, column: 4, scope: !778)
!1000 = !DILocation(line: 495, column: 7, scope: !1001)
!1001 = distinct !DILexicalBlock(scope: !778, file: !6, line: 495, column: 2)
!1002 = !DILocation(line: 495, column: 6, scope: !1001)
!1003 = !DILocation(line: 495, column: 11, scope: !1004)
!1004 = distinct !DILexicalBlock(scope: !1001, file: !6, line: 495, column: 2)
!1005 = !DILocation(line: 495, column: 12, scope: !1004)
!1006 = !DILocation(line: 495, column: 2, scope: !1001)
!1007 = !DILocation(line: 496, column: 6, scope: !1008)
!1008 = distinct !DILexicalBlock(scope: !1004, file: !6, line: 495, column: 23)
!1009 = !DILocation(line: 496, column: 8, scope: !1008)
!1010 = !DILocation(line: 496, column: 5, scope: !1008)
!1011 = !DILocation(line: 497, column: 9, scope: !1012)
!1012 = distinct !DILexicalBlock(scope: !1008, file: !6, line: 497, column: 6)
!1013 = !DILocation(line: 497, column: 8, scope: !1012)
!1014 = !DILocation(line: 497, column: 14, scope: !1012)
!1015 = !DILocation(line: 497, column: 12, scope: !1012)
!1016 = !DILocation(line: 497, column: 6, scope: !1008)
!1017 = !DILocation(line: 497, column: 40, scope: !1018)
!1018 = distinct !DILexicalBlock(scope: !1012, file: !6, line: 497, column: 17)
!1019 = !DILocation(line: 497, column: 21, scope: !1018)
!1020 = !DILocation(line: 497, column: 20, scope: !1018)
!1021 = !DILocation(line: 497, column: 44, scope: !1018)
!1022 = !DILocation(line: 498, column: 6, scope: !1023)
!1023 = distinct !DILexicalBlock(scope: !1008, file: !6, line: 498, column: 6)
!1024 = !DILocation(line: 498, column: 8, scope: !1023)
!1025 = !DILocation(line: 498, column: 6, scope: !1008)
!1026 = !DILocation(line: 498, column: 13, scope: !1027)
!1027 = distinct !DILexicalBlock(scope: !1023, file: !6, line: 498, column: 12)
!1028 = !DILocation(line: 499, column: 25, scope: !1008)
!1029 = !DILocation(line: 499, column: 6, scope: !1008)
!1030 = !DILocation(line: 499, column: 5, scope: !1008)
!1031 = !DILocation(line: 500, column: 6, scope: !1008)
!1032 = !DILocation(line: 500, column: 5, scope: !1008)
!1033 = !DILocation(line: 501, column: 2, scope: !1008)
!1034 = !DILocation(line: 495, column: 20, scope: !1004)
!1035 = !DILocation(line: 495, column: 2, scope: !1004)
!1036 = distinct !{!1036, !1006, !1037}
!1037 = !DILocation(line: 501, column: 2, scope: !1001)
!1038 = !DILocation(line: 513, column: 7, scope: !778)
!1039 = !DILocation(line: 513, column: 6, scope: !778)
!1040 = !DILocation(line: 514, column: 8, scope: !793)
!1041 = !DILocation(line: 514, column: 6, scope: !793)
!1042 = !DILocation(line: 514, column: 12, scope: !792)
!1043 = !DILocation(line: 514, column: 14, scope: !792)
!1044 = !DILocation(line: 514, column: 2, scope: !793)
!1045 = !DILocation(line: 516, column: 44, scope: !791)
!1046 = !DILocation(line: 516, column: 3, scope: !791)
!1047 = !DILocation(line: 523, column: 8, scope: !790)
!1048 = !DILocation(line: 523, column: 7, scope: !790)
!1049 = !DILocation(line: 523, column: 12, scope: !789)
!1050 = !DILocation(line: 523, column: 13, scope: !789)
!1051 = !DILocation(line: 523, column: 3, scope: !790)
!1052 = !DILocation(line: 524, column: 21, scope: !788)
!1053 = !DILocation(line: 524, column: 20, scope: !788)
!1054 = !DILocation(line: 524, column: 11, scope: !788)
!1055 = !DILocation(line: 524, column: 10, scope: !788)
!1056 = !DILocation(line: 524, column: 23, scope: !788)
!1057 = !DILocation(line: 524, column: 6, scope: !788)
!1058 = !DILocation(line: 525, column: 21, scope: !788)
!1059 = !DILocation(line: 525, column: 20, scope: !788)
!1060 = !DILocation(line: 525, column: 22, scope: !788)
!1061 = !DILocation(line: 525, column: 11, scope: !788)
!1062 = !DILocation(line: 525, column: 10, scope: !788)
!1063 = !DILocation(line: 525, column: 25, scope: !788)
!1064 = !DILocation(line: 525, column: 6, scope: !788)
!1065 = !DILocation(line: 526, column: 7, scope: !788)
!1066 = !DILocation(line: 526, column: 10, scope: !788)
!1067 = !DILocation(line: 526, column: 9, scope: !788)
!1068 = !DILocation(line: 526, column: 13, scope: !788)
!1069 = !DILocation(line: 526, column: 16, scope: !788)
!1070 = !DILocation(line: 526, column: 15, scope: !788)
!1071 = !DILocation(line: 526, column: 12, scope: !788)
!1072 = !DILocation(line: 526, column: 6, scope: !788)
!1073 = !DILocation(line: 527, column: 7, scope: !787)
!1074 = !DILocation(line: 527, column: 9, scope: !787)
!1075 = !DILocation(line: 527, column: 7, scope: !788)
!1076 = !DILocation(line: 528, column: 23, scope: !786)
!1077 = !DILocation(line: 227, column: 19, scope: !782, inlinedAt: !785)
!1078 = !DILocation(line: 227, column: 10, scope: !782, inlinedAt: !785)
!1079 = !DILocation(line: 528, column: 17, scope: !786)
!1080 = !DILocation(line: 528, column: 28, scope: !786)
!1081 = !DILocation(line: 528, column: 27, scope: !786)
!1082 = !DILocation(line: 894, column: 20, scope: !795, inlinedAt: !797)
!1083 = !DILocation(line: 894, column: 10, scope: !795, inlinedAt: !797)
!1084 = !DILocation(line: 528, column: 7, scope: !786)
!1085 = !DILocation(line: 529, column: 9, scope: !786)
!1086 = !DILocation(line: 529, column: 12, scope: !786)
!1087 = !DILocation(line: 529, column: 11, scope: !786)
!1088 = !DILocation(line: 529, column: 7, scope: !786)
!1089 = !DILocation(line: 530, column: 9, scope: !786)
!1090 = !DILocation(line: 530, column: 12, scope: !786)
!1091 = !DILocation(line: 530, column: 11, scope: !786)
!1092 = !DILocation(line: 530, column: 7, scope: !786)
!1093 = !DILocation(line: 531, column: 7, scope: !786)
!1094 = !DILocation(line: 589, column: 20, scope: !799, inlinedAt: !801)
!1095 = !DILocation(line: 589, column: 10, scope: !799, inlinedAt: !801)
!1096 = !DILocation(line: 589, column: 20, scope: !799, inlinedAt: !803)
!1097 = !DILocation(line: 589, column: 10, scope: !799, inlinedAt: !803)
!1098 = !DILocation(line: 589, column: 20, scope: !799, inlinedAt: !805)
!1099 = !DILocation(line: 589, column: 10, scope: !799, inlinedAt: !805)
!1100 = !DILocation(line: 589, column: 20, scope: !799, inlinedAt: !807)
!1101 = !DILocation(line: 589, column: 10, scope: !799, inlinedAt: !807)
!1102 = !DILocation(line: 531, column: 6, scope: !786)
!1103 = !DILocation(line: 532, column: 13, scope: !786)
!1104 = !DILocation(line: 532, column: 5, scope: !786)
!1105 = !DILocation(line: 532, column: 15, scope: !786)
!1106 = !DILocation(line: 533, column: 14, scope: !786)
!1107 = !DILocation(line: 533, column: 23, scope: !786)
!1108 = !DILocation(line: 533, column: 22, scope: !786)
!1109 = !DILocation(line: 533, column: 13, scope: !786)
!1110 = !DILocation(line: 534, column: 15, scope: !786)
!1111 = !DILocation(line: 534, column: 13, scope: !786)
!1112 = !DILocation(line: 535, column: 4, scope: !786)
!1113 = !DILocation(line: 536, column: 3, scope: !788)
!1114 = !DILocation(line: 523, column: 30, scope: !789)
!1115 = !DILocation(line: 523, column: 3, scope: !789)
!1116 = distinct !{!1116, !1051, !1117}
!1117 = !DILocation(line: 536, column: 3, scope: !790)
!1118 = !DILocation(line: 537, column: 2, scope: !791)
!1119 = !DILocation(line: 514, column: 22, scope: !792)
!1120 = !DILocation(line: 514, column: 24, scope: !792)
!1121 = !DILocation(line: 514, column: 21, scope: !792)
!1122 = !DILocation(line: 514, column: 2, scope: !792)
!1123 = distinct !{!1123, !1044, !1124}
!1124 = !DILocation(line: 537, column: 2, scope: !793)
!1125 = !DILocation(line: 551, column: 1, scope: !778)
!1126 = distinct !DISubprogram(name: "randlc_device", linkageName: "_Z13randlc_devicePdd", scope: !6, file: !6, line: 553, type: !1127, scopeLine: 554, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !7)
!1127 = !DISubroutineType(types: !1128)
!1128 = !{!157, !251, !157}
!1129 = !DILocalVariable(name: "x", arg: 1, scope: !1126, file: !6, line: 553, type: !251)
!1130 = !DILocation(line: 553, column: 41, scope: !1126)
!1131 = !DILocalVariable(name: "a", arg: 2, scope: !1126, file: !6, line: 554, type: !157)
!1132 = !DILocation(line: 554, column: 10, scope: !1126)
!1133 = !DILocalVariable(name: "t1", scope: !1126, file: !6, line: 555, type: !157)
!1134 = !DILocation(line: 555, column: 9, scope: !1126)
!1135 = !DILocalVariable(name: "t2", scope: !1126, file: !6, line: 555, type: !157)
!1136 = !DILocation(line: 555, column: 12, scope: !1126)
!1137 = !DILocalVariable(name: "t3", scope: !1126, file: !6, line: 555, type: !157)
!1138 = !DILocation(line: 555, column: 15, scope: !1126)
!1139 = !DILocalVariable(name: "t4", scope: !1126, file: !6, line: 555, type: !157)
!1140 = !DILocation(line: 555, column: 18, scope: !1126)
!1141 = !DILocalVariable(name: "a1", scope: !1126, file: !6, line: 555, type: !157)
!1142 = !DILocation(line: 555, column: 21, scope: !1126)
!1143 = !DILocalVariable(name: "a2", scope: !1126, file: !6, line: 555, type: !157)
!1144 = !DILocation(line: 555, column: 24, scope: !1126)
!1145 = !DILocalVariable(name: "x1", scope: !1126, file: !6, line: 555, type: !157)
!1146 = !DILocation(line: 555, column: 27, scope: !1126)
!1147 = !DILocalVariable(name: "x2", scope: !1126, file: !6, line: 555, type: !157)
!1148 = !DILocation(line: 555, column: 30, scope: !1126)
!1149 = !DILocalVariable(name: "z", scope: !1126, file: !6, line: 555, type: !157)
!1150 = !DILocation(line: 555, column: 33, scope: !1126)
!1151 = !DILocation(line: 556, column: 13, scope: !1126)
!1152 = !DILocation(line: 556, column: 11, scope: !1126)
!1153 = !DILocation(line: 556, column: 5, scope: !1126)
!1154 = !DILocation(line: 557, column: 12, scope: !1126)
!1155 = !DILocation(line: 557, column: 7, scope: !1126)
!1156 = !DILocation(line: 557, column: 5, scope: !1126)
!1157 = !DILocation(line: 558, column: 7, scope: !1126)
!1158 = !DILocation(line: 558, column: 17, scope: !1126)
!1159 = !DILocation(line: 558, column: 15, scope: !1126)
!1160 = !DILocation(line: 558, column: 9, scope: !1126)
!1161 = !DILocation(line: 558, column: 5, scope: !1126)
!1162 = !DILocation(line: 559, column: 15, scope: !1126)
!1163 = !DILocation(line: 559, column: 14, scope: !1126)
!1164 = !DILocation(line: 559, column: 11, scope: !1126)
!1165 = !DILocation(line: 559, column: 5, scope: !1126)
!1166 = !DILocation(line: 560, column: 12, scope: !1126)
!1167 = !DILocation(line: 560, column: 7, scope: !1126)
!1168 = !DILocation(line: 560, column: 5, scope: !1126)
!1169 = !DILocation(line: 561, column: 9, scope: !1126)
!1170 = !DILocation(line: 561, column: 8, scope: !1126)
!1171 = !DILocation(line: 561, column: 20, scope: !1126)
!1172 = !DILocation(line: 561, column: 18, scope: !1126)
!1173 = !DILocation(line: 561, column: 12, scope: !1126)
!1174 = !DILocation(line: 561, column: 5, scope: !1126)
!1175 = !DILocation(line: 562, column: 7, scope: !1126)
!1176 = !DILocation(line: 562, column: 12, scope: !1126)
!1177 = !DILocation(line: 562, column: 10, scope: !1126)
!1178 = !DILocation(line: 562, column: 17, scope: !1126)
!1179 = !DILocation(line: 562, column: 22, scope: !1126)
!1180 = !DILocation(line: 562, column: 20, scope: !1126)
!1181 = !DILocation(line: 562, column: 15, scope: !1126)
!1182 = !DILocation(line: 562, column: 5, scope: !1126)
!1183 = !DILocation(line: 563, column: 19, scope: !1126)
!1184 = !DILocation(line: 563, column: 17, scope: !1126)
!1185 = !DILocation(line: 563, column: 12, scope: !1126)
!1186 = !DILocation(line: 563, column: 7, scope: !1126)
!1187 = !DILocation(line: 563, column: 5, scope: !1126)
!1188 = !DILocation(line: 564, column: 6, scope: !1126)
!1189 = !DILocation(line: 564, column: 17, scope: !1126)
!1190 = !DILocation(line: 564, column: 15, scope: !1126)
!1191 = !DILocation(line: 564, column: 9, scope: !1126)
!1192 = !DILocation(line: 564, column: 4, scope: !1126)
!1193 = !DILocation(line: 565, column: 13, scope: !1126)
!1194 = !DILocation(line: 565, column: 11, scope: !1126)
!1195 = !DILocation(line: 565, column: 17, scope: !1126)
!1196 = !DILocation(line: 565, column: 22, scope: !1126)
!1197 = !DILocation(line: 565, column: 20, scope: !1126)
!1198 = !DILocation(line: 565, column: 15, scope: !1126)
!1199 = !DILocation(line: 565, column: 5, scope: !1126)
!1200 = !DILocation(line: 566, column: 19, scope: !1126)
!1201 = !DILocation(line: 566, column: 17, scope: !1126)
!1202 = !DILocation(line: 566, column: 12, scope: !1126)
!1203 = !DILocation(line: 566, column: 7, scope: !1126)
!1204 = !DILocation(line: 566, column: 5, scope: !1126)
!1205 = !DILocation(line: 567, column: 9, scope: !1126)
!1206 = !DILocation(line: 567, column: 20, scope: !1126)
!1207 = !DILocation(line: 567, column: 18, scope: !1126)
!1208 = !DILocation(line: 567, column: 12, scope: !1126)
!1209 = !DILocation(line: 567, column: 4, scope: !1126)
!1210 = !DILocation(line: 567, column: 7, scope: !1126)
!1211 = !DILocation(line: 568, column: 18, scope: !1126)
!1212 = !DILocation(line: 568, column: 17, scope: !1126)
!1213 = !DILocation(line: 568, column: 14, scope: !1126)
!1214 = !DILocation(line: 568, column: 2, scope: !1126)
!1215 = distinct !DISubprogram(name: "vranlc_device", linkageName: "_Z13vranlc_deviceiPddS_", scope: !6, file: !6, line: 647, type: !1216, scopeLine: 650, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !7)
!1216 = !DISubroutineType(types: !1217)
!1217 = !{null, !9, !251, !157, !251}
!1218 = !DILocalVariable(name: "n", arg: 1, scope: !1215, file: !6, line: 647, type: !9)
!1219 = !DILocation(line: 647, column: 35, scope: !1215)
!1220 = !DILocalVariable(name: "x_seed", arg: 2, scope: !1215, file: !6, line: 648, type: !251)
!1221 = !DILocation(line: 648, column: 11, scope: !1215)
!1222 = !DILocalVariable(name: "a", arg: 3, scope: !1215, file: !6, line: 649, type: !157)
!1223 = !DILocation(line: 649, column: 10, scope: !1215)
!1224 = !DILocalVariable(name: "y", arg: 4, scope: !1215, file: !6, line: 650, type: !251)
!1225 = !DILocation(line: 650, column: 11, scope: !1215)
!1226 = !DILocalVariable(name: "i", scope: !1215, file: !6, line: 651, type: !9)
!1227 = !DILocation(line: 651, column: 6, scope: !1215)
!1228 = !DILocalVariable(name: "x", scope: !1215, file: !6, line: 652, type: !157)
!1229 = !DILocation(line: 652, column: 9, scope: !1215)
!1230 = !DILocalVariable(name: "t1", scope: !1215, file: !6, line: 652, type: !157)
!1231 = !DILocation(line: 652, column: 11, scope: !1215)
!1232 = !DILocalVariable(name: "t2", scope: !1215, file: !6, line: 652, type: !157)
!1233 = !DILocation(line: 652, column: 14, scope: !1215)
!1234 = !DILocalVariable(name: "t3", scope: !1215, file: !6, line: 652, type: !157)
!1235 = !DILocation(line: 652, column: 17, scope: !1215)
!1236 = !DILocalVariable(name: "t4", scope: !1215, file: !6, line: 652, type: !157)
!1237 = !DILocation(line: 652, column: 20, scope: !1215)
!1238 = !DILocalVariable(name: "a1", scope: !1215, file: !6, line: 652, type: !157)
!1239 = !DILocation(line: 652, column: 23, scope: !1215)
!1240 = !DILocalVariable(name: "a2", scope: !1215, file: !6, line: 652, type: !157)
!1241 = !DILocation(line: 652, column: 26, scope: !1215)
!1242 = !DILocalVariable(name: "x1", scope: !1215, file: !6, line: 652, type: !157)
!1243 = !DILocation(line: 652, column: 29, scope: !1215)
!1244 = !DILocalVariable(name: "x2", scope: !1215, file: !6, line: 652, type: !157)
!1245 = !DILocation(line: 652, column: 32, scope: !1215)
!1246 = !DILocalVariable(name: "z", scope: !1215, file: !6, line: 652, type: !157)
!1247 = !DILocation(line: 652, column: 35, scope: !1215)
!1248 = !DILocation(line: 653, column: 13, scope: !1215)
!1249 = !DILocation(line: 653, column: 11, scope: !1215)
!1250 = !DILocation(line: 653, column: 5, scope: !1215)
!1251 = !DILocation(line: 654, column: 12, scope: !1215)
!1252 = !DILocation(line: 654, column: 7, scope: !1215)
!1253 = !DILocation(line: 654, column: 5, scope: !1215)
!1254 = !DILocation(line: 655, column: 7, scope: !1215)
!1255 = !DILocation(line: 655, column: 17, scope: !1215)
!1256 = !DILocation(line: 655, column: 15, scope: !1215)
!1257 = !DILocation(line: 655, column: 9, scope: !1215)
!1258 = !DILocation(line: 655, column: 5, scope: !1215)
!1259 = !DILocation(line: 656, column: 7, scope: !1215)
!1260 = !DILocation(line: 656, column: 6, scope: !1215)
!1261 = !DILocation(line: 656, column: 4, scope: !1215)
!1262 = !DILocation(line: 657, column: 7, scope: !1263)
!1263 = distinct !DILexicalBlock(scope: !1215, file: !6, line: 657, column: 2)
!1264 = !DILocation(line: 657, column: 6, scope: !1263)
!1265 = !DILocation(line: 657, column: 11, scope: !1266)
!1266 = distinct !DILexicalBlock(scope: !1263, file: !6, line: 657, column: 2)
!1267 = !DILocation(line: 657, column: 13, scope: !1266)
!1268 = !DILocation(line: 657, column: 12, scope: !1266)
!1269 = !DILocation(line: 657, column: 2, scope: !1263)
!1270 = !DILocation(line: 658, column: 14, scope: !1271)
!1271 = distinct !DILexicalBlock(scope: !1266, file: !6, line: 657, column: 20)
!1272 = !DILocation(line: 658, column: 12, scope: !1271)
!1273 = !DILocation(line: 658, column: 6, scope: !1271)
!1274 = !DILocation(line: 659, column: 13, scope: !1271)
!1275 = !DILocation(line: 659, column: 8, scope: !1271)
!1276 = !DILocation(line: 659, column: 6, scope: !1271)
!1277 = !DILocation(line: 660, column: 8, scope: !1271)
!1278 = !DILocation(line: 660, column: 18, scope: !1271)
!1279 = !DILocation(line: 660, column: 16, scope: !1271)
!1280 = !DILocation(line: 660, column: 10, scope: !1271)
!1281 = !DILocation(line: 660, column: 6, scope: !1271)
!1282 = !DILocation(line: 661, column: 8, scope: !1271)
!1283 = !DILocation(line: 661, column: 13, scope: !1271)
!1284 = !DILocation(line: 661, column: 11, scope: !1271)
!1285 = !DILocation(line: 661, column: 18, scope: !1271)
!1286 = !DILocation(line: 661, column: 23, scope: !1271)
!1287 = !DILocation(line: 661, column: 21, scope: !1271)
!1288 = !DILocation(line: 661, column: 16, scope: !1271)
!1289 = !DILocation(line: 661, column: 6, scope: !1271)
!1290 = !DILocation(line: 662, column: 20, scope: !1271)
!1291 = !DILocation(line: 662, column: 18, scope: !1271)
!1292 = !DILocation(line: 662, column: 13, scope: !1271)
!1293 = !DILocation(line: 662, column: 8, scope: !1271)
!1294 = !DILocation(line: 662, column: 6, scope: !1271)
!1295 = !DILocation(line: 663, column: 7, scope: !1271)
!1296 = !DILocation(line: 663, column: 18, scope: !1271)
!1297 = !DILocation(line: 663, column: 16, scope: !1271)
!1298 = !DILocation(line: 663, column: 10, scope: !1271)
!1299 = !DILocation(line: 663, column: 5, scope: !1271)
!1300 = !DILocation(line: 664, column: 14, scope: !1271)
!1301 = !DILocation(line: 664, column: 12, scope: !1271)
!1302 = !DILocation(line: 664, column: 18, scope: !1271)
!1303 = !DILocation(line: 664, column: 23, scope: !1271)
!1304 = !DILocation(line: 664, column: 21, scope: !1271)
!1305 = !DILocation(line: 664, column: 16, scope: !1271)
!1306 = !DILocation(line: 664, column: 6, scope: !1271)
!1307 = !DILocation(line: 665, column: 20, scope: !1271)
!1308 = !DILocation(line: 665, column: 18, scope: !1271)
!1309 = !DILocation(line: 665, column: 13, scope: !1271)
!1310 = !DILocation(line: 665, column: 8, scope: !1271)
!1311 = !DILocation(line: 665, column: 6, scope: !1271)
!1312 = !DILocation(line: 666, column: 7, scope: !1271)
!1313 = !DILocation(line: 666, column: 18, scope: !1271)
!1314 = !DILocation(line: 666, column: 16, scope: !1271)
!1315 = !DILocation(line: 666, column: 10, scope: !1271)
!1316 = !DILocation(line: 666, column: 5, scope: !1271)
!1317 = !DILocation(line: 667, column: 16, scope: !1271)
!1318 = !DILocation(line: 667, column: 14, scope: !1271)
!1319 = !DILocation(line: 667, column: 3, scope: !1271)
!1320 = !DILocation(line: 667, column: 5, scope: !1271)
!1321 = !DILocation(line: 667, column: 8, scope: !1271)
!1322 = !DILocation(line: 668, column: 2, scope: !1271)
!1323 = !DILocation(line: 657, column: 17, scope: !1266)
!1324 = !DILocation(line: 657, column: 2, scope: !1266)
!1325 = distinct !{!1325, !1269, !1326}
!1326 = !DILocation(line: 668, column: 2, scope: !1263)
!1327 = !DILocation(line: 669, column: 12, scope: !1215)
!1328 = !DILocation(line: 669, column: 3, scope: !1215)
!1329 = !DILocation(line: 669, column: 10, scope: !1215)
!1330 = !DILocation(line: 670, column: 1, scope: !1215)
