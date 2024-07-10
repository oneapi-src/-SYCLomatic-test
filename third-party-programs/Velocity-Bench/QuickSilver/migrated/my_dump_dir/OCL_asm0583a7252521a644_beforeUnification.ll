; ------------------------------------------------
; OCL_asm0583a7252521a644_beforeUnification.ll
; ------------------------------------------------
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"
target triple = "spir64-unknown-unknown"

; Function Attrs: nounwind
define spir_kernel void @_ZTSZ13warmup_kernelvEUlN4sycl3_V17nd_itemILi3EEEE_() #0 {
  call spir_func void @__itt_offload_wi_start_wrapper() #1
  %1 = call spir_func <3 x i64> @__builtin_spirv_BuiltInWorkgroupSize() #5
  %2 = extractelement <3 x i64> %1, i32 2
  %3 = call spir_func <3 x i64> @__builtin_spirv_BuiltInWorkgroupSize() #5
  %4 = extractelement <3 x i64> %3, i32 1
  %5 = call spir_func <3 x i64> @__builtin_spirv_BuiltInWorkgroupSize() #5
  %6 = extractelement <3 x i64> %5, i32 0
  %7 = call spir_func <3 x i64> @__builtin_spirv_BuiltInNumWorkgroups() #5
  %8 = extractelement <3 x i64> %7, i32 1
  %9 = call spir_func <3 x i64> @__builtin_spirv_BuiltInNumWorkgroups() #5
  %10 = extractelement <3 x i64> %9, i32 0
  %11 = call spir_func <3 x i64> @__builtin_spirv_BuiltInWorkgroupId() #5
  %12 = extractelement <3 x i64> %11, i32 2
  %13 = call spir_func <3 x i64> @__builtin_spirv_BuiltInWorkgroupId() #5
  %14 = extractelement <3 x i64> %13, i32 1
  %15 = call spir_func <3 x i64> @__builtin_spirv_BuiltInWorkgroupId() #5
  %16 = extractelement <3 x i64> %15, i32 0
  %17 = call spir_func <3 x i64> @__builtin_spirv_BuiltInLocalInvocationId() #5
  %18 = extractelement <3 x i64> %17, i32 2
  %19 = call spir_func <3 x i64> @__builtin_spirv_BuiltInLocalInvocationId() #5
  %20 = extractelement <3 x i64> %19, i32 1
  %21 = call spir_func <3 x i64> @__builtin_spirv_BuiltInLocalInvocationId() #5
  %22 = extractelement <3 x i64> %21, i32 0
  %23 = icmp ult i64 %16, 2147483648
  %i1promo = zext i1 %23 to i8
  %24 = trunc i8 %i1promo to i1
  call void @llvm.assume(i1 %24)
  %25 = icmp ult i64 %14, 2147483648
  %i1promo1 = zext i1 %25 to i8
  %26 = trunc i8 %i1promo1 to i1
  call void @llvm.assume(i1 %26)
  %27 = icmp ult i64 %10, 2147483648
  %i1promo2 = zext i1 %27 to i8
  %28 = trunc i8 %i1promo2 to i1
  call void @llvm.assume(i1 %28)
  %29 = icmp ult i64 %12, 2147483648
  %i1promo3 = zext i1 %29 to i8
  %30 = trunc i8 %i1promo3 to i1
  call void @llvm.assume(i1 %30)
  %31 = icmp ult i64 %8, 2147483648
  %i1promo4 = zext i1 %31 to i8
  %32 = trunc i8 %i1promo4 to i1
  call void @llvm.assume(i1 %32)
  %33 = icmp ult i64 %6, 2147483648
  %i1promo5 = zext i1 %33 to i8
  %34 = trunc i8 %i1promo5 to i1
  call void @llvm.assume(i1 %34)
  %35 = icmp ult i64 %4, 2147483648
  %i1promo6 = zext i1 %35 to i8
  %36 = trunc i8 %i1promo6 to i1
  call void @llvm.assume(i1 %36)
  %37 = icmp ult i64 %2, 2147483648
  %i1promo7 = zext i1 %37 to i8
  %38 = trunc i8 %i1promo7 to i1
  call void @llvm.assume(i1 %38)
  %39 = icmp ult i64 %18, 2147483648
  %i1promo8 = zext i1 %39 to i8
  %40 = trunc i8 %i1promo8 to i1
  call void @llvm.assume(i1 %40)
  %41 = icmp ult i64 %20, 2147483648
  %i1promo9 = zext i1 %41 to i8
  %42 = trunc i8 %i1promo9 to i1
  call void @llvm.assume(i1 %42)
  %43 = icmp ult i64 %22, 2147483648
  %i1promo10 = zext i1 %43 to i8
  %44 = trunc i8 %i1promo10 to i1
  call void @llvm.assume(i1 %44)
  call spir_func void @__itt_offload_wi_finish_wrapper() #1
  ret void
}

; Function Attrs: alwaysinline nounwind
define spir_func void @__itt_offload_wi_start_wrapper() #1 {
  %1 = alloca [3 x i64], align 8
  %2 = icmp eq i8 0, 0
  br i1 %2, label %25, label %3

3:                                                ; preds = %0
  %4 = bitcast [3 x i64]* %1 to i8*
  %5 = bitcast i8* %4 to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* %5)
  %6 = getelementptr inbounds [3 x i64], [3 x i64]* %1, i64 0, i64 0
  %7 = addrspacecast i64* %6 to i64 addrspace(4)*
  %8 = call spir_func <3 x i64> @__builtin_spirv_BuiltInWorkgroupId() #5
  %9 = extractelement <3 x i64> %8, i32 0
  store i64 %9, i64* %6, align 8
  %10 = getelementptr inbounds [3 x i64], [3 x i64]* %1, i64 0, i64 1
  %11 = extractelement <3 x i64> %8, i32 1
  store i64 %11, i64* %10, align 8
  %12 = getelementptr inbounds [3 x i64], [3 x i64]* %1, i64 0, i64 2
  %13 = extractelement <3 x i64> %8, i32 2
  store i64 %13, i64* %12, align 8
  %14 = call spir_func i64 @__builtin_spirv_BuiltInGlobalLinearId() #5
  %15 = call spir_func <3 x i64> @__builtin_spirv_BuiltInWorkgroupSize() #5
  %16 = extractelement <3 x i64> %15, i32 0
  %17 = call spir_func <3 x i64> @__builtin_spirv_BuiltInWorkgroupSize() #5
  %18 = extractelement <3 x i64> %17, i32 1
  %19 = mul i64 %16, %18
  %20 = call spir_func <3 x i64> @__builtin_spirv_BuiltInWorkgroupSize() #5
  %21 = extractelement <3 x i64> %20, i32 2
  %22 = mul i64 %19, %21
  %23 = trunc i64 %22 to i32
  call spir_func void @__itt_offload_wi_start_stub(i64 addrspace(4)* %7, i64 %14, i32 %23) #3
  %24 = bitcast i8* %4 to i8*
  call void @llvm.lifetime.end.p0i8(i64 24, i8* %24)
  br label %25

25:                                               ; preds = %3, %0
  ret void
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #2

; Function Attrs: noinline nounwind optnone
define spir_func void @__itt_offload_wi_start_stub(i64 addrspace(4)* %0, i64 %1, i32 %2) #3 {
  %4 = alloca i64 addrspace(4)*, align 8
  %5 = alloca i64, align 8
  %6 = alloca i32, align 4
  %7 = addrspacecast i64 addrspace(4)** %4 to i64 addrspace(4)* addrspace(4)*
  %8 = addrspacecast i64* %5 to i64 addrspace(4)*
  %9 = addrspacecast i32* %6 to i32 addrspace(4)*
  store i64 addrspace(4)* %0, i64 addrspace(4)* addrspace(4)* %7, align 8
  store i64 %1, i64 addrspace(4)* %8, align 8
  store i32 %2, i32 addrspace(4)* %9, align 4
  ret void
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #2

; Function Attrs: nounwind willreturn
declare void @llvm.assume(i1) #4

; Function Attrs: alwaysinline nounwind
define spir_func void @__itt_offload_wi_finish_wrapper() #1 {
  %1 = alloca [3 x i64], align 8
  %2 = icmp eq i8 0, 0
  br i1 %2, label %16, label %3

3:                                                ; preds = %0
  %4 = bitcast [3 x i64]* %1 to i8*
  %5 = bitcast i8* %4 to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* %5)
  %6 = getelementptr inbounds [3 x i64], [3 x i64]* %1, i64 0, i64 0
  %7 = addrspacecast i64* %6 to i64 addrspace(4)*
  %8 = call spir_func <3 x i64> @__builtin_spirv_BuiltInWorkgroupId() #5
  %9 = extractelement <3 x i64> %8, i32 0
  store i64 %9, i64* %6, align 8
  %10 = getelementptr inbounds [3 x i64], [3 x i64]* %1, i64 0, i64 1
  %11 = extractelement <3 x i64> %8, i32 1
  store i64 %11, i64* %10, align 8
  %12 = getelementptr inbounds [3 x i64], [3 x i64]* %1, i64 0, i64 2
  %13 = extractelement <3 x i64> %8, i32 2
  store i64 %13, i64* %12, align 8
  %14 = call spir_func i64 @__builtin_spirv_BuiltInGlobalLinearId() #5
  call spir_func void @__itt_offload_wi_finish_stub(i64 addrspace(4)* %7, i64 %14) #3
  %15 = bitcast i8* %4 to i8*
  call void @llvm.lifetime.end.p0i8(i64 24, i8* %15)
  br label %16

16:                                               ; preds = %3, %0
  ret void
}

; Function Attrs: noinline nounwind optnone
define spir_func void @__itt_offload_wi_finish_stub(i64 addrspace(4)* %0, i64 %1) #3 {
  %3 = alloca i64 addrspace(4)*, align 8
  %4 = alloca i64, align 8
  %5 = addrspacecast i64 addrspace(4)** %3 to i64 addrspace(4)* addrspace(4)*
  %6 = addrspacecast i64* %4 to i64 addrspace(4)*
  store i64 addrspace(4)* %0, i64 addrspace(4)* addrspace(4)* %5, align 8
  store i64 %1, i64 addrspace(4)* %6, align 8
  ret void
}

; Function Attrs: nounwind readnone
declare spir_func <3 x i64> @__builtin_spirv_BuiltInNumWorkgroups() #5

; Function Attrs: nounwind readnone
declare spir_func <3 x i64> @__builtin_spirv_BuiltInLocalInvocationId() #5

; Function Attrs: nounwind readnone
declare spir_func <3 x i64> @__builtin_spirv_BuiltInWorkgroupId() #5

; Function Attrs: nounwind readnone
declare spir_func i64 @__builtin_spirv_BuiltInGlobalLinearId() #5

; Function Attrs: nounwind readnone
declare spir_func <3 x i64> @__builtin_spirv_BuiltInWorkgroupSize() #5

attributes #0 = { nounwind }
attributes #1 = { alwaysinline nounwind }
attributes #2 = { argmemonly nounwind willreturn }
attributes #3 = { noinline nounwind optnone }
attributes #4 = { nounwind willreturn }
attributes #5 = { nounwind readnone }

!opencl.kernels = !{!0}
!IGCMetadata = !{!7}
!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!257}
!opencl.ocl.version = !{!257}
!opencl.used.extensions = !{!258}
!opencl.used.optional.core.features = !{!258}
!opencl.compiler.options = !{!258}
!igc.functions = !{}

!0 = !{void ()* @_ZTSZ13warmup_kernelvEUlN4sycl3_V17nd_itemILi3EEEE_, !1, !2, !3, !4, !5, !6}
!1 = !{!"kernel_arg_addr_space"}
!2 = !{!"kernel_arg_access_qual"}
!3 = !{!"kernel_arg_type"}
!4 = !{!"kernel_arg_type_qual"}
!5 = !{!"kernel_arg_base_type"}
!6 = !{!"kernel_arg_name"}
!7 = !{!"ModuleMD", !8, !9, !69, !129, !159, !175, !190, !200, !202, !203, !216, !217, !218, !219, !223, !224, !225, !226, !227, !228, !229, !230, !231, !232, !233, !234, !235, !237, !241, !242, !243, !244, !245, !246, !247, !114, !248, !249, !250, !252, !255, !256}
!8 = !{!"isPrecise", i1 false}
!9 = !{!"compOpt", !10, !11, !12, !13, !14, !15, !16, !17, !18, !19, !20, !21, !22, !23, !24, !25, !26, !27, !28, !29, !30, !31, !32, !33, !34, !35, !36, !37, !38, !39, !40, !41, !42, !43, !44, !45, !46, !47, !48, !49, !50, !51, !52, !53, !54, !55, !56, !57, !58, !59, !60, !61, !62, !63, !64, !65, !66, !67, !68}
!10 = !{!"DenormsAreZero", i1 false}
!11 = !{!"CorrectlyRoundedDivSqrt", i1 false}
!12 = !{!"OptDisable", i1 false}
!13 = !{!"MadEnable", i1 false}
!14 = !{!"NoSignedZeros", i1 false}
!15 = !{!"NoNaNs", i1 false}
!16 = !{!"FloatRoundingMode", i32 0}
!17 = !{!"FloatCvtIntRoundingMode", i32 3}
!18 = !{!"VISAPreSchedRPThreshold", i32 0}
!19 = !{!"SetLoopUnrollThreshold", i32 0}
!20 = !{!"UnsafeMathOptimizations", i1 false}
!21 = !{!"FiniteMathOnly", i1 false}
!22 = !{!"FastRelaxedMath", i1 false}
!23 = !{!"DashGSpecified", i1 false}
!24 = !{!"FastCompilation", i1 false}
!25 = !{!"UseScratchSpacePrivateMemory", i1 true}
!26 = !{!"RelaxedBuiltins", i1 false}
!27 = !{!"SubgroupIndependentForwardProgressRequired", i1 true}
!28 = !{!"GreaterThan2GBBufferRequired", i1 true}
!29 = !{!"GreaterThan4GBBufferRequired", i1 true}
!30 = !{!"DisableA64WA", i1 false}
!31 = !{!"ForceEnableA64WA", i1 false}
!32 = !{!"PushConstantsEnable", i1 true}
!33 = !{!"HasPositivePointerOffset", i1 false}
!34 = !{!"HasBufferOffsetArg", i1 false}
!35 = !{!"BufferOffsetArgOptional", i1 true}
!36 = !{!"HasSubDWAlignedPtrArg", i1 false}
!37 = !{!"replaceGlobalOffsetsByZero", i1 false}
!38 = !{!"forcePixelShaderSIMDMode", i32 0}
!39 = !{!"pixelShaderDoNotAbortOnSpill", i1 false}
!40 = !{!"UniformWGS", i1 false}
!41 = !{!"disableVertexComponentPacking", i1 false}
!42 = !{!"disablePartialVertexComponentPacking", i1 false}
!43 = !{!"PreferBindlessImages", i1 false}
!44 = !{!"UseBindlessMode", i1 false}
!45 = !{!"UseLegacyBindlessMode", i1 true}
!46 = !{!"disableMathRefactoring", i1 false}
!47 = !{!"atomicBranch", i1 false}
!48 = !{!"ForceInt32DivRemEmu", i1 false}
!49 = !{!"ForceInt32DivRemEmuSP", i1 false}
!50 = !{!"DisableFastestSingleCSSIMD", i1 false}
!51 = !{!"DisableFastestLinearScan", i1 false}
!52 = !{!"UseStatelessforPrivateMemory", i1 false}
!53 = !{!"EnableTakeGlobalAddress", i1 false}
!54 = !{!"IsLibraryCompilation", i1 false}
!55 = !{!"FastVISACompile", i1 false}
!56 = !{!"MatchSinCosPi", i1 false}
!57 = !{!"ExcludeIRFromZEBinary", i1 false}
!58 = !{!"EmitZeBinVISASections", i1 false}
!59 = !{!"FP64GenEmulationEnabled", i1 false}
!60 = !{!"allowDisableRematforCS", i1 false}
!61 = !{!"DisableIncSpillCostAllAddrTaken", i1 false}
!62 = !{!"DisableCPSOmaskWA", i1 false}
!63 = !{!"DisableFastestGopt", i1 false}
!64 = !{!"WaForceHalfPromotion", i1 false}
!65 = !{!"DisableConstantCoalescing", i1 false}
!66 = !{!"EnableUndefAlphaOutputAsRed", i1 true}
!67 = !{!"WaEnableALTModeVisaWA", i1 false}
!68 = !{!"NewSpillCostFunction", i1 false}
!69 = !{!"FuncMD", !70, !71}
!70 = !{!"FuncMDMap[0]", void ()* @_ZTSZ13warmup_kernelvEUlN4sycl3_V17nd_itemILi3EEEE_}
!71 = !{!"FuncMDValue[0]", !72, !73, !77, !78, !79, !100, !106, !107, !108, !109, !110, !111, !112, !113, !114, !115, !116, !117, !118, !119, !120, !121, !122, !123, !124, !125, !126, !127, !128}
!72 = !{!"localOffsets"}
!73 = !{!"workGroupWalkOrder", !74, !75, !76}
!74 = !{!"dim0", i32 0}
!75 = !{!"dim1", i32 0}
!76 = !{!"dim2", i32 0}
!77 = !{!"funcArgs"}
!78 = !{!"functionType", !"KernelFunction"}
!79 = !{!"rtInfo", !80, !81, !82, !83, !84, !85, !86, !87, !88, !89, !90, !91, !92, !93, !94, !95, !99}
!80 = !{!"callableShaderType", !"NumberOfCallableShaderTypes"}
!81 = !{!"isContinuation", i1 false}
!82 = !{!"hasTraceRayPayload", i1 false}
!83 = !{!"hasHitAttributes", i1 false}
!84 = !{!"hasCallableData", i1 false}
!85 = !{!"ShaderStackSize", i32 0}
!86 = !{!"ShaderHash", i64 0}
!87 = !{!"ShaderName", !""}
!88 = !{!"ParentName", !""}
!89 = !{!"SlotNum", i1* null}
!90 = !{!"NOSSize", i32 0}
!91 = !{!"globalRootSignatureSize", i32 0}
!92 = !{!"Entries"}
!93 = !{!"SpillUnions"}
!94 = !{!"CustomHitAttrSizeInBytes", i32 0}
!95 = !{!"Types", !96, !97, !98}
!96 = !{!"FrameStartTys"}
!97 = !{!"ArgumentTys"}
!98 = !{!"FullFrameTys"}
!99 = !{!"Aliases"}
!100 = !{!"resAllocMD", !101, !102, !103, !104, !105}
!101 = !{!"uavsNumType", i32 0}
!102 = !{!"srvsNumType", i32 0}
!103 = !{!"samplersNumType", i32 0}
!104 = !{!"argAllocMDList"}
!105 = !{!"inlineSamplersMD"}
!106 = !{!"maxByteOffsets"}
!107 = !{!"IsInitializer", i1 false}
!108 = !{!"IsFinalizer", i1 false}
!109 = !{!"CompiledSubGroupsNumber", i32 0}
!110 = !{!"hasInlineVmeSamplers", i1 false}
!111 = !{!"localSize", i32 0}
!112 = !{!"localIDPresent", i1 false}
!113 = !{!"groupIDPresent", i1 false}
!114 = !{!"privateMemoryPerWI", i32 0}
!115 = !{!"globalIDPresent", i1 false}
!116 = !{!"hasSyncRTCalls", i1 false}
!117 = !{!"hasNonKernelArgLoad", i1 false}
!118 = !{!"hasNonKernelArgStore", i1 false}
!119 = !{!"hasNonKernelArgAtomic", i1 false}
!120 = !{!"UserAnnotations"}
!121 = !{!"m_OpenCLArgAddressSpaces"}
!122 = !{!"m_OpenCLArgAccessQualifiers"}
!123 = !{!"m_OpenCLArgTypes"}
!124 = !{!"m_OpenCLArgBaseTypes"}
!125 = !{!"m_OpenCLArgTypeQualifiers"}
!126 = !{!"m_OpenCLArgNames"}
!127 = !{!"m_OpenCLArgScalarAsPointers"}
!128 = !{!"m_OptsToDisablePerFunc"}
!129 = !{!"pushInfo", !130, !131, !132, !135, !136, !137, !138, !139, !140, !141, !142, !155, !156, !157, !158}
!130 = !{!"pushableAddresses"}
!131 = !{!"bindlessPushInfo"}
!132 = !{!"dynamicBufferInfo", !133, !134}
!133 = !{!"firstIndex", i32 0}
!134 = !{!"numOffsets", i32 0}
!135 = !{!"MaxNumberOfPushedBuffers", i32 0}
!136 = !{!"inlineConstantBufferSlot", i32 -1}
!137 = !{!"inlineConstantBufferOffset", i32 -1}
!138 = !{!"inlineConstantBufferGRFOffset", i32 -1}
!139 = !{!"constants"}
!140 = !{!"inputs"}
!141 = !{!"constantReg"}
!142 = !{!"simplePushInfoArr", !143, !152, !153, !154}
!143 = !{!"simplePushInfoArrVec[0]", !144, !145, !146, !147, !148, !149, !150, !151}
!144 = !{!"cbIdx", i32 0}
!145 = !{!"pushableAddressGrfOffset", i32 -1}
!146 = !{!"pushableOffsetGrfOffset", i32 -1}
!147 = !{!"offset", i32 0}
!148 = !{!"size", i32 0}
!149 = !{!"isStateless", i1 false}
!150 = !{!"isBindless", i1 false}
!151 = !{!"simplePushLoads"}
!152 = !{!"simplePushInfoArrVec[1]", !144, !145, !146, !147, !148, !149, !150, !151}
!153 = !{!"simplePushInfoArrVec[2]", !144, !145, !146, !147, !148, !149, !150, !151}
!154 = !{!"simplePushInfoArrVec[3]", !144, !145, !146, !147, !148, !149, !150, !151}
!155 = !{!"simplePushBufferUsed", i32 0}
!156 = !{!"pushAnalysisWIInfos"}
!157 = !{!"inlineRTGlobalPtrOffset", i32 0}
!158 = !{!"rtSyncSurfPtrOffset", i32 0}
!159 = !{!"psInfo", !160, !161, !162, !163, !164, !165, !166, !167, !168, !169, !170, !171, !172, !173, !174}
!160 = !{!"BlendStateDisabledMask", i8 0}
!161 = !{!"SkipSrc0Alpha", i1 false}
!162 = !{!"DualSourceBlendingDisabled", i1 false}
!163 = !{!"ForceEnableSimd32", i1 false}
!164 = !{!"outputDepth", i1 false}
!165 = !{!"outputStencil", i1 false}
!166 = !{!"outputMask", i1 false}
!167 = !{!"blendToFillEnabled", i1 false}
!168 = !{!"forceEarlyZ", i1 false}
!169 = !{!"hasVersionedLoop", i1 false}
!170 = !{!"forceSingleSourceRTWAfterDualSourceRTW", i1 false}
!171 = !{!"NumSamples", i8 0}
!172 = !{!"blendOptimizationMode"}
!173 = !{!"colorOutputMask"}
!174 = !{!"WaDisableVRS", i1 false}
!175 = !{!"csInfo", !176, !177, !178, !179, !180, !18, !19, !181, !182, !183, !184, !185, !186, !187, !188, !47, !189}
!176 = !{!"maxWorkGroupSize", i32 0}
!177 = !{!"waveSize", i32 0}
!178 = !{!"ComputeShaderSecondCompile"}
!179 = !{!"forcedSIMDSize", i8 0}
!180 = !{!"forceTotalGRFNum", i32 0}
!181 = !{!"allowLowerSimd", i1 false}
!182 = !{!"disableSimd32Slicing", i1 false}
!183 = !{!"disableSplitOnSpill", i1 false}
!184 = !{!"forcedVISAPreRAScheduler", i1 false}
!185 = !{!"disableLocalIdOrderOptimizations", i1 false}
!186 = !{!"disableDispatchAlongY", i1 false}
!187 = !{!"neededThreadIdLayout", i1* null}
!188 = !{!"forceTileYWalk", i1 false}
!189 = !{!"ResForHfPacking"}
!190 = !{!"msInfo", !191, !192, !193, !194, !195, !196, !197, !198, !199}
!191 = !{!"PrimitiveTopology", i32 3}
!192 = !{!"MaxNumOfPrimitives", i32 0}
!193 = !{!"MaxNumOfVertices", i32 0}
!194 = !{!"MaxNumOfPerPrimitiveOutputs", i32 0}
!195 = !{!"MaxNumOfPerVertexOutputs", i32 0}
!196 = !{!"WorkGroupSize", i32 0}
!197 = !{!"WorkGroupMemorySizeInBytes", i32 0}
!198 = !{!"IndexFormat", i32 6}
!199 = !{!"SubgroupSize", i32 0}
!200 = !{!"taskInfo", !201, !196, !197, !199}
!201 = !{!"MaxNumOfOutputs", i32 0}
!202 = !{!"NBarrierCnt", i32 0}
!203 = !{!"rtInfo", !204, !205, !206, !207, !208, !209, !210, !211, !212, !213, !214, !215}
!204 = !{!"RayQueryAllocSizeInBytes", i32 0}
!205 = !{!"NumContinuations", i32 0}
!206 = !{!"RTAsyncStackAddrspace", i32 -1}
!207 = !{!"RTAsyncStackSurfaceStateOffset", i1* null}
!208 = !{!"SWHotZoneAddrspace", i32 -1}
!209 = !{!"SWHotZoneSurfaceStateOffset", i1* null}
!210 = !{!"SWStackAddrspace", i32 -1}
!211 = !{!"SWStackSurfaceStateOffset", i1* null}
!212 = !{!"RTSyncStackAddrspace", i32 -1}
!213 = !{!"RTSyncStackSurfaceStateOffset", i1* null}
!214 = !{!"doSyncDispatchRays", i1 false}
!215 = !{!"MemStyle", !"Xe"}
!216 = !{!"CurUniqueIndirectIdx", i32 0}
!217 = !{!"inlineDynTextures"}
!218 = !{!"inlineResInfoData"}
!219 = !{!"immConstant", !220, !221, !222}
!220 = !{!"data"}
!221 = !{!"sizes"}
!222 = !{!"zeroIdxs"}
!223 = !{!"stringConstants"}
!224 = !{!"inlineConstantBuffers"}
!225 = !{!"inlineGlobalBuffers"}
!226 = !{!"GlobalPointerProgramBinaryInfos"}
!227 = !{!"ConstantPointerProgramBinaryInfos"}
!228 = !{!"GlobalBufferAddressRelocInfo"}
!229 = !{!"ConstantBufferAddressRelocInfo"}
!230 = !{!"forceLscCacheList"}
!231 = !{!"SrvMap"}
!232 = !{!"RasterizerOrderedByteAddressBuffer"}
!233 = !{!"MinNOSPushConstantSize", i32 0}
!234 = !{!"inlineProgramScopeOffsets"}
!235 = !{!"shaderData", !236}
!236 = !{!"numReplicas", i32 0}
!237 = !{!"URBInfo", !238, !239, !240}
!238 = !{!"has64BVertexHeaderInput", i1 false}
!239 = !{!"has64BVertexHeaderOutput", i1 false}
!240 = !{!"hasVertexHeader", i1 true}
!241 = !{!"UseBindlessImage", i1 false}
!242 = !{!"enableRangeReduce", i1 false}
!243 = !{!"allowMatchMadOptimizationforVS", i1 false}
!244 = !{!"disableMemOptforNegativeOffsetLoads", i1 false}
!245 = !{!"enableThreeWayLoadSpiltOpt", i1 false}
!246 = !{!"statefulResourcesNotAliased", i1 false}
!247 = !{!"disableMixMode", i1 false}
!248 = !{!"PrivateMemoryPerFG"}
!249 = !{!"m_OptsToDisable"}
!250 = !{!"capabilities", !251}
!251 = !{!"globalVariableDecorationsINTEL", i1 false}
!252 = !{!"m_ShaderResourceViewMcsMask", !253, !254}
!253 = !{!"m_ShaderResourceViewMcsMaskVec[0]", i64 0}
!254 = !{!"m_ShaderResourceViewMcsMaskVec[1]", i64 0}
!255 = !{!"computedDepthMode", i32 0}
!256 = !{!"isHDCFastClearShader", i1 false}
!257 = !{i32 1, i32 0}
!258 = !{}
