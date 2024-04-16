; ------------------------------------------------
; OCL_asm0583a7252521a644_afterUnification.ll
; ------------------------------------------------
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "spir64-unknown-unknown"

; Function Attrs: convergent nounwind
define spir_kernel void @_ZTSZ13warmup_kernelvEUlN4sycl3_V17nd_itemILi3EEEE_(<8 x i32> %r0, <8 x i32> %payloadHeader, <3 x i32> %numWorkGroups, <3 x i32> %localSize, i16 %localIdX, i16 %localIdY, i16 %localIdZ) #0 {
  %scalar61 = extractelement <8 x i32> %r0, i32 0
  %scalar62 = extractelement <8 x i32> %r0, i32 1
  %scalar63 = extractelement <8 x i32> %r0, i32 2
  %scalar64 = extractelement <8 x i32> %r0, i32 3
  %scalar65 = extractelement <8 x i32> %r0, i32 4
  %scalar66 = extractelement <8 x i32> %r0, i32 5
  %scalar67 = extractelement <8 x i32> %r0, i32 6
  %scalar68 = extractelement <8 x i32> %r0, i32 7
  %scalar58 = extractelement <3 x i32> %numWorkGroups, i32 0
  %scalar59 = extractelement <3 x i32> %numWorkGroups, i32 1
  %scalar60 = extractelement <3 x i32> %numWorkGroups, i32 2
  %scalar = extractelement <3 x i32> %localSize, i32 0
  %scalar56 = extractelement <3 x i32> %localSize, i32 1
  %scalar57 = extractelement <3 x i32> %localSize, i32 2
  %1 = icmp sgt i32 %scalar62, -1
  call void @llvm.assume(i1 %1)
  %2 = icmp sgt i32 %scalar67, -1
  call void @llvm.assume(i1 %2)
  %3 = icmp sgt i32 %scalar58, -1
  call void @llvm.assume(i1 %3)
  %4 = icmp sgt i32 %scalar68, -1
  call void @llvm.assume(i1 %4)
  %5 = icmp sgt i32 %scalar59, -1
  call void @llvm.assume(i1 %5)
  %6 = icmp sgt i32 %scalar, -1
  call void @llvm.assume(i1 %6)
  %7 = icmp sgt i32 %scalar56, -1
  call void @llvm.assume(i1 %7)
  %8 = icmp sgt i32 %scalar57, -1
  call void @llvm.assume(i1 %8)
  ret void
}

; Function Attrs: nounwind willreturn
declare void @llvm.assume(i1) #1

; Function Attrs: convergent nounwind readnone
declare spir_func i32 @__builtin_IB_get_num_groups(i32) local_unnamed_addr #2

; Function Attrs: convergent nounwind readnone
declare spir_func i32 @__builtin_IB_get_local_size(i32) local_unnamed_addr #2

; Function Attrs: convergent nounwind readnone
declare spir_func i32 @__builtin_IB_get_group_id(i32) local_unnamed_addr #2

; Function Attrs: convergent nounwind readnone
declare spir_func i32 @__builtin_IB_get_local_id_x() local_unnamed_addr #2

; Function Attrs: convergent nounwind readnone
declare spir_func i32 @__builtin_IB_get_local_id_y() local_unnamed_addr #2

; Function Attrs: convergent nounwind readnone
declare spir_func i32 @__builtin_IB_get_local_id_z() local_unnamed_addr #2

attributes #0 = { convergent nounwind "less-precise-fpmad"="true" }
attributes #1 = { nounwind willreturn }
attributes #2 = { convergent nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!IGCMetadata = !{!0}
!igc.functions = !{!260}
!opencl.ocl.version = !{!271, !271, !271, !271, !271}
!opencl.spir.version = !{!271, !271, !271, !271, !271}
!llvm.ident = !{!272, !272, !272, !272, !272}
!llvm.module.flags = !{!273}

!0 = !{!"ModuleMD", !1, !2, !62, !132, !162, !178, !193, !203, !205, !206, !219, !220, !221, !222, !226, !227, !228, !229, !230, !231, !232, !233, !234, !235, !236, !237, !238, !240, !244, !245, !246, !247, !248, !249, !250, !117, !251, !252, !253, !255, !258, !259}
!1 = !{!"isPrecise", i1 false}
!2 = !{!"compOpt", !3, !4, !5, !6, !7, !8, !9, !10, !11, !12, !13, !14, !15, !16, !17, !18, !19, !20, !21, !22, !23, !24, !25, !26, !27, !28, !29, !30, !31, !32, !33, !34, !35, !36, !37, !38, !39, !40, !41, !42, !43, !44, !45, !46, !47, !48, !49, !50, !51, !52, !53, !54, !55, !56, !57, !58, !59, !60, !61}
!3 = !{!"DenormsAreZero", i1 false}
!4 = !{!"CorrectlyRoundedDivSqrt", i1 false}
!5 = !{!"OptDisable", i1 false}
!6 = !{!"MadEnable", i1 true}
!7 = !{!"NoSignedZeros", i1 false}
!8 = !{!"NoNaNs", i1 false}
!9 = !{!"FloatRoundingMode", i32 0}
!10 = !{!"FloatCvtIntRoundingMode", i32 3}
!11 = !{!"VISAPreSchedRPThreshold", i32 0}
!12 = !{!"SetLoopUnrollThreshold", i32 0}
!13 = !{!"UnsafeMathOptimizations", i1 false}
!14 = !{!"FiniteMathOnly", i1 false}
!15 = !{!"FastRelaxedMath", i1 false}
!16 = !{!"DashGSpecified", i1 false}
!17 = !{!"FastCompilation", i1 false}
!18 = !{!"UseScratchSpacePrivateMemory", i1 true}
!19 = !{!"RelaxedBuiltins", i1 false}
!20 = !{!"SubgroupIndependentForwardProgressRequired", i1 true}
!21 = !{!"GreaterThan2GBBufferRequired", i1 true}
!22 = !{!"GreaterThan4GBBufferRequired", i1 false}
!23 = !{!"DisableA64WA", i1 false}
!24 = !{!"ForceEnableA64WA", i1 false}
!25 = !{!"PushConstantsEnable", i1 true}
!26 = !{!"HasPositivePointerOffset", i1 false}
!27 = !{!"HasBufferOffsetArg", i1 true}
!28 = !{!"BufferOffsetArgOptional", i1 true}
!29 = !{!"HasSubDWAlignedPtrArg", i1 false}
!30 = !{!"replaceGlobalOffsetsByZero", i1 false}
!31 = !{!"forcePixelShaderSIMDMode", i32 0}
!32 = !{!"pixelShaderDoNotAbortOnSpill", i1 false}
!33 = !{!"UniformWGS", i1 false}
!34 = !{!"disableVertexComponentPacking", i1 false}
!35 = !{!"disablePartialVertexComponentPacking", i1 false}
!36 = !{!"PreferBindlessImages", i1 false}
!37 = !{!"UseBindlessMode", i1 false}
!38 = !{!"UseLegacyBindlessMode", i1 true}
!39 = !{!"disableMathRefactoring", i1 false}
!40 = !{!"atomicBranch", i1 false}
!41 = !{!"ForceInt32DivRemEmu", i1 false}
!42 = !{!"ForceInt32DivRemEmuSP", i1 false}
!43 = !{!"DisableFastestSingleCSSIMD", i1 false}
!44 = !{!"DisableFastestLinearScan", i1 false}
!45 = !{!"UseStatelessforPrivateMemory", i1 false}
!46 = !{!"EnableTakeGlobalAddress", i1 false}
!47 = !{!"IsLibraryCompilation", i1 false}
!48 = !{!"FastVISACompile", i1 false}
!49 = !{!"MatchSinCosPi", i1 false}
!50 = !{!"ExcludeIRFromZEBinary", i1 false}
!51 = !{!"EmitZeBinVISASections", i1 false}
!52 = !{!"FP64GenEmulationEnabled", i1 false}
!53 = !{!"allowDisableRematforCS", i1 false}
!54 = !{!"DisableIncSpillCostAllAddrTaken", i1 false}
!55 = !{!"DisableCPSOmaskWA", i1 false}
!56 = !{!"DisableFastestGopt", i1 false}
!57 = !{!"WaForceHalfPromotion", i1 false}
!58 = !{!"DisableConstantCoalescing", i1 false}
!59 = !{!"EnableUndefAlphaOutputAsRed", i1 true}
!60 = !{!"WaEnableALTModeVisaWA", i1 false}
!61 = !{!"NewSpillCostFunction", i1 false}
!62 = !{!"FuncMD", !63, !64}
!63 = !{!"FuncMDMap[0]", void (<8 x i32>, <8 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16)* @_ZTSZ13warmup_kernelvEUlN4sycl3_V17nd_itemILi3EEEE_}
!64 = !{!"FuncMDValue[0]", !65, !66, !70, !71, !72, !93, !109, !110, !111, !112, !113, !114, !115, !116, !117, !118, !119, !120, !121, !122, !123, !124, !125, !126, !127, !128, !129, !130, !131}
!65 = !{!"localOffsets"}
!66 = !{!"workGroupWalkOrder", !67, !68, !69}
!67 = !{!"dim0", i32 0}
!68 = !{!"dim1", i32 0}
!69 = !{!"dim2", i32 0}
!70 = !{!"funcArgs"}
!71 = !{!"functionType", !"KernelFunction"}
!72 = !{!"rtInfo", !73, !74, !75, !76, !77, !78, !79, !80, !81, !82, !83, !84, !85, !86, !87, !88, !92}
!73 = !{!"callableShaderType", !"NumberOfCallableShaderTypes"}
!74 = !{!"isContinuation", i1 false}
!75 = !{!"hasTraceRayPayload", i1 false}
!76 = !{!"hasHitAttributes", i1 false}
!77 = !{!"hasCallableData", i1 false}
!78 = !{!"ShaderStackSize", i32 0}
!79 = !{!"ShaderHash", i64 0}
!80 = !{!"ShaderName", !""}
!81 = !{!"ParentName", !""}
!82 = !{!"SlotNum", i1* null}
!83 = !{!"NOSSize", i32 0}
!84 = !{!"globalRootSignatureSize", i32 0}
!85 = !{!"Entries"}
!86 = !{!"SpillUnions"}
!87 = !{!"CustomHitAttrSizeInBytes", i32 0}
!88 = !{!"Types", !89, !90, !91}
!89 = !{!"FrameStartTys"}
!90 = !{!"ArgumentTys"}
!91 = !{!"FullFrameTys"}
!92 = !{!"Aliases"}
!93 = !{!"resAllocMD", !94, !95, !96, !97, !108}
!94 = !{!"uavsNumType", i32 0}
!95 = !{!"srvsNumType", i32 0}
!96 = !{!"samplersNumType", i32 0}
!97 = !{!"argAllocMDList", !98, !102, !103, !104, !105, !106, !107}
!98 = !{!"argAllocMDListVec[0]", !99, !100, !101}
!99 = !{!"type", i32 0}
!100 = !{!"extensionType", i32 -1}
!101 = !{!"indexType", i32 -1}
!102 = !{!"argAllocMDListVec[1]", !99, !100, !101}
!103 = !{!"argAllocMDListVec[2]", !99, !100, !101}
!104 = !{!"argAllocMDListVec[3]", !99, !100, !101}
!105 = !{!"argAllocMDListVec[4]", !99, !100, !101}
!106 = !{!"argAllocMDListVec[5]", !99, !100, !101}
!107 = !{!"argAllocMDListVec[6]", !99, !100, !101}
!108 = !{!"inlineSamplersMD"}
!109 = !{!"maxByteOffsets"}
!110 = !{!"IsInitializer", i1 false}
!111 = !{!"IsFinalizer", i1 false}
!112 = !{!"CompiledSubGroupsNumber", i32 0}
!113 = !{!"hasInlineVmeSamplers", i1 false}
!114 = !{!"localSize", i32 0}
!115 = !{!"localIDPresent", i1 false}
!116 = !{!"groupIDPresent", i1 false}
!117 = !{!"privateMemoryPerWI", i32 0}
!118 = !{!"globalIDPresent", i1 false}
!119 = !{!"hasSyncRTCalls", i1 false}
!120 = !{!"hasNonKernelArgLoad", i1 false}
!121 = !{!"hasNonKernelArgStore", i1 false}
!122 = !{!"hasNonKernelArgAtomic", i1 false}
!123 = !{!"UserAnnotations"}
!124 = !{!"m_OpenCLArgAddressSpaces"}
!125 = !{!"m_OpenCLArgAccessQualifiers"}
!126 = !{!"m_OpenCLArgTypes"}
!127 = !{!"m_OpenCLArgBaseTypes"}
!128 = !{!"m_OpenCLArgTypeQualifiers"}
!129 = !{!"m_OpenCLArgNames"}
!130 = !{!"m_OpenCLArgScalarAsPointers"}
!131 = !{!"m_OptsToDisablePerFunc"}
!132 = !{!"pushInfo", !133, !134, !135, !138, !139, !140, !141, !142, !143, !144, !145, !158, !159, !160, !161}
!133 = !{!"pushableAddresses"}
!134 = !{!"bindlessPushInfo"}
!135 = !{!"dynamicBufferInfo", !136, !137}
!136 = !{!"firstIndex", i32 0}
!137 = !{!"numOffsets", i32 0}
!138 = !{!"MaxNumberOfPushedBuffers", i32 0}
!139 = !{!"inlineConstantBufferSlot", i32 -1}
!140 = !{!"inlineConstantBufferOffset", i32 -1}
!141 = !{!"inlineConstantBufferGRFOffset", i32 -1}
!142 = !{!"constants"}
!143 = !{!"inputs"}
!144 = !{!"constantReg"}
!145 = !{!"simplePushInfoArr", !146, !155, !156, !157}
!146 = !{!"simplePushInfoArrVec[0]", !147, !148, !149, !150, !151, !152, !153, !154}
!147 = !{!"cbIdx", i32 0}
!148 = !{!"pushableAddressGrfOffset", i32 -1}
!149 = !{!"pushableOffsetGrfOffset", i32 -1}
!150 = !{!"offset", i32 0}
!151 = !{!"size", i32 0}
!152 = !{!"isStateless", i1 false}
!153 = !{!"isBindless", i1 false}
!154 = !{!"simplePushLoads"}
!155 = !{!"simplePushInfoArrVec[1]", !147, !148, !149, !150, !151, !152, !153, !154}
!156 = !{!"simplePushInfoArrVec[2]", !147, !148, !149, !150, !151, !152, !153, !154}
!157 = !{!"simplePushInfoArrVec[3]", !147, !148, !149, !150, !151, !152, !153, !154}
!158 = !{!"simplePushBufferUsed", i32 0}
!159 = !{!"pushAnalysisWIInfos"}
!160 = !{!"inlineRTGlobalPtrOffset", i32 0}
!161 = !{!"rtSyncSurfPtrOffset", i32 0}
!162 = !{!"psInfo", !163, !164, !165, !166, !167, !168, !169, !170, !171, !172, !173, !174, !175, !176, !177}
!163 = !{!"BlendStateDisabledMask", i8 0}
!164 = !{!"SkipSrc0Alpha", i1 false}
!165 = !{!"DualSourceBlendingDisabled", i1 false}
!166 = !{!"ForceEnableSimd32", i1 false}
!167 = !{!"outputDepth", i1 false}
!168 = !{!"outputStencil", i1 false}
!169 = !{!"outputMask", i1 false}
!170 = !{!"blendToFillEnabled", i1 false}
!171 = !{!"forceEarlyZ", i1 false}
!172 = !{!"hasVersionedLoop", i1 false}
!173 = !{!"forceSingleSourceRTWAfterDualSourceRTW", i1 false}
!174 = !{!"NumSamples", i8 0}
!175 = !{!"blendOptimizationMode"}
!176 = !{!"colorOutputMask"}
!177 = !{!"WaDisableVRS", i1 false}
!178 = !{!"csInfo", !179, !180, !181, !182, !183, !11, !12, !184, !185, !186, !187, !188, !189, !190, !191, !40, !192}
!179 = !{!"maxWorkGroupSize", i32 0}
!180 = !{!"waveSize", i32 0}
!181 = !{!"ComputeShaderSecondCompile"}
!182 = !{!"forcedSIMDSize", i8 0}
!183 = !{!"forceTotalGRFNum", i32 0}
!184 = !{!"allowLowerSimd", i1 false}
!185 = !{!"disableSimd32Slicing", i1 false}
!186 = !{!"disableSplitOnSpill", i1 false}
!187 = !{!"forcedVISAPreRAScheduler", i1 false}
!188 = !{!"disableLocalIdOrderOptimizations", i1 false}
!189 = !{!"disableDispatchAlongY", i1 false}
!190 = !{!"neededThreadIdLayout", i1* null}
!191 = !{!"forceTileYWalk", i1 false}
!192 = !{!"ResForHfPacking"}
!193 = !{!"msInfo", !194, !195, !196, !197, !198, !199, !200, !201, !202}
!194 = !{!"PrimitiveTopology", i32 3}
!195 = !{!"MaxNumOfPrimitives", i32 0}
!196 = !{!"MaxNumOfVertices", i32 0}
!197 = !{!"MaxNumOfPerPrimitiveOutputs", i32 0}
!198 = !{!"MaxNumOfPerVertexOutputs", i32 0}
!199 = !{!"WorkGroupSize", i32 0}
!200 = !{!"WorkGroupMemorySizeInBytes", i32 0}
!201 = !{!"IndexFormat", i32 6}
!202 = !{!"SubgroupSize", i32 0}
!203 = !{!"taskInfo", !204, !199, !200, !202}
!204 = !{!"MaxNumOfOutputs", i32 0}
!205 = !{!"NBarrierCnt", i32 0}
!206 = !{!"rtInfo", !207, !208, !209, !210, !211, !212, !213, !214, !215, !216, !217, !218}
!207 = !{!"RayQueryAllocSizeInBytes", i32 0}
!208 = !{!"NumContinuations", i32 0}
!209 = !{!"RTAsyncStackAddrspace", i32 -1}
!210 = !{!"RTAsyncStackSurfaceStateOffset", i1* null}
!211 = !{!"SWHotZoneAddrspace", i32 -1}
!212 = !{!"SWHotZoneSurfaceStateOffset", i1* null}
!213 = !{!"SWStackAddrspace", i32 -1}
!214 = !{!"SWStackSurfaceStateOffset", i1* null}
!215 = !{!"RTSyncStackAddrspace", i32 -1}
!216 = !{!"RTSyncStackSurfaceStateOffset", i1* null}
!217 = !{!"doSyncDispatchRays", i1 false}
!218 = !{!"MemStyle", !"Xe"}
!219 = !{!"CurUniqueIndirectIdx", i32 0}
!220 = !{!"inlineDynTextures"}
!221 = !{!"inlineResInfoData"}
!222 = !{!"immConstant", !223, !224, !225}
!223 = !{!"data"}
!224 = !{!"sizes"}
!225 = !{!"zeroIdxs"}
!226 = !{!"stringConstants"}
!227 = !{!"inlineConstantBuffers"}
!228 = !{!"inlineGlobalBuffers"}
!229 = !{!"GlobalPointerProgramBinaryInfos"}
!230 = !{!"ConstantPointerProgramBinaryInfos"}
!231 = !{!"GlobalBufferAddressRelocInfo"}
!232 = !{!"ConstantBufferAddressRelocInfo"}
!233 = !{!"forceLscCacheList"}
!234 = !{!"SrvMap"}
!235 = !{!"RasterizerOrderedByteAddressBuffer"}
!236 = !{!"MinNOSPushConstantSize", i32 0}
!237 = !{!"inlineProgramScopeOffsets"}
!238 = !{!"shaderData", !239}
!239 = !{!"numReplicas", i32 0}
!240 = !{!"URBInfo", !241, !242, !243}
!241 = !{!"has64BVertexHeaderInput", i1 false}
!242 = !{!"has64BVertexHeaderOutput", i1 false}
!243 = !{!"hasVertexHeader", i1 true}
!244 = !{!"UseBindlessImage", i1 false}
!245 = !{!"enableRangeReduce", i1 false}
!246 = !{!"allowMatchMadOptimizationforVS", i1 false}
!247 = !{!"disableMemOptforNegativeOffsetLoads", i1 false}
!248 = !{!"enableThreeWayLoadSpiltOpt", i1 false}
!249 = !{!"statefulResourcesNotAliased", i1 false}
!250 = !{!"disableMixMode", i1 false}
!251 = !{!"PrivateMemoryPerFG"}
!252 = !{!"m_OptsToDisable"}
!253 = !{!"capabilities", !254}
!254 = !{!"globalVariableDecorationsINTEL", i1 false}
!255 = !{!"m_ShaderResourceViewMcsMask", !256, !257}
!256 = !{!"m_ShaderResourceViewMcsMaskVec[0]", i64 0}
!257 = !{!"m_ShaderResourceViewMcsMaskVec[1]", i64 0}
!258 = !{!"computedDepthMode", i32 0}
!259 = !{!"isHDCFastClearShader", i1 false}
!260 = !{void (<8 x i32>, <8 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16)* @_ZTSZ13warmup_kernelvEUlN4sycl3_V17nd_itemILi3EEEE_, !261}
!261 = !{!262, !263}
!262 = !{!"function_type", i32 0}
!263 = !{!"implicit_arg_desc", !264, !265, !266, !267, !268, !269, !270}
!264 = !{i32 0}
!265 = !{i32 1}
!266 = !{i32 3}
!267 = !{i32 5}
!268 = !{i32 7}
!269 = !{i32 8}
!270 = !{i32 9}
!271 = !{i32 2, i32 0}
!272 = !{!"clang version 11.1.0"}
!273 = !{i32 1, !"wchar_size", i32 4}
