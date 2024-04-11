; ------------------------------------------------
; OCL_asm0583a7252521a644_push_analysis.ll
; ------------------------------------------------
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "spir64-unknown-unknown"

; Function Attrs: convergent nounwind
define spir_kernel void @_ZTSZ13warmup_kernelvEUlN4sycl3_V17nd_itemILi3EEEE_(<8 x i32> %r0, <8 x i32> %payloadHeader, <3 x i32> %numWorkGroups, <3 x i32> %localSize, i16 %localIdX, i16 %localIdY, i16 %localIdZ) #0 {
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
!igc.functions = !{!265}
!opencl.ocl.version = !{!276, !276, !276, !276, !276}
!opencl.spir.version = !{!276, !276, !276, !276, !276}
!llvm.ident = !{!277, !277, !277, !277, !277}
!llvm.module.flags = !{!278}

!0 = !{!"ModuleMD", !1, !2, !62, !135, !165, !181, !196, !206, !208, !209, !222, !223, !224, !225, !229, !230, !231, !232, !233, !234, !235, !236, !237, !238, !239, !240, !241, !243, !247, !248, !249, !250, !251, !252, !253, !117, !254, !257, !258, !260, !263, !264}
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
!18 = !{!"UseScratchSpacePrivateMemory", i1 false}
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
!131 = !{!"m_OptsToDisablePerFunc", !132, !133, !134}
!132 = !{!"m_OptsToDisablePerFuncSet[0]", !"IGC-AddressArithmeticSinking"}
!133 = !{!"m_OptsToDisablePerFuncSet[1]", !"IGC-AllowSimd32Slicing"}
!134 = !{!"m_OptsToDisablePerFuncSet[2]", !"IGC-SinkLoadOpt"}
!135 = !{!"pushInfo", !136, !137, !138, !141, !142, !143, !144, !145, !146, !147, !148, !161, !162, !163, !164}
!136 = !{!"pushableAddresses"}
!137 = !{!"bindlessPushInfo"}
!138 = !{!"dynamicBufferInfo", !139, !140}
!139 = !{!"firstIndex", i32 0}
!140 = !{!"numOffsets", i32 0}
!141 = !{!"MaxNumberOfPushedBuffers", i32 0}
!142 = !{!"inlineConstantBufferSlot", i32 -1}
!143 = !{!"inlineConstantBufferOffset", i32 -1}
!144 = !{!"inlineConstantBufferGRFOffset", i32 -1}
!145 = !{!"constants"}
!146 = !{!"inputs"}
!147 = !{!"constantReg"}
!148 = !{!"simplePushInfoArr", !149, !158, !159, !160}
!149 = !{!"simplePushInfoArrVec[0]", !150, !151, !152, !153, !154, !155, !156, !157}
!150 = !{!"cbIdx", i32 0}
!151 = !{!"pushableAddressGrfOffset", i32 -1}
!152 = !{!"pushableOffsetGrfOffset", i32 -1}
!153 = !{!"offset", i32 0}
!154 = !{!"size", i32 0}
!155 = !{!"isStateless", i1 false}
!156 = !{!"isBindless", i1 false}
!157 = !{!"simplePushLoads"}
!158 = !{!"simplePushInfoArrVec[1]", !150, !151, !152, !153, !154, !155, !156, !157}
!159 = !{!"simplePushInfoArrVec[2]", !150, !151, !152, !153, !154, !155, !156, !157}
!160 = !{!"simplePushInfoArrVec[3]", !150, !151, !152, !153, !154, !155, !156, !157}
!161 = !{!"simplePushBufferUsed", i32 0}
!162 = !{!"pushAnalysisWIInfos"}
!163 = !{!"inlineRTGlobalPtrOffset", i32 0}
!164 = !{!"rtSyncSurfPtrOffset", i32 0}
!165 = !{!"psInfo", !166, !167, !168, !169, !170, !171, !172, !173, !174, !175, !176, !177, !178, !179, !180}
!166 = !{!"BlendStateDisabledMask", i8 0}
!167 = !{!"SkipSrc0Alpha", i1 false}
!168 = !{!"DualSourceBlendingDisabled", i1 false}
!169 = !{!"ForceEnableSimd32", i1 false}
!170 = !{!"outputDepth", i1 false}
!171 = !{!"outputStencil", i1 false}
!172 = !{!"outputMask", i1 false}
!173 = !{!"blendToFillEnabled", i1 false}
!174 = !{!"forceEarlyZ", i1 false}
!175 = !{!"hasVersionedLoop", i1 false}
!176 = !{!"forceSingleSourceRTWAfterDualSourceRTW", i1 false}
!177 = !{!"NumSamples", i8 0}
!178 = !{!"blendOptimizationMode"}
!179 = !{!"colorOutputMask"}
!180 = !{!"WaDisableVRS", i1 false}
!181 = !{!"csInfo", !182, !183, !184, !185, !186, !11, !12, !187, !188, !189, !190, !191, !192, !193, !194, !40, !195}
!182 = !{!"maxWorkGroupSize", i32 0}
!183 = !{!"waveSize", i32 0}
!184 = !{!"ComputeShaderSecondCompile"}
!185 = !{!"forcedSIMDSize", i8 0}
!186 = !{!"forceTotalGRFNum", i32 0}
!187 = !{!"allowLowerSimd", i1 false}
!188 = !{!"disableSimd32Slicing", i1 false}
!189 = !{!"disableSplitOnSpill", i1 false}
!190 = !{!"forcedVISAPreRAScheduler", i1 false}
!191 = !{!"disableLocalIdOrderOptimizations", i1 false}
!192 = !{!"disableDispatchAlongY", i1 false}
!193 = !{!"neededThreadIdLayout", i1* null}
!194 = !{!"forceTileYWalk", i1 false}
!195 = !{!"ResForHfPacking"}
!196 = !{!"msInfo", !197, !198, !199, !200, !201, !202, !203, !204, !205}
!197 = !{!"PrimitiveTopology", i32 3}
!198 = !{!"MaxNumOfPrimitives", i32 0}
!199 = !{!"MaxNumOfVertices", i32 0}
!200 = !{!"MaxNumOfPerPrimitiveOutputs", i32 0}
!201 = !{!"MaxNumOfPerVertexOutputs", i32 0}
!202 = !{!"WorkGroupSize", i32 0}
!203 = !{!"WorkGroupMemorySizeInBytes", i32 0}
!204 = !{!"IndexFormat", i32 6}
!205 = !{!"SubgroupSize", i32 0}
!206 = !{!"taskInfo", !207, !202, !203, !205}
!207 = !{!"MaxNumOfOutputs", i32 0}
!208 = !{!"NBarrierCnt", i32 0}
!209 = !{!"rtInfo", !210, !211, !212, !213, !214, !215, !216, !217, !218, !219, !220, !221}
!210 = !{!"RayQueryAllocSizeInBytes", i32 0}
!211 = !{!"NumContinuations", i32 0}
!212 = !{!"RTAsyncStackAddrspace", i32 -1}
!213 = !{!"RTAsyncStackSurfaceStateOffset", i1* null}
!214 = !{!"SWHotZoneAddrspace", i32 -1}
!215 = !{!"SWHotZoneSurfaceStateOffset", i1* null}
!216 = !{!"SWStackAddrspace", i32 -1}
!217 = !{!"SWStackSurfaceStateOffset", i1* null}
!218 = !{!"RTSyncStackAddrspace", i32 -1}
!219 = !{!"RTSyncStackSurfaceStateOffset", i1* null}
!220 = !{!"doSyncDispatchRays", i1 false}
!221 = !{!"MemStyle", !"Xe"}
!222 = !{!"CurUniqueIndirectIdx", i32 0}
!223 = !{!"inlineDynTextures"}
!224 = !{!"inlineResInfoData"}
!225 = !{!"immConstant", !226, !227, !228}
!226 = !{!"data"}
!227 = !{!"sizes"}
!228 = !{!"zeroIdxs"}
!229 = !{!"stringConstants"}
!230 = !{!"inlineConstantBuffers"}
!231 = !{!"inlineGlobalBuffers"}
!232 = !{!"GlobalPointerProgramBinaryInfos"}
!233 = !{!"ConstantPointerProgramBinaryInfos"}
!234 = !{!"GlobalBufferAddressRelocInfo"}
!235 = !{!"ConstantBufferAddressRelocInfo"}
!236 = !{!"forceLscCacheList"}
!237 = !{!"SrvMap"}
!238 = !{!"RasterizerOrderedByteAddressBuffer"}
!239 = !{!"MinNOSPushConstantSize", i32 2}
!240 = !{!"inlineProgramScopeOffsets"}
!241 = !{!"shaderData", !242}
!242 = !{!"numReplicas", i32 0}
!243 = !{!"URBInfo", !244, !245, !246}
!244 = !{!"has64BVertexHeaderInput", i1 false}
!245 = !{!"has64BVertexHeaderOutput", i1 false}
!246 = !{!"hasVertexHeader", i1 true}
!247 = !{!"UseBindlessImage", i1 false}
!248 = !{!"enableRangeReduce", i1 false}
!249 = !{!"allowMatchMadOptimizationforVS", i1 false}
!250 = !{!"disableMemOptforNegativeOffsetLoads", i1 false}
!251 = !{!"enableThreeWayLoadSpiltOpt", i1 false}
!252 = !{!"statefulResourcesNotAliased", i1 false}
!253 = !{!"disableMixMode", i1 false}
!254 = !{!"PrivateMemoryPerFG", !255, !256}
!255 = !{!"PrivateMemoryPerFGMap[0]", void (<8 x i32>, <8 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16)* @_ZTSZ13warmup_kernelvEUlN4sycl3_V17nd_itemILi3EEEE_}
!256 = !{!"PrivateMemoryPerFGValue[0]", i32 0}
!257 = !{!"m_OptsToDisable"}
!258 = !{!"capabilities", !259}
!259 = !{!"globalVariableDecorationsINTEL", i1 false}
!260 = !{!"m_ShaderResourceViewMcsMask", !261, !262}
!261 = !{!"m_ShaderResourceViewMcsMaskVec[0]", i64 0}
!262 = !{!"m_ShaderResourceViewMcsMaskVec[1]", i64 0}
!263 = !{!"computedDepthMode", i32 0}
!264 = !{!"isHDCFastClearShader", i1 false}
!265 = !{void (<8 x i32>, <8 x i32>, <3 x i32>, <3 x i32>, i16, i16, i16)* @_ZTSZ13warmup_kernelvEUlN4sycl3_V17nd_itemILi3EEEE_, !266}
!266 = !{!267, !268}
!267 = !{!"function_type", i32 0}
!268 = !{!"implicit_arg_desc", !269, !270, !271, !272, !273, !274, !275}
!269 = !{i32 0}
!270 = !{i32 1}
!271 = !{i32 3}
!272 = !{i32 5}
!273 = !{i32 7}
!274 = !{i32 8}
!275 = !{i32 9}
!276 = !{i32 2, i32 0}
!277 = !{!"clang version 11.1.0"}
!278 = !{i32 1, !"wchar_size", i32 4}
