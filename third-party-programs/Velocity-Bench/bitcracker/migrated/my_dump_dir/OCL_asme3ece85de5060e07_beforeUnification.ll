; ------------------------------------------------
; OCL_asme3ece85de5060e07_beforeUnification.ll
; ------------------------------------------------
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"
target triple = "spir64-unknown-unknown"

%"class.sycl::_V1::nd_item" = type { %"class.sycl::_V1::item", %"class.sycl::_V1::item.13", %"class.sycl::_V1::group" }
%"class.sycl::_V1::item" = type { %"struct.sycl::_V1::detail::ItemBase" }
%"struct.sycl::_V1::detail::ItemBase" = type { %"class.sycl::_V1::range", %"class.sycl::_V1::range", %"class.sycl::_V1::range" }
%"class.sycl::_V1::range" = type { %"class.sycl::_V1::detail::array" }
%"class.sycl::_V1::detail::array" = type { [3 x i64] }
%"class.sycl::_V1::item.13" = type { %"struct.sycl::_V1::detail::ItemBase.14" }
%"struct.sycl::_V1::detail::ItemBase.14" = type { %"class.sycl::_V1::range", %"class.sycl::_V1::range" }
%"class.sycl::_V1::group" = type { %"class.sycl::_V1::range", %"class.sycl::_V1::range", %"class.sycl::_V1::range", %"class.sycl::_V1::range" }

; Function Attrs: nounwind
define spir_kernel void @_ZTSZ16evaluate_w_blockPhPjRdEUlN4sycl3_V17nd_itemILi3EEEE_(i8 addrspace(1)* %0, i8 addrspace(1)* %1, i32 addrspace(1)* %2) #0 {
  call spir_func void @__itt_offload_wi_start_wrapper() #1
  %4 = alloca %"class.sycl::_V1::nd_item", align 8
  %5 = addrspacecast i8 addrspace(1)* %0 to i8 addrspace(4)*
  %6 = addrspacecast i8 addrspace(1)* %1 to i8 addrspace(4)*
  %7 = addrspacecast i32 addrspace(1)* %2 to i32 addrspace(4)*
  %8 = call spir_func <3 x i64> @__builtin_spirv_BuiltInGlobalSize() #5
  %9 = extractelement <3 x i64> %8, i32 2
  %10 = call spir_func <3 x i64> @__builtin_spirv_BuiltInGlobalSize() #5
  %11 = extractelement <3 x i64> %10, i32 1
  %12 = call spir_func <3 x i64> @__builtin_spirv_BuiltInGlobalSize() #5
  %13 = extractelement <3 x i64> %12, i32 0
  %14 = call spir_func <3 x i64> @__builtin_spirv_BuiltInWorkgroupSize() #5
  %15 = extractelement <3 x i64> %14, i32 2
  %16 = call spir_func <3 x i64> @__builtin_spirv_BuiltInWorkgroupSize() #5
  %17 = extractelement <3 x i64> %16, i32 1
  %18 = call spir_func <3 x i64> @__builtin_spirv_BuiltInWorkgroupSize() #5
  %19 = extractelement <3 x i64> %18, i32 0
  %20 = call spir_func <3 x i64> @__builtin_spirv_BuiltInNumWorkgroups() #5
  %21 = extractelement <3 x i64> %20, i32 2
  %22 = call spir_func <3 x i64> @__builtin_spirv_BuiltInNumWorkgroups() #5
  %23 = extractelement <3 x i64> %22, i32 1
  %24 = call spir_func <3 x i64> @__builtin_spirv_BuiltInNumWorkgroups() #5
  %25 = extractelement <3 x i64> %24, i32 0
  %26 = call spir_func <3 x i64> @__builtin_spirv_BuiltInWorkgroupId() #5
  %27 = extractelement <3 x i64> %26, i32 2
  %28 = call spir_func <3 x i64> @__builtin_spirv_BuiltInWorkgroupId() #5
  %29 = extractelement <3 x i64> %28, i32 1
  %30 = call spir_func <3 x i64> @__builtin_spirv_BuiltInWorkgroupId() #5
  %31 = extractelement <3 x i64> %30, i32 0
  %32 = call spir_func <3 x i64> @__builtin_spirv_BuiltInGlobalInvocationId() #5
  %33 = extractelement <3 x i64> %32, i32 2
  %34 = call spir_func <3 x i64> @__builtin_spirv_BuiltInGlobalInvocationId() #5
  %35 = extractelement <3 x i64> %34, i32 1
  %36 = call spir_func <3 x i64> @__builtin_spirv_BuiltInGlobalInvocationId() #5
  %37 = extractelement <3 x i64> %36, i32 0
  %38 = call spir_func <3 x i64> @__builtin_spirv_BuiltInLocalInvocationId() #5
  %39 = extractelement <3 x i64> %38, i32 2
  %40 = call spir_func <3 x i64> @__builtin_spirv_BuiltInLocalInvocationId() #5
  %41 = extractelement <3 x i64> %40, i32 1
  %42 = call spir_func <3 x i64> @__builtin_spirv_BuiltInLocalInvocationId() #5
  %43 = extractelement <3 x i64> %42, i32 0
  %44 = call spir_func <3 x i64> @__builtin_spirv_BuiltInGlobalOffset() #5
  %45 = extractelement <3 x i64> %44, i32 2
  %46 = call spir_func <3 x i64> @__builtin_spirv_BuiltInGlobalOffset() #5
  %47 = extractelement <3 x i64> %46, i32 1
  %48 = call spir_func <3 x i64> @__builtin_spirv_BuiltInGlobalOffset() #5
  %49 = extractelement <3 x i64> %48, i32 0
  %50 = bitcast %"class.sycl::_V1::nd_item"* %4 to i8*
  %51 = bitcast i8* %50 to i8*
  call void @llvm.lifetime.start.p0i8(i64 216, i8* %51)
  %52 = getelementptr inbounds %"class.sycl::_V1::nd_item", %"class.sycl::_V1::nd_item"* %4, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0, i64 0
  store i64 %9, i64* %52, align 8
  %53 = getelementptr inbounds %"class.sycl::_V1::nd_item", %"class.sycl::_V1::nd_item"* %4, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0, i64 1
  store i64 %11, i64* %53, align 8
  %54 = getelementptr inbounds %"class.sycl::_V1::nd_item", %"class.sycl::_V1::nd_item"* %4, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0, i64 2
  store i64 %13, i64* %54, align 8
  %55 = getelementptr inbounds %"class.sycl::_V1::nd_item", %"class.sycl::_V1::nd_item"* %4, i64 0, i32 0, i32 0, i32 1, i32 0, i32 0, i64 0
  store i64 %33, i64* %55, align 8
  %56 = getelementptr inbounds %"class.sycl::_V1::nd_item", %"class.sycl::_V1::nd_item"* %4, i64 0, i32 0, i32 0, i32 1, i32 0, i32 0, i64 1
  store i64 %35, i64* %56, align 8
  %57 = getelementptr inbounds %"class.sycl::_V1::nd_item", %"class.sycl::_V1::nd_item"* %4, i64 0, i32 0, i32 0, i32 1, i32 0, i32 0, i64 2
  store i64 %37, i64* %57, align 8
  %58 = getelementptr inbounds %"class.sycl::_V1::nd_item", %"class.sycl::_V1::nd_item"* %4, i64 0, i32 0, i32 0, i32 2, i32 0, i32 0, i64 0
  store i64 %45, i64* %58, align 8
  %59 = getelementptr inbounds %"class.sycl::_V1::nd_item", %"class.sycl::_V1::nd_item"* %4, i64 0, i32 0, i32 0, i32 2, i32 0, i32 0, i64 1
  store i64 %47, i64* %59, align 8
  %60 = getelementptr inbounds %"class.sycl::_V1::nd_item", %"class.sycl::_V1::nd_item"* %4, i64 0, i32 0, i32 0, i32 2, i32 0, i32 0, i64 2
  store i64 %49, i64* %60, align 8
  %61 = getelementptr inbounds %"class.sycl::_V1::nd_item", %"class.sycl::_V1::nd_item"* %4, i64 0, i32 1, i32 0, i32 0, i32 0, i32 0, i64 0
  store i64 %15, i64* %61, align 8
  %62 = getelementptr inbounds %"class.sycl::_V1::nd_item", %"class.sycl::_V1::nd_item"* %4, i64 0, i32 1, i32 0, i32 0, i32 0, i32 0, i64 1
  store i64 %17, i64* %62, align 8
  %63 = getelementptr inbounds %"class.sycl::_V1::nd_item", %"class.sycl::_V1::nd_item"* %4, i64 0, i32 1, i32 0, i32 0, i32 0, i32 0, i64 2
  store i64 %19, i64* %63, align 8
  %64 = getelementptr inbounds %"class.sycl::_V1::nd_item", %"class.sycl::_V1::nd_item"* %4, i64 0, i32 1, i32 0, i32 1, i32 0, i32 0, i64 0
  store i64 %39, i64* %64, align 8
  %65 = getelementptr inbounds %"class.sycl::_V1::nd_item", %"class.sycl::_V1::nd_item"* %4, i64 0, i32 1, i32 0, i32 1, i32 0, i32 0, i64 1
  store i64 %41, i64* %65, align 8
  %66 = getelementptr inbounds %"class.sycl::_V1::nd_item", %"class.sycl::_V1::nd_item"* %4, i64 0, i32 1, i32 0, i32 1, i32 0, i32 0, i64 2
  store i64 %43, i64* %66, align 8
  %67 = getelementptr inbounds %"class.sycl::_V1::nd_item", %"class.sycl::_V1::nd_item"* %4, i64 0, i32 2, i32 0, i32 0, i32 0, i64 0
  store i64 %9, i64* %67, align 8
  %68 = getelementptr inbounds %"class.sycl::_V1::nd_item", %"class.sycl::_V1::nd_item"* %4, i64 0, i32 2, i32 0, i32 0, i32 0, i64 1
  store i64 %11, i64* %68, align 8
  %69 = getelementptr inbounds %"class.sycl::_V1::nd_item", %"class.sycl::_V1::nd_item"* %4, i64 0, i32 2, i32 0, i32 0, i32 0, i64 2
  store i64 %13, i64* %69, align 8
  %70 = getelementptr inbounds %"class.sycl::_V1::nd_item", %"class.sycl::_V1::nd_item"* %4, i64 0, i32 2, i32 1, i32 0, i32 0, i64 0
  store i64 %15, i64* %70, align 8
  %71 = getelementptr inbounds %"class.sycl::_V1::nd_item", %"class.sycl::_V1::nd_item"* %4, i64 0, i32 2, i32 1, i32 0, i32 0, i64 1
  store i64 %17, i64* %71, align 8
  %72 = getelementptr inbounds %"class.sycl::_V1::nd_item", %"class.sycl::_V1::nd_item"* %4, i64 0, i32 2, i32 1, i32 0, i32 0, i64 2
  store i64 %19, i64* %72, align 8
  %73 = getelementptr inbounds %"class.sycl::_V1::nd_item", %"class.sycl::_V1::nd_item"* %4, i64 0, i32 2, i32 2, i32 0, i32 0, i64 0
  store i64 %21, i64* %73, align 8
  %74 = getelementptr inbounds %"class.sycl::_V1::nd_item", %"class.sycl::_V1::nd_item"* %4, i64 0, i32 2, i32 2, i32 0, i32 0, i64 1
  store i64 %23, i64* %74, align 8
  %75 = getelementptr inbounds %"class.sycl::_V1::nd_item", %"class.sycl::_V1::nd_item"* %4, i64 0, i32 2, i32 2, i32 0, i32 0, i64 2
  store i64 %25, i64* %75, align 8
  %76 = getelementptr inbounds %"class.sycl::_V1::nd_item", %"class.sycl::_V1::nd_item"* %4, i64 0, i32 2, i32 3, i32 0, i32 0, i64 0
  store i64 %27, i64* %76, align 8
  %77 = getelementptr inbounds %"class.sycl::_V1::nd_item", %"class.sycl::_V1::nd_item"* %4, i64 0, i32 2, i32 3, i32 0, i32 0, i64 1
  store i64 %29, i64* %77, align 8
  %78 = getelementptr inbounds %"class.sycl::_V1::nd_item", %"class.sycl::_V1::nd_item"* %4, i64 0, i32 2, i32 3, i32 0, i32 0, i64 2
  store i64 %31, i64* %78, align 8
  %79 = addrspacecast %"class.sycl::_V1::nd_item"* %4 to %"class.sycl::_V1::nd_item" addrspace(4)*
  call spir_func void @_Z14kernel_w_blockPhS_PjRKN4sycl3_V17nd_itemILi3EEE(i8 addrspace(4)* nocapture readonly %5, i8 addrspace(4)* nocapture readonly %6, i32 addrspace(4)* nocapture %7, %"class.sycl::_V1::nd_item" addrspace(4)* nocapture readonly %79) #0
  %80 = bitcast i8* %50 to i8*
  call void @llvm.lifetime.end.p0i8(i64 216, i8* %80)
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

; Function Attrs: nounwind
define spir_func void @_Z14kernel_w_blockPhS_PjRKN4sycl3_V17nd_itemILi3EEE(i8 addrspace(4)* nocapture readonly %0, i8 addrspace(4)* nocapture readonly %1, i32 addrspace(4)* nocapture %2, %"class.sycl::_V1::nd_item" addrspace(4)* nocapture readonly %3) #0 {
  %5 = alloca [64 x i8], align 1
  %6 = getelementptr inbounds %"class.sycl::_V1::nd_item", %"class.sycl::_V1::nd_item" addrspace(4)* %3, i64 0, i32 2, i32 3, i32 0, i32 0, i64 2
  %7 = load i64, i64 addrspace(4)* %6, align 8
  %8 = icmp ult i64 %7, 2147483648
  %i1promo = zext i1 %8 to i8
  %9 = trunc i8 %i1promo to i1
  call void @llvm.assume(i1 %9)
  %10 = getelementptr inbounds %"class.sycl::_V1::nd_item", %"class.sycl::_V1::nd_item" addrspace(4)* %3, i64 0, i32 1, i32 0, i32 0, i32 0, i32 0, i64 2
  %11 = load i64, i64 addrspace(4)* %10, align 8
  %12 = icmp ult i64 %11, 2147483648
  %i1promo1 = zext i1 %12 to i8
  %13 = trunc i8 %i1promo1 to i1
  call void @llvm.assume(i1 %13)
  %14 = mul nuw nsw i64 %7, %11
  %15 = getelementptr inbounds %"class.sycl::_V1::nd_item", %"class.sycl::_V1::nd_item" addrspace(4)* %3, i64 0, i32 1, i32 0, i32 1, i32 0, i32 0, i64 2
  %16 = load i64, i64 addrspace(4)* %15, align 8
  %17 = icmp ult i64 %16, 2147483648
  %i1promo2 = zext i1 %17 to i8
  %18 = trunc i8 %i1promo2 to i1
  call void @llvm.assume(i1 %18)
  %19 = add nuw nsw i64 %14, %16
  %20 = and i64 %19, 4293918720
  %21 = icmp eq i64 %20, 0
  br i1 %21, label %22, label %1072

22:                                               ; preds = %4
  %23 = getelementptr inbounds [64 x i8], [64 x i8]* %5, i64 0, i64 0
  %24 = bitcast i8* %23 to i8*
  call void @llvm.lifetime.start.p0i8(i64 64, i8* %24)
  br label %25

25:                                               ; preds = %28, %22
  %26 = phi i32 [ 0, %22 ], [ %33, %28 ]
  %27 = icmp ult i32 %26, 16
  br i1 %27, label %28, label %.preheader

.preheader:                                       ; preds = %25
  br label %34

28:                                               ; preds = %25
  %29 = zext i32 %26 to i64
  %30 = getelementptr inbounds i8, i8 addrspace(4)* %0, i64 %29
  %31 = load i8, i8 addrspace(4)* %30, align 1
  %32 = getelementptr inbounds [64 x i8], [64 x i8]* %5, i64 0, i64 %29
  store i8 %31, i8* %32, align 1
  %33 = add nuw nsw i32 %26, 1
  br label %25

34:                                               ; preds = %37, %.preheader
  %35 = phi i32 [ %44, %37 ], [ 0, %.preheader ]
  %36 = icmp ult i32 %35, 40
  br i1 %36, label %37, label %45

37:                                               ; preds = %34
  %38 = zext i32 %35 to i64
  %39 = getelementptr inbounds i8, i8 addrspace(4)* %1, i64 %38
  %40 = load i8, i8 addrspace(4)* %39, align 1
  %41 = add nuw nsw i32 %35, 24
  %42 = zext i32 %41 to i64
  %43 = getelementptr inbounds [64 x i8], [64 x i8]* %5, i64 0, i64 %42
  store i8 %40, i8* %43, align 1
  %44 = add nuw nsw i32 %35, 1
  br label %34

45:                                               ; preds = %34
  %46 = getelementptr inbounds [64 x i8], [64 x i8]* %5, i64 0, i64 20
  %47 = getelementptr inbounds [64 x i8], [64 x i8]* %5, i64 0, i64 21
  %48 = getelementptr inbounds [64 x i8], [64 x i8]* %5, i64 0, i64 22
  %49 = getelementptr inbounds [64 x i8], [64 x i8]* %5, i64 0, i64 23
  %50 = shl i64 %19, 6
  %51 = and i64 %50, 4294967232
  %52 = load i8, i8* %23, align 1
  %53 = zext i8 %52 to i32
  %54 = shl nuw i32 %53, 24
  %55 = getelementptr inbounds [64 x i8], [64 x i8]* %5, i64 0, i64 1
  %56 = load i8, i8* %55, align 1
  %57 = zext i8 %56 to i32
  %58 = shl nuw nsw i32 %57, 16
  %59 = or i32 %54, %58
  %60 = getelementptr inbounds [64 x i8], [64 x i8]* %5, i64 0, i64 2
  %61 = load i8, i8* %60, align 1
  %62 = zext i8 %61 to i32
  %63 = shl nuw nsw i32 %62, 8
  %64 = or i32 %59, %63
  %65 = getelementptr inbounds [64 x i8], [64 x i8]* %5, i64 0, i64 3
  %66 = load i8, i8* %65, align 1
  %67 = zext i8 %66 to i32
  %68 = or i32 %64, %67
  %69 = getelementptr inbounds i32, i32 addrspace(4)* %2, i64 %51
  store i32 %68, i32 addrspace(4)* %69, align 4
  %70 = getelementptr inbounds [64 x i8], [64 x i8]* %5, i64 0, i64 4
  %71 = load i8, i8* %70, align 1
  %72 = zext i8 %71 to i32
  %73 = shl nuw i32 %72, 24
  %74 = getelementptr inbounds [64 x i8], [64 x i8]* %5, i64 0, i64 5
  %75 = load i8, i8* %74, align 1
  %76 = zext i8 %75 to i32
  %77 = shl nuw nsw i32 %76, 16
  %78 = or i32 %73, %77
  %79 = getelementptr inbounds [64 x i8], [64 x i8]* %5, i64 0, i64 6
  %80 = load i8, i8* %79, align 1
  %81 = zext i8 %80 to i32
  %82 = shl nuw nsw i32 %81, 8
  %83 = or i32 %78, %82
  %84 = getelementptr inbounds [64 x i8], [64 x i8]* %5, i64 0, i64 7
  %85 = load i8, i8* %84, align 1
  %86 = zext i8 %85 to i32
  %87 = or i32 %83, %86
  %88 = or i64 %51, 1
  %89 = getelementptr inbounds i32, i32 addrspace(4)* %2, i64 %88
  store i32 %87, i32 addrspace(4)* %89, align 4
  %90 = getelementptr inbounds [64 x i8], [64 x i8]* %5, i64 0, i64 8
  %91 = load i8, i8* %90, align 1
  %92 = zext i8 %91 to i32
  %93 = shl nuw i32 %92, 24
  %94 = getelementptr inbounds [64 x i8], [64 x i8]* %5, i64 0, i64 9
  %95 = load i8, i8* %94, align 1
  %96 = zext i8 %95 to i32
  %97 = shl nuw nsw i32 %96, 16
  %98 = or i32 %93, %97
  %99 = getelementptr inbounds [64 x i8], [64 x i8]* %5, i64 0, i64 10
  %100 = load i8, i8* %99, align 1
  %101 = zext i8 %100 to i32
  %102 = shl nuw nsw i32 %101, 8
  %103 = or i32 %98, %102
  %104 = getelementptr inbounds [64 x i8], [64 x i8]* %5, i64 0, i64 11
  %105 = load i8, i8* %104, align 1
  %106 = zext i8 %105 to i32
  %107 = or i32 %103, %106
  %108 = or i64 %51, 2
  %109 = getelementptr inbounds i32, i32 addrspace(4)* %2, i64 %108
  store i32 %107, i32 addrspace(4)* %109, align 4
  %110 = getelementptr inbounds [64 x i8], [64 x i8]* %5, i64 0, i64 12
  %111 = load i8, i8* %110, align 1
  %112 = zext i8 %111 to i32
  %113 = shl nuw i32 %112, 24
  %114 = getelementptr inbounds [64 x i8], [64 x i8]* %5, i64 0, i64 13
  %115 = load i8, i8* %114, align 1
  %116 = zext i8 %115 to i32
  %117 = shl nuw nsw i32 %116, 16
  %118 = or i32 %113, %117
  %119 = getelementptr inbounds [64 x i8], [64 x i8]* %5, i64 0, i64 14
  %120 = load i8, i8* %119, align 1
  %121 = zext i8 %120 to i32
  %122 = shl nuw nsw i32 %121, 8
  %123 = or i32 %118, %122
  %124 = getelementptr inbounds [64 x i8], [64 x i8]* %5, i64 0, i64 15
  %125 = load i8, i8* %124, align 1
  %126 = zext i8 %125 to i32
  %127 = or i32 %123, %126
  %128 = or i64 %51, 3
  %129 = getelementptr inbounds i32, i32 addrspace(4)* %2, i64 %128
  store i32 %127, i32 addrspace(4)* %129, align 4
  %130 = trunc i64 %19 to i32
  %131 = and i32 %130, 16777215
  %132 = call spir_func i32 @spirv.llvm_bswap_i32(i32 %131) #0
  %133 = or i64 %51, 4
  %134 = getelementptr inbounds i32, i32 addrspace(4)* %2, i64 %133
  store i32 %132, i32 addrspace(4)* %134, align 4
  %135 = load i8, i8* %46, align 1
  %136 = zext i8 %135 to i32
  %137 = shl nuw i32 %136, 24
  %138 = load i8, i8* %47, align 1
  %139 = zext i8 %138 to i32
  %140 = shl nuw nsw i32 %139, 16
  %141 = or i32 %137, %140
  %142 = load i8, i8* %48, align 1
  %143 = zext i8 %142 to i32
  %144 = shl nuw nsw i32 %143, 8
  %145 = or i32 %141, %144
  %146 = load i8, i8* %49, align 1
  %147 = zext i8 %146 to i32
  %148 = or i32 %145, %147
  %149 = or i64 %51, 5
  %150 = getelementptr inbounds i32, i32 addrspace(4)* %2, i64 %149
  store i32 %148, i32 addrspace(4)* %150, align 4
  %151 = getelementptr inbounds [64 x i8], [64 x i8]* %5, i64 0, i64 24
  %152 = load i8, i8* %151, align 1
  %153 = zext i8 %152 to i32
  %154 = shl nuw i32 %153, 24
  %155 = getelementptr inbounds [64 x i8], [64 x i8]* %5, i64 0, i64 25
  %156 = load i8, i8* %155, align 1
  %157 = zext i8 %156 to i32
  %158 = shl nuw nsw i32 %157, 16
  %159 = or i32 %154, %158
  %160 = getelementptr inbounds [64 x i8], [64 x i8]* %5, i64 0, i64 26
  %161 = load i8, i8* %160, align 1
  %162 = zext i8 %161 to i32
  %163 = shl nuw nsw i32 %162, 8
  %164 = or i32 %159, %163
  %165 = getelementptr inbounds [64 x i8], [64 x i8]* %5, i64 0, i64 27
  %166 = load i8, i8* %165, align 1
  %167 = zext i8 %166 to i32
  %168 = or i32 %164, %167
  %169 = or i64 %51, 6
  %170 = getelementptr inbounds i32, i32 addrspace(4)* %2, i64 %169
  store i32 %168, i32 addrspace(4)* %170, align 4
  %171 = getelementptr inbounds [64 x i8], [64 x i8]* %5, i64 0, i64 28
  %172 = load i8, i8* %171, align 1
  %173 = zext i8 %172 to i32
  %174 = shl nuw i32 %173, 24
  %175 = getelementptr inbounds [64 x i8], [64 x i8]* %5, i64 0, i64 29
  %176 = load i8, i8* %175, align 1
  %177 = zext i8 %176 to i32
  %178 = shl nuw nsw i32 %177, 16
  %179 = or i32 %174, %178
  %180 = getelementptr inbounds [64 x i8], [64 x i8]* %5, i64 0, i64 30
  %181 = load i8, i8* %180, align 1
  %182 = zext i8 %181 to i32
  %183 = shl nuw nsw i32 %182, 8
  %184 = or i32 %179, %183
  %185 = getelementptr inbounds [64 x i8], [64 x i8]* %5, i64 0, i64 31
  %186 = load i8, i8* %185, align 1
  %187 = zext i8 %186 to i32
  %188 = or i32 %184, %187
  %189 = or i64 %51, 7
  %190 = getelementptr inbounds i32, i32 addrspace(4)* %2, i64 %189
  store i32 %188, i32 addrspace(4)* %190, align 4
  %191 = getelementptr inbounds [64 x i8], [64 x i8]* %5, i64 0, i64 32
  %192 = load i8, i8* %191, align 1
  %193 = zext i8 %192 to i32
  %194 = shl nuw i32 %193, 24
  %195 = getelementptr inbounds [64 x i8], [64 x i8]* %5, i64 0, i64 33
  %196 = load i8, i8* %195, align 1
  %197 = zext i8 %196 to i32
  %198 = shl nuw nsw i32 %197, 16
  %199 = or i32 %194, %198
  %200 = getelementptr inbounds [64 x i8], [64 x i8]* %5, i64 0, i64 34
  %201 = load i8, i8* %200, align 1
  %202 = zext i8 %201 to i32
  %203 = shl nuw nsw i32 %202, 8
  %204 = or i32 %199, %203
  %205 = getelementptr inbounds [64 x i8], [64 x i8]* %5, i64 0, i64 35
  %206 = load i8, i8* %205, align 1
  %207 = zext i8 %206 to i32
  %208 = or i32 %204, %207
  %209 = or i64 %51, 8
  %210 = getelementptr inbounds i32, i32 addrspace(4)* %2, i64 %209
  store i32 %208, i32 addrspace(4)* %210, align 4
  %211 = getelementptr inbounds [64 x i8], [64 x i8]* %5, i64 0, i64 36
  %212 = load i8, i8* %211, align 1
  %213 = zext i8 %212 to i32
  %214 = shl nuw i32 %213, 24
  %215 = getelementptr inbounds [64 x i8], [64 x i8]* %5, i64 0, i64 37
  %216 = load i8, i8* %215, align 1
  %217 = zext i8 %216 to i32
  %218 = shl nuw nsw i32 %217, 16
  %219 = or i32 %214, %218
  %220 = getelementptr inbounds [64 x i8], [64 x i8]* %5, i64 0, i64 38
  %221 = load i8, i8* %220, align 1
  %222 = zext i8 %221 to i32
  %223 = shl nuw nsw i32 %222, 8
  %224 = or i32 %219, %223
  %225 = getelementptr inbounds [64 x i8], [64 x i8]* %5, i64 0, i64 39
  %226 = load i8, i8* %225, align 1
  %227 = zext i8 %226 to i32
  %228 = or i32 %224, %227
  %229 = or i64 %51, 9
  %230 = getelementptr inbounds i32, i32 addrspace(4)* %2, i64 %229
  store i32 %228, i32 addrspace(4)* %230, align 4
  %231 = getelementptr inbounds [64 x i8], [64 x i8]* %5, i64 0, i64 40
  %232 = load i8, i8* %231, align 1
  %233 = zext i8 %232 to i32
  %234 = shl nuw i32 %233, 24
  %235 = getelementptr inbounds [64 x i8], [64 x i8]* %5, i64 0, i64 41
  %236 = load i8, i8* %235, align 1
  %237 = zext i8 %236 to i32
  %238 = shl nuw nsw i32 %237, 16
  %239 = or i32 %234, %238
  %240 = getelementptr inbounds [64 x i8], [64 x i8]* %5, i64 0, i64 42
  %241 = load i8, i8* %240, align 1
  %242 = zext i8 %241 to i32
  %243 = shl nuw nsw i32 %242, 8
  %244 = or i32 %239, %243
  %245 = getelementptr inbounds [64 x i8], [64 x i8]* %5, i64 0, i64 43
  %246 = load i8, i8* %245, align 1
  %247 = zext i8 %246 to i32
  %248 = or i32 %244, %247
  %249 = or i64 %51, 10
  %250 = getelementptr inbounds i32, i32 addrspace(4)* %2, i64 %249
  store i32 %248, i32 addrspace(4)* %250, align 4
  %251 = getelementptr inbounds [64 x i8], [64 x i8]* %5, i64 0, i64 44
  %252 = load i8, i8* %251, align 1
  %253 = zext i8 %252 to i32
  %254 = shl nuw i32 %253, 24
  %255 = getelementptr inbounds [64 x i8], [64 x i8]* %5, i64 0, i64 45
  %256 = load i8, i8* %255, align 1
  %257 = zext i8 %256 to i32
  %258 = shl nuw nsw i32 %257, 16
  %259 = or i32 %254, %258
  %260 = getelementptr inbounds [64 x i8], [64 x i8]* %5, i64 0, i64 46
  %261 = load i8, i8* %260, align 1
  %262 = zext i8 %261 to i32
  %263 = shl nuw nsw i32 %262, 8
  %264 = or i32 %259, %263
  %265 = getelementptr inbounds [64 x i8], [64 x i8]* %5, i64 0, i64 47
  %266 = load i8, i8* %265, align 1
  %267 = zext i8 %266 to i32
  %268 = or i32 %264, %267
  %269 = or i64 %51, 11
  %270 = getelementptr inbounds i32, i32 addrspace(4)* %2, i64 %269
  store i32 %268, i32 addrspace(4)* %270, align 4
  %271 = getelementptr inbounds [64 x i8], [64 x i8]* %5, i64 0, i64 48
  %272 = load i8, i8* %271, align 1
  %273 = zext i8 %272 to i32
  %274 = shl nuw i32 %273, 24
  %275 = getelementptr inbounds [64 x i8], [64 x i8]* %5, i64 0, i64 49
  %276 = load i8, i8* %275, align 1
  %277 = zext i8 %276 to i32
  %278 = shl nuw nsw i32 %277, 16
  %279 = or i32 %274, %278
  %280 = getelementptr inbounds [64 x i8], [64 x i8]* %5, i64 0, i64 50
  %281 = load i8, i8* %280, align 1
  %282 = zext i8 %281 to i32
  %283 = shl nuw nsw i32 %282, 8
  %284 = or i32 %279, %283
  %285 = getelementptr inbounds [64 x i8], [64 x i8]* %5, i64 0, i64 51
  %286 = load i8, i8* %285, align 1
  %287 = zext i8 %286 to i32
  %288 = or i32 %284, %287
  %289 = or i64 %51, 12
  %290 = getelementptr inbounds i32, i32 addrspace(4)* %2, i64 %289
  store i32 %288, i32 addrspace(4)* %290, align 4
  %291 = getelementptr inbounds [64 x i8], [64 x i8]* %5, i64 0, i64 52
  %292 = load i8, i8* %291, align 1
  %293 = zext i8 %292 to i32
  %294 = shl nuw i32 %293, 24
  %295 = getelementptr inbounds [64 x i8], [64 x i8]* %5, i64 0, i64 53
  %296 = load i8, i8* %295, align 1
  %297 = zext i8 %296 to i32
  %298 = shl nuw nsw i32 %297, 16
  %299 = or i32 %294, %298
  %300 = getelementptr inbounds [64 x i8], [64 x i8]* %5, i64 0, i64 54
  %301 = load i8, i8* %300, align 1
  %302 = zext i8 %301 to i32
  %303 = shl nuw nsw i32 %302, 8
  %304 = or i32 %299, %303
  %305 = getelementptr inbounds [64 x i8], [64 x i8]* %5, i64 0, i64 55
  %306 = load i8, i8* %305, align 1
  %307 = zext i8 %306 to i32
  %308 = or i32 %304, %307
  %309 = or i64 %51, 13
  %310 = getelementptr inbounds i32, i32 addrspace(4)* %2, i64 %309
  store i32 %308, i32 addrspace(4)* %310, align 4
  %311 = getelementptr inbounds [64 x i8], [64 x i8]* %5, i64 0, i64 56
  %312 = load i8, i8* %311, align 1
  %313 = zext i8 %312 to i32
  %314 = shl nuw i32 %313, 24
  %315 = getelementptr inbounds [64 x i8], [64 x i8]* %5, i64 0, i64 57
  %316 = load i8, i8* %315, align 1
  %317 = zext i8 %316 to i32
  %318 = shl nuw nsw i32 %317, 16
  %319 = or i32 %314, %318
  %320 = getelementptr inbounds [64 x i8], [64 x i8]* %5, i64 0, i64 58
  %321 = load i8, i8* %320, align 1
  %322 = zext i8 %321 to i32
  %323 = shl nuw nsw i32 %322, 8
  %324 = or i32 %319, %323
  %325 = getelementptr inbounds [64 x i8], [64 x i8]* %5, i64 0, i64 59
  %326 = load i8, i8* %325, align 1
  %327 = zext i8 %326 to i32
  %328 = or i32 %324, %327
  %329 = or i64 %51, 14
  %330 = getelementptr inbounds i32, i32 addrspace(4)* %2, i64 %329
  store i32 %328, i32 addrspace(4)* %330, align 4
  %331 = getelementptr inbounds [64 x i8], [64 x i8]* %5, i64 0, i64 60
  %332 = load i8, i8* %331, align 1
  %333 = zext i8 %332 to i32
  %334 = shl nuw i32 %333, 24
  %335 = getelementptr inbounds [64 x i8], [64 x i8]* %5, i64 0, i64 61
  %336 = load i8, i8* %335, align 1
  %337 = zext i8 %336 to i32
  %338 = shl nuw nsw i32 %337, 16
  %339 = or i32 %334, %338
  %340 = getelementptr inbounds [64 x i8], [64 x i8]* %5, i64 0, i64 62
  %341 = load i8, i8* %340, align 1
  %342 = zext i8 %341 to i32
  %343 = shl nuw nsw i32 %342, 8
  %344 = or i32 %339, %343
  %345 = getelementptr inbounds [64 x i8], [64 x i8]* %5, i64 0, i64 63
  %346 = load i8, i8* %345, align 1
  %347 = zext i8 %346 to i32
  %348 = or i32 %344, %347
  %349 = or i64 %51, 15
  %350 = getelementptr inbounds i32, i32 addrspace(4)* %2, i64 %349
  store i32 %348, i32 addrspace(4)* %350, align 4
  %351 = or i64 %51, 16
  %352 = add i32 %68, %228
  %353 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %86, i32 %87, i32 25) #0
  %354 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %87, i32 %78, i32 14) #0
  %355 = xor i32 %353, %354
  %356 = lshr i32 %87, 3
  %357 = xor i32 %355, %356
  %358 = add i32 %352, %357
  %359 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %328, i32 %319, i32 15) #0
  %360 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %328, i32 %319, i32 13) #0
  %361 = xor i32 %359, %360
  %362 = lshr i32 %324, 10
  %363 = xor i32 %361, %362
  %364 = add i32 %358, %363
  %365 = getelementptr inbounds i32, i32 addrspace(4)* %2, i64 %351
  store i32 %364, i32 addrspace(4)* %365, align 4
  %366 = or i64 %51, 17
  %367 = add i32 %87, %248
  %368 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %106, i32 %107, i32 25) #0
  %369 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %107, i32 %98, i32 14) #0
  %370 = xor i32 %368, %369
  %371 = lshr i32 %107, 3
  %372 = xor i32 %370, %371
  %373 = add i32 %367, %372
  %374 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %348, i32 %339, i32 15) #0
  %375 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %348, i32 %339, i32 13) #0
  %376 = xor i32 %374, %375
  %377 = lshr i32 %344, 10
  %378 = xor i32 %376, %377
  %379 = add i32 %373, %378
  %380 = getelementptr inbounds i32, i32 addrspace(4)* %2, i64 %366
  store i32 %379, i32 addrspace(4)* %380, align 4
  %381 = or i64 %51, 18
  %382 = add i32 %107, %268
  %383 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %126, i32 %127, i32 25) #0
  %384 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %127, i32 %118, i32 14) #0
  %385 = xor i32 %383, %384
  %386 = lshr i32 %127, 3
  %387 = xor i32 %385, %386
  %388 = add i32 %382, %387
  %389 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %364, i32 %364, i32 15) #0
  %390 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %364, i32 %364, i32 13) #0
  %391 = xor i32 %389, %390
  %392 = lshr i32 %364, 10
  %393 = xor i32 %391, %392
  %394 = add i32 %388, %393
  %395 = getelementptr inbounds i32, i32 addrspace(4)* %2, i64 %381
  store i32 %394, i32 addrspace(4)* %395, align 4
  %396 = or i64 %51, 19
  %397 = add i32 %127, %288
  %398 = lshr i32 %132, 7
  %399 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %132, i32 %132, i32 14) #0
  %400 = xor i32 %398, %399
  %401 = lshr i32 %132, 3
  %402 = xor i32 %400, %401
  %403 = add i32 %397, %402
  %404 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %379, i32 %379, i32 15) #0
  %405 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %379, i32 %379, i32 13) #0
  %406 = xor i32 %404, %405
  %407 = lshr i32 %379, 10
  %408 = xor i32 %406, %407
  %409 = add i32 %403, %408
  %410 = getelementptr inbounds i32, i32 addrspace(4)* %2, i64 %396
  store i32 %409, i32 addrspace(4)* %410, align 4
  %411 = or i64 %51, 20
  %412 = add i32 %132, %308
  %413 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %147, i32 %148, i32 25) #0
  %414 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %148, i32 %141, i32 14) #0
  %415 = xor i32 %413, %414
  %416 = lshr i32 %148, 3
  %417 = xor i32 %415, %416
  %418 = add i32 %412, %417
  %419 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %394, i32 %394, i32 15) #0
  %420 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %394, i32 %394, i32 13) #0
  %421 = xor i32 %419, %420
  %422 = lshr i32 %394, 10
  %423 = xor i32 %421, %422
  %424 = add i32 %418, %423
  %425 = getelementptr inbounds i32, i32 addrspace(4)* %2, i64 %411
  store i32 %424, i32 addrspace(4)* %425, align 4
  %426 = or i64 %51, 21
  %427 = add i32 %148, %328
  %428 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %167, i32 %168, i32 25) #0
  %429 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %168, i32 %159, i32 14) #0
  %430 = xor i32 %428, %429
  %431 = lshr i32 %168, 3
  %432 = xor i32 %430, %431
  %433 = add i32 %427, %432
  %434 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %409, i32 %409, i32 15) #0
  %435 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %409, i32 %409, i32 13) #0
  %436 = xor i32 %434, %435
  %437 = lshr i32 %409, 10
  %438 = xor i32 %436, %437
  %439 = add i32 %433, %438
  %440 = getelementptr inbounds i32, i32 addrspace(4)* %2, i64 %426
  store i32 %439, i32 addrspace(4)* %440, align 4
  %441 = or i64 %51, 22
  %442 = add i32 %168, %348
  %443 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %187, i32 %188, i32 25) #0
  %444 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %188, i32 %179, i32 14) #0
  %445 = xor i32 %443, %444
  %446 = lshr i32 %188, 3
  %447 = xor i32 %445, %446
  %448 = add i32 %442, %447
  %449 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %424, i32 %424, i32 15) #0
  %450 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %424, i32 %424, i32 13) #0
  %451 = xor i32 %449, %450
  %452 = lshr i32 %424, 10
  %453 = xor i32 %451, %452
  %454 = add i32 %448, %453
  %455 = getelementptr inbounds i32, i32 addrspace(4)* %2, i64 %441
  store i32 %454, i32 addrspace(4)* %455, align 4
  %456 = or i64 %51, 23
  %457 = add i32 %188, %364
  %458 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %207, i32 %208, i32 25) #0
  %459 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %208, i32 %199, i32 14) #0
  %460 = xor i32 %458, %459
  %461 = lshr i32 %208, 3
  %462 = xor i32 %460, %461
  %463 = add i32 %457, %462
  %464 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %439, i32 %439, i32 15) #0
  %465 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %439, i32 %439, i32 13) #0
  %466 = xor i32 %464, %465
  %467 = lshr i32 %439, 10
  %468 = xor i32 %466, %467
  %469 = add i32 %463, %468
  %470 = getelementptr inbounds i32, i32 addrspace(4)* %2, i64 %456
  store i32 %469, i32 addrspace(4)* %470, align 4
  %471 = or i64 %51, 24
  %472 = add i32 %208, %379
  %473 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %227, i32 %228, i32 25) #0
  %474 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %228, i32 %219, i32 14) #0
  %475 = xor i32 %473, %474
  %476 = lshr i32 %228, 3
  %477 = xor i32 %475, %476
  %478 = add i32 %472, %477
  %479 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %454, i32 %454, i32 15) #0
  %480 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %454, i32 %454, i32 13) #0
  %481 = xor i32 %479, %480
  %482 = lshr i32 %454, 10
  %483 = xor i32 %481, %482
  %484 = add i32 %478, %483
  %485 = getelementptr inbounds i32, i32 addrspace(4)* %2, i64 %471
  store i32 %484, i32 addrspace(4)* %485, align 4
  %486 = or i64 %51, 25
  %487 = add i32 %228, %394
  %488 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %247, i32 %248, i32 25) #0
  %489 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %248, i32 %239, i32 14) #0
  %490 = xor i32 %488, %489
  %491 = lshr i32 %248, 3
  %492 = xor i32 %490, %491
  %493 = add i32 %487, %492
  %494 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %469, i32 %469, i32 15) #0
  %495 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %469, i32 %469, i32 13) #0
  %496 = xor i32 %494, %495
  %497 = lshr i32 %469, 10
  %498 = xor i32 %496, %497
  %499 = add i32 %493, %498
  %500 = getelementptr inbounds i32, i32 addrspace(4)* %2, i64 %486
  store i32 %499, i32 addrspace(4)* %500, align 4
  %501 = or i64 %51, 26
  %502 = add i32 %248, %409
  %503 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %267, i32 %268, i32 25) #0
  %504 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %268, i32 %259, i32 14) #0
  %505 = xor i32 %503, %504
  %506 = lshr i32 %268, 3
  %507 = xor i32 %505, %506
  %508 = add i32 %502, %507
  %509 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %484, i32 %484, i32 15) #0
  %510 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %484, i32 %484, i32 13) #0
  %511 = xor i32 %509, %510
  %512 = lshr i32 %484, 10
  %513 = xor i32 %511, %512
  %514 = add i32 %508, %513
  %515 = getelementptr inbounds i32, i32 addrspace(4)* %2, i64 %501
  store i32 %514, i32 addrspace(4)* %515, align 4
  %516 = or i64 %51, 27
  %517 = add i32 %268, %424
  %518 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %287, i32 %288, i32 25) #0
  %519 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %288, i32 %279, i32 14) #0
  %520 = xor i32 %518, %519
  %521 = lshr i32 %288, 3
  %522 = xor i32 %520, %521
  %523 = add i32 %517, %522
  %524 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %499, i32 %499, i32 15) #0
  %525 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %499, i32 %499, i32 13) #0
  %526 = xor i32 %524, %525
  %527 = lshr i32 %499, 10
  %528 = xor i32 %526, %527
  %529 = add i32 %523, %528
  %530 = getelementptr inbounds i32, i32 addrspace(4)* %2, i64 %516
  store i32 %529, i32 addrspace(4)* %530, align 4
  %531 = or i64 %51, 28
  %532 = add i32 %288, %439
  %533 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %307, i32 %308, i32 25) #0
  %534 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %308, i32 %299, i32 14) #0
  %535 = xor i32 %533, %534
  %536 = lshr i32 %308, 3
  %537 = xor i32 %535, %536
  %538 = add i32 %532, %537
  %539 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %514, i32 %514, i32 15) #0
  %540 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %514, i32 %514, i32 13) #0
  %541 = xor i32 %539, %540
  %542 = lshr i32 %514, 10
  %543 = xor i32 %541, %542
  %544 = add i32 %538, %543
  %545 = getelementptr inbounds i32, i32 addrspace(4)* %2, i64 %531
  store i32 %544, i32 addrspace(4)* %545, align 4
  %546 = or i64 %51, 29
  %547 = add i32 %308, %454
  %548 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %327, i32 %328, i32 25) #0
  %549 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %328, i32 %319, i32 14) #0
  %550 = xor i32 %548, %549
  %551 = lshr i32 %328, 3
  %552 = xor i32 %550, %551
  %553 = add i32 %547, %552
  %554 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %529, i32 %529, i32 15) #0
  %555 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %529, i32 %529, i32 13) #0
  %556 = xor i32 %554, %555
  %557 = lshr i32 %529, 10
  %558 = xor i32 %556, %557
  %559 = add i32 %553, %558
  %560 = getelementptr inbounds i32, i32 addrspace(4)* %2, i64 %546
  store i32 %559, i32 addrspace(4)* %560, align 4
  %561 = or i64 %51, 30
  %562 = add i32 %328, %469
  %563 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %347, i32 %348, i32 25) #0
  %564 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %348, i32 %339, i32 14) #0
  %565 = xor i32 %563, %564
  %566 = lshr i32 %348, 3
  %567 = xor i32 %565, %566
  %568 = add i32 %562, %567
  %569 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %544, i32 %544, i32 15) #0
  %570 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %544, i32 %544, i32 13) #0
  %571 = xor i32 %569, %570
  %572 = lshr i32 %544, 10
  %573 = xor i32 %571, %572
  %574 = add i32 %568, %573
  %575 = getelementptr inbounds i32, i32 addrspace(4)* %2, i64 %561
  store i32 %574, i32 addrspace(4)* %575, align 4
  %576 = or i64 %51, 31
  %577 = add i32 %348, %484
  %578 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %364, i32 %364, i32 25) #0
  %579 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %364, i32 %364, i32 14) #0
  %580 = xor i32 %578, %579
  %581 = lshr i32 %364, 3
  %582 = xor i32 %580, %581
  %583 = add i32 %577, %582
  %584 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %559, i32 %559, i32 15) #0
  %585 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %559, i32 %559, i32 13) #0
  %586 = xor i32 %584, %585
  %587 = lshr i32 %559, 10
  %588 = xor i32 %586, %587
  %589 = add i32 %583, %588
  %590 = getelementptr inbounds i32, i32 addrspace(4)* %2, i64 %576
  store i32 %589, i32 addrspace(4)* %590, align 4
  %591 = or i64 %51, 32
  %592 = add i32 %364, %499
  %593 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %379, i32 %379, i32 25) #0
  %594 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %379, i32 %379, i32 14) #0
  %595 = xor i32 %593, %594
  %596 = lshr i32 %379, 3
  %597 = xor i32 %595, %596
  %598 = add i32 %592, %597
  %599 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %574, i32 %574, i32 15) #0
  %600 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %574, i32 %574, i32 13) #0
  %601 = xor i32 %599, %600
  %602 = lshr i32 %574, 10
  %603 = xor i32 %601, %602
  %604 = add i32 %598, %603
  %605 = getelementptr inbounds i32, i32 addrspace(4)* %2, i64 %591
  store i32 %604, i32 addrspace(4)* %605, align 4
  %606 = or i64 %51, 33
  %607 = add i32 %379, %514
  %608 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %394, i32 %394, i32 25) #0
  %609 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %394, i32 %394, i32 14) #0
  %610 = xor i32 %608, %609
  %611 = lshr i32 %394, 3
  %612 = xor i32 %610, %611
  %613 = add i32 %607, %612
  %614 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %589, i32 %589, i32 15) #0
  %615 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %589, i32 %589, i32 13) #0
  %616 = xor i32 %614, %615
  %617 = lshr i32 %589, 10
  %618 = xor i32 %616, %617
  %619 = add i32 %613, %618
  %620 = getelementptr inbounds i32, i32 addrspace(4)* %2, i64 %606
  store i32 %619, i32 addrspace(4)* %620, align 4
  %621 = or i64 %51, 34
  %622 = add i32 %394, %529
  %623 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %409, i32 %409, i32 25) #0
  %624 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %409, i32 %409, i32 14) #0
  %625 = xor i32 %623, %624
  %626 = lshr i32 %409, 3
  %627 = xor i32 %625, %626
  %628 = add i32 %622, %627
  %629 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %604, i32 %604, i32 15) #0
  %630 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %604, i32 %604, i32 13) #0
  %631 = xor i32 %629, %630
  %632 = lshr i32 %604, 10
  %633 = xor i32 %631, %632
  %634 = add i32 %628, %633
  %635 = getelementptr inbounds i32, i32 addrspace(4)* %2, i64 %621
  store i32 %634, i32 addrspace(4)* %635, align 4
  %636 = or i64 %51, 35
  %637 = add i32 %409, %544
  %638 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %424, i32 %424, i32 25) #0
  %639 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %424, i32 %424, i32 14) #0
  %640 = xor i32 %638, %639
  %641 = lshr i32 %424, 3
  %642 = xor i32 %640, %641
  %643 = add i32 %637, %642
  %644 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %619, i32 %619, i32 15) #0
  %645 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %619, i32 %619, i32 13) #0
  %646 = xor i32 %644, %645
  %647 = lshr i32 %619, 10
  %648 = xor i32 %646, %647
  %649 = add i32 %643, %648
  %650 = getelementptr inbounds i32, i32 addrspace(4)* %2, i64 %636
  store i32 %649, i32 addrspace(4)* %650, align 4
  %651 = or i64 %51, 36
  %652 = add i32 %424, %559
  %653 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %439, i32 %439, i32 25) #0
  %654 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %439, i32 %439, i32 14) #0
  %655 = xor i32 %653, %654
  %656 = lshr i32 %439, 3
  %657 = xor i32 %655, %656
  %658 = add i32 %652, %657
  %659 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %634, i32 %634, i32 15) #0
  %660 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %634, i32 %634, i32 13) #0
  %661 = xor i32 %659, %660
  %662 = lshr i32 %634, 10
  %663 = xor i32 %661, %662
  %664 = add i32 %658, %663
  %665 = getelementptr inbounds i32, i32 addrspace(4)* %2, i64 %651
  store i32 %664, i32 addrspace(4)* %665, align 4
  %666 = or i64 %51, 37
  %667 = add i32 %439, %574
  %668 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %454, i32 %454, i32 25) #0
  %669 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %454, i32 %454, i32 14) #0
  %670 = xor i32 %668, %669
  %671 = lshr i32 %454, 3
  %672 = xor i32 %670, %671
  %673 = add i32 %667, %672
  %674 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %649, i32 %649, i32 15) #0
  %675 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %649, i32 %649, i32 13) #0
  %676 = xor i32 %674, %675
  %677 = lshr i32 %649, 10
  %678 = xor i32 %676, %677
  %679 = add i32 %673, %678
  %680 = getelementptr inbounds i32, i32 addrspace(4)* %2, i64 %666
  store i32 %679, i32 addrspace(4)* %680, align 4
  %681 = or i64 %51, 38
  %682 = add i32 %454, %589
  %683 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %469, i32 %469, i32 25) #0
  %684 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %469, i32 %469, i32 14) #0
  %685 = xor i32 %683, %684
  %686 = lshr i32 %469, 3
  %687 = xor i32 %685, %686
  %688 = add i32 %682, %687
  %689 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %664, i32 %664, i32 15) #0
  %690 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %664, i32 %664, i32 13) #0
  %691 = xor i32 %689, %690
  %692 = lshr i32 %664, 10
  %693 = xor i32 %691, %692
  %694 = add i32 %688, %693
  %695 = getelementptr inbounds i32, i32 addrspace(4)* %2, i64 %681
  store i32 %694, i32 addrspace(4)* %695, align 4
  %696 = or i64 %51, 39
  %697 = add i32 %469, %604
  %698 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %484, i32 %484, i32 25) #0
  %699 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %484, i32 %484, i32 14) #0
  %700 = xor i32 %698, %699
  %701 = lshr i32 %484, 3
  %702 = xor i32 %700, %701
  %703 = add i32 %697, %702
  %704 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %679, i32 %679, i32 15) #0
  %705 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %679, i32 %679, i32 13) #0
  %706 = xor i32 %704, %705
  %707 = lshr i32 %679, 10
  %708 = xor i32 %706, %707
  %709 = add i32 %703, %708
  %710 = getelementptr inbounds i32, i32 addrspace(4)* %2, i64 %696
  store i32 %709, i32 addrspace(4)* %710, align 4
  %711 = or i64 %51, 40
  %712 = add i32 %484, %619
  %713 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %499, i32 %499, i32 25) #0
  %714 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %499, i32 %499, i32 14) #0
  %715 = xor i32 %713, %714
  %716 = lshr i32 %499, 3
  %717 = xor i32 %715, %716
  %718 = add i32 %712, %717
  %719 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %694, i32 %694, i32 15) #0
  %720 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %694, i32 %694, i32 13) #0
  %721 = xor i32 %719, %720
  %722 = lshr i32 %694, 10
  %723 = xor i32 %721, %722
  %724 = add i32 %718, %723
  %725 = getelementptr inbounds i32, i32 addrspace(4)* %2, i64 %711
  store i32 %724, i32 addrspace(4)* %725, align 4
  %726 = or i64 %51, 41
  %727 = add i32 %499, %634
  %728 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %514, i32 %514, i32 25) #0
  %729 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %514, i32 %514, i32 14) #0
  %730 = xor i32 %728, %729
  %731 = lshr i32 %514, 3
  %732 = xor i32 %730, %731
  %733 = add i32 %727, %732
  %734 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %709, i32 %709, i32 15) #0
  %735 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %709, i32 %709, i32 13) #0
  %736 = xor i32 %734, %735
  %737 = lshr i32 %709, 10
  %738 = xor i32 %736, %737
  %739 = add i32 %733, %738
  %740 = getelementptr inbounds i32, i32 addrspace(4)* %2, i64 %726
  store i32 %739, i32 addrspace(4)* %740, align 4
  %741 = or i64 %51, 42
  %742 = add i32 %514, %649
  %743 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %529, i32 %529, i32 25) #0
  %744 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %529, i32 %529, i32 14) #0
  %745 = xor i32 %743, %744
  %746 = lshr i32 %529, 3
  %747 = xor i32 %745, %746
  %748 = add i32 %742, %747
  %749 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %724, i32 %724, i32 15) #0
  %750 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %724, i32 %724, i32 13) #0
  %751 = xor i32 %749, %750
  %752 = lshr i32 %724, 10
  %753 = xor i32 %751, %752
  %754 = add i32 %748, %753
  %755 = getelementptr inbounds i32, i32 addrspace(4)* %2, i64 %741
  store i32 %754, i32 addrspace(4)* %755, align 4
  %756 = or i64 %51, 43
  %757 = add i32 %529, %664
  %758 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %544, i32 %544, i32 25) #0
  %759 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %544, i32 %544, i32 14) #0
  %760 = xor i32 %758, %759
  %761 = lshr i32 %544, 3
  %762 = xor i32 %760, %761
  %763 = add i32 %757, %762
  %764 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %739, i32 %739, i32 15) #0
  %765 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %739, i32 %739, i32 13) #0
  %766 = xor i32 %764, %765
  %767 = lshr i32 %739, 10
  %768 = xor i32 %766, %767
  %769 = add i32 %763, %768
  %770 = getelementptr inbounds i32, i32 addrspace(4)* %2, i64 %756
  store i32 %769, i32 addrspace(4)* %770, align 4
  %771 = or i64 %51, 44
  %772 = add i32 %544, %679
  %773 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %559, i32 %559, i32 25) #0
  %774 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %559, i32 %559, i32 14) #0
  %775 = xor i32 %773, %774
  %776 = lshr i32 %559, 3
  %777 = xor i32 %775, %776
  %778 = add i32 %772, %777
  %779 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %754, i32 %754, i32 15) #0
  %780 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %754, i32 %754, i32 13) #0
  %781 = xor i32 %779, %780
  %782 = lshr i32 %754, 10
  %783 = xor i32 %781, %782
  %784 = add i32 %778, %783
  %785 = getelementptr inbounds i32, i32 addrspace(4)* %2, i64 %771
  store i32 %784, i32 addrspace(4)* %785, align 4
  %786 = or i64 %51, 45
  %787 = add i32 %559, %694
  %788 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %574, i32 %574, i32 25) #0
  %789 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %574, i32 %574, i32 14) #0
  %790 = xor i32 %788, %789
  %791 = lshr i32 %574, 3
  %792 = xor i32 %790, %791
  %793 = add i32 %787, %792
  %794 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %769, i32 %769, i32 15) #0
  %795 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %769, i32 %769, i32 13) #0
  %796 = xor i32 %794, %795
  %797 = lshr i32 %769, 10
  %798 = xor i32 %796, %797
  %799 = add i32 %793, %798
  %800 = getelementptr inbounds i32, i32 addrspace(4)* %2, i64 %786
  store i32 %799, i32 addrspace(4)* %800, align 4
  %801 = or i64 %51, 46
  %802 = add i32 %574, %709
  %803 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %589, i32 %589, i32 25) #0
  %804 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %589, i32 %589, i32 14) #0
  %805 = xor i32 %803, %804
  %806 = lshr i32 %589, 3
  %807 = xor i32 %805, %806
  %808 = add i32 %802, %807
  %809 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %784, i32 %784, i32 15) #0
  %810 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %784, i32 %784, i32 13) #0
  %811 = xor i32 %809, %810
  %812 = lshr i32 %784, 10
  %813 = xor i32 %811, %812
  %814 = add i32 %808, %813
  %815 = getelementptr inbounds i32, i32 addrspace(4)* %2, i64 %801
  store i32 %814, i32 addrspace(4)* %815, align 4
  %816 = or i64 %51, 47
  %817 = add i32 %589, %724
  %818 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %604, i32 %604, i32 25) #0
  %819 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %604, i32 %604, i32 14) #0
  %820 = xor i32 %818, %819
  %821 = lshr i32 %604, 3
  %822 = xor i32 %820, %821
  %823 = add i32 %817, %822
  %824 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %799, i32 %799, i32 15) #0
  %825 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %799, i32 %799, i32 13) #0
  %826 = xor i32 %824, %825
  %827 = lshr i32 %799, 10
  %828 = xor i32 %826, %827
  %829 = add i32 %823, %828
  %830 = getelementptr inbounds i32, i32 addrspace(4)* %2, i64 %816
  store i32 %829, i32 addrspace(4)* %830, align 4
  %831 = or i64 %51, 48
  %832 = add i32 %604, %739
  %833 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %619, i32 %619, i32 25) #0
  %834 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %619, i32 %619, i32 14) #0
  %835 = xor i32 %833, %834
  %836 = lshr i32 %619, 3
  %837 = xor i32 %835, %836
  %838 = add i32 %832, %837
  %839 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %814, i32 %814, i32 15) #0
  %840 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %814, i32 %814, i32 13) #0
  %841 = xor i32 %839, %840
  %842 = lshr i32 %814, 10
  %843 = xor i32 %841, %842
  %844 = add i32 %838, %843
  %845 = getelementptr inbounds i32, i32 addrspace(4)* %2, i64 %831
  store i32 %844, i32 addrspace(4)* %845, align 4
  %846 = or i64 %51, 49
  %847 = add i32 %619, %754
  %848 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %634, i32 %634, i32 25) #0
  %849 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %634, i32 %634, i32 14) #0
  %850 = xor i32 %848, %849
  %851 = lshr i32 %634, 3
  %852 = xor i32 %850, %851
  %853 = add i32 %847, %852
  %854 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %829, i32 %829, i32 15) #0
  %855 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %829, i32 %829, i32 13) #0
  %856 = xor i32 %854, %855
  %857 = lshr i32 %829, 10
  %858 = xor i32 %856, %857
  %859 = add i32 %853, %858
  %860 = getelementptr inbounds i32, i32 addrspace(4)* %2, i64 %846
  store i32 %859, i32 addrspace(4)* %860, align 4
  %861 = or i64 %51, 50
  %862 = add i32 %634, %769
  %863 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %649, i32 %649, i32 25) #0
  %864 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %649, i32 %649, i32 14) #0
  %865 = xor i32 %863, %864
  %866 = lshr i32 %649, 3
  %867 = xor i32 %865, %866
  %868 = add i32 %862, %867
  %869 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %844, i32 %844, i32 15) #0
  %870 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %844, i32 %844, i32 13) #0
  %871 = xor i32 %869, %870
  %872 = lshr i32 %844, 10
  %873 = xor i32 %871, %872
  %874 = add i32 %868, %873
  %875 = getelementptr inbounds i32, i32 addrspace(4)* %2, i64 %861
  store i32 %874, i32 addrspace(4)* %875, align 4
  %876 = or i64 %51, 51
  %877 = add i32 %649, %784
  %878 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %664, i32 %664, i32 25) #0
  %879 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %664, i32 %664, i32 14) #0
  %880 = xor i32 %878, %879
  %881 = lshr i32 %664, 3
  %882 = xor i32 %880, %881
  %883 = add i32 %877, %882
  %884 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %859, i32 %859, i32 15) #0
  %885 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %859, i32 %859, i32 13) #0
  %886 = xor i32 %884, %885
  %887 = lshr i32 %859, 10
  %888 = xor i32 %886, %887
  %889 = add i32 %883, %888
  %890 = getelementptr inbounds i32, i32 addrspace(4)* %2, i64 %876
  store i32 %889, i32 addrspace(4)* %890, align 4
  %891 = or i64 %51, 52
  %892 = add i32 %664, %799
  %893 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %679, i32 %679, i32 25) #0
  %894 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %679, i32 %679, i32 14) #0
  %895 = xor i32 %893, %894
  %896 = lshr i32 %679, 3
  %897 = xor i32 %895, %896
  %898 = add i32 %892, %897
  %899 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %874, i32 %874, i32 15) #0
  %900 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %874, i32 %874, i32 13) #0
  %901 = xor i32 %899, %900
  %902 = lshr i32 %874, 10
  %903 = xor i32 %901, %902
  %904 = add i32 %898, %903
  %905 = getelementptr inbounds i32, i32 addrspace(4)* %2, i64 %891
  store i32 %904, i32 addrspace(4)* %905, align 4
  %906 = or i64 %51, 53
  %907 = add i32 %679, %814
  %908 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %694, i32 %694, i32 25) #0
  %909 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %694, i32 %694, i32 14) #0
  %910 = xor i32 %908, %909
  %911 = lshr i32 %694, 3
  %912 = xor i32 %910, %911
  %913 = add i32 %907, %912
  %914 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %889, i32 %889, i32 15) #0
  %915 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %889, i32 %889, i32 13) #0
  %916 = xor i32 %914, %915
  %917 = lshr i32 %889, 10
  %918 = xor i32 %916, %917
  %919 = add i32 %913, %918
  %920 = getelementptr inbounds i32, i32 addrspace(4)* %2, i64 %906
  store i32 %919, i32 addrspace(4)* %920, align 4
  %921 = or i64 %51, 54
  %922 = add i32 %694, %829
  %923 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %709, i32 %709, i32 25) #0
  %924 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %709, i32 %709, i32 14) #0
  %925 = xor i32 %923, %924
  %926 = lshr i32 %709, 3
  %927 = xor i32 %925, %926
  %928 = add i32 %922, %927
  %929 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %904, i32 %904, i32 15) #0
  %930 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %904, i32 %904, i32 13) #0
  %931 = xor i32 %929, %930
  %932 = lshr i32 %904, 10
  %933 = xor i32 %931, %932
  %934 = add i32 %928, %933
  %935 = getelementptr inbounds i32, i32 addrspace(4)* %2, i64 %921
  store i32 %934, i32 addrspace(4)* %935, align 4
  %936 = or i64 %51, 55
  %937 = add i32 %709, %844
  %938 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %724, i32 %724, i32 25) #0
  %939 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %724, i32 %724, i32 14) #0
  %940 = xor i32 %938, %939
  %941 = lshr i32 %724, 3
  %942 = xor i32 %940, %941
  %943 = add i32 %937, %942
  %944 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %919, i32 %919, i32 15) #0
  %945 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %919, i32 %919, i32 13) #0
  %946 = xor i32 %944, %945
  %947 = lshr i32 %919, 10
  %948 = xor i32 %946, %947
  %949 = add i32 %943, %948
  %950 = getelementptr inbounds i32, i32 addrspace(4)* %2, i64 %936
  store i32 %949, i32 addrspace(4)* %950, align 4
  %951 = or i64 %51, 56
  %952 = add i32 %724, %859
  %953 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %739, i32 %739, i32 25) #0
  %954 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %739, i32 %739, i32 14) #0
  %955 = xor i32 %953, %954
  %956 = lshr i32 %739, 3
  %957 = xor i32 %955, %956
  %958 = add i32 %952, %957
  %959 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %934, i32 %934, i32 15) #0
  %960 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %934, i32 %934, i32 13) #0
  %961 = xor i32 %959, %960
  %962 = lshr i32 %934, 10
  %963 = xor i32 %961, %962
  %964 = add i32 %958, %963
  %965 = getelementptr inbounds i32, i32 addrspace(4)* %2, i64 %951
  store i32 %964, i32 addrspace(4)* %965, align 4
  %966 = or i64 %51, 57
  %967 = add i32 %739, %874
  %968 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %754, i32 %754, i32 25) #0
  %969 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %754, i32 %754, i32 14) #0
  %970 = xor i32 %968, %969
  %971 = lshr i32 %754, 3
  %972 = xor i32 %970, %971
  %973 = add i32 %967, %972
  %974 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %949, i32 %949, i32 15) #0
  %975 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %949, i32 %949, i32 13) #0
  %976 = xor i32 %974, %975
  %977 = lshr i32 %949, 10
  %978 = xor i32 %976, %977
  %979 = add i32 %973, %978
  %980 = getelementptr inbounds i32, i32 addrspace(4)* %2, i64 %966
  store i32 %979, i32 addrspace(4)* %980, align 4
  %981 = or i64 %51, 58
  %982 = add i32 %754, %889
  %983 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %769, i32 %769, i32 25) #0
  %984 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %769, i32 %769, i32 14) #0
  %985 = xor i32 %983, %984
  %986 = lshr i32 %769, 3
  %987 = xor i32 %985, %986
  %988 = add i32 %982, %987
  %989 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %964, i32 %964, i32 15) #0
  %990 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %964, i32 %964, i32 13) #0
  %991 = xor i32 %989, %990
  %992 = lshr i32 %964, 10
  %993 = xor i32 %991, %992
  %994 = add i32 %988, %993
  %995 = getelementptr inbounds i32, i32 addrspace(4)* %2, i64 %981
  store i32 %994, i32 addrspace(4)* %995, align 4
  %996 = or i64 %51, 59
  %997 = add i32 %769, %904
  %998 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %784, i32 %784, i32 25) #0
  %999 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %784, i32 %784, i32 14) #0
  %1000 = xor i32 %998, %999
  %1001 = lshr i32 %784, 3
  %1002 = xor i32 %1000, %1001
  %1003 = add i32 %997, %1002
  %1004 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %979, i32 %979, i32 15) #0
  %1005 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %979, i32 %979, i32 13) #0
  %1006 = xor i32 %1004, %1005
  %1007 = lshr i32 %979, 10
  %1008 = xor i32 %1006, %1007
  %1009 = add i32 %1003, %1008
  %1010 = getelementptr inbounds i32, i32 addrspace(4)* %2, i64 %996
  store i32 %1009, i32 addrspace(4)* %1010, align 4
  %1011 = or i64 %51, 60
  %1012 = add i32 %784, %919
  %1013 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %799, i32 %799, i32 25) #0
  %1014 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %799, i32 %799, i32 14) #0
  %1015 = xor i32 %1013, %1014
  %1016 = lshr i32 %799, 3
  %1017 = xor i32 %1015, %1016
  %1018 = add i32 %1012, %1017
  %1019 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %994, i32 %994, i32 15) #0
  %1020 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %994, i32 %994, i32 13) #0
  %1021 = xor i32 %1019, %1020
  %1022 = lshr i32 %994, 10
  %1023 = xor i32 %1021, %1022
  %1024 = add i32 %1018, %1023
  %1025 = getelementptr inbounds i32, i32 addrspace(4)* %2, i64 %1011
  store i32 %1024, i32 addrspace(4)* %1025, align 4
  %1026 = or i64 %51, 61
  %1027 = add i32 %799, %934
  %1028 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %814, i32 %814, i32 25) #0
  %1029 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %814, i32 %814, i32 14) #0
  %1030 = xor i32 %1028, %1029
  %1031 = lshr i32 %814, 3
  %1032 = xor i32 %1030, %1031
  %1033 = add i32 %1027, %1032
  %1034 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %1009, i32 %1009, i32 15) #0
  %1035 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %1009, i32 %1009, i32 13) #0
  %1036 = xor i32 %1034, %1035
  %1037 = lshr i32 %1009, 10
  %1038 = xor i32 %1036, %1037
  %1039 = add i32 %1033, %1038
  %1040 = getelementptr inbounds i32, i32 addrspace(4)* %2, i64 %1026
  store i32 %1039, i32 addrspace(4)* %1040, align 4
  %1041 = or i64 %51, 62
  %1042 = add i32 %814, %949
  %1043 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %829, i32 %829, i32 25) #0
  %1044 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %829, i32 %829, i32 14) #0
  %1045 = xor i32 %1043, %1044
  %1046 = lshr i32 %829, 3
  %1047 = xor i32 %1045, %1046
  %1048 = add i32 %1042, %1047
  %1049 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %1024, i32 %1024, i32 15) #0
  %1050 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %1024, i32 %1024, i32 13) #0
  %1051 = xor i32 %1049, %1050
  %1052 = lshr i32 %1024, 10
  %1053 = xor i32 %1051, %1052
  %1054 = add i32 %1048, %1053
  %1055 = getelementptr inbounds i32, i32 addrspace(4)* %2, i64 %1041
  store i32 %1054, i32 addrspace(4)* %1055, align 4
  %1056 = or i64 %51, 63
  %1057 = add i32 %829, %964
  %1058 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %844, i32 %844, i32 25) #0
  %1059 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %844, i32 %844, i32 14) #0
  %1060 = xor i32 %1058, %1059
  %1061 = lshr i32 %844, 3
  %1062 = xor i32 %1060, %1061
  %1063 = add i32 %1057, %1062
  %1064 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %1039, i32 %1039, i32 15) #0
  %1065 = call spir_func i32 @spirv.llvm_fshl_i32(i32 %1039, i32 %1039, i32 13) #0
  %1066 = xor i32 %1064, %1065
  %1067 = lshr i32 %1039, 10
  %1068 = xor i32 %1066, %1067
  %1069 = add i32 %1063, %1068
  %1070 = getelementptr inbounds i32, i32 addrspace(4)* %2, i64 %1056
  store i32 %1069, i32 addrspace(4)* %1070, align 4
  %1071 = bitcast i8* %23 to i8*
  call void @llvm.lifetime.end.p0i8(i64 64, i8* %1071)
  br label %1072

1072:                                             ; preds = %45, %4
  ret void
}

; Function Attrs: nounwind willreturn
declare void @llvm.assume(i1) #4

; Function Attrs: nounwind
define spir_func i32 @spirv.llvm_bswap_i32(i32 %0) #0 {
entry:
  %bswap.4 = shl i32 %0, 24
  %bswap.3 = shl i32 %0, 8
  %bswap.2 = lshr i32 %0, 8
  %bswap.1 = lshr i32 %0, 24
  %bswap.and3 = and i32 %bswap.3, 16711680
  %bswap.and2 = and i32 %bswap.2, 65280
  %bswap.or1 = or i32 %bswap.4, %bswap.and3
  %bswap.or2 = or i32 %bswap.and2, %bswap.1
  %bswap.i32 = or i32 %bswap.or1, %bswap.or2
  ret i32 %bswap.i32
}

; Function Attrs: nounwind
define spir_func i32 @spirv.llvm_fshl_i32(i32 %0, i32 %1, i32 %2) #0 {
rotate:
  %3 = urem i32 %2, 32
  %4 = shl i32 %0, %3
  %5 = sub i32 32, %3
  %6 = lshr i32 %1, %5
  %7 = or i32 %4, %6
  ret i32 %7
}

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
declare spir_func <3 x i64> @__builtin_spirv_BuiltInGlobalSize() #5

; Function Attrs: nounwind readnone
declare spir_func <3 x i64> @__builtin_spirv_BuiltInGlobalOffset() #5

; Function Attrs: nounwind readnone
declare spir_func <3 x i64> @__builtin_spirv_BuiltInNumWorkgroups() #5

; Function Attrs: nounwind readnone
declare spir_func <3 x i64> @__builtin_spirv_BuiltInGlobalInvocationId() #5

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
!IGCMetadata = !{!6}
!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!256}
!opencl.ocl.version = !{!256}
!opencl.used.extensions = !{!257}
!opencl.used.optional.core.features = !{!257}
!opencl.compiler.options = !{!257}
!igc.functions = !{}

!0 = !{void (i8 addrspace(1)*, i8 addrspace(1)*, i32 addrspace(1)*)* @_ZTSZ16evaluate_w_blockPhPjRdEUlN4sycl3_V17nd_itemILi3EEEE_, !1, !2, !3, !4, !5}
!1 = !{!"kernel_arg_addr_space", i32 1, i32 1, i32 1}
!2 = !{!"kernel_arg_access_qual", !"none", !"none", !"none"}
!3 = !{!"kernel_arg_type", !"char*", !"char*", !"int*"}
!4 = !{!"kernel_arg_type_qual", !"", !"", !""}
!5 = !{!"kernel_arg_base_type", !"char*", !"char*", !"int*"}
!6 = !{!"ModuleMD", !7, !8, !68, !128, !158, !174, !189, !199, !201, !202, !215, !216, !217, !218, !222, !223, !224, !225, !226, !227, !228, !229, !230, !231, !232, !233, !234, !236, !240, !241, !242, !243, !244, !245, !246, !113, !247, !248, !249, !251, !254, !255}
!7 = !{!"isPrecise", i1 false}
!8 = !{!"compOpt", !9, !10, !11, !12, !13, !14, !15, !16, !17, !18, !19, !20, !21, !22, !23, !24, !25, !26, !27, !28, !29, !30, !31, !32, !33, !34, !35, !36, !37, !38, !39, !40, !41, !42, !43, !44, !45, !46, !47, !48, !49, !50, !51, !52, !53, !54, !55, !56, !57, !58, !59, !60, !61, !62, !63, !64, !65, !66, !67}
!9 = !{!"DenormsAreZero", i1 false}
!10 = !{!"CorrectlyRoundedDivSqrt", i1 false}
!11 = !{!"OptDisable", i1 false}
!12 = !{!"MadEnable", i1 false}
!13 = !{!"NoSignedZeros", i1 false}
!14 = !{!"NoNaNs", i1 false}
!15 = !{!"FloatRoundingMode", i32 0}
!16 = !{!"FloatCvtIntRoundingMode", i32 3}
!17 = !{!"VISAPreSchedRPThreshold", i32 0}
!18 = !{!"SetLoopUnrollThreshold", i32 0}
!19 = !{!"UnsafeMathOptimizations", i1 false}
!20 = !{!"FiniteMathOnly", i1 false}
!21 = !{!"FastRelaxedMath", i1 false}
!22 = !{!"DashGSpecified", i1 false}
!23 = !{!"FastCompilation", i1 false}
!24 = !{!"UseScratchSpacePrivateMemory", i1 true}
!25 = !{!"RelaxedBuiltins", i1 false}
!26 = !{!"SubgroupIndependentForwardProgressRequired", i1 true}
!27 = !{!"GreaterThan2GBBufferRequired", i1 true}
!28 = !{!"GreaterThan4GBBufferRequired", i1 true}
!29 = !{!"DisableA64WA", i1 false}
!30 = !{!"ForceEnableA64WA", i1 false}
!31 = !{!"PushConstantsEnable", i1 true}
!32 = !{!"HasPositivePointerOffset", i1 false}
!33 = !{!"HasBufferOffsetArg", i1 false}
!34 = !{!"BufferOffsetArgOptional", i1 true}
!35 = !{!"HasSubDWAlignedPtrArg", i1 false}
!36 = !{!"replaceGlobalOffsetsByZero", i1 false}
!37 = !{!"forcePixelShaderSIMDMode", i32 0}
!38 = !{!"pixelShaderDoNotAbortOnSpill", i1 false}
!39 = !{!"UniformWGS", i1 false}
!40 = !{!"disableVertexComponentPacking", i1 false}
!41 = !{!"disablePartialVertexComponentPacking", i1 false}
!42 = !{!"PreferBindlessImages", i1 false}
!43 = !{!"UseBindlessMode", i1 false}
!44 = !{!"UseLegacyBindlessMode", i1 true}
!45 = !{!"disableMathRefactoring", i1 false}
!46 = !{!"atomicBranch", i1 false}
!47 = !{!"ForceInt32DivRemEmu", i1 false}
!48 = !{!"ForceInt32DivRemEmuSP", i1 false}
!49 = !{!"DisableFastestSingleCSSIMD", i1 false}
!50 = !{!"DisableFastestLinearScan", i1 false}
!51 = !{!"UseStatelessforPrivateMemory", i1 false}
!52 = !{!"EnableTakeGlobalAddress", i1 false}
!53 = !{!"IsLibraryCompilation", i1 false}
!54 = !{!"FastVISACompile", i1 false}
!55 = !{!"MatchSinCosPi", i1 false}
!56 = !{!"ExcludeIRFromZEBinary", i1 false}
!57 = !{!"EmitZeBinVISASections", i1 false}
!58 = !{!"FP64GenEmulationEnabled", i1 false}
!59 = !{!"allowDisableRematforCS", i1 false}
!60 = !{!"DisableIncSpillCostAllAddrTaken", i1 false}
!61 = !{!"DisableCPSOmaskWA", i1 false}
!62 = !{!"DisableFastestGopt", i1 false}
!63 = !{!"WaForceHalfPromotion", i1 false}
!64 = !{!"DisableConstantCoalescing", i1 false}
!65 = !{!"EnableUndefAlphaOutputAsRed", i1 true}
!66 = !{!"WaEnableALTModeVisaWA", i1 false}
!67 = !{!"NewSpillCostFunction", i1 false}
!68 = !{!"FuncMD", !69, !70}
!69 = !{!"FuncMDMap[0]", void (i8 addrspace(1)*, i8 addrspace(1)*, i32 addrspace(1)*)* @_ZTSZ16evaluate_w_blockPhPjRdEUlN4sycl3_V17nd_itemILi3EEEE_}
!70 = !{!"FuncMDValue[0]", !71, !72, !76, !77, !78, !99, !105, !106, !107, !108, !109, !110, !111, !112, !113, !114, !115, !116, !117, !118, !119, !120, !121, !122, !123, !124, !125, !126, !127}
!71 = !{!"localOffsets"}
!72 = !{!"workGroupWalkOrder", !73, !74, !75}
!73 = !{!"dim0", i32 0}
!74 = !{!"dim1", i32 0}
!75 = !{!"dim2", i32 0}
!76 = !{!"funcArgs"}
!77 = !{!"functionType", !"KernelFunction"}
!78 = !{!"rtInfo", !79, !80, !81, !82, !83, !84, !85, !86, !87, !88, !89, !90, !91, !92, !93, !94, !98}
!79 = !{!"callableShaderType", !"NumberOfCallableShaderTypes"}
!80 = !{!"isContinuation", i1 false}
!81 = !{!"hasTraceRayPayload", i1 false}
!82 = !{!"hasHitAttributes", i1 false}
!83 = !{!"hasCallableData", i1 false}
!84 = !{!"ShaderStackSize", i32 0}
!85 = !{!"ShaderHash", i64 0}
!86 = !{!"ShaderName", !""}
!87 = !{!"ParentName", !""}
!88 = !{!"SlotNum", i1* null}
!89 = !{!"NOSSize", i32 0}
!90 = !{!"globalRootSignatureSize", i32 0}
!91 = !{!"Entries"}
!92 = !{!"SpillUnions"}
!93 = !{!"CustomHitAttrSizeInBytes", i32 0}
!94 = !{!"Types", !95, !96, !97}
!95 = !{!"FrameStartTys"}
!96 = !{!"ArgumentTys"}
!97 = !{!"FullFrameTys"}
!98 = !{!"Aliases"}
!99 = !{!"resAllocMD", !100, !101, !102, !103, !104}
!100 = !{!"uavsNumType", i32 0}
!101 = !{!"srvsNumType", i32 0}
!102 = !{!"samplersNumType", i32 0}
!103 = !{!"argAllocMDList"}
!104 = !{!"inlineSamplersMD"}
!105 = !{!"maxByteOffsets"}
!106 = !{!"IsInitializer", i1 false}
!107 = !{!"IsFinalizer", i1 false}
!108 = !{!"CompiledSubGroupsNumber", i32 0}
!109 = !{!"hasInlineVmeSamplers", i1 false}
!110 = !{!"localSize", i32 0}
!111 = !{!"localIDPresent", i1 false}
!112 = !{!"groupIDPresent", i1 false}
!113 = !{!"privateMemoryPerWI", i32 0}
!114 = !{!"globalIDPresent", i1 false}
!115 = !{!"hasSyncRTCalls", i1 false}
!116 = !{!"hasNonKernelArgLoad", i1 false}
!117 = !{!"hasNonKernelArgStore", i1 false}
!118 = !{!"hasNonKernelArgAtomic", i1 false}
!119 = !{!"UserAnnotations"}
!120 = !{!"m_OpenCLArgAddressSpaces"}
!121 = !{!"m_OpenCLArgAccessQualifiers"}
!122 = !{!"m_OpenCLArgTypes"}
!123 = !{!"m_OpenCLArgBaseTypes"}
!124 = !{!"m_OpenCLArgTypeQualifiers"}
!125 = !{!"m_OpenCLArgNames"}
!126 = !{!"m_OpenCLArgScalarAsPointers"}
!127 = !{!"m_OptsToDisablePerFunc"}
!128 = !{!"pushInfo", !129, !130, !131, !134, !135, !136, !137, !138, !139, !140, !141, !154, !155, !156, !157}
!129 = !{!"pushableAddresses"}
!130 = !{!"bindlessPushInfo"}
!131 = !{!"dynamicBufferInfo", !132, !133}
!132 = !{!"firstIndex", i32 0}
!133 = !{!"numOffsets", i32 0}
!134 = !{!"MaxNumberOfPushedBuffers", i32 0}
!135 = !{!"inlineConstantBufferSlot", i32 -1}
!136 = !{!"inlineConstantBufferOffset", i32 -1}
!137 = !{!"inlineConstantBufferGRFOffset", i32 -1}
!138 = !{!"constants"}
!139 = !{!"inputs"}
!140 = !{!"constantReg"}
!141 = !{!"simplePushInfoArr", !142, !151, !152, !153}
!142 = !{!"simplePushInfoArrVec[0]", !143, !144, !145, !146, !147, !148, !149, !150}
!143 = !{!"cbIdx", i32 0}
!144 = !{!"pushableAddressGrfOffset", i32 -1}
!145 = !{!"pushableOffsetGrfOffset", i32 -1}
!146 = !{!"offset", i32 0}
!147 = !{!"size", i32 0}
!148 = !{!"isStateless", i1 false}
!149 = !{!"isBindless", i1 false}
!150 = !{!"simplePushLoads"}
!151 = !{!"simplePushInfoArrVec[1]", !143, !144, !145, !146, !147, !148, !149, !150}
!152 = !{!"simplePushInfoArrVec[2]", !143, !144, !145, !146, !147, !148, !149, !150}
!153 = !{!"simplePushInfoArrVec[3]", !143, !144, !145, !146, !147, !148, !149, !150}
!154 = !{!"simplePushBufferUsed", i32 0}
!155 = !{!"pushAnalysisWIInfos"}
!156 = !{!"inlineRTGlobalPtrOffset", i32 0}
!157 = !{!"rtSyncSurfPtrOffset", i32 0}
!158 = !{!"psInfo", !159, !160, !161, !162, !163, !164, !165, !166, !167, !168, !169, !170, !171, !172, !173}
!159 = !{!"BlendStateDisabledMask", i8 0}
!160 = !{!"SkipSrc0Alpha", i1 false}
!161 = !{!"DualSourceBlendingDisabled", i1 false}
!162 = !{!"ForceEnableSimd32", i1 false}
!163 = !{!"outputDepth", i1 false}
!164 = !{!"outputStencil", i1 false}
!165 = !{!"outputMask", i1 false}
!166 = !{!"blendToFillEnabled", i1 false}
!167 = !{!"forceEarlyZ", i1 false}
!168 = !{!"hasVersionedLoop", i1 false}
!169 = !{!"forceSingleSourceRTWAfterDualSourceRTW", i1 false}
!170 = !{!"NumSamples", i8 0}
!171 = !{!"blendOptimizationMode"}
!172 = !{!"colorOutputMask"}
!173 = !{!"WaDisableVRS", i1 false}
!174 = !{!"csInfo", !175, !176, !177, !178, !179, !17, !18, !180, !181, !182, !183, !184, !185, !186, !187, !46, !188}
!175 = !{!"maxWorkGroupSize", i32 0}
!176 = !{!"waveSize", i32 0}
!177 = !{!"ComputeShaderSecondCompile"}
!178 = !{!"forcedSIMDSize", i8 0}
!179 = !{!"forceTotalGRFNum", i32 0}
!180 = !{!"allowLowerSimd", i1 false}
!181 = !{!"disableSimd32Slicing", i1 false}
!182 = !{!"disableSplitOnSpill", i1 false}
!183 = !{!"forcedVISAPreRAScheduler", i1 false}
!184 = !{!"disableLocalIdOrderOptimizations", i1 false}
!185 = !{!"disableDispatchAlongY", i1 false}
!186 = !{!"neededThreadIdLayout", i1* null}
!187 = !{!"forceTileYWalk", i1 false}
!188 = !{!"ResForHfPacking"}
!189 = !{!"msInfo", !190, !191, !192, !193, !194, !195, !196, !197, !198}
!190 = !{!"PrimitiveTopology", i32 3}
!191 = !{!"MaxNumOfPrimitives", i32 0}
!192 = !{!"MaxNumOfVertices", i32 0}
!193 = !{!"MaxNumOfPerPrimitiveOutputs", i32 0}
!194 = !{!"MaxNumOfPerVertexOutputs", i32 0}
!195 = !{!"WorkGroupSize", i32 0}
!196 = !{!"WorkGroupMemorySizeInBytes", i32 0}
!197 = !{!"IndexFormat", i32 6}
!198 = !{!"SubgroupSize", i32 0}
!199 = !{!"taskInfo", !200, !195, !196, !198}
!200 = !{!"MaxNumOfOutputs", i32 0}
!201 = !{!"NBarrierCnt", i32 0}
!202 = !{!"rtInfo", !203, !204, !205, !206, !207, !208, !209, !210, !211, !212, !213, !214}
!203 = !{!"RayQueryAllocSizeInBytes", i32 0}
!204 = !{!"NumContinuations", i32 0}
!205 = !{!"RTAsyncStackAddrspace", i32 -1}
!206 = !{!"RTAsyncStackSurfaceStateOffset", i1* null}
!207 = !{!"SWHotZoneAddrspace", i32 -1}
!208 = !{!"SWHotZoneSurfaceStateOffset", i1* null}
!209 = !{!"SWStackAddrspace", i32 -1}
!210 = !{!"SWStackSurfaceStateOffset", i1* null}
!211 = !{!"RTSyncStackAddrspace", i32 -1}
!212 = !{!"RTSyncStackSurfaceStateOffset", i1* null}
!213 = !{!"doSyncDispatchRays", i1 false}
!214 = !{!"MemStyle", !"Xe"}
!215 = !{!"CurUniqueIndirectIdx", i32 0}
!216 = !{!"inlineDynTextures"}
!217 = !{!"inlineResInfoData"}
!218 = !{!"immConstant", !219, !220, !221}
!219 = !{!"data"}
!220 = !{!"sizes"}
!221 = !{!"zeroIdxs"}
!222 = !{!"stringConstants"}
!223 = !{!"inlineConstantBuffers"}
!224 = !{!"inlineGlobalBuffers"}
!225 = !{!"GlobalPointerProgramBinaryInfos"}
!226 = !{!"ConstantPointerProgramBinaryInfos"}
!227 = !{!"GlobalBufferAddressRelocInfo"}
!228 = !{!"ConstantBufferAddressRelocInfo"}
!229 = !{!"forceLscCacheList"}
!230 = !{!"SrvMap"}
!231 = !{!"RasterizerOrderedByteAddressBuffer"}
!232 = !{!"MinNOSPushConstantSize", i32 0}
!233 = !{!"inlineProgramScopeOffsets"}
!234 = !{!"shaderData", !235}
!235 = !{!"numReplicas", i32 0}
!236 = !{!"URBInfo", !237, !238, !239}
!237 = !{!"has64BVertexHeaderInput", i1 false}
!238 = !{!"has64BVertexHeaderOutput", i1 false}
!239 = !{!"hasVertexHeader", i1 true}
!240 = !{!"UseBindlessImage", i1 false}
!241 = !{!"enableRangeReduce", i1 false}
!242 = !{!"allowMatchMadOptimizationforVS", i1 false}
!243 = !{!"disableMemOptforNegativeOffsetLoads", i1 false}
!244 = !{!"enableThreeWayLoadSpiltOpt", i1 false}
!245 = !{!"statefulResourcesNotAliased", i1 false}
!246 = !{!"disableMixMode", i1 false}
!247 = !{!"PrivateMemoryPerFG"}
!248 = !{!"m_OptsToDisable"}
!249 = !{!"capabilities", !250}
!250 = !{!"globalVariableDecorationsINTEL", i1 false}
!251 = !{!"m_ShaderResourceViewMcsMask", !252, !253}
!252 = !{!"m_ShaderResourceViewMcsMaskVec[0]", i64 0}
!253 = !{!"m_ShaderResourceViewMcsMaskVec[1]", i64 0}
!254 = !{!"computedDepthMode", i32 0}
!255 = !{!"isHDCFastClearShader", i1 false}
!256 = !{i32 1, i32 0}
!257 = !{}
