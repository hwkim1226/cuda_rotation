# cuda_rotation
1. rotation_test_kernel.cu : Using CUDA Kernel (null stream ver.)
2. rotation_test_kernel_stream.cu : Using CUDA Kernel
3. rotation_test_nppi_stream.cu : Using NPPI (nppiRotate_8u_C1R)
4. rotation_test_nppi_stream_ctx.cu : Using NPPI (nppiRotate_8u_C1R_Ctx)
* nppiRotate_8u_C1R_Ctx is available in CUDA 10.2 and later.
* This codes work for grayscale images.
