#include <cudnn.h>
#include <cuda_fp16.h>

constexpr std::size_t size = 1lu << 20;

int main() {
	float *dA;
	half  *dB;

	cudaMalloc(&dA, sizeof(float) * size);
	cudaMalloc(&dB, sizeof(half ) * size);

	cudnnHandle_t cudnn_handle;
	cudnnCreate(&cudnn_handle);

	cudnnTensorDescriptor_t a_tensor_desc;
	cudnnTensorDescriptor_t b_tensor_desc;
	cudnnCreateTensorDescriptor(&a_tensor_desc);
	cudnnCreateTensorDescriptor(&b_tensor_desc);

	cudnnSetTensor4dDescriptorEx(
			a_tensor_desc,
			CUDNN_DATA_FLOAT,
			1, size,
			1, 1,
			size, 1, 1, 1);
	cudnnSetTensor4dDescriptorEx(
			b_tensor_desc,
			CUDNN_DATA_HALF,
			1, size,
			1, 1,
			size, 1, 1, 1);

	const float alpha = 1.f, beta = 0.f;
	cudnnTransformTensor(
			cudnn_handle,
			&alpha,
			a_tensor_desc,
			dA, &beta,
			b_tensor_desc,
			dB
			);

	cudnnDestroyTensorDescriptor(a_tensor_desc);
	cudnnDestroyTensorDescriptor(b_tensor_desc);

	cudnnDestroy(cudnn_handle);

	cudaFree(dA);
	cudaFree(dB);
}
