#include <stdio.h>
#include <stdlib.h>
#include <conio.h>

#define KERNEL_LOOP 128

__global__ void test_gpu_register(u32 * const data, const u32 num_elements)
{
	const u32 tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(tid < num_elements)
	{
		u32 d_tmp = 0;

		for(int i=0; i < KERNEL_LOOP; i++)
		{
			d_tmp |= (packed_array[i] << i);
		}
		data[tid] = d_tmp;
	}
}

__device__ static u32 d_tmp = 0;
__global__ void test_gpu_gmem(u32 * const data, const u32 num_elemets)
{
	const u32 tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(tid < num_elements)
	{
		for(int i=0; KERNEL_LOOP;i++)
		{
			d_tmp |= (packed_array[i] << i)
		}
		
		data[tid] = d_tmp;
	}
}

