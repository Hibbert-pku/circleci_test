#include<cuda_runtime.h>

#include<thrust/scan.h>
#include<thrust/functional.h>

#include<iostream>

int add_(int a, int b)
{
	return a+b;
}

__device__ int log_plus(int a, int b)
{
        return a+b;
}

__device__ int segscan_warp(int* ptr, bool* hd, int idx) {
        const unsigned int lane = idx & 31;
        if (lane >= 1) {
                ptr[idx] = hd[idx] ? ptr[idx] : log_plus(ptr[idx - 1] , ptr[idx]);
                hd[idx] = hd[idx - 1] | hd[idx]; }
        if (lane >= 2) {
                ptr[idx] = hd[idx] ? ptr[idx] : log_plus(ptr[idx - 2] , ptr[idx]);
                hd[idx] = hd[idx - 2] | hd[idx]; }
        if (lane >= 4) {
                ptr[idx] = hd[idx] ? ptr[idx] : log_plus(ptr[idx - 4] , ptr[idx]);
                hd[idx] = hd[idx - 4] | hd[idx]; }
        if (lane >= 8) {
                ptr[idx] = hd[idx] ? ptr[idx] : log_plus(ptr[idx - 8] , ptr[idx]);
                hd[idx] = hd[idx - 8] | hd[idx]; }
        if (lane >= 16) {
                ptr[idx] = hd[idx] ? ptr[idx] : log_plus(ptr[idx - 16] , ptr[idx]);
                hd[idx] = hd[idx - 16] | hd[idx];
        }
	return ptr[idx];
}

__device__ void segscan_block(int* ptr, bool* hd, int idx)
{
	unsigned int warpid = idx >> 5;
        unsigned int warp_first = warpid << 5;
        unsigned int warp_last = warp_first + 31;
        // Step 1a:
        // Before overwriting the input head flags, record whether // this warp begins with an "open" segment.
        bool warp_is_open = (hd[warp_first] == 0);
        __syncthreads ();
        // Step 1b:
        // Intra-warp segmented scan in each warp.
        int val = segscan_warp(ptr, hd, idx);
        // Step 2a:
        // Since ptr[] contains *inclusive* results, irrespective of Kind, // the last value is the correct partial result.
        int warp_total = ptr[warp_last];
        // Step 2b:
        // warp_flag is the OR-reduction of the flags in a warp and is
        // computed indirectly from the mindex values in hd[].
        // will_accumulate indicates that a thread will only accumulate a
        // partial result in Step 4 if there is no segment boundary to its left. 
        bool warp_flag = hd[warp_last]!=0 || !warp_is_open;
        bool will_accumulate = warp_is_open && hd[idx]==0;
        __syncthreads ();
        // Step 2c: The last thread in each warp writes partial results
        if( idx == warp_last ) {
                ptr[warpid] = warp_total;
                hd[warpid] = warp_flag;
        }
        __syncthreads ();
        // Step 3: One warp scans the per-warp results
        if( warpid == 0 ) segscan_warp(ptr, hd, idx);
        __syncthreads ();
        // Step 4: Accumulate results from
        if( warpid != 0 && will_accumulate)
                val = log_plus(ptr[warpid -1], val);
        __syncthreads ();
        ptr[idx] = val;
        __syncthreads ();
}

__global__ void kernel_1(int* array, int size, bool* key)
{
	int idx=threadIdx.x;
	int stt=blockIdx.x;
	segscan_block(array+stt*1838, key+stt*1838, idx);
	__syncthreads();
	if(threadIdx.x==0&&key[1024+stt*1838]==0)
	{
		key[1024+stt*1838]=1;
		array[1024+stt*1838]+=array[1023+stt*1838];
	}
	__syncthreads();
	if(threadIdx.x+1024<size)
	{
		segscan_block(array+1024+stt*1838, key+1024+stt*1838, idx);
	}
}
int main()
{
	int* data;
	bool* keys;
	int* vals;
	int* thrust_keys;
	int n=10;
	int array_size=1838*n;
	int num_of_rule=1838;
	printf("%f\n",-std::numeric_limits<float>::max());
	cudaMallocManaged(&thrust_keys, array_size*sizeof(int));
	cudaMallocManaged(&data, array_size*sizeof(int));
	cudaMallocManaged(&keys, array_size*sizeof(bool));
	cudaMallocManaged(&vals, array_size*sizeof(int));
	
	for(int i=0;i<num_of_rule;i++)
	{
		for(int j=0;j<n;j++)
		{
			data[i+j*num_of_rule] = 1;
			thrust_keys[i+j*num_of_rule]=i/10;
			if(i%10==0) keys[i+j*num_of_rule]=1;
			else keys[i+j*num_of_rule]=0;
		}
	}

	thrust::equal_to<int> binary_pred;
	for(int i=0;i<n;i++)
	thrust::inclusive_scan_by_key(thrust_keys+i*num_of_rule, thrust_keys +i*num_of_rule+ 1838, data+i*num_of_rule, vals+i*num_of_rule, binary_pred, add_);
	dim3 grid(n);
	dim3 block(1024);
	dim3 block2(1838-1024);
	kernel_1<<< grid, block >>>(data, 1838, keys);
	cudaDeviceSynchronize();
	for(int i=0;i<1838*n;i++) {
		if(vals[i]!=data[i]) printf("%d ", i);
		if((vals[i]-1)%10!=(i%1838)%10) printf("%d vs %d\n", vals[i], i);
	}
	printf("\n");
	return 0;
}
