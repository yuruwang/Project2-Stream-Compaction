#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"


namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
		__global__ void kernNaiveScan(int n, int d, int* odata, const int* idata) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) {
				return;
			}

			int dist = 1 << (d - 1);
			if (index >= dist) {
				odata[index] = idata[index - dist] + idata[index];
			} else{
				odata[index] = idata[index];
			}


		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			int* dev_idata;
			int* dev_odata;
			cudaMalloc((void**)&dev_idata, n * sizeof(int));
			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			dim3 fullBlocksPerGrid((n + blockSize - 1)/ blockSize);

			int depth = ilog2ceil(n);
            timer().startGpuTimer();
            // TODO
			for (int d = 1; d <= depth; ++d) {
				kernNaiveScan <<<fullBlocksPerGrid, blockSize>>> (n, d, dev_odata, dev_idata);
				std::swap(dev_idata, dev_odata);
			}
			std::swap(dev_idata, dev_odata);

			cudaMemcpy(odata + 1, dev_odata, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);
			odata[0] = 0;
            timer().endGpuTimer();

			cudaFree(dev_idata);
			cudaFree(dev_odata);
        }

    }
}
