#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#include <iostream>
using namespace std;


namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

		__global__ void kernUpSweep(int n, int d, int *data) {
			// old implementation
			//int index = threadIdx.x + (blockIdx.x * blockDim.x);
			//if (index >= n) {
			//	return;
			//}
			//int twoPowerDplus1 = 1 << (d + 1);
			//int twoPowerD = 1 << d;


			//if (index % twoPowerDplus1 == 0) {
			//	data[index + twoPowerDplus1 - 1] += data[index + twoPowerD - 1];
			//}

			// optimized implementation
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) {
				return;
			}
			int twoPowerDplus1 = 1 << (d + 1);
			int twoPowerD = 1 << d;

			data[index * twoPowerDplus1 + twoPowerDplus1 - 1] += data[index * twoPowerDplus1 + twoPowerD - 1];

		}

		__global__ void kernDownSweep(int n, int d, int *data) {
			// old implementation
			//int index = threadIdx.x + (blockIdx.x * blockDim.x);
			//if (index >= n) {
			//	return;
			//}

			//int twoPowerDplus1 = 1 << (d + 1);
			//int twoPowerD = 1 << d;

			//if (index % twoPowerDplus1 == 0) {
			//	int temp = data[index + twoPowerD - 1];
			//	data[index + twoPowerD - 1] = data[index + twoPowerDplus1 - 1];
			//	data[index + twoPowerDplus1 - 1] += temp;
			//}

			// optimized implementation
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) {
				return;
			}

			int twoPowerDplus1 = 1 << (d + 1);
			int twoPowerD = 1 << d;

			int temp = data[index * twoPowerDplus1 + twoPowerD - 1];
			data[index * twoPowerDplus1 + twoPowerD - 1] = data[index * twoPowerDplus1 + twoPowerDplus1 - 1];
			data[index * twoPowerDplus1 + twoPowerDplus1 - 1] += temp;
		}

		void efficientScan(int n, int depth, int *dev_tempData) {
			// old implementation
			//dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

			//for (int d = 0; d < depth; ++d) {
			//	kernUpSweep <<<fullBlocksPerGrid, blockSize >>> (n, d, dev_tempData);
			//}

			//cudaMemset(dev_tempData + n - 1, 0, sizeof(int));
			//for (int d = depth - 1; d >= 0; --d) {
			//	kernDownSweep << <fullBlocksPerGrid, blockSize >> > (n, d, dev_tempData);
			//}

			// optimized implementation
			for (int d = 0; d < depth; ++d) {
				int twoPowerDPluse1 = 1 << (d + 1);
				dim3 fullBlocksPerGrid((n / twoPowerDPluse1  + blockSize - 1) / blockSize);
				kernUpSweep <<<fullBlocksPerGrid, blockSize >>> (n / twoPowerDPluse1, d, dev_tempData);
			}

			cudaMemset(dev_tempData + n - 1, 0, sizeof(int));
			for (int d = depth - 1; d >= 0; --d) {
				int twoPowerDPluse1 = 1 << (d + 1);
				dim3 fullBlocksPerGrid((n / twoPowerDPluse1 + blockSize - 1) / blockSize);
				kernDownSweep << <fullBlocksPerGrid, blockSize >> > (n / twoPowerDPluse1, d, dev_tempData);
			}

		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			int depth = ilog2ceil(n);
			int totalLength = 1 << depth;

			int *dev_tempData;
			int *dataWithPadding = new int[totalLength];
			std::memcpy(dataWithPadding, idata, n * sizeof(int));

			for (int i = n; i < totalLength; ++i) {;
				dataWithPadding[i] = 0;
			}

			cudaMalloc((void**)&dev_tempData, totalLength * sizeof(int));
			cudaMemcpy(dev_tempData, dataWithPadding, totalLength * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            // TODO
			efficientScan(totalLength, depth, dev_tempData);

            timer().endGpuTimer();

			cudaMemcpy(odata, dev_tempData, n * sizeof(int), cudaMemcpyDeviceToHost);
			cudaFree(dev_tempData);

			delete[] dataWithPadding;
        }

		__global__ void kernCompact(int n, int *odata, int *mapping, int *scan ) {

		}

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
			// array size round up to power of two
			int depth = ilog2ceil(n);
			int totalLength = 1 << depth;
			int *initIndices = new int[totalLength];

			for (int i = 0; i < totalLength; ++i) {
				initIndices[i] = 0;
			}

			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

			int *dev_bools;
			int *dev_indices;
			int *dev_idata;
			int *dev_odata;


			cudaMalloc((void**)&dev_bools, n * sizeof(int));
			cudaMalloc((void**)&dev_indices, totalLength * sizeof(int));
			cudaMalloc((void**)&dev_idata, n * sizeof(int));
			cudaMalloc((void**)&dev_odata, n * sizeof(int));

			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(dev_indices, initIndices, totalLength * sizeof(int), cudaMemcpyHostToDevice);


            timer().startGpuTimer();
            // TODO
			// mapping
			StreamCompaction::Common::kernMapToBoolean <<<fullBlocksPerGrid, blockSize >>> (n, dev_bools, dev_idata);

			// scann
			cudaMemcpy(dev_indices, dev_bools, n * sizeof(int), cudaMemcpyDeviceToDevice);
			efficientScan(totalLength, depth, dev_indices);

			// scatter
			StreamCompaction::Common::kernScatter << <fullBlocksPerGrid, blockSize >> > (n, dev_odata, dev_idata, dev_bools, dev_indices);

            timer().endGpuTimer();

			cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
			int count = 0;
			for (int i = 0; i < n; i++) {
				if (odata[i]) {
					count++;
				} else {
					break;
				}
			}

			cudaFree(dev_bools);
			cudaFree(dev_indices);
			cudaFree(dev_idata);
			cudaFree(dev_odata);

			delete[]  initIndices;
            return count;
        }
    }
}
