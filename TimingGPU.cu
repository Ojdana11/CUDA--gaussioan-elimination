/*   Gaussian Elimination.
*    
*    Copyright (C) 2012-2013 Orange Owl Solutions.  
*
*    This file is part of Bluebird Library.
*    Gaussian Elimination is free software: you can redistribute it and/or modify
*    it under the terms of the Lesser GNU General Public License as published by
*    the Free Software Foundation, either version 3 of the License, or
*    (at your option) any later version.
*
*    Gaussian Elimination is distributed in the hope that it will be useful,
*    but WITHOUT ANY WARRANTY; without even the implied warranty of
*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*    Lesser GNU General Public License for more details.
*
*    You should have received a copy of the GNU General Public License
*    along with Gaussian Elimination.  If not, see <http://www.gnu.org/licenses/>.
*
*
*    For any request, question or bug reporting please visit http://www.orangeowlsolutions.com/
*    or send an e-mail to: info@orangeowlsolutions.com
*
*
*/


/**************/
/* TIMING GPU */
/**************/

#include "TimingGPU.cuh"

#include <cuda.h>
#include <cuda_runtime.h>

struct PrivateTimingGPU {
	cudaEvent_t		start;
	cudaEvent_t		stop;
};

// default constructor
TimingGPU::TimingGPU() { privateTimingGPU = new PrivateTimingGPU; }

// default destructor
TimingGPU::~TimingGPU() { }

void TimingGPU::StartCounter()
{
	cudaEventCreate(&((*privateTimingGPU).start));
	cudaEventCreate(&((*privateTimingGPU).stop));
	cudaEventRecord((*privateTimingGPU).start,0);
}

void TimingGPU::StartCounterFlags()
{
	int eventflags = cudaEventBlockingSync;

	cudaEventCreateWithFlags(&((*privateTimingGPU).start),eventflags);
	cudaEventCreateWithFlags(&((*privateTimingGPU).stop),eventflags);
	cudaEventRecord((*privateTimingGPU).start,0);
}

// Gets the counter in ms
float TimingGPU::GetCounter()
{
	float	time;
	cudaEventRecord((*privateTimingGPU).stop, 0);
	cudaEventSynchronize((*privateTimingGPU).stop);
	cudaEventElapsedTime(&time,(*privateTimingGPU).start,(*privateTimingGPU).stop);
	return time;
}

