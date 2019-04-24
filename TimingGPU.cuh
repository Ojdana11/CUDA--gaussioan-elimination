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

#ifndef __TIMING_CUH__
#define __TIMING_CUH__

/**************/
/* TIMING GPU */
/**************/

// Events are a part of CUDA API and provide a system independent way to measure execution times on CUDA devices with approximately 0.5 
// microsecond precision.

struct PrivateTimingGPU;

class TimingGPU
{
	private:
		PrivateTimingGPU *privateTimingGPU;

	public:
			
		TimingGPU();

		~TimingGPU();

		void StartCounter();
		void StartCounterFlags();

		float GetCounter();

}; // TimingCPU class

#endif 
