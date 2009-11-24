//
// Multiply count 4x4 matrices with a constant matrix
//

__kernel void mul4x4(__global float* input,
		     __global float* output,
		     const float16 a,
		     const unsigned int count)
{
    int i;

    i = get_global_id(0);
    if (i < count) {
        int j,k;
        __global float* b = input  + i*16;
	__global float* c = output + i*16;

	for (i=0; i<3; i++) {
	    for (j=0; j<3; j++) {
	    	float e = 0.0;
		for (k=0; k<3; k++)
		    e += a[3*i+k]*b[3*k+j];
		c[3*i+j] = e;
	    }
	}
    }
}


	 
