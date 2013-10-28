//
// Multiply count 4x4 matrices with a constant matrix
//

__kernel void mul4x4(__global float* input,
		     __global float* output,
		     const float16 a,
		     const unsigned int count)
{
    int ix;
    __global float* b;
    __global float* c;
    
    ix = get_global_id(0);
    if (ix < count) {
        int i,j,k;

	b = input  + ix*16;
	c = output + ix*16;

	for (i=0; i<4; i++) {
	    for (j=0; j<4; j++) {
	        float s1 = 0.0;
		for (k=0; k<4; k++) {
		    float t1 = a.s(4*i+k);
		    float t2 = b[4*k+j];
		    s1 += (t1*t2);
		}
    		c[4*i+j] = s1;
	    }
	}
	
    }
}


	 
