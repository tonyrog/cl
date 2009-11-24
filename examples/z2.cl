//
// Calculate mandelbrot
// f(0) = x+yi
// f(n) = f(n)^2 + c
//

__kernel void z2(const float x, const float y, 
		 const float xs, const float ys, 
		 const unsigned int n,
		 __global unsigned int* out)
{
    int i = get_global_id(0);
    int j = get_global_id(0);
    if ((i < n) && (j < n)) {
	int k = 0;
	float cx = x + i*xs;
	float cy = y + j*ys;
	float a = 0, b = 0;
	float a2 = 0, b2 = 0;

	while ((k < n) && ((a2 + b2) < 4)) {
	    a = a2-b2 + cx;
	    b = 2*a*b + cy;
	    a2 = a*a;
	    b2 = b*b;
	    k++;
	}
	out[i*n + j] = k;
    }
}
