//
// Multiply count 4x4 matrices with a constant matrix
//

__kernel void
mul4x4 (__global float*    input,
        __global float*    output,
        const float16      a,
        const unsigned int count)
{
    size_t ix;
    __global float* b;
    __global float* c;

    ix = get_global_id(0);
    if (ix < count) {
        b = input  + 16 * ix;
        c = output + 16 * ix;

        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                float s1 = 0.0;
                for (int k = 0; k < 4; ++k) {
                    float t1 = a[4*i + k];
                    float t2 = b[4*k + j];
                    s1 += (t1 * t2);
                }
                c[4*i + j] = s1;
            }
        }
    }
}
