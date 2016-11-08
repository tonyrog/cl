/* -*- c -*-
 *
 */

float4 pixel_over(float4 a, float4 b);
float4 pixel_blend(float4 a, float4 b);

float4 pixel_blend(float4 a, float4 b)
{
    return a.w*a + (1-a.w)*b;
}

float4 pixel_over(float4 a, float4 b)
{
    return a.w*a + (1-a.w)*b.w*b;
}

kernel void pixmap_over(read_write image2d_t a,
			read_write image2d_t b,
			read_write image2d_t c,
			uint w, uint h)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if ((x < (int)w) && (y < (int)h)) {
	int2 coord = {x,y};
	float4 ap = read_imagef(a, coord);
	float4 bp = read_imagef(b, coord);
	float4 cp = pixel_over(ap, bp);
	write_imagef(c, coord, cp);
    }
}

kernel void pixmap_blend(read_write image2d_t a,
			 read_write image2d_t b,
			 read_write image2d_t c,
			 uint w, uint h)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if ((x < (int)w) && (y < (int)h)) {
	int2 coord = {x,y};
	float4 ap = read_imagef(a, coord);
	float4 bp = read_imagef(b, coord);
	float4 cp = pixel_blend(ap, bp);
	write_imagef(c, coord, cp);
    }
}
