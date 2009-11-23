//
// CBUF testing
//
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

#include "erl_driver.h"
#define driver_alloc(size) malloc((size))
#define driver_realloc(ptr,size) realloc((ptr),(size))
#define driver_free(ptr) free((ptr))

ErlDrvBinary* driver_alloc_binary(int sz) 
{
    ErlDrvBinary* bp = driver_alloc(sizeof(ErlDrvBinary)+sz);
    bp->orig_size = sz;
    return bp;
}

ErlDrvBinary* driver_realloc_binary(ErlDrvBinary* bp, int sz) 
{
    bp = driver_realloc(bp, sz);
    bp->orig_size = sz;
    return bp;
}
void driver_free_binary(ErlDrvBinary* bp) 
{
    driver_free(bp);
}

#include "cbufv2.h"

u_int8_t  vu8[]  = {1,2,3,4,5,6,7,8,9,10};
u_int16_t vu16[] = {11,12,13,14,15,16,17,18,19,20};
u_int32_t vu32[] = {21,22,23,24,25,26,27,28,29,30};
u_int64_t vu64[] = {31,32,33,34,35,36,37,38,39,40};

float  vf32[] = {41.0,42.0,43.0,44.0,45.0,46.0,47.0,48.0,49.0,50.0};
double vf64[] = {51.0,52.0,53.0,54.0,55.0,56.0,57.0,58.0,59.0,60.0};

// Aligned not wrapped
SysIOVec iov1[6] =
{
    { (char*)vu8, sizeof(vu8) },
    { (char*)vu16, sizeof(vu16) },
    { (char*)vu32, sizeof(vu32) },
    { (char*)vu64, sizeof(vu64) },
    { (char*)vf32, sizeof(vf32) },
    { (char*)vf64, sizeof(vf64) }
};
    
ErlDrvBinary* binv1[6] = { 0, 0, 0, 0, 0, 0};

ErlIOVec vec1 = 
{
    6,
    (sizeof(vu8)+sizeof(vu16)+
     sizeof(vu32)+sizeof(vu64)+
     sizeof(vf32)+sizeof(vf64)),
    iov1,
    binv1
};

u_int8_t vx[] = { 1, 0, 0, 0, 2, 0, 0, 0, 3, 0 };
u_int8_t vy[] = { 0, 0, 4, 0, 0, 0, 5, 0 };
u_int8_t vz[] = { 0, 0, 6, 0, 7 };

SysIOVec iov2[3] =
{
    { (char*)vx, sizeof(vx) },
    { (char*)vy, sizeof(vy) },
    { (char*)vz, sizeof(vz) }
};
    
ErlDrvBinary* binv2[3] = { 0, 0, 0};

ErlIOVec vec2 = 
{
    3,
    (sizeof(vx)+sizeof(vy)+sizeof(vz)),
    iov2,
    binv2
};



void print_u8(cbuf_t* in, int n)
{
    int i;
    for (i = 0; i < n; i++) {
	u_int8_t val;
	get_uint8(in, &val);
	printf("%u ", val);
    }
    printf("\n");
}

void print_u16(cbuf_t* in, int n)
{
    int i;
    for (i = 0; i < n; i++) {
	u_int16_t val;
	get_uint16(in, &val);
	printf("%u ", val);
    }
    printf("\n");
}

void print_u32(cbuf_t* in, int n)
{
    int i;
    for (i = 0; i < n; i++) {
	u_int32_t val;
	get_uint32(in, &val);
	printf("%u ", val);
    }
    printf("\n");
}

void print_u64(cbuf_t* in, int n)
{
    int i;
    for (i = 0; i < n; i++) {
	u_int64_t val;
	get_uint64(in, &val);
	printf("%llu ", val);
    }
    printf("\n");
}

void print_f32(cbuf_t* in, int n)
{
    int i;
    for (i = 0; i < n; i++) {
	float val;
	get_float32(in, &val);
	printf("%f ", val);
    }
    printf("\n");
}


void print_f64(cbuf_t* in, int n)
{
    int i;
    for (i = 0; i < n; i++) {
	double val;
	get_float64(in, &val);
	printf("%f ", val);
    }
    printf("\n");
}

void read_buffer_test()
{
    cbuf_t in;
    printf("read_buffer_test vu8: BEGIN\n");
    cbuf_init(&in, &vu8, sizeof(vu8), 0, 0);
    print_u8(&in, 10);
    printf("read_buffer_test vu8: END%s\n",
	   !cbuf_eob(&in) ? " (data not consumed)" : "");
    cbuf_final(&in);

    printf("read_buffer_test vu16: BEGIN\n");
    cbuf_init(&in, &vu16, sizeof(vu16), 0, 0);
    print_u16(&in, 10);
    printf("read_buffer_test vu16: END%s\n",
	   !cbuf_eob(&in) ? " (data not consumed)" : "");
    cbuf_final(&in);

    printf("read_buffer_test vu32: BEGIN\n");
    cbuf_init(&in, &vu32, sizeof(vu32), 0, 0);
    print_u32(&in, 10);
    printf("read_buffer_test vu32: END%s\n",
	   !cbuf_eob(&in) ? " (data not consumed)" : "");
    cbuf_final(&in);

    printf("read_buffer_test vu64: BEGIN\n");
    cbuf_init(&in, &vu64, sizeof(vu64), 0, 0);
    print_u64(&in, 10);
    printf("read_buffer_test vu64: END%s\n",
	   !cbuf_eob(&in) ? " (data not consumed)" : "");
    cbuf_final(&in);

    printf("read_buffer_test vf32: BEGIN\n");
    cbuf_init(&in, &vf32, sizeof(vf32), 0, 0);
    print_f32(&in, 10);
    printf("read_buffer_test vu32: END%s\n",
	   !cbuf_eob(&in) ? " (data not consumed)" : "");
    cbuf_final(&in);

    printf("read_buffer_test vf64: BEGIN\n");
    cbuf_init(&in, &vf64, sizeof(vf64), 0, 0);
    print_f64(&in, 10);
    printf("read_buffer_test vf64: END%s\n",
	   !cbuf_eob(&in) ? " (data not consumed)" : "");
    cbuf_final(&in);
}

void read_vector_test()
{
    cbuf_t in;

    printf("read_vector_test1: BEGIN\n");
    cbuf_initv(&in, &vec1);
    cbuf_print(&in, "vec1");
    print_u8(&in, 10);
    print_u16(&in, 10);
    print_u32(&in, 10);
    print_u64(&in, 10);
    print_f32(&in, 10);
    print_f64(&in, 10);
    printf("read_vector_test1: END%s\n",
	   !cbuf_eob(&in) ? " (data not consumed)" : "");
    cbuf_print(&in, "vec1");
    cbuf_final(&in);

    printf("read_vector_test2: BEGIN\n");
    cbuf_initv(&in, &vec2);
    cbuf_print(&in, "vec2");
    print_u32(&in, 5);    
    print_u16(&in, 1);
    print_u8(&in, 1);
    printf("read_vector_test2: END%s\n",
	   !cbuf_eob(&in) ? " (data not consumed)" : "");
    cbuf_print(&in, "vec2");
    cbuf_final(&in);
}

void write_buffer_test()
{
    cbuf_t out;
    u_int8_t small_buf[10];
    u_int8_t i8;
    u_int16_t i16;

    printf("write_buffer_test small_buf: BEGIN\n");
    cbuf_init(&out, small_buf, sizeof(small_buf), 0, 0);
    for (i8 = 1; i8 <= 10; i8++)
	cbuf_write(&out, &i8, sizeof(i8));
    cbuf_reset(&out, 0);
    print_u8(&out, 10);
    cbuf_print(&out, "out_i8");
    cbuf_final(&out);
    printf("write_buffer_test: small_buf: END\n");

    printf("write_buffer_test alloc_buf: BEGIN\n");
    cbuf_init(&out, small_buf, sizeof(small_buf), 0, 0);
    for (i8 = 1; i8 <= 20; i8++)
	cbuf_write(&out, &i8, sizeof(i8));
    cbuf_reset(&out, 0);
    print_u8(&out, 20);
    cbuf_print(&out, "out_i8");
    cbuf_final(&out);
    printf("write_buffer_test: alloc_buf: END\n");

    printf("write_buffer_test realloc_buf: BEGIN\n");
    cbuf_init(&out, small_buf, sizeof(small_buf), 0, 0);
    cbuf_print(&out, "out_i16");
    for (i16 = 1; i16 <= 200; i16++)
	cbuf_write(&out, &i16, sizeof(i16));
    cbuf_reset(&out, 0);
    print_u16(&out, 200);
    cbuf_print(&out, "out_i16");
    cbuf_final(&out);
    printf("write_buffer_test: realloc_buf: END\n");

    // the same with empty inital buffer
    printf("write_buffer_test realloc_buf2: BEGIN\n");
    cbuf_init(&out, 0, 0, 0, 0);
    cbuf_print(&out, "out_i16");
    for (i16 = 1; i16 <= 200; i16++)
	cbuf_write(&out, &i16, sizeof(i16));
    cbuf_reset(&out, 0);
    print_u16(&out, 200);
    cbuf_print(&out, "out_i16");
    cbuf_final(&out);
    printf("write_buffer_test: realloc_buf2: END\n");

    printf("write_buffer_test binary_realloc_buf: BEGIN\n");
    cbuf_init(&out, small_buf, sizeof(small_buf), 0, CBUF_FLAG_BINARY);
    cbuf_print(&out, "out_i16");
    for (i16 = 1; i16 <= 200; i16++)
	cbuf_write(&out, &i16, sizeof(i16));
    cbuf_reset(&out, 0);
    print_u16(&out, 200);
    cbuf_print(&out, "out_i16");
    cbuf_final(&out);
    printf("write_buffer_test: binary_realloc_buf: END\n");

    

}

void write_vec_test()
{
    cbuf_t out;
    u_int8_t small_buf[10];
    u_int16_t i16;

    printf("write_vec_test1: binary BEGIN\n");
    cbuf_init(&out, small_buf, sizeof(small_buf), 0, CBUF_FLAG_BINARY);
    for (i16 = 1; i16 <= 20; i16++)
	cbuf_write(&out, &i16, sizeof(i16));
    cbuf_seg_add(&out);    
    for (i16 = 21; i16 <= 40; i16++)
	cbuf_write(&out, &i16, sizeof(i16));
    cbuf_seg_add(&out);    
    for (i16 = 41; i16 <= 60; i16++)
	cbuf_write(&out, &i16, sizeof(i16));
    cbuf_reset(&out, 0);
    print_u16(&out, 60);
    cbuf_print(&out, "out_i16");
    cbuf_final(&out);
    printf("write_vec_test1: binary END\n");    


    printf("write_vec_test2: BEGIN\n");
    cbuf_init(&out, small_buf, sizeof(small_buf), 0, 0);
    for (i16 = 1; i16 <= 20; i16++)
	cbuf_write(&out, &i16, sizeof(i16));
    cbuf_seg_add(&out);    
    for (i16 = 21; i16 <= 40; i16++)
	cbuf_write(&out, &i16, sizeof(i16));
    cbuf_seg_add(&out);    
    for (i16 = 41; i16 <= 60; i16++)
	cbuf_write(&out, &i16, sizeof(i16));
    cbuf_seg_add(&out);    
    for (i16 = 61; i16 <= 80; i16++)
	cbuf_write(&out, &i16, sizeof(i16));
    cbuf_seg_add(&out);    
    for (i16 = 81; i16 <= 100; i16++)
	cbuf_write(&out, &i16, sizeof(i16));
    cbuf_reset(&out, 0);
    print_u16(&out, 100);
    cbuf_print(&out, "out_i16");
    cbuf_final(&out);
    printf("write_vec_test2: END\n");    
}


main()
{
    read_buffer_test();
    read_vector_test();

    write_buffer_test();
    write_vec_test();

    exit(0);
}
