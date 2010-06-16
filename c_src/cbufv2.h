/*
 * control buffer managment
 *
 */
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <memory.h>
#include <unistd.h>
#include <sys/types.h>
#ifdef DARWIN
#include <machine/endian.h>
#endif

#define CBUF_USE_PUT_ETF   // pack ETF 
#define CBUF_USE_PUT_CTI   // pack CTI 

#define CBUF_FLAG_HEAP     0x01  // allocated heap memory
#define CBUF_FLAG_BINARY   0x02  // ErlDrvBinary 
#define CBUF_FLAG_PUT_CTI  0x00  // Put CTI data (default)
#define CBUF_FLAG_PUT_ETF  0x10  // Put ETF data
#define CBUF_FLAG_PUT_TRM  0x20  // Put ErlDrvTerm data
#define CBUF_FLAG_PUT_MASK 0x30  // put type value selection

#define CBUF_VEC_SIZE      4     // static vector size
#define CBUF_SEG_EXTRA     256
//
//
//
typedef struct {
    u_int8_t  flags;  // allocation status (HEAP|BINARY)
    u_int8_t* base;   // base pointer
    size_t    size;   // allocated length of segment
    size_t    len;    // used length of segment
    ErlDrvBinary* bp; // reference when segment is in a binary
} cbuf_segment_t;

typedef struct
{
    u_int8_t  flags;         // allocation flags
    size_t    ip;            // poistion in current segment
    size_t    iv;
    size_t    vlen;         // used length of v
    size_t    vsize;        // actual length of v
    cbuf_segment_t* v;      // current segment
    cbuf_segment_t  dv[CBUF_VEC_SIZE];
} cbuf_t;

// CBUF Tag Interface (CTI) (always used by cl_drv)
#define OK             1  // 'ok'
#define ERROR          2  // 'error'
#define EVENT          3  // 'event'
#define INT8           4   // int8_t
#define UINT8          5   // u_int8_t 
#define INT16          6   // int16_t
#define UINT16         7   // u_int16_t
#define INT32          8   // int32_t
#define UINT32         9   // u_int32_t
#define INT64          10  // int64_t
#define UINT64         11  // u_int64_t
#define BOOLEAN        12  // u_int8_t
#define FLOAT32        13  // float
#define FLOAT64        14  // double
#define STRING1        15  // len byte followed by UTF-8 chars 
#define STRING4        16  // 4-byte len followed by UTF-8 string 
#define ATOM           17  // len bytes followed by ASCII chars
#define BINARY         18  // binary 4-byte len followed by Octets
#define LIST           19  // list begin
#define LIST_END       20  // list end 
#define TUPLE          21  // tuple begin
#define TUPLE_END      22  // tuple end 
#define ENUM           23  // Encoded as INT32
#define BITFIELD       24  // Encoded as UINT64
#define HANDLE         25  // Encoded pointer 32/64 bit

// External Term Format (ETF)
// Version 131
#ifdef CBUF_USE_PUT_ETF

#define VERSION_MAGIC 131
#define SMALL_INTEGER_EXT 97  // 'a'
#define INTEGER_EXT       98  // 'b'
#define FLOAT_EXT         99  // 'c'
#define ATOM_EXT          100 // 'd'
#define SMALL_ATOM_EXT    115 // 's'
#define REFERENCE_EXT     101 // 'e'
#define NEW_REFERENCE_EXT 114 // 'r'
#define PORT_EXT          102 // 'f'
#define NEW_FLOAT_EXT     70  // 'F'
#define PID_EXT           103 // 'g'
#define SMALL_TUPLE_EXT   104 // 'h'
#define LARGE_TUPLE_EXT   105 // 'i'
#define NIL_EXT           106 // 'j'
#define STRING_EXT        107 // 'k'
#define LIST_EXT          108 // 'l'
#define BINARY_EXT        109 // 'm'
#define BIT_BINARY_EXT    77 // 'M'
#define SMALL_BIG_EXT     110 // 'n'
#define LARGE_BIG_EXT     111 // 'o'
#define NEW_FUN_EXT       112 // 'p'
#define EXPORT_EXT        113 // 'q'
#define FUN_EXT           117 // 'u'
#define DIST_HEADER       68  // 'D'
#define ATOM_CACHE_REF    82 // 'R'
#define COMPRESSED        80 // 'P'

#endif

// Debug
void cbuf_print(cbuf_t* cp,char* name)
{
    size_t i;
    FILE* f = stderr;

    fprintf(f,"cbuf %s = {\r\n", name);
    fprintf(f,"  flags:");
    if (cp->flags & CBUF_FLAG_BINARY)	fprintf(f," binary");
    if (cp->flags & CBUF_FLAG_HEAP)	fprintf(f," heap");
    fprintf(f,"\r\n");
    fprintf(f,"     iv: %lu\r\n", cp->iv);
    fprintf(f,"     ip: %lu\r\n", cp->ip);
    fprintf(f,"  vsize: %lu\r\n", cp->vsize);
    fprintf(f,"   vlen: %lu\r\n", cp->vlen);
    fprintf(f,"     dv: %s\r\n", (cp->v == cp->dv) ? "true" : "false");
    for (i = 0; i < cp->vlen; i++) {
	fprintf(f,"    v[%lu].flags:", i);
	if (cp->v[i].flags & CBUF_FLAG_BINARY)	fprintf(f," binary");
	if (cp->v[i].flags & CBUF_FLAG_HEAP)	fprintf(f," heap");
	fprintf(f,"\r\n");
	fprintf(f,"    v[%lu].base = %p\r\n",  i, cp->v[i].base);
	fprintf(f,"    v[%lu].size = %lu\r\n", i, cp->v[i].size);
	fprintf(f,"    v[%lu].len  = %lu\r\n", i, cp->v[i].len);
	fprintf(f,"    v[%lu].bp   = %p\r\n", i,  cp->v[i].bp);
    }
    fprintf(f,"};\r\n");
}

// copy src to dst. native-endian to big-endian
// src is a buffer holding a number in native endian order 
// dst is a buffer holding a number in big endian order
//
static inline void* memcpy_n2b(void* dst, void* src, size_t len)
{
#if BYTE_ORDER == BIG_ENDIAN
    return memcpy(dst, src, len);
#else
    u_int8_t* sp = ((u_int8_t*) src) + len;
    u_int8_t* dp = (u_int8_t*) dst;
    while(len--)
	*dp++ = *--sp;
    return dst;
#endif
}

// copy src to dst. native-endian to little-endian
// src is a buffer holding a number in native endian order 
// dst is a buffer holding a number in little endian order
//
static inline void* memcpy_n2l(void* dst, void* src, size_t len)
{
#if BYTE_ORDER == LITTLE_ENDIAN
    return memcpy(dst, src, len);
#else
    u_int8_t* sp = ((u_int8_t*) src) + len;
    u_int8_t* dp = (u_int8_t*) dst;
    while(len--)
	*dp++ = *--sp;
    return dst;
#endif
}

// Number of bytes written/read to current segment
static inline size_t cbuf_seg_used(cbuf_t* cp)
{
    return cp->ip;
}

// Return a pointer to current poistion
static inline u_int8_t* cbuf_seg_ptr(cbuf_t* cp)
{
    return (u_int8_t*) (cp->v[cp->iv].base + cp->ip);
}

// Number of byte available to read in current segment
static inline size_t cbuf_seg_r_avail(cbuf_t* cp)
{
    return (cp->iv >= cp->vlen) ? 0 : (cp->v[cp->iv].len - cp->ip);
}

// Total number of byte available for read
static size_t cbuf_r_avail(cbuf_t* cp) 
{
    size_t sz = cbuf_seg_r_avail(cp);
    size_t i = cp->iv + 1;
    while(i < cp->vlen) {
	sz += cp->v[i].len;
	i++;
    }
    return sz;
}

// Number of byte available to write in current segment
static inline size_t cbuf_seg_w_avail(cbuf_t* cp)
{
    return (cp->iv >= cp->vlen) ? 0 : (cp->v[cp->iv].size - cp->ip);
}

// return 1 if at end of buf 0 otherwise
static inline int cbuf_eob(cbuf_t* cp)
{
    return (cp->iv >= cp->vlen) || 
	((cp->iv == cp->vlen-1) && (cbuf_seg_r_avail(cp) == 0));
}

// Adjust position if end of segment to next segment
static inline void cbuf_adjust_r_ip(cbuf_t* cp)
{
    if (cp->ip >= cp->v[cp->iv].len) {
	cp->iv++;
	cp->ip = 0;
    }
}

// Adjust position if end of segment to next segment
static inline void cbuf_adjust_w_ip(cbuf_t* cp)
{
    if (cp->ip >= cp->v[cp->iv].size) {
	cp->iv++;
	cp->ip = 0;
    }
}


// Rest the cbuf to start & set read flag
static inline void cbuf_reset(cbuf_t* cp, u_int32_t skip)
{
    cp->iv = 0;
    cp->ip = 0;
    while(skip > cp->v[cp->iv].len) {
	skip -= cp->v[cp->iv].len;
	cp->iv++;
    }
    cp->ip = skip;
}

// resize (grow) current segment
static u_int8_t* cbuf_seg_realloc(cbuf_t* cp, size_t need)
{
    cbuf_segment_t* sp = &cp->v[cp->iv];
    size_t new_size;

    if (sp->len + need <= sp->size) {
	sp->len += need;
	return sp->base + cp->ip;
    }
    new_size = sp->size + need + CBUF_SEG_EXTRA;
    if (sp->flags & CBUF_FLAG_BINARY) { 
	// Data is allocated in ErlDrvBinary
	if (sp->bp) {
	    ErlDrvBinary* bp;
	    // fprintf(stderr, "realloc_binary: %lu\r\n", new_size);
	    if (!(bp = driver_realloc_binary(sp->bp, new_size)))
		return 0;
	    sp->bp = bp;
	}
	else {
	    // fprintf(stderr, "alloc_binary: %lu\r\n", new_size);
	    if (!(sp->bp = driver_alloc_binary(new_size)))
		return 0;
	    memcpy(sp->bp->orig_bytes, sp->base, sp->len);
	}
	sp->base = (u_int8_t*) sp->bp->orig_bytes;
    }
    else if (sp->flags & CBUF_FLAG_HEAP) {
	// Data is already dynamic binaries not used
	u_int8_t* dp;
	// fprintf(stderr, "realloc: %lu\r\n", new_size);
	if (!(dp = driver_realloc(sp->base, new_size)))
	    return 0;
	sp->base = dp;
    }
    else {
	// Move data from static buffer to dynamic
	u_int8_t* base = sp->base;
	u_int8_t* dp;

	// fprintf(stderr, "alloc: %lu\r\n", new_size);
	if (!(dp = driver_alloc(new_size)))
	    return 0;
	sp->base = dp;
	memcpy(sp->base, base, sp->len);
	sp->flags |= CBUF_FLAG_HEAP;
    }
    sp->size = new_size;
    return sp->base + cp->ip;
}

// grow the segment vector
static int cbuf_vec_grow(cbuf_t* cp)
{
    size_t vsize = 2*cp->vsize;
    cbuf_segment_t* sp;

    if (cp->v == cp->dv) {
	if (!(sp = driver_alloc(sizeof(cbuf_segment_t)*vsize)))
	    return 0;
	memcpy(sp,cp->dv,CBUF_VEC_SIZE*sizeof(cbuf_segment_t));
    }
    else {
	if (!(sp = driver_realloc(cp->v,sizeof(cbuf_segment_t)*vsize)))
	    return 0;
    }
    cp->v = sp;	
    cp->vsize = vsize;
    return 1;
}

// Terminate current segment (patch iov_len)
// add new segment and increase iv
static int cbuf_seg_add(cbuf_t* cp)
{
    cp->v[cp->iv].len = cp->ip;
    cp->iv++;
    cp->ip = 0;
    if (cp->iv >= cp->vlen) {
	cp->vlen++;
	if (cp->vlen >= cp->vsize) {
	    if (!cbuf_vec_grow(cp))
		return 0;
	}
	memset(&cp->v[cp->iv], 0, sizeof(cbuf_segment_t));
    }
    return 1;
}

// Allocate len contigous bytes in current segment
static inline u_int8_t* cbuf_seg_alloc(cbuf_t* cp, size_t len)
{
    u_int8_t* ptr;

    if (cbuf_seg_w_avail(cp) < len) {
	if (!cbuf_seg_realloc(cp, len))
	    return 0;
    }
    ptr = cbuf_seg_ptr(cp);
    cp->ip += len;
    cp->v[cp->iv].len = cp->ip;
    return ptr;
}

// segmented read & handle end of segment pointer
static int cbuf_seg_read(cbuf_t* cp, void* ptr, size_t len)
{
    size_t n;
    while((cp->iv < cp->vlen) && len && ((n=cbuf_seg_r_avail(cp)) < len)) {
	memcpy(ptr, cp->v[cp->iv].base + cp->ip, n);
	ptr = ((u_int8_t*) ptr) + n;
	len -= n;
	cp->iv++;
	cp->ip = 0;
    }
    if (cbuf_seg_r_avail(cp) < len)
	return 0;
    memcpy(ptr, cp->v[cp->iv].base + cp->ip, len);
    cp->ip += len;
    cbuf_adjust_r_ip(cp);
    return 1;
}

// read data from cbuf into ptr,len
static inline int cbuf_read(cbuf_t* cp, void* ptr, size_t len)
{
    if (cbuf_seg_r_avail(cp) > len) { // fast case
	memcpy(ptr, cp->v[cp->iv].base + cp->ip, len);
	cp->ip += len;
	return 1;
    }
    return cbuf_seg_read(cp, ptr, len);
}

//
// Write data into segments
// FIXME: add code to expand segments
//
static int cbuf_seg_write(cbuf_t* cp, void* ptr, size_t len)
{
    size_t n;
    while((cp->iv < cp->vlen) && len && ((n=cbuf_seg_w_avail(cp)) < len)) {
	memcpy(cp->v[cp->iv].base+cp->ip, ptr, n);
	ptr = ((u_int8_t*) ptr) + n;
	cp->v[cp->iv].len = cp->ip + n;
	len -= n;
	cp->iv++;
	cp->ip = 0;
    }
    if (cbuf_seg_w_avail(cp) < len)
	return 0;
    memcpy(cp->v[cp->iv].base + cp->ip, ptr, len);
    cp->ip += len;
    cp->v[cp->iv].len = cp->ip;
    cbuf_adjust_w_ip(cp);
    return 1;    
}

	    
// copy tag and data in ptr,len to cbuf, fix fill vector version
static inline int cbuf_twrite(cbuf_t* cp, u_int8_t tag, void* ptr, size_t len)
{
    u_int8_t* p;

    if (!(p = cbuf_seg_alloc(cp, 1+len)))
	return 0;
    p[0] = tag;
    switch(len) {
    case 4: p[4] = ((u_int8_t*)ptr)[3];
    case 3: p[3] = ((u_int8_t*)ptr)[2];
    case 2: p[2] = ((u_int8_t*)ptr)[1];
    case 1: p[1] = ((u_int8_t*)ptr)[0];
    case 0: break;
    default: memcpy(p+1, ptr, len); break;
    }
    return 1;
}

// write data to cbuf, fix fill vector version
static inline int cbuf_write(cbuf_t* cp, void* ptr, size_t len)
{
    u_int8_t* p;

    if (!(p = cbuf_seg_alloc(cp, len)))
	return 0;
    switch(len) {
    case 4: p[3] = ((u_int8_t*)ptr)[3];
    case 3: p[2] = ((u_int8_t*)ptr)[2];
    case 2: p[1] = ((u_int8_t*)ptr)[1];
    case 1: p[0] = ((u_int8_t*)ptr)[0];
    case 0: break;
    default: memcpy(p, ptr, len); break;
    }
    return 1;
}

//
// Initialize read/write buffer
// for write buffer 
//   flags = 0      => data will be resized with malloc
//   flags = HEAP   => data will be resize with realloc
//   flags = BINARY => data will be resize with binary_alloc
//   flags = PUT_CTI => put CTI format 
//   flags = PUT_ETF => put ETF format
//
static void cbuf_init(cbuf_t* cp, void* buf, size_t len, 
		      size_t skip, u_int8_t flags)
{
    cp->flags = (flags & CBUF_FLAG_PUT_MASK);
    cp->v     = cp->dv;
    cp->vlen  = 1;
    cp->vsize = CBUF_VEC_SIZE;
    
    cp->v[0].flags = flags;
    cp->v[0].base  = buf;
    cp->v[0].len   = len;
    cp->v[0].size  = len;
    cp->v[0].bp    = 0;

    cp->iv    = 0;           // current vector index
    cp->ip    = skip;        // current position in current vector
}

// IOV read only (or copy on write?)
static void cbuf_initv(cbuf_t* cp, ErlIOVec* vec)
{
    int i;
    cp->flags = 0;
    if (vec->vsize > CBUF_VEC_SIZE)
	cp->v = driver_alloc(sizeof(cbuf_segment_t)*vec->vsize);
    else
	cp->v = cp->dv;
    cp->vsize = vec->vsize;
    cp->vlen  = vec->vsize;
    for (i = 0; i < vec->vsize; i++) {
	cp->v[i].flags = 0;
	cp->v[i].base  = (u_int8_t*) vec->iov[i].iov_base;
	cp->v[i].size  = vec->iov[i].iov_len;
	cp->v[i].len   = vec->iov[i].iov_len;
	cp->v[i].bp    = vec->binv[i];
    }
    cp->iv = 0;
    cp->ip = 0;
}


// Create cbuf as a binary 
static cbuf_t* cbuf_new_bin(u_int8_t* buf,size_t len,size_t skip)
{
    cbuf_t* cp;
    ErlDrvBinary* bp;

    if (!(cp = (cbuf_t*) driver_alloc(sizeof(cbuf_t))))
	return 0;
    if (!(bp = driver_alloc_binary(len))) {
	driver_free(cp);
	return 0;
    }
    cbuf_init(cp,bp->orig_bytes,len,skip,CBUF_FLAG_BINARY);
    cp->flags = CBUF_FLAG_HEAP;  // cp is on heap
    cp->v[0].bp = bp; // the binary ref (after init!)
    if (buf) memcpy(cp->v[0].base, buf, len);
    return cp;
}

/* allocate a combi cbuf_t and buffer (non growing) */
static cbuf_t* cbuf_new(u_int8_t* buf, u_int32_t len, u_int32_t skip)
{
    cbuf_t* cp;
    char*   bp;

    if (!(cp = (cbuf_t*) driver_alloc(sizeof(cbuf_t))))
	return 0;
    if (!(bp = driver_alloc(len))) {
	driver_free(cp);
	return 0;
    }
    cbuf_init(cp,bp,len,skip,CBUF_FLAG_HEAP);
    cp->flags = CBUF_FLAG_HEAP;
    if (buf) memcpy(cp->v[0].base, buf, len);
    return cp;
}

//
// Cleanup dynamically created vectors etc
// 
static void cbuf_final(cbuf_t* cp)
{
    size_t i;
    for (i = 0; i < cp->vlen; i++) {
	cbuf_segment_t* sp = &cp->v[i];
	
	if (sp->flags & CBUF_FLAG_BINARY) {
	    if (sp->bp)
		driver_free_binary(sp->bp);
	}
	else if (sp->flags & CBUF_FLAG_HEAP)
	    driver_free(sp->base);
    }
    if (cp->v != cp->dv)
	driver_free(cp->v);
}

static inline void cbuf_free(cbuf_t* cp)
{
    cbuf_final(cp);
    if (cp->flags & CBUF_FLAG_HEAP)
	driver_free(cp);
}

// Trim buffer to used size (when binary)
// The control interface wont use the size return in the case
// of an allocated binary. THIS IS A BUG (I think)  
// FIXME: a bit dangerous since I do not know what the orig_size
//        the real fix is to reallocate!
static inline void cbuf_trim(cbuf_t* cp)
{
    if (cp->v[cp->iv].bp)
	cp->v[cp->iv].bp->orig_size = cbuf_seg_used(cp);
}

/* add "raw" data to cbuf_t buffer */
static inline void cbuf_add(cbuf_t* cp, u_int8_t* buf, u_int32_t len)
{
    u_int8_t* ptr = cbuf_seg_alloc(cp, len);
    memcpy(ptr, buf, len);
}

// skip "data" (reading) moving ptr forward 
static void cbuf_forward(cbuf_t* cp, size_t len)
{
    while(cp->iv < cp->vlen) {
	size_t n = cbuf_seg_r_avail(cp);
	if (n >= len) {
	    cp->ip += len;
	    cbuf_adjust_r_ip(cp);
	    return;
	}
	len -= n;
	cp->iv++;
	cp->ip = 0;
    }
}

// skip backward 
static void cbuf_backward(cbuf_t* cp, size_t len)
{
    while(len) {
	size_t n = cbuf_seg_used(cp);
	if (n >= len) {
	    cp->ip -= len;
	    return;
	}
	len -= n;
	if (cp->iv == 0) {
	    cp->ip = 0;
	    return;
	}
	cp->iv--;
	cp->ip = cp->v[cp->iv].len;
    }
    cbuf_adjust_r_ip(cp);
}

/*****************************************************************************
 *
 * PUT tagged data
 *
 *****************************************************************************/
static inline size_t cbuf_sizeof(u_int8_t tag)
{
    switch (tag) {
    case BOOLEAN:   return sizeof(u_int8_t);
    case UINT8:     return sizeof(u_int8_t);
    case UINT16:    return sizeof(u_int16_t);
    case UINT32:    return sizeof(u_int32_t);
    case UINT64:    return sizeof(u_int64_t);
    case STRING1:   return 0;  // variable
    case LIST:      return 0;  // variable
    case LIST_END:  return 0;  // variable
    case TUPLE:     return 0;  // variable
    case TUPLE_END: return 0;  // variable
    case ATOM:      return 0;  // variable
    case BINARY:    return 0;  // variable
    case INT8:      return sizeof(int8_t);
    case INT16:     return sizeof(int16_t);
    case INT32:     return sizeof(int32_t);
    case INT64:     return sizeof(int64_t);
    case FLOAT32:   return sizeof(float);
    case FLOAT64:   return sizeof(double);
    case STRING4:   return 0;
    case ENUM:      return sizeof(int32_t);
    case BITFIELD:  return sizeof(int64_t);
    case HANDLE:    return sizeof(intptr_t);
    default: return 0;
    }
}

#ifdef CBUF_USE_PUT_CTI

static inline int cbuf_cti_put_boolean(cbuf_t* cp, u_int8_t value)
{
    return cbuf_twrite(cp, BOOLEAN, &value, sizeof(value));
}

static inline int cbuf_cti_put_int8(cbuf_t* cp, int8_t value)
{
    return cbuf_twrite(cp, INT8, &value, sizeof(value));
}

static inline int cbuf_cti_put_int16(cbuf_t* cp, int16_t value)
{
    return cbuf_twrite(cp, INT16, &value, sizeof(value));
}

static inline int cbuf_cti_put_int32(cbuf_t* cp, int32_t value)
{
    return cbuf_twrite(cp, INT32, &value, sizeof(value));
}

static inline int cbuf_cti_put_int64(cbuf_t* cp, int64_t value)
{
    return cbuf_twrite(cp, INT64, &value, sizeof(value));
}

static inline int cbuf_cti_put_float32(cbuf_t* cp, float value)
{
    return cbuf_twrite(cp, FLOAT32, &value, sizeof(value));
}

static inline int cbuf_cti_put_float64(cbuf_t* cp, double value)
{
    return cbuf_twrite(cp, FLOAT64, &value, sizeof(value));
}

static inline int cbuf_cti_put_uint8(cbuf_t* cp, u_int8_t value)
{
    return cbuf_twrite(cp, UINT8, &value, sizeof(value));
}

static inline int cbuf_cti_put_uint16(cbuf_t* cp, u_int16_t value)
{
    return cbuf_twrite(cp, UINT16, &value, sizeof(value));
}

static inline int cbuf_cti_put_uint32(cbuf_t* cp, u_int32_t value)
{
    return cbuf_twrite(cp, UINT32, &value, sizeof(value));
}

static inline int cbuf_cti_put_uint64(cbuf_t* cp, u_int64_t value)
{
    return cbuf_twrite(cp, UINT64, &value, sizeof(value));
}

/* put special tag like TUPLE/LIST/TUPLE_END/TUPLE_END 
 * REPLY_OK/REPLY_ERROR/REPLY_EVENT etc
 */

static inline int cbuf_cti_put_tuple_begin(cbuf_t* cp, size_t n)
{
    (void) n;
    return cbuf_twrite(cp, TUPLE, 0, 0);    
}

static inline int cbuf_cti_put_tuple_end(cbuf_t* cp)
{
    return cbuf_twrite(cp, TUPLE_END, 0, 0);    
}

static inline int cbuf_cti_put_list_begin(cbuf_t* cp, size_t n)
{
    (void) n;
    return cbuf_twrite(cp, LIST, 0, 0);        
}

static inline int cbuf_cti_put_list_end(cbuf_t* cp)
{
    return cbuf_twrite(cp, LIST_END, 0, 0);
}

static inline int cbuf_cti_put_begin(cbuf_t* cp)
{
    (void) cp;
    return 1;
}

static inline int cbuf_cti_put_end(cbuf_t* cp)
{
    (void) cp;
    return 1;
}

static inline int cbuf_cti_put_tag_ok(cbuf_t* cp)
{
    return cbuf_twrite(cp, OK, 0, 0);
}

static inline int cbuf_cti_put_tag_error(cbuf_t* cp)
{
    return cbuf_twrite(cp, ERROR, 0, 0);
}

static inline int cbuf_cti_put_tag_event(cbuf_t* cp)
{
    return cbuf_twrite(cp, EVENT, 0, 0);
}

static inline int cbuf_cti_put_atom(cbuf_t* cp, const char* atom)
{
    u_int8_t* ptr;
    u_int32_t n = strlen(atom);

    if (n > 0xff) n = 0xff; // truncate error?
    if (!(ptr = cbuf_seg_alloc(cp, n+2)))
	return 0;
    ptr[0] = ATOM;
    ptr[1] = n;
    memcpy(&ptr[2], atom, n);
    return 1;
}

static inline int cbuf_cti_put_string(cbuf_t* cp, const char* string, int n)
{
    u_int8_t* ptr;

    if ((string == NULL) || (n == 0)) {
	if (!(ptr = cbuf_seg_alloc(cp, 2)))
	    return 0;
	ptr[0] = STRING1;
	ptr[1] = 0;
    }
    else {
	if (n <= 0xff) {
	    if (!(ptr = cbuf_seg_alloc(cp, n+2)))
		return 0;
	    ptr[0] = STRING1;
	    ptr[1] = n;
	    memcpy(&ptr[2], string, n);
	}
	else {
	    u_int32_t len = n;
	    if (!(ptr = cbuf_seg_alloc(cp, n+5)))
		return 0;
	    ptr[0] = STRING4;
	    memcpy(&ptr[1], &len, sizeof(len));
	    memcpy(&ptr[5], string, n);
	}
    }
    return 1;
}

static inline int cbuf_cti_put_binary(cbuf_t* cp, const u_int8_t* buf, u_int32_t len)
{
    u_int8_t* ptr;

    if (!(ptr = cbuf_seg_alloc(cp, len+5)))
	return 0;
    ptr[0] = BINARY;
    memcpy(ptr+1, &len, sizeof(len));
    memcpy(ptr+5, buf, len);
    return 1;
}
#endif // CBUF_USE_PUT_CTI


#ifdef CBUF_USE_PUT_ETF
//
// ETF implementation of reply data
//
static inline int etf_put_uint8(cbuf_t* cp, u_int8_t value)
{
    u_int8_t* p;
    if (!(p = cbuf_seg_alloc(cp, 2)))
	return 0;    
    p[0] = SMALL_INTEGER_EXT;    
    p[1] = value;
    return 1;
}

static inline int etf_put_int32(cbuf_t* cp, int32_t value)
{
    u_int8_t* p;
    if (!(p = cbuf_seg_alloc(cp, 5)))
	return 0;
    p[0] = INTEGER_EXT;
    memcpy_n2b(&p[1], &value, 4);
    return 1;
}

static inline int etf_put_u64(cbuf_t* cp,u_int8_t sign,u_int64_t value)
{
    u_int8_t* p;

    if (!(p = cbuf_seg_alloc(cp, 11)))
	return 0;
    p[0] = SMALL_BIG_EXT;
    p[1] = 8;
    p[2] = sign;
    memcpy_n2l(&p[3], &value, 8);
    return 1;
}

static inline int etf_put_int64(cbuf_t* cp, int64_t value)
{
    if (value < 0)
	return etf_put_u64(cp, 1, (u_int64_t) -value);
    else
	return etf_put_u64(cp, 0, (u_int64_t) value);
}

static inline int etf_put_float(cbuf_t* cp, double value)
{
    u_int8_t* p;
    if (!(p = cbuf_seg_alloc(cp, 9)))
	return 0;
    p[0] = NEW_FLOAT_EXT;
    memcpy_n2b(&p[1], &value, 8);
    return 1;
}


static inline int etf_put_atom(cbuf_t* cp, const char* atom, size_t len)
{
    u_int8_t* p;
    if (len > 255) len = 255;
    if (!(p = cbuf_seg_alloc(cp, len+2)))
	return 0;    
    p[0] = SMALL_ATOM_EXT;
    p[1] = len;
    memcpy(&p[2], atom, len);
    return 1;
}

static inline int cbuf_etf_put_atom(cbuf_t* cp, const char* atom)
{
    size_t n = strlen(atom);
    return etf_put_atom(cp, atom, n);
}

static inline int cbuf_etf_put_boolean(cbuf_t* cp, u_int8_t value)
{
    if (value) 
	return etf_put_atom(cp, "true", 4);
    else
	return etf_put_atom(cp, "false", 5);
}

static inline int cbuf_etf_put_int8(cbuf_t* cp, int8_t value)
{
    if (value >= 0)
	return etf_put_uint8(cp, (u_int8_t) value);
    else
	return etf_put_int32(cp, (int32_t) value);
}

static inline int cbuf_etf_put_int16(cbuf_t* cp, int16_t value)
{
    return etf_put_int32(cp, (int32_t) value);
}

static inline int cbuf_etf_put_int32(cbuf_t* cp, int32_t value)
{
    return etf_put_int32(cp, value);
}

static inline int cbuf_etf_put_int64(cbuf_t* cp, int64_t value)
{
    return etf_put_int64(cp, value);
}

static inline int cbuf_etf_put_float64(cbuf_t* cp, double value)
{
    return etf_put_float(cp, value);
}

static inline int cbuf_etf_put_float32(cbuf_t* cp, float value)
{
    return etf_put_float(cp, (double) value);
}

static inline int cbuf_etf_put_uint8(cbuf_t* cp, u_int8_t value)
{
    return etf_put_uint8(cp, value);
}

static inline int cbuf_etf_put_uint16(cbuf_t* cp, u_int16_t value)
{
    return etf_put_int32(cp, (int32_t) value);
}

static inline int cbuf_etf_put_uint32(cbuf_t* cp, u_int32_t value)
{
    if (value > 0x7fffffff)
	return etf_put_u64(cp, 0, (uint64_t) value);
    else
	return etf_put_int32(cp, (int32_t) value);
}

static inline int cbuf_etf_put_uint64(cbuf_t* cp, u_int64_t value)
{
    return etf_put_u64(cp, 0, value);
}

static inline int cbuf_etf_put_begin(cbuf_t* cp)
{
    u_int8_t* p;
    if (!(p = cbuf_seg_alloc(cp, 1)))
	return 0;
    p[0] = VERSION_MAGIC;
    return 1;
}

static inline int cbuf_etf_put_end(cbuf_t* cp)
{
    (void) cp;
    return 1;
}


static inline int cbuf_etf_put_tuple_begin(cbuf_t* cp, size_t n)
{
    u_int8_t* p;
    if (n > 0xFF) {
	if (!(p = cbuf_seg_alloc(cp, 5)))
	    return 0;
	p[0] = LARGE_TUPLE_EXT;
	memcpy_n2b(&p[1], &n, 4);
    }
    else {
	if (!(p = cbuf_seg_alloc(cp, 2)))
	    return 0;
	p[0] = SMALL_TUPLE_EXT;
	p[1] = n;
    }
    return 1;
}

static inline int cbuf_etf_put_tuple_end(cbuf_t* cp)
{
    (void) cp;
    return 1;
}

static inline int cbuf_etf_put_list_begin(cbuf_t* cp, size_t n)
{
    u_int8_t* p;
    if (!(p = cbuf_seg_alloc(cp, 5)))
	return 0;
    p[0] = LIST_EXT;
    memcpy_n2b(&p[1], &n, 4);
    return 1;
}

// proper list end!
static inline int cbuf_etf_put_list_end(cbuf_t* cp)
{
    u_int8_t* p;
    if (!(p = cbuf_seg_alloc(cp, 1)))
	return 0;
    p[0] = NIL_EXT;
    return 1;
}

static inline int cbuf_etf_put_tag_ok(cbuf_t* cp)
{
    return etf_put_atom(cp, "ok", 2);
}

static inline int cbuf_etf_put_tag_error(cbuf_t* cp)
{
    return etf_put_atom(cp, "error", 5);
}

static inline int cbuf_etf_put_tag_event(cbuf_t* cp)
{
    return etf_put_atom(cp, "event", 5);
}

static inline int cbuf_etf_put_string(cbuf_t* cp, const char* string, int n)
{
    u_int8_t* p;

    if ((string == NULL) || (n == 0)) {
	if (!(p = cbuf_seg_alloc(cp, 1)))
	    return 0;
	p[0] = NIL_EXT;
    }
    else {
	if (n > 0xFFFF) n = 0xFFFF; // warn?
	if (!(p = cbuf_seg_alloc(cp, n+3)))
	    return 0;
	p[0] = STRING_EXT;
	p[1] = n>>8;
	p[2] = n;
	memcpy(&p[3], string, n);
    }
    return 1;
}

// FIXME - if vectored interface add as binary part
static inline int cbuf_etf_put_binary(cbuf_t* cp, const u_int8_t* buf, 
				      u_int32_t len)
{
    u_int8_t* p;

    if (!(p = cbuf_seg_alloc(cp, len+5)))
	return 0;
    p[0] = BINARY_EXT;
    memcpy_n2b(&p[1], &len, 4);
    memcpy(&p[5], buf, len);
    return 1;
}

#endif // CBUF_USE_PUT_ETF

// Select ETF or CTI both in runtime and compile time
#if defined(CBUF_USE_PUT_ETF) && defined(CBUF_USE_PUT_CTI)

#define cbuf_put(what,cp) (						\
	(((cp)->flags & CBUF_FLAG_PUT_MASK) == CBUF_FLAG_PUT_ETF) ?	\
	(cbuf_etf_put_##what((cp))) :					\
	((((cp)->flags & CBUF_FLAG_PUT_MASK) == CBUF_FLAG_PUT_CTI) ?	\
	 (cbuf_cti_put_##what((cp))) : 0))

#define cbuf_put_value(what,cp,arg) (					\
	(((cp)->flags & CBUF_FLAG_PUT_MASK) == CBUF_FLAG_PUT_ETF) ?	\
	(cbuf_etf_put_##what((cp),(arg))) :				\
	((((cp)->flags & CBUF_FLAG_PUT_MASK) == CBUF_FLAG_PUT_CTI) ?	\
	 (cbuf_cti_put_##what((cp),(arg))) : 0))

#define cbuf_put_value2(what,cp,arg1,arg2) (				\
	(((cp)->flags & CBUF_FLAG_PUT_MASK) == CBUF_FLAG_PUT_ETF) ?	\
	(cbuf_etf_put_##what((cp),(arg1),(arg2))) :			\
	((((cp)->flags & CBUF_FLAG_PUT_MASK) == CBUF_FLAG_PUT_CTI) ?	\
	 (cbuf_cti_put_##what((cp),(arg1),(arg2))) : 0))

#elif defined(CBUF_USE_PUT_ETF)

#define cbuf_put(what,cp) 				 \
    (cbuf_etf_put_##what((cp)))
#define cbuf_put_value(what,cp,arg) 				 \
    (cbuf_etf_put_##what((cp),(arg)))
#define cbuf_put_value2(what,cp,arg1,arg2)		\
    (cbuf_etf_put_##what((cp),(arg1),(arg2)))

#elif defined(CBUF_USE_PUT_CTI)

#define cbuf_put(what,cp) 				 \
    (cbuf_cit_put_##what((cp)))
#define cbuf_put_value(what,cp,arg) 				 \
    (cbuf_cti_put_##what((cp),(arg)))
#define cbuf_put_value2(what,cp,arg1,arg2)	\
    (cbuf_cti_put_##what((cp),(arg1),(arg2)))

#else 
#error "must use either CTI or ETF"
#endif

static inline int cbuf_put_boolean(cbuf_t* cp, u_int8_t value)
{
    return cbuf_put_value(boolean, cp, value);
}

static inline int cbuf_put_int8(cbuf_t* cp, int8_t value)
{
    return cbuf_put_value(int8, cp, value);
}

static inline int cbuf_put_int16(cbuf_t* cp, int16_t value)
{
    return cbuf_put_value(int16, cp, value);
}

static inline int cbuf_put_int32(cbuf_t* cp, int32_t value)
{
    return cbuf_put_value(int32, cp, value);
}

static inline int cbuf_put_int64(cbuf_t* cp, int64_t value)
{
    return cbuf_put_value(int64, cp, value);
}
static inline int cbuf_put_float32(cbuf_t* cp, float value)
{
    return cbuf_put_value(float32, cp, value);
}
static inline int cbuf_put_float64(cbuf_t* cp, double value)
{
    return cbuf_put_value(float64, cp, value);
}
static inline int cbuf_put_uint8(cbuf_t* cp, u_int8_t value)
{
    return cbuf_put_value(uint8, cp, value);
}
static inline int cbuf_put_uint16(cbuf_t* cp, u_int16_t value)
{
    return cbuf_put_value(uint16, cp, value);
}

static inline int cbuf_put_uint32(cbuf_t* cp, u_int32_t value)
{
    return cbuf_put_value(uint32, cp, value);
}

static inline int cbuf_put_uint64(cbuf_t* cp, u_int64_t value)
{
    return cbuf_put_value(uint64, cp, value);
}

static inline int cbuf_put_atom(cbuf_t* cp, const char* value)
{
    return cbuf_put_value(atom, cp, value);
}

static inline int cbuf_put_tuple_begin(cbuf_t* cp, size_t n)
{
    return cbuf_put_value(tuple_begin, cp, n);
}

static inline int cbuf_put_tuple_end(cbuf_t* cp) 
{
    return cbuf_put(tuple_end, cp);
}

static inline int cbuf_put_list_begin(cbuf_t* cp, size_t n)
{
    return cbuf_put_value(list_begin, cp, n);
}

static inline int cbuf_put_list_end(cbuf_t* cp)
{
    return cbuf_put(list_end, cp);
}

static inline int cbuf_put_begin(cbuf_t* cp)
{
    return cbuf_put(begin, cp);
}

static inline int cbuf_put_end(cbuf_t* cp)
{
    return cbuf_put(end, cp);
}

static inline int cbuf_put_tag_ok(cbuf_t* cp)
{
    return cbuf_put(tag_ok, cp);
}

static inline int cbuf_put_tag_error(cbuf_t* cp)
{
    return cbuf_put(tag_error, cp);
}
static inline int cbuf_put_tag_event(cbuf_t* cp)
{
    return cbuf_put(tag_event, cp);
}

static inline int cbuf_put_string(cbuf_t* cp, const char* value, int n)
{
    return cbuf_put_value2(string, cp, value, n);
}

static inline int cbuf_put_binary(cbuf_t* cp, const u_int8_t* buf, u_int32_t len)
{
    return cbuf_put_value2(binary, cp, buf, len);
}


/*****************************************************************************
 *
 * GET untagged data
 *
 *****************************************************************************/

static inline int get_boolean(cbuf_t* cp, u_int8_t* val)
{
    u_int8_t v;
    if (cbuf_read(cp, &v, sizeof(*val))) {
	*val = (v != 0);
	return 1;
    }
    return 0;
}

static inline int get_uint8(cbuf_t* cp, u_int8_t* val)
{
    return cbuf_read(cp, val, sizeof(*val));
}

static inline int get_uint16(cbuf_t* cp, u_int16_t* val)
{
    return cbuf_read(cp, val, sizeof(*val));
}

static inline int get_int32(cbuf_t* cp, int32_t* val)
{
    return cbuf_read(cp, val, sizeof(*val));
}

static inline int get_uint32(cbuf_t* cp, u_int32_t* val)
{
    return cbuf_read(cp, val, sizeof(*val));
}


static inline int get_uint64(cbuf_t* cp, u_int64_t* val)
{
    return cbuf_read(cp, val, sizeof(*val));
}

static inline int get_float32(cbuf_t* cp, float* val)
{
    return cbuf_read(cp, val, sizeof(*val));
}

static inline int get_float64(cbuf_t* cp, double* val)
{
    return cbuf_read(cp, val, sizeof(*val));
}
