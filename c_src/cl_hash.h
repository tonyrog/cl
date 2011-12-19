#ifndef __ECL_HASH_H__
#define __ECL_HASH_H__

#include <stdint.h>

typedef uintptr_t lhash_value_t;

typedef struct _lhash_bucket_t {
    struct _lhash_bucket_t* next;
    lhash_value_t hvalue;
} lhash_bucket_t;

typedef struct {
    lhash_value_t (*hash)(void*);  // calculate hash
    int (*cmp)(void*, void*);      // compare data items
    void (*release)(void*);        // data release (free)
    void* (*copy)(void*);          // copy (may be used with insert)
} lhash_func_t;

typedef struct {
    lhash_func_t func;         // functions

    int is_allocated;
    char* name;

    unsigned int thres;        // Medium bucket chain len, for grow
    unsigned int szm;          // current size mask
    unsigned int nactive;      // Number of "active" slots
    unsigned int nslots;       // Total number of slots
    unsigned int nitems;       // Total number of items
    unsigned int p;            // Split position
    unsigned int nsegs;        // Number of segments
    unsigned int n_resize;     // Number of index realloc calls
    unsigned int n_seg_alloc;  // Number of segment allocations
    unsigned int n_seg_free;   // Number of segment destroy
    lhash_bucket_t*** seg;
} lhash_t;

extern lhash_t* lhash_new(char* name, int thres, lhash_func_t* func);
extern lhash_t* lhash_init(lhash_t* lh, char* name, int thres,
			   lhash_func_t* func);
extern void  lhash_delete(lhash_t* lh);
extern void* lhash_lookup(lhash_t* lh, void* key);
extern void* lhash_insert(lhash_t* lh, void* key, void* data);
extern void* lhash_insert_new(lhash_t* lh, void* key, void* data);
extern void* lhash_erase(lhash_t* lh, void* key);
extern void  lhash_each(lhash_t* lh,
			void (elem)(lhash_t* lh, void* elem, void* arg),
			void* arg);
extern void lhash_Info(lhash_t* lh);

#endif
