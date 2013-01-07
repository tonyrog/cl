/****** BEGIN COPYRIGHT *******************************************************
 *
 * Copyright (C) 2007 - 2012, Rogvall Invest AB, <tony@rogvall.se>
 *
 * This software is licensed as described in the file COPYRIGHT, which
 * you should have received as part of this distribution. The terms
 * are also available at http://www.rogvall.se/docs/copyright.txt.
 *
 * You may opt to use, copy, modify, merge, publish, distribute and/or sell
 * copies of the Software, and permit persons to whom the Software is
 * furnished to do so, under the terms of the COPYRIGHT file.
 *
 * This software is distributed on an "AS IS" basis, WITHOUT WARRANTY OF ANY
 * KIND, either express or implied.
 *
 ****** END COPYRIGHT ********************************************************/
/*
** Linear hash 
*/
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>

#include "cl_hash.h"

#define LHASH_SZEXP   8
#define LHASH_SEGSZ   (1 << LHASH_SZEXP)
#define LHASH_SZMASK  ((1 << LHASH_SZEXP)-1)

#define LHASH_SEG(i)  ((i)>>LHASH_SZEXP)
#define LHASH_POS(i)  ((i)&LHASH_SZMASK)

#define LHASH_SEG_LEN         256   /* When growing init segs */
#define LHASH_SEG_INCREAMENT  128   /* Number of segments to grow */

#define LHASH_BUCKET(lh, i) (lh)->seg[LHASH_SEG(i)][LHASH_POS(i)]

#define LHASH_IX(lh, hval) \
    (((((hval) & (lh)->szm)) < (lh)->p) ? \
       ((hval) & (((lh)->szm << 1) | 1)) : \
       (((hval) & (lh)->szm)))

#ifndef WIN32
#define INLINE inline
#else
#define INLINE
#endif

static lhash_bucket_t** lhash_alloc_seg(int seg_sz)
{
    lhash_bucket_t** bp;
    int sz = sizeof(lhash_bucket_t*)*seg_sz;

    bp = (lhash_bucket_t**) malloc(sz);
    memset(bp, 0, sz);
    return bp;
}

INLINE static lhash_bucket_t** lhash_HLOOKUP(lhash_t* lh,
					     lhash_value_t hval,
					     void* key)
{
    int ix = LHASH_IX(lh, hval);
    lhash_bucket_t** bpp = &LHASH_BUCKET(lh, ix);
    lhash_bucket_t* b = *bpp;

    while(b != (lhash_bucket_t*) 0) {
	if ((b->hvalue == hval) && (lh->func.cmp(key, (void*) b) == 0))
	    return bpp;
	bpp = &b->next;
	b = b->next;
    }
    return bpp;
}

/* scan bucket for key return bucket */
INLINE static lhash_bucket_t** lhash_LOOKUP(lhash_t* lh, void* key)
{
    return lhash_HLOOKUP(lh, lh->func.hash(key), key);
}


lhash_t* lhash_init(lhash_t* lh, char* name, int thres, lhash_func_t* func)
{
    lhash_bucket_t*** bp;

    if (!(bp = (lhash_bucket_t***) malloc(sizeof(lhash_bucket_t**))))
	return 0;
    lh->func    = *func;
    lh->is_allocated = 0;
    lh->name = name;
    lh->thres = thres;
    lh->szm = LHASH_SZMASK;
    lh->nactive = LHASH_SEGSZ;
    lh->nitems = 0;
    lh->p = 0;
    lh->nsegs = 1;
    lh->seg = bp;
    lh->seg[0] = lhash_alloc_seg(LHASH_SEGSZ);
    lh->nslots = LHASH_SEGSZ;
    lh->n_seg_alloc = 1;
    lh->n_seg_free  = 0;
    lh->n_resize    = 0;
    return lh;
}


static void lhash_grow(lhash_t* lh)
{
    lhash_bucket_t** bp;
    lhash_bucket_t** bps;
    lhash_bucket_t* b;
    unsigned int ix;
    unsigned int nszm = (lh->szm << 1) | 1;

    if (lh->nactive >= lh->nslots) {
	/* Time to get a new array */
	if (LHASH_POS(lh->nactive) == 0) {
	    unsigned int six = LHASH_SEG(lh->nactive);
	    if (six == lh->nsegs) {
		int i, sz;

		if (lh->nsegs == 1)
		    sz = LHASH_SEG_LEN;
		else
		    sz = lh->nsegs + LHASH_SEG_INCREAMENT;
		lh->seg = (lhash_bucket_t***) realloc(lh->seg,
						      sizeof(lhash_bucket_t**)*sz);
		lh->nsegs = sz;
		lh->n_resize++;
		for (i = six+1; i < sz; i++)
		    lh->seg[i] = 0;
	    }
	    lh->seg[six] = lhash_alloc_seg(LHASH_SEGSZ);
	    lh->nslots += LHASH_SEGSZ;
	    lh->n_seg_alloc++;
	}
    }

    ix = lh->p;
    bp = &LHASH_BUCKET(lh, ix);
    ix += (lh->szm+1);
    bps = &LHASH_BUCKET(lh, ix);
    b = *bp;

    while (b != 0) {
	ix = b->hvalue & nszm;

	if (ix == lh->p)
	    bp = &b->next;          /* object stay */
	else {
	    *bp = b->next;  	    /* unlink */
	    b->next = *bps;         /* link */
	    *bps = b;
	}
	b = *bp;
    }

    lh->nactive++;
    if (lh->p == lh->szm) {
	lh->p = 0;
	lh->szm = nszm;
    }
    else
	lh->p++;
}

/*
** Shrink the hash table
** Remove segments if they are empty
** but do not reallocate the segment index table !!!
*/
static void lhash_shrink(lhash_t* lh)
{
    lhash_bucket_t** bp;

    if (lh->nactive == LHASH_SEGSZ)
	return;

    lh->nactive--;
    if (lh->p == 0) {
	lh->szm >>= 1;
	lh->p = lh->szm;
    }
    else
	lh->p--;

    bp = &LHASH_BUCKET(lh, lh->p);
    while(*bp != 0) 
	bp = &(*bp)->next;

    *bp = LHASH_BUCKET(lh, lh->nactive);
    LHASH_BUCKET(lh, lh->nactive) = 0;

    if ((lh->nactive & LHASH_SZMASK) == LHASH_SZMASK) {
	int six = LHASH_SEG(lh->nactive)+1;

	free(lh->seg[six]);
	lh->seg[six] = 0;
	lh->nslots -= LHASH_SEGSZ;
	lh->n_seg_free++;
    }
}

lhash_t* lhash_new(char* name, int thres, lhash_func_t* func)
{
    lhash_t* tp;

    if (!(tp = (lhash_t*) malloc(sizeof(lhash_t))))
	return 0;
    
    if (!lhash_init(tp, name, thres, func)) {
	free(tp);
	return 0;
    }
    tp->is_allocated = 1;
    return tp;
}


void lhash_delete(lhash_t* lh)
{
    lhash_bucket_t*** sp = lh->seg;
    int n = lh->nsegs;

    while(n--) {
	lhash_bucket_t** bp = *sp;
	if (bp != 0) {
	    int m = LHASH_SEGSZ;
	    while(m--) {
		lhash_bucket_t* p = *bp++;
		while(p != 0) {
		    lhash_bucket_t* next = p->next;
		    if (lh->func.release)
			lh->func.release((void*) p);
		    p = next;
		}
	    }
	    free(*sp);
	}
	sp++;
    }
    free(lh->seg);

    if (lh->is_allocated)
	free(lh);
}

void* lhash_insert_new(lhash_t* lh, void* key, void* data)
{
    lhash_value_t hval = lh->func.hash(key);
    lhash_bucket_t** bpp = lhash_HLOOKUP(lh, hval, key);
    lhash_bucket_t* b = *bpp;

    if (b) {
	/* release data if copy function is not defined */
	if (!lh->func.copy) {
	    if (lh->func.release) lh->func.release(data);
	}
	return 0;
    }
    b = (lhash_bucket_t*) (lh->func.copy ? lh->func.copy(data) : data);
    b->hvalue = hval;
    b->next = *bpp;
    *bpp = b;
    lh->nitems++;

    if ((lh->nitems / lh->nactive) >= lh->thres)
	lhash_grow(lh);
    return (void*) b;
}

void* lhash_Insert(lhash_t* lh, void* key, void* data)
{
    lhash_value_t hval = lh->func.hash(key);
    lhash_bucket_t** bpp = lhash_HLOOKUP(lh, hval, key);
    lhash_bucket_t* b = *bpp;

    if (b) {
	lhash_bucket_t* b_next = b->next;
	if (lh->func.release) lh->func.release(b);
	b = (lhash_bucket_t*) (lh->func.copy ? lh->func.copy(data) : data);
	b->hvalue = hval;
	b->next = b_next;
	*bpp = b;
    }
    else {
	b = (lhash_bucket_t*) (lh->func.copy ? lh->func.copy(data) : data);
	b->hvalue = hval;
	b->next   = 0;
	*bpp = b;
	lh->nitems++;

	if ((lh->nitems / lh->nactive) >= lh->thres)
	    lhash_grow(lh);
    }
    return (void*) b;

}


void* lhash_lookup(lhash_t* lh, void* key)
{
    lhash_bucket_t** bpp = lhash_LOOKUP(lh, key);
    return *bpp;
}

/*
** Erase an item
*/
void* lhash_erase(lhash_t* lh, void* key)
{
    lhash_bucket_t** bpp = lhash_LOOKUP(lh, key);
    lhash_bucket_t* b = *bpp;

    if (b) {
	*bpp = b->next;  /* unlink */
	if (lh->func.release) lh->func.release((void*) b);
	lh->nitems--;
	if ((lh->nitems / lh->nactive) < lh->thres)
	    lhash_shrink(lh);
    }
    return (void*)b;
}

void lhash_each(lhash_t* lh, void (elem)(lhash_t* lh, void* elem, void* arg),
	       void* arg)
{
    int i;
    int nslots = lh->nslots;

    for (i = 0; i < nslots; i++) {
	lhash_bucket_t* list = LHASH_BUCKET(lh, i);
	while(list) {
	    lhash_bucket_t* next = list->next;
	    elem(lh, (void*) list, arg);
	    list = next;
	}
    }
}


void lhash_info(lhash_t* lh)
{
    unsigned int i;
    int depth = 0;

    for (i = 0; i < lh->nslots; i++) {
	lhash_bucket_t* list = LHASH_BUCKET(lh, i);
	int d = 0;

	while(list) {
 	    list = list->next;
	    d++;
	}
	if (d > depth)
	    depth = d;
    }
    printf("  Name: %s\r\n", lh->name);
    printf("  Size: %d\r\n", lh->szm+1);
    printf("Active: %d\r\n", lh->nactive);    
    printf(" Split: %d\r\n", lh->p);
    printf(" Items: %d\r\n", lh->nitems);
    printf(" Slots: %d\r\n", lh->nslots);
    printf("  Segs: %d\r\n", lh->nsegs);
    printf(" Thres: %d\r\n", lh->thres);
    printf(" Ratio: %e\r\n", (float) lh->nitems / (float) lh->nactive);
    printf("   Max: %d\r\n", depth);
    printf("Resize: %d\r\n", lh->n_resize);
    printf(" Alloc: %d\r\n", lh->n_seg_alloc);
    printf("  Free: %d\r\n", lh->n_seg_free);
}
