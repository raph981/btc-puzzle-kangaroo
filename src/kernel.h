// secp256k1 field arithmetic + the kangaroo stepping kernel.
//
// This header is compiled twice: by g++ for the CPU backend and by nvcc for
// the CUDA backend. Everything here is plain 64-bit integer code (the only
// platform-specific piece is MULHI), so the code that runs on the GPU is the
// exact code that the CPU tests exercise.
#pragma once
#include <stdint.h>

typedef uint64_t u64;
typedef uint32_t u32;

#ifdef __CUDACC__
#define DEV __device__ __forceinline__
#define MULHI(a, b) __umul64hi((a), (b))
#define ATOMIC_INC(p) atomicAdd((p), 1u)
#else
#define DEV static inline
#define MULHI(a, b) ((u64)(((unsigned __int128)(a) * (b)) >> 64))
#define ATOMIC_INC(p) ((*(p))++)
#endif

#define NJ 32                    // jump table size (power of two)
#define FE_C 0x1000003D1ULL      // 2^256 - p
#define FE_P0 0xFFFFFFFEFFFFFC2FULL

struct Fe { u64 v[4]; };         // little-endian limbs, always fully reduced mod p

// A distinguished point reported by the kernel.
struct DPRec {
    u32 idx;                     // kangaroo index
    u32 pad;
    u64 x[4];                    // x coordinate
    u64 dlo, dhi;                // travelled distance (128 bit)
};

DEV bool fe_is_zero(const Fe &a) { return (a.v[0] | a.v[1] | a.v[2] | a.v[3]) == 0; }
DEV bool fe_eq(const Fe &a, const Fe &b) {
    return a.v[0] == b.v[0] && a.v[1] == b.v[1] && a.v[2] == b.v[2] && a.v[3] == b.v[3];
}
DEV bool fe_gep(const Fe &a) {
    return a.v[3] == ~0ULL && a.v[2] == ~0ULL && a.v[1] == ~0ULL && a.v[0] >= FE_P0;
}
// r += C (mod 2^256), i.e. subtract p after an overflow
DEV void fe_addc(Fe &r) {
    u64 t = r.v[0] + FE_C;
    u64 c = (t < FE_C);
    r.v[0] = t;
    for (int i = 1; i < 4; i++) { r.v[i] += c; c = c && (r.v[i] == 0); }
}
DEV void fe_add(Fe &r, const Fe &a, const Fe &b) {
    u64 c = 0;
    for (int i = 0; i < 4; i++) {
        u64 bi = b.v[i];
        u64 t = a.v[i] + c; c = (t < c);
        t += bi; c += (t < bi);
        r.v[i] = t;
    }
    if (c || fe_gep(r)) fe_addc(r);
}
DEV void fe_sub(Fe &r, const Fe &a, const Fe &b) {
    u64 bw = 0;
    for (int i = 0; i < 4; i++) {
        u64 ai = a.v[i], bi = b.v[i];
        u64 t = ai - bi; u64 b1 = (ai < bi);
        u64 t2 = t - bw; u64 b2 = (t < bw);
        r.v[i] = t2; bw = b1 | b2;
    }
    if (bw) {                    // add p == subtract C (mod 2^256)
        u64 t = r.v[0];
        r.v[0] = t - FE_C;
        u64 c = (t < FE_C);
        for (int i = 1; i < 4; i++) { t = r.v[i]; r.v[i] = t - c; c = c && (t == 0); }
    }
}
// r = r + a*b + c, c = carry out
DEV void fe_mac(u64 a, u64 b, u64 &r, u64 &c) {
    u64 lo = a * b, hi = MULHI(a, b);
    lo += c; hi += (lo < c);
    r += lo; hi += (r < lo);
    c = hi;
}
// r = t mod p for a 512-bit t (8 limbs); folds the high half using 2^256 == C (mod p)
DEV void fe_reduce(Fe &r, u64 *t) {
    u64 c = 0;
    for (int i = 0; i < 4; i++) fe_mac(t[4 + i], FE_C, t[i], c);
    u64 lo = c * FE_C, hi = MULHI(c, FE_C);
    t[0] += lo; u64 k = (t[0] < lo);
    u64 s = t[1] + hi; u64 k2 = (s < hi);
    s += k; k2 += (s < k); t[1] = s;
    t[2] += k2; k = (t[2] < k2);
    t[3] += k; k = (t[3] < k);
    r.v[0] = t[0]; r.v[1] = t[1]; r.v[2] = t[2]; r.v[3] = t[3];
    if (k) fe_addc(r);
    if (fe_gep(r)) fe_addc(r);
}
DEV void fe_mul(Fe &r, const Fe &a, const Fe &b) {
    u64 t[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i < 4; i++) {
        u64 c = 0;
        for (int j = 0; j < 4; j++) fe_mac(a.v[i], b.v[j], t[i + j], c);
        t[i + 4] = c;
    }
    fe_reduce(r, t);
}
// Squaring: the 6 cross products are computed once and doubled, then the 4
// squares are added - 10 limb multiplications instead of 16.
DEV void fe_sqr(Fe &r, const Fe &a) {
    u64 t[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i < 3; i++) {
        u64 c = 0;
        for (int j = i + 1; j < 4; j++) fe_mac(a.v[i], a.v[j], t[i + j], c);
        t[i + 4] = c;
    }
    for (int i = 7; i > 0; i--) t[i] = (t[i] << 1) | (t[i - 1] >> 63);
    t[0] <<= 1;
    u64 c = 0;
    for (int i = 0; i < 4; i++) {
        u64 lo = a.v[i] * a.v[i], hi = MULHI(a.v[i], a.v[i]);
        u64 x = t[2 * i] + c; u64 k = (x < c);
        x += lo; k += (x < lo); t[2 * i] = x;
        u64 y = t[2 * i + 1] + k; u64 k2 = (y < k);
        y += hi; k2 += (y < hi); t[2 * i + 1] = y;
        c = k2;
    }
    fe_reduce(r, t);
}
DEV void fe_sqrn(Fe &r, const Fe &a, int n) {
    r = a;
    for (int i = 0; i < n; i++) fe_sqr(r, r);
}

// r = a^e (mod p)
DEV void fe_pow(Fe &r, const Fe &a, u64 e0, u64 e1, u64 e2, u64 e3) {
    u64 e[4] = {e0, e1, e2, e3};
    Fe x; x.v[0] = 1; x.v[1] = x.v[2] = x.v[3] = 0;
    for (int i = 255; i >= 0; i--) {
        fe_sqr(x, x);
        if ((e[i >> 6] >> (i & 63)) & 1) fe_mul(x, x, a);
    }
    r = x;
}
// r = a^(p-2) = 1/a (mod p), via the secp256k1 addition chain used by
// libsecp256k1: 255 squarings and 15 multiplications (vs ~500 for plain
// square-and-multiply). x_k below is a^(2^k - 1).
DEV void fe_inv(Fe &r, const Fe &a) {
    Fe x2, x3, x6, x9, x11, x22, x44, x88, x176, x220, x223, t;
    fe_sqr(x2, a);          fe_mul(x2, x2, a);
    fe_sqr(x3, x2);         fe_mul(x3, x3, a);
    fe_sqrn(x6, x3, 3);     fe_mul(x6, x6, x3);
    fe_sqrn(x9, x6, 3);     fe_mul(x9, x9, x3);
    fe_sqrn(x11, x9, 2);    fe_mul(x11, x11, x2);
    fe_sqrn(x22, x11, 11);  fe_mul(x22, x22, x11);
    fe_sqrn(x44, x22, 22);  fe_mul(x44, x44, x22);
    fe_sqrn(x88, x44, 44);  fe_mul(x88, x88, x44);
    fe_sqrn(x176, x88, 88); fe_mul(x176, x176, x88);
    fe_sqrn(x220, x176, 44); fe_mul(x220, x220, x44);
    fe_sqrn(x223, x220, 3); fe_mul(x223, x223, x3);
    fe_sqrn(t, x223, 23);   fe_mul(t, t, x22);
    fe_sqrn(t, t, 5);       fe_mul(t, t, a);
    fe_sqrn(t, t, 3);       fe_mul(t, t, x2);
    fe_sqrn(t, t, 2);       fe_mul(r, t, a);
}

// Advance n kangaroos by nsteps jumps each.
//
// All n affine additions of one step share a single field inversion
// (Montgomery's trick). dx/pre are scratch arrays of n elements.
// Points whose x has its low bits clear (x & dpmask == 0) are "distinguished"
// and are appended to out[]; *cnt may exceed maxOut, the excess is dropped.
DEV void kang_step(Fe *px, Fe *py, u64 *dlo, u64 *dhi, int n, int nsteps,
                   const Fe *jx, const Fe *jy, const u64 *jd, u64 dpmask,
                   DPRec *out, u32 *cnt, u32 maxOut, u32 base, Fe *dx, Fe *pre) {
    for (int s = 0; s < nsteps; s++) {
        for (int i = 0; i < n; i++) {
            u32 j = (u32)px[i].v[1] & (NJ - 1);
            fe_sub(dx[i], jx[j], px[i]);
            if (fe_is_zero(dx[i])) dx[i].v[0] = 1;      // P == +-J: skipped below
            if (i == 0) pre[0] = dx[0];
            else fe_mul(pre[i], pre[i - 1], dx[i]);
        }
        Fe inv;
        fe_inv(inv, pre[n - 1]);
        for (int i = n - 1; i >= 0; i--) {
            Fe idx;
            if (i > 0) { fe_mul(idx, inv, pre[i - 1]); fe_mul(inv, inv, dx[i]); }
            else idx = inv;
            Fe X = px[i];
            u32 j = (u32)X.v[1] & (NJ - 1);
            if (fe_eq(X, jx[j])) continue;
            Fe lam, t, x3, y3;
            fe_sub(t, jy[j], py[i]);
            fe_mul(lam, t, idx);
            fe_sqr(x3, lam);
            fe_sub(x3, x3, X);
            fe_sub(x3, x3, jx[j]);
            fe_sub(t, X, x3);
            fe_mul(y3, lam, t);
            fe_sub(y3, y3, py[i]);
            px[i] = x3; py[i] = y3;
            u64 d = dlo[i] + jd[j];
            dhi[i] += (d < jd[j]);
            dlo[i] = d;
            if ((x3.v[0] & dpmask) == 0) {
                u32 k = ATOMIC_INC(cnt);
                if (k < maxOut) {
                    out[k].idx = base + (u32)i; out[k].pad = 0;
                    out[k].x[0] = x3.v[0]; out[k].x[1] = x3.v[1];
                    out[k].x[2] = x3.v[2]; out[k].x[3] = x3.v[3];
                    out[k].dlo = d; out[k].dhi = dhi[i];
                }
            }
        }
    }
}
