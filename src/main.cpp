// Pollard's kangaroo (lambda) solver for the secp256k1 discrete logarithm in a
// known interval - the method of choice for the "Bitcoin puzzle" keys whose
// public key is known.
//
//   kangaroo --pubkey <hex> --bits 70            (key in [2^69, 2^70))
//   kangaroo --pubkey <hex> --start <hex> --end <hex>
//
// See README.md for details.
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "gpu.h"
#include "hash.h"
#include "kernel.h"

typedef unsigned __int128 u128;

// ---------------------------------------------------------------- utilities

static bool g_json = false;
static std::atomic<bool> g_stop(false);

[[noreturn]] static void die(const std::string &msg) {
    if (g_json) printf("{\"event\":\"error\",\"message\":\"%s\"}\n", msg.c_str());
    else fprintf(stderr, "error: %s\n", msg.c_str());
    fflush(stdout);
    exit(1);
}

static int hexval(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}
static std::string strip0x(std::string s) {
    if (s.size() > 1 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) s = s.substr(2);
    return s;
}
static u128 parse_u128(const std::string &in) {
    std::string s = strip0x(in);
    if (s.empty() || s.size() > 32) die("number must be 1-32 hex digits: " + in);
    u128 v = 0;
    for (char c : s) {
        int h = hexval(c);
        if (h < 0) die("bad hex number: " + in);
        v = (v << 4) | (u128)h;
    }
    return v;
}
static std::string hex_u128(u128 v) {
    char buf[40];
    u64 hi = (u64)(v >> 64), lo = (u64)v;
    if (hi) snprintf(buf, sizeof buf, "%llx%016llx", (unsigned long long)hi, (unsigned long long)lo);
    else snprintf(buf, sizeof buf, "%llx", (unsigned long long)lo);
    return buf;
}
static Fe fe_from_hex(const std::string &s) {   // exactly 64 hex digits, big endian
    Fe r = {{0, 0, 0, 0}};
    for (int i = 0; i < 64; i++) {
        int h = hexval(s[i]);
        if (h < 0) die("bad hex in field element");
        int bit = (63 - i) * 4;
        r.v[bit >> 6] |= (u64)h << (bit & 63);
    }
    return r;
}
static std::string fe_hex(const Fe &a) {
    char buf[80];
    snprintf(buf, sizeof buf, "%016llx%016llx%016llx%016llx", (unsigned long long)a.v[3],
             (unsigned long long)a.v[2], (unsigned long long)a.v[1], (unsigned long long)a.v[0]);
    return buf;
}

// ------------------------------------------------------- host curve arithmetic

struct Pt { Fe x, y; bool inf; };

static const Fe FE_ONE = {{1, 0, 0, 0}};
static Pt G;
static Pt Gtab[256];   // Gtab[i] = 2^i * G

static Pt pt_add(const Pt &a, const Pt &b) {
    if (a.inf) return b;
    if (b.inf) return a;
    Fe num, den, lam, t;
    if (fe_eq(a.x, b.x)) {
        if (!fe_eq(a.y, b.y) || fe_is_zero(a.y)) return Pt{{}, {}, true};
        fe_sqr(t, a.x);
        fe_add(num, t, t); fe_add(num, num, t);   // 3x^2
        fe_add(den, a.y, a.y);
    } else {
        fe_sub(num, b.y, a.y);
        fe_sub(den, b.x, a.x);
    }
    fe_inv(den, den);
    fe_mul(lam, num, den);
    Pt r; r.inf = false;
    fe_sqr(r.x, lam);
    fe_sub(r.x, r.x, a.x);
    fe_sub(r.x, r.x, b.x);
    fe_sub(t, a.x, r.x);
    fe_mul(r.y, lam, t);
    fe_sub(r.y, r.y, a.y);
    return r;
}
static Pt pt_neg(Pt a) {
    if (!a.inf) { Fe z = {{0, 0, 0, 0}}; fe_sub(a.y, z, a.y); }
    return a;
}
static bool pt_eq(const Pt &a, const Pt &b) {
    if (a.inf || b.inf) return a.inf == b.inf;
    return fe_eq(a.x, b.x) && fe_eq(a.y, b.y);
}
static Pt mulG(u128 k) {
    Pt r{{}, {}, true};
    for (int i = 0; i < 128; i++)
        if ((k >> i) & 1) r = pt_add(r, Gtab[i]);
    return r;
}
static void curve_init() {
    G.x = fe_from_hex("79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798");
    G.y = fe_from_hex("483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8");
    G.inf = false;
    Gtab[0] = G;
    for (int i = 1; i < 256; i++) Gtab[i] = pt_add(Gtab[i - 1], Gtab[i - 1]);
}

static bool on_curve(const Pt &p) {
    Fe l, r, seven = {{7, 0, 0, 0}};
    fe_sqr(l, p.y);
    fe_sqr(r, p.x); fe_mul(r, r, p.x); fe_add(r, r, seven);
    return fe_eq(l, r);
}
static Pt parse_pubkey(const std::string &in) {
    std::string s = strip0x(in);
    Pt p; p.inf = false;
    if (s.size() == 66 && s[0] == '0' && (s[1] == '2' || s[1] == '3')) {
        p.x = fe_from_hex(s.substr(2));
        Fe t, seven = {{7, 0, 0, 0}};
        fe_sqr(t, p.x); fe_mul(t, t, p.x); fe_add(t, t, seven);
        fe_pow(p.y, t, 0xFFFFFFFFBFFFFF0CULL, ~0ULL, ~0ULL, 0x3FFFFFFFFFFFFFFFULL);   // sqrt
        if ((p.y.v[0] & 1) != (u64)(s[1] == '3')) { Fe z = {{0, 0, 0, 0}}; fe_sub(p.y, z, p.y); }
    } else if (s.size() == 130 && s[0] == '0' && s[1] == '4') {
        p.x = fe_from_hex(s.substr(2, 64));
        p.y = fe_from_hex(s.substr(66, 64));
    } else {
        die("public key must be 33 bytes (02/03...) or 65 bytes (04...) in hex");
    }
    if (!on_curve(p)) die("public key is not a point on secp256k1");
    return p;
}
static std::string pubkey_hex(const Pt &p) {
    return std::string((p.y.v[0] & 1) ? "03" : "02") + fe_hex(p.x);
}
static void serialize_pub(const Pt &p, uint8_t pub[65], bool compressed, int &len) {
    if (compressed) {
        pub[0] = (p.y.v[0] & 1) ? 3 : 2;
        for (int i = 0; i < 32; i++) pub[1 + i] = (uint8_t)(p.x.v[3 - i / 8] >> (56 - 8 * (i % 8)));
        len = 33;
    } else {
        pub[0] = 4;
        for (int i = 0; i < 32; i++) pub[1 + i] = (uint8_t)(p.x.v[3 - i / 8] >> (56 - 8 * (i % 8)));
        for (int i = 0; i < 32; i++) pub[33 + i] = (uint8_t)(p.y.v[3 - i / 8] >> (56 - 8 * (i % 8)));
        len = 65;
    }
}
static std::string address_of(const Pt &p) {
    uint8_t pub[65]; int len;
    serialize_pub(p, pub, true, len);
    return hashes::p2pkh(pub, len);
}
// hash160 = RIPEMD160(SHA256(pubkey))
static void hash160_of(const Pt &p, bool compressed, uint8_t out[20]) {
    uint8_t pub[65], s[32]; int len;
    serialize_pub(p, pub, compressed, len);
    hashes::sha256(pub, len, s);
    hashes::ripemd160(s, 32, out);
}
// Decode a Base58Check P2PKH address to its 20-byte hash160, verifying the
// version byte and checksum. Returns false if the address is malformed.
static bool decode_address(const std::string &addr, uint8_t out[20]) {
    static const char *A = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    uint8_t num[32] = {0};
    size_t nlen = 0, zeros = 0;
    for (char c : addr) if (c == '1') zeros++; else break;
    for (char c : addr) {
        const char *q = strchr(A, c);
        if (!q || !c) return false;
        int carry = (int)(q - A);
        for (size_t i = 0; i < nlen; i++) { carry += num[i] * 58; num[i] = carry & 0xff; carry >>= 8; }
        while (carry) { if (nlen >= sizeof num) return false; num[nlen++] = carry & 0xff; carry >>= 8; }
    }
    std::vector<uint8_t> d(zeros, 0);
    for (size_t i = 0; i < nlen; i++) d.push_back(num[nlen - 1 - i]);
    if (d.size() != 25 || d[0] != 0x00) return false;   // 0x00 = mainnet P2PKH
    uint8_t h1[32], h2[32];
    hashes::sha256(d.data(), 21, h1);
    hashes::sha256(h1, 32, h2);
    if (memcmp(h2, d.data() + 21, 4) != 0) return false;
    memcpy(out, d.data() + 1, 20);
    return true;
}

// ------------------------------------------------------------------- solver

struct Rng {
    u64 s;
    u64 next() {   // splitmix64
        u64 z = (s += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    u128 below(u128 n) { return ((((u128)next()) << 64) | next()) % n; }
};

static u128 g_start, g_width;        // key = g_start + k', 0 <= k' < g_width
static Pt g_Q, g_Qs;                 // target, and target shifted by -start*G
static Fe g_jx[NJ], g_jy[NJ];
static u64 g_jd[NJ];
static u64 g_dpmask;
static std::vector<Fe> g_px, g_py;   // kangaroo i: even = tame, odd = wild
static std::vector<u64> g_dlo, g_dhi;

static std::atomic<u64> g_ops(0), g_resets(0);
static std::atomic<bool> g_found(false);
static u128 g_key;

// A fresh kangaroo. Tame ones know their absolute position (distance = scalar),
// wild ones only their offset from the unknown key.
static void new_kangaroo(u32 idx, Rng &rng, Pt &p, u128 &d) {
    for (;;) {
        u128 r = rng.below(g_width);
        if (idx & 1) { d = r; p = pt_add(g_Qs, mulG(r)); }
        else { d = g_width / 2 + r; p = mulG(d); }
        if (!p.inf) return;
    }
}

// Add the constant point T to the selected kangaroos with one shared inversion.
static void batch_add_const(const std::vector<u32> &ids, const Pt &T) {
    size_t n = ids.size();
    if (!n) return;
    std::vector<Fe> dx(n), pre(n);
    std::vector<char> slow(n, 0);
    for (size_t k = 0; k < n; k++) {
        fe_sub(dx[k], T.x, g_px[ids[k]]);
        if (fe_is_zero(dx[k])) { slow[k] = 1; dx[k] = FE_ONE; }
        if (k == 0) pre[0] = dx[0];
        else fe_mul(pre[k], pre[k - 1], dx[k]);
    }
    Fe inv;
    fe_inv(inv, pre[n - 1]);
    for (size_t k = n; k-- > 0;) {
        Fe idx;
        if (k > 0) { fe_mul(idx, inv, pre[k - 1]); fe_mul(inv, inv, dx[k]); }
        else idx = inv;
        u32 i = ids[k];
        if (slow[k]) {   // doubling (or cancellation): use the generic formula
            Pt r = pt_add(Pt{g_px[i], g_py[i], false}, T);
            if (r.inf) die("degenerate start point");
            g_px[i] = r.x; g_py[i] = r.y;
            continue;
        }
        Fe lam, t, x3, y3;
        fe_sub(t, T.y, g_py[i]);
        fe_mul(lam, t, idx);
        fe_sqr(x3, lam);
        fe_sub(x3, x3, g_px[i]);
        fe_sub(x3, x3, T.x);
        fe_sub(t, g_px[i], x3);
        fe_mul(y3, lam, t);
        fe_sub(y3, y3, g_py[i]);
        g_px[i] = x3; g_py[i] = y3;
    }
}

// ------------------------------------------- brute force (public key unknown)
//
// For the puzzles whose public key has never appeared on-chain the kangaroo
// method cannot be used: all we have is the address. The only option is to walk
// candidate private keys, derive each address, and compare. Consecutive keys
// differ by G, so a whole batch is advanced with one shared field inversion.

static uint8_t g_target160[20];
static bool g_brute_compressed = true, g_brute_both = false;

// Add the constant generator G to n local points, sharing one inversion.
static void step_addG(std::vector<Fe> &px, std::vector<Fe> &py, int n,
                      std::vector<Fe> &dx, std::vector<Fe> &pre, std::vector<char> &slow) {
    for (int i = 0; i < n; i++) {
        fe_sub(dx[i], G.x, px[i]);
        slow[i] = fe_is_zero(dx[i]);
        if (slow[i]) dx[i] = FE_ONE;
        if (i == 0) pre[0] = dx[0];
        else fe_mul(pre[i], pre[i - 1], dx[i]);
    }
    Fe inv;
    fe_inv(inv, pre[n - 1]);
    for (int i = n - 1; i >= 0; i--) {
        Fe idx;
        if (i > 0) { fe_mul(idx, inv, pre[i - 1]); fe_mul(inv, inv, dx[i]); }
        else idx = inv;
        if (slow[i]) {   // px == Gx: this is a doubling
            Pt r = pt_add(Pt{px[i], py[i], false}, G);
            px[i] = r.x; py[i] = r.y;
            continue;
        }
        Fe lam, t, x3, y3;
        fe_sub(t, G.y, py[i]);
        fe_mul(lam, t, idx);
        fe_sqr(x3, lam);
        fe_sub(x3, x3, px[i]);
        fe_sub(x3, x3, G.x);
        fe_sub(t, px[i], x3);
        fe_mul(y3, lam, t);
        fe_sub(y3, y3, py[i]);
        px[i] = x3; py[i] = y3;
    }
}

#define BRUTE_B 1024

static std::atomic<int> g_brute_done(0);   // workers that have finished their share

static void brute_scan(u128 lo, u128 hi) {
    if (lo >= hi) return;
    u128 span = hi - lo;
    int B = (int)std::min<u128>(BRUTE_B, span);
    u128 S = (span + B - 1) / B;         // keys handled per lane
    std::vector<Fe> px(B), py(B), dx(B), pre(B);
    std::vector<char> slow(B);
    std::vector<u128> base(B);
    for (int j = 0; j < B; j++) {
        base[j] = lo + (u128)j * S;
        Pt p = mulG(base[j]);            // worst case 2^125 is still fine
        px[j] = p.x; py[j] = p.y;
    }
    uint8_t h[20];
    for (u128 s = 0; s < S && !g_stop; s++) {
        u64 checked = 0;
        for (int j = 0; j < B; j++) {
            u128 key = base[j] + s;
            if (key >= hi) continue;     // lanes overhang the top of the range
            checked++;
            Pt P{px[j], py[j], false};
            hash160_of(P, g_brute_compressed, h);
            bool hit = memcmp(h, g_target160, 20) == 0;
            if (!hit && g_brute_both) { hash160_of(P, !g_brute_compressed, h); hit = memcmp(h, g_target160, 20) == 0; }
            if (hit) { g_key = key; g_ops += checked; g_found = true; g_stop = true; return; }
        }
        g_ops += checked;                // count real keys only, so a full sweep is exactly the range
        step_addG(px, py, B, dx, pre, slow);
    }
}

static void brute_worker(u128 lo, u128 hi) {
    brute_scan(lo, hi);
    g_brute_done++;
}

static void init_herd(u32 nk, int bits, Rng &rng) {
    g_px.resize(nk); g_py.resize(nk); g_dlo.resize(nk); g_dhi.resize(nk);
    std::vector<u128> r(nk);
    Pt B = mulG(g_width / 2);
    if (B.inf) {   // width 1: nothing to shift by
        for (u32 i = 0; i < nk; i++) {
            Pt p; u128 d; new_kangaroo(i, rng, p, d);
            g_px[i] = p.x; g_py[i] = p.y; g_dlo[i] = (u64)d; g_dhi[i] = (u64)(d >> 64);
        }
        return;
    }
    for (u32 i = 0; i < nk; i++) {
        r[i] = rng.below(g_width);
        const Pt &base = (i & 1) ? g_Qs : B;
        u128 d = (i & 1) ? r[i] : g_width / 2 + r[i];
        g_px[i] = base.x; g_py[i] = base.y;
        g_dlo[i] = (u64)d; g_dhi[i] = (u64)(d >> 64);
    }
    std::vector<u32> ids;
    for (int b = 0; b < bits; b++) {
        ids.clear();
        for (u32 i = 0; i < nk; i++)
            if ((r[i] >> b) & 1) ids.push_back(i);
        batch_add_const(ids, Gtab[b]);
    }
}

// Table of distinguished points. A tame and a wild kangaroo landing on the same
// point reveal the key: tame_dist = k' + wild_dist.
struct Collector {
    struct Ent { u64 x3; u128 d; uint8_t wild; };
    std::unordered_map<u64, Ent> tab;
    std::mutex mu;
    std::atomic<u64> count{0};

    // returns true if the submitting kangaroo is redundant and must be replaced
    bool submit(const DPRec &r) {
        u64 key = r.x[1] ^ (r.x[2] * 0x9E3779B97F4A7C15ULL);
        u128 d = ((u128)r.dhi << 64) | r.dlo;
        uint8_t wild = r.idx & 1;
        std::lock_guard<std::mutex> lk(mu);
        auto it = tab.find(key);
        if (it == tab.end() || it->second.x3 != r.x[3]) {
            tab[key] = Ent{r.x[3], d, wild};
            count++;
            return false;
        }
        if (it->second.wild == wild) return it->second.d != d;   // herd-mates merged
        u128 t = wild ? it->second.d : d, w = wild ? d : it->second.d;
        u128 k = g_start + t - w;
        if (pt_eq(mulG(k), g_Q)) {
            g_key = k;
            g_found = true;
            g_stop = true;
            return false;
        }
        return true;
    }
};
static Collector g_col;

#define CPU_GRP 1024
#define CPU_STEPS 32

static void cpu_worker(u32 lo, u32 hi, u64 seed) {
    Rng rng{seed};
    std::vector<Fe> dx(CPU_GRP), pre(CPU_GRP);
    std::vector<DPRec> out((size_t)CPU_GRP * CPU_STEPS);
    while (!g_stop) {
        for (u32 b = lo; b < hi && !g_stop; b += CPU_GRP) {
            int n = (int)std::min<u32>(CPU_GRP, hi - b);
            u32 cnt = 0;
            kang_step(&g_px[b], &g_py[b], &g_dlo[b], &g_dhi[b], n, CPU_STEPS, g_jx, g_jy, g_jd,
                      g_dpmask, out.data(), &cnt, (u32)out.size(), b, dx.data(), pre.data());
            g_ops += (u64)n * CPU_STEPS;
            for (u32 k = 0; k < cnt; k++) {
                if (!g_col.submit(out[k])) continue;
                u32 i = out[k].idx;
                Pt p; u128 d;
                new_kangaroo(i, rng, p, d);
                g_px[i] = p.x; g_py[i] = p.y; g_dlo[i] = (u64)d; g_dhi[i] = (u64)(d >> 64);
                g_resets++;
            }
        }
    }
}

static void gpu_worker(u32 nk, u32 maxOut, u64 seed) {
    Rng rng{seed};
    std::vector<DPRec> out(maxOut);
    const int steps = 64;
    while (!g_stop) {
        u32 cnt = gpu_run(steps, g_dpmask, out.data());
        g_ops += (u64)nk * steps;
        for (u32 k = 0; k < cnt; k++) {
            if (!g_col.submit(out[k])) continue;
            Pt p; u128 d;
            new_kangaroo(out[k].idx, rng, p, d);
            gpu_set(out[k].idx, p.x, p.y, (u64)d, (u64)(d >> 64));
            g_resets++;
        }
    }
}

// ----------------------------------------------------------------- self test

static int selftest() {
    int fails = 0;
    auto check = [&](bool ok, const char *what) {
        printf("  [%s] %s\n", ok ? " ok " : "FAIL", what);
        if (!ok) fails++;
    };
    check(on_curve(G), "generator is on the curve");
    check(fe_hex(mulG(2).x) == "c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5", "2*G matches the known value");
    Fe a = fe_from_hex("DEADBEEF00000000FFFFFFFFFFFFFFFF123456789ABCDEF00FEDCBA987654321"), ai, one;
    fe_inv(ai, a); fe_mul(one, a, ai);
    check(fe_eq(one, FE_ONE), "field inversion");
    check(pt_eq(pt_add(mulG(1000), mulG(2345)), mulG(3345)), "point addition is consistent with scalar multiplication");
    check(address_of(mulG(1)) == "1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH", "address of private key 1 (SHA-256, RIPEMD-160, Base58)");
    check(pt_eq(parse_pubkey(pubkey_hex(mulG(0xD2C55))), mulG(0xD2C55)), "compressed public key round trip");

    // the batched kernel must agree with the simple one-at-a-time formulas
    Rng rng{42};
    for (int j = 0; j < NJ; j++) {
        g_jd[j] = 1 + rng.next() % 100000;
        Pt p = mulG(g_jd[j]); g_jx[j] = p.x; g_jy[j] = p.y;
    }
    const int n = 37, steps = 50;
    std::vector<Fe> px(n), py(n), dx(n), pre(n);
    std::vector<u64> dlo(n), dhi(n, 0);
    std::vector<Pt> ref(n);
    std::vector<u128> refd(n);
    for (int i = 0; i < n; i++) {
        refd[i] = 1000000 + rng.next() % 1000000;
        ref[i] = mulG(refd[i]);
        px[i] = ref[i].x; py[i] = ref[i].y; dlo[i] = (u64)refd[i];
    }
    std::vector<DPRec> out(n * steps);
    u32 cnt = 0;
    kang_step(px.data(), py.data(), dlo.data(), dhi.data(), n, steps, g_jx, g_jy, g_jd, 0, out.data(), &cnt,
              (u32)out.size(), 0, dx.data(), pre.data());
    bool ok = (cnt == (u32)(n * steps));
    for (int i = 0; i < n; i++) {
        for (int s = 0; s < steps; s++) {
            int j = (int)(ref[i].x.v[1] & (NJ - 1));
            ref[i] = pt_add(ref[i], Pt{g_jx[j], g_jy[j], false});
            refd[i] += g_jd[j];
        }
        ok = ok && fe_eq(ref[i].x, px[i]) && fe_eq(ref[i].y, py[i]) && dlo[i] == (u64)refd[i];
        ok = ok && pt_eq(mulG(refd[i]), ref[i]);
    }
    check(ok, "kangaroo kernel (batched jumps) matches the reference implementation");

    uint8_t h160[20], d160[20];
    hash160_of(mulG(0xD2C55), true, h160);
    check(decode_address("1HsMJxNiV7TLxmoF6uJNkydxPFDog4NQum", d160) && memcmp(h160, d160, 20) == 0,
          "Base58 address decodes to the matching hash160");
    check(!decode_address("1HsMJxNiV7TLxmoF6uJNkydxPFDog4NQ0m", d160), "a bad checksum is rejected");
    // the incremental +G walk must land on the same points as scalar multiplication
    {
        const int bn = 40;
        std::vector<Fe> bx(bn), by(bn), bdx(bn), bpre(bn);
        std::vector<char> bslow(bn);
        for (int i = 0; i < bn; i++) { Pt p = mulG(500 + i); bx[i] = p.x; by[i] = p.y; }
        bool bok = true;
        for (int s = 1; s <= 30; s++) {
            step_addG(bx, by, bn, bdx, bpre, bslow);
            for (int i = 0; i < bn; i++) bok = bok && pt_eq(mulG(500 + i + s), Pt{bx[i], by[i], false});
        }
        check(bok, "brute-force +G stepping matches scalar multiplication");
    }
    printf(fails ? "SELF TEST FAILED\n" : "self test passed\n");
    return fails ? 1 : 0;
}

// ----------------------------------------------------------------------- main

static void usage() {
    puts("Solver for secp256k1 Bitcoin-puzzle keys in a known range\n"
         "\n"
         "  kangaroo --pubkey HEX --bits N [options]       known public key: kangaroo method\n"
         "  kangaroo --pubkey HEX --start HEX --end HEX    key in [start, end]\n"
         "  kangaroo --brute --address ADDR --bits N       public key unknown: brute force by address\n"
         "\n"
         "The kangaroo method needs the public key and is vastly faster (~2*sqrt(range)).\n"
         "Brute force is for puzzles whose public key has never been revealed - it has to\n"
         "try keys one by one (~range/2 on average), so only low puzzles are feasible.\n"
         "\n"
         "options:\n"
         "  --address ADDR    expected address; required for --brute, else used to verify\n"
         "  --hash160 HEX     target hash160 (40 hex) instead of --address, for --brute\n"
         "  --brute           search by address, without a public key\n"
         "  --uncompressed    brute force against the uncompressed address\n"
         "  --both            brute force against both compressed and uncompressed\n"
         "  --threads T       CPU threads (default: all cores)\n"
         "  --gpu             use the CUDA backend (binary must be built with `make gpu`)\n"
         "  --device N        CUDA device index (default 0)\n"
         "  --kangaroos N     herd size (default: 1024 per CPU thread, 2^20 on GPU)\n"
         "  --dp N            distinguished point bits (default: automatic)\n"
         "  --seed S          RNG seed (default: random)\n"
         "  --json            machine readable progress, one JSON object per line\n"
         "\n"
         "other modes:\n"
         "  --derive KEYHEX   print public key and address of a private key\n"
         "  --info            print build / device information as JSON\n"
         "  --selftest        run the built-in arithmetic tests");
}

static void on_signal(int) { g_stop = true; }

// Brute force by address (public key unknown). Returns the process exit code.
static int run_brute(const std::string &address, const std::string &hash160hex, int bits,
                     const std::string &s_start, const std::string &s_end, int threads,
                     bool uncompressed, bool both) {
    u128 start, end;
    if (bits) {
        if (bits < 2 || bits > 80) die("--bits must be between 2 and 80 for brute force");
        start = (u128)1 << (bits - 1);
        end = ((u128)1 << bits) - 1;
    } else {
        if (s_start.empty() || s_end.empty()) die("give either --bits or --start and --end");
        start = parse_u128(s_start);
        end = parse_u128(s_end);
        if (end < start) die("--end is below --start");
        if (end >> 80) die("brute-force range must be below 2^80");
    }
    if (!hash160hex.empty()) {
        std::string h = strip0x(hash160hex);
        if (h.size() != 40) die("--hash160 must be exactly 40 hex characters");
        for (int i = 0; i < 20; i++) g_target160[i] = (uint8_t)(hexval(h[2 * i]) << 4 | hexval(h[2 * i + 1]));
    } else if (!address.empty()) {
        if (!decode_address(address, g_target160)) die("could not decode address (bad Base58 or checksum, or not a P2PKH address)");
    } else {
        die("brute force needs --address or --hash160");
    }
    g_brute_compressed = !uncompressed;
    g_brute_both = both;
    if (threads < 1) threads = 1;
    if (threads > 256) threads = 256;

    u128 width = end - start + 1;
    double lw = std::log2((double)width);

    auto t0 = std::chrono::steady_clock::now();
    auto elapsed = [&]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };

    if (g_json)
        printf("{\"event\":\"start\",\"mode\":\"brute\",\"backend\":\"cpu: %d threads\",\"start\":\"%s\",\"end\":\"%s\","
               "\"range_bits\":%.2f,\"total_ops\":%.0f,\"expected_ops\":%.0f}\n",
               threads, hex_u128(start).c_str(), hex_u128(end).c_str(), lw, (double)width, (double)width / 2);
    else
        printf("mode        : brute force by address (public key unknown)\n"
               "range       : %s .. %s (2^%.2f keys)\nbackend     : cpu: %d threads\n"
               "match       : %s address\nexpected    : ~%.3g keys (half the range)\n",
               hex_u128(start).c_str(), hex_u128(end).c_str(), lw, threads,
               both ? "compressed or uncompressed" : (uncompressed ? "uncompressed" : "compressed"), (double)width / 2);
    fflush(stdout);

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    std::vector<std::thread> workers;
    u128 per = (width + threads - 1) / threads;
    for (int t = 0; t < threads; t++) {
        u128 lo = start + (u128)t * per;
        u128 hi = lo + per; if (hi > end + 1) hi = end + 1;
        if (lo < hi) workers.emplace_back(brute_worker, lo, hi);
    }

    u64 lastOps = 0; double lastT = 0, nextReport = 1.0;
    auto progress = [&](double t) {
        u64 ops = g_ops;
        double rate = t > lastT ? (ops - lastOps) / (t - lastT) : 0;
        lastOps = ops; lastT = t;
        if (g_json)
            printf("{\"event\":\"progress\",\"ops\":%llu,\"rate\":%.0f,\"elapsed\":%.2f}\n",
                   (unsigned long long)ops, rate, t);
        else
            printf("\r%7.1fs  %6.2f Mkeys/s  %llu keys tried (%.4f%% of the range)   ", t, rate / 1e6,
                   (unsigned long long)ops, 100.0 * (double)ops / (double)width);
        fflush(stdout);
    };
    const int nworkers = (int)workers.size();
    while (!g_stop && g_brute_done < nworkers) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        double t = elapsed();
        if (t < nextReport) continue;
        nextReport = t + 1.0;
        progress(t);
    }
    for (auto &w : workers) w.join();

    if (g_found) {
        Pt p = mulG(g_key);
        std::string addr_c = address_of(p);   // compressed address
        uint8_t h[20]; hash160_of(p, false, h);
        uint8_t v[21]; v[0] = 0; memcpy(v + 1, h, 20);
        std::string addr_u = hashes::base58check(v, 21);
        bool comp = (addr_c == address) || g_brute_compressed;
        std::string addr = comp ? addr_c : addr_u;
        bool match = address.empty() ? true : (addr_c == address || addr_u == address);
        if (g_json)
            printf("{\"event\":\"found\",\"key\":\"%s\",\"pubkey\":\"%s\",\"address\":\"%s\",\"address_checked\":%s,"
                   "\"address_match\":%s,\"ops\":%llu,\"elapsed\":%.3f}\n",
                   hex_u128(g_key).c_str(), pubkey_hex(p).c_str(), addr.c_str(), address.empty() ? "false" : "true",
                   match ? "true" : "false", (unsigned long long)g_ops.load(), elapsed());
        else
            printf("\nKEY FOUND\n  private key : %s\n  public key  : %s\n  address     : %s\n  work        : %llu keys in %.1f s\n",
                   hex_u128(g_key).c_str(), pubkey_hex(p).c_str(), addr.c_str(), (unsigned long long)g_ops.load(), elapsed());
        fflush(stdout);
        return match ? 0 : 3;
    }
    if (!g_stop) {   // every worker finished its share: the whole range was searched
        progress(elapsed());
        if (g_json)
            printf("{\"event\":\"exhausted\",\"ops\":%llu,\"elapsed\":%.2f}\n", (unsigned long long)g_ops.load(), elapsed());
        else
            printf("\nrange exhausted: the key is not in %s .. %s (%llu keys checked)\n", hex_u128(start).c_str(),
                   hex_u128(end).c_str(), (unsigned long long)g_ops.load());
        fflush(stdout);
        return 4;
    }
    if (g_json) printf("{\"event\":\"stopped\",\"ops\":%llu,\"elapsed\":%.2f}\n", (unsigned long long)g_ops.load(), elapsed());
    else printf("\nstopped\n");
    return 2;
}

int main(int argc, char **argv) {
    std::string pubkey, address, s_start, s_end, derive, hash160;
    int bits = 0, threads = (int)std::thread::hardware_concurrency(), device = 0, dp = -1;
    long long kangaroos = 0;
    bool use_gpu = false, info = false, st = false, brute = false, uncompressed = false, both = false;
    u64 seed = ((u64)std::random_device{}() << 32) ^ std::random_device{}();
    if (threads < 1) threads = 1;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto val = [&]() -> std::string {
            if (i + 1 >= argc) die("missing value for " + a);
            return argv[++i];
        };
        if (a == "--pubkey") pubkey = val();
        else if (a == "--bits") bits = atoi(val().c_str());
        else if (a == "--start") s_start = val();
        else if (a == "--end") s_end = val();
        else if (a == "--address") address = val();
        else if (a == "--hash160") hash160 = val();
        else if (a == "--brute") brute = true;
        else if (a == "--uncompressed") uncompressed = true;
        else if (a == "--both") both = true;
        else if (a == "--threads") threads = atoi(val().c_str());
        else if (a == "--gpu") use_gpu = true;
        else if (a == "--device") device = atoi(val().c_str());
        else if (a == "--kangaroos") kangaroos = atoll(val().c_str());
        else if (a == "--dp") dp = atoi(val().c_str());
        else if (a == "--seed") seed = strtoull(val().c_str(), nullptr, 0);
        else if (a == "--json") g_json = true;
        else if (a == "--derive") derive = val();
        else if (a == "--info") info = true;
        else if (a == "--selftest") st = true;
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else die("unknown option " + a);
    }

    if (info) {
        int n = gpu_device_count();
        printf("{\"cuda_build\":%s,\"threads\":%u,\"devices\":[",
#ifdef WITH_CUDA
               "true",
#else
               "false",
#endif
               std::thread::hardware_concurrency());
        for (int d = 0; d < n; d++) printf("%s\"%s\"", d ? "," : "", gpu_device_name(d));
        printf("]}\n");
        return 0;
    }

    curve_init();
    if (st) return selftest();
    if (!derive.empty()) {
        u128 k = parse_u128(derive);
        if (!k) die("private key must not be zero");
        Pt p = mulG(k);
        printf("private key : %s\npublic key  : %s\naddress     : %s\n", hex_u128(k).c_str(),
               pubkey_hex(p).c_str(), address_of(p).c_str());
        return 0;
    }
    // Brute force: no public key, only an address or hash160.
    if (brute || (pubkey.empty() && (!address.empty() || !hash160.empty()))) {
        if (!pubkey.empty()) die("--brute searches by address; do not pass --pubkey (use the kangaroo method instead)");
        if (use_gpu) die("the brute-force mode is CPU-only in this build");
        return run_brute(address, hash160, bits, s_start, s_end, threads, uncompressed, both);
    }
    if (pubkey.empty()) { usage(); return 1; }

    // ---- problem setup
    u128 end;
    if (bits) {
        if (bits < 2 || bits > 125) die("--bits must be between 2 and 125");
        g_start = (u128)1 << (bits - 1);
        end = ((u128)1 << bits) - 1;
    } else {
        if (s_start.empty() || s_end.empty()) die("give either --bits or --start and --end");
        g_start = parse_u128(s_start);
        end = parse_u128(s_end);
        if (end < g_start) die("--end is below --start");
        if (end >> 125) die("range end must be below 2^125");
    }
    g_width = end - g_start + 1;
    int wbits = 0;
    while (wbits < 126 && ((u128)1 << wbits) < g_width) wbits++;
    double lw = std::log2((double)g_width);
    double expected = 2.0 * std::sqrt((double)g_width);

    g_Q = parse_pubkey(pubkey);
    if (threads < 1) threads = 1;
    if (threads > 256) threads = 256;

    u32 nk;
    const u32 maxOut = 1u << 18;
    if (use_gpu) {
#ifndef WITH_CUDA
        die("this binary was built without CUDA support - run `make gpu` on a machine with the CUDA toolkit");
#endif
        if (gpu_device_count() <= device) die("CUDA device not found");
        nk = kangaroos > 0 ? (u32)kangaroos : (1u << 20);
        nk = (nk + GPU_GRP - 1) / GPU_GRP * GPU_GRP;
    } else {
        nk = kangaroos > 0 ? (u32)kangaroos : (u32)threads * 1024;
        nk = std::max<u32>(nk, 2 * (u32)threads);
    }
    if (kangaroos <= 0) {   // a small interval cannot feed a large herd: walks would just merge
        double cap = std::max(64.0, std::exp2(lw / 2 - 2));
        if (cap < nk) nk = (u32)cap;
        if (use_gpu) nk = (nk + GPU_GRP - 1) / GPU_GRP * GPU_GRP;
    }
    if (nk & 1) nk++;

    if (dp < 0) {
        double e = lw / 2 + 1;   // log2 of the expected work
        dp = (int)std::floor(std::max(e - std::log2((double)nk) - 3, e - 22));
        if (dp < 0) dp = 0;
    }
    if (dp > 60) dp = 60;
    g_dpmask = ((u64)1 << dp) - 1;

    auto t0 = std::chrono::steady_clock::now();
    auto elapsed = [&]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
    auto report_found = [&](u128 key, u64 ops) {
        Pt p = mulG(key);
        std::string addr = address_of(p);
        bool has = !address.empty(), match = (addr == address);
        if (g_json)
            printf("{\"event\":\"found\",\"key\":\"%s\",\"pubkey\":\"%s\",\"address\":\"%s\",\"address_checked\":%s,"
                   "\"address_match\":%s,\"ops\":%llu,\"elapsed\":%.3f}\n",
                   hex_u128(key).c_str(), pubkey_hex(p).c_str(), addr.c_str(), has ? "true" : "false",
                   (has && match) ? "true" : "false", (unsigned long long)ops, elapsed());
        else {
            printf("\nKEY FOUND\n  private key : %s\n  public key  : %s\n  address     : %s%s\n", hex_u128(key).c_str(),
                   pubkey_hex(p).c_str(), addr.c_str(),
                   has ? (match ? "  (matches the expected address)" : "  (DOES NOT match the expected address!)") : "");
            printf("  work        : %llu jumps in %.1f s\n", (unsigned long long)ops, elapsed());
        }
        fflush(stdout);
        return (!has || match) ? 0 : 3;
    };

    // trivial cases that the random walks cannot handle
    Pt startG = mulG(g_start);
    if (pt_eq(startG, g_Q)) return report_found(g_start, 0);
    g_Qs = pt_add(g_Q, pt_neg(startG));

    Rng rng{seed};
    int jb = std::max(1, std::min(62, (int)(lw / 2) + 1));   // jumps average ~sqrt(width)
    for (int j = 0; j < NJ; j++) {
        for (;;) {
            g_jd[j] = 1 + rng.next() % ((u64)1 << jb);
            bool dup = false;
            for (int k = 0; k < j; k++) dup = dup || g_jd[k] == g_jd[j];
            if (!dup || jb < 6) break;
        }
        Pt p = mulG(g_jd[j]);
        g_jx[j] = p.x; g_jy[j] = p.y;
    }

    std::string backend = use_gpu ? std::string("gpu: ") + gpu_device_name(device)
                                  : "cpu: " + std::to_string(threads) + " threads";
    if (g_json)
        printf("{\"event\":\"start\",\"backend\":\"%s\",\"start\":\"%s\",\"end\":\"%s\",\"range_bits\":%.2f,"
               "\"kangaroos\":%u,\"dp_bits\":%d,\"expected_ops\":%.0f}\n",
               backend.c_str(), hex_u128(g_start).c_str(), hex_u128(end).c_str(), lw, nk, dp, expected);
    else
        printf("range       : %s .. %s (2^%.2f keys)\nbackend     : %s\nkangaroos   : %u, DP bits %d\n"
               "expected    : ~2^%.1f jumps\n",
               hex_u128(g_start).c_str(), hex_u128(end).c_str(), lw, backend.c_str(), nk, dp, std::log2(expected));
    fflush(stdout);

    init_herd(nk, wbits, rng);

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    std::vector<std::thread> workers;
    if (use_gpu) {
        char err[256] = "";
        if (!gpu_init(device, nk, g_jx, g_jy, g_jd, maxOut, err, sizeof err)) die(std::string("CUDA init failed: ") + err);
        gpu_upload(g_px.data(), g_py.data(), g_dlo.data(), g_dhi.data());
        workers.emplace_back(gpu_worker, nk, maxOut, rng.next());
    } else {
        u32 per = (nk + threads - 1) / threads;
        for (int t = 0; t < threads; t++) {
            u32 lo = std::min(nk, (u32)t * per), hi = std::min(nk, lo + per);
            if (lo < hi) workers.emplace_back(cpu_worker, lo, hi, rng.next());
        }
    }

    t0 = std::chrono::steady_clock::now();
    u64 lastOps = 0;
    double lastT = 0, nextReport = 1.0;
    while (!g_stop) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        double t = elapsed();
        if (t < nextReport) continue;
        nextReport = t + 1.0;
        u64 ops = g_ops;
        double rate = (ops - lastOps) / (t - lastT);
        lastOps = ops; lastT = t;
        if (g_json)
            printf("{\"event\":\"progress\",\"ops\":%llu,\"rate\":%.0f,\"elapsed\":%.2f,\"dps\":%llu,\"resets\":%llu}\n",
                   (unsigned long long)ops, rate, t, (unsigned long long)g_col.count.load(),
                   (unsigned long long)g_resets.load());
        else
            printf("\r%7.1fs  %6.2f Mjumps/s  2^%.2f jumps (%.0f%% of expected)  %llu DPs   ", t, rate / 1e6,
                   std::log2((double)ops + 1), 100.0 * ops / expected, (unsigned long long)g_col.count.load());
        fflush(stdout);
    }
    for (auto &w : workers) w.join();
    if (use_gpu) gpu_free();

    if (g_found) return report_found(g_key, g_ops);
    if (g_json) printf("{\"event\":\"stopped\",\"ops\":%llu,\"elapsed\":%.2f}\n", (unsigned long long)g_ops.load(), elapsed());
    else printf("\nstopped\n");
    return 2;
}
