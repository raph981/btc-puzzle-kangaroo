// SHA-256, RIPEMD-160 and Base58Check - just enough to turn a public key
// into a legacy (P2PKH) Bitcoin address.
#pragma once
#include <stdint.h>
#include <string.h>
#include <string>
#include <vector>

namespace hashes {

static inline uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
static inline uint32_t rotl(uint32_t x, int n) { return (x << n) | (x >> (32 - n)); }

static inline void sha256(const uint8_t *msg, size_t len, uint8_t out[32]) {
    static const uint32_t K[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                     0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    std::vector<uint8_t> m(msg, msg + len);
    m.push_back(0x80);
    while (m.size() % 64 != 56) m.push_back(0);
    uint64_t bits = (uint64_t)len * 8;
    for (int i = 7; i >= 0; i--) m.push_back((uint8_t)(bits >> (i * 8)));
    for (size_t off = 0; off < m.size(); off += 64) {
        uint32_t w[64];
        for (int i = 0; i < 16; i++)
            w[i] = (uint32_t)m[off + 4 * i] << 24 | (uint32_t)m[off + 4 * i + 1] << 16 |
                   (uint32_t)m[off + 4 * i + 2] << 8 | m[off + 4 * i + 3];
        for (int i = 16; i < 64; i++) {
            uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; i++) {
            uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            uint32_t ch = (e & f) ^ (~e & g);
            uint32_t t1 = hh + S1 + ch + K[i] + w[i];
            uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t t2 = S0 + mj;
            hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 4; j++) out[4 * i + j] = (uint8_t)(h[i] >> (24 - 8 * j));
}

static inline void ripemd160(const uint8_t *msg, size_t len, uint8_t out[20]) {
    static const uint8_t RL[80] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        7, 4, 13, 1, 10, 6, 15, 3, 12, 0, 9, 5, 2, 14, 11, 8,
        3, 10, 14, 4, 9, 15, 8, 1, 2, 7, 0, 6, 13, 11, 5, 12,
        1, 9, 11, 10, 0, 8, 12, 4, 13, 3, 7, 15, 14, 5, 6, 2,
        4, 0, 5, 9, 7, 12, 2, 10, 14, 1, 3, 8, 11, 6, 15, 13};
    static const uint8_t RR[80] = {
        5, 14, 7, 0, 9, 2, 11, 4, 13, 6, 15, 8, 1, 10, 3, 12,
        6, 11, 3, 7, 0, 13, 5, 10, 14, 15, 8, 12, 4, 9, 1, 2,
        15, 5, 1, 3, 7, 14, 6, 9, 11, 8, 12, 2, 10, 0, 4, 13,
        8, 6, 4, 1, 3, 11, 15, 0, 5, 12, 2, 13, 9, 7, 10, 14,
        12, 15, 10, 4, 1, 5, 8, 7, 6, 2, 13, 14, 0, 3, 9, 11};
    static const uint8_t SL[80] = {
        11, 14, 15, 12, 5, 8, 7, 9, 11, 13, 14, 15, 6, 7, 9, 8,
        7, 6, 8, 13, 11, 9, 7, 15, 7, 12, 15, 9, 11, 7, 13, 12,
        11, 13, 6, 7, 14, 9, 13, 15, 14, 8, 13, 6, 5, 12, 7, 5,
        11, 12, 14, 15, 14, 15, 9, 8, 9, 14, 5, 6, 8, 6, 5, 12,
        9, 15, 5, 11, 6, 8, 13, 12, 5, 12, 13, 14, 11, 8, 5, 6};
    static const uint8_t SR[80] = {
        8, 9, 9, 11, 13, 15, 15, 5, 7, 7, 8, 11, 14, 14, 12, 6,
        9, 13, 15, 7, 12, 8, 9, 11, 7, 7, 12, 7, 6, 15, 13, 11,
        9, 7, 15, 11, 8, 6, 6, 14, 12, 13, 5, 14, 13, 13, 7, 5,
        15, 5, 8, 11, 14, 14, 6, 14, 6, 9, 12, 9, 12, 5, 15, 8,
        8, 5, 12, 9, 12, 5, 14, 6, 8, 13, 6, 5, 15, 13, 11, 11};
    static const uint32_t KL[5] = {0, 0x5A827999, 0x6ED9EBA1, 0x8F1BBCDC, 0xA953FD4E};
    static const uint32_t KR[5] = {0x50A28BE6, 0x5C4DD124, 0x6D703EF3, 0x7A6D76E9, 0};
    auto f = [](int r, uint32_t x, uint32_t y, uint32_t z) -> uint32_t {
        switch (r) {
        case 0: return x ^ y ^ z;
        case 1: return (x & y) | (~x & z);
        case 2: return (x | ~y) ^ z;
        case 3: return (x & z) | (y & ~z);
        default: return x ^ (y | ~z);
        }
    };
    uint32_t h[5] = {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0};
    std::vector<uint8_t> m(msg, msg + len);
    m.push_back(0x80);
    while (m.size() % 64 != 56) m.push_back(0);
    uint64_t bits = (uint64_t)len * 8;
    for (int i = 0; i < 8; i++) m.push_back((uint8_t)(bits >> (i * 8)));
    for (size_t off = 0; off < m.size(); off += 64) {
        uint32_t X[16];
        for (int i = 0; i < 16; i++)
            X[i] = (uint32_t)m[off + 4 * i] | (uint32_t)m[off + 4 * i + 1] << 8 |
                   (uint32_t)m[off + 4 * i + 2] << 16 | (uint32_t)m[off + 4 * i + 3] << 24;
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4];
        uint32_t a2 = a, b2 = b, c2 = c, d2 = d, e2 = e;
        for (int i = 0; i < 80; i++) {
            int r = i / 16;
            uint32_t t = rotl(a + f(r, b, c, d) + X[RL[i]] + KL[r], SL[i]) + e;
            a = e; e = d; d = rotl(c, 10); c = b; b = t;
            t = rotl(a2 + f(4 - r, b2, c2, d2) + X[RR[i]] + KR[r], SR[i]) + e2;
            a2 = e2; e2 = d2; d2 = rotl(c2, 10); c2 = b2; b2 = t;
        }
        uint32_t t = h[1] + c + d2;
        h[1] = h[2] + d + e2; h[2] = h[3] + e + a2; h[3] = h[4] + a + b2; h[4] = h[0] + b + c2;
        h[0] = t;
    }
    for (int i = 0; i < 5; i++)
        for (int j = 0; j < 4; j++) out[4 * i + j] = (uint8_t)(h[i] >> (8 * j));
}

static inline std::string base58check(const uint8_t *payload, size_t len) {
    static const char *A = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    std::vector<uint8_t> d(payload, payload + len);
    uint8_t h1[32], h2[32];
    sha256(d.data(), d.size(), h1);
    sha256(h1, 32, h2);
    d.insert(d.end(), h2, h2 + 4);
    std::string out;
    size_t zeros = 0;
    while (zeros < d.size() && d[zeros] == 0) zeros++;
    std::vector<uint8_t> num(d.begin() + zeros, d.end());
    while (!num.empty()) {
        int rem = 0;
        std::vector<uint8_t> q;
        for (uint8_t byte : num) {
            int cur = rem * 256 + byte;
            uint8_t dig = (uint8_t)(cur / 58);
            rem = cur % 58;
            if (!q.empty() || dig) q.push_back(dig);
        }
        out.push_back(A[rem]);
        num.swap(q);
    }
    out.append(zeros, '1');
    return std::string(out.rbegin(), out.rend());
}

// Legacy P2PKH address of a serialized public key.
static inline std::string p2pkh(const uint8_t *pub, size_t len) {
    uint8_t s[32], payload[21];
    sha256(pub, len, s);
    payload[0] = 0x00;
    ripemd160(s, 32, payload + 1);
    return base58check(payload, 21);
}

} // namespace hashes
