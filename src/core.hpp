#pragma once
// Exact arithmetic and the in-memory search domain. No file formats or saved state.
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

namespace p5 {
using I = __int128_t;
using U = __uint128_t;
constexpr uint32_t LOW = 10000, WIDTH = 100000;
constexpr uint32_t EMPTY = 0xffffffffu, END = 0x80000000u, MASK = 0x3fffffffu;

struct Spec {
    uint32_t modulus, ahi, bhi, chi, ne, nd, tiny, chunk_a;
    uint64_t ab_count() const {
        return uint64_t(ahi - LOW + 1) * (bhi - LOW + 1);
    }
    uint64_t pair_count() const {
        return uint64_t(ne) * nd;
    }
};
inline Spec spec(bool tiny) {
    // d = LOW + part + 10*i; e = LOW + j. All ten parts are searched.
    return tiny ? Spec{1009, 10019, 10019, 10039, 40, 4, 1, 8}
                : Spec{948423849, 99999, 88240, 88239, 78240, 7824, 0, 256};
}

inline I fifth(uint32_t x) {
    I y = x;
    return y * y * y * y * y;
}
inline I f(uint32_t j, uint32_t x) {
    I coefficient = 1;
    for (uint32_t k = 0; k < j; ++k)
        coefficient *= WIDTH;
    return fifth(x) - coefficient * x;
}
inline uint32_t mod(I n, uint32_t p) {
    I r = n % p;
    return uint32_t(r < 0 ? r + p : r);
}
inline bool original_solution(uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t e) {
    return f(4, a) + f(3, b) + f(2, c) + f(1, d) + f(0, e) == 0;
}

// Signed values are represented modulo 2^96 on the GPU. Every exact sum in
// this domain has magnitude below 2^95, so equality here is exact, not a hash.
struct Wide {
    uint32_t lo, mid, hi;
};
static_assert(sizeof(Wide) == 12);
inline Wide wide(I x) {
    U u = U(x);
    return {uint32_t(u), uint32_t(u >> 32), uint32_t(u >> 64)};
}
HD inline Wide add(Wide a, Wide b) {
    Wide r;
    r.lo = a.lo + b.lo;
    uint32_t carry = r.lo < a.lo;
    uint32_t middle = a.mid + b.mid, carry2 = middle < a.mid;
    r.mid = middle + carry;
    carry2 |= r.mid < middle;
    r.hi = a.hi + b.hi + carry2;
    return r;
}
HD inline Wide neg(Wide a) {
    return add({~a.lo, ~a.mid, ~a.hi}, {1, 0, 0});
}
HD inline bool eq(Wide a, Wide b) {
    return a.lo == b.lo && a.mid == b.mid && a.hi == b.hi;
}

// F_j(x) == 3*((7*x) mod 10) (mod 30) for j>0; F_0(x) == 0.
// Remove this residue before dividing by 30. The query adds the decimal
// carry of the four residues, preserving the original equation modulo p.
inline uint32_t reduced_key(uint32_t j, uint32_t x, uint32_t p) {
    uint32_t ell = j ? (7 * x) % 10 : 0;
    I value = f(j, x) - 3 * ell;
    if (value % 30)
        throw std::runtime_error("Nonintegral reduced key");
    return mod(value / 30, p);
}
HD inline uint32_t make_key(uint32_t base, uint32_t h, uint32_t p) {
    int32_t k = int32_t(base) - int32_t(h);
    return uint32_t(k < 0 ? k + int32_t(p) : k);
}
struct Aux {
    Spec s;
    std::vector<uint32_t> h;
    std::vector<Wide> exact;
    std::vector<uint8_t> r13;
    explicit Aux(Spec in) : s(in), h(5 * WIDTH), exact(5 * WIDTH), r13(5 * WIDTH) {
        for (uint32_t j = 0; j < 5; ++j)
            for (uint32_t x = 0; x < WIDTH; ++x) {
                size_t at = j * WIDTH + x;
                h[at] = reduced_key(j, x, s.modulus);
                exact[at] = wide(f(j, x));
                r13[at] = uint8_t(mod(f(j, x), 13));
            }
    }
};

// One packed interval per (a,b): low decade in bits 0..15, exclusive stop
// in bits 16..31. Rounding OUTWARD keeps every admissible c. Each part uses
// one c per decade; consequently every part has the same query count.
inline uint32_t pack(uint32_t low, uint32_t stop) {
    return low | (stop << 16);
}
inline uint32_t plan_count(uint32_t p) {
    return (p >> 16) - (p & 65535u);
}
inline uint32_t rounded_interval(const Spec &, uint32_t lo, uint32_t stop) {
    return lo >= stop ? 0 : pack((lo - LOW) / 10, (stop - LOW + 9) / 10);
}
inline std::pair<uint32_t, uint32_t> exact_interval(const Spec &s, uint32_t a, uint32_t b) {
    I target = -f(4, a) - f(3, b);
    I lower = target - f(1, LOW + 10 * s.nd - 1) - f(0, LOW + s.ne - 1);
    I upper = target - f(1, LOW) - f(0, LOW);
    uint32_t lo = LOW, hi = s.chi + 1;
    while (lo < hi) {
        uint32_t m = lo + (hi - lo) / 2;
        if (f(2, m) < lower)
            lo = m + 1;
        else
            hi = m;
    }
    uint32_t first = lo;
    lo = LOW;
    hi = s.chi + 1;
    while (lo < hi) {
        uint32_t m = lo + (hi - lo) / 2;
        if (f(2, m) <= upper)
            lo = m + 1;
        else
            hi = m;
    }
    return {first, lo};
}
inline std::vector<uint32_t> generate_plan(const Spec &s, uint32_t start, uint32_t count) {
    if (start < LOW || !count || start > s.ahi || count > s.ahi - start + 1 || count > s.chunk_a)
        throw std::runtime_error("Invalid in-memory plan chunk");
    uint32_t nb = s.bhi - LOW + 1, nc = s.chi - LOW + 1;
    std::vector<I> fc(nc), fb(nb), fa(count);
    for (uint32_t i = 0; i < nc; ++i)
        fc[i] = f(2, LOW + i);
    for (uint32_t i = 0; i < nb; ++i)
        fb[i] = f(3, LOW + i);
    for (uint32_t i = 0; i < count; ++i)
        fa[i] = f(4, start + i);
    I minpair = f(1, LOW) + f(0, LOW);
    I maxpair = f(1, LOW + 10 * s.nd - 1) + f(0, LOW + s.ne - 1);
    std::vector<uint32_t> plans(uint64_t(count) * nb);
    // Exact monotone sweep: as b increases, both c pointers only move down.
    // The optional OpenMP row parallelism is identical to the original sweep.
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t row = 0; row < count; ++row) {
        uint32_t lo = nc, stop = nc;
        for (uint32_t j = 0; j < nb; ++j) {
            uint32_t packed;
            if (s.tiny)
                packed = pack(0, nc / 10); // Tiny oracle: no pruning.
            else {
                I partial = -fa[size_t(row)] - fb[j];
                I lower = partial - maxpair, upper = partial - minpair;
                while (lo && fc[lo - 1] >= lower)
                    --lo;
                while (stop && fc[stop - 1] > upper)
                    --stop;
                packed = rounded_interval(s, LOW + lo, LOW + stop);
            }
            plans[size_t(row) * nb + j] = packed;
        }
    }
    // Independently binary-search sampled intervals in every generated row.
    if (!s.tiny)
        for (uint32_t row = 0; row < count; ++row)
            for (uint32_t q = 0; q < 16; ++q) {
                uint32_t b = LOW + uint32_t((uint64_t(q) * 7919 + row * 131) % nb);
                auto interval = exact_interval(s, start + row, b);
                if (plans[uint64_t(row) * nb + b - LOW] !=
                    rounded_interval(s, interval.first, interval.second))
                    throw std::runtime_error("Plan sweep/binary-search disagreement");
            }
    return plans;
}

// One collision-preserving linked table, reused for each part. END marks the
// final pair in a chain; all pair IDs fit under MASK. No disk representation.
struct Table {
    std::vector<uint32_t> head, next;
    explicit Table(const Spec &s) : head(s.modulus), next(s.pair_count()) {}
    void build(const Aux &aux, uint32_t part) {
        const Spec &s = aux.s;
        if (part > 9 || head.size() != s.modulus || next.size() != s.pair_count() ||
            s.pair_count() > MASK)
            throw std::runtime_error("Invalid pair table configuration");
        std::fill(head.begin(), head.end(), EMPTY);
        std::fill(next.begin(), next.end(), EMPTY);
        for (uint32_t i = 0; i < s.nd; ++i) {
            uint32_t kd = aux.h[WIDTH + LOW + part + 10 * i], id = i * s.ne;
            for (uint32_t j = 0; j < s.ne; ++j, ++id) {
                uint32_t key = kd + aux.h[LOW + j];
                if (key >= s.modulus)
                    key -= s.modulus;
                uint32_t old = head[key];
                if (old == EMPTY)
                    head[key] = id | END;
                else {
                    next[id] = old;
                    head[key] = id;
                }
            }
        }
    }
};

inline void verify_bounds() {
    I best = -f(4, LOW);
    for (uint32_t a = LOW + 1; a < WIDTH; ++a)
        best = std::max(best, -f(4, a));
    for (uint32_t j = 0; j < 4; ++j)
        for (uint32_t x = LOW + 1; x < WIDTH; ++x)
            if (f(j, x) <= f(j, x - 1))
                throw std::runtime_error("Monotonicity check failed");
    const uint32_t highs[4] = {88239, 88239, 88239, 88240};
    for (uint32_t j = 0; j < 4; ++j) {
        I minsum = 0;
        for (uint32_t k = 0; k < 4; ++k)
            if (k != j)
                minsum += f(k, LOW);
        if (f(j, highs[j] + 1) + minsum <= best)
            throw std::runtime_error("Upper bound certificate failed");
    }
}

// Unsigned checksum addition intentionally wraps modulo 2^64. It is independent
// of launch dimensions and is diagnostic, never part of the acceptance test.
struct Stats {
    uint64_t queries, candidates, checksum, passed13, solutions;
};
static_assert(std::is_trivial<Stats>::value && sizeof(Stats) == 40);
HD inline Stats plus(Stats a, Stats b) {
    return {a.queries + b.queries, a.candidates + b.candidates, a.checksum + b.checksum,
            a.passed13 + b.passed13, a.solutions + b.solutions};
}
struct Solution {
    uint32_t a, b, c, d, e;
};
} // namespace p5
