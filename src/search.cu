// One process, one GPU, all ten parts. All tables and plans live only in memory.
#include <cuda_runtime.h>
#include "core.hpp"
#include <climits>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>

using namespace p5;
#define CUDA_OK(expr)                                                                              \
    do {                                                                                           \
        cudaError_t err = (expr);                                                                  \
        if (err != cudaSuccess)                                                                    \
            throw std::runtime_error(std::string(#expr) + ": " + cudaGetErrorString(err));         \
    } while (0)

namespace {
constexpr unsigned RESULT_CAPACITY = 4096;
constexpr unsigned BATCH_ROWS = 1048576;

template <class T> class Device {
    T *data_ = nullptr;
    size_t size_;

  public:
    explicit Device(size_t n) : size_(n) {
        if (n)
            CUDA_OK(cudaMalloc(reinterpret_cast<void **>(&data_), n * sizeof(T)));
    }
    ~Device() {
        if (data_)
            cudaFree(data_);
    }
    Device(const Device &) = delete;
    Device &operator=(const Device &) = delete;
    T *get() const {
        return data_;
    }
    void upload(const T *src, size_t n) {
        if (n > size_)
            throw std::runtime_error("Device upload overflow");
        if (n)
            CUDA_OK(cudaMemcpy(data_, src, n * sizeof(T), cudaMemcpyHostToDevice));
    }
    void download(T *dst, size_t n) const {
        if (n > size_)
            throw std::runtime_error("Device download overflow");
        if (n)
            CUDA_OK(cudaMemcpy(dst, data_, n * sizeof(T), cudaMemcpyDeviceToHost));
    }
};
class Event {
    cudaEvent_t event_;

  public:
    Event() {
        CUDA_OK(cudaEventCreate(&event_));
    }
    ~Event() {
        cudaEventDestroy(event_);
    }
    Event(const Event &) = delete;
    Event &operator=(const Event &) = delete;
    cudaEvent_t get() const {
        return event_;
    }
};
struct View {
    const uint32_t *head, *next, *h;
    const Wide *exact;
    const uint8_t *r13;
    Spec s;
    uint32_t part;
};
__device__ inline uint64_t warp_sum(uint64_t x) {
    for (int shift = 16; shift; shift /= 2)
        x += __shfl_down_sync(0xffffffffu, (unsigned long long)x, shift);
    return x;
}
__device__ inline Stats warp_stats(Stats x) {
    return {warp_sum(x.queries), warp_sum(x.candidates), warp_sum(x.checksum), warp_sum(x.passed13),
            warp_sum(x.solutions)};
}
__device__ inline uint32_t reduce13(uint32_t x) {
    if (x >= 26)
        x -= 26;
    else if (x >= 13)
        x -= 13;
    return x;
}
struct Row {
    uint32_t a, b, c_first, n, base, r13;
    Wide target;
};

// The original production kernel: one block per (a,b), one thread per c
// modulo blockDim, and complete collision-chain traversal for every query.
// Filter=false exists only to cross-check the production forced-hit test.
template <bool Filter>
__global__ void search_kernel(View v, const uint32_t *plans, uint64_t first, Stats *partials,
                              Solution *solutions, unsigned *used) {
    const uint32_t packed = plans[blockIdx.x], lo = packed & 65535u, stop = packed >> 16;
    if (lo == stop) {
        if (!threadIdx.x)
            partials[blockIdx.x] = Stats{};
        return;
    }
    __shared__ Row row;
    __shared__ Stats warps[32];
    const uint64_t ab = first + blockIdx.x;
    if (!threadIdx.x) {
        const uint32_t nb = v.s.bhi - LOW + 1;
        row.a = LOW + uint32_t(ab / nb);
        row.b = LOW + uint32_t(ab % nb);
        uint32_t digit = (30 - row.a % 10 - row.b % 10 - v.part) % 10;
        row.c_first = LOW + 10 * lo + digit;
        row.n = stop - lo;
        uint32_t carry =
            ((7 * row.a) % 10 + (7 * row.b) % 10 + (7 * digit) % 10 + (7 * v.part) % 10) / 10;
        uint32_t sum = v.h[4 * WIDTH + row.a] + v.h[3 * WIDTH + row.b] + carry;
        if (sum >= v.s.modulus)
            sum -= v.s.modulus;
        if (sum >= v.s.modulus)
            sum -= v.s.modulus;
        row.base = sum ? v.s.modulus - sum : 0;
        uint32_t r = uint32_t(v.r13[4 * WIDTH + row.a]) + v.r13[3 * WIDTH + row.b];
        if (r >= 13)
            r -= 13;
        row.r13 = r ? 13 - r : 0;
        row.target = neg(add(v.exact[4 * WIDTH + row.a], v.exact[3 * WIDTH + row.b]));
    }
    __syncthreads();
    Stats local{};
    if (!threadIdx.x)
        local.queries = row.n;
    for (uint32_t i = threadIdx.x; i < row.n; i += blockDim.x) {
        const uint32_t c = row.c_first + 10 * i;
        uint32_t key = make_key(row.base, v.h[2 * WIDTH + c], v.s.modulus), word = v.head[key];
        if (word == EMPTY)
            continue;
        // Stable modulo-2^64 checksum: independent of batch and launch size.
        uint64_t weight = ab * ((v.s.chi - LOW + 1) / 10) + (lo + i) + 1;
        while (true) {
            uint32_t id = word & MASK, di = id / v.s.ne, ej = id - di * v.s.ne;
            uint32_t d = LOW + v.part + 10 * di, e = LOW + ej;
            ++local.candidates;
            local.checksum += (uint64_t(id) + 1) * weight;
            bool pass = true;
            if constexpr (Filter) {
                pass = reduce13(uint32_t(v.r13[WIDTH + d]) + v.r13[e] + v.r13[2 * WIDTH + c]) ==
                       row.r13;
                local.passed13 += pass;
            }
            if (pass &&
                eq(add(add(v.exact[WIDTH + d], v.exact[e]), v.exact[2 * WIDTH + c]), row.target)) {
                ++local.solutions;
                unsigned index = atomicAdd(used, 1u);
                if (index < RESULT_CAPACITY)
                    solutions[index] = {row.a, row.b, c, d, e};
            }
            if (word & END)
                break;
            word = v.next[id];
        }
    }
    local = warp_stats(local);
    if ((threadIdx.x & 31u) == 0)
        warps[threadIdx.x / 32] = local;
    __syncthreads();
    if (threadIdx.x < 32) {
        local = threadIdx.x < blockDim.x / 32 ? warps[threadIdx.x] : Stats{};
        local = warp_stats(local);
        if (!threadIdx.x)
            partials[blockIdx.x] = local;
    }
}
__global__ void reduce_kernel(const Stats *parts, size_t n, Stats *total) {
    __shared__ Stats warps[32];
    Stats s{};
    for (size_t i = threadIdx.x; i < n; i += blockDim.x)
        s = plus(s, parts[i]);
    s = warp_stats(s);
    if ((threadIdx.x & 31u) == 0)
        warps[threadIdx.x / 32] = s;
    __syncthreads();
    if (threadIdx.x < 32) {
        s = threadIdx.x < blockDim.x / 32 ? warps[threadIdx.x] : Stats{};
        s = warp_stats(s);
        if (!threadIdx.x)
            *total = s;
    }
}

__global__ void arithmetic_kernel(const Wide *xs, const Wide *expected, unsigned n,
                                  unsigned *errors) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * n)
        return;
    unsigned a = i / n, b = i % n;
    if (!eq(add(xs[a], xs[b]), expected[i]) || !eq(add(xs[a], neg(xs[a])), Wide{0, 0, 0}))
        atomicAdd(errors, 1u);
}
void arithmetic_test() {
    std::vector<I> integers = {0,
                               1,
                               -1,
                               I(0xffffffffu),
                               I(1) << 32,
                               -(I(1) << 32),
                               (I(1) << 64) - 1,
                               I(1) << 64,
                               -(I(1) << 80),
                               (I(1) << 84) - 1};
    std::vector<Wide> xs, expected;
    for (I x : integers)
        xs.push_back(wide(x));
    for (I a : integers)
        for (I b : integers)
            expected.push_back(wide(a + b));
    Device<Wide> dx(xs.size()), de(expected.size());
    Device<unsigned> errors(1);
    unsigned count = 0;
    dx.upload(xs.data(), xs.size());
    de.upload(expected.data(), expected.size());
    errors.upload(&count, 1);
    arithmetic_kernel<<<1, 256>>>(dx.get(), de.get(), unsigned(xs.size()), errors.get());
    CUDA_OK(cudaGetLastError());
    errors.download(&count, 1);
    if (count)
        throw std::runtime_error("GPU 96-bit arithmetic self-test failed");
}

struct Probe {
    uint32_t key, c, target13;
    Wide target;
};
__global__ void probe_kernel(View v, const Probe *probes, Stats *out, unsigned n) {
    unsigned q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= n)
        return;
    const Probe p = probes[q];
    Stats s{};
    uint32_t word = v.head[p.key];
    while (word != EMPTY) {
        uint32_t id = word & MASK, d = LOW + v.part + 10 * (id / v.s.ne), e = LOW + id % v.s.ne;
        ++s.candidates;
        s.checksum += uint64_t(id) + 1;
        if (reduce13(uint32_t(v.r13[WIDTH + d]) + v.r13[e] + v.r13[2 * WIDTH + p.c]) ==
            p.target13) {
            ++s.passed13;
            s.solutions +=
                eq(add(add(v.exact[WIDTH + d], v.exact[e]), v.exact[2 * WIDTH + p.c]), p.target);
        }
        if (word & END)
            break;
        word = v.next[id];
    }
    out[q] = s;
}
void validate_probes(const Table &table, View v) {
    // Independent pair oracle: sort e keys and invert each chosen key for
    // EVERY d. This retains all collisions without constructing a second table.
    std::vector<std::pair<uint32_t, uint32_t>> es;
    for (uint32_t j = 0; j < v.s.ne; ++j)
        es.emplace_back(reduced_key(0, LOW + j, v.s.modulus), j);
    std::sort(es.begin(), es.end());
    std::vector<Probe> probes;
    std::vector<Stats> wanted;
    for (unsigned q = 0; q < 32; ++q) {
        uint32_t di = (q * 139u) % v.s.nd, ej = (q * 1723u) % v.s.ne;
        uint32_t d0 = LOW + v.part + 10 * di, e0 = LOW + ej,
                 c = LOW + (q * 97u) % (v.s.chi - LOW + 1);
        I target = f(1, d0) + f(0, e0) + f(2, c) + (q % 2 ? 13 : 0);
        uint32_t key = reduced_key(1, d0, v.s.modulus) + reduced_key(0, e0, v.s.modulus);
        if (key >= v.s.modulus)
            key -= v.s.modulus;
        probes.push_back({key, c, mod(target, 13), wide(target)});
        Stats s{};
        std::vector<uint32_t> expected, actual;
        for (uint32_t i = 0; i < v.s.nd; ++i) {
            uint32_t d = LOW + v.part + 10 * i,
                     ke = make_key(key, reduced_key(1, d, v.s.modulus), v.s.modulus);
            auto it = std::lower_bound(es.begin(), es.end(), std::make_pair(ke, 0u));
            while (it != es.end() && it->first == ke) {
                uint32_t e = LOW + it->second, id = i * v.s.ne + it->second;
                I sum = f(1, d) + f(0, e) + f(2, c);
                expected.push_back(id);
                ++s.candidates;
                s.checksum += uint64_t(id) + 1;
                s.passed13 += mod(sum - target, 13) == 0;
                s.solutions += sum == target;
                ++it;
            }
        }
        // Compare actual host chain IDs, not just the GPU's checksum.
        uint32_t word = table.head[key];
        while (word != EMPTY) {
            uint32_t id = word & MASK;
            if (id >= v.s.pair_count() || actual.size() >= v.s.pair_count())
                throw std::runtime_error("Pair table cycle/out-of-range ID");
            actual.push_back(id);
            if (word & END)
                break;
            word = table.next[id];
        }
        std::sort(expected.begin(), expected.end());
        std::sort(actual.begin(), actual.end());
        if (actual != expected)
            throw std::runtime_error("Independent pair-table oracle failed");
        wanted.push_back(s);
    }
    Device<Probe> dp(probes.size());
    Device<Stats> ds(probes.size());
    dp.upload(probes.data(), probes.size());
    probe_kernel<<<1, 128>>>(v, dp.get(), ds.get(), unsigned(probes.size()));
    CUDA_OK(cudaGetLastError());
    std::vector<Stats> got(probes.size());
    ds.download(got.data(), got.size());
    for (size_t i = 0; i < got.size(); ++i)
        if (std::memcmp(&got[i], &wanted[i], sizeof(Stats)))
            throw std::runtime_error("GPU/independent pair oracle mismatch");
}
void validate_main_positive(const Aux &aux, View v, unsigned threads) {
    // Exercise the ACTUAL production kernel and result queue. Temporarily
    // replace F4(LOW) by a synthetic value forcing a table pair to solve a row.
    // This has no dependency on any solution of the research problem.
    const uint32_t a = LOW, b = LOW, c = LOW + (10 - v.part) % 10, d = LOW + v.part, e = LOW;
    const size_t at = 4 * WIDTH + a;
    const I forced = -f(3, b) - f(2, c) - f(1, d) - f(0, e);
    if (forced % 30)
        throw std::runtime_error("Synthetic key unexpectedly nonintegral");
    uint32_t key = mod(forced / 30, v.s.modulus);
    Wide exact = wide(forced);
    uint8_t r = uint8_t(mod(forced, 13));
    CUDA_OK(cudaMemcpy(const_cast<uint32_t *>(v.h) + at, &key, sizeof key, cudaMemcpyHostToDevice));
    CUDA_OK(
        cudaMemcpy(const_cast<Wide *>(v.exact) + at, &exact, sizeof exact, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(const_cast<uint8_t *>(v.r13) + at, &r, sizeof r, cudaMemcpyHostToDevice));
    Device<uint32_t> plan(1);
    uint32_t packed = pack(0, 1);
    plan.upload(&packed, 1);
    Device<Stats> stats(1);
    Device<Solution> records(RESULT_CAPACITY);
    Device<unsigned> used(1);
    Stats baseline{};
    for (unsigned negative = 0; negative < 2; ++negative) {
        if (negative) {
            // +13 preserves mod-13 acceptance. Keep the key fixed as well:
            // only the exact 96-bit comparison can reject this near miss.
            exact = wide(forced + 13);
            CUDA_OK(cudaMemcpy(const_cast<Wide *>(v.exact) + at, &exact, sizeof exact,
                               cudaMemcpyHostToDevice));
        }
        for (unsigned mode = 0; mode < 2; ++mode) {
            unsigned count = 0;
            used.upload(&count, 1);
            if (!mode)
                search_kernel<true>
                    <<<1, threads>>>(v, plan.get(), 0, stats.get(), records.get(), used.get());
            else
                search_kernel<false>
                    <<<1, threads>>>(v, plan.get(), 0, stats.get(), records.get(), used.get());
            CUDA_OK(cudaGetLastError());
            Stats got{};
            stats.download(&got, 1);
            used.download(&count, 1);
            if (got.queries != 1 || got.solutions != count || count > RESULT_CAPACITY)
                throw std::runtime_error("Production-kernel synthetic accounting failed");
            if (!negative && !mode)
                baseline = got;
            if (got.candidates != baseline.candidates || got.checksum != baseline.checksum)
                throw std::runtime_error("Synthetic test changed table traversal");
            if (negative) {
                if (count)
                    throw std::runtime_error("Production exact check accepted a near miss");
            } else {
                std::vector<Solution> found(count);
                records.download(found.data(), count);
                bool known = false;
                for (const auto &x : found)
                    known |= x.a == a && x.b == b && x.c == c && x.d == d && x.e == e;
                if (!known)
                    throw std::runtime_error("Production kernel/queue missed the forced solution");
            }
        }
    }
    CUDA_OK(cudaMemcpy(const_cast<uint32_t *>(v.h) + at, aux.h.data() + at, sizeof(uint32_t),
                       cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(const_cast<Wide *>(v.exact) + at, aux.exact.data() + at, sizeof(Wide),
                       cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(const_cast<uint8_t *>(v.r13) + at, aux.r13.data() + at, sizeof(uint8_t),
                       cudaMemcpyHostToDevice));
}

std::string json_quote(const std::string &s) {
    const char *hex = "0123456789abcdef";
    std::string out = "\"";
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') {
            out += '\\';
            out += char(c);
        } else if (c < 32) {
            out += "\\u00";
            out += hex[c >> 4];
            out += hex[c & 15];
        } else
            out += char(c);
    }
    return out + '"';
}
void print_solutions(const std::vector<Solution> &solutions) {
    std::cout << '[';
    for (size_t i = 0; i < solutions.size(); ++i) {
        if (i)
            std::cout << ',';
        const auto &s = solutions[i];
        std::cout << '[' << s.a << ',' << s.b << ',' << s.c << ',' << s.d << ',' << s.e << ']';
    }
    std::cout << ']';
}
void print_stats(const Stats &s, const std::vector<Solution> &solutions) {
    std::cout << ",\"queries\":" << s.queries << ",\"candidates\":" << s.candidates
              << ",\"checksum\":" << s.checksum << ",\"passed13\":" << s.passed13
              << ",\"solutions_count\":" << s.solutions << ",\"solutions\":";
    print_solutions(solutions);
}
unsigned parse_unsigned(const char *text) {
    std::string s = text;
    if (s.empty())
        throw std::runtime_error("Empty numeric option");
    uint64_t n = 0;
    for (char c : s) {
        if (c < '0' || c > '9')
            throw std::runtime_error("Expected unsigned decimal option");
        n = n * 10 + unsigned(c - '0');
        if (n > INT_MAX)
            throw std::runtime_error("Numeric option out of range");
    }
    return unsigned(n);
}
struct Options {
    bool tiny = false, help = false;
    unsigned device = 0, threads = 256;
};
Options options(int argc, char **argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help")
            o.help = true;
        else if (arg == "--tiny")
            o.tiny = true;
        else if (arg == "--device" || arg == "--threads") {
            if (++i == argc)
                throw std::runtime_error("Missing value for " + arg);
            unsigned n = parse_unsigned(argv[i]);
            if (arg == "--device")
                o.device = n;
            else
                o.threads = n;
        } else
            throw std::runtime_error("Unknown option: " + arg);
    }
    if (o.threads != 128 && o.threads != 256 && o.threads != 512)
        throw std::runtime_error("--threads must be 128, 256, or 512");
    return o;
}
void validate_hit(const Spec &s, const Solution &x, uint32_t part, uint64_t first, uint64_t next) {
    if (x.a < LOW || x.a > s.ahi || x.b < LOW || x.b > s.bhi || x.c < LOW || x.c > s.chi ||
        x.d < LOW || x.d > LOW + 10 * s.nd - 1 || x.e < LOW || x.e > LOW + s.ne - 1 ||
        x.d % 10 != part || !original_solution(x.a, x.b, x.c, x.d, x.e))
        throw std::runtime_error("GPU solution failed host signed128 verification");
    uint64_t ab = uint64_t(x.a - LOW) * (s.bhi - LOW + 1) + x.b - LOW;
    if (ab < first || ab >= next)
        throw std::runtime_error("GPU solution outside dispatched rows");
}
} // namespace

int main(int argc, char **argv) {
    try {
        Options o = options(argc, argv);
        if (o.help) {
            std::cout
                << "Usage: search [--tiny] [--device N] [--threads 128|256|512] [--help]\n"
                << "Default: full exhaustive search, all ten shards, device 0, 256 threads.\n"
                << "--tiny selects the diagnostic domain (400 rows, 1600 queries per shard).\n"
                << "No resume, checkpoints, saved plans, or partial-run controls.\n";
            return 0;
        }
        std::cout.exceptions(std::ios::badbit | std::ios::failbit);
        std::cout << std::setprecision(12) << std::unitbuf;
        const Spec s = spec(o.tiny);
        verify_bounds();
        CUDA_OK(cudaSetDevice(int(o.device)));
        cudaDeviceProp prop{};
        CUDA_OK(cudaGetDeviceProperties(&prop, int(o.device)));
        const unsigned capacity = unsigned(std::min<uint64_t>(BATCH_ROWS, s.ab_count()));
        if (o.threads > unsigned(prop.maxThreadsPerBlock) ||
            capacity > unsigned(prop.maxGridSize[0]) || prop.warpSize != 32)
            throw std::runtime_error("Unsupported GPU launch configuration");
        arithmetic_test();
        size_t free = 0, total = 0;
        CUDA_OK(cudaMemGetInfo(&free, &total));
        const uint64_t need = 4 * (uint64_t(s.modulus) + s.pair_count()) + 8500000ULL +
                              capacity * (sizeof(Stats) + sizeof(uint32_t)) +
                              RESULT_CAPACITY * sizeof(Solution) + sizeof(Stats) + sizeof(unsigned);
        if (free < need)
            throw std::runtime_error("Insufficient free GPU memory");
        Aux aux(s);
        Table table(s);
        Device<uint32_t> head(s.modulus), next(s.pair_count()), h(aux.h.size()), plan(capacity);
        Device<Wide> exact(aux.exact.size());
        Device<uint8_t> r13(aux.r13.size());
        Device<Stats> partials(capacity), sum(1);
        Device<Solution> found(RESULT_CAPACITY);
        Device<unsigned> used(1);
        h.upload(aux.h.data(), aux.h.size());
        exact.upload(aux.exact.data(), aux.exact.size());
        r13.upload(aux.r13.data(), aux.r13.size());
        Event begin, end;
        std::cout << "{\"kind\":\"start\",\"schema_version\":1,\"profile\":"
                  << json_quote(o.tiny ? "tiny" : "full") << ",\"rows_per_part\":" << s.ab_count()
                  << ",\"device\":" << json_quote(prop.name) << ",\"device_index\":" << o.device
                  << ",\"threads\":" << o.threads
                  << ",\"self_tests\":\"passed\",\"config\":{\"ahi\":" << s.ahi
                  << ",\"bhi\":" << s.bhi << ",\"chi\":" << s.chi << ",\"ne\":" << s.ne
                  << ",\"nd\":" << s.nd << ",\"modulus\":" << s.modulus
                  << ",\"chunk_a\":" << s.chunk_a << ",\"batch_rows\":" << capacity << "}}\n";
        Stats grand{};
        std::vector<Solution> all_solutions;
        uint64_t common_queries = 0;
        for (uint32_t part = 0; part < 10; ++part) {
            std::cout << "{\"kind\":\"shard_start\",\"part\":" << part << "}\n";
            table.build(aux, part);
            head.upload(table.head.data(), table.head.size());
            next.upload(table.next.data(), table.next.size());
            View view{head.get(), next.get(), h.get(), exact.get(), r13.get(), s, part};
            validate_probes(table, view);
            validate_main_positive(aux, view, o.threads);
            Stats shard{};
            std::vector<Solution> shard_solutions;
            uint64_t rows = 0, planned_queries = 0;
            for (uint32_t a = LOW; a <= s.ahi; a += s.chunk_a) {
                auto plans = generate_plan(s, a, std::min(s.chunk_a, s.ahi - a + 1));
                // Tally the entire generated chunk separately from GPU batches.
                for (uint32_t packed : plans) {
                    if ((packed & 65535u) > (packed >> 16) ||
                        (packed >> 16) > (s.chi - LOW + 1) / 10)
                        throw std::runtime_error("Invalid generated interval");
                    planned_queries += plan_count(packed);
                }
                const uint64_t chunk_first = uint64_t(a - LOW) * (s.bhi - LOW + 1);
                for (size_t offset = 0; offset < plans.size(); offset += capacity) {
                    unsigned n = unsigned(std::min<size_t>(capacity, plans.size() - offset));
                    uint64_t first = chunk_first + offset;
                    if (first != rows)
                        throw std::runtime_error("Noncontiguous dispatch");
                    uint64_t expected_queries = 0;
                    for (unsigned i = 0; i < n; ++i)
                        expected_queries += plan_count(plans[offset + i]);
                    plan.upload(plans.data() + offset, n);
                    unsigned count = 0;
                    used.upload(&count, 1);
                    CUDA_OK(cudaEventRecord(begin.get()));
                    search_kernel<true><<<n, o.threads>>>(view, plan.get(), first, partials.get(),
                                                          found.get(), used.get());
                    CUDA_OK(cudaGetLastError());
                    reduce_kernel<<<1, 256>>>(partials.get(), n, sum.get());
                    CUDA_OK(cudaGetLastError());
                    CUDA_OK(cudaEventRecord(end.get()));
                    CUDA_OK(cudaEventSynchronize(end.get()));
                    float milliseconds = 0;
                    CUDA_OK(cudaEventElapsedTime(&milliseconds, begin.get(), end.get()));
                    Stats batch{};
                    sum.download(&batch, 1);
                    used.download(&count, 1);
                    if (count > RESULT_CAPACITY || batch.solutions != count)
                        throw std::runtime_error(
                            "Solution buffer overflow/accounting mismatch; search incomplete");
                    if (batch.queries != expected_queries)
                        throw std::runtime_error("GPU query-coverage counter mismatch");
                    std::vector<Solution> solutions(count);
                    found.download(solutions.data(), count);
                    for (const auto &hit : solutions)
                        validate_hit(s, hit, part, first, first + n);
                    shard = plus(shard, batch);
                    shard_solutions.insert(shard_solutions.end(), solutions.begin(),
                                           solutions.end());
                    rows += n;
                    std::cout << "{\"kind\":\"batch\",\"part\":" << part
                              << ",\"first_pair\":" << first << ",\"next_pair\":" << rows;
                    print_stats(batch, solutions);
                    std::cout << ",\"device_seconds\":" << milliseconds / 1000.0 << "}\n";
                }
            }
            if (rows != s.ab_count() || shard.queries != planned_queries ||
                shard.solutions != shard_solutions.size())
                throw std::runtime_error("Shard coverage/accounting mismatch");
            if (part == 0)
                common_queries = planned_queries;
            else if (planned_queries != common_queries)
                throw std::runtime_error("Shards disagree on total planned queries");
            grand = plus(grand, shard);
            all_solutions.insert(all_solutions.end(), shard_solutions.begin(),
                                 shard_solutions.end());
            std::cout << "{\"kind\":\"shard_complete\",\"part\":" << part << ",\"rows\":" << rows;
            print_stats(shard, shard_solutions);
            std::cout << "}\n";
        }
        if (grand.queries != 10 * common_queries || grand.solutions != all_solutions.size())
            throw std::runtime_error("Final coverage/accounting mismatch");
        CUDA_OK(cudaDeviceSynchronize());
        std::cout << "{\"kind\":\"complete\",\"parts\":10,\"queries\":" << grand.queries
                  << ",\"solutions_count\":" << grand.solutions << ",\"solutions\":";
        print_solutions(all_solutions);
        std::cout << "}\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "search: " << e.what() << '\n';
        return 1;
    }
}
