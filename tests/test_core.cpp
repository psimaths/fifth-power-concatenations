#include "../src/core.hpp"
#include <cassert>
#include <iostream>

int main() {
    using namespace p5;
    verify_bounds();
    const I values[] = {
        0, 1, -1, (I(1) << 32) - 1, I(1) << 32, (I(1) << 64) - 1, -(I(1) << 80), (I(1) << 84) - 1};
    for (I a : values) {
        assert(eq(add(wide(a), neg(wide(a))), wide(0)));
        for (I b : values)
            assert(eq(add(wide(a), wide(b)), wide(a + b)));
    }
    const Spec tiny = spec(true);
    const Aux aux(tiny);
    Table table(tiny);
    for (unsigned part = 0; part < 10; ++part) {
        table.build(aux, part);
        std::vector<unsigned> seen(tiny.pair_count(), 0);
        for (unsigned key = 0; key < tiny.modulus; ++key) {
            unsigned word = table.head[key];
            while (word != EMPTY) {
                unsigned id = word & MASK;
                assert(id < seen.size() && ++seen[id] == 1);
                unsigned d = LOW + part + 10 * (id / tiny.ne), e = LOW + id % tiny.ne;
                I exact = f(1, d) + f(0, e) - 3 * ((7 * part) % 10);
                assert(exact % 30 == 0 && mod(exact / 30, tiny.modulus) == key);
                if (word & END)
                    break;
                word = table.next[id];
            }
        }
        for (auto count : seen)
            assert(count == 1);
    }
    const Spec full = spec(false);
    for (unsigned a : {10000u, 39168u, 66874u, 99999u}) {
        const auto plan = generate_plan(full, a, 1);
        for (unsigned b = LOW; b <= full.bhi; ++b) {
            auto interval = exact_interval(full, a, b);
            assert(plan[b - LOW] == rounded_interval(full, interval.first, interval.second));
        }
    }
    std::cout
        << "Host arithmetic, bounds, complete tiny tables, and production intervals passed.\n";
}
