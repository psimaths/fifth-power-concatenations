# Coverage and exactness

All intervals below have integer endpoints. The problem concerns ordered tuples
`(a,b,c,d,e)` with every coordinate in `[10000,99999]`; distinctness is not
required. Put `B=100000`, `L=10000`, and `F_j(x)=x^5-B^j x`.

## 1. Domain reduction

The original equation is equivalent to

    F_4(a)+F_3(b)+F_2(c)+F_1(d)+F_0(e)=0.

For `j <= 3`, `F_j` is strictly increasing on the five-digit domain, since
`5x^4-B^j > 0` there. Let `M=max_{L<=a<B}(-F_4(a))`. A necessary condition for
a coordinate `x_j`, for `j<=3`, is

    F_j(x_j) + sum_{0<=k<=3, k!=j} F_k(L) <= M.

The code evaluates `M` over all possible `a`, verifies monotonicity by exact
finite differences, and checks that this inequality fails at `x_j=88240` for
`j=0,1,2` and at `x_3=88241`. Monotonicity then excludes every larger value.
Thus the retained bounds are `a<=99999`, `b<=88240`, `c,d,e<=88239`.
In particular, `a=99999` is included. These checks use signed 128-bit integers.

## 2. Shards and the lookup key

For every integer `x`, `x^5=x (mod 30)`. Also `B^j=10 (mod 30)` for `j>=1`.
Let `ell(x)=(7x) mod 10`, with residues in `[0,9]`. Define

    Q_0(x) = F_0(x)/30
    Q_j(x) = (F_j(x)-3 ell(x))/30  for j>=1.

These divisions are exact. The original equation modulo 10 implies
`a+b+c+d=0 (mod 10)`. In part `r=d mod 10`, the only possible last digit of `c`
is `t=(-a-b-r) mod 10`. The ten parts are disjoint and cover every possible `d`.

The sum `ell(a)+ell(b)+ell(t)+ell(r)` is divisible by ten. Writing its quotient
as `carry`, a solution must satisfy

    Q_1(d)+Q_0(e) = -Q_4(a)-Q_3(b)-Q_2(c)-carry.

With `P=948423849` and `h_j(x)=Q_j(x) mod P`, the bucket queried is therefore

    (-h_4(a)-h_3(b)-h_2(c)-carry) mod P.

No primality or injectivity assumption is made about `P`. Modular equality is
only a necessary condition; collisions are resolved by exact verification.

Each part inserts every pair

    d=L+r+10i,  0<=i<7824
    e=L+j,     0<=j<78240.

The ID is `78240i+j`. Inserting a collision links the previous head instead of
replacing its contents. By induction, every bucket contains all of its pairs
exactly once. Successor IDs were inserted earlier, so correctly constructed
chains cannot cycle. The kernel traverses each whole chain. IDs fit below
`2^30`, leaving the termination flag disjoint from the ID bits.

## 3. Interval pruning

For fixed `(a,b)`, set `T=-F_4(a)-F_3(b)`,

    m = F_1(L)+F_0(L)
    U = F_1(88239)+F_0(88239).

Every solution satisfies `T-U <= F_2(c) <= T-m`. Strict monotonicity of `F_2`
gives an exact interval `[first,stop)` clipped to `[10000,88240)`. Equality at
both magnitude boundaries is retained. As `b` increases, both endpoints move
only downward; this permits the two-pointer sweep in `generate_plan`.
Independent binary searches check sampled intervals in each generated row.

The stored interval is rounded outward to whole decades:

    lo=floor((first-L)/10), hi=ceil((stop-L)/10).

For each `k` in `[lo,hi)`, part `r` examines `c=L+10k+t`. Rounding adds candidates,
never removes them. The domain contains a whole number of decades, so this
cannot leave the retained domain. Empty intervals are retained as empty rows.
Across all ten parts, each admissible `c` occurs in exactly its required shard.

Rows are flattened as

    row=(a-L)*78241+(b-L),  0<=row<7041690000.

The host dispatches contiguous nonoverlapping intervals of these rows, including
empty rows, in every part. It independently sums generated interval lengths and
checks the GPU batch and shard totals. Plans and tables exist only in memory.

## 4. Exact acceptance

A modulo-13 filter is applied before exact equality. It cannot reject an exact
solution. The GPU then compares

    F_1(d)+F_0(e)+F_2(c) = -F_4(a)-F_3(b).

Each side is represented by three unsigned 32-bit limbs, with negative values
encoded modulo `2^96`. For `0<=x<100000` and `0<=j<=4`, `|F_j(x)|<10^25`.
The difference between the two sides consequently has magnitude less than
`5*10^25<2^86`. It cannot be a nonzero multiple of `2^96`. Thus this limb equality
is an exact integer equality, not a probabilistic hash test.

Every returned tuple is checked again using host signed 128-bit arithmetic and
later Python arbitrary-precision integers. A bounded result queue holds 4096
solutions per batch. Overflow or inconsistent accounting aborts the invocation;
no successful completion record is emitted for an overflow.

## 5. Completeness statement

Assuming the documented arithmetic and table construction are executed correctly,
take any solution in the original five-digit domain. Section 1 retains its
coordinates; Section 2 assigns its unique shard and last digit of `c`; Section 3
retains its decade and row; the complete collision chain contains its `(d,e)`;
and Section 4 accepts it. Conversely, every accepted tuple satisfies the
original equation. Contiguous row coverage and disjoint shards avoid duplicate
enumeration.

Therefore, if a complete execution reports exactly one tuple, the computational
result is uniqueness among ordered tuples of five five-digit positive integers.
It does not concern other digit lengths or leading-zero blocks.

## 6. What the receipt does and does not establish

The evidence verifier checks log/accounting consistency: one start record,
ordered parts 0–9, contiguous row
coverage, batch-to-shard counters, equal planned query totals between parts,
solution equations and distinctness, and one final completion record. The
recorder also requires a zero exit, empty stderr, and unchanged source/binary
hashes. It records a clean compilation and binds its build manifest to the run.
It launches one process without automatic retries or resume inputs. The
standalone log verifier does not independently regenerate the full work plan;
the in-process host plan tally checks each GPU batch.

These checks are not a machine-checked proof of the CUDA implementation. The
query counter is derived from interval lengths; it is not an independent sensor
of every memory lookup. Checksums wrap modulo `2^64`. Startup arithmetic,
forced-positive/near-miss, and pair-oracle tests exercise important paths but do
not certify every hardware operation. A software-generated receipt cannot prove
that no external actor paused the operating system or fabricated a log.
Reproduction on independent hardware provides additional evidence.
