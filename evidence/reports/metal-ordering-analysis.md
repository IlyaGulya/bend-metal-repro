# Metal memory-ordering analysis — bendlang/bend#2094

Scope: GPU memory ordering and scheduler handoff in the **failing** compiler checkout
`be236f324ed0338fe907fe50474468e4d3b94157` (`bend2/comp.ts`). Passing checkout:
`d18110b6d6e9a70bb8814b2a8d8db092e301e540`. Artifacts: this directory.
No builds/tests were run for this analysis. Runtime observations below are quoted from
artifacts produced by the parent agent; static claims are mine and cite exact lines.

Primary spec: *Metal Shading Language Specification*, 2026-06-04 revision
(the PDF currently served at `developer.apple.com/metal/Metal-Shading-Language-Specification.pdf`),
cited as **MSL §x (p.N)**. C++ fence rules are cited from C++17 N4659 §32.9 (the
standard revision MSL says it implements a subset of) and, where they differ, from the
current working draft.

---

## 0. Verdict

1. **The Metal ordering protocol used by the compiler is, in principle, a valid
   C++17 fence protocol on Metal 3.2+/Apple silicon.** `coherent(device)` plain
   accesses + `atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst)`
   (default `thread_scope_device`) around relaxed 32-bit atomics is the documented
   way to get device-scope release/acquire on Metal. MSL only allows relaxed
   atomics outside fences (§6.16.1 p.299), so this design is not merely acceptable,
   it is the only documented one for the deployment range in this source
   (`__METAL_VERSION__ >= 320`, `comp.ts:3201-3216`).

2. **One ordering sequence in the failing checkout is genuinely invalid**: the ring
   consumption-index (`get`) handoff. The consumer loads the slot, then publishes
   `get` with a *relaxed* store (`comp.ts:4459`) and the producer reads `get` with a
   *relaxed* load and no acquire fence (`comp.ts:4209`). That permits the producer to
   reuse/write the slot while the consumer's second 32-bit slot load is still in
   flight, i.e. a torn 64-bit `Term`. It is fixed in the passing tree by commit
   `335f239a` ("a ring pop publishes get with release, a push reads it with acquire",
   passing `comp.ts:4213,4462`). **This is a real defect, but it is not the mechanism
   of this reproducer** (see §3): the first divergence happens with one lane, no ring
   wrap, no producer/consumer overlap, and in code that executes no such handoff.

3. **The other orderings are valid under the documented model, with two qualifications**:
   (a) the continuation refcount acquire uses a release fence + *RMW* and an acquire
   fence after an *RMW*; this is valid under C++17 `[atomics.fences]/2` but is
   excluded by the *current draft*'s added "Y is not an atomic modify-write operation"
   restriction, so it is fragile; (b) `term_peek`'s 64-bit plain read of a cell that
   other lanes bump with a 32-bit atomic RMW is outside the atomic model ("relies on
   a hardware/layout invariant", not on a guarantee). There is also a wrong barrier
   scope in `bank_pack` (threadgroup-only barrier around device-memory copies).

4. **#2094's observed failure is not a memory-ordering race.** The earliest divergence
   is at the device `WL_RETN` inside `FID_ir_eval`: the stack word for `FID_EXIT` is
   correct and has just been read as `0xee`, the result word is correct, and yet the
   dispatch to `FID_EXIT` does not happen; `work_loop` returns an undefined value
   (varies run to run), and the caller re-enters `FID_ENTER` with it, producing
   `ERR_FIDS` ("a function the device does not hold") or a drained frontier. The
   inspected AIR has a **source-correct** cleanup edge
   (`cleanup232=55 → cleanup233=4 → switch epilog → cleanup336=0 → outer for`), so the
   defect is **downstream of AIR generation** (Metal backend / AGX lowering, or its
   execution); the exact lowering step is not identified. It is not a missing
   acquire/release edge: a store that is never executed cannot be made visible by any
   fence. The nondeterminism and the issue's reported null-edit sensitivity are
   evidence of codegen sensitivity, **not** evidence of a data race; I do not claim a
   race from outcome variation.

---

## 1. What MSL actually guarantees

### 1.1 Coherence, address space, `coherent(device)`

MSL §4.8 (pp.122-123): memory in the **device address space defaults to
*threadgroup* coherence**; `coherent(device)` (Metal 3.2+) marks a device object as
*device* coherent, i.e. "memory operations are visible across threads on the device
**if you properly synchronize them**". MSL §2.6 (p.39): "Objects of atomic types are
free from data races"; atomics are race-free only when the accessed object is a Metal
atomic type, and atomic functions may only use Metal atomic data types.

Source consequences:

* The whole corpus `H` is a `u64` buffer reached through `DEV u64*`, where
  `DEV` expands to `coherent(device) device` for MSL ≥ 3.2 (`comp.ts:3195-3206`).
  The build in `run-info.json` reports Metal 4, so this branch is active and plain
  `H[...]` accesses are device-coherent **provided the accesses are synchronized**.
* Every cast that produces a pointer used for cross-thread access preserves `DEV`:
  `A32()` (`comp.ts:3478-3484`), `a32_at()` (`comp.ts:3539`), `bank_at()`
  (`comp.ts:3690`), `turn_set`'s `(DEV u32*)(H + H_GROUPS)` (`comp.ts:4238`).
* If the same source were compiled with `__METAL_VERSION__ < 320`, `DEV` degrades to
  plain `device` (threadgroup-coherent) and *none* of the cross-threadgroup plain
  payload publication in this file is guaranteed. That is a latent portability bug,
  not this build.

### 1.2 Atomics are relaxed; fences carry order

MSL §6.16.1 (p.299), verbatim: **"For atomic operations other than
atomic_thread_fence, memory_order_relaxed is the only enumeration value."** Table 6.26
marks acquire/release/acq_rel as Metal 4.1+ *values*, but the sentence above is the
operative restriction for the atomics used here; the code maps `REL`, `ACQ`, `ACR`
all to `RLX` on device and adds explicit fences (`comp.ts:3522-3525`). This matches
the documented model, and `a32_sub_rel`/`a32_store_rel` are intentionally
fence+relaxed-op, not release ops (`comp.ts:3537-3538`).

### 1.3 Fence signature, flags, scope

MSL §6.16.3 (pp.300-301):

```c
void atomic_thread_fence(mem_flags flags, memory_order order,
                         thread_scope scope = thread_scope_device);
```

* `mem_flags::mem_device` is required for the *device* address space; a fence only
  orders the address spaces named in its flags.
* Default scope is **`thread_scope_device`** (§6.16.3 p.301), which is what a
  cross-threadgroup handoff in `H` needs.
* `memory_order_seq_cst` is "a sequentially consistent acquire **and** release fence"
  (§6.16.3 p.301). The file's `FENCE()` is exactly
  `atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst)`
  (`comp.ts:3213`), i.e. device-scope, device-memory, acquire+release.
* Spec caveat: the prose says a seq_cst fence "prevents all preceding writes from
  moving past all subsequent stores within that scope" (§6.16.3 p.301). It does not
  spell out load→store ordering in that sentence; the stronger acquire/release fence
  semantics are stated as "establishes memory synchronization ordering of nonatomic
  and relaxed atomic accesses, according to the memory order and thread scope", and
  are ultimately inherited from C++17. This distinction matters in §2.3.

### 1.4 The C++17 fence rules the code relies on

C++17 N4659 §32.9 `[atomics.fences]`:

* **/2 (fence–fence):** release fence A synchronizes with acquire fence B if
  `A ≺ X`, `X` modifies `M`, `Y ≺ B`, and `Y` reads `X` (or a hypothetical release
  sequence headed by `X`).
* **/3 (fence–atomic):** release fence A synchronizes with an atomic operation B
  that performs an **acquire operation** on `M` and reads `X`/its release sequence.
* **/4 (atomic–fence):** release operation A synchronizes with acquire fence B if
  some `X ≺ B` reads `A`/its release sequence.

The currently published draft adds "*where Y is not an atomic modify-write
operation*" to /2 (`[atomics.fences]/2`). This is the one place where the code's
validity depends on which revision is implemented (see §2.4).

---

## 2. The sequences, one by one

Line numbers below are failing checkout `be236f32` unless marked `passing`.

### 2.1 A32 vocabulary

```c
#define DEV coherent(device) device                      // comp.ts:3203
#define FENCE() atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst) // :3213
#define REL RLX / ACQ RLX / ACR RLX                      // :3522-3524 (device)
#define a32_acq(p) FENCE()                               // :3525 (device; p unused)
#define w64_load(p) (*(p))                               // :3526 (device: plain 64-bit)
#define a32_sub_rel(p,v) (FENCE(), atomic_fetch_sub_explicit(A32(p),v,REL)) // :3537
#define a32_store_rel(p,v)(FENCE(), atomic_store_explicit(A32(p),v,REL))    // :3538
INLINE u32 a32_load_acq(DEV u32* p) { u32 v=DEVICE?a32_load(p):...ACQ; FENCE(); return v; } // :3541-3546
```

So on device every "release" is a device-scope seq_cst fence followed by a relaxed
atomic op, and every "acquire" is a relaxed atomic load followed by a device-scope
seq_cst fence. Loads go through a `volatile thread u32` temporary to work around a
pipeline-build issue (`comp.ts:3465-3506`); `volatile` contributes no ordering, only
the fence does.

### 2.2 Ring slot publication — **valid**

Producer `ring_push` (`comp.ts:4207-4216`):

```c
u32 pos = a32_add(ring_put(H,r), 1);                     // claim (relaxed RMW)
if (pos - a32_load(ring_get(H,r)) >= RING_LEN) ...       // capacity (see §2.3)
a32_store(lo, (u32)tsk);                                 // RLX store, low word
a32_store_rel(lo+1, high | lap<<31);                     // FENCE, then RLX store
```

Consumer `monk_step` (`comp.ts:4449-4455`):

```c
u32 hi = a32_load_acq(lo + 1);                           // RLX load hi, then FENCE
Term t = ((u64)hi << 32) | a32_load(lo);                 // RLX load lo
if ((hi >> 31) != ring_lap(*get)) return 0;              // lap validates hi
```

This is a textbook release-store/acquire-load (strictly: release fence + relaxed
store, satisfied by C++17 /3 as an acquire *fence* after a relaxed load, and MSL
§6.16.3). The lap bit prevents accepting a not-yet-written slot, and the acquire
fence orders the following `lo` load after the released `hi` value. **Valid.**
The emulation is also empirically exercised here: the host pushes the root task with
a release store (`comp.ts:5328`), the device accepts it with `a32_load_acq` and reads
a correct 64-bit task (`trace-be236f32.log`: `fid=239 r0=0500c60003108d08`), and the
task's continuation is correct `~0` by the time `FID_ir_eval` runs.

### 2.3 Ring consumption-index handoff — **invalid in be236f32** (real bug, separate)

Failing consumer (`comp.ts:4454-4459`):

```c
u32 hi = a32_load_acq(lo + 1);   // fence is HERE, between hi and lo
Term t = (((u64)hi << 32) | a32_load(lo)) & ~RFC_BIT;
if (...) return 0;
a32_store(get, *get + 1);        // RLX store — no release fence after the lo load
```

Failing producer (`comp.ts:4208-4215`):

```c
u32 pos = a32_add(ring_put(H,r), 1);           // claim
if (pos - a32_load(ring_get(H,r)) >= RING_LEN) // RLX load of get — no acquire fence
...
a32_store(lo, low); a32_store_rel(lo+1, high|lap);
```

The existing `a32_load_acq(lo+1)` fence sits **between** the `hi` load and the `lo`
load. It orders `lo` after `hi` (so it prevents a hi/lo tear *of the same
publication*), but it is sequenced **before** the `lo` load and cannot act as the
release fence for the later `get` store: C++17 /2 requires the release fence to be
sequenced before the released atomic store, and the fence's happens-before only covers
evaluations before it. Nothing then orders the consumer's `lo` load before its
`get` store; nothing on the producer side orders the producer's slot stores after its
`get` load. When the ring wraps and the producer reuses this slot, the consumer can
observe the new low word with the old high word (the old lap bit still validates),
producing a valid-looking but garbage `Term` — precisely an `ERR_FIDS`-class failure
in a heavily reused ring.

Passing `335f239a` fixes exactly the two missing edges (passing `comp.ts:4213,4462`):

```c
if (pos - a32_load_acq(ring_get(H,r)) >= RING_LEN)   // producer: load get, then FENCE
...
a32_store_rel(get, *get + 1);                        // consumer: FENCE, then store get
```

Now the consumer's fence is after both slot loads and immediately before the `get`
store, so the `lo` load is carried (C++17 /2 with Y = producer's relaxed `get` load;
Y is not an RMW, so this remains valid under the current draft too). **This is a
genuine static bug, and the fix is minimal and correct.**

Caveats for this specific report:

* It requires slot reuse (a full `RING_LEN` lap) while a consumer load is in flight.
  `RING_LEN = 1<<(17-CUBE_LOG)` = 1024 with the default `CUBE_LOG=7`
  (`comp.ts:3357-3361`); the reproducer pushes a handful of tasks.
* The failing trace shows the first divergence **before** any wrap and with no
  concurrent producer (only lane 0 is active, §3).
* It does not explain the parent's "null edit / CUBE_LOG=0 still fails" reports by
  itself; those are codegen-sensitivity observations.

### 2.4 Task contents and continuation delivery/refcount — **valid in intent; fragile**

`task_deliver` (`comp.ts:4264-4281`):

```c
for (j...) H[at + j] = v[j];                       // plain, coherent(device) stores
if (cont == TERM_HOLE) {
  a32_store_rel(H_ROOT_DONE, n+1); return 0;       // FENCE + RLX store
}
DEV u32* c = a32_at(H, task_tail(cont)+1);
if (a32_sub_rel(c,1) == 1) {                       // FENCE + RLX fetch_sub
  a32_acq(c);                                      // bare FENCE
  return cont;                                     // last decrementer runs cont
}
```

Two points:

* **Payload publication is valid** in the same sense as §2.2: the producer's writes
  are sequenced before the release fence; the last decrementer executes an acquire
  fence and then (in `FID_EXIT`, `original-be236f32.c:13392-13415`) reads the
  continuation frame. On Metal the payload stores are `coherent(device)` plain stores
  (`H` is `DEV u64*`), so the fence covers them (§4.8, §6.16.3).
* **Acquire side is the fragile part.** C++17 /2 permits Y to be the relaxed
  `fetch_sub` RMW: `Y ≺ B` and `Y` reads the previous decrementer's release-sequence
  value, so each producer's release fence synchronizes with the last decrementer's
  acquire fence. However the *current draft* explicitly requires
  "Y is not an atomic modify-write operation", and MSL's own §6.16.1.2 prose only
  states the store-release/load-acquire case, not fence/RMW chains. So this pattern
  is **valid under the C++17 subset MSL cites, but not robust across standard
  revisions and not spelled out in MSL itself**.

  Minimal hardening if this path is ever suspected: after the `fetch_sub`, perform a
  *non-RMW* relaxed load of `c` and then the acquire fence (Y becomes a load that
  reads the RMW's value, satisfying both the C++17 and current-draft /2). The same
  applies to `term_drop`'s refcount pair (`comp.ts:3930-3941`) and to the block
  refcount acquire at `comp.ts:4021-4026`.
* `a32_acq(c)` ignoring `c` on device is fine for fence–fence synchronization, but it
  does not itself load the counter; if a load were needed for the standard rule, it is
  not present.

This path is **not reached** before the observed failure: input 1 takes
`FID_calculate → FID_ir_eval` with `cont = TERM_HOLE` and zero forks
(`trace-be236f32.log`, steps 0-2); no refcount is decremented.

### 2.5 Root result and initial CPU→GPU publication — **valid, but API-level more than MSL-level**

Root result: `task_deliver` writes `H_ROOT_WORD[0..n-1]` then `a32_store_rel`
`H_ROOT_DONE = n+1` (`comp.ts:4265-4273`). Host `root_done()` is an acquire load
(`comp.ts:4304-4306`). Under C++17 /3 the device release fence + relaxed store
synchronizes with the host acquire load; in addition the host only polls after
`waitUntilCompleted` (`comp.ts:4980`), which is a stronger Metal API ordering point.

Initial publication (`corpus_eval`, `comp.ts:5323-5330`):

```c
H[tl] = TERM_HOLE;                       // plain store to the task node
a32_store(a32_at(H, H_CURSOR), 1);       // relaxed store (host)
ring_push(H, 0, t);                      // host: release store of the hi word
cube_run(H, true);                       // turn_take -> turn_set(H_FRONT/H_GROUPS) -> gpu_pass
```

`turn_take` is an atomic exchange (`comp.ts:4242`), `turn_set` writes `H_FRONT` and
the 15 indirect-dispatch group counts as plain stores (`comp.ts:4232-4239`), and
`gpu_run` encodes five dispatches whose group counts are read from `H_GROUPS` via
`dispatchThreadgroupsWithIndirectBuffer` (`comp.ts:4867-4871`, `4959-4965`). The
buffer is `MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked`
(`comp.ts:4933-4934`), and pass-to-pass ordering is enforced by
`memoryBarrierWithScope:MTLBarrierScopeBuffers` (`comp.ts:4964`); cross-command-buffer
ordering by the event wait/signal (`comp.ts:4987,4994`) and `waitUntilCompleted`.

This is **Metal-API synchronization, not an MSL fence**, and it is the only thing that
makes the CPU's plain `turn_set` stores and the host ring push visible to the first
kernel. It demonstrably works here (the kernel sees `H_GROUPS`, reads the root task
and its continuation correctly). A host↔device atomic protocol (`_Atomic u32` on the
host vs `atomic_uint` in MSL on the same buffer) is outside *both* language models;
the command-buffer boundary is what actually orders it. Not a suspect for #2094.

**Direct-return constraint for input 1:** `FID_calculate` on device jumps straight to
`FID_ir_eval` (`original-be236f32.c:12492-12508`); the `IInt` branch computes
`term_pak(CID_ir_VI, o[1])` and does `WL_RETN(1)` (`original-be236f32.c:9219-9224`).
`WL_RETN` is

```c
#define WL_RETN(N) { rn=(N); sp -= LANE_STEP; WL_DYN((u32)STK(0)); }   // comp.ts:3274
#define WL_DYN WL_JMP = { fid = (F); break; }                          // comp.ts:3255-3256
```

so the only way the value reaches the host is: next switch iteration with
`fid == FID_EXIT` (238), `cont == TERM_HOLE`, then the `H_ROOT_WORD` +
`H_ROOT_DONE` publication above. There is no alternate path. If that dispatch does
not happen, no memory ordering can compensate — the store never executes.

### 2.6 Bank publication and device scheduling — **one wrong barrier flag; otherwise valid by construction**

`bank_pack` (`comp.ts:4518-4535`) with **one** threadgroup (`turn_groups` gives
`TURN_ROWS(4)=1` for `at==4`, `comp.ts:4225-4229`; `gpu_run` maps pass 2 to `at=4`,
`comp.ts:4867-4871`) copies device memory:

```c
Term v = H[b->off + b->top + i + lane];
BAR();                                  // threadgroup_barrier(mem_flags::mem_threadgroup)
if (i+lane < n) H[b->off + rd + i + lane] = v;
...
BAR();
if (lane == 0) b->rd = b->wr = b->top = rd + n;
```

`H` is **device** memory, but `BAR()` is `threadgroup_barrier(mem_flags::mem_threadgroup)`
(`comp.ts:3214`). MSL §6.10.1/Table 6.13 (pp.215-216) says the `mem_device` flag is
what makes the GPU order memory operations to **device** memory; `mem_threadgroup`
orders only threadgroup memory. So the first barrier does not formally order the
device loads before the device stores that may overwrite the source region, and the
second does not formally order the stores before lane 0 publishes the new
`rd/wr/top`. **This is a genuine static ordering defect**; use `BARD()`
(`mem_device|mem_threadgroup`) or `threadgroup_barrier(mem_flags::mem_device)`.
It is likely benign in practice because the whole operation is a single threadgroup
on Apple silicon, and it is not implicated in #2094: with the tiny heap the bank
lengths are 0 (`n == 0`), so the two barriers bracket no device traffic.

`bank_push`/`bank_pop` (`comp.ts:3703-3725`) publish `wr`/`rd` with relaxed atomics
and access disjoint ends of the stack (`rd` downward, `wr` upward from `top`), so
within a pass a popper cannot observe a claimed-but-unwritten push slot; the
cross-dispatch visibility of the plain `head` store is provided by
`MTLBarrierScopeBuffers`/command boundaries, not by an MSL fence. Valid by
construction, but note the mixed atomic/plain access to `Bank::rd/wr/top`
(plain in `bank_pack`, atomic in `bank_pop/push`) would be UB if they ever ran
concurrently; they do not in this schedule.

### 2.7 Mixed atomic/plain and mixed-width accesses — **outside the model**

* **Redirect cell (strongest instance).** `rfc_bump` does a 32-bit relaxed atomic
  add to the low word of a 64-bit cell (`comp.ts:3866-3871`), while `term_peek` does
  a **plain 64-bit load** of the same cell on device (`w64_load(p) = *(p)`,
  `comp.ts:3526`, `3884-3893`). The comment acknowledges the tear and argues both
  halves hold the same `loc` (count lives in bits 0-23, `loc` is shifted by 24;
  `RFC_CNT`, `LOC_MASK` at `comp.ts:3346-3348`). The layout argument is correct, but
  this is a non-atomic read overlapping an atomic write, which is a data race / UB
  under any C++17-based model and is not blessed by MSL §2.6 ("atomic functions must
  operate on Metal atomic data"). It is *not* what fails here (single active lane),
  but it is a formal UB source in the same compilation unit.
* **Ring index plain reads.** `monk_step` reads `*get` with a plain load
  (`comp.ts:4450,4453,4456`) and writes it with an atomic store (`comp.ts:4459`);
  the kernel loop reads `*ring_get(...)` plainly (`comp.ts:4577`). On the device this
  is one thread per ring, so no cross-thread plain/atomic conflict on `get`; it is
  still mixed access to the same object and relies on the compiler not exploiting it.
* **Host pool path.** `row_grow` reads `*ring_put`/`*ring_get` non-atomically
  (`comp.ts:4669-4671`) while producer threads use atomic RMWs (`comp.ts:4208`).
  That is a C11 data race on the CPU path (which passes 3/3 only because the
  workload never rotates the rings); it is not a Metal issue but it is the same
  source compiled for the host, and it belongs in the UB audit.
* **Strict aliasing.** `H` is `uint64_t*` but is routinely accessed as `uint32_t*`,
  `Bank*`, or `atomic_uint*` (`a32_at`, `turn_set`, `bank_at`, `A32`). This is a
  language-level aliasing violation on both host and device compilers. It is not
  local to the failing `WL_RETN` path (which only touches type-consistent `u64`
  stack words), but it is the most plausible source-level UB that could perturb the
  giant function's optimization.
* **64-bit atomics are available** (`atomic_ulong`, MSL §2.6 p.39, Metal 2.4+), so
  the hi/lo split is a choice, not a necessity; a single `atomic_ulong` relaxed
  store + fence would remove the 32-bit tear window entirely.

### 2.8 SIMD divergence / convergence

* No SIMD-group functions (`simd_shuffle`, `simd_*_barrier`, `quad_*`) are used, so
  divergence does not interact with SIMD-group collectives here.
* `threadgroup_barrier` requires *all* threads of the threadgroup to reach it, and
  for each loop iteration if any thread reaches it (`MSL §6.10.1 pp.215-216`).
  In `bank_pack`, `n` and the loop trip count are read from the same device `Bank`
  fields by all 128 lanes, and pass 2 is dispatched with exactly one threadgroup, so
  the barriers are uniform. In the grow loop, `pass` is a uniform kernel argument and
  every lane reaches `BAR()`/`BARD()` each iteration after `monk_step` (which
  contains no barrier). The early return before the barriers is CUDA-only
  (`dev_front`), not in the Metal kernel (the Metal path dispatches group counts
  indirectly). **Safe as written.**
* If pass 2 were ever dispatched with more than one threadgroup, `bank_pack` would
  become a multi-writer of the same bank cells *and* the loop trip counts would still
  be uniform, but the concurrent identical writes would be a data race; the current
  `turn_groups` prevents that for `at==4`.

---

## 3. Why none of the above is #2094's mechanism

The parent's trace and boundary probe (artifacts quoted verbatim) show the failure
with **one active lane** and **no inter-thread handoff on the failing path**:

`trace-be236f32.log` (input 1):

```
TRACE 0 fid=239 sp=0 rn=0 r0=0500c60003108d08 seq=1 cont=0000000000000000
TRACE 1 fid=198 sp=3 rn=0 r0=0000000000000001 seq=3 cont=ffffffffffffffff
TRACE 2 fid=134 sp=3 rn=0 r0=0200120003108aa3 seq=3 cont=ffffffffffffffff
bend: frontier drained without a result
```

* Only lane 0 has debug records (the dumper loops all `LANES`).
* `fid=239` is `FID_ENTER` (`original-be236f32.c:625`), `198` is `FID_calculate`
  (`:585`), `134` is `FID_ir_eval` (`:521`); the root task and continuation are
  correct.
* No `FID_EXIT` (238) dispatch follows.

`boundary-be236f32.log` (input 1) adds the decisive stage 105 probe immediately before
`WL_RETN` in the `IInt` branch (`boundary-be236f32.c` diff, `original-be236f32.c`
lines 9219-9224):

```
BOUNDARY 1=0000000000000069 2=... 3=...0001 6=0000000000000001 10=3 11=00000000000000ee
TRACE 3 fid=239 sp=0 rn=0 r0=3d8fcfd53d79fad0 seq=0 cont=0000000000000000
TRACE 4 fid=36815 sp=3 ...
bend: a function the device does not hold
```

Reading it: marker 105 was written (`0x69`); the result word `r0` is correct (`1`,
i.e. `VI{1}`); `sp` at `WL_RETN` is 3; and `STK(-1)` is `0xee` = `FID_EXIT`.
`WL_RETN(1)` is `{rn=1; sp-=CUBE; fid=(u32)STK(0); break;}` (`comp.ts:3274`,
`3255-3256`), so after the decrement `STK(0)` is exactly that `0xee`. Yet the next
outer dispatch is not 238: `work_loop` returns an **undefined nonzero value** to
`monk_step` (`0x3c003c003c003c00`, `0x3d8fcfd53d79fad0`, `0x000000003f800000`,
`0x000000003c0030b5` across runs), which is then re-entered as a task, yielding
`FID_ENTER` with a garbage `r0` and `ERR_FIDS` (or, on the no-return variant, a
drained frontier). The same failure occurs for all three inputs, and the
`} break;` → `} continue;` variant still fails 45/45 (`continue-be236f32.log`), so it
is not explained by the outer switch break.

Memory-ordering arguments that fail to account for this:

* There is **no cross-thread edge** between stage 105 and the failure: the value at
  `sp[-1]` was written by this thread's earlier `WL_FRAME` (`comp.ts:3279-3284`) and
  read back by this thread; same-address own-thread store→load needs no fence.
* The fiber/stack region is device memory, but the load that is mispredicted is
  local to the thread; hardware and MSL guarantee it sees its own earlier store
  (coherence is at least thread-scoped, §4.8).
* No ring slot is reused; no refcount fork/`task_deliver` runs; no bank traffic with
  `n > 0`. The first `ERR_FIDS` is not preceded by any of the orderings in §2.
* The return value is not a torn ring `Term`; it never came from `ring_push` (only
  `monk_step` re-enters `work_loop` with `t = r` after a nonzero return, and the
  trace's new invocation is at `sp=0` with garbage). It is the undefined return
  register of a function that left through a path that never executed `FID_EXIT`.
  The AIR edge is **source-correct** (`cleanup232=55 → cleanup233=4 → switch epilog →
  cleanup336=0 → outer for`), and the pre-`WL_RETN` marker 105 is present while the
  after-inner-loop marker 106 is absent. The root cause therefore lies **downstream
  of AIR generation**, in the Metal backend/AGX lowering or its execution; the exact
  lowering step is not identified. *(Parent/AIR evidence; not independently re-dumped
  by me.)*

Confirmation that this is not ring-specific: the reported null edit (moving `wpoll`
into the loop header) flips the failure, and `CUBE_LOG=0` still fails; both are
codegen/layout perturbations, not changes that would fix or trigger the ring
release/acquire protocol. The task-ownership probe additionally verified that input 1
has **no inter-lane handoff and zero allocator activity after entry**. Also note the
first publication (host release store → device acquire fence) **worked**: the root
`Term` and its continuation arrived intact, which is a live counter-example to "the
fence protocol is broken on this machine".

Therefore: **strongest bounded conclusion = the #2094 failure is generated downstream
of source-correct AIR in the Metal backend/AGX lowering; it is not a Metal
memory-model violation and not a missing acquire/release edge.** Distinguish this from
source UB: the `WL_RETN` path itself is UB-free (type-consistent `u64` stack accesses,
in-bounds stack pointer, no aliasing, no race); the UB candidates found in §2.7 live
elsewhere in the same function and are the reason a "language UB vs compiler bug"
question remains open at the whole-function level.

---

## 4. UB boundary audit (for the source-vs-AIR question)

Present in the failing translation unit, but **not on the `WL_RETN` path**:

1. Strict-aliasing/effective-type violations from `u64*` → `u32*`, `Bank*`,
   `atomic_uint*` casts (`a32_at` `comp.ts:3539`, `turn_set` `:4238`, `bank_at`
   `:3690`, `A32` `:3478-3484`). This is the most likely license for aggressive
   optimization in this giant function; it does not by itself explain the specific
   `break`-to-`FID_EXIT` cleanup.
2. Non-atomic 64-bit `term_peek` overlapping 32-bit atomic `rfc_bump`
   (`comp.ts:3866-3871,3889-3893`).
3. Host pool plain ring counter reads vs atomic RMWs (`comp.ts:4669-4671` vs `:4208`).
4. Mixed atomic/plain `Bank::rd/wr/top` accesses (`comp.ts:4521-4532` vs `:3708-3724`).
5. Device `work_loop` has no trailing `return`; if the compiler creates a path to the
   function end, executing it is UB (falling off a non-void function). The observed
   garbage return is consistent with such a path being reached, but the AIR edge is
   source-correct, so this is a *downstream* lowering defect, not frontend license
   from source UB.

Path-local check for the failing `WL_RETN` (`original-be236f32.c:9219-9224`): all
writes/reads are `u64` (`Term`/`FID_*` stored as `u64` in stack slots), `sp` stays
within the lane's `STAK` region (`STAK_LEN=2048`, `sp` indices 2-3), `fid`/`rn` are
locals, and no pointer escapes. There is no local UB that would let the frontend
legitimately drop the `FID_EXIT` dispatch. Since the AIR cleanup chain is
source-correct and marker 106 is absent while marker 105 is present, the discrepancy
is introduced below AIR (Metal backend/AGX lowering or its execution). The exact
lowering step is not identified in this analysis.

---

## 5. Targeted discriminating experiments (minimal, in priority order)

No builds/tests were run here; these are the smallest probes that distinguish the
remaining hypotheses.

**P0 — load the exact AIR, bypassing the source frontend (in progress by parent).**
A native `air-loader` C program that calls `newLibraryWithData:` on the **unmodified**
AIR/metallib extracted from a fresh `boundary.gpu`, then runs the same three inputs
plain and under validation. If the failure persists from byte-identical AIR, the
source frontend and archive staleness are ruled out and the defect is in the
AIR→AGX lowering/execution; if it disappears, the frontend's emitted AIR is suspect
despite the source-correct cleanup edge. This is the decisive discriminator; do not
edit the AIR for this probe.

**P1 — isolate the control-flow lowering (only if a sentinel probe becomes possible).**
Extract the device `work_loop` skeleton into a standalone MSL kernel with the same
nesting (`for(;;)` outer, `switch(fid)`, segment-local `for(;;)` with `break`, the
`WL_RETN` expansion), but replace all payload work with constants and no atomics.
Have `FID_EXIT` write a sentinel. If the kernel can return without executing the
sentinel / returns garbage on the M1 Max while AIR still shows the correct PHI, the
compiler/cleanup lowering is proven independent of all memory-ordering and payload
UB. A re-encoded LLVM 21 bitcode attempt crashed the backend; that crash is **not**
evidence about the original bug and is not counted. Any future variant must leave the
original AIR unmodified.

**P2 — ring index ordering, in isolation (validates §2.3).**
One CPU producer + one CPU consumer MPSC ring with `RING_LEN=2`, 64-bit payload split
lo/hi exactly as `ring_push`/`monk_step`, and only the two lines under test toggled.
Expected under the failing pair (`comp.ts:4209,4459`): mixed lo/hi Terms observed
within 10^6 iterations. Expected under `335f239a` (`passing 4213,4462`): none. Run on
both CPU and a Metal two-lane variant; the Metal variant needs only lane 0 produce and
lane 1 consume.

**P3 — refcount fence/RMW pairing (validates §2.4).**
Two (or N) lanes: each writes a distinct payload word to `coherent(device)` memory,
`atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst)`, then relaxed
`atomic_fetch_sub` on a shared counter; the lane that observes old value 1 checks all
payload words after a fence. Run 10^6 times; compare against the variant with an extra
non-RMW `atomic_load(c, RLX)` before the acquire fence. A failure only in the first
variant pins the current-draft "Y not RMW" hazard.

**P4 — `term_peek` tear.**
One lane repeatedly `term_peek`s a redirect cell while another lane hammers
`rfc_bump`; check `loc` stability under `-O` builds. This tests the UB in §2.7 rather
than the model; if it fails, replace the 64-bit plain read with a 64-bit relaxed
atomic (`atomic_ulong`, available on Metal 2.4+).

**P5 — bank barrier scope.**
Force a bank to `n > 0` and overlapping copy regions, run `bank_pack` with `BAR()`
vs `BARD()`, and compare the packed cells. A difference proves the `mem_threadgroup`
flag is insufficient for device memory in practice (and justifies the `BARD()` fix);
no difference leaves it a model-only defect.

**P6 — ring-slot width.**
Replace the split lo/hi with a single `atomic_ulong` relaxed store after a device
fence and `atomic_ulong` relaxed load before a fence. If #2094 is untouched (expected),
it confirms the 32-bit split tear is not the mechanism; if a previously flaky
cross-lane workload changes, it validates the split as the hazard.

**P7 — attribution on the checkouts.**
(a) Apply only `335f239a`'s two lines to `be236f32`; (b) revert only those two lines in
`d18110b6`. Given the codegen sensitivity, interpret together with P1/P2: if (b) still
passes 45/45, the passing checkout's success is not due to the ring fix, supporting
the compiler-lowering explanation. Do not treat a flip as proof of a race, because
both edits shift inlining/ISel in a 14k-line Metal function.

---

## 6. Direct answers

**Are task contents/continuation publication and refcount/acquire sequences valid
under Metal's actual memory model?**

* Ring **slot contents**: yes for the publication itself (release fence + relaxed hi
  store, acquire fence after hi load), given `coherent(device)` and `mem_device`
  fences at device scope.
* Ring **consumption index**: no in `be236f32` — the release fence is on the wrong
  side of the second slot load and the producer has no acquire; that is a real
  ordering bug, fixed by `335f239a`. It is not the reproducer's earliest failure.
* Continuation **payload**: yes by intent and by C++17 `[atomics.fences]/2`, but the
  acquire rides on a relaxed RMW, which the current draft excludes and MSL does not
  spell out; add a non-RMW relaxed load before the acquire fence to make it robust.
* Refcount **atomicity** is sound (`relaxed` is sufficient and is all MSL allows,
  §6.16.1); only the payload-ordering rule is fragile.
* Root result: valid (release fence + relaxed store vs host acquire load, plus
  command-buffer completion).
* Initial CPU→GPU publication: valid in practice, but only Metal-API ordering
  (shared buffer + commit + event/wait + `MTLBarrierScopeBuffers`), not an MSL fence.
* Bank: publication-by-construction is sound, but `bank_pack` uses a threadgroup-only
  barrier for device-memory copies — wrong `mem_flags`, use `BARD()`.
* Mixed 32/64 atomic/plain (`term_peek`/`rfc_bump`, host ring counters) and the
  `u64*`↔`u32*` aliasing are outside the model; formal UB, not the observed bug.

**What is the strongest mechanism for #2094?**
The defect is generated downstream of source-correct AIR, in the Metal backend/AGX
lowering (or its execution): at the device `WL_RETN` the correct `FID_EXIT` stack word
and result are present, the dispatch is dropped, the function returns an undefined
value, and the caller resumes with it. Single-lane execution, the absence of any ring
reuse/refcount/bank handoff before the failure, and the task-ownership probe's
"no inter-lane handoff / zero allocator activity after entry" rule memory ordering
out; the source-correct AIR cleanup edge rules out a frontend miscompile. The exact
AGX lowering step remains unknown. Nondeterminism and null-edit sensitivity are
codegen evidence, not race evidence.

**Do not claim:** that #2094 is caused by the ring `get` race, by a Metal fence bug,
or by a missing acquire on continuation publication. The first two are real/latent
issues found in the source and are worth fixing, but the artifacts place the first
divergence before any of them can execute.
