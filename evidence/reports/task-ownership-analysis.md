# Task record and allocator ownership/lifetime analysis for bend #2094

Failing checkout: `/Users/ilyagulya/Projects/My/bond-lang-issue-2094-before` at
`be236f324ed0338fe907fe50474468e4d3b94157`.
Emitted failing program: `original-be236f32.c` in this directory.
All source citations are `bend2/comp.ts` in that checkout unless noted.
Static analysis only; I ran no build, test, or benchmark. Runtime observations
quoted below come from the parent agent's trace artifacts and are marked so.

## Verdict

For this failing program, and for input 1 in particular:

1. **No block is handed to two GPU lanes.** All GPU heap activity for this run
   happens under one lane identity (`me = 0`), and every transfer across the
   CPU/GPU boundary is a single-owner move.
2. **No task record is freed or reused before its continuation consumes it.**
   This program contains no `rem > 1` task record at all, so the multi-child
   (fork/join) paths are never exercised.
3. **Input 1 performs no heap allocation and no free between entering `ir_eval`
   and the failing `WL_RETN`.** The only allocator call on that path is the
   `FID_ENTER` free of the bang task record *before* `ir_eval` runs. Therefore
   the allocator cannot be the earliest cause of the input 1 failure.
4. The observed failure is a **control-flow deviation after `WL_RETN(1)`**:
   source and the extracted AIR agree; the hardware does not. See section 5.

## 1. Task record lifecycle

Layout and lifetime primitives (`comp.ts`):

- `task_node` allocates `cls_fit(ar + 2)` words and writes `ar` argument slots,
  then `cont`, then `(idx << 32) | rem` (4249-4258).
- `task_tail(t) = loc + fid_arity(fid)` is the `cont` word (4260-4261).
- `task_deliver` writes `n` result words at `term_loc(cont) + idx`, then
  decrements the low 32 bits of the `(idx, rem)` word with a release RMW; only
  the delivery that reads 1 as the old value performs an acquire and returns
  `cont` (4264-4281).
- `task_deal` is called only when `(u32)H[task_tail(r) + 1] != 0`, i.e. while
  `rem > 0`; it replaces each `TAG_TSK` argument slot with `TERM_HOLE` before
  `ring_push`ing the child (4283-4298, 4466-4482).
- A task record is freed only by `FID_ENTER`, after loading its arguments and
  frame and only on the path where its `rem` word has reached zero:
  `heap_free(e, cls_fit(war + 2), a)` (4372-4379, reached via 4466-4478).

Invariants that follow, with the exact reasons:

- **No free while `rem > 0`.** `FID_ENTER` is the only free site for a live task
  record, and `monk_step` reaches it only after `(u32)H[task_tail(r)+1] == 0`
  and after `task_deliver` has returned that record as the last delivery
  (4464-4479). `task_deal`'s slot overwrite happens before the child is pushed,
  so the parent does not retain a stale reference (4290-4296).
- **No continuation overwrite while `rem > 0`.** `task_deliver` writes result
  words only in `[idx, idx+n)`; `idx` is written once at creation (4253-4255)
  and read by `WL_FRAME` as `(H[tail+1] >> 32) & 0xFFFF` (3280-3284). The
  emitter guarantees `idx + n <= arity`: `idx` is the number of leading held
  parameters and `n` is the callee's result width (fork creation 2437-2450,
  continuation creation 2465-2478). The reads happen before the release
  decrement and the last reader acquires, so the eventual `FID_ENTER` sees all
  results.
- **Single owner at handoff.** The only device path that moves a completed task
  record to another lane is `monk_step` 4471-4478 (`ring_push` then
  `return 2`); the source of the record has already returned from `work_loop`
  and does not touch it again.

## 2. What this program actually creates

Counting the emitted `task_node` calls in `original-be236f32.c`:

- 352 calls total: 206 with `rem = 0`, 145 with `rem = 1`, plus the root
  `task_node(e, MAIN_FID, TERM_HOLE, 0, 0)` at the host entry.
- **Zero** calls with `rem > 1`.
- **Zero** stores of the form `e.mem[join + idx] = term_tsk(...)`; the
  fork/join emitter pattern (2437-2450) is not present in this program.

Consequences:

- There is no join record with multiple children, so `task_deal` can never find
  a `TAG_TSK` slot in this program. Its cross-ring `ring_push` (4289-4297) is
  dead.
- The only remaining cross-lane move is `monk_step` 4471-4478, which fires only
  for a completed `rem == 0` record that is `nofk` and under `stride != 0`.
  Input 1 never produces such a record (section 3); inputs 0 and 2 do.

## 3. Why input 1 never crosses lanes and never touches the allocator

Input 1 selects IR node `IInt{1}`. `FID_calculate` has a device guard:

- `original-be236f32.c:12492-12506`: `if (!DEVICE && !seq && ... )` creates a
  task record; on the device this is false, so the code falls through to
  `WL_JMP(FID_ir_eval)` (12504-12507). `calculate` is inlined into `ir_eval`;
  no task record is created between them.
- `original-be236f32.c:9219-9224`: the `_o_0[0] == 0` (`IInt`) branch calls
  `term_sink` on three values that are all static image terms
  (`STAT_OFF + 2467`, `STAT_OFF + 2469`, `term_pak(CID_Nil, 0)`) and then
  `WL_RETN(1)`. There is no `heap_alloc`, no `heap_free`, no `task_node`, and
  no `WL_AGAIN`.
  - `term_sink` is a no-op for `term_triv` values (`term_loc < HEAP_OFF`),
    which these static constructor cells are.
  - `spin_ir_nthNode` (called at 9216) only does `ctr_take`/`spare_free` on the
    same static list; `spare_free` ignores locations below `HEAP_OFF`
    (`comp.ts` 3808-3812).
- The only allocator action anywhere on the input 1 path is the `FID_ENTER`
  free of the CPU-allocated bang task record, and that happens before
  `ir_eval` (4372-4379). After that, the path allocates nothing that could be
  freed or reused.

GPU topology for this run is single-lane for ring 0:

- `gpu_run` dispatches passes `0,0,1,3,2` (4853-4858).
- The first pass-0 dispatch uses `turn_groups(f, 0) = 1` threadgroup
  (`TURN_ROWS(at) = at % 4 == 0 ? 1 : CUBE_G`, 4224-4228), so
  `stride = CUBE_G` and `me = row * CUBE_T + stride * lane` gives `me = 0` for
  `row = 0, lane = 0` (4564-4566).
- Ring `r = 0` is handled by that lane in pass 0; in passes 1 and 3,
  `ring_flip(0) = 0` and `me = 0` again. All heap/free-list operations for the
  single root task therefore use `Env.alc = H + ALC_OFF + 0`, one lane identity.

The only ownership crossings on the input 1 path:

1. CPU -> GPU for the bang task: `corpus_eval` saves `cont`/`idx`, writes
   `H[task_tail(t)] = TERM_HOLE`, pushes the record to ring 0 and runs the GPU
   (5312-5332). The GPU enters the record with `cont = TERM_HOLE`, runs the
   call, and frees the record in `FID_ENTER`. The CPU free list is the static
   `ALC` array (3403); the GPU free list is `H + ALC_OFF + me` (4565), so the
   same block is never on both lists.
2. GPU -> CPU result: `task_deliver` with `cont == TERM_HOLE` stores the result
   in `H_ROOT_WORD`/`H_ROOT_DONE` (4264-4273); the CPU then performs the real
   delivery with the saved `cont`/`idx` (5330-5333).

## 4. Allocator invariants: proof sketches

- **Single owner per block.** `heap_alloc` pops a lane chain head or takes a
  bank/bump generation (3744-3788); `heap_free` pushes a block back (3791-3805);
  `dev_cut` detaches exactly `KEEP(c)` blocks and pushes the head to the bank
  (4498-4516); `bank_pop` removes an entry by decrementing `rd`; `bank_push`
  appends at `wr` (3692-3716). No operation copies a live block between two
  free lists.
- **Generation arithmetic.** `KEEP(c) = KEEP_WORDS >> c` (floor 1) (3730);
  device `dev_cut` uses `gen = KEEP(c) << c` (4503); a bank hit sets
  `n = KEEP(cls)` and `ALC_LEN = (n-1) << cls` (3762, 3784-3786). A fresh
  device allocation uses `n = QUANTUM >> cls = 128 >> cls` (3370, 3762), whose
  `(n-1) << cls` is `128 - 2^cls < gen`, so a fresh chain is never handed off
  early. `KEEP_WORDS` is a single global chosen per run (5297), so host and
  device generations agree when the GPU is enabled.
- **Bank capacity is sufficient.** `corpus_lay` (5263-5278) gives class `c` a
  region of `2 * (cap >> ((c < NCLS ? NCLS : c) - PAGE_BITS))` u64 entries,
  `cap = (span - HEAP_OFF) / (PAGE_LEN + 10)` (5265). The maximum number of
  generation entries a class can ever need is
  `heap_words / (KEEP(c) << c) = cap/2` for `c <= 7` and
  `cap / 2^(c-7)` for `c >= 8` (heap words = `cap * 128`). The region is exactly
  twice that, and the `+ 10` divisor reserves ~`10 * cap` words for the banks
  (9 classes at `cap` plus the geometric tail). For the recorded 512 MB span,
  `cap ~ 113,716`: classes 0-8 can need at most 56,858 entries against 113,716
  slots, class 9 at most 28,429 against 56,858, and so on. Classes whose region
  is smaller than one entry (roughly `c >= 24`) cannot have a block at all in
  that span (`2^c` exceeds the heap words), and the boundary class 23 has a
  two-entry region for at most one block.
- **Bank compaction preserves the stack.** `bank_pack` reads chunk `i` from
  `[top, wr)` and writes it to `rd + i` after a barrier (4518-4533). Since
  `rd <= top`, the destination chunk is always below the source chunk and below
  every future source chunk, so there is no overwrite-before-read.
- **The task record invariants in section 1** hold for this program; in
  particular, the `rem` word is only changed by `task_deliver`, and `cont` is
  only changed at creation (and deliberately to `TERM_HOLE` by the CPU bang
  handoff before the GPU sees it, 5322-5325).

I found **no reachable sequence** in this program in which a block is given to
two lanes, freed/reused before the continuation consumes it, or has its `cont`
overwritten while `rem > 0`. The closest theoretical failure would be a
violation of the emitter's `idx + n <= arity` rule, which would make
`task_deliver` write into `cont`/`(idx, rem)`; it does not occur here because
there are no multi-child records, and the construction in 2437-2478 maintains
it.

## 5. Where the failure actually is (cross-check against parent artifacts)

Parent runtime trace (`boundary-be236f32.log`, probe source
`boundary-be236f32.c`):

- At `ir_eval` entry the frame continuation is `~0` (correct).
- Immediately before `WL_RETN`, stage 105 records `sp = 3` and
  `STK(-1) = 0xee` (`FID_EXIT`).
- No dispatch with `fid = 238` is recorded. The next trace entry is a fresh
  `work_loop` invocation (`sp = 0`) with `fid = FID_ENTER` and a garbage `r0`,
  followed by the invalid-FID default (`ERR_FIDS`).

### 5.1 Archive identity ties this AIR to the executed code

Parent-reported build facts, and the code paths that support them:

- The `boundary-be236f32.c` used for this AIR was a fresh copy and no unique
  `boundary-be236f32.bin`/`.gpu` existed before the build. The build command
  compiled the C (`--gpu-build`), then ran the same binary for all three
  inputs, with no source/binary/archive edits in between.
- `BEND_SRC` is `#embed __FILE__` at C compile time (`comp.ts:3407-3414`), so
  the executable fixes one source snapshot. `gpu_desc`/`gpu_load` create the
  4-cell pipeline from that same embedded snapshot (`comp.ts:4910-4958`);
  `gpu_pipe` requests `MTLPipelineOptionFailOnBinaryArchiveMiss` and falls
  back to `gpu_pipe(d, nil)`, which compiles the same embedded snapshot
  (`comp.ts:4927-4958`).
- Therefore the `.gpu` archive's AIR and the runtime pipeline are the same
  source snapshot: the archive is either loaded directly or the snapshot is
  recompiled in-process. The `llvm-dis` 21 experiment that preserved
  `air64_v28-apple-macosx26.6.0` (`boundary-preserved.ll`) is a writer-compat
  test only; its AGX pipeline crash after re-encoding is not evidence about the
  original binary.

### 5.2 Exact AIR path proof (machine-checked against `boundary-air.ll`)

The only path from the input-1 branch to the dispatch loop is:

| step | block | line | terminator |
|---|---|---|---|
| 1 | `if.end9379` | 23042 | `switch i32 %conv9381` case `0` -> `if.then9383` |
| 2 | `if.then9383` .. VI branch | 23075-23133 | sinks; `r0 = term_pak(CID_ir_VI, o[1])`; `fid = trunc(STK(-1))`; `br label %cleanup10681` |
| 3 | `cleanup10681` | 24733 | `switch i32 %cleanup.dest.slot.232`: literal edge `i32 55, label %cleanup10685.loopexit36280` (24749) |
| 4 | `cleanup10685.loopexit36280` | 24753 | `br label %cleanup10685`; literal phi entry `[ 4, %cleanup10685.loopexit36280 ]` for `%cleanup.dest.slot.233` (24780) |
| 5 | `cleanup10685` | 24778 | `%cond14985 = icmp eq i32 %cleanup.dest.slot.233, 4` (24789); `br i1 %cond14985, label %sw.epilog14914, label %cleanup14915` |
| 6 | `sw.epilog14914` | 33726 | `br label %cleanup14915`; `%fid.361` phi has literal entry `[ %fid.254, %cleanup10685 ]` (33730) |
| 7 | `cleanup14915` | 33750 | `%cond14935 = icmp eq i32 %cleanup.dest.slot.336, 0` (33770); phi entry `[ 0, %sw.epilog14914 ]` (33753); `br i1 %cond14935, label %for.cond, label %cleanup14918` |

`cleanup10681`, `cleanup10685`, `sw.epilog14914` and `cleanup14915` contain no
`ret` and no `unreachable`. Every phi value used on the edge is defined:
`%cleanup.dest.slot.232 = 55`, `%cleanup.dest.slot.233 = 4`,
`%cleanup.dest.slot.336 = 0`, `%fid.361 = %fid.254 = trunc(STK(-1))`.
The garbage value is exactly what an incorrectly selected cleanup returns:
`for.cond` carries `%retval.0 = phi i64 [ undef, %entry ]` (1711), and
`cleanup14918` returns it via `ret %retval.352` (33773-33775). Reaching
`cleanup14918` requires `cleanup.dest.slot.336 != 0`, which this path cannot
produce.

### 5.3 Source UB audit: no UB on this path

- `WL_SPIN` is `for (;;) { if (err_spun(...)) { return 0; }` and `WL_SPUN` is
  `} break;` (`original-be236f32.c:127-129`). `WL_RETN` is
  `{ rn = N; sp -= LANE_STEP; WL_DYN((u32)STK(0)); }` (137), and on the device
  `WL_DYN(F)` is `{ fid = F; break; }`. The VI branch (9219-9224) lies between
  `WL_SPIN` (9213) and `WL_SPUN` (9957), so its `break` exits the inner
  `for(;;)`, then `WL_SPUN`'s `break` exits the dispatch `switch`. This is
  ordinary, well-defined C/C++ (Metal C++) control flow.
- The outer dispatch loop is `for(;;)` (`comp.ts:4349-4351`) and no path
  `break`s it; the function cannot fall off its end in the abstract machine.
  (Falling off a non-void function is UB only if reached, and the AIR has no
  edge to `ret` on this path.)
- Pointer/index uses on the path stay in their objects: `sp` slots 0-2 are the
  current lane frame inside the 2048-slot stack region (`STAK_LEN`,
  `comp.ts:3362`), `STK(-1)` after `WL_RETN`'s `sp -= LANE_STEP` is the frame
  slot just written by `WL_FRAME` (not below the lane stack base), the task
  record words `[a, a+ar+1]` come from `heap_alloc` with `HEAP_OFF + p*128`
  and the cap check at `comp.ts:3765-3769`, and the list/env terms are static
  image cells below `HEAP_OFF`.
- No uninitialized read affects this path: `_o_0[1]` is written by
  `spin_ir_nthNode` (the `n == 0` store sequence, `original-be236f32.c:3263-3270`
  region), and the frame fields were written by `FID_ENTER`/`WL_FRAME`
  (`comp.ts:3279-3284`).
- Integer operations are defined for the used values: `(idx << 32) | rem`
  with `idx < 2^16` and `rem <= 1`, truncations to `u32`, `cls_fit(ar+2)` with
  small arities, and `(u32)` tag extraction. No signed overflow, shift-count,
  division-by-zero, or null dereference exists on the path.
- The `undef` carried in `%retval.0` is not used for any control decision on
  this path; it is only the function's return register if the wrong cleanup is
  selected. (Even in IR, returning `undef` is not UB, but it is observable as
  the garbage `r0` in the trace.)

The source and this AIR are both defined and correct on the path. The observed
behavior (the after-loop marker never runs and `work_loop` returns garbage)
contradicts both, so the deviation is downstream of AIR. The construct the
backend must lower is clang's cleanup-destination dispatch: a 239-case
`switch (fid)` containing, for each spin segment, a second
`switch (cleanup.dest.slot.N)` whose 55/56 values mean "left the inner loop" /
"stay in the inner loop". The two observed wrong outcomes (no `fid = 238`
dispatch; a fresh `work_loop` invocation with garbage `r0`) are exactly what a
mis-lowered/mis-executed cleanup selector or its edge to `sw.epilog14914`
produces. To isolate the backend step without MetalToolchain, the minimal
reproducer is a tiny MSL kernel with the same shape: an outer `for(;;)` over a
large `switch(fid)`, a per-case inner `for(;;)` with `break`, and a
`switch(cleanup.slot)` that routes "after inner loop" to the outer dispatch;
check whether a marker after the inner loop executes. If MetalToolchain becomes
available, recompiling `boundary-air.ll` at a lower optimization level and
re-running shell checkpoints would separate optimization-pass from
codegen/AGX. This is not a task-record ownership bug.

## 6. Smallest probes that could falsify the above

Each probe is an assertion/instrumentation, not a semantic change; they do not
perturb allocation order.

1. **Static inventory (already run):** assert every emitted `task_node` has
   `rem in {0, 1}` and that no `e.mem[...] = term_tsk(...)` child store exists.
   Falsifies "there is a multi-child join in this program".
2. **`FID_ENTER` assertion:** before `heap_free`, if
   `(u32)e.mem[task_tail(t) + 1] != 0`, `err_post(ERR_TAGS)`. One load and one
   branch; falsifies "a task record is entered/freed while `rem > 0`".
3. **Cross-lane counter:** in `monk_step`'s `ring_push` branch (4471-4478),
   record `(r, target_ring)` in a spare header word or a bounded side buffer.
   Expect zero for input 1 and nonzero for inputs 0/2. Falsifies the
   no-cross-lane claim.
4. **Bank capacity assertion:** in `bank_push`, compare `wr` against the
   class's entry capacity (computable from `H_CAP` and the 5263-5278 formula)
   and `err_post` on exceed. Expected never to fire; falsifies the capacity
   proof.
5. **Archive/AIR identity (already established by the fresh build):** the
   `BEND_SRC` snapshot is fixed at C compile time, the unique `.gpu` was made
   from that snapshot by `--gpu-build`, and the runs followed with no edits.
   The remaining discriminator is therefore not staleness but the Metal
   backend stage. Two probes: (a) a minimal MSL kernel with the same nested
   `for(;;)`/`switch(fid)` plus cleanup-slot shape, with a marker after the
   inner loop; (b) if `MetalToolchain` becomes available, recompile the same
   AIR at a lower optimization level (or with the archive/pipeline options
   varied) and check whether the marker executes. Either isolates
   optimization-pass vs codegen/AGX without touching Bend's allocation or task
   records.
