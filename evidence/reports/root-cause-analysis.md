# #2094: verified failure boundary

## Environment and scope

M1 Max, 32 GPU cores, 64 GiB RAM; macOS 26.6.2 (25G83); Apple clang 21.0.0; Bun 1.4.4. Runtime shader compiler identifies itself as Apple metal 32023.886 (metalfe-32023.886.1). AIR metadata is Metal 4.0, AIR 2.8, target `air64_v28-apple-macosx26.6.0`.

Compiler baseline: `be236f324ed0338fe907fe50474468e4d3b94157`. Reproducer: `ee1d7a858f54cdd7f182391c917ee41ca9f1c6d4`. No compiler fixes applied. Original source checkouts remain unchanged; diagnostic artifacts are separate generated C files. Initial runs used `/Users/ilyagulya/Projects/My/bond-lang/bond/guard.ts` with unchanged limits. The resumed offline phase uses the diagnostic guard described below; its sole approved timing relaxation is the process-list query timeout. Bun 1.4.4 consumes the first `--`, so invocation uses `bun guard.ts -- -- COMMAND ...`.

## Observed baseline

`results-be236f32.jsonl`: CPU 3/3 correct; Metal 0/45 correct (24 device-does-not-hold, 21 frontier-drained). `results-d18110b6.jsonl`: CPU 3/3 and Metal 45/45 correct, actual backend=1/events=3. Both used threads=4, Metal span=512MB, no validation.

## First divergence, not merely an error message

`boundary-be236f32.log`, input 1:

- Only lane 0 is active. Dispatch trace: FID_ENTER=239 -> calculate=198 -> ir_eval=134.
- Initial continuation is `TERM_HOLE`, not corrupted.
- `spin_ir_nthNode` returns success=1 and initialized output tag=0 / integer=1.
- All three `term_sink` calls complete. Their arguments refer to the static image; this path allocates nothing and does not transfer tasks between lanes.
- Immediately before `WL_RETN(1)`: stack depth=3 and return slot=`0xEE`=FID_EXIT=238.
- Expected next dispatch is FID_EXIT. Instead `work_loop` returns an arbitrary 64-bit word to `monk_step`. That word is then mistaken for a task; a fresh FID_ENTER receives the garbage word. This explains variable downstream error messages without requiring a race.

`loop-exit-be236f32.log` inserts a marker immediately after the inner ir_eval loop, before the switch break. The pre-return marker is reached on input 1, but the after-inner-loop marker is not. Instrumentation does not suppress the failure.

## Source and actual AIR agree on the expected edge

The source edge is `WL_RETN` -> break inner for -> `WL_SPUN` -> break switch -> next outer dispatch iteration. Both breaks are lexically well-defined. The frame slot is initialized and in bounds; lookup output is initialized; this input has no dynamic result/task allocation on the failing path. See `task-ownership-analysis.md` for the bounded ownership/UB analysis.

AIR was extracted without rewriting from the newly built `boundary-be236f32.bin.gpu`. `boundary-preserved.ll` was decoded using the already-installed LLVM 21 llvm-dis, preserving its AIR target and data layout. The earlier `boundary-air.ll` was printed by Apple clang, which overrides the printed target/data layout to ARM64; it was used only for reading, never GPU execution.

Relevant control-flow labels in the preserved AIR:

1. Literal VI branch computes r0 and loads fid=238 from the valid return slot.
2. `cleanup10681` receives cleanup destination 55 (line 23917).
3. Switch case 55 -> `cleanup10685.loopexit36280` (23932).
4. `cleanup10685` receives cleanup destination 4 (23936).
5. `cond14985` is 4==4 -> `sw.epilog14914` (23945).
6. `cleanup14915` receives cleanup destination 0 from that switch epilogue (32730).
7. `cond14935` is 0==0 -> `for.cond`, not function return (32747).

No return/unreachable/undefined operand exists on that selected edge. The function does carry an initially undefined return accumulator (`%retval.0` at 1675), but the defined path never returns it. A wrong cleanup/exit decision could expose that accumulator; this is a mechanism consistent with the observed garbage, not an observed GPU register value.

## Direct original-AIR replay

`boundary-original.metallib` is the unmodified air64 slice of the fresh GPU archive (SHA-256 recorded in `root-cause-artifacts.sha256`). `air-loader-be236f32.c` changes only the host library creation to `newLibraryWithData` for that exact metallib, not source compilation. A fresh, uniquely named loader binary/archive was built from it.

`original-air-runtime.log`: all three plain inputs still fail. Input 1 again has success=1/tag=0/value=1, depth=3/return fid=238, followed by garbage task re-entry. Thus the failure reproduces downstream of the retained AIR, without either Bend compilation or MSL source compilation in the replay. Fresh archive creation and FailOnBinaryArchiveMiss exclude a stale shader archive.

The direct-AIR loader and the original source binary both crash on the host when validation is enabled while loading their existing plain archives. The original process crash report places the null dereference in `MTLMetalScriptBuilder::initFromScript`, called by `_MTLBinaryArchive` loading through MetalTools debug wrappers, before shader execution. This is NOT evidence about the original compute defect and is not a passing validation control. A subsequent validation run with the plain archive temporarily removed reached runtime shader compilation but did not finish within the unchanged 120-second guard limit (`original-source-validation-no-archive.log`); no pass/fail result for that shader was obtained. The watchdog also reported EPERM while killing its group; a subsequent process inspection found no remaining reproducer or guard process. The original archive is restored separately.

## Negative experiments and limits

- Changing only WL_SPUN from `} break;` to `} continue;` is not a fix: CPU 3/3, Metal 0/45 (`continue-be236f32.log`).
- Offline `xcrun metal` initially could not run because Xcode's Metal Toolchain component was missing (`offline-metal.log`). After manual installation it reports Apple metal 32023.922 (metalfe-32023.922.4).
- Reassembling unchanged AIR with generic LLVM 21 produces a library that loads, but its pipeline compiler connection terminates. This is an unsupported writer-compatibility experiment, not evidence of the original defect. The sentinel-modified AIR was not executed.
- The installed `air-objdump` cannot initialize an AGX disassembler (`boundary-gpu-disasm.txt`). A specific register-allocation, cleanup-selector, or branch-table instruction has not been identified.
- The first guarded Metal Toolchain download reached 3.4 MB of 838.9 MB before the 120-second watchdog stopped it (`metal-toolchain-install.log`). User then installed the component manually. The initial installation attempt changed no guard limits.

## Conclusion

Established: the earliest divergence is an incorrect function exit at a defined nested-loop return edge, downstream of source-correct AIR. An allocator/task-record race is not the cause of the input-1 failure. The root-cause boundary is the Metal AIR-to-execution layer on this machine; incorrect backend control-flow lowering is the leading explanation, not yet an identified Apple optimizer pass/instruction. The passing newer Bend commit and source perturbations are mitigations/codegen changes, not proof that this underlying defect is fixed.

## Manual installation and retained state

User-selected next action, not executed by this investigation:

```sh
xcodebuild -downloadComponent MetalToolchain
```

After installation, `xcrun metal --version` identifies the available compiler. Forensic inputs are `boundary-original.metallib`, `boundary-air-410.bc`, `boundary-preserved.ll`, `boundary-be236f32.c`, and `root-cause-artifacts.sha256`. Exact source-correct cleanup labels and the failing direct-AIR replay are retained above. The next unresolved question is which AIR-to-AGX lowering/execution step loses the valid loop-break edge; no permanent compiler/runtime fix has been claimed.

## Resumed offline evidence: concrete native-compiler diagnostic

The original pipeline script was extracted from the metallib's `SLST` flatbuffer section. Only its placeholder library path was changed to the exact retained original metallib; pipeline settings and original AIR are unchanged (`captured-pipeline.bin`). The captured pipeline targets `applegpu_g13s` and macOS 26.6. The installed native-compiler configuration selects `libapplegpu25-nt.dylib` for macOS 26.x.

`offline-captured-selected-guard.log` records native compiler **exit 0** and:

```text
Max Unstruct ID 555 in function agc.main in shader 0 exceeds supported limit
```

The untouched original build reports the same 555 in a separate run (`offline-unpatched2.log`, 23 s); the earlier attempt exceeded the 120 s limit only because of machine load, not build cost. It nevertheless emits `offline-captured.gpu`. Loading that exact archive into a fresh copy of the original host/AIR loader reproduces the input-1 failure in all three runs, with the same valid stage105 result/stack/return-fid followed by garbage task re-entry (`offline-overlimit-runtime.log`). The runtime accepts this archive without a source-compilation fallback. Thus the warning describes an observably broken emitted binary, not merely an unused diagnostic.

Important negative control: the passing commit's exact retained AIR also compiles with an over-limit warning, now `Max Unstruct ID 548` (`offline-passing-completed.log`). That emitted archive is accepted without fallback and returns the correct 2/1/3 values on actual Metal (`offline-passing-runtime.log`). The supplied workaround regenerated from the failing source also stays over the limit at 489 (`offline-redispatch.log`) while passing all three inputs on real Metal 3/3 plus CPU 3/3 (`redispatch-runtime-control.log`). Therefore **exceeding the maximum region ID is not by itself sufficient to fail these inputs**; the earlier hypothesis that the reported maximum explains the defect is refuted as stated, and `555 > 255` cannot be presented as the proven cause.

What the limit does establish, with the code-level proof in `unstruct-limit-binary-analysis.md`, is that this GPU class exposes only a 255-region space to the legacy AGX control-flow lowering used for `applegpu_g13s`, that Bend's generated dispatcher needs several hundred such regions, and that the compiler reacts by warning and continuing instead of rejecting or restructuring. Which specific region identifier is mis-handled on the executed edge remains unproven, so the mechanism is still an unidentified defect inside that pass/encoder rather than an identified wrap.

### Diagnostic guard

The original guard sometimes aborts because its `ps -A` subprocess exceeds its own 1-second timeout, not because a measured RSS limit was exceeded. A local `guard-diagnostic.ts` retains the original command deadline (120 seconds), 2 GiB RSS cap, 20 GiB free-space requirement, 10% memory-headroom requirement, single-run lock, process-group accounting and fail-closed behavior. On macOS it selects the target process group directly with `ps -g`, plus the monitor PID to avoid a no-process exit race; it still filters/sums exactly the target group's RSS.

`guard-group-selection-proof.log` verifies identical group membership against `ps -A`, including a grandchild and excluding the monitor. An actual failure diagnostic confirms `spawnSync ps ETIMEDOUT`, status null, signal SIGTERM, no output. The direct group query also occasionally exceeds one second during native compilation. User explicitly approved increasing **only this health-query timeout to 5 seconds** in the local diagnostic guard. RSS enforcement can consequently be delayed up to five seconds; it remains a polling safeguard, not a hard OS memory limit. Main checkout `bond/guard.ts` is unchanged.

User then approved one single 10-minute compiler run to complete the region-ID measurement. Because the shipped guard's parser permits options to tighten limits only, that run used a local `guard-long.ts` (copy of the diagnostic guard with `--timeout` default 600 and nothing else changed; paper limits still 2 GiB, 20 GiB, 10%). Approval and the single-use scope are recorded here.

### Region-ID measurement: closed as unavailable

`air-nt` accepts `-mllvm -print-gpucfglower-flowanalyzer` without the wrapper's "Unsupported translator option" rejection, but produces no DAG output; the in-process flag write also could not be used safely (`cg_print_probe` faults during dyld image-add callbacks, and no crash frame shows it reaching the write). The compiler still reports only `Max Unstruct ID 555`, which is the function maximum, not the identifier of the region the failing break belongs to.

Two other discriminators were attempted and are excluded: `-mtranslator` is rejected by the wrapper whitelist, and reassembling AIR with the generic LLVM 21 writer produces an archive the pipeline compiler cannot ingest. Obtaining the per-node ID therefore needs either an AGX disassembler or symbolic debugging of the stripped 105 MB plugin, i.e. Apple-internal tooling.

## Final statement

1. Bend's generated C, its AIR, and the host-side scheduling are correct for the failing input. The first wrong action is on the GPU: a defined loop-break edge returns from `work_loop` instead of dispatching `FID_EXIT`, and the scheduler then treats the garbage return as a task.
2. The wrong action originates downstream of AIR, in Apple's Metal/AGX lowering for `applegpu_g13s`, not in Bend lowering, allocation, task ownership, or memory ordering.
3. The legacy AGX control-flow lowerer on this GPU class has a 255-region identifier space (`0xff` vs `0xffff` selected by a virtual target predicate; proven at `0x809f80–0x809fa8`). Bend's dispatcher needs 555 regions, and the compiler only warns.
4. That over-limit condition is not sufficient to fail on its own: the passing commit (548) and the supplied workaround (489) both exceed it and pass. Which region identifier is mis-handled remains unproven.
5. Practical consequence for this fork: on M1/M2 GPUs the emitted dispatcher exceeds the backend's region limit, the compiler accepts it, and correctness then depends on codegen luck. The later upstream commit and the workaround both changed MSL shape rather than respecting the limit, which matches the issue's reports that semantically null edits flip the outcome. A durable direction is to keep the `!`-dispatched Work loop out of the multi-hundred-region CFG (shorter dispatch, or per-segment redispatch), and to report the non-fatal over-limit acceptance to Apple.
6. Do not treat `d18110b6` passing as proof of repair: its shader still exceeds the G13 region limit, so the same class of failure can reappear with another codegen change.
