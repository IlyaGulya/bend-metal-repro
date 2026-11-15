# Evidence for bendlang/bend#2094

Files behind the issue comment
[`#issuecomment-6897282341`](https://github.com/bendlang/bend/issues/2094#issuecomment-6897282341).

Machine: M1 Max, 32 GPU cores, macOS 26.6.2 (25G83), Metal 4, Apple clang 21.0.0,
Bun 1.4.4. Compiler commits: `be236f32` (fails) and `d18110b6` (passes).
No compiler patches. Shader validation unset.

## reports

| file | content |
|---|---|
| `root-cause-analysis.md` | main write-up: method, evidence, limits, final statement |
| `task-ownership-analysis.md` | proof that task records and the allocator are not the cause |
| `metal-ordering-analysis.md` | Metal memory-model review of the runtime, and the ring handoff defect |
| `unstruct-limit-binary-analysis.md` | disassembly of the 255/65535 limit check in the AGX plugin |
| `unstruct-id-semantics.md` | what an unstructured region ID is, and how far the encoding is known |

## generated

`generated/original-be236f32.c` is the untouched C emitted by the failing compiler.
`generated/boundary-be236f32.c` is a copy with a per-lane trace buffer; it fails the
same way, so the trace does not hide the fault. Both files compile on macOS as shown
in the comment.

## air

`air/boundary-air-410.bc` is the AIR slice extracted from the metallib built from
`boundary-be236f32.c`, without rewriting. `air/boundary-preserved.ll` is that bitcode
read with LLVM 21 `llvm-dis`; the AIR target stays intact. The edge quoted in the
comment is at the `cleanup10681`/`cleanup10685`/`sw.epilog14914`/`cleanup14915` chain.

## metallib

`metallib/boundary-original.metallib` is the unmodified air64 slice of the GPU
archive. It was loaded directly with `newLibraryWithData` from a fresh host binary
with a unique archive name, and all three inputs failed the same way. This rules out
both the MSL front end and a stale `.gpu` archive.

## logs

| file | content |
|---|---|
| `results-be236f32.jsonl` | 45 Metal runs and 3 CPU runs on the failing commit |
| `results-d18110b6.jsonl` | the same matrix on the passing commit |
| `comparison.json` | summary of both, including error-message counts |
| `boundary-be236f32.log` | per-lane dispatch trace, stage markers before `WL_RETN` |
| `loop-exit-be236f32.log` | marker after the inner loop, which never runs |
| `original-air-runtime.log` | the direct AIR replay |
| `offline-overlimit-runtime.log` | the emitted 555 archive, loaded and failing |
| `offline-passing-completed.log` | native compiler output for `d18110b6`: ID 548 |
| `offline-passing-runtime.log` | that archive returning 2, 1, 3 on Metal |
| `offline-unpatched2.log` | native compiler output for untouched `be236f32`: ID 555 |
| `offline-redispatch.log` | the workaround build: ID 489 |
| `redispatch-runtime-control.log` | the workaround passing CPU 3/3 and Metal 3/3 |
| `root-cause-artifacts.sha256` | SHA-256 of the input artifacts |
| `run-info.json` | environment record for the retest matrix |

## tools

`tools/repack-air.py` lifts the pipeline script out of a metallib `SLST` section,
points its placeholder library path at a chosen metallib, and rebuilds the container
with corrected hashes and offsets. It is how the offline `air-nt` runs were made.

## Reproducing the backend diagnostic

```sh
xcrun air-objdump --metallib --script <archive>.metallib   # pipeline script
python3 tools/repack-air.py <archive>.metallib <new>.bc <out>.metallib
air-nt -arch applegpu_g13s -N <script> -platform_version macos 26.6 26.6
```

The 255/65535 comparison is in the legacy plugin selected for macOS 26.x,
`libapplegpu25-nt.dylib`, inside the `gpu-cfg-lower` pass.
