# "Max Unstruct ID" — native compiler binary analysis

Artifact: `/private/var/run/com.apple.security.cryptexd/mnt/com.apple.MobileAsset.MetalToolchain-v27.2.47.0.xECBT9/Metal.xctoolchain/usr/metal/32023/lib/libapplegpu25-nt.dylib`
(105.8 MB; selected for macOS 26.x by `lib/air-nt/config.yaml` lines 11–14, not by `libapplegpu-nt.dylib`).

The dylib is a fat Mach-O:

| slice | cputype | file offset | size |
|---|---|---|---|
| x86_64h | 0x01000007 / subtype 3 | 0x4000 | 0x366df60 |
| arm64 | 0x0100000c / subtype 0 | 0x3674000 | 0x3362e40 |

The M1 Max process loads the **arm64** slice. All VAs below are arm64-slice VAs; file offset = `0x3674000 + VA`. Section layout (arm64): `__text` VMA 0xa80 size 0x159a1a8; `__const` VMA 0x159bf20; `__cstring` VMA 0x2eaffa0 size 0xf3854. Function boundaries were taken from `LC_FUNCTION_STARTS` (dataoff 0x32dabc0, size 0x197f0) and validated against exported symbols (`_AIRNTGetDefaultArch` = 0x1714a0, etc.).

## 1. Exact numeric comparison — proven

The check is in the **GPU Control Flow Lowering** pass (`gpu-cfg-lower`, description "GPU Control Flow Lowering"; strings at VMA 0x2ee34a0 / 0x2ee34ae). The containing arm64 function starts at **0x809aec**, ends at **0x80cbe0** (size 0x30f4); it includes the inlined FlowAnalyzer / CFGUnloopifier logic (nearby strings "CFGUnloopifier" 0x2ee34c8, "ControlGraph" 0x2ee34e7, "FlowAnalyzer" 0x2ee3502).

```
0x809f80  60 7e 40 f9   ldr  x0, [x19, #0xf8]      ; target-info object of the pass
0x809f84  08 00 40 f9   ldr  x8, [x0]
0x809f88  08 71 40 f9   ldr  x8, [x8, #0xe0]       ; virtual method, vtable slot 0xe0
0x809f8c  00 01 3f d6   blr  x8                    ; returns bool in w0: chooses ID width
0x809f90  e8 1f 80 52   mov  w8, #0xff              ; 255
0x809f94  e9 ff 9f 52   mov  w9, #0xffff            ; 65535
0x809f98  1f 00 00 71   cmp  w0, #0
0x809f9c  28 11 88 1a   csel w8, w9, w8, ne         ; w8 = (feature()!=0) ? 65535 : 255
0x809fa0  e9 47 4a b9   ldr  w9, [sp, #0xa44]       ; Max Unstruct ID
0x809fa4  3f 01 08 6b   cmp  w9, w8
0x809fa8  29 0d 00 54   b.ls 0x80a14c              ; id <= limit: skip diagnostic
```

So the supported limit is **not a single constant**. It is selected at runtime by a virtual predicate on `[pass+0xf8]` (slot 0xe0): **255 if the predicate is false, 65535 if true**. The two values are literal instructions in the binary; 255/511/other guesses are not involved.

On this machine (applegpu_g13s) the observed diagnostic at ID 555 and at ID 548 proves the predicate returned **0**, i.e. the effective limit is **255** for G13s. The 65535 branch is the wide/next-generation path.

The report site is the same function; the check is **not** gated by a debug option:

* 0x809ecc–0x809ed8 reads a byte global at `0x32e9d20` and, if != 1, jumps over only the preceding dot/CFG dump block (0x809edc–0x809f7c) straight to **0x809f80**. The limit check at 0x809f80 always executes.

## 2. It only warns — proven

When `MaxID > limit`, execution falls into the diagnostic construction at 0x809fac and prints, in order:

| piece | length | string VMA |
|---|---|---|
| `"Max Unstruct ID "` | 0x10 | 0x2ee351d |
| decimal ID (`[sp+0xa44]`, loaded at 0x809fe0) | — | — |
| `" in function "` | 0x0d | 0x2ee352e |
| function name (from `[x20]` → 0xe657e0 / 0x106b9e4) | — | — |
| `" in shader "` | 0x0b | 0x2ee353c |
| shader index | — | — |
| `" exceeds supported limit\n"` | 0x19 | 0x2ee3548 |

(exact immediates: `add x1,x1,#0x51d; mov w2,#0x10` at 0x809fd0/0x809fd8; `add #0x52e; mov w2,#0xd` at 0x80a000/0x80a008; `add #0x53c; mov w2,#0xb` at 0x80a080/0x80a088; `add #0x548; mov w2,#0x19` at 0x80a13c/0x80a144).

There is no `report_fatal_error`, no abort/exit path, and no ID modification in this block. Both the `b.ls` (no warning) path and the warning path converge on the **same continuation at 0x80a14c** (0x80a148 `bl` falls through; 0x809fa8 `b.ls` targets it). This matches the observed exit code 0.

## 3. No clamp in the lowering pass — proven; 16-bit operand, not 8

The IDs are produced by a monotonic 32-bit counter in function **0x80f85c** (the CFG/control-graph builder called from 0x80f7ac via 0x80f824):

```
0x80fe4c  e9 4f 40 b9   ldr  w9, [sp, #0x4c]    ; counter
0x80fe50  29 05 00 11   add  w9, w9, #1
0x80fe54  e9 4f 00 b9   str  w9, [sp, #0x4c]
0x80fe58  69 22 00 b9   str  w9, [x19, #0x20]    ; node->UnstructuredID (32-bit)
```

* The node field is confirmed 32-bit by the dumper: `ldr w1, [x20, #0x20]` at 0x80da50 after printing `"UnstructuredID="` (0x2ee35f7), and the assignment/copy at 0x80fe04/0x80fe24 uses `[x19,#0x24]` for a second 32-bit counter.
* No `cmp #0xff`, no `csel/csmin` saturation, no mask is applied to the counter in 0x80f85c or in the warning block. (Only unrelated `and #0xfffe` / `orr #0x100` flag operations exist there.)

At instruction construction the ID is narrowed to **16 bits only**:

```
0x80a854  2a 01 08 2a   orr  w10, w9, w8        ; two unstructured IDs from node+0x20
0x80a858  5f 3d 00 72   tst  w10, #0xffff       ; presence test
0x80a880  44 3d 00 12   and  w4, w10, #0xffff   ; first ID operand
0x80a884  05 3d 00 12   and  w5, w8,  #0xffff   ; second ID operand
0x80a898  20 01 3f d6   blr  x9                ; vtable slot 0xb8: emit unstruct prolog/epilog
...
0x80ac70  f3 07 00 79   strh w19, [sp, #2]      ; ID passed as 16-bit stack argument
0x80ac98  20 01 3f d6   blr  x9                ; vtable slot 0x50: emit instruction
```

Thus within this dylib's control-flow lowering, IDs >255 are **neither clamped to 255 nor truncated to 8 bits**. They remain full 32-bit values in the graph and are handed to the target instruction emitter as 16-bit operands (555 fits in 16 bits). If the hardware/encoder field is 8-bit on G13, the loss of information happens below this pass, in the final AGX encoding. No range check or error was found there either: the only literal 255/65535 site in the binary is the check above, and a scan of `__cstring` for range/limit diagnostics found no unstructured-ID encoder message other than " exceeds supported limit\n" (0x2ee3548) itself.

## 4. Confidence

* **Certain:** comparison site, the literal candidates 255/65535, the feature-selected limit, effective G13s limit = 255, and non-fatal warn-and-continue behavior (same successor, exit 0, no abort call).
* **High:** no clamp/truncation in the gpu-cfg-lower pass; IDs are 32-bit in the graph and 16-bit at the emit call.
* **Strong inference (medium-high):** 255 vs 65535 are the max values of an 8-bit vs 16-bit encodable field; the target predicate selects the hardware's supported width. G13s selects 8-bit.
* **Unknown/not proven:** whether the final AGX encoder masks the ID to 8 bits, overlaps adjacent instruction bits, or leaves it to hardware. A byte scan of the produced GPU images found no little-endian 16-bit 0x022B (555) or 0x0224 (548) anywhere, so IDs are bit-packed, and static image scanning alone cannot settle this.

## 5. Emitted-image observation (useful for the runtime comparison)

`offline-captured.gpu` and `offline-passing.gpu` are outer GPU Mach-O executables (`cputype 0x01000013` = applegpu_g13s) whose `__compute` section is itself a nested Mach-O at outer file offset **0x810**. Nested `__text` is 0x141770 bytes at nested offset 0x2b0; symbols include `ltmp0` 0x4500 and `_agc.main` **0x4540** (outer file offset 0x5000). Metadata sections `__GPU_METADATA` (0x808), `__GPU_LD_MD` (0xe0), `__GPU_ARCH_LD_MD` (0x20) contain names/descriptors but no plain 555/548 values. Both images are otherwise structurally identical, so a direct image diff must be interpreted with the AGX ISA, not by literal ID search.

A bounded scan of all 64-bit-aligned words at every bit offset 0–48 found only 41 windows equal to 555 in the 555-warning image (chance level ≈ 123 for 165k words × 49 offsets) and 213 in the 548-warning image. There is therefore no evidence of a plain 16-bit ID field in the emitted code; notably the image whose diagnostic says 555 does not contain an excess of 555-bit patterns. This is consistent with the ID being packed into a narrower bitfield (or not emitted as a standalone 16-bit field), but it cannot distinguish 8-bit truncation from bitfield overlap without the AGX encoder layout.

## 6. Minimal discriminating experiments

1. **Selected-edge IDs (parent already planned):** the pass registers three dedicated CLI options (exact strings in `__cstring`, use with whatever forwards options to the plugin, e.g. `-mtranslator`):
   * `-print-gpucfglower-cfgunloopifier` (0x2ee33b1, help "Print a DAG showing the CFG unloopified representation" 0x2ee33d2)
   * `-print-gpucfglower-controlgraph` (0x2ee3409)
   * `-print-gpucfglower-flowanalyzer` (0x2ee344e, help "Print a DAG showing the flow analysis" 0x2ee346d)

   All three print node attributes including `UnstructuredID=` (0x2ee35f7), i.e. the executed edge's IDs and the ID value associated with the inner-loop-break/switch-break prolog and epilog. `-print-after=gpu-cfg-lower` (pass name string 0x2ee34a0) is the generic alternative but may be less reliable under the legacy pass manager. Then `-print-after` the final pre-RA/RA pass to see any later narrowing.
2. **Boundary shader:** synthetic shader whose executed unstructured region needs ID 255 vs 256. If 255 works and 256 fails on G13s, the 8-bit field is confirmed functionally.
3. **Wide-target codegen diff:** compile the identical AIR for a target whose predicate selects 65535 (G15 class) and compare the AGX words for the same function. The bit positions that change with the ID value give the field width directly (8 vs 16 bits) without needing an AGX disassembler.
4. **Encoder breakpoint (if native debugging is acceptable):** break at 0x809f80 to read w0 on g13s (expected 0), and at 0x80a898/0x80ac98 to capture the emitted operands; then break in the final MC encoder and inspect the encoded instruction word for ID 555. This is the only route that distinguishes "encoder masks to low 8 bits" from "hardware mis-indexes a 16-bit ID".
