# "Unstruct ID" semantics and limits (Apple GPU / AGX), bounded findings

Scope: what Apple's GPU backend means by an unstructured-control-flow ID, whether IDs live in
fixed-width fields, whether overflow can explain the observed "break dispatch returns garbage"
behaviour, and which ID-reduction strategies are semantically neutral. No target execution,
compilation, builds or tests were performed for this note. All binary reads were bounded
read-only mmap/string/short-range disassembly queries.

Status labels used below: **[observed]** = read directly from a primary artifact/source;
**[inferred]** = follows from observed facts but not directly shown; **[unknown]** = not established.

## 1. Where the number comes from (owner of the diagnostic)

The string is produced by Apple's AGX backend pass, not by Metal source compilation, not by Bend,
and not by any public LLVM pass.

**[observed]** Apple Metal toolchain 32023 (compiler 32023.922.4),
`.../Metal.xctoolchain/usr/metal/32023/lib/libapplegpu25-nt.dylib` (universal; arm64 slice):
the `__cstring` page contains, contiguously:

```
GPU Control Flow Lowering / gpu-cfg-lower / CFG Lowering
CFGUnloopifier / ControlGraph / FlowAnalyzer
Max Unstruct ID  in function  in shader  exceeds supported limit\n
EntryNestingLevel= ExitNestingLevel= UnstructuredID= PrologExtraCndEnds= LoopEndPrologs=
UnstructuredCndEnd SkipUnstructuredSuccessorEmit InvertPredicate LoopBegin IfBegin
IfElseBegin ElseBegin UnstructuredProlog LoopEndEpilog LoopBackedge LoopBreak
OptimizedUnstructuredLoopExit OptimizedLoopEpilog BranchUniformBegin BranchUniformElseBegin
BranchUniformControlled
```

Hidden debug options in the same binary: `print-gpucfglower-controlgraph`,
`print-gpucfglower-flowanalyzer`, `print-gpucfglower-cfgunloopifier` (descriptions: "Print a DAG
showing the control graph / flow analysis / CFG unloopified representation"), plus a neighbouring
pass `gpu-cond-flatten` ("GPU Conditional Flattener") and `gpu-itr-combine`. The same diagnostic
exists in `libapplegpu23-nt`, `libapplegpu24-nt`, `libapplegpuG9G12-nt` and the generic
`libapplegpu-nt`; it is a backend-wide facility, not a macOS-26-only path.

So an "unstruct ID" is a **compiler-assigned identifier for a node of the backend's ControlGraph**
(nodes also carry `EntryNestingLevel`, `ExitNestingLevel`, `UnstructuredProlog`, `UnstructuredCndEnd`,
loop-begin/end flags). It is a per-function bookkeeping index for lowering an arbitrary (unstructured)
CFG, not a source-language stack depth. Metal source cannot even express `goto` (MSL 4.1, §1.5.4:
`goto` statement not available), so the unstructured regions are created by the backend from
`switch`/`break`/`continue`/nested-exit shapes.

## 2. Numeric guard in the pass (corroboration only; UnstructLimit owns the comparator result)

**[observed]** In the arm64 slice of `libapplegpu25-nt.dylib`, the function containing the diagnostic
starts at slice-relative `0x809aec`. The check is:

```
0x809f80  ldr  x0, [x19, #0xf8]      ; target/device-info object
0x809f84  ldr  x8, [x0]
0x809f88  ldr  x8, [x8, #0xe0]       ; per-target virtual predicate
0x809f8c  blr  x8
0x809f90  mov  w8, #0xff
0x809f94  mov  w9, #0xffff
0x809f98  cmp  w0, #0
0x809f9c  csel w8, w9, w8, ne        ; limit = predicate ? 65535 : 255
0x809fa0  ldr  w9, [sp, #0xa44]      ; maximum UnstructuredID in this function
0x809fa4  cmp  w9, w8
0x809fa8  b.ls 0x80a14c              ; only warn when maxID > limit
```

The diagnostic is assembled at `0x809fd0`/`0x80a0a8` from the literals `Max Unstruct ID `,
` in function `, ` in shader `, ` exceeds supported limit\n`; the numeric value printed is the
`[sp,#0xa44]` word. Both observed warnings (555 on the boundary shader, 548 on the passing-commit
shader, from the parent's guarded offline compiles) therefore imply that **on `applegpu_g13s` the
predicate selected 255**, since both values are below 65535.

**[observed]** There is no abort on overflow: the warning path rejoins the no-warning path at
`0x80a14c`, which continues directly into graph/emission work. Parent observed exit 0 and a produced
`.gpu`. The compiler warns and keeps lowering with the same IDs.

**[observed, corroborated by UnstructLimit/parent]** IDs are monotonically assigned 32-bit values in
the lowering; the 16-bit `and`/`strh` above is operand marshalling, and **no 8-bit clamp or mask is
present in the lowering code**. Therefore any 8-bit wrap must occur in the final encoder/emitted
form, and that truncation is currently **unproven** (no AGX instruction printer is available in this
toolchain). H1 below is a candidate, not an established mechanism.

**Interpretation [inferred]**: 255 and 65535 are exactly the maxima of unsigned 8-bit and 16-bit
values. The code is selecting between an **8-bit and a 16-bit ID representation** for the target.
It cannot be distinguished from this code alone whether the 8/16-bit bound is a bitfield width in the
emitted instruction/metadata or the capacity of a per-shader table indexed by the ID; both are
"supported limit < 556" facts with the same consequence for out-of-range IDs.

## 3. Width of the ID along the pipeline

**[observed]** ControlGraph node layout (used by the DAG printer): `EntryNestingLevel` at node+0x18,
`ExitNestingLevel` at node+0x1c, `UnstructuredID` at node+0x20, stride 0x30. The printer loads all
three with `ldr w` (32-bit loads):

```
0x80d9b8  ldr w1, [x20, #0x18]   ; EntryNestingLevel=
0x80da00  ldr w1, [x20, #0x1c]   ; ExitNestingLevel=
0x80da50  ldr w1, [x20, #0x20]   ; UnstructuredID=
```

**[observed]** In the emitter part of the same function the ID is read with `ldr w` and handed to the
emission interface as a **16-bit** value:

```
0x80ac48  umaddl x8, w2, w0(=0x30), x19
0x80ac4c  ldr  w19, [x8, #0x20]       ; node->UnstructuredID
0x80ac6c  and  w7, w20, #0xffff
0x80ac70  strh w19, [sp, #0x2]        ; ID passed as uint16_t stack argument
...
0x80b14c  ldr  w9, [x9, #0x20]        ; second node's ID
0x80b16c  strh w19, [sp, #0x4]
0x80b174  strh w9,  [sp]              ; both IDs passed as uint16_t
```

Also `and w4, w10, #0xffff` / `and w5, w8, #0xffff` at `0x80a880`/`0x80a884` before another
emission call. So: **32-bit in the graph, 16-bit at the emitter ABI, and a target-dependent 8-bit
cap in the guard.** A 32-bit compiler-side truncation is ruled out; the narrowing, if any, happens in
the 8-bit target's emitted form.

## 4. Branch reach is not the limiting factor (rules out "offset overflow")

**[observed, Asahi Mesa]** `src/asahi/compiler/agx_pack.c` fixes up relative branches by patching a
full 32-bit PC-relative value two bytes into the instruction:

```c
/* Branch offset is 2 bytes into the jump instruction */
uint8_t *location = ((uint8_t *)emission->data) + fix.offset + 2;
int32_t patch = (int32_t)target - (int32_t)fix.offset;
memcpy(location, &patch, sizeof(patch));
```

`src/asahi/compiler/agx_opcodes.py` gives the same picture: `jmp_exec_any/none/none_after`,
`call`, and the structured `if_/else_/while_icmp` all take a `TARGET` immediate; AGX instructions are
variable-length 2–12 bytes. **[observed, dougallj G13 ISA notes]** the flow-control bit diagrams show
a 32-bit sign-extended `off` for `jmp_exec_any`, `jmp_exec_none`, `call`; only the small `n`
(exec-mask level increment) and `cc` fields are narrow.

Consequence: a 555-valued unstruct ID cannot be a branch *distance* overflow. The ID is a side-band
identity/index consumed by the backend's unstructured-CFG lowering; the failure mode to look for is
ID aliasing or a missing table entry, not a too-far branch.

## 5. Hardware context: structured CF versus Apple's unstructured lowering

**[observed, Asahi Mesa `src/asahi/compiler/README.md` + `agx_opcodes.py`]** AGX has `r0l` as the
hardware nesting counter; `if_icmp`/`else_icmp`/`while_icmp` update a per-thread count `v` in `r0l`
(`v += n`, `v = 1`, `v -= n` in `pop_exec`), and `exec_mask = (v == 0)`. Mesa lowers `break` as a
pseudo-instruction carrying `NEST` + `TARGET` that becomes an `r0l` write and `pop_exec 0`.

**[observed, dougallj]** same model: `r0l` tracks the execution-mask stack; a break is implemented by
moving a level count into `r0l` and executing `pop_exec 0`; `if/else/while` are the only known
exec-mask manipulators.

**[observed, Apple LLVM GPU Compiler talk, LLVM devmtg 2017, slides 46–48]** Apple's backend
"supports full execution of unstructured control-flow handled at MI-level with little overhead", so
they require only limited structurization (loops in LoopSimplify form), and use duplication-based
structurization for small unstructured blocks. This is the public statement that the AGX backend has
a machine-level unstructured-CFG mechanism distinct from the structured `if/while` encodings above.

**[inferred]** The `gpu-cfg-lower` pass is that mechanism: it unloopifies/flattens and assigns IDs to
unstructured regions so the target can represent cross-region exits (e.g. `break` out of a `switch`
inside nested loops). The 8-bit cap is a property of that target mechanism, and later GPU
generations evidently have a 16-bit variant (the same code selects 65535 for them).

**Negative documentation result [observed]:** neither Asahi Mesa's AGX compiler (`agx_pack.c`,
`agx_opcodes.py`, `README.md`) nor Dougall Johnson's G13 ISA notes name an "unstruct" field, an
8/16-bit ID operand, or a per-region ID table. The publicly documented G13 CF surface is the `r0l`
nesting counter, `if/else/while_*cmp`, `pop_exec`, and 32-bit relative branch/call offsets. Mesa's
compiler emits structured CF and does not model any ID; its packer explicitly says "We don't
implement indirect branches". So a docs-primary mapping of Apple's ID to a specific G13 hardware field
**does not exist in public sources**; the only evidence for the 8/16-bit split is Apple's own backend
binary, and the wrap point is unproven.

## 6. Does this explain the observed break mis-dispatch?

**Observed failure shape** (parent's artifacts): input 1 on the boundary shader completes
`spin_ir_nthNode` correctly (tag 0 / value 1), stack depth 3, return slot `0xEE` = FID_EXIT = 238, then
`work_loop` returns an arbitrary 64-bit word that is re-entered as a task. Expected edge is inner-loop
break -> switch break -> next dispatcher iteration. `boundary-preserved.ll` shows no
return/unreachable on that edge, but the function has an undefined `%retval.0` accumulator that the
good path never uses.

**Hypothesis H1 (best fit): ID aliasing / missing region metadata on the executed edge.**
If the 8-bit target encodes the unstructured ID in 8 bits (or indexes an 8-bit-bounded table), an ID
such as 555 narrows to 43. The branch would then carry another region's identity/target metadata.
Landing on a region whose "condition end" is the function epilogue, with the undefined `%retval.0`
live, reproduces "returns arbitrary word instead of continuing the loop" without any memory
corruption. **This is a mechanism consistent with the evidence, not an observed truncation.**
Critical caveat from the parent's control: the passing-commit shader also exceeds the cap
(max ID 548) yet executes all three simple inputs correctly. Therefore `maxID > limit` is **not**
sufficient; H1 requires that the *specific executed* region be the one that aliases badly, while the
new shader's executed regions are either below the cap or alias benignly. The global maximum alone
does not identify the failing edge.

Alternatives that remain open:
- H2: 8-bit ID is a table index, and over-cap entries are dropped rather than wrapped; a branch to a
  dropped entry behaves as fallthrough/wrong exit. Same observable class as H1.
- H3: the warning is incidental for these inputs, and the real defect is a different incorrect
  lowering of the same cleanup/unloopified edge (e.g. wrong `UnstructuredCndEnd`/nesting level).
  The old/new asymmetry (555 fails, 548 passes on simple inputs) is compatible with H3 too.

Discriminator (cheap, no new infrastructure): run the backend with
`-print-gpucfglower-flowanalyzer` (option string present in the plugin) on both captured pipelines;
the DAG prints `UnstructuredID=`, `EntryNestingLevel=`, `ExitNestingLevel=` per node. Map the failing
AIR cleanup/unloopified blocks to nodes and check (a) whether the failing break's node/condition-end
has ID >= 256, and (b) whether the passing shader's executed nodes stay below 256. This converts H1
from plausible to measured or refutes it.

Secondary discriminator: compile the same `captured-pipeline.bin` for a target whose predicate
selects 65535 (e.g. a G15/G16/G17 arch name). If the warning disappears, the cap is
generation-dependent. This must be compile-only (no run) and guarded by the parent.

## 7. Deterministic ways to stay under the cap without changing semantics

There is **[observed]** no `-max-unstruct*`/`-unstruct-limit` option in the plugin (the only
"unstruct" strings are the diagnostic and DAG-printer names), so the limit cannot be raised by a
flag. The only levers are the shape of the per-function CFG, or a toolchain/GPU whose predicate picks
16-bit.

Semantics-neutral candidates, in order of expected effect:
1. **Outlining / no-inline boundaries.** The diagnostic is per function ("in function agc.main in
   shader 0"), so IDs restart at 0 per function. If the AIR reaches AGX codegen with separate
   functions, moving the interpreter's cold cleanup/error arms into non-inlined device functions cuts
   the maximum ID of the hot function. Whether Metal honours `noinline` through the AIR optimizer and
   whether the backend names the merged function `agc.main` must be checked; if everything is inlined
   into `agc.main`, this does nothing.
2. **Eliminate cross-region exits in the generated MSL.** The unstructured regions come from
   `break`/`continue` out of nested constructs (switch-inside-loop). Rewriting the dispatch so each
   loop has a single exit and the inner-loop break becomes a loop-condition/predicate exit removes
   the need for an `UnstructuredCndEnd`/unloopified edge, hence removes IDs. This is a codegen-shape
   change and must be validated as a control, not sold as the root-cause fix.
3. **Reduce switch fan-out / split the interpreter state machine** into smaller per-state functions
   or a table-driven structured dispatch — again a shape change whose only claim is fewer
   unstructured regions per function.
4. **Toolchain/GPU generation:** if the 8/16-bit selection is a GPU-family property (M3+/M4 may pick
   16-bit), the same shader is within spec there. Untested; the parent can settle it with arch-only
   compile probes.

None of these is a "safe because it is semantically equivalent" fix in the sense of proving the
backend correct; they are ways to keep the program inside a representation the compiler actually
supports. The permanent fix belongs in either (a) the generated MSL shape (stay below the cap for
8-bit targets), or (b) the toolchain (reject or correctly lower over-cap shaders instead of warning
and emitting), or (c) a runtime/OS update that selects 16-bit on this GPU. Declaring any one-line
source perturbation "the fix" without the ID dump would repeat the perturbation trap.

## 8. Unresolved facts

- The exact instruction/metadata field or table that consumes the ID, and whether values above the
  cap are masked (`id & 0xff`) or dropped. Not located; the emitter is reached through virtual calls
  with the ID as a 16-bit argument, and AGX has no usable instruction printer in this toolchain
  (`air-objdump` reports `no instruction printer for target agxN---macho`).
- Which GPU generations select 65535. The predicate is a virtual method on the target object
  (`[obj+0xf8] -> vtable+0xe0`); its identity is stripped. M1 (g13s) demonstrably selects 255.
- Whether the cap is an encoding width or a table capacity (bitfield vs entry count).
- Whether ID assignment is dense/ordered by emission; the printer exists but no dump was produced
  here.
- The per-function scope at AGX time: `agc.main` suggests post-inlining merging; if so, MSL-level
  outlining cannot reduce the count without a no-inline barrier.

## 9. Sources

- Apple Metal toolchain 32023 (compiler 32023.922.4), `libapplegpu25-nt.dylib` arm64 slice:
  `__cstring` string cluster at slice offset `0x2ee351d` ff.; diagnostic function `0x809aec`;
  limiter `0x809f80–0x809fa8`; printer `0x80d9b8–0x80da50`; emitter ID marshalling
  `0x80ac48–0x80ac70`, `0x80b14c–0x80b174`. Slice at fat offset `0x3674000`.
- Apple LLVM GPU Compiler: Embedded Dragons (LLVM devmtg 2017), slides 46–48:
  https://llvm.org/devmtg/2017-10/slides/Chandrasekaran-Maggioni-Apple%20LLVM%20GPU%20Compiler.pdf
- Metal Shading Language Specification 4.1, §1.5.4 (goto unavailable):
  https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
- Asahi Mesa `src/asahi/compiler/agx_pack.c` (32-bit branch fixups):
  https://gitlab.freedesktop.org/asahi/mesa/-/raw/main/src/asahi/compiler/agx_pack.c
- Asahi Mesa `src/asahi/compiler/agx_opcodes.py` (CF opcodes, NEST/TARGET):
  https://gitlab.freedesktop.org/asahi/mesa/-/raw/main/src/asahi/compiler/agx_opcodes.py
- Asahi Mesa `src/asahi/compiler/README.md` (`r0l` hardware nesting counter):
  https://gitlab.freedesktop.org/asahi/mesa/-/raw/main/src/asahi/compiler/README.md
- Dougall Johnson, Apple G13 GPU architecture notes (flow control, exec-mask stack):
  https://dougallj.github.io/applegpu/docs.html
- Local AIR shape: `boundary-preserved.ll` `work_loop` = 3993 labels, 2027 conditional branches,
  36 switches (pre-inlining), vs a 255-entry cap.
