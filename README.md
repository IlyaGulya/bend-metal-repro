# Bend Metal evaluator reproducer

Reproduces [bendlang/bend#2094](https://github.com/bendlang/bend/issues/2094).

The same program returns `2`, `1`, `3` on CPU and fails on Metal on an M1 Max. The included compiler patch makes those GPU runs pass. This is a tested workaround, not a confirmed root-cause fix.

The input has three IR nodes. The full evaluator module is included because a smaller standalone evaluator did not reproduce the failure. `checker.bend` provides shared types; this example does not run the model checker.

## Build

Use Bun 1.4.4 on Linux with a working systemd user session. Clone this repository, then get the tested compiler in a separate directory:

```sh
git clone https://github.com/bendlang/bend.git ../bend
export COMPILER="$(cd ../bend && pwd)"
git -C "$COMPILER" checkout be236f324ed0338fe907fe50474468e4d3b94157

./scripts/bounded env BEND_LIB="$PWD/.cache/bend-lib" \
  bun --smol "$COMPILER/bend2/main.ts" repro.bend -o before.c

patch -d "$COMPILER/bend2" -p1 -F0 -t \
  < patches/bend-device-work-dispatch.patch

./scripts/bounded env BEND_LIB="$PWD/.cache/bend-lib" \
  bun --smol "$COMPILER/bend2/main.ts" repro.bend -o after.c
```

Copy the repository and both C files to the Mac. Requires Xcode command-line tools and Python 3. The macOS helper uses sudo to apply a memory limit; the compiler and program run as the original user.

```sh
sudo -v
./scripts/build-macos --metal before.c "$PWD/before"
./scripts/build-macos --metal after.c "$PWD/after"

./scripts/bounded-macos -- "$PWD/before" +RTS --threads 4 --gpu off -RTS 1
./scripts/bounded-macos -- "$PWD/before" +RTS --threads 4 --gpu 512MB -RTS 1
./scripts/bounded-macos -- "$PWD/after" +RTS --threads 4 --gpu 512MB -RTS 1
```

Repeat the last three commands with inputs `0` and `2`.

## Expected and observed

| Input | Expected value | Original CPU | Original Metal | Patched Metal |
|---|---|---|---|---|
| `0` | `2` | `2` | fails | `2` |
| `1` | `1` | `1` | fails | `1` |
| `2` | `3` | `3` | fails | `3` |

The recorded Metal error was `frontier drained without a result`. Earlier runs also reported `a function the device does not hold` or `memory fault (machine stack overflow?)`.

Tested on M1 Max, 32 GPU cores, Darwin 25.6.0, Apple clang 21. Patched runs reported actual Metal command submissions, not CPU fallback. Current Bend main and CUDA were not executed.

The patch replaces device Work self-loops with VM redispatch. Original inlining and native flat loops stay unchanged. Whether this repairs invalid lowering or avoids a Metal compiler defect is unknown. See the upstream issue for related reports.
